//! Point Cloud: positions with colors, normals and batch ids, as one node drawing `POINTS`.
//!
//! - Positions are `POSITION`, or `POSITION_QUANTIZED` times `QUANTIZED_VOLUME_SCALE / 65535`.
//!   They're turned Y up and kept as the file's single precision: the `RTC_CENTER` and the
//!   quantized volume's offset are the node's translation. Plain positions with neither, stored
//!   far from the origin, are measured from a whole number of meters near their middle, which
//!   the node carries instead.
//! - Colors are `RGBA`, `RGB`, `RGB565` or `CONSTANT_RGBA`, the first of them present, read as
//!   sRGB and kept linear as the other point-cloud readers keep theirs. A color every point
//!   shares is the mesh's `extras.constant.COLOR_0` rather than an attribute.
//! - Normals are `NORMAL`, or `NORMAL_OCT16P` decoded, turned Y up.
//! - `BATCH_ID` becomes the `_BATCHID` attribute, and the batch table is kept in the node's
//!   extras with one value for each batch. Without batch ids the batch table is per point: its
//!   binary properties (and JSON lists of numbers) become float custom attributes named for
//!   them (`_TEMPERATURE`; a vector's components `_COLOR_0`, `_COLOR_1`...), and what's left
//!   stays in the extras.
//! - `3DTILES_draco_point_compression` is decoded with the `draco` feature: the properties it
//!   names come from the Draco data, the rest from the binary bodies as usual.

use serde_json::{Map, Value, json};

use super::table::{Component, Tile};
use super::{Ctx, Stats, batch_extras, oct_decode, table, unset, y_up};
use crate::document::Document;
use crate::formats::points::{self, Cloud, Column};
use crate::num::{f32_of, usize_of};
use crate::source::SourceInfo;
use crate::{Context as _, ErrorKind, Result};

/// The extension that holds some properties as Draco data in the feature table's body.
const DRACO: &str = "3DTILES_draco_point_compression";

pub(super) fn read(ctx: &Ctx<'_>, data: &[u8], stats: &mut Stats) -> Result<Document> {
    let tile = table::parse(data, "pnts")?;
    let ft = &tile.feature;
    let count = ft
        .count("POINTS_LENGTH")?
        .context(ErrorKind::Corrupt, "pnts tile has no POINTS_LENGTH")?;
    points::check_count(count, ctx.keep.points(), "pnts")?;
    let n = usize_of(count);
    let draco = Draco::decode(&tile, n)?;
    // A property from the Draco data when it's there, else from the body.
    let column = |key: &str, comps: usize, component: Component, typed: bool| -> Result<Option<Vec<f32>>> {
        match draco.feature(key, n, comps)? {
            Some(v) => Ok(Some(v)),
            None => ft.floats(key, n, comps, component, typed),
        }
    };

    let rtc = ft.vec3("RTC_CENTER")?.unwrap_or([0.0; 3]);
    let mut base = rtc;
    let mut positions: Vec<[f32; 3]>;
    if let Some(p) = column("POSITION", 3, Component::F32, false)? {
        positions = p.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        if unset(base) {
            base = middle(&positions);
            if !unset(base) {
                for p in &mut positions {
                    *p = [0, 1, 2].map(|k| f32_of(f64::from(p[k]) - base[k]));
                }
            }
        }
    } else if let Some(q) = ft.floats("POSITION_QUANTIZED", n, 3, Component::U16, false)? {
        let (Some(offset), Some(scale)) = (ft.vec3("QUANTIZED_VOLUME_OFFSET")?, ft.vec3("QUANTIZED_VOLUME_SCALE")?)
        else {
            bail!(
                Corrupt,
                "pnts tile has POSITION_QUANTIZED without QUANTIZED_VOLUME_OFFSET and QUANTIZED_VOLUME_SCALE"
            );
        };
        base = [0, 1, 2].map(|k| rtc[k] + offset[k]);
        positions = q
            .chunks_exact(3)
            .map(|q| [0, 1, 2].map(|k| f32_of(f64::from(q[k]) * scale[k] / 65535.0)))
            .collect();
    } else {
        bail!(Corrupt, "pnts tile has neither POSITION nor POSITION_QUANTIZED");
    }
    for p in &mut positions {
        *p = [p[0], p[2], -p[1]];
    }

    let byte = |v: f32| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to a byte's range first"
        )]
        let b = v.round().clamp(0.0, 255.0) as u8;
        b
    };
    let colors: Vec<[u16; 4]> = if let Some(c) = column("RGBA", 4, Component::U8, false)? {
        c.chunks_exact(4)
            .map(|c| {
                let [r, g, b, _] = points::linear8([byte(c[0]), byte(c[1]), byte(c[2])]);
                [r, g, b, u16::from(byte(c[3])) * 257]
            })
            .collect()
    } else if let Some(c) = column("RGB", 3, Component::U8, false)? {
        c.chunks_exact(3)
            .map(|c| points::linear8([byte(c[0]), byte(c[1]), byte(c[2])]))
            .collect()
    } else if let Some(c) = ft.floats("RGB565", n, 1, Component::U16, false)? {
        c.iter()
            .map(|&v| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "read from 16 bits"
                )]
                let v = v as u16;
                let rgb = [
                    f32::from(v >> 11) / 31.0,
                    f32::from((v >> 5) & 63) / 63.0,
                    f32::from(v & 31) / 31.0,
                ];
                points::linear_f(rgb, 1.0)
            })
            .collect()
    } else if let Some(c) = ft.global("CONSTANT_RGBA", 4, Component::U8)? {
        let c: Vec<u8> = c.iter().map(|&c| byte(f32_of(c))).collect();
        let [r, g, b, _] = points::linear8([c[0], c[1], c[2]]);
        vec![[r, g, b, u16::from(c[3]) * 257]; n]
    } else {
        Vec::new()
    };

    let normals: Vec<[f32; 3]> = if let Some(v) = column("NORMAL", 3, Component::F32, false)? {
        v.chunks_exact(3).map(|v| [v[0], v[2], -v[1]]).collect()
    } else if let Some(v) = ft.floats("NORMAL_OCT16P", n, 2, Component::U8, false)? {
        v.chunks_exact(2)
            .map(|v| y_up(oct_decode(f64::from(v[0]), f64::from(v[1]), 255.0)).map(f32_of))
            .collect()
    } else {
        Vec::new()
    };

    let mut cloud = Cloud {
        positions,
        colors,
        normals,
        origin: y_up(base),
        ..Cloud::default()
    };
    let mut extras = json!({ "tile": "pnts" });
    if let Some(ids) = column("BATCH_ID", 1, Component::U16, true)? {
        // Points in batches: the table has a value for each batch.
        let length = ft.count("BATCH_LENGTH")?.unwrap_or(0);
        cloud.columns.push(Column {
            name: "BATCHID".into(),
            values: ids,
        });
        if ctx.keep.extras {
            extras = batch_extras("pnts", &tile.batch, length, usize_of(length))?;
        }
        stats.features += length;
    } else {
        // A value for each point: attributes, where they're numbers.
        let mut rest = Map::new();
        for (key, v) in &tile.batch.json {
            let values = match draco.batch(&tile.batch, key, v, n)? {
                Some(found) => Some(found),
                None => tile.batch.binary(key, v, n)?,
            };
            let listed = || {
                let list = v.as_array().filter(|l| l.len() == n)?;
                let numbers: Vec<f64> = list.iter().filter_map(Value::as_f64).collect();
                (numbers.len() == n).then_some((1, numbers))
            };
            let Some((comps, values)) = values.or_else(listed) else {
                rest.insert(key.clone(), v.clone());
                continue;
            };
            for k in 0..comps {
                let name = if comps == 1 { key.clone() } else { format!("{key}_{k}") };
                let values = values.iter().skip(k).step_by(comps).map(|&x| f32_of(x)).collect();
                cloud.columns.push(Column::new(&name, values));
            }
        }
        if !rest.is_empty() {
            extras["batchTable"] = Value::Object(rest);
        }
    }

    let mut doc = cloud.document(&SourceInfo::default(), Map::new(), Map::new());
    if let Some(node) = doc.base_mut().get_mut("nodes").and_then(|n| n.get_mut(0)) {
        node["name"] = json!("pnts");
        if ctx.keep.extras {
            node["extras"] = extras;
        }
    }
    stats.kind("pnts");
    stats.tiles += 1;
    Ok(doc)
}

/// Where to measure `positions` from when nothing in the file says: zero on each axis they're
/// near zero on, else a whole number near their middle.
fn middle(positions: &[[f32; 3]]) -> [f64; 3] {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            let v = f64::from(p[k]);
            if v.is_finite() {
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
            }
        }
    }
    points::origin(min, max)
}

/// The tile's Draco data, decoded.
#[cfg(feature = "draco")]
struct Draco<'a> {
    cloud: Option<draco_core::point_cloud::PointCloud>,
    /// The feature table's properties in the data, each with its attribute's id.
    feature: Option<&'a Value>,
    /// The batch table's.
    batch: Option<&'a Value>,
}

#[cfg(feature = "draco")]
impl<'a> Draco<'a> {
    fn decode(tile: &'a Tile<'_>, n: usize) -> Result<Self> {
        use draco_core::decode_limits::DecodeLimits;
        use draco_core::decoder_buffer::DecoderBuffer;
        use draco_core::point_cloud::PointCloud;
        use draco_core::point_cloud_decoder::PointCloudDecoder;

        let Some(ext) = tile.feature.extension(DRACO) else {
            return Ok(Self {
                cloud: None,
                feature: None,
                batch: None,
            });
        };
        let at = |key: &str| {
            ext.get(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
        };
        let bytes = at("byteOffset")
            .zip(at("byteLength"))
            .and_then(|(off, len)| tile.feature.body().get(off..off.checked_add(len)?))
            .context(
                ErrorKind::Truncated,
                "pnts Draco data runs past the feature table's binary body",
            )?;
        let limits = DecodeLimits::default().with_max_points(n as u64);
        let mut buffer = DecoderBuffer::new(bytes).with_limits(limits);
        let mut cloud = PointCloud::new();
        PointCloudDecoder::new()
            .decode(&mut buffer, &mut cloud)
            .context(ErrorKind::Corrupt, "pnts Draco data doesn't decode")?;
        ensure!(
            cloud.num_points() == n,
            Corrupt,
            "pnts Draco data holds {} points, not the {n} its POINTS_LENGTH says",
            cloud.num_points()
        );
        Ok(Self {
            cloud: Some(cloud),
            feature: ext.get("properties"),
            batch: tile.batch.extension(DRACO).and_then(|e| e.get("properties")),
        })
    }

    /// `n` values of `comps` components of the attribute `properties` names for `key`.
    fn values(&self, properties: Option<&Value>, key: &str, n: usize, comps: usize) -> Result<Option<Vec<f32>>> {
        let (Some(cloud), Some(id)) = (&self.cloud, properties.and_then(|p| p.get(key))) else {
            return Ok(None);
        };
        let attribute = id
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .and_then(|id| cloud.attribute_by_unique_id(id))
            .with_context(ErrorKind::Corrupt, || {
                format!("pnts Draco data has no attribute {id} for {key}")
            })?;
        Ok(Some(attribute.read_f32s(n, comps)))
    }

    /// Feature table property `key`, when the Draco data holds it.
    fn feature(&self, key: &str, n: usize, comps: usize) -> Result<Option<Vec<f32>>> {
        self.values(self.feature, key, n, comps)
    }

    /// Batch table property `key`, declared by `v`, when the Draco data holds it.
    fn batch(&self, table: &table::Table<'_>, key: &str, v: &Value, n: usize) -> Result<Option<(usize, Vec<f64>)>> {
        if self.cloud.is_none() || self.batch.and_then(|p| p.get(key)).is_none() {
            return Ok(None);
        }
        let (comps, _) = table.kind(key, v)?;
        Ok(self
            .values(self.batch, key, n, comps)?
            .map(|v| (comps, v.into_iter().map(f64::from).collect())))
    }
}

/// Without the `draco` feature there is no decoder: a tile with Draco data is refused.
#[cfg(not(feature = "draco"))]
struct Draco;

#[cfg(not(feature = "draco"))]
#[allow(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "the same calls as with the `draco` feature"
)]
impl Draco {
    fn decode(tile: &Tile<'_>, _: usize) -> Result<Self> {
        ensure!(
            tile.feature.extension(DRACO).is_none(),
            FeatureDisabled,
            "pnts tile is Draco-compressed ({DRACO}), which needs Polysmelt's `draco` feature"
        );
        Ok(Self)
    }

    fn feature(&self, _: &str, _: usize, _: usize) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    fn batch(&self, _: &table::Table<'_>, _: &str, _: &Value, _: usize) -> Result<Option<(usize, Vec<f64>)>> {
        Ok(None)
    }
}
