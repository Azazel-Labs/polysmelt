//! Point clouds as glTF: one node, translated to where the cloud sits, drawing one `POINTS`
//! primitive. The point-cloud readers (LAS and LAZ, PCD, the text clouds, E57) fill a
//! [`Cloud`] and turn it into a document here.
//!
//! Positions are single precision relative to the node's translation, so a cloud far from the
//! origin (a survey in projected coordinates) keeps its detail: the readers pick an origin near
//! the points and the node carries it. Colors are `COLOR_0` as normalized 16-bit RGBA, linear.
//! Per-point values glTF has no attribute for (intensity, classification and the rest) are float
//! custom attributes named `_` and the value's name in capitals. A column that holds the same
//! value at every point is dropped and its value recorded in the mesh's extras instead.

use std::sync::{Mutex, OnceLock, PoisonError};

use serde_json::{Map, Value, json};

use crate::document::{Builder, Document, Part};
use crate::source::SourceInfo;
use crate::{Context as _, ErrorKind, Result};

/// A point cloud on its way to a document. Every array that isn't empty has one entry per
/// point.
#[derive(Default)]
pub(crate) struct Cloud {
    /// Positions in glTF's axes and meters, relative to [`Cloud::origin`].
    pub positions: Vec<[f32; 3]>,
    /// Linear RGBA, normalized 16-bit.
    pub colors: Vec<[u16; 4]>,
    /// Unit normals in glTF's axes.
    pub normals: Vec<[f32; 3]>,
    /// The other per-point values, as custom attributes.
    pub columns: Vec<Column>,
    /// Where the positions are measured from, in glTF's axes and meters: the node's
    /// translation.
    pub origin: [f64; 3],
    /// Values every point shares, by attribute name, kept in the mesh's extras.
    pub constant: Map<String, Value>,
}

/// One per-point value.
pub(crate) struct Column {
    /// The value's name, as the attribute takes it after `_` (capitals, digits and `_`).
    pub name: String,
    /// One value per point.
    pub values: Vec<f32>,
}

impl Column {
    /// A column named after `name`, made fit for an attribute name.
    #[cfg_attr(
        not(any(feature = "pcd", feature = "xyz", feature = "e57")),
        allow(dead_code, reason = "not every reader uses it")
    )]
    pub fn new(name: &str, values: Vec<f32>) -> Self {
        Self {
            name: attribute_name(name),
            values,
        }
    }
}

/// `name` in capitals with anything but letters and digits as `_`.
pub(crate) fn attribute_name(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "VALUE".into() } else { s }
}

/// The sRGB to linear table for 16-bit values.
fn table() -> &'static [u16] {
    static TABLE: OnceLock<Box<[u16]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|v| {
                let c = crate::mesh::srgb_to_linear(f32::from(v) / 65535.0);
                // In 0..=1, so the product is in range.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the value is in 0..=65535"
                )]
                let q = (c * 65535.0).round().clamp(0.0, 65535.0) as u16;
                q
            })
            .collect()
    })
}

/// A 16-bit sRGB color as normalized 16-bit linear RGBA, opaque.
#[inline]
pub(crate) fn linear16(rgb: [u16; 3]) -> [u16; 4] {
    let t = table();
    [
        t[usize::from(rgb[0])],
        t[usize::from(rgb[1])],
        t[usize::from(rgb[2])],
        u16::MAX,
    ]
}

/// An 8-bit sRGB color as normalized 16-bit linear RGBA, opaque.
#[cfg_attr(not(any(feature = "pcd")), allow(dead_code, reason = "not every reader uses it"))]
#[inline]
pub(crate) fn linear8(rgb: [u8; 3]) -> [u16; 4] {
    linear16(rgb.map(|c| u16::from(c) * 257))
}

/// A float sRGB color in 0..=1 as normalized 16-bit linear RGBA, with `alpha` (linear already).
#[cfg_attr(
    not(any(feature = "pcd", feature = "xyz", feature = "e57")),
    allow(dead_code, reason = "not every reader uses it")
)]
#[inline]
pub(crate) fn linear_f(rgb: [f32; 3], alpha: f32) -> [u16; 4] {
    let q = |c: f32| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=65535 first"
        )]
        let v = (c.clamp(0.0, 1.0) * 65535.0).round() as u16;
        v
    };
    let [r, g, b, _] = linear16(rgb.map(q));
    [r, g, b, q(alpha)]
}

/// An origin near the points between `min` and `max` (in any axes): per axis, the middle
/// rounded to a whole unit when the points sit far enough from zero that single precision
/// would lose detail, otherwise zero.
pub(crate) fn origin(min: [f64; 3], max: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| {
        let far = min[k].abs().max(max[k].abs());
        if far.is_finite() && far > 1024.0 && min[k] <= max[k] {
            f64::midpoint(min[k], max[k]).round()
        } else {
            0.0
        }
    })
}

/// A Z-up right-handed position (east, north, up) in glTF's Y-up axes.
#[cfg_attr(
    not(any(feature = "las", feature = "xyz", feature = "e57")),
    allow(dead_code, reason = "not every reader uses it")
)]
#[inline]
pub(crate) fn z_up(p: [f64; 3]) -> [f64; 3] {
    [p[0], p[2], -p[1]]
}

/// Keeps the items of `v` whose `keep` is true, when there's one for each.
fn retain<T>(v: &mut Vec<T>, keep: &[bool]) {
    if v.len() == keep.len() {
        let mut k = keep.iter();
        v.retain(|_| k.next().copied().unwrap_or(false));
    }
}

impl Cloud {
    /// Drops the points whose position isn't finite, with their other values.
    fn drop_broken(&mut self) {
        let n = self.positions.len();
        if self.positions.iter().all(|p| p.iter().all(|c| c.is_finite())) {
            return;
        }
        let keep: Vec<bool> = self.positions.iter().map(|p| p.iter().all(|c| c.is_finite())).collect();
        retain(&mut self.colors, &keep);
        retain(&mut self.normals, &keep);
        for c in &mut self.columns {
            retain(&mut c.values, &keep);
        }
        retain(&mut self.positions, &keep);
        debug_assert!(self.positions.len() < n);
    }

    /// The document: `info` written over `source` (the format's own details) in the asset's
    /// extras, `facts` as given with the point count added.
    pub fn document(
        mut self,
        info: &SourceInfo,
        mut source: Map<String, Value>,
        mut facts: Map<String, Value>,
    ) -> Document {
        info.write(&mut source);
        let source = Value::Object(source);
        self.drop_broken();
        let n = self.positions.len();
        facts.insert("points".into(), json!(n));
        if n == 0 {
            let mut doc = Document::empty(facts);
            doc.json["asset"]["extras"] = json!({ "source": source });
            return doc;
        }
        let mut b = Builder::default();
        let mut constant = std::mem::take(&mut self.constant);
        let mut attributes = json!({ "POSITION": b.positions(std::mem::take(&mut self.positions)) });
        if self.normals.len() == n {
            for v in &mut self.normals {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                *v = if len.is_finite() && len > 1e-12 {
                    v.map(|c| c / len)
                } else {
                    [0.0, 1.0, 0.0]
                };
            }
            let a = b.accessor(
                Part::f32(std::mem::take(&mut self.normals).into_flattened()),
                5126,
                n,
                "VEC3",
                Some(34962),
            );
            attributes["NORMAL"] = json!(a);
        }
        if self.colors.len() == n {
            let first = self.colors[0];
            if self.colors.iter().all(|&c| c == first) {
                constant.insert("COLOR_0".into(), json!(first.map(|c| f32::from(c) / 65535.0)));
            } else {
                let a = b.accessor(
                    Part::u16(std::mem::take(&mut self.colors).into_flattened()),
                    5123,
                    n,
                    "VEC4",
                    Some(34962),
                );
                b.accessors[a]["normalized"] = json!(true);
                attributes["COLOR_0"] = json!(a);
            }
        }
        for c in std::mem::take(&mut self.columns) {
            let key = format!("_{}", c.name);
            if c.values.len() != n || attributes.get(&key).is_some() || constant.contains_key(&key) {
                continue;
            }
            let first = c.values[0];
            if c.values.iter().all(|v| v.to_bits() == first.to_bits()) {
                constant.insert(key, if first.is_finite() { json!(first) } else { Value::Null });
                continue;
            }
            attributes[key] = json!(b.accessor(Part::f32(c.values), 5126, n, "SCALAR", Some(34962)));
        }
        let mut mesh = json!({ "primitives": [{ "attributes": attributes, "mode": 0 }] });
        if !constant.is_empty() {
            mesh["extras"] = json!({ "constant": constant });
        }
        let mut node = json!({ "mesh": 0 });
        if self.origin.iter().any(|&c| c != 0.0 && c.is_finite()) {
            node["translation"] = json!(self.origin.map(|c| if c.is_finite() { c } else { 0.0 }));
        }
        let mut json = Map::new();
        let mut extras = Map::new();
        extras.insert("source".into(), source);
        json.insert("asset".into(), json!({ "extras": extras }));
        json.insert("scene".into(), json!(0));
        json.insert("scenes".into(), json!([{ "nodes": [0] }]));
        json.insert("nodes".into(), json!([node]));
        json.insert("meshes".into(), json!([mesh]));
        b.finish(json, facts)
    }
}

/// Room for `n` more items, or an error where there isn't memory for them.
pub(crate) fn reserve<T>(v: &mut Vec<T>, n: usize, what: &str) -> Result<()> {
    v.try_reserve(n).with_context(ErrorKind::LimitExceeded, || {
        format!("not enough memory for the {what} points")
    })
}

/// Refuses `n` points when it's more than `max` ([`Keep::max_points`](crate::document::Keep)).
pub(crate) fn check_count(n: u64, max: usize, what: &str) -> Result<()> {
    ensure!(
        usize::try_from(n).is_ok_and(|n| n <= max),
        LimitExceeded,
        "{n} points in the {what} file, more than the {max} allowed (Keep::max_points)"
    );
    Ok(())
}

/// `f` on each of `items`, on up to `threads` threads (this one among them), the results in
/// the items' order. Helpers that can't be started leave their share to the rest: on a
/// platform without threads, this one does it all.
#[cfg_attr(
    not(any(feature = "pcd", feature = "xyz", feature = "e57")),
    allow(dead_code, reason = "not every reader uses it")
)]
pub(crate) fn map<I: Send, T: Send>(items: Vec<I>, threads: usize, f: impl Fn(I) -> T + Sync) -> Vec<T> {
    let n = items.len();
    let helpers = threads.min(n).saturating_sub(1);
    let queue = Mutex::new(items.into_iter().enumerate());
    let done: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let run = || {
        loop {
            let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
            let Some((k, item)) = next else { break };
            let t = f(item);
            done.lock().unwrap_or_else(PoisonError::into_inner)[k] = Some(t);
        }
    };
    std::thread::scope(|s| {
        for _ in 0..helpers {
            if !crate::keep::spawn(s, run) {
                break;
            }
        }
        run();
    });
    done.into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .flatten()
        .collect()
}

/// Overwrites `k` random bytes of `b`, for tests of broken files.
#[cfg(test)]
pub(crate) fn corrupt(b: &mut [u8], rng: &mut crate::test_rng::Pcg32, k: usize) {
    if b.is_empty() {
        return;
    }
    for _ in 0..k {
        let at = rng.below(u32::try_from(b.len()).unwrap_or(u32::MAX)) as usize;
        b[at] = rng.next_u32().to_le_bytes()[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_go_linear_and_keep_their_ends() {
        assert_eq!(linear8([0, 255, 0]), [0, 65535, 0, 65535]);
        let mid = linear8([128, 128, 128])[0];
        // sRGB 128 is about 0.2158 linear.
        assert!((f32::from(mid) / 65535.0 - 0.2158).abs() < 0.001, "{mid}");
        assert_eq!(linear_f([1.0, 0.0, 2.0], 0.5), [65535, 0, 65535, 32768]);
    }

    #[test]
    fn the_origin_is_zero_near_zero_and_rounded_far_away() {
        let o = origin([500_000.25, 10.0, -3000.0], [500_100.75, 20.0, -2000.0]);
        assert_eq!(o, [500_051.0, 0.0, -2500.0]);
        assert_eq!(origin([f64::NAN; 3], [f64::NAN; 3]), [0.0; 3]);
    }

    #[test]
    fn constant_columns_and_broken_points_are_dropped() {
        let cloud = Cloud {
            positions: vec![[0.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [1.0, 2.0, 3.0]],
            colors: vec![[1, 2, 3, 4]; 3],
            normals: vec![[0.0, 0.0, 2.0], [0.0; 3], [0.0; 3]],
            columns: vec![
                Column::new("intensity", vec![1.0, 2.0, 3.0]),
                Column::new("class", vec![2.0, 9.0, 2.0]),
            ],
            origin: [10.0, 0.0, 0.0],
            constant: Map::new(),
        };
        let info = SourceInfo {
            format: Some("test".into()),
            ..SourceInfo::default()
        };
        let doc = cloud.document(&info, Map::new(), Map::new());
        assert_eq!(doc.source_info().format.as_deref(), Some("test"));
        assert_eq!(doc.facts["points"], 2);
        let mesh = &doc.json["meshes"][0];
        let attrs = &mesh["primitives"][0]["attributes"];
        assert!(attrs.get("_INTENSITY").is_some());
        assert!(attrs.get("_CLASS").is_none());
        assert!(attrs.get("COLOR_0").is_none());
        assert_eq!(mesh["extras"]["constant"]["_CLASS"], 2.0);
        assert_eq!(doc.json["nodes"][0]["translation"], json!([10.0, 0.0, 0.0]));
        let m = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        assert_eq!(m.prims.iter().map(|p| p.points.len()).sum::<usize>(), 2);
    }
}
