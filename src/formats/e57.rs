//! E57 laser scans (ASTM E2807), read with the `e57` crate: every scan in the file, posed where
//! the file puts it, in one `POINTS` primitive.
//!
//! Cartesian positions are used where they're valid, spherical ones (range, azimuth, elevation)
//! turned Cartesian where they aren't; points with neither are left out and counted in the facts.
//! Each scan's pose (a rotation and translation) is applied in double precision. E57 is in
//! meters, right-handed and Z-up by convention, so positions turn Y-up. Colors are scaled by the
//! scan's color limits (or the fields' own range) and taken as sRGB; a point whose color is
//! marked invalid is transparent black. Normals from the `nor` extension are `NORMAL`.
//! Intensity is `_INTENSITY`, as stored (the limits are in the extras); row and column indices,
//! return index and count, time stamps and any other field are float custom attributes, up to
//! [`COLUMNS`] of them. A file with several scans adds `_SCAN`, the scan's index.
//!
//! The source info has the format's meters (declared), right-handed, Z up (assumed), the library
//! that wrote the file as its tool, the file's creation time (GPS time) as its creation date in
//! UTC, and the coordinate system the file names, if any, as its georeference. Each scan's
//! metadata (names, sensor, pose, limits, bounds, acquisition times) and the images' names and
//! projections are kept beside them. The images themselves are left in the file.

use std::cell::Cell;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::rc::Rc;

use e57::{E57Reader, PointCloud, Projection, RecordDataType, RecordName, RecordValue, Transform};
use serde_json::{Map, Value, json};

use super::dates;
use super::points::{self, Cloud, Column};
use super::{Input, ReadSeek};
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// The most custom attributes made from fields that aren't position, normal or color.
const COLUMNS: usize = 64;
/// The most bytes read without a point coming of them. The `e57` crate queues the values of a
/// field whose neighbors haven't caught up; a broken file could fill that queue without end.
/// Real files interleave their fields packet by packet, and a packet is at most 64 KiB.
const STALL: u64 = 1 << 20;

/// Bytes read since the last point, and how many are allowed.
struct Meter {
    since: Cell<u64>,
    limit: Cell<u64>,
}

/// A stream that counts what's read from it and stops when [`Meter::limit`] is passed.
struct Watched<R> {
    inner: R,
    meter: Rc<Meter>,
}

impl<R: Read> Read for Watched<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let m = &self.meter;
        if m.since.get() > m.limit.get() {
            return Err(io::Error::other(format!(
                "{} bytes of data without a point",
                m.since.get()
            )));
        }
        let n = self.inner.read(buf)?;
        m.since.set(m.since.get().saturating_add(n as u64));
        Ok(n)
    }
}

impl<R: Seek> Seek for Watched<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

type Reader<'a> = E57Reader<Watched<BufReader<Box<dyn ReadSeek + 'a>>>>;

/// The file opened, with its metadata read.
fn open<'a>(input: &Input<'a>) -> Result<(Reader<'a>, Rc<Meter>)> {
    let meter = Rc::new(Meter {
        since: Cell::new(0),
        limit: Cell::new(u64::MAX),
    });
    let stream = input.seekable().context(ErrorKind::Corrupt, "reading the E57 file")?;
    let inner = BufReader::with_capacity(1 << 16, stream);
    let reader = E57Reader::new(Watched {
        inner,
        meter: Rc::clone(&meter),
    })
    .context(ErrorKind::Corrupt, "not a readable E57 file")?;
    Ok((reader, meter))
}

pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document> {
    let (mut reader, meter) = open(&input)?;
    let scans = reader.pointclouds();
    let total = scans.iter().fold(0_u64, |a, s| a.saturating_add(s.records));
    points::check_count(total, keep.points(), "E57")?;
    let threads = keep.thread_count();
    let parts: Vec<Part> = if threads > 1 && scans.len() > 1 {
        // Each thread opens the file for itself: a reader holds its place in the file.
        let jobs: Vec<usize> = (0..scans.len()).collect();
        points::map(jobs, threads, |k| {
            let (mut r, m) = open(&input)?;
            scan(&mut r, &m, &scans[k])
        })
        .into_iter()
        .collect::<Result<_>>()?
    } else {
        scans
            .iter()
            .map(|s| scan(&mut reader, &meter, s))
            .collect::<Result<_>>()?
    };
    document(&reader, &scans, parts)
}

/// Where each kind of value is in a scan's records.
#[derive(Default)]
struct Fields {
    cartesian: Option<[usize; 3]>,
    cartesian_state: Option<usize>,
    /// Range, azimuth, elevation.
    spherical: Option<[usize; 3]>,
    spherical_state: Option<usize>,
    color: Option<[usize; 3]>,
    color_bad: Option<usize>,
    normal: Option<[usize; 3]>,
    /// The custom attributes: the field, the field that marks its value invalid, the name.
    columns: Vec<(usize, Option<usize>, String)>,
    /// Fields past [`COLUMNS`].
    dropped: Vec<String>,
}

impl Fields {
    fn new(pc: &PointCloud) -> Fields {
        let at = |name: RecordName| pc.prototype.iter().position(|r| r.name == name);
        let unknown = |want: &str| {
            pc.prototype
                .iter()
                .position(|r| matches!(&r.name, RecordName::Unknown { name, .. } if name == want))
        };
        let three = |a: Option<usize>, b: Option<usize>, c: Option<usize>| Some([a?, b?, c?]);
        let mut f = Fields {
            cartesian: three(
                at(RecordName::CartesianX),
                at(RecordName::CartesianY),
                at(RecordName::CartesianZ),
            ),
            cartesian_state: at(RecordName::CartesianInvalidState),
            spherical: three(
                at(RecordName::SphericalRange),
                at(RecordName::SphericalAzimuth),
                at(RecordName::SphericalElevation),
            ),
            spherical_state: at(RecordName::SphericalInvalidState),
            color: three(
                at(RecordName::ColorRed),
                at(RecordName::ColorGreen),
                at(RecordName::ColorBlue),
            ),
            color_bad: at(RecordName::IsColorInvalid),
            normal: three(unknown("normalX"), unknown("normalY"), unknown("normalZ")),
            ..Fields::default()
        };
        let mut used: Vec<usize> = [f.cartesian, f.spherical, f.color, f.normal]
            .into_iter()
            .flatten()
            .flatten()
            .chain(
                [f.cartesian_state, f.spherical_state, f.color_bad]
                    .into_iter()
                    .flatten(),
            )
            .collect();
        let intensity_bad = at(RecordName::IsIntensityInvalid);
        let time_bad = at(RecordName::IsTimeStampInvalid);
        used.extend(intensity_bad);
        used.extend(time_bad);
        // Intensity first, then the rest in the file's order.
        if let Some(i) = at(RecordName::Intensity) {
            f.columns.push((i, intensity_bad, "INTENSITY".into()));
            used.push(i);
        }
        for (i, r) in pc.prototype.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            let (name, bad) = match &r.name {
                RecordName::RowIndex => ("ROW".to_string(), None),
                RecordName::ColumnIndex => ("COLUMN".into(), None),
                RecordName::ReturnIndex => ("RETURN_INDEX".into(), None),
                RecordName::ReturnCount => ("RETURN_COUNT".into(), None),
                RecordName::TimeStamp => ("TIME_STAMP".into(), time_bad),
                other => (points::attribute_name(&field_name(other)), None),
            };
            let taken = f.columns.iter().any(|c| c.2 == name) || matches!(name.as_str(), "SCAN" | "");
            if f.columns.len() >= COLUMNS || taken {
                f.dropped.push(field_name(&r.name));
            } else {
                f.columns.push((i, bad, name));
            }
        }
        f
    }
}

/// The name a field has in the file.
fn field_name(n: &RecordName) -> String {
    match n {
        RecordName::CartesianX => "cartesianX",
        RecordName::CartesianY => "cartesianY",
        RecordName::CartesianZ => "cartesianZ",
        RecordName::CartesianInvalidState => "cartesianInvalidState",
        RecordName::SphericalRange => "sphericalRange",
        RecordName::SphericalAzimuth => "sphericalAzimuth",
        RecordName::SphericalElevation => "sphericalElevation",
        RecordName::SphericalInvalidState => "sphericalInvalidState",
        RecordName::Intensity => "intensity",
        RecordName::IsIntensityInvalid => "isIntensityInvalid",
        RecordName::ColorRed => "colorRed",
        RecordName::ColorGreen => "colorGreen",
        RecordName::ColorBlue => "colorBlue",
        RecordName::IsColorInvalid => "isColorInvalid",
        RecordName::RowIndex => "rowIndex",
        RecordName::ColumnIndex => "columnIndex",
        RecordName::ReturnCount => "returnCount",
        RecordName::ReturnIndex => "returnIndex",
        RecordName::TimeStamp => "timeStamp",
        RecordName::IsTimeStampInvalid => "isTimeStampInvalid",
        RecordName::Unknown { namespace, name } => {
            return if namespace.is_empty() {
                name.clone()
            } else {
                format!("{namespace}:{name}")
            };
        }
    }
    .into()
}

/// A raw value as a number.
#[inline]
fn num(v: &RecordValue, dt: &RecordDataType) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "64-bit integers go to the nearest double")]
    match (v, dt) {
        (RecordValue::Single(f), _) => f64::from(*f),
        (RecordValue::Double(d), _) => *d,
        (RecordValue::ScaledInteger(i), RecordDataType::ScaledInteger { scale, offset, .. }) => {
            *i as f64 * scale + offset
        }
        (RecordValue::ScaledInteger(i) | RecordValue::Integer(i), _) => *i as f64,
    }
}

/// A limit's value as a number, for JSON and color scaling.
fn limit(v: Option<&RecordValue>) -> Option<f64> {
    #[allow(clippy::cast_precision_loss, reason = "64-bit integers go to the nearest double")]
    match v? {
        RecordValue::Single(f) => Some(f64::from(*f)),
        RecordValue::Double(d) => Some(*d),
        RecordValue::Integer(i) | RecordValue::ScaledInteger(i) => Some(*i as f64),
    }
}

/// The span a field's values take: its limits where the scan gives them, otherwise the field's
/// own range, otherwise 0 to 1. Returns the low end and one over the width.
fn span(lo: Option<&RecordValue>, hi: Option<&RecordValue>, dt: &RecordDataType) -> (f64, f64) {
    // Scaled limits don't say their scale; the field's range stands in for them.
    let given = match (lo, hi) {
        (Some(RecordValue::ScaledInteger(_)), _) | (_, Some(RecordValue::ScaledInteger(_))) => None,
        _ => limit(lo).zip(limit(hi)),
    };
    #[allow(clippy::cast_precision_loss, reason = "64-bit integers go to the nearest double")]
    let (a, b) = given.unwrap_or(match *dt {
        RecordDataType::Integer { min, max } => (min as f64, max as f64),
        RecordDataType::ScaledInteger {
            min,
            max,
            scale,
            offset,
        } => (min as f64 * scale + offset, max as f64 * scale + offset),
        RecordDataType::Single { min, max } => (min.map_or(0.0, f64::from), max.map_or(1.0, f64::from)),
        RecordDataType::Double { min, max } => (min.unwrap_or(0.0), max.unwrap_or(1.0)),
    });
    let w = b - a;
    (a, if w.is_finite() && w > 0.0 { 1.0 / w } else { 0.0 })
}

/// A scan's pose as a row-major rotation and a translation.
fn pose(t: Option<&Transform>) -> ([f64; 9], [f64; 3]) {
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let Some(t) = t else {
        return (identity, [0.0; 3]);
    };
    let q = &t.rotation;
    let len = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
    let rotation = if len.is_finite() && len > 1e-12 {
        let (w, x, y, z) = (q.w / len, q.x / len, q.y / len, q.z / len);
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    } else {
        identity
    };
    let p = &t.translation;
    let translation = [p.x, p.y, p.z].map(|c| if c.is_finite() { c } else { 0.0 });
    (rotation, translation)
}

#[inline]
fn rotate(m: &[f64; 9], p: [f64; 3]) -> [f64; 3] {
    [
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
        m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
        m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
    ]
}

/// One scan's points, relative to [`Part::origin`].
struct Part {
    origin: [f64; 3],
    positions: Vec<[f32; 3]>,
    colors: Vec<[u16; 4]>,
    normals: Vec<[f32; 3]>,
    columns: Vec<Column>,
    /// Points without a valid position.
    dropped: u64,
    /// Fields that weren't kept.
    dropped_fields: Vec<String>,
}

/// Reads one scan's points.
fn scan(reader: &mut Reader<'_>, meter: &Meter, pc: &PointCloud) -> Result<Part> {
    let f = Fields::new(pc);
    ensure!(
        f.cartesian.is_some() || f.spherical.is_some(),
        Corrupt,
        "an E57 scan has neither Cartesian nor spherical positions"
    );
    // When every field is constant, the `e57` crate can't tell how many values a packet holds.
    let constant = |dt: &RecordDataType| match *dt {
        RecordDataType::Integer { min, max } | RecordDataType::ScaledInteger { min, max, .. } => max <= min,
        _ => false,
    };
    let records = usize::try_from(pc.records).unwrap_or(usize::MAX);
    let mut part = Part {
        origin: [0.0; 3],
        positions: Vec::new(),
        colors: Vec::new(),
        normals: Vec::new(),
        columns: f.columns.iter().map(|c| Column::new(&c.2, Vec::new())).collect(),
        dropped: 0,
        dropped_fields: f.dropped.clone(),
    };
    if records == 0 || pc.prototype.iter().all(|r| constant(&r.data_type)) {
        part.dropped = pc.records;
        return Ok(part);
    }
    let types: Vec<RecordDataType> = pc.prototype.iter().map(|r| r.data_type.clone()).collect();
    let (rotation, translation) = pose(pc.transform.as_ref());
    let channels = f.color.map(|c| {
        let l = pc.color_limits.as_ref();
        [
            span(
                l.and_then(|l| l.red_min.as_ref()),
                l.and_then(|l| l.red_max.as_ref()),
                &types[c[0]],
            ),
            span(
                l.and_then(|l| l.green_min.as_ref()),
                l.and_then(|l| l.green_max.as_ref()),
                &types[c[1]],
            ),
            span(
                l.and_then(|l| l.blue_min.as_ref()),
                l.and_then(|l| l.blue_max.as_ref()),
                &types[c[2]],
            ),
        ]
    });
    // Room for the points up front, as far as memory and a sane guess allow; more as they come.
    let guess = records.min(1 << 22);
    points::reserve(&mut part.positions, guess, "E57")?;
    if f.color.is_some() {
        points::reserve(&mut part.colors, guess, "E57")?;
    }
    if f.normal.is_some() {
        points::reserve(&mut part.normals, guess, "E57")?;
    }
    for c in &mut part.columns {
        points::reserve(&mut c.values, guess, "E57")?;
    }
    let value = |p: &[RecordValue], k: usize| p.get(k).map_or(0.0, |v| num(v, &types[k]));
    let flag = |p: &[RecordValue], k: Option<usize>| k.is_some_and(|k| value(p, k) != 0.0);
    let mut origin = None;
    meter.since.set(0);
    meter.limit.set(STALL);
    let points = reader
        .pointcloud_raw(pc)
        .context(ErrorKind::Corrupt, "reading an E57 scan")?;
    for p in points {
        meter.since.set(0);
        let p = p.context(ErrorKind::Corrupt, "reading an E57 scan")?;
        let local = match (f.cartesian, f.spherical) {
            (Some(c), _) if !flag(&p, f.cartesian_state) => Some(c.map(|k| value(&p, k))),
            (_, Some([r, a, e])) if !flag(&p, f.spherical_state) => {
                let (r, a, e) = (value(&p, r), value(&p, a), value(&p, e));
                Some([r * e.cos() * a.cos(), r * e.cos() * a.sin(), r * e.sin()])
            }
            _ => None,
        };
        let Some(local) = local.filter(|l| l.iter().all(|c| c.is_finite())) else {
            part.dropped += 1;
            continue;
        };
        let r = rotate(&rotation, local);
        let world = points::z_up([r[0] + translation[0], r[1] + translation[1], r[2] + translation[2]]);
        let o = *origin.get_or_insert_with(|| points::origin(world, world));
        part.positions.push(std::array::from_fn(|k| f32_of(world[k] - o[k])));
        if let (Some(c), Some(s)) = (f.color, &channels) {
            part.colors.push(if flag(&p, f.color_bad) {
                [0; 4]
            } else {
                let rgb: [f32; 3] = std::array::from_fn(|k| f32_of((value(&p, c[k]) - s[k].0) * s[k].1));
                points::linear_f(rgb, 1.0)
            });
        }
        if let Some(n) = f.normal {
            let n = points::z_up(rotate(&rotation, n.map(|k| value(&p, k))));
            part.normals.push(n.map(f32_of));
        }
        for (col, (k, bad, _)) in part.columns.iter_mut().zip(&f.columns) {
            col.values
                .push(if flag(&p, *bad) { 0.0 } else { f32_of(value(&p, *k)) });
        }
    }
    meter.limit.set(u64::MAX);
    part.origin = origin.unwrap_or([0.0; 3]);
    Ok(part)
}

/// The scans together in one document.
fn document(reader: &Reader<'_>, scans: &[PointCloud], parts: Vec<Part>) -> Result<Document> {
    let n: usize = parts.iter().map(|p| p.positions.len()).sum();
    let origin = parts
        .iter()
        .find(|p| !p.positions.is_empty())
        .map_or([0.0; 3], |p| p.origin);
    let any_color = parts.iter().any(|p| !p.colors.is_empty());
    let all_normals = parts.iter().all(|p| p.positions.is_empty() || !p.normals.is_empty());
    let parts_had_normals = parts.iter().any(|p| !p.normals.is_empty());
    let mut names: Vec<String> = Vec::new();
    for p in &parts {
        for c in &p.columns {
            if !names.contains(&c.name) {
                names.push(c.name.clone());
            }
        }
    }
    let many = parts.len() > 1;
    let mut cloud = Cloud {
        origin,
        ..Cloud::default()
    };
    points::reserve(&mut cloud.positions, n, "E57")?;
    let mut columns: Vec<Column> = names.iter().map(|s| Column::new(s, Vec::new())).collect();
    let mut scan_column = Vec::new();
    let mut dropped = 0;
    let mut dropped_fields: Vec<String> = Vec::new();
    for (k, mut p) in parts.into_iter().enumerate() {
        let m = p.positions.len();
        dropped += p.dropped;
        for d in p.dropped_fields.drain(..) {
            if !dropped_fields.contains(&d) {
                dropped_fields.push(d);
            }
        }
        // Each scan is measured from its own origin; move it to the first one's.
        let shift = std::array::from_fn::<f32, 3, _>(|a| f32_of(p.origin[a] - origin[a]));
        if shift.iter().all(|&c| c == 0.0) {
            cloud.positions.append(&mut p.positions);
        } else {
            cloud
                .positions
                .extend(p.positions.iter().map(|q| std::array::from_fn(|a| q[a] + shift[a])));
        }
        if any_color {
            if p.colors.len() == m {
                cloud.colors.append(&mut p.colors);
            } else {
                cloud.colors.extend(std::iter::repeat_n([0; 4], m));
            }
        }
        if all_normals && p.normals.len() == m {
            cloud.normals.append(&mut p.normals);
        }
        for col in &mut columns {
            match p.columns.iter_mut().find(|c| c.name == col.name) {
                Some(c) if c.values.len() == m => col.values.append(&mut c.values),
                _ => col.values.extend(std::iter::repeat_n(0.0, m)),
            }
        }
        if many {
            #[allow(clippy::cast_precision_loss, reason = "scan counts are small")]
            scan_column.extend(std::iter::repeat_n(k as f32, m));
        }
    }
    if many {
        columns.push(Column::new("SCAN", scan_column));
    }
    cloud.columns = columns;

    let scan_list: Vec<Value> = scans.iter().map(describe).collect();
    let mut source = json!({ "scans": scan_list });
    let h = reader.header();
    source["guid"] = json!(reader.guid());
    // Meters and right-handed by the standard; Z up by convention.
    let mut info = SourceInfo {
        format: Some("E57".into()),
        version: Some(format!("{}.{}", h.major, h.minor)),
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Assumed(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool: reader
            .library_version()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_owned),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    if let Some(c) = reader.creation() {
        info.created = dates::gps(c.gps_time).filter(|_| c.gps_time > 0.0);
        if info.created.is_none() {
            source["creation"] = json!({ "gpsTime": c.gps_time, "atomicReference": c.atomic_reference });
        } else if c.atomic_reference {
            source["creationAtomicReference"] = json!(true);
        }
    }
    let images: Vec<Value> = reader
        .images()
        .iter()
        .map(|i| {
            let mut v = json!({
                "projection": match &i.projection {
                    Some(Projection::Pinhole(_)) => "pinhole",
                    Some(Projection::Spherical(_)) => "spherical",
                    Some(Projection::Cylindrical(_)) => "cylindrical",
                    None => "none",
                },
                "visualReference": i.visual_reference.is_some(),
            });
            put(&mut v, "name", i.name.as_ref());
            put(&mut v, "guid", i.guid.as_ref());
            put(&mut v, "scanGuid", i.pointcloud_guid.as_ref());
            put(&mut v, "description", i.description.as_ref());
            if let Some(t) = &i.transform {
                v["pose"] = pose_json(t);
            }
            v
        })
        .collect();
    if !images.is_empty() {
        source["images"] = json!(images);
        source["imagesRead"] = json!(false);
    }
    let extensions: Vec<Value> = reader
        .extensions()
        .iter()
        .map(|e| json!({ "namespace": e.namespace, "url": e.url }))
        .collect();
    if !extensions.is_empty() {
        source["extensions"] = json!(extensions);
    }
    if !dropped_fields.is_empty() {
        source["droppedFields"] = json!(dropped_fields);
    }
    if !all_normals && parts_had_normals {
        source["normals"] = json!("dropped: not every scan has them");
    }
    let mut facts = Map::new();
    if dropped > 0 {
        facts.insert("dropped_points".into(), json!(dropped));
    }
    if let Some(w) = reader.coordinate_metadata().map(str::trim).filter(|w| !w.is_empty()) {
        // glTF's axes back to the file's: x, -z, y.
        let source_origin = [origin[0], -origin[2], origin[1]].map(|c| c + 0.0);
        let mut g = json!({
            "wkt": w,
            "origin": source_origin,
            "units": "meters",
            "axes": "Z-up source turned Y-up: glTF x = east, y = up, z = south",
        });
        if let Some(crs) = super::wkt::Crs::parse(w) {
            g["kind"] = json!(crs.kind);
            if let Some(e) = crs.epsg {
                g["epsg"] = json!(e);
            }
            if let Some(e) = crs.vertical_epsg {
                g["verticalEpsg"] = json!(e);
            }
        }
        info.georeference = Some(g);
    }
    let Value::Object(source) = source else { unreachable!() };
    Ok(cloud.document(&info, source, facts))
}

fn put(v: &mut Value, key: &str, s: Option<&String>) {
    if let Some(s) = s.filter(|s| !s.is_empty()) {
        v[key] = json!(s);
    }
}

fn pose_json(t: &Transform) -> Value {
    let (q, p) = (&t.rotation, &t.translation);
    json!({ "rotation": [q.x, q.y, q.z, q.w], "translation": [p.x, p.y, p.z] })
}

/// What a scan says about itself.
fn describe(pc: &PointCloud) -> Value {
    let mut v = json!({
        "points": pc.records,
        "fields": pc.prototype.iter().map(|r| field_name(&r.name)).collect::<Vec<_>>(),
    });
    put(&mut v, "name", pc.name.as_ref());
    put(&mut v, "description", pc.description.as_ref());
    put(&mut v, "guid", pc.guid.as_ref());
    let mut sensor = json!({});
    put(&mut sensor, "vendor", pc.sensor_vendor.as_ref());
    put(&mut sensor, "model", pc.sensor_model.as_ref());
    put(&mut sensor, "serial", pc.sensor_serial.as_ref());
    put(&mut sensor, "hardwareVersion", pc.sensor_hw_version.as_ref());
    put(&mut sensor, "softwareVersion", pc.sensor_sw_version.as_ref());
    put(&mut sensor, "firmwareVersion", pc.sensor_fw_version.as_ref());
    if sensor.as_object().is_some_and(|o| !o.is_empty()) {
        v["sensor"] = sensor;
    }
    for (key, x) in [
        ("temperature", pc.temperature),
        ("humidity", pc.humidity),
        ("atmosphericPressure", pc.atmospheric_pressure),
    ] {
        if let Some(x) = x {
            v[key] = json!(x);
        }
    }
    if let Some(t) = &pc.transform {
        v["pose"] = pose_json(t);
    }
    if let Some(t) = &pc.acquisition_start {
        v["acquisitionStart"] = json!(t.gps_time);
    }
    if let Some(t) = &pc.acquisition_end {
        v["acquisitionEnd"] = json!(t.gps_time);
    }
    if let Some(l) = &pc.intensity_limits {
        v["intensityLimits"] = json!([limit(l.intensity_min.as_ref()), limit(l.intensity_max.as_ref())]);
    }
    if let Some(l) = &pc.color_limits {
        v["colorLimits"] = json!({
            "red": [limit(l.red_min.as_ref()), limit(l.red_max.as_ref())],
            "green": [limit(l.green_min.as_ref()), limit(l.green_max.as_ref())],
            "blue": [limit(l.blue_min.as_ref()), limit(l.blue_max.as_ref())],
        });
    }
    if let Some(b) = &pc.cartesian_bounds {
        v["cartesianBounds"] = json!({
            "min": [b.x_min, b.y_min, b.z_min],
            "max": [b.x_max, b.y_max, b.z_max],
        });
    }
    if let Some(b) = &pc.spherical_bounds {
        v["sphericalBounds"] = json!({
            "range": [b.range_min, b.range_max],
            "elevation": [b.elevation_min, b.elevation_max],
            "azimuth": [b.azimuth_start, b.azimuth_end],
        });
    }
    if let Some(b) = &pc.index_bounds {
        v["indexBounds"] = json!({
            "row": [b.row_min, b.row_max],
            "column": [b.column_min, b.column_max],
            "return": [b.return_min, b.return_max],
        });
    }
    v
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::format_collect,
    reason = "test data is small"
)]
mod tests {
    use std::io::Cursor;

    use e57::{E57Writer, Extension, Quaternion, Record, Translation};

    use super::*;

    fn rec(name: RecordName, data_type: RecordDataType) -> Record {
        Record { name, data_type }
    }

    fn double() -> RecordDataType {
        RecordDataType::Double { min: None, max: None }
    }

    fn int(min: i64, max: i64) -> RecordDataType {
        RecordDataType::Integer { min, max }
    }

    type Writer<'a> = E57Writer<&'a mut Cursor<Vec<u8>>>;

    /// A scan of `n` points on a line starting at `far` in x, with intensity, 8-bit colors and
    /// row indices.
    fn line(w: &mut Writer<'_>, guid: &str, n: usize, pose: Option<Transform>, far: f64) {
        let prototype = vec![
            rec(RecordName::CartesianX, double()),
            rec(RecordName::CartesianY, double()),
            rec(RecordName::CartesianZ, double()),
            rec(RecordName::Intensity, RecordDataType::Single { min: None, max: None }),
            rec(RecordName::ColorRed, int(0, 255)),
            rec(RecordName::ColorGreen, int(0, 255)),
            rec(RecordName::ColorBlue, int(0, 255)),
            rec(RecordName::RowIndex, int(0, 1 << 20)),
        ];
        let mut pc = w.add_pointcloud(guid, prototype).unwrap();
        pc.set_name(Some(format!("scan {guid}")));
        pc.set_sensor_vendor(Some("test".into()));
        pc.set_transform(pose);
        for i in 0..n {
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_wrap,
                reason = "small test numbers"
            )]
            pc.add_point(vec![
                RecordValue::Double(far + i as f64 * 0.001),
                RecordValue::Double(2.0),
                RecordValue::Double(3.0 + i as f64 * 0.5),
                RecordValue::Single((i % 10) as f32 / 10.0),
                RecordValue::Integer((i % 256) as i64),
                RecordValue::Integer(255),
                RecordValue::Integer(0),
                RecordValue::Integer(i as i64),
            ])
            .unwrap();
        }
        pc.finalize().unwrap();
    }

    /// A file of line scans, `(points, pose)` each.
    fn file(scans: &[(usize, Option<Transform>)], far: f64, wkt: Option<&str>) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        {
            let mut w = E57Writer::new(&mut out, "file-guid").unwrap();
            w.set_coordinate_metadata(wkt.map(String::from));
            w.set_creation(Some(e57::DateTime {
                gps_time: 1_300_000_018.0,
                atomic_reference: false,
            }));
            for (k, (n, pose)) in scans.iter().enumerate() {
                line(&mut w, &format!("g{k}"), *n, pose.clone(), far);
            }
            w.finalize().unwrap();
        }
        out.into_inner()
    }

    fn read(bytes: &[u8], keep: Keep) -> Result<Document> {
        load(Input::memory(bytes, "t.e57"), keep)
    }

    fn data(doc: &Document, key: &str) -> Vec<u8> {
        let j = &doc.json();
        let a = j["meshes"][0]["primitives"][0]["attributes"][key].as_u64().unwrap() as usize;
        let acc = &j["accessors"][a];
        let view = &j["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let start = view.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize
            + acc.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        doc.buffer(0).unwrap()[start..start + len].to_vec()
    }

    fn floats(doc: &Document, key: &str) -> Vec<f32> {
        data(doc, key)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn u16s(doc: &Document, key: &str) -> Vec<u16> {
        data(doc, key)
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn a_scan_reads_with_its_values() {
        let doc = read(&file(&[(300, None)], 0.0, None), Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 300);
        let p = floats(&doc, "POSITION");
        // Point 2 at (0.002, 2, 4) Z-up is (0.002, 4, -2) Y-up.
        assert_eq!(&p[6..9], &[0.002, 4.0, -2.0]);
        let i = floats(&doc, "_INTENSITY");
        assert!((i[3] - 0.3).abs() < 1e-6);
        assert_eq!(floats(&doc, "_ROW")[299], 299.0);
        let c = u16s(&doc, "COLOR_0");
        // Red 255 at point 255, green full everywhere, blue none, opaque.
        assert_eq!(&c[255 * 4..256 * 4], &[65535, 65535, 0, 65535]);
        assert_eq!(c[0], 0);
        let s = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(s["format"], "E57");
        assert_eq!(s["scans"][0]["name"], "scan g0");
        assert_eq!(s["scans"][0]["sensor"]["vendor"], "test");
        let info = doc.source_info();
        assert_eq!(info.version.as_deref(), Some("1.0"));
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
        assert!(info.converted.up_axis && !info.converted.units);
        assert!(info.tool.as_deref().is_some_and(|t| t.starts_with("Rust E57 Library")));
        assert_eq!(info.created.as_deref(), Some("2021-03-17T07:06:40Z"));
        assert!(info.georeference.is_none());
        assert!(
            doc.json()["meshes"][0]["primitives"][0]["attributes"]
                .get("_SCAN")
                .is_none()
        );
    }

    #[test]
    fn scans_are_posed_and_numbered() {
        // A quarter turn about z and a move: local (x, y, z) lands at (10 - y, 20 + x, z).
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let turn = Transform {
            rotation: Quaternion {
                w: h,
                x: 0.0,
                y: 0.0,
                z: h,
            },
            translation: Translation {
                x: 10.0,
                y: 20.0,
                z: 0.0,
            },
        };
        let bytes = file(&[(5, None), (5, Some(turn))], 0.0, None);
        for threads in [1, 2] {
            let doc = read(&bytes, Keep::ALL.with_threads(threads)).unwrap();
            assert_eq!(doc.facts["points"], 10);
            let p = floats(&doc, "POSITION");
            // The second scan's first point, local (0, 2, 3): file (8, 20, 3), glTF (8, 3, -20).
            for (a, b) in p[15..18].iter().zip([8.0, 3.0, -20.0]) {
                assert!((a - b).abs() < 1e-5, "{:?}", &p[15..18]);
            }
            let scan = floats(&doc, "_SCAN");
            assert_eq!((scan[4], scan[5]), (0.0, 1.0));
            assert!(doc.json()["asset"]["extras"]["source"]["scans"][1]["pose"].is_object());
        }
    }

    #[test]
    fn far_scans_keep_their_detail_and_their_coordinate_system() {
        let wkt = r#"PROJCS["WGS 84 / UTM zone 17N",GEOGCS["WGS 84"],UNIT["metre",1],AUTHORITY["EPSG","32617"]]"#;
        let doc = read(&file(&[(50, None)], 500_000.0, Some(wkt)), Keep::ALL).unwrap();
        let t = doc.json()["nodes"][0]["translation"].as_array().unwrap();
        assert_eq!(t[0], 500_000.0);
        let p = floats(&doc, "POSITION");
        // A millimeter apart, half a million meters out.
        assert!((p[3] - p[0] - 0.001).abs() < 1e-6, "{} {}", p[0], p[3]);
        assert!(doc.facts.get("georeference").is_none());
        let g = &doc.source_info().georeference.unwrap();
        assert_eq!(g["epsg"], 32617);
        assert_eq!(g["kind"], "projected");
        assert_eq!(g["origin"][0], 500_000.0);
    }

    #[test]
    fn spherical_positions_normals_and_invalid_points() {
        let mut out = Cursor::new(Vec::new());
        {
            let mut w = E57Writer::new(&mut out, "f").unwrap();
            w.register_extension(Extension::new(
                "nor",
                "http://www.libe57.org/E57_EXT_surface_normals.txt",
            ))
            .unwrap();
            let unknown = |name: &str| RecordName::Unknown {
                namespace: "nor".into(),
                name: name.into(),
            };
            let unit = || RecordDataType::Single {
                min: Some(-1.0),
                max: Some(1.0),
            };
            let prototype = vec![
                rec(RecordName::SphericalRange, double()),
                rec(RecordName::SphericalAzimuth, double()),
                rec(RecordName::SphericalElevation, double()),
                rec(RecordName::SphericalInvalidState, int(0, 2)),
                rec(RecordName::Intensity, int(0, 4095)),
                rec(RecordName::IsIntensityInvalid, int(0, 1)),
                rec(unknown("normalX"), unit()),
                rec(unknown("normalY"), unit()),
                rec(unknown("normalZ"), unit()),
            ];
            let mut pc = w.add_pointcloud("s", prototype).unwrap();
            let half = std::f64::consts::FRAC_PI_2;
            for (range, az, el, state, i, bad) in [
                (2.0, 0.0, 0.0, 0, 100, 0),
                (3.0, half, 0.0, 0, 200, 1),
                (4.0, 0.0, half, 2, 300, 0),
                (5.0, 0.0, -half, 0, 400, 0),
            ] {
                pc.add_point(vec![
                    RecordValue::Double(range),
                    RecordValue::Double(az),
                    RecordValue::Double(el),
                    RecordValue::Integer(state),
                    RecordValue::Integer(i),
                    RecordValue::Integer(bad),
                    RecordValue::Single(0.0),
                    RecordValue::Single(0.0),
                    RecordValue::Single(1.0),
                ])
                .unwrap();
            }
            pc.finalize().unwrap();
            w.finalize().unwrap();
        }
        let doc = read(out.get_ref(), Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 3);
        assert_eq!(doc.facts["dropped_points"], 1);
        let p = floats(&doc, "POSITION");
        let want = [[2.0, 0.0, 0.0], [0.0, 0.0, -3.0], [0.0, -5.0, 0.0]];
        for (got, want) in p.chunks_exact(3).zip(want) {
            for (a, b) in got.iter().zip(want) {
                assert!((a - b).abs() < 1e-5, "{got:?} {want:?}");
            }
        }
        // Z-up normals point up in glTF.
        assert_eq!(&floats(&doc, "NORMAL")[..3], &[0.0, 1.0, 0.0]);
        // The invalid intensity is zero.
        assert_eq!(floats(&doc, "_INTENSITY"), [100.0, 0.0, 400.0]);
    }

    #[test]
    fn too_many_points_are_refused() {
        let bytes = file(&[(100, None)], 0.0, None);
        let err = read(&bytes, Keep::ALL.with_max_points(99)).unwrap_err();
        assert!(err.to_string().contains("max_points"), "{err}");
    }

    #[test]
    fn broken_files_fail_cleanly() {
        let bytes = file(&[(2000, None), (500, None)], 0.0, None);
        for n in (0..bytes.len()).step_by(997) {
            let _ = read(&bytes[..n], Keep::ALL);
        }
        let mut rng = crate::test_rng::Pcg32::new(57);
        for _ in 0..200 {
            let mut b = bytes.clone();
            points::corrupt(&mut b, &mut rng, 4);
            if let Ok(doc) = read(&b, Keep::ALL) {
                let _ = doc.to_glb();
            }
        }
        assert!(read(b"not an e57 file at all, just some text", Keep::ALL).is_err());
    }

    /// Recomputes the CRC-32C at the end of each 1024-byte page.
    fn checksum(b: &mut [u8]) {
        for page in b.chunks_exact_mut(1024) {
            let crc = !page[..1020].iter().fold(!0_u32, |c, &x| {
                (0..8).fold(c ^ u32::from(x), |c, _| {
                    if c & 1 == 0 { c >> 1 } else { (c >> 1) ^ 0x82F6_3B78 }
                })
            });
            page[1020..].copy_from_slice(&crc.to_be_bytes());
        }
    }

    #[test]
    fn scans_of_constant_fields_dont_hang() {
        // Every field the same at every point takes no bits, so packets say nothing of how
        // many points they hold. The writer won't make such a file: make one with a bit per
        // field, then narrow the ranges.
        let mut out = Cursor::new(Vec::new());
        {
            let mut w = E57Writer::new(&mut out, "f").unwrap();
            let prototype = vec![
                rec(RecordName::CartesianX, int(5, 6)),
                rec(RecordName::CartesianY, int(5, 6)),
                rec(RecordName::CartesianZ, int(5, 6)),
            ];
            let mut pc = w.add_pointcloud("c", prototype).unwrap();
            for _ in 0..10 {
                pc.add_point(vec![RecordValue::Integer(5); 3]).unwrap();
            }
            pc.finalize().unwrap();
            w.finalize().unwrap();
        }
        let mut b = out.into_inner();
        let (from, to) = (b"maximum=\"6\"", b"maximum=\"5\"");
        let mut patched = 0;
        for k in 0..b.len().saturating_sub(from.len()) {
            if &b[k..k + from.len()] == from {
                b[k..k + to.len()].copy_from_slice(to);
                patched += 1;
            }
        }
        assert_eq!(patched, 3);
        checksum(&mut b);
        let doc = read(&b, Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 0);
        assert_eq!(doc.facts["dropped_points"], 10);
    }
}
