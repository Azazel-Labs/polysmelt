//! LAS lidar point clouds, versions 1.0 to 1.4, point formats 0 to 10, and (with the `laz`
//! feature) LAZ, their compressed form.
//!
//! Each point's position is its integer coordinates times the header's scale plus its offset,
//! east, north and up. The cloud turns Y-up (east, up, south), is scaled to meters when the
//! coordinate system says feet, and is measured from an origin near the header's bounds, which
//! the node's translation carries back. Colors become `COLOR_0` (8-bit values stored in the
//! 16-bit fields are told apart by their range, as writers of both kinds exist). Intensity,
//! classification, returns, scan angle, user data, point source, GPS time, near infrared and the
//! extra bytes the file describes become custom attributes. Waveform packets are left out.
//!
//! The source info has the coordinate system's horizontal linear unit as declared units (unknown
//! when it has none), Z up and right-handed by the format, the generating software as its tool,
//! the creation day and year as its creation date, and the scale, offset, coordinate system (EPSG
//! code and WKT) and units as its georeference. The header's other fields and the variable-length
//! records' names are kept beside them. A file declaring more points than [`Keep::max_points`] is refused before any
//! are read. Points are read a block at a time, on several threads when [`Keep::threads`] allows.

use std::borrow::Cow;
use std::fmt::Write as _;
use std::sync::{Mutex, PoisonError};

use serde_json::{Map, Value, json};

use super::Input;
use super::dates;
use super::points::{self, Cloud, Column};
use crate::document::{Document, Keep};
use crate::keep::spawn;
use crate::num::{f32_of, usize_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, Result};

/// Points decoded per job when the file sets no other size (LAS, or LAZ without chunks).
const BLOCK: usize = 1 << 16;
/// The smallest header: versions 1.0 to 1.2.
const HEADER: usize = 227;
/// A variable-length record's header.
const VLR: usize = 54;
/// An extended variable-length record's header.
const EVLR: usize = 60;
/// Each point format's record size, before extra bytes.
const SIZES: [usize; 11] = [20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67];
/// Feet in meters, international and US survey.
const FOOT: f64 = 0.3048;
const US_FOOT: f64 = 1200.0 / 3937.0;

pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    let head = input.head(64 * 1024)?;
    let h = Header::parse(&head)?;
    if h.compressed {
        return compressed(input, &h, keep);
    }
    match input.bytes() {
        Some(bytes) => read(&Source::Slice(bytes), &h, keep),
        None => read(&Source::File(Mutex::new(input.seekable()?)), &h, keep),
    }
}

#[cfg(feature = "laz")]
fn compressed(input: Input<'_>, h: &Header, keep: Keep) -> Result<Document<'static>> {
    let bytes = input.read()?;
    read(&Source::Laz(&bytes), h, keep)
}

#[cfg(not(feature = "laz"))]
fn compressed(_: Input<'_>, _: &Header, _: Keep) -> Result<Document<'static>> {
    bail!("a LAZ (compressed LAS) file, and this build lacks the `laz` feature")
}

/// Where the file's bytes come from.
enum Source<'a> {
    /// All of it, in memory.
    Slice(&'a [u8]),
    /// A file, read as needed.
    File(Mutex<Box<dyn super::ReadSeek + 'a>>),
    /// A LAZ file, all of it, in memory (compressed points are small).
    #[cfg(feature = "laz")]
    Laz(&'a [u8]),
}

impl Source<'_> {
    /// The source's size.
    fn len(&self) -> Result<u64> {
        Ok(match self {
            Source::Slice(b) => b.len() as u64,
            #[cfg(feature = "laz")]
            Source::Laz(b) => b.len() as u64,
            Source::File(f) => {
                use std::io::Seek as _;
                let mut f = f.lock().unwrap_or_else(PoisonError::into_inner);
                f.seek(std::io::SeekFrom::End(0))?
            }
        })
    }

    /// `n` bytes at `at`, or an error where the source ends first.
    fn at(&self, at: u64, n: usize) -> Result<Cow<'_, [u8]>> {
        let slice = |b: &'_ [u8]| -> Option<Vec<u8>> {
            let at = usize_of(at);
            b.get(at..at.checked_add(n)?).map(<[u8]>::to_vec)
        };
        match self {
            Source::Slice(b) => {
                let at = usize_of(at);
                at.checked_add(n)
                    .and_then(|end| b.get(at..end))
                    .map(Cow::Borrowed)
                    .context("LAS data ends early")
            }
            #[cfg(feature = "laz")]
            Source::Laz(b) => slice(b).map(Cow::Owned).context("LAS data ends early"),
            Source::File(f) => {
                use std::io::{Read as _, Seek as _};
                let _ = slice;
                let mut f = f.lock().unwrap_or_else(PoisonError::into_inner);
                f.seek(std::io::SeekFrom::Start(at))?;
                let mut out = Vec::new();
                out.try_reserve_exact(n)
                    .ok()
                    .context("not enough memory for the LAS data")?;
                (&mut *f).take(n as u64).read_to_end(&mut out)?;
                ensure!(out.len() == n, "LAS data ends early");
                Ok(Cow::Owned(out))
            }
        }
    }
}

/// The public header block.
#[derive(Clone)]
struct Header {
    version: (u8, u8),
    file_source: u16,
    encoding: u16,
    guid: [u8; 16],
    system: String,
    software: String,
    day: u16,
    year: u16,
    size: u16,
    data_offset: u64,
    vlrs: u32,
    format: u8,
    compressed: bool,
    record: usize,
    count: u64,
    by_return: Vec<u64>,
    scale: [f64; 3],
    offset: [f64; 3],
    min: [f64; 3],
    max: [f64; 3],
    waveform: u64,
    evlr_start: u64,
    evlrs: u32,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2).map_or(0, |s| u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    b.get(at..at + 8)
        .and_then(|s| s.try_into().ok())
        .map_or(0, u64::from_le_bytes)
}

fn f64_at(b: &[u8], at: usize) -> f64 {
    f64::from_bits(u64_at(b, at))
}

/// A fixed-size text field: up to its first NUL, trimmed.
fn text(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_string()
}

impl Header {
    fn parse(b: &[u8]) -> Result<Header> {
        ensure!(
            b.len() >= HEADER && b.starts_with(b"LASF"),
            "not a LAS file (no LASF header)"
        );
        let version = (b[24], b[25]);
        ensure!(
            version.0 == 1 && version.1 <= 4,
            "LAS version {}.{} isn't one this reads",
            version.0,
            version.1
        );
        let header_size = u16_at(b, 94);
        ensure!(
            usize::from(header_size) >= HEADER,
            "the LAS header is too short ({header_size} bytes)"
        );
        let raw_format = b[104];
        let format = raw_format & 0x3f;
        ensure!(
            usize::from(format) < SIZES.len(),
            "LAS point format {format} isn't one this reads"
        );
        let record = usize::from(u16_at(b, 105));
        ensure!(
            record >= SIZES[usize::from(format)],
            "LAS point records of {record} bytes are too short for point format {format}"
        );
        let legacy = u64::from(u32_at(b, 107));
        let v14 = version.1 >= 4 && header_size >= 375 && b.len() >= 375;
        let count = if v14 && (legacy == 0 || u64_at(b, 247) > 0) {
            u64_at(b, 247)
        } else {
            legacy
        };
        let by_return = if v14 {
            (0..15).map(|k| u64_at(b, 255 + 8 * k)).collect()
        } else {
            (0..5).map(|k| u64::from(u32_at(b, 111 + 4 * k))).collect()
        };
        let three = |at: usize| [f64_at(b, at), f64_at(b, at + 8), f64_at(b, at + 16)];
        let scale = three(131);
        ensure!(
            scale.iter().all(|s| s.is_finite() && *s != 0.0),
            "the LAS scale factors aren't usable: {scale:?}"
        );
        let offset = three(155);
        ensure!(offset.iter().all(|o| o.is_finite()), "the LAS offsets aren't finite");
        let mut guid = [0; 16];
        guid.copy_from_slice(&b[8..24]);
        let has = |end: usize| usize::from(header_size) >= end && b.len() >= end;
        Ok(Header {
            version,
            file_source: u16_at(b, 4),
            encoding: u16_at(b, 6),
            guid,
            system: text(&b[26..58]),
            software: text(&b[58..90]),
            day: u16_at(b, 90),
            year: u16_at(b, 92),
            size: header_size,
            data_offset: u64::from(u32_at(b, 96)),
            vlrs: u32_at(b, 100),
            format,
            compressed: raw_format & 0x80 != 0,
            record,
            count,
            by_return,
            scale,
            offset,
            min: [f64_at(b, 187), f64_at(b, 203), f64_at(b, 219)],
            max: [f64_at(b, 179), f64_at(b, 195), f64_at(b, 211)],
            waveform: if has(235) { u64_at(b, 227) } else { 0 },
            evlr_start: if has(247) { u64_at(b, 235) } else { 0 },
            evlrs: if has(247) { u32_at(b, 243) } else { 0 },
        })
    }
}

/// A variable-length record, as the file names it.
struct Record {
    user: String,
    id: u16,
    description: String,
    data: Vec<u8>,
}

/// The variable-length records after the header and the extended ones after the points.
fn records(src: &Source<'_>, h: &Header, len: u64) -> Result<Vec<Record>> {
    let mut out = Vec::new();
    let mut at = u64::from(h.size);
    // Every record takes at least its header, so the count is bounded by the room before the
    // points.
    for _ in 0..h.vlrs {
        if at + VLR as u64 > h.data_offset.min(len) {
            break;
        }
        let head = src.at(at, VLR)?;
        let n = usize::from(u16_at(&head, 20));
        at += VLR as u64;
        let data = if at + n as u64 <= len {
            src.at(at, n)?.into_owned()
        } else {
            Vec::new()
        };
        at += n as u64;
        out.push(Record {
            user: text(&head[2..18]),
            id: u16_at(&head, 18),
            description: text(&head[22..54]),
            data,
        });
    }
    let mut at = h.evlr_start;
    if h.version.1 >= 4 && at >= h.data_offset && at > 0 {
        for _ in 0..h.evlrs {
            let Some(end) = at.checked_add(EVLR as u64).filter(|&e| e <= len) else {
                break;
            };
            let head = src.at(at, EVLR)?;
            let n = u64_at(&head, 20);
            // Waveform data and other big records aren't read; only those small enough to be
            // metadata are.
            let data = if n <= 1 << 20 && end + n <= len {
                src.at(end, usize_of(n))?.into_owned()
            } else {
                Vec::new()
            };
            out.push(Record {
                user: text(&head[2..18]),
                id: u16_at(&head, 18),
                description: text(&head[28..60]),
                data,
            });
            at = end.saturating_add(n);
        }
    }
    Ok(out)
}

/// How a per-point value is read from a record.
#[derive(Clone, Copy)]
enum Get {
    U8(usize),
    U16(usize),
    /// `(byte >> shift) & mask`.
    Bits(usize, u8, u8),
    /// A signed byte, in degrees.
    I8(usize),
    /// A signed 16-bit angle in steps of 0.006 degrees.
    Angle(usize),
    /// A double, less the job's time base.
    Time(usize),
    /// An extra-bytes value: its type (1 to 10) and its scale and offset.
    Extra(usize, u8, f64, f64),
}

/// A custom attribute and how it's read.
struct Field {
    name: String,
    get: Get,
}

/// What a point format holds beyond the position, and where.
struct Layout {
    fields: Vec<Field>,
    rgb: Option<usize>,
    time: Option<usize>,
}

fn layout(format: u8) -> Layout {
    let f = |name: &str, get| Field { name: name.into(), get };
    let mut fields = vec![f("INTENSITY", Get::U16(12))];
    let (time, rgb, nir) = if format < 6 {
        fields.extend([
            f("RETURN_NUMBER", Get::Bits(14, 0, 7)),
            f("NUMBER_OF_RETURNS", Get::Bits(14, 3, 7)),
            f("SCAN_DIRECTION", Get::Bits(14, 6, 1)),
            f("EDGE_OF_FLIGHT_LINE", Get::Bits(14, 7, 1)),
            f("CLASSIFICATION", Get::Bits(15, 0, 31)),
            f("CLASSIFICATION_FLAGS", Get::Bits(15, 5, 7)),
            f("SCAN_ANGLE", Get::I8(16)),
            f("USER_DATA", Get::U8(17)),
            f("POINT_SOURCE_ID", Get::U16(18)),
        ]);
        match format {
            1 | 4 => (Some(20), None, None),
            2 => (None, Some(20), None),
            3 | 5 => (Some(20), Some(28), None),
            _ => (None, None, None),
        }
    } else {
        fields.extend([
            f("RETURN_NUMBER", Get::Bits(14, 0, 15)),
            f("NUMBER_OF_RETURNS", Get::Bits(14, 4, 15)),
            f("CLASSIFICATION_FLAGS", Get::Bits(15, 0, 15)),
            f("SCANNER_CHANNEL", Get::Bits(15, 4, 3)),
            f("SCAN_DIRECTION", Get::Bits(15, 6, 1)),
            f("EDGE_OF_FLIGHT_LINE", Get::Bits(15, 7, 1)),
            f("CLASSIFICATION", Get::U8(16)),
            f("USER_DATA", Get::U8(17)),
            f("SCAN_ANGLE", Get::Angle(18)),
            f("POINT_SOURCE_ID", Get::U16(20)),
        ]);
        match format {
            7 => (Some(22), Some(30), None),
            8 | 10 => (Some(22), Some(30), Some(36)),
            _ => (Some(22), None, None),
        }
    };
    if let Some(at) = time {
        fields.push(f("GPS_TIME", Get::Time(at)));
    }
    if let Some(at) = nir {
        fields.push(f("NIR", Get::U16(at)));
    }
    Layout { fields, rgb, time }
}

/// The extra bytes the file describes (record 4 of `LASF_Spec`), as fields from `at` on within
/// records of `record` bytes, and their descriptions for the extras.
fn extra_bytes(data: &[u8], mut at: usize, record: usize, fields: &mut Vec<Field>) -> Vec<Value> {
    const SIZE: [usize; 11] = [0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8];
    let mut described = Vec::new();
    for d in data.chunks_exact(192) {
        let ty = d[2];
        let options = d[3];
        let name = text(&d[4..36]);
        let (base, count) = match ty {
            1..=10 => (ty, 1),
            11..=20 => (ty - 10, 2),
            21..=30 => (ty - 20, 3),
            _ => (0, usize::from(options).max(1)),
        };
        let size = if base == 0 {
            usize::from(options)
        } else {
            SIZE[usize::from(base)] * count
        };
        described.push(json!({ "name": name, "type": ty, "description": text(&d[160..192]) }));
        if at + size > record {
            break;
        }
        if base != 0 {
            for k in 0..count {
                let scale = if options & 8 != 0 { f64_at(d, 112 + 8 * k) } else { 1.0 };
                let offset = if options & 16 != 0 { f64_at(d, 136 + 8 * k) } else { 0.0 };
                let name = if count == 1 {
                    name.clone()
                } else {
                    format!("{name}_{k}")
                };
                let (scale, offset) = (
                    if scale.is_finite() && scale != 0.0 { scale } else { 1.0 },
                    if offset.is_finite() { offset } else { 0.0 },
                );
                let pos = at + k * SIZE[usize::from(base)];
                fields.push(Field {
                    name: points::attribute_name(&name),
                    get: Get::Extra(pos, base, scale, offset),
                });
            }
        }
        at += size;
    }
    described
}

/// A column while it's read: one value so far, or many.
enum Store {
    Same(u32, usize),
    Many(Vec<f32>),
}

impl Store {
    fn add(&mut self, block: &[f32], total: usize) -> Result<()> {
        match self {
            Store::Same(bits, n) if *n == 0 && !block.is_empty() => {
                *bits = block[0].to_bits();
                if block.iter().all(|v| v.to_bits() == *bits) {
                    *n = block.len();
                    return Ok(());
                }
            }
            Store::Same(bits, n) if block.iter().all(|v| v.to_bits() == *bits) => {
                *n += block.len();
                return Ok(());
            }
            Store::Same(..) => {}
            Store::Many(v) => {
                points::reserve(v, block.len(), "LAS")?;
                v.extend_from_slice(block);
                return Ok(());
            }
        }
        if let Store::Same(bits, n) = *self {
            let mut v = Vec::new();
            points::reserve(&mut v, total.max(n + block.len()), "LAS")?;
            v.resize(n, f32::from_bits(bits));
            v.extend_from_slice(block);
            *self = Store::Many(v);
        }
        Ok(())
    }
}

/// What a job turns its points into.
#[derive(Default)]
struct Scratch {
    pos: Vec<[f32; 3]>,
    rgb: Vec<[u16; 3]>,
    cols: Vec<Vec<f32>>,
    /// The time base the job found, when it was asked to find one.
    base: Option<f64>,
    raw: Vec<u8>,
    result: Option<Result<()>>,
}

/// What every job reads its points with.
struct Plan<'a> {
    record: usize,
    fields: &'a [Field],
    rgb: Option<usize>,
    time: Option<usize>,
    /// Integer coordinates to position in glTF's axes and meters, relative to the origin:
    /// `(i * scale + shift) * unit` per source axis.
    scale: [f64; 3],
    shift: [f64; 3],
    unit: [f64; 3],
}

impl Plan<'_> {
    /// Decodes the records in `raw` into `out`, its time base `base` (found from the first
    /// record when `None`).
    fn decode(&self, raw: &[u8], out: &mut Scratch, base: Option<f64>) {
        let n = raw.len() / self.record;
        let recs = || raw.chunks_exact(self.record);
        out.pos.clear();
        out.pos.extend(recs().map(|r| {
            let c = |k: usize| {
                let i = i32::from_le_bytes([r[4 * k], r[4 * k + 1], r[4 * k + 2], r[4 * k + 3]]);
                (f64::from(i) * self.scale[k] + self.shift[k]) * self.unit[k]
            };
            let [x, y, z] = [c(0), c(1), c(2)];
            [f32_of(x), f32_of(z), f32_of(-y)]
        }));
        out.rgb.clear();
        if let Some(at) = self.rgb {
            out.rgb
                .extend(recs().map(|r| [u16_at(r, at), u16_at(r, at + 2), u16_at(r, at + 4)]));
        }
        let base = base.unwrap_or_else(|| {
            let t = self.time.and_then(|at| raw.get(..self.record).map(|r| f64_at(r, at)));
            t.filter(|t| t.is_finite()).map_or(0.0, f64::floor)
        });
        out.base = Some(base);
        out.cols.resize_with(self.fields.len(), Vec::new);
        for (f, col) in self.fields.iter().zip(&mut out.cols) {
            col.clear();
            col.reserve(n);
            match f.get {
                Get::U8(at) => col.extend(recs().map(|r| f32::from(r[at]))),
                Get::U16(at) => col.extend(recs().map(|r| f32::from(u16_at(r, at)))),
                Get::Bits(at, shift, mask) => col.extend(recs().map(|r| f32::from((r[at] >> shift) & mask))),
                Get::I8(at) => col.extend(recs().map(|r| f32::from(r[at].cast_signed()))),
                Get::Angle(at) => col.extend(recs().map(|r| f32::from(u16_at(r, at).cast_signed()) * 0.006)),
                Get::Time(at) => col.extend(recs().map(|r| f32_of(f64_at(r, at) - base))),
                Get::Extra(at, ty, scale, offset) => col.extend(recs().map(|r| {
                    let v = match ty {
                        1 => f64::from(r[at]),
                        2 => f64::from(r[at].cast_signed()),
                        3 => f64::from(u16_at(r, at)),
                        4 => f64::from(u16_at(r, at).cast_signed()),
                        5 => f64::from(u32_at(r, at)),
                        6 => f64::from(u32_at(r, at).cast_signed()),
                        #[allow(clippy::cast_precision_loss, reason = "stored as single precision anyway")]
                        7 => u64_at(r, at) as f64,
                        #[allow(clippy::cast_precision_loss, reason = "stored as single precision anyway")]
                        8 => u64_at(r, at).cast_signed() as f64,
                        9 => f64::from(f32::from_bits(u32_at(r, at))),
                        _ => f64_at(r, at),
                    };
                    f32_of(v * scale + offset)
                })),
            }
        }
    }
}

/// A run of points: the index of the first and how many.
#[derive(Clone, Copy)]
struct Job {
    first: u64,
    count: usize,
}

/// Where a worker finds its records.
#[derive(Clone, Copy)]
enum Mode<'a> {
    /// In the file's bytes.
    Slice(&'a [u8]),
    /// In its scratch's raw bytes, read for it.
    Raw,
    /// From its LAZ decompressor.
    #[cfg_attr(not(feature = "laz"), allow(dead_code, reason = "only LAZ decompresses"))]
    Laz,
}

/// One thread's state.
struct Worker<'a> {
    scratch: Scratch,
    #[cfg(feature = "laz")]
    laz: Option<laz_read::Laz<'a>>,
    #[cfg(not(feature = "laz"))]
    #[allow(dead_code, reason = "holds the lifetime a LAZ decompressor has")]
    laz: Option<std::marker::PhantomData<&'a ()>>,
}

fn read(src: &Source<'_>, h: &Header, keep: Keep) -> Result<Document<'static>> {
    let len = src.len()?;
    let records = records(src, h, len)?;
    let mut lay = layout(h.format);
    let mut extras_described = Vec::new();
    if let Some(r) = records.iter().find(|r| r.user == "LASF_Spec" && r.id == 4) {
        extras_described = extra_bytes(&r.data, SIZES[usize::from(h.format)], h.record, &mut lay.fields);
    }
    let geo = Geo::from_records(&records, h.encoding);
    points::check_count(h.count, keep.points(), "LAS")?;
    let n = usize_of(h.count);
    #[cfg(feature = "laz")]
    let laz = match src {
        Source::Laz(bytes) => Some(laz_read::Setup::new(bytes, h, &records, n)?),
        _ => None,
    };
    #[cfg(not(feature = "laz"))]
    let laz: Option<()> = None;
    if laz.is_none() {
        // Uncompressed points must all be in the file.
        let room = len.saturating_sub(h.data_offset) / h.record as u64;
        ensure!(
            h.count <= room,
            "LAS data ends early: {} points declared, room for {room}",
            h.count
        );
    }
    // The origin, per source axis, near the header's bounds; feet become meters.
    let ok = (0..3).all(|k| h.min[k].is_finite() && h.max[k].is_finite() && h.min[k] <= h.max[k]);
    let origin_src = if ok { points::origin(h.min, h.max) } else { [0.0; 3] };
    let unit = geo.unit;
    let plan = Plan {
        record: h.record,
        fields: &lay.fields,
        rgb: lay.rgb,
        time: lay.time,
        scale: h.scale,
        shift: std::array::from_fn(|k| h.offset[k] - origin_src[k]),
        unit,
    };
    // The jobs: LAZ chunks where the file has a chunk table, blocks otherwise.
    #[cfg(feature = "laz")]
    let jobs = match &laz {
        Some(l) => l.jobs(n),
        None => blocks(n),
    };
    #[cfg(not(feature = "laz"))]
    let jobs = blocks(n);
    #[cfg(feature = "laz")]
    let threads = if laz.as_ref().is_some_and(|l| !l.seekable()) {
        1
    } else {
        keep.threads()
    };
    #[cfg(not(feature = "laz"))]
    let threads = keep.threads();
    let threads = threads.clamp(1, jobs.len().max(1));
    let mut workers = Vec::new();
    for _ in 0..threads {
        workers.push(Mutex::new(Worker {
            scratch: Scratch::default(),
            #[cfg(feature = "laz")]
            laz: match &laz {
                Some(l) => Some(l.open()?),
                None => None,
            },
            #[cfg(not(feature = "laz"))]
            laz: None,
        }));
    }
    // What the workers read from: the source itself only when it's a slice.
    let mode = match src {
        Source::Slice(b) => Mode::Slice(b),
        Source::File(_) => Mode::Raw,
        #[cfg(feature = "laz")]
        Source::Laz(_) => Mode::Laz,
    };
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut rgb: Vec<[u16; 3]> = Vec::new();
    let mut stores: Vec<Store> = lay.fields.iter().map(|_| Store::Same(0, 0)).collect();
    let mut base: Option<f64> = None;
    // The first job runs alone, so the time base it finds is everyone's.
    let mut done = 0;
    let mut round = 1;
    while done < jobs.len() {
        let these = &jobs[done..(done + round).min(jobs.len())];
        // A file is read on this thread, then decoded on the workers.
        if let Source::File(_) = src {
            for (w, job) in workers.iter().zip(these) {
                let raw = src.at(h.data_offset + job.first * h.record as u64, job.count * h.record)?;
                let mut w = w.lock().unwrap_or_else(PoisonError::into_inner);
                w.scratch.raw.clear();
                w.scratch.raw.extend_from_slice(&raw);
            }
        }
        let work = |w: &Mutex<Worker<'_>>, job: Job| {
            let mut guard = w.lock().unwrap_or_else(PoisonError::into_inner);
            let w = &mut *guard;
            let result = (|| -> Result<()> {
                let s = &mut w.scratch;
                match mode {
                    Mode::Slice(b) => {
                        let at = usize_of(h.data_offset + job.first * h.record as u64);
                        let raw = b.get(at..at + job.count * h.record).context("LAS data ends early")?;
                        plan.decode(raw, s, base);
                    }
                    Mode::Raw => {
                        let raw = std::mem::take(&mut s.raw);
                        plan.decode(&raw, s, base);
                        s.raw = raw;
                    }
                    #[cfg(feature = "laz")]
                    Mode::Laz => {
                        let mut raw = std::mem::take(&mut s.raw);
                        let read = w
                            .laz
                            .as_mut()
                            .context("no LAZ decompressor")?
                            .read(job, h.record, &mut raw);
                        if read.is_ok() {
                            plan.decode(&raw, s, base);
                        }
                        s.raw = raw;
                        read?;
                    }
                    #[cfg(not(feature = "laz"))]
                    Mode::Laz => bail!("LAZ needs the `laz` feature"),
                }
                Ok(())
            })();
            w.scratch.result = Some(result);
        };
        std::thread::scope(|scope| {
            let mut inline = Vec::new();
            for (k, (w, &job)) in workers.iter().zip(these).enumerate() {
                if k == 0 || !spawn(scope, move || work(w, job)) {
                    inline.push((w, job));
                }
            }
            for (w, job) in inline {
                work(w, job);
            }
        });
        for w in workers.iter().take(these.len()) {
            let mut w = w.lock().unwrap_or_else(PoisonError::into_inner);
            let s = &mut w.scratch;
            s.result.take().context("a LAS job didn't run")??;
            if base.is_none() {
                base = s.base;
            }
            points::reserve(&mut positions, s.pos.len(), "LAS")?;
            positions.extend_from_slice(&s.pos);
            if lay.rgb.is_some() {
                points::reserve(&mut rgb, s.rgb.len(), "LAS")?;
                rgb.extend_from_slice(&s.rgb);
            }
            for (store, col) in stores.iter_mut().zip(&s.cols) {
                store.add(col, n)?;
            }
        }
        done += these.len();
        round = threads;
    }
    drop(workers);
    // Colors: 8-bit values in the 16-bit fields when none is past 255.
    let colors = if rgb.is_empty() {
        Vec::new()
    } else {
        let eight = rgb.iter().flatten().all(|&c| c <= 255);
        let mut colors = Vec::new();
        points::reserve(&mut colors, rgb.len(), "LAS")?;
        colors.extend(rgb.iter().map(|&c| {
            if eight {
                points::linear16(c.map(|v| v * 257))
            } else {
                points::linear16(c)
            }
        }));
        colors
    };
    drop(rgb);
    let mut constant = Map::new();
    let mut columns = Vec::new();
    for (f, store) in lay.fields.iter().zip(stores) {
        match store {
            Store::Same(bits, k) if k > 0 => {
                constant.insert(format!("_{}", f.name), json!(f32::from_bits(bits)));
            }
            Store::Same(..) => {}
            Store::Many(v) => columns.push(Column {
                name: f.name.clone(),
                values: v,
            }),
        }
    }
    let origin = points::z_up(std::array::from_fn(|k| origin_src[k] * unit[k]));
    let cloud = Cloud {
        positions,
        colors,
        normals: Vec::new(),
        columns,
        origin,
        constant,
    };
    let mut source = json!({
        "pointFormat": h.format,
        "pointRecordLength": h.record,
        "fileSourceId": h.file_source,
        "globalEncoding": h.encoding,
        "systemIdentifier": h.system,
        "pointsByReturn": h.by_return,
        "bounds": { "min": h.min, "max": h.max },
    });
    if h.guid.iter().any(|&c| c != 0) {
        source["projectId"] = json!(guid(&h.guid));
    }
    if let Some(b) = base.filter(|_| lay.time.is_some()) {
        source["gpsTimeBase"] = json!(b);
    }
    if [4, 5, 9, 10].contains(&h.format) || h.waveform != 0 {
        source["waveformPackets"] = json!("not read");
    }
    if !extras_described.is_empty() {
        source["extraBytes"] = json!(extras_described);
    }
    let list: Vec<Value> = records
        .iter()
        .map(|r| json!({ "userId": r.user, "recordId": r.id, "description": r.description, "length": r.data.len() }))
        .collect();
    if !list.is_empty() {
        source["records"] = json!(list);
    }
    let mut g = json!({
        "scale": h.scale,
        "offset": h.offset.map(|c| c + 0.0),
        "origin": origin_src,
        "units": geo.unit_name,
        "toMeters": unit,
        "axes": "Z-up source turned Y-up: glTF x = east, y = up, z = south",
    });
    if let Some(e) = geo.epsg {
        g["epsg"] = json!(e);
    }
    if let Some(v) = geo.vertical_epsg {
        g["verticalEpsg"] = json!(v);
    }
    if let Some(w) = &geo.wkt {
        g["wkt"] = json!(w);
    }
    if let Some(k) = geo.kind {
        g["kind"] = json!(k);
    }
    let mut info = SourceInfo {
        format: Some((if h.compressed { "LAZ" } else { "LAS" }).into()),
        version: Some(format!("{}.{}", h.version.0, h.version.1)),
        meters_per_unit: geo
            .horizontal
            .as_ref()
            .map_or(Fact::Unknown, |&(_, m)| Fact::Declared(m)),
        unit: geo.horizontal.as_ref().map(|(name, _)| name.replace("metre", "meter")),
        // East, north and up, by the format.
        up_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool: Some(h.software.clone()).filter(|t| !t.is_empty()),
        created: created(h.year, h.day),
        georeference: Some(g),
        ..SourceInfo::default()
    };
    info.converted.units = unit.iter().any(|&u| (u - 1.0).abs() > f64::EPSILON);
    info.converted.up_axis = true;
    let Value::Object(source) = source else { unreachable!() };
    Ok(cloud.document(&info, source, Map::new()))
}

/// The jobs for `n` points read a block at a time.
fn blocks(n: usize) -> Vec<Job> {
    (0..n.div_ceil(BLOCK))
        .map(|k| Job {
            first: (k * BLOCK) as u64,
            count: BLOCK.min(n - k * BLOCK),
        })
        .collect()
}

fn guid(g: &[u8; 16]) -> String {
    let hex = |b: &[u8]| {
        b.iter().fold(String::new(), |mut s, c| {
            let _ = write!(s, "{c:02x}");
            s
        })
    };
    // The first three groups are little-endian numbers.
    let d1 = u32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    let d2 = u16::from_le_bytes([g[4], g[5]]);
    let d3 = u16::from_le_bytes([g[6], g[7]]);
    format!("{d1:08x}-{d2:04x}-{d3:04x}-{}-{}", hex(&g[8..10]), hex(&g[10..16]))
}

/// The header's creation date (`year`, and `day` of it from 1) in ISO 8601, as written when it
/// isn't a date, and `None` when it's left out (zero).
fn created(year: u16, day: u16) -> Option<String> {
    if year == 0 || day == 0 {
        return None;
    }
    Some(dates::day_of_year(year, day).unwrap_or_else(|| format!("{year}, day {day}")))
}

/// The coordinate system, as far as the file says.
struct Geo {
    epsg: Option<u32>,
    vertical_epsg: Option<u32>,
    wkt: Option<String>,
    kind: Option<&'static str>,
    /// Meters per unit along each source axis.
    unit: [f64; 3],
    unit_name: String,
    /// The horizontal unit's name and meters, when the coordinate system has a linear one.
    horizontal: Option<(String, f64)>,
}

impl Geo {
    fn from_records(records: &[Record], encoding: u16) -> Geo {
        let mut g = Geo {
            epsg: None,
            vertical_epsg: None,
            wkt: None,
            kind: None,
            unit: [1.0; 3],
            unit_name: "unknown, taken as meters".into(),
            horizontal: None,
        };
        let find = |user: &str, id: u16| records.iter().find(|r| r.user == user && r.id == id);
        // WKT, which LAS 1.4 asks for (global encoding bit 4), or the GeoTIFF keys.
        let wkt = find("LASF_Projection", 2112)
            .map(|r| text(&r.data))
            .filter(|w| !w.is_empty());
        let keys = find("LASF_Projection", 34735);
        if let Some(w) = wkt.filter(|_| encoding & 16 != 0 || keys.is_none()) {
            if let Some(crs) = super::wkt::Crs::parse(&w) {
                g.epsg = crs.epsg;
                g.kind = Some(crs.kind);
                if let Some((name, h)) = crs.horizontal {
                    let v = crs.vertical.map_or(h, |(_, v)| v);
                    g.unit = [h, h, v];
                    g.horizontal = Some((name.clone(), h));
                    g.unit_name = name;
                } else if crs.kind == "geographic" {
                    g.unit_name = "degrees".into();
                }
                g.vertical_epsg = crs.vertical_epsg;
            }
            g.wkt = Some(w);
            return g;
        }
        let Some(keys) = keys else {
            return g;
        };
        let words: Vec<u16> = keys
            .data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let count = usize::from(words.get(3).copied().unwrap_or(0));
        let mut horizontal = None;
        let mut vertical = None;
        for k in words.get(4..).unwrap_or(&[]).chunks_exact(4).take(count) {
            let (id, location, value) = (k[0], k[1], k[3]);
            if location != 0 {
                continue;
            }
            match id {
                1024 => {
                    g.kind = match value {
                        1 => Some("projected"),
                        2 => Some("geographic"),
                        3 => Some("geocentric"),
                        _ => None,
                    };
                }
                3072 if value != 0 && value != 32767 => g.epsg = Some(u32::from(value)),
                2048 if g.epsg.is_none() && value != 0 && value != 32767 => g.epsg = Some(u32::from(value)),
                4096 if value != 0 && value != 32767 => g.vertical_epsg = Some(u32::from(value)),
                3076 => horizontal = linear_unit(value),
                4099 => vertical = linear_unit(value),
                _ => {}
            }
        }
        if let Some((name, h)) = horizontal {
            let v = vertical.map_or(h, |(_, v)| v);
            g.unit = [h, h, v];
            g.horizontal = Some((name.into(), h));
            g.unit_name = name.into();
        } else if g.kind == Some("geographic") {
            g.unit_name = "degrees".into();
            if let Some((_, v)) = vertical {
                g.unit[2] = v;
            }
        } else if let Some((name, v)) = vertical {
            g.unit[2] = v;
            g.unit_name = format!("meters across, {name} up");
        }
        g
    }
}

/// A `GeoTIFF` linear unit code's name and size in meters.
fn linear_unit(code: u16) -> Option<(&'static str, f64)> {
    Some(match code {
        9001 => ("meter", 1.0),
        9002 => ("foot", FOOT),
        9003 => ("US survey foot", US_FOOT),
        9005 => ("Clarke's foot", 0.304_797_265_4),
        9014 => ("fathom", 1.8288),
        9030 => ("nautical mile", 1852.0),
        9036 => ("kilometer", 1000.0),
        9096 => ("yard", 0.9144),
        _ => return None,
    })
}

#[cfg(feature = "laz")]
mod laz_read {
    //! LAZ points through the `laz` crate: one decompressor per thread, each seeking to the
    //! chunks it's given.

    use std::io::{Cursor, Seek as _, SeekFrom};

    use laz::{LasZipDecompressor, LazVlr};

    use super::{Header, Job, Record, u32_at, u64_at};
    use crate::num::usize_of;
    use crate::{Context as _, Result};

    /// What every decompressor is made from.
    pub(super) struct Setup<'a> {
        bytes: &'a [u8],
        start: u64,
        vlr: LazVlr,
        /// The points in each chunk, when the file has a chunk table.
        chunks: Option<Vec<u64>>,
    }

    pub(super) struct Laz<'a> {
        d: LasZipDecompressor<'a, Cursor<&'a [u8]>>,
        /// The point the decompressor reads next.
        next: u64,
    }

    impl<'a> Setup<'a> {
        pub fn new(bytes: &'a [u8], h: &Header, records: &[Record], n: usize) -> Result<Self> {
            let r = records
                .iter()
                .find(|r| r.user == "laszip encoded" && r.id == 22204)
                .context("a LAZ file without its laszip record")?;
            let vlr = LazVlr::from_buffer(&r.data).context("the LAZ laszip record")?;
            ensure!(
                usize::try_from(vlr.items_size()).is_ok_and(|s| s == h.record),
                "the LAZ items ({} bytes) don't match the point records ({} bytes)",
                vlr.items_size(),
                h.record
            );
            let start = h.data_offset;
            let len = bytes.len() as u64;
            ensure!(start.checked_add(8).is_some_and(|e| e <= len), "LAZ data ends early");
            let chunks = if vlr.chunk_size() == 0 && !vlr.uses_variable_size_chunks() {
                None
            } else {
                chunk_table(bytes, start, &vlr, n)?
            };
            Ok(Setup {
                bytes,
                start,
                vlr,
                chunks,
            })
        }

        /// Whether decompressors can seek, so chunks can be spread over threads.
        pub fn seekable(&self) -> bool {
            self.chunks.is_some()
        }

        pub fn jobs(&self, n: usize) -> Vec<Job> {
            let mut out = Vec::new();
            let mut first = 0u64;
            let total = n as u64;
            if let Some(chunks) = &self.chunks {
                for &c in chunks {
                    if first >= total {
                        break;
                    }
                    let count = c.min(total - first);
                    if count > 0 {
                        out.push(Job {
                            first,
                            count: usize_of(count),
                        });
                    }
                    first += count;
                }
            }
            // What the table doesn't cover (or all of it, without one) is read in order.
            while first < total {
                let count = (super::BLOCK as u64).min(total - first);
                out.push(Job {
                    first,
                    count: usize_of(count),
                });
                first += count;
            }
            out
        }

        pub fn open(&self) -> Result<Laz<'a>> {
            let mut c = Cursor::new(self.bytes);
            c.seek(SeekFrom::Start(self.start))?;
            let d = LasZipDecompressor::new(c, self.vlr.clone()).context("the LAZ data")?;
            Ok(Laz { d, next: 0 })
        }
    }

    impl Laz<'_> {
        /// Decompresses the job's points into `raw`.
        pub fn read(&mut self, job: Job, record: usize, raw: &mut Vec<u8>) -> Result<()> {
            if job.first != self.next {
                self.d.seek(job.first).context("seeking in the LAZ data")?;
            }
            let size = job.count.checked_mul(record).context("too many LAZ points")?;
            raw.clear();
            raw.try_reserve_exact(size)
                .ok()
                .context("not enough memory for the LAZ points")?;
            raw.resize(size, 0);
            self.next = u64::MAX;
            self.d.decompress_many(raw).context("the LAZ data")?;
            self.next = job.first + job.count as u64;
            Ok(())
        }
    }

    /// The point count of each chunk, from the chunk table, or `None` when the file has none
    /// (a writer that couldn't go back to record it). The table's size is checked against the
    /// bytes before it's read.
    fn chunk_table(bytes: &[u8], start: u64, vlr: &LazVlr, n: usize) -> Result<Option<Vec<u64>>> {
        let len = bytes.len() as u64;
        let at = usize_of(start);
        let mut table = u64_at(bytes, at).cast_signed();
        if table <= start.cast_signed() {
            table = u64_at(bytes, bytes.len().saturating_sub(8)).cast_signed();
            if table <= start.cast_signed() {
                ensure!(
                    !vlr.uses_variable_size_chunks(),
                    "a LAZ file of variable-size chunks without its chunk table"
                );
                return Ok(None);
            }
        }
        let table = table.cast_unsigned();
        ensure!(
            table.checked_add(8).is_some_and(|e| e <= len),
            "the LAZ chunk table is past the end"
        );
        let chunks = u64::from(u32_at(bytes, usize_of(table) + 4));
        // Each chunk takes some bytes of point data, and no chunk is empty.
        ensure!(
            chunks <= (table - start) && chunks <= n as u64 + 1,
            "the LAZ chunk table counts {chunks} chunks, more than the data holds"
        );
        let mut c = Cursor::new(bytes);
        c.seek(SeekFrom::Start(start))?;
        let t = laz::laszip::ChunkTable::read_from(&mut c, vlr).context("the LAZ chunk table")?;
        let counts = t.as_ref().iter().map(|e| e.point_count).collect();
        Ok(Some(counts))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::document::Pose;

    /// A LAS file: version 1.`minor`, point format `format`, the points' integer coordinates
    /// and a byte pattern for the rest of each record, and the variable-length records given.
    pub(crate) fn las(minor: u8, format: u8, extra: usize, pts: &[[i32; 3]], vlrs: &[(&str, u16, Vec<u8>)]) -> Vec<u8> {
        let header = if minor >= 4 {
            375
        } else if minor == 3 {
            235
        } else {
            227
        };
        let record = SIZES[usize::from(format)] + extra;
        let vlr_bytes: usize = vlrs.iter().map(|v| VLR + v.2.len()).sum();
        let offset = header + vlr_bytes;
        let mut b = vec![0u8; header];
        b[..4].copy_from_slice(b"LASF");
        b[24] = 1;
        b[25] = minor;
        b[26..30].copy_from_slice(b"test");
        b[94..96].copy_from_slice(&u16::try_from(header).unwrap().to_le_bytes());
        b[96..100].copy_from_slice(&u32::try_from(offset).unwrap().to_le_bytes());
        b[100..104].copy_from_slice(&u32::try_from(vlrs.len()).unwrap().to_le_bytes());
        b[104] = format;
        b[105..107].copy_from_slice(&u16::try_from(record).unwrap().to_le_bytes());
        let n = pts.len();
        if format < 6 {
            b[107..111].copy_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
        }
        if minor >= 4 {
            b[247..255].copy_from_slice(&(n as u64).to_le_bytes());
        }
        let put = |b: &mut Vec<u8>, at: usize, v: f64| b[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for k in 0..3 {
            put(&mut b, 131 + 8 * k, 0.01);
            put(&mut b, 155 + 8 * k, [500_000.0, 4_000_000.0, 0.0][k]);
        }
        let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in pts {
            for k in 0..3 {
                let v = f64::from(p[k]) * 0.01 + [500_000.0, 4_000_000.0, 0.0][k];
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
            }
        }
        for k in 0..3 {
            put(&mut b, 179 + 16 * k, max[k]);
            put(&mut b, 187 + 16 * k, min[k]);
        }
        for (user, id, data) in vlrs {
            let mut h = vec![0u8; VLR];
            h[2..2 + user.len()].copy_from_slice(user.as_bytes());
            h[18..20].copy_from_slice(&id.to_le_bytes());
            h[20..22].copy_from_slice(&u16::try_from(data.len()).unwrap().to_le_bytes());
            b.extend_from_slice(&h);
            b.extend_from_slice(data);
        }
        for (i, p) in pts.iter().enumerate() {
            let mut r = vec![0u8; record];
            for k in 0..3 {
                r[4 * k..4 * k + 4].copy_from_slice(&p[k].to_le_bytes());
            }
            for (j, c) in r.iter_mut().enumerate().skip(12) {
                *c = u8::try_from((i * 7 + j * 13) % 251).unwrap();
            }
            b.extend_from_slice(&r);
        }
        b
    }

    fn load_las(bytes: &[u8], keep: Keep) -> Result<Document<'static>> {
        super::load(Input::memory(bytes, "t.las"), keep)
    }

    fn pts(n: usize) -> Vec<[i32; 3]> {
        (0..n)
            .map(|i| {
                let i = i32::try_from(i).unwrap();
                [i * 100, -i * 50, 1000 + i]
            })
            .collect()
    }

    #[test]
    fn every_point_format_reads() {
        for format in 0..=10u8 {
            for minor in [2u8, 4] {
                if format > 5 && minor < 4 {
                    continue;
                }
                let bytes = las(minor, format, 0, &pts(5), &[]);
                let doc = load_las(&bytes, Keep::ALL).unwrap_or_else(|e| panic!("format {format}: {e}"));
                assert_eq!(doc.facts["points"], 5, "format {format}");
                let attrs = &doc.json["meshes"][0]["primitives"][0]["attributes"];
                assert!(attrs.get("_INTENSITY").is_some(), "format {format}: {attrs}");
                assert_eq!(
                    attrs.get("COLOR_0").is_some(),
                    [2, 3, 5, 7, 8, 10].contains(&format),
                    "{format}"
                );
                assert_eq!(attrs.get("_NIR").is_some(), [8, 10].contains(&format), "{format}");
                let m = doc.flatten(Pose::Rest).unwrap().unwrap();
                let p = &m.prims[0].positions;
                // The second point: 1 m east, 0.5 m south, 0.01 m higher than the first.
                let d: Vec<f32> = (0..3).map(|k| p[1][k] - p[0][k]).collect();
                assert!(
                    (d[0] - 1.0).abs() < 1e-3 && (d[1] - 0.01).abs() < 1e-3 && (d[2] - 0.5).abs() < 1e-3,
                    "{d:?}"
                );
            }
        }
    }

    #[test]
    fn far_points_keep_their_detail_through_the_translation() {
        let bytes = las(2, 0, 0, &pts(3), &[]);
        let doc = load_las(&bytes, Keep::ALL).unwrap();
        // No coordinate system: units unknown, and left as they are.
        let info = doc.source_info();
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert!(!info.converted.units && info.converted.up_axis);
        assert_eq!((info.tool, info.created), (None, None));
        let t = doc.json["nodes"][0]["translation"].as_array().unwrap();
        assert_eq!(t[0], 500_001.0);
        assert_eq!(t[2], -4_000_000.0);
        assert_eq!(doc.source_info().georeference.unwrap()["scale"][0], 0.01);
    }

    #[test]
    fn feet_become_meters() {
        // GeoTIFF keys: projected, EPSG 2263, US survey feet.
        let keys: Vec<u8> = [1u16, 1, 0, 3, 1024, 0, 1, 1, 3072, 0, 1, 2263, 3076, 0, 1, 9003]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut bytes = las(2, 0, 0, &pts(2), &[("LASF_Projection", 34735, keys)]);
        bytes[58..66].copy_from_slice(b"Scanner ");
        bytes[90..92].copy_from_slice(&60u16.to_le_bytes());
        bytes[92..94].copy_from_slice(&2024u16.to_le_bytes());
        let doc = load_las(&bytes, Keep::ALL).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("LAS"));
        assert_eq!(info.version.as_deref(), Some("1.2"));
        assert_eq!(info.meters_per_unit, Fact::Declared(US_FOOT));
        assert_eq!(info.unit.as_deref(), Some("US survey foot"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        assert!(info.converted.units && info.converted.up_axis);
        assert_eq!(info.tool.as_deref(), Some("Scanner"));
        assert_eq!(info.created.as_deref(), Some("2024-02-29"));
        assert!(doc.facts.get("georeference").is_none());
        let g = &info.georeference.unwrap();
        assert_eq!(g["epsg"], 2263);
        assert_eq!(g["kind"], "projected");
        assert!((g["toMeters"][0].as_f64().unwrap() - US_FOOT).abs() < 1e-12);
        let t = doc.json["nodes"][0]["translation"][0].as_f64().unwrap();
        assert!((t - 500_000.0 * US_FOOT).abs() < 1.0, "{t}");
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        let p = &m.prims[0].positions;
        // Single precision this far out is good to a centimeter or two.
        assert!((f64::from(p[1][0] - p[0][0]) - US_FOOT).abs() < 0.02);
    }

    #[test]
    fn wkt_names_the_units() {
        let wkt = br#"PROJCS["NAD83 / NY",GEOGCS["NAD83",UNIT["degree",0.0174532925199433]],UNIT["US survey foot",0.304800609601219,AUTHORITY["EPSG","9003"]],AUTHORITY["EPSG","2263"]]"#;
        let mut bytes = las(4, 6, 0, &pts(2), &[("LASF_Projection", 2112, wkt.to_vec())]);
        bytes[6] = 16;
        let doc = load_las(&bytes, Keep::ALL).unwrap();
        let g = &doc.source_info().georeference.unwrap();
        assert_eq!(g["epsg"], 2263);
        assert!((g["toMeters"][0].as_f64().unwrap() - US_FOOT).abs() < 1e-9);
    }

    #[test]
    fn extra_bytes_are_attributes() {
        let mut d = vec![0u8; 192];
        d[2] = 4; // i16
        d[3] = 8; // scaled
        d[4..9].copy_from_slice(b"Width");
        d[112..120].copy_from_slice(&0.5f64.to_le_bytes());
        let bytes = las(4, 6, 2, &pts(4), &[("LASF_Spec", 4, d)]);
        let doc = load_las(&bytes, Keep::ALL).unwrap();
        let attrs = &doc.json["meshes"][0]["primitives"][0]["attributes"];
        assert!(attrs.get("_WIDTH").is_some(), "{attrs}");
    }

    #[test]
    fn too_many_points_are_refused_before_reading() {
        let bytes = las(2, 0, 0, &pts(10), &[]);
        let e = load_las(
            &bytes,
            Keep {
                max_points: 5,
                ..Keep::ALL
            },
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("max_points"), "{e}");
    }

    #[test]
    fn threads_read_the_same_cloud() {
        let bytes = las(2, 3, 0, &pts(BLOCK * 3 + 17), &[]);
        let one = load_las(&bytes, Keep::ALL).unwrap().to_glb().unwrap();
        let many = load_las(
            &bytes,
            Keep {
                threads: 4,
                ..Keep::ALL
            },
        )
        .unwrap()
        .to_glb()
        .unwrap();
        assert_eq!(one, many);
        crate::keep::NO_THREADS.set(true);
        let none = load_las(
            &bytes,
            Keep {
                threads: 4,
                ..Keep::ALL
            },
        )
        .unwrap()
        .to_glb()
        .unwrap();
        crate::keep::NO_THREADS.set(false);
        assert_eq!(one, none);
    }

    #[test]
    fn broken_files_fail_cleanly() {
        let bytes = las(
            4,
            8,
            3,
            &pts(40),
            &[("LASF_Projection", 2112, b"PROJCS[\"x\"".to_vec())],
        );
        for cut in (0..bytes.len()).step_by(7) {
            let _ = load_las(&bytes[..cut], Keep::ALL);
        }
        let mut rng = crate::test_rng::Pcg32::new(5);
        for _ in 0..500 {
            let mut b = bytes.clone();
            points::corrupt(&mut b, &mut rng, 4);
            let _ = load_las(&b, Keep::ALL);
        }
    }

    #[cfg(feature = "laz")]
    pub(crate) fn laz(las: &[u8], chunk: u32) -> Vec<u8> {
        use std::io::Cursor;

        use laz::{LasZipCompressor, LazItemRecordBuilder, LazVlrBuilder};
        let h = Header::parse(las).unwrap();
        let extra = h.record - SIZES[usize::from(h.format)];
        let items = LazItemRecordBuilder::default_for_point_format_id(h.format, u16::try_from(extra).unwrap()).unwrap();
        let vlr = LazVlrBuilder::new(items).with_fixed_chunk_size(chunk).build();
        let mut vlr_data = Vec::new();
        vlr.write_to(&mut vlr_data).unwrap();
        let mut out = las[..usize::from(h.size)].to_vec();
        // The existing records, then the laszip one.
        let data = usize_of(h.data_offset);
        out.extend_from_slice(&las[usize::from(h.size)..data]);
        let mut head = vec![0u8; VLR];
        head[2..16].copy_from_slice(b"laszip encoded");
        head[18..20].copy_from_slice(&22204u16.to_le_bytes());
        head[20..22].copy_from_slice(&u16::try_from(vlr_data.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&head);
        out.extend_from_slice(&vlr_data);
        let offset = u32::try_from(out.len()).unwrap();
        out[96..100].copy_from_slice(&offset.to_le_bytes());
        out[100..104].copy_from_slice(&(h.vlrs + 1).to_le_bytes());
        out[104] |= 0x80;
        // The compressor records where its chunk table is from the start of the file.
        let mut c = Cursor::new(out);
        c.set_position(c.get_ref().len() as u64);
        {
            let mut z = LasZipCompressor::new(&mut c, vlr).unwrap();
            z.compress_many(&las[data..]).unwrap();
            z.done().unwrap();
        }
        c.into_inner()
    }

    #[cfg(feature = "laz")]
    #[test]
    fn laz_reads_as_its_las_does() {
        for format in [0u8, 1, 3, 6, 7, 8] {
            let minor = if format > 5 { 4 } else { 2 };
            let plain = las(minor, format, 0, &pts(1000), &[]);
            let packed = laz(&plain, 128);
            assert!(packed.len() < plain.len());
            let a = load_las(&plain, Keep::ALL).unwrap();
            let b = super::load(
                Input::memory(&packed, "t.laz"),
                Keep {
                    threads: 3,
                    ..Keep::ALL
                },
            )
            .unwrap();
            assert_eq!(b.json["asset"]["extras"]["source"]["format"], "LAZ");
            let fa = a.flatten(Pose::Rest).unwrap().unwrap();
            let fb = b.flatten(Pose::Rest).unwrap().unwrap();
            assert_eq!(fa.prims[0].positions, fb.prims[0].positions, "format {format}");
            assert_eq!(a.json["meshes"], b.json["meshes"], "format {format}");
        }
    }

    #[cfg(feature = "laz")]
    #[test]
    fn broken_laz_fails_cleanly() {
        let packed = laz(&las(4, 7, 0, &pts(300), &[]), 50);
        for cut in (0..packed.len()).step_by(5) {
            let _ = super::load(Input::memory(&packed[..cut], "t.laz"), Keep::ALL);
        }
        let mut rng = crate::test_rng::Pcg32::new(9);
        for _ in 0..250 {
            let mut b = packed.clone();
            points::corrupt(&mut b, &mut rng, 3);
            let _ = super::load(
                Input::memory(&b, "t.laz"),
                Keep {
                    threads: 2,
                    ..Keep::ALL
                },
            );
        }
    }
}
