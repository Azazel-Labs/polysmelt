//! PCD point clouds, versions 0.5 to 0.7: `ascii`, `binary` and `binary_compressed` data.
//!
//! `x`, `y` and `z` are the position, `normal_x`, `normal_y` and `normal_z` the normal, and a
//! packed `rgb` or `rgba` (or separate `r`, `g`, `b`, `a` fields) the color. Every other field
//! becomes a float custom attribute: `intensity` as `_INTENSITY`, `label` as `_LABEL`,
//! `curvature` as `_CURVATURE` and so on, a field with several values per point as one
//! attribute per value (`_NAME_0`, `_NAME_1`, ...), up to [`COLUMNS`] of them. `_` fields are
//! padding. Points whose position isn't a number (the holes in an organized cloud) are left out.
//!
//! The format says nothing of units or axes. The source info leaves the units unknown, and Z up is
//! assumed, as it is for the scanners' text formats (`.xyz`, `.pts`, `.ptx`, `.asc`) and as robotics and
//! surveying tools write their clouds: positions and normals are turned to glTF's Y up in the data, `(x, y, z)`
//! to `(x, z, -y)`, so the same scan lands the same way whichever of these formats it is saved in. `VIEWPOINT`
//! is the sensor's pose in the cloud's own frame, not a statement of which way is up: it is kept as the file
//! gives it, with the cloud's width and height and the fields, in the
//! asset's extras. Double-precision positions far from the origin are measured from a point near
//! the first one, which the node's translation carries back.

use serde_json::{Map, Value, json};

use super::Input;
use super::points::{self, Cloud, Column};
use super::stream::{f32_from, f64_from, space};
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// The most custom attributes made from the fields that aren't position, normal or color.
const COLUMNS: usize = 64;
/// The most bytes the header may take.
const HEADER: usize = 1 << 20;
/// The most fields.
const FIELDS: usize = 4096;
/// Bytes of text parsed per job.
const PIECE: usize = if cfg!(test) { 64 << 10 } else { 4 << 20 };

pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document> {
    let bytes = input.read().context(ErrorKind::Corrupt, "reading the PCD file")?;
    read(&bytes, keep)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    I,
    U,
    F,
}

struct Field {
    name: String,
    size: usize,
    kind: Kind,
    count: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Data {
    Ascii,
    Binary,
    Compressed,
}

struct Header {
    version: String,
    fields: Vec<Field>,
    width: u64,
    height: u64,
    viewpoint: [f64; 7],
    points: u64,
    data: Data,
    /// Where the data starts.
    start: usize,
}

impl Header {
    fn parse(b: &[u8]) -> Result<Header> {
        let mut at = 0;
        let mut version = String::new();
        let mut names: Vec<String> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut types: Vec<Kind> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let (mut width, mut height, mut points) = (None, None, None);
        let mut viewpoint = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        loop {
            if at >= b.len().min(HEADER) {
                // A header that started and never reached its DATA line: the file ends inside it.
                ensure!(
                    at < b.len() || (version.is_empty() && names.is_empty()),
                    Truncated,
                    "PCD header is cut short: it has no DATA line"
                );
                bail!(UnknownFormat, "not a PCD file: no DATA line");
            }
            let end = b[at..].iter().position(|&c| c == b'\n').map_or(b.len(), |n| at + n);
            let line = crate::bytes::text(&b[at..end]);
            at = (end + 1).min(b.len());
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut words = line.split_ascii_whitespace();
            let key = words.next().unwrap_or_default().to_ascii_uppercase();
            let rest: Vec<&str> = words.take(FIELDS + 1).collect();
            ensure!(rest.len() <= FIELDS, LimitExceeded, "PCD header line {key} is too long");
            let num = |s: &&str| s.parse::<u64>().ok();
            match key.as_str() {
                "VERSION" => version = rest.first().map(|s| (*s).to_string()).unwrap_or_default(),
                "FIELDS" | "COLUMNS" => names = rest.iter().map(|s| (*s).to_string()).collect(),
                "SIZE" => sizes = rest.iter().map(|s| s.parse().unwrap_or(0)).collect(),
                "TYPE" => {
                    types = rest
                        .iter()
                        .map(|s| match s.to_ascii_uppercase().as_str() {
                            "I" => Ok(Kind::I),
                            "U" => Ok(Kind::U),
                            "F" => Ok(Kind::F),
                            t => Err(err!(Corrupt, "PCD field type {t} is unknown")),
                        })
                        .collect::<Result<_>>()?;
                }
                "COUNT" => counts = rest.iter().map(|s| s.parse().unwrap_or(0)).collect(),
                "WIDTH" => width = rest.first().and_then(num),
                "HEIGHT" => height = rest.first().and_then(num),
                "POINTS" => points = rest.first().and_then(num),
                "VIEWPOINT" => {
                    for (v, s) in viewpoint.iter_mut().zip(&rest) {
                        *v = s.parse::<f64>().ok().filter(|v| v.is_finite()).unwrap_or(*v);
                    }
                }
                "DATA" => {
                    let data = match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
                        Some("ascii") => Data::Ascii,
                        Some("binary") => Data::Binary,
                        Some("binary_compressed") => Data::Compressed,
                        d => bail!(Corrupt, "PCD data kind {} is unknown", d.unwrap_or("(none)")),
                    };
                    ensure!(!names.is_empty(), Corrupt, "the PCD file names no fields");
                    let n = names.len();
                    // Old files leave out what defaults: four-byte floats, one value each.
                    let sizes = if sizes.is_empty() { vec![4; n] } else { sizes };
                    let types = if types.is_empty() { vec![Kind::F; n] } else { types };
                    let counts = if counts.is_empty() { vec![1; n] } else { counts };
                    ensure!(
                        sizes.len() == n && types.len() == n && counts.len() == n,
                        Corrupt,
                        "the PCD header's FIELDS, SIZE, TYPE and COUNT lines differ in length"
                    );
                    let mut fields = Vec::with_capacity(n);
                    for (((name, size), kind), count) in names.into_iter().zip(sizes).zip(types).zip(counts) {
                        let fits = match kind {
                            Kind::F => matches!(size, 4 | 8),
                            Kind::I | Kind::U => matches!(size, 1 | 2 | 4 | 8),
                        };
                        ensure!(fits, Corrupt, "PCD field {name} has type {kind:?} of size {size}");
                        ensure!(
                            (1..=1 << 20).contains(&count),
                            LimitExceeded,
                            "PCD field {name} has {count} values per point"
                        );
                        fields.push(Field {
                            name,
                            size,
                            kind,
                            count,
                        });
                    }
                    let width = width.unwrap_or(0);
                    let height = height.unwrap_or(1).max(1);
                    let points = points.or_else(|| width.checked_mul(height)).unwrap_or(0);
                    return Ok(Header {
                        version,
                        fields,
                        width,
                        height,
                        viewpoint,
                        points,
                        data,
                        start: at,
                    });
                }
                _ => {}
            }
        }
    }

    /// The bytes of one point.
    fn record(&self) -> Option<usize> {
        self.fields
            .iter()
            .try_fold(0usize, |n, f| n.checked_add(f.size.checked_mul(f.count)?))
    }
}

/// What a value of a field is for.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    /// The position's axis.
    Pos(usize),
    /// A packed color, with alpha in its top byte or not.
    Packed(bool),
    /// One of [`Layout::values`].
    Value(usize),
    Skip,
}

/// What a float value is for.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Use {
    Normal(usize),
    /// Red, green, blue or alpha.
    Channel(usize),
    Column,
}

/// One value of a field, as stored.
#[derive(Clone, Copy)]
struct Elem {
    role: Role,
    kind: Kind,
    size: usize,
    /// The first point's value, in the data.
    start: usize,
    /// Bytes from one point's value to the next.
    stride: usize,
}

struct Value1 {
    name: String,
    usage: Use,
    kind: Kind,
    size: usize,
}

struct Layout {
    /// Every value of every field, in order.
    elems: Vec<Elem>,
    values: Vec<Value1>,
    /// Whether all three axes are there, and in double precision.
    pos: bool,
    double: bool,
    packed: bool,
    /// Values past [`COLUMNS`], left out.
    dropped: Vec<String>,
}

impl Layout {
    /// The values of `h`'s fields, laid out for `data` (point after point, or, compressed, field
    /// after field) over `points` points.
    fn new(h: &Header, points: usize) -> Layout {
        let mut elems = Vec::new();
        let mut values: Vec<Value1> = Vec::new();
        let mut dropped = Vec::new();
        let mut columns = 0;
        let mut axes = [false; 3];
        let mut double = false;
        let mut packed = false;
        let record = h.record().unwrap_or(0);
        let (mut offset, mut base) = (0usize, 0usize);
        for f in &h.fields {
            let lower = f.name.to_ascii_lowercase();
            for k in 0..f.count {
                let (start, stride) = if h.data == Data::Compressed {
                    (base + k * f.size, f.size * f.count)
                } else {
                    (offset + k * f.size, record)
                };
                let single = f.count == 1;
                let mut usage = None;
                let role = match lower.as_str() {
                    "_" => Role::Skip,
                    "x" | "y" | "z" if single => {
                        let a = usize::from(lower.as_bytes()[0] - b'x');
                        if axes[a] {
                            Role::Skip
                        } else {
                            axes[a] = true;
                            double |= f.kind == Kind::F && f.size == 8;
                            Role::Pos(a)
                        }
                    }
                    "rgb" | "rgba" if single && f.size == 4 && !packed => {
                        packed = true;
                        Role::Packed(lower == "rgba")
                    }
                    "normal_x" | "normal_y" | "normal_z" if single => {
                        usage = Some(Use::Normal(usize::from(lower.as_bytes()[7] - b'x')));
                        Role::Skip
                    }
                    "r" | "red" if single => {
                        usage = Some(Use::Channel(0));
                        Role::Skip
                    }
                    "g" | "green" if single => {
                        usage = Some(Use::Channel(1));
                        Role::Skip
                    }
                    "b" | "blue" if single => {
                        usage = Some(Use::Channel(2));
                        Role::Skip
                    }
                    "a" | "alpha" if single => {
                        usage = Some(Use::Channel(3));
                        Role::Skip
                    }
                    _ => {
                        usage = Some(Use::Column);
                        Role::Skip
                    }
                };
                let name = if single {
                    f.name.clone()
                } else {
                    format!("{}_{k}", f.name)
                };
                let role = match usage {
                    // A second field doing the same job is kept as a plain value.
                    Some(u) if u != Use::Column && values.iter().any(|v| v.usage == u) => {
                        usage = Some(Use::Column);
                        Role::Skip
                    }
                    _ => role,
                };
                let role = match usage {
                    Some(Use::Column) if columns >= COLUMNS => {
                        dropped.push(name.clone());
                        Role::Skip
                    }
                    Some(u) => {
                        columns += usize::from(u == Use::Column);
                        values.push(Value1 {
                            name,
                            usage: u,
                            kind: f.kind,
                            size: f.size,
                        });
                        Role::Value(values.len() - 1)
                    }
                    None => role,
                };
                elems.push(Elem {
                    role,
                    kind: f.kind,
                    size: f.size,
                    start,
                    stride,
                });
            }
            offset += f.size * f.count;
            base = base.saturating_add(points.saturating_mul(f.size * f.count));
        }
        Layout {
            elems,
            values,
            pos: axes == [true; 3],
            double,
            packed,
            dropped,
        }
    }
}

/// Points decoded: positions relative to the origin, packed colors, and the other values.
#[derive(Default)]
struct Decoded {
    pos: Vec<[f32; 3]>,
    packed: Vec<u32>,
    values: Vec<Vec<f32>>,
}

impl Decoded {
    fn new(lay: &Layout, n: usize) -> Result<Decoded> {
        let mut d = Decoded::default();
        points::reserve(&mut d.pos, n, "PCD")?;
        if lay.packed {
            points::reserve(&mut d.packed, n, "PCD")?;
        }
        for _ in &lay.values {
            let mut v = Vec::new();
            points::reserve(&mut v, n, "PCD")?;
            d.values.push(v);
        }
        Ok(d)
    }

    fn append(&mut self, mut other: Decoded) {
        self.pos.append(&mut other.pos);
        self.packed.append(&mut other.packed);
        for (a, b) in self.values.iter_mut().zip(&mut other.values) {
            a.append(b);
        }
    }

    fn truncate(&mut self, n: usize) {
        self.pos.truncate(n);
        self.packed.truncate(n);
        for v in &mut self.values {
            v.truncate(n);
        }
    }
}

/// The stored value in `b` (little-endian), as a double.
#[inline]
#[allow(clippy::cast_precision_loss, reason = "64-bit integers go to the nearest double")]
fn number(b: &[u8], kind: Kind) -> f64 {
    let mut le = [0u8; 8];
    le[..b.len()].copy_from_slice(b);
    let u = u64::from_le_bytes(le);
    match (kind, b.len()) {
        (Kind::F, 4) => f64::from(f32::from_bits(bits(b))),
        (Kind::F, _) => f64::from_bits(u),
        (Kind::U, _) => u as f64,
        (Kind::I, n) => {
            // Sign-extend from n bytes.
            let shift = 64 - 8 * n.min(8);
            ((u << shift).cast_signed() >> shift) as f64
        }
    }
}

/// The 32 bits of a packed color.
#[inline]
fn bits(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read(bytes: &[u8], keep: Keep) -> Result<Document> {
    let h = Header::parse(bytes)?;
    points::check_count(h.points, keep.points(), "PCD")?;
    let n = usize::try_from(h.points).unwrap_or(usize::MAX);
    let lay = Layout::new(&h, n);
    ensure!(lay.pos, Corrupt, "the PCD file has no x, y and z fields");
    let body = &bytes[h.start.min(bytes.len())..];
    let record = h
        .record()
        .filter(|&r| r > 0)
        .context(ErrorKind::LimitExceeded, "PCD records are too big")?;
    let (mut d, origin) = match h.data {
        Data::Ascii => ascii(body, &h, &lay, n, keep.thread_count())?,
        Data::Binary => {
            let need = n
                .checked_mul(record)
                .filter(|&b| b <= body.len())
                .context(ErrorKind::Truncated, "PCD data is cut short")?;
            binary(&body[..need], &lay, n)?
        }
        Data::Compressed => {
            ensure!(body.len() >= 8, Truncated, "PCD data is cut short");
            let packed = usize_of(u32::from_le_bytes([body[0], body[1], body[2], body[3]]));
            let size = usize_of(u32::from_le_bytes([body[4], body[5], body[6], body[7]]));
            let need = n
                .checked_mul(record)
                .context(ErrorKind::LimitExceeded, "PCD records are too big")?;
            ensure!(
                size == need,
                Corrupt,
                "PCD compressed data holds {size} bytes, not the {need} its points need"
            );
            let data = body
                .get(8..8 + packed)
                .context(ErrorKind::Truncated, "PCD data is cut short")?;
            // LZF expands a byte to at most about 130 of them; anything more is a broken file.
            ensure!(
                size <= packed.saturating_mul(140).saturating_add(64),
                Corrupt,
                "PCD compressed data is broken"
            );
            let raw = lzf(data, size).context(ErrorKind::Corrupt, "PCD compressed data is broken")?;
            binary(&raw, &lay, n)?
        }
    };
    d.truncate(n);
    Ok(cloud(d, &h, &lay, origin))
}

fn usize_of(v: u32) -> usize {
    crate::num::usize_of(u64::from(v))
}

/// The first finite point's origin, for double-precision positions far from zero.
fn origin_near(p: [f64; 3]) -> [f64; 3] {
    if p.iter().all(|c| c.is_finite()) {
        points::origin(p, p)
    } else {
        [0.0; 3]
    }
}

/// Points stored one after another (or, from compressed data, field after field) in `data`.
fn binary(data: &[u8], lay: &Layout, n: usize) -> Result<(Decoded, [f64; 3])> {
    let mut d = Decoded::new(lay, n)?;
    let at = |e: &Elem, i: usize| &data[e.start + i * e.stride..e.start + i * e.stride + e.size];
    let axis: [Elem; 3] = std::array::from_fn(|a| {
        *lay.elems
            .iter()
            .find(|e| e.role == Role::Pos(a))
            .unwrap_or(&lay.elems[0])
    });
    // Every element's values must lie in the data; the caller sized it so.
    for e in &lay.elems {
        if e.role != Role::Skip && n > 0 {
            ensure!(
                e.start + (n - 1) * e.stride + e.size <= data.len(),
                Truncated,
                "PCD data is cut short"
            );
        }
    }
    let mut origin = [0.0; 3];
    if lay.double
        && let Some(i) = (0..n).find(|&i| axis.iter().all(|e| number(at(e, i), e.kind).is_finite()))
    {
        origin = origin_near(std::array::from_fn(|a| number(at(&axis[a], i), axis[a].kind)));
    }
    let float = axis.iter().all(|e| e.kind == Kind::F && e.size == 4);
    for i in 0..n {
        d.pos.push(if float {
            std::array::from_fn(|a| f32::from_bits(bits(at(&axis[a], i))))
        } else {
            std::array::from_fn(|a| f32_of(number(at(&axis[a], i), axis[a].kind) - origin[a]))
        });
    }
    for e in &lay.elems {
        match e.role {
            Role::Packed(_) => d.packed.extend((0..n).map(|i| bits(at(e, i)))),
            Role::Value(v) => {
                let out = &mut d.values[v];
                if e.kind == Kind::F && e.size == 4 {
                    out.extend((0..n).map(|i| f32::from_bits(bits(at(e, i)))));
                } else {
                    out.extend((0..n).map(|i| f32_of(number(at(e, i), e.kind))));
                }
            }
            Role::Pos(_) | Role::Skip => {}
        }
    }
    Ok((d, origin))
}

/// LZF-compressed `input`, which should expand to `size` bytes.
fn lzf(input: &[u8], size: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(size).ok()?;
    let mut i = 0;
    while i < input.len() {
        let ctrl = usize::from(input[i]);
        i += 1;
        if ctrl < 32 {
            let n = ctrl + 1;
            let lit = input.get(i..i + n)?;
            if out.len() + n > size {
                return None;
            }
            out.extend_from_slice(lit);
            i += n;
        } else {
            let mut n = ctrl >> 5;
            if n == 7 {
                n += usize::from(*input.get(i)?);
                i += 1;
            }
            n += 2;
            let back = ((ctrl & 0x1f) << 8) + usize::from(*input.get(i)?) + 1;
            i += 1;
            if back > out.len() || out.len() + n > size {
                return None;
            }
            let from = out.len() - back;
            if back >= n {
                out.extend_from_within(from..from + n);
            } else {
                for k in 0..n {
                    let b = out[from + k];
                    out.push(b);
                }
            }
        }
    }
    (out.len() == size).then_some(out)
}

/// The tokens of a line.
fn tokens(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(|&b| space(b)).filter(|t| !t.is_empty())
}

/// A packed color written as text: an integer, or (older writers) the float with its bits.
fn packed_text(t: &[u8]) -> u32 {
    if t.iter().all(u8::is_ascii_digit) {
        // Wider than 32 bits keeps the low ones.
        let mut v = 0u64;
        for &c in t.iter().take(20) {
            v = v.wrapping_mul(10).wrapping_add(u64::from(c - b'0'));
        }
        #[allow(clippy::cast_possible_truncation, reason = "the low 32 bits are the color")]
        let v = v as u32;
        v
    } else if let Some(v) = t.strip_prefix(b"-").filter(|r| r.iter().all(u8::is_ascii_digit)) {
        // A signed integer: its two's complement bits.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the low 32 bits are the color"
        )]
        let v = std::str::from_utf8(v)
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .map_or(0, |v| (-v) as u32);
        v
    } else {
        f32_from(t).map_or(0, f32::to_bits)
    }
}

/// Points as text, a line each, read on up to `threads` threads.
fn ascii(body: &[u8], h: &Header, lay: &Layout, n: usize, threads: usize) -> Result<(Decoded, [f64; 3])> {
    // Some writers leave padding fields out of the text.
    let first = body
        .split(|&b| b == b'\n')
        .find(|l| tokens(l).next().is_some_and(|t| t[0] != b'#'))
        .unwrap_or_default();
    let found = tokens(first).count();
    let elems: Vec<Elem> = if found == lay.elems.len() {
        lay.elems.clone()
    } else {
        let mut k = 0;
        let mut kept = Vec::new();
        for f in &h.fields {
            for _ in 0..f.count {
                if f.name != "_" {
                    kept.push(lay.elems[k]);
                }
                k += 1;
            }
        }
        kept
    };
    let axis: [Option<usize>; 3] = std::array::from_fn(|a| elems.iter().position(|e| e.role == Role::Pos(a)));
    let toks: Vec<&[u8]> = tokens(first).collect();
    let p: [f64; 3] = std::array::from_fn(|a| {
        axis[a]
            .and_then(|k| toks.get(k))
            .and_then(|t| f64_from(t))
            .unwrap_or(f64::NAN)
    });
    let origin = origin_near(p);
    // Pieces of whole lines.
    let mut cuts = vec![0];
    while let Some(&last) = cuts.last() {
        if last >= body.len() {
            break;
        }
        let want = last + PIECE;
        let cut = if want >= body.len() {
            body.len()
        } else {
            body[want..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(body.len(), |k| want + k + 1)
        };
        cuts.push(cut);
    }
    let pieces: Vec<&[u8]> = cuts.windows(2).map(|w| &body[w[0]..w[1]]).collect();
    let parts = points::map(pieces, threads, |text| {
        let mut d = Decoded {
            values: vec![Vec::new(); lay.values.len()],
            ..Decoded::default()
        };
        for line in text.split(|&b| b == b'\n') {
            if d.pos.len() >= n {
                break;
            }
            let mut t = tokens(line).peekable();
            match t.peek() {
                None => continue,
                Some(s) if s[0] == b'#' => continue,
                _ => {}
            }
            let mut p = [f32::NAN; 3];
            let mut color = 0;
            for e in &elems {
                let s = t.next();
                match e.role {
                    Role::Pos(a) => {
                        p[a] = s.and_then(f64_from).map_or(f32::NAN, |v| f32_of(v - origin[a]));
                    }
                    Role::Packed(_) => color = s.map_or(0, packed_text),
                    Role::Value(v) => d.values[v].push(s.and_then(f32_from).unwrap_or(f32::NAN)),
                    Role::Skip => {}
                }
            }
            d.pos.push(p);
            if lay.packed {
                d.packed.push(color);
            }
        }
        d
    });
    let mut d = Decoded::new(lay, 0)?;
    for part in parts {
        if d.pos.len() >= n {
            break;
        }
        d.append(part);
    }
    Ok((d, origin))
}

/// The scale that takes a color channel stored as `kind` of `size` bytes to 0..=1, given its
/// largest value.
fn channel_scale(kind: Kind, size: usize, max: f32) -> f32 {
    match (kind, size) {
        (Kind::F, _) if max <= 1.0 => 1.0,
        (Kind::U | Kind::I, 2) if max > 255.0 => 1.0 / 65535.0,
        _ => 1.0 / 255.0,
    }
}

fn cloud(d: Decoded, h: &Header, lay: &Layout, origin: [f64; 3]) -> Document {
    let n = d.pos.len();
    // Z up, as the text scanner formats are taken to be: turned to glTF's Y up in the data.
    let turn = |[x, y, z]: [f32; 3]| [x, z, -y];
    let mut cloud = Cloud {
        positions: d.pos.into_iter().map(turn).collect(),
        origin: points::z_up(origin),
        ..Cloud::default()
    };
    if lay.packed && d.packed.len() == n {
        let alpha = lay.elems.iter().any(|e| e.role == Role::Packed(true));
        cloud.colors = d
            .packed
            .iter()
            .map(|&v| {
                let [b, g, r, a] = v.to_le_bytes();
                let mut c = points::linear8([r, g, b]);
                if alpha {
                    c[3] = u16::from(a) * 257;
                }
                c
            })
            .collect();
    }
    let find = |u: Use| lay.values.iter().position(|v| v.usage == u);
    let mut values: Vec<Option<Vec<f32>>> = d.values.into_iter().map(Some).collect();
    if let [Some(x), Some(y), Some(z)] = [0, 1, 2].map(|a| find(Use::Normal(a))) {
        let (x, y, z) = (values[x].take(), values[y].take(), values[z].take());
        if let (Some(x), Some(y), Some(z)) = (x, y, z) {
            cloud.normals = x.iter().zip(&y).zip(&z).map(|((&x, &y), &z)| turn([x, y, z])).collect();
        }
    }
    if cloud.colors.is_empty()
        && let [Some(r), Some(g), Some(b)] = [0, 1, 2].map(|c| find(Use::Channel(c)))
    {
        let a = find(Use::Channel(3));
        let scale = |k: usize, v: &[f32]| {
            let max = v.iter().copied().filter(|v| v.is_finite()).fold(0.0, f32::max);
            channel_scale(lay.values[k].kind, lay.values[k].size, max)
        };
        let ch: Vec<(Vec<f32>, f32)> = [Some(r), Some(g), Some(b), a]
            .into_iter()
            .flatten()
            .filter_map(|k| values[k].take().map(|v| (scale(k, &v), v)).map(|(s, v)| (v, s)))
            .collect();
        if ch.len() >= 3 {
            cloud.colors = (0..n)
                .map(|i| {
                    let c = |k: usize| ch[k].0.get(i).copied().unwrap_or(0.0) * ch[k].1;
                    let alpha = if ch.len() > 3 { c(3) } else { 1.0 };
                    points::linear_f([c(0), c(1), c(2)], alpha)
                })
                .collect();
        }
    }
    for (v, info) in values.into_iter().zip(&lay.values) {
        if let Some(v) = v {
            cloud.columns.push(Column::new(&info.name, v));
        }
    }
    let fields: Vec<Value> = h
        .fields
        .iter()
        .map(|f| {
            let t = match f.kind {
                Kind::I => "I",
                Kind::U => "U",
                Kind::F => "F",
            };
            json!({ "name": f.name, "size": f.size, "type": t, "count": f.count })
        })
        .collect();
    let v = h.viewpoint;
    let mut source = json!({
        "data": match h.data {
            Data::Ascii => "ascii",
            Data::Binary => "binary",
            Data::Compressed => "binary_compressed",
        },
        "fields": fields,
        "width": h.width,
        "height": h.height,
        "points": h.points,
        "viewpoint": { "translation": [v[0], v[1], v[2]], "rotation": [v[4], v[5], v[6], v[3]] },
    });
    if !lay.dropped.is_empty() {
        source["droppedValues"] = json!(lay.dropped);
    }
    let mut facts = Map::new();
    if n as u64 != h.points {
        facts.insert("declared_points".into(), json!(h.points));
    }
    // The format says nothing of units or axes: Z up is assumed, as for the scanners' text formats.
    let mut info = SourceInfo {
        format: Some("PCD".into()),
        version: Some(h.version.clone()).filter(|v| !v.is_empty()),
        up_axis: Fact::Assumed(Axis::PosZ),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    let Value::Object(source) = source else { unreachable!() };
    cloud.document(&info, source, facts)
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
    use super::*;
    use crate::document::Pose;
    use crate::source::Fact;

    fn load(bytes: &[u8]) -> Result<Document> {
        read(bytes, Keep::ALL)
    }

    fn attrs(doc: &Document) -> Vec<String> {
        let mut v: Vec<String> = doc.json()["meshes"][0]["primitives"][0]["attributes"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// The values of attribute `name`, as floats (normalized ones scaled to 0..=1).
    fn values(doc: &Document, name: &str) -> Vec<f32> {
        let a = doc.json()["meshes"][0]["primitives"][0]["attributes"][name]
            .as_u64()
            .unwrap() as usize;
        let acc = &doc.json()["accessors"][a];
        let view = &doc.json()["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let buf = doc.buffer(view["buffer"].as_u64().unwrap_or(0) as usize).unwrap();
        let off = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let b = &buf[off..off + len];
        match acc["componentType"].as_u64() {
            Some(5126) => b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Some(5123) => b
                .chunks(2)
                .map(|c| f32::from(u16::from_le_bytes(c.try_into().unwrap())) / 65535.0)
                .collect(),
            t => panic!("{t:?}"),
        }
    }

    const ASCII: &str = "# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb intensity
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 1 2 3 1 0 0 0
POINTS 3
DATA ascii
0.5 1 -2 4278190335 7
1 2 3 4.2108e+06 9
nan nan nan 0 0
";

    #[test]
    fn ascii_reads_with_packed_colors() {
        let doc = load(ASCII.as_bytes()).unwrap();
        assert_eq!(doc.facts["points"], 2);
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("PCD"));
        assert_eq!(info.version.as_deref(), Some("0.7"));
        // Z up is assumed, as for a scanner's text formats, and turned in the data; a cloud has no root node.
        assert_eq!(
            (info.meters_per_unit, info.up_axis),
            (Fact::Unknown, Fact::Assumed(Axis::PosZ))
        );
        assert!(!info.converted.units && info.converted.up_axis);
        assert_eq!(doc.json()["nodes"].as_array().unwrap().len(), 1);
        assert!(doc.json()["nodes"][0].get("rotation").is_none());
        assert_eq!(attrs(&doc), ["COLOR_0", "POSITION", "_INTENSITY"]);
        let p = values(&doc, "POSITION");
        // (0.5, 1, -2) and (1, 2, 3) with Z up.
        assert_eq!(&p[..6], &[0.5, -2.0, -1.0, 1.0, 3.0, -2.0]);
        let c = values(&doc, "COLOR_0");
        // 0xFF0000FF: blue, with an alpha byte that rgb ignores.
        assert_eq!(&c[..4], &[0.0, 0.0, 1.0, 1.0]);
        // The float 4.2108e6 has the bits 0x4A808060 (the old way of writing rgb).
        let bits = 4.2108e6f32.to_bits().to_le_bytes();
        let want = points::linear8([bits[2], bits[1], bits[0]]);
        assert!((c[4] - f32::from(want[0]) / 65535.0).abs() < 1e-6);
        assert_eq!(values(&doc, "_INTENSITY"), [7.0, 9.0]);
        let vp = &doc.json()["asset"]["extras"]["source"]["viewpoint"];
        assert_eq!(vp["translation"], json!([1.0, 2.0, 3.0]));
        assert_eq!(vp["rotation"], json!([0.0, 0.0, 0.0, 1.0]));
    }

    /// A PCD file of `points` written as `data`, with an x y z (f32), normal (f32), label (u32),
    /// a padding byte field and a two-value histogram (u8).
    fn file(data: &str, pts: &[[f32; 7]]) -> Vec<u8> {
        let n = pts.len();
        let mut out = format!(
            "VERSION .7\nFIELDS x y z normal_x normal_y normal_z label _ hist\nSIZE 4 4 4 4 4 4 4 1 1\nTYPE F F F F F F U U U\nCOUNT 1 1 1 1 1 1 1 1 2\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA {data}\n"
        )
        .into_bytes();
        let record = |p: &[f32; 7], out: &mut Vec<u8>| {
            for v in &p[..6] {
                out.extend(v.to_le_bytes());
            }
            out.extend((p[6] as u32).to_le_bytes());
            out.extend([0xAA, 3, 4]);
        };
        match data {
            "ascii" => {
                for p in pts {
                    out.extend(
                        format!("{} {} {} {} {} {} {} 0 3 4\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6]).bytes(),
                    );
                }
            }
            "binary" => pts.iter().for_each(|p| record(p, &mut out)),
            _ => {
                // Field after field, stored as LZF literals.
                let mut soa = Vec::new();
                for k in 0..7 {
                    for p in pts {
                        if k == 6 {
                            soa.extend((p[6] as u32).to_le_bytes());
                        } else {
                            soa.extend(p[k].to_le_bytes());
                        }
                    }
                }
                soa.extend(std::iter::repeat_n(0xAA, n));
                for _ in pts {
                    soa.extend([3, 4]);
                }
                let mut lz = Vec::new();
                for c in soa.chunks(32) {
                    lz.push((c.len() - 1) as u8);
                    lz.extend(c);
                }
                out.extend((lz.len() as u32).to_le_bytes());
                out.extend((soa.len() as u32).to_le_bytes());
                out.extend(lz);
            }
        }
        out
    }

    fn sample() -> Vec<[f32; 7]> {
        (0..50)
            .map(|i| {
                let f = i as f32;
                [
                    f * 0.25,
                    -f,
                    3.5,
                    0.0,
                    0.0,
                    if i % 2 == 0 { 2.0 } else { -1.0 },
                    f * 3.0,
                ]
            })
            .collect()
    }

    #[test]
    fn every_kind_of_data_reads_alike() {
        let pts = sample();
        let mut first = None;
        for data in ["ascii", "binary", "binary_compressed"] {
            let doc = load(&file(data, &pts)).unwrap();
            assert_eq!(doc.facts["points"], 50, "{data}");
            assert_eq!(attrs(&doc), ["NORMAL", "POSITION", "_LABEL"], "{data}");
            let p = values(&doc, "POSITION");
            // (0.25, -1, 3.5) with Z up is (0.25, 3.5, 1) with Y up; a normal's z is its y.
            assert_eq!(&p[3..6], &[0.25, 3.5, 1.0]);
            let nz: Vec<f32> = values(&doc, "NORMAL").chunks(3).map(|c| c[1]).collect();
            assert_eq!(&nz[..2], &[1.0, -1.0]);
            assert_eq!(values(&doc, "_LABEL")[7], 21.0);
            let constant = &doc.json()["meshes"][0]["extras"]["constant"];
            assert_eq!(constant["_HIST_0"], 3.0, "{data}");
            assert_eq!(constant["_HIST_1"], 4.0);
            let got = (p, values(&doc, "_LABEL"));
            if let Some(f) = &first {
                assert_eq!(f, &got, "{data}");
            } else {
                first = Some(got);
            }
        }
    }

    #[test]
    fn lzf_back_references_expand() {
        // "abc" then a run copying back 3 bytes for 6: "abcabcabc".
        let lz = [2, b'a', b'b', b'c', 4 << 5, 2];
        assert_eq!(lzf(&lz, 9).unwrap(), b"abcabcabc");
        assert!(lzf(&lz, 8).is_none());
        assert!(lzf(&[1 << 5, 5], 3).is_none());
        assert!(lzf(&[5, 1], 6).is_none());
    }

    #[test]
    fn double_positions_far_away_keep_their_detail() {
        let mut out = b"VERSION .7\nFIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n".to_vec();
        for v in [637_012.24f64, 849_028.31, 431.66, 637_012.25, 849_028.32, 431.67] {
            out.extend(v.to_le_bytes());
        }
        let doc = load(&out).unwrap();
        let t = &doc.json()["nodes"][0]["translation"];
        assert_eq!(t, &json!([637_012.0, 0.0, -849_028.0]));
        let p = values(&doc, "POSITION");
        assert!((p[3] - p[0] - 0.01).abs() < 1e-5, "{p:?}");
        assert!((f64::from(p[2]) + 0.31).abs() < 1e-5);
    }

    #[test]
    fn separate_channels_and_organized_holes() {
        let text = "VERSION 0.7\nFIELDS x y z r g b\nSIZE 4 4 4 1 1 1\nTYPE F F F U U U\nWIDTH 2\nHEIGHT 2\nDATA ascii\n0 0 0 255 0 0\nnan nan nan 0 0 0\n1 0 0 0 255 0\n1 1 0 0 0 255\n";
        let doc = load(text.as_bytes()).unwrap();
        assert_eq!(doc.facts["points"], 3);
        let c = values(&doc, "COLOR_0");
        assert_eq!(&c[..8], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(doc.json()["asset"]["extras"]["source"]["height"], 2);
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(m.prims.iter().map(|p| p.points.len()).sum::<usize>(), 3);
    }

    #[test]
    fn threads_read_the_same_text() {
        let pts: Vec<[f32; 7]> = (0..40_000)
            .map(|i| [i as f32 * 0.5, 1.0, 2.0, 0.0, 1.0, 0.0, (i % 7) as f32])
            .collect();
        let bytes = file("ascii", &pts);
        assert!(bytes.len() > 2 * PIECE);
        let many = read(&bytes, Keep::ALL.with_threads(4)).unwrap();
        let one = read(&bytes, Keep::ALL).unwrap();
        crate::keep::NO_THREADS.set(true);
        let none = read(&bytes, Keep::ALL.with_threads(4)).unwrap();
        crate::keep::NO_THREADS.set(false);
        assert_eq!(many.facts["points"], 40_000);
        for d in [&one, &none] {
            assert_eq!(values(&many, "POSITION"), values(d, "POSITION"));
            assert_eq!(values(&many, "_LABEL"), values(d, "_LABEL"));
        }
    }

    #[test]
    fn too_many_points_are_refused() {
        let err = read(&file("binary", &sample()), Keep::ALL.with_max_points(10)).unwrap_err();
        assert!(err.to_string().contains("Keep::max_points"), "{err}");
    }

    #[test]
    fn broken_files_fail_cleanly() {
        let mut rng = crate::test_rng::Pcg32::new(7);
        for data in ["ascii", "binary", "binary_compressed"] {
            let good = file(data, &sample());
            for cut in (0..good.len()).step_by(7) {
                let _ = load(&good[..cut]);
            }
            for _ in 0..300 {
                let mut b = good.clone();
                points::corrupt(&mut b, &mut rng, 4);
                let _ = load(&b);
            }
        }
        for junk in [
            &b"DATA binary\n"[..],
            b"FIELDS x y z\nSIZE 4 4\nDATA ascii\n",
            b"FIELDS x\nCOUNT 99999999\nDATA ascii\n",
            b"FIELDS x y z\nPOINTS 99999999999\nDATA binary\n",
        ] {
            assert!(load(junk).is_err());
        }
    }
}
