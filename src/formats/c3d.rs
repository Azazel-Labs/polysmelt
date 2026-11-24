//! C3D motion capture as a glTF document: each marker a node, its track a translation animation.
//!
//! A C3D file is in 512-byte blocks. The first is the header: its first byte is the block the parameters start at
//! (counting from 1), its second 0x50. The parameter section's fourth byte names the processor that wrote the file,
//! and so how every number in it is stored: 84 is Intel (little-endian, IEEE floats), 85 DEC (little-endian, DEC
//! floats: the two 16-bit halves swapped and the value four times too big) and 86 MIPS (big-endian, IEEE floats).
//! The header gives the layout of the data: the number of markers ("points"), the analog values stored with each
//! frame, the first and last frame, a scale (negative when the data is floats), the block the data starts at, and
//! the frame rate. The parameters follow the header block, each record a name, a group or parameter number, an
//! offset to the next record, and for a parameter a type, dimensions and values; groups gather them (`POINT`,
//! `ANALOG`, `TRIAL`, …).
//!
//! Each frame stores four numbers per marker, X, Y, Z and a residual, then the frame's analog values. Integer data
//! is 16-bit, multiplied by the scale's magnitude to give lengths; float data is 32-bit as it is. A negative residual
//! marks a sample as missing.
//!
//! The document has a root node named `markers`, and a node per marker under it, named from `POINT:LABELS`
//! (continued in `LABELS2`, `LABELS3`, …), or `marker<n>` where it has none. A marker's rest position is its first
//! sample that isn't missing. Its track is a linear translation channel with a key at each frame where it isn't
//! missing, frame `k` (counting from 0) at `k / rate` seconds; glTF has no way to leave a gap, so a missing stretch
//! is filled in by the keys on each side, and the missing samples are counted. Lengths become meters from
//! `POINT:UNITS` (millimeters when it's absent). The root turns the capture so that `POINT:Y_SCREEN`, the axis that
//! points up on screen, is glTF's Y and `POINT:X_SCREEN` is its X; a file without them is taken to be Z-up, as
//! laboratory captures nearly all are.
//!
//! The source info has `POINT:UNITS` as the `unit` and its length as `metersPerUnit`, declared (assumed `mm` when
//! it's absent, unknown when it names a unit that isn't known, which leaves lengths as they are); `POINT:Y_SCREEN`
//! as the up axis, declared (assumed `+Z` without the screen axes); the header's rate, or `POINT:RATE`, as
//! `frameRate`, declared (an assumed default when neither gives one); and `MANUFACTURER:SOFTWARE` as the `tool`.
//! `MANUFACTURER:COMPANY`, `MANUFACTURER:VERSION_LABEL` and `SUBJECTS:NAMES` are kept as `company`,
//! `softwareVersion` and `subject`, `POINT:X_SCREEN` as `xScreen`.
//!
//! Analog channels (force plates, EMG and such) and the header's events have no glTF form, so they're left out and
//! counted. A file whose last frame doesn't fit in the header's 16 bits gives its frame count in `TRIAL` or
//! `POINT:FRAMES`. Frames are read only as far as the file holds them, so a cut-short file loses its tail, which is
//! counted; nothing is allocated from a count the file gives beyond what its bytes can back.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::Input;
use super::scene::{self, Animation, Channel, Interp, Node, Path, Scene};
use crate::Result;
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact, meters_in};

/// Frames per second when neither the header nor `POINT:RATE` gives one.
const DEFAULT_RATE: f64 = 30.0;
/// A C3D block's size in bytes.
const BLOCK: usize = 512;

/// How the numbers in a file are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Processor {
    Intel,
    Dec,
    Mips,
}

impl Processor {
    fn name(self) -> &'static str {
        match self {
            Self::Intel => "Intel",
            Self::Dec => "DEC",
            Self::Mips => "MIPS",
        }
    }
}

/// Reads numbers at offsets in the processor's byte order; `None` past the end.
#[derive(Clone, Copy)]
struct Reader<'a> {
    b: &'a [u8],
    p: Processor,
}

impl<'a> Reader<'a> {
    fn slice(&self, at: usize, n: usize) -> Option<&'a [u8]> {
        self.b.get(at..at.checked_add(n)?)
    }

    fn arr<const N: usize>(&self, at: usize) -> Option<[u8; N]> {
        self.slice(at, N)?.try_into().ok()
    }

    fn u16(&self, at: usize) -> Option<u16> {
        let a = self.arr::<2>(at)?;
        Some(if self.p == Processor::Mips {
            u16::from_be_bytes(a)
        } else {
            u16::from_le_bytes(a)
        })
    }

    fn i16(&self, at: usize) -> Option<i16> {
        self.u16(at).map(|v| i16::from_ne_bytes(v.to_ne_bytes()))
    }

    fn f32(&self, at: usize) -> Option<f32> {
        let a = self.arr::<4>(at)?;
        Some(match self.p {
            Processor::Intel => f32::from_le_bytes(a),
            Processor::Mips => f32::from_be_bytes(a),
            // The halves swapped give IEEE's layout with an exponent two higher.
            Processor::Dec => f32::from_le_bytes([a[2], a[3], a[0], a[1]]) / 4.0,
        })
    }
}

/// A parameter's type, dimensions and values.
struct Param<'a> {
    /// -1 characters, 1 bytes, 2 16-bit integers, 4 floats.
    kind: i8,
    dims: Vec<usize>,
    data: &'a [u8],
}

/// The parameters by `GROUP:NAME`, upper case.
struct Params<'a> {
    rd: Reader<'a>,
    map: HashMap<String, Param<'a>>,
    /// Records past the first that couldn't be read whole.
    cut_short: bool,
}

impl<'a> Params<'a> {
    /// Walks the records from `at` (just past the section's four-byte header) to the last, or the end of the
    /// section: `end`, or the end of the file. Some writers leave the last record's offset pointing on, past the
    /// section.
    fn read(rd: Reader<'a>, mut at: usize, end: usize) -> Self {
        let mut groups: HashMap<u8, String> = HashMap::new();
        let mut found: Vec<(u8, String, Param<'a>)> = Vec::new();
        let mut cut_short = false;
        while at < end {
            let Some(&[n, id]) = rd.slice(at, 2) else {
                cut_short = at < rd.b.len();
                break;
            };
            let len = usize::from(i8::from_ne_bytes([n]).unsigned_abs());
            if len == 0 {
                break;
            }
            let id = i8::from_ne_bytes([id]);
            let (Some(name), Some(offset)) = (rd.slice(at + 2, len), rd.i16(at + 2 + len)) else {
                cut_short = true;
                break;
            };
            let name = String::from_utf8_lossy(name).trim().to_ascii_uppercase();
            let body = at + 4 + len;
            if id < 0 {
                groups.entry(id.unsigned_abs()).or_insert(name);
            } else if id > 0 {
                let Some(p) = Self::param(rd, body) else {
                    cut_short = true;
                    break;
                };
                found.push((id.unsigned_abs(), name, p));
            }
            // An offset of 0 marks the last record; one that doesn't move forward ends the walk too.
            let Ok(step) = usize::try_from(offset) else { break };
            if step == 0 {
                break;
            }
            at = at + 2 + len + step;
        }
        let mut map = HashMap::new();
        for (g, name, p) in found {
            if let Some(group) = groups.get(&g) {
                map.entry(format!("{group}:{name}")).or_insert(p);
            }
        }
        Self { rd, map, cut_short }
    }

    /// The parameter whose type byte is at `at`; `None` if it runs past the end of the file.
    fn param(rd: Reader<'a>, at: usize) -> Option<Param<'a>> {
        let kind = i8::from_ne_bytes([*rd.b.get(at)?]);
        let ndims = usize::from(*rd.b.get(at + 1)?);
        let dims: Vec<usize> = rd.slice(at + 2, ndims)?.iter().map(|&d| usize::from(d)).collect();
        let count = dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d))?;
        let size = count.checked_mul(usize::from(kind.unsigned_abs()))?;
        let data = rd.slice(at + 2 + ndims, size)?;
        Some(Param { kind, dims, data })
    }

    fn get(&self, key: &str) -> Option<&Param<'a>> {
        self.map.get(key)
    }

    /// The parameter's values as numbers.
    fn numbers(&self, key: &str) -> Vec<f64> {
        let Some(p) = self.get(key) else { return Vec::new() };
        let rd = Reader {
            b: p.data,
            p: self.rd.p,
        };
        match p.kind {
            1 => p.data.iter().map(|&b| f64::from(b)).collect(),
            2 => (0..p.data.len() / 2)
                .filter_map(|i| rd.i16(i * 2))
                .map(f64::from)
                .collect(),
            4 => (0..p.data.len() / 4)
                .filter_map(|i| rd.f32(i * 4))
                .map(f64::from)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The parameter's first value as a number.
    fn number(&self, key: &str) -> Option<f64> {
        self.numbers(key).first().copied().filter(|v| v.is_finite())
    }

    /// A 16-bit integer parameter's values, read unsigned.
    fn words(&self, key: &str) -> Vec<u16> {
        let Some(p) = self.get(key).filter(|p| p.kind == 2) else {
            return Vec::new();
        };
        let rd = Reader {
            b: p.data,
            p: self.rd.p,
        };
        (0..p.data.len() / 2).filter_map(|i| rd.u16(i * 2)).collect()
    }

    /// The parameter's first value as a count: a 16-bit integer read unsigned, as files with more than 32767 of
    /// something store it.
    fn count(&self, key: &str) -> Option<usize> {
        match self.get(key)?.kind {
            2 => self.words(key).first().map(|&w| usize::from(w)),
            _ => whole(self.number(key)?),
        }
    }

    /// A character parameter's strings, each the first dimension long, trimmed.
    fn strings(&self, key: &str) -> Vec<String> {
        let Some(p) = self.get(key).filter(|p| p.kind == -1) else {
            return Vec::new();
        };
        let tidy = |b: &[u8]| {
            scene::text(b)
                .trim_matches(|c: char| c == '\0' || c.is_whitespace())
                .to_owned()
        };
        match p.dims.first() {
            None => vec![tidy(p.data)],
            Some(0) => Vec::new(),
            Some(&n) => p.data.chunks(n).map(tidy).collect(),
        }
    }

    fn string(&self, key: &str) -> Option<String> {
        self.strings(key).into_iter().next().filter(|s| !s.is_empty())
    }
}

/// `v` as a count, if it's a whole number that is one.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the range is checked first"
)]
fn whole(v: f64) -> Option<usize> {
    (0.0..4e9).contains(&v).then(|| v.round() as usize)
}

/// Frame `f`'s time in seconds.
#[allow(clippy::cast_precision_loss, reason = "frame numbers are far below 2^52")]
fn seconds(f: usize, rate: f64) -> f32 {
    f32_of(f as f64 / rate)
}

/// A screen axis (`+X`, `-Z`, `Y`, …) as a unit vector.
fn axis(s: &str) -> Option<[f64; 3]> {
    let s = s.trim().to_ascii_uppercase();
    let (sign, a) = match s.strip_prefix('-') {
        Some(rest) => (-1.0, rest),
        None => (1.0, s.trim_start_matches('+')),
    };
    let k = match a {
        "X" => 0,
        "Y" => 1,
        "Z" => 2,
        _ => return None,
    };
    let mut v = [0.0; 3];
    v[k] = sign;
    Some(v)
}

/// The rotation that takes screen right `x` to glTF's +X and screen up `y` to +Y; `None` if they're parallel.
fn screen_rotation(x: [f64; 3], y: [f64; 3]) -> Option<[f64; 4]> {
    let z = [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ];
    if z.iter().all(|c| *c == 0.0) {
        return None;
    }
    // Row r of the rotation is the file's direction for glTF axis r; stored column-major.
    let rows = [x, y, z];
    let mut m = [0.0; 9];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            m[c * 3 + r] = *v;
        }
    }
    Some(scene::matrix_quat(&m))
}

/// Reads a C3D file.
pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    let bytes = input.read()?;
    let b: &[u8] = &bytes;
    ensure!(
        b.len() >= 2 && b[1] == 0x50,
        UnknownFormat,
        "not a C3D file: its second byte isn't 0x50"
    );
    ensure!(
        b[0] >= 2,
        Corrupt,
        "C3D header points at parameter block {}, not one past the header",
        b[0]
    );
    let param_start = (usize::from(b[0]) - 1) * BLOCK;
    let Some(&proc_byte) = b.get(param_start + 3) else {
        bail!(Truncated, "C3D file is cut short: its parameters are missing");
    };
    let p = match proc_byte {
        84 => Some(Processor::Intel),
        85 => Some(Processor::Dec),
        86 => Some(Processor::Mips),
        _ => None,
    };
    let rd = Reader {
        b,
        p: p.unwrap_or(Processor::Intel),
    };
    let blocks = usize::from(b[param_start + 2]);
    let end = if blocks == 0 {
        b.len()
    } else {
        param_start + blocks * BLOCK
    };
    let params = Params::read(rd, param_start + 4, end);
    let word = |w: usize| rd.u16((w - 1) * 2).unwrap_or(0);
    let float = |w: usize| rd.f32((w - 1) * 2).map(f64::from).filter(|v| v.is_finite());

    // The layout: the header's, with the parameters where it has none.
    let points = match word(2) {
        0 => params.count("POINT:USED").unwrap_or(0).min(usize::from(u16::MAX)),
        n => usize::from(n),
    };
    let analog_words = usize::from(word(3));
    let header_scale = float(7).filter(|s| *s != 0.0);
    let scale = header_scale.or_else(|| params.number("POINT:SCALE")).unwrap_or(1.0);
    let floats = scale < 0.0;
    let point_scale = if scale == 0.0 { 1.0 } else { scale.abs() };
    let start_block = match word(9) {
        0 => params.count("POINT:DATA_START").unwrap_or(0),
        n => usize::from(n),
    };
    ensure!(start_block > 0, Corrupt, "C3D file doesn't say where its data starts");
    let data_start = (start_block - 1).saturating_mul(BLOCK);
    let rate_given = float(11)
        .filter(|r| *r > 0.0)
        .or_else(|| params.number("POINT:RATE").filter(|r| *r > 0.0));
    let rate = rate_given.unwrap_or(DEFAULT_RATE);
    let (first, last) = (usize::from(word(4)), usize::from(word(5)));
    let claimed = if last == usize::from(u16::MAX) || last < first {
        // Two 16-bit words, low then high.
        let field = |key: &str| match params.words(key)[..] {
            [lo, hi, ..] => Some(usize::from(lo) | usize::from(hi) << 16),
            _ => None,
        };
        match (field("TRIAL:ACTUAL_START_FIELD"), field("TRIAL:ACTUAL_END_FIELD")) {
            (Some(s), Some(e)) if e >= s => Some(e - s + 1),
            _ => params.count("POINT:FRAMES"),
        }
    } else {
        Some(last - first + 1)
    };
    let width = if floats { 4 } else { 2 };
    let frame_size = (points * 4 + analog_words) * width;
    let fit = b.len().saturating_sub(data_start).checked_div(frame_size).unwrap_or(0);
    let frames = match claimed {
        Some(c) => fit.min(c),
        None => fit,
    };
    let missing_frames = claimed.map_or(0, |c| c.saturating_sub(frames));

    // The markers' names and tracks.
    let mut labels = params.strings("POINT:LABELS");
    for k in 2.. {
        if labels.len() >= points {
            break;
        }
        let more = params.strings(&format!("POINT:LABELS{k}"));
        if more.is_empty() {
            break;
        }
        labels.extend(more);
    }
    let descriptions = params.strings("POINT:DESCRIPTIONS");
    let mut tracks: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); points];
    let mut invalid = 0usize;
    for f in 0..frames {
        let at = data_start + f * frame_size;
        let time = seconds(f, rate);
        for (m, track) in tracks.iter_mut().enumerate() {
            let at = at + m * 4 * width;
            let sample = if floats {
                let v = [0, 1, 2, 3].map(|i| rd.f32(at + i * 4).map_or(f64::NAN, f64::from));
                (v[3] >= 0.0).then_some([v[0], v[1], v[2]])
            } else {
                let v = [0, 1, 2, 3].map(|i| f64::from(rd.i16(at + i * 2).unwrap_or(-1)));
                (v[3] >= 0.0).then(|| [v[0], v[1], v[2]].map(|c| c * point_scale))
            };
            match sample.filter(|s| s.iter().all(|c| c.is_finite())) {
                Some(s) => {
                    track.0.push(time);
                    track.1.extend(s.map(f32_of));
                }
                None => invalid += 1,
            }
        }
    }

    let mut scene = Scene::default();
    let root = scene.add(Node::named("markers", None));
    let mut channels = Vec::new();
    let mut empty = 0usize;
    for (m, (times, values)) in tracks.into_iter().enumerate() {
        let name = match labels.get(m) {
            Some(l) if !l.is_empty() => l.clone(),
            _ => format!("marker{}", m + 1),
        };
        let mut node = Node::named(name, Some(root));
        if let Some(d) = descriptions.get(m).filter(|d| !d.is_empty()) {
            node.extras = Some(json!({ "description": d }));
        }
        match values.get(..3) {
            Some(v) => node.translation = [v[0], v[1], v[2]],
            None => empty += 1,
        }
        let i = scene.add(node);
        if !times.is_empty() {
            channels.push(Channel {
                node: i,
                path: Path::Translation,
                times,
                values,
                interp: Interp::Linear,
            });
        }
    }
    if keep.animations && !channels.is_empty() {
        scene.animations.push(Animation {
            name: "motion".into(),
            channels,
            extras: None,
        });
    }

    // Units and the up axis.
    let units = params.string("POINT:UNITS");
    let meters = match &units {
        Some(u) => meters_in(u),
        None => Some(0.001),
    };
    let x_screen = params.string("POINT:X_SCREEN");
    let y_screen = params.string("POINT:Y_SCREEN");
    let screen = match (x_screen.as_deref().and_then(axis), y_screen.as_deref().and_then(axis)) {
        (Some(x), Some(y)) => screen_rotation(x, y),
        _ => None,
    };
    let turn = screen.unwrap_or_else(|| scene::Z_UP_TO_Y_UP.map(f64::from));

    let info = &mut scene.info;
    info.format = Some("C3D".into());
    info.meters_per_unit = match (&units, meters) {
        (Some(_), Some(m)) => Fact::Declared(m),
        (None, Some(m)) => Fact::Assumed(m),
        (_, None) => Fact::Unknown,
    };
    info.unit = Some(units.clone().unwrap_or_else(|| "mm".into()));
    info.up_axis = match y_screen.as_deref().and_then(Axis::parse) {
        Some(up) if screen.is_some() => Fact::Declared(up),
        _ => Fact::Assumed(Axis::PosZ),
    };
    info.frame_rate = match rate_given {
        Some(r) => Fact::Declared(r),
        None => Fact::Assumed(rate),
    };
    info.tool = params.string("MANUFACTURER:SOFTWARE");
    let mut source = Map::new();
    source.insert("processor".into(), json!(rd.p.name()));
    source.insert("storage".into(), json!(if floats { "float" } else { "integer" }));
    if let Some(x) = &x_screen {
        source.insert("xScreen".into(), json!(x));
    }
    source.insert("firstFrame".into(), json!(first));
    source.insert("scale".into(), json!(scale));
    for (key, name) in [
        ("MANUFACTURER:COMPANY", "company"),
        ("MANUFACTURER:VERSION_LABEL", "softwareVersion"),
        ("SUBJECTS:NAMES", "subject"),
    ] {
        if let Some(v) = params.string(key) {
            source.insert(name.into(), Value::String(v));
        }
    }
    let analog_channels = params.count("ANALOG:USED").unwrap_or(0);
    let facts = &mut scene.facts;
    facts.insert("markers".into(), json!(points));
    facts.insert("frames".into(), json!(frames));
    if invalid > 0 {
        facts.insert("missing_samples".into(), json!(invalid));
    }
    if empty > 0 {
        facts.insert("markers_never_seen".into(), json!(empty));
    }
    if missing_frames > 0 {
        facts.insert("frames_cut_off".into(), json!(missing_frames));
    }
    if analog_words > 0 || analog_channels > 0 {
        facts.insert("analog_channels_dropped".into(), json!(analog_channels));
        if let Some(r) = params.number("ANALOG:RATE") {
            facts.insert("analog_rate".into(), json!(r));
        }
    }
    let events = usize::from(word(151)).min(18);
    if events > 0 {
        facts.insert("events_dropped".into(), json!(events));
    }
    if p.is_none() {
        facts.insert("processor_unknown".into(), json!(proc_byte));
    }
    if params.cut_short {
        facts.insert("parameters_cut_short".into(), json!(true));
    }
    scene.source = source;
    scene.scale(f32_of(meters.unwrap_or(1.0)));
    scene.rotate_roots(scene::normalize(turn));
    scene.finish(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a C3D file: `markers` names, `frames[f][m]` samples (`None` missing), as integers with `scale` or as
    /// floats when it's negative, for processor `p`.
    fn build(
        p: Processor,
        names: &[&str],
        frames: &[Vec<Option<[f32; 3]>>],
        scale: f32,
        extra: &[(&str, &str)],
    ) -> Vec<u8> {
        let u16b = |v: u16| {
            if p == Processor::Mips {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let f32b = |v: f32| match p {
            Processor::Intel => v.to_le_bytes(),
            Processor::Mips => v.to_be_bytes(),
            Processor::Dec => {
                let a = (v * 4.0).to_le_bytes();
                [a[2], a[3], a[0], a[1]]
            }
        };
        let mut params: Vec<u8> = vec![1, 0x50, 0, 83 + p as u8 + 1];
        let record = |params: &mut Vec<u8>, id: i8, name: &str, body: Vec<u8>, last: bool| {
            params.push(u8::try_from(name.len()).unwrap());
            params.extend(id.to_ne_bytes());
            params.extend(name.as_bytes());
            let offset = if last {
                0
            } else {
                u16::try_from(body.len() + 2).unwrap()
            };
            params.extend(u16b(offset));
            params.extend(body);
        };
        let chars = |s: &[&str]| {
            let n = s.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
            let mut body = vec![0xff, 2, u8::try_from(n).unwrap(), u8::try_from(s.len()).unwrap()];
            for s in s {
                body.extend(format!("{s:<n$}").as_bytes());
            }
            body.push(0);
            body
        };
        record(&mut params, -1, "POINT", vec![0], false);
        let mut used = vec![2, 0];
        used.extend(u16b(u16::try_from(names.len()).unwrap()));
        used.push(0);
        record(&mut params, 1, "USED", used, false);
        record(&mut params, 1, "LABELS", chars(names), false);
        for (k, v) in extra {
            record(&mut params, 1, k, chars(&[v]), false);
        }
        record(&mut params, -2, "ANALOG", vec![0], true);
        let blocks = params.len().div_ceil(BLOCK);
        params[2] = u8::try_from(blocks).unwrap();
        params.resize(blocks * BLOCK, 0);
        let mut header = vec![0u8; BLOCK];
        header[0] = 2;
        header[1] = 0x50;
        let mut put = |w: usize, v: [u8; 2]| header[(w - 1) * 2..w * 2].copy_from_slice(&v);
        put(2, u16b(u16::try_from(names.len()).unwrap()));
        put(4, u16b(1));
        put(5, u16b(u16::try_from(frames.len()).unwrap()));
        let s = f32b(scale);
        put(7, [s[0], s[1]]);
        put(8, [s[2], s[3]]);
        put(9, u16b(u16::try_from(2 + blocks).unwrap()));
        let r = f32b(10.0);
        put(11, [r[0], r[1]]);
        put(12, [r[2], r[3]]);
        let mut data = Vec::new();
        for frame in frames {
            for s in frame {
                if scale < 0.0 {
                    let v = s.map_or([0.0, 0.0, 0.0, -1.0], |v| [v[0], v[1], v[2], 0.0]);
                    for c in v {
                        data.extend(f32b(c));
                    }
                } else {
                    #[allow(clippy::cast_possible_truncation, reason = "test values fit")]
                    let q = |c: f32| (c / scale).round() as i16;
                    let v = s.map_or([0, 0, 0, -1], |v| [q(v[0]), q(v[1]), q(v[2]), 0]);
                    for c in v {
                        data.extend(u16b(u16::from_ne_bytes(c.to_ne_bytes())));
                    }
                }
            }
        }
        data.resize(data.len().div_ceil(BLOCK).max(1) * BLOCK, 0);
        [header, params, data].concat()
    }

    fn sample() -> Vec<Vec<Option<[f32; 3]>>> {
        vec![
            vec![Some([100.0, 200.0, 300.0]), None],
            vec![Some([110.0, 200.0, 300.0]), Some([0.0, 0.0, 1000.0])],
            vec![Some([120.0, 210.0, 290.0]), Some([0.0, 10.0, 1000.0])],
        ]
    }

    fn read(b: &[u8]) -> Result<Document<'static>> {
        load(Input::memory(b, "t.c3d"), Keep::ALL)
    }

    fn near(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn every_processor_and_storage_reads_the_same() {
        for p in [Processor::Intel, Processor::Dec, Processor::Mips] {
            for scale in [0.1, -1.0] {
                let b = build(p, &["LASI", ""], &sample(), scale, &[]);
                let doc = read(&b).unwrap();
                assert_eq!(doc.facts["markers"], json!(2), "{p:?}");
                assert_eq!(doc.facts["frames"], json!(3));
                assert_eq!(doc.facts["missing_samples"], json!(1));
                let info = doc.source_info();
                assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ), "{p:?}");
                assert_eq!(info.meters_per_unit, Fact::Assumed(0.001));
                assert_eq!(info.unit.as_deref(), Some("mm"));
                assert_eq!(info.frame_rate, Fact::Declared(10.0));
                assert!(info.converted.units && info.converted.up_axis);
                let rig = doc.rig().unwrap();
                assert_eq!(rig.nodes[1].name.as_deref(), Some("LASI"));
                assert_eq!(rig.nodes[2].name.as_deref(), Some("marker2"));
                // Z-up millimeters: (100, 200, 300) mm is (0.1, 0.3, -0.2) m in glTF.
                let pose = rig.pose(Some(0), 0.0);
                let world =
                    |i: usize| scene::rotate(pose[0].rotation.map(f64::from), pose[i].translation.map(f64::from));
                assert!(
                    near(&world(1).map(f32_of), &[0.1, 0.3, -0.2]),
                    "{p:?} {scale}: {:?}",
                    world(1)
                );
                let pose = rig.pose(Some(0), 0.2);
                let world =
                    |i: usize| scene::rotate(pose[0].rotation.map(f64::from), pose[i].translation.map(f64::from));
                assert!(near(&world(2).map(f32_of), &[0.0, 1.0, -0.01]), "{p:?} {scale}");
                assert_eq!(rig.animations[0].channels[1].times, [0.1, 0.2]);
            }
        }
    }

    #[test]
    fn screen_axes_and_units() {
        let b = build(
            Processor::Intel,
            &["a"],
            &sample()[..1].iter().map(|f| f[..1].to_vec()).collect::<Vec<_>>(),
            -1.0,
            &[("X_SCREEN", "+X"), ("Y_SCREEN", "+Y"), ("UNITS", "m")],
        );
        let doc = read(&b).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("C3D"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.unit.as_deref(), Some("m"));
        assert!(!info.converted.units);
        let rig = doc.rig().unwrap();
        assert!(near(&rig.nodes[1].rest.translation, &[100.0, 200.0, 300.0]));
        assert!(near(&rig.nodes[0].rest.rotation, &[0.0, 0.0, 0.0, 1.0]));
        // Screen right -Y and up +Z: file -Y is glTF X, +Z is Y, so +X is glTF Z... (−Y)×Z = −X toward the viewer.
        let q = screen_rotation(axis("-Y").unwrap(), axis("+Z").unwrap()).unwrap();
        assert!(near(&scene::rotate(q, [0.0, -1.0, 0.0]).map(f32_of), &[1.0, 0.0, 0.0]));
        assert!(near(&scene::rotate(q, [0.0, 0.0, 1.0]).map(f32_of), &[0.0, 1.0, 0.0]));
        assert!(near(&scene::rotate(q, [-1.0, 0.0, 0.0]).map(f32_of), &[0.0, 0.0, 1.0]));
        assert!(screen_rotation(axis("+Z").unwrap(), axis("-Z").unwrap()).is_none());
    }

    #[test]
    fn not_c3d_is_an_error() {
        for b in [&b""[..], b"hello", &[2, 0x50], &[0, 0x50, 0, 0]] {
            assert!(read(b).is_err(), "{b:?}");
        }
    }

    #[test]
    fn truncated_and_corrupted_files_never_panic() {
        let b = build(Processor::Mips, &["a", "b"], &sample(), 0.1, &[("UNITS", "mm")]);
        for cut in 0..b.len() {
            let _ = read(&b[..cut]);
        }
        let doc = read(&b[..b.len() - BLOCK + 20]).unwrap();
        assert_eq!(doc.facts["frames"], json!(1));
        assert_eq!(doc.facts["frames_cut_off"], json!(2));
        let doc = read(&b[..b.len() - BLOCK + 10]).unwrap();
        assert_eq!(doc.facts["frames"], json!(0));
        assert_eq!(doc.facts["frames_cut_off"], json!(3));
        let mut x: u32 = 1;
        for _ in 0..3000 {
            let mut c = b.clone();
            for _ in 0..4 {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let i = (x >> 8) as usize % c.len();
                c[i] = (x >> 24) as u8;
            }
            let _ = read(&c);
        }
    }
}
