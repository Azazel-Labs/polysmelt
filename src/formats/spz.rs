//! SPZ, a compressed gaussian splat format: a gzip stream of a 16-byte header and the splats a
//! property at a time, versions 1 to 3.
//!
//! Read: positions (half floats in version 1, 24-bit fixed point after), opacity, base color,
//! scale, rotation (three components in versions 1 and 2, the smallest three of four after) and
//! the spherical harmonics up to degree 3 (degree 4's are dropped: `KHR_gaussian_splatting`
//! stops at 3). The coordinate system extension record names the file's axes; without it they
//! are right, up and back, glTF's own. Axes that differ only in direction are flipped in the
//! data; those that also swap up and forward are turned on the node, as turning spherical
//! harmonics would cost precision. The source info has the header's version and the axes' up and
//! handedness (declared by the record, else glTF's, assumed); the spherical-harmonic degree, the
//! antialiasing flag and the record's coordinate system code stay in the asset's `extras.source`. Version 4 (zstd streams in a container
//! of their own) isn't read.
//!
//! The point count is checked against [`Keep::max_points`], and the data is inflated only as
//! far as the header says it reaches: a count the data doesn't fill is refused as cut short. The
//! splats are unpacked as the stream is inflated, which with [`Keep::threads`] above 1 runs on a
//! thread of its own.

use std::f32::consts::FRAC_1_SQRT_2;
use std::io::Read;
use std::sync::mpsc;

use serde_json::{Map, json};

use super::Input;
use crate::document::{Document, Keep};
use crate::gaussian::{REST, Splats, rotation};
use crate::keep::spawn;
use crate::num::usize_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// "NGSP", little-endian, which starts the header.
const MAGIC: u32 = 0x5053_474e;
const HEADER: usize = 16;
/// The most bytes of extension records read past the splats.
const RECORDS: u64 = 1 << 16;
/// The coordinate system extension record's type.
const COORDINATES: u32 = 0xadbe_0003;
/// The base color's quantization scale.
const COLOR_SCALE: f32 = 0.15;
/// Spherical-harmonic coefficients a channel has, by degree, beyond the base one.
const DIM: [usize; 5] = [0, 3, 8, 15, 24];

pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    decode(&input.read()?, keep.points(), keep.threads())
}

/// The splats in an SPZ file, up to `max` of them. With more than one thread, the stream is
/// inflated on another as the splats are unpacked.
fn decode(bytes: &[u8], max: usize, threads: usize) -> Result<Document<'static>> {
    if let Some(&[b'N', b'G', b'S', b'P', v0, v1, v2, v3]) = bytes.first_chunk::<8>() {
        let version = u32::from_le_bytes([v0, v1, v2, v3]);
        ensure!(
            version < 4,
            Unsupported,
            "SPZ version {version} (zstd streams) isn't supported; versions 1 to 3 (gzip) are"
        );
        bail!(UnknownFormat, "not an SPZ file: an SPZ header outside a gzip stream");
    }
    ensure!(
        bytes.starts_with(&[0x1f, 0x8b]),
        UnknownFormat,
        "not an SPZ file: no gzip stream"
    );
    let mut z = flate2::read::MultiGzDecoder::new(bytes);
    let mut head = Vec::with_capacity(HEADER);
    // What a broken stream gives before it breaks is checked for length below.
    let _ = (&mut z).take(HEADER as u64).read_to_end(&mut head);
    ensure!(head.len() == HEADER, Truncated, "SPZ data is cut short: no header");
    let word = |k: usize| u32::from_le_bytes([head[k], head[k + 1], head[k + 2], head[k + 3]]);
    ensure!(word(0) == MAGIC, UnknownFormat, "not an SPZ file: no NGSP header");
    let version = word(4);
    ensure!(
        (1..=3).contains(&version),
        Unsupported,
        "SPZ version {version} isn't supported"
    );
    let n = usize_of(u64::from(word(8)));
    let (degree, fractional, flags) = (usize::from(head[12]), head[13], head[14]);
    ensure!(
        degree < DIM.len(),
        Unsupported,
        "SPZ spherical-harmonic degree {degree} isn't supported"
    );
    ensure!(
        n <= max,
        LimitExceeded,
        "{n} gaussian splats, more than the {max} allowed (Keep::max_points)"
    );
    let mut source = Map::new();
    source.insert("antialiased".into(), json!(flags & 1 != 0));
    if degree > 3 {
        source.insert("droppedShDegree".into(), json!(degree));
    }
    let pos_size = if version == 1 { 6 } else { 9 };
    let rot_size = if version >= 3 { 4 } else { 3 };
    let dim = DIM[degree];
    let per = pos_size + 7 + rot_size + dim * 3;
    let core = n
        .checked_mul(per)
        .context(ErrorKind::LimitExceeded, "too many SPZ splats")?;
    let unpack = Unpack {
        version,
        unit: 0.5f32.powi(i32::from(fractional)),
        dim,
        sections: [pos_size, 1, 3, 3, rot_size, dim * 3].map(|size| (n * size, size)),
        at: 0,
        splats: Splats {
            sh: vec![Vec::new(); REST[degree.min(3)]],
            ..Splats::default()
        },
    };
    // Only what the stream holds is kept; a stream that breaks past the splats loses records.
    let (data, unpack) = inflate(z.take(core as u64 + RECORDS), unpack, threads);
    ensure!(
        data.len() >= core,
        Truncated,
        "SPZ data is cut short: {n} splats need {core} bytes, the stream holds {}",
        data.len()
    );
    let mut s = unpack.splats;
    let records = &data[core..];
    let system = if flags & 2 != 0 { coordinates(records) } else { 0 };
    if system != 0 {
        source.insert("coordinateSystem".into(), json!(system));
    }
    let (flip, turn) = axes(system);
    s.flip(flip);
    let (up_axis, handedness) = frame(system);
    let mut info = SourceInfo {
        format: Some("SPZ".into()),
        version: Some(version.to_string()),
        up_axis,
        handedness,
        ..SourceInfo::default()
    };
    info.converted.up_axis = up_axis.value() != Some(&Axis::PosY);
    info.converted.handedness = handedness.value() == Some(&Handedness::Left);
    Ok(s.document(&info, source, turn, Map::new()))
}

/// The splats unpacked as their bytes are inflated: the sizes of the sections of the stream (each a
/// property of every splat, in the order SPZ stores them) and of their items, and how far into
/// the stream the splats are unpacked.
struct Unpack {
    version: u32,
    /// A fixed-point position's step.
    unit: f32,
    /// Spherical-harmonic coefficients a channel has, beyond the base one.
    dim: usize,
    /// Positions, opacities, colors, scales, rotations and spherical harmonics.
    sections: [(usize, usize); 6],
    at: usize,
    splats: Splats,
}

impl Unpack {
    /// Unpacks the whole items of `data` (the stream so far) not yet unpacked.
    fn advance(&mut self, data: &[u8]) {
        let mut start = 0;
        for (k, (len, size)) in self.sections.into_iter().enumerate() {
            let end = start + len;
            let from = self.at.max(start);
            let to = data.len().min(end);
            if from < to {
                if from == start {
                    self.room(k);
                }
                let whole = from + (to - from) / size * size;
                self.section(k, &data[from..whole]);
                self.at = whole;
            }
            if self.at < end {
                return;
            }
            start = end;
        }
    }

    /// Room in section `k`'s columns for as many splats as there are positions, once these are
    /// all unpacked: the splats' count is taken from the data the stream holds, not the header.
    fn room(&mut self, k: usize) {
        let s = &mut self.splats;
        let n = s.positions.len();
        // Without room, the columns grow as they're filled.
        let _ = match k {
            1 => s.opacities.try_reserve_exact(n),
            2 => s.dc.try_reserve_exact(n),
            3 => s.scales.try_reserve_exact(n),
            4 => s.rotations.try_reserve_exact(n),
            5 => s.sh.iter_mut().try_for_each(|c| c.try_reserve_exact(n)),
            _ => Ok(()),
        };
    }

    /// Unpacks `items`, whole items of section `k`.
    fn section(&mut self, k: usize, items: &[u8]) {
        let s = &mut self.splats;
        match k {
            0 if self.version == 1 => s.positions.extend(
                items
                    .as_chunks::<6>()
                    .0
                    .iter()
                    .map(|p| std::array::from_fn(|k| f16([p[2 * k], p[2 * k + 1]]))),
            ),
            0 => s.positions.extend(items.as_chunks::<9>().0.iter().map(|p| {
                std::array::from_fn(|k| {
                    let v = i32::from_le_bytes([0, p[3 * k], p[3 * k + 1], p[3 * k + 2]]) >> 8;
                    #[allow(clippy::cast_precision_loss, reason = "a 24-bit integer, exact in an f32")]
                    let v = v as f32;
                    v * self.unit
                })
            })),
            1 => s.opacities.extend(items.iter().map(|&a| f32::from(a) / 255.0)),
            2 => s.dc.extend(
                items
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|c| c.map(|v| (f32::from(v) / 255.0 - 0.5) / COLOR_SCALE)),
            ),
            3 => s.scales.extend(
                items
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|c| c.map(|v| (f32::from(v) / 16.0 - 10.0).exp())),
            ),
            4 if self.version >= 3 => s.rotations.extend(
                items
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|r| smallest_three(u32::from_le_bytes(*r))),
            ),
            4 => s.rotations.extend(items.as_chunks::<3>().0.iter().map(|r| {
                let [x, y, z] = r.map(|v| f32::from(v) / 127.5 - 1.0);
                rotation(x, y, z, (1.0 - (x * x + y * y + z * z)).max(0.0).sqrt())
            })),
            _ => {
                let q = |v: u8| (f32::from(v) - 128.0) / 128.0;
                let per = self.dim * 3;
                for (j, col) in s.sh.iter_mut().enumerate() {
                    col.extend(
                        items
                            .chunks_exact(per)
                            .map(|p| [q(p[3 * j]), q(p[3 * j + 1]), q(p[3 * j + 2])]),
                    );
                }
            }
        }
    }
}

/// How many inflated bytes are unpacked at a time.
const CHUNK: u64 = 1 << 18;

/// The inflated stream, as much of it as there is, and `unpack` having unpacked it. With more than
/// one thread, inflating runs on another while this one unpacks; where that thread can't be
/// started, both run on this one.
fn inflate(mut z: impl Read + Send, mut unpack: Unpack, threads: usize) -> (Vec<u8>, Unpack) {
    let mut data = Vec::new();
    let threaded = threads > 1
        && std::thread::scope(|scope| {
            let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(8);
            let z = &mut z;
            // A thread that doesn't start takes nothing from the stream.
            let started = spawn(scope, move || {
                loop {
                    let mut chunk = Vec::new();
                    let read = (&mut *z).take(CHUNK).read_to_end(&mut chunk);
                    // What a broken stream gives before it breaks is kept.
                    if !chunk.is_empty() && tx.send(chunk).is_err() || !matches!(read, Ok(k) if k > 0) {
                        break;
                    }
                }
            });
            if started {
                for chunk in rx {
                    data.extend_from_slice(&chunk);
                    unpack.advance(&data);
                }
            }
            started
        });
    if !threaded {
        loop {
            let read = (&mut z).take(CHUNK).read_to_end(&mut data);
            unpack.advance(&data);
            if !matches!(read, Ok(k) if k > 0) {
                break;
            }
        }
    }
    (data, unpack)
}

/// The coordinate system extension record's value among `records`, or 0 (unnamed) without one.
fn coordinates(mut records: &[u8]) -> u32 {
    let word = |b: &[u8], k: usize| b.get(k..k + 4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    while let (Some(ty), Some(len)) = (word(records, 0), word(records, 4)) {
        let Some(payload) = records.get(8..).and_then(|r| r.get(..usize_of(u64::from(len)))) else {
            break;
        };
        if ty == COORDINATES {
            return word(payload, 0).unwrap_or(0);
        }
        records = &records[8 + payload.len()..];
    }
    0
}

/// The sign of each axis that takes a coordinate system's values to glTF's (right, up, back), and
/// for the systems whose second axis runs forward or back, the node rotation (x, y, z, w) that
/// stands their third axis up. An unknown system is taken to be glTF's own, as SPZ's default.
fn axes(system: u32) -> ([f32; 3], Option<[f32; 4]>) {
    let sign = |plus: bool| if plus { 1.0 } else { -1.0 };
    match system {
        // LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF: right, up and back are bits 0 to 2 (back clear).
        1..=8 => {
            let k = system - 1;
            ([sign(k & 1 != 0), sign(k & 2 != 0), sign(k & 4 == 0)], None)
        }
        // LFD, RFD, LFU, RFU, LBD, RBD, LBU, RBU: right, up (the third letter) and back (the
        // second). In right, back, up, a quarter turn about x stands the third axis up, after
        // the third flips to make the turn's input right, back, down.
        9..=16 => {
            let k = system - 9;
            let flip = [sign(k & 1 != 0), sign(k & 4 != 0), -sign(k & 2 != 0)];
            (flip, Some([FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2]))
        }
        _ => ([1.0; 3], None),
    }
}

/// A coordinate system's up axis and handedness: declared when the file names the system, else
/// glTF's own (+Y up, right-handed), assumed as SPZ's default.
fn frame(system: u32) -> (Fact<Axis>, Fact<Handedness>) {
    // Bit 0 is right (not left), bit 1 up (not down) and bit 2 forward (not back); the second
    // letter is up or down in systems 1 to 8, the third in 9 to 16. Either way, each of left, down
    // and forward flips a right-handed system (RUB, RFU).
    let (k, up, down) = match system {
        1..=8 => (system - 1, Axis::PosY, Axis::NegY),
        9..=16 => (system - 9, Axis::PosZ, Axis::NegZ),
        _ => return (Fact::Assumed(Axis::PosY), Fact::Assumed(Handedness::Right)),
    };
    let flips = u32::from(k & 1 == 0) + u32::from(k & 2 == 0) + u32::from(k & 4 != 0);
    let hand = if flips % 2 == 0 {
        Handedness::Right
    } else {
        Handedness::Left
    };
    (Fact::Declared(if k & 2 != 0 { up } else { down }), Fact::Declared(hand))
}

/// A rotation packed as its three smallest components, 10 bits each (9 of magnitude and a sign),
/// below the two bits that say which of x, y, z, w is the largest, which is positive.
fn smallest_three(mut packed: u32) -> [f32; 4] {
    const MASK: u32 = (1 << 9) - 1;
    let largest = (packed >> 30) as usize;
    let mut q = [0.0f32; 4];
    let mut sum = 0.0;
    for i in (0..4).rev() {
        if i == largest {
            continue;
        }
        #[allow(clippy::cast_precision_loss, reason = "a 9-bit integer")]
        let mag = (packed & MASK) as f32;
        let v = FRAC_1_SQRT_2 * mag / 511.0;
        q[i] = if packed & (1 << 9) != 0 { -v } else { v };
        sum += q[i] * q[i];
        packed >>= 10;
    }
    q[largest] = (1.0 - sum).max(0.0).sqrt();
    rotation(q[0], q[1], q[2], q[3])
}

/// A half-precision float's value.
fn f16(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 if man == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test files are small"
)]
mod tests {
    use std::io::Write as _;

    use super::*;
    use crate::document::Pose;
    use crate::test_rng::Pcg32;

    /// An SPZ file of `n` splats: version, degree and flags as given, the core streams from
    /// `fill`, then `records`.
    fn spz(version: u32, n: u32, degree: u8, flags: u8, records: &[u8]) -> Vec<u8> {
        let mut raw = Vec::new();
        raw.extend(MAGIC.to_le_bytes());
        raw.extend(version.to_le_bytes());
        raw.extend(n.to_le_bytes());
        raw.extend([degree, 12, flags, 0]);
        let pos = if version == 1 { 6 } else { 9 };
        let rot = if version >= 3 { 4 } else { 3 };
        let per = pos + 7 + rot + DIM[usize::from(degree)] * 3;
        let mut rng = Pcg32::new(5);
        for _ in 0..n as usize * per {
            raw.push(rng.below(256) as u8);
        }
        if version == 1 {
            // Finite half floats, under 2^15.
            for i in 0..n as usize * 3 {
                raw[HEADER + 2 * i + 1] &= 0x3b;
            }
        }
        raw.extend(records);
        gzip(&raw)
    }

    fn gzip(raw: &[u8]) -> Vec<u8> {
        let mut z = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(raw).unwrap();
        z.finish().unwrap()
    }

    fn splats(doc: &Document) -> u64 {
        doc.facts["splats"].as_u64().unwrap()
    }

    #[test]
    fn every_version_and_degree_reads() {
        for version in 1..=3 {
            for degree in 0..=4 {
                let doc = decode(&spz(version, 50, degree, 1, &[]), 1000, 1).unwrap();
                assert_eq!(splats(&doc), 50, "version {version}, degree {degree}");
                let a = &doc.json["meshes"][0]["primitives"][0]["attributes"];
                let has = |d: usize| a.get(format!("KHR_gaussian_splatting:SH_DEGREE_{d}_COEF_0")).is_some();
                let top = usize::from(degree.min(3));
                assert!(
                    (1..=3).all(|d| has(d) == (d <= top)),
                    "version {version}, degree {degree}"
                );
                assert_eq!(doc.json["asset"]["extras"]["source"]["antialiased"], true);
                let info = doc.source_info();
                assert_eq!(info.version, Some(version.to_string()));
                assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
                assert_eq!(info.meters_per_unit, Fact::Unknown);
                assert!(!info.converted.up_axis && !info.converted.handedness);
                doc.to_glb().unwrap();
                let m = doc.flatten(Pose::Rest).unwrap().unwrap();
                assert_eq!(m.points(), 50);
            }
        }
    }

    #[test]
    fn unpacking_as_another_thread_inflates_reads_the_same() {
        // Several chunks' worth, cut in the middle of every section once.
        let file = spz(3, 20_000, 3, 3, &[]);
        let one = decode(&file, 100_000, 1).unwrap().to_glb().unwrap();
        for threads in [2, 4] {
            assert!(decode(&file, 100_000, threads).unwrap().to_glb().unwrap() == one);
        }
        // Where the thread can't be started, this one inflates as well, from the start.
        crate::keep::NO_THREADS.set(true);
        let unthreaded = decode(&file, 100_000, 2).unwrap().to_glb().unwrap();
        crate::keep::NO_THREADS.set(false);
        assert!(unthreaded == one, "a load without threads differs");
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(&file[..]).read_to_end(&mut raw).unwrap();
        for cut in (0..raw.len()).step_by(raw.len() / 17) {
            let short = gzip(&raw[..cut]);
            let a = decode(&short, 100_000, 1).map(|d| d.to_glb().unwrap());
            let b = decode(&short, 100_000, 2).map(|d| d.to_glb().unwrap());
            assert_eq!(a.ok(), b.ok(), "cut at {cut}");
        }
    }

    #[test]
    fn values_decode_as_the_reference_does() {
        // Smallest three: w largest; x = +1/sqrt2 * 511/511, y = 0, z = -0.5/sqrt2 (about).
        let packed = (3u32 << 30) | (511 << 20) | (1 << 9) | 255;
        let q = smallest_three(packed);
        assert!((q[0] - FRAC_1_SQRT_2).abs() < 1e-3, "{q:?}");
        assert!(q[1].abs() < 1e-6 && q[2] < 0.0 && q[3] > 0.0, "{q:?}");
        let len: f32 = q.iter().map(|c| c * c).sum();
        assert!((len - 1.0).abs() < 1e-5);
        assert_eq!(f16([0x00, 0x3c]), 1.0);
        assert_eq!(f16([0x00, 0xc0]), -2.0);
        assert_eq!(f16([0x01, 0x00]), 2f32.powi(-24));
        assert_eq!(f16([0xff, 0x7b]), 65504.0);
        assert!(f16([0x01, 0x7c]).is_nan());
    }

    #[test]
    fn coordinate_systems_flip_and_turn() {
        assert_eq!(frame(0), (Fact::Assumed(Axis::PosY), Fact::Assumed(Handedness::Right)));
        // RUB, RDF and LUB; RFU, RBU and LFD.
        assert_eq!(
            frame(4),
            (Fact::Declared(Axis::PosY), Fact::Declared(Handedness::Right))
        );
        assert_eq!(
            frame(6),
            (Fact::Declared(Axis::NegY), Fact::Declared(Handedness::Right))
        );
        assert_eq!(frame(3), (Fact::Declared(Axis::PosY), Fact::Declared(Handedness::Left)));
        assert_eq!(
            frame(12),
            (Fact::Declared(Axis::PosZ), Fact::Declared(Handedness::Right))
        );
        assert_eq!(
            frame(16),
            (Fact::Declared(Axis::PosZ), Fact::Declared(Handedness::Left))
        );
        assert_eq!(
            frame(9),
            (Fact::Declared(Axis::NegZ), Fact::Declared(Handedness::Right))
        );
        assert_eq!(axes(0), ([1.0, 1.0, 1.0], None));
        assert_eq!(axes(4), ([1.0, 1.0, 1.0], None));
        assert_eq!(axes(6), ([1.0, -1.0, -1.0], None));
        assert_eq!(axes(7), ([-1.0, 1.0, -1.0], None));
        assert_eq!(axes(1), ([-1.0, -1.0, 1.0], None));
        // Right, front, up: z up is y up after the turn; front is -z.
        let (flip, turn) = axes(12);
        assert_eq!(flip, [1.0, -1.0, -1.0]);
        assert!(turn.is_some());
        assert_eq!(axes(16).0, [1.0, 1.0, -1.0]);
        assert_eq!(axes(9).0, [-1.0, -1.0, 1.0]);
        let mut records = Vec::new();
        records.extend(0xadbe_0002u32.to_le_bytes());
        records.extend(3u32.to_le_bytes());
        records.extend([1, 2, 3]);
        records.extend(COORDINATES.to_le_bytes());
        records.extend(4u32.to_le_bytes());
        records.extend(12u32.to_le_bytes());
        assert_eq!(coordinates(&records), 12);
        assert_eq!(coordinates(&records[..records.len() - 1]), 0);
        let doc = decode(&spz(3, 4, 0, 2, &records), 100, 1).unwrap();
        let node = &doc.json["nodes"][0];
        assert!(node["rotation"].is_array(), "{node}");
    }

    #[test]
    fn hostile_headers_are_refused_before_anything_is_allocated() {
        let mut raw = Vec::new();
        raw.extend(MAGIC.to_le_bytes());
        raw.extend(2u32.to_le_bytes());
        raw.extend(u32::MAX.to_le_bytes());
        raw.extend([3, 12, 0, 0]);
        let e = decode(&gzip(&raw), usize::MAX >> 8, 1).unwrap_err().to_string();
        assert!(e.contains("is cut short"), "{e}");
        let e = decode(&gzip(&raw), 1000, 1).unwrap_err().to_string();
        assert!(e.contains("max_points"), "{e}");
        raw[12] = 7;
        assert!(decode(&gzip(&raw), usize::MAX, 1).is_err());
        assert!(
            decode(b"NGSP\x04\0\0\0", 10, 1)
                .unwrap_err()
                .to_string()
                .contains("version 4")
        );
        assert!(decode(b"", 10, 1).is_err());
        let empty = decode(&spz(2, 0, 0, 0, &[]), 10, 1).unwrap();
        assert!(empty.flatten(Pose::Rest).unwrap().is_none());
    }

    #[test]
    fn survives_damage() {
        let good = spz(3, 300, 2, 3, &[]);
        let step = (good.len() / 200).max(1);
        for cut in (0..good.len()).step_by(step) {
            let _ = decode(&good[..cut], 10_000, 1 + cut % 2).map(|d| d.flatten(Pose::Rest));
        }
        let mut rng = Pcg32::new(11);
        for _ in 0..300 {
            let mut bad = good.clone();
            for _ in 0..4 {
                let i = rng.below(bad.len() as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            let _ = decode(&bad, 10_000, 1).map(|d| d.flatten(Pose::Rest));
        }
        // The raw stream damaged, then compressed, so the damage reaches the decoder.
        let raw = {
            use std::io::Read as _;
            let mut v = Vec::new();
            flate2::read::GzDecoder::new(&good[..]).read_to_end(&mut v).unwrap();
            v
        };
        for _ in 0..300 {
            let mut bad = raw.clone();
            for _ in 0..4 {
                let i = rng.below(bad.len() as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            let cut = rng.below(bad.len() as u32 + 1) as usize;
            if let Ok(doc) = decode(&gzip(&bad[..cut]), 10_000, 1) {
                doc.to_glb().unwrap();
                let _ = doc.flatten(Pose::Rest);
            }
        }
    }
}
