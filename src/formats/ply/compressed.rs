//! `SuperSplat`'s compressed PLY (`.compressed.ply`): gaussian splats quantized in chunks of 256.
//!
//! A `chunk` element gives each chunk's bounds: of position, of scale (a logarithm) and, in
//! files written since 2024, of color. A `vertex` element packs each splat in four 32-bit
//! words: position and scale as 11, 10 and 11 bits between the chunk's bounds; rotation as the
//! three smaller of its four components (w, x, y, z), 10 bits each, under 2 bits naming the
//! largest; and color and opacity as 8 bits each. An optional `sh` element holds the higher
//! spherical harmonics, a byte a coefficient and a color channel at a time, as `f_rest_*` are.
//! The layout and its decoding are those of `PlayCanvas`'s splat-transform (its reader and
//! writer of the format) and engine.
//!
//! The splats are in a 3D gaussian splatting PLY's axes, and are flipped to glTF's as those
//! are. The count is checked against [`Keep::max_points`](crate::document::Keep::max_points) and
//! against the bytes the file holds before anything is allocated: a file shorter than its
//! header's counts is refused.

use std::f64::consts::SQRT_2;

use serde_json::{Map, json};

use super::{Element, Format, Header, Ty, ordered, writer};
use crate::document::Document;
use crate::gaussian::{REST, Splats, degree_of, rotation};
use crate::num::f32_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// The splats a chunk holds.
const CHUNK: usize = 256;
/// A chunk's bounds every file has: of position, then of scale.
const BOUNDS: [&str; 12] = [
    "min_x",
    "min_y",
    "min_z",
    "max_x",
    "max_y",
    "max_z",
    "min_scale_x",
    "min_scale_y",
    "min_scale_z",
    "max_scale_x",
    "max_scale_y",
    "max_scale_z",
];
/// A chunk's bounds of color, which older files leave out: their colors run from 0 to 1.
const COLOR_BOUNDS: [&str; 6] = ["min_r", "min_g", "min_b", "max_r", "max_g", "max_b"];
/// A splat's packed words.
const PACKED: [&str; 4] = ["packed_position", "packed_rotation", "packed_scale", "packed_color"];
/// The zeroth spherical-harmonic basis function's constant, in the precision the format's own
/// tools decode with.
const SH_C0: f64 = 0.282_094_791_773_878_14;

/// Whether `elements` are a compressed PLY's: a `chunk` element with the bounds and a `vertex`
/// element with the packed words.
pub(super) fn is_compressed(elements: &[Element]) -> bool {
    let has = |name: &str, props: &[&str], ty: Ty| {
        elements
            .iter()
            .any(|e| e.name == name && props.iter().all(|p| slot(e, p, ty).is_some()))
    };
    has("chunk", &BOUNDS, Ty::F32) && has("vertex", &PACKED, Ty::U32)
}

/// Where the scalar property `name` of type `ty` starts in one of `el`'s items.
fn slot(el: &Element, name: &str, ty: Ty) -> Option<usize> {
    let k = el
        .props
        .iter()
        .position(|p| p.name == name && p.ty == ty && p.count.is_none())?;
    Some(el.props[..k].iter().map(|p| p.ty.size()).sum())
}

/// An element's items in the file: an item's size, and all of their bytes.
struct Items<'a> {
    el: &'a Element,
    size: usize,
    bytes: &'a [u8],
}

/// The value `bits` wide at the low end of `v`, from 0 to 1.
fn unorm(v: u32, bits: u32) -> f64 {
    let top = (1u32 << bits) - 1;
    f64::from(v & top) / f64::from(top)
}

/// The value `t` of the way from `a` to `b`, as the format's tools work it out.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

/// The three values 11, 10 and 11 bits wide in `v`, each from 0 to 1.
fn unpack_11_10_11(v: u32) -> [f64; 3] {
    [unorm(v >> 21, 11), unorm(v >> 11, 10), unorm(v, 11)]
}

/// The four values 8 bits wide in `v`, each from 0 to 1.
fn unpack_8888(v: u32) -> [f64; 4] {
    [unorm(v >> 24, 8), unorm(v >> 16, 8), unorm(v >> 8, 8), unorm(v, 8)]
}

/// A rotation (x, y, z, w) from its packed word: the top 2 bits name the largest of w, x, y
/// and z, and the others follow in that order, 10 bits each, from -1/sqrt(2) to 1/sqrt(2).
fn unpack_rotation(v: u32) -> [f32; 4] {
    let part = |shift: u32| (unorm(v >> shift, 10) - 0.5) * SQRT_2;
    let (a, b, c) = (part(20), part(10), part(0));
    let m = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();
    let [w, x, y, z] = match v >> 30 {
        0 => [m, a, b, c],
        1 => [a, m, b, c],
        2 => [a, b, m, c],
        _ => [a, b, c, m],
    };
    rotation(f32_of(x), f32_of(y), f32_of(z), f32_of(w))
}

/// A higher spherical-harmonic coefficient from its byte: the middle of the byte's step of the
/// range from -4 to 4, and the ends of the range for 0 and 255.
fn unpack_sh(v: u8) -> f32 {
    let n = match v {
        0 => 0.0,
        255 => 1.0,
        v => (f32::from(v) + 0.5) / 256.0,
    };
    (n - 0.5) * 8.0
}

/// The splats of a compressed PLY whose header is `head` and `elements`, from `data`, the
/// bytes past the header; up to `max` of them.
pub(super) fn load(data: &[u8], head: &Header, elements: &[Element], max: usize) -> Result<Document> {
    ensure!(
        head.format != Format::Ascii,
        Unsupported,
        "a compressed PLY written as text isn't supported"
    );
    let be = head.format == Format::Be;
    // Where each element's items are, when the file holds them all.
    let mut start = 0usize;
    let mut laid = Vec::with_capacity(elements.len());
    for el in elements {
        ensure!(
            el.props.iter().all(|p| p.count.is_none()),
            Corrupt,
            "compressed PLY element {} has a list property",
            el.name
        );
        let size: usize = el.props.iter().map(|p| p.ty.size()).sum();
        let end = el
            .count
            .checked_mul(size)
            .and_then(|len| start.checked_add(len))
            .context(ErrorKind::Corrupt, "compressed PLY counts are past any file's size")?;
        laid.push((el, size, start, end));
        start = end;
    }
    let find = |name: &str| -> Result<Option<Items<'_>>> {
        let Some(&(el, size, start, end)) = laid.iter().find(|(el, ..)| el.name == name) else {
            return Ok(None);
        };
        let bytes = data.get(start..end).with_context(ErrorKind::Truncated, || {
            format!(
                "compressed PLY data is cut short: its {name} items end at byte {end}, the file holds {}",
                data.len()
            )
        })?;
        Ok(Some(Items { el, size, bytes }))
    };
    let n = elements.iter().find(|e| e.name == "vertex").map_or(0, |e| e.count);
    ensure!(
        n <= max,
        LimitExceeded,
        "{n} gaussian splats, more than the {max} allowed (Keep::max_points)"
    );
    let (Some(chunks), Some(vertices)) = (find("chunk")?, find("vertex")?) else {
        bail!(Corrupt, "a compressed PLY needs chunk and vertex elements");
    };
    ensure!(
        chunks.el.count >= n.div_ceil(CHUNK),
        Corrupt,
        "{n} compressed PLY splats need {} chunks, the file has {}",
        n.div_ceil(CHUNK),
        chunks.el.count
    );
    let missing = || crate::Error::new(ErrorKind::Corrupt, "a compressed PLY property is missing");
    let mut bounds = [None; 18];
    for (k, name) in BOUNDS.iter().chain(&COLOR_BOUNDS).enumerate() {
        bounds[k] = slot(chunks.el, name, Ty::F32);
    }
    ensure!(
        bounds[..12].iter().all(Option::is_some),
        Corrupt,
        "a compressed PLY property is missing"
    );
    let colored = bounds[12..].iter().all(Option::is_some);
    let mut packed = [0usize; 4];
    for (k, name) in PACKED.iter().enumerate() {
        packed[k] = slot(vertices.el, name, Ty::U32).ok_or_else(missing)?;
    }
    // The higher harmonics: `f_rest_0` on, a color channel at a time, in whole degrees.
    let mut skipped = Vec::new();
    let sh = find("sh")?.filter(|sh| {
        let fits = sh.el.count >= n;
        if !fits {
            skipped.push("sh");
        }
        fits
    });
    let rest: Vec<usize> = sh.as_ref().map_or_else(Vec::new, |sh| {
        (0..)
            .map_while(|k| slot(sh.el, &format!("f_rest_{k}"), Ty::U8))
            .collect()
    });
    let per = if rest.len().is_multiple_of(3) {
        rest.len() / 3
    } else {
        0
    };
    let used = REST[degree_of(per)];

    let mut s = Splats::with_capacity(n, used)?;
    let float = |b: &[u8], at: usize| f64::from(f32::from_le_bytes(ordered(&b[at..], be)));
    let word = |b: &[u8], at: usize| u32::from_le_bytes(ordered(&b[at..], be));
    let splats = &vertices.bytes[..n * vertices.size];
    for (chunk, splats) in chunks
        .bytes
        .chunks_exact(chunks.size.max(1))
        .zip(splats.chunks(CHUNK * vertices.size.max(1)))
    {
        // Colors without bounds are the packed values themselves.
        let mut b = [0.0f64; 18];
        for (k, at) in bounds.iter().enumerate() {
            b[k] = match at {
                Some(at) if k < 12 || colored => float(chunk, *at),
                _ if k < 15 => 0.0,
                _ => 1.0,
            };
        }
        for splat in splats.chunks_exact(vertices.size) {
            let [position, turn, scale, color] = packed.map(|at| word(splat, at));
            let p = unpack_11_10_11(position);
            s.positions
                .push(std::array::from_fn(|k| f32_of(lerp(b[k], b[3 + k], p[k]))));
            s.rotations.push(unpack_rotation(turn));
            let e = unpack_11_10_11(scale);
            s.scales
                .push(std::array::from_fn(|k| f32_of(lerp(b[6 + k], b[9 + k], e[k])).exp()));
            let c = unpack_8888(color);
            s.dc.push(std::array::from_fn(|k| {
                f32_of((lerp(b[12 + k], b[15 + k], c[k]) - 0.5) / SH_C0)
            }));
            // The tools store the logit of this and take its sigmoid: the value itself.
            s.opacities.push(f32_of(c[3]));
        }
    }
    if let Some(sh) = &sh {
        for (j, col) in s.sh.iter_mut().enumerate() {
            let at = [rest[j], rest[per + j], rest[2 * per + j]];
            col.extend(
                sh.bytes
                    .chunks_exact(sh.size.max(1))
                    .take(n)
                    .map(|row| at.map(|a| unpack_sh(row[a]))),
            );
        }
    }
    s.flip([1.0, -1.0, -1.0]);

    let mut source = Map::new();
    source.insert(
        "encoding".into(),
        json!(if be {
            "binary_big_endian"
        } else {
            "binary_little_endian"
        }),
    );
    source.insert("compressed".into(), json!(true));
    if !head.obj_info.is_empty() {
        source.insert("objInfo".into(), json!(head.obj_info));
    }
    let others = elements
        .iter()
        .map(|e| e.name.as_str())
        .filter(|name| !matches!(*name, "chunk" | "vertex" | "sh"));
    let skipped: Vec<&str> = others.chain(skipped).collect();
    if !skipped.is_empty() {
        source.insert("skipped".into(), json!(skipped));
    }
    // Right, down and forward by convention, which the file doesn't state: turned to glTF's.
    let mut info = SourceInfo {
        format: Some("PLY".into()),
        version: Some(head.version.clone()).filter(|v| !v.is_empty()),
        up_axis: Fact::Assumed(Axis::NegY),
        handedness: Fact::Assumed(Handedness::Right),
        tool: head.comments.iter().find_map(|c| writer(c)),
        comments: head.comments.clone(),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    Ok(s.document(&info, source, None, Map::new()))
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test files are small"
)]
mod tests {
    use std::f32::consts::FRAC_1_SQRT_2;

    use super::*;
    use crate::document::{Keep, Pose};
    use crate::formats::Input;
    use crate::test_rng::Pcg32;

    fn load(bytes: &[u8]) -> Result<Document> {
        super::super::ply(
            Input::memory(bytes, "t.compressed.ply"),
            &crate::assets::Memory::new(),
            Keep::ALL,
        )
    }

    fn floats(doc: &Document, name: &str) -> Vec<f32> {
        let a = &doc.json()["meshes"][0]["primitives"][0]["attributes"][name];
        doc.read_floats(usize::try_from(a.as_u64().unwrap()).unwrap())
            .unwrap()
            .1
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} != {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{got:?} != {want:?}");
        }
    }

    /// The header of a file of `chunks` chunks (with color bounds when `colored`) and `n`
    /// splats, with an `sh` element of `rest` columns when there are any.
    fn header(chunks: usize, colored: bool, n: usize, rest: usize) -> Vec<u8> {
        let mut lines = vec![
            "ply".to_owned(),
            "format binary_little_endian 1.0".to_owned(),
            "comment Generated by SuperSplat 2.7.1".to_owned(),
            format!("element chunk {chunks}"),
        ];
        let bounds = BOUNDS.iter().chain(if colored { &COLOR_BOUNDS[..] } else { &[] });
        lines.extend(bounds.map(|p| format!("property float {p}")));
        lines.push(format!("element vertex {n}"));
        lines.extend(PACKED.iter().map(|p| format!("property uint {p}")));
        if rest > 0 {
            lines.push(format!("element sh {n}"));
            lines.extend((0..rest).map(|k| format!("property uchar f_rest_{k}")));
        }
        lines.push("end_header\n".to_owned());
        lines.join("\n").into_bytes()
    }

    fn le<T: Copy, const N: usize>(values: &[T], bytes: fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|&v| bytes(v)).collect()
    }

    /// The bounds the tests' chunks have: positions from (0, 0, 0) to (10, 20, 30), scales from
    /// ln 1 to ln 4, colors from 0.25 to 0.75.
    const CHUNK_BOUNDS: [f32; 18] = [
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
        0.0,
        0.0,
        0.0,
        1.386_294_4,
        1.386_294_4,
        1.386_294_4,
        0.25,
        0.25,
        0.25,
        0.75,
        0.75,
        0.75,
    ];

    /// A file of one chunk and `n` splats of random words, with `rest` columns of harmonics.
    fn file(n: usize, rest: usize) -> Vec<u8> {
        let mut rng = Pcg32::new(5);
        let mut out = header(n.div_ceil(CHUNK), true, n, rest);
        for _ in 0..n.div_ceil(CHUNK) {
            out.extend(le(&CHUNK_BOUNDS, f32::to_le_bytes));
        }
        for _ in 0..n * 4 {
            out.extend(rng.next_u32().to_le_bytes());
        }
        for _ in 0..n * rest {
            out.push(rng.below(256) as u8);
        }
        out
    }

    #[test]
    fn words_unpack_by_their_bit_fields() {
        assert_eq!(unpack_11_10_11(0xFFE0_0000), [1.0, 0.0, 0.0]);
        assert_eq!(unpack_11_10_11(0x001F_F800), [0.0, 1.0, 0.0]);
        assert_eq!(unpack_11_10_11(0x0000_07FF), [0.0, 0.0, 1.0]);
        let v = unpack_11_10_11(1023 << 21 | 511 << 11 | 1);
        assert_eq!(v, [1023.0 / 2047.0, 511.0 / 1023.0, 1.0 / 2047.0]);
        assert_eq!(unpack_8888(0x80FF_0040), [128.0 / 255.0, 1.0, 0.0, 64.0 / 255.0]);
        assert_eq!(unorm(0xFFFF_FFFF, 10), 1.0);
        assert_eq!(unpack_sh(0), -4.0);
        assert_eq!(unpack_sh(255), 4.0);
        assert_eq!(unpack_sh(128), 0.015_625);
        assert_eq!(unpack_sh(1), (1.5 / 256.0 - 0.5) * 8.0);
    }

    #[test]
    fn rotations_unpack_as_the_smallest_three_of_w_x_y_z() {
        // Each stored part at its middle, 512 of 1023: (512 / 1023 - 0.5) sqrt(2) = 0.000691.
        let mid = 0.000_691_2;
        let one = (1.0f32 - 3.0 * mid * mid).sqrt();
        // The largest is w; x, y and z follow.
        close(&unpack_rotation(512 << 20 | 512 << 10 | 512), &[mid, mid, mid, one]);
        // The largest is x: w, y and z follow. Then y: w, x, z. Then z: w, x, y.
        let (top, low) = (1023, 0);
        let q = unpack_rotation(1 << 30 | top << 20 | low << 10 | 512);
        close(&q, &[0.0, -FRAC_1_SQRT_2, mid, FRAC_1_SQRT_2]);
        let q = unpack_rotation(2 << 30 | top << 20 | low << 10 | 512);
        close(&q, &[-FRAC_1_SQRT_2, 0.0, mid, FRAC_1_SQRT_2]);
        let q = unpack_rotation(3 << 30 | top << 20 | low << 10 | 512);
        close(&q, &[-FRAC_1_SQRT_2, mid, 0.0, FRAC_1_SQRT_2]);
        // Parts whose squares pass 1 leave nothing for the largest.
        let q = unpack_rotation(top << 20 | top << 10 | top);
        close(&q, &[0.577_350_3, 0.577_350_3, 0.577_350_3, 0.0]);
    }

    #[test]
    fn splats_decode_between_their_chunks_bounds() {
        let mut bytes = header(1, true, 2, 9);
        bytes.extend(le(&CHUNK_BOUNDS, f32::to_le_bytes));
        // Splat 0: the bounds' ends; the identity; scale ln 4, ln 1, ln 1; color at both ends.
        let identity = 512 << 20 | 512 << 10 | 512;
        bytes.extend(le(
            &[0xFFE0_0000, identity, 0xFFE0_0000, 0xFF00_80FFu32],
            u32::to_le_bytes,
        ));
        // Splat 1: the middle of y (511 of 1023), a third of the way in x and z (682 of 2047).
        let third = 0x2AA << 21 | 0x1FF << 11 | 0x2AA;
        bytes.extend(le(
            &[third, 2 << 30 | identity, third, 0x0000_0033u32],
            u32::to_le_bytes,
        ));
        // The harmonics, a channel at a time: red's three coefficients, green's, blue's.
        bytes.extend([0, 255, 128, 64, 64, 64, 192, 192, 192]);
        bytes.extend([128; 9]);
        let doc = load(&bytes).unwrap();
        assert_eq!(doc.facts["splats"], 2);
        assert_eq!(doc.facts["points"], 2);
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("PLY"));
        assert_eq!(info.up_axis, crate::source::Fact::Assumed(crate::source::Axis::NegY));
        assert!(info.converted.up_axis && !info.converted.handedness);
        assert_eq!(info.tool.as_deref(), Some("SuperSplat 2.7.1"));
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!(source["compressed"], true);
        assert_eq!(source["shDegree"], 1);
        // Right, down, forward to right, up, back.
        let third_of = |top: f32| top * 682.0 / 2047.0;
        close(
            &floats(&doc, "POSITION"),
            &[10.0, 0.0, 0.0, third_of(10.0), -20.0 * 511.0 / 1023.0, -third_of(30.0)],
        );
        let ln4 = 1.386_294_4f32;
        close(
            &floats(&doc, "KHR_gaussian_splatting:SCALE"),
            &[
                4.0,
                1.0,
                1.0,
                third_of(ln4).exp(),
                (ln4 * 511.0 / 1023.0).exp(),
                third_of(ln4).exp(),
            ],
        );
        close(&floats(&doc, crate::gaussian::OPACITY), &[1.0, 0.2]);
        // Colors from 0.25 to 0.75, as coefficients: (c - 0.5) / 0.2820948.
        let c = |v: f32| (v - 0.5) / 0.282_094_8;
        close(
            &floats(&doc, crate::gaussian::SH0),
            &[
                c(0.75),
                c(0.25),
                c(0.25 + 0.5 * 128.0 / 255.0),
                c(0.25),
                c(0.25),
                c(0.25),
            ],
        );
        let q = floats(&doc, "KHR_gaussian_splatting:ROTATION");
        let mid = 0.000_691_2;
        close(&q[..4], &[mid, -mid, -mid, 1.0]);
        // Splat 1's largest is y: the flip turns its sign with z's.
        close(&q[4..], &[mid, -1.0, -mid, mid]);
        // Degree 1's coefficients are odd in y, z and x: the first two flip with the axes.
        let sh = |v: f32| ((v + 0.5) / 256.0 - 0.5) * 8.0;
        close(
            &floats(&doc, "KHR_gaussian_splatting:SH_DEGREE_1_COEF_0"),
            &[4.0, -sh(64.0), -sh(192.0), -sh(128.0), -sh(128.0), -sh(128.0)],
        );
        close(
            &floats(&doc, "KHR_gaussian_splatting:SH_DEGREE_1_COEF_1")[..3],
            &[-4.0, -sh(64.0), -sh(192.0)],
        );
        close(
            &floats(&doc, "KHR_gaussian_splatting:SH_DEGREE_1_COEF_2")[..3],
            &[sh(128.0), sh(64.0), sh(192.0)],
        );
        doc.to_glb().unwrap();
        assert_eq!(doc.flatten(Pose::Rest).unwrap().unwrap().points(), 2);
    }

    #[test]
    fn chunks_without_color_bounds_hold_colors_from_0_to_1() {
        let mut bytes = header(1, false, 1, 0);
        bytes.extend(le(&CHUNK_BOUNDS[..12], f32::to_le_bytes));
        bytes.extend(le(&[0u32, 0, 0, 0xFF00_33FF], u32::to_le_bytes));
        let doc = load(&bytes).unwrap();
        let c = |v: f32| (v - 0.5) / 0.282_094_8;
        close(&floats(&doc, crate::gaussian::SH0), &[c(1.0), c(0.0), c(0.2)]);
        assert_eq!(doc.json()["asset"]["extras"]["source"]["shDegree"], 0);
    }

    #[test]
    fn big_endian_files_and_other_elements_read_the_same() {
        let good = file(300, 0);
        let want = floats(&load(&good).unwrap(), "POSITION");
        // The same words, each turned around.
        let at = good.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let mut be = good[..at].to_vec();
        for word in good[at..].chunks_exact(4) {
            be.extend(word.iter().rev());
        }
        let text = String::from_utf8(be[..at].to_vec()).unwrap();
        let text = text.replace("binary_little_endian", "binary_big_endian");
        let mut be_file = text.into_bytes();
        be_file.extend(&be[at..]);
        let doc = load(&be_file).unwrap();
        assert_eq!(floats(&doc, "POSITION"), want);
        // An element the reader has no use for, before the others, is skipped by its size.
        let text = String::from_utf8(good[..at].to_vec()).unwrap();
        let text = text.replace(
            "element chunk",
            "element extra 3\nproperty double a\nproperty uchar b\nelement chunk",
        );
        let mut more = text.into_bytes();
        more.extend([7u8; 27]);
        more.extend(&good[at..]);
        let doc = load(&more).unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["skipped"][0], "extra");
        assert_eq!(floats(&doc, "POSITION"), want);
    }

    #[test]
    fn counts_are_checked_before_anything_is_allocated() {
        let good = file(300, 9);
        let swap = |from: &str, to: &str| {
            let at = good.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
            let text = String::from_utf8(good[..at].to_vec()).unwrap();
            assert!(text.contains(from), "{from}");
            let mut bad = text.replace(from, to).into_bytes();
            bad.extend(&good[at..]);
            bad
        };
        let fails = |bytes: &[u8], kind: ErrorKind, text: &str| {
            let e = load(bytes).unwrap_err();
            assert_eq!(e.kind(), kind, "{e}");
            assert!(e.to_string().contains(text), "{e}");
        };
        fails(
            &swap("element vertex 300", "element vertex 4000000000"),
            ErrorKind::LimitExceeded,
            "max_points",
        );
        // Within the limit, and far past the file's end.
        fails(
            &swap("element vertex 300", "element vertex 15000000"),
            ErrorKind::Truncated,
            "cut short",
        );
        fails(
            &swap("element sh 300", "element sh 15000000"),
            ErrorKind::Truncated,
            "cut short",
        );
        fails(
            &swap("element chunk 2", "element chunk 18446744073709551615"),
            ErrorKind::Corrupt,
            "past any file's size",
        );
        fails(
            &swap("element chunk 2", "element chunk 1"),
            ErrorKind::Corrupt,
            "need 2 chunks",
        );
        fails(
            &swap("property uchar f_rest_0", "property list uchar uchar f_rest_0"),
            ErrorKind::Corrupt,
            "list property",
        );
        fails(
            &swap("binary_little_endian", "ascii"),
            ErrorKind::Unsupported,
            "written as text",
        );
        let max = Keep::ALL.with_max_points(299);
        let e = super::super::ply(Input::memory(&good, "t.ply"), &crate::assets::Memory::new(), max).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::LimitExceeded);
        // Harmonics for fewer splats than there are say nothing of the rest: none are read.
        let mut short = swap("element sh 300", "element sh 299");
        short.truncate(short.len() - 9);
        let doc = load(&short).unwrap();
        assert_eq!(doc.facts["splats"], 300);
        let source = &doc.json()["asset"]["extras"]["source"];
        assert_eq!((&source["shDegree"], &source["skipped"][0]), (&json!(0), &json!("sh")));
        // Seven columns are no whole degree.
        let doc = load(&file(10, 7)).unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["shDegree"], 0);
        // No splats at all.
        let doc = load(&file(0, 0)).unwrap();
        assert!(doc.flatten(Pose::Rest).unwrap().is_none());
    }

    #[test]
    fn damaged_files_never_panic() {
        let good = file(300, 24);
        let body = good.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        for cut in (0..good.len()).step_by(11) {
            match load(&good[..cut]) {
                Ok(doc) => {
                    // A header cut before its end may leave another PLY, of no vertices.
                    assert!(cut < body, "a file cut at {cut} read");
                    let _ = doc.flatten(Pose::Rest);
                }
                Err(e) if cut >= body => assert_eq!(e.kind(), ErrorKind::Truncated, "{cut}: {e}"),
                Err(_) => {}
            }
        }
        let mut rng = Pcg32::new(23);
        for _ in 0..300 {
            let mut bad = good.clone();
            for _ in 0..4 {
                let i = body + rng.below((good.len() - body) as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            // Whatever the words and bounds are, the splats that are left are valid glTF.
            let doc = load(&bad).unwrap();
            doc.to_glb().unwrap();
            let _ = doc.flatten(Pose::Rest);
        }
        for _ in 0..300 {
            let mut bad = good.clone();
            let i = rng.below(body as u32) as usize;
            bad[i] = rng.below(256) as u8;
            if let Ok(doc) = load(&bad) {
                let _ = doc.to_glb();
                let _ = doc.flatten(Pose::Rest);
            }
        }
        // Bounds that aren't numbers drop their chunk's splats and leave the others.
        let mut bad = good.clone();
        bad[body..body + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let doc = load(&bad).unwrap();
        assert_eq!(doc.facts["splats"], 44);
        doc.to_glb().unwrap();
    }
}
