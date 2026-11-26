//! `.splat`, the plain gaussian splat format: 32 bytes a splat and no header. Each splat is its
//! position and linear scale (three floats each), its base color and opacity (a byte each, the
//! color as the sRGB value the base spherical harmonics give) and its rotation (w, x, y, z, a
//! byte each from -1 at 0 to 1 at 256). The axes are right, down and forward, as 3D gaussian
//! splatting's PLY files have them, and are flipped to glTF's; the source info has -Y up and
//! right-handed, assumed, and no units.
//!
//! The format has no view-dependent color, so the splats have degree 0 alone. With no header,
//! all that tells a `.splat` file is its length: a file that is empty or isn't a whole number
//! of splats long is refused as not one.

use serde_json::Map;

use super::Input;
use crate::Result;
use crate::document::{Document, Keep};
use crate::gaussian::{SH_C0, Splats, rotation};
use crate::source::{Axis, Fact, Handedness, SourceInfo};

/// The bytes a splat takes.
const SIZE: usize = 32;

pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document> {
    decode(&input.read()?, keep.points())
}

/// The splats in `bytes`, up to `max` of them.
fn decode(bytes: &[u8], max: usize) -> Result<Document> {
    let (splats, rest) = bytes.as_chunks::<SIZE>();
    let n = splats.len();
    // The format has no header to know it by: all there is to check is that it holds whole splats.
    ensure!(
        n > 0 && rest.is_empty(),
        UnknownFormat,
        "not a .splat file: {} bytes aren't a whole number of {SIZE}-byte splats",
        bytes.len()
    );
    ensure!(
        n <= max,
        LimitExceeded,
        "{n} gaussian splats, more than the {max} allowed (Keep::max_points)"
    );
    let mut s = Splats::with_capacity(n, 0)?;
    let f = |b: &[u8; SIZE], k: usize| f32::from_le_bytes([b[4 * k], b[4 * k + 1], b[4 * k + 2], b[4 * k + 3]]);
    for b in splats {
        s.positions.push([f(b, 0), f(b, 1), f(b, 2)]);
        s.scales.push([f(b, 3), f(b, 4), f(b, 5)]);
        s.dc.push([b[24], b[25], b[26]].map(|c| (f32::from(c) / 255.0 - 0.5) / SH_C0));
        s.opacities.push(f32::from(b[27]) / 255.0);
        let [w, x, y, z] = [b[28], b[29], b[30], b[31]].map(|c| (f32::from(c) - 128.0) / 128.0);
        s.rotations.push(rotation(x, y, z, w));
    }
    s.flip([1.0, -1.0, -1.0]);
    let source = Map::new();
    // Right, down and forward: the convention, which the file doesn't state.
    let mut info = SourceInfo {
        format: Some("splat".into()),
        up_axis: Fact::Assumed(Axis::NegY),
        handedness: Fact::Assumed(Handedness::Right),
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
    use super::*;
    use crate::document::Pose;
    use crate::test_rng::Pcg32;

    /// A `.splat` file of `n` splats with random bytes for color and rotation and small floats
    /// for position and scale.
    fn file(n: usize) -> Vec<u8> {
        let mut rng = Pcg32::new(9);
        let mut out = Vec::new();
        for _ in 0..n {
            for _ in 0..3 {
                out.extend((rng.f32() * 2.0 - 1.0).to_le_bytes());
            }
            for _ in 0..3 {
                out.extend((rng.f32() * 0.1).to_le_bytes());
            }
            for _ in 0..8 {
                out.push(rng.below(256) as u8);
            }
        }
        out
    }

    #[test]
    fn splats_read_with_their_color() {
        let mut bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 0.5, 0.25, 0.125] {
            bytes.extend(v.to_le_bytes());
        }
        bytes.extend([255, 128, 0, 64, 128, 128, 128, 255]);
        let doc = decode(&bytes, 10).unwrap();
        assert_eq!(doc.facts["splats"], 1);
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("splat"));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::NegY));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert!(info.converted.up_axis && !info.converted.handedness);
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        let p = &m.prims[0];
        assert_eq!(p.positions, [[1.0, -2.0, -3.0]]);
        let c = p.colors[0];
        assert!((c[0] - 1.0).abs() < 1e-3 && c[2].abs() < 1e-3, "{c:?}");
        assert!(
            (c[1] - crate::mesh::srgb_to_linear(128.0 / 255.0)).abs() < 1e-3,
            "{c:?}"
        );
        assert!((c[3] - 64.0 / 255.0).abs() < 1e-3, "{c:?}");
        doc.to_glb().unwrap();
    }

    #[test]
    fn counts_past_the_limit_are_refused() {
        let e = decode(&file(20), 10).unwrap_err().to_string();
        assert!(e.contains("max_points"), "{e}");
    }

    #[test]
    fn a_length_that_is_not_whole_splats_is_not_a_splat_file() {
        for bytes in [
            &[][..],
            &[0; 31],
            &[0; 33],
            b"just some text, not thirty-two bytes a row",
        ] {
            let e = decode(bytes, 10).unwrap_err();
            assert_eq!(e.kind(), crate::ErrorKind::UnknownFormat, "{}", bytes.len());
        }
        assert_eq!(decode(&[0; 64], 10).unwrap().facts["splats"], 2);
    }

    #[test]
    fn survives_damage() {
        let good = file(200);
        let step = (good.len() / 200).max(1);
        for cut in (0..good.len()).step_by(step) {
            match decode(&good[..cut], 1000) {
                Ok(doc) => {
                    assert!(cut > 0 && cut % SIZE == 0);
                    assert_eq!(doc.facts["splats"], cut / SIZE);
                    let _ = doc.flatten(Pose::Rest);
                }
                Err(_) => assert!(cut == 0 || cut % SIZE != 0),
            }
        }
        let mut rng = Pcg32::new(11);
        for _ in 0..300 {
            let mut bad = good.clone();
            for _ in 0..4 {
                let i = rng.below(bad.len() as u32) as usize;
                bad[i] ^= 1 << rng.below(8);
            }
            let doc = decode(&bad, 1000).unwrap();
            doc.to_glb().unwrap();
            let _ = doc.flatten(Pose::Rest);
        }
        // Broken floats: positions drop their splat, other values are made 0.
        let mut bad = good;
        bad[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        bad[SIZE + 12..SIZE + 16].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let doc = decode(&bad, 1000).unwrap();
        assert_eq!(doc.facts["splats"], 199);
        doc.to_glb().unwrap();
    }
}
