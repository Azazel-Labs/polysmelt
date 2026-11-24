//! Gaussian splats, as `KHR_gaussian_splatting` stores them: a point primitive whose points carry
//! each splat's rotation, scale, opacity and spherical-harmonic color beside its position.
//!
//! The splat readers gather splats in a [`Splats`] and write them with [`Splats::primitive`]. It
//! also gives the points the base color the extension suggests as a fallback (`COLOR_0`), so a
//! flat mesh shows each splat as a colored point. [`fallback`] works that color out for a splat
//! primitive that comes without one.

use std::sync::OnceLock;

use serde_json::{Map, Value, json};

use crate::document::{Builder, Document, Part};
use crate::mesh::srgb_to_linear;
use crate::source::SourceInfo;
use crate::{Context as _, Result};

/// The extension's name, which also prefixes its attributes.
pub(crate) const EXTENSION: &str = "KHR_gaussian_splatting";
/// Each splat's opacity, from 0 to 1.
pub(crate) const OPACITY: &str = "KHR_gaussian_splatting:OPACITY";
/// Each splat's base color: its degree-0 spherical-harmonic coefficients.
pub(crate) const SH0: &str = "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0";
const ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
const SCALE: &str = "KHR_gaussian_splatting:SCALE";

/// The zeroth spherical-harmonic basis function's constant, `1 / (2 sqrt(pi))`.
pub(crate) const SH_C0: f32 = 0.282_094_8;

/// How many coefficients a color channel has beyond the base one, for degrees 0 to 3.
#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
pub(crate) const REST: [usize; 4] = [0, 3, 8, 15];

/// The highest spherical-harmonic degree whose coefficients all fit in `rest` coefficients past
/// the base one.
#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
pub(crate) fn degree_of(rest: usize) -> usize {
    REST.iter().rposition(|&r| r <= rest).unwrap_or(0)
}

/// Gaussian splats, a column per property.
#[derive(Default)]
#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
pub(crate) struct Splats {
    /// Each splat's center.
    pub positions: Vec<[f32; 3]>,
    /// Each splat's orientation, a unit quaternion (x, y, z, w).
    pub rotations: Vec<[f32; 4]>,
    /// Each splat's extent along its own axes, linear.
    pub scales: Vec<[f32; 3]>,
    /// Each splat's opacity, from 0 to 1.
    pub opacities: Vec<f32>,
    /// Each splat's base color, its degree-0 coefficients (red, green, blue).
    pub dc: Vec<[f32; 3]>,
    /// The higher degrees' coefficients, a column each (red, green, blue), in the order
    /// `KHR_gaussian_splatting` numbers them: degree by degree, and within a degree from order
    /// -l to l. There are 3, 8 or 15 columns, or none.
    pub sh: Vec<Vec<[f32; 3]>>,
}

#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
impl Splats {
    /// Room for `n` splats with `rest` coefficients a channel past the base one, or an error
    /// where there isn't memory for them. Callers check `n` against the bytes they hold first.
    #[cfg_attr(
        not(any(feature = "ply", feature = "splat")),
        allow(dead_code, reason = "SPZ sizes its columns as it unpacks them")
    )]
    pub fn with_capacity(n: usize, rest: usize) -> Result<Splats> {
        fn room<T>(n: usize) -> Result<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n)
                .ok()
                .context("not enough memory for the splats")?;
            Ok(v)
        }
        let mut sh = Vec::with_capacity(rest);
        for _ in 0..rest {
            sh.push(room(n)?);
        }
        Ok(Splats {
            positions: room(n)?,
            rotations: room(n)?,
            scales: room(n)?,
            opacities: room(n)?,
            dc: room(n)?,
            sh,
        })
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Flips the axes whose entry in `s` is -1 (the others are 1): positions, orientations and
    /// the view-dependent color follow, so the splats look the same in the new axes.
    pub fn flip(&mut self, s: [f32; 3]) {
        let det = s[0] * s[1] * s[2];
        for p in &mut self.positions {
            for k in 0..3 {
                p[k] *= s[k];
            }
        }
        // The flipped rotation is S R S: its axis flips as a pseudovector does, its angle stays.
        for q in &mut self.rotations {
            for k in 0..3 {
                q[k] *= det * s[k];
            }
        }
        // Each basis function is even or odd in each axis; a coefficient changes sign once for
        // each odd axis flipped.
        let [x, y, z] = s;
        let signs = [y, z, x, x * y, y * z, 1.0, x * z, 1.0, y, x * y * z, y, z, x, z, x];
        for (col, sign) in self.sh.iter_mut().zip(signs) {
            if sign < 0.0 {
                for c in col {
                    *c = c.map(|v| -v);
                }
            }
        }
    }

    /// Drops splats whose position isn't a finite point, and brings other values into the ranges
    /// the extension allows: non-finite values become 0, scales their magnitude, and opacities
    /// are clamped to [0, 1]. Coefficient columns past the last whole degree are dropped. When
    /// splats were dropped, returns which were kept, one entry for each splat there was.
    pub fn sanitize(&mut self) -> Option<Vec<bool>> {
        let n = self.positions.len();
        self.sh.truncate(REST[degree_of(self.sh.len())]);
        if self.rotations.len() != n
            || self.scales.len() != n
            || self.opacities.len() != n
            || self.dc.len() != n
            || self.sh.iter().any(|c| c.len() != n)
        {
            // Columns that don't match say nothing reliable.
            *self = Splats::default();
            return Some(vec![false; n]);
        }
        let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
        let mut dropped = None;
        if self.positions.iter().any(|p| !p.iter().all(|c| c.is_finite())) {
            let keep: Vec<bool> = self.positions.iter().map(|p| p.iter().all(|c| c.is_finite())).collect();
            retain(&mut self.positions, &keep);
            retain(&mut self.rotations, &keep);
            retain(&mut self.scales, &keep);
            retain(&mut self.opacities, &keep);
            retain(&mut self.dc, &keep);
            for col in &mut self.sh {
                retain(col, &keep);
            }
            dropped = Some(keep);
        }
        for s in &mut self.scales {
            *s = s.map(|v| finite(v).abs());
        }
        for o in &mut self.opacities {
            *o = if *o >= 0.0 { o.min(1.0) } else { 0.0 };
        }
        for c in self.sh.iter_mut().chain([&mut self.dc]) {
            for v in c {
                *v = v.map(finite);
            }
        }
        dropped
    }

    /// The point primitive that draws the splats, with the attributes `KHR_gaussian_splatting`
    /// defines and the fallback `COLOR_0` (normalized 16-bit), its arrays moved into the buffer;
    /// `None` when no splat is left (see [`Splats::sanitize`]). The splats' colors are sRGB, as
    /// training on photographs makes them.
    pub fn primitive(mut self, b: &mut Builder) -> Option<Value> {
        self.sanitize();
        let n = self.len();
        if n == 0 {
            return None;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a value from 0 to 1, scaled to 16 bits and rounded"
        )]
        let colors: Vec<u16> = self
            .dc
            .iter()
            .zip(&self.opacities)
            .flat_map(|(&dc, &o)| fallback(dc, o, true).map(|c| (c * 65535.0 + 0.5) as u16))
            .collect();
        let mut attributes = json!({ "POSITION": b.positions(self.positions) });
        let color = b.accessor(Part::u16(colors), 5123, n, "VEC4", Some(34962));
        b.accessors[color]["normalized"] = json!(true);
        attributes["COLOR_0"] = json!(color);
        let mut float = |v: Vec<f32>, kind: &str| b.accessor(Part::f32(v), 5126, n, kind, Some(34962));
        attributes[ROTATION] = json!(float(self.rotations.into_flattened(), "VEC4"));
        attributes[SCALE] = json!(float(self.scales.into_flattened(), "VEC3"));
        attributes[OPACITY] = json!(float(self.opacities, "SCALAR"));
        attributes[SH0] = json!(float(self.dc.into_flattened(), "VEC3"));
        for (k, col) in self.sh.into_iter().enumerate() {
            let degree = REST.iter().position(|&r| r > k).unwrap_or(3);
            let coef = k - REST[degree - 1];
            let name = format!("{EXTENSION}:SH_DEGREE_{degree}_COEF_{coef}");
            attributes[name] = json!(float(col.into_flattened(), "VEC3"));
        }
        Some(json!({
            "attributes": attributes,
            "mode": 0,
            "extensions": { EXTENSION: { "kernel": "ellipse", "colorSpace": "srgb_rec709_display" } },
        }))
    }

    /// A document of the splats alone, on one node turned by `rotation` (x, y, z, w) when given.
    /// `info` written over `source` (the format's own details) goes in the asset's
    /// `extras.source`, and the facts get `points` and `splats`, the splats drawn. Without splats,
    /// the document is empty but for those.
    #[cfg_attr(
        not(any(feature = "splat", feature = "spz")),
        allow(dead_code, reason = "PLY splats share a document with the file's other elements")
    )]
    pub fn document(
        self,
        info: &SourceInfo,
        mut source: Map<String, Value>,
        rotation: Option<[f32; 4]>,
        mut facts: Map<String, Value>,
    ) -> Document {
        let mut b = Builder::default();
        let degree = degree_of(self.sh.len());
        info.write(&mut source);
        let Some(prim) = self.primitive(&mut b) else {
            facts.insert("points".into(), json!(0));
            facts.insert("splats".into(), json!(0));
            let mut doc = Document::empty(facts);
            doc.json["asset"]["extras"] = json!({ "source": source });
            return doc;
        };
        let n = prim["attributes"]["POSITION"]
            .as_u64()
            .and_then(|a| b.accessors[usize::try_from(a).ok()?]["count"].as_u64())
            .unwrap_or(0);
        facts.insert("points".into(), json!(n));
        facts.insert("splats".into(), json!(n));
        source.insert("shDegree".into(), json!(degree));
        let mut node = json!({ "mesh": 0 });
        if let Some(q) = rotation {
            node["rotation"] = json!(q);
        }
        let mut json = Map::new();
        json.insert("asset".into(), json!({ "extras": { "source": source } }));
        json.insert("extensionsUsed".into(), json!([EXTENSION]));
        json.insert("scene".into(), json!(0));
        json.insert("scenes".into(), json!([{ "nodes": [0] }]));
        json.insert("nodes".into(), json!([node]));
        json.insert("meshes".into(), json!([{ "primitives": [prim] }]));
        b.finish(json, facts)
    }
}

/// Keeps the items of `v` whose entry in `keep` is true.
#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
pub(crate) fn retain<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut k = keep.iter();
    v.retain(|_| k.next().copied().unwrap_or(false));
}

/// Adds the extension to `json`'s `extensionsUsed`.
#[cfg_attr(
    not(feature = "ply"),
    allow(dead_code, reason = "only the PLY reader adds splats to a document")
)]
pub(crate) fn use_extension(json: &mut Map<String, Value>) {
    let list = json.entry("extensionsUsed").or_insert_with(|| json!([]));
    if let Some(list) = list.as_array_mut()
        && !list.iter().any(|v| v == EXTENSION)
    {
        list.push(json!(EXTENSION));
    }
}

/// A unit quaternion (x, y, z, w) from components at any scale; the identity for a zero or
/// broken one, which says nothing.
#[cfg_attr(
    not(any(feature = "ply", feature = "splat", feature = "spz")),
    allow(dead_code, reason = "only the splat readers build splats")
)]
pub(crate) fn rotation(x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
    let len = (x * x + y * y + z * z + w * w).sqrt();
    if len > 0.0 && len.is_finite() {
        [x / len, y / len, z / len, w / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// The logistic function, which turns a stored logit into an opacity.
#[cfg_attr(not(feature = "ply"), allow(dead_code, reason = "only PLY splats store logits"))]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A splat's color where it isn't drawn as a splat, as `KHR_gaussian_splatting` suggests for
/// `COLOR_0`: its base color (`0.5 + SH_C0 * dc`, clamped to [0, 1]), made linear when the
/// splats' colors are sRGB (`srgb`), and its opacity as alpha.
pub(crate) fn fallback(dc: [f32; 3], opacity: f32, srgb: bool) -> [f32; 4] {
    let unit = |v: f32| if v >= 0.0 { v.min(1.0) } else { 0.0 };
    let c = dc.map(|d| {
        let c = unit(0.5 + SH_C0 * d);
        if srgb { linear(c) } else { c }
    });
    [c[0], c[1], c[2], unit(opacity)]
}

/// How many steps [`linear`]'s table has.
const STEPS: usize = 4096;

/// `c` (from 0 to 1) from sRGB to linear, interpolated in a table, as millions of splats each want
/// three. Within 1e-6 of the exact curve, far under a 16-bit step.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a table index from a value from 0 to 1"
)]
fn linear(c: f32) -> f32 {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    let t = TABLE.get_or_init(|| (0..=STEPS).map(|i| srgb_to_linear(i as f32 / STEPS as f32)).collect());
    let x = c * STEPS as f32;
    let i = (x as usize).min(STEPS - 1);
    let f = x - i as f32;
    t[i] + f * (t[i + 1] - t[i])
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test files are small"
)]
mod tests {
    use super::*;

    #[test]
    fn the_color_table_follows_the_srgb_curve() {
        for k in 0..=10_000u16 {
            let c = f32::from(k) / 10_000.0;
            assert!((linear(c) - srgb_to_linear(c)).abs() < 1e-6, "{c}");
        }
    }

    #[test]
    fn the_fallback_color_is_the_base_color_made_linear() {
        assert_eq!(fallback([0.0; 3], 0.25, false), [0.5, 0.5, 0.5, 0.25]);
        let c = fallback([1.0 / SH_C0, -1.0 / SH_C0, f32::NAN], f32::NAN, true);
        assert!(
            (c[0] - 1.0).abs() < 1e-6 && c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0,
            "{c:?}"
        );
    }

    #[test]
    fn degrees_by_their_coefficients() {
        let got: Vec<usize> = [0, 2, 3, 7, 8, 14, 15, 24, 45].map(degree_of).to_vec();
        assert_eq!(got, [0, 0, 1, 1, 2, 2, 3, 3, 3]);
    }

    /// A splat's color in direction `d`, from its coefficients, as the extension's reference
    /// evaluates it.
    fn shade(dc: [f32; 3], sh: &[[f32; 3]], d: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = d;
        let basis = [
            -0.488_602_5 * y,
            0.488_602_5 * z,
            -0.488_602_5 * x,
            1.092_548_4 * x * y,
            -1.092_548_4 * y * z,
            0.315_391_57 * (3.0 * z * z - 1.0),
            -1.092_548_4 * x * z,
            0.546_274_2 * (x * x - y * y),
            -0.590_043_6 * y * (3.0 * x * x - y * y),
            2.890_611_4 * x * y * z,
            -0.457_045_8 * y * (5.0 * z * z - 1.0),
            0.373_176_33 * z * (5.0 * z * z - 3.0),
            -0.457_045_8 * x * (5.0 * z * z - 1.0),
            1.445_305_7 * z * (x * x - y * y),
            -0.590_043_6 * x * (x * x - 3.0 * y * y),
        ];
        let mut c = dc.map(|v| SH_C0 * v);
        for (coef, b) in sh.iter().zip(basis) {
            for k in 0..3 {
                c[k] += b * coef[k];
            }
        }
        c
    }

    #[test]
    fn flipped_splats_look_the_same_from_the_flipped_direction() {
        let mut rng = crate::test_rng::Pcg32::new(3);
        let mut r = || rng.f32() * 2.0 - 1.0;
        let sh: Vec<[f32; 3]> = (0..15).map(|_| [r(), r(), r()]).collect();
        let dc = [r(), r(), r()];
        for s in [
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ] {
            let mut splats = Splats {
                positions: vec![[0.1, 0.2, 0.3]],
                rotations: vec![rotation(0.1, 0.2, 0.3, 0.9)],
                scales: vec![[1.0; 3]],
                opacities: vec![1.0],
                dc: vec![dc],
                sh: sh.iter().map(|&c| vec![c]).collect(),
            };
            splats.flip(s);
            let flipped: Vec<[f32; 3]> = splats.sh.iter().map(|c| c[0]).collect();
            for _ in 0..10 {
                let d = [r(), r(), r()];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let d = d.map(|v| v / len);
                let want = shade(dc, &sh, d);
                let got = shade(dc, &flipped, [d[0] * s[0], d[1] * s[1], d[2] * s[2]]);
                for k in 0..3 {
                    assert!((want[k] - got[k]).abs() < 1e-5, "{s:?}: {want:?} != {got:?}");
                }
            }
            // The covariance flips with the axes: S R S's columns are R's, flipped.
            let m = |q: [f32; 4]| {
                let [x, y, z, w] = q;
                [
                    [
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y + w * z),
                        2.0 * (x * z - w * y),
                    ],
                    [
                        2.0 * (x * y - w * z),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z + w * x),
                    ],
                    [
                        2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y),
                    ],
                ]
            };
            let (a, b) = (m(rotation(0.1, 0.2, 0.3, 0.9)), m(splats.rotations[0]));
            for col in 0..3 {
                for row in 0..3 {
                    let want = s[row] * a[col][row] * s[col];
                    assert!((b[col][row] - want).abs() < 1e-5, "{s:?}");
                }
            }
        }
    }

    #[test]
    fn broken_values_are_brought_into_range_and_broken_positions_dropped() {
        let splats = Splats {
            positions: vec![[0.0; 3], [f32::NAN, 0.0, 0.0], [1.0, 2.0, 3.0]],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]; 3],
            scales: vec![[-2.0, f32::INFINITY, 1.0]; 3],
            opacities: vec![2.0, 0.5, f32::NAN],
            dc: vec![[f32::NAN, 1.0, 2.0]; 3],
            sh: vec![vec![[1.0; 3]; 3]; 4],
        };
        let mut b = Builder::default();
        let p = splats.primitive(&mut b).unwrap();
        let a = &p["attributes"];
        assert_eq!(b.accessors[a["POSITION"].as_u64().unwrap() as usize]["count"], 2);
        // Four columns hold degree 1 and part of degree 2: degree 1 stays.
        assert!(a.get("KHR_gaussian_splatting:SH_DEGREE_1_COEF_2").is_some());
        assert!(a.get("KHR_gaussian_splatting:SH_DEGREE_2_COEF_0").is_none());
        assert_eq!(p["extensions"][EXTENSION]["kernel"], "ellipse");
    }
}
