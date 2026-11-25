//! `Line` and `NurbsCurve` geometry as line segments.

use serde_json::json;

use super::{Reader, child_attr, floats, ints, kind_of, name_of, num, string, vec3};
use crate::formats::fbx_tree::NodeHandle;
use crate::formats::nurbs::{self, MAX_CURVE_POINTS};
use crate::mesh::Prim;
use crate::num::f32_of;

impl Reader<'_, '_, '_> {
    /// The mesh for `Line` or `NurbsCurve` geometry `g` as model `m` (0 for none) draws it: its segments, in the
    /// model's first material, else unlit in the geometry's `Color` (white without one); `None` when it has none.
    pub(super) fn line_mesh(&mut self, g: i64, m: i64) -> Option<usize> {
        let mut mats = if m == 0 { Vec::new() } else { self.model_materials(m) };
        mats.truncate(1);
        if let Some(&i) = self.mesh_of.get(&(g, mats.clone())) {
            return Some(i);
        }
        let geom = self.objs.node(g)?;
        let prim = if kind_of(geom) == "NurbsCurve" {
            let (prim, drawn) = curve_geometry(geom, MAX_CURVE_POINTS.saturating_sub(self.curve_points));
            self.curve_points += drawn;
            prim
        } else {
            line_geometry(geom)
        };
        if prim.lines.is_empty() {
            return None;
        }
        let material = match mats.first() {
            Some(&i) => i,
            None if self.keep.materials => {
                let color = vec3(&self.objs.props_of(geom), "Color", [1.0; 3]);
                self.line_material(color)
            }
            None => self.default_material(),
        };
        let mut primitives = self.b.primitives(prim);
        for p in &mut primitives {
            p["material"] = json!(material);
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(name) = name_of(geom) {
            mesh["name"] = json!(name);
        }
        self.meshes.push(mesh);
        let i = self.meshes.len() - 1;
        self.mesh_of.insert((g, mats), i);
        Some(i)
    }
}

/// A `Line` geometry's polylines as one primitive's segments, on the control points (`Points`) they use. Each
/// runs through `PointsIndex` to an index written negative (`!i`), its last point; an index that isn't a whole
/// number, or a control point a float can hold, breaks it there.
fn line_geometry(geom: NodeHandle<'_>) -> Prim {
    let points = child_attr(geom, "Points").and_then(floats).unwrap_or_default();
    let index = child_attr(geom, "PointsIndex")
        .and_then(|a| ints(a).or_else(|| floats(a).map(|f| f.into_iter().map(whole).collect())))
        .unwrap_or_default();
    let point = |cp: usize| [points[cp * 3], points[cp * 3 + 1], points[cp * 3 + 2]].map(f32_of);
    let mut prim = Prim::default();
    // Control point → vertex, for those a segment uses: no more room than the points take.
    let mut vertex = vec![u32::MAX; points.len() / 3];
    let mut last = None;
    for i in index {
        let (cp, end) = if i < 0 { (!i, true) } else { (i, false) };
        let cp = usize::try_from(cp)
            .ok()
            .filter(|&cp| cp < vertex.len() && point(cp).iter().all(|c| c.is_finite()));
        if let (Some(a), Some(b)) = (last, cp)
            && a != b
        {
            for c in [a, b] {
                if vertex[c] == u32::MAX {
                    let Ok(v) = u32::try_from(prim.positions.len()) else {
                        return prim;
                    };
                    vertex[c] = v;
                    prim.positions.push(point(c));
                }
                prim.lines.push(vertex[c]);
            }
        }
        last = if end { None } else { cp };
    }
    prim
}

/// Whether `NurbsCurve` geometry has what drawing it takes: an `Order` (a whole number from 2) and `Points`.
pub(super) fn is_curve(geom: NodeHandle<'_>) -> bool {
    child_attr(geom, "Order")
        .and_then(num)
        .map(whole)
        .is_some_and(|o| (2..i64::MAX).contains(&o))
        && child_attr(geom, "Points")
            .and_then(floats)
            .is_some_and(|p| p.len() >= 8)
}

/// A `NurbsCurve` geometry as one primitive's segments: the curve of its `Order` (its degree plus one) through its
/// `Points` (x, y, z and a weight each) over its `KnotVector`, a `Periodic` one wrapped around to its first points
/// again. A curve whose order or knots don't fit its points, or that takes more than `budget` points, is drawn
/// through its control points; so is one with control points a float can't hold, broken there. The primitive, and
/// the points it took from the budget.
fn curve_geometry(geom: NodeHandle<'_>, budget: usize) -> (Prim, usize) {
    if !is_curve(geom) {
        return (Prim::default(), 0);
    }
    let raw = child_attr(geom, "Points").and_then(floats).unwrap_or_default();
    let mut ctrl: Vec<[f64; 3]> = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect();
    let mut weights: Vec<f64> = raw.chunks_exact(4).map(|c| c[3]).collect();
    let order = child_attr(geom, "Order").and_then(num).map_or(0, whole);
    let degree = order - 1;
    let knots = child_attr(geom, "KnotVector").and_then(floats).unwrap_or_default();
    let periodic = child_attr(geom, "Form").and_then(string) == Some("Periodic");
    let n = ctrl.len();
    // A periodic curve's knots run past its points by its degree: those points are its first ones again.
    if periodic
        && let Ok(p) = usize::try_from(degree)
        && p < n
        && knots.len() == n + 2 * p + 1
    {
        ctrl.extend_from_within(..p);
        weights.extend_from_within(..p);
    }
    let finite = ctrl.iter().flatten().all(|c| c.is_finite());
    let mut pts = Vec::new();
    let drawn = if let Some(Ok(drawn)) = finite.then(|| nurbs::curve(degree, &ctrl, &weights, &knots, budget, &mut pts))
    {
        drawn
    } else {
        pts.clone_from(&ctrl);
        if periodic {
            pts.truncate(n);
            pts.push(ctrl[0]);
        }
        0
    };
    let mut prim = Prim::default();
    // The vertex the polyline is at: none past a point that isn't a number. One point twice draws nothing.
    let mut last: Option<u32> = None;
    for p in pts {
        let p = p.map(f32_of);
        if !p.iter().all(|c| c.is_finite()) {
            last = None;
            continue;
        }
        #[allow(clippy::float_cmp, reason = "the same point twice, exactly")]
        if last.is_some_and(|v| prim.positions[v as usize] == p) {
            continue;
        }
        let Ok(v) = u32::try_from(prim.positions.len()) else {
            break;
        };
        prim.positions.push(p);
        if let Some(u) = last {
            prim.lines.extend_from_slice(&[u, v]);
        }
        last = Some(v);
    }
    (prim, drawn)
}

/// A whole number written as a float, as an index; anything else as one no array has.
pub(super) fn whole(f: f64) -> i64 {
    #[allow(clippy::cast_possible_truncation, reason = "a whole number in range")]
    if f.fract() == 0.0 && f.abs() < 2f64.powi(62) {
        f as i64
    } else {
        i64::MAX
    }
}
