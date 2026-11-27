//! Normals smoothed across the polygons that meet at a point, up to a crease angle: for formats that store a
//! smoothing angle rather than normals, and for every file that says nothing at all about how it is shaded.

use crate::vec3::{dot, unit};

/// The crease angle of a file with no normals and nothing of its own to say how it is smoothed, in degrees,
/// unless the caller asks for another ([`Keep::crease_angle`](crate::Keep::crease_angle), whose default this is;
/// `Keep::ALL` spells it out in hundredths): two faces that meet at a point share a normal there when they turn
/// less than this from each other. It is the usual default of modeling tools' automatic smoothing: edges meant to
/// be hard (a box's are 90 degrees) stay hard, and a curved surface cut into faces fine enough to read as
/// curved comes out smooth, whatever format it was stored in.
pub(crate) const DEFAULT_ANGLE: f32 = 30.0;

/// The most polygons around one point that are compared pair by pair without counting the cost.
const MAX_AROUND: usize = 512;

/// How many comparisons the points with more than [`MAX_AROUND`] polygons may cost in all, beside 64 for each
/// corner: enough for the tip of any cone or the middle of any fan a model has, and a bound on what a hostile
/// file costs. Past it, such a point's polygons that are within the angle of their mean share that mean.
const BUDGET: usize = 1 << 26;

/// Each corner's normal, in the order of `corners`. `polys` are (first corner, corner count) spans of `corners`,
/// whose values are point indices below `points`; `face` is each polygon's unit normal. A corner averages the
/// normals of the polygons around its point that `joins(own, other)` accepts and that are within the angle whose
/// cosine `cos(own)` gives; `None` keeps the polygon flat.
#[cfg_attr(
    not(any(feature = "lwo", feature = "ac3d", feature = "x3d")),
    allow(dead_code, reason = "only formats with a smoothing angle of their own use it")
)]
pub(crate) fn corner_normals(
    points: usize,
    polys: &[(usize, usize)],
    corners: &[u32],
    face: &[[f32; 3]],
    cos: impl Fn(usize) -> Option<f32>,
    joins: impl Fn(usize, usize) -> bool,
) -> Vec<[f32; 3]> {
    weighted(points, polys, corners, face, None, cos, joins)
}

/// The default normals of triangles `indices` over `positions`, one for each corner in the order of `indices`:
/// at each corner, the normals of the triangles around its vertex that are within `degrees` of its own, summed by
/// their areas. At 0 every triangle keeps its own, and at 180 every triangle around a vertex counts. A triangle
/// with no area has no normal: its corners are zero.
pub(crate) fn default_normals(positions: &[[f32; 3]], indices: &[u32], degrees: f32) -> Vec<[f32; 3]> {
    let tris = indices.len() / 3;
    let mut face = Vec::with_capacity(tris);
    let mut area = Vec::with_capacity(tris);
    for t in indices.as_chunks::<3>().0 {
        // In double precision, so a float's square can't overflow.
        let [a, b, c] = t.map(|i| positions.get(i as usize).copied().unwrap_or([0.0; 3]).map(f64::from));
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let f = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let l = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a unit vector, and a weight that only needs its size"
        )]
        if l > 0.0 && l.is_finite() {
            face.push(f.map(|x| (x / l) as f32));
            area.push(l);
        } else {
            face.push([0.0; 3]);
            area.push(0.0);
        }
    }
    // Weights relative to the largest, so they fit a float whatever the model's size.
    let most = area.iter().fold(0.0f64, |m, &a| m.max(a));
    #[allow(clippy::cast_possible_truncation, reason = "between 0 and 1")]
    let weight: Vec<f32> = area
        .iter()
        .map(|&a| if most > 0.0 { (a / most) as f32 } else { 0.0 })
        .collect();
    let polys: Vec<(usize, usize)> = (0..tris).map(|t| (t * 3, 3)).collect();
    // The ends are exact: no face joins another at 0, and none is left out at 180, whatever rounding does.
    let cos = if degrees <= 0.0 {
        None
    } else if degrees >= 180.0 {
        Some(-2.0)
    } else {
        Some(degrees.to_radians().cos())
    };
    let mut out = weighted(
        positions.len(),
        &polys,
        &indices[..tris * 3],
        &face,
        Some(&weight),
        |_| cos,
        |_, _| true,
    );
    // A triangle without area gave its corners its own lack of a normal.
    for (t, n) in out.chunks_exact_mut(3).enumerate() {
        if weight[t] == 0.0 {
            n.fill([0.0; 3]);
        }
    }
    out.resize(indices.len(), [0.0; 3]);
    out
}

/// [`corner_normals`], each polygon's normal counting for `weight[polygon]` (1 without).
fn weighted(
    points: usize,
    polys: &[(usize, usize)],
    corners: &[u32],
    face: &[[f32; 3]],
    weight: Option<&[f32]>,
    cos: impl Fn(usize) -> Option<f32>,
    joins: impl Fn(usize, usize) -> bool,
) -> Vec<[f32; 3]> {
    let span = |&(s, n): &(usize, usize)| corners.get(s..s.saturating_add(n)).unwrap_or(&[]);
    let weigh = |q: usize| weight.map_or(1.0, |w| w.get(q).copied().unwrap_or(0.0));
    // The polygons around each point, as ranges of one list.
    let mut start = vec![0usize; points + 1];
    for p in polys {
        for &c in span(p) {
            if let Some(s) = start.get_mut(c as usize + 1) {
                *s += 1;
            }
        }
    }
    for i in 1..start.len() {
        start[i] += start[i - 1];
    }
    let mut around = vec![0u32; start[points]];
    let mut fill = start.clone();
    for (i, p) in polys.iter().enumerate() {
        for &c in span(p).iter().filter(|&&c| (c as usize) < points) {
            let f = &mut fill[c as usize];
            around[*f] = u32::try_from(i).unwrap_or(u32::MAX);
            *f += 1;
        }
    }

    // What the crowded points may still cost, and the mean of each one that is past it.
    let mut budget = BUDGET.saturating_add(corners.len().saturating_mul(64));
    let mut means: std::collections::HashMap<usize, Option<[f32; 3]>> = std::collections::HashMap::new();
    let mut paid: Vec<bool> = Vec::new();
    let mut out = vec![[0.0; 3]; corners.len()];
    for (p, &(s, n)) in polys.iter().enumerate() {
        let own = face.get(p).copied().unwrap_or([0.0, 1.0, 0.0]);
        let cos = cos(p);
        for k in s..s.saturating_add(n).min(corners.len()) {
            let c = corners[k] as usize;
            let (Some(cos), Some(&a), Some(&b)) = (cos, start.get(c), start.get(c + 1)) else {
                out[k] = own;
                continue;
            };
            let list = &around[a..b];
            if list.len() > MAX_AROUND {
                // A crowded point: compared in full while the budget lasts, each point paying once.
                if paid.is_empty() {
                    paid = vec![false; points];
                }
                if !paid[c] && !means.contains_key(&c) {
                    let cost = list.len().saturating_mul(list.len());
                    if cost <= budget {
                        budget -= cost;
                        paid[c] = true;
                    } else {
                        let mut sum = [0.0f32; 3];
                        for &q in list {
                            if let Some(&f) = face.get(q as usize) {
                                let w = weigh(q as usize);
                                sum = [sum[0] + f[0] * w, sum[1] + f[1] * w, sum[2] + f[2] * w];
                            }
                        }
                        means.insert(c, unit(sum));
                    }
                }
                if !paid[c] {
                    // Past the budget: the polygons within the angle of the point's mean share it.
                    let mean = means.get(&c).copied().flatten();
                    out[k] = mean.filter(|m| dot(*m, own) >= cos).unwrap_or(own);
                    continue;
                }
            }
            let mut sum = [0.0f32; 3];
            let mut last = u32::MAX;
            for &q in list {
                // A polygon that comes back to the point counts once.
                if q == last {
                    continue;
                }
                last = q;
                let q = q as usize;
                let Some(&f) = face.get(q) else { continue };
                if q == p || (joins(p, q) && dot(f, own) >= cos) {
                    let w = weigh(q);
                    sum = [sum[0] + f[0] * w, sum[1] + f[1] * w, sum[2] + f[2] * w];
                }
            }
            out[k] = unit(sum).unwrap_or(own);
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "exact directions")]
mod tests {
    use super::*;

    #[test]
    fn creases_split_and_smooth() {
        // Two squares folded 90 degrees along their shared edge (points 1 and 2).
        let corners = [0, 1, 2, 3, 1, 4, 5, 2];
        let polys = [(0, 4), (4, 4)];
        let face = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let soft = corner_normals(6, &polys, &corners, &face, |_| Some(0.0), |_, _| true);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((soft[1][0] - h).abs() < 1e-6 && (soft[1][2] - h).abs() < 1e-6);
        assert_eq!(soft[0], face[0]);
        let hard = corner_normals(6, &polys, &corners, &face, |_| Some(0.5), |_, _| true);
        assert_eq!(hard[1], face[0]);
        assert_eq!(hard[4], face[1]);
        let apart = corner_normals(6, &polys, &corners, &face, |_| Some(0.0), |a, b| a == b);
        assert_eq!(apart[1], face[0]);
        let flat = corner_normals(6, &polys, &corners, &face, |_| None, |_, _| true);
        assert_eq!(flat[5], face[1]);
        // Out-of-range points and spans don't panic.
        let odd = corner_normals(2, &[(0, 9), (7, 3)], &corners, &face[..1], |_| Some(0.0), |_, _| true);
        assert_eq!(odd.len(), corners.len());
    }

    #[test]
    fn the_default_keeps_a_right_angle_hard_and_a_gentle_fold_smooth() {
        // Two triangles sharing the edge from 0 to 1: one on the ground, the other tilted by `angle` from it.
        let fold = |angle: f32| {
            let (s, c) = angle.to_radians().sin_cos();
            let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -c, s]];
            default_normals(&positions, &[0, 1, 2, 1, 0, 3], DEFAULT_ANGLE)
        };
        let hard = fold(90.0);
        assert_eq!(hard[0], [0.0, 0.0, 1.0]);
        assert!((hard[4][1] - 1.0).abs() < 1e-6, "{:?}", hard[4]);
        let soft = fold(20.0);
        // The shared corners take one normal, between the two faces'; the others keep their face's.
        assert_eq!(soft[0], soft[4]);
        assert_eq!(soft[1], soft[3]);
        assert!(soft[0][2] < 1.0 && soft[0][2] > 0.98, "{:?}", soft[0]);
        assert_eq!(soft[2], [0.0, 0.0, 1.0]);
        // A triangle with no area has no normal.
        let none = default_normals(&[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &[0, 1, 2], DEFAULT_ANGLE);
        assert_eq!(none, [[0.0; 3]; 3]);
    }

    #[test]
    fn the_tip_of_a_cone_with_many_faces_is_smooth() {
        // A shallow cone of 2000 faces: every face is within the angle of its neighbors, and the tip has far
        // more faces around it than are usually compared.
        let n = 2000u32;
        let mut positions = vec![[0.0, 0.0, 0.1]];
        for k in 0..n {
            #[allow(clippy::cast_precision_loss, reason = "a few thousand")]
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            positions.push([a.cos(), a.sin(), 0.0]);
        }
        let indices: Vec<u32> = (0..n).flat_map(|k| [0, 1 + k, 1 + (k + 1) % n]).collect();
        let normals = default_normals(&positions, &indices, DEFAULT_ANGLE);
        // At the tip, a face's corner isn't its own flat normal: it leans toward the axis with its neighbors.
        let face = crate::mesh::face_normal(positions[0], positions[1], positions[2]);
        assert!(normals[0][2] > face[2] + 1e-3, "{:?} {face:?}", normals[0]);
        // And on the rim, two faces share one normal.
        assert_eq!(normals[2], normals[4]);
    }
}
