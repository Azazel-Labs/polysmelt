//! Polygons with holes, triangulated in their plane: each hole joined to the outer loop by a two-way cut, then the
//! one outline ear-clipped, with an O(n²) ear walk and each triangle turned to face the same way as the polygon.
//! The readers whose faces carry holes (COLLADA's `<ph>`, `SketchUp`'s inner loops) share it; a port of openskp's
//! `triangulator.cpp` and `earcut.cpp` at heart.
//!
//! What it copes with: holes that touch the outline or each other at a corner, collinear and repeated corners,
//! holes wound either way, holes that overlap (clipped from the outline after), and degenerate or non-finite input,
//! which gives fewer triangles or none rather than a panic.

use crate::vec3::{self, cross, dot, sub, unit};

type V3 = [f64; 3];

/// Polygons past this many corners, counting each hole's bridge twice, are fanned from the outline's first corner
/// instead: ear clipping costs the square of the corners at best and the cube at worst.
const MAX_CLIP: usize = 4096;

/// The unit normal of a loop by Newell's method, which copes with concave and slightly non-planar loops.
pub(crate) fn newell(points: &[V3]) -> Option<V3> {
    unit(vec3::newell(points.iter().copied()))
}

/// Triangulates a polygon. `loops` holds the outer loop then its holes; the returned indices count through them in
/// that order. Each triangle winds counter-clockwise seen from the side `normal` points to, or the outer loop's
/// Newell normal when `normal` is `None` or has no direction. Slivers that draw nothing are left out.
pub(crate) fn triangulate_holes<L: AsRef<[V3]>>(loops: &[L], normal: Option<V3>) -> Vec<[usize; 3]> {
    let Some(outer) = loops.first().map(AsRef::as_ref) else {
        return Vec::new();
    };
    let Some(n) = normal.and_then(unit).or_else(|| newell(outer)) else {
        return Vec::new();
    };
    let points: Vec<V3> = loops.iter().flat_map(|l| l.as_ref().iter().copied()).collect();
    let scale = points
        .iter()
        .map(|&p| dot(sub(p, outer[0]), sub(p, outer[0])))
        .fold(0.0, f64::max);
    let mut tris = if loops.len() == 1 && outer.len() == 3 {
        vec![[0, 1, 2]]
    } else if loops.len() == 1 && outer.len() == 4 {
        split_quad(outer)
    } else if points.len() + 2 * (loops.len() - 1) > MAX_CLIP {
        (1..outer.len().saturating_sub(1)).map(|k| [0, k, k + 1]).collect()
    } else {
        // project onto the face plane
        let seed = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let Some(u) = unit(cross(n, seed)) else {
            return Vec::new();
        };
        let v = cross(n, u);
        let origin = outer[0];
        let mut id = 0;
        let flat: Vec<Vec<P>> = loops
            .iter()
            .map(|l| {
                l.as_ref()
                    .iter()
                    .map(|&p| {
                        let d = sub(p, origin);
                        id += 1;
                        P {
                            x: dot(d, u),
                            y: dot(d, v),
                            id: id - 1,
                        }
                    })
                    .collect()
            })
            .collect();
        // The tolerances are for polygons an inch or more across; smaller ones get them scaled down to size.
        earcut(flat, scale.min(1.0))
    };
    // slivers along a straight run of edges draw nothing and have no reliable facing, so they go
    tris.retain_mut(|t| {
        let c = cross(sub(points[t[1]], points[t[0]]), sub(points[t[2]], points[t[0]]));
        let facing = dot(c, n);
        if facing < 0.0 {
            t.swap(1, 2);
        }
        facing.abs() > scale * 1e-12
    });
    tris
}

/// Splits a quad along the diagonal that stays inside it: on a concave quad the other one leaves two halves
/// facing opposite ways.
fn split_quad(q: &[V3]) -> Vec<[usize; 3]> {
    let normal = |a: usize, b: usize, c: usize| cross(sub(q[b], q[a]), sub(q[c], q[a]));
    if dot(normal(0, 1, 2), normal(0, 2, 3)) >= 0.0 {
        vec![[0, 1, 2], [0, 2, 3]]
    } else {
        vec![[1, 2, 3], [1, 3, 0]]
    }
}

/// A 2D point and its index into the polygon's concatenated loops.
#[derive(Debug, Clone, Copy)]
struct P {
    x: f64,
    y: f64,
    id: usize,
}

/// Whether two points are the same, exactly: where a bridge comes back, or a hole touches the outline.
#[allow(clippy::float_cmp, reason = "the same point, exactly")]
#[inline]
fn same(a: P, b: P) -> bool {
    a.x == b.x && a.y == b.y
}

#[inline]
fn cross2(a: P, b: P, c: P) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn signed_area(poly: &[P]) -> f64 {
    let mut s = 0.0;
    for (i, a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        s += a.x * b.y - b.x * a.y;
    }
    s / 2.0
}

fn point_in_polygon(x: f64, y: f64, poly: &[P]) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for (i, a) in poly.iter().enumerate() {
        let b = poly[j];
        if (a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Triangulates the projected loops, outer first. `tol` scales the tolerances: 1 for a polygon an inch across or
/// more, less for a smaller one.
fn earcut(mut loops: Vec<Vec<P>>, tol: f64) -> Vec<[usize; 3]> {
    if loops.first().is_none_or(|l| l.len() < 3) {
        return Vec::new();
    }
    let mut outer = loops.remove(0);
    loops.retain(|l| l.len() >= 3);
    if signed_area(&outer) < 0.0 {
        outer.reverse();
    }
    for h in &mut loops {
        if signed_area(h) > 0.0 {
            h.reverse();
        }
    }
    if loops.len() > 1 && holes_overlap(&loops) {
        // Bridging overlapping holes makes a self-crossing polygon; clip the outline and cut the holes out after.
        // The outline's points are the first loop's, so its ids count from 0.
        let mut by_id = outer.clone();
        by_id.sort_by_key(|p| p.id);
        return clip(outer, tol)
            .into_iter()
            .filter(|t| {
                let x = t.iter().map(|&id| by_id[id].x).sum::<f64>() / 3.0;
                let y = t.iter().map(|&id| by_id[id].y).sum::<f64>() / 3.0;
                !loops.iter().any(|h| point_in_polygon(x, y, h))
            })
            .collect();
    }
    // bridge the holes rightmost first, so later bridges see earlier ones as part of the outline
    loops.sort_by(|a, b| max_x(b).total_cmp(&max_x(a)));
    for (k, hole) in loops.iter().enumerate() {
        outer = bridge(&outer, hole, &loops[k + 1..]);
    }
    clip(outer, tol)
}

fn max_x(poly: &[P]) -> f64 {
    poly.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
}

/// Whether a corner of one hole lies inside another. A corner one hole shares with another is where they touch,
/// which bridging copes with, so it doesn't count.
fn holes_overlap(holes: &[Vec<P>]) -> bool {
    let bounds: Vec<[f64; 4]> = holes
        .iter()
        .map(|h| {
            h.iter().fold(
                [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
                |b, p| [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)],
            )
        })
        .collect();
    holes.iter().enumerate().any(|(i, a)| {
        holes.iter().enumerate().any(|(j, b)| {
            let [x0, y0, x1, y1] = bounds[j];
            // a point outside the hole's bounds is outside the hole
            i != j
                && a.iter().any(|&p| {
                    !(p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1)
                        && point_in_polygon(p.x, p.y, b)
                        && !b.iter().any(|&q| same(p, q))
                })
        })
    })
}

fn segments_cross(a: P, b: P, c: P, d: P) -> bool {
    let d1 = cross2(a, b, c);
    let d2 = cross2(a, b, d);
    let d3 = cross2(c, d, a);
    let d4 = cross2(c, d, b);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Joins a hole to the outline with a two-way cut from the hole's rightmost point to the nearest outline point the
/// cut can reach without crossing an edge: one to its right if there is one, else any, else the nearest. A hole that
/// touches the outline is joined where it touches instead, as a cut from anywhere else would pass by the touch.
fn bridge(outer: &[P], hole: &[P], rest: &[Vec<P>]) -> Vec<P> {
    if let Some((oi, hi)) = touch(outer, hole) {
        let mut merged = Vec::with_capacity(outer.len() + hole.len());
        merged.extend_from_slice(&outer[..=oi]);
        merged.extend(hole[hi + 1..].iter().chain(&hole[..=hi]).copied());
        merged.extend_from_slice(&outer[oi + 1..]);
        return merged;
    }
    let mut hi = 0;
    for (i, p) in hole.iter().enumerate() {
        if p.x.total_cmp(&hole[hi].x).then(hole[hi].y.total_cmp(&p.y)).is_gt() {
            hi = i;
        }
    }
    let h = hole[hi];
    let visible = |o: P| {
        !std::iter::once(outer)
            .chain(std::iter::once(hole))
            .chain(rest.iter().map(Vec::as_slice))
            .any(|r| (0..r.len()).any(|i| segments_cross(h, o, r[i], r[(i + 1) % r.len()])))
    };
    let dist = |p: P| (p.x - h.x).powi(2) + (p.y - h.y).powi(2);
    // Nearest first (ties in outline order), so the first visible candidate is the one to take.
    let mut order: Vec<usize> = (0..outer.len()).collect();
    order.sort_by(|&a, &b| dist(outer[a]).total_cmp(&dist(outer[b])));
    let oi = order
        .iter()
        .copied()
        .find(|&i| outer[i].x >= h.x && visible(outer[i]))
        .or_else(|| order.iter().copied().find(|&i| outer[i].x < h.x && visible(outer[i])))
        .or_else(|| order.first().copied())
        .unwrap_or(0);
    let mut merged = Vec::with_capacity(outer.len() + hole.len() + 2);
    merged.extend_from_slice(&outer[..=oi]);
    merged.extend(hole[hi..].iter().chain(&hole[..hi]).copied());
    merged.push(h);
    merged.extend_from_slice(&outer[oi..]);
    merged
}

/// The first outline point (and the hole point) where a hole touches the outline, if it does.
fn touch(outer: &[P], hole: &[P]) -> Option<(usize, usize)> {
    let [x0, y0, x1, y1] = hole.iter().fold(
        [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        |b, p| [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)],
    );
    // a point outside the hole's bounds is on none of its points
    outer.iter().enumerate().find_map(|(oi, &o)| {
        if o.x < x0 || o.x > x1 || o.y < y0 || o.y > y1 {
            return None;
        }
        hole.iter().position(|&p| same(p, o)).map(|hi| (oi, hi))
    })
}

#[inline]
fn in_triangle(p: P, a: P, b: P, c: P, eps: f64) -> bool {
    cross2(a, b, p) >= -eps && cross2(b, c, p) >= -eps && cross2(c, a, p) >= -eps
}

/// Whether the corner at `at` is an ear: convex, its triangle holding no other corner. A corner on top of one of
/// the ear's (a bridge's way back, or where a hole touches) doesn't block it.
fn is_ear(poly: &[P], at: usize, eps: f64) -> bool {
    let len = poly.len();
    let (a, b, c) = (poly[(at + len - 1) % len], poly[at], poly[(at + 1) % len]);
    if cross2(a, b, c) <= eps {
        return false;
    }
    !poly.iter().any(|&p| {
        p.id != a.id
            && p.id != b.id
            && p.id != c.id
            && in_triangle(p, a, b, c, eps)
            && !same(p, a)
            && !same(p, b)
            && !same(p, c)
    })
}

/// Ear clipping of a simple counter-clockwise polygon (bridged holes included).
fn clip(mut poly: Vec<P>, tol: f64) -> Vec<[usize; 3]> {
    let eps = 1e-12 * tol;
    if signed_area(&poly) < 0.0 {
        poly.reverse();
    }
    let mut out = Vec::new();
    let mut i = 0;
    let mut misses = 0;
    while poly.len() > 3 {
        let n = poly.len();
        i %= n;
        if misses >= n {
            // no ear in a full lap: drop a collinear point, or clip anyway on a self-crossing loop
            if let Some(k) =
                (0..n).find(|&k| cross2(poly[(k + n - 1) % n], poly[k], poly[(k + 1) % n]).abs() < 1e-10 * tol)
            {
                poly.remove(k);
            } else {
                out.push([poly[(i + n - 1) % n].id, poly[i].id, poly[(i + 1) % n].id]);
                poly.remove(i);
            }
            misses = 0;
            continue;
        }
        if is_ear(&poly, i, eps) {
            out.push([poly[(i + n - 1) % n].id, poly[i].id, poly[(i + 1) % n].id]);
            poly.remove(i);
            // the previous point may have just become an ear
            i = (i + n - 2) % (n - 1);
            misses = 0;
        } else {
            i += 1;
            misses += 1;
        }
    }
    if let [a, b, c] = poly[..]
        && cross2(a, b, c).abs() > eps
    {
        out.push([a.id, b.id, c.id]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(points: &[V3], tris: &[[usize; 3]]) -> f64 {
        tris.iter()
            .map(|t| {
                dot(
                    cross(sub(points[t[1]], points[t[0]]), sub(points[t[2]], points[t[0]])),
                    [0.0, 0.0, 1.0],
                ) / 2.0
            })
            .sum()
    }

    /// Every triangle faces +z and has area, and they add up to `want`.
    fn covers(loops: &[Vec<V3>], tris: &[[usize; 3]], want: f64) {
        let points: Vec<V3> = loops.iter().flatten().copied().collect();
        let each: Vec<f64> = tris.iter().map(|t| area(&points, &[*t])).collect();
        assert!(each.iter().all(|&a| a > 0.0), "{each:?}");
        let total = area(&points, tris);
        assert!((total - want).abs() < want * 1e-9, "area {total}, want {want}");
    }

    fn square(x: f64, y: f64, s: f64) -> Vec<V3> {
        vec![[x, y, 0.0], [x + s, y, 0.0], [x + s, y + s, 0.0], [x, y + s, 0.0]]
    }

    const UP: Option<V3> = Some([0.0, 0.0, 1.0]);

    #[test]
    fn triangle_and_quad() {
        let tri = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(triangulate_holes(&[tri], UP), vec![[0, 1, 2]]);
        let q = square(0.0, 0.0, 2.0);
        let t = triangulate_holes(std::slice::from_ref(&q), UP);
        assert_eq!(t.len(), 2);
        assert!((area(&q, &t) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn winding_follows_the_normal() {
        let mut q = square(0.0, 0.0, 1.0);
        q.reverse();
        let t = triangulate_holes(std::slice::from_ref(&q), UP);
        assert!((area(&q, &t) - 1.0).abs() < 1e-9);
        // with no normal given, the outer loop's own winding
        let t = triangulate_holes(std::slice::from_ref(&q), None);
        assert!((area(&q, &t) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn concave_polygon() {
        // an L shape
        let l = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let t = triangulate_holes(std::slice::from_ref(&l), UP);
        assert_eq!(t.len(), 4);
        assert!((area(&l, &t) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn holes_are_cut_out() {
        let loops = vec![square(0.0, 0.0, 10.0), square(2.0, 2.0, 2.0), square(6.0, 6.0, 2.0)];
        covers(&loops, &triangulate_holes(&loops, UP), 92.0);
        // holes wound the same way as the outline, and no normal given
        let mut loops = loops;
        for h in &mut loops[1..] {
            h.reverse();
        }
        covers(&loops, &triangulate_holes(&loops, None), 92.0);
    }

    #[test]
    fn holes_in_a_tilted_plane() {
        // the frame of `holes_are_cut_out` stood up in the x = y plane, facing (1, -1, 0)
        let lift = |l: Vec<V3>| -> Vec<V3> { l.into_iter().map(|p| [p[0], p[0], p[1]]).collect() };
        let loops = vec![lift(square(0.0, 0.0, 10.0)), lift(square(2.0, 2.0, 2.0))];
        let points: Vec<V3> = loops.iter().flatten().copied().collect();
        let t = triangulate_holes(&loops, None);
        let n = [1.0, -1.0, 0.0];
        let facing: Vec<f64> = t
            .iter()
            .map(|t| {
                dot(
                    cross(sub(points[t[1]], points[t[0]]), sub(points[t[2]], points[t[0]])),
                    n,
                ) / 2.0
            })
            .collect();
        assert!(facing.iter().all(|&a| a > 0.0), "{facing:?}");
        // the plane is √2 wider than its x extent
        assert!((facing.iter().sum::<f64>() - 96.0 * 2.0).abs() < 1e-9);
    }

    #[test]
    fn holes_that_touch() {
        // two holes sharing a corner, and a third touching the outline at one of its corners
        let loops = vec![
            vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 5.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            square(2.0, 2.0, 2.0),
            square(4.0, 4.0, 2.0),
            vec![[10.0, 5.0, 0.0], [8.0, 6.0, 0.0], [8.0, 4.0, 0.0]],
        ];
        covers(&loops, &triangulate_holes(&loops, UP), 100.0 - 4.0 - 4.0 - 2.0);
    }

    #[test]
    fn overlapping_holes_fall_back_to_the_outline() {
        let loops = vec![square(0.0, 0.0, 30.0), square(2.0, 2.0, 4.0), square(4.0, 4.0, 4.0)];
        let t = triangulate_holes(&loops, UP);
        // no triangle reaches a hole's corner: they all come from the outline
        assert!(!t.is_empty() && t.iter().flatten().all(|&i| i < 4));
    }

    #[test]
    fn collinear_corners_on_holes() {
        let mut hole = vec![
            [2.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [2.0, 4.0, 0.0],
            [3.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
        ];
        hole.reverse();
        let loops = vec![
            vec![
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            hole,
        ];
        covers(&loops, &triangulate_holes(&loops, UP), 96.0);
    }

    #[test]
    fn tiny_polygons_keep_their_holes() {
        let s = 1e-7;
        let shrink = |l: Vec<V3>| -> Vec<V3> { l.into_iter().map(|p| p.map(|c| c * s)).collect() };
        let loops = vec![shrink(square(0.0, 0.0, 10.0)), shrink(square(2.0, 2.0, 2.0))];
        covers(&loops, &triangulate_holes(&loops, UP), 96.0 * s * s);
    }

    #[test]
    fn huge_polygons_are_fanned() {
        let n = u32::try_from(MAX_CLIP + 1).unwrap();
        let ring: Vec<V3> = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / f64::from(n);
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        let t = triangulate_holes(&[ring], UP);
        assert_eq!(t.len(), MAX_CLIP - 1);
        assert!(t.iter().all(|t| t[0] == 0));
    }

    #[test]
    fn degenerate_input_does_not_panic() {
        assert_eq!(triangulate_holes::<Vec<V3>>(&[], UP), Vec::<[usize; 3]>::new());
        let line = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let _ = triangulate_holes(std::slice::from_ref(&line), Some([0.0, 0.0, 0.0]));
        assert_eq!(triangulate_holes(&[line], UP), Vec::<[usize; 3]>::new());
        // a triangle with an extra vertex halfway along one side still covers its area, with no sliver
        let tri = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]];
        let areas: Vec<f64> = triangulate_holes(std::slice::from_ref(&tri), UP)
            .iter()
            .map(|t| cross(sub(tri[t[1]], tri[t[0]]), sub(tri[t[2]], tri[t[0]]))[2] / 2.0)
            .collect();
        assert!(
            areas.iter().all(|&a| a > 1.0) && (areas.iter().sum::<f64>() - 50.0).abs() < 1e-9,
            "{areas:?}"
        );
        let nan = vec![[f64::NAN; 3]; 5];
        let _ = triangulate_holes(std::slice::from_ref(&nan), UP);
        let _ = triangulate_holes(&[square(0.0, 0.0, 1.0), nan], UP);
        // an outline too short to be one leaves nothing, even with a hole to spare
        let stub = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(
            triangulate_holes(&[stub, square(0.0, 0.0, 1.0)], UP),
            Vec::<[usize; 3]>::new()
        );
    }
}
