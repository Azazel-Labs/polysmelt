//! Normals smoothed across the polygons that meet at a point, up to a crease angle: for formats that store a
//! smoothing angle rather than normals.

use crate::vec3::{dot, unit};

/// The most polygons around one point that are compared. The corners of a point shared by more stay flat, so a
/// hostile file can't cost the square of its size.
const MAX_AROUND: usize = 512;

/// Each corner's normal, in the order of `corners`. `polys` are (first corner, corner count) spans of `corners`,
/// whose values are point indices below `points`; `face` is each polygon's unit normal. A corner averages the
/// normals of the polygons around its point that `joins(own, other)` accepts and that are within the angle whose
/// cosine `cos(own)` gives; `None` keeps the polygon flat.
pub(crate) fn corner_normals(
    points: usize,
    polys: &[(usize, usize)],
    corners: &[u32],
    face: &[[f32; 3]],
    cos: impl Fn(usize) -> Option<f32>,
    joins: impl Fn(usize, usize) -> bool,
) -> Vec<[f32; 3]> {
    let span = |&(s, n): &(usize, usize)| corners.get(s..s.saturating_add(n)).unwrap_or(&[]);
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

    let mut out = vec![[0.0; 3]; corners.len()];
    for (p, &(s, n)) in polys.iter().enumerate() {
        let own = face.get(p).copied().unwrap_or([0.0, 1.0, 0.0]);
        let cos = cos(p);
        for k in s..s.saturating_add(n).min(corners.len()) {
            let c = corners[k] as usize;
            let list = match (cos, start.get(c), start.get(c + 1)) {
                (Some(_), Some(&a), Some(&b)) if b - a <= MAX_AROUND => &around[a..b],
                _ => {
                    out[k] = own;
                    continue;
                }
            };
            let cos = cos.unwrap_or(1.0);
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
                    sum = [sum[0] + f[0], sum[1] + f[1], sum[2] + f[2]];
                }
            }
            out[k] = unit(sum).unwrap_or(own);
        }
    }
    out
}

#[cfg(test)]
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
}
