//! Placing nodes by matrices, column-major as glTF's and `OpenGEX`'s: `m[col * 4 + row]`.

use serde_json::Value;

use crate::formats::xform::col::{IDENTITY, M4};
use crate::formats::xform::set_trs;
use crate::quat;
use crate::vec3::{cross, dot, unit};

/// The inverse of a matrix; `None` when it's singular.
pub(super) fn inverse(m: &M4) -> Option<M4> {
    // Cofactors by 2×2 minors of the top and bottom row pairs.
    let a = |c: usize, r: usize| m[c * 4 + r];
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(2, 1) - a(2, 0) * a(0, 1);
    let s2 = a(0, 0) * a(3, 1) - a(3, 0) * a(0, 1);
    let s3 = a(1, 0) * a(2, 1) - a(2, 0) * a(1, 1);
    let s4 = a(1, 0) * a(3, 1) - a(3, 0) * a(1, 1);
    let s5 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);
    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(1, 2) * a(3, 3) - a(3, 2) * a(1, 3);
    let c3 = a(1, 2) * a(2, 3) - a(2, 2) * a(1, 3);
    let c2 = a(0, 2) * a(3, 3) - a(3, 2) * a(0, 3);
    let c1 = a(0, 2) * a(2, 3) - a(2, 2) * a(0, 3);
    let c0 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);
    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if det.abs() < 1e-300 || !det.is_finite() {
        return None;
    }
    let d = 1.0 / det;
    // b(c, r): the inverse's element at column c, row r.
    let rows = [
        [
            (a(1, 1) * c5 - a(2, 1) * c4 + a(3, 1) * c3) * d,
            (-a(1, 0) * c5 + a(2, 0) * c4 - a(3, 0) * c3) * d,
            (a(1, 3) * s5 - a(2, 3) * s4 + a(3, 3) * s3) * d,
            (-a(1, 2) * s5 + a(2, 2) * s4 - a(3, 2) * s3) * d,
        ],
        [
            (-a(0, 1) * c5 + a(2, 1) * c2 - a(3, 1) * c1) * d,
            (a(0, 0) * c5 - a(2, 0) * c2 + a(3, 0) * c1) * d,
            (-a(0, 3) * s5 + a(2, 3) * s2 - a(3, 3) * s1) * d,
            (a(0, 2) * s5 - a(2, 2) * s2 + a(3, 2) * s1) * d,
        ],
        [
            (a(0, 1) * c4 - a(1, 1) * c2 + a(3, 1) * c0) * d,
            (-a(0, 0) * c4 + a(1, 0) * c2 - a(3, 0) * c0) * d,
            (a(0, 3) * s4 - a(1, 3) * s2 + a(3, 3) * s0) * d,
            (-a(0, 2) * s4 + a(1, 2) * s2 - a(3, 2) * s0) * d,
        ],
        [
            (-a(0, 1) * c3 + a(1, 1) * c1 - a(2, 1) * c0) * d,
            (a(0, 0) * c3 - a(1, 0) * c1 + a(2, 0) * c0) * d,
            (-a(0, 3) * s3 + a(1, 3) * s1 - a(2, 3) * s0) * d,
            (a(0, 2) * s3 - a(1, 2) * s1 + a(2, 2) * s0) * d,
        ],
    ];
    // `rows[r][c]` is the inverse's row r, column c.
    let mut out = [0.0; 16];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            out[c * 4 + r] = *v;
        }
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// A placement's translation, rotation and scale when the matrix is one (its axes orthogonal,
/// none of length 0, the bottom row `0 0 0 1`); `None` otherwise.
pub(super) fn decompose(m: &M4) -> Option<([f64; 3], [f64; 4], [f64; 3])> {
    if m.iter().any(|v| !v.is_finite()) || m[3] != 0.0 || m[7] != 0.0 || m[11] != 0.0 || (m[15] - 1.0).abs() > 0.0 {
        return None;
    }
    let cols = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let s = cols.map(|c| dot(c, c).sqrt());
    let orthogonal = s.iter().all(|&v| v > 1e-12)
        && [(0, 1), (0, 2), (1, 2)]
            .iter()
            .all(|&(a, b)| dot(cols[a], cols[b]).abs() <= 1e-6 * s[a] * s[b]);
    orthogonal.then(|| trs(m))
}

/// A matrix's nearest translation, rotation and scale: its axes' lengths as the scale (the
/// first negated for a mirror) and its axes made orthogonal as the rotation.
pub(super) fn trs(m: &M4) -> ([f64; 3], [f64; 4], [f64; 3]) {
    let t = [m[12], m[13], m[14]];
    let mut cols = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let mut s = cols.map(|c| dot(c, c).sqrt());
    if dot(cross(cols[0], cols[1]), cols[2]) < 0.0 {
        s[0] = -s[0];
        cols[0] = cols[0].map(|v| -v);
    }
    // Gram-Schmidt, falling back to the standard axes where an axis has no length.
    let x = unit(cols[0]).unwrap_or([1.0, 0.0, 0.0]);
    let along = dot(cols[1], x);
    let y = unit([0, 1, 2].map(|k| cols[1][k] - along * x[k]))
        .or_else(|| unit(cross([0.0, 0.0, 1.0], x)))
        .or_else(|| unit(cross([0.0, 1.0, 0.0], x)))
        .unwrap_or([0.0, 1.0, 0.0]);
    let z = cross(x, y);
    let q = quat::positive_w(quat::from_rotation(|r, c| [x, y, z][c][r]));
    let s = s.map(|v| if v.is_finite() { v } else { 1.0 });
    (t, q, s)
}

/// Sets a node's placement: its translation, rotation and scale, the nearest when the matrix
/// doesn't come apart into them (glTF's nodes can't shear); whether it did.
pub(super) fn place(j: &mut Value, m: &M4) -> bool {
    if is_identity(m) || m.iter().any(|v| !v.is_finite()) {
        return m.iter().all(|v| v.is_finite());
    }
    if let Some(parts) = decompose(m) {
        set_trs(j, parts);
        true
    } else {
        set_trs(j, trs(m));
        false
    }
}

/// Whether a matrix is the identity.
pub(super) fn is_identity(m: &M4) -> bool {
    m.iter().zip(IDENTITY).all(|(a, b)| (a - b).abs() < 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::xform::col::{mul, rotation, scaling, translation};

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn inverse_undoes_a_placement() {
        let q = quat::unit([0.3, -0.2, 0.5, 0.8]);
        let m = mul(
            &translation([1.0, 2.0, 3.0]),
            &mul(&rotation(q), &scaling([2.0, 3.0, 0.5])),
        );
        let inv = inverse(&m).unwrap();
        assert!(close(&mul(&m, &inv), &IDENTITY));
        let (t, r, s) = decompose(&m).unwrap();
        assert!(close(&t, &[1.0, 2.0, 3.0]));
        assert!(close(&r, &q) || close(&r, &q.map(|v| -v)));
        assert!(close(&s, &[2.0, 3.0, 0.5]));
        assert!(inverse(&scaling([1.0, 0.0, 1.0])).is_none());
        let mut shear = IDENTITY;
        shear[4] = 0.5;
        assert!(decompose(&shear).is_none());
        let (_, q, s) = trs(&scaling([-1.0, 1.0, 1.0]));
        assert!(close(&s, &[-1.0, 1.0, 1.0]) && close(&q, &[0.0, 0.0, 0.0, 1.0]));
    }
}
