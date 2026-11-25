//! Matrices and quaternions for placing nodes. Matrices are column-major, as glTF's and
//! `OpenGEX`'s: `m[col * 4 + row]`. Quaternions are `[x, y, z, w]`.

use serde_json::{Value, json};

use crate::vec3::{cross, dot, unit};

pub(super) type M4 = [f64; 16];

pub(super) const IDENTITY: M4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

pub(super) fn mul(a: &M4, b: &M4) -> M4 {
    let mut m = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            m[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    m
}

pub(super) fn translation(t: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[12..15].copy_from_slice(&t);
    m
}

pub(super) fn scaling(s: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[0] = s[0];
    m[5] = s[1];
    m[10] = s[2];
    m
}

/// The rotation of `angle` radians about unit `axis`, as a quaternion.
pub(super) fn quat_axis(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let (s, c) = (angle * 0.5).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

/// The rotation matrix of a unit quaternion.
pub(super) fn quat_matrix(q: [f64; 4]) -> M4 {
    let [x, y, z, w] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// The quaternion product `a * b`: the rotation `b`, then `a`.
pub(super) fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// A quaternion scaled to unit length; the identity when it has none.
pub(super) fn quat_unit(q: [f64; 4]) -> [f64; 4] {
    let l = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if l > 1e-12 && l.is_finite() {
        q.map(|v| v / l)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

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
    let q = quat_of([x, y, z]);
    let s = s.map(|v| if v.is_finite() { v } else { 1.0 });
    (t, q, s)
}

/// The unit quaternion of a rotation's columns, `w` not negative.
pub(super) fn quat_of(c: [[f64; 3]; 3]) -> [f64; 4] {
    let (m00, m11, m22) = (c[0][0], c[1][1], c[2][2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (c[1][2] - c[2][1]) / s,
            (c[2][0] - c[0][2]) / s,
            (c[0][1] - c[1][0]) / s,
            0.25 * s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * s,
            (c[1][0] + c[0][1]) / s,
            (c[2][0] + c[0][2]) / s,
            (c[1][2] - c[2][1]) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (c[1][0] + c[0][1]) / s,
            0.25 * s,
            (c[2][1] + c[1][2]) / s,
            (c[2][0] - c[0][2]) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (c[2][0] + c[0][2]) / s,
            (c[2][1] + c[1][2]) / s,
            0.25 * s,
            (c[0][1] - c[1][0]) / s,
        ]
    };
    let q = quat_unit(q);
    if q[3] < 0.0 { q.map(|v| -v) } else { q }
}

/// Sets a node's translation, rotation and scale, those that aren't the identity's.
pub(super) fn set_trs(j: &mut Value, (t, q, s): ([f64; 3], [f64; 4], [f64; 3])) {
    if t.iter().any(|&v| v != 0.0) {
        j["translation"] = json!(t);
    }
    if (q[3] - 1.0).abs() > 1e-12 {
        j["rotation"] = json!(q);
    }
    if s.iter().any(|&v| (v - 1.0).abs() > 1e-12) {
        j["scale"] = json!(s);
    }
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

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    #[test]
    fn inverse_undoes_a_placement() {
        let q = quat_unit([0.3, -0.2, 0.5, 0.8]);
        let m = mul(
            &translation([1.0, 2.0, 3.0]),
            &mul(&quat_matrix(q), &scaling([2.0, 3.0, 0.5])),
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

    #[test]
    fn quaternion_products_compose_rotations() {
        let a = quat_axis([0.0, 0.0, 1.0], 0.7);
        let b = quat_axis([1.0, 0.0, 0.0], -1.1);
        assert!(close(
            &quat_matrix(quat_mul(a, b)),
            &mul(&quat_matrix(a), &quat_matrix(b))
        ));
    }
}
