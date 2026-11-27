//! Affine transforms as 4x4 matrices (`m[row][col]`, column vectors, as glTF composes), and turning them into a
//! node's translation, rotation and scale.
//!
//! Readers that keep a matrix as glTF lists it, column by column in a flat `[f64; 16]`, use [`col`]; this module
//! converts between the two layouts with [`from_column_major`] and [`to_column_major`].

use serde_json::{Value, json};

use crate::quat;

/// The longest turn between two animation samples, in degrees, when a source's rotations are resampled as glTF
/// quaternions: FBX interpolates Euler angles and USD can too, while glTF takes the shorter way between quaternions,
/// so longer turns are cut into steps short enough that the two agree.
#[cfg(any(feature = "fbx", feature = "usd"))]
pub(super) const MAX_STEP_DEGREES: f64 = 10.0;

pub(crate) type M4 = [[f64; 4]; 4];

pub(crate) const IDENTITY: M4 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];

pub(crate) fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0; 4]; 4];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

pub(crate) fn translate(t: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[0][3] = t[0];
    m[1][3] = t[1];
    m[2][3] = t[2];
    m
}

pub(crate) fn scale(s: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[0][0] = s[0];
    m[1][1] = s[1];
    m[2][2] = s[2];
    m
}

pub(crate) fn det3(m: &M4) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The inverse of an affine transform.
pub(crate) fn inverse(m: &M4) -> M4 {
    let d = det3(m);
    if d.abs() < 1e-300 {
        return IDENTITY;
    }
    let mut r = IDENTITY;
    for (i, row) in r.iter_mut().enumerate().take(3) {
        for (j, v) in row.iter_mut().enumerate().take(3) {
            let (i1, i2, j1, j2) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
            *v = (m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]) / d;
        }
    }
    let t = [0, 1, 2].map(|i| -(0..3).map(|k| r[i][k] * m[k][3]).sum::<f64>());
    for (row, t) in r.iter_mut().zip(t) {
        row[3] = t;
    }
    r
}

/// A 4x4 matrix as glTF lists it: column by column.
pub(crate) fn to_column_major(m: &M4) -> [f64; 16] {
    std::array::from_fn(|k| m[k % 4][k / 4])
}

/// A 4x4 matrix listed column by column, as glTF does.
pub(crate) fn from_column_major(m: &[f64; 16]) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j * 4 + i]))
}

/// The rotation of unit quaternion `q` (x, y, z, w).
pub(crate) fn rotation(q: [f64; 4]) -> M4 {
    let r = quat::matrix(q);
    let mut m = IDENTITY;
    for (i, row) in m.iter_mut().enumerate().take(3) {
        for (j, v) in row.iter_mut().enumerate().take(3) {
            *v = r[j * 3 + i];
        }
    }
    m
}

/// An affine transform as translation, rotation (a unit quaternion, x y z w) and scale; `exact` is false when it
/// shears, which TRS can't hold.
pub(crate) struct Trs {
    pub(crate) t: [f64; 3],
    pub(crate) r: [f64; 4],
    pub(crate) s: [f64; 3],
    pub(crate) exact: bool,
}

pub(crate) fn decompose(m: &M4) -> Trs {
    let t = [m[0][3], m[1][3], m[2][3]];
    let col = |j: usize| [m[0][j], m[1][j], m[2][j]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut s = [len(col(0)), len(col(1)), len(col(2))];
    if det3(m) < 0.0 {
        s[0] = -s[0];
    }
    if s.iter().any(|v| v.abs() < 1e-300) {
        return Trs {
            t,
            r: [0.0, 0.0, 0.0, 1.0],
            s,
            exact: false,
        };
    }
    let rot: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| m[i][j] / s[j]));
    let mut exact = true;
    for a in 0..3 {
        for b in 0..3 {
            let dot: f64 = (0..3).map(|k| rot[k][a] * rot[k][b]).sum();
            let want = if a == b { 1.0 } else { 0.0 };
            if (dot - want).abs() > 1e-6 {
                exact = false;
            }
        }
    }
    Trs {
        t,
        r: quaternion(&rot),
        s,
        exact,
    }
}

/// A rotation matrix's unit quaternion (x, y, z, w).
pub(crate) fn quaternion(m: &[[f64; 3]; 3]) -> [f64; 4] {
    quat::unit_unchecked(quat::of_rotation(|r, c| m[r][c]))
}

/// Sets a node's transform: TRS where it holds one, else the matrix.
pub(crate) fn set_transform(node: &mut Value, m: &M4) {
    let d = decompose(m);
    if !d.exact {
        node["matrix"] = json!(to_column_major(m));
        return;
    }
    set_trs(node, (d.t, d.r, d.s));
}

/// Sets a node's translation, rotation and scale, leaving out each that is (all but) the identity.
pub(crate) fn set_trs(node: &mut Value, (t, r, s): ([f64; 3], [f64; 4], [f64; 3])) {
    if t.iter().any(|&c| c != 0.0) {
        node["translation"] = json!(t);
    }
    if (r[3] - 1.0).abs() > 1e-12 {
        node["rotation"] = json!(r);
    }
    if s.iter().any(|&c| (c - 1.0).abs() > 1e-12) {
        node["scale"] = json!(s);
    }
}

/// 4x4 matrices as glTF lists them, column by column in a flat array: element (row r, column c) at `c * 4 + r`.
pub(crate) mod col {
    use crate::quat;

    pub(crate) type M4 = [f64; 16];

    pub(crate) const IDENTITY: M4 = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    /// `a·b`.
    pub(crate) fn mul(a: &M4, b: &M4) -> M4 {
        let mut m = [0.0; 16];
        for c in 0..4 {
            for r in 0..4 {
                m[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
            }
        }
        m
    }

    pub(crate) fn translation(t: [f64; 3]) -> M4 {
        let mut m = IDENTITY;
        m[12..15].copy_from_slice(&t);
        m
    }

    pub(crate) fn scaling(s: [f64; 3]) -> M4 {
        let mut m = IDENTITY;
        m[0] = s[0];
        m[5] = s[1];
        m[10] = s[2];
        m
    }

    /// The rotation of unit quaternion `q` (x, y, z, w).
    pub(crate) fn rotation(q: [f64; 4]) -> M4 {
        let r = quat::matrix(q);
        let mut m = IDENTITY;
        for c in 0..3 {
            m[c * 4..c * 4 + 3].copy_from_slice(&r[c * 3..c * 3 + 3]);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_convert_both_ways() {
        let flat: [f64; 16] = std::array::from_fn(|k| f64::from(u8::try_from(k).unwrap()));
        let m = from_column_major(&flat);
        assert_eq!(m[1][0], 1.0);
        assert_eq!(m[0][1], 4.0);
        assert_eq!(m[2][3], 14.0);
        assert_eq!(to_column_major(&m), flat);
        let q = quat::unit([0.2, -0.4, 0.1, 0.9]);
        assert_eq!(to_column_major(&rotation(q)), col::rotation(q));
        let (a, b) = (rotation(q), translate([1.0, 2.0, 3.0]));
        assert_eq!(
            to_column_major(&mul(&a, &b)),
            col::mul(&to_column_major(&a), &to_column_major(&b))
        );
        assert_eq!(to_column_major(&scale([2.0, 3.0, 4.0])), col::scaling([2.0, 3.0, 4.0]));
        assert_eq!(
            to_column_major(&translate([2.0, 3.0, 4.0])),
            col::translation([2.0, 3.0, 4.0])
        );
    }
}
