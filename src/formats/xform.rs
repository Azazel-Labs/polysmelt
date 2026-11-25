//! Affine transforms as 4x4 matrices (`m[row][col]`, column vectors, as glTF composes), and turning them into a
//! node's translation, rotation and scale.

#![cfg_attr(
    not(any(feature = "fbx", feature = "usd")),
    allow(dead_code, reason = "3DS uses only some of these")
)]

use serde_json::{Value, json};

/// The longest turn between two animation samples, in degrees, when a source's rotations are resampled as glTF
/// quaternions: FBX interpolates Euler angles and USD can too, while glTF takes the shorter way between quaternions,
/// so longer turns are cut into steps short enough that the two agree.
#[cfg(any(feature = "fbx", feature = "usd"))]
pub(super) const MAX_STEP_DEGREES: f64 = 10.0;

pub(super) type M4 = [[f64; 4]; 4];

pub(super) const IDENTITY: M4 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]];

pub(super) fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0; 4]; 4];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

pub(super) fn translate(t: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[0][3] = t[0];
    m[1][3] = t[1];
    m[2][3] = t[2];
    m
}

pub(super) fn scale(s: [f64; 3]) -> M4 {
    let mut m = IDENTITY;
    m[0][0] = s[0];
    m[1][1] = s[1];
    m[2][2] = s[2];
    m
}

pub(super) fn det3(m: &M4) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The inverse of an affine transform.
pub(super) fn inverse(m: &M4) -> M4 {
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
pub(super) fn column_major(m: &M4) -> [f64; 16] {
    std::array::from_fn(|k| m[k % 4][k / 4])
}

/// An affine transform as translation, rotation (a unit quaternion, x y z w) and scale; `exact` is false when it
/// shears, which TRS can't hold.
pub(super) struct Trs {
    pub(super) t: [f64; 3],
    pub(super) r: [f64; 4],
    pub(super) s: [f64; 3],
    pub(super) exact: bool,
}

pub(super) fn decompose(m: &M4) -> Trs {
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
pub(super) fn quaternion(m: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            s / 4.0,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [
            s / 4.0,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [
            (m[0][1] + m[1][0]) / s,
            s / 4.0,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        ]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4.0,
            (m[1][0] - m[0][1]) / s,
        ]
    };
    let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    q.map(|c| c / n)
}

/// Sets a node's transform: TRS where it holds one, else the matrix.
pub(super) fn set_transform(node: &mut Value, m: &M4) {
    let d = decompose(m);
    if !d.exact {
        node["matrix"] = json!(column_major(m));
        return;
    }
    if d.t.iter().any(|&c| c != 0.0) {
        node["translation"] = json!(d.t);
    }
    if (d.r[3] - 1.0).abs() > 1e-12 {
        node["rotation"] = json!(d.r);
    }
    if d.s.iter().any(|&c| (c - 1.0).abs() > 1e-12) {
        node["scale"] = json!(d.s);
    }
}
