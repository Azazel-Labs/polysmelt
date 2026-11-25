//! Quaternions for rotations, `[x, y, z, w]` as glTF orders them, in double precision. A reader whose format lists
//! `w` first reorders before calling in.
//!
//! The arithmetic in each function is in one fixed order, so a result has the same bits whichever reader computes
//! it. Where readers differ in rounding that reaches their output (how Shepperd's method scales its positive-trace
//! branch, how slerp blends two nearly equal rotations), each variant has its own function.

/// No rotation.
pub(crate) const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// The Hamilton product `a·b`: rotating by `b`, then by `a`.
pub(crate) fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// The inverse rotation of unit quaternion `q`.
pub(crate) fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// The same rotation with `w` not negative.
pub(crate) fn positive_w(q: [f64; 4]) -> [f64; 4] {
    if q[3] < 0.0 { q.map(|c| -c) } else { q }
}

/// The rotation by `radians` about axis `axis` (0, 1 or 2 for X, Y or Z).
pub(crate) fn about(axis: usize, radians: f64) -> [f64; 4] {
    let (s, c) = (radians * 0.5).sin_cos();
    let mut q = [0.0, 0.0, 0.0, c];
    q[axis] = s;
    q
}

/// The rotation by `radians` about unit vector `axis`.
pub(crate) fn about_axis(axis: [f64; 3], radians: f64) -> [f64; 4] {
    let (s, c) = (radians * 0.5).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

/// The rotation Euler angles (radians) make: about X, then Y, then Z.
pub(crate) fn euler_xyz([x, y, z]: [f64; 3]) -> [f64; 4] {
    mul(about(2, z), mul(about(1, y), about(0, x)))
}

/// The angle in radians between the rotations of unit quaternions `a` and `b`.
pub(crate) fn angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let d = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>().abs().min(1.0);
    2.0 * d.acos()
}

/// `v` rotated by unit quaternion `q`:`v + 2w(u × v) + 2u × (u × v)`, `u` the vector part.
pub(crate) fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let [a, b, c] = v;
    let t = [2.0 * (y * c - z * b), 2.0 * (z * a - x * c), 2.0 * (x * b - y * a)];
    [
        a + w * t[0] + (y * t[2] - z * t[1]),
        b + w * t[1] + (z * t[0] - x * t[2]),
        c + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

/// The length of `q`, its components squared and summed in order.
pub(crate) fn length(q: [f64; 4]) -> f64 {
    q.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// `q` divided by its length, when that is finite and above `min`.
pub(crate) fn unit_above(q: [f64; 4], min: f64) -> Option<[f64; 4]> {
    let l = length(q);
    (l > min && l.is_finite()).then(|| q.map(|c| c / l))
}

/// `q` at unit length; the identity when its length isn't finite and above 1e-12.
pub(crate) fn unit(q: [f64; 4]) -> [f64; 4] {
    unit_above(q, 1e-12).unwrap_or(IDENTITY)
}

/// `q` divided by its length with no check: not finite for a zero one. For a quaternion that can't be zero, such as
/// one from [`of_rotation`].
pub(crate) fn unit_unchecked(q: [f64; 4]) -> [f64; 4] {
    let l = length(q);
    q.map(|c| c / l)
}

/// `q` at unit length in single precision; the identity for a zero or broken one (any length above 0 counts).
pub(crate) fn normalize(q: [f64; 4]) -> [f32; 4] {
    unit_above(q, 0.0).map_or([0.0, 0.0, 0.0, 1.0], |q| q.map(crate::num::f32_of))
}

/// The 3×3 rotation of unit quaternion `q`, column-major: element (row r, column c) at `c * 3 + r`.
pub(crate) fn matrix(q: [f64; 4]) -> [f64; 9] {
    let [x, y, z, w] = q;
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// The quaternion of a rotation matrix with elements `at(row, col)`, by Shepperd's method, before scaling to unit
/// length. A matrix with finite elements gives a quaternion at least 0.5 long, so it can always be scaled.
pub(crate) fn of_rotation(at: impl Fn(usize, usize) -> f64) -> [f64; 4] {
    let trace = at(0, 0) + at(1, 1) + at(2, 2);
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (at(2, 1) - at(1, 2)) / s,
            (at(0, 2) - at(2, 0)) / s,
            (at(1, 0) - at(0, 1)) / s,
            0.25 * s,
        ]
    } else {
        largest_diagonal(&at)
    }
}

/// [`of_rotation`], its positive-trace branch multiplying by `0.5 / √(trace + 1)` rather than dividing by
/// `2√(trace + 1)`: the same quaternion, rounded differently. The flat mesh's decomposition and the glTF rig reader
/// have always rounded this way, and keep it so their output doesn't change.
pub(crate) fn of_rotation_reciprocal(at: impl Fn(usize, usize) -> f64) -> [f64; 4] {
    let trace = at(0, 0) + at(1, 1) + at(2, 2);
    if trace > 0.0 {
        let k = 0.5 / (trace + 1.0).sqrt();
        [
            (at(2, 1) - at(1, 2)) * k,
            (at(0, 2) - at(2, 0)) * k,
            (at(1, 0) - at(0, 1)) * k,
            0.25 / k,
        ]
    } else {
        largest_diagonal(&at)
    }
}

/// Shepperd's method when the trace isn't positive: from the largest diagonal element.
fn largest_diagonal(at: &impl Fn(usize, usize) -> f64) -> [f64; 4] {
    if at(0, 0) > at(1, 1) && at(0, 0) > at(2, 2) {
        let s = (1.0 + at(0, 0) - at(1, 1) - at(2, 2)).sqrt() * 2.0;
        [
            0.25 * s,
            (at(0, 1) + at(1, 0)) / s,
            (at(0, 2) + at(2, 0)) / s,
            (at(2, 1) - at(1, 2)) / s,
        ]
    } else if at(1, 1) > at(2, 2) {
        let s = (1.0 + at(1, 1) - at(0, 0) - at(2, 2)).sqrt() * 2.0;
        [
            (at(0, 1) + at(1, 0)) / s,
            0.25 * s,
            (at(1, 2) + at(2, 1)) / s,
            (at(0, 2) - at(2, 0)) / s,
        ]
    } else {
        let s = (1.0 + at(2, 2) - at(0, 0) - at(1, 1)).sqrt() * 2.0;
        [
            (at(0, 2) + at(2, 0)) / s,
            (at(1, 2) + at(2, 1)) / s,
            0.25 * s,
            (at(1, 0) - at(0, 1)) / s,
        ]
    }
}

/// The unit quaternion of a rotation matrix with elements `at(row, col)`; the identity for one that isn't finite.
pub(crate) fn from_rotation(at: impl Fn(usize, usize) -> f64) -> [f64; 4] {
    unit(of_rotation(at))
}

/// The unit quaternion of a 3×3 rotation, column-major; the identity for one that isn't finite.
pub(crate) fn from_matrix(m: &[f64; 9]) -> [f64; 4] {
    from_rotation(|r, c| m[c * 3 + r])
}

/// `b`, or `-b` (the same rotation) when that is nearer `a`, and the cosine of the angle between `a` and it.
fn nearer(a: [f64; 4], b: [f64; 4]) -> ([f64; 4], f64) {
    let d: f64 = (0..4).map(|k| a[k] * b[k]).sum();
    if d < 0.0 { (b.map(|c| -c), -d) } else { (b, d) }
}

/// Slerp's weights for `a` and `b` at `t`, `d` the cosine between them; `None` when they are too near (cosine above
/// 0.9995) to divide by the sine of the angle.
fn weights(d: f64, t: f64) -> Option<(f64, f64)> {
    if d > 0.9995 {
        return None;
    }
    let theta = d.acos();
    let sin = theta.sin();
    Some((((1.0 - t) * theta).sin() / sin, (t * theta).sin() / sin))
}

/// Spherical linear interpolation from unit quaternion `a` (`t` 0) to `b` (`t` 1), the short way round. Rotations
/// too near to divide by the sine of the angle between are blended as `(1 - t)·a + t·b`. Not rescaled to unit length.
pub(crate) fn slerp(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let (b, d) = nearer(a, b);
    let (wa, wb) = weights(d, t).unwrap_or((1.0 - t, t));
    std::array::from_fn(|k| wa * a[k] + wb * b[k])
}

/// [`slerp`], nearly equal rotations stepped as `a + (b - a)·t` instead, which rounds differently. MMD motions
/// have always been sampled this way, and keep it so their output doesn't change.
pub(crate) fn slerp_stepped(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let (b, d) = nearer(a, b);
    match weights(d, t) {
        Some((wa, wb)) => std::array::from_fn(|k| wa * a[k] + wb * b[k]),
        None => std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    /// The same rotation: `q` or `-q`.
    fn same_turn(a: [f64; 4], b: [f64; 4]) -> bool {
        close(&a, &b, 1e-9) || close(&a, &b.map(|c| -c), 1e-9)
    }

    /// The 3×3 product `a·b`, column-major.
    fn mat_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
        std::array::from_fn(|k| {
            let (r, c) = (k % 3, k / 3);
            (0..3).map(|i| a[i * 3 + r] * b[c * 3 + i]).sum()
        })
    }

    #[test]
    fn matrices_and_quaternions_round_trip() {
        // Rotations in each of Shepperd's four branches: positive trace, then X, Y or Z largest on the diagonal.
        let turns = [
            unit([0.1, 0.7, -0.2, 0.6]),
            about(0, 3.0),
            about(1, -3.0),
            about(2, 3.1),
            unit([0.3, -0.5, 0.6, -0.2]),
            IDENTITY,
        ];
        for q in turns {
            let m = matrix(q);
            assert!(same_turn(from_matrix(&m), q), "{q:?}");
            assert!(close(&matrix(from_matrix(&m)), &m, 1e-12), "{q:?}");
            let by_rows = from_rotation(|r, c| m[c * 3 + r]);
            assert_eq!(by_rows.map(f64::to_bits), from_matrix(&m).map(f64::to_bits));
            // The reciprocal form is the same quaternion, to rounding.
            assert!(close(
                &unit(of_rotation_reciprocal(|r, c| m[c * 3 + r])),
                &by_rows,
                1e-12
            ));
            let unchecked = unit_unchecked(of_rotation(|r, c| m[c * 3 + r]));
            assert_eq!(unchecked.map(f64::to_bits), by_rows.map(f64::to_bits));
        }
        // The matrix turns a vector as `rotate` does.
        let q = unit([0.2, -0.4, 0.1, 0.9]);
        let m = matrix(q);
        let v = [1.0, -2.0, 0.5];
        let by_matrix: [f64; 3] = std::array::from_fn(|r| (0..3).map(|c| m[c * 3 + r] * v[c]).sum());
        assert!(close(&by_matrix, &rotate(q, v), 1e-12));
        // A broken matrix gives no rotation.
        assert_eq!(from_matrix(&[f64::NAN; 9]), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn products_compose_rotations() {
        let a = about_axis([0.0, 0.0, 1.0], 0.7);
        let b = about_axis([1.0, 0.0, 0.0], -1.1);
        assert!(close(&matrix(mul(a, b)), &mat_mul(&matrix(a), &matrix(b)), 1e-12));
        assert!(close(&mul(a, conjugate(a)), &IDENTITY, 1e-15));
        assert_eq!(about(1, 0.4), about_axis([0.0, 1.0, 0.0], 0.4));
        // A quarter turn about Z takes X to Y.
        assert!(close(
            &rotate(about(2, std::f64::consts::FRAC_PI_2), [1.0, 0.0, 0.0]),
            &[0.0, 1.0, 0.0],
            1e-15
        ));
        let e = euler_xyz([0.3, -0.2, 1.1]);
        assert_eq!(e, mul(about(2, 1.1), mul(about(1, -0.2), about(0, 0.3))));
        assert!((angle(about(0, 0.5), about(0, -0.25)) - 0.75).abs() < 1e-12);
        assert!(angle(e, e.map(|c| -c)) < 1e-6);
    }

    #[test]
    fn normalizing_zero_and_broken_quaternions() {
        assert_eq!(unit([0.0; 4]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(unit([1e-13, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(unit([f64::NAN, 0.0, 0.0, 1.0]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(unit([0.0, 0.0, 0.0, -2.0]), [0.0, 0.0, 0.0, -1.0]);
        assert_eq!(unit_above([1e-13, 0.0, 0.0, 0.0], 0.0), Some([1.0, 0.0, 0.0, 0.0]));
        assert_eq!(unit_above([0.0; 4], 0.0), None);
        assert_eq!(unit_above([f64::INFINITY, 0.0, 0.0, 0.0], 0.0), None);
        assert_eq!(normalize([0.0; 4]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(normalize([0.0, 3.0, 0.0, 4.0]), [0.0, 0.6, 0.0, 0.8]);
        assert!(unit_unchecked([0.0; 4]).iter().all(|c| c.is_nan()));
        assert_eq!(positive_w([1.0, 0.0, 0.0, -0.5]), [-1.0, -0.0, -0.0, 0.5]);
        assert_eq!(positive_w([1.0, 0.0, 0.0, 0.5]), [1.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn slerp_ends_and_takes_the_short_way() {
        let a = about(1, 0.2);
        let b = about(1, 1.4);
        for f in [slerp, slerp_stepped] {
            assert!(close(&f(a, b, 0.0), &a, 1e-15));
            assert!(close(&f(a, b, 1.0), &b, 1e-15));
            // Halfway is half the angle, whichever sign `b` comes in.
            let half = about(1, 0.8);
            assert!(close(&f(a, b, 0.5), &half, 1e-12));
            assert!(close(&f(a, b.map(|c| -c), 0.5), &half, 1e-12));
            // Nearly equal rotations are blended in a straight line, a hair short of unit length.
            let c = about(1, 0.2005);
            assert!(close(&f(a, c, 0.5), &about(1, 0.20025), 1e-7));
        }
        // The two near blends round differently, which is why both are kept.
        let (p, q) = ([0.0, 0.0, 0.6, 0.8], [0.0, 0.0, 0.600_003_5, 0.8]);
        let t: f64 = 0.3;
        let (blended, stepped) = (slerp(p, q, t)[2], slerp_stepped(p, q, t)[2]);
        assert_eq!(blended.to_bits(), ((1.0 - t) * p[2] + t * q[2]).to_bits());
        assert_eq!(stepped.to_bits(), (p[2] + (q[2] - p[2]) * t).to_bits());
        assert_ne!(blended.to_bits(), stepped.to_bits());
    }
}
