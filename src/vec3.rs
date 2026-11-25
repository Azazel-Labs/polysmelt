//! Three-component vector math for the readers, in `f32` or `f64` as each reader works. Every function does its
//! operations in one fixed order, so a result has the same bits whichever reader computes it.

use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// A float type the vector functions work in: `f32` or `f64`.
pub(crate) trait Real:
    Copy + PartialOrd + Add<Output = Self> + AddAssign + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    const ZERO: Self;
    /// The length a vector must exceed for [`unit`] to give it a direction.
    const TINY: Self;
    fn sqrt(self) -> Self;
    fn is_finite(self) -> bool;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const TINY: Self = 1e-20;
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const TINY: Self = 1e-12;
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

#[inline]
pub(crate) fn add<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn sub<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale<T: Real>(a: [T; 3], s: T) -> [T; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn dot<T: Real>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn cross<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn length<T: Real>(a: [T; 3]) -> T {
    dot(a, a).sqrt()
}

/// `v` divided by its length, when that is finite and above `min`.
#[inline]
pub(crate) fn unit_above<T: Real>(v: [T; 3], min: T) -> Option<[T; 3]> {
    let len = length(v);
    (len > min && len.is_finite()).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

/// `v` scaled to length one; `None` when it is too short to have a direction (1e-20 in `f32`, 1e-12 in `f64`) or
/// not finite.
#[inline]
pub(crate) fn unit<T: Real>(v: [T; 3]) -> Option<[T; 3]> {
    unit_above(v, T::TINY)
}

/// `v` divided by its length, or `fallback` when the length isn't above zero. An infinite length still divides.
#[inline]
pub(crate) fn unit_or<T: Real>(v: [T; 3], fallback: [T; 3]) -> [T; 3] {
    let len = length(v);
    if len > T::ZERO {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Adds the edge from `a` to `b` to Newell's normal `n`.
#[inline]
pub(crate) fn newell_edge<T: Real>(n: &mut [T; 3], a: [T; 3], b: [T; 3]) {
    n[0] += (a[1] - b[1]) * (a[2] + b[2]);
    n[1] += (a[2] - b[2]) * (a[0] + b[0]);
    n[2] += (a[0] - b[0]) * (a[1] + b[1]);
}

/// Newell's normal of a closed polygon: along its normal, twice its area long. It copes with concave and slightly
/// bent polygons. The edges are summed in order, the closing one last.
#[inline]
pub(crate) fn newell<T: Real>(points: impl IntoIterator<Item = [T; 3]>) -> [T; 3] {
    let mut n = [T::ZERO; 3];
    let mut points = points.into_iter();
    let Some(first) = points.next() else { return n };
    let mut prev = first;
    for p in points {
        newell_edge(&mut n, prev, p);
        prev = p;
    }
    newell_edge(&mut n, prev, first);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(dot([1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0);
        assert_eq!(unit([0.0, 3.0, 4.0]), Some([0.0, 0.6, 0.8]));
        assert_eq!(unit([0.0f32; 3]), None);
        assert_eq!(unit([f64::INFINITY, 0.0, 0.0]), None);
        assert_eq!(unit_or([0.0f32; 3], [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_eq!(unit_above([1e-15, 0.0, 0.0], 0.0), Some([1.0, 0.0, 0.0]));
    }

    #[test]
    fn newell_of_a_square() {
        let square = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]];
        assert_eq!(newell(square), [0.0, 0.0, 8.0]);
        assert_eq!(newell(square.into_iter().rev()), [0.0, 0.0, -8.0]);
        assert_eq!(newell(std::iter::empty::<[f32; 3]>()), [0.0; 3]);
    }
}
