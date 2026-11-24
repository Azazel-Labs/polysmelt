//! Numeric conversions that lose range or precision on purpose. Each says once why it is
//! safe, so a bare `as` that narrows stands out in review (the crate lints for them).

/// The nearest `f32`: meshes store single precision, and many files hold doubles. A value
/// beyond `f32`'s range becomes infinite.
#[allow(
    clippy::cast_possible_truncation,
    reason = "rounding to single precision is the point"
)]
#[inline]
pub(crate) fn f32_of(x: f64) -> f32 {
    x as f32
}

/// A vertex count or index as a GLB index. Readers stop at a load's triangle limit, at most
/// [`TRIANGLE_CEILING`](crate::mesh::TRIANGLE_CEILING), which keeps every count below 2^32.
#[allow(clippy::cast_possible_truncation, reason = "the triangle limit bounds every count")]
#[inline]
pub(crate) fn u32_of(n: usize) -> u32 {
    debug_assert!(u32::try_from(n).is_ok(), "{n} vertices is past the triangle limit");
    n as u32
}

/// A size, offset or index from a file as `usize`. One too big for this target (past 2^32 on
/// wasm32) saturates, so the range check that follows refuses it instead of the reader taking
/// wrapped-around bytes.
#[inline]
pub(crate) fn usize_of(n: u64) -> usize {
    usize::try_from(n).unwrap_or(usize::MAX)
}
