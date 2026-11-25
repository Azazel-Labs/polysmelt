//! Animations as animation stacks, each with a layer of curve nodes and curves.

use super::Writer;
use crate::Result;

/// The standard FBX time modes and their rates.
const MODES: [(i32, f64); 15] = [
    (6, 30.0),
    (11, 24.0),
    (10, 25.0),
    (3, 60.0),
    (4, 50.0),
    (5, 48.0),
    (1, 120.0),
    (2, 100.0),
    (12, 1000.0),
    (15, 96.0),
    (16, 72.0),
    (9, 30.0 / 1.001),
    (13, 24.0 / 1.001),
    (17, 60.0 / 1.001),
    (18, 120.0 / 1.001),
];

/// The FBX time mode for `fps` frames a second: a standard one within 0.1% of it, else custom (14) at `fps`.
pub(super) fn time_mode(fps: f64) -> (i32, f64) {
    MODES
        .iter()
        .find(|(_, r)| (fps / r - 1.0).abs() <= 1e-3)
        .map_or((14, fps), |&(m, r)| (m, r))
}

impl Writer<'_> {
    /// Writes the animations.
    #[allow(clippy::unnecessary_wraps, clippy::unused_self, reason = "written in a later step")]
    pub(super) fn animations(&mut self) -> Result<()> {
        Ok(())
    }
}
