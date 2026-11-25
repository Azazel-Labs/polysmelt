//! Skins and morph targets.

use super::Writer;
use super::geometry::Geom;
use crate::Result;

impl Writer<'_> {
    /// Writes node `i`'s skin.
    #[allow(clippy::unnecessary_wraps, clippy::unused_self, reason = "written in a later step")]
    pub(super) fn skin(&mut self, _i: usize, _g: &Geom) -> Result<()> {
        Ok(())
    }

    /// Writes mesh `mesh`'s morph targets, on node `i`.
    #[allow(clippy::unnecessary_wraps, clippy::unused_self, reason = "written in a later step")]
    pub(super) fn morphs(&mut self, _i: usize, _mesh: usize, _g: &Geom) -> Result<()> {
        Ok(())
    }
}
