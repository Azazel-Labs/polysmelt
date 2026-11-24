//! Pure-Rust 3D asset interchange, with glTF as the hub.
//!
//! - [`formats`]: readers for STL, PLY, `MagicaVoxel`, OBJ, FBX, 3DS, 3MF, USD, glTF,
//!   `SketchUp`, BVH, COLLADA, VRM, OFF, X3D/VRML, DXF, LAS/LAZ, PCD, E57, text point clouds,
//!   AMF, `LightWave`, AC3D, `OpenFlight` and `OpenGEX`, and for game and animation assets
//!   `MikuMikuDance` (PMX, PMD, VMD), `DirectX` `.x`, MD2, MD3, MD5, IQM, SMD, `MilkShape` 3D,
//!   `Blitz3D`, ASF/AMC, C3D, TRC, HTR and OGRE, each behind a cargo feature, producing a
//!   [`Document`](document::Document).
//! - [`document`]: that common form, a glTF 2.0 document (JSON and buffers) written out as a GLB
//!   as it is, flattened into a posed [`Mesh`](mesh::Mesh) on demand, and pruned to what a
//!   [`Keep`](document::Keep) asks for.
//! - [`mesh`]: the flat triangle-list model, written out as a GLB.
//! - [`rig`]: typed views of a document's node tree, skins and animations, for playing and
//!   retargeting motion.
//! - [`assets`]: how readers find the textures and material libraries a model refers to.
//! - [`images`]: handles on a document's images, for those a load found but didn't read.
//! - [`gltf`]: GLB packing, and meshopt compression added to or decoded away from a GLB.
//! - [`meshopt`]: the `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams.

use std::fmt::Display;

/// The shared byte buffer [`mesh::Image`] holds, from the `bytes` crate: cloning or slicing one
/// doesn't copy its bytes.
pub use bytes::Bytes;

/// Returns an [`Error::InvalidInput`] built from a format string.
macro_rules! bail {
    ($($t:tt)*) => {
        return Err($crate::Error::InvalidInput($crate::printable(format!($($t)*))))
    };
}

/// Bails unless the condition holds.
macro_rules! ensure {
    ($cond:expr, $($t:tt)*) => {
        if !$cond {
            bail!($($t)*)
        }
    };
}

/// An [`Error::InvalidInput`] built from a format string.
#[allow(unused_macros)]
macro_rules! invalid {
    ($($t:tt)*) => {
        $crate::Error::InvalidInput($crate::printable(format!($($t)*)))
    };
}

/// Text from a file as an error message shows it: quoted, escaped as Rust writes a string
/// literal (so control characters and NULs come out printable), and cut to 32 characters.
#[allow(dead_code, reason = "which readers use it depends on the enabled features")]
pub(crate) fn shown(text: &[u8]) -> String {
    const MOST: usize = 32;
    let text = String::from_utf8_lossy(text);
    let mut cut: String = text.chars().take(MOST).collect();
    if text.chars().nth(MOST).is_some() {
        cut.push('…');
    }
    format!("{cut:?}")
}

/// An error message with any control characters escaped, so file content that slipped into
/// it can't garble a terminal or a report.
pub(crate) fn printable(message: String) -> String {
    if !message.chars().any(char::is_control) {
        return message;
    }
    message
        .chars()
        .map(|c| {
            if c.is_control() {
                c.escape_debug().to_string()
            } else {
                c.to_string()
            }
        })
        .collect()
}

pub mod assets;
pub mod document;
mod flatten;
pub mod formats;
mod gaussian;
pub mod gltf;
pub mod images;
mod json;
mod keep;
mod material;
pub mod mesh;
pub mod meshopt;
mod num;
pub mod rig;
pub mod source;

#[cfg(any(
    feature = "webp",
    feature = "skp",
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds",
    feature = "amf"
))]
mod png;
#[cfg(any(
    feature = "tga",
    feature = "bmp",
    feature = "gif",
    feature = "tiff",
    feature = "psd",
    feature = "dds"
))]
mod raster;
#[cfg(test)]
mod test_rng;
#[cfg(feature = "webp")]
mod webp;

/// What can go wrong reading or writing an asset.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input is malformed or uses something unsupported.
    #[error("{0}")]
    InvalidInput(String),
    /// Reading a file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A result with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Turns a missing value or a foreign error into an [`Error::InvalidInput`] that says what was
/// being done.
#[allow(dead_code)]
pub(crate) trait Context<T> {
    fn context(self, what: impl Display) -> Result<T>;
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    fn context(self, what: impl Display) -> Result<T> {
        self.ok_or_else(|| Error::InvalidInput(printable(what.to_string())))
    }
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T> {
        self.ok_or_else(|| Error::InvalidInput(printable(what().to_string())))
    }
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl Display) -> Result<T> {
        self.map_err(|e| Error::InvalidInput(printable(format!("{what}: {e}"))))
    }
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T> {
        self.map_err(|e| Error::InvalidInput(printable(format!("{}: {e}", what()))))
    }
}

/// Foreign errors met while parsing, as invalid input.
macro_rules! invalid_from {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Error::InvalidInput(e.to_string())
            }
        })*
    };
}

invalid_from!(
    std::num::ParseIntError,
    std::num::ParseFloatError,
    std::num::TryFromIntError,
    std::array::TryFromSliceError,
    std::str::Utf8Error,
    serde_json::Error,
);
#[cfg(any(feature = "3mf", feature = "usd"))]
invalid_from!(zip::result::ZipError);
#[cfg(feature = "3mf")]
invalid_from!(quick_xml::Error);
