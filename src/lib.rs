//! Pure-Rust 3D asset interchange, with glTF as the hub.
//!
//! - [`formats`]: readers for STL, PLY, `MagicaVoxel`, OBJ, FBX, 3DS, 3MF, USD, glTF,
//!   `SketchUp`, BVH, COLLADA, VRM, OFF, X3D/VRML, DXF, LAS/LAZ, PCD, E57, text point clouds,
//!   AMF, `LightWave`, AC3D, `OpenFlight`, `OpenGEX`, Irrlicht and XGL, and for game and animation assets
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

/// The shared byte buffer [`mesh::Image`] holds, from the `bytes` crate: cloning or slicing one
/// doesn't copy its bytes.
pub use bytes::Bytes;

/// Returns an [`Error`] of an [`ErrorKind`] (named bare: `Corrupt`) with a message built from a
/// format string.
macro_rules! bail {
    ($kind:ident, $($t:tt)*) => {
        return Err(err!($kind, $($t)*))
    };
}

/// Bails with an error of the kind unless the condition holds.
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($t:tt)*) => {
        if !$cond {
            bail!($kind, $($t)*)
        }
    };
}

/// An [`Error`] of an [`ErrorKind`] (named bare: `Corrupt`) with a message built from a format
/// string. A constant message isn't allocated.
#[allow(unused_macros)]
macro_rules! err {
    ($kind:ident, $($t:tt)*) => {
        $crate::Error::from_args($crate::ErrorKind::$kind, format_args!($($t)*))
    };
}

pub mod assets;
pub mod document;
mod error;
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

#[allow(unused_imports, reason = "which readers use them depends on the enabled features")]
pub(crate) use error::{Context, Within, printable, shown};
pub use error::{Error, ErrorKind, Result};
