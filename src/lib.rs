//! Pure-Rust 3D asset interchange, with glTF as the hub: 60-some formats read into a glTF 2.0
//! [`Document`], and written back out as GLB.
//!
//! # Quick start
//!
//! ```no_run
//! use polysmelt::document::Pose;
//!
//! // Any format Polysmelt reads: told by the extension, or else by the file's first bytes. The
//! // textures and libraries the model names are found beside it.
//! let doc = polysmelt::open("models/robot.fbx")?;
//!
//! // What the file says about itself (format, units, axes, the tool that wrote it), and what
//! // the reader converted to glTF's conventions.
//! let info = doc.source_info();
//! println!("{:?} {:?}, up {:?}", info.format, info.version, info.up_axis);
//!
//! // What it draws, counted from the JSON alone, or flattened into posed triangles.
//! println!("{} triangles", doc.triangles());
//! if let Some(mesh) = doc.flatten(Pose::Rest)? {
//!     println!("{} vertices", mesh.prims.iter().map(|p| p.positions.len()).sum::<usize>());
//! }
//!
//! // And out as a GLB, everything it holds kept.
//! doc.write_glb(std::io::BufWriter::new(std::fs::File::create("robot.glb")?))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! A model in memory is [`read`], and a [`Loader`] chooses what [`open`] and [`read`] don't:
//! what to [`Keep`], where the files a model names are found, and its [`Format`].
//!
//! ```
//! use polysmelt::{Keep, Loader, assets::Memory};
//!
//! let obj = "mtllib box.mtl\nusemtl red\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
//! let mtl = "newmtl red\nKd 1 0 0\n";
//! let doc = Loader::new()
//!     .assets(Memory::new().with("box.mtl", mtl.as_bytes()))
//!     .keep(Keep::ALL.with_animations(false))
//!     .read(obj.as_bytes(), "box.obj")?;
//! assert_eq!(doc.triangles(), 1);
//! assert_eq!(doc.json()["materials"][0]["name"], "red");
//! let glb: Vec<u8> = doc.to_glb()?;
//! # Ok::<(), polysmelt::Error>(())
//! ```
//!
//! # What's here
//!
//! - [`open`], [`read`] and [`Loader`]: the ways in. [`Format`] names what they read: STL, PLY,
//!   `MagicaVoxel`, OBJ, FBX, 3DS, 3MF, USD, glTF, `SketchUp`, BVH, COLLADA, VRM, OFF,
//!   X3D/VRML, DXF, LAS/LAZ, PCD, E57, text point clouds, AMF, `LightWave`, AC3D,
//!   `OpenFlight`, `OpenGEX`, Irrlicht and XGL, and for game and animation assets
//!   `MikuMikuDance` (PMX, PMD, VMD), `DirectX` `.x`, MD2, MD3, MD5, IQM, SMD, `MilkShape` 3D,
//!   `Blitz3D`, ASF/AMC, C3D, TRC, HTR and OGRE, each behind a cargo feature.
//! - [`document`]: the [`Document`] they produce, a glTF 2.0 document (JSON and buffers)
//!   written out as a GLB as it is, flattened into a posed [`Mesh`](mesh::Mesh) on demand, and
//!   pruned to what a [`Keep`] asks for.
//! - [`mesh`]: the flat triangle-list model, written out as a GLB.
//! - [`rig`]: typed views of a document's node tree, skins and animations, for playing and
//!   retargeting motion.
//! - [`source`]: what a file said about itself: units, axes, the tool that wrote it.
//! - [`assets`]: how readers find the textures and material libraries a model refers to.
//! - [`images`]: handles on a document's images, for those a load found but didn't read.
//! - [`gltf`]: GLB packing, and meshopt compression added to or decoded away from a GLB.
//! - [`meshopt`]: the `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams.

/// The shared byte buffer [`mesh::Image`] holds, from the `bytes` crate: cloning or slicing one
/// doesn't copy its bytes.
pub use bytes::Bytes;
/// The JSON a [`Document`] is, from the `serde_json` crate, as [`Document::json`] hands it out.
pub use serde_json;

pub use document::{Document, Keep};
#[cfg(feature = "fs")]
pub use formats::open;
pub use formats::{Format, Loader, read};

/// The README's examples, compiled as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

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
mod formats;
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
