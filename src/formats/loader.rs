//! The ways in: [`open`] and [`read`], and [`Loader`] for the rest.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use super::{Format, Input};
use crate::assets::{Assets, Memory};
use crate::document::{Document, Keep};
use crate::{Bytes, Result};

/// Reads the model at `path`, all of it. The format is told by the path's extension (in any
/// case), or, when that names none, by the file's first bytes; the files it refers to (MTL
/// libraries, textures, glTF buffers) are found from its folder, as a
/// [`Folder`](crate::assets::Folder) rooted there finds them. [`Loader`] keeps less, finds them
/// elsewhere, or names the format.
///
/// ```no_run
/// let doc = polysmelt::open("models/Duck.DAE")?;
/// doc.write_glb(std::fs::File::create("Duck.glb")?)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// When the file can't be read, its format is none Polysmelt reads (or one whose feature is
/// off), or the model doesn't parse.
#[cfg(feature = "fs")]
pub fn open(path: impl AsRef<Path>) -> Result<Document> {
    Loader::new().open(path)
}

/// Reads a model in memory, all of it. Its format is told by `name`'s extension (in any case),
/// or, when that names none, by its first bytes; `name` can be a bare extension's worth
/// (`"model.glb"`) or empty. The model can't refer to other files: a [`Loader`] with
/// [`Assets`](Loader::assets) finds them.
///
/// The bytes are taken as [`Bytes`]: a `Vec<u8>` or `Bytes` goes in without a copy, and what
/// the document keeps of them (a GLB's binary chunk) is shared rather than copied.
///
/// ```
/// let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
/// let doc = polysmelt::read(&obj[..], "triangle.obj")?;
/// assert_eq!(doc.counts().triangles, 1);
/// # Ok::<(), polysmelt::Error>(())
/// ```
///
/// # Errors
///
/// When the format can't be told or is one whose feature is off, or the model doesn't parse.
pub fn read(bytes: impl Into<Bytes>, name: &str) -> Result<Document> {
    Loader::new().read(bytes, name)
}

/// How to read a model, for what [`open`] and [`read`] don't choose: what to [`keep`](Self::keep),
/// where its files are found ([`assets`](Self::assets)), and its [`format`](Self::format).
/// One loader reads any number of models, from any thread.
///
/// ```no_run
/// use polysmelt::{Format, Keep, Loader};
/// use polysmelt::assets::Memory;
///
/// // A model's meshes alone, without reading its textures.
/// let doc = Loader::new().keep(Keep::LIGHT).open("scene.fbx")?;
///
/// // A model in memory with the files it refers to, by their paths beside it.
/// let files = Memory::new()
///     .with("models/crate.mtl", std::fs::read("crate.mtl")?)
///     .with("models/crate.png", std::fs::read("crate.png")?);
/// let doc = Loader::new().assets(files).read(std::fs::read("crate.obj")?, "models/crate.obj")?;
///
/// // A model whose name says nothing of its format.
/// let doc = Loader::new().format(Format::Stl).read(std::fs::read("download.bin")?, "download.bin")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Default)]
pub struct Loader {
    keep: Keep,
    assets: Option<Arc<dyn Assets + Send + Sync>>,
    format: Option<Format>,
}

impl fmt::Debug for Loader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loader")
            .field("keep", &self.keep)
            .field("assets", &self.assets.is_some())
            .field("format", &self.format)
            .finish()
    }
}

impl Loader {
    /// A loader that keeps everything, finds a file's references from its folder (and a model
    /// in memory's nowhere), and tells the format from the name, then the bytes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps what `keep` says, rather than everything.
    #[must_use]
    pub fn keep(mut self, keep: Keep) -> Self {
        self.keep = keep;
        self
    }

    /// Finds the files a model refers to through `assets`: a
    /// [`Folder`](crate::assets::Folder) elsewhere than the model's own, a
    /// [`Memory`](crate::assets::Memory), or assets of your own.
    #[must_use]
    pub fn assets(mut self, assets: impl Assets + Send + Sync + 'static) -> Self {
        self.assets = Some(Arc::new(assets));
        self
    }

    /// Reads models as `format`, whatever their names say. An extension of the format's own
    /// (`.usdz` for [`Format::Usd`]) still says which form of it a model is in; otherwise its
    /// bytes do.
    #[must_use]
    pub fn format(mut self, format: Format) -> Self {
        self.format = Some(format);
        self
    }

    /// Reads the model at `path`, as [`open`] does with this loader's choices.
    ///
    /// # Errors
    ///
    /// When the file can't be read, its format is none Polysmelt reads (or one whose feature
    /// is off), or the model doesn't parse.
    #[cfg(feature = "fs")]
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Document> {
        // A bare file name's folder is the current one, which its references are found from.
        let path = std::path::absolute(path.as_ref())?;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let folder;
        let assets: &dyn Assets = if let Some(a) = &self.assets {
            &**a
        } else {
            folder = crate::assets::Folder::new(path.parent().unwrap_or(&path));
            &folder
        };
        super::load_as(&path, ext, self.format, assets, self.keep)
    }

    /// Reads a model in memory, as [`read`] does with this loader's choices. `name` is the
    /// model's path among the [`assets`](Self::assets): the files it refers to are found from
    /// its folder.
    ///
    /// # Errors
    ///
    /// When the format can't be told or is one whose feature is off, or the model doesn't
    /// parse.
    pub fn read(&self, bytes: impl Into<Bytes>, name: &str) -> Result<Document> {
        let bytes = bytes.into();
        let path = Path::new(name);
        let input = Input::shared(&bytes, path);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let ext = super::extension_as(input, ext, self.format);
        let none = Memory::new();
        let assets: &dyn Assets = match &self.assets {
            Some(a) => &**a,
            None => &none,
        };
        super::read_input(input, ext, assets, self.keep)
    }
}

#[cfg(all(test, feature = "obj"))]
mod tests {
    use super::*;

    const TRIANGLE: &[u8] = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";

    #[test]
    fn a_model_in_memory_is_read_by_its_name_or_its_format() {
        assert_eq!(read(TRIANGLE, "a.OBJ").unwrap().counts().triangles, 1);
        let told = read(TRIANGLE, "a.bin").unwrap();
        assert_eq!(
            told.counts().triangles,
            1,
            "an unknown extension's format is told by the bytes"
        );
        assert!(read(&b"nothing to tell"[..], "a.bin").is_err());
        assert!(
            read(TRIANGLE, "a.stl").is_err(),
            "a known extension wins over the bytes"
        );
        let doc = Loader::new().format(Format::Obj).read(TRIANGLE, "a.stl").unwrap();
        assert_eq!(doc.counts().triangles, 1);
    }

    #[test]
    fn what_a_loader_keeps_and_where_it_finds_files() {
        let obj = b"mtllib a.mtl\nusemtl red\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let mtl = b"newmtl red\nKd 1 0 0\n";
        let loader = Loader::new().assets(Memory::new().with("m/a.mtl", &mtl[..]));
        let doc = loader.read(&obj[..], "m/a.obj").unwrap();
        assert_eq!(doc.json()["materials"][0]["name"], "red");
        let doc = loader
            .clone()
            .keep(Keep::ALL.with_materials(false))
            .read(&obj[..], "m/a.obj")
            .unwrap();
        assert!(doc.json().get("materials").is_none());
        assert!(format!("{loader:?}").contains("Loader"));
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn a_glb_in_memory_keeps_its_bytes_shared() {
        let glb = Bytes::from(read(TRIANGLE, "t.obj").unwrap().to_glb().unwrap());
        let doc = read(glb.clone(), "m.glb").unwrap();
        let crate::document::Blob::Bytes(kept) = &doc.buffers[0] else {
            panic!("the binary chunk is held as bytes")
        };
        let (all, part) = (glb.as_ptr_range(), kept.as_ptr_range());
        assert!(
            all.start <= part.start && part.end <= all.end,
            "and shared with the GLB's"
        );
    }
}
