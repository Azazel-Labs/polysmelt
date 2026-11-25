//! Images a document names without having read them: what a load with
//! [`Keep::image_bytes`](crate::document::Keep::image_bytes) off leaves, as handles from
//! [`Document::images`].
//!
//! Each such image is a glTF image with a `uri`. Its handle says where the image is (a file
//! found through the [`Assets`], an entry in the model's archive, bytes inside the model) and
//! reads it on request; [`Document::embed_images`] reads them all into the document's buffers.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::assets::Assets;
use crate::document::Document;
use crate::gltf::{data_uri, mime_from_name, sniff_mime};
use crate::mesh::Image;
use crate::{Context as _, ErrorKind, Result};

/// Where an image is.
///
/// Non-exhaustive: other places an image can be may be added, so a `match` on one needs a
/// wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Location {
    /// A file, found through the [`Assets`].
    File(PathBuf),
    /// An entry in the model's own archive (a `.usdz` or `.3mf`), by its path there.
    Archive(String),
    /// Inside the model: in a buffer (a GLB's binary chunk), a `data:` URI, or embedded by the
    /// format (an FBX `Video`'s content).
    Embedded,
    /// Named by the model but not found.
    Missing,
}

static EMBEDDED: Location = Location::Embedded;
static MISSING: Location = Location::Missing;

/// An image a document names but hasn't read.
#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "ply",
        feature = "3ds",
        feature = "lwo",
        feature = "ac3d",
        feature = "fbx",
        feature = "usd",
        feature = "3mf",
        feature = "gltf"
    )),
    allow(dead_code, reason = "only readers with textures make references")
)]
#[derive(Debug, Clone)]
pub(crate) struct Reference {
    /// The name as the model wrote it.
    pub name: String,
    pub location: Location,
    /// The undecoded bytes of an image from the model itself (archive entry, embedded).
    pub bytes: Option<Vec<u8>>,
    /// Embedded as it is (a glTF image, whose extensions say its type), rather than turned into
    /// PNG or JPEG by the assets.
    pub raw: bool,
    /// For a file noted without looking for it: whether it's there, looked up the first time
    /// [`location`](Self::location) is asked. `None` when `location` is already settled.
    pub found: Option<std::sync::OnceLock<bool>>,
}

impl Reference {
    /// Where the image is. A file noted without looking is looked for the first time this is
    /// asked, and is [`Location::Missing`] when it isn't there.
    pub fn location(&self) -> &Location {
        match (&self.location, &self.found) {
            (Location::File(p), Some(found)) if !*found.get_or_init(|| p.is_file()) => &MISSING,
            (l, _) => l,
        }
    }
}

/// An image a reader collected for a material: read, or a reference to read later.
#[derive(Debug)]
pub(crate) enum Source {
    Image(Image),
    #[cfg_attr(
        not(any(
            feature = "obj",
            feature = "ply",
            feature = "3ds",
            feature = "lwo",
            feature = "ac3d",
            feature = "fbx",
            feature = "usd",
            feature = "3mf"
        )),
        allow(dead_code, reason = "only readers with textures make references")
    )]
    Ref(Reference, String),
}

impl From<Image> for Source {
    fn from(i: Image) -> Self {
        Source::Image(i)
    }
}

/// The images a reader collects: read and decoded, or with `lazy`, references.
#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "ply",
        feature = "3ds",
        feature = "lwo",
        feature = "ac3d",
        feature = "fbx",
        feature = "usd",
        feature = "3mf"
    )),
    allow(dead_code, reason = "only readers with textures collect images")
)]
#[derive(Debug, Default)]
pub(crate) struct Images {
    pub list: Vec<Source>,
    lazy: bool,
    /// The model's folder, which URIs of files are made relative to.
    #[cfg_attr(
        not(any(
            feature = "obj",
            feature = "ply",
            feature = "3ds",
            feature = "lwo",
            feature = "ac3d",
            feature = "fbx",
            feature = "usd"
        )),
        allow(dead_code, reason = "only readers of texture files use it")
    )]
    dir: PathBuf,
    /// The texture files read so far, so a file named twice is one image.
    files: std::collections::HashMap<PathBuf, Option<usize>>,
}

#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "ply",
        feature = "3ds",
        feature = "lwo",
        feature = "ac3d",
        feature = "fbx",
        feature = "usd",
        feature = "3mf"
    )),
    allow(dead_code, reason = "only readers with textures collect images")
)]
impl Images {
    /// Collects references when `lazy`; `dir` is the model's folder.
    pub fn new(lazy: bool, dir: &Path) -> Self {
        Self {
            list: Vec::new(),
            lazy,
            dir: dir.to_path_buf(),
            files: std::collections::HashMap::new(),
        }
    }

    fn push(&mut self, s: Source) -> usize {
        self.list.push(s);
        self.list.len() - 1
    }

    #[cfg_attr(
        not(any(
            feature = "obj",
            feature = "ply",
            feature = "3ds",
            feature = "lwo",
            feature = "ac3d",
            feature = "fbx",
            feature = "usd"
        )),
        allow(dead_code, reason = "only readers of texture files use it")
    )]
    /// The texture file at `path`, found by the assets for `reference`: its index, or `None`
    /// when it's read and isn't an image the assets can embed.
    pub fn file(&mut self, assets: &dyn Assets, path: PathBuf, reference: &str) -> Option<usize> {
        if let Some(&i) = self.files.get(&path) {
            return i;
        }
        let i = self.read_file(assets, path.clone(), reference);
        self.files.insert(path, i);
        i
    }

    #[cfg_attr(
        not(any(
            feature = "obj",
            feature = "ply",
            feature = "3ds",
            feature = "lwo",
            feature = "ac3d",
            feature = "fbx",
            feature = "usd"
        )),
        allow(dead_code, reason = "only readers of texture files use it")
    )]
    fn read_file(&mut self, assets: &dyn Assets, path: PathBuf, reference: &str) -> Option<usize> {
        if !self.lazy {
            let img = assets.load_image(&path)?;
            return Some(self.push(Source::Image(img)));
        }
        let uri = file_uri(&path, &self.dir);
        let r = Reference {
            name: reference.to_owned(),
            location: Location::File(path),
            bytes: None,
            raw: false,
            found: None,
        };
        Some(self.push(Source::Ref(r, uri)))
    }

    #[cfg_attr(
        not(any(feature = "fbx", feature = "usd", feature = "3mf")),
        allow(dead_code, reason = "only readers of archives and embedded textures use it")
    )]
    /// A texture the model holds itself, at `location`: `convert` makes the image when it's
    /// read now. Its index, or `None` when `convert` makes none.
    pub fn bytes(
        &mut self,
        bytes: Cow<'_, [u8]>,
        reference: &str,
        location: Location,
        convert: impl FnOnce(&[u8]) -> Option<Image>,
    ) -> Option<usize> {
        if !self.lazy {
            let img = convert(&bytes)?;
            return Some(self.push(Source::Image(img)));
        }
        let uri = match &location {
            Location::Archive(entry) => encode(entry),
            _ => encode(&reference.replace('\\', "/")),
        };
        let r = Reference {
            name: reference.to_owned(),
            location,
            bytes: Some(bytes.into_owned()),
            raw: false,
            found: None,
        };
        Some(self.push(Source::Ref(r, uri)))
    }
}

/// Whether encoded image bytes are a gray PNG or JPEG, by the header: a PNG's color type, a JPEG's component count.
#[cfg(any(feature = "obj", feature = "usd"))]
pub(crate) fn gray(b: &[u8]) -> bool {
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        // IHDR comes first; color type 0 is gray, 4 gray with alpha.
        return b.get(12..16) == Some(b"IHDR".as_slice()) && matches!(b.get(25), Some(0 | 4));
    }
    if !b.starts_with(&[0xFF, 0xD8]) {
        return false;
    }
    let mut i = 2usize;
    while i + 4 <= b.len() {
        if b[i] != 0xFF {
            return false;
        }
        let marker = b[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        // A start of frame (not a table): its component count follows the precision, height and width.
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            return b.get(i + 9) == Some(&1);
        }
        if matches!(marker, 0xD9 | 0xDA) {
            return false;
        }
        i = i.saturating_add(2 + usize::from(u16::from_be_bytes([b[i + 2], b[i + 3]])));
    }
    false
}

#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "ply",
        feature = "3ds",
        feature = "lwo",
        feature = "ac3d",
        feature = "fbx",
        feature = "usd"
    )),
    allow(dead_code, reason = "only readers of texture files use it")
)]
/// A URI for a file at `path`: relative to the model's folder when it's under it.
fn file_uri(path: &Path, dir: &Path) -> String {
    let rel = path.strip_prefix(dir).unwrap_or(path);
    let s = rel.to_string_lossy().replace('\\', "/");
    if rel.is_absolute() || s.starts_with('/') || s.as_bytes().get(1) == Some(&b':') {
        format!("file:///{}", encode(s.trim_start_matches('/')).replace("%3A", ":"))
    } else {
        encode(&s)
    }
}

#[cfg_attr(
    not(any(
        feature = "obj",
        feature = "ply",
        feature = "3ds",
        feature = "lwo",
        feature = "ac3d",
        feature = "fbx",
        feature = "usd",
        feature = "3mf"
    )),
    allow(dead_code, reason = "only readers with textures make references")
)]
/// `s` percent-encoded for a URI path: everything but unreserved characters and `/`.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// An image of one of the MIME types glTF images carry.
fn image_of(bytes: crate::Bytes, mime: Option<&str>) -> Option<Image> {
    let mime = sniff_mime(&bytes).or_else(|| mime_from_name(&format!("x.{}", mime?.strip_prefix("image/")?)))?;
    Some(Image { bytes, mime })
}

/// One of a document's images: read, or named and not yet read. See [`Document::images`].
#[derive(Debug, Clone, Copy)]
pub struct ImageHandle<'d> {
    index: usize,
    json: &'d Value,
    reference: Option<&'d Reference>,
    doc: &'d Document,
}

impl<'d> ImageHandle<'d> {
    /// Its index in the glTF `images`.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Its glTF `name`.
    #[must_use]
    pub fn name(&self) -> Option<&'d str> {
        self.json.get("name").and_then(Value::as_str)
    }

    /// Whether it's only named, not yet read: [`load`](Self::load) reads it, and
    /// [`Document::embed_images`] puts it in the document.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.reference.is_some()
    }

    /// The name the model gave it (a path as written, an archive entry, an embedded texture's
    /// file name), for an image not yet read.
    #[must_use]
    pub fn reference(&self) -> Option<&'d str> {
        self.reference.map(|r| r.name.as_str())
    }

    /// Where it is. A file a glTF names is looked for the first time this (or anything that
    /// needs it: [`path`](Self::path), [`mime`](Self::mime), [`load`](Self::load)) is asked,
    /// not when the document is loaded.
    #[must_use]
    pub fn location(&self) -> &'d Location {
        self.reference.map_or(&EMBEDDED, Reference::location)
    }

    /// Its file, when it's in one.
    #[must_use]
    pub fn path(&self) -> Option<&'d Path> {
        match self.location() {
            Location::File(p) => Some(p),
            _ => None,
        }
    }

    /// Its MIME type: as the document says, or from its bytes or its name.
    #[must_use]
    pub fn mime(&self) -> Option<&'d str> {
        if let Some(m) = self.json.get("mimeType").and_then(Value::as_str) {
            return Some(m);
        }
        let r = self.reference?;
        r.bytes
            .as_deref()
            .and_then(sniff_mime)
            .or_else(|| mime_from_name(&r.name))
            .or_else(|| self.path().and_then(|p| mime_from_name(&p.to_string_lossy())))
    }

    /// The image, read now: a file through `assets` (which also turn formats glTF doesn't carry
    /// into PNG or JPEG, as a load that reads images would have), or the bytes the document
    /// holds.
    ///
    /// # Errors
    ///
    /// When its file is missing or can't be read, or it isn't an image `assets` can embed.
    pub fn load(&self, assets: &dyn Assets) -> Result<Image> {
        let i = self.index;
        let Some(r) = self.reference else {
            let bytes = if let Some(v) = self.json.get("bufferView").and_then(Value::as_u64) {
                self.doc.view_shared(crate::num::usize_of(v))?
            } else {
                let uri = self.json.get("uri").and_then(Value::as_str);
                data_uri(uri.with_context(ErrorKind::Corrupt, || format!("image {i} has no bytes"))?)?
                    .1
                    .into()
            };
            return image_of(bytes, self.mime())
                .with_context(ErrorKind::Unsupported, || format!("image {i}: unknown image format"));
        };
        let bytes: Cow<'_, [u8]> = match (r.location(), &r.bytes) {
            (_, Some(b)) => Cow::Borrowed(b),
            (Location::File(p), None) => assets
                .read(p)
                .ok_or_else(|| crate::Error::missing(format!("image {i} can't be read")).at(p))?,
            _ => bail!(MissingAsset, "image {i} ({:?}) wasn't found", r.name),
        };
        let img = if r.raw {
            image_of(crate::Bytes::from(bytes.into_owned()), self.mime())
        } else {
            let name = self.path().map_or_else(|| PathBuf::from(&r.name), Path::to_path_buf);
            assets.image(&bytes, &name)
        };
        img.with_context(ErrorKind::Unsupported, || {
            format!("image {i} ({:?}) isn't an image that can be embedded", r.name)
        })
    }
}

impl Document {
    /// The document's images, read or not. A load with
    /// [`Keep::image_bytes`](crate::document::Keep::image_bytes) off leaves the images it found
    /// unread: each is a glTF image with a `uri` (relative to the model's folder, for a file
    /// under it), and its handle says where it is and reads it.
    pub fn images(&self) -> impl Iterator<Item = ImageHandle<'_>> {
        let list = self.base.get("images").and_then(Value::as_array);
        list.into_iter()
            .flatten()
            .enumerate()
            .map(move |(index, json)| ImageHandle {
                index,
                json,
                reference: self.refs.get(&index),
                doc: self,
            })
    }

    /// Reads every image not yet read into a buffer of its own, so the document stands alone
    /// (as a GLB from [`to_glb`](Self::to_glb) then does). Returns how many couldn't be read;
    /// they stay references by `uri`.
    pub fn embed_images(&mut self, assets: &dyn Assets) -> usize {
        let mut loaded = Vec::new();
        let mut failed = 0;
        for h in self.images().filter(ImageHandle::is_pending) {
            match h.load(assets) {
                Ok(img) => loaded.push((h.index, img)),
                Err(_) => failed += 1,
            }
        }
        for (i, Image { bytes, mime }) in loaded {
            let len = bytes.len();
            let buffer = self.push_buffer(bytes);
            let view = self.push_view(serde_json::json!({ "buffer": buffer, "byteLength": len }));
            if let Some(img) = self.base_mut()["images"][i].as_object_mut() {
                img.remove("uri");
                img.insert("bufferView".into(), view.into());
                img.insert("mimeType".into(), mime.into());
            }
            self.refs.remove(&i);
        }
        failed
    }
}

/// The references a document's images make, by image index.
pub(crate) type Refs = BTreeMap<usize, Reference>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris() {
        assert_eq!(file_uri(Path::new("m/tex/a b.png"), Path::new("m")), "tex/a%20b.png");
        assert_eq!(file_uri(Path::new("other/a.png"), Path::new("m")), "other/a.png");
        assert_eq!(file_uri(Path::new("/abs/a.png"), Path::new("m")), "file:///abs/a.png");
        assert_eq!(
            file_uri(Path::new("C:\\x\\a.png"), Path::new("m")),
            "file:///C:/x/a.png"
        );
    }
}
