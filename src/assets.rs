//! Finding the files a model refers to (MTL libraries, textures) and turning texture bytes into
//! embeddable images.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::mesh::Image;

/// Where a reader finds the files a model refers to, and how it turns texture files into
/// images it can embed.
pub trait Assets {
    /// The file a model in `model_dir` refers to as `reference`: a relative path, a bare name,
    /// or another machine's absolute path.
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf>;

    /// The bytes of a file [`find`](Self::find) returned, or of one a model names exactly (a
    /// glTF buffer, a USD sublayer). Reads the file by default.
    #[cfg(feature = "fs")]
    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>> {
        std::fs::read(path).ok().map(Cow::Owned)
    }

    /// The bytes of a file [`find`](Self::find) returned, or of one a model names exactly (a
    /// glTF buffer, a USD sublayer). Without the `fs` feature there is no default: the bytes
    /// come from wherever the assets keep them.
    #[cfg(not(feature = "fs"))]
    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>>;

    /// Texture bytes as an embeddable image; `name` is the file's name or path, for its format
    /// (TGA has no signature). By default PNG and JPEG pass, and WebP, TGA, BMP, GIF, TIFF, PSD
    /// and DDS are re-encoded as PNG, each with the feature of its name. Override to convert
    /// others.
    fn image(&self, bytes: &[u8], name: &Path) -> Option<Image> {
        let image = Image::png_or_jpeg(bytes).or_else(|| Image::webp(bytes));
        #[cfg(any(
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        ))]
        let image = image.or_else(|| crate::raster::decode(bytes, name));
        #[cfg(not(any(
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        )))]
        let _ = name;
        image
    }

    /// [`Assets::image`] for bytes already held as [`Bytes`](crate::Bytes) (a file read, or
    /// part of one). By default the same as `image`; [`Folder`] keeps PNG and JPEG bytes as they
    /// are, without a copy.
    fn image_shared(&self, bytes: crate::Bytes, name: &Path) -> Option<Image> {
        self.image(&bytes, name)
    }

    /// [`read`](Self::read) as [`Bytes`](crate::Bytes), for bytes a document keeps (a glTF
    /// buffer). By default the bytes `read` returns, copied only when they're borrowed; [`Memory`]
    /// shares the bytes it holds.
    fn read_shared(&self, path: &Path) -> Option<crate::Bytes> {
        self.read(path).map(|b| match b {
            Cow::Owned(v) => v.into(),
            Cow::Borrowed(s) => crate::Bytes::copy_from_slice(s),
        })
    }

    /// A texture file read as an embeddable image.
    fn load_image(&self, path: &Path) -> Option<Image> {
        self.image(&self.read(path)?, path)
    }

    /// A color texture for a model at `model` that names none, by naming convention, from the
    /// model's part of the pack: its folder's parent and below (`Models/` and `Textures/` side by
    /// side), so a sprite of the same name in another kit of a bundle isn't taken, and only from
    /// the model's folder, above it, or a folder named for textures, so a sprite or preview of the
    /// model beside it isn't either. `None` by default.
    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        let _ = model;
        None
    }

    /// The files directly in `dir` whose names end with `suffix` (any case), in name order, for a model whose
    /// side files aren't named by the model itself (OGRE material scripts). Empty by default.
    fn files_with_suffix(&self, dir: &Path, suffix: &str) -> Vec<PathBuf> {
        let _ = (dir, suffix);
        Vec::new()
    }
}

/// The most files [`Assets::files_with_suffix`] lists.
const SUFFIX_LIMIT: usize = 256;

/// Assets in a folder tree laid out like the pack a model came in: references are tried relative
/// to the model (`../Textures/x.png` too, inside `root`), then found under `root` by file name (or
/// the same name in another image format), the one nearest the model. A model naming no color
/// texture takes one by name from its part of the pack: `Barrel.png` or `Barrel_Albedo.png` for
/// `Barrel.obj`. Whichever order the folders list their files in, the same file is found.
///
/// The tree is listed once, the first time a search needs it, and that listing is kept: a model
/// with many textures costs one walk of the pack, not one for each. Make a new `Folder` (one
/// for each model, as a rule) to see files added since.
#[cfg(feature = "fs")]
#[derive(Debug, Clone)]
pub struct Folder {
    /// The top of the tree searched.
    pub root: PathBuf,
    /// The tree's files, in [`walk_files`] order, with their lowercase names, and whether
    /// listing them stopped short.
    files: std::sync::OnceLock<(Vec<(PathBuf, String)>, bool)>,
}

#[cfg(feature = "fs")]
impl Folder {
    /// A folder tree rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            files: std::sync::OnceLock::new(),
        }
    }

    /// The files under `under` (the root, or a folder inside it), in [`walk_files`] order, with
    /// their lowercase names.
    fn files_under(&self, under: &Path) -> Cow<'_, [(PathBuf, String)]> {
        let list = |root: &Path| {
            let mut files = Vec::new();
            let short = walk_files(root, &mut |p| {
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                files.push((p.to_path_buf(), name));
            });
            (files, short)
        };
        let (files, short) = self.files.get_or_init(|| list(&self.root));
        if under == self.root {
            Cow::Borrowed(files)
        } else if *short {
            // The listing of the whole tree may have stopped before reaching `under`.
            Cow::Owned(list(under).0)
        } else {
            Cow::Owned(files.iter().filter(|(p, _)| p.starts_with(under)).cloned().collect())
        }
    }
}

#[cfg(feature = "fs")]
impl Assets for Folder {
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf> {
        forms(reference)
            .iter()
            .find_map(|r| find_texture(r, model_dir, &self.root, || self.files_under(&self.root)))
    }

    fn image_shared(&self, bytes: crate::Bytes, name: &Path) -> Option<Image> {
        Image::from_png_or_jpeg(bytes.clone()).or_else(|| self.image(&bytes, name))
    }

    fn load_image(&self, path: &Path) -> Option<Image> {
        // The file as read becomes the image's bytes, rather than a copy of them.
        let bytes = self.read(path)?.into_owned();
        self.image_shared(bytes.into(), path)
    }

    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        // The model's folder's parent, when that's still inside the pack.
        let home = model
            .parent()
            .and_then(Path::parent)
            .filter(|h| h.starts_with(&self.root))
            .unwrap_or(&self.root);
        let images = self
            .files_under(home)
            .iter()
            .filter(|(_, name)| image_parts(name).is_some())
            .map(|(p, _)| p.clone())
            .collect::<Vec<_>>();
        color_texture_by_name(model, home, images)
    }

    fn files_with_suffix(&self, dir: &Path, suffix: &str) -> Vec<PathBuf> {
        let suffix = suffix.to_lowercase();
        let dir = if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut out: Vec<PathBuf> = entries
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .filter(|e| e.file_name().to_string_lossy().to_lowercase().ends_with(&suffix))
            .map(|e| e.path())
            .collect();
        out.sort();
        out.truncate(SUFFIX_LIMIT);
        out
    }
}

/// Files held in memory by path: assets for loading without file access, found the way
/// [`Folder`] finds them on disk. Paths are relative to the top of the pack and `/`-separated
/// (`\` and a leading `./` or `/` read the same). A reference is tried relative to the model,
/// then by file name, then as the same name in another image format, the one nearest the model;
/// a model naming no color texture takes one by name.
///
/// Files are held as [`Bytes`](crate::Bytes): a `Vec<u8>` or `Bytes` goes in without a copy, a
/// clone of the `Memory` shares them, and what a document keeps of them (a glTF buffer, a PNG
/// texture) is shared too.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    files: BTreeMap<String, crate::Bytes>,
}

impl Memory {
    /// No files.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the file at `path`.
    pub fn insert(&mut self, path: &str, bytes: impl Into<crate::Bytes>) -> &mut Self {
        self.files.insert(memory_key(path), bytes.into());
        self
    }

    /// With the file at `path` added (or replaced).
    #[must_use]
    pub fn with(mut self, path: &str, bytes: impl Into<crate::Bytes>) -> Self {
        self.insert(path, bytes);
        self
    }

    /// The file at `path`, as held: cloning it shares its bytes.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&crate::Bytes> {
        self.files.get(&memory_key(path))
    }

    /// The files' paths, normalized, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    /// The file one form of a reference names (see [`forms`]): relative to the model, then by file name, then the
    /// same name in another image format.
    fn find_form(&self, cleaned: &str, model_dir: &Path) -> Option<PathBuf> {
        if !cleaned.starts_with('/') && !cleaned.contains(':') {
            let rel = memory_key(&format!("{}/{cleaned}", model_dir.to_string_lossy()));
            if self.files.contains_key(&rel) {
                return Some(rel.into());
            }
        }
        let name = cleaned.rsplit('/').next()?.to_lowercase();
        let files: Vec<(&Path, String)> = self
            .files
            .keys()
            .map(|k| {
                (
                    Path::new(k.as_str()),
                    k.rsplit('/').next().unwrap_or_default().to_lowercase(),
                )
            })
            .collect();
        let by_name = files.iter().filter(|(_, n)| *n == name).map(|(p, _)| p.to_path_buf());
        nearest(model_dir, by_name).or_else(|| {
            let files = files.iter().map(|(p, n)| (*p, n.as_str()));
            same_name_other_format(&name, model_dir, files)
        })
    }

    fn images(&self) -> impl Iterator<Item = PathBuf> + '_ {
        self.files
            .keys()
            .map(Path::new)
            .filter(|p| image_stem(p).is_some())
            .map(Path::to_path_buf)
    }
}

impl Assets for Memory {
    fn find(&self, reference: &str, model_dir: &Path) -> Option<PathBuf> {
        forms(reference).iter().find_map(|r| self.find_form(r, model_dir))
    }

    fn read(&self, path: &Path) -> Option<Cow<'_, [u8]>> {
        self.files
            .get(&memory_key(&path.to_string_lossy()))
            .map(|b| Cow::Borrowed(&b[..]))
    }

    fn read_shared(&self, path: &Path) -> Option<crate::Bytes> {
        self.files.get(&memory_key(&path.to_string_lossy())).cloned()
    }

    fn image_shared(&self, bytes: crate::Bytes, name: &Path) -> Option<Image> {
        Image::from_png_or_jpeg(bytes.clone()).or_else(|| self.image(&bytes, name))
    }

    fn load_image(&self, path: &Path) -> Option<Image> {
        // The file as held becomes the image's bytes, rather than a copy of them.
        self.image_shared(self.read_shared(path)?, path)
    }

    fn color_texture_for(&self, model: &Path) -> Option<PathBuf> {
        // The model's folder's parent: `a/b` for `a/b/Models/Barrel.obj`.
        let key = memory_key(&model.to_string_lossy());
        let home = Path::new(key.rsplitn(3, '/').nth(2).unwrap_or_default());
        let images = self.images().filter(|p| p.starts_with(home));
        color_texture_by_name(Path::new(&key), home, images)
    }

    fn files_with_suffix(&self, dir: &Path, suffix: &str) -> Vec<PathBuf> {
        let dir = memory_key(&dir.to_string_lossy());
        let suffix = suffix.to_lowercase();
        self.files
            .keys()
            .filter(|k| {
                let (parent, name) = k.rsplit_once('/').unwrap_or(("", k.as_str()));
                parent == dir && name.to_lowercase().ends_with(&suffix)
            })
            .take(SUFFIX_LIMIT)
            .map(PathBuf::from)
            .collect()
    }
}

/// A model's reference to a file, trimmed, out of the quotes some writers put around a path with spaces, with `/`
/// between its parts, and a `file:` URI as the path it names (`file:///C:/Old%20Path/tex.png` is
/// `C:/Old Path/tex.png`).
fn cleaned(reference: &str) -> String {
    let r = reference.trim();
    let r = ['"', '\'']
        .iter()
        .find_map(|&q| r.strip_prefix(q)?.strip_suffix(q))
        .map_or(r, str::trim);
    let r = r.replace('\\', "/");
    let Some(rest) = r.get(..5).filter(|p| p.eq_ignore_ascii_case("file:")).map(|_| &r[5..]) else {
        return r;
    };
    // A URI is percent-encoded. After `file:` come two slashes and a host (none, or `localhost`, for this
    // machine), then the path; some writers leave the slashes out.
    let rest = crate::gltf::percent_decode(rest);
    let path = match rest.strip_prefix("//") {
        Some(after) => after.split_once('/').map_or("", |(_host, path)| path),
        None => rest.trim_start_matches('/'),
    };
    // `C:/...` is a path as it stands, and so is one that was relative; anything else was a path from the root.
    if path.as_bytes().get(1) == Some(&b':') || !rest.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

/// The path a model's reference names, as every reader takes it: [`cleaned`], and percent-decoded, for the
/// formats whose references are URIs (COLLADA, glTF, X3D).
#[cfg_attr(
    not(feature = "dae"),
    allow(dead_code, reason = "only the COLLADA reader needs the path itself")
)]
pub(crate) fn local_path(uri: &str) -> String {
    let c = cleaned(uri);
    if uri.trim().get(..5).is_some_and(|p| p.eq_ignore_ascii_case("file:")) {
        return c;
    }
    crate::gltf::percent_decode(&c)
}

/// The forms a reference is looked for in, in turn: as the model wrote it ([`cleaned`]), then percent-decoded
/// when that differs (`my%20tex.png` is `my tex.png` to a format that writes URIs, and a file's own name to one
/// that writes paths: the file that exists says which).
fn forms(reference: &str) -> Vec<String> {
    let c = cleaned(reference);
    if c.is_empty() {
        return Vec::new();
    }
    let decoded = crate::gltf::percent_decode(&c);
    if decoded == c { vec![c] } else { vec![c, decoded] }
}

/// A path as a [`Memory`] keys it: `/`-separated, without empty or `.` parts, and `..` stepping
/// up (no further than the top).
pub(crate) fn memory_key(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

/// A model's color texture by its name, from `images` in the pack's part under `home`: one in the
/// model's folder or above it, or in a folder named for textures (`Textures/`, `Materials/`,
/// `maps/`), never a sprite or preview of the model from beside it. A name that says it's the
/// color wins over one that says nothing (`Barrel.png`), then the nearest, then the exact name.
fn color_texture_by_name(model: &Path, home: &Path, images: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let stem = model.file_stem()?.to_string_lossy().to_lowercase();
    if stem.len() < 3 || stem.contains('@') {
        return None;
    }
    let model_dir = model.parent().unwrap_or(Path::new(""));
    // (role, not the exact name, where, path): the least is the pick.
    let mut best: Option<(u8, Closeness, bool, PathBuf)> = None;
    for p in images {
        let Some(file_stem) = image_stem(&p) else { continue };
        let Some(rest) = file_stem.strip_prefix(&stem) else {
            continue;
        };
        if !(rest.is_empty() || rest.starts_with(['_', '-', ' ', '.'])) {
            continue;
        }
        let role = match texture_role(&file_stem) {
            Some(Role::Color) => 0,
            None => 1,
            Some(Role::Other) => continue,
        };
        let dir = p.parent().unwrap_or(Path::new(""));
        let texture_folder = dir
            .strip_prefix(home)
            .unwrap_or(dir)
            .components()
            .any(|c| is_texture_folder(&c.as_os_str().to_string_lossy()));
        if !texture_folder && !model_dir.starts_with(dir) {
            continue;
        }
        let key = (role, closeness(model_dir, &p), !rest.is_empty(), p);
        if best.as_ref().is_none_or(|b| key < *b) {
            best = Some(key);
        }
    }
    best.map(|(.., p)| p)
}

/// A folder whose name says it holds textures.
fn is_texture_folder(name: &str) -> bool {
    let name = name.to_lowercase();
    ["tex", "material", "map", "skin", "albedo", "diffuse", "color", "colour"]
        .iter()
        .any(|w| name.contains(w))
}

/// How near a file is to a folder: the folders they share from the top (more is nearer), then
/// how deep the file is (less is nearer).
type Closeness = (std::cmp::Reverse<usize>, usize);

fn closeness(dir: &Path, p: &Path) -> Closeness {
    let parent = p.parent().unwrap_or(Path::new(""));
    let lower = |c: std::path::Component<'_>| c.as_os_str().to_string_lossy().to_lowercase();
    let shared = parent
        .components()
        .zip(dir.components())
        .take_while(|(a, b)| lower(*a) == lower(*b))
        .count();
    (std::cmp::Reverse(shared), parent.components().count())
}

/// Of `candidates`, the one nearest the model's folder `dir`, then first by path: the same pick
/// whatever order they come in.
fn nearest(dir: &Path, candidates: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    candidates
        .into_iter()
        .map(|p| (closeness(dir, &p), p))
        .min()
        .map(|(_, p)| p)
}

/// An image with the same stem as `name` (lowercase) in another format (`rock.png` for
/// `rock.tga`, as packs often ship), or in any format when `name` has none (Blender's FBX
/// exporter names packed images `x.fbm\rock`), preferring PNG and JPEG, then the nearest.
///
/// `files` are paths with their lowercase file names.
fn same_name_other_format<'a>(
    name: &str,
    dir: &Path,
    files: impl IntoIterator<Item = (&'a Path, &'a str)>,
) -> Option<PathBuf> {
    let stem = match name.rsplit_once('.') {
        Some((stem, ext)) if IMAGE_EXTS.contains(&ext) => stem,
        Some(_) => return None,
        None => name,
    };
    files
        .into_iter()
        .filter_map(|(p, n)| {
            let (s, ext) = image_parts(n)?;
            let other = !matches!(ext, "png" | "jpg" | "jpeg");
            (s == stem).then(|| (other, closeness(dir, p), p))
        })
        .min()
        .map(|(.., p)| p.to_path_buf())
}

/// A lowercase file name's stem and extension, when the extension is an image's: what
/// [`image_stem`] reads from a path.
fn image_parts(name: &str) -> Option<(&str, &str)> {
    let (stem, ext) = name.rsplit_once('.')?;
    (!stem.is_empty() && IMAGE_EXTS.contains(&ext)).then_some((stem, ext))
}

/// An image file's lowercased stem; `None` for a file that isn't an image.
fn image_stem(p: &Path) -> Option<String> {
    let ext = p.extension()?.to_string_lossy().to_lowercase();
    IMAGE_EXTS.contains(&ext.as_str()).then(|| {
        p.file_stem()
            .map(|x| x.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    })
}

#[derive(Debug, PartialEq, Eq)]
enum Role {
    Color,
    Other,
}

/// What a texture's name says it holds: `None` when it says nothing.
fn texture_role(stem: &str) -> Option<Role> {
    let tail = stem.rsplit(['-', '_', '.', ' ']).next().unwrap_or(stem);
    let split = tail.len() < stem.len();
    let ends = |list: &[&str]| split && list.contains(&tail);
    let has = |list: &[&str]| list.iter().any(|w| stem.contains(w));
    if has(&[
        "normal",
        "nrm",
        "emissi",
        "emit",
        "glow",
        "rough",
        "metal",
        "occlusion",
        "height",
        "disp",
        "bump",
    ]) || has(&["mask", "opacity", "alpha", "spec", "gloss", "smooth"])
        || ends(&[
            "n", "nor", "norm", "e", "r", "m", "mt", "ao", "occ", "h", "orm", "arm", "mra", "rma", "s", "g",
        ])
    {
        return Some(Role::Other);
    }
    if has(&[
        "albedo",
        "basecolor",
        "base_color",
        "base color",
        "basecolour",
        "diffuse",
    ]) || ends(&["diff", "dif", "col", "color", "colour", "d", "bc", "a", "c"])
    {
        return Some(Role::Color);
    }
    None
}

const IMAGE_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "tga", "bmp", "webp", "gif", "tif", "tiff", "dds", "psd",
];

/// Visit the files under `root` (at most 20 000 entries, each folder's in name order). True when
/// it stopped at that many.
#[cfg(feature = "fs")]
fn walk_files(root: &Path, visit: &mut dyn FnMut(&Path)) -> bool {
    let mut stack = vec![root.to_path_buf()];
    let mut seen = 0;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        // The listing says what each entry is; only a link needs looking up.
        let mut entries: Vec<(PathBuf, bool)> = entries
            .flatten()
            .map(|e| {
                let p = e.path();
                let dir = match e.file_type() {
                    Ok(t) if !t.is_symlink() => t.is_dir(),
                    _ => p.is_dir(),
                };
                (p, dir)
            })
            .collect();
        entries.sort_unstable();
        let mut dirs = Vec::new();
        for (p, dir) in entries {
            seen += 1;
            if seen > 20_000 {
                return true;
            }
            if dir {
                dirs.push(p);
            } else {
                visit(&p);
            }
        }
        stack.extend(dirs.into_iter().rev());
    }
    false
}

/// `path` with its `.` and `..` parts worked out.
#[cfg(feature = "fs")]
fn tidy(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// Find a texture a model refers to, in one of the [`forms`] its reference takes: relative to it (`../Textures/rock.png` too, inside `root`),
/// or under `root` by file name, the one nearest the model (artists' absolute paths like
/// `C:\Users\me\tex\rock.tga` survive only as names). Failing that, an image with the same name
/// in another format.
#[cfg(feature = "fs")]
fn find_texture<'a>(
    cleaned: &str,
    model_dir: &Path,
    root: &Path,
    files: impl FnOnce() -> Cow<'a, [(PathBuf, String)]>,
) -> Option<PathBuf> {
    if !cleaned.starts_with('/') && !cleaned.contains(':') {
        let rel = model_dir.join(cleaned);
        let up = cleaned.split('/').any(|c| c == "..");
        let inside = !up || tidy(&rel).starts_with(tidy(root));
        if inside && rel.is_file() {
            return Some(if up { tidy(&rel) } else { rel });
        }
    }
    let name = cleaned.rsplit('/').next()?.to_lowercase();
    let files = files();
    let named = files.iter().filter(|(_, n)| *n == name).map(|(p, _)| p.clone());
    nearest(model_dir, named).or_else(|| {
        let files = files.iter().map(|(p, n)| (p.as_path(), n.as_str()));
        same_name_other_format(&name, model_dir, files)
    })
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;

    #[test]
    fn texture_roles_by_name() {
        assert_eq!(texture_role("barrel_normal"), Some(Role::Other));
        assert_eq!(texture_role("barrel_d"), Some(Role::Color));
        assert_eq!(texture_role("barrel_albedo"), Some(Role::Color));
        assert_eq!(texture_role("barrel"), None);
    }

    #[test]
    fn quoted_references_are_found_without_their_quotes() {
        let m = Memory::new().with("Models/my tex.png", "png");
        let found = |r: &str| m.find(r, Path::new("Models")).map(|p| p.to_string_lossy().into_owned());
        for r in ["\"my tex.png\"", " 'my tex.png' ", "\"C:\\art\\my tex.png\""] {
            assert_eq!(found(r).as_deref(), Some("Models/my tex.png"), "{r}");
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("my tex.png"), "png").unwrap();
        let folder = Folder::new(dir.path());
        assert!(folder.find("\"my tex.png\"", dir.path()).is_some());
        assert_eq!(cleaned("\""), "\"");
    }

    #[test]
    fn file_uris_and_percent_escapes_name_the_file_they_name() {
        assert_eq!(cleaned("file:///C:/Old%20Path/tex.png"), "C:/Old Path/tex.png");
        assert_eq!(cleaned("FILE://localhost/home/me/a%20b.png"), "/home/me/a b.png");
        assert_eq!(cleaned("file:///home/me/tex.png"), "/home/me/tex.png");
        assert_eq!(cleaned("file:tex.png"), "tex.png");
        assert_eq!(local_path("textures/my%20tex.png"), "textures/my tex.png");
        // A plain reference is tried as written, then decoded: a file may be named with a percent sign.
        assert_eq!(forms("my%20tex.png"), ["my%20tex.png", "my tex.png"]);
        assert_eq!(forms("tex.png"), ["tex.png"]);
        let m = Memory::new()
            .with("Models/my tex.png", "a")
            .with("Models/100%25.png", "b");
        let found = |r: &str| m.find(r, Path::new("Models")).map(|p| p.to_string_lossy().into_owned());
        assert_eq!(found("my%20tex.png").as_deref(), Some("Models/my tex.png"));
        assert_eq!(found("100%25.png").as_deref(), Some("Models/100%25.png"));
        assert_eq!(
            found("file:///C:/Old%20Path/MY%20TEX.PNG").as_deref(),
            Some("Models/my tex.png")
        );
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("my tex.png"), "png").unwrap();
        let folder = Folder::new(dir.path());
        assert!(folder.find("file:///D:/art/my%20tex.png", dir.path()).is_some());
        assert!(folder.find("my%20tex.png", dir.path()).is_some());
    }

    #[test]
    fn memory_finds_like_a_folder() {
        let m = Memory::new()
            .with("Models/crate.obj", "")
            .with("Models\\textures\\Crate.PNG", "png")
            .with("textures/rock.jpg", "jpg")
            .with("Barrel_Albedo.png", "a")
            .with("Barrel_Normal.png", "n");
        let found = |r: &str| m.find(r, Path::new("Models")).map(|p| p.to_string_lossy().into_owned());
        let krate = Some("Models/textures/Crate.PNG");
        assert_eq!(found("textures/Crate.PNG").as_deref(), krate);
        assert_eq!(found("./textures\\Crate.PNG").as_deref(), krate);
        assert_eq!(found("D:\\Work\\crate.png").as_deref(), krate);
        assert_eq!(found("../textures/rock.jpg").as_deref(), Some("textures/rock.jpg"));
        assert_eq!(found("rock.tga").as_deref(), Some("textures/rock.jpg"));
        assert_eq!(found("missing.png"), None);
        assert_eq!(
            m.read(Path::new("./Models/textures/Crate.PNG")).as_deref(),
            Some(&b"png"[..])
        );
        assert_eq!(
            m.color_texture_for(Path::new("Barrel.obj")),
            Some(PathBuf::from("Barrel_Albedo.png"))
        );
    }

    #[test]
    fn color_textures_by_name_come_from_the_models_part_of_the_pack() {
        let m = Memory::new()
            .with("Bundle/Kit2D/Sprites/Stairs.png", "2d")
            .with("Bundle/Kit3D/Textures/Crate.png", "3d");
        let guess = |model: &str| m.color_texture_for(Path::new(model));
        assert_eq!(guess("Bundle/Kit3D/Models/Stairs.fbx"), None);
        assert_eq!(
            guess("Bundle/Kit3D/Models/Crate.fbx"),
            Some(PathBuf::from("Bundle/Kit3D/Textures/Crate.png"))
        );
        assert_eq!(guess("Models/Stairs.fbx"), None, "a sprite is never taken");
        assert_eq!(
            guess("Models/Crate.fbx"),
            Some(PathBuf::from("Bundle/Kit3D/Textures/Crate.png")),
            "a model near the top searches the whole pack"
        );
    }
}
