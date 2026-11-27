//! glTF 2.0, `.gltf` and `.glb`, read as the [`Document`] it already is. Nothing is decoded:
//! extensions (known or not), extras, and meshopt or Draco compression pass through as they are.
//!
//! - A GLB file is read whole, in one call, and its binary chunk shared from those bytes without a
//!   copy. A GLB in memory shares its bytes the same way when they're held as [`Bytes`].
//! - External buffers are found relative to the model's file (`..` too, but never an absolute
//!   path); for a model in memory they come from [`Assets::read_shared`]. A meshopt fallback
//!   buffer that isn't there holds nothing: views on it decode from their compressed data.
//! - Images named by URI (a data URI, or a file found as buffers are) move into a buffer, so the
//!   document stands on its own. A file that isn't where its URI says is looked for as every other format's
//!   textures are ([`Assets::find`]): by file name beside the model or in the pack, in any case, or in another
//!   image format, so another machine's absolute path or a `file:` URI finds its file. One that isn't found
//!   keeps its URI.
//! - A texture none of whose images Polysmelt both has and decodes (one in KTX2 or AVIF, or a file that
//!   isn't found) isn't drawn. Its images are counted in `asset.extras.source.droppedTextures`, each once
//!   however many textures use it, and those not found are listed in `missingFiles`, as every reader
//!   reports a texture it doesn't have. Nothing else of the document changes for it.
//! - `asset.extras.source` gets what every reader writes there: the format and version, and glTF's
//!   meters, +Y up, +Z forward and right-handedness, declared. A file that has a source already (a
//!   model Polysmelt converted, read back) keeps it.
//! - glTF 1.0 (and its binary GLB, `KHR_binary_glTF`) is upgraded to 2.0 first; see
//!   [`super::gltf1`]. An upgraded document is converted, not passed through, so it follows the rules every
//!   converting reader does ([`super::rules`]): it hangs from a `RootNode`, its animations start at 0 and its
//!   skins follow the one skinning rule. A glTF 2.0 file is left as it is.

use std::path::{Component, Path};

use serde_json::{Map, Value, json};

use super::{Input, gltf1};
use crate::assets::Assets;
use crate::document::{Blob, Document, Keep};
use crate::gltf as container;
use crate::images::{Location, Reference};
use crate::num::usize_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Bytes, Context as _, ErrorKind, Result};

/// Reads the glTF or GLB file `input`: a glTF 2.0 one as it is, and a glTF 1.0 one upgraded and held to the
/// reader rules, under a `RootNode`.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let (mut doc, upgraded) = read(input, assets, keep)?;
    if upgraded {
        super::frame::root(&mut doc, super::frame::Frame::IDENTITY);
        super::rules::conform(&mut doc, keep);
    }
    Ok(doc)
}

/// Reads the glTF or GLB `input` (glTF 1.0 upgraded to 2.0) as it is, for a reader that has its own use for
/// it; and whether it was upgraded.
pub(crate) fn read(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<(Document, bool)> {
    let (json, mut bin) = match input.bytes() {
        Some(b) if b.starts_with(b"glTF") => glb_in_memory(b, |part| input.share(part))?,
        Some(b) => (gltf_json(b)?, None),
        #[cfg(feature = "fs")]
        None => open_file(input.path)?,
        #[cfg(not(feature = "fs"))]
        None => unreachable!("a model without its bytes needs the `fs` feature"),
    };
    let (mut json, fix) = if gltf1::is_v1(&json) {
        let (json, fix) = gltf1::upgrade(json)?;
        (json, Some(fix))
    } else {
        (json, None)
    };
    let version = json
        .get("asset")
        .and_then(|a| a.get("version"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    if version.starts_with('1') {
        // A 1.0 label on a 2.0 document: read as the 2.0 it is, the label kept as its source's.
        json["asset"]["version"] = "2.0".into();
        source(&mut json)["version"] = version.clone().into();
    } else {
        ensure!(
            version.starts_with("2."),
            Unsupported,
            "glTF version {version:?} isn't 1.x or 2.x"
        );
    }
    if fix.is_none() {
        own_source(&mut json, &version);
    }
    // Told by what the file declares, before images are read.
    let undecodable = undecodable_images(&json);
    let mut doc = Document::new(json, Vec::new(), Map::new(), crate::images::Refs::new());
    let n = doc.base.get("buffers").and_then(Value::as_array).map_or(0, Vec::len);
    for i in 0..n {
        let b = &mut doc.base_mut()["buffers"][i];
        let declared = b.get("byteLength").and_then(Value::as_u64).unwrap_or(0);
        let fallback = container::is_fallback(b);
        let blob = match b.get("uri").and_then(Value::as_str) {
            Some(uri) if uri.starts_with("data:") => Blob::Bytes(container::data_uri(uri)?.1.into()),
            Some(uri) => match external(input, assets, uri) {
                Ok(blob) => blob,
                Err(_) if fallback => Blob::Missing(declared),
                Err(e) => return Err(e),
            },
            None if i == 0 && bin.is_some() => bin.take().expect("checked"),
            None if fallback => Blob::Missing(declared),
            // An empty buffer needs no data (a placeholder some writers leave).
            None if declared == 0 => Blob::Bytes(Bytes::new()),
            None => bail!(Corrupt, "glTF buffer {i} has no data"),
        };
        ensure!(
            blob.len() >= declared,
            Truncated,
            "glTF buffer {i} is cut short: it holds {} bytes but declares {declared}",
            blob.len()
        );
        if let Some(b) = b.as_object_mut() {
            b.remove("uri");
            // glTF 1.0 lets the length go unsaid.
            if declared == 0 {
                b.insert("byteLength".into(), json!(blob.len()));
            }
        }
        doc.buffers.push(blob);
    }
    let upgraded = fix.is_some();
    if let Some(fix) = fix {
        gltf1::finish(&mut doc, &fix)?;
    }
    // Images that would only be dropped aren't read, nor counted.
    if keep.materials && keep.textures {
        if keep.image_bytes {
            let lacks = embed_images(&mut doc, input, assets);
            report_dropped(&mut doc, &undecodable, |i| lacks.get(&i).copied());
        } else {
            reference_images(&mut doc, input, assets);
            let refs = std::mem::take(&mut doc.refs);
            report_dropped(&mut doc, &undecodable, |i| {
                refs.get(&i)
                    .is_some_and(|r| matches!(r.location(), Location::Missing))
                    .then_some(true)
            });
            doc.refs = refs;
        }
    }
    Ok((doc, upgraded))
}

/// The asset's `extras.source`, made if it isn't there. `json` has its `asset`.
fn source(json: &mut Map<String, Value>) -> &mut Value {
    let extras = &mut json["asset"]["extras"];
    if !extras.is_object() {
        *extras = Value::Object(Map::new());
    }
    let source = &mut extras["source"];
    if !source.is_object() {
        *source = Value::Object(Map::new());
    }
    source
}

/// Writes what a glTF 2.0 file says of itself by being one into its `asset.extras.source`, as every reader
/// does: the format and `version`, glTF's meters, +Y up, +Z forward and right-handedness (all declared: the
/// specification fixes them), its `generator` as the tool and its `copyright`. A file that carries a source
/// already (one Polysmelt converted from another format, read back) keeps it.
fn own_source(json: &mut Map<String, Value>, version: &str) {
    let asset = json.get("asset");
    if asset.and_then(|a| a.pointer("/extras/source/format")).is_some() {
        return;
    }
    let text = |k: &str| asset.and_then(|a| a.get(k)).and_then(Value::as_str).map(str::to_owned);
    let info = SourceInfo {
        format: Some("glTF".into()),
        version: Some(version.to_owned()),
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Declared(Axis::PosY),
        forward_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        tool: text("generator"),
        copyright: text("copyright"),
        ..SourceInfo::default()
    };
    if let Value::Object(source) = source(json) {
        info.write(source);
    }
}

/// Which images are in a format Polysmelt doesn't decode: KTX2 (`KHR_texture_basisu`), AVIF
/// (`EXT_texture_avif`), or WebP without the `webp` feature. An image's format is the type it declares,
/// or its name's, not its bytes', so the answer is the same however much of the file is read.
fn undecodable_images(json: &Map<String, Value>) -> Vec<bool> {
    let images = json
        .get("images")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let undecodable = |img: &Value| {
        let uri = img.get("uri").and_then(Value::as_str).unwrap_or("");
        let named = if let Some(rest) = uri.strip_prefix("data:") {
            rest.split([';', ',']).next().map(str::to_owned)
        } else {
            let name = container::percent_decode(uri).to_ascii_lowercase();
            let known = [
                (".ktx2", "image/ktx2"),
                (".avif", "image/avif"),
                (".webp", "image/webp"),
            ];
            known
                .iter()
                .find(|(ext, _)| name.ends_with(ext))
                .map(|(_, m)| (*m).to_owned())
        };
        match img.get("mimeType").and_then(Value::as_str).or(named.as_deref()) {
            Some("image/ktx2" | "image/avif") => true,
            Some("image/webp") => !cfg!(feature = "webp"),
            _ => false,
        }
    };
    images.iter().map(undecodable).collect()
}

/// Reports the textures the document can't draw as every reader reports textures it doesn't have
/// ([`crate::images::Missed`]), changing nothing else. A texture is drawn from its `source` or an
/// extension's; one with none that Polysmelt decodes ([`undecodable_images`]) and has isn't drawn.
/// Each image of such a texture is counted once in `droppedTextures`, however many textures and
/// materials use it (images that name one file by the same URI are one), and the URIs of those not
/// found are listed in `missingFiles`, as written.
/// `lacks(i)` is `Some(true)` for an image whose file wasn't found, `Some(false)` for one found that
/// isn't an image that can be embedded, and `None` for one that's there.
fn report_dropped(doc: &mut Document, undecodable: &[bool], lacks: impl Fn(usize) -> Option<bool>) {
    let list = |key: &str| {
        doc.base
            .get(key)
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
    };
    let images = list("images");
    let mut dropped = std::collections::BTreeSet::new();
    let mut missed = crate::images::Missed::default();
    for t in list("textures") {
        let extensions = t.get("extensions").and_then(Value::as_object);
        let sources: Vec<Option<usize>> = t
            .get("source")
            .into_iter()
            .chain(extensions.into_iter().flatten().filter_map(|(_, e)| e.get("source")))
            .map(|s| s.as_u64().map(usize_of).filter(|&i| i < images.len()))
            .collect();
        // A source that names no image is left to the rest of the reader, as before.
        let usable =
            |s: &Option<usize>| s.is_none_or(|i| !undecodable.get(i).copied().unwrap_or(false) && lacks(i).is_none());
        if sources.is_empty() || sources.iter().any(usable) {
            continue;
        }
        for i in sources.into_iter().flatten() {
            let uri = images[i]
                .get("uri")
                .and_then(Value::as_str)
                .filter(|u| !u.starts_with("data:"));
            if !dropped.insert(uri.ok_or(i)) {
                continue;
            }
            if let Some(uri) = uri {
                missed.names.push(uri.to_owned());
            }
            if let Some(uri) = uri.filter(|_| lacks(i) == Some(true)) {
                missed.files.push(uri.to_owned());
            }
        }
    }
    missed.dropped = dropped.len();
    if missed.is_empty() {
        return;
    }
    if let Value::Object(source) = source(doc.base_mut()) {
        let counted = !source.contains_key("droppedTextures");
        missed.write(source);
        doc.missed = Some(Box::new((missed, counted)));
    }
}

/// Moves each image named by URI into a buffer of its own, when it's found and its type known. Returns
/// the images it couldn't: `true` for a file that wasn't found, `false` for one found whose type isn't
/// known.
fn embed_images(doc: &mut Document, input: Input<'_>, assets: &dyn Assets) -> std::collections::HashMap<usize, bool> {
    let mut lacks = std::collections::HashMap::new();
    let n = doc.base.get("images").and_then(Value::as_array).map_or(0, Vec::len);
    for i in 0..n {
        let img = &doc.base["images"][i];
        let Some(uri) = img.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let declared = img.get("mimeType").and_then(Value::as_str);
        let typed =
            |bytes: &[u8], named: Option<&str>| container::sniff_mime(bytes).or(declared).or(named).map(str::to_owned);
        let found = if uri.starts_with("data:") {
            container::data_uri(uri)
                .ok()
                .and_then(|(named, bytes)| Some((typed(&bytes, named.as_deref())?, Blob::Bytes(bytes.into()))))
        } else {
            let named = container::mime_from_name(&container::percent_decode(uri));
            let Ok(bytes) = read_image(input, assets, uri) else {
                lacks.insert(i, true);
                continue;
            };
            let mime = typed(&bytes, named);
            if mime.is_none() {
                lacks.insert(i, false);
            }
            mime.map(|mime| (mime, Blob::Bytes(bytes)))
        };
        let Some((mime, blob)) = found else { continue };
        let len = blob.len();
        let buffer = doc.push_blob(blob);
        let view = doc.push_view(json!({ "buffer": buffer, "byteLength": len }));
        let img = doc.base_mut()["images"][i]
            .as_object_mut()
            .expect("an image with a uri is an object");
        img.remove("uri");
        img.insert("bufferView".into(), json!(view));
        img.insert("mimeType".into(), json!(mime));
    }
    lacks
}

/// Notes where each image named by URI (other than a `data:` one) is, without reading it. A file
/// on disk is only looked for when the image's location is first asked.
fn reference_images(doc: &mut Document, input: Input<'_>, assets: &dyn Assets) {
    let images = doc.base.get("images").and_then(Value::as_array);
    for (i, img) in images.into_iter().flatten().enumerate() {
        let Some(uri) = img
            .get("uri")
            .and_then(Value::as_str)
            .filter(|u| !u.starts_with("data:"))
        else {
            continue;
        };
        let name = container::percent_decode(uri);
        let path = checked(&name).ok().map(|rel| input.dir().join(rel));
        // A model read from disk has its image files looked for when they're first asked about,
        // since a lookup per image adds up; one read from memory asks its assets now.
        #[cfg(feature = "fs")]
        let later = input.bytes().is_none();
        #[cfg(not(feature = "fs"))]
        let later = false;
        // What isn't where the URI says (or isn't a plain relative path: another machine's absolute path, a
        // `file:` URI) is looked for as every reader's textures are.
        let found = path
            .filter(|p| later || assets.read(p).is_some())
            .or_else(|| assets.find(uri, input.dir()));
        let location = found.map_or(Location::Missing, Location::File);
        let r = Reference {
            found: (later && matches!(location, Location::File(_))).then(std::sync::OnceLock::new),
            name,
            location,
            bytes: None,
            raw: true,
        };
        doc.refs.insert(i, r);
    }
}

/// `rel` when it's a relative path: no root, drive, share or URI scheme. It may climb out of the
/// model's folder (`../shared/scene.bin`), as models that share their buffers do; a model in
/// memory gets only what its [`Assets`] give, and [`crate::gltf::files_in`] stays inside its
/// folder.
fn checked(rel: &str) -> Result<&str> {
    let safe = !rel.contains(':')
        && Path::new(rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir | Component::ParentDir));
    ensure!(safe, Unsupported, "refusing to read {rel:?}: not a relative path");
    Ok(rel)
}

/// An external buffer: from the model's folder, or for a model in memory from its assets.
fn external(input: Input<'_>, assets: &dyn Assets, uri: &str) -> Result<Blob> {
    read_external(input, assets, uri).map(Blob::Bytes)
}

/// An external file the model names by URI, whole: from the model's folder, or for a model in
/// memory from its assets (shared, not copied, when they hold it as [`Bytes`]).
fn read_external(input: Input<'_>, assets: &dyn Assets, uri: &str) -> Result<Bytes> {
    let rel = container::percent_decode(uri);
    let p = input.dir().join(checked(&rel)?);
    #[cfg(feature = "fs")]
    if input.bytes().is_none() {
        return read_file(&p, Some(&rel));
    }
    assets
        .read_shared(&p)
        .ok_or_else(|| crate::Error::missing(format!("glTF file {rel:?} isn't among the assets")).at(&p))
}

/// An external image the model names by URI: the file the URI names, or else the one the shared resolver finds
/// for it ([`Assets::find`]), as for every other format's textures: by file name beside the model or in the
/// pack, whatever the case, or in another image format; another machine's absolute path and a `file:` URI are
/// found by their file names.
fn read_image(input: Input<'_>, assets: &dyn Assets, uri: &str) -> Result<Bytes> {
    let exact = match read_external(input, assets, uri) {
        Ok(bytes) => return Ok(bytes),
        Err(e) => e,
    };
    let Some(p) = assets.find(uri, input.dir()) else {
        return Err(exact);
    };
    #[cfg(feature = "fs")]
    if input.bytes().is_none() {
        return read_file(&p, Some(uri));
    }
    assets.read_shared(&p).ok_or(exact)
}

/// The file at `path`, whole, in one read into memory reserved for it (failing, rather than
/// aborting, when there isn't that much). `asset` names a file the model refers to, as its
/// reference has it.
#[cfg(feature = "fs")]
fn read_file(path: &Path, asset: Option<&str>) -> Result<Bytes> {
    use std::io::Read as _;
    let io = |e: std::io::Error| match asset {
        Some(rel) => crate::Error::asset(e, path, format!("glTF file {rel:?}")),
        None => crate::Error::from(e),
    };
    let mut file = std::fs::File::open(path).map_err(io)?;
    let len = file.metadata().map_err(io)?.len();
    let mut all = Vec::new();
    all.try_reserve_exact(usize::try_from(len).unwrap_or(usize::MAX))
        .map_err(|_| err!(LimitExceeded, "glTF file of {len} bytes"))?;
    file.read_to_end(&mut all).map_err(io)?;
    Ok(all.into())
}

/// Where a GLB's binary chunk is: offset and length.
type Span = (u64, u64);

/// A GLB's JSON and where its binary chunk is. `read(off, len)` reads the container.
fn glb_layout<'b>(
    total: u64,
    read: impl Fn(u64, usize) -> Result<&'b [u8]>,
) -> Result<(Map<String, Value>, Option<Span>)> {
    let head = read(0, 20)?;
    let u32_at = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    if u32_at(head, 4) == 1 {
        // glTF 1.0's binary glTF: the JSON's length and format (0, JSON) in the header, then
        // the body, unpadded, to the end.
        ensure!(u32_at(head, 16) == 0, Corrupt, "binary glTF 1.0 content isn't JSON");
        let json_len = u64::from(u32_at(head, 12));
        let end = u64::from(u32_at(head, 8)).min(total);
        ensure!(20 + json_len <= end, Truncated, "GLB is cut short");
        let text = read(20, usize_of(json_len))?;
        let json = glb_json(text)?;
        return Ok((json, Some((20 + json_len, end - 20 - json_len))));
    }
    ensure!(
        u32_at(head, 4) == 2,
        Unsupported,
        "GLB version {} is not 1 or 2",
        u32_at(head, 4)
    );
    ensure!(&head[16..20] == b"JSON", Corrupt, "GLB doesn't start with a JSON chunk");
    let json_len = u64::from(u32_at(head, 12));
    ensure!(20 + json_len <= total, Truncated, "GLB is cut short");
    let text = read(20, usize_of(json_len))?;
    let json = glb_json(text)?;
    let at = (20 + json_len).next_multiple_of(4);
    let mut bin = None;
    if at + 8 <= total {
        let h = read(at, 8)?;
        if &h[4..8] == b"BIN\0" {
            let base = at + 8;
            bin = Some((base, u64::from(u32_at(h, 0)).min(total - base)));
        }
    }
    Ok((json, bin))
}

/// A GLB in memory, as its JSON and binary chunk, which `share` makes [`Bytes`] of (without a
/// copy, when it can).
fn glb_in_memory(bytes: &[u8], share: impl FnOnce(&[u8]) -> Bytes) -> Result<(Map<String, Value>, Option<Blob>)> {
    let total = bytes.len() as u64;
    let (json, bin) = glb_layout(total, |off, len| {
        bytes
            .get(usize_of(off)..usize_of(off).saturating_add(len))
            .context(ErrorKind::Truncated, "GLB is cut short")
    })?;
    let bin = bin.map(|(base, len)| Blob::Bytes(share(&bytes[usize_of(base)..usize_of(base + len)])));
    Ok((json, bin))
}

/// A GLB's JSON chunk, after any byte-order mark: the spec forbids one, but writers add it.
fn glb_json(text: &[u8]) -> Result<Map<String, Value>> {
    let text = text.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(text);
    // The chunk is padded with spaces, or by some writers with zeros.
    let end = text.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    let Value::Object(json) =
        crate::json::from_slice(&text[..end]).context(ErrorKind::Corrupt, "GLB JSON doesn't parse")?
    else {
        bail!(Corrupt, "glTF JSON is not an object")
    };
    Ok(json)
}

/// glTF JSON (a `.gltf`), after any byte-order mark.
fn gltf_json(bytes: &[u8]) -> Result<Map<String, Value>> {
    let text = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    match crate::json::from_slice(text) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => bail!(UnknownFormat, "not a glTF file: neither GLB nor glTF JSON"),
        Err(e) => match crate::Error::json(e, text, "glTF JSON") {
            e if e.kind() == ErrorKind::UnknownFormat => {
                bail!(UnknownFormat, "not a glTF file: neither GLB nor glTF JSON")
            }
            e => Err(e),
        },
    }
}

/// The JSON and binary chunk of a glTF or GLB file, read whole: the binary chunk is shared from
/// the bytes read, not copied out of them.
#[cfg(feature = "fs")]
fn open_file(path: &Path) -> Result<(Map<String, Value>, Option<Blob>)> {
    let all = read_file(path, None)?;
    if all.starts_with(b"glTF") {
        return glb_in_memory(&all, |part| all.slice_ref(part));
    }
    Ok((gltf_json(&all)?, None))
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    use crate::assets::Folder;
    use crate::document::{Document, Keep};

    /// A GLB's JSON (after any byte-order mark) and binary chunk.
    fn split(glb: &[u8]) -> (Value, &[u8]) {
        let n = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let text = &glb[20..20 + n];
        let json = serde_json::from_slice(text.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(text)).unwrap();
        let bin = glb.get(20 + n + 8..).unwrap_or(&[]);
        (json, bin)
    }

    /// The bytes of each buffer view.
    fn views(json: &Value, bin: &[u8]) -> Vec<Vec<u8>> {
        let empty = Vec::new();
        let list = json["bufferViews"].as_array().unwrap_or(&empty);
        list.iter()
            .map(|v| {
                if v["buffer"].as_u64() != Some(0) {
                    return Vec::new();
                }
                let off = usize::try_from(v["byteOffset"].as_u64().unwrap_or(0)).unwrap();
                let len = usize::try_from(v["byteLength"].as_u64().unwrap()).unwrap();
                bin[off..off + len].to_vec()
            })
            .collect()
    }

    /// The JSON without what writing may restate: views' default `byteOffset`, where in the
    /// binary chunk each view sits, and empty buffers at the end, which no view can use. Nor
    /// the textures it doesn't have, the one note reading adds.
    fn normalized(mut json: Value) -> Value {
        if let Some(asset) = json.get_mut("asset").and_then(Value::as_object_mut) {
            let extras = asset.get_mut("extras").and_then(Value::as_object_mut);
            if let Some(source) = extras.and_then(|e| e.get_mut("source")?.as_object_mut()) {
                source.remove("droppedTextures");
                source.remove("missingFiles");
                // What reading writes of the file itself.
                if source.get("format") == Some(&serde_json::json!("glTF")) {
                    for key in [
                        "format",
                        "version",
                        "metersPerUnit",
                        "unit",
                        "upAxis",
                        "forwardAxis",
                        "handedness",
                        "tool",
                        "copyright",
                    ] {
                        source.remove(key);
                    }
                }
            }
            if asset.get("extras") == Some(&serde_json::json!({ "source": {} })) {
                asset.remove("extras");
            }
        }
        if let Some(list) = json.get_mut("buffers").and_then(Value::as_array_mut) {
            while list.len() > 1 && list.last().and_then(|b| b["byteLength"].as_u64()) == Some(0) {
                list.pop();
            }
        }
        for v in json["bufferViews"].as_array_mut().into_iter().flatten() {
            let o = v.as_object_mut().unwrap();
            if o.get("buffer").and_then(Value::as_u64) == Some(0) {
                o.remove("byteOffset");
            }
        }
        if let Some(b) = json.get_mut("buffers").and_then(|b| b.get_mut(0)) {
            b.as_object_mut().unwrap().remove("byteLength");
        }
        json
    }

    fn glbs() -> Vec<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf");
        let mut out = Vec::new();
        // Case folders: a license sits beside them.
        for case in std::fs::read_dir(root).unwrap().flatten().filter(|e| e.path().is_dir()) {
            for f in std::fs::read_dir(case.path()).unwrap().flatten() {
                if f.path().extension().is_some_and(|e| e == "glb") {
                    out.push(f.path());
                }
            }
        }
        out.sort();
        out
    }

    #[cfg(feature = "fs")]
    #[test]
    fn a_buffer_in_a_parent_folder_is_read() {
        let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf/gltf_external_bin");
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("model")).unwrap();
        std::fs::create_dir_all(dir.path().join("data")).unwrap();
        std::fs::copy(case.join("cube.bin"), dir.path().join("data/cube.bin")).unwrap();
        let text = std::fs::read_to_string(case.join("cube.gltf")).unwrap();
        let gltf = dir.path().join("model/cube.gltf");
        std::fs::write(&gltf, text.replace("\"cube.bin\"", "\"../data/cube.bin\"")).unwrap();
        let doc = crate::formats::load(&gltf, "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
        assert_eq!(
            doc.flatten(crate::document::Pose::Rest).unwrap().unwrap().triangles(),
            12
        );
        std::fs::write(&gltf, text.replace("\"cube.bin\"", "\"/data/cube.bin\"")).unwrap();
        let err = crate::formats::load(&gltf, "gltf", &Folder::new(dir.path()), Keep::ALL)
            .err()
            .unwrap();
        assert!(err.to_string().contains("refusing"), "{err}");
    }

    #[cfg(feature = "fs")]
    #[test]
    fn image_files_are_written_as_they_are() {
        // Many images, each its own bytes; some named wrongly, so only their first bytes tell
        // their type.
        let n = 260;
        let dir = tempfile::tempdir().unwrap();
        let png = |i: usize| {
            let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
            b.extend((0..i % 300).map(|k| u8::try_from((k * 7 + i) % 251).unwrap()));
            b
        };
        let jpeg = b"\xff\xd8\xff\xe0 not really a jpeg".to_vec();
        let mut images = Vec::new();
        for i in 0..n {
            let name = if i % 5 == 0 {
                format!("t{i}.jpg")
            } else {
                format!("t{i}.png")
            };
            std::fs::write(dir.path().join(&name), png(i)).unwrap();
            images.push(serde_json::json!({ "uri": name }));
        }
        std::fs::write(dir.path().join("j.png"), &jpeg).unwrap();
        images.push(serde_json::json!({ "uri": "j.png", "mimeType": "image/png" }));
        images.push(serde_json::json!({ "uri": "missing.png" }));
        let gltf = serde_json::json!({ "asset": { "version": "2.0" }, "images": images, "nodes": [{ "name": "n" }] });
        let path = dir.path().join("m.gltf");
        std::fs::write(&path, gltf.to_string()).unwrap();
        let doc = crate::formats::load(&path, "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
        let glb = doc.to_glb().unwrap();
        let (json, bin) = split(&glb);
        let views = views(&json, bin);
        let images = json["images"].as_array().unwrap();
        assert_eq!(images.len(), n + 2);
        for (i, img) in images.iter().enumerate() {
            if i == n + 1 {
                assert_eq!(img["uri"], "missing.png", "what isn't found keeps its URI");
                continue;
            }
            let view = usize::try_from(img["bufferView"].as_u64().unwrap()).unwrap();
            let (bytes, mime) = if i == n {
                (jpeg.clone(), "image/jpeg")
            } else {
                (png(i), "image/png")
            };
            assert_eq!(views[view], bytes, "image {i}");
            assert_eq!(img["mimeType"], mime, "image {i}");
        }
    }

    #[test]
    fn a_glb_reads_the_same_from_its_file_and_from_memory() {
        let mut n = 0;
        for path in glbs() {
            let dir = path.parent().unwrap();
            let Ok(doc) = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL) else {
                continue;
            };
            let bytes = std::fs::read(&path).unwrap();
            // The files beside it, for what it names (a model in memory reads them from its assets).
            let mut assets = crate::assets::Memory::new();
            for f in std::fs::read_dir(dir).unwrap().flatten() {
                assets.insert(&f.file_name().to_string_lossy(), std::fs::read(f.path()).unwrap());
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let mem = crate::formats::load_bytes(&bytes, &name, "glb", &assets, Keep::ALL).unwrap();
            let shown = |d: &Document| {
                format!(
                    "{:?}",
                    d.flatten(crate::document::Pose::Rest).map_err(|e| e.to_string())
                )
            };
            assert_eq!(shown(&doc), shown(&mem), "{}", path.display());
            assert!(doc.to_glb().unwrap() == mem.to_glb().unwrap(), "{}", path.display());
            n += 1;
        }
        assert!(n > 20, "only {n} GLBs read");
    }

    #[test]
    fn glb_round_trips() {
        let mut n = 0;
        for path in glbs() {
            let dir = path.parent().unwrap();
            let Ok(doc) = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL) else {
                // Cases that are meant to fail.
                continue;
            };
            let original = std::fs::read(&path).unwrap();
            let written = doc.to_glb().unwrap();
            let name = path.display();
            // A glTF 1.0 file comes back as 2.0, and buffers in files come back inside: only the
            // second round trip must match.
            let (a, abin) = split(&original);
            let in_files = a["buffers"]
                .as_array()
                .is_some_and(|l| l.iter().any(|b| b.get("uri").is_some()));
            // Nor must one the repair pass rewrites, as it does invalid input.
            let mut fixed = doc.json().clone();
            let repaired = crate::flatten::repaired(&doc, &mut fixed, &mut None).is_some() || fixed != *doc.json();
            if original[4..8] != 1u32.to_le_bytes() && !in_files && !repaired {
                let (b, bbin) = split(&written);
                assert_eq!(normalized(a.clone()), normalized(b.clone()), "{name}: the JSON changed");
                assert!(
                    views(&a, abin) == views(&b, bbin),
                    "{name}: a buffer view's bytes changed"
                );
            }
            // And again, now exactly.
            let again = crate::formats::load_bytes(&written, "x.glb", "glb", &crate::assets::Memory::new(), Keep::ALL)
                .and_then(|d: Document| d.to_glb())
                .unwrap();
            assert!(again == written, "{name}: a second round trip changed it");
            n += 1;
        }
        assert!(n > 20, "only {n} GLBs round-tripped");
    }

    #[test]
    fn a_json_chunk_padded_with_zeros_reads() {
        let mut json =
            br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"n"}]}"#.to_vec();
        json.resize(json.len().next_multiple_of(4) + 4, 0);
        let len = u32::try_from(json.len()).unwrap();
        let glb = [
            &b"glTF"[..],
            &2u32.to_le_bytes(),
            &(20 + len).to_le_bytes(),
            &len.to_le_bytes(),
            b"JSON",
            &json,
        ]
        .concat();
        let assets = crate::assets::Memory::new();
        let doc = crate::formats::load_bytes(&glb, "x.glb", "glb", &assets, Keep::ALL).unwrap();
        assert_eq!(doc.json()["scene"], 0);
    }
}
