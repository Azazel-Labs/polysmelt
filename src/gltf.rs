//! glTF 2.0 containers: packing a `.gltf` and the files it names into one GLB, and adding or
//! decoding away meshopt compression (`EXT_meshopt_compression`, `KHR_meshopt_compression`).
//!
//! - [`decompress`]: a GLB (or self-contained `.gltf`) to a plain GLB that any glTF reader takes.
//! - [`pack_glb`]: a `.gltf` and its external buffers and images to one GLB, compression kept,
//!   with the meshopt extensions' buffer references rewritten.
//! - [`pack_glb_decompressed`]: both at once, without fetching meshopt fallback buffers.
//! - [`compress`]: the reverse, lossless: meshopt-encodes a GLB's index and vertex views with
//!   `EXT_meshopt_compression`. [`compress_with`] takes [`CompressOptions`], for
//!   `KHR_meshopt_compression` with its stronger attribute codec and `COLOR` filter.
//! - [`files_in`]: a `fetch` for the packers that reads files next to the `.gltf`, safely.
//!
//! Only buffer views change: accessors, their component types and every other extension stay
//! as they were. `KHR_mesh_quantization` data stays quantized.

use std::borrow::Cow;
#[cfg(feature = "fs")]
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::num::usize_of;
use crate::{Context as _, ErrorKind, Result, Within as _, meshopt};

/// The meshopt compression extensions, by name.
pub const MESHOPT_EXTENSIONS: [&str; 2] = ["EXT_meshopt_compression", "KHR_meshopt_compression"];

const GLB_MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: &[u8; 4] = b"JSON";
const CHUNK_BIN: &[u8; 4] = b"BIN\0";

/// Decodes the meshopt compression in a GLB (or a `.gltf` whose buffers are data URIs) and
/// returns a plain GLB: every buffer view holds its data uncompressed in the one binary chunk,
/// fallback buffers are gone and neither meshopt extension is listed any more. Quantized
/// attributes (`KHR_mesh_quantization`) stay quantized; filtered ones come out in the component
/// type their accessor declares. Images are left as they are.
///
/// A GLB that uses neither extension comes back byte for byte.
///
/// # Errors
///
/// When `input` isn't glTF that parses, or its compressed data doesn't decode.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>> {
    let doc = Doc::parse(input)?;
    if let Some(len) = doc.glb_len
        && !doc.uses_meshopt()
    {
        return Ok(input[..len].to_vec());
    }
    doc.decompress(&mut no_fetch, false)
}

/// Packs a `.gltf` (or a GLB) and the files it names into one GLB. `fetch` is called with each
/// external buffer and image URI, percent-decoded (`textures/my%20tex.png` arrives as
/// `textures/my tex.png`); data URIs are decoded here. Meshopt compression is kept: every
/// buffer moves into the binary chunk, and buffer views and their `EXT_meshopt_compression` /
/// `KHR_meshopt_compression` objects are pointed at it. A meshopt fallback buffer that can't be
/// fetched (or has no URI) becomes a URI-less placeholder and the extension is marked required.
///
/// [`files_in`] gives a `fetch` that reads files next to the `.gltf`.
///
/// # Errors
///
/// When `gltf` doesn't parse, or `fetch` fails for a file the GLB can't do without.
pub fn pack_glb(gltf: &[u8], mut fetch: impl FnMut(&str) -> Result<Vec<u8>>) -> Result<Vec<u8>> {
    Doc::parse(gltf)?.pack(&mut fetch)
}

/// [`pack_glb`] and [`decompress`] in one pass: a `.gltf` (or GLB) and the files it names to a
/// plain GLB with every image embedded. Meshopt fallback buffers are never fetched.
///
/// # Errors
///
/// When `gltf` doesn't parse, `fetch` fails for a file the GLB can't do without, or
/// compressed data doesn't decode.
pub fn pack_glb_decompressed(gltf: &[u8], mut fetch: impl FnMut(&str) -> Result<Vec<u8>>) -> Result<Vec<u8>> {
    Doc::parse(gltf)?.decompress(&mut fetch, true)
}

/// Meshopt-compresses a GLB (or a self-contained `.gltf`) losslessly with
/// `EXT_meshopt_compression`: triangle-list index views use the triangle codec, other index
/// views the index codec, and vertex attribute views whose stride is a multiple of 4 the
/// attribute codec (version 0), unfiltered. Triangles may come back starting at another corner
/// (winding is kept); every other byte decodes exactly. Already-compressed input is
/// decompressed first. The same as [`compress_with`] and [`CompressOptions::EXT`].
///
/// # Errors
///
/// When `input` isn't glTF that parses, or already-compressed data doesn't decode.
pub fn compress(input: &[u8]) -> Result<Vec<u8>> {
    compress_with(input, CompressOptions::EXT)
}

/// Which meshopt extension [`compress_with`] writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshoptExtension {
    /// `EXT_meshopt_compression`: attribute codec version 0 and no `COLOR` filter, so every
    /// meshopt-aware reader takes it.
    #[default]
    Ext,
    /// `KHR_meshopt_compression`: attribute codec version 1, which compresses attributes
    /// better, and the `COLOR` filter. Only newer readers take it.
    Khr,
}

impl MeshoptExtension {
    /// The extension's name, as it appears in the glTF.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            MeshoptExtension::Ext => MESHOPT_EXTENSIONS[0],
            MeshoptExtension::Khr => MESHOPT_EXTENSIONS[1],
        }
    }
}

/// How [`compress_with`] compresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressOptions {
    /// The extension written, and with it the attribute codec version.
    pub extension: MeshoptExtension,
    /// The attribute codec's level, 0 (fastest) to 3 (smallest), as in
    /// [`meshopt::encode_vertex_buffer_level`]; levels above 3 encode as 3. Only the
    /// `KHR_meshopt_compression` codec (version 1) has levels; `EXT_meshopt_compression`'s ignores
    /// it.
    pub level: u8,
    /// Whether to store vertex colors with the `COLOR` filter (`KHR_meshopt_compression` only):
    /// views read only as `COLOR_n` attributes that are normalized `VEC4` unsigned bytes or
    /// shorts, tightly packed. The filter's `YCoCg` form compresses better than RGBA but is
    /// lossy: each channel decodes within 1 of its original value (out of 255 or 65535).
    pub color_filter: bool,
}

impl CompressOptions {
    /// `EXT_meshopt_compression`, lossless: what [`compress`] writes.
    pub const EXT: CompressOptions = CompressOptions {
        extension: MeshoptExtension::Ext,
        level: 2,
        color_filter: false,
    };
    /// `KHR_meshopt_compression` at meshoptimizer's default level, 2, lossless (gltfpack's
    /// `-ce khr` codec settings). Level 3 and `color_filter` are gltfpack's `-cz` ones.
    pub const KHR: CompressOptions = CompressOptions {
        extension: MeshoptExtension::Khr,
        level: 2,
        color_filter: false,
    };
}

impl Default for CompressOptions {
    /// [`CompressOptions::EXT`].
    fn default() -> Self {
        CompressOptions::EXT
    }
}

/// [`compress`] with a choice of extension, attribute codec level and the `COLOR` filter. With
/// `KHR_meshopt_compression`, attribute views use the version 1 codec at `options.level`, and
/// with `options.color_filter` qualifying color views are filtered, the one lossy step (see
/// [`CompressOptions::color_filter`]).
///
/// # Errors
///
/// When `input` isn't glTF that parses, already-compressed data doesn't decode, or
/// `options.color_filter` is set with `EXT_meshopt_compression`, which has no `COLOR` filter.
pub fn compress_with(input: &[u8], options: CompressOptions) -> Result<Vec<u8>> {
    ensure!(
        !options.color_filter || options.extension == MeshoptExtension::Khr,
        Invalid,
        "the meshopt COLOR filter needs KHR_meshopt_compression"
    );
    let plain;
    let mut doc = Doc::parse(input)?;
    if doc.uses_meshopt() || doc.glb_len.is_none() {
        plain = decompress(input)?;
        doc = Doc::parse(&plain)?;
    }
    doc.compress(options)
}

/// A `fetch` for [`pack_glb`] and [`pack_glb_decompressed`] that reads URIs as paths relative
/// to `dir`. It refuses absolute paths and `..`, so a `.gltf` can't pull in files from outside
/// `dir`.
#[cfg(feature = "fs")]
pub fn files_in(dir: impl Into<PathBuf>) -> impl FnMut(&str) -> Result<Vec<u8>> {
    let dir = dir.into();
    move |uri: &str| {
        let rel = Path::new(uri);
        let safe = rel
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        if !safe || uri.contains(':') {
            bail!(
                Unsupported,
                "refusing to read {uri:?}: not a relative path inside the model's folder"
            );
        }
        std::fs::read(dir.join(rel)).with_context(ErrorKind::Corrupt, || format!("reading {uri:?}"))
    }
}

fn no_fetch(uri: &str) -> Result<Vec<u8>> {
    bail!(
        Unsupported,
        "{uri:?} is an external file; use pack_glb_decompressed to read it"
    )
}

/// A parsed GLB or `.gltf`.
struct Doc<'a> {
    json: Map<String, Value>,
    /// The GLB's binary chunk.
    bin: Option<&'a [u8]>,
    /// The GLB's length, when the input is one.
    glb_len: Option<usize>,
}

/// Where a buffer's bytes are.
enum Data<'a> {
    Bytes(Cow<'a, [u8]>),
    /// A meshopt fallback buffer that wasn't fetched: only its declared length.
    Placeholder,
}

impl<'a> Doc<'a> {
    fn parse(input: &'a [u8]) -> Result<Self> {
        if input.len() >= 4 && &input[..4] == GLB_MAGIC {
            return Self::parse_glb(input);
        }
        let text = input.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(input);
        match crate::json::from_slice(text).context(ErrorKind::UnknownFormat, "not a GLB or glTF JSON")? {
            Value::Object(json) => Ok(Doc {
                json,
                bin: None,
                glb_len: None,
            }),
            _ => bail!(Corrupt, "glTF JSON is not an object"),
        }
    }

    fn parse_glb(input: &'a [u8]) -> Result<Self> {
        let u32_at = |at: usize| -> Result<usize> {
            let b = input.get(at..at + 4).and_then(|b| b.try_into().ok());
            Ok(u32::from_le_bytes(b.context(ErrorKind::Truncated, "GLB is cut short")?) as usize)
        };
        ensure!(u32_at(4)? == 2, Unsupported, "GLB version {} is not 2", u32_at(4)?);
        let len = u32_at(8)?;
        ensure!(
            (12..=input.len()).contains(&len),
            Truncated,
            "GLB length {len} doesn't match the {} bytes given",
            input.len()
        );
        let input = &input[..len];
        let (mut json, mut bin) = (None, None);
        let mut at = 12;
        while at < len {
            let size = u32_at(at)?;
            let kind = input
                .get(at + 4..at + 8)
                .context(ErrorKind::Truncated, "GLB is cut short")?;
            let body = input
                .get(at + 8..at + 8 + size)
                .context(ErrorKind::Truncated, "GLB chunk runs past the end of the file")?;
            if json.is_none() {
                ensure!(kind == CHUNK_JSON, Corrupt, "GLB doesn't start with a JSON chunk");
                json = Some(body);
            } else if kind == CHUNK_BIN && bin.is_none() {
                bin = Some(body);
            }
            at += 8 + size.next_multiple_of(4).min(len - at - 8);
        }
        let json = json.context(ErrorKind::Corrupt, "GLB has no JSON chunk")?;
        let json = json.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(json);
        match crate::json::from_slice(json).context(ErrorKind::Corrupt, "bad GLB JSON")? {
            Value::Object(json) => Ok(Doc {
                json,
                bin,
                glb_len: Some(len),
            }),
            _ => bail!(Corrupt, "glTF JSON is not an object"),
        }
    }

    fn list(&self, key: &str) -> &[Value] {
        self.json.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
    }

    fn uses_meshopt(&self) -> bool {
        self.list("extensionsUsed")
            .iter()
            .any(|e| e.as_str().is_some_and(|e| MESHOPT_EXTENSIONS.contains(&e)))
            || self.list("bufferViews").iter().any(|v| meshopt_of(v).is_some())
    }

    /// Buffer `i`'s bytes. `lenient` makes a fallback buffer that can't be had a placeholder.
    fn buffer(&self, i: usize, fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>, lenient: bool) -> Result<Data<'a>> {
        let b = self
            .list("buffers")
            .get(i)
            .with_context(ErrorKind::Corrupt, || format!("buffer {i} doesn't exist"))?;
        let declared = b.get("byteLength").and_then(Value::as_u64).map_or(0, usize_of);
        let fallback = lenient && is_fallback(b);
        let bytes: Cow<'a, [u8]> = match b.get("uri").and_then(Value::as_str) {
            Some(uri) if uri.starts_with("data:") => Cow::Owned(data_uri(uri)?.1),
            Some(uri) => match fetch(&percent_decode(uri)) {
                Ok(bytes) => Cow::Owned(bytes),
                Err(_) if fallback => return Ok(Data::Placeholder),
                Err(e) => return Err(e),
            },
            None if i == 0 && self.bin.is_some() => Cow::Borrowed(self.bin.unwrap_or_default()),
            None if fallback => return Ok(Data::Placeholder),
            None if declared == 0 => Cow::Borrowed(&[]),
            None => bail!(Corrupt, "buffer {i} has no data"),
        };
        ensure!(
            bytes.len() >= declared,
            Truncated,
            "buffer {i} is cut short: it holds {} bytes but declares {declared}",
            bytes.len()
        );
        Ok(Data::Bytes(bytes))
    }

    /// Every buffer view's data, decoded, into one binary chunk.
    fn decompress(mut self, fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>, embed_images: bool) -> Result<Vec<u8>> {
        let views = self.list("bufferViews");
        // Only the buffers something reads from: fallbacks are skipped.
        let mut needed = vec![false; self.list("buffers").len()];
        for (i, v) in views.iter().enumerate() {
            let buffer = match meshopt_of(v) {
                Some((_, ext)) => ext.get("buffer"),
                None => v.get("buffer"),
            };
            let b = usize_of(
                buffer
                    .and_then(Value::as_u64)
                    .with_context(ErrorKind::Corrupt, || format!("buffer view {i} has no buffer"))?,
            );
            *needed.get_mut(b).with_context(ErrorKind::Corrupt, || {
                format!("buffer view {i} names missing buffer {b}")
            })? = true;
        }
        let mut data = Vec::with_capacity(needed.len());
        for (i, &need) in needed.iter().enumerate() {
            data.push(if need {
                Some(self.buffer(i, fetch, false)?)
            } else {
                None
            });
        }
        let bytes = |b: usize| match data.get(b) {
            Some(Some(Data::Bytes(d))) => Ok(&d[..]),
            _ => bail!(Corrupt, "buffer {b} has no data"),
        };

        // Every view's source is checked before anything is reserved, so a hostile length
        // can't make us allocate: plain views must lie inside their buffer, and no meshopt
        // codec expands its data more than 1024 times.
        let mut total = 0usize;
        for (i, v) in views.iter().enumerate() {
            let len = view_len(v)?;
            if let Some((name, ext)) = meshopt_of(v) {
                let src = slice(
                    bytes(get_usize(ext, "buffer")?)?,
                    ext.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of),
                    get_usize(ext, "byteLength")?,
                )
                .with_context(ErrorKind::Truncated, || {
                    format!("buffer view {i}: {name} data runs past the end of its buffer")
                })?;
                ensure!(
                    len / 1024 <= src.len(),
                    Truncated,
                    "buffer view {i}: {name} data is cut short"
                );
            } else {
                let b = get_usize(
                    v.as_object()
                        .context(ErrorKind::Corrupt, "buffer view is not an object")?,
                    "buffer",
                )?;
                let off = v.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of);
                slice(bytes(b)?, off, len).with_context(ErrorKind::Truncated, || {
                    format!("buffer view {i} runs past the end of its buffer")
                })?;
            }
            total = total.saturating_add(len.next_multiple_of(4));
        }
        let mut bin = Vec::with_capacity(total);
        let mut placed = Vec::with_capacity(views.len());
        for (i, v) in views.iter().enumerate() {
            let len = view_len(v)?;
            pad4(&mut bin);
            let at = bin.len();
            if let Some((name, ext)) = meshopt_of(v) {
                let src = slice(
                    bytes(get_usize(ext, "buffer")?)?,
                    ext.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of),
                    get_usize(ext, "byteLength")?,
                )
                .with_context(ErrorKind::Truncated, || {
                    format!("buffer view {i}: {name} data runs past the end of its buffer")
                })?;
                bin.resize(at + len, 0);
                decode_view(ext, src, &mut bin[at..]).with_within(|| format!("buffer view {i}"))?;
            } else {
                let src = slice(
                    bytes(usize_of(
                        v.get("buffer")
                            .and_then(Value::as_u64)
                            .context(ErrorKind::Corrupt, "missing buffer")?,
                    ))?,
                    v.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of),
                    len,
                )
                .with_context(ErrorKind::Truncated, || {
                    format!("buffer view {i} runs past the end of its buffer")
                })?;
                bin.extend_from_slice(src);
            }
            placed.push(at);
        }
        drop(data);

        let views = self.json.get_mut("bufferViews").and_then(Value::as_array_mut);
        for (v, at) in views.into_iter().flatten().zip(placed) {
            let v = v
                .as_object_mut()
                .context(ErrorKind::Corrupt, "buffer view is not an object")?;
            v.insert("buffer".into(), json!(0));
            v.insert("byteOffset".into(), json!(at));
            remove_extensions(v, &MESHOPT_EXTENSIONS);
        }
        unlist_extensions(&mut self.json, &MESHOPT_EXTENSIONS);
        if embed_images {
            embed_images_into(&mut self.json, &mut bin, fetch)?;
        }
        self.finish(&bin, Vec::new())
    }

    /// Every buffer into the binary chunk, compression kept.
    fn pack(mut self, fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>) -> Result<Vec<u8>> {
        let buffers = self.list("buffers").to_vec();
        let mut bin: Vec<u8> = Vec::new();
        // Each old buffer's new home: (buffer, offset).
        let mut home = Vec::with_capacity(buffers.len());
        let mut placeholders = Vec::new();
        for (i, b) in buffers.iter().enumerate() {
            match self.buffer(i, fetch, true)? {
                Data::Bytes(d) => {
                    pad4(&mut bin);
                    home.push((0, bin.len()));
                    bin.extend_from_slice(&d);
                }
                Data::Placeholder => {
                    let mut p = b.as_object().cloned().unwrap_or_default();
                    p.remove("uri");
                    placeholders.push(Value::Object(p));
                    home.push((placeholders.len(), 0));
                }
            }
        }
        let mut required = Vec::new();
        let views = self.json.get_mut("bufferViews").and_then(Value::as_array_mut);
        for (i, v) in views.into_iter().flatten().enumerate() {
            let v = v
                .as_object_mut()
                .context(ErrorKind::Corrupt, "buffer view is not an object")?;
            let (vb, _) = rehome(v, &home).with_within(|| format!("buffer view {i}"))?;
            let exts = v.get_mut("extensions").and_then(Value::as_object_mut);
            for (name, ext) in exts.into_iter().flatten() {
                if !MESHOPT_EXTENSIONS.contains(&name.as_str()) {
                    continue;
                }
                let ext = ext
                    .as_object_mut()
                    .context(ErrorKind::Corrupt, "bad meshopt extension")?;
                let (b, _) = rehome(ext, &home).with_within(|| format!("buffer view {i}"))?;
                ensure!(
                    b == 0,
                    Corrupt,
                    "buffer view {i}: compressed data is in a missing buffer"
                );
                if vb != 0 {
                    required.push(name.clone());
                }
            }
        }
        // Views on a placeholder need the extension to be read at all.
        for name in required {
            require_extension(&mut self.json, &name);
        }
        embed_images_into(&mut self.json, &mut bin, fetch)?;
        self.finish(&bin, placeholders)
    }

    /// Meshopt-encodes the index and vertex views of a plain GLB.
    fn compress(mut self, options: CompressOptions) -> Result<Vec<u8>> {
        let bin = self.bin.unwrap_or_default();
        let plans = plan_compression(&self, options.color_filter)?;
        let name = options.extension.name();
        let version = u8::from(options.extension == MeshoptExtension::Khr);
        let level = options.level.min(3);
        let views = self.list("bufferViews").to_vec();
        let mut out = Vec::with_capacity(bin.len() / 2);
        let mut fallback_len = 0usize;
        let mut new_views = Vec::with_capacity(views.len());
        for (i, (v, plan)) in views.iter().zip(&plans).enumerate() {
            let len = view_len(v)?;
            let off = v.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of);
            let src = slice(bin, off, len).with_context(ErrorKind::Corrupt, || format!("buffer view {i}"))?;
            let mut v = v
                .as_object()
                .cloned()
                .context(ErrorKind::Corrupt, "buffer view is not an object")?;
            pad4(&mut out);
            let at = out.len();
            match plan {
                Plan::Keep => {
                    out.extend_from_slice(src);
                    v.insert("buffer".into(), json!(0));
                    v.insert("byteOffset".into(), json!(at));
                }
                Plan::Encode { mode, stride, color } => {
                    let stride = *stride;
                    let count = len / stride;
                    let encoded = if *color {
                        meshopt::encode_vertex_buffer_level(&color_filtered(src, stride), stride, level, version)
                    } else if *mode == "ATTRIBUTES" {
                        meshopt::encode_vertex_buffer_level(src, stride, level, version)
                    } else {
                        let idx: Vec<u32> = if stride == 2 {
                            src.chunks_exact(2)
                                .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                                .collect()
                        } else {
                            src.chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        };
                        if *mode == "TRIANGLES" {
                            meshopt::encode_index_buffer(&idx)
                        } else {
                            meshopt::encode_index_sequence(&idx)?
                        }
                    };
                    out.extend_from_slice(&encoded);
                    fallback_len = fallback_len.next_multiple_of(4);
                    v.insert("buffer".into(), json!(1));
                    v.insert("byteOffset".into(), json!(fallback_len));
                    fallback_len += len;
                    let mut ext = json!({
                        "buffer": 0,
                        "byteOffset": at,
                        "byteLength": encoded.len(),
                        "byteStride": stride,
                        "count": count,
                        "mode": mode,
                    });
                    if *color {
                        ext["filter"] = json!("COLOR");
                    }
                    v.entry("extensions")
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .context(ErrorKind::Corrupt, "bad buffer view extensions")?
                        .insert(name.into(), ext);
                }
            }
            new_views.push(Value::Object(v));
        }
        let mut placeholders = Vec::new();
        if plans.iter().any(|p| !matches!(p, Plan::Keep)) {
            placeholders.push(json!({
                "byteLength": fallback_len,
                "extensions": { name: { "fallback": true } },
            }));
            require_extension(&mut self.json, name);
        }
        self.json.insert("bufferViews".into(), Value::Array(new_views));
        self.finish(&out, placeholders)
    }

    /// The GLB: `bin` as buffer 0, then `extra` URI-less buffers.
    fn finish(mut self, bin: &[u8], extra: Vec<Value>) -> Result<Vec<u8>> {
        let has_views = !self.list("bufferViews").is_empty();
        if bin.is_empty() && extra.is_empty() && !has_views {
            self.json.remove("buffers");
            return write_glb(&Value::Object(self.json), None);
        }
        let mut buffers = vec![json!({ "byteLength": bin.len() })];
        buffers.extend(extra);
        self.json.insert("buffers".into(), Value::Array(buffers));
        write_glb(&Value::Object(self.json), Some(bin))
    }
}

/// What [`Doc::compress`] does with a buffer view.
enum Plan {
    Keep,
    /// Encode in `mode`, elements of `stride` bytes, with the `COLOR` filter first if `color`.
    Encode {
        mode: &'static str,
        stride: usize,
        color: bool,
    },
}

/// A view of normalized RGBA colors (unsigned bytes at stride 4, unsigned shorts at 8) as the
/// `COLOR` filter's input at full width.
fn color_filtered(src: &[u8], stride: usize) -> Vec<u8> {
    let colors: Vec<[f32; 4]> = if stride == 4 {
        src.chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]].map(|v| f32::from(v) / 255.0))
            .collect()
    } else {
        src.chunks_exact(8)
            .map(|c| [0, 1, 2, 3].map(|i| f32::from(u16::from_le_bytes([c[2 * i], c[2 * i + 1]])) / 65535.0))
            .collect()
    };
    meshopt::encode_filter_color(&colors, stride, if stride == 4 { 8 } else { 16 })
}

/// How each buffer view of a plain glTF can be compressed, from what reads it; with `colors`,
/// which views the `COLOR` filter takes.
fn plan_compression(doc: &Doc, colors: bool) -> Result<Vec<Plan>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Use {
        None,
        /// Primitive indices: element size, and whether every reader is a triangle list.
        Index(usize, bool),
        /// Vertex attributes: element size, and whether every reader is a color the `COLOR`
        /// filter takes (a `COLOR_n` attribute, normalized unsigned `VEC4`, at offset 0).
        Vertex(usize, bool),
        Other,
    }
    let accessors = doc.list("accessors");
    let views = doc.list("bufferViews");
    let mut role = vec![Use::None; accessors.len()];
    let mut mark = |a: Option<&Value>, u: Use| {
        if let Some(r) = a.and_then(Value::as_u64).and_then(|a| role.get_mut(usize_of(a))) {
            *r = match (*r, u) {
                (Use::None, u) => u,
                (Use::Index(s, t), Use::Index(_, t2)) => Use::Index(s, t && t2),
                (Use::Vertex(s, c), Use::Vertex(_, c2)) => Use::Vertex(s, c && c2),
                _ => Use::Other,
            };
        }
    };
    for m in doc.list("meshes") {
        for p in m.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            let triangles = p.get("mode").and_then(Value::as_u64).unwrap_or(4) == 4;
            mark(p.get("indices"), Use::Index(0, triangles));
            let targets = p.get("targets").and_then(Value::as_array);
            let sets = std::iter::once(p.get("attributes")).chain(targets.into_iter().flatten().map(Some));
            // Morph target colors are deltas, not colors.
            for (i, set) in sets.enumerate() {
                for (name, a) in set.and_then(Value::as_object).into_iter().flatten() {
                    mark(Some(a), Use::Vertex(0, i == 0 && name.starts_with("COLOR_")));
                }
            }
        }
    }
    let mut plans: Vec<Option<Use>> = vec![None; views.len()];
    for (a, acc) in accessors.iter().enumerate() {
        let sparse = acc.get("sparse").is_some();
        let Some(v) = acc.get("bufferView").and_then(Value::as_u64).map(usize_of) else {
            continue;
        };
        let Some(slot) = plans.get_mut(v) else {
            continue;
        };
        let size = element_size(acc);
        let off = acc.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of);
        let count = acc.get("count").and_then(Value::as_u64).map_or(0, usize_of);
        let u = match (role[a], size) {
            _ if sparse => Use::Other,
            (Use::Index(_, tri), Some(s @ (2 | 4))) => {
                let aligned = off.is_multiple_of(3 * s) && count.is_multiple_of(3);
                Use::Index(s, tri && aligned)
            }
            (Use::Vertex(_, color), Some(s)) => {
                let normalized = acc.get("normalized").and_then(Value::as_bool) == Some(true);
                let unsigned = matches!(acc.get("componentType").and_then(Value::as_u64), Some(5121 | 5123));
                let vec4 = acc.get("type").and_then(Value::as_str) == Some("VEC4");
                Use::Vertex(s, color && normalized && unsigned && vec4 && off == 0)
            }
            _ => Use::Other,
        };
        *slot = Some(match (*slot, u) {
            (None, u) => u,
            (Some(Use::Index(s, t)), Use::Index(s2, t2)) if s == s2 => Use::Index(s, t && t2),
            (Some(Use::Vertex(s, c)), Use::Vertex(s2, c2)) if s == s2 => Use::Vertex(s, c && c2),
            (Some(Use::Vertex(..)), Use::Vertex(..)) if views[v].get("byteStride").is_some() => Use::Vertex(0, false),
            _ => Use::Other,
        });
    }
    // Sparse accessors read views for their indices and values: leave those alone.
    for acc in accessors {
        let Some(sparse) = acc.get("sparse") else {
            continue;
        };
        for part in ["indices", "values"] {
            let v = sparse
                .get(part)
                .and_then(|p| p.get("bufferView"))
                .and_then(Value::as_u64);
            if let Some(slot) = v.and_then(|v| plans.get_mut(usize_of(v))) {
                *slot = Some(Use::Other);
            }
        }
    }
    views
        .iter()
        .zip(plans)
        .map(|(v, u)| {
            let len = view_len(v)?;
            let stride = v.get("byteStride").and_then(Value::as_u64).map(usize_of);
            Ok(match u {
                Some(Use::Index(s, tri)) if stride.is_none() && len.is_multiple_of(s) => {
                    let tri = tri && len.is_multiple_of(3 * s);
                    Plan::Encode {
                        mode: if tri { "TRIANGLES" } else { "INDICES" },
                        stride: s,
                        color: false,
                    }
                }
                Some(Use::Vertex(size, color)) => match stride.unwrap_or(size) {
                    s if s > 0 && s <= 256 && s.is_multiple_of(4) && len.is_multiple_of(s) => Plan::Encode {
                        mode: "ATTRIBUTES",
                        stride: s,
                        // Tightly packed, so every byte of the view is a color channel.
                        color: colors && color && s == size,
                    },
                    _ => Plan::Keep,
                },
                _ => Plan::Keep,
            })
        })
        .collect()
}

/// An accessor's element size in bytes, unpadded.
fn element_size(acc: &Value) -> Option<usize> {
    let component = match acc.get("componentType")?.as_u64()? {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        _ => return None,
    };
    let n = match acc.get("type")?.as_str()? {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" | "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => return None,
    };
    // Matrix columns are 4-byte aligned.
    Some(match (acc.get("type")?.as_str()?, component) {
        ("MAT2", 1) => 8,
        ("MAT3", 1) => 12,
        ("MAT3", 2) => 24,
        _ => n * component,
    })
}

/// Decodes one compressed buffer view into `out`, its uncompressed length.
pub(crate) fn decode_view(ext: &Map<String, Value>, src: &[u8], out: &mut [u8]) -> Result<()> {
    let count = get_usize(ext, "count")?;
    let stride = get_usize(ext, "byteStride")?;
    ensure!(
        count.checked_mul(stride) == Some(out.len()),
        Corrupt,
        "meshopt count {count} x stride {stride} doesn't match the view's {} bytes",
        out.len()
    );
    let mode = ext.get("mode").and_then(Value::as_str).unwrap_or("");
    let filter = ext.get("filter").and_then(Value::as_str).unwrap_or("NONE");
    let indices = match mode {
        "ATTRIBUTES" => {
            ensure!(out.len() / 1024 <= src.len(), Truncated, "meshopt data is cut short");
            meshopt::decode_vertex_buffer_into(src, out, stride)?;
            return meshopt::apply_filter(filter, out, stride);
        }
        "TRIANGLES" => meshopt::decode_index_buffer(src, count)?,
        "INDICES" => meshopt::decode_index_sequence(src, count)?,
        m => bail!(Corrupt, "unknown meshopt mode {m:?}"),
    };
    ensure!(filter == "NONE", Corrupt, "meshopt filter {filter} on indices");
    match stride {
        2 => {
            for (o, i) in out.chunks_exact_mut(2).zip(indices) {
                let i = u16::try_from(i).map_err(|_| err!(Corrupt, "meshopt index {i} in a 16-bit view"))?;
                o.copy_from_slice(&i.to_le_bytes());
            }
        }
        4 => {
            for (o, i) in out.chunks_exact_mut(4).zip(indices) {
                o.copy_from_slice(&i.to_le_bytes());
            }
        }
        s => bail!(Corrupt, "meshopt index stride {s}"),
    }
    Ok(())
}

/// Moves images with a URI (data or external) into the binary chunk.
fn embed_images_into(
    json: &mut Map<String, Value>,
    bin: &mut Vec<u8>,
    fetch: &mut dyn FnMut(&str) -> Result<Vec<u8>>,
) -> Result<()> {
    let mut new_views = Vec::new();
    let first = json.get("bufferViews").and_then(Value::as_array).map_or(0, Vec::len);
    let images = json.get_mut("images").and_then(Value::as_array_mut);
    for (i, img) in images.into_iter().flatten().enumerate() {
        let Some(img) = img.as_object_mut() else {
            continue;
        };
        let Some(uri) = img.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let (declared, bytes) = if uri.starts_with("data:") {
            data_uri(uri).with_within(|| format!("image {i}"))?
        } else {
            let path = percent_decode(uri);
            let bytes = fetch(&path).with_within(|| format!("image {i}"))?;
            (mime_from_name(&path).map(str::to_owned), bytes)
        };
        let mime = sniff_mime(&bytes)
            .map(str::to_owned)
            .or_else(|| img.get("mimeType").and_then(Value::as_str).map(str::to_owned))
            .or(declared)
            .with_context(ErrorKind::Unsupported, || format!("image {i}: unknown image format"))?;
        pad4(bin);
        new_views.push(json!({ "buffer": 0, "byteOffset": bin.len(), "byteLength": bytes.len() }));
        bin.extend_from_slice(&bytes);
        img.remove("uri");
        img.insert("bufferView".into(), json!(first + new_views.len() - 1));
        img.insert("mimeType".into(), json!(mime));
    }
    if !new_views.is_empty() {
        json.entry("bufferViews")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .context(ErrorKind::Corrupt, "bufferViews is not a list")?
            .extend(new_views);
    }
    Ok(())
}

/// Points an object's `buffer`/`byteOffset` at its buffer's new home; returns the new buffer.
pub(crate) fn rehome(obj: &mut Map<String, Value>, home: &[(usize, usize)]) -> Result<(usize, usize)> {
    let b = get_usize(obj, "buffer")?;
    let &(nb, base) = home
        .get(b)
        .with_context(ErrorKind::Corrupt, || format!("missing buffer {b}"))?;
    let off = obj.get("byteOffset").and_then(Value::as_u64).map_or(0, usize_of);
    let at = off
        .checked_add(base)
        .context(ErrorKind::Truncated, "byteOffset is too large")?;
    obj.insert("buffer".into(), json!(nb));
    if nb == 0 || off != 0 {
        obj.insert("byteOffset".into(), json!(at));
    }
    Ok((nb, at))
}

/// A buffer view's meshopt extension, if it has one.
pub(crate) fn meshopt_of(v: &Value) -> Option<(&'static str, &Map<String, Value>)> {
    let exts = v.get("extensions")?.as_object()?;
    MESHOPT_EXTENSIONS
        .iter()
        .find_map(|&n| Some((n, exts.get(n)?.as_object()?)))
}

pub(crate) fn is_fallback(b: &Value) -> bool {
    // Key by key, without building a pointer string for each buffer.
    let Some(exts) = b.get("extensions") else {
        return false;
    };
    MESHOPT_EXTENSIONS.iter().any(|&n| {
        exts.get(n)
            .and_then(|e| e.get("fallback"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    })
}

fn view_len(v: &Value) -> Result<usize> {
    v.get("byteLength")
        .and_then(Value::as_u64)
        .map(usize_of)
        .context(ErrorKind::Corrupt, "buffer view has no byteLength")
}

fn get_usize(obj: &Map<String, Value>, key: &str) -> Result<usize> {
    obj.get(key)
        .and_then(Value::as_u64)
        .map(usize_of)
        .with_context(ErrorKind::Corrupt, || format!("missing {key}"))
}

fn slice(data: &[u8], off: usize, len: usize) -> Option<&[u8]> {
    data.get(off..off.checked_add(len)?)
}

fn pad4(bin: &mut Vec<u8>) {
    bin.resize(bin.len().next_multiple_of(4), 0);
}

fn remove_extensions(obj: &mut Map<String, Value>, names: &[&str]) {
    if let Some(exts) = obj.get_mut("extensions").and_then(Value::as_object_mut) {
        for n in names {
            exts.remove(*n);
        }
        if exts.is_empty() {
            obj.remove("extensions");
        }
    }
}

fn unlist_extensions(json: &mut Map<String, Value>, names: &[&str]) {
    for key in ["extensionsUsed", "extensionsRequired"] {
        if let Some(list) = json.get_mut(key).and_then(Value::as_array_mut) {
            list.retain(|e| !e.as_str().is_some_and(|e| names.contains(&e)));
            if list.is_empty() {
                json.remove(key);
            }
        }
    }
}

pub(crate) fn require_extension(json: &mut Map<String, Value>, name: &str) {
    for key in ["extensionsUsed", "extensionsRequired"] {
        let list = json.entry(key).or_insert_with(|| json!([]));
        if let Some(list) = list.as_array_mut()
            && !list.iter().any(|e| e.as_str() == Some(name))
        {
            list.push(json!(name));
        }
    }
}

/// Frames JSON and an optional binary chunk as a GLB.
#[allow(
    clippy::cast_possible_truncation,
    reason = "each chunk is smaller than `total`, which fits"
)]
pub(crate) fn write_glb(json: &Value, bin: Option<&[u8]>) -> Result<Vec<u8>> {
    let mut text = serde_json::to_vec(json).context(ErrorKind::Invalid, "writing the glTF JSON")?;
    text.resize(text.len().next_multiple_of(4), b' ');
    let bin_len = bin.map_or(0, |b| 8 + b.len().next_multiple_of(4));
    let total = 12 + 8 + text.len() + bin_len;
    let total32 = u32::try_from(total).map_err(|_| err!(LimitExceeded, "GLB would be over 4 GiB"))?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(GLB_MAGIC);
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total32.to_le_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(CHUNK_JSON);
    out.extend_from_slice(&text);
    if let Some(bin) = bin {
        let padded = bin.len().next_multiple_of(4);
        out.extend_from_slice(&(padded as u32).to_le_bytes());
        out.extend_from_slice(CHUNK_BIN);
        out.extend_from_slice(bin);
        out.resize(out.len() + padded - bin.len(), 0);
    }
    Ok(out)
}

/// A data URI's MIME type and bytes.
pub(crate) fn data_uri(uri: &str) -> Result<(Option<String>, Vec<u8>)> {
    let rest = uri
        .strip_prefix("data:")
        .context(ErrorKind::Corrupt, "not a data URI")?;
    let (head, payload) = rest
        .split_once(',')
        .context(ErrorKind::Corrupt, "data URI has no ','")?;
    let (mime, base64) = match head.strip_suffix(";base64") {
        Some(m) => (m, true),
        None => (head, false),
    };
    let mime = mime.split(';').next().filter(|m| !m.is_empty()).map(str::to_owned);
    let bytes = if base64 {
        decode_base64(payload)?
    } else {
        percent_decode_bytes(payload)
    };
    Ok((mime, bytes))
}

/// Standard or URL-safe base64, padding optional.
fn decode_base64(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => bail!(Corrupt, "bad base64 in data URI"),
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            #[allow(clippy::cast_possible_truncation, reason = "the low byte is the decoded one")]
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn percent_decode_bytes(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && let (Some(h), Some(l)) = (b.get(i + 1).and_then(|&c| hex(c)), b.get(i + 2).and_then(|&c| hex(c)))
        {
            #[allow(clippy::cast_possible_truncation, reason = "two hex digits are one byte")]
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

pub(crate) fn percent_decode(s: &str) -> String {
    match percent_decode_bytes(s) {
        b if b == s.as_bytes() => s.to_owned(),
        b => String::from_utf8_lossy(&b).into_owned(),
    }
}

/// An image's MIME type from its first bytes.
pub(crate) fn sniff_mime(b: &[u8]) -> Option<&'static str> {
    Some(match b {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xff, 0xd8, 0xff, ..] => "image/jpeg",
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => "image/webp",
        [0xab, b'K', b'T', b'X', b' ', b'2', b'0', 0xbb, ..] => "image/ktx2",
        _ => return None,
    })
}

pub(crate) fn mime_from_name(name: &str) -> Option<&'static str> {
    let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ktx2" => "image/ktx2",
        _ => return None,
    })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test grids are small")]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::mesh::{Image, Material, Mesh, Prim, to_glb};

    /// A grid with normals, UVs and a (fake) PNG texture, as a GLB.
    fn grid_glb(n: u32) -> Vec<u8> {
        let mut prim = Prim::default();
        for y in 0..=n {
            for x in 0..=n {
                let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
                prim.positions.push([u, (u * 7.0).sin() * 0.1, v]);
                prim.normals.push([0.0, 1.0, 0.0]);
                prim.uvs.push([u, v]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                prim.indices.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend([7u8; 37]);
        let mut material = Material::plain("grid", [1.0, 0.5, 0.25, 1.0]);
        material.color_image = Some(0);
        let mesh = Mesh {
            prims: vec![prim],
            materials: vec![material],
            images: vec![Image {
                bytes: png.into(),
                mime: "image/png",
            }],
            skeleton: Vec::new(),
        };
        to_glb(&mesh).unwrap()
    }

    fn u(v: &Value) -> usize {
        usize::try_from(v.as_u64().unwrap()).unwrap()
    }

    /// Accessor `a`'s elements, tightly packed.
    fn accessor(glb: &[u8], a: usize) -> Vec<u8> {
        let doc = Doc::parse(glb).unwrap();
        let acc = &doc.list("accessors")[a];
        let v = &doc.list("bufferViews")[u(&acc["bufferView"])];
        let size = element_size(acc).unwrap();
        let stride = v.get("byteStride").map_or(size, u);
        let base = v.get("byteOffset").map_or(0, u) + acc.get("byteOffset").map_or(0, u);
        let bin = doc.bin.unwrap();
        (0..u(&acc["count"]))
            .flat_map(|i| bin[base + i * stride..base + i * stride + size].to_vec())
            .collect()
    }

    /// Triangles from u32 index bytes, each starting at its lowest corner.
    fn triangles(bytes: &[u8]) -> Vec<[u32; 3]> {
        let idx: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        idx.chunks_exact(3)
            .map(|t| {
                let r = (0..3).min_by_key(|&k| t[k]).unwrap();
                [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
            })
            .collect()
    }

    fn json_of(glb: &[u8]) -> Map<String, Value> {
        Doc::parse(glb).unwrap().json
    }

    /// The bytes of image 0.
    fn image(glb: &[u8]) -> Vec<u8> {
        let d = Doc::parse(glb).unwrap();
        let v = &d.list("bufferViews")[u(&d.list("images")[0]["bufferView"])];
        let off = v.get("byteOffset").map_or(0, u);
        d.bin.unwrap()[off..off + view_len(v).unwrap()].to_vec()
    }

    /// Every accessor of `a` and `b` holds the same data (triangles up to their first corner).
    fn same_data(a: &[u8], b: &[u8]) {
        let (ja, jb) = (json_of(a), json_of(b));
        assert_eq!(ja["accessors"], jb["accessors"]);
        for (i, acc) in ja["accessors"].as_array().unwrap().iter().enumerate() {
            let (da, db) = (accessor(a, i), accessor(b, i));
            if acc["type"] == "SCALAR" && acc["componentType"] == 5125 {
                assert_eq!(triangles(&da), triangles(&db), "accessor {i}");
            } else {
                assert_eq!(da, db, "accessor {i}");
            }
        }
        assert_eq!(image(a), image(b));
    }

    #[test]
    fn compress_then_decompress_roundtrips() {
        let plain = grid_glb(40);
        let packed = compress(&plain).unwrap();
        assert!(packed.len() * 2 < plain.len(), "{} vs {}", packed.len(), plain.len());
        let j = json_of(&packed);
        assert_eq!(j["extensionsRequired"], json!(["EXT_meshopt_compression"]));
        assert_eq!(j["buffers"].as_array().unwrap().len(), 2);
        assert_eq!(
            j["buffers"][1]["extensions"]["EXT_meshopt_compression"]["fallback"],
            true
        );
        let modes: Vec<&str> = j["bufferViews"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.pointer("/extensions/EXT_meshopt_compression/mode")?.as_str())
            .collect();
        assert_eq!(modes, ["ATTRIBUTES", "ATTRIBUTES", "TRIANGLES", "ATTRIBUTES"]);
        // A plain glTF reader refuses the compressed file and takes the decompressed one.
        assert!(::gltf::Gltf::from_slice(&packed).is_err());
        let back = decompress(&packed).unwrap();
        let g = ::gltf::Gltf::from_slice(&back).unwrap();
        assert!(g.extensions_used().next().is_none());
        assert_eq!(g.buffers().count(), 1);
        same_data(&plain, &back);
        // Compressing twice is the same as compressing once.
        assert_eq!(compress(&packed).unwrap(), packed);
    }

    #[test]
    fn khr_compression_roundtrips() {
        let plain = grid_glb(40);
        let ext = compress(&plain).unwrap();
        for level in [0, 2, 3] {
            let options = CompressOptions {
                level,
                ..CompressOptions::KHR
            };
            let packed = compress_with(&plain, options).unwrap();
            let j = json_of(&packed);
            assert_eq!(j["extensionsUsed"], json!(["KHR_meshopt_compression"]));
            assert_eq!(j["extensionsRequired"], json!(["KHR_meshopt_compression"]));
            assert_eq!(
                j["buffers"][1]["extensions"]["KHR_meshopt_compression"]["fallback"],
                true
            );
            assert!(!serde_json::to_string(&j).unwrap().contains("EXT_meshopt"));
            // Attribute views use codec version 1.
            let bin = Doc::parse(&packed).unwrap().bin.unwrap();
            for v in j["bufferViews"].as_array().unwrap() {
                if let Some(e) = v.pointer("/extensions/KHR_meshopt_compression")
                    && e["mode"] == "ATTRIBUTES"
                {
                    assert_eq!(bin[u(&e["byteOffset"])], 0xa1);
                }
            }
            if level >= 2 {
                assert!(
                    packed.len() <= ext.len(),
                    "level {level}: {} vs {}",
                    packed.len(),
                    ext.len()
                );
            }
            same_data(&plain, &decompress(&packed).unwrap());
            // Recompressing EXT as KHR goes through the plain file.
            assert_eq!(compress_with(&ext, options).unwrap(), packed);
        }
        let colors = CompressOptions {
            color_filter: true,
            ..CompressOptions::EXT
        };
        assert!(compress_with(&plain, colors).is_err());
        assert_eq!(MeshoptExtension::Khr.name(), "KHR_meshopt_compression");
        assert_eq!(CompressOptions::default(), CompressOptions::EXT);
    }

    /// Triangles with 8- and 16-bit colors, float colors, and an 8-bit custom attribute.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "colors in 0 to 1 scaled to their integer range, and bytes that wrap on purpose"
    )]
    fn colored_glb(n: usize) -> Vec<u8> {
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut view = |bytes: Vec<u8>| {
            views.push(json!({ "buffer": 0, "byteOffset": bin.len(), "byteLength": bytes.len() }));
            bin.extend(bytes);
            views.len() - 1
        };
        let wave = |i: usize, k: usize| (i as f32 * 0.05 + k as f32).sin() * 0.5 + 0.5;
        let pos = view((0..n * 3).flat_map(|i| (i as f32).to_le_bytes()).collect());
        let c8 = view((0..n * 4).map(|i| (wave(i / 4, i % 4) * 255.0) as u8).collect());
        let c16 = view(
            (0..n * 4)
                .flat_map(|i| ((wave(i / 4, i % 4) * 65535.0) as u16).to_le_bytes())
                .collect(),
        );
        let cf = view((0..n * 4).flat_map(|i| wave(i / 4, i % 4).to_le_bytes()).collect());
        let custom = view((0..n * 4).map(|i| (i * 7) as u8).collect());
        let acc = |view: usize, ty: u32, kind: &str, normalized: bool| json!({ "bufferView": view, "componentType": ty, "type": kind, "count": n, "normalized": normalized });
        let json = json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": views,
            "accessors": [
                acc(pos, 5126, "VEC3", false),
                acc(c8, 5121, "VEC4", true),
                acc(c16, 5123, "VEC4", true),
                acc(cf, 5126, "VEC4", false),
                acc(custom, 5121, "VEC4", true),
            ],
            "meshes": [{ "primitives": [{ "attributes": {
                "POSITION": 0, "COLOR_0": 1, "COLOR_1": 2, "COLOR_2": 3, "_CUSTOM": 4,
            } }] }],
        });
        write_glb(&json, Some(&bin)).unwrap()
    }

    #[test]
    fn color_filter_is_close() {
        let plain = colored_glb(300);
        let options = CompressOptions {
            level: 3,
            color_filter: true,
            ..CompressOptions::KHR
        };
        let packed = compress_with(&plain, options).unwrap();
        let j = json_of(&packed);
        let filters: Vec<&str> = j["bufferViews"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                v.pointer("/extensions/KHR_meshopt_compression/filter")
                    .map_or("NONE", |f| f.as_str().unwrap())
            })
            .collect();
        assert_eq!(filters, ["NONE", "COLOR", "COLOR", "NONE", "NONE"]);
        let back = decompress(&packed).unwrap();
        assert_eq!(json_of(&back)["accessors"], json_of(&plain)["accessors"]);
        for a in [0, 3, 4] {
            assert_eq!(accessor(&back, a), accessor(&plain, a), "accessor {a}");
        }
        let (want, got) = (accessor(&plain, 1), accessor(&back, 1));
        assert!(want.iter().zip(&got).all(|(a, b)| a.abs_diff(*b) <= 1));
        assert_ne!(want, got);
        let (want, got) = (halves(&accessor(&plain, 2)), halves(&accessor(&back, 2)));
        assert!(want.iter().zip(&got).all(|(a, b)| a.abs_diff(*b) <= 1));
        // Without the filter, the same options are lossless.
        let lossless = compress_with(
            &plain,
            CompressOptions {
                color_filter: false,
                ..options
            },
        )
        .unwrap();
        let back = decompress(&lossless).unwrap();
        for a in 0..5 {
            assert_eq!(accessor(&back, a), accessor(&plain, a), "accessor {a}");
        }
    }

    fn halves(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn plain_glb_passes_through() {
        let plain = grid_glb(3);
        assert_eq!(decompress(&plain).unwrap(), plain);
    }

    /// A compressed GLB split into a `.gltf`, its binary as an external file (with a space in
    /// its name), a fallback buffer that isn't there and an external texture.
    fn split(packed: &[u8]) -> (Vec<u8>, HashMap<String, Vec<u8>>) {
        let doc = Doc::parse(packed).unwrap();
        let mut j = doc.json.clone();
        j["buffers"][0]["uri"] = json!("scene%20data.bin");
        j["buffers"][1]["uri"] = json!("fallback.bin");
        let png = image(packed);
        j["images"][0] = json!({ "uri": "tex/albedo.png" });
        let mut files = HashMap::new();
        files.insert("scene data.bin".to_owned(), doc.bin.unwrap().to_vec());
        files.insert("tex/albedo.png".to_owned(), png);
        (serde_json::to_vec(&j).unwrap(), files)
    }

    #[test]
    fn pack_keeps_compression_and_rewrites_references() {
        let packed = compress(&grid_glb(12)).unwrap();
        let (gltf, files) = split(&packed);
        let mut asked = Vec::new();
        let fetch = |uri: &str| {
            asked.push(uri.to_owned());
            files.get(uri).cloned().ok_or_else(|| err!(Corrupt, "no {uri}"))
        };
        let glb = pack_glb(&gltf, fetch).unwrap();
        assert_eq!(asked, ["scene data.bin", "fallback.bin", "tex/albedo.png"]);
        let j = json_of(&glb);
        assert_eq!(j["buffers"].as_array().unwrap().len(), 2);
        assert!(j["buffers"][0].get("uri").is_none());
        assert!(j["buffers"][1].get("uri").is_none());
        assert!(j["images"][0].get("uri").is_none());
        assert_eq!(j["images"][0]["mimeType"], "image/png");
        let required = j["extensionsRequired"].as_array().unwrap();
        assert!(required.contains(&json!("EXT_meshopt_compression")));
        same_data(&decompress(&packed).unwrap(), &decompress(&glb).unwrap());
    }

    #[test]
    fn pack_decompressed_never_fetches_fallbacks() {
        let packed = compress(&grid_glb(12)).unwrap();
        let (gltf, files) = split(&packed);
        let mut asked = Vec::new();
        let fetch = |uri: &str| {
            asked.push(uri.to_owned());
            files.get(uri).cloned().ok_or_else(|| err!(Corrupt, "no {uri}"))
        };
        let glb = pack_glb_decompressed(&gltf, fetch).unwrap();
        assert_eq!(asked, ["scene data.bin", "tex/albedo.png"]);
        let g = ::gltf::Gltf::from_slice(&glb).unwrap();
        assert_eq!(g.images().count(), 1);
        same_data(&decompress(&packed).unwrap(), &glb);
    }

    #[test]
    fn missing_file_is_an_error() {
        let packed = compress(&grid_glb(4)).unwrap();
        let (gltf, mut files) = split(&packed);
        files.remove("tex/albedo.png");
        let fetch = |uri: &str| files.get(uri).cloned().ok_or_else(|| err!(Corrupt, "no {uri}"));
        assert!(pack_glb_decompressed(&gltf, fetch).is_err());
        // decompress reads no files at all.
        assert!(decompress(&gltf).is_err());
    }

    fn base64(b: &[u8]) -> String {
        const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in b.chunks(3) {
            let n = (u32::from(c[0]) << 16)
                | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
                | u32::from(*c.get(2).unwrap_or(&0));
            for k in 0..4 {
                s.push(if k <= c.len() {
                    T[(n >> (18 - 6 * k)) as usize & 63] as char
                } else {
                    '='
                });
            }
        }
        s
    }

    #[test]
    fn data_uri_gltf_decompresses() {
        let packed = compress(&grid_glb(5)).unwrap();
        let doc = Doc::parse(&packed).unwrap();
        let mut j = doc.json.clone();
        let uri = format!("data:application/octet-stream;base64,{}", base64(doc.bin.unwrap()));
        j["buffers"][0]["uri"] = json!(uri);
        let back = decompress(&serde_json::to_vec(&j).unwrap()).unwrap();
        same_data(&decompress(&packed).unwrap(), &back);
    }

    /// A `KHR_meshopt_compression` view holding meshoptimizer's version 1 reference stream
    /// (`demo/tests.cpp`): four vertices of three u16s, two u8s and two u16s.
    #[test]
    fn khr_v1_stream_decodes() {
        let stream: &[u8] = &[
            0xa1, 0xee, 0xaa, 0xee, 0x00, 0x4b, 0x4b, 0x4b, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x7d, 0x7d, 0x7d, 0x00, 0x00,
            0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x62,
        ];
        let j = json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_meshopt_compression", "KHR_mesh_quantization"],
            "extensionsRequired": ["KHR_meshopt_compression", "KHR_mesh_quantization"],
            "buffers": [
                { "byteLength": stream.len() },
                { "byteLength": 48, "extensions": { "KHR_meshopt_compression": { "fallback": true } } },
            ],
            "bufferViews": [{
                "buffer": 1, "byteLength": 48, "byteStride": 12,
                "extensions": { "KHR_meshopt_compression": {
                    "buffer": 0, "byteLength": stream.len(), "byteStride": 12, "count": 4,
                    "mode": "ATTRIBUTES",
                } },
            }],
        });
        let glb = write_glb(&j, Some(stream)).unwrap();
        let back = decompress(&glb).unwrap();
        let d = Doc::parse(&back).unwrap();
        // Quantization stays.
        assert_eq!(d.json["extensionsUsed"], json!(["KHR_mesh_quantization"]));
        assert_eq!(d.json["extensionsRequired"], json!(["KHR_mesh_quantization"]));
        assert_eq!(d.json["buffers"], json!([{ "byteLength": 48 }]));
        let mut want = Vec::new();
        for (p, t) in [
            ([0u16, 0, 0], [0u16, 0]),
            ([300, 0, 0], [500, 0]),
            ([0, 300, 0], [0, 500]),
            ([300, 300, 0], [500, 500]),
        ] {
            want.extend(p.iter().flat_map(|v| v.to_le_bytes()));
            want.extend([0u8, 0]);
            want.extend(t.iter().flat_map(|v| v.to_le_bytes()));
        }
        assert_eq!(&d.bin.unwrap()[..48], &want[..]);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn files_in_stays_inside_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("tex")).unwrap();
        std::fs::write(dir.path().join("tex").join("a b.png"), b"png").unwrap();
        let mut fetch = files_in(dir.path());
        assert_eq!(fetch("tex/a b.png").unwrap(), b"png");
        assert_eq!(fetch("./tex/a b.png").unwrap(), b"png");
        for bad in [
            "../secret",
            "tex/../../secret",
            "/etc/passwd",
            "C:/Windows/win.ini",
            "\\\\host\\share\\x",
            "file:///etc/passwd",
        ] {
            assert!(fetch(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn malformed_input_is_an_error() {
        let packed = compress(&grid_glb(4)).unwrap();
        for cut in [0, 3, 11, 20, packed.len() / 2, packed.len() - 1] {
            assert!(decompress(&packed[..cut]).is_err(), "cut at {cut}");
        }
        // A view claiming far more data than its stream could hold is refused, not allocated.
        let mut j = json_of(&packed);
        j["bufferViews"][0]["byteLength"] = json!(12u64 << 40);
        j["bufferViews"][0]["extensions"]["EXT_meshopt_compression"]["count"] = json!(1u64 << 40);
        let bin = Doc::parse(&packed).unwrap().bin.unwrap().to_vec();
        let hostile = write_glb(&Value::Object(j), Some(&bin)).unwrap();
        assert!(decompress(&hostile).is_err());
    }
}
