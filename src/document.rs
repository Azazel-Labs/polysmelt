//! The common form every reader produces: a glTF 2.0 document, its JSON and its buffers' bytes,
//! carrying whatever the source had (scene graph, animation, skins, morph targets, materials,
//! cameras, lights, names, extras).
//!
//! - [`Document::to_glb`] and [`Document::write_glb`] serialize it: every buffer goes into the
//!   GLB's binary chunk as it is, compression included. Nothing is decoded or repacked.
//! - [`Document::flatten`] derives the flat, posed triangle [`Mesh`] (for counting, previews or
//!   tools that want plain triangles), and [`Document::counts`] counts what it draws from the
//!   JSON alone.
//! - [`Document::decode`] rewrites it as plain glTF 2.0: meshopt and Draco decoded, quantized
//!   attributes as floats.
//! - [`Document::prune`] drops what a [`Keep`] doesn't keep (animations, skins, materials, ...);
//!   the loaders in [`formats`](crate::formats) apply it.
//!
//! The JSON's buffers have no `uri`: each has its bytes here, in memory, in a file the document
//! reads as needed (a GLB's binary chunk), or, for a meshopt fallback buffer, nowhere. Images
//! name a buffer view, or keep their `uri` when the file they name wasn't found.

use std::borrow::Cow;
use std::fmt;
#[cfg(feature = "fs")]
use std::fs::File;
use std::io::Write;
#[cfg(feature = "fs")]
use std::sync::Mutex;

use serde_json::{Map, Value, json};

use crate::gltf as container;
use crate::images::{Refs, Source};
use crate::material::{Full, Writer};
use crate::mesh::{Image, Material, Mesh, Prim};
use crate::num::usize_of;
use crate::{Context as _, Result};

pub use crate::keep::Keep;

/// A model as a glTF 2.0 document: its JSON, its buffers' bytes, and facts about the source.
pub struct Document<'a> {
    /// The glTF JSON. Its buffers have no `uri`; their bytes are held by the document (see
    /// [`Document::buffer`] and [`Document::push_buffer`]).
    pub json: Map<String, Value>,
    /// Buffer `i`'s bytes, for each of the JSON's buffers.
    pub(crate) buffers: Vec<Blob<'a>>,
    /// Facts worth recording about the source: `triangles`, and per format `points`, `lines`,
    /// `vertices`, `voxels`, `voxel_size`, `models`, `objects`, `colors`, `application`.
    pub facts: Map<String, Value>,
    /// The images named but not read, by image index (see [`Document::images`]).
    pub(crate) refs: Refs,
}

impl fmt::Debug for Document<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Document")
            .field("json", &self.json)
            .field("buffers", &self.buffers.iter().map(Blob::len).collect::<Vec<_>>())
            .field("facts", &self.facts)
            .finish()
    }
}

/// Which pose [`Document::flatten`] draws.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Pose {
    /// The nodes' own transforms and morph weights.
    #[default]
    Rest,
    /// Animation `index` at `time` seconds: the channels it has override the rest pose; before
    /// its first key or after its last, the first or last value holds.
    Animation { index: usize, time: f32 },
}

/// What a document's default scene draws, every instance counted, as the JSON declares it:
/// indices aren't read, so triangles naming a missing vertex (which [`Document::flatten`] drops)
/// are counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub triangles: usize,
    pub points: usize,
    /// Line segments: a strip or loop of n points is n - 1 or n of them.
    pub lines: usize,
}

/// Where a buffer's bytes are.
pub(crate) enum Blob<'a> {
    /// In memory: borrowed from the input (a GLB's binary chunk, an asset), or owned.
    Bytes(Cow<'a, [u8]>),
    /// A stretch of a file, read as needed.
    #[cfg(feature = "fs")]
    File(FileSpan),
    /// Typed arrays a reader built, laid end to end.
    Parts(Parts),
    /// A meshopt fallback buffer that isn't there: only its length. Views on it decode from
    /// their compressed data.
    #[cfg_attr(not(feature = "gltf"), allow(dead_code, reason = "only the glTF reader makes one"))]
    Missing(u64),
}

/// A stretch of a file: `len` bytes from `base`.
#[cfg(feature = "fs")]
pub(crate) struct FileSpan {
    file: Mutex<Cursor>,
    base: u64,
    len: u64,
}

/// A [`FileSpan`]'s file and what was last read ahead of it.
#[cfg(feature = "fs")]
struct Cursor {
    file: File,
    /// Where the file's cursor is relative to the span's base, to skip needless seeks.
    at: u64,
    /// Bytes read ahead from `ahead_at`: small reads (animation keys, a skin's matrices, little meshes' accessors), one
    /// after another through the file, come from here rather than each its own system call.
    ahead: Vec<u8>,
    ahead_at: u64,
}

/// How far a small read reads ahead.
#[cfg(feature = "fs")]
const AHEAD: usize = 64 * 1024;

#[cfg(feature = "fs")]
impl FileSpan {
    /// `len` bytes of `file` from `base`; its cursor is at `base + at`.
    pub(crate) fn new(file: File, base: u64, len: u64, at: u64) -> Self {
        Self {
            file: Mutex::new(Cursor {
                file,
                at,
                ahead: Vec::new(),
                ahead_at: 0,
            }),
            base,
            len,
        }
    }

    /// Copies the bytes from `off` into `out`, which must lie within the span.
    fn read_at(&self, off: u64, out: &mut [u8]) -> Result<()> {
        let mut c = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let end = off + out.len() as u64;
        if off >= c.ahead_at && end <= c.ahead_at + c.ahead.len() as u64 {
            let from = usize_of(off - c.ahead_at);
            out.copy_from_slice(&c.ahead[from..from + out.len()]);
            return Ok(());
        }
        if out.len() >= AHEAD / 2 {
            return self.read_file(&mut c, off, out);
        }
        let n = usize_of((self.len - off).min(AHEAD as u64));
        let mut ahead = std::mem::take(&mut c.ahead);
        ahead.resize(n, 0);
        c.ahead_at = 0;
        let read = self.read_file(&mut c, off, &mut ahead);
        if read.is_ok() {
            out.copy_from_slice(&ahead[..out.len()]);
            c.ahead_at = off;
        } else {
            ahead.clear();
        }
        c.ahead = ahead;
        read
    }

    fn read_file(&self, c: &mut Cursor, off: u64, out: &mut [u8]) -> Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        if c.at != off {
            c.file.seek(SeekFrom::Start(self.base + off))?;
        }
        // Until the read succeeds, the cursor isn't known.
        c.at = u64::MAX;
        c.file.read_exact(out)?;
        c.at = off + out.len() as u64;
        Ok(())
    }
}

/// Typed arrays laid end to end, each at a 4-byte boundary, read as little-endian bytes in
/// place.
#[derive(Default)]
pub(crate) struct Parts {
    /// Each part and its offset.
    list: Vec<(u64, Part)>,
    len: u64,
}

/// One typed array of a [`Parts`].
pub(crate) enum Part {
    F32(Vec<f32>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    Bytes(Vec<u8>),
}

#[allow(dead_code, reason = "readers use the part types their formats need")]
impl Part {
    /// Floats, stored as they are on a little-endian target (and byte-swapped once otherwise).
    pub(crate) fn f32(v: Vec<f32>) -> Self {
        if cfg!(target_endian = "little") {
            Part::F32(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
    }

    pub(crate) fn u32(v: Vec<u32>) -> Self {
        if cfg!(target_endian = "little") {
            Part::U32(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
    }

    pub(crate) fn u16(v: Vec<u16>) -> Self {
        if cfg!(target_endian = "little") {
            Part::U16(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
    }

    pub(crate) fn bytes(v: Vec<u8>) -> Self {
        Part::Bytes(v)
    }

    /// The part as a GLB stores it. Typed parts exist only on little-endian targets, where
    /// their memory is that.
    fn as_bytes(&self) -> &[u8] {
        match self {
            Part::F32(v) => bytemuck::cast_slice(v),
            Part::U32(v) => bytemuck::cast_slice(v),
            Part::U16(v) => bytemuck::cast_slice(v),
            Part::Bytes(v) => v,
        }
    }
}

impl Parts {
    /// Adds a part at the next 4-byte boundary; returns its offset and length.
    pub(crate) fn push(&mut self, part: Part) -> (u64, u64) {
        let at = self.len.next_multiple_of(4);
        let n = part.as_bytes().len() as u64;
        self.list.push((at, part));
        self.len = at + n;
        (at, n)
    }

    /// The part holding `off`, if any: its index.
    fn part_at(&self, off: u64) -> Option<usize> {
        let i = self.list.partition_point(|(at, _)| *at <= off).checked_sub(1)?;
        Some(i)
    }

    fn slice(&self, off: u64, len: u64) -> Option<&[u8]> {
        let (at, part) = &self.list[self.part_at(off)?];
        let bytes = part.as_bytes();
        let start = usize_of(off - at);
        bytes.get(start..start.checked_add(usize_of(len))?)
    }

    fn read_at(&self, off: u64, out: &mut [u8]) {
        out.fill(0);
        let end = off + out.len() as u64;
        let first = self.part_at(off).unwrap_or(0);
        for (at, part) in &self.list[first..] {
            if *at >= end {
                break;
            }
            let bytes = part.as_bytes();
            let (from, to) = (off.max(*at), end.min(at + bytes.len() as u64));
            if from < to {
                out[usize_of(from - off)..usize_of(to - off)]
                    .copy_from_slice(&bytes[usize_of(from - at)..usize_of(to - at)]);
            }
        }
    }

    fn write_to(&self, len: u64, w: &mut dyn Write) -> Result<()> {
        let mut pos = 0;
        for (at, part) in &self.list {
            if pos >= len {
                break;
            }
            write_zeros(w, (*at).min(len) - pos)?;
            let bytes = part.as_bytes();
            let take = usize_of((len - at).min(bytes.len() as u64));
            w.write_all(&bytes[..take])?;
            pos = at + take as u64;
        }
        write_zeros(w, len.saturating_sub(pos))
    }
}

fn write_zeros(w: &mut dyn Write, n: u64) -> Result<()> {
    use std::io::Read as _;
    std::io::copy(&mut std::io::repeat(0).take(n), w)?;
    Ok(())
}

impl Blob<'_> {
    pub(crate) fn len(&self) -> u64 {
        match self {
            Blob::Bytes(b) => b.len() as u64,
            #[cfg(feature = "fs")]
            Blob::File(f) => f.len,
            Blob::Parts(p) => p.len,
            Blob::Missing(n) => *n,
        }
    }

    /// `len` bytes from `off`, in place, when they are in memory in one piece.
    pub(crate) fn slice(&self, off: u64, len: u64) -> Option<&[u8]> {
        match self {
            Blob::Bytes(b) => b.get(usize_of(off)..usize_of(off.checked_add(len)?)),
            Blob::Parts(p) => p.slice(off, len),
            #[cfg(feature = "fs")]
            Blob::File(_) => None,
            Blob::Missing(_) => None,
        }
    }

    /// Copies the bytes from `off` into `out`.
    pub(crate) fn read_at(&self, off: u64, out: &mut [u8]) -> Result<()> {
        let end = off
            .checked_add(out.len() as u64)
            .filter(|&e| e <= self.len())
            .context("glTF data runs past the end of its buffer")?;
        match self {
            Blob::Bytes(b) => out.copy_from_slice(&b[usize_of(off)..usize_of(end)]),
            #[cfg(feature = "fs")]
            Blob::File(f) => f.read_at(off, out)?,
            Blob::Parts(p) => p.read_at(off, out),
            Blob::Missing(_) => bail!("glTF data is in a meshopt fallback buffer, which holds none"),
        }
        Ok(())
    }

    /// Writes the first `len` bytes.
    fn write_to(&self, len: u64, w: &mut dyn Write) -> Result<()> {
        ensure!(
            len <= self.len(),
            "glTF buffer holds {} bytes but declares {len}",
            self.len()
        );
        match self {
            Blob::Bytes(b) => w.write_all(&b[..usize_of(len)])?,
            #[cfg(feature = "fs")]
            Blob::File(f) => {
                let mut chunk = vec![0; usize_of(len.min(1 << 20))];
                let mut at = 0;
                while at < len {
                    let n = usize_of((len - at).min(chunk.len() as u64));
                    f.read_at(at, &mut chunk[..n])?;
                    w.write_all(&chunk[..n])?;
                    at += n as u64;
                }
            }
            Blob::Parts(p) => p.write_to(len, w)?,
            Blob::Missing(_) => bail!("a meshopt fallback buffer has no bytes to write"),
        }
        Ok(())
    }

    fn into_owned(self) -> Blob<'static> {
        match self {
            Blob::Bytes(b) => Blob::Bytes(Cow::Owned(b.into_owned())),
            #[cfg(feature = "fs")]
            Blob::File(f) => Blob::File(f),
            Blob::Parts(p) => Blob::Parts(p),
            Blob::Missing(n) => Blob::Missing(n),
        }
    }
}

fn list<'v>(obj: &'v Map<String, Value>, key: &str) -> &'v [Value] {
    obj.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

impl<'a> Document<'a> {
    /// A document that draws nothing: only the `asset` glTF requires.
    #[must_use]
    pub fn empty(facts: Map<String, Value>) -> Document<'static> {
        let Value::Object(json) = json!({ "asset": { "version": "2.0", "generator": "polysmelt" } }) else {
            unreachable!()
        };
        Document {
            json,
            buffers: Vec::new(),
            facts,
            refs: Refs::new(),
        }
    }

    /// The document with every borrowed buffer copied, so it outlives its input.
    #[must_use]
    pub fn into_owned(self) -> Document<'static> {
        Document {
            json: self.json,
            buffers: self.buffers.into_iter().map(Blob::into_owned).collect(),
            facts: self.facts,
            refs: self.refs,
        }
    }

    /// Buffer `i`'s bytes, whole (borrowed when they are in memory in one piece).
    ///
    /// # Errors
    ///
    /// When buffer `i` doesn't exist, holds no bytes (a meshopt fallback buffer), or its file
    /// can't be read.
    pub fn buffer(&self, i: usize) -> Result<Cow<'_, [u8]>> {
        let b = self
            .buffers
            .get(i)
            .with_context(|| format!("glTF buffers[{i}] doesn't exist"))?;
        if let Some(s) = b.slice(0, b.len()) {
            return Ok(Cow::Borrowed(s));
        }
        let mut out = vec![0; usize_of(b.len())];
        b.read_at(0, &mut out)?;
        Ok(Cow::Owned(out))
    }

    /// Buffer view `i`'s bytes.
    pub(crate) fn view_bytes(&self, i: usize) -> Result<Cow<'_, [u8]>> {
        let v = list(&self.json, "bufferViews")
            .get(i)
            .with_context(|| format!("glTF bufferViews[{i}] doesn't exist"))?;
        let num = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
        let (offset, len) = (num("byteOffset"), num("byteLength"));
        let b = self
            .buffers
            .get(usize_of(num("buffer")))
            .context("glTF buffer view names a missing buffer")?;
        ensure!(
            offset.saturating_add(len) <= b.len(),
            "glTF buffer view runs past its buffer"
        );
        if let Some(s) = b.slice(offset, len) {
            return Ok(Cow::Borrowed(s));
        }
        let mut out = vec![0; usize_of(len)];
        b.read_at(offset, &mut out)?;
        Ok(Cow::Owned(out))
    }

    /// Adds a buffer holding `bytes`; returns its index.
    pub fn push_buffer(&mut self, bytes: Vec<u8>) -> usize {
        self.push_blob(Blob::Bytes(Cow::Owned(bytes)))
    }

    pub(crate) fn push_blob(&mut self, blob: Blob<'a>) -> usize {
        let buffers = self
            .json
            .entry("buffers")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("glTF buffers is a list");
        buffers.push(json!({ "byteLength": blob.len() }));
        self.buffers.push(blob);
        self.buffers.len() - 1
    }

    /// Adds a buffer view of a whole buffer; returns its index.
    pub(crate) fn push_view(&mut self, view: Value) -> usize {
        let views = self
            .json
            .entry("bufferViews")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("glTF bufferViews is a list");
        views.push(view);
        views.len() - 1
    }

    /// The document as a GLB.
    ///
    /// # Errors
    ///
    /// When a buffer holds fewer bytes than it declares, a buffer's file can't be read, or the
    /// GLB would be over 4 GiB.
    pub fn to_glb(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_glb(&mut out)?;
        Ok(out)
    }

    /// Writes the document as a GLB: the JSON with every buffer's bytes in its binary chunk,
    /// each where its buffer views (and meshopt data) now point.
    ///
    /// # Errors
    ///
    /// As [`Document::to_glb`], and when writing fails.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "each chunk is smaller than `total`, which is checked to fit"
    )]
    pub fn write_glb(&self, w: &mut dyn Write) -> Result<()> {
        let (json, layout, bin_len) = self.glb_json()?;
        let mut text = serde_json::to_vec(&Value::Object(json))?;
        text.resize(text.len().next_multiple_of(4), b' ');
        let padded = bin_len.map(|n| n.next_multiple_of(4));
        let total = 12 + 8 + text.len() as u64 + padded.map_or(0, |n| 8 + n);
        let total = u32::try_from(total).map_err(|_| invalid!("GLB would be over 4 GiB"))?;
        let mut head = Vec::with_capacity(28);
        head.extend_from_slice(b"glTF");
        head.extend_from_slice(&2u32.to_le_bytes());
        head.extend_from_slice(&total.to_le_bytes());
        head.extend_from_slice(&(text.len() as u32).to_le_bytes());
        head.extend_from_slice(b"JSON");
        w.write_all(&head)?;
        w.write_all(&text)?;
        if let Some(padded) = padded {
            w.write_all(&(padded as u32).to_le_bytes())?;
            w.write_all(b"BIN\0")?;
            let mut pos = 0;
            for (i, at, len) in layout {
                write_zeros(w, at - pos)?;
                self.buffers[i].write_to(len, w)?;
                pos = at + len;
            }
            write_zeros(w, padded - pos)?;
        }
        Ok(())
    }

    /// The GLB's JSON, where each buffer with bytes goes in the binary chunk (buffer, offset,
    /// length), and the binary chunk's length (`None` for none).
    #[allow(clippy::type_complexity, reason = "a private step of write_glb")]
    fn glb_json(&self) -> Result<(Map<String, Value>, Vec<(usize, u64, u64)>, Option<u64>)> {
        let mut json = self.json.clone();
        let buffers = list(&json, "buffers");
        ensure!(
            buffers.len() == self.buffers.len(),
            "glTF document lists {} buffers but holds {}",
            buffers.len(),
            self.buffers.len()
        );
        // Each buffer's new home: (buffer, offset). Buffers with bytes share the binary chunk;
        // missing fallbacks follow as buffers without data.
        let mut home = Vec::with_capacity(buffers.len());
        let mut layout = Vec::new();
        let mut placeholders = Vec::new();
        let mut bin_len = 0u64;
        for (i, (b, blob)) in buffers.iter().zip(&self.buffers).enumerate() {
            let declared = b.get("byteLength").and_then(Value::as_u64).unwrap_or(0);
            if let Blob::Missing(_) = blob {
                let mut p = b.as_object().cloned().unwrap_or_default();
                p.remove("uri");
                placeholders.push(Value::Object(p));
                home.push((placeholders.len(), 0));
                continue;
            }
            ensure!(
                blob.len() >= declared,
                "glTF buffer {i} holds {} bytes but declares {declared}",
                blob.len()
            );
            bin_len = bin_len.next_multiple_of(4);
            home.push((0, usize_of(bin_len)));
            layout.push((i, bin_len, declared));
            bin_len += declared;
        }
        let mut required = Vec::new();
        let views = json.get_mut("bufferViews").and_then(Value::as_array_mut);
        for (i, v) in views.into_iter().flatten().enumerate() {
            let v = v
                .as_object_mut()
                .with_context(|| format!("glTF buffer view {i} is not an object"))?;
            let (vb, _) = container::rehome(v, &home).with_context(|| format!("glTF buffer view {i}"))?;
            let exts = v.get_mut("extensions").and_then(Value::as_object_mut);
            for (name, ext) in exts.into_iter().flatten() {
                if !container::MESHOPT_EXTENSIONS.contains(&name.as_str()) {
                    continue;
                }
                let ext = ext.as_object_mut().context("bad meshopt extension")?;
                let (b, _) = container::rehome(ext, &home).with_context(|| format!("glTF buffer view {i}"))?;
                ensure!(
                    b == 0,
                    "glTF buffer view {i}: its compressed data is in a missing buffer"
                );
                if vb != 0 {
                    required.push(name.clone());
                }
            }
        }
        // Views on a missing fallback need the extension to be read at all.
        for name in required {
            container::require_extension(&mut json, &name);
        }
        let has_views = !list(&json, "bufferViews").is_empty();
        if layout.is_empty() && placeholders.is_empty() && !has_views {
            json.remove("buffers");
            return Ok((json, layout, None));
        }
        let mut buffers = vec![json!({ "byteLength": bin_len })];
        buffers.extend(placeholders);
        json.insert("buffers".into(), Value::Array(buffers));
        Ok((json, layout, Some(bin_len)))
    }

    /// The document for a triangle mesh: one node drawing one mesh, a primitive per material,
    /// its arrays moved into the document's buffer without a copy.
    #[must_use]
    pub fn from_mesh(mesh: Mesh, facts: Map<String, Value>) -> Document<'static> {
        let Mesh {
            prims,
            materials,
            images,
        } = mesh;
        let images = images.into_iter().map(Source::Image).collect();
        Self::from_named(vec![(None, prims)], &materials, images, None, facts)
    }

    /// The document for named triangle meshes sharing `materials` and `images`: a node drawing
    /// each, in order, skipping those with no triangles; `source` becomes the asset's
    /// `extras.source`.
    pub(crate) fn from_named(
        meshes: Vec<(Option<String>, Vec<Prim>)>,
        materials: &[Material],
        images: Vec<Source>,
        source: Option<Value>,
        facts: Map<String, Value>,
    ) -> Document<'static> {
        let mut b = Builder::default();
        let (mut nodes, mut out) = (Vec::new(), Vec::new());
        for (name, prims) in meshes {
            let Some(mut mesh) = b.mesh(prims, materials.len()) else {
                continue;
            };
            let mut node = json!({ "mesh": out.len() });
            if let Some(name) = name {
                mesh["name"] = json!(name);
                node["name"] = json!(name);
            }
            out.push(mesh);
            nodes.push(node);
        }
        if out.is_empty() {
            return Document::empty(facts);
        }
        let mut json = Map::new();
        if let Some(source) = source {
            json.insert("asset".into(), json!({ "extras": { "source": source } }));
        }
        json.insert("scene".into(), json!(0));
        json.insert(
            "scenes".into(),
            json!([{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }]),
        );
        json.insert("nodes".into(), Value::Array(nodes));
        json.insert("meshes".into(), Value::Array(out));
        b.materials(materials, images, &mut json);
        b.finish(json, facts)
    }
}

/// Builds a document's one buffer from typed arrays, with a view and an accessor for each.
#[derive(Default)]
pub(crate) struct Builder {
    pub parts: Parts,
    pub views: Vec<Value>,
    pub accessors: Vec<Value>,
    /// The images named but not read, by image index.
    pub refs: Refs,
}

impl Builder {
    /// A primitive's JSON (attributes and indices, no material), its arrays moved into the
    /// buffer; `None` when it has no triangles.
    pub(crate) fn primitive(&mut self, p: Prim) -> Option<Value> {
        if p.indices.is_empty() {
            return None;
        }
        let n = p.positions.len();
        let pos = self.positions(p.positions);
        let mut attributes = json!({ "POSITION": pos });
        // Without normals, glTF clients shade flat.
        if p.normals.len() == n {
            let a = self.accessor(Part::f32(p.normals.into_flattened()), 5126, n, "VEC3", Some(34962));
            attributes["NORMAL"] = json!(a);
        }
        let count = p.indices.len();
        let indices = self.accessor(Part::u32(p.indices), 5125, count, "SCALAR", Some(34963));
        if p.uvs.len() == n {
            let a = self.accessor(Part::f32(p.uvs.into_flattened()), 5126, n, "VEC2", Some(34962));
            attributes["TEXCOORD_0"] = json!(a);
        }
        if p.colors.len() == n {
            let mut colors = p.colors.into_flattened();
            for c in &mut colors {
                *c = c.clamp(0.0, 1.0);
            }
            let a = self.accessor(Part::f32(colors), 5126, n, "VEC4", Some(34962));
            attributes["COLOR_0"] = json!(a);
        }
        Some(json!({ "attributes": attributes, "indices": indices }))
    }

    /// A mesh of `prims`, each using its material (the last of `materials` when out of range);
    /// `None` when none has triangles.
    pub(crate) fn mesh(&mut self, prims: Vec<Prim>, materials: usize) -> Option<Value> {
        let last = materials.saturating_sub(1);
        let primitives: Vec<Value> = prims
            .into_iter()
            .filter_map(|p| {
                let material = p.material.min(last);
                let mut prim = self.primitive(p)?;
                prim["material"] = json!(material);
                Some(prim)
            })
            .collect();
        (!primitives.is_empty()).then(|| json!({ "primitives": primitives }))
    }

    /// A `VEC3` float accessor with the `min` and `max` glTF requires of positions (and of
    /// morph targets' position deltas).
    pub(crate) fn positions(&mut self, v: Vec<[f32; 3]>) -> usize {
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &v {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        if v.is_empty() {
            (min, max) = ([0.0; 3], [0.0; 3]);
        }
        let n = v.len();
        let a = self.accessor(Part::f32(v.into_flattened()), 5126, n, "VEC3", Some(34962));
        self.accessors[a]["min"] = json!(min);
        self.accessors[a]["max"] = json!(max);
        a
    }

    /// A float accessor of `kind` (`SCALAR`, `VEC3`, `VEC4`, `MAT4`) with no target, for skins
    /// and animations; `bounds` adds the `min` and `max` an animation's times need.
    #[cfg(any(feature = "fbx", feature = "usd", feature = "vox", feature = "3ds"))]
    pub(crate) fn floats(&mut self, v: Vec<f32>, kind: &str, bounds: bool) -> usize {
        let width = match kind {
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            _ => 1,
        };
        let count = v.len() / width;
        let range = bounds.then(|| {
            let mut min = vec![f32::MAX; width];
            let mut max = vec![f32::MIN; width];
            for e in v.chunks_exact(width) {
                for k in 0..width {
                    min[k] = min[k].min(e[k]);
                    max[k] = max[k].max(e[k]);
                }
            }
            (min, max)
        });
        let a = self.accessor(Part::f32(v), 5126, count, kind, None);
        if let Some((min, max)) = range {
            self.accessors[a]["min"] = json!(min);
            self.accessors[a]["max"] = json!(max);
        }
        a
    }

    /// Adds `materials` and `images` to `json`, with the textures and samplers they use (texture `i` is image
    /// `i`, repeating, and more for other wraps) and the extensions they use added to `extensionsUsed`. Nothing is
    /// added for an empty list.
    pub(crate) fn materials<M: Into<Full>>(
        &mut self,
        materials: impl IntoIterator<Item = M>,
        images: Vec<Source>,
        json: &mut Map<String, Value>,
    ) {
        let mut w = Writer::new(images.len());
        let materials: Vec<Value> = materials.into_iter().map(|m| w.material(&m.into())).collect();
        let images: Vec<Value> = images
            .into_iter()
            .enumerate()
            .map(|(i, s)| match s {
                Source::Image(Image { bytes, mime }) => {
                    let v = self.view(Part::bytes(bytes), None);
                    json!({ "bufferView": v, "mimeType": mime })
                }
                Source::Ref(r, uri) => {
                    self.refs.insert(i, r);
                    json!({ "uri": uri })
                }
            })
            .collect();
        let (textures, samplers, used) = w.finish();
        if !materials.is_empty() {
            json.insert("materials".into(), json!(materials));
        }
        if !used.is_empty() {
            let list = json.entry("extensionsUsed").or_insert_with(|| json!([]));
            if let Some(list) = list.as_array_mut() {
                for e in used {
                    if !list.iter().any(|v| v == e) {
                        list.push(json!(e));
                    }
                }
            }
        }
        if !images.is_empty() {
            json.insert("samplers".into(), json!(samplers));
            json.insert("textures".into(), json!(textures));
            json.insert("images".into(), json!(images));
        }
    }

    /// The document: `json` (scenes, nodes, meshes and the rest) with the asset, the accessors,
    /// the views and the one buffer they're on.
    pub(crate) fn finish(self, mut json: Map<String, Value>, facts: Map<String, Value>) -> Document<'static> {
        let asset = json.entry("asset").or_insert_with(|| json!({}));
        asset["version"] = json!("2.0");
        if asset.get("generator").is_none() {
            asset["generator"] = json!("polysmelt");
        }
        let mut buffers = Vec::new();
        if !self.views.is_empty() {
            json.insert("accessors".into(), Value::Array(self.accessors));
            json.insert("bufferViews".into(), Value::Array(self.views));
            json.insert("buffers".into(), json!([{ "byteLength": self.parts.len }]));
            buffers.push(Blob::Parts(self.parts));
        }
        Document {
            json,
            buffers,
            facts,
            refs: self.refs,
        }
    }

    /// A buffer view of `part`; returns its index.
    pub(crate) fn view(&mut self, part: Part, target: Option<u32>) -> usize {
        let (at, len) = self.parts.push(part);
        let mut v = json!({ "buffer": 0, "byteOffset": at, "byteLength": len });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }

    /// An accessor of `count` elements of `kind` over a view of `part`; returns its index.
    pub(crate) fn accessor(&mut self, part: Part, ctype: u32, count: usize, kind: &str, target: Option<u32>) -> usize {
        let v = self.view(part, target);
        self.accessors
            .push(json!({ "bufferView": v, "componentType": ctype, "count": count, "type": kind }));
        self.accessors.len() - 1
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use std::io::Write as _;

    use super::{AHEAD, FileSpan};

    #[test]
    fn a_file_span_reads_what_the_file_holds_however_it_is_read() {
        let bytes: Vec<u8> = (0..3 * AHEAD + 77)
            .map(|i| (i * 7 + i / 251).to_le_bytes()[0])
            .collect();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"head").unwrap();
        file.write_all(&bytes).unwrap();
        let span = FileSpan::new(file, 4, bytes.len() as u64, u64::MAX);
        let len = bytes.len();
        // Forward in small steps, backward, across the read-ahead's end, big reads, the very end.
        let mut reads: Vec<(usize, usize)> = (0..200).map(|k| (k * 997 % len, 1 + k % 300)).collect();
        reads.extend([
            (AHEAD - 3, 10),
            (5, AHEAD),
            (0, len),
            (len - 1, 1),
            (len - 40, 40),
            (100, 0),
            (AHEAD, 2),
        ]);
        reads.extend((0..50).rev().map(|k| (k * 3989, 64)));
        for (off, n) in reads {
            let n = n.min(len - off);
            let mut out = vec![0; n];
            span.read_at(off as u64, &mut out).unwrap();
            assert_eq!(out, bytes[off..off + n], "{n} bytes at {off}");
        }
    }
}
