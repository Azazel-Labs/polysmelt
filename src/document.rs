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

use crate::Bytes;
use crate::gltf as container;
use crate::images::{Refs, Source};
use crate::material::{Full, Writer};
use crate::mesh::{Bone, Image, MAX_POINTS, MAX_TRIANGLES, Material, Mesh, Prim, TRIANGLE_CEILING};
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
    /// `vertices`, `voxels`, `voxel_size`, `models`, `objects`, `colors`, `application`, and for
    /// motion capture `joints`, `channels`, `frames`, `frame_time` and `dropped_frames`.
    pub facts: Map<String, Value>,
    /// The images named but not read, by image index (see [`Document::images`]).
    pub(crate) refs: Refs,
    /// The most triangles [`Document::flatten`] draws (see [`Document::set_max_triangles`]).
    pub(crate) max_triangles: usize,
    /// Whether [`Document::flatten`] draws points and lines (see
    /// [`Document::set_lines_and_points`]).
    pub(crate) lines_and_points: bool,
    /// The most points and line segments [`Document::flatten`] draws (see
    /// [`Document::set_max_points`]).
    pub(crate) max_points: usize,
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
    /// As `Rest`, but skinned meshes as their files store them: in the pose they were bound in,
    /// their vertices not moved by their joints, and the skeleton where the inverse bind matrices
    /// put it. Joints and weights still come with them, to pose with your own skinning. Quicker
    /// than `Rest` on skinned meshes, and the same wherever the nodes stand as they were bound.
    Bind,
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
    /// In memory, borrowed from the input (a GLB's binary chunk, an asset).
    #[cfg_attr(not(feature = "gltf"), allow(dead_code, reason = "only the glTF reader makes one"))]
    Borrowed(&'a [u8]),
    /// In memory, owned and shareable: images slice it rather than copy it.
    Bytes(Bytes),
    /// A stretch of a file, read as needed.
    #[cfg(feature = "fs")]
    #[cfg_attr(not(feature = "gltf"), allow(dead_code, reason = "only the glTF reader makes one"))]
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

/// A [`FileSpan`]'s file and the blocks of it last read.
#[cfg(feature = "fs")]
struct Cursor {
    file: File,
    /// Where the file's cursor is relative to the span's base, to skip needless seeks.
    at: u64,
    /// Up to [`BLOCKS`] blocks of [`BLOCK`] bytes, each by its start (a multiple of `BLOCK` from
    /// the span's base), the most recently used last. Small reads (animation keys, a skin's
    /// matrices, little meshes' accessors) come from here rather than each its own system call,
    /// and since the blocks don't overlap, reads that hop between a few places in the file (a
    /// mesh's vertices and its indices) read each part of it once.
    blocks: Vec<(u64, Vec<u8>)>,
}

/// The size of the blocks small reads go through; reads this big or bigger go to the file.
#[cfg(feature = "fs")]
const BLOCK: usize = 64 * 1024;

/// How many blocks a [`FileSpan`] keeps.
#[cfg(feature = "fs")]
const BLOCKS: usize = 8;

#[cfg(feature = "fs")]
impl FileSpan {
    /// `len` bytes of `file` from `base`; its cursor is at `base + at`.
    #[cfg_attr(not(feature = "gltf"), allow(dead_code, reason = "only the glTF reader makes one"))]
    pub(crate) fn new(file: File, base: u64, len: u64, at: u64) -> Self {
        Self {
            file: Mutex::new(Cursor {
                file,
                at,
                blocks: Vec::new(),
            }),
            base,
            len,
        }
    }

    /// Copies the bytes from `off` into `out`, which must lie within the span.
    /// Writes bytes `from..to`, read through `chunk` a MiB at a time.
    fn write_range(&self, from: u64, to: u64, chunk: &mut Vec<u8>, w: &mut dyn Write) -> Result<()> {
        let mut at = from;
        while at < to {
            let n = usize_of((to - at).min(1 << 20));
            chunk.resize(n, 0);
            self.read_at(at, chunk)?;
            w.write_all(chunk)?;
            at += n as u64;
        }
        Ok(())
    }

    fn read_at(&self, off: u64, out: &mut [u8]) -> Result<()> {
        ensure!(
            off.checked_add(out.len() as u64).is_some_and(|end| end <= self.len),
            "read past the end of a file's glTF data"
        );
        let mut c = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if out.len() >= BLOCK {
            return self.read_file(&mut c, off, out);
        }
        let mut done = 0;
        while done < out.len() {
            let at = off + done as u64;
            let start = at - at % BLOCK as u64;
            let block = if let Some(i) = c.blocks.iter().position(|(s, _)| *s == start) {
                c.blocks.remove(i)
            } else {
                // Reuse the least recently used block's buffer once all are in use.
                let mut b = if c.blocks.len() >= BLOCKS {
                    c.blocks.remove(0).1
                } else {
                    Vec::new()
                };
                b.resize(usize_of((self.len - start).min(BLOCK as u64)), 0);
                self.read_file(&mut c, start, &mut b)?;
                (start, b)
            };
            let from = usize_of(at - start);
            let n = (out.len() - done).min(block.1.len() - from);
            out[done..done + n].copy_from_slice(&block.1[from..from + n]);
            done += n;
            c.blocks.push(block);
        }
        Ok(())
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
    Bytes(Bytes),
}

#[allow(dead_code, reason = "readers use the part types their formats need")]
impl Part {
    /// Floats, stored as they are on a little-endian target (and byte-swapped once otherwise).
    pub(crate) fn f32(v: Vec<f32>) -> Self {
        if cfg!(target_endian = "little") {
            Part::F32(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>().into())
        }
    }

    pub(crate) fn u32(v: Vec<u32>) -> Self {
        if cfg!(target_endian = "little") {
            Part::U32(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>().into())
        }
    }

    pub(crate) fn u16(v: Vec<u16>) -> Self {
        if cfg!(target_endian = "little") {
            Part::U16(v)
        } else {
            Part::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>().into())
        }
    }

    pub(crate) fn bytes(v: impl Into<Bytes>) -> Self {
        Part::Bytes(v.into())
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

    /// `len` bytes from `off`, shared without a copy, when they lie in one byte part.
    fn shared(&self, off: u64, len: u64) -> Option<Bytes> {
        let (at, Part::Bytes(bytes)) = &self.list[self.part_at(off)?] else {
            return None;
        };
        let start = usize_of(off - at);
        let end = start.checked_add(usize_of(len)).filter(|&e| e <= bytes.len())?;
        Some(bytes.slice(start..end))
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
            Blob::Borrowed(b) => b.len() as u64,
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
            Blob::Borrowed(b) => b.get(usize_of(off)..usize_of(off.checked_add(len)?)),
            Blob::Bytes(b) => b.get(usize_of(off)..usize_of(off.checked_add(len)?)),
            Blob::Parts(p) => p.slice(off, len),
            #[cfg(feature = "fs")]
            Blob::File(_) => None,
            Blob::Missing(_) => None,
        }
    }

    /// `len` bytes from `off`: shared without a copy when the buffer owns them in one piece,
    /// copied otherwise.
    pub(crate) fn shared(&self, off: u64, len: u64) -> Result<Bytes> {
        let end = off
            .checked_add(len)
            .filter(|&e| e <= self.len())
            .context("glTF data runs past the end of its buffer")?;
        let shared = match self {
            Blob::Bytes(b) => Some(b.slice(usize_of(off)..usize_of(end))),
            Blob::Parts(p) => p.shared(off, len),
            _ => None,
        };
        if let Some(b) = shared {
            return Ok(b);
        }
        if let Some(s) = self.slice(off, len) {
            return Ok(Bytes::copy_from_slice(s));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(usize_of(len))
            .map_err(|_| invalid!("glTF data of {len} bytes"))?;
        out.resize(usize_of(len), 0);
        self.read_at(off, &mut out)?;
        Ok(out.into())
    }

    /// Copies the bytes from `off` into `out`.
    pub(crate) fn read_at(&self, off: u64, out: &mut [u8]) -> Result<()> {
        let end = off
            .checked_add(out.len() as u64)
            .filter(|&e| e <= self.len())
            .context("glTF data runs past the end of its buffer")?;
        match self {
            Blob::Borrowed(b) => out.copy_from_slice(&b[usize_of(off)..usize_of(end)]),
            Blob::Bytes(b) => out.copy_from_slice(&b[usize_of(off)..usize_of(end)]),
            #[cfg(feature = "fs")]
            Blob::File(f) => f.read_at(off, out)?,
            Blob::Parts(p) => p.read_at(off, out),
            Blob::Missing(_) => bail!("glTF data is in a meshopt fallback buffer, which holds none"),
        }
        Ok(())
    }

    /// Writes the first `len` bytes.
    /// [`Blob::write_to`], with `kept` (sorted by offset) holding bytes already read from
    /// this blob's file, to write instead of reading them again.
    fn write_keeping(&self, len: u64, w: &mut dyn Write, kept: &[crate::flatten::Kept]) -> Result<()> {
        #[cfg(feature = "fs")]
        if let (Blob::File(f), false) = (self, kept.is_empty()) {
            ensure!(len <= f.len, "glTF buffer holds {} bytes but declares {len}", f.len);
            let mut chunk = Vec::new();
            let mut pos = 0;
            for (_, start, bytes) in kept {
                let end = (start + bytes.len() as u64).min(len);
                if end <= pos {
                    continue;
                }
                let from = (*start).max(pos);
                f.write_range(pos, from, &mut chunk, w)?;
                w.write_all(&bytes[usize_of(from - start)..usize_of(end - start)])?;
                pos = end;
            }
            return f.write_range(pos, len, &mut chunk, w);
        }
        let _ = kept;
        self.write_to(len, w)
    }

    fn write_to(&self, len: u64, w: &mut dyn Write) -> Result<()> {
        ensure!(
            len <= self.len(),
            "glTF buffer holds {} bytes but declares {len}",
            self.len()
        );
        match self {
            Blob::Borrowed(b) => w.write_all(&b[..usize_of(len)])?,
            Blob::Bytes(b) => w.write_all(&b[..usize_of(len)])?,
            #[cfg(feature = "fs")]
            Blob::File(f) => f.write_range(0, len, &mut Vec::new(), w)?,
            Blob::Parts(p) => p.write_to(len, w)?,
            Blob::Missing(_) => bail!("a meshopt fallback buffer has no bytes to write"),
        }
        Ok(())
    }

    fn into_owned(self) -> Blob<'static> {
        match self {
            Blob::Borrowed(b) => Blob::Bytes(Bytes::copy_from_slice(b)),
            Blob::Bytes(b) => Blob::Bytes(b),
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
            max_triangles: MAX_TRIANGLES,
            lines_and_points: true,
            max_points: MAX_POINTS,
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
            max_triangles: self.max_triangles,
            lines_and_points: self.lines_and_points,
            max_points: self.max_points,
        }
    }

    /// The most triangles [`Document::flatten`] draws before it refuses: the load's
    /// [`Keep::max_triangles`], or [`MAX_TRIANGLES`] for a document built otherwise.
    #[must_use]
    pub fn max_triangles(&self) -> usize {
        self.max_triangles
    }

    /// Sets the most triangles [`Document::flatten`] draws before it refuses, as
    /// [`Keep::max_triangles`] does for a load: values past [`TRIANGLE_CEILING`] are clamped.
    pub fn set_max_triangles(&mut self, n: usize) {
        self.max_triangles = n.min(TRIANGLE_CEILING);
    }

    /// Whether [`Document::flatten`] draws points and lines: the load's
    /// [`Keep::lines_and_points`], or `true` for a document built otherwise.
    #[must_use]
    pub fn lines_and_points(&self) -> bool {
        self.lines_and_points
    }

    /// Sets whether [`Document::flatten`] draws points and lines, as [`Keep::lines_and_points`]
    /// does for a load.
    pub fn set_lines_and_points(&mut self, on: bool) {
        self.lines_and_points = on;
    }

    /// The most points and line segments [`Document::flatten`] draws before it refuses: the
    /// load's [`Keep::max_points`], or [`MAX_POINTS`] for a document built otherwise.
    #[must_use]
    pub fn max_points(&self) -> usize {
        self.max_points
    }

    /// Sets the most points and line segments [`Document::flatten`] draws before it refuses, as
    /// [`Keep::max_points`] does for a load: values past [`TRIANGLE_CEILING`] are clamped.
    pub fn set_max_points(&mut self, n: usize) {
        self.max_points = n.min(TRIANGLE_CEILING);
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

    /// Buffer view `i`'s bytes, shared without a copy when its buffer owns them in one piece.
    pub(crate) fn view_shared(&self, i: usize) -> Result<Bytes> {
        let v = list(&self.json, "bufferViews")
            .get(i)
            .with_context(|| format!("glTF bufferViews[{i}] doesn't exist"))?;
        let num = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
        let b = self
            .buffers
            .get(usize_of(num("buffer")))
            .context("glTF buffer view names a missing buffer")?;
        let (offset, len) = (num("byteOffset"), num("byteLength"));
        ensure!(
            offset.saturating_add(len) <= b.len(),
            "glTF buffer view runs past its buffer"
        );
        b.shared(offset, len)
    }

    /// Adds a buffer holding `bytes` (a `Vec<u8>` or [`Bytes`], neither copied); returns its
    /// index.
    pub fn push_buffer(&mut self, bytes: impl Into<Bytes>) -> usize {
        self.push_blob(Blob::Bytes(bytes.into()))
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
        let mut json = self.json.clone();
        // What repair read from files is written from memory rather than read again.
        let (extra, mut kept) = crate::flatten::repaired_keeping(self, &mut json);
        let extra = extra.map(|b| Blob::Bytes(b.into()));
        kept.sort_unstable_by_key(|k| (k.0, k.1));
        let (json, layout, bin_len) = self.glb_json(json, extra.as_ref())?;
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
                let blob = self
                    .buffers
                    .get(i)
                    .or(extra.as_ref())
                    .context("glTF buffer is missing")?;
                let from = kept.partition_point(|k| k.0 < i);
                let to = kept.partition_point(|k| k.0 <= i);
                blob.write_keeping(len, w, &kept[from..to])?;
                pos = at + len;
            }
            write_zeros(w, padded - pos)?;
        }
        Ok(())
    }

    /// The GLB's JSON (from `json`, the document's, with `extra` its last buffer if given),
    /// where each buffer with bytes goes in the binary chunk (buffer, offset, length), and the
    /// binary chunk's length (`None` for none).
    #[allow(clippy::type_complexity, reason = "a private step of write_glb")]
    fn glb_json(
        &self,
        mut json: Map<String, Value>,
        extra: Option<&Blob<'_>>,
    ) -> Result<(Map<String, Value>, Vec<(usize, u64, u64)>, Option<u64>)> {
        let buffers = list(&json, "buffers");
        let held = self.buffers.len() + usize::from(extra.is_some());
        ensure!(
            buffers.len() == held,
            "glTF document lists {} buffers but holds {held}",
            buffers.len(),
        );
        // Each buffer's new home: (buffer, offset). Buffers with bytes share the binary chunk;
        // missing fallbacks follow as buffers without data.
        let mut home = Vec::with_capacity(buffers.len());
        let mut layout = Vec::new();
        let mut placeholders = Vec::new();
        let mut bin_len = 0u64;
        for (i, (b, blob)) in buffers.iter().zip(self.buffers.iter().chain(extra)).enumerate() {
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

    /// The document for a mesh: one node drawing one mesh, a primitive per material (and one for
    /// the lines and one for the points of each, sharing its vertices),
    /// its arrays moved into the document's buffer without a copy. Skinned primitives, when the
    /// mesh has a skeleton, are a second mesh on a node with a skin: a node per bone, and
    /// inverse bind matrices that bind the vertices where they are, so the document draws as the
    /// mesh does and moves with its bones.
    #[must_use]
    pub fn from_mesh(mesh: Mesh, facts: Map<String, Value>) -> Document<'static> {
        let Mesh {
            prims,
            materials,
            images,
            skeleton,
        } = mesh;
        let images = images.into_iter().map(Source::Image).collect();
        let (skinned, rigid): (Vec<Prim>, Vec<Prim>) = if skeleton.is_empty() {
            (Vec::new(), prims)
        } else {
            prims.into_iter().partition(Prim::skinned)
        };
        let mut meshes = vec![(None, rigid)];
        if !skinned.is_empty() {
            meshes.push((None, skinned));
        }
        Self::from_named(meshes, &materials, images, None, facts, &skeleton)
    }

    /// The document for named meshes sharing `materials` and `images`: a node drawing each, in
    /// order, skipping those that draw nothing; `source` becomes the asset's
    /// `extras.source`. A mesh whose primitives are all skinned is bound to `skeleton`, when
    /// there is one.
    pub(crate) fn from_named(
        mut meshes: Vec<(Option<String>, Vec<Prim>)>,
        materials: &[Material],
        images: Vec<Source>,
        source: Option<Value>,
        facts: Map<String, Value>,
        skeleton: &[Bone],
    ) -> Document<'static> {
        let materials = texcoords(&mut meshes, materials);
        let mut b = Builder::default();
        let rig = (!skeleton.is_empty() && meshes.iter().any(|(_, p)| !p.is_empty() && p.iter().all(Prim::skinned)))
            .then(|| Rigging::new(skeleton, meshes.iter().flat_map(|(_, p)| p)));
        let (mut nodes, mut out) = (Vec::new(), Vec::new());
        let mut skinned_node = None;
        for (name, prims) in meshes {
            let skinned = !prims.is_empty() && prims.iter().all(Prim::skinned);
            b.joints = rig.as_ref().filter(|_| skinned).map(|r| (r.offset, r.joints.len()));
            let Some(mut mesh) = b.mesh(prims, materials.len()) else {
                continue;
            };
            let mut node = json!({ "mesh": out.len() });
            if let Some(name) = name {
                mesh["name"] = json!(name);
                node["name"] = json!(name);
            }
            if b.joints.is_some() {
                node["skin"] = json!(0);
                skinned_node = Some(nodes.len());
            }
            out.push(mesh);
            nodes.push(node);
        }
        b.joints = None;
        if out.is_empty() {
            return Document::empty(facts);
        }
        let mut json = Map::new();
        if let Some(source) = source {
            json.insert("asset".into(), json!({ "extras": { "source": source } }));
        }
        let mut scene: Vec<usize> = (0..nodes.len()).collect();
        if let (Some(rig), Some(_)) = (&rig, skinned_node) {
            scene.extend(rig.write(&mut b, &mut nodes, &mut json));
        }
        json.insert("scene".into(), json!(0));
        json.insert("scenes".into(), json!([{ "nodes": scene }]));
        json.insert("nodes".into(), Value::Array(nodes));
        json.insert("meshes".into(), Value::Array(out));
        b.materials(&materials, images, &mut json);
        b.finish(json, facts)
    }
}

/// The glTF materials for `materials`, their textures reading the coordinate sets as each primitive using them
/// numbers its sets (the color texture's first, then the others that are there, in order). A material with a
/// texture on another set ([`Sampling::uv_set`](crate::mesh::Sampling::uv_set)) gets a copy for each numbering
/// the primitives give it, and the primitives are pointed at theirs.
fn texcoords(meshes: &mut [(Option<String>, Vec<Prim>)], materials: &[Material]) -> Vec<Full> {
    let mut out: Vec<Full> = materials.iter().map(Full::from).collect();
    let last = materials.len().saturating_sub(1);
    // The other sets each material's textures read, in order.
    let sets: Vec<Vec<usize>> = materials
        .iter()
        .map(|m| {
            let used = std::cell::RefCell::new(Vec::new());
            Full::of(m, |s| {
                used.borrow_mut().extend(s);
                Some(0)
            });
            let mut used = used.into_inner();
            used.sort_unstable();
            used.dedup();
            used
        })
        .collect();
    // (material, the numbers its sets get, the material written for them).
    let mut copies: Vec<(usize, Vec<Option<u32>>, usize)> = Vec::new();
    for p in meshes.iter_mut().flat_map(|(_, p)| p) {
        let i = p.material.min(last);
        let Some(want) = sets.get(i).filter(|s| !s.is_empty()) else {
            continue;
        };
        let n = p.positions.len();
        let first = u32::from(p.uvs.len() == n);
        let numbers: Vec<Option<u32>> = want
            .iter()
            .map(|&s| {
                p.uv_sets.get(s).filter(|v| v.len() == n)?;
                let before = p.uv_sets[..s].iter().filter(|v| v.len() == n).count();
                Some(first + u32::try_from(before).ok()?)
            })
            .collect();
        if let Some(c) = copies.iter().find(|c| c.0 == i && c.1 == numbers) {
            p.material = c.2;
            continue;
        }
        let full = Full::of(&materials[i], |s| match s {
            None => Some(0),
            Some(s) => want.iter().position(|&w| w == s).and_then(|k| numbers[k]),
        });
        let slot = if copies.iter().any(|c| c.0 == i) {
            out.push(full);
            out.len() - 1
        } else {
            out[i] = full;
            i
        };
        copies.push((i, numbers, slot));
        p.material = slot;
    }
    out
}

/// A skeleton as a skin's joints: the bones, after a root above them all when they have
/// several roots or some vertex is bound to none (glTF wants a skin's joints to share a root,
/// and every skinned vertex weighted).
struct Rigging<'s> {
    skeleton: &'s [Bone],
    /// 1 when a root comes first, 0 otherwise: what's added to a bone's index for its joint.
    offset: u32,
    /// Each joint's world transform, as written: decomposed into TRS and composed back, so the
    /// inverse bind matrices match what a reader composes, whatever shear was dropped.
    joints: Vec<crate::flatten::M4>,
    /// Each joint's local TRS.
    locals: Vec<([f64; 3], [f64; 4], [f64; 3])>,
}

impl<'s> Rigging<'s> {
    fn new<'p>(skeleton: &'s [Bone], prims: impl Iterator<Item = &'p Prim>) -> Self {
        use crate::flatten::{IDENTITY, affine_inverse, decompose, mul, trs};
        let roots = skeleton.iter().filter(|b| b.parent.is_none()).count();
        let unbound = || {
            prims.filter(|p| p.skinned()).any(|p| {
                p.skin_weights
                    .iter()
                    .any(|w| w.iter().sum::<f32>().partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater))
            })
        };
        let offset = u32::from(roots > 1 || unbound());
        let world = |b: &Bone| b.matrix.map(f64::from);
        let mut joints = Vec::with_capacity(skeleton.len() + 1);
        let mut locals = Vec::with_capacity(skeleton.len() + 1);
        if offset == 1 {
            joints.push(IDENTITY);
            locals.push(([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]));
        }
        for (i, b) in skeleton.iter().enumerate() {
            // A bone's parent comes before it; one that doesn't is taken as a root.
            let parent = b.parent.filter(|&p| p < i);
            let local = match parent {
                Some(p) => affine_inverse(&world(&skeleton[p])).map_or_else(|| world(b), |inv| mul(&inv, &world(b))),
                None => world(b),
            };
            let (t, q, s) = decompose(&local);
            let above = match parent {
                Some(p) => joints[p + offset as usize],
                None => IDENTITY,
            };
            joints.push(mul(&above, &trs(t, q, s)));
            locals.push((t, q, s));
        }
        Self {
            skeleton,
            offset,
            joints,
            locals,
        }
    }

    /// Adds a node per joint after `nodes`, the skin, and its inverse bind matrices; gives the
    /// root joints' nodes, for the scene.
    fn write(&self, b: &mut Builder, nodes: &mut Vec<Value>, json: &mut Map<String, Value>) -> Vec<usize> {
        use crate::flatten::{IDENTITY, affine_inverse};
        let first = nodes.len();
        let offset = self.offset as usize;
        let mut roots = Vec::new();
        let mut children = vec![Vec::new(); self.joints.len()];
        for (i, bone) in self.skeleton.iter().enumerate() {
            match bone.parent.filter(|&p| p < i) {
                Some(p) => children[p + offset].push(first + i + offset),
                None if offset == 1 => children[0].push(first + i + offset),
                None => roots.push(first + i),
            }
        }
        if offset == 1 {
            roots.push(first);
        }
        for (j, ((t, q, s), kids)) in self.locals.iter().zip(&children).enumerate() {
            let mut node = Map::new();
            let name = match j.checked_sub(offset) {
                Some(i) => self.skeleton[i].name.clone(),
                None => Some("Skeleton".to_owned()),
            };
            if let Some(name) = name {
                node.insert("name".into(), json!(name));
            }
            if !kids.is_empty() {
                node.insert("children".into(), json!(kids));
            }
            let f = |v: &[f64]| v.iter().map(|&x| crate::num::f32_of(x)).collect::<Vec<f32>>();
            if t.iter().any(|&x| x != 0.0) {
                node.insert("translation".into(), json!(f(t)));
            }
            if q[..3].iter().any(|&x| x != 0.0) {
                node.insert("rotation".into(), json!(f(q)));
            }
            // Written as f32, a scale that rounds to 1 is the default.
            if f(s) != [1.0; 3] {
                node.insert("scale".into(), json!(f(s)));
            }
            nodes.push(Value::Object(node));
        }
        let ibm: Vec<f32> = self
            .joints
            .iter()
            .flat_map(|m| affine_inverse(m).unwrap_or(IDENTITY).map(crate::num::f32_of))
            .collect();
        let n = self.joints.len();
        let ibm = b.accessor(Part::f32(ibm), 5126, n, "MAT4", None);
        json.insert(
            "skins".into(),
            json!([{ "joints": (first..first + n).collect::<Vec<_>>(), "inverseBindMatrices": ibm }]),
        );
        roots
    }
}

/// A skinned primitive's `JOINTS_0` and `WEIGHTS_0`, flattened: each bone index moved up by
/// `offset` (past the root a skin may have first), weights summing to 1 in single precision,
/// and a vertex no bone moves bound to the root, joint 0. A slot with no weight names joint 0,
/// as glTF's validator wants.
fn skin_attributes(joints: &[[u32; 4]], weights: &[[f32; 4]], offset: u32, count: usize) -> (Vec<u32>, Vec<f32>) {
    let mut js = Vec::with_capacity(joints.len() * 4);
    let mut ws = Vec::with_capacity(weights.len() * 4);
    let last = u32::try_from(count.saturating_sub(1)).unwrap_or(u32::MAX);
    for (j, w) in joints.iter().zip(weights) {
        let mut w = w.map(|x| if x > 0.0 && x.is_finite() { x } else { 0.0 });
        let mut j = j.map(|x| x.saturating_add(offset).min(last));
        let sum: f32 = w.iter().sum();
        if sum > 0.0 {
            w = w.map(|x| x / sum);
            // What rounding left over goes to the strongest.
            let top = (0..4).fold(0, |a, k| if w[k] > w[a] { k } else { a });
            w[top] += 1.0 - w.iter().sum::<f32>();
        } else {
            (j, w) = ([0; 4], [1.0, 0.0, 0.0, 0.0]);
        }
        for k in 0..4 {
            if w[k] == 0.0 {
                j[k] = 0;
            }
        }
        js.extend_from_slice(&j);
        ws.extend_from_slice(&w);
    }
    (js, ws)
}

/// Builds a document's one buffer from typed arrays, with a view and an accessor for each.
#[derive(Default)]
pub(crate) struct Builder {
    pub parts: Parts,
    pub views: Vec<Value>,
    pub accessors: Vec<Value>,
    /// The images named but not read, by image index.
    pub refs: Refs,
    /// While writing skinned primitives: what's added to a bone's index for its joint, and how
    /// many joints the skin has.
    pub joints: Option<(u32, usize)>,
}

impl Builder {
    /// A primitive's JSON (attributes and indices, no material) for its triangles, its arrays
    /// moved into the buffer; `None` when it has no triangles. Its lines and points are left
    /// out: [`Builder::primitives`] writes them too.
    #[cfg_attr(
        not(any(feature = "fbx", feature = "obj", feature = "ply", feature = "usd")),
        allow(dead_code, reason = "only readers that build primitives one at a time use it")
    )]
    pub(crate) fn primitive(&mut self, mut p: Prim) -> Option<Value> {
        if p.indices.is_empty() {
            return None;
        }
        p.lines = Vec::new();
        p.points = Vec::new();
        self.primitives(p).pop()
    }

    /// The JSON of a primitive's triangles, lines (mode 1) and points (mode 0), those it has, in
    /// that order, on one set of vertex attributes they share, and no material: its arrays moved
    /// into the buffer. Empty when it draws nothing.
    pub(crate) fn primitives(&mut self, mut p: Prim) -> Vec<Value> {
        if p.is_empty() {
            return Vec::new();
        }
        let n = p.positions.len();
        let draws: [(u64, Vec<u32>); 3] = [
            (4, std::mem::take(&mut p.indices)),
            (1, std::mem::take(&mut p.lines)),
            (0, std::mem::take(&mut p.points)),
        ];
        let pos = self.positions(p.positions);
        let mut attributes = json!({ "POSITION": pos });
        // Without normals, glTF clients shade flat.
        let has_normals = p.normals.len() == n;
        if has_normals {
            let a = self.accessor(Part::f32(p.normals.into_flattened()), 5126, n, "VEC3", Some(34962));
            attributes["NORMAL"] = json!(a);
        }
        // The index accessors next, as the buffer has always had them: (mode, accessor), an
        // accessor for all but points on every vertex in order (a point cloud), which need none.
        let draws: Vec<(u64, Option<usize>)> = draws
            .into_iter()
            .filter(|(_, idx)| !idx.is_empty())
            .map(|(mode, idx)| {
                let all = mode == 0 && idx.len() == n && idx.iter().enumerate().all(|(k, &i)| i as usize == k);
                (mode, (!all).then(|| self.indices(idx, n)))
            })
            .collect();
        // glTF wants tangents only beside normals, and unit length with a w of ±1.
        if has_normals && p.tangents.len() == n {
            let t: Vec<f32> = p
                .tangents
                .iter()
                .flat_map(|t| {
                    let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                    let w = if t[3] < 0.0 { -1.0 } else { 1.0 };
                    if len > 0.0 && len.is_finite() {
                        [t[0] / len, t[1] / len, t[2] / len, w]
                    } else {
                        [1.0, 0.0, 0.0, w]
                    }
                })
                .collect();
            let a = self.accessor(Part::f32(t), 5126, n, "VEC4", Some(34962));
            attributes["TANGENT"] = json!(a);
        }
        // Sets are numbered from 0 with no gaps, as glTF requires: the color texture's
        // coordinates first, which is the set the materials written here read, then the others
        // in order.
        let uvs = std::iter::once(p.uvs).chain(p.uv_sets).filter(|s| s.len() == n);
        for (k, set) in uvs.enumerate() {
            let a = self.accessor(Part::f32(set.into_flattened()), 5126, n, "VEC2", Some(34962));
            attributes[format!("TEXCOORD_{k}")] = json!(a);
        }
        let colors = std::iter::once(p.colors).chain(p.color_sets).filter(|s| s.len() == n);
        for (k, set) in colors.enumerate() {
            let mut colors = set.into_flattened();
            for c in &mut colors {
                *c = c.clamp(0.0, 1.0);
            }
            let a = self.accessor(Part::f32(colors), 5126, n, "VEC4", Some(34962));
            attributes[format!("COLOR_{k}")] = json!(a);
        }
        if let Some((offset, count)) = self.joints
            && p.skin_joints.len() == n
            && p.skin_weights.len() == n
        {
            let (joints, weights) = skin_attributes(&p.skin_joints, &p.skin_weights, offset, count);
            let j = if count <= 256 {
                let b: Vec<u8> = joints.iter().map(|&j| u8::try_from(j).unwrap_or(0)).collect();
                self.accessor(Part::bytes(b), 5121, n, "VEC4", Some(34962))
            } else {
                let b = joints.iter().map(|&j| u16::try_from(j).unwrap_or(0)).collect();
                self.accessor(Part::u16(b), 5123, n, "VEC4", Some(34962))
            };
            let w = self.accessor(Part::f32(weights), 5126, n, "VEC4", Some(34962));
            attributes["JOINTS_0"] = json!(j);
            attributes["WEIGHTS_0"] = json!(w);
        }
        let mut out = Vec::new();
        for (mode, indices) in draws {
            let mut prim = json!({ "attributes": attributes.clone() });
            if let Some(i) = indices {
                prim["indices"] = json!(i);
            }
            if mode != 4 {
                prim["mode"] = json!(mode);
            }
            out.push(prim);
        }
        out
    }

    /// An index accessor for `idx`, on `n` vertices.
    fn indices(&mut self, idx: Vec<u32>, n: usize) -> usize {
        let count = idx.len();
        // Two bytes an index when they'll do: under 65535 vertices, as glTF keeps the largest
        // value of each type for restarting strips.
        let narrow = (n < usize::from(u16::MAX))
            .then(|| idx.iter().map(|&i| u16::try_from(i).ok()).collect::<Option<Vec<u16>>>())
            .flatten();
        match narrow {
            Some(v) => self.accessor(Part::u16(v), 5123, count, "SCALAR", Some(34963)),
            None => self.accessor(Part::u32(idx), 5125, count, "SCALAR", Some(34963)),
        }
    }

    /// A mesh of `prims` (their triangles, lines and points), each using its material (the last
    /// of `materials` when out of range); `None` when none draws anything.
    pub(crate) fn mesh(&mut self, prims: Vec<Prim>, materials: usize) -> Option<Value> {
        let last = materials.saturating_sub(1);
        let mut primitives = Vec::new();
        for p in prims {
            let material = p.material.min(last);
            for mut prim in self.primitives(p) {
                prim["material"] = json!(material);
                primitives.push(prim);
            }
        }
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
    #[cfg(any(
        feature = "fbx",
        feature = "usd",
        feature = "vox",
        feature = "3ds",
        feature = "bvh",
        feature = "dae"
    ))]
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
        // A WebP image a flattened glTF kept is re-encoded as PNG now; one that doesn't decode is
        // left out, with the textures that use it.
        let images: Vec<Option<Source>> = images
            .into_iter()
            .map(|s| match s {
                Source::Image(i) => i.into_png_or_jpeg().map(Source::Image),
                r @ Source::Ref(..) => Some(r),
            })
            .collect();
        let mut next = 0;
        let written = images
            .iter()
            .map(|s| {
                s.as_ref().map(|_| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let images = images.into_iter().flatten();
        let mut w = Writer::new(written);
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
            max_triangles: MAX_TRIANGLES,
            lines_and_points: true,
            max_points: MAX_POINTS,
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

    use super::{BLOCK, FileSpan};

    #[test]
    fn a_file_span_reads_what_the_file_holds_however_it_is_read() {
        let bytes: Vec<u8> = (0..12 * BLOCK + 77)
            .map(|i| (i * 7 + i / 251).to_le_bytes()[0])
            .collect();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"head").unwrap();
        file.write_all(&bytes).unwrap();
        let span = FileSpan::new(file, 4, bytes.len() as u64, u64::MAX);
        let len = bytes.len();
        // Forward in small steps, hopping around more blocks than are kept, backward, across a
        // block's end, big reads, the very end.
        let mut reads: Vec<(usize, usize)> = (0..200).map(|k| (k * 997 % len, 1 + k % 300)).collect();
        reads.extend((0..100).map(|k| (k * 104_729 % len, 1 + k * 389 % BLOCK)));
        reads.extend([
            (BLOCK - 3, 10),
            (5, BLOCK),
            (BLOCK - 1, BLOCK - 1),
            (0, len),
            (len - 1, 1),
            (len - 40, 40),
            (100, 0),
            (BLOCK, 2),
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
