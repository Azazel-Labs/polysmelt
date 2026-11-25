//! Accessors read from their buffer views in place: strided, sparse, normalized, Meshopt-compressed or
//! Draco-decoded.

use std::sync::Arc;

use serde_json::{Map, Value};

use super::{Reader, at, index, items, list, missing, shape};
use crate::document::{Document, Tables};
use crate::gltf as container;
use crate::num::usize_of;
use crate::{Context as _, ErrorKind, Result, Within as _};

/// How much of a buffer is read at once.
const BLOCK: usize = 256 * 1024;

/// An accessor, resolved.
#[derive(Clone)]
pub(super) struct Acc {
    pub(super) view: Option<usize>,
    pub(super) offset: u64,
    pub(super) count: usize,
    pub(super) ctype: u64,
    /// Components per element.
    pub(super) n: usize,
    /// Components per column: a matrix's height, `n` for a scalar or vector.
    rows: usize,
    pub(super) normalized: bool,
    pub(super) sparse: Option<SparseAt>,
    /// Elements decoded from Draco, tightly packed, in place of a buffer view.
    pub(super) decoded: Option<Arc<[u8]>>,
}

/// A sparse accessor's substitutions, as far as they're given: what's missing is an error
/// only once they're read.
#[derive(Clone, Copy)]
pub(super) struct SparseAt {
    count: Option<usize>,
    indices: SparsePart,
    /// The indices' component type, 0 when not given.
    ictype: u64,
    values: SparsePart,
}

/// Where a sparse accessor's indices or values are.
#[derive(Clone, Copy)]
enum SparsePart {
    Missing,
    NoView,
    /// A buffer view and the offset into it.
    At(usize, u64),
}

impl SparseAt {
    fn of(s: &Value) -> Self {
        let part = |key: &str| match s.get(key) {
            None => SparsePart::Missing,
            Some(p) => match index(p, "bufferView") {
                None => SparsePart::NoView,
                Some(v) => SparsePart::At(v, p.get("byteOffset").and_then(Value::as_u64).unwrap_or(0)),
            },
        };
        Self {
            count: index(s, "count"),
            indices: part("indices"),
            ictype: at(s, &["indices", "componentType"])
                .and_then(Value::as_u64)
                .unwrap_or(0),
            values: part("values"),
        }
    }
}

/// A document's accessors and buffer views: the lists in its JSON, or the tables a reader
/// built them in.
#[derive(Clone, Copy)]
pub(super) enum Tabs<'d> {
    Json { accessors: &'d [Value], views: &'d [Value] },
    Typed(&'d Tables),
}

impl<'d> Tabs<'d> {
    pub(super) fn of(doc: &'d Document) -> Self {
        match doc.tables.as_deref() {
            Some(t) => Self::Typed(t),
            None => Self::json(&doc.base),
        }
    }

    pub(super) fn json(json: &'d Map<String, Value>) -> Self {
        Self::Json {
            accessors: list(json, "accessors"),
            views: list(json, "bufferViews"),
        }
    }

    pub(super) fn accessors(self) -> usize {
        match self {
            Self::Json { accessors, .. } => accessors.len(),
            Self::Typed(t) => t.accessors.len(),
        }
    }

    pub(super) fn views(self) -> usize {
        match self {
            Self::Json { views, .. } => views.len(),
            Self::Typed(t) => t.views.len(),
        }
    }

    /// Accessor `i`'s buffer view.
    pub(super) fn accessor_view(self, i: usize) -> Option<usize> {
        match self {
            Self::Json { accessors, .. } => index(accessors.get(i)?, "bufferView"),
            Self::Typed(t) => t.accessors.get(i)?.buffer_view,
        }
    }

    /// Accessor `i`'s `count`.
    pub(super) fn accessor_count(self, i: usize) -> Option<u64> {
        match self {
            Self::Json { accessors, .. } => accessors.get(i)?.get("count")?.as_u64(),
            Self::Typed(t) => Some(t.accessors.get(i)?.count as u64),
        }
    }

    /// Whether accessor `i` declares a `min` or `max`.
    pub(super) fn declares_bounds(self, i: usize) -> bool {
        match self {
            Self::Json { accessors, .. } => accessors
                .get(i)
                .is_some_and(|x| x.get("min").is_some() || x.get("max").is_some()),
            Self::Typed(t) => t.accessors.get(i).is_some_and(|x| x.min.is_some() || x.max.is_some()),
        }
    }

    /// Whether accessor `i` is sparse.
    pub(super) fn is_sparse(self, i: usize) -> bool {
        match self {
            Self::Json { accessors, .. } => accessors.get(i).is_some_and(|x| x.get("sparse").is_some()),
            Self::Typed(t) => t.accessors.get(i).is_some_and(|x| x.sparse.is_some()),
        }
    }

    /// Whether buffer view `v` gives a `byteStride`.
    pub(super) fn view_strided(self, v: usize) -> bool {
        match self {
            Self::Json { views, .. } => views.get(v).is_some_and(|v| v.get("byteStride").is_some()),
            Self::Typed(_) => false,
        }
    }

    /// Whether buffer view `v` has extensions.
    pub(super) fn view_extended(self, v: usize) -> bool {
        match self {
            Self::Json { views, .. } => views.get(v).is_some_and(|v| v.get("extensions").is_some()),
            Self::Typed(_) => false,
        }
    }
}

impl Acc {
    fn csize(&self) -> usize {
        match self.ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4,
        }
    }

    /// Bytes from one column to the next: a matrix's columns start 4-byte aligned (3.6.2.4).
    fn column(&self) -> usize {
        (self.rows * self.csize()).next_multiple_of(if self.rows < self.n { 4 } else { 1 })
    }

    pub(super) fn esize(&self) -> usize {
        self.column() * (self.n / self.rows.max(1))
    }

    /// Whether a matrix's columns are padded, so components aren't `csize` apart.
    pub(super) fn padded(&self) -> bool {
        self.esize() != self.csize() * self.n
    }

    /// Component `c`'s byte offset in an element.
    fn at(&self, c: usize) -> usize {
        (c / self.rows) * self.column() + (c % self.rows) * self.csize()
    }
}

/// One component as a float: normalized integers mapped to [0, 1] or [-1, 1].
#[allow(
    clippy::cast_precision_loss,
    reason = "past 2^24 an f32 attribute rounds, as in any viewer"
)]
fn component(ctype: u64, normalized: bool, b: &[u8]) -> f32 {
    match (ctype, normalized) {
        (5126, _) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (5121, true) => f32::from(b[0]) / 255.0,
        (5121, false) => f32::from(b[0]),
        (5120, true) => (f32::from(b[0].cast_signed()) / 127.0).max(-1.0),
        (5120, false) => f32::from(b[0].cast_signed()),
        (5123, true) => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
        (5123, false) => f32::from(u16::from_le_bytes([b[0], b[1]])),
        (5122, true) => (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0),
        (5122, false) => f32::from(i16::from_le_bytes([b[0], b[1]])),
        _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
    }
}

impl<'d> Reader<'d> {
    pub(super) fn get(&self, key: &str, i: usize) -> Result<&'d Value> {
        let all = match (key, self.tabs) {
            ("accessors", Tabs::Json { accessors, .. }) => accessors,
            ("bufferViews", Tabs::Json { views, .. }) => views,
            ("meshes", _) => self.meshes,
            _ => list(self.json, key),
        };
        all.get(i).with_context(ErrorKind::Corrupt, || missing(key, i))
    }

    /// A buffer view as (buffer, offset, length, stride), and whether it has extensions (as a
    /// meshopt-compressed one does).
    pub(super) fn view(&self, i: usize) -> Result<(usize, u64, u64, Option<usize>, bool)> {
        if let Tabs::Typed(t) = self.tabs {
            let v = t
                .views
                .get(i)
                .with_context(ErrorKind::Corrupt, || missing("bufferViews", i))?;
            return Ok((v.buffer, v.byte_offset.unwrap_or(0), v.byte_length, None, false));
        }
        let v = self.get("bufferViews", i)?;
        // One pass over its members rather than a lookup for each: these are read for every
        // accessor.
        let (mut buffer, mut offset, mut len, mut stride, mut ext) = (None, None, None, None, false);
        for (k, x) in v.as_object().into_iter().flatten() {
            match k.as_str() {
                "buffer" => buffer = Some(x),
                "byteOffset" => offset = Some(x),
                "byteLength" => len = Some(x),
                "byteStride" => stride = Some(x),
                "extensions" => ext = true,
                _ => {}
            }
        }
        let buffer = buffer
            .and_then(Value::as_u64)
            .map(usize_of)
            .context(ErrorKind::Corrupt, "glTF buffer view has no buffer")?;
        let offset = offset.and_then(Value::as_u64).unwrap_or(0);
        let len = len
            .and_then(Value::as_u64)
            .context(ErrorKind::Corrupt, "glTF buffer view has no byteLength")?;
        Ok((buffer, offset, len, stride.and_then(Value::as_u64).map(usize_of), ext))
    }

    /// Whether buffer view `i` is meshopt-compressed, decoding it (once) if so.
    pub(super) fn meshopt(&mut self, i: usize) -> Result<bool> {
        if self.meshopt.contains_key(&i) {
            return Ok(true);
        }
        if let Tabs::Typed(t) = self.tabs {
            // Only a file's views are compressed.
            t.views
                .get(i)
                .with_context(ErrorKind::Corrupt, || missing("bufferViews", i))?;
            return Ok(false);
        }
        let v = self.get("bufferViews", i)?;
        let Some((name, ext)) = container::meshopt_of(v) else {
            return Ok(false);
        };
        let len = usize_of(v.get("byteLength").and_then(Value::as_u64).unwrap_or(0));
        let b = ext
            .get("buffer")
            .and_then(Value::as_u64)
            .map(usize_of)
            .with_context(ErrorKind::Corrupt, || {
                format!("glTF buffer view {i}: {name} names no buffer")
            })?;
        let off = ext.get("byteOffset").and_then(Value::as_u64).unwrap_or(0);
        let src_len = ext
            .get("byteLength")
            .and_then(Value::as_u64)
            .with_context(ErrorKind::Corrupt, || {
                format!("glTF buffer view {i}: {name} has no byteLength")
            })?;
        let blob = self.doc.buffers.get(b).with_context(ErrorKind::Corrupt, || {
            format!("glTF buffer view {i}: {name} names missing buffer {b}")
        })?;
        let held;
        let src = if let Some(s) = blob.slice(off, src_len) {
            s
        } else {
            ensure!(
                off.saturating_add(src_len) <= blob.len(),
                Truncated,
                "glTF buffer view {i}: {name} data runs past the end of its buffer"
            );
            let mut v = vec![0; usize_of(src_len)];
            blob.read_at(off, &mut v)?;
            held = v;
            &held
        };
        // No meshopt codec expands its data more than 1024 times: a hostile length can't make
        // us allocate.
        ensure!(
            len / 1024 <= src.len(),
            Truncated,
            "glTF buffer view {i}: {name} data is cut short"
        );
        let mut out = vec![0; len];
        container::decode_view(ext, src, &mut out).with_within(|| format!("glTF buffer view {i}"))?;
        self.meshopt.insert(i, out);
        Ok(true)
    }

    /// A buffer view's bytes, shared with its buffer when they can be.
    pub(super) fn view_shared(&mut self, i: usize) -> Result<crate::Bytes> {
        if self.meshopt(i)? {
            return Ok(crate::Bytes::copy_from_slice(&self.meshopt[&i]));
        }
        let (buffer, offset, len, _, _) = self.view(i)?;
        let b = self
            .doc
            .buffers
            .get(buffer)
            .context(ErrorKind::Corrupt, "glTF buffer view names a missing buffer")?;
        ensure!(
            offset.saturating_add(len) <= b.len(),
            Truncated,
            "glTF buffer view runs past the end of its buffer"
        );
        usize::try_from(len).map_err(|_| err!(LimitExceeded, "glTF buffer view is too big to hold"))?;
        b.shared(offset, len)
    }

    pub(super) fn acc(&self, i: usize) -> Result<Acc> {
        if let Some((_, bytes, ctype, n)) = self.draco.iter().find(|d| d.0 == i) {
            let count = bytes.len() / (n * 4);
            return Ok(Acc {
                view: None,
                offset: 0,
                count,
                ctype: *ctype,
                n: *n,
                rows: *n,
                normalized: false,
                sparse: None,
                decoded: Some(bytes.clone()),
            });
        }
        if let Tabs::Typed(t) = self.tabs {
            let a = t
                .accessors
                .get(i)
                .with_context(ErrorKind::Corrupt, || missing("accessors", i))?;
            let ctype = u64::from(a.component_type);
            ensure!(
                matches!(ctype, 5120..=5123 | 5125 | 5126),
                Corrupt,
                "glTF accessor {i} has component type {ctype}"
            );
            let (n, rows) = shape(Some(a.kind)).with_context(ErrorKind::Corrupt, || {
                format!("glTF accessor {i} has type {:?}, not one read here", Some(a.kind))
            })?;
            return Ok(Acc {
                view: a.buffer_view,
                offset: a.byte_offset.unwrap_or(0),
                count: a.count,
                ctype,
                n,
                rows,
                normalized: a.normalized,
                sparse: a.sparse.map(|s| SparseAt {
                    count: Some(s.count),
                    indices: SparsePart::At(s.indices, 0),
                    ictype: u64::from(s.index_type),
                    values: SparsePart::At(s.values, 0),
                }),
                decoded: None,
            });
        }
        let a = self.get("accessors", i)?;
        // One pass over its members, as for a view.
        let (mut ctype, mut kind, mut view, mut offset, mut count, mut normalized, mut sparse) =
            (None, None, None, None, None, None, None);
        for (k, x) in a.as_object().into_iter().flatten() {
            match k.as_str() {
                "componentType" => ctype = Some(x),
                "type" => kind = Some(x),
                "bufferView" => view = Some(x),
                "byteOffset" => offset = Some(x),
                "count" => count = Some(x),
                "normalized" => normalized = Some(x),
                "sparse" => sparse = Some(x),
                _ => {}
            }
        }
        let ctype = ctype.and_then(Value::as_u64).unwrap_or(0);
        ensure!(
            matches!(ctype, 5120..=5123 | 5125 | 5126),
            Corrupt,
            "glTF accessor {i} has component type {ctype}"
        );
        let kind = kind.and_then(Value::as_str);
        let (n, rows) = shape(kind).with_context(ErrorKind::Corrupt, || {
            format!("glTF accessor {i} has type {kind:?}, not one read here")
        })?;
        let u = |x: Option<&Value>| x.and_then(Value::as_u64).map(usize_of);
        Ok(Acc {
            view: u(view),
            offset: offset.and_then(Value::as_u64).unwrap_or(0),
            count: u(count).context(ErrorKind::Corrupt, "glTF accessor has no count")?,
            ctype,
            n,
            rows,
            normalized: normalized.and_then(Value::as_bool).unwrap_or(false),
            sparse: sparse.map(SparseAt::of),
            decoded: None,
        })
    }

    /// Calls `f(first, bytes, stride)` over an accessor's elements a block at a time: element
    /// `first + j` is at `bytes[j * stride..]`. Nothing for an accessor without a buffer view.
    pub(super) fn blocks(&mut self, a: &Acc, mut f: impl FnMut(usize, &[u8], usize)) -> Result<()> {
        if let Some(bytes) = &a.decoded {
            f(0, bytes, a.esize());
            return Ok(());
        }
        let Some(view) = a.view else {
            ensure!(
                a.count <= self.max_vertices,
                LimitExceeded,
                "glTF accessor counts {} elements",
                a.count
            );
            return Ok(());
        };
        if a.count == 0 {
            return Ok(());
        }
        let (buffer, voff, vlen, stride, ext) = self.view(view)?;
        let esize = a.esize();
        let stride = stride.filter(|&s| s > 0).unwrap_or(esize);
        ensure!(
            stride >= esize,
            Corrupt,
            "glTF buffer view stride {stride} is under an element's {esize} bytes"
        );
        // The elements' bytes, from the first: the accessor's offset into its view is checked here, not counted twice.
        let span = (a.count as u64 - 1)
            .checked_mul(stride as u64)
            .and_then(|s| s.checked_add(esize as u64))
            .filter(|&s| a.offset.checked_add(s).is_some_and(|end| end <= vlen))
            .with_context(ErrorKind::Corrupt, || {
                format!(
                    "glTF accessor's {} elements run past the end of their buffer view",
                    a.count
                )
            })?;
        // Only a view with extensions can be meshopt-compressed.
        if ext && self.meshopt(view)? {
            let bytes = &self.meshopt[&view];
            let s = bytes.get(usize_of(a.offset)..usize_of(a.offset + span)).context(
                ErrorKind::Corrupt,
                "glTF accessor runs past the end of its decoded buffer view",
            )?;
            f(0, s, stride);
            return Ok(());
        }
        let start = voff + a.offset;
        let b = self
            .doc
            .buffers
            .get(buffer)
            .context(ErrorKind::Corrupt, "glTF buffer view names a missing buffer")?;
        ensure!(
            start + span <= b.len(),
            Truncated,
            "glTF accessor runs past the end of its buffer: bytes {start}..{} of buffer {buffer}, which holds {}",
            start + span,
            b.len()
        );
        if let Some(s) = b.slice(start, span) {
            f(0, s, stride);
            return Ok(());
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        let per = (BLOCK / stride).max(1);
        let mut i = 0;
        while i < a.count {
            let k = per.min(a.count - i);
            let len = (k - 1) * stride + esize;
            if scratch.len() < len {
                scratch.resize(len, 0);
            }
            b.read_at(start + (i * stride) as u64, &mut scratch[..len])?;
            f(i, &scratch[..len], stride);
            i += k;
        }
        self.scratch = scratch;
        Ok(())
    }

    /// An accessor's elements as floats, components past the accessor's from `fill`, integer
    /// components normalized when the accessor says so.
    pub(super) fn read<const N: usize>(&mut self, i: usize, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let a = self.acc(i)?;
        self.read_acc(&a, fill)
    }

    pub(super) fn read_acc<const N: usize>(&mut self, a: &Acc, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let mut zero = fill;
        for z in zero.iter_mut().take(a.n) {
            *z = 0.0;
        }
        if a.view.is_none() {
            ensure!(
                a.count <= self.max_vertices,
                LimitExceeded,
                "glTF accessor counts {} elements",
                a.count
            );
        }
        let count = a.count;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| err!(LimitExceeded, "glTF accessor counts {count} elements"))?;
        let m = a.n.min(N);
        let cs = a.csize();
        let (ctype, normalized) = (a.ctype, a.normalized);
        let offsets: Option<Vec<usize>> = a.padded().then(|| (0..m).map(|c| a.at(c)).collect());
        // Elements are appended as the blocks come, in order, rather than written over zeros
        // laid down first: that filling cost as much as the reading. Elements the data doesn't
        // reach (no view, a short decoded stream) are `zero`, filled in after.
        self.blocks(a, |first, bytes, stride| {
            if out.len() < first {
                out.resize(first.min(count), zero);
            }
            // Never past the accessor's count, whatever a decoded stream holds.
            let room = count - out.len();
            let bytes = &bytes[..bytes.len().min(room.saturating_mul(stride))];
            if let Some(offsets) = &offsets {
                // A small matrix's padded columns: rare, so one general loop.
                out.extend(bytes.chunks(stride).map(|e| {
                    let mut o = zero;
                    for (x, &at) in o.iter_mut().zip(offsets) {
                        *x = component(ctype, normalized, &e[at..]);
                    }
                    o
                }));
                return;
            }
            if ctype == 5126 && m == N && stride == 4 * N {
                // Tightly packed floats (what nearly every exporter writes): one flat loop.
                out.extend(bytes.chunks_exact(4 * N).map(|e| {
                    std::array::from_fn(|c| f32::from_le_bytes([e[4 * c], e[4 * c + 1], e[4 * c + 2], e[4 * c + 3]]))
                }));
                return;
            }
            let o = &mut out;
            match (ctype, normalized) {
                (5126, _) => put(o, bytes, stride, cs, m, zero, |b| {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                }),
                (5121, true) => put(o, bytes, stride, cs, m, zero, |b| f32::from(b[0]) / 255.0),
                (5121, false) => put(o, bytes, stride, cs, m, zero, |b| f32::from(b[0])),
                (5120, true) => put(o, bytes, stride, cs, m, zero, |b| {
                    (f32::from(b[0].cast_signed()) / 127.0).max(-1.0)
                }),
                (5120, false) => put(o, bytes, stride, cs, m, zero, |b| f32::from(b[0].cast_signed())),
                (5123, true) => put(o, bytes, stride, cs, m, zero, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0
                }),
                (5123, false) => put(o, bytes, stride, cs, m, zero, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]]))
                }),
                (5122, true) => put(o, bytes, stride, cs, m, zero, |b| {
                    (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0)
                }),
                (5122, false) => put(o, bytes, stride, cs, m, zero, |b| {
                    f32::from(i16::from_le_bytes([b[0], b[1]]))
                }),
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "past 2^24 an f32 attribute rounds, as in any viewer"
                )]
                _ => put(o, bytes, stride, cs, m, zero, |b| {
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32
                }),
            }
        })?;
        out.resize(count, zero);
        if let Some(s) = a.sparse {
            let (at, values) = self.sparse(a, &s)?;
            let values = self.read_acc(&values, fill)?;
            for (k, v) in at.into_iter().zip(values) {
                if let Some(o) = out.get_mut(k as usize) {
                    *o = v;
                }
            }
        }
        Ok(out)
    }

    /// An index accessor's values.
    /// Decodes the Draco stream holding accessor `i`, when it has no buffer view of its own and
    /// a primitive's `KHR_draco_mesh_compression` names it.
    pub(super) fn draco_holding(&mut self, i: usize) -> Result<()> {
        if self.draco.iter().any(|d| d.0 == i) {
            return Ok(());
        }
        let viewed = if let Tabs::Typed(t) = self.tabs {
            let a = t
                .accessors
                .get(i)
                .with_context(ErrorKind::Corrupt, || missing("accessors", i))?;
            a.buffer_view.is_some()
        } else {
            self.get("accessors", i)?.get("bufferView").is_some()
        };
        if viewed {
            return Ok(());
        }
        let meshes = self.meshes;
        let found = meshes.iter().flat_map(|m| items(m, "primitives")).find_map(|p| {
            let ext = p.get("extensions")?.get("KHR_draco_mesh_compression")?;
            let attrs = p.get("attributes")?;
            let names = ext.get("attributes").and_then(Value::as_object);
            let holds = index(p, "indices") == Some(i)
                || names
                    .into_iter()
                    .flat_map(|n| n.keys())
                    .any(|n| index(attrs, n) == Some(i));
            holds.then(|| (ext, attrs, index(p, "indices")))
        });
        let Some((ext, attrs, indices)) = found else {
            return Ok(());
        };
        #[cfg(feature = "draco")]
        {
            self.decode_draco(ext, attrs, indices)
        }
        #[cfg(not(feature = "draco"))]
        {
            let _ = (ext, attrs, indices);
            bail!(
                FeatureDisabled,
                "glTF accessor {i} is Draco-compressed, which needs Polysmelt's `draco` feature"
            )
        }
    }

    pub(super) fn read_indices(&mut self, i: usize) -> Result<Vec<u32>> {
        let a = self.acc(i)?;
        self.read_index_acc(i, &a)
    }

    /// Index accessor `i`'s values, `a` being that accessor.
    pub(super) fn read_index_acc(&mut self, i: usize, a: &Acc) -> Result<Vec<u32>> {
        ensure!(
            a.n == 1 && matches!(a.ctype, 5121 | 5123 | 5125),
            Corrupt,
            "glTF index accessor {i} isn't unsigned integer scalars"
        );
        self.read_u32s(a)
    }

    fn read_u32s(&mut self, a: &Acc) -> Result<Vec<u32>> {
        let count = a.count;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| err!(LimitExceeded, "glTF accessor counts {count} elements"))?;
        let ctype = a.ctype;
        // Appended in order, as `read_acc` does; what the data doesn't reach is 0.
        self.blocks(a, |first, bytes, stride| {
            if out.len() < first {
                out.resize(first.min(count), 0);
            }
            let room = count - out.len();
            let bytes = &bytes[..bytes.len().min(room.saturating_mul(stride))];
            let need = match ctype {
                5121 => 1,
                5123 => 2,
                _ => 4,
            };
            let whole = |e: &&[u8]| e.len() >= need;
            match ctype {
                5121 => out.extend(bytes.chunks(stride).map(|e| u32::from(e[0]))),
                5123 if stride == 2 => out.extend(
                    bytes
                        .chunks_exact(2)
                        .map(|e| u32::from(u16::from_le_bytes([e[0], e[1]]))),
                ),
                5125 if stride == 4 => out.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|e| u32::from_le_bytes([e[0], e[1], e[2], e[3]])),
                ),
                5123 => out.extend(
                    bytes
                        .chunks(stride)
                        .filter(whole)
                        .map(|e| u32::from(u16::from_le_bytes([e[0], e[1]]))),
                ),
                _ => out.extend(
                    bytes
                        .chunks(stride)
                        .filter(whole)
                        .map(|e| u32::from_le_bytes([e[0], e[1], e[2], e[3]])),
                ),
            }
        })?;
        out.resize(count, 0);
        if let Some(s) = a.sparse {
            let (at, values) = self.sparse(a, &s)?;
            let values = self.read_u32s(&values)?;
            for (k, v) in at.into_iter().zip(values) {
                if let Some(o) = out.get_mut(k as usize) {
                    *o = v;
                }
            }
        }
        Ok(out)
    }

    /// A sparse accessor's indices, and an accessor for its values.
    fn sparse(&mut self, a: &Acc, s: &SparseAt) -> Result<(Vec<u32>, Acc)> {
        let count = s
            .count
            .context(ErrorKind::Corrupt, "glTF sparse accessor has no count")?;
        ensure!(
            count <= a.count,
            Corrupt,
            "glTF sparse accessor replaces more elements than it has"
        );
        let part = |key: &str, p: SparsePart| -> Result<(usize, u64)> {
            match p {
                SparsePart::At(v, off) => Ok((v, off)),
                SparsePart::Missing => {
                    None.with_context(ErrorKind::Corrupt, || format!("glTF sparse accessor has no {key}"))
                }
                SparsePart::NoView => {
                    None.with_context(ErrorKind::Corrupt, || format!("glTF sparse {key} has no bufferView"))
                }
            }
        };
        let (iv, ioff) = part("indices", s.indices)?;
        let ictype = s.ictype;
        ensure!(
            matches!(ictype, 5121 | 5123 | 5125),
            Corrupt,
            "glTF sparse indices' component type {ictype}"
        );
        let idx = Acc {
            view: Some(iv),
            offset: ioff,
            count,
            ctype: ictype,
            n: 1,
            rows: 1,
            normalized: false,
            sparse: None,
            decoded: None,
        };
        let at = self.read_u32s(&idx)?;
        let (vv, voff) = part("values", s.values)?;
        let values = Acc {
            view: Some(vv),
            offset: voff,
            sparse: None,
            decoded: None,
            count,
            ..a.clone()
        };
        Ok((at, values))
    }
}

/// Appends each element in `bytes` to `out`: its first `m` components converted, the rest
/// (and the whole of an element cut short) from `zero`.
#[inline]
fn put<const N: usize>(
    out: &mut Vec<[f32; N]>,
    bytes: &[u8],
    stride: usize,
    csize: usize,
    m: usize,
    zero: [f32; N],
    conv: impl Fn(&[u8]) -> f32,
) {
    if m == N && stride == N * csize {
        // Every component given, packed tightly (joint indices, colors): whole elements only,
        // so the loop has no length check. A short element at the end is `zero`, as below.
        let elements = bytes.chunks_exact(stride);
        let short = !elements.remainder().is_empty();
        out.extend(elements.map(|e| std::array::from_fn(|c| conv(&e[c * csize..]))));
        if short {
            out.push(zero);
        }
        return;
    }
    if m == N {
        // Every component given (the usual case): a loop the compiler unrolls.
        out.extend(bytes.chunks(stride).map(|e| {
            if e.len() >= N * csize {
                std::array::from_fn(|c| conv(&e[c * csize..]))
            } else {
                zero
            }
        }));
        return;
    }
    out.extend(bytes.chunks(stride).map(|e| {
        let mut o = zero;
        if e.len() >= m * csize {
            for (c, x) in o.iter_mut().enumerate().take(m) {
                *x = conv(&e[c * csize..]);
            }
        }
        o
    }));
}
