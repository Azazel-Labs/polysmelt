//! [`Document::repair`]: what glTF validators reject, put right without changing what draws.
//!
//! Every GLB [`Document::write_glb`] writes goes through this, so documents read from files
//! that break the rules (and readers' own slips) come out valid. A document already valid is
//! left as it is, byte for byte.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::{Map, Value, json};

use super::{Acc, Reader, index, items, list, retain_groups, segments, triangles};
use crate::document::{Blob, Document};
use crate::num::{f32_of, usize_of};

/// The largest a unit vector's length may be off by before it's rewritten: well inside the
/// validator's 0.00674, so lengths that pass it are left alone.
const UNIT: f64 = 5e-4;

impl Document<'_> {
    /// Puts right what the glTF 2.0 specification forbids and can be fixed without changing
    /// what the document draws or how it moves:
    ///
    /// - indices naming no vertex dropped with the triangle, line or point using them (strips,
    ///   loops and fans becoming lists), and primitives left with none removed;
    /// - non-finite floats in vertex attributes, animation keys and inverse bind matrices set
    ///   to 0;
    /// - normals, tangents and rotation keys made unit length (a zero normal takes its faces'
    ///   normal), tangents' handedness made ±1, and each vertex's skin weights scaled to sum
    ///   to 1;
    /// - animation keys sorted by time, of keys at the same time the last kept;
    /// - animation and inverse bind matrix data copied out of interleaved buffer views, and
    ///   vertex attributes sharing a buffer view without a stride given views of their own;
    /// - `min` and `max` added where they're required and corrected where they're wrong
    ///   (per-instance attributes' too);
    /// - texture coordinate sets a material reads but a primitive lacks filled in (by another
    ///   set, or zeros), as are gaps in their numbering;
    /// - node hierarchies made trees (a second parent, or a cycle, loses the link), and scene
    ///   roots that are children dropped;
    /// - empty lists removed.
    ///
    /// Rewritten data goes into a new buffer. [`Document::write_glb`] and
    /// [`Document::to_glb`] do this to what they write, so calling it first is needed only to
    /// look at the result.
    pub fn repair(&mut self) {
        let mut json = self.json.clone();
        if let Some(bytes) = repaired(self, &mut json) {
            self.buffers.push(Blob::Bytes(bytes.into()));
        }
        self.json = json;
    }
}

/// Repairs `json`, a copy of `doc`'s, as [`Document::repair`] describes. Returns the bytes
/// of the new buffer that rewritten data went into, which `json` already lists after `doc`'s
/// buffers; `None` when nothing needed new data. A step that can't read what it needs is
/// skipped, so this never fails.
pub(crate) fn repaired(doc: &Document<'_>, json: &mut Map<String, Value>) -> Option<Vec<u8>> {
    repair_json(doc, json, false).0
}

/// [`repaired`], and the vertex and index data it read from files, for writing them without
/// reading them again: (buffer, offset, bytes).
pub(crate) fn repaired_keeping(doc: &Document<'_>, json: &mut Map<String, Value>) -> (Option<Vec<u8>>, Vec<Kept>) {
    repair_json(doc, json, true)
}

/// Bytes of buffer `.0` at offset `.1`, as read from its file.
pub(crate) type Kept = (usize, u64, Vec<u8>);

fn repair_json(doc: &Document<'_>, json: &mut Map<String, Value>, keep: bool) -> (Option<Vec<u8>>, Vec<Kept>) {
    let mut fix = Fix {
        doc,
        json,
        buffer: doc.buffers.len(),
        out: Vec::new(),
        done: BTreeSet::new(),
        keep,
        kept: Vec::new(),
    };
    let mut r = Reader::bare(doc);
    fix.strides(&mut r);
    let drop = fix.indices(&mut r);
    fix.vertices(&mut r);
    fix.weights(&mut r);
    fix.animations(&mut r);
    fix.skins(&mut r);
    fix.texcoords();
    fix.nodes();
    fix.prune(&drop);
    if fix.out.is_empty() {
        return (None, fix.kept);
    }
    fix.out.resize(fix.out.len().next_multiple_of(4), 0);
    let len = fix.out.len();
    push(fix.json, "buffers", json!({ "byteLength": len }));
    (Some(fix.out), fix.kept)
}

struct Fix<'d, 'j> {
    doc: &'d Document<'d>,
    /// The copy being repaired. Data is read from `doc`, whose accessors keep their indices.
    json: &'j mut Map<String, Value>,
    /// The new buffer's index.
    buffer: usize,
    out: Vec<u8>,
    /// Accessors already rewritten.
    done: BTreeSet<usize>,
    /// Whether to keep what's read from files, in `kept`.
    keep: bool,
    kept: Vec<Kept>,
}

/// What a vertex accessor is used as, for what's checked in it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Position,
    Normal,
    Tangent,
    Indices,
    /// Any other attribute, or a morph target's.
    Other,
    /// A node's per-instance attribute (`EXT_mesh_gpu_instancing`), whose view has no target.
    Instance,
}

/// The primitives as (mesh, primitive, JSON), Draco-compressed ones (whose accessors hold no
/// data) left out.
fn prims(json: &Map<String, Value>) -> Vec<(usize, usize, &Value)> {
    let mut out = Vec::new();
    for (m, mesh) in list(json, "meshes").iter().enumerate() {
        for (p, prim) in items(mesh, "primitives").iter().enumerate() {
            let draco = prim
                .get("extensions")
                .and_then(|e| e.get("KHR_draco_mesh_compression"))
                .is_some();
            if !draco {
                out.push((m, p, prim));
            }
        }
    }
    out
}

fn attribute_sets(prim: &Value) -> impl Iterator<Item = (bool, &Map<String, Value>)> {
    prim.get("attributes")
        .and_then(Value::as_object)
        .map(|a| (true, a))
        .into_iter()
        .chain(
            items(prim, "targets")
                .iter()
                .filter_map(Value::as_object)
                .map(|t| (false, t)),
        )
}

fn push(json: &mut Map<String, Value>, key: &str, v: Value) -> usize {
    let Some(l) = json.entry(key).or_insert_with(|| json!([])).as_array_mut() else {
        return 0;
    };
    l.push(v);
    l.len() - 1
}

fn accessor_mut(json: &mut Map<String, Value>, a: usize) -> Option<&mut Map<String, Value>> {
    json.get_mut("accessors")?.as_array_mut()?.get_mut(a)?.as_object_mut()
}

fn primitive_mut(json: &mut Map<String, Value>, m: usize, p: usize) -> Option<&mut Map<String, Value>> {
    json.get_mut("meshes")?
        .as_array_mut()?
        .get_mut(m)?
        .get_mut("primitives")?
        .as_array_mut()?
        .get_mut(p)?
        .as_object_mut()
}

fn csize(ctype: u64) -> usize {
    match ctype {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        _ => 4,
    }
}

/// What a scan of a vertex accessor found.
enum Scan {
    /// Unreadable: left as it is.
    Skip,
    /// Values to put right.
    Fix,
    /// Fine, with its bounds when they're checked.
    Fine(Option<(Vec<f64>, Vec<f64>)>),
}

/// An accessor's elements, tightly packed. `None` for one this doesn't rewrite: sparse,
/// without data, a matrix with padded columns, or unreadable.
/// In place when they're in memory in one piece, as most are: checking a valid document then
/// copies nothing.
fn raw<'d>(r: &mut Reader<'d>, a: &Acc) -> Option<Cow<'d, [u8]>> {
    located(r, a).map(|x| x.0)
}

/// [`raw`], and where in which buffer the bytes were read from when they were read from a
/// file in one piece.
fn located<'d>(r: &mut Reader<'d>, a: &Acc) -> Option<Located<'d>> {
    if a.sparse.is_some() || a.view.is_none() || a.padded() || a.decoded.is_some() {
        return None;
    }
    if let Some(s) = tight(r, a) {
        return Some(s);
    }
    let es = a.esize();
    let mut out = Vec::with_capacity(a.count.saturating_mul(es));
    r.blocks(a, |_, bytes, stride| {
        if stride == es {
            out.extend_from_slice(bytes);
        } else {
            let k = (bytes.len() - es) / stride + 1;
            for j in 0..k {
                out.extend_from_slice(&bytes[j * stride..j * stride + es]);
            }
        }
    })
    .ok()?;
    (out.len() == a.count * es).then_some((Cow::Owned(out), None))
}

/// Bytes from [`located`], and their buffer and offset when they were read from a file.
type Located<'d> = (Cow<'d, [u8]>, Option<(usize, u64)>);

/// `bytes` from [`located`], to keep when they were read from a file.
fn kept(keep: bool, bytes: Cow<'_, [u8]>, at: Option<(usize, u64)>) -> Option<Kept> {
    match (bytes, at) {
        // Small ones aren't worth it: writing around them costs more than reading them again.
        (Cow::Owned(v), Some((buffer, start))) if keep && v.len() >= 64 << 10 => Some((buffer, start, v)),
        _ => None,
    }
}

/// An accessor's elements when they're tightly packed in an uncompressed view: where they lie
/// when that's in memory in one piece, else in one read of the file.
fn tight<'d>(r: &mut Reader<'d>, a: &Acc) -> Option<Located<'d>> {
    let (buffer, voff, vlen, stride, ext) = r.view(a.view?).ok()?;
    let es = a.esize();
    if ext || a.count == 0 || stride.is_some_and(|s| s > 0 && s != es) {
        return None;
    }
    let span = (a.count as u64).checked_mul(es as u64)?;
    if a.offset.checked_add(span)? > vlen {
        return None;
    }
    let doc: &'d Document<'d> = r.doc;
    let blob = doc.buffers.get(buffer)?;
    let start = voff.checked_add(a.offset)?;
    if let Some(s) = blob.slice(start, span) {
        return Some((Cow::Borrowed(s), None));
    }
    // Checked against the bytes there before allocating: a view's length is only what the
    // file claims.
    if start.checked_add(span)? > blob.len() {
        return None;
    }
    let mut out = vec![0; usize::try_from(span).ok()?];
    blob.read_at(start, &mut out).ok()?;
    Some((Cow::Owned(out), Some((buffer, start))))
}

/// The components of a vertex accessor used as `role` that are to be unit length: 3 for
/// normals, 4 for tangents (the last their handedness), `None` for the rest.
fn unit_size(role: Role, n: usize) -> Option<usize> {
    match role {
        Role::Normal if n == 3 => Some(3),
        Role::Tangent if n == 4 => Some(4),
        _ => None,
    }
}

/// Whether every little-endian `f32` in `bytes` is finite: the exponent bits not all set.
fn all_finite(bytes: &[u8]) -> bool {
    !bytes.chunks_exact(4).fold(false, |bad, b| {
        bad | (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
    })
}

/// Whether every packed `f32` VEC3 (or the first three of each VEC4, when `n` is 4) is unit
/// length, and with `n` 4 has a handedness of ±1.
#[allow(clippy::float_cmp, reason = "handedness must be exactly ±1")]
fn all_unit(bytes: &[u8], n: usize) -> bool {
    let f = |b: &[u8], c: usize| f32::from_le_bytes([b[4 * c], b[4 * c + 1], b[4 * c + 2], b[4 * c + 3]]);
    !bytes.chunks_exact(4 * n).fold(false, |bad, e| {
        let sq = |c| f64::from(f(e, c)) * f64::from(f(e, c));
        let off = off_unit_sq(sq(0) + sq(1) + sq(2));
        bad | off | (n == 4 && f(e, 3) != 1.0 && f(e, 3) != -1.0)
    })
}

/// Whether every index is under `vcount` and none is its type's restart value.
#[allow(
    clippy::needless_bitwise_bool,
    reason = "without a branch per index, the loop vectorizes"
)]
fn indices_fit(bytes: &[u8], ctype: u64, vcount: u64) -> bool {
    let (top, restart) = match ctype {
        5121 => bytes
            .iter()
            .fold((0u32, false), |(m, r), &i| (m.max(u32::from(i)), r | (i == 0xFF))),
        5123 => bytes.chunks_exact(2).fold((0u32, false), |(m, r), b| {
            let i = u16::from_le_bytes([b[0], b[1]]);
            (m.max(u32::from(i)), r | (i == 0xFFFF))
        }),
        _ => bytes.chunks_exact(4).fold((0u32, false), |(m, r), b| {
            let i = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            (m.max(i), r | (i == u32::MAX))
        }),
    };
    bytes.is_empty() || (!restart && u64::from(top) < vcount)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bytes_of(f: &[f32]) -> Vec<u8> {
    f.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Sets non-finite floats to 0; whether there were any.
fn finite(f: &mut [f32]) -> bool {
    let mut changed = false;
    for x in f.iter_mut().filter(|x| !x.is_finite()) {
        *x = 0.0;
        changed = true;
    }
    changed
}

/// Component `c` of raw bytes as a number: integers exactly.
fn number(ctype: u64, b: &[u8]) -> f64 {
    match ctype {
        5120 => f64::from(b[0].cast_signed()),
        5121 => f64::from(b[0]),
        5122 => f64::from(i16::from_le_bytes([b[0], b[1]])),
        5123 => f64::from(u16::from_le_bytes([b[0], b[1]])),
        5125 => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        _ => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    }
}

/// Packed elements' `min` and `max`, as glTF writes them; `None` for no elements.
fn bounds(bytes: &[u8], ctype: u64, n: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let es = csize(ctype) * n;
    if es == 0 || bytes.len() < es {
        return None;
    }
    let (mut min, mut max) = match ctype {
        5120 => min_max::<1, i8>(bytes, n, |b| b[0].cast_signed()),
        5121 => min_max::<1, u8>(bytes, n, |b| b[0]),
        5122 => min_max::<2, i16>(bytes, n, i16::from_le_bytes),
        5123 => min_max::<2, u16>(bytes, n, u16::from_le_bytes),
        5125 => min_max::<4, u32>(bytes, n, u32::from_le_bytes),
        _ => min_max::<4, f32>(bytes, n, f32::from_le_bytes),
    };
    let fix = |v: &mut Vec<f64>| {
        for x in v.iter_mut().filter(|x| !x.is_finite()) {
            *x = 0.0;
        }
    };
    fix(&mut min);
    fix(&mut max);
    Some((min, max))
}

/// Each component's least and greatest over elements of `n` components of `CS` bytes, read
/// by `read`, compared as they're stored (a NaN is passed over) and given as `f64`: a loop of
/// its own for the common widths, so it runs without a branch per component. `bytes` holds
/// at least one element.
fn min_max<const CS: usize, T: Copy + PartialOrd + Into<f64>>(
    bytes: &[u8],
    n: usize,
    read: impl Fn([u8; CS]) -> T + Copy,
) -> (Vec<f64>, Vec<f64>) {
    fn fixed<const CS: usize, const N: usize, T: Copy + PartialOrd + Into<f64>>(
        bytes: &[u8],
        at: impl Fn(&[u8], usize) -> T,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut parts = bytes.chunks_exact(CS * N);
        let Some(first) = parts.next() else {
            return (Vec::new(), Vec::new());
        };
        let mut lo: [T; N] = std::array::from_fn(|c| at(first, c));
        let mut hi = lo;
        for e in parts {
            for c in 0..N {
                let x = at(e, c);
                // Selects, not branches: these compile to min and max instructions.
                lo[c] = if x < lo[c] { x } else { lo[c] };
                hi[c] = if x > hi[c] { x } else { hi[c] };
            }
        }
        (lo.map(Into::into).to_vec(), hi.map(Into::into).to_vec())
    }
    let at = |e: &[u8], c: usize| {
        let mut b = [0; CS];
        b.copy_from_slice(&e[c * CS..(c + 1) * CS]);
        read(b)
    };
    match n {
        1 => fixed::<CS, 1, T>(bytes, at),
        2 => fixed::<CS, 2, T>(bytes, at),
        3 => fixed::<CS, 3, T>(bytes, at),
        4 => fixed::<CS, 4, T>(bytes, at),
        _ => {
            let mut parts = bytes.chunks_exact(CS * n);
            let Some(first) = parts.next() else {
                return (Vec::new(), Vec::new());
            };
            let mut lo: Vec<T> = (0..n).map(|c| at(first, c)).collect();
            let mut hi = lo.clone();
            for e in parts {
                for c in 0..n {
                    let x = at(e, c);
                    lo[c] = if x < lo[c] { x } else { lo[c] };
                    hi[c] = if x > hi[c] { x } else { hi[c] };
                }
            }
            (
                lo.into_iter().map(Into::into).collect(),
                hi.into_iter().map(Into::into).collect(),
            )
        }
    }
}

fn bound_values(v: &[f64], ctype: u64) -> Value {
    #[allow(clippy::cast_possible_truncation, reason = "integer components' bounds are whole")]
    let ints = || v.iter().map(|&x| json!(x as i64)).collect();
    if ctype == 5126 {
        Value::Array(v.iter().map(|&x| json!(f32_of(x))).collect())
    } else {
        Value::Array(ints())
    }
}

/// Whether declared bounds match: floats as the `f32`s the data holds.
#[allow(clippy::float_cmp, reason = "bounds must match the data exactly")]
fn same_bounds(declared: Option<&Value>, want: &[f64], ctype: u64) -> bool {
    let Some(d) = declared.and_then(Value::as_array) else {
        return false;
    };
    d.len() == want.len()
        && d.iter().zip(want).all(|(d, &w)| {
            d.as_f64()
                .is_some_and(|d| if ctype == 5126 { f32_of(d) == f32_of(w) } else { d == w })
        })
}

/// Whether a buffer view in `doc` gives a stride, which animation and skin data mustn't have.
fn strided(json: &Map<String, Value>, view: Option<usize>) -> bool {
    view.and_then(|v| list(json, "bufferViews").get(v))
        .is_some_and(|v| v.get("byteStride").is_some())
}

/// Elements' lengths are checked against this when vertices are unit vectors.
fn off_unit(len: f64) -> bool {
    off_unit_sq(len * len)
}

/// [`off_unit`] from the squared length, without a square root.
fn off_unit_sq(sq: f64) -> bool {
    const LO: f64 = (1.0 - UNIT) * (1.0 - UNIT);
    const HI: f64 = (1.0 + UNIT) * (1.0 + UNIT);
    !(LO..=HI).contains(&sq)
}

impl Fix<'_, '_> {
    /// Puts `bytes` in the new buffer behind a new buffer view; returns the view.
    fn place(&mut self, bytes: &[u8], target: Option<u64>) -> usize {
        self.out.resize(self.out.len().next_multiple_of(4), 0);
        let at = self.out.len();
        self.out.extend_from_slice(bytes);
        let mut v = json!({ "buffer": self.buffer, "byteOffset": at, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        push(self.json, "bufferViews", v)
    }

    /// Points accessor `a` at new elements, tightly packed. Its bounds are recomputed when it
    /// declares them or `need_bounds`.
    #[allow(clippy::too_many_arguments, reason = "an accessor's every property")]
    fn rewrite(
        &mut self,
        a: usize,
        bytes: &[u8],
        ctype: u64,
        n: usize,
        count: usize,
        target: Option<u64>,
        need_bounds: bool,
    ) {
        if count == 0 {
            return;
        }
        let view = self.place(bytes, target);
        self.done.insert(a);
        let Some(acc) = accessor_mut(self.json, a) else { return };
        let declared = acc.contains_key("min") || acc.contains_key("max");
        acc.insert("bufferView".into(), json!(view));
        acc.insert("componentType".into(), json!(ctype));
        acc.insert("count".into(), json!(count));
        acc.remove("byteOffset");
        acc.remove("sparse");
        if (declared || need_bounds)
            && let Some((min, max)) = bounds(bytes, ctype, n)
        {
            acc.insert("min".into(), bound_values(&min, ctype));
            acc.insert("max".into(), bound_values(&max, ctype));
        }
    }

    /// Corrects accessor `a`'s bounds for elements `bytes` it holds unchanged: when they're
    /// declared but wrong, or `required` and missing.
    fn check_bounds(&mut self, a: usize, bytes: &[u8], ctype: u64, n: usize, required: bool) {
        let Some(acc) = accessor_mut(self.json, a) else { return };
        if acc.contains_key("sparse") {
            return;
        }
        let declared = acc.contains_key("min") || acc.contains_key("max");
        if !declared && !required {
            return;
        }
        if let Some((min, max)) = bounds(bytes, ctype, n) {
            self.set_bounds(a, &min, &max, ctype);
        }
    }

    /// Accessor `a`'s `min` and `max` made these, where they differ.
    fn set_bounds(&mut self, a: usize, min: &[f64], max: &[f64], ctype: u64) {
        let Some(acc) = accessor_mut(self.json, a) else { return };
        if same_bounds(acc.get("min"), min, ctype) && same_bounds(acc.get("max"), max, ctype) {
            return;
        }
        acc.insert("min".into(), bound_values(min, ctype));
        acc.insert("max".into(), bound_values(max, ctype));
    }

    /// Vertex attributes sharing a buffer view that gives no stride, which glTF forbids: each
    /// gets a view of its own on the same bytes.
    fn strides(&mut self, r: &mut Reader<'_>) {
        let json = &self.doc.json;
        let accessors = list(json, "accessors");
        let views = list(json, "bufferViews");
        let mut vertex = vec![false; accessors.len()];
        for (_, _, prim) in prims(json) {
            for (_, set) in attribute_sets(prim) {
                for a in set.values().filter_map(Value::as_u64).map(usize_of) {
                    if let Some(v) = vertex.get_mut(a) {
                        *v = true;
                    }
                }
            }
        }
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); views.len()];
        for (i, a) in accessors.iter().enumerate() {
            if let Some(u) = index(a, "bufferView").and_then(|v| users.get_mut(v)) {
                u.push(i);
            }
        }
        for (v, users) in users.iter().enumerate() {
            let view = &views[v];
            if users.len() < 2
                || view.get("byteStride").is_some()
                || view.get("extensions").is_some()
                || !users.iter().any(|&a| vertex[a])
            {
                continue;
            }
            let Ok((buffer, offset, len, _, _)) = r.view(v) else {
                continue;
            };
            for &a in users.iter().filter(|&&a| vertex[a]) {
                let Ok(acc) = r.acc(a) else { continue };
                let span = (acc.count * acc.esize()) as u64;
                if acc.count == 0 || acc.sparse.is_some() || acc.offset + span > len {
                    continue;
                }
                let nv = json!({
                    "buffer": buffer,
                    "byteOffset": offset + acc.offset,
                    "byteLength": span,
                    "target": 34962,
                });
                let i = push(self.json, "bufferViews", nv);
                if let Some(acc) = accessor_mut(self.json, a) {
                    acc.insert("bufferView".into(), json!(i));
                    acc.remove("byteOffset");
                }
            }
        }
    }

    /// Indices naming no vertex, dropped with what uses them. Returns the primitives left with
    /// nothing to draw.
    fn indices(&mut self, r: &mut Reader<'_>) -> BTreeSet<(usize, usize)> {
        let doc = self.doc;
        let accessors = list(&doc.json, "accessors");
        let mut drop = BTreeSet::new();
        // (indices, vertices, mode) → the new indices and mode, `None` for none left.
        let mut made: HashMap<(usize, usize, u64), Option<(usize, u64)>> = HashMap::new();
        let vertex_count = |prim: &Value| {
            let attrs = prim.get("attributes").and_then(Value::as_object);
            attrs
                .into_iter()
                .flat_map(|a| a.values())
                .filter_map(|a| accessors.get(usize_of(a.as_u64()?)))
                .filter_map(|a| a.get("count").and_then(Value::as_u64))
                .min()
        };
        // Each indexed primitive's (mesh, primitive, indices, vertices).
        let indexed: Vec<_> = prims(&doc.json)
            .into_iter()
            .filter_map(|(m, p, prim)| Some((m, p, index(prim, "indices")?, vertex_count(prim)?, prim)))
            .collect();
        // Which (indices, vertices) are fine (or unreadable), found together first: most are.
        let checks: BTreeSet<(usize, u64)> = indexed.iter().map(|x| (x.2, x.3)).collect();
        let checks: Vec<(usize, u64)> = checks.into_iter().collect();
        let keep = self.keep;
        let fits: Vec<_> = checks
            .iter()
            .map(|&(ia, vcount)| {
                let Ok(acc) = r.acc(ia) else { return (true, None) };
                if !matches!(acc.ctype, 5121 | 5123 | 5125) || acc.n != 1 {
                    return (true, None);
                }
                let Some((bytes, at)) = located(r, &acc) else {
                    return (true, None);
                };
                (indices_fit(&bytes, acc.ctype, vcount), kept(keep, bytes, at))
            })
            .collect();
        let mut fine = BTreeSet::new();
        for (check, (fit, k)) in checks.into_iter().zip(fits) {
            if fit {
                fine.insert(check);
            }
            self.kept.extend(k);
        }
        for (m, p, ia, vcount, prim) in indexed {
            if fine.contains(&(ia, vcount)) {
                continue;
            }
            let mode = super::mode(prim);
            let key = (ia, usize_of(vcount), mode);
            let result = if let Some(&r) = made.get(&key) {
                r
            } else {
                let Some(r) = self.filter(r, ia, vcount, mode) else {
                    continue;
                };
                made.insert(key, r);
                r
            };
            match result {
                None => {
                    drop.insert((m, p));
                }
                Some((a, mode)) => {
                    if let Some(prim) = primitive_mut(self.json, m, p) {
                        prim.insert("indices".into(), json!(a));
                        if mode == 4 {
                            prim.remove("mode");
                        } else {
                            prim.insert("mode".into(), json!(mode));
                        }
                    }
                }
            }
        }
        drop
    }

    /// Indices accessor `ia` for `vcount` vertices and `mode`, without those naming no vertex:
    /// `None` when they're all fine (or unreadable), else the new accessor and mode, or no
    /// accessor when nothing is left.
    #[allow(clippy::option_option, reason = "unchanged, and changed to nothing, differ")]
    fn filter(&mut self, r: &mut Reader<'_>, ia: usize, vcount: u64, mode: u64) -> Option<Option<(usize, u64)>> {
        let acc = r.acc(ia).ok()?;
        if !matches!(acc.ctype, 5121 | 5123 | 5125) || acc.n != 1 {
            return None;
        }
        let bytes = raw(r, &acc)?;
        if indices_fit(&bytes, acc.ctype, vcount) {
            return None;
        }
        let cs = csize(acc.ctype);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "indices are whole"
        )]
        let idx: Vec<u32> = bytes.chunks_exact(cs).map(|b| number(acc.ctype, b) as u32).collect();
        let ok = |i: u32| u64::from(i) < vcount;
        let (mut idx, mode) = match mode {
            0 => {
                let mut idx = idx;
                idx.retain(|&i| ok(i));
                (idx, 0)
            }
            1..=3 => {
                let mut idx = segments(mode, idx);
                retain_groups::<2>(&mut idx, |g| g.iter().all(|&i| ok(i)));
                (idx, 1)
            }
            4..=6 => {
                let mut idx = triangles(mode, idx);
                retain_groups::<3>(&mut idx, |g| g.iter().all(|&i| ok(i)));
                (idx, 4)
            }
            _ => return None,
        };
        if idx.is_empty() {
            return Some(None);
        }
        let top = idx.iter().copied().max().unwrap_or(0);
        let ctype = match acc.ctype {
            5121 if top < 0xFF => 5121,
            5121 | 5123 if top < 0xFFFF => 5123,
            _ => 5125,
        };
        let mut out = Vec::with_capacity(idx.len() * csize(ctype));
        #[allow(clippy::cast_possible_truncation, reason = "under the type's largest value")]
        for i in idx.drain(..) {
            match ctype {
                5121 => out.push(i as u8),
                5123 => out.extend_from_slice(&(i as u16).to_le_bytes()),
                _ => out.extend_from_slice(&i.to_le_bytes()),
            }
        }
        let template = list(&self.doc.json, "accessors").get(ia)?.clone();
        let a = push(self.json, "accessors", template);
        let count = out.len() / csize(ctype);
        self.rewrite(a, &out, ctype, 1, count, Some(34963), false);
        Some(Some((a, mode)))
    }

    /// Vertex attributes: non-finite floats, normals and tangents that aren't unit length,
    /// and bounds.
    fn vertices(&mut self, r: &mut Reader<'_>) {
        let doc = self.doc;
        let mut roles: BTreeMap<usize, Role> = BTreeMap::new();
        for (_, _, prim) in prims(&doc.json) {
            for (base, set) in attribute_sets(prim) {
                for (name, a) in set {
                    let Some(a) = a.as_u64().map(usize_of) else { continue };
                    let role = match name.as_str() {
                        "POSITION" => Role::Position,
                        "NORMAL" if base => Role::Normal,
                        "TANGENT" if base => Role::Tangent,
                        _ => Role::Other,
                    };
                    roles.entry(a).or_insert(role);
                }
            }
            if let Some(a) = index(prim, "indices") {
                roles.entry(a).or_insert(Role::Indices);
            }
        }
        for node in list(&doc.json, "nodes") {
            let instancing = node.get("extensions").and_then(|e| e.get("EXT_mesh_gpu_instancing"));
            let attrs = instancing.and_then(|e| e.get("attributes")).and_then(Value::as_object);
            for a in attrs.into_iter().flat_map(Map::values).filter_map(Value::as_u64) {
                roles.entry(usize_of(a)).or_insert(Role::Instance);
            }
        }
        let accessors = list(&doc.json, "accessors");
        let declared = |a: usize| {
            let x = accessors.get(a);
            x.is_some_and(|x| x.get("min").is_some() || x.get("max").is_some())
        };
        let mut jobs = Vec::new();
        for (a, role) in roles {
            if self.done.contains(&a) {
                continue;
            }
            let Ok(acc) = r.acc(a) else { continue };
            // Integers can't be non-finite, so without bounds to check there's nothing to read.
            if acc.ctype != 5126 && !declared(a) && role != Role::Position {
                continue;
            }
            jobs.push((a, role, acc.ctype));
        }
        // Valid data, most of it, is only read and scanned; what's to be put right is put
        // right after.
        let keep = self.keep;
        let scans: Vec<_> = jobs
            .iter()
            .map(|&(a, role, _)| {
                let Ok(acc) = r.acc(a) else { return (Scan::Skip, None) };
                let Some((bytes, at)) = located(r, &acc) else {
                    return (Scan::Skip, None);
                };
                let unit = unit_size(role, acc.n);
                if acc.ctype == 5126 && (!all_finite(&bytes) || unit.is_some_and(|n| !all_unit(&bytes, n))) {
                    return (Scan::Fix, None);
                }
                let sparse = accessors.get(a).is_some_and(|x| x.get("sparse").is_some());
                let checked = (role == Role::Position || declared(a)) && !sparse;
                let found = Scan::Fine(if checked {
                    bounds(&bytes, acc.ctype, acc.n)
                } else {
                    None
                });
                (found, kept(keep, bytes, at))
            })
            .collect();
        for (&(a, role, ctype), (scan, k)) in jobs.iter().zip(scans) {
            self.kept.extend(k);
            match scan {
                Scan::Skip | Scan::Fine(None) => {}
                Scan::Fine(Some((min, max))) => self.set_bounds(a, &min, &max, ctype),
                Scan::Fix => self.fix_vertices(r, a, role),
            }
        }
    }

    /// Vertex accessor `a`, used as `role`, with non-finite values made 0 and normals and
    /// tangents made unit length.
    fn fix_vertices(&mut self, r: &mut Reader<'_>, a: usize, role: Role) {
        let Ok(acc) = r.acc(a) else { return };
        let Some(bytes) = raw(r, &acc) else { return };
        let position = role == Role::Position;
        let mut f = f32s(&bytes);
        let mut changed = finite(&mut f);
        match unit_size(role, acc.n) {
            Some(3) => changed |= self.unit_normals(r, a, &mut f),
            Some(_) => changed |= unit_tangents(&mut f),
            None => {}
        }
        if changed {
            let target = (role != Role::Indices && role != Role::Instance).then_some(34962);
            self.rewrite(a, &bytes_of(&f), acc.ctype, acc.n, acc.count, target, position);
        } else {
            self.check_bounds(a, &bytes, acc.ctype, acc.n, position);
        }
    }

    /// Makes normals unit length; a zero one takes its faces' normal, or +Z. Whether any
    /// changed.
    fn unit_normals(&self, r: &mut Reader<'_>, a: usize, f: &mut [f32]) -> bool {
        let len = |v: &[f32]| (v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()).sqrt();
        if !f.chunks_exact(3).any(|v| off_unit(len(v))) {
            return false;
        }
        let mut faces: Option<Vec<[f64; 3]>> = None;
        for (k, v) in f.chunks_exact_mut(3).enumerate() {
            let l = len(v);
            if l > 1e-12 && l.is_finite() {
                if off_unit(l) {
                    for x in v.iter_mut() {
                        *x = f32_of(f64::from(*x) / l);
                    }
                }
                continue;
            }
            let faces = faces.get_or_insert_with(|| self.face_normals(r, a));
            let n = faces.get(k).copied().unwrap_or([0.0; 3]);
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = if l > 1e-30 && l.is_finite() {
                [n[0] / l, n[1] / l, n[2] / l]
            } else {
                [0.0, 0.0, 1.0]
            };
            for (x, y) in v.iter_mut().zip(n) {
                *x = f32_of(y);
            }
        }
        true
    }

    /// Each vertex's area-weighted face normal, over the triangles of every primitive using
    /// normals `a`.
    fn face_normals(&self, r: &mut Reader<'_>, a: usize) -> Vec<[f64; 3]> {
        let mut out: Vec<[f64; 3]> = Vec::new();
        for (_, _, prim) in prims(&self.doc.json) {
            let attrs = prim.get("attributes");
            let mode = super::mode(prim);
            if attrs.and_then(|x| index(x, "NORMAL")) != Some(a) || !(4..=6).contains(&mode) {
                continue;
            }
            let Some(pa) = attrs.and_then(|x| index(x, "POSITION")) else {
                continue;
            };
            let Ok(pos) = r.read::<3>(pa, [0.0; 3]) else { continue };
            let idx = match index(prim, "indices") {
                Some(i) => match r.read_indices(i) {
                    Ok(i) => i,
                    Err(_) => continue,
                },
                None => (0..crate::num::u32_of(pos.len())).collect(),
            };
            if out.len() < pos.len() {
                out.resize(pos.len(), [0.0; 3]);
            }
            for t in triangles(mode, idx).chunks_exact(3) {
                let [i, j, k] = [t[0], t[1], t[2]].map(|i| i as usize);
                let (Some(p), Some(q), Some(s)) = (pos.get(i), pos.get(j), pos.get(k)) else {
                    continue;
                };
                let d = |u: &[f32; 3], v: &[f32; 3]| [0, 1, 2].map(|c| f64::from(u[c]) - f64::from(v[c]));
                let (e, g) = (d(q, p), d(s, p));
                let n = [
                    e[1] * g[2] - e[2] * g[1],
                    e[2] * g[0] - e[0] * g[2],
                    e[0] * g[1] - e[1] * g[0],
                ];
                for v in [i, j, k] {
                    for c in 0..3 {
                        out[v][c] += n[c];
                    }
                }
            }
        }
        out
    }

    /// Skin weights: each vertex's, over all its sets, scaled to sum to 1.
    fn weights(&mut self, r: &mut Reader<'_>) {
        let doc = self.doc;
        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        for (_, _, prim) in prims(&doc.json) {
            let Some(attrs) = prim.get("attributes") else { continue };
            let sets: Vec<usize> = (0..).map_while(|k| index(attrs, &format!("WEIGHTS_{k}"))).collect();
            if sets.is_empty() || sets.iter().any(|a| self.done.contains(a)) || !seen.insert(sets.clone()) {
                continue;
            }
            let Some(accs) = sets.iter().map(|&a| r.acc(a).ok()).collect::<Option<Vec<Acc>>>() else {
                continue;
            };
            let first = &accs[0];
            if accs
                .iter()
                .any(|x| x.n != 4 || x.count != first.count || x.ctype != first.ctype)
            {
                continue;
            }
            let Some(data) = accs.iter().map(|x| raw(r, x)).collect::<Option<Vec<Cow<'_, [u8]>>>>() else {
                continue;
            };
            let fixed = match (first.ctype, first.normalized) {
                (5126, _) => normalize_float_weights(&data, first.count),
                (5121, true) => normalize_int_weights(&data, first.count, 1),
                (5123, true) => normalize_int_weights(&data, first.count, 2),
                _ => None,
            };
            let Some(fixed) = fixed else { continue };
            for ((&a, bytes), acc) in sets.iter().zip(fixed).zip(&accs) {
                self.rewrite(a, &bytes, acc.ctype, 4, acc.count, Some(34962), false);
            }
        }
    }

    /// Animation samplers: keys out of order or at the same time, interleaved data, and
    /// rotations that aren't unit quaternions.
    fn animations(&mut self, r: &mut Reader<'_>) {
        let doc = self.doc;
        // Each input's key order, once read: `None` for already in order.
        let mut inputs: HashMap<usize, Option<Option<Vec<usize>>>> = HashMap::new();
        let mut outputs: BTreeSet<usize> = BTreeSet::new();
        for anim in list(&doc.json, "animations") {
            let channels = items(anim, "channels");
            for (si, s) in items(anim, "samplers").iter().enumerate() {
                let (Some(ia), Some(oa)) = (index(s, "input"), index(s, "output")) else {
                    continue;
                };
                let cubic = s.get("interpolation").and_then(Value::as_str) == Some("CUBICSPLINE");
                let rotation = channels
                    .iter()
                    .find(|c| index(c, "sampler") == Some(si))
                    .and_then(|c| c.get("target")?.get("path")?.as_str())
                    == Some("rotation");
                let order = if let Some(o) = inputs.get(&ia) {
                    o.clone()
                } else {
                    let o = self.input(r, ia);
                    inputs.insert(ia, o.clone());
                    o
                };
                let Some(order) = order else { continue };
                if !outputs.insert(oa) || self.done.contains(&oa) {
                    continue;
                }
                let Ok(in_count) = r.acc(ia).map(|a| a.count) else {
                    continue;
                };
                self.output(r, oa, in_count, order.as_deref(), cubic, rotation);
            }
        }
    }

    /// Rewrites input `ia` as needed: returns the order its keys were put in (`None` if
    /// unchanged), or `None` if it couldn't be read.
    #[allow(clippy::option_option, reason = "unreadable, and in order, differ")]
    fn input(&mut self, r: &mut Reader<'_>, ia: usize) -> Option<Option<Vec<usize>>> {
        let acc = r.acc(ia).ok()?;
        if acc.ctype != 5126 || acc.n != 1 {
            return None;
        }
        let bytes = raw(r, &acc)?;
        let mut t = f32s(&bytes);
        let changed = finite(&mut t);
        let order = key_order(&t);
        if order.is_some() || changed || strided(&self.doc.json, acc.view) {
            let t: Vec<f32> = match &order {
                Some(o) => o.iter().map(|&k| t[k]).collect(),
                None => t,
            };
            self.rewrite(ia, &bytes_of(&t), 5126, 1, t.len(), None, true);
        } else {
            self.check_bounds(ia, &bytes, 5126, 1, true);
        }
        Some(order)
    }

    fn output(
        &mut self,
        r: &mut Reader<'_>,
        oa: usize,
        in_count: usize,
        order: Option<&[usize]>,
        cubic: bool,
        rotation: bool,
    ) {
        let Ok(acc) = r.acc(oa) else { return };
        if in_count == 0 || acc.count % in_count != 0 {
            return;
        }
        let Some(mut bytes) = raw(r, &acc) else { return };
        let per = acc.count / in_count;
        let es = acc.esize();
        let mut changed = false;
        if let Some(order) = order {
            let block = per * es;
            bytes = order
                .iter()
                .flat_map(|&k| &bytes[k * block..(k + 1) * block])
                .copied()
                .collect();
            changed = true;
        }
        if acc.ctype == 5126 {
            let mut f = f32s(&bytes);
            changed |= finite(&mut f);
            if rotation && acc.n == 4 {
                // A cubic spline's tangents aren't rotations: only each key's middle third is.
                let third = per / 3;
                for (e, q) in f.chunks_exact_mut(4).enumerate() {
                    let at = e % per;
                    if !cubic || (third..2 * third).contains(&at) {
                        changed |= unit_quaternion(q);
                    }
                }
            }
            if changed {
                bytes = Cow::Owned(bytes_of(&f));
            }
        }
        if changed || strided(&self.doc.json, acc.view) {
            let count = bytes.len() / es;
            self.rewrite(oa, &bytes, acc.ctype, acc.n, count, None, false);
        } else {
            self.check_bounds(oa, &bytes, acc.ctype, acc.n, false);
        }
    }

    /// Inverse bind matrices: interleaved, or not finite.
    fn skins(&mut self, r: &mut Reader<'_>) {
        let doc = self.doc;
        for skin in list(&doc.json, "skins") {
            let Some(a) = index(skin, "inverseBindMatrices") else {
                continue;
            };
            if self.done.contains(&a) {
                continue;
            }
            let Ok(acc) = r.acc(a) else { continue };
            if acc.ctype != 5126 {
                continue;
            }
            let Some(bytes) = raw(r, &acc) else { continue };
            let mut f = f32s(&bytes);
            if finite(&mut f) || strided(&doc.json, acc.view) {
                self.rewrite(a, &bytes_of(&f), 5126, acc.n, acc.count, None, false);
            } else {
                self.check_bounds(a, &bytes, 5126, acc.n, false);
            }
        }
    }

    /// Texture coordinate sets a primitive's material reads, and gaps in their numbering,
    /// filled by its first set, or zeros.
    fn texcoords(&mut self) {
        let doc = self.doc;
        let materials = list(&doc.json, "materials");
        let mut zeros: HashMap<u64, usize> = HashMap::new();
        for (m, p, prim) in prims(&doc.json) {
            let Some(attrs) = prim.get("attributes").and_then(Value::as_object) else {
                continue;
            };
            let need = index(prim, "material")
                .and_then(|i| materials.get(i))
                .map_or(0, uv_sets);
            let sets: BTreeMap<usize, &Value> = attrs
                .iter()
                .filter_map(|(k, v)| Some((k.strip_prefix("TEXCOORD_")?.parse::<usize>().ok()?, v)))
                .collect();
            let have = sets.keys().next_back().map_or(0, |&k| k + 1);
            let want = need.max(have).min(32);
            let missing: Vec<usize> = (0..want).filter(|k| !sets.contains_key(k)).collect();
            if missing.is_empty() {
                continue;
            }
            let source = if let Some(&v) = sets.values().next() {
                v.clone()
            } else {
                let accessors = list(&doc.json, "accessors");
                let Some(count) = index(prim.get("attributes").unwrap_or(&Value::Null), "POSITION")
                    .and_then(|a| accessors.get(a))
                    .and_then(|a| a.get("count")?.as_u64())
                else {
                    continue;
                };
                let a = if let Some(&a) = zeros.get(&count) {
                    a
                } else {
                    // An accessor without a view is zeros by the specification, but some readers (the `gltf` crate
                    // among them) refuse one that isn't sparse: the zeros are written out, unless there are more
                    // than the file's own data could hold positions for (a count that's only claimed).
                    let mut a = json!({ "componentType": 5126, "count": count, "type": "VEC2" });
                    let data: u64 = doc.buffers.iter().map(crate::document::Blob::len).sum();
                    let fits = |n: &u64| *n <= 1 << 26 && n * 8 <= data.max(1 << 20);
                    if let Some(len) = Some(count).filter(fits).and_then(|n| usize::try_from(n * 8).ok()) {
                        a["bufferView"] = json!(self.place(&vec![0; len], Some(34962)));
                    }
                    let a = push(self.json, "accessors", a);
                    zeros.insert(count, a);
                    a
                };
                json!(a)
            };
            if let Some(attrs) = primitive_mut(self.json, m, p)
                .and_then(|p| p.get_mut("attributes"))
                .and_then(Value::as_object_mut)
            {
                for k in missing {
                    attrs.insert(format!("TEXCOORD_{k}"), source.clone());
                }
            }
        }
    }

    /// Makes the node hierarchy a forest: a node listed as a child twice, or as its own
    /// ancestor, keeps only the first link that doesn't make a cycle. Scene roots that are
    /// children, or listed twice, are dropped.
    fn nodes(&mut self) {
        let Some(nodes) = self.json.get_mut("nodes").and_then(Value::as_array_mut) else {
            return;
        };
        let n = nodes.len();
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let ancestor = |parent: &[Option<usize>], c: usize, i: usize| {
            let mut x = Some(i);
            for _ in 0..=parent.len() {
                match x {
                    Some(y) if y == c => return true,
                    Some(y) => x = parent[y],
                    None => return false,
                }
            }
            true
        };
        for (i, node) in nodes.iter_mut().enumerate() {
            let Some(children) = node.get_mut("children").and_then(Value::as_array_mut) else {
                continue;
            };
            children.retain(|c| {
                let Some(c) = c.as_u64().map(usize_of).filter(|&c| c < n) else {
                    return false;
                };
                if parent[c].is_some() || ancestor(&parent, c, i) {
                    return false;
                }
                parent[c] = Some(i);
                true
            });
        }
        let scenes = self.json.get_mut("scenes").and_then(Value::as_array_mut);
        for s in scenes.into_iter().flatten() {
            let Some(roots) = s.get_mut("nodes").and_then(Value::as_array_mut) else {
                continue;
            };
            let mut seen = BTreeSet::new();
            roots.retain(|r| {
                r.as_u64()
                    .map(usize_of)
                    .is_some_and(|r| r < n && parent[r].is_none() && seen.insert(r))
            });
        }
    }

    /// Removes primitives left with nothing to draw (and meshes left with no primitives), and
    /// empty lists.
    fn prune(&mut self, drop: &BTreeSet<(usize, usize)>) {
        if let Some(meshes) = self.json.get_mut("meshes").and_then(Value::as_array_mut) {
            for &(m, p) in drop.iter().rev() {
                if let Some(prims) = meshes
                    .get_mut(m)
                    .and_then(|m| m.get_mut("primitives"))
                    .and_then(Value::as_array_mut)
                    && p < prims.len()
                {
                    prims.remove(p);
                }
            }
            let empty: Vec<bool> = meshes.iter().map(|m| items(m, "primitives").is_empty()).collect();
            if empty.iter().any(|&e| e) {
                let mut new = Vec::with_capacity(empty.len());
                let mut k = 0usize;
                for &e in &empty {
                    new.push((!e).then_some(k));
                    k += usize::from(!e);
                }
                let mut i = 0;
                meshes.retain(|_| {
                    i += 1;
                    !empty[i - 1]
                });
                let nodes = self.json.get_mut("nodes").and_then(Value::as_array_mut);
                for node in nodes.into_iter().flatten().filter_map(Value::as_object_mut) {
                    let Some(m) = node.get("mesh").and_then(Value::as_u64).map(usize_of) else {
                        continue;
                    };
                    match new.get(m).copied().flatten() {
                        Some(m) => {
                            node.insert("mesh".into(), json!(m));
                        }
                        None => {
                            for k in ["mesh", "skin", "weights"] {
                                node.remove(k);
                            }
                        }
                    }
                }
            }
        }
        let empty = |v: &Value| v.as_array().is_some_and(Vec::is_empty);
        let clear = |obj: &mut Value, keys: &[&str]| {
            if let Some(o) = obj.as_object_mut() {
                for k in keys {
                    if o.get(*k).is_some_and(empty) {
                        o.remove(*k);
                    }
                }
            }
        };
        for (key, inner) in [
            ("nodes", &["children", "weights"][..]),
            ("scenes", &["nodes"][..]),
            ("meshes", &["weights"][..]),
        ] {
            for v in self
                .json
                .get_mut(key)
                .and_then(Value::as_array_mut)
                .into_iter()
                .flatten()
            {
                clear(v, inner);
            }
        }
        let meshes = self.json.get_mut("meshes").and_then(Value::as_array_mut);
        for m in meshes.into_iter().flatten() {
            let prims = m.get_mut("primitives").and_then(Value::as_array_mut);
            for p in prims.into_iter().flatten() {
                clear(p, &["targets"]);
            }
        }
        self.json.retain(|k, v| {
            !(empty(v)
                && matches!(
                    k.as_str(),
                    "accessors"
                        | "animations"
                        | "bufferViews"
                        | "cameras"
                        | "images"
                        | "materials"
                        | "meshes"
                        | "nodes"
                        | "samplers"
                        | "scenes"
                        | "skins"
                        | "textures"
                        | "extensionsUsed"
                        | "extensionsRequired"
                ))
        });
    }
}

/// How many texture coordinate sets a material reads: one past the highest `texCoord` of its
/// textures (extensions' included), 0 for none.
fn uv_sets(v: &Value) -> usize {
    match v {
        Value::Object(o) => {
            let own = if o.get("index").is_some_and(Value::is_u64) || o.contains_key("texCoord") {
                o.get("texCoord")
                    .and_then(Value::as_u64)
                    .map_or(1, |t| usize_of(t).saturating_add(1))
            } else {
                0
            };
            o.iter()
                .filter(|(k, _)| k.as_str() != "extras")
                .map(|(_, v)| uv_sets(v))
                .fold(own, usize::max)
        }
        Value::Array(a) => a.iter().map(uv_sets).fold(0, usize::max),
        _ => 0,
    }
}

/// Makes tangents' directions unit length and their handedness ±1; whether any changed.
#[allow(clippy::float_cmp, reason = "handedness must be exactly ±1")]
fn unit_tangents(f: &mut [f32]) -> bool {
    let bad = |v: &[f32]| {
        let l = (v[..3].iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()).sqrt();
        off_unit(l) || (v[3] != 1.0 && v[3] != -1.0)
    };
    if !f.chunks_exact(4).any(bad) {
        return false;
    }
    for v in f.chunks_exact_mut(4) {
        let l = (v[..3].iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()).sqrt();
        if l > 1e-12 && l.is_finite() {
            if off_unit(l) {
                for x in &mut v[..3] {
                    *x = f32_of(f64::from(*x) / l);
                }
            }
        } else {
            v[..3].copy_from_slice(&[1.0, 0.0, 0.0]);
        }
        v[3] = if v[3] < 0.0 { -1.0 } else { 1.0 };
    }
    true
}

/// Makes a quaternion unit length (the identity for a zero one); whether it changed.
fn unit_quaternion(q: &mut [f32]) -> bool {
    let l = (q.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()).sqrt();
    if !off_unit(l) {
        return false;
    }
    if l > 1e-12 && l.is_finite() {
        for x in q.iter_mut() {
            *x = f32_of(f64::from(*x) / l);
        }
    } else {
        q.copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
    }
    true
}

/// The order to put animation keys in: by time, of keys at the same time only the last.
/// `None` when they're already strictly increasing.
fn key_order(t: &[f32]) -> Option<Vec<usize>> {
    if t.windows(2).all(|w| w[0] < w[1]) {
        return None;
    }
    let mut idx: Vec<usize> = (0..t.len()).collect();
    idx.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
    let mut out: Vec<usize> = Vec::with_capacity(idx.len());
    for k in idx {
        match out.last_mut() {
            Some(last) if t[*last] >= t[k] => *last = k,
            _ => out.push(k),
        }
    }
    Some(out)
}

/// Whether a vertex's weights sum to 1 as a validator checks it: in `f32`, allowing a rounding
/// for each weight that isn't 0.
fn sums_to_one(w: impl Iterator<Item = f32>) -> bool {
    let mut sum = 0.0f32;
    let mut nonzero = 0u16;
    for x in w {
        sum += x;
        nonzero += u16::from(x != 0.0);
    }
    f64::from((sum - 1.0).abs()) <= 2e-7 * f64::from(nonzero.max(1))
}

/// Float weight sets (each `count` VEC4s) scaled so each vertex's sum to 1, or `None` when
/// they already do. Negative and non-finite weights become 0; a vertex with none left is
/// given wholly to its first joint.
fn normalize_float_weights(sets: &[Cow<'_, [u8]>], count: usize) -> Option<Vec<Vec<u8>>> {
    let mut f: Vec<Vec<f32>> = sets.iter().map(|s| f32s(s)).collect();
    let vertex =
        |f: &[Vec<f32>], v: usize| -> Vec<f32> { f.iter().flat_map(|s| s[v * 4..v * 4 + 4].iter().copied()).collect() };
    let bad = (0..count).any(|v| {
        let w = vertex(&f, v);
        w.iter().any(|&x| !x.is_finite() || x < 0.0) || !sums_to_one(w.into_iter())
    });
    if !bad {
        return None;
    }
    for v in 0..count {
        let mut w: Vec<f32> = vertex(&f, v)
            .into_iter()
            .map(|x| if x.is_finite() && x > 0.0 { x } else { 0.0 })
            .collect();
        let sum: f64 = w.iter().map(|&x| f64::from(x)).sum();
        if sum > 0.0 {
            for x in &mut w {
                *x = f32_of(f64::from(*x) / sum);
            }
        } else {
            w.fill(0.0);
            w[0] = 1.0;
        }
        // What rounding leaves over goes to the heaviest weight.
        for _ in 0..4 {
            if sums_to_one(w.iter().copied()) {
                break;
            }
            let s: f32 = w.iter().sum();
            let top = (0..w.len()).fold(0, |b, k| if w[k] > w[b] { k } else { b });
            w[top] = (w[top] + (1.0 - s)).max(0.0);
        }
        for (k, x) in w.into_iter().enumerate() {
            f[k / 4][v * 4 + k % 4] = x;
        }
    }
    Some(f.iter().map(|s| bytes_of(s)).collect())
}

/// Normalized integer weight sets (`size` bytes a component) scaled so each vertex's sum to
/// the type's largest value, or `None` when they already do.
fn normalize_int_weights(sets: &[Cow<'_, [u8]>], count: usize, size: usize) -> Option<Vec<Vec<u8>>> {
    let full: u64 = if size == 1 { 0xFF } else { 0xFFFF };
    let get = |s: &[u8], k: usize| -> u64 {
        if size == 1 {
            u64::from(s[k])
        } else {
            u64::from(u16::from_le_bytes([s[2 * k], s[2 * k + 1]]))
        }
    };
    let vertex = |v: usize| -> Vec<u64> {
        sets.iter()
            .flat_map(|s| (0..4).map(move |c| get(s, v * 4 + c)))
            .collect()
    };
    if (0..count).all(|v| vertex(v).iter().sum::<u64>() == full) {
        return None;
    }
    let mut out: Vec<Vec<u8>> = sets.iter().map(|s| s.to_vec()).collect();
    for v in 0..count {
        let w = vertex(v);
        let sum: u64 = w.iter().sum();
        let mut n: Vec<u64> = if sum == 0 {
            let mut n = vec![0; w.len()];
            n[0] = full;
            n
        } else {
            w.iter().map(|&x| x * full / sum).collect()
        };
        let left = full - n.iter().sum::<u64>();
        let top = (0..n.len()).fold(0, |b, k| if n[k] > n[b] { k } else { b });
        n[top] += left;
        for (k, x) in n.into_iter().enumerate() {
            let (s, c) = (k / 4, v * 4 + k % 4);
            #[allow(clippy::cast_possible_truncation, reason = "at most the type's largest value")]
            if size == 1 {
                out[s][c] = x as u8;
            } else {
                out[s][2 * c..2 * c + 2].copy_from_slice(&(x as u16).to_le_bytes());
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document's buffer and the views and accessors on it, built up a piece at a time.
    #[derive(Default)]
    struct Build {
        bytes: Vec<u8>,
        views: Vec<Value>,
        accessors: Vec<Value>,
    }

    impl Build {
        /// Adds `data` as an accessor of `count` elements of `kind` on a view of its own;
        /// returns the accessor.
        fn add(&mut self, data: &[u8], ctype: u64, kind: &str, count: usize, extra: &Value) -> usize {
            self.bytes.resize(self.bytes.len().next_multiple_of(4), 0);
            let mut view = json!({ "buffer": 0, "byteOffset": self.bytes.len(), "byteLength": data.len() });
            self.bytes.extend_from_slice(data);
            if let Some(stride) = extra.get("byteStride") {
                view["byteStride"] = stride.clone();
            }
            self.views.push(view);
            let mut acc =
                json!({ "bufferView": self.views.len() - 1, "componentType": ctype, "count": count, "type": kind });
            for k in ["min", "max", "normalized"] {
                if let Some(v) = extra.get(k) {
                    acc[k] = v.clone();
                }
            }
            self.accessors.push(acc);
            self.accessors.len() - 1
        }

        fn floats(&mut self, f: &[f32], kind: &str, extra: &Value) -> usize {
            let n = match kind {
                "SCALAR" => 1,
                "VEC2" => 2,
                "VEC3" => 3,
                "VEC4" => 4,
                _ => 16,
            };
            self.add(&bytes_of(f), 5126, kind, f.len() / n, extra)
        }

        fn indices(&mut self, i: &[u16]) -> usize {
            let bytes: Vec<u8> = i.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.add(&bytes, 5123, "SCALAR", i.len(), &json!({}))
        }

        /// The document of `rest`, the JSON's other lists, over this buffer.
        fn doc(self, rest: Value) -> Document<'static> {
            let mut d = Document::empty(Map::new());
            d.push_buffer(self.bytes);
            d.json.insert("bufferViews".into(), json!(self.views));
            d.json.insert("accessors".into(), json!(self.accessors));
            if let Value::Object(o) = rest {
                d.json.extend(o);
            }
            d
        }
    }

    /// A triangle's corners, with their bounds.
    fn triangle(b: &mut Build) -> usize {
        let p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        b.floats(&p, "VEC3", &json!({ "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }))
    }

    fn values(d: &Document<'_>, a: usize) -> Vec<f32> {
        let mut r = Reader::bare(d);
        let acc = r.acc(a).unwrap();
        f32s(&raw(&mut r, &acc).unwrap())
    }

    fn repaired_doc(mut d: Document<'static>) -> Document<'static> {
        d.repair();
        d
    }

    #[test]
    fn a_valid_document_is_left_as_it_is() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let n = b.floats(&[0.0, 0.0, 1.0].repeat(3), "VEC3", &json!({}));
        let i = b.indices(&[0, 1, 2]);
        let d = b.doc(json!({
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": p, "NORMAL": n }, "indices": i }] }],
            "nodes": [{ "mesh": 0 }],
            "scenes": [{ "nodes": [0] }],
        }));
        let mut json = d.json.clone();
        assert!(repaired(&d, &mut json).is_none());
        assert_eq!(json, d.json);
    }

    #[test]
    fn instance_attributes_get_their_true_bounds() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let t = b.floats(
            &[-0.5, 0.0, 0.0, 1.0, 2.0, 0.0],
            "VEC3",
            &json!({ "min": [0.0, 0.0, 0.0], "max": [8.0, 8.0, 8.0] }),
        );
        let d = repaired_doc(b.doc(json!({
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": p } }] }],
            "nodes": [{ "mesh": 0, "extensions": { "EXT_mesh_gpu_instancing": { "attributes": { "TRANSLATION": t } } } }],
        })));
        assert_eq!(d.json["accessors"][t]["min"], json!([-0.5, 0.0, 0.0]));
        assert_eq!(d.json["accessors"][t]["max"], json!([1.0, 2.0, 0.0]));
    }

    #[test]
    fn animation_keys_are_sorted_keeping_the_last_at_one_time() {
        let mut b = Build::default();
        let input = b.floats(&[0.0, 2.0, 1.0, 1.0], "SCALAR", &json!({ "min": [0.0], "max": [2.0] }));
        let keys = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let output = b.floats(&keys, "VEC3", &json!({}));
        let d = repaired_doc(b.doc(json!({
            "nodes": [{}],
            "animations": [{
                "samplers": [{ "input": input, "output": output }],
                "channels": [{ "sampler": 0, "target": { "node": 0, "path": "translation" } }],
            }],
        })));
        assert_eq!(values(&d, input), [0.0, 1.0, 2.0]);
        assert_eq!(values(&d, output), [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        assert_eq!(d.json["accessors"][input]["max"], json!([2.0]));
    }

    #[test]
    fn inverse_bind_matrices_leave_a_strided_view() {
        let mut b = Build::default();
        let mut identity = [0.0f32; 16];
        identity[0] = 1.0;
        identity[5] = 1.0;
        identity[10] = 1.0;
        identity[15] = 1.0;
        let ibm = b.floats(&identity, "MAT4", &json!({ "byteStride": 64 }));
        let d = repaired_doc(b.doc(json!({
            "nodes": [{}],
            "skins": [{ "joints": [0], "inverseBindMatrices": ibm }],
        })));
        let view = usize_of(d.json["accessors"][ibm]["bufferView"].as_u64().unwrap());
        assert!(d.json["bufferViews"][view].get("byteStride").is_none());
        assert_eq!(values(&d, ibm), identity);
    }

    #[test]
    fn normals_are_made_unit_and_weights_sum_to_one() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let n = b.floats(&[0.0, 0.0, 2.0].repeat(3), "VEC3", &json!({}));
        let j = b.add(&[0; 12], 5121, "VEC4", 3, &json!({}));
        let w = b.floats(&[0.5; 12], "VEC4", &json!({}));
        let d = repaired_doc(b.doc(json!({
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": p, "NORMAL": n, "JOINTS_0": j, "WEIGHTS_0": w } }] }],
            "nodes": [{ "mesh": 0, "skin": 0 }, {}],
            "skins": [{ "joints": [1] }],
        })));
        assert_eq!(values(&d, n), [0.0, 0.0, 1.0].repeat(3));
        assert_eq!(values(&d, w), [0.25; 12]);
    }

    #[test]
    fn indices_naming_no_vertex_go_with_their_triangle() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let some = b.indices(&[0, 1, 2, 0, 1, 9]);
        let none = b.indices(&[7, 8, 9]);
        let d = repaired_doc(b.doc(json!({
            "meshes": [{ "primitives": [
                { "attributes": { "POSITION": p }, "indices": some },
                { "attributes": { "POSITION": p }, "indices": none },
            ] }],
        })));
        let prims = d.json["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 1);
        let i = usize_of(prims[0]["indices"].as_u64().unwrap());
        assert_eq!(d.json["accessors"][i]["count"], 3);
    }

    #[test]
    fn a_node_cycle_loses_the_link_that_closes_it() {
        let mut d = Build::default().doc(json!({
            "nodes": [{ "children": [1] }, { "children": [0] }],
            "scenes": [{ "nodes": [0, 1] }],
        }));
        d.repair();
        assert_eq!(d.json["nodes"][0]["children"], json!([1]));
        assert!(d.json["nodes"][1].get("children").is_none_or(|c| c == &json!([])));
        assert_eq!(d.json["scenes"][0]["nodes"], json!([0]));
    }

    #[test]
    fn a_texture_coordinate_set_a_material_reads_is_filled_in() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let uv = b.floats(&[0.0; 6], "VEC2", &json!({}));
        let d = repaired_doc(b.doc(json!({
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": p, "TEXCOORD_0": uv }, "material": 0 }] }],
            "materials": [{ "pbrMetallicRoughness": { "baseColorTexture": { "index": 0, "texCoord": 1 } } }],
            "textures": [{}],
        })));
        assert_eq!(
            d.json["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"],
            json!(uv)
        );
    }

    #[test]
    fn a_missing_texture_coordinate_set_is_written_out_as_zeros() {
        let mut b = Build::default();
        let p = triangle(&mut b);
        let d = repaired_doc(b.doc(json!({
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": p }, "material": 0 }] }],
            "materials": [{ "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } } }],
            "textures": [{}],
        })));
        let uv = index(&d.json["meshes"][0]["primitives"][0]["attributes"], "TEXCOORD_0").unwrap();
        let acc = &d.json["accessors"][uv];
        assert_eq!(acc["count"], 3);
        // Backed by a view, which readers that refuse a view-less accessor need.
        let view = index(acc, "bufferView").expect("a view");
        assert_eq!(d.json["bufferViews"][view]["byteLength"], 24);
    }

    #[test]
    fn a_claimed_vertex_count_writes_no_zeros_for_it() {
        let mut d = Build::default().doc(json!({
            "accessors": [{ "componentType": 5126, "count": 1_000_000_000_u64, "type": "VEC3" }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 }, "material": 0 }] }],
            "materials": [{ "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 } } }],
            "textures": [{}],
        }));
        d.repair();
        let uv = index(&d.json["meshes"][0]["primitives"][0]["attributes"], "TEXCOORD_0").unwrap();
        assert!(d.json["accessors"][uv].get("bufferView").is_none());
    }

    #[cfg(all(feature = "fs", feature = "gltf"))]
    #[test]
    fn what_repair_read_from_a_file_is_written_as_the_file_holds_it() {
        use crate::assets::{Folder, Memory};
        use crate::keep::Keep;
        // Big enough that repair keeps what it reads to write it again; wrong bounds and
        // normals too long, so the JSON changes and an extra buffer is written too.
        let n = 7998;
        let mut b = Build::default();
        let p: Vec<f32> = (0..n * 3)
            .map(|i| f32::from(u8::try_from(i % 97).unwrap()) * 0.5)
            .collect();
        let pa = b.floats(&p, "VEC3", &json!({ "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0] }));
        let normals: Vec<f32> = (0..n * 3).map(|i| if i % 3 == 2 { 2.0 } else { 0.0 }).collect();
        let na = b.floats(&normals, "VEC3", &json!({}));
        let idx: Vec<u16> = (0..u16::try_from(n).unwrap()).rev().collect();
        let ia = b.indices(&idx);
        let len = b.bytes.len();
        let gltf = json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": len }],
            "bufferViews": b.views,
            "accessors": b.accessors,
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": pa, "NORMAL": na }, "indices": ia }] }],
            "nodes": [{ "mesh": 0 }],
            "scenes": [{ "nodes": [0] }],
            "scene": 0,
        });
        let glb = crate::gltf::write_glb(&gltf, Some(&b.bytes)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.glb");
        std::fs::write(&path, &glb).unwrap();
        // Read lazily, however small: the binary chunk stays in the file.
        crate::formats::gltf::WHOLE_AT.set(0);
        let from_file = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL);
        crate::formats::gltf::WHOLE_AT.set(crate::formats::gltf::WHOLE);
        let from_file = from_file.unwrap();
        let memory = Memory::default();
        let in_memory = crate::formats::load_bytes(&glb, "t.glb", "glb", &memory, Keep::ALL).unwrap();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        from_file.write_glb(&mut a).unwrap();
        in_memory.write_glb(&mut b).unwrap();
        assert!(a == b, "the same GLB either way");
        let mut json = from_file.json.clone();
        let (extra, kept) = repaired_keeping(&from_file, &mut json);
        assert!(extra.is_some(), "the normals are rewritten");
        assert!(!kept.is_empty(), "what was read from the file is kept");
    }
}
