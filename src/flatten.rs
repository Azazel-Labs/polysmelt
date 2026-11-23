//! The flat view of a [`Document`]: the triangles its default scene draws, posed, as a
//! [`Mesh`], and how many triangles, points and lines that is, counted from the JSON alone.
//!
//! - The default scene is drawn (else scene 0; with no scenes, every root node; with no nodes,
//!   every mesh). Node transforms are baked into the vertices; a mirroring transform flips the
//!   winding back. `EXT_mesh_gpu_instancing` instances are baked the same way. A node
//!   `KHR_node_visibility` hides isn't drawn, nor anything under it.
//! - The pose is the nodes' own transforms and weights, or an animation's at a time
//!   ([`Pose`]): translation, rotation, scale and morph weights, with linear, step and
//!   cubic-spline interpolation.
//! - Skinned meshes take the pose their joints are in (linear blend skinning); morph targets
//!   take the node's or mesh's weights.
//! - Triangles, strips and fans; points and lines are counted, not drawn.
//! - `POSITION`, `NORMAL`, the color texture's `TEXCOORD_n` (its `KHR_texture_transform`
//!   baked in) and `COLOR_0`, in any component type `KHR_mesh_quantization` allows, sparse
//!   accessors included. Without `NORMAL` a primitive keeps no normals: glTF shades it flat.
//! - Materials: base color, metallic and roughness factors, emissive (with
//!   `KHR_materials_emissive_strength`), double-sidedness, and
//!   the color and normal textures (`KHR_materials_pbrSpecularGlossiness` as its diffuse), when
//!   their images are PNG or JPEG, or WebP (`EXT_texture_webp`) with the `webp` feature, which
//!   re-encodes it as PNG.
//! - Meshopt-compressed views are decoded as they are read; Draco
//!   (`KHR_draco_mesh_compression`) needs the `draco` feature.
//! - A required extension it can't honour is an error, unless it only changes how surfaces look.
//!
//! Accessors are read from a buffer in place when it's in memory, and a block at a time from a
//! file, so the peak is little more than the mesh.

mod decode;
mod rig;

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::document::{Counts, Document, Pose};
use crate::gltf as container;
use crate::mesh::{Image, MAX_TRIANGLES, Material, Mesh, Prim};
use crate::num::{f32_of, u32_of, usize_of};
use crate::{Context as _, Result};

/// How much of a buffer is read at once.
const BLOCK: usize = 256 * 1024;

/// The most vertices an accessor without data (all zeros) may claim.
const MAX_VERTICES: usize = 3 * MAX_TRIANGLES;

impl Document<'_> {
    /// The triangles the default scene draws in `pose`, as one mesh with every transform,
    /// instance, morph and skin baked in; `None` when it draws no triangles.
    ///
    /// # Errors
    ///
    /// When the document is malformed (an accessor out of range, a scene that doesn't exist),
    /// `pose` names an animation that doesn't exist, it uses Draco without the `draco`
    /// feature, or it draws more than [`MAX_TRIANGLES`].
    pub fn flatten(&self, pose: Pose) -> Result<Option<Mesh>> {
        let mut r = Reader::new(self)?;
        r.pose(pose)?;
        r.mesh()
    }

    /// How many triangles, points and line segments the default scene draws (every instance
    /// counted), from the JSON alone: nothing is read or decoded.
    #[must_use]
    pub fn counts(&self) -> Counts {
        counts(&self.json)
    }

    /// How many triangles the default scene draws: [`Document::counts`]' triangles.
    #[must_use]
    pub fn triangles(&self) -> usize {
        self.counts().triangles
    }
}

/// See [`Document::counts`]. A document whose default scene doesn't exist counts nothing.
fn counts(json: &Map<String, Value>) -> Counts {
    let mut c = Counts::default();
    let Ok(order) = scene_order(json) else {
        return c;
    };
    let meshes = list(json, "meshes");
    let accessors = list(json, "accessors");
    let count = |a: usize| accessors.get(a).and_then(|a| index(a, "count")).unwrap_or(0);
    let mut add = |mesh: &Value, k: usize| {
        for p in list(mesh.as_object().unwrap_or(&Map::new()), "primitives") {
            let Some(pos) = p.get("attributes").and_then(|a| index(a, "POSITION")) else {
                continue;
            };
            let n = index(p, "indices").map_or_else(|| count(pos), count);
            match p.get("mode").and_then(Value::as_u64).unwrap_or(4) {
                0 => c.points += n * k,
                1 => c.lines += n / 2 * k,
                2 => c.lines += n * k,
                3 => c.lines += n.saturating_sub(1) * k,
                4 => c.triangles += n / 3 * k,
                5 | 6 => c.triangles += n.saturating_sub(2) * k,
                _ => {}
            }
        }
    };
    let Some(order) = order else {
        for m in meshes {
            add(m, 1);
        }
        return c;
    };
    let nodes = list(json, "nodes");
    // A skinned mesh is drawn once per skin and weights, wherever its nodes are.
    let mut skinned: Vec<(usize, usize, Vec<u32>)> = Vec::new();
    for i in order {
        let n = &nodes[i];
        let Some(m) = index(n, "mesh") else { continue };
        let Some(mesh) = meshes.get(m) else { continue };
        let k = if let Some(skin) = index(n, "skin") {
            let w: Vec<u32> = default_weights(n, mesh).iter().map(|x| x.to_bits()).collect();
            if skinned.iter().any(|s| s.0 == m && s.1 == skin && s.2 == w) {
                continue;
            }
            skinned.push((m, skin, w));
            1
        } else {
            match instancing(n) {
                None => 1,
                Some(attrs) => ["TRANSLATION", "ROTATION", "SCALE"]
                    .iter()
                    .filter_map(|k| index(attrs, k))
                    .map(count)
                    .min()
                    .unwrap_or(0),
            }
        };
        add(mesh, k);
    }
    c
}

/// A node's `EXT_mesh_gpu_instancing` attributes.
fn instancing(node: &Value) -> Option<&Value> {
    node.get("extensions")?
        .get("EXT_mesh_gpu_instancing")?
        .get("attributes")
}

/// A node's morph weights: its own, else its mesh's.
fn default_weights(node: &Value, mesh: &Value) -> Vec<f32> {
    node.get("weights")
        .or_else(|| mesh.get("weights"))
        .and_then(Value::as_array)
        .map(|w| w.iter().map(|x| f32_of(x.as_f64().unwrap_or(0.0))).collect())
        .unwrap_or_default()
}

/// Whether each node is some node's child.
fn children(nodes: &[Value]) -> Vec<bool> {
    let mut child = vec![false; nodes.len()];
    for n in nodes {
        for c in list_of(n, "children") {
            if let Some(f) = child.get_mut(c) {
                *f = true;
            }
        }
    }
    child
}

/// The nodes the default scene draws, parents before children, each once, leaving out those
/// `KHR_node_visibility` hides and everything under them; `None` for a
/// document without nodes, which draws every mesh as it is.
fn scene_order(json: &Map<String, Value>) -> Result<Option<Vec<usize>>> {
    let nodes = list(json, "nodes");
    let scenes = list(json, "scenes");
    let roots: Vec<usize> = if scenes.is_empty() {
        if nodes.is_empty() {
            return Ok(None);
        }
        let child = children(nodes);
        (0..nodes.len()).filter(|&i| !child[i]).collect()
    } else {
        let s = json.get("scene").and_then(Value::as_u64).map_or(0, usize_of);
        list_of(scenes.get(s).context("glTF default scene doesn't exist")?, "nodes")
    };
    let mut seen = vec![false; nodes.len()];
    let mut order = Vec::new();
    let mut stack: Vec<usize> = roots.into_iter().rev().collect();
    while let Some(i) = stack.pop() {
        if i >= nodes.len() || seen[i] {
            continue;
        }
        seen[i] = true;
        // KHR_node_visibility: a hidden node hides everything under it.
        if nodes[i].pointer("/extensions/KHR_node_visibility/visible") == Some(&Value::Bool(false)) {
            continue;
        }
        order.push(i);
        stack.extend(list_of(&nodes[i], "children").into_iter().rev());
    }
    Ok(Some(order))
}

/// An accessor, resolved.
#[derive(Clone)]
struct Acc {
    view: Option<usize>,
    offset: u64,
    count: usize,
    ctype: u64,
    /// Components per element.
    n: usize,
    normalized: bool,
    sparse: Option<Value>,
    /// Elements decoded from Draco, tightly packed, in place of a buffer view.
    decoded: Option<Arc<[u8]>>,
}

impl Acc {
    fn csize(&self) -> usize {
        match self.ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4,
        }
    }

    fn esize(&self) -> usize {
        self.csize() * self.n
    }
}

/// A 4x4 matrix, column-major as glTF writes it.
type M4 = [f64; 16];

const IDENTITY: M4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Where one mesh is drawn: its transforms (several for instances), morph weights and skin.
struct Use {
    mesh: usize,
    weights: Vec<f32>,
    skin: Option<usize>,
    transforms: Vec<M4>,
}

struct Reader<'d> {
    doc: &'d Document<'d>,
    json: &'d Map<String, Value>,
    scratch: Vec<u8>,
    /// Meshopt-compressed buffer views, decoded once read.
    meshopt: HashMap<usize, Vec<u8>>,
    /// glTF material → mesh material.
    materials: Vec<Option<usize>>,
    /// The material for primitives that name none.
    default_material: Option<usize>,
    /// glTF image → mesh image, once tried: `None` until then, `Some(None)` if it didn't embed.
    #[allow(clippy::option_option, reason = "untried and tried-but-failed are both cached")]
    images: Vec<Option<Option<usize>>>,
    /// Every node's world transform, for joints.
    world: Vec<M4>,
    /// Each node's animated properties, in the pose asked for.
    posed: Vec<Posed>,
    /// Triangles still allowed.
    budget: usize,
    /// The current primitive's Draco-decoded accessors: (accessor, elements, component type,
    /// components).
    draco: Vec<(usize, Arc<[u8]>, u64, usize)>,
}

/// A node's properties an animation sets.
#[derive(Default, Clone)]
struct Posed {
    translation: Option<[f64; 3]>,
    rotation: Option<[f64; 4]>,
    scale: Option<[f64; 3]>,
    weights: Option<Vec<f32>>,
}

fn list<'v>(obj: &'v Map<String, Value>, key: &str) -> &'v [Value] {
    obj.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// `v[key]` as a list.
fn items<'v>(v: &'v Value, key: &str) -> &'v [Value] {
    v.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn index(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(usize_of)
}

fn floats<const N: usize>(v: Option<&Value>, default: [f64; N]) -> [f64; N] {
    let mut out = default;
    if let Some(a) = v.and_then(Value::as_array).filter(|a| a.len() == N) {
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_f64().unwrap_or(*o);
        }
    }
    out
}

/// Required extensions a flat view draws: the geometry ones it decodes, and any that only change how surfaces
/// look (materials, textures, lights), which it shows as best it can.
fn drawable(extension: &str) -> bool {
    const DECODED: &[&str] = &[
        "KHR_draco_mesh_compression",
        "EXT_meshopt_compression",
        "KHR_meshopt_compression",
        "KHR_mesh_quantization",
        "EXT_mesh_gpu_instancing",
        "KHR_animation_pointer",
        "KHR_node_visibility",
    ];
    const LOOKS: &[&str] = &[
        "KHR_materials_",
        "KHR_texture_",
        "EXT_texture_",
        "KHR_lights_",
        "KHR_xmp",
    ];
    DECODED.contains(&extension) || LOOKS.iter().any(|p| extension.starts_with(p))
}

fn f32s<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    floats(v, default.map(f64::from)).map(f32_of)
}

impl<'d> Reader<'d> {
    #[allow(clippy::unnecessary_wraps, reason = "it fails without the draco feature")]
    fn new(doc: &'d Document<'d>) -> Result<Self> {
        let json = &doc.json;
        #[cfg(not(feature = "draco"))]
        ensure!(
            !list(json, "extensionsRequired")
                .iter()
                .any(|e| e.as_str() == Some("KHR_draco_mesh_compression")),
            "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
        );
        if let Some(e) = list(json, "extensionsRequired")
            .iter()
            .filter_map(Value::as_str)
            .find(|e| !drawable(e))
        {
            bail!("glTF requires the `{e}` extension, which polysmelt can't draw");
        }
        Ok(Self::bare(doc))
    }

    /// A reader that only reads accessors: none of the checks drawing needs.
    fn bare(doc: &'d Document<'d>) -> Self {
        let json = &doc.json;
        let n = |key| list(json, key).len();
        Self {
            doc,
            json,
            scratch: Vec::new(),
            meshopt: HashMap::new(),
            materials: vec![None; n("materials")],
            default_material: None,
            images: vec![None; n("images")],
            world: Vec::new(),
            posed: vec![Posed::default(); n("nodes")],
            budget: MAX_TRIANGLES,
            draco: Vec::new(),
        }
    }

    fn get(&self, key: &str, i: usize) -> Result<&'d Value> {
        list(self.json, key)
            .get(i)
            .with_context(|| format!("glTF {key}[{i}] doesn't exist"))
    }

    /// A buffer view as (buffer, offset, length, stride).
    fn view(&self, i: usize) -> Result<(usize, u64, u64, Option<usize>)> {
        let v = self.get("bufferViews", i)?;
        let buffer = index(v, "buffer").context("glTF buffer view has no buffer")?;
        let offset = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0);
        let len = v
            .get("byteLength")
            .and_then(Value::as_u64)
            .context("glTF buffer view has no byteLength")?;
        Ok((buffer, offset, len, index(v, "byteStride")))
    }

    /// Whether buffer view `i` is meshopt-compressed, decoding it (once) if so.
    fn meshopt(&mut self, i: usize) -> Result<bool> {
        if self.meshopt.contains_key(&i) {
            return Ok(true);
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
            .with_context(|| format!("glTF buffer view {i}: {name} names no buffer"))?;
        let off = ext.get("byteOffset").and_then(Value::as_u64).unwrap_or(0);
        let src_len = ext
            .get("byteLength")
            .and_then(Value::as_u64)
            .with_context(|| format!("glTF buffer view {i}: {name} has no byteLength"))?;
        let blob = self
            .doc
            .buffers
            .get(b)
            .with_context(|| format!("glTF buffer view {i}: {name} names missing buffer {b}"))?;
        let held;
        let src = if let Some(s) = blob.slice(off, src_len) {
            s
        } else {
            ensure!(
                off.saturating_add(src_len) <= blob.len(),
                "glTF buffer view {i}: {name} data out of range"
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
            "glTF buffer view {i}: {name} data is truncated"
        );
        let mut out = vec![0; len];
        container::decode_view(ext, src, &mut out).with_context(|| format!("glTF buffer view {i}"))?;
        self.meshopt.insert(i, out);
        Ok(true)
    }

    /// A buffer view's bytes.
    fn view_bytes(&mut self, i: usize) -> Result<Vec<u8>> {
        if self.meshopt(i)? {
            return Ok(self.meshopt[&i].clone());
        }
        let (buffer, offset, len, _) = self.view(i)?;
        let b = self
            .doc
            .buffers
            .get(buffer)
            .context("glTF buffer view names a missing buffer")?;
        ensure!(
            offset.saturating_add(len) <= b.len(),
            "glTF buffer view runs past its buffer"
        );
        if let Some(s) = b.slice(offset, len) {
            return Ok(s.to_vec());
        }
        let mut out = vec![0; usize::try_from(len).context("glTF buffer view is too big to hold")?];
        b.read_at(offset, &mut out)?;
        Ok(out)
    }

    fn acc(&self, i: usize) -> Result<Acc> {
        if let Some((_, bytes, ctype, n)) = self.draco.iter().find(|d| d.0 == i) {
            let count = bytes.len() / (n * 4);
            return Ok(Acc {
                view: None,
                offset: 0,
                count,
                ctype: *ctype,
                n: *n,
                normalized: false,
                sparse: None,
                decoded: Some(bytes.clone()),
            });
        }
        let a = self.get("accessors", i)?;
        let ctype = a.get("componentType").and_then(Value::as_u64).unwrap_or(0);
        ensure!(
            matches!(ctype, 5120..=5123 | 5125 | 5126),
            "glTF accessor {i} has component type {ctype}"
        );
        let n = match a.get("type").and_then(Value::as_str) {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            t => bail!("glTF accessor {i} has type {t:?}, not one read here"),
        };
        Ok(Acc {
            view: index(a, "bufferView"),
            offset: a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0),
            count: index(a, "count").context("glTF accessor has no count")?,
            ctype,
            n,
            normalized: a.get("normalized").and_then(Value::as_bool).unwrap_or(false),
            sparse: a.get("sparse").cloned(),
            decoded: None,
        })
    }

    /// Calls `f(first, bytes, stride)` over an accessor's elements a block at a time: element
    /// `first + j` is at `bytes[j * stride..]`. Nothing for an accessor without a buffer view.
    fn blocks(&mut self, a: &Acc, mut f: impl FnMut(usize, &[u8], usize)) -> Result<()> {
        if let Some(bytes) = &a.decoded {
            f(0, bytes, a.esize());
            return Ok(());
        }
        let Some(view) = a.view else {
            ensure!(a.count <= MAX_VERTICES, "glTF accessor counts {} elements", a.count);
            return Ok(());
        };
        if a.count == 0 {
            return Ok(());
        }
        let (buffer, voff, vlen, stride) = self.view(view)?;
        let esize = a.esize();
        let stride = stride.filter(|&s| s > 0).unwrap_or(esize);
        ensure!(
            stride >= esize,
            "glTF buffer view stride {stride} is under an element's {esize} bytes"
        );
        // The elements' bytes, from the first: the accessor's offset into its view is checked here, not counted twice.
        let span = (a.count as u64 - 1)
            .checked_mul(stride as u64)
            .and_then(|s| s.checked_add(esize as u64))
            .filter(|&s| a.offset.checked_add(s).is_some_and(|end| end <= vlen))
            .with_context(|| format!("glTF accessor's {} elements run past their buffer view", a.count))?;
        if self.meshopt(view)? {
            let bytes = &self.meshopt[&view];
            let s = bytes
                .get(usize_of(a.offset)..usize_of(a.offset + span))
                .context("glTF accessor runs past its decoded buffer view")?;
            f(0, s, stride);
            return Ok(());
        }
        let start = voff + a.offset;
        let b = self
            .doc
            .buffers
            .get(buffer)
            .context("glTF buffer view names a missing buffer")?;
        ensure!(
            start + span <= b.len(),
            "glTF accessor runs past its buffer: bytes {start}..{} of buffer {buffer}, which holds {}",
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
    fn read<const N: usize>(&mut self, i: usize, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let a = self.acc(i)?;
        self.read_acc(&a, fill)
    }

    fn read_acc<const N: usize>(&mut self, a: &Acc, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let mut zero = fill;
        for z in zero.iter_mut().take(a.n) {
            *z = 0.0;
        }
        if a.view.is_none() {
            ensure!(a.count <= MAX_VERTICES, "glTF accessor counts {} elements", a.count);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(a.count)
            .map_err(|_| invalid!("glTF accessor counts {} elements", a.count))?;
        out.resize(a.count, zero);
        let m = a.n.min(N);
        let cs = a.csize();
        let (ctype, normalized) = (a.ctype, a.normalized);
        self.blocks(a, |first, bytes, stride| {
            let out = &mut out[first..];
            if ctype == 5126 && m == N && stride == 4 * N {
                // Tightly packed floats (what nearly every exporter writes): one flat loop.
                for (o, b) in out.as_flattened_mut().iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                return;
            }
            match (ctype, normalized) {
                (5126, _) => put(out, bytes, stride, cs, m, |b| {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                }),
                (5121, true) => put(out, bytes, stride, cs, m, |b| f32::from(b[0]) / 255.0),
                (5121, false) => put(out, bytes, stride, cs, m, |b| f32::from(b[0])),
                (5120, true) => put(out, bytes, stride, cs, m, |b| {
                    (f32::from(b[0].cast_signed()) / 127.0).max(-1.0)
                }),
                (5120, false) => put(out, bytes, stride, cs, m, |b| f32::from(b[0].cast_signed())),
                (5123, true) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0
                }),
                (5123, false) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]]))
                }),
                (5122, true) => put(out, bytes, stride, cs, m, |b| {
                    (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0)
                }),
                (5122, false) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(i16::from_le_bytes([b[0], b[1]]))
                }),
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "past 2^24 an f32 attribute rounds, as in any viewer"
                )]
                _ => put(out, bytes, stride, cs, m, |b| {
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32
                }),
            }
        })?;
        if let Some(s) = a.sparse.clone() {
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
    fn read_indices(&mut self, i: usize) -> Result<Vec<u32>> {
        let a = self.acc(i)?;
        ensure!(
            a.n == 1 && matches!(a.ctype, 5121 | 5123 | 5125),
            "glTF index accessor {i} isn't unsigned integer scalars"
        );
        self.read_u32s(&a)
    }

    fn read_u32s(&mut self, a: &Acc) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        out.try_reserve_exact(a.count)
            .map_err(|_| invalid!("glTF accessor counts {} elements", a.count))?;
        out.resize(a.count, 0u32);
        let ctype = a.ctype;
        self.blocks(a, |first, bytes, stride| {
            let out = &mut out[first..];
            match ctype {
                5121 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from(e[0]);
                    }
                }
                5123 if stride == 2 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                        *o = u32::from(u16::from_le_bytes([e[0], e[1]]));
                    }
                }
                5125 if stride == 4 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                        *o = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    }
                }
                5123 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from(u16::from_le_bytes([e[0], e[1]]));
                    }
                }
                _ => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    }
                }
            }
        })?;
        if let Some(s) = a.sparse.clone() {
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
    fn sparse(&mut self, a: &Acc, s: &Value) -> Result<(Vec<u32>, Acc)> {
        let count = index(s, "count").context("glTF sparse accessor has no count")?;
        ensure!(
            count <= a.count,
            "glTF sparse accessor replaces more elements than it has"
        );
        let part = |key: &str| -> Result<(usize, u64)> {
            let p = s
                .get(key)
                .with_context(|| format!("glTF sparse accessor has no {key}"))?;
            Ok((
                index(p, "bufferView").with_context(|| format!("glTF sparse {key} has no bufferView"))?,
                p.get("byteOffset").and_then(Value::as_u64).unwrap_or(0),
            ))
        };
        let (iv, ioff) = part("indices")?;
        let ictype = s["indices"].get("componentType").and_then(Value::as_u64).unwrap_or(0);
        ensure!(
            matches!(ictype, 5121 | 5123 | 5125),
            "glTF sparse indices' component type {ictype}"
        );
        let idx = Acc {
            view: Some(iv),
            offset: ioff,
            count,
            ctype: ictype,
            n: 1,
            normalized: false,
            sparse: None,
            decoded: None,
        };
        let at = self.read_u32s(&idx)?;
        let (vv, voff) = part("values")?;
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

    /// The scene's meshes, where they are drawn.
    fn uses(&mut self) -> Result<Vec<Use>> {
        let nodes = list(self.json, "nodes");
        let mut world = vec![None::<M4>; nodes.len()];
        // Parents first: every node reachable from a root takes its parent's transform.
        let child = children(nodes);
        let mut stack: Vec<(usize, M4)> = (0..nodes.len()).filter(|&i| !child[i]).map(|i| (i, IDENTITY)).collect();
        while let Some((i, parent)) = stack.pop() {
            if world[i].is_some() {
                continue;
            }
            let m = mul(&parent, &self.local(i, &nodes[i]));
            world[i] = Some(m);
            for c in list_of(&nodes[i], "children") {
                if c < nodes.len() && world[c].is_none() {
                    stack.push((c, m));
                }
            }
        }
        let Some(order) = scene_order(self.json)? else {
            // No nodes at all: every mesh, as it is.
            return Ok((0..list(self.json, "meshes").len())
                .map(|mesh| Use {
                    mesh,
                    weights: Vec::new(),
                    skin: None,
                    transforms: vec![IDENTITY],
                })
                .collect());
        };

        let mut uses: Vec<Use> = Vec::new();
        for i in order {
            let n = &nodes[i];
            let Some(mesh) = index(n, "mesh") else { continue };
            let skin = index(n, "skin");
            let weights = match &self.posed[i].weights {
                Some(w) => w.clone(),
                None => default_weights(n, list(self.json, "meshes").get(mesh).unwrap_or(&Value::Null)),
            };
            let w = world[i].unwrap_or(IDENTITY);
            let transforms = if skin.is_some() {
                // A skinned mesh is placed by its joints, not its node.
                vec![IDENTITY]
            } else {
                self.instances(n, &w)?
            };
            let same = uses.iter_mut().find(|u| {
                u.mesh == mesh
                    && u.skin == skin
                    && u.weights
                        .iter()
                        .map(|x| x.to_bits())
                        .eq(weights.iter().map(|x| x.to_bits()))
            });
            match same {
                Some(_) if skin.is_some() => {}
                Some(u) => u.transforms.extend(transforms),
                None => uses.push(Use {
                    mesh,
                    weights,
                    skin,
                    transforms,
                }),
            }
        }
        // Joints need every node's transform, in the scene or not.
        self.world = world.into_iter().map(|w| w.unwrap_or(IDENTITY)).collect();
        Ok(uses)
    }

    /// A node's transforms: its own, or one per `EXT_mesh_gpu_instancing` instance.
    fn instances(&mut self, node: &Value, world: &M4) -> Result<Vec<M4>> {
        let Some(attrs) = instancing(node) else {
            return Ok(vec![*world]);
        };
        let get = |k: &str| index(attrs, k);
        let t = get("TRANSLATION").map(|a| self.read(a, [0.0; 3])).transpose()?;
        let r = get("ROTATION")
            .map(|a| self.read(a, [0.0, 0.0, 0.0, 1.0]))
            .transpose()?;
        let s = get("SCALE").map(|a| self.read(a, [1.0; 3])).transpose()?;
        let count = [
            t.as_ref().map(Vec::len),
            r.as_ref().map(Vec::len),
            s.as_ref().map(Vec::len),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(0);
        ensure!(count <= MAX_TRIANGLES, "glTF node has {count} instances");
        Ok((0..count)
            .map(|k| {
                let t = t.as_ref().map_or([0.0; 3], |v| v[k].map(f64::from));
                let r = r.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |v| v[k].map(f64::from));
                let s = s.as_ref().map_or([1.0; 3], |v| v[k].map(f64::from));
                mul(world, &trs(t, r, s))
            })
            .collect())
    }
}

/// `node[key]` as indices.
fn list_of(v: &Value, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(usize_of).collect())
        .unwrap_or_default()
}

/// Writes `m` components of each element in `bytes` into `out`.
#[inline]
fn put<const N: usize>(
    out: &mut [[f32; N]],
    bytes: &[u8],
    stride: usize,
    csize: usize,
    m: usize,
    conv: impl Fn(&[u8]) -> f32,
) {
    if m == N {
        // Every component given (the usual case): a loop the compiler unrolls.
        for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
            if e.len() >= N * csize {
                *o = std::array::from_fn(|c| conv(&e[c * csize..]));
            }
        }
        return;
    }
    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
        for (c, x) in o.iter_mut().enumerate().take(m) {
            *x = conv(&e[c * csize..]);
        }
    }
}

fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [0.0; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}

/// Translation, rotation (a quaternion x y z w) and scale as one matrix.
fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [x, y, z, w] = if len > 0.0 {
        q.map(|c| c / len)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut m = IDENTITY;
    for col in 0..3 {
        for row in 0..3 {
            m[col * 4 + row] = r[row][col] * s[col];
        }
    }
    m[12..15].copy_from_slice(&t);
    m
}

/// The upper 3x3's determinant.
fn det3(m: &M4) -> f64 {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2])
}

/// The matrix normals go through: the upper 3x3's cofactors (its inverse transpose, scaled),
/// signed so a mirror doesn't turn normals inward. Row-major.
fn normal_matrix(m: &M4) -> [[f64; 3]; 3] {
    let a = |r: usize, c: usize| m[c * 4 + r];
    let sign = if det3(m) < 0.0 { -1.0 } else { 1.0 };
    let mut n = [[0.0; 3]; 3];
    for (r, row) in n.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
            let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
            *x = sign * (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1));
        }
    }
    n
}

fn apply_point(m: &M4, p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p.map(f64::from);
    [0, 1, 2].map(|r| f32_of(m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]))
}

fn apply_normal(n: &[[f64; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v.map(f64::from);
    let o = [0, 1, 2].map(|r| n[r][0] * x + n[r][1] * y + n[r][2] * z);
    let len = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    if len > 0.0 {
        o.map(|c| f32_of(c / len))
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn normalize(v: &mut [f32; 3]) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        for c in v.iter_mut() {
            *c /= len;
        }
    } else {
        *v = [0.0, 0.0, 1.0];
    }
}

/// One primitive's geometry before it is placed.
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// Triangle lists from a primitive's indices, by mode (4 triangles, 5 strip, 6 fan), per the
/// glTF 2.0 specification's section 3.7.2.1.
fn triangles(mode: u64, idx: Vec<u32>) -> Vec<u32> {
    match mode {
        5 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                if i % 2 == 0 {
                    out.extend_from_slice(&[idx[i], idx[i + 1], idx[i + 2]]);
                } else {
                    out.extend_from_slice(&[idx[i], idx[i + 2], idx[i + 1]]);
                }
            }
            out
        }
        6 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                out.extend_from_slice(&[idx[i + 1], idx[i + 2], idx[0]]);
            }
            out
        }
        _ => {
            let mut idx = idx;
            idx.truncate(idx.len() / 3 * 3);
            idx
        }
    }
}

impl Reader<'_> {
    fn mesh(&mut self) -> Result<Option<Mesh>> {
        let uses = self.uses()?;
        let mut mesh = Mesh::default();
        for u in &uses {
            for p in items(self.get("meshes", u.mesh)?, "primitives") {
                self.primitive(p, u, &mut mesh)?;
            }
        }
        Ok((mesh.triangles() > 0).then_some(mesh))
    }

    fn primitive(&mut self, p: &Value, u: &Use, mesh: &mut Mesh) -> Result<()> {
        let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
        let attrs = p.get("attributes").cloned().unwrap_or(Value::Null);
        let Some(pos) = index(&attrs, "POSITION") else {
            return Ok(());
        };
        if !matches!(mode, 4..=6) {
            // Points and lines are counted, not drawn.
            return Ok(());
        }

        self.draco.clear();
        if let Some(ext) = p.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")) {
            #[cfg(feature = "draco")]
            self.decode_draco(ext, &attrs, index(p, "indices"))?;
            #[cfg(not(feature = "draco"))]
            {
                // Files may carry uncompressed data beside the Draco stream, for readers without it.
                let _ = ext;
                ensure!(
                    self.acc(pos)?.view.is_some(),
                    "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
                );
            }
        }

        // Check the triangle count before reading any vertices.
        let corners = match index(p, "indices") {
            Some(i) => self.acc(i)?.count,
            None => self.acc(pos)?.count,
        };
        let tris = if mode == 4 {
            corners / 3
        } else {
            corners.saturating_sub(2)
        };
        let total = tris.saturating_mul(u.transforms.len());
        ensure!(total <= self.budget, "too many triangles to convert");

        let mut positions = self.read(pos, [0.0; 3])?;
        let n = positions.len();
        let idx = match index(p, "indices") {
            Some(i) => self.read_indices(i)?,
            None => (0..u32_of(n)).collect(),
        };
        let mut indices = triangles(mode, idx);
        if indices.iter().copied().max().is_some_and(|m| m as usize >= n) {
            // Drop triangles that name vertices that don't exist.
            let mut w = 0;
            for t in 0..indices.len() / 3 {
                let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
                if tri.iter().all(|&i| (i as usize) < n) {
                    indices[w..w + 3].copy_from_slice(&tri);
                    w += 3;
                }
            }
            indices.truncate(w);
        }
        if indices.is_empty() {
            return Ok(());
        }

        let material = index(p, "material");
        let (set, uv_matrix) = self.uv_set(material);
        let attr = |name: &str| index(&attrs, name);
        let mut normals = match attr("NORMAL") {
            Some(a) => self.read(a, [0.0; 3])?,
            None => Vec::new(),
        };
        let mut uvs = match attr(&format!("TEXCOORD_{set}")) {
            Some(a) => self.read(a, [0.0; 2])?,
            None => Vec::new(),
        };
        let colors = match attr("COLOR_0") {
            Some(a) => self.read(a, [0.0, 0.0, 0.0, 1.0])?,
            None => Vec::new(),
        };
        // Attributes that don't match the positions are dropped rather than trusted.
        if normals.len() != n {
            normals = Vec::new();
        }
        if uvs.len() != n {
            uvs = Vec::new();
        }
        let colors = if colors.len() == n { colors } else { Vec::new() };
        if let Some(t) = uv_matrix {
            for uv in &mut uvs {
                let [u0, v0] = *uv;
                *uv = [t[0] * u0 + t[1] * v0 + t[2], t[3] * u0 + t[4] * v0 + t[5]];
            }
        }

        // Morph targets at their default weights.
        for (k, target) in items(p, "targets").iter().enumerate() {
            let w = u.weights.get(k).copied().unwrap_or(0.0);
            if w == 0.0 {
                continue;
            }
            if let Some(a) = index(target, "POSITION") {
                for (v, d) in positions.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
            if !normals.is_empty()
                && let Some(a) = index(target, "NORMAL")
            {
                for (v, d) in normals.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
        }
        if let Some(skin) = u.skin {
            self.skin(skin, &attrs, &mut positions, &mut normals)?;
        } else if !u.weights.is_empty() {
            normals.iter_mut().for_each(normalize);
        }

        let g = Geometry {
            positions,
            normals,
            uvs,
            colors,
            indices,
        };
        let material = self.material(material, mesh);
        self.budget -= g.indices.len() / 3 * u.transforms.len();
        mesh.prims.push(place(g, &u.transforms, material));
        Ok(())
    }

    /// Decodes a primitive's Draco stream, so its indices and the attributes it compresses read
    /// from the decoded values instead of their buffer views. Positions and other quantized
    /// attributes come out dequantized; integer attributes are normalized here when their
    /// accessor says so.
    #[cfg(feature = "draco")]
    fn decode_draco(&mut self, ext: &Value, attrs: &Value, indices: Option<usize>) -> Result<()> {
        use draco_core::decode_limits::DecodeLimits;
        use draco_core::decoder_buffer::DecoderBuffer;
        use draco_core::draco_types::DataType;
        use draco_core::mesh_decoder::MeshDecoder;

        let view = index(ext, "bufferView").context("glTF Draco primitive has no bufferView")?;
        let bytes = self.view_bytes(view)?;
        let limits = DecodeLimits::default()
            .with_max_faces(self.budget as u64)
            .with_max_points(MAX_VERTICES as u64);
        let mut buffer = DecoderBuffer::new(&bytes).with_limits(limits);
        let mut dm = draco_core::mesh::Mesh::new();
        MeshDecoder::new()
            .decode(&mut buffer, &mut dm)
            .map_err(|e| invalid!("glTF Draco data doesn't decode: {e}"))?;
        drop(bytes);

        if let Some(i) = indices {
            let mut out = Vec::with_capacity(dm.num_faces() * 12);
            for face in dm.faces() {
                for p in face {
                    out.extend_from_slice(&p.0.to_le_bytes());
                }
            }
            self.draco.push((i, out.into(), 5125, 1));
        }
        let points = dm.num_points();
        let mut decoded = Vec::new();
        for (name, id) in ext.get("attributes").and_then(Value::as_object).into_iter().flatten() {
            let (Some(acc), Some(id)) = (index(attrs, name), id.as_u64().and_then(|v| u32::try_from(v).ok())) else {
                continue;
            };
            let attribute = dm
                .attribute_by_unique_id(id)
                .with_context(|| format!("glTF Draco stream has no attribute {id} for {name}"))?;
            let a = self.acc(acc)?;
            let mut values = attribute.read_f32s(points, a.n);
            if a.normalized || attribute.normalized() {
                let max = match attribute.data_type() {
                    DataType::Uint8 => Some((255.0, 0.0)),
                    DataType::Int8 => Some((127.0, -1.0)),
                    DataType::Uint16 => Some((65535.0, 0.0)),
                    DataType::Int16 => Some((32767.0, -1.0)),
                    _ => None,
                };
                if let Some((max, floor)) = max {
                    for v in &mut values {
                        *v = (*v / max).max(floor);
                    }
                }
            }
            let mut out = Vec::with_capacity(values.len() * 4);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            decoded.push((acc, out.into(), 5126, a.n));
        }
        self.draco.extend(decoded);
        Ok(())
    }

    /// Poses a skinned primitive: each vertex goes through its joints' matrices, blended by
    /// weight.
    fn skin(&mut self, skin: usize, attrs: &Value, positions: &mut [[f32; 3]], normals: &mut [[f32; 3]]) -> Result<()> {
        let s = self.get("skins", skin)?.clone();
        let joints = list_of(&s, "joints");
        let ibm = match index(&s, "inverseBindMatrices") {
            Some(a) => self.read(a, IDENTITY.map(f32_of))?,
            None => Vec::new(),
        };
        let mats: Vec<M4> = joints
            .iter()
            .enumerate()
            .map(|(k, &j)| {
                let w = self.world.get(j).copied().unwrap_or(IDENTITY);
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                mul(&w, &b)
            })
            .collect();
        let (Some(ja), Some(wa)) = (index(attrs, "JOINTS_0"), index(attrs, "WEIGHTS_0")) else {
            return Ok(());
        };
        // Joint indices are integers, read as floats exactly.
        let ji = self.read(ja, [0.0; 4])?;
        let wi = self.read(wa, [0.0; 4])?;
        if ji.len() != positions.len() || wi.len() != positions.len() {
            return Ok(());
        }
        for (v, (j, w)) in ji.iter().zip(&wi).enumerate() {
            let sum: f32 = w.iter().sum();
            if sum <= 0.0 {
                continue;
            }
            let mut m = [0.0; 16];
            for k in 0..4 {
                if w[k] == 0.0 {
                    continue;
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "joint indices are small whole numbers; `as` saturates bad ones and `get` refuses them"
                )]
                let jm = mats.get(j[k] as usize).unwrap_or(&IDENTITY);
                let f = f64::from(w[k] / sum);
                for (o, x) in m.iter_mut().zip(jm) {
                    *o += f * x;
                }
            }
            positions[v] = apply_point(&m, positions[v]);
            if let Some(nv) = normals.get_mut(v) {
                *nv = apply_normal(&normal_matrix(&m), *nv);
            }
        }
        Ok(())
    }

    /// The texture coordinate set a material's color texture reads, and its
    /// `KHR_texture_transform` as a 2x3 matrix (row-major).
    fn uv_set(&self, material: Option<usize>) -> (u64, Option<[f32; 6]>) {
        let Some(info) = material
            .and_then(|m| list(self.json, "materials").get(m))
            .and_then(color_texture)
        else {
            return (0, None);
        };
        let mut set = info.get("texCoord").and_then(Value::as_u64).unwrap_or(0);
        let Some(t) = info.get("extensions").and_then(|e| e.get("KHR_texture_transform")) else {
            return (set, None);
        };
        if let Some(s) = t.get("texCoord").and_then(Value::as_u64) {
            set = s;
        }
        let [ox, oy] = f32s(t.get("offset"), [0.0; 2]);
        let [sx, sy] = f32s(t.get("scale"), [1.0; 2]);
        let r = f32_of(t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0));
        let (sin, cos) = r.sin_cos();
        // translation * rotation * scale, as the extension defines them.
        (set, Some([cos * sx, sin * sy, ox, -sin * sx, cos * sy, oy]))
    }

    /// The mesh material for glTF material `i` (or glTF's default material).
    fn material(&mut self, i: Option<usize>, mesh: &mut Mesh) -> usize {
        let slot = match i {
            Some(i) if i < self.materials.len() => self.materials[i],
            _ => self.default_material,
        };
        if let Some(s) = slot {
            return s;
        }
        let mut m = Material {
            name: String::new(),
            color: [1.0; 4],
            color_image: None,
            normal_image: None,
            emissive: [0.0; 3],
            metallic: 1.0,
            roughness: 1.0,
            double_sided: false,
        };
        let def = i.and_then(|i| list(self.json, "materials").get(i)).cloned();
        if let Some(d) = &def {
            d.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .clone_into(&mut m.name);
            let sg = d
                .get("extensions")
                .and_then(|e| e.get("KHR_materials_pbrSpecularGlossiness"));
            match (d.get("pbrMetallicRoughness"), sg) {
                (None, Some(sg)) => {
                    m.color = f32s(sg.get("diffuseFactor"), [1.0; 4]);
                    m.metallic = 0.0;
                    m.roughness = 1.0 - f32_of(sg.get("glossinessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
                (pbr, _) => {
                    let pbr = pbr.cloned().unwrap_or(Value::Null);
                    m.color = f32s(pbr.get("baseColorFactor"), [1.0; 4]);
                    m.metallic = f32_of(pbr.get("metallicFactor").and_then(Value::as_f64).unwrap_or(1.0));
                    m.roughness = f32_of(pbr.get("roughnessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
            }
            let strength = d
                .pointer("/extensions/KHR_materials_emissive_strength/emissiveStrength")
                .and_then(Value::as_f64)
                .map_or(1.0, f32_of);
            m.emissive = f32s(d.get("emissiveFactor"), [0.0; 3]).map(|e| e * strength);
            m.double_sided = d.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
            if d.get("alphaMode").and_then(Value::as_str).unwrap_or("OPAQUE") == "OPAQUE" {
                m.color[3] = 1.0;
            }
            if let Some(t) = color_texture(d).and_then(|t| index(t, "index")) {
                m.color_image = self.texture(t, mesh);
            }
            if let Some(t) = d.get("normalTexture").and_then(|t| index(t, "index")) {
                m.normal_image = self.texture(t, mesh);
            }
        }
        mesh.materials.push(m);
        let s = mesh.materials.len() - 1;
        match i {
            Some(i) if i < self.materials.len() => self.materials[i] = Some(s),
            _ => self.default_material = Some(s),
        }
        s
    }

    /// The mesh image for glTF texture `t`: its source, or failing that one an extension
    /// names (`EXT_texture_webp`, `KHR_texture_basisu`, ...), if an image can be made of it.
    fn texture(&mut self, t: usize, mesh: &mut Mesh) -> Option<usize> {
        let tex = list(self.json, "textures").get(t)?.clone();
        let mut sources: Vec<usize> = index(&tex, "source").into_iter().collect();
        if let Some(ext) = tex.get("extensions").and_then(Value::as_object) {
            sources.extend(ext.values().filter_map(|e| index(e, "source")));
        }
        sources.into_iter().find_map(|s| self.image(s, mesh))
    }

    /// The mesh image for glTF image `s`, when it's PNG or JPEG (or WebP, with the `webp`
    /// feature).
    fn image(&mut self, s: usize, mesh: &mut Mesh) -> Option<usize> {
        if let Some(done) = *self.images.get(s)? {
            return done;
        }
        let img = list(self.json, "images").get(s)?;
        let bytes = if let Some(v) = index(img, "bufferView") {
            self.view_bytes(v).ok()
        } else {
            // A file the loader didn't find stays a URI, and isn't drawn.
            img.get("uri")
                .and_then(Value::as_str)
                .filter(|u| u.starts_with("data:"))
                .and_then(|u| container::data_uri(u).ok())
                .map(|(_, b)| b)
        };
        let out = bytes.and_then(Image::decode).map(|i| {
            mesh.images.push(i);
            mesh.images.len() - 1
        });
        self.images[s] = Some(out);
        out
    }
}

/// A material's color texture info: `baseColorTexture`, or the specular-glossiness diffuse.
fn color_texture(m: &Value) -> Option<&Value> {
    m.get("pbrMetallicRoughness")
        .and_then(|p| p.get("baseColorTexture"))
        .or_else(|| {
            m.get("extensions")?
                .get("KHR_materials_pbrSpecularGlossiness")?
                .get("diffuseTexture")
        })
}

/// A primitive's geometry at each of `transforms`, as one primitive.
#[allow(clippy::float_cmp_const, reason = "only the exact identity may skip the transform")]
fn place(g: Geometry, transforms: &[M4], material: usize) -> Prim {
    if let [t] = transforms
        && *t == IDENTITY
    {
        return Prim {
            positions: g.positions,
            normals: g.normals,
            uvs: g.uvs,
            colors: g.colors,
            indices: g.indices,
            material,
        };
    }
    let n = g.positions.len();
    let k = transforms.len();
    let mut p = Prim {
        positions: Vec::with_capacity(n * k),
        normals: Vec::with_capacity(g.normals.len() * k),
        uvs: Vec::with_capacity(g.uvs.len() * k),
        colors: Vec::with_capacity(g.colors.len() * k),
        indices: Vec::with_capacity(g.indices.len() * k),
        material,
    };
    for t in transforms {
        let base = u32_of(p.positions.len());
        p.positions.extend(g.positions.iter().map(|&v| apply_point(t, v)));
        let nm = normal_matrix(t);
        p.normals.extend(g.normals.iter().map(|&v| apply_normal(&nm, v)));
        p.uvs.extend_from_slice(&g.uvs);
        p.colors.extend_from_slice(&g.colors);
        let flip = det3(t) < 0.0;
        for tri in g.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] + base, tri[1] + base, tri[2] + base];
            p.indices.extend_from_slice(&if flip { [a, c, b] } else { [a, b, c] });
        }
    }
    p
}

impl Reader<'_> {
    /// Sets each node's animated properties for `pose`.
    fn pose(&mut self, pose: Pose) -> Result<()> {
        let Pose::Animation { index: a, time } = pose else {
            return Ok(());
        };
        let anim = self.get("animations", a)?;
        let samplers = items(anim, "samplers");
        for ch in items(anim, "channels") {
            let Some(target) = ch.get("target") else { continue };
            let (Some(node), Some(path)) = (index(target, "node"), target.get("path").and_then(Value::as_str)) else {
                // A channel for an extension's target (KHR_animation_pointer), or none.
                continue;
            };
            if node >= self.posed.len() {
                continue;
            }
            let s = index(ch, "sampler")
                .and_then(|s| samplers.get(s))
                .context("glTF animation channel names a missing sampler")?;
            let (Some(input), Some(output)) = (index(s, "input"), index(s, "output")) else {
                bail!("glTF animation sampler has no input or output");
            };
            let interpolation = s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR");
            let times: Vec<f32> = self.read::<1>(input, [0.0])?.into_flattened();
            let values: Vec<f32> = match path {
                "translation" | "scale" => self.read::<3>(output, [0.0; 3])?.into_flattened(),
                "rotation" => self.read::<4>(output, [0.0, 0.0, 0.0, 1.0])?.into_flattened(),
                "weights" => self.read::<1>(output, [0.0])?.into_flattened(),
                _ => continue,
            };
            let Some(v) = sample(&times, &values, interpolation, time, path == "rotation") else {
                continue;
            };
            let p = &mut self.posed[node];
            match path {
                "translation" => p.translation = Some([v[0], v[1], v[2]]),
                "scale" => p.scale = Some([v[0], v[1], v[2]]),
                "rotation" => p.rotation = Some([v[0], v[1], v[2], v[3]]),
                _ => p.weights = Some(v.into_iter().map(f32_of).collect()),
            }
        }
        Ok(())
    }

    /// Node `i`'s own transform, as posed.
    fn local(&self, i: usize, node: &Value) -> M4 {
        let p = &self.posed[i];
        // A node with a matrix can't be animated.
        if let Some(m) = node.get("matrix").and_then(Value::as_array).filter(|m| m.len() == 16) {
            let mut out = IDENTITY;
            for (o, x) in out.iter_mut().zip(m) {
                *o = x.as_f64().unwrap_or(*o);
            }
            return out;
        }
        trs(
            p.translation
                .unwrap_or_else(|| floats(node.get("translation"), [0.0; 3])),
            p.rotation
                .unwrap_or_else(|| floats(node.get("rotation"), [0.0, 0.0, 0.0, 1.0])),
            p.scale.unwrap_or_else(|| floats(node.get("scale"), [1.0; 3])),
        )
    }
}

/// An animation sampler's value at `t`: `values` holds one element per key (three, in-tangent,
/// value and out-tangent, for `CUBICSPLINE`), each `values.len() / keys` wide. Before the
/// first key it's the first value, after the last the last. `None` for a sampler without keys
/// or with values that don't fit them.
pub(crate) fn sample(times: &[f32], values: &[f32], interpolation: &str, t: f32, rotation: bool) -> Option<Vec<f64>> {
    let keys = times.len();
    let cubic = interpolation == "CUBICSPLINE";
    let per = if cubic { 3 } else { 1 };
    if keys == 0 || values.is_empty() || !values.len().is_multiple_of(keys * per) {
        return None;
    }
    let w = values.len() / (keys * per);
    // The value of key k (its middle element for a cubic spline), and its tangents.
    let at = |k: usize, part: usize| -> Vec<f64> {
        let e = k * per + part;
        values[e * w..(e + 1) * w].iter().map(|&x| f64::from(x)).collect()
    };
    let value = |k: usize| at(k, usize::from(cubic));
    let k = times.partition_point(|&x| x <= t);
    if k == 0 {
        return Some(value(0));
    }
    if k == keys {
        return Some(value(keys - 1));
    }
    let (k, (t0, t1)) = (k - 1, (f64::from(times[k - 1]), f64::from(times[k])));
    let dt = t1 - t0;
    let s = if dt > 0.0 { (f64::from(t) - t0) / dt } else { 0.0 };
    let mut out = match interpolation {
        "STEP" => value(k),
        "CUBICSPLINE" => {
            // Hermite, per the glTF 2.0 specification's appendix C.
            let (v0, b0, a1, v1) = (at(k, 1), at(k, 2), at(k + 1, 0), at(k + 1, 1));
            let (s2, s3) = (s * s, s * s * s);
            let h = [
                2.0 * s3 - 3.0 * s2 + 1.0,
                (s3 - 2.0 * s2 + s) * dt,
                -2.0 * s3 + 3.0 * s2,
                (s3 - s2) * dt,
            ];
            (0..w)
                .map(|c| h[0] * v0[c] + h[1] * b0[c] + h[2] * v1[c] + h[3] * a1[c])
                .collect()
        }
        _ if rotation && w == 4 => slerp(&value(k), &value(k + 1), s),
        _ => {
            let (a, b) = (value(k), value(k + 1));
            a.iter().zip(&b).map(|(x, y)| x + (y - x) * s).collect()
        }
    };
    if rotation && w == 4 {
        let len = out.iter().map(|c| c * c).sum::<f64>().sqrt();
        if len > 0.0 {
            for c in &mut out {
                *c /= len;
            }
        }
    }
    Some(out)
}

/// Spherical linear interpolation between unit quaternions, the short way round.
fn slerp(a: &[f64], b: &[f64], s: f64) -> Vec<f64> {
    let mut d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };
    d *= sign;
    let (wa, wb) = if d > 0.9995 {
        // Nearly the same rotation: a straight line is as good, and stable.
        (1.0 - s, s)
    } else {
        let theta = d.acos();
        let sin = theta.sin();
        (((1.0 - s) * theta).sin() / sin, (s * theta).sin() / sin)
    };
    a.iter().zip(b).map(|(x, y)| wa * x + wb * sign * y).collect()
}

#[cfg(all(test, feature = "fs"))]
#[allow(clippy::cast_precision_loss, reason = "test grids are small")]
mod tests {
    use crate::assets::Folder;
    use crate::document::{Keep, Pose};
    use crate::mesh::{Material, Mesh, Prim, to_glb};

    /// A bumpy grid with normals, UVs and colors, in two materials.
    fn grid(n: u32) -> Mesh {
        let mut prims = Vec::new();
        for m in 0..2u32 {
            let mut p = Prim {
                material: m as usize,
                ..Prim::default()
            };
            for y in 0..=n {
                for x in 0..=n {
                    let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
                    p.positions.push([u + m as f32, (u * 7.0).sin() * 0.1, v]);
                    p.normals.push([0.0, 1.0, 0.0]);
                    p.uvs.push([u, v]);
                    p.colors.push([u, v, 0.5, 1.0]);
                }
            }
            for y in 0..n {
                for x in 0..n {
                    let i = y * (n + 1) + x;
                    p.indices
                        .extend_from_slice(&[i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
                }
            }
            prims.push(p);
        }
        let mut red = Material::plain("Red", [1.0, 0.0, 0.0, 1.0]);
        red.metallic = 0.5;
        red.roughness = 0.25;
        red.emissive = [0.1, 0.2, 0.3];
        red.double_sided = true;
        Mesh {
            prims,
            materials: vec![red, Material::plain("Blue", [0.0, 0.0, 1.0, 0.5])],
            images: Vec::new(),
        }
    }

    /// Each triangle's corners, rotated to start at its smallest, so triangles that begin at
    /// another corner (as meshopt's triangle codec may write them) compare equal.
    fn triangles(p: &Prim) -> Vec<[[u32; 12]; 3]> {
        let key = |i: u32| {
            let i = i as usize;
            let f = |v: &[f32]| v.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
            let mut k = [0u32; 12];
            let all = [f(&p.positions[i]), f(&p.normals[i]), f(&p.uvs[i]), f(&p.colors[i])].concat();
            k.copy_from_slice(&all);
            k
        };
        let mut out: Vec<_> = p
            .indices
            .chunks(3)
            .map(|t| {
                let c = [key(t[0]), key(t[1]), key(t[2])];
                let r = (0..3).min_by_key(|&s| c[s]).unwrap();
                [c[r], c[(r + 1) % 3], c[(r + 2) % 3]]
            })
            .collect();
        out.sort_unstable();
        out
    }

    fn load(dir: &std::path::Path, name: &str, bytes: &[u8]) -> Mesh {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let ext = name.rsplit('.').next().unwrap();
        let doc = crate::formats::load(&path, ext, &Folder::new(dir), Keep::ALL).unwrap();
        // From memory, with the folder's files as the assets, it reads the same.
        let mut memory = crate::assets::Memory::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            memory.insert(&p.file_name().unwrap().to_string_lossy(), std::fs::read(&p).unwrap());
        }
        let in_memory = crate::formats::load_bytes(bytes, name, ext, &memory, Keep::ALL).unwrap();
        assert!(
            doc.to_glb().unwrap() == in_memory.to_glb().unwrap(),
            "{name} reads differently from memory"
        );
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert!(
            to_glb(&mesh).unwrap() == to_glb(&in_memory.flatten(Pose::Rest).unwrap().unwrap()).unwrap(),
            "{name} flattens differently from memory"
        );
        mesh
    }

    fn same(a: &Mesh, b: &Mesh) {
        assert_eq!(a.prims.len(), b.prims.len());
        for (pa, pb) in a.prims.iter().zip(&b.prims) {
            assert_eq!(triangles(pa), triangles(pb));
            assert_eq!(pa.material, pb.material);
        }
        for (ma, mb) in a.materials.iter().zip(&b.materials) {
            assert_eq!(
                (
                    &ma.name,
                    ma.color,
                    ma.emissive,
                    ma.metallic,
                    ma.roughness,
                    ma.double_sided
                ),
                (
                    &mb.name,
                    mb.color,
                    mb.emissive,
                    mb.metallic,
                    mb.roughness,
                    mb.double_sided
                )
            );
        }
    }

    #[test]
    fn reads_back_what_to_glb_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(30);
        let glb = to_glb(&mesh).unwrap();
        same(&mesh, &load(dir.path(), "plain.glb", &glb));
    }

    #[test]
    fn streamed_glb_cut_short_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let glb = to_glb(&grid(60)).unwrap();
        assert!(glb.len() > 2 * 64 * 1024);
        let path = dir.path().join("cut.glb");
        std::fs::write(&path, &glb[..glb.len() - 1000]).unwrap();
        let err = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap_err();
        assert!(err.to_string().contains("declares"), "{err}");
    }

    #[test]
    fn reads_meshopt_compressed_glb() {
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(30);
        let packed = crate::gltf::compress(&to_glb(&mesh).unwrap()).unwrap();
        same(&mesh, &load(dir.path(), "packed.glb", &packed));
    }

    #[test]
    fn decode_leaves_plain_gltf() {
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(30);
        let packed = crate::gltf::compress(&to_glb(&mesh).unwrap()).unwrap();
        let path = dir.path().join("packed.glb");
        std::fs::write(&path, &packed).unwrap();
        let mut doc = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap();
        assert_eq!(doc.json["buffers"].as_array().unwrap().len(), 2, "data and fallback");
        let before = doc.flatten(Pose::Rest).unwrap().unwrap();
        doc.decode().unwrap();
        let text = serde_json::to_string(&doc.json).unwrap();
        assert!(!text.contains("meshopt"), "{text}");
        // The compressed data and the fallback are gone: only the decoded buffer is left.
        assert_eq!(doc.json["buffers"].as_array().unwrap().len(), 1);
        let after = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(to_glb(&before).unwrap(), to_glb(&after).unwrap());
        same(&mesh, &load(dir.path(), "decoded.glb", &doc.to_glb().unwrap()));
        // Decoding plain glTF changes nothing.
        let glb = doc.to_glb().unwrap();
        doc.decode().unwrap();
        assert_eq!(glb, doc.to_glb().unwrap());
    }

    #[test]
    fn decode_widens_indices_holding_their_types_largest_value() {
        // 256 vertices on a line, u8 indices naming the last one: 255, a restart in WebGL 2.
        let dir = tempfile::tempdir().unwrap();
        let mut bin = Vec::new();
        for i in 0..256u16 {
            for v in [f32::from(i), f32::from(i % 2), 0.0] {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let index_offset = bin.len();
        bin.extend_from_slice(&[0, 1, 2, 253, 254, 255, 0, 0]);
        std::fs::write(dir.path().join("line.bin"), &bin).unwrap();
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"uri": "line.bin", "byteLength": bin.len()}],
            "bufferViews": [
                {"buffer": 0, "byteLength": index_offset},
                {"buffer": 0, "byteOffset": index_offset, "byteLength": 6},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 256, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [255.0, 1.0, 0.0]},
                {"bufferView": 1, "componentType": 5121, "count": 6, "type": "SCALAR"},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        });
        let path = dir.path().join("line.gltf");
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let mut doc = crate::formats::load(&path, "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
        let before = doc.flatten(Pose::Rest).unwrap().unwrap();
        doc.decode().unwrap();
        assert_eq!(doc.json["accessors"][1]["componentType"], 5123);
        let after = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(to_glb(&before).unwrap(), to_glb(&after).unwrap());
    }

    #[test]
    fn reads_meshopt_compressed_gltf_with_external_buffers() {
        // A compressed GLB split into a .gltf and its buffer files.
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(10);
        let packed = crate::gltf::compress(&to_glb(&mesh).unwrap()).unwrap();
        let json_len = u32::from_le_bytes(packed[12..16].try_into().unwrap()) as usize;
        let mut json: serde_json::Value = serde_json::from_slice(&packed[20..20 + json_len]).unwrap();
        let bin = &packed[20 + json_len + 8..];
        json["buffers"][0]["uri"] = "packed.bin".into();
        std::fs::write(dir.path().join("packed.bin"), bin).unwrap();
        same(
            &mesh,
            &load(dir.path(), "packed.gltf", &serde_json::to_vec(&json).unwrap()),
        );
    }

    #[test]
    fn emissive_brighter_than_one_round_trips_as_emissive_strength() {
        let mut mesh = grid(2);
        mesh.materials[0].emissive = [4.0, 2.0, 0.0];
        let doc = crate::document::Document::from_mesh(mesh, serde_json::Map::new());
        let m = &doc.json["materials"][0];
        assert_eq!(m["emissiveFactor"], serde_json::json!([1.0, 0.5, 0.0]));
        assert_eq!(
            m["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
            4.0
        );
        assert_eq!(
            doc.json["extensionsUsed"],
            serde_json::json!(["KHR_materials_emissive_strength"])
        );
        let back = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(back.materials[0].emissive, [4.0, 2.0, 0.0]);
    }

    #[test]
    fn a_required_extension_it_cant_draw_is_an_error() {
        let mut doc = crate::document::Document::from_mesh(grid(2), serde_json::Map::new());
        doc.json
            .insert("extensionsRequired".into(), serde_json::json!(["KHR_materials_unlit"]));
        assert!(
            doc.flatten(Pose::Rest).unwrap().is_some(),
            "a look is drawn as best it can be"
        );
        doc.json.insert(
            "extensionsRequired".into(),
            serde_json::json!(["EXT_some_future_geometry"]),
        );
        let err = doc.flatten(Pose::Rest).unwrap_err().to_string();
        assert!(err.contains("EXT_some_future_geometry"), "{err}");
    }
}
