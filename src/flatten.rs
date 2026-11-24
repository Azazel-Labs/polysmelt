//! The flat view of a [`Document`]: the triangles, lines and points its default scene draws,
//! posed, as a [`Mesh`], and how many of each that is, counted from the JSON alone.
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
//! - Triangles, strips and fans; lines, line strips and loops (as segments) and points, unless
//!   [`Keep::lines_and_points`] leaves them out. A primitive drawn on the same vertices as the
//!   one before it, as Polysmelt writes a primitive's lines and points, joins it rather than
//!   repeat them. Draco-compressed points and lines aren't drawn.
//! - `POSITION`, `NORMAL`, the color texture's `TEXCOORD_n` (its `KHR_texture_transform`
//!   baked in) and `COLOR_0`, in any component type `KHR_mesh_quantization` allows, sparse
//!   accessors included. Without `NORMAL` a primitive keeps no normals: glTF shades it flat.
//! - Materials: base color, metallic and roughness factors, emissive (with
//!   `KHR_materials_emissive_strength`), alpha mode, double-sidedness, unlit, the other
//!   `KHR_materials_*` extensions (ior, specular, transmission, clearcoat, sheen, volume,
//!   anisotropy, iridescence, dispersion), and every texture
//!   (`KHR_materials_pbrSpecularGlossiness` as its diffuse) with its sampler, coordinate set and
//!   `KHR_texture_transform` (relative to the color texture's, which the coordinates have baked
//!   in), when their images are PNG or JPEG, or a format a feature re-encodes as PNG (WebP with
//!   `EXT_texture_webp`, TGA, BMP, GIF, TIFF, PSD, DDS).
//! - Meshopt-compressed views are decoded as they are read; Draco
//!   (`KHR_draco_mesh_compression`) needs the `draco` feature.
//! - A required extension it can't honour is an error, unless it only changes how surfaces look.
//!
//! Accessors are read from their buffers in place, so the peak is little more than the mesh.

mod decode;
mod repair;
mod rig;

pub(crate) use repair::repaired;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{Map, Value};

#[cfg(doc)]
use crate::document::Keep;
use crate::document::{Bounds, Counts, Document, Pose};
use crate::gltf as container;
use crate::mesh::{
    Anisotropy, Clearcoat, Extensions, Factor, Image, Iridescence, Material, Mesh, Prim, Sampling, Sheen, Specular,
    Texture, UvTransform, Volume,
};
use crate::num::{f32_of, u32_of, usize_of};
use crate::{Context as _, Result};

/// How much of a buffer is read at once.
const BLOCK: usize = 256 * 1024;

impl Document {
    /// The triangles, lines and points the default scene draws in `pose`, as one mesh with every
    /// transform, instance, morph and skin baked in; `None` when it draws nothing. Lines and
    /// points are left out when [`Document::lines_and_points`] says so.
    ///
    /// # Errors
    ///
    /// When the document is malformed (an accessor out of range, a scene that doesn't exist),
    /// `pose` names an animation that doesn't exist, it uses Draco without the `draco`
    /// feature, or it draws more than [`Document::max_triangles`] triangles or
    /// [`Document::max_points`] points and line segments.
    pub fn flatten(&self, pose: Pose) -> Result<Option<Mesh>> {
        let mut r = Reader::new(self)?;
        r.bind = pose == Pose::Bind;
        r.pose(pose)?;
        r.mesh()
    }

    /// What the default scene draws with each node in the transform `nodes` gives it,
    /// as [`Document::flatten`] does: for a pose worked out elsewhere, such as one retargeted
    /// from another rig, blended, or solved by IK. `nodes` is indexed as [`Rig::nodes`] (as
    /// [`Rig::pose_into`] fills it); nodes past its end keep their own transforms, and morph
    /// weights are the nodes' own.
    ///
    /// # Errors
    ///
    /// As [`Document::flatten`].
    ///
    /// [`Rig::nodes`]: crate::rig::Rig::nodes
    /// [`Rig::pose_into`]: crate::rig::Rig::pose_into
    pub fn flatten_nodes(&self, nodes: &[crate::rig::Trs]) -> Result<Option<Mesh>> {
        let mut r = Reader::new(self)?;
        for (p, n) in r.posed.iter_mut().zip(nodes) {
            p.translation = Some(n.translation.map(f64::from));
            p.rotation = Some(n.rotation.map(f64::from));
            p.scale = Some(n.scale.map(f64::from));
            p.given = true;
        }
        r.mesh()
    }

    /// Accessor `accessor`'s elements as floats, `width` to an element (a matrix's columns in
    /// order, without padding), as a renderer reads them: meshopt-compressed views decoded,
    /// sparse substitutions applied, normalized integers mapped to [0, 1] or [-1, 1] and other
    /// integers as they are (exact below 2^24). An accessor without a buffer view reads as
    /// zeros under its sparse values, unless a Draco stream holds it: then it reads as that stream
    /// decodes (which needs the `draco` feature).
    ///
    /// # Errors
    ///
    /// When the accessor doesn't exist, is malformed or runs past its buffer, or its
    /// compressed view doesn't decode.
    pub fn read_floats(&self, accessor: usize) -> Result<(usize, Vec<f32>)> {
        let mut r = Reader::bare(self);
        r.draco_holding(accessor)?;
        let a = r.acc(accessor)?;
        let values = match a.n {
            1 => r.read_acc(&a, [0.0])?.into_flattened(),
            2 => r.read_acc(&a, [0.0; 2])?.into_flattened(),
            3 => r.read_acc(&a, [0.0; 3])?.into_flattened(),
            4 => r.read_acc(&a, [0.0; 4])?.into_flattened(),
            9 => r.read_acc(&a, [0.0; 9])?.into_flattened(),
            _ => r.read_acc(&a, [0.0; 16])?.into_flattened(),
        };
        Ok((a.n, values))
    }

    /// Accessor `accessor`'s unsigned integer scalars (indices, or anything else stored so),
    /// exactly, read as [`Document::read_floats`] reads.
    ///
    /// # Errors
    ///
    /// As [`Document::read_floats`], and when the accessor isn't unsigned integer scalars.
    pub fn read_indices(&self, accessor: usize) -> Result<Vec<u32>> {
        let mut r = Reader::bare(self);
        r.draco_holding(accessor)?;
        r.read_indices(accessor)
    }

    /// How many triangles, points and line segments the default scene draws (every instance
    /// counted), from the JSON alone: nothing is read or decoded.
    #[must_use]
    pub fn counts(&self) -> Counts {
        counts(&self.json)
    }

    /// The box around every vertex the default scene draws in `pose`, transforms, instances,
    /// morphs and skins applied, exactly as [`Document::flatten`] places them (lines and points
    /// too, unless [`Document::lines_and_points`] leaves them out); `None` when it draws nothing.
    /// Only positions are read: no other attribute, material or image. A file's own bounds
    /// (glTF's accessor `min` and `max`) aren't trusted, since files get them wrong.
    ///
    /// # Errors
    ///
    /// As [`Document::flatten`], except that no triangle or point limit applies.
    pub fn bounds(&self, pose: Pose) -> Result<Option<Bounds>> {
        let mut r = Reader::new(self)?;
        r.bind = pose == Pose::Bind;
        r.pose(pose)?;
        r.bounds()
    }

    /// How many triangles the default scene draws: [`Document::counts`]' triangles.
    #[must_use]
    pub fn triangles(&self) -> usize {
        self.counts().triangles
    }

    /// The box around every vertex the default scene draws while animation `index` plays: the
    /// [`Document::bounds`] of its poses at `samples` times spread evenly from its first key to
    /// its last, both included (one sample is the first key alone), together. A pose between
    /// samples can reach a little past the box. `None` when no pose draws anything.
    ///
    /// ```no_run
    /// let doc = polysmelt::open("walk.fbx")?;
    /// // Where a camera should stand to frame the character through its whole walk.
    /// let frame = doc.animation_bounds(0, 30)?;
    /// # Ok::<(), polysmelt::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// When the document has no animation `index`, and as [`Document::bounds`].
    #[allow(clippy::cast_precision_loss, reason = "sample numbers are small")]
    pub fn animation_bounds(&self, index: usize, samples: usize) -> Result<Option<Bounds>> {
        let (first, last) = Reader::bare(self).span(index)?;
        let n = samples.max(1);
        let mut out: Option<Bounds> = None;
        for k in 0..n {
            let time = if k == 0 {
                first
            } else {
                first + (last - first) * (k as f32 / (n - 1) as f32)
            };
            let Some(b) = self.bounds(Pose::Animation { index, time })? else {
                continue;
            };
            out = Some(out.map_or(b, |o| Bounds {
                min: std::array::from_fn(|c| o.min[c].min(b.min[c])),
                max: std::array::from_fn(|c| o.max[c].max(b.max[c])),
            }));
        }
        Ok(out)
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
            c.vertices += count(pos) * k;
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
    let mut skinned: HashSet<(usize, usize, Vec<u32>)> = HashSet::new();
    for i in order {
        let n = &nodes[i];
        let Some(m) = index(n, "mesh") else { continue };
        let Some(mesh) = meshes.get(m) else { continue };
        let k = if let Some(skin) = index(n, "skin") {
            let w: Vec<u32> = default_weights(n, mesh).iter().map(|x| x.to_bits()).collect();
            if !skinned.insert((m, skin, w)) {
                continue;
            }
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
    weights_of(node.get("weights").or_else(|| mesh.get("weights")))
}

/// Morph weights, as listed.
fn weights_of(w: Option<&Value>) -> Vec<f32> {
    w.and_then(Value::as_array)
        .map(|w| w.iter().map(|x| f32_of(x.as_f64().unwrap_or(0.0))).collect())
        .unwrap_or_default()
}

/// The members of a node that place it and say what it draws, found in one pass over it rather
/// than a lookup for each: a scene can have many thousands of nodes.
#[derive(Default, Clone, Copy)]
struct NodeView<'v> {
    mesh: Option<&'v Value>,
    skin: Option<&'v Value>,
    children: Option<&'v Value>,
    matrix: Option<&'v Value>,
    translation: Option<&'v Value>,
    rotation: Option<&'v Value>,
    scale: Option<&'v Value>,
    weights: Option<&'v Value>,
    extensions: Option<&'v Value>,
}

impl<'v> NodeView<'v> {
    fn of(node: &'v Value) -> Self {
        let mut v = Self::default();
        for (k, x) in node.as_object().into_iter().flatten() {
            let slot = match k.as_str() {
                "mesh" => &mut v.mesh,
                "skin" => &mut v.skin,
                "children" => &mut v.children,
                "matrix" => &mut v.matrix,
                "translation" => &mut v.translation,
                "rotation" => &mut v.rotation,
                "scale" => &mut v.scale,
                "weights" => &mut v.weights,
                "extensions" => &mut v.extensions,
                _ => continue,
            };
            *slot = Some(x);
        }
        v
    }

    fn children(&self) -> impl Iterator<Item = usize> + 'v {
        self.children
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .filter_map(Value::as_u64)
            .map(usize_of)
    }
}

/// Whether each node is some node's child.
fn children(nodes: &[Value]) -> Vec<bool> {
    let mut child = vec![false; nodes.len()];
    for n in nodes {
        for c in ids(n, "children") {
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
        // (Looked up key by key: `Value::pointer` allocates for each step, on every node.)
        let visible = nodes[i]
            .get("extensions")
            .and_then(|e| e.get("KHR_node_visibility"))
            .and_then(|v| v.get("visible"));
        if visible == Some(&Value::Bool(false)) {
            continue;
        }
        order.push(i);
        if let Some(kids) = nodes[i].get("children").and_then(Value::as_array) {
            stack.extend(kids.iter().rev().filter_map(Value::as_u64).map(usize_of));
        }
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
    /// Components per column: a matrix's height, `n` for a scalar or vector.
    rows: usize,
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

    /// Bytes from one column to the next: a matrix's columns start 4-byte aligned (3.6.2.4).
    fn column(&self) -> usize {
        (self.rows * self.csize()).next_multiple_of(if self.rows < self.n { 4 } else { 1 })
    }

    fn esize(&self) -> usize {
        self.column() * (self.n / self.rows.max(1))
    }

    /// Whether a matrix's columns are padded, so components aren't `csize` apart.
    fn padded(&self) -> bool {
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

/// A 4x4 matrix, column-major as glTF writes it.
pub(crate) type M4 = [f64; 16];

pub(crate) const IDENTITY: M4 = [
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
    doc: &'d Document,
    json: &'d Map<String, Value>,
    /// The accessors and buffer views, looked up once: every attribute read names one of each.
    accessors: &'d [Value],
    views: &'d [Value],
    /// The other top-level lists read for each mesh, node or material, looked up once too.
    meshes: &'d [Value],
    nodes: &'d [Value],
    materials_json: &'d [Value],
    textures: &'d [Value],
    samplers: &'d [Value],
    images_json: &'d [Value],
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
    /// Skinned meshes are left in their bind pose ([`Pose::Bind`]).
    bind: bool,
    /// For [`Pose::Bind`], each joint's transform in its bind pose (its inverse bind matrix's
    /// inverse), from the first skin to list it.
    bound: Vec<Option<M4>>,
    /// Triangles still allowed.
    budget: usize,
    /// Points and line segments still allowed.
    point_budget: usize,
    /// The most vertices an accessor without data (all zeros) may claim: three to each
    /// triangle the document allows, or one to each point.
    max_vertices: usize,
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
    /// Set by the caller rather than an animation, so it holds for a node written as a matrix
    /// too.
    given: bool,
}

fn list<'v>(obj: &'v Map<String, Value>, key: &str) -> &'v [Value] {
    obj.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// `v[k0][k1]...`: a JSON pointer without the allocation `Value::pointer` makes per lookup.
fn at<'v>(v: &'v Value, path: &[&str]) -> Option<&'v Value> {
    path.iter().try_fold(v, |v, k| v.get(k))
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
        "KHR_gaussian_splatting",
    ];
    const LOOKS: &[&str] = &[
        "VRMC_",
        "KHR_materials_",
        "KHR_texture_",
        "EXT_texture_",
        "KHR_lights_",
        "KHR_xmp",
        "EXT_materials_",
        "EXT_lights_",
        "MSFT_texture_",
        "MSFT_packing_",
        "ADOBE_materials_",
        // Shader programs for a material (`KHR_techniques_webgl`); its own values stand in.
        "KHR_technique",
    ];
    // Levels of detail: the node itself is the finest, which is what's drawn.
    // VRM 0.x names its extension plainly, without a vendor prefix.
    const ALSO: &[&str] = &["MSFT_lod", "VRM"];
    DECODED.contains(&extension) || ALSO.contains(&extension) || LOOKS.iter().any(|p| extension.starts_with(p))
}

fn f32s<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    floats(v, default.map(f64::from)).map(f32_of)
}

impl<'d> Reader<'d> {
    #[allow(clippy::unnecessary_wraps, reason = "it fails without the draco feature")]
    fn new(doc: &'d Document) -> Result<Self> {
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
    fn bare(doc: &'d Document) -> Self {
        let json = &doc.json;
        let n = |key| list(json, key).len();
        Self {
            doc,
            json,
            accessors: list(json, "accessors"),
            views: list(json, "bufferViews"),
            meshes: list(json, "meshes"),
            nodes: list(json, "nodes"),
            materials_json: list(json, "materials"),
            textures: list(json, "textures"),
            samplers: list(json, "samplers"),
            images_json: list(json, "images"),
            scratch: Vec::new(),
            meshopt: HashMap::new(),
            materials: vec![None; n("materials")],
            default_material: None,
            images: vec![None; n("images")],
            world: Vec::new(),
            posed: vec![Posed::default(); n("nodes")],
            bind: false,
            bound: Vec::new(),
            budget: doc.max_triangles,
            point_budget: doc.max_points,
            max_vertices: doc.max_triangles.saturating_mul(3).max(if doc.lines_and_points {
                doc.max_points
            } else {
                0
            }),
            draco: Vec::new(),
        }
    }

    fn get(&self, key: &str, i: usize) -> Result<&'d Value> {
        let all = match key {
            "accessors" => self.accessors,
            "bufferViews" => self.views,
            "meshes" => self.meshes,
            _ => list(self.json, key),
        };
        all.get(i).with_context(|| format!("glTF {key}[{i}] doesn't exist"))
    }

    /// A buffer view as (buffer, offset, length, stride), and whether it has extensions (as a
    /// meshopt-compressed one does).
    fn view(&self, i: usize) -> Result<(usize, u64, u64, Option<usize>, bool)> {
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
            .context("glTF buffer view has no buffer")?;
        let offset = offset.and_then(Value::as_u64).unwrap_or(0);
        let len = len
            .and_then(Value::as_u64)
            .context("glTF buffer view has no byteLength")?;
        Ok((buffer, offset, len, stride.and_then(Value::as_u64).map(usize_of), ext))
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

    /// A buffer view's bytes, shared with its buffer when they can be.
    fn view_shared(&mut self, i: usize) -> Result<crate::Bytes> {
        if self.meshopt(i)? {
            return Ok(crate::Bytes::copy_from_slice(&self.meshopt[&i]));
        }
        let (buffer, offset, len, _, _) = self.view(i)?;
        let b = self
            .doc
            .buffers
            .get(buffer)
            .context("glTF buffer view names a missing buffer")?;
        ensure!(
            offset.saturating_add(len) <= b.len(),
            "glTF buffer view runs past its buffer"
        );
        usize::try_from(len).map_err(|_| invalid!("glTF buffer view is too big to hold"))?;
        b.shared(offset, len)
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
                rows: *n,
                normalized: false,
                sparse: None,
                decoded: Some(bytes.clone()),
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
            "glTF accessor {i} has component type {ctype}"
        );
        let (n, rows) = match kind.and_then(Value::as_str) {
            Some("SCALAR") => (1, 1),
            Some("VEC2") => (2, 2),
            Some("VEC3") => (3, 3),
            Some("VEC4") => (4, 4),
            Some("MAT2") => (4, 2),
            Some("MAT3") => (9, 3),
            Some("MAT4") => (16, 4),
            t => bail!("glTF accessor {i} has type {t:?}, not one read here"),
        };
        let u = |x: Option<&Value>| x.and_then(Value::as_u64).map(usize_of);
        Ok(Acc {
            view: u(view),
            offset: offset.and_then(Value::as_u64).unwrap_or(0),
            count: u(count).context("glTF accessor has no count")?,
            ctype,
            n,
            rows,
            normalized: normalized.and_then(Value::as_bool).unwrap_or(false),
            sparse: sparse.cloned(),
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
            ensure!(
                a.count <= self.max_vertices,
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
            "glTF buffer view stride {stride} is under an element's {esize} bytes"
        );
        // The elements' bytes, from the first: the accessor's offset into its view is checked here, not counted twice.
        let span = (a.count as u64 - 1)
            .checked_mul(stride as u64)
            .and_then(|s| s.checked_add(esize as u64))
            .filter(|&s| a.offset.checked_add(s).is_some_and(|end| end <= vlen))
            .with_context(|| format!("glTF accessor's {} elements run past their buffer view", a.count))?;
        // Only a view with extensions can be meshopt-compressed.
        if ext && self.meshopt(view)? {
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
            ensure!(
                a.count <= self.max_vertices,
                "glTF accessor counts {} elements",
                a.count
            );
        }
        let count = a.count;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| invalid!("glTF accessor counts {count} elements"))?;
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
    /// Decodes the Draco stream holding accessor `i`, when it has no buffer view of its own and
    /// a primitive's `KHR_draco_mesh_compression` names it.
    fn draco_holding(&mut self, i: usize) -> Result<()> {
        if self.draco.iter().any(|d| d.0 == i) || self.get("accessors", i)?.get("bufferView").is_some() {
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
            bail!("glTF accessor {i} is Draco-compressed, which needs Polysmelt's `draco` feature")
        }
    }

    fn read_indices(&mut self, i: usize) -> Result<Vec<u32>> {
        let a = self.acc(i)?;
        self.read_index_acc(i, &a)
    }

    /// Index accessor `i`'s values, `a` being that accessor.
    fn read_index_acc(&mut self, i: usize, a: &Acc) -> Result<Vec<u32>> {
        ensure!(
            a.n == 1 && matches!(a.ctype, 5121 | 5123 | 5125),
            "glTF index accessor {i} isn't unsigned integer scalars"
        );
        self.read_u32s(a)
    }

    fn read_u32s(&mut self, a: &Acc) -> Result<Vec<u32>> {
        let count = a.count;
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| invalid!("glTF accessor counts {count} elements"))?;
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
            rows: 1,
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
        let nodes = self.nodes;
        let views: Vec<NodeView<'_>> = nodes.iter().map(NodeView::of).collect();
        let mut world = vec![None::<M4>; nodes.len()];
        // Parents first: every node reachable from a root takes its parent's transform.
        let mut child = vec![false; nodes.len()];
        for v in &views {
            for c in v.children() {
                if let Some(f) = child.get_mut(c) {
                    *f = true;
                }
            }
        }
        let mut stack: Vec<(usize, M4)> = (0..nodes.len()).filter(|&i| !child[i]).map(|i| (i, IDENTITY)).collect();
        while let Some((i, parent)) = stack.pop() {
            if world[i].is_some() {
                continue;
            }
            let m = mul(&parent, &self.local(i, &views[i]));
            world[i] = Some(m);
            for c in views[i].children() {
                if c < nodes.len() && world[c].is_none() {
                    stack.push((c, m));
                }
            }
        }
        let Some(order) = scene_order(self.json)? else {
            // No nodes at all: every mesh, as it is.
            return Ok((0..self.meshes.len())
                .map(|mesh| Use {
                    mesh,
                    weights: Vec::new(),
                    skin: None,
                    transforms: vec![IDENTITY],
                })
                .collect());
        };

        let mut uses: Vec<Use> = Vec::new();
        // Each use by (mesh, skin, weights' bits): a scan of every use per node is quadratic
        // in a scene of many meshes.
        let mut seen: HashMap<(usize, Option<usize>, Vec<u32>), usize> = HashMap::new();
        for i in order {
            let n = &views[i];
            let Some(mesh) = n.mesh.and_then(Value::as_u64).map(usize_of) else {
                continue;
            };
            let skin = n.skin.and_then(Value::as_u64).map(usize_of);
            let weights = match &self.posed[i].weights {
                Some(w) => w.clone(),
                None => weights_of(
                    n.weights
                        .or_else(|| self.meshes.get(mesh).and_then(|m| m.get("weights"))),
                ),
            };
            let w = world[i].unwrap_or(IDENTITY);
            let transforms = if skin.is_some() {
                // A skinned mesh is placed by its joints, not its node.
                vec![IDENTITY]
            } else {
                self.instances(n.extensions, &w)?
            };
            let key = (mesh, skin, weights.iter().map(|x| x.to_bits()).collect());
            match seen.get(&key) {
                Some(_) if skin.is_some() => {}
                Some(&u) => uses[u].transforms.extend(transforms),
                None => {
                    seen.insert(key, uses.len());
                    uses.push(Use {
                        mesh,
                        weights,
                        skin,
                        transforms,
                    });
                }
            }
        }
        // Joints need every node's transform, in the scene or not.
        self.world = world.into_iter().map(|w| w.unwrap_or(IDENTITY)).collect();
        Ok(uses)
    }

    /// A node's transforms: its own, or one per `EXT_mesh_gpu_instancing` instance.
    fn instances(&mut self, extensions: Option<&Value>, world: &M4) -> Result<Vec<M4>> {
        let Some(attrs) = extensions
            .and_then(|e| e.get("EXT_mesh_gpu_instancing"))
            .and_then(|e| e.get("attributes"))
        else {
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
        ensure!(count <= self.doc.max_triangles, "glTF node has {count} instances");
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

/// `v[key]`'s indices, without collecting them.
fn ids<'v>(v: &'v Value, key: &str) -> impl Iterator<Item = usize> + 'v {
    items(v, key).iter().filter_map(Value::as_u64).map(usize_of)
}

/// `node[key]` as indices.
fn list_of(v: &Value, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(usize_of).collect())
        .unwrap_or_default()
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

pub(crate) fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [0.0; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}

/// Translation, rotation (a quaternion x y z w) and scale as one matrix.
pub(crate) fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
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

/// The inverse of an affine matrix (last row 0 0 0 1), `None` when its 3x3 part is singular.
pub(crate) fn affine_inverse(m: &M4) -> Option<M4> {
    let det = det3(m);
    if !(det.abs() > 1e-30 && det.is_finite()) {
        return None;
    }
    // The 3x3's inverse is its cofactors, transposed, over the determinant.
    let a = |r: usize, c: usize| m[c * 4 + r];
    let mut out = IDENTITY;
    for r in 0..3 {
        for c in 0..3 {
            let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
            let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
            // Cofactor (r, c) goes to (c, r).
            out[r * 4 + c] = (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1)) / det;
        }
    }
    for r in 0..3 {
        out[12 + r] = -(0..3).map(|k| out[k * 4 + r] * m[12 + k]).sum::<f64>();
    }
    Some(out)
}

/// An affine matrix as translation, rotation (a unit quaternion x y z w) and scale, a mirror
/// taken as a negative X scale. Shear, which glTF's TRS can't hold, is dropped.
pub(crate) fn decompose(m: &M4) -> ([f64; 3], [f64; 4], [f64; 3]) {
    let t = [m[12], m[13], m[14]];
    let col = |c: usize| [m[c * 4], m[c * 4 + 1], m[c * 4 + 2]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut s = [len(col(0)), len(col(1)), len(col(2))];
    if det3(m) < 0.0 {
        s[0] = -s[0];
    }
    if s.iter().any(|&x| !(x.abs() > 1e-30 && x.is_finite())) {
        return (t, [0.0, 0.0, 0.0, 1.0], s.map(|x| if x.is_finite() { x } else { 0.0 }));
    }
    // The rotation's rows and columns: each column over its scale.
    let r = |row: usize, c: usize| m[c * 4 + row] / s[c];
    let trace = r(0, 0) + r(1, 1) + r(2, 2);
    let q = if trace > 0.0 {
        let k = 0.5 / (trace + 1.0).sqrt();
        [
            (r(2, 1) - r(1, 2)) * k,
            (r(0, 2) - r(2, 0)) * k,
            (r(1, 0) - r(0, 1)) * k,
            0.25 / k,
        ]
    } else if r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2) {
        let k = 2.0 * (1.0 + r(0, 0) - r(1, 1) - r(2, 2)).max(0.0).sqrt();
        [
            0.25 * k,
            (r(0, 1) + r(1, 0)) / k,
            (r(0, 2) + r(2, 0)) / k,
            (r(2, 1) - r(1, 2)) / k,
        ]
    } else if r(1, 1) > r(2, 2) {
        let k = 2.0 * (1.0 + r(1, 1) - r(0, 0) - r(2, 2)).max(0.0).sqrt();
        [
            (r(0, 1) + r(1, 0)) / k,
            0.25 * k,
            (r(1, 2) + r(2, 1)) / k,
            (r(0, 2) - r(2, 0)) / k,
        ]
    } else {
        let k = 2.0 * (1.0 + r(2, 2) - r(0, 0) - r(1, 1)).max(0.0).sqrt();
        [
            (r(0, 2) + r(2, 0)) / k,
            (r(1, 2) + r(2, 1)) / k,
            0.25 * k,
            (r(1, 0) - r(0, 1)) / k,
        ]
    };
    let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let q = if n > 0.0 && n.is_finite() {
        q.map(|c| c / n)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    (t, q, s)
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

/// A mesh's vertex positions, normals and tangents, to move together.
type Frames<'a> = (&'a mut [[f32; 3]], &'a mut [[f32; 3]], &'a mut [[f32; 4]]);

/// Moves `range` of the vertices by `m`, their normals by its normal matrix `nm` and their
/// tangents by `m` itself. A matrix that moves nothing leaves them exactly as they are, as an
/// unskinned mesh's are, but for normals and tangents a morph changed (`morphed`), which are
/// renormalized.
fn transform(
    m: &M4,
    nm: &[[f64; 3]; 3],
    morphed: bool,
    (positions, normals, tangents): Frames<'_>,
    range: std::ops::Range<usize>,
) {
    let still = m[..12] == IDENTITY[..12] && m[12..15] == [0.0; 3];
    if !still {
        for p in &mut positions[range.clone()] {
            *p = apply_point(m, *p);
        }
    }
    let end = range.end.min(tangents.len());
    if (morphed || !still) && range.start < end {
        let flip = det3(m) < 0.0;
        for t in &mut tangents[range.start..end] {
            *t = apply_tangent(m, *t, flip);
        }
    }
    let end = range.end.min(normals.len());
    if (morphed || !still) && range.start < end {
        // Single precision: within a unit in the last place of double, at a fraction of the cost.
        // A block at a time, with no branch in the loop (one stops it vectorizing, for a quarter
        // of the speed); a block where single precision couldn't square a normal is redone. Rows
        // written out, as in `apply_point`.
        let n = nm.map(|r| r.map(f32_of));
        let turn = |[x, y, z]: [f32; 3]| {
            [
                n[0][0] * x + n[0][1] * y + n[0][2] * z,
                n[1][0] * x + n[1][1] * y + n[1][2] * z,
                n[2][0] * x + n[2][1] * y + n[2][2] * z,
            ]
        };
        let fits = |len2: f32| len2 > 0.0 && len2.is_finite();
        let normals = &mut normals[range.start..end];
        if normals.len() < 8 {
            // A few normals, as a skinned vertex's often are: one at a time, each with the same
            // operations the blocks below make.
            for nv in normals {
                let [x, y, z] = turn(*nv);
                let len2 = x * x + y * y + z * z;
                *nv = if fits(len2) {
                    let inv = 1.0 / len2.sqrt();
                    [x * inv, y * inv, z * inv]
                } else {
                    apply_normal(nm, *nv)
                };
            }
            return;
        }
        for block in normals.chunks_mut(64) {
            let mut kept = [[0.0; 3]; 64];
            kept[..block.len()].copy_from_slice(block);
            let mut all = true;
            for nv in block.iter_mut() {
                let [x, y, z] = turn(*nv);
                let len2 = x * x + y * y + z * z;
                all &= fits(len2);
                let inv = 1.0 / len2.sqrt();
                *nv = [x * inv, y * inv, z * inv];
            }
            if !all {
                for (nv, &k) in block.iter_mut().zip(&kept) {
                    let o = turn(k);
                    if !fits(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) {
                        // Too small or large for single precision to square: double's answer.
                        *nv = apply_normal(nm, k);
                    }
                }
            }
        }
    }
}

fn apply_point(m: &M4, p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p.map(f64::from);
    // Row by row, written out: built with `map`, the result shuffles through registers at a
    // third of the speed.
    [
        f32_of(m[0] * x + m[4] * y + m[8] * z + m[12]),
        f32_of(m[1] * x + m[5] * y + m[9] * z + m[13]),
        f32_of(m[2] * x + m[6] * y + m[10] * z + m[14]),
    ]
}

fn apply_normal(n: &[[f64; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v.map(f64::from);
    // Written out, as in `apply_point`.
    let (a, b, c) = (
        n[0][0] * x + n[0][1] * y + n[0][2] * z,
        n[1][0] * x + n[1][1] * y + n[1][2] * z,
        n[2][0] * x + n[2][1] * y + n[2][2] * z,
    );
    let len = (a * a + b * b + c * c).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        [f32_of(a * inv), f32_of(b * inv), f32_of(c * inv)]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// A tangent turned by `m` (its upper 3x3, as surface directions turn) and made unit length;
/// its handedness flips under a mirroring `m` (`flip`), as the triangles' winding does.
fn apply_tangent(m: &M4, t: [f32; 4], flip: bool) -> [f32; 4] {
    let [x, y, z] = [t[0], t[1], t[2]].map(f64::from);
    let (a, b, c) = (
        m[0] * x + m[4] * y + m[8] * z,
        m[1] * x + m[5] * y + m[9] * z,
        m[2] * x + m[6] * y + m[10] * z,
    );
    let w = if (t[3] < 0.0) == flip { 1.0 } else { -1.0 };
    let len = (a * a + b * b + c * c).sqrt();
    if len > 0.0 && len.is_finite() {
        let inv = 1.0 / len;
        [f32_of(a * inv), f32_of(b * inv), f32_of(c * inv), w]
    } else {
        [1.0, 0.0, 0.0, w]
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
    tangents: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    uv_sets: Vec<Vec<[f32; 2]>>,
    colors: Vec<[f32; 4]>,
    color_sets: Vec<Vec<[f32; 4]>>,
    draws: Draws,
    joints: Vec<u32>,
    /// Each vertex's four strongest joints as nodes (`u32::MAX` for none), and their weights.
    skin_joints: Vec<[u32; 4]>,
    skin_weights: Vec<[f32; 4]>,
}

/// What a primitive draws on its vertices, as indices into them.
#[derive(Default)]
struct Draws {
    /// Three to a triangle.
    triangles: Vec<u32>,
    /// Two to a segment.
    lines: Vec<u32>,
    points: Vec<u32>,
}

impl Draws {
    fn is_empty(&self) -> bool {
        self.triangles.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }
}

/// Whether glTF primitive `p` is drawn on the same vertices as `q`: the same attributes,
/// material, morph targets and extensions, and no Draco stream of its own.
fn same_vertices(p: &Value, q: &Value) -> bool {
    let draco = |v: &Value| at(v, &["extensions", "KHR_draco_mesh_compression"]).is_some();
    ["attributes", "material", "targets", "extensions"]
        .iter()
        .all(|k| p.get(k) == q.get(k))
        && !draco(p)
        && !draco(q)
}

/// Line segments, in pairs, from a primitive's indices, by mode (1 lines, 2 loop, 3 strip), per
/// the glTF 2.0 specification's section 3.7.2.1.
fn segments(mode: u64, idx: Vec<u32>) -> Vec<u32> {
    match mode {
        2 | 3 => {
            let n = idx.len();
            let mut out = Vec::with_capacity(n * 2);
            for w in idx.windows(2) {
                out.extend_from_slice(w);
            }
            if mode == 2 && n >= 2 {
                out.extend_from_slice(&[idx[n - 1], idx[0]]);
            }
            out
        }
        _ => {
            let mut idx = idx;
            idx.truncate(idx.len() / 2 * 2);
            idx
        }
    }
}

/// Keeps the groups of `N` indices in `idx` that `keep` takes, in order.
fn retain_groups<const N: usize>(idx: &mut Vec<u32>, keep: impl Fn(&[u32; N]) -> bool) {
    let mut w = 0;
    for g in 0..idx.len() / N {
        let group: [u32; N] = std::array::from_fn(|k| idx[g * N + k]);
        if keep(&group) {
            idx[w..w + N].copy_from_slice(&group);
            w += N;
        }
    }
    idx.truncate(w);
}

/// What [`Reader::skin`] gives each vertex: its strongest joint's node, and its four strongest
/// joints' nodes (`u32::MAX` for none) and weights, summing to 1.
#[derive(Default)]
struct Skinned {
    strongest: Vec<u32>,
    joints: Vec<[u32; 4]>,
    weights: Vec<[f32; 4]>,
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
        // Room for every primitive and material up front: a scene of thousands of meshes
        // otherwise copies these, a primitive or material at a time, as they grow.
        let prims: usize = uses
            .iter()
            .map(|u| self.meshes.get(u.mesh).map_or(0, |m| items(m, "primitives").len()))
            .sum();
        mesh.prims.reserve(prims);
        mesh.materials.reserve(self.materials.len().min(prims) + 1);
        for u in &uses {
            // The primitive last drawn, and its place in the mesh: points and lines on the same
            // vertices join it.
            let mut last: Option<(&Value, usize)> = None;
            for p in items(self.get("meshes", u.mesh)?, "primitives") {
                if let Some((q, at)) = last
                    && mode(p) < 4
                    && same_vertices(p, q)
                {
                    self.join(p, u, &mut mesh.prims[at])?;
                    continue;
                }
                let before = mesh.prims.len();
                self.primitive(p, u, &mut mesh)?;
                last = (mesh.prims.len() > before).then_some((p, before));
            }
        }
        self.skeleton(&mut mesh);
        Ok(mesh.prims.iter().any(|p| !p.is_empty()).then_some(mesh))
    }

    /// See [`Document::bounds`].
    fn bounds(&mut self) -> Result<Option<Bounds>> {
        let uses = self.uses()?;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for u in &uses {
            for p in items(self.get("meshes", u.mesh)?, "primitives") {
                let mode = mode(p);
                let attrs = p.get("attributes").unwrap_or(&Value::Null);
                let Some(pos) = index(attrs, "POSITION") else {
                    continue;
                };
                if !self.drawn(mode) {
                    continue;
                }
                self.draco.clear();
                if let Some(ext) = p.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")) {
                    if mode < 4 {
                        // Not drawn: see `primitive`.
                        continue;
                    }
                    #[cfg(feature = "draco")]
                    self.decode_draco(ext, attrs, index(p, "indices"))?;
                    #[cfg(not(feature = "draco"))]
                    {
                        let _ = ext;
                        ensure!(
                            self.acc(pos)?.view.is_some(),
                            "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
                        );
                    }
                }
                let mut positions = self.read(pos, [0.0; 3])?;
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
                }
                if let Some(skin) = u.skin {
                    let morphed = !u.weights.is_empty();
                    self.skin(skin, attrs, &mut positions, &mut [], &mut [], morphed)?;
                }
                for m in &u.transforms {
                    let still = m[..12] == IDENTITY[..12] && m[12..15] == [0.0; 3];
                    for &v in &positions {
                        let v = if still { v } else { apply_point(m, v) };
                        if v.iter().all(|x| x.is_finite()) {
                            for c in 0..3 {
                                lo[c] = lo[c].min(v[c]);
                                hi[c] = hi[c].max(v[c]);
                            }
                        }
                    }
                }
            }
        }
        Ok((lo[0] <= hi[0]).then_some(Bounds { min: lo, max: hi }))
    }

    /// Whether a primitive in `mode` is drawn: triangles, strips and fans, and points and lines
    /// when the document draws them.
    fn drawn(&self, mode: u64) -> bool {
        matches!(mode, 4..=6) || (mode < 4 && self.doc.lines_and_points)
    }

    /// Checks that a primitive with index accessor `indices` and position accessor `pos`, in
    /// `mode`, drawn `k` times, stays within what may still be drawn, before anything is read:
    /// what its accessors declare is counted. Gives the accessor it counted, to be read next
    /// rather than looked up again: the indices', or else the positions'.
    fn afford(&self, indices: Option<usize>, pos: usize, mode: u64, k: usize) -> Result<Acc> {
        let counted = self.acc(indices.unwrap_or(pos))?;
        let corners = counted.count;
        let n = match mode {
            0 | 2 => corners,
            1 => corners / 2,
            3 => corners.saturating_sub(1),
            4 => corners / 3,
            _ => corners.saturating_sub(2),
        };
        let total = n.saturating_mul(k);
        if mode < 4 {
            ensure!(
                total <= self.point_budget,
                "too many points and lines to convert (more than {})",
                self.doc.max_points
            );
        } else {
            ensure!(
                total <= self.budget,
                "too many triangles to convert (more than {})",
                self.doc.max_triangles
            );
        }
        Ok(counted)
    }

    /// What primitive `p` draws in `mode` on its `n` vertices: triangles, line segments or
    /// points. Those that name a vertex that doesn't exist are dropped, and segments from a
    /// vertex to itself.
    fn draws(&mut self, indices: Option<(usize, &Acc)>, mode: u64, n: usize) -> Result<Draws> {
        let idx = match indices {
            Some((i, a)) => self.read_index_acc(i, a)?,
            None => (0..u32_of(n)).collect(),
        };
        let fits = |i: u32| (i as usize) < n;
        let mut d = Draws::default();
        match mode {
            0 => {
                d.points = idx;
                d.points.retain(|&i| fits(i));
            }
            1..=3 => {
                d.lines = segments(mode, idx);
                retain_groups(&mut d.lines, |&[a, b]: &[u32; 2]| a != b && fits(a) && fits(b));
            }
            _ => {
                d.triangles = triangles(mode, idx);
                if d.triangles.iter().copied().max().is_some_and(|m| !fits(m)) {
                    // Drop triangles that name vertices that don't exist.
                    retain_groups(&mut d.triangles, |t: &[u32; 3]| t.iter().all(|&i| fits(i)));
                }
            }
        }
        Ok(d)
    }

    /// Takes what `d`, drawn `k` times, draws from what may still be drawn.
    fn charge(&mut self, d: &Draws, k: usize) {
        self.budget -= d.triangles.len() / 3 * k;
        self.point_budget -= (d.lines.len() / 2 + d.points.len()) * k;
    }

    /// Adds the points or lines of primitive `p`, drawn on the same vertices `prim` was made
    /// from, to `prim`.
    fn join(&mut self, p: &Value, u: &Use, prim: &mut Prim) -> Result<()> {
        let mode = mode(p);
        let Some(pos) = p.get("attributes").and_then(|a| index(a, "POSITION")) else {
            return Ok(());
        };
        if !self.drawn(mode) {
            return Ok(());
        }
        let k = u.transforms.len();
        let indices = index(p, "indices");
        let counted = self.afford(indices, pos, mode, k)?;
        let n = prim.positions.len() / k.max(1);
        let d = self.draws(indices.map(|i| (i, &counted)), mode, n)?;
        self.charge(&d, k);
        extend(prim, &d, n, &u.transforms);
        Ok(())
    }

    /// The mesh's skeleton: every node a vertex is bound to, and their ancestors, parents first,
    /// where they are in the pose drawn. Each primitive's joints, nodes until now, become bones.
    fn skeleton(&self, mesh: &mut Mesh) {
        let nodes = self.nodes;
        let mut used = vec![false; nodes.len()];
        for p in mesh.prims.iter().filter(|p| p.skinned()) {
            for &j in p.skin_joints.as_flattened() {
                if let Some(u) = used.get_mut(j as usize) {
                    *u = true;
                }
            }
        }
        // Each node's parent: the first node listing it as a child, as `Document::rig` has it.
        let mut parent = vec![None; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for c in ids(n, "children") {
                if c < nodes.len() && c != i && parent[c].is_none() {
                    parent[c] = Some(i);
                }
            }
        }
        // Ancestors too, so the bones hang together as the nodes do. Bounded, against cycles.
        for i in 0..nodes.len() {
            if !used[i] {
                continue;
            }
            let mut at = parent[i];
            for _ in 0..nodes.len() {
                let Some(p) = at else { break };
                if used[p] {
                    break;
                }
                used[p] = true;
                at = parent[p];
            }
        }
        // Parents first: down from each bone whose parent isn't one; a cycle's nodes, with no
        // such bone above them, become roots after.
        let mut kids = vec![Vec::new(); nodes.len()];
        for i in (0..nodes.len()).filter(|&i| used[i]) {
            if let Some(p) = parent[i].filter(|&p| used[p]) {
                kids[p].push(i);
            }
        }
        let mut bone_of = vec![None::<usize>; nodes.len()];
        let mut order = Vec::new();
        let roots: Vec<usize> = (0..nodes.len())
            .filter(|&i| used[i] && parent[i].is_none_or(|p| !used[p]))
            .collect();
        let cyclic: Vec<usize> = (0..nodes.len()).filter(|&i| used[i]).collect();
        for root in roots.into_iter().chain(cyclic) {
            if bone_of[root].is_some() {
                continue;
            }
            let mut stack = vec![root];
            while let Some(i) = stack.pop() {
                if bone_of[i].is_some() {
                    continue;
                }
                bone_of[i] = Some(order.len());
                order.push(i);
                stack.extend(kids[i].iter().rev().copied());
            }
        }
        mesh.skeleton = order
            .iter()
            .map(|&i| crate::mesh::Bone {
                name: nodes[i].get("name").and_then(Value::as_str).map(str::to_owned),
                parent: parent[i]
                    .and_then(|p| bone_of[p])
                    .filter(|&b| bone_of[i].is_some_and(|me| b < me)),
                node: i,
                matrix: self
                    .bound
                    .get(i)
                    .copied()
                    .flatten()
                    .or_else(|| self.world.get(i).copied())
                    .unwrap_or(IDENTITY)
                    .map(f32_of),
            })
            .collect();
        // Each node's bone, as a flat table: one load per joint, no branches.
        let bone: Vec<u32> = bone_of.iter().map(|b| b.map_or(0, u32_of)).collect();
        for p in &mut mesh.prims {
            if mesh.skeleton.is_empty() || !p.skinned() {
                p.skin_joints = Vec::new();
                p.skin_weights = Vec::new();
                continue;
            }
            // Slots no joint fills keep weight 0 and name the first bone.
            for (j, w) in p.skin_joints.iter_mut().zip(&p.skin_weights) {
                *j = std::array::from_fn(|k| {
                    let b = bone.get(j[k] as usize).copied().unwrap_or(0);
                    if w[k] > 0.0 { b } else { 0 }
                });
            }
        }
    }

    /// A gaussian splat primitive's colors when it doesn't give them (`KHR_gaussian_splatting`
    /// asks for a `COLOR_0`, but doesn't require one): each splat's base color and opacity, as
    /// the extension works them out. Empty for other primitives.
    fn splat_colors(&mut self, p: &Value, attrs: &Value) -> Result<Vec<[f32; 4]>> {
        let Some(ext) = at(p, &["extensions", "KHR_gaussian_splatting"]) else {
            return Ok(Vec::new());
        };
        let (Some(dc), Some(opacity)) = (
            index(attrs, crate::gaussian::SH0),
            index(attrs, crate::gaussian::OPACITY),
        ) else {
            return Ok(Vec::new());
        };
        let srgb = ext.get("colorSpace").and_then(Value::as_str) != Some("lin_rec709_display");
        let dc = self.read(dc, [0.0; 3])?;
        let opacity = self.read(opacity, [1.0])?;
        if dc.len() != opacity.len() {
            return Ok(Vec::new());
        }
        let fallback = |(&d, &[o]): (&[f32; 3], &[f32; 1])| crate::gaussian::fallback(d, o, srgb);
        Ok(dc.iter().zip(&opacity).map(fallback).collect())
    }

    fn primitive(&mut self, p: &Value, u: &Use, mesh: &mut Mesh) -> Result<()> {
        let mode = mode(p);
        let attrs = p.get("attributes").unwrap_or(&Value::Null);
        let Some(pos) = index(attrs, "POSITION") else {
            return Ok(());
        };
        if !self.drawn(mode) {
            return Ok(());
        }

        self.draco.clear();
        if let Some(ext) = p.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")) {
            if mode < 4 {
                // Draco point clouds and lines aren't decoded.
                return Ok(());
            }
            #[cfg(feature = "draco")]
            self.decode_draco(ext, attrs, index(p, "indices"))?;
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

        // Check the count before reading any vertices.
        let indices = index(p, "indices");
        let counted = self.afford(indices, pos, mode, u.transforms.len())?;

        let mut positions = match indices {
            Some(_) => self.read(pos, [0.0; 3])?,
            None => self.read_acc(&counted, [0.0; 3])?,
        };
        let n = positions.len();
        let draws = self.draws(indices.map(|i| (i, &counted)), mode, n)?;
        if draws.is_empty() {
            return Ok(());
        }

        let material = index(p, "material");
        let (set, uv_matrix) = self.uv_set(material);
        let attr = |name: &str| index(attrs, name);
        let mut normals = match attr("NORMAL") {
            Some(a) => self.read(a, [0.0; 3])?,
            None => Vec::new(),
        };
        let mut uvs = match attr(&texcoord(set)) {
            Some(a) => self.read(a, [0.0; 2])?,
            None => Vec::new(),
        };
        let colors = match attr("COLOR_0") {
            Some(a) => self.read(a, [0.0, 0.0, 0.0, 1.0])?,
            None => self.splat_colors(p, attrs)?,
        };
        // Attributes that don't match the positions are dropped rather than trusted.
        if normals.len() != n {
            normals = Vec::new();
        }
        if uvs.len() != n {
            uvs = Vec::new();
        }
        let colors = if colors.len() == n { colors } else { Vec::new() };
        // Tangents mean nothing without the normals they're perpendicular to.
        let mut tangents = match attr("TANGENT") {
            Some(a) if !normals.is_empty() => self.read(a, [1.0, 0.0, 0.0, 1.0])?,
            _ => Vec::new(),
        };
        if tangents.len() != n {
            tangents = Vec::new();
        }
        // The other sets: one that can't be read, or doesn't fit, is dropped, not the model.
        let uv_sets = numbered(attrs, "TEXCOORD_", |k| k != set, |a| self.read(a, [0.0; 2]).ok(), n);
        let color_sets = numbered(
            attrs,
            "COLOR_",
            |k| k != 0,
            |a| self.read(a, [0.0, 0.0, 0.0, 1.0]).ok(),
            n,
        );
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
            if !tangents.is_empty()
                && let Some(a) = index(target, "TANGENT")
            {
                for (v, d) in tangents.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
        }
        let mut skinned = Skinned::default();
        let morphed = !u.weights.is_empty();
        if let Some(skin) = u.skin {
            skinned = self.skin(skin, attrs, &mut positions, &mut normals, &mut tangents, morphed)?;
        } else if morphed {
            normals.iter_mut().for_each(normalize);
            for t in &mut tangents {
                *t = apply_tangent(&IDENTITY, *t, false);
            }
        }

        let g = Geometry {
            positions,
            normals,
            tangents,
            uvs,
            uv_sets,
            colors,
            color_sets,
            draws,
            joints: skinned.strongest,
            skin_joints: skinned.joints,
            skin_weights: skinned.weights,
        };
        let material = self.material(material, mesh);
        self.charge(&g.draws, u.transforms.len());
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
        let bytes = self.view_shared(view)?;
        let limits = DecodeLimits::default()
            .with_max_faces(self.budget as u64)
            .with_max_points(self.max_vertices as u64);
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
    /// weight over every `JOINTS_n`/`WEIGHTS_n` set, not just the first four. Gives each vertex's
    /// joints, or nothing if the weights can't be read. `morphed` normals are renormalized even
    /// where the joints don't move them.
    fn skin(
        &mut self,
        skin: usize,
        attrs: &Value,
        positions: &mut [[f32; 3]],
        normals: &mut [[f32; 3]],
        tangents: &mut [[f32; 4]],
        morphed: bool,
    ) -> Result<Skinned> {
        let s = self.get("skins", skin)?;
        let joints = list_of(s, "joints");
        let ibm = match index(s, "inverseBindMatrices") {
            Some(a) => self.read(a, IDENTITY.map(f32_of))?,
            None => Vec::new(),
        };
        if self.bind {
            if self.bound.is_empty() {
                self.bound = vec![None; self.posed.len()];
            }
            for (k, &j) in joints.iter().enumerate() {
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                if let Some(slot @ None) = self.bound.get_mut(j) {
                    *slot = Some(affine_inverse(&b).unwrap_or(IDENTITY));
                }
            }
        }
        let mats: Vec<M4> = joints
            .iter()
            .enumerate()
            .map(|(k, &j)| {
                let w = self.world.get(j).copied().unwrap_or(IDENTITY);
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                mul(&w, &b)
            })
            .collect();
        // Every set of four influences, JOINTS_0/WEIGHTS_0 then JOINTS_1/WEIGHTS_1 and on, as
        // long as both halves are there and fit the vertices. Joint indices are integers, read as
        // floats exactly.
        let mut sets = Vec::new();
        for n in 0.. {
            let (Some(ja), Some(wa)) = (
                index(attrs, &format!("JOINTS_{n}")),
                index(attrs, &format!("WEIGHTS_{n}")),
            ) else {
                break;
            };
            let ji = self.read(ja, [0.0; 4])?;
            let wi = self.read(wa, [0.0; 4])?;
            if ji.len() != positions.len() || wi.len() != positions.len() {
                break;
            }
            sets.push((ji, wi));
        }
        if sets.is_empty() {
            return Ok(Skinned::default());
        }
        // A skin's joint index as its node, `None` past the skin's joints.
        let node = |j: f32| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "joint indices are small whole numbers; `as` saturates bad ones and `get` refuses them"
            )]
            joints.get(j as usize).copied()
        };
        let n = positions.len();
        let mut out = Skinned {
            strongest: Vec::with_capacity(n),
            joints: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
        };
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "joint indices are small whole numbers; `as` saturates bad ones and `get` refuses them"
        )]
        let joint = |j: f32| j as usize;
        // Each joint's normal matrix, made the first time a vertex follows that joint alone.
        let mut rigid: Vec<Option<[[f64; 3]; 3]>> = vec![None; mats.len()];
        let unmoved = normal_matrix(&IDENTITY);
        // The previous vertex's influences and blend, reused when the next one's are the same:
        // neighbouring vertices often share their weights.
        let mut last: Vec<(f32, f32)> = Vec::with_capacity(4 * sets.len());
        let (mut m, mut nm) = (IDENTITY, unmoved);
        // Vertices `start..v` all go through `m` (when `moving`; they stay put otherwise). They
        // are moved together when it changes, in one tight loop rather than one at a time.
        let (mut start, mut moving) = (0, false);
        // In the bind pose nothing moves; morphed normals are still renormalized, below.
        let bind = self.bind;
        let bits = |a: [f32; 4]| a.map(f32::to_bits);
        // One vertex's influences, (joint index, weight), reused.
        let mut inf: Vec<(f32, f32)> = Vec::with_capacity(4 * sets.len());
        // The previous vertex's output, kept at hand for the next one that repeats it.
        let mut prev = (u32::MAX, [u32::MAX; 4], [0.0f32; 4]);
        for v in 0..n {
            // The same rows as the previous vertex: everything it worked out holds for this one.
            if v > 0
                && sets
                    .iter()
                    .all(|(ji, wi)| bits(ji[v]) == bits(ji[v - 1]) && bits(wi[v]) == bits(wi[v - 1]))
            {
                out.strongest.push(prev.0);
                out.joints.push(prev.1);
                out.weights.push(prev.2);
                continue;
            }
            inf.clear();
            for (ji, wi) in &sets {
                for k in 0..4 {
                    let (j, w) = (ji[v][k], wi[v][k]);
                    // Negative and NaN weights move nothing.
                    if w > 0.0 {
                        match inf.iter_mut().find(|(i, _)| i.to_bits() == j.to_bits()) {
                            Some((_, x)) => *x += w,
                            None => inf.push((j, w)),
                        }
                    }
                }
            }
            // Strongest first, the earlier on a tie, as the file lists them. Weights here are
            // positive, so `<` orders them; a few of them sort fastest by insertion.
            for i in 1..inf.len() {
                let mut k = i;
                while k > 0 && inf[k - 1].1 < inf[k].1 {
                    inf.swap(k - 1, k);
                    k -= 1;
                }
            }
            prev.0 = inf.first().and_then(|&(j, _)| node(j)).map_or(u32::MAX, u32_of);
            out.strongest.push(prev.0);
            // The four strongest that name a joint, for binding the vertex back to them.
            let mut top = [(u32::MAX, 0.0f32); 4];
            let mut slots = top.iter_mut();
            for &(j, w) in &inf {
                if let Some(nd) = node(j) {
                    let Some(slot) = slots.next() else { break };
                    *slot = (u32_of(nd), w);
                }
            }
            let kept: f32 = top.iter().map(|t| t.1).sum();
            prev.1 = top.map(|t| t.0);
            prev.2 = if kept > 0.0 { top.map(|t| t.1 / kept) } else { [0.0; 4] };
            out.joints.push(prev.1);
            out.weights.push(prev.2);

            let sum: f32 = inf.iter().map(|t| t.1).sum();
            if bind || !(sum > 0.0 && sum.is_finite()) {
                if moving {
                    transform(&m, &nm, morphed, (positions, normals, tangents), start..v);
                    moving = false;
                }
                continue;
            }
            let same = inf.len() == last.len()
                && inf
                    .iter()
                    .zip(&last)
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits());
            if !(same && moving) {
                if moving {
                    transform(&m, &nm, morphed, (positions, normals, tangents), start..v);
                }
                (start, moving) = (v, true);
            }
            if !same {
                if let [(j, _)] = inf[..] {
                    // One joint: its matrix as it is (its weight over the sum is exactly 1).
                    let j = joint(j);
                    m = mats.get(j).copied().unwrap_or(IDENTITY);
                    nm = match rigid.get_mut(j) {
                        Some(slot) => *slot.get_or_insert_with(|| normal_matrix(&m)),
                        None => unmoved,
                    };
                } else if let Some(first) = inf
                    .iter()
                    .map(|&(j, _)| mats.get(joint(j)).unwrap_or(&IDENTITY))
                    .try_fold(None::<&M4>, |seen, jm| match seen {
                        #[allow(clippy::float_cmp, reason = "the very same matrix, bit for bit, is what's sought")]
                        Some(s) if s != jm => Err(()),
                        _ => Ok(Some(jm)),
                    })
                    .ok()
                    .flatten()
                {
                    // Joints that all sit the same way: their blend is that matrix exactly,
                    // which rounding the weights' sum would miss.
                    m = *first;
                    nm = normal_matrix(&m);
                } else {
                    // Only the upper three rows: the bottom one is never read.
                    m = [0.0; 16];
                    for &(j, w) in &inf {
                        let jm = mats.get(joint(j)).unwrap_or(&IDENTITY);
                        let f = f64::from(w / sum);
                        for c in 0..4 {
                            for r in 0..3 {
                                m[c * 4 + r] += f * jm[c * 4 + r];
                            }
                        }
                    }
                    nm = normal_matrix(&m);
                }
                last.clone_from(&inf);
            }
        }
        if moving {
            transform(&m, &nm, morphed, (positions, normals, tangents), start..n);
        }
        if bind && morphed {
            normals.iter_mut().for_each(normalize);
            for t in tangents.iter_mut() {
                *t = apply_tangent(&IDENTITY, *t, false);
            }
        }
        Ok(out)
    }

    /// The texture coordinate set a material's color texture reads, and its
    /// `KHR_texture_transform` as a 2x3 matrix (row-major).
    fn uv_set(&self, material: Option<usize>) -> (u64, Option<[f32; 6]>) {
        material
            .and_then(|m| self.materials_json.get(m))
            .and_then(color_texture)
            .map_or((0, None), mapping)
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
            color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            ..Material::plain("", [1.0; 4])
        };
        if let Some(d) = i.and_then(|i| self.materials_json.get(i)) {
            d.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .clone_into(&mut m.name);
            let ext = d.get("extensions");
            let sg = ext.and_then(|e| e.get("KHR_materials_pbrSpecularGlossiness"));
            match (d.get("pbrMetallicRoughness"), sg) {
                (None, Some(sg)) => {
                    m.color = f32s(sg.get("diffuseFactor"), [1.0; 4]);
                    m.metallic = 0.0;
                    m.roughness = 1.0 - f32_of(sg.get("glossinessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
                (pbr, _) => {
                    let pbr = pbr.unwrap_or(&Value::Null);
                    m.color = f32s(pbr.get("baseColorFactor"), [1.0; 4]);
                    m.metallic = f32_of(pbr.get("metallicFactor").and_then(Value::as_f64).unwrap_or(1.0));
                    m.roughness = f32_of(pbr.get("roughnessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
            }
            let strength = ext
                .and_then(|e| at(e, &["KHR_materials_emissive_strength", "emissiveStrength"]))
                .and_then(Value::as_f64)
                .map_or(1.0, f32_of);
            m.emissive = f32s(d.get("emissiveFactor"), [0.0; 3]).map(|e| e * strength);
            m.double_sided = d.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
            match d.get("alphaMode").and_then(Value::as_str) {
                Some("BLEND") => {}
                Some("MASK") => {
                    let cutoff = d.get("alphaCutoff").and_then(Value::as_f64).map_or(0.5, f32_of);
                    m.alpha_cutoff = Some(if cutoff.is_finite() { cutoff.max(0.0) } else { 0.5 });
                }
                _ => m.color[3] = 1.0,
            }
            m.unlit = ext.and_then(|e| e.get("KHR_materials_unlit")).is_some();
            // `Prim::uvs` are the color texture's coordinates with its transform baked in: the
            // other textures read them through what's left of their own transform, or read
            // another set.
            let color = color_texture(d);
            let uv = color.map_or((0, None), mapping);
            let mut take = |info: Option<&Value>| self.texture_info(info?, uv, mesh);
            let mut slot = |info: Option<&Value>, image: &mut Option<usize>, sampling: &mut Sampling| {
                if let Some(t) = take(info) {
                    *image = Some(t.image);
                    *sampling = t.sampling;
                }
            };
            let s = &mut m.sampling;
            slot(color, &mut m.color_image, &mut s.color);
            slot(d.get("normalTexture"), &mut m.normal_image, &mut s.normal);
            slot(d.get("occlusionTexture"), &mut m.occlusion_image, &mut s.occlusion);
            slot(d.get("emissiveTexture"), &mut m.emissive_image, &mut s.emissive);
            slot(
                at(d, &["pbrMetallicRoughness", "metallicRoughnessTexture"]),
                &mut m.metallic_roughness_image,
                &mut s.metallic_roughness,
            );
            if m.normal_image.is_some() {
                m.normal_scale = at(d, &["normalTexture", "scale"])
                    .and_then(Value::as_f64)
                    .map_or(1.0, f32_of);
            }
            if m.occlusion_image.is_some() {
                m.occlusion_strength = at(d, &["occlusionTexture", "strength"])
                    .and_then(Value::as_f64)
                    .map_or(1.0, f32_of);
            }
            if let Some(ext) = ext {
                m.extensions = self.extensions(ext, uv, mesh).map(Box::new);
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

    /// A texture info's image and how it's sampled, given the color texture's coordinate set and
    /// transform `color` (which [`Prim::uvs`] have baked in); `None` when the image can't be
    /// made, or the texture reads the color's coordinates through a transform that can't be
    /// written relative to the color's (a shear).
    fn texture_info(&mut self, info: &Value, color: (u64, Option<[f32; 6]>), mesh: &mut Mesh) -> Option<Texture> {
        let t = index(info, "index")?;
        let (set, own) = mapping(info);
        let (uv_set, transform) = if set != color.0 {
            (Some(usize::try_from(set).ok()?), uv_transform(info))
        } else if color.1.is_none() {
            (None, uv_transform(info))
        } else if own == color.1 {
            (None, None)
        } else {
            let relative = relative(own.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), color.1?)?;
            (None, Some(uv_decompose(relative)?))
        };
        let image = self.texture(t, mesh)?;
        let sampler = self
            .textures
            .get(t)
            .and_then(|tex| index(tex, "sampler"))
            .and_then(|s| self.samplers.get(s));
        let num = |k: &str| {
            sampler
                .and_then(|s| s.get(k))
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
        };
        Some(Texture {
            image,
            sampling: Sampling {
                uv_set,
                transform: transform.filter(|x| *x != UvTransform::IDENTITY),
                wrap: [num("wrapS").unwrap_or(10497), num("wrapT").unwrap_or(10497)],
                filter: [num("magFilter"), num("minFilter")],
            },
        })
    }

    /// The `KHR_materials_*` extensions in a material's `extensions`, with their textures;
    /// `None` when it has none of them.
    fn extensions(&mut self, ext: &Value, uv: (u64, Option<[f32; 6]>), mesh: &mut Mesh) -> Option<Extensions> {
        let num = |e: &Value, k: &str, default: f32| e.get(k).and_then(Value::as_f64).map_or(default, f32_of);
        let mut tex = |e: &Value, k: &str| e.get(k).and_then(|info| self.texture_info(info, uv, mesh));
        let get = |name: &str| ext.get(name).filter(|e| e.is_object());
        let mut out = Extensions {
            ior: get("KHR_materials_ior").map(|e| num(e, "ior", 1.5)),
            dispersion: get("KHR_materials_dispersion").map(|e| num(e, "dispersion", 0.0)),
            ..Extensions::default()
        };
        if let Some(e) = get("KHR_materials_specular") {
            out.specular = Some(Specular {
                factor: Factor {
                    factor: num(e, "specularFactor", 1.0),
                    tex: tex(e, "specularTexture"),
                },
                color: f32s(e.get("specularColorFactor"), [1.0; 3]),
                color_tex: tex(e, "specularColorTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_transmission") {
            out.transmission = Some(Factor {
                factor: num(e, "transmissionFactor", 0.0),
                tex: tex(e, "transmissionTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_clearcoat") {
            out.clearcoat = Some(Clearcoat {
                factor: Factor {
                    factor: num(e, "clearcoatFactor", 0.0),
                    tex: tex(e, "clearcoatTexture"),
                },
                roughness: Factor {
                    factor: num(e, "clearcoatRoughnessFactor", 0.0),
                    tex: tex(e, "clearcoatRoughnessTexture"),
                },
                normal: tex(e, "clearcoatNormalTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_sheen") {
            out.sheen = Some(Sheen {
                color: f32s(e.get("sheenColorFactor"), [0.0; 3]),
                color_tex: tex(e, "sheenColorTexture"),
                roughness: Factor {
                    factor: num(e, "sheenRoughnessFactor", 0.0),
                    tex: tex(e, "sheenRoughnessTexture"),
                },
            });
        }
        if let Some(e) = get("KHR_materials_volume") {
            out.volume = Some(Volume {
                thickness: Factor {
                    factor: num(e, "thicknessFactor", 0.0),
                    tex: tex(e, "thicknessTexture"),
                },
                attenuation_distance: e
                    .get("attenuationDistance")
                    .and_then(Value::as_f64)
                    .map(f32_of)
                    .filter(|d| d.is_finite() && *d > 0.0),
                attenuation_color: f32s(e.get("attenuationColor"), [1.0; 3]),
            });
        }
        if let Some(e) = get("KHR_materials_anisotropy") {
            out.anisotropy = Some(Anisotropy {
                strength: Factor {
                    factor: num(e, "anisotropyStrength", 0.0),
                    tex: tex(e, "anisotropyTexture"),
                },
                rotation: num(e, "anisotropyRotation", 0.0),
            });
        }
        if let Some(e) = get("KHR_materials_iridescence") {
            out.iridescence = Some(Iridescence {
                factor: Factor {
                    factor: num(e, "iridescenceFactor", 0.0),
                    tex: tex(e, "iridescenceTexture"),
                },
                ior: num(e, "iridescenceIor", 1.3),
                thickness: [
                    num(e, "iridescenceThicknessMinimum", 100.0),
                    num(e, "iridescenceThicknessMaximum", 400.0),
                ],
                thickness_tex: tex(e, "iridescenceThicknessTexture"),
            });
        }
        (out != Extensions::default()).then_some(out)
    }

    /// The mesh image for glTF texture `t`: its source, or failing that one an extension
    /// names (`EXT_texture_webp`, `KHR_texture_basisu`, ...), if an image can be made of it.
    fn texture(&mut self, t: usize, mesh: &mut Mesh) -> Option<usize> {
        let tex = self.textures.get(t)?;
        let mut sources: Vec<usize> = index(tex, "source").into_iter().collect();
        if let Some(ext) = tex.get("extensions").and_then(Value::as_object) {
            sources.extend(ext.values().filter_map(|e| index(e, "source")));
        }
        sources.into_iter().find_map(|s| self.image(s, mesh))
    }

    /// The mesh image for glTF image `s`, when it's PNG or JPEG, or a format a feature
    /// re-encodes (WebP, TGA, BMP, GIF, TIFF, PSD, DDS). A WebP image is kept as it is, and
    /// re-encoded only when the mesh is written.
    fn image(&mut self, s: usize, mesh: &mut Mesh) -> Option<usize> {
        if let Some(done) = *self.images.get(s)? {
            return done;
        }
        let img = self.images_json.get(s)?;
        // TGA has no signature: its MIME type, or a file name, says what it is.
        let name = match img.get("mimeType").and_then(Value::as_str) {
            Some("image/x-tga" | "image/tga" | "image/x-targa" | "image/targa") => "image.tga",
            _ => img
                .get("uri")
                .and_then(Value::as_str)
                .filter(|u| !u.starts_with("data:"))
                .unwrap_or(""),
        };
        let name = std::path::PathBuf::from(name);
        let bytes = if let Some(v) = index(img, "bufferView") {
            self.view_shared(v).ok()
        } else {
            // A file the loader didn't find stays a URI, and isn't drawn.
            img.get("uri")
                .and_then(Value::as_str)
                .filter(|u| u.starts_with("data:"))
                .and_then(|u| container::data_uri(u).ok())
                .map(|(_, b)| b.into())
        };
        let out = bytes.and_then(|b: crate::Bytes| Image::kept_named(b, &name)).map(|i| {
            mesh.images.push(i);
            mesh.images.len() - 1
        });
        self.images[s] = Some(out);
        out
    }
}

/// The node and property (`translation`, `rotation`, `scale` or `weights`) an animation
/// channel's target animates: named outright, or by a `KHR_animation_pointer` to one of them.
pub(crate) fn node_target(target: &Value) -> Option<(usize, &str)> {
    if let (Some(node), Some(path)) = (index(target, "node"), target.get("path").and_then(Value::as_str))
        && path != "pointer"
    {
        return Some((node, path));
    }
    let pointer = at(target, &["extensions", "KHR_animation_pointer", "pointer"]).and_then(Value::as_str)?;
    let (node, path) = pointer.strip_prefix("/nodes/")?.split_once('/')?;
    // An index the way JSON pointers write them: no sign, no leading zeros.
    if node.is_empty() || !node.bytes().all(|b| b.is_ascii_digit()) || (node.len() > 1 && node.starts_with('0')) {
        return None;
    }
    let path = ["translation", "rotation", "scale", "weights"]
        .into_iter()
        .find(|&p| p == path)?;
    Some((node.parse().ok()?, path))
}

/// The texture coordinate set a texture info reads, and its `KHR_texture_transform` as a 2x3
/// matrix (row-major).
fn mapping(info: &Value) -> (u64, Option<[f32; 6]>) {
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

/// A texture info's `KHR_texture_transform` as it's written; `None` without one.
fn uv_transform(info: &Value) -> Option<UvTransform> {
    let t = info.get("extensions")?.get("KHR_texture_transform")?;
    Some(UvTransform {
        offset: f32s(t.get("offset"), [0.0; 2]),
        rotation: f32_of(t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0)),
        scale: f32s(t.get("scale"), [1.0; 2]),
    })
}

/// The 2x3 transform (row-major) that takes coordinates `color` has moved to where `own` moves
/// them: `own * color^-1`; `None` when `color` collapses them.
fn relative(own: [f32; 6], color: [f32; 6]) -> Option<[f32; 6]> {
    let [a, b, e, c, d, f] = color.map(f64::from);
    let det = a * d - b * c;
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    // The inverse: its linear part, then the translation it undoes.
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    let (ie, if_) = (-(ia * e + ib * f), -(ic * e + id * f));
    let [p, q, s, r, t, u] = own.map(f64::from);
    let out = [
        p * ia + q * ic,
        p * ib + q * id,
        p * ie + q * if_ + s,
        r * ia + t * ic,
        r * ib + t * id,
        r * ie + t * if_ + u,
    ];
    #[allow(clippy::cast_possible_truncation, reason = "back to the f32s the matrices came in")]
    Some(out.map(|v| v as f32))
}

/// A 2x3 transform (row-major) as `KHR_texture_transform` writes one, `translate * rotate *
/// scale`; `None` when it shears, which that can't write.
fn uv_decompose(m: [f32; 6]) -> Option<UvTransform> {
    let [a, b, e, c, d, f] = m;
    // The first column is (cos, -sin) * scale x; the second (sin, cos) * scale y.
    let sx = a.hypot(c);
    if sx == 0.0 || !sx.is_finite() {
        return None;
    }
    let rotation = (-c).atan2(a);
    let (sin, cos) = rotation.sin_cos();
    let sy = b * sin + d * cos;
    let tolerance = 1e-4 * sy.abs().max(1.0);
    if (b - sin * sy).abs() > tolerance || (d - cos * sy).abs() > tolerance {
        return None;
    }
    Some(UvTransform {
        offset: [e, f],
        rotation,
        scale: [sx, sy],
    })
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

/// The most sets of one attribute (`TEXCOORD_n`, `COLOR_n`) read: past any real model's, and a
/// bound on what a hostile name like `TEXCOORD_4000000000` could make room for.
const MAX_SETS: usize = 64;

/// A primitive's numbered attribute sets (`prefix` then the number), each at its number, for
/// the numbers `want` picks: a set that `read` can't read or that doesn't have `n` elements is
/// left empty, and so are the gaps. Empty when no set is kept.
fn numbered<T>(
    attrs: &Value,
    prefix: &str,
    want: impl Fn(u64) -> bool,
    mut read: impl FnMut(usize) -> Option<Vec<T>>,
    n: usize,
) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = Vec::new();
    let Some(attrs) = attrs.as_object() else { return out };
    for (name, a) in attrs {
        let Some(k) = name.strip_prefix(prefix).and_then(|k| k.parse::<u64>().ok()) else {
            continue;
        };
        let (Some(a), Ok(slot)) = (a.as_u64().and_then(|a| usize::try_from(a).ok()), usize::try_from(k)) else {
            continue;
        };
        if slot >= MAX_SETS || !want(k) {
            continue;
        }
        if let Some(v) = read(a).filter(|v| v.len() == n) {
            if out.len() <= slot {
                out.resize_with(slot + 1, Vec::new);
            }
            out[slot] = v;
        }
    }
    out
}

/// A primitive's geometry at each of `transforms`, as one primitive.
#[allow(clippy::float_cmp_const, reason = "only the exact identity may skip the transform")]
fn place(mut g: Geometry, transforms: &[M4], material: usize) -> Prim {
    if let [t] = transforms {
        // One transform: in place, rather than into new buffers.
        if *t != IDENTITY {
            for v in &mut g.positions {
                *v = apply_point(t, *v);
            }
            let nm = normal_matrix(t);
            for v in &mut g.normals {
                *v = apply_normal(&nm, *v);
            }
            let flip = det3(t) < 0.0;
            for v in &mut g.tangents {
                *v = apply_tangent(t, *v, flip);
            }
            // As `extend` writes them: whole triangles, wound back when mirrored.
            let whole = g.draws.triangles.len() / 3 * 3;
            g.draws.triangles.truncate(whole);
            if flip {
                for tri in g.draws.triangles.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
        }
        return Prim {
            positions: g.positions,
            normals: g.normals,
            tangents: g.tangents,
            uvs: g.uvs,
            uv_sets: g.uv_sets,
            colors: g.colors,
            color_sets: g.color_sets,
            indices: g.draws.triangles,
            lines: g.draws.lines,
            points: g.draws.points,
            material,
            joints: g.joints,
            skin_joints: g.skin_joints,
            skin_weights: g.skin_weights,
        };
    }
    let n = g.positions.len();
    let k = transforms.len();
    let mut p = Prim {
        positions: Vec::with_capacity(n * k),
        normals: Vec::with_capacity(g.normals.len() * k),
        tangents: Vec::with_capacity(g.tangents.len() * k),
        uvs: Vec::with_capacity(g.uvs.len() * k),
        uv_sets: g.uv_sets.iter().map(|s| Vec::with_capacity(s.len() * k)).collect(),
        colors: Vec::with_capacity(g.colors.len() * k),
        color_sets: g.color_sets.iter().map(|s| Vec::with_capacity(s.len() * k)).collect(),
        indices: Vec::with_capacity(g.draws.triangles.len() * k),
        lines: Vec::with_capacity(g.draws.lines.len() * k),
        points: Vec::with_capacity(g.draws.points.len() * k),
        material,
        joints: Vec::with_capacity(g.joints.len() * k),
        skin_joints: Vec::with_capacity(g.skin_joints.len() * k),
        skin_weights: Vec::with_capacity(g.skin_weights.len() * k),
    };
    for t in transforms {
        p.positions.extend(g.positions.iter().map(|&v| apply_point(t, v)));
        let nm = normal_matrix(t);
        p.normals.extend(g.normals.iter().map(|&v| apply_normal(&nm, v)));
        let flip = det3(t) < 0.0;
        p.tangents.extend(g.tangents.iter().map(|&v| apply_tangent(t, v, flip)));
        p.uvs.extend_from_slice(&g.uvs);
        for (to, from) in p.uv_sets.iter_mut().zip(&g.uv_sets) {
            to.extend_from_slice(from);
        }
        p.colors.extend_from_slice(&g.colors);
        for (to, from) in p.color_sets.iter_mut().zip(&g.color_sets) {
            to.extend_from_slice(from);
        }
        p.joints.extend_from_slice(&g.joints);
        p.skin_joints.extend_from_slice(&g.skin_joints);
        p.skin_weights.extend_from_slice(&g.skin_weights);
    }
    extend(&mut p, &g.draws, n, transforms);
    p
}

/// Adds what `d` draws on `n` vertices to `p`, once for each of `transforms`, where `p` has those
/// vertices at each in turn: the `k`th's from `k * n`. A mirroring transform flips the triangles'
/// winding back.
fn extend(p: &mut Prim, d: &Draws, n: usize, transforms: &[M4]) {
    for (k, t) in transforms.iter().enumerate() {
        let base = u32_of(k * n);
        let flip = det3(t) < 0.0;
        for tri in d.triangles.chunks_exact(3) {
            let [a, b, c] = [tri[0] + base, tri[1] + base, tri[2] + base];
            p.indices.extend_from_slice(&if flip { [a, c, b] } else { [a, b, c] });
        }
        p.lines.extend(d.lines.iter().map(|&i| i + base));
        p.points.extend(d.points.iter().map(|&i| i + base));
    }
}

/// The name of texture coordinate set `set`'s attribute.
fn texcoord(set: u64) -> std::borrow::Cow<'static, str> {
    const NAMES: [&str; 4] = ["TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2", "TEXCOORD_3"];
    match usize::try_from(set).ok().and_then(|s| NAMES.get(s)) {
        Some(name) => (*name).into(),
        None => format!("TEXCOORD_{set}").into(),
    }
}

/// A glTF primitive's mode: 4 (triangles) when it doesn't say.
fn mode(p: &Value) -> u64 {
    p.get("mode").and_then(Value::as_u64).unwrap_or(4)
}

impl Reader<'_> {
    /// When animation `a`'s first key is, and its last, over the channels that move nodes:
    /// `(0, 0)` for one without keys.
    fn span(&mut self, a: usize) -> Result<(f32, f32)> {
        let anim = self.get("animations", a)?;
        let samplers = items(anim, "samplers");
        let (mut first, mut last) = (f32::INFINITY, f32::NEG_INFINITY);
        for ch in items(anim, "channels") {
            if ch.get("target").and_then(node_target).is_none() {
                continue;
            }
            let Some(input) = index(ch, "sampler")
                .and_then(|s| samplers.get(s))
                .and_then(|s| index(s, "input"))
            else {
                continue;
            };
            let times = self.read::<1>(input, [0.0])?;
            if let (Some([a]), Some([b])) = (times.first(), times.last()) {
                first = first.min(*a);
                last = last.max(*b);
            }
        }
        Ok(if first <= last { (first, last) } else { (0.0, 0.0) })
    }

    /// Sets each node's animated properties for `pose`.
    fn pose(&mut self, pose: Pose) -> Result<()> {
        let Pose::Animation { index: a, time } = pose else {
            return Ok(());
        };
        let anim = self.get("animations", a)?;
        let samplers = items(anim, "samplers");
        for ch in items(anim, "channels") {
            let Some((node, path)) = ch.get("target").and_then(node_target) else {
                // A channel for another extension's target, or none.
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
            // A sampler with fewer outputs than keys samples narrower than its path: skipped,
            // as `Rig::pose` skips it, rather than read past its values.
            let p = &mut self.posed[node];
            match (path, v.as_slice()) {
                ("translation", &[x, y, z]) => p.translation = Some([x, y, z]),
                ("scale", &[x, y, z]) => p.scale = Some([x, y, z]),
                ("rotation", &[x, y, z, w]) => p.rotation = Some([x, y, z, w]),
                ("weights", _) => p.weights = Some(v.into_iter().map(f32_of).collect()),
                _ => {}
            }
        }
        Ok(())
    }

    /// Node `i`'s own transform, as posed.
    fn local(&self, i: usize, node: &NodeView<'_>) -> M4 {
        let p = &self.posed[i];
        // A node with a matrix can't be animated.
        if let Some(m) = node
            .matrix
            .and_then(Value::as_array)
            .filter(|m| m.len() == 16 && !p.given)
        {
            let mut out = IDENTITY;
            for (o, x) in out.iter_mut().zip(m) {
                *o = x.as_f64().unwrap_or(*o);
            }
            return out;
        }
        trs(
            p.translation.unwrap_or_else(|| floats(node.translation, [0.0; 3])),
            p.rotation
                .unwrap_or_else(|| floats(node.rotation, [0.0, 0.0, 0.0, 1.0])),
            p.scale.unwrap_or_else(|| floats(node.scale, [1.0; 3])),
        )
    }
}

/// An animation sampler's value at `t`: `values` holds one element per key (three, in-tangent,
/// value and out-tangent, for `CUBICSPLINE`), each `values.len() / keys` wide. Before the
/// first key it's the first value, after the last the last. `None` for a sampler without keys
/// or with values that don't fit them.
pub(crate) fn sample(times: &[f32], values: &[f32], interpolation: &str, t: f32, rotation: bool) -> Option<Vec<f64>> {
    let mut out = vec![0.0; width(times, values, interpolation)?];
    sample_into(times, values, interpolation, t, rotation, &mut out).then_some(out)
}

/// How wide each of a sampler's elements is: `None` for a sampler without keys or with values
/// that don't fit them.
pub(crate) fn width(times: &[f32], values: &[f32], interpolation: &str) -> Option<usize> {
    let keys = times.len() * if interpolation == "CUBICSPLINE" { 3 } else { 1 };
    (keys > 0 && !values.is_empty() && values.len().is_multiple_of(keys)).then(|| values.len() / keys)
}

/// [`sample`] written into `out`, which must be the sampler's [`width`]: `false`, leaving `out`
/// as it was, if it isn't or the sampler can't be read. Allocates nothing. A rotation comes out
/// unit length at every time, its first and last keys too.
pub(crate) fn sample_into(
    times: &[f32],
    values: &[f32],
    interpolation: &str,
    t: f32,
    rotation: bool,
    out: &mut [f64],
) -> bool {
    let k = times.partition_point(|&x| x <= t);
    sample_after(times, values, interpolation, t, rotation, k, out)
}

/// [`sample_into`] given `k`, how many keys are at or before `t`, as a caller walking through time in order can
/// know without searching.
pub(crate) fn sample_after(
    times: &[f32],
    values: &[f32],
    interpolation: &str,
    t: f32,
    rotation: bool,
    k: usize,
    out: &mut [f64],
) -> bool {
    let Some(w) = width(times, values, interpolation) else {
        return false;
    };
    if out.len() != w {
        return false;
    }
    let keys = times.len();
    let cubic = interpolation == "CUBICSPLINE";
    let per = if cubic { 3 } else { 1 };
    // The value of key k (its middle element for a cubic spline), and its tangents.
    let at = |k: usize, part: usize| -> &[f32] {
        let e = k * per + part;
        &values[e * w..(e + 1) * w]
    };
    let value = |k: usize| at(k, usize::from(cubic));
    let copy = |out: &mut [f64], v: &[f32]| {
        for (o, &x) in out.iter_mut().zip(v) {
            *o = f64::from(x);
        }
    };
    if k == 0 {
        copy(out, value(0));
    } else if k == keys {
        copy(out, value(keys - 1));
    } else {
        let (k, (t0, t1)) = (k - 1, (f64::from(times[k - 1]), f64::from(times[k])));
        let dt = t1 - t0;
        let s = if dt > 0.0 { (f64::from(t) - t0) / dt } else { 0.0 };
        match interpolation {
            // On a key every interpolation gives its value: sampling at the keys, as a clip baked frame by frame
            // does, needs no slerp or spline.
            _ if s == 0.0 => copy(out, value(k)),
            "STEP" => copy(out, value(k)),
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
                for (c, o) in out.iter_mut().enumerate() {
                    let f = |v: &[f32]| f64::from(v[c]);
                    *o = h[0] * f(v0) + h[1] * f(b0) + h[2] * f(v1) + h[3] * f(a1);
                }
            }
            _ if rotation && w == 4 => slerp(value(k), value(k + 1), s, out),
            _ => {
                for ((o, &x), &y) in out.iter_mut().zip(value(k)).zip(value(k + 1)) {
                    let (x, y) = (f64::from(x), f64::from(y));
                    *o = x + (y - x) * s;
                }
            }
        }
    }
    if rotation && w == 4 {
        let len = out.iter().map(|c| c * c).sum::<f64>().sqrt();
        if len > 0.0 {
            for c in out.iter_mut() {
                *c /= len;
            }
        }
    }
    true
}

/// Spherical linear interpolation between unit quaternions, the short way round, into `out`.
fn slerp(a: &[f32], b: &[f32], s: f64, out: &mut [f64]) {
    let mut d: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
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
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = wa * f64::from(x) + wb * sign * f64::from(y);
    }
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
            skeleton: Vec::new(),
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
    fn tangents_and_every_set_are_read_and_follow_the_node() {
        // One triangle with a tangent, three UV sets and a second color set (but no first), its
        // color texture reading set 1 at twice the scale, on a node mirrored in x.
        let floats: Vec<f32> = [
            &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..],         // POSITION
            &[0., 0., 1., 0., 0., 1., 0., 0., 1.],             // NORMAL
            &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.], // TANGENT
            &[0., 0., 1., 0., 0., 1.],                         // TEXCOORD_0
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],                   // TEXCOORD_1
            &[7., 8., 7., 8., 7., 8.],                         // TEXCOORD_2
            &[1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1.], // COLOR_1
        ]
        .concat();
        let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let parts = [
            ("VEC3", 3),
            ("VEC3", 3),
            ("VEC4", 4),
            ("VEC2", 2),
            ("VEC2", 2),
            ("VEC2", 2),
            ("VEC4", 4),
        ];
        let mut offset = 0;
        let mut accessors = Vec::new();
        let mut views = Vec::new();
        for (i, (kind, width)) in parts.iter().enumerate() {
            views.push(serde_json::json!({ "buffer": 0, "byteOffset": offset, "byteLength": 12 * width }));
            let mut a = serde_json::json!({ "bufferView": i, "componentType": 5126, "count": 3, "type": kind });
            if i == 0 {
                a["min"] = serde_json::json!([0, 0, 0]);
                a["max"] = serde_json::json!([1, 1, 0]);
            }
            accessors.push(a);
            offset += 12 * width;
        }
        let gltf = serde_json::json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "uri": "t.bin", "byteLength": bin.len() }],
            "bufferViews": views,
            "accessors": accessors,
            "materials": [{ "pbrMetallicRoughness": { "baseColorTexture": {
                "index": 0, "texCoord": 0,
                "extensions": { "KHR_texture_transform": { "texCoord": 1, "scale": [2, 2] } }
            } } }],
            "meshes": [{ "primitives": [{ "material": 0, "attributes": {
                "POSITION": 0, "NORMAL": 1, "TANGENT": 2,
                "TEXCOORD_0": 3, "TEXCOORD_1": 4, "TEXCOORD_2": 5, "COLOR_1": 6, "TEXCOORD_99999999999": 5
            } }] }],
            "nodes": [{ "mesh": 0, "scale": [-1, 1, 1] }],
            "scenes": [{ "nodes": [0] }],
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("t.bin"), &bin).unwrap();
        let mesh = load(dir.path(), "t.gltf", gltf.to_string().as_bytes());
        let p = &mesh.prims[0];
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
        assert!(
            near(p.uvs[0], [0.2, 0.4]) && near(p.uvs[2], [1.0, 1.2]),
            "set 1, transformed: {:?}",
            p.uvs
        );
        assert_eq!(
            p.uv_sets.len(),
            3,
            "set 0, set 1 (in `uvs`), set 2; a set number past any real one is ignored"
        );
        assert_eq!(
            p.uv_sets[0],
            [[0., 0.], [1., 0.], [0., 1.]],
            "the other sets as the file has them"
        );
        assert_eq!(p.uv_sets[1], Vec::<[f32; 2]>::new());
        assert_eq!(p.uv_sets[2], [[7., 8.]; 3]);
        assert_eq!(p.colors, Vec::<[f32; 4]>::new());
        assert_eq!(p.color_sets.len(), 2);
        assert_eq!(p.color_sets[1][2], [0., 0., 1., 1.]);
        // Mirrored, the tangent turns with the node and its handedness flips, so the bitangent,
        // cross(normal, tangent) * w, still points up +y as the node moves it.
        assert_eq!(p.tangents, [[-1., 0., 0., -1.]; 3]);
        assert_eq!(p.normals, [[0., 0., 1.]; 3]);

        // Written out, the sets are renumbered from 0 with no gaps, the color texture's first.
        let glb = to_glb(&mesh).unwrap();
        let back = load(dir.path(), "back.glb", &glb);
        let q = &back.prims[0];
        assert_eq!(q.uvs, p.uvs);
        assert_eq!(q.uv_sets, [vec![], p.uv_sets[0].clone(), p.uv_sets[2].clone()]);
        assert_eq!(q.colors, p.color_sets[1]);
        assert_eq!(q.tangents, p.tangents);
    }

    #[test]
    fn a_cut_out_unlit_material_keeps_its_textures_through_a_round_trip() {
        let floats: Vec<f32> = [
            &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..], // POSITION
            &[0., 0., 1., 0., 0., 1.],                 // TEXCOORD_0
            &[0., 0., 1., 0., 0., 1.],                 // TEXCOORD_1
        ]
        .concat();
        let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let gltf = serde_json::json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "uri": "m.bin", "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 24 },
                { "buffer": 0, "byteOffset": 60, "byteLength": 24 },
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 0] },
                { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2" },
                { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
            ],
            "images": [{ "uri": "a.png" }, { "uri": "b.png" }, { "uri": "c.png" }],
            "textures": [{ "source": 0 }, { "source": 1 }, { "source": 2 }],
            "materials": [{
                "alphaMode": "MASK", "alphaCutoff": 0.3,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 1, 1, 0.8],
                    "baseColorTexture": { "index": 0 },
                    "metallicRoughnessTexture": { "index": 1 },
                },
                "occlusionTexture": { "index": 1 },
                // Read through the other set: kept with the set it reads.
                "emissiveTexture": { "index": 2, "texCoord": 1 },
                "extensions": { "KHR_materials_unlit": {} },
            }],
            "meshes": [{ "primitives": [{ "material": 0,
                "attributes": { "POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2 } }] }],
            "nodes": [{ "mesh": 0 }],
            "scenes": [{ "nodes": [0] }],
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.bin"), &bin).unwrap();
        for (i, name) in ["a.png", "b.png", "c.png"].into_iter().enumerate() {
            let png = [
                &b"\x89PNG\r\n\x1a\n"[..],
                &[0, 0, 0, 13, b'I', b'H', b'D', b'R', u8::try_from(i).unwrap()],
            ]
            .concat();
            std::fs::write(dir.path().join(name), png).unwrap();
        }
        let mesh = load(dir.path(), "m.gltf", gltf.to_string().as_bytes());
        let check = |mesh: &Mesh| {
            let m = &mesh.materials[mesh.prims[0].material];
            assert_eq!(m.alpha_cutoff, Some(0.3));
            assert!((m.color[3] - 0.8).abs() < 1e-6, "a cut-out material keeps its alpha");
            assert!(m.unlit);
            let (Some(c), Some(o)) = (m.color_image, m.occlusion_image) else {
                panic!("{m:?}")
            };
            assert_ne!(mesh.images[c], mesh.images[o]);
            assert_eq!(m.metallic_roughness_image, Some(o), "one image, two textures");
            let e = m.emissive_image.expect("the emissive texture on the other set");
            assert!(mesh.images[e] != mesh.images[c] && mesh.images[e] != mesh.images[o]);
            let set = m.sampling.emissive.uv_set.expect("its own set");
            assert_eq!(mesh.prims[0].uv_sets[set], [[0., 0.], [1., 0.], [0., 1.]]);
        };
        check(&mesh);
        let back = load(dir.path(), "back.glb", &to_glb(&mesh).unwrap());
        check(&back);
    }

    #[test]
    fn an_animation_pointer_to_a_node_moves_it() {
        let floats: Vec<f32> = [
            &[0., 0., 0., 1., 0., 0., 0., 1., 0.][..], // POSITION
            &[0., 1.],                                 // times
            &[0., 0., 0., 2., 0., 0.],                 // translations
        ]
        .concat();
        let bin: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let pointer = |p: &str| {
            serde_json::json!({ "sampler": 0, "target": { "path": "pointer",
            "extensions": { "KHR_animation_pointer": { "pointer": p } } } })
        };
        let gltf = serde_json::json!({
            "asset": { "version": "2.0" },
            "extensionsUsed": ["KHR_animation_pointer"],
            "extensionsRequired": ["KHR_animation_pointer"],
            "buffers": [{ "uri": "a.bin", "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 8 },
                { "buffer": 0, "byteOffset": 44, "byteLength": 24 },
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 0] },
                { "bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] },
                { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3" },
            ],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }],
            "nodes": [{ "mesh": 0 }],
            "scenes": [{ "nodes": [0] }],
            "animations": [{
                "samplers": [{ "input": 1, "output": 2 }],
                // The first is the one that counts; the others aren't a node's, or aren't
                // indices as JSON pointers write them.
                "channels": [pointer("/nodes/0/translation"), pointer("/materials/0/emissiveFactor"),
                             pointer("/nodes/00/translation"), pointer("/nodes/0/extras")],
            }],
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), &bin).unwrap();
        std::fs::write(dir.path().join("a.gltf"), gltf.to_string()).unwrap();
        let doc =
            crate::formats::load(&dir.path().join("a.gltf"), "gltf", &Folder::new(dir.path()), Keep::ALL).unwrap();
        let mesh = doc.flatten(Pose::Animation { index: 0, time: 0.5 }).unwrap().unwrap();
        assert_eq!(mesh.prims[0].positions[0], [1., 0., 0.]);
        let rig = doc.rig().unwrap();
        let ch = &rig.animations[0].channels;
        assert_eq!(ch.len(), 1);
        assert_eq!((ch[0].node, ch[0].path), (0, crate::rig::Path::Translation));
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
    fn reads_accessors_matrices_padded() {
        // MAT2 of bytes: each column 2 bytes, padded to 4. MAT3 of normalized shorts: each
        // column 6 bytes, padded to 8. A sparse VEC3 of floats with one element replaced.
        let mut bin = vec![1, 2, 0xAA, 0xAA, 3, 4, 0xAA, 0xAA];
        for c in 0..3i16 {
            for r in 0..3i16 {
                bin.extend_from_slice(&(c * 3 + r - 4).saturating_mul(8191).to_le_bytes());
            }
            bin.extend_from_slice(&[0xAA, 0xAA]);
        }
        let floats = bin.len();
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 8.0, 7.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let sparse = bin.len();
        bin.extend_from_slice(&[1, 0, 0, 0]);
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "bufferViews": [
                {"buffer": 0, "byteLength": 8},
                {"buffer": 0, "byteOffset": 8, "byteLength": 24},
                {"buffer": 0, "byteOffset": floats, "byteLength": 24},
                {"buffer": 0, "byteOffset": floats + 24, "byteLength": 12},
                {"buffer": 0, "byteOffset": sparse, "byteLength": 4},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5121, "count": 1, "type": "MAT2"},
                {"bufferView": 1, "componentType": 5122, "normalized": true, "count": 1, "type": "MAT3"},
                {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC3",
                 "sparse": {"count": 1, "indices": {"bufferView": 4, "componentType": 5121},
                            "values": {"bufferView": 3}}},
                {"bufferView": 4, "componentType": 5121, "count": 1, "type": "SCALAR"},
            ],
        });
        let mut doc = crate::document::Document::empty(serde_json::Map::new());
        doc.json = json.as_object().unwrap().clone();
        doc.push_buffer(bin);
        assert_eq!(doc.read_floats(0).unwrap(), (4, vec![1.0, 2.0, 3.0, 4.0]));
        let (width, m3) = doc.read_floats(1).unwrap();
        assert_eq!(width, 9);
        let want: Vec<f32> = (0..9).map(|k| ((k - 4) * 8191) as f32 / 32767.0).collect();
        assert!(m3.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6), "{m3:?}");
        assert_eq!(doc.read_floats(2).unwrap(), (3, vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0]));
        assert_eq!(doc.read_indices(3).unwrap(), vec![1]);
        assert!(doc.read_indices(0).is_err(), "a matrix isn't indices");
        assert!(doc.read_floats(9).is_err());
    }

    #[test]
    fn decode_keeps_required_extensions_it_cant_draw() {
        let dir = tempfile::tempdir().unwrap();
        let packed = crate::gltf::compress(&to_glb(&grid(10)).unwrap()).unwrap();
        let path = dir.path().join("packed.glb");
        std::fs::write(&path, &packed).unwrap();
        let mut doc = crate::formats::load(&path, "glb", &Folder::new(dir.path()), Keep::ALL).unwrap();
        for key in ["extensionsUsed", "extensionsRequired"] {
            let list = doc.json.entry(key).or_insert_with(|| serde_json::json!([]));
            list.as_array_mut().unwrap().push("EXT_made_up".into());
        }
        assert!(
            doc.flatten(Pose::Rest).is_err(),
            "drawing needs every required extension"
        );
        doc.decode().unwrap();
        assert_eq!(doc.json["extensionsRequired"], serde_json::json!(["EXT_made_up"]));
        assert!(!serde_json::to_string(&doc.json).unwrap().contains("meshopt"));
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
        doc.json.insert(
            "extensionsRequired".into(),
            serde_json::json!(["KHR_materials_unlit", "KHR_techniques_webgl"]),
        );
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

    /// A document of a unit square's four corners (accessor 0), with index accessors 1
    /// (`[3, 2, 1, 0]`), 2 (two triangles) and 3 (`[0, 0, 1, 2, 3]`), drawn by `meshes` on
    /// `nodes`.
    fn corners(meshes: serde_json::Value, nodes: serde_json::Value) -> crate::document::Document {
        let mut bin: Vec<u8> = [[0f32, 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        bin.extend([3u16, 2, 1, 0].iter().flat_map(|i| i.to_le_bytes()));
        bin.extend([0u8, 1, 2, 0, 2, 3, 0, 0, 1, 2, 3, 0]);
        let n = (0..nodes.as_array().unwrap().len()).collect::<Vec<_>>();
        let mut json = serde_json::json!({
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteLength": 48 },
                { "buffer": 0, "byteOffset": 48, "byteLength": 8 },
                { "buffer": 0, "byteOffset": 56, "byteLength": 6 },
                { "buffer": 0, "byteOffset": 62, "byteLength": 5 },
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, 1, 0] },
                { "bufferView": 1, "componentType": 5123, "count": 4, "type": "SCALAR" },
                { "bufferView": 2, "componentType": 5121, "count": 6, "type": "SCALAR" },
                { "bufferView": 3, "componentType": 5121, "count": 5, "type": "SCALAR" },
            ],
            "scenes": [{ "nodes": n }],
        });
        json["meshes"] = meshes;
        json["nodes"] = nodes;
        let mut doc = crate::document::Document::empty(serde_json::Map::new());
        let serde_json::Value::Object(json) = json else {
            unreachable!()
        };
        doc.json = json;
        doc.push_buffer(bin);
        doc
    }

    #[test]
    fn point_only_mesh_flattens_with_points() {
        let meshes = serde_json::json!([{ "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 0 }] }]);
        let mut doc = corners(meshes, serde_json::json!([{ "mesh": 0 }]));
        let mesh = doc.flatten(Pose::Rest).unwrap().expect("points are drawn");
        assert_eq!(mesh.prims.len(), 1);
        let p = &mesh.prims[0];
        assert_eq!(p.points, [0, 1, 2, 3]);
        assert!(p.indices.is_empty() && p.lines.is_empty());
        assert_eq!(p.positions.len(), 4);
        assert!(p.normals.is_empty(), "no normals are made up for points");
        assert_eq!((mesh.triangles(), mesh.lines(), mesh.points()), (0, 0, 4));

        doc.set_max_points(3);
        let err = doc.flatten(Pose::Rest).unwrap_err().to_string();
        assert!(err.contains("too many points"), "{err}");
        doc.set_lines_and_points(false);
        assert!(doc.flatten(Pose::Rest).unwrap().is_none(), "left out, nothing is drawn");
    }

    #[test]
    fn strips_and_loops_expand_to_segments() {
        let meshes = serde_json::json!([
            { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 3, "indices": 1 }] },
            { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 2 }] },
            { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 1, "indices": 3 }] },
            { "primitives": [{ "attributes": { "POSITION": 0 }, "mode": 1 }] },
        ]);
        let nodes = serde_json::json!([{ "mesh": 0 }, { "mesh": 1 }, { "mesh": 2 }, { "mesh": 3 }]);
        let mesh = corners(meshes, nodes).flatten(Pose::Rest).unwrap().unwrap();
        let lines: Vec<&[u32]> = mesh.prims.iter().map(|p| p.lines.as_slice()).collect();
        assert_eq!(lines[0], [3, 2, 2, 1, 1, 0], "a strip: each index to the next");
        assert_eq!(lines[1], [0, 1, 1, 2, 2, 3, 3, 0], "a loop: closed back to the first");
        assert_eq!(lines[2], [1, 2], "a segment to itself and an odd index out are dropped");
        assert_eq!(
            lines[3],
            [0, 1, 2, 3],
            "plain lines, without indices, pair the vertices"
        );
        assert_eq!(mesh.lines(), 3 + 4 + 1 + 2);
        assert_eq!(mesh.triangles(), 0);
    }

    #[test]
    fn node_transforms_move_lines_and_points() {
        let meshes = serde_json::json!([{ "primitives": [
            { "attributes": { "POSITION": 0 }, "indices": 2 },
            { "attributes": { "POSITION": 0 }, "mode": 3, "indices": 1 },
            { "attributes": { "POSITION": 0 }, "mode": 0 },
        ] }]);
        let nodes = serde_json::json!([
            { "mesh": 0, "translation": [10, 0, 0] },
            { "mesh": 0, "scale": [-1, 1, 1] },
        ]);
        let mesh = corners(meshes, nodes).flatten(Pose::Rest).unwrap().unwrap();
        // Both nodes draw the mesh: one primitive, the vertices once in each place.
        assert_eq!(
            mesh.prims.len(),
            1,
            "lines and points on a primitive's vertices join it"
        );
        let p = &mesh.prims[0];
        assert_eq!(p.positions[2], [11.0, 1.0, 0.0], "moved");
        assert_eq!(p.positions[5], [-1.0, 0.0, 0.0], "mirrored");
        assert_eq!(
            p.indices,
            [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6],
            "a mirror flips triangles back"
        );
        assert_eq!(p.lines, [3, 2, 2, 1, 1, 0, 7, 6, 6, 5, 5, 4], "but not lines");
        assert_eq!(p.points, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!((mesh.triangles(), mesh.lines(), mesh.points()), (4, 6, 8));
    }

    #[test]
    fn instances_repeat_lines_and_points() {
        let meshes = serde_json::json!([{ "primitives": [
            { "attributes": { "POSITION": 0 }, "mode": 1, "indices": 1 },
            { "attributes": { "POSITION": 0 }, "mode": 0 },
        ] }]);
        let mut doc = corners(meshes, serde_json::json!([{ "mesh": 0 }]));
        // Two instances, 5 apart along x.
        let bin: Vec<u8> = [0f32, 0., 0., 5., 0., 0.]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let b = doc.push_buffer(bin);
        let json = &mut doc.json;
        json["buffers"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "byteLength": 24 }));
        json["bufferViews"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "buffer": b, "byteLength": 24 }));
        json["accessors"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC3" }));
        json["nodes"][0]["extensions"] =
            serde_json::json!({ "EXT_mesh_gpu_instancing": { "attributes": { "TRANSLATION": 4 } } });
        json.insert("extensionsUsed".into(), serde_json::json!(["EXT_mesh_gpu_instancing"]));
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.prims.len(), 1);
        let p = &mesh.prims[0];
        assert_eq!(p.positions.len(), 8);
        assert_eq!(p.positions[5], [6.0, 0.0, 0.0]);
        assert_eq!(
            p.lines,
            [3, 2, 1, 0, 7, 6, 5, 4],
            "each instance's segments on its own vertices"
        );
        assert_eq!(p.points, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn glb_round_trip_keeps_lines_and_points() {
        let mut mesh = grid(2);
        mesh.prims[0].lines = vec![0, 1, 1, 2, 2, 5];
        mesh.prims[0].points = vec![4, 8];
        mesh.prims[1].indices.clear();
        mesh.prims[1].normals.clear();
        mesh.prims[1].points = (0..9).collect();
        let glb = to_glb(&mesh).unwrap();
        let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json: serde_json::Value = serde_json::from_slice(&glb[20..20 + len]).unwrap();
        let prims = json["meshes"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|m| m["primitives"].as_array().unwrap().clone())
            .collect::<Vec<_>>();
        let modes: Vec<u64> = prims.iter().map(|p| p["mode"].as_u64().unwrap_or(4)).collect();
        assert_eq!(modes, [4, 1, 0, 0]);
        assert_eq!(prims[0]["attributes"], prims[1]["attributes"], "one set of vertices");
        assert_eq!(prims[0]["attributes"], prims[2]["attributes"]);
        assert_eq!(prims[1]["material"], prims[0]["material"]);
        let lines = &json["accessors"][usize::try_from(prims[1]["indices"].as_u64().unwrap()).unwrap()];
        assert_eq!(
            (lines["componentType"].as_u64(), lines["count"].as_u64()),
            (Some(5123), Some(6))
        );
        assert!(prims[3].get("indices").is_none(), "a point cloud needs no indices");

        let none = crate::assets::Memory::new();
        let back = crate::formats::load_bytes(&glb, "m.glb", "glb", &none, Keep::ALL)
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
            .unwrap();
        assert_eq!(
            back.prims.len(),
            2,
            "lines and points join the triangles they share vertices with"
        );
        assert_eq!(back.prims[0].indices, mesh.prims[0].indices);
        assert_eq!(back.prims[0].lines, mesh.prims[0].lines);
        assert_eq!(back.prims[0].points, mesh.prims[0].points);
        assert_eq!(back.prims[0].positions, mesh.prims[0].positions);
        assert_eq!(back.prims[1].points, mesh.prims[1].points);
        assert!(back.prims[1].normals.is_empty(), "no normals are made up for points");
        assert!(to_glb(&back).unwrap() == glb, "written again, the same bytes");
    }

    /// The box around a flattened mesh's positions.
    fn box_of(mesh: Option<&Mesh>) -> Option<crate::document::Bounds> {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in mesh.into_iter().flat_map(|m| &m.prims).flat_map(|p| &p.positions) {
            for c in 0..3 {
                lo[c] = lo[c].min(v[c]);
                hi[c] = hi[c].max(v[c]);
            }
        }
        (lo[0] <= hi[0]).then_some(crate::document::Bounds { min: lo, max: hi })
    }

    #[cfg(all(feature = "gltf", feature = "fs"))]
    #[test]
    fn bounds_hold_what_flatten_draws_in_every_pose() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf");
        let cases = [
            "skin_animated/skin.glb",
            "skin_rest_pose",
            "skin_eight_influences",
            "blender_shape_key/shape_key.glb",
            "morph_default_weights",
            "gpu_instancing/inst.glb",
            "mirror_x/cube.glb",
            "node_hierarchy",
            "node_matrix",
            "points_and_lines",
            "quantized_normalized_i16",
            "sparse",
            "blender_draco/draco.glb",
        ];
        for case in cases {
            let mut path = root.join(case);
            if path.is_dir() {
                path = std::fs::read_dir(&path)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .find(|p| p.extension().is_some_and(|e| e == "glb" || e == "gltf"))
                    .unwrap();
            }
            let doc = crate::open(&path).unwrap();
            let mut poses = vec![Pose::Rest, Pose::Bind];
            let animations = super::list(&doc.json, "animations").len();
            poses.extend((0..animations).map(|index| Pose::Animation { index, time: 0.4 }));
            for pose in poses {
                let Ok(mesh) = doc.flatten(pose) else { continue };
                let want = box_of(mesh.as_ref());
                assert_eq!(doc.bounds(pose).unwrap(), want, "{case} in {pose:?}");
                assert!(want.is_some(), "{case} draws something");
            }
        }
    }

    #[cfg(all(feature = "gltf", feature = "fs"))]
    #[test]
    fn animation_bounds_hold_every_sampled_pose() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf/skin_animated/skin.glb");
        let doc = crate::open(&path).unwrap();
        let (first, last) = doc.rig().unwrap().animations[0].span();
        assert!(first < last);
        let all = doc.animation_bounds(0, 5).unwrap().unwrap();
        for k in 0..5u8 {
            let time = first + (last - first) * f32::from(k) / 4.0;
            let b = doc.bounds(Pose::Animation { index: 0, time }).unwrap().unwrap();
            assert!(
                (0..3).all(|c| all.min[c] <= b.min[c] && b.max[c] <= all.max[c]),
                "{b:?} in {all:?}"
            );
        }
        let start = doc.bounds(Pose::Animation { index: 0, time: first }).unwrap();
        assert_eq!(doc.animation_bounds(0, 1).unwrap(), start);
        assert_eq!(doc.animation_bounds(0, 0).unwrap(), start);
        assert!(doc.animation_bounds(1, 5).is_err());
    }

    #[test]
    fn bounds_and_vertices_count_every_instance() {
        let meshes = serde_json::json!([{ "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 2 }] }]);
        let doc = corners(
            meshes,
            serde_json::json!([{ "mesh": 0 }, { "mesh": 0, "translation": [5, 0, -2], "rotation": [0, 0, 0.707_106_77, 0.707_106_77] }]),
        );
        let b = doc.bounds(Pose::Rest).unwrap().unwrap();
        assert_eq!(b, box_of(doc.flatten(Pose::Rest).unwrap().as_ref()).unwrap());
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(near(b.min, [0.0, 0.0, -2.0]) && near(b.max, [5.0, 1.0, 0.0]), "{b:?}");
        assert!((b.center()[0] - 2.5).abs() < 1e-6 && (b.size()[2] - 2.0).abs() < 1e-6);
        let c = doc.counts();
        assert_eq!((c.vertices, c.triangles), (8, 4));
        let empty = corners(serde_json::json!([]), serde_json::json!([]));
        assert_eq!(empty.bounds(Pose::Rest).unwrap(), None);
    }

    #[cfg(feature = "draco")]
    #[test]
    fn draco_compressed_accessors_read_as_their_stream_decodes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf/blender_draco/draco.glb");
        let dir = path.parent().unwrap();
        let doc = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL).unwrap();
        let mut decoded = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL).unwrap();
        decoded.decode().unwrap();
        let prim = &doc.json["meshes"][0]["primitives"][0];
        let position = usize::try_from(prim["attributes"]["POSITION"].as_u64().unwrap()).unwrap();
        let indices = usize::try_from(prim["indices"].as_u64().unwrap()).unwrap();
        assert!(
            doc.json["accessors"][position].get("bufferView").is_none(),
            "the data is in the stream"
        );
        let (n, floats) = doc.read_floats(position).unwrap();
        assert_eq!((n, &floats), (3, &decoded.read_floats(position).unwrap().1));
        assert!(floats.iter().any(|&f| f != 0.0), "not zeros");
        assert_eq!(
            doc.read_indices(indices).unwrap(),
            decoded.read_indices(indices).unwrap()
        );
    }
}
