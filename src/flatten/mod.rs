//! The flat view of a [`Document`]: the triangles, lines and points its default scene draws,
//! posed, as a [`Mesh`], and how many of each that is, counted from the JSON alone.
//!
//! - The default scene is drawn (else scene 0; with no scenes, every root node; with no nodes,
//!   every mesh). Node transforms are baked into the vertices; a mirroring transform flips the
//!   winding back. `EXT_mesh_gpu_instancing` instances are baked the same way. A node
//!   `KHR_node_visibility` hides isn't drawn, nor anything under it; an animation can show
//!   or hide a node through `KHR_animation_pointer`.
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
//! - A volume's thickness (`KHR_materials_volume`) is in its mesh's space and scales with the node, so
//!   it is scaled as the vertices are: by the cube root of the transform's volume change, which is
//!   the scale when it is uniform and the mean of the three otherwise. A material drawn at several
//!   scales becomes a material for each. A skinned mesh takes the mean scale of its skin's joints.
//!   The attenuation distance is in world space, and stays.
//! - Meshopt-compressed views are decoded as they are read; Draco
//!   (`KHR_draco_mesh_compression`) needs the `draco` feature.
//! - A required extension it can't honour is an error, unless it only changes how surfaces look
//!   or changes nothing drawn (interaction, metadata).
//!
//! Accessors are read from their buffers in place, so the peak is little more than the mesh.

mod accessor;
mod animation;
mod decode;
mod material;
mod mesh;
mod repair;
mod rig;
#[cfg(all(test, feature = "fs"))]
#[allow(clippy::cast_precision_loss, reason = "test grids are small")]
mod tests;

pub(crate) use animation::{sample_after, width};
pub(crate) use repair::repaired;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{Map, Value};

#[cfg(doc)]
use crate::document::Keep;
use crate::document::{Bounds, Counts, Document, Pose};
use crate::mesh::Mesh;
use crate::num::{f32_of, usize_of};
use crate::quat;
use crate::{Context as _, ErrorKind, Result};
use accessor::Tabs;

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
        counts(&self.base, Tabs::of(self))
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
        let (first, last) = Reader::whole(self).span(index)?;
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
fn counts(json: &Map<String, Value>, tabs: Tabs<'_>) -> Counts {
    let mut c = Counts::default();
    let Ok(order) = scene_order(json, |_| None) else {
        return c;
    };
    let meshes = list(json, "meshes");
    let count = |a: usize| tabs.accessor_count(a).map_or(0, usize_of);
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
/// document without nodes, which draws every mesh as it is. `posed` is whether a pose shows or
/// hides a node, over what the node says.
fn scene_order(json: &Map<String, Value>, posed: impl Fn(usize) -> Option<bool>) -> Result<Option<Vec<usize>>> {
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
        list_of(
            scenes
                .get(s)
                .context(ErrorKind::Corrupt, "glTF default scene doesn't exist")?,
            "nodes",
        )
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
        if !posed(i).unwrap_or(visible != Some(&Value::Bool(false))) {
            continue;
        }
        order.push(i);
        if let Some(kids) = nodes[i].get("children").and_then(Value::as_array) {
            stack.extend(kids.iter().rev().filter_map(Value::as_u64).map(usize_of));
        }
    }
    Ok(Some(order))
}

/// 4x4 matrices, column-major as glTF writes them.
pub(crate) use crate::formats::xform::col::{IDENTITY, M4, mul};

/// Where one mesh is drawn: its transforms (several for instances), morph weights and skin.
struct Use {
    mesh: usize,
    weights: Vec<f32>,
    skin: Option<usize>,
    transforms: Vec<M4>,
    /// What the transforms scale a volume's thickness by: one scale for them all.
    thickness: f32,
}

struct Reader<'d> {
    doc: &'d Document,
    json: &'d Map<String, Value>,
    /// The accessors and buffer views, looked up once: every attribute read names one of each.
    tabs: Tabs<'d>,
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
    /// (glTF material, thickness scale's bits) → mesh material, for a volume drawn at a scale other than 1.
    scaled_materials: HashMap<(usize, u32), usize>,
    /// The thickness scales met so far: one within rounding of another is taken as it.
    scales: Vec<f32>,
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
    /// Shown or hidden (`KHR_node_visibility`).
    visible: Option<bool>,
    /// Set by the caller rather than an animation, so it holds for a node written as a matrix
    /// too.
    given: bool,
}

/// The error text for a missing list entry.
fn missing(key: &str, i: usize) -> String {
    format!("glTF {key}[{i}] doesn't exist")
}

/// An accessor type's components, and components per column.
fn shape(kind: Option<&str>) -> Option<(usize, usize)> {
    Some(match kind? {
        "SCALAR" => (1, 1),
        "VEC2" => (2, 2),
        "VEC3" => (3, 3),
        "VEC4" => (4, 4),
        "MAT2" => (4, 2),
        "MAT3" => (9, 3),
        "MAT4" => (16, 4),
        _ => return None,
    })
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

/// Required extensions a flat view draws: the geometry ones it decodes, any that only change how surfaces
/// look (materials, textures, lights), which it shows as best it can, and those that change nothing drawn.
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
    // What draws the same without them: how a viewer answers the pointer, behavior graphs
    // (the scene at rest is drawn), metadata and feature IDs, a promise about the node
    // hierarchy or the mesh's topology, and edges to outline, which aren't drawn.
    const SAME: &[&str] = &[
        "KHR_node_selectability",
        "KHR_node_hoverability",
        "KHR_interactivity",
        "EXT_structural_metadata",
        "EXT_mesh_features",
        "GODOT_single_root",
        "CESIUM_primitive_outline",
        "EXT_mesh_manifold",
    ];
    DECODED.contains(&extension)
        || ALSO.contains(&extension)
        || SAME.contains(&extension)
        || LOOKS.iter().any(|p| extension.starts_with(p))
}

fn f32s<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    floats(v, default.map(f64::from)).map(f32_of)
}

impl<'d> Reader<'d> {
    #[allow(clippy::unnecessary_wraps, reason = "it fails without the draco feature")]
    fn new(doc: &'d Document) -> Result<Self> {
        let json = doc.json();
        #[cfg(not(feature = "draco"))]
        ensure!(
            !list(json, "extensionsRequired")
                .iter()
                .any(|e| e.as_str() == Some("KHR_draco_mesh_compression")),
            FeatureDisabled,
            "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
        );
        if let Some(e) = list(json, "extensionsRequired")
            .iter()
            .filter_map(Value::as_str)
            .find(|e| !drawable(e))
        {
            bail!(
                Unsupported,
                "glTF requires the `{e}` extension, which polysmelt can't draw"
            );
        }
        Ok(Self::whole(doc))
    }

    /// A reader that only reads accessors: none of the checks drawing needs. It reads the
    /// animations only when they're in the JSON, as [`Reader::whole`] has them.
    fn bare(doc: &'d Document) -> Self {
        Self::over(doc, &doc.base, Tabs::of(doc))
    }

    /// A reader of the whole JSON, animations too, but none of the checks drawing needs.
    fn whole(doc: &'d Document) -> Self {
        let json = doc.json();
        Self::over(doc, json, Tabs::json(json))
    }

    fn over(doc: &'d Document, json: &'d Map<String, Value>, tabs: Tabs<'d>) -> Self {
        let n = |key| list(json, key).len();
        Self {
            doc,
            json,
            tabs,
            meshes: list(json, "meshes"),
            nodes: list(json, "nodes"),
            materials_json: list(json, "materials"),
            textures: list(json, "textures"),
            samplers: list(json, "samplers"),
            images_json: list(json, "images"),
            scratch: Vec::new(),
            meshopt: HashMap::new(),
            materials: vec![None; n("materials")],
            scaled_materials: HashMap::new(),
            scales: vec![1.0],
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
        let Some(order) = scene_order(self.json, |i| self.posed[i].visible)? else {
            // No nodes at all: every mesh, as it is.
            return Ok((0..self.meshes.len())
                .map(|mesh| Use {
                    mesh,
                    weights: Vec::new(),
                    skin: None,
                    transforms: vec![IDENTITY],
                    thickness: 1.0,
                })
                .collect());
        };

        // The meshes with a volume's thickness, which their nodes scale: a use is of one scale.
        let thick: Vec<bool> = self.materials_json.iter().map(material::thick).collect();
        let thick: Vec<bool> = if thick.contains(&true) {
            let is = |p: &Value| index(p, "material").is_some_and(|m| thick.get(m) == Some(&true));
            let any = |m: &Value| items(m, "primitives").iter().any(is);
            self.meshes.iter().map(any).collect()
        } else {
            Vec::new()
        };
        let mut uses: Vec<Use> = Vec::new();
        // Each use by (mesh, skin, weights' bits, thickness scale's bits): a scan of every use per
        // node is quadratic in a scene of many meshes.
        let mut seen: HashMap<(usize, Option<usize>, Vec<u32>, u32), usize> = HashMap::new();
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
            let bits: Vec<u32> = weights.iter().map(|x| x.to_bits()).collect();
            // A skin's joints scale its thickness, not its node.
            let scaled = skin.is_none() && thick.get(mesh) == Some(&true);
            let mut rest = transforms;
            while !rest.is_empty() {
                // The transforms of the first one's scale, in the order they came.
                let (transforms, thickness) = if scaled {
                    let s = self.thickness_scale(&rest[0]);
                    let same = |t: &M4| self.thickness_scale(t).to_bits() == s.to_bits();
                    let (same, other) = rest.into_iter().partition(same);
                    rest = other;
                    (same, s)
                } else {
                    (std::mem::take(&mut rest), 1.0)
                };
                let key = (mesh, skin, bits.clone(), thickness.to_bits());
                match seen.get(&key) {
                    Some(_) if skin.is_some() => {}
                    Some(&u) => uses[u].transforms.extend(transforms),
                    None => {
                        seen.insert(key, uses.len());
                        uses.push(Use {
                            mesh,
                            weights: weights.clone(),
                            skin,
                            transforms,
                            thickness,
                        });
                    }
                }
            }
        }
        // Joints need every node's transform, in the scene or not.
        self.world = world.into_iter().map(|w| w.unwrap_or(IDENTITY)).collect();
        Ok(uses)
    }

    /// What a transform scales a volume's thickness by: the cube root of how much it scales volumes,
    /// which is its scale when that is uniform.
    fn thickness_scale(&mut self, m: &M4) -> f32 {
        self.scale_met(det3(m).abs().cbrt())
    }

    /// A thickness scale as the one within rounding of it met before, if any: 1 at first, so rotations
    /// don't make materials of their own.
    fn scale_met(&mut self, s: f64) -> f32 {
        let s = f32_of(s);
        if !s.is_finite() {
            return 1.0;
        }
        if let Some(&near) = self.scales.iter().find(|&&k| (k - s).abs() <= 1e-6 * k) {
            return near;
        }
        self.scales.push(s);
        s
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
        ensure!(
            count <= self.doc.max_triangles,
            LimitExceeded,
            "glTF node has {count} instances"
        );
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

/// Translation, rotation (a quaternion x y z w) and scale as one matrix.
pub(crate) fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
    let q = if quat::length(q) > 0.0 {
        quat::unit_unchecked(q)
    } else {
        quat::IDENTITY
    };
    let r = quat::matrix(q);
    let mut m = IDENTITY;
    for col in 0..3 {
        for row in 0..3 {
            m[col * 4 + row] = r[col * 3 + row] * s[col];
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
    let q = quat::unit(quat::of_rotation_reciprocal(|row, c| m[c * 4 + row] / s[c]));
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
