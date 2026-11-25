//! Writes a [`Model`] as binary glTF: one node per component instance carrying its transform, one mesh per
//! definition and inherited material, normals averaged across smooth edges and split at hard ones, and `SketchUp`'s
//! inches and Z-up turned into meters and Y-up. Hidden entities, and entities on hidden tags, are kept as nodes
//! that `KHR_node_visibility` hides.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::{Context as _, ErrorKind, Result};
use serde_json::{Value, json};

use super::model::{Definition, Face, Id, Instance, Model};
use super::triangulate::{newell, triangulate};
use crate::vec3::{cross, dot, sub, unit};

const INCH: f64 = 0.0254;
/// More nodes than any real model has; a file that nests big instance arrays inside each other could otherwise ask
/// for billions.
const MAX_NODES: usize = 1_000_000;
/// Total vertices across all meshes for a load that takes `triangles` triangles: four to each, which bounds the
/// binary buffer to a few hundred megabytes at the default limit, and within what a 32-bit index reaches.
fn max_vertices(triangles: usize) -> usize {
    triangles
        .saturating_mul(4)
        .min(usize::try_from(u32::MAX).unwrap_or(usize::MAX))
}
/// Component nesting deeper than this is skipped rather than recursed into.
const MAX_NESTING: usize = 64;
/// `SketchUp`'s default front-face color, for faces nobody painted.
const DEFAULT_RGB: [u8; 3] = [255, 255, 255];

/// The image type of an embedded texture, from its leading bytes. Anything other than PNG or JPEG is dropped,
/// since glTF viewers only have to understand those two.
pub(crate) fn image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else {
        None
    }
}

/// Whether a PNG has an alpha channel (grayscale-alpha or RGBA in its header), so cut-out textures like leaves
/// and fences keep their holes.
fn png_has_alpha(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n") && matches!(bytes.get(25), Some(4 | 6))
}

/// The color factor that shows a colorized texture near the color it was given: the material's color over the
/// image's average brightness, so the image's light and dark parts keep their spread around that color. Without a
/// recorded average the color is used as it is. `SketchUp` shifts the image's hue instead, which would need the
/// image decoded and re-encoded.
fn tint(rgb: [u8; 3], average: Option<[u8; 3]>) -> [f64; 3] {
    let target = rgb.map(srgb_to_linear);
    let Some(average) = average else { return target };
    let [r, g, b] = average.map(srgb_to_linear);
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if luma < 1e-4 {
        return target;
    }
    target.map(|c| (c / luma).clamp(0.0, 1.0))
}

fn srgb_to_linear(c: u8) -> f64 {
    let c = f64::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// `SketchUp`'s inches, Z up, to glTF's meters, Y up. A rotation, so it keeps winding.
fn to_gltf(p: [f64; 3]) -> [f64; 3] {
    [p[0] * INCH, p[2] * INCH, -p[1] * INCH]
}

/// `SketchUp`'s 13-value transform as a column-major glTF matrix in meters and Y up, or `None` when it holds
/// non-finite values. Index 12 is a uniform scale that multiplies the rest.
fn node_matrix(m: &[f64; 13]) -> Option<[f64; 16]> {
    let s = if m[12] == 0.0 { 1.0 } else { m[12] };
    let a = |r: usize, c: usize| m[r * 3 + c] * s;
    // swap axes on both sides: R * A * R^T with R = (x, y, z) -> (x, z, -y)
    let src = [0, 2, 1];
    let sign = [1.0, 1.0, -1.0];
    let mut out = [0.0; 16];
    for r in 0..3 {
        for c in 0..3 {
            out[c * 4 + r] = sign[r] * sign[c] * a(src[r], src[c]);
        }
    }
    out[12] = m[9] * s * INCH;
    out[13] = m[11] * s * INCH;
    out[14] = -m[10] * s * INCH;
    out[15] = 1.0;
    out.iter().all(|v| v.is_finite()).then_some(out)
}

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Face-plane axes for `SketchUp`'s default texture projection: U along Z × n, V along n × U, snapping near-flat
/// faces to world X and Y.
fn uv_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (cx, cy) = (-n[1], n[0]);
    let len = cx.hypot(cy);
    if len < 1e-3 {
        return ([if n[2] >= 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0], [0.0, 1.0, 0.0]);
    }
    let xr = [cx / len, cy / len, 0.0];
    (xr, cross(n, xr))
}

#[expect(
    clippy::many_single_char_names,
    reason = "the textbook names for a 3x3 matrix's entries"
)]
fn invert_3x3(m: &[f64; 9]) -> [f64; 9] {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 || !det.is_finite() {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let k = 1.0 / det;
    [
        (e * i - f * h) * k,
        (c * h - b * i) * k,
        (b * f - c * e) * k,
        (f * g - d * i) * k,
        (a * i - c * g) * k,
        (c * d - a * f) * k,
        (d * h - e * g) * k,
        (b * g - a * h) * k,
        (a * e - b * d) * k,
    ]
}

/// A point's texture coordinate, with V flipped because glTF images start at the top and `SketchUp`'s at the bottom.
fn face_uv(p: [f64; 3], basis: ([f64; 3], [f64; 3]), inv: Option<&[f64; 9]>, tile: [f64; 2]) -> [f32; 2] {
    let (px, py) = (dot(p, basis.0), dot(p, basis.1));
    let (u, v) = match inv {
        None => (px, py),
        Some(i) => {
            let q = px * i[2] + py * i[5] + i[8];
            let q = if q.abs() < 1e-12 { 1.0 } else { q };
            ((px * i[0] + py * i[3] + i[6]) / q, (px * i[1] + py * i[4] + i[7]) / q)
        }
    };
    #[expect(clippy::cast_possible_truncation, reason = "glTF texture coordinates are f32")]
    [(u / tile[0]) as f32, (1.0 - v / tile[1]) as f32]
}

#[derive(Default)]
struct Prim {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    indices: Vec<u32>,
}

/// A material as glTF sees it: the `SketchUp` material (`None` for unpainted faces) and whether it shows on both
/// sides of a face.
type MaterialKey = (Option<Id>, bool);

/// Hidden geometry, set aside until the visible model is written so that it can never push that over the budget.
enum Pending<'m> {
    /// A definition's hidden faces, as a child of node `parent`.
    Faces {
        parent: usize,
        def: &'m Definition,
        key: Option<Id>,
        inherited: Option<Id>,
    },
    /// A hidden instance, as a child of node `parent`, with the definitions that were open above it.
    Instance {
        parent: usize,
        inst: &'m Instance,
        inherited: Option<Id>,
        active: Vec<Id>,
    },
}

struct Writer<'m> {
    model: &'m Model,
    hidden_layers: HashSet<Id>,
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    meshes: Vec<Value>,
    nodes: Vec<Value>,
    materials: Vec<Value>,
    material_index: HashMap<MaterialKey, usize>,
    images: Vec<Value>,
    image_index: HashMap<&'m [u8], usize>,
    /// Meshes by definition, inherited material and whether they hold the hidden faces.
    mesh_cache: HashMap<(Option<Id>, Option<Id>, bool), Option<usize>>,
    active: Vec<Id>,
    vertices: usize,
    /// The most vertices the meshes may have.
    max_vertices: usize,
    pending: Vec<Pending<'m>>,
    /// Set while writing hidden geometry: it is built straight away rather than set aside, and what would go over
    /// a budget is left out rather than failing the load.
    lenient: bool,
    /// Whether any node is hidden, so the file declares `KHR_node_visibility`.
    hidden_nodes: bool,
}

/// Converts the model to a GLB file, or `None` when it has no visible faces; `max_triangles` (the load's limit)
/// bounds its vertices.
pub(crate) fn write(model: &Model, max_triangles: usize) -> Result<Option<Vec<u8>>> {
    let mut w = Writer {
        model,
        hidden_layers: model.hidden_layers.iter().copied().collect(),
        bin: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
        meshes: Vec::new(),
        nodes: Vec::new(),
        materials: Vec::new(),
        material_index: HashMap::new(),
        images: Vec::new(),
        image_index: HashMap::new(),
        mesh_cache: HashMap::new(),
        active: Vec::new(),
        vertices: 0,
        max_vertices: max_vertices(max_triangles),
        pending: Vec::new(),
        lenient: false,
        hidden_nodes: false,
    };
    let root = w.node(&model.root, None, None, "SketchUp model")?;
    if w.vertices == 0 {
        return Ok(None);
    }
    w.lenient = true;
    for p in std::mem::take(&mut w.pending) {
        let (parent, child) = match p {
            Pending::Faces {
                parent,
                def,
                key,
                inherited,
            } => (parent, w.hidden_faces(def, key, inherited)?),
            Pending::Instance {
                parent,
                inst,
                inherited,
                active,
            } => {
                w.active = active;
                (parent, w.instance(inst, inherited, true)?)
            }
        };
        if let Some(child) = child
            && let Some(node) = w.nodes.get_mut(parent)
        {
            match node.get_mut("children").and_then(Value::as_array_mut) {
                Some(children) => children.push(json!(child)),
                None => node["children"] = json!([child]),
            }
        }
    }
    w.finish(root).map(Some)
}

impl<'m> Writer<'m> {
    fn visible(&self, hidden: bool, layer: Option<Id>) -> bool {
        !hidden && layer.is_none_or(|l| !self.hidden_layers.contains(&l))
    }

    /// The node for one placement of `def`, with nodes for the instances inside it as its children.
    fn node(&mut self, def: &'m Definition, key: Option<Id>, inherited: Option<Id>, name: &str) -> Result<usize> {
        let mesh = self.mesh(def, key, inherited, false)?;
        let mut children = Vec::new();
        let mut hidden = Vec::new();
        for inst in &def.instances {
            if self.active.len() >= MAX_NESTING {
                break;
            }
            if self.visible(inst.hidden, inst.layer) {
                children.extend(self.instance(inst, inherited, false)?);
            } else if self.lenient {
                children.extend(self.instance(inst, inherited, true)?);
            } else {
                hidden.push(inst);
            }
        }
        let hidden_faces = def.faces.iter().any(|f| !self.visible(f.hidden, f.layer));
        if hidden_faces && self.lenient {
            children.extend(self.hidden_faces(def, key, inherited)?);
        }
        let mut node = json!({ "name": name });
        if let Some(mesh) = mesh {
            node["mesh"] = json!(mesh);
        }
        if !children.is_empty() {
            node["children"] = json!(children);
        }
        self.nodes.push(node);
        let parent = self.nodes.len() - 1;
        if !self.lenient {
            if hidden_faces {
                self.pending.push(Pending::Faces {
                    parent,
                    def,
                    key,
                    inherited,
                });
            }
            for inst in hidden {
                let active = self.active.clone();
                self.pending.push(Pending::Instance {
                    parent,
                    inst,
                    inherited,
                    active,
                });
            }
        }
        Ok(parent)
    }

    /// The node for one instance, hidden with `KHR_node_visibility` when `hidden` is set, or `None` when it cannot
    /// be placed.
    fn instance(&mut self, inst: &'m Instance, inherited: Option<Id>, hidden: bool) -> Result<Option<usize>> {
        let Some(child) = self.model.definitions.get(&inst.definition) else {
            return Ok(None);
        };
        let Some(matrix) = node_matrix(&inst.matrix) else {
            return Ok(None);
        };
        // a definition that contains itself would recurse forever
        if self.active.contains(&inst.definition) {
            return Ok(None);
        }
        if self.nodes.len() >= MAX_NODES {
            ensure!(
                self.lenient,
                LimitExceeded,
                "the SketchUp model has too many component instances"
            );
            return Ok(None);
        }
        self.active.push(inst.definition);
        let name = if inst.name.is_empty() { &child.name } else { &inst.name };
        let index = self.node(child, Some(inst.definition), inst.material.or(inherited), name);
        self.active.pop();
        let index = index?;
        if let Some(node) = self.nodes.get_mut(index) {
            #[allow(clippy::float_cmp_const, reason = "only the exact identity may leave the matrix out")]
            if matrix != IDENTITY {
                node["matrix"] = json!(matrix);
            }
            if hidden {
                node["extensions"] = json!({ "KHR_node_visibility": { "visible": false } });
                self.hidden_nodes = true;
            }
        }
        Ok(Some(index))
    }

    /// A hidden node holding a definition's hidden faces, or `None` when they would go over a budget.
    fn hidden_faces(&mut self, def: &'m Definition, key: Option<Id>, inherited: Option<Id>) -> Result<Option<usize>> {
        if self.nodes.len() >= MAX_NODES {
            return Ok(None);
        }
        let Some(mesh) = self.mesh(def, key, inherited, true)? else {
            return Ok(None);
        };
        self.nodes.push(json!({
            "name": "Hidden geometry",
            "mesh": mesh,
            "extensions": { "KHR_node_visibility": { "visible": false } },
        }));
        self.hidden_nodes = true;
        Ok(Some(self.nodes.len() - 1))
    }

    /// A definition's visible faces as a mesh, or its hidden ones when `hidden` is set, painting unpainted faces
    /// with the material the placement inherited. Built once per definition and inherited material, then shared by
    /// every node that needs it.
    ///
    /// A face whose back is painted differently from its front becomes two single-sided faces back to back: the
    /// front, and the back with its winding and normals reversed and its own material and texture position. A face
    /// with only its front painted, or both sides the same, shows its front on both sides, since `SketchUp`'s
    /// default back color is only a cue for which way a face points.
    fn mesh(
        &mut self,
        def: &'m Definition,
        key: Option<Id>,
        inherited: Option<Id>,
        hidden: bool,
    ) -> Result<Option<usize>> {
        if let Some(&cached) = self.mesh_cache.get(&(key, inherited, hidden)) {
            return Ok(cached);
        }
        let faces: Vec<&Face> = def
            .faces
            .iter()
            .filter(|f| self.visible(f.hidden, f.layer) != hidden)
            .collect();
        let normals = smooth_normals(&faces);
        let start = self.vertices;
        let mut prims: Vec<(MaterialKey, Prim)> = Vec::new();
        let mut corner = 0;
        for face in faces {
            let corners = face.loops.iter().map(Vec::len).sum::<usize>();
            let smoothed = normals.as_ref().and_then(|n| n.get(corner..corner + corners));
            corner += corners;
            let painted = |m: Option<Id>| m.or(inherited).filter(|m| self.model.materials.contains_key(m));
            let front = painted(face.material);
            let back = painted(face.back_material);
            let both = back.is_none() || back == front;
            if self.lenient && self.vertices.saturating_add(corners * if both { 1 } else { 2 }) > self.max_vertices {
                self.vertices = start;
                self.mesh_cache.insert((key, inherited, hidden), None);
                return Ok(None);
            }
            let sides: &[(Option<Id>, bool)] = if both {
                &[(front, false)]
            } else {
                &[(front, false), (back, true)]
            };
            for &(mat, reverse) in sides {
                let key = (mat, both);
                let slot = if let Some(i) = prims.iter().position(|p| p.0 == key) {
                    i
                } else {
                    prims.push((key, Prim::default()));
                    prims.len() - 1
                };
                let texture = mat
                    .and_then(|m| self.model.materials.get(&m))
                    .and_then(|m| m.texture.as_ref());
                let tile = texture.map(|t| t.tile.map(|s| if s.abs() > 1e-9 && s.is_finite() { s } else { 1.0 }));
                self.vertices += add_face(&mut prims[slot].1, face, tile, reverse, smoothed);
            }
            ensure!(
                self.vertices <= self.max_vertices,
                LimitExceeded,
                "the SketchUp model is too large to convert"
            );
        }
        let mut primitives = Vec::new();
        for (key, prim) in prims {
            if prim.indices.is_empty() {
                continue;
            }
            let material = self.material(key);
            primitives.push(self.primitive(&prim, material));
        }
        let mesh = if primitives.is_empty() {
            None
        } else {
            let mut m = json!({ "primitives": primitives });
            if !def.name.is_empty() {
                m["name"] = json!(def.name);
            }
            self.meshes.push(m);
            Some(self.meshes.len() - 1)
        };
        self.mesh_cache.insert((key, inherited, hidden), mesh);
        Ok(mesh)
    }

    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let mut v = json!({ "buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.bin.extend_from_slice(bytes);
        self.views.push(v);
        self.views.len() - 1
    }

    fn floats(&mut self, values: &[f32], kind: &str, bounds: bool) -> usize {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(34962));
        let width = if kind == "VEC3" { 3 } else { 2 };
        let mut a = json!({
            "bufferView": view, "componentType": 5126, "count": values.len() / width, "type": kind,
        });
        if bounds {
            let mut min = vec![f32::INFINITY; width];
            let mut max = vec![f32::NEG_INFINITY; width];
            for chunk in values.chunks_exact(width) {
                for (k, &v) in chunk.iter().enumerate() {
                    min[k] = min[k].min(v);
                    max[k] = max[k].max(v);
                }
            }
            a["min"] = json!(min);
            a["max"] = json!(max);
        }
        self.accessors.push(a);
        self.accessors.len() - 1
    }

    fn primitive(&mut self, prim: &Prim, material: usize) -> Value {
        let position = self.floats(&prim.positions, "VEC3", true);
        let normal = self.floats(&prim.normals, "VEC3", false);
        let mut attributes = json!({ "POSITION": position, "NORMAL": normal });
        if !prim.uvs.is_empty() {
            attributes["TEXCOORD_0"] = json!(self.floats(&prim.uvs, "VEC2", false));
        }
        let bytes: Vec<u8> = prim.indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(34963));
        self.accessors.push(json!({
            "bufferView": view, "componentType": 5125, "count": prim.indices.len(), "type": "SCALAR",
        }));
        json!({ "attributes": attributes, "indices": self.accessors.len() - 1, "material": material })
    }

    /// The glTF material for a `SketchUp` material (unpainted faces when `None`), shown on both sides of a face or
    /// on its front only.
    fn material(&mut self, key: MaterialKey) -> usize {
        if let Some(&i) = self.material_index.get(&key) {
            return i;
        }
        let (id, double_sided) = key;
        let model = self.model;
        let (name, rgb, alpha, texture, colorize) = match id.and_then(|m| model.materials.get(&m)) {
            Some(m) => (m.name.as_str(), m.rgb, m.alpha, m.texture.as_ref(), m.colorize),
            None => ("Default", DEFAULT_RGB, 1.0, None, false),
        };
        let alpha = if alpha.is_finite() { alpha.clamp(0.0, 1.0) } else { 1.0 };
        let mut pbr = json!({ "metallicFactor": 0.0, "roughnessFactor": 0.9 });
        let mut cutout = false;
        if let Some(tex) = texture {
            let [r, g, b] = if colorize { tint(rgb, tex.average) } else { [1.0; 3] };
            pbr["baseColorFactor"] = json!([r, g, b, alpha]);
            pbr["baseColorTexture"] = json!({ "index": self.texture(&tex.bytes) });
            cutout = png_has_alpha(&tex.bytes);
        } else {
            let [r, g, b] = rgb.map(srgb_to_linear);
            pbr["baseColorFactor"] = json!([r, g, b, alpha]);
        }
        let mut m = json!({ "name": name, "pbrMetallicRoughness": pbr, "doubleSided": double_sided });
        if alpha < 0.999 {
            m["alphaMode"] = json!("BLEND");
        } else if cutout {
            m["alphaMode"] = json!("MASK");
            m["alphaCutoff"] = json!(0.5);
        }
        self.materials.push(m);
        let i = self.materials.len() - 1;
        self.material_index.insert(key, i);
        i
    }

    /// The texture for an embedded image, adding the image once however many materials share it.
    fn texture(&mut self, bytes: &'m [u8]) -> usize {
        if let Some(&i) = self.image_index.get(bytes) {
            return i;
        }
        let view = self.view(bytes, None);
        let mime = image_mime(bytes).unwrap_or("image/png");
        self.images.push(json!({ "bufferView": view, "mimeType": mime }));
        let i = self.images.len() - 1;
        self.image_index.insert(bytes, i);
        i
    }

    fn finish(self, root: usize) -> Result<Vec<u8>> {
        let mut doc = json!({
            "asset": { "version": "2.0", "generator": "polysmelt" },
            "scene": 0,
            "scenes": [{ "nodes": [root] }],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{ "byteLength": self.bin.len() }],
        });
        if !self.images.is_empty() {
            let textures: Vec<Value> = (0..self.images.len())
                .map(|i| json!({ "source": i, "sampler": 0 }))
                .collect();
            doc["images"] = json!(self.images);
            doc["textures"] = json!(textures);
            doc["samplers"] = json!([{ "wrapS": 10497, "wrapT": 10497 }]);
        }
        if self.hidden_nodes {
            doc["extensionsUsed"] = json!(["KHR_node_visibility"]);
        }
        let mut text = serde_json::to_vec(&doc).context(ErrorKind::Invalid, "writing the glTF JSON")?;
        while !text.len().is_multiple_of(4) {
            text.push(b' ');
        }
        let mut bin = self.bin;
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + text.len() + 8 + bin.len();
        let (Ok(total32), Ok(text32), Ok(bin32)) = (
            u32::try_from(total),
            u32::try_from(text.len()),
            u32::try_from(bin.len()),
        ) else {
            bail!(LimitExceeded, "the converted model is larger than a GLB file can hold");
        };
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&total32.to_le_bytes());
        out.extend_from_slice(&text32.to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&text);
        out.extend_from_slice(&bin32.to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
        Ok(out)
    }
}

/// A face's unit normal: the one the file gives, or failing that its outer loop's.
fn face_normal(face: &Face) -> Option<[f64; 3]> {
    unit(face.normal).or_else(|| face.loops.first().and_then(|l| newell(l)))
}

/// The root of `x`'s set, halving the path on the way.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

/// Per-corner normals for `faces`, one face after another and each face's loops flattened, as `SketchUp` shades
/// them: a corner's normal averages the normals of the faces around it that smooth edges join, each weighted by
/// its angle at that corner, and hard edges split them. `None` when no edge is smooth, so every face keeps its flat
/// normal without the cost of working any of this out.
///
/// Corners are joined across a smooth edge only when the faces either side run along it in opposite directions,
/// as consistently oriented neighbours do; a face flipped against its neighbour would cancel their normals out.
fn smooth_normals(faces: &[&Face]) -> Option<Vec<[f64; 3]>> {
    if faces.iter().all(|f| f.smooth.is_empty()) {
        return None;
    }
    let total: usize = faces.iter().map(|f| f.loops.iter().map(Vec::len).sum::<usize>()).sum();
    let count = u32::try_from(total).ok()?;
    // vertices welded by position, for the faces that have a smooth edge; -0.0 + 0.0 is 0.0, so both zeros weld
    let mut weld: HashMap<[u64; 3], u32> = HashMap::with_capacity(total);
    let mut ids: Vec<u32> = Vec::with_capacity(total);
    let joins = |f: &Face| f.smooth.len() == f.loops.iter().map(Vec::len).sum::<usize>();
    for f in faces {
        let joined = joins(f);
        for p in f.loops.iter().flatten() {
            let id = if joined {
                let next = u32::try_from(weld.len()).unwrap_or(u32::MAX);
                *weld.entry(p.map(|v| (v + 0.0).to_bits())).or_insert(next)
            } else {
                u32::MAX
            };
            ids.push(id);
        }
    }
    drop(weld);
    let mut parent: Vec<u32> = (0..count).collect();
    // each smooth edge by its two vertices: the corners at its start and end on the first face seen along it
    let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(total / 2);
    let mut base = 0u32;
    for f in faces {
        let corners = f.loops.iter().map(Vec::len).sum::<usize>();
        if joins(f) {
            let mut at = base;
            for l in &f.loops {
                let Ok(len) = u32::try_from(l.len()) else { return None };
                for k in 0..len {
                    let (c, d) = (at + k, at + (k + 1) % len);
                    if !f.smooth[(c - base) as usize] {
                        continue;
                    }
                    let (a, b) = (ids[c as usize], ids[d as usize]);
                    if a == b {
                        continue;
                    }
                    match edges.entry((a.min(b), a.max(b))) {
                        Entry::Vacant(e) => {
                            e.insert((c, d));
                        }
                        Entry::Occupied(e) => {
                            let (c2, d2) = *e.get();
                            if ids[c2 as usize] == b {
                                for (x, y) in [(c, d2), (d, c2)] {
                                    let (x, y) = (find(&mut parent, x), find(&mut parent, y));
                                    parent[x as usize] = y;
                                }
                            }
                        }
                    }
                }
                at += len;
            }
        }
        base += u32::try_from(corners).ok()?;
    }
    drop(edges);
    // angle-weighted sums of face normals, gathered at each set's root
    let mut sums = vec![[0.0f64; 3]; total];
    let mut flat = vec![[0.0f64; 3]; total];
    let mut c = 0u32;
    for f in faces {
        let n = face_normal(f).unwrap_or_default();
        for l in &f.loops {
            for (k, &p) in l.iter().enumerate() {
                let prev = l[(k + l.len() - 1) % l.len()];
                let next = l[(k + 1) % l.len()];
                let (e1, e2) = (sub(prev, p), sub(next, p));
                let x = cross(e1, e2);
                let angle = dot(x, x).sqrt().atan2(dot(e1, e2));
                let weight = if angle.is_finite() { angle } else { 0.0 };
                let root = find(&mut parent, c) as usize;
                for (s, v) in sums[root].iter_mut().zip(n) {
                    *s += v * weight;
                }
                flat[c as usize] = n;
                c += 1;
            }
        }
    }
    for c in 0..count {
        let root = find(&mut parent, c) as usize;
        let n = flat[c as usize];
        if let Some(s) = unit(sums[root]).filter(|s| dot(*s, n) > 0.0) {
            flat[c as usize] = s;
        }
    }
    Some(flat)
}

/// Triangulates one face into `prim`, its back when `reverse` is set, with `smoothed` giving each corner's normal
/// in place of the face's own. Returns how many vertices it added; faces with non-finite points add none.
#[expect(clippy::cast_possible_truncation, reason = "glTF vertex data is f32")]
fn add_face(
    prim: &mut Prim,
    face: &Face,
    tile: Option<[f64; 2]>,
    reverse: bool,
    smoothed: Option<&[[f64; 3]]>,
) -> usize {
    let points: Vec<[f64; 3]> = face.loops.iter().flatten().copied().collect();
    let out: Vec<[f32; 3]> = points.iter().map(|&p| to_gltf(p).map(|v| v as f32)).collect();
    if out.iter().flatten().any(|v| !v.is_finite()) {
        return 0;
    }
    let Some(n) = face_normal(face) else {
        return 0;
    };
    let tris = triangulate(&face.loops, n);
    let Ok(base) = u32::try_from(prim.positions.len() / 3) else {
        return 0;
    };
    if tris.is_empty()
        || u32::try_from(points.len())
            .ok()
            .and_then(|c| base.checked_add(c))
            .is_none()
    {
        return 0;
    }
    let sign = if reverse { -1.0 } else { 1.0 };
    let gltf_normal = |n: [f64; 3]| to_gltf(n).map(|v| (sign * v / INCH) as f32);
    let normal = gltf_normal(n);
    let matrix = if reverse { face.back_uv_matrix } else { face.uv_matrix };
    let inv = matrix.as_ref().map(invert_3x3);
    let basis = uv_basis(n);
    let smoothed = smoothed.filter(|s| s.len() == points.len());
    for (i, (p, q)) in points.iter().zip(&out).enumerate() {
        prim.positions.extend_from_slice(q);
        match smoothed.and_then(|s| s.get(i)) {
            Some(&s) => prim.normals.extend_from_slice(&gltf_normal(s)),
            None => prim.normals.extend_from_slice(&normal),
        }
        if let Some(tile) = tile {
            prim.uvs.extend_from_slice(&face_uv(*p, basis, inv.as_ref(), tile));
        }
    }
    for t in tris {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the face's vertex count was checked to fit"
        )]
        let [a, b, c] = t.map(|i| base + i as u32);
        prim.indices.extend(if reverse { [a, c, b] } else { [a, b, c] });
    }
    points.len()
}

#[cfg(test)]
#[allow(clippy::float_cmp_const, reason = "the conversions are exact")]
mod tests {
    use super::super::model::{Instance, Material};
    use super::*;
    use crate::mesh::MAX_TRIANGLES;

    fn square_face(material: Option<Id>) -> Face {
        Face {
            loops: vec![vec![
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]],
            smooth: Vec::new(),
            normal: [0.0, 0.0, 1.0],
            material,
            back_material: None,
            uv_matrix: None,
            back_uv_matrix: None,
            hidden: false,
            layer: None,
        }
    }

    #[test]
    fn axis_swap_is_a_rotation() {
        assert_eq!(to_gltf([1.0, 0.0, 0.0]), [INCH, 0.0, 0.0]);
        // SketchUp's up becomes glTF's up, and SketchUp's north (+Y) points away from a glTF viewer (-Z)
        assert_eq!(to_gltf([0.0, 0.0, 1.0]), [0.0, INCH, 0.0]);
        assert_eq!(to_gltf([0.0, 1.0, 0.0]), [0.0, 0.0, -INCH]);
    }

    #[test]
    fn node_matrices_match_baked_points() {
        // rotate 90 degrees about Z, then move by (1, 2, 3) inches
        let m = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 1.0];
        let g = node_matrix(&m).unwrap();
        let p = [5.0, 7.0, 11.0];
        let moved = [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[9],
            m[3] * p[0] + m[4] * p[1] + m[5] * p[2] + m[10],
            m[6] * p[0] + m[7] * p[1] + m[8] * p[2] + m[11],
        ];
        let want = to_gltf(moved);
        let q = to_gltf(p);
        for r in 0..3 {
            let got = g[r] * q[0] + g[4 + r] * q[1] + g[8 + r] * q[2] + g[12 + r];
            assert!((got - want[r]).abs() < 1e-12, "row {r}: {got} vs {}", want[r]);
        }
        assert!(node_matrix(&[f64::NAN; 13]).is_none());
    }

    /// The JSON chunk of a GLB.
    fn json_of(glb: &[u8]) -> Value {
        let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        serde_json::from_slice(&glb[20..20 + len]).unwrap()
    }

    #[test]
    fn writes_a_glb_with_instances_and_hides_hidden_ones() {
        let mut model = Model::default();
        model.materials.insert(
            7,
            Material {
                name: "Red".to_owned(),
                rgb: [255, 0, 0],
                alpha: 0.5,
                texture: None,
                colorize: false,
            },
        );
        model.definitions.insert(
            3,
            Definition {
                name: "Box".to_owned(),
                faces: vec![square_face(None)],
                instances: vec![],
            },
        );
        let mut inst = Instance {
            definition: 3,
            name: String::new(),
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            material: Some(7),
            hidden: false,
            layer: None,
        };
        model.root.instances.push(inst.clone());
        inst.hidden = true;
        model.root.instances.push(inst.clone());
        inst.hidden = false;
        inst.layer = Some(9);
        model.hidden_layers.push(9);
        model.root.instances.push(inst);
        model.root.faces.push(square_face(None));
        let glb = write(&model, MAX_TRIANGLES).unwrap().unwrap();
        let doc = gltf::Gltf::from_slice(&glb).unwrap();
        assert_eq!(doc.buffers().count(), 1);
        let root = doc.default_scene().unwrap().nodes().next().unwrap();
        // the visible instance, then the hidden one and the one on a hidden tag
        assert_eq!(root.children().count(), 3);
        let json = json_of(&glb);
        assert_eq!(json["extensionsUsed"], json!(["KHR_node_visibility"]));
        let hidden: Vec<bool> = root
            .children()
            .map(|c| json["nodes"][c.index()]["extensions"]["KHR_node_visibility"]["visible"] == json!(false))
            .collect();
        assert_eq!(hidden, [false, true, true]);
        let mats: Vec<_> = json["materials"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert!(mats.contains(&"Red") && mats.contains(&"Default"));
    }

    #[test]
    fn the_triangle_limit_bounds_the_vertices() {
        // Two squares: four triangles, and eight vertices (four to each face, for its flat normal).
        let mut model = Model::default();
        model.root.faces.extend([square_face(None), square_face(None)]);
        let err = write(&model, 1).unwrap_err();
        assert!(format!("{err:#}").contains("too large"), "{err:#}");
        assert!(write(&model, 2).unwrap().is_some());
        assert_eq!(max_vertices(MAX_TRIANGLES), 4 * MAX_TRIANGLES);
        assert_eq!(max_vertices(crate::mesh::TRIANGLE_CEILING), u32::MAX as usize);
    }

    #[test]
    fn hidden_faces_are_a_hidden_node_and_never_break_the_budget() {
        let mut model = Model::default();
        let mut hidden = square_face(None);
        hidden.hidden = true;
        model.root.faces.extend([square_face(None), hidden]);
        let json = json_of(&write(&model, 2).unwrap().unwrap());
        assert_eq!(json["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(json["nodes"][0]["children"], json!([1]));
        assert_eq!(
            json["nodes"][1]["extensions"]["KHR_node_visibility"]["visible"],
            json!(false)
        );
        // with room for the visible square only, the hidden one is left out rather than failing the load
        let json = json_of(&write(&model, 1).unwrap().unwrap());
        assert_eq!(json["nodes"].as_array().unwrap().len(), 1);
        assert!(json.get("extensionsUsed").is_none());
    }

    #[test]
    fn a_differently_painted_back_is_its_own_single_sided_face() {
        let mut model = Model::default();
        for (id, name) in [(7, "Front"), (8, "Back")] {
            let m = Material {
                name: name.to_owned(),
                rgb: [255, 0, 0],
                alpha: 1.0,
                texture: None,
                colorize: false,
            };
            model.materials.insert(id, m);
        }
        let mut f = square_face(Some(7));
        f.back_material = Some(8);
        model.root.faces.push(f);
        let glb = write(&model, MAX_TRIANGLES).unwrap().unwrap();
        let json = json_of(&glb);
        let mats = json["materials"].as_array().unwrap();
        assert_eq!(mats.len(), 2);
        assert!(mats.iter().all(|m| m["doubleSided"] == json!(false)));
        let doc = gltf::Gltf::from_slice(&glb).unwrap();
        let bin = doc.blob.as_deref().unwrap();
        let prims: Vec<_> = doc.meshes().next().unwrap().primitives().collect();
        assert_eq!(prims.len(), 2);
        let read = |p: &gltf::Primitive<'_>| {
            let r = p.reader(|_| Some(bin));
            let normals: Vec<[f32; 3]> = r.read_normals().unwrap().collect();
            let indices: Vec<u32> = r.read_indices().unwrap().into_u32().collect();
            (normals, indices)
        };
        let (front_n, front_i) = read(&prims[0]);
        let (back_n, back_i) = read(&prims[1]);
        assert_eq!(front_n[0], [0.0, 1.0, 0.0]);
        assert_eq!(back_n[0], [0.0, -1.0, 0.0]);
        for (f, b) in front_i.chunks_exact(3).zip(back_i.chunks_exact(3)) {
            assert_eq!([f[0], f[2], f[1]], b);
        }

        // painted the same on both sides, or on the front only, it stays one face showing on both
        for back in [Some(7), None] {
            model.root.faces[0].back_material = back;
            let json = json_of(&write(&model, MAX_TRIANGLES).unwrap().unwrap());
            assert_eq!(json["materials"].as_array().unwrap().len(), 1);
            assert_eq!(json["materials"][0]["doubleSided"], json!(true));
        }
    }

    /// A floor facing up and a wall facing +X meeting along the Y axis, the edge between them smooth when
    /// `smooth` is set.
    fn fold(smooth: bool) -> [Face; 2] {
        let mut floor = square_face(None);
        let mut wall = square_face(None);
        wall.loops = vec![vec![
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0],
        ]];
        wall.normal = [1.0, 0.0, 0.0];
        if smooth {
            // the floor runs along the shared edge from its last corner, the wall from its first
            floor.smooth = vec![false, false, false, true];
            wall.smooth = vec![true, false, false, false];
        }
        [floor, wall]
    }

    #[test]
    fn smooth_edges_average_normals_and_hard_edges_split_them() {
        assert!(smooth_normals(&fold(false).iter().collect::<Vec<_>>()).is_none());
        let faces = fold(true);
        let n = smooth_normals(&faces.iter().collect::<Vec<_>>()).unwrap();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let close = |a: [f64; 3], b: [f64; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12);
        let want = [
            [h, 0.0, h],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [h, 0.0, h],
            [h, 0.0, h],
            [h, 0.0, h],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        for (got, want) in n.iter().zip(want) {
            assert!(close(*got, want), "{got:?} vs {want:?}");
        }

        // a wall flipped against the floor would cancel it out, so the edge stays hard
        let [floor, mut wall] = fold(true);
        wall.loops[0].reverse();
        wall.normal = [-1.0, 0.0, 0.0];
        wall.smooth = vec![false, false, true, false];
        let n = smooth_normals(&[&floor, &wall]).unwrap();
        assert!(close(n[0], [0.0, 0.0, 1.0]) && close(n[4], [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn smoothing_weights_faces_by_their_angle_at_the_corner() {
        // around the corner at the origin the floor has a right angle and the triangle beside it 45 degrees
        let mut floor = square_face(None);
        floor.smooth = vec![false, false, false, true];
        let mut wall = square_face(None);
        wall.loops = vec![vec![[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 10.0, 10.0]]];
        wall.normal = [1.0, 0.0, 0.0];
        wall.smooth = vec![true, false, false];
        let n = smooth_normals(&[&floor, &wall]).unwrap();
        let want = unit([1.0, 0.0, 2.0]).unwrap();
        assert!((0..3).all(|i| (n[0][i] - want[i]).abs() < 1e-12), "{:?}", n[0]);
    }

    #[test]
    fn written_normals_follow_the_smoothing() {
        let mut model = Model::default();
        model.root.faces.extend(fold(true));
        let glb = write(&model, MAX_TRIANGLES).unwrap().unwrap();
        let doc = gltf::Gltf::from_slice(&glb).unwrap();
        let bin = doc.blob.as_deref().unwrap();
        let prim = doc.meshes().next().unwrap().primitives().next().unwrap();
        let normals: Vec<[f32; 3]> = prim.reader(|_| Some(bin)).read_normals().unwrap().collect();
        // glTF's Y is SketchUp's Z
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((normals[0][0] - h).abs() < 1e-6 && (normals[0][1] - h).abs() < 1e-6);
        assert_eq!(normals[1], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn colorized_textures_take_their_tint() {
        let gray = tint([64, 64, 64], Some([128, 128, 128]));
        let want = srgb_to_linear(64) / srgb_to_linear(128);
        assert!(gray.iter().all(|c| (c - want).abs() < 1e-12));
        assert_eq!(tint([255, 0, 0], Some([128, 128, 128])), [1.0, 0.0, 0.0]);
        assert_eq!(tint([255, 255, 255], None), [1.0, 1.0, 1.0]);
        assert_eq!(tint([255, 0, 0], Some([0, 0, 0])), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_model_with_nothing_visible_has_no_glb() {
        let mut model = Model::default();
        let mut f = square_face(None);
        f.hidden = true;
        model.root.faces.push(f);
        assert!(write(&model, MAX_TRIANGLES).unwrap().is_none());
    }

    #[test]
    fn recursive_definitions_terminate() {
        let mut model = Model::default();
        let inst = Instance {
            definition: 1,
            name: String::new(),
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            material: None,
            hidden: false,
            layer: None,
        };
        model.definitions.insert(
            1,
            Definition {
                name: "Loop".to_owned(),
                faces: vec![square_face(None)],
                instances: vec![inst.clone()],
            },
        );
        model.root.instances.push(inst);
        assert!(write(&model, MAX_TRIANGLES).is_ok());
    }

    #[test]
    fn sniffs_images() {
        assert_eq!(image_mime(&[0xff, 0xd8, 0xff, 0xe0]), Some("image/jpeg"));
        assert_eq!(image_mime(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(image_mime(b"BM"), None);
    }
}
