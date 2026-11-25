//! A document as binary FBX 7.4 (7.5 past 4 GiB): see [`Document::write_fbx`](crate::Document::write_fbx).
//!
//! The default scene's nodes become models (with the trees of any skin's joints), in node order, each with its
//! translation, rotation (XYZ Euler angles) and scale; joints are limb nodes with a skeleton attribute. Meshes become
//! geometry shared by every model drawing them: control points are the primitives' vertices, triangles (from lists,
//! strips and fans) are polygons, and normals, tangents and binormals, every UV set (`TEXCOORD_n`) and every color
//! set (`COLOR_n`) are layer elements indexed by polygon vertex, with a material per polygon. Lines become `Line`
//! geometry; points aren't written (FBX has no geometry for them). Draco, meshopt, sparse and quantized data are
//! decoded first.
//!
//! Materials are Phong materials (so any tool shows something like them) with the metal/rough model alongside
//! under `glTF|` properties, which this crate's reader takes back exactly: base color, metalness, roughness,
//! emission and its strength, alpha mode and cutoff, double-sidedness, unlit, IOR, transmission, clearcoat, sheen,
//! specular, volume, anisotropy, iridescence and dispersion factors. Textures are embedded in the file, each with its
//! wrap modes, `KHR_texture_transform` and UV set.
//!
//! Skins become skin deformers with a cluster per joint and a bind pose; morph targets become blend shapes, a channel
//! per target; animations become an animation stack and layer each (with a take), translation and scale curves keyed
//! as glTF keys them, rotation as Euler curves keyed densely enough to follow the quaternion path, and morph weights
//! as each channel's `DeformPercent`. Cameras and `KHR_lights_punctual` lights become camera and light attributes,
//! and node extras become user properties.
//!
//! The file is Y up, in meters (`UnitScaleFactor` 100), and the same document always writes the same bytes.

mod anim;
mod deform;
mod geometry;
mod material;
mod records;

use std::collections::HashMap;
use std::io::Write;

use serde_json::{Map, Value};

use super::xform::{IDENTITY, M4, decompose, from_column_major, mul, rotation, scale, translate};
use crate::document::Document;
use crate::{Result, quat};
use records::{Node, p, p_f64, p_int, p_str, p_time, p_vec, props70};

/// FBX time units per second.
pub(super) const KTIME: f64 = 46_186_158_000.0;

/// Writes `doc` as binary FBX to `w`.
pub(crate) fn write(doc: &Document, w: &mut dyn Write) -> Result<()> {
    let mut s = Writer::new(doc);
    s.scene()?;
    let top = s.finish();
    records::write_file(&top, w)
}

/// What a camera or light needs of its model: FBX cameras look down +X and lights shine down -Y, glTF's both down
/// -Z, so the model is turned by `turn` after the node's own rotation (and its scale permuted to match), and its
/// children are turned back.
#[derive(Clone, Copy)]
struct Turn {
    q: [f64; 4],
    /// Scale axis `i` of the model is the node's axis `perm[i]`.
    perm: [usize; 3],
}

/// A turn of 90° about Y, for a camera.
const CAMERA_TURN: Turn = Turn {
    q: [
        0.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
        std::f64::consts::FRAC_1_SQRT_2,
    ],
    perm: [2, 1, 0],
};

/// A turn of 90° about X, for a light.
const LIGHT_TURN: Turn = Turn {
    q: [
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        std::f64::consts::FRAC_1_SQRT_2,
    ],
    perm: [0, 2, 1],
};

/// A node written as a model.
struct Model {
    id: i64,
    name: String,
    parent: Option<usize>,
    /// The node's own translation, rotation and scale (from its matrix, if it has one).
    t: [f64; 3],
    r: [f64; 4],
    s: [f64; 3],
    /// The node's world transform at rest.
    world: M4,
    /// The model's turn for the camera or light it holds itself.
    turn: Option<Turn>,
    /// The inverse of the parent's turn, applied before the node's own transform.
    back: Option<[f64; 4]>,
    joint: bool,
}

impl Model {
    /// The model's translation, rotation and scale for the node's own, as FBX holds them.
    fn fbx_trs(&self, t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> ([f64; 3], [f64; 4], [f64; 3]) {
        let (mut t, mut r, mut s) = (t, r, s);
        if let Some(b) = self.back {
            t = quat::rotate(b, t);
            r = quat::mul(b, r);
        }
        if let Some(turn) = self.turn {
            r = quat::mul(r, turn.q);
            s = turn.perm.map(|i| s[i]);
        }
        (t, r, s)
    }
}

/// The writer: the objects and connections made so far, and how the document's parts map to them.
struct Writer<'d> {
    doc: &'d Document,
    json: &'d Map<String, Value>,
    next_id: i64,
    objects: Vec<Node>,
    links: Vec<Node>,
    /// Objects of each class, in the order classes first appear.
    counts: Vec<(&'static str, usize)>,
    /// By node index, the node's model.
    models: Vec<Option<Model>>,
    /// Material index → material object (`None` for one that failed).
    materials: HashMap<Option<usize>, i64>,
    /// Texture key → texture object.
    textures: HashMap<String, i64>,
    /// Image → video object.
    videos: HashMap<usize, Option<i64>>,
    /// Mesh → its geometry, when it's shared (neither skinned nor morphed).
    shared_geometry: HashMap<usize, Option<geometry::Geom>>,
    /// By node index, the blend shape channel objects of its morph targets.
    channels: HashMap<usize, Vec<i64>>,
    /// Takes, for the `Takes` section: (name, start, stop) in FBX time.
    takes: Vec<(String, i64, i64)>,
    /// Frames per second of the animations.
    fps: f64,
    /// The scene's time span, in FBX time.
    span: Option<(i64, i64)>,
}

/// An array of the JSON, or nothing.
fn list<'a>(json: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    json.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

/// A number field.
fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// A list of numbers, `n` long, or `default`.
fn numbers<const N: usize>(v: &Value, key: &str, default: [f64; N]) -> [f64; N] {
    v.get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() == N)
        .map_or(default, |a| {
            std::array::from_fn(|i| a[i].as_f64().filter(|x| x.is_finite()).unwrap_or(default[i]))
        })
}

/// An index field.
fn index(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok())
}

/// XYZ Euler angles in degrees (X applied first, as FBX's default order) for unit quaternion `q`.
fn euler(q: [f64; 4]) -> [f64; 3] {
    let m = quat::matrix(q);
    let at = |r: usize, c: usize| m[c * 3 + r];
    let sy = (-at(2, 0)).clamp(-1.0, 1.0);
    let y = sy.asin();
    let (x, z) = if at(2, 0).abs() < 1.0 - 1e-12 {
        (at(2, 1).atan2(at(2, 2)), at(1, 0).atan2(at(0, 0)))
    } else {
        ((-at(1, 2)).atan2(at(1, 1)), 0.0)
    };
    [x, y, z].map(f64::to_degrees)
}

/// A linear color channel sRGB-encoded.
fn to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// An object's name, `name\0\x01Class` as binary FBX writes it.
fn fbx_name(name: &str, class: &str) -> String {
    let name = name.replace('\0', "");
    format!("{name}\0\x01{class}")
}

/// A node's local transform as translation, rotation and scale.
fn node_trs(node: &Value) -> ([f64; 3], [f64; 4], [f64; 3]) {
    if let Some(m) = node.get("matrix").and_then(Value::as_array).filter(|m| m.len() == 16) {
        let flat: [f64; 16] = std::array::from_fn(|i| m[i].as_f64().unwrap_or(0.0));
        let d = decompose(&from_column_major(&flat));
        return (d.t, d.r, d.s);
    }
    let r = numbers(node, "rotation", [0.0, 0.0, 0.0, 1.0]);
    let r = quat::unit_above(r, 1e-12).unwrap_or(quat::IDENTITY);
    (
        numbers(node, "translation", [0.0; 3]),
        r,
        numbers(node, "scale", [1.0; 3]),
    )
}

/// The matrix of a translation, rotation and scale.
fn trs_matrix((t, r, s): ([f64; 3], [f64; 4], [f64; 3])) -> M4 {
    mul(&translate(t), &mul(&rotation(r), &scale(s)))
}

impl<'d> Writer<'d> {
    fn new(doc: &'d Document) -> Self {
        Self {
            doc,
            json: doc.json(),
            next_id: 1_000_000,
            objects: Vec::new(),
            links: Vec::new(),
            counts: Vec::new(),
            models: Vec::new(),
            materials: HashMap::new(),
            textures: HashMap::new(),
            videos: HashMap::new(),
            shared_geometry: HashMap::new(),
            channels: HashMap::new(),
            takes: Vec::new(),
            fps: 30.0,
            span: None,
        }
    }

    /// A new object id.
    fn id(&mut self) -> i64 {
        self.next_id += 1;
        self.next_id
    }

    /// Adds an object of `class` (as `Definitions` counts it).
    fn object(&mut self, class: &'static str, node: Node) {
        match self.counts.iter_mut().find(|(c, _)| *c == class) {
            Some((_, n)) => *n += 1,
            None => self.counts.push((class, 1)),
        }
        self.objects.push(node);
    }

    /// Connects object `child` to object `parent`.
    fn oo(&mut self, child: i64, parent: i64) {
        self.links.push(Node::new("C").str("OO").i64(child).i64(parent));
    }

    /// Connects object `child` to property `prop` of object `parent`.
    fn op(&mut self, child: i64, parent: i64, prop: &str) {
        self.links
            .push(Node::new("C").str("OP").i64(child).i64(parent).str(prop));
    }

    /// The nodes written: the default scene's (every root, without a scene) and the trees holding its skins'
    /// joints, by index.
    fn written_nodes(&self) -> Vec<bool> {
        let nodes = list(self.json, "nodes");
        let mut keep = vec![false; nodes.len()];
        let scenes = list(self.json, "scenes");
        let scene = index(&Value::Object(Map::new()), "scene");
        let scene = self
            .json
            .get("scene")
            .and_then(Value::as_u64)
            .and_then(|s| usize::try_from(s).ok())
            .or(scene)
            .or((!scenes.is_empty()).then_some(0));
        let mut roots: Vec<usize> = match scene.and_then(|s| scenes.get(s)) {
            Some(s) => s
                .get("nodes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_u64().and_then(|i| usize::try_from(i).ok()))
                .collect(),
            None => {
                let mut child = vec![false; nodes.len()];
                for n in nodes {
                    for c in children(n) {
                        if let Some(c) = child.get_mut(c) {
                            *c = true;
                        }
                    }
                }
                (0..nodes.len()).filter(|&i| !child[i]).collect()
            }
        };
        let mark = |roots: &[usize], keep: &mut Vec<bool>| {
            let mut stack: Vec<usize> = roots.to_vec();
            while let Some(n) = stack.pop() {
                if n >= keep.len() || keep[n] {
                    continue;
                }
                keep[n] = true;
                stack.extend(children(&nodes[n]));
            }
        };
        mark(&roots, &mut keep);
        // The joints of skins the scene uses, with the trees they're in.
        let parents = parents(nodes);
        let mut more = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            let Some(skin) = index(n, "skin").and_then(|s| list(self.json, "skins").get(s)) else {
                continue;
            };
            for j in skin.get("joints").and_then(Value::as_array).into_iter().flatten() {
                let Some(mut j) = j.as_u64().and_then(|j| usize::try_from(j).ok()) else {
                    continue;
                };
                if j >= nodes.len() || keep[j] {
                    continue;
                }
                let mut steps = 0;
                while let Some(&p) = parents.get(j).and_then(Option::as_ref) {
                    if keep[p] || steps > nodes.len() {
                        break;
                    }
                    j = p;
                    steps += 1;
                }
                more.push(j);
            }
        }
        roots.clear();
        mark(&more, &mut keep);
        keep
    }

    fn scene(&mut self) -> Result<()> {
        let nodes = list(self.json, "nodes");
        let keep = self.written_nodes();
        let parents = parents(nodes);
        let mut joint = vec![false; nodes.len()];
        for skin in list(self.json, "skins") {
            for j in skin.get("joints").and_then(Value::as_array).into_iter().flatten() {
                if let Some(j) = j.as_u64().and_then(|j| usize::try_from(j).ok())
                    && let Some(f) = joint.get_mut(j)
                {
                    *f = true;
                }
            }
        }
        // Models, in node order, so a reader numbering them in file order numbers them as the nodes.
        self.models = (0..nodes.len()).map(|_| None).collect();
        for (i, node) in nodes.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            let (t, r, s) = node_trs(node);
            let parent = parents[i].filter(|&p| keep[p]);
            let id = self.id();
            let name = node
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .map_or_else(|| format!("node_{i}"), str::to_owned);
            self.models[i] = Some(Model {
                id,
                name,
                parent,
                t,
                r,
                s,
                world: IDENTITY,
                turn: self.turn_of(i, joint[i]),
                back: None,
                joint: joint[i],
            });
        }
        // World transforms and turns, parents first.
        let order = topological(&self.models);
        for &i in &order {
            let Some(m) = &self.models[i] else { continue };
            let parent = m.parent;
            let local = trs_matrix((m.t, m.r, m.s));
            let (pw, back) = parent
                .and_then(|p| self.models[p].as_ref())
                .map_or((IDENTITY, None), |p| (p.world, p.turn.map(|t| quat::conjugate(t.q))));
            let m = self.models[i].as_mut().expect("written above");
            m.world = mul(&pw, &local);
            m.back = back;
        }
        for i in 0..nodes.len() {
            if self.models[i].is_some() {
                self.model(i);
            }
        }
        for i in 0..nodes.len() {
            if self.models[i].is_some() {
                self.content(i)?;
            }
        }
        self.animations()?;
        Ok(())
    }

    /// The turn node `i`'s model takes for what it holds: a camera's or else a light's, when that's all it holds.
    fn turn_of(&self, i: usize, joint: bool) -> Option<Turn> {
        let node = &list(self.json, "nodes")[i];
        if joint || node.get("mesh").is_some() || node.get("skin").is_some() {
            return None;
        }
        if index(node, "camera").is_some_and(|c| c < list(self.json, "cameras").len()) {
            Some(CAMERA_TURN)
        } else if self.light_of(node).is_some() {
            Some(LIGHT_TURN)
        } else {
            None
        }
    }

    /// The `KHR_lights_punctual` light a node holds.
    fn light_of(&self, node: &Value) -> Option<&'d Value> {
        let l = node.pointer("/extensions/KHR_lights_punctual/light")?.as_u64()?;
        self.json
            .get("extensions")?
            .get("KHR_lights_punctual")?
            .get("lights")?
            .as_array()?
            .get(usize::try_from(l).ok()?)
    }

    /// Writes node `i`'s model and connects it to its parent.
    fn model(&mut self, i: usize) {
        let node = &list(self.json, "nodes")[i];
        let m = self.models[i].as_ref().expect("a written node");
        let kind = if m.joint {
            "LimbNode"
        } else if node.get("mesh").is_some() {
            "Mesh"
        } else if m.turn.is_some_and(|t| t.perm == CAMERA_TURN.perm) {
            "Camera"
        } else if m.turn.is_some() {
            "Light"
        } else {
            "Null"
        };
        let (t, r, s) = m.fbx_trs(m.t, m.r, m.s);
        let hidden = node.pointer("/extensions/KHR_node_visibility/visible") == Some(&Value::Bool(false));
        let mut props = vec![
            p_int("RotationActive", "bool", 1),
            p_int("InheritType", "enum", 1),
            p_vec("ScalingMax", "Vector3D", "Vector", "", [0.0; 3]),
            p_int("DefaultAttributeIndex", "int", 0),
            p_vec("Lcl Translation", "Lcl Translation", "", "A", t),
            p_vec("Lcl Rotation", "Lcl Rotation", "", "A", euler(r)),
            p_vec("Lcl Scaling", "Lcl Scaling", "", "A", s),
        ];
        if hidden {
            props.push(p_f64("Visibility", "Visibility", "A", 0.0));
        }
        if let Some(extras) = node.get("extras").and_then(Value::as_object) {
            user_props(extras, &mut props);
        }
        let (id, name, parent) = (m.id, fbx_name(&m.name, "Model"), m.parent);
        let model = Node::new("Model")
            .i64(id)
            .str(&name)
            .str(kind)
            .child("Version", |n| n.i32(232))
            .with(props70(props))
            .child("Shading", |n| n.bool(true))
            .child("Culling", |n| n.str("CullingOff"));
        self.object("Model", model);
        let parent = parent.and_then(|p| self.models[p].as_ref()).map_or(0, |p| p.id);
        self.oo(id, parent);
        if self.models[i].as_ref().is_some_and(|m| m.joint) {
            let attr = self.id();
            let n = Node::new("NodeAttribute")
                .i64(attr)
                .str(&fbx_name(
                    &self.models[i].as_ref().expect("written").name,
                    "NodeAttribute",
                ))
                .str("LimbNode")
                .with(props70(vec![p_f64("Size", "double", "", 100.0)]))
                .child("TypeFlags", |n| n.str("Skeleton"));
            self.object("NodeAttribute", n);
            self.oo(attr, id);
        }
    }

    /// Writes what node `i` holds: its mesh (with skin and morph targets), camera and light.
    fn content(&mut self, i: usize) -> Result<()> {
        let node = &list(self.json, "nodes")[i];
        let model = self.models[i].as_ref().expect("a written node").id;
        if let Some(mesh) = index(node, "mesh").filter(|&m| m < list(self.json, "meshes").len()) {
            self.mesh(i, mesh, model)?;
        }
        self.attributes(i, model);
        Ok(())
    }

    /// A new model under `parent` with `local` as its transform, for what a node holds that its own model can't.
    fn helper(&mut self, parent: i64, name: &str, kind: &str, t: [f64; 3], r: [f64; 4]) -> i64 {
        let id = self.id();
        let props = vec![
            p_int("RotationActive", "bool", 1),
            p_int("InheritType", "enum", 1),
            p_int("DefaultAttributeIndex", "int", 0),
            p_vec("Lcl Translation", "Lcl Translation", "", "A", t),
            p_vec("Lcl Rotation", "Lcl Rotation", "", "A", euler(r)),
            p_vec("Lcl Scaling", "Lcl Scaling", "", "A", [1.0; 3]),
        ];
        let model = Node::new("Model")
            .i64(id)
            .str(&fbx_name(name, "Model"))
            .str(kind)
            .child("Version", |n| n.i32(232))
            .with(props70(props))
            .child("Shading", |n| n.bool(true))
            .child("Culling", |n| n.str("CullingOff"));
        self.object("Model", model);
        self.oo(id, parent);
        id
    }

    /// Writes mesh `mesh` as node `i` draws it, on model `model`.
    fn mesh(&mut self, i: usize, mesh: usize, model: i64) -> Result<()> {
        let node = &list(self.json, "nodes")[i];
        let mesh_json = &list(self.json, "meshes")[mesh];
        let skinned = index(node, "skin").is_some_and(|s| s < list(self.json, "skins").len());
        let morphed = mesh_json
            .get("primitives")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|p| {
                p.get("targets")
                    .and_then(Value::as_array)
                    .is_some_and(|t| !t.is_empty())
            });
        // A plain mesh's geometry is shared by every model drawing it; a skinned or morphed one's is the node's own,
        // since its deformers are.
        let geom = if skinned || morphed {
            self.geometry(mesh)?
        } else if let Some(g) = self.shared_geometry.get(&mesh) {
            g.clone()
        } else {
            let g = self.geometry(mesh)?;
            self.shared_geometry.insert(mesh, g.clone());
            g
        };
        let Some(g) = geom else {
            return Ok(());
        };
        if let Some(gid) = g.id {
            self.oo(gid, model);
            for slot in &g.slots {
                let m = self.material(*slot)?;
                self.oo(m, model);
            }
            if skinned {
                self.skin(i, &g)?;
            }
            if morphed {
                self.morphs(i, mesh, &g)?;
            }
        }
        if let Some((line, mat)) = g.lines {
            // A model's lines take its first material: beside triangles, they go on a model of their own.
            let host = if g.id.is_some() {
                let name = format!("{}_lines", self.models[i].as_ref().expect("written").name);
                self.helper(model, &name, "Mesh", [0.0; 3], quat::IDENTITY)
            } else {
                model
            };
            self.oo(line, host);
            let m = self.material(mat)?;
            self.oo(m, host);
        }
        Ok(())
    }

    /// Writes node `i`'s camera and light, on model `model`: on the model itself when it's turned for the one it
    /// holds, else on a child turned for it.
    fn attributes(&mut self, i: usize, model: i64) {
        let node = &list(self.json, "nodes")[i];
        let m = self.models[i].as_ref().expect("a written node");
        let (name, turn) = (m.name.clone(), m.turn.map(|t| t.perm));
        if let Some(cam) = index(node, "camera").and_then(|c| list(self.json, "cameras").get(c)) {
            let host = if turn == Some(CAMERA_TURN.perm) {
                model
            } else {
                self.helper(model, &format!("{name}_camera"), "Camera", [0.0; 3], CAMERA_TURN.q)
            };
            let attr = self.id();
            self.object("NodeAttribute", camera(attr, &name, cam));
            self.oo(attr, host);
        }
        if let Some(light) = self.light_of(node) {
            let host = if turn == Some(LIGHT_TURN.perm) {
                model
            } else {
                self.helper(model, &format!("{name}_light"), "Light", [0.0; 3], LIGHT_TURN.q)
            };
            let attr = self.id();
            self.object("NodeAttribute", light_attribute(attr, &name, light));
            self.oo(attr, host);
        }
    }

    /// The scene's top-level records, once everything is written.
    fn finish(mut self) -> Vec<Node> {
        let mut top = vec![
            self.header(7400),
            Node::new("FileId").raw(&[
                0x28, 0xb3, 0x2a, 0xeb, 0xb6, 0x24, 0xcc, 0xc2, 0xbf, 0xc8, 0xb0, 0x2a, 0xa9, 0x2b, 0xfc, 0xf1,
            ]),
            Node::new("CreationTime").str("1970-01-01 10:00:00:000"),
            Node::new("Creator").str(&creator()),
            self.global_settings(),
            self.documents(),
            Node::new("References"),
            self.definitions(),
            Node::new("Objects"),
            Node::new("Connections"),
            self.takes_node(),
        ];
        let objects = std::mem::take(&mut self.objects);
        let links = std::mem::take(&mut self.links);
        for o in objects {
            top[8].push(o);
        }
        for l in links {
            top[9].push(l);
        }
        if records::wide(&top) {
            top[0] = self.header(7500);
        }
        top
    }

    fn header(&self, version: i32) -> Node {
        let source = self.json.get("asset").and_then(|a| a.get("extras"));
        let meta = |key: &str| {
            source
                .and_then(|s| s.get(key))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned()
        };
        let creator = creator();
        let app = env!("CARGO_PKG_VERSION");
        let date = "01/01/1970 10:00:00.000";
        Node::new("FBXHeaderExtension")
            .child("FBXHeaderVersion", |n| n.i32(1003))
            .child("FBXVersion", |n| n.i32(version))
            .child("EncryptionType", |n| n.i32(0))
            .with(
                Node::new("CreationTimeStamp")
                    .child("Version", |n| n.i32(1000))
                    .child("Year", |n| n.i32(1970))
                    .child("Month", |n| n.i32(1))
                    .child("Day", |n| n.i32(1))
                    .child("Hour", |n| n.i32(10))
                    .child("Minute", |n| n.i32(0))
                    .child("Second", |n| n.i32(0))
                    .child("Millisecond", |n| n.i32(0)),
            )
            .child("Creator", |n| n.str(&creator))
            .with(
                Node::new("SceneInfo")
                    .str("GlobalInfo\0\x01SceneInfo")
                    .str("UserData")
                    .child("Type", |n| n.str("UserData"))
                    .child("Version", |n| n.i32(100))
                    .with(
                        Node::new("MetaData")
                            .child("Version", |n| n.i32(100))
                            .child("Title", |n| n.str(&meta("title")))
                            .child("Subject", |n| n.str(&meta("subject")))
                            .child("Author", |n| n.str(&meta("author")))
                            .child("Keywords", |n| n.str(&meta("keywords")))
                            .child("Revision", |n| n.str(&meta("revision")))
                            .child("Comment", |n| n.str(&meta("comment"))),
                    )
                    .with(props70(vec![
                        p("DocumentUrl", "KString", "Url", "").str("scene.fbx"),
                        p("SrcDocumentUrl", "KString", "Url", "").str("scene.fbx"),
                        p("Original", "Compound", "", ""),
                        p_str("Original|ApplicationVendor", ""),
                        p_str("Original|ApplicationName", "polysmelt"),
                        p_str("Original|ApplicationVersion", app),
                        p("Original|DateTime_GMT", "DateTime", "", "").str(date),
                        p_str("Original|FileName", "scene.fbx"),
                        p("LastSaved", "Compound", "", ""),
                        p_str("LastSaved|ApplicationVendor", ""),
                        p_str("LastSaved|ApplicationName", "polysmelt"),
                        p_str("LastSaved|ApplicationVersion", app),
                        p("LastSaved|DateTime_GMT", "DateTime", "", "").str(date),
                    ])),
            )
    }

    fn global_settings(&self) -> Node {
        let (mode, custom) = anim::time_mode(self.fps);
        let (start, stop) = self.span.unwrap_or((0, (KTIME * 10.0 / self.fps) as i64));
        Node::new("GlobalSettings")
            .child("Version", |n| n.i32(1000))
            .with(props70(vec![
                p_int("UpAxis", "int", 1),
                p_int("UpAxisSign", "int", 1),
                p_int("FrontAxis", "int", 2),
                p_int("FrontAxisSign", "int", 1),
                p_int("CoordAxis", "int", 0),
                p_int("CoordAxisSign", "int", 1),
                p_int("OriginalUpAxis", "int", 1),
                p_int("OriginalUpAxisSign", "int", 1),
                p_f64("UnitScaleFactor", "double", "", 100.0),
                p_f64("OriginalUnitScaleFactor", "double", "", 100.0),
                p_vec("AmbientColor", "ColorRGB", "Color", "", [0.0; 3]),
                p_str("DefaultCamera", "Producer Perspective"),
                p_int("TimeMode", "enum", mode),
                p_int("TimeProtocol", "enum", 2),
                p_int("SnapOnFrameMode", "enum", 0),
                p_time("TimeSpanStart", start),
                p_time("TimeSpanStop", stop),
                p_f64("CustomFrameRate", "double", "", custom),
                p("TimeMarker", "Compound", "", ""),
                p_int("CurrentTimeMarker", "int", -1),
            ]))
    }

    fn documents(&self) -> Node {
        let active = self.takes.first().map_or("", |t| t.0.as_str());
        Node::new("Documents").child("Count", |n| n.i32(1)).with(
            Node::new("Document")
                .i64(1_000_000)
                .str("")
                .str("Scene")
                .with(props70(vec![
                    p("SourceObject", "object", "", ""),
                    p_str("ActiveAnimStackName", active),
                ]))
                .child("RootNode", |n| n.i64(0)),
        )
    }

    fn definitions(&self) -> Node {
        let total: usize = self.counts.iter().map(|c| c.1).sum::<usize>() + 1;
        let mut defs = Node::new("Definitions")
            .child("Version", |n| n.i32(100))
            .child("Count", |n| n.i32(i32::try_from(total).unwrap_or(i32::MAX)));
        defs.push(
            Node::new("ObjectType")
                .str("GlobalSettings")
                .child("Count", |n| n.i32(1)),
        );
        for &(class, count) in &self.counts {
            let mut t = Node::new("ObjectType")
                .str(class)
                .child("Count", |n| n.i32(i32::try_from(count).unwrap_or(i32::MAX)));
            if let Some((name, props)) = template(class) {
                t.push(Node::new("PropertyTemplate").str(name).with(props70(props)));
            }
            defs.push(t);
        }
        defs
    }

    fn takes_node(&self) -> Node {
        let mut n = Node::new("Takes").child("Current", |n| n.str(""));
        for (name, start, stop) in &self.takes {
            n.push(
                Node::new("Take")
                    .str(name)
                    .child("FileName", |n| n.str(&format!("{}.tak", name.replace(' ', "_"))))
                    .child("LocalTime", |n| n.i64(*start).i64(*stop))
                    .child("ReferenceTime", |n| n.i64(*start).i64(*stop)),
            );
        }
        n
    }
}

/// A glTF camera as a camera attribute. The field of view is given as an angle and, for tools that work from a lens,
/// as the focal length a 24 mm high film would need for it.
fn camera(id: i64, name: &str, cam: &Value) -> Node {
    const FILM_HEIGHT: f64 = 24.0 / 25.4;
    let ortho = cam.get("type").and_then(Value::as_str) == Some("orthographic");
    let o = cam.get("orthographic").unwrap_or(&Value::Null);
    let p = cam.get("perspective").unwrap_or(&Value::Null);
    let (aspect, near, far) = if ortho {
        let (x, y) = (number(o, "xmag").unwrap_or(1.0), number(o, "ymag").unwrap_or(1.0));
        let aspect = if x > 0.0 && y > 0.0 { x / y } else { 1.0 };
        (
            aspect,
            number(o, "znear").unwrap_or(0.01),
            number(o, "zfar").unwrap_or(1000.0),
        )
    } else {
        let aspect = number(p, "aspectRatio").filter(|a| *a > 0.0).unwrap_or(16.0 / 9.0);
        // Without a far plane the camera sees forever: as far as can be held in practice.
        let near = number(p, "znear").filter(|n| *n > 0.0).unwrap_or(0.01);
        (
            aspect,
            near,
            number(p, "zfar").filter(|f| *f > near).unwrap_or(near * 1e7),
        )
    };
    let yfov = number(p, "yfov")
        .filter(|f| *f > 0.0 && *f < std::f64::consts::PI)
        .unwrap_or(0.8);
    let focal = FILM_HEIGHT * 25.4 / 2.0 / (yfov / 2.0).tan();
    let mut props = vec![
        p_int("CameraProjectionType", "enum", i32::from(ortho)),
        p_int("ApertureMode", "enum", 2),
        p_f64("FieldOfView", "FieldOfView", "A", yfov.to_degrees()),
        p_f64("FocalLength", "Number", "A", focal),
        p_f64("FilmWidth", "Number", "", FILM_HEIGHT * aspect),
        p_f64("FilmHeight", "Number", "", FILM_HEIGHT),
        p_f64("FilmAspectRatio", "double", "", aspect),
        p_int("AspectRatioMode", "enum", 0),
        p_f64("AspectWidth", "double", "", aspect),
        p_f64("AspectHeight", "double", "", 1.0),
        p_f64("NearPlane", "double", "", near),
        p_f64("FarPlane", "double", "", far),
    ];
    if ortho {
        props.push(p_f64("OrthoZoom", "double", "", number(o, "ymag").unwrap_or(1.0)));
    }
    if let Some(extras) = cam.get("extras").and_then(Value::as_object) {
        user_props(extras, &mut props);
    }
    Node::new("NodeAttribute")
        .i64(id)
        .str(&fbx_name(name, "NodeAttribute"))
        .str("Camera")
        .with(props70(props))
        .child("TypeFlags", |n| n.str("Camera"))
        .child("GeometryVersion", |n| n.i32(124))
        .child("Position", |n| n.f64(0.0).f64(0.0).f64(0.0))
        .child("Up", |n| n.f64(0.0).f64(1.0).f64(0.0))
        .child("LookAt", |n| n.f64(1.0).f64(0.0).f64(0.0))
        .child("ShowInfoOnMoving", |n| n.i32(1))
        .child("ShowAudio", |n| n.i32(0))
        .child("AudioColor", |n| n.f64(0.0).f64(1.0).f64(0.0))
        .child("CameraOrthoZoom", |n| n.f64(1.0))
}

/// A `KHR_lights_punctual` light as a light attribute: intensity in percent, as FBX keeps it, falling off with the
/// square of distance. A range has no FBX equivalent and is kept as a user property.
fn light_attribute(id: i64, name: &str, light: &Value) -> Node {
    let kind = match light.get("type").and_then(Value::as_str) {
        Some("directional") => 1,
        Some("spot") => 2,
        _ => 0,
    };
    let color = numbers(light, "color", [1.0; 3]).map(to_srgb);
    let mut props = vec![
        p_int("LightType", "enum", kind),
        p_int("CastLight", "bool", 1),
        p_vec("Color", "Color", "", "A", color),
        p_f64(
            "Intensity",
            "Number",
            "A",
            number(light, "intensity").unwrap_or(1.0) * 100.0,
        ),
        p_int("DecayType", "enum", 2),
        p_int("CastShadows", "bool", 1),
    ];
    if kind == 2 {
        let spot = light.get("spot").unwrap_or(&Value::Null);
        let outer = number(spot, "outerConeAngle").unwrap_or(std::f64::consts::FRAC_PI_4);
        let inner = number(spot, "innerConeAngle").unwrap_or(0.0);
        props.push(p_f64("InnerAngle", "Number", "A", (inner * 2.0).to_degrees()));
        props.push(p_f64("OuterAngle", "Number", "A", (outer * 2.0).to_degrees()));
    }
    let mut user = Map::new();
    if let Some(range) = light.get("range") {
        user.insert("range".to_owned(), range.clone());
    }
    if let Some(extras) = light.get("extras").and_then(Value::as_object) {
        user.extend(extras.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    user_props(&user, &mut props);
    Node::new("NodeAttribute")
        .i64(id)
        .str(&fbx_name(name, "NodeAttribute"))
        .str("Light")
        .with(props70(props))
        .child("TypeFlags", |n| n.str("Light"))
        .child("GeometryVersion", |n| n.i32(124))
}

/// Who wrote the file.
fn creator() -> String {
    concat!("polysmelt ", env!("CARGO_PKG_VERSION")).to_owned()
}

/// A node's children.
fn children(node: &Value) -> impl Iterator<Item = usize> + '_ {
    node.get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_u64().and_then(|i| usize::try_from(i).ok()))
}

/// Each node's parent (its first, in a broken document listing a node twice).
fn parents(nodes: &[Value]) -> Vec<Option<usize>> {
    let mut out = vec![None; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        for c in children(n) {
            if c < out.len() && out[c].is_none() && c != i {
                out[c] = Some(i);
            }
        }
    }
    out
}

/// The written nodes, each after its parent; a node in a cycle (a broken document) comes after the rest.
fn topological(models: &[Option<Model>]) -> Vec<usize> {
    let mut depth = vec![usize::MAX; models.len()];
    for i in 0..models.len() {
        let mut chain = Vec::new();
        let mut at = Some(i);
        while let Some(n) = at {
            if models[n].is_none() || depth[n] != usize::MAX || chain.len() > models.len() {
                break;
            }
            chain.push(n);
            at = models[n].as_ref().and_then(|m| m.parent);
        }
        let mut d = at.map_or(0, |n| if depth[n] == usize::MAX { 0 } else { depth[n] + 1 });
        for &n in chain.iter().rev() {
            if depth[n] == usize::MAX {
                depth[n] = d;
                d += 1;
            }
        }
    }
    let mut order: Vec<usize> = (0..models.len()).filter(|&i| models[i].is_some()).collect();
    order.sort_by_key(|&i| (depth[i], i));
    order
}

/// Extras as user properties: numbers, booleans (as integers, as FBX holds them), strings, lists of numbers, and
/// anything else as its JSON text.
fn user_props(extras: &Map<String, Value>, out: &mut Vec<Node>) {
    for (k, v) in extras {
        let entry = match v {
            Value::Bool(b) => p(k, "Bool", "", "A+U").i32(i32::from(*b)),
            Value::Number(n) => match n.as_i64().and_then(|i| i32::try_from(i).ok()) {
                Some(i) => p(k, "Integer", "", "A+U").i32(i),
                None => p(k, "Number", "", "A+U").f64(n.as_f64().unwrap_or(0.0)),
            },
            Value::String(s) => p(k, "KString", "", "U").str(s),
            Value::Array(a) if !a.is_empty() && a.len() <= 16 && a.iter().all(Value::is_number) => {
                let mut e = p(k, if a.len() == 3 { "Vector" } else { "Number" }, "", "A+U");
                for x in a {
                    e = e.f64(x.as_f64().unwrap_or(0.0));
                }
                e
            }
            Value::Null => continue,
            other => p(k, "KString", "", "U").str(&other.to_string()),
        };
        out.push(entry);
    }
}

/// The property template of a class: the FBX SDK's defaults for the properties its objects leave out.
fn template(class: &str) -> Option<(&'static str, Vec<Node>)> {
    Some(match class {
        "Model" => (
            "FbxNode",
            vec![
                p_int("QuaternionInterpolate", "enum", 0),
                p_vec("RotationOffset", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("RotationPivot", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("ScalingOffset", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("ScalingPivot", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("PreRotation", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("PostRotation", "Vector3D", "Vector", "", [0.0; 3]),
                p_int("RotationOrder", "enum", 0),
                p_int("RotationActive", "bool", 0),
                p_int("InheritType", "enum", 0),
                p_vec("GeometricTranslation", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("GeometricRotation", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("GeometricScaling", "Vector3D", "Vector", "", [1.0; 3]),
                p_vec("Lcl Translation", "Lcl Translation", "", "A", [0.0; 3]),
                p_vec("Lcl Rotation", "Lcl Rotation", "", "A", [0.0; 3]),
                p_vec("Lcl Scaling", "Lcl Scaling", "", "A", [1.0; 3]),
                p_f64("Visibility", "Visibility", "A", 1.0),
                p_int("Visibility Inheritance", "Visibility Inheritance", 1),
            ],
        ),
        "Geometry" => (
            "FbxMesh",
            vec![
                p_vec("Color", "ColorRGB", "Color", "", [0.8; 3]),
                p_vec("BBoxMin", "Vector3D", "Vector", "", [0.0; 3]),
                p_vec("BBoxMax", "Vector3D", "Vector", "", [0.0; 3]),
                p_int("Primary Visibility", "bool", 1),
                p_int("Casts Shadows", "bool", 1),
                p_int("Receive Shadows", "bool", 1),
            ],
        ),
        "Material" => (
            "FbxSurfacePhong",
            vec![
                p_str("ShadingModel", "Phong"),
                p_int("MultiLayer", "bool", 0),
                p_vec("EmissiveColor", "Color", "", "A", [0.0; 3]),
                p_f64("EmissiveFactor", "Number", "A", 1.0),
                p_vec("AmbientColor", "Color", "", "A", [0.2; 3]),
                p_f64("AmbientFactor", "Number", "A", 1.0),
                p_vec("DiffuseColor", "Color", "", "A", [0.8; 3]),
                p_f64("DiffuseFactor", "Number", "A", 1.0),
                p_vec("TransparentColor", "Color", "", "A", [0.0; 3]),
                p_f64("TransparencyFactor", "Number", "A", 0.0),
                p_vec("SpecularColor", "Color", "", "A", [0.2; 3]),
                p_f64("SpecularFactor", "Number", "A", 1.0),
                p_f64("Shininess", "Number", "A", 20.0),
                p_f64("ShininessExponent", "Number", "A", 20.0),
                p_vec("ReflectionColor", "Color", "", "A", [0.0; 3]),
                p_f64("ReflectionFactor", "Number", "A", 1.0),
            ],
        ),
        "Texture" => (
            "FbxFileTexture",
            vec![
                p_int("TextureTypeUse", "enum", 0),
                p_f64("Texture alpha", "Number", "A", 1.0),
                p_int("CurrentMappingType", "enum", 0),
                p_int("WrapModeU", "enum", 0),
                p_int("WrapModeV", "enum", 0),
                p_int("UVSwap", "bool", 0),
                p_int("PremultiplyAlpha", "bool", 1),
                p_vec("Translation", "Vector", "", "A", [0.0; 3]),
                p_vec("Rotation", "Vector", "", "A", [0.0; 3]),
                p_vec("Scaling", "Vector", "", "A", [1.0; 3]),
                p_int("UseMaterial", "bool", 0),
                p_int("UseMipMap", "bool", 0),
                p_int("CurrentTextureBlendMode", "enum", 1),
                p_str("UVSet", "default"),
            ],
        ),
        "Video" => ("FbxVideo", vec![p_str("Path", ""), p_int("ImageSequence", "bool", 0)]),
        "AnimationStack" => (
            "FbxAnimStack",
            vec![
                p_str("Description", ""),
                p_time("LocalStart", 0),
                p_time("LocalStop", 0),
                p_time("ReferenceStart", 0),
                p_time("ReferenceStop", 0),
            ],
        ),
        "AnimationLayer" => (
            "FbxAnimLayer",
            vec![
                p_f64("Weight", "Number", "A", 100.0),
                p_int("Mute", "bool", 0),
                p_int("Solo", "bool", 0),
                p_int("Lock", "bool", 0),
                p_vec("Color", "ColorRGB", "Color", "", [0.8; 3]),
                p_int("BlendMode", "enum", 0),
                p_int("RotationAccumulationMode", "enum", 0),
                p_int("ScaleAccumulationMode", "enum", 0),
                p("BlendModeBypass", "ULongLong", "", "").i64(0),
            ],
        ),
        "AnimationCurveNode" => ("FbxAnimCurveNode", vec![p("d", "Compound", "", "")]),
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
