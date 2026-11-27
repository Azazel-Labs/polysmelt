//! FBX (binary 7.x, and ASCII) as a glTF document: the model hierarchy with each model's transform (pivots, offsets,
//! pre- and post-rotation, rotation order; geometric transforms on a child node), meshes in their own space and shared
//! between instances, materials and their textures (embedded or found in the pack), skins from skin clusters, morph
//! targets from blend shapes, animation stacks sampled into translation, rotation, scale and weights channels,
//! cameras, lights, and user properties as node extras. The file's axis system and unit become a root node that
//! carries them to glTF's Y-up meters, and are recorded in `asset.extras.source` (see [`crate::source`]), with its
//! frame rate (`TimeMode`), the applications that made and last saved it, `SceneInfo`'s title, author, subject,
//! keywords and comment, and when it was made (`CreationTimeStamp`) and last saved. `coordAxis` (the file's right
//! axis), `creator` (the SDK that wrote it, when an application is named), `revision`, `originalApplication` and
//! `originalUnitScaleFactor` are kept as the file gives them.
//!
//! `Line` geometry is drawn as line segments, in its model's first material or else unlit in its own color, and so
//! is `NurbsCurve` geometry: each knot span in up to 32 segments, by how far it turns. NURBS surfaces and patches,
//! and curves with no order or points, aren't read; `asset.extras.source.skipped` counts them by kind.
//!
//! Colors are sRGB, as the FBX SDK's tools write them, except that Blender writes material colors linear; vertex
//! colors are sRGB from every tool, Blender included (its exporter's default).

mod animation;
mod curves;
mod deform;
mod geometry;
mod material;
mod settings;
#[cfg(all(test, feature = "fs"))]
mod tests;
mod walker;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{Map, Value, json};

use super::fbx_tree::{Attr as A, NodeHandle};
use super::xform::{IDENTITY, set_transform};
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::Images;
use crate::material::{Full, Tex};
use crate::{Context as _, ErrorKind, Result};
use curves::is_curve;
use material::TexField;

fn floats(v: &A) -> Option<Vec<f64>> {
    match v {
        A::ArrF64(a) => a.to_vec(),
        A::ArrF32(a) => a.map(f64::from),
        _ => None,
    }
}

fn ints(v: &A) -> Option<Vec<i64>> {
    match v {
        A::ArrI32(a) => a.map(i64::from),
        A::ArrI64(a) => a.to_vec(),
        _ => None,
    }
}

/// An object ID. FBX writes them as 64-bit integers, often hashes that use every bit, so they
/// are read as integers: through a double, IDs past 2^53 would round and collide.
fn id(v: &A) -> Option<i64> {
    match v {
        A::I64(i) => Some(*i),
        A::I32(i) => Some(i64::from(*i)),
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a whole number written as a float; `as` saturates"
        )]
        v => num(v).map(|f| f as i64),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "a property value as a double, like the FBX SDK reads it"
)]
fn num(v: &A) -> Option<f64> {
    match v {
        A::F64(f) => Some(*f),
        A::F32(f) => Some(f64::from(*f)),
        A::I32(i) => Some(f64::from(*i)),
        A::I64(i) => Some(*i as f64),
        A::I16(i) => Some(f64::from(*i)),
        A::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn string<'a>(v: &'a A<'_>) -> Option<&'a str> {
    if let A::String(s) = v { Some(s.as_ref()) } else { None }
}

fn child_attr<'a>(node: NodeHandle<'a>, name: &str) -> Option<&'a A<'a>> {
    node.first_child_by_name(name)?.attributes().first()
}

/// `Properties70` → name → values (after the name, type, label and flags strings).
fn props(node: NodeHandle<'_>) -> HashMap<String, Vec<A<'_>>> {
    let mut out = HashMap::new();
    if let Some(p70) = node.first_child_by_name("Properties70") {
        for p in p70.children_by_name("P") {
            let a = p.attributes();
            if let Some(name) = a.first().and_then(string) {
                out.insert(name.to_owned(), a.get(4..).unwrap_or_default().to_vec());
            }
        }
    }
    out
}

/// An object's properties: its own `Properties70`, over the defaults its class's `Definitions` template gives.
#[derive(Default)]
struct Props<'t> {
    own: HashMap<String, Vec<A<'t>>>,
    template: Option<&'t HashMap<String, Vec<A<'t>>>>,
}

impl<'t> Props<'t> {
    /// An object's own properties, without a template.
    fn of(node: NodeHandle<'t>) -> Self {
        Self {
            own: props(node),
            template: None,
        }
    }

    fn get(&self, name: &str) -> Option<&Vec<A<'t>>> {
        self.own.get(name).or_else(|| self.template?.get(name))
    }

    /// A number the object gives itself, ignoring its template.
    fn own_scalar(&self, name: &str) -> Option<f64> {
        self.own
            .get(name)
            .and_then(|v| v.first())
            .and_then(num)
            .filter(|v| v.is_finite())
    }

    /// A string property.
    fn text(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(|v| v.first()).and_then(string)
    }
}

/// A vector property; a component that isn't a finite number takes the default's.
fn vec3(p: &Props<'_>, name: &str, default: [f64; 3]) -> [f64; 3] {
    match p.get(name) {
        Some(v) if v.len() >= 3 => std::array::from_fn(|k| num(&v[k]).filter(|x| x.is_finite()).unwrap_or(default[k])),
        _ => default,
    }
}

/// A number property, when it's there and finite.
fn scalar(p: &Props<'_>, name: &str) -> Option<f64> {
    p.get(name)
        .and_then(|v| v.first())
        .and_then(num)
        .filter(|v| v.is_finite())
}

/// Whether a flag property is set, `default` when the property isn't there.
fn flag(p: &Props<'_>, name: &str, default: bool) -> bool {
    scalar(p, name).map_or(default, |v| v != 0.0)
}

/// An enum property, `default` when it isn't there.
fn choice(p: &Props<'_>, name: &str, default: i32) -> i32 {
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    scalar(p, name).map_or(default, |v| v as i32)
}

/// Reads an FBX file (binary 7.x, or ASCII) as a glTF document.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    // A file read is held as shared bytes, so that a big texture embedded in it can point into
    // it rather than be copied.
    let (borrowed, file) = match input.read()? {
        Cow::Borrowed(b) => (b, None),
        Cow::Owned(v) => (&[][..], Some(crate::Bytes::from(v))),
    };
    let data: &[u8] = file.as_deref().unwrap_or(borrowed);
    let tree = super::fbx_tree::Tree::parse(data)?;
    let version = tree.version();
    let rootnode = tree.root();
    let creator = |n: Option<NodeHandle<'_>>| {
        n.and_then(|n| n.first_child_by_name("Creator"))
            .and_then(|c| c.attributes().first().and_then(string).map(str::to_owned))
            .unwrap_or_default()
    };
    let creator = [
        creator(Some(rootnode)),
        creator(rootnode.first_child_by_name("FBXHeaderExtension")),
    ]
    .into_iter()
    .find(|c| !c.is_empty())
    .unwrap_or_default();
    let objects_node = rootnode
        .first_child_by_name("Objects")
        .context(ErrorKind::Empty, "FBX has no Objects")?;
    let mut objs = Objects {
        file: file.clone(),
        nodes: HashMap::new(),
        order: Vec::new(),
        parents: HashMap::new(),
        children: HashMap::new(),
        templates: HashMap::new(),
    };
    // Property templates: the defaults of every property an object of a class doesn't give itself.
    if let Some(defs) = rootnode.first_child_by_name("Definitions") {
        for class in defs.children_by_name("ObjectType") {
            let Some(name) = class.attributes().first().and_then(string) else {
                continue;
            };
            for t in class.children_by_name("PropertyTemplate") {
                let kind = t.attributes().first().and_then(string).unwrap_or("").to_owned();
                objs.templates
                    .entry(name.to_owned())
                    .or_default()
                    .push((kind, props(t)));
            }
        }
    }
    for o in objects_node.children() {
        if let Some(id) = o.attributes().first().and_then(id)
            && objs.nodes.insert(id, (o, o.name().to_owned())).is_none()
        {
            objs.order.push(id);
        }
    }
    if let Some(conns) = rootnode.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            let (Some(kind), Some(child), Some(parent)) =
                (a.first().and_then(string), a.get(1).and_then(id), a.get(2).and_then(id))
            else {
                continue;
            };
            let prop = if kind == "OP" {
                a.get(3).and_then(string).unwrap_or("").to_owned()
            } else {
                String::new()
            };
            if kind == "OO" {
                objs.parents.entry(child).or_default().push(parent);
            }
            objs.children.entry(parent).or_default().push((child, prop));
        }
    }

    let (basis, unit, info, mut details) = settings::read(rootnode, version, &creator);

    let mut r = Reader {
        objs: &objs,
        linear: creator.contains("Blender"),
        dir: input.dir(),
        assets,
        keep,
        limit: input.limit,
        b: Builder::new(keep.crease_angle()),
        nodes: Vec::new(),
        node_of: HashMap::new(),
        meshes: Vec::new(),
        mesh_of: HashMap::new(),
        materials: Vec::new(),
        images: Images::new(!keep.image_bytes, input.dir()),
        material_index: HashMap::new(),
        image_index: HashMap::new(),
        default_material: None,
        textured: Vec::new(),
        skins: Vec::new(),
        skin_of: HashMap::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        channel_users: HashMap::new(),
        morphs: HashMap::new(),
        channel_full: HashMap::new(),
        unit,
        skinned_roots: Vec::new(),
        skinned_meshes: std::collections::HashSet::new(),
        hidden: std::collections::HashSet::new(),
        color_uv: HashMap::new(),
        material_uvs: HashMap::new(),
        line_materials: HashMap::new(),
        curve_points: 0,
    };
    let models: Vec<i64> = objs.order.iter().copied().filter(|id| objs.is("Model", *id)).collect();
    // A level of detail group shows its first level, the finest; the others are kept, hidden.
    for &m in &models {
        if objs
            .children_of(m)
            .any(|a| objs.is_kind("NodeAttribute", a, "LodGroup"))
        {
            let levels = objs.children_of(m).filter(|c| objs.is("Model", *c));
            r.hidden.extend(levels.skip(1));
        }
    }
    for &m in &models {
        let node = r.model_node(m);
        r.nodes.push(node);
        r.node_of.insert(m, r.nodes.len() - 1);
    }
    // Parents, once every model has its node. A model whose parent is under it (a broken file) stays a root: each
    // node's tree is tracked as it's joined to its parent's, and a node is always a root when it's joined, so joining
    // makes a loop only when the parent's root is the node itself.
    let mut roots = Vec::new();
    let mut tree: Vec<usize> = (0..r.nodes.len()).collect();
    let root_of = |tree: &mut Vec<usize>, mut n: usize| {
        let mut top = n;
        while tree[top] != top {
            top = tree[top];
        }
        while tree[n] != top {
            n = std::mem::replace(&mut tree[n], top);
        }
        top
    };
    for &m in &models {
        let n = r.node_of[&m];
        match objs.model_parent(m).and_then(|p| r.node_of.get(&p)) {
            Some(&p) if root_of(&mut tree, p) != n => {
                r.add_child(p, n);
                tree[n] = root_of(&mut tree, p);
            }
            _ => roots.push(n),
        }
    }
    for &m in &models {
        r.attach(m)?;
    }
    // A geometry no model holds is drawn where it is. Geometry that isn't a mesh or a line is counted.
    let mut skipped: BTreeMap<&str, usize> = BTreeMap::new();
    for &g in &objs.order {
        let Some(geom) = objs.node(g).filter(|n| n.name() == "Geometry") else {
            continue;
        };
        let mesh = match kind_of(geom) {
            "NurbsCurve" if !is_curve(geom) => {
                *skipped.entry("NurbsCurve").or_default() += 1;
                continue;
            }
            "Mesh" | "Line" | "NurbsCurve" if objs.parents_of(g).any(|p| objs.is("Model", p)) => continue,
            "Mesh" => r.mesh_for(g, 0)?,
            "Line" | "NurbsCurve" => match r.line_mesh(g, 0) {
                Some(mesh) => mesh,
                None => continue,
            },
            "Shape" | "" => continue,
            kind => {
                *skipped.entry(kind).or_default() += 1;
                continue;
            }
        };
        r.nodes.push(json!({ "mesh": mesh }));
        roots.push(r.nodes.len() - 1);
    }
    if !skipped.is_empty() {
        details.insert("skipped".into(), json!(skipped));
    }
    let animations = if keep.animations { r.animations() } else { Vec::new() };
    let mut scene_nodes = roots;
    if basis != IDENTITY {
        let mut root = json!({ "name": "RootNode", "children": scene_nodes });
        set_transform(&mut root, &basis);
        r.nodes.push(root);
        scene_nodes = vec![r.nodes.len() - 1];
    }
    scene_nodes.append(&mut r.skinned_roots);
    ensure!(!r.nodes.is_empty(), Empty, "FBX has no mesh geometry");

    if r.keep.materials
        && r.keep.textures
        && !r.textured.is_empty()
        && !r.materials.iter().any(|m| m.base_color_tex.is_some())
        && super::plain_colors(r.textured.iter().map(|&m| r.materials[m].base_color))
        && let Some(i) = super::color_texture(input.path, assets, &mut r.images)
    {
        // A model that names no color texture takes one by name (see `guess_color_texture`).
        for &m in &r.textured {
            r.materials[m].base_color_tex = Some(Tex::new(i));
            r.materials[m].base_color[..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        }
    }

    let mut json = Map::new();
    info.write(&mut details);
    json.insert("asset".into(), json!({ "extras": { "source": details } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": scene_nodes }]));
    json.insert("nodes".into(), Value::Array(std::mem::take(&mut r.nodes)));
    if !r.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut r.meshes)));
    }
    if !r.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut r.skins)));
    }
    if !r.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut r.cameras)));
    }
    let mut used = Vec::new();
    if !r.lights.is_empty() {
        used.push("KHR_lights_punctual");
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    if !r.hidden.is_empty() {
        used.push("KHR_node_visibility");
    }
    if !used.is_empty() {
        json.insert("extensionsUsed".into(), json!(used));
    }
    let (materials, images) = (std::mem::take(&mut r.materials), std::mem::take(&mut r.images));
    let mut b = r.b;
    b.animations = animations;
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// Object id → (node, class), and how objects connect.
struct Objects<'a> {
    /// The file's bytes, when it was read from disk.
    file: Option<crate::Bytes>,
    nodes: HashMap<i64, (NodeHandle<'a>, String)>,
    /// Object ids in file order.
    order: Vec<i64>,
    /// child → parents (object-object connections), in file order.
    parents: HashMap<i64, Vec<i64>>,
    /// parent → children with the property they plug into ("" for object-object).
    children: HashMap<i64, Vec<(i64, String)>>,
    /// Class (`Model`, ...) → its property templates, by name (`FbxNode`, ...).
    templates: HashMap<String, Vec<Template<'a>>>,
}

/// A property template: its name (`FbxNode`, ...), and its properties' values by name.
type Template<'a> = (String, HashMap<String, Vec<A<'a>>>);

impl<'a> Objects<'a> {
    fn node(&self, id: i64) -> Option<NodeHandle<'a>> {
        self.nodes.get(&id).map(|(n, _)| *n)
    }

    /// Object `id`'s properties, over its class's template.
    fn props(&self, id: i64) -> Props<'_> {
        self.node(id).map(|n| self.props_of(n)).unwrap_or_default()
    }

    fn props_of(&self, node: NodeHandle<'a>) -> Props<'_> {
        Props {
            own: props(node),
            template: self.template(node),
        }
    }

    /// The template for an object of `node`'s class: the class's only one, or the one its kind or shading model
    /// names when there are more.
    fn template(&self, node: NodeHandle<'_>) -> Option<&HashMap<String, Vec<A<'a>>>> {
        let all = self.templates.get(node.name())?;
        let kind = kind_of(node);
        // A line's or curve's template is never a mesh's, nor a mesh's a line's or curve's.
        if let [(name, one)] = all.as_slice()
            && (node.name() != "Geometry"
                || match kind {
                    "Line" => name == "FbxLine",
                    "NurbsCurve" => name == "FbxNurbsCurve",
                    _ => name != "FbxLine" && name != "FbxNurbsCurve",
                })
        {
            return Some(one);
        }
        let named = |want: &str| all.iter().find(|(n, _)| n == want).map(|(_, p)| p);
        match node.name() {
            "Model" => named("FbxNode"),
            "Geometry" if kind == "Shape" => named("FbxShape"),
            "Geometry" if kind == "Line" => named("FbxLine"),
            "Geometry" if kind == "NurbsCurve" => named("FbxNurbsCurve"),
            "Geometry" => named("FbxMesh"),
            "NodeAttribute" => match kind {
                "LimbNode" => named("FbxSkeleton"),
                "LodGroup" => named("FbxLODGroup"),
                _ => all
                    .iter()
                    .find(|(n, _)| n.strip_prefix("Fbx") == Some(kind))
                    .map(|(_, p)| p),
            },
            "Material" => {
                let model = child_attr(node, "ShadingModel").and_then(string).unwrap_or("");
                if model.eq_ignore_ascii_case("lambert") {
                    named("FbxSurfaceLambert")
                } else {
                    named("FbxSurfacePhong")
                }
            }
            _ => None,
        }
    }

    /// Whether object `id` is a `class` node (`Model`, `Geometry`, `Deformer`, ...).
    fn is(&self, class: &str, id: i64) -> bool {
        self.node(id).is_some_and(|n| n.name() == class)
    }

    /// Whether object `id` is a `class` node of `kind` (`Geometry: id, "Name", "Mesh"`).
    fn is_kind(&self, class: &str, id: i64, kind: &str) -> bool {
        self.node(id).is_some_and(|n| n.name() == class && kind_of(n) == kind)
    }

    fn parents_of(&self, id: i64) -> impl Iterator<Item = i64> + '_ {
        self.parents.get(&id).into_iter().flatten().copied()
    }

    /// Object-object children of `id`.
    fn children_of(&self, id: i64) -> impl Iterator<Item = i64> + '_ {
        self.children
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(_, p)| p.is_empty())
            .map(|(c, _)| *c)
    }

    /// The child plugged into `id`'s property `prop`.
    fn child_at<'s>(&'s self, id: i64, prop: &'s str) -> impl Iterator<Item = i64> + 's {
        self.children
            .get(&id)
            .into_iter()
            .flatten()
            .filter(move |(_, p)| p == prop)
            .map(|(c, _)| *c)
    }

    fn model_parent(&self, id: i64) -> Option<i64> {
        self.parents_of(id).find(|p| self.is("Model", *p))
    }
}

/// An object's name, without its `\0\x01Class` suffix.
fn name_of(node: NodeHandle<'_>) -> Option<&str> {
    let name = node.attributes().get(1).and_then(string)?.split('\0').next()?;
    (!name.is_empty()).then_some(name)
}

/// Builds the document as the file's objects are visited.
struct Reader<'o, 'a, 'x> {
    objs: &'o Objects<'a>,
    /// Blender writes material colors linear; everyone else sRGB.
    linear: bool,
    dir: &'x Path,
    assets: &'x dyn Assets,
    keep: Keep,
    /// The most triangles one geometry may have.
    limit: usize,
    b: Builder,
    nodes: Vec<Value>,
    /// Model id → node.
    node_of: HashMap<i64, usize>,
    meshes: Vec<Value>,
    /// (geometry, its materials) → mesh.
    mesh_of: HashMap<(i64, Vec<usize>), usize>,
    materials: Vec<Full>,
    images: Images,
    material_index: HashMap<i64, usize>,
    image_index: HashMap<i64, Option<usize>>,
    /// The gray material primitives without one take.
    default_material: Option<usize>,
    /// Materials a primitive with UVs uses.
    textured: Vec<usize>,
    skins: Vec<Value>,
    /// (skin deformer, model) → skin.
    skin_of: HashMap<(i64, i64), usize>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    /// Blend shape channel → (node, target) pairs it drives.
    channel_users: HashMap<i64, Vec<(usize, usize)>>,
    /// Mesh → the blend shape channels its targets come from, in order, with each one's first target.
    morphs: HashMap<usize, Vec<(i64, usize)>>,
    /// Blend shape channel → the weights its targets are reached in full at.
    channel_full: HashMap<i64, Vec<f64>>,
    /// Meters per file unit.
    unit: f64,
    /// Nodes holding skinned meshes, which go at the scene's root, outside the axis and unit conversion.
    skinned_roots: Vec<usize>,
    /// Meshes carrying joints and weights.
    skinned_meshes: std::collections::HashSet<usize>,
    /// Models hidden, by their own visibility or as a level of detail not shown.
    hidden: std::collections::HashSet<i64>,
    /// Material → the UV set its color texture reads, by name.
    color_uv: HashMap<usize, String>,
    /// Material → its other textures that read a UV set by name, and the set; taken once the material is bound to
    /// a mesh's sets.
    material_uvs: HashMap<usize, Vec<(TexField, String)>>,
    /// A line's linear color, by its components' bits → the unlit material drawing it.
    line_materials: HashMap<[u32; 3], usize>,
    /// The points NURBS curves have drawn, out of `MAX_CURVE_POINTS`.
    curve_points: usize,
}

/// "Mesh" in `Geometry: 123, "Geometry::Name", "Mesh"`.
fn kind_of(node: NodeHandle<'_>) -> &str {
    node.attributes().get(2).and_then(string).unwrap_or("")
}

#[cfg(all(test, feature = "fs"))]
#[test]
#[ignore = "writes files; run by hand"]
fn write_fbx_fixture() {
    let Some(out) = std::env::var_os("POLYSMELT_FIXTURE_OUT") else {
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("floor.fbx"), tests::z_up_fbx()).unwrap();
}
