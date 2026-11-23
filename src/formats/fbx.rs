//! FBX (binary 7.x, and ASCII) as a glTF document: the model hierarchy with each model's transform (pivots, offsets,
//! pre- and post-rotation, rotation order; geometric transforms on a child node), meshes in their own space and shared
//! between instances, materials and their textures (embedded or found in the pack), skins from skin clusters, morph
//! targets from blend shapes, animation stacks sampled into translation, rotation, scale and weights channels,
//! cameras, lights, and user properties as node extras. The file's axis system and unit become a root node that
//! carries them to glTF's Y-up meters, and are recorded in `asset.extras.source`.
//!
//! Colors are sRGB, as the FBX SDK's tools write them, except that Blender writes material colors linear; vertex
//! colors are sRGB from every tool, Blender included (its exporter's default).

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use crate::{Context as _, Result};
use fbxcel::low::v7400::AttributeValue as A;
use fbxcel::tree::v7400::NodeHandle;
use serde_json::{Map, Value, json};

use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::{Images, Location};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Sheen, Tex, UvTransform};
use crate::mesh::{GRAY, Material, Prim, smooth_normals, srgb_to_linear, triangulate};
use crate::num::f32_of;

use super::xform::{IDENTITY, M4, Trs, column_major, decompose, det3, inverse, mul, scale, set_transform, translate};

const MAX_TRIANGLES: usize = 4_000_000;

/// Euler rotation in degrees, FBX rotation order (0 = XYZ: X applied first).
fn rotate(deg: [f64; 3], order: i32) -> M4 {
    let [x, y, z] = deg.map(f64::to_radians);
    let rx = [
        [1., 0., 0., 0.],
        [0., x.cos(), -x.sin(), 0.],
        [0., x.sin(), x.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let ry = [
        [y.cos(), 0., y.sin(), 0.],
        [0., 1., 0., 0.],
        [-y.sin(), 0., y.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let rz = [
        [z.cos(), -z.sin(), 0., 0.],
        [z.sin(), z.cos(), 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    match order {
        1 => mul(&ry, &mul(&rz, &rx)), // XZY
        2 => mul(&rx, &mul(&rz, &ry)), // YZX
        3 => mul(&rz, &mul(&rx, &ry)), // YXZ
        4 => mul(&ry, &mul(&rx, &rz)), // ZXY
        5 => mul(&rx, &mul(&ry, &rz)), // ZYX
        _ => mul(&rz, &mul(&ry, &rx)), // XYZ
    }
}

fn floats(v: &A) -> Option<Vec<f64>> {
    match v {
        A::ArrF64(a) => Some(a.clone()),
        A::ArrF32(a) => Some(a.iter().map(|&f| f64::from(f)).collect()),
        _ => None,
    }
}

fn ints(v: &A) -> Option<Vec<i64>> {
    match v {
        A::ArrI32(a) => Some(a.iter().map(|&i| i64::from(i)).collect()),
        A::ArrI64(a) => Some(a.clone()),
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

fn string(v: &A) -> Option<&str> {
    if let A::String(s) = v { Some(s.as_str()) } else { None }
}

fn child_attr<'a>(node: NodeHandle<'a>, name: &str) -> Option<&'a A> {
    node.first_child_by_name(name)?.attributes().first()
}

/// `Properties70` → name → values (after the name, type, label and flags strings).
fn props(node: NodeHandle<'_>) -> HashMap<String, Vec<A>> {
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

fn vec3(p: &HashMap<String, Vec<A>>, name: &str, default: [f64; 3]) -> [f64; 3] {
    match p.get(name) {
        Some(v) if v.len() >= 3 => [num(&v[0]), num(&v[1]), num(&v[2])].map(|x| x.unwrap_or(0.0)),
        _ => default,
    }
}

fn scalar(p: &HashMap<String, Vec<A>>, name: &str) -> Option<f64> {
    p.get(name).and_then(|v| v.first()).and_then(num)
}

/// A layer element's values and how they map onto polygon vertices.
struct Layer {
    values: Vec<f64>,
    index: Option<Vec<i64>>,
    mapping: String,
    width: usize,
}

impl Layer {
    fn read(geom: NodeHandle<'_>, element: &str, values: &str, index: &str, width: usize) -> Option<Self> {
        let el = geom
            .children_by_name(element)
            .find(|e| e.attributes().first().and_then(num).unwrap_or(0.0) == 0.0)?;
        Some(Self {
            values: floats(child_attr(el, values)?)?,
            index: child_attr(el, index).and_then(ints),
            mapping: child_attr(el, "MappingInformationType")
                .and_then(string)
                .unwrap_or("ByPolygonVertex")
                .to_owned(),
            width,
        })
    }

    /// The value for polygon-vertex `pv` (control point `cp`, polygon `poly`).
    fn get(&self, pv: usize, cp: usize, poly: usize) -> Option<(usize, &[f64])> {
        let i = match self.mapping.as_str() {
            "ByVertice" | "ByVertex" | "ByControlPoint" => cp,
            "ByPolygon" => poly,
            "AllSame" => 0,
            _ => pv,
        };
        let i = match &self.index {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        Some((i, self.values.get(i * self.width..i * self.width + self.width)?))
    }
}

/// FBX time units per second.
const KTIME: f64 = 46_186_158_000.0;

/// The most degrees a rotation curve may turn between two samples: FBX interpolates Euler
/// angles, glTF quaternions, so long turns are cut into steps short enough to agree.
const MAX_STEP_DEGREES: f64 = 10.0;

/// Reads an FBX file (binary 7.x, or ASCII) as a glTF document.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let mut data = input.read()?;
    if super::fbx_ascii::is_ascii(&data) {
        data = super::fbx_ascii::to_binary(&data)
            .context("couldn't read this ASCII FBX file")?
            .into();
    } else if data.starts_with(b"Kaydara FBX Binary")
        && let Some(v) = data.get(23..27)
    {
        let v = u32::from_le_bytes(v.try_into().unwrap_or_default());
        ensure!(
            v >= 7000,
            "this is FBX {}.{}, older than FBX 7, which this reader and Blender cannot read",
            v / 1000,
            v % 1000 / 100
        );
    }
    let fbxcel::tree::any::AnyTree::V7400(version, tree, _) =
        fbxcel::tree::any::AnyTree::from_seekable_reader(std::io::Cursor::new(data))
            .map_err(|e| invalid!("not a readable FBX file: {e}"))?
    else {
        bail!("unsupported FBX version")
    };
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
    let objects_node = rootnode.first_child_by_name("Objects").context("FBX has no Objects")?;
    let mut objs = Objects {
        nodes: HashMap::new(),
        order: Vec::new(),
        parents: HashMap::new(),
        children: HashMap::new(),
    };
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

    // The file's axis system and unit, mapped to glTF (X right, Y up, Z toward the viewer,
    // meters) by a root node.
    let settings = rootnode
        .first_child_by_name("GlobalSettings")
        .map(props)
        .unwrap_or_default();
    let axis = |name: &str, sign: &str, default: i32| -> (usize, f64) {
        #[allow(clippy::cast_possible_truncation, reason = "an axis number; `as` saturates bad ones")]
        let a = scalar(&settings, name).map_or(default, |v| v as i32);
        let s = scalar(&settings, sign).unwrap_or(1.0);
        (
            usize::try_from(a.clamp(0, 2)).unwrap_or(1),
            if s < 0.0 { -1.0 } else { 1.0 },
        )
    };
    let (up, up_s) = axis("UpAxis", "UpAxisSign", 1);
    let (front, front_s) = axis("FrontAxis", "FrontAxisSign", 2);
    let (coord, coord_s) = axis("CoordAxis", "CoordAxisSign", 0);
    let unit_cm = scalar(&settings, "UnitScaleFactor").unwrap_or(1.0);
    let unit = unit_cm / 100.0;
    let mut basis = [[0.0; 4]; 4];
    basis[0][coord] = coord_s * unit;
    basis[1][up] = up_s * unit;
    basis[2][front] = front_s * unit;
    basis[3][3] = 1.0;
    // Degenerate axis settings (two axes the same): fall back to Y-up.
    if det3(&basis).abs() < 1e-12 {
        basis = scale([unit; 3]);
    }
    let named = |a: usize, s: f64| format!("{}{}", if s < 0.0 { '-' } else { '+' }, ["X", "Y", "Z"][a]);
    let source = json!({
        "format": "FBX",
        "version": format!("{}.{}", version.major(), version.minor()),
        "creator": creator,
        "unitMeters": unit,
        "upAxis": named(up, up_s),
        "frontAxis": named(front, front_s),
        "coordAxis": named(coord, coord_s),
    });

    let mut r = Reader {
        objs: &objs,
        linear: creator.contains("Blender"),
        dir: input.dir(),
        assets,
        keep,
        b: Builder::default(),
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
        unit,
        skinned_roots: Vec::new(),
    };
    let models: Vec<i64> = objs.order.iter().copied().filter(|id| objs.is("Model", *id)).collect();
    for &m in &models {
        let node = r.model_node(m);
        r.nodes.push(node);
        r.node_of.insert(m, r.nodes.len() - 1);
    }
    // Parents, once every model has its node.
    let mut roots = Vec::new();
    for &m in &models {
        let n = r.node_of[&m];
        match objs.model_parent(m).and_then(|p| r.node_of.get(&p)) {
            Some(&p) if p != n && !r.is_ancestor(n, p) => r.add_child(p, n),
            _ => roots.push(n),
        }
    }
    for &m in &models {
        r.attach(m)?;
    }
    // A geometry no model holds is drawn where it is.
    for &g in &objs.order {
        if objs.is_kind("Geometry", g, "Mesh") && !objs.parents_of(g).any(|p| objs.is("Model", p)) {
            let mesh = r.mesh_for(g, 0)?;
            r.nodes.push(json!({ "mesh": mesh }));
            roots.push(r.nodes.len() - 1);
        }
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
    ensure!(!r.nodes.is_empty(), "FBX has no mesh geometry");

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
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": scene_nodes }]));
    json.insert("nodes".into(), Value::Array(std::mem::take(&mut r.nodes)));
    if !r.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut r.meshes)));
    }
    if !r.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut r.skins)));
    }
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    if !r.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut r.cameras)));
    }
    if !r.lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    let (materials, images) = (std::mem::take(&mut r.materials), std::mem::take(&mut r.images));
    let mut b = r.b;
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// Object id → (node, class), and how objects connect.
struct Objects<'a> {
    nodes: HashMap<i64, (NodeHandle<'a>, String)>,
    /// Object ids in file order.
    order: Vec<i64>,
    /// child → parents (object-object connections), in file order.
    parents: HashMap<i64, Vec<i64>>,
    /// parent → children with the property they plug into ("" for object-object).
    children: HashMap<i64, Vec<(i64, String)>>,
}

impl<'a> Objects<'a> {
    fn node(&self, id: i64) -> Option<NodeHandle<'a>> {
        self.nodes.get(&id).map(|(n, _)| *n)
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

/// A model's local transform: `T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1`, the FBX SDK's
/// order, with `t`, `r` and `s` standing in for its `Lcl` properties when given.
fn local_of(p: &Props, t: Option<[f64; 3]>, r: Option<[f64; 3]>, s: Option<[f64; 3]>) -> M4 {
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    let order = scalar(p, "RotationOrder").map_or(0, |v| v as i32);
    let r_pre = rotate(vec3(p, "PreRotation", [0.0; 3]), 0);
    let r_post_inv = transpose3(&rotate(vec3(p, "PostRotation", [0.0; 3]), 0));
    let rp = vec3(p, "RotationPivot", [0.0; 3]);
    let sp = vec3(p, "ScalingPivot", [0.0; 3]);
    [
        translate(t.unwrap_or_else(|| vec3(p, "Lcl Translation", [0.0; 3]))),
        translate(vec3(p, "RotationOffset", [0.0; 3])),
        translate(rp),
        r_pre,
        rotate(r.unwrap_or_else(|| vec3(p, "Lcl Rotation", [0.0; 3])), order),
        r_post_inv,
        translate(rp.map(|c| -c)),
        translate(vec3(p, "ScalingOffset", [0.0; 3])),
        translate(sp),
        scale(s.unwrap_or_else(|| vec3(p, "Lcl Scaling", [1.0; 3]))),
        translate(sp.map(|c| -c)),
    ]
    .iter()
    .fold(IDENTITY, |acc, m| mul(&acc, m))
}

/// The transpose of a rotation (its inverse).
fn transpose3(r: &M4) -> M4 {
    let mut t = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = r[j][i];
        }
    }
    t
}

/// A model's geometric transform: applied to what it holds, not passed to its children.
fn geometric_of(p: &Props) -> M4 {
    mul(
        &translate(vec3(p, "GeometricTranslation", [0.0; 3])),
        &mul(
            &rotate(vec3(p, "GeometricRotation", [0.0; 3]), 0),
            &scale(vec3(p, "GeometricScaling", [1.0; 3])),
        ),
    )
}
/// A property's values as JSON: a number, a string, or a list of them.
fn prop_json(vals: &[A]) -> Option<Value> {
    let one = |v: &A| match v {
        A::String(s) => Some(json!(s.split('\0').next().unwrap_or(""))),
        A::Bool(b) => Some(json!(b)),
        A::I16(i) => Some(json!(i)),
        A::I32(i) => Some(json!(i)),
        A::I64(i) => Some(json!(i)),
        A::F32(f) if f.is_finite() => Some(json!(f)),
        A::F64(f) if f.is_finite() => Some(json!(f)),
        _ => None,
    };
    match vals {
        [] => None,
        [v] => one(v),
        vs => vs.iter().map(one).collect::<Option<Vec<_>>>().map(Value::Array),
    }
}

/// A model's user-defined properties (flagged `U`), for its node's extras.
fn user_props(node: NodeHandle<'_>) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(p70) = node.first_child_by_name("Properties70") {
        for p in p70.children_by_name("P") {
            let a = p.attributes();
            if let (Some(name), Some(flags)) = (a.first().and_then(string), a.get(3).and_then(string))
                && flags.contains('U')
                && let Some(v) = prop_json(a.get(4..).unwrap_or_default())
            {
                out.insert(name.to_owned(), v);
            }
        }
    }
    out
}

/// An animation curve: keys in seconds, and whether each holds its value until the next.
struct Curve {
    times: Vec<f64>,
    values: Vec<f64>,
    constant: Vec<bool>,
}

impl Curve {
    fn read(node: NodeHandle<'_>) -> Option<Self> {
        let times: Vec<f64> = child_attr(node, "KeyTime")
            .and_then(ints)?
            .into_iter()
            .map(|t| {
                #[allow(clippy::cast_precision_loss, reason = "FBX time to seconds, as the SDK does")]
                let t = t as f64;
                t / KTIME
            })
            .collect();
        let values = floats(child_attr(node, "KeyValueFloat")?)?;
        if times.is_empty() || times.len() != values.len() || times.windows(2).any(|w| w[1] < w[0]) {
            return None;
        }
        // Flags are shared by runs of keys: flag i covers the next `KeyAttrRefCount[i]` keys.
        let flags = child_attr(node, "KeyAttrFlags").and_then(ints).unwrap_or_default();
        let counts = child_attr(node, "KeyAttrRefCount").and_then(ints).unwrap_or_default();
        let mut constant = Vec::with_capacity(times.len());
        for (f, n) in flags.iter().zip(&counts) {
            for _ in 0..usize::try_from(*n).unwrap_or(0) {
                if constant.len() < times.len() {
                    constant.push(f & 0x2 != 0);
                }
            }
        }
        constant.resize(times.len(), false);
        Some(Self {
            times,
            values,
            constant,
        })
    }

    /// The value at `t`: the first or last key's outside them, linear between keys (as
    /// Blender's and Assimp's importers read FBX) unless the earlier key is constant.
    fn at(&self, t: f64) -> f64 {
        let i = self.times.partition_point(|&k| k <= t);
        if i == 0 {
            return self.values[0];
        }
        if i == self.times.len() {
            return self.values[i - 1];
        }
        let (t0, t1) = (self.times[i - 1], self.times[i]);
        if self.constant[i - 1] || t1 <= t0 {
            return self.values[i - 1];
        }
        let f = (t - t0) / (t1 - t0);
        self.values[i - 1] + (self.values[i] - self.values[i - 1]) * f
    }

    /// Euler angles in degrees turned the short way where a curve crosses ±180° between two keys. Exporters
    /// bake an angle going past 180 as its other side (179 then -179), which read literally is a whole
    /// turn back in one frame, not two degrees on. A turn over 180° is only a wrap when it dwarfs the
    /// curve's usual step: a coin keyed 0 then 360 still spins.
    fn unwrap_degrees(&mut self) {
        let mut steps: Vec<f64> = self
            .values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut by = 0.0;
        for i in 1..self.values.len() {
            let raw = self.values[i] + by;
            let d = raw - self.values[i - 1];
            if d.abs() > 180.0 && d.abs() > usual * 4.0 {
                let turns = (d / 360.0).round() * 360.0;
                by -= turns;
            }
            self.values[i] += by;
        }
    }

    /// Times just before each key that follows a constant one, so a linear sampler steps.
    fn steps(&self, out: &mut Vec<f64>) {
        for i in 1..self.times.len() {
            if self.constant[i - 1] {
                let dt = self.times[i] - self.times[i - 1];
                out.push(self.times[i] - (dt / 2.0).min(1e-3));
            }
        }
    }
}

/// An animated vector property (`Lcl Translation`, ...): a curve or a fixed value per component.
struct Animated {
    curves: [Option<Curve>; 3],
    fixed: [f64; 3],
}

impl Animated {
    fn at(&self, t: f64) -> [f64; 3] {
        std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.at(t)))
    }

    fn times(&self, out: &mut Vec<f64>) {
        for c in self.curves.iter().flatten() {
            out.extend_from_slice(&c.times);
            c.steps(out);
        }
    }

    /// Euler rotation keys turned the short way, as Maya's Euler filter turns them. Exporters bake each
    /// frame's rotation as any of the angles that make it: an angle past 180 as its other side (179 then
    /// -179), or a limb raised past straight up as the other set of three (two axes half a turn on, the
    /// middle one mirrored). Read literally, either is a spin in one frame. Where a key jumps more than
    /// a quarter turn from the one before, and far more than the curve's usual step (so a coin keyed 0 then
    /// 360 still spins), it's given the angles making the same rotation that lie nearest the last key.
    fn unwrap_rotation(&mut self, order: i32) {
        let Some(times) = self.curves.iter().flatten().next().map(|c| c.times.clone()) else {
            return;
        };
        let shared = self
            .curves
            .iter()
            .flatten()
            .all(|c| c.times.len() == times.len() && c.times.iter().zip(&times).all(|(a, b)| (a - b).abs() < 1e-9));
        if !shared {
            // The three can't be read together key by key: each turns the short way on its own.
            self.curves.iter_mut().flatten().for_each(Curve::unwrap_degrees);
            return;
        }
        let raw: Vec<[f64; 3]> = (0..times.len())
            .map(|i| std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.values[i])))
            .collect();
        let apart = |a: &[f64; 3], b: &[f64; 3]| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
        let mut steps: Vec<f64> = raw
            .windows(2)
            .map(|w| apart(&w[0], &w[1]))
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut out = raw.clone();
        for i in 1..raw.len() {
            let (prev, key) = (out[i - 1], raw[i]);
            let jump = apart(&prev, &key);
            if jump <= 90.0 || jump <= usual * 4.0 {
                continue;
            }
            let nearest = |v: [f64; 3]| -> [f64; 3] {
                std::array::from_fn(|k| v[k] + 360.0 * ((prev[k] - v[k]) / 360.0).round())
            };
            let made = rotate(key, order);
            let mut best = nearest(key);
            // The other set of three: whichever axis is the middle one for this order is mirrored.
            for middle in 0..3 {
                let other: [f64; 3] =
                    std::array::from_fn(|k| if k == middle { 180.0 - key[k] } else { key[k] + 180.0 });
                let m = rotate(other, order);
                if (0..3).all(|r| (0..3).all(|c| (m[r][c] - made[r][c]).abs() < 1e-6)) {
                    let other = nearest(other);
                    if apart(&prev, &other) < apart(&prev, &best) {
                        best = other;
                    }
                }
            }
            out[i] = best;
        }
        let constant = self
            .curves
            .iter()
            .flatten()
            .next()
            .map(|c| c.constant.clone())
            .unwrap_or_default();
        for k in 0..3 {
            let values: Vec<f64> = out.iter().map(|v| v[k]).collect();
            match &mut self.curves[k] {
                Some(c) => c.values = values,
                // An axis with no curve of its own gets one when its angle had to change.
                None if values.iter().any(|v| (v - self.fixed[k]).abs() > 1e-9) => {
                    self.curves[k] = Some(Curve {
                        times: times.clone(),
                        values,
                        constant: constant.clone(),
                    });
                }
                None => {}
            }
        }
    }
}

/// Sorted, de-duplicated times, shifted so none is negative.
fn timeline(mut times: Vec<f64>) -> Vec<f64> {
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    times
}

/// A mesh built for one geometry and the materials of the model holding it.
struct Built {
    mesh: usize,
    /// Whether it carries joints and weights (and so needs a skin).
    skinned: bool,
}

type Props = HashMap<String, Vec<A>>;

/// A blend shape channel's name, weight (0 to 1), and offsets at full weight by control point.
type Shape = (String, f64, HashMap<usize, [f32; 3]>);

/// Builds the document as the file's objects are visited.
struct Reader<'o, 'a, 'x> {
    objs: &'o Objects<'a>,
    /// Blender writes material colors linear; everyone else sRGB.
    linear: bool,
    dir: &'x Path,
    assets: &'x dyn Assets,
    keep: Keep,
    b: Builder,
    nodes: Vec<Value>,
    /// Model id → node.
    node_of: HashMap<i64, usize>,
    meshes: Vec<Value>,
    /// (geometry, its materials) → mesh.
    mesh_of: HashMap<(i64, Vec<usize>), Built>,
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
    /// Mesh → the blend shape channels its targets come from, in order.
    morphs: HashMap<usize, Vec<i64>>,
    /// Meters per file unit.
    unit: f64,
    /// Nodes holding skinned meshes, which go at the scene's root, outside the axis and unit conversion.
    skinned_roots: Vec<usize>,
}

impl Reader<'_, '_, '_> {
    /// A model's node: its name, local transform and user properties.
    fn model_node(&self, m: i64) -> Value {
        let Some(node) = self.objs.node(m) else {
            return json!({});
        };
        let p = props(node);
        let mut out = json!({});
        if let Some(name) = name_of(node) {
            out["name"] = json!(name);
        }
        set_transform(&mut out, &local_of(&p, None, None, None));
        let extras = user_props(node);
        if !extras.is_empty() {
            out["extras"] = Value::Object(extras);
        }
        out
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let kids = self.nodes[parent]
            .as_object_mut()
            .expect("a node is an object")
            .entry("children")
            .or_insert_with(|| json!([]));
        kids.as_array_mut().expect("children is a list").push(json!(child));
    }

    /// Whether `a` is `b` or one of its ancestors, through the children set so far.
    fn is_ancestor(&self, a: usize, b: usize) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![a];
        while let Some(n) = stack.pop() {
            if n == b {
                return true;
            }
            if std::mem::replace(&mut seen[n], true) {
                continue;
            }
            stack.extend(
                self.nodes[n]["children"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64)
                    .filter_map(|c| usize::try_from(c).ok()),
            );
        }
        false
    }

    /// What model `m` holds: its geometry (with skin and morph targets), camera and light.
    fn attach(&mut self, m: i64) -> Result<()> {
        let node = self.node_of[&m];
        let p = self.objs.node(m).map(props).unwrap_or_default();
        let geometric = geometric_of(&p);
        let geometries: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|g| self.objs.is_kind("Geometry", *g, "Mesh"))
            .collect();
        for g in geometries {
            let built = self.mesh_for(g, m)?;
            let skin = self
                .objs
                .children_of(g)
                .find(|d| self.objs.is_kind("Deformer", *d, "Skin"));
            let skinned = self.mesh_of.values().any(|b| b.mesh == built && b.skinned);
            let target = if skinned && let Some(skin) = skin {
                // A skinned mesh's node transform is ignored, its joints place it (the geometric transform
                // goes into the inverse bind matrices), so it gets a root node of its own, as glTF asks; the
                // model's node stays where it is, a joint for vertices no bone weights.
                let s = self.skin(skin, m, &geometric);
                let mut host = json!({ "skin": s });
                if let Some(name) = self.nodes[node].get("name") {
                    host["name"] = name.clone();
                }
                self.nodes.push(host);
                self.skinned_roots.push(self.nodes.len() - 1);
                self.nodes.len() - 1
            } else if geometric == IDENTITY && self.nodes[node].get("mesh").is_none() {
                node
            } else {
                self.child(node, Some(&geometric))
            };
            self.nodes[target]["mesh"] = json!(built);
            for (k, ch) in self
                .morphs
                .get(&built)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .enumerate()
            {
                self.channel_users.entry(ch).or_default().push((target, k));
            }
        }
        let attributes: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|a| self.objs.is("NodeAttribute", *a))
            .collect();
        for a in attributes {
            let Some(attr) = self.objs.node(a) else { continue };
            match kind_of(attr) {
                "Camera" if self.keep.cameras => {
                    let cam = camera(&props(attr), self.unit);
                    self.cameras.push(cam);
                    // FBX cameras look down +X, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        0.0,
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["camera"] = json!(self.cameras.len() - 1);
                }
                "Light" if self.keep.lights => {
                    let Some(light) = light(&props(attr)) else { continue };
                    self.lights.push(light);
                    // FBX lights shine down -Y, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["extensions"] = json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A new child of `parent`, with transform `m`; returns it.
    fn child(&mut self, parent: usize, m: Option<&M4>) -> usize {
        let mut n = json!({});
        if let Some(m) = m {
            set_transform(&mut n, m);
        }
        self.nodes.push(n);
        let c = self.nodes.len() - 1;
        self.add_child(parent, c);
        c
    }

    /// Model `m`'s materials, in connection order: a geometry's material layer indexes into this list.
    fn model_materials(&mut self, m: i64) -> Vec<usize> {
        let ids: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|c| self.objs.is("Material", *c))
            .collect();
        if !self.keep.materials {
            return Vec::new();
        }
        ids.into_iter()
            .map(|mid| {
                if let Some(&i) = self.material_index.get(&mid) {
                    return i;
                }
                let mat = read_material(
                    mid,
                    self.objs,
                    self.linear,
                    self.dir,
                    self.keep.textures.then_some(self.assets),
                    &mut self.images,
                    &mut self.image_index,
                );
                self.materials.push(mat);
                self.material_index.insert(mid, self.materials.len() - 1);
                self.materials.len() - 1
            })
            .collect()
    }

    fn default_material(&mut self) -> usize {
        *self.default_material.get_or_insert_with(|| {
            self.materials.push(Material::plain("default", GRAY).into());
            self.materials.len() - 1
        })
    }

    /// The blend shape channels deforming geometry `g`, in target order.
    fn morph_channels(&self, g: i64) -> Vec<i64> {
        self.objs
            .children_of(g)
            .filter(|d| self.objs.is_kind("Deformer", *d, "BlendShape"))
            .flat_map(|d| self.objs.children_of(d).collect::<Vec<_>>())
            .filter(|c| self.objs.is_kind("Deformer", *c, "BlendShapeChannel"))
            .collect()
    }

    /// The mesh for geometry `g` as model `m` (0 for none) uses it.
    fn mesh_for(&mut self, g: i64, m: i64) -> Result<usize> {
        let mats = if m == 0 { Vec::new() } else { self.model_materials(m) };
        if let Some(b) = self.mesh_of.get(&(g, mats.clone())) {
            return Ok(b.mesh);
        }
        let geom = self.objs.node(g).context("missing geometry")?;
        let prims = geometry(geom)?;
        // Skin weights per control point: (joint, weight), heaviest first.
        let influences = if self.keep.skins {
            self.influences(g)
        } else {
            HashMap::new()
        };
        let channels = if self.keep.morph_targets {
            self.morph_channels(g)
        } else {
            Vec::new()
        };
        let (morph_ids, shapes): (Vec<i64>, Vec<Shape>) =
            channels.iter().filter_map(|&c| Some((c, self.shape(c)?))).unzip();
        let joint_sets = influences
            .values()
            .map(|v| v.len().div_ceil(4))
            .max()
            .unwrap_or(0)
            .min(4);
        let mut primitives = Vec::new();
        for (slot, prim, cps) in prims {
            let material = match mats.get(slot) {
                Some(&i) => i,
                None => self.default_material(),
            };
            if !prim.uvs.is_empty() && !self.textured.contains(&material) {
                self.textured.push(material);
            }
            let Some(mut json) = self.b.primitive(prim) else {
                continue;
            };
            json["material"] = json!(material);
            if joint_sets > 0 {
                let unweighted = influences.len() + 1; // stands for the model itself; see `skin`
                for set in 0..joint_sets {
                    let mut joints = Vec::with_capacity(cps.len() * 4);
                    let mut weights = Vec::with_capacity(cps.len() * 4);
                    for cp in &cps {
                        let inf = influences.get(cp).map_or(&[][..], Vec::as_slice);
                        for k in set * 4..set * 4 + 4 {
                            match inf.get(k) {
                                Some(&(j, w)) => {
                                    joints.push(u16::try_from(j).unwrap_or(0));
                                    weights.push(w);
                                }
                                None if k == 0 => {
                                    joints.push(u16::try_from(unweighted).unwrap_or(0));
                                    weights.push(1.0);
                                }
                                None => {
                                    joints.push(0);
                                    weights.push(0.0);
                                }
                            }
                        }
                    }
                    let n = cps.len();
                    let j = self.b.accessor(Part::u16(joints), 5123, n, "VEC4", Some(34962));
                    let w = self.b.accessor(Part::f32(weights), 5126, n, "VEC4", Some(34962));
                    json["attributes"][format!("JOINTS_{set}")] = json!(j);
                    json["attributes"][format!("WEIGHTS_{set}")] = json!(w);
                }
            }
            if !shapes.is_empty() {
                let targets: Vec<Value> = shapes
                    .iter()
                    .map(|(_, _, deltas)| {
                        let d: Vec<[f32; 3]> = cps
                            .iter()
                            .map(|cp| deltas.get(cp).copied().unwrap_or([0.0; 3]))
                            .collect();
                        json!({ "POSITION": self.b.positions(d) })
                    })
                    .collect();
                json["targets"] = json!(targets);
            }
            primitives.push(json);
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(name) = name_of(geom) {
            mesh["name"] = json!(name);
        }
        if !shapes.is_empty() {
            mesh["weights"] = json!(shapes.iter().map(|s| s.1).collect::<Vec<_>>());
            mesh["extras"] = json!({ "targetNames": shapes.iter().map(|s| s.0.clone()).collect::<Vec<_>>() });
        }
        self.meshes.push(mesh);
        let built = Built {
            mesh: self.meshes.len() - 1,
            skinned: joint_sets > 0,
        };
        let i = built.mesh;
        self.mesh_of.insert((g, mats), built);
        if !morph_ids.is_empty() {
            self.morphs.insert(i, morph_ids);
        }
        Ok(i)
    }

    /// The skin deformer's clusters on geometry `g`, as control point → (cluster number, weight), heaviest first and
    /// summing to one. Clusters whose bone isn't a model are left out.
    fn influences(&self, g: i64) -> HashMap<usize, Vec<(usize, f32)>> {
        let mut out: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        let Some(skin) = self
            .objs
            .children_of(g)
            .find(|d| self.objs.is_kind("Deformer", *d, "Skin"))
        else {
            return out;
        };
        for (j, cluster) in self.clusters(skin).into_iter().enumerate() {
            let Some(node) = self.objs.node(cluster) else { continue };
            let idx = child_attr(node, "Indexes").and_then(ints).unwrap_or_default();
            let w = child_attr(node, "Weights").and_then(floats).unwrap_or_default();
            for (&cp, &w) in idx.iter().zip(&w) {
                if let Ok(cp) = usize::try_from(cp)
                    && w > 0.0
                {
                    out.entry(cp).or_default().push((j, f32_of(w)));
                }
            }
        }
        for inf in out.values_mut() {
            inf.sort_by(|a, b| b.1.total_cmp(&a.1));
            inf.truncate(16);
            let sum: f32 = inf.iter().map(|x| x.1).sum();
            for x in inf.iter_mut() {
                x.1 /= sum;
            }
        }
        out
    }

    /// A skin deformer's clusters that name a bone.
    fn clusters(&self, skin: i64) -> Vec<i64> {
        self.objs
            .children_of(skin)
            .filter(|c| self.objs.is_kind("Deformer", *c, "Cluster"))
            .filter(|c| self.objs.children_of(*c).any(|b| self.node_of.contains_key(&b)))
            .collect()
    }

    /// The skin for deformer `skin` on model `m`: a joint per cluster, and the model's own node last, for
    /// vertices no cluster weights.
    fn skin(&mut self, skin: i64, m: i64, geometric: &M4) -> usize {
        if let Some(&s) = self.skin_of.get(&(skin, m)) {
            return s;
        }
        let mut joints = Vec::new();
        let mut ibms = Vec::new();
        for cluster in self.clusters(skin) {
            let Some(node) = self.objs.node(cluster) else { continue };
            let bone = self
                .objs
                .children_of(cluster)
                .find(|b| self.node_of.contains_key(b))
                .unwrap_or(0);
            let matrix = |name: &str| {
                child_attr(node, name)
                    .and_then(floats)
                    .filter(|v| v.len() == 16)
                    .map(|v| {
                        let mut m = IDENTITY;
                        for (k, x) in v.iter().enumerate() {
                            m[k % 4][k / 4] = *x;
                        }
                        m
                    })
            };
            // `Transform` takes the model's space to the bone's at bind time: the file stores it relative to
            // `TransformLink` (the bone's bind-time world), though the FBX SDK hands back `TransformLink ·
            // Transform`, the model's world. Without it, the bind pose is the pose the file is in.
            let transform = matrix("Transform").unwrap_or_else(|| {
                let link = matrix("TransformLink").unwrap_or_else(|| self.world_of(bone));
                mul(&inverse(&link), &self.world_of(m))
            });
            // The vertex goes geometry → model → bone space.
            ibms.extend(column_major(&mul(&transform, geometric)).map(f32_of));
            joints.push(self.node_of[&bone]);
        }
        // The model's own node: at bind time it's where `Transform` says, so its inverse bind matrix is the
        // geometric transform.
        joints.push(self.node_of[&m]);
        ibms.extend(column_major(geometric).map(f32_of));
        let acc = self.b.floats(ibms, "MAT4", false);
        self.skins.push(json!({ "joints": joints, "inverseBindMatrices": acc }));
        let s = self.skins.len() - 1;
        self.skin_of.insert((skin, m), s);
        s
    }

    /// Model `m`'s world transform in the file's space: its parents' times its own.
    fn world_of(&self, m: i64) -> M4 {
        let mut world = IDENTITY;
        let mut at = Some(m);
        // At most one step per model, so a cycle in a broken file ends.
        for _ in 0..=self.node_of.len() {
            let Some(id) = at else { break };
            let p = self.objs.node(id).map(props).unwrap_or_default();
            world = mul(&local_of(&p, None, None, None), &world);
            at = self.objs.model_parent(id).filter(|&p| p != id);
        }
        world
    }

    /// A blend shape channel's name, weight (0 to 1) and full-weight shape, as control point → offset.
    fn shape(&self, channel: i64) -> Option<Shape> {
        let node = self.objs.node(channel)?;
        let shape = self
            .objs
            .children_of(channel)
            .filter(|s| self.objs.is_kind("Geometry", *s, "Shape"))
            .last()?;
        let shape = self.objs.node(shape)?;
        let idx = child_attr(shape, "Indexes").and_then(ints)?;
        let v = child_attr(shape, "Vertices").and_then(floats)?;
        let mut deltas = HashMap::new();
        for (k, &cp) in idx.iter().enumerate() {
            if let (Ok(cp), Some(d)) = (usize::try_from(cp), v.get(k * 3..k * 3 + 3)) {
                deltas.insert(cp, [d[0], d[1], d[2]].map(f32_of));
            }
        }
        let weight = scalar(&props(node), "DeformPercent").unwrap_or(0.0) / 100.0;
        Some((name_of(node).unwrap_or("").to_owned(), weight.clamp(0.0, 1.0), deltas))
    }

    /// Every animation stack as a glTF animation, its curves sampled at their keys.
    fn animations(&mut self) -> Vec<Value> {
        let stacks: Vec<i64> = self
            .objs
            .order
            .iter()
            .copied()
            .filter(|s| self.objs.is("AnimationStack", *s))
            .collect();
        let mut out = Vec::new();
        for stack in stacks {
            let layers: Vec<i64> = self
                .objs
                .children_of(stack)
                .filter(|l| self.objs.is("AnimationLayer", *l))
                .collect();
            let curve_nodes: Vec<i64> = layers
                .iter()
                .flat_map(|&l| self.objs.children_of(l).collect::<Vec<_>>())
                .filter(|c| self.objs.is("AnimationCurveNode", *c))
                .collect();
            let in_stack = |c: &i64| curve_nodes.contains(c);
            let mut channels = Vec::new();
            let mut samplers = Vec::new();
            let mut per_node: Vec<(usize, [Option<Animated>; 3])> = Vec::new();
            let models: Vec<(i64, usize)> = self.node_of.iter().map(|(&m, &n)| (m, n)).collect();
            for (m, node) in models {
                let tracks = ["Lcl Translation", "Lcl Rotation", "Lcl Scaling"].map(|prop| {
                    let cn = self.objs.child_at(m, prop).find(in_stack)?;
                    let mut a = self.animated(cn, ["d|X", "d|Y", "d|Z"])?;
                    if prop == "Lcl Rotation" {
                        let p = self.objs.node(m).map(props).unwrap_or_default();
                        #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
                        let order = scalar(&p, "RotationOrder").map_or(0, |v| v as i32);
                        a.unwrap_rotation(order);
                    }
                    Some(a)
                });
                if tracks.iter().any(Option::is_some) {
                    per_node.push((node, tracks));
                }
            }
            per_node.sort_by_key(|(n, _)| *n);
            let model_of: HashMap<usize, i64> = self.node_of.iter().map(|(&m, &n)| (n, m)).collect();
            for (node, tracks) in per_node {
                let p = self.objs.node(model_of[&node]).map(props).unwrap_or_default();
                self.transform_channels(node, &p, &tracks, &mut channels, &mut samplers);
            }
            // Morph weights: every channel of a mesh at once, per node drawing it.
            let mut weight_nodes: HashMap<usize, Vec<(usize, Animated)>> = HashMap::new();
            let users: Vec<(i64, Vec<(usize, usize)>)> =
                self.channel_users.iter().map(|(c, u)| (*c, u.clone())).collect();
            for (ch, users) in users {
                let Some(cn) = self.objs.child_at(ch, "DeformPercent").find(in_stack) else {
                    continue;
                };
                for (node, target) in users {
                    if let Some(a) = self.animated(cn, ["d|DeformPercent", "", ""]) {
                        weight_nodes.entry(node).or_default().push((target, a));
                    }
                }
            }
            let mut weight_nodes: Vec<_> = weight_nodes.into_iter().collect();
            weight_nodes.sort_by_key(|(n, _)| *n);
            for (node, tracks) in weight_nodes {
                self.weight_channel(node, &tracks, &mut channels, &mut samplers);
            }
            if channels.is_empty() {
                continue;
            }
            let mut anim = json!({ "channels": channels, "samplers": samplers });
            if let Some(name) = self.objs.node(stack).and_then(name_of) {
                anim["name"] = json!(name);
            }
            out.push(anim);
        }
        out
    }

    /// A curve node's curves for its components (`d|X`, ...; "" for none), with its own values where a component
    /// has no curve.
    fn animated(&self, cn: i64, components: [&str; 3]) -> Option<Animated> {
        let node = self.objs.node(cn)?;
        let p = props(node);
        let curves = components.map(|c| {
            if c.is_empty() {
                return None;
            }
            let curve = self.objs.child_at(cn, c).find(|k| self.objs.is("AnimationCurve", *k))?;
            Curve::read(self.objs.node(curve)?)
        });
        if curves.iter().all(Option::is_none) {
            return None;
        }
        let fixed = components.map(|c| scalar(&p, c).unwrap_or(0.0));
        Some(Animated { curves, fixed })
    }

    /// Translation, rotation and scale channels for a node whose `Lcl` properties are animated: its whole local
    /// transform is evaluated at each time (pivots and pre-rotations included) and taken apart again.
    fn transform_channels(
        &mut self,
        node: usize,
        p: &Props,
        tracks: &[Option<Animated>; 3],
        channels: &mut Vec<Value>,
        samplers: &mut Vec<Value>,
    ) {
        let mut times = Vec::new();
        for t in tracks.iter().flatten() {
            t.times(&mut times);
        }
        let mut times = timeline(times);
        // Cut long turns into short steps.
        if let Some(r) = &tracks[1] {
            let mut fine = Vec::with_capacity(times.len());
            for w in times.windows(2) {
                fine.push(w[0]);
                let (a, b) = (r.at(w[0]), r.at(w[1] - 1e-9));
                let turn = (0..3).map(|k| (b[k] - a[k]).abs()).fold(0.0, f64::max);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a small step count"
                )]
                let n = ((turn / MAX_STEP_DEGREES).ceil() as usize).min(360);
                for k in 1..n {
                    #[allow(clippy::cast_precision_loss, reason = "a small step count")]
                    fine.push(w[0] + (w[1] - w[0]) * k as f64 / n as f64);
                }
            }
            fine.extend(times.last());
            times = fine;
        }
        if times.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let at = |k: usize, t: f64| tracks[k].as_ref().map(|a| a.at(t));
        let mut trs: Vec<Trs> = times
            .iter()
            .map(|&t| decompose(&local_of(p, at(0, t), at(1, t), at(2, t))))
            .collect();
        // Keep each quaternion on the same side as the last, so slerp takes the short way.
        for i in 1..trs.len() {
            let prev = trs[i - 1].r;
            let q = &mut trs[i].r;
            if (0..4).map(|k| prev[k] * q[k]).sum::<f64>() < 0.0 {
                *q = q.map(|c| -c);
            }
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let rest = decompose(&local_of(p, None, None, None));
        let paths: [(&str, Vec<f64>, Vec<f64>); 3] = [
            ("translation", trs.iter().flat_map(|x| x.t).collect(), rest.t.to_vec()),
            ("rotation", trs.iter().flat_map(|x| x.r).collect(), rest.r.to_vec()),
            ("scale", trs.iter().flat_map(|x| x.s).collect(), rest.s.to_vec()),
        ];
        for (path, values, rest) in paths {
            let width = rest.len();
            let still = values
                .chunks_exact(width)
                .all(|v| v.iter().zip(&rest).all(|(a, b)| (a - b).abs() < 1e-7));
            if still {
                continue;
            }
            let output = self.b.floats(
                values.into_iter().map(f32_of).collect(),
                if width == 4 { "VEC4" } else { "VEC3" },
                false,
            );
            samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
            channels.push(json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": path } }));
        }
        // An animated node holds TRS, never a matrix.
        if self.nodes[node]
            .as_object_mut()
            .and_then(|n| n.remove("matrix"))
            .is_some()
        {
            let n = &mut self.nodes[node];
            n["translation"] = json!(rest.t);
            n["rotation"] = json!(rest.r);
            n["scale"] = json!(rest.s);
        }
    }

    /// A weights channel for a node whose mesh's blend shape channels are animated.
    fn weight_channel(
        &mut self,
        node: usize,
        tracks: &[(usize, Animated)],
        channels: &mut Vec<Value>,
        samplers: &mut Vec<Value>,
    ) {
        let Some(mesh) = self.nodes[node]["mesh"].as_u64().and_then(|m| usize::try_from(m).ok()) else {
            return;
        };
        let rest: Vec<f64> = self.meshes[mesh]["weights"]
            .as_array()
            .map(|w| w.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let mut times = Vec::new();
        for (_, a) in tracks {
            a.times(&mut times);
        }
        let times = timeline(times);
        if times.is_empty() || rest.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let mut values = Vec::with_capacity(times.len() * rest.len());
        for &t in &times {
            let mut w = rest.clone();
            for (k, a) in tracks {
                if let Some(x) = w.get_mut(*k) {
                    *x = (a.at(t)[0] / 100.0).clamp(0.0, 1.0);
                }
            }
            values.extend(w.into_iter().map(f32_of));
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let output = self.b.floats(values, "SCALAR", false);
        samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
        channels.push(json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": "weights" } }));
    }
}

/// A camera's glTF form. FBX gives the field of view horizontally unless its aperture mode says otherwise.
fn camera(p: &Props, unit: f64) -> Value {
    let aspect = match (scalar(p, "AspectWidth"), scalar(p, "AspectHeight")) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => w / h,
        _ => 16.0 / 9.0,
    };
    // Planes are in the file's units, but a node's scale doesn't reach a camera's planes: they're converted here.
    let near = scalar(p, "NearPlane").filter(|n| *n > 0.0).unwrap_or(10.0);
    let far = scalar(p, "FarPlane").filter(|f| *f > near).unwrap_or(near * 10_000.0);
    let (near, far) = (near * unit, far * unit);
    if scalar(p, "CameraProjectionType") == Some(1.0) {
        let zoom = scalar(p, "OrthoZoom").filter(|z| *z > 0.0).unwrap_or(1.0);
        return json!({ "type": "orthographic", "orthographic": { "xmag": zoom * aspect, "ymag": zoom, "znear": near, "zfar": far } });
    }
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    let mode = scalar(p, "ApertureMode").map_or(1, |m| m as i32);
    let deg = scalar(p, "FieldOfView")
        .filter(|f| *f > 0.0 && *f < 180.0)
        .unwrap_or(40.0);
    let yfov = match mode {
        2 => deg.to_radians(),
        0 => scalar(p, "FieldOfViewY")
            .filter(|f| *f > 0.0 && *f < 180.0)
            .unwrap_or(deg)
            .to_radians(),
        3 => match (scalar(p, "FocalLength"), scalar(p, "FilmHeight")) {
            (Some(f), Some(h)) if f > 0.0 && h > 0.0 => 2.0 * (h * 25.4 / 2.0 / f).atan(),
            _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
        },
        _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
    };
    json!({ "type": "perspective", "perspective": { "yfov": yfov, "aspectRatio": aspect, "znear": near, "zfar": far } })
}

/// A light's `KHR_lights_punctual` form: point, directional or spot (area and volume lights have none).
fn light(p: &Props) -> Option<Value> {
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    let kind = scalar(p, "LightType").map_or(0, |t| t as i32);
    let color = vec3(p, "Color", [1.0; 3]).map(|c| srgb_to_linear(f32_of(c.clamp(0.0, 1.0))));
    let intensity = scalar(p, "Intensity").unwrap_or(100.0).max(0.0) / 100.0;
    let mut out = json!({ "color": color, "intensity": intensity });
    match kind {
        0 => out["type"] = json!("point"),
        1 => out["type"] = json!("directional"),
        2 => {
            let half = std::f64::consts::FRAC_PI_2;
            let outer = (scalar(p, "OuterAngle").unwrap_or(45.0).to_radians() / 2.0).clamp(1e-4, half);
            let inner = (scalar(p, "InnerAngle").unwrap_or(0.0).to_radians() / 2.0).clamp(0.0, outer * 0.999);
            out["type"] = json!("spot");
            out["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
        }
        _ => return None,
    }
    Some(out)
}

/// "Mesh" in `Geometry: 123, "Geometry::Name", "Mesh"`.
fn kind_of(node: NodeHandle<'_>) -> &str {
    node.attributes().get(2).and_then(string).unwrap_or("")
}

/// The material slot a texture connects to, from the property it connects as: the classic slots (`DiffuseColor`,
/// `NormalMap`, ...), Stingray PBS's (`Maya|TEX_color_map`, ...) and 3ds Max Physical's
/// (`3dsMax|Parameters|base_color_map`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Color,
    Normal,
    Emissive,
    Occlusion,
    Metallic,
    Roughness,
    Opacity,
}

fn slot(prop: &str) -> Option<Slot> {
    let s: String = prop
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase();
    Some(
        if s.contains("emissive") || s.contains("emitcolor") || s.contains("emission") {
            Slot::Emissive
        } else if s.contains("normal") || s == "bump" || s.ends_with("bumpmap") {
            Slot::Normal
        } else if s.contains("aomap") || s.contains("ambientocclusion") {
            Slot::Occlusion
        } else if s.contains("metallic") || s.contains("metalness") {
            Slot::Metallic
        } else if s.contains("roughness") {
            Slot::Roughness
        } else if s.contains("transparentcolor")
            || s.contains("transparencyfactor")
            || s.contains("opacity")
            || s.contains("cutout")
        {
            Slot::Opacity
        } else if s.contains("diffusecolor") || s == "diffuse" || s.contains("basecolor") || s.ends_with("texcolormap")
        {
            Slot::Color
        } else {
            return None;
        },
    )
}

#[allow(clippy::too_many_lines, reason = "one property after another")]
fn read_material(
    id: i64,
    objs: &Objects<'_>,
    linear: bool,
    dir: &Path,
    // None when textures aren't kept: they aren't read.
    assets: Option<&dyn Assets>,
    images: &mut Images,
    image_index: &mut HashMap<i64, Option<usize>>,
) -> Full {
    let (node, _) = objs.nodes[&id];
    let p = props(node);
    let name = node
        .attributes()
        .get(1)
        .and_then(string)
        .unwrap_or("material")
        .split('\0')
        .next()
        .unwrap_or("material")
        .to_owned();
    let to_linear = |c: [f64; 3]| {
        c.map(|v| {
            let v = f32_of(v.clamp(0.0, 1.0));
            if linear { v } else { srgb_to_linear(v) }
        })
    };
    let d = to_linear(vec3(&p, "DiffuseColor", vec3(&p, "Diffuse", [0.8, 0.8, 0.8])));
    let alpha =
        scalar(&p, "Opacity").unwrap_or_else(|| 1.0 - scalar(&p, "TransparencyFactor").unwrap_or(0.0).clamp(0.0, 1.0));
    let mut mat = Full::from(Material::plain(
        &name,
        [d[0], d[1], d[2], f32_of(alpha.clamp(0.0, 1.0))],
    ));
    // The emissive color is a color (sRGB unless Blender's); its factor is a linear intensity.
    let factor = f32_of(scalar(&p, "EmissiveFactor").unwrap_or(1.0).max(0.0));
    mat.emissive = to_linear(vec3(&p, "EmissiveColor", vec3(&p, "Emissive", [0.0; 3]))).map(|c| c * factor);
    // A Phong exponent as roughness the way Blender's importer maps it (and its exporter inverts): 1 - sqrt(n) / 10.
    if let Some(n) = scalar(&p, "Shininess").or_else(|| scalar(&p, "ShininessExponent")) {
        mat.roughness = f32_of(1.0 - n.clamp(0.0, 100.0).sqrt() / 10.0);
    }
    // Blender keeps its metallic in ReflectionFactor; other tools' reflectivity is a mirror blend, not metal.
    if linear && let Some(m) = scalar(&p, "ReflectionFactor") {
        mat.metallic = f32_of(m.clamp(0.0, 1.0));
    }
    let f = |k: &str| scalar(&p, k).map(f32_of);
    // Stingray PBS (Maya, and 3ds Max's export of it).
    if p.contains_key("Maya|base_color") {
        let c = to_linear(vec3(&p, "Maya|base_color", [1.0; 3]));
        mat.base_color[..3].copy_from_slice(&c);
        mat.metallic = f("Maya|metallic").unwrap_or(0.0);
        mat.roughness = f("Maya|roughness").unwrap_or(0.5);
        let e = to_linear(vec3(&p, "Maya|emissive", [0.0; 3]));
        mat.emissive = e.map(|c| c * f("Maya|emissive_intensity").unwrap_or(1.0).max(0.0));
    }
    // 3ds Max's Physical Material.
    let k = |name: &str| format!("3dsMax|Parameters|{name}");
    if p.contains_key(&k("base_color")) {
        let c = to_linear(vec3(&p, &k("base_color"), [1.0; 3]));
        let w = f(&k("base_weight")).unwrap_or(1.0);
        mat.base_color[..3].copy_from_slice(&c.map(|c| c * w));
        mat.metallic = f(&k("metalness")).unwrap_or(0.0);
        let r = f(&k("roughness")).unwrap_or(0.0);
        // Its roughness may be a glossiness.
        mat.roughness = if f(&k("roughness_inv")).is_some_and(|i| i > 0.5) {
            1.0 - r
        } else {
            r
        };
        let e = to_linear(vec3(&p, &k("emit_color"), [0.0; 3]));
        mat.emissive = e.map(|c| c * f(&k("emission")).unwrap_or(0.0).max(0.0));
        if let Some(t) = f(&k("trans_weight")).filter(|t| *t > 0.0) {
            mat.transmission = Some(Factor::of(t));
        }
        mat.ior = f(&k("trans_ior")).filter(|i| *i >= 1.0);
        if let Some(c) = f(&k("coat")).filter(|c| *c > 0.0) {
            mat.clearcoat = Some(Clearcoat {
                factor: Factor::of(c),
                roughness: Factor::of(f(&k("coat_roughness")).unwrap_or(0.0)),
                normal: None,
            });
        }
        if let Some(s) = f(&k("sheen")).filter(|s| *s > 0.0) {
            mat.sheen = Some(Sheen {
                color: to_linear(vec3(&p, &k("sheen_color"), [1.0; 3])).map(|c| c * s),
                color_tex: None,
                roughness: Factor::of(f(&k("sheen_roughness")).unwrap_or(0.3)),
            });
        }
    }
    let (mut metallic, mut roughness, mut opacity) = (None, None, None);
    for (tex, prop) in objs.children.get(&id).cloned().unwrap_or_default() {
        let Some(slot) = slot(&prop) else {
            continue;
        };
        let Some(assets) = assets else {
            continue;
        };
        let img = *image_index
            .entry(tex)
            .or_insert_with(|| texture_image(tex, objs, dir, assets, images));
        let Some(t) = img.map(|i| texture_tex(tex, objs, i)) else {
            continue;
        };
        match slot {
            Slot::Color if mat.base_color_tex.is_none() => {
                // A textured material's diffuse factor is usually a leftover; white shows the texture as painted.
                if !p.contains_key("Maya|base_color") {
                    mat.base_color[..3].copy_from_slice(&[1.0, 1.0, 1.0]);
                }
                mat.base_color_tex = Some(t);
            }
            Slot::Normal if mat.normal_tex.is_none() => mat.normal_tex = Some(t),
            Slot::Emissive if mat.emissive_tex.is_none() => mat.emissive_tex = Some(t),
            Slot::Occlusion if mat.occlusion_tex.is_none() => mat.occlusion_tex = Some(t),
            Slot::Metallic => metallic = Some(t),
            Slot::Roughness => roughness = Some(t),
            Slot::Opacity => opacity = Some(t),
            _ => {}
        }
    }
    // glTF reads roughness and metalness from one image's green and blue.
    if let (Some(m), Some(r)) = (&metallic, &roughness)
        && m.image == r.image
    {
        mat.metallic_roughness_tex.clone_from(&roughness);
        mat.metallic = 1.0;
        mat.roughness = 1.0;
    }
    // Opacity from the color texture is its alpha, glTF's own arrangement.
    if let (Some(o), Some(c)) = (&opacity, &mat.base_color_tex)
        && o.image == c.image
        && mat.alpha == Alpha::Opaque
    {
        mat.alpha = Alpha::Blend;
    }
    mat
}

/// A texture of `image` as texture `id` lays it out: its wrap modes, and its translation, rotation and scaling (on
/// UVs with V up, as `uv' = rotate * scale * uv + translate`) as a transform of the document's V-down UVs.
fn texture_tex(id: i64, objs: &Objects<'_>, image: usize) -> Tex {
    let mut t = Tex::new(image);
    let Some((node, _)) = objs.nodes.get(&id) else {
        return t;
    };
    let p = props(*node);
    let wrap = |k: &str| match scalar(&p, k) {
        Some(w) if w != 0.0 => material::CLAMP,
        _ => material::REPEAT,
    };
    t.wrap = [wrap("WrapModeU"), wrap("WrapModeV")];
    let [tu, tv, _] = vec3(&p, "Translation", [0.0; 3]).map(f32_of);
    let [_, _, angle] = vec3(&p, "Rotation", [0.0; 3]).map(f32_of);
    let [su, sv, _] = vec3(&p, "Scaling", [1.0; 3]).map(f32_of);
    let (su, sv) = (if su == 0.0 { 1.0 } else { su }, if sv == 0.0 { 1.0 } else { sv });
    let angle = angle.to_radians();
    if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
        let (sin, cos) = angle.sin_cos();
        t.transform = Some(UvTransform {
            offset: [tu - sv * sin, 1.0 - tv - sv * cos],
            rotation: angle,
            scale: [su, sv],
        });
    }
    t
}

/// A texture's pixels: embedded in a connected `Video`'s `Content`, or the file it names, found in the pack.
fn texture_image(id: i64, objs: &Objects<'_>, dir: &Path, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    let (node, _) = objs.nodes.get(&id)?;
    let names: Vec<String> = ["RelativeFilename", "FileName"]
        .iter()
        .filter_map(|k| child_attr(*node, k).and_then(string).map(str::to_owned))
        .filter(|s| !s.is_empty())
        .collect();
    for (vid, _) in objs.children.get(&id).cloned().unwrap_or_default() {
        if let Some((video, _)) = objs.nodes.get(&vid)
            && video.name() == "Video"
            && let Some(A::Binary(bytes)) = child_attr(*video, "Content")
            && !bytes.is_empty()
        {
            let hint = names.first().map_or_else(
                || Path::new("t.png").to_path_buf(),
                |n| Path::new(&n.replace('\\', "/")).to_path_buf(),
            );
            let name = hint.to_string_lossy().into_owned();
            let convert = |b: &[u8]| assets.image(b, &hint);
            if let Some(i) = images.bytes(Cow::Borrowed(&bytes[..]), &name, Location::Embedded, convert) {
                return Some(i);
            }
        }
    }
    let (name, path) = names.iter().find_map(|n| Some((n, assets.find(n, dir)?)))?;
    images.file(assets, path, name)
}

/// Vertices already emitted for a primitive, by (control point, normal index, uv index, color index).
type Seen = HashMap<(usize, usize, usize, usize), u32>;

/// A geometry's triangles in its own space, a primitive per material slot (sorted), each vertex's control point
/// alongside so skins and blend shapes can find it.
fn geometry(geom: NodeHandle<'_>) -> Result<Vec<(usize, Prim, Vec<usize>)>> {
    let verts = floats(child_attr(geom, "Vertices").context("mesh has no vertices")?).context("bad vertices")?;
    let poly_index =
        ints(child_attr(geom, "PolygonVertexIndex").context("mesh has no polygons")?).context("bad polygons")?;
    let normals = Layer::read(geom, "LayerElementNormal", "Normals", "NormalsIndex", 3);
    let uvs = Layer::read(geom, "LayerElementUV", "UV", "UVIndex", 2);
    let colors = Layer::read(geom, "LayerElementColor", "Colors", "ColorIndex", 4);
    let mut tris = Vec::new();
    let mut corner_pos = Vec::new();
    let mat_layer = geom.children_by_name("LayerElementMaterial").next().and_then(|el| {
        Some(Layer {
            values: ints(child_attr(el, "Materials")?)?
                .into_iter()
                .map(|i| {
                    #[allow(clippy::cast_precision_loss, reason = "a material slot, far below 2^53")]
                    let i = i as f64;
                    i
                })
                .collect(),
            index: None,
            mapping: child_attr(el, "MappingInformationType")
                .and_then(string)
                .unwrap_or("AllSame")
                .to_owned(),
            width: 1,
        })
    });
    // Per material slot: a primitive with de-duplicated vertices, and each vertex's control point.
    let mut prims: HashMap<usize, (Prim, Seen, Vec<usize>)> = HashMap::new();
    let mut poly_start = 0;
    let mut poly = 0;
    for (k, &raw) in poly_index.iter().enumerate() {
        if raw >= 0 {
            continue;
        }
        // A negative index ends a polygon (stored as !index).
        let corners: Vec<(usize, usize)> = (poly_start..=k)
            .map(|pv| {
                let v = poly_index[pv];
                (pv, usize::try_from(if v < 0 { !v } else { v }).unwrap_or(usize::MAX))
            })
            .collect();
        poly_start = k + 1;
        poly += 1;
        // A polygon naming a control point that isn't there, or too few to make a triangle, is left out.
        if corners.len() < 3 || corners.iter().any(|&(_, cp)| cp >= verts.len() / 3) {
            continue;
        }
        let slot = mat_layer
            .as_ref()
            .and_then(|l| l.get(k, 0, poly - 1))
            .map_or(0, |(_, v)| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a material slot; `as` saturates bad ones and `get` refuses them"
                )]
                let slot = v[0] as usize;
                slot
            });
        let (prim, seen, cps) = prims.entry(slot).or_insert_with(|| {
            (
                Prim {
                    material: slot,
                    ..Prim::default()
                },
                HashMap::new(),
                Vec::new(),
            )
        });
        let mut ids = Vec::with_capacity(corners.len());
        corner_pos.clear();
        for &(pv, cp) in &corners {
            let pos = &verts[cp * 3..cp * 3 + 3];
            let p = [pos[0], pos[1], pos[2]].map(f32_of);
            corner_pos.push(p);
            let n = normals.as_ref().and_then(|l| l.get(pv, cp, poly - 1));
            let t = uvs.as_ref().and_then(|l| l.get(pv, cp, poly - 1));
            let c = colors.as_ref().and_then(|l| l.get(pv, cp, poly - 1));
            let key = (
                cp,
                n.map_or(usize::MAX, |x| x.0),
                t.map_or(usize::MAX, |x| x.0),
                c.map_or(usize::MAX, |x| x.0),
            );
            let id = *seen.entry(key).or_insert_with(|| {
                prim.positions.push(p);
                cps.push(cp);
                if let Some((_, nv)) = n {
                    prim.normals.push(unit([nv[0], nv[1], nv[2]]).map(f32_of));
                }
                if let Some((_, tv)) = t {
                    // FBX's texture origin is bottom-left, glTF's top-left.
                    prim.uvs.push([f32_of(tv[0]), 1.0 - f32_of(tv[1])]);
                }
                if let Some((_, cv)) = c {
                    let [r, g, b, a] = [cv[0], cv[1], cv[2], cv[3]].map(|v| f32_of(v.clamp(0.0, 1.0)));
                    prim.colors
                        .push([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]);
                }
                u32::try_from(prim.positions.len() - 1).unwrap_or(u32::MAX)
            });
            ids.push(id);
        }
        tris.clear();
        triangulate(&corner_pos, &mut tris);
        for &[a, b, c] in &tris {
            prim.indices.extend([a, b, c].map(|i| ids[i as usize]));
        }
    }
    let mut out: Vec<(usize, Prim, Vec<usize>)> = prims
        .into_iter()
        .filter(|(_, (p, _, _))| !p.indices.is_empty())
        .map(|(slot, (mut prim, _, cps))| {
            if prim.normals.len() != prim.positions.len() {
                prim.normals = smooth_normals(&prim.positions, &prim.indices);
            }
            if prim.uvs.len() != prim.positions.len() {
                prim.uvs.clear();
            }
            if prim.colors.len() != prim.positions.len() {
                prim.colors.clear();
            }
            (slot, prim, cps)
        })
        .collect();
    out.sort_unstable_by_key(|x| x.0);
    let triangles: usize = out.iter().map(|x| x.1.indices.len() / 3).sum();
    ensure!(triangles <= MAX_TRIANGLES, "too many triangles to convert");
    Ok(out)
}

/// `v` scaled to length one (straight up when it has none).
fn unit(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 { v.map(|c| c / len) } else { [0.0, 1.0, 0.0] }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::document::Pose;
    use crate::mesh::Mesh;
    use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

    /// The file at `p`, with assets under `dir`, flattened in its rest pose.
    fn flat(p: &Path, dir: &Path) -> Mesh {
        let doc = load(
            crate::formats::Input::file(p),
            &crate::assets::Folder::new(dir),
            Keep::ALL,
        )
        .unwrap();
        doc.flatten(Pose::Rest).unwrap().unwrap()
    }

    fn png(rgb: [u8; 3]) -> Vec<u8> {
        let mut b = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb(rgb))
            .write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png)
            .unwrap();
        b
    }

    /// A Z-up, centimeter FBX (3ds Max style): one textured quad on the ground, its model raised 100 cm.
    pub(crate) fn z_up_fbx() -> Vec<u8> {
        let mut w = Writer::new(std::io::Cursor::new(Vec::new()), fbxcel::low::FbxVersion::V7_4).unwrap();
        let p = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, name: &str, vals: &[A]| {
            let mut a = w.new_node("P").unwrap();
            for s in [name, name, "", "A"] {
                a.append_string_direct(s).unwrap();
            }
            for v in vals {
                match v {
                    A::F64(f) => a.append_f64(*f).unwrap(),
                    A::I32(i) => a.append_i32(*i).unwrap(),
                    _ => unreachable!(),
                }
            }
            w.close_node().unwrap();
        };
        let object = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, kind: &str, id: i64, name: &str, sub: &str| {
            let mut a = w.new_node(kind).unwrap();
            a.append_i64(id).unwrap();
            a.append_string_direct(&format!("{name}\0\x01{kind}")).unwrap();
            a.append_string_direct(sub).unwrap();
        };
        // Global settings: Z up, -Y front, X right, centimeters.
        w.new_node("GlobalSettings").unwrap();
        w.new_node("Properties70").unwrap();
        for (k, v) in [
            ("UpAxis", 2),
            ("UpAxisSign", 1),
            ("FrontAxis", 1),
            ("FrontAxisSign", -1),
            ("CoordAxis", 0),
            ("CoordAxisSign", 1),
        ] {
            p(&mut w, k, &[A::I32(v)]);
        }
        p(&mut w, "UnitScaleFactor", &[A::F64(1.0)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        w.new_node("Objects").unwrap();
        object(&mut w, "Geometry", 10, "Quad", "Mesh");
        {
            let mut a = w.new_node("Vertices").unwrap();
            a.append_arr_f64_from_iter(None, [0., 0., 0., 100., 0., 0., 100., 100., 0., 0., 100., 0.])
                .unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("PolygonVertexIndex").unwrap();
            a.append_arr_i32_from_iter(None, [0, 1, 2, !3]).unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("LayerElementUV").unwrap();
            a.append_i32(0).unwrap();
        }
        for (name, s) in [
            ("MappingInformationType", "ByPolygonVertex"),
            ("ReferenceInformationType", "IndexToDirect"),
        ] {
            w.new_node(name).unwrap().append_string_direct(s).unwrap();
            w.close_node().unwrap();
        }
        w.new_node("UV")
            .unwrap()
            .append_arr_f64_from_iter(None, [0., 0., 1., 0., 1., 1., 0., 1.])
            .unwrap();
        w.close_node().unwrap();
        w.new_node("UVIndex")
            .unwrap()
            .append_arr_i32_from_iter(None, [0, 1, 2, 3])
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap(); // LayerElementUV
        w.close_node().unwrap(); // Geometry

        object(&mut w, "Model", 20, "Floor", "Mesh");
        w.new_node("Properties70").unwrap();
        p(&mut w, "Lcl Translation", &[A::F64(0.0), A::F64(0.0), A::F64(100.0)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Material", 30, "Stone", "");
        w.new_node("Properties70").unwrap();
        p(&mut w, "DiffuseColor", &[A::F64(0.5), A::F64(0.4), A::F64(0.3)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Texture", 40, "StoneTex", "");
        w.new_node("RelativeFilename")
            .unwrap()
            .append_string_direct("C:\\art\\stone.png")
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Video", 50, "StoneVideo", "Clip");
        w.new_node("Content")
            .unwrap()
            .append_binary_direct(&png([10, 200, 30]))
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap(); // Objects

        w.new_node("Connections").unwrap();
        for (kind, child, parent, prop) in [
            ("OO", 10, 20, ""),
            ("OO", 20, 0, ""),
            ("OO", 30, 20, ""),
            ("OP", 40, 30, "DiffuseColor"),
            ("OO", 50, 40, ""),
        ] {
            let mut a = w.new_node("C").unwrap();
            a.append_string_direct(kind).unwrap();
            a.append_i64(child).unwrap();
            a.append_i64(parent).unwrap();
            if !prop.is_empty() {
                a.append_string_direct(prop).unwrap();
            }
            w.close_node().unwrap();
        }
        w.close_node().unwrap();
        w.finalize_and_flush(&FbxFooter::default()).unwrap().into_inner()
    }

    #[test]
    fn z_up_centimeters_become_y_up_meters_with_embedded_texture() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("floor.fbx");
        std::fs::write(&p, z_up_fbx()).unwrap();
        let mesh = flat(&p, dir.path());
        assert_eq!(mesh.triangles(), 2);
        let prim = &mesh.prims[0];
        for v in &prim.positions {
            assert!((v[1] - 1.0).abs() < 1e-5, "raised 100 cm = 1 m up: {v:?}");
        }
        let xs: Vec<f32> = prim.positions.iter().map(|v| v[0]).collect();
        let zs: Vec<f32> = prim.positions.iter().map(|v| v[2]).collect();
        assert!(
            xs.iter().any(|&x| (x - 1.0).abs() < 1e-5) && zs.iter().any(|&z| (z + 1.0).abs() < 1e-5),
            "{:?}",
            prim.positions
        );
        // Ground-facing quad in Z-up faces +Y once converted.
        assert!(prim.normals.iter().all(|n| n[1] > 0.99), "{:?}", prim.normals);
        assert_eq!(prim.uvs.len(), prim.positions.len());
        assert!(
            prim.uvs.iter().any(|t| (t[1] - 1.0).abs() < 1e-6),
            "V flipped to glTF's origin"
        );
        let mat = &mesh.materials[prim.material];
        assert_eq!(mat.name, "Stone");
        assert_eq!(mat.color_image, Some(0));
        assert_eq!(mesh.images.len(), 1);
        let glb = crate::mesh::to_glb(&mesh).unwrap();
        let doc: serde_json::Value = {
            let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
            serde_json::from_slice(&glb[20..20 + len]).unwrap()
        };
        assert_eq!(
            doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            0
        );
        assert!(doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"].is_number());
    }

    /// `z_up_fbx` as ASCII FBX, the way Asset Forge writes it.
    fn z_up_ascii_fbx() -> String {
        use base64::Engine as _;
        let png = base64::engine::general_purpose::STANDARD.encode(png([10, 200, 30]));
        let (head, tail) = png.split_at(png.len() / 2);
        format!(
            r#"; FBX 7.3.0 project file
FBXHeaderExtension:  {{
	FBXHeaderVersion: 1003
	FBXVersion: 7300
}}
GlobalSettings:  {{
	Version: 1000
	Properties70:  {{
		P: "UpAxis", "int", "Integer", "",2
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",1
		P: "FrontAxisSign", "int", "Integer", "",-1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
	}}
}}
Objects:  {{
	Geometry: 10, "Geometry::Quad", "Mesh" {{
		Vertices: *12 {{
			a: 0,0,0,100,0,0,100,100,0,0,100,0
		}}
		PolygonVertexIndex: *4 {{
			a: 0,1,2,-4
		}}
		LayerElementUV: 0 {{
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *8 {{
				a: 0,0,1,0,1,1,0,1
			}}
			UVIndex: *4 {{
				a: 0,1,2,3
			}}
		}}
	}}
	Model: 20, "Model::Floor", "Mesh" {{
		Properties70:  {{
			P: "Lcl Translation", "Lcl Translation", "", "A",0,0,100
		}}
	}}
	Material: 30, "Material::Stone", "" {{
		Properties70:  {{
			P: "DiffuseColor", "Color", "", "A",0.5,0.4,0.3
		}}
	}}
	Texture: 40, "Texture::StoneTex", "" {{
		RelativeFilename: "C:\art\stone.png"
	}}
	Video: 50, "Video::StoneVideo", "Clip" {{
		Content: , "{head}",
			"{tail}"
	}}
}}
Connections:  {{
	;Geometry::Quad, Model::Floor
	C: "OO",10,20
	C: "OO",20,0
	C: "OO",30,20
	C: "OP",40,30, "DiffuseColor"
	C: "OO",50,40
}}
"#
        )
    }

    #[test]
    fn ascii_fbx_reads_like_binary() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("ascii.fbx"), dir.path().join("binary.fbx"));
        std::fs::write(&a, z_up_ascii_fbx()).unwrap();
        std::fs::write(&b, z_up_fbx()).unwrap();
        let (a, b) = (flat(&a, dir.path()), flat(&b, dir.path()));
        assert_eq!(a.triangles(), 2);
        assert_eq!(a.prims[0].positions, b.prims[0].positions);
        assert_eq!(a.prims[0].normals, b.prims[0].normals);
        assert_eq!(a.prims[0].uvs, b.prims[0].uvs);
        assert_eq!(a.materials[a.prims[0].material].name, "Stone");
        assert_eq!(a.materials[a.prims[0].material].color_image, Some(0));
        assert_eq!(a.images.len(), 1);
    }

    #[test]
    fn object_ids_past_2_pow_53_stay_distinct() {
        // As doubles these are one number, so the model and its material would share a key.
        let (model, material) = ("9007199254740992", "9007199254740993");
        let text = z_up_ascii_fbx()
            .replace("Model: 20,", &format!("Model: {model},"))
            .replace("Material: 30,", &format!("Material: {material},"))
            .replace("C: \"OO\",10,20", &format!("C: \"OO\",10,{model}"))
            .replace("C: \"OO\",20,0", &format!("C: \"OO\",{model},0"))
            .replace("C: \"OO\",30,20", &format!("C: \"OO\",{material},{model}"))
            .replace("C: \"OP\",40,30,", &format!("C: \"OP\",40,{material},"));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big_ids.fbx");
        std::fs::write(&p, text).unwrap();
        let mesh = flat(&p, dir.path());
        assert_eq!(mesh.triangles(), 2);
        assert!(
            mesh.prims[0].positions.iter().all(|v| (v[1] - 1.0).abs() < 1e-5),
            "placed by its model"
        );
        assert_eq!(mesh.materials[mesh.prims[0].material].name, "Stone");
    }

    fn curve(values: &[f64]) -> Curve {
        #[allow(clippy::cast_precision_loss, reason = "a few keys")]
        let times = (0..values.len()).map(|i| i as f64 / 30.0).collect();
        Curve {
            times,
            values: values.to_vec(),
            constant: vec![false; values.len()],
        }
    }

    #[test]
    fn a_turn_past_180_baked_as_its_other_side_goes_the_short_way() {
        // A sway through 180 (a model turned round), written as -179 after 179, and back.
        let mut c = curve(&[175.0, 177.0, 179.0, -179.0, -177.0, -179.0, 179.0, 177.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [175.0, 177.0, 179.0, 181.0, 183.0, 181.0, 179.0, 177.0]);
        // Past 180 twice over, on a curve that steps a degree at a time.
        let mut c = curve(&[0.0, 1.0, 2.0, 361.0, 362.0, 1.0, 0.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn a_limb_raised_past_straight_up_keeps_turning_the_short_way() {
        // An arm raised from 70° through straight up to 110° about Y (XYZ order), baked past 90 as the
        // other set of three: X and Z half a turn on, Y mirrored (Kenney's kicks are written so).
        let keys = [
            [10.0, 70.0, 20.0],
            [10.0, 85.0, 20.0],
            [-170.0, 80.0, -160.0],
            [-170.0, 70.0, -160.0],
        ];
        let mut a = Animated {
            curves: std::array::from_fn(|k| Some(curve(&keys.map(|v| v[k])))),
            fixed: [0.0; 3],
        };
        a.unwrap_rotation(0);
        let got: Vec<[f64; 3]> = (0..4)
            .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().unwrap().values[i]))
            .collect();
        let want = [
            [10.0, 70.0, 20.0],
            [10.0, 85.0, 20.0],
            [10.0, 100.0, 20.0],
            [10.0, 110.0, 20.0],
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((0..3).all(|k| (g[k] - w[k]).abs() < 1e-9), "{got:?}");
        }
    }

    #[test]
    fn a_whole_turn_keyed_as_one_still_spins() {
        // A coin keyed at 0, 360 and 720: each step is the curve's usual one.
        let mut c = curve(&[0.0, 360.0, 720.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 360.0, 720.0]);
        let mut c = curve(&[0.0, 360.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 360.0]);
        let mut a = rotation(&[[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), [[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
    }

    /// An animated rotation with a curve on every axis, one key per row (a thirtieth of a second apart).
    fn rotation(rows: &[[f64; 3]]) -> Animated {
        Animated {
            curves: std::array::from_fn(|k| Some(curve(&rows.iter().map(|v| v[k]).collect::<Vec<_>>()))),
            fixed: [0.0; 3],
        }
    }

    /// The angles at each key: an axis without a curve holds its fixed value.
    fn keys(a: &Animated) -> Vec<[f64; 3]> {
        let n = a.curves.iter().flatten().map(|c| c.values.len()).max().unwrap_or(0);
        (0..n)
            .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().map_or(a.fixed[k], |c| c.values[i])))
            .collect()
    }

    fn same_rotation(a: [f64; 3], b: [f64; 3], order: i32) -> bool {
        let (m, n) = (rotate(a, order), rotate(b, order));
        (0..3).all(|r| (0..3).all(|c| (m[r][c] - n[r][c]).abs() < 1e-6))
    }

    fn near(got: &[[f64; 3]], want: &[[f64; 3]]) -> bool {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (0..3).all(|k| (g[k] - w[k]).abs() < 1e-6))
    }

    /// An angle as an exporter writes it: in (-180, 180].
    fn baked(a: f64) -> f64 {
        let w = (a + 180.0).rem_euclid(360.0) - 180.0;
        if w <= -180.0 { w + 360.0 } else { w }
    }

    /// The other angles making the same rotation: the order's middle axis mirrored, the others half a turn on.
    fn flipped(v: [f64; 3], order: i32) -> [f64; 3] {
        let middle = middle_axis(order);
        std::array::from_fn(|k| if k == middle { 180.0 - v[k] } else { v[k] + 180.0 })
    }

    /// The axis applied second: XYZ and ZYX turn about Y in the middle, XZY and YZX about Z, YXZ and ZXY about X.
    fn middle_axis(order: i32) -> usize {
        match order {
            1 | 2 => 2,
            3 | 4 => 0,
            _ => 1,
        }
    }

    #[test]
    fn a_limb_raised_past_straight_up_turns_the_short_way_in_every_rotation_order() {
        for order in 0..=5 {
            let middle = middle_axis(order);
            // The middle axis swings from 60° through straight up to 120°, the others hold; past 90 the
            // exporter writes the other set of three.
            let truth: Vec<[f64; 3]> = [60.0, 70.0, 80.0, 95.0, 105.0, 120.0, 105.0, 85.0]
                .iter()
                .map(|&m| std::array::from_fn(|k| if k == middle { m } else { [10.0, -30.0, 20.0][k] }))
                .collect();
            let written: Vec<[f64; 3]> = truth
                .iter()
                .map(|v| {
                    if v[middle] > 90.0 {
                        flipped(*v, order).map(baked)
                    } else {
                        *v
                    }
                })
                .collect();
            for (t, w) in truth.iter().zip(&written) {
                assert!(
                    same_rotation(*t, *w, order),
                    "order {order}: the test's flip makes the same rotation"
                );
            }
            let mut a = rotation(&written);
            a.unwrap_rotation(order);
            assert!(near(&keys(&a), &truth), "order {order}: {:?}", keys(&a));
        }
    }

    #[test]
    fn a_steady_spin_baked_into_half_turns_either_side_keeps_turning_one_way() {
        for step in [30.0, -30.0, 45.0, -7.5] {
            let truth: Vec<[f64; 3]> = (0..40).map(|i| [0.0, 0.0, 15.0 + step * f64::from(i)]).collect();
            let written: Vec<[f64; 3]> = truth.iter().map(|v| v.map(baked)).collect();
            // The spin about Z alone, with no curves on X and Y.
            let mut a = Animated {
                curves: [
                    None,
                    None,
                    Some(curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>())),
                ],
                fixed: [0.0; 3],
            };
            a.unwrap_rotation(0);
            assert!(near(&keys(&a), &truth), "{step}° a key: {:?}", keys(&a));
            assert!(
                a.curves[0].is_none() && a.curves[1].is_none(),
                "axes that keep their angle get no curve"
            );
            // Each curve on its own turns the short way too.
            let mut c = curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>());
            c.unwrap_degrees();
            assert!(
                c.values.iter().zip(&truth).all(|(g, w)| (g - w[2]).abs() < 1e-9),
                "{step}°: {:?}",
                c.values
            );
        }
    }

    #[test]
    fn an_axis_without_a_curve_gets_one_when_a_flip_turns_it() {
        // Y holds 80 with no curve of its own while X and Z, turning a little each frame, flip half a turn:
        // the rotation the keys make is Y going on to 100, so Y needs keys.
        let written = [
            [10.0, 80.0, 20.0],
            [12.0, 80.0, 22.0],
            [-166.0, 80.0, -156.0],
            [-164.0, 80.0, -154.0],
        ];
        let mut a = Animated {
            curves: [
                Some(curve(&written.map(|v| v[0]))),
                None,
                Some(curve(&written.map(|v| v[2]))),
            ],
            fixed: [0.0, 80.0, 0.0],
        };
        a.unwrap_rotation(0);
        let want = [
            [10.0, 80.0, 20.0],
            [12.0, 80.0, 22.0],
            [14.0, 100.0, 24.0],
            [16.0, 100.0, 26.0],
        ];
        assert!(near(&keys(&a), &want), "{:?}", keys(&a));
        let y = a.curves[1].as_ref().expect("Y has keys now");
        assert_eq!(y.times, a.curves[0].as_ref().unwrap().times);
    }

    #[test]
    fn a_cut_or_a_fast_spin_is_kept_as_keyed() {
        // A cut: a degree a frame, then a jump to somewhere else entirely. The nearest angles making that
        // rotation are the ones written.
        let cut = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],
            [151.0, 0.0, 0.0],
        ];
        let mut a = rotation(&cut);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), cut);
        // A propeller keyed every third of a turn: every step is its usual one.
        let spin: Vec<[f64; 3]> = (0..6).map(|i| [0.0, 120.0 * f64::from(i), 0.0]).collect();
        let mut a = rotation(&spin);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), spin);
        let mut c = curve(&spin.iter().map(|v| v[1]).collect::<Vec<_>>());
        c.unwrap_degrees();
        assert_eq!(c.values, spin.iter().map(|v| v[1]).collect::<Vec<_>>());
    }

    #[test]
    fn curves_keyed_at_different_times_each_turn_the_short_way() {
        let mut x = curve(&[30.0, 30.0]);
        x.times = vec![0.0, 0.5];
        let z = curve(&[150.0, 165.0, 180.0, -165.0, -150.0]);
        let mut a = Animated {
            curves: [Some(x), None, Some(z)],
            fixed: [0.0; 3],
        };
        a.unwrap_rotation(0);
        assert_eq!(a.curves[0].as_ref().unwrap().values, [30.0, 30.0]);
        assert_eq!(
            a.curves[2].as_ref().unwrap().values,
            [150.0, 165.0, 180.0, 195.0, 210.0]
        );
    }

    #[test]
    fn odd_rotation_curves_are_left_alone() {
        let cases: [&[f64]; 6] = [
            &[42.0],
            &[5.0, 5.0, 5.0],
            &[0.0, f64::NAN, 10.0],
            &[0.0, f64::INFINITY, 0.0],
            &[f64::NEG_INFINITY, f64::INFINITY],
            &[1e300, -1e300, 1e300],
        ];
        for values in cases {
            let mut c = curve(values);
            c.unwrap_degrees();
            assert_eq!(format!("{:?}", c.values), format!("{values:?}"));
            let rows: Vec<[f64; 3]> = values.iter().map(|&v| [v, 0.0, 0.0]).collect();
            for order in [0, 3, 6, -1, 99] {
                let mut a = rotation(&rows);
                a.unwrap_rotation(order);
                assert_eq!(format!("{:?}", keys(&a)), format!("{rows:?}"), "order {order}");
            }
        }
        let mut none = Animated {
            curves: [None, None, None],
            fixed: [1.0, 2.0, 3.0],
        };
        none.unwrap_rotation(0);
        assert!(none.curves.iter().all(Option::is_none));
    }

    #[test]
    fn turning_keys_the_short_way_never_moves_a_key() {
        // Whatever angles the filter picks, each key makes the rotation it was written as: random keys in any
        // order, some wild, some with axes missing or keyed at other times.
        let mut rng = crate::test_rng::Pcg32::new(11);
        let angle = |rng: &mut crate::test_rng::Pcg32| f64::from(rng.f32()) * 1440.0 - 720.0;
        for trial in 0..500 {
            let order = [0, 1, 2, 3, 4, 5, 6, 42][rng.below(8) as usize];
            let n = 1 + rng.below(24) as usize;
            let mut a = Animated {
                curves: [None, None, None],
                fixed: std::array::from_fn(|_| angle(&mut rng)),
            };
            let mut last = [0.0; 3];
            for (k, last) in last.iter_mut().enumerate() {
                if rng.below(4) == 0 {
                    continue;
                }
                let values = (0..n)
                    .map(|_| {
                        if rng.below(3) == 0 {
                            return angle(&mut rng);
                        }
                        *last += f64::from(rng.f32()) * 20.0 - 10.0;
                        if rng.below(5) == 0 { baked(*last) } else { *last }
                    })
                    .collect::<Vec<_>>();
                let mut c = curve(&values);
                if rng.below(6) == 0 {
                    c.times.iter_mut().for_each(|t| *t *= 2.0);
                }
                a.curves[k] = Some(c);
            }
            let before = keys(&a);
            let shared = a.curves.iter().flatten().all(|c| {
                let first = a.curves.iter().flatten().next().unwrap();
                c.times == first.times
            });
            a.unwrap_rotation(order);
            let after = keys(&a);
            if shared {
                for (i, (b, g)) in before.iter().zip(&after).enumerate() {
                    assert!(
                        same_rotation(*b, *g, order),
                        "trial {trial}, key {i}: {b:?} became {g:?}"
                    );
                }
            } else {
                for (b, g) in before.iter().flatten().zip(after.iter().flatten()) {
                    let turns = (g - b) / 360.0;
                    assert!((turns - turns.round()).abs() < 1e-9, "trial {trial}: {b} became {g}");
                }
            }
        }
    }

    #[test]
    fn a_baked_turn_comes_back_as_smooth_as_it_was() {
        // A limb wandering a few degrees a frame for four seconds, written as an exporter bakes it: each angle
        // in (-180, 180], and now and then as the other set of three. The filter gives back a path as smooth
        // as the one wandered, making the same rotation at every key.
        let mut rng = crate::test_rng::Pcg32::new(3);
        for trial in 0..200 {
            let order = rng.below(6).cast_signed();
            let middle = middle_axis(order);
            let mut v: [f64; 3] = std::array::from_fn(|_| f64::from(rng.f32()) * 360.0 - 180.0);
            v[middle] = f64::from(rng.f32()) * 120.0 - 60.0;
            let mut truth = Vec::new();
            for _ in 0..120 {
                truth.push(v);
                for a in &mut v {
                    *a += f64::from(rng.f32()) * 8.0 - 4.0;
                }
                v[middle] = v[middle].clamp(-60.0, 60.0);
            }
            let written: Vec<[f64; 3]> = truth
                .iter()
                .map(|t| if rng.below(10) == 0 { flipped(*t, order) } else { *t }.map(baked))
                .collect();
            let mut a = rotation(&written);
            a.unwrap_rotation(order);
            let got = keys(&a);
            for (i, w) in got.windows(2).enumerate() {
                let step = (0..3).map(|k| (w[1][k] - w[0][k]).abs()).fold(0.0, f64::max);
                assert!(
                    step <= 4.0 + 1e-6,
                    "trial {trial} (order {order}), key {}: {:?} to {:?}",
                    i + 1,
                    w[0],
                    w[1]
                );
            }
            for (t, g) in truth.iter().zip(&got) {
                assert!(same_rotation(*t, *g, order), "trial {trial}: {t:?} became {g:?}");
            }
        }
    }
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
