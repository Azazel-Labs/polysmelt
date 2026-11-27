//! glTF 1.0, upgraded to 2.0 as it's read, so the rest of the library sees one glTF.
//!
//! - Objects keyed by ID become lists (in ID order, the binary glTF buffer first), each
//!   reference an index, and each ID the object's `name` when it has none.
//! - `KHR_binary_glTF` GLBs (version 1) read their body as buffer 0, and images in it as views.
//! - An accessor's `byteStride` moves to a view of its own; float `JOINT`s become unsigned
//!   shorts; a skin's `bindShapeMatrix` is baked into its inverse bind matrices; positions
//!   without bounds get them.
//! - `WEB3D_quantized_attributes`: an accessor with a `decodeMatrix` becomes floats, decoded.
//!   `CESIUM_RTC`: its center is added to every root node's translation.
//! - Semantics take 2.0's names (`JOINT` is `JOINTS_0`, `TEXCOORD` is `TEXCOORD_0`); others
//!   take a leading underscore. A node's several meshes become one mesh of all their primitives,
//!   and its `skeletons` and the skin's `jointNames` become the skin's joints.
//! - Materials: `KHR_materials_common`, or a technique's values (its parameters' defaults
//!   under the material's own): diffuse as the base color, emission, shininess as roughness,
//!   transparency, double-sidedness and blending; `CONSTANT` is unlit. Colors are taken as sRGB.
//!   `KHR_materials_common` lights become `KHR_lights_punctual` ones.
//! - What 2.0 has no place for (techniques, programs, shaders, texture formats, the material's
//!   own values, joint names) is kept in `extras.source`, beside the normalized source keys:
//!   the 1.0 version as written, the meter, +Y up and right-handedness declared (1.0 fixes them), the generator as `tool`
//!   and the copyright.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::document::Document;
use crate::mesh::srgb_to_linear;
use crate::num::{f32_of, usize_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// Whether `json` is glTF 1.0 (or an earlier draft).
pub(crate) fn is_v1(json: &Map<String, Value>) -> bool {
    // Lists are 2.0's, whatever the label says.
    if ["meshes", "nodes", "accessors"]
        .iter()
        .any(|k| json.get(*k).is_some_and(Value::is_array))
    {
        return false;
    }
    match json.get("asset").and_then(|a| a.get("version")) {
        Some(Value::String(v)) => v.starts_with('1') || v.starts_with("0."),
        Some(Value::Number(n)) => n.as_f64().is_some_and(|v| v < 2.0),
        Some(_) => false,
        None => ["meshes", "nodes", "accessors"]
            .iter()
            .any(|k| json.get(*k).is_some_and(Value::is_object)),
    }
}

/// The work left once the buffers are read: data 2.0 wants in another form.
#[derive(Default)]
pub(crate) struct Fixups {
    /// Accessors of float joint indices.
    joints: Vec<usize>,
    /// Skins whose bind shape matrix isn't the identity, and that matrix.
    bind_shapes: Vec<(usize, [f32; 16])>,
    /// Position accessors without `min` and `max`.
    bounds: Vec<usize>,
    /// Accessors of quantized values (`WEB3D_quantized_attributes`), and each one's decode
    /// matrix: column-major, one row and column more than the accessor has components.
    quantized: Vec<(usize, Vec<f64>)>,
}

/// One kind of 1.0 object: its entries in order, and each ID's index.
struct Kind {
    list: Vec<(String, Value)>,
    at: HashMap<String, usize>,
}

impl Kind {
    fn take(json: &mut Map<String, Value>, key: &str) -> Self {
        let mut list: Vec<(String, Value)> = match json.remove(key) {
            Some(Value::Object(m)) => m.into_iter().collect(),
            Some(Value::Array(a)) => a.into_iter().enumerate().map(|(i, v)| (i.to_string(), v)).collect(),
            _ => Vec::new(),
        };
        if key == "buffers"
            && let Some(i) = list.iter().position(|(id, _)| id == BINARY)
        {
            let b = list.remove(i);
            list.insert(0, b);
        }
        let at = list.iter().enumerate().map(|(i, (id, _))| (id.clone(), i)).collect();
        Self { list, at }
    }

    /// The index `v` (an ID) names.
    fn index(&self, v: Option<&Value>) -> Option<usize> {
        self.at.get(v?.as_str()?).copied()
    }

    fn ids<'a>(&'a self, v: Option<&'a Value>) -> impl Iterator<Item = usize> + 'a {
        v.and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|id| self.index(Some(id)))
    }
}

/// The buffer `KHR_binary_glTF` names for the GLB's body.
const BINARY: &str = "binary_glTF";

/// A new 2.0 object: `name` (or the ID), and the 1.0 object's extras, plus `source` when
/// there's anything 2.0 has no place for.
fn object(id: &str, v: &Value, source: Map<String, Value>) -> Map<String, Value> {
    let mut o = Map::new();
    o.insert("name".into(), v.get("name").cloned().unwrap_or_else(|| json!(id)));
    let mut extras = v.get("extras").and_then(Value::as_object).cloned().unwrap_or_default();
    if !source.is_empty() {
        extras.insert("source".into(), Value::Object(source));
    }
    if !extras.is_empty() {
        o.insert("extras".into(), Value::Object(extras));
    }
    o
}

/// Copies `keys` that `from` has into `to`.
fn copy(from: &Value, to: &mut Map<String, Value>, keys: &[&str]) {
    for k in keys {
        if let Some(v) = from.get(*k) {
            to.insert((*k).into(), v.clone());
        }
    }
}

fn components(kind: &str) -> usize {
    match kind {
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" | "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => 1,
    }
}

fn component_size(ctype: u64) -> usize {
    match ctype {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        _ => 4,
    }
}

/// A 1.0 attribute semantic's 2.0 name.
fn semantic(s: &str, kind: &str) -> String {
    let (base, n) = match s.rsplit_once('_') {
        Some((b, n)) if n.parse::<u32>().is_ok() => (b, n),
        _ => (s, "0"),
    };
    match (base, n) {
        ("POSITION" | "NORMAL", "0") => base.into(),
        ("TANGENT", "0") if kind == "VEC4" => base.into(),
        ("TEXCOORD", _) => format!("TEXCOORD_{n}"),
        ("COLOR", _) => format!("COLOR_{n}"),
        ("JOINT" | "JOINTS", _) => format!("JOINTS_{n}"),
        ("WEIGHT" | "WEIGHTS", _) => format!("WEIGHTS_{n}"),
        _ => format!("_{s}"),
    }
}

/// `json`, glTF 1.0, as glTF 2.0, and what's left to do once its buffers are read.
#[allow(clippy::too_many_lines, reason = "one pass over each kind of object, in order")]
pub(crate) fn upgrade(mut v1: Map<String, Value>) -> Result<(Map<String, Value>, Fixups)> {
    let buffers = Kind::take(&mut v1, "buffers");
    let views = Kind::take(&mut v1, "bufferViews");
    let accessors = Kind::take(&mut v1, "accessors");
    let meshes = Kind::take(&mut v1, "meshes");
    let nodes = Kind::take(&mut v1, "nodes");
    let skins = Kind::take(&mut v1, "skins");
    let cameras = Kind::take(&mut v1, "cameras");
    let images = Kind::take(&mut v1, "images");
    let samplers = Kind::take(&mut v1, "samplers");
    let textures = Kind::take(&mut v1, "textures");
    let materials = Kind::take(&mut v1, "materials");
    let techniques = Kind::take(&mut v1, "techniques");
    let animations = Kind::take(&mut v1, "animations");
    let scenes = Kind::take(&mut v1, "scenes");
    let mut fix = Fixups::default();
    let mut used: Vec<&str> = Vec::new();

    // What 2.0 drops, kept whole.
    let mut source = Map::new();
    let mut asset = match v1.remove("asset") {
        Some(Value::Object(a)) => a,
        _ => Map::new(),
    };
    let version = match asset.remove("version") {
        Some(Value::String(v)) => v,
        Some(Value::Number(n)) => n.to_string(),
        _ => "1.0".to_owned(),
    };
    let text = |k: &str| asset.get(k).and_then(Value::as_str).map(str::to_owned);
    SourceInfo {
        format: Some("glTF".into()),
        version: Some(version),
        // 1.0 fixes the meter, +Y up and right-handedness, as 2.0 does.
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Declared(Axis::PosY),
        handedness: Fact::Declared(Handedness::Right),
        tool: text("generator"),
        copyright: text("copyright"),
        ..SourceInfo::default()
    }
    .write(&mut source);
    for k in ["premultipliedAlpha", "profile"] {
        if let Some(v) = asset.remove(k) {
            source.insert(k.into(), v);
        }
    }
    if !techniques.list.is_empty() {
        source.insert(
            "techniques".into(),
            Value::Object(techniques.list.iter().cloned().collect()),
        );
    }
    for k in [
        "programs",
        "shaders",
        "extensionsUsed",
        "extensionsRequired",
        "glExtensionsUsed",
    ] {
        if let Some(v) = v1.remove(k) {
            source.insert(k.into(), v);
        }
    }
    let mut root_ext = match v1.remove("extensions") {
        Some(Value::Object(e)) => e,
        _ => Map::new(),
    };

    let mut out = Map::new();

    out.insert(
        "buffers".into(),
        buffers
            .list
            .iter()
            .map(|(id, b)| {
                let mut o = object(id, b, Map::new());
                copy(b, &mut o, &["byteLength"]);
                if id != BINARY {
                    copy(b, &mut o, &["uri"]);
                }
                Value::Object(o)
            })
            .collect(),
    );

    let mut out_views: Vec<Value> = views
        .list
        .iter()
        .map(|(id, v)| {
            let mut o = object(id, v, Map::new());
            o.insert("buffer".into(), json!(buffers.index(v.get("buffer"))));
            copy(v, &mut o, &["byteOffset", "byteLength", "target"]);
            Value::Object(o)
        })
        .collect();
    for (i, (id, v)) in views.list.iter().enumerate() {
        ensure!(
            out_views[i]["buffer"].is_u64(),
            Corrupt,
            "glTF 1.0 bufferView {id:?} names no buffer there is ({})",
            v.get("buffer").unwrap_or(&Value::Null)
        );
    }

    // Accessors; a stride moves to a copy of its view.
    let mut strided: HashMap<(usize, u64), usize> = HashMap::new();
    let mut out_accessors = Vec::with_capacity(accessors.list.len());
    for (id, a) in &accessors.list {
        let mut o = object(id, a, Map::new());
        copy(
            a,
            &mut o,
            &["byteOffset", "componentType", "count", "type", "max", "min"],
        );
        if let Some(view) = views.index(a.get("bufferView")) {
            let kind = a.get("type").and_then(Value::as_str).unwrap_or("SCALAR");
            let ctype = a.get("componentType").and_then(Value::as_u64).unwrap_or(5126);
            let packed = (components(kind) * component_size(ctype)) as u64;
            let stride = a.get("byteStride").and_then(Value::as_u64).unwrap_or(0);
            let view = if stride == 0 || stride == packed {
                view
            } else {
                *strided.entry((view, stride)).or_insert_with(|| {
                    let mut copy = out_views[view].clone();
                    copy["byteStride"] = json!(stride);
                    out_views.push(copy);
                    out_views.len() - 1
                })
            };
            o.insert("bufferView".into(), json!(view));
        }
        if let Some(q) = a.get("extensions").and_then(|e| e.get("WEB3D_quantized_attributes")) {
            let n = components(a.get("type").and_then(Value::as_str).unwrap_or("SCALAR"));
            let m: Vec<f64> = q
                .get("decodeMatrix")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_f64)
                .collect();
            ensure!(
                n <= 4 && m.len() == (n + 1) * (n + 1),
                Corrupt,
                "glTF 1.0 accessor {id:?}: its WEB3D_quantized_attributes decodeMatrix has {} numbers, not {}",
                m.len(),
                (n + 1) * (n + 1)
            );
            fix.quantized.push((out_accessors.len(), m));
        }
        out_accessors.push(Value::Object(o));
    }

    // Meshes; a node's several become one.
    let mut out_meshes: Vec<Value> = Vec::with_capacity(meshes.list.len());
    for (id, m) in &meshes.list {
        let mut o = object(id, m, Map::new());
        let prims: Vec<Value> = m
            .get("primitives")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|p| primitive(p, &accessors, &materials, &mut out_accessors, &mut fix))
            .collect();
        o.insert("primitives".into(), Value::Array(prims));
        out_meshes.push(Value::Object(o));
    }
    let mut merged: HashMap<Vec<usize>, usize> = HashMap::new();

    // Each node's parent, to find a skin's joints under its skeletons.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nodes.list.len()];
    for (i, (_, n)) in nodes.list.iter().enumerate() {
        children[i] = nodes.ids(n.get("children")).collect();
    }
    let joint_name = |i: usize| nodes.list[i].1.get("jointName").and_then(Value::as_str);
    let find_joint = |roots: &[usize], name: &str| -> Option<usize> {
        let mut stack: Vec<usize> = roots.to_vec();
        if stack.is_empty() {
            stack = (0..nodes.list.len()).rev().collect();
        }
        // Each node is looked at once: a node listed among its own descendants would go round forever.
        let mut seen = vec![false; nodes.list.len()];
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            if joint_name(i) == Some(name) {
                return Some(i);
            }
            stack.extend(children[i].iter().rev());
        }
        None
    };

    let mut out_skins: Vec<Value> = Vec::new();
    let mut skin_of: HashMap<(usize, Vec<usize>), Option<usize>> = HashMap::new();
    let mut out_nodes = Vec::with_capacity(nodes.list.len());
    let mut lights_used = false;
    for (i, (id, n)) in nodes.list.iter().enumerate() {
        let mut src = Map::new();
        for k in ["jointName", "skeletons", "skin", "instanceSkin"] {
            if let Some(v) = n.get(k) {
                src.insert(k.into(), v.clone());
            }
        }
        let mut o = object(id, n, src);
        if !children[i].is_empty() {
            o.insert("children".into(), json!(children[i]));
        }
        copy(n, &mut o, &["matrix", "translation", "rotation", "scale"]);
        if let Some(c) = cameras.index(n.get("camera")) {
            o.insert("camera".into(), json!(c));
        }
        // An early draft's instanceSkin holds the skin, skeletons and meshes.
        let skinned = n.get("instanceSkin").unwrap_or(n);
        let list: Vec<usize> = meshes.ids(skinned.get("meshes").or_else(|| n.get("meshes"))).collect();
        let mesh = match list.as_slice() {
            [] => None,
            [m] => Some(*m),
            _ => Some(*merged.entry(list.clone()).or_insert_with(|| {
                let prims: Vec<Value> = list
                    .iter()
                    .flat_map(|&m| out_meshes[m]["primitives"].as_array().cloned().unwrap_or_default())
                    .collect();
                let name = list
                    .iter()
                    .filter_map(|&m| out_meshes[m]["name"].as_str())
                    .collect::<Vec<_>>()
                    .join("+");
                out_meshes.push(json!({ "name": name, "primitives": prims }));
                out_meshes.len() - 1
            })),
        };
        if let Some(m) = mesh {
            o.insert("mesh".into(), json!(m));
            if let Some(s) = skins.index(skinned.get("skin")) {
                let roots: Vec<usize> = nodes.ids(skinned.get("skeletons")).collect();
                let skin = *skin_of.entry((s, roots.clone())).or_insert_with(|| {
                    let skin = &skins.list[s].1;
                    let joints: Option<Vec<usize>> = skin
                        .get("jointNames")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .map(|name| find_joint(&roots, name.as_str()?))
                        .collect();
                    let joints = joints.filter(|j| !j.is_empty())?;
                    let mut src = Map::new();
                    copy(skin, &mut src, &["bindShapeMatrix", "jointNames"]);
                    let mut o = object(&skins.list[s].0, skin, src);
                    if let Some(ibm) = accessors.index(skin.get("inverseBindMatrices")) {
                        o.insert("inverseBindMatrices".into(), json!(ibm));
                    }
                    if let [root, ..] = roots.as_slice() {
                        o.insert("skeleton".into(), json!(root));
                    }
                    o.insert("joints".into(), json!(joints));
                    out_skins.push(Value::Object(o));
                    let at = out_skins.len() - 1;
                    if let Some(m) = matrix(skin.get("bindShapeMatrix"))
                        && differs(&m, &IDENTITY)
                    {
                        fix.bind_shapes.push((at, m));
                    }
                    Some(at)
                });
                if let Some(skin) = skin {
                    o.insert("skin".into(), json!(skin));
                }
            }
        }
        if let Some(l) = n
            .get("extensions")
            .and_then(|e| e.get("KHR_materials_common"))
            .and_then(|c| c.get("light"))
            .and_then(Value::as_str)
        {
            o.insert("extensions".into(), json!({ "KHR_lights_punctual": { "light": l } }));
            lights_used = true;
        }
        out_nodes.push(Value::Object(o));
    }

    // KHR_materials_common lights, by ID, as punctual lights.
    if let Some(common) = root_ext.remove("KHR_materials_common") {
        source.insert("KHR_materials_common".into(), common.clone());
        let lights: Vec<(String, Value)> = common
            .get("lights")
            .and_then(Value::as_object)
            .map(|l| l.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let mut at = HashMap::new();
        let mut out_lights = Vec::new();
        for (id, l) in &lights {
            if let Some(light) = light(id, l) {
                at.insert(id.clone(), out_lights.len());
                out_lights.push(light);
            }
        }
        for n in &mut out_nodes {
            let idx = n
                .pointer("/extensions/KHR_lights_punctual/light")
                .and_then(Value::as_str)
                .map(|id| at.get(id).copied());
            match idx {
                Some(Some(i)) => n["extensions"]["KHR_lights_punctual"]["light"] = json!(i),
                Some(None) => {
                    n.as_object_mut().expect("a node").remove("extensions");
                }
                None => {}
            }
        }
        if !out_lights.is_empty() {
            used.push("KHR_lights_punctual");
            out.insert(
                "extensions".into(),
                json!({ "KHR_lights_punctual": { "lights": out_lights } }),
            );
        }
    } else if lights_used {
        for n in &mut out_nodes {
            n.as_object_mut().expect("a node").remove("extensions");
        }
    }
    // CESIUM_RTC: positions are relative to a center, which every root node moves to.
    if let Some(center) = root_ext.get("CESIUM_RTC").and_then(|e| e.get("center")) {
        let c: Vec<f64> = center
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_f64)
            .collect();
        ensure!(
            c.len() == 3 && c.iter().all(|x| x.is_finite()),
            Corrupt,
            "glTF 1.0 CESIUM_RTC center isn't three numbers"
        );
        let mut root = vec![true; out_nodes.len()];
        for kids in &children {
            for &k in kids {
                root[k] = false;
            }
        }
        for (n, _) in out_nodes.iter_mut().zip(root).filter(|(_, root)| *root) {
            let at = |v: &Value, i: usize| v.get(i).and_then(Value::as_f64).unwrap_or(0.0);
            if let Some(m) = n
                .get_mut("matrix")
                .filter(|m| m.as_array().is_some_and(|m| m.len() == 16))
            {
                for (k, x) in c.iter().enumerate() {
                    m[12 + k] = json!(at(m, 12 + k) + x);
                }
            } else {
                let t = n.get("translation").cloned().unwrap_or(Value::Null);
                n["translation"] = json!([at(&t, 0) + c[0], at(&t, 1) + c[1], at(&t, 2) + c[2]]);
            }
        }
    }
    if !root_ext.is_empty() {
        source.insert("extensions".into(), Value::Object(root_ext));
    }

    out.insert(
        "cameras".into(),
        cameras
            .list
            .iter()
            .map(|(id, c)| {
                let mut o = object(id, c, Map::new());
                copy(c, &mut o, &["type", "perspective", "orthographic"]);
                Value::Object(o)
            })
            .collect(),
    );

    out.insert(
        "images".into(),
        images
            .list
            .iter()
            .map(|(id, img)| {
                let mut o = object(id, img, Map::new());
                if let Some(b) = img.get("extensions").and_then(|e| e.get("KHR_binary_glTF")) {
                    if let Some(v) = views.index(b.get("bufferView")) {
                        o.insert("bufferView".into(), json!(v));
                    }
                    copy(b, &mut o, &["mimeType"]);
                } else {
                    copy(img, &mut o, &["uri"]);
                }
                Value::Object(o)
            })
            .collect(),
    );

    out.insert(
        "samplers".into(),
        samplers
            .list
            .iter()
            .map(|(id, s)| {
                let mut o = object(id, s, Map::new());
                copy(s, &mut o, &["magFilter", "minFilter", "wrapS", "wrapT"]);
                Value::Object(o)
            })
            .collect(),
    );

    out.insert(
        "textures".into(),
        textures
            .list
            .iter()
            .map(|(id, t)| {
                let mut src = Map::new();
                copy(t, &mut src, &["format", "internalFormat", "target", "type"]);
                let mut o = object(id, t, src);
                if let Some(s) = samplers.index(t.get("sampler")) {
                    o.insert("sampler".into(), json!(s));
                }
                if let Some(s) = images.index(t.get("source")) {
                    o.insert("source".into(), json!(s));
                }
                Value::Object(o)
            })
            .collect(),
    );

    let mut unlit = false;
    out.insert(
        "materials".into(),
        materials
            .list
            .iter()
            .map(|(id, m)| {
                let (o, is_unlit) = material(id, m, &techniques, &textures);
                unlit |= is_unlit;
                o
            })
            .collect(),
    );
    if unlit {
        used.push("KHR_materials_unlit");
    }

    let mut out_animations = Vec::new();
    for (id, a) in &animations.list {
        let mut src = Map::new();
        copy(a, &mut src, &["parameters"]);
        let mut o = object(id, a, src);
        let params = a.get("parameters");
        let param = |name: Option<&Value>| accessors.index(params?.get(name?.as_str()?));
        let sampler_ids: Vec<(String, Value)> = match a.get("samplers") {
            Some(Value::Object(s)) => s.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => Vec::new(),
        };
        let mut at = HashMap::new();
        let mut out_samplers = Vec::new();
        for (sid, s) in &sampler_ids {
            if let (Some(input), Some(output)) = (param(s.get("input")), param(s.get("output"))) {
                at.insert(sid.as_str(), out_samplers.len());
                let interpolation = s.get("interpolation").cloned().unwrap_or_else(|| json!("LINEAR"));
                out_samplers.push(json!({ "input": input, "output": output, "interpolation": interpolation }));
            }
        }
        let channels: Vec<Value> = a
            .get("channels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|c| {
                let sampler = at.get(c.get("sampler")?.as_str()?)?;
                let target = c.get("target")?;
                let node = nodes.index(target.get("id").or_else(|| target.get("node")))?;
                let path = target.get("path")?.as_str()?;
                matches!(path, "translation" | "rotation" | "scale")
                    .then(|| json!({ "sampler": sampler, "target": { "node": node, "path": path } }))
            })
            .collect();
        if channels.is_empty() {
            continue;
        }
        o.insert("samplers".into(), Value::Array(out_samplers));
        o.insert("channels".into(), Value::Array(channels));
        out_animations.push(Value::Object(o));
    }

    out.insert(
        "scenes".into(),
        scenes
            .list
            .iter()
            .map(|(id, s)| {
                let mut o = object(id, s, Map::new());
                o.insert("nodes".into(), json!(nodes.ids(s.get("nodes")).collect::<Vec<_>>()));
                Value::Object(o)
            })
            .collect(),
    );
    if let Some(s) = scenes.index(v1.get("scene")) {
        out.insert("scene".into(), json!(s));
    }

    out.insert("bufferViews".into(), Value::Array(out_views));
    out.insert("accessors".into(), Value::Array(out_accessors));
    out.insert("meshes".into(), Value::Array(out_meshes));
    out.insert("nodes".into(), Value::Array(out_nodes));
    out.insert("skins".into(), Value::Array(out_skins));
    out.insert("animations".into(), Value::Array(out_animations));
    out.retain(|_, v| !v.as_array().is_some_and(Vec::is_empty));
    if !used.is_empty() {
        out.insert("extensionsUsed".into(), json!(used));
    }
    if let Some(e) = v1.remove("extras") {
        out.insert("extras".into(), e);
    }
    asset.insert("version".into(), json!("2.0"));
    let mut extras = asset
        .remove("extras")
        .and_then(|e| e.as_object().cloned())
        .unwrap_or_default();
    extras.insert("source".into(), Value::Object(source));
    asset.insert("extras".into(), Value::Object(extras));
    out.insert("asset".into(), Value::Object(asset));
    Ok((out, fix))
}

/// A 1.0 primitive as 2.0's, noting the accessors that need their data changed.
fn primitive(p: &Value, accessors: &Kind, materials: &Kind, out_accessors: &mut [Value], fix: &mut Fixups) -> Value {
    let mut o = Map::new();
    let mut attrs = Map::new();
    for (name, id) in p.get("attributes").and_then(Value::as_object).into_iter().flatten() {
        let Some(a) = accessors.index(Some(id)) else { continue };
        let acc = &mut out_accessors[a];
        let kind = acc.get("type").and_then(Value::as_str).unwrap_or("SCALAR").to_owned();
        let ctype = acc.get("componentType").and_then(Value::as_u64).unwrap_or(5126);
        let s = semantic(name, &kind);
        if s.starts_with("JOINTS_") && ctype == 5126 && !fix.joints.contains(&a) {
            fix.joints.push(a);
        }
        // Quantized values are decoded by their matrix, as the integers they are.
        if s.starts_with("COLOR_") && ctype != 5126 && !fix.quantized.iter().any(|(q, _)| *q == a) {
            acc["normalized"] = json!(true);
        }
        if s == "POSITION" && (acc.get("min").is_none() || acc.get("max").is_none()) && !fix.bounds.contains(&a) {
            fix.bounds.push(a);
        }
        attrs.insert(s, json!(a));
    }
    o.insert("attributes".into(), Value::Object(attrs));
    if let Some(i) = accessors.index(p.get("indices")) {
        o.insert("indices".into(), json!(i));
    }
    if let Some(m) = materials.index(p.get("material")) {
        o.insert("material".into(), json!(m));
    }
    if let Some(m) = p.get("mode").or_else(|| p.get("primitive")) {
        o.insert("mode".into(), m.clone());
    }
    Value::Object(o)
}

/// Whether `a` and `b` differ anywhere by more than rounding.
fn differs(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6)
}

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn matrix(v: Option<&Value>) -> Option<[f32; 16]> {
    let a = v?.as_array()?;
    let mut m = [0.0; 16];
    for (o, x) in m.iter_mut().zip(a) {
        *o = f32_of(x.as_f64()?);
    }
    (a.len() == 16).then_some(m)
}

/// A material's 2.0 form, and whether it's unlit.
fn material(id: &str, m: &Value, techniques: &Kind, textures: &Kind) -> (Value, bool) {
    let common = m.get("extensions").and_then(|e| e.get("KHR_materials_common"));
    let technique = techniques.index(m.get("technique")).map(|t| &techniques.list[t].1);
    // The values: KHR_materials_common's, or the technique's defaults under the material's.
    let mut values = Map::new();
    let kind;
    let mut double_sided = false;
    let mut blend = false;
    if let Some(c) = common {
        kind = c.get("technique").and_then(Value::as_str).unwrap_or("BLINN").to_owned();
        values.extend(c.get("values").and_then(Value::as_object).cloned().unwrap_or_default());
        double_sided = c.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
        blend = c.get("transparent").and_then(Value::as_bool).unwrap_or(false);
    } else {
        kind = "TECHNIQUE".to_owned();
        if let Some(t) = technique {
            for (name, p) in t.get("parameters").and_then(Value::as_object).into_iter().flatten() {
                if let Some(v) = p.get("value") {
                    values.insert(name.clone(), v.clone());
                }
            }
            let enabled = t.pointer("/states/enable").and_then(Value::as_array);
            if let Some(enabled) = enabled {
                double_sided = !enabled.iter().any(|e| e.as_u64() == Some(2884));
                blend = enabled.iter().any(|e| e.as_u64() == Some(3042));
            }
        }
        values.extend(m.get("values").and_then(Value::as_object).cloned().unwrap_or_default());
    }

    let mut src = Map::new();
    copy(m, &mut src, &["technique", "values"]);
    if let Some(c) = common {
        src.insert("KHR_materials_common".into(), c.clone());
    }
    let mut o = object(id, m, src);
    let color = |v: &Value| -> Option<[f32; 4]> {
        let a = v.as_array()?;
        let mut c = [0.0, 0.0, 0.0, 1.0];
        for (i, x) in a.iter().take(4).enumerate() {
            let x = f32_of(x.as_f64()?);
            c[i] = if i < 3 {
                srgb_to_linear(x.clamp(0.0, 1.0))
            } else {
                x.clamp(0.0, 1.0)
            };
        }
        (a.len() >= 3).then_some(c)
    };
    let texture = |v: &Value| textures.index(Some(v)).map(|t| json!({ "index": t }));

    let unlit = kind == "CONSTANT";
    let mut pbr = Map::new();
    let mut base = [1.0, 1.0, 1.0, 1.0];
    let main = if unlit {
        values.get("emission").or_else(|| values.get("diffuse"))
    } else {
        values.get("diffuse")
    };
    if let Some(d) = main {
        if let Some(c) = color(d) {
            base = c;
        } else if let Some(t) = texture(d) {
            pbr.insert("baseColorTexture".into(), t);
        }
    }
    if let Some(t) = values.get("transparency").and_then(Value::as_f64) {
        base[3] *= f32_of(t).clamp(0.0, 1.0);
    }
    if differs(&base, &[1.0; 4]) {
        pbr.insert("baseColorFactor".into(), json!(base));
    }
    pbr.insert("metallicFactor".into(), json!(0.0));
    let shininess = values
        .get("shininess")
        .and_then(|s| s.as_f64().or_else(|| s.get(0)?.as_f64()));
    let roughness = match (kind.as_str(), shininess) {
        ("LAMBERT", _) | (_, None) => 1.0,
        (_, Some(n)) => (1.0 - f32_of(n.max(0.0)).sqrt() / 10.0).clamp(0.0, 1.0),
    };
    pbr.insert("roughnessFactor".into(), json!(roughness));
    o.insert("pbrMetallicRoughness".into(), Value::Object(pbr));
    if unlit {
        o.insert("extensions".into(), json!({ "KHR_materials_unlit": {} }));
    } else if let Some(e) = values.get("emission") {
        if let Some(c) = color(e) {
            if c[..3] != [0.0; 3] {
                o.insert("emissiveFactor".into(), json!(c[..3]));
            }
        } else if let Some(t) = texture(e) {
            o.insert("emissiveTexture".into(), t);
            o.insert("emissiveFactor".into(), json!([1.0, 1.0, 1.0]));
        }
    }
    if blend || base[3] < 1.0 {
        o.insert("alphaMode".into(), json!("BLEND"));
    }
    if double_sided {
        o.insert("doubleSided".into(), json!(true));
    }
    (Value::Object(o), unlit)
}

/// A `KHR_materials_common` light as a punctual one; ambient light has none.
fn light(id: &str, l: &Value) -> Option<Value> {
    let kind = l.get("type")?.as_str()?;
    let body = l.get(kind);
    let color = body
        .and_then(|b| b.get("color"))
        .and_then(Value::as_array)
        .filter(|c| c.len() >= 3)
        .map_or(json!([1.0, 1.0, 1.0]), |c| json!(c[..3]));
    let mut o = json!({ "name": l.get("name").cloned().unwrap_or_else(|| json!(id)), "color": color });
    match kind {
        "directional" => o["type"] = json!("directional"),
        "point" => o["type"] = json!("point"),
        "spot" => {
            o["type"] = json!("spot");
            // The falloff angle is the cone's whole width.
            let falloff = body.and_then(|b| b.get("falloffAngle")).and_then(Value::as_f64);
            let outer = falloff.map_or(std::f64::consts::FRAC_PI_4, |a| {
                (a / 2.0).clamp(1e-3, std::f64::consts::FRAC_PI_2)
            });
            o["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": outer });
        }
        _ => return None,
    }
    if let Some(b) = body.and_then(Value::as_object) {
        let mut extras = b.clone();
        extras.remove("color");
        if !extras.is_empty() {
            o["extras"] = json!({ "source": extras });
        }
    }
    Some(o)
}

/// Makes the changes the upgrade left for when the buffers are read.
pub(crate) fn finish(doc: &mut Document, fix: &Fixups) -> Result<()> {
    for (a, m) in &fix.quantized {
        // The integers as they are, then the matrix: its last column is the offset.
        let (n, raw) = doc.read_floats(*a)?;
        let rows = n + 1;
        ensure!(
            m.len() == rows * rows,
            Corrupt,
            "glTF 1.0 accessor {a}: its decode matrix doesn't fit its type"
        );
        let mut min = vec![f32::INFINITY; n];
        let mut max = vec![f32::NEG_INFINITY; n];
        let mut bytes = Vec::with_capacity(raw.len() * 4);
        for v in raw.chunks_exact(n) {
            for r in 0..n {
                let x: f64 = (0..n).map(|c| m[c * rows + r] * f64::from(v[c])).sum::<f64>() + m[n * rows + r];
                let x = f32_of(x);
                min[r] = min[r].min(x);
                max[r] = max[r].max(x);
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        let view = push(doc, bytes);
        let acc = doc.json_mut()["accessors"][*a].as_object_mut().expect("an accessor");
        acc.insert("bufferView".into(), json!(view));
        acc.insert("byteOffset".into(), json!(0));
        acc.insert("componentType".into(), json!(5126));
        acc.remove("normalized");
        if min.iter().all(|x| x.is_finite()) {
            acc.insert("min".into(), json!(min));
            acc.insert("max".into(), json!(max));
        } else {
            acc.remove("min");
            acc.remove("max");
        }
    }
    for &a in &fix.joints {
        let (floats, n) = floats(doc, a)?;
        let joints: Vec<u16> = floats
            .iter()
            .map(|&j| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "rounded and clamped"
                )]
                let j = j.round().clamp(0.0, f32::from(u16::MAX)) as u16;
                j
            })
            .collect();
        let bytes: Vec<u8> = joints.iter().flat_map(|j| j.to_le_bytes()).collect();
        let view = push(doc, bytes);
        let acc = doc.json_mut()["accessors"][a].as_object_mut().expect("an accessor");
        acc.insert("bufferView".into(), json!(view));
        acc.insert("byteOffset".into(), json!(0));
        acc.insert("componentType".into(), json!(5123));
        acc.insert("count".into(), json!(n));
        acc.remove("min");
        acc.remove("max");
    }
    for &a in &fix.bounds {
        let (p, _) = floats(doc, a)?;
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in p.chunks_exact(3) {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        if min[0].is_finite() {
            let acc = &mut doc.json_mut()["accessors"][a];
            acc["min"] = json!(min);
            acc["max"] = json!(max);
        }
    }
    for &(s, bsm) in &fix.bind_shapes {
        let skin = &doc.base["skins"][s];
        let joints = skin["joints"].as_array().map_or(0, Vec::len);
        let ibms: Vec<f32> = match skin.get("inverseBindMatrices").and_then(Value::as_u64) {
            Some(a) => floats(doc, usize_of(a))?.0,
            None => IDENTITY.repeat(joints),
        };
        let mut baked = Vec::with_capacity(ibms.len() * 4);
        for ibm in ibms.chunks_exact(16) {
            for c in 0..4 {
                for r in 0..4 {
                    let x: f32 = (0..4).map(|k| ibm[k * 4 + r] * bsm[c * 4 + k]).sum();
                    baked.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        let count = baked.len() / 64;
        let view = push(doc, baked);
        let accessors = doc
            .json_mut()
            .entry("accessors")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("glTF accessors is a list");
        accessors.push(json!({ "bufferView": view, "componentType": 5126, "count": count, "type": "MAT4" }));
        let a = accessors.len() - 1;
        doc.base_mut()["skins"][s]["inverseBindMatrices"] = json!(a);
    }
    Ok(())
}

/// A new buffer and a view of it.
fn push(doc: &mut Document, bytes: Vec<u8>) -> usize {
    let len = bytes.len();
    let b = doc.push_buffer(bytes);
    doc.push_view(json!({ "buffer": b, "byteLength": len }))
}

/// Float accessor `a`'s components, and its count.
fn floats(doc: &Document, a: usize) -> Result<(Vec<f32>, usize)> {
    let acc = &doc.json()["accessors"][a];
    let get = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    ensure!(
        get(acc, "componentType") == 5126,
        Corrupt,
        "glTF 1.0 accessor {a} isn't float"
    );
    let count = usize_of(get(acc, "count"));
    let n = components(acc.get("type").and_then(Value::as_str).unwrap_or("SCALAR"));
    // 1.0 requires a view: without one, an accessor's count isn't backed by any data.
    let Some(v) = acc.get("bufferView").and_then(Value::as_u64) else {
        ensure!(count == 0, Corrupt, "glTF 1.0 accessor {a} has no bufferView");
        return Ok((Vec::new(), 0));
    };
    let view = &doc.json()["bufferViews"][usize_of(v)];
    let stride = match usize_of(get(view, "byteStride")) {
        0 => n * 4,
        s => s,
    };
    let start = usize_of(get(view, "byteOffset").saturating_add(get(acc, "byteOffset")));
    let buf = doc.buffer(usize_of(get(view, "buffer")))?;
    // The last element must be in the buffer before room is made for them all.
    let end = count.checked_sub(1).map_or(Some(start), |last| {
        last.checked_mul(stride)?.checked_add(start)?.checked_add(n * 4)
    });
    ensure!(
        end.is_some_and(|e| e <= buf.len()),
        Truncated,
        "glTF 1.0 accessor {a} runs past the end of its buffer"
    );
    let mut out = Vec::with_capacity(count * n);
    for i in 0..count {
        for k in 0..n {
            let at = start + i * stride + k * 4;
            let b = buf.get(at..at + 4).with_context(ErrorKind::Truncated, || {
                format!("glTF 1.0 accessor {a} runs past the end of its buffer")
            })?;
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
    }
    Ok((out, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantics_take_2_0_names() {
        for (a, b) in [
            ("POSITION", "POSITION"),
            ("NORMAL", "NORMAL"),
            ("TEXCOORD", "TEXCOORD_0"),
            ("TEXCOORD_1", "TEXCOORD_1"),
            ("COLOR", "COLOR_0"),
            ("JOINT", "JOINTS_0"),
            ("WEIGHT", "WEIGHTS_0"),
            ("TANGENT", "_TANGENT"),
            ("BATCHID", "_BATCHID"),
        ] {
            assert_eq!(semantic(a, "VEC3"), b);
        }
    }

    /// A glTF 1.0 file of one triangle whose positions have no bounds, so they're read for them, with `accessor`'s
    /// fields over the position accessor's.
    fn triangle(accessor: &Value) -> Result<Document> {
        use base64::Engine as _;
        let floats: Vec<u8> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&floats)
        );
        let mut position =
            json!({ "bufferView": "v", "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3" });
        for (k, v) in accessor.as_object().unwrap() {
            if v.is_null() {
                position.as_object_mut().unwrap().remove(k);
            } else {
                position[k] = v.clone();
            }
        }
        let gltf = json!({
            "asset": { "version": "1.0" },
            "buffers": { "b": { "byteLength": 36, "uri": uri } },
            "bufferViews": { "v": { "buffer": "b", "byteOffset": 0, "byteLength": 36 } },
            "accessors": { "p": position },
            "meshes": { "m": { "primitives": [{ "attributes": { "POSITION": "p" }, "mode": 4 }] } },
            "nodes": { "n": { "meshes": ["m"] } },
            "scenes": { "s": { "nodes": ["n"] } },
            "scene": "s"
        });
        load(&gltf)
    }

    #[test]
    fn a_node_that_is_its_own_child_ends_the_search_for_a_joint() {
        use base64::Engine as _;
        let floats: Vec<u8> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&floats)
        );
        let file = |children: Value| {
            json!({
                "asset": { "version": "1.0" },
                "buffers": { "b": { "byteLength": 36, "uri": uri } },
                "bufferViews": { "v": { "buffer": "b", "byteOffset": 0, "byteLength": 36 } },
                "accessors": { "p": { "bufferView": "v", "byteOffset": 0, "componentType": 5126, "count": 3,
                    "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0] } },
                "meshes": { "m": { "primitives": [{ "attributes": { "POSITION": "p" }, "mode": 4 }] } },
                "skins": { "sk": { "jointNames": ["root", "missing"] } },
                "nodes": {
                    "n": { "meshes": ["m"], "skin": "sk", "skeletons": ["r"] },
                    "r": { "jointName": "root", "children": children },
                    "c": { "jointName": "leaf", "children": ["r"] }
                },
                "scenes": { "s": { "nodes": ["n", "r"] } },
                "scene": "s"
            })
        };
        // The joint `missing` is under no node, so the search runs through every node: it must end, with an error
        // or without the skin.
        for children in [json!(["r"]), json!(["c"])] {
            if let Ok(doc) = load(&file(children)) {
                let _ = doc.flatten(crate::document::Pose::Rest);
            }
        }
    }

    fn load(gltf: &Value) -> Result<Document> {
        let bytes = serde_json::to_vec(gltf).unwrap();
        crate::formats::load_bytes(
            &bytes,
            "x.gltf",
            "gltf",
            &crate::assets::Memory::new(),
            crate::document::Keep::ALL,
        )
    }

    /// A glTF 1.0 file of one triangle with texture coordinates, both unsigned shorts over their whole range
    /// with the decode matrices given.
    fn quantized(position: &Value, uv: &Value) -> Result<Document> {
        use base64::Engine as _;
        let shorts: [u16; 15] = [0, 0, 0, 65535, 0, 0, 0, 65535, 32768, 0, 0, 65535, 0, 0, 65535];
        let data: Vec<u8> = shorts.iter().flat_map(|s| s.to_le_bytes()).collect();
        let uri = format!(
            "data:application/octet-stream;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&data)
        );
        let accessor = |offset: usize, kind: &str, matrix: &Value| {
            json!({ "bufferView": "v", "byteOffset": offset, "byteStride": 0, "componentType": 5123, "count": 3,
                "type": kind, "min": [0, 0, 0], "max": [65535, 65535, 65535],
                "extensions": { "WEB3D_quantized_attributes": {
                    "decodeMatrix": matrix, "decodedMin": [-1, -1, -1], "decodedMax": [1, 1, 1] } } })
        };
        load(&json!({
            "asset": { "version": "1.0" },
            "extensionsUsed": ["WEB3D_quantized_attributes"],
            "buffers": { "b": { "byteLength": 30, "uri": uri } },
            "bufferViews": { "v": { "buffer": "b", "byteOffset": 0, "byteLength": 30 } },
            "accessors": { "p": accessor(0, "VEC3", position), "t": accessor(18, "VEC2", uv) },
            "meshes": { "m": { "primitives": [{ "attributes": { "POSITION": "p", "TEXCOORD_0": "t" }, "mode": 4 }] } },
            "nodes": { "n": { "meshes": ["m"] } },
            "scenes": { "s": { "nodes": ["n"] } },
            "scene": "s"
        }))
    }

    #[test]
    fn quantized_attributes_are_decoded_by_their_matrix() {
        // The 16-bit range onto [-1, 1], as the extension's overview works it out: a scale down the diagonal
        // and the minimum in the last column. Texture coordinates take a 3x3 matrix, onto [0, 2] and [1, 0].
        let s = 2.0 / 65535.0;
        let position = json!([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, -1, -1, -1, 1]);
        let uv = json!([s, 0, 0, 0, -1.0 / 65535.0, 0, 0, 1, 1]);
        let doc = quantized(&position, &uv).unwrap();
        let p = &doc.json()["accessors"][0];
        assert_eq!(p["componentType"], 5126);
        // The middle of the range is half a step past the middle of what it decodes to.
        let half = 1.0 / 65535.0;
        assert_eq!(
            (&p["min"], &p["max"]),
            (&json!([-1.0, -1.0, -1.0]), &json!([1.0, 1.0, half]))
        );
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let prim = &mesh.prims[0];
        assert_eq!(prim.positions, [[-1., -1., -1.], [1., -1., -1.], [-1., 1., half]]);
        assert_eq!(prim.uvs, [[0., 1.], [2., 1.], [0., 0.]]);
        // A matrix of the wrong size for the accessor's type.
        let err = quantized(&uv, &uv).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Corrupt);
        assert!(err.to_string().contains("decodeMatrix"), "{err}");
    }

    #[test]
    fn positions_relative_to_a_center_move_to_it() {
        // The center of the extension's example: a point on the equator, in meters from the Earth's center.
        let mut gltf = json!({
            "asset": { "version": "1.0" },
            "extensionsUsed": ["CESIUM_RTC"],
            "extensions": { "CESIUM_RTC": { "center": [6_378_137.0, 0.0, 0.0] } },
            "meshes": {},
            "nodes": {
                "a": { "children": ["b"], "translation": [1, 2, 3] },
                "b": { "translation": [1, 0, 0] },
                "c": { "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1] },
                "d": {},
            },
            "scenes": { "s": { "nodes": ["a", "c", "d"] } },
            "scene": "s"
        });
        let (out, _) = upgrade(gltf.as_object().unwrap().clone()).unwrap();
        let nodes = &out["nodes"];
        assert_eq!(nodes[0]["translation"], json!([6_378_138.0, 2.0, 3.0]));
        assert_eq!(nodes[1]["translation"], json!([1, 0, 0]), "a child follows its parent");
        assert_eq!(nodes[2]["matrix"][12], 6_378_137.0);
        assert_eq!(nodes[2]["matrix"][13], 5.0);
        assert_eq!(nodes[3]["translation"], json!([6_378_137.0, 0.0, 0.0]));
        assert_eq!(
            out["asset"]["extras"]["source"]["extensions"]["CESIUM_RTC"]["center"][0],
            6_378_137.0
        );
        gltf["extensions"]["CESIUM_RTC"]["center"] = json!([0, 0]);
        assert!(upgrade(gltf.as_object().unwrap().clone()).is_err());
    }

    #[test]
    fn huge_counts_are_errors_not_allocations() {
        let doc = triangle(&json!({})).unwrap();
        assert_eq!(doc.json()["accessors"][0]["max"], json!([1.0, 1.0, 0.0]));
        // Counts no buffer holds, with and without a view, and offsets past any buffer.
        for bad in [
            json!({ "count": 1u64 << 40 }),
            json!({ "count": u64::MAX }),
            json!({ "count": 1u64 << 40, "bufferView": null }),
            json!({ "byteOffset": u64::MAX }),
            json!({ "count": 4 }),
        ] {
            assert!(triangle(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn versions() {
        let v = |j: Value| is_v1(j.as_object().unwrap());
        assert!(v(json!({ "asset": { "version": "1.0" } })));
        assert!(v(json!({ "asset": { "version": "1.0.1" } })));
        assert!(v(json!({ "asset": { "version": 1 } })));
        assert!(v(json!({ "meshes": {} })));
        assert!(!v(json!({ "asset": { "version": "2.0" } })));
        assert!(!v(json!({ "meshes": [] })));
    }

    #[test]
    fn the_source_is_1_0s() {
        let v1 = json!({
            "asset": { "version": "1.0.1", "generator": "Exporter 2", "copyright": "Ann" },
            "meshes": {},
        });
        let (out, _) = upgrade(v1.as_object().unwrap().clone()).unwrap();
        assert_eq!(out["asset"]["version"], "2.0");
        assert_eq!(out["asset"]["generator"], "Exporter 2");
        let info = SourceInfo::from_json(out["asset"]["extras"]["source"].as_object().unwrap());
        assert_eq!(info.format.as_deref(), Some("glTF"));
        assert_eq!(info.version.as_deref(), Some("1.0.1"));
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.unit.as_deref(), Some("meter"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.tool.as_deref(), Some("Exporter 2"));
        assert_eq!(info.copyright.as_deref(), Some("Ann"));
        assert_eq!(info.converted, crate::source::Converted::default());
    }
}
