//! Irrlicht scenes: a tree of nodes, built in Irrlicht's coordinates and mirrored whole.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_2, TAU};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::Ctx;
use super::graft::{self, Placed};
use super::material::{Attrs, IrrMaterial};
use super::mesh::{self, Mesh};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::formats::scene::{self, Animation, Channel, Interp, M4, Node, Path as Key, Primitive};
use crate::formats::xml::{El, Xml};
use crate::mesh::srgb_to_linear;

/// The most keys a channel has.
const MAX_KEYS: usize = 4096;
/// Keys per turn of a circle.
const CIRCLE_KEYS: f64 = 72.0;
/// Keys per segment of a spline.
const SPLINE_KEYS: f64 = 16.0;
/// The most a rotation turns between keys, in degrees.
const ROTATION_STEP: f64 = 10.0;

/// A camera, aimed once the nodes' places are known.
struct Camera {
    node: usize,
    name: String,
    target: [f32; 3],
    up: [f32; 3],
    json: Value,
}

/// A file another reader loads, grafted under a node.
struct Graft {
    node: usize,
    path: PathBuf,
    name: String,
    materials: Option<Vec<usize>>,
}

struct World<'a> {
    ctx: Ctx<'a>,
    assets: &'a dyn Assets,
    dir: PathBuf,
    limit: usize,
    keep: Keep,
    /// Irrlicht mesh files read, by path (`None` when unreadable).
    files: HashMap<PathBuf, Option<Mesh>>,
    /// Meshes made from them, by path and the materials that replace theirs.
    made: HashMap<(PathBuf, String), usize>,
    cameras: Vec<Camera>,
    lights: Vec<(usize, Value)>,
    grafts: Vec<Graft>,
    /// Every file the scene names, in order.
    named: Vec<String>,
    missing: Vec<String>,
    unsupported_nodes: BTreeMap<String, usize>,
    unsupported_animators: BTreeMap<String, usize>,
    hidden: usize,
    dropped: usize,
    nodes: usize,
}

pub(super) fn load<'a>(text: &str, dir: &Path, limit: usize, assets: &dyn Assets, keep: Keep) -> Result<Document<'a>> {
    let xml = Xml::parse(text)?;
    let root = xml.root();
    ensure!(
        root.is("irr_scene"),
        UnknownFormat,
        "not an Irrlicht scene: the root is <{}>",
        root.name()
    );
    let mut w = World {
        ctx: Ctx::new(assets, keep, dir),
        assets,
        dir: dir.to_path_buf(),
        limit,
        keep,
        files: HashMap::new(),
        made: HashMap::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        grafts: Vec::new(),
        named: Vec::new(),
        missing: Vec::new(),
        unsupported_nodes: BTreeMap::new(),
        unsupported_animators: BTreeMap::new(),
        hidden: 0,
        dropped: 0,
        nodes: 0,
    };
    let ambient = root
        .children()
        .find(|c| c.is("attributes"))
        .map(|a| Attrs::of(a).colorf("AmbientLight", [0.0; 4]));
    for n in root.children().filter(|c| c.is("node")) {
        w.node(n, None)?;
    }
    ensure!(w.nodes > 0 || w.hidden > 0, Empty, "Irrlicht scene has no nodes");
    w.aim_cameras();

    let World {
        mut ctx,
        cameras,
        lights,
        grafts,
        named,
        mut missing,
        unsupported_nodes,
        unsupported_animators,
        hidden,
        dropped,
        nodes,
        ..
    } = w;
    let scene = &mut ctx.scene;
    scene.info.format = Some("Irrlicht scene".into());
    if let Some(a) = ambient.filter(|a| a[..3].iter().any(|&c| c != 0.0)) {
        scene.source.insert("ambientLight".into(), json!(a));
    }
    if !named.is_empty() {
        scene.source.insert("files".into(), json!(named));
    }
    let mut doc = ctx.finish()?;
    let json = &mut doc.json;

    if keep.cameras && !cameras.is_empty() {
        let mut list = Vec::new();
        for c in cameras {
            if let Some(n) = json.get_mut("nodes").and_then(|n| n.get_mut(c.node)) {
                n["camera"] = json!(list.len());
                list.push(c.json);
            }
        }
        json.insert("cameras".into(), json!(list));
    }
    if keep.lights && !lights.is_empty() {
        let mut list = Vec::new();
        for (node, light) in lights {
            if let Some(n) = json.get_mut("nodes").and_then(|n| n.get_mut(node)) {
                n["extensions"] = json!({ "KHR_lights_punctual": { "light": list.len() } });
                list.push(light);
            }
        }
        let ext = json.entry("extensions").or_insert_with(|| json!({}));
        ext["KHR_lights_punctual"] = json!({ "lights": list });
        let used = json.entry("extensionsUsed").or_insert_with(|| json!([]));
        if let Some(u) = used.as_array_mut() {
            u.push(json!("KHR_lights_punctual"));
        }
    }

    // Files other readers load, each read once while it can be shared.
    let mut placed: HashMap<(PathBuf, Option<Vec<usize>>), Placed> = HashMap::new();
    let sub = Keep {
        max_triangles: limit,
        ..keep
    };
    for g in grafts {
        let key = (g.path, g.materials);
        if let Some(p) = placed.get(&key).filter(|p| p.shareable) {
            graft::repeat(&mut doc, p, g.node);
            continue;
        }
        let ext = key
            .0
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let path = key.0.to_string_lossy().into_owned();
        let Some(bytes) = assets.read(&key.0) else {
            missing.push(g.name);
            continue;
        };
        match crate::formats::read(crate::formats::Input::memory(&bytes, &path), &ext, assets, sub) {
            Ok(d) => {
                let p = graft::graft(&mut doc, d.into_owned(), g.node, key.1.as_deref());
                placed.insert(key, p);
            }
            Err(e) => missing.push(format!("{}: {e:#}", g.name)),
        }
        ensure!(
            doc.counts().triangles <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
    }

    let triangles = doc.counts().triangles;
    let facts = &mut doc.facts;
    facts.insert("triangles".into(), json!(triangles));
    facts.insert("nodes".into(), json!(nodes));
    if hidden > 0 {
        facts.insert("hidden_nodes".into(), json!(hidden));
    }
    if dropped > 0 {
        facts.insert("dropped_triangles".into(), json!(dropped));
    }
    if !unsupported_nodes.is_empty() {
        facts.insert("unsupported_nodes".into(), json!(unsupported_nodes));
    }
    if !unsupported_animators.is_empty() {
        facts.insert("unsupported_animators".into(), json!(unsupported_animators));
    }
    if !missing.is_empty() {
        facts.insert("missing_files".into(), json!(missing));
    }
    Ok(doc)
}

/// Irrlicht's rotation in degrees (about X, then Y, then Z) as a quaternion.
fn euler(d: [f64; 3]) -> [f32; 4] {
    let [x, y, z] = d.map(f64::to_radians);
    scene::normalize(scene::quat_mul(
        scene::about(2, z),
        scene::quat_mul(scene::about(1, y), scene::about(0, x)),
    ))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = dot(a, a).sqrt();
    (l > 1e-12 && l.is_finite()).then(|| a.map(|c| c / l))
}

fn wide(v: [f32; 3]) -> [f64; 3] {
    v.map(f64::from)
}

fn narrow(v: [f64; 3]) -> [f32; 3] {
    v.map(crate::num::f32_of)
}

impl World<'_> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "segment counts, clamped"
    )]
    fn node(&mut self, el: El<'_, '_>, parent: Option<usize>) -> Result<()> {
        let kind = el.attr("type").map(|t| t.to_ascii_lowercase()).unwrap_or_default();
        let attrs = el
            .children()
            .find(|c| c.is("attributes"))
            .map_or_else(|| Attrs::of(el.child("\u{0}").unwrap_or(el)), Attrs::of);
        if !attrs.bool("Visible", true) {
            self.hidden += 1;
            return Ok(());
        }
        self.nodes += 1;
        let name = attrs
            .get("Name")
            .filter(|n| !n.is_empty())
            .map_or_else(|| format!("{kind} {}", self.ctx.scene.nodes.len()), str::to_owned);
        let mut node = Node::named(name.clone(), parent);
        node.translation = attrs.vec3("Position", [0.0; 3]);
        let rest = wide(attrs.vec3("Rotation", [0.0; 3]));
        node.rotation = euler(rest);
        node.scale = attrs.vec3("Scale", [1.0; 3]);
        let index = self.ctx.scene.add(node);
        let materials: Vec<IrrMaterial> = el
            .children()
            .find(|c| c.is("materials"))
            .map(|m| {
                m.children()
                    .filter(|a| a.is("attributes"))
                    .map(|a| IrrMaterial::parse(&Attrs::of(a)))
                    .collect()
            })
            .unwrap_or_default();

        match kind.as_str() {
            "mesh" | "octtree" | "animatedmesh" => self.mesh_node(index, &name, &attrs, materials)?,
            "cube" => {
                let s = f64::from(attrs.f32("Size", 10.0));
                self.shape(index, &name, vec![cube(s)], &materials)?;
            }
            "sphere" => {
                let r = f64::from(attrs.f32("Radius", 5.0));
                let both = attrs.f32("PolyCount", 16.0);
                let px = attrs.f32("PolyCountX", both);
                let py = attrs.f32("PolyCountY", both);
                let count = |v: f32| if v.is_finite() { (v as usize).clamp(3, 256) } else { 16 };
                self.shape(index, &name, vec![sphere(r, count(px), count(py))], &materials)?;
            }
            "skybox" => self.shape(index, &name, skybox(), &materials)?,
            "billboard" => {
                let w = f64::from(attrs.f32("Width", 10.0));
                let h = f64::from(attrs.f32("Height", 10.0));
                let top = attrs.color("Shade_Top", 0xFFFF_FFFF);
                let down = attrs.color("Shade_Down", 0xFFFF_FFFF);
                self.billboard(index, &name, w, h, top, down, &materials)?;
                if self.keep.extras {
                    self.ctx.scene.nodes[index].extras = Some(json!({ "billboard": true }));
                }
            }
            "light" => self.light(index, &name, &attrs),
            "camera" => self.camera(index, &name, &attrs),
            "empty" | "dummytransformation" => {}
            other => *self.unsupported_nodes.entry(other.to_owned()).or_default() += 1,
        }

        if let Some(a) = el.children().find(|c| c.is("animators")) {
            self.animate(index, &name, rest, kind == "camera", a);
        }
        for c in el.children().filter(|c| c.is("node")) {
            self.node(c, Some(index))?;
        }
        Ok(())
    }

    /// A node's generated primitives, the node's materials on them in order.
    fn shape(&mut self, node: usize, name: &str, prims: Vec<Primitive>, materials: &[IrrMaterial]) -> Result<()> {
        let mut mesh = scene::SceneMesh {
            name: name.to_owned(),
            ..Default::default()
        };
        let dir = self.dir.clone();
        for (i, mut p) in prims.into_iter().enumerate() {
            let m = materials.get(i).or(materials.first()).cloned().unwrap_or_default();
            p.material = Some(self.ctx.material(&m, Some(0xFFFF_FFFF), 1, &dir));
            mesh.prims.push(p);
        }
        self.add_mesh(node, mesh)
    }

    fn add_mesh(&mut self, node: usize, mesh: scene::SceneMesh) -> Result<()> {
        let scene = &mut self.ctx.scene;
        scene.meshes.push(mesh);
        scene.nodes[node].mesh = Some(scene.meshes.len() - 1);
        let limit = self.limit;
        ensure!(
            scene.triangles() <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation, reason = "glTF's coordinates are f32")]
    #[allow(clippy::too_many_arguments, reason = "a billboard's settings")]
    fn billboard(
        &mut self,
        node: usize,
        name: &str,
        w: f64,
        h: f64,
        top: u32,
        down: u32,
        materials: &[IrrMaterial],
    ) -> Result<()> {
        let (x, y) = ((w * 0.5) as f32, (h * 0.5) as f32);
        let mut p = Primitive {
            positions: vec![[-x, -y, 0.0], [x, -y, 0.0], [x, y, 0.0], [-x, y, 0.0]],
            normals: vec![[0.0, 0.0, -1.0]; 4],
            uvs: vec![vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
            ..Primitive::default()
        };
        quad(&mut p, 0);
        let m = materials.first().cloned().unwrap_or_default();
        let dir = self.dir.clone();
        let uniform = (top == down).then_some(top);
        if uniform.is_none() {
            let alpha = m.vertex_alpha();
            let c = |v: u32| {
                let mut l = super::material::linear(v);
                if !alpha {
                    l[3] = 1.0;
                }
                l
            };
            p.colors = vec![vec![c(down), c(down), c(top), c(top)]];
        }
        p.material = Some(self.ctx.material(&m, uniform, 1, &dir));
        let mesh = scene::SceneMesh {
            name: name.to_owned(),
            prims: vec![p],
            ..Default::default()
        };
        self.add_mesh(node, mesh)
    }

    /// A node that shows a mesh file.
    fn mesh_node(&mut self, node: usize, name: &str, attrs: &Attrs, materials: Vec<IrrMaterial>) -> Result<()> {
        let Some(file) = attrs.get("Mesh").map(str::trim).filter(|m| !m.is_empty()) else {
            return Ok(());
        };
        if !self.named.iter().any(|n| n == file) {
            self.named.push(file.to_owned());
        }
        let wanted = file.rsplit(['/', '\\']).next().unwrap_or(file).to_ascii_lowercase();
        let found = self.assets.find(file, &self.dir).filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().to_ascii_lowercase() == wanted)
        });
        let Some(path) = found else {
            self.missing.push(file.to_owned());
            return Ok(());
        };
        let overrides = (!attrs.bool("ReadOnlyMaterials", false) && !materials.is_empty()).then_some(materials);
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let sniffed = if ext == "irrmesh" || ext == "irr" || crate::formats::EXTENSIONS.contains(&ext.as_str()) {
            ext.clone()
        } else {
            String::new()
        };
        match sniffed.as_str() {
            "irrmesh" | "xml" => {
                let assets = self.assets;
                let limit = self.limit;
                let entry = self.files.entry(path.clone()).or_insert_with(|| {
                    let bytes = assets.read(&path)?;
                    mesh::parse(&crate::formats::xml::text(&bytes), limit).ok()
                });
                let Some(m) = entry.as_ref() else {
                    if sniffed == "xml" {
                        // Not an Irrlicht mesh: some other XML format.
                        self.grafts.push(Graft {
                            node,
                            path,
                            name: file.to_owned(),
                            materials: None,
                        });
                    } else {
                        self.missing.push(file.to_owned());
                    }
                    return Ok(());
                };
                let key = (path.clone(), format!("{overrides:?}"));
                let index = if let Some(&i) = self.made.get(&key) {
                    i
                } else {
                    let mdir = path.parent().unwrap_or(Path::new("")).to_path_buf();
                    let dir = if overrides.is_some() { self.dir.clone() } else { mdir };
                    let stem = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let built = self.ctx.mesh(&stem, &m.buffers, overrides.as_deref(), &dir);
                    self.dropped += m.dropped;
                    let scene = &mut self.ctx.scene;
                    scene.meshes.push(built);
                    self.made.insert(key, scene.meshes.len() - 1);
                    scene.meshes.len() - 1
                };
                self.ctx.scene.nodes[node].mesh = Some(index);
                let limit = self.limit;
                ensure!(
                    self.ctx.scene.triangles() <= limit,
                    LimitExceeded,
                    "too many triangles to convert (more than {limit})"
                );
            }
            "irr" => *self.unsupported_nodes.entry("scene file".into()).or_default() += 1,
            "" => self.missing.push(file.to_owned()),
            _ => {
                let materials = overrides.map(|o| {
                    let dir = self.dir.clone();
                    o.iter()
                        .map(|m| self.ctx.material(m, Some(0xFFFF_FFFF), 1, &dir))
                        .collect()
                });
                // Irrlicht reads OBJ's X negated where glTF readers keep it, so the file is seen turned half
                // around Y from the mirror that converts the rest.
                let at = if ext == "obj" {
                    let mut holder = Node::named(format!("{name} (OBJ)"), Some(node));
                    holder.rotation = [0.0, 1.0, 0.0, 0.0];
                    self.ctx.scene.add(holder)
                } else {
                    node
                };
                self.grafts.push(Graft {
                    node: at,
                    path,
                    name: file.to_owned(),
                    materials,
                });
            }
        }
        Ok(())
    }

    fn light(&mut self, node: usize, name: &str, a: &Attrs) {
        let kind = match a.get("LightType").map(|t| t.trim().to_ascii_lowercase()).as_deref() {
            Some("spot") => "spot",
            Some("directional") => "directional",
            _ => "point",
        };
        let c = a.colorf("DiffuseColor", [1.0; 4]);
        let mut light = json!({
            "name": name,
            "type": kind,
            "color": [srgb_to_linear(c[0].clamp(0.0, 1.0)), srgb_to_linear(c[1].clamp(0.0, 1.0)), srgb_to_linear(c[2].clamp(0.0, 1.0))],
            "intensity": 1.0,
        });
        let radius = a.f32("Radius", 100.0);
        if kind != "directional" && radius > 0.0 {
            light["range"] = json!(radius);
        }
        if kind == "spot" {
            let outer = f64::from(a.f32("OuterCone", 45.0)).to_radians().clamp(1e-3, FRAC_PI_2);
            let inner = f64::from(a.f32("InnerCone", 0.0))
                .to_radians()
                .clamp(0.0, outer * 0.999);
            light["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
        }
        if self.keep.extras {
            light["extras"] = json!({
                "ambient": a.colorf("AmbientColor", [0.0, 0.0, 0.0, 1.0]),
                "specular": a.colorf("SpecularColor", [1.0; 4]),
                "attenuation": a.vec3("Attenuation", [1.0, 0.0, 0.0]),
                "castShadows": a.bool("CastShadows", true),
            });
        }
        self.lights.push((node, light));
    }

    fn camera(&mut self, node: usize, name: &str, a: &Attrs) {
        let znear = a.f32("ZNear", 1.0).max(1e-4);
        let zfar = a.f32("ZFar", 3000.0);
        let yfov = a.f32("Fovy", std::f32::consts::PI / 2.5);
        let aspect = a.f32("Aspect", 4.0 / 3.0);
        let mut p =
            json!({ "yfov": if yfov > 0.0 && yfov < std::f32::consts::PI { yfov } else { 1.0 }, "znear": znear });
        if zfar > znear {
            p["zfar"] = json!(zfar);
        }
        if aspect > 0.0 {
            p["aspectRatio"] = json!(aspect);
        }
        self.cameras.push(Camera {
            node,
            name: name.to_owned(),
            target: a.vec3("Target", [0.0, 0.0, 100.0]),
            up: a.vec3("UpVector", [0.0, 1.0, 0.0]),
            json: json!({ "name": name, "type": "perspective", "perspective": p }),
        });
    }

    /// Puts each camera on a child of its node, aimed from where the node is at rest at the camera's target
    /// (Irrlicht aims a camera by its target and up vector, both in world space, not by its node's rotation).
    fn aim_cameras(&mut self) {
        if self.cameras.is_empty() {
            return;
        }
        let worlds = self.ctx.scene.world();
        for c in &mut self.cameras {
            let world: M4 = worlds[c.node];
            let pos = [world[12], world[13], world[14]];
            let z = unit(sub(wide(c.target), pos)).unwrap_or([0.0, 0.0, 1.0]);
            let x = unit(cross(wide(c.up), z))
                .or_else(|| unit(cross([0.0, 1.0, 0.0], z)))
                .or_else(|| unit(cross([1.0, 0.0, 0.0], z)))
                .unwrap_or([1.0, 0.0, 0.0]);
            let y = cross(z, x);
            let mut aim = scene::IDENTITY;
            for (col, v) in [x, y, z].iter().enumerate() {
                aim[col * 4..col * 4 + 3].copy_from_slice(v);
            }
            aim[12..15].copy_from_slice(&pos);
            let local = scene::mul(&scene::invert(&world), &aim);
            let (t, r, s) = scene::decompose(&local);
            let mut n = Node::named(c.name.clone(), Some(c.node));
            n.translation = t;
            n.rotation = r;
            n.scale = s;
            c.node = self.ctx.scene.add(n);
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "key counts clamped to MAX_KEYS, and glTF's times are f32"
    )]
    fn animate(&mut self, node: usize, name: &str, rest: [f64; 3], camera: bool, el: El<'_, '_>) {
        let mut rate = [0.0f64; 3];
        let mut mover: Option<Mover> = None;
        for a in el.children().filter(|c| c.is("attributes")) {
            let a = Attrs::of(a);
            let kind = a.get("Type").unwrap_or_default().trim().to_owned();
            match kind.to_ascii_lowercase().as_str() {
                // A camera's rotation doesn't aim it.
                "rotation" if camera => {}
                "rotation" => {
                    let r = wide(a.vec3("Rotation", [0.0; 3]));
                    for (s, r) in rate.iter_mut().zip(r) {
                        *s += r;
                    }
                }
                "flycircle" => mover = Some(Mover::circle(&a)),
                "flystraight" => mover = Some(Mover::straight(&a)),
                "followspline" => mover = Some(Mover::spline(&a)),
                _ => *self.unsupported_animators.entry(kind).or_default() += 1,
            }
        }
        let mut periods: Vec<f64> = rate
            .iter()
            .filter(|r| r.abs() > 1e-9)
            .map(|r| 3600.0 / r.abs())
            .collect();
        let mut once = 0.0f64;
        if let Some(m) = &mover {
            match m.span() {
                Span::Repeats(p) => periods.push(p),
                Span::Once(d) => once = d,
                Span::Still => {
                    self.ctx.scene.nodes[node].translation = narrow(m.at(0.0));
                }
            }
        }
        let total = common(&periods).max(once);
        if total <= 0.0 || !total.is_finite() {
            return;
        }
        let mut channels = Vec::new();
        if rate.iter().any(|r| r.abs() > 1e-9) {
            let turn: f64 = rate.iter().map(|r| r.abs()).sum::<f64>() / 10.0;
            let steps = ((total * turn / ROTATION_STEP).ceil() as usize).clamp(1, MAX_KEYS - 1);
            let mut times = Vec::with_capacity(steps + 1);
            let mut values = Vec::with_capacity((steps + 1) * 4);
            for i in 0..=steps {
                let t = total * i as f64 / steps as f64;
                times.push((t / 1000.0) as f32);
                let e = [0, 1, 2].map(|k| rest[k] + rate[k] * t / 10.0);
                values.extend_from_slice(&euler(e));
            }
            channels.push(Channel {
                node,
                path: Key::Rotation,
                times,
                values,
                interp: Interp::Linear,
            });
        }
        if let Some(m) = &mover {
            let keys = m.keys(total);
            if !keys.is_empty() {
                channels.push(Channel {
                    node,
                    path: Key::Translation,
                    times: keys.iter().map(|(t, _)| (t / 1000.0) as f32).collect(),
                    values: keys.iter().flat_map(|(_, p)| narrow(*p)).collect(),
                    interp: Interp::Linear,
                });
            }
        }
        if !channels.is_empty() {
            self.ctx.scene.animations.push(Animation {
                name: name.to_owned(),
                channels,
                extras: None,
            });
        }
    }
}

/// How long until channels repeating every `periods` (milliseconds) all repeat together: the shortest multiple
/// of the longest, up to 8 of them, that each fits within 1%, or else the longest.
fn common(periods: &[f64]) -> f64 {
    let Some(longest) = periods.iter().copied().filter(|p| p.is_finite()).reduce(f64::max) else {
        return 0.0;
    };
    (1..=8)
        .map(|n| f64::from(n) * longest)
        .find(|t| {
            periods.iter().all(|p| {
                let k = t / p;
                (k - k.round()).abs() < 0.01 * k.max(1.0)
            })
        })
        .unwrap_or(longest)
}

/// How a position animator runs.
enum Span {
    /// Over and over, each round this many milliseconds.
    Repeats(f64),
    /// Once, for this many milliseconds, then holds.
    Once(f64),
    /// Not at all: it holds one place.
    Still,
}

/// The animators that set a node's position.
enum Mover {
    Circle {
        center: [f64; 3],
        radius: f64,
        radius2: f64,
        speed: f64,
        u: [f64; 3],
        v: [f64; 3],
    },
    Straight {
        start: [f64; 3],
        end: [f64; 3],
        time: f64,
        looping: bool,
        ping_pong: bool,
    },
    Spline {
        points: Vec<[f64; 3]>,
        speed: f64,
        tightness: f64,
        looping: bool,
        ping_pong: bool,
    },
}

impl Mover {
    fn circle(a: &Attrs) -> Self {
        let dir = unit(wide(a.vec3("Direction", [0.0, 1.0, 0.0]))).unwrap_or([0.0, 1.0, 0.0]);
        let v = if dir[1] == 0.0 {
            cross([0.0, 50.0, 0.0], dir)
        } else {
            cross([50.0, 0.0, 0.0], dir)
        };
        let v = unit(v).unwrap_or([1.0, 0.0, 0.0]);
        let u = unit(cross(v, dir)).unwrap_or([0.0, 0.0, 1.0]);
        let radius = f64::from(a.f32("Radius", 100.0));
        let r2 = f64::from(a.f32("RadiusEllipsoid", 0.0));
        Self::Circle {
            center: wide(a.vec3("Center", [0.0; 3])),
            radius,
            radius2: if r2 == 0.0 { radius } else { r2 },
            speed: f64::from(a.f32("Speed", 0.001)),
            u,
            v,
        }
    }

    fn straight(a: &Attrs) -> Self {
        Self::Straight {
            start: wide(a.vec3("Start", [0.0; 3])),
            end: wide(a.vec3("End", [0.0; 3])),
            time: f64::from(a.f32("TimeForWay", 3000.0)),
            looping: a.bool("Loop", false),
            ping_pong: a.bool("PingPong", false),
        }
    }

    fn spline(a: &Attrs) -> Self {
        let mut numbered: Vec<(u32, [f64; 3])> = a
            .iter()
            .filter_map(|(_, n, v)| {
                let k = n.get(..5).filter(|p| p.eq_ignore_ascii_case("point"))?;
                let _ = k;
                let i = n[5..].parse::<u32>().ok()?;
                Some((i, wide(super::material::floats::<3>(v))))
            })
            .collect();
        numbered.sort_by_key(|(i, _)| *i);
        Self::Spline {
            points: numbered.into_iter().map(|(_, p)| p).collect(),
            speed: f64::from(a.f32("Speed", 1.0)),
            tightness: f64::from(a.f32("Tightness", 0.5)),
            looping: a.bool("Loop", true),
            ping_pong: a.bool("PingPong", false),
        }
    }

    #[allow(clippy::cast_precision_loss, reason = "a handful of points")]
    fn span(&self) -> Span {
        match *self {
            Self::Circle { speed, .. } if speed.abs() > 1e-12 => Span::Repeats(TAU / speed.abs()),
            Self::Straight {
                time,
                looping,
                ping_pong,
                ..
            } if time > 0.0 => {
                let round = if ping_pong { 2.0 * time } else { time };
                if looping {
                    Span::Repeats(round)
                } else {
                    Span::Once(round)
                }
            }
            Self::Spline {
                ref points,
                speed,
                looping,
                ping_pong,
                ..
            } if points.len() > 1 && speed > 1e-9 => {
                let n = points.len() as f64;
                let segment = 1000.0 / speed;
                match (looping, ping_pong) {
                    (true, false) => Span::Repeats(n * segment),
                    (true, true) => Span::Repeats(2.0 * (n - 1.0) * segment),
                    (false, _) => Span::Once((n - 1.0) * segment),
                }
            }
            _ => Span::Still,
        }
    }

    /// The position `t` milliseconds in, by Irrlicht's formulas.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "a handful of points, and indices clamped to them"
    )]
    fn at(&self, t: f64) -> [f64; 3] {
        match self {
            Self::Circle {
                center,
                radius,
                radius2,
                speed,
                u,
                v,
            } => {
                let (s, c) = (t * speed).sin_cos();
                [0, 1, 2].map(|k| center[k] + radius * c * u[k] + radius2 * s * v[k])
            }
            Self::Straight {
                start,
                end,
                time,
                looping,
                ping_pong,
            } => {
                let time = time.max(1e-9);
                let lerp = |f: f64| [0, 1, 2].map(|k| start[k] + (end[k] - start[k]) * f);
                if !looping && !ping_pong && t >= time {
                    return *end;
                }
                if !looping && *ping_pong && t >= 2.0 * time {
                    return *start;
                }
                let f = (t % time) / time;
                if *ping_pong && t % (2.0 * time) >= time {
                    lerp(1.0 - f)
                } else {
                    lerp(f)
                }
            }
            Self::Spline {
                points,
                speed,
                tightness,
                looping,
                ping_pong,
            } => {
                let size = points.len() as i64;
                match size {
                    0 => return [0.0; 3],
                    1 => return points[0],
                    _ => {}
                }
                let dt = t * speed * 0.001;
                let whole = dt.floor() as i64;
                if !looping && whole >= size - 1 {
                    return points[(size - 1) as usize];
                }
                let pong = *ping_pong && (whole / (size - 1)) % 2 != 0;
                let fract = dt - dt.floor();
                let u = if pong { 1.0 - fract } else { fract };
                let idx = if pong {
                    (size - 2) - whole % (size - 1)
                } else if *ping_pong {
                    whole % (size - 1)
                } else {
                    whole % size
                };
                let wrap = |i: i64| {
                    let i = if i < 0 {
                        size + i
                    } else if i >= size {
                        i - size
                    } else {
                        i
                    };
                    points[i.clamp(0, size - 1) as usize]
                };
                let (p0, p1, p2, p3) = (wrap(idx - 1), wrap(idx), wrap(idx + 1), wrap(idx + 2));
                let (u2, u3) = (u * u, u * u * u);
                let h1 = 2.0 * u3 - 3.0 * u2 + 1.0;
                let h2 = -2.0 * u3 + 3.0 * u2;
                let h3 = u3 - 2.0 * u2 + u;
                let h4 = u3 - u2;
                [0, 1, 2].map(|k| {
                    let t1 = (p2[k] - p0[k]) * tightness;
                    let t2 = (p3[k] - p1[k]) * tightness;
                    p1[k] * h1 + p2[k] * h2 + t1 * h3 + t2 * h4
                })
            }
        }
    }

    /// Keys over `total` milliseconds, or over the animator's one run when that's shorter.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "key counts, clamped"
    )]
    fn keys(&self, total: f64) -> Vec<(f64, [f64; 3])> {
        let span = match self.span() {
            Span::Repeats(_) => total,
            Span::Once(d) => d,
            Span::Still => return Vec::new(),
        };
        match self {
            Self::Circle { speed, .. } => {
                let turns = span * speed.abs() / TAU;
                self.sampled(span, (turns * CIRCLE_KEYS).ceil() as usize)
            }
            Self::Spline { speed, .. } => {
                let segments = span * speed * 0.001;
                self.sampled(span, (segments * SPLINE_KEYS).ceil() as usize)
            }
            Self::Straight {
                start,
                end,
                time,
                ping_pong,
                looping,
            } => {
                // Straight lines between turns: a key at each, and at a loop's jump back two.
                let rounds = ((span / time).round() as usize).clamp(1, MAX_KEYS / 2 - 1);
                let mut keys = vec![(0.0, *start)];
                for i in 1..=rounds {
                    let t = *time * i as f64;
                    if *ping_pong {
                        keys.push((t, if i % 2 == 1 { *end } else { *start }));
                    } else {
                        keys.push((t, *end));
                        if *looping && i < rounds {
                            keys.push((t, *start));
                        }
                    }
                }
                keys
            }
        }
    }

    #[allow(clippy::cast_precision_loss, reason = "key counts are small")]
    fn sampled(&self, span: f64, steps: usize) -> Vec<(f64, [f64; 3])> {
        let steps = steps.clamp(1, MAX_KEYS - 1);
        (0..=steps)
            .map(|i| {
                let t = span * i as f64 / steps as f64;
                (t, self.at(t))
            })
            .collect()
    }
}

/// Adds the two triangles of the quad whose four vertices start at `first`, wound to face their normals (the
/// cross product of the edges along the normal, as Irrlicht's own meshes have it, so the mirror into glTF turns
/// them the right way).
fn quad(p: &mut Primitive, first: u32) {
    for tri in [[0, 1, 2], [0, 2, 3]] {
        let [a, b, c] = tri.map(|i| first + i);
        let pos = |i: u32| wide(p.positions[i as usize]);
        let n = cross(sub(pos(b), pos(a)), sub(pos(c), pos(a)));
        let want = wide(p.normals[a as usize]);
        if dot(n, n) < 1e-18 {
            continue;
        }
        if dot(n, want) >= 0.0 {
            p.indices.extend_from_slice(&[a, b, c]);
        } else {
            p.indices.extend_from_slice(&[a, c, b]);
        }
    }
}

/// The four corners of a face at `n` (outward) with `right` and `up` across it, half-size `h`, each with its
/// normal and coordinates (u along `right`, v down `up`).
#[allow(
    clippy::cast_possible_truncation,
    reason = "a few vertices, and glTF's coordinates are f32"
)]
fn face(p: &mut Primitive, n: [f64; 3], right: [f64; 3], up: [f64; 3], h: f64, normal: [f64; 3]) {
    let first = p.positions.len() as u32;
    for (sr, su) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let v = [0, 1, 2].map(|k| (n[k] + right[k] * sr + up[k] * su) * h);
        p.positions.push(narrow(v));
        p.normals.push(narrow(normal));
        p.uvs[0].push([f64::midpoint(sr, 1.0) as f32, ((1.0 - su) * 0.5) as f32]);
    }
    quad(p, first);
}

/// The cube of edge `size` about its node, a face at a time with flat normals.
fn cube(size: f64) -> Primitive {
    let mut p = Primitive {
        uvs: vec![Vec::new()],
        ..Primitive::default()
    };
    let h = size * 0.5;
    // Each face seen from outside: in Irrlicht's left-handed space right = up × forward, forward = -n.
    for (n, up) in [
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
    ] {
        let right = cross(up, n.map(|c: f64| -c));
        face(&mut p, n, right, up, h, n);
    }
    p
}

/// A sphere of `radius`, `around` segments about Y and `down` from pole to pole.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "segment counts are at most 256, and glTF's coordinates are f32"
)]
fn sphere(radius: f64, around: usize, down: usize) -> Primitive {
    let mut p = Primitive {
        uvs: vec![Vec::new()],
        ..Primitive::default()
    };
    for j in 0..=down {
        let theta = std::f64::consts::PI * j as f64 / down as f64;
        for i in 0..=around {
            let phi = TAU * i as f64 / around as f64;
            let n = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
            p.positions.push(narrow(n.map(|c| c * radius)));
            p.normals.push(narrow(n));
            p.uvs[0].push([(i as f64 / around as f64) as f32, (j as f64 / down as f64) as f32]);
        }
    }
    let row = (around + 1) as u32;
    let mut quads = Primitive::default();
    for j in 0..down as u32 {
        for i in 0..around as u32 {
            let corners = [j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i];
            quads.positions = corners.iter().map(|&c| p.positions[c as usize]).collect();
            quads.normals = corners.iter().map(|&c| p.normals[c as usize]).collect();
            quads.indices.clear();
            quad(&mut quads, 0);
            p.indices.extend(quads.indices.iter().map(|&k| corners[k as usize]));
        }
    }
    p
}

/// Irrlicht's sky box: six quads of half-size 10 facing in, front, left, back, right, top and bottom, each its
/// own primitive for its own material.
fn skybox() -> Vec<Primitive> {
    let l = 10.0;
    // The face's direction from the center, and its up across it; the normal points in.
    [
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ]
    .into_iter()
    .map(|(n, up)| {
        let mut p = Primitive {
            uvs: vec![Vec::new()],
            ..Primitive::default()
        };
        // Seen from inside, looking along n: right = up × n.
        let right = cross(up, n);
        face(&mut p, n, right, up, l, n.map(|c: f64| -c));
        p
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_face_out() {
        for p in [cube(2.0), sphere(1.0, 8, 6)] {
            assert_ne!(p.indices, [] as [u32; 0]);
            for t in p.indices.chunks(3) {
                let v = |i: u32| wide(p.positions[i as usize]);
                let n = cross(sub(v(t[1]), v(t[0])), sub(v(t[2]), v(t[0])));
                let c = [0, 1, 2].map(|k| v(t[0])[k] + v(t[1])[k] + v(t[2])[k]);
                assert!(dot(n, c) > 0.0);
            }
        }
        assert_eq!(cube(2.0).indices.len(), 36);
        let sky = skybox();
        assert_eq!(sky.len(), 6);
        for p in &sky {
            let t = &p.indices[..3];
            let v = |i: u32| wide(p.positions[i as usize]);
            let n = cross(sub(v(t[1]), v(t[0])), sub(v(t[2]), v(t[0])));
            assert!(dot(n, v(t[0])) < 0.0, "a sky box faces in");
        }
    }

    #[test]
    fn animators_follow_irrlicht() {
        // Irrlicht's circle about +Y starts at U = normalize(V × D), V = normalize((50,0,0) × D) = (0,0,-1).
        let x =
            Xml::parse(r#"<a><vector3d name="Direction" value="0,1,0"/><float name="Radius" value="2"/></a>"#).unwrap();
        let c = Mover::circle(&Attrs::of(x.root()));
        let p = c.at(0.0);
        assert!((p[0] + 2.0).abs() < 1e-9 && p[2].abs() < 1e-9, "{p:?}");
        assert!(matches!(c.span(), Span::Repeats(t) if (t - TAU * 1000.0).abs() < 1e-2));
        // A spline through its points, a second a segment.
        let x = Xml::parse(
            r#"<a><vector3d name="Point1" value="0,0,0"/><vector3d name="Point2" value="10,0,0"/><vector3d name="Point3" value="10,10,0"/></a>"#,
        )
        .unwrap();
        let s = Mover::spline(&Attrs::of(x.root()));
        assert_eq!(s.at(1000.0), [10.0, 0.0, 0.0]);
        assert!(matches!(s.span(), Span::Repeats(t) if (t - 3000.0).abs() < 1e-6));
        assert!((common(&[12000.0, 6000.0]) - 12000.0).abs() < 1e-9);
        assert!((common(&[3000.0, 2000.0]) - 6000.0).abs() < 1e-9);
        // Irrlicht's rotation: about X, then Y, then Z.
        let q = euler([0.0, 90.0, 0.0]).map(f64::from);
        let v = scene::rotate(q, [1.0, 0.0, 0.0]);
        assert!((v[2] + 1.0).abs() < 1e-6, "{v:?}");
    }
}
