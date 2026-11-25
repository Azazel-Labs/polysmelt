//! XGL (`.xgl`, or `.xml` with a `<WORLD>` root) and zipped XGL (`.zgl`: the same XML, zlib-compressed).
//!
//! XGL is a scene of XML tags, written in upper case but read in any. A `<WORLD>` holds the lighting
//! (`<AMBIENT>`, `<DIRECTIONALLIGHT>` with its `<DIRECTION>`, `<DIFFUSE>` and `<SPECULAR>`), the background
//! (`<BACKCOLOR>`), materials (`<MAT>`), meshes (`<MESH>`) and objects (`<OBJECT>`), which nest. A mesh lists
//! points (`<P ID>`), normals (`<N ID>`) and texture coordinates (`<TC ID>`), then triangles (`<F>`, with
//! `<FV1>` to `<FV3>`), line segments (`<L>`, `<LV1>` and `<LV2>`) and points (`<P>` with a `<PV1>`), each
//! vertex referring to them by ID (`<PREF>`, `<NREF>`, `<TCREF>`) or giving its own, each primitive its material
//! (`<MATREF>` or a `<MAT>` of its own). A `<PATCH>` gathers primitives, a `<MATREF>` or `<MAT>` in it the
//! material of those that name none. A mesh with an ID is a definition, placed by each `<MESHREF>` to it (or,
//! when nothing refers to it, where it's written); one without is placed where it is.
//!
//! An object's `<TRANSFORM>` gives its `<FORWARD>` and `<UP>` directions, `<POSITION>` and uniform `<SCALE>`:
//! its local +Z points forward, +Y up and +X along up × forward. Materials take the diffuse color, `<ALPHA>`
//! (blending below one), the emissive color and the shininess (a Phong exponent up to 128) as roughness.
//!
//! XGL is right-handed with Y up, as glTF is, so nothing is converted; the source info has both assumed, and the
//! unit's length unknown. Textures (`<TEXTURE>`, `<TEXTURERGB>`, `<TEXTURERGBA>`), object references and line
//! and point styles are counted as unsupported tags.

use std::collections::{BTreeMap, HashMap};

use serde_json::{Value, json};

use super::scene::{self, M4, Node, Primitive, Scene, SceneMesh};
use super::xml::{El, Xml};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::mesh::{Material, srgb_to_linear};
use crate::source::{Axis, Fact, Handedness};

pub(crate) fn load(input: super::Input<'_>, ext: &str, _assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let raw;
    let bytes: &[u8] = if ext == "zgl" || zipped(&bytes) {
        let mut out = Vec::new();
        // A zlib header (two bytes) before the deflate stream; a stream without one is read as it is.
        let body = if zipped(&bytes) { &bytes[2..] } else { &bytes[..] };
        super::inflate::inflate(body, &mut out, MAX_INFLATED)?;
        raw = out;
        &raw
    } else {
        &bytes
    };
    let text = super::xml::text(bytes);
    read(&text, input.limit, keep)
}

/// The most a zipped file inflates to.
const MAX_INFLATED: usize = 1 << 31;

/// Whether `b` starts with a zlib header: deflate, and a check that holds.
fn zipped(b: &[u8]) -> bool {
    b.len() > 2 && b[0] & 0x0F == 8 && b[0] >> 4 <= 7 && (u16::from(b[0]) << 8 | u16::from(b[1])) % 31 == 0
}

/// A vertex: where it is, and its normal and coordinates when it has them.
#[derive(Clone, Copy, PartialEq)]
struct Vertex {
    p: [f32; 3],
    n: Option<[f32; 3]>,
    tc: Option<[f32; 2]>,
}

/// What primitives a mesh splits into: by material, by kind (3 triangles, 2 lines, 1 points), and by whether
/// the vertices have normals and coordinates.
type Key = (Option<usize>, u8, bool, bool);

/// A primitive being built, its vertices shared by their bits.
#[derive(Default)]
struct Builder {
    prim: Primitive,
    seen: HashMap<[u32; 8], u32>,
}

impl Builder {
    fn vertex(&mut self, v: Vertex) -> u32 {
        let n = v.n.unwrap_or_default();
        let tc = v.tc.unwrap_or_default();
        let bits = [v.p[0], v.p[1], v.p[2], n[0], n[1], n[2], tc[0], tc[1]].map(f32::to_bits);
        let prim = &mut self.prim;
        *self.seen.entry(bits).or_insert_with(|| {
            prim.positions.push(v.p);
            if v.n.is_some() {
                prim.normals.push(n);
            }
            if v.tc.is_some() {
                // XGL's coordinates start at the bottom left, glTF's at the top left.
                prim.uvs[0].push([tc[0], 1.0 - tc[1]]);
            }
            crate::num::u32_of(prim.positions.len() - 1)
        })
    }
}

struct Reader {
    scene: Scene,
    keep: Keep,
    limit: usize,
    /// Materials by ID.
    mats: HashMap<String, usize>,
    /// Mesh definitions by ID: the scene mesh, and whether anything placed it.
    defs: HashMap<String, (usize, bool)>,
    /// Where each mesh with an ID is written, to place it there when nothing refers to it.
    homes: Vec<(String, usize)>,
    default_material: Option<usize>,
    triangles: usize,
    dropped: usize,
    unsupported: BTreeMap<String, usize>,
}

fn read(text: &str, limit: usize, keep: Keep) -> Result<Document> {
    let xml = Xml::parse(text)?;
    let root = xml.root();
    ensure!(
        root.is("world"),
        UnknownFormat,
        "not XGL: the root is <{}>",
        root.name()
    );
    let mut r = Reader {
        scene: Scene::default(),
        keep,
        limit,
        mats: HashMap::new(),
        defs: HashMap::new(),
        homes: Vec::new(),
        default_material: None,
        triangles: 0,
        dropped: 0,
        unsupported: BTreeMap::new(),
    };
    let world = r.scene.add(Node::named("WORLD", None));
    // Materials anywhere outside meshes first, so references before definitions find them.
    r.materials(root);
    r.definitions(root)?;
    let mut lights = Vec::new();
    for c in root.children() {
        if c.is("lighting") {
            r.lighting(c, world, &mut lights);
        } else if c.is("background")
            && let Some(color) = c.children().find(|b| b.is("backcolor")).map(|b| vec3(&b.text()))
            && keep.extras
        {
            r.scene.source.insert("backgroundColor".into(), json!(color));
        }
    }
    r.object(root, world)?;
    // Definitions nothing placed, where they're written.
    for (id, node) in std::mem::take(&mut r.homes) {
        if let Some((mesh, false)) = r.defs.get(&id).copied() {
            r.place(node, mesh, &id);
        }
    }
    ensure!(
        r.scene.meshes.iter().any(|m| !m.prims.is_empty()),
        Empty,
        "no geometry in the XGL file"
    );
    let Reader {
        mut scene,
        dropped,
        unsupported,
        ..
    } = r;
    scene.info.format = Some("XGL".into());
    scene.info.up_axis = Fact::Assumed(Axis::PosY);
    scene.info.handedness = Fact::Assumed(Handedness::Right);
    if dropped > 0 {
        scene.facts.insert("dropped_primitives".into(), json!(dropped));
    }
    if !unsupported.is_empty() {
        scene.facts.insert("unsupported_tags".into(), json!(unsupported));
    }
    let mut doc = scene.finish(keep)?;
    // The triangles drawn: a definition placed twice counts twice.
    let drawn = doc.counts().triangles;
    doc.facts.insert("triangles".into(), json!(drawn));
    if keep.lights && !lights.is_empty() {
        let json = doc.base_mut();
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
    Ok(doc)
}

/// Three numbers, separated by commas (or spaces); missing ones are zero.
fn vec3(s: &str) -> [f32; 3] {
    let mut out = [0.0; 3];
    for (o, t) in out
        .iter_mut()
        .zip(s.split([',', ' ', '\t', '\n', '\r']).filter(|t| !t.is_empty()))
    {
        *o = super::stream::f32_from(t.as_bytes()).unwrap_or(0.0);
    }
    out
}

fn vec2(s: &str) -> [f32; 2] {
    let v = vec3(s);
    [v[0], v[1]]
}

fn num(e: &El<'_, '_>) -> f32 {
    super::stream::f32_from(e.text().trim().as_bytes()).unwrap_or(0.0)
}

/// An element's ID attribute, in any case.
fn id(e: &El<'_, '_>) -> Option<String> {
    e.attr_ci("id").map(|v| v.trim().to_owned())
}

fn color(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)))
}

impl Reader {
    /// Reads every `<MAT>` that isn't inside a primitive, by ID.
    fn materials(&mut self, e: El<'_, '_>) {
        for c in e.children() {
            if c.is("mat") {
                self.material(c);
            } else if c.is("mesh") || c.is("object") || c.is("patch") {
                self.materials(c);
            }
        }
    }

    fn material(&mut self, e: El<'_, '_>) -> usize {
        let index = self.scene.materials.len();
        let name = id(&e).map_or_else(|| format!("material {index}"), |i| format!("material {i}"));
        let mut m = Material::plain(&name, [1.0; 4]);
        let mut spec = [0.0f32; 3];
        let mut shine = 0.0f32;
        for c in e.children() {
            match c.local().to_ascii_lowercase().as_str() {
                "diff" => m.color[..3].copy_from_slice(&color(vec3(&c.text()))),
                "emiss" => m.emissive = color(vec3(&c.text())),
                "spec" => spec = vec3(&c.text()),
                "shine" => shine = num(&c),
                "alpha" => m.color[3] = num(&c).clamp(0.0, 1.0),
                _ => {}
            }
        }
        // Phong's exponent as GGX roughness: alpha = sqrt(2 / (n + 2)), roughness = sqrt(alpha); a material with
        // no highlight stays rough.
        if shine > 0.0 && spec.iter().any(|&s| s > 0.0) {
            m.roughness = (2.0 / (shine.clamp(0.0, 128.0) + 2.0)).powf(0.25);
        }
        self.scene.materials.push(m);
        if let Some(i) = id(&e) {
            self.mats.insert(i, index);
        }
        index
    }

    fn default_material(&mut self) -> usize {
        *self.default_material.get_or_insert_with(|| {
            self.scene.materials.push(Material::plain("default", [1.0; 4]));
            self.scene.materials.len() - 1
        })
    }

    /// Reads every mesh with an ID, wherever it's written, so references before definitions find them.
    fn definitions(&mut self, e: El<'_, '_>) -> Result<()> {
        for c in e.children() {
            if c.is("mesh")
                && let Some(i) = id(&c)
            {
                let mesh = self.mesh(c)?;
                self.scene.meshes.push(mesh);
                self.defs.insert(i, (self.scene.meshes.len() - 1, false));
            } else if c.is("object") {
                self.definitions(c)?;
            }
        }
        Ok(())
    }

    fn lighting(&mut self, e: El<'_, '_>, world: usize, lights: &mut Vec<(usize, Value)>) {
        for c in e.children() {
            if c.is("ambient") {
                if self.keep.extras {
                    self.scene.source.insert("ambientLight".into(), json!(vec3(&c.text())));
                }
            } else if c.is("directionallight") {
                let mut dir = [0.0, 0.0, -1.0];
                let mut diffuse = [1.0; 3];
                let mut specular = None;
                for p in c.children() {
                    match p.local().to_ascii_lowercase().as_str() {
                        "direction" => dir = vec3(&p.text()),
                        "diffuse" => diffuse = vec3(&p.text()),
                        "specular" => specular = Some(vec3(&p.text())),
                        _ => {}
                    }
                }
                let count = lights.len();
                let mut node = Node::named(format!("light {count}"), Some(world));
                // The light shines along its node's -Z: turn -Z to the direction it travels.
                node.rotation = toward(dir.map(f64::from));
                let node = self.scene.add(node);
                let mut light = json!({ "type": "directional", "color": color(diffuse), "intensity": 1.0 });
                if let Some(s) = specular.filter(|_| self.keep.extras) {
                    light["extras"] = json!({ "specular": s });
                }
                lights.push((node, light));
            } else if c.is("spheremap") {
                *self.unsupported.entry("spheremap".into()).or_default() += 1;
            }
        }
    }

    /// An object's (or the world's) contents under `node`.
    fn object(&mut self, e: El<'_, '_>, node: usize) -> Result<()> {
        for c in e.children() {
            match c.local().to_ascii_lowercase().as_str() {
                "name" if node != 0 || !e.is("world") => {
                    let name = c.text().trim().to_owned();
                    if !name.is_empty() {
                        self.scene.nodes[node].name = name;
                    }
                }
                "transform" => {
                    if let Some(m) = transform(c) {
                        let (t, r, s) = scene::decompose(&m);
                        let n = &mut self.scene.nodes[node];
                        n.translation = t;
                        n.rotation = r;
                        n.scale = s;
                    }
                }
                "mesh" => {
                    if let Some(i) = id(&c) {
                        self.homes.push((i, node));
                    } else {
                        let mesh = self.mesh(c)?;
                        let index = self.scene.meshes.len();
                        self.scene.meshes.push(mesh);
                        self.attach(node, index);
                    }
                }
                "meshref" => {
                    let i = c.text().trim().to_owned();
                    match self.defs.get_mut(&i) {
                        Some(d) => {
                            d.1 = true;
                            let mesh = d.0;
                            self.place(node, mesh, &i);
                        }
                        None => *self.unsupported.entry("meshref to nothing".into()).or_default() += 1,
                    }
                }
                "object" => {
                    let count = self.scene.nodes.len();
                    let child = self.scene.add(Node::named(format!("object {count}"), Some(node)));
                    self.object(c, child)?;
                }
                tag @ ("objectref" | "texture" | "texturergb" | "texturergba" | "linestyle" | "pointstyle") => {
                    *self.unsupported.entry(tag.to_owned()).or_default() += 1;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Puts mesh `mesh` on `node`, or on a child of it when the node already has one.
    fn attach(&mut self, node: usize, mesh: usize) {
        if self.scene.nodes[node].mesh.is_none() {
            self.scene.nodes[node].mesh = Some(mesh);
        } else {
            let name = self.scene.meshes[mesh].name.clone();
            let mut n = Node::named(name, Some(node));
            n.mesh = Some(mesh);
            self.scene.add(n);
        }
    }

    fn place(&mut self, node: usize, mesh: usize, _id: &str) {
        self.attach(node, mesh);
    }

    fn mesh(&mut self, e: El<'_, '_>) -> Result<SceneMesh> {
        let mut points: HashMap<String, [f32; 3]> = HashMap::new();
        let mut normals: HashMap<String, [f32; 3]> = HashMap::new();
        let mut uvs: HashMap<String, [f32; 2]> = HashMap::new();
        for c in e.children() {
            let Some(i) = id(&c) else { continue };
            if c.is("p") {
                points.insert(i, vec3(&c.text()));
            } else if c.is("n") {
                normals.insert(i, vec3(&c.text()));
            } else if c.is("tc") {
                uvs.insert(i, vec2(&c.text()));
            }
        }
        let tables = Tables {
            points: &points,
            normals: &normals,
            uvs: &uvs,
        };
        let mut built: BTreeMap<Key, Builder> = BTreeMap::new();
        self.primitives(e, None, &tables, &mut built)?;
        let name = id(&e).map_or_else(|| format!("mesh {}", self.scene.meshes.len()), |i| format!("mesh {i}"));
        let mut out = SceneMesh {
            name,
            ..SceneMesh::default()
        };
        for ((material, kind, _, _), b) in built {
            let mut prim = b.prim;
            prim.material = Some(material.unwrap_or_else(|| self.default_material()));
            prim.lines = kind == 2;
            prim.points = kind == 1;
            if prim.uvs.first().is_some_and(Vec::is_empty) {
                prim.uvs.clear();
            }
            out.prims.push(prim);
        }
        Ok(out)
    }

    /// The primitives in `e` (a mesh or a patch), their material `material` when they name none.
    fn primitives(
        &mut self,
        e: El<'_, '_>,
        material: Option<usize>,
        t: &Tables<'_>,
        built: &mut BTreeMap<Key, Builder>,
    ) -> Result<()> {
        let mut material = material;
        // A patch's own material, written before or after its primitives.
        if e.is("patch") {
            for c in e.children() {
                if c.is("matref") {
                    material = self.mats.get(c.text().trim()).copied().or(material);
                } else if c.is("mat") {
                    material = Some(self.material(c));
                }
            }
        }
        for c in e.children() {
            let kind: u8 = match c.local().to_ascii_lowercase().as_str() {
                "f" => 3,
                "l" => 2,
                "p" if id(&c).is_none() => 1,
                "patch" => {
                    self.primitives(c, material, t, built)?;
                    continue;
                }
                _ => continue,
            };
            let mut own = None;
            let mut verts: [Option<Vertex>; 3] = [None; 3];
            let mut bad = false;
            for v in c.children() {
                let slot = match v.local().to_ascii_lowercase().as_str() {
                    "fv1" | "lv1" | "pv1" => 0,
                    "fv2" | "lv2" => 1,
                    "fv3" => 2,
                    "matref" => {
                        own = self.mats.get(v.text().trim()).copied();
                        continue;
                    }
                    "mat" => {
                        own = Some(self.material(v));
                        continue;
                    }
                    _ => continue,
                };
                match vertex(v, t) {
                    Some(x) => verts[slot] = Some(x),
                    None => bad = true,
                }
            }
            let need = usize::from(kind);
            let Some(vs) = verts[..need]
                .iter()
                .copied()
                .collect::<Option<Vec<Vertex>>>()
                .filter(|_| !bad)
            else {
                self.dropped += 1;
                continue;
            };
            if kind == 3 {
                self.triangles += 1;
                let limit = self.limit;
                ensure!(
                    self.triangles <= limit,
                    LimitExceeded,
                    "too many triangles to convert (more than {limit})"
                );
            }
            let normals = vs.iter().all(|v| v.n.is_some());
            let coords = vs.iter().all(|v| v.tc.is_some());
            let b = built
                .entry((own.or(material), kind, normals, coords))
                .or_insert_with(|| Builder {
                    prim: Primitive {
                        uvs: vec![Vec::new()],
                        ..Primitive::default()
                    },
                    ..Builder::default()
                });
            for mut v in vs {
                if !normals {
                    v.n = None;
                }
                if !coords {
                    v.tc = None;
                }
                let i = b.vertex(v);
                b.prim.indices.push(i);
            }
        }
        Ok(())
    }
}

/// A mesh's points, normals and coordinates by ID.
struct Tables<'a> {
    points: &'a HashMap<String, [f32; 3]>,
    normals: &'a HashMap<String, [f32; 3]>,
    uvs: &'a HashMap<String, [f32; 2]>,
}

/// A face, line or point vertex: references to the mesh's tables, or values of its own. `None` when it has no
/// position, or refers to one that isn't there.
fn vertex(e: El<'_, '_>, t: &Tables<'_>) -> Option<Vertex> {
    let mut p = None;
    let mut n = None;
    let mut tc = None;
    for c in e.children() {
        let text = c.text();
        let key = text.trim();
        match c.local().to_ascii_lowercase().as_str() {
            "pref" => p = Some(*t.points.get(key)?),
            "nref" => n = Some(*t.normals.get(key)?),
            "tcref" => tc = Some(*t.uvs.get(key)?),
            "p" => p = Some(vec3(key)),
            "n" => n = Some(vec3(key)),
            "tc" => tc = Some(vec2(key)),
            _ => {}
        }
    }
    Some(Vertex { p: p?, n, tc })
}

/// A `<TRANSFORM>`'s matrix: +Z forward, +Y up, +X up × forward, a uniform scale, then the position. `None` for
/// directions that are zero or not square to each other.
fn transform(e: El<'_, '_>) -> Option<M4> {
    let mut forward = [0.0, 0.0, 1.0];
    let mut up = [0.0, 1.0, 0.0];
    let mut position = [0.0; 3];
    let mut scale = 1.0;
    for c in e.children() {
        match c.local().to_ascii_lowercase().as_str() {
            "forward" => forward = vec3(&c.text()).map(f64::from),
            "up" => up = vec3(&c.text()).map(f64::from),
            "position" => position = vec3(&c.text()).map(f64::from),
            "scale" => scale = f64::from(num(&c)),
            _ => {}
        }
    }
    let unit = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (l > 1e-6 && l.is_finite()).then(|| v.map(|c| c / l))
    };
    let z = unit(forward)?;
    let y = unit(up)?;
    if (z[0] * y[0] + z[1] * y[1] + z[2] * y[2]).abs() > 1e-3 {
        return None;
    }
    let x = [
        y[1] * z[2] - y[2] * z[1],
        y[2] * z[0] - y[0] * z[2],
        y[0] * z[1] - y[1] * z[0],
    ];
    let s = if scale > 0.0 && scale.is_finite() { scale } else { 1.0 };
    let mut m = scene::IDENTITY;
    for (col, v) in [x, y, z].iter().enumerate() {
        for k in 0..3 {
            m[col * 4 + k] = v[k] * s;
        }
    }
    m[12..15].copy_from_slice(&position);
    Some(m)
}

/// The rotation that turns -Z to `dir`.
fn toward(dir: [f64; 3]) -> [f32; 4] {
    let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if l < 1e-9 || !l.is_finite() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let d = dir.map(|c| c / l);
    let from = [0.0, 0.0, -1.0];
    let dot = -d[2];
    if dot < -0.999_999 {
        // Opposite: half a turn about Y.
        return [0.0, 1.0, 0.0, 0.0];
    }
    // q = (from × d, 1 + from · d), normalized.
    let c = [
        from[1] * d[2] - from[2] * d[1],
        from[2] * d[0] - from[0] * d[2],
        from[0] * d[1] - from[1] * d[0],
    ];
    scene::normalize([c[0], c[1], c[2], 1.0 + dot])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;

    const CUBE: &str = r#"<WORLD>
<BACKGROUND><BACKCOLOR>1,1,1</BACKCOLOR></BACKGROUND>
<LIGHTING><AMBIENT>0.1,0.1,0.1</AMBIENT>
<DIRECTIONALLIGHT><DIFFUSE>1,1,1</DIFFUSE><DIRECTION>0,-1,0</DIRECTION></DIRECTIONALLIGHT></LIGHTING>
<MESH ID="0">
<MAT ID="0"><DIFF>0,1,0</DIFF><ALPHA>0.5</ALPHA><SPEC>1,1,1</SPEC><SHINE>64</SHINE></MAT>
<P ID="0">0,0,0</P><P ID="1">1,0,0</P><P ID="2">0,1,0</P>
<N ID="0">0,0,1</N>
<TC ID="0">0,0</TC>
<F><MATREF>0</MATREF><FV1><PREF>0</PREF><NREF>0</NREF><TCREF>0</TCREF></FV1>
<FV2><PREF>1</PREF><NREF>0</NREF><TCREF>0</TCREF></FV2><FV3><PREF>2</PREF><NREF>0</NREF><TCREF>0</TCREF></FV3></F>
<F><FV1><PREF>0</PREF></FV1><FV2><PREF>9</PREF></FV2><FV3><PREF>2</PREF></FV3></F>
<L><LV1><PREF>0</PREF></LV1><LV2><P>5,5,5</P></LV2></L>
<P><PV1><PREF>1</PREF></PV1></P>
</MESH>
<OBJECT><NAME>a</NAME><TRANSFORM><FORWARD>1,0,0</FORWARD><UP>0,1,0</UP><POSITION>0,0,10</POSITION><SCALE>2</SCALE></TRANSFORM>
<MESHREF>0</MESHREF></OBJECT>
<OBJECT><NAME>b</NAME><MESHREF>0</MESHREF></OBJECT>
</WORLD>"#;

    fn load(bytes: &[u8], name: &str) -> Result<Document> {
        let ext = name.rsplit('.').next().unwrap();
        let assets = Memory::new();
        crate::formats::load_bytes(bytes, name, ext, &assets, Keep::ALL)
    }

    #[test]
    fn reads_meshes_objects_and_lights() {
        let doc = load(CUBE.as_bytes(), "cube.xgl").unwrap();
        let j = &doc.json();
        assert_eq!(doc.counts().triangles, 2);
        assert_eq!(doc.facts["dropped_primitives"], 1);
        let prims = j["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 3, "{prims:?}");
        let m = &j["materials"][0];
        assert_eq!(m["alphaMode"], "BLEND");
        assert_eq!(
            j["extensions"]["KHR_lights_punctual"]["lights"][0]["type"],
            "directional"
        );
        // The light's -Z turned straight down.
        let light = j["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n.get("extensions").is_some())
            .unwrap();
        let q = light["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let d = scene::rotate([q[0], q[1], q[2], q[3]], [0.0, 0.0, -1.0]);
        assert!((d[1] + 1.0).abs() < 1e-6, "{d:?}");
        // Object a: +Z forward along +X, so local +X (up × forward) points to -Z; scaled 2, at z = 10.
        let a = j["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == "a")
            .unwrap();
        let q = a["rotation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let x = scene::rotate([q[0], q[1], q[2], q[3]], [1.0, 0.0, 0.0]);
        assert!((x[2] + 1.0).abs() < 1e-6, "{x:?}");
        assert_eq!(a["scale"][0], 2.0);
        assert_eq!(j["asset"]["extras"]["source"]["backgroundColor"][0], 1.0);
        let flat = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_ne!(flat.prims.len(), 0);
    }

    #[test]
    fn reads_zipped_and_as_xml() {
        let plain = load(CUBE.as_bytes(), "cube.xgl").unwrap().to_glb().unwrap();
        assert_eq!(load(CUBE.as_bytes(), "cube.xml").unwrap().to_glb().unwrap(), plain);
        // A zlib stream of stored blocks.
        let mut z = vec![0x78, 0x01];
        let data = CUBE.as_bytes();
        let chunks: Vec<&[u8]> = data.chunks(60000).collect();
        for (i, c) in chunks.iter().enumerate() {
            z.push(u8::from(i + 1 == chunks.len()));
            let n = u16::try_from(c.len()).unwrap();
            z.extend_from_slice(&n.to_le_bytes());
            z.extend_from_slice(&(!n).to_le_bytes());
            z.extend_from_slice(c);
        }
        z.extend_from_slice(&[0; 4]);
        assert_eq!(load(&z, "cube.zgl").unwrap().to_glb().unwrap(), plain);
    }

    #[test]
    fn refuses_what_isnt_xgl() {
        assert!(load(b"<WORLD></WORLD>", "a.xgl").is_err());
        assert!(load(b"<nothing/>", "a.xgl").is_err());
        assert!(load(b"\x78\x01garbage", "a.zgl").is_err());
    }
}
