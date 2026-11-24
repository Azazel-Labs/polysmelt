//! X3D and VRML97 scenes: the classic encoding (`.wrl`, `.x3dv`), the XML one (`.x3d`), and
//! either gzipped (`.wrz`, `.x3dz`, or any of the others when the bytes say so).
//!
//! Both encodings are read into one scene graph of nodes and their fields, which is then
//! walked from its roots:
//!
//! - grouping nodes (`Transform`, `Group`, `Anchor`, `Billboard`, `Collision`, `StaticGroup`,
//!   the CAD nodes, `Switch` by its choice, `LOD` by its finest level) become nodes, a
//!   transform's center folded into its translation;
//! - a `Shape` becomes a mesh, its `Appearance` a material: `Material`'s diffuse and emissive
//!   colors and transparency, `PhysicalMaterial` and `UnlitMaterial` as they are, an
//!   `ImageTexture` by its relative URL (clamped where it doesn't repeat), a `TextureTransform`
//!   baked into the texture coordinates;
//! - the geometry: `IndexedFaceSet` (polygons of any size, separate normal, color and texture
//!   coordinate indices, per-vertex or per-face colors and normals, `ccw`, `creaseAngle`), the
//!   triangle, strip, fan and quad sets, `ElevationGrid`, `Extrusion` (its spine-aligned
//!   cross-section planes as the specification computes them), and `Box`, `Sphere`, `Cylinder`
//!   and `Cone` tessellated, and the filled 2D nodes (`Rectangle2D`, `Disk2D`, `ArcClose2D`,
//!   `TriangleSet2D`) in the Z = 0 plane; `solid FALSE` makes the material double-sided;
//! - the line and point sets (`IndexedLineSet` with its per-vertex or per-polyline colors,
//!   `LineSet`, `PointSet`, and the 2D `Circle2D`, `Arc2D`, `Polyline2D` and `Polypoint2D`) as
//!   lines and points, unlit as X3D draws them: in their colors, else in the material's emissive
//!   color (white with no material);
//! - `Viewpoint` and `OrthoViewpoint` become cameras, the point, spot and directional lights
//!   `KHR_lights_punctual` lights;
//! - `DEF` names nodes and `USE` instances them (a shape's mesh is made once), a node that
//!   contains itself stops there; `Inline` reads the file it names through the assets, a few
//!   levels deep;
//! - prototypes (`PROTO`, `ProtoDeclare`) are instanced, and external ones (`EXTERNPROTO`,
//!   `ExternProtoDeclare`) read from the file they name through the assets (see `proto`).
//!
//! Scripts, routes, sensors and interpolators are skipped, as is any node not named here
//! (`Text`, `NurbsPatchSurface`): the asset's `extras.source.skipped` counts them.
//! Colors are taken as sRGB. `UNIT length` scales the scene.
//!
//! The asset's `extras.source` records the encoding, the version and the length unit (meters
//! unless `UNIT length` names another, a scene in another unit scaled to meters), +Y up as the
//! convention, the first `WorldInfo`'s title and `info` strings (as comments), and the `META`
//! statements (`meta` elements): `title`, `creator` or `author`, `generator` as the tool,
//! `description`, `copyright` or `rights`, `license`, `created`, `modified` and `keywords` by
//! their meanings, any other under `meta` as written.

mod classic;
mod geometry;
mod proto;
mod xml;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read as _;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::Input;
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::Images;
use crate::material::{Alpha, CLAMP, Full, REPEAT, Tex};
use crate::mesh::{GRAY, Material, srgb_to_linear};
use crate::num::f32_of;
use crate::source::{Axis, Converted, Fact, Handedness, SourceInfo};

/// An absent index.
pub(super) const NONE: u32 = u32::MAX;

/// Nodes nested deeper than this are skipped, in parsing and in walking the graph.
pub(super) const MAX_DEPTH: usize = 200;

/// Files an `Inline` may lead through.
const MAX_INLINE: usize = 8;

/// Nodes a scene may make in all, counting each instance.
const MAX_NODES: usize = 1 << 20;

/// The most a gzipped file may inflate to.
const MAX_INFLATED: u64 = 1 << 30;

/// An index field's value: [`NONE`] for a negative one or one that isn't a number.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "checked to be in range"
)]
#[inline]
pub(super) fn idx(x: f64) -> u32 {
    if (0.0..4_294_967_295.0).contains(&x) {
        x as u32
    } else {
        NONE
    }
}

/// A field's value, its type told from how it's written.
#[derive(Clone, Debug)]
pub(super) enum Val {
    Nums(Vec<f64>),
    Strs(Vec<String>),
    Bools(Vec<bool>),
    Nodes(Vec<usize>),
    /// In a prototype's body: the interface field whose value this field takes (`IS`).
    Is(String),
}

#[derive(Debug, Default)]
pub(super) struct Node {
    pub kind: String,
    /// Its `DEF` name.
    pub name: Option<String>,
    pub fields: Vec<(String, Val)>,
}

/// A prototype: its interface's fields with their defaults, and the nodes of its body.
#[derive(Clone, Debug, Default)]
pub(super) struct Proto {
    pub name: String,
    /// Each field's default; `None` for one given none.
    pub fields: Vec<(String, Option<Val>)>,
    /// The body's nodes: an instance is the first, the others only there to be used by it.
    pub body: Vec<usize>,
}

/// A file's scene: its nodes, the ones at the top, and what was skipped reading it.
#[derive(Debug, Default)]
pub(super) struct Graph {
    pub nodes: Vec<Node>,
    pub roots: Vec<usize>,
    /// Meters a length unit is, from `UNIT length`.
    pub unit: Option<f64>,
    /// The length unit's name, likewise.
    pub unit_name: Option<String>,
    /// The file's `META` statements (`meta` elements), each name and content.
    pub meta: Vec<(String, String)>,
    /// Statements and nodes not read, by kind.
    pub skipped: BTreeMap<String, usize>,
    pub version: String,
    /// Prototypes declared, in order.
    pub protos: Vec<Proto>,
    /// External prototypes declared: each one's name and URLs.
    pub externs: Vec<(String, Vec<String>)>,
    pub notes: BTreeSet<&'static str>,
}

impl Graph {
    pub(super) fn add(&mut self, kind: &str, name: Option<String>) -> usize {
        self.nodes.push(Node {
            kind: kind.to_owned(),
            name,
            fields: Vec::new(),
        });
        self.nodes.len() - 1
    }

    /// Sets field `name` of node `id`; the last of a field given twice counts.
    pub(super) fn set(&mut self, id: usize, name: &str, v: Val) {
        let fields = &mut self.nodes[id].fields;
        match fields.iter_mut().find(|(n, _)| n == name) {
            Some(f) => f.1 = v,
            None => fields.push((name.to_owned(), v)),
        }
    }

    /// Adds `child` to node field `field` of node `id`.
    pub(super) fn push_child(&mut self, id: usize, field: &str, child: usize) {
        let fields = &mut self.nodes[id].fields;
        match fields.iter_mut().find(|(n, _)| n == field) {
            Some((_, Val::Nodes(list))) => list.push(child),
            Some(f) => f.1 = Val::Nodes(vec![child]),
            None => fields.push((field.to_owned(), Val::Nodes(vec![child]))),
        }
    }

    pub(super) fn skip(&mut self, what: &str) {
        *self.skipped.entry(what.to_owned()).or_default() += 1;
    }

    /// The prototype named `name`: the last declared of that name.
    pub(super) fn proto(&self, name: &str) -> Option<usize> {
        self.protos.iter().rposition(|p| p.name == name)
    }

    fn field(&self, id: usize, name: &str) -> Option<&Val> {
        let node = self.nodes.get(id)?;
        node.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    pub(super) fn kind(&self, id: usize) -> &str {
        self.nodes.get(id).map_or("", |n| n.kind.as_str())
    }

    fn name(&self, id: usize) -> Option<&str> {
        self.nodes.get(id)?.name.as_deref()
    }

    pub(super) fn nums(&self, id: usize, name: &str) -> &[f64] {
        match self.field(id, name) {
            Some(Val::Nums(v)) => v,
            _ => &[],
        }
    }

    /// A number field's value, or `default` when it's absent or not finite.
    pub(super) fn num(&self, id: usize, name: &str, default: f64) -> f64 {
        self.nums(id, name)
            .first()
            .copied()
            .filter(|x| x.is_finite())
            .unwrap_or(default)
    }

    /// A vector field's value, or `default` when it's absent, short or not finite.
    pub(super) fn vec<const N: usize>(&self, id: usize, name: &str, default: [f64; N]) -> [f64; N] {
        match self.nums(id, name).first_chunk::<N>() {
            Some(v) if v.iter().all(|x| x.is_finite()) => *v,
            _ => default,
        }
    }

    pub(super) fn flag(&self, id: usize, name: &str, default: bool) -> bool {
        match self.field(id, name) {
            Some(Val::Bools(v)) => v.first().copied().unwrap_or(default),
            _ => default,
        }
    }

    pub(super) fn nodes(&self, id: usize, name: &str) -> &[usize] {
        match self.field(id, name) {
            Some(Val::Nodes(v)) => v,
            _ => &[],
        }
    }

    pub(super) fn node(&self, id: usize, name: &str) -> Option<usize> {
        self.nodes(id, name).first().copied()
    }

    pub(super) fn strs(&self, id: usize, name: &str) -> &[String] {
        match self.field(id, name) {
            Some(Val::Strs(v)) => v,
            _ => &[],
        }
    }

    /// A field of 3D points or vectors.
    pub(super) fn vec3s(&self, id: usize, name: &str) -> Vec<[f32; 3]> {
        let v = self.nums(id, name);
        v.as_chunks::<3>().0.iter().map(|c| c.map(f32_of)).collect()
    }

    /// A `Color` or `ColorRGBA` node's colors, linear.
    pub(super) fn colors(&self, id: usize) -> Vec<[f32; 4]> {
        let lin = |c: f64| srgb_to_linear(f32_of(c).clamp(0.0, 1.0));
        let v = self.nums(id, "color");
        if self.kind(id) == "ColorRGBA" {
            let c = v.as_chunks::<4>().0;
            c.iter()
                .map(|c| [lin(c[0]), lin(c[1]), lin(c[2]), f32_of(c[3]).clamp(0.0, 1.0)])
                .collect()
        } else {
            v.as_chunks::<3>()
                .0
                .iter()
                .map(|c| [lin(c[0]), lin(c[1]), lin(c[2]), 1.0])
                .collect()
        }
    }

    /// A `TextureCoordinate` node's coordinates, or the first set of a
    /// `MultiTextureCoordinate`.
    pub(super) fn tex_coords(&self, id: usize) -> Vec<[f32; 2]> {
        let id = match self.kind(id) {
            "MultiTextureCoordinate" => match self.node(id, "texCoord") {
                Some(t) => t,
                None => return Vec::new(),
            },
            _ => id,
        };
        if self.kind(id) != "TextureCoordinate" {
            return Vec::new();
        }
        let v = self.nums(id, "point");
        v.as_chunks::<2>().0.iter().map(|c| c.map(f32_of)).collect()
    }
}

/// What a file is, for the asset's `extras.source`.
pub(super) struct Info {
    format: &'static str,
    encoding: &'static str,
    gzip: bool,
}

/// A file's scene graph: inflated when gzipped, then by its first character the XML encoding
/// or the classic one.
fn parse(bytes: &[u8]) -> Result<(Graph, Info)> {
    let mut inflated = Vec::new();
    let gzip = bytes.starts_with(&[0x1f, 0x8b]);
    let bytes = if gzip {
        let mut r = flate2::read::MultiGzDecoder::new(bytes).take(MAX_INFLATED);
        // A file cut short reads as far as it inflates.
        if r.read_to_end(&mut inflated).is_err() && inflated.is_empty() {
            bail!("not a gzipped X3D or VRML file");
        }
        inflated.as_slice()
    } else {
        bytes
    };
    let decoded = crate::formats::xml::utf8(bytes);
    let bytes = &*decoded;
    let start = bytes
        .iter()
        .position(|&b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let bytes = &bytes[start..];
    if bytes.first() == Some(&b'<') {
        let g = xml::parse(bytes);
        let info = Info {
            format: "X3D",
            encoding: "XML",
            gzip,
        };
        return Ok((g, info));
    }
    let line = bytes.split(|&b| b == b'\n' || b == b'\r').next().unwrap_or_default();
    let mut words = line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty());
    let (head, version) = (words.next().unwrap_or_default(), words.next().unwrap_or_default());
    let version = String::from_utf8_lossy(version.strip_prefix(b"V").unwrap_or(version)).into_owned();
    let format = match head {
        b"#VRML" if version.starts_with('1') => bail!("VRML 1.0 isn't supported, only VRML97 and X3D"),
        b"#VRML" => "VRML97",
        b"#X3D" => "X3D",
        _ if bytes.is_empty() => bail!("an empty file"),
        _ => "VRML97",
    };
    let mut g = classic::parse(bytes);
    if matches!(head, b"#VRML" | b"#X3D") {
        g.version = version;
    }
    let info = Info {
        format,
        encoding: "classic",
        gzip,
    };
    Ok((g, info))
}

pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let (mut graph, info) = parse(&input.read()?)?;
    let dir = input.path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut library = proto::Library::new();
    proto::resolve(&mut graph, assets, &dir, &mut library, 0);
    proto::expand(&mut graph);
    let mut c = Convert {
        assets,
        keep,
        limit: input.limit,
        images: Images::new(!keep.image_bytes, &dir),
        graphs: vec![graph],
        dirs: vec![dir],
        library,
        inline_of: HashMap::new(),
        b: Builder::default(),
        nodes: Vec::new(),
        meshes: Vec::new(),
        mesh_tris: Vec::new(),
        mesh_of: HashMap::new(),
        materials: Vec::new(),
        material_of: HashMap::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        triangles: 0,
        visiting: Vec::new(),
        skipped: BTreeMap::new(),
        notes: BTreeSet::new(),
        title: None,
        info: Vec::new(),
    };
    c.inline_of.insert(input.path.to_path_buf(), Some(0));
    let mut roots = Vec::new();
    for r in c.graphs[0].roots.clone() {
        roots.extend(c.visit(0, r, 0)?);
    }
    if let Some(u) = c.graphs[0].unit.filter(|u| u.is_finite() && *u > 0.0)
        && (u - 1.0).abs() > 1e-12
        && !roots.is_empty()
    {
        roots = vec![c.push_node(json!({ "name": "RootNode", "children": roots, "scale": [u, u, u] }))];
    }
    Ok(c.finish(&roots, &info))
}

/// Walks scene graphs into a glTF document.
struct Convert<'a> {
    assets: &'a dyn Assets,
    keep: Keep,
    limit: usize,
    images: Images,
    /// The file's graph first, then those of the files inlined.
    graphs: Vec<Graph>,
    /// The folder of each graph's file, which its references are from.
    dirs: Vec<PathBuf>,
    /// The files external prototypes were read from.
    library: proto::Library,
    inline_of: HashMap<PathBuf, Option<usize>>,
    b: Builder,
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    /// Each mesh's triangles.
    mesh_tris: Vec<usize>,
    /// Each shape's mesh, by graph and node.
    mesh_of: HashMap<(usize, usize), Option<usize>>,
    materials: Vec<Full>,
    /// By graph, appearance, whether double-sided and whether vertex-colored.
    material_of: HashMap<(usize, usize, bool, bool, bool), usize>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    /// Triangles drawn so far, each instance counted.
    triangles: usize,
    /// The nodes being walked, by graph and node, outermost first.
    visiting: Vec<(usize, usize)>,
    skipped: BTreeMap<String, usize>,
    notes: BTreeSet<&'static str>,
    title: Option<String>,
    /// The first `WorldInfo`'s `info` strings.
    info: Vec<String>,
}

impl Convert<'_> {
    fn push_node(&mut self, node: Value) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn skip(&mut self, what: &str) {
        *self.skipped.entry(what.to_owned()).or_default() += 1;
    }

    /// The glTF node for node `id` of graph `g`, and what's under it; `None` when nothing is
    /// drawn or placed there.
    fn visit(&mut self, g: usize, id: usize, inline: usize) -> Result<Option<usize>> {
        if self.visiting.contains(&(g, id)) {
            self.notes.insert("a node that contains itself was drawn once");
            return Ok(None);
        }
        if self.visiting.len() >= MAX_DEPTH {
            self.notes.insert("nodes nested too deep were skipped");
            return Ok(None);
        }
        if self.nodes.len() >= MAX_NODES {
            self.notes.insert("instances past the node limit were skipped");
            return Ok(None);
        }
        self.visiting.push((g, id));
        let out = self.node(g, id, inline);
        self.visiting.pop();
        out
    }

    fn node(&mut self, g: usize, id: usize, inline: usize) -> Result<Option<usize>> {
        let graph = &self.graphs[g];
        let kind = graph.kind(id);
        let kids = |field: &str| graph.nodes(id, field).to_vec();
        let (children, transform) = match kind {
            "Transform" | "CADPart" | "HAnimSegment" => (kids("children"), true),
            "Group" | "StaticGroup" | "Anchor" | "Billboard" | "Collision" | "CADAssembly" | "CADLayer" => {
                (kids("children"), false)
            }
            "CADFace" => (kids("shape"), false),
            "Switch" => {
                let list = if graph.nodes(id, "choice").is_empty() {
                    kids("children")
                } else {
                    kids("choice")
                };
                let which = idx(graph.num(id, "whichChoice", -1.0));
                (list.get(which as usize).map(|&c| vec![c]).unwrap_or_default(), false)
            }
            "LOD" => {
                let list = if graph.nodes(id, "level").is_empty() {
                    kids("children")
                } else {
                    kids("level")
                };
                (list.first().map(|&c| vec![c]).unwrap_or_default(), false)
            }
            "Shape" => return self.shape(g, id),
            "Inline" => return self.inline(g, id, inline),
            "Viewpoint" | "OrthoViewpoint" => return Ok(Some(self.camera(g, id))),
            "DirectionalLight" | "PointLight" | "SpotLight" => return Ok(self.light(g, id)),
            "WorldInfo" => {
                if self.title.is_none() {
                    self.title = graph.strs(id, "title").first().filter(|t| !t.is_empty()).cloned();
                }
                if self.info.is_empty() {
                    self.info = graph
                        .strs(id, "info")
                        .iter()
                        .filter(|t| !t.is_empty())
                        .cloned()
                        .collect();
                }
                return Ok(None);
            }
            "NavigationInfo" | "Background" | "Fog" | "TextureBackground" | "MetadataSet" | "MetadataString"
            | "MetadataFloat" | "MetadataInteger" | "MetadataDouble" | "MetadataBoolean" => return Ok(None),
            _ => {
                let kind = kind.to_owned();
                self.skip(&kind);
                return Ok(None);
            }
        };
        let mut node = Map::new();
        if let Some(name) = graph.name(id) {
            node.insert("name".into(), json!(name));
        }
        if transform {
            self.place(g, id, &mut node);
        }
        let mut list = Vec::new();
        for c in children {
            list.extend(self.visit(g, c, inline)?);
        }
        if list.is_empty() {
            return Ok(None);
        }
        node.insert("children".into(), json!(list));
        Ok(Some(self.push_node(Value::Object(node))))
    }

    /// A transform's translation, rotation and scale on `node`. Its center is folded into the
    /// translation; a scale orientation is kept only where the scale is uniform, as glTF has
    /// no place for it otherwise.
    fn place(&mut self, g: usize, id: usize, node: &mut Map<String, Value>) {
        let graph = &self.graphs[g];
        let t = graph.vec(id, "translation", [0.0; 3]);
        let c = graph.vec(id, "center", [0.0; 3]);
        let r = quat(graph.vec(id, "rotation", [0.0, 0.0, 1.0, 0.0]));
        let s = graph.vec(id, "scale", [1.0; 3]);
        let so = quat(graph.vec(id, "scaleOrientation", [0.0, 0.0, 1.0, 0.0]));
        let uniform = (s[0] - s[1]).abs() < 1e-12 && (s[1] - s[2]).abs() < 1e-12;
        if !uniform && (so[3].abs() - 1.0).abs() > 1e-9 {
            self.notes
                .insert("a scale orientation under an uneven scale was dropped");
        }
        // T·C·R·S·-C moves the origin to T + C - R·S·C.
        let rsc = rotate(r, [s[0] * c[0], s[1] * c[1], s[2] * c[2]]);
        let at = [0, 1, 2].map(|k| t[k] + c[k] - rsc[k]);
        if at.iter().any(|v| v.abs() > 0.0) {
            node.insert("translation".into(), json!(at));
        }
        if (r[3].abs() - 1.0).abs() > 1e-12 {
            node.insert("rotation".into(), json!(r));
        }
        if s.iter().any(|v| (v - 1.0).abs() > 0.0) {
            node.insert("scale".into(), json!(s));
        }
    }

    fn shape(&mut self, g: usize, id: usize) -> Result<Option<usize>> {
        let mesh = if let Some(&m) = self.mesh_of.get(&(g, id)) {
            m
        } else {
            let m = self.mesh(g, id)?;
            self.mesh_of.insert((g, id), m);
            m
        };
        let Some(m) = mesh else { return Ok(None) };
        self.triangles += self.mesh_tris[m];
        ensure!(
            self.triangles <= self.limit,
            "too many triangles to convert (more than {})",
            self.limit
        );
        let mut node = json!({ "mesh": m });
        if let Some(name) = self.graphs[g].name(id) {
            node["name"] = json!(name);
        }
        Ok(Some(self.push_node(node)))
    }

    /// A shape's mesh: its geometry with its appearance's material.
    fn mesh(&mut self, g: usize, id: usize) -> Result<Option<usize>> {
        let graph = &self.graphs[g];
        let Some(geo) = graph.node(id, "geometry") else {
            return Ok(None);
        };
        let app = graph.node(id, "appearance");
        let textured = texture(graph, app).is_some();
        let Some((mut prim, double)) = geometry::geometry(graph, geo, textured, self.limit)? else {
            let kind = graph.kind(geo).to_owned();
            self.skip(&kind);
            return Ok(None);
        };
        if prim.is_empty() {
            return Ok(None);
        }
        let name = graph.name(id).or_else(|| graph.name(geo)).map(str::to_owned);
        let transform = app
            .and_then(|a| graph.node(a, "textureTransform"))
            .filter(|&t| graph.kind(t) == "TextureTransform")
            .map(|t| UvMap::of(graph, t));
        for uv in &mut prim.uvs {
            if let Some(m) = &transform {
                *uv = m.apply(*uv);
            }
            uv[1] = 1.0 - uv[1];
        }
        let colored = !prim.colors.is_empty();
        let lines = prim.indices.is_empty();
        prim.material = self.material(g, app, double, colored, lines);
        let tris = prim.indices.len() / 3;
        let Some(mut mesh) = self.b.mesh(vec![prim], self.materials.len()) else {
            return Ok(None);
        };
        if let Some(name) = name {
            mesh["name"] = json!(name);
        }
        self.meshes.push(mesh);
        self.mesh_tris.push(tris);
        Ok(Some(self.meshes.len() - 1))
    }

    fn material(&mut self, g: usize, app: Option<usize>, double: bool, colored: bool, lines: bool) -> usize {
        let key = (g, app.unwrap_or(usize::MAX), double, colored, lines);
        if let Some(&m) = self.material_of.get(&key) {
            return m;
        }
        let m = if lines {
            self.line_material(g, app, colored)
        } else {
            self.make_material(g, app, double, colored)
        };
        self.materials.push(m);
        self.material_of.insert(key, self.materials.len() - 1);
        self.materials.len() - 1
    }

    /// An appearance as a material: no appearance is lit gray; one without a material unlit;
    /// a texture or vertex colors replace `Material`'s diffuse color.
    fn make_material(&mut self, g: usize, app: Option<usize>, double: bool, colored: bool) -> Full {
        let graph = &self.graphs[g];
        let mut m = Full::from(&Material::plain("", GRAY));
        m.double_sided = double;
        let Some(a) = app else {
            if colored {
                m.base_color = [1.0; 4];
            }
            return m;
        };
        let lin = |c: [f64; 3]| c.map(|x| srgb_to_linear(f32_of(x).clamp(0.0, 1.0)));
        let mat = graph.node(a, "material");
        let tex = texture(graph, app);
        let opacity = |k: usize| 1.0 - f32_of(graph.num(k, "transparency", 0.0)).clamp(0.0, 1.0);
        let replaced = tex.is_some() || colored;
        match mat.map(|k| (k, graph.kind(k))) {
            Some((k, "Material" | "TwoSidedMaterial")) => {
                let [r, gr, b] = if replaced {
                    [1.0; 3]
                } else {
                    lin(graph.vec(k, "diffuseColor", [0.8; 3]))
                };
                m.base_color = [r, gr, b, opacity(k)];
                m.emissive = lin(graph.vec(k, "emissiveColor", [0.0; 3]));
                m.metallic = 0.0;
                m.roughness = (1.0 - f32_of(graph.num(k, "shininess", 0.2))).clamp(0.0, 1.0);
            }
            Some((k, "PhysicalMaterial")) => {
                let [r, gr, b] = lin(graph.vec(k, "baseColor", [1.0; 3]));
                m.base_color = [r, gr, b, opacity(k)];
                m.emissive = lin(graph.vec(k, "emissiveColor", [0.0; 3]));
                m.metallic = f32_of(graph.num(k, "metallic", 1.0)).clamp(0.0, 1.0);
                m.roughness = f32_of(graph.num(k, "roughness", 1.0)).clamp(0.0, 1.0);
            }
            Some((k, "UnlitMaterial")) => {
                let [r, gr, b] = lin(graph.vec(k, "emissiveColor", [1.0; 3]));
                m.base_color = [r, gr, b, opacity(k)];
                m.unlit = true;
            }
            _ => {
                m.base_color = [1.0; 4];
                m.unlit = true;
            }
        }
        if m.base_color[3] < 1.0 {
            m.alpha = Alpha::Blend;
        }
        if let Some(name) = mat.and_then(|k| graph.name(k)).or_else(|| graph.name(a)) {
            name.clone_into(&mut m.name);
        }
        if let Some(t) = tex
            && self.keep.textures
        {
            let dir = &self.dirs[g];
            let image = graph.strs(t, "url").iter().find_map(|u| {
                let found = self.assets.find(u, dir)?;
                self.images.file(self.assets, found, u)
            });
            if let Some(image) = image {
                let mut tex = Tex::new(image);
                tex.wrap = [graph.flag(t, "repeatS", true), graph.flag(t, "repeatT", true)]
                    .map(|r| if r { REPEAT } else { CLAMP });
                m.base_color_tex = Some(tex);
            }
        }
        m
    }

    /// The material of lines and points, which X3D leaves unlit and untextured: white under
    /// their own colors, else the material's emissive color, else white with no material.
    fn line_material(&self, g: usize, app: Option<usize>, colored: bool) -> Full {
        let graph = &self.graphs[g];
        let mut m = Full::from(&Material::plain("", GRAY));
        m.unlit = true;
        m.metallic = 0.0;
        let mat = app.and_then(|a| graph.node(a, "material"));
        let lin = |c: [f64; 3]| c.map(|x| srgb_to_linear(f32_of(x).clamp(0.0, 1.0)));
        let (rgb, opacity) = match mat.map(|k| (k, graph.kind(k))) {
            Some((k, kind @ ("Material" | "TwoSidedMaterial" | "PhysicalMaterial" | "UnlitMaterial"))) => {
                let default = if kind == "UnlitMaterial" { 1.0 } else { 0.0 };
                let opacity = 1.0 - f32_of(graph.num(k, "transparency", 0.0)).clamp(0.0, 1.0);
                (lin(graph.vec(k, "emissiveColor", [default; 3])), opacity)
            }
            _ => ([1.0; 3], 1.0),
        };
        let [r, gr, b] = if colored { [1.0; 3] } else { rgb };
        m.base_color = [r, gr, b, opacity];
        if opacity < 1.0 {
            m.alpha = Alpha::Blend;
        }
        if let Some(name) = mat
            .and_then(|k| graph.name(k))
            .or_else(|| app.and_then(|a| graph.name(a)))
        {
            name.clone_into(&mut m.name);
        }
        m
    }

    /// The scene of the first file an `Inline` names that can be read.
    fn inline(&mut self, g: usize, id: usize, depth: usize) -> Result<Option<usize>> {
        if depth >= MAX_INLINE {
            self.notes.insert("inlines nested too deep were skipped");
            return Ok(None);
        }
        let urls = self.graphs[g].strs(id, "url").to_vec();
        let dir = self.dirs[g].clone();
        let mut child = None;
        for u in &urls {
            let Some(path) = self.assets.find(u, &dir) else {
                continue;
            };
            let c = if let Some(&c) = self.inline_of.get(&path) {
                c
            } else {
                let graph = self
                    .assets
                    .read(&path)
                    .and_then(|b| parse(&b).ok())
                    .map(|(graph, _)| graph);
                let c = graph.map(|mut graph| {
                    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
                    proto::resolve(&mut graph, self.assets, &dir, &mut self.library, depth + 1);
                    proto::expand(&mut graph);
                    self.graphs.push(graph);
                    self.dirs.push(dir);
                    self.graphs.len() - 1
                });
                self.inline_of.insert(path, c);
                c
            };
            if c.is_some() {
                child = c;
                break;
            }
        }
        let Some(c) = child else {
            self.notes.insert("an inlined file wasn't found or read");
            return Ok(None);
        };
        let mut list = Vec::new();
        for r in self.graphs[c].roots.clone() {
            list.extend(self.visit(c, r, depth + 1)?);
        }
        if list.is_empty() {
            return Ok(None);
        }
        let mut node = json!({ "children": list });
        if let Some(name) = self.graphs[g].name(id) {
            node["name"] = json!(name);
        }
        if let Some(u) = self.graphs[c]
            .unit
            .filter(|u| u.is_finite() && *u > 0.0 && (u - 1.0).abs() > 1e-12)
        {
            node["scale"] = json!([u, u, u]);
        }
        Ok(Some(self.push_node(node)))
    }

    fn camera(&mut self, g: usize, id: usize) -> usize {
        let graph = &self.graphs[g];
        let position = graph.vec(id, "position", [0.0, 0.0, 10.0]);
        let r = quat(graph.vec(id, "orientation", [0.0, 0.0, 1.0, 0.0]));
        let name = graph
            .name(id)
            .map(str::to_owned)
            .or_else(|| graph.strs(id, "description").first().cloned())
            .unwrap_or_else(|| "Viewpoint".into());
        let camera = if graph.kind(id) == "OrthoViewpoint" {
            let [x0, y0, x1, y1] = graph.vec(id, "fieldOfView", [-1.0, -1.0, 1.0, 1.0]);
            let half = |a: f64, b: f64| {
                let h = (b - a).abs() / 2.0;
                if h > 0.0 { h } else { 1.0 }
            };
            json!({ "name": name, "type": "orthographic",
                    "orthographic": { "xmag": half(x0, x1), "ymag": half(y0, y1), "znear": 0.01, "zfar": 1000.0 } })
        } else {
            let fov = graph
                .num(id, "fieldOfView", std::f64::consts::FRAC_PI_4)
                .clamp(0.01, 3.1);
            json!({ "name": name, "type": "perspective", "perspective": { "yfov": fov, "znear": 0.01 } })
        };
        self.cameras.push(camera);
        let mut node = json!({ "name": name, "camera": self.cameras.len() - 1 });
        if position.iter().any(|v| v.abs() > 0.0) {
            node["translation"] = json!(position);
        }
        if (r[3].abs() - 1.0).abs() > 1e-12 {
            node["rotation"] = json!(r);
        }
        self.push_node(node)
    }

    fn light(&mut self, g: usize, id: usize) -> Option<usize> {
        let graph = &self.graphs[g];
        if !graph.flag(id, "on", true) {
            return None;
        }
        let kind = graph.kind(id);
        let color = graph
            .vec(id, "color", [1.0; 3])
            .map(|x| srgb_to_linear(f32_of(x).clamp(0.0, 1.0)));
        let intensity = graph.num(id, "intensity", 1.0).max(0.0);
        let mut light = json!({ "color": color, "intensity": intensity });
        let mut node = Map::new();
        if let Some(name) = graph.name(id) {
            light["name"] = json!(name);
            node.insert("name".into(), json!(name));
        }
        if kind != "DirectionalLight" {
            let at = graph.vec(id, "location", [0.0; 3]);
            if at.iter().any(|v| v.abs() > 0.0) {
                node.insert("translation".into(), json!(at));
            }
            let radius = graph.num(id, "radius", 100.0);
            if radius > 0.0 {
                light["range"] = json!(radius);
            }
        }
        if kind != "PointLight" {
            let d = graph.vec(id, "direction", [0.0, 0.0, -1.0]);
            if let Some(q) = toward(d) {
                node.insert("rotation".into(), json!(q));
            }
        }
        light["type"] = json!(match kind {
            "DirectionalLight" => "directional",
            "PointLight" => "point",
            _ => {
                let half = std::f64::consts::FRAC_PI_2;
                let outer = graph.num(id, "cutOffAngle", half).clamp(1e-4, half);
                let inner = graph
                    .num(id, "beamWidth", std::f64::consts::FRAC_PI_4)
                    .clamp(0.0, outer * 0.999);
                light["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
                "spot"
            }
        });
        self.lights.push(light);
        node.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } }),
        );
        Some(self.push_node(Value::Object(node)))
    }

    fn finish(mut self, roots: &[usize], info: &Info) -> Document<'static> {
        let mut source = Map::new();
        source.insert("encoding".into(), json!(info.encoding));
        if info.gzip {
            source.insert("gzip".into(), json!(true));
        }
        let main = &self.graphs[0];
        let unit = main.unit.filter(|u| u.is_finite() && *u > 0.0);
        let mut si = SourceInfo {
            format: Some(info.format.into()),
            version: (!main.version.is_empty()).then(|| main.version.clone()),
            // Lengths are meters unless a `UNIT` says otherwise.
            meters_per_unit: Fact::Declared(unit.unwrap_or(1.0)),
            unit: Some(
                main.unit_name
                    .clone()
                    .filter(|_| unit.is_some())
                    .unwrap_or_else(|| "meter".into()),
            ),
            // +Y up is the default viewpoint's, a convention; the coordinates are right-handed.
            up_axis: Fact::Assumed(Axis::PosY),
            handedness: Fact::Declared(Handedness::Right),
            converted: Converted {
                units: unit.is_some_and(|u| (u - 1.0).abs() > 1e-12) && !roots.is_empty(),
                ..Converted::default()
            },
            title: self.title.clone(),
            comments: std::mem::take(&mut self.info),
            ..SourceInfo::default()
        };
        let mut other = Map::new();
        for (name, content) in &main.meta {
            let content = content.trim();
            if content.is_empty() {
                continue;
            }
            let join = |slot: &mut Option<String>| {
                *slot = Some(match slot.take() {
                    Some(old) if old != content => format!("{old}; {content}"),
                    Some(old) => old,
                    None => content.to_owned(),
                });
            };
            match name.to_ascii_lowercase().as_str() {
                "title" => {
                    if si.title.is_none() {
                        si.title = Some(content.to_owned());
                    }
                }
                "creator" | "author" => join(&mut si.author),
                "generator" => join(&mut si.tool),
                "description" => join(&mut si.description),
                "copyright" | "rights" => join(&mut si.copyright),
                "license" => join(&mut si.license),
                "created" => join(&mut si.created),
                "modified" => join(&mut si.modified),
                "keywords" => si.keywords.extend(
                    content
                        .split([',', ';'])
                        .map(str::trim)
                        .filter(|k| !k.is_empty())
                        .map(str::to_owned),
                ),
                _ => {
                    // Kept as the file has it; a name given twice keeps each content.
                    match other.get_mut(name) {
                        Some(Value::Array(a)) => a.push(json!(content)),
                        Some(v) => *v = json!([v.take(), content]),
                        None => {
                            other.insert(name.clone(), json!(content));
                        }
                    }
                }
            }
        }
        if !other.is_empty() {
            source.insert("meta".into(), Value::Object(other));
        }
        si.write(&mut source);
        let mut skipped = std::mem::take(&mut self.skipped);
        for g in &self.graphs {
            for (k, n) in &g.skipped {
                *skipped.entry(k.clone()).or_default() += n;
            }
            self.notes.extend(&g.notes);
        }
        if !skipped.is_empty() {
            source.insert("skipped".into(), json!(skipped));
        }
        if !self.notes.is_empty() {
            source.insert("notes".into(), json!(self.notes));
        }
        let mut json = Map::new();
        json.insert("asset".into(), json!({ "extras": { "source": source } }));
        json.insert("scene".into(), json!(0));
        let scene = if roots.is_empty() {
            json!({})
        } else {
            json!({ "nodes": roots })
        };
        json.insert("scenes".into(), json!([scene]));
        if !self.nodes.is_empty() {
            json.insert("nodes".into(), Value::Array(self.nodes));
        }
        if !self.meshes.is_empty() {
            json.insert("meshes".into(), Value::Array(self.meshes));
        }
        if !self.cameras.is_empty() {
            json.insert("cameras".into(), Value::Array(self.cameras));
        }
        if !self.lights.is_empty() {
            json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
            json.insert(
                "extensions".into(),
                json!({ "KHR_lights_punctual": { "lights": self.lights } }),
            );
        }
        self.b.materials(self.materials, self.images.list, &mut json);
        let mut facts = Map::new();
        facts.insert("triangles".into(), json!(self.triangles));
        self.b.finish(json, facts)
    }
}

/// An appearance's color texture: an `ImageTexture` as its texture, or the first in a
/// `MultiTexture`, or a material's own (X3D 4's `baseTexture`, `diffuseTexture`,
/// `emissiveTexture`).
fn texture(graph: &Graph, app: Option<usize>) -> Option<usize> {
    let a = app?;
    let image = |t: usize| (graph.kind(t) == "ImageTexture").then_some(t);
    let own = graph.node(a, "material").and_then(|m| {
        ["baseTexture", "diffuseTexture", "emissiveTexture"]
            .iter()
            .find_map(|f| graph.node(m, f).and_then(image))
    });
    own.or_else(|| {
        let t = graph.node(a, "texture")?;
        if graph.kind(t) == "MultiTexture" {
            graph.nodes(t, "texture").iter().find_map(|&t| image(t))
        } else {
            image(t)
        }
    })
}

/// A `TextureTransform`: a point is moved by `translation`, then scaled by `scale` and turned by
/// `rotation` about `center`.
struct UvMap {
    center: [f32; 2],
    translation: [f32; 2],
    scale: [f32; 2],
    sin: f32,
    cos: f32,
}

impl UvMap {
    fn of(g: &Graph, id: usize) -> Self {
        let r = f32_of(g.num(id, "rotation", 0.0));
        UvMap {
            center: g.vec(id, "center", [0.0; 2]).map(f32_of),
            translation: g.vec(id, "translation", [0.0; 2]).map(f32_of),
            scale: g.vec(id, "scale", [1.0; 2]).map(f32_of),
            sin: r.sin(),
            cos: r.cos(),
        }
    }

    /// X3D's `-C × S × R × C × T`, applied right to left.
    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let [x, y] = [0, 1].map(|k| p[k] + self.translation[k] + self.center[k]);
        let (x, y) = (x * self.cos - y * self.sin, x * self.sin + y * self.cos);
        [x * self.scale[0] - self.center[0], y * self.scale[1] - self.center[1]]
    }
}

/// An axis and angle as a unit quaternion, `[x, y, z, w]`; no turn for a zero axis.
fn quat(r: [f64; 4]) -> [f64; 4] {
    let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if !(len > 1e-12 && len.is_finite() && r[3].is_finite()) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (s, c) = (r[3] / 2.0).sin_cos();
    [r[0] / len * s, r[1] / len * s, r[2] / len * s, c]
}

/// `v` turned by unit quaternion `q`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    // v + 2w(u × v) + 2u × (u × v), u the vector part.
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

/// The turn from -Z, where glTF lights and cameras point, to `d`; `None` for none or no
/// direction.
fn toward(d: [f64; 3]) -> Option<[f64; 4]> {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if !(len > 1e-12 && len.is_finite()) {
        return None;
    }
    let d = d.map(|c| c / len);
    // From a = (0, 0, -1): a × d and 1 + a · d, normalized.
    let dot = -d[2];
    if dot > 1.0 - 1e-12 {
        return None;
    }
    if dot < -1.0 + 1e-12 {
        return Some([0.0, 1.0, 0.0, 0.0]);
    }
    let q = [d[1], -d[0], 0.0, 1.0 + dot];
    let n = (q[0] * q[0] + q[1] * q[1] + q[3] * q[3]).sqrt();
    Some(q.map(|c| c / n))
}

#[cfg(test)]
mod tests;
