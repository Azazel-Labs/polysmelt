//! 3MF, the 3D-printing format: a zip of XML models. Read here: meshes, components (in the same model or another
//! part of the zip) and the build's placed items, in the model's unit (scaled to glTF's meters); the 3MF materials
//! extension's base materials, color groups (per triangle or blended across one), 2D textures, composite materials
//! (mixed into a color) and multiproperties (layered into a color, or a texture's coordinates); the beam lattice
//! extension's beams and balls, made into triangles (see [`lattice`]); and what slicers keep beside them: the
//! filaments' colors, which filament each object and part prints in (Bambu Studio, `OrcaSlicer`, `PrusaSlicer`), the
//! filaments painted onto triangles, and which parts and volumes aren't printed but change how the rest prints
//! (modifiers, negative volumes, support blockers and enforcers).
//!
//! The build is a node per item and component, each with its 3MF transform, under a root node that
//! turns the model's Z-up millimeters (or its unit) into glTF's Y-up meters; an object's triangles
//! are a mesh in its own space, shared wherever it's placed in the same filament, its corners
//! sharing vertices as the object's triangles do. A part or volume that isn't printed is a hidden
//! node (`KHR_node_visibility`) named for what it is. A component that leads back to an object it's
//! inside is left out. Object names, types and part numbers, the model's metadata and the slicer's filament
//! colors go in extras.
//!
//! An object of the volumetric extension, a level set of implicit functions, has no surface to read: a build of
//! nothing else fails, saying so.
//!
//! The source's units are declared by the model's `unit`, or millimeters assumed when it has none, as the
//! specification says; its up axis is +Z and its handedness right, as the specification fixes them. The
//! model's metadata the specification names (`Title`, `Designer`, `Description`, `Copyright`,
//! `LicenseTerms`, `CreationDate`, `ModificationDate`, `Application`) are the source's title, author,
//! description, copyright, license, dates and tool; the rest stay in its `metadata`.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read};

use crate::document::{Builder, Document, Keep};
use crate::images::{Images, Location};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result, Within as _};
use quick_xml::events::{BytesStart, Event};
use serde_json::{Map, Value, json};
use zip::ZipArchive;

/// The 3MF package.
type Zip<'a> = ZipArchive<Box<dyn super::ReadSeek + 'a>>;

use crate::material::{self, Factor, Full, Specular, Tex, Volume};
use crate::mesh::{GRAY, Image, Material, Prim, srgb_to_linear};

mod lattice;
use lattice::{Attr, BallMode, BeamLook, Cap, Clip, Clipper, Lattice};

/// A 3MF transform: the first three columns of a 4x4 matrix, row by row (points are row vectors).
type Mat = [f32; 12];
const IDENTITY: Mat = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.];
/// The most read from one part of the zip, unpacked: more is a zip bomb, not a model.
const MAX_PART: u64 = 2 << 30;
/// Slicer settings files are small; anything bigger isn't one.
const MAX_CONFIG: u64 = 64 << 20;
/// Filament colors for a slicer project that assigns filaments without saying what color they are.
const FALLBACK: [[u8; 3]; 6] = [
    [0xB8, 0xBC, 0xC4],
    [0xD9, 0x48, 0x3B],
    [0x3B, 0x7D, 0xD8],
    [0x4C, 0xAF, 0x50],
    [0xF2, 0xB7, 0x05],
    [0x8E, 0x5B, 0xC4],
];

#[derive(Default)]
struct Tri {
    v: [u32; 3],
    pid: Option<u32>,
    /// The property of each corner in the group (p2 and p3 default to p1).
    p1: Option<u32>,
    p2: Option<u32>,
    p3: Option<u32>,
    /// The filament painted on it (0: none, the part's own).
    paint: u8,
}

struct Component {
    path: Option<String>,
    id: u32,
    transform: Mat,
}

#[derive(Default)]
struct Object {
    /// The `name`, `type` and `partnumber` attributes.
    name: Option<String>,
    kind: Option<String>,
    partnumber: Option<String>,
    pid: Option<u32>,
    pindex: Option<u32>,
    verts: Vec<[f32; 3]>,
    tris: Vec<Tri>,
    components: Vec<Component>,
    /// Its beam lattice.
    lattice: Option<Box<Lattice>>,
    /// Whether it's a volumetric level set.
    levelset: bool,
}

/// A property group of the materials extension.
enum Group {
    /// Base materials or a color group: each entry's color (sRGB, with alpha) and, for base
    /// materials, name.
    Colors(Vec<(Option<[u8; 4]>, Option<String>)>),
    /// A 2D texture's coordinates (origin bottom-left).
    Coords { texture: u32, uvs: Vec<[f32; 2]> },
    /// Composite materials, until mixed into colors: each entry's proportions (from `starts[entry]` in `values`)
    /// of the base materials `indices` of group `base`.
    Composite {
        base: u32,
        indices: Vec<u32>,
        values: Vec<f32>,
        starts: Vec<usize>,
    },
    /// Multiproperties, until layered into colors or texture coordinates: each entry's index (from
    /// `starts[entry]` in `indices`) into each of the groups `pids`, every layer after the first multiplied or,
    /// by default, mixed over the ones below.
    Multi {
        pids: Vec<u32>,
        multiply: Vec<bool>,
        indices: Vec<u32>,
        starts: Vec<usize>,
    },
}

/// How a base material or color looks beyond its color, from the materials extension's display properties
/// (their floats kept as bits, so looks can be told apart).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Finish {
    /// `pbmetallic`: metalness and roughness.
    Metal { metallic: u32, roughness: u32 },
    /// `pbspecular`: the specular color (sRGB) and glossiness.
    Specular { color: [u8; 3], glossiness: u32 },
    /// `translucent`: attenuation per unit length in RGB, the index of refraction in RGB, and roughness.
    Translucent {
        attenuation: [u32; 3],
        ior: [u32; 3],
        roughness: u32,
    },
}

impl Finish {
    fn full(self, mut m: Full) -> Full {
        let f = f32::from_bits;
        match self {
            Finish::Metal { metallic, roughness } => {
                m.metallic = f(metallic);
                m.roughness = f(roughness);
            }
            Finish::Specular { color, glossiness } => {
                m.metallic = 0.0;
                m.roughness = 1.0 - f(glossiness);
                // glTF's specular color scales the 4% a dielectric reflects head on.
                let color = color.map(|c| srgb_to_linear(f32::from(c) / 255.0) / 0.04);
                m.specular = Some(Specular {
                    factor: Factor::of(1.0),
                    color,
                    color_tex: None,
                });
            }
            Finish::Translucent {
                attenuation,
                ior,
                roughness,
            } => {
                m.metallic = 0.0;
                m.roughness = f(roughness);
                m.transmission = Some(Factor::of(1.0));
                let ior = ior.map(f);
                m.ior = Some((ior[0] + ior[1] + ior[2]) / 3.0);
                let a = attenuation.map(|a| f(a).max(0.0));
                let most = a[0].max(a[1]).max(a[2]);
                if most > 0.0 && most.is_finite() {
                    // As far as the most absorbed channel takes to fall to 1/e, and each channel's color there.
                    let d = 1.0 / most;
                    m.volume = Some(Volume {
                        thickness: Factor::of(d),
                        attenuation_distance: Some(d),
                        attenuation_color: a.map(|a| (-a * d).exp()),
                    });
                }
            }
        }
        m
    }
}

/// A texture's wrap from its `tilestyleu` or `tilestylev`.
fn tile_style(v: Option<&str>) -> u32 {
    match v {
        Some("mirror") => material::MIRROR,
        Some("clamp" | "none") => material::CLAMP,
        _ => material::REPEAT,
    }
}

/// A build item: an object (in another model file, by the production extension) placed by a transform.
struct Item {
    path: Option<String>,
    id: u32,
    transform: Mat,
    partnumber: Option<String>,
}

/// One model file of the zip.
#[derive(Default)]
struct Part {
    objects: HashMap<u32, Object>,
    groups: HashMap<u32, Group>,
    /// 2D textures by resource id: the zip part holding the image, and its wrap.
    textures: HashMap<u32, (String, [u32; 2])>,
    /// Display properties by resource id, an entry for each of a group's.
    displays: HashMap<u32, Vec<Finish>>,
    /// The display properties of a group, by group id.
    group_display: HashMap<u32, u32>,
    build: Vec<Item>,
    /// Every `<metadata name>` and its text.
    metadata: Vec<(String, String)>,
    unit: Option<String>,
}

/// Meters per unit, for the units the core specification allows (millimeters when unsaid).
fn meters(unit: Option<&str>) -> Option<f64> {
    Some(match unit.unwrap_or("millimeter") {
        "micron" => 1e-6,
        "millimeter" => 1e-3,
        "centimeter" => 1e-2,
        "inch" => 0.0254,
        "foot" => 0.3048,
        "meter" => 1.0,
        _ => return None,
    })
}

/// What a slicer project says about filaments: their colors, and which one each object, part (Bambu Studio,
/// `OrcaSlicer`) or range of triangles (`PrusaSlicer`'s volumes) prints in, what each part or volume is for, and
/// its name. Filaments count from 1.
#[derive(Default)]
struct Slicer {
    filaments: Vec<[u8; 3]>,
    objects: HashMap<u32, usize>,
    parts: HashMap<(u32, u32), Assigned>,
    /// Each object's volumes, in order of their first triangle.
    volumes: HashMap<u32, Vec<SlicerVolume>>,
}

impl Slicer {
    fn filament(&self, n: usize) -> [u8; 3] {
        let i = n.saturating_sub(1);
        self.filaments.get(i).copied().unwrap_or(FALLBACK[i % FALLBACK.len()])
    }
}

/// What a slicer's part or volume is for: printed, or one of the shapes that only change how the rest prints.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum Role {
    #[default]
    Normal,
    /// Different settings inside it.
    Modifier,
    /// Cut out of the parts.
    Negative,
    /// No supports inside it.
    Blocker,
    /// Supports inside it.
    Enforcer,
}

impl Role {
    /// Bambu Studio's and `OrcaSlicer`'s `subtype` (`modifier_part`, `negative_part`, `support_blocker`, ...) or
    /// `PrusaSlicer`'s `volume_type` (`ParameterModifier`, `NegativeVolume`, `SupportBlocker`, ...).
    fn parse(s: &str) -> Option<Role> {
        let s = s.trim().to_ascii_lowercase();
        Some(if s.contains("modifier") {
            Role::Modifier
        } else if s.contains("negative") {
            Role::Negative
        } else if s.contains("blocker") {
            Role::Blocker
        } else if s.contains("enforcer") {
            Role::Enforcer
        } else if s.contains("normal") || s.contains("model") {
            Role::Normal
        } else {
            return None;
        })
    }

    /// As a node's name says it.
    fn label(self) -> &'static str {
        match self {
            Role::Normal => "part",
            Role::Modifier => "modifier",
            Role::Negative => "negative volume",
            Role::Blocker => "support blocker",
            Role::Enforcer => "support enforcer",
        }
    }

    /// As a node's `extras.volumeType` says it.
    fn key(self) -> &'static str {
        match self {
            Role::Normal => "part",
            Role::Modifier => "modifier",
            Role::Negative => "negative_volume",
            Role::Blocker => "support_blocker",
            Role::Enforcer => "support_enforcer",
        }
    }
}

/// A slicer's settings for a part or volume.
#[derive(Default, Clone)]
struct Assigned {
    extruder: Option<usize>,
    role: Role,
    name: Option<String>,
}

/// `PrusaSlicer`'s volume: an object's triangles `first..=last`.
struct SlicerVolume {
    first: u32,
    last: u32,
    a: Assigned,
}

/// The volume triangle `i` is in, of `volumes` in order of their first triangle.
fn volume_of(volumes: &[SlicerVolume], i: u32) -> Option<&SlicerVolume> {
    let k = volumes.partition_point(|v| v.first <= i);
    volumes[..k].last().filter(|v| i <= v.last)
}

type Fast = std::hash::BuildHasherDefault<crate::mesh::FxHasher>;

// Load a 3MF: the build as nodes, an object's triangles as a mesh with a primitive per look, and
/// the facts worth keeping.
pub(crate) fn load(input: super::Input<'_>, keep: Keep) -> Result<Document> {
    let mut zip: Zip<'_> =
        ZipArchive::new(input.seekable()?).context(ErrorKind::UnknownFormat, "not a 3MF file (it isn't a zip)")?;
    let names: HashMap<String, String> = zip
        .file_names()
        .filter_map(Result::ok)
        .map(|n| (norm(&n), n.into_owned()))
        .collect();
    let root = text(&mut zip, &names, "_rels/.rels", MAX_CONFIG)
        .and_then(|rels| root_target(&rels))
        .map_or_else(|| "3d/3dmodel.model".to_owned(), |t| norm(&t));

    // The root model, then every model file its components point into.
    let mut parts: HashMap<String, Part> = HashMap::new();
    let mut todo = vec![root.clone()];
    let mut triangles = 0;
    while let Some(key) = todo.pop() {
        if parts.contains_key(&key) {
            continue;
        }
        ensure!(parts.len() < 1024, LimitExceeded, "too many model files in this 3MF");
        let Some(real) = names.get(&key) else {
            ensure!(key != root, Corrupt, "no 3D model in this 3MF");
            continue;
        };
        let reader = zip
            .by_name(real)
            .with_context(ErrorKind::Corrupt, || format!("reading {real}"))?
            .take(MAX_PART);
        let part = parse_part(reader, &mut triangles, input.limit).with_within(|| format!("reading {real}"))?;
        for o in part.objects.values() {
            todo.extend(o.components.iter().filter_map(|c| c.path.as_deref()).map(norm));
        }
        todo.extend(part.build.iter().filter_map(|i| i.path.as_deref()).map(norm));
        parts.insert(key, part);
    }
    let slicer = read_slicer(&mut zip, &names);
    let main = &parts[&root];
    ensure!(!main.build.is_empty(), Empty, "this 3MF builds nothing");
    let unit = main.unit.clone().unwrap_or_else(|| "millimeter".to_owned());
    let Some(scale) = meters(Some(&unit)) else {
        bail!(Corrupt, "unknown 3MF unit {unit:?}")
    };

    let mut out = Emit {
        parts: &parts,
        slicer: &slicer,
        looks: HashMap::new(),
        order: Vec::new(),
        names: HashMap::new(),
        textures: Vec::new(),
        distinct: HashSet::new(),
        b: Builder::default(),
        meshes: Vec::new(),
        mesh_triangles: Vec::new(),
        mesh_of: HashMap::new(),
        nodes: Vec::new(),
        path: Vec::new(),
        visits: 0,
        dropped: 0,
        hidden: false,
        triangles: 0,
        built: 0,
        limit: input.limit,
        lattices: HashMap::new(),
        unclipped: 0,
        levelsets: 0,
    };
    let mut items = Vec::new();
    for item in &main.build {
        let key = item.path.as_deref().map_or_else(|| root.clone(), norm);
        let filament = slicer.objects.get(&item.id).copied();
        let at = At {
            item: item.id,
            filament,
            depth: 0,
            visible: true,
        };
        if let Some(n) = out.node(&key, item.id, &item.transform, at)? {
            if let Some(pn) = &item.partnumber {
                out.nodes[n]["extras"]["itemPartnumber"] = json!(pn);
            }
            items.push(n);
        }
    }
    if out.triangles == 0 {
        ensure!(
            out.levelsets == 0,
            Unsupported,
            "this 3MF builds only volumetric level sets (shapes given by implicit functions), which have no \
             triangles to read"
        );
        bail!(Empty, "this 3MF has no triangles");
    }

    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(out.triangles));
    facts.insert("objects".into(), json!(main.build.len()));
    if out.distinct.len() > 1 {
        facts.insert("colors".into(), json!(out.distinct.len()));
    }
    let (beams, balls) = out.lattices.values().fold((0, 0), |(a, b), &(x, y)| (a + x, b + y));
    if beams > 0 {
        facts.insert("beams".into(), json!(beams));
    }
    if balls > 0 {
        facts.insert("balls".into(), json!(balls));
    }

    let mut images = Images::new(!keep.image_bytes, std::path::Path::new(""));
    let mut image_of: HashMap<usize, Option<usize>> = HashMap::new();
    let mut materials: Vec<Full> = Vec::new();
    for (i, look) in out.order.iter().enumerate() {
        let named = |fallback: String| out.names.get(look).cloned().unwrap_or(fallback);
        let material = match *look {
            Look::Plain => Material::plain("default", GRAY).into(),
            Look::Color(c, finish) => {
                let m = Full::from(Material::plain(&named(format!("color_{i}")), linear(c)));
                match finish {
                    Some(f) => f.full(m),
                    None => m,
                }
            }
            Look::Vertex => Material::plain("vertex_colors", [1.0; 4]).into(),
            Look::Texture(t) => {
                let image = *image_of.entry(t).or_insert_with(|| {
                    // A texture that would only be dropped isn't unzipped.
                    if !(keep.materials && keep.textures) {
                        return None;
                    }
                    let real = names.get(&out.textures[t].0)?;
                    let mut bytes = Vec::new();
                    zip.by_name(real).ok()?.take(MAX_PART).read_to_end(&mut bytes).ok()?;
                    let location = Location::Archive(real.clone());
                    images.bytes(bytes.into(), &out.textures[t].0, location, Image::png_or_jpeg)
                });
                match image {
                    Some(i) => Full {
                        base_color_tex: Some(Tex {
                            wrap: out.textures[t].1,
                            ..Tex::new(i)
                        }),
                        ..Material::plain(&format!("texture_{t}"), [1.0; 4]).into()
                    },
                    None => Material::plain(&format!("texture_{t}"), GRAY).into(),
                }
            }
        };
        materials.push(material);
    }

    let mut info = SourceInfo {
        format: Some("3MF".into()),
        meters_per_unit: if main.unit.is_some() {
            Fact::Declared(scale)
        } else {
            Fact::Assumed(scale)
        },
        unit: Some(unit),
        up_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        ..SourceInfo::default()
    };
    info.converted.units = (scale - 1.0).abs() > f64::EPSILON;
    info.converted.up_axis = true;
    let mut source = json!({});
    let mut metadata = Map::new();
    for (k, v) in &main.metadata {
        let slot = match k.to_ascii_lowercase().as_str() {
            "title" => &mut info.title,
            "designer" => &mut info.author,
            "description" => &mut info.description,
            "copyright" => &mut info.copyright,
            "licenseterms" => &mut info.license,
            "creationdate" => &mut info.created,
            "modificationdate" => &mut info.modified,
            "application" => &mut info.tool,
            _ => {
                metadata.insert(k.clone(), json!(v));
                continue;
            }
        };
        if slot.is_none() && !v.is_empty() {
            *slot = Some(v.clone());
        } else {
            metadata.insert(k.clone(), json!(v));
        }
    }
    if !metadata.is_empty() {
        source["metadata"] = Value::Object(metadata);
    }
    if !slicer.filaments.is_empty() {
        let hex: Vec<String> = slicer
            .filaments
            .iter()
            .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
            .collect();
        source["filaments"] = json!(hex);
    }
    if out.dropped > 0 {
        source["droppedComponents"] = json!(out.dropped);
    }
    if out.unclipped > 0 {
        source["unclippedLattices"] = json!(out.unclipped);
    }
    if let Value::Object(m) = &mut source {
        info.write(m);
    }
    let Emit {
        mut b,
        meshes,
        mut nodes,
        hidden,
        ..
    } = out;
    // Z-up to Y-up (a quarter turn about X), and the unit to meters.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    nodes.push(
        json!({ "name": "RootNode", "children": items, "rotation": [-h, 0.0, 0.0, h], "scale": [scale, scale, scale] }),
    );
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [nodes.len() - 1] }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    if hidden {
        json.insert("extensionsUsed".into(), json!(["KHR_node_visibility"]));
    }
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, facts))
}

/// How a triangle looks, which decides its primitive.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Look {
    Plain,
    Color([u8; 4], Option<Finish>),
    /// Colors blended across it from its corners.
    Vertex,
    /// Textured: an index into `Emit::textures`.
    Texture(usize),
}

/// The build's objects as nodes, their triangles as meshes with a primitive per look.
struct Emit<'a> {
    parts: &'a HashMap<String, Part>,
    slicer: &'a Slicer,
    /// Each look's material, and the looks in that order.
    looks: HashMap<Look, usize>,
    order: Vec<Look>,
    /// Base material names, for the looks that come from one.
    names: HashMap<Look, String>,
    /// The zip parts of the textures used, and their wraps.
    textures: Vec<(String, [u32; 2])>,
    /// The colors shown, for the facts.
    distinct: HashSet<[u8; 3]>,
    b: Builder,
    meshes: Vec<Value>,
    mesh_triangles: Vec<usize>,
    /// The mesh of an object (model file, id) in a filament, at the top of an item or not, and for one slicer
    /// volume's triangles only or not.
    mesh_of: HashMap<MeshKey, Option<usize>>,
    nodes: Vec<Value>,
    /// The objects being placed, outermost first, so a component that leads back to one is dropped.
    path: Vec<(String, u32)>,
    /// Objects visited, against a build that fans out without end.
    visits: usize,
    /// Components and slicer volumes left out: cyclic, nested too deep, or past the most nodes or volumes.
    dropped: usize,
    /// Whether any node is hidden (a slicer's modifier, negative volume or support part).
    hidden: bool,
    /// Triangles drawn, every visible placement counted.
    triangles: usize,
    /// Triangles in meshes.
    built: usize,
    /// The most triangles the meshes may have.
    limit: usize,
    /// The beams and balls drawn of each object (model file, id) with a lattice.
    lattices: HashMap<(String, u32), (usize, usize)>,
    /// Lattices to be clipped by a mesh that isn't there, drawn whole.
    unclipped: usize,
    /// Level sets placed, which draw nothing.
    levelsets: usize,
}

/// An object's mesh as [`Emit::mesh_of_object`] is asked for it: model file, object id, filament, whether at the
/// top of an item, and the slicer volume's triangles.
type MeshKey = (String, u32, Option<usize>, bool, Option<(u32, u32)>);

/// How deep components may nest.
const MAX_DEPTH: u32 = 16;
/// The most nodes a build may place.
const MAX_NODES: usize = 1 << 20;
/// The most volumes of an object kept as nodes of their own: each is a pass over the object's vertices.
const MAX_VOLUMES: usize = 4096;
/// The most objects a build may visit, placed or not.
const MAX_VISITS: usize = 1 << 24;
/// The most triangles an object's beam lattice is made into, while the limit allows: its curves are coarsened to
/// fit.
const MAX_LATTICE_TRIANGLES: usize = 1 << 22;

/// Where a node is placed: inside the build item for object `item` (the one slicers' settings are kept against),
/// printing in `filament` unless its parts or triangles say otherwise, `depth` components down, and whether it's
/// shown (a slicer's modifier, negative volume or support part, and everything in it, isn't).
#[derive(Clone, Copy)]
struct At {
    item: u32,
    filament: Option<usize>,
    depth: u32,
    visible: bool,
}

/// A vertex of an object as first used in a mesh: its primitive (`u32::MAX`: not yet), its index there, and the
/// texture coordinates or color it was used with.
#[derive(Clone, Copy)]
struct Slot {
    prim: u32,
    index: u32,
    attr: u64,
}

impl Emit<'_> {
    /// The node for object `id` of model `key` placed by `m`; `None` when the object isn't there, or when placing
    /// it would nest too deep or make too many nodes, which leaves it out rather than failing the load.
    fn node(&mut self, key: &str, id: u32, m: &Mat, at: At) -> Result<Option<usize>> {
        self.visits += 1;
        if at.depth >= MAX_DEPTH || self.nodes.len() >= MAX_NODES || self.visits > MAX_VISITS {
            self.dropped += 1;
            return Ok(None);
        }
        let parts = self.parts;
        let Some(o) = parts.get(key).and_then(|p| p.objects.get(&id)) else {
            return Ok(None);
        };
        let mut node = Map::new();
        if let Some(name) = &o.name {
            node.insert("name".into(), json!(name));
        }
        #[allow(clippy::float_cmp_const, reason = "only the exact identity is left out")]
        if *m != IDENTITY {
            // Row vectors, so the rows are glTF's columns.
            let g = [
                m[0], m[1], m[2], 0.0, m[3], m[4], m[5], 0.0, m[6], m[7], m[8], 0.0, m[9], m[10], m[11], 1.0,
            ];
            node.insert("matrix".into(), json!(g));
        }
        let mut extras = json!({ "objectId": id });
        if let Some(k) = &o.kind {
            extras["type"] = json!(k);
        }
        if let Some(pn) = &o.partnumber {
            extras["partnumber"] = json!(pn);
        }
        node.insert("extras".into(), extras);
        let mut children = Vec::new();
        self.path.push((key.to_owned(), id));
        for c in &o.components {
            let sub = c.path.as_deref().map_or_else(|| key.to_owned(), norm);
            if self.path.iter().any(|(k, i)| *i == c.id && *k == sub) {
                self.dropped += 1;
                continue;
            }
            let assigned = self.slicer.parts.get(&(at.item, c.id));
            let role = assigned.map_or(Role::Normal, |a| a.role);
            let inner = At {
                filament: assigned.and_then(|a| a.extruder).or(at.filament),
                depth: at.depth + 1,
                visible: at.visible && role == Role::Normal,
                ..at
            };
            if let Some(n) = self.node(&sub, c.id, &c.transform, inner)? {
                if let Some(a) = assigned {
                    self.name_part(n, a);
                }
                children.push(n);
            }
        }
        self.path.pop();
        if o.levelset && o.tris.is_empty() && o.lattice.is_none() {
            self.levelsets += 1;
        }
        if !o.tris.is_empty() || o.lattice.is_some() {
            let top = at.depth == 0;
            if let Some(mesh) = self.mesh_of_object(key, o, id, at.filament, top, None)? {
                node.insert("mesh".into(), json!(mesh));
                if at.visible {
                    self.triangles += self.mesh_triangles[mesh];
                }
            }
            // A slicer's modifiers, negative volumes and support parts that are ranges of the object's triangles:
            // each a hidden node of its own.
            let volumes = if top { self.slicer.volumes.get(&id) } else { None };
            let unprinted = volumes.into_iter().flatten().filter(|v| v.a.role != Role::Normal);
            for (k, v) in unprinted.enumerate() {
                if self.nodes.len() >= MAX_NODES || k >= MAX_VOLUMES {
                    self.dropped += 1;
                    break;
                }
                let Some(mesh) = self.mesh_of_object(key, o, id, at.filament, top, Some((v.first, v.last)))? else {
                    continue;
                };
                let name = v.a.name.as_deref().or(o.name.as_deref()).unwrap_or("volume");
                self.nodes.push(json!({
                    "name": format!("{name} ({})", v.a.role.label()),
                    "mesh": mesh,
                    "extensions": { "KHR_node_visibility": { "visible": false } },
                    "extras": { "volumeType": v.a.role.key(), "triangles": [v.first, v.last] },
                }));
                self.hidden = true;
                children.push(self.nodes.len() - 1);
            }
        }
        if !children.is_empty() {
            node.insert("children".into(), json!(children));
        }
        self.nodes.push(Value::Object(node));
        Ok(Some(self.nodes.len() - 1))
    }

    /// Names node `n`, a part, as its slicer's settings do; one that isn't printed (a modifier, negative volume or
    /// support part) is hidden, and its name says what it is.
    fn name_part(&mut self, n: usize, a: &Assigned) {
        let node = &mut self.nodes[n];
        let name = a
            .name
            .clone()
            .or_else(|| node.get("name").and_then(Value::as_str).map(str::to_owned));
        if a.role == Role::Normal {
            if let Some(name) = name {
                node["name"] = json!(name);
            }
            return;
        }
        node["name"] = json!(format!("{} ({})", name.as_deref().unwrap_or("part"), a.role.label()));
        node["extensions"] = json!({ "KHR_node_visibility": { "visible": false } });
        node["extras"]["volumeType"] = json!(a.role.key());
        self.hidden = true;
    }

    /// [`Emit::mesh`], made once for each way it's asked for.
    fn mesh_of_object(
        &mut self,
        key: &str,
        o: &Object,
        id: u32,
        filament: Option<usize>,
        top: bool,
        range: Option<(u32, u32)>,
    ) -> Result<Option<usize>> {
        let at = (key.to_owned(), id, filament, top, range);
        if let Some(&mesh) = self.mesh_of.get(&at) {
            return Ok(mesh);
        }
        let part = self.parts.get(key).context(ErrorKind::Corrupt, "model file")?;
        let mesh = self.mesh(part, key, o, id, filament, top, range)?;
        self.mesh_of.insert(at, mesh);
        Ok(mesh)
    }

    /// Object `id`'s triangles, in its own space, as a mesh with a primitive per look; `top` when it is a build
    /// item's own object, which slicers' volumes divide. With `range`, only those triangles (a volume that isn't
    /// printed); without, every triangle but those of such volumes.
    ///
    /// Corners share a vertex wherever the object's triangles do, unless their texture coordinates or colors
    /// differ. There are no normals: printed models are faceted, which glTF clients shade flat.
    #[allow(clippy::too_many_arguments)]
    fn mesh(
        &mut self,
        part: &Part,
        key: &str,
        o: &Object,
        id: u32,
        filament: Option<usize>,
        top: bool,
        range: Option<(u32, u32)>,
    ) -> Result<Option<usize>> {
        let volumes = if top { self.slicer.volumes.get(&id) } else { None }.map_or(&[][..], Vec::as_slice);
        let mut prims: Vec<Prim> = Vec::new();
        // Each material's primitive.
        let mut prim_of: Vec<u32> = Vec::new();
        let mut slots = vec![
            Slot {
                prim: u32::MAX,
                index: 0,
                attr: 0
            };
            o.verts.len()
        ];
        // A vertex used again in another primitive, or with other texture coordinates or colors.
        let mut more: HashMap<(u64, u64), u32, Fast> = HashMap::default();
        let (first, last) = range.map_or((0, o.tris.len()), |(a, b)| {
            (a as usize, (b as usize).saturating_add(1).min(o.tris.len()))
        });
        let mut count = 0;
        for (i, t) in o.tris.iter().enumerate().take(last).skip(first) {
            ensure!(
                self.built < self.limit,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                self.limit
            );
            let [Some(a), Some(b), Some(c)] = t.v.map(|k| o.verts.get(k as usize)) else {
                continue;
            };
            let volume = volume_of(volumes, u32::try_from(i).unwrap_or(u32::MAX));
            if range.is_none() && volume.is_some_and(|v| v.a.role != Role::Normal) {
                continue;
            }
            let filament = if t.paint > 0 {
                Some(usize::from(t.paint))
            } else {
                volume.and_then(|v| v.a.extruder).or(filament)
            };
            let (look, uvs, colors) = match filament {
                Some(n) => {
                    let [r, g, b] = self.slicer.filament(n);
                    (Look::Color([r, g, b, 255], None), None, None)
                }
                None => self.properties(part, t.pid.or(o.pid), [t.p1.or(o.pindex), t.p2, t.p3]),
            };
            self.note(look, colors.as_ref());
            let material = self.material(look);
            if prim_of.len() <= material {
                prim_of.resize(material + 1, u32::MAX);
            }
            if prim_of[material] == u32::MAX {
                prim_of[material] =
                    u32::try_from(prims.len()).context(ErrorKind::LimitExceeded, "too many primitives")?;
                prims.push(Prim {
                    material,
                    ..Prim::default()
                });
            }
            let pi = prim_of[material];
            let p = &mut prims[pi as usize];
            let pos = [*a, *b, *c];
            for k in 0..3 {
                let attr = match (&uvs, &colors) {
                    (Some(uv), _) => u64::from(uv[k][0].to_bits()) | u64::from(uv[k][1].to_bits()) << 32,
                    (_, Some(cs)) => u64::from(u32::from_le_bytes(cs[k])),
                    _ => 0,
                };
                let v = t.v[k] as usize;
                let slot = slots[v];
                let found = if slot.prim == pi && slot.attr == attr {
                    Some(slot.index)
                } else if slot.prim == u32::MAX {
                    None
                } else {
                    more.get(&((v as u64) << 32 | u64::from(pi), attr)).copied()
                };
                let index = if let Some(index) = found {
                    index
                } else {
                    let index = u32::try_from(p.positions.len())
                        .context(ErrorKind::LimitExceeded, "too many vertices for 32-bit indices")?;
                    p.positions.push(pos[k]);
                    if let Some(uv) = &uvs {
                        p.uvs.push([uv[k][0], 1.0 - uv[k][1]]);
                    }
                    if let Some(cs) = &colors {
                        p.colors.push(linear(cs[k]));
                    }
                    if slot.prim == u32::MAX {
                        slots[v] = Slot { prim: pi, index, attr };
                    } else {
                        more.insert(((v as u64) << 32 | u64::from(pi), attr), index);
                    }
                    index
                };
                p.indices.push(index);
            }
            count += 1;
            self.built += 1;
        }
        let mut extras = None;
        if range.is_none()
            && let Some(l) = o.lattice.as_deref()
        {
            let (made, beams, balls) = self.lattice(part, o, id, l, filament, &mut prims);
            count += made;
            self.built += made;
            ensure!(
                self.built <= self.limit,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                self.limit
            );
            self.lattices.insert((key.to_owned(), id), (beams, balls));
            let mut e = json!({ "beams": beams });
            if balls > 0 {
                e["balls"] = json!(balls);
            }
            if !l.sets.is_empty() {
                let sets: Vec<Value> = l
                    .sets
                    .iter()
                    .map(|s| json!({ "name": s.name, "identifier": s.identifier, "beams": s.beams, "balls": s.balls }))
                    .collect();
                e["beamsets"] = json!(sets);
            }
            extras = Some(e);
        }
        // Materials are numbered as looks turn up, so every one used is already there.
        let Some(mut mesh) = self.b.mesh(prims, usize::MAX) else {
            return Ok(None);
        };
        if let Some(name) = &o.name {
            mesh["name"] = json!(name);
        }
        if let Some(e) = extras {
            mesh["extras"] = e;
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(count);
        Ok(Some(self.meshes.len() - 1))
    }

    /// The material of `look`, numbered as looks turn up.
    fn material(&mut self, look: Look) -> usize {
        *self.looks.entry(look).or_insert_with(|| {
            self.order.push(look);
            self.order.len() - 1
        })
    }

    /// Counts the colors `look` shows.
    fn note(&mut self, look: Look, colors: Option<&[[u8; 4]; 3]>) {
        match look {
            Look::Color(c, _) => {
                self.distinct.insert([c[0], c[1], c[2]]);
            }
            Look::Vertex => {
                for c in colors.into_iter().flatten() {
                    self.distinct.insert([c[0], c[1], c[2]]);
                }
            }
            _ => {}
        }
    }

    /// The primitive of `look` among a lattice's, `prim_of` each look's so far.
    fn lattice_prim(&mut self, prims: &mut Vec<Prim>, prim_of: &mut HashMap<Look, usize>, look: Look) -> usize {
        let material = self.material(look);
        *prim_of.entry(look).or_insert_with(|| {
            prims.push(Prim {
                material,
                ..Prim::default()
            });
            prims.len() - 1
        })
    }

    /// A lattice's look, in property group `pid` from entry `p[0]` to `p[1]`, and the attributes at either end;
    /// for colors that blend, each end's color too. An object's filament overrides it.
    fn lattice_look(
        &mut self,
        part: &Part,
        filament: Option<usize>,
        pid: Option<u32>,
        p: [Option<u32>; 2],
    ) -> (Look, [Attr; 2], [[u8; 4]; 2]) {
        let (look, uvs, colors) = match filament {
            Some(n) => {
                let [r, g, b] = self.slicer.filament(n);
                (Look::Color([r, g, b, 255], None), None, None)
            }
            None => self.properties(part, pid, [p[0], p[1], p[1]]),
        };
        self.note(look, colors.as_ref());
        match (look, uvs, colors) {
            (Look::Texture(_), Some(uv), _) => {
                let at = |k: usize| Attr::Uv([uv[k][0], 1.0 - uv[k][1]]);
                (look, [at(0), at(1)], [[0; 4]; 2])
            }
            (Look::Vertex, _, Some(cs)) => (
                look,
                [Attr::Color(linear(cs[0])), Attr::Color(linear(cs[1]))],
                [cs[0], cs[1]],
            ),
            _ => (look, [Attr::None; 2], [[0; 4]; 2]),
        }
    }

    /// Object `o`'s beam lattice made into triangles, in primitives added to `prims` (with normals, so apart from
    /// the faceted triangles'), clipped by its clipping mesh. Returns the triangles, and the beams and balls drawn.
    ///
    /// A beam's properties are its own, else the lattice's, else the object's, the second end's defaulting to the
    /// first's; colors that differ blend along it, and a display property (which can't blend) is the first end's
    /// throughout. A sphere at a vertex takes the properties of the end or ball that gave it its radius.
    fn lattice(
        &mut self,
        part: &Part,
        o: &Object,
        id: u32,
        l: &Lattice,
        filament: Option<usize>,
        prims: &mut Vec<Prim>,
    ) -> (usize, usize, usize) {
        let plan = lattice::plan(l, &o.verts);
        if plan.beams.is_empty() && plan.spheres.is_empty() {
            return (0, 0, 0);
        }
        let first = prims.len();
        let mut prim_of: HashMap<Look, usize> = HashMap::new();
        let mut looks = Vec::with_capacity(plan.beams.len());
        for &i in &plan.beams {
            let b = &l.beams[i];
            let pid = b.pid.or(l.pid).or(o.pid);
            let p1 = b.p[0].or(l.pindex).or(o.pindex);
            let p2 = if pid.is_some_and(|g| part.group_display.contains_key(&g)) {
                None
            } else {
                b.p[1]
            };
            let (look, side_attr, colors) = self.lattice_look(part, filament, pid, [p1, p2]);
            let side = self.lattice_prim(prims, &mut prim_of, look);
            // Where colors blend along the side, each end's cap is its end's color.
            let ends = if look == Look::Vertex {
                colors.map(|c| (self.lattice_prim(prims, &mut prim_of, Look::Color(c, None)), Attr::None))
            } else {
                [(side, side_attr[0]), (side, side_attr[1])]
            };
            looks.push(BeamLook { side, side_attr, ends });
        }
        let mut sphere_looks = Vec::with_capacity(plan.spheres.len());
        for s in &plan.spheres {
            let (pid, p) = match s.from {
                lattice::From::End(k, end) => {
                    sphere_looks.push(looks[k].ends[end]);
                    continue;
                }
                lattice::From::Ball(k) => {
                    let ball = &l.balls[k];
                    (ball.pid.or(l.pid).or(o.pid), ball.p.or(l.pindex).or(o.pindex))
                }
                lattice::From::Lattice => (l.pid.or(o.pid), l.pindex.or(o.pindex)),
            };
            let (look, attr, colors) = self.lattice_look(part, filament, pid, [p, None]);
            let look = if look == Look::Vertex {
                Look::Color(colors[0], None)
            } else {
                look
            };
            sphere_looks.push((self.lattice_prim(prims, &mut prim_of, look), attr[0]));
        }
        let budget = self.limit.saturating_sub(self.built).min(MAX_LATTICE_TRIANGLES);
        let mut made = lattice::build(l, &o.verts, &plan, &looks, &sphere_looks, prims, budget);
        if l.clip != Clip::None {
            // The clipping mesh is in the lattice's own space; it can't be the lattice's own object.
            let clipping = l.clipping_mesh.filter(|&c| c != id).and_then(|c| part.objects.get(&c));
            match clipping.and_then(|m| Clipper::new(&m.verts, m.tris.iter().map(|t| t.v))) {
                Some(clipper) => {
                    // What refining the cut may add comes out of what's left of the budget.
                    let mut room = budget.saturating_sub(made);
                    made = 0;
                    for p in &mut prims[first..] {
                        let before = p.indices.len() / 3;
                        let after = lattice::clip(p, &clipper, l.clip == Clip::Inside, room);
                        room = room.saturating_sub(after.saturating_sub(before));
                        made += after;
                    }
                }
                None => self.unclipped += 1,
            }
        }
        (made, plan.beams.len(), plan.balls)
    }

    /// A look from the materials extension: property group `pid` and the entry for each corner (`p[0]`, the
    /// second and third defaulting to the first), as already defaulted to the object's.
    #[allow(clippy::type_complexity)]
    fn properties(
        &mut self,
        part: &Part,
        pid: Option<u32>,
        p: [Option<u32>; 3],
    ) -> (Look, Option<[[f32; 2]; 3]>, Option<[[u8; 4]; 3]>) {
        let plain = (Look::Plain, None, None);
        let Some((gid, group)) = pid.and_then(|g| Some((g, part.groups.get(&g)?))) else {
            return plain;
        };
        let p1 = p[0].unwrap_or(0) as usize;
        let at = [p1, p[1].map_or(p1, |v| v as usize), p[2].map_or(p1, |v| v as usize)];
        match group {
            Group::Colors(cs) => {
                let c = at.map(|i| cs.get(i).and_then(|e| e.0));
                if c[0] == c[1] && c[1] == c[2] {
                    let finish = part
                        .group_display
                        .get(&gid)
                        .and_then(|d| part.displays.get(d))
                        .and_then(|d| d.get(p1).copied());
                    let look = c[0].map_or(Look::Plain, |c| Look::Color(c, finish));
                    if let Some(name) = cs.get(p1).and_then(|e| e.1.as_ref())
                        && look != Look::Plain
                    {
                        self.names.entry(look).or_insert_with(|| name.clone());
                    }
                    (look, None, None)
                } else {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a constant in 0..=1"
                    )]
                    let gray = GRAY.map(|v| (v * 255.0) as u8);
                    (Look::Vertex, None, Some(c.map(|c| c.unwrap_or(gray))))
                }
            }
            // Mixed or layered into the others when read.
            Group::Composite { .. } | Group::Multi { .. } => plain,
            Group::Coords { texture, uvs } => {
                let (Some(path), [Some(a), Some(b), Some(c)]) =
                    (part.textures.get(texture), at.map(|i| uvs.get(i).copied()))
                else {
                    return plain;
                };
                let key = (norm(&path.0), path.1);
                let t = if let Some(t) = self.textures.iter().position(|k| *k == key) {
                    t
                } else {
                    self.textures.push(key);
                    self.textures.len() - 1
                };
                (Look::Texture(t), Some([a, b, c]), None)
            }
        }
    }
}

/// A zip entry name as looked up: no leading slash, lower case (3MF part names aren't case-sensitive).
fn norm(name: &str) -> String {
    name.trim_start_matches('/').to_ascii_lowercase()
}

fn text(zip: &mut Zip<'_>, names: &HashMap<String, String>, name: &str, limit: u64) -> Option<String> {
    let mut s = String::new();
    zip.by_name(names.get(&norm(name))?)
        .ok()?
        .take(limit)
        .read_to_string(&mut s)
        .ok()?;
    Some(s)
}

fn each_attr(e: &BytesStart<'_>, mut f: impl FnMut(&[u8], &[u8])) {
    for a in e.attributes().with_checks(false).flatten() {
        f(a.key.local_name().as_ref(), &a.value);
    }
}

fn attr(e: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    let mut out = None;
    each_attr(e, |k, v| {
        if k == name {
            out = Some(String::from_utf8_lossy(v).into_owned());
        }
    });
    out
}

fn num<T: std::str::FromStr>(v: &[u8]) -> Option<T> {
    std::str::from_utf8(v).ok()?.trim().parse().ok()
}

fn transform(v: &[u8]) -> Mat {
    let n: Vec<f32> = std::str::from_utf8(v)
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    n.try_into().unwrap_or(IDENTITY)
}

/// `#RRGGBB` or `#RRGGBBAA`.
fn hex_rgba(s: &str) -> Option<[u8; 4]> {
    let h = s.trim().strip_prefix('#')?;
    if h.len() != 6 && h.len() != 8 {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    let a = if h.len() == 8 { byte(6)? } else { 255 };
    Some([byte(0)?, byte(2)?, byte(4)?, a])
}

fn hex_color(s: &str) -> Option<[u8; 3]> {
    hex_rgba(s).map(|[r, g, b, _]| [r, g, b])
}

/// The root model's name, from the package's relationships.
fn root_target(rels: &str) -> Option<String> {
    let mut reader = quick_xml::Reader::from_str(rels);
    loop {
        match reader.read_event() {
            Ok(Event::Start(e) | Event::Empty(e)) if e.local_name().as_ref() == b"Relationship" => {
                if attr(&e, b"Type").is_some_and(|t| t.ends_with("/3dmodel")) {
                    return attr(&e, b"Target");
                }
            }
            Ok(Event::Eof) | Err(_) => return None,
            _ => {}
        }
    }
}

fn parse_part(r: impl Read, triangles: &mut usize, limit: usize) -> Result<Part> {
    let mut reader = quick_xml::Reader::from_reader(crate::formats::xml::Utf8Read::new(BufReader::new(r))?);
    let mut buf = Vec::new();
    let mut part = Part::default();
    let mut object: Option<(u32, Object)> = None;
    let mut group: Option<u32> = None;
    // The display properties whose entries are being read.
    let mut display: Option<u32> = None;
    let mut in_build = false;
    // The `<metadata name>` whose text is being read.
    let mut meta: Option<String> = None;
    // Inside an object's `<beamlattice>`, and inside one of its `<beamset>`s.
    let (mut in_lattice, mut in_set) = (false, false);
    loop {
        let (e, empty) = match reader
            .read_event_into(&mut buf)
            .context(ErrorKind::Corrupt, "bad 3MF XML")?
        {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::Text(t) => {
                if let Some(name) = meta.take() {
                    let text = String::from_utf8_lossy(&t).trim().to_owned();
                    part.metadata.push((name, text));
                }
                buf.clear();
                continue;
            }
            Event::End(e) => {
                match e.local_name().as_ref() {
                    b"object" => {
                        if let Some((id, o)) = object.take() {
                            part.objects.insert(id, o);
                        }
                    }
                    b"basematerials"
                    | b"colorgroup"
                    | b"texture2dgroup"
                    | b"compositematerials"
                    | b"multiproperties" => {
                        group = None;
                    }
                    b"pbmetallicdisplayproperties"
                    | b"pbspeculardisplayproperties"
                    | b"translucentdisplayproperties" => {
                        display = None;
                    }
                    b"build" => in_build = false,
                    b"metadata" => meta = None,
                    b"beamlattice" => in_lattice = false,
                    b"beamset" => in_set = false,
                    _ => {}
                }
                buf.clear();
                continue;
            }
            Event::Eof => break,
            _ => {
                buf.clear();
                continue;
            }
        };
        match e.local_name().as_ref() {
            b"model" => part.unit = attr(&e, b"unit"),
            b"object" => {
                let mut o = Object::default();
                let mut id = None;
                each_attr(&e, |k, v| match k {
                    b"id" => id = num(v),
                    b"pid" => o.pid = num(v),
                    b"pindex" => o.pindex = num(v),
                    b"name" => o.name = Some(String::from_utf8_lossy(v).into_owned()),
                    b"type" => o.kind = Some(String::from_utf8_lossy(v).into_owned()),
                    b"partnumber" => o.partnumber = Some(String::from_utf8_lossy(v).into_owned()),
                    _ => {}
                });
                if let Some(id) = id {
                    if empty {
                        part.objects.insert(id, o);
                    } else {
                        object = Some((id, o));
                    }
                }
            }
            b"vertex" => {
                if let Some((_, o)) = &mut object {
                    let mut p = [0.0f32; 3];
                    each_attr(&e, |k, v| match k {
                        b"x" => p[0] = num(v).unwrap_or(0.0),
                        b"y" => p[1] = num(v).unwrap_or(0.0),
                        b"z" => p[2] = num(v).unwrap_or(0.0),
                        _ => {}
                    });
                    o.verts.push(p);
                }
            }
            b"triangle" => {
                if let Some((_, o)) = &mut object {
                    let mut t = Tri {
                        v: [u32::MAX; 3],
                        ..Tri::default()
                    };
                    each_attr(&e, |k, v| match k {
                        b"v1" => t.v[0] = num(v).unwrap_or(u32::MAX),
                        b"v2" => t.v[1] = num(v).unwrap_or(u32::MAX),
                        b"v3" => t.v[2] = num(v).unwrap_or(u32::MAX),
                        b"pid" => t.pid = num(v),
                        b"p1" => t.p1 = num(v),
                        b"p2" => t.p2 = num(v),
                        b"p3" => t.p3 = num(v),
                        // Bambu Studio and OrcaSlicer; PrusaSlicer (`slic3rpe:`).
                        b"paint_color" | b"mmu_segmentation" => t.paint = paint_state(v),
                        _ => {}
                    });
                    *triangles += 1;
                    ensure!(
                        *triangles <= limit,
                        LimitExceeded,
                        "too many triangles to preview (more than {limit})"
                    );
                    o.tris.push(t);
                }
            }
            b"beamlattice" => {
                if let Some((_, o)) = &mut object {
                    let mut l = Lattice::default();
                    each_attr(&e, |k, v| match k {
                        b"minlength" => l.minlength = num(v).unwrap_or(0.0),
                        b"radius" => l.radius = num(v).unwrap_or(f32::NAN),
                        b"cap" => l.cap = Cap::parse(v).unwrap_or(Cap::Sphere),
                        b"ballmode" => {
                            l.ball_mode = match v.trim_ascii() {
                                b"all" => BallMode::All,
                                b"mixed" => BallMode::Mixed,
                                _ => BallMode::None,
                            };
                        }
                        b"ballradius" => l.ball_radius = num(v).unwrap_or(f32::NAN),
                        // `clipping` is what drafts of the extension called it.
                        b"clippingmode" | b"clipping" => {
                            l.clip = match v.trim_ascii() {
                                b"inside" => Clip::Inside,
                                b"outside" => Clip::Outside,
                                _ => Clip::None,
                            };
                        }
                        b"clippingmesh" => l.clipping_mesh = num(v),
                        b"pid" => l.pid = num(v),
                        b"pindex" => l.pindex = num(v),
                        _ => {}
                    });
                    o.lattice = Some(Box::new(l));
                    in_lattice = !empty;
                }
            }
            b"beam" if in_lattice => {
                if let Some(l) = object.as_mut().and_then(|(_, o)| o.lattice.as_deref_mut()) {
                    let (mut v, mut r, mut cap, mut pid, mut p) =
                        ([u32::MAX; 2], [None; 2], [None; 2], None, [None; 2]);
                    each_attr(&e, |k, x| match k {
                        b"v1" => v[0] = num(x).unwrap_or(u32::MAX),
                        b"v2" => v[1] = num(x).unwrap_or(u32::MAX),
                        b"r1" => r[0] = num::<f32>(x),
                        b"r2" => r[1] = num::<f32>(x),
                        b"cap1" => cap[0] = Cap::parse(x),
                        b"cap2" => cap[1] = Cap::parse(x),
                        b"pid" => pid = num(x),
                        b"p1" => p[0] = num(x),
                        b"p2" => p[1] = num(x),
                        _ => {}
                    });
                    let r1 = r[0].unwrap_or(l.radius);
                    *triangles += 1;
                    ensure!(
                        *triangles <= limit,
                        LimitExceeded,
                        "too many beams to preview (more than {limit})"
                    );
                    l.beams.push(lattice::Beam {
                        v,
                        r: [r1, r[1].unwrap_or(r1)],
                        cap: cap.map(|c| c.unwrap_or(l.cap)),
                        pid,
                        p,
                    });
                }
            }
            b"ball" if in_lattice => {
                if let Some(l) = object.as_mut().and_then(|(_, o)| o.lattice.as_deref_mut()) {
                    let (mut v, mut r, mut pid, mut p) = (u32::MAX, None, None, None);
                    each_attr(&e, |k, x| match k {
                        b"vindex" => v = num(x).unwrap_or(u32::MAX),
                        b"r" => r = num::<f32>(x),
                        b"pid" => pid = num(x),
                        b"p" => p = num(x),
                        _ => {}
                    });
                    *triangles += 1;
                    ensure!(
                        *triangles <= limit,
                        LimitExceeded,
                        "too many balls to preview (more than {limit})"
                    );
                    l.balls.push(lattice::Ball {
                        v,
                        r: r.unwrap_or(l.ball_radius),
                        pid,
                        p,
                    });
                }
            }
            b"beamset" if in_lattice => {
                if let Some(l) = object.as_mut().and_then(|(_, o)| o.lattice.as_deref_mut()) {
                    l.sets.push(lattice::BeamSet {
                        name: attr(&e, b"name"),
                        identifier: attr(&e, b"identifier"),
                        ..lattice::BeamSet::default()
                    });
                    in_set = !empty;
                }
            }
            kind @ (b"ref" | b"ballref") if in_set => {
                if let Some(set) = object
                    .as_mut()
                    .and_then(|(_, o)| o.lattice.as_deref_mut())
                    .and_then(|l| l.sets.last_mut())
                {
                    if kind == b"ref" {
                        set.beams += 1;
                    } else {
                        set.balls += 1;
                    }
                }
            }
            b"levelset" => {
                if let Some((_, o)) = &mut object {
                    o.levelset = true;
                }
            }
            b"component" => {
                if let Some((_, o)) = &mut object {
                    let mut c = Component {
                        path: None,
                        id: u32::MAX,
                        transform: IDENTITY,
                    };
                    each_attr(&e, |k, v| match k {
                        b"objectid" => c.id = num(v).unwrap_or(u32::MAX),
                        b"transform" => c.transform = transform(v),
                        b"path" => c.path = Some(String::from_utf8_lossy(v).into_owned()),
                        _ => {}
                    });
                    o.components.push(c);
                }
            }
            kind @ (b"basematerials" | b"colorgroup" | b"texture2dgroup") => {
                group = attr(&e, b"id").and_then(|v| v.parse().ok());
                if let (Some(g), Some(d)) = (group, attr(&e, b"displaypropertiesid").and_then(|v| v.parse().ok())) {
                    part.group_display.insert(g, d);
                }
                if let Some(g) = group {
                    let new = if kind == b"texture2dgroup" {
                        let texture = attr(&e, b"texid").and_then(|v| v.parse().ok());
                        Group::Coords {
                            texture: texture.unwrap_or(u32::MAX),
                            uvs: Vec::new(),
                        }
                    } else {
                        Group::Colors(Vec::new())
                    };
                    part.groups.insert(g, new);
                }
                if empty {
                    group = None;
                }
            }
            kind @ (b"compositematerials" | b"multiproperties") => {
                group = attr(&e, b"id").and_then(|v| v.parse().ok());
                let list = |k: &[u8]| attr(&e, k).unwrap_or_default();
                let ids = |k: &[u8]| {
                    list(k)
                        .split_whitespace()
                        .map(|w| w.parse().unwrap_or(u32::MAX))
                        .collect()
                };
                if let Some(g) = group {
                    let new = if kind == b"compositematerials" {
                        if let Some(d) = attr(&e, b"displaypropertiesid").and_then(|v| v.parse().ok()) {
                            part.group_display.insert(g, d);
                        }
                        Group::Composite {
                            base: attr(&e, b"matid").and_then(|v| v.parse().ok()).unwrap_or(u32::MAX),
                            indices: ids(b"matindices"),
                            values: Vec::new(),
                            starts: Vec::new(),
                        }
                    } else {
                        Group::Multi {
                            pids: ids(b"pids"),
                            multiply: list(b"blendmethods")
                                .split_whitespace()
                                .map(|w| w == "multiply")
                                .collect(),
                            indices: Vec::new(),
                            starts: Vec::new(),
                        }
                    };
                    part.groups.insert(g, new);
                }
                if empty {
                    group = None;
                }
            }
            b"composite" => {
                if let Some(Group::Composite { values, starts, .. }) = group.and_then(|g| part.groups.get_mut(&g)) {
                    starts.push(values.len());
                    let v = attr(&e, b"values").unwrap_or_default();
                    values.extend(v.split_whitespace().map(|w| w.parse::<f32>().unwrap_or(0.0)));
                }
            }
            b"multi" => {
                if let Some(Group::Multi { indices, starts, .. }) = group.and_then(|g| part.groups.get_mut(&g)) {
                    starts.push(indices.len());
                    let v = attr(&e, b"pindices").unwrap_or_default();
                    indices.extend(v.split_whitespace().map(|w| w.parse::<u32>().unwrap_or(0)));
                }
            }
            name @ (b"base" | b"color") => {
                if let Some(Group::Colors(cs)) = group.and_then(|g| part.groups.get_mut(&g)) {
                    let key: &[u8] = if name == b"base" { b"displaycolor" } else { b"color" };
                    let label = if name == b"base" { attr(&e, b"name") } else { None };
                    cs.push((attr(&e, key).and_then(|v| hex_rgba(&v)), label));
                }
            }
            b"tex2coord" => {
                if let Some(Group::Coords { uvs, .. }) = group.and_then(|g| part.groups.get_mut(&g)) {
                    let mut uv = [0.0f32; 2];
                    each_attr(&e, |k, v| match k {
                        b"u" => uv[0] = num(v).unwrap_or(0.0),
                        b"v" => uv[1] = num(v).unwrap_or(0.0),
                        _ => {}
                    });
                    uvs.push(uv);
                }
            }
            b"texture2d" => {
                if let (Some(id), Some(path)) = (attr(&e, b"id").and_then(|v| v.parse().ok()), attr(&e, b"path")) {
                    let wrap = [b"tilestyleu", b"tilestylev"].map(|k| tile_style(attr(&e, k).as_deref()));
                    part.textures.insert(id, (path, wrap));
                }
            }
            b"pbmetallicdisplayproperties" | b"pbspeculardisplayproperties" | b"translucentdisplayproperties" => {
                display = attr(&e, b"id").and_then(|v| v.parse().ok());
                if let Some(d) = display {
                    part.displays.insert(d, Vec::new());
                }
                if empty {
                    display = None;
                }
            }
            kind @ (b"pbmetallic" | b"pbspecular" | b"translucent") => {
                if let Some(list) = display.and_then(|d| part.displays.get_mut(&d)) {
                    let f = |k: &[u8], d: f32| attr(&e, k).and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(d);
                    let rgb = |k: &[u8], d: f32| {
                        let v = attr(&e, k).unwrap_or_default();
                        let mut n = v.split_whitespace().filter_map(|w| w.parse::<f32>().ok());
                        let first = n.next().unwrap_or(d);
                        [first, n.next().unwrap_or(first), n.next().unwrap_or(first)].map(f32::to_bits)
                    };
                    list.push(match kind {
                        b"pbmetallic" => Finish::Metal {
                            metallic: f(b"metallicness", 0.0).to_bits(),
                            roughness: f(b"roughness", 1.0).to_bits(),
                        },
                        b"pbspecular" => {
                            let c = attr(&e, b"specularcolor")
                                .and_then(|v| hex_rgba(&v))
                                .unwrap_or([0x38, 0x38, 0x38, 255]);
                            Finish::Specular {
                                color: [c[0], c[1], c[2]],
                                glossiness: f(b"glossiness", 0.0).to_bits(),
                            }
                        }
                        _ => Finish::Translucent {
                            attenuation: rgb(b"attenuation", 0.0),
                            ior: rgb(b"refractiveindex", 1.0),
                            roughness: f(b"roughness", 0.0).to_bits(),
                        },
                    });
                }
            }
            b"build" => in_build = !empty,
            b"item" if in_build => {
                let (mut id, mut transform_, mut path, mut partnumber) = (None, IDENTITY, None, None);
                each_attr(&e, |k, v| match k {
                    b"objectid" => id = num(v),
                    b"transform" => transform_ = transform(v),
                    b"path" => path = Some(String::from_utf8_lossy(v).into_owned()),
                    b"partnumber" => partnumber = Some(String::from_utf8_lossy(v).into_owned()),
                    _ => {}
                });
                if let Some(id) = id {
                    part.build.push(Item {
                        path,
                        id,
                        transform: transform_,
                        partnumber,
                    });
                }
            }
            // The model's own metadata, not an object's or item's (`metadatagroup`).
            b"metadata" if object.is_none() && !in_build => meta = if empty { None } else { attr(&e, b"name") },
            _ => {}
        }
        buf.clear();
    }
    if let Some((id, o)) = object {
        part.objects.insert(id, o);
    }
    resolve(&mut part.groups);
    Ok(part)
}

/// Linear RGBA of an sRGB color.
fn linear(c: [u8; 4]) -> [f32; 4] {
    let [r, g, b] = [c[0], c[1], c[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
    [r, g, b, f32::from(c[3]) / 255.0]
}

/// An sRGB color of linear RGBA.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 0..=1 first, so in 0..=255"
)]
fn srgb(c: [f32; 4]) -> [u8; 4] {
    let encode = |v: f32| {
        let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };
    let alpha = if c[3].is_nan() { 1.0 } else { c[3].clamp(0.0, 1.0) };
    [encode(c[0]), encode(c[1]), encode(c[2]), alpha].map(|v| (v * 255.0).round() as u8)
}

/// Mixes composite materials into colors, then layers multiproperties into colors or, with a texture among their
/// layers, into that texture's coordinates (the colors over it are lost).
fn resolve(groups: &mut HashMap<u32, Group>) {
    // Layers read of a multiproperties group: the materials extension expects a handful, and every entry is
    // layered from all of them.
    const MAX_LAYERS: usize = 16;
    let composites: Vec<u32> = groups
        .iter()
        .filter(|(_, g)| matches!(g, Group::Composite { .. }))
        .map(|(id, _)| *id)
        .collect();
    for id in composites {
        let Some(Group::Composite {
            base,
            indices,
            values,
            starts,
        }) = groups.remove(&id)
        else {
            continue;
        };
        let colors = match groups.get(&base) {
            Some(Group::Colors(cs)) => cs.as_slice(),
            _ => &[],
        };
        // Each composite: its base materials' linear colors, weighted by its proportions.
        let mixed = (0..starts.len())
            .map(|k| {
                let end = starts.get(k + 1).copied().unwrap_or(values.len());
                let (mut sum, mut total) = ([0.0f32; 4], 0.0f32);
                for (j, &w) in values.get(starts[k]..end).unwrap_or_default().iter().enumerate() {
                    let c = indices.get(j).and_then(|&i| colors.get(i as usize)).and_then(|e| e.0);
                    let (Some(c), true) = (c, w.is_finite() && w > 0.0) else {
                        continue;
                    };
                    for (s, l) in sum.iter_mut().zip(linear(c)) {
                        *s += l * w;
                    }
                    total += w;
                }
                ((total > 0.0).then(|| srgb(sum.map(|s| s / total))), None)
            })
            .collect();
        groups.insert(id, Group::Colors(mixed));
    }

    let multis: Vec<u32> = groups
        .iter()
        .filter(|(_, g)| matches!(g, Group::Multi { .. }))
        .map(|(id, _)| *id)
        .collect();
    for id in multis {
        let Some(Group::Multi {
            mut pids,
            multiply,
            indices,
            starts,
        }) = groups.remove(&id)
        else {
            continue;
        };
        pids.truncate(MAX_LAYERS);
        // Entry `k`'s index into layer `j`; 0 when it lists too few.
        let index = |k: usize, j: usize| {
            let end = starts.get(k + 1).copied().unwrap_or(indices.len());
            let at = starts[k].saturating_add(j);
            if at < end {
                indices.get(at).map_or(0, |&i| i as usize)
            } else {
                0
            }
        };
        let texture = pids.iter().enumerate().find_map(|(j, p)| match groups.get(p) {
            Some(Group::Coords { texture, uvs }) => Some((j, *texture, uvs)),
            _ => None,
        });
        let resolved = if let Some((j, texture, uvs)) = texture {
            let uvs = (0..starts.len())
                .map(|k| uvs.get(index(k, j)).copied().unwrap_or([0.0; 2]))
                .collect();
            Group::Coords { texture, uvs }
        } else {
            let colors = (0..starts.len())
                .map(|k| {
                    let mut out: Option<[f32; 4]> = None;
                    for (j, p) in pids.iter().enumerate() {
                        let Some(Group::Colors(cs)) = groups.get(p) else {
                            continue;
                        };
                        let Some(c) = cs.get(index(k, j)).and_then(|e| e.0).map(linear) else {
                            continue;
                        };
                        let times = j.checked_sub(1).and_then(|b| multiply.get(b)).copied().unwrap_or(false);
                        out = Some(match out {
                            None => c,
                            Some(o) if times => [o[0] * c[0], o[1] * c[1], o[2] * c[2], o[3] * c[3]],
                            // Mixed by the upper layer's alpha.
                            Some(o) => {
                                let a = c[3];
                                [
                                    o[0] * (1.0 - a) + c[0] * a,
                                    o[1] * (1.0 - a) + c[1] * a,
                                    o[2] * (1.0 - a) + c[2] * a,
                                    a + o[3] * (1.0 - a),
                                ]
                            }
                        });
                    }
                    (out.map(srgb), None)
                })
                .collect();
            Group::Colors(colors)
        };
        groups.insert(id, resolved);
    }
}

/// The filament most of a painted triangle prints in (0: the part's own). Slicers store the painting as a tree of
/// subdivisions in hex, its nibbles read from the end: the low two bits of a node say how many sides it splits (none:
/// a leaf; else into that many plus one), the high two a leaf's filament (3: the next nibble plus 3).
fn paint_state(code: &[u8]) -> u8 {
    fn hex(c: u8) -> Option<u8> {
        char::from(c).to_digit(16).and_then(|d| u8::try_from(d).ok())
    }
    fn walk(n: &impl Fn(usize) -> Option<u8>, pos: &mut usize, w: f32, area: &mut [f32; 19], depth: u32) -> Option<()> {
        let code = n(*pos)?;
        *pos += 1;
        let sides = code & 0b11;
        if sides == 0 {
            let state = if code >> 2 == 3 {
                *pos += 1;
                n(*pos - 1)? + 3
            } else {
                code >> 2
            };
            area[usize::from(state)] += w;
        } else {
            if depth > 24 {
                return None;
            }
            let children = sides + 1;
            for _ in 0..children {
                walk(n, pos, w / f32::from(children), area, depth + 1)?;
            }
        }
        Some(())
    }
    let mut area = [0.0f32; 19];
    if code.iter().all(|&c| hex(c).is_some()) {
        // Read in place, the usual case.
        walk(
            &|i| code.len().checked_sub(i + 1).and_then(|k| hex(code[k])),
            &mut 0,
            1.0,
            &mut area,
            0,
        );
    } else {
        let nibbles: Vec<u8> = code.iter().rev().filter_map(|&c| hex(c)).collect();
        walk(&|i| nibbles.get(i).copied(), &mut 0, 1.0, &mut area, 0);
    }
    let mut best = 0;
    for (i, a) in area.iter().enumerate() {
        if *a > area[best] {
            best = i;
        }
    }
    u8::try_from(best).unwrap_or(0)
}

/// Filament colors and assignments from a slicer project's settings files, when it has them.
fn read_slicer(zip: &mut Zip<'_>, names: &HashMap<String, String>) -> Slicer {
    let mut s = Slicer::default();
    // Bambu Studio and OrcaSlicer: JSON.
    if let Some(cfg) = text(zip, names, "Metadata/project_settings.config", MAX_CONFIG)
        && let Ok(v) = serde_json::from_str::<Value>(&cfg)
    {
        s.filaments = v
            .get("filament_colour")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|c| c.as_str().and_then(hex_color).unwrap_or(FALLBACK[0]))
            .collect();
    }
    // PrusaSlicer: `; key = value` lines; an extruder's own color wins over its filament's.
    if s.filaments.is_empty()
        && let Some(cfg) = text(zip, names, "Metadata/Slic3r_PE.config", MAX_CONFIG)
    {
        let list = |key: &str| -> Vec<Option<[u8; 3]>> {
            cfg.lines()
                .find_map(|l| l.strip_prefix("; ")?.strip_prefix(key)?.trim_start().strip_prefix('='))
                .map(|v| v.trim().trim_matches('"').split(';').map(hex_color).collect())
                .unwrap_or_default()
        };
        let (extruder, filament) = (list("extruder_colour"), list("filament_colour"));
        s.filaments = (0..extruder.len().max(filament.len()))
            .map(|i| {
                extruder
                    .get(i)
                    .copied()
                    .flatten()
                    .or(filament.get(i).copied().flatten())
                    .unwrap_or(FALLBACK[i % 6])
            })
            .collect();
    }
    for name in ["Metadata/model_settings.config", "Metadata/Slic3r_PE_model.config"] {
        if let Some(xml) = text(zip, names, name, MAX_CONFIG) {
            read_assignments(&xml, &mut s);
        }
    }
    for list in s.volumes.values_mut() {
        list.sort_by_key(|v| v.first);
    }
    s
}

/// `<object id>` with `<metadata key="extruder" value>`, inside it Bambu's `<part id subtype>` and Prusa's
/// `<volume firstid lastid>`, each of which may name its own extruder, what it's for (`volume_type`, or the older
/// `modifier`) and its name.
fn read_assignments(xml: &str, s: &mut Slicer) {
    let mut reader = quick_xml::Reader::from_str(xml);
    // The object, and the part (its id) or volume (its place in the object's list) being read.
    let (mut object, mut part, mut volume) = (None::<u32>, None::<u32>, None::<usize>);
    loop {
        let (e, empty) = match reader.read_event() {
            Ok(Event::Start(e)) => (e, false),
            Ok(Event::Empty(e)) => (e, true),
            Ok(Event::End(e)) => {
                match e.local_name().as_ref() {
                    b"object" => object = None,
                    b"part" => part = None,
                    b"volume" => volume = None,
                    _ => {}
                }
                continue;
            }
            Ok(Event::Eof) | Err(_) => return,
            _ => continue,
        };
        let id = |k: &[u8]| attr(&e, k).and_then(|v| v.parse::<u32>().ok());
        match e.local_name().as_ref() {
            b"object" if !empty => object = id(b"id"),
            b"part" => {
                let (Some(o), Some(p)) = (object, id(b"id")) else {
                    continue;
                };
                let a = s.parts.entry((o, p)).or_default();
                if let Some(role) = attr(&e, b"subtype")
                    .or_else(|| attr(&e, b"type"))
                    .and_then(|v| Role::parse(&v))
                {
                    a.role = role;
                }
                part = (!empty).then_some(p);
            }
            b"volume" => {
                let (Some(o), Some((first, last))) = (object, id(b"firstid").zip(id(b"lastid"))) else {
                    continue;
                };
                let list = s.volumes.entry(o).or_default();
                list.push(SlicerVolume {
                    first,
                    last,
                    a: Assigned::default(),
                });
                volume = (!empty).then_some(list.len() - 1);
            }
            b"metadata" => {
                let (Some(o), Some(key), Some(value)) = (object, attr(&e, b"key"), attr(&e, b"value")) else {
                    continue;
                };
                let a = match (part, volume) {
                    (Some(p), _) => s.parts.get_mut(&(o, p)),
                    (None, Some(v)) => s.volumes.get_mut(&o).and_then(|l| l.get_mut(v)).map(|v| &mut v.a),
                    (None, None) => {
                        if key == "extruder"
                            && let Some(n) = value.trim().parse::<usize>().ok().filter(|&n| n > 0)
                        {
                            s.objects.insert(o, n);
                        }
                        continue;
                    }
                };
                let Some(a) = a else {
                    continue;
                };
                match key.as_str() {
                    "extruder" => {
                        if let Some(n) = value.trim().parse::<usize>().ok().filter(|&n| n > 0) {
                            a.extruder = Some(n);
                        }
                    }
                    "volume_type" | "subtype" => {
                        if let Some(role) = Role::parse(&value) {
                            a.role = role;
                        }
                    }
                    // Older projects only mark modifiers.
                    "modifier" if value.trim() == "1" && a.role == Role::Normal => a.role = Role::Modifier,
                    "name" if !value.is_empty() => a.name = Some(value),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn zip_of(files: &[(&str, &str)]) -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new().suffix(".3mf").tempfile().unwrap();
        let mut zip = zip::ZipWriter::new(f.reopen().unwrap());
        for (name, body) in files {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        f
    }

    const RELS: &str = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>"#;

    const QUAD: &str = r#"<vertices><vertex x="0" y="0" z="0"/><vertex x="10" y="0" z="0"/><vertex x="10" y="10" z="0"/><vertex x="0" y="10" z="0"/></vertices>"#;

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates to 0..=255"
    )]
    fn colors(mesh: &crate::mesh::Mesh) -> Vec<[u8; 3]> {
        let to = |c: f32| (c.powf(1.0 / 2.2) * 255.0).round() as u8;
        mesh.materials
            .iter()
            .map(|m| [to(m.color[0]), to(m.color[1]), to(m.color[2])])
            .collect()
    }

    fn flat(doc: &Document) -> crate::mesh::Mesh {
        doc.flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    #[test]
    fn paint_states() {
        assert_eq!(paint_state(b"4"), 1);
        assert_eq!(paint_state(b"8"), 2);
        assert_eq!(paint_state(b"0C"), 3);
        assert_eq!(paint_state(b"1C"), 4);
        // Split on three sides into four: filament 1 twice, 2 once, unpainted once.
        assert_eq!(paint_state(b"08443"), 1);
        assert_eq!(paint_state(b""), 0);
        // Anything but hex digits is skipped.
        assert_eq!(paint_state(b"0 C"), 3);
    }

    #[test]
    fn base_material_colors_and_build_transforms() {
        let model = format!(
            r##"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
            <metadata name="Application">Test Exporter</metadata>
            <resources><basematerials id="1"><base name="Red" displaycolor="#FF0000"/><base name="Blue" displaycolor="#0000FFFF"/></basematerials>
            <object id="2" type="model" pid="1" pindex="0"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3" pid="1" p1="1" p2="1" p3="1"/></triangles></mesh></object></resources>
            <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 0 0"/><item objectid="2"/></build></model>"##
        );
        let f = zip_of(&[("_rels/.rels", RELS), ("3D/3dmodel.model", &model)]);
        let doc = load(crate::formats::Input::file(f.path()), Keep::ALL).unwrap();
        let (mesh, facts) = (flat(&doc), &doc.facts);
        assert_eq!(facts["triangles"], 4);
        assert_eq!(facts["objects"], 2);
        assert_eq!(facts["colors"], 2);
        let info = doc.source_info();
        assert_eq!(info.tool.as_deref(), Some("Test Exporter"));
        assert!(doc.json["asset"]["extras"]["source"].get("metadata").is_none());
        assert_eq!(info.meters_per_unit, Fact::Declared(0.001));
        assert_eq!(info.unit.as_deref(), Some("millimeter"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        assert!(info.converted.units && info.converted.up_axis);
        assert!(facts.get("application").is_none() && facts.get("unit").is_none());
        assert_eq!(colors(&mesh), [[255, 0, 0], [0, 0, 255]]);
        // The first item moved 100 mm along x.
        assert!(mesh.prims[0].positions.iter().any(|p| p[0] >= 0.1));
    }

    #[test]
    fn the_triangle_limit_refuses_bigger_models() {
        let model = format!(
            r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
            <resources><object id="2" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3"/></triangles></mesh></object></resources>
            <build><item objectid="2"/><item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 0 0"/></build></model>"#
        );
        let f = zip_of(&[("_rels/.rels", RELS), ("3D/3dmodel.model", &model)]);
        let bytes = std::fs::read(f.path()).unwrap();
        let assets = crate::assets::Memory::new();
        let read = |max_triangles| {
            let keep = Keep::ALL.with_max_triangles(max_triangles);
            crate::formats::load_bytes(&bytes, "q.3mf", "3mf", &assets, keep)
        };
        // The file's two triangles are refused as they're parsed, under a limit of two.
        let err = read(1).unwrap_err();
        assert!(format!("{err:#}").contains("too many triangles"), "{err:#}");
        // Drawn twice, they're four: the flat mesh refuses them under a limit of four.
        let doc = read(2).unwrap();
        assert!(doc.flatten(crate::document::Pose::Rest).is_err());
        assert_eq!(flat(&read(4).unwrap()).triangles(), 4);
    }

    #[test]
    fn slicer_filaments_parts_and_paint() {
        // Bambu Studio's layout: the root object's parts live in another model file.
        let root = r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">
            <resources><object id="5" type="model"><components>
              <component p:path="/3D/Objects/object_1.model" objectid="1"/>
              <component p:path="/3D/Objects/object_1.model" objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 20"/>
            </components></object></resources><build><item objectid="5"/></build></model>"#;
        let parts = format!(
            r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>
            <object id="1" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3" paint_color="0C"/></triangles></mesh></object>
            <object id="2" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
            </resources></model>"#
        );
        let settings = r#"<?xml version="1.0"?><config><object id="5"><metadata key="extruder" value="1"/>
            <part id="1" subtype="normal_part"><metadata key="extruder" value="1"/></part>
            <part id="2" subtype="normal_part"><metadata key="extruder" value="2"/></part></object></config>"#;
        let project = r##"{"filament_colour": ["#FFFFFF", "#FF0000", "#00FF00"]}"##;
        let f = zip_of(&[
            ("_rels/.rels", RELS),
            ("3D/3dmodel.model", root),
            ("3D/Objects/object_1.model", &parts),
            ("Metadata/model_settings.config", settings),
            ("Metadata/project_settings.config", project),
        ]);
        let doc = load(crate::formats::Input::file(f.path()), Keep::ALL).unwrap();
        assert_eq!(doc.facts["triangles"], 3);
        // Part 1 in white with one triangle painted green; part 2 in red.
        assert_eq!(colors(&flat(&doc)), [[255, 255, 255], [0, 255, 0], [255, 0, 0]]);
    }

    #[test]
    fn prusa_volumes() {
        let model = format!(
            r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>
            <object id="1" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3"/></triangles></mesh></object>
            </resources><build><item objectid="1"/></build></model>"#
        );
        let cfg = r#"<?xml version="1.0"?><config><object id="1" instances_count="1"><metadata type="object" key="extruder" value="1"/>
            <volume firstid="0" lastid="0"><metadata type="volume" key="extruder" value="2"/></volume>
            <volume firstid="1" lastid="1"></volume></object></config>"#;
        let settings = "; extruder_colour = \"#0000FF;\"\n; filament_colour = \"#FFFFFF;#FFFF00\"\n";
        let f = zip_of(&[
            ("3D/3dmodel.model", &model),
            ("Metadata/Slic3r_PE_model.config", cfg),
            ("Metadata/Slic3r_PE.config", settings),
        ]);
        let doc = load(crate::formats::Input::file(f.path()), Keep::ALL).unwrap();
        // Volume 0 in extruder 2 (no color of its own: its filament's yellow); volume 1 in the object's extruder 1.
        assert_eq!(colors(&flat(&doc)), [[255, 255, 0], [0, 0, 255]]);
    }

    const CORE: &str = r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02""#;

    fn load_zip(files: &[(&str, &str)]) -> Document {
        let f = zip_of(files);
        load(crate::formats::Input::file(f.path()), Keep::ALL).unwrap()
    }

    /// The hidden nodes' names and `extras.volumeType`.
    fn hidden(doc: &Document) -> Vec<(String, String)> {
        let nodes = doc.json["nodes"].as_array().unwrap();
        let mut out: Vec<(String, String)> = nodes
            .iter()
            .filter(|n| n["extensions"]["KHR_node_visibility"]["visible"] == false)
            .map(|n| {
                let s = |v: &Value| v.as_str().unwrap_or_default().to_owned();
                (s(&n["name"]), s(&n["extras"]["volumeType"]))
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn slicer_parts_that_are_not_printed_are_hidden() {
        let root = format!(
            r#"{CORE} xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"><resources><object id="9" type="model"><components>
              <component p:path="/3D/Objects/object_1.model" objectid="1"/><component p:path="/3D/Objects/object_1.model" objectid="2"/>
              <component p:path="/3D/Objects/object_1.model" objectid="3"/><component p:path="/3D/Objects/object_1.model" objectid="4"/>
              <component p:path="/3D/Objects/object_1.model" objectid="5"/>
            </components></object></resources><build><item objectid="9"/></build></model>"#
        );
        let one = format!(
            r#"type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>"#
        );
        let parts = format!(
            r#"{CORE}><resources><object id="1" name="Body" {one}<object id="2" name="Cube" {one}<object id="3" {one}<object id="4" {one}<object id="5" {one}</resources></model>"#
        );
        let settings = r#"<?xml version="1.0"?><config><object id="9"><metadata key="name" value="Thing"/>
            <part id="1" subtype="normal_part"><metadata key="name" value="Body"/></part>
            <part id="2" subtype="modifier_part"><metadata key="name" value="Infill"/><metadata key="extruder" value="2"/></part>
            <part id="3" subtype="negative_part"/>
            <part id="4" subtype="support_blocker"><metadata key="name" value="No supports"/></part>
            <part id="5" subtype="support_enforcer"/></object></config>"#;
        let doc = load_zip(&[
            ("_rels/.rels", RELS),
            ("3D/3dmodel.model", &root),
            ("3D/Objects/object_1.model", &parts),
            ("Metadata/model_settings.config", settings),
        ]);
        // Only the printed part is drawn and counted.
        assert_eq!(doc.facts["triangles"], 1);
        assert_eq!(flat(&doc).triangles(), 1);
        let want = [
            ("Infill (modifier)", "modifier"),
            ("No supports (support blocker)", "support_blocker"),
            ("part (negative volume)", "negative_volume"),
            ("part (support enforcer)", "support_enforcer"),
        ];
        assert_eq!(hidden(&doc), want.map(|(a, b)| (a.to_owned(), b.to_owned())));
        assert!(
            doc.json["extensionsUsed"]
                .as_array()
                .unwrap()
                .contains(&json!("KHR_node_visibility"))
        );
    }

    #[test]
    fn prusa_volume_types() {
        let model = format!(
            r#"{CORE}><resources><object id="1" name="Box" type="model"><mesh>{QUAD}<triangles>
            <triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3"/><triangle v1="0" v2="1" v3="3"/><triangle v1="1" v2="2" v3="3"/>
            </triangles></mesh></object></resources><build><item objectid="1"/></build></model>"#
        );
        let cfg = r#"<?xml version="1.0"?><config><object id="1" instances_count="1"><metadata type="object" key="name" value="Box"/>
            <volume firstid="3" lastid="3"><metadata type="volume" key="modifier" value="1"/></volume>
            <volume firstid="0" lastid="1"><metadata type="volume" key="volume_type" value="ModelPart"/></volume>
            <volume firstid="2" lastid="2"><metadata type="volume" key="name" value="Hole"/><metadata type="volume" key="volume_type" value="NegativeVolume"/></volume>
            </object></config>"#;
        let doc = load_zip(&[("3D/3dmodel.model", &model), ("Metadata/Slic3r_PE_model.config", cfg)]);
        assert_eq!(doc.facts["triangles"], 2);
        assert_eq!(flat(&doc).triangles(), 2);
        let want = [
            ("Box (modifier)", "modifier"),
            ("Hole (negative volume)", "negative_volume"),
        ];
        assert_eq!(hidden(&doc), want.map(|(a, b)| (a.to_owned(), b.to_owned())));
    }

    #[test]
    fn composites_and_multiproperties_become_colors() {
        let model = format!(
            r##"{CORE}><resources>
            <basematerials id="1"><base name="Red" displaycolor="#FF0000"/><base name="Blue" displaycolor="#0000FF"/></basematerials>
            <compositematerials id="2" matid="1" matindices="0 1"><composite values="0.5 0.5"/><composite values="0 1"/></compositematerials>
            <colorgroup id="3"><color color="#808080"/><color color="#00FF0080"/></colorgroup>
            <multiproperties id="4" pids="1 3" blendmethods="multiply"><multi pindices="0 0"/></multiproperties>
            <multiproperties id="5" pids="1 3"><multi pindices="0 1"/></multiproperties>
            <object id="6" type="model"><mesh>{QUAD}<triangles>
            <triangle v1="0" v2="1" v3="2" pid="2" p1="0"/><triangle v1="0" v2="2" v3="3" pid="2" p1="1"/>
            <triangle v1="0" v2="1" v3="3" pid="4" p1="0"/><triangle v1="1" v2="2" v3="3" pid="5" p1="0"/>
            </triangles></mesh></object></resources><build><item objectid="6"/></build></model>"##
        );
        let doc = load_zip(&[("3D/3dmodel.model", &model)]);
        let got = colors(&flat(&doc));
        // Half red and half blue mixed in linear light; all blue; red times mid gray; half-transparent green over red.
        let want = [[188, 0, 188], [0, 0, 255], [128, 0, 0], [188, 188, 0]];
        assert_eq!(got.len(), want.len(), "{got:?}");
        for (g, w) in got.iter().zip(want) {
            assert!(g.iter().zip(w).all(|(&g, w)| g.abs_diff(w) <= 3), "{got:?}");
        }
    }

    #[test]
    fn component_cycles_are_dropped() {
        let model = format!(
            r#"{CORE}><resources><object id="1" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
            <object id="2" type="model"><components><component objectid="1"/><component objectid="3"/><component objectid="2"/></components></object>
            <object id="3" type="model"><components><component objectid="2"/><component objectid="1"/></components></object>
            </resources><build><item objectid="2"/></build></model>"#
        );
        let doc = load_zip(&[("3D/3dmodel.model", &model)]);
        // Object 1 under 2, and under 3 under 2; 2 inside itself and inside 3 left out.
        assert_eq!(doc.facts["triangles"], 2);
        assert_eq!(doc.json["asset"]["extras"]["source"]["droppedComponents"], 2);
    }

    /// Corners share a vertex unless their colors differ.
    #[test]
    fn vertices_are_shared() {
        let model = format!(
            r##"{CORE}><resources><colorgroup id="1"><color color="#FF0000"/><color color="#0000FF"/></colorgroup>
            <object id="2" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3"/>
            <triangle v1="0" v2="1" v3="2" pid="1" p1="0" p2="0" p3="1"/><triangle v1="0" v2="2" v3="3" pid="1" p1="1" p2="0" p3="0"/>
            </triangles></mesh></object></resources><build><item objectid="2"/></build></model>"##
        );
        let doc = load_zip(&[("3D/3dmodel.model", &model)]);
        let mesh = flat(&doc);
        let verts: Vec<usize> = mesh.prims.iter().map(|p| p.positions.len()).collect();
        // A plain quad on four corners; a blended one whose corners 0 and 2 change color between its triangles.
        assert_eq!(verts, [4, 6]);
    }

    /// A 3MF of `files`, in memory.
    fn zip_bytes(files: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, body) in files {
            let stored = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            zip.start_file(*name, stored).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    /// Run with `cargo test --release -- --ignored --nocapture measure_sharing`.
    #[test]
    #[ignore = "a measurement, not a check"]
    #[allow(clippy::cast_precision_loss, reason = "small coordinates")]
    fn measure_sharing() {
        use std::fmt::Write as _;
        let (w, h) = (1000u32, 500u32);
        let mut model = String::from(
            r##"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><basematerials id="1"><base name="A" displaycolor="#FF0000"/><base name="B" displaycolor="#00FF00"/></basematerials><object id="2" type="model" pid="1" pindex="0"><mesh><vertices>"##,
        );
        for y in 0..=h {
            for x in 0..=w {
                let z = (x as f32 / 40.0).sin() * (y as f32 / 30.0).cos() * 5.0;
                write!(model, r#"<vertex x="{x}" y="{y}" z="{z}"/>"#).unwrap();
            }
        }
        model.push_str("</vertices><triangles>");
        for y in 0..h {
            for x in 0..w {
                let a = y * (w + 1) + x;
                let (b, c, d) = (a + 1, a + w + 2, a + w + 1);
                // Bands of the second color, every 50 rows.
                let p = if (y / 50) % 2 == 1 { r#" pid="1" p1="1""# } else { "" };
                write!(
                    model,
                    r#"<triangle v1="{a}" v2="{b}" v3="{c}"{p}/><triangle v1="{a}" v2="{c}" v3="{d}"{p}/>"#
                )
                .unwrap();
            }
        }
        model.push_str("</triangles></mesh></object></resources><build><item objectid=\"2\"/></build></model>");
        let bytes = zip_bytes(&[("_rels/.rels", RELS), ("3D/3dmodel.model", &model)]);
        let assets = crate::assets::Memory::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let doc = crate::formats::load_bytes(&bytes, "q.3mf", "3mf", &assets, Keep::ALL).unwrap();
            let load = start.elapsed();
            let mesh = flat(&doc);
            let verts: usize = mesh.prims.iter().map(|p| p.positions.len()).sum();
            println!(
                "{} triangles, {verts} vertices, load {load:?}, {} buffer bytes",
                mesh.triangles(),
                doc.buffer(0).map_or(0, |b| b.len())
            );
        }
    }
}
