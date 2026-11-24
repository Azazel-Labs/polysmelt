//! `MagicaVoxel` `.vox` (the format as documented in ephtracy/voxel-model, versions 150 and 200).
//!
//! Each model becomes a mesh of the faces between a voxel and empty space, a primitive per
//! palette color, in its own space: centered on its pivot, `floor(size / 2)`, when a scene graph
//! places it, as `MagicaVoxel` shows it; a file without a scene graph keeps the raw voxel
//! coordinates, a node per model. Same-colored faces on a plane merge into rectangles that share
//! their vertices, split so no vertex sits inside another triangle's edge (see [`mesh_model`]).
//! The scene graph becomes nodes: each `nTRN` transform (first frame, its name, layer and
//! attributes in extras) with the `nGRP` group or `nSHP` shape under it; a placed model's nodes
//! share its mesh. A shape with a model per animation frame draws frame 0's and keeps each other
//! frame's as a hidden child node named for its frame. Hidden nodes and nodes on hidden layers
//! are kept, hidden with `KHR_node_visibility`. More than one transform frame is a step animation
//! at [`FRAMES_PER_SECOND`]. Colors are sRGB (the default palette when the file has none), and
//! `MATL` (or legacy `MATT`) metal, roughness, emission and glass carry over, glass's
//! transparency as opacity too. The asset's `extras.source` keeps the palette, every material,
//! the layers, render settings, cameras, notes and index map. Coordinates turn from Z-up to
//! glTF's Y-up. The source info has Z up and right-handed, and the step animation's frame rate,
//! all assumed; a voxel's size is unknown.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::document::{Builder, Document, Keep};
use crate::material::{Factor, Full};
use crate::mesh::{Material, Prim, srgb_to_linear};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// The most voxels a file may hold, unless [`Keep::max_triangles`] is raised past its default,
/// which raises this in proportion.
const MAX_VOXELS: usize = 4_000_000;

/// The palette `MagicaVoxel` uses when a file has no `RGBA` chunk, indexed by color index (0 is
/// empty): a 6x6x6 cube of sRGB levels, then ramps of red, green, blue and gray.
fn default_palette() -> [[u8; 4]; 256] {
    let mut p = [[0u8; 4]; 256];
    let levels = [0xff, 0xcc, 0x99, 0x66, 0x33, 0x00];
    let mut i = 1;
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                if i < 216 {
                    p[i] = [r, g, b, 0xff];
                    i += 1;
                }
            }
        }
    }
    let ramp = [0xee, 0xdd, 0xbb, 0xaa, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11];
    for channel in 0..4 {
        for &v in &ramp {
            p[i] = match channel {
                0 => [v, 0, 0, 0xff],
                1 => [0, v, 0, 0xff],
                2 => [0, 0, v, 0xff],
                _ => [v, v, v, 0xff],
            };
            i += 1;
        }
    }
    p
}

struct Model {
    size: [i32; 3],
    voxels: Vec<[u8; 4]>,
}

/// A rotation (rows of -1/0/1) and translation, all integers.
#[derive(Clone, Copy)]
struct Xform {
    r: [[i32; 3]; 3],
    t: [i32; 3],
}

impl Xform {
    const IDENTITY: Xform = Xform {
        r: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t: [0; 3],
    };

    fn mirrors(&self) -> bool {
        let r = &self.r;
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        det < 0
    }

    /// The rotation byte of an `nTRN` frame's `_r`.
    fn from_rotation_byte(b: u8) -> Xform {
        let first = usize::from(b & 3);
        let second = usize::from((b >> 2) & 3);
        let mut r = [[0; 3]; 3];
        if first > 2 || second > 2 || first == second {
            return Xform::IDENTITY;
        }
        let third = 3 - first - second;
        let sign = |bit: u8| if b >> bit & 1 == 1 { -1 } else { 1 };
        r[0][first] = sign(4);
        r[1][second] = sign(5);
        r[2][third] = sign(6);
        Xform { r, t: [0; 3] }
    }
}

/// Reads little-endian values and `DICT`s from a chunk body.
struct Body<'a> {
    b: &'a [u8],
    at: usize,
}

impl Body<'_> {
    fn u32(&mut self) -> Result<u32> {
        let v = self.b.get(self.at..self.at + 4).and_then(|v| v.try_into().ok());
        let v = v.context(ErrorKind::Truncated, "VOX chunk is cut short")?;
        self.at += 4;
        Ok(u32::from_le_bytes(v))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()?.cast_signed())
    }

    fn string(&mut self) -> Result<&str> {
        let n = self.u32()? as usize;
        let s = self
            .b
            .get(self.at..self.at.saturating_add(n))
            .context(ErrorKind::Truncated, "VOX chunk is cut short")?;
        self.at += n;
        Ok(std::str::from_utf8(s).unwrap_or(""))
    }

    fn dict(&mut self) -> Result<HashMap<String, String>> {
        let n = self.u32()?;
        ensure!(n <= 4096, LimitExceeded, "VOX dictionary too large");
        let mut d = HashMap::new();
        for _ in 0..n {
            let k = self.string()?.to_owned();
            let v = self.string()?.to_owned();
            d.insert(k, v);
        }
        Ok(d)
    }
}

/// A chunk's `DICT`.
type Dict = HashMap<String, String>;

/// One `nTRN` frame: its frame number and where it puts the child.
struct Frame {
    number: u32,
    xform: Xform,
}

enum Node {
    Transform {
        attrs: Dict,
        child: u32,
        layer: i32,
        frames: Vec<Frame>,
    },
    Group {
        attrs: Dict,
        children: Vec<u32>,
    },
    Shape {
        attrs: Dict,
        models: Vec<(u32, Dict)>,
    },
}

/// A `MATL` chunk as glTF has places for it: metal's metalness and roughness, glass's transmission and its
/// roughness, an emitter's glow, and the index of refraction (written as how far above one it is). A blend has
/// what it mixes.
#[derive(Default, Clone, Copy)]
struct Matl {
    metal: f32,
    rough: Option<f32>,
    emit: f32,
    transmission: Option<f32>,
    ior: Option<f32>,
    /// Glass's or a blend's opacity, from the transparency it gives (`_trans`, or the older
    /// `_alpha`), so a client without transmission still shows it see-through; never below
    /// [`MIN_OPACITY`], so clear glass doesn't vanish there.
    opacity: Option<f32>,
}

/// The least opacity glass is given.
const MIN_OPACITY: f32 = 0.1;

impl Matl {
    fn new(d: &Dict) -> Matl {
        let f = |k: &str| d.get(k).and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite());
        let ty = d.get("_type").map_or("_diffuse", String::as_str);
        let (metal, glass, emit) = match ty {
            "_metal" => (true, false, false),
            "_glass" => (false, true, false),
            "_emit" => (false, false, true),
            "_blend" => (true, true, true),
            _ => (false, false, false),
        };
        Matl {
            metal: if metal { f("_metal").unwrap_or(0.0) } else { 0.0 },
            rough: (metal || glass).then(|| f("_rough")).flatten(),
            emit: if emit {
                f("_emit").unwrap_or(0.0) * (1.0 + f("_flux").unwrap_or(0.0))
            } else {
                0.0
            },
            transmission: glass.then(|| f("_trans").unwrap_or(if ty == "_glass" { 1.0 } else { 0.0 })),
            ior: (metal || glass).then(|| f("_ior")).flatten().map(|i| 1.0 + i),
            opacity: glass
                .then(|| f("_trans").or_else(|| f("_alpha")))
                .flatten()
                .map(|t| (1.0 - t.clamp(0.0, 1.0)).max(MIN_OPACITY)),
        }
    }

    /// The `MATL` dictionary of a legacy `MATT` chunk's material: its type and weight (metal's
    /// metalness, glass's transparency, an emitter's glow) and, when given, its roughness.
    fn legacy(b: &mut Body) -> Result<(u32, Dict)> {
        let id = b.u32()?;
        let kind = b.u32()?;
        let weight = f32::from_bits(b.u32()?);
        let bits = b.u32()?;
        let mut d = Dict::new();
        let (ty, key) = match kind {
            1 => ("_metal", "_metal"),
            2 => ("_glass", "_trans"),
            3 => ("_emit", "_emit"),
            _ => ("_diffuse", ""),
        };
        d.insert("_type".into(), ty.into());
        if !key.is_empty() {
            d.insert(key.into(), weight.to_string());
        }
        // Values follow for each set bit below the seventh, in bit order: plastic, roughness,
        // specular, index of refraction, attenuation, power, glow.
        for bit in 0..7 {
            if bits >> bit & 1 == 1 {
                let value = f32::from_bits(b.u32()?);
                if bit == 1 {
                    d.insert("_rough".into(), value.to_string());
                }
            }
        }
        Ok((id, d))
    }
}

/// `.vox` has no frame rate; animation frames are this many to a second.
const FRAMES_PER_SECOND: f32 = 10.0;

/// The most glTF nodes a scene graph may unfold into (a group listed twice is walked twice).
const MAX_NODES: usize = 1 << 16;

fn dict_json(d: &Dict) -> Value {
    Value::Object(d.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
}

/// Everything a `.vox` file holds, read.
#[derive(Default)]
struct Vox {
    version: u32,
    models: Vec<Model>,
    palette: Option<[[u8; 4]; 256]>,
    nodes: HashMap<u32, Node>,
    layers: Vec<(i32, Dict)>,
    matl: Vec<(u32, Dict)>,
    /// Legacy `MATT` materials, as `MATL` dictionaries; a `MATL` for the same color wins.
    matt: Vec<(u32, Dict)>,
    render: Vec<Value>,
    cameras: Vec<Value>,
    notes: Vec<String>,
    index_map: Vec<u8>,
    skipped: Vec<String>,
}

/// `MagicaVoxel`: each model a mesh of the faces between its voxels and empty space, one
/// primitive per color; the scene graph as nodes, hidden ones kept but hidden.
pub(crate) fn vox(data: &[u8], keep: Keep) -> Result<Document<'static>> {
    let per_voxel = crate::mesh::MAX_TRIANGLES / MAX_VOXELS;
    let v = read(data, (keep.triangles() / per_voxel).max(MAX_VOXELS))?;
    let scene = v.nodes.contains_key(&0);
    let models = &v.models;

    // Each model in its own space: centered on its pivot when a scene graph places it.
    let mut meshed: Vec<Vec<(u8, Prim)>> = Vec::with_capacity(models.len());
    for m in models {
        let mut prims: HashMap<u8, Prim> = HashMap::new();
        mesh_model(m, scene, &mut prims)?;
        let mut prims: Vec<(u8, Prim)> = prims.into_iter().collect();
        prims.sort_by_key(|(c, _)| *c);
        meshed.push(prims);
    }
    let mut colors: Vec<u8> = meshed.iter().flatten().map(|(c, _)| *c).collect();
    colors.sort_unstable();
    colors.dedup();

    let mut facts = Map::new();
    let s = models[0].size;
    facts.insert("voxel_size".into(), json!(format!("{}x{}x{}", s[0], s[1], s[2])));
    if models.len() > 1 {
        facts.insert("models".into(), json!(models.len()));
    }
    if colors.is_empty() {
        facts.insert("voxels".into(), json!(0));
        return Ok(Document::empty(facts));
    }

    let mut b = Builder::default();
    let mut meshes = Vec::new();
    let mut mesh_of = Vec::with_capacity(models.len());
    for (m, prims) in meshed.into_iter().enumerate() {
        let prims = prims
            .into_iter()
            .map(|(c, mut p)| {
                p.material = colors.binary_search(&c).unwrap_or(0);
                p
            })
            .collect();
        mesh_of.push(b.mesh(prims, colors.len()).map(|mut mesh| {
            mesh["name"] = json!(format!("model_{m}"));
            meshes.push(mesh);
            meshes.len() - 1
        }));
    }

    let mut walk = Walk {
        v: &v,
        mesh_of: &mesh_of,
        out: Vec::new(),
        animated: Vec::new(),
        voxels: 0,
        hidden: false,
    };
    let roots = if scene {
        walk.visit(0, 0, true)?.into_iter().collect()
    } else {
        // No scene graph: every model where its voxels are.
        for (m, model) in models.iter().enumerate() {
            walk.voxels += model.voxels.len();
            if let Some(mesh) = mesh_of[m] {
                walk.out.push(json!({ "mesh": mesh }));
            }
        }
        (0..walk.out.len()).collect::<Vec<_>>()
    };
    facts.insert("voxels".into(), json!(walk.voxels));
    let Walk {
        out: nodes,
        animated,
        hidden,
        ..
    } = walk;

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source(&v) } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": roots }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    if hidden {
        json.insert("extensionsUsed".into(), json!(["KHR_node_visibility"]));
    }
    if keep.animations && !animated.is_empty() {
        json.insert("animations".into(), json!([animation(&mut b, &animated)]));
    }

    let palette = v.palette.unwrap_or_else(default_palette);
    let matl: HashMap<u32, Matl> = v.matt.iter().chain(&v.matl).map(|(i, d)| (*i, Matl::new(d))).collect();
    let materials: Vec<Full> = colors
        .iter()
        .map(|&c| {
            let rgba = palette[usize::from(c)];
            let [r, g, b] = [rgba[0], rgba[1], rgba[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
            let x = matl.get(&u32::from(c));
            let alpha = x.and_then(|x| x.opacity).unwrap_or(1.0);
            let mut mat = Full::from(Material::plain(&format!("color_{c}"), [r, g, b, alpha]));
            if let Some(x) = x {
                mat.metallic = x.metal;
                if let Some(rough) = x.rough {
                    mat.roughness = rough;
                }
                // Brighter than one goes in KHR_materials_emissive_strength.
                let e = x.emit.max(0.0);
                mat.emissive = [r * e, g * e, b * e];
                mat.transmission = x.transmission.filter(|t| *t > 0.0).map(Factor::of);
                mat.ior = x.ior.filter(|i| *i >= 1.0);
            }
            mat
        })
        .collect();
    b.materials(&materials, Vec::new(), &mut json);
    if let Some(list) = json.get_mut("materials").and_then(Value::as_array_mut) {
        for (mat, c) in list.iter_mut().zip(&colors) {
            mat["extras"] = json!({ "paletteIndex": c });
        }
    }
    Ok(b.finish(json, facts))
}

/// The asset's `extras.source`: what the file says that the nodes and materials don't.
fn source(v: &Vox) -> Value {
    let mut s = Map::new();
    if let Some(p) = &v.palette {
        let hex: Vec<String> = p[1..]
            .iter()
            .map(|c| format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3]))
            .collect();
        s.insert("palette".into(), json!(hex));
    }
    if !v.matl.is_empty() {
        let m: Map<String, Value> = v.matl.iter().map(|(i, d)| (i.to_string(), dict_json(d))).collect();
        s.insert("materials".into(), Value::Object(m));
    }
    if !v.matt.is_empty() {
        let m: Map<String, Value> = v.matt.iter().map(|(i, d)| (i.to_string(), dict_json(d))).collect();
        s.insert("legacyMaterials".into(), Value::Object(m));
    }
    if !v.layers.is_empty() {
        let layers: Vec<Value> = v
            .layers
            .iter()
            .map(|(id, d)| {
                let mut l = dict_json(d);
                l["id"] = json!(id);
                l
            })
            .collect();
        s.insert("layers".into(), json!(layers));
    }
    for (key, list) in [("render", &v.render), ("cameras", &v.cameras)] {
        if !list.is_empty() {
            s.insert(key.into(), json!(list));
        }
    }
    if !v.notes.is_empty() {
        s.insert("notes".into(), json!(v.notes));
    }
    if !v.index_map.is_empty() {
        s.insert("indexMap".into(), json!(v.index_map));
    }
    if !v.skipped.is_empty() {
        s.insert("skipped".into(), json!(v.skipped));
    }
    // Z up and right-handed is the editor's convention, not the file's word; a voxel has no size.
    let mut info = SourceInfo {
        format: Some("VOX".into()),
        version: Some(v.version.to_string()),
        up_axis: Fact::Assumed(Axis::PosZ),
        handedness: Fact::Assumed(Handedness::Right),
        frame_rate: if v.nodes.contains_key(&0) {
            Fact::Assumed(f64::from(FRAMES_PER_SECOND))
        } else {
            Fact::Unknown
        },
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    info.write(&mut s);
    Value::Object(s)
}

/// Reads every chunk, refusing a file of more than `max_voxels` voxels.
fn read(data: &[u8], max_voxels: usize) -> Result<Vox> {
    ensure!(
        data.len() >= 20 && &data[..4] == b"VOX ",
        UnknownFormat,
        "not a MagicaVoxel file"
    );
    let le = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            data.get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .context(ErrorKind::Truncated, "VOX file is cut short")?,
        ))
    };
    ensure!(&data[8..12] == b"MAIN", Corrupt, "VOX file has no MAIN chunk");
    let mut v = Vox {
        version: le(4)?,
        ..Vox::default()
    };
    let mut size: Option<[i32; 3]> = None;
    let mut total = 0usize;
    // MAIN's children follow its 12-byte chunk header, after the 8-byte file header.
    let mut at = 8 + 12;
    while at + 12 <= data.len() {
        let id = &data[at..at + 4];
        let content = le(at + 4)? as usize;
        let start = at + 12;
        let end = start
            .checked_add(content)
            .context(ErrorKind::Corrupt, "VOX chunk overflow")?;
        let next = end
            .checked_add(le(at + 8)? as usize)
            .context(ErrorKind::Corrupt, "VOX chunk overflow")?;
        let body = data
            .get(start..end)
            .context(ErrorKind::Truncated, "VOX chunk is cut short")?;
        let mut b = Body { b: body, at: 0 };
        match id {
            b"SIZE" => size = Some([b.i32()?, b.i32()?, b.i32()?]),
            b"XYZI" => {
                let n = b.u32()? as usize;
                total = total.saturating_add(n);
                ensure!(
                    total <= max_voxels,
                    LimitExceeded,
                    "too many voxels to convert ({total})"
                );
                let raw = body
                    .get(4..4 + n * 4)
                    .context(ErrorKind::Truncated, "VOX voxels are cut short")?;
                let size = size.take().context(ErrorKind::Corrupt, "VOX voxels without a SIZE")?;
                v.models.push(Model {
                    size,
                    voxels: raw.as_chunks::<4>().0.to_vec(),
                });
            }
            b"RGBA" => {
                // Entry i is color index i + 1.
                let raw = body
                    .get(..1024)
                    .context(ErrorKind::Truncated, "VOX palette is cut short")?;
                let mut p = [[0u8; 4]; 256];
                for (i, c) in raw.as_chunks::<4>().0.iter().take(255).enumerate() {
                    p[i + 1] = *c;
                }
                v.palette = Some(p);
            }
            b"nTRN" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let child = b.u32()?;
                let _reserved = b.i32()?;
                let layer = b.i32()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    Truncated,
                    "VOX transform is cut short: it lists more frames than it holds"
                );
                let mut frames = Vec::new();
                for k in 0..n {
                    let f = b.dict()?;
                    let mut xform = Xform::IDENTITY;
                    if let Some(r) = f.get("_r").and_then(|r| r.trim().parse::<u8>().ok()) {
                        xform = Xform::from_rotation_byte(r);
                    }
                    if let Some(t) = f.get("_t") {
                        let v: Vec<i32> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                        if let [x, y, z] = v[..] {
                            xform.t = [x, y, z];
                        }
                    }
                    let number = f.get("_f").and_then(|f| f.trim().parse().ok()).unwrap_or(k);
                    frames.push(Frame { number, xform });
                }
                v.nodes.insert(
                    id,
                    Node::Transform {
                        attrs,
                        child,
                        layer,
                        frames,
                    },
                );
            }
            b"nGRP" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    Truncated,
                    "VOX group is cut short: it lists more children than it holds"
                );
                let children = (0..n).map(|_| b.u32()).collect::<Result<_>>()?;
                v.nodes.insert(id, Node::Group { attrs, children });
            }
            b"nSHP" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    Truncated,
                    "VOX shape is cut short: it lists more models than it holds"
                );
                let mut models = Vec::new();
                for _ in 0..n {
                    models.push((b.u32()?, b.dict()?));
                }
                v.nodes.insert(id, Node::Shape { attrs, models });
            }
            b"LAYR" => {
                let id = b.i32()?;
                v.layers.push((id, b.dict()?));
            }
            b"MATL" => {
                let id = b.u32()?;
                v.matl.push((id, b.dict()?));
            }
            b"MATT" => v.matt.push(Matl::legacy(&mut b)?),
            b"rOBJ" => v.render.push(dict_json(&b.dict()?)),
            b"rCAM" => {
                let id = b.u32()?;
                let mut cam = dict_json(&b.dict()?);
                cam["id"] = json!(id);
                v.cameras.push(cam);
            }
            b"NOTE" => {
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    Truncated,
                    "VOX note is cut short: it lists more names than it holds"
                );
                for _ in 0..n {
                    let note = b.string()?.to_owned();
                    v.notes.push(note);
                }
            }
            b"IMAP" => {
                v.index_map = body
                    .get(..256)
                    .context(ErrorKind::Truncated, "VOX index map is cut short")?
                    .to_vec();
            }
            b"PACK" => {}
            _ => {
                let name = String::from_utf8_lossy(id).into_owned();
                if !v.skipped.contains(&name) {
                    v.skipped.push(name);
                }
            }
        }
        at = next;
    }
    ensure!(!v.models.is_empty(), Empty, "VOX file has no model");
    Ok(v)
}

/// Unfolds the scene graph into glTF nodes.
struct Walk<'a> {
    v: &'a Vox,
    mesh_of: &'a [Option<usize>],
    out: Vec<Value>,
    /// Each node with more than one frame, and those frames.
    animated: Vec<(usize, &'a [Frame])>,
    /// Voxels the visible shapes draw.
    voxels: usize,
    /// Whether any node is hidden.
    hidden: bool,
}

impl Walk<'_> {
    /// The glTF node for scene node `id` (a transform with the group or shape under it), its
    /// index; `None` for an id that isn't there.
    fn visit(&mut self, id: u32, depth: usize, visible: bool) -> Result<Option<usize>> {
        ensure!(depth < 64, LimitExceeded, "VOX scene graph too deep");
        ensure!(
            self.out.len() < MAX_NODES,
            LimitExceeded,
            "VOX scene graph unfolds into too many nodes"
        );
        let Some(node) = self.v.nodes.get(&id) else {
            return Ok(None);
        };
        let mut out = Map::new();
        let mut extras = Map::new();
        let mut visible = visible;
        let content = match node {
            Node::Transform {
                attrs,
                child,
                layer,
                frames,
            } => {
                if let Some(name) = attrs.get("_name") {
                    out.insert("name".into(), json!(name));
                }
                let layer_hidden = self
                    .v
                    .layers
                    .iter()
                    .any(|(l, d)| l == layer && d.get("_hidden").is_some_and(|h| h == "1"));
                if attrs.get("_hidden").is_some_and(|h| h == "1") || layer_hidden {
                    out.insert(
                        "extensions".into(),
                        json!({ "KHR_node_visibility": { "visible": false } }),
                    );
                    self.hidden = true;
                    visible = false;
                }
                if let Some(f) = frames.first() {
                    trs(&f.xform, &mut out);
                }
                let rest: Dict = attrs
                    .iter()
                    .filter(|(k, _)| *k != "_name" && *k != "_hidden")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if !rest.is_empty() {
                    extras.insert("transform".into(), dict_json(&rest));
                }
                if *layer >= 0 {
                    extras.insert("layer".into(), json!(layer));
                }
                self.v.nodes.get(child).map(|n| (n, *child))
            }
            _ => Some((node, id)),
        };
        let mut children = Vec::new();
        match content {
            Some((Node::Group { attrs, children: ids }, _)) => {
                if !attrs.is_empty() {
                    extras.insert("group".into(), dict_json(attrs));
                }
                for &c in ids {
                    children.extend(self.visit(c, depth + 1, visible)?);
                }
            }
            Some((Node::Shape { attrs, models }, _)) => {
                if !attrs.is_empty() {
                    extras.insert("shape".into(), dict_json(attrs));
                }
                // A model per animation frame (`_f`): the earliest is drawn, each other one a
                // hidden child to show in its place.
                let frame = |d: &Dict| d.get("_f").and_then(|f| f.trim().parse::<u32>().ok()).unwrap_or(0);
                let shown = (0..models.len()).min_by_key(|&k| frame(&models[k].1));
                for (k, (m, d)) in models.iter().enumerate() {
                    let m = *m as usize;
                    let (Some(model), Some(&mesh)) = (self.v.models.get(m), self.mesh_of.get(m)) else {
                        continue;
                    };
                    if Some(k) == shown {
                        if visible {
                            self.voxels += model.voxels.len();
                        }
                        if let Some(mesh) = mesh {
                            out.insert("mesh".into(), json!(mesh));
                        }
                    } else if let Some(mesh) = mesh
                        && self.out.len() < MAX_NODES
                    {
                        self.out.push(json!({
                            "name": format!("frame_{}", frame(d)),
                            "mesh": mesh,
                            "extensions": { "KHR_node_visibility": { "visible": false } },
                            "extras": { "frame": frame(d), "model": m },
                        }));
                        self.hidden = true;
                        children.push(self.out.len() - 1);
                    }
                }
                // Every frame's model and attributes, by number.
                if models.len() > 1 || models.first().is_some_and(|(_, d)| !d.is_empty()) {
                    let list: Vec<Value> = models
                        .iter()
                        .map(|(m, d)| {
                            let mut e = dict_json(d);
                            e["model"] = json!(m);
                            e
                        })
                        .collect();
                    extras.insert("models".into(), json!(list));
                }
            }
            Some((Node::Transform { .. }, child)) => {
                children.extend(self.visit(child, depth + 1, visible)?);
            }
            None => {}
        }
        if !children.is_empty() {
            out.insert("children".into(), json!(children));
        }
        if !extras.is_empty() {
            out.insert("extras".into(), Value::Object(extras));
        }
        self.out.push(Value::Object(out));
        let index = self.out.len() - 1;
        if let Node::Transform { frames, .. } = node
            && frames.len() > 1
        {
            self.animated.push((index, frames));
        }
        Ok(Some(index))
    }
}

/// The glTF translation, rotation and scale of `x`, a `MagicaVoxel` (Z-up) transform: a
/// mirror is a scale of -1 on X after a proper rotation.
fn gltf_trs(x: &Xform) -> ([f32; 3], [f32; 4], [f32; 3]) {
    // (x, y, z) -> (x, z, -y), as the vertices: C R C^T, C t.
    const C: [[i32; 3]; 3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];
    let mul = |a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]| -> [[i32; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let ct: [[i32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| C[j][i]));
    let mut r = mul(&mul(&C, &x.r), &ct);
    let mut scale = [1.0; 3];
    if x.mirrors() {
        for row in &mut r {
            row[0] = -row[0];
        }
        scale[0] = -1.0;
    }
    #[allow(clippy::cast_precision_loss, reason = "translations are small integers")]
    let t = [x.t[0] as f32, x.t[2] as f32, -x.t[1] as f32];
    (t, quaternion(&r), scale)
}

/// Sets a node's `translation`, `rotation` and `scale` from `x`, leaving out the defaults.
#[allow(clippy::float_cmp, reason = "exact: every value comes from small integers")]
fn trs(x: &Xform, out: &mut Map<String, Value>) {
    let (t, q, s) = gltf_trs(x);
    if t != [0.0; 3] {
        out.insert("translation".into(), json!(t));
    }
    if q != [0.0, 0.0, 0.0, 1.0] {
        out.insert("rotation".into(), json!(q));
    }
    if s != [1.0; 3] {
        out.insert("scale".into(), json!(s));
    }
}

/// The unit quaternion (x, y, z, w) of a rotation matrix of -1, 0 and 1.
#[allow(clippy::cast_precision_loss, reason = "entries are -1, 0 or 1")]
fn quaternion(r: &[[i32; 3]; 3]) -> [f32; 4] {
    let m = |i: usize, j: usize| r[i][j] as f32;
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m(2, 1) - m(1, 2)) / s,
            (m(0, 2) - m(2, 0)) / s,
            (m(1, 0) - m(0, 1)) / s,
            s / 4.0,
        ]
    } else if m(0, 0) >= m(1, 1) && m(0, 0) >= m(2, 2) {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        [
            s / 4.0,
            (m(0, 1) + m(1, 0)) / s,
            (m(0, 2) + m(2, 0)) / s,
            (m(2, 1) - m(1, 2)) / s,
        ]
    } else if m(1, 1) >= m(2, 2) {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        [
            (m(0, 1) + m(1, 0)) / s,
            s / 4.0,
            (m(1, 2) + m(2, 1)) / s,
            (m(0, 2) - m(2, 0)) / s,
        ]
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        [
            (m(0, 2) + m(2, 0)) / s,
            (m(1, 2) + m(2, 1)) / s,
            s / 4.0,
            (m(1, 0) - m(0, 1)) / s,
        ]
    };
    // Exact for quarter turns, so the identity comes out as (0, 0, 0, 1).
    q.map(|c| if c.abs() < 1e-6 { 0.0 } else { c })
}

/// One animation stepping each animated node through its frames.
#[allow(clippy::float_cmp, reason = "exact: a scale is -1 or 1")]
fn animation(b: &mut Builder, animated: &[(usize, &[Frame])]) -> Value {
    let (mut channels, mut samplers) = (Vec::new(), Vec::new());
    for &(node, frames) in animated {
        let mut frames: Vec<&Frame> = frames.iter().collect();
        frames.sort_by_key(|f| f.number);
        frames.dedup_by_key(|f| f.number);
        #[allow(clippy::cast_precision_loss, reason = "frame numbers are small")]
        let times: Vec<f32> = frames.iter().map(|f| f.number as f32 / FRAMES_PER_SECOND).collect();
        let input = b.floats(times, "SCALAR", true);
        let trs: Vec<_> = frames.iter().map(|f| gltf_trs(&f.xform)).collect();
        let mirrors = trs.iter().any(|(.., s)| *s != [1.0; 3]);
        let mut paths = vec![
            (
                "translation",
                trs.iter().flat_map(|(t, ..)| *t).collect::<Vec<_>>(),
                "VEC3",
            ),
            ("rotation", trs.iter().flat_map(|(_, q, _)| *q).collect(), "VEC4"),
        ];
        if mirrors {
            paths.push(("scale", trs.iter().flat_map(|(.., s)| *s).collect(), "VEC3"));
        }
        for (path, values, kind) in paths {
            let output = b.floats(values, kind, false);
            channels.push(json!({ "sampler": samplers.len(), "target": { "node": node, "path": path } }));
            samplers.push(json!({ "input": input, "output": output, "interpolation": "STEP" }));
        }
    }
    json!({ "name": "frames", "channels": channels, "samplers": samplers })
}

/// Each face direction as the axis it faces along, whether it faces the positive way, and the
/// in-plane axes `u` and `v`, ordered so counter-clockwise in `(u, v)` is counter-clockwise seen
/// from outside; Z-up.
const DIRS: [(usize, bool, usize, usize); 6] = [
    (0, true, 1, 2),
    (0, false, 2, 1),
    (1, true, 2, 0),
    (1, false, 0, 2),
    (2, true, 0, 1),
    (2, false, 1, 0),
];

type Fast = std::hash::BuildHasherDefault<crate::mesh::FxHasher>;

/// A rectangle of same-colored faces on one plane: `[u0, u1) x [v0, v1)` in that plane's axes.
#[derive(Clone, Copy)]
struct Rect {
    /// The direction (an index into [`DIRS`]) times 1025, plus the plane.
    bucket: u16,
    color: u8,
    u0: u16,
    u1: u16,
    v0: u16,
    v1: u16,
}

impl Rect {
    /// The lattice point at `(u, v)` on this rectangle's plane.
    fn point(&self, u: usize, v: usize) -> [usize; 3] {
        let (d, plane) = (usize::from(self.bucket) / 1025, usize::from(self.bucket) % 1025);
        let (axis, _, ua, va) = DIRS[d];
        let mut p = [0; 3];
        p[axis] = plane;
        p[ua] = u;
        p[va] = v;
        p
    }
}

/// The lattice points of a model `size` voxels across that are some rectangle's corner, one bit
/// each.
struct Corners {
    bits: Vec<u64>,
    side: [usize; 2],
}

impl Corners {
    fn new(size: [usize; 3]) -> Self {
        let side = [size[0] + 1, size[1] + 1];
        let bits = vec![0; (side[0] * side[1] * (size[2] + 1)).div_ceil(64)];
        Self { bits, side }
    }

    fn at(&self, p: [usize; 3]) -> usize {
        (p[2] * self.side[1] + p[1]) * self.side[0] + p[0]
    }

    fn insert(&mut self, p: [usize; 3]) {
        let i = self.at(p);
        self.bits[i / 64] |= 1 << (i % 64);
    }

    fn contains(&self, p: [usize; 3]) -> bool {
        let i = self.at(p);
        self.bits[i / 64] >> (i % 64) & 1 == 1
    }
}

/// The vertex of each color at each point of one plane at a time: the first color to reach a
/// point keeps it in a dense table, which is marked with the plane's turn so moving to the next
/// plane clears nothing; others, at points where colors meet, go in a map.
struct Shared {
    /// The turn, color and vertex at each point, `u` fastest.
    slots: Vec<(u32, u8, u32)>,
    others: HashMap<u32, u32, Fast>,
    turn: u32,
    width: usize,
}

impl Shared {
    /// Room for every plane of a model `size` voxels across.
    fn new(size: [usize; 3]) -> Self {
        let most = DIRS.iter().map(|&(_, _, ua, va)| (size[ua] + 1) * (size[va] + 1)).max();
        Self {
            slots: vec![(0, 0, 0); most.unwrap_or(0)],
            others: HashMap::default(),
            turn: 0,
            width: 0,
        }
    }

    /// Starts a plane whose `u` runs to `width - 1`.
    fn plane(&mut self, width: usize) {
        self.turn += 1;
        self.width = width;
        self.others.clear();
    }

    /// The vertex of `color` at `(u, v)`, made by `make` the first time.
    #[allow(clippy::cast_possible_truncation, reason = "plane coordinates are at most 1024")]
    fn get(&mut self, color: u8, u: usize, v: usize, make: impl FnOnce() -> Result<u32>) -> Result<u32> {
        let slot = &mut self.slots[v * self.width + u];
        if slot.0 != self.turn {
            let id = make()?;
            *slot = (self.turn, color, id);
            return Ok(id);
        }
        if slot.1 == color {
            return Ok(slot.2);
        }
        let key = u32::from(color) | (u as u32) << 8 | (v as u32) << 19;
        if let Some(&id) = self.others.get(&key) {
            return Ok(id);
        }
        let id = make()?;
        self.others.insert(key, id);
        Ok(id)
    }
}

/// Triangles between two chains of `ring` points on parallel sides, `a` and `b`, both in order
/// of coordinate `k`, `b` to the left of `a`'s direction; `ids` are the points' vertices.
fn strip(p: &mut Prim, ring: &[[usize; 2]], ids: &[u32], a: &[usize], b: &[usize], k: usize) {
    let (mut i, mut j) = (0, 0);
    while i + 1 < a.len() || j + 1 < b.len() {
        if j + 1 == b.len() || (i + 1 < a.len() && ring[a[i + 1]][k] <= ring[b[j + 1]][k]) {
            p.indices.extend([ids[a[i]], ids[a[i + 1]], ids[b[j]]]);
            i += 1;
        } else {
            p.indices.extend([ids[a[i]], ids[b[j + 1]], ids[b[j]]]);
            j += 1;
        }
    }
}

/// Adds a model's exposed faces to the primitive of each voxel's color, centered on its pivot
/// when `centered`.
///
/// Same-colored faces on a plane merge into rectangles: runs along each row, then runs of equal
/// extent on the rows after. A rectangle's side is split wherever another rectangle has a corner
/// on it, and at every voxel along the model's bounding box, so no vertex sits inside another
/// triangle's edge (a T-junction, which shows as a crack), within the model or where models
/// placed side by side meet. Vertices are shared between a plane's rectangles of one color.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "model sizes are checked to 1..=1024, so every coordinate is small and signs are checked"
)]
fn mesh_model(model: &Model, centered: bool, prims: &mut HashMap<u8, Prim>) -> Result<()> {
    let [sx, sy, sz] = model.size;
    ensure!(
        (1..=1024).contains(&sx) && (1..=1024).contains(&sy) && (1..=1024).contains(&sz),
        Corrupt,
        "VOX model size {sx}x{sy}x{sz} is out of range"
    );
    let size = [sx as usize, sy as usize, sz as usize];
    let [sx, sy, sz] = size;
    let cell = |x: usize, y: usize, z: usize| (z * sy + y) * sx + x;
    let mut occupied = vec![0u64; (sx * sy * sz).div_ceil(64)];
    let inside = |v: &[u8; 4]| (v[0] as usize) < sx && (v[1] as usize) < sy && (v[2] as usize) < sz && v[3] != 0;
    for v in model.voxels.iter().filter(|v| inside(v)) {
        let c = cell(v[0].into(), v[1].into(), v[2].into());
        occupied[c / 64] |= 1 << (c % 64);
    }
    let filled = |p: [i32; 3]| {
        if p.iter().zip(size).any(|(&c, s)| c < 0 || c as usize >= s) {
            return false;
        }
        let c = cell(p[0] as usize, p[1] as usize, p[2] as usize);
        occupied[c / 64] >> (c % 64) & 1 == 1
    };

    // Every exposed face as one number: its bucket (direction and plane), row, column and color,
    // so sorting gathers each plane's faces row by row.
    let mut faces: Vec<u64> = Vec::with_capacity(model.voxels.len());
    let mut seen = vec![0u64; occupied.len()];
    for v in model.voxels.iter().filter(|v| inside(v)) {
        // A voxel listed twice draws once.
        let c = cell(v[0].into(), v[1].into(), v[2].into());
        if seen[c / 64] >> (c % 64) & 1 == 1 {
            continue;
        }
        seen[c / 64] |= 1 << (c % 64);
        let p = [usize::from(v[0]), usize::from(v[1]), usize::from(v[2])];
        for (d, &(axis, positive, ua, va)) in DIRS.iter().enumerate() {
            let mut n = p.map(|c| c as i32);
            n[axis] += if positive { 1 } else { -1 };
            if filled(n) {
                continue;
            }
            let bucket = d * 1025 + p[axis] + usize::from(positive);
            faces.push((bucket as u64) << 28 | (p[va] as u64) << 18 | (p[ua] as u64) << 8 | u64::from(v[3]));
        }
    }
    faces.sort_unstable();

    // Rectangles: runs of one color along a row, grown a row at a time while the next row has a
    // run of the same extent and color.
    let mut rects: Vec<Rect> = Vec::new();
    let (mut open, mut next): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < faces.len() {
        let bucket = faces[i] >> 28;
        let end = i + faces[i..].partition_point(|f| f >> 28 == bucket);
        open.clear();
        while i < end {
            let line = faces[i] >> 18;
            next.clear();
            let mut k = 0;
            while i < end && faces[i] >> 18 == line {
                let (u0, color) = ((faces[i] >> 8) & 0x3ff, faces[i] as u8);
                let mut u1 = u0 + 1;
                i += 1;
                while i < end && faces[i] >> 18 == line && (faces[i] >> 8) & 0x3ff == u1 && faces[i] as u8 == color {
                    u1 += 1;
                    i += 1;
                }
                let (u0, u1, row) = (u0 as u16, u1 as u16, (line & 0x3ff) as u16);
                while k < open.len() && rects[open[k]].u0 < u0 {
                    k += 1;
                }
                if let Some(&o) = open.get(k)
                    && rects[o].v1 == row
                    && rects[o].u0 == u0
                    && rects[o].u1 == u1
                    && rects[o].color == color
                {
                    rects[o].v1 += 1;
                    next.push(o);
                    k += 1;
                    continue;
                }
                rects.push(Rect {
                    bucket: bucket as u16,
                    color,
                    u0,
                    u1,
                    v0: row,
                    v1: row + 1,
                });
                next.push(rects.len() - 1);
            }
            std::mem::swap(&mut open, &mut next);
        }
    }

    let mut corners = Corners::new(size);
    let mut per_color = [0usize; 256];
    for r in &rects {
        let (u0, u1, v0, v1) = (r.u0.into(), r.u1.into(), r.v0.into(), r.v1.into());
        for (u, v) in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)] {
            corners.insert(r.point(u, v));
        }
        per_color[usize::from(r.color)] += 1;
    }
    // Most rectangles are two triangles on four corners, half of them shared.
    for (color, &n) in per_color.iter().enumerate().filter(|(_, n)| **n > 0) {
        let p = prims.entry(color as u8).or_default();
        p.positions.reserve(n * 2);
        p.normals.reserve(n * 2);
        p.indices.reserve(n * 6);
    }

    let pivot = if centered {
        size.map(|p| (p / 2) as f32)
    } else {
        [0.0; 3]
    };
    // MagicaVoxel is Z-up, glTF Y-up: (x, y, z) -> (x, z, -y).
    let to_gltf = |p: [f32; 3]| [p[0] - pivot[0], p[2] - pivot[2], pivot[1] - p[1]];
    let on_box = |axis: usize, c: usize| c == 0 || c == size[axis];
    // The current plane's vertices, by color and point.
    let mut shared = Shared::new(size);
    let mut current = u16::MAX;
    // A rectangle's outline counter-clockwise from (u0, v0), each point's vertex, and two chains
    // of outline points to strip between.
    let mut ring: Vec<[usize; 2]> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let (mut a, mut b): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for r in &rects {
        let (axis, positive, ua, va) = DIRS[usize::from(r.bucket) / 1025];
        if r.bucket != current {
            shared.plane(size[ua] + 1);
            current = r.bucket;
        }
        let plane = usize::from(r.bucket) % 1025;
        let (u0, u1, v0, v1) = (r.u0.into(), r.u1.into(), r.v0.into(), r.v1.into());
        // The sides, counter-clockwise: from, to, and whether it runs along `u`.
        let sides: [([usize; 2], [usize; 2], bool); 4] = [
            ([u0, v0], [u1, v0], true),
            ([u1, v0], [u1, v1], false),
            ([u1, v1], [u0, v1], true),
            ([u0, v1], [u0, v0], false),
        ];
        ring.clear();
        let mut at = [0usize; 4];
        let mut clean = [true; 4];
        for (s, &(from, to, along_u)) in sides.iter().enumerate() {
            at[s] = ring.len();
            ring.push(from);
            let (fixed_axis, fixed, a, b) = if along_u {
                (va, from[1], from[0], to[0])
            } else {
                (ua, from[0], from[1], to[1])
            };
            let every = on_box(axis, plane) || on_box(fixed_axis, fixed);
            for t in 1..a.abs_diff(b) {
                let c = if b > a { a + t } else { a - t };
                let q = if along_u { [c, fixed] } else { [fixed, c] };
                if every || corners.contains(r.point(q[0], q[1])) {
                    ring.push(q);
                    clean[s] = false;
                }
            }
        }
        let p = prims.entry(r.color).or_default();
        let mut normal = [0.0f32; 3];
        normal[axis] = if positive { 1.0 } else { -1.0 };
        let normal = [normal[0], normal[2], -normal[1]];
        ids.clear();
        for q in &ring {
            let id = shared.get(r.color, q[0], q[1], || {
                let id = u32::try_from(p.positions.len())
                    .context(ErrorKind::LimitExceeded, "too many vertices for 32-bit indices")?;
                p.positions.push(to_gltf(r.point(q[0], q[1]).map(|c| c as f32)));
                p.normals.push(normal);
                Ok(id)
            })?;
            ids.push(id);
        }
        let n = ids.len();
        if let Some(c) = (0..4).find(|&c| clean[c] && clean[(c + 3) % 4]) {
            // A corner between two unsplit sides fans to every other point without a sliver.
            for t in 1..n - 1 {
                p.indices
                    .extend([ids[at[c]], ids[(at[c] + t) % n], ids[(at[c] + t + 1) % n]]);
            }
        } else if clean[1] && clean[3] {
            // Unsplit left and right: a strip between bottom and top, both walked along `u`.
            a.clear();
            b.clear();
            a.extend(at[0]..=at[1]);
            b.extend((at[2]..=at[3]).rev());
            strip(p, &ring, &ids, &a, &b, 0);
        } else if clean[0] && clean[2] {
            // Unsplit bottom and top: a strip between right and left, both walked along `v`.
            a.clear();
            b.clear();
            a.extend(at[1]..=at[2]);
            b.push(0);
            b.extend((at[3]..n).rev());
            strip(p, &ring, &ids, &a, &b, 1);
        } else {
            // A point in the middle, which no edge of any other rectangle passes through.
            let id = u32::try_from(p.positions.len())
                .context(ErrorKind::LimitExceeded, "too many vertices for 32-bit indices")?;
            let mut m = r.point(0, 0).map(|c| c as f32);
            m[ua] = (u0 + u1) as f32 / 2.0;
            m[va] = (v0 + v1) as f32 / 2.0;
            p.positions.push(to_gltf(m));
            p.normals.push(normal);
            for t in 0..n {
                p.indices.extend([id, ids[t], ids[(t + 1) % n]]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "small test models, and exact unit normals"
)]
mod tests {
    use super::*;

    /// Every rotation byte's translation, rotation and scale rebuild its matrix, turned Y-up.
    #[test]
    fn every_rotation_byte_round_trips_through_its_quaternion() {
        let mut seen = 0;
        for byte in 0..=127u8 {
            let (first, second) = (byte & 3, (byte >> 2) & 3);
            if first > 2 || second > 2 || first == second {
                continue;
            }
            seen += 1;
            let x = Xform::from_rotation_byte(byte);
            let (_, [qx, qy, qz, qw], s) = gltf_trs(&x);
            let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "byte {byte}: |q| = {len}");
            // The quaternion's matrix, times the scale on X.
            let q = [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ];
            // Y-up: rows and columns (x, y, z) -> (x, z, -y).
            let c = |i: usize| match i {
                0 => (0, 1),
                1 => (2, 1),
                _ => (1, -1),
            };
            for (i, row) in q.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let ((ri, si), (rj, sj)) = (c(i), c(j));
                    #[allow(clippy::cast_precision_loss, reason = "-1, 0 or 1")]
                    let want = (x.r[ri][rj] * si * sj) as f32;
                    let got = v * s[j];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "byte {byte}: [{i}][{j}] is {got}, expected {want}"
                    );
                }
            }
        }
        assert_eq!(seen, 48);
    }

    /// A `.vox` file of one model: its size, voxels and, when given, palette.
    fn file(size: [i32; 3], voxels: &[[u8; 4]], extra: &[u8]) -> Vec<u8> {
        let chunk = |id: &[u8], content: &[u8]| {
            let mut c = id.to_vec();
            c.extend(u32::try_from(content.len()).unwrap().to_le_bytes());
            c.extend(0u32.to_le_bytes());
            c.extend(content);
            c
        };
        let mut children = chunk(b"SIZE", &size.map(i32::to_le_bytes).concat());
        let mut xyzi = u32::try_from(voxels.len()).unwrap().to_le_bytes().to_vec();
        xyzi.extend(voxels.concat());
        children.extend(chunk(b"XYZI", &xyzi));
        children.extend(extra);
        let mut f = b"VOX ".to_vec();
        f.extend(150u32.to_le_bytes());
        f.extend(b"MAIN");
        f.extend(0u32.to_le_bytes());
        f.extend(u32::try_from(children.len()).unwrap().to_le_bytes());
        f.extend(children);
        f
    }

    /// A shell `thick` voxels deep inside a ball filling `n`³, banded in eight colors by height.
    fn ball(n: i32, thick: f32) -> Vec<[u8; 4]> {
        let r = n as f32 / 2.0;
        let mut v = Vec::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d = [x, y, z].map(|c| c as f32 + 0.5 - r);
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if len <= r && len > r - thick {
                        v.push([x as u8, y as u8, z as u8, 1 + (z * 8 / n) as u8]);
                    }
                }
            }
        }
        v
    }

    /// Rolling hills `n` wide, colored by height, solid below.
    fn terrain(n: i32) -> Vec<[u8; 4]> {
        let mut v = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let h = 24.0 + 12.0 * ((x as f32 / 23.0).sin() + (y as f32 / 31.0).cos());
                for z in 0..h as i32 {
                    v.push([x as u8, y as u8, z as u8, 1 + (z / 6) as u8]);
                }
            }
        }
        v
    }

    /// Run with `cargo test --release -- --ignored --nocapture measure_meshing`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn measure_meshing() {
        for (name, n, thick) in [
            ("256 shell", 256, 6.0),
            ("196 ball", 196, 196.0),
            ("256 terrain", 256, 0.0),
        ] {
            let voxels = if thick > 0.0 { ball(n, thick) } else { terrain(n) };
            let bytes = file([n; 3], &voxels, &[]);
            let start = std::time::Instant::now();
            let doc = vox(&bytes, Keep::ALL).unwrap();
            let load = start.elapsed();
            let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
            let verts: usize = mesh.prims.iter().map(|p| p.positions.len()).sum();
            println!(
                "{name}: {} voxels, {verts} vertices, {} triangles, {load:?}, {} buffer bytes",
                voxels.len(),
                mesh.triangles(),
                doc.buffer(0).map_or(0, |b| b.len())
            );
        }
    }

    /// Checks a meshed model against the voxels it came from: every triangle faces the way its
    /// normal says, the area facing each way is the number of exposed faces that way, no vertex
    /// sits inside another triangle's edge, and every edge is walked as often each way.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "small test models"
    )]
    fn check_surface(model: &Model, prims: &HashMap<u8, Prim>) {
        let key = |p: [f32; 3]| p.map(|c| (c * 2.0).round() as i32);
        let mut points: Vec<[i32; 3]> = Vec::new();
        let mut area = [0.0f32; 6];
        let mut edges: HashMap<([i32; 3], [i32; 3]), i32> = HashMap::new();
        for p in prims.values() {
            points.extend(p.positions.iter().map(|&q| key(q)));
            for t in p.indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|k| p.positions[t[k] as usize]);
                let n = crate::mesh::face_normal(a, b, c);
                let want = p.normals[t[0] as usize];
                assert!(
                    (0..3).all(|k| (n[k] - want[k]).abs() < 1e-4),
                    "triangle faces {n:?}, not {want:?}"
                );
                let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let x = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let twice = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                assert!(twice > 1e-4, "a sliver triangle");
                let dir = (0..6)
                    .find(|&d| {
                        let (axis, positive, _, _) = DIRS[d];
                        let mut m = [0.0; 3];
                        m[axis] = if positive { 1.0 } else { -1.0 };
                        [m[0], m[2], -m[1]] == want
                    })
                    .unwrap();
                area[dir] += twice / 2.0;
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    *edges.entry((key(u), key(v))).or_default() += 1;
                    *edges.entry((key(v), key(u))).or_default() -= 1;
                }
            }
        }
        for ((u, v), n) in &edges {
            assert_eq!(*n, 0, "edge {u:?}-{v:?} isn't closed");
            for q in &points {
                // Strictly inside the segment: collinear and between the ends.
                let d = [v[0] - u[0], v[1] - u[1], v[2] - u[2]];
                let w = [q[0] - u[0], q[1] - u[1], q[2] - u[2]];
                let cross = [
                    d[1] * w[2] - d[2] * w[1],
                    d[2] * w[0] - d[0] * w[2],
                    d[0] * w[1] - d[1] * w[0],
                ];
                let along = d[0] * w[0] + d[1] * w[1] + d[2] * w[2];
                let len = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                assert!(
                    cross != [0; 3] || along <= 0 || along >= len,
                    "T-junction: {q:?} inside edge {u:?}-{v:?}"
                );
            }
        }
        let [sx, sy, sz] = model.size.map(|s| s as usize);
        let mut grid = vec![false; sx * sy * sz];
        for v in &model.voxels {
            grid[(usize::from(v[2]) * sy + usize::from(v[1])) * sx + usize::from(v[0])] = true;
        }
        let filled = |p: [i32; 3]| {
            p.iter().zip([sx, sy, sz]).all(|(&c, s)| c >= 0 && (c as usize) < s)
                && grid[(p[2] as usize * sy + p[1] as usize) * sx + p[0] as usize]
        };
        let mut want = [0.0f32; 6];
        for z in 0..sz as i32 {
            for y in 0..sy as i32 {
                for x in 0..sx as i32 {
                    if !filled([x, y, z]) {
                        continue;
                    }
                    for (d, &(axis, positive, _, _)) in DIRS.iter().enumerate() {
                        let mut n = [x, y, z];
                        n[axis] += if positive { 1 } else { -1 };
                        if !filled(n) {
                            want[d] += 1.0;
                        }
                    }
                }
            }
        }
        for d in 0..6 {
            assert!(
                (area[d] - want[d]).abs() < 1e-3,
                "direction {d}: area {} of {}",
                area[d],
                want[d]
            );
        }
    }

    /// Random models of a few colors mesh into closed surfaces without T-junctions, fewer
    /// triangles than two to a face.
    #[test]
    fn meshing_is_closed_and_free_of_t_junctions() {
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..40 {
            let size = [
                3 + (next() % 9) as i32,
                3 + (next() % 9) as i32,
                3 + (next() % 9) as i32,
            ];
            let colors = 1 + next() % 4;
            let fill = 30 + next() % 70;
            let mut voxels = Vec::new();
            for z in 0..size[2] as u8 {
                for y in 0..size[1] as u8 {
                    for x in 0..size[0] as u8 {
                        // Lower layers fuller, so there are big flat runs as well as noise.
                        if next() % 100 < fill.saturating_sub(u32::from(z) * 8).max(5) {
                            voxels.push([x, y, z, 1 + (next() % colors) as u8]);
                        }
                    }
                }
            }
            let model = Model { size, voxels };
            for centered in [false, true] {
                let mut prims = HashMap::new();
                mesh_model(&model, centered, &mut prims).unwrap();
                check_surface(&model, &prims);
                let tris: usize = prims.values().map(|p| p.indices.len() / 3).sum();
                assert!(tris > 0 || model.voxels.is_empty(), "round {round}: nothing meshed");
            }
        }
    }

    /// A solid block of one color clear of its model's box is six rectangles: twelve triangles
    /// on eight shared corners. Filling the box, its sides split at every voxel, to meet the
    /// faces of a model placed against it.
    #[test]
    fn a_block_is_twelve_triangles() {
        let block = |at: u8| {
            let mut voxels = Vec::new();
            for z in 0..4 {
                for y in 0..5 {
                    for x in 0..6 {
                        voxels.push([x + at, y + at, z + at, 7]);
                    }
                }
            }
            voxels
        };
        let model = Model {
            size: [8, 7, 6],
            voxels: block(1),
        };
        let mut prims = HashMap::new();
        mesh_model(&model, false, &mut prims).unwrap();
        check_surface(&model, &prims);
        assert_eq!((prims[&7].positions.len(), prims[&7].indices.len()), (24, 36));

        let model = Model {
            size: [6, 5, 4],
            voxels: block(0),
        };
        let mut prims = HashMap::new();
        mesh_model(&model, true, &mut prims).unwrap();
        check_surface(&model, &prims);
        // The +Z side: 6 + 5 + 6 + 5 outline points around a middle one.
        let top = prims[&7].normals.iter().filter(|n| n[1] > 0.5).count();
        assert_eq!(top, 2 * (6 + 1) + 2 * (5 + 1) - 4 + 1);
    }

    /// A chunk with no children.
    fn chunk(id: &[u8], content: &[u8]) -> Vec<u8> {
        let mut c = id.to_vec();
        c.extend(u32::try_from(content.len()).unwrap().to_le_bytes());
        c.extend(0u32.to_le_bytes());
        c.extend(content);
        c
    }

    /// A `DICT` of `pairs`.
    fn dict(pairs: &[(&str, &str)]) -> Vec<u8> {
        let mut d = u32::try_from(pairs.len()).unwrap().to_le_bytes().to_vec();
        for s in pairs.iter().flat_map(|(k, v)| [k, v]) {
            d.extend(u32::try_from(s.len()).unwrap().to_le_bytes());
            d.extend(s.as_bytes());
        }
        d
    }

    /// Glass takes its opacity from its transparency, a legacy `MATT` chunk's glass too, and a
    /// shape with a model per frame draws the first and keeps the second as a hidden child.
    #[test]
    fn glass_opacity_and_frame_models() {
        let mut extra = chunk(b"SIZE", &[2i32, 2, 2].map(i32::to_le_bytes).concat());
        extra.extend(chunk(b"XYZI", &[2, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2]));
        let mut trn = 0u32.to_le_bytes().to_vec();
        trn.extend(dict(&[]));
        for v in [1i32, -1, -1, 1] {
            trn.extend(v.to_le_bytes());
        }
        trn.extend(dict(&[]));
        extra.extend(chunk(b"nTRN", &trn));
        let mut shp = 1u32.to_le_bytes().to_vec();
        shp.extend(dict(&[]));
        shp.extend(2u32.to_le_bytes());
        shp.extend(1u32.to_le_bytes());
        shp.extend(dict(&[("_f", "3")]));
        shp.extend(0u32.to_le_bytes());
        shp.extend(dict(&[("_f", "0")]));
        extra.extend(chunk(b"nSHP", &shp));
        let mut matl = 1u32.to_le_bytes().to_vec();
        matl.extend(dict(&[("_type", "_glass"), ("_trans", "0.75")]));
        extra.extend(chunk(b"MATL", &matl));
        let mut matt = Vec::new();
        for v in [2, 2, 0.5f32.to_bits(), 0b10, 0.25f32.to_bits()] {
            matt.extend(v.to_le_bytes());
        }
        extra.extend(chunk(b"MATT", &matt));

        let doc = vox(&file([1, 1, 1], &[[0, 0, 0, 1]], &extra), Keep::ALL).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("VOX"));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert_eq!(info.handedness, Fact::Assumed(Handedness::Right));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert!(info.converted.up_axis && !info.converted.units);
        let materials = doc.json["materials"].as_array().unwrap();
        let alpha = |m: &Value| m["pbrMetallicRoughness"]["baseColorFactor"][3].as_f64().unwrap();
        assert!((alpha(&materials[0]) - 0.25).abs() < 1e-6, "{}", materials[0]);
        assert_eq!(materials[0]["alphaMode"], "BLEND");
        assert!((alpha(&materials[1]) - 0.5).abs() < 1e-6, "{}", materials[1]);
        assert!(
            (materials[1]["pbrMetallicRoughness"]["roughnessFactor"]
                .as_f64()
                .unwrap()
                - 0.25)
                .abs()
                < 1e-6
        );

        let nodes = doc.json["nodes"].as_array().unwrap();
        let shape = nodes.iter().find(|n| n.get("children").is_some()).unwrap();
        let alternate = &nodes[shape["children"][0].as_u64().unwrap() as usize];
        assert_eq!(alternate["name"], "frame_3");
        assert_eq!(alternate["extensions"]["KHR_node_visibility"]["visible"], false);
        assert_ne!(alternate["mesh"], shape["mesh"]);
        assert_eq!(doc.facts["voxels"], 1);
        // Only frame 0's one voxel is drawn, not frame 3's two.
        assert_eq!(
            doc.flatten(crate::document::Pose::Rest).unwrap().unwrap().triangles(),
            12
        );
    }

    /// The voxel limit rises with the triangle limit: a file claiming one voxel past the default is refused for its
    /// size, while a higher limit gets as far as finding the voxels missing.
    #[test]
    fn a_triangle_limit_past_the_default_lets_more_voxels_in() {
        let mut f = file([1, 1, 1], &[], &[]);
        let at = f.windows(4).position(|w| w == b"XYZI").unwrap() + 12;
        f[at..at + 4].copy_from_slice(&u32::try_from(MAX_VOXELS + 1).unwrap().to_le_bytes());
        let err = |keep| format!("{:#}", vox(&f, keep).unwrap_err());
        assert!(err(Keep::ALL).contains("too many voxels"), "{}", err(Keep::ALL));
        let lower = Keep {
            max_triangles: 10,
            ..Keep::ALL
        };
        assert!(err(lower).contains("too many voxels"), "{}", err(lower));
        let higher = Keep {
            max_triangles: crate::mesh::MAX_TRIANGLES * 2,
            ..Keep::ALL
        };
        assert!(err(higher).contains("voxels are cut short"), "{}", err(higher));
    }
}
