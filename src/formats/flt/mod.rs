//! `OpenFlight` (`.flt`): the binary scene format of visual simulation, a stream of big-endian
//! records: a header, palettes (colors, materials, textures, light sources), a vertex palette,
//! and a tree of groups, objects, levels of detail, degrees of freedom, switches, instances and
//! references to other files, whose faces and meshes name the vertices they join. Versions 14.2
//! to 16.x are read, and older files (their integer vertices and smaller palettes) as far as
//! they go.
//!
//! Each node of the tree becomes a glTF node, its faces one mesh, a primitive for each
//! material. A face's material is its material palette entry (diffuse as the base color,
//! emissive, specular color and shininess as the highlight, alpha), its texture, whether it has two sides (its draw
//! type), its transparency and whether it's lit (its light mode); the face's color, or its
//! vertices' colors, go in `COLOR_0`, which glTF multiplies with the base color as `OpenFlight`
//! does. Wireframe faces are drawn as lines, light points as points. Texture layers have
//! coordinate sets of their own; a bump-map layer is the normal texture, the others stay in
//! `extras`. A morph vertex list makes a morph target. SGI images (`.rgb`, `.rgba`, `.int`,
//! `.inta`) are decoded and embedded as PNG.
//!
//! Matrix records, degrees of freedom (at their current values) and replication place nodes.
//! Instances share their meshes. Referenced files are read through the assets, with their own
//! units and, as the reference's flags say, their own or the parent's palettes. Of the levels
//! of detail sharing a center the most detailed is kept, the ranges in `extras`; a switch keeps
//! the children its current mask shows; an animated group shows its children in turn, each
//! scaled to nothing when it's not its turn. Light sources become `KHR_lights_punctual` lights.
//!
//! `OpenFlight` is Z up, counterclockwise from the front, in the header's units: the document's
//! `RootNode`, over the header's node, turns it Y up and scales it to meters. What has no glTF slot (the header's placement on the
//! earth, ranges, comments, record kinds) stays in `extras`.
//!
//! The asset's `extras.source` records the version, the header's units, Z up, what was
//! converted, the header's date as `created`, and, when the header places the database on the
//! earth, its origin, corners, projection and ellipsoid as the `georeference`.

mod parse;
mod sgi;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::frame::{self, Frame};
use super::keys::{self, Interp, Track};
use super::xform::col::{self, IDENTITY, mul, scaling, translation};
use super::xform::set_trs;
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::{Images, Location};
use crate::legacy_material::{self, Phong, Rgb, Transparency};
use crate::material::{Alpha, Full, Tex};
use crate::mesh::{FastMap, Image, Prim, linear_u8, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::quat;
use crate::source::{Axis, Converted, Fact, Handedness, SourceInfo};
use crate::vec3::{self, cross, dot, sub, unit};
use parse::{Db, F_HIDDEN, F_PACKED, Face, Kind, Shape, V_INDEX, V_NORMAL, V_PACKED, V_UV};

/// The most glTF nodes a scene expands to (instances and references multiply them).
const MAX_NODES: usize = 1 << 18;
/// How deep instances may nest in instances.
const MAX_INSTANCE_DEPTH: u8 = 32;
/// How deep references to files may nest.
const MAX_EXTERNAL_DEPTH: u8 = 16;
/// The most files a scene may read.
const MAX_FILES: usize = 4096;
/// Ear clipping costs the cube of a polygon's corners; past this much in all, polygons are fanned.
const CLIP_BUDGET: u64 = 1 << 31;
/// A mesh further than this from its node's origin is moved near it, the node placing it back.
const FAR: f64 = 65536.0;
/// The most frames an animated group is animated with.
const MAX_FRAMES: usize = 1024;
/// Face flag: no color (white).
const F_NO_COLOR: u32 = 0x8000_0000 >> 1;
/// Texture layer effect: a bump map.
const BUMP: i16 = 1;

/// A file of the scene.
struct File {
    db: Db,
    dir: PathBuf,
    name: String,
}

/// The files whose palettes a part of the scene reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Pal {
    colors: u32,
    materials: u32,
    textures: u32,
    lights: u32,
}

/// What makes a glTF material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MatKey {
    /// The material palette's file and index; `None` for the default.
    material: Option<(u32, i16)>,
    /// The texture palette's file and index.
    texture: Option<(u32, i16)>,
    /// The texture layers (multitexture): their textures and effects.
    layers: [(i16, i16); 7],
    /// The layers whose coordinates the primitive has.
    layer_mask: u8,
    double_sided: bool,
    blend: bool,
    unlit: bool,
}

/// A primitive being built.
#[derive(Default)]
struct Part2 {
    prim: Prim,
    /// Each layer's coordinates, for the layers the key has.
    layers: [Vec<[f32; 2]>; 7],
    /// Each vertex's morph target delta.
    morph: Vec<[f32; 3]>,
    /// A vertex has coordinates, or the primitive is textured.
    uvs: bool,
    /// A vertex isn't opaque white.
    colored: bool,
    /// Each vertex by its vertex, its face (when it takes the face's normal or color) and its
    /// corner (when it has a corner's layer coordinates or morph target).
    at: FastMap<(u32, u32, u32), u32>,
}

/// A texture found: its image (`None` when textures aren't kept) and whether it has
/// transparent pixels.
type Found = (Option<usize>, bool);

/// A mesh built for a node: its index and the offset its positions are relative to.
type MeshOf = Option<(usize, [f64; 3])>;

/// The state of a load.
struct Ctx<'a> {
    assets: &'a dyn Assets,
    keep: Keep,
    limit: usize,
    files: Vec<File>,
    by_path: FastMap<PathBuf, Option<u32>>,
    b: Builder,
    images: Images,
    image_of: FastMap<(u32, i16), Option<Found>>,
    materials: Vec<Full>,
    material_at: FastMap<MatKey, usize>,
    meshes: Vec<Value>,
    /// Each built mesh and the triangles it draws.
    mesh_of: FastMap<(u32, u32, Pal), (MeshOf, usize)>,
    lights: Vec<Value>,
    clip: u64,
    /// The triangles drawn, each placement of a mesh counted.
    triangles: usize,
    /// The triangles built, each mesh counted once; bounded by the limit.
    built_triangles: usize,
    lines_or_points: bool,
    /// sRGB bytes in linear light.
    lut: [f32; 256],
    stats: Stats,
}

/// What a load drops or meets, counted.
#[derive(Default)]
struct Stats {
    hidden_faces: usize,
    dropped_textures: usize,
    /// The external references' files that weren't found, by the paths the file gives.
    missing_files: Vec<String>,
    missing_instances: usize,
    cycles: usize,
    too_deep: usize,
    dropped_nodes: usize,
    dropped_lods: usize,
    switched_off: usize,
    disabled_lights: usize,
    faces: usize,
}

pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let db = parse::parse(&data)?;
    drop(data);
    let dir = input.dir().to_path_buf();
    let name = input
        .path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let mut lut = [0.0f32; 256];
    for (k, v) in lut.iter_mut().enumerate() {
        *v = linear_u8(u8::try_from(k).unwrap_or(0));
    }
    let mut c = Ctx {
        assets,
        keep,
        limit: input.limit,
        files: vec![File {
            db,
            dir: dir.clone(),
            name,
        }],
        by_path: FastMap::default(),
        b: Builder::new(keep.crease_angle()),
        images: Images::new(!keep.image_bytes, &dir),
        image_of: FastMap::default(),
        materials: Vec::new(),
        material_at: FastMap::default(),
        meshes: Vec::new(),
        mesh_of: FastMap::default(),
        lights: Vec::new(),
        clip: CLIP_BUDGET,
        triangles: 0,
        built_triangles: 0,
        lines_or_points: false,
        lut,
        stats: Stats::default(),
    };
    // The file itself, as references to it would find it.
    c.by_path.insert(input.path.to_path_buf(), Some(0));
    if let Some(n) = input.path.file_name() {
        c.by_path.insert(dir.join(n), Some(0));
    }
    let scene = expand(&mut c)?;
    if c.meshes.is_empty() && c.lights.is_empty() {
        let cut = c.files.iter().any(|f| f.db.truncated);
        return Err(crate::Error::nothing_left(
            cut,
            "OpenFlight",
            "this OpenFlight file has no geometry",
        ));
    }
    Ok(finish(c, scene))
}

/// The expanded scene: glTF nodes, each node's children, and the animated groups' frames.
struct Scene {
    nodes: Vec<Value>,
    kids: Vec<Vec<usize>>,
    /// Animated groups: their playback and their frame nodes in order.
    sequences: Vec<(parse::Sequence, Vec<usize>)>,
}

/// A node of a file to place in the scene.
#[derive(Clone, Copy)]
struct Item {
    file: u32,
    node: u32,
    pal: Pal,
    parent: Option<usize>,
    /// The instances and files it's inside: an entry of `chain`.
    chain: u32,
    instances: u8,
    externals: u8,
    /// For a replicated copy, one more than the times its matrix moves it; 0 for the node
    /// itself.
    copy: u16,
    /// The scale from its file's units to its parent file's.
    scale: f64,
}

const NO_CHAIN: u32 = u32::MAX;

fn expand(c: &mut Ctx<'_>) -> Result<Scene> {
    let mut s = Scene {
        nodes: Vec::new(),
        kids: Vec::new(),
        sequences: Vec::new(),
    };
    // What each item is inside: (instance or file key, the entry it's in).
    let mut chain: Vec<(u64, u32)> = Vec::new();
    let mut stack = vec![Item {
        file: 0,
        node: 0,
        pal: Pal::default(),
        parent: None,
        chain: NO_CHAIN,
        instances: 0,
        externals: 0,
        copy: 0,
        scale: 1.0,
    }];
    while let Some(it) = stack.pop() {
        if s.nodes.len() >= MAX_NODES {
            c.stats.dropped_nodes += 1;
            continue;
        }
        let (fi, ni) = (it.file as usize, it.node as usize);
        let attach = |s: &mut Scene, json: Value| -> usize {
            let out = s.nodes.len();
            s.nodes.push(json);
            s.kids.push(Vec::new());
            if let Some(p) = it.parent {
                s.kids[p].push(out);
            }
            out
        };
        let node = &c.files[fi].db.nodes[ni];
        // A replicated node: a group of its copies, the first in place and each other moved by
        // the matrix once more.
        if it.copy == 0 && node.replicate > 0 && node.matrix.is_some() {
            let out = attach(&mut s, json!({ "name": display_name(node) }));
            for k in (1..=node.replicate.saturating_add(1)).rev() {
                if s.nodes.len() + stack.len() >= MAX_NODES {
                    c.stats.dropped_nodes += 1;
                    continue;
                }
                stack.push(Item {
                    parent: Some(out),
                    copy: k,
                    ..it
                });
            }
            continue;
        }
        let top = it.parent.is_none();
        let mut j = json!({ "name": display_name(node) });
        // Placement.
        let mut m = local_matrix(node, it.copy);
        if let Kind::Light { pos, yaw, pitch, .. } = node.kind {
            let d = light_direction(yaw, pitch);
            let r = col::rotation(from_to([0.0, 0.0, -1.0], d));
            m = mul(&m, &mul(&translation(pos), &r));
        }
        // The header stands where the file's space does: the root node over it turns and scales that.
        if !top {
            if (it.scale - 1.0).abs() > 0.0 {
                m = mul(&m, &scaling([it.scale; 3]));
            }
            place(&mut j, &m);
        }
        if c.keep.extras {
            j["extras"] = node_extras(&c.files[fi], node);
        }
        // A light source.
        if let Kind::Light { palette, flags, .. } = node.kind {
            light(c, it.pal, palette, flags, &mut j);
        }
        let out = attach(&mut s, j);
        // Its faces.
        let mesh = mesh_of(c, it.file, it.node, it.pal)?;
        if let Some((mesh, offset)) = mesh {
            if offset.iter().all(|&v| v == 0.0) {
                s.nodes[out]["mesh"] = json!(mesh);
            } else {
                let name = format!("{} mesh", s.nodes[out]["name"].as_str().unwrap_or_default());
                let k = s.nodes.len();
                s.nodes
                    .push(json!({ "name": name, "translation": offset, "mesh": mesh }));
                s.kids.push(Vec::new());
                s.kids[out].push(k);
            }
        }
        // What's under it.
        let node = &c.files[fi].db.nodes[ni];
        match &node.kind {
            Kind::InstanceRef(number) => {
                let Some(&def) = c.files[fi].db.instances.get(number) else {
                    c.stats.missing_instances += 1;
                    continue;
                };
                let key = (1u64 << 63) | (u64::from(it.file) << 32) | u64::from(def);
                if inside(&chain, it.chain, key) {
                    c.stats.cycles += 1;
                    continue;
                }
                if it.instances >= MAX_INSTANCE_DEPTH {
                    c.stats.too_deep += 1;
                    continue;
                }
                chain.push((key, it.chain));
                stack.push(Item {
                    node: def,
                    parent: Some(out),
                    chain: u32_of(chain.len() - 1),
                    instances: it.instances + 1,
                    copy: 0,
                    scale: 1.0,
                    ..it
                });
            }
            Kind::External { file, flags } => {
                let (path, target) = split_reference(file);
                let (path, target) = (path.to_owned(), target.map(str::to_owned));
                let flags = *flags;
                let Some(ef) = external(c, it.file, &path) else {
                    if !c.stats.missing_files.contains(&path) {
                        c.stats.missing_files.push(path);
                    }
                    continue;
                };
                let key = u64::from(ef);
                if inside(&chain, it.chain, key) || ef == 0 {
                    c.stats.cycles += 1;
                    continue;
                }
                if it.externals >= MAX_EXTERNAL_DEPTH {
                    c.stats.too_deep += 1;
                    continue;
                }
                chain.push((key, it.chain));
                let parent_db = &c.files[fi].db;
                // A clear bit: the parent's palette.
                let flags = if parent_db.version == 1541 { u32::MAX } else { flags };
                let own = |bit: u32| flags & (0x8000_0000 >> bit) != 0;
                let pal = Pal {
                    colors: if own(0) { ef } else { it.pal.colors },
                    materials: if own(1) { ef } else { it.pal.materials },
                    textures: if own(2) { ef } else { it.pal.textures },
                    lights: if own(5) || parent_db.version < 1510 {
                        ef
                    } else {
                        it.pal.lights
                    },
                };
                let edb = &c.files[ef as usize].db;
                let scale = edb.meters() / parent_db.meters();
                let node = target
                    .and_then(|t| edb.nodes.iter().position(|n| n.name == t))
                    .map_or(0, u32_of);
                stack.push(Item {
                    file: ef,
                    node,
                    pal,
                    parent: Some(out),
                    chain: u32_of(chain.len() - 1),
                    instances: it.instances,
                    externals: it.externals + 1,
                    copy: 0,
                    scale,
                });
            }
            kind => {
                let db = &c.files[fi].db;
                let mut kids = pick_lods(db, &node.kids, &mut c.stats.dropped_lods);
                if let Kind::Switch { current, words, masks } = kind {
                    let before = kids.len();
                    kids = switch_kids(&kids, *current, *words, masks);
                    c.stats.switched_off += before - kids.len();
                }
                let sequence = kind.sequence().filter(|_| kids.len() > 1);
                let mut frames = Vec::new();
                for (k, &kid) in kids.iter().enumerate().rev() {
                    if s.nodes.len() + stack.len() >= MAX_NODES {
                        c.stats.dropped_nodes += 1;
                        continue;
                    }
                    let mut parent = out;
                    if sequence.is_some() {
                        // A frame node, shown in its turn.
                        let f = s.nodes.len();
                        let mut j = json!({ "name": format!("frame{k}") });
                        if k > 0 {
                            j["scale"] = json!([0.0, 0.0, 0.0]);
                        }
                        s.nodes.push(j);
                        s.kids.push(Vec::new());
                        frames.push(f);
                        parent = f;
                    }
                    stack.push(Item {
                        node: kid,
                        parent: Some(parent),
                        copy: 0,
                        scale: 1.0,
                        ..it
                    });
                }
                if let Some(seq) = sequence {
                    frames.reverse();
                    s.kids[out].extend(&frames);
                    s.sequences.push((seq, frames));
                }
            }
        }
    }
    Ok(s)
}

/// Whether `key` is among the instances and files `at` is inside.
fn inside(chain: &[(u64, u32)], mut at: u32, key: u64) -> bool {
    while let Some(&(k, up)) = chain.get(at as usize) {
        if k == key {
            return true;
        }
        at = up;
    }
    false
}

/// A reference's file and, after it in angle brackets, the node of it to use.
fn split_reference(r: &str) -> (&str, Option<&str>) {
    if let Some(open) = r.find('<')
        && r.ends_with('>')
    {
        return (r[..open].trim(), Some(&r[open + 1..r.len() - 1]));
    }
    (r.trim(), None)
}

/// The file another file refers to, read; `None` when it's missing or not `OpenFlight`.
fn external(c: &mut Ctx<'_>, from: u32, reference: &str) -> Option<u32> {
    let dir = c.files[from as usize].dir.clone();
    let path = c.assets.find(reference, &dir)?;
    if let Some(&f) = c.by_path.get(&path) {
        return f;
    }
    let f = if c.files.len() >= MAX_FILES {
        None
    } else {
        c.assets.read(&path).and_then(|b| parse::parse(&b).ok()).map(|db| {
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            let name = path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            c.files.push(File { db, dir, name });
            u32_of(c.files.len() - 1)
        })
    };
    c.by_path.insert(path, f);
    f
}

/// The kids to keep of levels of detail: of those sharing a center, the most detailed (the
/// smallest switch-out distance); other kids stay.
fn pick_lods(db: &Db, kids: &[u32], dropped: &mut usize) -> Vec<u32> {
    let lod = |k: u32| match db.nodes[k as usize].kind {
        Kind::Lod { switch_out, center, .. } => Some((center.map(|c| (c * 1000.0).round().to_bits()), switch_out)),
        _ => None,
    };
    if kids.iter().filter(|&&k| lod(k).is_some()).count() < 2 {
        return kids.to_vec();
    }
    let mut best: FastMap<[u64; 3], f64> = FastMap::default();
    for &k in kids {
        if let Some((c, out)) = lod(k) {
            let e = best.entry(c).or_insert(out);
            *e = e.min(out);
        }
    }
    let keep: Vec<u32> = kids
        .iter()
        .copied()
        .filter(|&k| lod(k).is_none_or(|(c, out)| best.get(&c).is_none_or(|&b| out <= b)))
        .collect();
    *dropped += kids.len() - keep.len();
    keep
}

/// The kids a switch's current mask shows (all of them when it has no such mask).
fn switch_kids(kids: &[u32], current: u32, words: u32, masks: &[u32]) -> Vec<u32> {
    let words = words as usize;
    let Some(base) = (current as usize).checked_mul(words) else {
        return kids.to_vec();
    };
    if words == 0 || base.saturating_add(words) > masks.len() {
        return kids.to_vec();
    }
    kids.iter()
        .enumerate()
        .filter(|&(n, _)| masks.get(base + n / 32).is_some_and(|w| w & (1 << (n % 32)) != 0))
        .map(|(_, &k)| k)
        .collect()
}

/// A node's name: its own, or its record's.
fn display_name(n: &parse::Node) -> String {
    match &n.kind {
        _ if !n.name.is_empty() => n.name.clone(),
        Kind::InstanceRef(k) => format!("instance{k}"),
        Kind::External { file, .. } => file.clone(),
        k => k.name().to_owned(),
    }
}

/// A node's `extras`: its record and what glTF has no place for.
fn node_extras(f: &File, n: &parse::Node) -> Value {
    let mut e = Map::new();
    e.insert("type".into(), json!(n.kind.name()));
    match &n.kind {
        Kind::Root => {
            e.insert("file".into(), json!(f.name));
        }
        Kind::Group {
            flags,
            loop_count,
            loop_duration,
            last_frame,
        } => {
            if *flags != 0 {
                e.insert("flags".into(), json!(flags));
            }
            if n.kind.sequence().is_some() {
                e.insert("loopCount".into(), json!(loop_count));
                e.insert("loopDuration".into(), json!(loop_duration));
                e.insert("lastFrameDuration".into(), json!(last_frame));
            }
        }
        Kind::Object { flags, transparency } => {
            if *flags != 0 {
                e.insert("flags".into(), json!(flags));
            }
            if *transparency != 0 {
                e.insert("transparency".into(), json!(transparency));
            }
        }
        Kind::Lod {
            switch_in,
            switch_out,
            center,
            flags,
        } => {
            e.insert("switchIn".into(), json!(switch_in));
            e.insert("switchOut".into(), json!(switch_out));
            e.insert("center".into(), json!(center));
            if *flags != 0 {
                e.insert("flags".into(), json!(flags));
            }
        }
        Kind::Dof(d) => {
            e.insert("origin".into(), json!(d.origin));
            let names = ["z", "y", "x", "pitch", "roll", "yaw", "scaleZ", "scaleY", "scaleX"];
            let mut r = Map::new();
            for (name, v) in names.iter().zip(&d.ranges) {
                if v.iter().any(|&x| x != 0.0) {
                    r.insert(
                        (*name).into(),
                        json!({ "min": v[0], "max": v[1], "current": v[2], "increment": v[3] }),
                    );
                }
            }
            e.insert("ranges".into(), Value::Object(r));
            e.insert("flags".into(), json!(d.flags));
        }
        Kind::Switch { current, words, masks } => {
            e.insert("currentMask".into(), json!(current));
            if *words > 0 {
                let masks: Vec<&[u32]> = masks.chunks(*words as usize).take(256).collect();
                e.insert("masks".into(), json!(masks));
            }
        }
        Kind::InstanceDef(k) | Kind::InstanceRef(k) => {
            e.insert("instance".into(), json!(k));
        }
        Kind::External { file, flags } => {
            e.insert("file".into(), json!(file));
            e.insert("flags".into(), json!(flags));
        }
        Kind::Light { palette, flags, .. } => {
            e.insert("palette".into(), json!(palette));
            e.insert("flags".into(), json!(flags));
        }
        _ => {}
    }
    if let Some(m) = &n.comment {
        e.insert("comment".into(), json!(m));
    }
    if n.replicate > 0 {
        e.insert("replicate".into(), json!(n.replicate));
    }
    for (k, v) in &n.extras {
        e.insert(k.clone(), v.clone());
    }
    json!({ "openflight": e })
}

/// A light source's light, added to `j`.
fn light(c: &mut Ctx<'_>, pal: Pal, palette: i32, flags: u32, j: &mut Value) {
    if flags & 0x8000_0000 == 0 {
        c.stats.disabled_lights += 1;
        return;
    }
    let lp = c.files[pal.lights as usize]
        .db
        .lights
        .get(&palette)
        .cloned()
        .unwrap_or_default();
    // In linear light, as the material colors are.
    let rgb = [0, 1, 2].map(|k| srgb_to_linear(lp.diffuse[k].clamp(0.0, 1.0)));
    let color = if rgb.iter().all(|&v| v <= 0.0) { [1.0; 3] } else { rgb };
    let mut l = json!({ "color": color, "intensity": 1.0 });
    match lp.kind {
        0 => l["type"] = json!("directional"),
        2 => {
            l["type"] = json!("spot");
            let outer = f64::from(lp.spot_cutoff)
                .to_radians()
                .clamp(1e-3, std::f64::consts::FRAC_PI_2);
            l["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": outer });
        }
        _ => l["type"] = json!("point"),
    }
    if !lp.name.is_empty() {
        l["name"] = json!(lp.name);
    }
    if c.keep.extras {
        l["extras"] = json!({ "openflight": {
            "ambient": lp.ambient, "specular": lp.specular, "spotExponent": lp.spot_exponent,
            "attenuation": lp.attenuation, "global": flags & 0x4000_0000 != 0,
        } });
    }
    c.lights.push(l);
    j["extensions"] = json!({ "KHR_lights_punctual": { "light": c.lights.len() - 1 } });
}

/// The direction a light at `yaw` and `pitch` (degrees) shines: yaw from +Y toward -X, pitch up.
fn light_direction(yaw: f32, pitch: f32) -> [f64; 3] {
    let (y, p) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
    [-y.sin() * p.cos(), y.cos() * p.cos(), p.sin()]
}

// Matrices are column-major, as glTF's: `m[col * 4 + row]`.

/// A rotation of `deg` degrees about axis `axis` (0 x, 1 y, 2 z).
fn rotation(axis: usize, deg: f64) -> [f64; 16] {
    let (s, c) = deg.to_radians().sin_cos();
    let mut m = IDENTITY;
    let (a, b) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    m[a * 4 + a] = c;
    m[a * 4 + b] = s;
    m[b * 4 + a] = -s;
    m[b * 4 + b] = c;
    m
}

/// The shortest rotation taking unit vector `a` to `b`, as a quaternion.
fn from_to(a: [f64; 3], b: [f64; 3]) -> [f64; 4] {
    let len = dot(b, b).sqrt();
    if len <= 0.0 || !len.is_finite() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let b = b.map(|v| v / len);
    let d = dot(a, b);
    if d < -0.999_999 {
        // Opposite: half a turn about an axis across `a`.
        let axis = if a[0].abs() < 0.9 {
            cross(a, [1.0, 0.0, 0.0])
        } else {
            cross(a, [0.0, 1.0, 0.0])
        };
        let l = dot(axis, axis).sqrt();
        return [axis[0] / l, axis[1] / l, axis[2] / l, 0.0];
    }
    let x = cross(a, b);
    let q = [x[0], x[1], x[2], 1.0 + d];
    let l = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    q.map(|v| v / l)
}

/// A node's own placement: its degree of freedom at its current values, under its matrix
/// record (for a replicated copy, the matrix raised to the power `copy - 1`).
fn local_matrix(n: &parse::Node, copy: u16) -> [f64; 16] {
    let mut m = n.matrix.map_or(IDENTITY, |mut m| {
        // A projective row isn't a placement.
        m[3] = 0.0;
        m[7] = 0.0;
        m[11] = 0.0;
        m[15] = 1.0;
        m
    });
    if copy > 0 {
        // The matrix to the power `copy - 1`, by squaring.
        let (mut base, mut k, mut out) = (m, copy - 1, IDENTITY);
        while k > 0 {
            if k & 1 == 1 {
                out = mul(&out, &base);
            }
            base = mul(&base, &base);
            k >>= 1;
        }
        m = out;
    }
    if let Kind::Dof(d) = &n.kind {
        m = mul(&m, &dof_matrix(d));
    }
    m
}

/// A degree of freedom's transform at its current values: within its frame (its origin, x
/// axis toward its second point, xy plane through its third), translated, rotated (yaw about
/// z, pitch about x, roll about y) and scaled.
fn dof_matrix(d: &parse::Dof) -> [f64; 16] {
    let o = d.origin;
    let frame = (|| {
        let x = unit(sub(d.on_x, o))?;
        let z = unit(cross(x, sub(d.in_xy, o)))?;
        let y = cross(z, x);
        Some([
            x[0], x[1], x[2], 0.0, y[0], y[1], y[2], 0.0, z[0], z[1], z[2], 0.0, o[0], o[1], o[2], 1.0,
        ])
    })()
    .unwrap_or_else(|| translation(o));
    // Its inverse: the transposed rotation, and the origin taken back.
    let mut inv = IDENTITY;
    for r in 0..3 {
        for col in 0..3 {
            inv[col * 4 + r] = frame[r * 4 + col];
        }
    }
    for r in 0..3 {
        inv[12 + r] = -(0..3).map(|k| inv[k * 4 + r] * frame[12 + k]).sum::<f64>();
    }
    let cur = |k: usize| d.ranges[k][2];
    let scale = |k: usize| if cur(k) == 0.0 { 1.0 } else { cur(k) };
    let local = mul(
        &translation([cur(2), cur(1), cur(0)]),
        &mul(
            &mul(&rotation(2, cur(5)), &mul(&rotation(0, cur(3)), &rotation(1, cur(4)))),
            &scaling([scale(8), scale(7), scale(6)]),
        ),
    );
    mul(&frame, &mul(&local, &inv))
}

/// Sets a node's placement: translation, rotation and scale when the matrix comes apart into
/// them, or the matrix.
fn place(j: &mut Value, m: &[f64; 16]) {
    if m.iter().zip(IDENTITY).all(|(a, b)| (a - b).abs() < 1e-12) || m.iter().any(|v| !v.is_finite()) {
        return;
    }
    let t = [m[12], m[13], m[14]];
    let mut cols = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let mut s = cols.map(|c| dot(c, c).sqrt());
    let orthogonal = s.iter().all(|&v| v > 1e-12)
        && [(0, 1), (0, 2), (1, 2)]
            .iter()
            .all(|&(a, b)| dot(cols[a], cols[b]).abs() <= 1e-6 * s[a] * s[b]);
    if !orthogonal {
        j["matrix"] = json!(m);
        return;
    }
    if dot(cross(cols[0], cols[1]), cols[2]) < 0.0 {
        s[0] = -s[0];
    }
    for k in 0..3 {
        cols[k] = cols[k].map(|v| v / s[k]);
    }
    let q = quat::positive_w(quat::unit_unchecked(quat::of_rotation(|r, c| cols[c][r])));
    set_trs(j, (t, q, s));
}

/// The color of palette entry `index` (with its intensity in the low 7 bits), as sRGB bytes.
fn palette(db: &Db, index: u32) -> [u8; 3] {
    let (entry, intensity) = if db.old_colors && index & 0x1000 != 0 {
        ((index & 0xFFF) as usize + 32, None)
    } else {
        ((index >> 7) as usize, Some(index & 0x7F))
    };
    let Some(&c) = db.colors.get(entry) else {
        return [255; 3];
    };
    match intensity {
        Some(i) => c.map(|v| u8::try_from(u32::from(v) * i / 127).unwrap_or(255)),
        None => c,
    }
}

/// A face's own color, as sRGB bytes.
fn face_color(db: &Db, f: &Face) -> [u8; 3] {
    if (f.texwhite && f.texture >= 0) || f.flags & F_NO_COLOR != 0 {
        [255; 3]
    } else if f.flags & F_PACKED != 0 {
        f.packed
    } else if f.color == u32::MAX {
        [255; 3]
    } else {
        palette(db, f.color)
    }
}

/// The mesh of a node's faces, read with `pal`'s palettes, built once.
fn mesh_of(c: &mut Ctx<'_>, file: u32, node: u32, pal: Pal) -> Result<MeshOf> {
    if c.files[file as usize].db.nodes[node as usize].faces.is_empty() {
        return Ok(None);
    }
    let (m, triangles) = if let Some(&m) = c.mesh_of.get(&(file, node, pal)) {
        m
    } else {
        let before = c.built_triangles;
        let m = build_mesh(c, file, node, pal)?;
        let built = (m, c.built_triangles - before);
        c.mesh_of.insert((file, node, pal), built);
        built
    };
    if m.is_some() {
        c.triangles = c.triangles.saturating_add(triangles);
    }
    Ok(m)
}

/// The glTF material for a key; its index.
fn material_of(c: &mut Ctx<'_>, key: MatKey, image: Option<usize>) -> usize {
    if let Some(&m) = c.material_at.get(&key) {
        return m;
    }
    let palette = key
        .material
        .and_then(|(f, i)| c.files[f as usize].db.materials.get(&i32::from(i)))
        .cloned();
    let src = palette.clone().unwrap_or_default();
    // A face with no material is drawn in its own color (white when it gives none), which the vertex colors
    // hold: the default material, white under them. A palette material is OpenGL's, its shininess (0 to 128)
    // the Blinn-Phong exponent.
    let look = match &palette {
        Some(m) => {
            let mut phong = Phong::new(Rgb::Srgb(m.diffuse));
            phong.opacity = m.alpha;
            phong.emissive = Rgb::Srgb(m.emissive);
            phong.specular = Some(Rgb::Srgb(m.specular));
            phong.exponent = Some(m.shininess.clamp(0.0, 128.0));
            phong.pbr().full("")
        }
        None => legacy_material::default_material(true).into(),
    };
    let alpha = look.base_color[3];
    let (base, emissive, specular) = if key.unlit {
        ([1.0; 3], [0.0; 3], None)
    } else {
        (
            [look.base_color[0], look.base_color[1], look.base_color[2]],
            look.emissive,
            look.specular,
        )
    };
    let roughness = look.roughness;
    let tex_name = |c: &Ctx<'_>, f: u32, i: i16| c.files[f as usize].db.textures.get(&i32::from(i)).cloned();
    let texture = key.texture.and_then(|(f, i)| tex_name(c, f, i));
    let name = if !src.name.is_empty() {
        src.name.clone()
    } else if let Some(t) = &texture {
        t.rsplit(['/', '\\']).next().unwrap_or(t).to_owned()
    } else if let Some((_, i)) = key.material {
        format!("material{i}")
    } else {
        legacy_material::DEFAULT_NAME.to_owned()
    };
    // Layer coordinates follow the base set, in layer order.
    let texcoord = |l: usize| 1 + (key.layer_mask & ((1u8 << l) - 1)).count_ones();
    let mut normal_tex = None;
    let mut layers = Vec::new();
    for (l, &(t, effect)) in key.layers.iter().enumerate() {
        if t < 0 {
            continue;
        }
        let tfile = key.texture.map_or(0, |(f, _)| f);
        let uv = (key.layer_mask & (1 << l) != 0).then(|| texcoord(l));
        if effect == BUMP
            && normal_tex.is_none()
            && let Some(uv) = uv
            && let Some((Some(img), _)) = texture_image(c, tfile, t)
        {
            let mut tex = Tex::new(img);
            tex.uv = uv;
            normal_tex = Some(tex);
            continue;
        }
        layers.push(json!({ "layer": l + 1, "texture": tex_name(c, tfile, t), "effect": effect, "texCoord": uv }));
    }
    let mut full = Full {
        name,
        base_color: [base[0], base[1], base[2], alpha],
        base_color_tex: image.map(Tex::new),
        metallic: 0.0,
        roughness,
        normal_tex,
        emissive,
        specular,
        alpha: if key.blend { Alpha::Blend } else { Alpha::Opaque },
        double_sided: key.double_sided,
        unlit: key.unlit,
        ..Full::default()
    };
    if c.keep.extras {
        let mut e = json!({});
        if let Some((_, i)) = key.material {
            e["materialIndex"] = json!(i);
            e["flags"] = json!(src.flags);
            e["ambient"] = json!(src.ambient);
            e["diffuse"] = json!(src.diffuse);
            e["specular"] = json!(src.specular);
            e["emissive"] = json!(src.emissive);
            e["shininess"] = json!(src.shininess);
            e["alpha"] = json!(src.alpha);
        }
        if let Some(t) = texture {
            e["texture"] = json!(t);
        }
        if !layers.is_empty() {
            e["layers"] = json!(layers);
        }
        full.extras = Some(json!({ "openflight": e }));
    }
    c.materials.push(full);
    c.material_at.insert(key, c.materials.len() - 1);
    c.materials.len() - 1
}

/// The image of a texture palette entry (`None` when textures aren't kept), and whether it has
/// transparent pixels (known for SGI images); `None` when it's missing. A face whose texture is
/// found is textured whether or not textures are kept, so its primitive is the one a full read
/// makes.
fn texture_image(c: &mut Ctx<'_>, file: u32, index: i16) -> Option<Found> {
    if let Some(&i) = c.image_of.get(&(file, index)) {
        return i;
    }
    let f = &c.files[file as usize];
    let found = f.db.textures.get(&i32::from(index)).cloned().and_then(|name: String| {
        let Some(path) = c.assets.find(&name, &f.dir) else {
            c.stats.dropped_textures += 1;
            return None;
        };
        if !(c.keep.materials && c.keep.textures) {
            return Some((None, false));
        }
        let got = if sgi::is_sgi_name(&name) {
            // SGI images are decoded now, as PNG.
            c.assets.read(&path).and_then(|b| {
                let (w, h, channels, px) = sgi::decode(&b)?;
                let alpha = channels % 2 == 0 && px.chunks_exact(channels).any(|p| p[channels - 1] < 255);
                let png = sgi::png(w, h, channels, &px)?;
                let i = c
                    .images
                    .bytes(Cow::Owned(png), &name, Location::File(path), Image::png_or_jpeg)?;
                Some((Some(i), alpha))
            })
        } else {
            c.images.file(c.assets, path, &name).map(|i| (Some(i), false))
        };
        if got.is_none() {
            c.stats.dropped_textures += 1;
        }
        got
    });
    c.image_of.insert((file, index), found);
    found
}

/// Newell's normal of a polygon, unit length; `None` when it has no area.
fn face_normal(db: &Db, corners: &[u32]) -> Option<[f64; 3]> {
    unit(vec3::newell(corners.iter().map(|&a| db.verts[a as usize].pos)))
}

/// A node's faces as a mesh, a primitive per material.
#[allow(clippy::too_many_lines, reason = "one pass over the faces")]
fn build_mesh(c: &mut Ctx<'_>, file: u32, node: u32, pal: Pal) -> Result<MeshOf> {
    let fi = file as usize;
    let faces = c.files[fi].db.nodes[node as usize].faces.clone();
    let object_alpha = match c.files[fi].db.nodes[node as usize].kind {
        Kind::Object { transparency, .. } => 1.0 - f32::from(transparency) / 65535.0,
        _ => 1.0,
    };
    // Where the mesh is: far from the origin, it's moved near it.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    {
        let db = &c.files[fi].db;
        for &f in &faces {
            let f = &db.faces[f as usize];
            for &v in &db.corners[f.start as usize..(f.start + f.len) as usize] {
                let p = db.verts[v as usize].pos;
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
    }
    let center = [0, 1, 2].map(|k| f64::midpoint(lo[k], hi[k]));
    let offset = if lo[0] <= hi[0] && center.iter().any(|v| v.abs() > FAR) {
        center.map(f64::round)
    } else {
        [0.0; 3]
    };
    let mut parts: Vec<(MatKey, Option<usize>, Part2)> = Vec::new();
    let mut part_of: FastMap<MatKey, usize> = FastMap::default();
    let mut morphs = false;
    let (mut poly, mut tris, mut local): (Vec<[f32; 3]>, Vec<[u32; 3]>, Vec<u32>) =
        (Vec::new(), Vec::new(), Vec::new());
    for &face in &faces {
        let f = c.files[fi].db.faces[face as usize].clone();
        if f.flags & F_HIDDEN != 0 {
            c.stats.hidden_faces += 1;
            continue;
        }
        if f.len == 0 {
            continue;
        }
        c.stats.faces += 1;
        // Its material.
        let image = (f.texture >= 0)
            .then(|| texture_image(c, pal.textures, f.texture))
            .flatten();
        let points = f.shape == Shape::Points || matches!(f.draw, 8..=10);
        let lines = !points && matches!(f.draw, 2 | 3);
        let textured = image.is_some();
        let mat = (f.material >= 0)
            .then_some((pal.materials, f.material))
            .filter(|&(mf, i)| c.files[mf as usize].db.materials.contains_key(&i32::from(i)));
        let mat_alpha = mat.map_or(1.0, |(mf, i)| c.files[mf as usize].db.materials[&i32::from(i)].alpha);
        let alpha = (1.0 - f32::from(f.transparency) / 65535.0) * object_alpha;
        let key = MatKey {
            material: mat,
            texture: image.map(|_| (pal.textures, f.texture)),
            layers: if textured || f.layer_mask != 0 {
                f.layer_tex
            } else {
                [(-1, 0); 7]
            },
            layer_mask: if f.layers == u32::MAX { 0 } else { f.layer_mask },
            double_sided: f.draw == 1 && !points && !lines,
            // The face's own transparency is in its vertices' colors; template 1 is fixed alpha blending,
            // and 2 and 4 billboards with it.
            blend: Transparency {
                opacity: mat_alpha,
                vertex_alpha: alpha < 1.0,
                texture_alpha: f.template == 1
                    || (textured && matches!(f.template, 2 | 4))
                    || image.is_some_and(|(_, a)| a),
                cutout: None,
            }
            .mode()
                == Alpha::Blend,
            unlit: f.light_mode <= 1 || points || lines,
        };
        let pi = *part_of.entry(key).or_insert_with(|| {
            parts.push((key, image.and_then(|(i, _)| i), Part2::default()));
            parts.len() - 1
        });
        let db = &c.files[fi].db;
        let colors = &c.files[pal.colors as usize].db;
        let corners = &db.corners[f.start as usize..(f.start + f.len) as usize];
        let fcolor = face_color(colors, &f).map(|v| c.lut[usize::from(v)]);
        let vertex_colors = matches!(f.light_mode, 1 | 3);
        // A face with no normal of its own, or a vertex whose normal has no length, is left zero, and takes its
        // faces' normal as the primitive is written.
        let normal = face_normal(db, corners).unwrap_or([0.0; 3]).map(f32_of);
        let part = &mut parts[pi].2;
        part.uvs |= textured;
        let per_corner = f.layers != u32::MAX || f.morph != u32::MAX;
        morphs |= f.morph != u32::MAX && c.keep.morph_targets;
        // Each corner's vertex in the primitive.
        tris.clear();
        poly.clear();
        local.clear();
        for (k, &vi) in corners.iter().enumerate() {
            let v = &db.verts[vi as usize];
            let own_color = vertex_colors && v.has & (V_PACKED | V_INDEX) != 0;
            let by_face = v.has & V_NORMAL == 0 || !own_color;
            let key = (
                vi,
                if by_face { face } else { u32::MAX },
                if per_corner { u32_of(k) } else { u32::MAX },
            );
            let at = *part.at.entry(key).or_insert_with(|| {
                let p = &mut part.prim;
                let pos = [0, 1, 2].map(|i| f32_of(v.pos[i] - offset[i]));
                p.positions.push(pos);
                let n = if v.has & V_NORMAL != 0 {
                    let n = v.normal;
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if l > 1e-6 && l.is_finite() {
                        n.map(|x| x / l)
                    } else {
                        [0.0; 3]
                    }
                } else {
                    normal
                };
                p.normals.push(n);
                let uv = if v.has & V_UV != 0 {
                    part.uvs = true;
                    [v.uv[0], 1.0 - v.uv[1]]
                } else {
                    [0.0, 0.0]
                };
                p.uvs.push(uv);
                let rgb = if own_color {
                    let b = if v.has & V_PACKED != 0 {
                        v.packed
                    } else {
                        palette(colors, v.index)
                    };
                    b.map(|x| c.lut[usize::from(x)])
                } else {
                    fcolor
                };
                let color = [rgb[0], rgb[1], rgb[2], alpha];
                part.colored |= color.iter().any(|&v| v < 1.0);
                p.colors.push(color);
                for (l, set) in part.layers.iter_mut().enumerate() {
                    if key_layer(f.layers, f.layer_mask, l) {
                        let uv = db
                            .layer_uvs
                            .get(f.layers as usize + k * 7 + l)
                            .copied()
                            .unwrap_or_default();
                        set.push([uv[0], 1.0 - uv[1]]);
                    }
                }
                let delta = if f.morph == u32::MAX {
                    [0.0; 3]
                } else {
                    let m = db.morph.get(f.morph as usize + k).map_or(vi, |&m| m);
                    let q = db.verts[m as usize].pos;
                    [0, 1, 2].map(|i| f32_of(q[i] - v.pos[i]))
                };
                part.morph.push(delta);
                u32_of(p.positions.len() - 1)
            });
            local.push(at);
        }
        let p = &mut part.prim;
        let n = local.len();
        if points || f.shape == Shape::Points || n == 1 {
            p.points.extend_from_slice(&local);
        } else if lines || n == 2 {
            for k in 0..n - 1 {
                p.lines.extend_from_slice(&[local[k], local[k + 1]]);
            }
            // A closed wireframe, and a two-sided polygon drawn as a line, close the loop.
            if n > 2 && f.draw != 3 {
                p.lines.extend_from_slice(&[local[n - 1], local[0]]);
            }
        } else if f.shape == Shape::Triangles {
            for t in local.chunks_exact(3) {
                p.indices.extend_from_slice(t);
            }
            c.built_triangles += n / 3;
        } else {
            let cost = if n > 8 { (n as u64).pow(3) } else { 0 };
            if cost <= c.clip {
                c.clip -= cost;
                poly.extend(local.iter().map(|&i| p.positions[i as usize]));
                crate::mesh::triangulate(&poly, &mut tris);
                for t in &tris {
                    p.indices.extend(t.iter().map(|&k| local[k as usize]));
                }
                c.built_triangles += tris.len();
            } else {
                for k in 1..n - 1 {
                    p.indices.extend_from_slice(&[local[0], local[k], local[k + 1]]);
                }
                c.built_triangles += n - 2;
            }
        }
        ensure!(
            c.built_triangles <= c.limit,
            LimitExceeded,
            "more than {} triangles",
            c.limit
        );
    }
    let mut primitives = Vec::new();
    for (key, image, mut part) in parts {
        let material = material_of(c, key, image);
        let n = part.prim.positions.len();
        if !part.uvs && key.layer_mask == 0 {
            part.prim.uvs = Vec::new();
        }
        if !part.colored {
            part.prim.colors = Vec::new();
        }
        if key.layer_mask != 0 {
            part.prim.uv_sets.push(Vec::new());
            for l in 0..7 {
                if key.layer_mask & (1 << l) != 0 {
                    let mut set = std::mem::take(&mut part.layers[l]);
                    set.resize(n, [0.0; 2]);
                    part.prim.uv_sets.push(set);
                }
            }
        }
        c.lines_or_points |= !part.prim.lines.is_empty() || !part.prim.points.is_empty();
        let target = morphs.then(|| c.b.positions(std::mem::take(&mut part.morph)));
        for mut j in c.b.primitives(part.prim) {
            j["material"] = json!(material);
            if let Some(t) = target {
                j["targets"] = json!([{ "POSITION": t }]);
            }
            primitives.push(j);
        }
    }
    if primitives.is_empty() {
        return Ok(None);
    }
    let name = &c.files[fi].db.nodes[node as usize].name;
    let mut mesh = json!({ "primitives": primitives });
    if !name.is_empty() {
        mesh["name"] = json!(name);
    }
    if morphs {
        mesh["weights"] = json!([0.0]);
        mesh["extras"] = json!({ "targetNames": ["morph"] });
    }
    c.meshes.push(mesh);
    Ok(Some((c.meshes.len() - 1, offset)))
}

/// Whether a face with layer coordinates at `layers` has layer `l`'s.
fn key_layer(layers: u32, mask: u8, l: usize) -> bool {
    layers != u32::MAX && mask & (1 << l) != 0
}

/// Animated groups as an animation: each frame node scaled to full size in its turn and to
/// nothing otherwise, stepping.
fn animation(sequences: &[(parse::Sequence, Vec<usize>)]) -> (Vec<Track>, bool) {
    let mut tracks = Vec::new();
    let mut assumed = false;
    for (seq, frames) in sequences {
        let n = frames.len();
        if !(2..=MAX_FRAMES).contains(&n) {
            continue;
        }
        let mut order: Vec<usize> = (0..n).collect();
        if seq.swing {
            order.extend((1..n - 1).rev());
        }
        if seq.backward {
            order.reverse();
        }
        let steps = order.len();
        let count = f32_of(f64::from(u32_of(n)));
        let frame = if seq.loop_duration > 0.0 {
            seq.loop_duration / count
        } else {
            // No duration given: the readers' shared assumed frame rate.
            assumed = true;
            f32_of(1.0 / keys::ASSUMED_FRAME_RATE)
        };
        let mut times: Vec<f32> = (0..u32_of(steps)).map(|s| f32_of(f64::from(s)) * frame).collect();
        let last = if seq.last_frame > 0.0 { seq.last_frame } else { frame };
        times.push(times[steps - 1] + last);
        if times.windows(2).any(|w| w[1] <= w[0] || !w[1].is_finite()) {
            continue;
        }
        for (k, &node) in frames.iter().enumerate() {
            let mut v = Vec::with_capacity((steps + 1) * 3);
            for &shown in order.iter().chain(std::iter::once(&order[0])) {
                let s = if shown == k { 1.0 } else { 0.0 };
                v.extend_from_slice(&[s, s, s]);
            }
            tracks.push(Track {
                interp: Interp::Step,
                ..Track::linear(node, keys::Path::Scale, times.clone(), v)
            });
        }
    }
    (tracks, assumed)
}

/// A date as the header writes it, C's `ctime` form (`"Tue Mar 05 14:30:00 2024"`), in ISO 8601.
fn ctime_date(s: &str) -> Option<String> {
    const MONTHS: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let w: Vec<&str> = s.split_whitespace().collect();
    let [_, month, day, time, year] = w.as_slice() else {
        return None;
    };
    let month = MONTHS.iter().position(|m| month.eq_ignore_ascii_case(m))? + 1;
    let day: u32 = day.parse().ok().filter(|d| (1..=31).contains(d))?;
    let year: u32 = year.parse().ok().filter(|y| (1000..=9999).contains(y))?;
    let hms: Vec<u32> = time.split(':').map(str::parse).collect::<Result<_, _>>().ok()?;
    let [h, m, sec] = hms.as_slice() else {
        return None;
    };
    (*h < 24 && *m < 60 && *sec < 61).then(|| format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{sec:02}"))
}

/// Where the header places the database on the earth: its origin's latitude and longitude, its
/// corners', the projection and the ellipsoid. `None` when they're all zero, as an unplaced
/// database leaves them.
fn georeference(header: &Map<String, Value>) -> Option<Value> {
    let pair = |k: &str| {
        let v = header.get(k)?.as_array()?;
        Some([v.first()?.as_f64()?, v.get(1)?.as_f64()?])
    };
    let origin = pair("originLatLong");
    let sw = pair("southwestLatLong");
    let ne = pair("northeastLatLong");
    if [origin, sw, ne].iter().flatten().flatten().all(|&x| x == 0.0) {
        return None;
    }
    let mut g = Map::new();
    if let Some([lat, lon]) = origin {
        g.insert("latitude".into(), json!(lat));
        g.insert("longitude".into(), json!(lon));
    }
    if let (Some(sw), Some(ne)) = (sw, ne) {
        g.insert("southwest".into(), json!(sw));
        g.insert("northeast".into(), json!(ne));
    }
    let code = |k: &str| header.get(k).and_then(Value::as_i64);
    if let Some(p) = code("projection") {
        let name = match p {
            0 => "flat earth",
            1 => "trapezoidal",
            2 => "round earth",
            3 => "Lambert",
            4 => "UTM",
            5 => "geodetic",
            6 => "geocentric",
            _ => "unknown",
        };
        g.insert("projection".into(), json!(name));
    }
    if let Some(e) = code("ellipsoid") {
        let name = match e {
            0 => "WGS 84",
            1 => "WGS 72",
            2 => "Bessel",
            3 => "Clarke 1866",
            4 => "NAD 27",
            -1 => "user defined",
            _ => "unknown",
        };
        g.insert("ellipsoid".into(), json!(name));
    }
    if code("projection") == Some(4)
        && let Some(z) = code("utmZone")
    {
        g.insert("utmZone".into(), json!(z));
    }
    Some(Value::Object(g))
}

fn finish(mut c: Ctx<'_>, mut s: Scene) -> Document {
    for (k, kids) in s.kids.iter().enumerate() {
        if !kids.is_empty() {
            s.nodes[k]["children"] = json!(kids);
        }
    }
    let root = &c.files[0].db;
    let mut source = Map::new();
    if !root.id.is_empty() {
        source.insert("id".into(), json!(root.id));
    }
    if c.keep.extras {
        for (k, v) in &root.header {
            if k != "date" {
                source.insert(k.clone(), v.clone());
            }
        }
    }
    if c.files.len() > 1 {
        source.insert("files".into(), json!(c.files.len()));
    }
    let meters = root.meters();
    let unit = root.unit_name();
    SourceInfo {
        format: Some("OpenFlight".into()),
        version: Some(root.version_name()),
        // A unit code the format doesn't define is read as meters.
        meters_per_unit: if unit.is_some() {
            Fact::Declared(meters)
        } else {
            Fact::Assumed(meters)
        },
        unit: unit.map(str::to_owned),
        up_axis: Fact::Declared(Axis::PosZ),
        handedness: Fact::Declared(Handedness::Right),
        converted: Converted {
            units: (meters - 1.0).abs() > 0.0,
            up_axis: true,
            handedness: false,
        },
        created: root
            .header
            .get("date")
            .and_then(Value::as_str)
            .map(|d| ctime_date(d).unwrap_or_else(|| d.to_owned())),
        georeference: georeference(&root.header),
        ..SourceInfo::default()
    }
    .write(&mut source);
    let counts = c.files.iter().fold(parse::Counts::default(), |mut a, f| {
        let n = &f.db.counts;
        a.unknown_records += n.unknown_records;
        a.dropped += n.dropped;
        a.extensions += n.extensions;
        a.subfaces += n.subfaces;
        a.billboards += n.billboards;
        a.morph_faces += n.morph_faces;
        a
    });
    let st = &c.stats;
    for (key, n) in [
        ("hiddenFaces", st.hidden_faces),
        ("droppedTextures", st.dropped_textures),
        ("missingInstances", st.missing_instances),
        ("cycles", st.cycles),
        ("tooDeep", st.too_deep),
        ("droppedNodes", st.dropped_nodes),
        ("droppedLevelsOfDetail", st.dropped_lods),
        ("switchedOff", st.switched_off),
        ("disabledLights", st.disabled_lights),
        ("unknownRecords", counts.unknown_records),
        (crate::geometry::DROPPED, counts.dropped),
        ("extensions", counts.extensions),
        ("subfaces", counts.subfaces),
        ("billboards", counts.billboards),
    ] {
        if n > 0 {
            source.insert(key.into(), json!(n));
        }
    }
    if !st.missing_files.is_empty() {
        source.insert("missingFiles".into(), json!(st.missing_files));
    }
    if c.files.iter().any(|f| f.db.truncated) {
        source.insert("truncated".into(), json!(true));
    }
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(c.triangles));
    facts.insert("faces".into(), json!(st.faces));
    facts.insert("nodes".into(), json!(s.nodes.len()));
    facts.insert("materials".into(), json!(c.materials.len()));

    if c.keep.animations {
        let (tracks, assumed) = animation(&s.sequences);
        let mut writer = keys::Writer::new(c.keep.extras);
        let written = writer.animation(&mut c.b, None, None, tracks).is_some();
        writer.finish(&mut c.b);
        if written && assumed {
            // A sequence that gives no duration plays at the readers' shared assumed rate.
            let mut info = SourceInfo::from_json(&source);
            info.frame_rate = Fact::Assumed(keys::ASSUMED_FRAME_RATE);
            info.write(&mut source);
        }
    }

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [0] }]));
    json.insert("nodes".into(), Value::Array(s.nodes));
    if !c.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut c.meshes)));
    }
    if !c.lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut c.lights) } }),
        );
    }
    let images = std::mem::replace(&mut c.images, Images::new(true, Path::new(""))).list;
    c.b.materials(&c.materials, images, &mut json);
    let mut doc = c.b.finish(json, facts);
    if c.lines_or_points {
        doc.set_lines_and_points(true);
    }
    // Z up to Y up, and the file's units to meters.
    frame::root(&mut doc, Frame::z_up().scaled(meters));
    doc
}
