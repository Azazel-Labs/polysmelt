//! Every loader the evidence binaries run: Polysmelt, a competitor crate for each format one
//! exists for, and (with the `assimp` feature) assimp, each reading a file into its own in-memory
//! result.
//!
//! A loader reads either a file on disk (`Src::File`, through the library's own file API where it
//! has one: the speed benchmarks) or bytes already in memory (`Src::Bytes`, with the files a model
//! refers to found beside `path`: the resilience run). It returns its result boxed, so the result's
//! heap stays live until measured; how much it read (triangles for meshes, voxels for VOX, joint
//! channels times frames for BVH); and a function that counts what the result holds, called after
//! the clock stops, for the count-parity tables.

#![allow(dead_code)]

use std::any::Any;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::Path;

use serde_json::{Map, Value};

pub enum Src<'a> {
    File(&'a Path),
    Bytes(&'a [u8], &'a Path),
}

impl Src<'_> {
    pub fn path(&self) -> &Path {
        match self {
            Src::File(p) | Src::Bytes(_, p) => p,
        }
    }
    pub fn dir(&self) -> &Path {
        self.path().parent().unwrap_or(Path::new("."))
    }
    pub fn bytes(&self) -> Result<Cow<'_, [u8]>, String> {
        match self {
            Src::File(p) => std::fs::read(p).map(Cow::Owned).map_err(|e| e.to_string()),
            Src::Bytes(b, _) => Ok(Cow::Borrowed(b)),
        }
    }
    pub fn ext(&self) -> String {
        ext_of(self.path())
    }
}

pub fn ext_of(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// What a result holds, by name: `meshes`, `vertices`, `triangles`, `materials`, `nodes`,
/// `joints`, `animations` and so on. A loader sets only what its result tells.
pub type Stats = BTreeMap<&'static str, u64>;

pub struct Loaded {
    pub result: Box<dyn Any>,
    pub units: usize,
    pub stats: fn(&dyn Any) -> Stats,
}

pub type Out = Result<Loaded, String>;
pub type Run = fn(&Src) -> Out;

#[derive(Clone, Copy)]
pub struct Loader {
    pub name: &'static str,
    pub run: Run,
}

fn e(x: impl std::fmt::Display) -> String {
    x.to_string()
}

fn loaded<T: Any>(result: T, units: usize, stats: fn(&dyn Any) -> Stats) -> Out {
    Ok(Loaded { result: Box::new(result), units, stats })
}

fn no_stats(_: &dyn Any) -> Stats {
    Stats::new()
}

fn n(x: usize) -> u64 {
    x as u64
}

// --- Polysmelt ------------------------------------------------------------------------------

pub const POLYSMELT: &str = "polysmelt";
pub const POLYSMELT_GLB: &str = "polysmelt → GLB";

pub fn polysmelt_reads(ext: &str) -> bool {
    polysmelt::formats::EXTENSIONS.contains(&ext)
}

fn polysmelt_doc(src: &Src, keep: polysmelt::document::Keep) -> Result<polysmelt::document::Document<'static>, String> {
    let ext = src.ext();
    let folder = polysmelt::assets::Folder::new(src.dir());
    match src {
        Src::File(p) => polysmelt::formats::load(p, &ext, &folder, keep).map_err(e),
        Src::Bytes(b, p) => {
            // The model's path, so that what it references resolves from its folder.
            let name = p.to_string_lossy().into_owned();
            polysmelt::formats::load_bytes(b, &name, &ext, &folder, keep).map(|d| d.into_owned()).map_err(e)
        }
    }
}

fn list<'a>(json: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    json.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn idx(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|i| i as usize)
}

fn tri_count(mode: u64, n: usize) -> usize {
    match mode {
        4 => n / 3,
        5 | 6 => n.saturating_sub(2),
        _ => 0,
    }
}

/// What a glTF JSON declares (every glTF-shaped Polysmelt document): each mesh counted once,
/// whatever number of nodes draw it, as the other libraries count.
pub fn json_stats(json: &Map<String, Value>) -> Stats {
    let accessors = list(json, "accessors");
    let count = |a: Option<usize>| a.and_then(|a| accessors.get(a)).and_then(|a| idx(a, "count")).unwrap_or(0);
    let (mut prims, mut verts, mut tris, mut lines, mut points) = (0, 0, 0, 0, 0);
    for m in list(json, "meshes") {
        for p in m.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            prims += 1;
            let v = count(p.get("attributes").and_then(|a| idx(a, "POSITION")));
            verts += v;
            let k = idx(p, "indices").map_or(v, |i| count(Some(i)));
            let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
            tris += tri_count(mode, k);
            match mode {
                0 => points += k,
                1 => lines += k / 2,
                2 => lines += k,
                3 => lines += k.saturating_sub(1),
                _ => {}
            }
        }
    }
    let joints: BTreeSet<u64> = list(json, "skins")
        .iter()
        .flat_map(|s| s.get("joints").and_then(Value::as_array).into_iter().flatten())
        .filter_map(Value::as_u64)
        .collect();
    let channels: usize = list(json, "animations")
        .iter()
        .map(|a| a.get("channels").and_then(Value::as_array).map_or(0, Vec::len))
        .sum();
    let lights = json
        .get("extensions")
        .and_then(|e| e.get("KHR_lights_punctual"))
        .and_then(|l| l.get("lights"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    Stats::from([
        ("meshes", n(list(json, "meshes").len())),
        ("primitives", n(prims)),
        ("vertices", n(verts)),
        ("triangles", n(tris)),
        ("lines", n(lines)),
        ("points", n(points)),
        ("materials", n(list(json, "materials").len())),
        ("nodes", n(list(json, "nodes").len())),
        ("joints", n(joints.len())),
        ("skins", n(list(json, "skins").len())),
        ("animations", n(list(json, "animations").len())),
        ("channels", n(channels)),
        ("images", n(list(json, "images").len())),
        ("cameras", n(list(json, "cameras").len())),
        ("lights", n(lights)),
    ])
}

fn doc_stats(doc: &polysmelt::document::Document) -> Stats {
    let mut s = json_stats(&doc.json);
    let fact = |k: &str| doc.facts.get(k).and_then(Value::as_u64);
    for (k, name) in [("voxels", "voxels"), ("frames", "frames"), ("channels", "bvh_channels")] {
        if let Some(v) = fact(k) {
            s.insert(name, v);
        }
    }
    s.insert("scene_triangles", n(doc.triangles()));
    s
}

fn polysmelt_doc_stats(r: &dyn Any) -> Stats {
    if let Some(doc) = r.downcast_ref::<polysmelt::document::Document<'static>>() {
        return doc_stats(doc);
    }
    if let Some((doc, _)) = r.downcast_ref::<(polysmelt::document::Document<'static>, Option<polysmelt::mesh::Mesh>)>() {
        return doc_stats(doc);
    }
    if let Some((doc, _)) = r.downcast_ref::<(polysmelt::document::Document<'static>, Vec<u8>)>() {
        return doc_stats(doc);
    }
    Stats::new()
}

/// Voxels for VOX, channels × frames for BVH: what a model without triangles holds.
fn units_of(doc: &polysmelt::document::Document) -> usize {
    let f = |k: &str| doc.facts.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    f("voxels").max(f("channels") * f("frames"))
}

/// Load with `keep`. A glTF document leaves its arrays in the file, so it's flattened: the
/// triangles drawn, decoded, as the `gltf` crate's readers give them. Other formats' documents
/// already hold their arrays, as their competitors' results do.
fn polysmelt_keep(src: &Src, keep: polysmelt::document::Keep) -> Out {
    use polysmelt::document::Pose;
    let doc = polysmelt_doc(src, keep)?;
    if matches!(src.ext().as_str(), "gltf" | "glb" | "vrm") {
        let mesh = doc.flatten(Pose::Rest).map_err(e)?;
        let k = mesh.as_ref().map_or(0, polysmelt::mesh::Mesh::triangles);
        return loaded((doc, mesh), k, polysmelt_doc_stats);
    }
    let k = doc.triangles().max(units_of(&doc));
    loaded(doc, k, polysmelt_doc_stats)
}

pub fn polysmelt(src: &Src) -> Out {
    use polysmelt::document::Keep;
    polysmelt_keep(src, Keep { threads: 1, ..Keep::ALL })
}

pub fn polysmelt_all_cores(src: &Src) -> Out {
    use polysmelt::document::Keep;
    polysmelt_keep(src, Keep { threads: 0, ..Keep::ALL })
}

/// Without reading the bytes of the image files a model names: the work libraries that keep only
/// an image's path (assimp, `gltf` without image decoding) do.
pub fn polysmelt_no_image_bytes(src: &Src) -> Out {
    use polysmelt::document::Keep;
    polysmelt_keep(src, Keep { threads: 1, image_bytes: false, ..Keep::ALL })
}

/// A conversion: load, and write the document as a GLB in memory.
pub fn polysmelt_glb(src: &Src) -> Out {
    use polysmelt::document::Keep;
    let doc = polysmelt_doc(src, Keep { threads: 1, ..Keep::ALL })?;
    let glb = doc.to_glb().map_err(e)?;
    let k = glb.len();
    loaded((doc, glb), k, polysmelt_doc_stats)
}

/// Everything a conversion does: load, flatten at rest, and write both the document and the
/// flat mesh as GLBs. What the resilience run puts each mutated file through.
pub fn polysmelt_convert(src: &Src) -> Out {
    use polysmelt::document::{Keep, Pose};
    let doc = polysmelt_doc(src, Keep::ALL)?;
    let glb = doc.to_glb().map_err(e)?;
    let mesh = doc.flatten(Pose::Rest).map_err(e)?;
    let k = mesh.as_ref().map_or(0, polysmelt::mesh::Mesh::triangles);
    if let Some(m) = &mesh {
        polysmelt::mesh::to_glb(m).map_err(e)?;
    }
    if doc.json.get("animations").is_some() {
        doc.rig().map_err(e)?;
    }
    loaded((glb, mesh), k, no_stats)
}

/// Both GLBs a conversion writes, the document's and the flat mesh's (`None` when the model
/// draws nothing), or why it failed.
pub fn polysmelt_glbs(path: &Path) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
    use polysmelt::document::{Keep, Pose};
    let doc = polysmelt_doc(&Src::File(path), Keep::ALL)?;
    let glb = doc.to_glb().map_err(e)?;
    let mesh = match doc.flatten(Pose::Rest).map_err(e)? {
        Some(m) => Some(polysmelt::mesh::to_glb(&m).map_err(e)?),
        None => None,
    };
    Ok((glb, mesh))
}

// --- gltf -----------------------------------------------------------------------------------

/// Every primitive's attributes and indices read into vectors, the arrays a renderer uploads.
fn gltf_read(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> (Box<dyn Any>, usize) {
    #[allow(dead_code)]
    #[derive(Default)]
    struct Prim {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        tangents: Vec<[f32; 4]>,
        uvs: Vec<Vec<[f32; 2]>>,
        colors: Vec<[f32; 4]>,
        indices: Vec<u32>,
        joints: Vec<[u16; 4]>,
        weights: Vec<[f32; 4]>,
        targets: usize,
    }
    let get = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);
    let mut prims = Vec::new();
    let mut tris = 0;
    for mesh in doc.meshes() {
        for p in mesh.primitives() {
            let r = p.reader(get);
            let mut out = Prim {
                positions: r.read_positions().map(Iterator::collect).unwrap_or_default(),
                normals: r.read_normals().map(Iterator::collect).unwrap_or_default(),
                tangents: r.read_tangents().map(Iterator::collect).unwrap_or_default(),
                colors: r.read_colors(0).map(|c| c.into_rgba_f32().collect()).unwrap_or_default(),
                indices: r.read_indices().map(|i| i.into_u32().collect()).unwrap_or_default(),
                joints: r.read_joints(0).map(|j| j.into_u16().collect()).unwrap_or_default(),
                weights: r.read_weights(0).map(|w| w.into_f32().collect()).unwrap_or_default(),
                ..Prim::default()
            };
            let mut k = 0;
            while let Some(t) = r.read_tex_coords(k) {
                out.uvs.push(t.into_f32().collect());
                k += 1;
            }
            for (pos, nor, tan) in r.read_morph_targets() {
                out.targets += pos.map_or(0, Iterator::count) + nor.map_or(0, Iterator::count) + tan.map_or(0, Iterator::count);
            }
            let k = if p.indices().is_some() { out.indices.len() } else { out.positions.len() };
            tris += tri_count(u64::from(p.mode().as_gl_enum()), k);
            prims.push(out);
        }
    }
    let ibms: Vec<Vec<[[f32; 4]; 4]>> = doc
        .skins()
        .map(|s| s.reader(get).read_inverse_bind_matrices().map(Iterator::collect).unwrap_or_default())
        .collect();
    let keys: Vec<(Vec<f32>, usize)> = doc
        .animations()
        .flat_map(|a| a.channels().collect::<Vec<_>>())
        .map(|c| {
            let r = c.reader(get);
            let times = r.read_inputs().map(Iterator::collect).unwrap_or_default();
            let outs = r.read_outputs().map_or(0, |o| match o {
                gltf::animation::util::ReadOutputs::Translations(t) => t.count(),
                gltf::animation::util::ReadOutputs::Rotations(t) => t.into_f32().count(),
                gltf::animation::util::ReadOutputs::Scales(t) => t.count(),
                gltf::animation::util::ReadOutputs::MorphTargetWeights(t) => t.into_f32().count(),
            });
            (times, outs)
        })
        .collect();
    (Box::new((prims, ibms, keys)), tris)
}

/// Counts from the `gltf` crate's document, read through its own API (the counts it found, not
/// what the JSON says).
fn gltf_doc_stats(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Stats {
    let get = |b: gltf::Buffer| buffers.get(b.index()).map(|d| &d.0[..]);
    let (mut prims, mut verts, mut tris) = (0, 0, 0);
    for mesh in doc.meshes() {
        for p in mesh.primitives() {
            prims += 1;
            let r = p.reader(get);
            let v = r.read_positions().map_or(0, Iterator::count);
            verts += v;
            let k = if p.indices().is_some() { r.read_indices().map_or(0, |i| i.into_u32().count()) } else { v };
            tris += tri_count(u64::from(p.mode().as_gl_enum()), k);
        }
    }
    let joints: BTreeSet<usize> = doc.skins().flat_map(|s| s.joints().map(|j| j.index()).collect::<Vec<_>>()).collect();
    Stats::from([
        ("meshes", n(doc.meshes().count())),
        ("primitives", n(prims)),
        ("vertices", n(verts)),
        ("triangles", n(tris)),
        ("materials", n(doc.materials().count())),
        ("nodes", n(doc.nodes().count())),
        ("joints", n(joints.len())),
        ("skins", n(doc.skins().count())),
        ("animations", n(doc.animations().count())),
        ("channels", n(doc.animations().map(|a| a.channels().count()).sum())),
        ("images", n(doc.images().count())),
        ("cameras", n(doc.cameras().count())),
        ("lights", n(doc.lights().map_or(0, Iterator::count))),
    ])
}

type GltfImport = (gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>, Box<dyn Any>);
type GltfNoImages = (gltf::Document, Vec<gltf::buffer::Data>, Box<dyn Any>);

fn gltf_stats(r: &dyn Any) -> Stats {
    if let Some((doc, buffers, _, _)) = r.downcast_ref::<GltfImport>() {
        return gltf_doc_stats(doc, buffers);
    }
    if let Some((doc, buffers, _)) = r.downcast_ref::<GltfNoImages>() {
        return gltf_doc_stats(doc, buffers);
    }
    Stats::new()
}

/// `gltf::import`: the document, its buffers, and every image decoded to pixels.
pub fn gltf_import(src: &Src) -> Out {
    let (doc, buffers, images) = match src {
        Src::File(p) => gltf::import(p).map_err(e)?,
        Src::Bytes(b, p) => {
            let g = gltf::Gltf::from_slice(b).map_err(e)?;
            let buffers = gltf::import_buffers(&g.document, p.parent(), g.blob).map_err(e)?;
            let images = gltf::import_images(&g.document, p.parent(), &buffers).map_err(e)?;
            (g.document, buffers, images)
        }
    };
    let (arrays, k) = gltf_read(&doc, &buffers);
    let r: GltfImport = (doc, buffers, images, arrays);
    loaded(r, k, gltf_stats)
}

/// The document and its buffers, images left undecoded: the work Polysmelt's flattened mesh
/// does (it embeds image files as they are, without decoding their pixels).
pub fn gltf_no_images(src: &Src) -> Out {
    let bytes = src.bytes()?;
    let g = gltf::Gltf::from_slice(&bytes).map_err(e)?;
    let buffers = gltf::import_buffers(&g.document, src.path().parent(), g.blob).map_err(e)?;
    let (arrays, k) = gltf_read(&g.document, &buffers);
    let r: GltfNoImages = (g.document, buffers, arrays);
    loaded(r, k, gltf_stats)
}

// --- OBJ, STL, PLY --------------------------------------------------------------------------

type Tobj = (Vec<tobj::Model>, Option<Vec<tobj::Material>>);

fn tobj_stats(r: &dyn Any) -> Stats {
    let Some((models, mats)) = r.downcast_ref::<Tobj>() else { return Stats::new() };
    Stats::from([
        ("meshes", n(models.len())),
        ("vertices", n(models.iter().map(|m| m.mesh.positions.len() / 3).sum())),
        ("triangles", n(models.iter().map(|m| m.mesh.indices.len() / 3).sum())),
        ("materials", n(mats.as_ref().map_or(0, Vec::len))),
    ])
}

/// tobj with `GPU_LOAD_OPTIONS` (triangulated, one index per vertex) and its MTL libraries.
pub fn tobj(src: &Src) -> Out {
    let (models, mats) = match src {
        Src::File(p) => tobj::load_obj(p, &tobj::GPU_LOAD_OPTIONS).map_err(e)?,
        Src::Bytes(b, p) => {
            let dir = p.parent().unwrap_or(Path::new(".")).to_path_buf();
            tobj::load_obj_buf(&mut Cursor::new(b), &tobj::GPU_LOAD_OPTIONS, |m| tobj::load_mtl(dir.join(m)))
                .map_err(e)?
        }
    };
    let k = models.iter().map(|m| m.mesh.indices.len() / 3).sum();
    let r: Tobj = (models, mats.ok());
    loaded(r, k, tobj_stats)
}

fn stl_stats(r: &dyn Any) -> Stats {
    let Some(m) = r.downcast_ref::<stl_io::IndexedMesh>() else { return Stats::new() };
    Stats::from([("meshes", 1), ("vertices", n(m.vertices.len())), ("triangles", n(m.faces.len()))])
}

pub fn stl_io(src: &Src) -> Out {
    let mesh = match src {
        Src::File(p) => stl_io::read_stl(&mut BufReader::new(File::open(p).map_err(e)?)).map_err(e)?,
        Src::Bytes(b, _) => stl_io::read_stl(&mut Cursor::new(b)).map_err(e)?,
    };
    let k = mesh.faces.len();
    loaded(mesh, k, stl_stats)
}

type Ply = ply_rs::ply::Ply<ply_rs::ply::DefaultElement>;

fn ply_faces(ply: &Ply) -> usize {
    use ply_rs::ply::Property;
    ply.payload
        .get("face")
        .map(|faces| {
            faces
                .iter()
                .map(|f| match f.get("vertex_indices").or_else(|| f.get("vertex_index")) {
                    Some(Property::ListInt(v)) => v.len().saturating_sub(2),
                    Some(Property::ListUInt(v)) => v.len().saturating_sub(2),
                    Some(Property::ListShort(v)) => v.len().saturating_sub(2),
                    Some(Property::ListUShort(v)) => v.len().saturating_sub(2),
                    Some(Property::ListChar(v)) => v.len().saturating_sub(2),
                    Some(Property::ListUChar(v)) => v.len().saturating_sub(2),
                    _ => 0,
                })
                .sum()
        })
        .unwrap_or(0)
}

fn ply_stats(r: &dyn Any) -> Stats {
    let Some(ply) = r.downcast_ref::<Ply>() else { return Stats::new() };
    Stats::from([
        ("meshes", 1),
        ("vertices", n(ply.payload.get("vertex").map_or(0, Vec::len))),
        ("triangles", n(ply_faces(ply))),
    ])
}

/// ply-rs's `read_ply` with its generic elements: the only path that reads any PLY.
pub fn ply_rs(src: &Src) -> Out {
    use ply_rs::parser::Parser;
    use ply_rs::ply::DefaultElement;
    let p = Parser::<DefaultElement>::new();
    let ply = match src {
        Src::File(path) => p.read_ply(&mut BufReader::new(File::open(path).map_err(e)?)).map_err(e)?,
        Src::Bytes(b, _) => p.read_ply(&mut Cursor::new(b)).map_err(e)?,
    };
    let k = ply_faces(&ply);
    loaded(ply, k, ply_stats)
}

// --- FBX, 3MF, VOX, COLLADA, BVH ------------------------------------------------------------

#[cfg(feature = "evidence")]
type Fbx = (fbxcel_dom::v7400::Document, Vec<Vec<[f64; 3]>>);

#[cfg(feature = "evidence")]
fn fbx_stats(r: &dyn Any) -> Stats {
    let Some((doc, meshes)) = r.downcast_ref::<Fbx>() else { return Stats::new() };
    let (mut materials, mut models, mut stacks, mut limbs) = (0, 0, 0, 0);
    for o in doc.objects() {
        match o.class() {
            "Material" => materials += 1,
            "Model" => {
                models += 1;
                if o.subclass() == "LimbNode" {
                    limbs += 1;
                }
            }
            "AnimationStack" => stacks += 1,
            _ => {}
        }
    }
    Stats::from([
        ("meshes", n(meshes.len())),
        ("triangles", n(meshes.iter().map(|m| m.len() / 3).sum())),
        ("materials", n(materials)),
        ("nodes", n(models)),
        ("limb_nodes", n(limbs)),
        ("animations", n(stacks)),
    ])
}

/// fbxcel-dom: the DOM, then each mesh geometry's polygons fan-triangulated into positions. Less
/// than Polysmelt does: no normals, UVs, materials, skins, animation or transforms.
#[cfg(feature = "evidence")]
pub fn fbxcel_dom(src: &Src) -> Out {
    use fbxcel_dom::any::AnyDocument;
    use fbxcel_dom::v7400::data::mesh::{PolygonVertexIndex, PolygonVertices};
    use fbxcel_dom::v7400::object::TypedObjectHandle;
    use fbxcel_dom::v7400::object::geometry::TypedGeometryHandle;
    let doc = match src {
        Src::File(p) => AnyDocument::from_seekable_reader(BufReader::new(File::open(p).map_err(e)?)).map_err(e)?,
        Src::Bytes(b, _) => AnyDocument::from_seekable_reader(Cursor::new(b)).map_err(e)?,
    };
    let AnyDocument::V7400(_, doc) = doc else { return Err("unsupported FBX version".into()) };
    let fan = |_: &PolygonVertices<'_>, poly: &[PolygonVertexIndex], out: &mut Vec<[PolygonVertexIndex; 3]>| {
        for i in 1..poly.len().saturating_sub(1) {
            out.push([poly[0], poly[i], poly[i + 1]]);
        }
        Ok(())
    };
    let mut meshes: Vec<Vec<[f64; 3]>> = Vec::new();
    for obj in doc.objects() {
        if let TypedObjectHandle::Geometry(TypedGeometryHandle::Mesh(m)) = obj.get_typed() {
            let pv = m.polygon_vertices().map_err(e)?;
            let tris = pv.triangulate_each(fan).map_err(e)?;
            let mut pos = Vec::with_capacity(tris.len());
            for tvi in tris.triangle_vertex_indices() {
                let p = tris.control_point(tvi).ok_or("triangle vertex without a control point")?;
                pos.push([p.x, p.y, p.z]);
            }
            meshes.push(pos);
        }
    }
    let k = meshes.iter().map(|m| m.len() / 3).sum();
    let r: Fbx = (*doc, meshes);
    loaded(r, k, fbx_stats)
}

#[cfg(feature = "evidence")]
fn threemf_stats(r: &dyn Any) -> Stats {
    let Some(models) = r.downcast_ref::<Vec<threemf::model::Model>>() else { return Stats::new() };
    let meshes: Vec<&threemf::model::Mesh> =
        models.iter().flat_map(|m| &m.resources.object).filter_map(|o| o.mesh.as_ref()).collect();
    Stats::from([
        ("meshes", n(meshes.len())),
        ("vertices", n(meshes.iter().map(|m| m.vertices.vertex.len()).sum())),
        ("triangles", n(meshes.iter().map(|m| m.triangles.triangle.len()).sum())),
    ])
}

#[cfg(feature = "evidence")]
pub fn threemf(src: &Src) -> Out {
    let models = match src {
        Src::File(p) => threemf::read(BufReader::new(File::open(p).map_err(e)?)).map_err(e)?,
        Src::Bytes(b, _) => threemf::read(Cursor::new(b)).map_err(e)?,
    };
    let k = models
        .iter()
        .flat_map(|m| &m.resources.object)
        .filter_map(|o| o.mesh.as_ref())
        .map(|m| m.triangles.triangle.len())
        .sum();
    loaded(models, k, threemf_stats)
}

#[cfg(feature = "evidence")]
fn vox_stats(r: &dyn Any) -> Stats {
    let Some(d) = r.downcast_ref::<dot_vox::DotVoxData>() else { return Stats::new() };
    Stats::from([("meshes", n(d.models.len())), ("voxels", n(d.models.iter().map(|m| m.voxels.len()).sum()))])
}

#[cfg(feature = "evidence")]
pub fn dot_vox(src: &Src) -> Out {
    let bytes = src.bytes()?;
    let data = dot_vox::load_bytes(&bytes).map_err(e)?;
    let k = data.models.iter().map(|m| m.voxels.len()).sum();
    loaded(data, k, vox_stats)
}

#[cfg(feature = "evidence")]
type Collada = (
    Option<collada::ObjSet>,
    Option<Vec<collada::Animation>>,
    Option<Vec<collada::Skeleton>>,
    Option<collada::BindDataSet>,
);

#[cfg(feature = "evidence")]
fn collada_tris(objs: Option<&collada::ObjSet>) -> usize {
    objs.map(|s| {
        s.objects
            .iter()
            .flat_map(|o| &o.geometry)
            .flat_map(|g| &g.mesh)
            .map(|m| match m {
                collada::PrimitiveElement::Triangles(t) => t.vertices.len(),
                collada::PrimitiveElement::Polylist(p) => {
                    p.shapes.iter().filter(|s| matches!(s, collada::Shape::Triangle(..))).count()
                }
            })
            .sum()
    })
    .unwrap_or(0)
}

#[cfg(feature = "evidence")]
fn collada_stats(r: &dyn Any) -> Stats {
    let Some((objs, anims, skels, _)) = r.downcast_ref::<Collada>() else { return Stats::new() };
    Stats::from([
        ("meshes", n(objs.as_ref().map_or(0, |s| s.objects.len()))),
        ("vertices", n(objs.as_ref().map_or(0, |s| s.objects.iter().map(|o| o.vertices.len()).sum()))),
        ("triangles", n(collada_tris(objs.as_ref()))),
        ("materials", n(objs.as_ref().map_or(0, |s| s.material_library.as_ref().map_or(0, |_| 1)))),
        ("joints", n(skels.as_ref().map_or(0, |s| s.iter().map(|k| k.joints.len()).sum()))),
        ("channels", n(anims.as_ref().map_or(0, Vec::len))),
    ])
}

/// The collada crate: the document, then its meshes (`get_obj_set`), animations, skeletons and
/// skin bindings, as its users read them.
#[cfg(feature = "evidence")]
pub fn collada(src: &Src) -> Out {
    let bytes = src.bytes()?;
    let text = String::from_utf8_lossy(&bytes);
    let doc = collada::document::ColladaDocument::from_str(&text).map_err(e)?;
    let objs = doc.get_obj_set();
    let anims = doc.get_animations();
    let skels = doc.get_skeletons();
    let binds = doc.get_bind_data_set();
    let k = collada_tris(objs.as_ref());
    let r: Collada = (objs, anims, skels, binds);
    loaded(r, k, collada_stats)
}

#[cfg(feature = "evidence")]
fn bvh_stats(r: &dyn Any) -> Stats {
    let Some(b) = r.downcast_ref::<bvh_anim::Bvh>() else { return Stats::new() };
    let channels: usize = b.joints().map(|j| j.data().channels().len()).sum();
    Stats::from([
        ("joints", n(b.joints().count())),
        ("animations", 1),
        ("frames", n(b.frames().count())),
        ("bvh_channels", n(channels)),
    ])
}

#[cfg(feature = "evidence")]
pub fn bvh_anim(src: &Src) -> Out {
    let bytes = src.bytes()?;
    let bvh = bvh_anim::from_bytes(&bytes[..]).map_err(e)?;
    let channels: usize = bvh.joints().map(|j| j.data().channels().len()).sum();
    let k = channels * bvh.frames().count();
    loaded(bvh, k, bvh_stats)
}

// --- assimp ---------------------------------------------------------------------------------

pub const ASSIMP: &str = "assimp";
pub const ASSIMP_GLB: &str = "assimp → GLB";

#[cfg(feature = "assimp")]
pub mod assimp {
    use std::any::Any;
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::path::Path;
    use std::sync::OnceLock;

    use super::{Loaded, Out, Src, Stats};

    #[repr(C)]
    #[derive(Default)]
    struct ShimStats {
        meshes: u64,
        vertices: u64,
        triangles: u64,
        lines: u64,
        points: u64,
        polygons: u64,
        materials: u64,
        nodes: u64,
        joints: u64,
        animations: u64,
        channels: u64,
        textures: u64,
        cameras: u64,
        lights: u64,
        incomplete: u64,
    }

    unsafe extern "C" {
        fn shim_read(path: *const c_char, data: *const u8, len: usize, flags: u32, err: *mut c_char, errlen: usize) -> *mut c_void;
        fn shim_stats(h: *mut c_void, out: *mut ShimStats) -> i32;
        fn shim_free(h: *mut c_void);
        fn shim_export_glb(h: *mut c_void, out: *const c_char, err: *mut c_char, errlen: usize) -> i64;
        fn shim_extensions() -> *const c_char;
        fn shim_version() -> *const c_char;
    }

    /// `aiProcess_Triangulate`: the least post-processing that gives triangles, as Polysmelt's
    /// readers do. Everything else (joining identical vertices, normals, validation) is off, the
    /// fastest assimp can read a model.
    pub const FLAGS: u32 = 0x8;

    /// An importer holding a scene; freed on drop.
    pub struct Scene(*mut c_void);

    impl Drop for Scene {
        fn drop(&mut self) {
            unsafe { shim_free(self.0) }
        }
    }

    fn cpath(p: &Path) -> Result<CString, String> {
        CString::new(p.to_string_lossy().as_bytes()).map_err(|e| e.to_string())
    }

    fn from_err(buf: &[u8]) -> String {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let s = String::from_utf8_lossy(&buf[..end]).trim().to_owned();
        if s.is_empty() { "assimp failed without a message".into() } else { s }
    }

    pub fn read(src: &Src) -> Result<Scene, String> {
        let path = cpath(src.path())?;
        let mut err = vec![0u8; 1024];
        let (data, len) = match src {
            Src::File(_) => (std::ptr::null(), 0),
            Src::Bytes(b, _) => (b.as_ptr(), b.len()),
        };
        let h = unsafe { shim_read(path.as_ptr(), data, len, FLAGS, err.as_mut_ptr().cast(), err.len()) };
        if h.is_null() { Err(from_err(&err)) } else { Ok(Scene(h)) }
    }

    pub fn scene_stats(s: &Scene) -> Stats {
        let mut st = ShimStats::default();
        if unsafe { shim_stats(s.0, &raw mut st) } != 0 {
            return Stats::new();
        }
        Stats::from([
            ("meshes", st.meshes),
            ("vertices", st.vertices),
            ("triangles", st.triangles),
            ("lines", st.lines),
            ("points", st.points),
            ("polygons", st.polygons),
            ("materials", st.materials),
            ("nodes", st.nodes),
            ("joints", st.joints),
            ("animations", st.animations),
            ("channels", st.channels),
            ("images", st.textures),
            ("cameras", st.cameras),
            ("lights", st.lights),
            ("incomplete", st.incomplete),
        ])
    }

    fn stats(r: &dyn Any) -> Stats {
        if let Some(s) = r.downcast_ref::<Scene>() {
            return scene_stats(s);
        }
        if let Some((s, _)) = r.downcast_ref::<(Scene, i64)>() {
            return scene_stats(s);
        }
        Stats::new()
    }

    /// The scene as a GLB, written to `out` (or, `None`, to memory and dropped): its size.
    pub fn export(s: &Scene, out: Option<&Path>) -> Result<i64, String> {
        let out = out.map(cpath).transpose()?;
        let mut err = vec![0u8; 1024];
        let k = unsafe {
            shim_export_glb(s.0, out.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()), err.as_mut_ptr().cast(), err.len())
        };
        if k < 0 { Err(from_err(&err)) } else { Ok(k) }
    }

    pub fn load(src: &Src) -> Out {
        let s = read(src)?;
        let mut st = ShimStats::default();
        let k = if unsafe { shim_stats(s.0, &raw mut st) } == 0 { st.triangles as usize } else { 0 };
        Ok(Loaded { result: Box::new(s), units: k, stats })
    }

    /// A conversion: read, and write the scene as a GLB in memory (`glb2`, as `assimp export x.glb`).
    pub fn load_glb(src: &Src) -> Out {
        let s = read(src)?;
        let k = export(&s, None)?;
        Ok(Loaded { result: Box::new((s, k)), units: k as usize, stats })
    }

    /// Every extension assimp reads, lowercase.
    pub fn extensions() -> &'static [String] {
        static EXTS: OnceLock<Vec<String>> = OnceLock::new();
        EXTS.get_or_init(|| {
            let s = unsafe { CStr::from_ptr(shim_extensions()) }.to_string_lossy().into_owned();
            s.split(';').map(|e| e.trim().trim_start_matches("*.").to_ascii_lowercase()).filter(|e| !e.is_empty()).collect()
        })
    }

    pub fn version() -> String {
        unsafe { CStr::from_ptr(shim_version()) }.to_string_lossy().into_owned()
    }
}

/// Whether assimp reads files with this extension (never, without the `assimp` feature).
pub fn assimp_reads(ext: &str) -> bool {
    #[cfg(feature = "assimp")]
    {
        assimp::extensions().iter().any(|e| e == ext)
    }
    #[cfg(not(feature = "assimp"))]
    {
        let _ = ext;
        false
    }
}

pub fn assimp_version() -> Option<String> {
    #[cfg(feature = "assimp")]
    {
        Some(assimp::version())
    }
    #[cfg(not(feature = "assimp"))]
    {
        None
    }
}

/// The competitor crates for a file extension, and assimp when it reads it.
pub fn competitors(ext: &str) -> Vec<Loader> {
    let l = |name, run| Loader { name, run };
    #[cfg_attr(not(feature = "assimp"), allow(unused_mut))]
    let mut out: Vec<Loader> = match ext {
        "gltf" | "glb" => vec![l("gltf (import)", gltf_import as Run), l("gltf (no image decode)", gltf_no_images)],
        "obj" => vec![l("tobj", tobj)],
        "stl" => vec![l("stl_io", stl_io)],
        "ply" => vec![l("ply-rs", ply_rs)],
        #[cfg(feature = "evidence")]
        "fbx" => vec![l("fbxcel-dom", fbxcel_dom)],
        #[cfg(feature = "evidence")]
        "3mf" => vec![l("threemf", threemf)],
        #[cfg(feature = "evidence")]
        "vox" => vec![l("dot_vox", dot_vox)],
        #[cfg(feature = "evidence")]
        "dae" => vec![l("collada", collada)],
        #[cfg(feature = "evidence")]
        "bvh" => vec![l("bvh_anim", bvh_anim)],
        _ => vec![],
    };
    #[cfg(feature = "assimp")]
    if assimp_reads(ext) {
        out.push(l(ASSIMP, assimp::load));
    }
    out
}

/// assimp's conversion to GLB, when it reads this extension.
pub fn assimp_glb(ext: &str) -> Option<Loader> {
    #[cfg(feature = "assimp")]
    if assimp_reads(ext) {
        return Some(Loader { name: ASSIMP_GLB, run: assimp::load_glb });
    }
    let _ = ext;
    None
}

/// Every loader by name, for a worker told a name and a file.
pub fn by_name(name: &str, ext: &str) -> Option<Loader> {
    let l = |name, run| Some(Loader { name, run });
    match name {
        POLYSMELT => l(POLYSMELT, polysmelt),
        POLYSMELT_GLB => l(POLYSMELT_GLB, polysmelt_glb),
        "polysmelt (threads: 0, all cores)" => l("polysmelt (threads: 0, all cores)", polysmelt_all_cores),
        "polysmelt (no image bytes)" => l("polysmelt (no image bytes)", polysmelt_no_image_bytes),
        "polysmelt (convert)" => l("polysmelt (convert)", polysmelt_convert),
        _ => competitors(ext).into_iter().chain(assimp_glb(ext)).find(|c| c.name == name),
    }
}
