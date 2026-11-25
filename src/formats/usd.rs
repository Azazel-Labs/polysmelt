//! USD (`.usd`, `.usda`, `.usdc`, `.usdz`) as a glTF document, read with the pure-Rust `openusd` crate. The composed
//! stage (references, payloads, variants and instances resolved) becomes the node hierarchy, each prim a node with its
//! local transform. Meshes are in their prim's own space and shared between instances; a point instancer places its
//! prototypes under a node per instance. The stage's up axis and units become a root node that carries them to glTF's
//! Y-up meters, and are recorded in `asset.extras.source` (see [`crate::source`]; USD's 0.01 meters and Y up are
//! assumed when the stage doesn't say), with its frame rate (`framesPerSecond`, else `timeCodesPerSecond`), its `doc`
//! as the description and its `comment`. `startTimeCode`, `endTimeCode`, `defaultPrim` and `customLayerData` are
//! kept as the stage gives them, and `timeCodesPerSecond` when the frame rate is `framesPerSecond`.
//!
//! Meshes keep their polygons' outlines (concave faces are ear-clipped, `holeIndices` faces left out) and their
//! normals, texture coordinates and display colors in any interpolation; with no normals a subdivision surface is
//! smoothed and a polygonal one faceted. The implicit surfaces (Cube, Sphere, Cylinder, Cone, Capsule, Plane) are
//! tessellated. `BasisCurves` become lines: linear curves through their points, cubic ones (Bézier, B-spline and
//! Catmull-Rom, with their wrap) sampled at eight steps a segment, which is what draws them where they are rather
//! than along their control hulls. `NurbsCurves` are sampled at eight steps a knot span within their ranges,
//! rational by their point weights, up to order [`MAX_ORDER`]; `HermiteCurves` at eight steps a segment, leaving
//! each point along its tangent. Past [`MAX_CURVE_POINTS`] samples in a stage, curves are drawn through their
//! control points, as are those whose order, knots or tangents don't make a curve. `Points` become points. Curves
//! and points take their display colors (per curve, per point or per segment end) and their bound material, and
//! leave widths out. Values are read at the stage's start time when it gives a time range, else at the earliest sample,
//! so the rest pose is the first frame.
//!
//! `UsdSkel` skeletons become joint nodes under their skeleton's node, skinned meshes glTF skins (joint influences
//! and the geometry's bind transform), blend shapes morph targets. Animated transforms, skeletal animations and
//! blend shape weights become one animation, sampled where the file has samples (long turns cut into short steps).
//! Cameras and lights (`KHR_lights_punctual`) come across with their prims; `userProperties:` attributes and
//! `customData` become node extras.
//!
//! Materials come from `UsdPreviewSurface`: the diffuse color or texture, the normal map, opacity, emissive color,
//! metallic and roughness (with the shader's fallbacks), textures read from the `.usdz` itself or found in the pack.
//! A surface with no material shows its display color: one color as the material, more as vertex colors.
//!
//! A file is read with `openusd`'s own resolver. A model in memory is read with [`MemResolver`], which serves the
//! layers it names from [`Assets::read`].

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::io::{self, Read as _};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Context as _, ErrorKind, Result};
use openusd::usd::SchemaBase as _;
use openusd::{ar, gf, sdf, tf, usd};
use openusd_schemas::geom::{
    self, CapsuleSchema, ConeSchema, CubeSchema, CylinderSchema, GprimSchema, ImageableExt, MeshSchema, PlaneSchema,
    PointBasedSchema, PointInstancerSchema, PrimvarsAPI, SphereSchema, SubsetSchema, XformQuery,
};
use openusd_schemas::shade::{self, MaterialBindingAPI};
use serde_json::{Map, Value, json};

use super::Input;
use super::xform::{IDENTITY, M4, Trs, decompose, inverse, mul, scale, set_transform, translate};
use crate::assets::{Assets, memory_key};
use crate::document::{self, Animation, Builder, Document, Keep, Part, Sampler};
use crate::images::{Images, Location, gray};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Specular, Tex, UvTransform};
use crate::mesh::{GRAY, Material, Prim, triangulate};
use crate::num::f32_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};

/// Prim types that are no node: shading networks, and what the prims using them read (subsets, blend shapes,
/// skeletal animations).
const NOT_NODES: &[&str] = &[
    "Material",
    "Shader",
    "NodeGraph",
    "GeomSubset",
    "BlendShape",
    "SkelAnimation",
];

/// Prim types drawn as surfaces: meshes and the implicit surfaces.
const GPRIMS: &[&str] = &[
    "Mesh",
    "Cube",
    "Sphere",
    "Cylinder",
    "Cylinder_1",
    "Cone",
    "Capsule",
    "Capsule_1",
    "Plane",
];

/// The steps each segment of a cubic curve is sampled at.
const CURVE_STEPS: u16 = 8;

/// The most points sampled curves add to a stage: each control point is a few bytes of the file but
/// [`CURVE_STEPS`] points drawn, so this bounds what a stage of curves costs. Past it, a curve is drawn through its
/// control points.
const MAX_CURVE_POINTS: usize = 1 << 22;

/// The highest order of NURBS curve sampled, where its control points' weights still fit a [`Sample`]: degree
/// seven, past the cubics nearly every curve is. Higher orders are drawn through their control points.
const MAX_ORDER: usize = 8;

/// The most texture coordinate sets, and the most color sets, a primitive keeps: past what any real model has, and
/// a bound on what a file with thousands of primvars costs.
const MAX_SETS: usize = 8;

/// The longest turn between two animation samples, in degrees: longer ones are cut into steps.
const MAX_STEP_DEGREES: f64 = 10.0;

/// The most nodes a stage makes: point instancers can place millions of instances.
const MAX_NODES: usize = 1 << 20;

/// How deeply point instancers nest in each other's prototypes.
const MAX_NESTING: usize = 8;

/// The stage as a document; one with nothing to show (a material library, a layer of overrides) has no nodes.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    load_capped(input, assets, keep, MAX_NODES)
}

/// [`load`], making at most `max_nodes` nodes: past them the rest of the stage is left out, and
/// `asset.extras.source` says so.
fn load_capped(input: Input<'_>, assets: &dyn Assets, keep: Keep, max_nodes: usize) -> Result<Document> {
    let path = input.path;
    let (stage, paths) = match input.bytes() {
        Some(bytes) => open_in_memory(bytes, input, assets)?,
        #[cfg(feature = "fs")]
        None => {
            let stage = usd::Stage::builder()
                .schema_registry(openusd_schemas::schema_registry())
                .open(path.to_str().context(ErrorKind::Corrupt, "the path isn't UTF-8")?)
                .context(ErrorKind::Corrupt, "opening the USD stage")?;
            let paths = prim_paths(&stage)?;
            (stage, paths)
        }
        #[cfg(not(feature = "fs"))]
        None => unreachable!("a model without its bytes needs the `fs` feature"),
    };
    let z_up = matches!(stage.stage_metadata("upAxis"), Ok(Some(v)) if v.as_str() == Some("Z"));
    let meters = match stage.stage_metadata("metersPerUnit") {
        Ok(Some(sdf::Value::Double(m))) if m > 0.0 => m,
        Ok(Some(sdf::Value::Float(m))) if m > 0.0 => f64::from(m),
        _ => 0.01,
    };
    // A value that isn't animated reads the same at any time; an animated one shows the first frame.
    let time = if stage.has_authored_time_code_range() {
        usd::TimeCode::new(stage.start_time_code())
    } else {
        usd::TimeCode::EARLIEST
    };
    let tcps = Some(stage.time_codes_per_second())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(24.0);
    let prims: Vec<usd::Prim> = paths.iter().filter_map(|p| stage.prim(p.clone()).ok()).collect();
    let mut r = Reader {
        stage: &stage,
        file: path,
        assets,
        keep,
        time,
        meters,
        usdz: Usdz::open(input),
        b: Builder::default(),
        nodes: Vec::new(),
        meshes: Vec::new(),
        mesh_triangles: Vec::new(),
        mesh_of: HashMap::new(),
        materials: Vec::new(),
        material_of: HashMap::new(),
        material_reads: Vec::new(),
        images: Images::new(!keep.image_bytes, input.dir()),
        image_of: HashMap::new(),
        skins: Vec::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        animated: Vec::new(),
        skeletons: HashMap::new(),
        pending: Vec::new(),
        morphed: Vec::new(),
        skinned_roots: Vec::new(),
        drawn: 0,
        curve_points: 0,
        limit: input.limit,
        max_nodes,
        truncated: false,
    };
    let roots = r.walk(&prims, &prims, None, 0)?;
    r.skin_meshes()?;
    r.pose_weights();
    if r.nodes.is_empty() {
        return Ok(Document::empty(Map::new()));
    }
    let animations = if keep.animations {
        r.animations(tcps)
    } else {
        Vec::new()
    };

    let mut basis = if z_up {
        [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]]
    } else {
        IDENTITY
    };
    basis = mul(&basis, &scale([meters; 3]));
    let mut scene_nodes = roots;
    #[allow(clippy::float_cmp, reason = "only the exact identity needs no root")]
    if basis != IDENTITY {
        let mut root = json!({ "name": "RootNode", "children": scene_nodes });
        set_transform(&mut root, &basis);
        r.nodes.push(root);
        scene_nodes = vec![r.nodes.len() - 1];
    }
    scene_nodes.append(&mut r.skinned_roots);

    let mut source = Map::new();
    if r.truncated {
        source.insert(
            "truncated".into(),
            json!(format!("over {max_nodes} nodes: the rest are left out")),
        );
    }
    if stage.has_authored_time_code_range() {
        source.insert("startTimeCode".into(), json!(stage.start_time_code()));
        source.insert("endTimeCode".into(), json!(stage.end_time_code()));
    }
    if let Some(p) = stage.default_prim() {
        source.insert("defaultPrim".into(), json!(p.as_str()));
    }
    if authored(&stage, "framesPerSecond") {
        // The frame rate is the frames', and time codes count at their own rate.
        source.insert("timeCodesPerSecond".into(), json!(tcps));
    }
    stage_info(&stage, z_up, meters, tcps).write(&mut source);
    if let Some(Value::Object(data)) = stage_metadata_json(&stage, "customLayerData") {
        source.insert("customLayerData".into(), Value::Object(data));
    }
    let mut json = Map::new();
    let mut asset = json!({ "extras": { "source": source } });
    if let Some(c) = asset["extras"]["source"].get("copyright").cloned() {
        asset["copyright"] = c;
    }
    json.insert("asset".into(), asset);
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
    if !r.lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    let (materials, images) = (std::mem::take(&mut r.materials), std::mem::take(&mut r.images));
    let mut b = r.b;
    b.animations = animations;
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// Whether the stage's root or session layer authors the metadata `field`.
fn authored(stage: &usd::Stage, field: &str) -> bool {
    let has = |layer: &sdf::Layer| layer.pseudo_root().is_some_and(|r| r.has_field(field));
    has(&stage.root_layer()) || stage.session_layer().is_some_and(|l| has(&l))
}

/// The stage metadata `field` as JSON, when it's authored.
fn stage_metadata_json(stage: &usd::Stage, field: &str) -> Option<Value> {
    if !authored(stage, field) {
        return None;
    }
    let value = stage.stage_metadata(field).ok().flatten()?;
    json_of(&value)
}

/// What the stage says of itself: its up axis (USD's Y when it doesn't say), its `metersPerUnit` (0.01 when it
/// doesn't), its frame rate (`framesPerSecond`, else `timeCodesPerSecond`, else USD's 24), its `doc` as the
/// description and its `comment`.
fn stage_info(stage: &usd::Stage, z_up: bool, meters: f64, tcps: f64) -> SourceInfo {
    let fact = |field: &str, v: f64| {
        if authored(stage, field) {
            Fact::Declared(v)
        } else {
            Fact::Assumed(v)
        }
    };
    let up = if z_up { Axis::PosZ } else { Axis::PosY };
    let up_given = matches!(stage.stage_metadata("upAxis"), Ok(Some(v)) if matches!(v.as_str(), Some("Y" | "Z")));
    let text = |field: &str| match stage_metadata_json(stage, field) {
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s.trim().to_owned()),
        _ => None,
    };
    let fps = Some(stage.frames_per_second()).filter(|f| f.is_finite() && *f > 0.0);
    let mut info = SourceInfo {
        format: Some("USD".into()),
        meters_per_unit: fact("metersPerUnit", meters),
        unit: crate::source::unit_name(meters).map(str::to_owned),
        up_axis: if up_given && authored(stage, "upAxis") {
            Fact::Declared(up)
        } else {
            Fact::Assumed(up)
        },
        // USD's coordinate system is right-handed by definition.
        handedness: Fact::Declared(Handedness::Right),
        frame_rate: match fps {
            Some(f) if authored(stage, "framesPerSecond") => Fact::Declared(f),
            _ => fact("timeCodesPerSecond", tcps),
        },
        description: text("documentation").or_else(|| text("doc")),
        comments: text("comment").into_iter().collect(),
        ..SourceInfo::default()
    };
    info.converted.units = (meters - 1.0).abs() > 1e-12;
    info.converted.up_axis = z_up;
    info
}

/// The paths of the stage's prims, composed (which reads the layers they reference), parents before children.
fn prim_paths(stage: &usd::Stage) -> Result<Vec<sdf::Path>> {
    let mut paths = Vec::new();
    stage
        .traverse(usd::PrimPredicate::DEFAULT.with_instance_proxies(true), |p| {
            paths.push(p.clone());
        })
        .context(ErrorKind::Corrupt, "composing the USD stage")?;
    Ok(paths)
}

/// The prim's schema type name.
fn kind(prim: &usd::Prim) -> Option<String> {
    prim.type_name().ok().flatten().map(|t| t.as_str().to_owned())
}

/// Whether `path` is `root` or beneath it.
fn within(path: &sdf::Path, root: &sdf::Path) -> bool {
    let (p, r) = (path.as_str(), root.as_str());
    p == r || (p.starts_with(r) && p.as_bytes().get(r.len()) == Some(&b'/'))
}

/// The last part of a path: a prim's name, or a joint's.
fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A USD matrix (row vectors) as an [`M4`] (column vectors): its transpose.
fn m4(m: &gf::Matrix4d) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| m.0[j * 4 + i]))
}

/// The rotation for a unit quaternion (x, y, z, w).
fn rotation([x, y, z, w]: [f64; 4]) -> M4 {
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            0.0,
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            0.0,
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A quaternion (w, x, y, z), as USD lists one, as glTF's unit (x, y, z, w); `None` when it has no length.
fn unit_quaternion([w, x, y, z]: [f32; 4]) -> Option<[f64; 4]> {
    let q = [x, y, z, w].map(f64::from);
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    (len > 0.0 && len.is_finite()).then(|| q.map(|c| c / len))
}

/// The angle between two unit quaternions' rotations, in degrees.
fn turn(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dot: f64 = (0..4).map(|k| a[k] * b[k]).sum();
    2.0 * dot.abs().min(1.0).acos().to_degrees()
}

/// Times sorted, without repeats or ones that aren't finite.
fn timeline(mut times: Vec<f64>) -> Vec<f64> {
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    times
}

/// A `UsdSkel` skeleton drawn as joint nodes.
struct Skel {
    /// Joint paths (`Hips/Spine`), in the skeleton's order.
    joints: Vec<String>,
    /// Each joint's node.
    nodes: Vec<usize>,
    /// Each joint's bind transform in the skeleton's space (row vectors), identity where the file has none.
    binds: Vec<gf::Matrix4d>,
    /// Each joint's local transform with no animation: its rest transform, or what its bind transform makes it.
    rest: Vec<M4>,
    /// The `SkelAnimation` driving it.
    animation: Option<sdf::Path>,
}

/// A skinned mesh waiting for its skeleton's joint nodes.
struct Pending {
    /// The mesh prim's node, which keeps its place in the hierarchy.
    node: usize,
    mesh: usize,
    prim: usd::Prim,
    skeleton: String,
}

/// A skeletal animation's values at one time, per joint of `joints`.
struct SkelSample {
    joints: Vec<String>,
    translations: Vec<[f32; 3]>,
    /// (w, x, y, z).
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
}

impl SkelSample {
    /// Joint `k`'s local transform as translation, rotation and scale, `rest`'s where the animation has none.
    fn trs(&self, k: usize, rest: &Trs) -> ([f64; 3], [f64; 4], [f64; 3]) {
        let t = self.translations.get(k).map_or(rest.t, |v| v.map(f64::from));
        let r = self
            .rotations
            .get(k)
            .and_then(|q| unit_quaternion(*q))
            .unwrap_or(rest.r);
        let s = self.scales.get(k).map_or(rest.s, |v| v.map(f64::from));
        (t, r, s)
    }
}

/// One animation channel's samples, before they go into the buffer.
struct Channel {
    node: usize,
    path: &'static str,
    times: Vec<f64>,
    values: Vec<f64>,
}

struct Reader<'a> {
    stage: &'a usd::Stage,
    file: &'a Path,
    assets: &'a dyn Assets,
    keep: Keep,
    time: usd::TimeCode,
    meters: f64,
    usdz: Option<Usdz<'a>>,
    b: Builder,
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    /// Each mesh's triangles.
    mesh_triangles: Vec<usize>,
    /// A surface (its prototype's path when it's an instance, its material binding, whether it's skinned) to its
    /// mesh, `None` when it has no faces.
    mesh_of: HashMap<String, Option<usize>>,
    materials: Vec<Full>,
    /// Material prim path (or `display:<rgba>` for a display color), and whether it's double-sided, to its index
    /// in `materials`.
    material_of: HashMap<String, usize>,
    /// What each material reads from the surface it's on, by index in `materials`.
    material_reads: Vec<Reads>,
    images: Images,
    /// Texture asset path to its index in `images`, or `None` when it couldn't be found.
    image_of: HashMap<String, Option<usize>>,
    skins: Vec<Value>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    /// Nodes whose transforms are animated, and their prims.
    animated: Vec<(usize, usd::Prim)>,
    /// Skeleton prim path to its joints.
    skeletons: HashMap<String, Skel>,
    pending: Vec<Pending>,
    /// Nodes drawing a mesh with blend shapes, and the mesh prim.
    morphed: Vec<(usize, usd::Prim)>,
    /// Nodes for skinned meshes, at the scene's root: glTF places a skinned mesh by its joints alone.
    skinned_roots: Vec<usize>,
    /// Triangles drawn, every instance counted.
    drawn: usize,
    /// Points sampled on curves, up to [`MAX_CURVE_POINTS`].
    curve_points: usize,
    /// The most triangles the stage may draw.
    limit: usize,
    /// The most nodes the stage may make.
    max_nodes: usize,
    /// Whether nodes were left out for `max_nodes`.
    truncated: bool,
}

impl Reader<'_> {
    /// Adds a node; `None` once there are `max_nodes`, which leaves the rest of the stage out.
    fn push_node(&mut self, node: Value) -> Option<usize> {
        if self.nodes.len() >= self.max_nodes {
            self.truncated = true;
            return None;
        }
        self.nodes.push(node);
        Some(self.nodes.len() - 1)
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let p = &mut self.nodes[parent];
        match p.get_mut("children").and_then(Value::as_array_mut) {
            Some(c) => c.push(json!(child)),
            None => p["children"] = json!([child]),
        }
    }

    /// Nodes for `prims` (parents before children): all of them when `scope` is `None`, else the prototype rooted
    /// at `scope.0`, hung under node `scope.1`. `all` is every prim, where point instancers find their prototypes.
    /// Returns the nodes with no parent node.
    fn walk(
        &mut self,
        prims: &[usd::Prim],
        all: &[usd::Prim],
        scope: Option<(&sdf::Path, usize)>,
        depth: usize,
    ) -> Result<Vec<usize>> {
        let mut node_of: HashMap<&str, usize> = HashMap::new();
        let mut roots = Vec::new();
        // A point instancer's prototypes show only where it places them.
        let mut instancers: Vec<&sdf::Path> = Vec::new();
        for prim in prims {
            let path = prim.path();
            if scope.is_some_and(|(root, _)| !within(path, root)) || instancers.iter().any(|i| within(path, i)) {
                continue;
            }
            let Some(kind) = kind(prim) else { continue };
            if NOT_NODES.contains(&kind.as_str()) {
                continue;
            }
            let parent = match scope {
                Some((root, top)) if path == root => Some(top),
                _ => match prim.parent() {
                    Some(p) if p.path().as_str() != "/" => match node_of.get(p.path().as_str()) {
                        Some(&n) => Some(n),
                        // Its parent isn't drawn.
                        None => continue,
                    },
                    _ if scope.is_some() => continue,
                    _ => None,
                },
            };
            if !self.shown(prim) {
                continue;
            }
            let query = XformQuery::for_prim(prim).ok();
            let local = query
                .as_ref()
                .and_then(|q| q.local_transformation(self.time).ok())
                .map_or(IDENTITY, |m| m4(&m));
            let mut node = json!({ "name": leaf(path.as_str()) });
            set_transform(&mut node, &local);
            if self.keep.extras {
                let extras = self.extras(prim);
                if !extras.is_empty() {
                    node["extras"] = Value::Object(extras);
                }
            }
            let Some(n) = self.push_node(node) else {
                return Ok(roots);
            };
            // A prim that resets the transform stack ignores its parents' transforms.
            let resets = query.as_ref().is_some_and(XformQuery::resets_xform_stack);
            match (parent, scope) {
                (_, Some((_, top))) if resets => self.add_child(top, n),
                (Some(p), _) if !resets => self.add_child(p, n),
                _ => roots.push(n),
            }
            node_of.insert(path.as_str(), n);
            if self.keep.animations
                && query
                    .as_ref()
                    .is_some_and(|q| q.transform_might_be_time_varying().unwrap_or(false))
            {
                self.animated.push((n, prim.clone()));
            }
            match kind.as_str() {
                "PointInstancer" => {
                    instancers.push(path);
                    if depth < MAX_NESTING {
                        self.instancer(prim, all, n, depth)?;
                    }
                }
                "Skeleton" => self.skeleton(prim, n),
                "Camera" if self.keep.cameras => {
                    let camera = self.camera(prim);
                    self.cameras.push(camera);
                    self.nodes[n]["camera"] = json!(self.cameras.len() - 1);
                }
                "DistantLight" | "SphereLight" | "DiskLight" | "RectLight" | "CylinderLight" if self.keep.lights => {
                    let light = self.light(prim, &kind);
                    self.lights.push(light);
                    self.nodes[n]["extensions"] = json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } });
                }
                // glTF has no light all around: the dome's color, intensity and texture go in the node's extras.
                "DomeLight" | "DomeLight_1" if self.keep.lights => {
                    let dome = self.dome_light(prim);
                    self.nodes[n]["extras"]["domeLight"] = dome;
                }
                k if GPRIMS.contains(&k) => self.surface(prim, n)?,
                "BasisCurves" | "NurbsCurves" | "HermiteCurves" | "Points" => self.loose(prim, n),
                _ => {}
            }
        }
        Ok(roots)
    }

    /// A point instancer's instances: a node each under `node`, holding its prototype's nodes.
    fn instancer(&mut self, prim: &usd::Prim, all: &[usd::Prim], node: usize, depth: usize) -> Result<()> {
        let pi = geom::PointInstancer::from_prim_unchecked(prim.clone());
        let protos = pi.prototypes_rel().targets().unwrap_or_default();
        let which = ints(self.value(&pi.proto_indices_attr()));
        let positions = self.value(&pi.positions_attr()).and_then(vec3s).unwrap_or_default();
        let orients = self
            .value(&prim.attribute("orientationsf"))
            .or_else(|| self.value(&pi.orientations_attr()))
            .and_then(quats)
            .unwrap_or_default();
        let scales = self
            .value(&prim.attribute("scales"))
            .and_then(vec3s)
            .unwrap_or_default();
        let ids = int64s(self.value(&prim.attribute("ids")));
        let hidden = int64s(self.value(&prim.attribute("invisibleIds")));
        // Each prototype's prims, gathered once.
        let parts: Vec<Vec<usd::Prim>> = protos
            .iter()
            .map(|root| all.iter().filter(|p| within(p.path(), root)).cloned().collect())
            .collect();
        for (i, &w) in which.iter().enumerate() {
            let id = ids
                .get(i)
                .copied()
                .unwrap_or_else(|| i64::try_from(i).unwrap_or(i64::MAX));
            if hidden.contains(&id) {
                continue;
            }
            let Some(w) = usize::try_from(w).ok().filter(|&w| w < protos.len()) else {
                continue;
            };
            let s = scales.get(i).copied().unwrap_or([1.0; 3]);
            let q = orients.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let t = positions.get(i).copied().unwrap_or_default();
            let mut inst = json!({ "name": format!("instance_{i}") });
            set_transform(&mut inst, &m4(&instance_matrix(s, q, t)));
            let Some(n) = self.push_node(inst) else {
                return Ok(());
            };
            self.add_child(node, n);
            self.walk(&parts[w], all, Some((&protos[w], n)), depth + 1)?;
        }
        Ok(())
    }

    /// A skeleton's joints, as nodes under its node `node`.
    fn skeleton(&mut self, prim: &usd::Prim, node: usize) {
        let joints = tokens(self.value(&prim.attribute("joints")));
        let binds = matrices(self.value(&prim.attribute("bindTransforms")));
        let rests = matrices(self.value(&prim.attribute("restTransforms")));
        let binds: Vec<gf::Matrix4d> = if binds.len() == joints.len() {
            binds
        } else {
            vec![gf::Matrix4d::IDENTITY; joints.len()]
        };
        let index: HashMap<&str, usize> = joints.iter().enumerate().map(|(i, j)| (j.as_str(), i)).collect();
        let parents: Vec<Option<usize>> = joints
            .iter()
            .enumerate()
            .map(|(i, j)| {
                j.rsplit_once('/')
                    .and_then(|(p, _)| index.get(p).copied())
                    .filter(|&p| p < i)
            })
            .collect();
        let rest: Vec<M4> = (0..joints.len())
            .map(|i| {
                if rests.len() == joints.len() {
                    return m4(&rests[i]);
                }
                let bind = m4(&binds[i]);
                parents[i].map_or(bind, |p| mul(&inverse(&m4(&binds[p])), &bind))
            })
            .collect();
        let animation = inherited_target(prim, "skel:animationSource");
        let posed = animation
            .as_ref()
            .and_then(|a| self.skel_sample(a, self.time))
            .map(|s| {
                let at: HashMap<&str, usize> = s.joints.iter().enumerate().map(|(k, j)| (j.as_str(), k)).collect();
                joints
                    .iter()
                    .enumerate()
                    .map(|(i, j)| {
                        let k = *at.get(j.as_str())?;
                        let (t, r, sc) = s.trs(k, &decompose(&rest[i]));
                        Some(mul(&mul(&translate(t), &rotation(r)), &scale(sc)))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut nodes = Vec::with_capacity(joints.len());
        for (i, j) in joints.iter().enumerate() {
            let local = posed.get(i).copied().flatten().unwrap_or(rest[i]);
            let mut n = json!({ "name": leaf(j) });
            set_transform(&mut n, &local);
            // Past the node cap the skeleton isn't made: its surfaces go unskinned.
            let Some(n) = self.push_node(n) else {
                return;
            };
            self.add_child(parents[i].map_or(node, |p| nodes[p]), n);
            nodes.push(n);
        }
        self.skeletons.insert(
            prim.path().as_str().to_owned(),
            Skel {
                joints,
                nodes,
                binds,
                rest,
                animation,
            },
        );
    }

    /// A skeletal animation's values at `time`.
    fn skel_sample(&self, animation: &sdf::Path, time: usd::TimeCode) -> Option<SkelSample> {
        let a = self.stage.prim(animation.clone()).ok()?;
        let at = |name: &str| a.attribute(name).get_at::<sdf::Value>(time).ok().flatten();
        Some(SkelSample {
            joints: tokens(at("joints")),
            translations: at("translations").and_then(vec3s).unwrap_or_default(),
            rotations: at("rotations").and_then(quats).unwrap_or_default(),
            scales: at("scales").and_then(vec3s).unwrap_or_default(),
        })
    }

    /// A camera's glTF form. Apertures and focal length are in tenths of a scene unit; the clipping planes are in
    /// scene units, which a node's scale doesn't reach, so they're converted here.
    fn camera(&self, prim: &usd::Prim) -> Value {
        let num = |name: &str, fallback: f64| f64::from(self.number(&prim.attribute(name), fallback));
        let focal = num("focalLength", 50.0).max(1e-6);
        let h = num("horizontalAperture", 20.955).max(1e-6);
        let v = num("verticalAperture", 15.2908).max(1e-6);
        let [near, far] = self
            .value(&prim.attribute("clippingRange"))
            .as_ref()
            .and_then(pair)
            .unwrap_or([1.0, 1_000_000.0]);
        let near = (near * self.meters).max(1e-6);
        let far = (far * self.meters).max(near * 1.000_001);
        if self.token(&prim.attribute("projection")).as_deref() == Some("orthographic") {
            return json!({ "type": "orthographic", "orthographic": {
                "xmag": h / 20.0 * self.meters, "ymag": v / 20.0 * self.meters, "znear": near, "zfar": far,
            } });
        }
        json!({ "type": "perspective", "perspective": {
            "yfov": 2.0 * (v / 2.0 / focal).atan(), "aspectRatio": h / v, "znear": near, "zfar": far,
        } })
    }

    /// A light's `KHR_lights_punctual` form. A disk or rectangle lights one side, as a spot with a hemisphere's cone.
    fn light(&self, prim: &usd::Prim, kind: &str) -> Value {
        let number = |name: &str, fallback: f64| self.light_number(prim, name, fallback);
        let (color, intensity) = self.light_color(prim);
        let mut out = json!({ "color": color, "intensity": intensity });
        let half = std::f64::consts::FRAC_PI_2;
        match kind {
            "DistantLight" => out["type"] = json!("directional"),
            "DiskLight" | "RectLight" => {
                out["type"] = json!("spot");
                out["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": half });
            }
            _ => {
                let cone = number("shaping:cone:angle", 90.0);
                if kind == "SphereLight" && cone < 90.0 {
                    let outer = cone.to_radians().clamp(1e-4, half);
                    let soft = number("shaping:cone:softness", 0.0).clamp(0.0, 1.0);
                    out["type"] = json!("spot");
                    out["spot"] = json!({ "innerConeAngle": outer * (1.0 - soft) * 0.999, "outerConeAngle": outer });
                } else {
                    out["type"] = json!("point");
                }
            }
        }
        out
    }

    /// A light's input: by its `inputs:` name, or the name it had before that prefix.
    fn light_input(&self, prim: &usd::Prim, name: &str) -> Option<sdf::Value> {
        self.value(&prim.attribute(format!("inputs:{name}").as_str()))
            .or_else(|| self.value(&prim.attribute(name)))
    }

    fn light_number(&self, prim: &usd::Prim, name: &str, fallback: f64) -> f64 {
        match self.light_input(prim, name) {
            Some(sdf::Value::Float(v)) => f64::from(v),
            Some(sdf::Value::Double(v)) => v,
            Some(sdf::Value::Half(v)) => f64::from(v.to_f32()),
            _ => fallback,
        }
    }

    /// A light's color and its intensity with its exposure applied.
    fn light_color(&self, prim: &usd::Prim) -> ([f32; 3], f64) {
        let color = self
            .light_input(prim, "color")
            .and_then(vec3s)
            .and_then(|c| c.first().copied())
            .unwrap_or([1.0; 3]);
        let intensity =
            self.light_number(prim, "intensity", 1.0).max(0.0) * self.light_number(prim, "exposure", 0.0).exp2();
        (color, intensity)
    }

    /// A dome light, which glTF has no light for: its color, intensity, and environment map, for the node's extras.
    fn dome_light(&self, prim: &usd::Prim) -> Value {
        let (color, intensity) = self.light_color(prim);
        let mut out = json!({ "color": color, "intensity": intensity });
        if let Some(sdf::Value::AssetPath(a)) = self.light_input(prim, "texture:file") {
            out["texture"] = json!(a.authored_path);
        }
        if let Some(format) = self.token(&prim.attribute("inputs:texture:format")) {
            out["format"] = json!(format);
        }
        out
    }

    /// Whether a surface is drawn from both sides: its `doubleSided`, off unless authored.
    fn double_sided(&self, prim: &usd::Prim) -> bool {
        matches!(self.value(&prim.attribute("doubleSided")), Some(sdf::Value::Bool(true)))
    }

    /// A prim's `userProperties:` attributes (by their names after the prefix) and its `customData`.
    fn extras(&self, prim: &usd::Prim) -> Map<String, Value> {
        let mut out = Map::new();
        for attr in prim
            .authored_attributes_in_namespace("userProperties")
            .unwrap_or_default()
        {
            let name = attr.name();
            let name = name.strip_prefix("userProperties:").unwrap_or(name);
            if let Some(v) = self.value(&attr).as_ref().and_then(json_of) {
                out.insert(name.to_owned(), v);
            }
        }
        if let Ok(Some(data)) = prim.custom_data()
            && let Some(v) = json_of(&data).filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
        {
            out.insert("customData".into(), v);
        }
        out
    }

    /// A surface on node `node`: its mesh there, or for a skinned surface, on a node of its own once its skeleton
    /// is drawn.
    fn surface(&mut self, prim: &usd::Prim, node: usize) -> Result<()> {
        let skin = if self.keep.skins { self.skin_binding(prim) } else { None };
        let Some(mesh) = self.mesh(prim, skin.as_ref().map(|s| s.1))? else {
            return Ok(());
        };
        self.drawn += self.mesh_triangles[mesh];
        ensure!(
            self.drawn <= self.limit,
            LimitExceeded,
            "too many triangles (over {})",
            self.limit
        );
        if let Some((skeleton, _)) = skin {
            self.pending.push(Pending {
                node,
                mesh,
                prim: prim.clone(),
                skeleton,
            });
        } else {
            self.hold(node, mesh, prim);
        }
        Ok(())
    }

    /// Curves or points on node `node`: a mesh of their lines or points, made once per prototype and material
    /// binding.
    fn loose(&mut self, prim: &usd::Prim, node: usize) {
        let source = match prim.prim_in_prototype() {
            Ok(Some(p)) if prim.is_instance_proxy().unwrap_or(false) => p.path().as_str().to_owned(),
            _ => prim.path().as_str().to_owned(),
        };
        let binding = bound(prim).map(|b| b.as_str().to_owned()).unwrap_or_default();
        let key = format!("{source}|{binding}|loose");
        let mesh = if let Some(&m) = self.mesh_of.get(&key) {
            m
        } else {
            let m = self.build_loose(prim);
            self.mesh_of.insert(key, m);
            m
        };
        if let Some(m) = mesh {
            self.nodes[node]["mesh"] = json!(m);
        }
    }

    /// The mesh of a curves prim's lines or a `Points` prim's points; `None` when it draws nothing.
    fn build_loose(&mut self, prim: &usd::Prim) -> Option<usize> {
        let mut points = vec3s(self.value(&prim.attribute("points"))?)?;
        let count = points.len();
        let type_name = kind(prim);
        // A Hermite curve's tangents, one a point, follow its points.
        let hermite = type_name.as_deref() == Some("HermiteCurves")
            && match self.value(&prim.attribute("tangents")).and_then(vec3s) {
                Some(tangents) if tangents.len() == count => {
                    points.extend(tangents);
                    true
                }
                _ => false,
            };
        let api = PrimvarsAPI::from_prim_unchecked(prim.clone());
        let display = self.curve_primvar(&api, "displayColor", vec3s);
        let opacity = self.curve_primvar(&api, "displayOpacity", floats);
        let vertex_colors =
            display.as_ref().is_some_and(|d| !d.constant()) || opacity.as_ref().is_some_and(|o| !o.constant());
        // Display colors that vary are vertex colors, whatever material is bound, as on surfaces.
        let colored = vertex_colors;
        let material = match bound(prim) {
            Some(path) => self.material(Some(path), GRAY, false),
            None if vertex_colors => self.material(None, [1.0; 4], false),
            None => {
                let [r, g, b] = display
                    .as_ref()
                    .and_then(|d| d.values.first().copied())
                    .unwrap_or([GRAY[0], GRAY[1], GRAY[2]]);
                let a = opacity.as_ref().and_then(|o| o.values.first().copied()).unwrap_or(1.0);
                self.material(None, [r, g, b, a.clamp(0.0, 1.0)], false)
            }
        };
        let curves = matches!(
            type_name.as_deref(),
            Some("BasisCurves" | "NurbsCurves" | "HermiteCurves")
        );
        let mut out = Prim {
            material,
            ..Prim::default()
        };
        // The curve's first vertex, and the vertex before, while the curve is unbroken.
        let (mut first, mut prev) = (None, None);
        let mut emit = |s: &Sample| {
            if s.starts {
                (first, prev) = (None, None);
            }
            let p = s.eval(|i| points.get(i).copied());
            let Some(p) = p.filter(|p| p.iter().all(|x| x.is_finite())) else {
                // A point that isn't there breaks the curve.
                prev = None;
                return;
            };
            let Ok(v) = u32::try_from(out.positions.len()) else {
                return;
            };
            out.positions.push(p);
            if colored {
                let [r, g, b] = display.as_ref().and_then(|d| along(d, s, |c| c)).unwrap_or([1.0; 3]);
                let [a] = opacity.as_ref().and_then(|o| along(o, s, |a| [a])).unwrap_or([1.0]);
                out.colors.push([r, g, b, a].map(|c| c.clamp(0.0, 1.0)));
            }
            if !curves {
                out.points.push(v);
                return;
            }
            if let Some(u) = prev {
                out.lines.extend_from_slice(&[u, v]);
            }
            first = first.or(Some(v));
            prev = Some(v);
            if s.closes
                && let Some(f) = first
                && f + 1 < v
            {
                out.lines.extend_from_slice(&[v, f]);
            }
        };
        if curves {
            let counts = ints(self.value(&prim.attribute("curveVertexCounts")));
            let budget = MAX_CURVE_POINTS.saturating_sub(self.curve_points);
            self.curve_points += match type_name.as_deref() {
                Some("NurbsCurves") => {
                    let nurbs = Nurbs {
                        orders: ints(self.value(&prim.attribute("order"))),
                        knots: self
                            .value(&prim.attribute("knots"))
                            .and_then(doubles)
                            .unwrap_or_default(),
                        ranges: self
                            .value(&prim.attribute("ranges"))
                            .and_then(double2s)
                            .unwrap_or_default(),
                        weights: self
                            .value(&prim.attribute("pointWeights"))
                            .and_then(doubles)
                            .unwrap_or_default(),
                    };
                    nurbs_samples(&counts, count, &nurbs, budget, &mut emit)
                }
                // Without a tangent for each point, Hermite curves are drawn through their points.
                Some("HermiteCurves") => curve_samples(
                    &counts,
                    count,
                    hermite.then_some(Basis::Hermite),
                    None,
                    budget,
                    &mut emit,
                ),
                _ => {
                    let cubic = match self.token(&prim.attribute("type")).as_deref() {
                        Some("linear") => None,
                        _ => Some(match self.token(&prim.attribute("basis")).as_deref() {
                            Some("bspline") => Basis::Bspline,
                            Some("catmullRom") => Basis::CatmullRom,
                            _ => Basis::Bezier,
                        }),
                    };
                    let wrap = self.token(&prim.attribute("wrap"));
                    curve_samples(&counts, count, cubic, wrap.as_deref(), budget, &mut emit)
                }
            };
        } else {
            for i in 0..count {
                emit(&Sample::at(0, i, i));
            }
        }
        if out.is_empty() {
            return None;
        }
        let mut primitives = self.b.primitives(out);
        for p in &mut primitives {
            p["material"] = json!(material);
        }
        self.meshes
            .push(json!({ "name": leaf(prim.path().as_str()), "primitives": primitives }));
        self.mesh_triangles.push(0);
        Some(self.meshes.len() - 1)
    }

    /// A primvar of curves or points. Varying (and face-varying) values are per segment end, so they're looked up
    /// by corner.
    fn curve_primvar<T>(
        &self,
        api: &PrimvarsAPI,
        name: &str,
        decode: fn(sdf::Value) -> Option<Vec<T>>,
    ) -> Option<Primvar<T>> {
        let pv = api.primvar(name).ok()?;
        let mut p = self.read_primvar(&pv, decode)?;
        if pv
            .interpolation()
            .is_ok_and(|t| matches!(t.as_token(), "varying" | "faceVarying"))
        {
            p.spread = Spread::FaceVarying;
        }
        Some(p)
    }

    /// Puts `mesh` on node `node`.
    fn hold(&mut self, node: usize, mesh: usize, prim: &usd::Prim) {
        self.nodes[node]["mesh"] = json!(mesh);
        if self.meshes[mesh].get("weights").is_some() {
            self.morphed.push((node, prim.clone()));
        }
    }

    /// The skeleton a skinned surface is bound to and how many joints its influences name, when it has joint
    /// influences and a skeleton.
    fn skin_binding(&self, prim: &usd::Prim) -> Option<(String, usize)> {
        self.value(&prim.attribute("primvars:skel:jointIndices"))?;
        let path = inherited_target(prim, "skel:skeleton")?;
        let skel = self.stage.prim(path.clone()).ok()?;
        if kind(&skel).as_deref() != Some("Skeleton") {
            return None;
        }
        let own = tokens(self.value(&prim.attribute("skel:joints")));
        let count = if own.is_empty() {
            tokens(self.value(&skel.attribute("joints"))).len()
        } else {
            own.len()
        };
        (count > 0).then(|| (path.as_str().to_owned(), count))
    }

    /// Skins for the skinned surfaces, now that every skeleton has its joint nodes.
    fn skin_meshes(&mut self) -> Result<()> {
        for p in std::mem::take(&mut self.pending) {
            let Some(skel) = self.skeletons.get(&p.skeleton).filter(|s| !s.nodes.is_empty()) else {
                // The skeleton isn't drawn: the surface goes unskinned where it is.
                if let Some(mesh) = self.mesh(&p.prim, None)? {
                    self.hold(p.node, mesh, &p.prim);
                }
                continue;
            };
            if self.nodes.len() >= self.max_nodes {
                // No room for the skinned node: the surface stays where it is, unskinned.
                self.truncated = true;
                if let Some(mesh) = self.mesh(&p.prim, None)? {
                    self.hold(p.node, mesh, &p.prim);
                }
                continue;
            }
            let own = tokens(self.value(&p.prim.attribute("skel:joints")));
            let names = if own.is_empty() { &skel.joints } else { &own };
            let geom_bind = match self.value(&p.prim.attribute("primvars:skel:geomBindTransform")) {
                Some(sdf::Value::Matrix4d(m)) => m,
                _ => gf::Matrix4d::IDENTITY,
            };
            let mut joints = Vec::with_capacity(names.len());
            let mut ibms = Vec::with_capacity(names.len() * 16);
            for name in names {
                // A joint the skeleton doesn't have takes its first.
                let j = skel.joints.iter().position(|s| s == name).unwrap_or(0);
                joints.push(skel.nodes[j]);
                // Mesh space to skeleton space at bind time, then to the joint's: row vectors, whose layout is
                // glTF's column-major one for column vectors.
                let ibm = geom_bind * skel.binds[j].inverse().unwrap_or(gf::Matrix4d::IDENTITY);
                ibms.extend(ibm.0.map(f32_of));
            }
            let acc = self.b.floats(ibms, "MAT4", false);
            let name = self.nodes[p.node]["name"].clone();
            self.skins
                .push(json!({ "joints": joints, "inverseBindMatrices": acc, "skeleton": skel.nodes[0] }));
            let Some(n) = self.push_node(json!({ "name": name, "skin": self.skins.len() - 1 })) else {
                continue;
            };
            self.hold(n, p.mesh, &p.prim);
            self.skinned_roots.push(n);
        }
        Ok(())
    }

    /// Blend shape weights at the rest time, on the nodes drawing meshes with blend shapes.
    fn pose_weights(&mut self) {
        for (node, prim) in std::mem::take(&mut self.morphed) {
            if let Some(w) = self.weights_at(node, &prim, self.time) {
                self.nodes[node]["weights"] = json!(w);
            }
            self.morphed.push((node, prim));
        }
    }

    /// The `SkelAnimation` that sets a surface's blend shape weights: its skeleton's.
    fn blend_animation(&self, prim: &usd::Prim) -> Option<usd::Prim> {
        let skel = self.stage.prim(inherited_target(prim, "skel:skeleton")?).ok()?;
        self.stage.prim(inherited_target(&skel, "skel:animationSource")?).ok()
    }

    /// The weights of the mesh on `node` at `time`, in its targets' order, from the surface's animation.
    fn weights_at(&self, node: usize, prim: &usd::Prim, time: usd::TimeCode) -> Option<Vec<f32>> {
        let anim = self.blend_animation(prim)?;
        let mesh = usize::try_from(self.nodes[node]["mesh"].as_u64()?).ok()?;
        let names = self.meshes[mesh].pointer("/extras/targetNames")?.as_array()?;
        let animated = tokens(anim.attribute("blendShapes").get_at::<sdf::Value>(time).ok().flatten());
        let w = floats(anim.attribute("blendShapeWeights").get_at::<sdf::Value>(time).ok()??)?;
        Some(
            names
                .iter()
                .map(|n| {
                    animated
                        .iter()
                        .position(|a| Some(a.as_str()) == n.as_str())
                        .and_then(|k| w.get(k).copied())
                        .unwrap_or(0.0)
                })
                .collect(),
        )
    }

    /// One animation: animated transforms, skeletal animations and blend shape weights.
    fn animations(&mut self, tcps: f64) -> Vec<Animation> {
        let mut channels = Vec::new();
        for (node, prim) in std::mem::take(&mut self.animated) {
            self.transform_channels(node, &prim, &mut channels);
        }
        let mut skels: Vec<&Skel> = self.skeletons.values().filter(|s| s.animation.is_some()).collect();
        skels.sort_by_key(|s| s.nodes.first().copied());
        for skel in skels {
            let Some(a) = skel.animation.as_ref().and_then(|a| self.stage.prim(a.clone()).ok()) else {
                continue;
            };
            let attrs = ["translations", "rotations", "scales"].map(|n| a.attribute(n));
            let times = timeline(usd::Attribute::unioned_time_samples(&attrs).unwrap_or_default());
            if times.len() < 2 {
                continue;
            }
            let path = skel.animation.clone().unwrap_or_else(|| a.path().clone());
            let samples: Vec<SkelSample> = times
                .iter()
                .filter_map(|&t| self.skel_sample(&path, usd::TimeCode::new(t)))
                .collect();
            if samples.len() != times.len() {
                continue;
            }
            for (j, name) in skel.joints.iter().enumerate() {
                let Some(k) = samples[0].joints.iter().position(|n| n == name) else {
                    continue;
                };
                let rest = decompose(&skel.rest[j]);
                let mut t = Vec::with_capacity(times.len() * 3);
                let mut r: Vec<[f64; 4]> = Vec::with_capacity(times.len());
                let mut s = Vec::with_capacity(times.len() * 3);
                for sample in &samples {
                    let (tv, mut rv, sv) = sample.trs(k, &rest);
                    if let Some(prev) = r.last()
                        && (0..4).map(|c| prev[c] * rv[c]).sum::<f64>() < 0.0
                    {
                        rv = rv.map(|c| -c);
                    }
                    t.extend(tv);
                    r.push(rv);
                    s.extend(sv);
                }
                let node = skel.nodes[j];
                for (path, values) in [("translation", t), ("rotation", r.into_flattened()), ("scale", s)] {
                    let width = if path == "rotation" { 4 } else { 3 };
                    let first = &values[..width];
                    if values
                        .chunks_exact(width)
                        .all(|v| v.iter().zip(first).all(|(a, b)| (a - b).abs() < 1e-7))
                    {
                        continue;
                    }
                    channels.push(Channel {
                        node,
                        path,
                        times: times.clone(),
                        values,
                    });
                }
            }
        }
        for (node, prim) in self.morphed.clone() {
            self.weight_channel(node, &prim, &mut channels);
        }
        if channels.is_empty() {
            return Vec::new();
        }
        let shift = channels.iter().map(|c| c.times[0]).fold(0.0, f64::min);
        let mut inputs: HashMap<Vec<u64>, usize> = HashMap::new();
        let (mut samplers, mut out) = (Vec::new(), Vec::new());
        for c in channels {
            let key: Vec<u64> = c.times.iter().map(|t| t.to_bits()).collect();
            let input = if let Some(&i) = inputs.get(&key) {
                i
            } else {
                let secs = c.times.iter().map(|&t| f32_of((t - shift) / tcps)).collect();
                let i = self.b.floats(secs, "SCALAR", true);
                inputs.insert(key, i);
                i
            };
            let kind = match c.path {
                "translation" | "scale" => "VEC3",
                "rotation" => "VEC4",
                _ => "SCALAR",
            };
            let output = self.b.floats(c.values.into_iter().map(f32_of).collect(), kind, false);
            samplers.push(Sampler {
                input,
                output,
                interpolation: "LINEAR",
            });
            out.push(document::Channel {
                sampler: samplers.len() - 1,
                node: c.node,
                path: c.path,
            });
        }
        vec![Animation {
            channels: out,
            samplers,
            name: None,
            extras: None,
        }]
    }

    /// Translation, rotation and scale channels for a node whose prim's transform is animated: its local transform
    /// at each sample, taken apart.
    fn transform_channels(&mut self, node: usize, prim: &usd::Prim, channels: &mut Vec<Channel>) {
        const PROBES: usize = 8;
        let Ok(q) = XformQuery::for_prim(prim) else { return };
        let mut times = timeline(q.time_samples().unwrap_or_default());
        if times.len() < 2 {
            return;
        }
        let at = |t: f64| {
            q.local_transformation(usd::TimeCode::new(t))
                .map_or(IDENTITY, |m| m4(&m))
        };
        // Cut long turns into short steps; probing within each span catches a full turn.
        let mut fine = Vec::with_capacity(times.len());
        for w in times.windows(2) {
            fine.push(w[0]);
            #[allow(clippy::cast_precision_loss, reason = "a small step count")]
            let probes: Vec<[f64; 4]> = (0..=PROBES)
                .map(|k| decompose(&at(w[0] + (w[1] - w[0]) * k as f64 / PROBES as f64)).r)
                .collect();
            let total: f64 = probes.windows(2).map(|p| turn(&p[0], &p[1])).sum();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a small step count"
            )]
            let n = ((total / MAX_STEP_DEGREES).ceil() as usize).min(360);
            for k in 1..n {
                #[allow(clippy::cast_precision_loss, reason = "a small step count")]
                fine.push(w[0] + (w[1] - w[0]) * k as f64 / n as f64);
            }
        }
        fine.extend(times.last());
        times = fine;
        let mut trs: Vec<Trs> = times.iter().map(|&t| decompose(&at(t))).collect();
        // Keep each quaternion on the same side as the last, so slerp takes the short way.
        for i in 1..trs.len() {
            let prev = trs[i - 1].r;
            let r = &mut trs[i].r;
            if (0..4).map(|k| prev[k] * r[k]).sum::<f64>() < 0.0 {
                *r = r.map(|c| -c);
            }
        }
        let rest = decompose(&q.local_transformation(self.time).map_or(IDENTITY, |m| m4(&m)));
        let paths: [(&'static str, Vec<f64>, Vec<f64>); 3] = [
            ("translation", trs.iter().flat_map(|x| x.t).collect(), rest.t.to_vec()),
            ("rotation", trs.iter().flat_map(|x| x.r).collect(), rest.r.to_vec()),
            ("scale", trs.iter().flat_map(|x| x.s).collect(), rest.s.to_vec()),
        ];
        for (path, values, rest) in paths {
            let still = values
                .chunks_exact(rest.len())
                .all(|v| v.iter().zip(&rest).all(|(a, b)| (a - b).abs() < 1e-7));
            if !still {
                channels.push(Channel {
                    node,
                    path,
                    times: times.clone(),
                    values,
                });
            }
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

    /// A weights channel for a node whose mesh's blend shape weights are animated.
    fn weight_channel(&self, node: usize, prim: &usd::Prim, channels: &mut Vec<Channel>) {
        let Some(anim) = self.blend_animation(prim) else { return };
        let times = timeline(
            anim.attribute("blendShapeWeights")
                .time_sample_times()
                .unwrap_or_default(),
        );
        if times.len() < 2 {
            return;
        }
        let mut values = Vec::new();
        for &t in &times {
            let Some(w) = self.weights_at(node, prim, usd::TimeCode::new(t)) else {
                return;
            };
            values.extend(w.into_iter().map(f64::from));
        }
        channels.push(Channel {
            node,
            path: "weights",
            times,
            values,
        });
    }
}

/// The first target of relationship `name` on `prim` or the nearest ancestor that has one: `UsdSkel`'s bindings are
/// inherited.
fn inherited_target(prim: &usd::Prim, name: &str) -> Option<sdf::Path> {
    let mut at = Some(prim.clone());
    while let Some(p) = at {
        if let Some(t) = p.relationship(name).targets().ok().and_then(|t| t.into_iter().next()) {
            return Some(t);
        }
        at = p.parent();
    }
    None
}

/// How a primvar's values spread over the mesh.
#[derive(Clone, Copy, PartialEq)]
enum Spread {
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
}

impl Spread {
    fn of(token: Option<&str>) -> Self {
        match token {
            Some("constant") => Self::Constant,
            Some("uniform") => Self::Uniform,
            Some("faceVarying") => Self::FaceVarying,
            _ => Self::Vertex,
        }
    }

    /// The value index for one face corner: `face` is the face, `point` the point it uses, `corner` its place
    /// among all the face corners.
    fn index(self, face: usize, point: usize, corner: usize) -> usize {
        match self {
            Self::Constant => 0,
            Self::Uniform => face,
            Self::Vertex => point,
            Self::FaceVarying => corner,
        }
    }
}

/// A primvar's values with its indices and interpolation.
#[derive(Clone)]
struct Primvar<T> {
    values: Vec<T>,
    indices: Option<Vec<i32>>,
    spread: Spread,
}

impl<T: Copy> Primvar<T> {
    fn at(&self, face: usize, point: usize, corner: usize) -> Option<T> {
        let i = self.spread.index(face, point, corner);
        let i = match &self.indices {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        self.values.get(i).copied()
    }

    /// One value for the whole surface.
    fn constant(&self) -> bool {
        self.spread == Spread::Constant || (self.values.len() == 1 && self.indices.is_none())
    }
}

/// The basis of a cubic curve.
#[derive(Clone, Copy, PartialEq)]
enum Basis {
    Bezier,
    Bspline,
    CatmullRom,
    /// A segment between two points, each leaving along its tangent.
    Hermite,
}

impl Basis {
    /// The weights of a segment's four control points at `t`. Each set sums to one, but for a Hermite segment's,
    /// whose four are its start, its start's tangent, its end and its end's tangent: its points' sum to one.
    fn weights(self, t: f32) -> [f32; 4] {
        let (t2, t3) = (t * t, t * t * t);
        let u = 1.0 - t;
        match self {
            Basis::Bezier => [u * u * u, 3.0 * t * u * u, 3.0 * t2 * u, t3],
            Basis::Bspline => [
                u * u * u / 6.0,
                (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                t3 / 6.0,
            ],
            Basis::CatmullRom => [
                0.5 * (-t3 + 2.0 * t2 - t),
                1.5 * t3 - 2.5 * t2 + 1.0,
                0.5 * (-3.0 * t3 + 4.0 * t2 + t),
                0.5 * (t3 - t2),
            ],
            Basis::Hermite => [
                2.0 * t3 - 3.0 * t2 + 1.0,
                t3 - 2.0 * t2 + t,
                -2.0 * t3 + 3.0 * t2,
                t3 - t2,
            ],
        }
    }
}

/// A point drawn on a curve, or a point of a point cloud: a weighted sum of control points.
struct Sample {
    curve: usize,
    /// Control points and their weights, the first `n`.
    terms: [(usize, f32); 8],
    n: usize,
    /// The segment ends it lies between, as corners for per-segment-end values, and how far along it is.
    ends: (usize, usize, f32),
    /// Whether it starts a curve.
    starts: bool,
    /// Whether the curve closes from it back to its start.
    closes: bool,
    /// Where tangents start among the control points: they carry no primvar values.
    tangents: usize,
}

impl Sample {
    /// Control point `point`, at corner `corner`, of curve `curve`.
    fn at(curve: usize, point: usize, corner: usize) -> Self {
        let mut terms = [(0, 0.0); 8];
        terms[0] = (point, 1.0);
        Sample {
            curve,
            terms,
            n: 1,
            ends: (corner, corner, 0.0),
            starts: false,
            closes: false,
            tangents: usize::MAX,
        }
    }

    /// Adds `w` of control point `i`.
    fn add(&mut self, i: usize, w: f32) {
        if let Some(slot) = self.terms[..self.n].iter_mut().find(|(k, _)| *k == i) {
            slot.1 += w;
        } else if let Some(slot) = self.terms.get_mut(self.n) {
            *slot = (i, w);
            self.n += 1;
        }
    }

    /// The weighted sum of what `get` gives for each control point.
    fn eval<const N: usize>(&self, get: impl Fn(usize) -> Option<[f32; N]>) -> Option<[f32; N]> {
        let mut out = [0.0; N];
        for &(i, w) in &self.terms[..self.n] {
            let v = get(i)?;
            for (o, x) in out.iter_mut().zip(v) {
                *o += w * x;
            }
        }
        Some(out)
    }
}

/// A primvar's value at a sample: per curve, per control point (weighted as the point is), or per segment end
/// (between the two ends).
fn along<T: Copy, const N: usize>(pv: &Primvar<T>, s: &Sample, to: impl Fn(T) -> [f32; N]) -> Option<[f32; N]> {
    match pv.spread {
        Spread::FaceVarying => {
            let (a, b, t) = s.ends;
            let (a, b) = (to(pv.at(s.curve, 0, a)?), to(pv.at(s.curve, 0, b)?));
            Some(std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t))
        }
        // A tangent adds nothing: what's left is the points' share, which sums to one.
        _ => s.eval(|i| {
            if i >= s.tangents {
                Some([0.0; N])
            } else {
                pv.at(s.curve, i, 0).map(&to)
            }
        }),
    }
}

/// Emits the points drawn on curves of `counts` control points each (of `points` in all), in order: linear
/// curves' control points, cubic ones' sampled at [`CURVE_STEPS`] a segment while `budget` lasts and through their
/// control points after. A cubic curve with too few points for its basis and wrap is drawn through them; the
/// curves past the points are left out. Hermite curves' tangents follow the points, one each. Returns the points
/// sampled on cubic curves.
fn curve_samples(
    counts: &[i32],
    points: usize,
    cubic: Option<Basis>,
    wrap: Option<&str>,
    budget: usize,
    emit: &mut dyn FnMut(&Sample),
) -> usize {
    let periodic = wrap == Some("periodic");
    // Pinned curves meet their end points: B-spline and Catmull-Rom curves get a phantom point past each end.
    let pinned = wrap == Some("pinned") && cubic.is_some_and(|b| b != Basis::Bezier);
    let steps = usize::from(CURVE_STEPS);
    let mut sampled = 0usize;
    let (mut base, mut corners) = (0usize, 0usize);
    for (curve, &n) in counts.iter().enumerate() {
        let Ok(n) = usize::try_from(n) else { break };
        let Some(end) = base.checked_add(n).filter(|&e| e <= points) else {
            break;
        };
        let segments = match cubic {
            None => 0,
            Some(Basis::Hermite) => n.saturating_sub(1),
            Some(Basis::Bezier) if periodic => {
                if n >= 3 && n % 3 == 0 {
                    n / 3
                } else {
                    0
                }
            }
            Some(Basis::Bezier) => {
                if n >= 4 && (n - 4) % 3 == 0 {
                    (n - 1) / 3
                } else {
                    0
                }
            }
            Some(_) if periodic => {
                if n >= 3 {
                    n
                } else {
                    0
                }
            }
            Some(_) if pinned => n.saturating_sub(1),
            Some(_) => n.saturating_sub(3),
        };
        let want = segments.saturating_mul(steps).saturating_add(1);
        // Corners for per-segment-end values: one at each segment's end, the last the first when periodic, or one
        // a control point on linear curves.
        let span = match cubic {
            Some(_) if segments > 0 && periodic => segments,
            Some(_) if segments > 0 => segments + 1,
            _ => n,
        };
        match cubic {
            Some(basis) if segments > 0 && sampled.saturating_add(want) <= budget => {
                sampled += want;
                let stride = if basis == Basis::Bezier { 3 } else { 1 };
                for seg in 0..segments {
                    let last = if seg + 1 == segments { steps } else { steps - 1 };
                    for j in 0..=last {
                        let t = f32::from(u16::try_from(j).unwrap_or(0)) / f32::from(CURVE_STEPS);
                        let mut s = Sample::at(curve, 0, 0);
                        s.n = 0;
                        for (k, w) in basis.weights(t).into_iter().enumerate() {
                            let q = seg * stride + k;
                            if basis == Basis::Hermite {
                                // A point, then its tangent, then the next point and its tangent.
                                let at = base + seg + k / 2;
                                s.add(if k % 2 == 0 { at } else { points + at }, w);
                                s.tangents = points;
                            } else if pinned && q == 0 {
                                s.add(base, 2.0 * w);
                                s.add(base + 1, -w);
                            } else if pinned && q == n + 1 {
                                s.add(base + n - 1, 2.0 * w);
                                s.add(base + n - 2, -w);
                            } else if pinned {
                                s.add(base + q - 1, w);
                            } else {
                                s.add(base + q % n, w);
                            }
                        }
                        s.ends = (corners + seg, corners + (seg + 1) % span, t);
                        s.starts = seg == 0 && j == 0;
                        emit(&s);
                    }
                }
            }
            _ => through(curve, base, n, (corners, span), periodic, emit),
        }
        corners += span;
        base = end;
    }
    sampled
}

/// Emits curve `curve`'s `n` control points from `base`, closing back to the first when `closes`. A curve drawn
/// through its control points takes the nearest of the `span` segment ends' values from corner `corners.0`.
fn through(
    curve: usize,
    base: usize,
    n: usize,
    (corners, span): (usize, usize),
    closes: bool,
    emit: &mut dyn FnMut(&Sample),
) {
    for i in 0..n {
        let mut s = Sample::at(curve, base + i, corners + (i * span / n).min(span.saturating_sub(1)));
        s.starts = i == 0;
        s.closes = closes && i + 1 == n;
        emit(&s);
    }
}

/// What a `NurbsCurves` prim draws its curves by, each per curve but the weights, per control point.
struct Nurbs {
    orders: Vec<i32>,
    /// Each curve's knots in turn, as many as its control points and its order.
    knots: Vec<f64>,
    /// Each curve's parameter range; without one, the whole of its knots'.
    ranges: Vec<[f64; 2]>,
    /// Every control point's weight; without them, or with one that isn't positive, curves aren't rational.
    weights: Vec<f64>,
}

/// Emits the points drawn on NURBS curves of `counts` control points each (of `points` in all), in order: each
/// knot span within the curve's range sampled at [`CURVE_STEPS`] while `budget` lasts. A curve past the budget,
/// of order past [`MAX_ORDER`], or whose knots or range don't make a curve is drawn through its control points;
/// the curves past the points are left out. Returns the points sampled.
fn nurbs_samples(counts: &[i32], points: usize, nurbs: &Nurbs, budget: usize, emit: &mut dyn FnMut(&Sample)) -> usize {
    let steps = usize::from(CURVE_STEPS);
    let mut sampled = 0usize;
    let (mut base, mut corners, mut knot_at) = (0usize, 0usize, 0usize);
    for (curve, &n) in counts.iter().enumerate() {
        let Ok(n) = usize::try_from(n) else { break };
        let Some(end) = base.checked_add(n).filter(|&e| e <= points) else {
            break;
        };
        let order = nurbs.orders.get(curve).and_then(|&k| usize::try_from(k).ok());
        // This curve's knots, and where the next one's start: past them all once one is missing.
        let knot_end = order.and_then(|k| knot_at.checked_add(n)?.checked_add(k));
        let knots = knot_end.and_then(|e| nurbs.knots.get(knot_at..e));
        knot_at = knot_end.unwrap_or(usize::MAX);
        let curve_of = order
            .zip(knots)
            .and_then(|(k, knots)| NurbsCurve::new(knots, k, n, nurbs.ranges.get(curve).copied()));
        let Some(c) = curve_of else {
            through(curve, base, n, (corners, n), false, emit);
            corners += n;
            base = end;
            continue;
        };
        // Per-segment-end values are one at each knot span's end, whether the range draws the span or not.
        let span = n - c.order + 2;
        let spans = (c.order - 1..n).filter(|&j| c.clipped(j).is_some()).count();
        let want = spans.saturating_mul(steps).saturating_add(1);
        if sampled.saturating_add(want) > budget {
            through(curve, base, n, (corners, span), false, emit);
            corners += span;
            base = end;
            continue;
        }
        sampled += want;
        let rational =
            nurbs.weights.len() == points && nurbs.weights[base..end].iter().all(|w| w.is_finite() && *w > 0.0);
        let weight = |i: usize| if rational { nurbs.weights[i] } else { 1.0 };
        let mut drawn = 0;
        for j in c.order - 1..n {
            let Some((a, b)) = c.clipped(j) else { continue };
            drawn += 1;
            let last = if drawn == spans { steps } else { steps - 1 };
            for step in 0..=last {
                let t = f64::from(u16::try_from(step).unwrap_or(0)) / f64::from(CURVE_STEPS);
                let u = a * (1.0 - t) + b * t;
                // The first control point of the span's `order`.
                let first = base + j + 1 - c.order;
                let mut terms = c.basis(j, u);
                for (r, x) in terms.iter_mut().take(c.order).enumerate() {
                    *x *= weight(first + r);
                }
                let total: f64 = terms.iter().sum();
                let mut s = Sample::at(curve, 0, 0);
                s.n = 0;
                for (r, x) in terms.iter().take(c.order).enumerate() {
                    s.add(first + r, f32_of(x / total));
                }
                let along = (u - c.knots[j]) / (c.knots[j + 1] - c.knots[j]);
                let at = corners + j + 1 - c.order;
                s.ends = (at, at + 1, f32_of(along));
                s.starts = drawn == 1 && step == 0;
                emit(&s);
            }
        }
        corners += span;
        base = end;
    }
    sampled
}

/// A NURBS curve its knots and range make: of order 2 to [`MAX_ORDER`], at least as many control points, knots
/// that never decrease, and a range of some length within them.
struct NurbsCurve<'a> {
    knots: &'a [f64],
    order: usize,
    /// The parameter range drawn.
    range: (f64, f64),
}

impl<'a> NurbsCurve<'a> {
    fn new(knots: &'a [f64], order: usize, n: usize, range: Option<[f64; 2]>) -> Option<Self> {
        if !(2..=MAX_ORDER).contains(&order)
            || n < order
            || knots.len() != n + order
            || !knots.iter().all(|k| k.is_finite())
            || knots.windows(2).any(|w| w[0] > w[1])
        {
            return None;
        }
        let (lo, hi) = (knots[order - 1], knots[n]);
        let range = match range {
            Some([a, b]) if a.is_finite() && b.is_finite() && a < b => (a.max(lo), b.min(hi)),
            _ => (lo, hi),
        };
        (range.0 < range.1).then_some(NurbsCurve { knots, order, range })
    }

    /// The part of knot span `j` (from knot `j` to the next) the range draws, if any.
    fn clipped(&self, j: usize) -> Option<(f64, f64)> {
        let (a, b) = (self.knots[j].max(self.range.0), self.knots[j + 1].min(self.range.1));
        (a < b).then_some((a, b))
    }

    /// The weights at `u` in knot span `j` of the `order` control points from `j + 1 - order`, by Cox and de Boor's
    /// recurrence.
    fn basis(&self, j: usize, u: f64) -> [f64; MAX_ORDER] {
        let k = self.knots;
        let mut n = [0.0; MAX_ORDER];
        let (mut left, mut right) = ([0.0; MAX_ORDER], [0.0; MAX_ORDER]);
        n[0] = 1.0;
        for d in 1..self.order {
            left[d] = u - k[j + 1 - d];
            right[d] = k[j + d] - u;
            let mut saved = 0.0;
            for r in 0..d {
                let gap = right[r + 1] + left[d - r];
                let x = if gap == 0.0 { 0.0 } else { n[r] / gap };
                n[r] = saved + right[r + 1] * x;
                saved = left[d - r] * x;
            }
            n[d] = saved;
        }
        n
    }
}

/// One material's part of a mesh, with the point each of its vertices came from.
struct Part_ {
    prim: Prim,
    points: Vec<usize>,
    /// Its texture coordinate sets in `TEXCOORD_n` order, as indices into the shape's.
    uvs: Vec<usize>,
    /// Its color sets in `COLOR_n` order.
    colors: Vec<Paint>,
}

/// Where a primitive's color set comes from.
#[derive(Clone, Copy, PartialEq)]
enum Paint {
    /// `displayColor` and `displayOpacity`.
    Display,
    /// One of the shape's color sets.
    Set(usize),
    /// What the material's primvar reader reads where the surface lacks the primvar.
    Fallback([f32; 4]),
}

/// What a material reads from the surface it's on.
#[derive(Clone, Default)]
struct Reads {
    /// Its texture coordinate sets by primvar name, in `TEXCOORD_n` order: the color texture's first, empty when
    /// it names none. Empty when no texture names a set.
    uvs: Vec<String>,
    /// The color primvar its base color reads, and what it reads where the surface lacks it.
    color: Option<(String, [f32; 4])>,
}

/// A blend shape: its name, and its offsets and normal offsets per point.
struct Target {
    name: String,
    offsets: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
}

/// A primitive's texture coordinate sets, as indices into the shape's in `TEXCOORD_n` order, and its color sets in
/// `COLOR_n` order, for a material that reads `reads`. A set the material names and the surface lacks reads the
/// first, so the material's set numbers hold; the sets no texture reads follow. `display` is whether the surface
/// has a display color, `vertex_colors` whether it varies, and `bound` whether the material is one bound to it
/// rather than made from its display color.
fn layout(reads: &Reads, shape: &Shape, display: bool, vertex_colors: bool, bound: bool) -> (Vec<usize>, Vec<Paint>) {
    let find = |name: &str| shape.uvs.iter().position(|(n, _)| n == name);
    let mut uvs = Vec::with_capacity(shape.uvs.len() + reads.uvs.len());
    if !shape.uvs.is_empty() {
        let first = reads.uvs.first().and_then(|n| find(n)).unwrap_or(0);
        uvs.push(first);
        uvs.extend(reads.uvs.iter().skip(1).map(|n| find(n).unwrap_or(first)));
        for k in 0..shape.uvs.len() {
            if !uvs.contains(&k) {
                uvs.push(k);
            }
        }
    }
    let mut colors = Vec::with_capacity(shape.colors.len() + 1);
    match &reads.color {
        Some((name, _)) if name == "displayColor" && display => colors.push(Paint::Display),
        Some((name, fallback)) => colors.push(
            shape
                .colors
                .iter()
                .position(|(n, _)| n == name)
                .map_or(Paint::Fallback(*fallback), Paint::Set),
        ),
        None => {}
    }
    // glTF multiplies `COLOR_0` into the base color. A material that reads no color set must not be tinted by
    // one, so white goes first and the sets follow as `COLOR_1` on; only display colors on a surface with no
    // material of its own are its color.
    if colors.is_empty() && (bound || !vertex_colors) && (vertex_colors || !shape.colors.is_empty()) {
        colors.push(Paint::Fallback([1.0; 4]));
    }
    if vertex_colors && !colors.contains(&Paint::Display) {
        colors.push(Paint::Display);
    }
    for k in 0..shape.colors.len() {
        if !colors.contains(&Paint::Set(k)) {
            colors.push(Paint::Set(k));
        }
    }
    (uvs, colors)
}

/// `n` empty sets, slot 0 and those to fill; none for one set, which is slot 0.
fn empty_sets<T: Clone>(n: usize) -> Vec<Vec<T>> {
    if n > 1 { vec![Vec::new(); n] } else { Vec::new() }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Reader<'_> {
    fn primvar<T>(
        &self,
        api: &PrimvarsAPI,
        name: &str,
        decode: impl Fn(sdf::Value) -> Option<Vec<T>>,
    ) -> Option<Primvar<T>> {
        self.read_primvar(&api.primvar(name).ok()?, decode)
    }

    /// A primvar's values, indices and interpolation.
    fn read_primvar<T>(&self, pv: &geom::Primvar, decode: impl Fn(sdf::Value) -> Option<Vec<T>>) -> Option<Primvar<T>> {
        let values = decode(pv.attribute().get_at::<sdf::Value>(self.time).ok()??)?;
        let indices = pv.indices(self.time).ok().flatten().filter(|i| !i.is_empty());
        let spread = pv
            .interpolation()
            .ok()
            .map_or(Spread::Vertex, |i| Spread::of(Some(i.as_token())));
        Some(Primvar {
            values,
            indices,
            spread,
        })
    }

    fn value(&self, attr: &usd::Attribute) -> Option<sdf::Value> {
        attr.get_at::<sdf::Value>(self.time).ok().flatten()
    }

    /// A number attribute, double or float, or its fallback.
    fn number(&self, attr: &usd::Attribute, fallback: f64) -> f32 {
        f32_of(match self.value(attr) {
            Some(sdf::Value::Double(v)) => v,
            Some(sdf::Value::Float(v)) => f64::from(v),
            Some(sdf::Value::Half(v)) => f64::from(v.to_f32()),
            Some(sdf::Value::Int(v)) => f64::from(v),
            _ => fallback,
        })
    }

    fn token(&self, attr: &usd::Attribute) -> Option<String> {
        attr.get_at::<tf::Token>(self.time)
            .ok()
            .flatten()
            .map(|t| t.as_str().to_owned())
    }

    /// Visible, and drawn for rendering: guides and proxies stand in for the real thing.
    fn shown(&self, prim: &usd::Prim) -> bool {
        let img = geom::Imageable::from_prim_unchecked(prim.clone());
        let visible = !matches!(img.compute_visibility(self.time), Ok(geom::Visibility::Invisible));
        let purpose = img.compute_purpose().unwrap_or_default();
        visible && !matches!(purpose, geom::Purpose::Guide | geom::Purpose::Proxy)
    }

    /// The prim's surface as polygons in its own space: a mesh's, or an implicit surface's tessellation.
    fn shape(&self, prim: &usd::Prim) -> Option<Shape> {
        let t = |attr: &usd::Attribute| self.token(attr);
        match kind(prim)?.as_str() {
            "Mesh" => self.mesh_shape(&geom::Mesh::from_prim_unchecked(prim.clone())),
            "Cube" => {
                let s = self.number(&geom::Cube::from_prim_unchecked(prim.clone()).size_attr(), 2.0);
                Some(Tess::cube(s))
            }
            "Sphere" => {
                let r = self.number(&geom::Sphere::from_prim_unchecked(prim.clone()).radius_attr(), 1.0);
                Some(Tess::lathe(&Tess::sphere_rows(r, r, 0.0), None))
            }
            "Cylinder" | "Cylinder_1" => {
                let c = geom::Cylinder::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 1.0);
                let h = self.number(&c.height_attr(), 2.0);
                let (bottom, top) = if kind(prim)?.as_str() == "Cylinder_1" {
                    (
                        self.number(&prim.attribute("radiusBottom"), f64::from(r)),
                        self.number(&prim.attribute("radiusTop"), f64::from(r)),
                    )
                } else {
                    (r, r)
                };
                Some(Tess::lathe(
                    &Tess::frustum_rows(bottom, top, h),
                    t(&c.axis_attr()).as_deref(),
                ))
            }
            "Cone" => {
                let c = geom::Cone::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 1.0);
                let h = self.number(&c.height_attr(), 2.0);
                Some(Tess::lathe(
                    &Tess::frustum_rows(r, 0.0, h),
                    t(&c.axis_attr()).as_deref(),
                ))
            }
            "Capsule" | "Capsule_1" => {
                let c = geom::Capsule::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 0.5);
                let h = self.number(&c.height_attr(), 1.0);
                let (bottom, top) = if kind(prim)?.as_str() == "Capsule_1" {
                    (
                        self.number(&prim.attribute("radiusBottom"), f64::from(r)),
                        self.number(&prim.attribute("radiusTop"), f64::from(r)),
                    )
                } else {
                    (r, r)
                };
                let mut rows = Tess::sphere_rows(bottom, top, h);
                #[allow(clippy::float_cmp, reason = "identical rows, not near ones, make an empty band")]
                rows.dedup_by(|a, b| a == b);
                Some(Tess::lathe(&rows, t(&c.axis_attr()).as_deref()))
            }
            "Plane" => {
                let p = geom::Plane::from_prim_unchecked(prim.clone());
                let w = self.number(&p.width_attr(), 2.0);
                let l = self.number(&p.length_attr(), 2.0);
                Some(Tess::plane(w, l, t(&p.axis_attr()).as_deref()))
            }
            _ => None,
        }
    }

    fn mesh_shape(&self, mesh: &geom::Mesh) -> Option<Shape> {
        let points = vec3s(self.value(&mesh.points_attr())?)?;
        let counts = ints(self.value(&mesh.face_vertex_counts_attr()));
        let corners = ints(self.value(&mesh.face_vertex_indices_attr()));
        if points.is_empty() || counts.is_empty() {
            return None;
        }
        let holes = ints(self.value(&mesh.hole_indices_attr()));
        let left_handed = self.token(&mesh.orientation_attr()).as_deref() == Some("leftHanded");
        let api = PrimvarsAPI::from_prim_unchecked(mesh.prim().clone());
        let (uvs, colors) = self.surface_sets(&api);
        let tangents = ["tangents", "tangent"]
            .iter()
            .find_map(|n| self.primvar(&api, n, tangent4s));
        let bitangents = ["bitangents", "binormals", "bitangent", "binormal"]
            .iter()
            .find_map(|n| self.primvar(&api, n, vec3s));
        let normals = self.primvar(&api, "normals", vec3s).or_else(|| {
            let attr = mesh.normals_attr();
            let values = vec3s(self.value(&attr)?)?;
            let spread = attr.get_metadata::<sdf::Value>("interpolation").ok().flatten();
            Some(Primvar {
                values,
                indices: None,
                spread: Spread::of(spread.as_ref().and_then(sdf::Value::as_str)),
            })
        });
        // No normals: a subdivision surface is smooth, a polygon mesh faceted.
        let smooth = normals.is_none() && self.token(&mesh.subdivision_scheme_attr()).as_deref() != Some("none");
        let mut hole = vec![false; counts.len()];
        for h in holes {
            if let Some(slot) = usize::try_from(h).ok().and_then(|h| hole.get_mut(h)) {
                *slot = true;
            }
        }
        Some(Shape {
            points,
            counts,
            corners,
            hole,
            normals,
            uvs,
            colors,
            tangents,
            bitangents,
            left_handed,
            smooth,
        })
    }

    /// A mesh's texture coordinate sets and its color sets other than the display color, by primvar name, at most
    /// [`MAX_SETS`] of each: the usual names for texture coordinates first, then the rest by name. Colors the same
    /// over the whole surface aren't sets.
    #[allow(clippy::type_complexity, reason = "two lists of named primvars")]
    fn surface_sets(&self, api: &PrimvarsAPI) -> (Vec<(String, Primvar<[f32; 2]>)>, Vec<(String, Primvar<[f32; 4]>)>) {
        let mut uv = Vec::new();
        let mut color = Vec::new();
        for pv in api.authored_primvars().unwrap_or_default() {
            let Ok(Some(ty)) = pv.attribute().type_name() else {
                continue;
            };
            match ty.as_str() {
                "texCoord2f[]" | "texCoord2d[]" | "texCoord2h[]" | "float2[]" | "double2[]" | "half2[]" => uv.push(pv),
                "color3f[]" | "color3d[]" | "color3h[]" | "color4f[]" | "color4d[]" | "color4h[]"
                    if pv.primvar_name() != "displayColor" =>
                {
                    color.push(pv);
                }
                _ => {}
            }
        }
        uv.sort_by_cached_key(|pv| {
            let name = pv.primvar_name();
            (
                UV_NAMES.iter().position(|n| *n == name).unwrap_or(UV_NAMES.len()),
                name.to_owned(),
            )
        });
        color.sort_by(|a, b| a.primvar_name().cmp(b.primvar_name()));
        let uvs = uv
            .iter()
            .filter_map(|pv| Some((pv.primvar_name().to_owned(), self.read_primvar(pv, vec2s)?)))
            .take(MAX_SETS)
            .collect();
        let colors = color
            .iter()
            .filter_map(|pv| Some((pv.primvar_name().to_owned(), self.read_primvar(pv, vec4s)?)))
            .filter(|(_, c)| !c.constant())
            .take(MAX_SETS)
            .collect();
        (uvs, colors)
    }

    /// The mesh for a surface, made once per prototype, material binding and skinning; `None` when it has no
    /// faces. `joints` is how many joints a skinned surface's influences name. The geometry is read only when the
    /// mesh is made, so each instance of a prototype costs a lookup.
    fn mesh(&mut self, prim: &usd::Prim, joints: Option<usize>) -> Result<Option<usize>> {
        let source = match prim.prim_in_prototype() {
            Ok(Some(p)) if prim.is_instance_proxy().unwrap_or(false) => p.path().as_str().to_owned(),
            _ => prim.path().as_str().to_owned(),
        };
        let binding = bound(prim).map(|b| b.as_str().to_owned()).unwrap_or_default();
        let key = format!("{source}|{binding}|{}", joints.is_some());
        if let Some(&m) = self.mesh_of.get(&key) {
            return Ok(m);
        }
        let m = match self.shape(prim) {
            Some(shape) => self.build_mesh(prim, &shape, joints)?,
            None => None,
        };
        self.mesh_of.insert(key, m);
        Ok(m)
    }

    #[allow(clippy::too_many_lines, reason = "one pass over the faces, filling every attribute")]
    fn build_mesh(&mut self, prim: &usd::Prim, shape: &Shape, joints: Option<usize>) -> Result<Option<usize>> {
        let points = &shape.points;
        let flip = shape.left_handed;
        let (counts, corners) = (&shape.counts, &shape.corners);

        // The faces to draw, by their first corner.
        let mut faces = Vec::with_capacity(counts.len());
        let mut start = 0usize;
        let mut triangles = 0usize;
        for (face, &n) in counts.iter().enumerate() {
            let n = usize::try_from(n).unwrap_or(0);
            let end = start.saturating_add(n);
            let ok = n >= 3
                && end <= corners.len()
                && !shape.hole.get(face).copied().unwrap_or(false)
                && corners[start..end]
                    .iter()
                    .all(|&c| usize::try_from(c).is_ok_and(|c| c < points.len()));
            if ok {
                faces.push((face, start, n));
                triangles += n - 2;
                ensure!(
                    triangles <= self.limit,
                    LimitExceeded,
                    "too many triangles (over {})",
                    self.limit
                );
            }
            start = end;
        }
        if faces.is_empty() {
            return Ok(None);
        }
        let point_normals = if shape.smooth {
            smooth_normals(points, corners, &faces, flip)
        } else {
            Vec::new()
        };

        let api = PrimvarsAPI::from_prim_unchecked(prim.clone());
        let display = self.primvar(&api, "displayColor", vec3s);
        let opacity = self.primvar(&api, "displayOpacity", floats);
        let binding = bound(prim);
        // Display colors that vary over the surface are its vertex colors, whatever material is bound.
        let vertex_colors =
            display.as_ref().is_some_and(|d| !d.constant()) || opacity.as_ref().is_some_and(|o| !o.constant());
        let double = self.double_sided(prim);
        let own = match &binding {
            Some(path) => self.material(Some(path.clone()), GRAY, double),
            None if vertex_colors => self.material(None, [1.0; 4], double),
            None => {
                let [r, g, b] = display
                    .as_ref()
                    .and_then(|d| d.values.first().copied())
                    .unwrap_or([GRAY[0], GRAY[1], GRAY[2]]);
                let a = opacity.as_ref().and_then(|o| o.values.first().copied()).unwrap_or(1.0);
                self.material(None, [r, g, b, a.clamp(0.0, 1.0)], double)
            }
        };

        // Each face's material: a material-binding subset's, else the surface's own.
        let mut face_material = vec![own; counts.len()];
        if let Ok(children) = prim.children() {
            for child in children {
                let Ok(Some(subset)) = geom::Subset::get(self.stage, child.path().clone()) else {
                    continue;
                };
                let family = self.token(&subset.family_name_attr());
                if family.as_deref().is_some_and(|f| f != "materialBind") {
                    continue;
                }
                let material = match bound(&child) {
                    Some(path) => self.material(Some(path), GRAY, double),
                    None => own,
                };
                for i in ints(self.value(&subset.indices_attr())) {
                    if let Some(slot) = usize::try_from(i).ok().and_then(|i| face_material.get_mut(i)) {
                        *slot = material;
                    }
                }
            }
        }

        let mut parts: Vec<Part_> = Vec::new();
        let mut part_of: HashMap<usize, usize> = HashMap::new();
        let mut tris = Vec::new();
        let mut poly = Vec::new();
        for &(face, start, n) in &faces {
            poly.clear();
            #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
            poly.extend(corners[start..start + n].iter().map(|&c| points[c as usize]));
            let flat = (shape.normals.is_none() && !shape.smooth).then(|| polygon_normal(&poly, flip));
            let material = face_material[face];
            let next = parts.len();
            let i = *part_of.entry(material).or_insert(next);
            if i == next {
                let none = Reads::default();
                let reads = self.material_reads.get(material).unwrap_or(&none);
                let bound = binding.is_some() || material != own;
                let (uvs, colors) = layout(reads, shape, display.is_some(), vertex_colors, bound);
                parts.push(Part_ {
                    prim: Prim {
                        material,
                        uv_sets: empty_sets(uvs.len()),
                        color_sets: empty_sets(colors.len()),
                        ..Prim::default()
                    },
                    points: Vec::new(),
                    uvs,
                    colors,
                });
            }
            let part = &mut parts[i];
            let prim = &mut part.prim;
            let first = u32::try_from(prim.positions.len()).context(ErrorKind::LimitExceeded, "too many vertices")?;
            for (c, &corner) in corners.iter().enumerate().skip(start).take(n) {
                #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
                let point = corner as usize;
                prim.positions.push(points[point]);
                part.points.push(point);
                let normal = match (&shape.normals, flat) {
                    (Some(nv), _) => nv.at(face, point, c).map(|v| unit(v.map(f64::from), false)),
                    (None, Some(f)) => Some(f),
                    (None, None) => point_normals.get(point).copied(),
                };
                let normal = normal.unwrap_or([0.0, 1.0, 0.0]);
                prim.normals.push(normal);
                if let Some(tv) = &shape.tangents {
                    let t = tv.at(face, point, c).unwrap_or([1.0, 0.0, 0.0, 1.0]);
                    // Handedness as authored, else from the bitangent: glTF's is cross(normal, tangent) * w.
                    let w = if t[3] < 0.0 {
                        -1.0
                    } else if t[3] > 0.0 {
                        1.0
                    } else {
                        match shape.bitangents.as_ref().and_then(|b| b.at(face, point, c)) {
                            Some(b) if dot(cross(normal, [t[0], t[1], t[2]]), b) < 0.0 => -1.0,
                            _ => 1.0,
                        }
                    };
                    prim.tangents.push([t[0], t[1], t[2], w]);
                }
                for (slot, &k) in part.uvs.iter().enumerate() {
                    let [u, v] = shape.uvs[k].1.at(face, point, c).unwrap_or_default();
                    let uv = [u, 1.0 - v];
                    match slot {
                        0 => prim.uvs.push(uv),
                        _ => prim.uv_sets[slot].push(uv),
                    }
                }
                for (slot, &paint) in part.colors.iter().enumerate() {
                    let color = match paint {
                        Paint::Display => {
                            let [r, g, b] = display.as_ref().and_then(|d| d.at(face, point, c)).unwrap_or([1.0; 3]);
                            let a = opacity.as_ref().and_then(|o| o.at(face, point, c)).unwrap_or(1.0);
                            [r, g, b, a.clamp(0.0, 1.0)]
                        }
                        Paint::Set(k) => shape.colors[k].1.at(face, point, c).unwrap_or([1.0; 4]),
                        Paint::Fallback(f) => f,
                    };
                    match slot {
                        0 => prim.colors.push(color),
                        _ => prim.color_sets[slot].push(color),
                    }
                }
            }
            tris.clear();
            triangulate(&poly, &mut tris);
            for &[a, b, c] in &tris {
                let tri = if flip { [a, c, b] } else { [a, b, c] };
                prim.indices.extend(tri.map(|k| first + k));
            }
        }

        let influences = joints.and_then(|j| self.influences(prim, points.len(), j));
        let joint_sets = influences.as_ref().map_or(0, |inf| {
            inf.iter().map(|v| v.len().div_ceil(4)).max().unwrap_or(0).min(4)
        });
        let targets = if self.keep.morph_targets {
            self.blend_shapes(prim, points.len(), shape.normals.is_some())
        } else {
            Vec::new()
        };
        let target_normals = targets.iter().any(|t| t.normals.is_some());
        let mut primitives = Vec::new();
        let mut drawn = 0;
        for Part_ { prim: p, points, .. } in parts {
            let material = p.material;
            drawn += p.indices.len() / 3;
            let Some(mut json) = self.b.primitive(p) else { continue };
            json["material"] = json!(material);
            if let Some(inf) = &influences {
                for set in 0..joint_sets {
                    let mut js = Vec::with_capacity(points.len() * 4);
                    let mut ws = Vec::with_capacity(points.len() * 4);
                    for &pt in &points {
                        for k in set * 4..set * 4 + 4 {
                            let (j, w) = inf[pt].get(k).copied().unwrap_or((0, 0.0));
                            js.push(j);
                            ws.push(w);
                        }
                    }
                    let n = points.len();
                    let j = self.b.accessor(Part::u16(js), 5123, n, "VEC4", Some(34962));
                    let w = self.b.accessor(Part::f32(ws), 5126, n, "VEC4", Some(34962));
                    json["attributes"][format!("JOINTS_{set}")] = json!(j);
                    json["attributes"][format!("WEIGHTS_{set}")] = json!(w);
                }
            }
            if !targets.is_empty() {
                let t: Vec<Value> = targets
                    .iter()
                    .map(|t| {
                        let d = points.iter().map(|&pt| t.offsets[pt]).collect();
                        let mut target = json!({ "POSITION": self.b.positions(d) });
                        // Every target has normals when one does: a target without moves none.
                        if target_normals {
                            let d = match &t.normals {
                                Some(n) => points.iter().flat_map(|&pt| n[pt]).collect(),
                                None => vec![0.0; points.len() * 3],
                            };
                            let a = self.b.accessor(Part::f32(d), 5126, points.len(), "VEC3", Some(34962));
                            target["NORMAL"] = json!(a);
                        }
                        target
                    })
                    .collect();
                json["targets"] = json!(t);
            }
            primitives.push(json);
        }
        if primitives.is_empty() {
            return Ok(None);
        }
        let mut mesh = json!({ "name": leaf(prim.path().as_str()), "primitives": primitives });
        if !targets.is_empty() {
            mesh["weights"] = json!(vec![0.0; targets.len()]);
            mesh["extras"] = json!({ "targetNames": targets.iter().map(|t| t.name.as_str()).collect::<Vec<_>>() });
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(drawn);
        Ok(Some(self.meshes.len() - 1))
    }

    /// A skinned surface's joint influences per point: (joint, weight), heaviest first, up to 16, summing to one.
    /// A point no joint weighs follows the first joint.
    fn influences(&self, prim: &usd::Prim, points: usize, joints: usize) -> Option<Vec<Vec<(u16, f32)>>> {
        let ji = prim.attribute("primvars:skel:jointIndices");
        let index = ints(self.value(&ji));
        let weight = floats(self.value(&prim.attribute("primvars:skel:jointWeights"))?)?;
        if index.is_empty() || index.len() != weight.len() {
            return None;
        }
        let size = match ji.get_metadata::<sdf::Value>("elementSize") {
            Ok(Some(sdf::Value::Int(n))) if n > 0 => usize::try_from(n).unwrap_or(1),
            _ => 1,
        };
        let constant = matches!(
            ji.get_metadata::<sdf::Value>("interpolation"),
            Ok(Some(v)) if v.as_str() == Some("constant")
        );
        let mut out = Vec::with_capacity(points);
        for p in 0..points {
            let base = if constant { 0 } else { p * size };
            let mut inf: Vec<(u16, f32)> = (base..base + size)
                .filter_map(|k| {
                    let j = usize::try_from(*index.get(k)?).ok().filter(|&j| j < joints)?;
                    let w = *weight.get(k)?;
                    (w > 0.0 && w.is_finite()).then(|| Some((u16::try_from(j).ok()?, w)))?
                })
                .collect();
            inf.sort_by(|a, b| b.1.total_cmp(&a.1));
            inf.truncate(16);
            let sum: f32 = inf.iter().map(|x| x.1).sum();
            if sum > 0.0 {
                for x in &mut inf {
                    x.1 /= sum;
                }
            } else {
                inf = vec![(0, 1.0)];
            }
            out.push(inf);
        }
        Some(out)
    }

    /// A surface's blend shapes: each one's name, its offset per point, and with `normals` (the surface has
    /// authored normals), its normal offsets per point where it has them.
    fn blend_shapes(&self, prim: &usd::Prim, points: usize, normals: bool) -> Vec<Target> {
        let names = tokens(self.value(&prim.attribute("skel:blendShapes")));
        let targets = prim
            .relationship("skel:blendShapeTargets")
            .targets()
            .unwrap_or_default();
        names
            .into_iter()
            .zip(targets)
            .filter_map(|(name, path)| {
                let shape = self.stage.prim(path).ok()?;
                let at = ints(self.value(&shape.attribute("pointIndices")));
                let per_point = |values: &[[f32; 3]]| {
                    let mut d = vec![[0.0; 3]; points];
                    if at.is_empty() {
                        for (slot, o) in d.iter_mut().zip(values) {
                            *slot = *o;
                        }
                    } else {
                        for (&i, o) in at.iter().zip(values) {
                            if let Some(slot) = usize::try_from(i).ok().and_then(|i| d.get_mut(i)) {
                                *slot = *o;
                            }
                        }
                    }
                    d
                };
                let offsets = per_point(&vec3s(self.value(&shape.attribute("offsets"))?)?);
                let normals = normals
                    .then(|| vec3s(self.value(&shape.attribute("normalOffsets"))?))
                    .flatten()
                    .filter(|n| !n.is_empty())
                    .map(|n| per_point(&n));
                Some(Target { name, offsets, normals })
            })
            .collect()
    }

    /// The material at `bound`, made once per material prim and sidedness (a material on one- and two-sided
    /// surfaces is two); with none, a plain one in `display`.
    fn material(&mut self, bound: Option<sdf::Path>, display: [f32; 4], double: bool) -> usize {
        let key = match &bound {
            Some(p) => format!("{}|{double}", p.as_str()),
            None => format!("display:{display:?}|{double}"),
        };
        if let Some(&i) = self.material_of.get(&key) {
            return i;
        }
        let (mut material, reads) = match bound {
            // Without materials kept, the shader isn't read; the plain one goes in the prune.
            Some(path) if !self.keep.materials => (
                Full::from(Material::plain(leaf(path.as_str()), display)),
                Reads::default(),
            ),
            Some(path) => self.preview_surface(&path),
            None => (Full::from(Material::plain("default", display)), Reads::default()),
        };
        material.double_sided = double;
        let i = self.materials.len();
        self.materials.push(material);
        self.material_reads.push(reads);
        self.material_of.insert(key, i);
        i
    }

    /// A `UsdPreviewSurface` material, and what it reads from the surface it's on.
    #[allow(clippy::too_many_lines, reason = "one input after another")]
    fn preview_surface(&mut self, path: &sdf::Path) -> (Full, Reads) {
        let name = leaf(path.as_str()).to_owned();
        let Ok(Some(ps)) = shade::read_preview_surface(self.stage, path) else {
            return (Material::plain(&name, GRAY).into(), Reads::default());
        };
        let shader = self.stage.prim(ps.shader.as_str()).ok();
        let source = |input: &str| shader.as_ref().and_then(|s| self.texture_source(s, input));
        let (diffuse, normal, emissive, occlusion) = (
            source("diffuseColor"),
            source("normal"),
            source("emissiveColor"),
            source("occlusion"),
        );
        let (metallic_src, roughness_src, opacity_src) = (source("metallic"), source("roughness"), source("opacity"));
        let specular_src = source("specularColor");
        let specular_workflow = shader
            .as_ref()
            .and_then(|s| match self.value(&s.attribute("inputs:useSpecularWorkflow")) {
                Some(sdf::Value::Int(v)) => Some(v != 0),
                _ => None,
            })
            .unwrap_or(false);
        // A base color from a primvar reader is the surface's vertex colors.
        let color_read = shader
            .as_ref()
            .and_then(|s| self.connected(s, "inputs:diffuseColor"))
            .filter(|(r, _)| {
                self.token(&r.attribute("info:id"))
                    .is_some_and(|id| id.starts_with("UsdPrimvarReader"))
            })
            .and_then(|(r, _)| {
                let fallback = self
                    .value(&r.attribute("inputs:fallback"))
                    .and_then(vec4s)
                    .and_then(|v| v.first().copied())
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                Some((self.varname(&r)?, fallback))
            });
        // The texture coordinate sets the textures read, by primvar name: the color texture's first. Without a
        // color texture, the first set another texture names.
        let first = if ps.diffuse_color.texture().is_some() {
            diffuse.as_ref().and_then(|d| d.uv.clone())
        } else {
            [
                &normal,
                &emissive,
                &occlusion,
                &roughness_src,
                &metallic_src,
                &specular_src,
            ]
            .into_iter()
            .find_map(|s| s.as_ref().and_then(|s| s.uv.clone()))
        };
        let mut sets = vec![first.unwrap_or_default()];

        // UsdPreviewSurface colors are linear, as glTF's are. Unauthored inputs take the shader's fallbacks.
        let base_color_tex = self.tex(ps.diffuse_color.texture(), diffuse.as_ref(), &mut sets);
        let mut color = match ps.diffuse_color.value() {
            _ if color_read.is_some() => [1.0; 4],
            Some(c) => [c.x, c.y, c.z, 1.0],
            // A texture's scale tints it.
            None if ps.diffuse_color.texture().is_some() => diffuse.as_ref().map_or([1.0; 4], |d| d.scale),
            None => [0.18, 0.18, 0.18, 1.0],
        };
        let mut alpha = Alpha::Opaque;
        if let Some(&o) = ps.opacity.value() {
            color[3] = o.clamp(0.0, 1.0);
            if o < 1.0 {
                alpha = Alpha::Blend;
            }
        } else if let (Some(o), Some(d)) = (&opacity_src, &diffuse) {
            // Opacity from the color texture's alpha is glTF's own arrangement.
            if o.file == d.file && o.channel == "a" {
                alpha = Alpha::Blend;
            }
        }
        if let Some(&t) = ps.opacity_threshold.value().filter(|t| **t > 0.0) {
            alpha = Alpha::Mask(t);
        }
        let mut metallic = if specular_workflow {
            0.0
        } else {
            ps.metallic.value().copied().unwrap_or(0.0)
        };
        let mut roughness = ps.roughness.value().copied().unwrap_or(0.5);
        // glTF reads roughness from green and metalness from blue of one image. A gray roughness image reads the
        // same in every channel, so it serves where nothing is metal.
        let mut metallic_roughness_tex = None;
        if let Some(r) = roughness_src.as_ref()
            && let Some(file) = ps.roughness.texture()
        {
            let metal_too = r.channel == "g"
                && metallic_src
                    .as_ref()
                    .is_some_and(|m| m.channel == "b" && m.file == r.file);
            let alone = ps.metallic.texture().is_none();
            let usable = if r.channel == "g" {
                metal_too || alone
            } else {
                alone
                    && (specular_workflow || metallic <= 0.0)
                    && self.image(file).is_some_and(|i| self.single_channel(i))
            };
            if usable {
                metallic_roughness_tex = self.tex(Some(file), Some(r), &mut sets);
                if metallic_roughness_tex.is_some() {
                    // The texture's scale scales what it reads, where nothing is added to it.
                    roughness = r.factor(1);
                    if metal_too && !specular_workflow {
                        metallic = metallic_src.as_ref().map_or(1.0, |m| m.factor(2));
                    }
                }
            }
        }
        let emissive_tex = self.tex(ps.emissive_color.texture(), emissive.as_ref(), &mut sets);
        let emissive_color = match ps.emissive_color.value() {
            Some(c) => [c.x, c.y, c.z],
            None if emissive_tex.is_some() => emissive
                .as_ref()
                .map_or([1.0; 3], |e| [e.scale[0], e.scale[1], e.scale[2]]),
            None => [0.0; 3],
        };
        // glTF reads occlusion from red: another channel serves only from a gray image, where every channel is red.
        let occlusion_tex = occlusion.as_ref().and_then(|o| {
            let file = ps.occlusion.texture()?;
            let red = matches!(o.channel.as_str(), "r" | "rgb");
            if !red && !self.image(file).is_some_and(|i| self.single_channel(i)) {
                return None;
            }
            let mut t = self.tex(Some(file), Some(o), &mut sets)?;
            // A scale `s` and bias `1 - s` is glTF's strength `s`.
            let k = o.channel_index();
            let s = o.scale[k];
            if (o.bias[k] - (1.0 - s)).abs() < 1e-4 && (0.0..=1.0).contains(&s) {
                t.scale = s;
            }
            Some(t)
        });
        // A normal map's usual scale 2 and bias -1 is glTF's own reading; twice that a normal scale of 2.
        let normal_tex = self.tex(ps.normal.texture(), normal.as_ref(), &mut sets).map(|mut t| {
            if let Some(n) = &normal {
                let [sx, sy] = [n.scale[0], n.scale[1]];
                let matched = (sx - sy).abs() < 1e-4
                    && sx > 0.0
                    && (n.bias[0] + sx / 2.0).abs() < 1e-4
                    && (n.bias[1] + sy / 2.0).abs() < 1e-4;
                if matched {
                    t.scale = sx / 2.0;
                }
            }
            t
        });
        let specular_tex = if specular_workflow {
            self.tex(ps.specular_color.texture(), specular_src.as_ref(), &mut sets)
        } else {
            None
        };
        let specular = specular_workflow.then(|| {
            // UsdPreviewSurface's specular color is what reflects head on; glTF's scales a dielectric's 4%.
            let c = ps.specular_color.value().map_or([0.0; 3], |c| [c.x, c.y, c.z]);
            Specular {
                factor: Factor::of(1.0),
                color: if ps.specular_color.texture().is_some() {
                    [1.0 / 0.04; 3]
                } else {
                    c.map(|c| c / 0.04)
                },
                color_tex: specular_tex,
            }
        });
        let clearcoat = ps.clearcoat.value().filter(|c| **c > 0.0).map(|&c| Clearcoat {
            factor: Factor::of(c),
            roughness: Factor::of(ps.clearcoat_roughness.value().copied().unwrap_or(0.01)),
            normal: None,
        });
        let full = Full {
            name,
            base_color: color,
            base_color_tex,
            metallic,
            roughness,
            metallic_roughness_tex,
            normal_tex,
            occlusion_tex,
            emissive: emissive_color,
            emissive_tex,
            alpha,
            specular,
            clearcoat,
            ior: ps.ior.value().copied(),
            ..Full::default()
        };
        let reads = Reads {
            uvs: if sets.iter().all(String::is_empty) {
                Vec::new()
            } else {
                sets
            },
            color: color_read,
        };
        (full, reads)
    }

    /// A texture of `file` as `src` reads it, its coordinates numbered by their place in `sets`.
    fn tex(&mut self, file: Option<&str>, src: Option<&Source>, sets: &mut Vec<String>) -> Option<Tex> {
        let image = self.image(file?)?;
        let mut t = src.map_or_else(|| Tex::new(image), |s| s.tex(image));
        if let Some(name) = src.and_then(|s| s.uv.as_deref()).filter(|n| !n.is_empty()) {
            let k = match sets.iter().position(|s| s == name) {
                Some(k) => k,
                None if sets.first().is_some_and(String::is_empty) && sets.len() == 1 && t.uv == 0 => 0,
                None => {
                    sets.push(name.to_owned());
                    sets.len() - 1
                }
            };
            t.uv = u32::try_from(k).unwrap_or(0);
        }
        Some(t)
    }

    /// Whether an image is known, from its header, to be gray: one channel, with or without alpha, so its red,
    /// green and blue read the same. Unknown for an image not yet read.
    fn single_channel(&self, image: usize) -> bool {
        match self.images.list.get(image) {
            Some(crate::images::Source::Image(i)) => gray(&i.bytes),
            Some(crate::images::Source::Ref(r, _)) => r.bytes.as_deref().is_some_and(gray),
            None => false,
        }
    }

    /// The `UsdUVTexture` an input of `shader` reads, through node graphs: the output it uses (`rgb`, `r`, ...),
    /// its wrap, its scale, and the `UsdTransform2d` on its coordinates.
    fn texture_source(&self, shader: &usd::Prim, input: &str) -> Option<Source> {
        let (tex, channel) = self.connected(shader, &format!("inputs:{input}"))?;
        let file = match self.value(&tex.attribute("inputs:file")) {
            Some(sdf::Value::AssetPath(a)) => Some(a.authored_path),
            _ => None,
        };
        let wrap = |k: &str| match self.token(&tex.attribute(k)).as_deref() {
            Some("mirror") => material::MIRROR,
            Some("clamp" | "black") => material::CLAMP,
            _ => material::REPEAT,
        };
        let four = |name: &str, fallback: [f32; 4]| match self.value(&tex.attribute(name)) {
            Some(sdf::Value::Vec4f(v)) => [v.x, v.y, v.z, v.w],
            #[allow(clippy::cast_possible_truncation, reason = "colors")]
            Some(sdf::Value::Vec4d(v)) => [v.x as f32, v.y as f32, v.z as f32, v.w as f32],
            _ => fallback,
        };
        let (scale, bias) = (four("inputs:scale", [1.0; 4]), four("inputs:bias", [0.0; 4]));
        let mut transform = None;
        if let Some((t, _)) = self.connected(&tex, "inputs:st")
            && self.token(&t.attribute("info:id")).as_deref() == Some("UsdTransform2d")
        {
            let [su, sv] = self.vec2(&t.attribute("inputs:scale")).unwrap_or([1.0; 2]);
            let [tu, tv] = self.vec2(&t.attribute("inputs:translation")).unwrap_or([0.0; 2]);
            let angle = self.number(&t.attribute("inputs:rotation"), 0.0).to_radians();
            if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
                // USD's `st` has V up; the document's UVs are flipped.
                let (sin, cos) = angle.sin_cos();
                transform = Some(UvTransform {
                    offset: [tu - sv * sin, 1.0 - tv - sv * cos],
                    rotation: angle,
                    scale: [su, sv],
                });
            }
        }
        Some(Source {
            file,
            channel,
            wrap: [wrap("inputs:wrapS"), wrap("inputs:wrapT")],
            scale,
            bias,
            transform,
            uv: self.st_name(&tex),
        })
    }

    /// The texture coordinate set a `UsdUVTexture` reads, by primvar name: the primvar reader on its `st`, through
    /// any `UsdTransform2d`.
    fn st_name(&self, tex: &usd::Prim) -> Option<String> {
        let (mut at, _) = self.connected(tex, "inputs:st")?;
        for _ in 0..8 {
            let id = self.token(&at.attribute("info:id"))?;
            if id.starts_with("UsdPrimvarReader") {
                return self.varname(&at);
            }
            if id != "UsdTransform2d" {
                return None;
            }
            at = self.connected(&at, "inputs:in")?.0;
        }
        None
    }

    /// The primvar a `UsdPrimvarReader_*` reads: its `varname`, a token or string, or the material input it's
    /// connected to.
    fn varname(&self, reader: &usd::Prim) -> Option<String> {
        let mut attr = reader.attribute("inputs:varname");
        for _ in 0..8 {
            if let Some(path) = attr.connections().ok().and_then(|c| c.into_iter().next()) {
                let (at, prop) = path.as_str().rsplit_once('.')?;
                attr = self.stage.prim(at).ok()?.attribute(prop);
                continue;
            }
            let name = match self.value(&attr)? {
                sdf::Value::Token(t) => t.as_str().to_owned(),
                sdf::Value::String(s) => s,
                _ => return None,
            };
            let name = name.strip_prefix("primvars:").unwrap_or(&name);
            return (!name.is_empty()).then(|| name.to_owned());
        }
        None
    }

    /// The shader and output an attribute connects to, followed through node graphs' outputs.
    fn connected(&self, prim: &usd::Prim, attr: &str) -> Option<(usd::Prim, String)> {
        let mut path = prim.attribute(attr).connections().ok()?.into_iter().next()?;
        for _ in 0..8 {
            let (at, prop) = path.as_str().rsplit_once('.')?;
            let p = self.stage.prim(at).ok()?;
            let next = p.attribute(prop).connections().ok().and_then(|c| c.into_iter().next());
            match next {
                Some(n) if prop.starts_with("outputs:") => path = n,
                _ => return Some((p, prop.strip_prefix("outputs:").unwrap_or(prop).to_owned())),
            }
        }
        None
    }

    fn vec2(&self, attr: &usd::Attribute) -> Option<[f32; 2]> {
        match self.value(attr)? {
            sdf::Value::Vec2f(v) => Some([v.x, v.y]),
            #[allow(clippy::cast_possible_truncation, reason = "texture coordinates")]
            sdf::Value::Vec2d(v) => Some([v.x as f32, v.y as f32]),
            _ => None,
        }
    }
}

/// A `UsdUVTexture` as an input reads it.
struct Source {
    file: Option<String>,
    /// The output read: `rgb`, `r`, `g`, `b` or `a`.
    channel: String,
    wrap: [u32; 2],
    /// `inputs:scale`, which tints what it reads.
    scale: [f32; 4],
    /// `inputs:bias`, added after the scale.
    bias: [f32; 4],
    transform: Option<UvTransform>,
    /// The texture coordinate set it reads, by primvar name, when a primvar reader names it.
    uv: Option<String>,
}

impl Source {
    /// The channel read, as an index into `scale` and `bias`: red for `rgb`.
    fn channel_index(&self) -> usize {
        match self.channel.as_str() {
            "g" => 1,
            "b" => 2,
            "a" => 3,
            _ => 0,
        }
    }

    /// The factor a glTF texture of this channel takes: the scale of channel `k`, where no bias is added.
    fn factor(&self, k: usize) -> f32 {
        let k = if matches!(self.channel.as_str(), "r" | "g" | "b" | "a") {
            self.channel_index()
        } else {
            k
        };
        if self.bias.iter().all(|b| *b == 0.0) {
            self.scale[k].clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    fn tex(&self, image: usize) -> Tex {
        Tex {
            transform: self.transform,
            wrap: self.wrap,
            ..Tex::new(image)
        }
    }
}

/// A value as JSON, for extras: numbers, strings, tokens, asset paths, vectors, matrices, arrays of them and
/// dictionaries.
fn json_of(v: &sdf::Value) -> Option<Value> {
    use sdf::Value as V;
    let h = |x: gf::f16| x.to_f32();
    Some(match v {
        V::Bool(b) => json!(b),
        V::Uchar(x) => json!(x),
        V::Int(x) => json!(x),
        V::Uint(x) => json!(x),
        V::Int64(x) => json!(x),
        V::Uint64(x) => json!(x),
        V::Half(x) => json!(h(*x)),
        V::Float(x) => json!(x),
        V::Double(x) => json!(x),
        V::String(s) => json!(s),
        V::Token(t) => json!(t.as_str()),
        V::AssetPath(a) => json!(a.authored_path),
        V::BoolVec(v) => json!(v),
        V::UcharVec(v) => json!(v),
        V::IntVec(v) => json!(v),
        V::UintVec(v) => json!(v),
        V::Int64Vec(v) => json!(v),
        V::Uint64Vec(v) => json!(v),
        V::HalfVec(v) => json!(v.iter().map(|x| h(*x)).collect::<Vec<_>>()),
        V::FloatVec(v) => json!(v),
        V::DoubleVec(v) => json!(v),
        V::StringVec(v) => json!(v),
        V::TokenVec(v) => json!(v.iter().map(tf::Token::as_str).collect::<Vec<_>>()),
        V::AssetPathVec(v) => json!(v.iter().map(|a| a.authored_path.as_str()).collect::<Vec<_>>()),
        V::Vec2f(p) => json!([p.x, p.y]),
        V::Vec2d(p) => json!([p.x, p.y]),
        V::Vec2i(p) => json!([p.x, p.y]),
        V::Vec2h(p) => json!([h(p.x), h(p.y)]),
        V::Vec3f(p) => json!([p.x, p.y, p.z]),
        V::Vec3d(p) => json!([p.x, p.y, p.z]),
        V::Vec3i(p) => json!([p.x, p.y, p.z]),
        V::Vec3h(p) => json!([h(p.x), h(p.y), h(p.z)]),
        V::Vec4f(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4d(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4i(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4h(p) => json!([h(p.x), h(p.y), h(p.z), h(p.w)]),
        V::Vec2fVec(v) => json!(v.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
        V::Vec2dVec(v) => json!(v.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
        V::Vec3fVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>()),
        V::Vec3dVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>()),
        V::Vec4fVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z, p.w]).collect::<Vec<_>>()),
        V::Vec4dVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z, p.w]).collect::<Vec<_>>()),
        V::Matrix4d(m) => json!(m.0),
        V::Dictionary(d) => {
            let mut keys: Vec<&String> = d.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .filter_map(|k| Some((k.clone(), json_of(&d[k])?)))
                    .collect(),
            )
        }
        _ => return None,
    })
}

/// Token values (a `token[]`, or `string[]`).
fn tokens(v: Option<sdf::Value>) -> Vec<String> {
    match v {
        Some(sdf::Value::TokenVec(v)) => v.iter().map(|t| t.as_str().to_owned()).collect(),
        Some(sdf::Value::StringVec(v)) => v,
        _ => Vec::new(),
    }
}

fn matrices(v: Option<sdf::Value>) -> Vec<gf::Matrix4d> {
    match v {
        Some(sdf::Value::Matrix4dVec(v)) => v,
        _ => Vec::new(),
    }
}

/// Two numbers, as a clipping range is.
fn pair(v: &sdf::Value) -> Option<[f64; 2]> {
    match v {
        sdf::Value::Vec2f(p) => Some([f64::from(p.x), f64::from(p.y)]),
        sdf::Value::Vec2d(p) => Some([p.x, p.y]),
        _ => None,
    }
}

/// A surface as polygons in its prim's space, with what it carries per point or corner.
#[derive(Clone)]
struct Shape {
    points: Vec<[f32; 3]>,
    counts: Vec<i32>,
    corners: Vec<i32>,
    /// Faces `holeIndices` leaves out.
    hole: Vec<bool>,
    normals: Option<Primvar<[f32; 3]>>,
    /// Texture coordinate sets by primvar name, the likeliest color texture's first.
    uvs: Vec<(String, Primvar<[f32; 2]>)>,
    /// Color sets other than the display color, by primvar name.
    colors: Vec<(String, Primvar<[f32; 4]>)>,
    /// Authored tangents; a w of 0 is a handedness the file leaves to `bitangents`.
    tangents: Option<Primvar<[f32; 4]>>,
    bitangents: Option<Primvar<[f32; 3]>>,
    /// Faces wound clockwise.
    left_handed: bool,
    /// Smoothed when it has no normals: a subdivision surface.
    smooth: bool,
}

/// How finely the implicit surfaces are tessellated: around their axis, and from pole to pole. Multiples of four
/// and two, so the tessellation reaches the surface's extent on every axis.
const AROUND: usize = 32;
const STACKS: usize = 16;

/// A tessellation of an implicit surface, built around Z and then turned to the surface's axis.
#[derive(Default)]
struct Tess {
    points: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    counts: Vec<i32>,
    corners: Vec<i32>,
}

impl Tess {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "a tessellated shape has AROUND * (STACKS + 1) points at most"
    )]
    fn face(&mut self, corners: &[([f32; 3], [f32; 3], [f32; 2])]) {
        self.counts.push(corners.len() as i32);
        for &(p, n, uv) in corners {
            self.corners.push(self.points.len() as i32);
            self.points.push(p);
            self.normals.push(n);
            self.uvs.push(uv);
        }
    }

    /// The shape, its Z axis turned to `axis` (a cyclic swap, which keeps the winding).
    fn finish(self, axis: Option<&str>) -> Shape {
        let turn = |v: [f32; 3]| match axis {
            Some("X") => [v[2], v[0], v[1]],
            Some("Y") => [v[1], v[2], v[0]],
            _ => v,
        };
        Shape {
            points: self.points.into_iter().map(turn).collect(),
            counts: self.counts,
            corners: self.corners,
            hole: Vec::new(),
            normals: Some(Primvar {
                values: self.normals.into_iter().map(turn).collect(),
                indices: None,
                spread: Spread::FaceVarying,
            }),
            uvs: vec![(
                "st".to_owned(),
                Primvar {
                    values: self.uvs,
                    indices: None,
                    spread: Spread::FaceVarying,
                },
            )],
            colors: Vec::new(),
            tangents: None,
            bitangents: None,
            left_handed: false,
            smooth: false,
        }
    }

    /// A cube of edge `size` about the origin, each face's texture coordinates covering the unit square.
    fn cube(size: f32) -> Shape {
        let h = size / 2.0;
        let mut t = Self::default();
        for k in 0..3 {
            let (u, v) = ((k + 1) % 3, (k + 2) % 3);
            for s in [1.0f32, -1.0] {
                let mut quad: Vec<([f32; 3], [f32; 3], [f32; 2])> =
                    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                        .iter()
                        .map(|&(a, b)| {
                            let mut p = [0.0; 3];
                            let mut n = [0.0; 3];
                            p[k] = s * h;
                            p[u] = a * h;
                            p[v] = b * h;
                            n[k] = s;
                            (p, n, [f32::midpoint(a, 1.0), f32::midpoint(b, 1.0)])
                        })
                        .collect();
                if s < 0.0 {
                    quad.reverse();
                }
                t.face(&quad);
            }
        }
        t.finish(None)
    }

    /// A plane facing `axis`: `width` along X (Z when facing X), `length` along Y (Z when facing Y). U runs
    /// along the width and V along the length, over the unit square.
    fn plane(width: f32, length: f32, axis: Option<&str>) -> Shape {
        let (w, l) = (width / 2.0, length / 2.0);
        let (corners, n, uvs): ([[f32; 3]; 4], [f32; 3], [[f32; 2]; 4]) = match axis {
            Some("X") => (
                [[0.0, -l, -w], [0.0, l, -w], [0.0, l, w], [0.0, -l, w]],
                [1.0, 0.0, 0.0],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            ),
            Some("Y") => (
                [[-w, 0.0, -l], [-w, 0.0, l], [w, 0.0, l], [w, 0.0, -l]],
                [0.0, 1.0, 0.0],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            ),
            _ => (
                [[-w, -l, 0.0], [w, -l, 0.0], [w, l, 0.0], [-w, l, 0.0]],
                [0.0, 0.0, 1.0],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ),
        };
        let mut t = Self::default();
        t.face(&[0, 1, 2, 3].map(|k| (corners[k], n, uvs[k])));
        t.finish(None)
    }

    /// Rows of a sphere (`h` 0) or capsule from bottom to top: radius, height, and the normal's radial and axial
    /// parts. The bottom half has radius `bottom` about -h/2, the top half `top` about h/2.
    fn sphere_rows(bottom: f32, top: f32, h: f32) -> Vec<[f32; 4]> {
        let mut rows = Vec::new();
        let half = STACKS / 2;
        for k in 0..=STACKS {
            let (s, c) = match k {
                0 => (-1.0, 0.0),
                k if k == STACKS => (1.0, 0.0),
                #[allow(clippy::cast_precision_loss, reason = "a small loop counter")]
                k => (std::f32::consts::PI * (k as f32 - half as f32) / STACKS as f32).sin_cos(),
            };
            if k <= half {
                rows.push([bottom * c, -h / 2.0 + bottom * s, c, s]);
            }
            if k >= half {
                rows.push([top * c, h / 2.0 + top * s, c, s]);
            }
        }
        rows
    }

    /// Rows of a capped cylinder, cone (`top` 0) or frustum of height `h`.
    fn frustum_rows(bottom: f32, top: f32, h: f32) -> Vec<[f32; 4]> {
        let len = h.hypot(bottom - top).max(f32::MIN_POSITIVE);
        let (nr, nz) = (h / len, (bottom - top) / len);
        vec![
            [0.0, -h / 2.0, 0.0, -1.0],
            [bottom, -h / 2.0, 0.0, -1.0],
            [bottom, -h / 2.0, nr, nz],
            [top, h / 2.0, nr, nz],
            [top, h / 2.0, 0.0, 1.0],
            [0.0, h / 2.0, 0.0, 1.0],
        ]
    }

    /// A surface of revolution about Z through `rows`, turned to `axis`. U runs once around, V along the
    /// profile by its length from the bottom.
    #[allow(
        clippy::float_cmp,
        reason = "a radius of exactly 0 is a pole; any other closes a band"
    )]
    fn lathe(rows: &[[f32; 4]], axis: Option<&str>) -> Shape {
        let mut along = Vec::with_capacity(rows.len());
        let mut length = 0.0f32;
        for (k, row) in rows.iter().enumerate() {
            if let Some(prev) = k.checked_sub(1).map(|p| rows[p]) {
                length += (row[0] - prev[0]).hypot(row[1] - prev[1]);
            }
            along.push(length);
        }
        let length = if length > 0.0 && length.is_finite() {
            length
        } else {
            1.0
        };
        let ring = |k: usize, j: usize| {
            let row = rows[k];
            #[allow(clippy::cast_precision_loss, reason = "a small loop counter")]
            let (s, c) = (std::f32::consts::TAU * (j % AROUND) as f32 / AROUND as f32).sin_cos();
            #[allow(clippy::cast_precision_loss, reason = "a small loop counter")]
            let u = j as f32 / AROUND as f32;
            (
                [row[0] * c, row[0] * s, row[1]],
                [row[2] * c, row[2] * s, row[3]],
                [u, along[k] / length],
            )
        };
        let mut t = Self::default();
        for k in 0..rows.len().saturating_sub(1) {
            let (a, b) = (rows[k], rows[k + 1]);
            if (a[0] == b[0] && a[1] == b[1]) || (a[0] == 0.0 && b[0] == 0.0) {
                continue;
            }
            let (a, b) = (k, k + 1);
            for i in 0..AROUND {
                if rows[a][0] == 0.0 {
                    t.face(&[ring(a, i), ring(b, i + 1), ring(b, i)]);
                } else if rows[b][0] == 0.0 {
                    t.face(&[ring(a, i), ring(a, i + 1), ring(b, i)]);
                } else {
                    t.face(&[ring(a, i), ring(a, i + 1), ring(b, i + 1), ring(b, i)]);
                }
            }
        }
        t.finish(axis)
    }
}

/// The material bound to `prim` (directly or through a parent), if any.
fn bound(prim: &usd::Prim) -> Option<sdf::Path> {
    MaterialBindingAPI::from_prim_unchecked(prim.clone())
        .compute_bound_material("")
        .ok()
        .flatten()
}

/// A point instancer's placement of one instance: scale, then orientation (w, x, y, z), then translation, as
/// row vectors compose.
fn instance_matrix(s: [f32; 3], q: [f32; 4], t: [f32; 3]) -> gf::Matrix4d {
    let len = q.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
    let [w, x, y, z] = if len > 0.0 {
        q.map(|c| f64::from(c) / len)
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    // The rotation for column vectors; row vectors take its transpose.
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut m = gf::Matrix4d::IDENTITY;
    for row in 0..3 {
        // USD matrices are row vectors, so row `row` of the transform is column `row` of `r`.
        for (col, r_col) in r.iter().enumerate() {
            m.0[row * 4 + col] = f64::from(s[row]) * r_col[row];
        }
        m.0[12 + row] = f64::from(t[row]);
    }
    m
}

/// Smooth normals as Hydra makes them: each face corner adds the cross product of its two edges to its point.
#[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
fn smooth_normals(points: &[[f32; 3]], corners: &[i32], faces: &[(usize, usize, usize)], flip: bool) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f64; 3]; points.len()];
    for &(_, start, n) in faces {
        let ix = &corners[start..start + n];
        for k in 0..n {
            let (p, c, nx) = (ix[(k + n - 1) % n] as usize, ix[k] as usize, ix[(k + 1) % n] as usize);
            let at = |i: usize| points[i].map(f64::from);
            let (a, b, o) = (at(nx), at(p), at(c));
            let (u, v) = (
                [a[0] - o[0], a[1] - o[1], a[2] - o[2]],
                [b[0] - o[0], b[1] - o[1], b[2] - o[2]],
            );
            let x = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            for (s, d) in acc[c].iter_mut().zip(x) {
                *s += d;
            }
        }
    }
    acc.into_iter().map(|n| unit(n, flip)).collect()
}

/// A polygon's normal (Newell's), unit length.
fn polygon_normal(poly: &[[f32; 3]], flip: bool) -> [f32; 3] {
    let mut n = [0f64; 3];
    for (i, a) in poly.iter().enumerate() {
        let (a, b) = (a.map(f64::from), poly[(i + 1) % poly.len()].map(f64::from));
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    unit(n, flip)
}

fn unit(n: [f64; 3], flip: bool) -> [f32; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let s = if flip { -1.0 } else { 1.0 };
    if len > 0.0 {
        n.map(|x| f32_of(s * x / len))
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn vec3s(v: sdf::Value) -> Option<Vec<[f32; 3]>> {
    Some(match v {
        sdf::Value::Vec3fVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z]).collect(),
        sdf::Value::Vec3dVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z].map(f32_of)).collect(),
        sdf::Value::Vec3hVec(v) => v
            .into_iter()
            .map(|p| [p.x.to_f32(), p.y.to_f32(), p.z.to_f32()])
            .collect(),
        sdf::Value::Vec3f(p) => vec![[p.x, p.y, p.z]],
        sdf::Value::Vec3d(p) => vec![[p.x, p.y, p.z].map(f32_of)],
        _ => return None,
    })
}

fn vec2s(v: sdf::Value) -> Option<Vec<[f32; 2]>> {
    Some(match v {
        sdf::Value::Vec2fVec(v) => v.into_iter().map(|p| [p.x, p.y]).collect(),
        sdf::Value::Vec2dVec(v) => v.into_iter().map(|p| [p.x, p.y].map(f32_of)).collect(),
        sdf::Value::Vec2hVec(v) => v.into_iter().map(|p| [p.x.to_f32(), p.y.to_f32()]).collect(),
        _ => return None,
    })
}

/// Four-component values; three-component ones (a color without alpha) get a fourth of 1.
fn vec4s(v: sdf::Value) -> Option<Vec<[f32; 4]>> {
    Some(match v {
        sdf::Value::Vec4fVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z, p.w]).collect(),
        sdf::Value::Vec4dVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z, p.w].map(f32_of)).collect(),
        sdf::Value::Vec4hVec(v) => v
            .into_iter()
            .map(|p| [p.x.to_f32(), p.y.to_f32(), p.z.to_f32(), p.w.to_f32()])
            .collect(),
        v => vec3s(v)?.into_iter().map(|[x, y, z]| [x, y, z, 1.0]).collect(),
    })
}

/// Tangents: a 4-vector's w is its handedness; a 3-vector gets a w of 0, handedness unknown.
fn tangent4s(v: sdf::Value) -> Option<Vec<[f32; 4]>> {
    match v {
        v @ (sdf::Value::Vec4fVec(_) | sdf::Value::Vec4dVec(_) | sdf::Value::Vec4hVec(_)) => vec4s(v),
        v => Some(vec3s(v)?.into_iter().map(|[x, y, z]| [x, y, z, 0.0]).collect()),
    }
}

/// Numbers in double precision, as knots want.
fn doubles(v: sdf::Value) -> Option<Vec<f64>> {
    Some(match v {
        sdf::Value::DoubleVec(v) => v,
        sdf::Value::FloatVec(v) => v.into_iter().map(f64::from).collect(),
        sdf::Value::HalfVec(v) => v.into_iter().map(|h| f64::from(h.to_f32())).collect(),
        _ => return None,
    })
}

fn double2s(v: sdf::Value) -> Option<Vec<[f64; 2]>> {
    Some(match v {
        sdf::Value::Vec2dVec(v) => v.into_iter().map(|p| [p.x, p.y]).collect(),
        sdf::Value::Vec2fVec(v) => v.into_iter().map(|p| [p.x, p.y].map(f64::from)).collect(),
        _ => return None,
    })
}

fn floats(v: sdf::Value) -> Option<Vec<f32>> {
    Some(match v {
        sdf::Value::FloatVec(v) => v,
        sdf::Value::DoubleVec(v) => v.into_iter().map(f32_of).collect(),
        sdf::Value::HalfVec(v) => v.into_iter().map(openusd::gf::f16::to_f32).collect(),
        sdf::Value::Float(x) => vec![x],
        sdf::Value::Double(x) => vec![f32_of(x)],
        _ => return None,
    })
}

/// Quaternions as (w, x, y, z).
fn quats(v: sdf::Value) -> Option<Vec<[f32; 4]>> {
    Some(match v {
        sdf::Value::QuathVec(v) => v
            .into_iter()
            .map(|q| [q.w.to_f32(), q.x.to_f32(), q.y.to_f32(), q.z.to_f32()])
            .collect(),
        sdf::Value::QuatfVec(v) => v.into_iter().map(|q| [q.w, q.x, q.y, q.z]).collect(),
        _ => return None,
    })
}

fn ints(v: Option<sdf::Value>) -> Vec<i32> {
    match v {
        Some(sdf::Value::IntVec(v)) => v,
        _ => Vec::new(),
    }
}

fn int64s(v: Option<sdf::Value>) -> Vec<i64> {
    match v {
        Some(sdf::Value::Int64Vec(v)) => v,
        Some(sdf::Value::IntVec(v)) => v.into_iter().map(i64::from).collect(),
        _ => Vec::new(),
    }
}

/// Names texture coordinates go by: USD's own `st`, then what exporters write.
const UV_NAMES: &[&str] = &["st", "st0", "st_0", "UVMap", "uv", "uv0", "map1", "UVChannel_1"];

impl Reader<'_> {
    /// A texture by its asset path: inside the `.usdz`, beside the file, or anywhere in the pack by name.
    fn image(&mut self, asset: &str) -> Option<usize> {
        if !(self.keep.materials && self.keep.textures) {
            return None;
        }
        if let Some(&i) = self.image_of.get(asset) {
            return i;
        }
        let i = if asset.contains("<UDIM>") || asset.contains("<udim>") {
            // A tiled texture, as its first tile there is.
            (1001..=1010).find_map(|tile| {
                let tile = tile.to_string();
                self.find_image(&asset.replace("<UDIM>", &tile).replace("<udim>", &tile))
            })
        } else {
            self.find_image(asset)
        };
        self.image_of.insert(asset.to_owned(), i);
        i
    }

    /// An image by its asset path, read when found.
    fn find_image(&mut self, reference: &str) -> Option<usize> {
        let reference = reference.trim_start_matches("./");
        let assets = self.assets;
        self.usdz
            .as_mut()
            .and_then(|z| z.read(reference))
            .and_then(|(entry, bytes)| {
                let convert = |b: &[u8]| assets.image(b, Path::new(reference));
                self.images
                    .bytes(bytes.into(), reference, Location::Archive(entry), convert)
            })
            .or_else(|| {
                let dir = self.file.parent().unwrap_or(Path::new(""));
                self.images.file(assets, assets.find(reference, dir)?, reference)
            })
    }
}

/// A `.usdz` opened for its textures.
struct Usdz<'a>(zip::ZipArchive<Box<dyn super::ReadSeek + 'a>>);

impl<'a> Usdz<'a> {
    fn open(input: Input<'a>) -> Option<Self> {
        let is_usdz = input.path.extension().is_some_and(|e| e.eq_ignore_ascii_case("usdz"));
        let stream = input.seekable().ok().filter(|_| is_usdz)?;
        zip::ZipArchive::new(stream).ok().map(Self)
    }

    /// An entry by its path in the archive, or failing that by its file name.
    fn read(&mut self, reference: &str) -> Option<(String, Vec<u8>)> {
        let name = reference.rsplit('/').next()?.to_lowercase();
        let names: Vec<String> = self
            .0
            .file_names()
            .filter_map(Result::ok)
            .map(std::borrow::Cow::into_owned)
            .collect();
        let entry = names.iter().find(|n| n.eq_ignore_ascii_case(reference)).or_else(|| {
            names
                .iter()
                .find(|n| n.rsplit('/').next().is_some_and(|f| f.to_lowercase() == name))
        })?;
        let mut f = self.0.by_name(entry).ok()?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).ok()?;
        Some((entry.clone(), bytes))
    }
}

/// How many times a model in memory is reopened for the layers it names: each round fetches the layers the last
/// one found missing, so this is the depth of references a model can have.
const FETCH_ROUNDS: usize = 32;

/// The stage of a model in memory, and its prims' paths: opened and composed, then again with whatever layers it
/// named that `assets` has, until it names no more.
fn open_in_memory(bytes: &[u8], input: Input<'_>, assets: &dyn Assets) -> Result<(usd::Stage, Vec<sdf::Path>)> {
    let mut root = memory_key(&input.path.to_string_lossy());
    let ext = input.path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !["usd", "usda", "usdc", "usdz"]
        .iter()
        .any(|e| ext.eq_ignore_ascii_case(e))
    {
        // The format is read from the name: a model named without its extension is given one.
        root.push_str(if bytes.starts_with(b"PK") { ".usdz" } else { ".usd" });
    }
    let mut files = HashMap::new();
    files.insert(root.clone(), Arc::<[u8]>::from(bytes));
    let mut tried = BTreeSet::new();
    let mut round = 0;
    loop {
        let resolver = MemResolver::new(files.clone());
        let missing = Rc::clone(&resolver.missing);
        let stage = usd::Stage::builder()
            .schema_registry(openusd_schemas::schema_registry())
            .resolver(resolver)
            .open(&root)
            .context(ErrorKind::Corrupt, "opening the USD stage")?;
        let paths = prim_paths(&stage)?;
        round += 1;
        let mut found = false;
        for id in missing.take() {
            if tried.insert(id.clone())
                && let Some(b) = assets.read(Path::new(&id))
            {
                files.insert(id, Arc::from(b.as_ref()));
                found = true;
            }
        }
        if !found || round == FETCH_ROUNDS {
            return Ok((stage, paths));
        }
    }
}

/// Resolves a model in memory: its layers are `files`, keyed by their paths among the assets (`/`-separated,
/// relative), and a package's layers are its entries (`pkg.usdz[inner.usda]`). A layer it doesn't have is noted in
/// `missing`, for the next round to fetch.
struct MemResolver {
    files: HashMap<String, Arc<[u8]>>,
    missing: Rc<RefCell<BTreeSet<String>>>,
    id: u64,
}

impl MemResolver {
    fn new(files: HashMap<String, Arc<[u8]>>) -> Self {
        // Ids apart from the ones `openusd` gives its own resolvers, which count up from zero.
        static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
        Self {
            files,
            missing: Rc::default(),
            id: NEXT.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Whether the plain path (no package) is here, noting it missing when it isn't.
    fn has(&self, path: &str) -> bool {
        let here = self.files.contains_key(path);
        if !here {
            self.missing.borrow_mut().insert(path.to_owned());
        }
        here
    }
}

/// `leaf` placed inside the deepest package of `base`: `pkg.usdz[inner.usdz]` and `a.usda` give
/// `pkg.usdz[inner.usdz[a.usda]]`.
fn nest(base: &str, leaf: &str) -> String {
    match ar::split_package_relative_path_outer(base) {
        Some((package, inner)) => ar::join_package_relative_path(&package, &nest(&inner, leaf)),
        None => ar::join_package_relative_path(base, leaf),
    }
}

impl ar::Resolver for MemResolver {
    fn create_identifier(&self, asset_path: &str, anchor: Option<&ar::ResolvedPath>) -> String {
        if asset_path.is_empty() {
            return String::new();
        }
        if let Some((package, inner)) = ar::split_package_relative_path_outer(asset_path) {
            return nest(&self.create_identifier(&package, anchor), &memory_key(&inner));
        }
        let Some(anchor) = anchor.map(|a| a.to_string_lossy().replace('\\', "/")) else {
            return memory_key(asset_path);
        };
        if let Some((package, inner)) = ar::split_package_relative_path_inner(&anchor) {
            // A layer inside a package refers to its neighbours in the package.
            let dir = inner.rsplit_once('/').map_or("", |(dir, _)| dir);
            return nest(&package, &memory_key(&format!("{dir}/{asset_path}")));
        }
        if anchor.to_ascii_lowercase().ends_with(".usdz") {
            // A package's root layer refers to the package's entries.
            return ar::join_package_relative_path(&anchor, &memory_key(asset_path));
        }
        if asset_path.starts_with(['/', '\\']) {
            return memory_key(asset_path);
        }
        let dir = anchor.rsplit_once('/').map_or("", |(dir, _)| dir);
        memory_key(&format!("{dir}/{asset_path}"))
    }

    fn resolve(&self, asset_path: &str) -> Option<ar::ResolvedPath> {
        if asset_path.is_empty() {
            return None;
        }
        let found = match ar::split_package_relative_path_outer(asset_path) {
            Some((package, inner)) => self.has(&package) && openusd::usdz::package_contains(self, &package, &inner),
            None => self.has(asset_path),
        };
        found.then(|| ar::ResolvedPath::new(asset_path))
    }

    fn resolve_for_new_asset(&self, _asset_path: &str) -> Option<ar::ResolvedPath> {
        None
    }

    fn open_asset(&self, resolved_path: &ar::ResolvedPath) -> io::Result<Box<dyn ar::Asset>> {
        let path = resolved_path.to_string_lossy();
        let bytes = match ar::split_package_relative_path_outer(&path) {
            Some((package, inner)) => openusd::usdz::package_entry(self, &package, &inner)?,
            None => match self.files.get(path.as_ref()) {
                Some(b) => ar::AssetBuffer::from(Arc::clone(b)),
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("{path} isn't among the assets"),
                    ));
                }
            },
        };
        Ok(Box::new(io::Cursor::new(bytes)))
    }

    fn get_modification_timestamp(&self, _: &str, _: &ar::ResolvedPath) -> Option<std::time::SystemTime> {
        None
    }

    fn identity(&self) -> String {
        format!("polysmelt-memory-{}", self.id)
    }

    fn cache_id(&self) -> Option<u64> {
        Some(self.id)
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::document::Pose;
    use std::fmt::Write as _;

    fn read(path: &Path, dir: &Path) -> Option<crate::mesh::Mesh> {
        load(Input::file(path), &crate::assets::Folder::new(dir), Keep::ALL)
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
    }

    const CUBE: &str = r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World"
{
    double3 xformOp:translate = (10, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Quad" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        uniform token subdivisionScheme = "none"
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "faceVarying")
        rel material:binding = </World/Looks/Red>
    }

    def Mesh "Hidden"
    {
        token visibility = "invisible"
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }

    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.8, 0.1, 0.1)
                token outputs:surface
            }
        }
    }
}
"#;

    #[test]
    fn reads_meshes_transforms_and_materials() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quad.usda");
        std::fs::write(&path, CUBE).unwrap();
        let mesh = read(&path, dir.path()).unwrap();
        assert_eq!(mesh.prims.len(), 1, "the invisible mesh is left out");
        assert_eq!(mesh.triangles(), 2);
        let prim = &mesh.prims[0];
        // Translated 10 along X; Z up becomes Y up, so the quad in the XY plane lies in XZ.
        for (got, want) in [
            (prim.positions[1], [11.0, 0.0, 0.0]),
            (prim.positions[2], [11.0, 0.0, -1.0]),
        ] {
            assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-5), "{got:?}");
        }
        let n = prim.normals[0];
        assert!((n[1] - 1.0).abs() < 1e-5, "the +Z face normal points up: {n:?}");
        assert_eq!(prim.uvs[2], [1.0, 0.0], "V flipped to glTF's top-left origin");
        let m = &mesh.materials[prim.material];
        assert_eq!(m.name, "Red");
        assert!((m.color[0] - 0.8).abs() < 1e-6 && (m.color[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_material_library_has_no_mesh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("looks.usda");
        std::fs::write(
            &path,
            "#usda 1.0
def Material \"Planks\"
{
}
",
        )
        .unwrap();
        assert!(read(&path, dir.path()).is_none());
    }

    #[test]
    fn a_mesh_with_no_material_shows_its_display_color() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.usda");
        std::fs::write(
            &path,
            "#usda 1.0\ndef Mesh \"Tri\"\n{\n    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n    color3f[] primvars:displayColor = [(0, 0.5, 1)] (interpolation = \"constant\")\n}\n",
        )
        .unwrap();
        let mesh = read(&path, dir.path()).unwrap();
        assert_eq!(mesh.materials[mesh.prims[0].material].color, [0.0, 0.5, 1.0, 1.0]);
        // Default units are centimeters.
        assert_eq!(mesh.prims[0].positions[1], [0.01, 0.0, 0.0]);
    }

    /// A stage of `usda` text in a folder of its own, beside `images` (each a gray or a color PNG by name).
    fn stage(usda: &str, images: &[(&str, bool)]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        for &(name, gray) in images {
            let path = dir.path().join(name);
            if gray {
                image::GrayImage::from_pixel(2, 2, image::Luma([200]))
                    .save(path)
                    .unwrap();
            } else {
                image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
                    .save(path)
                    .unwrap();
            }
        }
        let path = dir.path().join("stage.usda");
        std::fs::write(&path, usda).unwrap();
        (dir, path)
    }

    fn doc(usda: &str, images: &[(&str, bool)]) -> Document {
        let (dir, path) = stage(usda, images);
        load(Input::file(&path), &crate::assets::Folder::new(dir.path()), Keep::ALL).unwrap()
    }

    fn flat(usda: &str, images: &[(&str, bool)]) -> crate::mesh::Mesh {
        doc(usda, images).flatten(Pose::Rest).unwrap().unwrap()
    }

    #[test]
    fn what_the_stage_says_of_itself() {
        use crate::source::{Axis, Fact, Handedness};
        let mesh = "def Mesh \"Quad\"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
";
        let usda = format!(
            "#usda 1.0
(
    doc = \"A crate\"
    comment = \"first pass\"
    upAxis = \"Z\"
    metersPerUnit = 0.0254
    framesPerSecond = 30
    timeCodesPerSecond = 60
    customLayerData = {{
        string creator = \"someone\"
    }}
)

{mesh}"
        );
        let d = doc(&usda, &[]);
        let info = d.source_info();
        assert_eq!(info.format.as_deref(), Some("USD"));
        assert_eq!(info.meters_per_unit, Fact::Declared(0.0254));
        assert_eq!(info.unit.as_deref(), Some("inch"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
        assert_eq!(info.frame_rate, Fact::Declared(30.0));
        assert!(info.converted.units && info.converted.up_axis);
        assert_eq!(info.description.as_deref(), Some("A crate"));
        assert_eq!(info.comments, ["first pass"]);
        assert_eq!(
            d.json()["asset"]["extras"]["source"]["customLayerData"]["creator"],
            "someone"
        );
        assert_eq!(d.json()["asset"]["extras"]["source"]["timeCodesPerSecond"], 60.0);

        // A stage that says nothing: USD's centimeters, Y up and 24 time codes a second, assumed.
        let d = doc(
            &format!(
                "#usda 1.0

{mesh}"
            ),
            &[],
        );
        let info = d.source_info();
        assert_eq!(info.meters_per_unit, Fact::Assumed(0.01));
        assert_eq!(info.unit.as_deref(), Some("centimeter"));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.frame_rate, Fact::Assumed(24.0));
        assert!(info.converted.units && !info.converted.up_axis);
        assert_eq!(info.description, None);
    }

    fn close<const N: usize>(a: [f32; N], b: [f32; N]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5)
    }

    /// A unit quad in the XY plane with `body` in it and `rest` after it, in meters, Y up.
    fn quad(body: &str, rest: &str) -> String {
        format!(
            "#usda 1.0\n(\n    upAxis = \"Y\"\n    metersPerUnit = 1\n)\n\ndef Mesh \"Quad\" (prepend apiSchemas = [\"MaterialBindingAPI\"])\n{{\n    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]\n    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n    uniform token subdivisionScheme = \"none\"\n{body}\n}}\n{rest}"
        )
    }

    #[test]
    fn the_color_texture_reads_the_set_its_primvar_reader_names() {
        let usda = quad(
            r#"    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
    texCoord2f[] primvars:st1 = [(0.25, 0.25), (0.75, 0.75)] (interpolation = "faceVarying")
    int[] primvars:st1:indices = [0, 1, 1, 0]
    rel material:binding = </Looks/M>"#,
            r#"def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Looks/M/T.outputs:rgb>
            token outputs:surface
        }
        def Shader "T"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @c.png@
            float2 inputs:st.connect = </Looks/M/R.outputs:result>
            float3 outputs:rgb
        }
        def Shader "R"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st1"
            float2 outputs:result
        }
    }
}
"#,
        );
        let mesh = flat(&usda, &[("c.png", false)]);
        let prim = &mesh.prims[0];
        assert!(mesh.materials[prim.material].color_image.is_some());
        assert_eq!(prim.uvs.len(), prim.positions.len());
        assert_eq!(prim.uv_sets.len(), 2, "st kept as the second set");
        assert!(prim.uv_sets[0].is_empty(), "slot 0 is `uvs`");
        for (i, p) in prim.positions.iter().enumerate() {
            // The indexed st1 reads 0.25 at x = 0 and 0.75 at x = 1; st is the corner itself.
            let u = 0.25 + 0.5 * p[0];
            assert!(close(prim.uvs[i], [u, 1.0 - u]), "{:?} at {p:?}", prim.uvs[i]);
            assert!(
                close(prim.uv_sets[1][i], [p[0], 1.0 - p[1]]),
                "{:?} at {p:?}",
                prim.uv_sets[1][i]
            );
        }
    }

    #[test]
    fn vertex_colors_stay_under_a_bound_material() {
        let usda = quad(
            r#"    color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] (interpolation = "vertex")
    color4f[] primvars:paint = [(0, 0, 0, 0.5), (0, 0, 0, 0.5), (0, 0, 0, 0.5), (0, 0, 0, 0.5)] (interpolation = "vertex")
    rel material:binding = </Looks/M>"#,
            r#"def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.5, 0.5, 0.5)
            token outputs:surface
        }
    }
}
"#,
        );
        let mesh = flat(&usda, &[]);
        let prim = &mesh.prims[0];
        assert_eq!(mesh.materials[prim.material].name, "M");
        assert!(
            prim.colors.iter().all(|&c| close(c, [1.0; 4])),
            "the material reads no color set, so COLOR_0 doesn't tint it"
        );
        assert_eq!(prim.color_sets.len(), 3);
        assert!(prim.color_sets[0].is_empty(), "slot 0 is `colors`");
        assert_eq!(prim.color_sets[1].len(), prim.positions.len());
        for (c, p) in prim.color_sets[1].iter().zip(&prim.positions) {
            let want = match (p[0] > 0.5, p[1] > 0.5) {
                (false, false) => [1.0, 0.0, 0.0, 1.0],
                (true, false) => [0.0, 1.0, 0.0, 1.0],
                (true, true) => [0.0, 0.0, 1.0, 1.0],
                (false, true) => [1.0, 1.0, 1.0, 1.0],
            };
            assert!(close(*c, want), "{c:?} at {p:?}");
        }
        assert!(prim.color_sets[2].iter().all(|c| close(*c, [0.0, 0.0, 0.0, 0.5])));
    }

    #[test]
    fn authored_tangents_are_kept_with_their_handedness() {
        let usda = quad(
            r#"    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
    float3[] primvars:tangents = [(2, 0, 0)] (interpolation = "constant")
    float3[] primvars:bitangents = [(0, -1, 0)] (interpolation = "constant")"#,
            "",
        );
        let mesh = flat(&usda, &[]);
        let prim = &mesh.prims[0];
        assert_eq!(prim.tangents.len(), prim.positions.len());
        // cross(normal, tangent) is +Y; the bitangent points down, so the frame is mirrored.
        assert!(
            prim.tangents.iter().all(|t| close(*t, [1.0, 0.0, 0.0, -1.0])),
            "{:?}",
            prim.tangents
        );
    }

    #[test]
    fn emissive_occlusion_and_cutoff_are_read() {
        let usda = quad(
            r#"    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
    rel material:binding = </Looks/M>"#,
            r#"def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:emissiveColor.connect = </Looks/M/E.outputs:rgb>
            float inputs:occlusion.connect = </Looks/M/O.outputs:g>
            float inputs:opacity = 1
            float inputs:opacityThreshold = 0.4
            token outputs:surface
        }
        def Shader "E"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @glow.png@
            float3 outputs:rgb
        }
        def Shader "O"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @ao.png@
            float outputs:g
        }
    }
}
"#,
        );
        let d = doc(&usda, &[("glow.png", false), ("ao.png", true)]);
        let m = &d.json()["materials"][0];
        assert!(m.get("emissiveTexture").is_some(), "the emissive texture: {m}");
        assert_eq!(m["emissiveFactor"], json!([1.0, 1.0, 1.0]));
        assert!(
            m.get("occlusionTexture").is_some(),
            "a gray image reads the same in green: {m}"
        );
        assert_eq!(m["alphaMode"], "MASK");
        assert!((m["alphaCutoff"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn a_double_sided_surface_gets_a_double_sided_material() {
        let looks = r#"def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            token outputs:surface
        }
    }
}
def Mesh "Back" (prepend apiSchemas = ["MaterialBindingAPI"])
{
    point3f[] points = [(5, 0, 0), (6, 0, 0), (6, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    uniform bool doubleSided = 1
    rel material:binding = </Looks/M>
}
"#;
        let mesh = flat(&quad("    rel material:binding = </Looks/M>", looks), &[]);
        assert_eq!(mesh.prims.len(), 2);
        assert_eq!(mesh.materials.len(), 2, "one material, one- and two-sided");
        for prim in &mesh.prims {
            let m = &mesh.materials[prim.material];
            assert_eq!(m.name, "M");
            assert_eq!(m.double_sided, prim.positions[0][0] > 4.0);
        }
    }

    #[test]
    fn instances_share_one_mesh() {
        let mut usda = String::from(
            r#"#usda 1.0
def Xform "Root"
{
    class Xform "Proto"
    {
        def Cube "Box"
        {
        }
    }
"#,
        );
        for i in 0..20 {
            write!(
                usda,
                "    def Xform \"I{i}\" (\n        instanceable = true\n        prepend inherits = </Root/Proto>\n    )\n    {{\n        double3 xformOp:translate = ({i}, 0, 0)\n        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }}\n"
            )
            .unwrap();
        }
        usda.push_str("}\n");
        let d = doc(&usda, &[]);
        assert_eq!(d.json()["meshes"].as_array().unwrap().len(), 1);
        let held = d.json()["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| n.get("mesh").is_some())
            .count();
        assert_eq!(held, 20);
    }

    #[test]
    fn past_the_node_cap_the_rest_is_left_out() {
        let mut usda = String::from("#usda 1.0\n");
        for i in 0..10 {
            write!(usda, "def Cube \"C{i}\"\n{{\n}}\n").unwrap();
        }
        let (dir, path) = stage(&usda, &[]);
        let d = load_capped(
            Input::file(&path),
            &crate::assets::Folder::new(dir.path()),
            Keep::ALL,
            4,
        )
        .unwrap();
        assert!(
            d.json()["nodes"].as_array().unwrap().len() <= 5,
            "4 and a root for the units"
        );
        assert!(d.json()["asset"]["extras"]["source"]["truncated"].is_string());
        let mesh = d.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!(mesh.triangles(), 4 * 12, "what was made is kept");
    }

    #[test]
    fn implicit_shapes_have_texture_coordinates() {
        let mesh = flat(
            "#usda 1.0\ndef Cube \"C\"\n{\n}\ndef Sphere \"S\"\n{\n}\ndef Cylinder \"Y\"\n{\n}\ndef Plane \"P\"\n{\n}\n",
            &[],
        );
        assert_ne!(mesh.prims.len(), 0);
        for prim in &mesh.prims {
            assert_eq!(prim.uvs.len(), prim.positions.len());
            assert!(prim.uvs.iter().flatten().all(|v| (0.0..=1.0).contains(v)));
            assert!(prim.uvs.iter().any(|uv| uv[0] > 0.5) && prim.uvs.iter().any(|uv| uv[0] < 0.5));
        }
    }

    #[test]
    fn a_dome_light_is_noted_on_its_node() {
        let d = doc(
            "#usda 1.0\ndef DomeLight \"Sky\"\n{\n    asset inputs:texture:file = @sky.hdr@\n    float inputs:intensity = 2\n}\n",
            &[],
        );
        let sky = d.json()["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == "Sky")
            .unwrap();
        assert_eq!(sky["extras"]["domeLight"]["texture"], "sky.hdr");
        assert_eq!(sky["extras"]["domeLight"]["intensity"], 2.0);
    }

    #[test]
    fn a_udim_texture_reads_its_first_tile() {
        let usda = quad(
            r#"    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
    rel material:binding = </Looks/M>"#,
            r#"def Scope "Looks"
{
    def Material "M"
    {
        token outputs:surface.connect = </Looks/M/S.outputs:surface>
        def Shader "S"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Looks/M/T.outputs:rgb>
            token outputs:surface
        }
        def Shader "T"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @tiles.<UDIM>.png@
            float3 outputs:rgb
        }
    }
}
"#,
        );
        let mesh = flat(&usda, &[("tiles.1003.png", false)]);
        assert!(mesh.materials[mesh.prims[0].material].color_image.is_some());
    }

    #[test]
    fn blend_shapes_carry_their_normal_offsets() {
        let usda = quad(
            r#"    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = "vertex")
    uniform token[] skel:blendShapes = ["lift"]
    rel skel:blendShapeTargets = </Quad/Lift>
    def BlendShape "Lift"
    {
        uniform vector3f[] offsets = [(0, 0, 1)]
        uniform vector3f[] normalOffsets = [(0, 1, 0)]
        uniform int[] pointIndices = [2]
    }"#,
            "",
        )
        .replace("MaterialBindingAPI", "SkelBindingAPI");
        let d = doc(&usda, &[]);
        let target = &d.json()["meshes"][0]["primitives"][0]["targets"][0];
        assert!(
            target.get("POSITION").is_some() && target.get("NORMAL").is_some(),
            "{target}"
        );
    }

    /// A sample drawn on a curve: its curve, the weight of each control point, and whether it
    /// starts the curve and closes it.
    type Drawn = (usize, Vec<f32>, bool, bool);

    /// The samples drawn on curves, and the count sampled on cubic curves.
    fn samples(
        counts: &[i32],
        points: usize,
        cubic: Option<Basis>,
        wrap: Option<&str>,
        budget: usize,
    ) -> (Vec<Drawn>, usize) {
        let mut out = Vec::new();
        let sampled = curve_samples(counts, points, cubic, wrap, budget, &mut |s| {
            let mut w = vec![0.0; points];
            for &(i, x) in &s.terms[..s.n] {
                w[i] += x;
            }
            out.push((s.curve, w, s.starts, s.closes));
        });
        (out, sampled)
    }

    #[test]
    fn curve_samples_by_basis_and_wrap() {
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        // Linear curves are their control points; a periodic one closes from its last.
        let (s, n) = samples(&[3, 2], 5, None, Some("periodic"), 0);
        assert_eq!(n, 0);
        assert_eq!(
            s.iter().map(|s| (s.0, s.2, s.3)).collect::<Vec<_>>(),
            [
                (0, true, false),
                (0, false, false),
                (0, false, true),
                (1, true, false),
                (1, false, true)
            ]
        );
        // A uniform B-spline of five points is two segments of eight steps; it starts at
        // (P0 + 4 P1 + P2) / 6.
        let (s, n) = samples(&[5], 5, Some(Basis::Bspline), None, usize::MAX);
        assert_eq!((s.len(), n), (17, 17));
        assert!(
            close(&s[0].1, &[1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]),
            "{:?}",
            s[0].1
        );
        assert!(s[0].2 && !s[1].2 && s.iter().all(|s| !s.3));
        // Pinned, it runs from its first point to its last.
        let (s, n) = samples(&[3], 3, Some(Basis::Bspline), Some("pinned"), usize::MAX);
        assert_eq!((s.len(), n), (17, 17));
        assert!(close(&s[0].1, &[1.0, 0.0, 0.0]) && close(&s[16].1, &[0.0, 0.0, 1.0]));
        // A periodic Bezier curve of six points is two segments, back to its first point.
        let (s, _) = samples(&[6], 6, Some(Basis::Bezier), Some("periodic"), usize::MAX);
        assert_eq!(s.len(), 17);
        assert!(close(&s[16].1, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        // Too few points for the basis, or past the budget: drawn through the control points.
        for (count, basis, budget) in [
            (5, Basis::Bezier, usize::MAX),
            (2, Basis::Bspline, usize::MAX),
            (5, Basis::Bspline, 16),
        ] {
            let (s, n) = samples(&[count], 5, Some(basis), None, budget);
            assert_eq!((s.len(), n), (usize::try_from(count).unwrap(), 0), "{count} {budget}");
        }
        // A negative count, or a curve past the points, ends the curves.
        assert_eq!(samples(&[2, -1, 2], 6, None, None, 0).0.len(), 2);
        assert_eq!(samples(&[2, 9, 2], 6, None, None, 0).0.len(), 2);
        assert_eq!(
            samples(&[i32::MAX], 6, Some(Basis::CatmullRom), None, usize::MAX)
                .0
                .len(),
            0
        );
    }

    #[test]
    fn hermite_curves_leave_each_point_along_its_tangent() {
        // Three points are two segments; a tangent's index follows the points.
        let mut out = Vec::new();
        let n = curve_samples(&[3], 3, Some(Basis::Hermite), None, usize::MAX, &mut |s| {
            let mut w = [0.0; 6];
            for &(i, x) in &s.terms[..s.n] {
                w[i] += x;
            }
            out.push((w, s.starts, s.tangents, s.ends));
        });
        assert_eq!((out.len(), n), (17, 17));
        assert_eq!(out[0].0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[16].0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        // Halfway along the first: half of each end, and an eighth of each tangent, the second's taken away.
        assert_eq!(out[4].0, [0.5, 0.5, 0.0, 0.125, -0.125, 0.0]);
        assert!(out[0].1 && out.iter().skip(1).all(|o| !o.1 && o.2 == 3));
        assert_eq!(out[12].3, (1, 2, 0.5));
        // Past the budget, through the points.
        let mut drawn = 0;
        assert_eq!(
            curve_samples(&[3], 3, Some(Basis::Hermite), None, 16, &mut |_| drawn += 1),
            0
        );
        assert_eq!(drawn, 3);
    }

    /// The samples drawn on NURBS curves, each its curve, the weight of each control point and whether it starts
    /// the curve, and the count sampled.
    fn nurbs(counts: &[i32], points: usize, nurbs: &Nurbs, budget: usize) -> (Vec<(usize, Vec<f32>, bool)>, usize) {
        let mut out = Vec::new();
        let sampled = nurbs_samples(counts, points, nurbs, budget, &mut |s| {
            let mut w = vec![0.0; points];
            for &(i, x) in &s.terms[..s.n] {
                w[i] += x;
            }
            out.push((s.curve, w, s.starts));
        });
        (out, sampled)
    }

    #[test]
    fn nurbs_curves_sample_their_knot_spans_in_range() {
        let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        let curve = |orders: &[i32], knots: &[f64]| Nurbs {
            orders: orders.to_vec(),
            knots: knots.to_vec(),
            ranges: Vec::new(),
            weights: Vec::new(),
        };
        // A cubic with clamped knots is a Bezier segment: from its first point to its last, through
        // (P0 + 3 P1 + 3 P2 + P3) / 8 halfway.
        let bezier = curve(&[4], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let (s, n) = nurbs(&[4], 4, &bezier, usize::MAX);
        assert_eq!((s.len(), n), (9, 9));
        assert!(close(&s[0].1, &[1.0, 0.0, 0.0, 0.0]) && close(&s[8].1, &[0.0, 0.0, 0.0, 1.0]));
        assert!(close(&s[4].1, &[0.125, 0.375, 0.375, 0.125]), "{:?}", s[4].1);
        assert!(s[0].2 && s.iter().skip(1).all(|s| !s.2));
        // Weights make it rational: a quarter circle, its middle control point weighted by cos 45°, is
        // halfway at (P0 + √2 P1 + P2) / (2 + √2).
        let mut arc = curve(&[3], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        arc.weights = vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0];
        let (s, _) = nurbs(&[3], 3, &arc, usize::MAX);
        let (w, sum) = (std::f32::consts::SQRT_2, 2.0 + std::f32::consts::SQRT_2);
        assert!(close(&s[4].1, &[1.0 / sum, w / sum, 1.0 / sum]), "{:?}", s[4].1);
        // A weight that isn't positive leaves the curve not rational.
        arc.weights[1] = 0.0;
        assert!(close(&nurbs(&[3], 3, &arc, usize::MAX).0[4].1, &[0.25, 0.5, 0.25]));
        // Five points of order two with a repeated knot: three spans drawn (an empty one isn't), and the range
        // cuts off the first half of the first and all of the last.
        let mut line = curve(&[2], &[0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let (s, n) = nurbs(&[5], 5, &line, usize::MAX);
        assert_eq!((s.len(), n), (25, 25));
        line.ranges = vec![[0.5, 2.0]];
        let (s, _) = nurbs(&[5], 5, &line, usize::MAX);
        assert_eq!(s.len(), 17);
        assert!(close(&s[0].1, &[0.5, 0.5, 0.0, 0.0, 0.0]) && close(&s[16].1, &[0.0, 0.0, 1.0, 0.0, 0.0]));
        // A range that isn't one draws the whole.
        line.ranges = vec![[2.0, 1.0]];
        assert_eq!(nurbs(&[5], 5, &line, usize::MAX).0.len(), 25);
        // Orders, knots and ranges that don't make a curve, or past the budget: through the control points. A
        // curve whose knots run out leaves every curve after it so.
        for (orders, knots, ranges, budget) in [
            (&[1][..], &[0.0, 1.0, 2.0, 3.0, 4.0][..], &[][..], usize::MAX),
            (&[9], &[0.0; 13], &[], usize::MAX),
            (&[5], &[0.0; 9], &[], usize::MAX),
            (&[4], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[], 8),
            (
                &[4],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                &[[5.0, 6.0]],
                usize::MAX,
            ),
            (&[4], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0][..7], &[], usize::MAX),
            (&[4], &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[], usize::MAX),
            (&[4], &[0.0, 0.0, 0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0], &[], usize::MAX),
            (&[4], &[0.0; 8], &[], usize::MAX),
            (&[-4], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[], usize::MAX),
            (&[], &[], &[], usize::MAX),
        ] {
            let mut c = curve(orders, knots);
            c.ranges = ranges.to_vec();
            let (s, n) = nurbs(&[4, 4], 8, &c, budget);
            assert_eq!((s.len(), n), (8, 0), "{orders:?} {knots:?} {ranges:?}");
        }
        // The second curve's knots follow the first's.
        let two = curve(&[2, 4], &[0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]);
        let (s, n) = nurbs(&[2, 4], 6, &two, usize::MAX);
        assert_eq!((s.len(), n), (18, 18));
        assert!(s[9].2 && s[9].0 == 1 && close(&s[9].1, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]));
        // A negative count, or a curve past the points, ends the curves.
        assert_eq!(nurbs(&[2, -1, 2], 6, &curve(&[], &[]), usize::MAX).0.len(), 2);
        assert_eq!(nurbs(&[i32::MAX], 6, &bezier, usize::MAX).0.len(), 0);
    }

    /// A stage holding `prims`, in meters, Y up.
    fn stage_of(prims: &str) -> String {
        format!("#usda 1.0\n(\n    upAxis = \"Y\"\n    metersPerUnit = 1\n)\n\n{prims}")
    }

    #[test]
    fn nurbs_and_hermite_curves_are_drawn_as_lines() {
        let usda = stage_of(
            r#"def NurbsCurves "Arc"
{
    int[] curveVertexCounts = [3]
    point3f[] points = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] order = [3]
    double[] knots = [0, 0, 0, 1, 1, 1]
    double[] pointWeights = [1, 0.70710678, 1]
    color3f[] primvars:displayColor = [(1, 0, 0)]
}

def HermiteCurves "Wave"
{
    int[] curveVertexCounts = [2]
    point3f[] points = [(0, 0, 5), (1, 0, 5)]
    vector3f[] tangents = [(0, 4, 0), (0, 4, 0)]
    color3f[] primvars:displayColor = [(0, 0, 1), (0, 1, 0)] (interpolation = "vertex")
}
"#,
        );
        let d = doc(&usda, &[]);
        assert_eq!(d.json()["meshes"].as_array().unwrap().len(), 2);
        let m = d.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!((m.triangles(), m.lines()), (0, 16));
        // The arc lies on the unit circle; the Hermite segment rises and falls between its points.
        let arc = m
            .prims
            .iter()
            .find(|p| p.positions.iter().all(|q| q[2] == 0.0))
            .unwrap();
        assert_eq!(arc.positions.len(), 9);
        assert!(
            arc.positions.iter().all(|q| (q[0].hypot(q[1]) - 1.0).abs() < 1e-5),
            "{:?}",
            arc.positions
        );
        let wave = m
            .prims
            .iter()
            .find(|p| p.positions.iter().all(|q| (q[2] - 5.0).abs() < 1e-6))
            .unwrap();
        assert!(close(wave.positions[4], [0.5, 0.0, 5.0]), "{:?}", wave.positions[4]);
        assert!(
            close(wave.positions[2], [0.15625, 0.375, 5.0]),
            "{:?}",
            wave.positions[2]
        );
        // Its colors run between its points', the tangents adding none.
        let [r, g, b, a] = wave.colors[4];
        assert!(close([r, g, b, a], [0.0, 0.5, 0.5, 1.0]), "{:?}", wave.colors[4]);

        // Tangents that aren't one a point, and knots that don't make a curve: through the points.
        let usda = stage_of(
            r#"def NurbsCurves "Arc"
{
    int[] curveVertexCounts = [3]
    point3f[] points = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] order = [3]
    double[] knots = [0, 0, 1]
}

def HermiteCurves "Wave"
{
    int[] curveVertexCounts = [2]
    point3f[] points = [(0, 0, 5), (1, 0, 5)]
    vector3f[] tangents = [(0, 4, 0)]
}
"#,
        );
        let m = flat(&usda, &[]);
        assert_eq!(
            (m.lines(), m.prims.iter().map(|p| p.positions.len()).sum::<usize>()),
            (3, 5)
        );
    }

    #[test]
    fn hostile_curves_never_panic() {
        let attrs = [
            "int[] curveVertexCounts = [2147483647, -1, 0, 3]",
            "int[] curveVertexCounts = [0, 0, 0, 0, 0, 0, 4]",
            "int[] order = [-2147483648, 2147483647, 0, 1]",
            "int[] order = [4, 4, 4, 4, 4, 4, 4]",
            "double[] knots = [nan, inf, -inf, 1e308, -1e308, 0, 0, 0]",
            "double[] knots = [-1e308, -1e308, -1e308, -1e308, 1e308, 1e308, 1e308, 1e308]",
            "double[] knots = [0, 0, 0, 0, 1e-300, 1e-300, 1e-300, 1e-300]",
            "double2[] ranges = [(nan, 1), (1, -1), (inf, inf)]",
            "double2[] ranges = [(-1e308, 1e308)]",
            "double[] pointWeights = [0, -1, nan, 1e308]",
            "double[] pointWeights = [1e-308, 1e308, 1e-308, 1e308]",
            "vector3f[] tangents = [(1e38, 1e38, 1e38), (nan, 0, 0), (0, 0, 0), (1, 1, 1)]",
            "vector3f[] tangents = []",
            "color3f[] primvars:displayColor = [(1, 0, 0)] (interpolation = \"varying\")",
            "color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0)] (interpolation = \"vertex\")",
        ];
        let points = "point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1e38, 0, 0)]";
        for kind in ["NurbsCurves", "HermiteCurves"] {
            for (i, a) in attrs.iter().enumerate() {
                for b in &attrs[i..] {
                    let usda = stage_of(&format!("def {kind} \"C\"\n{{\n    {points}\n    {a}\n    {b}\n}}\n"));
                    let (dir, path) = stage(&usda, &[]);
                    if let Ok(d) = load(Input::file(&path), &crate::assets::Folder::new(dir.path()), Keep::ALL) {
                        let _flat = d.flatten(Pose::Rest);
                    }
                }
            }
        }
    }
}
