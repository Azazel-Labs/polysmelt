//! `OpenGEX` (`.ogex`), the Open Game Engine Exchange format: a scene written in `OpenDDL`, a
//! tree of nodes (plain, bone, geometry, light and camera nodes) placed by transforms, the
//! objects they reference (meshes, lights, cameras), materials, and animation tracks. The format
//! is read as version 3.0 of its specification gives it, which extends the earlier ones; a file doesn't say
//! which version it was written to, so none is checked or reported.
//!
//! Each node becomes a glTF node, placed by its node transforms (matrices, translations,
//! rotations and scales, multiplied in order); its object transforms, which only its object
//! takes, go on a child node that holds the object; a file with no nodes has each of its objects
//! placed under the root by itself. A geometry object's most detailed mesh
//! becomes a glTF mesh, a primitive for each index array, its material the one the geometry
//! node's material reference with that index names. Points, lines, line strips, triangles,
//! triangle strips and quads are read; positions, normals, tangents (their handedness from the
//! bitangents), texture coordinate sets and color sets keep their indices. Morph targets
//! become glTF morph targets, the one with the lowest index the base, relative targets kept
//! relative to theirs; morph weights the node's weights. Skins become glTF skins, their bind
//! poses inverse bind matrices, each vertex's four strongest bones kept.
//!
//! Materials map onto glTF's metallic-roughness model: diffuse is the base color, metalness
//! and roughness theirs, emission emissive, opacity alpha, transparency transmission,
//! clearcoat, sheen and the index of refraction their extensions, and a material with only a
//! specular color and power (no roughness or metalness) is the classic material every reader
//! reads one way (`legacy_material`): the power its roughness, the color its highlight's.
//! Faces no material reference covers take the default material. Textures keep their coordinate set, addressing
//! and transform. Lights become `KHR_lights_punctual` lights, their attenuation the spot cone
//! and range; cameras glTF perspective cameras. A camera's `fov` (or `fovx`) is its horizontal
//! field of view: with a `fovy` the two give the aspect ratio, and without one a 16:9 frame is
//! assumed, the vertical angle worked out from it and the ratio written beside it.
//!
//! Animation tracks (of transforms and morph weights) become glTF animations, one for each
//! clip. Tracks that map onto a node's translation, rotation or scale, with linear times, are
//! carried over exactly: linear and constant values as they are, Bézier and TCB values as
//! cubic splines. Others (a track inside a stack of transforms glTF can't split, Bézier
//! times) are sampled, at every key and four times between.
//!
//! `OpenGEX` is Z up by default, in its distance metric's units: the document's `RootNode`, its first node, turns it Y up
//! and scales it to meters (and has no transform for a file that is Y up, in meters). Angles and times are scaled by their metrics, and colors are
//! converted from the file's primaries to sRGB's. What has no glTF slot stays in `extras`.
//!
//! The asset's `extras.source` records the distance metric (a meter, assumed, when there's
//! none), the up axis (+Z, assumed, likewise) and the forward axis the file gives, what was
//! converted, the angle and time metrics, primaries other than sRGB's, and counts of what
//! couldn't be read.

mod anim;
mod ddl;
mod geometry;
mod material;
mod math;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::frame::{self, Frame};
use super::xform::col::{self, IDENTITY, M4, mul, scaling, translation};
use super::xform::set_trs;
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Limits, Part};
use crate::images::Images;
use crate::material::Full;
use crate::mesh::FastMap;
use crate::num::f32_of;
use crate::quat;
use crate::source::{Axis, Converted, Fact, Handedness, SourceInfo};
use ddl::{ROOT, Tree};
use math::{is_identity, place, trs};

/// The structures that are nodes.
const NODE_KINDS: [&str; 5] = ["Node", "BoneNode", "GeometryNode", "LightNode", "CameraNode"];
/// The frame a camera is taken to have when the file gives only its horizontal field of view.
const ASSUMED_ASPECT: f64 = 16.0 / 9.0;

/// The units, directions and color space of a file.
struct Metric {
    /// Meters in a unit of distance.
    distance: f64,
    /// Radians in a unit of angle.
    angle: f64,
    /// Seconds in a unit of time.
    time: f64,
    /// Whether +Y is up (else +Z).
    y_up: bool,
    /// The forward axis, when the file gives one.
    forward: Option<Axis>,
    /// Whether the file gives its distance metric, and its up axis.
    has_distance: bool,
    has_up: bool,
    /// The file's red, green and blue primaries and white point, when they aren't sRGB's.
    primaries: Option<[[f64; 2]; 4]>,
    /// The file's linear RGB to linear sRGB.
    to_srgb: Option<[[f64; 3]; 3]>,
}

/// A value the file declares, or else the format's default.
fn fact<T>(declared: bool, v: T) -> Fact<T> {
    if declared { Fact::Declared(v) } else { Fact::Assumed(v) }
}

/// A forward direction as the file writes it: `"x"`, `"-x"`, ... `"-z"`.
fn forward_axis(s: &str) -> Option<Axis> {
    let (negative, letter) = s.strip_prefix('-').map_or((false, s), |l| (true, l));
    let axis = match (letter, negative) {
        ("x", false) => Axis::PosX,
        ("x", true) => Axis::NegX,
        ("y", false) => Axis::PosY,
        ("y", true) => Axis::NegY,
        ("z", false) => Axis::PosZ,
        ("z", true) => Axis::NegZ,
        _ => return None,
    };
    Some(axis)
}

/// A number stored as a `float` as the decimal it was most likely written as (0.01 rather
/// than 0.009999999776482582); other numbers as they are.
fn decimal(v: f64) -> f64 {
    let f = f32_of(v);
    if f64::from(f).to_bits() == v.to_bits() {
        f.to_string().parse().unwrap_or(v)
    } else {
        v
    }
}

impl Metric {
    fn read(t: &Tree<'_>) -> Self {
        let mut m = Metric {
            distance: 1.0,
            angle: 1.0,
            time: 1.0,
            y_up: false,
            forward: None,
            has_distance: false,
            has_up: false,
            primaries: None,
            to_srgb: None,
        };
        let mut prim = SRGB;
        let mut custom = false;
        for k in t.kids(ROOT).filter(|&k| t.ident(k) == "Metric") {
            let (v, _) = t.values(k);
            let scale = v.first().copied().filter(|x| x.is_finite() && *x > 0.0).map(decimal);
            match t.prop_str(k, "key") {
                Some("distance") => {
                    m.distance = scale.unwrap_or(1.0);
                    m.has_distance = scale.is_some();
                }
                Some("angle") => m.angle = scale.unwrap_or(1.0),
                Some("time") => m.time = scale.unwrap_or(1.0),
                Some("up") => {
                    m.y_up = t.string(k) == Some("y");
                    m.has_up = matches!(t.string(k), Some("y" | "z"));
                }
                Some("forward") => m.forward = t.string(k).and_then(forward_axis),
                Some(key @ ("red" | "green" | "blue" | "white")) => {
                    let at = ["red", "green", "blue", "white"]
                        .iter()
                        .position(|&n| n == key)
                        .unwrap_or(0);
                    if v.len() >= 2 && v[..2].iter().all(|x| x.is_finite()) {
                        prim[at] = [v[0], v[1]];
                        custom |= (0..2).any(|i| (prim[at][i] - SRGB[at][i]).abs() > 1e-9);
                    }
                }
                _ => {}
            }
        }
        if custom {
            m.primaries = Some(prim);
            m.to_srgb = rgb_to_xyz(&prim).and_then(|to_xyz| {
                let from_xyz = inverse3(&rgb_to_xyz(&SRGB)?)?;
                Some(mul3(&from_xyz, &to_xyz))
            });
        }
        m
    }

    /// A color from the file's color space to linear sRGB.
    fn color(&self, c: [f64; 3]) -> [f64; 3] {
        match &self.to_srgb {
            Some(m) => [0, 1, 2].map(|r| (0..3).map(|k| m[r][k] * c[k]).sum()),
            None => c,
        }
    }
}

/// The sRGB primaries' and white point's chromaticities.
const SRGB: [[f64; 2]; 4] = [[0.64, 0.33], [0.3, 0.6], [0.15, 0.06], [0.3127, 0.329]];

/// The matrix from linear RGB with primaries and white point `p` to XYZ.
fn rgb_to_xyz(p: &[[f64; 2]; 4]) -> Option<[[f64; 3]; 3]> {
    let xyz = |[x, y]: [f64; 2]| (y.abs() > 1e-12).then(|| [x / y, 1.0, (1.0 - x - y) / y]);
    let cols = [xyz(p[0])?, xyz(p[1])?, xyz(p[2])?];
    let w = xyz(p[3])?;
    let m = [0, 1, 2].map(|r| [cols[0][r], cols[1][r], cols[2][r]]);
    let inv = inverse3(&m)?;
    let s: [f64; 3] = [0, 1, 2].map(|r| (0..3).map(|k| inv[r][k] * w[k]).sum());
    Some([0, 1, 2].map(|r| [0, 1, 2].map(|c| m[r][c] * s[c])))
}

fn mul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

fn inverse3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
    };
    let det: f64 = (0..3).map(|k| m[0][k] * c(0, k)).sum();
    if det.abs() < 1e-300 || !det.is_finite() {
        return None;
    }
    // The inverse is the transposed cofactors over the determinant.
    Some([0, 1, 2].map(|r| [0, 1, 2].map(|k| c(k, r) / det)))
}

/// A component a translation or scale sets: one axis, or all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Comp {
    One(usize),
    All,
}

/// How a rotation is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rot {
    /// An angle about the x, y or z axis.
    Axis(usize),
    /// An angle and an axis.
    AxisAngle,
    Quaternion,
}

/// What a transform structure is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A matrix of this many values (16, 12, 9, 6 or 4).
    Matrix(usize),
    Translate(Comp),
    Rotate(Rot),
    Scale(Comp),
}

impl Kind {
    /// How many values it holds.
    fn width(self) -> usize {
        match self {
            Kind::Matrix(w) => w,
            Kind::Translate(Comp::One(_)) | Kind::Scale(Comp::One(_)) | Kind::Rotate(Rot::Axis(_)) => 1,
            Kind::Translate(Comp::All) | Kind::Scale(Comp::All) => 3,
            Kind::Rotate(_) => 4,
        }
    }
}

/// A transform structure of a node or texture.
#[derive(Clone, Debug)]
struct Xf {
    ddl: u32,
    /// An object transform, which only the node's object takes.
    object: bool,
    kind: Kind,
    /// Its values, as many as its kind holds.
    value: Vec<f64>,
}

/// The transform structures inside `i`, in order; those that can't be read are counted.
fn transforms(t: &Tree<'_>, i: u32, bad: &mut usize) -> Vec<Xf> {
    let mut out = Vec::new();
    for k in t.kids(i) {
        let ident = t.ident(k);
        let comp = |default: Comp| match t.prop_str(k, "kind") {
            Some("x") => Some(Comp::One(0)),
            Some("y") => Some(Comp::One(1)),
            Some("z") => Some(Comp::One(2)),
            Some("xyz") => Some(Comp::All),
            None => Some(default),
            Some(_) => None,
        };
        let (v, w) = t.values(k);
        let kind = match ident {
            "Transform" => {
                let w = if w == 0 { v.len() } else { w as usize };
                [16, 12, 9, 6, 4].contains(&w).then_some(Kind::Matrix(w))
            }
            "Translation" => comp(Comp::All).map(Kind::Translate),
            "Scale" => comp(Comp::All).map(Kind::Scale),
            "Rotation" => match t.prop_str(k, "kind") {
                Some("x") => Some(Kind::Rotate(Rot::Axis(0))),
                Some("y") => Some(Kind::Rotate(Rot::Axis(1))),
                Some("z") => Some(Kind::Rotate(Rot::Axis(2))),
                Some("axis") | None => Some(Kind::Rotate(Rot::AxisAngle)),
                Some("quaternion") => Some(Kind::Rotate(Rot::Quaternion)),
                Some(_) => None,
            },
            _ => continue,
        };
        let Some(kind) = kind.filter(|k| v.len() >= k.width() && v[..k.width()].iter().all(|x| x.is_finite())) else {
            *bad += 1;
            continue;
        };
        out.push(Xf {
            ddl: k,
            object: t.prop_bool(k, "object").unwrap_or(false),
            kind,
            value: v[..kind.width()].to_vec(),
        });
    }
    out
}

/// A matrix of `width` values (column-major, as written) as a 4×4 matrix.
fn expand(v: &[f64], width: usize) -> M4 {
    let mut m = IDENTITY;
    match width {
        16 => m.copy_from_slice(&v[..16]),
        12 => {
            for col in 0..4 {
                m[col * 4..col * 4 + 3].copy_from_slice(&v[col * 3..col * 3 + 3]);
            }
        }
        9 => {
            for col in 0..3 {
                m[col * 4..col * 4 + 3].copy_from_slice(&v[col * 3..col * 3 + 3]);
            }
        }
        6 => {
            // A 2D placement: the first two rows of the first, second and fourth columns.
            m[0..2].copy_from_slice(&v[0..2]);
            m[4..6].copy_from_slice(&v[2..4]);
            m[12..14].copy_from_slice(&v[4..6]);
        }
        4 => {
            m[0..2].copy_from_slice(&v[0..2]);
            m[4..6].copy_from_slice(&v[2..4]);
        }
        _ => {}
    }
    m
}

/// The unit axis an axis-angle rotation's values name, if they name one.
fn axis_of(v: &[f64]) -> Option<[f64; 3]> {
    crate::vec3::unit([v[1], v[2], v[3]])
}

/// The matrix of a transform of `kind` with values `v`, angles in units of `angle` radians.
fn matrix(kind: Kind, v: &[f64], angle: f64) -> M4 {
    match kind {
        Kind::Matrix(w) => expand(v, w),
        Kind::Translate(Comp::One(k)) => {
            let mut t = [0.0; 3];
            t[k] = v[0];
            translation(t)
        }
        Kind::Translate(Comp::All) => translation([v[0], v[1], v[2]]),
        Kind::Scale(Comp::One(k)) => {
            let mut s = [1.0; 3];
            s[k] = v[0];
            scaling(s)
        }
        Kind::Scale(Comp::All) => scaling([v[0], v[1], v[2]]),
        Kind::Rotate(r) => col::rotation(rotation(r, v, angle)),
    }
}

/// The quaternion of a rotation of kind `r` with values `v`.
fn rotation(r: Rot, v: &[f64], angle: f64) -> [f64; 4] {
    match r {
        Rot::Axis(k) => {
            let mut axis = [0.0; 3];
            axis[k] = 1.0;
            quat::about_axis(axis, v[0] * angle)
        }
        Rot::AxisAngle => axis_of(v).map_or([0.0, 0.0, 0.0, 1.0], |a| quat::about_axis(a, v[0] * angle)),
        Rot::Quaternion => quat::unit([v[0], v[1], v[2], v[3]]),
    }
}

/// The product of the transforms `xs`, in order.
fn product<'x>(xs: impl Iterator<Item = &'x Xf>, angle: f64) -> M4 {
    xs.fold(IDENTITY, |m, x| mul(&m, &matrix(x.kind, &x.value, angle)))
}

/// A node as placed: its structure, its glTF node, the glTF node holding its object (itself,
/// or a child placed by its object transforms) and its transforms.
struct Placed {
    ddl: u32,
    node: usize,
    holder: usize,
    xfs: Vec<Xf>,
}

/// A geometry node's morph targets as its glTF node has them: that node, the morph target
/// index of each glTF target, and each target's weight when nothing animates it.
struct Morphs {
    holder: usize,
    targets: Vec<u32>,
    weights: Vec<f64>,
}

/// The state of a load.
struct Ctx<'a> {
    t: &'a Tree<'a>,
    assets: &'a dyn Assets,
    keep: Keep,
    limit: usize,
    dir: PathBuf,
    metric: Metric,
    b: Builder,
    images: Images,
    image_of: FastMap<String, Option<usize>>,
    /// The material structures whose missing textures are counted, when materials aren't kept.
    counted: std::collections::HashSet<u32>,
    materials: Vec<Full>,
    /// How many texture coordinate sets each material reads (its highest set and one).
    material_uvs: Vec<usize>,
    /// The material of faces with none, and the one for those with vertex colors.
    default_materials: [Option<usize>; 2],
    material_of: FastMap<u32, Option<usize>>,
    meshes: Vec<Value>,
    geoms: FastMap<u32, Option<geometry::Geom>>,
    mesh_of: FastMap<(u32, Vec<Option<usize>>), usize>,
    nodes: Vec<Value>,
    kids: Vec<Vec<usize>>,
    node_of: FastMap<u32, usize>,
    placed: Vec<Placed>,
    geometry_nodes: Vec<(u32, usize)>,
    morphs: FastMap<u32, Morphs>,
    skins: Vec<Value>,
    lights: Vec<Value>,
    light_of: FastMap<u32, Option<usize>>,
    cameras: Vec<Value>,
    camera_of: FastMap<u32, Option<usize>>,
    /// The triangles drawn, each placement of a mesh counted.
    triangles: usize,
    /// The triangles built, each mesh counted once; bounded by the limit.
    built_triangles: usize,
    lines_or_points: bool,
    stats: Stats,
}

/// What a load drops or meets, counted.
#[derive(Default)]
struct Stats {
    missing_objects: usize,
    missing_materials: usize,
    missing_textures: usize,
    dropped_lods: usize,
    bad_transforms: usize,
    sheared_transforms: usize,
    bad_arrays: usize,
    unknown_arrays: usize,
    bad_primitives: usize,
    unknown_primitives: usize,
    bad_skins: usize,
    dropped_influences: usize,
    dropped_targets: usize,
    morph_weights_off: usize,
    bad_tracks: usize,
    missing_targets: usize,
    sampled_tracks: usize,
    dropped_keys: usize,
    sheared_textures: usize,
    animated_textures: usize,
    spectra: usize,
}

/// An `OpenDDL` parse error, of the kind other readers give the same failure: text with no structure in it (no
/// brace at all) isn't `OpenGEX`, and a parse that ran into the end of the file with structures still open is a
/// file cut short.
fn same_failure(e: crate::Error, data: &[u8]) -> crate::Error {
    if e.kind() != crate::ErrorKind::Corrupt {
        return e;
    }
    if !data.contains(&b'{') {
        return e.of_kind(crate::ErrorKind::UnknownFormat).within("not an OpenGEX file");
    }
    // Braces outside strings: more opened than closed means the file ends inside a structure.
    let (mut depth, mut quoted, mut escaped) = (0i64, false, false);
    for &c in data {
        match c {
            _ if escaped => escaped = false,
            b'\\' if quoted => escaped = true,
            b'"' => quoted = !quoted,
            b'{' if !quoted => depth += 1,
            b'}' if !quoted => depth -= 1,
            _ => {}
        }
    }
    if depth > 0 || quoted {
        return e
            .of_kind(crate::ErrorKind::Truncated)
            .within("OpenGEX file is cut short");
    }
    e
}

pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let t = Tree::parse(&data).map_err(|e| same_failure(e, &data))?;
    let dir = input.dir().to_path_buf();
    let mut c = Ctx {
        t: &t,
        assets,
        keep,
        limit: input.limit,
        metric: Metric::read(&t),
        b: Builder::new(keep.crease_angle()),
        images: Images::new(!keep.image_bytes, &dir),
        dir,
        image_of: FastMap::default(),
        counted: std::collections::HashSet::new(),
        materials: Vec::new(),
        material_uvs: Vec::new(),
        default_materials: [None; 2],
        material_of: FastMap::default(),
        meshes: Vec::new(),
        geoms: FastMap::default(),
        mesh_of: FastMap::default(),
        nodes: Vec::new(),
        kids: Vec::new(),
        node_of: FastMap::default(),
        placed: Vec::new(),
        geometry_nodes: Vec::new(),
        morphs: FastMap::default(),
        skins: Vec::new(),
        lights: Vec::new(),
        light_of: FastMap::default(),
        cameras: Vec::new(),
        camera_of: FastMap::default(),
        triangles: 0,
        built_triangles: 0,
        lines_or_points: false,
        stats: Stats::default(),
    };
    let scene = nodes(&mut c);
    geometry::meshes(&mut c)?;
    if keep.animations {
        anim::animations(&mut c);
    }
    // A file of nodes alone (a skeleton, or empty groups) is a scene still; one with nothing is not.
    ensure!(
        !c.meshes.is_empty() || !c.lights.is_empty() || !c.cameras.is_empty() || scene,
        Empty,
        "this OpenGEX file has no geometry"
    );
    Ok(finish(c))
}

/// A node's name: its `Name`, or its `OpenDDL` name.
fn node_name(t: &Tree<'_>, n: u32) -> Option<String> {
    t.kid(n, "Name")
        .and_then(|k| t.string(k))
        .or_else(|| t.name(n))
        .map(str::to_owned)
}

/// The structure the reference inside `i`'s first `id` structure names.
fn referenced(t: &Tree<'_>, i: u32, id: &str) -> Option<u32> {
    let k = t.kid(i, id)?;
    let r = t.reference(k)?;
    t.resolve(k, r)
}

/// The structures the tracks of `n`'s animations target.
fn track_targets(t: &Tree<'_>, n: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for a in t.kids(n).filter(|&a| t.ident(a) == "Animation") {
        for tr in t.kids(a).filter(|&k| t.ident(k) == "Track") {
            if let Some(target) = t.prop_ref(tr, "target").and_then(|r| t.resolve(tr, r)) {
                out.push(target);
            }
        }
    }
    out
}

/// What brings the file to glTF's Y up and meters: its `up` and `distance` metrics.
fn frame_of(metric: &Metric) -> Frame {
    let up = if metric.y_up {
        Frame::IDENTITY
    } else {
        Frame::up(Axis::PosZ)
    };
    up.scaled(metric.distance)
}

/// Builds the glTF nodes: the root (turning the file Y up, in meters), then every node. Returns whether that's a
/// scene even without meshes: the file has nodes, or lights or cameras to place without them.
fn nodes(c: &mut Ctx<'_>) -> bool {
    let t = c.t;
    c.nodes.push(frame::node(frame_of(&c.metric)));
    c.kids.push(Vec::new());
    let mut stack: Vec<(u32, usize)> = Vec::new();
    let push_kids = |stack: &mut Vec<(u32, usize)>, n: u32, parent: usize| {
        let at = stack.len();
        stack.extend(
            t.kids(n)
                .filter(|&k| NODE_KINDS.contains(&t.ident(k)))
                .map(|k| (k, parent)),
        );
        stack[at..].reverse();
    };
    push_kids(&mut stack, ROOT, 0);
    while let Some((n, parent)) = stack.pop() {
        let ident = t.ident(n);
        let name = node_name(t, n);
        let xfs = transforms(t, n, &mut c.stats.bad_transforms);
        let targets = if c.keep.animations {
            track_targets(t, n)
        } else {
            Vec::new()
        };
        let animated = |object: bool| xfs.iter().any(|x| x.object == object && targets.contains(&x.ddl));
        let angle = c.metric.angle;
        let node_m = product(xfs.iter().filter(|x| !x.object), angle);
        let object_m = product(xfs.iter().filter(|x| x.object), angle);
        let mut j = json!({});
        if let Some(name) = &name {
            j["name"] = json!(name);
        }
        if animated(false) {
            set_trs(&mut j, trs(&node_m));
        } else if !place(&mut j, &node_m) {
            c.stats.sheared_transforms += 1;
        }
        if c.keep.extras {
            let mut extras = Map::new();
            for key in ["visible", "shadow", "motion_blur"] {
                if let Some(v) = t.prop_bool(n, key) {
                    extras.insert(key.into(), json!(v));
                }
            }
            if let Some(ddl) = t.name(n) {
                extras.insert("ddlName".into(), json!(ddl));
            }
            if ident == "BoneNode" {
                extras.insert("bone".into(), json!(true));
            }
            if !extras.is_empty() {
                j["extras"] = Value::Object(extras);
            }
        }
        let k = attach(c, j, parent);
        c.node_of.insert(n, k);
        let has_object = matches!(ident, "GeometryNode" | "LightNode" | "CameraNode");
        let holder = if has_object && (animated(true) || !is_identity(&object_m)) {
            let mut o = json!({ "name": format!("{} object", name.as_deref().unwrap_or("node")) });
            if animated(true) {
                set_trs(&mut o, trs(&object_m));
            } else if !place(&mut o, &object_m) {
                c.stats.sheared_transforms += 1;
            }
            attach(c, o, k)
        } else {
            k
        };
        match ident {
            "GeometryNode" => c.geometry_nodes.push((n, holder)),
            "LightNode" if c.keep.lights => {
                if let Some(l) = referenced(t, n, "ObjectRef").and_then(|o| light(c, o)) {
                    c.nodes[holder]["extensions"] = json!({ "KHR_lights_punctual": { "light": l } });
                } else {
                    c.stats.missing_objects += 1;
                }
            }
            "CameraNode" if c.keep.cameras => {
                if let Some(cam) = referenced(t, n, "ObjectRef").and_then(|o| camera(c, o)) {
                    c.nodes[holder]["camera"] = json!(cam);
                } else {
                    c.stats.missing_objects += 1;
                }
            }
            _ => {}
        }
        c.placed.push(Placed {
            ddl: n,
            node: k,
            holder,
            xfs,
        });
        push_kids(&mut stack, n, k);
    }
    c.nodes.len() > 1 || objects(c)
}

/// Places a file's objects when it has no nodes to place them: each under the root, untransformed, so a file of
/// only cameras or lights (or meshes) still shows them. Returns whether it placed a light or camera.
fn objects(c: &mut Ctx<'_>) -> bool {
    let t = c.t;
    let mut seen = false;
    for o in t.kids(ROOT).collect::<Vec<_>>() {
        let ident = t.ident(o);
        let kind = match ident {
            "GeometryObject" => "geometry",
            "LightObject" => "light",
            "CameraObject" => "camera",
            _ => continue,
        };
        seen |= kind != "geometry";
        let name = node_name(t, o).unwrap_or_else(|| kind.to_owned());
        let k = attach(c, json!({ "name": name }), 0);
        match ident {
            "GeometryObject" => c.geometry_nodes.push((o, k)),
            "LightObject" if c.keep.lights => {
                if let Some(l) = light(c, o) {
                    c.nodes[k]["extensions"] = json!({ "KHR_lights_punctual": { "light": l } });
                }
            }
            "CameraObject" if c.keep.cameras => {
                if let Some(cam) = camera(c, o) {
                    c.nodes[k]["camera"] = json!(cam);
                }
            }
            _ => {}
        }
    }
    seen
}

/// Adds a glTF node under `parent`; returns its index.
fn attach(c: &mut Ctx<'_>, j: Value, parent: usize) -> usize {
    c.nodes.push(j);
    c.kids.push(Vec::new());
    let k = c.nodes.len() - 1;
    c.kids[parent].push(k);
    k
}

/// The `Param` inside `i` with attribute `attrib`.
fn param(t: &Tree<'_>, i: u32, attrib: &str) -> Option<f64> {
    t.kids(i)
        .filter(|&k| t.ident(k) == "Param" && t.prop_str(k, "attrib") == Some(attrib))
        .find_map(|k| t.values(k).0.first().copied())
        .filter(|v| v.is_finite())
}

/// The `Color` inside `i` with attribute `attrib`, RGBA in linear sRGB.
fn color(c: &Ctx<'_>, i: u32, attrib: &str) -> Option<[f64; 4]> {
    let t = c.t;
    t.kids(i)
        .filter(|&k| t.ident(k) == "Color" && t.prop_str(k, "attrib") == Some(attrib))
        .find_map(|k| {
            let (v, _) = t.values(k);
            (v.len() >= 3 && v.iter().take(4).all(|x| x.is_finite())).then(|| {
                let rgb = c.metric.color([v[0], v[1], v[2]]);
                [rgb[0], rgb[1], rgb[2], v.get(3).copied().unwrap_or(1.0)]
            })
        })
}

/// A light object as a `KHR_lights_punctual` light; its index.
fn light(c: &mut Ctx<'_>, o: u32) -> Option<usize> {
    if let Some(&l) = c.light_of.get(&o) {
        return l;
    }
    let t = c.t;
    let out = (t.ident(o) == "LightObject").then(|| {
        let kind = t.prop_str(o, "type").unwrap_or("point");
        let rgb = color(c, o, "light").map_or([1.0; 3], |v| [v[0], v[1], v[2]]);
        // glTF's color is at most 1; what's brighter goes into the intensity.
        let top = rgb.iter().copied().fold(0.0f64, f64::max);
        let (rgb, gain) = if top > 1.0 {
            (rgb.map(|v| v / top), top)
        } else {
            (rgb.map(|v| v.max(0.0)), 1.0)
        };
        let key = if kind == "infinite" { "illuminance" } else { "intensity" };
        let intensity = param(t, o, key)
            .or_else(|| param(t, o, if kind == "infinite" { "intensity" } else { "illuminance" }))
            .unwrap_or(1.0)
            .max(0.0)
            * gain;
        let mut l = json!({ "color": rgb, "intensity": intensity });
        let mut extras = Map::new();
        extras.insert("type".into(), json!(kind));
        if let Some(s) = t.prop_bool(o, "shadow") {
            extras.insert("shadow".into(), json!(s));
        }
        let mut cone: Option<(f64, f64)> = None;
        let mut range: Option<f64> = None;
        let mut attens = Vec::new();
        for a in t.kids(o).filter(|&k| t.ident(k) == "Atten") {
            let akind = t.prop_str(a, "kind").unwrap_or("distance");
            let curve = t.prop_str(a, "curve").unwrap_or("linear");
            let p = |name: &str| param(t, a, name);
            let mut e = json!({ "kind": akind, "curve": curve });
            for name in [
                "begin",
                "end",
                "scale",
                "offset",
                "constant",
                "linear",
                "quadratic",
                "power",
            ] {
                if let Some(v) = p(name) {
                    e[name] = json!(v);
                }
            }
            attens.push(e);
            let (begin, end) = (p("begin").unwrap_or(0.0), p("end").unwrap_or(1.0));
            match akind {
                "angle" => cone = Some((begin * c.metric.angle, end * c.metric.angle)),
                "cos_angle" => cone = Some((begin.clamp(-1.0, 1.0).acos(), end.clamp(-1.0, 1.0).acos())),
                _ if matches!(curve, "linear" | "smooth") && end > 0.0 => range = Some(end * c.metric.distance),
                _ => {}
            }
        }
        if !attens.is_empty() {
            extras.insert("attenuation".into(), json!(attens));
        }
        let spectra = t.kids(o).filter(|&k| t.ident(k) == "Spectrum").count();
        c.stats.spectra += spectra;
        if let Some(tex) = t.kids(o).find(|&k| t.ident(k) == "Texture")
            && let Some(f) = t.string(tex)
        {
            extras.insert("projection".into(), json!(f));
        }
        match kind {
            "infinite" => l["type"] = json!("directional"),
            "spot" => {
                l["type"] = json!("spot");
                let half = std::f64::consts::FRAC_PI_2;
                let (inner, outer) = cone.unwrap_or((0.0, std::f64::consts::FRAC_PI_4));
                let outer = if outer.is_finite() {
                    outer.clamp(1e-4, half)
                } else {
                    std::f64::consts::FRAC_PI_4
                };
                let inner = if inner.is_finite() {
                    inner.clamp(0.0, outer * 0.9999)
                } else {
                    0.0
                };
                l["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
            }
            _ => l["type"] = json!("point"),
        }
        if kind != "infinite"
            && let Some(r) = range.filter(|r| r.is_finite() && *r > 0.0)
        {
            l["range"] = json!(r);
        }
        if let Some(name) = t.name(o) {
            l["name"] = json!(name);
        }
        if c.keep.extras {
            l["extras"] = json!({ "opengex": extras });
        }
        c.lights.push(l);
        c.lights.len() - 1
    });
    c.light_of.insert(o, out);
    out
}

/// A camera object as a glTF perspective camera; its index.
fn camera(c: &mut Ctx<'_>, o: u32) -> Option<usize> {
    if let Some(&cam) = c.camera_of.get(&o) {
        return cam;
    }
    let t = c.t;
    let out = (t.ident(o) == "CameraObject").then(|| {
        let angle = |name: &str| {
            param(t, o, name)
                .map(|v| v * c.metric.angle)
                .filter(|v| *v > 0.0 && *v < std::f64::consts::PI)
        };
        let fovx = angle("fovx").or_else(|| angle("fov"));
        let fovy = angle("fovy");
        let mut p = Map::new();
        let mut extras = Map::new();
        match (fovx, fovy) {
            (Some(x), Some(y)) => {
                p.insert("yfov".into(), json!(y));
                let aspect = (x * 0.5).tan() / (y * 0.5).tan();
                if aspect.is_finite() && aspect > 0.0 {
                    p.insert("aspectRatio".into(), json!(aspect));
                }
            }
            (None, Some(y)) => {
                p.insert("yfov".into(), json!(y));
            }
            (Some(x), None) => {
                // The horizontal field of view alone ("fov" and "fovx" are both it): glTF wants the
                // vertical, which needs the aspect ratio the file leaves to the viewer. A 16:9 frame
                // is assumed and written with it, so the two together are the file's angle across.
                p.insert("yfov".into(), json!(2.0 * ((x * 0.5).tan() / ASSUMED_ASPECT).atan()));
                p.insert("aspectRatio".into(), json!(ASSUMED_ASPECT));
                extras.insert("horizontalFov".into(), json!(x));
            }
            (None, None) => {
                p.insert("yfov".into(), json!(std::f64::consts::FRAC_PI_4));
            }
        }
        let d = c.metric.distance;
        let near = param(t, o, "near").map(|v| v * d).filter(|v| *v > 0.0).unwrap_or(0.1);
        p.insert("znear".into(), json!(near));
        if let Some(far) = param(t, o, "far").map(|v| v * d).filter(|v| v.is_finite() && *v > near) {
            p.insert("zfar".into(), json!(far));
        }
        let mut cam = json!({ "type": "perspective", "perspective": p });
        if let Some(name) = t.name(o) {
            cam["name"] = json!(name);
        }
        if c.keep.extras && !extras.is_empty() {
            cam["extras"] = json!({ "opengex": extras });
        }
        c.cameras.push(cam);
        c.cameras.len() - 1
    });
    c.camera_of.insert(o, out);
    out
}

/// A float accessor of `kind` (`SCALAR`, `VEC3`, `VEC4` or `MAT4`) with no target, for skins
/// and animations; `bounds` adds the `min` and `max` an animation's times need.
fn floats(b: &mut Builder, v: Vec<f32>, kind: &'static str, bounds: bool) -> usize {
    let width = match kind {
        "VEC3" => 3,
        "VEC4" => 4,
        "MAT4" => 16,
        _ => 1,
    };
    let count = v.len() / width;
    let range = bounds.then(|| {
        let mut min = vec![f32::MAX; width];
        let mut max = vec![f32::MIN; width];
        for e in v.chunks_exact(width) {
            for k in 0..width {
                min[k] = min[k].min(e[k]);
                max[k] = max[k].max(e[k]);
            }
        }
        (min, max)
    });
    let a = b.accessor(Part::f32(v), 5126, count, kind, None);
    if let Some((min, max)) = range {
        b.accessors[a].min = Some(Limits::floats(&min));
        b.accessors[a].max = Some(Limits::floats(&max));
    }
    a
}

fn finish(mut c: Ctx<'_>) -> Document {
    let mut nodes = std::mem::take(&mut c.nodes);
    for (k, kids) in c.kids.iter().enumerate() {
        if !kids.is_empty() {
            nodes[k]["children"] = json!(kids);
        }
    }
    let m = &c.metric;
    let mut source = Map::new();
    source.insert("radiansPerUnit".into(), json!(m.angle));
    source.insert("secondsPerUnit".into(), json!(m.time));
    if let Some(p) = &m.primaries {
        source.insert(
            "primaries".into(),
            json!({ "red": p[0], "green": p[1], "blue": p[2], "white": p[3] }),
        );
    }
    SourceInfo {
        format: Some("OpenGEX".into()),
        // A unit is a meter, and +Z is up, unless the file's metrics say otherwise.
        meters_per_unit: fact(m.has_distance, m.distance),
        unit: crate::source::unit_name(m.distance).map(str::to_owned),
        up_axis: fact(m.has_up, if m.y_up { Axis::PosY } else { Axis::PosZ }),
        forward_axis: m.forward.map_or(Fact::Unknown, Fact::Declared),
        handedness: Fact::Declared(Handedness::Right),
        converted: Converted {
            units: (m.distance - 1.0).abs() > 0.0,
            up_axis: !m.y_up,
            handedness: false,
        },
        ..SourceInfo::default()
    }
    .write(&mut source);
    let st = &c.stats;
    for (key, n) in [
        ("missingObjects", st.missing_objects),
        ("missingMaterials", st.missing_materials),
        ("droppedTextures", st.missing_textures),
        ("droppedLevelsOfDetail", st.dropped_lods),
        ("badTransforms", st.bad_transforms),
        ("shearedTransforms", st.sheared_transforms),
        ("badArrays", st.bad_arrays),
        ("unknownArrays", st.unknown_arrays),
        (crate::geometry::DROPPED, st.bad_primitives),
        ("unknownPrimitives", st.unknown_primitives),
        ("badSkins", st.bad_skins),
        ("droppedInfluences", st.dropped_influences),
        ("droppedMorphTargets", st.dropped_targets),
        ("morphWeightsNotSummingToOne", st.morph_weights_off),
        ("badTracks", st.bad_tracks),
        ("missingTrackTargets", st.missing_targets),
        ("sampledTracks", st.sampled_tracks),
        ("droppedKeys", st.dropped_keys),
        ("shearedTextureTransforms", st.sheared_textures),
        ("animatedTextureTransforms", st.animated_textures),
        ("spectra", st.spectra),
        ("stateIdentifiers", c.t.states),
    ] {
        if n > 0 {
            source.insert(key.into(), json!(n));
        }
    }
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(c.triangles));
    facts.insert("nodes".into(), json!(nodes.len()));
    // The file's materials: the default one that faces with none take isn't one of them.
    let defaults = c.default_materials.iter().flatten().count();
    facts.insert("materials".into(), json!(c.materials.len() - defaults));

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [0] }]));
    crate::skin::set_skeletons(&nodes, &mut c.skins);
    json.insert("nodes".into(), Value::Array(nodes));
    if !c.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut c.meshes)));
    }
    if !c.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut c.skins)));
    }
    if !c.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut c.cameras)));
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
    let lines_or_points = c.lines_or_points;
    let frame = frame_of(&c.metric);
    let mut doc = c.b.finish(json, facts);
    frame::placed(&mut doc, frame);
    if lines_or_points {
        doc.set_lines_and_points(true);
    }
    doc
}
