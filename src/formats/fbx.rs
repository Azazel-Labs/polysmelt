//! FBX (binary 7.x, and ASCII) as a glTF document: the model hierarchy with each model's transform (pivots, offsets,
//! pre- and post-rotation, rotation order; geometric transforms on a child node), meshes in their own space and shared
//! between instances, materials and their textures (embedded or found in the pack), skins from skin clusters, morph
//! targets from blend shapes, animation stacks sampled into translation, rotation, scale and weights channels,
//! cameras, lights, and user properties as node extras. The file's axis system and unit become a root node that
//! carries them to glTF's Y-up meters, and are recorded in `asset.extras.source` (see [`crate::source`]), with its
//! frame rate (`TimeMode`), the applications that made and last saved it, `SceneInfo`'s title, author, subject,
//! keywords and comment, and when it was made (`CreationTimeStamp`) and last saved. `coordAxis` (the file's right
//! axis), `creator` (the SDK that wrote it, when an application is named), `revision`, `originalApplication` and
//! `originalUnitScaleFactor` are kept as the file gives them.
//!
//! `Line` geometry is drawn as line segments, in its model's first material or else unlit in its own color, and so
//! is `NurbsCurve` geometry: each knot span in up to 32 segments, by how far it turns. NURBS surfaces and patches,
//! and curves with no order or points, aren't read; `asset.extras.source.skipped` counts them by kind.
//!
//! Colors are sRGB, as the FBX SDK's tools write them, except that Blender writes material colors linear; vertex
//! colors are sRGB from every tool, Blender included (its exporter's default).

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use super::fbx_tree::{Attr as A, NodeHandle};
use super::nurbs;
use crate::{Context as _, ErrorKind, Result};
use serde_json::{Map, Value, json};

use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::{Images, Location};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Sheen, Tex, UvTransform};
use crate::mesh::{FastMap, GRAY, Material, Prim, smooth_normals, srgb_to_linear, triangulate};
use crate::num::f32_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::vec3;

use super::xform::{IDENTITY, M4, Trs, column_major, decompose, det3, inverse, mul, scale, set_transform, translate};

/// Euler rotation in degrees, FBX rotation order (0 = XYZ: X applied first).
fn rotate(deg: [f64; 3], order: i32) -> M4 {
    let [x, y, z] = deg.map(f64::to_radians);
    let rx = [
        [1., 0., 0., 0.],
        [0., x.cos(), -x.sin(), 0.],
        [0., x.sin(), x.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let ry = [
        [y.cos(), 0., y.sin(), 0.],
        [0., 1., 0., 0.],
        [-y.sin(), 0., y.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let rz = [
        [z.cos(), -z.sin(), 0., 0.],
        [z.sin(), z.cos(), 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    match order {
        1 => mul(&ry, &mul(&rz, &rx)), // XZY
        2 => mul(&rx, &mul(&rz, &ry)), // YZX
        3 => mul(&rz, &mul(&rx, &ry)), // YXZ
        4 => mul(&ry, &mul(&rx, &rz)), // ZXY
        5 => mul(&rx, &mul(&ry, &rz)), // ZYX
        _ => mul(&rz, &mul(&ry, &rx)), // XYZ
    }
}

fn floats(v: &A) -> Option<Vec<f64>> {
    match v {
        A::ArrF64(a) => a.to_vec(),
        A::ArrF32(a) => a.map(f64::from),
        _ => None,
    }
}

fn ints(v: &A) -> Option<Vec<i64>> {
    match v {
        A::ArrI32(a) => a.map(i64::from),
        A::ArrI64(a) => a.to_vec(),
        _ => None,
    }
}

/// An object ID. FBX writes them as 64-bit integers, often hashes that use every bit, so they
/// are read as integers: through a double, IDs past 2^53 would round and collide.
fn id(v: &A) -> Option<i64> {
    match v {
        A::I64(i) => Some(*i),
        A::I32(i) => Some(i64::from(*i)),
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a whole number written as a float; `as` saturates"
        )]
        v => num(v).map(|f| f as i64),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "a property value as a double, like the FBX SDK reads it"
)]
fn num(v: &A) -> Option<f64> {
    match v {
        A::F64(f) => Some(*f),
        A::F32(f) => Some(f64::from(*f)),
        A::I32(i) => Some(f64::from(*i)),
        A::I64(i) => Some(*i as f64),
        A::I16(i) => Some(f64::from(*i)),
        A::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn string<'a>(v: &'a A<'_>) -> Option<&'a str> {
    if let A::String(s) = v { Some(s.as_ref()) } else { None }
}

fn child_attr<'a>(node: NodeHandle<'a>, name: &str) -> Option<&'a A<'a>> {
    node.first_child_by_name(name)?.attributes().first()
}

/// `Properties70` → name → values (after the name, type, label and flags strings).
fn props(node: NodeHandle<'_>) -> HashMap<String, Vec<A<'_>>> {
    let mut out = HashMap::new();
    if let Some(p70) = node.first_child_by_name("Properties70") {
        for p in p70.children_by_name("P") {
            let a = p.attributes();
            if let Some(name) = a.first().and_then(string) {
                out.insert(name.to_owned(), a.get(4..).unwrap_or_default().to_vec());
            }
        }
    }
    out
}

/// An object's properties: its own `Properties70`, over the defaults its class's `Definitions` template gives.
#[derive(Default)]
struct Props<'t> {
    own: HashMap<String, Vec<A<'t>>>,
    template: Option<&'t HashMap<String, Vec<A<'t>>>>,
}

impl<'t> Props<'t> {
    /// An object's own properties, without a template.
    fn of(node: NodeHandle<'t>) -> Self {
        Self {
            own: props(node),
            template: None,
        }
    }

    fn get(&self, name: &str) -> Option<&Vec<A<'t>>> {
        self.own.get(name).or_else(|| self.template?.get(name))
    }

    /// A number the object gives itself, ignoring its template.
    fn own_scalar(&self, name: &str) -> Option<f64> {
        self.own
            .get(name)
            .and_then(|v| v.first())
            .and_then(num)
            .filter(|v| v.is_finite())
    }

    /// A string property.
    fn text(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(|v| v.first()).and_then(string)
    }
}

/// A vector property; a component that isn't a finite number takes the default's.
fn vec3(p: &Props<'_>, name: &str, default: [f64; 3]) -> [f64; 3] {
    match p.get(name) {
        Some(v) if v.len() >= 3 => std::array::from_fn(|k| num(&v[k]).filter(|x| x.is_finite()).unwrap_or(default[k])),
        _ => default,
    }
}

/// A number property, when it's there and finite.
fn scalar(p: &Props<'_>, name: &str) -> Option<f64> {
    p.get(name)
        .and_then(|v| v.first())
        .and_then(num)
        .filter(|v| v.is_finite())
}

/// Whether a flag property is set, `default` when the property isn't there.
fn flag(p: &Props<'_>, name: &str, default: bool) -> bool {
    scalar(p, name).map_or(default, |v| v != 0.0)
}

/// An enum property, `default` when it isn't there.
fn choice(p: &Props<'_>, name: &str, default: i32) -> i32 {
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    scalar(p, name).map_or(default, |v| v as i32)
}

/// Whether a model's own properties hide it: `Visibility` under one half, or `Show` off.
fn hidden(p: &Props<'_>) -> bool {
    scalar(p, "Visibility").is_some_and(|v| v < 0.5) || !flag(p, "Show", true)
}

/// A model's rotation order. Without `RotationActive` the order, and the pre- and post-rotations, are ignored: the
/// rotation is XYZ. A file that doesn't say is taken to mean them.
fn rotation_order(p: &Props<'_>) -> i32 {
    if flag(p, "RotationActive", true) {
        choice(p, "RotationOrder", 0)
    } else {
        0
    }
}

/// A layer element's values and how they map onto polygon vertices.
struct Layer {
    values: Vec<f64>,
    index: Option<Vec<i64>>,
    mapping: Mapping,
    width: usize,
}

/// What a layer element holds a value for.
#[derive(Clone, Copy)]
enum Mapping {
    ControlPoint,
    Polygon,
    AllSame,
    PolygonVertex,
}

impl Mapping {
    fn of(el: NodeHandle<'_>, default: Self) -> Self {
        match child_attr(el, "MappingInformationType").and_then(string) {
            Some("ByVertice" | "ByVertex" | "ByControlPoint") => Self::ControlPoint,
            Some("ByPolygon") => Self::Polygon,
            Some("AllSame") => Self::AllSame,
            Some(_) => Self::PolygonVertex,
            None => default,
        }
    }
}

impl Layer {
    fn read(geom: NodeHandle<'_>, element: &str, values: &str, index: &str, width: usize) -> Option<Self> {
        let el = geom
            .children_by_name(element)
            .find(|e| e.attributes().first().and_then(num).unwrap_or(0.0) == 0.0)?;
        Self::of(el, values, index, width)
    }

    /// Every layer of `element` (UV sets, color sets) with its node, by its number, lowest first: the first is
    /// the one [`Layer::read`] gives. One that can't be read is left out.
    fn read_all<'a>(
        geom: NodeHandle<'a>,
        element: &str,
        values: &str,
        index: &str,
        width: usize,
    ) -> Vec<(NodeHandle<'a>, Self)> {
        let mut all: Vec<(f64, NodeHandle<'a>, Self)> = geom
            .children_by_name(element)
            .filter_map(|el| {
                let n = el.attributes().first().and_then(num).unwrap_or(0.0);
                Some((n, el, Self::of(el, values, index, width)?))
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.into_iter().map(|(_, el, l)| (el, l)).collect()
    }

    fn of(el: NodeHandle<'_>, values: &str, index: &str, width: usize) -> Option<Self> {
        Some(Self {
            values: floats(child_attr(el, values)?)?,
            index: child_attr(el, index).and_then(ints),
            mapping: Mapping::of(el, Mapping::PolygonVertex),
            width,
        })
    }

    /// The value for polygon-vertex `pv` (control point `cp`, polygon `poly`).
    fn get(&self, pv: usize, cp: usize, poly: usize) -> Option<(usize, &[f64])> {
        let i = match self.mapping {
            Mapping::ControlPoint => cp,
            Mapping::Polygon => poly,
            Mapping::AllSame => 0,
            Mapping::PolygonVertex => pv,
        };
        let i = match &self.index {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        Some((i, self.values.get(i * self.width..i * self.width + self.width)?))
    }
}

/// FBX time units per second.
const KTIME: f64 = 46_186_158_000.0;

/// The most points NURBS curves add to a file; past them, a curve is drawn through its control points. Each control
/// point is a few bytes of the file but up to 32 segments, so this bounds what a file of curves costs.
const MAX_CURVE_POINTS: usize = 1 << 22;

/// The most degrees a rotation curve may turn between two samples: FBX interpolates Euler
/// angles, glTF quaternions, so long turns are cut into steps short enough to agree.
const MAX_STEP_DEGREES: f64 = 10.0;

/// The scene's frame rate, from `TimeMode` (the FBX SDK's `EMode`), or `CustomFrameRate` when that's the mode.
fn frame_rate(settings: &Props<'_>) -> Fact<f64> {
    const RATES: [f64; 19] = [
        -1.0,
        120.0,
        100.0,
        60.0,
        50.0,
        48.0,
        30.0,
        30.0,
        30.0 / 1.001,
        30.0 / 1.001,
        25.0,
        24.0,
        1000.0,
        24.0 / 1.001,
        -1.0,
        96.0,
        72.0,
        60.0 / 1.001,
        120.0 / 1.001,
    ];
    if scalar(settings, "TimeMode").is_none() {
        return Fact::Unknown;
    }
    let custom = scalar(settings, "CustomFrameRate").filter(|r| *r > 0.0);
    match usize::try_from(choice(settings, "TimeMode", 0))
        .ok()
        .and_then(|m| RATES.get(m))
    {
        Some(&r) if r > 0.0 => Fact::Declared(r),
        Some(_) => custom.map_or(Fact::Unknown, Fact::Declared),
        None => Fact::Unknown,
    }
}

/// An FBX date as ISO 8601: `DD/MM/YYYY HH:MM:SS.mmm` in UTC, as `SceneInfo` writes them. `None` for the epoch,
/// which exporters write when they don't know.
fn fbx_date(s: &str) -> Option<String> {
    let s = s.trim();
    let parse = || -> Option<String> {
        let (date, time) = s.split_once(' ')?;
        let mut d = date.split('/').map(|p| p.parse::<u32>().ok());
        let (day, month, year) = (d.next()??, d.next()??, d.next()??);
        let (hms, ms) = time.split_once('.').unwrap_or((time, ""));
        let mut t = hms.split(':').map(|p| p.parse::<u32>().ok());
        let (h, mi, sec) = (t.next()??, t.next()??, t.next()??);
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) || h > 23 || mi > 59 || sec > 60 {
            return None;
        }
        let ms = ms.parse::<u32>().ok().filter(|ms| *ms > 0);
        Some(match ms {
            Some(ms) => format!("{year:04}-{month:02}-{day:02}T{h:02}:{mi:02}:{sec:02}.{ms:03}Z"),
            None => format!("{year:04}-{month:02}-{day:02}T{h:02}:{mi:02}:{sec:02}Z"),
        })
    };
    match parse() {
        Some(iso) if iso.starts_with("1970-01-01T00:00:00") => None,
        Some(iso) => Some(iso),
        None => (!s.is_empty()).then(|| s.to_owned()),
    }
}

/// What the file's header says of it: `SceneInfo`'s `MetaData` (title, author, subject, keywords, comment,
/// revision) and the applications that made and last saved it, and `CreationTimeStamp`.
fn scene_info(root: NodeHandle<'_>, creator: &str, info: &mut SourceInfo, details: &mut Map<String, Value>) {
    let header = root.first_child_by_name("FBXHeaderExtension");
    let scene = header.and_then(|h| h.first_child_by_name("SceneInfo"));
    let text = |n: Option<NodeHandle<'_>>, name: &str| {
        n.and_then(|n| child_attr(n, name))
            .and_then(string)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    let meta = scene.and_then(|s| s.first_child_by_name("MetaData"));
    info.title = text(meta, "Title");
    info.author = text(meta, "Author");
    info.description = text(meta, "Subject");
    info.keywords = text(meta, "Keywords")
        .map(|k| {
            k.split([',', ';'])
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    info.comments = text(meta, "Comment").into_iter().collect();
    if let Some(r) = text(meta, "Revision") {
        details.insert("revision".into(), json!(r));
    }
    let props = scene.map(Props::of).unwrap_or_default();
    let app = |when: &str| {
        let name = props.text(&format!("{when}|ApplicationName")).map_or("", str::trim);
        let version = props.text(&format!("{when}|ApplicationVersion")).map_or("", str::trim);
        (!name.is_empty()).then(|| format!("{name} {version}").trim_end().to_owned())
    };
    let saved = app("LastSaved");
    let original = app("Original");
    info.tool = saved.clone().or_else(|| original.clone());
    if saved.is_some() && original.is_some() && original != saved {
        details.insert("originalApplication".into(), json!(original));
    }
    if !creator.is_empty() {
        if info.tool.is_some() {
            details.insert("creator".into(), json!(creator));
        } else {
            info.tool = Some(creator.to_owned());
        }
    }
    info.modified = props.text("LastSaved|DateTime_GMT").and_then(fbx_date);
    let stamp = header.and_then(|h| h.first_child_by_name("CreationTimeStamp"));
    let part = |name: &str| {
        stamp
            .and_then(|s| child_attr(s, name))
            .and_then(num)
            .filter(|v| v.is_finite() && *v >= 0.0 && *v < 1e5)
    };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small, non-negative numbers"
    )]
    let created = match ["Year", "Month", "Day", "Hour", "Minute", "Second"].map(|k| part(k).map(|v| v as u32)) {
        [Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(s)]
            if (1..=12).contains(&mo) && (1..=31).contains(&d) && h < 24 && mi < 60 && s <= 60 =>
        {
            Some(format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}"))
        }
        _ => None,
    };
    info.created = created
        .or_else(|| props.text("Original|DateTime_GMT").and_then(fbx_date))
        .filter(|c| !c.starts_with("1970-01-01T00:00:00"));
}

/// Reads an FBX file (binary 7.x, or ASCII) as a glTF document.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    // A file read is held as shared bytes, so that a big texture embedded in it can point into
    // it rather than be copied.
    let (borrowed, file) = match input.read()? {
        Cow::Borrowed(b) => (b, None),
        Cow::Owned(v) => (&[][..], Some(crate::Bytes::from(v))),
    };
    let data: &[u8] = file.as_deref().unwrap_or(borrowed);
    let tree = super::fbx_tree::Tree::parse(data)?;
    let version = tree.version();
    let rootnode = tree.root();
    let creator = |n: Option<NodeHandle<'_>>| {
        n.and_then(|n| n.first_child_by_name("Creator"))
            .and_then(|c| c.attributes().first().and_then(string).map(str::to_owned))
            .unwrap_or_default()
    };
    let creator = [
        creator(Some(rootnode)),
        creator(rootnode.first_child_by_name("FBXHeaderExtension")),
    ]
    .into_iter()
    .find(|c| !c.is_empty())
    .unwrap_or_default();
    let objects_node = rootnode
        .first_child_by_name("Objects")
        .context(ErrorKind::Empty, "FBX has no Objects")?;
    let mut objs = Objects {
        file: file.clone(),
        nodes: HashMap::new(),
        order: Vec::new(),
        parents: HashMap::new(),
        children: HashMap::new(),
        templates: HashMap::new(),
    };
    // Property templates: the defaults of every property an object of a class doesn't give itself.
    if let Some(defs) = rootnode.first_child_by_name("Definitions") {
        for class in defs.children_by_name("ObjectType") {
            let Some(name) = class.attributes().first().and_then(string) else {
                continue;
            };
            for t in class.children_by_name("PropertyTemplate") {
                let kind = t.attributes().first().and_then(string).unwrap_or("").to_owned();
                objs.templates
                    .entry(name.to_owned())
                    .or_default()
                    .push((kind, props(t)));
            }
        }
    }
    for o in objects_node.children() {
        if let Some(id) = o.attributes().first().and_then(id)
            && objs.nodes.insert(id, (o, o.name().to_owned())).is_none()
        {
            objs.order.push(id);
        }
    }
    if let Some(conns) = rootnode.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            let (Some(kind), Some(child), Some(parent)) =
                (a.first().and_then(string), a.get(1).and_then(id), a.get(2).and_then(id))
            else {
                continue;
            };
            let prop = if kind == "OP" {
                a.get(3).and_then(string).unwrap_or("").to_owned()
            } else {
                String::new()
            };
            if kind == "OO" {
                objs.parents.entry(child).or_default().push(parent);
            }
            objs.children.entry(parent).or_default().push((child, prop));
        }
    }

    // The file's axis system and unit, mapped to glTF (X right, Y up, Z toward the viewer,
    // meters) by a root node.
    let settings = rootnode
        .first_child_by_name("GlobalSettings")
        .map(Props::of)
        .unwrap_or_default();
    let axis = |name: &str, sign: &str, default: i32| -> (usize, f64) {
        #[allow(clippy::cast_possible_truncation, reason = "an axis number; `as` saturates bad ones")]
        let a = scalar(&settings, name).map_or(default, |v| v as i32);
        let s = scalar(&settings, sign).unwrap_or(1.0);
        (
            usize::try_from(a.clamp(0, 2)).unwrap_or(1),
            if s < 0.0 { -1.0 } else { 1.0 },
        )
    };
    let (up, up_s) = axis("UpAxis", "UpAxisSign", 1);
    let (front, front_s) = axis("FrontAxis", "FrontAxisSign", 2);
    let (coord, coord_s) = axis("CoordAxis", "CoordAxisSign", 0);
    let unit_cm = scalar(&settings, "UnitScaleFactor")
        .filter(|u| *u > 0.0 && (1.0 / u).is_finite())
        .unwrap_or(1.0);
    let unit = unit_cm / 100.0;
    let mut basis = [[0.0; 4]; 4];
    basis[0][coord] = coord_s * unit;
    basis[1][up] = up_s * unit;
    basis[2][front] = front_s * unit;
    basis[3][3] = 1.0;
    // Degenerate axis settings (two axes the same): fall back to Y-up.
    let degenerate = det3(&basis).abs() < 1e-12;
    if degenerate {
        basis = scale([unit; 3]);
    }
    let signed = |a: usize, s: f64| match (a, s < 0.0) {
        (0, false) => Axis::PosX,
        (0, true) => Axis::NegX,
        (1, false) => Axis::PosY,
        (1, true) => Axis::NegY,
        (_, false) => Axis::PosZ,
        (_, true) => Axis::NegZ,
    };
    let declared = |name: &str, a: Axis| {
        if scalar(&settings, name).is_some() {
            Fact::Declared(a)
        } else {
            Fact::Assumed(a)
        }
    };
    let mut details = Map::new();
    details.insert("coordAxis".into(), json!(signed(coord, coord_s).as_str()));
    let mut info = SourceInfo {
        format: Some("FBX".into()),
        version: Some(format!("{}.{}", version.major(), version.minor())),
        meters_per_unit: if scalar(&settings, "UnitScaleFactor").is_some() {
            Fact::Declared(unit)
        } else {
            Fact::Assumed(unit)
        },
        unit: crate::source::unit_name(unit).map(str::to_owned),
        up_axis: declared("UpAxis", signed(up, up_s)),
        forward_axis: declared("FrontAxis", signed(front, front_s)),
        frame_rate: frame_rate(&settings),
        ..SourceInfo::default()
    };
    if !degenerate {
        let right = det3(&basis) > 0.0;
        let any = ["UpAxis", "FrontAxis", "CoordAxis"]
            .iter()
            .any(|k| scalar(&settings, k).is_some());
        let h = if right { Handedness::Right } else { Handedness::Left };
        info.handedness = if any { Fact::Declared(h) } else { Fact::Assumed(h) };
        info.converted.handedness = !right;
        info.converted.up_axis = !(up == 1 && up_s > 0.0 && coord == 0 && coord_s > 0.0 && front == 2);
    }
    info.converted.units = (unit - 1.0).abs() > 1e-9;
    if let Some(original) = scalar(&settings, "OriginalUnitScaleFactor").filter(|o| (o - unit_cm).abs() > 1e-9) {
        details.insert("originalUnitScaleFactor".into(), json!(original));
    }
    scene_info(rootnode, &creator, &mut info, &mut details);

    let mut r = Reader {
        objs: &objs,
        linear: creator.contains("Blender"),
        dir: input.dir(),
        assets,
        keep,
        limit: input.limit,
        b: Builder::default(),
        nodes: Vec::new(),
        node_of: HashMap::new(),
        meshes: Vec::new(),
        mesh_of: HashMap::new(),
        materials: Vec::new(),
        images: Images::new(!keep.image_bytes, input.dir()),
        material_index: HashMap::new(),
        image_index: HashMap::new(),
        default_material: None,
        textured: Vec::new(),
        skins: Vec::new(),
        skin_of: HashMap::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        channel_users: HashMap::new(),
        morphs: HashMap::new(),
        channel_full: HashMap::new(),
        unit,
        skinned_roots: Vec::new(),
        skinned_meshes: std::collections::HashSet::new(),
        hidden: std::collections::HashSet::new(),
        color_uv: HashMap::new(),
        material_uvs: HashMap::new(),
        line_materials: HashMap::new(),
        curve_points: 0,
    };
    let models: Vec<i64> = objs.order.iter().copied().filter(|id| objs.is("Model", *id)).collect();
    // A level of detail group shows its first level, the finest; the others are kept, hidden.
    for &m in &models {
        if objs
            .children_of(m)
            .any(|a| objs.is_kind("NodeAttribute", a, "LodGroup"))
        {
            let levels = objs.children_of(m).filter(|c| objs.is("Model", *c));
            r.hidden.extend(levels.skip(1));
        }
    }
    for &m in &models {
        let node = r.model_node(m);
        r.nodes.push(node);
        r.node_of.insert(m, r.nodes.len() - 1);
    }
    // Parents, once every model has its node. A model whose parent is under it (a broken file) stays a root: each
    // node's tree is tracked as it's joined to its parent's, and a node is always a root when it's joined, so joining
    // makes a loop only when the parent's root is the node itself.
    let mut roots = Vec::new();
    let mut tree: Vec<usize> = (0..r.nodes.len()).collect();
    let root_of = |tree: &mut Vec<usize>, mut n: usize| {
        let mut top = n;
        while tree[top] != top {
            top = tree[top];
        }
        while tree[n] != top {
            n = std::mem::replace(&mut tree[n], top);
        }
        top
    };
    for &m in &models {
        let n = r.node_of[&m];
        match objs.model_parent(m).and_then(|p| r.node_of.get(&p)) {
            Some(&p) if root_of(&mut tree, p) != n => {
                r.add_child(p, n);
                tree[n] = root_of(&mut tree, p);
            }
            _ => roots.push(n),
        }
    }
    for &m in &models {
        r.attach(m)?;
    }
    // A geometry no model holds is drawn where it is. Geometry that isn't a mesh or a line is counted.
    let mut skipped: BTreeMap<&str, usize> = BTreeMap::new();
    for &g in &objs.order {
        let Some(geom) = objs.node(g).filter(|n| n.name() == "Geometry") else {
            continue;
        };
        let mesh = match kind_of(geom) {
            "NurbsCurve" if !is_curve(geom) => {
                *skipped.entry("NurbsCurve").or_default() += 1;
                continue;
            }
            "Mesh" | "Line" | "NurbsCurve" if objs.parents_of(g).any(|p| objs.is("Model", p)) => continue,
            "Mesh" => r.mesh_for(g, 0)?,
            "Line" | "NurbsCurve" => match r.line_mesh(g, 0) {
                Some(mesh) => mesh,
                None => continue,
            },
            "Shape" | "" => continue,
            kind => {
                *skipped.entry(kind).or_default() += 1;
                continue;
            }
        };
        r.nodes.push(json!({ "mesh": mesh }));
        roots.push(r.nodes.len() - 1);
    }
    if !skipped.is_empty() {
        details.insert("skipped".into(), json!(skipped));
    }
    let animations = if keep.animations { r.animations() } else { Vec::new() };
    let mut scene_nodes = roots;
    if basis != IDENTITY {
        let mut root = json!({ "name": "RootNode", "children": scene_nodes });
        set_transform(&mut root, &basis);
        r.nodes.push(root);
        scene_nodes = vec![r.nodes.len() - 1];
    }
    scene_nodes.append(&mut r.skinned_roots);
    ensure!(!r.nodes.is_empty(), Empty, "FBX has no mesh geometry");

    if r.keep.materials
        && r.keep.textures
        && !r.textured.is_empty()
        && !r.materials.iter().any(|m| m.base_color_tex.is_some())
        && super::plain_colors(r.textured.iter().map(|&m| r.materials[m].base_color))
        && let Some(i) = super::color_texture(input.path, assets, &mut r.images)
    {
        // A model that names no color texture takes one by name (see `guess_color_texture`).
        for &m in &r.textured {
            r.materials[m].base_color_tex = Some(Tex::new(i));
            r.materials[m].base_color[..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        }
    }

    let mut json = Map::new();
    info.write(&mut details);
    json.insert("asset".into(), json!({ "extras": { "source": details } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": scene_nodes }]));
    json.insert("nodes".into(), Value::Array(std::mem::take(&mut r.nodes)));
    if !r.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut r.meshes)));
    }
    if !r.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut r.skins)));
    }
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    if !r.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut r.cameras)));
    }
    let mut used = Vec::new();
    if !r.lights.is_empty() {
        used.push("KHR_lights_punctual");
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    if !r.hidden.is_empty() {
        used.push("KHR_node_visibility");
    }
    if !used.is_empty() {
        json.insert("extensionsUsed".into(), json!(used));
    }
    let (materials, images) = (std::mem::take(&mut r.materials), std::mem::take(&mut r.images));
    let mut b = r.b;
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// Object id → (node, class), and how objects connect.
struct Objects<'a> {
    /// The file's bytes, when it was read from disk.
    file: Option<crate::Bytes>,
    nodes: HashMap<i64, (NodeHandle<'a>, String)>,
    /// Object ids in file order.
    order: Vec<i64>,
    /// child → parents (object-object connections), in file order.
    parents: HashMap<i64, Vec<i64>>,
    /// parent → children with the property they plug into ("" for object-object).
    children: HashMap<i64, Vec<(i64, String)>>,
    /// Class (`Model`, ...) → its property templates, by name (`FbxNode`, ...).
    templates: HashMap<String, Vec<Template<'a>>>,
}

/// A property template: its name (`FbxNode`, ...), and its properties' values by name.
type Template<'a> = (String, HashMap<String, Vec<A<'a>>>);

impl<'a> Objects<'a> {
    fn node(&self, id: i64) -> Option<NodeHandle<'a>> {
        self.nodes.get(&id).map(|(n, _)| *n)
    }

    /// Object `id`'s properties, over its class's template.
    fn props(&self, id: i64) -> Props<'_> {
        self.node(id).map(|n| self.props_of(n)).unwrap_or_default()
    }

    fn props_of(&self, node: NodeHandle<'a>) -> Props<'_> {
        Props {
            own: props(node),
            template: self.template(node),
        }
    }

    /// The template for an object of `node`'s class: the class's only one, or the one its kind or shading model
    /// names when there are more.
    fn template(&self, node: NodeHandle<'_>) -> Option<&HashMap<String, Vec<A<'a>>>> {
        let all = self.templates.get(node.name())?;
        let kind = kind_of(node);
        // A line's or curve's template is never a mesh's, nor a mesh's a line's or curve's.
        if let [(name, one)] = all.as_slice()
            && (node.name() != "Geometry"
                || match kind {
                    "Line" => name == "FbxLine",
                    "NurbsCurve" => name == "FbxNurbsCurve",
                    _ => name != "FbxLine" && name != "FbxNurbsCurve",
                })
        {
            return Some(one);
        }
        let named = |want: &str| all.iter().find(|(n, _)| n == want).map(|(_, p)| p);
        match node.name() {
            "Model" => named("FbxNode"),
            "Geometry" if kind == "Shape" => named("FbxShape"),
            "Geometry" if kind == "Line" => named("FbxLine"),
            "Geometry" if kind == "NurbsCurve" => named("FbxNurbsCurve"),
            "Geometry" => named("FbxMesh"),
            "NodeAttribute" => match kind {
                "LimbNode" => named("FbxSkeleton"),
                "LodGroup" => named("FbxLODGroup"),
                _ => all
                    .iter()
                    .find(|(n, _)| n.strip_prefix("Fbx") == Some(kind))
                    .map(|(_, p)| p),
            },
            "Material" => {
                let model = child_attr(node, "ShadingModel").and_then(string).unwrap_or("");
                if model.eq_ignore_ascii_case("lambert") {
                    named("FbxSurfaceLambert")
                } else {
                    named("FbxSurfacePhong")
                }
            }
            _ => None,
        }
    }

    /// Whether object `id` is a `class` node (`Model`, `Geometry`, `Deformer`, ...).
    fn is(&self, class: &str, id: i64) -> bool {
        self.node(id).is_some_and(|n| n.name() == class)
    }

    /// Whether object `id` is a `class` node of `kind` (`Geometry: id, "Name", "Mesh"`).
    fn is_kind(&self, class: &str, id: i64, kind: &str) -> bool {
        self.node(id).is_some_and(|n| n.name() == class && kind_of(n) == kind)
    }

    fn parents_of(&self, id: i64) -> impl Iterator<Item = i64> + '_ {
        self.parents.get(&id).into_iter().flatten().copied()
    }

    /// Object-object children of `id`.
    fn children_of(&self, id: i64) -> impl Iterator<Item = i64> + '_ {
        self.children
            .get(&id)
            .into_iter()
            .flatten()
            .filter(|(_, p)| p.is_empty())
            .map(|(c, _)| *c)
    }

    /// The child plugged into `id`'s property `prop`.
    fn child_at<'s>(&'s self, id: i64, prop: &'s str) -> impl Iterator<Item = i64> + 's {
        self.children
            .get(&id)
            .into_iter()
            .flatten()
            .filter(move |(_, p)| p == prop)
            .map(|(c, _)| *c)
    }

    fn model_parent(&self, id: i64) -> Option<i64> {
        self.parents_of(id).find(|p| self.is("Model", *p))
    }
}

/// An object's name, without its `\0\x01Class` suffix.
fn name_of(node: NodeHandle<'_>) -> Option<&str> {
    let name = node.attributes().get(1).and_then(string)?.split('\0').next()?;
    (!name.is_empty()).then_some(name)
}

/// A model's local transform: `T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1`, the FBX SDK's
/// order, with its fixed parts (offsets, pivots, pre- and post-rotation) multiplied out, for evaluating it again at
/// each animation time: `T · before · R · between · S · after`.
struct Local {
    t: [f64; 3],
    r: [f64; 3],
    s: [f64; 3],
    order: i32,
    before: M4,
    between: M4,
    after: M4,
    /// Whether the parent's own scale is taken off before this model's rotation and scale (`InheritType` 2, a
    /// joint's segment scale compensation): it still moves the model, but doesn't stretch it.
    ignore_parent_scale: bool,
}

impl Local {
    fn new(p: &Props<'_>) -> Self {
        let order = rotation_order(p);
        let active = flag(p, "RotationActive", true);
        let turn = |name: &str| if active { vec3(p, name, [0.0; 3]) } else { [0.0; 3] };
        let r_pre = rotate(turn("PreRotation"), 0);
        let r_post_inv = transpose3(&rotate(turn("PostRotation"), 0));
        let rp = vec3(p, "RotationPivot", [0.0; 3]);
        let sp = vec3(p, "ScalingPivot", [0.0; 3]);
        let product = |ms: &[M4]| ms.iter().fold(IDENTITY, |acc, m| mul(&acc, m));
        Self {
            t: vec3(p, "Lcl Translation", [0.0; 3]),
            r: vec3(p, "Lcl Rotation", [0.0; 3]),
            s: vec3(p, "Lcl Scaling", [1.0; 3]),
            order,
            before: product(&[translate(vec3(p, "RotationOffset", [0.0; 3])), translate(rp), r_pre]),
            between: product(&[
                r_post_inv,
                translate(rp.map(|c| -c)),
                translate(vec3(p, "ScalingOffset", [0.0; 3])),
                translate(sp),
            ]),
            after: translate(sp.map(|c| -c)),
            ignore_parent_scale: choice(p, "InheritType", 0) == 2,
        }
    }

    /// The transform with translation, rotation and scale as given, or as the model holds them, under a parent
    /// scaled `parent` (its own scale, not its parents').
    fn at(&self, t: Option<[f64; 3]>, r: Option<[f64; 3]>, s: Option<[f64; 3]>, parent: [f64; 3]) -> M4 {
        let mut m = translate(t.unwrap_or(self.t));
        if self.ignore_parent_scale && parent.iter().all(|c| c.abs() > 1e-12 && (1.0 / c).is_finite()) {
            m = mul(&m, &scale(parent.map(|c| 1.0 / c)));
        }
        let m = mul(&m, &self.before);
        let m = mul(&m, &rotate(r.unwrap_or(self.r), self.order));
        let m = mul(&m, &self.between);
        let m = mul(&m, &scale(s.unwrap_or(self.s)));
        mul(&m, &self.after)
    }
}

/// The transpose of a rotation (its inverse).
fn transpose3(r: &M4) -> M4 {
    let mut t = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = r[j][i];
        }
    }
    t
}

/// A model's geometric transform: applied to what it holds, not passed to its children.
fn geometric_of(p: &Props<'_>) -> M4 {
    mul(
        &translate(vec3(p, "GeometricTranslation", [0.0; 3])),
        &mul(
            &rotate(vec3(p, "GeometricRotation", [0.0; 3]), 0),
            &scale(vec3(p, "GeometricScaling", [1.0; 3])),
        ),
    )
}
/// A property's values as JSON: a number, a string, or a list of them.
fn prop_json(vals: &[A]) -> Option<Value> {
    let one = |v: &A| match v {
        A::String(s) => Some(json!(s.split('\0').next().unwrap_or(""))),
        A::Bool(b) => Some(json!(b)),
        A::I16(i) => Some(json!(i)),
        A::I32(i) => Some(json!(i)),
        A::I64(i) => Some(json!(i)),
        A::F32(f) if f.is_finite() => Some(json!(f)),
        A::F64(f) if f.is_finite() => Some(json!(f)),
        _ => None,
    };
    match vals {
        [] => None,
        [v] => one(v),
        vs => vs.iter().map(one).collect::<Option<Vec<_>>>().map(Value::Array),
    }
}

/// A model's user-defined properties (flagged `U`), for its node's extras.
fn user_props(node: NodeHandle<'_>) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(p70) = node.first_child_by_name("Properties70") {
        for p in p70.children_by_name("P") {
            let a = p.attributes();
            if let (Some(name), Some(flags)) = (a.first().and_then(string), a.get(3).and_then(string))
                && flags.contains('U')
                && let Some(v) = prop_json(a.get(4..).unwrap_or_default())
            {
                out.insert(name.to_owned(), v);
            }
        }
    }
    out
}

/// How a key's value runs on to the next key's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Interp {
    /// Holds its value until the next key.
    Constant,
    /// Takes the next key's value straight after it.
    ConstantNext,
    Linear,
    /// A Bezier curve leaving the key along its slope out and reaching the next along that one's slope in.
    Cubic,
}

impl Interp {
    /// A key's interpolation, from its `KeyAttrFlags`; a key that names none is linear.
    fn of(flags: i64) -> Self {
        if flags & 0x2 != 0 {
            if flags & 0x100 != 0 {
                Self::ConstantNext
            } else {
                Self::Constant
            }
        } else if flags & 0x8 != 0 {
            Self::Cubic
        } else {
            Self::Linear
        }
    }
}

/// How far along its segment an unweighted Bezier handle reaches.
const THIRD: f64 = 1.0 / 3.0;

/// The handles of a segment whose keys have no slopes: flat, a third of the way along.
const FLAT: [f64; 4] = [0.0, THIRD, 0.0, THIRD];

/// An animation curve: keys in seconds, how each runs on to the next, and for a cubic one, its handles.
struct Curve {
    times: Vec<f64>,
    values: Vec<f64>,
    interp: Vec<Interp>,
    /// Per key, the handles of the segment after it: the key's slope out (value per second) and how far along the
    /// segment its handle reaches (a fraction of its length), then the same for the next key's way in.
    handles: Vec<[f64; 4]>,
    /// Where the last lookup landed among the keys, tried first since times are mostly asked for in order.
    cursor: Cell<usize>,
}

impl Curve {
    /// A curve running straight from key to key.
    fn linear(times: Vec<f64>, values: Vec<f64>) -> Self {
        let n = times.len();
        Self {
            times,
            values,
            interp: vec![Interp::Linear; n],
            handles: vec![FLAT; n],
            cursor: Cell::new(0),
        }
    }

    /// How many keys are at or before `t`.
    fn keys_until(&self, t: f64) -> usize {
        let n = self.times.len();
        let fits = |i: usize| (i == 0 || self.times[i - 1] <= t) && (i == n || self.times[i] > t);
        let c = self.cursor.get();
        if c <= n && fits(c) {
            return c;
        }
        if c < n && fits(c + 1) {
            self.cursor.set(c + 1);
            return c + 1;
        }
        let i = self.times.partition_point(|&k| k <= t);
        self.cursor.set(i);
        i
    }

    fn read(node: NodeHandle<'_>) -> Option<Self> {
        let times: Vec<f64> = child_attr(node, "KeyTime")
            .and_then(ints)?
            .into_iter()
            .map(|t| {
                #[allow(clippy::cast_precision_loss, reason = "FBX time to seconds, as the SDK does")]
                let t = t as f64;
                t / KTIME
            })
            .collect();
        let mut values = floats(child_attr(node, "KeyValueFloat")?)?;
        if times.is_empty() || times.len() != values.len() || times.windows(2).any(|w| w[1] < w[0]) {
            return None;
        }
        // A value that isn't a number takes the one before it (the first number, for the first).
        let mut last = values.iter().copied().find(|v| v.is_finite())?;
        for v in &mut values {
            if v.is_finite() {
                last = *v;
            } else {
                *v = last;
            }
        }
        // Flags and data are shared by runs of keys: run i covers the next `KeyAttrRefCount[i]` keys, with four
        // numbers of `KeyAttrDataFloat`: the slope out, the next key's slope in, and the handles' packed lengths
        // (or a TCB key's tension, continuity and bias).
        let flags = child_attr(node, "KeyAttrFlags").and_then(ints).unwrap_or_default();
        let counts = child_attr(node, "KeyAttrRefCount").and_then(ints).unwrap_or_default();
        let data = child_attr(node, "KeyAttrDataFloat")
            .and_then(floats)
            .unwrap_or_default();
        let n = times.len();
        let mut curve = Self::linear(times, values);
        let mut tcb = vec![None; n];
        let mut k = 0;
        for (run, (&f, &count)) in flags.iter().zip(&counts).enumerate() {
            let d: [f64; 4] = std::array::from_fn(|j| {
                let x = run
                    .checked_mul(4)
                    .and_then(|at| data.get(at + j))
                    .copied()
                    .unwrap_or(0.0);
                if x.is_finite() { x } else { 0.0 }
            });
            for _ in 0..usize::try_from(count).unwrap_or(0) {
                if k == n {
                    break;
                }
                curve.interp[k] = Interp::of(f);
                if curve.interp[k] == Interp::Cubic {
                    if f & 0x200 != 0 {
                        tcb[k] = Some([d[0], d[1], d[2]]);
                    } else {
                        let (out, next_in) = handle_lengths(d[2]);
                        curve.handles[k] = [
                            d[0],
                            if f & 0x0100_0000 != 0 { out } else { THIRD },
                            d[1],
                            if f & 0x0200_0000 != 0 { next_in } else { THIRD },
                        ];
                    }
                }
                k += 1;
            }
            if k == n {
                break;
            }
        }
        if tcb.iter().any(Option::is_some) {
            curve.tcb(&tcb);
        }
        Some(curve)
    }

    /// Slopes for the keys given as tension, continuity and bias, from the slopes of the segments either side, as
    /// a Kochanek-Bartels spline makes them. A key after one is reached along its own TCB slope in, plain (a
    /// Catmull-Rom slope) if it has no TCB values of its own.
    fn tcb(&mut self, params: &[Option<[f64; 3]>]) {
        let n = self.times.len();
        let segment = |i: usize| {
            let dt = self.times[i + 1] - self.times[i];
            (dt > 0.0).then(|| (self.values[i + 1] - self.values[i]) / dt)
        };
        let slopes: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let [t, c, b] = params[i].unwrap_or([0.0; 3]);
                let before = if i > 0 { segment(i - 1) } else { None };
                let after = if i + 1 < n { segment(i) } else { None };
                let (before, after) = match (before, after) {
                    (Some(p), Some(q)) => (p, q),
                    (Some(p), None) => (p, p),
                    (None, Some(q)) => (q, q),
                    (None, None) => (0.0, 0.0),
                };
                let into = (1.0 - t) * ((1.0 - c) * (1.0 + b) * before + (1.0 + c) * (1.0 - b) * after) / 2.0;
                let out = (1.0 - t) * ((1.0 + c) * (1.0 + b) * before + (1.0 - c) * (1.0 - b) * after) / 2.0;
                (into, out)
            })
            .collect();
        for i in 0..n {
            if params[i].is_some() {
                let next_in = slopes.get(i + 1).map_or(0.0, |s| s.0);
                self.handles[i] = [slopes[i].1, THIRD, next_in, THIRD];
            }
        }
    }

    /// The value at `t`: the first or last key's outside them, else as the key before `t` runs on to the next.
    fn at(&self, t: f64) -> f64 {
        let i = self.keys_until(t);
        if i == 0 {
            return self.values[0];
        }
        if i == self.times.len() {
            return self.values[i - 1];
        }
        self.segment(i - 1, t)
    }

    /// The value at `t` as key `i` runs on to the next, for `t` from that key up to the next.
    fn segment(&self, i: usize, t: f64) -> f64 {
        let (t0, t1, v0, v1) = (self.times[i], self.times[i + 1], self.values[i], self.values[i + 1]);
        let dt = t1 - t0;
        if t <= t0 || dt <= 0.0 {
            return v0;
        }
        let x = (t - t0) / dt;
        match self.interp[i] {
            Interp::Constant => v0,
            Interp::ConstantNext => v1,
            Interp::Linear => v0 + (v1 - v0) * x,
            Interp::Cubic => {
                let [s0, w0, s1, w1] = self.handles[i];
                let (w0, w1) = (w0.clamp(0.0, 1.0), w1.clamp(0.0, 1.0));
                // Handles a third of the way along make time run evenly along the curve; others are solved for.
                #[allow(
                    clippy::float_cmp_const,
                    reason = "a third is set exactly, as a marker, never computed"
                )]
                let u = if w0 == THIRD && w1 == THIRD {
                    x
                } else {
                    along(w0, w1, x)
                };
                bezier(v0, v0 + s0 * w0 * dt, v1 - s1 * w1 * dt, v1, u)
            }
        }
    }

    /// Euler angles in degrees turned the short way where a curve crosses ±180° between two keys. Exporters
    /// bake an angle going past 180 as its other side (179 then -179), which read literally is a whole
    /// turn back in one frame, not two degrees on. A turn over 180° is only a wrap when it dwarfs the
    /// curve's usual step: a coin keyed 0 then 360 still spins.
    fn unwrap_degrees(&mut self) {
        let mut steps: Vec<f64> = self
            .values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut by = 0.0;
        for i in 1..self.values.len() {
            let raw = self.values[i] + by;
            let d = raw - self.values[i - 1];
            if d.abs() > 180.0 && d.abs() > usual * 4.0 {
                let turns = (d / 360.0).round() * 360.0;
                by -= turns;
            }
            self.values[i] += by;
        }
    }

    /// The times a linear sampler needs to follow the curve: its keys, a moment either side of each step, and
    /// along a cubic segment enough points (up to 32) that straight lines between them stray from it by no more
    /// than a thousandth of the curve's range where checked.
    fn samples(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.times);
        let (lo, hi) = self
            .values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let tolerance = 1e-3 * (hi - lo).max(1e-6);
        for i in 1..self.times.len() {
            let (t0, t1) = (self.times[i - 1], self.times[i]);
            let dt = t1 - t0;
            if dt <= 0.0 {
                continue;
            }
            match self.interp[i - 1] {
                Interp::Constant => out.push(t1 - (dt / 2.0).min(1e-3)),
                Interp::ConstantNext => out.push(t0 + (dt / 2.0).min(1e-3)),
                Interp::Linear => {}
                Interp::Cubic => {
                    let at = |k: u32, n: u32| t0 + dt * f64::from(k) / f64::from(n);
                    // Inside this segment it's evaluated directly; at its end (or past it, by rounding) looked up.
                    let value = |t: f64| {
                        if t0 <= t && t < t1 {
                            self.segment(i - 1, t)
                        } else {
                            self.at(t)
                        }
                    };
                    let mut ends = [0.0; 33];
                    let mut n = 1;
                    while n < 32 {
                        for k in 0..=n {
                            ends[k as usize] = value(at(k, n));
                        }
                        // Checked a quarter, half and three quarters of the way along each piece: an ease in and
                        // out is on the line at its middle.
                        let strays = (0..n).any(|k| {
                            let (a, b) = (at(k, n), at(k + 1, n));
                            let (va, vb) = (ends[k as usize], ends[k as usize + 1]);
                            [0.25, 0.5, 0.75]
                                .into_iter()
                                .any(|f| (value(a + (b - a) * f) - (va + (vb - va) * f)).abs() > tolerance)
                        });
                        if !strays {
                            break;
                        }
                        n *= 2;
                    }
                    out.extend((1..n).map(|k| at(k, n)));
                }
            }
        }
    }
}

/// How far along a segment's curve time has run a fraction `x` of the segment, for handles reaching `w0` and `w1`
/// of the way along, found by halving the range 48 times.
fn along(w0: f64, w1: f64, x: f64) -> f64 {
    let c = 1.0 - w1;
    // The time at `u`: `bezier(0, w0, c, 1, u)` without its first term, which is zero.
    let before = |u: f64| {
        let v = 1.0 - u;
        3.0 * w0 * v * v * u + 3.0 * c * v * u * u + u * u * u < x
    };
    let (mut lo, mut hi) = (0.0, 1.0);
    // Two halvings at a time: the midpoints of both halves are tried alongside the middle, so they don't wait on it.
    for _ in 0..24 {
        let mid = f64::midpoint(lo, hi);
        let (a, b) = (f64::midpoint(lo, mid), f64::midpoint(mid, hi));
        let (m, below_a, below_b) = (before(mid), before(a), before(b));
        if m {
            (lo, hi) = if below_b { (b, hi) } else { (mid, b) };
        } else {
            (lo, hi) = if below_a { (a, mid) } else { (lo, a) };
        }
    }
    f64::midpoint(lo, hi)
}

/// A key's packed handle lengths, in ten-thousandths of its segment: its own way out in the low 16 bits, the next
/// key's way in in the high 16. Binary files keep the bits in a float; text files write them as a whole number.
fn handle_lengths(x: f64) -> (f64, f64) {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "packed bits; `as` saturates bad ones"
    )]
    let bits = if x >= 1.0 {
        x.round() as u32
    } else {
        (x as f32).to_bits()
    };
    let length = |b: u32| {
        let w = f64::from(b & 0xFFFF) / 9999.0;
        // A third written in ten-thousandths, or through a float, is a third.
        if (w - THIRD).abs() < 1e-3 {
            THIRD
        } else {
            w.clamp(0.0, 1.0)
        }
    };
    (length(bits), length(bits >> 16))
}

/// A cubic Bezier through `a`, `b`, `c` and `d` at `u` (0 to 1).
fn bezier(a: f64, b: f64, c: f64, d: f64, u: f64) -> f64 {
    let v = 1.0 - u;
    a * v * v * v + 3.0 * b * v * v * u + 3.0 * c * v * u * u + d * u * u * u
}

/// An animated vector property (`Lcl Translation`, ...): a curve or a fixed value per component.
struct Animated {
    curves: [Option<Curve>; 3],
    fixed: [f64; 3],
}

impl Animated {
    fn at(&self, t: f64) -> [f64; 3] {
        std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.at(t)))
    }

    fn times(&self, out: &mut Vec<f64>) {
        for c in self.curves.iter().flatten() {
            c.samples(out);
        }
    }

    /// Euler rotation keys turned the short way, as Maya's Euler filter turns them. Exporters bake each
    /// frame's rotation as any of the angles that make it: an angle past 180 as its other side (179 then
    /// -179), or a limb raised past straight up as the other set of three (two axes half a turn on, the
    /// middle one mirrored). Read literally, either is a spin in one frame. Where a key jumps more than
    /// a quarter turn from the one before, and far more than the curve's usual step (so a coin keyed 0 then
    /// 360 still spins), it's given the angles making the same rotation that lie nearest the last key.
    fn unwrap_rotation(&mut self, order: i32) {
        let Some(times) = self.curves.iter().flatten().next().map(|c| c.times.clone()) else {
            return;
        };
        let shared = self
            .curves
            .iter()
            .flatten()
            .all(|c| c.times.len() == times.len() && c.times.iter().zip(&times).all(|(a, b)| (a - b).abs() < 1e-9));
        if !shared {
            // The three can't be read together key by key: each turns the short way on its own.
            self.curves.iter_mut().flatten().for_each(Curve::unwrap_degrees);
            return;
        }
        let raw: Vec<[f64; 3]> = (0..times.len())
            .map(|i| std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.values[i])))
            .collect();
        let apart = |a: &[f64; 3], b: &[f64; 3]| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
        let mut steps: Vec<f64> = raw
            .windows(2)
            .map(|w| apart(&w[0], &w[1]))
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut out = raw.clone();
        // The axis mirrored at each key given the other set of three: its slopes turn the other way.
        let mut mirrored = vec![None; raw.len()];
        for i in 1..raw.len() {
            let (prev, key) = (out[i - 1], raw[i]);
            let jump = apart(&prev, &key);
            if jump <= 90.0 || jump <= usual * 4.0 {
                continue;
            }
            let nearest = |v: [f64; 3]| -> [f64; 3] {
                std::array::from_fn(|k| v[k] + 360.0 * ((prev[k] - v[k]) / 360.0).round())
            };
            let made = rotate(key, order);
            let mut best = nearest(key);
            // The other set of three: whichever axis is the middle one for this order is mirrored.
            for middle in 0..3 {
                let other: [f64; 3] =
                    std::array::from_fn(|k| if k == middle { 180.0 - key[k] } else { key[k] + 180.0 });
                let m = rotate(other, order);
                if (0..3).all(|r| (0..3).all(|c| (m[r][c] - made[r][c]).abs() < 1e-6)) {
                    let other = nearest(other);
                    if apart(&prev, &other) < apart(&prev, &best) {
                        best = other;
                        mirrored[i] = Some(middle);
                    }
                }
            }
            out[i] = best;
        }
        let interp: Vec<Interp> = self
            .curves
            .iter()
            .flatten()
            .next()
            .map(|c| {
                c.interp
                    .iter()
                    .map(|&i| if i == Interp::Cubic { Interp::Linear } else { i })
                    .collect()
            })
            .unwrap_or_default();
        for k in 0..3 {
            let values: Vec<f64> = out.iter().map(|v| v[k]).collect();
            match &mut self.curves[k] {
                Some(c) => {
                    c.values = values;
                    for (i, m) in mirrored.iter().enumerate() {
                        if *m == Some(k) {
                            c.handles[i][0] = -c.handles[i][0];
                            if i > 0 {
                                c.handles[i - 1][2] = -c.handles[i - 1][2];
                            }
                        }
                    }
                }
                // An axis with no curve of its own gets one when its angle had to change, keeping the others'
                // steps.
                None if values.iter().any(|v| (v - self.fixed[k]).abs() > 1e-9) => {
                    let mut c = Curve::linear(times.clone(), values);
                    c.interp.clone_from(&interp);
                    self.curves[k] = Some(c);
                }
                None => {}
            }
        }
    }
}

/// How an animation layer's value meets the value of the layers under it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Blend {
    /// Takes its place, by the layer's weight.
    Override,
    /// Adds to it, by the layer's weight.
    Add,
    /// Multiplies it, by the layer's weight (an additive layer's scale).
    Multiply,
}

/// An animation layer of a stack: its object, how it blends, and its weight (0 to 1).
#[derive(Clone, Copy)]
struct AnimLayer {
    id: i64,
    additive: bool,
    weight: f64,
    /// Whether an additive layer multiplies scale rather than adding to it.
    multiply_scale: bool,
}

/// A property animated through a stack's layers: its value at rest, then each layer's curves over it in order.
struct Layered {
    rest: [f64; 3],
    layers: Vec<(Blend, f64, Animated)>,
}

impl Layered {
    fn at(&self, t: f64) -> [f64; 3] {
        let mut v = self.rest;
        for (blend, w, a) in &self.layers {
            let x = a.at(t);
            for k in 0..3 {
                v[k] = match blend {
                    Blend::Override if *w >= 1.0 => x[k],
                    Blend::Override => v[k] + (x[k] - v[k]) * w,
                    Blend::Add => v[k] + x[k] * w,
                    Blend::Multiply if x[k] > 0.0 => v[k] * x[k].powf(*w),
                    Blend::Multiply => v[k] * (1.0 + (x[k] - 1.0) * w),
                };
            }
        }
        v
    }

    fn times(&self, out: &mut Vec<f64>) {
        for (_, _, a) in &self.layers {
            a.times(out);
        }
    }
}

/// Sorted, de-duplicated times, shifted so none is negative.
fn timeline(mut times: Vec<f64>) -> Vec<f64> {
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    times
}

/// A blend shape target: its name, and by control point, its offsets to position and normal.
struct Target {
    name: String,
    positions: HashMap<usize, [f32; 3]>,
    normals: HashMap<usize, [f32; 3]>,
}

/// A blend shape channel: its weight (a fraction, not clamped), and its targets with the weight each is reached
/// in full at, ascending: the in-betweens, then the full shape.
struct Channel {
    weight: f64,
    full: Vec<f64>,
    targets: Vec<Target>,
}

/// A shape's offsets by control point, to position and to normal; ones that aren't numbers are left out.
fn offsets(shape: NodeHandle<'_>) -> (HashMap<usize, [f32; 3]>, HashMap<usize, [f32; 3]>) {
    let idx = child_attr(shape, "Indexes").and_then(ints).unwrap_or_default();
    let read = |name: &str| {
        let data = child_attr(shape, name).and_then(floats).unwrap_or_default();
        let mut out = HashMap::new();
        for (k, &cp) in idx.iter().enumerate() {
            if let (Ok(cp), Some(d)) = (usize::try_from(cp), data.get(k * 3..k * 3 + 3))
                && d.iter().all(|x| x.is_finite())
            {
                out.insert(cp, [d[0], d[1], d[2]].map(f32_of));
            }
        }
        out
    };
    (read("Vertices"), read("Normals"))
}

/// Builds the document as the file's objects are visited.
struct Reader<'o, 'a, 'x> {
    objs: &'o Objects<'a>,
    /// Blender writes material colors linear; everyone else sRGB.
    linear: bool,
    dir: &'x Path,
    assets: &'x dyn Assets,
    keep: Keep,
    /// The most triangles one geometry may have.
    limit: usize,
    b: Builder,
    nodes: Vec<Value>,
    /// Model id → node.
    node_of: HashMap<i64, usize>,
    meshes: Vec<Value>,
    /// (geometry, its materials) → mesh.
    mesh_of: HashMap<(i64, Vec<usize>), usize>,
    materials: Vec<Full>,
    images: Images,
    material_index: HashMap<i64, usize>,
    image_index: HashMap<i64, Option<usize>>,
    /// The gray material primitives without one take.
    default_material: Option<usize>,
    /// Materials a primitive with UVs uses.
    textured: Vec<usize>,
    skins: Vec<Value>,
    /// (skin deformer, model) → skin.
    skin_of: HashMap<(i64, i64), usize>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    /// Blend shape channel → (node, target) pairs it drives.
    channel_users: HashMap<i64, Vec<(usize, usize)>>,
    /// Mesh → the blend shape channels its targets come from, in order, with each one's first target.
    morphs: HashMap<usize, Vec<(i64, usize)>>,
    /// Blend shape channel → the weights its targets are reached in full at.
    channel_full: HashMap<i64, Vec<f64>>,
    /// Meters per file unit.
    unit: f64,
    /// Nodes holding skinned meshes, which go at the scene's root, outside the axis and unit conversion.
    skinned_roots: Vec<usize>,
    /// Meshes carrying joints and weights.
    skinned_meshes: std::collections::HashSet<usize>,
    /// Models hidden, by their own visibility or as a level of detail not shown.
    hidden: std::collections::HashSet<i64>,
    /// Material → the UV set its color texture reads, by name.
    color_uv: HashMap<usize, String>,
    /// Material → its other textures that read a UV set by name, and the set; taken once the material is bound to
    /// a mesh's sets.
    material_uvs: HashMap<usize, Vec<(TexField, String)>>,
    /// A line's linear color, by its components' bits → the unlit material drawing it.
    line_materials: HashMap<[u32; 3], usize>,
    /// The points NURBS curves have drawn, out of `MAX_CURVE_POINTS`.
    curve_points: usize,
}

/// Sets a node's transform, unless it isn't finite (a broken file): the node stays where its parent is.
fn place(node: &mut Value, m: &M4) {
    if m.iter().flatten().all(|c| c.is_finite()) {
        set_transform(node, m);
    }
}

/// Marks a node hidden.
fn hide(node: &mut Value) {
    node["extensions"]["KHR_node_visibility"] = json!({ "visible": false });
}

impl Reader<'_, '_, '_> {
    /// A model's node: its name, local transform, visibility and user properties.
    fn model_node(&mut self, m: i64) -> Value {
        let Some(node) = self.objs.node(m) else {
            return json!({});
        };
        let mut out = json!({});
        if let Some(name) = name_of(node) {
            out["name"] = json!(name);
        }
        place(&mut out, &self.rest_local(m));
        if hidden(&self.objs.props(m)) {
            self.hidden.insert(m);
        }
        if self.hidden.contains(&m) {
            hide(&mut out);
        }
        let extras = user_props(node);
        if !extras.is_empty() {
            out["extras"] = Value::Object(extras);
        }
        out
    }

    /// Model `m`'s local transform at rest.
    fn rest_local(&self, m: i64) -> M4 {
        let local = Local::new(&self.objs.props(m));
        local.at(None, None, None, self.parent_scale(&local, m))
    }

    /// The scale of model `m`'s parent, when `m` takes it off (one otherwise).
    fn parent_scale(&self, local: &Local, m: i64) -> [f64; 3] {
        match self.objs.model_parent(m) {
            Some(p) if local.ignore_parent_scale && p != m => vec3(&self.objs.props(p), "Lcl Scaling", [1.0; 3]),
            _ => [1.0; 3],
        }
    }

    /// Whether model `m` or one of its parents is hidden.
    fn hidden_under(&self, m: i64) -> bool {
        let mut at = Some(m);
        for _ in 0..=self.node_of.len() {
            let Some(id) = at else { break };
            if self.hidden.contains(&id) {
                return true;
            }
            at = self.objs.model_parent(id).filter(|&p| p != id);
        }
        false
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let kids = self.nodes[parent]
            .as_object_mut()
            .expect("a node is an object")
            .entry("children")
            .or_insert_with(|| json!([]));
        kids.as_array_mut().expect("children is a list").push(json!(child));
    }

    /// What model `m` holds: its geometry (with skin and morph targets), camera and light.
    fn attach(&mut self, m: i64) -> Result<()> {
        let node = self.node_of[&m];
        let geometric = geometric_of(&self.objs.props(m));
        let geometries: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|&g| {
                self.objs
                    .node(g)
                    .is_some_and(|n| n.name() == "Geometry" && matches!(kind_of(n), "Mesh" | "Line" | "NurbsCurve"))
            })
            .collect();
        for g in geometries {
            let built = if self.objs.is_kind("Geometry", g, "Mesh") {
                self.mesh_for(g, m)?
            } else {
                match self.line_mesh(g, m) {
                    Some(built) => built,
                    None => continue,
                }
            };
            let skin = self
                .objs
                .children_of(g)
                .find(|d| self.objs.is_kind("Deformer", *d, "Skin"));
            let skinned = self.skinned_meshes.contains(&built);
            let target = if skinned && let Some(skin) = skin {
                // A skinned mesh's node transform is ignored, its joints place it (the geometric transform
                // goes into the inverse bind matrices), so it gets a root node of its own, as glTF asks; the
                // model's node stays where it is, a joint for vertices no bone weights. Out of its model's
                // tree, it's hidden when that is.
                let s = self.skin(skin, m, &geometric);
                let mut host = json!({ "skin": s });
                if let Some(name) = self.nodes[node].get("name") {
                    host["name"] = name.clone();
                }
                if self.hidden_under(m) {
                    hide(&mut host);
                }
                self.nodes.push(host);
                self.skinned_roots.push(self.nodes.len() - 1);
                self.nodes.len() - 1
            } else if geometric == IDENTITY && self.nodes[node].get("mesh").is_none() {
                node
            } else {
                self.child(node, Some(&geometric))
            };
            self.nodes[target]["mesh"] = json!(built);
            for (ch, first) in self.morphs.get(&built).cloned().unwrap_or_default() {
                self.channel_users.entry(ch).or_default().push((target, first));
            }
        }
        let attributes: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|a| self.objs.is("NodeAttribute", *a))
            .collect();
        for a in attributes {
            let Some(attr) = self.objs.node(a) else { continue };
            match kind_of(attr) {
                "Camera" if self.keep.cameras => {
                    let cam = camera(&self.objs.props(a), self.unit);
                    self.cameras.push(cam);
                    // FBX cameras look down +X, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        0.0,
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["camera"] = json!(self.cameras.len() - 1);
                }
                "Light" if self.keep.lights => {
                    let Some(light) = light(&self.objs.props(a)) else {
                        continue;
                    };
                    self.lights.push(light);
                    // FBX lights shine down -Y, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["extensions"] = json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A new child of `parent`, with transform `m`; returns it.
    fn child(&mut self, parent: usize, m: Option<&M4>) -> usize {
        let mut n = json!({});
        if let Some(m) = m {
            place(&mut n, m);
        }
        self.nodes.push(n);
        let c = self.nodes.len() - 1;
        self.add_child(parent, c);
        c
    }

    /// Model `m`'s materials, in connection order: a geometry's material layer indexes into this list.
    fn model_materials(&mut self, m: i64) -> Vec<usize> {
        let ids: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|c| self.objs.is("Material", *c))
            .collect();
        if !self.keep.materials {
            return Vec::new();
        }
        ids.into_iter()
            .map(|mid| {
                if let Some(&i) = self.material_index.get(&mid) {
                    return i;
                }
                let (mat, color_set, sets) = read_material(
                    mid,
                    self.objs,
                    self.linear,
                    self.dir,
                    self.keep.textures.then_some(self.assets),
                    &mut self.images,
                    &mut self.image_index,
                );
                let i = self.materials.len();
                self.materials.push(mat);
                self.material_index.insert(mid, i);
                if let Some(set) = color_set {
                    self.color_uv.insert(i, set);
                }
                if !sets.is_empty() {
                    self.material_uvs.insert(i, sets);
                }
                i
            })
            .collect()
    }

    fn default_material(&mut self) -> usize {
        *self.default_material.get_or_insert_with(|| {
            self.materials.push(Material::plain("default", GRAY).into());
            self.materials.len() - 1
        })
    }

    /// An unlit material for lines of sRGB `color`, one for each color.
    fn line_material(&mut self, color: [f64; 3]) -> usize {
        let rgb = color.map(|c| srgb_to_linear(f32_of(c).clamp(0.0, 1.0)));
        let key = rgb.map(f32::to_bits);
        if let Some(&i) = self.line_materials.get(&key) {
            return i;
        }
        let mut m = Full::from(&Material::plain("", GRAY));
        m.unlit = true;
        m.metallic = 0.0;
        m.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
        self.materials.push(m);
        self.line_materials.insert(key, self.materials.len() - 1);
        self.materials.len() - 1
    }

    /// The mesh for `Line` or `NurbsCurve` geometry `g` as model `m` (0 for none) draws it: its segments, in the
    /// model's first material, else unlit in the geometry's `Color` (white without one); `None` when it has none.
    fn line_mesh(&mut self, g: i64, m: i64) -> Option<usize> {
        let mut mats = if m == 0 { Vec::new() } else { self.model_materials(m) };
        mats.truncate(1);
        if let Some(&i) = self.mesh_of.get(&(g, mats.clone())) {
            return Some(i);
        }
        let geom = self.objs.node(g)?;
        let prim = if kind_of(geom) == "NurbsCurve" {
            let (prim, drawn) = curve_geometry(geom, MAX_CURVE_POINTS.saturating_sub(self.curve_points));
            self.curve_points += drawn;
            prim
        } else {
            line_geometry(geom)
        };
        if prim.lines.is_empty() {
            return None;
        }
        let material = match mats.first() {
            Some(&i) => i,
            None if self.keep.materials => {
                let color = vec3(&self.objs.props_of(geom), "Color", [1.0; 3]);
                self.line_material(color)
            }
            None => self.default_material(),
        };
        let mut primitives = self.b.primitives(prim);
        for p in &mut primitives {
            p["material"] = json!(material);
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(name) = name_of(geom) {
            mesh["name"] = json!(name);
        }
        self.meshes.push(mesh);
        let i = self.meshes.len() - 1;
        self.mesh_of.insert((g, mats), i);
        Some(i)
    }

    /// The blend shape channels deforming geometry `g`, in target order.
    fn morph_channels(&self, g: i64) -> Vec<i64> {
        self.objs
            .children_of(g)
            .filter(|d| self.objs.is_kind("Deformer", *d, "BlendShape"))
            .flat_map(|d| self.objs.children_of(d).collect::<Vec<_>>())
            .filter(|c| self.objs.is_kind("Deformer", *c, "BlendShapeChannel"))
            .collect()
    }

    /// Points material `material`'s textures that read a UV set by name at that set's number in `prim`, whose
    /// texture coordinates (`uvs`) are UV layer `color_set` of the geometry's layers `names`; the first layer then
    /// takes that one's number. A set the primitive doesn't have leaves the texture on the first. A material
    /// shared by meshes is bound by the first.
    fn bind_uv_sets(&mut self, material: usize, prim: &Prim, names: &[String], color_set: usize) {
        let Some(sets) = self.material_uvs.remove(&material) else {
            return;
        };
        for (field, name) in sets {
            let Some(layer) = names.iter().position(|n| *n == name) else {
                continue;
            };
            let number = match layer {
                l if l == color_set => 0,
                0 => color_set,
                l => l,
            };
            let there = if number == 0 {
                !prim.uvs.is_empty()
            } else {
                prim.uv_sets.get(number).is_some_and(|s| !s.is_empty())
            };
            if there
                && let Some(t) = field(&mut self.materials[material])
                && let Ok(n) = u32::try_from(number)
            {
                t.uv = n;
            }
        }
    }

    /// The mesh for geometry `g` as model `m` (0 for none) uses it.
    fn mesh_for(&mut self, g: i64, m: i64) -> Result<usize> {
        let mats = if m == 0 { Vec::new() } else { self.model_materials(m) };
        if let Some(&i) = self.mesh_of.get(&(g, mats.clone())) {
            return Ok(i);
        }
        let geom = self.objs.node(g).context(ErrorKind::Corrupt, "missing geometry")?;
        // The UV set each material slot's color texture reads, by name.
        let color_uv: HashMap<usize, &str> = mats
            .iter()
            .enumerate()
            .filter_map(|(slot, m)| Some((slot, self.color_uv.get(m)?.as_str())))
            .collect();
        let named = !color_uv.is_empty() || mats.iter().any(|m| self.material_uvs.contains_key(m));
        let (prims, uv_names) = geometry(geom, self.limit, &color_uv, named)?;
        // Skin weights per control point: (joint, weight), heaviest first.
        let (clusters, influences) = if self.keep.skins {
            self.influences(g)
        } else {
            (0, Vec::new())
        };
        let channels = if self.keep.morph_targets {
            self.morph_channels(g)
        } else {
            Vec::new()
        };
        let channels: Vec<(i64, Channel)> = channels.iter().filter_map(|&c| Some((c, self.channel(c)?))).collect();
        let shapes: Vec<&Target> = channels.iter().flat_map(|(_, c)| &c.targets).collect();
        // Offsets to normals only for a geometry with normals of its own, and only when some aren't zero.
        let shape_normals = geom.children_by_name("LayerElementNormal").next().is_some()
            && shapes.iter().any(|t| t.normals.values().flatten().any(|c| *c != 0.0));
        let joint_sets = influences.iter().map(|v| v.len().div_ceil(4)).max().unwrap_or(0).min(4);
        let mut primitives = Vec::new();
        for (slot, prim, cps, color_set) in prims {
            let material = match mats.get(slot) {
                Some(&i) => i,
                None => self.default_material(),
            };
            self.bind_uv_sets(material, &prim, &uv_names, color_set);
            if !prim.uvs.is_empty() && !self.textured.contains(&material) {
                self.textured.push(material);
            }
            let normals = shape_normals && !prim.normals.is_empty();
            let Some(mut json) = self.b.primitive(prim) else {
                continue;
            };
            json["material"] = json!(material);
            if joint_sets > 0 {
                let unweighted = clusters; // stands for the model itself, after the clusters; see `skin`
                for set in 0..joint_sets {
                    let mut joints = Vec::with_capacity(cps.len() * 4);
                    let mut weights = Vec::with_capacity(cps.len() * 4);
                    for cp in &cps {
                        let inf = influences.get(*cp).map_or(&[][..], Vec::as_slice);
                        for k in set * 4..set * 4 + 4 {
                            match inf.get(k) {
                                Some(&(j, w)) => {
                                    joints.push(u16::try_from(j).unwrap_or(0));
                                    weights.push(w);
                                }
                                None if k == 0 => {
                                    joints.push(u16::try_from(unweighted).unwrap_or(0));
                                    weights.push(1.0);
                                }
                                None => {
                                    joints.push(0);
                                    weights.push(0.0);
                                }
                            }
                        }
                    }
                    let n = cps.len();
                    let j = self.b.accessor(Part::u16(joints), 5123, n, "VEC4", Some(34962));
                    let w = self.b.accessor(Part::f32(weights), 5126, n, "VEC4", Some(34962));
                    json["attributes"][format!("JOINTS_{set}")] = json!(j);
                    json["attributes"][format!("WEIGHTS_{set}")] = json!(w);
                }
            }
            if !shapes.is_empty() {
                let targets: Vec<Value> = shapes
                    .iter()
                    .map(|t| {
                        let d: Vec<[f32; 3]> = cps
                            .iter()
                            .map(|cp| t.positions.get(cp).copied().unwrap_or([0.0; 3]))
                            .collect();
                        let mut target = json!({ "POSITION": self.b.positions(d) });
                        if normals {
                            let n: Vec<f32> = cps
                                .iter()
                                .flat_map(|cp| t.normals.get(cp).copied().unwrap_or([0.0; 3]))
                                .collect();
                            target["NORMAL"] = json!(self.b.floats(n, "VEC3", false));
                        }
                        target
                    })
                    .collect();
                json["targets"] = json!(targets);
            }
            primitives.push(json);
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(name) = name_of(geom) {
            mesh["name"] = json!(name);
        }
        if !shapes.is_empty() {
            let mut weights = vec![0.0; shapes.len()];
            let mut first = 0;
            let mut morph_ids = Vec::with_capacity(channels.len());
            for (id, c) in &channels {
                split_weight(c.weight, &c.full, &mut weights[first..first + c.targets.len()]);
                morph_ids.push((*id, first));
                self.channel_full.insert(*id, c.full.clone());
                first += c.targets.len();
            }
            mesh["weights"] = json!(weights);
            mesh["extras"] = json!({ "targetNames": shapes.iter().map(|t| t.name.as_str()).collect::<Vec<_>>() });
            self.morphs.insert(self.meshes.len(), morph_ids);
        }
        self.meshes.push(mesh);
        let i = self.meshes.len() - 1;
        if joint_sets > 0 {
            self.skinned_meshes.insert(i);
        }
        self.mesh_of.insert((g, mats), i);
        Ok(i)
    }

    /// The skin deformer's clusters on geometry `g`: how many there are, and by control point, (cluster number,
    /// weight), heaviest first and summing to one. Clusters whose bone isn't a model are left out.
    fn influences(&self, g: i64) -> (usize, Vec<Vec<(usize, f32)>>) {
        let mut out: Vec<Vec<(usize, f32)>> = Vec::new();
        let Some(skin) = self
            .objs
            .children_of(g)
            .find(|d| self.objs.is_kind("Deformer", *d, "Skin"))
        else {
            return (0, out);
        };
        let clusters = self.clusters(skin);
        // Indexes past the control points weight nothing: no more room than those take.
        let cps = self
            .objs
            .node(g)
            .and_then(|n| child_attr(n, "Vertices"))
            .map_or(0, |v| match v {
                A::ArrF64(a) => a.len() / 3,
                A::ArrF32(a) => a.len() / 3,
                _ => 0,
            });
        for (j, &cluster) in clusters.iter().enumerate() {
            let Some(node) = self.objs.node(cluster) else { continue };
            let (Some(idx), Some(w)) = (child_attr(node, "Indexes"), child_attr(node, "Weights")) else {
                continue;
            };
            let (Some(idx), Some(w)) = (ints(idx), floats(w)) else {
                continue;
            };
            for (&cp, &w) in idx.iter().zip(&w) {
                if let Ok(cp) = usize::try_from(cp)
                    && cp < cps
                    && w > 0.0
                    && w.is_finite()
                {
                    if out.len() <= cp {
                        out.resize_with(cp + 1, Vec::new);
                    }
                    out[cp].push((j, f32_of(w)));
                }
            }
        }
        for inf in &mut out {
            inf.sort_by(|a, b| b.1.total_cmp(&a.1));
            inf.truncate(16);
            let sum: f32 = inf.iter().map(|x| x.1).sum();
            if !sum.is_finite() {
                // Weights too large to add up: kept in proportion to the largest.
                let top = inf.first().map_or(1.0, |x| x.1);
                for x in inf.iter_mut() {
                    x.1 /= top;
                }
                let sum: f32 = inf.iter().map(|x| x.1).sum();
                for x in inf.iter_mut() {
                    x.1 /= sum;
                }
                continue;
            }
            for x in inf.iter_mut() {
                x.1 /= sum;
            }
        }
        (clusters.len(), out)
    }

    /// A skin deformer's clusters that name a bone.
    fn clusters(&self, skin: i64) -> Vec<i64> {
        self.objs
            .children_of(skin)
            .filter(|c| self.objs.is_kind("Deformer", *c, "Cluster"))
            .filter(|c| self.objs.children_of(*c).any(|b| self.node_of.contains_key(&b)))
            .collect()
    }

    /// The skin for deformer `skin` on model `m`: a joint per cluster, and the model's own node last, for
    /// vertices no cluster weights.
    fn skin(&mut self, skin: i64, m: i64, geometric: &M4) -> usize {
        if let Some(&s) = self.skin_of.get(&(skin, m)) {
            return s;
        }
        let mut joints = Vec::new();
        let mut ibms = Vec::new();
        for cluster in self.clusters(skin) {
            let Some(node) = self.objs.node(cluster) else { continue };
            let bone = self
                .objs
                .children_of(cluster)
                .find(|b| self.node_of.contains_key(b))
                .unwrap_or(0);
            let matrix = |name: &str| {
                child_attr(node, name)
                    .and_then(floats)
                    .filter(|v| v.len() == 16 && v.iter().all(|x| x.is_finite()))
                    .map(|v| {
                        let mut m = IDENTITY;
                        for (k, x) in v.iter().enumerate() {
                            m[k % 4][k / 4] = *x;
                        }
                        m
                    })
            };
            // `Transform` takes the model's space to the bone's at bind time: the file stores it relative to
            // `TransformLink` (the bone's bind-time world), though the FBX SDK hands back `TransformLink ·
            // Transform`, the model's world. Without it, the bind pose is the pose the file is in.
            let transform = matrix("Transform").unwrap_or_else(|| {
                let link = matrix("TransformLink").unwrap_or_else(|| self.world_of(bone));
                mul(&inverse(&link), &self.world_of(m))
            });
            // The vertex goes geometry → model → bone space.
            ibms.extend(column_major(&mul(&transform, geometric)).map(f32_of));
            joints.push(self.node_of[&bone]);
        }
        // The model's own node: at bind time it's where `Transform` says, so its inverse bind matrix is the
        // geometric transform.
        joints.push(self.node_of[&m]);
        ibms.extend(column_major(geometric).map(f32_of));
        let acc = self.b.floats(ibms, "MAT4", false);
        let mut out = json!({ "joints": joints, "inverseBindMatrices": acc });
        // glTF skins blend linearly; a skin meant to blend otherwise (dual quaternion, or a mix) says so.
        if let Some(kind) = self
            .objs
            .node(skin)
            .and_then(|n| child_attr(n, "SkinningType"))
            .and_then(string)
            .filter(|k| !matches!(*k, "" | "Linear" | "Rigid"))
        {
            out["extras"] = json!({ "skinningType": kind });
        }
        self.skins.push(out);
        let s = self.skins.len() - 1;
        self.skin_of.insert((skin, m), s);
        s
    }

    /// Model `m`'s world transform in the file's space: its parents' times its own.
    fn world_of(&self, m: i64) -> M4 {
        let mut world = IDENTITY;
        let mut at = Some(m);
        // At most one step per model, so a cycle in a broken file ends.
        for _ in 0..=self.node_of.len() {
            let Some(id) = at else { break };
            world = mul(&self.rest_local(id), &world);
            at = self.objs.model_parent(id).filter(|&p| p != id);
        }
        world
    }

    /// A blend shape channel: its weight, and its shapes as targets, ordered by the weight each is reached in full
    /// at (its `FullWeights`, in percent; evenly spread, in the order connected, when those are missing or don't
    /// rise). The full shape is named for the channel, an in-between for its own shape.
    fn channel(&self, channel: i64) -> Option<Channel> {
        let node = self.objs.node(channel)?;
        let shapes: Vec<i64> = self
            .objs
            .children_of(channel)
            .filter(|s| self.objs.is_kind("Geometry", *s, "Shape"))
            .collect();
        if shapes.is_empty() {
            return None;
        }
        let given = child_attr(node, "FullWeights").and_then(floats).unwrap_or_default();
        let mut paired: Vec<(f64, i64)> = shapes
            .iter()
            .enumerate()
            .map(|(k, &s)| (given.get(k).map_or(f64::NAN, |w| w / 100.0), s))
            .collect();
        paired.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rising =
            paired.iter().all(|(w, _)| w.is_finite() && *w > 0.0) && paired.windows(2).all(|w| w[1].0 > w[0].0);
        if !rising {
            #[allow(clippy::cast_precision_loss, reason = "a few shapes")]
            let even = |k: usize| (k + 1) as f64 / shapes.len() as f64;
            paired = shapes.iter().enumerate().map(|(k, &s)| (even(k), s)).collect();
        }
        let name = name_of(node).unwrap_or("");
        let targets = paired
            .iter()
            .enumerate()
            .map(|(k, &(_, s))| {
                let shape = self.objs.node(s);
                let (positions, normals) = shape.map(offsets).unwrap_or_default();
                let name = match shape.and_then(name_of) {
                    Some(own) if k + 1 < paired.len() => own,
                    _ => name,
                };
                Target {
                    name: name.to_owned(),
                    positions,
                    normals,
                }
            })
            .collect();
        let weight = scalar(&self.objs.props(channel), "DeformPercent").unwrap_or(0.0) / 100.0;
        Some(Channel {
            weight,
            full: paired.iter().map(|p| p.0).collect(),
            targets,
        })
    }

    /// Every animation stack as a glTF animation, its curves sampled at their keys and blended through its layers.
    fn animations(&mut self) -> Vec<Value> {
        let stacks: Vec<i64> = self
            .objs
            .order
            .iter()
            .copied()
            .filter(|s| self.objs.is("AnimationStack", *s))
            .collect();
        let mut out = Vec::new();
        for stack in stacks {
            let layers = self.layers(stack);
            // Each curve node of the stack, by the layer it's on.
            let mut layer_of: HashMap<i64, usize> = HashMap::new();
            for (k, l) in layers.iter().enumerate() {
                for c in self.objs.children_of(l.id) {
                    if self.objs.is("AnimationCurveNode", c) {
                        layer_of.entry(c).or_insert(k);
                    }
                }
            }
            if layer_of.is_empty() {
                continue;
            }
            let stack_of = (layers.as_slice(), &layer_of);
            let mut channels = Vec::new();
            let mut samplers = Vec::new();
            let mut models: Vec<(i64, usize)> = self.node_of.iter().map(|(&m, &n)| (m, n)).collect();
            models.sort_by_key(|(_, n)| *n);
            for (m, node) in models {
                let local = Local::new(&self.objs.props(m));
                let rest = [local.t, local.r, local.s];
                let tracks: [Option<Layered>; 3] = std::array::from_fn(|k| {
                    let prop = ["Lcl Translation", "Lcl Rotation", "Lcl Scaling"][k];
                    self.layered(
                        m,
                        prop,
                        ["d|X", "d|Y", "d|Z"],
                        rest[k],
                        k == 2,
                        (k == 1).then_some(local.order),
                        stack_of,
                    )
                });
                // A model taking its parent's scale off moves with that scale's animation too.
                let parent_rest = self.parent_scale(&local, m);
                let parent = if local.ignore_parent_scale {
                    self.objs.model_parent(m).filter(|&p| p != m).and_then(|p| {
                        self.layered(
                            p,
                            "Lcl Scaling",
                            ["d|X", "d|Y", "d|Z"],
                            parent_rest,
                            true,
                            None,
                            stack_of,
                        )
                    })
                } else {
                    None
                };
                if tracks.iter().any(Option::is_some) || parent.is_some() {
                    let under = (parent.as_ref(), parent_rest);
                    self.transform_channels(node, &local, &tracks, under, &mut channels, &mut samplers);
                }
            }
            // Morph weights: every channel of a mesh at once, per node drawing it.
            let mut weight_nodes: HashMap<usize, Vec<(usize, Vec<f64>, Layered)>> = HashMap::new();
            let users: Vec<(i64, Vec<(usize, usize)>)> =
                self.channel_users.iter().map(|(c, u)| (*c, u.clone())).collect();
            for (ch, users) in users {
                let percent = scalar(&self.objs.props(ch), "DeformPercent").unwrap_or(0.0);
                let full = self.channel_full.get(&ch).cloned().unwrap_or_else(|| vec![1.0]);
                for (node, first) in users {
                    let rest = [percent, 0.0, 0.0];
                    if let Some(a) = self.layered(
                        ch,
                        "DeformPercent",
                        ["d|DeformPercent", "", ""],
                        rest,
                        false,
                        None,
                        stack_of,
                    ) {
                        weight_nodes.entry(node).or_default().push((first, full.clone(), a));
                    }
                }
            }
            let mut weight_nodes: Vec<_> = weight_nodes.into_iter().collect();
            weight_nodes.sort_by_key(|(n, _)| *n);
            for (node, tracks) in weight_nodes {
                self.weight_channel(node, &tracks, &mut channels, &mut samplers);
            }
            if channels.is_empty() {
                continue;
            }
            let mut anim = json!({ "channels": channels, "samplers": samplers });
            if let Some(name) = self.objs.node(stack).and_then(name_of) {
                anim["name"] = json!(name);
            }
            // The stack's own span, which may be wider or narrower than its keys.
            let p = self.objs.props(stack);
            if let (Some(start), Some(stop)) = (scalar(&p, "LocalStart"), scalar(&p, "LocalStop"))
                && stop > start
            {
                anim["extras"] = json!({ "start": start / KTIME, "stop": stop / KTIME });
            }
            out.push(anim);
        }
        out
    }

    /// A stack's animation layers that play, bottom first: muted ones are left out, and when any is soloed, all but
    /// those. The bottom one is the base, which always overrides.
    fn layers(&self, stack: i64) -> Vec<AnimLayer> {
        let all: Vec<(i64, Props<'_>)> = self
            .objs
            .children_of(stack)
            .filter(|l| self.objs.is("AnimationLayer", *l))
            .map(|l| (l, self.objs.props(l)))
            .collect();
        let solo = all.iter().any(|(_, p)| flag(p, "Solo", false));
        let mut out: Vec<AnimLayer> = Vec::new();
        for (id, p) in all {
            if flag(&p, "Mute", false) || (solo && !flag(&p, "Solo", false)) {
                continue;
            }
            out.push(AnimLayer {
                id,
                additive: !out.is_empty() && choice(&p, "BlendMode", 0) == 0,
                weight: (scalar(&p, "Weight").unwrap_or(100.0) / 100.0).clamp(0.0, 1.0),
                multiply_scale: choice(&p, "ScaleAccumulationMode", 0) == 0,
            });
        }
        out
    }

    /// Property `prop` of object `owner` as a stack animates it through its layers, from its value at rest; `None`
    /// when no layer of the stack animates it. A rotation (given its order) is turned the short way layer by layer.
    #[allow(clippy::too_many_arguments, reason = "one property's whole description")]
    fn layered(
        &self,
        owner: i64,
        prop: &str,
        components: [&str; 3],
        rest: [f64; 3],
        is_scale: bool,
        order: Option<i32>,
        (layers, layer_of): (&[AnimLayer], &HashMap<i64, usize>),
    ) -> Option<Layered> {
        let mut nodes: Vec<(usize, i64)> = self
            .objs
            .child_at(owner, prop)
            .filter_map(|cn| layer_of.get(&cn).map(|&k| (k, cn)))
            .collect();
        if nodes.is_empty() {
            return None;
        }
        nodes.sort_unstable();
        nodes.dedup_by_key(|(k, _)| *k);
        let mut out = Layered {
            rest,
            layers: Vec::new(),
        };
        for (k, cn) in nodes {
            let Some(mut a) = self.animated(cn, components) else {
                continue;
            };
            if let Some(order) = order {
                a.unwrap_rotation(order);
            }
            let l = layers[k];
            let blend = match (l.additive, is_scale && l.multiply_scale) {
                (false, _) => Blend::Override,
                (true, true) => Blend::Multiply,
                (true, false) => Blend::Add,
            };
            out.layers.push((blend, l.weight, a));
        }
        (!out.layers.is_empty()).then_some(out)
    }

    /// A curve node's curves for its components (`d|X`, ...; "" for none), with its own values where a component
    /// has no curve.
    fn animated(&self, cn: i64, components: [&str; 3]) -> Option<Animated> {
        let p = self.objs.props(cn);
        let curves = components.map(|c| {
            if c.is_empty() {
                return None;
            }
            let curve = self.objs.child_at(cn, c).find(|k| self.objs.is("AnimationCurve", *k))?;
            Curve::read(self.objs.node(curve)?)
        });
        if curves.iter().all(Option::is_none) {
            return None;
        }
        let fixed = components.map(|c| scalar(&p, c).unwrap_or(0.0));
        Some(Animated { curves, fixed })
    }

    /// Translation, rotation and scale channels for a node whose `Lcl` properties are animated (or whose parent's
    /// scale is, when it takes that off): its whole local transform is evaluated at each time (pivots and
    /// pre-rotations included) and taken apart again.
    fn transform_channels(
        &mut self,
        node: usize,
        local: &Local,
        tracks: &[Option<Layered>; 3],
        (parent, parent_rest): (Option<&Layered>, [f64; 3]),
        channels: &mut Vec<Value>,
        samplers: &mut Vec<Value>,
    ) {
        let mut times = Vec::new();
        for t in tracks.iter().flatten().chain(parent) {
            t.times(&mut times);
        }
        let mut times = timeline(times);
        // Cut long turns into short steps.
        if let Some(r) = &tracks[1] {
            let mut fine = Vec::with_capacity(times.len());
            for w in times.windows(2) {
                fine.push(w[0]);
                let (a, b) = (r.at(w[0]), r.at(w[1] - 1e-9));
                let turn = (0..3).map(|k| (b[k] - a[k]).abs()).fold(0.0, f64::max);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a small step count"
                )]
                let n = ((turn / MAX_STEP_DEGREES).ceil() as usize).min(360);
                for k in 1..n {
                    #[allow(clippy::cast_precision_loss, reason = "a small step count")]
                    fine.push(w[0] + (w[1] - w[0]) * k as f64 / n as f64);
                }
            }
            fine.extend(times.last());
            times = fine;
        }
        if times.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let finite = |m: M4| m.iter().flatten().all(|c| c.is_finite()).then_some(m);
        let rest_m = finite(local.at(None, None, None, parent_rest)).unwrap_or(IDENTITY);
        let at = |k: usize, t: f64| tracks[k].as_ref().map(|a| a.at(t));
        let mut trs: Vec<Trs> = times
            .iter()
            .map(|&t| {
                let under = parent.map_or(parent_rest, |p| p.at(t));
                decompose(&finite(local.at(at(0, t), at(1, t), at(2, t), under)).unwrap_or(rest_m))
            })
            .collect();
        // Keep each quaternion on the same side as the last, so slerp takes the short way.
        for i in 1..trs.len() {
            let prev = trs[i - 1].r;
            let q = &mut trs[i].r;
            if (0..4).map(|k| prev[k] * q[k]).sum::<f64>() < 0.0 {
                *q = q.map(|c| -c);
            }
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let rest = decompose(&rest_m);
        let paths: [(&str, Vec<f64>, Vec<f64>); 3] = [
            ("translation", trs.iter().flat_map(|x| x.t).collect(), rest.t.to_vec()),
            ("rotation", trs.iter().flat_map(|x| x.r).collect(), rest.r.to_vec()),
            ("scale", trs.iter().flat_map(|x| x.s).collect(), rest.s.to_vec()),
        ];
        for (path, values, rest) in paths {
            let width = rest.len();
            let still = values
                .chunks_exact(width)
                .all(|v| v.iter().zip(&rest).all(|(a, b)| (a - b).abs() < 1e-7));
            if still {
                continue;
            }
            let output = self.b.floats(
                values.into_iter().map(f32_of).collect(),
                if width == 4 { "VEC4" } else { "VEC3" },
                false,
            );
            samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
            channels.push(json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": path } }));
        }
        // An animated node holds TRS, never a matrix.
        if self.nodes[node]
            .as_object_mut()
            .and_then(|n| n.remove("matrix"))
            .is_some()
            && [rest.t.as_slice(), &rest.r, &rest.s]
                .iter()
                .all(|v| v.iter().all(|c| c.is_finite()))
        {
            let n = &mut self.nodes[node];
            n["translation"] = json!(rest.t);
            n["rotation"] = json!(rest.r);
            n["scale"] = json!(rest.s);
        }
    }

    /// A weights channel for a node whose mesh's blend shape channels are animated: per channel, its first target,
    /// the weights its targets are reached at, and its weight in percent.
    fn weight_channel(
        &mut self,
        node: usize,
        tracks: &[(usize, Vec<f64>, Layered)],
        channels: &mut Vec<Value>,
        samplers: &mut Vec<Value>,
    ) {
        let Some(mesh) = self.nodes[node]["mesh"].as_u64().and_then(|m| usize::try_from(m).ok()) else {
            return;
        };
        let rest: Vec<f64> = self.meshes[mesh]["weights"]
            .as_array()
            .map(|w| w.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let mut times = Vec::new();
        for (_, _, a) in tracks {
            a.times(&mut times);
        }
        let times = timeline(times);
        if times.is_empty() || rest.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let mut values = Vec::with_capacity(times.len() * rest.len());
        let mut w = rest.clone();
        for &t in &times {
            w.copy_from_slice(&rest);
            for (first, full, a) in tracks {
                if let Some(out) = w.get_mut(*first..first + full.len()) {
                    split_weight(a.at(t)[0] / 100.0, full, out);
                }
            }
            values.extend(w.iter().copied().map(f32_of));
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let output = self.b.floats(values, "SCALAR", false);
        samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
        channels.push(json!({ "sampler": samplers.len() - 1, "target": { "node": node, "path": "weights" } }));
    }
}

/// A blend shape channel's weight (a fraction; not clamped, so a rig may push past full or below none) shared
/// among its targets, each reached in full at its weight in `full` (ascending): between two, the two blend; past
/// the last (or before the first), the nearest two carry on as they were going.
fn split_weight(w: f64, full: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    let n = full.len().min(out.len());
    if n == 0 || !w.is_finite() {
        return;
    }
    if n == 1 || w <= full[0] {
        out[0] = w / full[0];
        return;
    }
    let i = full[..n].partition_point(|&f| f < w).clamp(1, n - 1);
    let (lo, hi) = (full[i - 1], full[i]);
    out[i - 1] = (hi - w) / (hi - lo);
    out[i] = (w - lo) / (hi - lo);
}

/// A camera's glTF form. FBX gives the field of view horizontally unless its aperture mode says otherwise.
fn camera(p: &Props<'_>, unit: f64) -> Value {
    let aspect = match (scalar(p, "AspectWidth"), scalar(p, "AspectHeight")) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => w / h,
        _ => 16.0 / 9.0,
    };
    // Planes are in the file's units, but a node's scale doesn't reach a camera's planes: they're converted here.
    let near = scalar(p, "NearPlane").filter(|n| *n > 0.0).unwrap_or(10.0);
    let far = scalar(p, "FarPlane").filter(|f| *f > near).unwrap_or(near * 10_000.0);
    let (near, far) = (near * unit, far * unit);
    if scalar(p, "CameraProjectionType") == Some(1.0) {
        let zoom = scalar(p, "OrthoZoom").filter(|z| *z > 0.0).unwrap_or(1.0);
        return json!({ "type": "orthographic", "orthographic": { "xmag": zoom * aspect, "ymag": zoom, "znear": near, "zfar": far } });
    }
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    let mode = scalar(p, "ApertureMode").map_or(1, |m| m as i32);
    let deg = scalar(p, "FieldOfView")
        .filter(|f| *f > 0.0 && *f < 180.0)
        .unwrap_or(40.0);
    let yfov = match mode {
        2 => deg.to_radians(),
        0 => scalar(p, "FieldOfViewY")
            .filter(|f| *f > 0.0 && *f < 180.0)
            .unwrap_or(deg)
            .to_radians(),
        3 => match (scalar(p, "FocalLength"), scalar(p, "FilmHeight")) {
            (Some(f), Some(h)) if f > 0.0 && h > 0.0 => 2.0 * (h * 25.4 / 2.0 / f).atan(),
            _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
        },
        _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
    };
    json!({ "type": "perspective", "perspective": { "yfov": yfov, "aspectRatio": aspect, "znear": near, "zfar": far } })
}

/// A light's `KHR_lights_punctual` form: point, directional or spot. An area light, which shines from one side of
/// its shape, becomes a spot opening to a half sphere, and a volume light a point; their own kind is kept in the
/// light's extras, as is any falloff other than the inverse square.
fn light(p: &Props<'_>) -> Option<Value> {
    let kind = choice(p, "LightType", 0);
    let color = vec3(p, "Color", [1.0; 3]).map(|c| srgb_to_linear(f32_of(c.clamp(0.0, 1.0))));
    let intensity = scalar(p, "Intensity").unwrap_or(100.0).max(0.0) / 100.0;
    let mut out = json!({ "color": color, "intensity": intensity });
    match kind {
        0 => out["type"] = json!("point"),
        1 => out["type"] = json!("directional"),
        2 => {
            let half = std::f64::consts::FRAC_PI_2;
            let outer = (scalar(p, "OuterAngle").unwrap_or(45.0).to_radians() / 2.0).clamp(1e-4, half);
            let inner = (scalar(p, "InnerAngle").unwrap_or(0.0).to_radians() / 2.0).clamp(0.0, outer * 0.999);
            out["type"] = json!("spot");
            out["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
        }
        3 => {
            out["type"] = json!("spot");
            out["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": std::f64::consts::FRAC_PI_2 });
            let shape = if choice(p, "AreaLightShape", 0) == 1 {
                "sphere"
            } else {
                "rectangle"
            };
            out["extras"] = json!({ "lightType": "area", "areaLightShape": shape });
        }
        4 => {
            out["type"] = json!("point");
            out["extras"] = json!({ "lightType": "volume" });
        }
        _ => return None,
    }
    // FBX's falloff: none, linear, quadratic (glTF's own) or cubic.
    let decay = choice(p, "DecayType", 2);
    if decay != 2 && kind != 1 {
        let name = match decay {
            0 => "none",
            1 => "linear",
            3 => "cubic",
            _ => "unknown",
        };
        out["extras"]["decayType"] = json!(name);
    }
    Some(out)
}

/// "Mesh" in `Geometry: 123, "Geometry::Name", "Mesh"`.
fn kind_of(node: NodeHandle<'_>) -> &str {
    node.attributes().get(2).and_then(string).unwrap_or("")
}

/// A `Line` geometry's polylines as one primitive's segments, on the control points (`Points`) they use. Each
/// runs through `PointsIndex` to an index written negative (`!i`), its last point; an index that isn't a whole
/// number, or a control point a float can hold, breaks it there.
fn line_geometry(geom: NodeHandle<'_>) -> Prim {
    let points = child_attr(geom, "Points").and_then(floats).unwrap_or_default();
    let index = child_attr(geom, "PointsIndex")
        .and_then(|a| ints(a).or_else(|| floats(a).map(|f| f.into_iter().map(whole).collect())))
        .unwrap_or_default();
    let point = |cp: usize| [points[cp * 3], points[cp * 3 + 1], points[cp * 3 + 2]].map(f32_of);
    let mut prim = Prim::default();
    // Control point → vertex, for those a segment uses: no more room than the points take.
    let mut vertex = vec![u32::MAX; points.len() / 3];
    let mut last = None;
    for i in index {
        let (cp, end) = if i < 0 { (!i, true) } else { (i, false) };
        let cp = usize::try_from(cp)
            .ok()
            .filter(|&cp| cp < vertex.len() && point(cp).iter().all(|c| c.is_finite()));
        if let (Some(a), Some(b)) = (last, cp)
            && a != b
        {
            for c in [a, b] {
                if vertex[c] == u32::MAX {
                    let Ok(v) = u32::try_from(prim.positions.len()) else {
                        return prim;
                    };
                    vertex[c] = v;
                    prim.positions.push(point(c));
                }
                prim.lines.push(vertex[c]);
            }
        }
        last = if end { None } else { cp };
    }
    prim
}

/// Whether `NurbsCurve` geometry has what drawing it takes: an `Order` (a whole number from 2) and `Points`.
fn is_curve(geom: NodeHandle<'_>) -> bool {
    child_attr(geom, "Order")
        .and_then(num)
        .map(whole)
        .is_some_and(|o| (2..i64::MAX).contains(&o))
        && child_attr(geom, "Points")
            .and_then(floats)
            .is_some_and(|p| p.len() >= 8)
}

/// A `NurbsCurve` geometry as one primitive's segments: the curve of its `Order` (its degree plus one) through its
/// `Points` (x, y, z and a weight each) over its `KnotVector`, a `Periodic` one wrapped around to its first points
/// again. A curve whose order or knots don't fit its points, or that takes more than `budget` points, is drawn
/// through its control points; so is one with control points a float can't hold, broken there. The primitive, and
/// the points it took from the budget.
fn curve_geometry(geom: NodeHandle<'_>, budget: usize) -> (Prim, usize) {
    if !is_curve(geom) {
        return (Prim::default(), 0);
    }
    let raw = child_attr(geom, "Points").and_then(floats).unwrap_or_default();
    let mut ctrl: Vec<[f64; 3]> = raw.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).collect();
    let mut weights: Vec<f64> = raw.chunks_exact(4).map(|c| c[3]).collect();
    let order = child_attr(geom, "Order").and_then(num).map_or(0, whole);
    let degree = order - 1;
    let knots = child_attr(geom, "KnotVector").and_then(floats).unwrap_or_default();
    let periodic = child_attr(geom, "Form").and_then(string) == Some("Periodic");
    let n = ctrl.len();
    // A periodic curve's knots run past its points by its degree: those points are its first ones again.
    if periodic
        && let Ok(p) = usize::try_from(degree)
        && p < n
        && knots.len() == n + 2 * p + 1
    {
        ctrl.extend_from_within(..p);
        weights.extend_from_within(..p);
    }
    let finite = ctrl.iter().flatten().all(|c| c.is_finite());
    let mut pts = Vec::new();
    let drawn = if let Some(Ok(drawn)) = finite.then(|| nurbs::curve(degree, &ctrl, &weights, &knots, budget, &mut pts))
    {
        drawn
    } else {
        pts.clone_from(&ctrl);
        if periodic {
            pts.truncate(n);
            pts.push(ctrl[0]);
        }
        0
    };
    let mut prim = Prim::default();
    // The vertex the polyline is at: none past a point that isn't a number. One point twice draws nothing.
    let mut last: Option<u32> = None;
    for p in pts {
        let p = p.map(f32_of);
        if !p.iter().all(|c| c.is_finite()) {
            last = None;
            continue;
        }
        #[allow(clippy::float_cmp, reason = "the same point twice, exactly")]
        if last.is_some_and(|v| prim.positions[v as usize] == p) {
            continue;
        }
        let Ok(v) = u32::try_from(prim.positions.len()) else {
            break;
        };
        prim.positions.push(p);
        if let Some(u) = last {
            prim.lines.extend_from_slice(&[u, v]);
        }
        last = Some(v);
    }
    (prim, drawn)
}

/// A whole number written as a float, as an index; anything else as one no array has.
fn whole(f: f64) -> i64 {
    #[allow(clippy::cast_possible_truncation, reason = "a whole number in range")]
    if f.fract() == 0.0 && f.abs() < 2f64.powi(62) {
        f as i64
    } else {
        i64::MAX
    }
}

/// The material slot a texture connects to, from the property it connects as: the classic slots (`DiffuseColor`,
/// `NormalMap`, ...) and PBR shaders' (`Maya|TEX_color_map`, `Maya|baseColor`, `3dsMax|main|base_color_map`,
/// `3dsMax|Parameters|base_color_map`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Color,
    Normal,
    Emissive,
    Occlusion,
    Metallic,
    Roughness,
    /// Roughness in green and metalness in blue, glTF's own arrangement.
    MetalRough,
    /// Occlusion, roughness and metalness in red, green and blue.
    OcclusionMetalRough,
    Opacity,
    Transmission,
    Coat,
    CoatRoughness,
    CoatNormal,
    SheenColor,
    SheenRoughness,
}

fn slot(prop: &str) -> Option<Slot> {
    let s: String = prop
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| s.contains(w));
    Some(if has(&["occlusionroughnessmetal", "aoroughnessmetal"]) {
        Slot::OcclusionMetalRough
    } else if has(&["metallicroughness", "metalroughness", "roughnessmetal"]) {
        Slot::MetalRough
    } else if has(&["coatnormal", "coatbump"]) {
        Slot::CoatNormal
    } else if has(&["coatroughness"]) {
        Slot::CoatRoughness
    } else if has(&["sheenroughness", "fuzzroughness"]) {
        Slot::SheenRoughness
    } else if has(&["sheencolor", "fuzzcolor"]) {
        Slot::SheenColor
    } else if has(&["coatcolor", "coatior", "coataffect", "transmissioncolor", "transcolor"]) {
        return None;
    } else if s.ends_with("coat") || s.ends_with("coatweight") || has(&["coatmap", "coatweightmap"]) {
        Slot::Coat
    } else if s.ends_with("transmission") || has(&["transmissionweight", "transweight", "transmissionmap"]) {
        Slot::Transmission
    } else if has(&["emissive", "emitcolor", "emission"]) {
        Slot::Emissive
    } else if s.contains("normal") || s == "bump" || s.ends_with("bumpmap") || s.ends_with("normmap") {
        Slot::Normal
    } else if has(&["aomap", "ambientocclusion"]) {
        Slot::Occlusion
    } else if has(&["metallic", "metalness"]) {
        Slot::Metallic
    } else if s.contains("roughness") {
        Slot::Roughness
    } else if has(&["transparentcolor", "transparencyfactor", "opacity", "cutout"]) {
        Slot::Opacity
    } else if s.contains("diffusecolor") || s == "diffuse" || s.contains("basecolor") || s.ends_with("texcolormap") {
        Slot::Color
    } else {
        return None;
    })
}

/// A PBR shader's parameters, as tools namespace them on a material (`Maya|baseColor`, `3dsMax|main|basecolor`,
/// `3dsMax|Parameters|base_color`, ...), found by their last part's name without case or underscores.
struct Pbr<'p> {
    by: HashMap<String, &'p [A<'p>]>,
}

impl<'p> Pbr<'p> {
    fn of(p: &'p Props<'_>) -> Self {
        let by = p
            .own
            .iter()
            .filter_map(|(k, v)| {
                let (_, last) = k.rsplit_once('|')?;
                let name: String = last
                    .chars()
                    .filter(char::is_ascii_alphanumeric)
                    .collect::<String>()
                    .to_lowercase();
                Some((name, v.as_slice()))
            })
            .collect();
        Self { by }
    }

    fn get(&self, names: &[&str]) -> Option<&'p [A<'p>]> {
        names.iter().find_map(|n| self.by.get(*n).copied())
    }

    /// A number: the first of the names given that's there and finite.
    fn f(&self, names: &[&str]) -> Option<f32> {
        names
            .iter()
            .find_map(|n| self.by.get(*n)?.first().and_then(num).filter(|v| v.is_finite()))
            .map(f32_of)
    }

    /// A color (or a number, as a gray).
    fn color(&self, names: &[&str]) -> Option<[f64; 3]> {
        let v = self.get(names)?;
        let c: Vec<f64> = v.iter().filter_map(num).filter(|x| x.is_finite()).collect();
        match c.as_slice() {
            [r, g, b, ..] => Some([*r, *g, *b]),
            [x] => Some([*x; 3]),
            _ => None,
        }
    }
}

/// A material's PBR shader, when it has one: standard surfaces, physical materials, metal/rough and spec/gloss
/// PBR materials, glTF materials and `OpenPBR`. They name their parameters alike, so one reading covers them;
/// each name list holds the synonyms the shaders use. Gives the base color's weight.
fn read_pbr(mat: &mut Full, p: &Props<'_>, to_linear: &dyn Fn([f64; 3]) -> [f32; 3]) -> Option<f32> {
    let pbr = Pbr::of(p);
    let base = pbr.color(&["basecolor", "basecolour"])?;
    let weight = pbr.f(&["baseweight", "base"]).unwrap_or(1.0).max(0.0);
    let c = to_linear(base);
    mat.base_color[..3].copy_from_slice(&c.map(|c| c * weight));
    // Spec/gloss has no metalness: its specular color stands for it.
    mat.metallic = pbr
        .f(&["metalness", "metallic", "basemetalness"])
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let glossy = pbr.f(&["roughnessinv", "useglossiness"]).is_some_and(|i| i > 0.5);
    mat.roughness = match (pbr.f(&["roughness", "specularroughness"]), pbr.f(&["glossiness"])) {
        (Some(r), _) if glossy => 1.0 - r,
        (Some(r), _) => r,
        (None, Some(g)) => 1.0 - g,
        (None, None) => 0.5,
    }
    .clamp(0.0, 1.0);
    let emit = pbr
        .color(&["emissive", "emissioncolor", "emitcolor"])
        .map_or([0.0; 3], to_linear);
    let strength = pbr
        .f(&["emissiveintensity", "emission", "emissionweight", "emissionluminance"])
        .unwrap_or(1.0)
        .max(0.0);
    mat.emissive = emit.map(|c| c * strength);
    if let Some(t) = pbr
        .f(&["transmission", "transmissionweight", "transweight"])
        .filter(|t| *t > 0.0)
    {
        mat.transmission = Some(Factor::of(t.min(1.0)));
    }
    mat.ior = pbr
        .f(&["specularior", "ior", "transior", "trans_ior"])
        .filter(|i| *i >= 1.0);
    if let Some(c) = pbr.f(&["coat", "coatweight"]).filter(|c| *c > 0.0) {
        mat.clearcoat = Some(Clearcoat {
            factor: Factor::of(c.min(1.0)),
            roughness: Factor::of(pbr.f(&["coatroughness"]).unwrap_or(0.0).clamp(0.0, 1.0)),
            normal: None,
        });
    }
    if let Some(s) = pbr.f(&["sheen", "sheenweight", "fuzzweight"]).filter(|s| *s > 0.0) {
        let color = pbr.color(&["sheencolor", "fuzzcolor"]).map_or([1.0; 3], to_linear);
        mat.sheen = Some(Sheen {
            color: color.map(|c| c * s),
            color_tex: None,
            roughness: Factor::of(
                pbr.f(&["sheenroughness", "fuzzroughness"])
                    .unwrap_or(0.3)
                    .clamp(0.0, 1.0),
            ),
        });
    }
    // A specular color without metalness (spec/gloss): the tint of its reflections.
    if pbr.f(&["metalness", "metallic", "basemetalness"]).is_none()
        && let Some(s) = pbr.color(&["specular", "specularcolor"])
    {
        mat.specular = Some(material::Specular {
            factor: Factor::of(1.0),
            color: to_linear(s),
            color_tex: None,
        });
    }
    if let Some(o) = pbr.color(&["opacity", "geometryopacity"]) {
        let a = f32_of((o[0] + o[1] + o[2]) / 3.0).clamp(0.0, 1.0);
        mat.base_color[3] = a;
        if a < 1.0 {
            mat.alpha = Alpha::Blend;
        }
    }
    if pbr.f(&["doublesided"]).is_some_and(|d| d > 0.5) {
        mat.double_sided = true;
    }
    if pbr.f(&["unlit"]).is_some_and(|u| u > 0.5) {
        mat.unlit = true;
    }
    Some(weight)
}

/// The texture a material's slot takes from object `id`: the object itself, or for a layered texture, its first
/// layer.
fn texture_of(id: i64, objs: &Objects<'_>) -> i64 {
    if objs.is("LayeredTexture", id) {
        objs.children_of(id).find(|t| objs.is("Texture", *t)).unwrap_or(id)
    } else {
        id
    }
}

/// The UV set a texture reads, when it names one other than the first.
fn texture_uv_set(id: i64, objs: &Objects<'_>) -> Option<String> {
    let p = objs.props(id);
    let name = p.text("UVSet")?.split('\0').next()?;
    (!name.is_empty() && name != "default").then(|| name.to_owned())
}

/// A texture field of a material.
type TexField = fn(&mut Full) -> Option<&mut Tex>;

/// A material, the UV set its color texture reads by name, and the sets its other textures read, where they name
/// one.
type Read = (Full, Option<String>, Vec<(TexField, String)>);

#[allow(clippy::too_many_lines, reason = "one property after another")]
fn read_material(
    id: i64,
    objs: &Objects<'_>,
    linear: bool,
    dir: &Path,
    // None when textures aren't kept: they aren't read.
    assets: Option<&dyn Assets>,
    images: &mut Images,
    image_index: &mut HashMap<i64, Option<usize>>,
) -> Read {
    let (node, _) = objs.nodes[&id];
    let p = objs.props_of(node);
    let name = node
        .attributes()
        .get(1)
        .and_then(string)
        .unwrap_or("material")
        .split('\0')
        .next()
        .unwrap_or("material")
        .to_owned();
    let to_linear = |c: [f64; 3]| {
        c.map(|v| {
            let v = f32_of(v.clamp(0.0, 1.0));
            if linear { v } else { srgb_to_linear(v) }
        })
    };
    let d = to_linear(vec3(&p, "DiffuseColor", vec3(&p, "Diffuse", [0.8, 0.8, 0.8])));
    // Opacity and reflection are read only where the material sets them: the file's defaults for them are no
    // statement about this one.
    let alpha = p
        .own_scalar("Opacity")
        .unwrap_or_else(|| 1.0 - p.own_scalar("TransparencyFactor").unwrap_or(0.0).clamp(0.0, 1.0));
    let mut mat = Full::from(Material::plain(
        &name,
        [d[0], d[1], d[2], f32_of(alpha.clamp(0.0, 1.0))],
    ));
    // The emissive color is a color (sRGB unless Blender's); its factor is a linear intensity.
    let factor = f32_of(scalar(&p, "EmissiveFactor").unwrap_or(1.0).max(0.0));
    mat.emissive = to_linear(vec3(&p, "EmissiveColor", vec3(&p, "Emissive", [0.0; 3]))).map(|c| c * factor);
    // A Phong exponent as roughness, 1 - sqrt(n) / 10, the inverse of the common export mapping.
    if let Some(n) = scalar(&p, "Shininess").or_else(|| scalar(&p, "ShininessExponent")) {
        mat.roughness = f32_of(1.0 - n.clamp(0.0, 100.0).sqrt() / 10.0);
    }
    // Blender keeps its metallic in ReflectionFactor; other tools' reflectivity is a mirror blend, not metal.
    if linear && let Some(m) = p.own_scalar("ReflectionFactor") {
        mat.metallic = f32_of(m.clamp(0.0, 1.0));
    }
    let pbr = read_pbr(&mut mat, &p, &to_linear);
    let mut uv_sets: Vec<(TexField, String)> = Vec::new();
    let mut color_set = None;
    let (mut metallic, mut roughness, mut opacity) = (None, None, None);
    for (link, prop) in objs.children.get(&id).cloned().unwrap_or_default() {
        let Some(slot) = slot(&prop) else {
            continue;
        };
        let Some(assets) = assets else {
            continue;
        };
        let tex = texture_of(link, objs);
        let img = *image_index
            .entry(tex)
            .or_insert_with(|| texture_image(tex, objs, dir, assets, images));
        let Some(t) = img.map(|i| texture_tex(tex, objs, i)) else {
            continue;
        };
        let field: Option<TexField> = match slot {
            Slot::Color if mat.base_color_tex.is_none() => {
                // A texture takes the place of the color it's connected to (a classic material's diffuse factor is
                // usually a leftover): it's shown as painted, scaled only by a PBR shader's base weight.
                mat.base_color[..3].fill(pbr.unwrap_or(1.0));
                mat.base_color_tex = Some(t);
                Some(|m| m.base_color_tex.as_mut())
            }
            Slot::Normal if mat.normal_tex.is_none() => {
                mat.normal_tex = Some(t);
                Some(|m| m.normal_tex.as_mut())
            }
            Slot::Emissive if mat.emissive_tex.is_none() => {
                // An emissive map with no emissive color would show nothing: it's shown as painted.
                if mat.emissive.iter().all(|c| *c == 0.0) {
                    mat.emissive = [1.0; 3];
                }
                mat.emissive_tex = Some(t);
                Some(|m| m.emissive_tex.as_mut())
            }
            Slot::Occlusion if mat.occlusion_tex.is_none() => {
                mat.occlusion_tex = Some(t);
                Some(|m| m.occlusion_tex.as_mut())
            }
            Slot::MetalRough | Slot::OcclusionMetalRough if mat.metallic_roughness_tex.is_none() => {
                if slot == Slot::OcclusionMetalRough && mat.occlusion_tex.is_none() {
                    mat.occlusion_tex = Some(t.clone());
                }
                mat.metallic_roughness_tex = Some(t);
                mat.metallic = 1.0;
                mat.roughness = 1.0;
                Some(|m| m.metallic_roughness_tex.as_mut())
            }
            Slot::Metallic => {
                metallic = Some(t);
                None
            }
            Slot::Roughness => {
                roughness = Some(t);
                None
            }
            Slot::Opacity => {
                opacity = Some(t);
                None
            }
            Slot::Transmission => {
                let tr = mat.transmission.get_or_insert_with(|| Factor::of(1.0));
                tr.tex.get_or_insert(t);
                Some(|m| m.transmission.as_mut()?.tex.as_mut())
            }
            Slot::Coat => {
                let coat = mat.clearcoat.get_or_insert_with(|| Clearcoat {
                    factor: Factor::of(1.0),
                    roughness: Factor::of(0.0),
                    normal: None,
                });
                coat.factor.tex.get_or_insert(t);
                Some(|m| m.clearcoat.as_mut()?.factor.tex.as_mut())
            }
            Slot::CoatRoughness => {
                if let Some(coat) = &mut mat.clearcoat {
                    coat.roughness.tex.get_or_insert(t);
                }
                Some(|m| m.clearcoat.as_mut()?.roughness.tex.as_mut())
            }
            Slot::CoatNormal => {
                if let Some(coat) = &mut mat.clearcoat {
                    coat.normal.get_or_insert(t);
                }
                Some(|m| m.clearcoat.as_mut()?.normal.as_mut())
            }
            Slot::SheenColor => {
                if let Some(sheen) = &mut mat.sheen {
                    sheen.color_tex.get_or_insert(t);
                }
                Some(|m| m.sheen.as_mut()?.color_tex.as_mut())
            }
            Slot::SheenRoughness => {
                if let Some(sheen) = &mut mat.sheen {
                    sheen.roughness.tex.get_or_insert(t);
                }
                Some(|m| m.sheen.as_mut()?.roughness.tex.as_mut())
            }
            _ => None,
        };
        if let (Some(field), Some(set)) = (field, texture_uv_set(tex, objs)) {
            if slot == Slot::Color {
                color_set = Some(set);
            } else {
                uv_sets.push((field, set));
            }
        }
    }
    // glTF reads roughness and metalness from one image's green and blue.
    if mat.metallic_roughness_tex.is_none()
        && let (Some(m), Some(r)) = (&metallic, &roughness)
        && m.image == r.image
    {
        mat.metallic_roughness_tex.clone_from(&roughness);
        mat.metallic = 1.0;
        mat.roughness = 1.0;
    }
    // Opacity from the color texture is its alpha, glTF's own arrangement.
    if let (Some(o), Some(c)) = (&opacity, &mat.base_color_tex)
        && o.image == c.image
        && mat.alpha == Alpha::Opaque
    {
        mat.alpha = Alpha::Blend;
    }
    (mat, color_set, uv_sets)
}

/// A texture of `image` as texture `id` lays it out: its wrap modes, and its translation, rotation and scaling (on
/// UVs with V up, as `uv' = rotate * scale * uv + translate`) as a transform of the document's V-down UVs.
fn texture_tex(id: i64, objs: &Objects<'_>, image: usize) -> Tex {
    let mut t = Tex::new(image);
    let p = objs.props(id);
    let wrap = |k: &str| match scalar(&p, k) {
        Some(w) if w != 0.0 => material::CLAMP,
        _ => material::REPEAT,
    };
    t.wrap = [wrap("WrapModeU"), wrap("WrapModeV")];
    let [tu, tv, _] = vec3(&p, "Translation", [0.0; 3]).map(f32_of);
    let [_, _, angle] = vec3(&p, "Rotation", [0.0; 3]).map(f32_of);
    let [su, sv, _] = vec3(&p, "Scaling", [1.0; 3]).map(f32_of);
    let (su, sv) = (if su == 0.0 { 1.0 } else { su }, if sv == 0.0 { 1.0 } else { sv });
    let angle = angle.to_radians();
    if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
        let (sin, cos) = angle.sin_cos();
        t.transform = Some(UvTransform {
            offset: [tu - sv * sin, 1.0 - tv - sv * cos],
            rotation: angle,
            scale: [su, sv],
        });
    }
    t
}

/// A texture's pixels: embedded in a connected `Video`'s `Content`, or the file it names, found in the pack.
fn texture_image(id: i64, objs: &Objects<'_>, dir: &Path, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    let (node, _) = objs.nodes.get(&id)?;
    let names: Vec<String> = ["RelativeFilename", "FileName"]
        .iter()
        .filter_map(|k| child_attr(*node, k).and_then(string).map(str::to_owned))
        .filter(|s| !s.is_empty())
        .collect();
    for (vid, _) in objs.children.get(&id).cloned().unwrap_or_default() {
        if let Some((video, _)) = objs.nodes.get(&vid)
            && video.name() == "Video"
            && let Some(A::Binary(bytes)) = child_attr(*video, "Content")
            && !bytes.is_empty()
        {
            let hint = names.first().map_or_else(
                || Path::new("t.png").to_path_buf(),
                |n| Path::new(&n.replace('\\', "/")).to_path_buf(),
            );
            let name = hint.to_string_lossy().into_owned();
            let convert = |b: &[u8]| match objs.file.as_ref().and_then(|f| shared_part(f, b)) {
                Some(part) => assets.image_shared(part, &hint),
                None => assets.image(b, &hint),
            };
            if let Some(i) = images.bytes(Cow::Borrowed(&bytes[..]), &name, Location::Embedded, convert) {
                return Some(i);
            }
        }
    }
    let (name, path) = names.iter().find_map(|n| Some((n, assets.find(n, dir)?)))?;
    images.file(assets, path, name)
}

/// `part` of `file`, sharing its bytes, when it lies inside them and is at least a quarter of them: a small texture
/// is copied instead, so that it doesn't keep a big file's bytes alive.
fn shared_part(file: &crate::Bytes, part: &[u8]) -> Option<crate::Bytes> {
    let (start, at) = (file.as_ptr().addr(), part.as_ptr().addr());
    let inside = at >= start && at - start + part.len() <= file.len();
    (inside && part.len() >= file.len() / 4).then(|| file.slice_ref(part))
}

/// Vertices already emitted at a crowded control point, by primitive, control point and what else they're matched on.
type Seen = FastMap<(usize, usize, [usize; 5 + EXTRA_SETS]), u32>;

/// The most vertices chained at one control point before the rest are looked up by hash instead.
const MAX_CHAIN: usize = 32;

/// A material slot's primitive, each vertex's control point, and how many corners and triangles its polygons have.
struct SlotPrim {
    slot: usize,
    prim: Prim,
    cps: Vec<usize>,
    corners: usize,
    triangles: usize,
}

/// How many UV and color sets past the first of each a geometry's vertices are matched on; one with more gives
/// each polygon corner its own vertex instead.
const EXTRA_SETS: usize = 3;

/// The most UV or color sets read from one geometry.
const MAX_SETS: usize = 64;

/// A primitive per material slot, each vertex's control point, and the UV layer its `uvs` hold.
type Prims = Vec<(usize, Prim, Vec<usize>, usize)>;

/// A geometry's triangles in its own space, a primitive per material slot (sorted), each vertex's control point
/// alongside so skins and blend shapes can find it; more than `limit` triangles is refused. A slot whose color
/// texture reads a UV layer by name (`color_uv`) takes that layer as its texture coordinates, and the first layer
/// in its place among the other sets. With `named`, the UV layers' names come back too, by number.
fn geometry(
    geom: NodeHandle<'_>,
    limit: usize,
    color_uv: &HashMap<usize, &str>,
    named: bool,
) -> Result<(Prims, Vec<String>)> {
    let verts = floats(child_attr(geom, "Vertices").context(ErrorKind::Corrupt, "mesh has no vertices")?)
        .context(ErrorKind::Corrupt, "bad vertices")?;
    let poly_index = ints(child_attr(geom, "PolygonVertexIndex").context(ErrorKind::Corrupt, "mesh has no polygons")?)
        .context(ErrorKind::Corrupt, "bad polygons")?;
    let normals = Layer::read(geom, "LayerElementNormal", "Normals", "NormalsIndex", 3);
    // Tangents need the normals they're perpendicular to; the binormals only give their handedness.
    let tangents = normals
        .as_ref()
        .and_then(|_| Layer::read(geom, "LayerElementTangent", "Tangents", "TangentsIndex", 3));
    let binormals = tangents
        .as_ref()
        .and_then(|_| Layer::read(geom, "LayerElementBinormal", "Binormals", "BinormalsIndex", 3));
    let uv_all = Layer::read_all(geom, "LayerElementUV", "UV", "UVIndex", 2);
    // The layers' names, in the same order.
    let uv_names: Vec<String> = if named {
        uv_all
            .iter()
            .take(MAX_SETS)
            .map(|(el, _)| {
                let name = child_attr(*el, "Name").and_then(string).unwrap_or("");
                name.split('\0').next().unwrap_or("").to_owned()
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut uv_layers: Vec<Layer> = uv_all.into_iter().map(|(_, l)| l).collect();
    let mut color_layers: Vec<Layer> = Layer::read_all(geom, "LayerElementColor", "Colors", "ColorIndex", 4)
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    uv_layers.truncate(MAX_SETS);
    color_layers.truncate(MAX_SETS);
    let mut uv_layers = uv_layers.into_iter();
    let mut color_layers = color_layers.into_iter();
    let (uvs, colors) = (uv_layers.next(), color_layers.next());
    let (uv_more, color_more): (Vec<Layer>, Vec<Layer>) = (uv_layers.collect(), color_layers.collect());
    // A polygon naming a control point that isn't there or whose position isn't a number, too few to make a
    // triangle, or a hole, is left out.
    let holes = holes(geom);
    let usable = |poly: usize, range: &std::ops::Range<usize>| {
        range.len() >= 3
            && holes
                .as_ref()
                .is_none_or(|h| h.get(if h.len() == 1 { 0 } else { poly }) != Some(&true))
            && range.clone().all(|pv| {
                let cp = corner_cp(poly_index[pv]);
                cp < verts.len() / 3 && verts[cp * 3..cp * 3 + 3].iter().all(|c| c.is_finite())
            })
    };
    // Without normals of its own, a geometry with smoothing has them made from it.
    let smooth = if normals.is_none() {
        smoothing(geom, &verts, &poly_index, &usable)
    } else {
        None
    };
    let mut tris = Vec::new();
    let mut corner_pos = Vec::new();
    let mat_layer = geom.children_by_name("LayerElementMaterial").next().and_then(|el| {
        Some(Layer {
            values: ints(child_attr(el, "Materials")?)?
                .into_iter()
                .map(|i| {
                    #[allow(clippy::cast_precision_loss, reason = "a material slot, far below 2^53")]
                    let i = i as f64;
                    i
                })
                .collect(),
            index: None,
            mapping: Mapping::of(el, Mapping::AllSame),
            width: 1,
        })
    });
    // A layer holding a value per polygon vertex, unindexed, gives every corner its own vertex: no need to look for
    // one already made.
    let unique = [&normals, &tangents, &binormals, &uvs, &colors]
        .into_iter()
        .flatten()
        .chain(uv_more.iter().chain(&color_more))
        .any(|l| matches!(l.mapping, Mapping::PolygonVertex) && l.index.is_none())
        || uv_more.len() + color_more.len() > EXTRA_SETS;
    let slot_of = |pv: usize, poly: usize| {
        mat_layer.as_ref().and_then(|l| l.get(pv, 0, poly)).map_or(0, |(_, v)| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a material slot; `as` saturates bad ones and `get` refuses them"
            )]
            let slot = v[0] as usize;
            slot
        })
    };
    // Each polygon's primitive (one per material slot), or none when it's left out, found first so each
    // primitive's buffers are sized once.
    let mut slots: Vec<SlotPrim> = Vec::new();
    let mut slot_index: FastMap<usize, usize> = FastMap::default();
    let mut last: Option<(usize, usize)> = None;
    let mut poly_slot = Vec::new();
    let mut total = 0;
    for (poly, range) in polygons(&poly_index) {
        if !usable(poly, &range) {
            poly_slot.push(usize::MAX);
            continue;
        }
        let slot = slot_of(range.end - 1, poly);
        let s = match last {
            Some((n, s)) if n == slot => s,
            _ => {
                let s = *slot_index.entry(slot).or_insert_with(|| {
                    slots.push(SlotPrim {
                        slot,
                        prim: Prim {
                            material: slot,
                            ..Prim::default()
                        },
                        cps: Vec::new(),
                        corners: 0,
                        triangles: 0,
                    });
                    slots.len() - 1
                });
                last = Some((slot, s));
                s
            }
        };
        slots[s].corners += range.len();
        slots[s].triangles += range.len() - 2;
        total += range.len();
        poly_slot.push(s);
    }
    let cp_count = verts.len() / 3;
    for s in &mut slots {
        // Without de-duplication every corner is a vertex; with it, about its share of the control points.
        let n = if unique {
            s.corners
        } else {
            (s.corners / (total / cp_count.max(1)).max(1)).min(s.corners)
        };
        let p = &mut s.prim;
        p.positions.reserve_exact(n);
        s.cps.reserve_exact(n);
        if normals.is_some() || smooth.is_some() {
            p.normals.reserve_exact(n);
        }
        if tangents.is_some() {
            p.tangents.reserve_exact(n);
        }
        if uvs.is_some() {
            p.uvs.reserve_exact(n);
        }
        if colors.is_some() {
            p.colors.reserve_exact(n);
        }
        p.indices.reserve_exact(s.triangles * 3);
    }
    // Besides its primitive and control point, a vertex is matched on the index of each layer whose value can
    // differ between the corners at one control point: not one by control point, or the same for all.
    let varies = |l: &Layer| matches!(l.mapping, Mapping::Polygon | Mapping::PolygonVertex);
    let main_varies = [
        normals.as_ref().map_or(smooth.is_some(), varies),
        tangents.as_ref().is_some_and(varies),
        binormals.as_ref().is_some_and(varies),
        uvs.as_ref().is_some_and(varies),
        colors.as_ref().is_some_and(varies),
    ];
    let extra_varies: Vec<&Layer> = uv_more.iter().chain(&color_more).filter(|l| varies(l)).collect();
    let stride = main_varies.iter().filter(|v| **v).count() + extra_varies.len();
    // Vertices already made, chained by control point (the next at the same one, its primitive and its number),
    // with what each was matched on in `keys`; past `MAX_CHAIN` at one control point, in `crowded`.
    let mut head = if unique { Vec::new() } else { vec![usize::MAX; cp_count] };
    let mut chain: Vec<(usize, usize, u32)> = Vec::new();
    let mut keys: Vec<usize> = Vec::new();
    let mut crowded = Seen::default();
    if !unique {
        chain.reserve(cp_count.min(total));
        keys.reserve(cp_count.min(total) * stride);
    }
    let mut corners: Vec<(usize, usize)> = Vec::new();
    let mut ids = Vec::new();
    for (poly, range) in polygons(&poly_index) {
        let s = poly_slot[poly];
        if s == usize::MAX {
            continue;
        }
        let SlotPrim { prim, cps, .. } = &mut slots[s];
        corners.clear();
        corners.extend(range.map(|pv| (pv, corner_cp(poly_index[pv]))));
        ids.clear();
        corner_pos.clear();
        for &(pv, cp) in &corners {
            let pos = &verts[cp * 3..cp * 3 + 3];
            let p = [pos[0], pos[1], pos[2]].map(f32_of);
            corner_pos.push(p);
            let (n, tg, bn, t, c) = [&normals, &tangents, &binormals, &uvs, &colors]
                .map(|l| l.as_ref().and_then(|l| l.get(pv, cp, poly)))
                .into();
            let n = n.or_else(|| {
                smooth
                    .as_ref()
                    .map(|(group, normal)| (group[pv], normal[group[pv]].as_slice()))
            });
            let i = |x: Option<(usize, &[f64])>| x.map_or(usize::MAX, |x| x.0);
            let mut new = || {
                prim.positions.push(p);
                cps.push(cp);
                if let Some((_, nv)) = n {
                    let nv = vec3::unit_or([nv[0], nv[1], nv[2]], [0.0, 1.0, 0.0]);
                    prim.normals.push(nv.map(f32_of));
                    if let Some((_, tv)) = tg {
                        prim.tangents.push(tangent(nv, [tv[0], tv[1], tv[2]], bn.map(|b| b.1)));
                    }
                }
                if let Some((_, tv)) = t {
                    prim.uvs.push(uv(tv));
                }
                if let Some((_, cv)) = c {
                    prim.colors.push(color(cv));
                }
                let k = prim.positions.len() - 1;
                for (s, l) in uv_more.iter().enumerate() {
                    if let Some((_, tv)) = l.get(pv, cp, poly) {
                        set_at(&mut prim.uv_sets, s + 1, k, uv(tv));
                    }
                }
                for (s, l) in color_more.iter().enumerate() {
                    if let Some((_, cv)) = l.get(pv, cp, poly) {
                        set_at(&mut prim.color_sets, s + 1, k, color(cv));
                    }
                }
                u32::try_from(k).unwrap_or(u32::MAX)
            };
            let id = if unique {
                new()
            } else {
                let mut key = [usize::MAX; 5 + EXTRA_SETS];
                let mut kn = 0;
                for (x, v) in [n, tg, bn, t, c].into_iter().zip(main_varies) {
                    if v {
                        key[kn] = i(x);
                        kn += 1;
                    }
                }
                for l in &extra_varies {
                    key[kn] = i(l.get(pv, cp, poly));
                    kn += 1;
                }
                let mut e = head[cp];
                let mut steps = 0;
                loop {
                    if e == usize::MAX {
                        if steps < MAX_CHAIN {
                            let id = new();
                            chain.push((head[cp], s, id));
                            keys.extend_from_slice(&key[..stride]);
                            head[cp] = chain.len() - 1;
                            break id;
                        }
                        break *crowded.entry((s, cp, key)).or_insert_with(new);
                    }
                    let (next, es, id) = chain[e];
                    if es == s && keys[e * stride..e * stride + stride] == key[..stride] {
                        break id;
                    }
                    e = next;
                    steps += 1;
                }
            };
            ids.push(id);
        }
        tris.clear();
        triangulate(&corner_pos, &mut tris);
        for &[a, b, c] in &tris {
            prim.indices.extend([a, b, c].map(|i| ids[i as usize]));
        }
    }
    let mut out: Prims = slots
        .into_iter()
        .filter(|s| !s.prim.indices.is_empty())
        .map(|s| {
            let (slot, mut prim, cps) = (s.slot, s.prim, s.cps);
            let n = prim.positions.len();
            if prim.normals.len() != n {
                prim.normals = smooth_normals(&prim.positions, &prim.indices);
                // Made up here, they wouldn't match the file's tangents.
                prim.tangents.clear();
            }
            if prim.tangents.len() != n {
                prim.tangents.clear();
            }
            if prim.uvs.len() != n {
                prim.uvs.clear();
            }
            if prim.colors.len() != n {
                prim.colors.clear();
            }
            // A set some corner had no value for is dropped, and so are trailing empty ones.
            for set in &mut prim.uv_sets {
                if set.len() != n {
                    set.clear();
                }
            }
            for set in &mut prim.color_sets {
                if set.len() != n {
                    set.clear();
                }
            }
            // The layer the color texture reads becomes the texture coordinates; the first takes its number.
            let mut color_set = 0;
            if let Some(c) = color_uv
                .get(&slot)
                .and_then(|name| uv_names.iter().position(|n| n == name))
                .filter(|&c| c > 0 && prim.uv_sets.get(c).is_some_and(|s| s.len() == n))
            {
                let first = std::mem::take(&mut prim.uvs);
                prim.uvs = std::mem::replace(&mut prim.uv_sets[c], first);
                color_set = c;
            }
            while prim.uv_sets.last().is_some_and(Vec::is_empty) {
                prim.uv_sets.pop();
            }
            while prim.color_sets.last().is_some_and(Vec::is_empty) {
                prim.color_sets.pop();
            }
            (slot, prim, cps, color_set)
        })
        .collect();
    out.sort_unstable_by_key(|x| x.0);
    let triangles: usize = out.iter().map(|x| x.1.indices.len() / 3).sum();
    ensure!(
        triangles <= limit,
        LimitExceeded,
        "too many triangles to convert (more than {limit})"
    );
    Ok((out, uv_names))
}

/// The control point a `PolygonVertexIndex` entry names (a polygon's last is stored as `!index`).
fn corner_cp(v: i64) -> usize {
    usize::try_from(if v < 0 { !v } else { v }).unwrap_or(usize::MAX)
}

/// Each polygon of a `PolygonVertexIndex`: its number, and its polygon vertices.
fn polygons(poly_index: &[i64]) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
    let mut start = 0;
    poly_index
        .iter()
        .enumerate()
        .filter(|(_, v)| **v < 0)
        .enumerate()
        .map(move |(poly, (k, _))| {
            let range = start..k + 1;
            start = k + 1;
            (poly, range)
        })
}

/// Which polygons are holes, by polygon (or one for all).
fn holes(geom: NodeHandle<'_>) -> Option<Vec<bool>> {
    let el = geom.children_by_name("LayerElementHole").next()?;
    match child_attr(el, "Hole")? {
        A::ArrBool(b) => b.map(|x| x & 1 != 0),
        v => ints(v)
            .map(|h| h.iter().map(|&x| x != 0).collect())
            .or_else(|| floats(v).map(|h| h.iter().map(|&x| x != 0.0).collect())),
    }
}

/// Normals for a geometry without its own, from its smoothing: per polygon vertex, a group, and per group, the
/// sum of its polygons' normals, each as long as its polygon is large. By polygon, the corners at a control point
/// whose polygons are in the same smoothing group share a normal (group 0 is flat; groups are matched whole, not
/// bit by bit); by edge, the corners of polygons meeting at an edge that isn't hard do. `None` without smoothing.
fn smoothing(
    geom: NodeHandle<'_>,
    verts: &[f64],
    poly_index: &[i64],
    usable: &dyn Fn(usize, &std::ops::Range<usize>) -> bool,
) -> Option<(Vec<usize>, Vec<[f64; 3]>)> {
    let el = geom.children_by_name("LayerElementSmoothing").next()?;
    let values = child_attr(el, "Smoothing").and_then(ints)?;
    let mapping = child_attr(el, "MappingInformationType").and_then(string);
    let n = poly_index.len();
    let cp = |pv: usize| corner_cp(poly_index[pv]);
    // The polygon vertex after each, round its polygon.
    let mut next = vec![usize::MAX; n];
    for (_, range) in polygons(poly_index) {
        for pv in range.clone() {
            next[pv] = if pv + 1 == range.end { range.start } else { pv + 1 };
        }
    }
    let mut group = vec![usize::MAX; n];
    if mapping == Some("ByEdge") {
        // An edge is its two control points, lowest first; `Edges` names each by the polygon vertex it starts at.
        let edge = |pv: usize| {
            let (a, b) = (cp(pv), cp(next[pv]));
            (a.min(b), a.max(b))
        };
        let edges = child_attr(geom, "Edges").and_then(ints).unwrap_or_default();
        let mut hard: Vec<(usize, usize)> = edges
            .iter()
            .zip(&values)
            .filter(|(_, smooth)| **smooth == 0)
            .filter_map(|(&e, _)| usize::try_from(e).ok().filter(|&pv| pv < n && next[pv] != usize::MAX))
            .map(edge)
            .collect();
        hard.sort_unstable();
        // Every polygon's side: its edge, and its corners at the edge's first and second control point.
        let mut sides: Vec<((usize, usize), usize, usize)> = Vec::new();
        for (poly, range) in polygons(poly_index) {
            if !usable(poly, &range) {
                continue;
            }
            for pv in range {
                let (a, b) = (cp(pv), cp(next[pv]));
                if a != b {
                    let (lo, hi) = if a < b { (pv, next[pv]) } else { (next[pv], pv) };
                    sides.push((edge(pv), lo, hi));
                }
            }
        }
        sides.sort_unstable();
        let mut parent: Vec<usize> = (0..n).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };
        for run in sides.chunk_by(|a, b| a.0 == b.0) {
            if hard.binary_search(&run[0].0).is_ok() {
                continue;
            }
            for w in run.windows(2) {
                for (x, y) in [(w[0].1, w[1].1), (w[0].2, w[1].2)] {
                    let (x, y) = (find(&mut parent, x), find(&mut parent, y));
                    parent[x] = y;
                }
            }
        }
        for (poly, range) in polygons(poly_index) {
            if usable(poly, &range) {
                for pv in range {
                    group[pv] = find(&mut parent, pv);
                }
            }
        }
    } else {
        let mut ids: FastMap<(usize, i64), usize> = FastMap::default();
        for (poly, range) in polygons(poly_index) {
            if !usable(poly, &range) {
                continue;
            }
            let at = if mapping == Some("AllSame") { 0 } else { poly };
            let mask = values.get(at).copied().unwrap_or(0);
            // Group 0 is flat: each polygon its own.
            let key = if mask == 0 {
                -1 - i64::try_from(poly).unwrap_or(i64::MAX - 1)
            } else {
                mask
            };
            for pv in range {
                let len = ids.len();
                group[pv] = *ids.entry((cp(pv), key)).or_insert(len);
            }
        }
    }
    let mut normal = vec![[0.0; 3]; n];
    let pos = |pv: usize| {
        let c = cp(pv);
        [verts[c * 3], verts[c * 3 + 1], verts[c * 3 + 2]]
    };
    for (poly, range) in polygons(poly_index) {
        if !usable(poly, &range) {
            continue;
        }
        // Newell's method: twice the polygon's area, along its normal.
        let mut sum = [0.0; 3];
        for pv in range.clone() {
            vec3::newell_edge(&mut sum, pos(pv), pos(next[pv]));
        }
        for pv in range {
            let g = &mut normal[group[pv]];
            for k in 0..3 {
                g[k] += sum[k];
            }
        }
    }
    Some((group, normal))
}

/// An FBX texture coordinate the glTF way: FBX's texture origin is bottom-left, glTF's top-left.
fn uv(v: &[f64]) -> [f32; 2] {
    [f32_of(v[0]), 1.0 - f32_of(v[1])]
}

/// An FBX vertex color, sRGB, in linear light.
fn color(v: &[f64]) -> [f32; 4] {
    let [r, g, b, a] = [v[0], v[1], v[2], v[3]].map(|v| f32_of(v.clamp(0.0, 1.0)));
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
}

/// Sets vertex `k` of set `s` of `sets` to `v`, when every vertex before it has a value in that set: a set with a
/// gap can't line up with the vertices, and is dropped at the end.
fn set_at<T: Copy>(sets: &mut Vec<Vec<T>>, s: usize, k: usize, v: T) {
    if sets.len() <= s {
        sets.resize_with(s + 1, Vec::new);
    }
    if sets[s].len() == k {
        sets[s].push(v);
    }
}

/// A glTF tangent from FBX's unit normal `n`, tangent `t` and, if there is one, binormal `b`: `t` made
/// perpendicular to `n` and unit length, with `w` the side `b` is on (glTF's bitangent is `cross(n, t) * w`).
/// glTF's V runs down where FBX's runs up, but both keep the bitangent pointing up the image, so it doesn't flip.
fn tangent(n: [f64; 3], t: [f64; 3], b: Option<&[f64]>) -> [f32; 4] {
    let d = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
    let t = [t[0] - n[0] * d, t[1] - n[1] * d, t[2] - n[2] * d];
    let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t = if len > 0.0 && len.is_finite() {
        t.map(|c| c / len)
    } else {
        [1.0, 0.0, 0.0]
    };
    let w = b.map_or(1.0, |b| {
        let c = [
            n[1] * t[2] - n[2] * t[1],
            n[2] * t[0] - n[0] * t[2],
            n[0] * t[1] - n[1] * t[0],
        ];
        if c[0] * b[0] + c[1] * b[1] + c[2] * b[2] < 0.0 {
            -1.0
        } else {
            1.0
        }
    });
    [f32_of(t[0]), f32_of(t[1]), f32_of(t[2]), w]
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::document::Pose;
    use crate::mesh::Mesh;
    use fbxcel::writer::v7400::binary::{FbxFooter, Writer};

    /// The file at `p`, with assets under `dir`, flattened in its rest pose.
    fn flat(p: &Path, dir: &Path) -> Mesh {
        let doc = load(
            crate::formats::Input::file(p),
            &crate::assets::Folder::new(dir),
            Keep::ALL,
        )
        .unwrap();
        doc.flatten(Pose::Rest).unwrap().unwrap()
    }

    fn png(rgb: [u8; 3]) -> Vec<u8> {
        let mut b = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb(rgb))
            .write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png)
            .unwrap();
        b
    }

    /// A Z-up, centimeter FBX (3ds Max style): one textured quad on the ground, its model raised 100 cm.
    pub(crate) fn z_up_fbx() -> Vec<u8> {
        let mut w = Writer::new(std::io::Cursor::new(Vec::new()), fbxcel::low::FbxVersion::V7_4).unwrap();
        let p = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, name: &str, vals: &[A]| {
            let mut a = w.new_node("P").unwrap();
            for s in [name, name, "", "A"] {
                a.append_string_direct(s).unwrap();
            }
            for v in vals {
                match v {
                    A::F64(f) => a.append_f64(*f).unwrap(),
                    A::I32(i) => a.append_i32(*i).unwrap(),
                    _ => unreachable!(),
                }
            }
            w.close_node().unwrap();
        };
        let object = |w: &mut Writer<std::io::Cursor<Vec<u8>>>, kind: &str, id: i64, name: &str, sub: &str| {
            let mut a = w.new_node(kind).unwrap();
            a.append_i64(id).unwrap();
            a.append_string_direct(&format!("{name}\0\x01{kind}")).unwrap();
            a.append_string_direct(sub).unwrap();
        };
        // Global settings: Z up, -Y front, X right, centimeters.
        w.new_node("GlobalSettings").unwrap();
        w.new_node("Properties70").unwrap();
        for (k, v) in [
            ("UpAxis", 2),
            ("UpAxisSign", 1),
            ("FrontAxis", 1),
            ("FrontAxisSign", -1),
            ("CoordAxis", 0),
            ("CoordAxisSign", 1),
        ] {
            p(&mut w, k, &[A::I32(v)]);
        }
        p(&mut w, "UnitScaleFactor", &[A::F64(1.0)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        w.new_node("Objects").unwrap();
        object(&mut w, "Geometry", 10, "Quad", "Mesh");
        {
            let mut a = w.new_node("Vertices").unwrap();
            a.append_arr_f64_from_iter(None, [0., 0., 0., 100., 0., 0., 100., 100., 0., 0., 100., 0.])
                .unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("PolygonVertexIndex").unwrap();
            a.append_arr_i32_from_iter(None, [0, 1, 2, !3]).unwrap();
        }
        w.close_node().unwrap();
        {
            let mut a = w.new_node("LayerElementUV").unwrap();
            a.append_i32(0).unwrap();
        }
        for (name, s) in [
            ("MappingInformationType", "ByPolygonVertex"),
            ("ReferenceInformationType", "IndexToDirect"),
        ] {
            w.new_node(name).unwrap().append_string_direct(s).unwrap();
            w.close_node().unwrap();
        }
        w.new_node("UV")
            .unwrap()
            .append_arr_f64_from_iter(None, [0., 0., 1., 0., 1., 1., 0., 1.])
            .unwrap();
        w.close_node().unwrap();
        w.new_node("UVIndex")
            .unwrap()
            .append_arr_i32_from_iter(None, [0, 1, 2, 3])
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap(); // LayerElementUV
        w.close_node().unwrap(); // Geometry

        object(&mut w, "Model", 20, "Floor", "Mesh");
        w.new_node("Properties70").unwrap();
        p(&mut w, "Lcl Translation", &[A::F64(0.0), A::F64(0.0), A::F64(100.0)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Material", 30, "Stone", "");
        w.new_node("Properties70").unwrap();
        p(&mut w, "DiffuseColor", &[A::F64(0.5), A::F64(0.4), A::F64(0.3)]);
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Texture", 40, "StoneTex", "");
        w.new_node("RelativeFilename")
            .unwrap()
            .append_string_direct("C:\\art\\stone.png")
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap();

        object(&mut w, "Video", 50, "StoneVideo", "Clip");
        w.new_node("Content")
            .unwrap()
            .append_binary_direct(&png([10, 200, 30]))
            .unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap();
        w.close_node().unwrap(); // Objects

        w.new_node("Connections").unwrap();
        for (kind, child, parent, prop) in [
            ("OO", 10, 20, ""),
            ("OO", 20, 0, ""),
            ("OO", 30, 20, ""),
            ("OP", 40, 30, "DiffuseColor"),
            ("OO", 50, 40, ""),
        ] {
            let mut a = w.new_node("C").unwrap();
            a.append_string_direct(kind).unwrap();
            a.append_i64(child).unwrap();
            a.append_i64(parent).unwrap();
            if !prop.is_empty() {
                a.append_string_direct(prop).unwrap();
            }
            w.close_node().unwrap();
        }
        w.close_node().unwrap();
        w.finalize_and_flush(&FbxFooter::default()).unwrap().into_inner()
    }

    #[test]
    fn z_up_centimeters_become_y_up_meters_with_embedded_texture() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("floor.fbx");
        std::fs::write(&p, z_up_fbx()).unwrap();
        let mesh = flat(&p, dir.path());
        assert_eq!(mesh.triangles(), 2);
        let prim = &mesh.prims[0];
        for v in &prim.positions {
            assert!((v[1] - 1.0).abs() < 1e-5, "raised 100 cm = 1 m up: {v:?}");
        }
        let xs: Vec<f32> = prim.positions.iter().map(|v| v[0]).collect();
        let zs: Vec<f32> = prim.positions.iter().map(|v| v[2]).collect();
        assert!(
            xs.iter().any(|&x| (x - 1.0).abs() < 1e-5) && zs.iter().any(|&z| (z + 1.0).abs() < 1e-5),
            "{:?}",
            prim.positions
        );
        // Ground-facing quad in Z-up faces +Y once converted.
        assert!(prim.normals.iter().all(|n| n[1] > 0.99), "{:?}", prim.normals);
        assert_eq!(prim.uvs.len(), prim.positions.len());
        assert!(
            prim.uvs.iter().any(|t| (t[1] - 1.0).abs() < 1e-6),
            "V flipped to glTF's origin"
        );
        let mat = &mesh.materials[prim.material];
        assert_eq!(mat.name, "Stone");
        assert_eq!(mat.color_image, Some(0));
        assert_eq!(mesh.images.len(), 1);
        let glb = mesh.to_glb().unwrap();
        let doc: serde_json::Value = {
            let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
            serde_json::from_slice(&glb[20..20 + len]).unwrap()
        };
        assert_eq!(
            doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
            0
        );
        assert!(doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"].is_number());
    }

    /// `z_up_fbx` as ASCII FBX, the way Asset Forge writes it.
    fn z_up_ascii_fbx() -> String {
        use base64::Engine as _;
        let png = base64::engine::general_purpose::STANDARD.encode(png([10, 200, 30]));
        let (head, tail) = png.split_at(png.len() / 2);
        format!(
            r#"; FBX 7.3.0 project file
FBXHeaderExtension:  {{
	FBXHeaderVersion: 1003
	FBXVersion: 7300
}}
GlobalSettings:  {{
	Version: 1000
	Properties70:  {{
		P: "UpAxis", "int", "Integer", "",2
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",1
		P: "FrontAxisSign", "int", "Integer", "",-1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
	}}
}}
Objects:  {{
	Geometry: 10, "Geometry::Quad", "Mesh" {{
		Vertices: *12 {{
			a: 0,0,0,100,0,0,100,100,0,0,100,0
		}}
		PolygonVertexIndex: *4 {{
			a: 0,1,2,-4
		}}
		LayerElementUV: 0 {{
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *8 {{
				a: 0,0,1,0,1,1,0,1
			}}
			UVIndex: *4 {{
				a: 0,1,2,3
			}}
		}}
	}}
	Model: 20, "Model::Floor", "Mesh" {{
		Properties70:  {{
			P: "Lcl Translation", "Lcl Translation", "", "A",0,0,100
		}}
	}}
	Material: 30, "Material::Stone", "" {{
		Properties70:  {{
			P: "DiffuseColor", "Color", "", "A",0.5,0.4,0.3
		}}
	}}
	Texture: 40, "Texture::StoneTex", "" {{
		RelativeFilename: "C:\art\stone.png"
	}}
	Video: 50, "Video::StoneVideo", "Clip" {{
		Content: , "{head}",
			"{tail}"
	}}
}}
Connections:  {{
	;Geometry::Quad, Model::Floor
	C: "OO",10,20
	C: "OO",20,0
	C: "OO",30,20
	C: "OP",40,30, "DiffuseColor"
	C: "OO",50,40
}}
"#
        )
    }

    #[test]
    fn ascii_fbx_reads_like_binary() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("ascii.fbx"), dir.path().join("binary.fbx"));
        std::fs::write(&a, z_up_ascii_fbx()).unwrap();
        std::fs::write(&b, z_up_fbx()).unwrap();
        let (a, b) = (flat(&a, dir.path()), flat(&b, dir.path()));
        assert_eq!(a.triangles(), 2);
        assert_eq!(a.prims[0].positions, b.prims[0].positions);
        assert_eq!(a.prims[0].normals, b.prims[0].normals);
        assert_eq!(a.prims[0].uvs, b.prims[0].uvs);
        assert_eq!(a.materials[a.prims[0].material].name, "Stone");
        assert_eq!(a.materials[a.prims[0].material].color_image, Some(0));
        assert_eq!(a.images.len(), 1);
    }

    #[test]
    fn every_uv_and_color_layer_and_the_tangents_are_read() {
        let layers = r#"		LayerElementUV: 1 {
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "Direct"
			UV: *8 {
				a: 0,0,2,0,2,2,0,2
			}
		}
		LayerElementNormal: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Normals: *12 {
				a: 0,0,1,0,0,1,0,0,1,0,0,1
			}
		}
		LayerElementTangent: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Tangents: *12 {
				a: 1,0,0.5,1,0,0.5,1,0,0.5,1,0,0.5
			}
		}
		LayerElementBinormal: 0 {
			MappingInformationType: "ByControlPoint"
			ReferenceInformationType: "Direct"
			Binormals: *12 {
				a: 0,-1,0,0,-1,0,0,-1,0,0,-1,0
			}
		}
		LayerElementColor: 0 {
			MappingInformationType: "AllSame"
			ReferenceInformationType: "Direct"
			Colors: *4 {
				a: 1,0,0,1
			}
		}
		LayerElementColor: 1 {
			MappingInformationType: "AllSame"
			ReferenceInformationType: "Direct"
			Colors: *4 {
				a: 0,0,1,0.5
			}
		}
		LayerElementUV: 0 {"#;
        let text = z_up_ascii_fbx().replacen("\t\tLayerElementUV: 0 {", layers, 1);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("layers.fbx");
        std::fs::write(&p, text).unwrap();
        let mesh = flat(&p, dir.path());
        let prim = &mesh.prims[0];
        let n = prim.positions.len();
        assert_eq!(prim.uvs.len(), n);
        assert_eq!(prim.uv_sets.len(), 2, "slot 0 is `uvs`, slot 1 the second layer");
        assert!(prim.uv_sets[0].is_empty(), "slot 0 is `uvs`");
        assert_eq!(prim.uv_sets[1].len(), n);
        for (a, b) in prim.uvs.iter().zip(&prim.uv_sets[1]) {
            assert_eq!(
                [a[0] * 2.0, 1.0 - (1.0 - a[1]) * 2.0],
                *b,
                "V flipped, like the first set"
            );
        }
        assert_eq!(prim.colors, vec![[1.0, 0.0, 0.0, 1.0]; n]);
        assert_eq!(prim.color_sets.len(), 2);
        assert!(prim.color_sets[0].is_empty(), "slot 0 is `colors`");
        assert_eq!(prim.color_sets[1], vec![[0.0, 0.0, 1.0, 0.5]; n]);
        // The tangent is made perpendicular to the normal (+Z in the file, +Y here), and the
        // binormal on the far side of cross(normal, tangent) makes it left-handed.
        assert_eq!(prim.tangents.len(), n);
        for (t, nrm) in prim.tangents.iter().zip(&prim.normals) {
            assert!(
                (t[0] - 1.0).abs() < 1e-6 && t[1].abs() < 1e-6 && t[2].abs() < 1e-6,
                "{t:?}"
            );
            assert_eq!(t[3], -1.0);
            assert!(nrm[1] > 0.99);
        }
        // Written out, every set survives, numbered from 0 without gaps.
        let glb = mesh.to_glb().unwrap();
        let memory = crate::assets::Memory::new();
        let doc = crate::formats::load_bytes(&glb, "x.glb", "glb", &memory, Keep::ALL).unwrap();
        let again = doc.flatten(Pose::Rest).unwrap().unwrap();
        let back = &again.prims[0];
        assert_eq!(back.uvs, prim.uvs);
        assert_eq!(back.uv_sets[1], prim.uv_sets[1]);
        assert_eq!(back.color_sets[1], prim.color_sets[1]);
        assert_eq!(back.tangents, prim.tangents);
    }

    #[test]
    fn object_ids_past_2_pow_53_stay_distinct() {
        // As doubles these are one number, so the model and its material would share a key.
        let (model, material) = ("9007199254740992", "9007199254740993");
        let text = z_up_ascii_fbx()
            .replace("Model: 20,", &format!("Model: {model},"))
            .replace("Material: 30,", &format!("Material: {material},"))
            .replace("C: \"OO\",10,20", &format!("C: \"OO\",10,{model}"))
            .replace("C: \"OO\",20,0", &format!("C: \"OO\",{model},0"))
            .replace("C: \"OO\",30,20", &format!("C: \"OO\",{material},{model}"))
            .replace("C: \"OP\",40,30,", &format!("C: \"OP\",40,{material},"));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big_ids.fbx");
        std::fs::write(&p, text).unwrap();
        let mesh = flat(&p, dir.path());
        assert_eq!(mesh.triangles(), 2);
        assert!(
            mesh.prims[0].positions.iter().all(|v| (v[1] - 1.0).abs() < 1e-5),
            "placed by its model"
        );
        assert_eq!(mesh.materials[mesh.prims[0].material].name, "Stone");
    }

    fn curve(values: &[f64]) -> Curve {
        #[allow(clippy::cast_precision_loss, reason = "a few keys")]
        let times = (0..values.len()).map(|i| i as f64 / 30.0).collect();
        Curve::linear(times, values.to_vec())
    }

    #[test]
    fn a_turn_past_180_baked_as_its_other_side_goes_the_short_way() {
        // A sway through 180 (a model turned round), written as -179 after 179, and back.
        let mut c = curve(&[175.0, 177.0, 179.0, -179.0, -177.0, -179.0, 179.0, 177.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [175.0, 177.0, 179.0, 181.0, 183.0, 181.0, 179.0, 177.0]);
        // Past 180 twice over, on a curve that steps a degree at a time.
        let mut c = curve(&[0.0, 1.0, 2.0, 361.0, 362.0, 1.0, 0.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn a_limb_raised_past_straight_up_keeps_turning_the_short_way() {
        // An arm raised from 70° through straight up to 110° about Y (XYZ order), baked past 90 as the
        // other set of three: X and Z half a turn on, Y mirrored (Kenney's kicks are written so).
        let keys = [
            [10.0, 70.0, 20.0],
            [10.0, 85.0, 20.0],
            [-170.0, 80.0, -160.0],
            [-170.0, 70.0, -160.0],
        ];
        let mut a = Animated {
            curves: std::array::from_fn(|k| Some(curve(&keys.map(|v| v[k])))),
            fixed: [0.0; 3],
        };
        a.unwrap_rotation(0);
        let got: Vec<[f64; 3]> = (0..4)
            .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().unwrap().values[i]))
            .collect();
        let want = [
            [10.0, 70.0, 20.0],
            [10.0, 85.0, 20.0],
            [10.0, 100.0, 20.0],
            [10.0, 110.0, 20.0],
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((0..3).all(|k| (g[k] - w[k]).abs() < 1e-9), "{got:?}");
        }
    }

    #[test]
    fn a_whole_turn_keyed_as_one_still_spins() {
        // A coin keyed at 0, 360 and 720: each step is the curve's usual one.
        let mut c = curve(&[0.0, 360.0, 720.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 360.0, 720.0]);
        let mut c = curve(&[0.0, 360.0]);
        c.unwrap_degrees();
        assert_eq!(c.values, [0.0, 360.0]);
        let mut a = rotation(&[[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), [[0.0, 0.0, 0.0], [0.0, 360.0, 0.0], [0.0, 720.0, 0.0]]);
    }

    /// An animated rotation with a curve on every axis, one key per row (a thirtieth of a second apart).
    fn rotation(rows: &[[f64; 3]]) -> Animated {
        Animated {
            curves: std::array::from_fn(|k| Some(curve(&rows.iter().map(|v| v[k]).collect::<Vec<_>>()))),
            fixed: [0.0; 3],
        }
    }

    /// The angles at each key: an axis without a curve holds its fixed value.
    fn keys(a: &Animated) -> Vec<[f64; 3]> {
        let n = a.curves.iter().flatten().map(|c| c.values.len()).max().unwrap_or(0);
        (0..n)
            .map(|i| std::array::from_fn(|k| a.curves[k].as_ref().map_or(a.fixed[k], |c| c.values[i])))
            .collect()
    }

    fn same_rotation(a: [f64; 3], b: [f64; 3], order: i32) -> bool {
        let (m, n) = (rotate(a, order), rotate(b, order));
        (0..3).all(|r| (0..3).all(|c| (m[r][c] - n[r][c]).abs() < 1e-6))
    }

    fn near(got: &[[f64; 3]], want: &[[f64; 3]]) -> bool {
        got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (0..3).all(|k| (g[k] - w[k]).abs() < 1e-6))
    }

    /// An angle as an exporter writes it: in (-180, 180].
    fn baked(a: f64) -> f64 {
        let w = (a + 180.0).rem_euclid(360.0) - 180.0;
        if w <= -180.0 { w + 360.0 } else { w }
    }

    /// The other angles making the same rotation: the order's middle axis mirrored, the others half a turn on.
    fn flipped(v: [f64; 3], order: i32) -> [f64; 3] {
        let middle = middle_axis(order);
        std::array::from_fn(|k| if k == middle { 180.0 - v[k] } else { v[k] + 180.0 })
    }

    /// The axis applied second: XYZ and ZYX turn about Y in the middle, XZY and YZX about Z, YXZ and ZXY about X.
    fn middle_axis(order: i32) -> usize {
        match order {
            1 | 2 => 2,
            3 | 4 => 0,
            _ => 1,
        }
    }

    #[test]
    fn a_limb_raised_past_straight_up_turns_the_short_way_in_every_rotation_order() {
        for order in 0..=5 {
            let middle = middle_axis(order);
            // The middle axis swings from 60° through straight up to 120°, the others hold; past 90 the
            // exporter writes the other set of three.
            let truth: Vec<[f64; 3]> = [60.0, 70.0, 80.0, 95.0, 105.0, 120.0, 105.0, 85.0]
                .iter()
                .map(|&m| std::array::from_fn(|k| if k == middle { m } else { [10.0, -30.0, 20.0][k] }))
                .collect();
            let written: Vec<[f64; 3]> = truth
                .iter()
                .map(|v| {
                    if v[middle] > 90.0 {
                        flipped(*v, order).map(baked)
                    } else {
                        *v
                    }
                })
                .collect();
            for (t, w) in truth.iter().zip(&written) {
                assert!(
                    same_rotation(*t, *w, order),
                    "order {order}: the test's flip makes the same rotation"
                );
            }
            let mut a = rotation(&written);
            a.unwrap_rotation(order);
            assert!(near(&keys(&a), &truth), "order {order}: {:?}", keys(&a));
        }
    }

    #[test]
    fn a_steady_spin_baked_into_half_turns_either_side_keeps_turning_one_way() {
        for step in [30.0, -30.0, 45.0, -7.5] {
            let truth: Vec<[f64; 3]> = (0..40).map(|i| [0.0, 0.0, 15.0 + step * f64::from(i)]).collect();
            let written: Vec<[f64; 3]> = truth.iter().map(|v| v.map(baked)).collect();
            // The spin about Z alone, with no curves on X and Y.
            let mut a = Animated {
                curves: [
                    None,
                    None,
                    Some(curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>())),
                ],
                fixed: [0.0; 3],
            };
            a.unwrap_rotation(0);
            assert!(near(&keys(&a), &truth), "{step}° a key: {:?}", keys(&a));
            assert!(
                a.curves[0].is_none() && a.curves[1].is_none(),
                "axes that keep their angle get no curve"
            );
            // Each curve on its own turns the short way too.
            let mut c = curve(&written.iter().map(|v| v[2]).collect::<Vec<_>>());
            c.unwrap_degrees();
            assert!(
                c.values.iter().zip(&truth).all(|(g, w)| (g - w[2]).abs() < 1e-9),
                "{step}°: {:?}",
                c.values
            );
        }
    }

    #[test]
    fn an_axis_without_a_curve_gets_one_when_a_flip_turns_it() {
        // Y holds 80 with no curve of its own while X and Z, turning a little each frame, flip half a turn:
        // the rotation the keys make is Y going on to 100, so Y needs keys.
        let written = [
            [10.0, 80.0, 20.0],
            [12.0, 80.0, 22.0],
            [-166.0, 80.0, -156.0],
            [-164.0, 80.0, -154.0],
        ];
        let mut a = Animated {
            curves: [
                Some(curve(&written.map(|v| v[0]))),
                None,
                Some(curve(&written.map(|v| v[2]))),
            ],
            fixed: [0.0, 80.0, 0.0],
        };
        a.unwrap_rotation(0);
        let want = [
            [10.0, 80.0, 20.0],
            [12.0, 80.0, 22.0],
            [14.0, 100.0, 24.0],
            [16.0, 100.0, 26.0],
        ];
        assert!(near(&keys(&a), &want), "{:?}", keys(&a));
        let y = a.curves[1].as_ref().expect("Y has keys now");
        assert_eq!(y.times, a.curves[0].as_ref().unwrap().times);
    }

    #[test]
    fn a_cut_or_a_fast_spin_is_kept_as_keyed() {
        // A cut: a degree a frame, then a jump to somewhere else entirely. The nearest angles making that
        // rotation are the ones written.
        let cut = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [150.0, 0.0, 0.0],
            [151.0, 0.0, 0.0],
        ];
        let mut a = rotation(&cut);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), cut);
        // A propeller keyed every third of a turn: every step is its usual one.
        let spin: Vec<[f64; 3]> = (0..6).map(|i| [0.0, 120.0 * f64::from(i), 0.0]).collect();
        let mut a = rotation(&spin);
        a.unwrap_rotation(0);
        assert_eq!(keys(&a), spin);
        let mut c = curve(&spin.iter().map(|v| v[1]).collect::<Vec<_>>());
        c.unwrap_degrees();
        assert_eq!(c.values, spin.iter().map(|v| v[1]).collect::<Vec<_>>());
    }

    #[test]
    fn curves_keyed_at_different_times_each_turn_the_short_way() {
        let mut x = curve(&[30.0, 30.0]);
        x.times = vec![0.0, 0.5];
        let z = curve(&[150.0, 165.0, 180.0, -165.0, -150.0]);
        let mut a = Animated {
            curves: [Some(x), None, Some(z)],
            fixed: [0.0; 3],
        };
        a.unwrap_rotation(0);
        assert_eq!(a.curves[0].as_ref().unwrap().values, [30.0, 30.0]);
        assert_eq!(
            a.curves[2].as_ref().unwrap().values,
            [150.0, 165.0, 180.0, 195.0, 210.0]
        );
    }

    #[test]
    fn odd_rotation_curves_are_left_alone() {
        let cases: [&[f64]; 6] = [
            &[42.0],
            &[5.0, 5.0, 5.0],
            &[0.0, f64::NAN, 10.0],
            &[0.0, f64::INFINITY, 0.0],
            &[f64::NEG_INFINITY, f64::INFINITY],
            &[1e300, -1e300, 1e300],
        ];
        for values in cases {
            let mut c = curve(values);
            c.unwrap_degrees();
            assert_eq!(format!("{:?}", c.values), format!("{values:?}"));
            let rows: Vec<[f64; 3]> = values.iter().map(|&v| [v, 0.0, 0.0]).collect();
            for order in [0, 3, 6, -1, 99] {
                let mut a = rotation(&rows);
                a.unwrap_rotation(order);
                assert_eq!(format!("{:?}", keys(&a)), format!("{rows:?}"), "order {order}");
            }
        }
        let mut none = Animated {
            curves: [None, None, None],
            fixed: [1.0, 2.0, 3.0],
        };
        none.unwrap_rotation(0);
        assert!(none.curves.iter().all(Option::is_none));
    }

    #[test]
    fn turning_keys_the_short_way_never_moves_a_key() {
        // Whatever angles the filter picks, each key makes the rotation it was written as: random keys in any
        // order, some wild, some with axes missing or keyed at other times.
        let mut rng = crate::test_rng::Pcg32::new(11);
        let angle = |rng: &mut crate::test_rng::Pcg32| f64::from(rng.f32()) * 1440.0 - 720.0;
        for trial in 0..500 {
            let order = [0, 1, 2, 3, 4, 5, 6, 42][rng.below(8) as usize];
            let n = 1 + rng.below(24) as usize;
            let mut a = Animated {
                curves: [None, None, None],
                fixed: std::array::from_fn(|_| angle(&mut rng)),
            };
            let mut last = [0.0; 3];
            for (k, last) in last.iter_mut().enumerate() {
                if rng.below(4) == 0 {
                    continue;
                }
                let values = (0..n)
                    .map(|_| {
                        if rng.below(3) == 0 {
                            return angle(&mut rng);
                        }
                        *last += f64::from(rng.f32()) * 20.0 - 10.0;
                        if rng.below(5) == 0 { baked(*last) } else { *last }
                    })
                    .collect::<Vec<_>>();
                let mut c = curve(&values);
                if rng.below(6) == 0 {
                    c.times.iter_mut().for_each(|t| *t *= 2.0);
                }
                a.curves[k] = Some(c);
            }
            let before = keys(&a);
            let shared = a.curves.iter().flatten().all(|c| {
                let first = a.curves.iter().flatten().next().unwrap();
                c.times == first.times
            });
            a.unwrap_rotation(order);
            let after = keys(&a);
            if shared {
                for (i, (b, g)) in before.iter().zip(&after).enumerate() {
                    assert!(
                        same_rotation(*b, *g, order),
                        "trial {trial}, key {i}: {b:?} became {g:?}"
                    );
                }
            } else {
                for (b, g) in before.iter().flatten().zip(after.iter().flatten()) {
                    let turns = (g - b) / 360.0;
                    assert!((turns - turns.round()).abs() < 1e-9, "trial {trial}: {b} became {g}");
                }
            }
        }
    }

    #[test]
    fn a_baked_turn_comes_back_as_smooth_as_it_was() {
        // A limb wandering a few degrees a frame for four seconds, written as an exporter bakes it: each angle
        // in (-180, 180], and now and then as the other set of three. The filter gives back a path as smooth
        // as the one wandered, making the same rotation at every key.
        let mut rng = crate::test_rng::Pcg32::new(3);
        for trial in 0..200 {
            let order = rng.below(6).cast_signed();
            let middle = middle_axis(order);
            let mut v: [f64; 3] = std::array::from_fn(|_| f64::from(rng.f32()) * 360.0 - 180.0);
            v[middle] = f64::from(rng.f32()) * 120.0 - 60.0;
            let mut truth = Vec::new();
            for _ in 0..120 {
                truth.push(v);
                for a in &mut v {
                    *a += f64::from(rng.f32()) * 8.0 - 4.0;
                }
                v[middle] = v[middle].clamp(-60.0, 60.0);
            }
            let written: Vec<[f64; 3]> = truth
                .iter()
                .map(|t| if rng.below(10) == 0 { flipped(*t, order) } else { *t }.map(baked))
                .collect();
            let mut a = rotation(&written);
            a.unwrap_rotation(order);
            let got = keys(&a);
            for (i, w) in got.windows(2).enumerate() {
                let step = (0..3).map(|k| (w[1][k] - w[0][k]).abs()).fold(0.0, f64::max);
                assert!(
                    step <= 4.0 + 1e-6,
                    "trial {trial} (order {order}), key {}: {:?} to {:?}",
                    i + 1,
                    w[0],
                    w[1]
                );
            }
            for (t, g) in truth.iter().zip(&got) {
                assert!(same_rotation(*t, *g, order), "trial {trial}: {t:?} became {g:?}");
            }
        }
    }

    // --- Files built here as ASCII FBX, read from memory -------------------------------------------------------

    /// An ASCII FBX file, Y up in meters, with these objects and connections.
    fn ascii(objects: &str, connections: &str) -> String {
        format!(
            r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {{
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}}
GlobalSettings:  {{
	Version: 1000
	Properties70:  {{
		P: "UnitScaleFactor", "double", "Number", "",100
	}}
}}
Objects:  {{
{objects}
}}
Connections:  {{
{connections}
}}
"#
        )
    }

    /// A model with these `Properties70` lines.
    fn model(id: i64, name: &str, props: &str) -> String {
        format!("\tModel: {id}, \"Model::{name}\", \"Mesh\" {{\n\t\tProperties70:  {{\n{props}\n\t\t}}\n\t}}\n")
    }

    /// A property line.
    fn prop(name: &str, values: &str) -> String {
        format!("\t\t\tP: \"{name}\", \"{name}\", \"\", \"A\",{values}\n")
    }

    /// Geometry 10: a unit quad facing +Z, with `layers` inside.
    fn quad(layers: &str) -> String {
        format!(
            "\tGeometry: 10, \"Geometry::Quad\", \"Mesh\" {{\n\t\tVertices: *12 {{\n\t\t\ta: 0,0,0,1,0,0,1,1,0,0,1,0\n\t\t}}\n\t\tPolygonVertexIndex: *4 {{\n\t\t\ta: 0,1,2,-4\n\t\t}}\n{layers}\t}}\n"
        )
    }

    /// A video with an embedded PNG.
    fn video(id: i64) -> String {
        use base64::Engine as _;
        let png = base64::engine::general_purpose::STANDARD.encode(png([10, 200, 30]));
        format!("\tVideo: {id}, \"Video::V{id}\", \"Clip\" {{\n\t\tContent: , \"{png}\"\n\t}}\n")
    }

    /// Connection lines: `(child, parent)` object links, and `(child, parent, property)` property links.
    fn links(oo: &[(i64, i64)], op: &[(i64, i64, &str)]) -> String {
        let oo = oo.iter().map(|(c, p)| format!("\tC: \"OO\",{c},{p}\n"));
        let op = op
            .iter()
            .map(|(c, p, name)| format!("\tC: \"OP\",{c},{p}, \"{name}\"\n"));
        oo.chain(op).collect()
    }

    fn read(text: &str) -> Document {
        let memory = crate::assets::Memory::new();
        crate::formats::load_bytes(text.as_bytes(), "t.fbx", "fbx", &memory, Keep::ALL).unwrap()
    }

    fn index(v: &Value) -> usize {
        usize::try_from(v.as_u64().unwrap()).unwrap()
    }

    /// A float accessor's numbers.
    fn accessor(doc: &Document, i: &Value) -> Vec<f32> {
        let a = &doc.json["accessors"][index(i)];
        let view = &doc.json["bufferViews"][index(&a["bufferView"])];
        let bytes = doc.buffer(index(&view["buffer"])).unwrap();
        let offset = |v: &Value| usize::try_from(v.as_u64().unwrap_or(0)).unwrap();
        let at = offset(&view["byteOffset"]) + offset(&a["byteOffset"]);
        let width = match a["type"].as_str().unwrap() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            _ => 16,
        };
        bytes[at..][..index(&a["count"]) * width * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// The node named `name`.
    fn node<'d>(doc: &'d Document, name: &str) -> &'d Value {
        doc.json["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == name)
            .unwrap_or_else(|| panic!("no node {name}"))
    }

    fn vector(v: &Value, default: &[f64]) -> Vec<f64> {
        v.as_array()
            .map_or_else(|| default.to_vec(), |a| a.iter().map(|x| x.as_f64().unwrap()).collect())
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    fn visible(n: &Value) -> bool {
        n["extensions"]["KHR_node_visibility"]["visible"] != false
    }

    #[test]
    fn hidden_models_and_coarser_levels_of_detail_are_kept_hidden() {
        let objects = [
            quad(""),
            model(20, "Shown", ""),
            model(21, "Gone", &prop("Visibility", "0")),
            model(22, "Lods", ""),
            "\tNodeAttribute: 60, \"NodeAttribute::\", \"LodGroup\" {\n\t}\n".to_owned(),
            model(23, "Fine", ""),
            model(24, "Coarse", ""),
            model(25, "UnderGone", ""),
        ]
        .concat();
        let doc = read(&ascii(
            &objects,
            &links(
                &[
                    (10, 20),
                    (10, 23),
                    (10, 24),
                    (20, 0),
                    (21, 0),
                    (22, 0),
                    (60, 22),
                    (23, 22),
                    (24, 22),
                    (25, 21),
                ],
                &[],
            ),
        ));
        assert!(visible(node(&doc, "Shown")));
        assert!(!visible(node(&doc, "Gone")));
        assert!(visible(node(&doc, "UnderGone")), "hidden by its parent, not itself");
        assert!(visible(node(&doc, "Fine")), "the finest level shows");
        assert!(!visible(node(&doc, "Coarse")));
        assert!(
            doc.json["extensionsUsed"]
                .as_array()
                .unwrap()
                .contains(&json!("KHR_node_visibility"))
        );
    }

    #[test]
    fn rotation_active_off_ignores_pre_rotation_and_order() {
        let pre = prop("PreRotation", "90,0,0");
        let off = pre.clone() + &prop("RotationActive", "0");
        let objects = [model(20, "On", &pre), model(21, "Off", &off)].concat();
        let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0)], &[])));
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!(close(
            &vector(&node(&doc, "On")["rotation"], &[0.0, 0.0, 0.0, 1.0]),
            &[h, 0.0, 0.0, h]
        ));
        assert!(close(
            &vector(&node(&doc, "Off")["rotation"], &[0.0, 0.0, 0.0, 1.0]),
            &[0.0, 0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn a_child_that_ignores_its_parents_scale_is_not_scaled_by_it() {
        let objects = [
            model(20, "Parent", &prop("Lcl Scaling", "2,2,2")),
            model(
                21,
                "Child",
                &(prop("InheritType", "2") + &prop("Lcl Translation", "1,0,0")),
            ),
        ]
        .concat();
        let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 20)], &[])));
        let child = node(&doc, "Child");
        assert!(close(&vector(&child["scale"], &[1.0; 3]), &[0.5; 3]), "{child}");
        assert!(
            close(&vector(&child["translation"], &[0.0; 3]), &[1.0, 0.0, 0.0]),
            "{child}"
        );
    }

    #[test]
    fn properties_a_model_leaves_out_come_from_its_template() {
        let definitions = "Definitions:  {\n\tObjectType: \"Model\" {\n\t\tPropertyTemplate: \"FbxNode\" {\n\t\t\tProperties70:  {\n\t\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,5,0\n\t\t\t}\n\t\t}\n\t}\n}\nObjects:  {";
        let objects = [
            model(20, "Templated", ""),
            model(21, "Own", &prop("Lcl Translation", "1,0,0")),
        ]
        .concat();
        let text = ascii(&objects, &links(&[(20, 0), (21, 0)], &[])).replacen("Objects:  {", definitions, 1);
        let doc = read(&text);
        assert!(close(
            &vector(&node(&doc, "Templated")["translation"], &[0.0; 3]),
            &[0.0, 5.0, 0.0]
        ));
        assert!(close(
            &vector(&node(&doc, "Own")["translation"], &[0.0; 3]),
            &[1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn transforms_that_overflow_or_are_not_numbers_never_write_null() {
        let objects = [
            model(20, "Infinite", &prop("Lcl Translation", "1e400,0,0")),
            model(
                21,
                "Overflowing",
                &(prop("Lcl Scaling", "1e200,1e200,1e200") + &prop("ScalingPivot", "1e200,1e200,1e200")),
            ),
        ]
        .concat();
        let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0)], &[])));
        let text = serde_json::to_string(&doc.json).unwrap();
        assert!(!text.contains("null"), "{text}");
    }

    #[test]
    fn a_deep_chain_listed_leaf_first_loads_quickly() {
        const N: i64 = 20_000;
        use std::fmt::Write as _;
        let (mut objects, mut connections) = (String::new(), String::new());
        for i in 0..N {
            let parent = if i + 1 == N { 0 } else { 101 + i };
            writeln!(objects, "\tModel: {}, \"Model::M{i}\", \"Null\" {{\n\t}}", 100 + i).unwrap();
            writeln!(connections, "\tC: \"OO\",{},{parent}", 100 + i).unwrap();
        }
        let text = ascii(&objects, &connections);
        let start = std::time::Instant::now();
        let doc = read(&text);
        let took = start.elapsed();
        assert_eq!(doc.json["nodes"].as_array().unwrap().len(), 20_000);
        assert!(took < std::time::Duration::from_secs(10), "took {took:?}");
    }

    /// The flattened mesh's normals at the corners at `p`.
    fn normals_at(mesh: &Mesh, p: [f32; 3]) -> Vec<[f32; 3]> {
        let prim = &mesh.prims[0];
        prim.positions
            .iter()
            .zip(&prim.normals)
            .filter(|(q, _)| (0..3).all(|k| (q[k] - p[k]).abs() < 1e-5))
            .map(|(_, n)| *n)
            .collect()
    }

    /// A wall (facing +Z) and a floor (facing +Y) meeting along the edge from the origin to +X, with `layers`.
    fn fold(layers: &str) -> Mesh {
        let geometry = format!(
            "\tGeometry: 10, \"Geometry::Fold\", \"Mesh\" {{\n\t\tVertices: *18 {{\n\t\t\ta: 0,0,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,1\n\t\t}}\n\t\tPolygonVertexIndex: *8 {{\n\t\t\ta: 0,1,2,-4,1,0,5,-5\n\t\t}}\n\t\tEdges: *7 {{\n\t\t\ta: 0,1,2,3,5,6,7\n\t\t}}\n{layers}\t}}\n"
        );
        let text = ascii(&(geometry + &model(20, "Fold", "")), &links(&[(10, 20), (20, 0)], &[]));
        read(&text).flatten(Pose::Rest).unwrap().unwrap()
    }

    fn smoothing_layer(mapping: &str, values: &str) -> String {
        let n = values.split(',').count();
        format!(
            "\t\tLayerElementSmoothing: 0 {{\n\t\t\tMappingInformationType: \"{mapping}\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tSmoothing: *{n} {{\n\t\t\t\ta: {values}\n\t\t\t}}\n\t\t}}\n"
        )
    }

    fn all_near(normals: &[[f32; 3]], want: [f32; 3]) -> bool {
        !normals.is_empty() && normals.iter().all(|n| (0..3).all(|k| (n[k] - want[k]).abs() < 1e-4))
    }

    #[test]
    fn smoothing_groups_by_polygon_make_the_normals() {
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let shared = fold(&smoothing_layer("ByPolygon", "1,1"));
        assert!(
            all_near(&normals_at(&shared, [0.0; 3]), [0.0, h, h]),
            "{:?}",
            shared.prims[0].normals
        );
        assert!(
            all_near(&normals_at(&shared, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]),
            "a corner of the wall alone"
        );
        for apart in ["1,2", "0,0"] {
            let mesh = fold(&smoothing_layer("ByPolygon", apart));
            let at = normals_at(&mesh, [0.0; 3]);
            assert_eq!(at.len(), 2, "{apart}: {at:?}");
            assert!(
                at.iter().any(|n| n[2] > 0.99) && at.iter().any(|n| n[1] > 0.99),
                "{apart}: {at:?}"
            );
        }
    }

    #[test]
    fn smoothing_by_edge_splits_the_normals_at_hard_edges() {
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let soft = fold(&smoothing_layer("ByEdge", "1,1,1,1,1,1,1"));
        assert!(
            all_near(&normals_at(&soft, [1.0, 0.0, 0.0]), [0.0, h, h]),
            "{:?}",
            soft.prims[0].normals
        );
        let hard = fold(&smoothing_layer("ByEdge", "0,1,1,1,1,1,1"));
        let at = normals_at(&hard, [1.0, 0.0, 0.0]);
        assert!(
            at.iter().any(|n| n[2] > 0.99) && at.iter().any(|n| n[1] > 0.99),
            "{at:?}"
        );
    }

    #[test]
    fn holes_are_left_out() {
        let hole = "\t\tLayerElementHole: 0 {\n\t\t\tMappingInformationType: \"ByPolygon\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tHole: *2 {\n\t\t\t\ta: 0,1\n\t\t\t}\n\t\t}\n";
        let mesh = fold(hole);
        assert_eq!(mesh.triangles(), 2);
        assert!(
            mesh.prims[0].positions.iter().all(|p| p[2].abs() < 1e-6),
            "only the wall is left"
        );
    }

    #[test]
    fn textures_read_their_uv_set_by_name() {
        let uv = |n: u32, name: &str, values: &str| {
            format!(
                "\t\tLayerElementUV: {n} {{\n\t\t\tName: \"{name}\"\n\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tUV: *8 {{\n\t\t\t\ta: {values}\n\t\t\t}}\n\t\t}}\n"
            )
        };
        let layers = uv(0, "map1", "0,0,1,0,1,1,0,1") + &uv(1, "detail", "0,0,2,0,2,2,0,2");
        let texture = |id: i64, set: &str| {
            format!(
                "\tTexture: {id}, \"Texture::T{id}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"UVSet\", \"KString\", \"\", \"\", \"{set}\"\n\t\t}}\n\t}}\n"
            )
        };
        let objects = [
            quad(&layers),
            model(20, "Quad", ""),
            "\tMaterial: 30, \"Material::M\", \"\" {\n\t}\n".to_owned(),
            texture(40, "detail"),
            texture(41, "map1"),
            video(50),
        ]
        .concat();
        let doc = read(&ascii(
            &objects,
            &links(
                &[(10, 20), (20, 0), (30, 20), (50, 40), (50, 41)],
                &[(40, 30, "DiffuseColor"), (41, 30, "NormalMap")],
            ),
        ));
        let mat = &doc.json["materials"][0];
        assert_eq!(
            mat["pbrMetallicRoughness"]["baseColorTexture"]["texCoord"]
                .as_u64()
                .unwrap_or(0),
            0
        );
        assert_eq!(mat["normalTexture"]["texCoord"], 1, "{mat}");
        let attributes = &doc.json["meshes"][0]["primitives"][0]["attributes"];
        let max_u = |set: &str| {
            accessor(&doc, &attributes[set])
                .chunks_exact(2)
                .map(|t| t[0])
                .fold(0.0, f32::max)
        };
        assert!(
            (max_u("TEXCOORD_0") - 2.0).abs() < 1e-6,
            "the color texture's set comes first"
        );
        assert!((max_u("TEXCOORD_1") - 1.0).abs() < 1e-6);
    }

    /// The glTF material made from a material with these properties and `(texture id, property)` links, each
    /// texture showing video 50.
    fn material(props: &str, textures: &[(i64, &str)]) -> Value {
        let mut objects = [
            quad(""),
            model(20, "Quad", ""),
            format!("\tMaterial: 30, \"Material::M\", \"\" {{\n\t\tProperties70:  {{\n{props}\n\t\t}}\n\t}}\n"),
            video(50),
        ]
        .concat();
        let mut oo = vec![(10, 20), (20, 0), (30, 20)];
        for &(id, _) in textures {
            objects.extend(format!("\tTexture: {id}, \"Texture::T{id}\", \"\" {{\n\t}}\n").chars());
            oo.push((50, id));
        }
        let op: Vec<(i64, i64, &str)> = textures.iter().map(|&(id, p)| (id, 30, p)).collect();
        let doc = read(&ascii(&objects, &links(&oo, &op)));
        doc.json["materials"][0].clone()
    }

    fn number(v: &Value, default: f64) -> f64 {
        v.as_f64().unwrap_or(default)
    }

    #[test]
    fn a_standard_surface_reads_as_metal_rough() {
        let props = [
            prop("Maya|baseColor", "1,1,1"),
            prop("Maya|base", "0.5"),
            prop("Maya|metalness", "1"),
            prop("Maya|specularRoughness", "0.25"),
        ]
        .concat();
        let m = material(&props, &[]);
        let pbr = &m["pbrMetallicRoughness"];
        assert!(
            close(&vector(&pbr["baseColorFactor"], &[1.0; 4])[..3], &[0.5; 3]),
            "{m}"
        );
        assert!((number(&pbr["metallicFactor"], 1.0) - 1.0).abs() < 1e-6, "{m}");
        assert!((number(&pbr["roughnessFactor"], 1.0) - 0.25).abs() < 1e-6, "{m}");
    }

    #[test]
    fn a_pbr_material_with_inverted_roughness_reads_as_metal_rough() {
        let props = [
            prop("3dsMax|main|basecolor", "1,0,0"),
            prop("3dsMax|main|metalness", "0"),
            prop("3dsMax|main|roughness", "0.2"),
            prop("3dsMax|main|roughness_inv", "1"),
        ]
        .concat();
        let m = material(&props, &[]);
        let pbr = &m["pbrMetallicRoughness"];
        assert!(
            close(&vector(&pbr["baseColorFactor"], &[1.0; 4]), &[1.0, 0.0, 0.0, 1.0]),
            "{m}"
        );
        assert!(number(&pbr["metallicFactor"], 1.0).abs() < 1e-6, "{m}");
        assert!((number(&pbr["roughnessFactor"], 1.0) - 0.8).abs() < 1e-6, "{m}");
    }

    #[test]
    fn a_packed_occlusion_roughness_metal_texture_feeds_both() {
        let props = prop("3dsMax|Parameters|base_color", "1,1,1");
        let m = material(&props, &[(40, "3dsMax|Parameters|ao_roughness_metalness_map")]);
        let mr = &m["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"];
        assert!(mr.is_number(), "{m}");
        assert_eq!(&m["occlusionTexture"]["index"], mr, "{m}");
    }

    #[test]
    fn a_layered_texture_shows_its_first_layer() {
        let layered =
            "\tLayeredTexture: 45, \"LayeredTexture::L\", \"\" {\n\t}\n\tTexture: 46, \"Texture::T46\", \"\" {\n\t}\n";
        // The layered texture holds texture 46, which shows video 50.
        let objects = [
            quad(""),
            model(20, "Quad", ""),
            "\tMaterial: 30, \"Material::M\", \"\" {\n\t}\n".to_owned(),
            video(50),
            layered.to_owned(),
        ]
        .concat();
        let doc = read(&ascii(
            &objects,
            &links(
                &[(10, 20), (20, 0), (30, 20), (46, 45), (50, 46)],
                &[(45, 30, "DiffuseColor")],
            ),
        ));
        assert!(doc.json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"].is_number());
    }

    #[test]
    fn in_between_shapes_and_their_normals_become_targets() {
        let normals = "\t\tLayerElementNormal: 0 {\n\t\t\tMappingInformationType: \"ByControlPoint\"\n\t\t\tReferenceInformationType: \"Direct\"\n\t\t\tNormals: *12 {\n\t\t\t\ta: 0,0,1,0,0,1,0,0,1,0,0,1\n\t\t\t}\n\t\t}\n";
        let shape = |id: i64, z: f64| {
            format!(
                "\tGeometry: {id}, \"Geometry::S{id}\", \"Shape\" {{\n\t\tIndexes: *1 {{\n\t\t\ta: 2\n\t\t}}\n\t\tVertices: *3 {{\n\t\t\ta: 0,0,{z}\n\t\t}}\n\t\tNormals: *3 {{\n\t\t\ta: 0,0.5,0\n\t\t}}\n\t}}\n"
            )
        };
        let objects = [
            quad(normals),
            model(20, "Quad", ""),
            "\tDeformer: 70, \"Deformer::B\", \"BlendShape\" {\n\t}\n".to_owned(),
            "\tDeformer: 71, \"SubDeformer::Bulge\", \"BlendShapeChannel\" {\n\t\tDeformPercent: 25\n\t\tFullWeights: *2 {\n\t\t\ta: 50,100\n\t\t}\n\t\tProperties70:  {\n\t\t\tP: \"DeformPercent\", \"Number\", \"\", \"A\",25\n\t\t}\n\t}\n".to_owned(),
            shape(80, 1.0),
            shape(81, 4.0),
        ]
        .concat();
        let doc = read(&ascii(
            &objects,
            &links(&[(10, 20), (20, 0), (70, 10), (71, 70), (80, 71), (81, 71)], &[]),
        ));
        let mesh = &doc.json["meshes"][0];
        let targets = mesh["primitives"][0]["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 2, "{mesh}");
        let reach = |t: &Value, key: &str, k: usize| {
            accessor(&doc, &t[key])
                .chunks_exact(3)
                .map(|v| v[k])
                .fold(0.0, f32::max)
        };
        assert!(
            (reach(&targets[0], "POSITION", 2) - 1.0).abs() < 1e-6,
            "the in-between first"
        );
        assert!((reach(&targets[1], "POSITION", 2) - 4.0).abs() < 1e-6);
        assert!(reach(&targets[0], "NORMAL", 1) > 0.1, "{mesh}");
        // A quarter of the way is half the in-between.
        assert!(close(&vector(&mesh["weights"], &[]), &[0.5, 0.0]), "{mesh}");
    }

    #[test]
    fn a_channel_weight_is_shared_among_its_shapes_and_not_clamped() {
        let mut out = [0.0; 2];
        for (w, want) in [
            (0.25, [0.5, 0.0]),
            (0.75, [0.5, 0.5]),
            (1.0, [0.0, 1.0]),
            (1.5, [-1.0, 2.0]),
            (-0.2, [-0.4, 0.0]),
        ] {
            split_weight(w, &[0.5, 1.0], &mut out);
            assert!(close(&out, &want), "{w}: {out:?}");
        }
        let mut one = [0.0];
        split_weight(1.5, &[1.0], &mut one);
        assert!(close(&one, &[1.5]));
        split_weight(f64::NAN, &[1.0], &mut one);
        assert!(close(&one, &[0.0]));
    }

    fn cubic(handles: [f64; 4]) -> Curve {
        let mut c = Curve::linear(vec![0.0, 1.0], vec![0.0, 1.0]);
        c.interp[0] = Interp::Cubic;
        c.handles[0] = handles;
        c
    }

    #[test]
    fn cubic_keys_follow_their_slopes_and_handle_lengths() {
        let flat = cubic(FLAT);
        assert!((flat.at(0.5) - 0.5).abs() < 1e-9);
        assert!((flat.at(0.25) - 0.156_25).abs() < 1e-9);
        // Leaving at a slope of two: the first handle at (1/3, 2/3).
        let steep = cubic([2.0, THIRD, 0.0, THIRD]);
        assert!((steep.at(0.5) - 0.75).abs() < 1e-9);
        // Longer handles, the same either side: still half way at half time, but slower to leave.
        let long = cubic([0.0, 0.5, 0.0, 0.5]);
        assert!((long.at(0.5) - 0.5).abs() < 1e-9);
        assert!(long.at(0.25) < flat.at(0.25));
        assert!(long.at(0.0).abs() < 1e-12 && (long.at(1.0) - 1.0).abs() < 1e-12);
        // Steps: held until the next key, or jumping to it at once.
        let mut step = Curve::linear(vec![0.0, 1.0], vec![0.0, 1.0]);
        step.interp[0] = Interp::Constant;
        assert!(step.at(0.99).abs() < 1e-12 && (step.at(1.0) - 1.0).abs() < 1e-12);
        step.interp[0] = Interp::ConstantNext;
        assert!((step.at(0.01) - 1.0).abs() < 1e-12 && step.at(0.0).abs() < 1e-12);
    }

    #[test]
    fn tcb_keys_make_catmull_rom_slopes() {
        let mut c = Curve::linear(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]);
        c.interp.fill(Interp::Cubic);
        c.tcb(&[Some([0.0; 3]); 3]);
        // Out of the first key at its segment's slope (1), into the peak flat.
        assert!((c.at(0.5) - 0.625).abs() < 1e-9, "{}", c.at(0.5));
        assert!((c.at(1.5) - 0.625).abs() < 1e-9, "{}", c.at(1.5));
    }

    #[test]
    fn packed_handle_lengths_read_from_text_and_binary() {
        // 3333 out, 5000 in.
        let packed = f64::from(0x1388_0D05_u32);
        let (out, next_in) = handle_lengths(packed);
        assert!((out - THIRD).abs() < 1e-12 && (next_in - 5000.0 / 9999.0).abs() < 1e-12);
        // 2500 out, 3333 in.
        let bits = f64::from(f32::from_bits(0x0D05_09C4_u32));
        let (out, next_in) = handle_lengths(bits);
        assert!((out - 2500.0 / 9999.0).abs() < 1e-12 && (next_in - THIRD).abs() < 1e-12);
    }

    #[test]
    fn cubic_curves_in_a_file_are_sampled_along_their_shape() {
        // Lcl Translation X eases from 0 to 1 over a second.
        let curve = "\tAnimationStack: 100, \"AnimStack::Take\", \"\" {\n\t}\n\tAnimationLayer: 101, \"AnimLayer::Base\", \"\" {\n\t}\n\tAnimationCurveNode: 102, \"AnimCurveNode::T\", \"\" {\n\t\tProperties70:  {\n\t\t\tP: \"d|X\", \"Number\", \"\", \"A\",0\n\t\t}\n\t}\n\tAnimationCurve: 103, \"AnimCurve::\", \"\" {\n\t\tKeyTime: *2 {\n\t\t\ta: 0,46186158000\n\t\t}\n\t\tKeyValueFloat: *2 {\n\t\t\ta: 0,1\n\t\t}\n\t\tKeyAttrFlags: *1 {\n\t\t\ta: 8\n\t\t}\n\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,218434821,0\n\t\t}\n\t\tKeyAttrRefCount: *1 {\n\t\t\ta: 2\n\t\t}\n\t}\n";
        let objects = model(20, "Mover", "") + curve;
        let doc = read(&ascii(
            &objects,
            &links(
                &[(20, 0), (101, 100), (102, 101)],
                &[(102, 20, "Lcl Translation"), (103, 102, "d|X")],
            ),
        ));
        let animation = &doc.json["animations"][0];
        let sampler = animation["channels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["target"]["path"] == "translation")
            .map(|c| &animation["samplers"][index(&c["sampler"])])
            .unwrap();
        let times = accessor(&doc, &sampler["input"]);
        let values = accessor(&doc, &sampler["output"]);
        assert!(times.len() > 4, "{times:?}");
        let flat = cubic(FLAT);
        for (t, v) in times.iter().zip(values.chunks_exact(3)) {
            assert!((f64::from(v[0]) - flat.at(f64::from(*t))).abs() < 1e-5, "{t}: {v:?}");
        }
    }

    #[test]
    fn layers_blend_by_their_weight_and_mode() {
        let steady = |v: f64| Animated {
            curves: [Some(Curve::linear(vec![0.0, 1.0], vec![v, v])), None, None],
            fixed: [0.0; 3],
        };
        let layered = Layered {
            rest: [0.0; 3],
            layers: vec![
                (Blend::Override, 1.0, steady(1.0)),
                (Blend::Add, 0.5, steady(2.0)),
                (Blend::Override, 0.25, steady(10.0)),
            ],
        };
        // 1, plus half of 2, then a quarter of the way to 10.
        assert!((layered.at(0.5)[0] - 4.0).abs() < 1e-9);
        let scaled = Layered {
            rest: [1.0; 3],
            layers: vec![(Blend::Override, 1.0, steady(2.0)), (Blend::Multiply, 0.5, steady(4.0))],
        };
        assert!((scaled.at(0.5)[0] - 4.0).abs() < 1e-9, "2 times the square root of 4");
    }

    #[test]
    fn area_and_volume_lights_are_kept() {
        let light = |id: i64, kind: u32| {
            format!(
                "\tNodeAttribute: {id}, \"NodeAttribute::L\", \"Light\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"LightType\", \"enum\", \"\", \"\",{kind}\n\t\t\tP: \"DecayType\", \"enum\", \"\", \"\",1\n\t\t}}\n\t}}\n"
            )
        };
        let objects = [
            model(20, "Area", ""),
            model(21, "Volume", ""),
            light(60, 3),
            light(61, 4),
        ]
        .concat();
        let doc = read(&ascii(&objects, &links(&[(20, 0), (21, 0), (60, 20), (61, 21)], &[])));
        let lights = doc.json["extensions"]["KHR_lights_punctual"]["lights"]
            .as_array()
            .unwrap();
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0]["type"], "spot");
        assert_eq!(lights[0]["extras"]["lightType"], "area");
        assert_eq!(lights[0]["extras"]["decayType"], "linear");
        assert_eq!(lights[1]["type"], "point");
        assert_eq!(lights[1]["extras"]["lightType"], "volume");
    }

    // --- Lines ---------------------------------------------------------------------------------------------------

    /// Geometry 10: a `Line` through `points` (x, y, z each) by `index`, with these `Properties70` lines.
    fn line(points: &str, index: &str, props: &str) -> String {
        let count = |a: &str| a.split(',').count();
        format!(
            "\tGeometry: 10, \"Geometry::Wire\", \"Line\" {{\n\t\tProperties70:  {{\n{props}\t\t}}\n\t\tPoints: *{} {{\n\t\t\ta: {points}\n\t\t}}\n\t\tPointsIndex: *{} {{\n\t\t\ta: {index}\n\t\t}}\n\t}}\n",
            count(points),
            count(index)
        )
    }

    /// The corners of a unit square, then a point off it.
    const SQUARE: &str = "0,0,0,1,0,0,1,1,0,0,1,0,5,5,5";

    #[test]
    fn line_geometry_draws_its_polylines_unlit_in_its_color() {
        // Round three sides of the square to its last corner, written negative, then a polyline from that corner
        // back to the first. The point off the square isn't used.
        let objects = [
            line(SQUARE, "0,1,2,-4,3,-1", &prop("Color", "1,0.5,0")),
            model(20, "Wire", &prop("Lcl Translation", "10,0,0")),
        ]
        .concat();
        let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0)], &[])));
        let mesh = &doc.json["meshes"][0];
        assert_eq!(mesh["name"], "Wire");
        assert_eq!(mesh["primitives"].as_array().unwrap().len(), 1);
        assert_eq!(mesh["primitives"][0]["mode"], 1);
        assert_eq!(node(&doc, "Wire")["mesh"], 0);
        let material = &doc.json["materials"][index(&mesh["primitives"][0]["material"])];
        assert!(material["extensions"]["KHR_materials_unlit"].is_object());
        let color = vector(&material["pbrMetallicRoughness"]["baseColorFactor"], &[]);
        assert!(
            close(&color, &[1.0, f64::from(srgb_to_linear(0.5)), 0.0, 1.0]),
            "{color:?}"
        );
        let m = doc.flatten(Pose::Rest).unwrap().unwrap();
        assert_eq!((m.triangles(), m.lines()), (0, 4));
        assert_eq!(m.prims[0].positions.len(), 4);
        assert!(
            m.prims[0]
                .positions
                .iter()
                .all(|p| (10.0..=11.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]) && p[2] == 0.0),
            "{:?}",
            m.prims[0].positions
        );
    }

    #[test]
    fn line_indices_that_are_not_points_break_the_polyline() {
        // The sixth point is too far out for a float.
        let points = [SQUARE, "1e300,0,0"].join(",");
        for (index, lines) in [
            // 9 isn't a point: 0 to 1, then 2 to 3 to the fifth point, written negative.
            ("0,1,9,2,3,-5", 3),
            // Negative, past every point: it ends nothing more.
            ("0,1,-2147483648,2,3", 2),
            ("0,5,1,2", 1),
            ("-1,-2,-3", 0),
            ("0,0,1", 1),
        ] {
            let objects = [line(&points, index, ""), model(20, "Wire", "")].concat();
            let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0)], &[])));
            let drawn = doc.flatten(Pose::Rest).unwrap().map_or(0, |m| m.lines());
            assert_eq!(drawn, lines, "{index}");
            assert_eq!(doc.json.get("meshes").is_some(), lines > 0, "{index}");
        }
        // Indexes written as floats: whole numbers are indexes, other numbers no point's.
        assert_eq!((whole(3.0), whole(-1.0)), (3, -1));
        for f in [0.5, f64::NAN, f64::INFINITY, -1e300] {
            assert_eq!(whole(f), i64::MAX, "{f}");
        }
    }

    #[test]
    fn a_line_takes_its_models_first_material() {
        let material = |id: i64, name: &str| format!("\tMaterial: {id}, \"Material::{name}\", \"\" {{\n\t}}\n");
        let objects = [
            line(SQUARE, "0,1,-3", &prop("Color", "1,0,0")),
            model(20, "Wire", ""),
            material(30, "First"),
            material(31, "Second"),
        ]
        .concat();
        let doc = read(&ascii(&objects, &links(&[(10, 20), (20, 0), (30, 20), (31, 20)], &[])));
        let materials = doc.json["materials"].as_array().unwrap();
        let used = &materials[index(&doc.json["meshes"][0]["primitives"][0]["material"])];
        assert_eq!(used["name"], "First");
        assert!(materials.iter().all(|m| m.get("extensions").is_none()));
    }

    #[test]
    fn a_line_no_model_holds_is_drawn_in_its_templates_color_and_other_curves_are_counted() {
        let template = |name: &str, color: &str| {
            format!(
                "\t\tPropertyTemplate: \"{name}\" {{\n\t\t\tProperties70:  {{\n\t\t\t\tP: \"Color\", \"ColorRGB\", \"Color\", \"\",{color}\n\t\t\t}}\n\t\t}}\n"
            )
        };
        let curves = "\tGeometry: 11, \"Geometry::C\", \"NurbsCurve\" {\n\t}\n\tGeometry: 12, \"Geometry::S\", \"NurbsSurface\" {\n\t}\n\tGeometry: 13, \"Geometry::D\", \"NurbsCurve\" {\n\t}\n";
        // A mesh's template isn't a line's, even the only one there is.
        for (templates, want) in [
            (
                template("FbxMesh", "0,0,1") + &template("FbxLine", "0,1,0"),
                [0.0, 1.0, 0.0, 1.0],
            ),
            (template("FbxMesh", "0,0,1"), [1.0; 4]),
        ] {
            let definitions =
                format!("Definitions:  {{\n\tObjectType: \"Geometry\" {{\n{templates}\t}}\n}}\nObjects:  {{");
            let objects = [line(SQUARE, "0,1,2,-4", ""), curves.to_owned()].concat();
            let doc = read(&ascii(&objects, "").replacen("Objects:  {", &definitions, 1));
            assert_eq!(doc.json["nodes"][0]["mesh"], 0);
            let material = &doc.json["materials"][index(&doc.json["meshes"][0]["primitives"][0]["material"])];
            let color = vector(&material["pbrMetallicRoughness"]["baseColorFactor"], &[]);
            assert!(close(&color, &want), "{color:?}");
            assert_eq!(
                doc.json["asset"]["extras"]["source"]["skipped"],
                json!({ "NurbsCurve": 2, "NurbsSurface": 1 })
            );
            assert_eq!(doc.flatten(Pose::Rest).unwrap().unwrap().lines(), 3);
        }
    }

    #[test]
    fn what_the_file_says_of_itself_reads_from_ascii_and_binary() {
        use crate::source::{Axis, Fact, Handedness};
        let header = r#"FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
	CreationTimeStamp:  {
		Version: 1000
		Year: 2024
		Month: 3
		Day: 9
		Hour: 14
		Minute: 5
		Second: 7
		Millisecond: 0
	}
	Creator: "FBX SDK/FBX Plugins version 2020.3"
	SceneInfo: "SceneInfo::GlobalInfo", "UserData" {
		Type: "UserData"
		Version: 100
		MetaData:  {
			Version: 100
			Title: "Crate"
			Subject: "A wooden crate"
			Author: "Someone"
			Keywords: "prop, wood"
			Revision: "rev. 2"
			Comment: "first pass"
		}
		Properties70:  {
			P: "Original|ApplicationName", "KString", "", "", "Maya"
			P: "Original|ApplicationVersion", "KString", "", "", "2020"
			P: "Original|DateTime_GMT", "DateTime", "", "", "09/03/2024 14:05:07.000"
			P: "LastSaved|ApplicationName", "KString", "", "", "Maya"
			P: "LastSaved|ApplicationVersion", "KString", "", "", "2022"
			P: "LastSaved|DateTime_GMT", "DateTime", "", "", "10/03/2024 08:00:00.250"
		}
	}
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",2
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",1
		P: "FrontAxisSign", "int", "Integer", "",-1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",2.54
		P: "TimeMode", "enum", "", "",11
	}
}
"#;
        let text = ascii(
            &[quad(""), model(20, "Box", "")].concat(),
            &links(&[(10, 20), (20, 0)], &[]),
        );
        let start = text.find("Objects:").unwrap();
        let text = format!("; FBX 7.4.0 project file\n{header}{}", &text[start..]);
        let binary = crate::formats::fbx_ascii::to_binary(text.as_bytes()).unwrap();
        let memory = crate::assets::Memory::new();
        for data in [text.as_bytes(), &binary] {
            let doc = crate::formats::load_bytes(data, "t.fbx", "fbx", &memory, Keep::ALL).unwrap();
            let info = doc.source_info();
            assert_eq!(info.format.as_deref(), Some("FBX"));
            assert_eq!(info.meters_per_unit, Fact::Declared(0.0254));
            assert_eq!(info.unit.as_deref(), Some("inch"));
            assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
            assert_eq!(info.forward_axis, Fact::Declared(Axis::NegY));
            assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
            assert_eq!(info.frame_rate, Fact::Declared(24.0));
            assert!(info.converted.units && info.converted.up_axis && !info.converted.handedness);
            assert_eq!(info.tool.as_deref(), Some("Maya 2022"));
            assert_eq!(info.title.as_deref(), Some("Crate"));
            assert_eq!(info.author.as_deref(), Some("Someone"));
            assert_eq!(info.description.as_deref(), Some("A wooden crate"));
            assert_eq!(info.keywords, ["prop", "wood"]);
            assert_eq!(info.comments, ["first pass"]);
            assert_eq!(info.created.as_deref(), Some("2024-03-09T14:05:07"));
            assert_eq!(info.modified.as_deref(), Some("2024-03-10T08:00:00.250Z"));
            let source = &doc.json["asset"]["extras"]["source"];
            assert_eq!(source["creator"], "FBX SDK/FBX Plugins version 2020.3");
            assert_eq!(source["originalApplication"], "Maya 2020");
            assert_eq!(source["revision"], "rev. 2");
            assert_eq!(source["coordAxis"], "+X");
        }

        // Settings that say nothing: FBX's centimeters and Y up, assumed; the creator is the tool.
        let doc = read(
            &ascii(
                &[quad(""), model(20, "Box", "")].concat(),
                &links(&[(10, 20), (20, 0)], &[]),
            )
            .replace("\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",100\n", ""),
        );
        let info = doc.source_info();
        assert_eq!(info.meters_per_unit, Fact::Assumed(0.01));
        assert_eq!(info.unit.as_deref(), Some("centimeter"));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.handedness, Fact::Assumed(Handedness::Right));
        assert_eq!(info.frame_rate, Fact::Unknown);
        assert!(info.converted.units && !info.converted.up_axis);
        assert_eq!(info.tool, None);
    }

    #[test]
    fn damaged_line_geometry_never_panics() {
        let objects = [
            line(SQUARE, "0,1,2,-4,3,-1", &prop("Color", "1,0,0")),
            model(20, "Wire", ""),
            "\tGeometry: 11, \"Geometry::C\", \"NurbsCurve\" {\n\t}\n".to_owned(),
        ]
        .concat();
        let text = ascii(&objects, &links(&[(10, 20), (20, 0), (11, 20)], &[]));
        let binary = crate::formats::fbx_ascii::to_binary(text.as_bytes()).unwrap();
        let memory = crate::assets::Memory::new();
        let load = |data: &[u8]| {
            if let Ok(doc) = crate::formats::load_bytes(data, "t.fbx", "fbx", &memory, Keep::ALL) {
                let _flat = doc.flatten(Pose::Rest);
            }
        };
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            usize::try_from(seed >> 40).unwrap()
        };
        for data in [text.into_bytes(), binary] {
            for len in (0..data.len()).step_by(7) {
                load(&data[..len]);
            }
            for _ in 0..300 {
                let mut bad = data.clone();
                for _ in 0..=rand() % 8 {
                    let at = rand() % bad.len();
                    bad[at] = u8::try_from(rand() & 0xff).unwrap();
                }
                load(&bad);
            }
        }
    }
}

#[cfg(all(test, feature = "fs"))]
#[test]
#[ignore = "writes files; run by hand"]
fn write_fbx_fixture() {
    let Some(out) = std::env::var_os("POLYSMELT_FIXTURE_OUT") else {
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("floor.fbx"), tests::z_up_fbx()).unwrap();
}
