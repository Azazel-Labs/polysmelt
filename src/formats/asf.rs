//! Acclaim ASF/AMC motion capture as a glTF document: the `.asf` skeleton as nodes, an `.amc` motion as one
//! animation.
//!
//! An ASF file is a list of `:`-sections. `:units` gives `length` (a factor every length is multiplied by
//! 1/`length` of to get inches), `angle` (`deg` or `rad`) and `mass`. `:root` gives the root's `order` (the columns
//! its motion lines hold, `TX TY TZ RX RY RZ` mostly), `axis` (the order of its orientation's rotations),
//! `position` and `orientation`. `:bonedata` holds one `begin` … `end` block per bone: its `name`, `direction` (a
//! unit vector in world space at rest), `length`, `axis` (three angles and their order, which give the bone's own
//! axes), `dof` (the rotations, translations and length change its motion lines hold, in their order) and `limits`
//! (a `(min max)` pair per dof). `:hierarchy` holds `begin` … `end` lines naming a parent and its children; the root
//! is `root`. `#` starts a comment.
//!
//! Each bone becomes a node where it starts, which is where its parent ends: its translation is the parent's
//! direction times its length (zero under the root, which has no length), and its rotation at rest is the identity,
//! so the rest pose is the one the ASF defines. A bone with no children ends in a node named `<bone>_end` at its own
//! direction times length. The root is a node named `root` at `position`. Bones come in the order a walk of the
//! hierarchy from the root meets them, each bone's children in the order listed; a bone the hierarchy doesn't reach
//! from the root (one never listed as a child, or in a loop) hangs from the root, and is counted. A bone's `dof`,
//! `axis` and `limits` are kept in its node's extras.
//!
//! An AMC file is a motion for one skeleton. After its header (`#` comments, `:FULLY-SPECIFIED`, `:DEGREES` or
//! `:RADIANS`, which override the ASF's angle unit for the motion, and `:SAMPLES-PER-SECOND`) come frames: a frame
//! number alone on a line, then a line per bone with its name and a value per dof (per `order` entry for the root).
//! Frame k (counting the frames as they come, not by their numbers) is at k / 120 seconds, the rate motion capture
//! studios record AMC at, unless `:SAMPLES-PER-SECOND` gives another (the source info's `frameRate`, assumed when the
//! file doesn't give it). A bone
//! moves by `C·M·C⁻¹`, where `C` is the rotation its `axis` gives and `M` the product of its dof rotations; both
//! apply their rotations in the order listed (`rx ry rz` is Rz·Ry·Rx on column vectors). The root's `C` comes from
//! its `orientation`, and its translation columns set its position outright. A bone's translation dofs move it along
//! its own axes, from where it starts. The length dof (`l`) has no glTF counterpart and is dropped, and counted.
//!
//! The motion becomes one linear animation: a rotation channel for each bone with rotation dofs, and a translation
//! channel for the root and any bone with translation dofs, keyed at the frames whose lines name the bone. A bone a
//! frame leaves out has no key there, so its motion runs straight from the frame before to the one after. A line
//! with too few values, or one that isn't a finite number, is dropped, and so is a line naming no bone the skeleton
//! has; both are counted. Quaternions are kept on one side, so interpolation takes the short way.
//!
//! An AMC file doesn't hold its skeleton: it's found beside it by the file its `#!OML:ASF` line names, then by the
//! motion's own name (`walk.asf` for `walk.amc`), then by the name up to its last `_` (`02.asf` for `02_01.amc`, as
//! subject-and-trial collections name them). Not finding it is an error.
//!
//! Lengths become meters (an inch is 0.0254 m, divided by `length`): the source info's `metersPerUnit`, declared
//! when `:units` gives `length` and assumed (an inch) when it doesn't. The skeleton's `name` is its `title` and its
//! `version` the source's. Capture is Y up, as glTF is; the file doesn't say so, so `+Y` is assumed. Nothing is
//! allocated from a count the file gives: the bones are capped, and keys take room as lines are read.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::Input;
use super::scene::{self, Animation, Channel, Interp, Node, Path, Scene};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact};

/// Bones a skeleton may have: far past any capture rig.
const MAX_BONES: usize = 32_768;
/// Dofs one bone (or the root's `order`) may list. The format defines seven.
const MAX_DOFS: usize = 64;
/// Frames per second when the motion doesn't say.
const DEFAULT_RATE: f64 = 120.0;
/// Meters in an inch, the unit ASF lengths are in once divided by `length`.
const INCH: f64 = 0.0254;

/// What one motion column sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dof {
    /// A translation along axis 0, 1 or 2 (X, Y or Z).
    Translate(usize),
    /// A rotation about axis 0, 1 or 2.
    Rotate(usize),
    /// A change of the bone's length.
    Length,
    /// A column the format doesn't define: skipped.
    Other,
}

impl Dof {
    fn parse(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let axis = |c: u8| match c {
            b'x' => Some(0),
            b'y' => Some(1),
            b'z' => Some(2),
            _ => None,
        };
        match lower.as_bytes() {
            [b't', c] => axis(*c).map_or(Self::Other, Self::Translate),
            [b'r', c] => axis(*c).map_or(Self::Other, Self::Rotate),
            [b'l'] => Self::Length,
            _ => Self::Other,
        }
    }
}

/// A bone, or the root, as the ASF defines it.
#[derive(Debug, Clone)]
struct Bone {
    name: String,
    id: Option<String>,
    /// Unit length, in world space at rest.
    direction: [f64; 3],
    length: f64,
    /// The angles of the bone's axes about X, Y and Z, in the file's angle unit.
    axis: [f64; 3],
    /// The order those rotations apply in: axis indices.
    order: Vec<usize>,
    /// The dofs as written, and what each sets.
    dof_names: Vec<String>,
    dofs: Vec<Dof>,
    /// A `(min, max)` pair per dof; `None` for an infinite bound.
    limits: Vec<[Option<f64>; 2]>,
}

impl Bone {
    fn new(name: String) -> Self {
        Self {
            name,
            id: None,
            direction: [0.0; 3],
            length: 0.0,
            axis: [0.0; 3],
            order: vec![0, 1, 2],
            dof_names: Vec::new(),
            dofs: Vec::new(),
            limits: Vec::new(),
        }
    }

    /// The rotation its axes make with the world's: `axis` applied in `order`.
    fn frame(&self, radians: f64) -> [f64; 4] {
        let mut q = [0.0, 0.0, 0.0, 1.0];
        for &a in &self.order {
            q = scene::quat_mul(scene::about(a, self.axis[a] * radians), q);
        }
        q
    }
}

/// The skeleton an ASF file defines.
#[derive(Debug)]
struct Skeleton {
    name: Option<String>,
    version: Option<String>,
    /// `:units length`: lengths times 1/`length` are inches.
    length: f64,
    /// Whether `:units` gave `length` (else it's 1, inches).
    length_given: bool,
    mass: Option<f64>,
    /// Whether angles are in radians (`angle rad`), not degrees.
    radians: bool,
    /// The root: `order` as its dofs, `axis` as its order, `orientation` as its axis angles.
    root: Bone,
    position: [f64; 3],
    bones: Vec<Bone>,
    /// The `:hierarchy` lines: a parent's name and its children's.
    hierarchy: Vec<(String, Vec<String>)>,
}

/// The section of an ASF file being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Units,
    Root,
    Bones,
    Hierarchy,
    /// `:documentation` or a section the format doesn't define: its lines are skipped.
    Other,
}

/// Up to 40 characters of `s`, for an error message.
fn show(s: &str) -> String {
    s.chars().take(40).collect()
}

/// A number the line needs.
fn number(tok: Option<&str>, what: &str, no: usize) -> Result<f64> {
    match tok.and_then(|t| t.parse::<f64>().ok()).filter(|v| v.is_finite()) {
        Some(v) => Ok(v),
        None => bail!(
            Corrupt,
            "ASF line {no}: {what} needs a number, not `{}`",
            show(tok.unwrap_or("the end of the line"))
        ),
    }
}

/// Three numbers the line needs.
fn three<'a>(tokens: &mut impl Iterator<Item = &'a str>, what: &str, no: usize) -> Result<[f64; 3]> {
    Ok([
        number(tokens.next(), what, no)?,
        number(tokens.next(), what, no)?,
        number(tokens.next(), what, no)?,
    ])
}

/// Axis letters (`XYZ`, `zyx`) as axis indices; `None` if any isn't one.
fn axis_order(s: &str) -> Option<Vec<usize>> {
    s.bytes()
        .map(|c| match c.to_ascii_lowercase() {
            b'x' => Some(0),
            b'y' => Some(1),
            b'z' => Some(2),
            _ => None,
        })
        .collect()
}

/// A line without its comment.
fn uncomment(line: &str) -> &str {
    line.split('#').next().unwrap_or("")
}

/// Reads an ASF file.
fn parse_asf(text: &str) -> Result<Skeleton> {
    let mut sk = Skeleton {
        name: None,
        version: None,
        length: 1.0,
        length_given: false,
        mass: None,
        radians: false,
        root: Bone::new("root".into()),
        position: [0.0; 3],
        bones: Vec::new(),
        hierarchy: Vec::new(),
    };
    sk.root.dofs.clear();
    let mut section = Section::None;
    let mut seen = false;
    // The bone being read, between `begin` and `end`.
    let mut bone: Option<Bone> = None;
    // Whether the last keyword was `limits`, which goes on over the lines after it.
    let mut in_limits = false;
    let mut names: HashMap<String, usize> = HashMap::new();
    for (k, raw) in text.lines().enumerate() {
        let no = k + 1;
        let line = uncomment(raw).trim();
        let mut tokens = line.split_whitespace();
        let Some(first) = tokens.next() else { continue };
        if let Some(key) = first.strip_prefix(':') {
            in_limits = false;
            ensure!(
                bone.is_none(),
                Corrupt,
                "ASF line {no}: `{}` inside a bone's begin … end",
                show(first)
            );
            let rest = line[first.len()..].trim();
            section = match key.to_ascii_lowercase().as_str() {
                "version" => {
                    sk.version = Some(rest.to_owned());
                    Section::Other
                }
                "name" => {
                    sk.name = Some(rest.to_owned());
                    Section::Other
                }
                "units" => Section::Units,
                "root" => Section::Root,
                "bonedata" => Section::Bones,
                "hierarchy" => Section::Hierarchy,
                _ => Section::Other,
            };
            seen = true;
            continue;
        }
        let key = first.to_ascii_lowercase();
        match section {
            Section::None => bail!(
                UnknownFormat,
                "not an ASF file: it starts with `{}`, not a `:` section",
                show(first)
            ),
            Section::Other => {}
            Section::Units => match key.as_str() {
                "length" => {
                    let v = number(tokens.next(), "length", no)?;
                    ensure!(
                        v.is_finite() && v > 0.0,
                        Corrupt,
                        "ASF line {no}: the length unit must be above 0, not {v}"
                    );
                    sk.length = v;
                    sk.length_given = true;
                }
                "mass" => sk.mass = Some(number(tokens.next(), "mass", no)?),
                "angle" => {
                    let unit = tokens.next().unwrap_or("").to_ascii_lowercase();
                    sk.radians = match unit.as_str() {
                        "deg" | "degree" | "degrees" => false,
                        "rad" | "radian" | "radians" => true,
                        _ => bail!(
                            Corrupt,
                            "ASF line {no}: the angle unit `{}` isn't deg or rad",
                            show(&unit)
                        ),
                    };
                }
                _ => {}
            },
            Section::Root => match key.as_str() {
                "order" => {
                    sk.root.dof_names = tokens.map(str::to_owned).collect();
                    ensure!(
                        sk.root.dof_names.len() <= MAX_DOFS,
                        LimitExceeded,
                        "ASF line {no}: the root's order lists more than {MAX_DOFS} columns"
                    );
                    sk.root.dofs = sk.root.dof_names.iter().map(|d| Dof::parse(d)).collect();
                }
                "axis" => {
                    let order = tokens.next().unwrap_or("");
                    let Some(order) = axis_order(order).filter(|o| !o.is_empty()) else {
                        bail!(
                            Corrupt,
                            "ASF line {no}: the root's axis `{}` isn't an order of X, Y and Z",
                            show(order)
                        );
                    };
                    sk.root.order = order;
                }
                "position" => sk.position = three(&mut tokens, "position", no)?,
                "orientation" => sk.root.axis = three(&mut tokens, "orientation", no)?,
                _ => {}
            },
            Section::Bones => {
                if key == "begin" {
                    ensure!(
                        bone.is_none(),
                        Corrupt,
                        "ASF line {no}: `begin` inside a bone's begin … end"
                    );
                    ensure!(
                        sk.bones.len() < MAX_BONES,
                        LimitExceeded,
                        "ASF has more than {MAX_BONES} bones"
                    );
                    bone = Some(Bone::new(String::new()));
                    in_limits = false;
                    continue;
                }
                let Some(b) = bone.as_mut() else {
                    bail!(Corrupt, "ASF line {no}: `{}` outside a bone's begin … end", show(first));
                };
                if key == "end" {
                    let mut b = bone.take().unwrap_or_else(|| Bone::new(String::new()));
                    if b.name.is_empty() {
                        b.name = format!("bone{}", sk.bones.len() + 1);
                    }
                    ensure!(
                        !b.name.eq_ignore_ascii_case("root") && !names.contains_key(&b.name),
                        Corrupt,
                        "ASF line {no}: bone `{}` is defined twice",
                        show(&b.name)
                    );
                    names.insert(b.name.clone(), sk.bones.len());
                    sk.bones.push(b);
                    in_limits = false;
                    continue;
                }
                let continues = in_limits && first.starts_with('(');
                in_limits = false;
                if continues || key == "limits" || key.starts_with("limits(") {
                    // `limits (a b)`, its pairs going on over the lines after it, parentheses touching or not.
                    let body = if continues { line } else { &line[6..] };
                    let cleaned = body.replace(['(', ')'], " ");
                    for pair in cleaned.split_whitespace().collect::<Vec<_>>().chunks(2) {
                        let bound = |t: &str| match t.parse::<f64>() {
                            Ok(v) if v.is_finite() => Ok(Some(v)),
                            Ok(v) if v.is_infinite() => Ok(None),
                            _ => Err(err!(
                                Corrupt,
                                "ASF line {no}: a limit needs a number, not `{}`",
                                show(t)
                            )),
                        };
                        let [lo, hi] = [pair[0], pair.get(1).copied().unwrap_or("")];
                        ensure!(
                            b.limits.len() < MAX_DOFS,
                            LimitExceeded,
                            "ASF line {no}: bone `{}` has more than {MAX_DOFS} limits",
                            show(&b.name)
                        );
                        b.limits.push([bound(lo)?, bound(hi)?]);
                    }
                    in_limits = true;
                    continue;
                }
                match key.as_str() {
                    "id" => b.id = tokens.next().map(str::to_owned),
                    "name" => {
                        let name = line[first.len()..].trim();
                        ensure!(!name.is_empty(), Corrupt, "ASF line {no}: a bone's name is empty");
                        name.clone_into(&mut b.name);
                    }
                    "direction" => {
                        let d = three(&mut tokens, "direction", no)?;
                        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        b.direction = if len > 0.0 { d.map(|c| c / len) } else { [0.0; 3] };
                    }
                    "length" => b.length = number(tokens.next(), "length", no)?,
                    "axis" => {
                        b.axis = three(&mut tokens, "axis", no)?;
                        if let Some(o) = tokens.next() {
                            let Some(order) = axis_order(o).filter(|o| !o.is_empty()) else {
                                bail!(
                                    Corrupt,
                                    "ASF line {no}: axis order `{}` isn't an order of X, Y and Z",
                                    show(o)
                                );
                            };
                            b.order = order;
                        }
                    }
                    "dof" => {
                        b.dof_names = tokens.map(str::to_owned).collect();
                        ensure!(
                            b.dof_names.len() <= MAX_DOFS,
                            LimitExceeded,
                            "ASF line {no}: bone `{}` lists more than {MAX_DOFS} dofs",
                            show(&b.name)
                        );
                        b.dofs = b.dof_names.iter().map(|d| Dof::parse(d)).collect();
                    }
                    _ => {}
                }
            }
            Section::Hierarchy => {
                if key == "begin" || key == "end" {
                    continue;
                }
                sk.hierarchy
                    .push((first.to_owned(), tokens.map(str::to_owned).collect()));
            }
        }
    }
    ensure!(seen, UnknownFormat, "not an ASF file: it has no `:` sections");
    if let Some(b) = bone {
        bail!(
            Truncated,
            "ASF file is cut short inside bone `{}`, or its `end` is missing",
            show(&b.name)
        );
    }
    Ok(sk)
}

/// Axis indices as letters: `XYZ`.
fn order_name(order: &[usize]) -> String {
    order.iter().map(|&a| ["X", "Y", "Z"][a]).collect()
}

/// Where a bone hangs: from the root, or from another bone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parent {
    Root,
    Bone(usize),
}

/// The skeleton's nodes in a scene, and what the motion needs to find them.
struct Rig {
    /// Each bone's node.
    nodes: Vec<usize>,
    root: usize,
    /// Bones the hierarchy doesn't reach from the root.
    unattached: usize,
}

/// Adds the skeleton's nodes to the scene, in the order a walk from the root meets them.
fn build(sk: &Skeleton, scene: &mut Scene) -> Rig {
    let index: HashMap<&str, usize> = sk.bones.iter().enumerate().map(|(i, b)| (b.name.as_str(), i)).collect();
    let n = sk.bones.len();
    let mut parent: Vec<Option<Parent>> = vec![None; n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut root_children = Vec::new();
    for (p, kids) in &sk.hierarchy {
        let from = if p.eq_ignore_ascii_case("root") {
            Some(Parent::Root)
        } else {
            index.get(p.as_str()).map(|&i| Parent::Bone(i))
        };
        let Some(from) = from else { continue };
        for kid in kids {
            let Some(&c) = index.get(kid.as_str()) else { continue };
            // A bone keeps the first parent named for it, and can't hang from itself.
            if parent[c].is_some() || from == Parent::Bone(c) {
                continue;
            }
            parent[c] = Some(from);
            match from {
                Parent::Root => root_children.push(c),
                Parent::Bone(b) => children[b].push(c),
            }
        }
    }
    let mut root_node = Node::named("root", None);
    root_node.translation = sk.position.map(f32_of);
    root_node.extras = Some(json!({
        "order": sk.root.dof_names,
        "axis": order_name(&sk.root.order),
        "orientation": sk.root.axis,
    }));
    let root = scene.add(root_node);
    let mut nodes = vec![usize::MAX; n];
    let mut unattached = 0;
    // A walk from the root first, then from each bone left over, in file order: a walk only enters bones not met
    // yet, so a loop in the hierarchy is cut where the walk closes it. Each entry is a bone to add, the node it
    // hangs from, and where it starts in that node's space.
    let mut stack: Vec<(usize, usize, [f64; 3])> = root_children.iter().rev().map(|&c| (c, root, [0.0; 3])).collect();
    let mut loose = 0;
    loop {
        while let Some((b, under, at)) = stack.pop() {
            if nodes[b] != usize::MAX {
                continue;
            }
            let bone = &sk.bones[b];
            let mut node = Node::named(bone.name.clone(), Some(under));
            node.translation = at.map(f32_of);
            let mut extras = Map::new();
            if let Some(id) = &bone.id {
                extras.insert("id".into(), json!(id));
            }
            extras.insert("dof".into(), json!(bone.dof_names));
            extras.insert("axis".into(), json!(bone.axis));
            extras.insert("axisOrder".into(), json!(order_name(&bone.order)));
            if !bone.limits.is_empty() {
                extras.insert("limits".into(), json!(bone.limits));
            }
            node.extras = Some(Value::Object(extras));
            let me = scene.add(node);
            nodes[b] = me;
            let tip = bone.direction.map(|c| c * bone.length);
            if children[b].is_empty() {
                let mut end = Node::named(format!("{}_end", bone.name), Some(me));
                end.translation = tip.map(f32_of);
                scene.add(end);
            }
            for &c in children[b].iter().rev() {
                stack.push((c, me, tip));
            }
        }
        while loose < n && nodes[loose] != usize::MAX {
            loose += 1;
        }
        if loose == n {
            break;
        }
        unattached += 1;
        stack.push((loose, root, [0.0; 3]));
    }
    Rig {
        nodes,
        root,
        unattached,
    }
}

/// One bone's (or the root's) keys.
#[derive(Default)]
struct Track {
    /// The frames it has keys at.
    frames: Vec<usize>,
    rotations: Vec<f32>,
    translations: Vec<f32>,
    /// The frame its last key is at.
    last: Option<usize>,
}

/// What reading the motion found, besides the keys.
#[derive(Default)]
struct Motion {
    frames: usize,
    /// Seconds per frame, and whether the file gave it.
    frame_time: f64,
    rate_given: bool,
    dropped: usize,
    unknown: usize,
}

/// The rotation dofs of a line's values make, applied in the order listed.
fn rotation(dofs: &[Dof], values: &[f64], radians: f64) -> [f64; 4] {
    let mut q = [0.0, 0.0, 0.0, 1.0];
    for (d, v) in dofs.iter().zip(values) {
        if let Dof::Rotate(a) = *d {
            q = scene::quat_mul(scene::about(a, v * radians), q);
        }
    }
    q
}

/// `q` turned around: the inverse of a unit quaternion.
fn conjugate(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Reads an AMC motion for `sk`, whose nodes are in `rig`, as an animation.
fn parse_amc(text: &str, sk: &Skeleton, rig: &Rig, scene: &Scene) -> (Option<Animation>, Motion) {
    let mut motion = Motion {
        frame_time: 1.0 / DEFAULT_RATE,
        ..Motion::default()
    };
    // The bones by name, then the root; the root is the last track.
    let mut by_name: HashMap<&str, usize> = sk.bones.iter().enumerate().map(|(i, b)| (b.name.as_str(), i)).collect();
    let root = sk.bones.len();
    by_name.insert("root", root);
    let lower: HashMap<String, usize> = by_name.iter().map(|(k, &v)| (k.to_ascii_lowercase(), v)).collect();
    let mut tracks: Vec<Track> = std::iter::repeat_with(Track::default).take(root + 1).collect();
    let mut angle = if sk.radians { 1.0 } else { std::f64::consts::PI / 180.0 };
    let skeleton_angle = if sk.radians { 1.0 } else { std::f64::consts::PI / 180.0 };
    let mut values: Vec<f64> = Vec::new();
    let mut frame: Option<usize> = None;
    for raw in text.lines() {
        let line = raw.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(key) = line.strip_prefix(':') {
            let mut parts = key.split_whitespace();
            match parts.next().unwrap_or("").to_ascii_uppercase().as_str() {
                "DEGREES" => angle = std::f64::consts::PI / 180.0,
                "RADIANS" => angle = 1.0,
                "SAMPLES-PER-SECOND" => {
                    if let Some(rate) = parts
                        .next()
                        .and_then(|t| t.parse::<f64>().ok())
                        .filter(|r| r.is_finite() && *r > 0.0)
                    {
                        motion.frame_time = 1.0 / rate;
                        motion.rate_given = true;
                    }
                }
                _ => {}
            }
            continue;
        }
        let mut tokens = line.split_whitespace();
        let Some(name) = tokens.next() else { continue };
        if tokens.clone().next().is_none() && name.bytes().all(|c| c.is_ascii_digit()) {
            frame = Some(motion.frames);
            motion.frames += 1;
            continue;
        }
        let Some(f) = frame else {
            // A bone line before any frame number.
            motion.dropped += 1;
            continue;
        };
        let Some(&b) = by_name.get(name).or_else(|| lower.get(&name.to_ascii_lowercase())) else {
            motion.unknown += 1;
            continue;
        };
        let bone = if b == root { &sk.root } else { &sk.bones[b] };
        values.clear();
        for t in tokens.by_ref().take(bone.dofs.len()) {
            match t.parse::<f64>() {
                Ok(v) if v.is_finite() => values.push(v),
                _ => break,
            }
        }
        if values.len() < bone.dofs.len() {
            motion.dropped += 1;
            continue;
        }
        let track = &mut tracks[b];
        if track.last == Some(f) {
            // A second line for the bone in one frame replaces the first.
            track.frames.pop();
            let r = track.rotations.len().saturating_sub(4);
            track.rotations.truncate(r);
            let t = track.translations.len().saturating_sub(3);
            track.translations.truncate(t);
        }
        track.last = Some(f);
        track.frames.push(f);
        let c = bone.frame(skeleton_angle);
        let m = rotation(&bone.dofs, &values, angle);
        let q = scene::quat_mul(scene::quat_mul(c, m), conjugate(c));
        if bone.dofs.iter().any(|d| matches!(d, Dof::Rotate(_))) {
            track.rotations.extend(scene::normalize(q));
        }
        if bone.dofs.iter().any(|d| matches!(d, Dof::Translate(_))) {
            let node = if b == root { rig.root } else { rig.nodes[b] };
            let rest = scene.nodes[node].translation.map(f64::from);
            let mut t = if b == root { rest } else { [0.0; 3] };
            for (d, v) in bone.dofs.iter().zip(&values) {
                if let Dof::Translate(a) = *d {
                    t[a] = *v;
                }
            }
            // The root's columns set its position (an axis without one keeps the ASF's); a bone's move it along
            // its own axes, from where it starts.
            let at = if b == root {
                t
            } else {
                let along = scene::rotate(c, t);
                [rest[0] + along[0], rest[1] + along[1], rest[2] + along[2]]
            };
            track.translations.extend(at.map(f32_of));
        }
    }
    let mut channels = Vec::new();
    for (b, track) in tracks.into_iter().enumerate() {
        if track.frames.is_empty() {
            continue;
        }
        #[allow(clippy::cast_precision_loss, reason = "frame counts are far below 2^52")]
        let times: Vec<f32> = track
            .frames
            .iter()
            .map(|&f| f32_of(f as f64 * motion.frame_time))
            .collect();
        let node = if b == root { rig.root } else { rig.nodes[b] };
        if !track.rotations.is_empty() {
            channels.push(Channel {
                node,
                path: Path::Rotation,
                times: times.clone(),
                values: track.rotations,
                interp: Interp::Linear,
            });
        }
        if !track.translations.is_empty() {
            channels.push(Channel {
                node,
                path: Path::Translation,
                times,
                values: track.translations,
                interp: Interp::Linear,
            });
        }
    }
    let animation = (!channels.is_empty()).then(|| Animation {
        name: "motion".into(),
        channels,
        extras: None,
    });
    (animation, motion)
}

/// The text of a file: UTF-8, or Latin-1 when it isn't, without a byte order mark.
fn decode(bytes: &[u8]) -> String {
    let mut s = scene::text(bytes);
    if s.starts_with('\u{feff}') {
        s.drain(..'\u{feff}'.len_utf8());
    }
    s
}

/// The names an AMC motion's skeleton may go by: the `#!OML:ASF` line's file, the motion's own name, and its name
/// up to the last `_`.
fn skeleton_names(amc: &str, path: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    for line in amc.lines().take(64) {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("#!OML:ASF") {
            let r = rest.trim();
            if !r.is_empty() {
                out.push(r.to_owned());
            }
        }
        if line.starts_with(':') || line.bytes().next().is_some_and(|c| c.is_ascii_digit()) {
            break;
        }
    }
    if let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) {
        out.push(format!("{stem}.asf"));
        if let Some((subject, _)) = stem.rsplit_once('_')
            && !subject.is_empty()
        {
            out.push(format!("{subject}.asf"));
        }
    }
    out
}

/// The `name` part of a path someone else's machine wrote, `\` or `/` separated.
fn file_name(reference: &str) -> &str {
    reference.rsplit(['/', '\\']).next().unwrap_or(reference)
}

pub(crate) fn load(input: Input<'_>, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let bytes = input.read()?;
    let text = decode(&bytes);
    let amc = ext.eq_ignore_ascii_case("amc");
    let (sk, skeleton_file) = if amc {
        let names = skeleton_names(&text, input.path);
        let mut found = None;
        for name in &names {
            if let Some(path) = assets.find(name, input.dir())
                && let Some(b) = assets.read(&path)
            {
                found = Some((decode(&b), path));
                break;
            }
        }
        let Some((asf, path)) = found else {
            bail!(
                MissingAsset,
                "AMC motion needs its ASF skeleton beside it, and none was found (looked for {})",
                names.iter().map(|n| file_name(n)).collect::<Vec<_>>().join(", ")
            );
        };
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        (parse_asf(&asf)?, name)
    } else {
        (parse_asf(&text)?, None)
    };

    let mut scene = Scene::default();
    let rig = build(&sk, &mut scene);
    let mut source = Map::new();
    let info = &mut scene.info;
    info.format = Some(if amc { "AMC" } else { "ASF" }.into());
    info.version.clone_from(&sk.version);
    info.title.clone_from(&sk.name);
    let meters = INCH / sk.length;
    info.meters_per_unit = if sk.length_given {
        Fact::Declared(meters)
    } else {
        Fact::Assumed(meters)
    };
    if (sk.length - 1.0).abs() < 1e-12 {
        info.unit = Some("inch".into());
    }
    // Motion capture stands Y up, as the format's own examples do, but the file doesn't say so.
    info.up_axis = Fact::Assumed(Axis::PosY);
    source.insert("length".into(), json!(sk.length));
    source.insert("angle".into(), json!(if sk.radians { "rad" } else { "deg" }));
    if let Some(m) = sk.mass {
        source.insert("mass".into(), json!(m));
    }
    if let Some(f) = &skeleton_file {
        source.insert("skeleton".into(), json!(f));
    }
    let facts = &mut scene.facts;
    facts.insert("bones".into(), json!(sk.bones.len()));
    let dofs: usize = sk.root.dofs.len() + sk.bones.iter().map(|b| b.dofs.len()).sum::<usize>();
    facts.insert("channels".into(), json!(dofs));
    let lengths = sk.bones.iter().filter(|b| b.dofs.contains(&Dof::Length)).count();
    if lengths > 0 {
        facts.insert("length_dofs_dropped".into(), json!(lengths));
    }
    if rig.unattached > 0 {
        facts.insert("unattached_bones".into(), json!(rig.unattached));
    }
    if amc {
        let (animation, motion) = parse_amc(&text, &sk, &rig, &scene);
        let facts = &mut scene.facts;
        facts.insert("frames".into(), json!(motion.frames));
        if motion.dropped > 0 {
            facts.insert("dropped_lines".into(), json!(motion.dropped));
        }
        if motion.unknown > 0 {
            facts.insert("unknown_bone_lines".into(), json!(motion.unknown));
        }
        let rate = 1.0 / motion.frame_time;
        scene.info.frame_rate = if motion.rate_given {
            Fact::Declared(rate)
        } else {
            Fact::Assumed(rate)
        };
        source.insert("frames".into(), json!(motion.frames));
        if keep.animations
            && let Some(a) = animation
        {
            scene.animations.push(a);
        }
    }
    scene.source = source;
    scene.scale(f32_of(INCH / sk.length));
    scene.finish(keep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;

    const ASF: &str = "# a two-bone arm
:version 1.10
:name test
:units
  mass 1.0
  length 0.0254
  angle deg
:documentation
  begin end
:root
  order TX TY TZ RX RY RZ
  axis XYZ
  position 0 0 0
  orientation 0 0 0
:bonedata
  begin
    id 1
    name upper
    direction 1 0 0
    length 2
    axis 0 0 90 XYZ
    dof rx ry rz
    limits (-180 180)
           (-inf inf)
           (-90.0 90.0)
  end
  begin
    id 2
    name lower
    direction 0 1 0
    length 3
    axis 0 0 0 XYZ
    dof rz l
    limits (-10 10) (0 1)
  end
:hierarchy
  begin
    root upper
    upper lower
  end
";

    fn asf(text: &str) -> Result<Document<'static>> {
        load(
            Input::memory(text.as_bytes(), "t.asf"),
            "asf",
            &Memory::new(),
            Keep::ALL,
        )
    }

    fn amc(asf_text: &str, motion: &str) -> Result<Document<'static>> {
        let assets = Memory::new().with("t.asf", asf_text.as_bytes());
        load(Input::memory(motion.as_bytes(), "t.amc"), "amc", &assets, Keep::ALL)
    }

    fn error(r: Result<Document<'static>>) -> String {
        match r {
            Ok(_) => panic!("read without error"),
            Err(e) => e.to_string(),
        }
    }

    fn near(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn skeleton() {
        let doc = asf(ASF).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        let names: Vec<_> = nodes.iter().map(|n| n["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["root", "upper", "lower", "lower_end"]);
        // `length 0.0254` makes a unit a meter.
        assert_eq!(nodes[2]["translation"], json!([2.0, 0.0, 0.0]));
        assert_eq!(nodes[3]["translation"], json!([0.0, 3.0, 0.0]));
        assert!(nodes[1].get("translation").is_none());
        assert_eq!(doc.facts["bones"], json!(2));
        assert_eq!(doc.facts["channels"], json!(11));
        assert_eq!(doc.facts["length_dofs_dropped"], json!(1));
        assert_eq!(
            nodes[1]["extras"]["limits"],
            json!([[-180.0, 180.0], [null, null], [-90.0, 90.0]])
        );
        assert_eq!(nodes[2]["extras"]["dof"], json!(["rz", "l"]));
        assert_eq!(doc.json["asset"]["extras"]["source"]["format"], "ASF");
        assert!(doc.json.get("animations").is_none());
        let info = doc.source_info();
        // `length 0.0254` makes the unit a meter, as the file declares.
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.frame_rate, Fact::Unknown);
        assert_eq!(info.title.as_deref(), Some("test"));
        assert_eq!(info.version.as_deref(), Some("1.10"));
        assert!(!info.converted.units && !info.converted.up_axis);
    }

    #[test]
    fn motion() {
        let text = "#!OML:ASF elsewhere/t.asf\n:FULLY-SPECIFIED\n:DEGREES\n1\nroot 1 2 3 0 0 0\nupper 0 0 0\nlower 0 0\n2\nroot 1 2 3 0 0 0\nupper 90 0 0\nlower 45 0.5\n";
        let doc = amc(ASF, text).unwrap();
        assert_eq!(doc.facts["frames"], json!(2));
        assert!(doc.facts.get("frame_time_guessed").is_none());
        assert_eq!(doc.source_info().frame_rate, Fact::Assumed(120.0));
        assert_eq!(doc.json["asset"]["extras"]["source"]["skeleton"], "t.asf");
        let rig = doc.rig().unwrap();
        let a = &rig.animations[0];
        assert_eq!(a.channels.len(), 4);
        let pose = rig.pose(Some(0), 1.0 / 120.0);
        // Root translation sets the position, in meters.
        assert!(near(&pose[0].translation, &[1.0, 2.0, 3.0]));
        // upper: axis turned 90° about Z, so its rx is a turn about world Y: C·Rx·C⁻¹ = Ry(90)... about C's X axis,
        // which is world Y.
        let q = pose[1].rotation;
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!(near(&q, &[0.0, s, 0.0, s]) || near(&q, &[0.0, -s, 0.0, -s]), "{q:?}");
        // lower: 45° about Z.
        let h = (std::f32::consts::PI / 8.0).sin();
        let c = (std::f32::consts::PI / 8.0).cos();
        assert!(near(&pose[2].rotation, &[0.0, 0.0, h, c]));
    }

    #[test]
    fn skeleton_by_name() {
        let motion = ":DEGREES\n1\nroot 0 0 0 0 0 0\n";
        let assets = Memory::new().with("02.asf", ASF.as_bytes());
        let doc = load(Input::memory(motion.as_bytes(), "02_01.amc"), "amc", &assets, Keep::ALL).unwrap();
        assert_eq!(doc.json["asset"]["extras"]["source"]["skeleton"], "02.asf");
        let e = error(load(
            Input::memory(motion.as_bytes(), "x.amc"),
            "amc",
            &assets,
            Keep::ALL,
        ));
        assert!(e.contains("x.asf"), "{e}");
    }

    #[test]
    fn radians_and_rate() {
        let text = ":RADIANS\n:SAMPLES-PER-SECOND 60\n1\nlower 1.5707963267948966 0\n2\nlower 0 0\n";
        let doc = amc(ASF, text).unwrap();
        let info = doc.source_info();
        assert_eq!(info.frame_rate, Fact::Declared(60.0));
        assert_eq!(info.format.as_deref(), Some("AMC"));
        let rig = doc.rig().unwrap();
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!(near(&rig.pose(Some(0), 0.0)[2].rotation, &[0.0, 0.0, s, s]));
        assert_eq!(rig.animations[0].channels[0].times, [0.0, 1.0 / 60.0]);
    }

    #[test]
    fn bad_lines_are_dropped() {
        let text = "upper 1 2 3\n1\nupper 1 2\nupper 1 nan 3\nnobody 1\nlower 1 2\n2\n3\nlower 4 5\n";
        let doc = amc(ASF, text).unwrap();
        assert_eq!(doc.facts["frames"], json!(3));
        assert_eq!(doc.facts["dropped_lines"], json!(3));
        assert_eq!(doc.facts["unknown_bone_lines"], json!(1));
        let rig = doc.rig().unwrap();
        // Only `lower` has keys, at frames 0 and 2.
        assert_eq!(rig.animations[0].channels.len(), 1);
        assert_eq!(rig.animations[0].channels[0].times, [0.0, 2.0 / 120.0]);
    }

    #[test]
    fn loops_and_loose_bones_hang_from_the_root() {
        let text = ASF.replace("root upper\n", "lower upper\n");
        let doc = asf(&text).unwrap();
        assert_eq!(doc.facts["unattached_bones"], json!(1));
        let rig = doc.rig().unwrap();
        let upper = rig.find("upper").unwrap();
        let lower = rig.find("lower").unwrap();
        assert_eq!(rig.nodes[upper].parent, Some(0));
        assert_eq!(rig.nodes[lower].parent, Some(upper));
    }

    #[test]
    fn bad_skeletons_are_errors() {
        let cases = [
            ("", "no `:` sections"),
            ("hello\n", "not an ASF file"),
            (":bonedata\nbegin\nname a\n", "is cut short inside bone"),
            (":bonedata\nname a\n", "outside a bone"),
            (":bonedata\nbegin\nname a\nlength x\nend\n", "length needs a number"),
            (":bonedata\nbegin\nname a\ndirection 1 0\nend\n", "direction needs"),
            (":bonedata\nbegin\nname a\nend\nbegin\nname a\nend\n", "defined twice"),
            (":bonedata\nbegin\nname root\nend\n", "defined twice"),
            (":units\nangle grad\n", "angle unit"),
            (":units\nlength 0\n", "above 0"),
            (":bonedata\nbegin\nname a\naxis 0 0 0 XQ\nend\n", "axis order"),
            (":bonedata\nbegin\nname a\nlimits (0 x)\nend\n", "limit"),
            (":bonedata\nbegin\n:hierarchy\n", "inside a bone"),
        ];
        for (text, want) in cases {
            let e = error(asf(text));
            assert!(e.contains(want), "{text:?}: {e}");
        }
        let many = format!(":bonedata\n{}", "begin\nend\n".repeat(MAX_BONES + 1));
        assert!(error(asf(&many)).contains("more than"));
    }

    #[test]
    fn truncated_files_never_panic() {
        let motion = "1\nroot 1 2 3 4 5 6\nupper 1 2 3\nlower 4 5\n2\nroot 1 2 3 4 5 6\n";
        for cut in 0..ASF.len() {
            let _ = asf(&ASF[..cut]);
        }
        for cut in 0..motion.len() {
            let _ = amc(ASF, &motion[..cut]).unwrap();
        }
    }

    #[test]
    fn keeping_no_animations_keeps_the_facts() {
        let text = "1\nroot 1 2 3 0 0 0\n";
        let assets = Memory::new().with("t.asf", ASF.as_bytes());
        let keep = Keep {
            animations: false,
            ..Keep::ALL
        };
        let none = load(Input::memory(text.as_bytes(), "t.amc"), "amc", &assets, keep).unwrap();
        let all = amc(ASF, text).unwrap();
        assert!(none.json.get("animations").is_none());
        assert_eq!(none.facts, all.facts);
    }
}
