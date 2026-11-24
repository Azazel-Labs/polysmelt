//! HTR (Hierarchical Translation-Rotation) motion capture as a glTF document: the segments as nodes, the motion as
//! one animation.
//!
//! The file is in `[`sections`]`, `#` starting a comment. `[Header]` holds `name value` lines: `NumSegments`,
//! `NumFrames`, `DataFrameRate` (frames per second), `EulerRotationOrder` (`ZYX` is Rz·Ry·Rx on column vectors: the
//! last letter's rotation applies first), `CalibrationUnits` (`mm`, `cm`, `m`, `in`, or a fraction of one such as
//! `1/10.000 m`), `RotationUnits` (`Degrees` or `Radians`), `GlobalAxisofGravity` (the up axis), `BoneLengthAxis`
//! and `ScaleFactor`. `[SegmentNames&Hierarchy]` lists each segment and its parent, `GLOBAL` for a root.
//! `[BasePosition]` gives each segment's base pose: a translation, three angles and its bone length. Then each
//! segment has a section of its own, named after it, with a row per frame: the frame number, a translation, three
//! angles and a scale factor. `[EndOfFile]` ends the file.
//!
//! Each segment becomes a node, in the order the hierarchy lists them, its base pose the rest pose: its translation
//! is the base translation and its rotation the base angles'. A segment without children ends in a node named
//! `<segment>_end`, its bone length along the bone length axis. A frame's row is relative to the base pose: the
//! segment's translation is the base's plus the row's, and its rotation is the base rotation followed by the row's
//! (`T(base + row)·R(base)·R(row)` on column vectors). A file without a hierarchy (the global form) has each segment
//! named in `[BasePosition]` as a root. A parent that isn't a segment makes a root too.
//!
//! The rows become one linear animation, a translation and a rotation channel per segment with rows, frame `n` at
//! `(n − first) / DataFrameRate` seconds, `first` being the lowest frame number in the file. A row with too few
//! numbers, or one that isn't a finite number, or whose frame number doesn't follow the row before it, is left out
//! and counted; so are rows in a section that names no segment. The scale factor column stretches the bone, which
//! glTF has no way to say, so rows whose factor isn't 1 are counted, and the factor is dropped. The header's
//! `ScaleFactor` is recorded and not applied. A missing or bad `DataFrameRate` is taken as 30.
//!
//! Lengths become meters, and a Z (or X) gravity axis is turned to glTF's Y up. The source info has
//! `CalibrationUnits` as the `unit` and its length as `metersPerUnit`, declared (unknown when it's missing or not a
//! unit, which leaves lengths as they are), `GlobalAxisofGravity` as the up axis, declared (`+Y` assumed when it's
//! missing, unknown when it isn't an axis), `DataFrameRate` as `frameRate`, declared (30 assumed), and
//! `FileVersion` as the version. Nothing is allocated from a count
//! the file gives: the segments are capped, and rows take room as they're read.

use std::collections::HashMap;

use serde_json::{Map, json};

use super::Input;
use super::scene::{self, Animation, Channel, Interp, Node, Path, Scene};
use crate::Result;
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact, meters_in};

/// Segments a file may have: far past any capture rig.
const MAX_SEGMENTS: usize = 32_768;
/// Frames per second when the file doesn't say.
const DEFAULT_RATE: f64 = 30.0;

/// A segment and its rows.
struct Segment {
    name: String,
    parent: Option<String>,
    /// Translation, then angles about X, Y and Z.
    base: [f64; 6],
    bone_length: f64,
    /// The kept rows' frame numbers, and each row's translation and angles.
    frames: Vec<f64>,
    rows: Vec<[f64; 6]>,
}

/// The section being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Header,
    Hierarchy,
    Base,
    /// A segment's rows; `None` for a section naming no segment.
    Rows(Option<usize>),
    End,
}

/// Up to 40 characters of `s`, for an error message.
fn show(s: &str) -> String {
    s.chars().take(40).collect()
}

/// The length of a calibration unit in meters: `mm`, `cm`, `m`, `in`, `ft`, or a fraction of one (`1/10.000 m`).
fn unit_length(s: &str) -> Option<f64> {
    let s = s.trim();
    let (amount, unit) = if let Some((a, u)) = s.rsplit_once(char::is_whitespace) {
        (a.trim(), u)
    } else {
        let split = s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len());
        (s[..split].trim(), &s[split..])
    };
    let base = meters_in(unit)?;
    let factor = if amount.is_empty() {
        1.0
    } else if let Some((n, d)) = amount.split_once('/') {
        n.trim().parse::<f64>().ok()? / d.trim().parse::<f64>().ok()?
    } else {
        amount.parse::<f64>().ok()?
    };
    let m = factor * base;
    (m.is_finite() && m > 0.0).then_some(m)
}

/// The rotation that takes up axis `axis` (`Y`, `Z`, `-X`, …) to +Y; `None` if it isn't one.
fn up_rotation(axis: &str) -> Option<[f64; 4]> {
    use std::f64::consts::{FRAC_PI_2, PI};
    let a = axis.trim().to_ascii_uppercase();
    let q = match a.trim_start_matches('+') {
        "Y" => [0.0, 0.0, 0.0, 1.0],
        "-Y" => scene::about(0, PI),
        "Z" => scene::about(0, -FRAC_PI_2),
        "-Z" => scene::about(0, FRAC_PI_2),
        "X" => scene::about(2, FRAC_PI_2),
        "-X" => scene::about(2, -FRAC_PI_2),
        _ => return None,
    };
    Some(q)
}

/// The rotation of angles `[x, y, z]` (radians) in Euler order `order`: the first axis's rotation outermost.
fn euler(order: &[usize; 3], angles: [f64; 3]) -> [f64; 4] {
    let mut q = [0.0, 0.0, 0.0, 1.0];
    for &a in order {
        q = scene::quat_mul(q, scene::about(a, angles[a]));
    }
    q
}

/// Up to `n` finite numbers from `tokens`; fewer if one isn't.
fn numbers<'a>(tokens: impl Iterator<Item = &'a str>, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    for t in tokens.take(n) {
        match t.parse::<f64>() {
            Ok(v) if v.is_finite() => out.push(v),
            _ => break,
        }
    }
    out
}

/// Reads an HTR file.
pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    let bytes = input.read()?;
    let text = scene::text(&bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut header: HashMap<String, String> = HashMap::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut section: Option<Section> = None;
    let mut has_hierarchy = false;
    let (mut dropped, mut unknown_rows, mut scaled_rows) = (0usize, 0usize, 0usize);
    let mut add = |segments: &mut Vec<Segment>, name: &str, parent: Option<String>| -> Result<usize> {
        if let Some(&i) = index.get(name) {
            return Ok(i);
        }
        ensure!(
            segments.len() < MAX_SEGMENTS,
            "HTR has more than {MAX_SEGMENTS} segments"
        );
        index.insert(name.to_owned(), segments.len());
        segments.push(Segment {
            name: name.to_owned(),
            parent,
            base: [0.0; 6],
            bone_length: 0.0,
            frames: Vec::new(),
            rows: Vec::new(),
        });
        Ok(segments.len() - 1)
    };
    let mut lookup: HashMap<String, usize> = HashMap::new();
    for (k, raw) in text.lines().enumerate() {
        let no = k + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let name = name.trim();
            let next = match name.to_ascii_lowercase().as_str() {
                "header" => Section::Header,
                "segmentnames&hierarchy" => {
                    has_hierarchy = true;
                    Section::Hierarchy
                }
                "baseposition" => Section::Base,
                "endoffile" => Section::End,
                _ => {
                    ensure!(
                        section.is_some(),
                        "not an HTR file: it starts with `[{}]`, not [Header]",
                        show(name)
                    );
                    if lookup.len() != segments.len() {
                        lookup = segments.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
                    }
                    Section::Rows(lookup.get(name).copied())
                }
            };
            section = Some(next);
            if next == Section::End {
                break;
            }
            continue;
        }
        let Some(at) = section else {
            bail!("not an HTR file: it starts with `{}`, not [Header]", show(line));
        };
        let mut tokens = line.split_whitespace();
        let first = tokens.next().unwrap_or("");
        match at {
            Section::Header => {
                let value = line[first.len()..].trim();
                header.insert(first.to_ascii_lowercase(), value.to_owned());
            }
            Section::Hierarchy => {
                let parent = tokens
                    .next()
                    .filter(|p| !p.eq_ignore_ascii_case("GLOBAL") && *p != first);
                add(&mut segments, first, parent.map(str::to_owned))?;
            }
            Section::Base => {
                let v = numbers(tokens, 7);
                ensure!(
                    v.len() == 7,
                    "HTR line {no}: segment `{}`'s base position needs seven numbers",
                    show(first)
                );
                let i = add(&mut segments, first, None)?;
                segments[i].base.copy_from_slice(&v[..6]);
                segments[i].bone_length = v[6];
            }
            Section::Rows(None) => unknown_rows += 1,
            Section::Rows(Some(i)) => {
                let v = numbers(line.split_whitespace(), 8);
                let s = &mut segments[i];
                if v.len() < 7 || s.frames.last().is_some_and(|&f| v[0] <= f) {
                    dropped += 1;
                    continue;
                }
                if v.len() > 7 && (v[7] - 1.0).abs() > 1e-9 {
                    scaled_rows += 1;
                }
                s.frames.push(v[0]);
                s.rows.push([v[1], v[2], v[3], v[4], v[5], v[6]]);
            }
            Section::End => {}
        }
    }
    let Some(_) = section else {
        bail!("not an HTR file: it's empty");
    };
    ensure!(!segments.is_empty(), "HTR has no segments");

    let get = |k: &str| header.get(&k.to_ascii_lowercase()).map(String::as_str);
    let order_text = get("EulerRotationOrder").unwrap_or("ZYX");
    let order: Vec<usize> = order_text
        .bytes()
        .filter_map(|c| match c.to_ascii_uppercase() {
            b'X' => Some(0),
            b'Y' => Some(1),
            b'Z' => Some(2),
            _ => None,
        })
        .collect();
    let Ok(order) = <[usize; 3]>::try_from(order) else {
        bail!(
            "HTR EulerRotationOrder `{}` isn't three of X, Y and Z",
            show(order_text)
        );
    };
    ensure!(
        order_text.trim().len() == 3,
        "HTR EulerRotationOrder `{}` isn't three of X, Y and Z",
        show(order_text)
    );
    let radians = match get("RotationUnits").map(str::to_ascii_lowercase).as_deref() {
        None => false,
        Some(u) if u.starts_with("deg") => false,
        Some(u) if u.starts_with("rad") => true,
        Some(u) => bail!("HTR RotationUnits `{}` isn't Degrees or Radians", show(u)),
    };
    let angle = if radians { 1.0 } else { std::f64::consts::PI / 180.0 };
    let rate = get("DataFrameRate")
        .and_then(|r| r.parse::<f64>().ok())
        .filter(|r| r.is_finite() && *r > 0.0);
    let frame_time = 1.0 / rate.unwrap_or(DEFAULT_RATE);
    let units = get("CalibrationUnits");
    let meters = units.and_then(unit_length);
    let gravity = get("GlobalAxisofGravity").unwrap_or("Y");
    let up = up_rotation(gravity);
    let length_axis_text = get("BoneLengthAxis").unwrap_or("Y");
    let length_axis = up_rotation(length_axis_text).map(|_| {
        let a = length_axis_text.trim().to_ascii_uppercase();
        let sign = if a.starts_with('-') { -1.0 } else { 1.0 };
        let mut v = [0.0; 3];
        let k = match a.trim_start_matches(['+', '-']) {
            "X" => 0,
            "Z" => 2,
            _ => 1,
        };
        v[k] = sign;
        v
    });

    let mut scene = Scene::default();
    let base_rotation = |s: &Segment| euler(&order, [s.base[3] * angle, s.base[4] * angle, s.base[5] * angle]);
    let mut has_children = vec![false; segments.len()];
    let parents: Vec<Option<usize>> = segments
        .iter()
        .map(|s| s.parent.as_ref().and_then(|p| index.get(p).copied()))
        .collect();
    for p in parents.iter().flatten() {
        has_children[*p] = true;
    }
    let unknown_parents = segments
        .iter()
        .zip(&parents)
        .filter(|(s, p)| s.parent.is_some() && p.is_none())
        .count();
    for (s, &parent) in segments.iter().zip(&parents) {
        let mut node = Node::named(s.name.clone(), parent);
        node.translation = [s.base[0], s.base[1], s.base[2]].map(f32_of);
        node.rotation = scene::normalize(base_rotation(s));
        scene.add(node);
    }
    for (i, s) in segments.iter().enumerate() {
        if !has_children[i] {
            let mut end = Node::named(format!("{}_end", s.name), Some(i));
            let axis = length_axis.unwrap_or([0.0, 1.0, 0.0]);
            end.translation = axis.map(|c| f32_of(c * s.bone_length));
            scene.add(end);
        }
    }
    let first = segments
        .iter()
        .filter_map(|s| s.frames.first().copied())
        .fold(f64::INFINITY, f64::min);
    let mut channels = Vec::new();
    for (i, s) in segments.iter().enumerate() {
        if s.frames.is_empty() {
            continue;
        }
        let times: Vec<f32> = s.frames.iter().map(|&f| f32_of((f - first) * frame_time)).collect();
        let qb = base_rotation(s);
        let mut translations = Vec::with_capacity(s.rows.len() * 3);
        let mut rotations = Vec::with_capacity(s.rows.len() * 4);
        for r in &s.rows {
            translations.extend([s.base[0] + r[0], s.base[1] + r[1], s.base[2] + r[2]].map(f32_of));
            let qf = euler(&order, [r[3] * angle, r[4] * angle, r[5] * angle]);
            rotations.extend(scene::normalize(scene::quat_mul(qb, qf)));
        }
        channels.push(Channel {
            node: i,
            path: Path::Translation,
            times: times.clone(),
            values: translations,
            interp: Interp::Linear,
        });
        channels.push(Channel {
            node: i,
            path: Path::Rotation,
            times,
            values: rotations,
            interp: Interp::Linear,
        });
    }
    let frames = segments.iter().map(|s| s.frames.len()).max().unwrap_or(0);
    if keep.animations && !channels.is_empty() {
        scene.animations.push(Animation {
            name: "motion".into(),
            channels,
            extras: None,
        });
    }

    let info = &mut scene.info;
    info.format = Some("HTR".into());
    info.version = get("FileVersion").map(str::to_owned);
    info.unit = units.map(str::to_owned);
    info.meters_per_unit = meters.map_or(Fact::Unknown, Fact::Declared);
    info.up_axis = match (get("GlobalAxisofGravity"), Axis::parse(gravity)) {
        (Some(_), Some(a)) if up.is_some() => Fact::Declared(a),
        (None, _) => Fact::Assumed(Axis::PosY),
        _ => Fact::Unknown,
    };
    info.frame_rate = rate.map_or(Fact::Assumed(DEFAULT_RATE), Fact::Declared);
    let mut source = Map::new();
    for (key, name) in [
        ("DataType", "dataType"),
        ("RotationUnits", "rotationUnits"),
        ("EulerRotationOrder", "eulerRotationOrder"),
        ("BoneLengthAxis", "boneLengthAxis"),
    ] {
        if let Some(v) = get(key) {
            source.insert(name.into(), json!(v));
        }
    }
    for (key, name) in [("ScaleFactor", "scaleFactor"), ("NumFrames", "frames")] {
        if let Some(v) = get(key).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite()) {
            source.insert(name.into(), json!(v));
        }
    }
    let facts = &mut scene.facts;
    facts.insert("segments".into(), json!(segments.len()));
    facts.insert("frames".into(), json!(frames));
    if dropped > 0 {
        facts.insert("dropped_rows".into(), json!(dropped));
    }
    if unknown_rows > 0 {
        facts.insert("unknown_segment_rows".into(), json!(unknown_rows));
    }
    if scaled_rows > 0 {
        facts.insert("scale_factors_dropped".into(), json!(scaled_rows));
    }
    if unknown_parents > 0 {
        facts.insert("unknown_parents".into(), json!(unknown_parents));
    }
    if length_axis.is_none() {
        facts.insert("bone_length_axis_unknown".into(), json!(true));
    }
    if !has_hierarchy {
        facts.insert("global".into(), json!(true));
    }
    scene.source = source;
    scene.scale(f32_of(meters.unwrap_or(1.0)));
    // Y up needs no turn; every other axis's quaternion has w below 1.
    if let Some(q) = up.filter(|q| q[3] < 1.0) {
        scene.rotate_roots(scene::normalize(q));
    }
    scene.finish(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HTR: &str = "# a two-segment test
[Header]
FileType htr
DataType HTRS
FileVersion 1
NumSegments 2
NumFrames 2
DataFrameRate 10
EulerRotationOrder ZYX
CalibrationUnits mm
RotationUnits Degrees
GlobalAxisofGravity Y
BoneLengthAxis Y
ScaleFactor 1.0
[SegmentNames&Hierarchy]
hip GLOBAL
knee hip
[BasePosition]
hip 0 1000 0 0 0 0 400
knee 0 -400 0 0 0 90 300
[hip]
1 0 0 0 0 0 0 1
2 100 0 0 0 90 0 1
[knee]
1 0 0 0 0 0 0 1
2 0 0 0 90 0 0 1.5
[EndOfFile]
";

    fn read(text: &str) -> Result<Document<'static>> {
        load(Input::memory(text.as_bytes(), "t.htr"), Keep::ALL)
    }

    fn error(text: &str) -> String {
        match read(text) {
            Ok(_) => panic!("read without error"),
            Err(e) => e.to_string(),
        }
    }

    fn near(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn two_segments() {
        let doc = read(HTR).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        let names: Vec<_> = nodes.iter().map(|n| n["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["hip", "knee", "knee_end"]);
        assert_eq!(doc.facts["segments"], json!(2));
        assert_eq!(doc.facts["frames"], json!(2));
        assert_eq!(doc.facts["scale_factors_dropped"], json!(1));
        let rig = doc.rig().unwrap();
        assert!(near(&rig.nodes[0].rest.translation, &[0.0, 1.0, 0.0]));
        let s = std::f32::consts::FRAC_1_SQRT_2;
        assert!(near(&rig.nodes[1].rest.rotation, &[0.0, 0.0, s, s]));
        assert!(near(&rig.nodes[2].rest.translation, &[0.0, 0.3, 0.0]));
        let pose = rig.pose(Some(0), 0.1);
        assert!(near(&pose[0].translation, &[0.1, 1.0, 0.0]));
        assert!(near(&pose[0].rotation, &[0.0, s, 0.0, s]));
        // knee: the base's 90° about Z, then the row's 90° about X.
        let want = scene::normalize(scene::quat_mul(
            scene::about(2, std::f64::consts::FRAC_PI_2),
            scene::about(0, std::f64::consts::FRAC_PI_2),
        ));
        assert!(near(&pose[1].rotation, &want), "{:?}", pose[1].rotation);
        assert_eq!(rig.animations[0].channels[0].times, [0.0, 0.1]);
    }

    #[test]
    fn euler_order() {
        // ZYX: Rz·Ry·Rx, so X turns first. Rx(90) then Rz(90) takes Y to Z, then Z stays.
        let q = euler(
            &[2, 1, 0],
            [std::f64::consts::FRAC_PI_2, 0.0, std::f64::consts::FRAC_PI_2],
        );
        let v = scene::rotate(q, [0.0, 1.0, 0.0]);
        assert!(near(&v.map(f32_of), &[0.0, 0.0, 1.0]));
        let v = scene::rotate(q, [1.0, 0.0, 0.0]);
        assert!(near(&v.map(f32_of), &[0.0, 1.0, 0.0]));
    }

    #[test]
    fn z_up_and_units() {
        let text = HTR
            .replace("GlobalAxisofGravity Y", "GlobalAxisofGravity Z")
            .replace("CalibrationUnits mm", "CalibrationUnits 1/10.000 m");
        let doc = read(&text).unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("HTR"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosZ));
        assert_eq!(info.meters_per_unit, Fact::Declared(0.1));
        assert_eq!(info.unit.as_deref(), Some("1/10.000 m"));
        assert_eq!(info.frame_rate, Fact::Declared(10.0));
        assert!(info.converted.units && info.converted.up_axis);
        let rig = doc.rig().unwrap();
        // 1000 tenths of a meter up Z is 100 m up Y.
        assert!(near(&rig.nodes[0].rest.translation, &[0.0, 0.0, -100.0]));
        assert_eq!(unit_length("mm"), Some(0.001));
        assert_eq!(unit_length("1/10.000 m"), Some(0.1));
        assert!((unit_length("2.54cm").unwrap() - 0.0254).abs() < 1e-12);
        assert_eq!(unit_length("furlongs"), None);
        assert_eq!(unit_length("1/0 m"), None);
    }

    #[test]
    fn bad_rows_are_dropped() {
        let text = HTR
            .replace(
                "2 0 0 0 90 0 0 1.5",
                "2 0 0 x\n2 0 0 0 0 0 0\n1 0 0 0 0 0 0\n3 0 0 0 0 0 0",
            )
            .replace("[EndOfFile]", "[nobody]\n1 2 3\n[EndOfFile]\njunk");
        let doc = read(&text).unwrap();
        assert_eq!(doc.facts["dropped_rows"], json!(2));
        assert_eq!(doc.facts["unknown_segment_rows"], json!(1));
        let rig = doc.rig().unwrap();
        assert_eq!(rig.animations[0].channels[2].times, [0.0, 0.1, 0.2]);
    }

    #[test]
    fn global_form() {
        let text =
            "[Header]\nDataFrameRate 0\n[BasePosition]\na 0 0 0 0 0 0 1\nb 1 0 0 0 0 0 1\n[a]\n0 0 0 0 0 0 0 1\n";
        let doc = read(text).unwrap();
        assert_eq!(doc.facts["global"], json!(true));
        let info = doc.source_info();
        assert_eq!(info.frame_rate, Fact::Assumed(30.0));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(doc.json["scenes"][0]["nodes"], json!([0, 1]));
    }

    #[test]
    fn bad_files_are_errors() {
        let cases = [
            ("", "empty"),
            ("hello\n", "not an HTR file"),
            ("[hip]\n", "not an HTR file"),
            ("[Header]\n", "no segments"),
            ("[Header]\n[BasePosition]\na 0 0\n", "seven numbers"),
            (
                "[Header]\nEulerRotationOrder ZY\n[BasePosition]\na 0 0 0 0 0 0 1\n",
                "EulerRotationOrder",
            ),
            (
                "[Header]\nEulerRotationOrder ZYXQ\n[BasePosition]\na 0 0 0 0 0 0 1\n",
                "EulerRotationOrder",
            ),
            (
                "[Header]\nRotationUnits grads\n[BasePosition]\na 0 0 0 0 0 0 1\n",
                "RotationUnits",
            ),
        ];
        for (text, want) in cases {
            let e = error(text);
            assert!(e.contains(want), "{text:?}: {e}");
        }
        let mut many = String::new();
        for i in 0..=MAX_SEGMENTS {
            many.push('s');
            many.push_str(&i.to_string());
            many.push_str(" GLOBAL\n");
        }
        assert!(error(&format!("[Header]\n[SegmentNames&Hierarchy]\n{many}")).contains("more than"));
    }

    #[test]
    fn loops_and_unknown_parents() {
        let text = HTR
            .replace("hip GLOBAL", "hip knee")
            .replace("knee hip", "knee hip\nfoot ankle");
        let doc = read(&text).unwrap();
        assert_eq!(doc.facts["unknown_parents"], json!(1));
        assert!(doc.rig().is_ok());
    }

    #[test]
    fn truncated_files_never_panic() {
        for cut in 0..HTR.len() {
            let _ = read(&HTR[..cut]);
        }
    }
}
