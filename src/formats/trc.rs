//! TRC (track row column) marker motion as a glTF document: each marker a node, its track a translation animation.
//!
//! A TRC file is tab-separated text. Its first line starts with `PathFileType`. The second names the header's
//! fields and the third gives their values: `DataRate` (frames per second), `NumFrames`, `NumMarkers`, `Units` and
//! others. The fourth starts `Frame#` and `Time`, then names each marker over its three columns; the fifth labels the
//! columns `X1 Y1 Z1 X2 …`. Each line after that is a frame: its number, its time in seconds, then X, Y and Z for each
//! marker, an empty cell (or `NaN`) where the marker wasn't seen. A file with no tabs is split on spaces instead,
//! and then a short row leaves its last markers unseen.
//!
//! The document has a root node named `markers`, and a node per marker under it, named from the fourth line, or
//! `marker<n>` where it names none. A marker's rest position is its first sample. Its track is a linear translation
//! channel with a key at each frame where it was seen, the first frame at 0 seconds; glTF has no way to leave a gap,
//! so a missing stretch is filled in by the keys on each side, and the missing samples are counted. A frame's time
//! is its `Time` cell, or `(Frame# − 1) / DataRate` without one; a row with neither, or whose time doesn't follow the
//! row before it, is left out and counted. Lengths become meters from `Units` (millimeters when it's absent).
//! TRC files don't say which way is up; they're taken to be Y-up, as biomechanics tools write them, and that's
//! recorded. Nothing is allocated from a count the file gives: markers come from the names and columns there are,
//! and frames from the rows there are.

use serde_json::{Map, Value, json};

use super::Input;
use super::scene::{self, Animation, Channel, Interp, Node, Path, Scene};
use crate::Result;
use crate::document::{Document, Keep};
use crate::num::f32_of;

/// Frames per second when the header doesn't give one.
const DEFAULT_RATE: f64 = 60.0;

/// Up to 40 characters of `s`, for an error message.
fn show(s: &str) -> String {
    s.chars().take(40).collect()
}

/// A line's cells: split on tabs when there are any, else on runs of spaces.
fn cells(line: &str, tabs: bool) -> Vec<&str> {
    if tabs {
        line.split('\t').map(str::trim).collect()
    } else {
        line.split_whitespace().collect()
    }
}

/// The length of a unit name in meters.
fn unit_length(s: &str) -> Option<f64> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => 0.001,
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => 0.01,
        "dm" => 0.1,
        "m" | "meter" | "meters" | "metre" | "metres" => 1.0,
        "in" | "inch" | "inches" => 0.0254,
        "ft" | "foot" | "feet" => 0.3048,
        _ => return None,
    })
}

/// A cell as a finite number.
fn number(cell: Option<&&str>) -> Option<f64> {
    cell?.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Reads a TRC file.
pub(crate) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    let bytes = input.read()?;
    let text = scene::text(&bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let tabs = text.contains('\t');
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    ensure!(
        first.split_whitespace().next() == Some("PathFileType"),
        "not a TRC file: it starts with `{}`, not PathFileType",
        show(first)
    );
    let (Some(keys), Some(values)) = (lines.next(), lines.next()) else {
        bail!("TRC file is cut short: its header is missing");
    };
    let header: Vec<(String, String)> = cells(keys, tabs)
        .into_iter()
        .zip(cells(values, tabs))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.to_owned()))
        .collect();
    let get = |k: &str| header.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
    let rate_given = get("datarate")
        .and_then(|r| r.parse::<f64>().ok())
        .filter(|r| r.is_finite() && *r > 0.0);
    let rate = rate_given.unwrap_or(DEFAULT_RATE);
    let units = get("units").filter(|u| !u.is_empty());
    let meters = units.map_or(Some(0.001), unit_length);

    // The marker names: every non-empty cell after Frame# and Time, each over three columns.
    let names_line = lines.next().unwrap_or("");
    let row = cells(names_line, tabs);
    ensure!(
        row.first().is_some_and(|c| c.eq_ignore_ascii_case("Frame#")),
        "TRC file's fourth line starts with `{}`, not Frame#",
        show(names_line)
    );
    let names: Vec<&str> = if tabs {
        // Each name is in the first of its three columns; the cells between are empty.
        row.iter().skip(2).step_by(3).copied().collect()
    } else {
        row.iter().skip(2).copied().collect()
    };
    let coords = lines
        .next()
        .map_or(0, |l| cells(l, tabs).iter().filter(|c| !c.is_empty()).count());
    let mut markers = names.len().max(coords / 3);
    while markers > 0 && names.get(markers - 1).is_none_or(|n| n.is_empty()) && markers * 3 > coords {
        markers -= 1;
    }

    let mut tracks: Vec<(Vec<f64>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); markers];
    let (mut frames, mut dropped, mut missing) = (0usize, 0usize, 0usize);
    let mut last: Option<f64> = None;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let row = cells(line, tabs);
        let time = number(row.get(1)).or_else(|| number(row.first()).map(|f| (f - 1.0) / rate));
        let Some(time) = time.filter(|t| last.is_none_or(|l| *t > l)) else {
            dropped += 1;
            continue;
        };
        last = Some(time);
        frames += 1;
        for (m, track) in tracks.iter_mut().enumerate() {
            let at = 2 + m * 3;
            match (number(row.get(at)), number(row.get(at + 1)), number(row.get(at + 2))) {
                (Some(x), Some(y), Some(z)) => {
                    track.0.push(time);
                    track.1.extend([x, y, z].map(f32_of));
                }
                _ => missing += 1,
            }
        }
    }
    let start = tracks
        .iter()
        .filter_map(|t| t.0.first().copied())
        .fold(f64::INFINITY, f64::min);

    let mut scene = Scene::default();
    let root = scene.add(Node::named("markers", None));
    let mut channels = Vec::new();
    let mut empty = 0usize;
    for (m, (times, values)) in tracks.into_iter().enumerate() {
        let name = match names.get(m) {
            Some(n) if !n.is_empty() => (*n).to_owned(),
            _ => format!("marker{}", m + 1),
        };
        let mut node = Node::named(name, Some(root));
        match values.get(..3) {
            Some(v) => node.translation = [v[0], v[1], v[2]],
            None => empty += 1,
        }
        let i = scene.add(node);
        if !times.is_empty() {
            channels.push(Channel {
                node: i,
                path: Path::Translation,
                times: times.iter().map(|t| f32_of(t - start)).collect(),
                values,
                interp: Interp::Linear,
            });
        }
    }
    if keep.animations && !channels.is_empty() {
        scene.animations.push(Animation {
            name: "motion".into(),
            channels,
            extras: None,
        });
    }

    let mut source = Map::new();
    source.insert("format".into(), json!("TRC"));
    source.insert("upAxis".into(), json!("Y"));
    source.insert("units".into(), json!(units.unwrap_or("mm")));
    if let Some(kind) = first.split('\t').nth(1).or_else(|| first.split_whitespace().nth(1)) {
        source.insert("pathFileType".into(), json!(kind.trim()));
    }
    for (key, name) in [
        ("cameraRate", "camerarate"),
        ("origDataRate", "origdatarate"),
        ("origDataStartFrame", "origdatastartframe"),
        ("origNumFrames", "orignumframes"),
    ] {
        if let Some(v) = get(name).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite()) {
            source.insert(key.into(), json!(v));
        }
    }
    source.insert("frameRate".into(), json!(rate));
    if start.is_finite() {
        source.insert("startTime".into(), json!(start));
    }
    let facts = &mut scene.facts;
    facts.insert("markers".into(), json!(markers));
    facts.insert("frames".into(), json!(frames));
    facts.insert("frame_rate".into(), json!(rate));
    facts.insert("frame_time".into(), json!(1.0 / rate));
    if rate_given.is_none() {
        facts.insert("frame_time_guessed".into(), json!(true));
    }
    if missing > 0 {
        facts.insert("missing_samples".into(), json!(missing));
    }
    if empty > 0 {
        facts.insert("markers_never_seen".into(), json!(empty));
    }
    if dropped > 0 {
        facts.insert("dropped_rows".into(), json!(dropped));
    }
    if meters.is_none() {
        facts.insert("units_unknown".into(), Value::Bool(true));
    }
    facts.insert("up_axis_assumed".into(), json!(true));
    scene.source = source;
    scene.scale(f32_of(meters.unwrap_or(1.0)));
    scene.finish(keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRC: &str = "PathFileType\t4\t(X/Y/Z)\twalk.trc
DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames
10.00\t10.00\t3\t2\tmm\t10.00\t1\t3
Frame#\tTime\tL.Knee\t\t\tR Knee\t\t\t
\t\tX1\tY1\tZ1\tX2\tY2\tZ2

1\t0.500\t100\t200\t300\t\t\t
2\t0.600\t110\t200\t300\t0\t500\t0
3\t0.700\t120\t210\t290\t0\t510\tNaN
";

    fn read(text: &str) -> Result<Document<'static>> {
        load(Input::memory(text.as_bytes(), "t.trc"), Keep::ALL)
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
    fn markers() {
        let doc = read(TRC).unwrap();
        assert_eq!(doc.facts["markers"], json!(2));
        assert_eq!(doc.facts["frames"], json!(3));
        assert_eq!(doc.facts["missing_samples"], json!(2));
        let rig = doc.rig().unwrap();
        assert_eq!(rig.nodes[1].name.as_deref(), Some("L.Knee"));
        assert_eq!(rig.nodes[2].name.as_deref(), Some("R Knee"));
        assert!(near(&rig.nodes[1].rest.translation, &[0.1, 0.2, 0.3]));
        assert!(near(&rig.nodes[2].rest.translation, &[0.0, 0.5, 0.0]));
        assert!(near(&rig.animations[0].channels[0].times, &[0.0, 0.1, 0.2]));
        assert!(near(&rig.animations[0].channels[1].times, &[0.1]));
        let pose = rig.pose(Some(0), 0.15);
        assert!(near(&pose[1].translation, &[0.115, 0.205, 0.295]));
        assert_eq!(doc.json["asset"]["extras"]["source"]["startTime"], json!(0.5));
    }

    #[test]
    fn spaces_units_and_frame_numbers() {
        let text = "PathFileType 4 (X/Y/Z) x.trc\nDataRate NumFrames Units\n4 2 cm\nFrame# Time a b\nX1 Y1 Z1 X2 Y2 Z2\n\
                    1 x 1 2 3 4 5 6\n3 x 1 2 3\n2 x 0 0 0 0 0 0\n";
        let doc = read(text).unwrap();
        assert_eq!(doc.facts["dropped_rows"], json!(1));
        assert_eq!(doc.facts["missing_samples"], json!(1));
        let rig = doc.rig().unwrap();
        assert!(near(&rig.nodes[2].rest.translation, &[0.04, 0.05, 0.06]));
        assert!(near(&rig.animations[0].channels[0].times, &[0.0, 0.5]));
    }

    #[test]
    fn unnamed_columns_and_odd_units() {
        let text = "PathFileType\t4\nDataRate\tUnits\tNumMarkers\n0\tfurlongs\t99\nFrame#\tTime\ta\n\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n1\t0\t1\t2\t3\t4\t5\t6\n";
        let doc = read(text).unwrap();
        assert_eq!(doc.facts["markers"], json!(2));
        assert_eq!(doc.facts["units_unknown"], json!(true));
        assert_eq!(doc.facts["frame_time_guessed"], json!(true));
        let rig = doc.rig().unwrap();
        assert_eq!(rig.nodes[2].name.as_deref(), Some("marker2"));
    }

    #[test]
    fn bad_files_are_errors() {
        for (text, want) in [
            ("", "not a TRC file"),
            ("Frame#\tTime\n", "not a TRC file"),
            ("PathFileType\t4\nDataRate\n", "cut short"),
            ("PathFileType\t4\nDataRate\n60\n1\t0\t1\t2\t3\n", "Frame#"),
        ] {
            let e = error(text);
            assert!(e.contains(want), "{text:?}: {e}");
        }
    }

    #[test]
    fn no_frames_is_no_animation() {
        let doc = read(&TRC[..TRC.find("\n1\t").unwrap()]).unwrap();
        assert_eq!(doc.facts["frames"], json!(0));
        assert_eq!(doc.facts["markers_never_seen"], json!(2));
        assert!(doc.json.get("animations").is_none());
    }

    #[test]
    fn truncated_files_never_panic() {
        for cut in 0..TRC.len() {
            let _ = read(&TRC[..cut]);
        }
    }
}
