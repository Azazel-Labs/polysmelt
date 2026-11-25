//! The file's axis system, unit and frame rate (`GlobalSettings`) and what its header says of it, as the root
//! node's transform and `asset.extras.source`.

use serde_json::{Map, Value, json};

use super::{Props, child_attr, choice, num, scalar, string};
use crate::formats::fbx_tree::{NodeHandle, Version};
use crate::formats::xform::{M4, det3, scale};
use crate::source::{Axis, Fact, Handedness, SourceInfo};

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

/// The file's axis system and unit, mapped to glTF (X right, Y up, Z toward the viewer, meters) by a root node: the
/// node's transform and the meters per file unit, with what `asset.extras.source` records of the file.
pub(super) fn read(
    rootnode: NodeHandle<'_>,
    version: Version,
    creator: &str,
) -> (M4, f64, SourceInfo, Map<String, Value>) {
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
    scene_info(rootnode, creator, &mut info, &mut details);
    (basis, unit, info, details)
}
