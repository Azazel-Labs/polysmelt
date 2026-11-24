//! What a model's file says about itself, the same way whatever the format: its units, which way
//! is up, its frame rate, who made it and with what, and when.
//!
//! Every reader records this in `asset.extras.source`, under the keys listed here, so it survives
//! being written out as glTF and read back. [`Document::source_info`](crate::document::Document::source_info)
//! reads it as a [`SourceInfo`]. Each value says whether the file declared it ([`Fact::Declared`]),
//! the reader took the format's convention or a guess ([`Fact::Assumed`]), or nothing is known
//! ([`Fact::Unknown`]): an STL's units are unknown, not millimeters.
//!
//! The values describe the source file as it was written. What the reader did about them is in
//! [`SourceInfo::converted`]: a COLLADA file in centimeters, Z up, is read into glTF's meters, Y up,
//! so its `metersPerUnit` is `0.01` and its `upAxis` `+Z`, and `converted` says both were
//! converted. [`SourceInfo::output_meters_per_unit`] answers the question most users have: how long
//! is a unit of the geometry as it is now.
//!
//! The keys in `asset.extras.source`:
//!
//! | key | value |
//! |---|---|
//! | `format`, `version` | the format's name (`"FBX"`) and the file's version of it, as strings |
//! | `metersPerUnit`, `unit` | the length of the file's unit in meters, and its name (`"centimeter"`) |
//! | `upAxis`, `forwardAxis` | signed axes, `"+Y"`, `"-Z"` |
//! | `handedness` | `"right"` or `"left"` |
//! | `frameRate` | frames (or keys, or ticks) per second |
//! | `assumed` | the keys above that are a convention or a guess, not declared by the file |
//! | `converted` | which of `units`, `upAxis`, `handedness` the reader converted to glTF's |
//! | `tool`, `author`, `title`, `description`, `copyright`, `license`, `created`, `modified` | strings; dates in ISO 8601 when the file's can be read as dates, else as written |
//! | `keywords`, `comments` | arrays of strings |
//! | `georeference` | where a scan or survey sits on Earth: `wkt`, `epsg`, `verticalEpsg`, `origin` and the like |
//!
//! Every other key is the format's own detail, kept as the reader found it.

use serde_json::{Map, Value, json};

/// The keys [`SourceInfo`] reads and writes: these stay in `asset.extras.source` when
/// [`Keep::extras`](crate::document::Keep) is off, and the format's own details go.
pub const KEYS: [&str; 21] = [
    "format",
    "version",
    "metersPerUnit",
    "unit",
    "upAxis",
    "forwardAxis",
    "handedness",
    "frameRate",
    "assumed",
    "converted",
    "tool",
    "author",
    "title",
    "description",
    "copyright",
    "license",
    "created",
    "modified",
    "keywords",
    "comments",
    "georeference",
];

/// A value as the file gave it: declared, assumed (the format's convention, or a guess), or unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Fact<T> {
    /// The file says so.
    Declared(T),
    /// The file doesn't say; this is the format's convention or the reader's guess.
    Assumed(T),
    /// Nothing is known.
    #[default]
    Unknown,
}

impl<T> Fact<T> {
    /// The value, declared or assumed.
    #[must_use]
    pub fn value(&self) -> Option<&T> {
        match self {
            Fact::Declared(v) | Fact::Assumed(v) => Some(v),
            Fact::Unknown => None,
        }
    }

    /// The value, only when the file declared it.
    #[must_use]
    pub fn declared(&self) -> Option<&T> {
        match self {
            Fact::Declared(v) => Some(v),
            _ => None,
        }
    }

    /// Whether anything is known, declared or assumed.
    #[must_use]
    pub fn is_known(&self) -> bool {
        !matches!(self, Fact::Unknown)
    }

    /// The value mapped, keeping whether it was declared.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Fact<U> {
        match self {
            Fact::Declared(v) => Fact::Declared(f(v)),
            Fact::Assumed(v) => Fact::Assumed(f(v)),
            Fact::Unknown => Fact::Unknown,
        }
    }
}

/// A signed axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// `+X`
    PosX,
    /// `-X`
    NegX,
    /// `+Y`
    PosY,
    /// `-Y`
    NegY,
    /// `+Z`
    PosZ,
    /// `-Z`
    NegZ,
}

impl Axis {
    /// As written in `asset.extras.source`: `"+Y"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Axis::PosX => "+X",
            Axis::NegX => "-X",
            Axis::PosY => "+Y",
            Axis::NegY => "-Y",
            Axis::PosZ => "+Z",
            Axis::NegZ => "-Z",
        }
    }

    /// `"+Y"`, `"-z"` or a bare `"Y"` (positive).
    #[must_use]
    pub fn parse(s: &str) -> Option<Axis> {
        let s = s.trim();
        let (neg, letter) = match s.as_bytes() {
            [b'+', c] | [c] => (false, *c),
            [b'-', c] => (true, *c),
            _ => return None,
        };
        Some(match (letter.to_ascii_uppercase(), neg) {
            (b'X', false) => Axis::PosX,
            (b'X', true) => Axis::NegX,
            (b'Y', false) => Axis::PosY,
            (b'Y', true) => Axis::NegY,
            (b'Z', false) => Axis::PosZ,
            (b'Z', true) => Axis::NegZ,
            _ => return None,
        })
    }

    /// The unit vector along it.
    #[must_use]
    pub fn vector(self) -> [f32; 3] {
        match self {
            Axis::PosX => [1.0, 0.0, 0.0],
            Axis::NegX => [-1.0, 0.0, 0.0],
            Axis::PosY => [0.0, 1.0, 0.0],
            Axis::NegY => [0.0, -1.0, 0.0],
            Axis::PosZ => [0.0, 0.0, 1.0],
            Axis::NegZ => [0.0, 0.0, -1.0],
        }
    }
}

/// Which way a coordinate system turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handedness {
    /// glTF's.
    Right,
    /// Direct3D's and Unreal's, and LWO files'.
    Left,
}

/// What a reader converted to glTF's conventions (meters, +Y up, right-handed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Converted {
    /// Lengths were scaled from the file's unit to meters.
    pub units: bool,
    /// The scene was turned so that the file's up is +Y.
    pub up_axis: bool,
    /// The scene was mirrored from left-handed to right-handed.
    pub handedness: bool,
}

/// What a model's file says about itself (see the [module](self) docs).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceInfo {
    /// The format's name: `"FBX"`, `"COLLADA"`, `"Wavefront OBJ"`.
    pub format: Option<String>,
    /// The file's version of its format, as it gives it.
    pub version: Option<String>,
    /// The length of the file's unit, in meters.
    pub meters_per_unit: Fact<f64>,
    /// The unit's name, as the file or its format calls it (`"millimeter"`, `"inch"`).
    pub unit: Option<String>,
    /// The file's up axis.
    pub up_axis: Fact<Axis>,
    /// The file's forward axis, where it has one.
    pub forward_axis: Fact<Axis>,
    /// The file's handedness.
    pub handedness: Fact<Handedness>,
    /// Frames (or keys, or ticks) per second.
    pub frame_rate: Fact<f64>,
    /// What the reader converted to glTF's conventions.
    pub converted: Converted,
    /// The program that wrote the file.
    pub tool: Option<String>,
    /// Who made it.
    pub author: Option<String>,
    /// Its title or name.
    pub title: Option<String>,
    /// A description or subject.
    pub description: Option<String>,
    /// Its copyright notice.
    pub copyright: Option<String>,
    /// Its license terms.
    pub license: Option<String>,
    /// When it was made: ISO 8601 when the file's date can be read as one, else as written.
    pub created: Option<String>,
    /// When it was last changed, likewise.
    pub modified: Option<String>,
    /// Keywords or tags.
    pub keywords: Vec<String>,
    /// Comments the file holds, in order.
    pub comments: Vec<String>,
    /// Where a scan or survey sits on Earth.
    pub georeference: Option<Value>,
}

impl SourceInfo {
    /// Reads the keys of `asset.extras.source` it knows; the rest are ignored.
    #[must_use]
    pub fn from_json(source: &Map<String, Value>) -> SourceInfo {
        let string = |k: &str| match source.get(k) {
            Some(Value::String(s)) => Some(s.clone()),
            Some(v @ Value::Number(_)) => Some(v.to_string()),
            _ => None,
        };
        let strings = |k: &str| -> Vec<String> {
            match source.get(k) {
                Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
                Some(Value::String(s)) => vec![s.clone()],
                _ => Vec::new(),
            }
        };
        let assumed = strings("assumed");
        let converted = strings("converted");
        let fact = |k: &str, v: Option<f64>| match v {
            Some(v) if assumed.iter().any(|a| a == k) => Fact::Assumed(v),
            Some(v) => Fact::Declared(v),
            None => Fact::Unknown,
        };
        let axis = |k: &str| match source.get(k).and_then(Value::as_str).and_then(Axis::parse) {
            Some(a) if assumed.iter().any(|x| x == k) => Fact::Assumed(a),
            Some(a) => Fact::Declared(a),
            None => Fact::Unknown,
        };
        let handed = match source.get("handedness").and_then(Value::as_str) {
            Some("right") => Some(Handedness::Right),
            Some("left") => Some(Handedness::Left),
            _ => None,
        };
        SourceInfo {
            format: string("format"),
            version: string("version"),
            meters_per_unit: fact("metersPerUnit", source.get("metersPerUnit").and_then(Value::as_f64)),
            unit: string("unit"),
            up_axis: axis("upAxis"),
            forward_axis: axis("forwardAxis"),
            handedness: match handed {
                Some(h) if assumed.iter().any(|x| x == "handedness") => Fact::Assumed(h),
                Some(h) => Fact::Declared(h),
                None => Fact::Unknown,
            },
            frame_rate: fact("frameRate", source.get("frameRate").and_then(Value::as_f64)),
            converted: Converted {
                units: converted.iter().any(|c| c == "units"),
                up_axis: converted.iter().any(|c| c == "upAxis"),
                handedness: converted.iter().any(|c| c == "handedness"),
            },
            tool: string("tool"),
            author: string("author"),
            title: string("title"),
            description: string("description"),
            copyright: string("copyright"),
            license: string("license"),
            created: string("created"),
            modified: string("modified"),
            keywords: strings("keywords"),
            comments: strings("comments"),
            georeference: source.get("georeference").cloned(),
        }
    }

    /// Writes what's known into `source` (as `asset.extras.source`), over what those keys held, leaving the
    /// rest: the format's own keys, and any of these that are unknown here.
    pub fn write(&self, source: &mut Map<String, Value>) {
        let mut assumed = Vec::new();
        let mut put = |k: &str, v: Option<Value>| {
            if let Some(v) = v {
                source.insert(k.to_owned(), v);
            }
        };
        let mut fact = |k: &'static str, f: Option<(Value, bool)>, put: &mut dyn FnMut(&str, Option<Value>)| {
            if let Some((v, declared)) = f {
                if !declared {
                    assumed.push(k);
                }
                put(k, Some(v));
            }
        };
        let num = |f: Fact<f64>| match f {
            Fact::Declared(v) => Some((json!(v), true)),
            Fact::Assumed(v) => Some((json!(v), false)),
            Fact::Unknown => None,
        };
        let axis = |f: Fact<Axis>| match f {
            Fact::Declared(a) => Some((json!(a.as_str()), true)),
            Fact::Assumed(a) => Some((json!(a.as_str()), false)),
            Fact::Unknown => None,
        };
        let s = |v: &Option<String>| v.as_ref().map(|s| json!(s));
        let list = |v: &Vec<String>| (!v.is_empty()).then(|| json!(v));
        put("format", s(&self.format));
        put("version", s(&self.version));
        fact("metersPerUnit", num(self.meters_per_unit), &mut put);
        put("unit", s(&self.unit));
        fact("upAxis", axis(self.up_axis), &mut put);
        fact("forwardAxis", axis(self.forward_axis), &mut put);
        let h = |h: Handedness| json!(if h == Handedness::Right { "right" } else { "left" });
        fact(
            "handedness",
            match self.handedness {
                Fact::Declared(x) => Some((h(x), true)),
                Fact::Assumed(x) => Some((h(x), false)),
                Fact::Unknown => None,
            },
            &mut put,
        );
        fact("frameRate", num(self.frame_rate), &mut put);
        if !assumed.is_empty() {
            put("assumed", Some(json!(assumed)));
        }
        let c = self.converted;
        let converted: Vec<&str> = [(c.units, "units"), (c.up_axis, "upAxis"), (c.handedness, "handedness")]
            .into_iter()
            .filter_map(|(on, k)| on.then_some(k))
            .collect();
        if !converted.is_empty() {
            put("converted", Some(json!(converted)));
        }
        put("tool", s(&self.tool));
        put("author", s(&self.author));
        put("title", s(&self.title));
        put("description", s(&self.description));
        put("copyright", s(&self.copyright));
        put("license", s(&self.license));
        put("created", s(&self.created));
        put("modified", s(&self.modified));
        put("keywords", list(&self.keywords));
        put("comments", list(&self.comments));
        put("georeference", self.georeference.clone());
    }

    /// The length of a unit of the geometry as read, in meters: 1 when the reader converted the
    /// file's units, else the file's own unit (known, assumed or not).
    #[must_use]
    pub fn output_meters_per_unit(&self) -> Fact<f64> {
        if self.converted.units {
            Fact::Declared(1.0)
        } else {
            self.meters_per_unit
        }
    }
}

/// The length in meters of a unit by its name, in the spellings formats use (`"mm"`, `"millimeter"`,
/// `"millimetres"`, `"in"`, `"feet"`, `"us survey foot"`...). `None` for a name it doesn't know.
#[allow(dead_code, reason = "readers that name their units use it")]
pub(crate) fn meters_in(name: &str) -> Option<f64> {
    let n = name.trim().to_ascii_lowercase().replace(['_', '-'], " ");
    let n = n.trim_end_matches('.');
    Some(match n {
        "m" | "meter" | "meters" | "metre" | "metres" => 1.0,
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => 0.001,
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => 0.01,
        "dm" | "decimeter" | "decimeters" | "decimetre" | "decimetres" => 0.1,
        "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => 1000.0,
        "um" | "µm" | "micron" | "microns" | "micrometer" | "micrometers" | "micrometre" | "micrometres" => 1e-6,
        "nm" | "nanometer" | "nanometers" | "nanometre" | "nanometres" => 1e-9,
        "in" | "inch" | "inches" => 0.0254,
        "ft" | "foot" | "feet" => 0.3048,
        "us survey foot" | "us survey feet" | "survey foot" | "survey feet" | "us ft" | "ftus" => 1200.0 / 3937.0,
        "yd" | "yard" | "yards" => 0.9144,
        "mi" | "mile" | "miles" => 1609.344,
        "mil" | "mils" | "thou" => 0.000_025_4,
        "angstrom" | "angstroms" => 1e-10,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_written_reads_back_the_same() {
        let info = SourceInfo {
            format: Some("COLLADA".into()),
            version: Some("1.4.1".into()),
            meters_per_unit: Fact::Declared(0.01),
            unit: Some("centimeter".into()),
            up_axis: Fact::Declared(Axis::PosZ),
            forward_axis: Fact::Unknown,
            handedness: Fact::Assumed(Handedness::Right),
            frame_rate: Fact::Assumed(30.0),
            converted: Converted {
                units: true,
                up_axis: true,
                handedness: false,
            },
            tool: Some("Blender 4.2".into()),
            author: Some("someone".into()),
            created: Some("2024-01-02T03:04:05Z".into()),
            keywords: vec!["a".into(), "b".into()],
            ..SourceInfo::default()
        };
        let mut map = Map::new();
        map.insert("detail".into(), json!(7));
        info.write(&mut map);
        assert_eq!(map["upAxis"], "+Z");
        assert_eq!(map["assumed"], json!(["handedness", "frameRate"]));
        assert_eq!(map["converted"], json!(["units", "upAxis"]));
        assert_eq!(map["detail"], 7, "the format's own keys stay");
        assert_eq!(SourceInfo::from_json(&map), info);
        assert_eq!(info.output_meters_per_unit(), Fact::Declared(1.0));
    }

    #[test]
    fn nothing_known_writes_nothing() {
        let mut map = Map::new();
        SourceInfo::default().write(&mut map);
        assert!(map.is_empty());
        let info = SourceInfo::from_json(&map);
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.output_meters_per_unit(), Fact::Unknown);
    }

    #[test]
    fn axes_read_signed_or_bare() {
        assert_eq!(Axis::parse("+Y"), Some(Axis::PosY));
        assert_eq!(Axis::parse("-z"), Some(Axis::NegZ));
        assert_eq!(Axis::parse("X"), Some(Axis::PosX));
        assert_eq!(Axis::parse("Z_UP"), None);
    }

    #[test]
    fn unit_names_in_their_usual_spellings() {
        assert_eq!(meters_in("Millimetres"), Some(0.001));
        assert_eq!(meters_in("inch"), Some(0.0254));
        assert_eq!(meters_in("US_Survey_Foot"), Some(1200.0 / 3937.0));
        assert_eq!(meters_in("parsec"), None);
    }
}
