//! DXF drawings, as text or binary: their faces, meshes, lines and points.
//!
//! Read are `3DFACE`s; `POLYLINE` polyface meshes (flag 64: position vertices, then face
//! records of up to four corners) and polygon meshes (flag 16: M by N vertices, closed either
//! way, or the surface fitted to them when there is one); `MESH` at its control mesh (level 0);
//! and `INSERT`s of blocks, nested, by insertion point, scale, rotation, extrusion (the
//! arbitrary axis algorithm) and column and row arrays, a block's mesh made once.
//!
//! Lines and points are read too: `LINE`, `POINT`, `LWPOLYLINE` and the 2D and 3D `POLYLINE`s,
//! open or closed, a 2D one in its object coordinate system at its elevation. A bulge (an arc
//! between two vertices) is drawn as straight segments, one for each sixteenth of a half turn
//! it sweeps; past [`MAX_ARC_POINTS`] in a drawing, arcs are drawn as their chords. A spline
//! fitted polyline is drawn through its fitted vertices, not its frame. Widths and thickness
//! are left out.
//!
//! Curves are drawn as lines too: `CIRCLE` and `ARC` in their object coordinate system, `ELLIPSE`
//! (whole or a part, by its major axis, ratio and extrusion) with a segment for each sixteenth
//! of a half turn, and `SPLINE` as the NURBS curve of its degree, knots and weights, each knot
//! span with a segment for each sixteenth of a half turn its control points turn, up to 32; a
//! spline with only fit points is drawn through them. These count against [`MAX_ARC_POINTS`]
//! too: past it, a circle, arc or ellipse has a segment for each quarter turn and a spline is
//! drawn through its control points, and the asset notes it.
//!
//! Faces, lines and points take a material for each layer and color: the entity's color index or true color, or
//! its layer's (`BYLAYER`), or its block reference's (`BYBLOCK`); in a block, layer `0` is the
//! reference's layer. Materials are named after their layers and are double-sided, as faces in
//! a drawing aren't wound any one way. Layers turned off or frozen and paper space are left out.
//!
//! The drawing sits under a `RootNode` that turns it from Z-up to glTF's Y-up and scales it to
//! meters by `$INSUNITS` (unitless drawings are left as they are, their units unknown). Text,
//! hatches, dimensions and ACIS solids and surfaces aren't read: the asset's
//! `extras.source.skipped` counts them.
//!
//! The source info has the drawing's `$ACADVER` as its version, its `$INSUNITS` as declared
//! units, Z up (assumed) and right-handed, `$LASTSAVEDBY` as its author, and its creation and
//! update dates (`$TDUCREATE` and `$TDUUPDATE` in UTC, else the local `$TDCREATE` and
//! `$TDUPDATE`) in ISO 8601. `$MEASUREMENT` is kept as `measurement`, imperial or metric.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::{FRAC_PI_2, TAU};

use serde_json::{Map, Value, json};

use super::Input;
use super::dates;
use super::nurbs;
use super::stream::f64_from;
use crate::Result;
use crate::document::{Builder, Document};
use crate::mesh::{FastMap, Material, Prim, srgb_to_linear, triangulate};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::vec3::{self, cross};

/// What a binary DXF file starts with.
const SENTINEL: &[u8] = b"AutoCAD Binary DXF\r\n\x1a\0";
/// The deepest block references go in one another.
const MAX_DEPTH: usize = 16;
/// The most nodes block references make.
const MAX_NODES: usize = 1 << 20;
/// The most points arcs in polylines, circles, arcs, ellipses and splines add to a drawing; past
/// them, a bulge is its chord, a circle, arc or ellipse a segment for each quarter turn and a
/// spline its control points. Each is a few bytes of the file but up to 32 segments (a spline
/// up to 32 for each knot span), so this bounds what a file of curves costs.
const MAX_ARC_POINTS: usize = 1 << 22;
/// The turn each segment of an arc stands for, at most: a sixteenth of a half turn.
const ARC_STEP: f64 = std::f64::consts::PI / 16.0;
const WHITE: u32 = 0xFF_FFFF;

/// A group's value: text's bytes, still to be read as a number when one is wanted, or a binary
/// file's typed value.
#[derive(Clone, Copy)]
enum V<'d> {
    Text(&'d [u8]),
    Real(f64),
    Int(i64),
}

impl<'d> V<'d> {
    fn real(self) -> f64 {
        match self {
            V::Text(t) => f64_from(t).unwrap_or(f64::NAN),
            V::Real(v) => v,
            #[allow(clippy::cast_precision_loss, reason = "a coordinate written as an integer")]
            V::Int(i) => i as f64,
        }
    }

    fn int(self) -> i64 {
        match self {
            V::Text(t) => int(t),
            V::Int(i) => i,
            #[allow(clippy::cast_possible_truncation, reason = "an integer written as a real, saturated")]
            V::Real(v) => v as i64,
        }
    }

    fn text(self) -> &'d [u8] {
        match self {
            V::Text(t) => t,
            _ => b"",
        }
    }
}

/// An integer's text: digits after an optional sign, or a real (truncated); 0 for neither.
fn int(t: &[u8]) -> i64 {
    let (neg, digits) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if !digits.is_empty() && digits.len() < 19 && digits.iter().all(u8::is_ascii_digit) {
        let v = digits.iter().fold(0i64, |v, &d| v * 10 + i64::from(d - b'0'));
        return if neg { -v } else { v };
    }
    V::Real(f64_from(t).unwrap_or(0.0)).int()
}

/// A group code's text.
fn code(t: &[u8]) -> Option<i32> {
    let (neg, digits) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || digits.len() > 6 || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let v = digits.iter().fold(0i32, |v, &d| v * 10 + i32::from(d - b'0'));
    Some(if neg { -v } else { v })
}

/// How a binary file writes a group's value, by its code.
enum Kind {
    Text,
    Real,
    I16,
    I32,
    I64,
    Byte,
    Chunk,
}

fn kind(code: i32) -> Kind {
    match code {
        10..=59 | 110..=149 | 210..=239 | 460..=469 | 1010..=1059 => Kind::Real,
        60..=79 | 170..=179 | 270..=289 | 370..=389 | 400..=409 | 1060..=1070 => Kind::I16,
        90..=99 | 420..=429 | 440..=459 | 1071 => Kind::I32,
        160..=169 => Kind::I64,
        290..=299 => Kind::Byte,
        310..=319 | 1004 => Kind::Chunk,
        _ => Kind::Text,
    }
}

/// A file's groups, one at a time: a code and its value.
struct Pairs<'d> {
    d: &'d [u8],
    at: usize,
    binary: bool,
    /// A binary file's codes are two bytes (one, in the oldest files).
    wide: bool,
    back: Option<(i32, V<'d>)>,
}

impl<'d> Pairs<'d> {
    /// The next line, without its end (`\n`, `\r\n` or a lone `\r`).
    #[inline]
    fn line(&mut self) -> Option<&'d [u8]> {
        let d = self.d;
        let rest = d.get(self.at..).filter(|r| !r.is_empty())?;
        let n = rest
            .iter()
            .position(|&b| b == b'\n' || b == b'\r')
            .unwrap_or(rest.len());
        let mut at = self.at + n;
        if at < d.len() {
            at += if d[at] == b'\r' && d.get(at + 1) == Some(&b'\n') {
                2
            } else {
                1
            };
        }
        self.at = at;
        Some(&rest[..n])
    }

    #[inline]
    fn next(&mut self) -> Option<(i32, V<'d>)> {
        if let Some(p) = self.back.take() {
            return Some(p);
        }
        if self.binary {
            return self.next_binary();
        }
        let c = code(self.line()?.trim_ascii())?;
        let v = self.line()?.trim_ascii();
        Some((c, V::Text(v)))
    }

    fn next_binary(&mut self) -> Option<(i32, V<'d>)> {
        let d = self.d;
        let short = |at: usize| {
            d.get(at..)
                .and_then(|r| r.first_chunk::<2>())
                .map(|b| i16::from_le_bytes(*b))
        };
        let c = if self.wide {
            let c = short(self.at)?;
            self.at += 2;
            i32::from(c)
        } else {
            let c = *d.get(self.at)?;
            self.at += 1;
            if c == 255 {
                let c = short(self.at)?;
                self.at += 2;
                i32::from(c)
            } else {
                i32::from(c)
            }
        };
        let rest = d.get(self.at..)?;
        let (v, len) = match kind(c) {
            Kind::Text => {
                let n = rest.iter().position(|&b| b == 0)?;
                (V::Text(rest[..n].trim_ascii()), n + 1)
            }
            Kind::Real => (V::Real(f64::from_le_bytes(*rest.first_chunk()?)), 8),
            Kind::I16 => (V::Int(i64::from(i16::from_le_bytes(*rest.first_chunk()?))), 2),
            Kind::I32 => (V::Int(i64::from(i32::from_le_bytes(*rest.first_chunk()?))), 4),
            Kind::I64 => (V::Int(i64::from_le_bytes(*rest.first_chunk()?)), 8),
            Kind::Byte => (V::Int(i64::from(*rest.first()?)), 1),
            Kind::Chunk => {
                let n = usize::from(*rest.first()?);
                (V::Text(rest.get(1..=n)?), n + 1)
            }
        };
        self.at += len;
        Some((c, v))
    }

    /// The group after, left to read.
    fn peek(&mut self) -> Option<(i32, V<'d>)> {
        if self.back.is_none() {
            self.back = self.next();
        }
        self.back
    }

    /// The groups up to the next entity, record or section boundary (code 0).
    fn fields(&mut self) -> Fields<'d> {
        let mut f = Fields {
            reals: [1.0, 1.0, 1.0, 0.0, 0.0],
            extrusion: [0.0, 0.0, 1.0],
            ..Fields::default()
        };
        while let Some((c, v)) = self.peek() {
            if c == 0 {
                break;
            }
            self.back = None;
            match c {
                2 => f.name = v.text(),
                8 => f.layer = v.text(),
                10..=13 => {
                    f.p[slot(c, 10)][0] = v.real();
                    f.given |= 1 << slot(c, 10);
                }
                20..=23 => f.p[slot(c, 20)][1] = v.real(),
                30..=33 => f.p[slot(c, 30)][2] = v.real(),
                41..=45 => {
                    f.reals[slot(c, 41)] = v.real();
                    if c == 42 {
                        f.bulge = v.real();
                    }
                }
                50 => f.angle = v.real(),
                62 => f.color = Some(v.int()),
                67 => f.paper = v.int() == 1,
                70..=75 => f.ints[slot(c, 70)] = v.int(),
                210 => f.extrusion[0] = v.real(),
                220 => f.extrusion[1] = v.real(),
                230 => f.extrusion[2] = v.real(),
                420 => f.true_color = Some(v.int()),
                _ => {}
            }
        }
        f
    }
}

fn slot(code: i32, base: i32) -> usize {
    usize::try_from(code - base).unwrap_or(0)
}

/// The groups of an entity or table record that this reader uses.
#[derive(Default)]
struct Fields<'d> {
    name: &'d [u8],
    layer: &'d [u8],
    /// Points 10 to 13.
    p: [[f64; 3]; 4],
    /// Which of those were given, a bit each.
    given: u8,
    /// Groups 41 to 45: scales, then column and row spacing.
    reals: [f64; 5],
    /// A vertex's bulge (group 42): the tangent of a quarter of the arc to the next vertex.
    bulge: f64,
    angle: f64,
    color: Option<i64>,
    true_color: Option<i64>,
    paper: bool,
    /// Groups 70 to 75.
    ints: [i64; 6],
    extrusion: [f64; 3],
}

/// An entity's color: its own, its layer's or its block reference's.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Spec {
    ByLayer,
    ByBlock,
    Rgb(u32),
}

impl Fields<'_> {
    fn spec(&self) -> Spec {
        if let Some(c) = self.true_color {
            return Spec::Rgb(u32::try_from(c & 0xFF_FFFF).unwrap_or(WHITE));
        }
        match self.color.map(i64::abs) {
            None | Some(256..) => Spec::ByLayer,
            Some(0) => Spec::ByBlock,
            Some(c) => Spec::Rgb(aci(c)),
        }
    }
}

/// The RGB of a color index (1 to 255): seven named colors, two grays, 24 hues at five
/// brightnesses, full and half saturated, and six more grays.
fn aci(c: i64) -> u32 {
    let rgb = |r: i64, g: i64, b: i64| u32::try_from((r << 16) | (g << 8) | b).unwrap_or(WHITE);
    match c {
        1 => 0xFF_0000,
        2 => 0xFF_FF00,
        3 => 0x00_FF00,
        4 => 0x00_FFFF,
        5 => 0x00_00FF,
        6 => 0xFF_00FF,
        8 => 0x80_8080,
        9 => 0xC0_C0C0,
        10..=249 => {
            let (h, level) = ((c - 10) / 10, (c - 10) % 10);
            let v = [255, 165, 127, 76, 38][usize::try_from(level / 2).unwrap_or(0)];
            let up = |k: i64| v * k / 4;
            let [r, g, b] = match h {
                0..=3 => [v, up(h), 0],
                4..=7 => [up(8 - h), v, 0],
                8..=11 => [0, v, up(h - 8)],
                12..=15 => [0, up(16 - h), v],
                16..=19 => [up(h - 16), 0, v],
                _ => [v, 0, up(24 - h)],
            };
            if level % 2 == 1 {
                rgb(r + (v - r) / 2, g + (v - g) / 2, b + (v - b) / 2)
            } else {
                rgb(r, g, b)
            }
        }
        250..=255 => {
            let v = [51, 80, 105, 130, 190, 255][usize::try_from(c - 250).unwrap_or(0)];
            rgb(v, v, v)
        }
        _ => WHITE,
    }
}

struct Layer {
    name: String,
    rgb: u32,
    hidden: bool,
}

/// Faces, lines and points of one layer and color in a block or model space.
struct Bucket {
    layer: u32,
    spec: Spec,
    positions: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    lines: Vec<[u32; 2]>,
    points: Vec<u32>,
    seen: FastMap<[u32; 3], u32>,
}

impl Bucket {
    #[inline]
    fn vertex(&mut self, p: [f64; 3]) -> u32 {
        // Adding zero makes -0 +0, so the two are one point.
        let q = p.map(|x| f32_of(x) + 0.0);
        let positions = &mut self.positions;
        *self.seen.entry(q.map(f32::to_bits)).or_insert_with(|| {
            positions.push(q);
            u32_of(positions.len() - 1)
        })
    }

    /// Adds a polygon; the triangles it made.
    fn polygon(&mut self, pts: &[[f64; 3]], s: &mut Scratch) -> usize {
        if pts.len() < 3 || pts.iter().flatten().any(|x| !x.is_finite()) {
            return 0;
        }
        s.ids.clear();
        for &p in pts {
            let v = self.vertex(p);
            s.ids.push(v);
        }
        s.ids.dedup();
        if s.ids.len() > 1 && s.ids.first() == s.ids.last() {
            s.ids.pop();
        }
        let before = self.tris.len();
        if s.ids.len() == 3 {
            self.tris.push([s.ids[0], s.ids[1], s.ids[2]]);
        } else if s.ids.len() > 3 {
            s.corners.clear();
            s.corners.extend(s.ids.iter().map(|&v| self.positions[v as usize]));
            s.tris.clear();
            triangulate(&s.corners, &mut s.tris);
            for t in &s.tris {
                let t = t.map(|k| s.ids[k as usize]);
                if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                    self.tris.push(t);
                }
            }
        }
        self.tris.len() - before
    }

    /// Adds the segments between a run of points, and from the last back to the first when
    /// it's `closed`; a point that isn't finite breaks the run there.
    fn polyline(&mut self, pts: &[[f64; 3]], closed: bool) {
        let mut last = None;
        for &p in pts {
            let v = p.iter().all(|x| x.is_finite()).then(|| self.vertex(p));
            if let (Some(a), Some(b)) = (last, v)
                && a != b
            {
                self.lines.push([a, b]);
            }
            last = v;
        }
        if closed && pts.len() > 2 {
            self.polyline(&[pts[pts.len() - 1], pts[0]], false);
        }
    }

    fn point(&mut self, p: [f64; 3]) {
        if p.iter().all(|x| x.is_finite()) {
            let v = self.vertex(p);
            self.points.push(v);
        }
    }
}

#[derive(Default)]
struct Scratch {
    ids: Vec<u32>,
    corners: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    pts: Vec<[f64; 3]>,
}

/// A block reference.
struct Insert {
    block: Vec<u8>,
    layer: u32,
    spec: Spec,
    at: [f64; 3],
    scale: [f64; 3],
    /// Radians about the extrusion.
    angle: f64,
    extrusion: [f64; 3],
    cols: u32,
    rows: u32,
    gap: [f64; 2],
}

/// Model space (the first) or a block.
#[derive(Default)]
struct Space {
    name: String,
    base: [f64; 3],
    buckets: Vec<Bucket>,
    bucket_of: FastMap<(u32, Spec), usize>,
    inserts: Vec<Insert>,
}

impl Space {
    fn bucket(&mut self, layer: u32, spec: Spec) -> &mut Bucket {
        let n = self.buckets.len();
        let k = *self.bucket_of.entry((layer, spec)).or_insert(n);
        if k == n {
            self.buckets.push(Bucket {
                layer,
                spec,
                positions: Vec::new(),
                tris: Vec::new(),
                lines: Vec::new(),
                points: Vec::new(),
                seen: FastMap::default(),
            });
        }
        &mut self.buckets[k]
    }
}

/// A drawing, read.
#[derive(Default)]
struct Model {
    layers: Vec<Layer>,
    layer_of: HashMap<Vec<u8>, u32>,
    spaces: Vec<Space>,
    block_of: HashMap<Vec<u8>, usize>,
    units: i64,
    version: String,
    /// `$TDCREATE` and `$TDUPDATE` (local Julian dates), and `$TDUCREATE` and `$TDUUPDATE` (UTC).
    created: Option<f64>,
    modified: Option<f64>,
    created_utc: Option<f64>,
    modified_utc: Option<f64>,
    last_saved_by: String,
    /// `$MEASUREMENT`: 0 imperial, 1 metric.
    measurement: Option<i64>,
    /// Triangles read, each once.
    triangles: usize,
    /// Points arcs added, up to [`MAX_ARC_POINTS`].
    arc_points: usize,
    limit: usize,
    ended: bool,
    skipped: BTreeMap<Vec<u8>, usize>,
    notes: BTreeSet<&'static str>,
    scratch: Scratch,
}

impl Model {
    /// The layer named (made, white, when the drawing doesn't define it).
    fn layer(&mut self, name: &[u8]) -> u32 {
        let name = if name.is_empty() { b"0" } else { name };
        let key = name.to_ascii_uppercase();
        if let Some(&l) = self.layer_of.get(&key) {
            return l;
        }
        self.layers.push(Layer {
            name: String::from_utf8_lossy(name).into_owned(),
            rgb: WHITE,
            hidden: false,
        });
        let l = u32_of(self.layers.len() - 1);
        self.layer_of.insert(key, l);
        l
    }

    fn skip(&mut self, kind: &[u8]) {
        if let Some(n) = self.skipped.get_mut(kind) {
            *n += 1;
        } else {
            self.skipped.insert(kind.to_vec(), 1);
        }
    }

    fn header(&mut self, p: &mut Pairs<'_>) {
        let mut var: &[u8] = b"";
        while let Some((c, v)) = p.next() {
            match c {
                0 => {
                    if v.text() != b"ENDSEC" {
                        p.back = Some((c, v));
                    }
                    return;
                }
                9 => var = v.text(),
                70 if var == b"$INSUNITS" => self.units = v.int(),
                1 if var == b"$ACADVER" => self.version = String::from_utf8_lossy(v.text()).into_owned(),
                1 if var == b"$LASTSAVEDBY" => {
                    String::from_utf8_lossy(v.text())
                        .trim()
                        .clone_into(&mut self.last_saved_by);
                }
                40 if var == b"$TDCREATE" => self.created = Some(v.real()),
                40 if var == b"$TDUPDATE" => self.modified = Some(v.real()),
                40 if var == b"$TDUCREATE" => self.created_utc = Some(v.real()),
                40 if var == b"$TDUUPDATE" => self.modified_utc = Some(v.real()),
                70 if var == b"$MEASUREMENT" => self.measurement = Some(v.int()),
                _ => {}
            }
        }
    }

    /// What the drawing says of itself: `units` from `$INSUNITS`, and whether a root turns it to
    /// Y-up (and scales it, unless its unit is the meter).
    fn info(&self, units: Option<(f64, &'static str)>, rooted: bool) -> SourceInfo {
        // A UTC date when the drawing has one, else a local one.
        let date = |utc: Option<f64>, local: Option<f64>| {
            utc.and_then(|d| dates::julian(d, true))
                .or_else(|| local.and_then(|d| dates::julian(d, false)))
        };
        let mut info = SourceInfo {
            format: Some("DXF".into()),
            version: Some(self.version.clone()).filter(|v| !v.is_empty()),
            meters_per_unit: units.map_or(Fact::Unknown, |(m, _)| Fact::Declared(m)),
            unit: units.map(|(_, name)| name.to_owned()),
            up_axis: Fact::Assumed(Axis::PosZ),
            handedness: Fact::Declared(Handedness::Right),
            author: Some(self.last_saved_by.clone()).filter(|a| !a.is_empty()),
            created: date(self.created_utc, self.created),
            modified: date(self.modified_utc, self.modified),
            ..SourceInfo::default()
        };
        info.converted.up_axis = rooted;
        info.converted.units = rooted && units.is_some_and(|(m, _)| (m - 1.0).abs() > 1e-12);
        info
    }

    fn tables(&mut self, p: &mut Pairs<'_>) {
        while let Some((c, v)) = p.next() {
            if c != 0 {
                continue;
            }
            match v.text() {
                b"LAYER" => {
                    let f = p.fields();
                    let l = self.layer(f.name) as usize;
                    let color = f.color.unwrap_or(7);
                    let rgb = match f.true_color {
                        Some(t) => u32::try_from(t & 0xFF_FFFF).unwrap_or(WHITE),
                        None => aci(color.abs()),
                    };
                    self.layers[l].rgb = rgb;
                    self.layers[l].hidden = color < 0 || f.ints[0] & 1 != 0;
                }
                b"ENDSEC" => return,
                b"EOF" => {
                    p.back = Some((c, v));
                    return;
                }
                _ => {}
            }
        }
    }

    fn blocks(&mut self, p: &mut Pairs<'_>) -> Result<()> {
        while let Some((c, v)) = p.next() {
            if c != 0 {
                continue;
            }
            match v.text() {
                b"BLOCK" => {
                    let f = p.fields();
                    if f.ints[0] & 4 != 0 {
                        self.notes.insert("external references weren't followed");
                    }
                    let key = f.name.to_ascii_uppercase();
                    self.spaces.push(Space {
                        name: String::from_utf8_lossy(f.name).into_owned(),
                        base: f.p[0],
                        ..Space::default()
                    });
                    let s = self.spaces.len() - 1;
                    self.block_of.entry(key).or_insert(s);
                    self.entities(p, s, b"ENDBLK")?;
                }
                b"ENDSEC" => return Ok(()),
                b"EOF" => {
                    p.back = Some((c, v));
                    return Ok(());
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Reads entities into space `s` up to `end` (`ENDSEC` or `ENDBLK`).
    fn entities(&mut self, p: &mut Pairs<'_>, s: usize, end: &[u8]) -> Result<()> {
        while let Some((c, v)) = p.next() {
            if c != 0 {
                continue;
            }
            let kind = v.text();
            if kind == end {
                p.fields();
                return Ok(());
            }
            match kind {
                b"ENDSEC" | b"ENDBLK" | b"EOF" | b"BLOCK" | b"SECTION" => {
                    p.back = Some((c, v));
                    return Ok(());
                }
                b"3DFACE" => {
                    let mut f = p.fields();
                    if self.paper(&f) {
                        continue;
                    }
                    if f.given & 8 == 0 {
                        f.p[3] = f.p[2];
                    }
                    let layer = self.layer(f.layer);
                    let bucket = self.spaces[s].bucket(layer, f.spec());
                    self.triangles += bucket.polygon(&f.p, &mut self.scratch);
                }
                b"LINE" | b"POINT" => {
                    let f = p.fields();
                    if self.paper(&f) {
                        continue;
                    }
                    let layer = self.layer(f.layer);
                    let bucket = self.spaces[s].bucket(layer, f.spec());
                    if kind == b"LINE" {
                        bucket.polyline(&f.p[..2], false);
                    } else {
                        bucket.point(f.p[0]);
                    }
                }
                b"LWPOLYLINE" => self.lwpolyline(p, s),
                b"POLYLINE" => self.polyline(p, s),
                b"CIRCLE" | b"ARC" | b"ELLIPSE" => self.conic(kind, p, s),
                b"SPLINE" => self.spline(p, s),
                b"MESH" => self.mesh(p, s),
                b"INSERT" => {
                    let f = p.fields();
                    if self.paper(&f) {
                        continue;
                    }
                    let count = |n: i64| u32::try_from(n.clamp(1, 32_767)).unwrap_or(1);
                    let insert = Insert {
                        block: f.name.to_ascii_uppercase(),
                        layer: self.layer(f.layer),
                        spec: f.spec(),
                        at: f.p[0],
                        scale: [f.reals[0], f.reals[1], f.reals[2]],
                        angle: f.angle.to_radians(),
                        extrusion: f.extrusion,
                        cols: count(f.ints[0]),
                        rows: count(f.ints[1]),
                        gap: [f.reals[3], f.reals[4]],
                    };
                    self.spaces[s].inserts.push(insert);
                }
                // Parts of a polyline or a block reference, read with it.
                b"VERTEX" | b"SEQEND" | b"ATTRIB" => {
                    p.fields();
                }
                _ => {
                    self.skip(kind);
                    p.fields();
                }
            }
            ensure!(
                self.triangles <= self.limit,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                self.limit
            );
        }
        Ok(())
    }

    /// Whether an entity is in paper space, and so left out.
    fn paper(&mut self, f: &Fields<'_>) -> bool {
        if f.paper {
            self.notes.insert("paper space was left out");
        }
        f.paper
    }

    /// A 2D polyline's outline in world space: its vertices (in the object coordinate system,
    /// at `elevation`) with each bulge's arc drawn as segments, into `out`.
    fn outline(
        &mut self,
        verts: &[([f64; 2], f64)],
        closed: bool,
        elevation: f64,
        extrusion: [f64; 3],
        out: &mut Vec<[f64; 3]>,
    ) {
        let [ax, ay, az] = axes(extrusion);
        let world =
            |q: [f64; 2]| -> [f64; 3] { std::array::from_fn(|k| ax[k] * q[0] + ay[k] * q[1] + az[k] * elevation) };
        out.clear();
        let mut arc = Vec::new();
        for (i, &(a, bulge)) in verts.iter().enumerate() {
            out.push(world(a));
            let next = if i + 1 < verts.len() {
                verts[i + 1].0
            } else if closed && verts.len() > 1 {
                verts[0].0
            } else {
                break;
            };
            if self.arc_points >= MAX_ARC_POINTS {
                if bulge != 0.0 {
                    self.notes.insert("arcs past the most points were drawn as chords");
                }
                continue;
            }
            arc.clear();
            bulge_arc(a, next, bulge, &mut arc);
            self.arc_points += arc.len();
            out.extend(arc.iter().map(|&q| world(q)));
        }
    }

    /// An `LWPOLYLINE`: its vertices (`10`, `20`) and their bulges (`42`) in the object
    /// coordinate system at its elevation (`38`), closed by flag 1.
    fn lwpolyline(&mut self, p: &mut Pairs<'_>, s: usize) {
        let mut f = Fields {
            extrusion: [0.0, 0.0, 1.0],
            ..Fields::default()
        };
        let mut elevation = 0.0;
        let mut verts: Vec<([f64; 2], f64)> = Vec::new();
        while let Some((c, v)) = p.peek() {
            if c == 0 {
                break;
            }
            p.back = None;
            match c {
                8 => f.layer = v.text(),
                38 => elevation = v.real(),
                62 => f.color = Some(v.int()),
                67 => f.paper = v.int() == 1,
                70 => f.ints[0] = v.int(),
                10 => verts.push(([v.real(), 0.0], 0.0)),
                20 => {
                    if let Some(q) = verts.last_mut() {
                        q.0[1] = v.real();
                    }
                }
                42 => {
                    if let Some(q) = verts.last_mut() {
                        q.1 = v.real();
                    }
                }
                210 => f.extrusion[0] = v.real(),
                220 => f.extrusion[1] = v.real(),
                230 => f.extrusion[2] = v.real(),
                420 => f.true_color = Some(v.int()),
                _ => {}
            }
        }
        if self.paper(&f) {
            return;
        }
        let closed = f.ints[0] & 1 != 0;
        let mut pts = std::mem::take(&mut self.scratch.pts);
        self.outline(&verts, closed, elevation, f.extrusion, &mut pts);
        let layer = self.layer(f.layer);
        self.spaces[s].bucket(layer, f.spec()).polyline(&pts, closed);
        self.scratch.pts = pts;
    }

    /// A `POLYLINE` and its vertices: a polyface or polygon mesh, or a 2D or 3D polyline.
    fn polyline(&mut self, p: &mut Pairs<'_>, s: usize) {
        let f = p.fields();
        let mut verts = Vec::new();
        while let Some((0, v)) = p.peek() {
            match v.text() {
                b"VERTEX" => {
                    p.back = None;
                    verts.push(p.fields());
                }
                b"SEQEND" => {
                    p.back = None;
                    p.fields();
                    break;
                }
                _ => break,
            }
        }
        if self.paper(&f) {
            return;
        }
        let flags = f.ints[0];
        let layer = self.layer(f.layer);
        let spec = f.spec();
        let mut pts = std::mem::take(&mut self.scratch.pts);
        if flags & 64 != 0 {
            // Position vertices, then faces of their (1-based) numbers, negative for an
            // invisible edge.
            let mut positions = Vec::new();
            let mut faces = Vec::new();
            for v in &verts {
                let corners = &v.ints[1..5];
                if v.ints[0] & 64 != 0 || corners.iter().all(|&k| k == 0) {
                    positions.push(v.p[0]);
                } else {
                    faces.push(v);
                }
            }
            for v in faces {
                pts.clear();
                for &k in &v.ints[1..5] {
                    let k = usize::try_from(k.unsigned_abs()).unwrap_or(0);
                    if let Some(&q) = k.checked_sub(1).and_then(|k| positions.get(k)) {
                        pts.push(q);
                    }
                }
                let layer = if v.layer.is_empty() { layer } else { self.layer(v.layer) };
                let spec = if v.color.is_some() || v.true_color.is_some() {
                    v.spec()
                } else {
                    spec
                };
                let bucket = self.spaces[s].bucket(layer, spec);
                self.triangles += bucket.polygon(&pts, &mut self.scratch);
            }
        } else if flags & 16 != 0 {
            // M rows of N, or the surface fitted to them (`73` by `74`).
            let fitted = flags & 4 != 0 && f.ints[5] != 0 && verts.iter().any(|v| v.ints[0] & 8 != 0);
            let (grid, m, n): (Vec<[f64; 3]>, i64, i64) = if fitted {
                let g = verts.iter().filter(|v| v.ints[0] & 8 != 0).map(|v| v.p[0]).collect();
                (g, f.ints[3], f.ints[4])
            } else {
                let g = verts.iter().filter(|v| v.ints[0] & 16 == 0).map(|v| v.p[0]).collect();
                (g, f.ints[1], f.ints[2])
            };
            let n = usize::try_from(n).unwrap_or(0);
            let m = usize::try_from(m)
                .unwrap_or(0)
                .min(grid.len().checked_div(n).unwrap_or(0));
            let (closed_m, closed_n) = (flags & 1 != 0, flags & 32 != 0);
            let bucket = self.spaces[s].bucket(layer, spec);
            if m >= 2 && n >= 2 {
                let rows = if closed_m { m } else { m - 1 };
                let cols = if closed_n { n } else { n - 1 };
                for i in 0..rows {
                    for j in 0..cols {
                        let (i2, j2) = ((i + 1) % m, (j + 1) % n);
                        pts.clear();
                        pts.extend([grid[i * n + j], grid[i * n + j2], grid[i2 * n + j2], grid[i2 * n + j]]);
                        self.triangles += bucket.polygon(&pts, &mut self.scratch);
                    }
                }
            }
        } else {
            // A spline fitted polyline has its frame's vertices (flag 16) beside the fitted ones:
            // the fitted ones are drawn.
            let framed = verts.iter().any(|v| v.ints[0] & 16 != 0) && verts.iter().any(|v| v.ints[0] & 16 == 0);
            let verts = verts.iter().filter(|v| !framed || v.ints[0] & 16 == 0);
            let closed = flags & 1 != 0;
            if flags & 8 != 0 {
                pts.clear();
                pts.extend(verts.map(|v| v.p[0]));
            } else {
                let plane: Vec<([f64; 2], f64)> = verts.map(|v| ([v.p[0][0], v.p[0][1]], v.bulge)).collect();
                self.outline(&plane, closed, f.p[0][2], f.extrusion, &mut pts);
            }
            self.spaces[s].bucket(layer, spec).polyline(&pts, closed);
        }
        self.scratch.pts = pts;
    }

    /// A `CIRCLE`, `ARC` or `ELLIPSE`, drawn as segments. A circle or arc is at its center
    /// (`10`) and radius (`40`) in the object coordinate system, the center's height its
    /// elevation, an arc from its start angle (`50`) to its end (`51`) counterclockwise, in
    /// degrees. An ellipse is at its center and major axis (`11`, from the center) in world
    /// space, its minor axis `40` times as long and a quarter turn counterclockwise about its
    /// extrusion, from its start parameter (`41`) to its end (`42`), in radians.
    fn conic(&mut self, kind: &[u8], p: &mut Pairs<'_>, s: usize) {
        let ellipse = kind == b"ELLIPSE";
        let mut f = Fields {
            extrusion: [0.0, 0.0, 1.0],
            ..Fields::default()
        };
        let (mut size, mut from, mut to) = (0.0, 0.0, if ellipse { TAU } else { 0.0 });
        while let Some((c, v)) = p.peek() {
            if c == 0 {
                break;
            }
            p.back = None;
            match c {
                8 => f.layer = v.text(),
                10 | 11 => f.p[slot(c, 10)][0] = v.real(),
                20 | 21 => f.p[slot(c, 20)][1] = v.real(),
                30 | 31 => f.p[slot(c, 30)][2] = v.real(),
                40 => size = v.real(),
                41 | 50 if ellipse == (c == 41) => from = v.real(),
                42 | 51 if ellipse == (c == 42) => to = v.real(),
                62 => f.color = Some(v.int()),
                67 => f.paper = v.int() == 1,
                210 => f.extrusion[0] = v.real(),
                220 => f.extrusion[1] = v.real(),
                230 => f.extrusion[2] = v.real(),
                420 => f.true_color = Some(v.int()),
                _ => {}
            }
        }
        if self.paper(&f) {
            return;
        }
        let [ax, ay, az] = axes(f.extrusion);
        let [c, m] = [f.p[0], f.p[1]];
        // The curve is the center plus `u` times the cosine of its parameter and `v` times the sine.
        let (center, u, v, start, sweep, closed) = if ellipse {
            let minor = [
                az[1] * m[2] - az[2] * m[1],
                az[2] * m[0] - az[0] * m[2],
                az[0] * m[1] - az[1] * m[0],
            ];
            let sweep = turned(from, to, TAU);
            (c, m, minor.map(|x| x * size), from, sweep, sweep >= TAU)
        } else {
            let center = std::array::from_fn(|k| ax[k] * c[0] + ay[k] * c[1] + az[k] * c[2]);
            let (start, sweep, closed) = if kind == b"ARC" {
                let sweep = turned(from, to, 360.0);
                (from.to_radians(), sweep.to_radians(), sweep >= 360.0)
            } else {
                (0.0, TAU, true)
            };
            (center, ax.map(|x| x * size), ay.map(|x| x * size), start, sweep, closed)
        };
        let mut n = segments(sweep, ARC_STEP);
        if self.arc_points.saturating_add(n) > MAX_ARC_POINTS {
            self.notes.insert("curves past the most points were drawn coarsely");
            n = segments(sweep, FRAC_PI_2);
        } else {
            self.arc_points += n;
        }
        let mut pts = std::mem::take(&mut self.scratch.pts);
        pts.clear();
        let ends = if closed { n } else { n + 1 };
        pts.extend((0..ends).map(|i| {
            #[allow(clippy::cast_precision_loss, reason = "at most 32 segments")]
            let t = start + sweep * i as f64 / n as f64;
            let (sin, cos) = t.sin_cos();
            std::array::from_fn(|k| center[k] + u[k] * cos + v[k] * sin)
        }));
        let layer = self.layer(f.layer);
        self.spaces[s].bucket(layer, f.spec()).polyline(&pts, closed);
        self.scratch.pts = pts;
    }

    /// A `SPLINE`: a NURBS curve of its degree (`71`) through its control points (`10`, in
    /// world space), knots (`40`) and weights (`41`), drawn as segments; one with only fit
    /// points (`11`) is drawn through them. Closed by flag 1. The counts it gives (`72` to
    /// `74`) aren't trusted: the values are counted as they're read.
    fn spline(&mut self, p: &mut Pairs<'_>, s: usize) {
        let mut f = Fields::default();
        let (mut degree, mut flags) = (3, 0);
        let (mut knots, mut weights) = (Vec::new(), Vec::new());
        let (mut ctrl, mut fit): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (Vec::new(), Vec::new());
        while let Some((c, v)) = p.peek() {
            if c == 0 {
                break;
            }
            p.back = None;
            let set = |pts: &mut Vec<[f64; 3]>, k: usize| {
                if let Some(q) = pts.last_mut() {
                    q[k] = v.real();
                }
            };
            match c {
                8 => f.layer = v.text(),
                10 => ctrl.push([v.real(), 0.0, 0.0]),
                20 => set(&mut ctrl, 1),
                30 => set(&mut ctrl, 2),
                11 => fit.push([v.real(), 0.0, 0.0]),
                21 => set(&mut fit, 1),
                31 => set(&mut fit, 2),
                40 => knots.push(v.real()),
                41 => weights.push(v.real()),
                62 => f.color = Some(v.int()),
                67 => f.paper = v.int() == 1,
                70 => flags = v.int(),
                71 => degree = v.int(),
                420 => f.true_color = Some(v.int()),
                _ => {}
            }
        }
        if self.paper(&f) {
            return;
        }
        let closed = flags & 1 != 0;
        let mut pts = std::mem::take(&mut self.scratch.pts);
        if ctrl.is_empty() {
            pts.clear();
            pts.extend_from_slice(&fit);
        } else {
            let budget = MAX_ARC_POINTS.saturating_sub(self.arc_points);
            match nurbs::curve(degree, &ctrl, &weights, &knots, budget, &mut pts) {
                Ok(n) => self.arc_points += n,
                Err(fail) => {
                    self.notes.insert(match fail {
                        nurbs::Fail::Unfit => {
                            "splines whose degree or knots don't fit were drawn through their control points"
                        }
                        nurbs::Fail::Budget => "curves past the most points were drawn coarsely",
                    });
                    pts.clear();
                    pts.extend_from_slice(&ctrl);
                }
            }
        }
        let layer = self.layer(f.layer);
        self.spaces[s].bucket(layer, f.spec()).polyline(&pts, closed);
        self.scratch.pts = pts;
    }

    /// A `MESH`: its vertices and faces (a count, then that many vertex numbers), the control
    /// mesh drawn as it is.
    fn mesh(&mut self, p: &mut Pairs<'_>, s: usize) {
        #[derive(PartialEq)]
        enum At {
            Other,
            Vertices,
            Faces,
        }
        let (mut at, mut want) = (At::Other, 0usize);
        let mut layer: &[u8] = b"";
        let mut spec = Fields::default();
        let mut paper = false;
        let mut level = 0;
        let mut verts: Vec<[f64; 3]> = Vec::new();
        let mut list: Vec<i64> = Vec::new();
        while let Some((c, v)) = p.peek() {
            if c == 0 {
                break;
            }
            p.back = None;
            match c {
                8 => layer = v.text(),
                62 => spec.color = Some(v.int()),
                420 => spec.true_color = Some(v.int()),
                67 => paper = v.int() == 1,
                91 => level = v.int(),
                92 => (at, want) = (At::Vertices, usize::try_from(v.int()).unwrap_or(0)),
                93 => (at, want) = (At::Faces, usize::try_from(v.int()).unwrap_or(0)),
                94..=95 => at = At::Other,
                10 if at == At::Vertices && verts.len() < want => verts.push([v.real(), 0.0, 0.0]),
                20 if at == At::Vertices => {
                    if let Some(q) = verts.last_mut() {
                        q[1] = v.real();
                    }
                }
                30 if at == At::Vertices => {
                    if let Some(q) = verts.last_mut() {
                        q[2] = v.real();
                    }
                }
                90 if at == At::Faces && list.len() < want => list.push(v.int()),
                _ => {}
            }
        }
        if paper {
            self.notes.insert("paper space was left out");
            return;
        }
        if level > 0 {
            self.notes
                .insert("meshes are drawn at their control mesh, not subdivided");
        }
        let layer = self.layer(layer);
        let bucket = self.spaces[s].bucket(layer, spec.spec());
        let mut pts = std::mem::take(&mut self.scratch.pts);
        let mut k = 0;
        while k < list.len() {
            let n = usize::try_from(list[k]).unwrap_or(0);
            let face = list.get(k + 1..k + 1 + n).unwrap_or_default();
            k += n + 1;
            pts.clear();
            pts.extend(
                face.iter()
                    .filter_map(|&i| usize::try_from(i).ok().and_then(|i| verts.get(i))),
            );
            if pts.len() == n {
                self.triangles += bucket.polygon(&pts, &mut self.scratch);
            }
        }
        self.scratch.pts = pts;
    }
}

/// The meters in a drawing unit (`$INSUNITS`), and the unit's name.
fn unit(units: i64) -> Option<(f64, &'static str)> {
    Some(match units {
        1 => (0.0254, "inch"),
        2 => (0.3048, "foot"),
        3 => (1609.344, "mile"),
        4 => (0.001, "millimeter"),
        5 => (0.01, "centimeter"),
        6 => (1.0, "meter"),
        7 => (1000.0, "kilometer"),
        8 => (2.54e-8, "microinch"),
        9 => (2.54e-5, "mil"),
        10 => (0.9144, "yard"),
        11 => (1e-10, "angstrom"),
        12 => (1e-9, "nanometer"),
        13 => (1e-6, "micron"),
        14 => (0.1, "decimeter"),
        15 => (10.0, "decameter"),
        16 => (100.0, "hectometer"),
        17 => (1e9, "gigameter"),
        18 => (1.495_978_707e11, "astronomical unit"),
        19 => (9.460_730_472_580_8e15, "light year"),
        20 => (3.085_677_581_491_367e16, "parsec"),
        21 => (1200.0 / 3937.0, "US survey foot"),
        22 => (100.0 / 3937.0, "US survey inch"),
        23 => (3600.0 / 3937.0, "US survey yard"),
        24 => (6_336_000.0 / 3937.0, "US survey mile"),
        _ => return None,
    })
}

pub(crate) fn load(input: Input<'_>) -> Result<Document> {
    let data = input.read()?;
    let d = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&data);
    ensure!(!d.iter().all(u8::is_ascii_whitespace), Empty, "DXF file is empty");
    let binary = d.starts_with(SENTINEL);
    let mut p = Pairs {
        d,
        at: if binary { SENTINEL.len() } else { 0 },
        binary,
        wide: d.get(SENTINEL.len() + 1) == Some(&0),
        back: None,
    };
    let mut m = Model {
        limit: input.limit,
        spaces: vec![Space::default()],
        ..Model::default()
    };
    let mut sections = 0;
    while let Some((c, v)) = p.next() {
        if c != 0 {
            continue;
        }
        match v.text() {
            b"SECTION" => {
                sections += 1;
                let name = match p.peek() {
                    Some((2, v)) => {
                        p.back = None;
                        v.text()
                    }
                    _ => b"",
                };
                match name {
                    b"HEADER" => m.header(&mut p),
                    b"TABLES" => m.tables(&mut p),
                    b"BLOCKS" => m.blocks(&mut p)?,
                    b"ENTITIES" => m.entities(&mut p, 0, b"ENDSEC")?,
                    _ => {}
                }
            }
            b"EOF" => {
                m.ended = true;
                break;
            }
            _ => {}
        }
    }
    ensure!(sections > 0, UnknownFormat, "not a DXF file: it has no sections");
    Build::new(m, binary).finish()
}

/// The layer and color a block reference passes to what's in its block.
#[derive(Clone, Copy)]
struct Ctx {
    layer: u32,
    rgb: u32,
}

/// A block reference, its block found.
#[derive(Clone, Copy)]
struct Inst {
    block: Option<usize>,
    layer: u32,
    spec: Spec,
    at: [f64; 3],
    scale: [f64; 3],
    angle: f64,
    extrusion: [f64; 3],
    cols: u32,
    rows: u32,
    gap: [f64; 2],
}

/// Turns a drawing into a glTF document.
struct Build {
    m: Model,
    binary: bool,
    layer0: u32,
    insts: Vec<Vec<Inst>>,
    b: Builder,
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    mesh_tris: Vec<usize>,
    mesh_of: HashMap<(usize, u32, u32), Option<usize>>,
    materials: Vec<Material>,
    material_of: HashMap<(u32, u32), usize>,
    triangles: usize,
    visiting: Vec<usize>,
}

impl Build {
    fn new(mut m: Model, binary: bool) -> Self {
        let layer0 = m.layer(b"0");
        let insts = m
            .spaces
            .iter_mut()
            .map(|s| {
                std::mem::take(&mut s.inserts)
                    .into_iter()
                    .map(|i| Inst {
                        block: m.block_of.get(&i.block).copied(),
                        layer: i.layer,
                        spec: i.spec,
                        at: i.at,
                        scale: i.scale,
                        angle: i.angle,
                        extrusion: i.extrusion,
                        cols: i.cols,
                        rows: i.rows,
                        gap: i.gap,
                    })
                    .collect()
            })
            .collect();
        for s in &mut m.spaces {
            for b in &mut s.buckets {
                b.seen = FastMap::default();
            }
        }
        Self {
            m,
            binary,
            layer0,
            insts,
            b: Builder::default(),
            nodes: Vec::new(),
            meshes: Vec::new(),
            mesh_tris: Vec::new(),
            mesh_of: HashMap::new(),
            materials: Vec::new(),
            material_of: HashMap::new(),
            triangles: 0,
            visiting: Vec::new(),
        }
    }

    fn resolve(&self, spec: Spec, layer: u32, ctx: Ctx) -> u32 {
        match spec {
            Spec::Rgb(c) => c,
            Spec::ByLayer => self.m.layers[layer as usize].rgb,
            Spec::ByBlock => ctx.rgb,
        }
    }

    fn material(&mut self, layer: u32, rgb: u32) -> usize {
        if let Some(&k) = self.material_of.get(&(layer, rgb)) {
            return k;
        }
        let channel =
            |shift: u32| srgb_to_linear(f32::from(u8::try_from((rgb >> shift) & 0xFF).unwrap_or(255)) / 255.0);
        let mut mat = Material::plain(
            &self.m.layers[layer as usize].name,
            [channel(16), channel(8), channel(0), 1.0],
        );
        mat.double_sided = true;
        self.materials.push(mat);
        self.material_of.insert((layer, rgb), self.materials.len() - 1);
        self.materials.len() - 1
    }

    /// The mesh of space `s`'s faces, as context `ctx` colors them.
    fn mesh(&mut self, s: usize, ctx: Ctx) -> Option<usize> {
        let space = &self.m.spaces[s];
        let key = (
            s,
            if space.buckets.iter().any(|b| b.layer == self.layer0) {
                ctx.layer
            } else {
                u32::MAX
            },
            if space.buckets.iter().any(|b| b.spec == Spec::ByBlock) {
                ctx.rgb
            } else {
                u32::MAX
            },
        );
        if let Some(&m) = self.mesh_of.get(&key) {
            return m;
        }
        let looks: Vec<(usize, u32, u32)> = space
            .buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.tris.is_empty() || !b.lines.is_empty() || !b.points.is_empty())
            .filter_map(|(k, b)| {
                let layer = if b.layer == self.layer0 { ctx.layer } else { b.layer };
                (!self.m.layers[layer as usize].hidden).then(|| (k, layer, self.resolve(b.spec, layer, ctx)))
            })
            .collect();
        let mut prims = Vec::new();
        let mut tris = 0;
        for (k, layer, rgb) in looks {
            let material = self.material(layer, rgb);
            let bucket = &mut self.m.spaces[s].buckets[k];
            // Model space is drawn once, so its faces can be moved.
            let (positions, faces, lines, points) = if s == 0 {
                (
                    std::mem::take(&mut bucket.positions),
                    std::mem::take(&mut bucket.tris),
                    std::mem::take(&mut bucket.lines),
                    std::mem::take(&mut bucket.points),
                )
            } else {
                (
                    bucket.positions.clone(),
                    bucket.tris.clone(),
                    bucket.lines.clone(),
                    bucket.points.clone(),
                )
            };
            tris += faces.len();
            prims.push(Prim {
                positions,
                indices: faces.into_flattened(),
                lines: lines.into_flattened(),
                points,
                material,
                ..Prim::default()
            });
        }
        let mesh = if prims.is_empty() {
            None
        } else {
            self.b.mesh(prims, self.materials.len()).map(|mut mesh| {
                if s > 0 {
                    mesh["name"] = json!(self.m.spaces[s].name);
                }
                self.meshes.push(mesh);
                self.mesh_tris.push(tris);
                self.meshes.len() - 1
            })
        };
        self.mesh_of.insert(key, mesh);
        mesh
    }

    /// What space `s` draws: its mesh and a node for each block reference in it.
    fn content(&mut self, s: usize, ctx: Ctx, depth: usize) -> Result<(Option<usize>, Vec<usize>)> {
        let mesh = self.mesh(s, ctx);
        if let Some(m) = mesh {
            self.triangles += self.mesh_tris[m];
            ensure!(
                self.triangles <= self.m.limit,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                self.m.limit
            );
        }
        let mut children = Vec::new();
        for k in 0..self.insts[s].len() {
            let ins = self.insts[s][k];
            self.insert(ins, ctx, depth, &mut children)?;
        }
        Ok((mesh, children))
    }

    /// Nodes for block reference `ins`: one for each place in its array.
    fn insert(&mut self, ins: Inst, ctx: Ctx, depth: usize, out: &mut Vec<usize>) -> Result<()> {
        let Some(b) = ins.block else {
            self.m.notes.insert("a block reference names a block that isn't there");
            return Ok(());
        };
        if self.visiting.contains(&b) {
            self.m.notes.insert("a block that contains itself was drawn once");
            return Ok(());
        }
        if depth >= MAX_DEPTH {
            self.m.notes.insert("blocks nested too deep were left out");
            return Ok(());
        }
        let layer = if ins.layer == self.layer0 { ctx.layer } else { ins.layer };
        if self.m.layers[layer as usize].hidden {
            return Ok(());
        }
        let inner = Ctx {
            layer,
            rgb: self.resolve(ins.spec, layer, ctx),
        };
        let [ax, ay, az] = axes(ins.extrusion);
        let rotation = mul(
            from_axes(ax, ay, az),
            [0.0, 0.0, (ins.angle / 2.0).sin(), (ins.angle / 2.0).cos()],
        );
        let base = self.m.spaces[b].base;
        let s = ins.scale;
        let (sin, cos) = ins.angle.sin_cos();
        let finite = s
            .iter()
            .chain(&ins.at)
            .chain(&base)
            .chain(&rotation)
            .chain(&ins.gap)
            .all(|x| x.is_finite());
        if !finite || s.contains(&0.0) {
            self.m
                .notes
                .insert("block references that aren't finite or are scaled to nothing were left out");
            return Ok(());
        }
        let name = self.m.spaces[b].name.clone();
        self.visiting.push(b);
        'array: for r in 0..ins.rows {
            for c in 0..ins.cols {
                if self.nodes.len() >= MAX_NODES {
                    self.m
                        .notes
                        .insert("block references past the most nodes were left out");
                    break 'array;
                }
                // p ↦ OCS(at + Rz(offset + S(p − base))): the base point and the array's place
                // go in the translation.
                let local = [
                    f64::from(c) * ins.gap[0] - s[0] * base[0],
                    f64::from(r) * ins.gap[1] - s[1] * base[1],
                    -s[2] * base[2],
                ];
                let o = [
                    ins.at[0] + cos * local[0] - sin * local[1],
                    ins.at[1] + sin * local[0] + cos * local[1],
                    ins.at[2] + local[2],
                ];
                let t: [f64; 3] = std::array::from_fn(|k| ax[k] * o[0] + ay[k] * o[1] + az[k] * o[2]);
                let (mesh, children) = self.content(b, inner, depth + 1)?;
                if mesh.is_none() && children.is_empty() {
                    break 'array;
                }
                let mut node = json!({ "name": name });
                if t.iter().any(|&x| x != 0.0) {
                    node["translation"] = json!(t);
                }
                if rotation[3] < 1.0 - 1e-15 {
                    node["rotation"] = json!(rotation);
                }
                if s.iter().any(|&x| (x - 1.0).abs() > 1e-15) {
                    node["scale"] = json!(s);
                }
                if let Some(m) = mesh {
                    node["mesh"] = json!(m);
                }
                if !children.is_empty() {
                    node["children"] = json!(children);
                }
                self.nodes.push(node);
                out.push(self.nodes.len() - 1);
            }
        }
        self.visiting.pop();
        Ok(())
    }

    fn finish(mut self) -> Result<Document> {
        let top = Ctx {
            layer: self.layer0,
            rgb: WHITE,
        };
        let (mesh, children) = self.content(0, top, 0)?;
        let units = unit(self.m.units);
        let mut roots = Vec::new();
        if mesh.is_some() || !children.is_empty() {
            let h = std::f64::consts::FRAC_1_SQRT_2;
            let mut root = json!({ "name": "RootNode", "rotation": [-h, 0.0, 0.0, h] });
            if let Some((u, _)) = units.filter(|&(u, _)| (u - 1.0).abs() > 1e-12) {
                root["scale"] = json!([u, u, u]);
            }
            if let Some(m) = mesh {
                root["mesh"] = json!(m);
            }
            if !children.is_empty() {
                root["children"] = json!(children);
            }
            self.nodes.push(root);
            roots.push(self.nodes.len() - 1);
        }
        let mut source = json!({ "encoding": if self.binary { "binary" } else { "ASCII" } });
        if let Some(m) = self.m.measurement.filter(|m| matches!(m, 0 | 1)) {
            source["measurement"] = json!(if m == 0 { "imperial" } else { "metric" });
        }
        if !self.m.ended {
            source["truncated"] = json!(true);
        }
        if !self.m.skipped.is_empty() {
            let skipped: Map<String, Value> = self
                .m
                .skipped
                .iter()
                .map(|(k, &n)| (String::from_utf8_lossy(k).into_owned(), json!(n)))
                .collect();
            source["skipped"] = Value::Object(skipped);
        }
        if !self.m.notes.is_empty() {
            source["notes"] = json!(self.m.notes);
        }
        if let Value::Object(map) = &mut source {
            self.m.info(units, !roots.is_empty()).write(map);
        }
        let mut json = Map::new();
        json.insert("asset".into(), json!({ "extras": { "source": source } }));
        json.insert("scene".into(), json!(0));
        let scene = if roots.is_empty() {
            json!({})
        } else {
            json!({ "nodes": roots })
        };
        json.insert("scenes".into(), json!([scene]));
        if !self.nodes.is_empty() {
            json.insert("nodes".into(), Value::Array(self.nodes));
        }
        if !self.meshes.is_empty() {
            json.insert("meshes".into(), Value::Array(self.meshes));
        }
        self.b.materials(self.materials, Vec::new(), &mut json);
        let mut facts = Map::new();
        facts.insert("triangles".into(), json!(self.triangles));
        Ok(self.b.finish(json, facts))
    }
}

/// The points inside the arc from `a` to `b` of `bulge` (the tangent of a quarter of its
/// sweep, counterclockwise when positive), one for each sixteenth of a half turn it sweeps;
/// none for a straight segment.
fn bulge_arc(a: [f64; 2], b: [f64; 2], bulge: f64, out: &mut Vec<[f64; 2]>) {
    let d = [b[0] - a[0], b[1] - a[1]];
    if bulge.abs() < 1e-9 || !bulge.is_finite() || d == [0.0, 0.0] {
        return;
    }
    let sweep = 4.0 * bulge.atan();
    // The center is off the chord's middle along its left normal, by (1 - bulge²) / (4 bulge)
    // of the chord's length.
    let k = (1.0 - bulge * bulge) / (4.0 * bulge);
    let center = [
        f64::midpoint(a[0], b[0]) - d[1] * k,
        f64::midpoint(a[1], b[1]) + d[0] * k,
    ];
    let radius = (a[0] - center[0]).hypot(a[1] - center[1]);
    let start = (a[1] - center[1]).atan2(a[0] - center[0]);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a sweep under a turn, in steps of a 32nd of one"
    )]
    let n = (sweep.abs() / ARC_STEP - 1e-9).ceil().clamp(1.0, 32.0) as u32;
    for i in 1..n {
        let t = start + sweep * f64::from(i) / f64::from(n);
        out.push([center[0] + radius * t.cos(), center[1] + radius * t.sin()]);
    }
}

/// How far counterclockwise it is from `from` to `to`, in a `turn` (360 degrees or a turn's
/// radians): a whole turn when they're the same or aren't numbers.
fn turned(from: f64, to: f64, turn: f64) -> f64 {
    let sweep = (to - from).rem_euclid(turn);
    if sweep.is_finite() && sweep > turn * 1e-9 && sweep < turn * (1.0 - 1e-9) {
        sweep
    } else {
        turn
    }
}

/// The segments an arc of `sweep` (at most a turn) is drawn with, one for each `step`: 1 to 32.
fn segments(sweep: f64, step: f64) -> usize {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 1 to 32"
    )]
    let n = (sweep / step - 1e-9).ceil().clamp(1.0, 32.0) as usize;
    n.max(1)
}

/// The object coordinate system's axes for an extrusion (the arbitrary axis algorithm).
fn axes(extrusion: [f64; 3]) -> [[f64; 3]; 3] {
    let Some(n) = vec3::unit(extrusion) else {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    };
    let ax = if n[0].abs() < 1.0 / 64.0 && n[1].abs() < 1.0 / 64.0 {
        cross([0.0, 1.0, 0.0], n)
    } else {
        cross([0.0, 0.0, 1.0], n)
    };
    let ax = vec3::unit(ax).unwrap_or([1.0, 0.0, 0.0]);
    let ay = vec3::unit(cross(n, ax)).unwrap_or([0.0, 1.0, 0.0]);
    [ax, ay, n]
}

/// The rotation whose matrix has columns `x`, `y` and `z`, as a quaternion.
fn from_axes(x: [f64; 3], y: [f64; 3], z: [f64; 3]) -> [f64; 4] {
    let (m00, m11, m22) = (x[0], y[1], z[2]);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(y[2] - z[1]) / s, (z[0] - x[2]) / s, (x[1] - y[0]) / s, s / 4.0]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [s / 4.0, (y[0] + x[1]) / s, (z[0] + x[2]) / s, (y[2] - z[1]) / s]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [(y[0] + x[1]) / s, s / 4.0, (z[1] + y[2]) / s, (z[0] - x[2]) / s]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [(z[0] + x[2]) / s, (z[1] + y[2]) / s, s / 4.0, (x[1] - y[0]) / s]
    };
    let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    let q = q.map(|c| c / n);
    if q[3] < 0.0 { q.map(|c| -c) } else { q }
}

/// The quaternion product `a b` (`b` turned, then `a`).
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let q = [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
    if q[3] < 0.0 { q.map(|c| -c) } else { q }
}

#[cfg(test)]
mod tests;
