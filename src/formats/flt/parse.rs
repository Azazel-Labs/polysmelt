//! The record stream of one `OpenFlight` file, read into a [`Db`]: its header, palettes, vertices,
//! node tree and faces. Every record starts with a big-endian opcode and length (which counts
//! those four bytes); a continuation record (23) extends the one before it. Fields a short
//! record (an older version's) doesn't reach read as their defaults.

use std::borrow::Cow;

use serde_json::{Map, Value, json};

use crate::Result;
use crate::mesh::FastMap;
use crate::num::u32_of;

/// Opcodes this reader acts on.
mod op {
    pub const HEADER: u16 = 1;
    pub const GROUP: u16 = 2;
    pub const OLD_LOD: u16 = 3;
    pub const OBJECT: u16 = 4;
    pub const FACE: u16 = 5;
    pub const OLD_VERTEX: u16 = 6;
    pub const OLD_SHADED_VERTEX: u16 = 7;
    pub const OLD_NORMAL_VERTEX: u16 = 8;
    pub const PUSH: u16 = 10;
    pub const POP: u16 = 11;
    pub const DOF: u16 = 14;
    pub const PUSH_SUBFACE: u16 = 19;
    pub const POP_SUBFACE: u16 = 20;
    pub const PUSH_EXTENSION: u16 = 21;
    pub const POP_EXTENSION: u16 = 22;
    pub const CONTINUATION: u16 = 23;
    pub const COMMENT: u16 = 31;
    pub const COLOR_PALETTE: u16 = 32;
    pub const LONG_ID: u16 = 33;
    pub const MATRIX: u16 = 49;
    pub const MULTITEXTURE: u16 = 52;
    pub const UV_LIST: u16 = 53;
    pub const BSP: u16 = 55;
    pub const REPLICATE: u16 = 60;
    pub const INSTANCE_REF: u16 = 61;
    pub const INSTANCE_DEF: u16 = 62;
    pub const EXTERNAL_REF: u16 = 63;
    pub const TEXTURE_PALETTE: u16 = 64;
    pub const OLD_MATERIAL_PALETTE: u16 = 66;
    pub const VERTEX_PALETTE: u16 = 67;
    pub const VERTEX_C: u16 = 68;
    pub const VERTEX_CN: u16 = 69;
    pub const VERTEX_CNT: u16 = 70;
    pub const VERTEX_CT: u16 = 71;
    pub const VERTEX_LIST: u16 = 72;
    pub const LOD: u16 = 73;
    pub const MESH: u16 = 84;
    pub const LOCAL_VERTEX_POOL: u16 = 85;
    pub const MESH_PRIMITIVE: u16 = 86;
    pub const MORPH_VERTEX_LIST: u16 = 89;
    pub const SWITCH: u16 = 96;
    pub const EXTENSION: u16 = 100;
    pub const LIGHT_SOURCE: u16 = 101;
    pub const LIGHT_SOURCE_PALETTE: u16 = 102;
    pub const LIGHT_POINT: u16 = 111;
    pub const MATERIAL_PALETTE: u16 = 113;
    pub const PUSH_ATTRIBUTE: u16 = 122;
    pub const POP_ATTRIBUTE: u16 = 123;
    pub const INDEXED_LIGHT_POINT: u16 = 130;
    pub const LIGHT_POINT_SYSTEM: u16 = 131;
    pub const INDEXED_STRING: u16 = 132;
    /// A pop level written little-endian, as some exporters do.
    pub const POP_SWAPPED: u16 = 0x0B00;
}

/// Records that start a node (a "primary" record, which push and pop levels nest), beyond the
/// ones read for what they are: sound, road segments, paths and construction, text, clip
/// regions, curves and CAT records. They become empty nodes so their children stay in place.
const OTHER_PRIMARY: &[u16] = &[87, 91, 92, 95, 97, 98, 115, 126, 127];

/// The bits a vertex keeps of what it has.
pub(super) const V_NORMAL: u8 = 1;
pub(super) const V_UV: u8 = 2;
/// Its color is the packed one, not a palette index.
pub(super) const V_PACKED: u8 = 4;
/// Its color is the palette's at `index`.
pub(super) const V_INDEX: u8 = 8;
/// Its color index is a mesh's, the palette's entry in the low 24 bits (no intensity).
pub(super) const V_MESH_INDEX: u8 = 16;

/// Face flags, from the left: packed color, and hidden.
pub(super) const F_PACKED: u32 = 0x8000_0000 >> 3;
pub(super) const F_HIDDEN: u32 = 0x8000_0000 >> 5;
/// Group flags: forward animation, swing, and backward animation (15.8 on).
const G_FORWARD: u32 = 0x8000_0000 >> 1;
const G_SWING: u32 = 0x8000_0000 >> 2;
const G_BACKWARD: u32 = 0x8000_0000 >> 6;

/// The most levels push records nest; deeper ones are read as if at this level.
const MAX_LEVELS: usize = 4096;

#[inline]
fn arr<const N: usize>(d: &[u8], at: usize) -> Option<[u8; N]> {
    d.get(at..at.checked_add(N)?)?.try_into().ok()
}

#[inline]
pub(super) fn u8_at(d: &[u8], at: usize) -> u8 {
    d.get(at).copied().unwrap_or(0)
}

#[inline]
pub(super) fn u16_at(d: &[u8], at: usize) -> u16 {
    arr(d, at).map_or(0, u16::from_be_bytes)
}

#[inline]
pub(super) fn i16_or(d: &[u8], at: usize, default: i16) -> i16 {
    arr(d, at).map_or(default, i16::from_be_bytes)
}

#[inline]
pub(super) fn u32_at(d: &[u8], at: usize) -> u32 {
    arr(d, at).map_or(0, u32::from_be_bytes)
}

#[inline]
pub(super) fn i32_or(d: &[u8], at: usize, default: i32) -> i32 {
    arr(d, at).map_or(default, i32::from_be_bytes)
}

#[inline]
pub(super) fn f32_or(d: &[u8], at: usize, default: f32) -> f32 {
    arr(d, at)
        .map(f32::from_be_bytes)
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

#[inline]
pub(super) fn f64_or(d: &[u8], at: usize, default: f64) -> f64 {
    arr(d, at)
        .map(f64::from_be_bytes)
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn vec3d(d: &[u8], at: usize) -> [f64; 3] {
    [f64_or(d, at, 0.0), f64_or(d, at + 8, 0.0), f64_or(d, at + 16, 0.0)]
}

fn vec3f(d: &[u8], at: usize) -> [f32; 3] {
    [f32_or(d, at, 0.0), f32_or(d, at + 4, 0.0), f32_or(d, at + 8, 0.0)]
}

/// A packed color, stored as alpha, blue, green, red: its red, green and blue.
fn packed(d: &[u8], at: usize) -> [u8; 3] {
    [u8_at(d, at + 3), u8_at(d, at + 2), u8_at(d, at + 1)]
}

/// A fixed-size text field: up to its first NUL, Latin-1 where it isn't UTF-8.
pub(super) fn text(d: &[u8], at: usize, len: usize) -> String {
    let end = at.saturating_add(len).min(d.len());
    let b = d.get(at..end).unwrap_or_default();
    let b = &b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())];
    match std::str::from_utf8(b) {
        Ok(s) => s.trim_end().to_owned(),
        Err(_) => b
            .iter()
            .map(|&c| char::from(c))
            .collect::<String>()
            .trim_end()
            .to_owned(),
    }
}

/// A material palette entry.
#[derive(Clone, Debug)]
pub(super) struct Mat {
    pub name: String,
    pub flags: u32,
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    pub emissive: [f32; 3],
    pub shininess: f32,
    pub alpha: f32,
}

impl Default for Mat {
    /// The material of a face that names none, or one that isn't in the palette: white.
    fn default() -> Self {
        Self {
            name: String::new(),
            flags: 0,
            ambient: [1.0; 3],
            diffuse: [1.0; 3],
            specular: [0.0; 3],
            emissive: [0.0; 3],
            shininess: 0.0,
            alpha: 1.0,
        }
    }
}

/// A light source palette entry.
#[derive(Clone, Debug, Default)]
pub(super) struct LightPal {
    pub name: String,
    pub ambient: [f32; 4],
    pub diffuse: [f32; 4],
    pub specular: [f32; 4],
    /// 0 infinite, 1 local, 2 spot.
    pub kind: i32,
    pub spot_exponent: f32,
    pub spot_cutoff: f32,
    pub attenuation: [f32; 3],
}

/// A vertex of the vertex palette (or a mesh's local pool, or an old inline vertex).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Vert {
    pub pos: [f64; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Its packed color's red, green and blue.
    pub packed: [u8; 3],
    /// Its color palette index (with intensity, as the version stores it).
    pub index: u32,
    /// What it has: the `V_` bits.
    pub has: u8,
}

/// A degree-of-freedom node: its frame and its nine ranges (min, max, current, increment) for
/// z, y, x, pitch, roll, yaw, and z, y, x scale.
#[derive(Clone, Debug)]
pub(super) struct Dof {
    pub origin: [f64; 3],
    pub on_x: [f64; 3],
    pub in_xy: [f64; 3],
    pub ranges: [[f64; 4]; 9],
    pub flags: u32,
}

/// What kind of node.
#[derive(Clone, Debug)]
pub(super) enum Kind {
    /// The file's header: the top of its tree.
    Root,
    Group {
        flags: u32,
        loop_count: i32,
        loop_duration: f32,
        last_frame: f32,
    },
    Object {
        flags: u32,
        transparency: u16,
    },
    Lod {
        switch_in: f64,
        switch_out: f64,
        center: [f64; 3],
        flags: u32,
    },
    Dof(Box<Dof>),
    Switch {
        current: u32,
        words: u32,
        masks: Vec<u32>,
    },
    InstanceDef(u16),
    InstanceRef(u16),
    External {
        file: String,
        flags: u32,
    },
    Light {
        palette: i32,
        flags: u32,
        pos: [f64; 3],
        yaw: f32,
        pitch: f32,
    },
    LightPoint,
    Bsp,
    /// Another primary record, by opcode.
    Other(u16),
}

impl Kind {
    /// The record's name, for `extras`.
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Root => "header",
            Kind::Group { .. } => "group",
            Kind::Object { .. } => "object",
            Kind::Lod { .. } => "lod",
            Kind::Dof(_) => "dof",
            Kind::Switch { .. } => "switch",
            Kind::InstanceDef(_) => "instanceDefinition",
            Kind::InstanceRef(_) => "instanceReference",
            Kind::External { .. } => "externalReference",
            Kind::Light { .. } => "lightSource",
            Kind::LightPoint => "lightPoint",
            Kind::Bsp => "bsp",
            Kind::Other(87) => "roadSegment",
            Kind::Other(91) => "sound",
            Kind::Other(92) => "roadPath",
            Kind::Other(95) => "text",
            Kind::Other(97) => "cat",
            Kind::Other(98) => "clipRegion",
            Kind::Other(115) => "catData",
            Kind::Other(126) => "curve",
            Kind::Other(127) => "roadConstruction",
            Kind::Other(_) => "other",
        }
    }

    /// Whether this group plays its children in turn.
    pub fn sequence(&self) -> Option<Sequence> {
        let Kind::Group {
            flags,
            loop_duration,
            last_frame,
            ..
        } = *self
        else {
            return None;
        };
        (flags & (G_FORWARD | G_SWING | G_BACKWARD) != 0).then_some(Sequence {
            swing: flags & G_SWING != 0,
            backward: flags & G_BACKWARD != 0 && flags & G_FORWARD == 0,
            loop_duration,
            last_frame,
        })
    }
}

/// An animated group's playback.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sequence {
    pub swing: bool,
    pub backward: bool,
    pub loop_duration: f32,
    pub last_frame: f32,
}

/// A node of the tree.
#[derive(Clone, Debug)]
pub(super) struct Node {
    pub kind: Kind,
    pub name: String,
    pub kids: Vec<u32>,
    /// Faces (and mesh primitives, light point strings) drawn here.
    pub faces: Vec<u32>,
    /// The matrix record's, column-major.
    pub matrix: Option<[f64; 16]>,
    /// More copies the replicate record asks for, each moved once more by the matrix.
    pub replicate: u16,
    pub comment: Option<String>,
    /// Further `extras` for its record.
    pub extras: Map<String, Value>,
}

impl Node {
    fn new(kind: Kind, name: String) -> Self {
        Self {
            kind,
            name,
            kids: Vec::new(),
            faces: Vec::new(),
            matrix: None,
            replicate: 0,
            comment: None,
            extras: Map::new(),
        }
    }
}

/// How a face's corners make shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// One polygon (or line, by its draw type).
    Polygon,
    /// Triangles, three corners each (a mesh's strips and fans, unrolled).
    Triangles,
    /// Points (light points).
    Points,
}

/// A face, or a mesh primitive with its mesh's attributes, or a light point string.
#[derive(Clone, Debug)]
pub(super) struct Face {
    /// The node it's drawn in.
    pub node: u32,
    /// Its corners: `Db::corners[start..start + len]`, vertex indices.
    pub start: u32,
    pub len: u32,
    pub shape: Shape,
    pub draw: u8,
    pub template: u8,
    pub texwhite: bool,
    pub light_mode: u8,
    pub texture: i16,
    pub detail: i16,
    pub material: i16,
    pub transparency: u16,
    pub flags: u32,
    /// The palette color: before 15.1 the color name index, after it the primary color index.
    pub color: u32,
    pub packed: [u8; 3],
    /// Its texture layers 1 to 7 (multitexture): texture and effect.
    pub layer_tex: [(i16, i16); 7],
    /// The layers it has (bit k for layer k + 1).
    pub layer_mask: u8,
    /// Its corners' layer coordinates: `Db::layer_uvs[layers + k]`, or `u32::MAX`.
    pub layers: u32,
    /// Its corners' morph targets (the vertex at 100%): `Db::morph[morph + k]`, or `u32::MAX`.
    pub morph: u32,
    /// How many subface levels down it is.
    pub subface: u8,
    pub shader: i16,
}

impl Face {
    fn new(node: u32) -> Self {
        Self {
            node,
            start: 0,
            len: 0,
            shape: Shape::Polygon,
            draw: 1,
            template: 0,
            texwhite: false,
            light_mode: 0,
            texture: -1,
            detail: -1,
            material: -1,
            transparency: 0,
            flags: 0,
            color: u32::MAX,
            packed: [255; 3],
            layer_tex: [(-1, 0); 7],
            layer_mask: 0,
            layers: u32::MAX,
            morph: u32::MAX,
            subface: 0,
            shader: -1,
        }
    }
}

/// What a file holds that the model leaves out, counted.
#[derive(Clone, Debug, Default)]
pub(super) struct Counts {
    pub unknown_records: usize,
    pub bad_corners: usize,
    pub extensions: usize,
    pub subfaces: usize,
    pub billboards: usize,
    pub morph_faces: usize,
}

/// One `OpenFlight` file, read.
#[derive(Debug, Default)]
pub(super) struct Db {
    /// The format revision (1420 for 14.2, 1640 for 16.4; 13 or less for old files).
    pub version: i32,
    pub id: String,
    /// The header's units: 0 meters, 1 kilometers, 4 feet, 5 inches, 8 nautical miles.
    pub units: u8,
    /// The header's other fields, for `extras`.
    pub header: Map<String, Value>,
    pub colors: Vec<[u8; 3]>,
    /// The color palette is the old one (version 13 or less).
    pub old_colors: bool,
    pub materials: FastMap<i32, Mat>,
    pub textures: FastMap<i32, String>,
    pub lights: FastMap<i32, LightPal>,
    pub verts: Vec<Vert>,
    /// The vertex palette's vertices by their offset in it, in order.
    pub vert_at: Vec<(u32, u32)>,
    pub nodes: Vec<Node>,
    pub faces: Vec<Face>,
    pub corners: Vec<u32>,
    pub layer_uvs: Vec<[f32; 2]>,
    pub morph: Vec<u32>,
    pub instances: FastMap<u16, u32>,
    pub counts: Counts,
    /// The records ended inside one, or the levels didn't close.
    pub truncated: bool,
}

impl Db {
    /// The vertex at a vertex palette offset.
    fn vertex(&self, offset: u32) -> Option<u32> {
        self.vert_at
            .binary_search_by_key(&offset, |&(o, _)| o)
            .ok()
            .map(|k| self.vert_at[k].1)
    }

    /// The meters in one of the file's units.
    pub fn meters(&self) -> f64 {
        match self.units {
            1 => 1000.0,
            4 => 0.3048,
            5 => 0.0254,
            8 => 1852.0,
            _ => 1.0,
        }
    }

    /// The units' name; `None` for a code the format doesn't define.
    pub fn unit_name(&self) -> Option<&'static str> {
        Some(match self.units {
            0 => "meter",
            1 => "kilometer",
            4 => "foot",
            5 => "inch",
            8 => "nautical mile",
            _ => return None,
        })
    }

    /// The version as written: 16.4 for 1640, 15.4.1 for 1541.
    pub fn version_name(&self) -> String {
        let v = self.version;
        if v >= 1000 {
            let (major, minor, patch) = (v / 100, v / 10 % 10, v % 10);
            if patch == 0 {
                format!("{major}.{minor}")
            } else {
                format!("{major}.{minor}.{patch}")
            }
        } else {
            v.to_string()
        }
    }
}

/// Where the next record goes: the last primary record read.
#[derive(Clone, Copy, Debug)]
enum Cur {
    Node(u32),
    Face(u32),
    /// A mesh: its attributes, as a face the primitives copy.
    Mesh(u32),
}

/// A record: its opcode, its bytes from its header on, and its offset in the file.
type Record<'a> = (u16, Cow<'a, [u8]>, usize);

/// The records of `data`, each with its continuations joined on.
struct Records<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Records<'a> {
    /// The next record (opcode, bytes from its header on, offset in the file), or `None` at the
    /// end; `Err` when its length is broken.
    fn next(&mut self) -> Option<std::result::Result<Record<'a>, ()>> {
        let head = arr::<4>(self.d, self.at)?;
        let opcode = u16::from_be_bytes([head[0], head[1]]);
        let len = usize::from(u16::from_be_bytes([head[2], head[3]]));
        if len < 4 {
            return Some(Err(()));
        }
        let at = self.at;
        let end = at + len;
        let Some(body) = self.d.get(at..end) else {
            return Some(Err(()));
        };
        self.at = end;
        let mut rec = Cow::Borrowed(body);
        // Continuations add to the record's body.
        while let Some(h) = arr::<4>(self.d, self.at) {
            if u16::from_be_bytes([h[0], h[1]]) != op::CONTINUATION {
                break;
            }
            let n = usize::from(u16::from_be_bytes([h[2], h[3]]));
            let Some(more) = self.d.get(self.at + 4..self.at + n.max(4)) else {
                self.at = self.d.len();
                break;
            };
            rec.to_mut().extend_from_slice(more);
            self.at += n.max(4);
        }
        Some(Ok((opcode, rec, at)))
    }
}

/// Reads one `OpenFlight` file.
pub(super) fn parse(data: &[u8]) -> Result<Db> {
    ensure!(
        data.len() >= 4 && u16::from_be_bytes([data[0], data[1]]) == op::HEADER,
        UnknownFormat,
        "not an OpenFlight file"
    );
    let mut db = Db {
        version: 1640,
        ..Db::default()
    };
    db.nodes.push(Node::new(Kind::Root, String::new()));
    let mut records = Records { d: data, at: 0 };
    let mut cur = Cur::Node(0);
    let mut levels: Vec<Cur> = Vec::new();
    // Levels pushed past the limit, popped before the real ones.
    let mut deep = 0usize;
    let mut subface = 0u8;
    // The vertex palette's offset in the file.
    let mut palette_at: Option<usize> = None;
    // A mesh's local vertex pool: its first vertex, count, and layer coordinates.
    let mut pool: (u32, u32, Vec<[f32; 2]>, u8) = (0, 0, Vec::new(), 0);
    let mut first = true;

    while let Some(rec) = records.next() {
        let Ok((opcode, rec, at)) = rec else {
            db.truncated = true;
            break;
        };
        let d: &[u8] = &rec;
        if first {
            first = false;
            header(&mut db, d);
            continue;
        }
        // The node that a primary record read now joins.
        let parent = |db: &Db, levels: &[Cur]| -> u32 {
            match levels.last() {
                None => 0,
                Some(Cur::Node(n)) => *n,
                Some(Cur::Face(f) | Cur::Mesh(f)) => db.faces[*f as usize].node,
            }
        };
        match opcode {
            op::PUSH => {
                if levels.len() < MAX_LEVELS {
                    levels.push(cur);
                } else {
                    deep += 1;
                }
            }
            op::POP | op::POP_SWAPPED => {
                if deep > 0 {
                    deep -= 1;
                } else if let Some(c) = levels.pop() {
                    cur = c;
                }
            }
            op::PUSH_SUBFACE => {
                subface = subface.saturating_add(1);
                if levels.len() < MAX_LEVELS {
                    levels.push(cur);
                } else {
                    deep += 1;
                }
            }
            op::POP_SUBFACE => {
                subface = subface.saturating_sub(1);
                if deep > 0 {
                    deep -= 1;
                } else if let Some(c) = levels.pop() {
                    cur = c;
                }
            }
            op::PUSH_EXTENSION | op::PUSH_ATTRIBUTE => {
                // Skip to the matching pop: what's inside is another application's.
                let close = if opcode == op::PUSH_EXTENSION {
                    op::POP_EXTENSION
                } else {
                    op::POP_ATTRIBUTE
                };
                db.counts.extensions += 1;
                let mut depth = 1usize;
                while let Some(r) = records.next() {
                    let Ok((o, _, _)) = r else {
                        db.truncated = true;
                        break;
                    };
                    if o == opcode {
                        depth += 1;
                    } else if o == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
            }
            op::POP_EXTENSION | op::POP_ATTRIBUTE => {}
            op::GROUP => {
                let mut n = Node::new(
                    Kind::Group {
                        flags: u32_at(d, 16),
                        loop_count: i32_or(d, 32, 0),
                        loop_duration: f32_or(d, 36, 0.0),
                        last_frame: f32_or(d, 40, 0.0),
                    },
                    text(d, 4, 8),
                );
                put_nonzero(&mut n.extras, "priority", i64::from(i16_or(d, 12, 0)));
                put_nonzero(&mut n.extras, "significance", i64::from(i16_or(d, 24, 0)));
                put_nonzero(&mut n.extras, "layer", i64::from(u8_at(d, 26)));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::OBJECT => {
                let mut n = Node::new(
                    Kind::Object {
                        flags: u32_at(d, 12),
                        transparency: u16_at(d, 18),
                    },
                    text(d, 4, 8),
                );
                put_nonzero(&mut n.extras, "priority", i64::from(i16_or(d, 16, 0)));
                put_nonzero(&mut n.extras, "significance", i64::from(i16_or(d, 24, 0)));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::LOD | op::OLD_LOD => {
                let kind = if opcode == op::LOD {
                    Kind::Lod {
                        switch_in: f64_or(d, 16, 0.0),
                        switch_out: f64_or(d, 24, 0.0),
                        center: vec3d(d, 40),
                        flags: u32_at(d, 36),
                    }
                } else {
                    Kind::Lod {
                        switch_in: f64::from(u32_at(d, 12)),
                        switch_out: f64::from(u32_at(d, 16)),
                        center: [i32_or(d, 28, 0), i32_or(d, 32, 0), i32_or(d, 36, 0)].map(f64::from),
                        flags: u32_at(d, 24),
                    }
                };
                let mut n = Node::new(kind, text(d, 4, 8));
                if opcode == op::LOD && d.len() >= 72 {
                    n.extras.insert("transitionRange".into(), json!(f64_or(d, 64, 0.0)));
                }
                if opcode == op::LOD && d.len() >= 80 {
                    n.extras.insert("significantSize".into(), json!(f64_or(d, 72, 0.0)));
                }
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::DOF => {
                let mut ranges = [[0.0; 4]; 9];
                for (k, r) in ranges.iter_mut().enumerate() {
                    for (j, v) in r.iter_mut().enumerate() {
                        *v = f64_or(d, 88 + k * 32 + j * 8, 0.0);
                    }
                }
                let dof = Dof {
                    origin: vec3d(d, 16),
                    on_x: vec3d(d, 40),
                    in_xy: vec3d(d, 64),
                    ranges,
                    flags: u32_at(d, 376),
                };
                let n = Node::new(Kind::Dof(Box::new(dof)), text(d, 4, 8));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::SWITCH => {
                let current = u32_at(d, 16);
                let count = u32_at(d, 20) as usize;
                let words = u32_at(d, 24);
                let room = d.len().saturating_sub(28) / 4;
                let n = count.saturating_mul(words as usize).min(room);
                let masks = (0..n).map(|k| u32_at(d, 28 + k * 4)).collect();
                let node = Node::new(Kind::Switch { current, words, masks }, text(d, 4, 8));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, node, p)
                });
            }
            op::INSTANCE_DEF => {
                let number = u16_at(d, 6);
                // A definition is placed only by references to it.
                let i = u32_of(db.nodes.len());
                db.nodes
                    .push(Node::new(Kind::InstanceDef(number), format!("instance{number}")));
                db.instances.entry(number).or_insert(i);
                cur = Cur::Node(i);
            }
            op::INSTANCE_REF => {
                let number = u16_at(d, 6);
                let n = Node::new(Kind::InstanceRef(number), String::new());
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::EXTERNAL_REF => {
                let len = if db.version < 1400 { 80 } else { 200 };
                let file = text(d, 4, len);
                // Without flags, the file uses its own palettes.
                let flags = arr::<4>(d, 4 + len + 4).map_or(u32::MAX, u32::from_be_bytes);
                let n = Node::new(Kind::External { file, flags }, String::new());
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::LIGHT_SOURCE => {
                let n = Node::new(
                    Kind::Light {
                        palette: i32_or(d, 16, -1),
                        flags: u32_at(d, 24),
                        pos: vec3d(d, 32),
                        yaw: f32_or(d, 56, 0.0),
                        pitch: f32_or(d, 60, 0.0),
                    },
                    text(d, 4, 8),
                );
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::LIGHT_POINT | op::INDEXED_LIGHT_POINT => {
                let mut n = Node::new(Kind::LightPoint, text(d, 4, 8));
                if opcode == op::INDEXED_LIGHT_POINT {
                    n.extras.insert("appearance".into(), json!(i32_or(d, 12, -1)));
                    n.extras.insert("animation".into(), json!(i32_or(d, 16, -1)));
                } else {
                    n.extras.insert("displayMode".into(), json!(i32_or(d, 20, 0)));
                    n.extras.insert("intensity".into(), json!(f32_or(d, 24, 1.0)));
                }
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::LIGHT_POINT_SYSTEM => {
                let mut n = Node::new(Kind::Other(op::LIGHT_POINT_SYSTEM), text(d, 4, 8));
                n.extras.insert("intensity".into(), json!(f32_or(d, 12, 1.0)));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::BSP => {
                let n = Node::new(Kind::Bsp, text(d, 4, 8));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::EXTENSION => {
                db.counts.extensions += 1;
                let mut n = Node::new(Kind::Other(op::EXTENSION), text(d, 4, 8));
                n.extras.insert("site".into(), json!(text(d, 12, 8)));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            o if OTHER_PRIMARY.contains(&o) => {
                let n = Node::new(Kind::Other(o), text(d, 4, 8));
                cur = Cur::Node({
                    let p = parent(&db, &levels);
                    add(&mut db, n, p)
                });
            }
            op::FACE | op::MESH => {
                let node = parent(&db, &levels);
                let mut f = Face::new(node);
                // A mesh's fields sit four bytes further on than a face's.
                let s = if opcode == op::MESH { 4 } else { 0 };
                f.draw = d.get(18 + s).copied().unwrap_or(1);
                f.texwhite = u8_at(d, 19 + s) != 0;
                let name_index = u16_at(d, 20 + s);
                f.template = u8_at(d, 25 + s);
                f.detail = i16_or(d, 26 + s, -1);
                f.texture = i16_or(d, 28 + s, -1);
                f.material = i16_or(d, 30 + s, -1);
                f.transparency = u16_at(d, 40 + s);
                f.flags = u32_at(d, 44 + s);
                f.light_mode = u8_at(d, 48 + s);
                f.packed = packed(d, 56 + s);
                f.color = if db.version >= 1510 {
                    arr::<4>(d, 68 + s).map_or(u32::MAX, u32::from_be_bytes)
                } else {
                    u32::from(name_index)
                };
                f.shader = i16_or(d, 78 + s, -1);
                f.subface = subface;
                if subface > 0 {
                    db.counts.subfaces += 1;
                }
                if matches!(f.template, 2 | 4) {
                    db.counts.billboards += 1;
                }
                let i = u32_of(db.faces.len());
                db.faces.push(f);
                if opcode == op::MESH {
                    cur = Cur::Mesh(i);
                } else {
                    db.nodes[node as usize].faces.push(i);
                    cur = Cur::Face(i);
                }
            }
            op::VERTEX_PALETTE => palette_at = Some(at),
            op::VERTEX_C | op::VERTEX_CN | op::VERTEX_CNT | op::VERTEX_CT => {
                let Some(p) = palette_at else {
                    db.counts.unknown_records += 1;
                    continue;
                };
                let v = vertex(opcode, d);
                db.vert_at
                    .push((u32::try_from(at - p).unwrap_or(u32::MAX), u32_of(db.verts.len())));
                db.verts.push(v);
            }
            op::OLD_VERTEX | op::OLD_SHADED_VERTEX | op::OLD_NORMAL_VERTEX => {
                // Old files put vertices right after their face.
                let Cur::Face(f) = cur else { continue };
                let v = old_vertex(opcode, d);
                let vi = u32_of(db.verts.len());
                db.verts.push(v);
                let face = &mut db.faces[f as usize];
                if face.len == 0 {
                    face.start = u32_of(db.corners.len());
                }
                if face.start as usize + face.len as usize == db.corners.len() {
                    db.corners.push(vi);
                    face.len += 1;
                }
            }
            op::VERTEX_LIST | op::MORPH_VERTEX_LIST => {
                let morph = opcode == op::MORPH_VERTEX_LIST;
                let step = if morph { 8 } else { 4 };
                let f = match cur {
                    Cur::Face(f) => f,
                    Cur::Node(n) if matches!(db.nodes[n as usize].kind, Kind::LightPoint) => {
                        // A light point's vertices: a string of points.
                        let mut f = Face::new(n);
                        f.shape = Shape::Points;
                        f.light_mode = 1;
                        let i = u32_of(db.faces.len());
                        db.faces.push(f);
                        db.nodes[n as usize].faces.push(i);
                        cur = Cur::Face(i);
                        i
                    }
                    _ => {
                        db.counts.unknown_records += 1;
                        continue;
                    }
                };
                let start = u32_of(db.corners.len());
                let morph_start = u32_of(db.morph.len());
                let mut k = 4;
                while k + step <= d.len() {
                    let v = db.vertex(u32_at(d, k));
                    let m = if morph { db.vertex(u32_at(d, k + 4)) } else { v };
                    match (v, m) {
                        (Some(v), Some(m)) => {
                            db.corners.push(v);
                            if morph {
                                db.morph.push(m);
                            }
                        }
                        _ => db.counts.bad_corners += 1,
                    }
                    k += step;
                }
                let face = &mut db.faces[f as usize];
                if face.len > 0 {
                    // A second list for the same face: its corners stand alone, so start over.
                    db.counts.unknown_records += 1;
                }
                face.start = start;
                face.len = u32_of(db.corners.len()) - start;
                if morph {
                    face.morph = morph_start;
                    db.counts.morph_faces += 1;
                }
            }
            op::MULTITEXTURE => {
                let (Cur::Face(f) | Cur::Mesh(f)) = cur else { continue };
                let mask = u32_at(d, 4);
                let face = &mut db.faces[f as usize];
                let mut at = 8;
                for layer in 0..7 {
                    if mask & (0x8000_0000 >> layer) != 0 {
                        face.layer_tex[layer] = (i16_or(d, at, -1), i16_or(d, at + 2, 0));
                        face.layer_mask |= 1 << layer;
                        at += 8;
                    }
                }
            }
            op::UV_LIST => {
                let Cur::Face(f) = cur else { continue };
                let mask = u32_at(d, 4);
                let layers = (0..7).filter(|l| mask & (0x8000_0000 >> l) != 0).count();
                let face = &db.faces[f as usize];
                let n = face.len as usize;
                if layers == 0 || d.len() < 8 + n * layers * 8 {
                    db.counts.unknown_records += 1;
                    continue;
                }
                // Stored here as 7 layers a corner, the missing ones zero.
                let start = u32_of(db.layer_uvs.len());
                let mut at = 8;
                for _ in 0..n {
                    for l in 0..7 {
                        if mask & (0x8000_0000 >> l) != 0 {
                            db.layer_uvs.push([f32_or(d, at, 0.0), f32_or(d, at + 4, 0.0)]);
                            at += 8;
                        } else {
                            db.layer_uvs.push([0.0; 2]);
                        }
                    }
                }
                let face = &mut db.faces[f as usize];
                face.layers = start;
                // The record's mask says which layers have coordinates; a face without a
                // multitexture record still has them.
                face.layer_mask |= u8::try_from(mask >> 25).unwrap_or(0);
            }
            op::LOCAL_VERTEX_POOL => {
                let count = u32_at(d, 4);
                let mask = u32_at(d, 8);
                pool = local_pool(&mut db, d, count, mask);
            }
            op::MESH_PRIMITIVE => {
                let Cur::Mesh(m) = cur else { continue };
                mesh_primitive(&mut db, d, m, &pool);
            }
            op::LONG_ID => {
                let name = text(d, 4, d.len());
                if let Cur::Node(n) = cur {
                    db.nodes[n as usize].name = name;
                }
            }
            op::COMMENT => {
                if let Cur::Node(n) = cur {
                    let c = text(d, 4, d.len());
                    db.nodes[n as usize].comment = Some(c);
                }
            }
            op::MATRIX => {
                if let Cur::Node(n) = cur
                    && d.len() >= 68
                {
                    let mut m = [0.0; 16];
                    for (k, v) in m.iter_mut().enumerate() {
                        *v = f64::from(f32_or(d, 4 + k * 4, 0.0));
                    }
                    db.nodes[n as usize].matrix = Some(m);
                }
            }
            op::REPLICATE => {
                if let Cur::Node(n) = cur {
                    db.nodes[n as usize].replicate = u16::try_from(i16_or(d, 4, 0)).unwrap_or(0);
                }
            }
            op::INDEXED_STRING => {
                // A switch's mask names, in order.
                if let Cur::Node(n) = cur
                    && matches!(db.nodes[n as usize].kind, Kind::Switch { .. })
                {
                    let names = db.nodes[n as usize]
                        .extras
                        .entry("maskNames")
                        .or_insert_with(|| json!({}));
                    if let Some(o) = names.as_object_mut()
                        && o.len() < 1024
                    {
                        o.insert(u32_at(d, 4).to_string(), json!(text(d, 8, d.len())));
                    }
                }
            }
            op::COLOR_PALETTE => color_palette(&mut db, d),
            op::MATERIAL_PALETTE => {
                let m = Mat {
                    name: text(d, 8, 12),
                    flags: u32_at(d, 20),
                    ambient: vec3f(d, 24),
                    diffuse: vec3f(d, 36),
                    specular: vec3f(d, 48),
                    emissive: vec3f(d, 60),
                    shininess: f32_or(d, 72, 0.0),
                    alpha: f32_or(d, 76, 1.0),
                };
                db.materials.insert(i32_or(d, 4, 0), m);
            }
            op::OLD_MATERIAL_PALETTE => {
                // 64 materials of 3 colors, shininess, alpha, flags, a name and reserved space.
                const SIZE: usize = 4 * 12 + 4 + 4 + 4 + 12 + 28 * 4;
                for k in 0..64usize {
                    let b = 4 + k * SIZE;
                    if b + 76 > d.len() {
                        break;
                    }
                    let m = Mat {
                        ambient: vec3f(d, b),
                        diffuse: vec3f(d, b + 12),
                        specular: vec3f(d, b + 24),
                        emissive: vec3f(d, b + 36),
                        shininess: f32_or(d, b + 48, 0.0),
                        alpha: f32_or(d, b + 52, 1.0),
                        flags: u32_at(d, b + 56),
                        name: text(d, b + 60, 12),
                    };
                    db.materials.insert(i32::try_from(k).unwrap_or(0), m);
                }
            }
            op::TEXTURE_PALETTE => {
                let len = if db.version < 1400 { 80 } else { 200 };
                db.textures.insert(i32_or(d, 4 + len, 0), text(d, 4, len));
            }
            op::LIGHT_SOURCE_PALETTE => {
                let v4 = |at: usize| [0, 4, 8, 12].map(|k| f32_or(d, at + k, 0.0));
                let l = LightPal {
                    name: text(d, 16, 20),
                    ambient: v4(40),
                    diffuse: v4(56),
                    specular: v4(72),
                    kind: i32_or(d, 88, 0),
                    spot_exponent: f32_or(d, 132, 0.0),
                    spot_cutoff: f32_or(d, 136, 180.0),
                    attenuation: [f32_or(d, 148, 1.0), f32_or(d, 152, 0.0), f32_or(d, 156, 0.0)],
                };
                db.lights.insert(i32_or(d, 4, -1), l);
            }
            _ => db.counts.unknown_records += 1,
        }
    }
    if !levels.is_empty() || deep > 0 {
        // Unclosed levels are common at the end of a file; only the header's own is expected.
        db.truncated |= levels.len() > 1;
    }
    Ok(db)
}

fn put_nonzero(m: &mut Map<String, Value>, key: &str, v: i64) {
    if v != 0 {
        m.insert(key.into(), json!(v));
    }
}

/// Adds a node under `parent`; returns its index.
fn add(db: &mut Db, n: Node, parent: u32) -> u32 {
    let i = u32_of(db.nodes.len());
    db.nodes.push(n);
    db.nodes[parent as usize].kids.push(i);
    i
}

/// The header record: the version, units and placement on the earth.
fn header(db: &mut Db, d: &[u8]) {
    db.id = text(d, 4, 8);
    db.version = i32_or(d, 12, 1640);
    db.units = u8_at(d, 62);
    db.nodes[0].name.clone_from(&db.id);
    let h = &mut db.header;
    let date = text(d, 20, 32);
    if !date.is_empty() {
        h.insert("date".into(), json!(date));
    }
    h.insert("editRevision".into(), json!(i32_or(d, 16, 0)));
    let flags = u32_at(d, 64);
    if flags != 0 {
        h.insert("flags".into(), json!(flags));
    }
    if d.len() >= 96 {
        h.insert("projection".into(), json!(i32_or(d, 92, 0)));
    }
    if d.len() >= 132 {
        h.insert("databaseOrigin".into(), json!(i32_or(d, 128, 0)));
    }
    if d.len() >= 164 {
        h.insert("southwestXY".into(), json!([f64_or(d, 132, 0.0), f64_or(d, 140, 0.0)]));
        h.insert("deltaXY".into(), json!([f64_or(d, 148, 0.0), f64_or(d, 156, 0.0)]));
    }
    if d.len() >= 236 {
        h.insert(
            "southwestLatLong".into(),
            json!([f64_or(d, 188, 0.0), f64_or(d, 196, 0.0)]),
        );
        h.insert(
            "northeastLatLong".into(),
            json!([f64_or(d, 204, 0.0), f64_or(d, 212, 0.0)]),
        );
        h.insert(
            "originLatLong".into(),
            json!([f64_or(d, 220, 0.0), f64_or(d, 228, 0.0)]),
        );
    }
    if d.len() >= 272 {
        h.insert("ellipsoid".into(), json!(i32_or(d, 268, 0)));
    }
    if d.len() >= 278 {
        h.insert("utmZone".into(), json!(i16_or(d, 276, 0)));
    }
    if d.len() >= 292 {
        h.insert("deltaZ".into(), json!(f64_or(d, 284, 0.0)));
    }
}

/// The color palette: 1024 colors (512 before 15.1, as many as the record holds), or the old
/// 32 variable and 56 fixed intensity colors (version 13 or less).
fn color_palette(db: &mut Db, d: &[u8]) {
    db.colors.clear();
    if db.version <= 13 {
        db.old_colors = true;
        for k in 0..88 {
            let at = 4 + k * 6;
            if at + 6 > d.len() {
                break;
            }
            let c = |o: usize| u8::try_from(u16_at(d, at + o).min(255)).unwrap_or(255);
            db.colors.push([c(0), c(2), c(4)]);
        }
        return;
    }
    let most = if db.version >= 1510 { 1024 } else { 512 };
    let n = (d.len().saturating_sub(132) / 4).min(most);
    db.colors.extend((0..n).map(|k| packed(d, 132 + k * 4)));
}

/// A vertex palette record.
fn vertex(opcode: u16, d: &[u8]) -> Vert {
    let flags = u16_at(d, 6);
    let mut v = Vert {
        pos: vec3d(d, 8),
        ..Vert::default()
    };
    let (color_at, normal, uv) = match opcode {
        op::VERTEX_C => (32, None, None),
        op::VERTEX_CN => (44, Some(32), None),
        op::VERTEX_CNT => (52, Some(32), Some(44)),
        _ => (40, None, Some(32)),
    };
    if let Some(at) = normal {
        v.normal = vec3f(d, at);
        v.has |= V_NORMAL;
    }
    if let Some(at) = uv {
        v.uv = [f32_or(d, at, 0.0), f32_or(d, at + 4, 0.0)];
        v.has |= V_UV;
    }
    v.packed = packed(d, color_at);
    let index = i32_or(d, color_at + 4, -1);
    if flags & 0x1000 != 0 {
        v.has |= V_PACKED;
    } else if flags & 0x2000 == 0
        && let Ok(i) = u32::try_from(index)
    {
        v.index = i;
        v.has |= V_INDEX;
    }
    v
}

/// An old inline vertex (before version 14): integer coordinates, maybe a normal (scaled by
/// 2^30) and texture coordinates.
fn old_vertex(opcode: u16, d: &[u8]) -> Vert {
    let mut v = Vert {
        pos: [i32_or(d, 4, 0), i32_or(d, 8, 0), i32_or(d, 12, 0)].map(f64::from),
        ..Vert::default()
    };
    let uv_at = match opcode {
        op::OLD_VERTEX => 16,
        op::OLD_SHADED_VERTEX => {
            if let Ok(i) = u32::try_from(i16_or(d, 18, -1)) {
                v.index = i;
                v.has |= V_INDEX;
            }
            20
        }
        _ => {
            if let Ok(i) = u32::try_from(i16_or(d, 18, -1)) {
                v.index = i;
                v.has |= V_INDEX;
            }
            let scale = f64::from(1u32 << 30);
            #[allow(clippy::cast_possible_truncation, reason = "a direction, single precision")]
            let n = if d.len() >= 44 {
                vec3d(d, 20).map(|c| (c / scale) as f32)
            } else {
                [i32_or(d, 20, 0), i32_or(d, 24, 0), i32_or(d, 28, 0)].map(|c| (f64::from(c) / scale) as f32)
            };
            v.normal = n;
            v.has |= V_NORMAL;
            if d.len() >= 44 { 44 } else { 32 }
        }
    };
    if d.len() >= uv_at + 8 {
        v.uv = [f32_or(d, uv_at, 0.0), f32_or(d, uv_at + 4, 0.0)];
        v.has |= V_UV;
    }
    v
}

/// A mesh's local vertex pool, added to the file's vertices: (first vertex, count, the
/// vertices' layer coordinates, 7 a vertex, and which layers there are).
fn local_pool(db: &mut Db, d: &[u8], count: u32, mask: u32) -> (u32, u32, Vec<[f32; 2]>, u8) {
    let bit = |k: u32| mask & (0x8000_0000 >> k) != 0;
    let mut stride = 0usize;
    for (k, size) in [(0, 24), (1, 4), (2, 4), (3, 12), (4, 8)] {
        if bit(k) {
            stride += size;
        }
    }
    let layer_mask: u8 = (0..7u8).filter(|&l| bit(5 + u32::from(l))).fold(0, |m, l| m | (1 << l));
    stride += 8 * layer_mask.count_ones() as usize;
    let first = u32_of(db.verts.len());
    if stride == 0 {
        return (first, 0, Vec::new(), 0);
    }
    let n = (count as usize).min(d.len().saturating_sub(12) / stride);
    db.verts.reserve(n);
    let mut layers = Vec::new();
    if layer_mask != 0 {
        layers.reserve(n * 7);
    }
    let mut at = 12;
    for _ in 0..n {
        let mut v = Vert::default();
        if bit(0) {
            v.pos = vec3d(d, at);
            at += 24;
        }
        if bit(1) {
            let ci = u32_at(d, at);
            v.index = ci & 0x00FF_FFFF;
            v.has |= V_INDEX | V_MESH_INDEX;
            at += 4;
        }
        if bit(2) {
            v.packed = packed(d, at);
            v.has |= V_PACKED;
            at += 4;
        }
        if bit(3) {
            v.normal = vec3f(d, at);
            v.has |= V_NORMAL;
            at += 12;
        }
        if bit(4) {
            v.uv = [f32_or(d, at, 0.0), f32_or(d, at + 4, 0.0)];
            v.has |= V_UV;
            at += 8;
        }
        if layer_mask != 0 {
            for l in 0..7 {
                if layer_mask & (1 << l) != 0 {
                    layers.push([f32_or(d, at, 0.0), f32_or(d, at + 4, 0.0)]);
                    at += 8;
                } else {
                    layers.push([0.0; 2]);
                }
            }
        }
        db.verts.push(v);
    }
    (first, u32_of(n), layers, layer_mask)
}

/// A mesh primitive: a strip, fan, quad strip or polygon of the local pool's vertices, as a
/// face with the mesh's attributes.
fn mesh_primitive(db: &mut Db, d: &[u8], mesh: u32, pool: &(u32, u32, Vec<[f32; 2]>, u8)) {
    let kind = i16_or(d, 4, 0);
    let size = usize::from(u16_at(d, 6));
    let count = u32_at(d, 8) as usize;
    if !matches!(size, 1 | 2 | 4) {
        db.counts.unknown_records += 1;
        return;
    }
    let n = count.min(d.len().saturating_sub(12) / size);
    let (first, len, layers, layer_mask) = pool;
    let mut idx: Vec<u32> = Vec::with_capacity(n);
    for k in 0..n {
        let at = 12 + k * size;
        let i = match size {
            1 => u32::from(u8_at(d, at)),
            2 => u32::from(u16_at(d, at)),
            _ => u32_at(d, at),
        };
        if i < *len {
            idx.push(i);
        } else {
            db.counts.bad_corners += 1;
        }
    }
    let mut corners: Vec<u32> = Vec::new();
    let shape = match kind {
        // A triangle strip: every other triangle turned to keep the winding.
        1 => {
            for k in 2..idx.len() {
                if k % 2 == 0 {
                    corners.extend([idx[k - 2], idx[k - 1], idx[k]]);
                } else {
                    corners.extend([idx[k - 1], idx[k - 2], idx[k]]);
                }
            }
            Shape::Triangles
        }
        2 => {
            for k in 2..idx.len() {
                corners.extend([idx[0], idx[k - 1], idx[k]]);
            }
            Shape::Triangles
        }
        // A quad strip: each pair of pairs a quad, as two triangles.
        3 => {
            let mut k = 2;
            while k + 1 < idx.len() {
                let (a, b, c, e) = (idx[k - 2], idx[k - 1], idx[k + 1], idx[k]);
                corners.extend([a, b, c, a, c, e]);
                k += 2;
            }
            Shape::Triangles
        }
        4 => {
            corners = idx;
            Shape::Polygon
        }
        _ => {
            db.counts.unknown_records += 1;
            return;
        }
    };
    if corners.is_empty() {
        return;
    }
    let mut f = db.faces[mesh as usize].clone();
    f.shape = shape;
    f.start = u32_of(db.corners.len());
    f.len = u32_of(corners.len());
    if *layer_mask != 0 {
        f.layers = u32_of(db.layer_uvs.len());
        f.layer_mask |= layer_mask;
        for &c in &corners {
            let at = c as usize * 7;
            db.layer_uvs.extend_from_slice(&layers[at..at + 7]);
        }
    }
    db.corners.extend(corners.iter().map(|&c| first + c));
    let node = f.node;
    let i = u32_of(db.faces.len());
    db.faces.push(f);
    db.nodes[node as usize].faces.push(i);
}
