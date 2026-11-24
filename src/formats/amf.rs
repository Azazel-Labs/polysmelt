//! AMF, the additive manufacturing file format (ISO/ASTM 52915): XML, as it is or zipped. Read here:
//! objects (a mesh each, a primitive per volume and texture), their vertices' normals and colors, the
//! colors of triangles, volumes, objects and materials (in that order of precedence, as the format
//! asks), materials' metadata and the colors of composite materials mixed by their proportions, 2D
//! textures (gray channels combined into one image) with each triangle's texture coordinates, and
//! constellations, placed as nodes wherever an instance puts them. The model's Z-up unit (millimeters
//! when unsaid) turns into glTF's Y-up meters at a root node. The source's units are declared when the
//! `unit` attribute names one the format knows, and millimeters assumed otherwise; its up axis is +Z
//! by the format's convention (assumed). The model's metadata of the kinds the standard names (`name`,
//! `description`, `author`, `cad`) and the common others (`copyright`, `license`, creation and
//! modification dates) are the source's title, description, author, tool and so on; the rest stay in
//! its `metadata`.
//!
//! Dropped, and counted in the source extras: the curved edges of curved triangles (their flat
//! triangles stay), colors and proportions given as formulas, 3D textures past their first layer,
//! texture coordinates' third (w) axis, triangles whose corners aren't vertices, and instances that
//! lead back to a constellation they're inside. Metadata of the model, objects, volumes, materials,
//! constellations and instances go in extras.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};

use quick_xml::events::{BytesStart, Event};
use serde_json::{Map, Value, json};

use crate::document::{Builder, Document};
use crate::images::Source;
use crate::material::{Alpha, CLAMP, Full, REPEAT, Tex};
use crate::mesh::{GRAY, Image, Material, Prim, srgb_to_linear};
use crate::source::{Axis, Fact, SourceInfo};
use crate::{Context as _, Result};

/// The most unzipped from a zipped AMF: more is a zip bomb, not a model.
const MAX_PART: u64 = 2 << 30;
/// The most nodes placed: nested constellations multiply, and past this they place nothing new.
const MAX_NODES: usize = 1 << 16;
/// How deep constellations nest inside each other.
const MAX_NESTING: usize = 64;
/// The most bytes of one texture's pixels.
const MAX_TEXTURE: usize = 1 << 28;
/// Marks a vertex without a normal or color, in arrays that hold them.
const NONE: f32 = f32::NAN;

/// Loads an AMF: an object's triangles as a mesh, constellations as nodes, and the facts worth keeping.
pub(crate) fn load(input: super::Input<'_>) -> Result<Document> {
    let zipped = input.head(4)?.starts_with(b"PK\x03\x04");
    let amf = if zipped {
        let mut zip = zip::ZipArchive::new(input.seekable()?).context("this zipped AMF can't be opened")?;
        let names: Vec<String> = zip
            .file_names()
            .filter_map(Result::ok)
            .map(std::borrow::Cow::into_owned)
            .collect();
        let name = names
            .iter()
            .find(|n| n.to_ascii_lowercase().ends_with(".amf"))
            .or_else(|| names.iter().find(|n| !n.ends_with('/')))
            .context("this zip holds no AMF")?
            .clone();
        let entry = zip
            .by_name(&name)
            .with_context(|| format!("reading {name}"))?
            .take(MAX_PART);
        parse(BufReader::with_capacity(1 << 16, entry), input.limit).with_context(|| format!("reading {name}"))?
    } else {
        parse(BufReader::with_capacity(1 << 16, input.seekable()?), input.limit)?
    };
    build(&amf, zipped)
}

/// An object: its vertices and its volumes.
#[derive(Default)]
struct Object {
    id: String,
    name: Option<String>,
    metadata: Map<String, Value>,
    color: Option<[f32; 4]>,
    verts: Vec<[f32; 3]>,
    /// Each vertex's normal, [`NONE`] for one without; as long as the last vertex with one.
    normals: Vec<[f32; 3]>,
    /// Each vertex's sRGB color, as `normals`.
    colors: Vec<[f32; 4]>,
    volumes: Vec<Volume>,
}

/// A volume: triangles of one material.
#[derive(Default)]
struct Volume {
    material: Option<String>,
    name: Option<String>,
    metadata: Map<String, Value>,
    color: Option<[f32; 4]>,
    tris: Vec<[u32; 3]>,
    /// Each triangle's sRGB color, as [`Object::normals`].
    colors: Vec<[f32; 4]>,
    /// Each triangle's texture map in `maps` (`u32::MAX` for none), as [`Object::normals`].
    map_of: Vec<u32>,
    maps: Vec<Texmap>,
}

/// A triangle's textures (red, green, blue and alpha: indices into [`Amf::texture_ids`], `u32::MAX` for none)
/// and its corners' texture coordinates.
#[derive(Clone, Copy)]
struct Texmap {
    tex: [u32; 4],
    uv: [[f32; 2]; 3],
}

#[derive(Default)]
struct Mat {
    id: String,
    name: Option<String>,
    metadata: Map<String, Value>,
    color: Option<[f32; 4]>,
    /// The materials mixed in and their proportions (`None` for a formula).
    composites: Vec<(String, Option<f32>)>,
}

/// A texture: gray pixels, `width` by `height` by `depth`, x fastest.
struct Texture {
    id: String,
    width: u32,
    height: u32,
    depth: u32,
    tiled: bool,
    pixels: Vec<u8>,
}

#[derive(Default)]
struct Constellation {
    id: String,
    name: Option<String>,
    metadata: Map<String, Value>,
    instances: Vec<Instance>,
}

/// An object or constellation placed: moved by `delta` after turning about x, y and z (in degrees, in that order).
#[derive(Default)]
struct Instance {
    target: String,
    delta: [f64; 3],
    turn: [f64; 3],
    metadata: Map<String, Value>,
}

/// The model, as read.
#[derive(Default)]
struct Amf {
    unit: Option<String>,
    version: Option<String>,
    metadata: Map<String, Value>,
    objects: Vec<Object>,
    materials: Vec<Mat>,
    textures: Vec<Texture>,
    constellations: Vec<Constellation>,
    /// The texture ids triangles name, each once.
    texture_ids: Vec<String>,
    /// What's dropped.
    curved: usize,
    formulas: usize,
    bad_textures: usize,
    /// The XML ended inside an element.
    truncated: bool,
    vertices: usize,
    triangles: usize,
}

/// The elements read; any other is skipped over.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tag {
    Amf,
    Object,
    Vertex,
    Coordinates,
    Normal,
    Color,
    Volume,
    Triangle,
    Texmap,
    Material,
    Composite,
    Texture,
    Constellation,
    Instance,
    Metadata,
    Edge,
    Leaf(Leaf, usize),
    Other,
}

/// Elements holding one number, and which of three or four they are.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leaf {
    Coord,
    Normal,
    Channel,
    Corner,
    U,
    V,
    Delta,
    Turn,
}

fn tag(name: &[u8]) -> Tag {
    use Leaf::{Channel, Coord, Corner, Delta, Normal, Turn, U, V};
    let leaf = |l, k| Tag::Leaf(l, k);
    match name {
        b"amf" => Tag::Amf,
        b"object" => Tag::Object,
        b"vertex" => Tag::Vertex,
        b"coordinates" => Tag::Coordinates,
        b"normal" => Tag::Normal,
        b"color" => Tag::Color,
        b"volume" => Tag::Volume,
        b"triangle" => Tag::Triangle,
        b"texmap" => Tag::Texmap,
        b"material" => Tag::Material,
        b"composite" => Tag::Composite,
        b"texture" => Tag::Texture,
        b"constellation" => Tag::Constellation,
        b"instance" => Tag::Instance,
        b"metadata" => Tag::Metadata,
        b"edge" => Tag::Edge,
        b"x" => leaf(Coord, 0),
        b"y" => leaf(Coord, 1),
        b"z" => leaf(Coord, 2),
        b"nx" => leaf(Normal, 0),
        b"ny" => leaf(Normal, 1),
        b"nz" => leaf(Normal, 2),
        b"r" => leaf(Channel, 0),
        b"g" => leaf(Channel, 1),
        b"b" => leaf(Channel, 2),
        b"a" => leaf(Channel, 3),
        b"v1" => leaf(Corner, 0),
        b"v2" => leaf(Corner, 1),
        b"v3" => leaf(Corner, 2),
        b"utex1" => leaf(U, 0),
        b"utex2" => leaf(U, 1),
        b"utex3" => leaf(U, 2),
        b"vtex1" => leaf(V, 0),
        b"vtex2" => leaf(V, 1),
        b"vtex3" => leaf(V, 2),
        b"deltax" => leaf(Delta, 0),
        b"deltay" => leaf(Delta, 1),
        b"deltaz" => leaf(Delta, 2),
        b"rx" => leaf(Turn, 0),
        b"ry" => leaf(Turn, 1),
        b"rz" => leaf(Turn, 2),
        _ => Tag::Other,
    }
}

impl Tag {
    /// Whether its text is kept.
    fn has_text(self) -> bool {
        matches!(self, Tag::Leaf(..) | Tag::Composite | Tag::Texture | Tag::Metadata)
    }
}

/// What's being read: the elements open, and the text of the innermost.
#[derive(Default)]
struct State {
    text: Vec<u8>,
    object: Option<Object>,
    volume: Option<Volume>,
    material: Option<Mat>,
    constellation: Option<Constellation>,
    instance: Option<Instance>,
    texture: Option<Texture>,
    position: [f32; 3],
    normal: [Option<f32>; 3],
    vertex_normal: Option<[f32; 3]>,
    vertex_color: Option<[f32; 4]>,
    channels: [Option<f32>; 4],
    formula: bool,
    corners: [u32; 3],
    tri_color: Option<[f32; 4]>,
    map: Option<Texmap>,
    tri_map: Option<Texmap>,
    composite: Option<String>,
    meta: String,
    texture_of: HashMap<String, u32>,
}

fn parse(r: impl BufRead, limit: usize) -> Result<Amf> {
    let mut reader = quick_xml::Reader::from_reader(crate::formats::xml::Utf8Read::new(r)?);
    let mut buf = Vec::new();
    let mut a = Amf::default();
    let mut s = State::default();
    let mut stack: Vec<Tag> = Vec::new();
    let mut root = false;
    loop {
        let at = reader.buffer_position();
        let event = match reader.read_event_into(&mut buf) {
            // A tag, comment or CDATA the input ends inside: the file was cut short there.
            Err(quick_xml::Error::Syntax(e)) if root && e != quick_xml::errors::SyntaxError::InvalidBangMarkup => {
                Event::Eof
            }
            event => event.with_context(|| format!("bad XML near byte {at}"))?,
        };
        match event {
            Event::Start(e) => {
                let t = tag(e.local_name().as_ref());
                ensure!(root || t == Tag::Amf, "not an AMF file (it doesn't start with <amf>)");
                root = true;
                s.text.clear();
                open(&mut a, &mut s, t, &e);
                stack.push(t);
            }
            Event::Empty(e) => {
                let t = tag(e.local_name().as_ref());
                ensure!(root || t == Tag::Amf, "not an AMF file (it doesn't start with <amf>)");
                root = true;
                s.text.clear();
                open(&mut a, &mut s, t, &e);
                close(&mut a, &mut s, t, stack.last().copied(), limit)?;
            }
            Event::End(_) => {
                if let Some(t) = stack.pop() {
                    close(&mut a, &mut s, t, stack.last().copied(), limit)?;
                }
            }
            Event::Text(t) => {
                if stack.last().is_some_and(|t| t.has_text()) {
                    s.text.extend_from_slice(&t);
                }
            }
            Event::CData(t) => {
                if stack.last().is_some_and(|t| t.has_text()) {
                    s.text.extend_from_slice(&t);
                }
            }
            Event::GeneralRef(r) => {
                if stack.last().is_some_and(|t| t.has_text()) {
                    reference(&r, &mut s.text);
                }
            }
            Event::Eof => {
                a.truncated = !stack.is_empty();
                // Keep the open volume, object or constellation, without the part cut short.
                while let Some(t) = stack.pop() {
                    if matches!(t, Tag::Volume | Tag::Object | Tag::Constellation) {
                        close(&mut a, &mut s, t, stack.last().copied(), limit)?;
                    }
                }
                break;
            }
            _ => {}
        }
        buf.clear();
    }
    ensure!(root, "not an AMF file (it has no <amf>)");
    Ok(a)
}

/// An entity or character reference's text.
fn reference(name: &[u8], out: &mut Vec<u8>) {
    let fixed: &[u8] = match name {
        b"amp" => b"&",
        b"lt" => b"<",
        b"gt" => b">",
        b"quot" => b"\"",
        b"apos" => b"'",
        _ => {
            let code = match name {
                [b'#', b'x' | b'X', hex @ ..] => std::str::from_utf8(hex)
                    .ok()
                    .and_then(|h| u32::from_str_radix(h, 16).ok()),
                [b'#', dec @ ..] => std::str::from_utf8(dec).ok().and_then(|d| d.parse().ok()),
                _ => None,
            };
            if let Some(c) = code.and_then(char::from_u32) {
                out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
            }
            return;
        }
    };
    out.extend_from_slice(fixed);
}

fn attr(e: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    e.attributes()
        .with_checks(false)
        .flatten()
        .find(|a| a.key.local_name().as_ref() == name)
        .map(|a| String::from_utf8_lossy(&a.value).trim().to_owned())
}

/// A finite number.
fn number(text: &[u8]) -> Option<f32> {
    std::str::from_utf8(text)
        .ok()?
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
}

fn text(t: &[u8]) -> String {
    String::from_utf8_lossy(t).trim().to_owned()
}

fn open(a: &mut Amf, s: &mut State, t: Tag, e: &BytesStart<'_>) {
    let id = || attr(e, b"id").unwrap_or_default();
    match t {
        Tag::Amf => {
            a.unit = attr(e, b"unit");
            a.version = attr(e, b"version");
        }
        Tag::Object => {
            s.object = Some(Object {
                id: id(),
                ..Object::default()
            });
        }
        Tag::Vertex => {
            s.position = [0.0; 3];
            s.vertex_normal = None;
            s.vertex_color = None;
        }
        Tag::Normal => s.normal = [None; 3],
        Tag::Color => {
            s.channels = [None; 4];
            s.formula = false;
        }
        Tag::Volume => {
            s.volume = Some(Volume {
                material: attr(e, b"materialid"),
                ..Volume::default()
            });
        }
        Tag::Triangle => {
            s.corners = [u32::MAX; 3];
            s.tri_color = None;
            s.tri_map = None;
        }
        Tag::Texmap => {
            let mut tex = [u32::MAX; 4];
            for (k, name) in [&b"rtexid"[..], b"gtexid", b"btexid", b"atexid"]
                .into_iter()
                .enumerate()
            {
                if let Some(v) = attr(e, name) {
                    let n = s.texture_of.len();
                    tex[k] = *s.texture_of.entry(v.clone()).or_insert_with(|| {
                        a.texture_ids.push(v);
                        u32::try_from(n).unwrap_or(u32::MAX)
                    });
                }
            }
            s.map = Some(Texmap { tex, uv: [[0.0; 2]; 3] });
        }
        Tag::Material => {
            s.material = Some(Mat {
                id: id(),
                ..Mat::default()
            });
        }
        Tag::Composite => s.composite = attr(e, b"materialid"),
        Tag::Texture => {
            let size = |name: &[u8]| attr(e, name).and_then(|v| v.parse::<u32>().ok());
            s.texture = Some(Texture {
                id: id(),
                width: size(b"width").unwrap_or(0),
                height: size(b"height").unwrap_or(0),
                depth: size(b"depth").unwrap_or(1),
                tiled: attr(e, b"tiled").is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1"),
                pixels: Vec::new(),
            });
        }
        Tag::Constellation => {
            s.constellation = Some(Constellation {
                id: id(),
                ..Constellation::default()
            });
        }
        Tag::Instance => {
            s.instance = Some(Instance {
                target: attr(e, b"objectid").unwrap_or_default(),
                ..Instance::default()
            });
        }
        Tag::Metadata => s.meta = attr(e, b"type").unwrap_or_default(),
        Tag::Edge => a.curved += 1,
        _ => {}
    }
}

/// Keeps a metadata entry: a name as the name, the rest by type.
fn keep_meta(name: Option<&mut Option<String>>, map: &mut Map<String, Value>, key: String, value: String) {
    match name {
        Some(n) if key.eq_ignore_ascii_case("name") => *n = Some(value),
        _ => {
            map.insert(key, json!(value));
        }
    }
}

/// Grows a per-item array to `len`, marking the items added as having none.
fn pad<T: Copy>(v: &mut Vec<T>, len: usize, none: T) {
    if v.len() < len {
        v.resize(len, none);
    }
}

fn close(a: &mut Amf, s: &mut State, t: Tag, parent: Option<Tag>, limit: usize) -> Result<()> {
    match t {
        Tag::Leaf(leaf, k) => {
            let v = number(&s.text);
            match (leaf, parent) {
                (Leaf::Coord, Some(Tag::Coordinates)) => s.position[k] = v.unwrap_or(0.0),
                (Leaf::Normal, Some(Tag::Normal)) => s.normal[k] = v,
                (Leaf::Channel, Some(Tag::Color)) => {
                    s.formula |= v.is_none() && !s.text.trim_ascii().is_empty();
                    s.channels[k] = v;
                }
                (Leaf::Corner, Some(Tag::Triangle)) => {
                    s.corners[k] = std::str::from_utf8(&s.text)
                        .ok()
                        .and_then(|t| t.trim().parse().ok())
                        .unwrap_or(u32::MAX);
                }
                (Leaf::U, Some(Tag::Texmap)) => {
                    if let Some(m) = &mut s.map {
                        m.uv[k][0] = v.unwrap_or(0.0);
                    }
                }
                (Leaf::V, Some(Tag::Texmap)) => {
                    if let Some(m) = &mut s.map {
                        m.uv[k][1] = v.unwrap_or(0.0);
                    }
                }
                (Leaf::Delta, Some(Tag::Instance)) => {
                    if let Some(i) = &mut s.instance {
                        i.delta[k] = v.map_or(0.0, f64::from);
                    }
                }
                (Leaf::Turn, Some(Tag::Instance)) => {
                    if let Some(i) = &mut s.instance {
                        i.turn[k] = v.map_or(0.0, f64::from);
                    }
                }
                _ => {}
            }
        }
        Tag::Normal => {
            if let [Some(x), Some(y), Some(z)] = s.normal {
                s.vertex_normal = Some([x, y, z]);
            }
        }
        Tag::Color => {
            let c = match s.channels {
                [Some(r), Some(g), Some(b), alpha] => Some([r, g, b, alpha.unwrap_or(1.0)].map(|v| v.clamp(0.0, 1.0))),
                _ => None,
            };
            if s.formula {
                a.formulas += 1;
            }
            match parent {
                Some(Tag::Vertex) => s.vertex_color = c,
                Some(Tag::Triangle) => s.tri_color = c,
                Some(Tag::Volume) => {
                    if let Some(v) = &mut s.volume {
                        v.color = c;
                    }
                }
                Some(Tag::Object) => {
                    if let Some(o) = &mut s.object {
                        o.color = c;
                    }
                }
                Some(Tag::Material) => {
                    if let Some(m) = &mut s.material {
                        m.color = c;
                    }
                }
                _ => {}
            }
        }
        Tag::Vertex => {
            if let Some(o) = &mut s.object {
                a.vertices += 1;
                ensure!(
                    a.vertices <= limit.saturating_mul(3),
                    "too many vertices to convert (more than three for each of {limit} triangles)"
                );
                let i = o.verts.len();
                o.verts.push(s.position);
                if let Some(n) = s.vertex_normal {
                    pad(&mut o.normals, i, [NONE; 3]);
                    o.normals.push(n);
                }
                if let Some(c) = s.vertex_color {
                    pad(&mut o.colors, i, [NONE; 4]);
                    o.colors.push(c);
                }
            }
        }
        Tag::Texmap => {
            if parent == Some(Tag::Triangle) {
                s.tri_map = s.map.take();
            }
        }
        Tag::Triangle => {
            if let Some(v) = &mut s.volume {
                ensure!(a.triangles < limit, "too many triangles to convert (more than {limit})");
                a.triangles += 1;
                let k = v.tris.len();
                v.tris.push(s.corners);
                if let Some(c) = s.tri_color {
                    pad(&mut v.colors, k, [NONE; 4]);
                    v.colors.push(c);
                }
                if let Some(m) = s.tri_map.take() {
                    pad(&mut v.map_of, k, u32::MAX);
                    v.map_of.push(u32::try_from(v.maps.len()).unwrap_or(u32::MAX));
                    v.maps.push(m);
                }
            }
        }
        Tag::Volume => {
            if let (Some(v), Some(o)) = (s.volume.take(), &mut s.object) {
                o.volumes.push(v);
            }
        }
        Tag::Object => {
            if let Some(o) = s.object.take() {
                a.objects.push(o);
            }
        }
        Tag::Material => {
            if let Some(m) = s.material.take() {
                a.materials.push(m);
            }
        }
        Tag::Composite => {
            if let (Some(id), Some(m)) = (s.composite.take(), &mut s.material) {
                let share = number(&s.text);
                if share.is_none() {
                    a.formulas += 1;
                }
                m.composites.push((id, share));
            }
        }
        Tag::Texture => {
            if let Some(mut t) = s.texture.take() {
                if let Some(pixels) = base64(&s.text) {
                    t.pixels = pixels;
                    a.textures.push(t);
                } else {
                    a.bad_textures += 1;
                }
            }
        }
        Tag::Instance => {
            if let (Some(i), Some(c)) = (s.instance.take(), &mut s.constellation) {
                c.instances.push(i);
            }
        }
        Tag::Constellation => {
            if let Some(c) = s.constellation.take() {
                a.constellations.push(c);
            }
        }
        Tag::Metadata => {
            let key = std::mem::take(&mut s.meta);
            let value = text(&s.text);
            match parent {
                Some(Tag::Amf) => keep_meta(None, &mut a.metadata, key, value),
                Some(Tag::Object) => {
                    if let Some(o) = &mut s.object {
                        keep_meta(Some(&mut o.name), &mut o.metadata, key, value);
                    }
                }
                Some(Tag::Volume) => {
                    if let Some(v) = &mut s.volume {
                        keep_meta(Some(&mut v.name), &mut v.metadata, key, value);
                    }
                }
                Some(Tag::Material) => {
                    if let Some(m) = &mut s.material {
                        keep_meta(Some(&mut m.name), &mut m.metadata, key, value);
                    }
                }
                Some(Tag::Constellation) => {
                    if let Some(c) = &mut s.constellation {
                        keep_meta(Some(&mut c.name), &mut c.metadata, key, value);
                    }
                }
                Some(Tag::Instance) => {
                    if let Some(i) = &mut s.instance {
                        keep_meta(None, &mut i.metadata, key, value);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(())
}

/// Base64 text (whitespace and padding as they come) as bytes.
fn base64(text: &[u8]) -> Option<Vec<u8>> {
    use base64::Engine as _;
    use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
    let clean: Vec<u8> = text.iter().copied().filter(|c| !c.is_ascii_whitespace()).collect();
    let config = GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent);
    GeneralPurpose::new(&base64::alphabet::STANDARD, config)
        .decode(clean)
        .ok()
}

/// Moves the model's metadata of the kinds [`SourceInfo`] has from `metadata` into `info`: the standard's
/// `name`, `description`, `author` and `cad`, and the copyright, license and dates other writers add.
fn take_metadata(info: &mut SourceInfo, metadata: &mut Map<String, Value>) {
    let keys: Vec<String> = metadata.keys().cloned().collect();
    for key in keys {
        let slot = match key.to_ascii_lowercase().as_str() {
            "name" | "title" => &mut info.title,
            "description" => &mut info.description,
            "author" | "designer" => &mut info.author,
            "cad" | "application" => &mut info.tool,
            "copyright" => &mut info.copyright,
            "license" | "licenseterms" => &mut info.license,
            "creationdate" | "created" => &mut info.created,
            "modificationdate" | "modified" => &mut info.modified,
            _ => continue,
        };
        if slot.is_some() {
            continue;
        }
        if let Some(Value::String(v)) = metadata.remove(&key) {
            *slot = Some(v.trim().to_owned());
        }
    }
}

/// Meters per unit: the units the format names, and a few spelled other ways.
fn meters(unit: &str) -> Option<f64> {
    Some(match unit.to_ascii_lowercase().as_str() {
        "millimeter" | "millimeters" | "millimetre" | "mm" => 1e-3,
        "centimeter" | "centimeters" | "centimetre" | "cm" => 1e-2,
        "meter" | "meters" | "metre" | "m" => 1.0,
        "inch" | "inches" | "in" => 0.0254,
        "feet" | "foot" | "ft" => 0.3048,
        "micron" | "microns" | "micrometer" | "micrometre" | "um" => 1e-6,
        _ => return None,
    })
}

/// Linear RGBA of an sRGB color.
fn linear(c: [f32; 4]) -> [f32; 4] {
    [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]
}

fn some<const N: usize>(v: Option<&[f32; N]>) -> Option<[f32; N]> {
    v.filter(|v| !v[0].is_nan()).copied()
}

/// Each material's linear color: its own, else its composites' mixed by their proportions.
fn material_colors(materials: &[Mat], index: &HashMap<&str, usize>) -> Vec<Option<[f32; 4]>> {
    fn color(materials: &[Mat], index: &HashMap<&str, usize>, i: usize, depth: usize) -> Option<[f32; 4]> {
        let m = &materials[i];
        if let Some(c) = m.color {
            return Some(linear(c));
        }
        if depth >= 8 {
            return None;
        }
        let mut sum = [0.0f32; 4];
        let mut total = 0.0;
        for (id, share) in &m.composites {
            let (Some(share), Some(&j)) = (share, index.get(id.as_str())) else {
                continue;
            };
            if *share <= 0.0 {
                continue;
            }
            if let Some(c) = color(materials, index, j, depth + 1) {
                for k in 0..4 {
                    sum[k] += c[k] * share;
                }
                total += share;
            }
        }
        (total > 0.0).then(|| sum.map(|v| v / total))
    }
    (0..materials.len()).map(|i| color(materials, index, i, 0)).collect()
}

/// A vertex of a primitive being built: the primitive it's in, its index there and what else it carries.
#[derive(Clone, Copy)]
struct Slot {
    prim: u32,
    index: u32,
    key: u64,
}

/// The images made of textures, each once. They're made whether or not textures are kept: they're
/// small, held in the file, and which triangles have one decides the primitives and their texture
/// coordinates, which stay; the document drops them afterward.
struct Textures<'a> {
    amf: &'a Amf,
    by_id: HashMap<&'a str, usize>,
    made: HashMap<[u32; 4], Option<usize>>,
    images: Vec<Source>,
    /// Each image's wrap.
    wraps: Vec<[u32; 2]>,
    dropped: usize,
}

impl Textures<'_> {
    /// The image of a triangle's textures: one RGB(A) image of their gray channels.
    fn image(&mut self, tex: [u32; 4]) -> Option<usize> {
        if let Some(&i) = self.made.get(&tex) {
            return i;
        }
        let i = self.make(tex);
        if i.is_none() {
            self.dropped += 1;
        }
        self.made.insert(tex, i);
        i
    }

    fn make(&mut self, tex: [u32; 4]) -> Option<usize> {
        let channels: Vec<Option<&Texture>> = tex
            .iter()
            .map(|&k| {
                let id = self.amf.texture_ids.get(k as usize)?;
                self.by_id.get(id.as_str()).map(|&t| &self.amf.textures[t])
            })
            .collect();
        let first = channels.iter().flatten().next()?;
        let (w, h) = (first.width, first.height);
        let n = usize::try_from(u64::from(w) * u64::from(h)).ok()?;
        let alpha = channels[3].is_some();
        let stride = if alpha { 4 } else { 3 };
        if n == 0 || n.checked_mul(stride)? > MAX_TEXTURE {
            return None;
        }
        // Every channel the same size, with a layer's worth of pixels.
        for t in channels.iter().flatten() {
            if t.width != w || t.height != h || t.depth == 0 || t.pixels.len() < n {
                return None;
            }
        }
        let mut px = vec![255u8; n * stride];
        for (k, t) in channels.iter().enumerate().take(stride) {
            if let Some(t) = t {
                for (p, &v) in px.chunks_exact_mut(stride).zip(&t.pixels[..n]) {
                    p[k] = v;
                }
            }
        }
        let png = crate::png::encode(w, h, stride, &px)?;
        let tiled = channels.iter().flatten().any(|t| t.tiled);
        self.images.push(Source::Image(Image::from_png_or_jpeg(png)?));
        self.wraps.push(if tiled { [REPEAT; 2] } else { [CLAMP; 2] });
        Some(self.images.len() - 1)
    }
}

/// A material, made once for each look.
#[derive(PartialEq, Eq, Hash)]
struct Look {
    material: Option<usize>,
    color: [u32; 4],
    image: Option<usize>,
    blend: bool,
}

fn build(a: &Amf, zipped: bool) -> Result<Document> {
    let unit = a.unit.clone().unwrap_or_else(|| "millimeter".to_owned());
    let known = meters(&unit);
    let scale = known.unwrap_or(1e-3);
    let mat_index: HashMap<&str, usize> = a
        .materials
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id.as_str(), i))
        .collect();
    let mat_colors = material_colors(&a.materials, &mat_index);
    let mut tex = Textures {
        amf: a,
        by_id: a.textures.iter().enumerate().map(|(i, t)| (t.id.as_str(), i)).collect(),
        made: HashMap::new(),
        images: Vec::new(),
        wraps: Vec::new(),
        dropped: 0,
    };
    let mut b = Builder::default();
    let mut looks: HashMap<Look, usize> = HashMap::new();
    let mut materials: Vec<Full> = Vec::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut mesh_of: Vec<Option<usize>> = Vec::with_capacity(a.objects.len());
    // Each mesh's triangles, for counting them where they're placed.
    let mut mesh_triangles: Vec<usize> = Vec::new();
    let mut triangles = 0usize;
    let mut bad = 0usize;
    let mut prim_count = 0u32;
    let mut more: HashMap<(u32, u64), u32> = HashMap::new();
    for o in &a.objects {
        let mut slots = vec![
            Slot {
                prim: u32::MAX,
                index: 0,
                key: 0
            };
            o.verts.len()
        ];
        // Triangle colors, numbered for telling corners apart.
        let mut palette: HashMap<[u32; 4], u64> = HashMap::new();
        let mut primitives = Vec::new();
        let before = triangles;
        for vol in &o.volumes {
            let material = vol.material.as_deref().and_then(|id| mat_index.get(id)).copied();
            let base = vol
                .color
                .or(o.color)
                .map(linear)
                .or_else(|| material.and_then(|m| mat_colors[m]))
                .unwrap_or(GRAY);
            // Each triangle's image (when textured), and the triangles in order of it.
            let image_of: Vec<Option<usize>> = (0..vol.tris.len())
                .map(|t| {
                    let m = vol.map_of.get(t).filter(|&&m| m != u32::MAX)?;
                    tex.image(vol.maps.get(*m as usize)?.tex)
                })
                .collect();
            let mut order: Vec<u32> = (0..u32::try_from(vol.tris.len())?).collect();
            if image_of.iter().any(Option::is_some) {
                order.sort_by_key(|&t| image_of[t as usize].map_or(0, |i| i + 1));
            }
            let mut start = 0;
            while start < order.len() {
                let image = image_of[order[start] as usize];
                let end = order[start..]
                    .iter()
                    .position(|&t| image_of[t as usize] != image)
                    .map_or(order.len(), |n| start + n);
                let group = &order[start..end];
                start = end;
                let colored = image.is_none()
                    && group.iter().any(|&t| {
                        let t = t as usize;
                        some(vol.colors.get(t)).is_some()
                            || vol.tris[t].iter().any(|&v| some(o.colors.get(v as usize)).is_some())
                    });
                let pi = prim_count;
                prim_count = prim_count.wrapping_add(1);
                more.clear();
                let mut p = Prim::default();
                let mut normals = true;
                let mut blend = false;
                for &t in group {
                    let t = t as usize;
                    let corners = vol.tris[t];
                    if corners.iter().any(|&v| v as usize >= o.verts.len()) {
                        bad += 1;
                        continue;
                    }
                    let tri_color = some(vol.colors.get(t));
                    let map = image
                        .and_then(|_| vol.map_of.get(t))
                        .and_then(|&m| vol.maps.get(m as usize));
                    for (k, &v) in corners.iter().enumerate() {
                        let key = if let Some(m) = map {
                            u64::from(m.uv[k][0].to_bits()) | u64::from(m.uv[k][1].to_bits()) << 32
                        } else if let Some(c) = tri_color.filter(|_| colored) {
                            let n = palette.len() as u64 + 1;
                            *palette.entry(c.map(f32::to_bits)).or_insert(n)
                        } else {
                            0
                        };
                        let slot = slots[v as usize];
                        let found = if slot.prim == pi && slot.key == key {
                            Some(slot.index)
                        } else if slot.prim == pi {
                            more.get(&(v, key)).copied()
                        } else {
                            None
                        };
                        let index = if let Some(i) = found {
                            i
                        } else {
                            let index = u32::try_from(p.positions.len())?;
                            p.positions.push(o.verts[v as usize]);
                            match some(o.normals.get(v as usize)) {
                                Some(n) => p.normals.push(n),
                                None => normals = false,
                            }
                            if let Some(m) = map {
                                p.uvs.push(m.uv[k]);
                            }
                            if colored {
                                let c = tri_color
                                    .or_else(|| some(o.colors.get(v as usize)))
                                    .map_or(base, linear);
                                blend |= c[3] < 1.0;
                                p.colors.push(c);
                            }
                            if slot.prim == pi {
                                more.insert((v, key), index);
                            } else {
                                slots[v as usize] = Slot { prim: pi, index, key };
                            }
                            index
                        };
                        p.indices.push(index);
                    }
                    triangles += 1;
                }
                if p.indices.is_empty() {
                    continue;
                }
                if !normals {
                    p.normals = Vec::new();
                }
                let color = if image.is_some() || colored { [1.0; 4] } else { base };
                let look = Look {
                    material,
                    color: color.map(f32::to_bits),
                    image,
                    blend,
                };
                let next = materials.len();
                let mi = *looks.entry(look).or_insert(next);
                if mi == next {
                    materials.push(full(
                        a,
                        material,
                        color,
                        image,
                        blend,
                        &tex.wraps,
                        vol.color.or(o.color).is_some(),
                    ));
                }
                let extras = volume_extras(vol);
                for mut prim in b.primitives(p) {
                    prim["material"] = json!(mi);
                    if let Some(x) = &extras {
                        prim["extras"] = x.clone();
                    }
                    primitives.push(prim);
                }
            }
        }
        if primitives.is_empty() {
            mesh_of.push(None);
        } else {
            let mut mesh = json!({ "primitives": primitives });
            if let Some(n) = &o.name {
                mesh["name"] = json!(n);
            }
            meshes.push(mesh);
            mesh_triangles.push(triangles - before);
            mesh_of.push(Some(meshes.len() - 1));
        }
    }
    ensure!(triangles > 0, "this AMF has no triangles");

    let mut place = Place {
        a,
        mesh_of: &mesh_of,
        objects: a.objects.iter().enumerate().map(|(i, o)| (o.id.as_str(), i)).collect(),
        constellations: a
            .constellations
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.as_str(), i))
            .collect(),
        nodes: Vec::new(),
        path: Vec::new(),
        seen: HashSet::new(),
        dropped: 0,
    };
    let placed: HashSet<&str> = a
        .constellations
        .iter()
        .flat_map(|c| c.instances.iter().map(|i| i.target.as_str()))
        .collect();
    let mut roots = Vec::new();
    for (i, c) in a.constellations.iter().enumerate() {
        if !placed.contains(c.id.as_str()) {
            roots.extend(place.constellation(i, None));
        }
    }
    for (i, o) in a.objects.iter().enumerate() {
        if !placed.contains(o.id.as_str()) {
            roots.extend(place.object(i, None));
        }
    }
    // Constellations that only place each other: each loop is cut where it comes back around.
    for i in 0..a.constellations.len() {
        if !place.seen.contains(&i) {
            roots.extend(place.constellation(i, None));
        }
    }
    let Place { mut nodes, dropped, .. } = place;

    let mut info = SourceInfo {
        format: Some("AMF".into()),
        version: a.version.clone(),
        meters_per_unit: match known {
            Some(m) if a.unit.is_some() => Fact::Declared(m),
            _ => Fact::Assumed(scale),
        },
        unit: Some(unit),
        up_axis: Fact::Assumed(Axis::PosZ),
        ..SourceInfo::default()
    };
    info.converted.units = (scale - 1.0).abs() > f64::EPSILON;
    info.converted.up_axis = true;
    let mut metadata = a.metadata.clone();
    take_metadata(&mut info, &mut metadata);
    let mut source = json!({});
    if zipped {
        source["zipped"] = json!(true);
    }
    if !metadata.is_empty() {
        source["metadata"] = Value::Object(metadata);
    }
    if let Value::Object(m) = &mut source {
        info.write(m);
    }
    for (key, n) in [
        ("droppedCurvedEdges", a.curved),
        ("droppedFormulas", a.formulas),
        ("droppedTriangles", bad),
        ("droppedInstances", dropped),
        ("droppedTextures", tex.dropped + a.bad_textures),
    ] {
        if n > 0 {
            source[key] = json!(n);
        }
    }
    if a.truncated {
        source["truncated"] = json!(true);
    }
    let mut facts = Map::new();
    // The triangles of the scene: each mesh's, as many times as it's placed.
    let in_scene: usize = nodes
        .iter()
        .filter_map(|n| n["mesh"].as_u64())
        .filter_map(|m| mesh_triangles.get(usize::try_from(m).ok()?))
        .sum();
    facts.insert("triangles".into(), json!(in_scene));
    facts.insert("objects".into(), json!(a.objects.len()));

    // Z-up to Y-up (a quarter turn about X), and the unit to meters.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    nodes.push(
        json!({ "name": "RootNode", "children": roots, "rotation": [-h, 0.0, 0.0, h], "scale": [scale, scale, scale] }),
    );
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [nodes.len() - 1] }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    b.materials(&materials, tex.images, &mut json);
    Ok(b.finish(json, facts))
}

/// A look's glTF material.
#[allow(clippy::float_cmp, reason = "white is the exact value vertex colors set")]
fn full(
    a: &Amf,
    material: Option<usize>,
    color: [f32; 4],
    image: Option<usize>,
    blend: bool,
    wraps: &[[u32; 2]],
    own: bool,
) -> Full {
    let m = material.map(|i| &a.materials[i]);
    let name = match m {
        Some(m) => m.name.clone().unwrap_or_else(|| format!("material_{}", m.id)),
        None if image.is_some() => "texture".to_owned(),
        None if own || color == [1.0; 4] => "color".to_owned(),
        None => "default".to_owned(),
    };
    let mut full = Full::from(Material::plain(&name, color));
    if let Some(i) = image {
        full.base_color_tex = Some(Tex {
            wrap: wraps.get(i).copied().unwrap_or([REPEAT; 2]),
            ..Tex::new(i)
        });
    }
    if blend {
        full.alpha = Alpha::Blend;
    }
    if let Some(m) = m {
        let mut x = json!({ "amfId": m.id });
        if !m.metadata.is_empty() {
            x["metadata"] = Value::Object(m.metadata.clone());
        }
        if !m.composites.is_empty() {
            let list: Vec<Value> = m
                .composites
                .iter()
                .map(|(id, share)| json!({ "materialid": id, "proportion": share }))
                .collect();
            x["composites"] = json!(list);
        }
        full.extras = Some(x);
    }
    full
}

fn volume_extras(v: &Volume) -> Option<Value> {
    let mut x = Map::new();
    if let Some(n) = &v.name {
        x.insert("name".into(), json!(n));
    }
    if let Some(m) = &v.material {
        x.insert("materialid".into(), json!(m));
    }
    if !v.metadata.is_empty() {
        x.insert("metadata".into(), Value::Object(v.metadata.clone()));
    }
    (!x.is_empty()).then_some(Value::Object(x))
}

/// Places objects and constellations as nodes.
struct Place<'a> {
    a: &'a Amf,
    mesh_of: &'a [Option<usize>],
    objects: HashMap<&'a str, usize>,
    constellations: HashMap<&'a str, usize>,
    nodes: Vec<Value>,
    /// The constellations being placed, outermost first.
    path: Vec<usize>,
    /// The constellations placed at least once.
    seen: HashSet<usize>,
    dropped: usize,
}

impl Place<'_> {
    #[allow(clippy::float_cmp, reason = "only an exact identity is left out")]
    fn push(&mut self, mut node: Value, at: Option<&Instance>) -> usize {
        if let Some(i) = at {
            let [x, y, z] = i.turn.map(|d| (d.to_radians() / 2.0).sin_cos());
            // Turned about x, then y, then z: z * y * x.
            let qx = [x.0, 0.0, 0.0, x.1];
            let qy = [0.0, y.0, 0.0, y.1];
            let qz = [0.0, 0.0, z.0, z.1];
            let q = mul(mul(qz, qy), qx);
            if q != [0.0, 0.0, 0.0, 1.0] {
                node["rotation"] = json!(q);
            }
            if i.delta != [0.0; 3] {
                node["translation"] = json!(i.delta);
            }
            if !i.metadata.is_empty() {
                node["extras"]["instanceMetadata"] = Value::Object(i.metadata.clone());
            }
        }
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn object(&mut self, i: usize, at: Option<&Instance>) -> Option<usize> {
        if self.nodes.len() >= MAX_NODES {
            self.dropped += 1;
            return None;
        }
        let o = &self.a.objects[i];
        let mut node = json!({ "name": o.name.clone().unwrap_or_else(|| format!("object_{}", o.id)) });
        if let Some(m) = self.mesh_of[i] {
            node["mesh"] = json!(m);
        }
        node["extras"]["amfId"] = json!(o.id);
        if !o.metadata.is_empty() {
            node["extras"]["metadata"] = Value::Object(o.metadata.clone());
        }
        Some(self.push(node, at))
    }

    fn constellation(&mut self, i: usize, at: Option<&Instance>) -> Option<usize> {
        if self.nodes.len() >= MAX_NODES || self.path.contains(&i) || self.path.len() >= MAX_NESTING {
            self.dropped += 1;
            return None;
        }
        let a = self.a;
        let c = &a.constellations[i];
        self.seen.insert(i);
        self.path.push(i);
        let mut children = Vec::new();
        for inst in &c.instances {
            let child = if let Some(&o) = self.objects.get(inst.target.as_str()) {
                self.object(o, Some(inst))
            } else if let Some(&k) = self.constellations.get(inst.target.as_str()) {
                self.constellation(k, Some(inst))
            } else {
                self.dropped += 1;
                None
            };
            children.extend(child);
        }
        self.path.pop();
        let mut node = json!({ "name": c.name.clone().unwrap_or_else(|| format!("constellation_{}", c.id)) });
        if !children.is_empty() {
            node["children"] = json!(children);
        }
        node["extras"]["amfId"] = json!(c.id);
        if !c.metadata.is_empty() {
            node["extras"]["metadata"] = Value::Object(c.metadata.clone());
        }
        Some(self.push(node, at))
    }
}

/// The product of quaternions (x, y, z, w).
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;
    use crate::formats::Input;

    #[test]
    fn millimeters_are_assumed_when_unsaid_and_metadata_is_source_info() {
        let cube = |head: &str| {
            format!(
                "<amf{head}><metadata type=\"cad\">a tool</metadata><metadata type=\"tolerance\">0.1</metadata>                 <object id=\"1\"><mesh><vertices>{}</vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle>                 </volume></mesh></object></amf>",
                "<vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex>                 <vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex>                 <vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex>"
            )
        };
        let doc = load_str(&cube("")).unwrap();
        let info = doc.source_info();
        assert_eq!(info.meters_per_unit, Fact::Assumed(0.001));
        assert_eq!(info.unit.as_deref(), Some("millimeter"));
        assert_eq!(info.tool.as_deref(), Some("a tool"));
        assert_eq!(source(&doc)["metadata"], json!({ "tolerance": "0.1" }));
        assert!(doc.facts.get("unit").is_none());
        let info = load_str(&cube(" unit=\"meter\"")).unwrap().source_info();
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert!(!info.converted.units, "meters need no scaling");
        let info = load_str(&cube(" unit=\"furlong\"")).unwrap().source_info();
        assert_eq!(info.meters_per_unit, Fact::Assumed(0.001), "read as millimeters");
        assert_eq!(info.unit.as_deref(), Some("furlong"));
    }

    fn load_str(xml: &str) -> Result<Document> {
        load(Input::memory(xml.as_bytes(), "t.amf"))
    }

    fn flat(doc: &Document) -> crate::mesh::Mesh {
        doc.flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    const PYRAMID: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<amf unit="inch" version="1.1">
  <metadata type="name">Split Pyramid</metadata>
  <metadata type="author">A &amp; B</metadata>
  <object id="1">
    <metadata type="name">Pyramid</metadata>
    <mesh>
      <vertices>
        <vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates><normal><nx>0</nx><ny>0</ny><nz>-1</nz></normal></vertex>
        <vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates><color><r>1</r><g>0</g><b>0</b></color></vertex>
        <vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex>
        <vertex><coordinates><x>1</x><y>1</y><z>0</z></coordinates></vertex>
        <vertex><coordinates><x>0.5</x><y>0.5</y><z>1</z></coordinates></vertex>
        <edge><v1>0</v1><dx1>0</dx1><dy1>0</dy1><dz1>1</dz1><v2>1</v2><dx2>0</dx2><dy2>0</dy2><dz2>1</dz2></edge>
      </vertices>
      <volume materialid="2">
        <metadata type="name">Hard side</metadata>
        <triangle><v1>2</v1><v2>1</v2><v3>0</v3></triangle>
        <triangle><v1>0</v1><v2>1</v2><v3>4</v3></triangle>
      </volume>
      <volume materialid="3">
        <triangle><v1>2</v1><v2>3</v2><v3>1</v3></triangle>
        <triangle><color><r>0</r><g>0</g><b>1</b><a>0.5</a></color><v1>1</v1><v2>3</v2><v3>4</v3></triangle>
        <triangle><v1>4</v1><v2>3</v2><v3>9</v3></triangle>
      </volume>
      <volume materialid="4">
        <triangle><v1>4</v1><v2>2</v2><v3>3</v3></triangle>
      </volume>
    </mesh>
  </object>
  <material id="2"><metadata type="name">Hard</metadata><color><r>0.1</r><g>0.1</g><b>0.1</b></color></material>
  <material id="3"><color><r>0</r><g>0.9</g><b>0.9</b><a>0.5</a></color></material>
  <material id="5"><color><r>1</r><g>1</g><b>1</b></color></material>
  <material id="6"><color><r>0</r><g>0</g><b>0</b></color></material>
  <material id="4"><composite materialid="5">3</composite><composite materialid="6">1</composite><composite materialid="5">x*2</composite></material>
</amf>"#;

    fn source(doc: &Document) -> &Value {
        &doc.json["asset"]["extras"]["source"]
    }

    #[test]
    fn objects_volumes_colors_and_units() {
        let doc = load_str(PYRAMID).unwrap();
        let src = source(&doc);
        assert_eq!(src["unit"], "inch");
        assert_eq!(src["metersPerUnit"], 0.0254);
        let info = doc.source_info();
        assert_eq!(info.meters_per_unit, Fact::Declared(0.0254));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert!(info.converted.units && info.converted.up_axis && !info.converted.handedness);
        assert_eq!(info.output_meters_per_unit(), Fact::Declared(1.0));
        assert_eq!(info.author.as_deref(), Some("A & B"));
        assert_eq!(info.title.as_deref(), Some("Split Pyramid"));
        assert!(src.get("metadata").is_none(), "nothing else to keep");
        assert_eq!(src["droppedCurvedEdges"], 1);
        assert_eq!(src["droppedTriangles"], 1);
        assert_eq!(src["droppedFormulas"], 1);
        assert_eq!(doc.facts["triangles"], 5);
        let mesh = &doc.json["meshes"][0];
        assert_eq!(mesh["name"], "Pyramid");
        let prims = mesh["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 3);
        assert_eq!(prims[0]["extras"]["name"], "Hard side");
        let mats = doc.json["materials"].as_array().unwrap();
        let mat = |p: &Value| &mats[usize::try_from(p["material"].as_u64().unwrap()).unwrap()];
        // The first volume has a red vertex: its colors go on the vertices.
        assert!(prims[0]["attributes"].get("COLOR_0").is_some());
        assert_eq!(mat(&prims[0])["name"], "Hard");
        assert_eq!(mat(&prims[0])["extras"]["amfId"], "2");
        // The second: a translucent triangle among the material's color.
        assert!(prims[1]["attributes"].get("COLOR_0").is_some());
        assert_eq!(mat(&prims[1])["alphaMode"], "BLEND");
        // The third: three parts white to one part black, mixed in linear light.
        let c = mat(&prims[2])["pbrMetallicRoughness"]["baseColorFactor"][0]
            .as_f64()
            .unwrap();
        assert!((c - 0.75).abs() < 1e-6, "{c}");
        // Normals only where every vertex has one.
        assert!(prims.iter().all(|p| p["attributes"].get("NORMAL").is_none()));

        let m = flat(&doc);
        assert_eq!(m.triangles(), 5);
        // The apex, 1 inch up in Z, is 0.0254 m up in Y.
        let top = m
            .prims
            .iter()
            .flat_map(|p| &p.positions)
            .fold(f32::MIN, |a, p| a.max(p[1]));
        assert!((top - 0.0254).abs() < 1e-6, "{top}");
        // The translucent blue triangle keeps its color on its corners.
        let blue: usize = m
            .prims
            .iter()
            .map(|p| {
                p.colors
                    .iter()
                    .filter(|c| c[2] > 0.99 && c[0] < 0.01 && c[3] < 0.6)
                    .count()
            })
            .sum();
        assert_eq!(blue, 3);
    }

    const CONSTELLATIONS: &str = r#"<amf>
  <object id="1"><mesh><vertices>
    <vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex>
    <vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex>
    <vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex>
  </vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle></volume></mesh></object>
  <constellation id="2">
    <instance objectid="1"><deltax>10</deltax><deltay>0</deltay><deltaz>0</deltaz><rx>0</rx><ry>0</ry><rz>90</rz></instance>
    <instance objectid="1"><deltax>-10</deltax></instance>
  </constellation>
  <constellation id="3">
    <metadata type="name">Pair of pairs</metadata>
    <instance objectid="2"><deltaz>5</deltaz></instance>
    <instance objectid="2"><deltaz>-5</deltaz></instance>
    <instance objectid="7"/>
  </constellation>
  <constellation id="8"><instance objectid="9"/></constellation>
  <constellation id="9"><instance objectid="8"/></constellation>
</amf>"#;

    #[test]
    fn constellations_place_instances() {
        let doc = load_str(CONSTELLATIONS).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        // Four placed triangles; one mesh, shared.
        assert_eq!(doc.json["meshes"].as_array().unwrap().len(), 1);
        assert_eq!(nodes.iter().filter(|n| n.get("mesh").is_some()).count(), 4);
        assert!(nodes.iter().any(|n| n["name"] == "Pair of pairs"));
        // The missing object and the loop between 8 and 9 are dropped.
        assert_eq!(source(&doc)["droppedInstances"], 2);
        assert_eq!(doc.facts["triangles"], 4);
        let m = flat(&doc);
        assert_eq!(m.triangles(), 4);
        #[allow(clippy::cast_possible_truncation, reason = "small test values")]
        let mut xs: Vec<i32> = m
            .prims
            .iter()
            .flat_map(|p| &p.positions)
            .map(|p| (p[0] * 1000.0).round() as i32)
            .collect();
        xs.sort_unstable();
        xs.dedup();
        // In meters: the turned copy's corners at 10 and 9 mm, the other's at -10 and -9.
        assert_eq!(xs, [-10, -9, 9, 10]);
    }

    fn zipped(name: &str, body: &str) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        zip.start_file(name, opts).unwrap();
        zip.write_all(body.as_bytes()).unwrap();
        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn zipped_files_read_the_same() {
        let bytes = zipped("model.amf", PYRAMID);
        let doc = load(Input::memory(&bytes, "t.amf")).unwrap();
        assert_eq!(source(&doc)["zipped"], true);
        assert_eq!(doc.facts["triangles"], 5);
        let bytes = zipped("model.txt", "not xml at all");
        assert!(load(Input::memory(&bytes, "t.amf")).is_err());
    }

    #[test]
    fn textures_combine_their_channels() {
        use base64::Engine as _;
        let b64 = |px: &[u8]| base64::engine::general_purpose::STANDARD.encode(px);
        let xml = format!(
            r#"<amf>
  <object id="1"><mesh><vertices>
    <vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex>
    <vertex><coordinates><x>1</x><y>0</y><z>0</z></coordinates></vertex>
    <vertex><coordinates><x>0</x><y>1</y><z>0</z></coordinates></vertex>
  </vertices><volume>
    <triangle><v1>0</v1><v2>1</v2><v3>2</v3>
      <texmap rtexid="1" gtexid="2" btexid="2"><utex1>0</utex1><utex2>1</utex2><utex3>0</utex3><vtex1>0</vtex1><vtex2>0</vtex2><vtex3>1</vtex3></texmap>
    </triangle>
    <triangle><v1>2</v1><v2>1</v2><v3>0</v3></triangle>
  </volume></mesh></object>
  <texture id="1" width="2" height="2" depth="1" tiled="true" type="grayscale">{}</texture>
  <texture id="2" width="2" height="2" depth="1" type="grayscale">{}</texture>
</amf>"#,
            b64(&[255, 0, 0, 255]),
            b64(&[0, 0, 0, 0])
        );
        let doc = load_str(&xml).unwrap();
        assert_eq!(doc.json["images"].as_array().unwrap().len(), 1);
        let prims = doc.json["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(prims.len(), 2);
        let textured = prims
            .iter()
            .find(|p| p["attributes"].get("TEXCOORD_0").is_some())
            .unwrap();
        let mat = &doc.json["materials"][usize::try_from(textured["material"].as_u64().unwrap()).unwrap()];
        assert_eq!(mat["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0);
        assert_eq!(doc.json["samplers"][0]["wrapS"], REPEAT);
        let m = flat(&doc);
        let rgb = [255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0];
        assert_eq!(m.images[0].bytes, crate::png::encode(2, 2, 3, &rgb).unwrap());
        let uvs = &m.prims.iter().find(|p| !p.uvs.is_empty()).unwrap().uvs;
        assert!(uvs.contains(&[1.0, 0.0]) && uvs.contains(&[0.0, 1.0]));
    }

    #[test]
    fn limits_and_broken_files() {
        let input = Input {
            limit: 3,
            ..Input::memory(PYRAMID.as_bytes(), "t.amf")
        };
        assert!(load(input).is_err());
        assert!(load_str("<model/>").is_err());
        assert!(load_str("").is_err());
        assert!(load_str("PK\u{3}\u{4}junk").is_err());
        // Cut short, what's whole is kept.
        let cut = &PYRAMID[..PYRAMID.find("<volume materialid=\"3\"").unwrap()];
        let doc = load_str(cut).unwrap();
        assert_eq!(source(&doc)["truncated"], true);
        assert_eq!(doc.facts["triangles"], 2);
        // Cut inside a tag, likewise.
        let cut = &PYRAMID[..PYRAMID.find("<volume materialid=\"3\"").unwrap() + 9];
        let doc = load_str(cut).unwrap();
        assert_eq!(source(&doc)["truncated"], true);
        assert_eq!(doc.facts["triangles"], 2);
        // Every prefix: an error or a document, never a panic.
        for n in 0..CONSTELLATIONS.len() {
            if let Ok(doc) = load(Input::memory(&CONSTELLATIONS.as_bytes()[..n], "t.amf")) {
                doc.to_glb().unwrap();
            }
        }
        // Bytes changed at random, likewise.
        let mut rng = crate::test_rng::Pcg32::new(7);
        let zip = zipped("a.amf", PYRAMID);
        for base in [PYRAMID.as_bytes(), CONSTELLATIONS.as_bytes(), &zip] {
            for _ in 0..300 {
                let mut b = base.to_vec();
                for _ in 0..=rng.below(8) {
                    let at = rng.below(u32::try_from(b.len()).unwrap()) as usize;
                    b[at] = rng.next_u32().to_le_bytes()[0];
                }
                let cut = rng.below(u32::try_from(b.len()).unwrap()) as usize;
                for bytes in [&b[..], &b[..cut]] {
                    if let Ok(doc) = load(Input::memory(bytes, "t.amf")) {
                        doc.to_glb().unwrap();
                    }
                }
            }
        }
    }
}
