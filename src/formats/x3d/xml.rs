//! X3D's XML encoding: a node is an element, its fields attributes, and a child node goes in
//! the field its `containerField` names, or the one its kind goes in by default. A
//! `ProtoInstance` is a node of the kind its `name` gives, its `fieldValue`s its fields.

use std::collections::{HashMap, HashSet};

use quick_xml::events::{BytesStart, Event};

use super::{Graph, MAX_DEPTH, Proto, Val};
use crate::formats::stream::f64_from;
use crate::formats::xml::{attr, attrs, unescape_bytes};

/// What an open element is.
#[derive(Clone, Copy)]
enum Frame {
    Node(usize),
    /// `X3D`, `Scene` and `head`: their nodes are the scene's roots.
    Top,
    /// Not read, nor anything in it.
    Skip,
    /// `ProtoDeclare`, for the prototype it declares.
    Proto(usize),
    /// `ProtoInterface`, of a prototype.
    Interface(usize),
    /// A `field` of a prototype's interface: the prototype, and the field's place in it.
    Field(usize, usize),
    /// `ProtoBody`, of a prototype.
    Body(usize),
    /// `IS`, in a node.
    Is(usize),
    /// A `fieldValue` of a prototype instance: the node, and the field's place in it.
    FieldValue(usize, usize),
}

/// An XML file's scene graph, as much of it as reads.
pub(super) fn parse(d: &[u8]) -> Graph {
    let mut reader = quick_xml::Reader::from_reader(d);
    reader.config_mut().check_end_names = false;
    let mut p = Parser {
        g: Graph::default(),
        defs: HashMap::new(),
        saved: Vec::new(),
        strings: HashSet::new(),
        stack: Vec::new(),
    };
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let f = p.open(&e);
                p.stack.push(f);
            }
            Ok(Event::Empty(e)) => {
                let f = p.open(&e);
                p.close(f);
            }
            Ok(Event::End(_)) => {
                if let Some(f) = p.stack.pop() {
                    p.close(f);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            Ok(_) => {}
        }
    }
    p.g
}

struct Parser {
    g: Graph,
    defs: HashMap<Vec<u8>, usize>,
    /// The file's names while a prototype's own are in `defs`.
    saved: Vec<HashMap<Vec<u8>, usize>>,
    /// The prototype fields whose type is a string, by prototype and field.
    strings: HashSet<(String, String)>,
    stack: Vec<Frame>,
}

impl Parser {
    /// Ends element `f`.
    fn close(&mut self, f: Frame) {
        if let Frame::Proto(_) = f
            && let Some(defs) = self.saved.pop()
        {
            self.defs = defs;
        }
    }

    /// Reads element `e` into the graph: what it is, for the elements inside it.
    fn open(&mut self, e: &BytesStart<'_>) -> Frame {
        let parent = self.stack.last().copied();
        if matches!(parent, Some(Frame::Skip)) || self.stack.len() >= MAX_DEPTH {
            return Frame::Skip;
        }
        let local = e.local_name();
        let kind = local.as_ref();
        let text = |name: &[u8]| attr(e, name).map(|v| String::from_utf8_lossy(v.trim_ascii()).into_owned());
        match (parent, kind) {
            (Some(Frame::Proto(i)), b"ProtoInterface") => return Frame::Interface(i),
            (Some(Frame::Proto(i)), b"ProtoBody") => return Frame::Body(i),
            (Some(Frame::Interface(i)), b"field") => {
                let Some(name) = text(b"name") else {
                    return Frame::Skip;
                };
                if matches!(text(b"accessType").as_deref(), Some("inputOnly" | "outputOnly")) {
                    return Frame::Skip;
                }
                let proto = self.g.protos[i].name.clone();
                let typ = text(b"type").unwrap_or_default();
                let v = attr(e, b"value").map(|raw| match typ.as_str() {
                    "MFString" => Val::Strs(strings(raw.trim_ascii())),
                    "SFString" => Val::Strs(vec![String::from_utf8_lossy(&raw).into_owned()]),
                    _ => value(name.as_bytes(), &raw),
                });
                if matches!(typ.as_str(), "SFString" | "MFString") {
                    self.strings.insert((proto, name.clone()));
                }
                let fields = &mut self.g.protos[i].fields;
                fields.push((name, v));
                return Frame::Field(i, fields.len() - 1);
            }
            (Some(Frame::Node(n)), b"IS") => return Frame::Is(n),
            (Some(Frame::Is(n)), b"connect") => {
                if let (Some(field), Some(from)) = (text(b"nodeField"), text(b"protoField")) {
                    self.g.set(n, &field, Val::Is(from));
                }
                return Frame::Skip;
            }
            (Some(Frame::Node(n)), b"fieldValue") => {
                let Some(name) = text(b"name") else {
                    return Frame::Skip;
                };
                let stringy = self.strings.contains(&(self.g.nodes[n].kind.clone(), name.clone()));
                let v = match attr(e, b"value") {
                    Some(raw) if stringy => Val::Strs(strings(raw.trim_ascii())),
                    Some(raw) => value(name.as_bytes(), &raw),
                    None => Val::Nodes(Vec::new()),
                };
                self.g.set(n, &name, v);
                let slot = self.g.nodes[n].fields.iter().position(|(k, _)| *k == name).unwrap_or(0);
                return Frame::FieldValue(n, slot);
            }
            (Some(Frame::Proto(_) | Frame::Interface(_) | Frame::Is(_)), _) => return Frame::Skip,
            _ => {}
        }
        match kind {
            b"X3D" => {
                if let Some(v) = attr(e, b"version") {
                    self.g.version = String::from_utf8_lossy(&v).into_owned();
                }
                return Frame::Top;
            }
            b"Scene" | b"head" => return Frame::Top,
            b"unit" => {
                if attr(e, b"category").as_deref() == Some(b"length")
                    && let Some(f) = attr(e, b"conversionFactor").and_then(|f| f64_from(f.trim_ascii()))
                {
                    self.g.unit = Some(f);
                    self.g.unit_name = text(b"name");
                }
                return Frame::Skip;
            }
            b"meta" => {
                if matches!(parent, Some(Frame::Top))
                    && let (Some(name), Some(content)) = (text(b"name"), text(b"content"))
                {
                    self.g.meta.push((name, content));
                }
                return Frame::Skip;
            }
            b"component" | b"IS" | b"field" | b"fieldValue" | b"connect" => return Frame::Skip,
            b"ProtoDeclare" => {
                let name = text(b"name").unwrap_or_default();
                self.g.protos.push(Proto {
                    name,
                    ..Proto::default()
                });
                // Its names are its own, apart from the file's.
                self.saved.push(std::mem::take(&mut self.defs));
                return Frame::Proto(self.g.protos.len() - 1);
            }
            b"ExternProtoDeclare" => {
                if let Some(name) = text(b"name") {
                    let urls = attr(e, b"url").map(|u| strings(u.trim_ascii()));
                    self.g.externs.push((name, urls.unwrap_or_default()));
                }
                return Frame::Skip;
            }
            b"ROUTE" | b"Script" | b"IMPORT" | b"EXPORT" => {
                self.g.skip(&String::from_utf8_lossy(kind));
                return Frame::Skip;
            }
            _ => {}
        }
        let instance = kind == b"ProtoInstance";
        let kind = if instance {
            let Some(name) = text(b"name") else {
                self.g.skip("ProtoInstance");
                return Frame::Skip;
            };
            name
        } else {
            String::from_utf8_lossy(kind).into_owned()
        };
        let mut container: Option<String> = None;
        let mut def: Option<Vec<u8>> = None;
        let mut fields: Vec<(String, Val)> = Vec::new();
        for (key, raw) in attrs(e) {
            match key {
                b"DEF" => def = Some(unescape_bytes(&raw).into_owned()),
                b"USE" => {
                    // An instance: the node named, with nothing of its own.
                    let name = unescape_bytes(&raw);
                    if let Some(&id) = self.defs.get(name.trim_ascii()) {
                        self.place(parent, &kind, container.as_deref(), id, e);
                    }
                    return Frame::Skip;
                }
                b"containerField" => container = Some(String::from_utf8_lossy(&unescape_bytes(&raw)).into_owned()),
                b"class" | b"id" | b"style" | b"xmlns" => {}
                b"name" if instance => {}
                _ => fields.push((
                    String::from_utf8_lossy(key).into_owned(),
                    value(key, &unescape_bytes(&raw)),
                )),
            }
        }
        let name = def
            .as_ref()
            .map(|d| String::from_utf8_lossy(d.trim_ascii()).into_owned());
        let id = self.g.add(&kind, name);
        self.g.nodes[id].fields = fields;
        if let Some(d) = def {
            self.defs.insert(d.trim_ascii().to_vec(), id);
        }
        self.place(parent, &kind, container.as_deref(), id, e);
        Frame::Node(id)
    }

    /// Puts node `id` in its parent's field, or among the roots.
    fn place(&mut self, parent: Option<Frame>, kind: &str, container: Option<&str>, id: usize, e: &BytesStart<'_>) {
        // `containerField` may come after `USE`.
        let container = container
            .map(str::to_owned)
            .or_else(|| attr(e, b"containerField").map(|c| String::from_utf8_lossy(c.trim_ascii()).into_owned()));
        match parent {
            Some(Frame::Node(p)) => {
                let field = container.as_deref().unwrap_or_else(|| default_container(kind));
                self.g.push_child(p, field, id);
            }
            Some(Frame::Body(i)) => self.g.protos[i].body.push(id),
            Some(Frame::Field(i, k)) => match &mut self.g.protos[i].fields[k].1 {
                Some(Val::Nodes(list)) => list.push(id),
                v => *v = Some(Val::Nodes(vec![id])),
            },
            Some(Frame::FieldValue(n, slot)) => {
                if let Some((_, v)) = self.g.nodes[n].fields.get_mut(slot) {
                    match v {
                        Val::Nodes(list) => list.push(id),
                        v => *v = Val::Nodes(vec![id]),
                    }
                }
            }
            Some(Frame::Top) | None => self.g.roots.push(id),
            Some(Frame::Skip | Frame::Proto(_) | Frame::Interface(_) | Frame::Is(_)) => {}
        }
    }
}

/// The field a child node goes in when it doesn't say.
fn default_container(kind: &str) -> &'static str {
    match kind {
        "Appearance" => "appearance",
        "Material" | "PhysicalMaterial" | "UnlitMaterial" | "TwoSidedMaterial" => "material",
        "ImageTexture" | "PixelTexture" | "MovieTexture" | "MultiTexture" => "texture",
        "TextureTransform" | "MultiTextureTransform" => "textureTransform",
        "TextureProperties" => "textureProperties",
        "Coordinate" | "CoordinateDouble" => "coord",
        "Normal" => "normal",
        "Color" | "ColorRGBA" => "color",
        "TextureCoordinate" | "MultiTextureCoordinate" | "TextureCoordinateGenerator" => "texCoord",
        "Box"
        | "Cone"
        | "Cylinder"
        | "Sphere"
        | "IndexedFaceSet"
        | "IndexedTriangleSet"
        | "IndexedTriangleStripSet"
        | "IndexedTriangleFanSet"
        | "IndexedQuadSet"
        | "TriangleSet"
        | "TriangleStripSet"
        | "TriangleFanSet"
        | "QuadSet"
        | "ElevationGrid"
        | "Extrusion"
        | "Text"
        | "PointSet"
        | "IndexedLineSet"
        | "LineSet"
        | "Arc2D"
        | "ArcClose2D"
        | "Circle2D"
        | "Disk2D"
        | "Polyline2D"
        | "Polypoint2D"
        | "Rectangle2D"
        | "TriangleSet2D" => "geometry",
        k if k.starts_with("Metadata") => "metadata",
        _ => "children",
    }
}

/// Whether `b` separates the values in an attribute.
fn gap(b: u8) -> bool {
    b <= b' ' || b == b','
}

/// An attribute's value (its references decoded), its type told from how it's written: quoted strings for
/// `url` and any value starting with a quote, numbers, `true` and `false`, or else text.
fn value(key: &[u8], text: &[u8]) -> Val {
    let t = text.trim_ascii();
    if key == b"url" || t.first() == Some(&b'"') {
        return Val::Strs(strings(t));
    }
    if matches!(key, b"title" | b"description" | b"name" | b"value") {
        return Val::Strs(vec![String::from_utf8_lossy(t).into_owned()]);
    }
    let mut nums = Vec::new();
    let mut numeric = true;
    for w in t.split(|&b| gap(b)).filter(|w| !w.is_empty()) {
        let hex = || {
            let h = w.strip_prefix(b"0x").or_else(|| w.strip_prefix(b"0X"))?;
            u32::from_str_radix(std::str::from_utf8(h).ok()?, 16)
                .ok()
                .map(f64::from)
        };
        if let Some(v) = f64_from(w).or_else(hex) {
            nums.push(v);
        } else {
            numeric = false;
            break;
        }
    }
    if numeric {
        return Val::Nums(nums);
    }
    let bools: Option<Vec<bool>> = t
        .split(|&b| gap(b))
        .filter(|w| !w.is_empty())
        .map(|w| {
            if w.eq_ignore_ascii_case(b"true") {
                Some(true)
            } else if w.eq_ignore_ascii_case(b"false") {
                Some(false)
            } else {
                None
            }
        })
        .collect();
    match bools {
        Some(b) => Val::Bools(b),
        None => Val::Strs(vec![String::from_utf8_lossy(t).into_owned()]),
    }
}

/// An `MFString`'s strings: each in quotes, `\"` and `\\` escaped; text with no quotes is one
/// string.
fn strings(t: &[u8]) -> Vec<String> {
    if !t.contains(&b'"') {
        return if t.is_empty() {
            Vec::new()
        } else {
            vec![String::from_utf8_lossy(t).into_owned()]
        };
    }
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(open) = t[i..].iter().position(|&c| c == b'"') {
        let mut j = i + open + 1;
        let mut s = Vec::new();
        while j < t.len() && t[j] != b'"' {
            if t[j] == b'\\' && j + 1 < t.len() {
                j += 1;
            }
            s.push(t[j]);
            j += 1;
        }
        out.push(String::from_utf8_lossy(&s).into_owned());
        i = (j + 1).min(t.len());
    }
    out
}
