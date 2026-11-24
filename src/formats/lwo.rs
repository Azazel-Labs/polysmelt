//! LightWave objects (`.lwo`): LWOB and its layered form LWLO (LightWave 5), LWO2 (LightWave 6 to 2015) and LWO3
//! (2015 on). An object is IFF chunks: layers, each with its points, polygons and vertex maps, then the surfaces
//! (materials) and image clips they use.
//!
//! Each layer becomes a node at its pivot (relative to its parent layer's), its points relative to it; a hidden
//! layer says so in `extras`. Faces, and the cages of subdivision patches, are triangulated; two-point polygons are
//! lines and one-point polygons points. Curves, bones and metaballs have no glTF form and are counted.
//!
//! A surface is a material: its color times its diffuse level, its luminosity as emission, its transparency as
//! alpha (with its refractive index then), its glossiness as roughness (fully rough without specularity), and two
//! sides when it says so. Polygons are smoothed across shared points up to the surface's smoothing angle, within a
//! smoothing group, unless vertex normal maps give the normals; a surface without the angle is flat. A vertex color
//! map blends into the color by its intensity. The surface's top image layer on the color channel is the base
//! color texture, through a UV map (polygon-specific VMAD values over the VMAP's) or projected as LightWave
//! projects it (planar, cylindrical, spherical, cubic). An LWO3 surface made in the node editor takes its values
//! from the Principled BSDF or Standard node that feeds the surface, and its texture from the image node that feeds
//! that node's color. Morph maps, relative and absolute, are morph targets. What has no glTF slot (other texture
//! layers, weight and selection maps, parts, smoothing groups, every raw value) stays in `extras`.
//!
//! LightWave is Y up and in meters but left-handed: Z is mirrored, and polygons, clockwise seen from the front, are
//! turned around. Colors in the classic chunks are sRGB; node values, from LightWave's linear workflow, are linear.
//!
//! The source info has the format's meters, Y up and left-handed (declared, as the format fixes them; the handedness
//! converted), the description chunk (`DESC`) as its description and the comment chunk (`TEXT`) as its comments.

// LightWave is a name, not code.
#![allow(clippy::doc_markdown, reason = "LightWave is a name")]

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::crease;
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::Images;
use crate::material::{Alpha, CLAMP, Clearcoat, Factor, Full, MIRROR, REPEAT, Tex};
use crate::mesh::{FastMap, Prim, srgb_to_linear};
use crate::num::u32_of;
use crate::source::{Axis, Fact, Handedness, SourceInfo};

const FORM: [u8; 4] = *b"FORM";
/// The id of the chunk holding an LWO3 form's own data.
const BLANK: [u8; 4] = *b"    ";
/// LightWave's default surface color, 200 of 255.
const DEFAULT_COLOR: f32 = 200.0 / 255.0;
/// No tag, surface or group.
const NONE: u32 = u32::MAX;
/// Ear clipping can cost the cube of a polygon's corners; past this much of that work in one file, the rest of
/// the polygons are fanned.
const CLIP_BUDGET: u64 = 1 << 31;

/// The file's flavor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Lwob,
    Lwlo,
    Lwo2,
    Lwo3,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Lwob => "LWOB",
            Self::Lwlo => "LWLO",
            Self::Lwo2 => "LWO2",
            Self::Lwo3 => "LWO3",
        }
    }

    /// LightWave 5's, with 1-based surfaces in the polygons and fixed-point surface values.
    fn classic(self) -> bool {
        matches!(self, Self::Lwob | Self::Lwlo)
    }
}

/// A cursor over big-endian bytes: reading past the end gives `None`.
#[derive(Clone, Copy)]
struct Rd<'a> {
    d: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, i: 0 }
    }

    fn left(&self) -> usize {
        self.d.len().saturating_sub(self.i)
    }

    fn rest(&self) -> &'a [u8] {
        self.d.get(self.i..).unwrap_or(&[])
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.i..self.i.checked_add(n)?)?;
        self.i += n;
        Some(s)
    }

    fn u1(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u2(&mut self) -> Option<u16> {
        Some(u16::from_be_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u4(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }

    fn f4(&mut self) -> Option<f32> {
        self.u4().map(f32::from_bits)
    }

    fn f8(&mut self) -> Option<f64> {
        Some(f64::from_be_bytes(self.take(8)?.try_into().ok()?))
    }

    fn id(&mut self) -> Option<[u8; 4]> {
        self.take(4)?.try_into().ok()
    }

    fn vec(&mut self) -> Option<[f32; 3]> {
        Some([self.f4()?, self.f4()?, self.f4()?])
    }

    /// A variable-length index: two bytes, or four when the first is 0xFF.
    fn vx(&mut self) -> Option<u32> {
        if *self.d.get(self.i)? == 0xFF {
            Some(self.u4()? & 0x00FF_FFFF)
        } else {
            self.u2().map(u32::from)
        }
    }

    /// A string: up to a zero byte (or the end), padded to an even length. UTF-8 when it is, else Latin-1.
    fn s0(&mut self) -> Option<String> {
        let rest = self.d.get(self.i..).filter(|r| !r.is_empty())?;
        let n = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        let b = &rest[..n];
        let s = match std::str::from_utf8(b) {
            Ok(s) => s.to_owned(),
            Err(_) => b.iter().map(|&c| char::from(c)).collect(),
        };
        self.i = (self.i + ((n + 2) & !1)).min(self.d.len());
        Some(s)
    }
}

/// The chunks in `d` as (id, body): sizes take four bytes (`wide`) or two, and bodies are padded to an even
/// length. A chunk longer than what is left gets what is left.
fn chunks(d: &[u8], wide: bool) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut r = Rd::new(d);
    std::iter::from_fn(move || {
        let id = r.id()?;
        let n = if wide { r.u4()? as usize } else { usize::from(r.u2()?) };
        let body = r.take(n.min(r.left()))?;
        if n % 2 == 1 {
            r.i = (r.i + 1).min(r.d.len());
        }
        Some((id, body))
    })
}

/// A `FORM` chunk's type and contents.
fn form(body: &[u8]) -> Option<([u8; 4], &[u8])> {
    let t: [u8; 4] = body.get(..4)?.try_into().ok()?;
    Some((t, &body[4..]))
}

/// What kind of polygon.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Face,
    /// A subdivision patch's cage.
    Patch,
    /// Kept only so that later indices into the polygon list still line up.
    Dropped,
}

#[derive(Clone, Copy, Debug)]
struct Poly {
    start: u32,
    len: u16,
    shape: Shape,
    /// Its surface's tag.
    surf: u32,
    /// Its smoothing group's tag.
    group: u32,
}

/// A vertex map (VMAP) or a polygon-specific one (VMAD).
#[derive(Debug)]
struct VMap {
    kind: [u8; 4],
    /// The values per entry, or 0 when only the map's name and size are kept.
    dim: usize,
    name: String,
    per_poly: bool,
    /// Each entry's (point, polygon), indices into the layer.
    at: Vec<(u32, u32)>,
    values: Vec<f32>,
    entries: usize,
}

#[derive(Debug, Default)]
struct Layer {
    number: u16,
    hidden: bool,
    pivot: [f32; 3],
    name: Option<String>,
    parent: Option<u16>,
    points: Vec<[f32; 3]>,
    polys: Vec<Poly>,
    corners: Vec<u32>,
    maps: Vec<VMap>,
    /// Part tags used.
    parts: Vec<u32>,
    /// Where the latest point and polygon lists start, which later chunks count from.
    points_from: u32,
    polys_from: usize,
}

/// An image clip: a file, or another clip's.
#[derive(Debug, Clone)]
enum Clip {
    File(String),
    Ref(u32),
}

/// A texture layer.
#[derive(Debug, Clone)]
struct Block {
    ordinal: Vec<u8>,
    /// `IMAP`, `PROC`, `GRAD` or `SHDR`.
    kind: [u8; 4],
    chan: [u8; 4],
    enabled: bool,
    opacity: f32,
    blend: u16,
    negative: bool,
    /// 0 planar, 1 cylindrical, 2 spherical, 3 cubic, 4 front, 5 UV.
    proj: u16,
    /// 0 X, 1 Y, 2 Z.
    axis: u16,
    clip: Option<u32>,
    file: Option<String>,
    /// 0 reset, 1 repeat, 2 mirror, 3 edge, across and down.
    wrap: [u16; 2],
    /// How many times a cylindrical or spherical image goes around, and over.
    amount: [f32; 2],
    uv: Option<String>,
    center: [f32; 3],
    size: [f32; 3],
    /// Heading, pitch and bank, in radians.
    rotation: [f32; 3],
    world: bool,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            ordinal: Vec::new(),
            kind: *b"IMAP",
            chan: *b"COLR",
            enabled: true,
            opacity: 1.0,
            blend: 0,
            negative: false,
            proj: 0,
            axis: 2,
            clip: None,
            file: None,
            wrap: [1, 1],
            amount: [1.0, 1.0],
            uv: None,
            center: [0.0; 3],
            size: [1.0; 3],
            rotation: [0.0; 3],
            world: false,
        }
    }
}

/// An LWO3 node graph's shader: the node feeding the surface.
#[derive(Debug, Default)]
struct Nodes {
    server: String,
    name: String,
    values: Vec<(String, Vec<f64>)>,
    /// The image node feeding its color.
    image: Option<Block>,
    /// The image (or normal map) node feeding its normal.
    normal: Option<Block>,
    /// Every image node feeding it, as (input, image).
    inputs: Vec<(String, Block)>,
    /// Every node, as (server, name).
    all: Vec<(String, String)>,
}

impl Nodes {
    fn get(&self, name: &str) -> Option<&[f64]> {
        self.values.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }

    #[allow(clippy::cast_possible_truncation, reason = "material values are f32")]
    fn num(&self, name: &str) -> Option<f32> {
        self.get(name)?.first().map(|&v| v as f32).filter(|v| v.is_finite())
    }

    #[allow(clippy::cast_possible_truncation, reason = "material values are f32")]
    fn rgb(&self, name: &str) -> Option<[f32; 3]> {
        match self.get(name)? {
            &[r, g, b, ..] => Some([r as f32, g as f32, b as f32]).filter(|c| c.iter().all(|v| v.is_finite())),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct Surf {
    name: String,
    source: String,
    color: Option<[f32; 3]>,
    diff: Option<f32>,
    lumi: Option<f32>,
    spec: Option<f32>,
    refl: Option<f32>,
    tran: Option<f32>,
    trnl: Option<f32>,
    glos: Option<f32>,
    bump: Option<f32>,
    sides: Option<u16>,
    sman: Option<f32>,
    rind: Option<f32>,
    /// Intensity, map type and map name.
    vcol: Option<(f32, [u8; 4], String)>,
    blocks: Vec<Block>,
    nodes: Option<Nodes>,
    /// LWOB flags.
    flags: Option<u16>,
}

#[derive(Debug)]
struct Model {
    kind: Kind,
    tags: Vec<String>,
    layers: Vec<Layer>,
    surfs: Vec<Surf>,
    clips: HashMap<u32, Clip>,
    description: Option<String>,
    comment: Option<String>,
    curves: usize,
    bones: usize,
    metaballs: usize,
    other_polys: usize,
    bad_polys: usize,
    patches: usize,
    envelopes: usize,
    truncated: bool,
}

impl Model {
    fn layer(&mut self) -> &mut Layer {
        if self.layers.is_empty() {
            self.layers.push(Layer::default());
        }
        let n = self.layers.len() - 1;
        &mut self.layers[n]
    }
}

fn parse(data: &[u8]) -> Result<Model> {
    let mut r = Rd::new(data);
    ensure!(r.id() == Some(FORM), UnknownFormat, "not a LightWave object");
    let size = r.u4().unwrap_or(0) as usize;
    let kind = match &r.id().unwrap_or_default() {
        b"LWOB" => Kind::Lwob,
        b"LWLO" => Kind::Lwlo,
        b"LWO2" => Kind::Lwo2,
        b"LWO3" => Kind::Lwo3,
        _ => bail!(UnknownFormat, "not a LightWave object"),
    };
    let body = r.rest();
    let want = size.saturating_sub(4);
    let mut m = Model {
        kind,
        tags: Vec::new(),
        layers: Vec::new(),
        surfs: Vec::new(),
        clips: HashMap::new(),
        description: None,
        comment: None,
        curves: 0,
        bones: 0,
        metaballs: 0,
        other_polys: 0,
        bad_polys: 0,
        patches: 0,
        envelopes: 0,
        truncated: want > body.len(),
    };
    let body = &body[..want.min(body.len())];
    for (id, b) in chunks(body, true) {
        match &id {
            b"FORM" => {
                let Some((t, inner)) = form(b) else { continue };
                match &t {
                    b"SURF" => m.surfs.push(surf_form(inner)),
                    b"CLIP" => clip_form(inner, &mut m.clips),
                    b"ENVL" => m.envelopes += 1,
                    _ => {}
                }
            }
            b"LAYR" => {
                let mut r = Rd::new(b);
                let number = r.u2().unwrap_or(0);
                let flags = r.u2().unwrap_or(0);
                let pivot = if kind == Kind::Lwlo { None } else { r.vec() };
                let name = r.s0().filter(|s| !s.is_empty());
                let parent = r.u2().filter(|&p| p != 0xFFFF);
                m.layers.push(Layer {
                    number,
                    hidden: flags & 1 != 0,
                    pivot: pivot.filter(|p| p.iter().all(|c| c.is_finite())).unwrap_or_default(),
                    name,
                    parent,
                    ..Layer::default()
                });
            }
            b"PNTS" => {
                let l = m.layer();
                l.points_from = u32::try_from(l.points.len()).unwrap_or(u32::MAX);
                let more = b.len() / 12;
                // At most 2^24 points can be named by a polygon, the most an index holds.
                let room = (1usize << 24).saturating_sub(l.points.len());
                l.points.reserve(more.min(room));
                for c in b.chunks_exact(12).take(room) {
                    let f = |i: usize| f32::from_be_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
                    l.points.push([f(0), f(4), f(8)]);
                }
            }
            b"POLS" if kind.classic() => polys_classic(&mut m, b),
            b"POLS" => polys(&mut m, b),
            b"PTAG" => {
                let mut r = Rd::new(b);
                let t = r.id().unwrap_or_default();
                let l = m.layer();
                while r.left() > 0 {
                    let (Some(p), Some(tag)) = (r.vx(), r.u2()) else { break };
                    let Some(poly) = l.polys.get_mut(l.polys_from + p as usize) else {
                        continue;
                    };
                    let tag = u32::from(tag);
                    match &t {
                        b"SURF" => poly.surf = tag,
                        b"SMGP" => poly.group = tag,
                        b"PART" if !l.parts.contains(&tag) && l.parts.len() < 4096 => l.parts.push(tag),
                        _ => {}
                    }
                }
            }
            b"VMAP" | b"VMAD" => vmap(m.layer(), b, &id == b"VMAD"),
            b"TAGS" | b"SRFS" => {
                let mut r = Rd::new(b);
                while let Some(s) = r.s0() {
                    m.tags.push(s);
                }
            }
            b"SURF" if kind.classic() => m.surfs.push(surf_classic(b)),
            b"SURF" => {
                let mut r = Rd::new(b);
                let name = r.s0().unwrap_or_default();
                let source = r.s0().unwrap_or_default();
                let mut s = Surf {
                    name,
                    source,
                    ..Surf::default()
                };
                for (id, c) in chunks(r.rest(), false) {
                    s.read(id, c, false);
                }
                m.surfs.push(s);
            }
            b"CLIP" => {
                let mut r = Rd::new(b);
                let Some(index) = r.u4() else { continue };
                for (id, c) in chunks(r.rest(), false) {
                    if let Some(clip) = clip_entry(id, c) {
                        m.clips.insert(index, clip);
                    }
                }
            }
            b"ENVL" => m.envelopes += 1,
            b"DESC" => m.description = Rd::new(b).s0(),
            b"TEXT" => m.comment = Rd::new(b).s0(),
            _ => {}
        }
    }
    Ok(m)
}

/// An LWO2 or LWO3 polygon list.
fn polys(m: &mut Model, b: &[u8]) {
    let mut r = Rd::new(b);
    let t = r.id().unwrap_or_default();
    let shape = match &t {
        b"FACE" => Shape::Face,
        b"PTCH" | b"SUBD" => Shape::Patch,
        _ => Shape::Dropped,
    };
    let (mut kept, mut bad) = (0usize, 0usize);
    let l = m.layer();
    l.polys_from = l.polys.len();
    let base = l.points_from as usize;
    while r.left() > 0 {
        let Some(n) = r.u2() else { break };
        let n = n & 0x03FF;
        let start = l.corners.len();
        let mut ok = shape != Shape::Dropped;
        for _ in 0..n {
            let Some(v) = r.vx() else {
                ok = false;
                break;
            };
            let v = base + v as usize;
            ok &= v < l.points.len();
            if ok {
                l.corners.push(u32_of(v));
            }
        }
        if !ok {
            l.corners.truncate(start);
            if shape != Shape::Dropped {
                bad += 1;
            }
        }
        kept += usize::from(ok);
        l.polys.push(Poly {
            start: u32_of(start),
            len: if ok { n } else { 0 },
            shape: if ok { shape } else { Shape::Dropped },
            surf: NONE,
            group: NONE,
        });
    }
    let all = l.polys.len() - l.polys_from;
    m.bad_polys += bad;
    match &t {
        b"FACE" => {}
        b"PTCH" | b"SUBD" => m.patches += kept,
        b"CURV" => m.curves += all,
        b"BONE" => m.bones += all,
        b"MBAL" => m.metaballs += all,
        _ => m.other_polys += all,
    }
}

/// A LightWave 5 polygon list: each polygon ends with its 1-based surface, negative when detail polygons (which are
/// read as polygons in their own right) follow.
fn polys_classic(m: &mut Model, b: &[u8]) {
    let mut r = Rd::new(b);
    let mut bad = 0usize;
    let l = m.layer();
    l.polys_from = l.polys.len();
    let base = l.points_from as usize;
    while r.left() > 0 {
        let Some(n) = r.u2() else { break };
        let start = l.corners.len();
        let mut ok = true;
        for _ in 0..n {
            let Some(v) = r.u2() else {
                ok = false;
                break;
            };
            let v = base + usize::from(v);
            ok &= v < l.points.len();
            if ok {
                l.corners.push(u32_of(v));
            }
        }
        let Some(surf) = r.u2() else { break };
        #[allow(clippy::cast_possible_wrap, reason = "the surface is a signed 16-bit number")]
        let mut surf = surf as i16;
        if surf < 0 {
            surf = surf.saturating_neg();
            // The count of detail polygons, which are read as they come.
            r.u2();
        }
        if !ok || n > 1023 {
            l.corners.truncate(start);
            bad += 1;
        }
        let ok = ok && n <= 1023;
        l.polys.push(Poly {
            start: u32_of(start),
            len: if ok { n } else { 0 },
            shape: if ok { Shape::Face } else { Shape::Dropped },
            surf: u32::try_from(surf - 1).unwrap_or(NONE),
            group: NONE,
        });
    }
    m.bad_polys += bad;
}

/// The vertex map types whose values are read, and how many each entry has.
fn map_dim(kind: [u8; 4]) -> Option<usize> {
    match &kind {
        b"TXUV" => Some(2),
        b"RGB " | b"NORM" | b"MORF" | b"SPOT" => Some(3),
        b"RGBA" => Some(4),
        _ => None,
    }
}

fn vmap(l: &mut Layer, b: &[u8], per_poly: bool) {
    let mut r = Rd::new(b);
    let (Some(kind), Some(dim)) = (r.id(), r.u2()) else {
        return;
    };
    let dim = usize::from(dim);
    let name = r.s0().unwrap_or_default();
    let keep = map_dim(kind) == Some(dim);
    let mut map = VMap {
        kind,
        dim: if keep { dim } else { 0 },
        name,
        per_poly,
        at: Vec::new(),
        values: Vec::new(),
        entries: 0,
    };
    let base = l.points_from as usize;
    while r.left() > 0 {
        let Some(p) = r.vx() else { break };
        let poly = if per_poly {
            let Some(f) = r.vx() else { break };
            l.polys_from + f as usize
        } else {
            0
        };
        let Some(v) = r.take(dim * 4) else { break };
        map.entries += 1;
        let p = base + p as usize;
        if keep && p < l.points.len() && poly <= l.polys.len().max(1) {
            map.at.push((u32_of(p), u32::try_from(poly).unwrap_or(NONE)));
            map.values
                .extend(v.chunks_exact(4).map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]])));
        }
    }
    l.maps.push(map);
}

/// One subchunk of an LWO2 image clip.
fn clip_entry(id: [u8; 4], c: &[u8]) -> Option<Clip> {
    let mut r = Rd::new(c);
    match &id {
        b"STIL" => r.s0().filter(|s| !s.is_empty()).map(Clip::File),
        b"ISEQ" => {
            let digits = usize::from(r.u1()?.min(10));
            let _flags = r.u1()?;
            let _offset = r.u2()?;
            let _reserved = r.u2()?;
            #[allow(clippy::cast_possible_wrap, reason = "the first frame is a signed 16-bit number")]
            let start = r.u2()? as i16;
            let _end = r.u2()?;
            let prefix = r.s0().unwrap_or_default();
            let suffix = r.s0().unwrap_or_default();
            Some(Clip::File(format!("{prefix}{start:0digits$}{suffix}")))
        }
        b"XREF" => r.u4().map(Clip::Ref),
        _ => None,
    }
}

/// An LWO3 clip form: its index in the blank chunk, then a still image form whose blank chunk names the file.
fn clip_form(d: &[u8], clips: &mut HashMap<u32, Clip>) {
    let mut index = None;
    let mut clip = None;
    for (id, c) in chunks(d, true) {
        match &id {
            b"    " => index = Rd::new(c).u4(),
            b"FORM" => {
                if let Some((t, inner)) = form(c)
                    && &t == b"STIL"
                {
                    clip = still(inner).map(Clip::File);
                }
            }
            b"XREF" => clip = Rd::new(c).u4().map(Clip::Ref),
            _ => {}
        }
    }
    if let (Some(i), Some(c)) = (index, clip) {
        clips.insert(i, c);
    }
}

/// The file an LWO3 still image form names.
fn still(d: &[u8]) -> Option<String> {
    chunks(d, true)
        .find(|(id, _)| *id == BLANK)
        .and_then(|(_, c)| Rd::new(c).s0())
        .filter(|s| !s.is_empty())
}

/// An LWO3 surface form: its name and source in the blank chunk, then the classic subchunks (with four-byte
/// sizes) and the node graph.
fn surf_form(d: &[u8]) -> Surf {
    let mut s = Surf::default();
    for (id, c) in chunks(d, true) {
        match &id {
            b"    " => {
                let mut r = Rd::new(c);
                s.name = r.s0().unwrap_or_default();
                s.source = r.s0().unwrap_or_default();
            }
            b"FORM" => {
                let Some((t, inner)) = form(c) else { continue };
                match &t {
                    b"NODS" => s.nodes = Some(nodes(inner)),
                    b"BLOK" => s.blocks.push(block(inner, true)),
                    _ => {}
                }
            }
            _ => s.read(id, c, true),
        }
    }
    s
}

impl Surf {
    /// One LWO2 (or classic LWO3) surface subchunk.
    fn read(&mut self, id: [u8; 4], c: &[u8], wide: bool) {
        let mut r = Rd::new(c);
        let f = r.f4().filter(|v| v.is_finite());
        match &id {
            b"COLR" => self.color = Rd::new(c).vec().filter(|v| v.iter().all(|c| c.is_finite())),
            b"DIFF" => self.diff = f,
            b"LUMI" => self.lumi = f,
            b"SPEC" => self.spec = f,
            b"REFL" => self.refl = f,
            b"TRAN" => self.tran = f,
            b"TRNL" => self.trnl = f,
            b"GLOS" => self.glos = f,
            b"BUMP" => self.bump = f,
            b"SMAN" => self.sman = f,
            b"RIND" => self.rind = f,
            b"SIDE" => self.sides = Rd::new(c).u2(),
            b"VCOL" => {
                let (Some(i), Some(_env)) = (f, r.vx()) else { return };
                let (Some(t), Some(name)) = (r.id(), r.s0()) else {
                    return;
                };
                self.vcol = Some((i, t, name));
            }
            b"BLOK" => self.blocks.push(block(c, wide)),
            _ => {}
        }
    }
}

/// A texture layer: its header (the layer's type, with its ordinal, channel, opacity and whether it's on), then
/// the mapping.
fn block(d: &[u8], wide: bool) -> Block {
    let mut b = Block::default();
    let mut first = true;
    for (id, c) in chunks(d, wide) {
        // In LWO3 a nested chunk can be a form of its own.
        let (id, c, is_form) = match form(c).filter(|_| id == FORM) {
            Some((t, inner)) => (t, inner, true),
            None => (id, c, false),
        };
        if first {
            first = false;
            b.kind = id;
            let mut r = Rd::new(c);
            if !is_form {
                b.ordinal = r.s0().unwrap_or_default().into_bytes();
            }
            for (h, x) in chunks(r.rest(), wide) {
                let mut r = Rd::new(x);
                match &h {
                    b"    " => b.ordinal = r.s0().unwrap_or_default().into_bytes(),
                    b"CHAN" => b.chan = r.id().unwrap_or(b.chan),
                    b"ENAB" => b.enabled = r.u2() != Some(0),
                    b"OPAC" => {
                        b.blend = r.u2().unwrap_or(0);
                        b.opacity = r.f4().filter(|v| v.is_finite()).unwrap_or(1.0);
                    }
                    b"NEGA" => b.negative = r.u2().is_some_and(|v| v != 0),
                    _ => {}
                }
            }
            continue;
        }
        let mut r = Rd::new(c);
        match &id {
            b"TMAP" => {
                for (t, x) in chunks(c, wide) {
                    let mut r = Rd::new(x);
                    let v = r.vec().filter(|v| v.iter().all(|c| c.is_finite()));
                    match &t {
                        b"CNTR" => b.center = v.unwrap_or(b.center),
                        b"SIZE" => b.size = v.unwrap_or(b.size),
                        b"ROTA" => b.rotation = v.unwrap_or(b.rotation),
                        b"CSYS" => b.world = Rd::new(x).u2().is_some_and(|v| v != 0),
                        _ => {}
                    }
                }
            }
            b"PROJ" => b.proj = r.u2().unwrap_or(b.proj),
            b"AXIS" => b.axis = r.u2().unwrap_or(b.axis),
            b"IMAG" => b.clip = r.vx(),
            b"WRAP" => b.wrap = [r.u2().unwrap_or(1), r.u2().unwrap_or(1)],
            b"WRPW" => b.amount[0] = r.f4().filter(|v| v.is_finite()).unwrap_or(1.0),
            b"WRPH" => b.amount[1] = r.f4().filter(|v| v.is_finite()).unwrap_or(1.0),
            b"VMAP" => b.uv = r.s0().filter(|s| !s.is_empty()),
            _ => {}
        }
    }
    b
}

/// A LightWave 5 surface: fixed-point levels (256 is all), which their float forms replace, and textures named by
/// file, each followed by the chunks that set it up.
fn surf_classic(d: &[u8]) -> Surf {
    let mut r = Rd::new(d);
    let mut s = Surf {
        name: r.s0().unwrap_or_default(),
        ..Surf::default()
    };
    let mut floats = [false; 5];
    let mut tex: Option<Block> = None;
    for (id, c) in chunks(r.rest(), false) {
        let mut r = Rd::new(c);
        let fixed = Rd::new(c).u2().map(|v| f32::from(v) / 256.0);
        let float = Rd::new(c).f4().filter(|v| v.is_finite());
        let level = |id: &[u8; 4]| match id {
            b"LUMI" | b"VLUM" => Some(0),
            b"DIFF" | b"VDIF" => Some(1),
            b"SPEC" | b"VSPC" => Some(2),
            b"REFL" | b"VRFL" => Some(3),
            b"TRAN" | b"VTRN" => Some(4),
            _ => None,
        };
        match &id {
            b"COLR" => {
                if let Some(rgb) = r.take(3) {
                    s.color = Some([rgb[0], rgb[1], rgb[2]].map(|v| f32::from(v) / 255.0));
                }
            }
            b"FLAG" => s.flags = r.u2(),
            b"LUMI" | b"DIFF" | b"SPEC" | b"REFL" | b"TRAN" => {
                let i = level(&id).unwrap_or(0);
                if !floats[i] && fixed.is_some() {
                    *slot(&mut s, i) = fixed;
                }
            }
            b"VLUM" | b"VDIF" | b"VSPC" | b"VRFL" | b"VTRN" => {
                let i = level(&id).unwrap_or(0);
                if float.is_some() {
                    floats[i] = true;
                    *slot(&mut s, i) = float;
                }
            }
            b"GLOS" => {
                // A Phong exponent: 16, 64, 256 or 1024 for low to very high, glossiness 0.2 to 0.8.
                let n = r.u2().unwrap_or(0);
                if n > 0 {
                    s.glos = Some(((f32::from(n).log2() - 2.0) / 10.0).clamp(0.0, 1.0));
                }
            }
            b"SMAN" => s.sman = float,
            b"RIND" => s.rind = float,
            b"CTEX" | b"DTEX" | b"STEX" | b"RTEX" | b"TTEX" | b"LTEX" | b"BTEX" => {
                if let Some(t) = tex.take() {
                    s.blocks.push(t);
                }
                let what = r.s0().unwrap_or_default();
                let chan = match id[0] {
                    b'C' => *b"COLR",
                    b'D' => *b"DIFF",
                    b'S' => *b"SPEC",
                    b'R' => *b"REFL",
                    b'T' => *b"TRAN",
                    b'L' => *b"LUMI",
                    _ => *b"BUMP",
                };
                let proj = ["Planar", "Cylindrical", "Spherical", "Cubic", "Front Projection"]
                    .iter()
                    .position(|p| what.starts_with(p));
                tex = Some(Block {
                    ordinal: format!("{:04}", s.blocks.len()).into_bytes(),
                    kind: if what.ends_with("Image Map") {
                        *b"IMAP"
                    } else {
                        *b"PROC"
                    },
                    chan,
                    proj: proj.and_then(|p| u16::try_from(p).ok()).unwrap_or(0),
                    uv: Some(what),
                    ..Block::default()
                });
            }
            _ => {
                let Some(t) = tex.as_mut() else { continue };
                match &id {
                    b"TIMG" => t.file = r.s0().filter(|f| !f.is_empty() && f != "(none)"),
                    b"TFLG" => {
                        let f = r.u2().unwrap_or(0);
                        t.axis = if f & 1 != 0 {
                            0
                        } else if f & 2 != 0 {
                            1
                        } else {
                            2
                        };
                        t.world = f & 8 != 0;
                        t.negative = f & 16 != 0;
                    }
                    b"TSIZ" => t.size = r.vec().filter(|v| v.iter().all(|c| c.is_finite())).unwrap_or(t.size),
                    b"TCTR" => t.center = r.vec().filter(|v| v.iter().all(|c| c.is_finite())).unwrap_or(t.center),
                    // Black, clamp, repeat, mirror: as LWO2's reset, edge, repeat, mirror.
                    b"TWRP" => {
                        let w = |v: Option<u16>| match v {
                            Some(0) => 0,
                            Some(1) => 3,
                            Some(3) => 2,
                            _ => 1,
                        };
                        t.wrap = [w(r.u2()), w(r.u2())];
                    }
                    b"TFP0" => t.amount[0] = float.unwrap_or(1.0),
                    b"TFP1" => t.amount[1] = float.unwrap_or(1.0),
                    b"TOPC" => t.opacity = float.unwrap_or(1.0),
                    _ => {}
                }
            }
        }
    }
    if let Some(t) = tex {
        s.blocks.push(t);
    }
    // A classic texture's type name isn't a UV map.
    for b in &mut s.blocks {
        b.uv = None;
    }
    if let Some(f) = s.flags {
        if f & 256 != 0 {
            s.sides = Some(3);
        }
        // Smoothing is a flag, with its angle (89.5 degrees unless given) beside it.
        s.sman = if f & 4 != 0 {
            Some(s.sman.unwrap_or(89.5f32.to_radians()))
        } else {
            Some(0.0)
        };
    }
    s
}

/// The level a LightWave 5 surface chunk sets: luminosity, diffuse, specular, reflection or transparency.
fn slot(s: &mut Surf, i: usize) -> &mut Option<f32> {
    match i {
        0 => &mut s.lumi,
        1 => &mut s.diff,
        2 => &mut s.spec,
        3 => &mut s.refl,
        _ => &mut s.tran,
    }
}

/// An LWO3 node graph: the nodes (their servers and values), and the connections between them.
fn nodes(d: &[u8]) -> Nodes {
    #[derive(Default)]
    struct Node {
        server: String,
        name: String,
        values: Vec<(String, Vec<f64>)>,
        image: Option<Block>,
    }
    let mut list: Vec<Node> = Vec::new();
    // (to node, to input, from node).
    let mut links: Vec<[String; 3]> = Vec::new();
    for (id, c) in chunks(d, true) {
        let Some((t, inner)) = form(c).filter(|_| id == FORM) else {
            continue;
        };
        match &t {
            b"NNDS" => {
                let mut server = String::new();
                for (id, c) in chunks(inner, true) {
                    match &id {
                        b"NSRV" => server = Rd::new(c).s0().unwrap_or_default(),
                        b"FORM" => {
                            let Some((b"NTAG", tag)) = form(c).as_ref().map(|(t, x)| (t, *x)) else {
                                continue;
                            };
                            if list.len() >= 4096 {
                                continue;
                            }
                            let mut n = Node {
                                server: server.clone(),
                                ..Node::default()
                            };
                            for (id, c) in chunks(tag, true) {
                                match &id {
                                    b"NNME" => n.name = Rd::new(c).s0().unwrap_or_default(),
                                    b"FORM" => {
                                        if let Some((b"NDTA", x)) = form(c).as_ref().map(|(t, x)| (t, *x)) {
                                            node_data(x, &mut n.values, &mut n.image, 0);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            list.push(n);
                        }
                        _ => {}
                    }
                }
            }
            b"NCON" => {
                for (id, c) in chunks(inner, true) {
                    let s = Rd::new(c).s0().unwrap_or_default();
                    match &id {
                        b"INME" if links.len() < 4096 => links.push([s, String::new(), String::new()]),
                        b"IINM" => {
                            if let Some(l) = links.last_mut() {
                                l[1] = s;
                            }
                        }
                        b"IINN" => {
                            if let Some(l) = links.last_mut() {
                                l[2] = s;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let feeding = |node: &str, input: &str| {
        let from = links
            .iter()
            .find(|l| l[0] == node && l[1] == input)
            .map(|l| l[2].as_str())?;
        list.iter().position(|n| n.name == from)
    };
    let shader = feeding("Surface", "Material").or_else(|| {
        list.iter()
            .position(|n| n.server == "Principled BSDF" || n.server == "Standard")
    });
    let all = list.iter().map(|n| (n.server.clone(), n.name.clone())).collect();
    let Some(s) = shader else {
        return Nodes {
            all,
            ..Nodes::default()
        };
    };
    let image = feeding(&list[s].name, "Color").and_then(|i| list[i].image.clone());
    let normal = feeding(&list[s].name, "Normal").and_then(|i| list[i].image.clone());
    let inputs = links
        .iter()
        .filter(|l| l[0] == list[s].name)
        .filter_map(|l| {
            let from = list.iter().find(|n| n.name == l[2])?;
            Some((l[1].clone(), from.image.clone().filter(|b| b.file.is_some())?))
        })
        .collect();
    let n = std::mem::take(&mut list[s]);
    Nodes {
        server: n.server,
        name: n.name,
        values: n.values,
        image,
        normal,
        inputs,
        all,
    }
}

/// A node's data: its attribute values wherever they are, and an image node's settings.
fn node_data(d: &[u8], values: &mut Vec<(String, Vec<f64>)>, image: &mut Option<Block>, depth: usize) {
    if depth > 8 {
        return;
    }
    let vec3 = |x: &[u8]| -> Option<[f32; 3]> {
        // A parameter form: its value is the first three floats of the blank chunk in its value form.
        let (_, vprm) = chunks(x, true).find(|(id, _)| *id == FORM)?;
        let (_, vpvl) = form(vprm)?;
        let (_, inner) = chunks(vpvl, true).find(|(id, _)| *id == FORM)?;
        let (_, val) = form(inner)?;
        let (_, blank) = chunks(val, true).find(|(id, _)| *id == BLANK)?;
        Rd::new(blank).vec().filter(|v| v.iter().all(|c| c.is_finite()))
    };
    for (id, c) in chunks(d, true) {
        if id == FORM {
            let Some((t, x)) = form(c) else { continue };
            match &t {
                b"ENTR" => {
                    if values.len() < 256
                        && let Some(e) = entry(x)
                    {
                        values.push(e);
                    }
                }
                b"IIMG" => image.get_or_insert_with(Block::default).file = find_still(x, 0),
                b"ISCL" | b"IPOS" | b"IROT" => {
                    let b = image.get_or_insert_with(Block::default);
                    if let Some(v) = vec3(c) {
                        match &t {
                            b"ISCL" => b.size = v,
                            b"IPOS" => b.center = v,
                            _ => b.rotation = v,
                        }
                    }
                }
                _ => node_data(x, values, image, depth + 1),
            }
            continue;
        }
        let mut r = Rd::new(c);
        match &id {
            b"IUVI" => image.get_or_insert_with(Block::default).uv = r.s0().filter(|s| !s.is_empty()),
            b"IMAP" => {
                image.get_or_insert_with(Block::default).proj = r.u4().and_then(|v| u16::try_from(v).ok()).unwrap_or(5);
            }
            b"IAXS" => {
                image.get_or_insert_with(Block::default).axis = r.u4().and_then(|v| u16::try_from(v).ok()).unwrap_or(2);
            }
            b"IUTL" | b"IVTL" => {
                let b = image.get_or_insert_with(Block::default);
                let w = r.u4().and_then(|v| u16::try_from(v).ok()).unwrap_or(1);
                b.wrap[usize::from(&id == b"IVTL")] = w;
            }
            _ => {}
        }
    }
}

/// The file named by the first still image form under `d`.
fn find_still(d: &[u8], depth: usize) -> Option<String> {
    if depth > 8 {
        return None;
    }
    chunks(d, true).filter(|(id, _)| *id == FORM).find_map(|(_, c)| {
        let (t, x) = form(c)?;
        if &t == b"STIL" {
            still(x)
        } else {
            find_still(x, depth + 1)
        }
    })
}

/// An attribute: its name and value (numbers, as many as it has).
fn entry(d: &[u8]) -> Option<(String, Vec<f64>)> {
    let mut name = None;
    let mut values = Vec::new();
    for (id, c) in chunks(d, true) {
        match &id {
            b"NAME" => name = Rd::new(c).s0(),
            b"FORM" => {
                if let Some((b"VALU", x)) = form(c).as_ref().map(|(t, x)| (t, *x)) {
                    value(x, &mut values, 0);
                }
            }
            _ => {}
        }
    }
    Some((name?, values))
}

/// The numbers in a value form: its blank chunk is a type name and, for numbers, the number; a parameter's value
/// form within holds a count and that many doubles.
fn value(d: &[u8], out: &mut Vec<f64>, depth: usize) {
    if depth > 4 {
        return;
    }
    for (id, c) in chunks(d, true) {
        match &id {
            b"    " => {
                let mut r = Rd::new(c);
                if depth > 0 {
                    let n = r.u4().unwrap_or(0) as usize;
                    for _ in 0..n.min(r.left() / 8).min(16) {
                        out.extend(r.f8());
                    }
                    continue;
                }
                match r.s0().as_deref() {
                    Some("int") => out.extend(r.u4().map(|v| f64::from(v.cast_signed()))),
                    Some("double") => out.extend(r.f8()),
                    Some("float") => out.extend(r.f4().map(f64::from)),
                    _ => {}
                }
            }
            b"FORM" => {
                if let Some((b"VALU", x)) = form(c).as_ref().map(|(t, x)| (t, *x)) {
                    value(x, out, depth + 1);
                }
            }
            _ => {}
        }
    }
}

/// How a surface looks on the layer's polygons.
struct Look {
    material: usize,
    /// The cosine of the smoothing angle; `None` is flat.
    cos: Option<f32>,
    /// The color texture's mapping, when there is one.
    tex: Option<Block>,
    /// The vertex color map: intensity and name.
    vcol: Option<(f32, String)>,
    /// The surface's linear color, which vertex colors blend with.
    color: [f32; 4],
}

/// Phong's highlight size as roughness: glossiness `g` is an exponent of 2^(10g + 2), Blinn-Phong's match for a
/// GGX alpha of sqrt(2 / (n + 2)), which is roughness squared.
fn gloss_roughness(g: f32) -> f32 {
    let n = (10.0 * g.clamp(0.0, 1.0) + 2.0).exp2();
    (2.0 / (n + 2.0)).sqrt().sqrt()
}

/// The glTF material for a surface, its color texture `tex` and whether vertex colors carry its color, with the
/// surface's linear color and alpha.
fn material(s: &Surf, tex: Option<Tex>, vcol: bool) -> (Full, [f32; 4]) {
    let lin = |c: [f32; 3]| c.map(srgb_to_linear);
    let legacy = lin(s.color.unwrap_or([DEFAULT_COLOR; 3]));
    let node = s
        .nodes
        .as_ref()
        .filter(|n| matches!(n.server.as_str(), "Principled BSDF" | "Standard"));
    let principled = node.filter(|n| n.server == "Principled BSDF");
    let num = |k: &str| node.and_then(|n| n.num(k));
    let color = node.and_then(|n| n.rgb("Color")).unwrap_or(legacy).map(|c| c.max(0.0));
    let diff = if principled.is_some() {
        1.0
    } else {
        num("Diffuse").or(s.diff).unwrap_or(1.0).max(0.0)
    };
    let tran = num("Transparency").or(s.tran).unwrap_or(0.0).clamp(0.0, 1.0);
    let alpha = 1.0 - tran;
    let (emissive, metallic, roughness) = if let Some(n) = principled {
        let c = n.rgb("Luminous Color").unwrap_or([1.0; 3]);
        let k = n.num("Luminous").unwrap_or(0.0).max(0.0);
        (
            c.map(|v| v.max(0.0) * k),
            n.num("Metallic").unwrap_or(0.0).clamp(0.0, 1.0),
            n.num("Roughness").unwrap_or(0.5).clamp(0.0, 1.0),
        )
    } else {
        let k = num("Luminosity").or(s.lumi).unwrap_or(0.0).max(0.0);
        let spec = num("Specular").or(s.spec).unwrap_or(0.0);
        let glos = num("Glossiness").or(s.glos).unwrap_or(0.4);
        (
            color.map(|v| v * k),
            0.0,
            if spec > 0.0 { gloss_roughness(glos) } else { 1.0 },
        )
    };
    let rind = num("Refraction Index").or(s.rind).filter(|&r| r >= 1.0 && tran > 0.0);
    let clearcoat = principled.and_then(|n| {
        let k = n.num("Clearcoat").filter(|&k| k > 0.0)?;
        let gloss = n.num("Clearcoat Gloss").unwrap_or(1.0).clamp(0.0, 1.0);
        Some(Clearcoat {
            factor: Factor::of(k.min(1.0)),
            roughness: Factor::of(1.0 - gloss),
            normal: None,
        })
    });
    let tint = if tex.is_some() || vcol { [1.0; 3] } else { color };
    let base = tint.map(|c| (c * diff).clamp(0.0, 1.0));
    let full = Full {
        name: s.name.clone(),
        base_color: [base[0], base[1], base[2], alpha],
        base_color_tex: tex,
        metallic,
        roughness,
        emissive,
        alpha: if alpha < 1.0 { Alpha::Blend } else { Alpha::Opaque },
        double_sided: s.sides == Some(3),
        ior: rind,
        clearcoat,
        ..Full::default()
    };
    let lit = color.map(|c| (c * diff).clamp(0.0, 1.0));
    (full, [lit[0], lit[1], lit[2], alpha])
}

fn clean(v: f32) -> Value {
    let r = (f64::from(v) * 1e6).round() / 1e6;
    json!(r)
}

fn cleans(v: &[f32]) -> Value {
    Value::Array(v.iter().map(|&c| clean(c)).collect())
}

fn id_str(id: [u8; 4]) -> String {
    String::from_utf8_lossy(&id).trim_end().to_owned()
}

/// Everything a texture layer says, for `extras`.
fn block_extras(b: &Block, image: Option<&str>) -> Value {
    let proj = ["planar", "cylindrical", "spherical", "cubic", "front", "uv"];
    let axis = ["x", "y", "z"].get(usize::from(b.axis)).copied().unwrap_or("unknown");
    let mut e = json!({
        "type": id_str(b.kind),
        "channel": id_str(b.chan),
        "enabled": b.enabled,
        "opacity": clean(b.opacity),
        "projection": proj.get(usize::from(b.proj)).copied().unwrap_or("unknown"),
        "axis": axis,
        "wrap": b.wrap,
    });
    if b.blend != 0 {
        e["blend"] = json!(b.blend);
    }
    if b.negative {
        e["negative"] = json!(true);
    }
    if let Some(i) = image {
        e["image"] = json!(i);
    }
    if let Some(uv) = &b.uv {
        e["uvMap"] = json!(uv);
    }
    if b.proj != 5 {
        e["center"] = cleans(&b.center);
        e["size"] = cleans(&b.size);
        if b.rotation.iter().any(|&r| r != 0.0) {
            e["rotation"] = cleans(&b.rotation);
        }
        if b.proj == 1 || b.proj == 2 {
            e["wrapAmount"] = cleans(&b.amount);
        }
        if b.world {
            e["worldCoordinates"] = json!(true);
        }
    }
    e
}

/// Every value a surface holds, for `extras`.
fn surf_extras(s: &Surf, image_of: &dyn Fn(&Block) -> Option<String>) -> Value {
    let mut e = Map::new();
    if !s.source.is_empty() {
        e.insert("source".into(), json!(s.source));
    }
    if let Some(c) = s.color {
        e.insert("color".into(), cleans(&c));
    }
    for (k, v) in [
        ("diffuse", s.diff),
        ("luminosity", s.lumi),
        ("specular", s.spec),
        ("glossiness", s.glos),
        ("reflection", s.refl),
        ("transparency", s.tran),
        ("translucency", s.trnl),
        ("bump", s.bump),
        ("refractiveIndex", s.rind),
    ] {
        if let Some(v) = v {
            e.insert(k.into(), clean(v));
        }
    }
    if let Some(a) = s.sman {
        e.insert("smoothingAngle".into(), clean(a.to_degrees()));
    }
    if let Some(n) = s.sides {
        e.insert("sides".into(), json!(n));
    }
    if let Some(f) = s.flags {
        e.insert("flags".into(), json!(f));
    }
    if let Some((i, t, name)) = &s.vcol {
        e.insert(
            "vertexColor".into(),
            json!({ "map": name, "type": id_str(*t), "intensity": clean(*i) }),
        );
    }
    if !s.blocks.is_empty() {
        let list: Vec<Value> = s
            .blocks
            .iter()
            .map(|b| block_extras(b, image_of(b).as_deref()))
            .collect();
        e.insert("textures".into(), Value::Array(list));
    }
    if let Some(n) = &s.nodes {
        let mut v = Map::new();
        for (k, x) in &n.values {
            let x: Vec<Value> = x.iter().map(|&f| json!((f * 1e6).round() / 1e6)).collect();
            v.insert(k.clone(), if x.len() == 1 { x[0].clone() } else { Value::Array(x) });
        }
        let mut ne = json!({ "shader": n.server, "name": n.name, "values": v });
        ne["nodes"] = n
            .all
            .iter()
            .map(|(s, name)| json!({ "server": s, "name": name }))
            .collect();
        if !n.inputs.is_empty() {
            let mut inputs = Map::new();
            for (input, b) in &n.inputs {
                inputs.insert(input.clone(), block_extras(b, image_of(b).as_deref()));
            }
            ne["imageInputs"] = Value::Object(inputs);
        }
        e.insert("nodes".into(), ne);
    }
    Value::Object(e)
}

/// The values of a vertex map, by point and by (polygon, point).
#[derive(Default)]
struct Lookup {
    point: FastMap<u32, [f32; 4]>,
    corner: FastMap<(u32, u32), [f32; 4]>,
}

impl Lookup {
    fn new(l: &Layer, kinds: &[[u8; 4]], name: &str) -> Self {
        let mut lk = Self::default();
        for m in l
            .maps
            .iter()
            .filter(|m| m.dim > 0 && kinds.contains(&m.kind) && m.name == name)
        {
            for (k, &(pt, poly)) in m.at.iter().enumerate() {
                let Some(v) = m.values.get(k * m.dim..(k + 1) * m.dim) else {
                    break;
                };
                let mut a = [0.0, 0.0, 0.0, 1.0];
                for (d, &x) in a.iter_mut().zip(v) {
                    *d = if x.is_finite() { x } else { 0.0 };
                }
                if m.per_poly {
                    lk.corner.insert((poly, pt), a);
                } else {
                    lk.point.insert(pt, a);
                }
            }
        }
        lk
    }

    fn get(&self, poly: u32, point: u32) -> Option<[f32; 4]> {
        self.corner
            .get(&(poly, point))
            .or_else(|| self.point.get(&point))
            .copied()
    }
}

/// Where a look's texture coordinates come from.
enum UvFrom<'a> {
    None,
    Map(&'a Lookup),
    Project(&'a Block),
}

/// The angles LightWave's projections use: heading about Y from +Z, and pitch up from the XZ plane.
fn heading(x: f32, z: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI};
    if x == 0.0 && z == 0.0 {
        0.0
    } else if z == 0.0 {
        if x < 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 }
    } else if z < 0.0 {
        -(x / z).atan() + PI
    } else {
        -(x / z).atan()
    }
}

fn pitch(x: f32, y: f32, z: f32) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    let d = x.hypot(z);
    if d == 0.0 {
        if y < 0.0 {
            -FRAC_PI_2
        } else if y > 0.0 {
            FRAC_PI_2
        } else {
            0.0
        }
    } else {
        (y / d).atan()
    }
}

/// Where LightWave's projection puts point `p` (in the object's left-handed space) on the image, as glTF texture
/// coordinates (down from the top); `n` is the polygon's normal, which picks a cubic map's side.
fn project(b: &Block, p: [f32; 3], n: [f32; 3]) -> [f32; 2] {
    let mut q = [p[0] - b.center[0], p[1] - b.center[1], p[2] - b.center[2]];
    if b.rotation.iter().any(|&r| r != 0.0) {
        // Undo heading (about Y), pitch (about X) and bank (about Z), last first.
        let [h, pi, bk] = b.rotation;
        let (s, c) = (-h).sin_cos();
        q = [q[0] * c + q[2] * s, q[1], -q[0] * s + q[2] * c];
        let (s, c) = (-pi).sin_cos();
        q = [q[0], q[1] * c - q[2] * s, q[1] * s + q[2] * c];
        let (s, c) = (-bk).sin_cos();
        q = [q[0] * c - q[1] * s, q[0] * s + q[1] * c, q[2]];
    }
    let size = b.size.map(|s| if s.abs() > 1e-12 { s } else { 1.0 });
    let [x, y, z] = q;
    let planar = |axis: u16| match axis {
        0 => [z / size[2] + 0.5, -y / size[1] + 0.5],
        1 => [x / size[0] + 0.5, -z / size[2] + 0.5],
        _ => [x / size[0] + 0.5, -y / size[1] + 0.5],
    };
    let around = |spherical: bool| {
        use std::f32::consts::{PI, TAU};
        let (hx, hy, hz, t) = match b.axis {
            0 => (z, x, -y, -x / size[0] + 0.5),
            1 => (-x, y, z, -y / size[1] + 0.5),
            _ => (-x, z, -y, -z / size[2] + 0.5),
        };
        let lon = (1.0 - heading(hx, hz) / TAU).rem_euclid(1.0);
        let v = if spherical {
            (0.5 - pitch(hx, hy, hz) / PI) * b.amount[1]
        } else {
            t
        };
        [lon * b.amount[0], v]
    };
    match b.proj {
        1 => around(false),
        2 => around(true),
        3 => {
            let a = if n[0].abs() >= n[1].abs() && n[0].abs() >= n[2].abs() {
                0
            } else if n[1].abs() >= n[2].abs() {
                1
            } else {
                2
            };
            planar(a)
        }
        // Front projection follows a camera the object doesn't have: planar along Z stands in.
        4 => planar(2),
        _ => planar(b.axis),
    }
}

/// glTF's wrap mode for LightWave's.
fn wrap_mode(w: u16) -> u32 {
    match w {
        1 => REPEAT,
        2 => MIRROR,
        _ => CLAMP,
    }
}

/// One layer's primitives, each with the point every vertex came from.
struct Built {
    prims: Vec<(Prim, Vec<u32>)>,
}

/// The newell normal of polygon `p`, right-handed over LightWave's order (outward), mirrored into glTF's space.
fn face_normal(l: &Layer, p: &Poly) -> [f32; 3] {
    let s = p.start as usize;
    let cs = l.corners.get(s..s + usize::from(p.len)).unwrap_or(&[]);
    let mut n = [0.0f32; 3];
    for (i, &a) in cs.iter().enumerate() {
        let a = l.points[a as usize];
        let b = l.points[cs[(i + 1) % cs.len()] as usize];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    crease::unit([n[0], n[1], -n[2]]).unwrap_or([0.0, 1.0, 0.0])
}

/// The bits of a vertex's attributes, which vertices are shared by.
type Key = (u32, [u32; 3], [u32; 2], [u32; 4]);

struct Budget {
    clip: u64,
    triangles: usize,
    limit: usize,
}

/// Triangulates `pos` into `out` (indices into it), ear clipping while the budget lasts.
fn triangles(pos: &[[f32; 3]], budget: &mut Budget, out: &mut Vec<[u32; 3]>) {
    let n = pos.len() as u64;
    let cost = if n > 8 { n * n * n } else { 0 };
    if cost > budget.clip {
        out.extend((1..u32_of(pos.len()) - 1).map(|k| [0, k, k + 1]));
        return;
    }
    budget.clip -= cost;
    crate::mesh::triangulate(pos, out);
}

#[allow(clippy::too_many_arguments, reason = "the layer's parts, all needed")]
fn build(l: &Layer, looks: &[Look], look_of: &[usize], budget: &mut Budget) -> Result<Built> {
    let pivot = l.pivot;
    let out = |p: [f32; 3]| [p[0] - pivot[0], p[1] - pivot[1], -(p[2] - pivot[2])];
    let face: Vec<[f32; 3]> = l.polys.iter().map(|p| face_normal(l, p)).collect();
    let used = |p: usize| l.polys[p].shape != Shape::Dropped && l.polys[p].len > 0;
    let smooth_any = (0..l.polys.len()).any(|p| used(p) && looks[look_of[p]].cos.is_some());
    let spans: Vec<(usize, usize)> = l
        .polys
        .iter()
        .map(|p| {
            (
                p.start as usize,
                if p.shape == Shape::Dropped {
                    0
                } else {
                    usize::from(p.len)
                },
            )
        })
        .collect();
    let smooth = if smooth_any {
        crease::corner_normals(
            l.points.len(),
            &spans,
            &l.corners,
            &face,
            |p| looks[look_of[p]].cos.filter(|_| l.polys[p].len >= 3),
            |a, b| look_of[a] == look_of[b] && l.polys[a].group == l.polys[b].group,
        )
    } else {
        Vec::new()
    };
    let norms = l
        .maps
        .iter()
        .find(|m| &m.kind == b"NORM" && m.dim == 3)
        .map(|m| Lookup::new(l, &[*b"NORM"], &m.name));
    let first_uv = l
        .maps
        .iter()
        .find(|m| &m.kind == b"TXUV" && m.dim == 2)
        .map(|m| m.name.as_str());
    let mut uv_maps: HashMap<&str, Lookup> = HashMap::new();
    let mut color_maps: HashMap<&str, Lookup> = HashMap::new();
    for look in looks {
        let uv = match &look.tex {
            Some(b) if b.proj == 5 => b.uv.as_deref().or(first_uv),
            Some(_) => None,
            None => first_uv,
        };
        if let Some(name) = uv
            && !uv_maps.contains_key(name)
        {
            uv_maps.insert(name, Lookup::new(l, &[*b"TXUV"], name));
        }
        if let Some((_, name)) = &look.vcol
            && !color_maps.contains_key(name.as_str())
        {
            color_maps.insert(name, Lookup::new(l, &[*b"RGB ", *b"RGBA"], name));
        }
    }

    let mut order: Vec<usize> = (0..l.polys.len()).filter(|&p| used(p)).collect();
    order.sort_by_key(|&p| look_of[p]);
    let mut prims = Vec::new();
    let (mut corner_idx, mut pos, mut uvs, mut tris) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for group in order.chunk_by(|&a, &b| look_of[a] == look_of[b]) {
        let look = &looks[look_of[group[0]]];
        let uv_from = match &look.tex {
            Some(b) if b.proj != 5 => UvFrom::Project(b),
            Some(b) => {
                b.uv.as_deref()
                    .or(first_uv)
                    .and_then(|n| uv_maps.get(n))
                    .map_or(UvFrom::None, UvFrom::Map)
            }
            None => first_uv.and_then(|n| uv_maps.get(n)).map_or(UvFrom::None, UvFrom::Map),
        };
        // A textured look always has coordinates, zero where the map has none.
        let has_uv = look.tex.is_some() || !matches!(uv_from, UvFrom::None);
        let colors = look
            .vcol
            .as_ref()
            .and_then(|(i, n)| Some((*i, color_maps.get(n.as_str())?)));
        let normals_on = look.cos.is_some() || norms.is_some();
        let mut prim = Prim {
            material: look.material,
            ..Prim::default()
        };
        let mut points_of: Vec<u32> = Vec::new();
        let mut index: FastMap<Key, u32> = FastMap::default();
        for &p in group {
            let poly = l.polys[p];
            let s = poly.start as usize;
            let cs = &l.corners[s..s + usize::from(poly.len)];
            let pu = u32_of(p);
            uvs.clear();
            match uv_from {
                UvFrom::None => {}
                UvFrom::Map(lk) => {
                    uvs.extend(
                        cs.iter()
                            .map(|&c| lk.get(pu, c).map_or([0.0, 0.0], |v| [v[0], 1.0 - v[1]])),
                    );
                }
                UvFrom::Project(b) => {
                    uvs.extend(cs.iter().map(|&c| project(b, l.points[c as usize], face[p])));
                    // Around a cylinder or sphere, a polygon across the seam takes the far side's coordinates.
                    if b.proj == 1 || b.proj == 2 {
                        let w = b.amount[0].abs().max(1e-6);
                        let (lo, hi) = uvs
                            .iter()
                            .fold((f32::MAX, f32::MIN), |(lo, hi), t| (lo.min(t[0]), hi.max(t[0])));
                        if hi - lo > w * 0.5 {
                            for t in &mut uvs {
                                if t[0] < w * 0.5 {
                                    t[0] += w;
                                }
                            }
                        }
                    }
                }
            }
            if has_uv && uvs.len() < cs.len() {
                uvs.resize(cs.len(), [0.0, 0.0]);
            }
            corner_idx.clear();
            pos.clear();
            // LightWave's corners run clockwise; turned around, they run counter-clockwise in glTF's space.
            for j in (0..cs.len()).rev() {
                let c = cs[j];
                let normal = if normals_on {
                    let given = norms
                        .as_ref()
                        .and_then(|n| n.get(pu, c))
                        .and_then(|v| crease::unit([v[0], v[1], -v[2]]));
                    given.unwrap_or_else(|| if look.cos.is_some() { smooth[s + j] } else { face[p] })
                } else {
                    [0.0; 3]
                };
                let uv = if has_uv { uvs[j] } else { [0.0; 2] };
                let color = colors.map(|(i, lk)| match lk.get(pu, c) {
                    Some(v) => {
                        let v = [srgb_to_linear(v[0]), srgb_to_linear(v[1]), srgb_to_linear(v[2]), v[3]];
                        std::array::from_fn(|k| look.color[k] * (1.0 - i) + v[k] * i)
                    }
                    None => look.color,
                });
                let key: Key = (
                    c,
                    normal.map(f32::to_bits),
                    uv.map(f32::to_bits),
                    color.unwrap_or_default().map(f32::to_bits),
                );
                let v = *index.entry(key).or_insert_with(|| {
                    prim.positions.push(out(l.points[c as usize]));
                    if normals_on {
                        prim.normals.push(normal);
                    }
                    if has_uv {
                        prim.uvs.push(uv);
                    }
                    if let Some(col) = color {
                        prim.colors.push(col.map(|x| x.clamp(0.0, 1.0)));
                    }
                    points_of.push(c);
                    u32_of(points_of.len() - 1)
                });
                corner_idx.push(v);
                pos.push(out(l.points[c as usize]));
            }
            match cs.len() {
                1 => prim.points.push(corner_idx[0]),
                2 => prim.lines.extend_from_slice(&corner_idx),
                _ => {
                    tris.clear();
                    triangles(&pos, budget, &mut tris);
                    budget.triangles += tris.len();
                    ensure!(
                        budget.triangles <= budget.limit,
                        LimitExceeded,
                        "more than {} triangles",
                        budget.limit
                    );
                    prim.indices
                        .extend(tris.iter().flatten().map(|&k| corner_idx[k as usize]));
                }
            }
        }
        if !prim.is_empty() {
            prims.push((prim, points_of));
        }
    }
    Ok(Built { prims })
}

/// The layers' order as a forest: each layer's parent (a layer naming one that isn't there, itself, or a loop is a
/// root), by layer index.
fn parents(layers: &[Layer]) -> Vec<Option<usize>> {
    let mut by_number: HashMap<u16, usize> = HashMap::new();
    for (i, l) in layers.iter().enumerate() {
        by_number.entry(l.number).or_insert(i);
    }
    let mut parent: Vec<Option<usize>> = layers
        .iter()
        .enumerate()
        .map(|(i, l)| l.parent.and_then(|n| by_number.get(&n).copied()).filter(|&p| p != i))
        .collect();
    // 0 unseen, 1 on the current path, 2 done.
    let mut state = vec![0u8; layers.len()];
    for i in 0..layers.len() {
        let mut path = Vec::new();
        let mut j = i;
        while state[j] == 0 {
            state[j] = 1;
            path.push(j);
            match parent[j] {
                Some(p) if state[p] == 1 => {
                    parent[j] = None;
                    break;
                }
                Some(p) => j = p,
                None => break,
            }
        }
        for k in path {
            state[k] = 2;
        }
    }
    parent
}

/// Reads a LightWave object.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let data = input.read()?;
    let m = parse(&data)?;
    let dir = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir);
    let mut image_of: HashMap<String, Option<usize>> = HashMap::new();
    let textures = keep.materials && keep.textures;
    let clip_file = |b: &Block| -> Option<String> {
        if let Some(f) = &b.file {
            return Some(f.clone());
        }
        let mut c = b.clip?;
        for _ in 0..8 {
            match m.clips.get(&c)? {
                Clip::File(f) => return Some(f.clone()),
                Clip::Ref(r) => c = *r,
            }
        }
        None
    };
    // Relative to the object, or to the content folder above the object's folder.
    let find = |name: &str| {
        assets
            .find(name, dir)
            .or_else(|| assets.find(name, dir.parent().unwrap_or(dir)))
    };
    let mut image_for = |name: &str| -> Option<usize> {
        if !textures {
            return None;
        }
        *image_of
            .entry(name.to_owned())
            .or_insert_with(|| images.file(assets, find(name)?, name))
    };

    // The materials and looks, a surface's by its index; tags naming no surface get LightWave's default.
    let mut materials: Vec<Full> = Vec::new();
    let mut looks: Vec<Look> = Vec::new();
    let mut dropped_textures = 0usize;
    for s in &m.surfs {
        // The top enabled image layer on the color channel.
        let mut color_blocks: Vec<&Block> = s
            .blocks
            .iter()
            .filter(|b| b.enabled && &b.kind == b"IMAP" && &b.chan == b"COLR" && clip_file(b).is_some())
            .collect();
        color_blocks.sort_by(|a, b| a.ordinal.cmp(&b.ordinal));
        let block = s
            .nodes
            .as_ref()
            .and_then(|n| n.image.as_ref())
            .filter(|b| b.file.is_some())
            .or_else(|| color_blocks.last().copied());
        let image = block.and_then(|b| image_for(&clip_file(b)?));
        // A missing image is counted whether or not textures are kept, as a full read counts it.
        let missing = || block.and_then(clip_file).is_none_or(|f| find(&f).is_none());
        if block.is_some() && image.is_none() && (textures || missing()) {
            dropped_textures += 1;
        }
        let tex = image.zip(block).map(|(i, b)| {
            let mut t = Tex::new(i);
            t.wrap = [wrap_mode(b.wrap[0]), wrap_mode(b.wrap[1])];
            if b.proj == 1 || b.proj == 2 {
                t.wrap[0] = REPEAT;
            }
            t
        });
        let vcol = s
            .vcol
            .as_ref()
            .filter(|(i, _, n)| *i > 0.0 && !n.is_empty())
            .map(|(i, _, n)| (i.min(1.0), n.clone()));
        // A node graph's normal map, on the color texture's coordinates.
        let normal_block = s
            .nodes
            .as_ref()
            .and_then(|n| n.normal.as_ref())
            .filter(|n| n.file.is_some() && n.proj == 5)
            .filter(|n| image.and(block).is_none_or(|b| b.proj == 5 && b.uv == n.uv));
        let normal = normal_block.and_then(|b| image_for(&clip_file(b)?));
        let (mut full, color) = material(s, tex, vcol.is_some());
        if let Some(i) = normal {
            let mut t = Tex::new(i);
            t.wrap = normal_block.map_or([REPEAT; 2], |b| [wrap_mode(b.wrap[0]), wrap_mode(b.wrap[1])]);
            full.normal_tex = Some(t);
        }
        if keep.extras {
            full.extras = Some(surf_extras(s, &|b: &Block| clip_file(b)));
        }
        let sman = s.sman.unwrap_or(0.0);
        looks.push(Look {
            material: materials.len(),
            cos: (sman > 0.0).then(|| sman.min(std::f32::consts::PI).cos()),
            tex: image.and(block).or(normal.and(normal_block)).cloned(),
            vcol,
            color,
        });
        materials.push(full);
    }
    let mut surf_by_name: HashMap<&str, usize> = HashMap::new();
    for (i, s) in m.surfs.iter().enumerate() {
        surf_by_name.entry(s.name.as_str()).or_insert(i);
    }
    let mut look_by_tag: HashMap<u32, usize> = HashMap::new();
    let mut default_look = |tag: u32, looks: &mut Vec<Look>, materials: &mut Vec<Full>| -> usize {
        let name = m.tags.get(tag as usize).map_or("Default", String::as_str);
        if let Some(&s) = surf_by_name.get(name) {
            return s;
        }
        *look_by_tag.entry(tag).or_insert_with(|| {
            let s = Surf {
                name: name.to_owned(),
                ..Surf::default()
            };
            let (full, color) = material(&s, None, false);
            looks.push(Look {
                material: materials.len(),
                cos: None,
                tex: None,
                vcol: None,
                color,
            });
            materials.push(full);
            looks.len() - 1
        })
    };

    let mut budget = Budget {
        clip: CLIP_BUDGET,
        triangles: 0,
        limit: input.limit,
    };
    let mut b = Builder::default();
    let mut meshes: Vec<Value> = Vec::new();
    let mut mesh_of: Vec<Option<usize>> = Vec::with_capacity(m.layers.len());
    let mut lines_or_points = false;
    for l in &m.layers {
        let look_of: Vec<usize> = l
            .polys
            .iter()
            .map(|p| default_look(p.surf, &mut looks, &mut materials))
            .collect();
        let built = build(l, &looks, &look_of, &mut budget)?;
        // Morph maps: relative offsets, or absolute positions.
        let mut morphs: Vec<(&str, bool)> = Vec::new();
        if keep.morph_targets {
            for map in l
                .maps
                .iter()
                .filter(|x| x.dim == 3 && matches!(&x.kind, b"MORF" | b"SPOT"))
            {
                let t = (map.name.as_str(), &map.kind == b"SPOT");
                if !morphs.contains(&t) && morphs.len() < 256 {
                    morphs.push(t);
                }
            }
        }
        let lookups: Vec<Lookup> = morphs
            .iter()
            .map(|&(n, spot)| Lookup::new(l, &[if spot { *b"SPOT" } else { *b"MORF" }], n))
            .collect();
        let mut primitives = Vec::new();
        for (prim, points_of) in built.prims {
            lines_or_points |= !prim.lines.is_empty() || !prim.points.is_empty();
            let material = prim.material;
            let targets: Vec<Vec<[f32; 3]>> = lookups
                .iter()
                .zip(&morphs)
                .map(|(lk, &(_, spot))| {
                    points_of
                        .iter()
                        .map(|&c| {
                            let p = l.points[c as usize];
                            let d = lk.point.get(&c).map_or([0.0; 3], |v| {
                                if spot {
                                    [v[0] - p[0], v[1] - p[1], v[2] - p[2]]
                                } else {
                                    [v[0], v[1], v[2]]
                                }
                            });
                            [d[0], d[1], -d[2]]
                        })
                        .collect()
                })
                .collect();
            let jsons = b.primitives(prim);
            let targets: Vec<Value> = targets
                .into_iter()
                .map(|t| json!({ "POSITION": b.positions(t) }))
                .collect();
            for mut j in jsons {
                j["material"] = json!(material);
                if !targets.is_empty() {
                    j["targets"] = Value::Array(targets.clone());
                }
                primitives.push(j);
            }
        }
        if primitives.is_empty() {
            mesh_of.push(None);
            continue;
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(n) = &l.name {
            mesh["name"] = json!(n);
        }
        if !morphs.is_empty() {
            mesh["weights"] = json!(vec![0.0; morphs.len()]);
            mesh["extras"] = json!({ "targetNames": morphs.iter().map(|m| m.0).collect::<Vec<_>>() });
        }
        meshes.push(mesh);
        mesh_of.push(Some(meshes.len() - 1));
    }
    ensure!(!meshes.is_empty(), Empty, "this LightWave object has no polygons");

    // The layers as nodes, each at its pivot, relative to its parent's.
    let parent = parents(&m.layers);
    let mut nodes: Vec<Value> = Vec::with_capacity(m.layers.len());
    let mut roots = Vec::new();
    for (i, l) in m.layers.iter().enumerate() {
        let name = l.name.clone().unwrap_or_else(|| format!("Layer {}", l.number));
        let mut node = json!({ "name": name });
        let base = parent[i].map_or([0.0; 3], |p| m.layers[p].pivot);
        let t = [l.pivot[0] - base[0], l.pivot[1] - base[1], -(l.pivot[2] - base[2])];
        if t.iter().any(|&c| c != 0.0) {
            node["translation"] = json!(t);
        }
        if let Some(mesh) = mesh_of[i] {
            node["mesh"] = json!(mesh);
        }
        let children: Vec<usize> = (0..m.layers.len()).filter(|&c| parent[c] == Some(i)).collect();
        if !children.is_empty() {
            node["children"] = json!(children);
        }
        if parent[i].is_none() {
            roots.push(i);
        }
        if keep.extras {
            let mut e = json!({ "layer": l.number });
            if l.hidden {
                e["hidden"] = json!(true);
            }
            if let Some(p) = l.parent.filter(|_| parent[i].is_none()) {
                e["missingParent"] = json!(p);
            }
            let parts: Vec<&str> = l
                .parts
                .iter()
                .filter_map(|&t| m.tags.get(t as usize))
                .map(String::as_str)
                .collect();
            if !parts.is_empty() {
                e["parts"] = json!(parts);
            }
            let maps: Vec<Value> = l
                .maps
                .iter()
                .map(|x| {
                    json!({ "type": id_str(x.kind), "name": x.name, "entries": x.entries, "perPolygon": x.per_poly })
                })
                .collect();
            if !maps.is_empty() {
                e["vertexMaps"] = Value::Array(maps);
            }
            let groups = l.polys.iter().filter(|p| p.group != NONE).count();
            if groups > 0 {
                e["smoothingGroupPolygons"] = json!(groups);
            }
            node["extras"] = e;
        }
        nodes.push(node);
    }

    // The format fixes the units and axes, so they are declared.
    let mut info = SourceInfo {
        format: Some(m.kind.name().into()),
        meters_per_unit: Fact::Declared(1.0),
        unit: Some("meter".into()),
        up_axis: Fact::Declared(Axis::PosY),
        handedness: Fact::Declared(Handedness::Left),
        description: m.description.clone(),
        comments: m.comment.iter().cloned().collect(),
        ..SourceInfo::default()
    };
    info.converted.handedness = true;
    let mut source = json!({});
    for (key, n) in [
        ("subdivisionPatches", m.patches),
        ("droppedCurves", m.curves),
        ("droppedBones", m.bones),
        ("droppedMetaballs", m.metaballs),
        ("droppedPolygons", m.bad_polys + m.other_polys),
        ("droppedEnvelopes", m.envelopes),
        ("droppedTextures", dropped_textures),
    ] {
        if n > 0 {
            source[key] = json!(n);
        }
    }
    if m.truncated {
        source["truncated"] = json!(true);
    }
    if let Value::Object(map) = &mut source {
        info.write(map);
    }
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(budget.triangles));
    facts.insert("layers".into(), json!(m.layers.len()));
    facts.insert("surfaces".into(), json!(m.surfs.len()));

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": roots }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    b.materials(&materials, images.list, &mut json);
    let mut doc = b.finish(json, facts);
    if lines_or_points {
        doc.set_lines_and_points(true);
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;
    use crate::formats::Input;
    use crate::mesh::Mesh;

    /// A chunk with a four-byte size.
    fn ck(id: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        v.extend(u32::try_from(body.len()).unwrap().to_be_bytes());
        v.extend(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    }

    /// A subchunk with a two-byte size.
    fn sub(id: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        v.extend(u16::try_from(body.len()).unwrap().to_be_bytes());
        v.extend(body);
        if body.len() % 2 == 1 {
            v.push(0);
        }
        v
    }

    fn file(kind: [u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut body = kind.to_vec();
        for c in chunks {
            body.extend(c);
        }
        ck(*b"FORM", &body)
    }

    fn s0(s: &str) -> Vec<u8> {
        let mut v = s.as_bytes().to_vec();
        v.push(0);
        if v.len() % 2 == 1 {
            v.push(0);
        }
        v
    }

    fn f4(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_be_bytes()).collect()
    }

    fn u2(v: u16) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    fn vx(v: u32) -> Vec<u8> {
        if v < 0xFF00 {
            u2(u16::try_from(v).unwrap())
        } else {
            (v | 0xFF00_0000).to_be_bytes().to_vec()
        }
    }

    fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    /// LightWave points for glTF ones: z mirrored.
    fn pnts(q: &[[f32; 3]]) -> Vec<u8> {
        ck(
            *b"PNTS",
            &f4(&q.iter().flat_map(|p| [p[0], p[1], -p[2]]).collect::<Vec<_>>()),
        )
    }

    /// An LWO2 polygon list of glTF counterclockwise polygons, written the LightWave way round.
    fn pols(kind: [u8; 4], polys: &[&[u32]]) -> Vec<u8> {
        let mut b = kind.to_vec();
        for p in polys {
            b.extend(u2(u16::try_from(p.len()).unwrap()));
            for &v in p.iter().rev() {
                b.extend(vx(v));
            }
        }
        ck(*b"POLS", &b)
    }

    fn layr(number: u16, pivot: [f32; 3], name: &str, parent: Option<u16>) -> Vec<u8> {
        let mut b = cat(&[u2(number), u2(0), f4(&pivot), s0(name)]);
        if let Some(p) = parent {
            b.extend(u2(p));
        }
        ck(*b"LAYR", &b)
    }

    const CUBE: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    const FACES: [&[u32]; 6] = [
        &[0, 3, 2, 1],
        &[4, 5, 6, 7],
        &[0, 1, 5, 4],
        &[3, 7, 6, 2],
        &[0, 4, 7, 3],
        &[1, 2, 6, 5],
    ];

    /// A textured unit cube in a layer pivoted at (1, 2, 3) in glTF terms, with UVs (and one per polygon at
    /// point 0), a morph and a smoothing angle.
    fn cube(sman: f32) -> Vec<u8> {
        let mut ptag = b"SURF".to_vec();
        for i in 0..6 {
            ptag.extend(cat(&[vx(i), u2(0)]));
        }
        let mut txuv = cat(&[b"TXUV".to_vec(), u2(2), s0("uv")]);
        for (i, p) in CUBE.iter().enumerate() {
            txuv.extend(cat(&[vx(u32::try_from(i).unwrap()), f4(&[p[0], p[1]])]));
        }
        let vmad = cat(&[b"TXUV".to_vec(), u2(2), s0("uv"), vx(0), vx(0), f4(&[0.5, 0.25])]);
        let morf = cat(&[b"MORF".to_vec(), u2(3), s0("smile"), vx(6), f4(&[0.0, 0.0, 1.0])]);
        let imap = cat(&[s0("\u{80}"), sub(*b"CHAN", b"COLR"), sub(*b"ENAB", &u2(1))]);
        let blok = cat(&[
            sub(*b"IMAP", &imap),
            sub(*b"PROJ", &u2(5)),
            sub(*b"IMAG", &vx(1)),
            sub(*b"WRAP", &cat(&[u2(1), u2(2)])),
            sub(*b"VMAP", &s0("uv")),
        ]);
        let surf = cat(&[
            s0("Skin"),
            s0(""),
            sub(*b"COLR", &cat(&[f4(&[1.0, 0.0, 0.0]), vx(0)])),
            sub(*b"DIFF", &cat(&[f4(&[0.8]), vx(0)])),
            sub(*b"SMAN", &cat(&[f4(&[sman]), vx(0)])),
            sub(*b"BLOK", &blok),
        ]);
        file(
            *b"LWO2",
            &[
                ck(*b"TAGS", &s0("Skin")),
                layr(0, [1.0, 2.0, -3.0], "Box", None),
                pnts(&CUBE),
                pols(*b"FACE", &FACES),
                ck(*b"PTAG", &ptag),
                ck(*b"VMAP", &txuv),
                ck(*b"VMAD", &vmad),
                ck(*b"VMAP", &morf),
                ck(
                    *b"CLIP",
                    &cat(&[1u32.to_be_bytes().to_vec(), sub(*b"STIL", &s0("skin.png"))]),
                ),
                ck(*b"SURF", &surf),
                ck(*b"DESC", &s0("a box")),
            ],
        )
    }

    fn png() -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    fn read(bytes: &[u8], assets: &Memory) -> Result<Document<'static>> {
        load(Input::memory(bytes, "t.lwo"), assets, Keep::ALL)
    }

    fn flat(doc: &Document<'_>) -> Mesh {
        doc.flatten(Pose::Rest).unwrap().unwrap()
    }

    /// The (area, signed volume) of the mesh's triangles.
    fn measure(m: &Mesh) -> (f64, f64) {
        let (mut area, mut vol) = (0.0, 0.0);
        for p in &m.prims {
            let at = |i: u32| p.positions[i as usize].map(f64::from);
            for t in p.indices.chunks_exact(3) {
                let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                area += (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() / 2.0;
                vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    / 6.0;
            }
        }
        (area, vol)
    }

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    fn near(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
    }

    #[test]
    fn cube_converts_whole() {
        let assets = Memory::new().with("skin.png", png());
        let doc = read(&cube(0.5), &assets).unwrap();
        let m = flat(&doc);
        let (area, vol) = measure(&m);
        assert!((area - 6.0).abs() < 1e-5 && (vol - 1.0).abs() < 1e-5, "{area} {vol}");
        assert_eq!(doc.facts["triangles"], 12);
        // The layer's node sits at its pivot, and its points are relative to it.
        let node = &doc.json["nodes"][0];
        assert_eq!(node["name"], "Box");
        assert_eq!(node["translation"], json!([1.0, 2.0, 3.0]));
        let p = &m.prims[0];
        assert!(
            p.positions
                .iter()
                .all(|q| q.iter().all(|c| (-0.001..1.001).contains(c)))
        );
        // Below the smoothing angle every corner stays flat.
        for (q, n) in p.positions.iter().zip(&p.normals) {
            assert!(n.iter().filter(|c| c.abs() > 0.999).count() == 1, "{q:?} {n:?}");
        }
        // UVs flip to glTF's v, and the per-polygon one wins at its corner.
        for (q, uv) in p.positions.iter().zip(&p.uvs) {
            let flipped = [q[0], 1.0 - q[1]];
            assert!(close(*uv, flipped) || close(*uv, [0.5, 0.75]), "{q:?} {uv:?}");
        }
        assert_eq!(p.uvs.iter().filter(|uv| close(**uv, [0.5, 0.75])).count(), 1);
        // The textured surface: white under the image, which repeats across and mirrors down.
        let mat = &m.materials[0];
        assert_eq!(mat.name, "Skin");
        assert_eq!(mat.color, [0.8, 0.8, 0.8, 1.0]);
        assert_eq!(m.images.len(), 1);
        assert_eq!(doc.json["samplers"][0]["wrapS"], REPEAT);
        assert_eq!(doc.json["samplers"][0]["wrapT"], MIRROR);
        // The morph is named and starts at rest.
        let mesh = &doc.json["meshes"][0];
        assert_eq!(mesh["extras"]["targetNames"], json!(["smile"]));
        assert_eq!(mesh["weights"], json!([0.0]));
        let source = &doc.json["asset"]["extras"]["source"];
        assert_eq!(source["format"], "LWO2");
        assert_eq!(source["description"], "a box");
        let info = doc.source_info();
        assert_eq!(info.meters_per_unit, Fact::Declared(1.0));
        assert_eq!(info.unit.as_deref(), Some("meter"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Left));
        assert!(info.converted.handedness && !info.converted.units && !info.converted.up_axis);
        assert_eq!(info.description.as_deref(), Some("a box"));
        assert_eq!(
            doc.json["nodes"][0]["extras"]["vertexMaps"].as_array().unwrap().len(),
            3
        );

        // Without the image the texture is dropped and counted, and the surface keeps its color.
        let doc = read(&cube(0.5), &Memory::new()).unwrap();
        assert_eq!(doc.json["asset"]["extras"]["source"]["droppedTextures"], 1);
        let m = flat(&doc);
        let red = srgb_to_linear(1.0) * 0.8;
        assert_eq!(m.materials[0].color, [red, 0.0, 0.0, 1.0]);
        assert_eq!(m.images.len(), 0);
    }

    #[test]
    fn smoothing_angle_joins_corners() {
        let doc = read(&cube(1.6), &Memory::new()).unwrap();
        let m = flat(&doc);
        let p = &m.prims[0];
        let k = 1.0 / 3f32.sqrt();
        for (q, n) in p.positions.iter().zip(&p.normals) {
            let want = q.map(|c| if c > 0.5 { k } else { -k });
            assert!(near(*n, want), "{q:?} {n:?}");
        }
    }

    #[test]
    fn layers_nest_at_their_pivots() {
        let tri: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let shifted = |d: [f32; 3]| tri.map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]]);
        let bytes = file(
            *b"LWO2",
            &[
                layr(1, [0.0, 0.0, 0.0], "Base", None),
                pnts(&tri),
                pols(*b"FACE", &[&[0, 1, 2]]),
                layr(2, [1.0, 2.0, -3.0], "Arm", Some(1)),
                pnts(&shifted([1.0, 2.0, 3.0])),
                pols(*b"FACE", &[&[0, 1, 2]]),
                layr(3, [1.0, 3.0, -3.0], "", Some(2)),
                pnts(&shifted([1.0, 3.0, 3.0])),
                pols(*b"FACE", &[&[0, 1, 2]]),
                // A two-point polygon is a line; a curve is dropped.
                pols(*b"FACE", &[&[0, 1]]),
                pols(*b"CURV", &[&[0, 1, 2]]),
            ],
        );
        let doc = read(&bytes, &Memory::new()).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        assert_eq!(doc.json["scenes"][0]["nodes"], json!([0]));
        assert_eq!(nodes[0]["children"], json!([1]));
        assert_eq!(nodes[1]["children"], json!([2]));
        assert_eq!(nodes[1]["translation"], json!([1.0, 2.0, 3.0]));
        assert_eq!(nodes[2]["translation"], json!([0.0, 1.0, 0.0]));
        assert_eq!(nodes[2]["name"], "Layer 3");
        assert_eq!(doc.json["asset"]["extras"]["source"]["droppedCurves"], 1);
        let m = flat(&doc);
        assert_eq!(m.triangles(), 3);
        assert_eq!(m.lines(), 1);
        // Each layer's triangle lands where its points were.
        for d in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 3.0, 3.0]] {
            let found = m.prims.iter().any(|p| p.positions.iter().any(|q| near(*q, d)));
            assert!(found, "{d:?}");
        }

        // Parents in a loop, or missing: every layer still ends up under a root.
        let looped = file(
            *b"LWO2",
            &[
                layr(1, [0.0; 3], "A", Some(2)),
                pnts(&tri),
                pols(*b"FACE", &[&[0, 1, 2]]),
                layr(2, [0.0; 3], "B", Some(1)),
                pnts(&tri),
                pols(*b"FACE", &[&[0, 1, 2]]),
                layr(3, [0.0; 3], "C", Some(9)),
            ],
        );
        let doc = read(&looped, &Memory::new()).unwrap();
        assert_eq!(flat(&doc).triangles(), 2);
        assert_eq!(doc.json["scenes"][0]["nodes"].as_array().unwrap().len(), 2);
        assert_eq!(doc.json["nodes"][2]["extras"]["missingParent"], 9);
    }

    #[test]
    fn bridged_holes_triangulate() {
        // A square with a square hole, joined by an edge walked both ways, as modelers write it.
        let pts: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
        ];
        let bytes = file(
            *b"LWO2",
            &[pnts(&pts), pols(*b"FACE", &[&[0, 1, 2, 3, 0, 4, 7, 6, 5, 4]])],
        );
        let m = flat(&read(&bytes, &Memory::new()).unwrap());
        let (area, _) = measure(&m);
        assert!((area - 12.0).abs() < 1e-4, "{area}");
        assert!(m.prims[0].normals.iter().all(|n| near(*n, [0.0, 0.0, 1.0])));
    }

    /// The cube as a LightWave 5 object, its polygons on surface 1 (with `detail` polygons said to follow each).
    fn classic(detail: bool, surf: &[Vec<u8>]) -> Vec<u8> {
        let mut pols = Vec::new();
        for f in FACES {
            pols.extend(u2(4));
            for &v in f.iter().rev() {
                pols.extend(u2(u16::try_from(v).unwrap()));
            }
            if detail {
                pols.extend(u2(0xFFFF));
                pols.extend(u2(1));
            } else {
                pols.extend(u2(1));
            }
        }
        file(
            *b"LWOB",
            &[
                pnts(&CUBE),
                ck(*b"SRFS", &s0("Red")),
                ck(*b"POLS", &pols),
                ck(*b"SURF", &cat(surf)),
            ],
        )
    }

    #[test]
    fn reads_lightwave_5() {
        let bytes = classic(
            false,
            &[
                s0("Red"),
                sub(*b"COLR", &[255, 0, 0, 0]),
                sub(*b"DIFF", &u2(128)),
                sub(*b"VDIF", &f4(&[0.75])),
                sub(*b"SMAN", &f4(&[0.5])),
            ],
        );
        let doc = read(&bytes, &Memory::new()).unwrap();
        let m = flat(&doc);
        let (area, vol) = measure(&m);
        assert!((area - 6.0).abs() < 1e-5 && (vol - 1.0).abs() < 1e-5, "{area} {vol}");
        assert_eq!(m.materials[0].name, "Red");
        assert_eq!(m.materials[0].color, [0.75, 0.0, 0.0, 1.0]);
        assert_eq!(doc.json["asset"]["extras"]["source"]["format"], "LWOB");
    }

    #[test]
    fn reads_lightwave_2015_forms() {
        let name = ck(*b"    ", &cat(&[s0("Gold"), s0("")]));
        let colr = ck(*b"COLR", &cat(&[f4(&[1.0, 1.0, 0.0]), vx(0)]));
        let surf = ck(*b"FORM", &cat(&[b"SURF".to_vec(), name, colr]));
        let bytes = file(
            *b"LWO3",
            &[
                ck(*b"TAGS", &s0("Gold")),
                layr(0, [0.0; 3], "Only", None),
                pnts(&CUBE),
                pols(*b"FACE", &FACES),
                ck(*b"PTAG", &cat(&[b"SURF".to_vec(), vx(0), u2(0), vx(5), u2(0)])),
                surf,
            ],
        );
        let doc = read(&bytes, &Memory::new()).unwrap();
        let m = flat(&doc);
        assert_eq!(m.triangles(), 12);
        let gold = m.materials.iter().find(|x| x.name == "Gold").unwrap();
        assert_eq!(gold.color, [1.0, 1.0, 0.0, 1.0]);
        // Untagged polygons get LightWave's default surface.
        assert!(m.materials.iter().any(|x| x.name == "Default"));
        assert_eq!(doc.json["asset"]["extras"]["source"]["format"], "LWO3");
    }

    #[test]
    fn rejects_what_is_not_an_object() {
        assert!(read(b"", &Memory::new()).is_err());
        assert!(read(b"FORM\0\0\0\x04ILBM", &Memory::new()).is_err());
        // An object with nothing drawn.
        assert!(read(&file(*b"LWO2", &[layr(0, [0.0; 3], "", None)]), &Memory::new()).is_err());
    }

    #[test]
    fn hostile_files_never_panic() {
        let assets = Memory::new().with("skin.png", png());
        let mut cut_short = cube(0.5);
        cut_short.truncate(cut_short.len() - 40);
        let doc = read(&cut_short, &assets).unwrap();
        assert_eq!(doc.json["asset"]["extras"]["source"]["truncated"], true);
        // Huge counts and indices.
        let wild = file(
            *b"LWO2",
            &[
                pnts(&CUBE),
                ck(*b"POLS", &cat(&[b"FACE".to_vec(), u2(0x3FF), vx(0xFF_FFFF), vx(1)])),
                ck(
                    *b"POLS",
                    &cat(&[b"FACE".to_vec(), u2(3), vx(0), vx(1), vx(0x00FE_FFFF)]),
                ),
                ck(
                    *b"VMAD",
                    &cat(&[b"TXUV".to_vec(), u2(2), s0("uv"), vx(0), vx(0xFF_FFFF), f4(&[0.0, 0.0])]),
                ),
                ck(*b"VMAP", &cat(&[b"TXUV".to_vec(), u2(9), s0("uv"), vx(0)])),
                ck(*b"PTAG", &cat(&[b"SURF".to_vec(), vx(0xFF_FFFF), u2(0xFFFF)])),
            ],
        );
        assert!(read(&wild, &assets).is_err());
        let textured = classic(
            true,
            &[
                s0("Red"),
                sub(*b"CTEX", &s0("Planar Image Map")),
                sub(*b"TIMG", &s0("skin.png")),
            ],
        );
        let mut rng = crate::test_rng::Pcg32::new(23);
        for base in [cube(0.5), cube(2.0), textured] {
            for n in 0..base.len() {
                if let Ok(doc) = read(&base[..n], &assets) {
                    doc.to_glb().unwrap();
                }
            }
            for _ in 0..400 {
                let mut b = base.clone();
                for _ in 0..=rng.below(8) {
                    let at = rng.below(u32::try_from(b.len()).unwrap()) as usize;
                    b[at] = rng.next_u32().to_le_bytes()[0];
                }
                if let Ok(doc) = read(&b, &assets) {
                    doc.to_glb().unwrap();
                }
            }
        }
    }
}
