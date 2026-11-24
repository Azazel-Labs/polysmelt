//! Meshes: sources read through their accessors, the primitives' index lists turned into triangles (or line
//! segments) over vertices that share every attribute, and polygons with holes bridged and ear-clipped.

use std::collections::HashMap;
use std::rc::Rc;

use super::xml::{self, Dom};
use crate::mesh::{Prim, srgb_to_linear, triangulate};
use crate::num::u32_of;

/// The values of a `<source>`, read through its accessor: `width` numbers to an element.
#[derive(Debug, Default)]
pub(super) struct Acc<T> {
    pub width: usize,
    pub data: Vec<T>,
}

impl<T> Acc<T> {
    pub(super) fn count(&self) -> usize {
        self.data.len().checked_div(self.width).unwrap_or(0)
    }

    /// Element `i`, when there is one.
    pub(super) fn get(&self, i: u32) -> Option<&[T]> {
        let start = (i as usize).checked_mul(self.width)?;
        self.data.get(start..start.checked_add(self.width)?)
    }
}

/// How many numbers a `<param>` of `type` takes.
fn param_size(kind: Option<&str>) -> usize {
    match kind.unwrap_or("float") {
        "float2" | "double2" | "int2" => 2,
        "float3" | "double3" | "int3" => 3,
        "float4" | "double4" | "int4" | "float2x2" | "double2x2" => 4,
        "float3x3" | "double3x3" => 9,
        "float4x4" | "double4x4" => 16,
        _ => 1,
    }
}

/// The array a source's accessor reads (or the source's own array) and the accessor.
fn array_of(dom: &Dom<'_>, source: usize) -> (Option<usize>, Option<usize>) {
    let accessor = dom.path(source, &["technique_common", "accessor"]);
    let array = accessor
        .and_then(|a| dom.link(a, "source"))
        .filter(|&a| dom.parent(a) == Some(source) || dom.name(a).ends_with(b"_array"))
        .or_else(|| dom.children(source).find(|&c| dom.name(c).ends_with(b"_array")));
    (array, accessor)
}

/// Reads a numeric source through its accessor: unnamed params skipped, stride and offset honored. A source with no
/// accessor is its array, `fallback` numbers to an element.
pub(super) fn read<T: Copy>(dom: &Dom<'_>, source: usize, fallback: usize, parse: impl Fn(&[u8]) -> Vec<T>) -> Acc<T> {
    let (array, accessor) = array_of(dom, source);
    let values = array.map(|a| parse(dom.text(a))).unwrap_or_default();
    let Some(a) = accessor else {
        let width = fallback.max(1);
        let n = values.len() / width * width;
        let mut data = values;
        data.truncate(n);
        return Acc { width, data };
    };
    let stride = dom.attr_num::<usize>(a, "stride").unwrap_or(1).max(1);
    let offset = dom.attr_num::<usize>(a, "offset").unwrap_or(0);
    let mut picks = Vec::new();
    let mut at = 0;
    for p in dom.kids(a, "param") {
        let size = param_size(dom.attr(p, "type"));
        if dom.attr(p, "name").is_some_and(|n| !n.is_empty()) {
            picks.extend(at..at + size);
        }
        at += size;
    }
    // Unnamed params are skipped, unless none is named: then the accessor is read whole.
    if picks.is_empty() {
        picks.extend(0..if at == 0 { stride } else { at });
    }
    picks.retain(|&p| p < stride);
    let Some(&last) = picks.last() else {
        return Acc {
            width: 0,
            data: Vec::new(),
        };
    };
    let available = if values.len() > offset + last {
        (values.len() - offset - last - 1) / stride + 1
    } else {
        0
    };
    let count = dom
        .attr_num::<usize>(a, "count")
        .map_or(available, |c| c.min(available));
    let width = picks.len();
    let contiguous = offset == 0 && width == stride;
    let data = if contiguous {
        let mut data = values;
        data.truncate(count * stride);
        data
    } else {
        let mut data = Vec::with_capacity(count * width);
        for i in 0..count {
            let base = offset + i * stride;
            data.extend(picks.iter().map(|&p| values[base + p]));
        }
        data
    };
    Acc { width, data }
}

/// The words of a `Name_array` or `IDREF_array` source.
pub(super) fn names(dom: &Dom<'_>, source: usize) -> Vec<String> {
    let (array, _) = array_of(dom, source);
    array
        .map(|a| xml::words(dom.text(a)).map(str::to_owned).collect())
        .unwrap_or_default()
}

/// What an input feeds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sem {
    Position,
    Normal,
    Texcoord,
    Color,
    Tangent,
    Binormal,
}

impl Sem {
    fn of(name: &str) -> Option<Self> {
        Some(match name {
            "POSITION" => Sem::Position,
            "NORMAL" => Sem::Normal,
            "TEXCOORD" | "UV" => Sem::Texcoord,
            "COLOR" => Sem::Color,
            "TANGENT" | "TEXTANGENT" => Sem::Tangent,
            "BINORMAL" | "TEXBINORMAL" => Sem::Binormal,
            _ => return None,
        })
    }

    /// Numbers to an element when a source has no accessor.
    fn width(self) -> usize {
        match self {
            Sem::Texcoord => 2,
            Sem::Color => 4,
            _ => 3,
        }
    }
}

/// An input of a primitive, with its values.
struct Stream {
    sem: Sem,
    set: u32,
    offset: usize,
    acc: Rc<Acc<f32>>,
}

/// One primitive of a mesh, ready for the builder.
pub(super) struct Piece {
    pub prim: Prim,
    /// The material symbol it's drawn with.
    pub symbol: Option<String>,
    /// The `set` of each `TEXCOORD` input, in the order of the primitive's `TEXCOORD_n`.
    pub sets: Vec<u32>,
    /// Each vertex's index into the mesh's `<vertices>`, which skins and morph targets index.
    pub points: Vec<u32>,
    pub triangles: usize,
}

/// Source values by element, read once.
pub(super) type Sources = HashMap<usize, Rc<Acc<f32>>>;

pub(super) fn source(dom: &Dom<'_>, sources: &mut Sources, el: usize, fallback: usize) -> Rc<Acc<f32>> {
    sources
        .entry(el)
        .or_insert_with(|| Rc::new(read(dom, el, fallback, xml::floats32)))
        .clone()
}

/// The position source a mesh's `<vertices>` names.
pub(super) fn positions(dom: &Dom<'_>, sources: &mut Sources, mesh: usize) -> Option<Rc<Acc<f32>>> {
    let vertices = dom.child(mesh, "vertices")?;
    let input = dom
        .kids(vertices, "input")
        .find(|&i| dom.attr(i, "semantic") == Some("POSITION"))?;
    let el = dom.link(input, "source")?;
    Some(source(dom, sources, el, 3))
}

/// The primitives of a `<mesh>` (or `<convex_mesh>`), drawing at most `budget` triangles; `Err` with the
/// number drawn when they'd draw more.
pub(super) fn mesh(dom: &Dom<'_>, sources: &mut Sources, mesh: usize, budget: usize) -> Result<Vec<Piece>, usize> {
    let mut out = Vec::new();
    let mut drawn = 0usize;
    for el in dom.children(mesh) {
        let kind = match dom.name(el) {
            b"triangles" => Kind::Triangles,
            b"polylist" => Kind::Polylist,
            b"polygons" => Kind::Polygons,
            b"trifans" => Kind::Fans,
            b"tristrips" => Kind::Strips,
            b"lines" => Kind::Lines,
            b"linestrips" => Kind::Linestrips,
            _ => continue,
        };
        let piece = primitive(dom, sources, el, kind, budget - drawn).map_err(|n| drawn + n)?;
        if let Some(p) = piece {
            drawn += p.triangles;
            out.push(p);
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Triangles,
    Polylist,
    Polygons,
    Fans,
    Strips,
    Lines,
    Linestrips,
}

/// The inputs of a primitive: its streams, the offset of `VERTEX` (the points skins index) and the `<p>` stride.
fn inputs(dom: &Dom<'_>, sources: &mut Sources, el: usize) -> (Vec<Stream>, Option<usize>, usize) {
    let mut streams = Vec::new();
    let mut vertex_offset = None;
    let mut stride = 1;
    let mut add = |sem: Sem, set: u32, offset: usize, src: usize, streams: &mut Vec<Stream>| {
        let acc = source(dom, sources, src, sem.width());
        streams.push(Stream { sem, set, offset, acc });
    };
    for input in dom.kids(el, "input") {
        let Some(offset) = dom.attr_num::<usize>(input, "offset").or(Some(0)) else {
            continue;
        };
        // A wild offset would make every corner huge.
        if offset > 64 {
            continue;
        }
        stride = stride.max(offset + 1);
        let set = dom.attr_num::<u32>(input, "set").unwrap_or(0);
        let semantic = dom.attr(input, "semantic").unwrap_or("");
        // A mesh has one `<vertices>`, which is what `VERTEX` reads however its URL is written.
        let src = dom.link(input, "source").or_else(|| {
            (semantic == "VERTEX")
                .then(|| dom.parent(el).and_then(|m| dom.child(m, "vertices")))
                .flatten()
        });
        let Some(src) = src else {
            continue;
        };
        if semantic == "VERTEX" && dom.is(src, "source") {
            // The positions themselves.
            vertex_offset.get_or_insert(offset);
            add(Sem::Position, set, offset, src, &mut streams);
        } else if semantic == "VERTEX" {
            vertex_offset.get_or_insert(offset);
            for vi in dom.kids(src, "input") {
                let Some(sem) = dom.attr(vi, "semantic").and_then(Sem::of) else {
                    continue;
                };
                let Some(vs) = dom.link(vi, "source") else {
                    continue;
                };
                let set = dom.attr_num::<u32>(vi, "set").unwrap_or(set);
                add(sem, set, offset, vs, &mut streams);
            }
        } else if let Some(sem) = Sem::of(semantic) {
            if sem == Sem::Position {
                vertex_offset.get_or_insert(offset);
            }
            add(sem, set, offset, src, &mut streams);
        }
    }
    // One position stream, the first.
    let mut seen = false;
    streams.retain(|s| {
        if s.sem != Sem::Position {
            return true;
        }
        let keep = !seen;
        seen = true;
        keep
    });
    (streams, vertex_offset, stride)
}

fn primitive(
    dom: &Dom<'_>,
    sources: &mut Sources,
    el: usize,
    kind: Kind,
    budget: usize,
) -> Result<Option<Piece>, usize> {
    let (streams, vertex_offset, stride) = inputs(dom, sources, el);
    let Some(pos) = streams.iter().position(|s| s.sem == Sem::Position) else {
        return Ok(None);
    };
    let symbol = dom.attr(el, "material").map(str::to_owned);
    let mut c = Corners::new(streams, pos, vertex_offset, stride, budget);
    match kind {
        Kind::Triangles => {
            for p in dom.kids(el, "p") {
                let idx = xml::indices(dom.text(p));
                for tri in idx.chunks_exact(stride * 3) {
                    c.triangle(&tri[..stride], &tri[stride..2 * stride], &tri[2 * stride..]);
                    if c.over {
                        break;
                    }
                }
            }
        }
        Kind::Polylist => {
            let counts = dom
                .child(el, "vcount")
                .map(|v| xml::indices(dom.text(v)))
                .unwrap_or_default();
            if let Some(p) = dom.child(el, "p") {
                let idx = xml::indices(dom.text(p));
                let mut at = 0usize;
                for &n in &counts {
                    let n = n as usize;
                    let Some(end) = n.checked_mul(stride).and_then(|len| at.checked_add(len)) else {
                        break;
                    };
                    if end > idx.len() {
                        break;
                    }
                    c.polygon(&idx[at..end]);
                    at = end;
                    if c.over {
                        break;
                    }
                }
            }
        }
        Kind::Polygons => {
            for child in dom.children(el) {
                if c.over {
                    break;
                }
                if dom.is(child, "p") {
                    let idx = xml::indices(dom.text(child));
                    c.polygon(&idx[..idx.len() / stride * stride]);
                } else if dom.is(child, "ph") {
                    let Some(outer) = dom.child(child, "p") else {
                        continue;
                    };
                    let outer = xml::indices(dom.text(outer));
                    let holes: Vec<Vec<u32>> = dom.kids(child, "h").map(|h| xml::indices(dom.text(h))).collect();
                    c.polygon_with_holes(&outer, &holes);
                }
            }
        }
        Kind::Fans | Kind::Strips => {
            for p in dom.kids(el, "p") {
                let idx = xml::indices(dom.text(p));
                let corners: Vec<&[u32]> = idx.chunks_exact(stride).collect();
                for k in 2..corners.len() {
                    let (a, b, d) = if kind == Kind::Fans {
                        (corners[0], corners[k - 1], corners[k])
                    } else if k % 2 == 0 {
                        (corners[k - 2], corners[k - 1], corners[k])
                    } else {
                        (corners[k - 1], corners[k - 2], corners[k])
                    };
                    c.triangle(a, b, d);
                    if c.over {
                        break;
                    }
                }
            }
        }
        // Segments come from the file's own indices, so they cost no more than the file; the triangle budget
        // doesn't count them.
        Kind::Lines => {
            for p in dom.kids(el, "p") {
                let idx = xml::indices(dom.text(p));
                for seg in idx.chunks_exact(stride * 2) {
                    c.segment(&seg[..stride], &seg[stride..]);
                }
            }
        }
        Kind::Linestrips => {
            for p in dom.kids(el, "p") {
                let idx = xml::indices(dom.text(p));
                let corners: Vec<&[u32]> = idx.chunks_exact(stride).collect();
                for pair in corners.windows(2) {
                    c.segment(pair[0], pair[1]);
                }
            }
        }
    }
    if c.over {
        return Err(c.triangles);
    }
    Ok(c.finish(symbol))
}

/// The empty slot of the corner table.
const EMPTY: u32 = u32::MAX;

/// Builds a primitive's vertices from corners (a `<p>` tuple each), one vertex for each distinct combination of the
/// indices its inputs read.
struct Corners {
    streams: Vec<Stream>,
    pos: usize,
    stride: usize,
    vertex_offset: usize,
    /// The offsets some input reads: what tells corners apart.
    keyed: Vec<usize>,
    /// With one keyed offset, the vertex of each index.
    direct: Vec<u32>,
    /// Otherwise, an open-addressed table of vertex numbers, and each vertex's key.
    table: Vec<u32>,
    keys: Vec<u32>,
    key: Vec<u32>,
    budget: usize,
    triangles: usize,
    over: bool,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 3]>,
    binormals: Vec<[f32; 3]>,
    uvs: Vec<Vec<[f32; 2]>>,
    colors: Vec<Vec<[f32; 4]>>,
    points: Vec<u32>,
    indices: Vec<u32>,
    /// Line segments, a pair of vertices each.
    lines: Vec<u32>,
    /// Which output each stream fills: its set's number among the inputs of its kind.
    slot: Vec<usize>,
    sets: Vec<u32>,
    // Scratch.
    poly: Vec<u32>,
    outline: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
}

impl Corners {
    fn new(mut streams: Vec<Stream>, pos: usize, vertex_offset: Option<usize>, stride: usize, budget: usize) -> Self {
        // Texture coordinate and color sets in order of their set numbers.
        let mut order: Vec<usize> = (0..streams.len()).collect();
        order.sort_by_key(|&i| (streams[i].sem as u8, streams[i].set));
        let mut slot = vec![0; streams.len()];
        let mut sets = Vec::new();
        let (mut nuv, mut ncolor) = (0, 0);
        let (mut normal, mut tangent, mut binormal) = (false, false, false);
        let mut drop = vec![false; streams.len()];
        for &i in &order {
            match streams[i].sem {
                Sem::Texcoord => {
                    slot[i] = nuv;
                    nuv += 1;
                    sets.push(streams[i].set);
                }
                Sem::Color => {
                    slot[i] = ncolor;
                    ncolor += 1;
                }
                // One of each of the others.
                Sem::Normal => drop[i] = std::mem::replace(&mut normal, true),
                Sem::Tangent => drop[i] = std::mem::replace(&mut tangent, true),
                Sem::Binormal => drop[i] = std::mem::replace(&mut binormal, true),
                Sem::Position => {}
            }
        }
        let pos_offset = streams[pos].offset;
        let mut keyed: Vec<usize> = streams
            .iter()
            .zip(&drop)
            .filter(|(_, d)| !**d)
            .map(|(s, _)| s.offset)
            .collect();
        keyed.sort_unstable();
        keyed.dedup();
        let positions = streams[pos].acc.count();
        let direct = if keyed.len() == 1 {
            vec![EMPTY; positions]
        } else {
            Vec::new()
        };
        // Dropped streams read nothing.
        let mut k = 0;
        streams.retain(|_| {
            k += 1;
            !drop[k - 1]
        });
        let mut k = 0;
        slot.retain(|_| {
            k += 1;
            !drop[k - 1]
        });
        let pos = streams.iter().position(|s| s.sem == Sem::Position).unwrap_or(0);
        Self {
            streams,
            pos,
            stride,
            vertex_offset: vertex_offset.unwrap_or(pos_offset),
            keyed,
            direct,
            table: Vec::new(),
            keys: Vec::new(),
            key: Vec::new(),
            budget,
            triangles: 0,
            over: false,
            positions: Vec::new(),
            normals: Vec::new(),
            tangents: Vec::new(),
            binormals: Vec::new(),
            uvs: vec![Vec::new(); nuv],
            colors: vec![Vec::new(); ncolor],
            points: Vec::new(),
            indices: Vec::new(),
            lines: Vec::new(),
            slot,
            sets,
            poly: Vec::new(),
            outline: Vec::new(),
            tris: Vec::new(),
        }
    }

    /// The vertex of a corner, made when it's new; `None` when the corner names no position.
    #[inline]
    fn vertex(&mut self, corner: &[u32]) -> Option<u32> {
        let pi = *corner.get(self.streams[self.pos].offset)?;
        if pi as usize >= self.streams[self.pos].acc.count() {
            return None;
        }
        if self.keyed.len() == 1 {
            let at = &mut self.direct[pi as usize];
            if *at == EMPTY {
                *at = u32_of(self.positions.len());
                self.emit(corner);
            }
            return Some(self.direct[pi as usize]);
        }
        self.key.clear();
        for &o in &self.keyed {
            self.key.push(*corner.get(o).unwrap_or(&EMPTY));
        }
        if self.positions.len() * 2 >= self.table.len() {
            self.grow();
        }
        let mask = self.table.len() - 1;
        let w = self.keyed.len();
        let mut h = hash(&self.key) & mask;
        loop {
            let v = self.table[h];
            if v == EMPTY {
                let n = u32_of(self.positions.len());
                self.table[h] = n;
                self.keys.extend_from_slice(&self.key);
                self.emit(corner);
                return Some(n);
            }
            let start = v as usize * w;
            // Compared number by number: keys are short, and a call to compare memory costs more than they do.
            if self.keys[start..start + w].iter().zip(&self.key).all(|(a, b)| a == b) {
                return Some(v);
            }
            h = (h + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let size = (self.table.len() * 2).max(1024);
        self.table = vec![EMPTY; size];
        let mask = size - 1;
        let w = self.keyed.len();
        for v in 0..self.positions.len() {
            let mut h = hash(&self.keys[v * w..v * w + w]) & mask;
            while self.table[h] != EMPTY {
                h = (h + 1) & mask;
            }
            self.table[h] = u32_of(v);
        }
    }

    /// Appends a new vertex's attributes.
    fn emit(&mut self, corner: &[u32]) {
        self.points.push(corner.get(self.vertex_offset).copied().unwrap_or(0));
        for (s, &slot) in self.streams.iter().zip(&self.slot) {
            let v = corner.get(s.offset).and_then(|&i| s.acc.get(i)).unwrap_or(&[]);
            let at = |k: usize| v.get(k).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
            match s.sem {
                Sem::Position => self.positions.push([at(0), at(1), at(2)]),
                Sem::Normal => self.normals.push(unit([at(0), at(1), at(2)])),
                Sem::Tangent => self.tangents.push([at(0), at(1), at(2)]),
                Sem::Binormal => self.binormals.push([at(0), at(1), at(2)]),
                Sem::Texcoord => self.uvs[slot].push([at(0), 1.0 - at(1)]),
                Sem::Color => {
                    let a = if v.len() >= 4 { at(3) } else { 1.0 };
                    self.colors[slot].push([srgb_to_linear(at(0)), srgb_to_linear(at(1)), srgb_to_linear(at(2)), a]);
                }
            }
        }
    }

    #[inline]
    fn push(&mut self, a: u32, b: u32, c: u32) {
        if a == b || b == c || a == c {
            return;
        }
        self.triangles += 1;
        if self.triangles > self.budget {
            self.over = true;
            return;
        }
        self.indices.extend_from_slice(&[a, b, c]);
    }

    fn triangle(&mut self, a: &[u32], b: &[u32], c: &[u32]) {
        let (Some(a), Some(b), Some(c)) = (self.vertex(a), self.vertex(b), self.vertex(c)) else {
            return;
        };
        self.push(a, b, c);
    }

    /// A line segment; one that starts and ends at the same vertex draws nothing.
    fn segment(&mut self, a: &[u32], b: &[u32]) {
        let (Some(a), Some(b)) = (self.vertex(a), self.vertex(b)) else {
            return;
        };
        if a != b {
            self.lines.extend_from_slice(&[a, b]);
        }
    }

    /// The vertices of a polygon's corners; `false` when one names no position.
    fn outline(&mut self, idx: &[u32], out: &mut Vec<u32>) -> bool {
        out.clear();
        for corner in idx.chunks_exact(self.stride) {
            let Some(v) = self.vertex(corner) else {
                return false;
            };
            // A corner repeated in place adds nothing.
            if out.last() != Some(&v) {
                out.push(v);
            }
        }
        while out.len() > 1 && out.first() == out.last() {
            out.pop();
        }
        true
    }

    fn polygon(&mut self, idx: &[u32]) {
        let n = idx.len() / self.stride;
        if n == 3 {
            let s = self.stride;
            self.triangle(&idx[..s], &idx[s..2 * s], &idx[2 * s..3 * s]);
            return;
        }
        if n < 3 {
            return;
        }
        let mut poly = std::mem::take(&mut self.poly);
        if self.outline(idx, &mut poly) && poly.len() >= 3 {
            if poly.len() == 3 {
                self.push(poly[0], poly[1], poly[2]);
            } else {
                self.outline.clear();
                self.outline.extend(poly.iter().map(|&v| self.positions[v as usize]));
                self.tris.clear();
                triangulate(&self.outline, &mut self.tris);
                let tris = std::mem::take(&mut self.tris);
                for t in &tris {
                    self.push(poly[t[0] as usize], poly[t[1] as usize], poly[t[2] as usize]);
                }
                self.tris = tris;
            }
        }
        self.poly = poly;
    }

    fn polygon_with_holes(&mut self, outer: &[u32], holes: &[Vec<u32>]) {
        let stride = self.stride;
        let mut ring = Vec::new();
        if !self.outline(&outer[..outer.len() / stride * stride], &mut ring) || ring.len() < 3 {
            return;
        }
        let mut hole_rings = Vec::new();
        for h in holes {
            let mut r = Vec::new();
            if self.outline(&h[..h.len() / stride * stride], &mut r) && r.len() >= 3 {
                hole_rings.push(r);
            }
        }
        if hole_rings.is_empty() {
            let flat: Vec<u32> = outer[..outer.len() / stride * stride].to_vec();
            self.polygon(&flat);
            return;
        }
        let mut tris = Vec::new();
        bridge_and_clip(&ring, &hole_rings, &self.positions, &mut tris);
        for t in tris {
            self.push(t[0], t[1], t[2]);
        }
    }

    fn finish(self, symbol: Option<String>) -> Option<Piece> {
        if self.indices.is_empty() && self.lines.is_empty() {
            return None;
        }
        let n = self.positions.len();
        let tangents = if self.tangents.len() == n && self.normals.len() == n {
            self.tangents
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let w = match (self.binormals.get(i), self.normals.get(i)) {
                        (Some(b), Some(nm)) if self.binormals.len() == n => {
                            let c = cross(*nm, *t);
                            if c[0] * b[0] + c[1] * b[1] + c[2] * b[2] < 0.0 {
                                -1.0
                            } else {
                                1.0
                            }
                        }
                        _ => 1.0,
                    };
                    [t[0], t[1], t[2], w]
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut uvs = self.uvs.into_iter();
        let first_uv = uvs.next().unwrap_or_default();
        let rest: Vec<Vec<[f32; 2]>> = uvs.collect();
        let uv_sets = if rest.is_empty() {
            Vec::new()
        } else {
            std::iter::once(Vec::new()).chain(rest).collect()
        };
        let mut colors = self.colors.into_iter();
        let first_color = colors.next().unwrap_or_default();
        let rest: Vec<Vec<[f32; 4]>> = colors.collect();
        let color_sets = if rest.is_empty() {
            Vec::new()
        } else {
            std::iter::once(Vec::new()).chain(rest).collect()
        };
        let triangles = self.indices.len() / 3;
        Some(Piece {
            prim: Prim {
                positions: self.positions,
                normals: self.normals,
                tangents,
                uvs: first_uv,
                uv_sets,
                colors: first_color,
                color_sets,
                indices: self.indices,
                lines: self.lines,
                ..Prim::default()
            },
            symbol,
            sets: self.sets,
            points: self.points,
            triangles,
        })
    }
}

#[inline]
fn hash(key: &[u32]) -> usize {
    let mut h: u64 = 0;
    for &k in key {
        h = (h.rotate_left(5) ^ u64::from(k)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    // The high bits mix best.
    usize::try_from(h >> 32).unwrap_or(0) ^ usize::try_from(h & 0xFFFF_FFFF).unwrap_or(0).rotate_left(16)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-20 && len.is_finite() {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Polygons with holes past this many corners are fanned rather than clipped.
const MAX_CLIP: usize = 4096;

/// Triangulates a polygon with holes: each hole joined to the outline by a bridge to a vertex it sees (holes with
/// the furthest-out vertex first), then the one outline ear-clipped, the bridges' doubled points ignored as
/// blockers. Triangles keep the outline's winding.
pub(super) fn bridge_and_clip(outer: &[u32], holes: &[Vec<u32>], pos: &[[f32; 3]], out: &mut Vec<[u32; 3]>) {
    // The plane of the outline (Newell's normal), and a 2D view of it with the outline counter-clockwise.
    let mut nrm = [0f64; 3];
    for i in 0..outer.len() {
        let a = pos[outer[i] as usize];
        let b = pos[outer[(i + 1) % outer.len()] as usize];
        nrm[0] += f64::from((a[1] - b[1]) * (a[2] + b[2]));
        nrm[1] += f64::from((a[2] - b[2]) * (a[0] + b[0]));
        nrm[2] += f64::from((a[0] - b[0]) * (a[1] + b[1]));
    }
    let axis = (0..3).fold(0, |m, k| if nrm[k].abs() > nrm[m].abs() { k } else { m });
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let flip = nrm[axis] < 0.0;
    let p2 = |i: u32| -> [f64; 2] {
        let p = pos[i as usize];
        let x = f64::from(p[u]);
        [if flip { -x } else { x }, f64::from(p[v])]
    };
    let area = |ring: &[u32]| -> f64 {
        (0..ring.len())
            .map(|i| {
                let a = p2(ring[i]);
                let b = p2(ring[(i + 1) % ring.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum()
    };
    let total = outer.len() + holes.iter().map(|h| h.len() + 2).sum::<usize>();
    if total > MAX_CLIP {
        for k in 1..outer.len() - 1 {
            out.push([outer[0], outer[k], outer[k + 1]]);
        }
        return;
    }
    let mut holes: Vec<Vec<u32>> = holes
        .iter()
        .map(|h| {
            let mut h = h.clone();
            if area(&h) > 0.0 {
                h.reverse();
            }
            h
        })
        .collect();
    let max_x = |h: &Vec<u32>| h.iter().map(|&i| p2(i)[0]).fold(f64::MIN, f64::max);
    holes.sort_by(|a, b| max_x(b).total_cmp(&max_x(a)));
    let mut poly: Vec<u32> = outer.to_vec();
    for (hi, hole) in holes.iter().enumerate() {
        let m = (0..hole.len()).fold(0, |b, k| if p2(hole[k])[0] > p2(hole[b])[0] { k } else { b });
        let mp = p2(hole[m]);
        let mut candidates: Vec<usize> = (0..poly.len()).collect();
        let dist = |i: usize| {
            let q = p2(poly[i]);
            (q[0] - mp[0]).powi(2) + (q[1] - mp[1]).powi(2)
        };
        candidates.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
        let blocked = |a: [f64; 2], b: [f64; 2]| -> bool {
            let rings = std::iter::once(&poly).chain(holes[hi..].iter());
            for ring in rings {
                for k in 0..ring.len() {
                    let c = p2(ring[k]);
                    let d = p2(ring[(k + 1) % ring.len()]);
                    if crosses(a, b, c, d) {
                        return true;
                    }
                }
            }
            false
        };
        let pick = candidates
            .iter()
            .take(64)
            .copied()
            .find(|&i| !blocked(mp, p2(poly[i])))
            .unwrap_or(candidates[0]);
        let mut merged = Vec::with_capacity(poly.len() + hole.len() + 2);
        merged.extend_from_slice(&poly[..=pick]);
        merged.extend_from_slice(&hole[m..]);
        merged.extend_from_slice(&hole[..=m]);
        merged.extend_from_slice(&poly[pick..]);
        poly = merged;
    }
    ear_clip(&poly, &p2, out);
}

/// Whether segments `ab` and `cd` cross at a point inside both (touching at an end doesn't count).
#[allow(clippy::float_cmp, reason = "the same point, exactly")]
fn crosses(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let orient = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    if a == c || a == d || b == c || b == d {
        return false;
    }
    let (d1, d2) = (orient(a, b, c), orient(a, b, d));
    let (d3, d4) = (orient(c, d, a), orient(c, d, b));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Ear-clips a simple outline (counter-clockwise in `p2`'s view) that may touch itself at doubled points.
#[allow(clippy::float_cmp, reason = "the same point, exactly")]
fn ear_clip(poly: &[u32], p2: &dyn Fn(u32) -> [f64; 2], out: &mut Vec<[u32; 3]>) {
    let pts: Vec<[f64; 2]> = poly.iter().map(|&i| p2(i)).collect();
    let mut idx: Vec<usize> = (0..poly.len()).collect();
    let orient = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    while idx.len() > 3 {
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (ia, ib, ic) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (pts[ia], pts[ib], pts[ic]);
            if orient(a, b, c) <= 0.0 {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                let p = pts[j];
                if j == ia || j == ib || j == ic || p == a || p == b || p == c {
                    return false;
                }
                orient(a, b, p) >= 0.0 && orient(b, c, p) >= 0.0 && orient(c, a, p) >= 0.0
            });
            if inside {
                continue;
            }
            out.push([poly[ia], poly[ib], poly[ic]]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            // Nothing is an ear (a degenerate or self-crossing outline): drop a corner that adds no area, or
            // clip one anyway.
            let k = (0..m)
                .find(|&k| orient(pts[idx[(k + m - 1) % m]], pts[idx[k]], pts[idx[(k + 1) % m]]).abs() < 1e-30)
                .unwrap_or(0);
            if k != 0 || orient(pts[idx[m - 1]], pts[idx[0]], pts[idx[1]]).abs() >= 1e-30 {
                out.push([poly[idx[(k + m - 1) % m]], poly[idx[k]], poly[idx[(k + 1) % m]]]);
            }
            idx.remove(k);
        }
    }
    if idx.len() == 3 {
        out.push([poly[idx[0]], poly[idx[1]], poly[idx[2]]]);
    }
}
