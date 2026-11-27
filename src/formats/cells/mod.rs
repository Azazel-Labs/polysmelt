//! Meshes of cells, as VTK, Gmsh and Medit files hold them: points, the cells that name them, and data on
//! either, drawn as triangles, lines and points.
//!
//! A reader fills a [`Model`] with [`Piece`]s and [`build`] makes the document. Cells go by VTK's type numbers
//! ([`shapes`]): vertex cells draw as points, line cells as lines, surface cells as triangles, and volume cells
//! as the faces only one of them has. Points no drawn cell uses are left out. Point data follows its points and
//! cell data splits a vertex where the cells around it differ; data at the cells' nodes gives each cell its own
//! vertices. Normals, texture coordinates and colors become `NORMAL`, `TEXCOORD_0` and `COLOR_0`, and every
//! other array a float custom attribute. Pieces with a name are a mesh and a node each; the rest are one mesh.

#[cfg(any(feature = "msh", feature = "medit"))]
mod groups;
mod shapes;

use serde_json::{Map, Value, json};

use super::stream;
use crate::Result;
use crate::document::{Builder, Document, Part};
use crate::mesh::{FastMap, Prim, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::source::SourceInfo;
#[cfg(any(feature = "msh", feature = "medit"))]
pub(super) use groups::{Elements, Group};
use shapes::{Draw, Faces, Shape};

/// No point: an index that's negative or out of range, or an unused slot.
pub(super) const BAD: u32 = u32::MAX;

/// The integer `b` holds, four bytes in either order.
#[cfg(any(feature = "msh", feature = "medit"))]
#[inline]
pub(super) fn int_of(b: &[u8], big: bool) -> i32 {
    let b = [b[0], b[1], b[2], b[3]];
    if big {
        i32::from_be_bytes(b)
    } else {
        i32::from_le_bytes(b)
    }
}

/// The double `b` holds, eight bytes in either order, as a float.
#[cfg(any(feature = "msh", feature = "medit"))]
#[inline]
pub(super) fn real_of(b: &[u8], big: bool) -> f32 {
    let b = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    f32_of(if big {
        f64::from_be_bytes(b)
    } else {
        f64::from_le_bytes(b)
    })
}

/// The unsigned integer `b` holds, four or eight bytes in either order.
#[cfg(any(feature = "msh", feature = "medit"))]
#[inline]
pub(super) fn word_of(b: &[u8], big: bool) -> u64 {
    if b.len() == 4 {
        let b = [b[0], b[1], b[2], b[3]];
        u64::from(if big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    } else {
        let b = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        if big {
            u64::from_be_bytes(b)
        } else {
            u64::from_le_bytes(b)
        }
    }
}

/// A tag as an index's width: [`BAD`] for one that's negative or too big.
#[cfg(any(feature = "msh", feature = "medit"))]
#[inline]
pub(super) fn tag_of<T: TryInto<u32>>(v: T) -> u32 {
    v.try_into().unwrap_or(BAD)
}

/// The most components an array has to be kept as custom attributes: a full tensor's nine.
const MAX_COMPONENTS: usize = 9;

/// The most triangles an image or rectilinear grid draws when the file holds no data for its points or cells:
/// such a file is a few lines however many points it counts, so this bounds what a few lines can ask for.
const MAX_BARE_TRIANGLES: usize = 1 << 22;

/// A place in a file's text: the legacy keywords and numbers, and the XML arrays written as text.
pub(super) struct Cursor<'a> {
    pub d: &'a [u8],
    pub at: usize,
    /// The format's name, for errors.
    format: &'static str,
}

impl<'a> Cursor<'a> {
    pub fn new(d: &'a [u8], format: &'static str) -> Self {
        Self { d, at: 0, format }
    }

    #[inline]
    pub fn skip_space(&mut self) {
        while self.at < self.d.len() && stream::space(self.d[self.at]) {
            self.at += 1;
        }
    }

    /// The next token, across line ends.
    #[inline]
    pub fn token(&mut self) -> Option<&'a [u8]> {
        self.skip_space();
        let start = self.at;
        while self.at < self.d.len() && !stream::space(self.d[self.at]) {
            self.at += 1;
        }
        (self.at > start).then(|| &self.d[start..self.at])
    }

    /// The next line that isn't blank, and the cursor past its end: binary data starts right there.
    pub fn line(&mut self) -> Option<&'a [u8]> {
        loop {
            let line = stream::trim(stream::next_line(self.d, &mut self.at)?);
            if !line.is_empty() {
                return Some(line);
            }
        }
    }

    /// How many values the text left can hold: each takes a character and a separator.
    pub fn room(&self) -> usize {
        (self.d.len() - self.at.min(self.d.len())).div_ceil(2)
    }

    /// `n` numbers as floats.
    pub fn floats(&mut self, n: usize, what: &str) -> Result<Vec<f32>> {
        ensure!(
            n <= self.room(),
            Truncated,
            "{} file is cut short: {what} claims {n} values, more than it holds",
            self.format
        );
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            self.skip_space();
            if let Some((v, len)) = stream::f32_lead(&self.d[self.at..]) {
                out.push(v);
                self.at += len;
                continue;
            }
            // `nan`, `inf` and the like.
            let Some(t) = self.token() else {
                bail!(Truncated, "{} file is cut short: {what} ends early", self.format);
            };
            let Some(v) = stream::f32_lenient(t) else {
                bail!(Corrupt, "{} {what}: {} is not a number", self.format, crate::shown(t));
            };
            out.push(v);
        }
        Ok(out)
    }

    /// The next number as a float; `None` at the end.
    #[cfg(any(feature = "msh", feature = "medit"))]
    #[inline]
    pub fn float(&mut self, what: &str) -> Result<Option<f32>> {
        self.skip_space();
        if let Some((v, len)) = stream::f32_lead(&self.d[self.at..]) {
            self.at += len;
            return Ok(Some(v));
        }
        let Some(t) = self.token() else {
            return Ok(None);
        };
        // `nan`, `inf`, and Fortran's `1.5D+00`.
        let v = stream::f32_lenient(t).or_else(|| {
            let e: Vec<u8> = t
                .iter()
                .map(|&b| if b == b'D' || b == b'd' { b'e' } else { b })
                .collect();
            stream::f32_from(&e)
        });
        let Some(v) = v else {
            bail!(Corrupt, "{} {what}: {} is not a number", self.format, crate::shown(t));
        };
        Ok(Some(v))
    }

    /// Moves past the end of the line the cursor is in.
    #[cfg(any(feature = "msh", feature = "medit"))]
    pub fn skip_line(&mut self) {
        let rest = &self.d[self.at.min(self.d.len())..];
        self.at += rest.iter().position(|&b| b == b'\n').map_or(rest.len(), |n| n + 1);
    }

    /// `n` whole numbers as indices: [`BAD`] for a negative one or one too big.
    pub fn ids(&mut self, n: usize, what: &str) -> Result<Vec<u32>> {
        ensure!(
            n <= self.room(),
            Truncated,
            "{} file is cut short: {what} claims {n} values, more than it holds",
            self.format
        );
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(v) = self.id(what)? else {
                bail!(Truncated, "{} file is cut short: {what} ends early", self.format);
            };
            out.push(v);
        }
        Ok(out)
    }

    /// The next whole number as an index, [`BAD`] for a negative one or one too big; `None` at the end.
    #[inline]
    pub fn id(&mut self, what: &str) -> Result<Option<u32>> {
        self.skip_space();
        let d = self.d;
        let start = self.at;
        let mut v = 0u64;
        while self.at < d.len() && self.at - start < 10 && d[self.at].is_ascii_digit() {
            v = v * 10 + u64::from(d[self.at] - b'0');
            self.at += 1;
        }
        if self.at > start && d.get(self.at).is_none_or(|&b| stream::space(b)) {
            return Ok(Some(u32::try_from(v).unwrap_or(BAD)));
        }
        // A sign, or more digits than an index has.
        self.at = start;
        let Some(t) = self.token() else {
            return Ok(None);
        };
        let Some(v) = stream::int::<i64>(t) else {
            bail!(
                Corrupt,
                "{} {what}: {} is not a whole number",
                self.format,
                crate::shown(t)
            );
        };
        Ok(Some(u32::try_from(v).unwrap_or(BAD)))
    }

    /// Every number left, as floats, expecting about `expected` of them.
    pub fn all_floats(&mut self, expected: usize, what: &str) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(expected.min(self.room()));
        loop {
            self.skip_space();
            if self.at >= self.d.len() {
                return Ok(out);
            }
            if let Some((v, len)) = stream::f32_lead(&self.d[self.at..]) {
                out.push(v);
                self.at += len;
                continue;
            }
            let t = self.token().unwrap_or_default();
            let Some(v) = stream::f32_lenient(t) else {
                bail!(Corrupt, "{} {what}: {} is not a number", self.format, crate::shown(t));
            };
            out.push(v);
        }
    }

    /// Every whole number left, as indices, expecting about `expected` of them.
    pub fn all_ids(&mut self, expected: usize, what: &str) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(expected.min(self.room()));
        while let Some(v) = self.id(what)? {
            out.push(v);
        }
        Ok(out)
    }
}

/// Where a piece's points are.
pub(super) enum Points {
    /// Each one given.
    List(Vec<[f32; 3]>),
    /// An image's: evenly spaced from an origin.
    Image {
        dims: [u32; 3],
        origin: [f64; 3],
        spacing: [f64; 3],
    },
    /// A rectilinear grid's: the coordinates along each axis.
    Rect { dims: [u32; 3], axes: [Vec<f32>; 3] },
}

impl Default for Points {
    fn default() -> Self {
        Points::List(Vec::new())
    }
}

impl Points {
    pub fn len(&self) -> usize {
        match self {
            Points::List(p) => p.len(),
            Points::Image { dims, .. } | Points::Rect { dims, .. } => {
                crate::num::usize_of(dims.iter().map(|&n| u64::from(n)).product())
            }
        }
    }

    /// Point `i`, which must be one of them.
    #[inline]
    fn at(&self, i: u32) -> [f32; 3] {
        let split = |dims: &[u32; 3]| [i % dims[0], i / dims[0] % dims[1], i / dims[0] / dims[1]];
        match self {
            Points::List(p) => p[i as usize],
            Points::Image { dims, origin, spacing } => {
                let at = split(dims);
                [0, 1, 2].map(|k| f32_of(origin[k] + f64::from(at[k]) * spacing[k]))
            }
            Points::Rect { dims, axes } => {
                let at = split(dims);
                [0, 1, 2].map(|k| axes[k][at[k] as usize])
            }
        }
    }
}

/// What an array is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    Normals,
    TCoords,
    Colors,
    Other,
}

/// A named array of point or cell data.
pub(super) struct Array {
    pub name: String,
    /// Values per point or cell.
    pub comps: usize,
    pub role: Role,
    pub values: Vec<f32>,
    /// What brings a color's values to 0-1: 1/255 for bytes.
    pub scale: f32,
    /// The lookup table its scalars name.
    pub table: Option<String>,
}

impl Array {
    pub fn new(name: &str, comps: usize, role: Role, values: Vec<f32>) -> Self {
        Self {
            name: name.to_owned(),
            comps,
            role,
            values,
            scale: 1.0,
            table: None,
        }
    }

    /// An array of unsigned bytes with three or four components is a color, as VTK draws it: called for such an
    /// array.
    pub fn bytes_are_colors(&mut self) {
        if matches!(self.comps, 3 | 4) && self.role == Role::Other {
            self.role = Role::Colors;
            self.scale = 1.0 / 255.0;
        }
    }
}

/// Which cells a list holds. Poly data numbers its cells in this order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Kind {
    Verts,
    Lines,
    Polys,
    Strips,
    /// An unstructured grid's, each with its own type.
    Mixed,
}

/// A list of cells: cell `i`'s nodes are `conn[offsets[i]..offsets[i + 1]]`.
pub(super) struct Cells {
    pub kind: Kind,
    pub conn: Vec<u32>,
    pub offsets: Vec<u32>,
    /// Each cell's type, for [`Kind::Mixed`].
    pub types: Vec<u8>,
}

impl Cells {
    pub fn len(&self) -> usize {
        let n = self.offsets.len().saturating_sub(1);
        if self.kind == Kind::Mixed {
            n.min(self.types.len())
        } else {
            n
        }
    }

    /// Cell `i`'s type and nodes; `None` for offsets that don't hold together.
    #[inline]
    fn cell(&self, i: usize) -> (u8, Option<&[u32]>) {
        let ty = match self.kind {
            Kind::Verts => 2,
            Kind::Lines => 4,
            Kind::Polys => 7,
            Kind::Strips => 6,
            Kind::Mixed => self.types[i],
        };
        (
            ty,
            self.conn.get(self.offsets[i] as usize..self.offsets[i + 1] as usize),
        )
    }

    /// Cells from the counted lists older legacy files hold: each cell its node count and then its nodes.
    pub fn from_counted(kind: Kind, cells: usize, list: &[u32]) -> Result<Self> {
        let mut conn = Vec::with_capacity(list.len().saturating_sub(cells));
        let mut offsets = Vec::with_capacity(cells.min(list.len()) + 1);
        offsets.push(0);
        let mut at = 0;
        for _ in 0..cells {
            let nodes = list
                .get(at)
                .and_then(|&n| list.get(at + 1..(at + 1).checked_add(n as usize)?));
            let Some(nodes) = nodes else {
                bail!(Corrupt, "cells run past their list");
            };
            conn.extend_from_slice(nodes);
            offsets.push(u32_of(conn.len()));
            at += 1 + nodes.len();
        }
        Ok(Self {
            kind,
            conn,
            offsets,
            types: Vec::new(),
        })
    }
}

/// One piece of a dataset: its points, its cells and their data.
#[derive(Default)]
pub(super) struct Piece {
    pub points: Points,
    /// A structured dataset's points along each axis; its cells are implied.
    pub dims: Option<[u32; 3]>,
    pub cells: Vec<Cells>,
    pub point_data: Vec<Array>,
    pub cell_data: Vec<Array>,
    /// Data at each node of each cell, as the one list of cells lists them: `conn` gives each value's point.
    pub corner_data: Vec<Array>,
    /// The piece's name, when it's a mesh of its own: a group of a file's cells. Pieces with no name are one
    /// mesh together.
    pub name: Option<String>,
    /// What the mesh's `extras` hold.
    pub extras: Option<Value>,
}

/// A file as read.
#[derive(Default)]
pub(super) struct Model {
    pub pieces: Vec<Piece>,
    /// Lookup tables by name, as written: sRGB and alpha, 0-1.
    pub tables: Vec<(String, Vec<[f32; 4]>)>,
    pub info: SourceInfo,
    /// The format's own notes for the asset's `extras.source`.
    pub source: Map<String, Value>,
}

/// One piece as drawn: a primitive's arrays, and the custom attributes beside them.
struct Drawn {
    prim: Prim,
    /// Each attribute's name, components and values.
    custom: Vec<(String, usize, Vec<f32>)>,
}

/// What the pieces add up to, for the facts and the source notes.
struct Totals {
    triangles: usize,
    lines: usize,
    points: usize,
    dropped: usize,
    counts: [usize; 256],
    skipped_arrays: Vec<String>,
}

/// The document `model` draws as. `format` is the format's name, for errors and the source notes.
pub(super) fn build(model: Model, format: &'static str, limit: usize) -> Result<Document> {
    let Model {
        pieces,
        tables,
        mut info,
        mut source,
    } = model;
    let mut totals = Totals {
        triangles: 0,
        lines: 0,
        points: 0,
        dropped: 0,
        counts: [0; 256],
        skipped_arrays: Vec::new(),
    };
    let tables: Vec<(String, Vec<[f32; 4]>)> = tables
        .into_iter()
        .map(|(name, colors)| (name, colors.iter().map(|c| linear(c, 1.0)).collect()))
        .collect();
    let mut b = Builder::default();
    // Each mesh's name, extras and primitives.
    let mut meshes: Vec<(Option<String>, Option<Value>, Vec<Value>)> = Vec::new();
    let mut colored = false;
    let mut any_points = false;
    let count = pieces.len();
    for mut piece in pieces {
        any_points |= piece.points.len() > 0;
        let (name, extras) = (piece.name.take(), piece.extras.take());
        let Some(drawn) = draw(piece, &tables, format, limit, &mut totals)? else {
            continue;
        };
        let Drawn { prim, custom } = drawn;
        let n = prim.positions.len();
        colored |= prim.colors.len() == n;
        let mut prims = b.primitives(prim);
        for (name, comps, values) in custom {
            let kind = match comps {
                1 => "SCALAR",
                2 => "VEC2",
                3 => "VEC3",
                _ => "VEC4",
            };
            let a = b.accessor(Part::f32(values), 5126, n, kind, Some(34962));
            for p in &mut prims {
                p["attributes"][&name] = json!(a);
            }
        }
        // Pieces with no name are one mesh.
        let shared = meshes.iter().position(|(n, ..)| n.is_none() && name.is_none());
        let at = shared.unwrap_or_else(|| {
            meshes.push((name, extras, Vec::new()));
            meshes.len() - 1
        });
        for mut p in prims {
            p["material"] = json!(0);
            meshes[at].2.push(p);
        }
    }
    let skipped: Map<String, Value> = (0..=255u8)
        .filter(|&ty| totals.counts[usize::from(ty)] > 0 && matches!(shapes::shape(ty), Shape::Unknown))
        .map(|ty| (shapes::name(ty), json!(totals.counts[usize::from(ty)])))
        .collect();
    ensure!(any_points, Empty, "{format} file has no points");
    if meshes.is_empty() {
        let names: Vec<&String> = skipped.keys().collect();
        ensure!(
            names.is_empty(),
            Unsupported,
            "{format} file has only cells that aren't drawn: {names:?}"
        );
        bail!(Empty, "{format} file has nothing to draw");
    }
    let kinds: Map<String, Value> = (0..=255u8)
        .filter(|&ty| totals.counts[usize::from(ty)] > 0)
        .map(|ty| (shapes::name(ty), json!(totals.counts[usize::from(ty)])))
        .collect();
    if !kinds.is_empty() {
        source.insert("cells".into(), Value::Object(kinds));
    }
    if !skipped.is_empty() {
        source.insert("skippedCells".into(), Value::Object(skipped));
    }
    if totals.dropped > 0 {
        source.insert("droppedCells".into(), json!(totals.dropped));
    }
    if !totals.skipped_arrays.is_empty() {
        source.insert("skippedArrays".into(), json!(totals.skipped_arrays));
    }
    if count > 1 {
        source.insert("pieces".into(), json!(count));
    }
    info.format = Some(format.into());
    info.write(&mut source);
    let mut material = crate::legacy_material::default_material(colored);
    material.double_sided = true;
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(totals.triangles));
    for (key, n) in [("lines", totals.lines), ("points", totals.points)] {
        if n > 0 {
            facts.insert(key.into(), json!(n));
        }
    }
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    b.materials([&material], Vec::new(), &mut json);
    json.insert("scene".into(), json!(0));
    let nodes: Vec<usize> = (0..meshes.len()).collect();
    json.insert("scenes".into(), json!([{ "nodes": nodes }]));
    let nodes: Vec<Value> = meshes
        .iter()
        .enumerate()
        .map(|(i, (name, ..))| match name {
            Some(name) => json!({ "mesh": i, "name": name }),
            None => json!({ "mesh": i }),
        })
        .collect();
    json.insert("nodes".into(), Value::Array(nodes));
    let meshes: Vec<Value> = meshes
        .into_iter()
        .map(|(name, extras, primitives)| {
            let mut mesh = json!({ "primitives": primitives });
            if let Some(name) = name {
                mesh["name"] = json!(name);
            }
            if let Some(extras) = extras {
                mesh["extras"] = extras;
            }
            mesh
        })
        .collect();
    json.insert("meshes".into(), Value::Array(meshes));
    Ok(b.finish(json, facts))
}

/// Draws one piece; `None` when it draws nothing.
fn draw(
    mut piece: Piece,
    tables: &[(String, Vec<[f32; 4]>)],
    format: &str,
    limit: usize,
    totals: &mut Totals,
) -> Result<Option<Drawn>> {
    let np = piece.points.len();
    if np == 0 {
        return Ok(None);
    }
    ensure!(np < BAD as usize, LimitExceeded, "{format} file has too many points");
    piece.cells.sort_by_key(|c| c.kind);
    let cell_count = match piece.dims {
        Some(dims) => dims.iter().map(|&n| (n as usize - 1).max(1)).product(),
        None => piece.cells.iter().map(Cells::len).sum::<usize>(),
    };
    ensure!(
        cell_count < BAD as usize,
        LimitExceeded,
        "{format} file has too many cells"
    );
    let fits = |a: &Array, n: usize| a.comps > 0 && n.checked_mul(a.comps) == Some(a.values.len());
    piece.point_data.retain(|a| fits(a, np));
    // With no cells there's nothing for cell data to be on, whatever an empty array claims.
    piece.cell_data.retain(|a| cell_count > 0 && fits(a, cell_count));
    // Data at the cells' nodes goes by the one list that has them.
    let slots = match (&piece.cells[..], piece.dims) {
        ([cells], None) => cells.conn.len(),
        _ => 0,
    };
    piece.corner_data.retain(|a| slots > 0 && fits(a, slots));
    let cornered = !piece.corner_data.is_empty();
    // Points that aren't finite take the cells at them along.
    let broken: Vec<bool> = match &piece.points {
        Points::List(p) if p.iter().any(|p| !p.iter().all(|c| c.is_finite())) => {
            p.iter().map(|p| !p.iter().all(|c| c.is_finite())).collect()
        }
        _ => Vec::new(),
    };
    let fine = |ids: &[u32]| {
        ids.iter()
            .all(|&p| (p as usize) < np && broken.get(p as usize) != Some(&true))
    };
    let over = |tris: &[u32]| tris.len() / 3 > limit.saturating_sub(totals.triangles);
    let mut d = Draw::new(!piece.cell_data.is_empty() || cornered);
    if let Some(dims) = piece.dims {
        // A structured grid's cells are hexahedra; an image's and a rectilinear grid's are voxels.
        let hexahedra = matches!(piece.points, Points::List(_));
        // Points that are only counted, with no data in the file either, are held to fewer triangles.
        let bare = !hexahedra && piece.point_data.is_empty() && piece.cell_data.is_empty();
        let most = limit.saturating_sub(totals.triangles);
        structured(
            &mut d,
            dims,
            if bare { most.min(MAX_BARE_TRIANGLES) } else { most },
            hexahedra,
        )?;
    } else {
        let mut first = 0usize;
        for cells in &piece.cells {
            let mut faces = Faces::default();
            for i in 0..cells.len() {
                let (ty, ids) = cells.cell(i);
                let Some(ids) = ids.filter(|ids| fine(ids)) else {
                    d.dropped += 1;
                    continue;
                };
                d.cell(ty, ids, u32_of(first + i), &piece.points, &mut faces);
                ensure!(
                    !over(&d.tris),
                    LimitExceeded,
                    "too many triangles to convert (more than {limit})"
                );
            }
            if !faces.is_empty() {
                faces.boundary(
                    |c| {
                        let (ty, ids) = cells.cell(c as usize - first);
                        (ty, ids.unwrap_or_default())
                    },
                    |ids, cell| d.fan(ids, cell),
                );
                ensure!(
                    !over(&d.tris),
                    LimitExceeded,
                    "too many triangles to convert (more than {limit})"
                );
            }
            first += cells.len();
        }
        if cell_count == 0 {
            d.verts = (0..u32_of(np)).filter(|&p| fine(&[p])).collect();
        }
    }
    for (total, n) in totals.counts.iter_mut().zip(d.counts) {
        *total += n;
    }
    totals.dropped += d.dropped;
    if d.tris.is_empty() && d.lines.is_empty() && d.verts.is_empty() {
        return Ok(None);
    }
    totals.triangles += d.tris.len() / 3;
    totals.lines += d.lines.len() / 2;
    totals.points += d.verts.len();
    let classes = d.track.then(|| {
        if cornered {
            // Each cell has its own values at a point, so its own vertices.
            (0..u32_of(cell_count)).collect()
        } else {
            classes(&piece.cell_data, cell_count, &d)
        }
    });
    let dense = matches!(piece.points, Points::List(_));
    let (src_points, src_cells) = weld(&mut d, np, classes.as_deref(), dense);
    drop(classes);
    let positions: Vec<[f32; 3]> = match (&src_points, std::mem::take(&mut piece.points)) {
        (None, Points::List(p)) => p,
        (None, points) => (0..u32_of(np)).map(|i| points.at(i)).collect(),
        (Some(src), points) => src.iter().map(|&i| points.at(i)).collect(),
    };
    let n = positions.len();
    let mut prim = Prim {
        positions,
        indices: std::mem::take(&mut d.tris),
        lines: std::mem::take(&mut d.lines),
        points: std::mem::take(&mut d.verts),
        ..Prim::default()
    };
    drop(d);
    let mut custom: Vec<(String, usize, Vec<f32>)> = Vec::new();
    let mut table_colors = None;
    // Where each vertex's cell lists its point: the place of its values in data at the cells' nodes.
    let src_slots: Vec<u32> = match (&piece.cells[..], &src_points) {
        ([cells], Some(points)) if cornered => points
            .iter()
            .zip(&src_cells)
            .map(|(&p, &c)| {
                let first = cells.offsets[c as usize];
                let nodes = cells.cell(c as usize).1.unwrap_or_default();
                first + u32_of(nodes.iter().position(|&n| n == p).unwrap_or(0))
            })
            .collect(),
        _ => Vec::new(),
    };
    let point_data = std::mem::take(&mut piece.point_data).into_iter();
    let cell_data = std::mem::take(&mut piece.cell_data).into_iter();
    let corner_data = std::mem::take(&mut piece.corner_data).into_iter();
    let all = point_data
        .map(|a| (a, src_points.as_ref(), false))
        .chain(cell_data.map(|a| (a, Some(&src_cells), true)))
        .chain(corner_data.map(|a| (a, Some(&src_slots), true)));
    for (mut a, src, per_cell) in all {
        // The scalars' range is the whole array's, drawn or not.
        let table = a
            .table
            .as_ref()
            .filter(|_| a.comps == 1 && table_colors.is_none())
            .and_then(|name| tables.iter().find(|(t, colors)| t == name && !colors.is_empty()))
            .map(|(_, colors)| (colors, range(&a.values)));
        let comps = a.comps;
        let values: Vec<f32> = match src {
            None => std::mem::take(&mut a.values),
            Some(src) => {
                let mut out = Vec::with_capacity(src.len() * comps);
                for &i in src {
                    out.extend_from_slice(&a.values[i as usize * comps..][..comps]);
                }
                out
            }
        };
        debug_assert_eq!(values.len(), n * comps);
        match (a.role, comps) {
            (Role::Normals, 3) if prim.normals.is_empty() => {
                prim.normals = values.as_chunks::<3>().0.iter().map(|&v| unit(v)).collect();
            }
            (Role::TCoords, 2) if prim.uvs.is_empty() => {
                // VTK's coordinates start at the bottom left, glTF's at the top left.
                prim.uvs = values.as_chunks::<2>().0.iter().map(|&[u, v]| [u, 1.0 - v]).collect();
            }
            (Role::Colors, 3) if prim.colors.is_empty() => {
                prim.colors = values
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|&[r, g, b]| linear(&[r, g, b, f32::NAN], a.scale))
                    .collect();
            }
            (Role::Colors, 4) if prim.colors.is_empty() => {
                prim.colors = values.as_chunks::<4>().0.iter().map(|c| linear(c, a.scale)).collect();
            }
            _ => {
                if let Some((colors, (min, max))) = table {
                    let last = colors.len() - 1;
                    #[allow(clippy::cast_precision_loss, reason = "a lookup table is small")]
                    let scale = if max > min {
                        colors.len() as f32 / (max - min)
                    } else {
                        0.0
                    };
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "clamped to the table"
                    )]
                    let pick = |v: f32| colors[(((v - min) * scale).max(0.0) as usize).min(last)];
                    table_colors = Some(values.iter().map(|&v| pick(v)).collect::<Vec<_>>());
                }
                let mut key = format!("_{}", attribute_name(&a.name));
                if custom
                    .iter()
                    .any(|(k, ..)| k == &key || k.starts_with(&format!("{key}_0")))
                {
                    key.push_str(if per_cell { "_CELL" } else { "_2" });
                }
                if comps > MAX_COMPONENTS || custom.iter().any(|(k, ..)| k == &key) {
                    totals.skipped_arrays.push(a.name);
                    continue;
                }
                let mut values = values;
                for v in &mut values {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
                if comps <= 4 {
                    custom.push((key, comps, values));
                } else {
                    for k in 0..comps {
                        let column = values.iter().skip(k).step_by(comps).copied().collect();
                        custom.push((format!("{key}_{k}"), 1, column));
                    }
                }
            }
        }
    }
    if prim.colors.is_empty()
        && let Some(colors) = table_colors
    {
        prim.colors = colors;
    }
    Ok(Some(Drawn { prim, custom }))
}

/// A structured dataset's outside: the six faces of a block, the quads of a sheet, the segments of a row, or
/// its one point.
fn structured(d: &mut Draw, dims: [u32; 3], limit: usize, hexahedra: bool) -> Result<()> {
    let n = dims.map(|n| n as usize);
    // Cells along each axis.
    let c = n.map(|n| (n - 1).max(1));
    let point = |i: [usize; 3]| u32_of(i[0] + n[0] * (i[1] + n[1] * i[2]));
    let cell = |i: [usize; 3]| u32_of(i[0] + c[0] * (i[1] + c[1] * i[2]));
    let real: Vec<usize> = (0..3).filter(|&a| n[a] > 1).collect();
    let quads: u64 = match real[..] {
        [u, v] => c[u] as u64 * c[v] as u64,
        [_, _, _] => 2 * (c[0] as u64 * c[1] as u64 + c[1] as u64 * c[2] as u64 + c[0] as u64 * c[2] as u64),
        // A row's segments count against the same limit.
        [a] => c[a] as u64 / 2,
        _ => 0,
    };
    ensure!(
        quads.saturating_mul(2) <= limit as u64,
        LimitExceeded,
        "too many triangles to convert (more than {limit})"
    );
    match real[..] {
        [] => {
            d.verts.push(0);
            if d.track {
                d.vert_cells.push(0);
            }
        }
        [a] => {
            for k in 0..c[a] {
                let mut i = [0; 3];
                i[a] = k;
                let from = point(i);
                let at = cell(i);
                i[a] = k + 1;
                d.lines.extend_from_slice(&[from, point(i)]);
                if d.track {
                    d.line_cells.push(at);
                }
            }
        }
        [u, v] => {
            // Each quad's corners as a pixel's: along the first axis, then the second.
            for iv in 0..c[v] {
                for iu in 0..c[u] {
                    let corner = |du: usize, dv: usize| {
                        let mut p = [0; 3];
                        p[u] = iu + du;
                        p[v] = iv + dv;
                        point(p)
                    };
                    let mut k = [0; 3];
                    k[u] = iu;
                    k[v] = iv;
                    d.fan(&[corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)], cell(k));
                }
            }
        }
        _ => {
            // The outside is each cell's face on the side it's at the edge of, its corners in the order VTK's
            // own cell has them. Both cells list their faces low and high across x, then y, then z.
            let volume = if hexahedra { &shapes::HEXAHEDRON } else { &shapes::VOXEL };
            for (side, face) in volume.faces.iter().enumerate() {
                let (a, far) = (side / 2, side % 2 == 1);
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                for iv in 0..c[v] {
                    for iu in 0..c[u] {
                        let mut k = [0; 3];
                        k[a] = if far { c[a] - 1 } else { 0 };
                        k[u] = iu;
                        k[v] = iv;
                        let corners = [0, 1, 2, 3].map(|q| {
                            // A hexahedron's nodes go around its bottom and then its top; a voxel's run along x,
                            // then y, then z.
                            let node = if hexahedra {
                                [0, 1, 3, 2, 4, 5, 7, 6][usize::from(face[q])]
                            } else {
                                usize::from(face[q])
                            };
                            point([k[0] + (node & 1), k[1] + (node >> 1 & 1), k[2] + (node >> 2)])
                        });
                        d.fan(&corners, cell(k));
                    }
                }
            }
        }
    }
    Ok(())
}

/// A class for each drawn cell (the rest are [`BAD`]): two cells are in one when every value they carry is the
/// same, so a point they share needs only one vertex.
fn classes(arrays: &[Array], cells: usize, d: &Draw) -> Vec<u32> {
    let mut class = vec![BAD; cells];
    for &c in d.tri_cells.iter().chain(&d.line_cells).chain(&d.vert_cells) {
        class[c as usize] = 0;
    }
    let mut seen: FastMap<(u32, u32), u32> = FastMap::default();
    for a in arrays {
        for k in 0..a.comps {
            seen.clear();
            for (c, class) in class.iter_mut().enumerate().filter(|(_, class)| **class != BAD) {
                let next = u32_of(seen.len());
                *class = *seen
                    .entry((*class, a.values[c * a.comps + k].to_bits()))
                    .or_insert(next);
            }
        }
    }
    class
}

/// Gives what's drawn its own vertices, rewriting its indices: the points used, in the file's order when cells
/// carry no data, otherwise in the order they're drawn, a point once for each class of the cells around it. Returns the point each vertex came from (`None` when the vertices are the points as they
/// are) and, with classes, a cell it came from.
fn weld(d: &mut Draw, points: usize, classes: Option<&[u32]>, dense: bool) -> (Option<Vec<u32>>, Vec<u32>) {
    let Draw {
        tris,
        lines,
        verts,
        tri_cells,
        line_cells,
        vert_cells,
        ..
    } = d;
    if classes.is_none() && dense {
        let mut remap = vec![BAD; points];
        for &i in tris.iter().chain(lines.iter()).chain(verts.iter()) {
            remap[i as usize] = 0;
        }
        let mut src = Vec::new();
        for (i, r) in remap.iter_mut().enumerate().filter(|(_, r)| **r == 0) {
            *r = u32_of(src.len());
            src.push(u32_of(i));
        }
        if src.len() == points {
            return (None, Vec::new());
        }
        for i in tris.iter_mut().chain(lines.iter_mut()).chain(verts.iter_mut()) {
            *i = remap[*i as usize];
        }
        return (Some(src), Vec::new());
    }
    let mut seen: FastMap<(u32, u32), u32> = FastMap::default();
    let mut src_points = Vec::new();
    let mut src_cells = Vec::new();
    for (items, cells, per) in [
        (tris, &*tri_cells, 3),
        (lines, &*line_cells, 2),
        (verts, &*vert_cells, 1),
    ] {
        for (k, item) in items.chunks_exact_mut(per).enumerate() {
            let cell = cells.get(k).copied().unwrap_or(0);
            let class = classes.map_or(0, |c| c[cell as usize]);
            for i in item {
                let next = u32_of(src_points.len());
                let at = *seen.entry((*i, class)).or_insert(next);
                if at == next {
                    src_points.push(*i);
                    if classes.is_some() {
                        src_cells.push(cell);
                    }
                }
                *i = at;
            }
        }
    }
    if classes.is_none() {
        // The points used, back in the file's order.
        let mut order: Vec<u32> = (0..u32_of(src_points.len())).collect();
        order.sort_unstable_by_key(|&v| src_points[v as usize]);
        let mut place = vec![0; order.len()];
        for (to, &from) in order.iter().enumerate() {
            place[from as usize] = u32_of(to);
        }
        let Draw { tris, lines, verts, .. } = d;
        for i in tris.iter_mut().chain(lines.iter_mut()).chain(verts.iter_mut()) {
            *i = place[*i as usize];
        }
        src_points = order.iter().map(|&v| src_points[v as usize]).collect();
    }
    (Some(src_points), src_cells)
}

/// The least and greatest finite values.
fn range(values: &[f32]) -> (f32, f32) {
    let finite = values.iter().filter(|v| v.is_finite());
    finite.fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)))
}

/// A normal made unit length; one with no direction points up.
fn unit(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len.is_finite() && len > 1e-12 {
        v.map(|c| c / len)
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// A color as written, `scale` bringing it to 0-1, as linear RGBA. A missing alpha (NaN) is opaque.
fn linear(c: &[f32; 4], scale: f32) -> [f32; 4] {
    let part = |x: f32| {
        if x.is_finite() {
            (x * scale).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let a = if c[3].is_nan() { 1.0 } else { part(c[3]) };
    [
        srgb_to_linear(part(c[0])),
        srgb_to_linear(part(c[1])),
        srgb_to_linear(part(c[2])),
        a,
    ]
}

/// `name` in capitals with anything but letters and digits as `_`.
pub(super) fn attribute_name(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "VALUE".into() } else { s }
}
