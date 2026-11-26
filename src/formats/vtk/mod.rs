//! VTK files: legacy `.vtk`, and XML `.vtp` (`PolyData`) and `.vtu` (`UnstructuredGrid`).
//!
//! Legacy files (versions 1.0 to 5.1) are ASCII or big-endian binary: `POLYDATA`, `UNSTRUCTURED_GRID` (cells as
//! counted lists, or as version 5's `OFFSETS` and `CONNECTIVITY`), `STRUCTURED_GRID`, `STRUCTURED_POINTS` and
//! `RECTILINEAR_GRID`. XML files hold their arrays as text, as base64 beside the tags or appended (raw or
//! base64), in either byte order, with 32-bit or 64-bit block headers, plain or zlib-compressed, in any number
//! of pieces. LZ4 and LZMA compression are refused by name. The parallel and multi-block files (`.pvtu`, `.vtm`)
//! and the other XML datasets (`.vti`, `.vtr`, `.vts`) aren't read.
//!
//! What's drawn: vertex cells as points, line cells as lines, and triangles, quads, pixels, polygons (cut in
//! their plane) and strips as triangles. Volume cells (tetrahedra, voxels, hexahedra, wedges, pyramids, and
//! pentagonal and hexagonal prisms) are drawn as the mesh's boundary: the faces only one cell has, wound to
//! face outward as VTK winds them. Quadratic, Lagrange and Bezier cells are drawn by their corner nodes.
//! Polyhedra and convex point sets aren't drawn; their counts are in the source notes. A structured grid,
//! image or rectilinear grid is its six outer faces, or all of its quads when it's one layer thick, a line when
//! it's one row, and a point when it's one point. An image or rectilinear grid with no point or cell data,
//! which costs a file nothing however large, draws at most four million triangles. A file of points with no
//! cells draws its points.
//!
//! Points no drawn cell uses are left out, the rest keeping their order. Point data follows its points: normals
//! (made unit length), two-component texture coordinates (flipped to glTF's top-left origin) and colors become
//! `NORMAL`, `TEXCOORD_0` and `COLOR_0`; every other array is a float custom attribute named `_` and its name
//! in capitals, `SCALAR` to `VEC4` by its components (more, up to nine, are an attribute each: `_NAME_0`,
//! `_NAME_1`). Colors are `COLOR_SCALARS` of three or four values, or an unsigned byte array of three or four
//! components, taken as sRGB; without those, scalars naming a `LOOKUP_TABLE` the file holds are colored
//! through it, across their range. Cell data is kept the same way: a vertex is split where the cells around
//! it differ in any value, so each cell's values sit on its own corners.
//!
//! No normals are made: without them glTF shades flat, which is what VTK draws. The material is double-sided,
//! as VTK lights both sides. Values that aren't finite are zero in custom attributes, and a cell at a point that
//! isn't finite is dropped. Dataset field data of a few values goes to the source notes with the title, the
//! version, the encoding and the cell counts by type. VTK names no units and no up axis, so both are unknown,
//! and geometry is kept as written.

mod cells;
mod legacy;
#[cfg(test)]
mod tests;
mod xml;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream;
use crate::document::{Builder, Document, Part};
use crate::mesh::{FastMap, GRAY, Material, Prim, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::source::SourceInfo;
use crate::{Context as _, ErrorKind, Result};
use cells::{Draw, Faces, Shape};

/// No point: an index that's negative or out of range, or an unused slot.
const BAD: u32 = u32::MAX;

/// The most components an array has to be kept as custom attributes: a full tensor's nine.
const MAX_COMPONENTS: usize = 9;

/// The most triangles an image or rectilinear grid draws when the file holds no data for its points or cells:
/// such a file is a few lines however many points it counts, so this bounds what a few lines can ask for.
const MAX_BARE_TRIANGLES: usize = 1 << 22;

/// The most values of dataset field data kept in the source notes.
const MAX_FIELD_VALUES: usize = 16;

/// A value's type in a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    /// One bit, eight to a byte from the top; legacy files only.
    Bit,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Ty {
    /// The bytes one value takes (a bit counts as one here; see [`packed`]).
    fn size(self) -> usize {
        match self {
            Ty::Bit | Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::I64 | Ty::U64 | Ty::F64 => 8,
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "attributes are single precision; whole numbers past 2^24 round"
)]
fn wide(x: i64) -> f32 {
    x as f32
}

#[allow(
    clippy::cast_precision_loss,
    reason = "attributes are single precision; whole numbers past 2^24 round"
)]
fn wide_unsigned(x: u64) -> f32 {
    x as f32
}

/// Decodes `b`, whole values of `$n` bytes each in either byte order, with `$f` on each.
macro_rules! decode {
    ($b:expr, $big:expr, $n:literal, $t:ty, $f:expr) => {{
        let (chunks, _) = $b.as_chunks::<$n>();
        if $big {
            chunks.iter().map(|&c| $f(<$t>::from_be_bytes(c))).collect()
        } else {
            chunks.iter().map(|&c| $f(<$t>::from_le_bytes(c))).collect()
        }
    }};
}

/// The values of type `ty` in `b` as floats.
fn floats_of(b: &[u8], ty: Ty, big: bool) -> Vec<f32> {
    match ty {
        Ty::Bit | Ty::U8 => b.iter().map(|&x| f32::from(x)).collect(),
        Ty::I8 => b.iter().map(|&x| f32::from(x.cast_signed())).collect(),
        Ty::I16 => decode!(b, big, 2, i16, f32::from),
        Ty::U16 => decode!(b, big, 2, u16, f32::from),
        Ty::I32 => decode!(b, big, 4, i32, |x| wide(i64::from(x))),
        Ty::U32 => decode!(b, big, 4, u32, |x| wide(i64::from(x))),
        Ty::I64 => decode!(b, big, 8, i64, wide),
        Ty::U64 => decode!(b, big, 8, u64, wide_unsigned),
        Ty::F32 => decode!(b, big, 4, f32, |x| x),
        Ty::F64 => decode!(b, big, 8, f64, f32_of),
    }
}

/// The values of type `ty` in `b` as indices: [`BAD`] for one that's negative, too big, or no whole number.
fn ids_of(b: &[u8], ty: Ty, big: bool) -> Vec<u32> {
    fn id<T: TryInto<u32>>(x: T) -> u32 {
        x.try_into().unwrap_or(BAD)
    }
    match ty {
        Ty::Bit | Ty::U8 => b.iter().map(|&x| u32::from(x)).collect(),
        Ty::I8 => b.iter().map(|&x| id(x.cast_signed())).collect(),
        Ty::I16 => decode!(b, big, 2, i16, id),
        Ty::U16 => decode!(b, big, 2, u16, u32::from),
        Ty::I32 => decode!(b, big, 4, i32, id),
        Ty::U32 => decode!(b, big, 4, u32, |x| x),
        Ty::I64 => decode!(b, big, 8, i64, id),
        Ty::U64 => decode!(b, big, 8, u64, id),
        Ty::F32 | Ty::F64 => vec![BAD; b.len() / ty.size()],
    }
}

/// `n` bits packed eight to a byte from the top, a byte each.
fn packed(b: &[u8], n: usize) -> Vec<u8> {
    (0..n).map(|i| (b[i / 8] >> (7 - i % 8)) & 1).collect()
}

/// A place in a file's text: the legacy keywords and numbers, and the XML arrays written as text.
struct Cursor<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, at: 0 }
    }

    #[inline]
    fn skip_space(&mut self) {
        while self.at < self.d.len() && stream::space(self.d[self.at]) {
            self.at += 1;
        }
    }

    /// The next token, across line ends.
    #[inline]
    fn token(&mut self) -> Option<&'a [u8]> {
        self.skip_space();
        let start = self.at;
        while self.at < self.d.len() && !stream::space(self.d[self.at]) {
            self.at += 1;
        }
        (self.at > start).then(|| &self.d[start..self.at])
    }

    /// The next line that isn't blank, and the cursor past its end: binary data starts right there.
    fn line(&mut self) -> Option<&'a [u8]> {
        loop {
            let line = stream::trim(stream::next_line(self.d, &mut self.at)?);
            if !line.is_empty() {
                return Some(line);
            }
        }
    }

    /// How many values the text left can hold: each takes a character and a separator.
    fn room(&self) -> usize {
        (self.d.len() - self.at.min(self.d.len())).div_ceil(2)
    }

    /// `n` numbers as floats.
    fn floats(&mut self, n: usize, what: &str) -> Result<Vec<f32>> {
        ensure!(
            n <= self.room(),
            Truncated,
            "VTK file is cut short: {what} claims {n} values, more than it holds"
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
                bail!(Truncated, "VTK file is cut short: {what} ends early");
            };
            let Some(v) = stream::f32_lenient(t) else {
                bail!(Corrupt, "VTK {what}: {} is not a number", crate::shown(t));
            };
            out.push(v);
        }
        Ok(out)
    }

    /// `n` whole numbers as indices: [`BAD`] for a negative one or one too big.
    fn ids(&mut self, n: usize, what: &str) -> Result<Vec<u32>> {
        ensure!(
            n <= self.room(),
            Truncated,
            "VTK file is cut short: {what} claims {n} values, more than it holds"
        );
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(v) = self.id(what)? else {
                bail!(Truncated, "VTK file is cut short: {what} ends early");
            };
            out.push(v);
        }
        Ok(out)
    }

    /// The next whole number as an index, [`BAD`] for a negative one or one too big; `None` at the end.
    #[inline]
    fn id(&mut self, what: &str) -> Result<Option<u32>> {
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
            bail!(Corrupt, "VTK {what}: {} is not a whole number", crate::shown(t));
        };
        Ok(Some(u32::try_from(v).unwrap_or(BAD)))
    }

    /// Every number left, as floats, expecting about `expected` of them.
    fn all_floats(&mut self, expected: usize, what: &str) -> Result<Vec<f32>> {
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
                bail!(Corrupt, "VTK {what}: {} is not a number", crate::shown(t));
            };
            out.push(v);
        }
    }

    /// Every whole number left, as indices, expecting about `expected` of them.
    fn all_ids(&mut self, expected: usize, what: &str) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(expected.min(self.room()));
        while let Some(v) = self.id(what)? {
            out.push(v);
        }
        Ok(out)
    }
}

/// Where a piece's points are.
enum Points {
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
    fn len(&self) -> usize {
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
enum Role {
    Normals,
    TCoords,
    Colors,
    Other,
}

/// A named array of point or cell data.
struct Array {
    name: String,
    /// Values per point or cell.
    comps: usize,
    role: Role,
    values: Vec<f32>,
    /// What brings a color's values to 0-1: 1/255 for bytes.
    scale: f32,
    /// The lookup table its scalars name.
    table: Option<String>,
}

impl Array {
    fn new(name: &str, comps: usize, role: Role, values: Vec<f32>) -> Self {
        Self {
            name: name.to_owned(),
            comps,
            role,
            values,
            scale: 1.0,
            table: None,
        }
    }

    /// An array of unsigned bytes with three or four components is a color, as VTK draws it.
    fn bytes_are_colors(&mut self, ty: Ty) {
        if ty == Ty::U8 && matches!(self.comps, 3 | 4) && self.role == Role::Other {
            self.role = Role::Colors;
            self.scale = 1.0 / 255.0;
        }
    }
}

/// Which cells a list holds. Poly data numbers its cells in this order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Kind {
    Verts,
    Lines,
    Polys,
    Strips,
    /// An unstructured grid's, each with its own type.
    Mixed,
}

/// A list of cells: cell `i`'s nodes are `conn[offsets[i]..offsets[i + 1]]`.
struct Cells {
    kind: Kind,
    conn: Vec<u32>,
    offsets: Vec<u32>,
    /// Each cell's type, for [`Kind::Mixed`].
    types: Vec<u8>,
}

impl Cells {
    fn len(&self) -> usize {
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
    fn from_counted(kind: Kind, cells: usize, list: &[u32]) -> Result<Self> {
        let mut conn = Vec::with_capacity(list.len().saturating_sub(cells));
        let mut offsets = Vec::with_capacity(cells.min(list.len()) + 1);
        offsets.push(0);
        let mut at = 0;
        for _ in 0..cells {
            let nodes = list
                .get(at)
                .and_then(|&n| list.get(at + 1..(at + 1).checked_add(n as usize)?));
            let Some(nodes) = nodes else {
                bail!(Corrupt, "VTK cells run past their list");
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
struct Piece {
    points: Points,
    /// A structured dataset's points along each axis; its cells are implied.
    dims: Option<[u32; 3]>,
    cells: Vec<Cells>,
    point_data: Vec<Array>,
    cell_data: Vec<Array>,
}

/// Checks a structured dataset's dimensions: each at least one, and few enough points to index.
fn check_dims(dims: [u32; 3]) -> Result<()> {
    let points = dims.iter().try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)));
    ensure!(
        dims.iter().all(|&d| d >= 1),
        Corrupt,
        "VTK dimensions {dims:?} have a zero"
    );
    ensure!(
        points.is_some_and(|n| n < u64::from(BAD)),
        LimitExceeded,
        "VTK dimensions {dims:?} are too many points"
    );
    Ok(())
}

/// A file as read.
#[derive(Default)]
struct Model {
    pieces: Vec<Piece>,
    /// Lookup tables by name, as written: sRGB and alpha, 0-1.
    tables: Vec<(String, Vec<[f32; 4]>)>,
    info: SourceInfo,
    /// The format's own notes for the asset's `extras.source`.
    source: Map<String, Value>,
}

pub(crate) fn load(input: Input<'_>) -> Result<Document> {
    let data = input.read()?;
    let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&data);
    let start = data.iter().position(|&b| !stream::space(b)).unwrap_or(data.len());
    let model = match data.get(start) {
        // XML, or UTF-16's byte order mark before it.
        Some(b'<' | 0xFE | 0xFF) => xml::read(data)?,
        Some(b'#') => legacy::read(data)?,
        Some(_) => bail!(UnknownFormat, "not a VTK file"),
        None => bail!(Empty, "VTK file is empty"),
    };
    build(model, input.limit)
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

fn build(model: Model, limit: usize) -> Result<Document> {
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
    let mut primitives = Vec::new();
    let mut colored = false;
    let mut any_points = false;
    let count = pieces.len();
    for piece in pieces {
        any_points |= piece.points.len() > 0;
        let Some(drawn) = draw(piece, &tables, limit, &mut totals)? else {
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
        for mut p in prims {
            p["material"] = json!(0);
            primitives.push(p);
        }
    }
    let skipped: Map<String, Value> = (0..=255u8)
        .filter(|&ty| totals.counts[usize::from(ty)] > 0 && matches!(cells::shape(ty), Shape::Unknown))
        .map(|ty| (cells::name(ty), json!(totals.counts[usize::from(ty)])))
        .collect();
    ensure!(any_points, Empty, "VTK file has no points");
    if primitives.is_empty() {
        let names: Vec<&String> = skipped.keys().collect();
        ensure!(
            names.is_empty(),
            Unsupported,
            "VTK file has only cells that aren't drawn: {names:?}"
        );
        bail!(Empty, "VTK file has nothing to draw");
    }
    let kinds: Map<String, Value> = (0..=255u8)
        .filter(|&ty| totals.counts[usize::from(ty)] > 0)
        .map(|ty| (cells::name(ty), json!(totals.counts[usize::from(ty)])))
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
    info.format = Some("VTK".into());
    info.write(&mut source);
    let mut material = Material::plain("default", if colored { [1.0; 4] } else { GRAY });
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
    json.insert("scenes".into(), json!([{ "nodes": [0] }]));
    json.insert("nodes".into(), json!([{ "mesh": 0 }]));
    json.insert("meshes".into(), json!([{ "primitives": primitives }]));
    Ok(b.finish(json, facts))
}

/// Draws one piece; `None` when it draws nothing.
fn draw(
    mut piece: Piece,
    tables: &[(String, Vec<[f32; 4]>)],
    limit: usize,
    totals: &mut Totals,
) -> Result<Option<Drawn>> {
    let np = piece.points.len();
    if np == 0 {
        return Ok(None);
    }
    ensure!(np < BAD as usize, LimitExceeded, "VTK file has too many points");
    piece.cells.sort_by_key(|c| c.kind);
    let cell_count = match piece.dims {
        Some(dims) => dims.iter().map(|&n| (n as usize - 1).max(1)).product(),
        None => piece.cells.iter().map(Cells::len).sum::<usize>(),
    };
    ensure!(cell_count < BAD as usize, LimitExceeded, "VTK file has too many cells");
    let fits = |a: &Array, n: usize| a.comps > 0 && n.checked_mul(a.comps) == Some(a.values.len());
    piece.point_data.retain(|a| fits(a, np));
    piece.cell_data.retain(|a| fits(a, cell_count));
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
    let mut d = Draw::new(!piece.cell_data.is_empty());
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
    let classes = d.track.then(|| classes(&piece.cell_data, cell_count, &d));
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
    let point_data = std::mem::take(&mut piece.point_data).into_iter().map(|a| (a, false));
    let cell_data = std::mem::take(&mut piece.cell_data).into_iter().map(|a| (a, true));
    for (mut a, per_cell) in point_data.chain(cell_data) {
        let src = if per_cell {
            Some(&src_cells)
        } else {
            src_points.as_ref()
        };
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
            let volume = if hexahedra { &cells::HEXAHEDRON } else { &cells::VOXEL };
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
fn attribute_name(name: &str) -> String {
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

/// A few values of dataset field data for the source notes; `None` for more than a few.
fn field_note(values: &[f32]) -> Option<Value> {
    (!values.is_empty() && values.len() <= MAX_FIELD_VALUES).then(|| {
        let list: Vec<Value> = values
            .iter()
            .map(|&v| if v.is_finite() { json!(v) } else { Value::Null })
            .collect();
        Value::Array(list)
    })
}

/// A count a file gives as `usize`: refused when it's negative or no number.
fn count(token: Option<&[u8]>, what: &str) -> Result<usize> {
    token
        .and_then(stream::int::<usize>)
        .with_context(ErrorKind::Corrupt, || format!("VTK {what} count is missing"))
}
