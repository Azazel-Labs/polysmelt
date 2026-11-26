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

mod legacy;
#[cfg(test)]
mod tests;
mod xml;

use serde_json::{Value, json};

use super::Input;
use super::cells::{self, Array, BAD, Cells, Cursor, Kind, Model, Piece, Points, Role};
use super::stream;
use crate::document::Document;
use crate::num::f32_of;
use crate::{Context as _, ErrorKind, Result};

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
    cells::build(model, "VTK", input.limit)
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
