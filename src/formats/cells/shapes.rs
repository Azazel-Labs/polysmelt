//! VTK's cells as what's drawn of them: points, lines, triangles, and the boundary faces of volume cells.
//!
//! A cell type's number picks its [`Shape`]. Quadratic and higher-order cells list their corner nodes first, so
//! each draws as the linear cell its corners make. A volume cell gives its faces, wound to face outward as VTK's
//! own face tables wind them, to [`Faces`], which is left with the ones only one cell has: the mesh's boundary.

use super::{BAD, Points};
use crate::mesh::{quad, triangulate};
use crate::num::u32_of;

/// A volume cell: how many corner nodes it has, and its faces as indices into them.
pub(super) struct Volume {
    corners: usize,
    pub faces: &'static [&'static [u8]],
}

const TETRA: Volume = Volume {
    corners: 4,
    faces: &[&[0, 1, 3], &[1, 2, 3], &[2, 0, 3], &[0, 2, 1]],
};
pub(super) const HEXAHEDRON: Volume = Volume {
    corners: 8,
    faces: &[
        &[0, 4, 7, 3],
        &[1, 2, 6, 5],
        &[0, 1, 5, 4],
        &[3, 7, 6, 2],
        &[0, 3, 2, 1],
        &[4, 5, 6, 7],
    ],
};
/// A voxel's nodes run along x, then y, then z. Its faces are the low and the high one across x, then y, then
/// z, as a structured grid's cells have them too.
pub(super) const VOXEL: Volume = Volume {
    corners: 8,
    faces: &[
        &[2, 0, 4, 6],
        &[1, 3, 7, 5],
        &[0, 1, 5, 4],
        &[3, 2, 6, 7],
        &[1, 0, 2, 3],
        &[4, 5, 7, 6],
    ],
};
const WEDGE: Volume = Volume {
    corners: 6,
    faces: &[&[0, 2, 1], &[3, 4, 5], &[0, 1, 4, 3], &[1, 2, 5, 4], &[2, 0, 3, 5]],
};
const PYRAMID: Volume = Volume {
    corners: 5,
    faces: &[&[0, 3, 2, 1], &[0, 1, 4], &[1, 2, 4], &[2, 3, 4], &[3, 0, 4]],
};
const PENTAGONAL_PRISM: Volume = Volume {
    corners: 10,
    faces: &[
        &[0, 4, 3, 2, 1],
        &[5, 6, 7, 8, 9],
        &[0, 1, 6, 5],
        &[1, 2, 7, 6],
        &[2, 3, 8, 7],
        &[3, 4, 9, 8],
        &[4, 0, 5, 9],
    ],
};
const HEXAGONAL_PRISM: Volume = Volume {
    corners: 12,
    faces: &[
        &[0, 5, 4, 3, 2, 1],
        &[6, 7, 8, 9, 10, 11],
        &[0, 1, 7, 6],
        &[1, 2, 8, 7],
        &[2, 3, 9, 8],
        &[3, 4, 10, 9],
        &[4, 5, 11, 10],
        &[5, 0, 6, 11],
    ],
};

/// The most corners a face has (a hexagonal prism's ends).
const MAX_FACE: usize = 6;

/// What a cell draws as.
pub(super) enum Shape {
    /// Nothing: an empty cell.
    Nothing,
    /// A point at each node.
    Points,
    /// A line through all the nodes.
    Polyline,
    /// A line between the first two nodes: the ends of a curved edge.
    Edge,
    /// A polygon of the first so many nodes.
    Polygon(usize),
    /// A polygon of all the nodes.
    AnyPolygon,
    /// A polygon of the first half of the nodes: the corners of a quadratic polygon.
    HalfPolygon,
    /// A quad whose nodes run along x and then y.
    Pixel,
    /// A triangle strip.
    Strip,
    Volume(&'static Volume),
    /// A cell that isn't drawn: a polyhedron, a convex point set, or a number VTK doesn't define.
    Unknown,
}

/// The shape of cell type `ty`.
pub(super) fn shape(ty: u8) -> Shape {
    match ty {
        0 => Shape::Nothing,
        1 | 2 => Shape::Points,
        3 | 4 => Shape::Polyline,
        // Quadratic and cubic edges, Lagrange and Bezier curves.
        21 | 35 | 68 | 75 => Shape::Edge,
        // Triangles: linear, quadratic, biquadratic, Lagrange and Bezier.
        5 | 22 | 34 | 69 | 76 => Shape::Polygon(3),
        6 => Shape::Strip,
        7 => Shape::AnyPolygon,
        8 => Shape::Pixel,
        // Quads: linear, quadratic, biquadratic, quadratic-linear, Lagrange and Bezier.
        9 | 23 | 28 | 30 | 70 | 77 => Shape::Polygon(4),
        36 => Shape::HalfPolygon,
        10 | 24 | 71 | 78 => Shape::Volume(&TETRA),
        11 => Shape::Volume(&VOXEL),
        12 | 25 | 29 | 33 | 72 | 79 => Shape::Volume(&HEXAHEDRON),
        13 | 26 | 31 | 32 | 73 | 80 => Shape::Volume(&WEDGE),
        14 | 27 | 37 | 74 | 81 => Shape::Volume(&PYRAMID),
        15 => Shape::Volume(&PENTAGONAL_PRISM),
        16 => Shape::Volume(&HEXAGONAL_PRISM),
        _ => Shape::Unknown,
    }
}

/// A cell type's name, for the source notes.
pub(super) fn name(ty: u8) -> String {
    let known = match ty {
        0 => "empty",
        1 => "vertex",
        2 => "polyVertex",
        3 => "line",
        4 => "polyLine",
        5 => "triangle",
        6 => "triangleStrip",
        7 => "polygon",
        8 => "pixel",
        9 => "quad",
        10 => "tetra",
        11 => "voxel",
        12 => "hexahedron",
        13 => "wedge",
        14 => "pyramid",
        15 => "pentagonalPrism",
        16 => "hexagonalPrism",
        21 => "quadraticEdge",
        22 => "quadraticTriangle",
        23 => "quadraticQuad",
        24 => "quadraticTetra",
        25 => "quadraticHexahedron",
        26 => "quadraticWedge",
        27 => "quadraticPyramid",
        28 => "biquadraticQuad",
        29 => "triquadraticHexahedron",
        30 => "quadraticLinearQuad",
        31 => "quadraticLinearWedge",
        32 => "biquadraticQuadraticWedge",
        33 => "biquadraticQuadraticHexahedron",
        34 => "biquadraticTriangle",
        35 => "cubicLine",
        36 => "quadraticPolygon",
        37 => "triquadraticPyramid",
        41 => "convexPointSet",
        42 => "polyhedron",
        68 => "lagrangeCurve",
        69 => "lagrangeTriangle",
        70 => "lagrangeQuadrilateral",
        71 => "lagrangeTetrahedron",
        72 => "lagrangeHexahedron",
        73 => "lagrangeWedge",
        74 => "lagrangePyramid",
        75 => "bezierCurve",
        76 => "bezierTriangle",
        77 => "bezierQuadrilateral",
        78 => "bezierTetrahedron",
        79 => "bezierHexahedron",
        80 => "bezierWedge",
        81 => "bezierPyramid",
        _ => return format!("type{ty}"),
    };
    known.to_owned()
}

/// One face of a volume cell, until another cell turns out to have it too.
#[derive(Clone, Copy)]
struct Face {
    /// Its corners, sorted: what makes two cells' faces the same one.
    key: [u32; 4],
    /// The cell it's a face of; [`BAD`] marks an empty slot.
    cell: u32,
    /// Which of the cell's faces.
    local: u32,
}

const NO_FACE: Face = Face {
    key: [BAD; 4],
    cell: BAD,
    local: 0,
};

/// The faces of the volume cells read so far that only one cell has. A face goes in when a cell brings it and
/// comes out when a second cell brings it again, so what's left at the end is the boundary. (A face three
/// cells share, which no proper mesh has, is left in.) Cells near each other in a file share their faces, so
/// the faces waiting for their second cell stay few and in the cache, however many the mesh has.
pub(super) struct Faces {
    /// Open addressing, linear probing.
    table: Vec<Face>,
    held: usize,
    /// 64 minus log2 of the table's size: a hash's top bits pick the slot.
    shift: u32,
}

impl Default for Faces {
    fn default() -> Self {
        Self {
            table: vec![NO_FACE; 1024],
            held: 0,
            shift: 64 - 10,
        }
    }
}

impl Faces {
    #[inline]
    fn slot(&self, key: [u32; 4]) -> usize {
        let h = (u64::from(key[0]) | u64::from(key[1]) << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (u64::from(key[2]) | u64::from(key[3]) << 32).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "`shift` leaves only as many bits as the table has slots"
        )]
        let slot = (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        slot
    }

    /// Whether no face is waiting.
    pub fn is_empty(&self) -> bool {
        self.held == 0
    }

    /// Adds face `local` of `cell`, whose sorted corners are `key`, or takes out the same face of another cell.
    #[inline]
    fn add(&mut self, key: [u32; 4], cell: u32, local: u32) {
        let mask = self.table.len() - 1;
        let mut i = self.slot(key);
        while self.table[i].cell != BAD {
            if self.table[i].key == key {
                self.remove(i);
                return;
            }
            i = (i + 1) & mask;
        }
        self.table[i] = Face { key, cell, local };
        self.held += 1;
        // Keep the table at most 5/8 full.
        if self.held * 8 > self.table.len() * 5 {
            self.grow();
        }
    }

    /// Empties slot `at`, moving back the faces after it that probing would no longer find.
    #[inline]
    fn remove(&mut self, at: usize) {
        let mask = self.table.len() - 1;
        let mut hole = at;
        let mut i = at;
        loop {
            i = (i + 1) & mask;
            if self.table[i].cell == BAD {
                break;
            }
            // A face stays where it is when its own slot lies after the hole.
            let home = self.slot(self.table[i].key);
            if (i.wrapping_sub(home) & mask) >= (i.wrapping_sub(hole) & mask) {
                self.table[hole] = self.table[i];
                hole = i;
            }
        }
        self.table[hole] = NO_FACE;
        self.held -= 1;
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.table);
        self.table = vec![NO_FACE; old.len() * 2];
        self.shift -= 1;
        let mask = self.table.len() - 1;
        for face in old.into_iter().filter(|f| f.cell != BAD) {
            let mut i = self.slot(face.key);
            while self.table[i].cell != BAD {
                i = (i + 1) & mask;
            }
            self.table[i] = face;
        }
    }

    /// Adds the faces of `cell`, a volume cell of nodes `ids`; `false` when it has too few nodes.
    #[inline]
    pub fn cell(&mut self, volume: &Volume, ids: &[u32], cell: u32) -> bool {
        let Some(ids) = ids.get(..volume.corners) else {
            return false;
        };
        for (local, face) in volume.faces.iter().enumerate() {
            let mut key = [BAD; 4];
            if face.len() <= 4 {
                for (k, &c) in key.iter_mut().zip(face.iter()) {
                    *k = ids[usize::from(c)];
                }
                key.sort_unstable();
            } else {
                // The three lowest corners, and the rest folded into the fourth.
                let mut all = [BAD; MAX_FACE];
                for (k, &c) in all.iter_mut().zip(face.iter()) {
                    *k = ids[usize::from(c)];
                }
                all.sort_unstable();
                key[..3].copy_from_slice(&all[..3]);
                key[3] = all[3..]
                    .iter()
                    .fold(0x8000_0000, |h, &i| (h ^ i).wrapping_mul(0x9E37_79B1) | 0x8000_0000);
            }
            if face.len() <= 4 && key.windows(2).any(|w| w[0] == w[1] && w[0] != BAD) {
                // Corners repeated, where a cell is collapsed (a wedge written as a hexahedron): the face is
                // the triangle left, or nothing at all.
                let mut distinct = [BAD; 4];
                let mut n = 0;
                for &k in key.iter().filter(|&&k| k != BAD) {
                    if n == 0 || distinct[n - 1] != k {
                        distinct[n] = k;
                        n += 1;
                    }
                }
                if n < 3 {
                    continue;
                }
                key = distinct;
            }
            self.add(key, cell, u32_of(local));
        }
        true
    }

    /// Calls `f` with the corners of each face only one cell has, wound to face out of it, and that cell, in the
    /// cells' order. `nodes` gives a cell's type and nodes.
    pub fn boundary<'c>(&self, nodes: impl Fn(u32) -> (u8, &'c [u32]), mut f: impl FnMut(&[u32], u32)) {
        let mut left: Vec<(u32, u32)> = self
            .table
            .iter()
            .filter(|f| f.cell != BAD)
            .map(|f| (f.cell, f.local))
            .collect();
        left.sort_unstable();
        let mut corners = [0u32; MAX_FACE];
        for (cell, local) in left {
            let (ty, ids) = nodes(cell);
            let Shape::Volume(volume) = shape(ty) else { continue };
            let local = volume.faces[local as usize];
            let mut n = 0;
            for &c in local {
                let id = ids[usize::from(c)];
                // A corner repeated, where a cell is collapsed, is taken once.
                if n == 0 || (corners[n - 1] != id && corners[0] != id) {
                    corners[n] = id;
                    n += 1;
                }
            }
            if n >= 3 {
                f(&corners[..n], cell);
            }
        }
    }
}

/// What's drawn: triangles, line segments and points as indices into a piece's points, and the cell each came
/// from when cells carry data.
pub(super) struct Draw {
    pub tris: Vec<u32>,
    pub lines: Vec<u32>,
    pub verts: Vec<u32>,
    pub tri_cells: Vec<u32>,
    pub line_cells: Vec<u32>,
    pub vert_cells: Vec<u32>,
    /// Whether to note each item's cell.
    pub track: bool,
    /// Cells left out: a node out of range, or too few nodes for the type.
    pub dropped: usize,
    /// Cells drawn or skipped, by type.
    pub counts: [usize; 256],
    scratch: Vec<[f32; 3]>,
    local: Vec<[u32; 3]>,
}

impl Draw {
    /// Nothing drawn yet; `track` says whether to note each item's cell.
    pub fn new(track: bool) -> Self {
        Self {
            tris: Vec::new(),
            lines: Vec::new(),
            verts: Vec::new(),
            tri_cells: Vec::new(),
            line_cells: Vec::new(),
            vert_cells: Vec::new(),
            track,
            dropped: 0,
            counts: [0; 256],
            scratch: Vec::new(),
            local: Vec::new(),
        }
    }

    #[inline]
    fn tri(&mut self, t: [u32; 3], cell: u32) {
        self.tris.extend_from_slice(&t);
        if self.track {
            self.tri_cells.push(cell);
        }
    }

    #[inline]
    fn line(&mut self, a: u32, b: u32, cell: u32) {
        self.lines.extend_from_slice(&[a, b]);
        if self.track {
            self.line_cells.push(cell);
        }
    }

    /// A face of a volume cell or of a grid: convex, so fanned from its first corner.
    #[inline]
    pub fn fan(&mut self, ids: &[u32], cell: u32) {
        for k in 1..ids.len().saturating_sub(1) {
            self.tri([ids[0], ids[k], ids[k + 1]], cell);
        }
    }

    /// A polygon cell, triangulated in its plane.
    fn polygon(&mut self, ids: &[u32], cell: u32, points: &Points) {
        match *ids {
            [] | [_] => self.dropped += 1,
            [a, b] => self.line(a, b, cell),
            [a, b, c] => self.tri([a, b, c], cell),
            [a, b, c, d] => {
                let p = [a, b, c, d].map(|i| points.at(i));
                if let Some(two) = quad(&p) {
                    for t in two {
                        self.tri(t.map(|k| ids[k as usize]), cell);
                    }
                } else {
                    self.ear_clip(ids, cell, points);
                }
            }
            _ => self.ear_clip(ids, cell, points),
        }
    }

    fn ear_clip(&mut self, ids: &[u32], cell: u32, points: &Points) {
        let mut pts = std::mem::take(&mut self.scratch);
        let mut local = std::mem::take(&mut self.local);
        pts.clear();
        pts.extend(ids.iter().map(|&i| points.at(i)));
        local.clear();
        triangulate(&pts, &mut local);
        for t in &local {
            self.tri(t.map(|k| ids[k as usize]), cell);
        }
        self.scratch = pts;
        self.local = local;
    }

    /// Draws `cell`, of type `ty` and nodes `ids` (each within the points): its volume faces go to `faces`.
    #[inline]
    pub fn cell(&mut self, ty: u8, ids: &[u32], cell: u32, points: &Points, faces: &mut Faces) {
        self.counts[usize::from(ty)] += 1;
        match shape(ty) {
            Shape::Nothing | Shape::Unknown => {}
            Shape::Points => {
                self.verts.extend_from_slice(ids);
                if self.track {
                    self.vert_cells.extend(std::iter::repeat_n(cell, ids.len()));
                }
            }
            Shape::Polyline => {
                if ids.len() < 2 {
                    self.dropped += 1;
                }
                for w in ids.windows(2) {
                    self.line(w[0], w[1], cell);
                }
            }
            Shape::Edge => match ids {
                [a, b, ..] => self.line(*a, *b, cell),
                _ => self.dropped += 1,
            },
            Shape::Polygon(n) => match ids.get(..n) {
                Some(corners) => self.polygon(corners, cell, points),
                None => self.dropped += 1,
            },
            Shape::AnyPolygon => self.polygon(ids, cell, points),
            Shape::HalfPolygon => self.polygon(&ids[..ids.len() / 2], cell, points),
            Shape::Pixel => match *ids {
                [a, b, c, d, ..] => self.polygon(&[a, b, d, c], cell, points),
                _ => self.dropped += 1,
            },
            Shape::Strip => {
                if ids.len() < 3 {
                    self.dropped += 1;
                }
                for (k, w) in ids.windows(3).enumerate() {
                    // Every other triangle is turned, so all wind the same way.
                    let t = if k % 2 == 0 {
                        [w[0], w[1], w[2]]
                    } else {
                        [w[1], w[0], w[2]]
                    };
                    // A strip's triangle on one point twice restarts the strip, and draws nothing.
                    if crate::geometry::distinct(t) {
                        self.tri(t, cell);
                    }
                }
            }
            Shape::Volume(volume) => {
                if !faces.cell(volume, ids, cell) {
                    self.dropped += 1;
                }
            }
        }
    }
}
