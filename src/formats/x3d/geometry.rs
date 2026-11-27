//! X3D geometry nodes as primitives: the face and triangle sets, the elevation grid, the
//! extrusion, the box, sphere, cylinder and cone tessellated, the line and point sets, and the
//! 2D nodes.

use std::f32::consts::{PI, TAU};

use super::{Graph, NONE, idx, quat, rotate};
use crate::Result;
use crate::formats::crease::corner_normals;
use crate::mesh::{FastMap, Prim, face_normal, smooth_normals, triangulate};
use crate::num::{f32_of, u32_of};
use crate::vec3::{self, cross, dot, length, sub};

/// How a set's vertex normals are found.
#[derive(Clone, Copy, PartialEq)]
enum Shading {
    /// From the file's normals.
    Given,
    /// Each face's own normal at its corners.
    Flat,
    /// Averaged over every face at a point.
    Smooth,
    /// Averaged over the faces at a point within the crease angle (its cosine) of each other.
    Crease(f32),
}

/// Polygons with an index into each attribute array at every corner.
struct Polys {
    points: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    /// X3D's convention: `t` up.
    uvs: Vec<[f32; 2]>,
    /// Position, normal, color and texture coordinate indices; [`NONE`] where absent.
    corners: Vec<[u32; 4]>,
    /// Where each face's corners start, and after the last, where they end.
    starts: Vec<u32>,
    shading: Shading,
    ccw: bool,
    /// Whether every corner's other indices are its position index or absent, so a vertex is
    /// one per position.
    by_position: bool,
}

impl Polys {
    fn new(g: &Graph, id: usize, points: Vec<[f32; 3]>) -> Self {
        let normals = g.node(id, "normal").map(|n| g.vec3s(n, "vector")).unwrap_or_default();
        let colors = g.node(id, "color").map(|c| g.colors(c)).unwrap_or_default();
        let uvs = g.node(id, "texCoord").map(|t| g.tex_coords(t)).unwrap_or_default();
        Polys {
            points,
            normals,
            colors,
            uvs,
            corners: Vec::new(),
            starts: vec![0],
            shading: Shading::Flat,
            ccw: g.flag(id, "ccw", true),
            by_position: true,
        }
    }

    fn end_face(&mut self) {
        let n = u32_of(self.corners.len());
        if self.starts.last() != Some(&n) {
            self.starts.push(n);
        }
    }

    /// The shading when the file gives no normals: smooth or flat by `per_vertex`, as the
    /// triangle sets have it.
    fn shade(&mut self, per_vertex: bool) {
        self.shading = if !self.normals.is_empty() {
            Shading::Given
        } else if per_vertex {
            Shading::Smooth
        } else {
            Shading::Flat
        };
    }
}

/// Indices into an attribute: per vertex, from an index list or the position's; per face, from
/// an index list or the face's number.
struct Lookup<'g> {
    present: bool,
    per_vertex: bool,
    index: &'g [f64],
}

impl Lookup<'_> {
    #[inline]
    fn at(&self, corner: usize, position: u32, face: usize) -> u32 {
        if !self.present {
            NONE
        } else if self.per_vertex {
            if self.index.is_empty() {
                position
            } else {
                self.index.get(corner).map_or(NONE, |&i| idx(i))
            }
        } else if self.index.is_empty() {
            u32_of(face)
        } else {
            self.index.get(face).map_or(NONE, |&i| idx(i))
        }
    }

    fn by_position(&self) -> bool {
        !self.present || (self.per_vertex && self.index.is_empty())
    }
}

/// The primitive for geometry node `id`, its texture coordinates in X3D's convention, and
/// whether it's seen from both sides; `None` for a kind not drawn. `textured` asks for
/// default texture coordinates where the node has none.
pub(super) fn geometry(
    g: &Graph,
    id: usize,
    textured: bool,
    limit: usize,
    dropped: &mut usize,
) -> Result<Option<(Prim, bool)>> {
    let solid = g.flag(id, "solid", true);
    let points = || g.node(id, "coord").map(|c| g.vec3s(c, "point")).unwrap_or_default();
    let polys = match g.kind(id) {
        "IndexedFaceSet" => {
            let mut p = Polys::new(g, id, points());
            face_set(g, id, &mut p);
            p
        }
        "IndexedTriangleSet" | "IndexedQuadSet" => {
            let mut p = Polys::new(g, id, points());
            let side = if g.kind(id) == "IndexedQuadSet" { 4 } else { 3 };
            let index: Vec<u32> = g.nums(id, "index").iter().map(|&i| idx(i)).collect();
            polygons(g, id, &mut p, index.chunks_exact(side));
            p
        }
        "TriangleSet" | "QuadSet" => {
            let mut p = Polys::new(g, id, points());
            let side = if g.kind(id) == "QuadSet" { 4 } else { 3 };
            let index: Vec<u32> = (0..u32_of(p.points.len())).collect();
            polygons(g, id, &mut p, index.chunks_exact(side));
            p
        }
        kind @ ("IndexedTriangleStripSet" | "IndexedTriangleFanSet") => {
            let mut p = Polys::new(g, id, points());
            let index = g.nums(id, "index");
            let runs: Vec<Vec<u32>> = index
                .split(|&i| i < 0.0)
                .map(|r| r.iter().map(|&i| idx(i)).collect())
                .collect();
            let tris = strips(&runs, kind.ends_with("FanSet"));
            polygons(g, id, &mut p, tris.chunks_exact(3));
            p
        }
        kind @ ("TriangleStripSet" | "TriangleFanSet") => {
            let mut p = Polys::new(g, id, points());
            let field = if kind == "TriangleFanSet" {
                "fanCount"
            } else {
                "stripCount"
            };
            let mut runs = Vec::new();
            let mut at = 0u32;
            for &n in g.nums(id, field) {
                let n = idx(n).min(u32_of(p.points.len()).saturating_sub(at));
                runs.push((at..at + n).collect::<Vec<u32>>());
                at += n;
            }
            let tris = strips(&runs, kind == "TriangleFanSet");
            polygons(g, id, &mut p, tris.chunks_exact(3));
            p
        }
        "IndexedLineSet" | "LineSet" | "PointSet" => return Ok(Some((line_set(g, id, dropped), false))),
        "ElevationGrid" => match elevation_grid(g, id) {
            Some(p) => p,
            None => return Ok(None),
        },
        "Extrusion" => extrusion(g, id, textured, limit)?,
        "Rectangle2D" | "Disk2D" | "ArcClose2D" | "TriangleSet2D" | "Circle2D" | "Arc2D" | "Polyline2D"
        | "Polypoint2D" => {
            let prim = flat(g, id, limit)?;
            // 2D nodes are seen from both sides unless they say otherwise.
            let double = !prim.indices.is_empty() && !g.flag(id, "solid", false);
            return Ok(Some((prim, double)));
        }
        "Box" => {
            let s = g.vec(id, "size", [2.0, 2.0, 2.0]).map(f32_of);
            return Ok(Some((box_prim(s), !solid)));
        }
        "Sphere" => {
            let r = f32_of(g.num(id, "radius", 1.0));
            return Ok(Some((sphere(r), !solid)));
        }
        "Cylinder" => {
            let r = f32_of(g.num(id, "radius", 1.0));
            let h = f32_of(g.num(id, "height", 2.0));
            let parts = [
                g.flag(id, "side", true),
                g.flag(id, "top", true),
                g.flag(id, "bottom", true),
            ];
            return Ok(Some((round(r, r, h, parts), !solid)));
        }
        "Cone" => {
            let r = f32_of(g.num(id, "bottomRadius", 1.0));
            let top = f32_of(g.num(id, "topRadius", 0.0));
            let h = f32_of(g.num(id, "height", 2.0));
            let parts = [g.flag(id, "side", true), false, g.flag(id, "bottom", true)];
            return Ok(Some((round(r, top, h, parts), !solid)));
        }
        _ => return Ok(None),
    };
    let prim = to_prim(&polys, textured, limit, dropped)?;
    Ok(Some((prim, !solid)))
}

/// Vertices of a line or point set, one for each pair of position and color index used.
struct Loose {
    points: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    seen: FastMap<(u32, u32), u32>,
    prim: Prim,
}

impl Loose {
    /// The vertex of position `v` with color `c`; `None` for a position that's missing or not
    /// finite.
    fn vertex(&mut self, v: u32, c: u32) -> Option<u32> {
        let &p = self.points.get(v as usize)?;
        if !p.iter().all(|x| x.is_finite()) {
            return None;
        }
        let n = u32_of(self.prim.positions.len());
        let at = *self.seen.entry((v, c)).or_insert(n);
        if at == n {
            self.prim.positions.push(p);
            if !self.colors.is_empty() {
                // A color index past the colors is white, as an unlit line is with no color.
                self.prim
                    .colors
                    .push(self.colors.get(c as usize).copied().unwrap_or([1.0; 4]));
            }
        }
        Some(at)
    }
}

/// An `IndexedLineSet`'s polylines opened into segments, a `LineSet`'s runs of `vertexCount`
/// points the same way, or a `PointSet`'s points; colors per vertex, or per polyline where
/// `colorPerVertex` is false. Each vertex is read from the file, so none costs more than the
/// file does.
fn line_set(g: &Graph, id: usize, dropped: &mut usize) -> Prim {
    let points = g.node(id, "coord").map(|c| g.vec3s(c, "point")).unwrap_or_default();
    let colors = g.node(id, "color").map(|c| g.colors(c)).unwrap_or_default();
    let n = u32_of(points.len());
    let mut l = Loose {
        points,
        colors,
        seen: FastMap::default(),
        prim: Prim::default(),
    };
    match g.kind(id) {
        "IndexedLineSet" => {
            let color = Lookup {
                present: !l.colors.is_empty(),
                per_vertex: g.flag(id, "colorPerVertex", true),
                index: g.nums(id, "colorIndex"),
            };
            let mut line = 0;
            let mut last = None;
            for (k, &ci) in g.nums(id, "coordIndex").iter().enumerate() {
                if ci < 0.0 {
                    line += 1;
                    last = None;
                    continue;
                }
                let v = idx(ci);
                // A bad index breaks the polyline there.
                let at = l.vertex(v, color.at(k, v, line));
                *dropped += usize::from(at.is_none());
                if let (Some(a), Some(b)) = (last, at)
                    && a != b
                {
                    l.prim.lines.extend_from_slice(&[a, b]);
                }
                last = at;
            }
        }
        "LineSet" => {
            let mut start = 0u32;
            for &count in g.nums(id, "vertexCount") {
                let count = idx(count).min(n - start);
                let mut last = None;
                // A run of fewer than two points draws nothing.
                for v in (start..start + count).filter(|_| count >= 2) {
                    let at = l.vertex(v, v);
                    *dropped += usize::from(at.is_none());
                    if let (Some(a), Some(b)) = (last, at) {
                        l.prim.lines.extend_from_slice(&[a, b]);
                    }
                    last = at;
                }
                start += count;
            }
        }
        _ => {
            for v in 0..n {
                match l.vertex(v, v) {
                    Some(at) => l.prim.points.push(at),
                    None => *dropped += 1,
                }
            }
        }
    }
    l.prim
}

/// An `IndexedFaceSet`'s faces: `coordIndex` with -1 between faces, and the other index lists
/// beside it.
fn face_set(g: &Graph, id: usize, p: &mut Polys) {
    let coord_index = g.nums(id, "coordIndex");
    let crease = g.num(id, "creaseAngle", 0.0);
    let normal_per_vertex = g.flag(id, "normalPerVertex", true);
    let color_per_vertex = g.flag(id, "colorPerVertex", true);
    let normal = Lookup {
        present: !p.normals.is_empty(),
        per_vertex: normal_per_vertex,
        index: g.nums(id, "normalIndex"),
    };
    let color = Lookup {
        present: !p.colors.is_empty(),
        per_vertex: color_per_vertex,
        index: g.nums(id, "colorIndex"),
    };
    let uv = Lookup {
        present: !p.uvs.is_empty(),
        per_vertex: true,
        index: g.nums(id, "texCoordIndex"),
    };
    p.by_position = normal.by_position() && color.by_position() && uv.by_position();
    p.shading = shading(normal.present, crease);
    p.corners.reserve(coord_index.len());
    let mut face = 0;
    for (k, &ci) in coord_index.iter().enumerate() {
        if ci < 0.0 {
            p.end_face();
            face = p.starts.len() - 1;
            continue;
        }
        let v = idx(ci);
        p.corners
            .push([v, normal.at(k, v, face), color.at(k, v, face), uv.at(k, v, face)]);
    }
    p.end_face();
}

/// Polygons given as position indices, the attributes per vertex (by the same index) or per
/// face, as the triangle and quad sets have them.
fn polygons<'i>(g: &Graph, id: usize, p: &mut Polys, faces: impl Iterator<Item = &'i [u32]>) {
    let normal_per_vertex = g.flag(id, "normalPerVertex", true);
    let normal = Lookup {
        present: !p.normals.is_empty(),
        per_vertex: normal_per_vertex,
        index: &[],
    };
    let color = Lookup {
        present: !p.colors.is_empty(),
        per_vertex: g.flag(id, "colorPerVertex", true),
        index: &[],
    };
    let uv = Lookup {
        present: !p.uvs.is_empty(),
        per_vertex: true,
        index: &[],
    };
    p.by_position = normal.by_position() && color.by_position();
    p.shade(normal_per_vertex);
    for (face, f) in faces.enumerate() {
        for &v in f {
            p.corners
                .push([v, normal.at(0, v, face), color.at(0, v, face), uv.at(0, v, face)]);
        }
        p.end_face();
    }
}

/// Triangles from strips or fans of position indices, each strip's alternate triangles turned
/// to keep one winding. A triangle on one index twice restarts the strip, and draws nothing.
fn strips(runs: &[Vec<u32>], fan: bool) -> Vec<u32> {
    let mut out = Vec::new();
    for r in runs {
        for k in 2..r.len() {
            let t = if fan {
                [r[0], r[k - 1], r[k]]
            } else if k % 2 == 0 {
                [r[k - 2], r[k - 1], r[k]]
            } else {
                [r[k - 1], r[k - 2], r[k]]
            };
            if crate::geometry::distinct(t) {
                out.extend(t);
            }
        }
    }
    out
}

/// An `ElevationGrid`: heights over a grid in X and Z, a quad between each four.
fn elevation_grid(g: &Graph, id: usize) -> Option<Polys> {
    let nx = idx(g.num(id, "xDimension", 0.0)) as usize;
    let nz = idx(g.num(id, "zDimension", 0.0)) as usize;
    let heights = g.nums(id, "height");
    if nx < 2 || nz < 2 || nx.checked_mul(nz)? > heights.len() {
        return None;
    }
    let (sx, sz) = (f32_of(g.num(id, "xSpacing", 1.0)), f32_of(g.num(id, "zSpacing", 1.0)));
    let mut points = Vec::with_capacity(nx * nz);
    for z in 0..nz {
        for x in 0..nx {
            points.push([count(x) * sx, f32_of(heights[z * nx + x]), count(z) * sz]);
        }
    }
    let mut p = Polys::new(g, id, points);
    let textured = !p.uvs.is_empty();
    if !textured {
        let (dx, dz) = (count(nx - 1), count(nz - 1));
        p.uvs = (0..nx * nz).map(|i| [count(i % nx) / dx, count(i / nx) / dz]).collect();
    }
    let crease = g.num(id, "creaseAngle", 0.0);
    let normal = Lookup {
        present: !p.normals.is_empty(),
        per_vertex: g.flag(id, "normalPerVertex", true),
        index: &[],
    };
    let color = Lookup {
        present: !p.colors.is_empty(),
        per_vertex: g.flag(id, "colorPerVertex", true),
        index: &[],
    };
    let uv = Lookup {
        present: true,
        per_vertex: true,
        index: &[],
    };
    p.by_position = normal.by_position() && color.by_position();
    p.shading = shading(normal.present, crease);
    let at = |x: usize, z: usize| u32_of(z * nx + x);
    for z in 0..nz - 1 {
        for x in 0..nx - 1 {
            let face = z * (nx - 1) + x;
            for v in [at(x, z), at(x, z + 1), at(x + 1, z + 1), at(x + 1, z)] {
                p.corners
                    .push([v, normal.at(0, v, face), color.at(0, v, face), uv.at(0, v, face)]);
            }
            p.end_face();
        }
    }
    Some(p)
}

/// The shading of a set with a crease angle: from its normals if it has them, else flat for
/// no angle, smooth for one past a half turn, and creased between.
fn shading(normals: bool, crease: f64) -> Shading {
    if normals {
        Shading::Given
    } else if crease <= 0.0 || crease.is_nan() {
        Shading::Flat
    } else if crease >= f64::from(PI) {
        Shading::Smooth
    } else {
        Shading::Crease(f32_of(crease.cos()))
    }
}

/// A field of numbers in groups of `N`, or `default` when it's absent.
fn groups<const N: usize>(g: &Graph, id: usize, name: &str, default: &[[f64; N]]) -> Vec<[f64; N]> {
    match g.field(id, name) {
        None => default.to_vec(),
        Some(_) => g.nums(id, name).as_chunks::<N>().0.to_vec(),
    }
}

/// Whether two points are the very same, as a closed spine or cross-section's ends are.
#[allow(clippy::float_cmp, reason = "the same point, exactly")]
fn same<const N: usize>(a: [f64; N], b: [f64; N]) -> bool {
    a == b
}

/// The axes (X, Y, Z) of the spine-aligned cross-section plane at each spine point, as the
/// specification computes them: Y along the spine at the point, Z square to the spine's bend
/// there (the last bend's at a straight stretch, the first's before any), turned to keep with
/// the last, and X = Y × Z. A straight spine turns the plane from +Y to its direction; points
/// that coincide share a plane, and a spine that ends where it starts is closed.
fn spine_planes(spine: &[[f64; 3]]) -> Vec<[[f64; 3]; 3]> {
    const STILL: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut of = Vec::with_capacity(spine.len());
    for &s in spine {
        if points.last() != Some(&s) {
            points.push(s);
        }
        of.push(points.len().saturating_sub(1));
    }
    let n = points.len();
    if n < 2 {
        return vec![STILL; spine.len()];
    }
    let closed = n > 2 && same(points[0], points[n - 1]);
    let mut ys = Vec::with_capacity(n);
    let mut zs: Vec<Option<[f64; 3]>> = Vec::with_capacity(n);
    for i in 0..n {
        let here = points[i];
        let (prev, next) = if closed && (i == 0 || i == n - 1) {
            (Some(points[n - 2]), Some(points[1]))
        } else {
            (i.checked_sub(1).map(|k| points[k]), points.get(i + 1).copied())
        };
        let y = match (prev, next) {
            (Some(p), Some(q)) => vec3::unit(sub(q, p)),
            _ => None,
        };
        let step = next.map_or_else(|| sub(here, prev.unwrap_or(here)), |q| sub(q, here));
        ys.push(y.or_else(|| vec3::unit(step)).unwrap_or([0.0, 1.0, 0.0]));
        zs.push(match (prev, next) {
            (Some(p), Some(q)) => {
                let (a, b) = (sub(q, here), sub(p, here));
                let c = cross(a, b);
                // Not where the spine runs straight on.
                if length(c) > 1e-9 * length(a) * length(b) {
                    vec3::unit(c)
                } else {
                    None
                }
            }
            _ => None,
        });
    }
    let planes: Vec<[[f64; 3]; 3]> = if let Some(first) = zs.iter().position(Option::is_some) {
        let mut z = zs[first].unwrap_or([0.0, 0.0, 1.0]);
        let mut out = Vec::with_capacity(n);
        for (i, (&y, zi)) in ys.iter().zip(&zs).enumerate() {
            if i > first
                && let Some(zi) = *zi
            {
                z = if dot(zi, z) < 0.0 { zi.map(|c| -c) } else { zi };
            }
            // Square to Y, as the specification's rules make it but at a straight stretch.
            let zy = vec3::unit(sub(z, y.map(|c| c * dot(z, y)))).unwrap_or(z);
            out.push([cross(y, zy), y, zy]);
        }
        out
    } else {
        // A straight spine: the turn from +Y to its direction.
        let d = vec3::unit(sub(points[1], points[0])).unwrap_or([0.0, 1.0, 0.0]);
        let q = if d[1] < -1.0 + 1e-12 {
            quat([1.0, 0.0, 0.0, std::f64::consts::PI])
        } else {
            quat([d[2], 0.0, -d[0], d[1].clamp(-1.0, 1.0).acos()])
        };
        vec![STILL.map(|a| rotate(q, a)); n]
    };
    of.iter().map(|&k| planes[k]).collect()
}

/// An `Extrusion`: its cross-section scaled, turned and set in the spine-aligned plane at each
/// spine point, a quad between each two copies, and caps at the ends as asked (none on a
/// closed spine, where they'd meet inside). Textured, the texture runs around the
/// cross-section and along the spine, by length, and over each cap by the cross-section's
/// bounds.
fn extrusion(g: &Graph, id: usize, textured: bool, limit: usize) -> Result<Polys> {
    let section = groups(
        g,
        id,
        "crossSection",
        &[[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
    );
    let spine = groups(g, id, "spine", &[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let scale = groups(g, id, "scale", &[[1.0, 1.0]]);
    let orientation = groups(g, id, "orientation", &[[0.0, 0.0, 1.0, 0.0]]);
    let mut p = Polys::new(g, id, Vec::new());
    p.by_position = false;
    p.shading = shading(false, g.num(id, "creaseAngle", 0.0));
    let (ns, nc) = (spine.len(), section.len());
    if ns < 2 || nc < 2 {
        return Ok(p);
    }
    let spine_closed = ns > 2 && same(spine[0], spine[ns - 1]);
    let section_closed = nc > 2 && same(section[0], section[nc - 1]);
    let cap_points = if section_closed { nc - 1 } else { nc };
    let caps = !spine_closed && cap_points >= 3;
    let begin = caps && g.flag(id, "beginCap", true);
    let end = caps && g.flag(id, "endCap", true);
    let tris = (ns - 1)
        .checked_mul(nc - 1)
        .and_then(|q| q.checked_mul(2))
        .and_then(|t| t.checked_add((usize::from(begin) + usize::from(end)) * cap_points));
    ensure!(
        tris.is_some_and(|t| t <= limit),
        LimitExceeded,
        "too many triangles to convert (more than {limit})"
    );
    let planes = spine_planes(&spine);
    p.points.reserve(ns * nc);
    for (i, (&s, [x, y, z])) in spine.iter().zip(&planes).enumerate() {
        let [sx, sz] = scale.get(i).or(scale.last()).copied().unwrap_or([1.0; 2]);
        let turn = quat(
            orientation
                .get(i)
                .or(orientation.last())
                .copied()
                .unwrap_or([0.0, 0.0, 1.0, 0.0]),
        );
        for &[cx, cz] in &section {
            let v = rotate(turn, [cx * sx, 0.0, cz * sz]);
            p.points
                .push([0, 1, 2].map(|k| f32_of(s[k] + x[k] * v[0] + y[k] * v[1] + z[k] * v[2])));
        }
    }
    // A closed spine whose ends meet, and a closed cross-section, share their seam's points
    // (so smoothing reaches across it).
    let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() <= 1e-6 * (1.0 + a[k].abs()));
    let rows_meet = spine_closed && (0..nc).all(|j| near(p.points[j], p.points[(ns - 1) * nc + j]));
    let at = |i: usize, j: usize| {
        let i = if rows_meet && i == ns - 1 { 0 } else { i };
        let j = if section_closed && j == nc - 1 { 0 } else { j };
        u32_of(i * nc + j)
    };
    // Texture coordinates: the sides' by length, then the caps'.
    let along = |lengths: &mut dyn Iterator<Item = f64>, n: usize| {
        let mut run = vec![0.0];
        for l in lengths {
            run.push(run.last().copied().unwrap_or(0.0) + if l.is_finite() { l } else { 0.0 });
        }
        let total = run.last().copied().unwrap_or(0.0);
        let steps = f64::from(u32_of(n - 1));
        run.iter()
            .enumerate()
            .map(|(k, r)| {
                if total > 0.0 {
                    r / total
                } else {
                    f64::from(u32_of(k)) / steps
                }
            })
            .collect::<Vec<f64>>()
    };
    let su = along(
        &mut section.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])),
        nc,
    );
    let tv = along(&mut spine.windows(2).map(|w| length(sub(w[1], w[0]))), ns);
    p.uvs.reserve(ns * nc + nc);
    for &t in &tv {
        p.uvs.extend(su.iter().map(|&s| [f32_of(s), f32_of(t)]));
    }
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for c in section.iter().filter(|c| c.iter().all(|v| v.is_finite())) {
        for k in 0..2 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    let size = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let size = if size > 0.0 { size } else { 1.0 };
    p.uvs.extend(
        section
            .iter()
            .map(|c| [f32_of((c[0] - lo[0]) / size), f32_of((c[1] - lo[1]) / size)]),
    );
    let uv = |n: usize| if textured { u32_of(n) } else { NONE };
    if !textured {
        p.uvs.clear();
    }
    p.corners.reserve(tris.unwrap_or(0) * 2);
    for i in 0..ns - 1 {
        for j in 0..nc - 1 {
            for (a, b) in [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)] {
                p.corners.push([at(a, b), NONE, NONE, uv(a * nc + b)]);
            }
            p.end_face();
        }
    }
    if begin {
        for j in (0..cap_points).rev() {
            p.corners.push([at(0, j), NONE, NONE, uv(ns * nc + j)]);
        }
        p.end_face();
    }
    if end {
        for j in 0..cap_points {
            p.corners.push([at(ns - 1, j), NONE, NONE, uv(ns * nc + j)]);
        }
        p.end_face();
    }
    Ok(p)
}

/// A start angle and the turn counterclockwise from it to an end angle: a whole turn for equal
/// angles or angles a turn or more apart.
fn arc(g: &Graph, id: usize) -> (f64, f64) {
    use std::f64::consts::{FRAC_PI_2, TAU};
    let start = g.num(id, "startAngle", 0.0);
    let end = g.num(id, "endAngle", FRAC_PI_2);
    let span = end - start;
    if span.abs() < 1e-9 || span.abs() >= TAU {
        (start, TAU)
    } else {
        (start, span.rem_euclid(TAU))
    }
}

/// The points of an arc of radius `r`: as many segments as its share of a whole turn of
/// [`AROUND`], at least two; a whole turn ends where it starts.
fn arc_points(r: f64, (start, span): (f64, f64)) -> Vec<[f32; 2]> {
    let n = (f64::from(AROUND) * span / std::f64::consts::TAU)
        .ceil()
        .clamp(2.0, f64::from(AROUND));
    let steps = 0..=idx(n);
    steps
        .map(|k| {
            let a = start + span * f64::from(k) / n;
            [f32_of(r * a.cos()), f32_of(r * a.sin())]
        })
        .collect()
}

/// A 2D node in the Z = 0 plane: the filled ones facing +Z, mapping the texture over their
/// bounds; the others as lines or points.
fn flat(g: &Graph, id: usize, limit: usize) -> Result<Prim> {
    let mut m = Mesher::default();
    let up = [0.0, 0.0, 1.0];
    // A fill: a fan of points around the first, each mapped by the square of side `size`
    // around the origin.
    let fan = |m: &mut Mesher, points: &[[f32; 2]], size: f32| {
        let size = if size > 0.0 { size } else { 1.0 };
        let uv = |q: [f32; 2]| [q[0] / size + 0.5, q[1] / size + 0.5];
        let v: Vec<u32> = points.iter().map(|&q| m.vert([q[0], q[1], 0.0], up, uv(q))).collect();
        for k in 1..v.len().saturating_sub(1) {
            m.tri(v[0], v[k], v[k + 1]);
        }
    };
    // Lines through points, back to the first if `closed`.
    let lines = |m: &mut Mesher, points: &[[f32; 2]]| {
        let mut last: Option<u32> = None;
        for q in points {
            let at = q.iter().all(|c| c.is_finite()).then(|| {
                m.prim.positions.push([q[0], q[1], 0.0]);
                u32_of(m.prim.positions.len() - 1)
            });
            if let (Some(a), Some(b)) = (last, at) {
                m.prim.lines.extend([a, b]);
            }
            last = at;
        }
    };
    let radius = |name: &str, default: f64| g.num(id, name, default).abs();
    match g.kind(id) {
        "Rectangle2D" => {
            let [w, h] = g.vec(id, "size", [2.0, 2.0]).map(|s| f32_of(s.abs() / 2.0));
            let a = m.vert([-w, -h, 0.0], up, [0.0, 0.0]);
            let b = m.vert([w, -h, 0.0], up, [1.0, 0.0]);
            let c = m.vert([w, h, 0.0], up, [1.0, 1.0]);
            let d = m.vert([-w, h, 0.0], up, [0.0, 1.0]);
            m.quad(a, b, c, d);
        }
        "Disk2D" => {
            let (a, b) = (radius("innerRadius", 0.0), radius("outerRadius", 1.0));
            let (inner, outer) = (a.min(b), a.max(b));
            let ring = |r: f64| arc_points(r, (0.0, std::f64::consts::TAU));
            if outer == 0.0 {
                // Nothing to draw.
            } else if outer - inner <= 1e-12 * outer {
                // A circle.
                lines(&mut m, &ring(outer));
            } else if inner == 0.0 {
                let mut points = vec![[0.0; 2]];
                points.extend(ring(outer));
                fan(&mut m, &points, f32_of(outer * 2.0));
            } else {
                let size = f32_of(outer * 2.0);
                let uv = |q: [f32; 2]| [q[0] / size + 0.5, q[1] / size + 0.5];
                let rows: Vec<Vec<u32>> = [inner, outer]
                    .iter()
                    .map(|&r| {
                        ring(r)
                            .into_iter()
                            .map(|q| m.vert([q[0], q[1], 0.0], up, uv(q)))
                            .collect()
                    })
                    .collect();
                for k in 0..rows[0].len() - 1 {
                    m.quad(rows[0][k], rows[1][k], rows[1][k + 1], rows[0][k + 1]);
                }
            }
        }
        "ArcClose2D" => {
            let r = radius("radius", 1.0);
            let pie = !g
                .strs(id, "closureType")
                .first()
                .is_some_and(|c| c.trim().eq_ignore_ascii_case("CHORD"));
            let (start, span) = arc(g, id);
            let mut points = if pie { vec![[0.0; 2]] } else { Vec::new() };
            points.extend(arc_points(r, (start, span)));
            if !pie && span >= std::f64::consts::TAU {
                // A whole turn's chord: its end is its start.
                points.pop();
            }
            fan(&mut m, &points, f32_of(r * 2.0));
        }
        "TriangleSet2D" => {
            let vertices: Vec<[f32; 2]> = g
                .nums(id, "vertices")
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| v.map(f32_of))
                .collect();
            let tris = vertices.as_chunks::<3>().0;
            ensure!(
                tris.len() <= limit,
                LimitExceeded,
                "too many triangles to convert (more than {limit})"
            );
            let finite = |q: &[f32; 2]| q.iter().all(|c| c.is_finite());
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for q in vertices.iter().filter(|q| finite(q)) {
                for k in 0..2 {
                    lo[k] = lo[k].min(q[k]);
                    hi[k] = hi[k].max(q[k]);
                }
            }
            let size = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            let size = if size > 0.0 { size } else { 1.0 };
            for t in tris.iter().filter(|t| t.iter().all(finite)) {
                let [a, b, c] =
                    t.map(|q| m.vert([q[0], q[1], 0.0], up, [(q[0] - lo[0]) / size, (q[1] - lo[1]) / size]));
                m.tri(a, b, c);
            }
        }
        "Circle2D" => lines(&mut m, &arc_points(radius("radius", 1.0), (0.0, std::f64::consts::TAU))),
        "Arc2D" => lines(&mut m, &arc_points(radius("radius", 1.0), arc(g, id))),
        "Polyline2D" => {
            let points: Vec<[f32; 2]> = g
                .nums(id, "lineSegments")
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| v.map(f32_of))
                .collect();
            lines(&mut m, &points);
        }
        _ => {
            for q in g.nums(id, "point").as_chunks::<2>().0 {
                let q = q.map(f32_of);
                if q.iter().all(|c| c.is_finite()) {
                    m.prim.positions.push([q[0], q[1], 0.0]);
                    m.prim.points.push(u32_of(m.prim.positions.len() - 1));
                }
            }
        }
    }
    Ok(m.prim)
}

/// A count as a coordinate.
fn count(n: usize) -> f32 {
    f32_of(f64::from(u32_of(n)))
}

/// The primitive for polygons: each triangulated in its plane, the corners made vertices (one
/// per distinct set of indices), with normals as the shading says. A face with a corner that
/// names no point, or one that isn't finite, is left out and counted in `dropped`.
fn to_prim(p: &Polys, textured: bool, limit: usize, dropped: &mut usize) -> Result<Prim> {
    let n_points = p.points.len();
    let finite: Vec<bool> = p.points.iter().map(|q| q.iter().all(|x| x.is_finite())).collect();
    // Triangles as corner numbers.
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    // The faces drawn, as spans of corners, and each one's normal: what flat and creased shading go by.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut face_normals: Vec<[f32; 3]> = Vec::new();
    // The cosine of the crease angle, or none for flat faces; `None` when the normals aren't each corner's own.
    let cos = match p.shading {
        Shading::Crease(cos) => Some(Some(cos)),
        Shading::Flat => Some(None),
        _ => None,
    };
    let need_faces = cos.is_some();
    for w in p.starts.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        let face = &p.corners[a..b];
        if face.len() < 3 {
            continue;
        }
        if !face.iter().all(|c| (c[0] as usize) < n_points && finite[c[0] as usize]) {
            *dropped += 1;
            continue;
        }
        let before = tris.len();
        if face.len() == 3 {
            tris.push([0, 1, 2].map(|k| u32_of(a + k)));
        } else {
            pts.clear();
            pts.extend(face.iter().map(|c| p.points[c[0] as usize]));
            local.clear();
            triangulate(&pts, &mut local);
            tris.extend(local.iter().map(|t| t.map(|k| u32_of(a) + k)));
        }
        if !p.ccw {
            for t in &mut tris[before..] {
                t.swap(1, 2);
            }
        }
        if need_faces {
            let mut nrm = newell(face.iter().map(|c| p.points[c[0] as usize]));
            if !p.ccw {
                nrm = nrm.map(|x| -x);
            }
            spans.push((a, face.len()));
            face_normals.push(nrm);
        }
        ensure!(
            tris.len() <= limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
    }
    let corner_pos = |c: u32| p.corners[c as usize][0];
    let smooth = if p.shading == Shading::Smooth || p.shading == Shading::Given {
        let flat: Vec<u32> = tris.iter().flatten().map(|&c| corner_pos(c)).collect();
        smooth_normals(&p.points, &flat)
    } else {
        Vec::new()
    };
    // Each corner's own normal: its face's, or the faces' at its point near enough its own face.
    let crease = match cos {
        Some(cos) => {
            let at: Vec<u32> = p.corners.iter().map(|c| c[0]).collect();
            corner_normals(n_points, &spans, &at, &face_normals, |_| cos, |_, _| true)
        }
        None => Vec::new(),
    };
    let default_uvs = if textured && p.uvs.is_empty() {
        box_mapping(&p.points)
    } else {
        Vec::new()
    };
    let mut out = Prim::default();
    let has_colors = !p.colors.is_empty();
    let has_uvs = !p.uvs.is_empty() || !default_uvs.is_empty();
    let push = |out: &mut Prim, c: [u32; 4], crease_n: Option<[f32; 3]>| {
        let v = c[0] as usize;
        out.positions.push(p.points[v]);
        out.normals.push(match p.shading {
            // One with no length stays zero, and takes its faces' as the primitive is written.
            Shading::Given => p
                .normals
                .get(c[1] as usize)
                .map_or(smooth[v], |&n| vec3::unit_above(n, 0.0).unwrap_or([0.0; 3])),
            Shading::Crease(_) | Shading::Flat => crease_n.unwrap_or(crate::geometry::UP),
            Shading::Smooth => smooth[v],
        });
        if has_colors {
            out.colors
                .push(p.colors.get(c[2] as usize).copied().unwrap_or([1.0; 4]));
        }
        if has_uvs {
            let uv = if default_uvs.is_empty() {
                p.uvs.get(c[3] as usize).copied().unwrap_or([0.0; 2])
            } else {
                default_uvs[v]
            };
            out.uvs.push(uv);
        }
        u32_of(out.positions.len() - 1)
    };
    out.indices.reserve(tris.len() * 3);
    if p.by_position && crease.is_empty() {
        let mut remap = vec![NONE; n_points];
        for &c in tris.iter().flatten() {
            let corner = p.corners[c as usize];
            let r = &mut remap[corner[0] as usize];
            if *r == NONE {
                *r = push(&mut out, corner, None);
            }
            out.indices.push(*r);
        }
    } else {
        let mut seen: FastMap<[u32; 6], u32> = FastMap::default();
        for &c in tris.iter().flatten() {
            let corner = p.corners[c as usize];
            let n = crease.get(c as usize).copied();
            let bits = n.map_or([0; 3], |n| n.map(f32::to_bits));
            // A crease normal stands in for the normal index.
            let key = if n.is_some() {
                [corner[0], bits[0], bits[1], bits[2], corner[2], corner[3]]
            } else {
                [corner[0], corner[1], corner[2], corner[3], 0, 0]
            };
            let r = if let Some(&r) = seen.get(&key) {
                r
            } else {
                let r = push(&mut out, corner, n);
                seen.insert(key, r);
                r
            };
            out.indices.push(r);
        }
    }
    Ok(out)
}

/// A polygon's unit normal by Newell's method.
fn newell(points: impl Iterator<Item = [f32; 3]>) -> [f32; 3] {
    unit(vec3::newell(points))
}

/// `n` at unit length; up when it has no length or isn't finite.
fn unit(n: [f32; 3]) -> [f32; 3] {
    vec3::unit_above(n, 0.0).unwrap_or([0.0, 1.0, 0.0])
}

/// X3D's default texture coordinates for a face set: the bounding box's longest side is `s`,
/// its second longest `t`, both from its minimum, scaled by the longest.
fn box_mapping(points: &[[f32; 3]]) -> Vec<[f32; 2]> {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for q in points.iter().filter(|q| q.iter().all(|x| x.is_finite())) {
        for k in 0..3 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    let size = [0, 1, 2].map(|k| (hi[k] - lo[k]).max(0.0));
    let mut axes = [0usize, 1, 2];
    axes.sort_by(|&a, &b| size[b].total_cmp(&size[a]));
    let (s, t) = (axes[0], axes[1]);
    let scale = if size[s] > 0.0 { size[s] } else { 1.0 };
    points
        .iter()
        .map(|q| [(q[s] - lo[s]) / scale, (q[t] - lo[t]) / scale])
        .collect()
}

/// Builds a primitive from vertices with normals and texture coordinates.
#[derive(Default)]
struct Mesher {
    prim: Prim,
}

impl Mesher {
    fn vert(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        self.prim.positions.push(p);
        self.prim.normals.push(n);
        self.prim.uvs.push(uv);
        u32_of(self.prim.positions.len() - 1)
    }

    /// A triangle, turned to face the way its vertices' normals do.
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        let pos = |i: u32| self.prim.positions[i as usize];
        let f = face_normal(pos(a), pos(b), pos(c));
        let n = [a, b, c].map(|i| self.prim.normals[i as usize]);
        let dot: f32 = (0..3).map(|k| f[k] * (n[0][k] + n[1][k] + n[2][k])).sum();
        if dot < 0.0 {
            self.prim.indices.extend([a, c, b]);
        } else {
            self.prim.indices.extend([a, b, c]);
        }
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }
}

/// A box centered on the origin, each face mapping the whole texture upright.
fn box_prim(size: [f32; 3]) -> Prim {
    let h = size.map(|s| s.abs() / 2.0);
    let mut m = Mesher::default();
    // Each face: its normal, and its right and up directions as seen from outside.
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    for (n, r, u) in faces {
        let corner = |s: f32, t: f32| [0, 1, 2].map(|k| (n[k] + r[k] * s + u[k] * t) * h[k]);
        let a = m.vert(corner(-1.0, -1.0), n, [0.0, 0.0]);
        let b = m.vert(corner(1.0, -1.0), n, [1.0, 0.0]);
        let c = m.vert(corner(1.0, 1.0), n, [1.0, 1.0]);
        let d = m.vert(corner(-1.0, 1.0), n, [0.0, 1.0]);
        m.quad(a, b, c, d);
    }
    m.prim
}

/// Segments around a sphere, cylinder or cone.
const AROUND: u32 = 32;
/// Rings from pole to pole of a sphere.
const RINGS: u32 = 16;

/// The point at angle `u` (0-1) around the Y axis, starting from -Z and turning
/// counterclockwise seen from above, as X3D wraps textures.
fn around(u: f32) -> [f32; 2] {
    let a = u * TAU;
    [-a.sin(), -a.cos()]
}

fn sphere(r: f32) -> Prim {
    let r = r.abs();
    let mut m = Mesher::default();
    let mut rows = Vec::new();
    for j in 0..=RINGS {
        let v = f32_of(f64::from(j)) / f32_of(f64::from(RINGS));
        let phi = v * PI;
        let (y, ring) = (-phi.cos(), phi.sin());
        let row: Vec<u32> = (0..=AROUND)
            .map(|i| {
                let u = f32_of(f64::from(i)) / f32_of(f64::from(AROUND));
                let [x, z] = around(u);
                let n = [x * ring, y, z * ring];
                m.vert(n.map(|c| c * r), n, [u, v])
            })
            .collect();
        rows.push(row);
    }
    for j in 0..RINGS as usize {
        for i in 0..AROUND as usize {
            let (a, b, c, d) = (rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]);
            if j == 0 {
                m.tri(a, c, d);
            } else if j + 1 == RINGS as usize {
                m.tri(a, b, c);
            } else {
                m.quad(a, b, c, d);
            }
        }
    }
    m.prim
}

/// A cylinder or cone (radius `top` at the top) of height `h` centered on the origin: its
/// side, top and bottom as `parts` asks.
fn round(bottom: f32, top: f32, h: f32, parts: [bool; 3]) -> Prim {
    let (bottom, top, y) = (bottom.abs(), top.abs(), h.abs() / 2.0);
    let mut m = Mesher::default();
    let steps = || (0..=AROUND).map(|i| f32_of(f64::from(i)) / f32_of(f64::from(AROUND)));
    if parts[0] {
        // The side's normal leans by the slope.
        let slope = (bottom - top) / (2.0 * y).max(f32::MIN_POSITIVE);
        let lean = 1.0 / (1.0 + slope * slope).sqrt();
        let mut prev: Option<(u32, u32)> = None;
        for u in steps() {
            let [x, z] = around(u);
            let n = [x * lean, slope * lean, z * lean];
            let lo = m.vert([x * bottom, -y, z * bottom], n, [u, 0.0]);
            let hi = m.vert([x * top, y, z * top], n, [u, 1.0]);
            if let Some((plo, phi)) = prev {
                if top > 0.0 {
                    m.quad(plo, lo, hi, phi);
                } else {
                    m.tri(plo, lo, hi);
                }
            }
            prev = Some((lo, hi));
        }
    }
    for (cap, at, radius, ny) in [(parts[1], y, top, 1.0f32), (parts[2], -y, bottom, -1.0)] {
        if !cap || radius <= 0.0 {
            continue;
        }
        let n = [0.0, ny, 0.0];
        let center = m.vert([0.0, at, 0.0], n, [0.5, 0.5]);
        let ring: Vec<u32> = steps()
            .map(|u| {
                let [x, z] = around(u);
                // Caps map the texture as seen from outside, upright toward -Z on top.
                m.vert([x * radius, at, z * radius], n, [0.5 + x / 2.0, 0.5 - ny * z / 2.0])
            })
            .collect();
        for w in ring.windows(2) {
            m.tri(center, w[0], w[1]);
        }
    }
    m.prim
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outward(p: &Prim) -> bool {
        p.indices.as_chunks::<3>().0.iter().all(|t| {
            let [a, b, c] = t.map(|i| p.positions[i as usize]);
            let n = face_normal(a, b, c);
            let mid = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0);
            n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] >= -1e-6
        })
    }

    #[test]
    fn primitives_face_outward() {
        assert!(outward(&box_prim([2.0, 1.0, 3.0])));
        assert!(outward(&sphere(1.5)));
        assert!(outward(&round(1.0, 1.0, 2.0, [true; 3])));
        assert!(outward(&round(1.0, 0.0, 2.0, [true, false, true])));
        assert_eq!(box_prim([2.0; 3]).indices.len(), 36);
    }

    #[test]
    fn strips_and_fans_keep_one_winding() {
        assert_eq!(strips(&[vec![0, 1, 2, 3]], false), [0, 1, 2, 2, 1, 3]);
        assert_eq!(strips(&[vec![0, 1, 2, 3]], true), [0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn sphere_texture_starts_at_the_back() {
        let s = sphere(1.0);
        // u = 0 at -Z, u = 0.25 at -X.
        let at = |u: f32| {
            let i = s
                .uvs
                .iter()
                .position(|t| (t[0] - u).abs() < 1e-6 && (t[1] - 0.5).abs() < 1e-6)
                .unwrap();
            s.positions[i]
        };
        assert!(at(0.0)[2] < -0.99);
        assert!(at(0.25)[0] < -0.99);
    }
}
