//! What every reader's geometry is held to, so a document is the same kind of thing whatever format it came
//! from. The readers that build documents ([`crate::document::Builder`] and the scene writer) apply both rules as
//! they write a primitive; a reader that writes its accessors itself calls them.
//!
//! **What can't be drawn is dropped, and counted.** A triangle, line segment or point that names a vertex that
//! isn't there, or one whose position isn't finite, is left out ([`sound`]): never moved to the origin, and never
//! an error for the whole file. Each is counted in `asset.extras.source.droppedGeometry` ([`DROPPED`]), with the
//! faces a reader left out itself for the same reasons. A vertex that isn't a finite point, which nothing draws
//! any more, reads as the origin, so no position in a document is NaN or infinite. Triangles with no area or a
//! repeated corner are kept as the file has them: they hold a mesh's topology together.
//!
//! **Every triangle primitive has unit normals.** A file's own normals are used, made unit length ([`normals`]);
//! one the file gives as zero or not finite takes the sum of the normals of the triangles that share its vertex,
//! weighted by their areas. Where the file has none, a reader first follows what its format says about smoothing
//! (smoothing groups, a crease angle, a default of flat shading). Otherwise the one default applies
//! ([`default_normals`]): a crease angle, 30 degrees unless the caller sets another
//! ([`Keep::crease_angle`](crate::Keep::crease_angle)). The angle is handed to each builder as it is made, from
//! the `Keep` its reader was given, so it holds on whatever thread a reader does its work.
//! Each corner takes the normals of the triangles around its point that turn less than that angle from its
//! own, summed by their areas, and a vertex is split where its corners' normals differ. The triangles around a
//! point are those of the primitive with a corner at that position, whether or not they share the vertex, so a
//! vertex the file (or its reader) split for its texture coordinates shows no seam. So a box comes out with hard
//! edges and a finely cut curved surface smooth, whatever format either was stored in. A vertex with no
//! triangle of any area around it points up, `[0, 1, 0]`. Points and lines get no normals.

/// The key in `asset.extras.source` that counts what was dropped for naming a missing vertex or a position that
/// isn't finite: triangles, line segments and points, and each polygon a reader left out whole, once.
pub(crate) const DROPPED: &str = "droppedGeometry";

/// The normal of a vertex with nothing to take one from.
pub(crate) const UP: [f32; 3] = [0.0, 1.0, 0.0];

/// Whether a triangle made by unrolling a strip or a fan is drawn: one that names a vertex twice isn't, as that
/// is how a strip is restarted (and such a triangle has no area). This is for the triangles a reader makes
/// itself; a triangle the file lists with a repeated corner is kept.
#[allow(dead_code, reason = "which readers unroll strips depends on the enabled features")]
pub(crate) fn distinct(t: [u32; 3]) -> bool {
    t[0] != t[1] && t[1] != t[2] && t[0] != t[2]
}

/// Whether every position is finite: one pass over the bits (an exponent of all ones is infinite or NaN).
pub(crate) fn finite(v: &[[f32; 3]]) -> bool {
    let worst = v.as_flattened().iter().fold(0, |m, x| m.max(x.to_bits() & 0x7fff_ffff));
    worst < 0x7f80_0000
}

/// Drops from `list`, groups of `width` indices (3 for triangles, 2 for segments, 1 for points), each group
/// naming a vertex past `positions` or one whose position isn't finite; an incomplete last group goes too.
/// `finite` is whether every position is ([`finite`]). Positions that aren't, which nothing draws once every list
/// on them has been through here, are then set to the origin by [`settle`]. Returns how many groups were dropped.
pub(crate) fn sound(positions: &[[f32; 3]], finite: bool, list: &mut Vec<u32>, width: usize) -> usize {
    let n = positions.len();
    let width = width.max(1);
    list.truncate(list.len() - list.len() % width);
    let top = list.iter().fold(0, |m, &i| m.max(i));
    if list.is_empty() || (finite && (top as usize) < n) {
        return 0;
    }
    let ok = |i: u32| {
        positions
            .get(i as usize)
            .is_some_and(|p| finite || p.iter().all(|c| c.is_finite()))
    };
    let before = list.len() / width;
    let mut kept = 0;
    for g in 0..before {
        let at = g * width;
        if list[at..at + width].iter().all(|&i| ok(i)) {
            list.copy_within(at..at + width, kept);
            kept += width;
        }
    }
    list.truncate(kept);
    before - kept / width
}

/// Sets every position that isn't finite to the origin, once [`sound`] has dropped what drew them.
pub(crate) fn settle(positions: &mut [[f32; 3]]) {
    for p in positions {
        if !p.iter().all(|c| c.is_finite()) {
            *p = [0.0; 3];
        }
    }
}

/// Each vertex's sum of the normals of the triangles around it, each as long as twice its triangle's area (in a
/// scale of its own when the model is very large or very small, so nothing overflows): the direction a smooth
/// normal takes. Zero where a vertex has no triangle with any area.
fn sums(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sum = vec![[0.0f32; 3]; positions.len()];
    let extent = positions.as_flattened().iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let scale = if (1e-12..=1e12).contains(&extent) || extent == 0.0 || !extent.is_finite() {
        1.0
    } else {
        1.0 / extent
    };
    for t in indices.as_chunks::<3>().0 {
        let [i, j, k] = t.map(|i| i as usize);
        let (Some(a), Some(b), Some(c)) = (positions.get(i), positions.get(j), positions.get(k)) else {
            continue;
        };
        let u = [(b[0] - a[0]) * scale, (b[1] - a[1]) * scale, (b[2] - a[2]) * scale];
        let v = [(c[0] - a[0]) * scale, (c[1] - a[1]) * scale, (c[2] - a[2]) * scale];
        let f = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        if !(f[0].is_finite() && f[1].is_finite() && f[2].is_finite()) {
            continue;
        }
        for v in [i, j, k] {
            let s = &mut sum[v];
            *s = [s[0] + f[0], s[1] + f[1], s[2] + f[2]];
        }
    }
    sum
}

/// Makes the normals of points or lines unit length, as [`normals`] does a triangle's; `false`, with the list
/// part done, when one has no length or isn't finite, as there are no faces to take one from.
pub(crate) fn unit(normals: &mut [[f32; 3]], slack: f32) -> bool {
    for v in normals {
        let sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if (sq - 1.0).abs() <= slack {
            continue;
        }
        let l = sq.sqrt();
        if !(l > 0.0 && l.is_finite()) {
            return false;
        }
        *v = v.map(|c| c / l);
    }
    true
}

/// `v` at unit length, or [`UP`] when it has no length or isn't finite.
fn unit_or_up(v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v.map(f64::from);
    let l = (x * x + y * y + z * z).sqrt();
    if l > 0.0 && l.is_finite() {
        #[allow(clippy::cast_possible_truncation, reason = "a unit vector's components fit")]
        let out = [(x / l) as f32, (y / l) as f32, (z / l) as f32];
        out
    } else {
        UP
    }
}

/// The default normals for triangles `indices` over `positions`, one per vertex: smooth, weighted by area (see
/// the module's notes).
pub(crate) fn smooth(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut out = sums(positions, indices);
    for n in &mut out {
        *n = unit_or_up(*n);
    }
    out
}

/// The default normals of triangles `indices` over `positions`, for a file that has none and says nothing of
/// its own about smoothing (see the module's notes), at a crease angle of `degrees`, the caller's
/// ([`Keep::crease_angle`](crate::Keep::crease_angle)): one for every vertex, and the vertices added where a
/// vertex's corners take different normals, each named by the vertex it is a copy of. The copies come after the
/// vertices there were, in order, and `indices` is changed to name them.
///
/// The work and the room grow with the model, not faster: a pass over the vertices to find those at one
/// position (a hash map keyed by each position's bits), one over the triangles for their normals, a list of the
/// triangles around each position, and for each corner a look at the triangles around its position (six or so,
/// on a surface: nothing is searched; a position with more than 512 triangles draws on a budget, see
/// `formats::crease`). While it works it holds, beside the normals it returns, about thirty bytes for each
/// vertex, forty for each triangle and twenty for each corner: some hundred bytes a triangle.
pub(crate) fn crease(positions: &[[f32; 3]], indices: &mut [u32], degrees: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n = positions.len();
    // The triangles as they meet in space: each vertex stands for the first at its position.
    let mut first: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
    let at: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            *first
                .entry(p.map(|c| (c + 0.0).to_bits()))
                .or_insert(crate::num::u32_of(i))
        })
        .collect();
    drop(first);
    let welded: Vec<u32> = indices
        .iter()
        .map(|&i| at.get(i as usize).copied().unwrap_or(i))
        .collect();
    let corner = crate::formats::crease::default_normals(positions, &welded, degrees);
    drop((welded, at));
    let mut normals = vec![[0.0f32; 3]; n];
    let mut set = vec![false; n];
    let mut added: Vec<u32> = Vec::new();
    let mut more: std::collections::HashMap<(u32, [u32; 3]), u32> = std::collections::HashMap::new();
    for (i, normal) in indices.iter_mut().zip(&corner) {
        let v = *i as usize;
        // A triangle without area has no normal: its corners stay on the vertices they name.
        if v >= n || normal.iter().all(|c| *c == 0.0) {
            continue;
        }
        let normal = normal.map(|c| c + 0.0);
        if !set[v] {
            set[v] = true;
            normals[v] = normal;
        } else if normals[v].map(f32::to_bits) != normal.map(f32::to_bits) {
            let next = crate::num::u32_of(n + added.len());
            let at = *more.entry((*i, normal.map(f32::to_bits))).or_insert(next);
            if at == next {
                added.push(*i);
                normals.push(normal);
            }
            *i = at;
        }
    }
    for (normal, set) in normals.iter_mut().zip(set) {
        if !set {
            *normal = UP;
        }
    }
    (normals, added)
}

/// A copy of each of `added`'s vertices at the end of `list`, when it is one per vertex (`n` of them): what a
/// list beside the positions needs once [`crease`] has added vertices.
pub(crate) fn follow<T: Copy>(list: &mut Vec<T>, n: usize, added: &[u32]) {
    if list.len() == n {
        list.reserve_exact(added.len());
        for &s in added {
            list.push(list[s as usize]);
        }
    }
}

/// Gives a primitive whose file has no normals the default ones ([`crease`], at `degrees`), every list of its
/// vertices following the vertices added. Returns those vertices' sources, for what a reader keeps beside the
/// primitive (morph targets, the file's own vertex numbers).
pub(crate) fn default_normals(p: &mut crate::mesh::Prim, degrees: f32) -> Vec<u32> {
    let n = p.positions.len();
    let (normals, added) = crease(&p.positions, &mut p.indices, degrees);
    p.normals = normals;
    // Tangents mean nothing without the normals they were made beside.
    p.tangents = Vec::new();
    follow(&mut p.positions, n, &added);
    follow(&mut p.uvs, n, &added);
    follow(&mut p.colors, n, &added);
    follow(&mut p.skin_joints, n, &added);
    follow(&mut p.skin_weights, n, &added);
    for set in &mut p.uv_sets {
        follow(set, n, &added);
    }
    for set in &mut p.color_sets {
        follow(set, n, &added);
    }
    added
}

/// Makes a file's own `normals`, one per vertex, what a triangle primitive must have: unit length. They are made
/// so when they're further from it than `slack` (as a difference in squared length; 0 for every one that isn't
/// exactly unit), and one that is zero or not finite takes the area-weighted normal of its vertex's triangles.
pub(crate) fn normals(positions: &[[f32; 3]], indices: &[u32], normals: &mut [[f32; 3]], slack: f32) {
    let mut bad = Vec::new();
    for (i, v) in normals.iter_mut().enumerate() {
        let sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if (sq - 1.0).abs() <= slack {
            continue;
        }
        let l = sq.sqrt();
        if l > 0.0 && l.is_finite() {
            *v = v.map(|c| c / l);
        } else {
            bad.push(i);
        }
    }
    if bad.is_empty() {
        return;
    }
    let sum = sums(positions, indices);
    for i in bad {
        normals[i] = unit_or_up(sum[i]);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::float_cmp_const, reason = "exact directions")]
mod tests {
    use super::*;

    #[test]
    fn what_cannot_be_drawn_is_dropped() {
        let mut pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [f32::NAN, 0.0, 0.0]];
        // A good triangle, one on the NaN vertex, one past the end, a repeated corner, and a stray index.
        let mut tris = vec![0, 1, 2, 0, 1, 3, 0, 1, 9, 0, 1, 1, 2];
        assert!(!finite(&pos));
        assert_eq!(sound(&pos, false, &mut tris, 3), 2);
        assert_eq!(tris, [0, 1, 2, 0, 1, 1]);
        let mut lines = vec![0, 1, 1, 3, 2, 4];
        assert_eq!(sound(&pos, false, &mut lines, 2), 2);
        assert_eq!(lines, [0, 1]);
        let mut points = vec![3, 2, 7];
        assert_eq!(sound(&pos, false, &mut points, 1), 2);
        assert_eq!(points, [2]);
        settle(&mut pos);
        assert_eq!(pos[3], [0.0; 3]);
        // Nothing wrong: nothing changes.
        let mut fine = vec![0, 1, 2];
        assert!(finite(&pos));
        assert_eq!(sound(&pos, true, &mut fine, 3), 0);
        assert_eq!(fine, [0, 1, 2]);
        assert_eq!(sound(&[], true, &mut Vec::new(), 3), 0);
    }

    #[test]
    fn the_default_normals_are_weighted_by_area() {
        // A unit triangle in the XY plane and one four times its area in the XZ plane, sharing vertex 0.
        let pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0],
            [5.0, 5.0, 5.0],
        ];
        let n = smooth(&pos, &[0, 1, 2, 0, 3, 4]);
        let l = 17f32.sqrt();
        assert!((n[0][2] - 1.0 / l).abs() < 1e-6 && (n[0][1] - 4.0 / l).abs() < 1e-6);
        assert_eq!(n[1], [0.0, 0.0, 1.0]);
        assert_eq!(n[3], [0.0, 1.0, 0.0]);
        // No triangle: up.
        assert_eq!(n[5], UP);
        // A model far too large for a float's square still gets its normals.
        let big = pos.map(|p| p.map(|c| c * 1e30));
        assert_eq!(smooth(&big, &[0, 1, 2])[1], [0.0, 0.0, 1.0]);
        let small = pos.map(|p| p.map(|c| c * 1e-30));
        assert_eq!(smooth(&small, &[0, 1, 2])[1], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_files_normals_are_made_unit_and_broken_ones_replaced() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]];
        let mut n = vec![[0.0, 0.0, 2.0], [0.0; 3], [f32::NAN, 0.0, 0.0], [0.0; 3]];
        normals(&pos, &[0, 1, 2], &mut n, 1e-6);
        assert_eq!(n, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], UP]);
    }

    #[test]
    fn the_default_normals_split_a_vertex_at_a_hard_edge() {
        // A square on the ground and one standing on its edge from 1 to 2, sharing those two vertices; vertex 6
        // is drawn by nothing.
        let mut p = crate::mesh::Prim {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
            ],
            uvs: (0..7u8).map(|k| [f32::from(k), 0.0]).collect(),
            indices: vec![0, 1, 2, 0, 2, 3, 1, 4, 5, 1, 5, 2],
            ..crate::mesh::Prim::default()
        };
        let added = default_normals(&mut p, 30.0);
        // The wall's corners at 1 and 2 are copies, after the vertices there were.
        assert_eq!(added, [1, 2]);
        assert_eq!(p.indices, [0, 1, 2, 0, 2, 3, 7, 4, 5, 7, 5, 8]);
        assert_eq!((p.positions.len(), p.normals.len(), p.uvs.len()), (9, 9, 9));
        assert_eq!((p.positions[7], p.uvs[8]), ([1.0, 0.0, 0.0], [2.0, 0.0]));
        assert_eq!((p.normals[1], p.normals[7]), ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]));
        assert_eq!(p.normals[6], UP);
        // Two vertices at one point (split for their texture coordinates, say) across a gentle fold: no seam.
        let (s, c) = 20f32.to_radians().sin_cos();
        let mut seam = crate::mesh::Prim {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -c, s],
            ],
            indices: vec![0, 1, 2, 4, 3, 5],
            ..crate::mesh::Prim::default()
        };
        assert_eq!(default_normals(&mut seam, 30.0), [0u32; 0]);
        assert_eq!((seam.normals[0], seam.normals[1]), (seam.normals[3], seam.normals[4]));
        assert!(seam.normals[0][2] < 1.0);
        // A flat square alone keeps its four vertices.
        let mut flat = crate::mesh::Prim {
            positions: p.positions[..4].to_vec(),
            indices: vec![0, 1, 2, 0, 2, 3],
            ..crate::mesh::Prim::default()
        };
        assert_eq!(default_normals(&mut flat, 30.0), [0u32; 0]);
        assert_eq!(flat.normals, [[0.0, 0.0, 1.0]; 4]);
    }
}
