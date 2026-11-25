//! Primitives drawn: their attributes and what they draw read, skinned and morphed, then placed at each of a
//! mesh's transforms.

use serde_json::Value;

use super::accessor::Acc;
use super::{IDENTITY, M4, Reader, Use, affine_inverse, at, det3, ids, index, items, list_of, mul, normal_matrix};
use crate::Result;
use crate::document::Bounds;
#[cfg(doc)]
use crate::document::Document;
use crate::mesh::{Mesh, Prim};
use crate::num::{f32_of, u32_of};
use crate::vec3;
#[cfg(feature = "draco")]
use crate::{Context as _, ErrorKind};

/// A mesh's vertex positions, normals and tangents, to move together.
type Frames<'a> = (&'a mut [[f32; 3]], &'a mut [[f32; 3]], &'a mut [[f32; 4]]);

/// Moves `range` of the vertices by `m`, their normals by its normal matrix `nm` and their
/// tangents by `m` itself. A matrix that moves nothing leaves them exactly as they are, as an
/// unskinned mesh's are, but for normals and tangents a morph changed (`morphed`), which are
/// renormalized.
fn transform(
    m: &M4,
    nm: &[[f64; 3]; 3],
    morphed: bool,
    (positions, normals, tangents): Frames<'_>,
    range: std::ops::Range<usize>,
) {
    let still = m[..12] == IDENTITY[..12] && m[12..15] == [0.0; 3];
    if !still {
        for p in &mut positions[range.clone()] {
            *p = apply_point(m, *p);
        }
    }
    let end = range.end.min(tangents.len());
    if (morphed || !still) && range.start < end {
        let flip = det3(m) < 0.0;
        for t in &mut tangents[range.start..end] {
            *t = apply_tangent(m, *t, flip);
        }
    }
    let end = range.end.min(normals.len());
    if (morphed || !still) && range.start < end {
        // Single precision: within a unit in the last place of double, at a fraction of the cost.
        // A block at a time, with no branch in the loop (one stops it vectorizing, for a quarter
        // of the speed); a block where single precision couldn't square a normal is redone. Rows
        // written out, as in `apply_point`.
        let n = nm.map(|r| r.map(f32_of));
        let turn = |[x, y, z]: [f32; 3]| {
            [
                n[0][0] * x + n[0][1] * y + n[0][2] * z,
                n[1][0] * x + n[1][1] * y + n[1][2] * z,
                n[2][0] * x + n[2][1] * y + n[2][2] * z,
            ]
        };
        let fits = |len2: f32| len2 > 0.0 && len2.is_finite();
        let normals = &mut normals[range.start..end];
        if normals.len() < 8 {
            // A few normals, as a skinned vertex's often are: one at a time, each with the same
            // operations the blocks below make.
            for nv in normals {
                let [x, y, z] = turn(*nv);
                let len2 = x * x + y * y + z * z;
                *nv = if fits(len2) {
                    let inv = 1.0 / len2.sqrt();
                    [x * inv, y * inv, z * inv]
                } else {
                    apply_normal(nm, *nv)
                };
            }
            return;
        }
        for block in normals.chunks_mut(64) {
            let mut kept = [[0.0; 3]; 64];
            kept[..block.len()].copy_from_slice(block);
            let mut all = true;
            for nv in block.iter_mut() {
                let [x, y, z] = turn(*nv);
                let len2 = x * x + y * y + z * z;
                all &= fits(len2);
                let inv = 1.0 / len2.sqrt();
                *nv = [x * inv, y * inv, z * inv];
            }
            if !all {
                for (nv, &k) in block.iter_mut().zip(&kept) {
                    let o = turn(k);
                    if !fits(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) {
                        // Too small or large for single precision to square: double's answer.
                        *nv = apply_normal(nm, k);
                    }
                }
            }
        }
    }
}

fn apply_point(m: &M4, p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p.map(f64::from);
    // Row by row, written out: built with `map`, the result shuffles through registers at a
    // third of the speed.
    [
        f32_of(m[0] * x + m[4] * y + m[8] * z + m[12]),
        f32_of(m[1] * x + m[5] * y + m[9] * z + m[13]),
        f32_of(m[2] * x + m[6] * y + m[10] * z + m[14]),
    ]
}

fn apply_normal(n: &[[f64; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v.map(f64::from);
    // Written out, as in `apply_point`.
    let (a, b, c) = (
        n[0][0] * x + n[0][1] * y + n[0][2] * z,
        n[1][0] * x + n[1][1] * y + n[1][2] * z,
        n[2][0] * x + n[2][1] * y + n[2][2] * z,
    );
    let len = (a * a + b * b + c * c).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        [f32_of(a * inv), f32_of(b * inv), f32_of(c * inv)]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// A tangent turned by `m` (its upper 3x3, as surface directions turn) and made unit length;
/// its handedness flips under a mirroring `m` (`flip`), as the triangles' winding does.
fn apply_tangent(m: &M4, t: [f32; 4], flip: bool) -> [f32; 4] {
    let [x, y, z] = [t[0], t[1], t[2]].map(f64::from);
    let (a, b, c) = (
        m[0] * x + m[4] * y + m[8] * z,
        m[1] * x + m[5] * y + m[9] * z,
        m[2] * x + m[6] * y + m[10] * z,
    );
    let w = if (t[3] < 0.0) == flip { 1.0 } else { -1.0 };
    let len = (a * a + b * b + c * c).sqrt();
    if len > 0.0 && len.is_finite() {
        let inv = 1.0 / len;
        [f32_of(a * inv), f32_of(b * inv), f32_of(c * inv), w]
    } else {
        [1.0, 0.0, 0.0, w]
    }
}

fn normalize(v: &mut [f32; 3]) {
    *v = vec3::unit_or(*v, [0.0, 0.0, 1.0]);
}

/// One primitive's geometry before it is placed.
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    tangents: Vec<[f32; 4]>,
    uvs: Vec<[f32; 2]>,
    uv_sets: Vec<Vec<[f32; 2]>>,
    colors: Vec<[f32; 4]>,
    color_sets: Vec<Vec<[f32; 4]>>,
    draws: Draws,
    joints: Vec<u32>,
    /// Each vertex's four strongest joints as nodes (`u32::MAX` for none), and their weights.
    skin_joints: Vec<[u32; 4]>,
    skin_weights: Vec<[f32; 4]>,
}

/// What a primitive draws on its vertices, as indices into them.
#[derive(Default)]
struct Draws {
    /// Three to a triangle.
    triangles: Vec<u32>,
    /// Two to a segment.
    lines: Vec<u32>,
    points: Vec<u32>,
}

impl Draws {
    fn is_empty(&self) -> bool {
        self.triangles.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }
}

/// Whether glTF primitive `p` is drawn on the same vertices as `q`: the same attributes,
/// material, morph targets and extensions, and no Draco stream of its own.
fn same_vertices(p: &Value, q: &Value) -> bool {
    let draco = |v: &Value| at(v, &["extensions", "KHR_draco_mesh_compression"]).is_some();
    ["attributes", "material", "targets", "extensions"]
        .iter()
        .all(|k| p.get(k) == q.get(k))
        && !draco(p)
        && !draco(q)
}

/// Line segments, in pairs, from a primitive's indices, by mode (1 lines, 2 loop, 3 strip), per
/// the glTF 2.0 specification's section 3.7.2.1.
pub(super) fn segments(mode: u64, idx: Vec<u32>) -> Vec<u32> {
    match mode {
        2 | 3 => {
            let n = idx.len();
            let mut out = Vec::with_capacity(n * 2);
            for w in idx.windows(2) {
                out.extend_from_slice(w);
            }
            if mode == 2 && n >= 2 {
                out.extend_from_slice(&[idx[n - 1], idx[0]]);
            }
            out
        }
        _ => {
            let mut idx = idx;
            idx.truncate(idx.len() / 2 * 2);
            idx
        }
    }
}

/// Keeps the groups of `N` indices in `idx` that `keep` takes, in order.
pub(super) fn retain_groups<const N: usize>(idx: &mut Vec<u32>, keep: impl Fn(&[u32; N]) -> bool) {
    let mut w = 0;
    for g in 0..idx.len() / N {
        let group: [u32; N] = std::array::from_fn(|k| idx[g * N + k]);
        if keep(&group) {
            idx[w..w + N].copy_from_slice(&group);
            w += N;
        }
    }
    idx.truncate(w);
}

/// What [`Reader::skin`] gives each vertex: its strongest joint's node, and its four strongest
/// joints' nodes (`u32::MAX` for none) and weights, summing to 1.
#[derive(Default)]
struct Skinned {
    strongest: Vec<u32>,
    joints: Vec<[u32; 4]>,
    weights: Vec<[f32; 4]>,
}

/// Triangle lists from a primitive's indices, by mode (4 triangles, 5 strip, 6 fan), per the
/// glTF 2.0 specification's section 3.7.2.1.
pub(super) fn triangles(mode: u64, idx: Vec<u32>) -> Vec<u32> {
    match mode {
        5 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                if i % 2 == 0 {
                    out.extend_from_slice(&[idx[i], idx[i + 1], idx[i + 2]]);
                } else {
                    out.extend_from_slice(&[idx[i], idx[i + 2], idx[i + 1]]);
                }
            }
            out
        }
        6 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                out.extend_from_slice(&[idx[i + 1], idx[i + 2], idx[0]]);
            }
            out
        }
        _ => {
            let mut idx = idx;
            idx.truncate(idx.len() / 3 * 3);
            idx
        }
    }
}

impl Reader<'_> {
    pub(super) fn mesh(&mut self) -> Result<Option<Mesh>> {
        let uses = self.uses()?;
        let mut mesh = Mesh::default();
        // Room for every primitive and material up front: a scene of thousands of meshes
        // otherwise copies these, a primitive or material at a time, as they grow.
        let prims: usize = uses
            .iter()
            .map(|u| self.meshes.get(u.mesh).map_or(0, |m| items(m, "primitives").len()))
            .sum();
        mesh.prims.reserve(prims);
        mesh.materials.reserve(self.materials.len().min(prims) + 1);
        for u in &uses {
            // The primitive last drawn, and its place in the mesh: points and lines on the same
            // vertices join it.
            let mut last: Option<(&Value, usize)> = None;
            for p in items(self.get("meshes", u.mesh)?, "primitives") {
                if let Some((q, at)) = last
                    && mode(p) < 4
                    && same_vertices(p, q)
                {
                    self.join(p, u, &mut mesh.prims[at])?;
                    continue;
                }
                let before = mesh.prims.len();
                self.primitive(p, u, &mut mesh)?;
                last = (mesh.prims.len() > before).then_some((p, before));
            }
        }
        self.skeleton(&mut mesh);
        Ok(mesh.prims.iter().any(|p| !p.is_empty()).then_some(mesh))
    }

    /// See [`Document::bounds`].
    pub(super) fn bounds(&mut self) -> Result<Option<Bounds>> {
        let uses = self.uses()?;
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for u in &uses {
            for p in items(self.get("meshes", u.mesh)?, "primitives") {
                let mode = mode(p);
                let attrs = p.get("attributes").unwrap_or(&Value::Null);
                let Some(pos) = index(attrs, "POSITION") else {
                    continue;
                };
                if !self.drawn(mode) {
                    continue;
                }
                self.draco.clear();
                if let Some(ext) = p.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")) {
                    if mode < 4 {
                        // Not drawn: see `primitive`.
                        continue;
                    }
                    #[cfg(feature = "draco")]
                    self.decode_draco(ext, attrs, index(p, "indices"))?;
                    #[cfg(not(feature = "draco"))]
                    {
                        let _ = ext;
                        ensure!(
                            self.acc(pos)?.view.is_some(),
                            FeatureDisabled,
                            "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
                        );
                    }
                }
                let mut positions = self.read(pos, [0.0; 3])?;
                for (k, target) in items(p, "targets").iter().enumerate() {
                    let w = u.weights.get(k).copied().unwrap_or(0.0);
                    if w == 0.0 {
                        continue;
                    }
                    if let Some(a) = index(target, "POSITION") {
                        for (v, d) in positions.iter_mut().zip(self.read(a, [0.0; 3])?) {
                            for c in 0..3 {
                                v[c] += w * d[c];
                            }
                        }
                    }
                }
                if let Some(skin) = u.skin {
                    let morphed = !u.weights.is_empty();
                    self.skin(skin, attrs, &mut positions, &mut [], &mut [], morphed)?;
                }
                for m in &u.transforms {
                    let still = m[..12] == IDENTITY[..12] && m[12..15] == [0.0; 3];
                    for &v in &positions {
                        let v = if still { v } else { apply_point(m, v) };
                        if v.iter().all(|x| x.is_finite()) {
                            for c in 0..3 {
                                lo[c] = lo[c].min(v[c]);
                                hi[c] = hi[c].max(v[c]);
                            }
                        }
                    }
                }
            }
        }
        Ok((lo[0] <= hi[0]).then_some(Bounds { min: lo, max: hi }))
    }

    /// Whether a primitive in `mode` is drawn: triangles, strips and fans, and points and lines
    /// when the document draws them.
    fn drawn(&self, mode: u64) -> bool {
        matches!(mode, 4..=6) || (mode < 4 && self.doc.lines_and_points)
    }

    /// Checks that a primitive with index accessor `indices` and position accessor `pos`, in
    /// `mode`, drawn `k` times, stays within what may still be drawn, before anything is read:
    /// what its accessors declare is counted. Gives the accessor it counted, to be read next
    /// rather than looked up again: the indices', or else the positions'.
    fn afford(&self, indices: Option<usize>, pos: usize, mode: u64, k: usize) -> Result<Acc> {
        let counted = self.acc(indices.unwrap_or(pos))?;
        let corners = counted.count;
        let n = match mode {
            0 | 2 => corners,
            1 => corners / 2,
            3 => corners.saturating_sub(1),
            4 => corners / 3,
            _ => corners.saturating_sub(2),
        };
        let total = n.saturating_mul(k);
        if mode < 4 {
            ensure!(
                total <= self.point_budget,
                LimitExceeded,
                "too many points and lines to convert (more than {})",
                self.doc.max_points
            );
        } else {
            ensure!(
                total <= self.budget,
                LimitExceeded,
                "too many triangles to convert (more than {})",
                self.doc.max_triangles
            );
        }
        Ok(counted)
    }

    /// What primitive `p` draws in `mode` on its `n` vertices: triangles, line segments or
    /// points. Those that name a vertex that doesn't exist are dropped, and segments from a
    /// vertex to itself.
    fn draws(&mut self, indices: Option<(usize, &Acc)>, mode: u64, n: usize) -> Result<Draws> {
        let idx = match indices {
            Some((i, a)) => self.read_index_acc(i, a)?,
            None => (0..u32_of(n)).collect(),
        };
        let fits = |i: u32| (i as usize) < n;
        let mut d = Draws::default();
        match mode {
            0 => {
                d.points = idx;
                d.points.retain(|&i| fits(i));
            }
            1..=3 => {
                d.lines = segments(mode, idx);
                retain_groups(&mut d.lines, |&[a, b]: &[u32; 2]| a != b && fits(a) && fits(b));
            }
            _ => {
                d.triangles = triangles(mode, idx);
                if d.triangles.iter().copied().max().is_some_and(|m| !fits(m)) {
                    // Drop triangles that name vertices that don't exist.
                    retain_groups(&mut d.triangles, |t: &[u32; 3]| t.iter().all(|&i| fits(i)));
                }
            }
        }
        Ok(d)
    }

    /// Takes what `d`, drawn `k` times, draws from what may still be drawn.
    fn charge(&mut self, d: &Draws, k: usize) {
        self.budget -= d.triangles.len() / 3 * k;
        self.point_budget -= (d.lines.len() / 2 + d.points.len()) * k;
    }

    /// Adds the points or lines of primitive `p`, drawn on the same vertices `prim` was made
    /// from, to `prim`.
    fn join(&mut self, p: &Value, u: &Use, prim: &mut Prim) -> Result<()> {
        let mode = mode(p);
        let Some(pos) = p.get("attributes").and_then(|a| index(a, "POSITION")) else {
            return Ok(());
        };
        if !self.drawn(mode) {
            return Ok(());
        }
        let k = u.transforms.len();
        let indices = index(p, "indices");
        let counted = self.afford(indices, pos, mode, k)?;
        let n = prim.positions.len() / k.max(1);
        let d = self.draws(indices.map(|i| (i, &counted)), mode, n)?;
        self.charge(&d, k);
        extend(prim, &d, n, &u.transforms);
        Ok(())
    }

    /// The mesh's skeleton: every node a vertex is bound to, and their ancestors, parents first,
    /// where they are in the pose drawn. Each primitive's joints, nodes until now, become bones.
    fn skeleton(&self, mesh: &mut Mesh) {
        let nodes = self.nodes;
        let mut used = vec![false; nodes.len()];
        for p in mesh.prims.iter().filter(|p| p.is_skinned()) {
            for &j in p.skin_joints.as_flattened() {
                if let Some(u) = used.get_mut(j as usize) {
                    *u = true;
                }
            }
        }
        // Each node's parent: the first node listing it as a child, as `Document::rig` has it.
        let mut parent = vec![None; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for c in ids(n, "children") {
                if c < nodes.len() && c != i && parent[c].is_none() {
                    parent[c] = Some(i);
                }
            }
        }
        // Ancestors too, so the bones hang together as the nodes do. Bounded, against cycles.
        for i in 0..nodes.len() {
            if !used[i] {
                continue;
            }
            let mut at = parent[i];
            for _ in 0..nodes.len() {
                let Some(p) = at else { break };
                if used[p] {
                    break;
                }
                used[p] = true;
                at = parent[p];
            }
        }
        // Parents first: down from each bone whose parent isn't one; a cycle's nodes, with no
        // such bone above them, become roots after.
        let mut kids = vec![Vec::new(); nodes.len()];
        for i in (0..nodes.len()).filter(|&i| used[i]) {
            if let Some(p) = parent[i].filter(|&p| used[p]) {
                kids[p].push(i);
            }
        }
        let mut bone_of = vec![None::<usize>; nodes.len()];
        let mut order = Vec::new();
        let roots: Vec<usize> = (0..nodes.len())
            .filter(|&i| used[i] && parent[i].is_none_or(|p| !used[p]))
            .collect();
        let cyclic: Vec<usize> = (0..nodes.len()).filter(|&i| used[i]).collect();
        for root in roots.into_iter().chain(cyclic) {
            if bone_of[root].is_some() {
                continue;
            }
            let mut stack = vec![root];
            while let Some(i) = stack.pop() {
                if bone_of[i].is_some() {
                    continue;
                }
                bone_of[i] = Some(order.len());
                order.push(i);
                stack.extend(kids[i].iter().rev().copied());
            }
        }
        mesh.skeleton = order
            .iter()
            .map(|&i| crate::mesh::Bone {
                name: nodes[i].get("name").and_then(Value::as_str).map(str::to_owned),
                parent: parent[i]
                    .and_then(|p| bone_of[p])
                    .filter(|&b| bone_of[i].is_some_and(|me| b < me)),
                node: i,
                matrix: self
                    .bound
                    .get(i)
                    .copied()
                    .flatten()
                    .or_else(|| self.world.get(i).copied())
                    .unwrap_or(IDENTITY)
                    .map(f32_of),
            })
            .collect();
        // Each node's bone, as a flat table: one load per joint, no branches.
        let bone: Vec<u32> = bone_of.iter().map(|b| b.map_or(0, u32_of)).collect();
        for p in &mut mesh.prims {
            if mesh.skeleton.is_empty() || !p.is_skinned() {
                p.skin_joints = Vec::new();
                p.skin_weights = Vec::new();
                continue;
            }
            // Slots no joint fills keep weight 0 and name the first bone.
            for (j, w) in p.skin_joints.iter_mut().zip(&p.skin_weights) {
                *j = std::array::from_fn(|k| {
                    let b = bone.get(j[k] as usize).copied().unwrap_or(0);
                    if w[k] > 0.0 { b } else { 0 }
                });
            }
        }
    }

    /// A gaussian splat primitive's colors when it doesn't give them (`KHR_gaussian_splatting`
    /// asks for a `COLOR_0`, but doesn't require one): each splat's base color and opacity, as
    /// the extension works them out. Empty for other primitives.
    fn splat_colors(&mut self, p: &Value, attrs: &Value) -> Result<Vec<[f32; 4]>> {
        let Some(ext) = at(p, &["extensions", "KHR_gaussian_splatting"]) else {
            return Ok(Vec::new());
        };
        let (Some(dc), Some(opacity)) = (
            index(attrs, crate::gaussian::SH0),
            index(attrs, crate::gaussian::OPACITY),
        ) else {
            return Ok(Vec::new());
        };
        let srgb = ext.get("colorSpace").and_then(Value::as_str) != Some("lin_rec709_display");
        let dc = self.read(dc, [0.0; 3])?;
        let opacity = self.read(opacity, [1.0])?;
        if dc.len() != opacity.len() {
            return Ok(Vec::new());
        }
        let fallback = |(&d, &[o]): (&[f32; 3], &[f32; 1])| crate::gaussian::fallback(d, o, srgb);
        Ok(dc.iter().zip(&opacity).map(fallback).collect())
    }

    fn primitive(&mut self, p: &Value, u: &Use, mesh: &mut Mesh) -> Result<()> {
        let mode = mode(p);
        let attrs = p.get("attributes").unwrap_or(&Value::Null);
        let Some(pos) = index(attrs, "POSITION") else {
            return Ok(());
        };
        if !self.drawn(mode) {
            return Ok(());
        }

        self.draco.clear();
        if let Some(ext) = p.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")) {
            if mode < 4 {
                // Draco point clouds and lines aren't decoded.
                return Ok(());
            }
            #[cfg(feature = "draco")]
            self.decode_draco(ext, attrs, index(p, "indices"))?;
            #[cfg(not(feature = "draco"))]
            {
                // Files may carry uncompressed data beside the Draco stream, for readers without it.
                let _ = ext;
                ensure!(
                    self.acc(pos)?.view.is_some(),
                    FeatureDisabled,
                    "glTF uses Draco compression, which needs Polysmelt's `draco` feature"
                );
            }
        }

        // Check the count before reading any vertices.
        let indices = index(p, "indices");
        let counted = self.afford(indices, pos, mode, u.transforms.len())?;

        let mut positions = match indices {
            Some(_) => self.read(pos, [0.0; 3])?,
            None => self.read_acc(&counted, [0.0; 3])?,
        };
        let n = positions.len();
        let draws = self.draws(indices.map(|i| (i, &counted)), mode, n)?;
        if draws.is_empty() {
            return Ok(());
        }

        let material = index(p, "material");
        let (set, uv_matrix) = self.uv_set(material);
        let attr = |name: &str| index(attrs, name);
        let mut normals = match attr("NORMAL") {
            Some(a) => self.read(a, [0.0; 3])?,
            None => Vec::new(),
        };
        let mut uvs = match attr(&texcoord(set)) {
            Some(a) => self.read(a, [0.0; 2])?,
            None => Vec::new(),
        };
        let colors = match attr("COLOR_0") {
            Some(a) => self.read(a, [0.0, 0.0, 0.0, 1.0])?,
            None => self.splat_colors(p, attrs)?,
        };
        // Attributes that don't match the positions are dropped rather than trusted.
        if normals.len() != n {
            normals = Vec::new();
        }
        if uvs.len() != n {
            uvs = Vec::new();
        }
        let colors = if colors.len() == n { colors } else { Vec::new() };
        // Tangents mean nothing without the normals they're perpendicular to.
        let mut tangents = match attr("TANGENT") {
            Some(a) if !normals.is_empty() => self.read(a, [1.0, 0.0, 0.0, 1.0])?,
            _ => Vec::new(),
        };
        if tangents.len() != n {
            tangents = Vec::new();
        }
        // The other sets: one that can't be read, or doesn't fit, is dropped, not the model.
        let uv_sets = numbered(attrs, "TEXCOORD_", |k| k != set, |a| self.read(a, [0.0; 2]).ok(), n);
        let color_sets = numbered(
            attrs,
            "COLOR_",
            |k| k != 0,
            |a| self.read(a, [0.0, 0.0, 0.0, 1.0]).ok(),
            n,
        );
        if let Some(t) = uv_matrix {
            for uv in &mut uvs {
                let [u0, v0] = *uv;
                *uv = [t[0] * u0 + t[1] * v0 + t[2], t[3] * u0 + t[4] * v0 + t[5]];
            }
        }

        // Morph targets at their default weights.
        for (k, target) in items(p, "targets").iter().enumerate() {
            let w = u.weights.get(k).copied().unwrap_or(0.0);
            if w == 0.0 {
                continue;
            }
            if let Some(a) = index(target, "POSITION") {
                for (v, d) in positions.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
            if !normals.is_empty()
                && let Some(a) = index(target, "NORMAL")
            {
                for (v, d) in normals.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
            if !tangents.is_empty()
                && let Some(a) = index(target, "TANGENT")
            {
                for (v, d) in tangents.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
        }
        let mut skinned = Skinned::default();
        let morphed = !u.weights.is_empty();
        if let Some(skin) = u.skin {
            skinned = self.skin(skin, attrs, &mut positions, &mut normals, &mut tangents, morphed)?;
        } else if morphed {
            normals.iter_mut().for_each(normalize);
            for t in &mut tangents {
                *t = apply_tangent(&IDENTITY, *t, false);
            }
        }

        let g = Geometry {
            positions,
            normals,
            tangents,
            uvs,
            uv_sets,
            colors,
            color_sets,
            draws,
            joints: skinned.strongest,
            skin_joints: skinned.joints,
            skin_weights: skinned.weights,
        };
        let material = self.material(material, mesh);
        self.charge(&g.draws, u.transforms.len());
        mesh.prims.push(place(g, &u.transforms, material));
        Ok(())
    }

    /// Decodes a primitive's Draco stream, so its indices and the attributes it compresses read
    /// from the decoded values instead of their buffer views. Positions and other quantized
    /// attributes come out dequantized; integer attributes are normalized here when their
    /// accessor says so.
    #[cfg(feature = "draco")]
    pub(super) fn decode_draco(&mut self, ext: &Value, attrs: &Value, indices: Option<usize>) -> Result<()> {
        use draco_core::decode_limits::DecodeLimits;
        use draco_core::decoder_buffer::DecoderBuffer;
        use draco_core::draco_types::DataType;
        use draco_core::mesh_decoder::MeshDecoder;

        let view = index(ext, "bufferView").context(ErrorKind::Corrupt, "glTF Draco primitive has no bufferView")?;
        let bytes = self.view_shared(view)?;
        let limits = DecodeLimits::default()
            .with_max_faces(self.budget as u64)
            .with_max_points(self.max_vertices as u64);
        let mut buffer = DecoderBuffer::new(&bytes).with_limits(limits);
        let mut dm = draco_core::mesh::Mesh::new();
        MeshDecoder::new()
            .decode(&mut buffer, &mut dm)
            .context(ErrorKind::Corrupt, "glTF Draco data doesn't decode")?;
        drop(bytes);

        if let Some(i) = indices {
            let mut out = Vec::with_capacity(dm.num_faces() * 12);
            for face in dm.faces() {
                for p in face {
                    out.extend_from_slice(&p.0.to_le_bytes());
                }
            }
            self.draco.push((i, out.into(), 5125, 1));
        }
        let points = dm.num_points();
        let mut decoded = Vec::new();
        for (name, id) in ext.get("attributes").and_then(Value::as_object).into_iter().flatten() {
            let (Some(acc), Some(id)) = (index(attrs, name), id.as_u64().and_then(|v| u32::try_from(v).ok())) else {
                continue;
            };
            let attribute = dm.attribute_by_unique_id(id).with_context(ErrorKind::Corrupt, || {
                format!("glTF Draco stream has no attribute {id} for {name}")
            })?;
            let a = self.acc(acc)?;
            let mut values = attribute.read_f32s(points, a.n);
            if a.normalized || attribute.normalized() {
                let max = match attribute.data_type() {
                    DataType::Uint8 => Some((255.0, 0.0)),
                    DataType::Int8 => Some((127.0, -1.0)),
                    DataType::Uint16 => Some((65535.0, 0.0)),
                    DataType::Int16 => Some((32767.0, -1.0)),
                    _ => None,
                };
                if let Some((max, floor)) = max {
                    for v in &mut values {
                        *v = (*v / max).max(floor);
                    }
                }
            }
            let mut out = Vec::with_capacity(values.len() * 4);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            decoded.push((acc, out.into(), 5126, a.n));
        }
        self.draco.extend(decoded);
        Ok(())
    }

    /// Poses a skinned primitive: each vertex goes through its joints' matrices, blended by
    /// weight over every `JOINTS_n`/`WEIGHTS_n` set, not just the first four. Gives each vertex's
    /// joints, or nothing if the weights can't be read. `morphed` normals are renormalized even
    /// where the joints don't move them.
    fn skin(
        &mut self,
        skin: usize,
        attrs: &Value,
        positions: &mut [[f32; 3]],
        normals: &mut [[f32; 3]],
        tangents: &mut [[f32; 4]],
        morphed: bool,
    ) -> Result<Skinned> {
        let s = self.get("skins", skin)?;
        let joints = list_of(s, "joints");
        let ibm = match index(s, "inverseBindMatrices") {
            Some(a) => self.read(a, IDENTITY.map(f32_of))?,
            None => Vec::new(),
        };
        if self.bind {
            if self.bound.is_empty() {
                self.bound = vec![None; self.posed.len()];
            }
            for (k, &j) in joints.iter().enumerate() {
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                if let Some(slot @ None) = self.bound.get_mut(j) {
                    *slot = Some(affine_inverse(&b).unwrap_or(IDENTITY));
                }
            }
        }
        let mats: Vec<M4> = joints
            .iter()
            .enumerate()
            .map(|(k, &j)| {
                let w = self.world.get(j).copied().unwrap_or(IDENTITY);
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                mul(&w, &b)
            })
            .collect();
        // Every set of four influences, JOINTS_0/WEIGHTS_0 then JOINTS_1/WEIGHTS_1 and on, as
        // long as both halves are there and fit the vertices. Joint indices are integers, read as
        // floats exactly.
        let mut sets = Vec::new();
        for n in 0.. {
            let (Some(ja), Some(wa)) = (
                index(attrs, &format!("JOINTS_{n}")),
                index(attrs, &format!("WEIGHTS_{n}")),
            ) else {
                break;
            };
            let ji = self.read(ja, [0.0; 4])?;
            let wi = self.read(wa, [0.0; 4])?;
            if ji.len() != positions.len() || wi.len() != positions.len() {
                break;
            }
            sets.push((ji, wi));
        }
        if sets.is_empty() {
            return Ok(Skinned::default());
        }
        // A skin's joint index as its node, `None` past the skin's joints.
        let node = |j: f32| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "joint indices are small whole numbers; `as` saturates bad ones and `get` refuses them"
            )]
            joints.get(j as usize).copied()
        };
        let n = positions.len();
        let mut out = Skinned {
            strongest: Vec::with_capacity(n),
            joints: Vec::with_capacity(n),
            weights: Vec::with_capacity(n),
        };
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "joint indices are small whole numbers; `as` saturates bad ones and `get` refuses them"
        )]
        let joint = |j: f32| j as usize;
        // Each joint's normal matrix, made the first time a vertex follows that joint alone.
        let mut rigid: Vec<Option<[[f64; 3]; 3]>> = vec![None; mats.len()];
        let unmoved = normal_matrix(&IDENTITY);
        // The previous vertex's influences and blend, reused when the next one's are the same:
        // neighbouring vertices often share their weights.
        let mut last: Vec<(f32, f32)> = Vec::with_capacity(4 * sets.len());
        let (mut m, mut nm) = (IDENTITY, unmoved);
        // Vertices `start..v` all go through `m` (when `moving`; they stay put otherwise). They
        // are moved together when it changes, in one tight loop rather than one at a time.
        let (mut start, mut moving) = (0, false);
        // In the bind pose nothing moves; morphed normals are still renormalized, below.
        let bind = self.bind;
        let bits = |a: [f32; 4]| a.map(f32::to_bits);
        // One vertex's influences, (joint index, weight), reused.
        let mut inf: Vec<(f32, f32)> = Vec::with_capacity(4 * sets.len());
        // The previous vertex's output, kept at hand for the next one that repeats it.
        let mut prev = (u32::MAX, [u32::MAX; 4], [0.0f32; 4]);
        for v in 0..n {
            // The same rows as the previous vertex: everything it worked out holds for this one.
            if v > 0
                && sets
                    .iter()
                    .all(|(ji, wi)| bits(ji[v]) == bits(ji[v - 1]) && bits(wi[v]) == bits(wi[v - 1]))
            {
                out.strongest.push(prev.0);
                out.joints.push(prev.1);
                out.weights.push(prev.2);
                continue;
            }
            inf.clear();
            for (ji, wi) in &sets {
                for k in 0..4 {
                    let (j, w) = (ji[v][k], wi[v][k]);
                    // Negative and NaN weights move nothing.
                    if w > 0.0 {
                        match inf.iter_mut().find(|(i, _)| i.to_bits() == j.to_bits()) {
                            Some((_, x)) => *x += w,
                            None => inf.push((j, w)),
                        }
                    }
                }
            }
            // Strongest first, the earlier on a tie, as the file lists them. Weights here are
            // positive, so `<` orders them; a few of them sort fastest by insertion.
            for i in 1..inf.len() {
                let mut k = i;
                while k > 0 && inf[k - 1].1 < inf[k].1 {
                    inf.swap(k - 1, k);
                    k -= 1;
                }
            }
            prev.0 = inf.first().and_then(|&(j, _)| node(j)).map_or(u32::MAX, u32_of);
            out.strongest.push(prev.0);
            // The four strongest that name a joint, for binding the vertex back to them.
            let mut top = [(u32::MAX, 0.0f32); 4];
            let mut slots = top.iter_mut();
            for &(j, w) in &inf {
                if let Some(nd) = node(j) {
                    let Some(slot) = slots.next() else { break };
                    *slot = (u32_of(nd), w);
                }
            }
            let kept: f32 = top.iter().map(|t| t.1).sum();
            prev.1 = top.map(|t| t.0);
            prev.2 = if kept > 0.0 { top.map(|t| t.1 / kept) } else { [0.0; 4] };
            out.joints.push(prev.1);
            out.weights.push(prev.2);

            let sum: f32 = inf.iter().map(|t| t.1).sum();
            if bind || !(sum > 0.0 && sum.is_finite()) {
                if moving {
                    transform(&m, &nm, morphed, (positions, normals, tangents), start..v);
                    moving = false;
                }
                continue;
            }
            let same = inf.len() == last.len()
                && inf
                    .iter()
                    .zip(&last)
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits());
            if !(same && moving) {
                if moving {
                    transform(&m, &nm, morphed, (positions, normals, tangents), start..v);
                }
                (start, moving) = (v, true);
            }
            if !same {
                if let [(j, _)] = inf[..] {
                    // One joint: its matrix as it is (its weight over the sum is exactly 1).
                    let j = joint(j);
                    m = mats.get(j).copied().unwrap_or(IDENTITY);
                    nm = match rigid.get_mut(j) {
                        Some(slot) => *slot.get_or_insert_with(|| normal_matrix(&m)),
                        None => unmoved,
                    };
                } else if let Some(first) = inf
                    .iter()
                    .map(|&(j, _)| mats.get(joint(j)).unwrap_or(&IDENTITY))
                    .try_fold(None::<&M4>, |seen, jm| match seen {
                        #[allow(clippy::float_cmp, reason = "the very same matrix, bit for bit, is what's sought")]
                        Some(s) if s != jm => Err(()),
                        _ => Ok(Some(jm)),
                    })
                    .ok()
                    .flatten()
                {
                    // Joints that all sit the same way: their blend is that matrix exactly,
                    // which rounding the weights' sum would miss.
                    m = *first;
                    nm = normal_matrix(&m);
                } else {
                    // Only the upper three rows: the bottom one is never read.
                    m = [0.0; 16];
                    for &(j, w) in &inf {
                        let jm = mats.get(joint(j)).unwrap_or(&IDENTITY);
                        let f = f64::from(w / sum);
                        for c in 0..4 {
                            for r in 0..3 {
                                m[c * 4 + r] += f * jm[c * 4 + r];
                            }
                        }
                    }
                    nm = normal_matrix(&m);
                }
                last.clone_from(&inf);
            }
        }
        if moving {
            transform(&m, &nm, morphed, (positions, normals, tangents), start..n);
        }
        if bind && morphed {
            normals.iter_mut().for_each(normalize);
            for t in tangents.iter_mut() {
                *t = apply_tangent(&IDENTITY, *t, false);
            }
        }
        Ok(out)
    }
}

/// The most sets of one attribute (`TEXCOORD_n`, `COLOR_n`) read: past any real model's, and a
/// bound on what a hostile name like `TEXCOORD_4000000000` could make room for.
const MAX_SETS: usize = 64;

/// A primitive's numbered attribute sets (`prefix` then the number), each at its number, for
/// the numbers `want` picks: a set that `read` can't read or that doesn't have `n` elements is
/// left empty, and so are the gaps. Empty when no set is kept.
fn numbered<T>(
    attrs: &Value,
    prefix: &str,
    want: impl Fn(u64) -> bool,
    mut read: impl FnMut(usize) -> Option<Vec<T>>,
    n: usize,
) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = Vec::new();
    let Some(attrs) = attrs.as_object() else { return out };
    for (name, a) in attrs {
        let Some(k) = name.strip_prefix(prefix).and_then(|k| k.parse::<u64>().ok()) else {
            continue;
        };
        let (Some(a), Ok(slot)) = (a.as_u64().and_then(|a| usize::try_from(a).ok()), usize::try_from(k)) else {
            continue;
        };
        if slot >= MAX_SETS || !want(k) {
            continue;
        }
        if let Some(v) = read(a).filter(|v| v.len() == n) {
            if out.len() <= slot {
                out.resize_with(slot + 1, Vec::new);
            }
            out[slot] = v;
        }
    }
    out
}

/// A primitive's geometry at each of `transforms`, as one primitive.
#[allow(clippy::float_cmp_const, reason = "only the exact identity may skip the transform")]
fn place(mut g: Geometry, transforms: &[M4], material: usize) -> Prim {
    if let [t] = transforms {
        // One transform: in place, rather than into new buffers.
        if *t != IDENTITY {
            for v in &mut g.positions {
                *v = apply_point(t, *v);
            }
            let nm = normal_matrix(t);
            for v in &mut g.normals {
                *v = apply_normal(&nm, *v);
            }
            let flip = det3(t) < 0.0;
            for v in &mut g.tangents {
                *v = apply_tangent(t, *v, flip);
            }
            // As `extend` writes them: whole triangles, wound back when mirrored.
            let whole = g.draws.triangles.len() / 3 * 3;
            g.draws.triangles.truncate(whole);
            if flip {
                for tri in g.draws.triangles.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
        }
        return Prim {
            positions: g.positions,
            normals: g.normals,
            tangents: g.tangents,
            uvs: g.uvs,
            uv_sets: g.uv_sets,
            colors: g.colors,
            color_sets: g.color_sets,
            indices: g.draws.triangles,
            lines: g.draws.lines,
            points: g.draws.points,
            material,
            joints: g.joints,
            skin_joints: g.skin_joints,
            skin_weights: g.skin_weights,
        };
    }
    let n = g.positions.len();
    let k = transforms.len();
    let mut p = Prim {
        positions: Vec::with_capacity(n * k),
        normals: Vec::with_capacity(g.normals.len() * k),
        tangents: Vec::with_capacity(g.tangents.len() * k),
        uvs: Vec::with_capacity(g.uvs.len() * k),
        uv_sets: g.uv_sets.iter().map(|s| Vec::with_capacity(s.len() * k)).collect(),
        colors: Vec::with_capacity(g.colors.len() * k),
        color_sets: g.color_sets.iter().map(|s| Vec::with_capacity(s.len() * k)).collect(),
        indices: Vec::with_capacity(g.draws.triangles.len() * k),
        lines: Vec::with_capacity(g.draws.lines.len() * k),
        points: Vec::with_capacity(g.draws.points.len() * k),
        material,
        joints: Vec::with_capacity(g.joints.len() * k),
        skin_joints: Vec::with_capacity(g.skin_joints.len() * k),
        skin_weights: Vec::with_capacity(g.skin_weights.len() * k),
    };
    for t in transforms {
        p.positions.extend(g.positions.iter().map(|&v| apply_point(t, v)));
        let nm = normal_matrix(t);
        p.normals.extend(g.normals.iter().map(|&v| apply_normal(&nm, v)));
        let flip = det3(t) < 0.0;
        p.tangents.extend(g.tangents.iter().map(|&v| apply_tangent(t, v, flip)));
        p.uvs.extend_from_slice(&g.uvs);
        for (to, from) in p.uv_sets.iter_mut().zip(&g.uv_sets) {
            to.extend_from_slice(from);
        }
        p.colors.extend_from_slice(&g.colors);
        for (to, from) in p.color_sets.iter_mut().zip(&g.color_sets) {
            to.extend_from_slice(from);
        }
        p.joints.extend_from_slice(&g.joints);
        p.skin_joints.extend_from_slice(&g.skin_joints);
        p.skin_weights.extend_from_slice(&g.skin_weights);
    }
    extend(&mut p, &g.draws, n, transforms);
    p
}

/// Adds what `d` draws on `n` vertices to `p`, once for each of `transforms`, where `p` has those
/// vertices at each in turn: the `k`th's from `k * n`. A mirroring transform flips the triangles'
/// winding back.
fn extend(p: &mut Prim, d: &Draws, n: usize, transforms: &[M4]) {
    for (k, t) in transforms.iter().enumerate() {
        let base = u32_of(k * n);
        let flip = det3(t) < 0.0;
        for tri in d.triangles.chunks_exact(3) {
            let [a, b, c] = [tri[0] + base, tri[1] + base, tri[2] + base];
            p.indices.extend_from_slice(&if flip { [a, c, b] } else { [a, b, c] });
        }
        p.lines.extend(d.lines.iter().map(|&i| i + base));
        p.points.extend(d.points.iter().map(|&i| i + base));
    }
}

/// The name of texture coordinate set `set`'s attribute.
fn texcoord(set: u64) -> std::borrow::Cow<'static, str> {
    const NAMES: [&str; 4] = ["TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2", "TEXCOORD_3"];
    match usize::try_from(set).ok().and_then(|s| NAMES.get(s)) {
        Some(name) => (*name).into(),
        None => format!("TEXCOORD_{set}").into(),
    }
}

/// A glTF primitive's mode: 4 (triangles) when it doesn't say.
pub(super) fn mode(p: &Value) -> u64 {
    p.get("mode").and_then(Value::as_u64).unwrap_or(4)
}
