//! The 3MF beam lattice extension, made into triangles: each beam a conical frustum between two of the mesh's
//! vertices, from one radius to the other, closed at each end by a sphere, a hemisphere or a flat disk; balls at
//! the beams' ends; and the lattice clipped to the inside or outside of another mesh.
//!
//! Sides and round ends are smooth, with normals across the curve, and only as fine as they need to be: no chord
//! strays from the true surface by more than a small fraction of the lattice's size, coarser where the lattice
//! would otherwise pass its budget of triangles. The spheres at a vertex (its beams' sphere ends and its ball) are
//! concentric, so their union is the largest of them, made once. Pieces overlap rather than being merged into one
//! surface, as a renderer needs no more.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use crate::mesh::Prim;

/// How a beam's end is closed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Cap {
    Sphere,
    Hemisphere,
    Butt,
}

impl Cap {
    pub(super) fn parse(v: &[u8]) -> Option<Cap> {
        match v.trim_ascii() {
            b"sphere" => Some(Cap::Sphere),
            b"hemisphere" => Some(Cap::Hemisphere),
            b"butt" => Some(Cap::Butt),
            _ => None,
        }
    }
}

/// Where the lattice has balls: nowhere, where a `<ball>` says, or at every beam's ends.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum BallMode {
    #[default]
    None,
    Mixed,
    All,
}

/// Which side of its clipping mesh the lattice keeps.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Clip {
    #[default]
    None,
    Inside,
    Outside,
}

/// A `<beamlattice>`, its radii resolved: a beam's or ball's own, else the lattice's default (`NaN` when neither
/// says, which leaves it out).
pub(super) struct Lattice {
    pub(super) minlength: f32,
    pub(super) radius: f32,
    pub(super) cap: Cap,
    pub(super) ball_mode: BallMode,
    pub(super) ball_radius: f32,
    pub(super) clip: Clip,
    pub(super) clipping_mesh: Option<u32>,
    pub(super) pid: Option<u32>,
    pub(super) pindex: Option<u32>,
    pub(super) beams: Vec<Beam>,
    pub(super) balls: Vec<Ball>,
    pub(super) sets: Vec<BeamSet>,
}

impl Default for Lattice {
    fn default() -> Self {
        Lattice {
            minlength: 0.0,
            radius: f32::NAN,
            cap: Cap::Sphere,
            ball_mode: BallMode::None,
            ball_radius: f32::NAN,
            clip: Clip::None,
            clipping_mesh: None,
            pid: None,
            pindex: None,
            beams: Vec::new(),
            balls: Vec::new(),
            sets: Vec::new(),
        }
    }
}

pub(super) struct Beam {
    pub(super) v: [u32; 2],
    pub(super) r: [f32; 2],
    pub(super) cap: [Cap; 2],
    pub(super) pid: Option<u32>,
    pub(super) p: [Option<u32>; 2],
}

pub(super) struct Ball {
    pub(super) v: u32,
    pub(super) r: f32,
    pub(super) pid: Option<u32>,
    pub(super) p: Option<u32>,
}

/// A `<beamset>`: its name, identifier, and how many beams and balls it groups.
#[derive(Default)]
pub(super) struct BeamSet {
    pub(super) name: Option<String>,
    pub(super) identifier: Option<String>,
    pub(super) beams: usize,
    pub(super) balls: usize,
}

/// What gives a sphere at a vertex its radius and properties: a beam's end (by its place among the kept beams, and
/// which end), a `<ball>`, or the lattice's default ball.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum From {
    End(usize, usize),
    Ball(usize),
    Lattice,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Sphere {
    pub(super) v: u32,
    pub(super) r: f32,
    pub(super) from: From,
}

/// What a lattice draws: the beams kept (indices into its beams, in order), and a sphere at each vertex with one.
pub(super) struct Plan {
    pub(super) beams: Vec<usize>,
    pub(super) spheres: Vec<Sphere>,
    /// Vertices with a ball.
    pub(super) balls: usize,
}

/// A vertex attribute a piece carries besides its position and normal: none, texture coordinates (glTF's
/// convention), or a linear color.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Attr {
    None,
    Uv([f32; 2]),
    Color([f32; 4]),
}

/// The primitives a beam's pieces go in, by index, and their attributes: the side, from one end's to the other's,
/// and each end's cap.
#[derive(Clone, Copy, Debug)]
pub(super) struct BeamLook {
    pub(super) side: usize,
    pub(super) side_attr: [Attr; 2],
    pub(super) ends: [(usize, Attr); 2],
}

fn finite(p: [f32; 3]) -> bool {
    p.iter().all(|c| c.is_finite())
}

fn positive(r: f32) -> bool {
    r.is_finite() && r > 0.0
}

/// The beams that are drawn and the spheres at the vertices. A beam is left out when it names a vertex that isn't
/// there, joins a vertex to itself, has a radius that isn't positive, or is shorter than the lattice's least length
/// (which the specification says to ignore).
pub(super) fn plan(l: &Lattice, verts: &[[f32; 3]]) -> Plan {
    let at = |v: u32| verts.get(v as usize).copied().filter(|p| finite(*p));
    let mut beams = Vec::new();
    for (i, b) in l.beams.iter().enumerate() {
        let (Some(a), Some(c)) = (at(b.v[0]), at(b.v[1])) else {
            continue;
        };
        let len = length(sub(c, a));
        let long = len > 0.0 && len >= l.minlength;
        if b.v[0] == b.v[1] || !long || !positive(b.r[0]) || !positive(b.r[1]) {
            continue;
        }
        beams.push(i);
    }
    // The sphere at each vertex: its ball (the lattice's default at every beam end when it has balls at all of
    // them, unless a `<ball>` says otherwise), then any larger sphere end; a later beam's end wins a tie.
    let mut best: HashMap<u32, (f32, From)> = HashMap::new();
    if l.ball_mode == BallMode::All && positive(l.ball_radius) {
        for &i in &beams {
            for v in l.beams[i].v {
                best.insert(v, (l.ball_radius, From::Lattice));
            }
        }
    }
    if l.ball_mode != BallMode::None {
        for (k, ball) in l.balls.iter().enumerate() {
            if at(ball.v).is_some() && positive(ball.r) {
                best.insert(ball.v, (ball.r, From::Ball(k)));
            }
        }
    }
    let balls = best.len();
    for (k, &i) in beams.iter().enumerate() {
        let b = &l.beams[i];
        for end in 0..2 {
            if b.cap[end] == Cap::Sphere {
                let r = b.r[end];
                if best.get(&b.v[end]).is_none_or(|&(have, _)| have <= r) {
                    best.insert(b.v[end], (r, From::End(k, end)));
                }
            }
        }
    }
    let mut spheres: Vec<Sphere> = best.into_iter().map(|(v, (r, from))| Sphere { v, r, from }).collect();
    spheres.sort_unstable_by_key(|s| s.v);
    Plan { beams, spheres, balls }
}

/// The most segments around a beam or sphere, and the fewest while the budget allows.
const MAX_SEGMENTS: u32 = 48;
const MIN_SEGMENTS: u32 = 8;
/// The fewest segments when the budget is tight: a square section.
const FLOOR_SEGMENTS: u32 = 4;

/// Segments around a circle of radius `r` so no chord strays more than `tol` from it, at least `min`; a multiple of
/// four, so a beam or sphere reaches its full radius along its frame's axes, and a sphere has an equator.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "clamped to a few dozen first"
)]
fn segments(r: f32, tol: f32, min: u32) -> u32 {
    // A chord of a circle of radius r spanning angle 2π/n strays r(1 − cos(π/n)) ≈ r π² / 2n² from it.
    let n = (PI * (r / (2.0 * tol)).sqrt()).ceil();
    let n = if n.is_finite() {
        n.clamp(min as f32, MAX_SEGMENTS as f32) as u32
    } else {
        MAX_SEGMENTS
    };
    n.div_ceil(4) * 4
}

/// Triangles of each piece with `n` segments around.
fn side_triangles(n: u32) -> usize {
    2 * n as usize
}
fn butt_triangles(n: u32) -> usize {
    n as usize
}
fn hemisphere_triangles(n: u32) -> usize {
    n as usize * (2 * (n as usize / 4) - 1)
}
fn sphere_triangles(n: u32) -> usize {
    2 * n as usize * (n as usize / 2 - 1)
}

/// The lattice's triangles, added to `prims` as `looks` (one for each kept beam) and `sphere_looks` (one for each
/// sphere) say; returns how many. `budget` is how many it should keep to, coarsening its curves until it does
/// (though never below a square section).
pub(super) fn build(
    l: &Lattice,
    verts: &[[f32; 3]],
    plan: &Plan,
    looks: &[BeamLook],
    sphere_looks: &[(usize, Attr)],
    prims: &mut [Prim],
    budget: usize,
) -> usize {
    // The lattice's extent sets how closely its curves are followed.
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let mut grow = |p: [f32; 3], r: f32| {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k] - r);
            hi[k] = hi[k].max(p[k] + r);
        }
    };
    for &i in &plan.beams {
        let b = &l.beams[i];
        grow(verts[b.v[0] as usize], b.r[0]);
        grow(verts[b.v[1] as usize], b.r[1]);
    }
    for s in &plan.spheres {
        grow(verts[s.v as usize], s.r);
    }
    let diagonal = length(sub(hi, lo));
    if !(diagonal.is_finite() && diagonal > 0.0) {
        return 0;
    }
    let estimate = |tol: f32, min: u32| -> usize {
        let mut sum = 0usize;
        for &i in &plan.beams {
            let b = &l.beams[i];
            let n = segments(b.r[0].max(b.r[1]), tol, min);
            sum = sum.saturating_add(side_triangles(n));
            for cap in b.cap {
                sum = sum.saturating_add(match cap {
                    Cap::Sphere => 0,
                    Cap::Butt => butt_triangles(n),
                    Cap::Hemisphere => hemisphere_triangles(n),
                });
            }
        }
        for s in &plan.spheres {
            sum = sum.saturating_add(sphere_triangles(segments(s.r, tol, min)));
        }
        sum
    };
    let (mut tol, mut min) = (diagonal / 4000.0, MIN_SEGMENTS);
    for _ in 0..64 {
        if estimate(tol, min) <= budget {
            break;
        }
        tol *= 2.0;
        min = FLOOR_SEGMENTS;
    }

    let mut triangles = 0;
    for (k, &i) in plan.beams.iter().enumerate() {
        let b = &l.beams[i];
        let n = segments(b.r[0].max(b.r[1]), tol, min);
        let look = looks[k];
        triangles += beam(prims, verts[b.v[0] as usize], verts[b.v[1] as usize], b, n, &look);
    }
    for (s, &(prim, attr)) in plan.spheres.iter().zip(sphere_looks) {
        let n = segments(s.r, tol, min);
        triangles += sphere(&mut prims[prim], verts[s.v as usize], s.r, n, attr);
    }
    triangles
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: [f32; 3]) -> f32 {
    a[0].hypot(a[1]).hypot(a[2])
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { [0.0, 0.0, 1.0] }
}

/// Two unit vectors that with unit `d` make a right-handed frame (`u × w = d`).
fn basis(d: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let a = d.map(f32::abs);
    let helper = if a[0] <= a[1] && a[0] <= a[2] {
        [1.0, 0.0, 0.0]
    } else if a[1] <= a[2] {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = normalize(cross(d, helper));
    (u, cross(d, u))
}

/// The unit directions of `n` segments around the frame `u`, `w`, counter-clockwise about `u × w`.
#[allow(clippy::cast_precision_loss, reason = "a few dozen at most")]
fn around(u: [f32; 3], w: [f32; 3], n: u32) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let (s, c) = (TAU * i as f32 / n as f32).sin_cos();
            add(scale(u, c), scale(w, s))
        })
        .collect()
}

fn push(p: &mut Prim, pos: [f32; 3], normal: [f32; 3], attr: Attr) -> u32 {
    let i = u32::try_from(p.positions.len()).unwrap_or(u32::MAX);
    p.positions.push(pos);
    p.normals.push(normal);
    match attr {
        Attr::Uv(uv) => p.uvs.push(uv),
        Attr::Color(c) => p.colors.push(c),
        Attr::None => {}
    }
    i
}

/// A beam from `a` to `c` with `n` segments around: its side, and its butt or hemisphere ends (sphere ends are the
/// vertices' spheres). Returns its triangles.
fn beam(prims: &mut [Prim], a: [f32; 3], c: [f32; 3], b: &Beam, n: u32, look: &BeamLook) -> usize {
    let axis = sub(c, a);
    let len = length(axis);
    let d = scale(axis, 1.0 / len);
    let (u, w) = basis(d);
    let dirs = around(u, w, n);
    let (r0, r1) = (b.r[0], b.r[1]);
    // The side's normal leans toward the narrower end by the slope.
    let slope = r0 - r1;
    let inv = 1.0 / len.hypot(slope);
    let p = &mut prims[look.side];
    let base = u32::try_from(p.positions.len()).unwrap_or(u32::MAX);
    for (end, center, r) in [(0, a, r0), (1, c, r1)] {
        for &dir in &dirs {
            let normal = add(scale(dir, len * inv), scale(d, slope * inv));
            push(p, add(center, scale(dir, r)), normal, look.side_attr[end]);
        }
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (a0, b0, c0, e0) = (base + i, base + j, base + n + i, base + n + j);
        p.indices.extend_from_slice(&[a0, b0, e0, a0, e0, c0]);
    }
    let mut triangles = side_triangles(n);
    for (end, center, r, out) in [(0, a, r0, scale(d, -1.0)), (1, c, r1, d)] {
        let (prim, attr) = look.ends[end];
        let p = &mut prims[prim];
        // Directions turn counter-clockwise about d, so clockwise about the first end's outward axis.
        let flip = end == 0;
        match b.cap[end] {
            Cap::Sphere => {}
            Cap::Butt => {
                triangles += butt(p, center, out, &dirs, r, attr, flip);
            }
            Cap::Hemisphere => {
                triangles += hemisphere(p, center, out, &dirs, r, attr, flip);
            }
        }
    }
    triangles
}

/// A flat disk closing a beam's end, facing `out`.
fn butt(p: &mut Prim, center: [f32; 3], out: [f32; 3], dirs: &[[f32; 3]], r: f32, attr: Attr, flip: bool) -> usize {
    let mid = push(p, center, out, attr);
    for &dir in dirs {
        push(p, add(center, scale(dir, r)), out, attr);
    }
    let n = u32::try_from(dirs.len()).unwrap_or(0);
    for i in 0..n {
        let (a, b) = (mid + 1 + i, mid + 1 + (i + 1) % n);
        let t = if flip { [mid, b, a] } else { [mid, a, b] };
        p.indices.extend_from_slice(&t);
    }
    dirs.len()
}

/// A half sphere closing a beam's end, bulging toward `out`; its rim is the beam's ring there, point for point.
#[allow(clippy::cast_precision_loss, reason = "a few dozen at most")]
fn hemisphere(
    p: &mut Prim,
    center: [f32; 3],
    out: [f32; 3],
    dirs: &[[f32; 3]],
    r: f32,
    attr: Attr,
    flip: bool,
) -> usize {
    let n = u32::try_from(dirs.len()).unwrap_or(0);
    let bands = (n / 4).max(1);
    let base = u32::try_from(p.positions.len()).unwrap_or(u32::MAX);
    for k in 0..bands {
        let (s, c) = (FRAC_PI_2 * k as f32 / bands as f32).sin_cos();
        for &dir in dirs {
            let normal = add(scale(dir, c), scale(out, s));
            push(p, add(add(center, scale(dir, r * c)), scale(out, r * s)), normal, attr);
        }
    }
    let pole = push(p, add(center, scale(out, r)), out, attr);
    let mut tri = |t: [u32; 3]| {
        let t = if flip { [t[0], t[2], t[1]] } else { t };
        p.indices.extend_from_slice(&t);
    };
    for k in 0..bands {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = (base + k * n + i, base + k * n + j);
            if k + 1 < bands {
                let (c, e) = (a + n, b + n);
                tri([a, b, e]);
                tri([a, e, c]);
            } else {
                tri([a, b, pole]);
            }
        }
    }
    hemisphere_triangles(n)
}

/// A whole sphere about `center`, its poles along Z.
#[allow(clippy::cast_precision_loss, reason = "a few dozen at most")]
fn sphere(p: &mut Prim, center: [f32; 3], r: f32, n: u32, attr: Attr) -> usize {
    let dirs = around([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], n);
    let z = [0.0, 0.0, 1.0];
    let rings = n / 2;
    let south = push(p, add(center, [0.0, 0.0, -r]), [0.0, 0.0, -1.0], attr);
    let base = south + 1;
    for k in 1..rings {
        let (s, c) = (-FRAC_PI_2 + PI * k as f32 / rings as f32).sin_cos();
        for &dir in &dirs {
            let normal = add(scale(dir, c), scale(z, s));
            push(p, add(center, scale(normal, r)), normal, attr);
        }
    }
    let north = push(p, add(center, [0.0, 0.0, r]), z, attr);
    for i in 0..n {
        let j = (i + 1) % n;
        p.indices.extend_from_slice(&[south, base + j, base + i]);
        for k in 0..rings - 2 {
            let (a, b) = (base + k * n + i, base + k * n + j);
            let (c, e) = (a + n, b + n);
            p.indices.extend_from_slice(&[a, b, e, a, e, c]);
        }
        let top = base + (rings - 2) * n;
        p.indices.extend_from_slice(&[top + i, top + j, north]);
    }
    sphere_triangles(n)
}

/// A closed mesh asked whether points are in it, by the positive fill rule: a ray up from the point crosses its
/// surface more often leaving than entering. Its triangles are binned by where they lie seen from above.
pub(super) struct Clipper {
    tris: Vec<[[f64; 3]; 3]>,
    lo: [f64; 2],
    cell: [f64; 2],
    grid: usize,
    /// Its bounds.
    bounds: [[f32; 3]; 2],
    /// The longest edge a cut looks along for the surface: a cell's size, or as much of its height.
    step: f32,
    /// The triangles over each cell: `items[start[c]..start[c + 1]]`.
    start: Vec<u32>,
    items: Vec<u32>,
}

/// The most triangle-to-cell entries a clipping mesh's grid may hold.
const MAX_GRID_ITEMS: usize = 1 << 24;

impl Clipper {
    /// `None` when there are no triangles to clip by.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "cell numbers, clamped to the grid"
    )]
    pub(super) fn new(verts: &[[f32; 3]], tris: impl Iterator<Item = [u32; 3]>) -> Option<Clipper> {
        let tris: Vec<[[f64; 3]; 3]> = tris
            .filter_map(|t| {
                let [a, b, c] = t.map(|i| verts.get(i as usize).copied().filter(|p| finite(*p)));
                Some([a?, b?, c?].map(|p| p.map(f64::from)))
            })
            .collect();
        if tris.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in tris.iter().flatten() {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let bounds = [lo, hi].map(|b| b.map(|v| v as f32));
        let (lo, hi) = ([lo[0], lo[1]], [hi[0], hi[1], hi[2] - lo[2]]);
        let mut grid = ((tris.len() as f64).sqrt() as usize).clamp(1, 512);
        loop {
            let cell = [0, 1].map(|k| ((hi[k] - lo[k]) / grid as f64).max(f64::MIN_POSITIVE));
            let span = |t: &[[f64; 3]; 3], k: usize| {
                let (mn, mx) = t
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p[k]), b.max(p[k])));
                let at = |v: f64| (((v - lo[k]) / cell[k]) as usize).min(grid - 1);
                (at(mn), at(mx))
            };
            let total: usize = tris
                .iter()
                .map(|t| {
                    let ((x0, x1), (y0, y1)) = (span(t, 0), span(t, 1));
                    (x1 - x0 + 1) * (y1 - y0 + 1)
                })
                .sum();
            if total > MAX_GRID_ITEMS && grid > 1 {
                grid /= 2;
                continue;
            }
            let mut count = vec![0u32; grid * grid + 1];
            let spans: Vec<_> = tris.iter().map(|t| (span(t, 0), span(t, 1))).collect();
            for &((x0, x1), (y0, y1)) in &spans {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        count[y * grid + x + 1] += 1;
                    }
                }
            }
            for i in 1..count.len() {
                count[i] += count[i - 1];
            }
            let mut fill = count.clone();
            let mut items = vec![0u32; total];
            for (t, &((x0, x1), (y0, y1))) in spans.iter().enumerate() {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let c = y * grid + x;
                        items[fill[c] as usize] = u32::try_from(t).unwrap_or(u32::MAX);
                        fill[c] += 1;
                    }
                }
            }
            // Not finer than a thousandth of the mesh's size, however flat it is.
            let size = hi[0] - lo[0] + hi[1] - lo[1] + hi[2];
            let step = cell
                .into_iter()
                .chain([hi[2] / grid as f64])
                .filter(|&v| v > size * 1e-3)
                .fold(size, f64::min);
            return Some(Clipper {
                tris,
                lo,
                cell,
                grid,
                bounds,
                step: step as f32,
                start: count,
                items,
            });
        }
    }

    /// Whether `p` is inside: the surface above it, counted +1 where it faces up and −1 where down, is positive.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "cell numbers, checked against the grid"
    )]
    pub(super) fn inside(&self, p: [f32; 3]) -> bool {
        let [x, y, z] = p.map(f64::from);
        let (fx, fy) = ((x - self.lo[0]) / self.cell[0], (y - self.lo[1]) / self.cell[1]);
        if !(fx >= 0.0 && fy >= 0.0) {
            return false;
        }
        let (cx, cy) = (fx as usize, fy as usize);
        // The far edges of the grid belong to its last cells.
        let at = |c: usize, f: f64| {
            if c == self.grid && f <= self.grid as f64 {
                c - 1
            } else {
                c
            }
        };
        let (cx, cy) = (at(cx, fx), at(cy, fy));
        if cx >= self.grid || cy >= self.grid {
            return false;
        }
        let c = cy * self.grid + cx;
        let mut winding = 0i32;
        for &t in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
            let [a, b, c] = self.tris[t as usize];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area == 0.0 {
                continue;
            }
            // Counter-clockwise seen from above; facing up when it already was.
            let (b, c, up) = if area > 0.0 { (b, c, 1) } else { (c, b, -1) };
            let edge = |p0: [f64; 3], p1: [f64; 3]| (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0]);
            // A point on an edge two triangles share is in just one of them: the one the edge is a left or top
            // edge of.
            let owns = |p0: [f64; 3], p1: [f64; 3]| {
                let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
                dy < 0.0 || (dy == 0.0 && dx < 0.0)
            };
            let (w0, w1, w2) = (edge(b, c), edge(c, a), edge(a, b));
            let covers = |w: f64, p0, p1| w > 0.0 || (w == 0.0 && owns(p0, p1));
            if !(covers(w0, b, c) && covers(w1, c, a) && covers(w2, a, b)) {
                continue;
            }
            let sum = w0 + w1 + w2;
            if sum <= 0.0 {
                continue;
            }
            let height = (w0 * a[2] + w1 * b[2] + w2 * c[2]) / sum;
            if height > z {
                winding += up;
            }
        }
        winding > 0
    }
}

/// Cuts `p` to the side of `clipper` it keeps (`inside` or not): triangles wholly on the other side go, and those
/// across the surface are cut where their edges cross it (found by bisection), their new corners' normals and
/// attributes blended. The cut leaves the beams open there. Triangles within the clipping mesh's bounds are first
/// split until no edge is longer than its step (while `room` more triangles allow), so an edge that passes through
/// the surface and out again is unlikely. Returns the triangles it has afterward.
pub(super) fn clip(p: &mut Prim, clipper: &Clipper, inside: bool, room: usize) -> usize {
    refine(p, clipper, room);
    let keep: Vec<bool> = p.positions.iter().map(|&q| clipper.inside(q) == inside).collect();
    if keep.iter().all(|&k| k) {
        return p.indices.len() / 3;
    }
    let old = std::mem::take(&mut p.indices);
    let mut cuts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut poly: Vec<u32> = Vec::with_capacity(4);
    for t in old.chunks_exact(3) {
        let k = [0, 1, 2].map(|i| keep[t[i] as usize]);
        if k == [true; 3] {
            p.indices.extend_from_slice(t);
            continue;
        }
        if k == [false; 3] {
            continue;
        }
        poly.clear();
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            if k[i] {
                poly.push(a);
            }
            if k[i] != k[(i + 1) % 3] {
                let key = (a.min(b), a.max(b));
                let v = if let Some(&v) = cuts.get(&key) {
                    v
                } else {
                    let (from, to) = if k[i] { (a, b) } else { (b, a) };
                    let v = crossing(p, clipper, inside, from, to);
                    cuts.insert(key, v);
                    v
                };
                poly.push(v);
            }
        }
        for i in 1..poly.len().saturating_sub(1) {
            p.indices.extend_from_slice(&[poly[0], poly[i], poly[i + 1]]);
        }
    }
    compact(p);
    p.indices.len() / 3
}

/// Splits the triangles of `p` that overlap the clipping mesh's bounds at the middle of their longest edge, until
/// none is longer than its step or `room` more triangles have been made. Neighbors split a shared edge at the same
/// new vertex.
fn refine(p: &mut Prim, clipper: &Clipper, room: usize) {
    let step2 = clipper.step * clipper.step;
    if !step2.is_finite() || step2 <= 0.0 {
        return;
    }
    let [blo, bhi] = clipper.bounds;
    let near = |p: &Prim, t: [u32; 3]| {
        (0..3).all(|k| {
            let v = t.map(|i| p.positions[i as usize][k]);
            v[0].min(v[1]).min(v[2]) <= bhi[k] && v[0].max(v[1]).max(v[2]) >= blo[k]
        })
    };
    let mut todo: Vec<[u32; 3]> = p
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .filter(|&t| near(p, t))
        .collect();
    if todo.is_empty() {
        return;
    }
    let mut out: Vec<u32> = p
        .indices
        .chunks_exact(3)
        .filter(|t| !near(p, [t[0], t[1], t[2]]))
        .flatten()
        .copied()
        .collect();
    let mut mids: HashMap<(u32, u32), u32> = HashMap::new();
    let mut left = room;
    while let Some(t) = todo.pop() {
        let len2 = |i: usize| {
            let d = sub(p.positions[t[(i + 1) % 3] as usize], p.positions[t[i] as usize]);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        };
        let e = (0..3).fold(0, |e, i| if len2(i) > len2(e) { i } else { e });
        let long = len2(e) > step2;
        if !long || left == 0 {
            out.extend_from_slice(&t);
            continue;
        }
        left -= 1;
        let (a, b, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
        let key = (a.min(b), a.max(b));
        let m = if let Some(&m) = mids.get(&key) {
            m
        } else {
            let m = blend(p, key.0, key.1, 0.5);
            mids.insert(key, m);
            m
        };
        todo.push([a, m, c]);
        todo.push([m, b, c]);
    }
    p.indices = out;
}

/// A new vertex where the edge from kept vertex `from` to dropped vertex `to` crosses the clipping surface.
fn crossing(p: &mut Prim, clipper: &Clipper, inside: bool, from: u32, to: u32) -> u32 {
    let (a, b) = (p.positions[from as usize], p.positions[to as usize]);
    let at = |s: f32| add(a, scale(sub(b, a), s));
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..32 {
        let mid = f32::midpoint(lo, hi);
        if clipper.inside(at(mid)) == inside {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    blend(p, from, to, lo)
}

/// A new vertex `s` of the way from vertex `from` to `to`, its normal and attributes blended.
fn blend(p: &mut Prim, from: u32, to: u32, s: f32) -> u32 {
    let (f, t) = (from as usize, to as usize);
    let (a, b) = (p.positions[f], p.positions[t]);
    let at = |s: f32| add(a, scale(sub(b, a), s));
    let lerp = |x: f32, y: f32| x + (y - x) * s;
    let i = u32::try_from(p.positions.len()).unwrap_or(u32::MAX);
    p.positions.push(at(s));
    if p.normals.len() > t {
        let (m, n) = (p.normals[f], p.normals[t]);
        p.normals.push(normalize([0, 1, 2].map(|k| lerp(m[k], n[k]))));
    }
    if p.uvs.len() > t {
        let (m, n) = (p.uvs[f], p.uvs[t]);
        p.uvs.push([lerp(m[0], n[0]), lerp(m[1], n[1])]);
    }
    if p.colors.len() > t {
        let (m, n) = (p.colors[f], p.colors[t]);
        p.colors.push([0, 1, 2, 3].map(|k| lerp(m[k], n[k])));
    }
    i
}

/// Drops the vertices no triangle uses.
fn compact(p: &mut Prim) {
    fn keep<T: Copy>(v: &mut Vec<T>, map: &[u32], n: usize) {
        if v.len() != map.len() {
            return;
        }
        let mut out = vec![v[0]; n];
        for (k, &m) in map.iter().enumerate() {
            if m != u32::MAX {
                out[m as usize] = v[k];
            }
        }
        *v = out;
    }
    let mut map = vec![u32::MAX; p.positions.len()];
    let mut next = 0u32;
    for i in &mut p.indices {
        let m = &mut map[*i as usize];
        if *m == u32::MAX {
            *m = next;
            next += 1;
        }
        *i = *m;
    }
    let n = next as usize;
    if n == 0 {
        *p = Prim {
            material: p.material,
            ..Prim::default()
        };
        return;
    }
    keep(&mut p.positions, &map, n);
    keep(&mut p.normals, &map, n);
    keep(&mut p.uvs, &map, n);
    keep(&mut p.colors, &map, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn beam_of(r: [f32; 2], cap: [Cap; 2]) -> Beam {
        Beam {
            v: [0, 1],
            r,
            cap,
            pid: None,
            p: [None, None],
        }
    }

    /// Signed volume and whether every edge is shared by exactly two triangles, once each way (positions welded).
    fn volume_and_closed(p: &Prim) -> (f64, bool) {
        // Welded by value: -0 and 0 are one place.
        let key = |i: u32| p.positions[i as usize].map(|x| (x + 0.0).to_bits());
        let mut edges: HashMap<([u32; 3], [u32; 3]), i32> = HashMap::new();
        let mut vol = 0.0f64;
        for t in p.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| p.positions[i as usize].map(f64::from));
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
            for i in 0..3 {
                let (x, y) = (key(t[i]), key(t[(i + 1) % 3]));
                if x == y {
                    continue;
                }
                *edges.entry((x, y)).or_default() += 1;
            }
        }
        let closed = edges
            .iter()
            .all(|(&(x, y), &n)| n == 1 && edges.get(&(y, x)) == Some(&1));
        (vol, closed)
    }

    fn winding_agrees(p: &Prim) -> bool {
        p.indices.chunks_exact(3).all(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| p.positions[i as usize]);
            let f = cross(sub(b, a), sub(c, a));
            let n = [t[0], t[1], t[2]].map(|i| p.normals[i as usize]);
            n.iter().map(|m| f[0] * m[0] + f[1] * m[1] + f[2] * m[2]).sum::<f32>() >= 0.0
        })
    }

    fn one(cap: [Cap; 2], r: [f32; 2], c: [f32; 3]) -> Prim {
        let look = BeamLook {
            side: 0,
            side_attr: [Attr::None; 2],
            ends: [(0, Attr::None); 2],
        };
        let mut prims = vec![Prim::default()];
        let n = segments(r[0].max(r[1]), 0.001, MIN_SEGMENTS);
        beam(&mut prims, [0.0; 3], c, &beam_of(r, cap), n, &look);
        prims.pop().unwrap()
    }

    #[test]
    fn butt_beam_is_a_closed_frustum() {
        for c in [[0.0, 0.0, 10.0], [3.0, -4.0, 5.0], [10.0, 0.0, 0.0]] {
            let p = one([Cap::Butt; 2], [2.0, 1.0], c);
            let (vol, closed) = volume_and_closed(&p);
            assert!(closed);
            assert!(winding_agrees(&p));
            let h = f64::from(length(c));
            let exact = std::f64::consts::PI * h / 3.0 * (4.0 + 2.0 + 1.0);
            assert!(vol > 0.0 && (vol - exact).abs() / exact < 0.01, "{vol} vs {exact}");
        }
    }

    #[test]
    fn hemisphere_beam_is_closed() {
        let p = one([Cap::Hemisphere, Cap::Butt], [1.0, 1.0], [0.0, 5.0, 0.0]);
        let (vol, closed) = volume_and_closed(&p);
        assert!(closed);
        assert!(winding_agrees(&p));
        // A cylinder and a half ball.
        let exact = std::f64::consts::PI * 5.0 + 2.0 / 3.0 * std::f64::consts::PI;
        assert!((vol - exact).abs() / exact < 0.01, "{vol} vs {exact}");
    }

    #[test]
    fn sphere_is_closed() {
        let mut p = Prim::default();
        sphere(&mut p, [1.0, 2.0, 3.0], 2.0, 32, Attr::None);
        assert_eq!(p.indices.len() / 3, sphere_triangles(32));
        let (vol, closed) = volume_and_closed(&p);
        assert!(closed);
        assert!(winding_agrees(&p));
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 8.0;
        assert!(vol > 0.0 && (vol - exact).abs() / exact < 0.05, "{vol} vs {exact}");
    }

    #[test]
    fn segments_grow_with_radius_and_stay_bounded() {
        assert_eq!(segments(1e-6, 1.0, MIN_SEGMENTS), MIN_SEGMENTS);
        assert_eq!(segments(1e6, 1e-6, MIN_SEGMENTS), MAX_SEGMENTS);
        assert!(segments(1.0, 0.001, MIN_SEGMENTS) > segments(0.1, 0.001, MIN_SEGMENTS));
        assert_eq!(segments(f32::NAN, 1.0, MIN_SEGMENTS), MAX_SEGMENTS);
        assert_eq!(segments(1.0, 0.001, MIN_SEGMENTS) % 4, 0);
    }

    fn lattice(beams: Vec<Beam>) -> Lattice {
        Lattice {
            radius: 1.0,
            minlength: 0.5,
            beams,
            ..Lattice::default()
        }
    }

    #[test]
    fn plan_skips_bad_beams_and_keeps_the_largest_sphere() {
        let verts = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.1, 0.0], [0.0, 10.0, 0.0]];
        let mut l = lattice(vec![
            beam_of([1.0, 2.0], [Cap::Sphere; 2]),
            // Shorter than minlength, to a vertex that isn't there, to itself, with no radius.
            Beam {
                v: [1, 2],
                ..beam_of([1.0, 1.0], [Cap::Sphere; 2])
            },
            Beam {
                v: [0, 9],
                ..beam_of([1.0, 1.0], [Cap::Sphere; 2])
            },
            Beam {
                v: [3, 3],
                ..beam_of([1.0, 1.0], [Cap::Sphere; 2])
            },
            Beam {
                v: [0, 3],
                ..beam_of([f32::NAN, 1.0], [Cap::Sphere; 2])
            },
            Beam {
                v: [0, 3],
                ..beam_of([3.0, 1.0], [Cap::Butt, Cap::Sphere])
            },
        ]);
        l.ball_mode = BallMode::Mixed;
        l.balls.push(Ball {
            v: 3,
            r: 5.0,
            pid: None,
            p: None,
        });
        let plan = plan(&l, &verts);
        assert_eq!(plan.beams, [0, 5]);
        assert_eq!(plan.balls, 1);
        let got: Vec<(u32, f32, From)> = plan.spheres.iter().map(|s| (s.v, s.r, s.from)).collect();
        // Vertex 0's butt end of the last beam makes no sphere; vertex 3's ball is larger than the beam's end.
        assert_eq!(
            got,
            [
                (0, 1.0, From::End(0, 0)),
                (1, 2.0, From::End(0, 1)),
                (3, 5.0, From::Ball(0))
            ]
        );
    }

    #[test]
    fn every_end_gets_the_default_ball() {
        let verts = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let mut l = lattice(vec![beam_of([1.0, 1.0], [Cap::Butt; 2])]);
        l.ball_mode = BallMode::All;
        l.ball_radius = 1.5;
        l.balls.push(Ball {
            v: 1,
            r: 0.5,
            pid: None,
            p: None,
        });
        let plan = plan(&l, &verts);
        let got: Vec<(u32, f32, From)> = plan.spheres.iter().map(|s| (s.v, s.r, s.from)).collect();
        assert_eq!(got, [(0, 1.5, From::Lattice), (1, 0.5, From::Ball(0))]);
    }

    #[test]
    fn budget_coarsens() {
        let verts: Vec<[f32; 3]> = (0..101u16).map(|i| [f32::from(i), 0.0, 0.0]).collect();
        let l = lattice(
            (0..100)
                .map(|i| Beam {
                    v: [i, i + 1],
                    ..beam_of([0.4, 0.4], [Cap::Sphere; 2])
                })
                .collect(),
        );
        let plan = plan(&l, &verts);
        let looks = vec![
            BeamLook {
                side: 0,
                side_attr: [Attr::None; 2],
                ends: [(0, Attr::None); 2]
            };
            plan.beams.len()
        ];
        let spheres = vec![(0, Attr::None); plan.spheres.len()];
        let mut fine = vec![Prim::default()];
        let all = build(&l, &verts, &plan, &looks, &spheres, &mut fine, usize::MAX);
        let mut coarse = vec![Prim::default()];
        let few = build(&l, &verts, &plan, &looks, &spheres, &mut coarse, 5000);
        assert_eq!(all, fine[0].indices.len() / 3);
        assert!(few <= 5000 && few < all, "{few} of {all}");
    }

    #[test]
    fn clipping_cuts_a_beam_at_a_box() {
        // A box from -5 to 10 along X, ±5 across, wound outward.
        let (x0, x1, h) = (-5.0f32, 10.0f32, 5.0f32);
        let verts: Vec<[f32; 3]> = (0..8)
            .map(|i| {
                [
                    if i & 1 == 0 { x0 } else { x1 },
                    if i & 2 == 0 { -h } else { h },
                    if i & 4 == 0 { -h } else { h },
                ]
            })
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let tris = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        let clipper = Clipper::new(&verts, tris).unwrap();
        assert!(clipper.inside([0.0, 0.0, 0.0]));
        assert!(clipper.inside([9.9, 4.9, -4.9]));
        assert!(!clipper.inside([10.1, 0.0, 0.0]));
        assert!(!clipper.inside([0.0, 0.0, 5.1]));
        assert!(!clipper.inside([0.0, 0.0, -5.1]));
        // Exactly over an edge the top's two triangles share.
        assert!(clipper.inside([2.5, 0.0, 0.0]));

        let mut p = one([Cap::Butt; 2], [1.0, 1.0], [30.0, 0.0, 0.0]);
        let before = p.indices.len() / 3;
        let after = clip(&mut p, &clipper, true, 1 << 16);
        assert!(after > 0 && after < before * 64);
        let max_x = p.positions.iter().map(|q| q[0]).fold(f32::MIN, f32::max);
        let min_x = p.positions.iter().map(|q| q[0]).fold(f32::MAX, f32::min);
        assert!((max_x - 10.0).abs() < 1e-3, "{max_x}");
        assert!(min_x.abs() < 1e-6);
        assert!(winding_agrees(&p));
        assert_eq!(p.positions.len(), p.normals.len());

        let mut q = one([Cap::Butt; 2], [1.0, 1.0], [30.0, 0.0, 0.0]);
        clip(&mut q, &clipper, false, 1 << 16);
        let min_x = q.positions.iter().map(|v| v[0]).fold(f32::MAX, f32::min);
        assert!((min_x - 10.0).abs() < 1e-3, "{min_x}");
    }
}
