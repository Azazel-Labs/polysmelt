//! Geometry objects: their most detailed mesh as a glTF mesh, with its morph targets and skin.

use serde_json::{Value, json};

use super::math::{IDENTITY, M4, inverse, mul};
use super::{Ctx, Morphs, expand, referenced};
use crate::Result;
use crate::document::{Builder, Part};
use crate::mesh::{FastMap, Prim};
use crate::num::{f32_of, u32_of};

/// The most morph targets a mesh keeps.
const MAX_TARGETS: usize = 256;
/// The most floats all of a mesh's morph targets hold together.
const MAX_TARGET_FLOATS: usize = 1 << 24;
/// The highest texture coordinate or color set kept.
const MAX_SET: usize = 7;
/// The most bone influences a skin may list.
const MAX_INFLUENCES: usize = 1 << 26;

/// A geometry object as written into the buffer.
pub(super) struct Geom {
    /// Each primitive's JSON (no material) and its index array's material index.
    prims: Vec<(Value, u32)>,
    /// The morph target index of each glTF target, and the base's.
    targets: Vec<u32>,
    base: u32,
    /// The absolute targets among `targets`.
    absolute: Vec<bool>,
    names: Vec<String>,
    skin: Option<usize>,
    extras: Option<Value>,
    /// The triangles it draws, counted again for each node that places it.
    triangles: usize,
}

/// A vertex attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Attrib {
    Position,
    Normal,
    Tangent,
    Bitangent,
    Color,
    Texcoord,
}

/// A vertex array: its attribute, set index, morph target index and structure.
struct Array {
    attrib: Attrib,
    index: usize,
    morph: u32,
    ddl: u32,
}

/// Builds the meshes of the geometry nodes (or of geometry objects placed by themselves) and puts them on their
/// nodes.
pub(super) fn meshes(c: &mut Ctx<'_>) -> Result<()> {
    let t = c.t;
    let mut uses = Vec::new();
    let mut need: FastMap<u32, usize> = FastMap::default();
    for (n, holder) in std::mem::take(&mut c.geometry_nodes) {
        // A geometry object no node places is placed by itself (see `objects`).
        let g = if t.ident(n) == "GeometryObject" {
            Some(n)
        } else {
            referenced(t, n, "ObjectRef").filter(|&g| t.ident(g) == "GeometryObject")
        };
        let Some(g) = g else {
            c.stats.missing_objects += 1;
            continue;
        };
        let mut mats: Vec<(u32, usize)> = Vec::new();
        let mut uvs = 0;
        if c.keep.materials {
            for k in t.kids(n).filter(|&k| t.ident(k) == "MaterialRef") {
                let index = t.prop_num(k, "index").map_or(0, index_of);
                let m = t
                    .reference(k)
                    .and_then(|r| t.resolve(k, r))
                    .and_then(|m| super::material::material(c, m));
                match m {
                    Some(m) => {
                        uvs = uvs.max(c.material_uvs[m]);
                        if !mats.iter().any(|&(i, _)| i == index) {
                            mats.push((index, m));
                        }
                    }
                    None => c.stats.missing_materials += 1,
                }
            }
        } else {
            // What a full read would count, counted.
            for k in t.kids(n).filter(|&k| t.ident(k) == "MaterialRef") {
                match t.reference(k).and_then(|r| t.resolve(k, r)) {
                    Some(m) if t.ident(m) == "Material" => super::material::count_missing(c, m),
                    _ => c.stats.missing_materials += 1,
                }
            }
        }
        let e = need.entry(g).or_insert(0);
        *e = (*e).max(uvs);
        uses.push((n, holder, g, mats));
    }
    for (n, holder, g, mats) in uses {
        if !c.geoms.contains_key(&g) {
            let geom = build(c, g, need.get(&g).copied().unwrap_or(0))?;
            c.geoms.insert(g, geom);
        }
        let Some(Some(geom)) = c.geoms.get(&g) else {
            continue;
        };
        let materials: Vec<Option<usize>> = geom
            .prims
            .iter()
            .map(|&(_, m)| mats.iter().find(|&&(i, _)| i == m).map(|&(_, k)| k))
            .collect();
        let skin = geom.skin;
        c.triangles = c.triangles.saturating_add(geom.triangles);
        let (targets, base, absolute) = (geom.targets.clone(), geom.base, geom.absolute.clone());
        let key = (g, materials);
        let mesh = if let Some(&m) = c.mesh_of.get(&key) {
            m
        } else {
            let mut prims = Vec::with_capacity(geom.prims.len());
            for ((p, _), m) in geom.prims.iter().zip(&key.1) {
                let mut p = p.clone();
                if let Some(m) = m {
                    p["material"] = json!(m);
                }
                prims.push(p);
            }
            let mut mesh = json!({ "primitives": prims });
            if let Some(name) = t.name(g) {
                mesh["name"] = json!(name);
            }
            let mut extras = geom.extras.clone().unwrap_or_else(|| json!({}));
            if !geom.names.is_empty() {
                extras["targetNames"] = json!(geom.names);
            }
            if extras.as_object().is_some_and(|e| !e.is_empty()) {
                mesh["extras"] = extras;
            }
            c.meshes.push(mesh);
            c.mesh_of.insert(key, c.meshes.len() - 1);
            c.meshes.len() - 1
        };
        c.nodes[holder]["mesh"] = json!(mesh);
        if let Some(s) = skin {
            c.nodes[holder]["skin"] = json!(s);
        }
        if !targets.is_empty() {
            let given: Vec<(u32, f64)> = t
                .kids(n)
                .filter(|&k| t.ident(k) == "MorphWeight")
                .filter_map(|k| {
                    let w = t.values(k).0.first().copied().filter(|w| w.is_finite())?;
                    Some((t.prop_num(k, "index").map_or(0, index_of), w))
                })
                .collect();
            let weight = |i: u32| given.iter().rev().find(|&&(k, _)| k == i).map_or(0.0, |&(_, w)| w);
            let weights: Vec<f64> = targets.iter().map(|&k| weight(k)).collect();
            // The base's weight is what's left after the absolute targets' when the weights sum
            // to one, as glTF has it; it's noted when they don't.
            let base_weight = if given.is_empty() {
                if base == 0 { 1.0 } else { 0.0 }
            } else {
                weight(base)
            };
            let rest: f64 = weights.iter().zip(&absolute).filter(|(_, a)| **a).map(|(w, _)| w).sum();
            if (base_weight + rest - 1.0).abs() > 1e-4 {
                c.stats.morph_weights_off += 1;
            }
            c.nodes[holder]["weights"] = json!(weights.iter().map(|&w| f32_of(w)).collect::<Vec<_>>());
            c.morphs.insert(
                n,
                Morphs {
                    holder,
                    targets,
                    weights,
                },
            );
        }
    }
    Ok(())
}

/// A number as an index: 0 when it's negative or not whole.
pub(super) fn index_of(v: f64) -> u32 {
    if v >= 0.0 && v.fract() == 0.0 && v < f64::from(u32::MAX) {
        // In range and whole, checked above.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked to be in range"
        )]
        let i = v as u32;
        i
    } else {
        0
    }
}

/// A vertex index: `None` when it isn't a whole number below `n`.
fn vertex(v: f64, n: usize) -> Option<u32> {
    #[allow(clippy::cast_precision_loss, reason = "vertex counts are far below 2^52")]
    let ok = v >= 0.0 && v.fract() == 0.0 && v < n as f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked to be in range"
    )]
    ok.then_some(v as u32)
}

/// The vertex arrays of `mesh`; unknown attributes are counted.
fn arrays(c: &mut Ctx<'_>, mesh: u32) -> Vec<Array> {
    let t = c.t;
    let mut out = Vec::new();
    for k in t.kids(mesh).filter(|&k| t.ident(k) == "VertexArray") {
        let Some(full) = t.prop_str(k, "attrib") else {
            c.stats.unknown_arrays += 1;
            continue;
        };
        // An older form names the set in the attribute: "texcoord[1]".
        let (name, inline) = match full.split_once('[') {
            Some((name, rest)) => (name, rest.trim_end_matches(']').parse::<usize>().ok()),
            None => (full, None),
        };
        let attrib = match name {
            "position" => Attrib::Position,
            "normal" => Attrib::Normal,
            "tangent" => Attrib::Tangent,
            "bitangent" => Attrib::Bitangent,
            "color" => Attrib::Color,
            "texcoord" => Attrib::Texcoord,
            _ => {
                c.stats.unknown_arrays += 1;
                continue;
            }
        };
        let index = t
            .prop_num(k, "index")
            .map(|v| index_of(v) as usize)
            .or(inline)
            .unwrap_or(0);
        if index > MAX_SET {
            c.stats.unknown_arrays += 1;
            continue;
        }
        let morph = t.prop_num(k, "morph").map_or(0, index_of);
        out.push(Array {
            attrib,
            index,
            morph,
            ddl: k,
        });
    }
    out
}

/// The vertex array for attribute `a` and set `index` of morph target `morph`: its own, or
/// the one of the lowest morph target index that has one.
fn pick(arrays: &[Array], a: Attrib, index: usize, morph: u32) -> Option<u32> {
    let mut best: Option<&Array> = None;
    for x in arrays.iter().filter(|x| x.attrib == a && x.index == index) {
        if x.morph == morph {
            return Some(x.ddl);
        }
        if best.is_none_or(|b| x.morph < b.morph) {
            best = Some(x);
        }
    }
    best.map(|x| x.ddl)
}

/// A vertex array's values and their width, when it holds `n` vertices.
fn values<'t>(c: &Ctx<'t>, k: u32, n: usize) -> Option<(&'t [f64], usize)> {
    let (v, w) = c.t.values(k);
    let w = if w == 0 { 1 } else { w as usize };
    (w <= 4 && v.len() == n * w).then_some((v, w))
}

/// The attribute values of morph target `morph` in the form glTF stores them.
#[derive(Default)]
struct Attrs {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
}

/// Positions (and unit normals, when `normals`) of morph target `morph`.
fn attrs(c: &mut Ctx<'_>, arrays: &[Array], morph: u32, n: usize, normals: bool) -> Option<Attrs> {
    let mut out = Attrs::default();
    let (v, w) = values(c, pick(arrays, Attrib::Position, 0, morph)?, n)?;
    out.positions = v.chunks_exact(w).map(|p| vec3(p, 0.0)).collect();
    if normals && let Some(k) = pick(arrays, Attrib::Normal, 0, morph) {
        match values(c, k, n) {
            Some((v, w)) => out.normals = v.chunks_exact(w).map(unit_normal).collect(),
            None => c.stats.bad_arrays += 1,
        }
    }
    Some(out)
}

/// The first three of `p`, `fill` for those it lacks.
fn vec3(p: &[f64], fill: f64) -> [f32; 3] {
    [0, 1, 2].map(|k| f32_of(p.get(k).copied().unwrap_or(fill)))
}

/// A normal at unit length; +Z when it has none.
fn unit_normal(p: &[f64]) -> [f32; 3] {
    let v = [0, 1, 2].map(|k| p.get(k).copied().unwrap_or(0.0));
    super::math::unit(v).map_or([0.0, 0.0, 1.0], |u| u.map(f32_of))
}

/// What one index array draws: triangles, line segments or points.
#[derive(Default)]
struct Draws {
    triangles: Vec<u32>,
    lines: Vec<u32>,
    points: Vec<u32>,
}

impl Draws {
    fn is_empty(&self) -> bool {
        self.triangles.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }
}

/// Assembles the primitives of kind `primitive` from indices `idx` (on `n` vertices); bad
/// indices are counted and what they'd be in dropped.
fn assemble(
    primitive: &str,
    idx: impl Iterator<Item = f64>,
    n: usize,
    restart: Option<f64>,
    cw: bool,
    bad: &mut usize,
) -> Draws {
    let mut d = Draws::default();
    let mut v = |x: f64| {
        let i = vertex(x, n);
        if i.is_none() {
            *bad += 1;
        }
        i
    };
    match primitive {
        "points" => d.points.extend(idx.filter_map(&mut v)),
        "lines" | "triangles" | "quads" => {
            let k = match primitive {
                "lines" => 2,
                "triangles" => 3,
                _ => 4,
            };
            let mut buf = [0u32; 4];
            let mut at = 0;
            let mut ok = true;
            for x in idx {
                match v(x) {
                    Some(i) => buf[at] = i,
                    None => ok = false,
                }
                at += 1;
                if at < k {
                    continue;
                }
                if ok {
                    match k {
                        2 => d.lines.extend_from_slice(&buf[..2]),
                        3 if cw => d.triangles.extend_from_slice(&[buf[0], buf[2], buf[1]]),
                        3 => d.triangles.extend_from_slice(&buf[..3]),
                        _ if cw => d
                            .triangles
                            .extend_from_slice(&[buf[0], buf[2], buf[1], buf[0], buf[3], buf[2]]),
                        _ => d
                            .triangles
                            .extend_from_slice(&[buf[0], buf[1], buf[2], buf[0], buf[2], buf[3]]),
                    }
                }
                at = 0;
                ok = true;
            }
        }
        // Strips: a restart index, or a bad one, starts a new strip.
        "line_strip" | "triangle_strip" => {
            let tri = primitive == "triangle_strip";
            let (mut a, mut b): (Option<u32>, Option<u32>) = (None, None);
            let mut parity = false;
            for x in idx {
                if restart == Some(x) {
                    (a, b, parity) = (None, None, false);
                    continue;
                }
                let Some(i) = v(x) else {
                    (a, b, parity) = (None, None, false);
                    continue;
                };
                if !tri {
                    if let Some(p) = b {
                        d.lines.extend_from_slice(&[p, i]);
                    }
                    b = Some(i);
                    continue;
                }
                if let (Some(p), Some(q)) = (a, b) {
                    // Every other triangle of a strip is wound the other way.
                    let (p0, p1) = if parity == cw { (p, q) } else { (q, p) };
                    if p != q && q != i && p != i {
                        d.triangles.extend_from_slice(&[p0, p1, i]);
                    }
                    parity = !parity;
                }
                (a, b) = (b, Some(i));
            }
        }
        _ => {}
    }
    d
}

/// An index accessor for `idx` on `n` vertices.
fn index_accessor(b: &mut Builder, idx: Vec<u32>, n: usize) -> usize {
    let count = idx.len();
    if n < usize::from(u16::MAX) {
        let v: Vec<u16> = idx.iter().map(|&i| u16::try_from(i).unwrap_or(0)).collect();
        b.accessor(Part::u16(v), 5123, count, "SCALAR", Some(34963))
    } else {
        b.accessor(Part::u32(idx), 5125, count, "SCALAR", Some(34963))
    }
}

/// Writes geometry object `g` (padding its texture coordinate sets to `need_uv`).
#[allow(clippy::too_many_lines, reason = "one pass over a mesh's arrays")]
fn build(c: &mut Ctx<'_>, g: u32, need_uv: usize) -> Result<Option<Geom>> {
    let t = c.t;
    let meshes: Vec<u32> = t.kids(g).filter(|&k| t.ident(k) == "Mesh").collect();
    let Some(&mesh) = meshes.iter().min_by_key(|&&m| t.prop_num(m, "lod").map_or(0, index_of)) else {
        c.stats.missing_objects += 1;
        return Ok(None);
    };
    c.stats.dropped_lods += meshes.len() - 1;
    let primitive = t.prop_str(mesh, "primitive").unwrap_or("triangles");
    if !matches!(
        primitive,
        "points" | "lines" | "line_strip" | "triangles" | "triangle_strip" | "quads"
    ) {
        c.stats.unknown_primitives += 1;
        return Ok(None);
    }
    let arrays = arrays(c, mesh);
    let mut morphs: Vec<u32> = arrays.iter().map(|a| a.morph).collect();
    morphs.sort_unstable();
    morphs.dedup();
    let Some(&base) = morphs.first() else {
        c.stats.bad_arrays += 1;
        return Ok(None);
    };
    let Some(pos) = pick(&arrays, Attrib::Position, 0, base) else {
        c.stats.bad_arrays += 1;
        return Ok(None);
    };
    let (pv, pw) = t.values(pos);
    let pw = if pw == 0 { 1 } else { pw as usize };
    if pw > 4 || pv.is_empty() || pv.len() % pw != 0 {
        c.stats.bad_arrays += 1;
        return Ok(None);
    }
    let n = pv.len() / pw;
    if u32::try_from(n).is_err() {
        c.stats.bad_arrays += 1;
        return Ok(None);
    }

    let mut p = Prim {
        positions: pv.chunks_exact(pw).map(|p| vec3(p, 0.0)).collect(),
        ..Prim::default()
    };
    let mut bad_arrays = 0;
    let mut get = |a: Attrib, index: usize| {
        let k = pick(&arrays, a, index, base)?;
        let got = values(c, k, n);
        if got.is_none() {
            bad_arrays += 1;
        }
        got
    };
    if let Some((v, w)) = get(Attrib::Normal, 0) {
        p.normals = v.chunks_exact(w).map(unit_normal).collect();
    }
    if !p.normals.is_empty()
        && let Some((v, w)) = get(Attrib::Tangent, 0)
    {
        let bitangents = get(Attrib::Bitangent, 0);
        p.tangents = v
            .chunks_exact(w)
            .enumerate()
            .map(|(i, x)| {
                let d = vec3(x, 0.0);
                let sign = if w == 4 {
                    x[3]
                } else if let Some((bv, bw)) = bitangents {
                    // The handedness: whether the bitangent agrees with normal × tangent.
                    let nrm = p.normals[i].map(f64::from);
                    let cr = super::math::cross(nrm, d.map(f64::from));
                    super::math::dot(
                        cr,
                        [0, 1, 2].map(|k| bv[i * bw..i * bw + bw].get(k).copied().unwrap_or(0.0)),
                    )
                } else {
                    1.0
                };
                [d[0], d[1], d[2], if sign < 0.0 { -1.0 } else { 1.0 }]
            })
            .collect();
    }
    // Texture coordinate sets, flipped to glTF's top-left origin, with any gap (and any set a
    // material reads that the mesh lacks) filled with zeros so the sets keep their numbers.
    let highest_uv = arrays
        .iter()
        .filter(|a| a.attrib == Attrib::Texcoord)
        .map(|a| a.index + 1)
        .max()
        .unwrap_or(0);
    let uv_count = highest_uv.max(need_uv.min(MAX_SET + 1));
    for set in 0..uv_count {
        let uv: Vec<[f32; 2]> = match get(Attrib::Texcoord, set) {
            Some((v, w)) => v
                .chunks_exact(w)
                .map(|x| [f32_of(x[0]), f32_of(1.0 - x.get(1).copied().unwrap_or(0.0))])
                .collect(),
            None => vec![[0.0; 2]; n],
        };
        if set == 0 {
            p.uvs = uv;
        } else {
            p.uv_sets.resize(set, Vec::new());
            p.uv_sets.push(uv);
        }
    }
    let highest_color = arrays
        .iter()
        .filter(|a| a.attrib == Attrib::Color)
        .map(|a| a.index + 1)
        .max()
        .unwrap_or(0);
    for set in 0..highest_color {
        let col: Vec<[f32; 4]> = match get(Attrib::Color, set) {
            Some((v, w)) => v
                .chunks_exact(w)
                .map(|x| {
                    let ch = |k: usize| x.get(k).copied().unwrap_or(if w == 1 { x[0] } else { 0.0 });
                    let rgb = c.metric.color([ch(0), ch(1), ch(2)]);
                    let a = if w == 4 { x[3] } else { 1.0 };
                    [f32_of(rgb[0]), f32_of(rgb[1]), f32_of(rgb[2]), f32_of(a)]
                })
                .collect(),
            None => vec![[1.0; 4]; n],
        };
        if set == 0 {
            p.colors = col;
        } else {
            p.color_sets.resize(set, Vec::new());
            p.color_sets.push(col);
        }
    }
    c.stats.bad_arrays += bad_arrays;

    // The index arrays' primitives.
    let mut draws: Vec<(Draws, u32)> = Vec::new();
    let mut bad = 0;
    let index_arrays: Vec<u32> = t.kids(mesh).filter(|&k| t.ident(k) == "IndexArray").collect();
    if index_arrays.is_empty() {
        #[allow(clippy::cast_precision_loss, reason = "vertex counts are far below 2^52")]
        let all = (0..n).map(|i| i as f64);
        draws.push((assemble(primitive, all, n, None, false, &mut bad), 0));
    }
    for &k in &index_arrays {
        let material = t.prop_num(k, "material").map_or(0, index_of);
        let restart = t.prop_num(k, "restart");
        let cw = t.prop_str(k, "front") == Some("cw");
        let (v, _) = t.values(k);
        draws.push((
            assemble(primitive, v.iter().copied(), n, restart, cw, &mut bad),
            material,
        ));
    }
    c.stats.bad_primitives += bad;
    draws.retain(|(d, _)| !d.is_empty());
    if draws.is_empty() {
        return Ok(None);
    }
    let triangles: usize = draws.iter().map(|(d, _)| d.triangles.len() / 3).sum();
    c.built_triangles += triangles;
    ensure!(
        c.built_triangles <= c.limit,
        LimitExceeded,
        "more than {} triangles",
        c.limit
    );
    if draws.iter().any(|(d, _)| !d.lines.is_empty() || !d.points.is_empty()) {
        c.lines_or_points = true;
    }

    let skin = if c.keep.skins { skin(c, mesh, n, &mut p) } else { None };

    let has_normals = !p.normals.is_empty();
    // The primitives: the first index array's written with the vertex arrays, the others on
    // the same arrays with their own indices.
    let mut iter = draws.into_iter();
    let (first, first_material) = iter.next().unwrap_or_default();
    p.indices = first.triangles;
    p.lines = first.lines;
    p.points = first.points;
    if let Some((s, _)) = &skin {
        let count = c.skins[*s]["joints"].as_array().map_or(0, Vec::len);
        c.b.joints = Some((0, count));
    }
    let written = c.b.primitives(p);
    c.b.joints = None;
    // What only a skin or morph targets use is written after what's always written, so a read
    // that skips them makes the same mesh.
    let skin = skin.map(|(s, ibms)| {
        c.skins[s]["inverseBindMatrices"] = json!(super::floats(&mut c.b, ibms, "MAT4", false));
        s
    });
    let mut prims = Vec::new();
    let attributes = written.first().map(|w| w["attributes"].clone());
    for w in written {
        prims.push((w, first_material));
    }
    if let Some(attributes) = attributes {
        for (d, material) in iter {
            for (mode, idx) in [(4, d.triangles), (1, d.lines), (0, d.points)] {
                if idx.is_empty() {
                    continue;
                }
                let mut j = json!({ "attributes": attributes.clone(), "indices": index_accessor(&mut c.b, idx, n) });
                if mode != 4 {
                    j["mode"] = json!(mode);
                }
                prims.push((j, material));
            }
        }
    }
    // Morph targets: every morph target index but the base's, as differences from the base
    // (absolute targets) or from their own base (relative ones).
    let mut targets: Vec<u32> = if c.keep.morph_targets {
        morphs[1..].to_vec()
    } else {
        Vec::new()
    };
    let floats_each = n * 6;
    let most = (MAX_TARGET_FLOATS / floats_each.max(1)).min(MAX_TARGETS);
    if targets.len() > most {
        c.stats.dropped_targets += targets.len() - most;
        targets.truncate(most);
    }
    let morph_of = |k: u32| {
        t.kids(g)
            .filter(|&m| t.ident(m) == "Morph")
            .find(|&m| t.prop_num(m, "index").map_or(0, index_of) == k)
    };
    let mut target_json = Vec::new();
    let mut names = Vec::new();
    let mut absolute = Vec::new();
    let mut kept = Vec::new();
    let mut cache: FastMap<u32, Attrs> = FastMap::default();
    for &k in &targets {
        let m = morph_of(k);
        let rel = m.and_then(|m| t.prop_num(m, "base")).map(index_of);
        let from = rel.unwrap_or(base);
        if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(from) {
            let Some(a) = attrs(c, &arrays, from, n, has_normals) else {
                c.stats.dropped_targets += 1;
                continue;
            };
            slot.insert(a);
        }
        let Some(a) = attrs(c, &arrays, k, n, has_normals) else {
            c.stats.dropped_targets += 1;
            continue;
        };
        let r = &cache[&from];
        let dp: Vec<[f32; 3]> = a
            .positions
            .iter()
            .zip(&r.positions)
            .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
            .collect();
        let mut tj = json!({ "POSITION": c.b.positions(dp) });
        if has_normals && a.normals.len() == n && r.normals.len() == n {
            let dn: Vec<f32> = a
                .normals
                .iter()
                .zip(&r.normals)
                .flat_map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
                .collect();
            tj["NORMAL"] = json!(c.b.accessor(Part::f32(dn), 5126, n, "VEC3", Some(34962)));
        }
        target_json.push(tj);
        names.push(
            m.and_then(|m| t.kid(m, "Name"))
                .and_then(|k| t.string(k))
                .map_or_else(|| format!("morph{k}"), str::to_owned),
        );
        absolute.push(rel.is_none());
        kept.push(k);
    }

    if !target_json.is_empty() {
        for (j, _) in &mut prims {
            j["targets"] = json!(target_json);
        }
    }
    let mut extras = serde_json::Map::new();
    for key in ["visible", "shadow", "motion_blur"] {
        if let Some(v) = t.prop_bool(g, key) {
            extras.insert(key.into(), json!(v));
        }
    }
    Ok(Some(Geom {
        prims,
        targets: kept,
        base,
        absolute,
        names,
        skin,
        extras: (c.keep.extras && !extras.is_empty()).then_some(Value::Object(extras)),
        triangles,
    }))
}

/// A mesh's skin as a glTF skin, its bone influences put on `p`: its index, and its inverse
/// bind matrices for the caller to write. `None` when it has none, or it can't be read
/// (counted).
fn skin(c: &mut Ctx<'_>, mesh: u32, n: usize, p: &mut Prim) -> Option<(usize, Vec<f32>)> {
    let t = c.t;
    let s = t.kid(mesh, "Skin")?;
    let got = read_skin(c, s, n, p);
    if got.is_none() {
        c.stats.bad_skins += 1;
        p.skin_joints = Vec::new();
        p.skin_weights = Vec::new();
    }
    got
}

fn read_skin(c: &mut Ctx<'_>, s: u32, n: usize, p: &mut Prim) -> Option<(usize, Vec<f32>)> {
    let t = c.t;
    let bind = match t.kid(s, "Transform") {
        Some(k) => {
            let (v, w) = t.values(k);
            let w = if w == 0 { v.len() } else { w as usize };
            ([16, 12, 9].contains(&w) && v.len() >= w && v[..w].iter().all(|x| x.is_finite())).then(|| expand(v, w))?
        }
        None => IDENTITY,
    };
    let skel = t.kid(s, "Skeleton")?;
    let refs_at = t.kid(skel, "BoneRefArray")?;
    let refs = t.data_of(refs_at).map(|d| t.refs(d))?;
    let bones = refs.len();
    if bones == 0 {
        return None;
    }
    let (poses, _) = t.values(t.kid(skel, "Transform")?);
    if poses.len() != bones * 16 || poses.iter().any(|x| !x.is_finite()) {
        return None;
    }
    // Each bone's joint: bones naming the same node share one.
    let mut joints: Vec<usize> = Vec::new();
    let mut ibms: Vec<f32> = Vec::new();
    let mut slot = Vec::with_capacity(bones);
    for (b, r) in refs.iter().enumerate() {
        let node = t.resolve(refs_at, r).and_then(|d| c.node_of.get(&d).copied())?;
        if let Some(j) = joints.iter().position(|&x| x == node) {
            slot.push(u32_of(j));
            continue;
        }
        let mut pose: M4 = [0.0; 16];
        pose.copy_from_slice(&poses[b * 16..b * 16 + 16]);
        let mut ibm = mul(&inverse(&pose)?, &bind);
        // glTF wants an affine matrix.
        (ibm[3], ibm[7], ibm[11], ibm[15]) = (0.0, 0.0, 0.0, 1.0);
        ibms.extend(ibm.map(f32_of));
        slot.push(u32_of(joints.len()));
        joints.push(node);
    }
    let counts = t.values(t.kid(s, "BoneCountArray")?).0;
    let indices = t.values(t.kid(s, "BoneIndexArray")?).0;
    let weights = t.values(t.kid(s, "BoneWeightArray")?).0;
    if counts.len() != n {
        return None;
    }
    let mut total = 0usize;
    for &k in counts {
        total = total.checked_add(vertex(k, MAX_INFLUENCES)? as usize)?;
    }
    if total > MAX_INFLUENCES || indices.len() != total || weights.len() != total {
        return None;
    }
    p.skin_joints = Vec::with_capacity(n);
    p.skin_weights = Vec::with_capacity(n);
    let mut at = 0;
    let mut dropped = 0;
    for &k in counts {
        // Checked above to be a whole number in range.
        let k = vertex(k, MAX_INFLUENCES).map_or(0, |k| k as usize);
        // The four strongest influences, strongest first.
        let mut top = [(0u32, 0.0f64); 4];
        for i in at..at + k {
            let w = weights[i];
            let Some(b) = vertex(indices[i], bones) else {
                dropped += 1;
                continue;
            };
            if !(w > 0.0 && w.is_finite()) {
                continue;
            }
            let j = slot[b as usize];
            // Two influences of one joint add up.
            if let Some(e) = top.iter_mut().find(|e| e.1 > 0.0 && e.0 == j) {
                e.1 += w;
            } else if w > top[3].1 {
                if top[3].1 > 0.0 {
                    dropped += 1;
                }
                top[3] = (j, w);
            } else {
                dropped += 1;
                continue;
            }
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        at += k;
        p.skin_joints.push(top.map(|e| e.0));
        p.skin_weights.push(top.map(|e| f32_of(e.1)));
    }
    c.stats.dropped_influences += dropped;
    c.skins.push(json!({ "joints": joints }));
    Some((c.skins.len() - 1, ibms))
}
