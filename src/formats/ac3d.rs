//! AC3D (`.ac`): a text format of materials, then a tree of objects (the world, groups, polygon objects and
//! lights). An object has a name, a placement relative to its parent (`loc` and the 3×3 `rot`), a texture, its
//! vertices and its surfaces: polygons, open and closed lines, and triangle strips, each with a material and
//! texture coordinates.
//!
//! Each object becomes a node, the world's children the scene's roots. A surface's material, the object's texture
//! (placed by `texrep` and `texoff`) and whether the surface has two sides make a glTF material: the color, with
//! the texture over it as AC3D modulates it, the emissive color, the transparency as alpha, and the specular color
//! and shininess as the highlight (the classic material every reader reads one way, `legacy_material`); a surface
//! naming a material the file doesn't have takes the default one. Shaded surfaces are smoothed across shared vertices up to the object's crease angle (AC3D's 61°
//! when it gives none); flat ones keep their faces' normals. Polygons are triangulated, lines drawn as lines.
//! What has no glTF slot (the ambient color, the subdivision level, `data`, `url`, hidden, locked and
//! folded flags, lights) stays in `extras`.
//!
//! AC3D works in OpenGL's space, Y up and right-handed, with polygons counterclockwise from the front, as glTF
//! does: the source's up axis is +Y and its handedness right (declared: the program's space is every file's).
//! Its units are whatever the model's are, unknown, and lengths are kept as written. Colors are sRGB.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::crease;
use super::stream::{self, f32_from};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep};
use crate::images::Images;
use crate::legacy_material::{self, Phong, Rgb};
use crate::material::{Full, Tex};
use crate::mesh::{FastMap, Prim};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};
use crate::vec3;

/// AC3D's crease angle, in degrees, for an object that gives none.
const DEFAULT_CREASE: f32 = 61.0;
/// Surface types: the low four bits of its flags. Anything else is a polygon.
const CLOSED_LINE: u32 = 1;
const LINE: u32 = 2;
const STRIP: u32 = 4;
/// Surface flags: smooth shading, and two sides.
const SHADED: u32 = 0x10;
const TWO_SIDED: u32 = 0x20;
/// Ear clipping costs the cube of a polygon's corners; past this much in all, polygons are fanned.
const CLIP_BUDGET: u64 = 1 << 31;

/// A material line.
#[derive(Clone, Debug)]
struct Mat {
    name: String,
    rgb: [f32; 3],
    amb: [f32; 3],
    emis: [f32; 3],
    spec: [f32; 3],
    shi: f32,
    trans: f32,
    /// Other values it gives, raw.
    other: Map<String, Value>,
}

impl Default for Mat {
    fn default() -> Self {
        Self {
            name: String::new(),
            rgb: [1.0; 3],
            amb: [0.2; 3],
            emis: [0.0; 3],
            spec: [0.5; 3],
            shi: 10.0,
            trans: 0.0,
            other: Map::new(),
        }
    }
}

/// A surface: its corners are `refs[start..start + len]`.
#[derive(Clone, Copy, Debug)]
struct Surf {
    flags: u32,
    mat: u32,
    start: u32,
    len: u32,
}

#[derive(Debug, Default)]
struct Obj {
    kind: String,
    name: Option<String>,
    data: Option<String>,
    texture: Option<String>,
    /// Further textures (AC3D 8 can give several), with their kinds.
    more_textures: Vec<(String, Option<String>)>,
    texrep: Option<[f32; 2]>,
    texoff: Option<[f32; 2]>,
    rot: Option<[f32; 9]>,
    loc: Option<[f32; 3]>,
    url: Option<String>,
    crease: Option<f32>,
    subdiv: Option<u32>,
    hidden: bool,
    locked: bool,
    folded: bool,
    verts: Vec<[f32; 3]>,
    /// Each surface corner's vertex and texture coordinates.
    refs: Vec<(u32, [f32; 2])>,
    surfs: Vec<Surf>,
    kids: Vec<usize>,
    /// Surfaces dropped for naming a vertex that isn't there.
    bad_surfs: usize,
}

struct Model {
    version: Option<char>,
    mats: Vec<Mat>,
    objs: Vec<Obj>,
    roots: Vec<usize>,
    /// The file ended before every object it promised.
    truncated: bool,
    unknown_lines: usize,
}

/// The text as lines, with the byte offset after each, so `data` blocks can be taken by their length.
struct Lines<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Lines<'a> {
    /// The next line, without its line break; `None` at the end.
    fn next_line(&mut self) -> Option<&'a [u8]> {
        stream::next_line(self.d, &mut self.at)
    }

    /// The next line with something on it.
    fn next_full(&mut self) -> Option<&'a [u8]> {
        loop {
            let l = self.next_line()?;
            if l.iter().any(|c| !c.is_ascii_whitespace()) {
                return Some(l);
            }
        }
    }

    /// The next `n` bytes, as they are (or what's left).
    fn take(&mut self, n: usize) -> &'a [u8] {
        let end = self.at.saturating_add(n).min(self.d.len());
        let b = &self.d[self.at..end];
        self.at = end;
        b
    }
}

/// The words of a line, a quoted string being one word (without its quotes).
fn words(line: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < line.len() {
        if line[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if line[i] == b'"' {
            let rest = &line[i + 1..];
            let n = rest.iter().position(|&c| c == b'"').unwrap_or(rest.len());
            out.push(&rest[..n]);
            i += n + 2;
        } else {
            let n = line[i..]
                .iter()
                .position(u8::is_ascii_whitespace)
                .unwrap_or(line.len() - i);
            out.push(&line[i..i + n]);
            i += n;
        }
    }
    out
}

/// An `f32` as the decimal it was written as (0.2, not 0.20000000298023224).
fn clean(x: f32) -> Value {
    json!(x.to_string().parse::<f64>().unwrap_or(0.0))
}

fn text(b: &[u8]) -> String {
    crate::bytes::text(b).into_owned()
}

/// A finite number.
fn num(b: Option<&[u8]>) -> Option<f32> {
    f32_from(b?).filter(|v| v.is_finite())
}

/// A whole number, decimal or `0x` hexadecimal.
fn int(b: Option<&[u8]>) -> Option<u32> {
    match b? {
        [b'0', b'x' | b'X', h @ ..] if !h.is_empty() => h.iter().try_fold(0u32, |n, &c| {
            let d = char::from(c).to_digit(16)?;
            n.checked_mul(16)?.checked_add(d)
        }),
        b => stream::int(b),
    }
}

/// The words of a line without quotes: the fast way for vertex and corner lines.
fn fields(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty())
}

fn nums<const N: usize>(w: &[&[u8]], from: usize) -> Option<[f32; N]> {
    let mut out = [0.0; N];
    for (k, o) in out.iter_mut().enumerate() {
        *o = num(w.get(from + k).copied())?;
    }
    Some(out)
}

/// A material's key-value list: `rgb 1 1 1  amb ...`, or one key per line in a `MAT` block.
fn mat_values(m: &mut Mat, w: &[&[u8]]) {
    let mut i = 0;
    while i < w.len() {
        let key = w[i];
        let three = nums::<3>(w, i + 1);
        match key {
            b"rgb" | b"amb" | b"emis" | b"spec" => {
                if let Some(v) = three {
                    *match key {
                        b"rgb" => &mut m.rgb,
                        b"amb" => &mut m.amb,
                        b"emis" => &mut m.emis,
                        _ => &mut m.spec,
                    } = v;
                }
                i += 4;
            }
            b"shi" => {
                m.shi = num(w.get(i + 1).copied()).unwrap_or(m.shi);
                i += 2;
            }
            b"trans" => {
                m.trans = num(w.get(i + 1).copied()).unwrap_or(m.trans);
                i += 2;
            }
            _ => {
                // An unknown key, with the numbers after it.
                let vals: Vec<Value> = w[i + 1..]
                    .iter()
                    .map_while(|v| f32_from(v).filter(|f| f.is_finite()))
                    .map(clean)
                    .collect();
                let n = vals.len();
                if m.other.len() < 64 {
                    let v = match n {
                        0 => Value::Bool(true),
                        1 => vals[0].clone(),
                        _ => Value::Array(vals),
                    };
                    m.other.insert(text(key), v);
                }
                i += 1 + n;
            }
        }
    }
}

fn parse(data: &[u8]) -> Result<Model> {
    let data = data.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(data);
    ensure!(data.starts_with(b"AC3D"), UnknownFormat, "not an AC3D file");
    let mut lines = Lines { d: data, at: 0 };
    let head = lines.next_line().unwrap_or_default();
    let mut m = Model {
        version: head.get(4).map(|&c| char::from(c)),
        mats: Vec::new(),
        objs: Vec::new(),
        roots: Vec::new(),
        truncated: false,
        unknown_lines: 0,
    };
    // Objects still owed children: (object, how many).
    let mut owed: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<usize> = None;
    while let Some(line) = lines.next_full() {
        let w = words(line);
        let Some(&key) = w.first() else { continue };
        match key {
            b"MATERIAL" => {
                let mut mat = Mat {
                    name: w.get(1).map(|n| text(n)).unwrap_or_default(),
                    ..Mat::default()
                };
                mat_values(&mut mat, w.get(2..).unwrap_or_default());
                m.mats.push(mat);
            }
            b"MAT" => {
                let mut mat = Mat {
                    name: w.get(1).map(|n| text(n)).unwrap_or_default(),
                    ..Mat::default()
                };
                while let Some(l) = lines.next_full() {
                    let w = words(l);
                    if w.first() == Some(&&b"ENDMAT"[..]) {
                        break;
                    }
                    mat_values(&mut mat, &w);
                }
                m.mats.push(mat);
            }
            b"OBJECT" => {
                while owed.last().is_some_and(|&(_, n)| n == 0) {
                    owed.pop();
                }
                let i = m.objs.len();
                m.objs.push(Obj {
                    kind: w.get(1).map(|k| text(k)).unwrap_or_default(),
                    ..Obj::default()
                });
                match owed.last_mut() {
                    Some((p, n)) => {
                        *n -= 1;
                        m.objs[*p].kids.push(i);
                    }
                    None => m.roots.push(i),
                }
                cur = Some(i);
            }
            b"kids" => {
                let n = int(w.get(1).copied()).unwrap_or(0) as usize;
                if let Some(c) = cur.take()
                    && n > 0
                {
                    owed.push((c, n));
                }
            }
            _ => {
                let Some(c) = cur else {
                    m.unknown_lines += 1;
                    continue;
                };
                object_line(&mut m.objs[c], &w, &mut lines, &mut m.unknown_lines);
            }
        }
    }
    // Children still owed, or an object whose `kids` line never came: the file stops early.
    m.truncated = owed.iter().any(|&(_, n)| n > 0) || cur.is_some();
    Ok(m)
}

/// One line of an object's, taking the lines that follow `numvert`, `numsurf` and `data` with it.
fn object_line(o: &mut Obj, w: &[&[u8]], lines: &mut Lines<'_>, unknown: &mut usize) {
    let s = |i: usize| w.get(i).map(|v| text(v));
    match w[0] {
        b"name" => o.name = s(1),
        b"data" => {
            let n = int(w.get(1).copied()).unwrap_or(0) as usize;
            let d = lines.take(n);
            o.data = Some(text(d));
        }
        b"texture" => match (&o.texture, s(1)) {
            (None, Some(t)) => o.texture = Some(t),
            (Some(_), Some(t)) if o.more_textures.len() < 16 => o.more_textures.push((t, s(2))),
            _ => {}
        },
        b"texrep" => o.texrep = nums(w, 1),
        b"texoff" => o.texoff = nums(w, 1),
        b"rot" => o.rot = nums(w, 1),
        b"loc" => o.loc = nums(w, 1),
        b"url" => o.url = s(1),
        b"crease" => o.crease = num(w.get(1).copied()),
        b"subdiv" => o.subdiv = int(w.get(1).copied()),
        b"hidden" => o.hidden = true,
        b"locked" => o.locked = true,
        b"folded" => o.folded = true,
        b"numvert" => {
            let n = int(w.get(1).copied()).unwrap_or(0) as usize;
            // Each vertex takes six bytes at least, so what's left bounds the count.
            o.verts.reserve(n.min(lines.d.len().saturating_sub(lines.at) / 6));
            for _ in 0..n {
                let back = lines.at;
                let Some(l) = lines.next_full() else { break };
                let mut f = fields(l);
                let v = (|| Some([num(f.next())?, num(f.next())?, num(f.next())?]))();
                let Some(v) = v else {
                    // The vertices ran short: what's there belongs to the object.
                    lines.at = back;
                    break;
                };
                o.verts.push(v);
            }
        }
        b"numsurf" => {
            let n = int(w.get(1).copied()).unwrap_or(0) as usize;
            let mut done = 0;
            let mut flags = 0;
            let mut mat = 0;
            while done < n {
                let back = lines.at;
                let Some(l) = lines.next_full() else { break };
                let mut f = fields(l);
                let (key, value) = (f.next(), int(f.next()));
                match key {
                    Some(b"SURF") => flags = value.unwrap_or(0),
                    Some(b"mat") => mat = value.unwrap_or(0),
                    Some(b"refs") => {
                        let count = value.unwrap_or(0) as usize;
                        let start = o.refs.len();
                        let mut ok = true;
                        for _ in 0..count {
                            let back = lines.at;
                            let Some(l) = lines.next_full() else {
                                ok = false;
                                break;
                            };
                            let mut f = fields(l);
                            let Some(v) = int(f.next()) else {
                                // Not a corner: the corners ran short.
                                lines.at = back;
                                ok = false;
                                break;
                            };
                            ok &= (v as usize) < o.verts.len();
                            let uv = num(f.next()).zip(num(f.next())).map_or([0.0; 2], |(u, v)| [u, v]);
                            o.refs.push((v, uv));
                        }
                        done += 1;
                        if !ok || count == 0 {
                            o.refs.truncate(start);
                            o.bad_surfs += 1;
                        } else if flags & 0xF == STRIP {
                            strip(o, flags, mat, start);
                        } else {
                            o.surfs.push(Surf {
                                flags,
                                mat,
                                start: u32_of(start),
                                len: u32_of(count),
                            });
                        }
                        flags = 0;
                        mat = 0;
                    }
                    // Something else: the surfaces ran short; it's the object's, or the next one's.
                    _ => {
                        lines.at = back;
                        break;
                    }
                }
            }
        }
        _ => *unknown += 1,
    }
}

/// A triangle strip's corners, from `start` in the object's, as triangles of their own, each turned to face the
/// way the first does; triangles that repeat a vertex are left out.
fn strip(o: &mut Obj, flags: u32, mat: u32, start: usize) {
    let corners = o.refs.split_off(start);
    let flags = flags & !0xF;
    for k in 2..corners.len() {
        let t = if k % 2 == 0 {
            [k - 2, k - 1, k]
        } else {
            [k - 1, k - 2, k]
        }
        .map(|c| corners[c]);
        if t[0].0 == t[1].0 || t[1].0 == t[2].0 || t[0].0 == t[2].0 {
            continue;
        }
        let at = o.refs.len();
        o.refs.extend(t);
        o.surfs.push(Surf {
            flags,
            mat,
            start: u32_of(at),
            len: 3,
        });
    }
}

/// The normal of a polygon (Newell's method), up when it has none.
fn face_normal(pos: &[[f32; 3]], corners: &[(u32, [f32; 2])]) -> [f32; 3] {
    let n = vec3::newell(corners.iter().map(|&(a, _)| pos[a as usize].map(f64::from)));
    vec3::unit(n.map(f32_of)).unwrap_or([0.0, 1.0, 0.0])
}

/// A material: the AC3D material's index (or none for AC3D's default), the texture's image, and two sides.
type MatKey = (Option<u32>, Option<usize>, bool);

/// A vertex: its AC3D vertex, normal and texture coordinates, as bits.
type Key = (u32, [u32; 3], [u32; 2]);

/// How much ear clipping a polygon of `n` corners costs; small ones are free.
fn clip_cost(n: u32) -> u64 {
    let n = u64::from(n);
    if n > 8 { n * n * n } else { 0 }
}

/// The object's surfaces, a primitive per material: `mats` holds each surface's material (or the one material of
/// an object with vertices but no surfaces, which become points), and `clip` how much ear clipping it may spend.
fn build(o: &Obj, mats: &[usize], textured: bool, mut clip: u64) -> Vec<Prim> {
    if o.surfs.is_empty() {
        return vec![Prim {
            material: mats.first().copied().unwrap_or(0),
            positions: o.verts.clone(),
            points: (0..u32_of(o.verts.len())).collect(),
            ..Prim::default()
        }];
    }
    let polygon = |s: &Surf| !matches!(s.flags & 0xF, LINE | CLOSED_LINE) && s.len >= 3;
    let corners = |s: &Surf| &o.refs[s.start as usize..(s.start + s.len) as usize];
    let face: Vec<[f32; 3]> = o
        .surfs
        .iter()
        .map(|s| {
            if polygon(s) {
                face_normal(&o.verts, corners(s))
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect();
    let cos = o.crease.unwrap_or(DEFAULT_CREASE).clamp(0.0, 180.0).to_radians().cos();
    let shaded = |s: &Surf| polygon(s) && s.flags & SHADED != 0;
    let smooth = if o.surfs.iter().any(shaded) {
        let spans: Vec<(usize, usize)> = o.surfs.iter().map(|s| (s.start as usize, s.len as usize)).collect();
        let points: Vec<u32> = o.refs.iter().map(|r| r.0).collect();
        crease::corner_normals(
            o.verts.len(),
            &spans,
            &points,
            &face,
            |p| shaded(&o.surfs[p]).then_some(cos),
            |a, b| shaded(&o.surfs[a]) && shaded(&o.surfs[b]),
        )
    } else {
        Vec::new()
    };
    let rep = o.texrep.unwrap_or([1.0; 2]);
    let off = o.texoff.unwrap_or([0.0; 2]);

    let mut by_mat: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for (i, &m) in mats.iter().enumerate().take(o.surfs.len()) {
        let k = *slot.entry(m).or_insert_with(|| {
            by_mat.push((m, Vec::new()));
            by_mat.len() - 1
        });
        by_mat[k].1.push(i);
    }
    let mut prims = Vec::new();
    let (mut idx, mut pos, mut tris) = (Vec::new(), Vec::new(), Vec::new());
    for (material, surfs) in by_mat {
        let mut prim = Prim {
            material,
            ..Prim::default()
        };
        let lit = surfs.iter().any(|&i| polygon(&o.surfs[i]));
        let mut index: FastMap<Key, u32> = FastMap::default();
        for i in surfs {
            let s = o.surfs[i];
            let cs = corners(&s);
            let kind = s.flags & 0xF;
            idx.clear();
            pos.clear();
            for (k, &(v, uv)) in cs.iter().enumerate() {
                let normal = if !lit {
                    [0.0; 3]
                } else if shaded(&s) {
                    smooth[s.start as usize + k]
                } else {
                    face[i]
                };
                let uv = if textured {
                    [uv[0] * rep[0] + off[0], 1.0 - (uv[1] * rep[1] + off[1])]
                } else {
                    [0.0; 2]
                };
                let key = (v, normal.map(f32::to_bits), uv.map(f32::to_bits));
                let at = *index.entry(key).or_insert_with(|| {
                    prim.positions.push(o.verts[v as usize]);
                    if lit {
                        prim.normals.push(normal);
                    }
                    if textured {
                        prim.uvs.push(uv);
                    }
                    u32_of(prim.positions.len() - 1)
                });
                idx.push(at);
                pos.push(o.verts[v as usize]);
            }
            match kind {
                LINE | CLOSED_LINE => {
                    for k in 1..idx.len() {
                        prim.lines.extend([idx[k - 1], idx[k]]);
                    }
                    if kind == CLOSED_LINE && idx.len() > 2 {
                        prim.lines.extend([idx[idx.len() - 1], idx[0]]);
                    }
                }
                _ if idx.len() == 1 => prim.points.push(idx[0]),
                _ if idx.len() == 2 => prim.lines.extend_from_slice(&idx),
                _ => {
                    tris.clear();
                    let cost = clip_cost(s.len);
                    if cost > clip {
                        tris.extend((1..s.len - 1).map(|k| [0, k, k + 1]));
                    } else {
                        clip -= cost;
                        crate::mesh::triangulate(&pos, &mut tris);
                    }
                    prim.indices.extend(tris.iter().flatten().map(|&k| idx[k as usize]));
                }
            }
        }
        if !prim.is_empty() {
            prims.push(prim);
        }
    }
    prims
}

/// The glTF material for an AC3D one, with its texture and sides; for surfaces whose material the file doesn't
/// have (`None`), the default one, white under a texture.
fn material(m: Option<&Mat>, tex: Option<Tex>, two_sided: bool, extras: bool) -> Full {
    let Some(m) = m else {
        return Full {
            base_color_tex: tex.clone(),
            double_sided: two_sided,
            ..legacy_material::default_material(tex.is_some()).into()
        };
    };
    // `MATERIAL name rgb r g b amb r g b emis r g b spec r g b shi n trans t`: OpenGL's material, whose
    // shininess (0 to 128) is the Blinn-Phong exponent.
    let mut phong = Phong::new(Rgb::Srgb(m.rgb));
    phong.opacity = 1.0 - m.trans;
    phong.emissive = Rgb::Srgb(m.emis);
    phong.specular = Some(Rgb::Srgb(m.spec));
    phong.exponent = Some(m.shi.clamp(0.0, 128.0));
    let mut full = Full {
        base_color_tex: tex,
        double_sided: two_sided,
        ..phong.pbr().full(&m.name)
    };
    if extras {
        let mut e = json!({ "ambient": m.amb.map(clean), "specular": m.spec.map(clean), "shininess": clean(m.shi), "transparency": clean(m.trans) });
        for (k, v) in &m.other {
            e[k] = v.clone();
        }
        full.extras = Some(json!({ "ac3d": e }));
    }
    full
}

/// The node's matrix (glTF's column order, which is AC3D's), or `None` when `rot` is the identity.
fn matrix(o: &Obj) -> Option<[f32; 16]> {
    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let r = o
        .rot
        .filter(|r| r.iter().zip(IDENTITY).any(|(a, b)| a.to_bits() != b.to_bits()))?;
    let t = o.loc.unwrap_or_default();
    Some([
        r[0], r[1], r[2], 0.0, r[3], r[4], r[5], 0.0, r[6], r[7], r[8], 0.0, t[0], t[1], t[2], 1.0,
    ])
}

pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let m = parse(&data)?;
    let dir = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir);
    let textures = keep.materials && keep.textures;
    let mut image_of: HashMap<String, Option<usize>> = HashMap::new();
    let mut dropped_textures = 0usize;
    let mut materials: Vec<Full> = Vec::new();
    let mut material_at: HashMap<MatKey, usize> = HashMap::new();
    let mut b = Builder::default();
    let mut meshes: Vec<Value> = Vec::new();
    let mut mesh_of: Vec<Option<usize>> = vec![None; m.objs.len()];
    let bad_surfs: usize = m.objs.iter().map(|o| o.bad_surfs).sum();

    // First, in order, each object's materials, the triangles it will make and the ear clipping it may spend.
    let mut jobs = Vec::new();
    let (mut clip, mut triangles, mut corners) = (CLIP_BUDGET, 0usize, 0usize);
    for (oi, o) in m.objs.iter().enumerate() {
        if o.surfs.is_empty() && o.verts.is_empty() {
            continue;
        }
        // A texture that isn't there is counted whether or not textures are kept, as a full read
        // counts it.
        let image = o.texture.as_ref().and_then(|name| {
            *image_of.entry(name.clone()).or_insert_with(|| {
                let Some(path) = assets.find(name, dir) else {
                    dropped_textures += 1;
                    return None;
                };
                let found = if textures {
                    images.file(assets, path, name)
                } else {
                    None
                };
                if found.is_none() && textures {
                    dropped_textures += 1;
                }
                found
            })
        });
        // Surfaces keep their own primitives without materials too: those are dropped afterward.
        let mut material_of = |mat: u32, flags: u32| -> usize {
            let mat = (mat as usize).lt(&m.mats.len()).then_some(mat);
            let two = flags & TWO_SIDED != 0;
            *material_at.entry((mat, image, two)).or_insert_with(|| {
                let src = mat.map(|i| &m.mats[i as usize]);
                materials.push(material(src, image.map(Tex::new), two, keep.extras));
                materials.len() - 1
            })
        };
        // Vertices without surfaces are points, in the default material.
        let mats: Vec<usize> = if o.surfs.is_empty() {
            vec![material_of(u32::MAX, 0)]
        } else {
            o.surfs.iter().map(|s| material_of(s.mat, s.flags)).collect()
        };
        let mut need = 0u64;
        for s in o
            .surfs
            .iter()
            .filter(|s| !matches!(s.flags & 0xF, LINE | CLOSED_LINE) && s.len >= 3)
        {
            triangles += s.len as usize - 2;
            need = need.saturating_add(clip_cost(s.len));
        }
        ensure!(
            triangles <= input.limit,
            LimitExceeded,
            "more than {} triangles",
            input.limit
        );
        let take = need.min(clip);
        clip -= take;
        corners += o.refs.len() + o.verts.len();
        jobs.push((oi, mats, take));
    }

    // Then the objects' primitives, on threads when there's enough to share.
    let threads = if corners > 200_000 { keep.thread_count() } else { 1 };
    let built = crate::keep::par_map_with(
        &jobs,
        threads,
        || (),
        |(), (oi, mats, clip)| {
            let o = &m.objs[*oi];
            build(o, mats, o.texture.is_some(), *clip)
        },
    );
    let mut triangles = 0;
    for ((oi, _, _), prims) in jobs.iter().zip(built) {
        let o = &m.objs[*oi];
        let mut primitives = Vec::new();
        for prim in prims {
            triangles += prim.indices.len() / 3;
            let material = prim.material;
            for mut j in b.primitives(prim) {
                j["material"] = json!(material);
                primitives.push(j);
            }
        }
        if primitives.is_empty() {
            continue;
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(n) = &o.name {
            mesh["name"] = json!(n);
        }
        meshes.push(mesh);
        mesh_of[*oi] = Some(meshes.len() - 1);
    }
    if meshes.is_empty() {
        return Err(crate::Error::nothing_left(
            m.truncated,
            "AC3D",
            "this AC3D file has no geometry",
        ));
    }

    // A plain world (no placement or surfaces of its own) gives way to its children.
    let plain = |i: usize| {
        let o = &m.objs[i];
        o.kind == "world" && o.rot.is_none() && o.loc.is_none() && mesh_of[i].is_none()
    };
    let mut roots: Vec<usize> = Vec::new();
    for &r in &m.roots {
        if plain(r) {
            roots.extend(&m.objs[r].kids);
        } else {
            roots.push(r);
        }
    }
    // Nodes in the order they're reached, skipping plain worlds.
    let mut node_of: Vec<Option<usize>> = vec![None; m.objs.len()];
    let mut order: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = roots.iter().rev().copied().collect();
    while let Some(i) = stack.pop() {
        if node_of[i].is_some() {
            continue;
        }
        node_of[i] = Some(order.len());
        order.push(i);
        stack.extend(m.objs[i].kids.iter().rev());
    }
    let mut nodes: Vec<Value> = Vec::with_capacity(order.len());
    for &i in &order {
        let o = &m.objs[i];
        let mut node = json!({ "name": o.name.clone().unwrap_or_else(|| o.kind.clone()) });
        if let Some(mx) = matrix(o) {
            node["matrix"] = json!(mx);
        } else if let Some(t) = o.loc.filter(|t| t.iter().any(|&c| c != 0.0)) {
            node["translation"] = json!(t);
        }
        if let Some(mesh) = mesh_of[i] {
            node["mesh"] = json!(mesh);
        }
        let kids: Vec<usize> = o.kids.iter().filter_map(|&k| node_of[k]).collect();
        if !kids.is_empty() {
            node["children"] = json!(kids);
        }
        if keep.extras {
            let mut e = json!({ "type": o.kind });
            let mut put = |k: &str, v: Value| e[k] = v;
            if let Some(d) = &o.data {
                put("data", json!(d));
            }
            if let Some(u) = &o.url {
                put("url", json!(u));
            }
            if let Some(t) = &o.texture {
                put("texture", json!(t));
            }
            if !o.more_textures.is_empty() {
                put("moreTextures", json!(o.more_textures));
            }
            if let Some(r) = o.texrep {
                put("texrep", json!(r.map(clean)));
            }
            if let Some(r) = o.texoff {
                put("texoff", json!(r.map(clean)));
            }
            if let Some(c) = o.crease {
                put("crease", clean(c));
            }
            if let Some(s) = o.subdiv {
                put("subdiv", json!(s));
            }
            for (k, on) in [("hidden", o.hidden), ("locked", o.locked), ("folded", o.folded)] {
                if on {
                    put(k, json!(true));
                }
            }
            node["extras"] = json!({ "ac3d": e });
        }
        nodes.push(node);
    }
    let roots: Vec<usize> = roots.iter().filter_map(|&r| node_of[r]).collect();

    let mut source = Map::new();
    let info = SourceInfo {
        format: Some("AC3D".into()),
        version: m.version.map(|v| v.to_string()),
        up_axis: Fact::Declared(Axis::PosY),
        handedness: Fact::Declared(Handedness::Right),
        ..SourceInfo::default()
    };
    info.write(&mut source);
    let mut source = Value::Object(source);
    let lights = m.objs.iter().filter(|o| o.kind == "light").count();
    let subdivided = m.objs.iter().filter(|o| o.subdiv.is_some_and(|s| s > 0)).count();
    for (key, n) in [
        ("droppedSurfaces", bad_surfs),
        ("droppedTextures", dropped_textures),
        ("lights", lights),
        ("subdividedObjects", subdivided),
        ("unknownLines", m.unknown_lines),
    ] {
        if n > 0 {
            source[key] = json!(n);
        }
    }
    if m.truncated {
        source["truncated"] = json!(true);
    }
    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(triangles));
    facts.insert("objects".into(), json!(m.objs.len()));
    facts.insert("materials".into(), json!(m.mats.len()));

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": roots }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    b.materials(&materials, images.list, &mut json);
    // Whether lines and points are drawn is the caller's choice (`Keep::lines_and_points`), which the load
    // records on the document after this.
    Ok(b.finish(json, facts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;
    use crate::formats::Input;
    use crate::mesh::Mesh;
    use crate::mesh::srgb_to_linear;

    fn png() -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    fn read(text: &str, assets: &Memory) -> Result<Document> {
        load(Input::memory(text.as_bytes(), "t.ac"), assets, Keep::ALL)
    }

    fn flat(doc: &Document) -> Mesh {
        doc.flatten(Pose::Rest).unwrap().unwrap()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// Every corner position of the mesh's triangles.
    fn corners(m: &Mesh) -> Vec<[f32; 3]> {
        m.prims
            .iter()
            .flat_map(|p| p.indices.iter().map(|&i| p.positions[i as usize]))
            .collect()
    }

    /// A unit square in the XY plane facing +Z, at `x`, as an object.
    fn square(name: &str, flags: &str, extra: &str) -> String {
        format!(
            "OBJECT poly\nname \"{name}\"\n{extra}numvert 4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\nnumsurf 1\nSURF {flags}\nmat 0\nrefs 4\n\
             0 0 0\n1 1 0\n2 1 1\n3 0 1\nkids 0\n"
        )
    }

    const MAT: &str = "MATERIAL \"red\" rgb 1 0 0  amb 0.2 0.2 0.2  emis 0 0 0  spec 0.5 0.5 0.5  shi 64  trans 0\n";

    #[test]
    fn hierarchy_and_placement() {
        // A group turned a quarter about Y (X goes to -Z) and moved, holding a square moved along its own X.
        let text = format!(
            "AC3Db\n{MAT}OBJECT world\nkids 1\nOBJECT group\nname \"arm\"\nrot 0 0 -1  0 1 0  1 0 0\nloc 0 2 0\nkids 1\n{}",
            square("hand", "0x00", "loc 3 0 0\n")
        );
        let doc = read(&text, &Memory::new()).unwrap();
        let nodes = doc.json()["nodes"].as_array().unwrap();
        // The plain world gives way: the group is the root.
        assert_eq!(nodes.len(), 2);
        assert_eq!(doc.json()["scenes"][0]["nodes"], json!([0]));
        assert_eq!(nodes[0]["name"], "arm");
        assert_eq!(nodes[0]["children"], json!([1]));
        assert_eq!(nodes[1]["translation"], json!([3.0, 0.0, 0.0]));
        assert_eq!(nodes[1]["extras"]["ac3d"]["type"], "poly");
        let m = flat(&doc);
        // The square's corner (1, 1, 0) lands at R·(4, 1, 0) + (0, 2, 0) = (0, 3, -4).
        assert!(
            corners(&m).iter().any(|&c| close(c, [0.0, 3.0, -4.0])),
            "{:?}",
            corners(&m)
        );
        // Its face, +Z in the file, now faces +X.
        let n = m.prims[0].normals[0];
        assert!(close(n, [1.0, 0.0, 0.0]), "{n:?}");
        assert_eq!(m.triangles(), 2);
        // A world that's placed is kept.
        let placed = format!(
            "AC3Db\n{MAT}OBJECT world\nloc 0 1 0\nkids 1\n{}",
            square("s", "0x00", "")
        );
        let doc = read(&placed, &Memory::new()).unwrap();
        assert_eq!(doc.json()["nodes"][0]["extras"]["ac3d"]["type"], "world");
        assert_eq!(doc.json()["nodes"][0]["children"], json!([1]));
    }

    #[test]
    fn materials_and_textures() {
        let text = format!(
            "AC3Dc\n{MAT}MATERIAL \"glass\" rgb 0.5 0.5 1 amb 0 0 0 emis 1 0.5 0 spec 0 0 0 shi 0 trans 0.25 glow 3\n\
             OBJECT world\nkids 2\n{}{}",
            square("plain", "0x10", "texture \"skin.png\"\ntexrep 2 2\ntexoff 0.5 0\n"),
            square("both", "0x20", "").replace("mat 0", "mat 1")
        );
        let assets = Memory::new().with("skin.png", png());
        let doc = read(&text, &assets).unwrap();
        let m = flat(&doc);
        assert_eq!(m.materials.len(), 2);
        assert_eq!(m.images.len(), 1);
        let red = &m.materials[0];
        assert_eq!(red.name, "red");
        assert_eq!(red.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(red.color_image, Some(0));
        assert!(!red.double_sided);
        assert_eq!(red.metallic, 0.0);
        assert!(red.roughness > 0.3 && red.roughness < 0.5, "{}", red.roughness);
        let glass = &m.materials[1];
        assert!(glass.double_sided);
        assert!((glass.color[3] - 0.75).abs() < 1e-6);
        assert!((glass.color[0] - srgb_to_linear(0.5)).abs() < 1e-6);
        assert_eq!(glass.emissive[0], 1.0);
        assert_eq!(glass.roughness, 1.0);
        assert_eq!(glass.color_image, None);
        assert_eq!(doc.json()["materials"][1]["extras"]["ac3d"]["glow"], 3.0);
        assert_eq!(doc.json()["materials"][1]["extras"]["ac3d"]["transparency"], 0.25);
        // Texture coordinates are repeated, offset and flipped for glTF.
        let textured = m.prims.iter().find(|p| p.material == 0).unwrap();
        let mut uvs = textured.uvs.clone();
        uvs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(uvs, vec![[0.5, -1.0], [0.5, 1.0], [2.5, -1.0], [2.5, 1.0]]);
        // The untextured square has none.
        assert_eq!(
            m.prims.iter().find(|p| p.material == 1).unwrap().uvs,
            Vec::<[f32; 2]>::new()
        );
        // A missing texture is counted, and the material keeps its color.
        let doc = read(&text, &Memory::new()).unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["droppedTextures"], 1);
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("AC3D"));
        assert_eq!(info.up_axis, Fact::Declared(Axis::PosY));
        assert_eq!(info.handedness, Fact::Declared(Handedness::Right));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert!(info.unit.is_none());
        assert_eq!(info.converted, crate::source::Converted::default());
        assert_eq!(flat(&doc).materials[0].color_image, None);
    }

    #[test]
    fn lines_strips_and_points() {
        let text = "AC3Db\nOBJECT poly\nnumvert 5\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n2 0 0\nnumsurf 5\n\
             SURF 0x02\nrefs 3\n0 0 0\n1 0 0\n2 0 0\n\
             SURF 0x01\nrefs 3\n0 0 0\n1 0 0\n2 0 0\n\
             SURF 0x14\nrefs 4\n0 0 0\n1 0 0\n2 0 0\n3 0 0\n\
             SURF 0x00\nrefs 1\n4 0 0\n\
             SURF 0x00\nrefs 2\n3 0 0\n4 0 0\n\
             kids 0\nOBJECT poly\nname \"dots\"\nnumvert 2\n5 5 5\n6 6 6\nkids 0\n";
        let doc = read(text, &Memory::new()).unwrap();
        let m = flat(&doc);
        let lines: usize = m.prims.iter().map(|p| p.lines.len() / 2).sum();
        let points: usize = m.prims.iter().map(|p| p.points.len()).sum();
        // Open: 2 segments, closed: 3, the two-corner polygon: 1.
        assert_eq!(lines, 6);
        // The one-corner polygon and the surfaceless object's two vertices.
        assert_eq!(points, 3);
        // The strip makes two triangles, both facing +Z.
        assert_eq!(m.triangles(), 2);
        for c in m.prims.iter().flat_map(|p| {
            p.indices
                .chunks_exact(3)
                .map(move |t| [t[0], t[1], t[2]].map(|i| p.positions[i as usize]))
        }) {
            let (u, v) = (
                [c[1][0] - c[0][0], c[1][1] - c[0][1]],
                [c[2][0] - c[0][0], c[2][1] - c[0][1]],
            );
            assert!(u[0] * v[1] - u[1] * v[0] > 0.0, "{c:?}");
        }
        assert_eq!(doc.json()["meshes"][1]["name"], "dots");
        // Lines and points are drawn or not as the caller asks, whatever the file holds.
        for on in [true, false] {
            let keep = Keep::ALL.with_lines_and_points(on);
            let doc = crate::formats::load_bytes(text.as_bytes(), "t.ac", "ac", &Memory::new(), keep).unwrap();
            assert_eq!(doc.lines_and_points(), on);
            let m = flat(&doc);
            assert_eq!(m.prims.iter().any(|p| !p.lines.is_empty() || !p.points.is_empty()), on);
            assert_eq!(m.triangles(), 2);
        }
    }

    #[test]
    fn crease_angle_smooths() {
        // Two squares folded 90 degrees along x = 1, shaded.
        let fold = |crease: &str| {
            format!(
                "AC3Db\n{MAT}OBJECT poly\n{crease}numvert 6\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n1 0 -1\n1 1 -1\nnumsurf 2\n\
                 SURF 0x10\nmat 0\nrefs 4\n0 0 0\n1 0 0\n2 0 0\n3 0 0\n\
                 SURF 0x10\nmat 0\nrefs 4\n1 0 0\n4 0 0\n5 0 0\n2 0 0\nkids 0\n"
            )
        };
        let normal_at = |m: &Mesh, at: [f32; 3]| -> Vec<[f32; 3]> {
            let p = &m.prims[0];
            p.positions
                .iter()
                .zip(&p.normals)
                .filter(|(q, _)| close(**q, at))
                .map(|(_, n)| *n)
                .collect()
        };
        // AC3D's default 61 degrees keeps the fold sharp.
        let sharp = flat(&read(&fold(""), &Memory::new()).unwrap());
        assert_eq!(normal_at(&sharp, [1.0, 0.0, 0.0]).len(), 2);
        assert_eq!(sharp.prims[0].positions.len(), 8);
        // A wider angle smooths across it.
        let soft = flat(&read(&fold("crease 120\n"), &Memory::new()).unwrap());
        let n = normal_at(&soft, [1.0, 0.0, 0.0]);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(n.len(), 1);
        assert!(close(n[0], [h, 0.0, h]), "{n:?}");
        // Flat surfaces stay flat whatever the angle.
        let flat_text = fold("crease 179\n").replace("SURF 0x10", "SURF 0x00");
        assert_eq!(
            flat(&read(&flat_text, &Memory::new()).unwrap()).prims[0]
                .positions
                .len(),
            8
        );
    }

    #[test]
    fn odd_but_valid_files() {
        // A byte order mark, CRLF lines, a MAT block, a data block holding words that look like keys, and blank lines.
        let text = format!(
            "\u{feff}AC3Db\r\nMAT \"blue\"\r\nrgb 0 0 1\r\ntrans 0.5\r\nENDMAT\r\n\r\nOBJECT world\r\nkids 1\r\n{}",
            square("s", "0x30", "data 15\nOBJECT poly\nkids 9\n").replace('\n', "\r\n")
        );
        let doc = read(&text, &Memory::new()).unwrap();
        let m = flat(&doc);
        assert_eq!(m.triangles(), 2);
        assert_eq!(m.materials[0].name, "blue");
        assert!(m.materials[0].double_sided);
        assert!((m.materials[0].color[3] - 0.5).abs() < 1e-6);
        assert_eq!(doc.json()["nodes"][0]["extras"]["ac3d"]["data"], "OBJECT poly\r\nki");
        // No materials: the default one, as every reader gives faces without one.
        let bare = format!("AC3Db\n{}", square("s", "0x00", "").replace("mat 0\n", ""));
        let m = flat(&read(&bare, &Memory::new()).unwrap());
        let d = &m.materials[0];
        assert_eq!(
            (d.name.as_str(), d.color, d.roughness),
            ("default", crate::mesh::GRAY, 0.85)
        );
        // A concave polygon is triangulated, not fanned.
        let ell = "AC3Db\nOBJECT poly\nnumvert 6\n0 0 0\n2 0 0\n2 1 0\n1 1 0\n1 2 0\n0 2 0\nnumsurf 1\nSURF 0\nrefs 6\n\
                   1 0 0\n2 0 0\n3 0 0\n4 0 0\n5 0 0\n0 0 0\nkids 0\n";
        let m = flat(&read(ell, &Memory::new()).unwrap());
        let mut area = 0.0;
        for t in corners(&m).chunks_exact(3) {
            area += ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) / 2.0;
        }
        assert!((area - 3.0).abs() < 1e-6, "{area}");
    }

    #[test]
    fn rejects_what_is_not_a_model() {
        assert!(read("solid cube\n", &Memory::new()).is_err());
        assert!(read("AC3Db\nMATERIAL \"\" rgb 1 1 1\nOBJECT world\nkids 0\n", &Memory::new()).is_err());
    }

    #[test]
    fn hostile_files_never_panic() {
        let assets = Memory::new().with("skin.png", png());
        // Counts far past what's there.
        let wild = "AC3Db\nOBJECT poly\nnumvert 4000000000\n0 0 0\n1 0 0\n0 1 0\nnumsurf 99999999\nSURF 0x10\nmat 77\n\
                    refs 4000000000\n0 0 0\n1 0 0\n2 0 0\nkids 99999999\n";
        let doc = read(wild, &assets).unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["truncated"], true);
        // Vertex indices past the vertices drop their surface.
        let bad = "AC3Db\nOBJECT poly\nnumvert 3\n0 0 0\n1 0 0\n0 1 0\nnumsurf 2\nSURF 0\nrefs 3\n0 0 0\n1 0 0\n9 0 0\n\
                   SURF 0\nrefs 3\n0 0 0\n1 0 0\n2 0 0\nkids 0\n";
        let doc = read(bad, &assets).unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["droppedSurfaces"], 1);
        assert_eq!(flat(&doc).triangles(), 1);
        // Deep nesting doesn't recurse.
        let deep = format!(
            "AC3Db\n{}{}",
            "OBJECT group\nkids 1\n".repeat(100_000),
            square("s", "0", "")
        );
        assert_eq!(flat(&read(&deep, &assets).unwrap()).triangles(), 2);

        let base = format!(
            "AC3Db\n{MAT}OBJECT world\nkids 2\nOBJECT group\nrot 0 0 -1 0 1 0 1 0 0\nkids 1\n{}{}",
            square("a", "0x10", "texture \"skin.png\"\ntexrep 2 2\ncrease 30\n"),
            square("b", "0x24", "loc 1 2 3\n")
        );
        let base = base.into_bytes();
        let mut rng = crate::test_rng::Pcg32::new(31);
        for n in 0..base.len() {
            if let Ok(doc) = load(Input::memory(&base[..n], "t.ac"), &assets, Keep::ALL) {
                doc.to_glb().unwrap();
            }
        }
        for _ in 0..2000 {
            let mut b = base.clone();
            for _ in 0..=rng.below(8) {
                let at = rng.below(u32::try_from(b.len()).unwrap()) as usize;
                b[at] = b" \n0123456789.-x\"SURFrefskidsOBJECT"[rng.below(34) as usize];
            }
            if let Ok(doc) = load(Input::memory(&b, "t.ac"), &assets, Keep::ALL) {
                doc.to_glb().unwrap();
            }
        }
    }

    #[test]
    fn threads_give_the_same_model() {
        // Eight shaded grids of 20,000 triangles each: enough corners to share out.
        use std::fmt::Write as _;
        let mut text = format!("AC3Db\n{MAT}OBJECT world\nkids 8\n");
        let side = 101;
        for k in 0..8 {
            write!(
                text,
                "OBJECT poly\nname \"grid{k}\"\nloc {k} 0 0\nnumvert {}\n",
                side * side
            )
            .unwrap();
            for i in 0..side {
                for j in 0..side {
                    writeln!(text, "{i}e-2 {j}e-2 {}e-2", (i * j + k) % 7).unwrap();
                }
            }
            writeln!(text, "numsurf {}", (side - 1) * (side - 1)).unwrap();
            for i in 0..side - 1 {
                for j in 0..side - 1 {
                    let a = i * side + j;
                    let (b, c, d) = (a + side, a + side + 1, a + 1);
                    write!(text, "SURF 0x10\nmat 0\nrefs 4\n{a} 0 0\n{b} 1 0\n{c} 1 1\n{d} 0 1\n").unwrap();
                }
            }
            text += "kids 0\n";
        }
        let with = |threads: usize| {
            let keep = Keep::ALL.with_threads(threads);
            load(Input::memory(text.as_bytes(), "t.ac"), &Memory::new(), keep)
                .unwrap()
                .to_glb()
                .unwrap()
        };
        let one = with(1);
        assert_eq!(with(4), one);
        crate::keep::NO_THREADS.set(true);
        let none = with(4);
        crate::keep::NO_THREADS.set(false);
        assert_eq!(none, one);
    }
}
