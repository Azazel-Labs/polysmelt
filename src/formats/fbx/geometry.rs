//! Meshes: polygons triangulated into a primitive per material slot, over the layer elements (normals, tangents,
//! UV and color sets) and normals made from smoothing.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::deform::{Channel, Target, split_weight};
use super::{Reader, child_attr, floats, ints, name_of, num, string};
use crate::document::Part;
use crate::formats::fbx_tree::{Attr as A, NodeHandle};
use crate::mesh::{FastMap, Prim, srgb_to_linear, triangulate};
use crate::num::f32_of;
use crate::vec3;
use crate::{Context as _, ErrorKind, Result};

/// A layer element's values and how they map onto polygon vertices.
struct Layer {
    values: Vec<f64>,
    index: Option<Vec<i64>>,
    mapping: Mapping,
    width: usize,
}

/// What a layer element holds a value for.
#[derive(Clone, Copy)]
enum Mapping {
    ControlPoint,
    Polygon,
    AllSame,
    PolygonVertex,
}

impl Mapping {
    fn of(el: NodeHandle<'_>, default: Self) -> Self {
        match child_attr(el, "MappingInformationType").and_then(string) {
            Some("ByVertice" | "ByVertex" | "ByControlPoint") => Self::ControlPoint,
            Some("ByPolygon") => Self::Polygon,
            Some("AllSame") => Self::AllSame,
            Some(_) => Self::PolygonVertex,
            None => default,
        }
    }
}

impl Layer {
    fn read(geom: NodeHandle<'_>, element: &str, values: &str, index: &str, width: usize) -> Option<Self> {
        let el = geom
            .children_by_name(element)
            .find(|e| e.attributes().first().and_then(num).unwrap_or(0.0) == 0.0)?;
        Self::of(el, values, index, width)
    }

    /// Every layer of `element` (UV sets, color sets) with its node, by its number, lowest first: the first is
    /// the one [`Layer::read`] gives. One that can't be read is left out.
    fn read_all<'a>(
        geom: NodeHandle<'a>,
        element: &str,
        values: &str,
        index: &str,
        width: usize,
    ) -> Vec<(NodeHandle<'a>, Self)> {
        let mut all: Vec<(f64, NodeHandle<'a>, Self)> = geom
            .children_by_name(element)
            .filter_map(|el| {
                let n = el.attributes().first().and_then(num).unwrap_or(0.0);
                Some((n, el, Self::of(el, values, index, width)?))
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        all.into_iter().map(|(_, el, l)| (el, l)).collect()
    }

    fn of(el: NodeHandle<'_>, values: &str, index: &str, width: usize) -> Option<Self> {
        Some(Self {
            values: floats(child_attr(el, values)?)?,
            index: child_attr(el, index).and_then(ints),
            mapping: Mapping::of(el, Mapping::PolygonVertex),
            width,
        })
    }

    /// The value for polygon-vertex `pv` (control point `cp`, polygon `poly`).
    fn get(&self, pv: usize, cp: usize, poly: usize) -> Option<(usize, &[f64])> {
        let i = match self.mapping {
            Mapping::ControlPoint => cp,
            Mapping::Polygon => poly,
            Mapping::AllSame => 0,
            Mapping::PolygonVertex => pv,
        };
        let i = match &self.index {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        Some((i, self.values.get(i * self.width..i * self.width + self.width)?))
    }
}

impl Reader<'_, '_, '_> {
    /// Points material `material`'s textures that read a UV set by name at that set's number in `prim`, whose
    /// texture coordinates (`uvs`) are UV layer `color_set` of the geometry's layers `names`; the first layer then
    /// takes that one's number. A set the primitive doesn't have leaves the texture on the first. A material
    /// shared by meshes is bound by the first.
    fn bind_uv_sets(&mut self, material: usize, prim: &Prim, names: &[String], color_set: usize) {
        let Some(sets) = self.material_uvs.remove(&material) else {
            return;
        };
        for (field, name) in sets {
            let Some(layer) = names.iter().position(|n| *n == name) else {
                continue;
            };
            let number = match layer {
                l if l == color_set => 0,
                0 => color_set,
                l => l,
            };
            let there = if number == 0 {
                !prim.uvs.is_empty()
            } else {
                prim.uv_sets.get(number).is_some_and(|s| !s.is_empty())
            };
            if there
                && let Some(t) = field(&mut self.materials[material])
                && let Ok(n) = u32::try_from(number)
            {
                t.uv = n;
            }
        }
    }

    /// The mesh for geometry `g` as model `m` (0 for none) uses it.
    pub(super) fn mesh_for(&mut self, g: i64, m: i64) -> Result<usize> {
        let mats = if m == 0 { Vec::new() } else { self.model_materials(m) };
        if let Some(&i) = self.mesh_of.get(&(g, mats.clone())) {
            return Ok(i);
        }
        let geom = self.objs.node(g).context(ErrorKind::Corrupt, "missing geometry")?;
        // The UV set each material slot's color texture reads, by name.
        let color_uv: HashMap<usize, &str> = mats
            .iter()
            .enumerate()
            .filter_map(|(slot, m)| Some((slot, self.color_uv.get(m)?.as_str())))
            .collect();
        let named = !color_uv.is_empty() || mats.iter().any(|m| self.material_uvs.contains_key(m));
        let (prims, uv_names, dropped) = geometry(geom, self.limit, &color_uv, named, self.b.crease)?;
        self.b.dropped += dropped;
        // Skin weights per control point: (joint, weight), heaviest first.
        let (clusters, influences) = if self.keep.skins {
            self.influences(g)
        } else {
            (0, Vec::new())
        };
        let channels = if self.keep.morph_targets {
            self.morph_channels(g)
        } else {
            Vec::new()
        };
        let channels: Vec<(i64, Channel)> = channels.iter().filter_map(|&c| Some((c, self.channel(c)?))).collect();
        let shapes: Vec<&Target> = channels.iter().flat_map(|(_, c)| &c.targets).collect();
        // Offsets to normals only for a geometry with normals of its own, and only when some aren't zero.
        let shape_normals = geom.children_by_name("LayerElementNormal").next().is_some()
            && shapes.iter().any(|t| t.normals.values().flatten().any(|c| *c != 0.0));
        let joint_sets = influences.iter().map(|v| v.len().div_ceil(4)).max().unwrap_or(0).min(4);
        let mut primitives = Vec::new();
        for (slot, prim, cps, color_set) in prims {
            let material = match mats.get(slot) {
                Some(&i) => i,
                None => self.default_material(),
            };
            self.bind_uv_sets(material, &prim, &uv_names, color_set);
            if !prim.uvs.is_empty() && !self.textured.contains(&material) {
                self.textured.push(material);
            }
            let normals = shape_normals && !prim.normals.is_empty();
            let Some(mut json) = self.b.primitive(prim) else {
                continue;
            };
            json["material"] = json!(material);
            if joint_sets > 0 {
                let unweighted = clusters; // stands for the model itself, after the clusters; see `skin`
                for set in 0..joint_sets {
                    let mut joints = Vec::with_capacity(cps.len() * 4);
                    let mut weights = Vec::with_capacity(cps.len() * 4);
                    for cp in &cps {
                        let inf = influences.get(*cp).map_or(&[][..], Vec::as_slice);
                        for k in set * 4..set * 4 + 4 {
                            match inf.get(k) {
                                Some(&(j, w)) => {
                                    joints.push(u16::try_from(j).unwrap_or(0));
                                    weights.push(w);
                                }
                                None if k == 0 => {
                                    joints.push(u16::try_from(unweighted).unwrap_or(0));
                                    weights.push(1.0);
                                }
                                None => {
                                    joints.push(0);
                                    weights.push(0.0);
                                }
                            }
                        }
                    }
                    let n = cps.len();
                    let j = self.b.accessor(Part::u16(joints), 5123, n, "VEC4", Some(34962));
                    let w = self.b.accessor(Part::f32(weights), 5126, n, "VEC4", Some(34962));
                    json["attributes"][format!("JOINTS_{set}")] = json!(j);
                    json["attributes"][format!("WEIGHTS_{set}")] = json!(w);
                }
            }
            if !shapes.is_empty() {
                let targets: Vec<Value> = shapes
                    .iter()
                    .map(|t| {
                        let d: Vec<[f32; 3]> = cps
                            .iter()
                            .map(|cp| t.positions.get(cp).copied().unwrap_or([0.0; 3]))
                            .collect();
                        let mut target = json!({ "POSITION": self.b.positions(d) });
                        if normals {
                            let n: Vec<f32> = cps
                                .iter()
                                .flat_map(|cp| t.normals.get(cp).copied().unwrap_or([0.0; 3]))
                                .collect();
                            target["NORMAL"] = json!(self.b.floats(n, "VEC3", false));
                        }
                        target
                    })
                    .collect();
                json["targets"] = json!(targets);
            }
            primitives.push(json);
        }
        let mut mesh = json!({ "primitives": primitives });
        if let Some(name) = name_of(geom) {
            mesh["name"] = json!(name);
        }
        if !shapes.is_empty() {
            let mut weights = vec![0.0; shapes.len()];
            let mut first = 0;
            let mut morph_ids = Vec::with_capacity(channels.len());
            for (id, c) in &channels {
                split_weight(c.weight, &c.full, &mut weights[first..first + c.targets.len()]);
                morph_ids.push((*id, first));
                self.channel_full.insert(*id, c.full.clone());
                first += c.targets.len();
            }
            mesh["weights"] = json!(weights);
            mesh["extras"] = json!({ "targetNames": shapes.iter().map(|t| t.name.as_str()).collect::<Vec<_>>() });
            self.morphs.insert(self.meshes.len(), morph_ids);
        }
        self.meshes.push(mesh);
        let i = self.meshes.len() - 1;
        if joint_sets > 0 {
            self.skinned_meshes.insert(i);
        }
        self.mesh_of.insert((g, mats), i);
        Ok(i)
    }
}

/// Vertices already emitted at a crowded control point, by primitive, control point and what else they're matched on.
type Seen = FastMap<(usize, usize, [usize; 5 + EXTRA_SETS]), u32>;

/// The most vertices chained at one control point before the rest are looked up by hash instead.
const MAX_CHAIN: usize = 32;

/// A material slot's primitive, each vertex's control point, and how many corners and triangles its polygons have.
struct SlotPrim {
    slot: usize,
    prim: Prim,
    cps: Vec<usize>,
    corners: usize,
    triangles: usize,
}

/// How many UV and color sets past the first of each a geometry's vertices are matched on; one with more gives
/// each polygon corner its own vertex instead.
const EXTRA_SETS: usize = 3;

/// The most UV or color sets read from one geometry.
const MAX_SETS: usize = 64;

/// A primitive per material slot, each vertex's control point, and the UV layer its `uvs` hold.
type Prims = Vec<(usize, Prim, Vec<usize>, usize)>;

/// A geometry's triangles in its own space, a primitive per material slot (sorted), each vertex's control point
/// alongside so skins and blend shapes can find it; more than `limit` triangles is refused. A slot whose color
/// texture reads a UV layer by name (`color_uv`) takes that layer as its texture coordinates, and the first layer
/// in its place among the other sets. With `named`, the UV layers' names come back too, by number. Last, how
/// many polygons were left out for naming a control point that isn't there or isn't a finite point.
fn geometry(
    geom: NodeHandle<'_>,
    limit: usize,
    color_uv: &HashMap<usize, &str>,
    named: bool,
    crease: f32,
) -> Result<(Prims, Vec<String>, usize)> {
    let verts = floats(child_attr(geom, "Vertices").context(ErrorKind::Corrupt, "mesh has no vertices")?)
        .context(ErrorKind::Corrupt, "bad vertices")?;
    let poly_index = ints(child_attr(geom, "PolygonVertexIndex").context(ErrorKind::Corrupt, "mesh has no polygons")?)
        .context(ErrorKind::Corrupt, "bad polygons")?;
    let normals = Layer::read(geom, "LayerElementNormal", "Normals", "NormalsIndex", 3);
    // Tangents need the normals they're perpendicular to; the binormals only give their handedness.
    let tangents = normals
        .as_ref()
        .and_then(|_| Layer::read(geom, "LayerElementTangent", "Tangents", "TangentsIndex", 3));
    let binormals = tangents
        .as_ref()
        .and_then(|_| Layer::read(geom, "LayerElementBinormal", "Binormals", "BinormalsIndex", 3));
    let uv_all = Layer::read_all(geom, "LayerElementUV", "UV", "UVIndex", 2);
    // The layers' names, in the same order.
    let uv_names: Vec<String> = if named {
        uv_all
            .iter()
            .take(MAX_SETS)
            .map(|(el, _)| {
                let name = child_attr(*el, "Name").and_then(string).unwrap_or("");
                name.split('\0').next().unwrap_or("").to_owned()
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut uv_layers: Vec<Layer> = uv_all.into_iter().map(|(_, l)| l).collect();
    let mut color_layers: Vec<Layer> = Layer::read_all(geom, "LayerElementColor", "Colors", "ColorIndex", 4)
        .into_iter()
        .map(|(_, l)| l)
        .collect();
    uv_layers.truncate(MAX_SETS);
    color_layers.truncate(MAX_SETS);
    let mut uv_layers = uv_layers.into_iter();
    let mut color_layers = color_layers.into_iter();
    let (uvs, colors) = (uv_layers.next(), color_layers.next());
    let (uv_more, color_more): (Vec<Layer>, Vec<Layer>) = (uv_layers.collect(), color_layers.collect());
    // A polygon naming a control point that isn't there or whose position isn't a number, too few to make a
    // triangle, or a hole, is left out.
    let holes = holes(geom);
    let whole = |poly: usize, range: &std::ops::Range<usize>| {
        range.len() >= 3
            && holes
                .as_ref()
                .is_none_or(|h| h.get(if h.len() == 1 { 0 } else { poly }) != Some(&true))
    };
    let sound = |range: &std::ops::Range<usize>| {
        range.clone().all(|pv| {
            let cp = corner_cp(poly_index[pv]);
            cp < verts.len() / 3 && verts[cp * 3..cp * 3 + 3].iter().all(|c| c.is_finite())
        })
    };
    let usable = |poly: usize, range: &std::ops::Range<usize>| whole(poly, range) && sound(range);
    let mut dropped = 0;
    // Without normals of its own, a geometry with smoothing has them made from it.
    let smooth = if normals.is_none() {
        smoothing(geom, &verts, &poly_index, &usable)
    } else {
        None
    };
    let mut tris = Vec::new();
    let mut corner_pos = Vec::new();
    let mat_layer = geom.children_by_name("LayerElementMaterial").next().and_then(|el| {
        Some(Layer {
            values: ints(child_attr(el, "Materials")?)?
                .into_iter()
                .map(|i| {
                    #[allow(clippy::cast_precision_loss, reason = "a material slot, far below 2^53")]
                    let i = i as f64;
                    i
                })
                .collect(),
            index: None,
            mapping: Mapping::of(el, Mapping::AllSame),
            width: 1,
        })
    });
    // A layer holding a value per polygon vertex, unindexed, gives every corner its own vertex: no need to look for
    // one already made.
    let unique = [&normals, &tangents, &binormals, &uvs, &colors]
        .into_iter()
        .flatten()
        .chain(uv_more.iter().chain(&color_more))
        .any(|l| matches!(l.mapping, Mapping::PolygonVertex) && l.index.is_none())
        || uv_more.len() + color_more.len() > EXTRA_SETS;
    let slot_of = |pv: usize, poly: usize| {
        mat_layer.as_ref().and_then(|l| l.get(pv, 0, poly)).map_or(0, |(_, v)| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a material slot; `as` saturates bad ones and `get` refuses them"
            )]
            let slot = v[0] as usize;
            slot
        })
    };
    // Each polygon's primitive (one per material slot), or none when it's left out, found first so each
    // primitive's buffers are sized once.
    let mut slots: Vec<SlotPrim> = Vec::new();
    let mut slot_index: FastMap<usize, usize> = FastMap::default();
    let mut last: Option<(usize, usize)> = None;
    let mut poly_slot = Vec::new();
    let mut total = 0;
    for (poly, range) in polygons(&poly_index) {
        if !usable(poly, &range) {
            dropped += usize::from(whole(poly, &range));
            poly_slot.push(usize::MAX);
            continue;
        }
        let slot = slot_of(range.end - 1, poly);
        let s = match last {
            Some((n, s)) if n == slot => s,
            _ => {
                let s = *slot_index.entry(slot).or_insert_with(|| {
                    slots.push(SlotPrim {
                        slot,
                        prim: Prim {
                            material: slot,
                            ..Prim::default()
                        },
                        cps: Vec::new(),
                        corners: 0,
                        triangles: 0,
                    });
                    slots.len() - 1
                });
                last = Some((slot, s));
                s
            }
        };
        slots[s].corners += range.len();
        slots[s].triangles += range.len() - 2;
        total += range.len();
        poly_slot.push(s);
    }
    let cp_count = verts.len() / 3;
    for s in &mut slots {
        // Without de-duplication every corner is a vertex; with it, about its share of the control points.
        let n = if unique {
            s.corners
        } else {
            (s.corners / (total / cp_count.max(1)).max(1)).min(s.corners)
        };
        let p = &mut s.prim;
        p.positions.reserve_exact(n);
        s.cps.reserve_exact(n);
        if normals.is_some() || smooth.is_some() {
            p.normals.reserve_exact(n);
        }
        if tangents.is_some() {
            p.tangents.reserve_exact(n);
        }
        if uvs.is_some() {
            p.uvs.reserve_exact(n);
        }
        if colors.is_some() {
            p.colors.reserve_exact(n);
        }
        p.indices.reserve_exact(s.triangles * 3);
    }
    // Besides its primitive and control point, a vertex is matched on the index of each layer whose value can
    // differ between the corners at one control point: not one by control point, or the same for all.
    let varies = |l: &Layer| matches!(l.mapping, Mapping::Polygon | Mapping::PolygonVertex);
    let main_varies = [
        normals.as_ref().map_or(smooth.is_some(), varies),
        tangents.as_ref().is_some_and(varies),
        binormals.as_ref().is_some_and(varies),
        uvs.as_ref().is_some_and(varies),
        colors.as_ref().is_some_and(varies),
    ];
    let extra_varies: Vec<&Layer> = uv_more.iter().chain(&color_more).filter(|l| varies(l)).collect();
    let stride = main_varies.iter().filter(|v| **v).count() + extra_varies.len();
    // Vertices already made, chained by control point (the next at the same one, its primitive and its number),
    // with what each was matched on in `keys`; past `MAX_CHAIN` at one control point, in `crowded`.
    let mut head = if unique { Vec::new() } else { vec![usize::MAX; cp_count] };
    let mut chain: Vec<(usize, usize, u32)> = Vec::new();
    let mut keys: Vec<usize> = Vec::new();
    let mut crowded = Seen::default();
    if !unique {
        chain.reserve(cp_count.min(total));
        keys.reserve(cp_count.min(total) * stride);
    }
    let mut corners: Vec<(usize, usize)> = Vec::new();
    let mut ids = Vec::new();
    for (poly, range) in polygons(&poly_index) {
        let s = poly_slot[poly];
        if s == usize::MAX {
            continue;
        }
        let SlotPrim { prim, cps, .. } = &mut slots[s];
        corners.clear();
        corners.extend(range.map(|pv| (pv, corner_cp(poly_index[pv]))));
        ids.clear();
        corner_pos.clear();
        for &(pv, cp) in &corners {
            let pos = &verts[cp * 3..cp * 3 + 3];
            let p = [pos[0], pos[1], pos[2]].map(f32_of);
            corner_pos.push(p);
            let (n, tg, bn, t, c) = [&normals, &tangents, &binormals, &uvs, &colors]
                .map(|l| l.as_ref().and_then(|l| l.get(pv, cp, poly)))
                .into();
            let n = n.or_else(|| {
                smooth
                    .as_ref()
                    .map(|(group, normal)| (group[pv], normal[group[pv]].as_slice()))
            });
            let i = |x: Option<(usize, &[f64])>| x.map_or(usize::MAX, |x| x.0);
            let mut new = || {
                prim.positions.push(p);
                cps.push(cp);
                if let Some((_, nv)) = n {
                    // One with no length stays zero, and takes its faces' as the primitive is written.
                    let nv = vec3::unit_or([nv[0], nv[1], nv[2]], [0.0; 3]);
                    prim.normals.push(nv.map(f32_of));
                    if let Some((_, tv)) = tg {
                        prim.tangents.push(tangent(nv, [tv[0], tv[1], tv[2]], bn.map(|b| b.1)));
                    }
                }
                if let Some((_, tv)) = t {
                    prim.uvs.push(uv(tv));
                }
                if let Some((_, cv)) = c {
                    prim.colors.push(color(cv));
                }
                let k = prim.positions.len() - 1;
                for (s, l) in uv_more.iter().enumerate() {
                    if let Some((_, tv)) = l.get(pv, cp, poly) {
                        set_at(&mut prim.uv_sets, s + 1, k, uv(tv));
                    }
                }
                for (s, l) in color_more.iter().enumerate() {
                    if let Some((_, cv)) = l.get(pv, cp, poly) {
                        set_at(&mut prim.color_sets, s + 1, k, color(cv));
                    }
                }
                u32::try_from(k).unwrap_or(u32::MAX)
            };
            let id = if unique {
                new()
            } else {
                let mut key = [usize::MAX; 5 + EXTRA_SETS];
                let mut kn = 0;
                for (x, v) in [n, tg, bn, t, c].into_iter().zip(main_varies) {
                    if v {
                        key[kn] = i(x);
                        kn += 1;
                    }
                }
                for l in &extra_varies {
                    key[kn] = i(l.get(pv, cp, poly));
                    kn += 1;
                }
                let mut e = head[cp];
                let mut steps = 0;
                loop {
                    if e == usize::MAX {
                        if steps < MAX_CHAIN {
                            let id = new();
                            chain.push((head[cp], s, id));
                            keys.extend_from_slice(&key[..stride]);
                            head[cp] = chain.len() - 1;
                            break id;
                        }
                        break *crowded.entry((s, cp, key)).or_insert_with(new);
                    }
                    let (next, es, id) = chain[e];
                    if es == s && keys[e * stride..e * stride + stride] == key[..stride] {
                        break id;
                    }
                    e = next;
                    steps += 1;
                }
            };
            ids.push(id);
        }
        tris.clear();
        triangulate(&corner_pos, &mut tris);
        for &[a, b, c] in &tris {
            prim.indices.extend([a, b, c].map(|i| ids[i as usize]));
        }
    }
    let mut out: Prims = slots
        .into_iter()
        .filter(|s| !s.prim.indices.is_empty())
        .map(|s| {
            let (slot, mut prim, mut cps) = (s.slot, s.prim, s.cps);
            let mut n = prim.positions.len();
            if prim.normals.len() != n {
                // Neither normals nor smoothing: the default ones, whose added vertices keep their control
                // points. Made up here, they wouldn't match the file's tangents, which go.
                let added = crate::geometry::default_normals(&mut prim, crease);
                crate::geometry::follow(&mut cps, n, &added);
                n = prim.positions.len();
            }
            if prim.tangents.len() != n {
                prim.tangents.clear();
            }
            if prim.uvs.len() != n {
                prim.uvs.clear();
            }
            if prim.colors.len() != n {
                prim.colors.clear();
            }
            // A set some corner had no value for is dropped, and so are trailing empty ones.
            for set in &mut prim.uv_sets {
                if set.len() != n {
                    set.clear();
                }
            }
            for set in &mut prim.color_sets {
                if set.len() != n {
                    set.clear();
                }
            }
            // The layer the color texture reads becomes the texture coordinates; the first takes its number.
            let mut color_set = 0;
            if let Some(c) = color_uv
                .get(&slot)
                .and_then(|name| uv_names.iter().position(|n| n == name))
                .filter(|&c| c > 0 && prim.uv_sets.get(c).is_some_and(|s| s.len() == n))
            {
                let first = std::mem::take(&mut prim.uvs);
                prim.uvs = std::mem::replace(&mut prim.uv_sets[c], first);
                color_set = c;
            }
            while prim.uv_sets.last().is_some_and(Vec::is_empty) {
                prim.uv_sets.pop();
            }
            while prim.color_sets.last().is_some_and(Vec::is_empty) {
                prim.color_sets.pop();
            }
            (slot, prim, cps, color_set)
        })
        .collect();
    out.sort_unstable_by_key(|x| x.0);
    let triangles: usize = out.iter().map(|x| x.1.indices.len() / 3).sum();
    ensure!(
        triangles <= limit,
        LimitExceeded,
        "too many triangles to convert (more than {limit})"
    );
    Ok((out, uv_names, dropped))
}

/// The control point a `PolygonVertexIndex` entry names (a polygon's last is stored as `!index`).
fn corner_cp(v: i64) -> usize {
    usize::try_from(if v < 0 { !v } else { v }).unwrap_or(usize::MAX)
}

/// Each polygon of a `PolygonVertexIndex`: its number, and its polygon vertices.
fn polygons(poly_index: &[i64]) -> impl Iterator<Item = (usize, std::ops::Range<usize>)> + '_ {
    let mut start = 0;
    poly_index
        .iter()
        .enumerate()
        .filter(|(_, v)| **v < 0)
        .enumerate()
        .map(move |(poly, (k, _))| {
            let range = start..k + 1;
            start = k + 1;
            (poly, range)
        })
}

/// Which polygons are holes, by polygon (or one for all).
fn holes(geom: NodeHandle<'_>) -> Option<Vec<bool>> {
    let el = geom.children_by_name("LayerElementHole").next()?;
    match child_attr(el, "Hole")? {
        A::ArrBool(b) => b.map(|x| x & 1 != 0),
        v => ints(v)
            .map(|h| h.iter().map(|&x| x != 0).collect())
            .or_else(|| floats(v).map(|h| h.iter().map(|&x| x != 0.0).collect())),
    }
}

/// Normals for a geometry without its own, from its smoothing: per polygon vertex, a group, and per group, the
/// sum of its polygons' normals, each as long as its polygon is large. By polygon, the corners at a control point
/// whose polygons are in the same smoothing group share a normal (group 0 is flat; groups are matched whole, not
/// bit by bit); by edge, the corners of polygons meeting at an edge that isn't hard do. `None` without smoothing.
fn smoothing(
    geom: NodeHandle<'_>,
    verts: &[f64],
    poly_index: &[i64],
    usable: &dyn Fn(usize, &std::ops::Range<usize>) -> bool,
) -> Option<(Vec<usize>, Vec<[f64; 3]>)> {
    let el = geom.children_by_name("LayerElementSmoothing").next()?;
    let values = child_attr(el, "Smoothing").and_then(ints)?;
    let mapping = child_attr(el, "MappingInformationType").and_then(string);
    let n = poly_index.len();
    let cp = |pv: usize| corner_cp(poly_index[pv]);
    // The polygon vertex after each, round its polygon.
    let mut next = vec![usize::MAX; n];
    for (_, range) in polygons(poly_index) {
        for pv in range.clone() {
            next[pv] = if pv + 1 == range.end { range.start } else { pv + 1 };
        }
    }
    let mut group = vec![usize::MAX; n];
    if mapping == Some("ByEdge") {
        // An edge is its two control points, lowest first; `Edges` names each by the polygon vertex it starts at.
        let edge = |pv: usize| {
            let (a, b) = (cp(pv), cp(next[pv]));
            (a.min(b), a.max(b))
        };
        let edges = child_attr(geom, "Edges").and_then(ints).unwrap_or_default();
        let mut hard: Vec<(usize, usize)> = edges
            .iter()
            .zip(&values)
            .filter(|(_, smooth)| **smooth == 0)
            .filter_map(|(&e, _)| usize::try_from(e).ok().filter(|&pv| pv < n && next[pv] != usize::MAX))
            .map(edge)
            .collect();
        hard.sort_unstable();
        // Every polygon's side: its edge, and its corners at the edge's first and second control point.
        let mut sides: Vec<((usize, usize), usize, usize)> = Vec::new();
        for (poly, range) in polygons(poly_index) {
            if !usable(poly, &range) {
                continue;
            }
            for pv in range {
                let (a, b) = (cp(pv), cp(next[pv]));
                if a != b {
                    let (lo, hi) = if a < b { (pv, next[pv]) } else { (next[pv], pv) };
                    sides.push((edge(pv), lo, hi));
                }
            }
        }
        sides.sort_unstable();
        let mut parent: Vec<usize> = (0..n).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        };
        for run in sides.chunk_by(|a, b| a.0 == b.0) {
            if hard.binary_search(&run[0].0).is_ok() {
                continue;
            }
            for w in run.windows(2) {
                for (x, y) in [(w[0].1, w[1].1), (w[0].2, w[1].2)] {
                    let (x, y) = (find(&mut parent, x), find(&mut parent, y));
                    parent[x] = y;
                }
            }
        }
        for (poly, range) in polygons(poly_index) {
            if usable(poly, &range) {
                for pv in range {
                    group[pv] = find(&mut parent, pv);
                }
            }
        }
    } else {
        let mut ids: FastMap<(usize, i64), usize> = FastMap::default();
        for (poly, range) in polygons(poly_index) {
            if !usable(poly, &range) {
                continue;
            }
            let at = if mapping == Some("AllSame") { 0 } else { poly };
            let mask = values.get(at).copied().unwrap_or(0);
            // Group 0 is flat: each polygon its own.
            let key = if mask == 0 {
                -1 - i64::try_from(poly).unwrap_or(i64::MAX - 1)
            } else {
                mask
            };
            for pv in range {
                let len = ids.len();
                group[pv] = *ids.entry((cp(pv), key)).or_insert(len);
            }
        }
    }
    let mut normal = vec![[0.0; 3]; n];
    let pos = |pv: usize| {
        let c = cp(pv);
        [verts[c * 3], verts[c * 3 + 1], verts[c * 3 + 2]]
    };
    for (poly, range) in polygons(poly_index) {
        if !usable(poly, &range) {
            continue;
        }
        // Newell's method: twice the polygon's area, along its normal.
        let mut sum = [0.0; 3];
        for pv in range.clone() {
            vec3::newell_edge(&mut sum, pos(pv), pos(next[pv]));
        }
        for pv in range {
            let g = &mut normal[group[pv]];
            for k in 0..3 {
                g[k] += sum[k];
            }
        }
    }
    Some((group, normal))
}

/// An FBX texture coordinate the glTF way: FBX's texture origin is bottom-left, glTF's top-left.
fn uv(v: &[f64]) -> [f32; 2] {
    [f32_of(v[0]), 1.0 - f32_of(v[1])]
}

/// An FBX vertex color, sRGB, in linear light.
fn color(v: &[f64]) -> [f32; 4] {
    let [r, g, b, a] = [v[0], v[1], v[2], v[3]].map(|v| f32_of(v.clamp(0.0, 1.0)));
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
}

/// Sets vertex `k` of set `s` of `sets` to `v`, when every vertex before it has a value in that set: a set with a
/// gap can't line up with the vertices, and is dropped at the end.
fn set_at<T: Copy>(sets: &mut Vec<Vec<T>>, s: usize, k: usize, v: T) {
    if sets.len() <= s {
        sets.resize_with(s + 1, Vec::new);
    }
    if sets[s].len() == k {
        sets[s].push(v);
    }
}

/// A glTF tangent from FBX's unit normal `n`, tangent `t` and, if there is one, binormal `b`: `t` made
/// perpendicular to `n` and unit length, with `w` the side `b` is on (glTF's bitangent is `cross(n, t) * w`).
/// glTF's V runs down where FBX's runs up, but both keep the bitangent pointing up the image, so it doesn't flip.
fn tangent(n: [f64; 3], t: [f64; 3], b: Option<&[f64]>) -> [f32; 4] {
    let d = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
    let t = [t[0] - n[0] * d, t[1] - n[1] * d, t[2] - n[2] * d];
    let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let t = if len > 0.0 && len.is_finite() {
        t.map(|c| c / len)
    } else {
        [1.0, 0.0, 0.0]
    };
    let w = b.map_or(1.0, |b| {
        let c = [
            n[1] * t[2] - n[2] * t[1],
            n[2] * t[0] - n[0] * t[2],
            n[0] * t[1] - n[1] * t[0],
        ];
        if c[0] * b[0] + c[1] * b[1] + c[2] * b[2] < 0.0 {
            -1.0
        } else {
            1.0
        }
    });
    [f32_of(t[0]), f32_of(t[1]), f32_of(t[2]), w]
}
