//! Skinning as every reader writes it, so a skinned mesh is the same kind of thing whatever format held it.
//!
//! - **Influences.** [`sets`] takes each vertex's joints and weights as the file lists them and gives glTF's
//!   `JOINTS_n` and `WEIGHTS_n`: joints that don't exist and weights that aren't positive numbers are dropped, a
//!   joint listed twice is one with the weights added, the [`MAX_INFLUENCES`] strongest are kept, strongest first
//!   (the lower joint first between equals), and the weights are scaled to sum to 1. A slot with no weight names
//!   joint 0, as glTF's validator wants. No reader cuts a vertex's list shorter first.
//! - **A vertex no joint moves** is bound wholly to a *rest joint*: a joint that stands where the node drawing the
//!   mesh stands, with the inverse bind matrix that keeps the vertex where it is. glTF ignores a skinned node's own
//!   transform, so binding such a vertex to the skin's first joint instead (as some readers did) made it follow
//!   that joint's animation, where in the file it stays with its mesh. The rest joint comes after the file's
//!   joints, and only in a skin that has such a vertex: [`Sets::unbound`] says whether one is needed.
//! - **`skin.skeleton`** is always written, as the joints' common root ([`common_root`]): the nearest node that
//!   every joint is, or is under.
//! - **Inverse bind matrices** are always written and finite ([`inverse_bind`]): one that isn't finite is the
//!   identity, and the last row is glTF's `0 0 0 1`.

use serde_json::{Value, json};

use crate::document::{Builder, Part};

pub(crate) use crate::mesh::MAX_INFLUENCES;

/// Merges a vertex's repeated joints (adding their weights in the order they come) and keeps the
/// [`MAX_INFLUENCES`] strongest, strongest first and the lower joint first between equals.
pub(crate) fn strongest(list: &mut Vec<(u32, f32)>) {
    list.sort_by_key(|e| e.0);
    list.dedup_by(|later, kept| {
        let same = later.0 == kept.0;
        if same {
            kept.1 += later.1;
        }
        same
    });
    list.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    list.truncate(MAX_INFLUENCES);
}

/// A primitive's joints and weights in sets of four, as glTF wants them.
pub(crate) struct Sets {
    /// Set after set, four joints to a vertex.
    pub joints: Vec<u32>,
    /// The weights of `joints`.
    pub weights: Vec<f32>,
    /// How many sets: `JOINTS_0` to `JOINTS_{sets - 1}`.
    pub groups: usize,
    pub vertices: usize,
    /// Whether some vertex had no joint and was bound to the rest joint: the skin needs that joint.
    pub unbound: bool,
}

/// Each of `vertices` vertices' joints and weights in sets of four: see the module's notes. `fill` adds vertex
/// `v`'s joints and weights to an empty list, as the file gives them; `joints` is how many joints the file's skin
/// has, and `rest` the joint a vertex with none is bound to (which may come after them).
pub(crate) fn sets(
    vertices: usize,
    joints: usize,
    rest: u32,
    mut fill: impl FnMut(usize, &mut Vec<(u32, f32)>),
) -> Sets {
    let n = vertices;
    // Every vertex's list, end to end, and where each ends: no list of its own for each vertex.
    let mut flat: Vec<(u32, f32)> = Vec::with_capacity(n * 4);
    let mut ends: Vec<usize> = Vec::with_capacity(n);
    let mut list: Vec<(u32, f32)> = Vec::new();
    let mut most = 1;
    let mut unbound = false;
    for v in 0..n {
        list.clear();
        fill(v, &mut list);
        list.retain(|&(j, wt)| wt > 0.0 && wt.is_finite() && (j as usize) < joints);
        strongest(&mut list);
        let sum: f32 = list.iter().map(|e| e.1).sum();
        if list.is_empty() || sum <= 0.0 {
            unbound = true;
            flat.push((rest, 1.0));
        } else if sum.is_finite() {
            flat.extend(list.iter().map(|&(j, wt)| (j, wt / sum)));
        } else {
            // Weights too large to add up: kept in proportion to the largest.
            let top = list[0].1;
            let sum: f32 = list.iter().map(|e| e.1 / top).sum();
            flat.extend(list.iter().map(|&(j, wt)| (j, wt / top / sum)));
        }
        most = most.max(flat.len() - ends.last().copied().unwrap_or(0));
        ends.push(flat.len());
    }
    let groups = most.div_ceil(4);
    let mut js = vec![0u32; groups * n * 4];
    let mut ws = vec![0f32; groups * n * 4];
    let starts = std::iter::once(0).chain(ends.iter().copied());
    for (v, (start, end)) in starts.zip(ends.iter().copied()).enumerate() {
        for (k, &(j, wt)) in flat[start..end].iter().enumerate() {
            // A weight that rounds to nothing names joint 0, as a slot with no weight does.
            if wt > 0.0 {
                let at = (k / 4) * n * 4 + v * 4 + k % 4;
                js[at] = j;
                ws[at] = wt;
            }
        }
    }
    Sets {
        joints: js,
        weights: ws,
        groups,
        vertices: n,
        unbound,
    }
}

impl Sets {
    /// Gives each vertex a reader added after these were made (split from vertex `added[k]` for its own normal)
    /// the joints and weights of the vertex it came from.
    pub fn follow(&mut self, added: &[u32]) {
        if added.is_empty() {
            return;
        }
        let n = self.vertices;
        let m = n + added.len();
        let mut joints = Vec::with_capacity(self.groups * m * 4);
        let mut weights = Vec::with_capacity(self.groups * m * 4);
        for s in 0..self.groups {
            let (at, start) = (s * n * 4, joints.len());
            joints.extend_from_slice(&self.joints[at..at + n * 4]);
            weights.extend_from_slice(&self.weights[at..at + n * 4]);
            for &a in added {
                // A source may itself be an added vertex: it is in the new list already.
                let from = start + (a as usize).min(m - 1) * 4;
                joints.extend_from_within(from..from + 4);
                weights.extend_from_within(from..from + 4);
            }
        }
        self.joints = joints;
        self.weights = weights;
        self.vertices = m;
    }

    /// Writes the accessors for a skin of `count` joints (bytes when they fit, else shorts), returning the
    /// attributes they make.
    pub fn write(self, b: &mut Builder, count: usize) -> Vec<(String, usize)> {
        let n = self.vertices;
        let per = n * 4;
        let mut out = Vec::new();
        for s in 0..self.groups {
            let jpart = &self.joints[s * per..(s + 1) * per];
            let j = if count <= 256 {
                let v: Vec<u8> = jpart.iter().map(|&j| u8::try_from(j).unwrap_or(0)).collect();
                b.accessor(Part::bytes(v), 5121, n, "VEC4", Some(34962))
            } else {
                let v = jpart.iter().map(|&j| u16::try_from(j).unwrap_or(0)).collect();
                b.accessor(Part::u16(v), 5123, n, "VEC4", Some(34962))
            };
            let w = b.accessor(
                Part::f32(self.weights[s * per..(s + 1) * per].to_vec()),
                5126,
                n,
                "VEC4",
                Some(34962),
            );
            out.push((format!("JOINTS_{s}"), j));
            out.push((format!("WEIGHTS_{s}"), w));
        }
        out
    }
}

/// The joints' common root, for `skin.skeleton`: the nearest node that every joint is or is under, given each
/// node's parent. `None` when there are no joints, or they share no root (which glTF doesn't allow of a skin).
pub(crate) fn common_root(parents: &[Option<usize>], joints: &[usize]) -> Option<usize> {
    // A node's ancestors, itself first; a loop in the parents ends the walk.
    let chain = |j: usize| {
        let mut out = vec![j];
        let mut at = j;
        while let Some(p) = parents.get(at).copied().flatten() {
            if out.len() > parents.len() {
                break;
            }
            out.push(p);
            at = p;
        }
        out
    };
    let (&first, rest) = joints.split_first()?;
    // The first joint's chain from the root down, cut to what each other joint's shares.
    let mut common = chain(first);
    common.reverse();
    for &j in rest {
        if common.contains(&j) {
            // An ancestor of the first joint, or itself: everything above it is shared already.
            let at = common.iter().position(|&c| c == j)?;
            common.truncate(at + 1);
            continue;
        }
        let mut other = chain(j);
        other.reverse();
        let shared = common.iter().zip(&other).take_while(|(a, b)| a == b).count();
        common.truncate(shared);
        if common.is_empty() {
            return None;
        }
    }
    common.last().copied()
}

/// Sets every skin's `skeleton` to its joints' common root, in a document's `nodes` and `skins` as JSON. A skin
/// whose joints share no root is left without one.
pub(crate) fn set_skeletons(nodes: &[Value], skins: &mut [Value]) {
    if skins.is_empty() {
        return;
    }
    let mut parents: Vec<Option<usize>> = vec![None; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        for c in n.get("children").and_then(Value::as_array).into_iter().flatten() {
            if let Some(c) = c.as_u64().and_then(|c| usize::try_from(c).ok())
                && c < parents.len()
            {
                parents[c] = Some(i);
            }
        }
    }
    for s in skins {
        let joints: Vec<usize> = s
            .get("joints")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|j| j.as_u64().and_then(|j| usize::try_from(j).ok()))
            .collect();
        match (common_root(&parents, &joints), s.as_object_mut()) {
            (Some(root), Some(o)) => {
                o.insert("skeleton".into(), json!(root));
            }
            (None, Some(o)) => {
                o.remove("skeleton");
            }
            _ => {}
        }
    }
}

/// A primitive's attributes as [`conform`] rewrites them: each name and accessor.
#[cfg(feature = "gltf")]
type Attributes = Vec<(String, usize)>;

/// Holds the skins of a glTF document that a reader converts (glTF 1.0, VRM, 3D Tiles) to the rule every reader's
/// follow (see the module's notes): each skinned primitive's joints and weights go through [`sets`], a vertex no
/// joint moves is bound to a rest joint (the node that draws the mesh, or a node of its own under it when that
/// node is one of the skin's joints already, with the identity for its inverse bind matrix), and every skin's
/// `skeleton` is its joints' common root ([`set_skeletons`]).
///
/// A primitive whose every vertex already has at most [`MAX_INFLUENCES`] joints, none of them twice, all of them
/// the skin's, and weights that are positive and sum to 1, is left as the file has it, whatever order its joints
/// are in. What's rewritten goes into a new buffer. Left as they are: a primitive compressed with Draco (its
/// joints and weights are in the compressed stream), one whose joints or weights can't be read, and a mesh drawn
/// with skins of different numbers of joints (its joint indices can't mean the same in each).
#[cfg(feature = "gltf")]
pub(crate) fn conform(doc: &mut crate::Document) {
    let plan = Plan::of(doc);
    let Plan { rewrites, rests, skins } = plan;
    let mut added = crate::document::Added::new(doc);
    // Each rewritten primitive: its mesh and place, how many sets it had, and its new attributes.
    let mut attributes: Vec<(usize, usize, usize, Attributes)> = Vec::new();
    for r in rewrites {
        let wide = r.count + usize::from(r.sets.unbound) > 256;
        let s = &r.sets;
        let per = s.vertices * 4;
        let mut out = Vec::new();
        for k in 0..s.groups {
            let joints = &s.joints[k * per..(k + 1) * per];
            let j = if wide {
                let b: Vec<u8> = joints
                    .iter()
                    .flat_map(|&j| u16::try_from(j).unwrap_or(0).to_le_bytes())
                    .collect();
                added.accessor(&b, 5123, s.vertices, "VEC4", Some(34962))
            } else {
                let b: Vec<u8> = joints.iter().map(|&j| u8::try_from(j).unwrap_or(0)).collect();
                added.accessor(&b, 5121, s.vertices, "VEC4", Some(34962))
            };
            let w: Vec<u8> = s.weights[k * per..(k + 1) * per]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            let w = added.accessor(&w, 5126, s.vertices, "VEC4", Some(34962));
            out.push((format!("JOINTS_{k}"), j));
            out.push((format!("WEIGHTS_{k}"), w));
        }
        attributes.push((r.mesh, r.prim, r.had, out));
    }
    // A rest joint's inverse bind matrix is the identity: the vertex stays where it is in the node's space.
    let rest_ibms: Vec<Option<usize>> = rests
        .iter()
        .map(|r| {
            r.ibms.as_ref().map(|ibms| {
                let mut all = ibms.clone();
                all.extend(IDENTITY);
                added.floats(&all, "MAT4", None)
            })
        })
        .collect();
    added.finish(doc);

    let json = doc.json_mut();
    for (m, p, had, out) in attributes {
        let Some(Value::Object(attrs)) = json
            .get_mut("meshes")
            .and_then(|l| l.get_mut(m))
            .and_then(|mesh| mesh.get_mut("primitives")?.get_mut(p))
            .and_then(|prim| prim.get_mut("attributes"))
        else {
            continue;
        };
        for k in 0..had {
            attrs.remove(&format!("JOINTS_{k}"));
            attrs.remove(&format!("WEIGHTS_{k}"));
        }
        for (key, a) in out {
            attrs.insert(key, json!(a));
        }
    }
    for (r, ibm) in rests.into_iter().zip(rest_ibms) {
        let Some(Value::Array(nodes)) = json.get_mut("nodes") else {
            break;
        };
        let mut joints = skins[r.skin].clone();
        // The node that draws the mesh is the rest joint, unless it is one of the skin's joints already: then a
        // node of its own under it, which stands where it does.
        let rest = if joints.contains(&r.node) {
            nodes.push(json!({}));
            let child = nodes.len() - 1;
            if let Some(node) = nodes[r.node].as_object_mut() {
                match node.entry("children").or_insert_with(|| json!([])) {
                    Value::Array(kids) => kids.push(json!(child)),
                    other => *other = json!([child]),
                }
            }
            child
        } else {
            r.node
        };
        joints.push(rest);
        let Some(Value::Array(list)) = json.get_mut("skins") else {
            break;
        };
        let mut skin = list[r.skin].clone();
        skin["joints"] = json!(joints);
        if let Some(a) = ibm {
            skin["inverseBindMatrices"] = json!(a);
        }
        if r.copy {
            list.push(skin);
            let new = list.len() - 1;
            if let Some(Value::Array(nodes)) = json.get_mut("nodes") {
                nodes[r.node]["skin"] = json!(new);
            }
        } else {
            list[r.skin] = skin;
        }
    }
    let nodes = json.get("nodes").and_then(Value::as_array).cloned().unwrap_or_default();
    if let Some(Value::Array(skins)) = json.get_mut("skins") {
        set_skeletons(&nodes, skins);
    }
}

/// The identity, column-major.
#[cfg(feature = "gltf")]
const IDENTITY: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];

/// What [`conform`] changes in a document, worked out before anything is.
#[cfg(feature = "gltf")]
struct Plan {
    rewrites: Vec<Rewrite>,
    rests: Vec<Rest>,
    /// Each skin's joints.
    skins: Vec<Vec<usize>>,
}

/// A skinned primitive whose joints and weights are rewritten.
#[cfg(feature = "gltf")]
struct Rewrite {
    mesh: usize,
    prim: usize,
    /// How many sets of joints and weights it had.
    had: usize,
    sets: Sets,
    /// How many joints its skin has.
    count: usize,
}

/// A rest joint to add.
#[cfg(feature = "gltf")]
struct Rest {
    /// The node that draws the mesh.
    node: usize,
    skin: usize,
    /// The skin's inverse bind matrices, when it has an accessor of them (without, every one is the identity).
    ibms: Option<Vec<f32>>,
    /// Whether the joint goes into a copy of the skin, which another node draws with too.
    copy: bool,
}

#[cfg(feature = "gltf")]
impl Plan {
    fn of(doc: &crate::Document) -> Plan {
        let index = |v: &Value, k: &str| v.get(k)?.as_u64().and_then(|i| usize::try_from(i).ok());
        let json = doc.json();
        let list = |k: &str| json.get(k).and_then(Value::as_array).map_or(&[][..], Vec::as_slice);
        let (nodes, skin_list, meshes) = (list("nodes"), list("skins"), list("meshes"));
        let skins: Vec<Vec<usize>> = skin_list
            .iter()
            .map(|s| {
                s.get("joints")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|j| j.as_u64().and_then(|j| usize::try_from(j).ok()))
                    .collect()
            })
            .collect();
        // Each node that draws a mesh with a skin: the node, the mesh and the skin.
        let drawn: Vec<(usize, usize, usize)> = nodes
            .iter()
            .enumerate()
            .filter_map(|(n, node)| {
                let (m, s) = (index(node, "mesh")?, index(node, "skin")?);
                (m < meshes.len() && s < skins.len()).then_some((n, m, s))
            })
            .collect();
        // Each skinned mesh's number of joints: the same in every skin it's drawn with, or `None`.
        let mut counts: std::collections::BTreeMap<usize, Option<usize>> = std::collections::BTreeMap::new();
        for &(_, m, s) in &drawn {
            let n = skins[s].len();
            counts
                .entry(m)
                .and_modify(|c| {
                    if *c != Some(n) {
                        *c = None;
                    }
                })
                .or_insert(Some(n));
        }
        let mut rewrites = Vec::new();
        for (&m, &count) in &counts {
            let Some(count) = count.filter(|&c| c > 0) else {
                continue;
            };
            let prims = meshes[m].get("primitives").and_then(Value::as_array);
            for (p, prim) in prims.into_iter().flatten().enumerate() {
                if prim.pointer("/extensions/KHR_draco_mesh_compression").is_some() {
                    continue;
                }
                if let Some((had, sets)) = weighed(doc, prim, count) {
                    rewrites.push(Rewrite {
                        mesh: m,
                        prim: p,
                        had,
                        sets,
                        count,
                    });
                }
            }
        }
        let unbound: Vec<usize> = rewrites.iter().filter(|r| r.sets.unbound).map(|r| r.mesh).collect();
        let mut rests = Vec::new();
        for &(n, _, s) in drawn.iter().filter(|d| unbound.contains(&d.1)) {
            let ibms = match index(&skin_list[s], "inverseBindMatrices") {
                Some(a) => match doc.read_floats(a) {
                    Ok((16, v)) if v.len() == 16 * skins[s].len() => Some(v),
                    // Inverse bind matrices that can't be read: the skin is left as it is.
                    _ => continue,
                },
                None => None,
            };
            rests.push(Rest {
                node: n,
                skin: s,
                ibms,
                copy: drawn.iter().filter(|d| d.2 == s).count() > 1,
            });
        }
        Plan { rewrites, rests, skins }
    }
}

/// A skinned primitive's joints and weights through [`sets`], for a skin of `count` joints, and how many sets
/// it had; `None` when they are as the rule has them already, or can't be read.
#[cfg(feature = "gltf")]
fn weighed(doc: &crate::Document, prim: &Value, count: usize) -> Option<(usize, Sets)> {
    let attrs = prim.get("attributes")?;
    let index = |k: &str| attrs.get(k)?.as_u64().and_then(|i| usize::try_from(i).ok());
    let mut read = Vec::new();
    for k in 0.. {
        let (Some(j), Some(w)) = (index(&format!("JOINTS_{k}")), index(&format!("WEIGHTS_{k}"))) else {
            break;
        };
        match (doc.read_floats(j), doc.read_floats(w)) {
            (Ok((4, j)), Ok((4, w))) if j.len() == w.len() => read.push((j, w)),
            _ => return None,
        }
    }
    let vertices = read.first()?.0.len() / 4;
    if read.iter().any(|(j, _)| j.len() != vertices * 4) {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a joint index, an unsigned integer read exactly"
    )]
    let fill = |v: usize, out: &mut Vec<(u32, f32)>| {
        for (j, w) in &read {
            out.extend((v * 4..v * 4 + 4).map(|i| (j[i] as u32, w[i])));
        }
    };
    let mut one = Vec::new();
    let sound = (0..vertices).all(|v| {
        one.clear();
        fill(v, &mut one);
        if one.iter().any(|&(_, w)| !w.is_finite() || w < 0.0) {
            return false;
        }
        one.retain(|&(_, w)| w > 0.0);
        // As a validator checks a sum: in single precision, allowing a rounding for each weight.
        let sum: f32 = one.iter().map(|e| e.1).sum();
        let unit = f64::from((sum - 1.0).abs()) <= 2e-7 * f64::from(crate::num::u32_of(one.len().max(1)));
        one.sort_unstable_by_key(|e| e.0);
        unit && one.len() <= MAX_INFLUENCES
            && one.iter().all(|&(j, _)| (j as usize) < count)
            && one.windows(2).all(|w| w[0].0 != w[1].0)
    });
    (!sound).then(|| (read.len(), sets(vertices, count, crate::num::u32_of(count), fill)))
}

/// An inverse bind matrix (column-major) as glTF holds it: the identity when any of it isn't finite, and its last
/// row `0 0 0 1`.
pub(crate) fn inverse_bind(m: [f64; 16]) -> [f32; 16] {
    let mut m = if m.iter().all(|v| v.is_finite()) {
        m
    } else {
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]
    };
    (m[3], m[7], m[11], m[15]) = (0.0, 0.0, 0.0, 1.0);
    m.map(crate::num::f32_of)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(lists: &[&[(u32, f32)]], joints: usize, rest: u32) -> Sets {
        sets(lists.len(), joints, rest, |v, out| out.extend_from_slice(lists[v]))
    }

    #[test]
    fn influences_are_merged_sorted_and_sum_to_one() {
        // Joint 1 twice, joint 7 out of range; then a vertex with nothing.
        let s = of(&[&[(1, 0.25), (1, 0.25), (2, 1.0), (7, 1.0)], &[(0, 0.0)]], 3, 2);
        assert_eq!(s.groups, 1);
        assert_eq!(&s.joints[..4], &[2, 1, 0, 0]);
        assert!((s.weights[0] - 2.0 / 3.0).abs() < 1e-6 && (s.weights[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(&s.joints[4..8], &[2, 0, 0, 0]);
        assert_eq!(&s.weights[4..8], &[1.0, 0.0, 0.0, 0.0]);
        assert!(s.unbound);
    }

    #[test]
    fn sixteen_influences_are_kept_in_four_sets_and_no_more() {
        let many: Vec<(u32, f32)> = (0..20u16).map(|j| (u32::from(j), 1.0 + f32::from(j))).collect();
        let s = of(&[&many, &[(3, 2.0)]], 20, 0);
        assert_eq!(s.groups, 4);
        assert!(!s.unbound);
        // The strongest first: joint 19 down to joint 4, the four weakest dropped.
        let first: Vec<u32> = (0..4).map(|set| s.joints[set * 8]).collect();
        assert_eq!(first, [19, 15, 11, 7]);
        assert_eq!(s.joints[3 * 8 + 3], 4);
        let sum: f32 = (0..4).flat_map(|set| s.weights[set * 8..set * 8 + 4].to_vec()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        // The second vertex has one: the slots it leaves name joint 0 with no weight.
        assert_eq!(&s.joints[4..8], &[3, 0, 0, 0]);
        assert_eq!(&s.weights[4..8], &[1.0, 0.0, 0.0, 0.0]);
        assert!(s.joints[12..16].iter().chain(&s.joints[20..24]).all(|&j| j == 0));
    }

    #[test]
    fn a_vertex_no_joint_moves_goes_to_the_rest_joint_after_the_files() {
        // The rest joint is one past the file's two: a file's own weight on a joint 2 is out of range, not it.
        let s = of(&[&[(2, 1.0)], &[(1, 1.0)]], 2, 2);
        assert_eq!(&s.joints[..8], &[2, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&s.weights[..8], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(s.unbound);
        assert!(!of(&[&[(1, 1.0)]], 2, 2).unbound);
    }

    #[test]
    fn added_vertices_take_their_sources_joints() {
        let many: Vec<(u32, f32)> = (0..5).map(|j| (j, 1.0)).collect();
        let mut s = of(&[&[(1, 1.0)], &many], 5, 0);
        // Vertex 2 from vertex 1, vertex 3 from vertex 2.
        s.follow(&[1, 2]);
        assert_eq!((s.vertices, s.groups), (4, 2));
        assert_eq!(&s.joints[..16], &[1, 0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(&s.joints[16..], &[0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]);
        assert_eq!(&s.weights[28..], &[0.2, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn weights_too_large_to_add_still_sum_to_one() {
        let s = of(&[&[(0, f32::MAX), (1, f32::MAX)]], 2, 0);
        assert_eq!(&s.weights[..2], &[0.5, 0.5]);
    }

    #[test]
    fn the_common_root_is_the_nearest_node_above_every_joint() {
        //      0
        //    1   4
        //   2 3
        let parents = [None, Some(0), Some(1), Some(1), Some(0), None];
        assert_eq!(common_root(&parents, &[2, 3]), Some(1));
        assert_eq!(common_root(&parents, &[1, 2, 3]), Some(1));
        assert_eq!(common_root(&parents, &[2, 1]), Some(1));
        assert_eq!(common_root(&parents, &[2, 4]), Some(0));
        assert_eq!(common_root(&parents, &[3]), Some(3));
        assert_eq!(common_root(&parents, &[2, 5]), None);
        assert_eq!(common_root(&parents, &[]), None);
    }

    #[test]
    fn an_inverse_bind_matrix_is_finite_and_affine() {
        let mut m = [0.0; 16];
        m[0] = f64::NAN;
        let id = inverse_bind(m);
        assert_eq!(id[0], 1.0);
        assert_eq!(id[15], 1.0);
        let mut m = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        m[15] = 7.0;
        let out = inverse_bind(m);
        assert_eq!([out[3], out[7], out[11], out[15]], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(out[12], 12.0);
    }
}
