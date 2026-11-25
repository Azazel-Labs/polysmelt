//! Skins as skin deformers with a cluster per joint and a bind pose; morph targets as blend shapes, a channel and
//! a shape per target.

use serde_json::Value;

use super::geometry::Geom;
use super::records::{Node, p_f64, props70};
use super::{Trs, Writer, fbx_name, index, list, trs_matrix};
use crate::Result;
use crate::formats::xform::{IDENTITY, M4, decompose, from_column_major, inverse, mul, to_column_major};

/// A primitive's attribute accessor.
fn attribute(prim: &Value, name: &str) -> Option<usize> {
    prim.get("attributes").and_then(|a| index(a, name))
}

impl Writer<'_> {
    /// A skin's inverse bind matrices; none when it has none (each joint's is then the identity).
    fn inverse_binds(&self, skin: &Value) -> Result<Vec<M4>> {
        let Some(a) = index(skin, "inverseBindMatrices") else {
            return Ok(Vec::new());
        };
        let (w, m) = self.doc.read_floats(a)?;
        Ok(if w == 16 {
            m.chunks_exact(16)
                .map(|c| from_column_major(&std::array::from_fn(|k| f64::from(c[k]))))
                .collect()
        } else {
            Vec::new()
        })
    }

    /// By node index, the joint a skinned node's mesh is folded onto, with the geometric transform it takes there.
    ///
    /// This crate's reader keeps a skinned mesh's model as the skin's last joint (for vertices no cluster weights)
    /// and draws the mesh from a root node of the same name, as glTF asks. Writing the two as one model again
    /// writes the file they came from, where two models would read back as three nodes. It's done only when that
    /// loses nothing: the node holds just its mesh and skin, with no children, extras or animated transform; the
    /// joint holds nothing; every vertex on the joint is on it alone; and its inverse bind matrix is a plain
    /// transform, which becomes the geometric one.
    pub(super) fn folds(&self, keep: &[bool], parents: &[Option<usize>]) -> Result<Vec<Option<(usize, Trs)>>> {
        let nodes = list(self.json, "nodes");
        let mut out = vec![None; nodes.len()];
        let mut taken = vec![false; nodes.len()];
        let mut moved = vec![false; nodes.len()];
        for anim in list(self.json, "animations") {
            for c in anim.get("channels").and_then(Value::as_array).into_iter().flatten() {
                let target = c.get("target").unwrap_or(&Value::Null);
                if target.get("path").and_then(Value::as_str) != Some("weights")
                    && let Some(f) = index(target, "node").and_then(|n| moved.get_mut(n))
                {
                    *f = true;
                }
            }
        }
        let hidden = |n: &Value| n.pointer("/extensions/KHR_node_visibility/visible") == Some(&Value::Bool(false));
        let hidden_under = |mut n: usize| {
            for _ in 0..=nodes.len() {
                if hidden(&nodes[n]) {
                    return true;
                }
                match parents.get(n).copied().flatten() {
                    Some(p) => n = p,
                    None => break,
                }
            }
            false
        };
        let name = |n: usize| nodes[n].get("name").and_then(Value::as_str).unwrap_or("");
        let holds = |n: &Value| {
            ["mesh", "skin", "camera"].iter().any(|k| n.get(*k).is_some())
                || n.pointer("/extensions/KHR_lights_punctual").is_some()
        };
        for (i, node) in nodes.iter().enumerate() {
            let Some(obj) = node.as_object() else { continue };
            let mesh = index(node, "mesh").and_then(|m| list(self.json, "meshes").get(m));
            let skin = index(node, "skin").and_then(|s| list(self.json, "skins").get(s));
            let (Some(mesh), Some(skin)) = (mesh, skin) else {
                continue;
            };
            let plain = obj.iter().all(|(k, v)| match k.as_str() {
                "name" | "mesh" | "skin" | "weights" | "translation" | "rotation" | "scale" | "matrix" => true,
                "extensions" => v
                    .as_object()
                    .is_some_and(|e| e.keys().all(|k| k == "KHR_node_visibility")),
                _ => false,
            });
            let joints = skin
                .get("joints")
                .and_then(Value::as_array)
                .map_or(&[][..], Vec::as_slice);
            let Some(j) = joints
                .last()
                .and_then(Value::as_u64)
                .and_then(|j| usize::try_from(j).ok())
            else {
                continue;
            };
            if !plain
                || !keep[i]
                || moved[i]
                || j == i
                || j >= nodes.len()
                || !keep[j]
                || taken[j]
                || holds(&nodes[j])
                || !(name(i).is_empty() || name(i) == name(j))
                || hidden(node) != hidden_under(j)
            {
                continue;
            }
            let ibm = self
                .inverse_binds(skin)?
                .get(joints.len() - 1)
                .copied()
                .unwrap_or(IDENTITY);
            let d = decompose(&ibm);
            let g = (d.t, d.r, d.s);
            let back = trs_matrix(g);
            let tol = 1e-6 * (1.0 + ibm.iter().flatten().fold(0.0_f64, |a, x| a.max(x.abs())));
            let plain_transform = d.exact
                && ibm
                    .iter()
                    .flatten()
                    .zip(back.iter().flatten())
                    .all(|(a, b)| (a - b).abs() <= tol);
            if !plain_transform || !self.alone_on(mesh, joints.len() - 1)? {
                continue;
            }
            taken[j] = true;
            out[i] = Some((j, g));
        }
        Ok(out)
    }

    /// Whether every vertex of `mesh` that skin joint `joint` weights is weighted by it alone, and some vertex by
    /// another joint: without one, no cluster would weight anything and the mesh would read back unskinned.
    fn alone_on(&self, mesh: &Value, joint: usize) -> Result<bool> {
        let mut others = false;
        for prim in mesh.get("primitives").and_then(Value::as_array).into_iter().flatten() {
            let mut sets = Vec::new();
            for set in 0.. {
                let (Some(ja), Some(wa)) = (
                    attribute(prim, &format!("JOINTS_{set}")),
                    attribute(prim, &format!("WEIGHTS_{set}")),
                ) else {
                    break;
                };
                let (jw, js) = self.doc.read_floats(ja)?;
                let (ww, ws) = self.doc.read_floats(wa)?;
                if jw != 4 || ww != 4 || js.len() != ws.len() {
                    return Ok(false);
                }
                sets.push((js, ws));
            }
            let count = sets.first().map_or(0, |s| s.0.len() / 4);
            for v in 0..count {
                let (mut on, mut off) = (0.0_f32, 0.0_f32);
                for (js, ws) in &sets {
                    for k in v * 4..v * 4 + 4 {
                        let w = ws.get(k).copied().unwrap_or(0.0).max(0.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a joint index")]
                        let mine = js.get(k).is_some_and(|&j| j as usize == joint);
                        if mine {
                            on += w;
                        } else {
                            off += w;
                        }
                    }
                }
                if on > 0.0 && off > 1e-6 {
                    return Ok(false);
                }
                others |= off > 0.0;
            }
        }
        Ok(others)
    }

    /// Writes node `i`'s skin as a deformer of geometry `g`, with a cluster for each joint and a bind pose.
    pub(super) fn skin(&mut self, i: usize, g: &Geom) -> Result<()> {
        let Some(gid) = g.id else { return Ok(()) };
        let node = &list(self.json, "nodes")[i];
        let Some(skin) = index(node, "skin").and_then(|s| list(self.json, "skins").get(s)) else {
            return Ok(());
        };
        let joints: Vec<usize> = skin
            .get("joints")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|j| j.as_u64().and_then(|j| usize::try_from(j).ok()))
            .collect();
        let ibms = self.inverse_binds(skin)?;
        // Each joint's control points and weights, a vertex's weights for one joint summed.
        let mut per_joint: Vec<(Vec<i32>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); joints.len()];
        let mesh = index(node, "mesh").map_or(&Value::Null, |m| &list(self.json, "meshes")[m]);
        let prims = mesh
            .get("primitives")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        for (p, prim) in prims.iter().enumerate() {
            let Some(base) = g.bases.get(p).copied().flatten() else {
                continue;
            };
            for set in 0.. {
                let (Some(ja), Some(wa)) = (
                    attribute(prim, &format!("JOINTS_{set}")),
                    attribute(prim, &format!("WEIGHTS_{set}")),
                ) else {
                    break;
                };
                let (jw, js) = self.doc.read_floats(ja)?;
                let (ww, ws) = self.doc.read_floats(wa)?;
                if jw != 4 || ww != 4 {
                    continue;
                }
                for (v, (jv, wv)) in js.chunks_exact(4).zip(ws.chunks_exact(4)).enumerate() {
                    let Ok(cp) = i32::try_from(base + v) else { break };
                    for k in 0..4 {
                        let w = f64::from(wv[k]);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a joint index")]
                        let j = jv[k] as usize;
                        if w <= 0.0 || j >= joints.len() {
                            continue;
                        }
                        let (ix, ws) = &mut per_joint[j];
                        if ix.last() == Some(&cp) {
                            *ws.last_mut().expect("beside its index") += w;
                        } else {
                            ix.push(cp);
                            ws.push(w);
                        }
                    }
                }
            }
        }
        // Weights from a second set of four land after the first's: put each joint's in control point order.
        for (ix, ws) in &mut per_joint {
            if !ix.is_sorted() {
                let mut pairs: Vec<(i32, f64)> = ix.iter().copied().zip(ws.iter().copied()).collect();
                pairs.sort_by_key(|p| p.0);
                let mut merged: Vec<(i32, f64)> = Vec::with_capacity(pairs.len());
                for (c, w) in pairs {
                    match merged.last_mut() {
                        Some(last) if last.0 == c => last.1 += w,
                        _ => merged.push((c, w)),
                    }
                }
                (*ix, *ws) = merged.into_iter().unzip();
            }
        }
        // A mesh folded onto its skin's last joint has that joint for its model, and the vertices on the joint alone
        // are the ones no cluster weights, so it gets no cluster; the geometric transform comes off the others'.
        let at = self.at(i);
        let m = self.models[at].as_ref().expect("a written node");
        let (mesh_world, mesh_model, name) = (m.world, m.id, m.name.clone());
        let unbind = inverse(&m.geometric.map_or(IDENTITY, trs_matrix));
        let skin_id = self.id();
        let skin_node = Node::new("Deformer")
            .i64(skin_id)
            .str(&fbx_name(&name, "Deformer"))
            .str("Skin")
            .child("Version", |n| n.i32(101))
            .child("Link_DeformAcuracy", |n| n.f64(50.0))
            .child("SkinningType", |n| n.str("Linear"));
        self.object("Deformer", skin_node);
        self.oo(skin_id, gid);
        let mut pose = vec![(mesh_model, mesh_world)];
        for (k, &j) in joints.iter().enumerate() {
            if at != i && j == at {
                continue;
            }
            let Some(joint) = self.models.get(j).and_then(Option::as_ref) else {
                continue;
            };
            let (jid, jname) = (joint.id, joint.name.clone());
            let ibm = mul(&ibms.get(k).copied().unwrap_or(IDENTITY), &unbind);
            // The joint's world transform at bind time, as the mesh's world transform and the inverse bind matrix
            // give it.
            let link = mul(&mesh_world, &inverse(&ibm));
            let (ix, ws) = std::mem::take(&mut per_joint[k]);
            let cid = self.id();
            let mut c = Node::new("Deformer")
                .i64(cid)
                .str(&fbx_name(&jname, "SubDeformer"))
                .str("Cluster")
                .child("Version", |n| n.i32(100))
                .child("UserData", |n| n.str("").str(""));
            if !ix.is_empty() {
                c = c.child("Indexes", |n| n.arr(&ix)).child("Weights", |n| n.arr(&ws));
            }
            let c = c
                .child("Transform", |n| n.arr(&to_column_major(&ibm)))
                .child("TransformLink", |n| n.arr(&to_column_major(&link)));
            self.object("Deformer", c);
            self.oo(cid, skin_id);
            self.oo(jid, cid);
            if !pose.iter().any(|p| p.0 == jid) {
                pose.push((jid, link));
            }
        }
        let pid = self.id();
        let mut p = Node::new("Pose")
            .i64(pid)
            .str(&fbx_name(&name, "Pose"))
            .str("BindPose")
            .child("Type", |n| n.str("BindPose"))
            .child("Version", |n| n.i32(100))
            .child("NbPoseNodes", |n| n.i32(i32::try_from(pose.len()).unwrap_or(i32::MAX)));
        for (id, m) in pose {
            p.push(
                Node::new("PoseNode")
                    .child("Node", |n| n.i64(id))
                    .child("Matrix", |n| n.arr(&to_column_major(&m))),
            );
        }
        self.object("Pose", p);
        Ok(())
    }

    /// Writes mesh `mesh`'s morph targets, as node `i` draws them, as a blend shape of geometry `g`: a channel per
    /// target, at the node's weight, holding a shape of the control points the target moves.
    pub(super) fn morphs(&mut self, i: usize, mesh: usize, g: &Geom) -> Result<()> {
        let Some(gid) = g.id else { return Ok(()) };
        let mesh_json = &list(self.json, "meshes")[mesh];
        let node = &list(self.json, "nodes")[i];
        let prims = mesh_json
            .get("primitives")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice);
        let targets = prims
            .iter()
            .filter_map(|p| p.get("targets").and_then(Value::as_array).map(Vec::len))
            .max()
            .unwrap_or(0);
        let weights: Vec<f64> = node
            .get("weights")
            .or_else(|| mesh_json.get("weights"))
            .and_then(Value::as_array)
            .map(|w| w.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
            .unwrap_or_default();
        let names = mesh_json.pointer("/extras/targetNames").and_then(Value::as_array);
        let model_name = self.models[self.at(i)].as_ref().expect("a written node").name.clone();
        let bs = self.id();
        let blend = Node::new("Deformer")
            .i64(bs)
            .str(&fbx_name(&model_name, "Deformer"))
            .str("BlendShape")
            .child("Version", |n| n.i32(100));
        self.object("Deformer", blend);
        self.oo(bs, gid);
        let mut channels = Vec::with_capacity(targets);
        for t in 0..targets {
            let mut ix: Vec<i32> = Vec::new();
            let mut dv: Vec<f64> = Vec::new();
            let mut dn: Vec<f64> = Vec::new();
            let mut normals = false;
            for (p, prim) in prims.iter().enumerate() {
                let Some(base) = g.bases.get(p).copied().flatten() else {
                    continue;
                };
                let Some(target) = prim.get("targets").and_then(Value::as_array).and_then(|a| a.get(t)) else {
                    continue;
                };
                let pos = match index(target, "POSITION") {
                    Some(a) => Some(self.doc.read_floats(a)?).filter(|r| r.0 == 3).map(|r| r.1),
                    None => None,
                };
                let nrm = match index(target, "NORMAL") {
                    Some(a) => Some(self.doc.read_floats(a)?).filter(|r| r.0 == 3).map(|r| r.1),
                    None => None,
                };
                normals |= nrm.is_some();
                let count = pos.as_ref().or(nrm.as_ref()).map_or(0, |v| v.len() / 3);
                for v in 0..count {
                    let get = |a: &Option<Vec<f32>>| {
                        a.as_ref()
                            .and_then(|a| a.get(v * 3..v * 3 + 3))
                            .map_or([0.0; 3], |d| [d[0], d[1], d[2]].map(f64::from))
                    };
                    let (d, n) = (get(&pos), get(&nrm));
                    if d.iter().chain(&n).all(|&x| x == 0.0) {
                        continue;
                    }
                    let Ok(cp) = i32::try_from(base + v) else { break };
                    ix.push(cp);
                    dv.extend_from_slice(&d);
                    dn.extend_from_slice(&n);
                }
            }
            // A target that moves nothing still holds a point, so it keeps its place among the channels.
            if ix.is_empty() {
                ix.push(0);
                dv.extend_from_slice(&[0.0; 3]);
                dn.extend_from_slice(&[0.0; 3]);
            }
            let name = names
                .and_then(|n| n.get(t))
                .and_then(Value::as_str)
                .map_or_else(|| format!("target_{t}"), str::to_owned);
            let percent = weights.get(t).copied().unwrap_or(0.0) * 100.0;
            let ch = self.id();
            let channel = Node::new("Deformer")
                .i64(ch)
                .str(&fbx_name(&name, "SubDeformer"))
                .str("BlendShapeChannel")
                .child("Version", |n| n.i32(100))
                .with(props70(vec![p_f64("DeformPercent", "Number", "A", percent)]))
                .child("DeformPercent", |n| n.f64(percent))
                .child("FullWeights", |n| n.arr(&[100.0f64]));
            self.object("Deformer", channel);
            self.oo(ch, bs);
            let sid = self.id();
            let mut shape = Node::new("Geometry")
                .i64(sid)
                .str(&fbx_name(&name, "Geometry"))
                .str("Shape")
                .with(props70(Vec::new()))
                .child("Version", |n| n.i32(100))
                .child("Indexes", |n| n.arr(&ix))
                .child("Vertices", |n| n.arr(&dv));
            if normals {
                shape = shape.child("Normals", |n| n.arr(&dn));
            }
            self.object("Geometry", shape);
            self.oo(sid, ch);
            channels.push(ch);
        }
        self.channels.insert(i, channels);
        Ok(())
    }
}
