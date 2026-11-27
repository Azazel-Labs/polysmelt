//! Skins from skin deformers' clusters, and morph targets from blend shape channels.

use std::collections::HashMap;

use serde_json::json;

use super::{Reader, child_attr, floats, ints, name_of, scalar, string};
use crate::formats::fbx_tree::{Attr as A, NodeHandle};
use crate::formats::xform::{M4, from_column_major, inverse, mul, to_column_major};
use crate::num::f32_of;

/// A blend shape target: its name, and by control point, its offsets to position and normal.
pub(super) struct Target {
    pub(super) name: String,
    pub(super) positions: HashMap<usize, [f32; 3]>,
    pub(super) normals: HashMap<usize, [f32; 3]>,
}

/// A blend shape channel: its weight (a fraction, not clamped), and its targets with the weight each is reached
/// in full at, ascending: the in-betweens, then the full shape.
pub(super) struct Channel {
    pub(super) weight: f64,
    pub(super) full: Vec<f64>,
    pub(super) targets: Vec<Target>,
}

/// A shape's offsets by control point, to position and to normal; ones that aren't numbers are left out.
fn offsets(shape: NodeHandle<'_>) -> (HashMap<usize, [f32; 3]>, HashMap<usize, [f32; 3]>) {
    let idx = child_attr(shape, "Indexes").and_then(ints).unwrap_or_default();
    let read = |name: &str| {
        let data = child_attr(shape, name).and_then(floats).unwrap_or_default();
        let mut out = HashMap::new();
        for (k, &cp) in idx.iter().enumerate() {
            if let (Ok(cp), Some(d)) = (usize::try_from(cp), data.get(k * 3..k * 3 + 3))
                && d.iter().all(|x| x.is_finite())
            {
                out.insert(cp, [d[0], d[1], d[2]].map(f32_of));
            }
        }
        out
    };
    (read("Vertices"), read("Normals"))
}

impl Reader<'_, '_, '_> {
    /// The blend shape channels deforming geometry `g`, in target order.
    pub(super) fn morph_channels(&self, g: i64) -> Vec<i64> {
        self.objs
            .children_of(g)
            .filter(|d| self.objs.is_kind("Deformer", *d, "BlendShape"))
            .flat_map(|d| self.objs.children_of(d).collect::<Vec<_>>())
            .filter(|c| self.objs.is_kind("Deformer", *c, "BlendShapeChannel"))
            .collect()
    }

    /// The skin deformer's clusters on geometry `g`: how many there are, and by control point, (cluster number,
    /// weight) as the clusters list them. Clusters whose bone isn't a model are left out. The shared writer keeps
    /// the strongest and makes them sum to one ([`crate::skin`]).
    pub(super) fn influences(&self, g: i64) -> (usize, Vec<Vec<(u32, f32)>>) {
        let mut out: Vec<Vec<(u32, f32)>> = Vec::new();
        let Some(skin) = self
            .objs
            .children_of(g)
            .find(|d| self.objs.is_kind("Deformer", *d, "Skin"))
        else {
            return (0, out);
        };
        let clusters = self.clusters(skin);
        // Indexes past the control points weight nothing: no more room than those take.
        let cps = self
            .objs
            .node(g)
            .and_then(|n| child_attr(n, "Vertices"))
            .map_or(0, |v| match v {
                A::ArrF64(a) => a.len() / 3,
                A::ArrF32(a) => a.len() / 3,
                _ => 0,
            });
        for (j, &cluster) in clusters.iter().enumerate() {
            let Some(node) = self.objs.node(cluster) else { continue };
            let (Some(idx), Some(w)) = (child_attr(node, "Indexes"), child_attr(node, "Weights")) else {
                continue;
            };
            let (Some(idx), Some(w)) = (ints(idx), floats(w)) else {
                continue;
            };
            for (&cp, &w) in idx.iter().zip(&w) {
                if let Ok(cp) = usize::try_from(cp)
                    && cp < cps
                    && w > 0.0
                    && w.is_finite()
                {
                    if out.len() <= cp {
                        out.resize_with(cp + 1, Vec::new);
                    }
                    out[cp].push((crate::num::u32_of(j), f32_of(w)));
                }
            }
        }
        (clusters.len(), out)
    }

    /// A skin deformer's clusters that name a bone.
    fn clusters(&self, skin: i64) -> Vec<i64> {
        self.objs
            .children_of(skin)
            .filter(|c| self.objs.is_kind("Deformer", *c, "Cluster"))
            .filter(|c| self.objs.children_of(*c).any(|b| self.node_of.contains_key(&b)))
            .collect()
    }

    /// The skin for deformer `skin` on model `m`: a joint per cluster, and when `unbound` (some vertex has no
    /// cluster weighting it) the model's own node last, which those vertices stay with.
    pub(super) fn skin(&mut self, skin: i64, m: i64, geometric: &M4, unbound: bool) -> usize {
        if let Some(&s) = self.skin_of.get(&(skin, m)) {
            return s;
        }
        let mut joints = Vec::new();
        let mut ibms = Vec::new();
        for cluster in self.clusters(skin) {
            let Some(node) = self.objs.node(cluster) else { continue };
            let bone = self
                .objs
                .children_of(cluster)
                .find(|b| self.node_of.contains_key(b))
                .unwrap_or(0);
            let matrix = |name: &str| {
                child_attr(node, name)
                    .and_then(floats)
                    .and_then(|v| <[f64; 16]>::try_from(&v[..]).ok())
                    .filter(|v| v.iter().all(|x| x.is_finite()))
                    .map(|v| from_column_major(&v))
            };
            // `Transform` takes the model's space to the bone's at bind time: the file stores it relative to
            // `TransformLink` (the bone's bind-time world), though the FBX SDK hands back `TransformLink ·
            // Transform`, the model's world. Without it, the bind pose is the pose the file is in.
            let transform = matrix("Transform").unwrap_or_else(|| {
                let link = matrix("TransformLink").unwrap_or_else(|| self.world_of(bone));
                mul(&inverse(&link), &self.world_of(m))
            });
            // The vertex goes geometry → model → bone space.
            ibms.extend(crate::skin::inverse_bind(to_column_major(
                &self.seen(&mul(&transform, geometric)),
            )));
            joints.push(self.node_of[&bone]);
        }
        // The model's own node: at bind time it's where `Transform` says, so its inverse bind matrix is the
        // geometric transform.
        if unbound || joints.is_empty() {
            joints.push(self.node_of[&m]);
            ibms.extend(crate::skin::inverse_bind(to_column_major(&self.seen(geometric))));
        }
        let acc = self.b.floats(ibms, "MAT4", false);
        let mut out = json!({ "joints": joints, "inverseBindMatrices": acc });
        // glTF skins blend linearly; a skin meant to blend otherwise (dual quaternion, or a mix) says so.
        if let Some(kind) = self
            .objs
            .node(skin)
            .and_then(|n| child_attr(n, "SkinningType"))
            .and_then(string)
            .filter(|k| !matches!(*k, "" | "Linear" | "Rigid"))
        {
            out["extras"] = json!({ "skinningType": kind });
        }
        self.skins.push(out);
        let s = self.skins.len() - 1;
        self.skin_of.insert((skin, m), s);
        s
    }

    /// A blend shape channel: its weight, and its shapes as targets, ordered by the weight each is reached in full
    /// at (its `FullWeights`, in percent; evenly spread, in the order connected, when those are missing or don't
    /// rise). The full shape is named for the channel, an in-between for its own shape.
    pub(super) fn channel(&self, channel: i64) -> Option<Channel> {
        let node = self.objs.node(channel)?;
        let shapes: Vec<i64> = self
            .objs
            .children_of(channel)
            .filter(|s| self.objs.is_kind("Geometry", *s, "Shape"))
            .collect();
        if shapes.is_empty() {
            return None;
        }
        let given = child_attr(node, "FullWeights").and_then(floats).unwrap_or_default();
        let mut paired: Vec<(f64, i64)> = shapes
            .iter()
            .enumerate()
            .map(|(k, &s)| (given.get(k).map_or(f64::NAN, |w| w / 100.0), s))
            .collect();
        paired.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rising =
            paired.iter().all(|(w, _)| w.is_finite() && *w > 0.0) && paired.windows(2).all(|w| w[1].0 > w[0].0);
        if !rising {
            #[allow(clippy::cast_precision_loss, reason = "a few shapes")]
            let even = |k: usize| (k + 1) as f64 / shapes.len() as f64;
            paired = shapes.iter().enumerate().map(|(k, &s)| (even(k), s)).collect();
        }
        let name = name_of(node).unwrap_or("");
        let targets = paired
            .iter()
            .enumerate()
            .map(|(k, &(_, s))| {
                let shape = self.objs.node(s);
                let (positions, normals) = shape.map(offsets).unwrap_or_default();
                let name = match shape.and_then(name_of) {
                    Some(own) if k + 1 < paired.len() => own,
                    _ => name,
                };
                Target {
                    name: name.to_owned(),
                    positions,
                    normals,
                }
            })
            .collect();
        let weight = scalar(&self.objs.props(channel), "DeformPercent").unwrap_or(0.0) / 100.0;
        Some(Channel {
            weight,
            full: paired.iter().map(|p| p.0).collect(),
            targets,
        })
    }
}

/// A blend shape channel's weight (a fraction; not clamped, so a rig may push past full or below none) shared
/// among its targets, each reached in full at its weight in `full` (ascending): between two, the two blend; past
/// the last (or before the first), the nearest two carry on as they were going.
pub(super) fn split_weight(w: f64, full: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    let n = full.len().min(out.len());
    if n == 0 || !w.is_finite() {
        return;
    }
    if n == 1 || w <= full[0] {
        out[0] = w / full[0];
        return;
    }
    let i = full[..n].partition_point(|&f| f < w).clamp(1, n - 1);
    let (lo, hi) = (full[i - 1], full[i]);
    out[i - 1] = (hi - w) / (hi - lo);
    out[i] = (w - lo) / (hi - lo);
}
