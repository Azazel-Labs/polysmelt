//! [`Document::rig`]: the node tree, skins and animations as typed values.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use serde_json::Value;

use super::{Reader, floats, index, items, list, list_of};
use crate::document::Document;
use crate::num::f32_of;
use crate::quat;
use crate::rig::{Animation, Channel, Interpolation, Node, Property, Rig, Skin, Trs};
use crate::{Context as _, ErrorKind, Result};

impl Document {
    /// The document's nodes, skins and animations, accessors read. A document loaded with a
    /// [`Keep`](crate::document::Keep) that drops animation, skins or morphs has none of those.
    ///
    /// # Errors
    ///
    /// When a skin or animation names an accessor, buffer or node that doesn't exist, or
    /// whose data runs past its buffer.
    pub fn rig(&self) -> Result<Rig> {
        let mut r = Reader::whole(self);
        let json = self.json();
        let name = |v: &Value| v.get("name").and_then(Value::as_str).map(str::to_owned);

        let raw = list(json, "nodes");
        let mut nodes: Vec<Node> = raw
            .iter()
            .map(|n| Node {
                name: name(n),
                parent: None,
                children: list_of(n, "children").into_iter().filter(|&c| c < raw.len()).collect(),
                rest: rest(n),
                mesh: index(n, "mesh"),
                skin: index(n, "skin"),
            })
            .collect();
        for i in 0..nodes.len() {
            for c in nodes[i].children.clone() {
                if c != i && nodes[c].parent.is_none() {
                    nodes[c].parent = Some(i);
                }
            }
        }

        let mut skins = Vec::new();
        for (k, s) in list(json, "skins").iter().enumerate() {
            let joints = list_of(s, "joints");
            if let Some(&j) = joints.iter().find(|&&j| j >= nodes.len()) {
                bail!(Corrupt, "glTF skin {k} names node {j}, which doesn't exist");
            }
            let inverse_bind = match index(s, "inverseBindMatrices") {
                Some(a) => {
                    let m = r.read::<16>(a, IDENTITY)?;
                    ensure!(
                        m.len() >= joints.len(),
                        Corrupt,
                        "glTF skin {k} has {} inverse bind matrices for {} joints",
                        m.len(),
                        joints.len()
                    );
                    m.into_iter().take(joints.len()).collect()
                }
                None => vec![IDENTITY; joints.len()],
            };
            skins.push(Skin {
                name: name(s),
                joints,
                inverse_bind,
                skeleton: index(s, "skeleton").filter(|&n| n < nodes.len()),
            });
        }

        let mut animations = Vec::new();
        let file_rate = self.source_info().frame_rate.value().copied();
        // Key times by accessor: channels mostly share them, every bone keyed on the same frames.
        let mut read_times: HashMap<usize, Vec<f32>> = HashMap::new();
        for (k, a) in list(json, "animations").iter().enumerate() {
            let samplers = items(a, "samplers");
            let mut channels = Vec::new();
            for ch in items(a, "channels") {
                let Some((node, path)) = ch.get("target").and_then(super::node_target) else {
                    // A channel for another extension's target, or none.
                    continue;
                };
                let property = match path {
                    "translation" => Property::Translation,
                    "rotation" => Property::Rotation,
                    "scale" => Property::Scale,
                    "weights" => Property::Weights,
                    _ => continue,
                };
                ensure!(
                    node < nodes.len(),
                    Corrupt,
                    "glTF animation {k} animates node {node}, which doesn't exist"
                );
                let s = index(ch, "sampler")
                    .and_then(|s| samplers.get(s))
                    .with_context(ErrorKind::Corrupt, || {
                        format!("glTF animation {k}: a channel names a missing sampler")
                    })?;
                let (Some(input), Some(output)) = (index(s, "input"), index(s, "output")) else {
                    bail!(Corrupt, "glTF animation {k}: a sampler has no input or output");
                };
                let interpolation = match s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR") {
                    "STEP" => Interpolation::Step,
                    "CUBICSPLINE" => Interpolation::CubicSpline,
                    _ => Interpolation::Linear,
                };
                let times = match read_times.entry(input) {
                    Entry::Occupied(t) => t.get().clone(),
                    Entry::Vacant(v) => v.insert(r.read::<1>(input, [0.0])?.into_flattened()).clone(),
                };
                let values = match property {
                    Property::Translation | Property::Scale => r.read::<3>(output, [0.0; 3])?.into_flattened(),
                    Property::Rotation => r.read::<4>(output, [0.0, 0.0, 0.0, 1.0])?.into_flattened(),
                    Property::Weights => r.read::<1>(output, [0.0])?.into_flattened(),
                };
                channels.push(Channel {
                    node,
                    property,
                    interpolation,
                    times,
                    values,
                });
            }
            let own_rate = a.pointer("/extras/frameRate").and_then(Value::as_f64);
            animations.push(Animation {
                name: name(a),
                channels,
                frame_rate: own_rate.or(file_rate).filter(|r| r.is_finite() && *r > 0.0),
            });
        }
        Ok(Rig {
            nodes,
            skins,
            animations,
        })
    }
}

const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// A node's own transform; a matrix is decomposed.
fn rest(n: &Value) -> Trs {
    if let Some(m) = n.get("matrix").and_then(Value::as_array).filter(|m| m.len() == 16) {
        let mut out = [0.0f64; 16];
        for (o, x) in out.iter_mut().zip(m) {
            *o = x.as_f64().unwrap_or(0.0);
        }
        return decompose(&out);
    }
    Trs {
        translation: floats(n.get("translation"), [0.0; 3]).map(f32_of),
        rotation: floats(n.get("rotation"), [0.0, 0.0, 0.0, 1.0]).map(f32_of),
        scale: floats(n.get("scale"), [1.0; 3]).map(f32_of),
    }
}

/// A column-major affine matrix as translation, rotation and scale. A mirroring matrix
/// negates the x scale; shear is lost.
fn decompose(m: &[f64; 16]) -> Trs {
    let col = |c: usize| [m[4 * c], m[4 * c + 1], m[4 * c + 2]];
    let len = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let (x, y, z) = (col(0), col(1), col(2));
    let det =
        x[0] * (y[1] * z[2] - y[2] * z[1]) - y[0] * (x[1] * z[2] - x[2] * z[1]) + z[0] * (x[1] * y[2] - x[2] * y[1]);
    let mut s = [len(x), len(y), len(z)];
    if det < 0.0 {
        s[0] = -s[0];
    }
    let unit = |v: [f64; 3], k: f64| if k == 0.0 { [0.0; 3] } else { v.map(|c| c / k) };
    let (x, y, z) = (unit(x, s[0]), unit(y, s[1]), unit(z, s[2]));
    // The rotation's columns are x, y and z.
    let q = quat::unit(quat::of_rotation_reciprocal(|r, c| [x, y, z][c][r]));
    Trs {
        translation: [m[12], m[13], m[14]].map(f32_of),
        rotation: q.map(f32_of),
        scale: s.map(f32_of),
    }
}

#[cfg(test)]
mod tests {
    use super::decompose;

    #[test]
    fn decompose_undoes_trs() {
        // Scale 2, 3, 4; 90° about z; moved 1, 2, 3.
        let m = [
            0.0, 2.0, 0.0, 0.0, //
            -3.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 4.0, 0.0, //
            1.0, 2.0, 3.0, 1.0,
        ];
        let t = decompose(&m);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(t.translation, [1.0, 2.0, 3.0]);
        assert!(t.scale.iter().zip([2.0, 3.0, 4.0]).all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(
            t.rotation
                .iter()
                .zip([0.0, 0.0, h, h])
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }
}
