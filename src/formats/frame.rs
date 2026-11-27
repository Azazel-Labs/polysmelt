//! The one way every reader brings a file's coordinate system and unit to glTF's (+Y up, right-handed, meters).
//!
//! A format with a scene (nodes, meshes, skeletons, animation, cameras, lights) is read as it is written, in the
//! file's own axes and units, and every root of the file is put under one node, [`ROOT`], which carries the
//! [`Frame`]: the turn that brings the file's up axis to +Y, and the scale from the file's unit to meters. The node
//! is always there, whether or not the file needs converting, so a document has the same shape whatever the file's
//! axes and units: one top-level node named `RootNode`, and the file's own roots as its children. A file that
//! needs nothing has a `RootNode` with no transform. Skinned meshes, skeleton roots, cameras and lights are under
//! it like everything else: a skinned mesh is placed by its joints, and they are under the root.
//!
//! [`root`] builds that node and is the only thing that does. It also records what it converted in
//! `asset.extras.source.converted`: `upAxis` when the frame turns, `units` when it scales, and neither for a
//! frame that changes nothing.
//!
//! A left-handed file is mirrored where it is read, into the vertex data, the transforms and the keys, with
//! triangles wound the other way: a mirror is never left as a negative scale on a node.
//!
//! Sampled data with no scene (a point cloud, gaussian splats, a voxel grid) is converted in its data instead, and
//! has no `RootNode`: [`flat`] says so. So is glTF itself, which is already in glTF's frame and is kept as it is.

use serde_json::{Map, Value, json};

use super::xform;
use crate::document::Document;

/// The name of the node every converted scene hangs from.
pub(crate) const ROOT: &str = "RootNode";

/// What brings a file's coordinates to glTF's: a turn (a unit quaternion, x y z w) and then nothing but a uniform
/// scale, the meters in one of the file's units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Frame {
    pub rotation: [f64; 4],
    pub scale: f64,
}

impl Default for Frame {
    fn default() -> Self {
        Frame::IDENTITY
    }
}

#[cfg_attr(
    not(feature = "default"),
    allow(dead_code, reason = "each reader uses the parts its format needs")
)]
impl Frame {
    /// A file already in glTF's axes, in meters or in no known unit.
    pub(crate) const IDENTITY: Frame = Frame {
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: 1.0,
    };

    /// A file with +Z up and +X right: a quarter turn about X.
    pub(crate) fn z_up() -> Frame {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        Frame {
            rotation: [-h, 0.0, 0.0, h],
            scale: 1.0,
        }
    }

    /// The frame for a file whose up axis is `up`, with X to the right for `+Y` and `+Z` up, in meters or no
    /// known unit.
    pub(crate) fn up(up: crate::source::Axis) -> Frame {
        use crate::source::Axis;
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let rotation = match up {
            Axis::PosY => [0.0, 0.0, 0.0, 1.0],
            Axis::PosZ => [-h, 0.0, 0.0, h],
            Axis::NegZ => [h, 0.0, 0.0, h],
            Axis::NegY => [1.0, 0.0, 0.0, 0.0],
            Axis::PosX => [0.0, 0.0, h, h],
            Axis::NegX => [0.0, 0.0, -h, h],
        };
        Frame { rotation, scale: 1.0 }
    }

    /// The frame a basis matrix is: its turn and its scale (taken as uniform). A basis that mirrors, which no
    /// frame holds, also gives the axis the reader is to mirror the scene through where it reads it
    /// (`basis = turn · scale · mirror`): the one that leaves the least turn, so a left-handed file that is
    /// Y up already is mirrored and not turned.
    pub(crate) fn of(basis: &xform::M4) -> (Frame, Option<usize>) {
        let frame = |m: &xform::M4| {
            let d = xform::decompose(m);
            Frame {
                rotation: d.r,
                scale: d.s[1].abs(),
            }
        };
        if xform::det3(basis) >= 0.0 {
            return (frame(basis), None);
        }
        let mut best = (frame(basis), 0, -1.0);
        for axis in 0..3 {
            // The basis with that axis mirrored first: a turn and a scale.
            let mut m = *basis;
            for row in m.iter_mut().take(3) {
                row[axis] = -row[axis];
            }
            let f = frame(&m);
            if f.rotation[3].abs() > best.2 {
                best = (f, axis, f.rotation[3].abs());
            }
        }
        (best.0, Some(best.1))
    }

    /// This frame turned by unit quaternion `q` instead.
    pub(crate) fn turned(self, q: [f64; 4]) -> Frame {
        Frame { rotation: q, ..self }
    }

    /// This frame for a file whose unit is `meters` long. A length that isn't a positive number is no unit at
    /// all: nothing is scaled.
    pub(crate) fn scaled(self, meters: f64) -> Frame {
        let scale = if meters.is_finite() && meters > 0.0 {
            meters
        } else {
            1.0
        };
        Frame { scale, ..self }
    }

    /// Whether the frame turns the scene.
    pub(crate) fn turns(&self) -> bool {
        (self.rotation[3].abs() - 1.0).abs() > 1e-12
    }

    /// Whether the frame scales the scene.
    pub(crate) fn scales(&self) -> bool {
        (self.scale - 1.0).abs() > 1e-12
    }
}

/// Puts every root of `doc`'s scenes under one [`ROOT`] node that carries `frame`, and records what the frame
/// converts in the document's source facts. Every reader of a format with a scene ends here, whatever the
/// file's axes and units; a scene with nothing in it stays empty.
pub(crate) fn root(doc: &mut Document, frame: Frame) {
    wrap(doc.base_mut(), frame);
    placed(doc, frame);
}

/// As [`root`], for a frame whose scale is a factor the file gives and not the length of a unit: the root node
/// scales by it, and no unit is said to be converted, since none is known.
#[cfg_attr(not(feature = "3ds"), allow(dead_code, reason = "only 3DS has such a factor"))]
pub(crate) fn root_by_factor(doc: &mut Document, frame: Frame) {
    wrap(doc.base_mut(), frame);
    placed(doc, Frame { scale: 1.0, ..frame });
}

/// The root node itself, with no children yet: [`ROOT`] carrying `frame`, each part left out when it changes
/// nothing. [`root`] makes it; a reader that builds its nodes from the root down makes it first and says so with
/// [`placed`].
pub(crate) fn node(frame: Frame) -> Value {
    let mut node = json!({ "name": ROOT });
    xform::set_trs(&mut node, ([0.0; 3], frame.rotation.map(exact), [frame.scale; 3]));
    node
}

/// A part of a turn between axes as its exact value, so the same turn is the same numbers whichever reader
/// worked it out, in single precision or from a matrix: 0, a half, a root of a half or 1, of either sign.
fn exact(c: f64) -> f64 {
    [0.0, 0.5, std::f64::consts::FRAC_1_SQRT_2, 1.0]
        .into_iter()
        .find(|v| (c.abs() - v).abs() < 1e-7)
        .map_or(c, |v| v.copysign(c) + 0.0)
}

/// Records that `doc`'s scene is under a root node made by [`node`] for `frame`, and what the frame converts.
pub(crate) fn placed(doc: &mut Document, frame: Frame) {
    doc.framed = true;
    let json = doc.base_mut();
    let converted = [(frame.scales(), "units"), (frame.turns(), "upAxis")];
    if converted.iter().any(|c| c.0)
        && let Some(Value::Object(source)) = json.get_mut("asset").and_then(|a| a.pointer_mut("/extras/source"))
    {
        let had: Vec<String> = match source.get("converted") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
            _ => Vec::new(),
        };
        let now: Vec<&str> = ["units", "upAxis", "handedness"]
            .into_iter()
            .filter(|k| had.iter().any(|h| h == k) || converted.iter().any(|c| c.0 && c.1 == *k))
            .collect();
        source.insert("converted".into(), json!(now));
    }
}

/// Marks `doc` as one that has no [`ROOT`] by design: sampled data converted in place, or glTF kept as it is.
#[cfg_attr(
    not(feature = "default"),
    allow(dead_code, reason = "only readers of sampled data and of glTF use it")
)]
pub(crate) fn flat(doc: &mut Document) {
    doc.framed = true;
}

/// The root node over each scene's roots. Scenes that share their roots share the node; a scene whose roots are
/// partly another's (which no reader writes) is left as it is, since a node has one parent.
fn wrap(json: &mut Map<String, Value>, frame: Frame) {
    let scenes: Vec<Vec<u64>> = match json.get("scenes") {
        Some(Value::Array(s)) => s
            .iter()
            .map(|s| {
                s.get("nodes")
                    .and_then(Value::as_array)
                    .map(|n| n.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default()
            })
            .collect(),
        _ => return,
    };
    let Some(Value::Array(nodes)) = json.get_mut("nodes") else {
        return;
    };
    let mut made: Vec<(Vec<u64>, usize)> = Vec::new();
    let mut claimed: Vec<u64> = Vec::new();
    let mut tops: Vec<Option<usize>> = Vec::new();
    for roots in scenes {
        if roots.is_empty() {
            tops.push(None);
            continue;
        }
        if let Some((_, i)) = made.iter().find(|(r, _)| *r == roots) {
            tops.push(Some(*i));
            continue;
        }
        if roots.iter().any(|r| claimed.contains(r)) {
            tops.push(None);
            continue;
        }
        let mut top = node(frame);
        top["children"] = json!(roots);
        nodes.push(top);
        claimed.extend(&roots);
        made.push((roots, nodes.len() - 1));
        tops.push(Some(nodes.len() - 1));
    }
    if let Some(Value::Array(scenes)) = json.get_mut("scenes") {
        for (scene, top) in scenes.iter_mut().zip(tops) {
            if let (Some(top), Some(scene)) = (top, scene.as_object_mut()) {
                scene.insert("nodes".into(), json!([top]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(v: Value) -> Document {
        let Value::Object(json) = v else { unreachable!() };
        Document::new(json, Vec::new(), Map::new(), crate::images::Refs::new())
    }

    #[test]
    fn every_root_goes_under_one_node_that_carries_the_frame() {
        let mut d = doc(json!({
            "asset": { "extras": { "source": { "format": "X" } } },
            "scenes": [{ "nodes": [0, 2] }],
            "nodes": [{ "name": "a", "children": [1] }, { "name": "b" }, { "name": "c" }],
        }));
        root(&mut d, Frame::z_up().scaled(0.01));
        let j = d.json();
        assert_eq!(j["scenes"], json!([{ "nodes": [3] }]));
        let top = &j["nodes"][3];
        assert_eq!(top["name"], ROOT);
        assert_eq!(top["children"], json!([0, 2]));
        assert_eq!(top["scale"], json!([0.01, 0.01, 0.01]));
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert_eq!(top["rotation"], json!([-h, 0.0, 0.0, h]));
        assert_eq!(j["asset"]["extras"]["source"]["converted"], json!(["units", "upAxis"]));
    }

    #[test]
    fn a_file_that_needs_nothing_still_has_the_node_and_converts_nothing() {
        let mut d = doc(json!({
            "asset": { "extras": { "source": { "format": "X", "converted": ["handedness"] } } },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "name": "a" }],
        }));
        root(&mut d, Frame::IDENTITY);
        let j = d.json();
        assert_eq!(j["nodes"][1], json!({ "name": ROOT, "children": [0] }));
        // What the reader mirrored stays said; the frame adds nothing.
        assert_eq!(j["asset"]["extras"]["source"]["converted"], json!(["handedness"]));
        // Only a turn: only the up axis is converted, and it is listed in the usual order.
        root(&mut d, Frame::z_up());
        assert_eq!(
            d.json()["asset"]["extras"]["source"]["converted"],
            json!(["upAxis", "handedness"])
        );
    }

    #[test]
    fn scenes_share_a_root_only_when_they_share_their_roots() {
        let mut d = doc(json!({
            "asset": {},
            "scenes": [{ "nodes": [0] }, { "nodes": [0] }, { "nodes": [1] }, {}, { "nodes": [1, 0] }],
            "nodes": [{ "name": "a" }, { "name": "b" }],
        }));
        root(&mut d, Frame::IDENTITY);
        let j = d.json();
        assert_eq!(
            j["scenes"],
            json!([{ "nodes": [2] }, { "nodes": [2] }, { "nodes": [3] }, {}, { "nodes": [1, 0] }])
        );
        assert_eq!(j["nodes"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn a_basis_is_a_turn_and_a_scale_and_a_mirror_when_it_is_left_handed() {
        // Z up, right-handed, centimeters.
        let z_up = [
            [0.01, 0., 0., 0.],
            [0., 0., 0.01, 0.],
            [0., -0.01, 0., 0.],
            [0., 0., 0., 1.],
        ];
        let (f, mirror) = Frame::of(&z_up);
        assert_eq!(mirror, None);
        assert!((f.scale - 0.01).abs() < 1e-15 && f.turns());
        let v = crate::quat::rotate(f.rotation, [0.0, 0.0, 1.0]);
        assert!((v[1] - 1.0).abs() < 1e-12, "{v:?}");
        // Y up with Z the other way: mirrored through Z, and not turned at all.
        let left = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]];
        let (f, mirror) = Frame::of(&left);
        assert_eq!(mirror, Some(2));
        assert!(!f.turns() && !f.scales(), "{f:?}");
        // Z up and left-handed: a mirror, then the quarter turn.
        let both = [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]];
        let (f, mirror) = Frame::of(&both);
        let axis = mirror.unwrap();
        for p in [[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]] {
            let mut q = p;
            q[axis] = -q[axis];
            let got = crate::quat::rotate(f.rotation, q);
            let want = [p[0], p[2], p[1]];
            assert!((0..3).all(|k| (got[k] - want[k]).abs() < 1e-12), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn each_up_axis_turns_to_y() {
        use crate::source::Axis;
        for axis in [Axis::PosX, Axis::NegX, Axis::PosY, Axis::NegY, Axis::PosZ, Axis::NegZ] {
            let f = Frame::up(axis);
            let v = crate::quat::rotate(f.rotation, axis.vector().map(f64::from));
            assert!(
                v[0].abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6 && v[2].abs() < 1e-6,
                "{axis}: {v:?}"
            );
            assert_eq!(f.turns(), axis != Axis::PosY);
        }
    }
}
