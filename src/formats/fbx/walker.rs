//! The scene walker: each model's node and local transform (pivots, offsets, pre- and post-rotation, rotation
//! order), what it holds, and its cameras and lights.

use serde_json::{Map, Value, json};

use super::{Props, Reader, choice, flag, kind_of, name_of, scalar, string, vec3};
use crate::Result;
use crate::formats::fbx_tree::{Attr as A, NodeHandle};
use crate::formats::xform::{IDENTITY, M4, mul, scale, set_transform, translate};
use crate::mesh::srgb_to_linear;
use crate::num::f32_of;

/// Euler rotation in degrees, FBX rotation order (0 = XYZ: X applied first).
pub(super) fn rotate(deg: [f64; 3], order: i32) -> M4 {
    let [x, y, z] = deg.map(f64::to_radians);
    let rx = [
        [1., 0., 0., 0.],
        [0., x.cos(), -x.sin(), 0.],
        [0., x.sin(), x.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let ry = [
        [y.cos(), 0., y.sin(), 0.],
        [0., 1., 0., 0.],
        [-y.sin(), 0., y.cos(), 0.],
        [0., 0., 0., 1.],
    ];
    let rz = [
        [z.cos(), -z.sin(), 0., 0.],
        [z.sin(), z.cos(), 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    match order {
        1 => mul(&ry, &mul(&rz, &rx)), // XZY
        2 => mul(&rx, &mul(&rz, &ry)), // YZX
        3 => mul(&rz, &mul(&rx, &ry)), // YXZ
        4 => mul(&ry, &mul(&rx, &rz)), // ZXY
        5 => mul(&rx, &mul(&ry, &rz)), // ZYX
        _ => mul(&rz, &mul(&ry, &rx)), // XYZ
    }
}

/// Whether a model's own properties hide it: `Visibility` under one half, or `Show` off.
fn hidden(p: &Props<'_>) -> bool {
    scalar(p, "Visibility").is_some_and(|v| v < 0.5) || !flag(p, "Show", true)
}

/// A model's rotation order. Without `RotationActive` the order, and the pre- and post-rotations, are ignored: the
/// rotation is XYZ. A file that doesn't say is taken to mean them.
fn rotation_order(p: &Props<'_>) -> i32 {
    if flag(p, "RotationActive", true) {
        choice(p, "RotationOrder", 0)
    } else {
        0
    }
}

/// A model's local transform: `T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1`, the FBX SDK's
/// order, with its fixed parts (offsets, pivots, pre- and post-rotation) multiplied out, for evaluating it again at
/// each animation time: `T · before · R · between · S · after`.
pub(super) struct Local {
    pub(super) t: [f64; 3],
    pub(super) r: [f64; 3],
    pub(super) s: [f64; 3],
    pub(super) order: i32,
    before: M4,
    between: M4,
    after: M4,
    /// Whether the parent's own scale is taken off before this model's rotation and scale (`InheritType` 2, a
    /// joint's segment scale compensation): it still moves the model, but doesn't stretch it.
    pub(super) ignore_parent_scale: bool,
}

impl Local {
    pub(super) fn new(p: &Props<'_>) -> Self {
        let order = rotation_order(p);
        let active = flag(p, "RotationActive", true);
        let turn = |name: &str| if active { vec3(p, name, [0.0; 3]) } else { [0.0; 3] };
        let r_pre = rotate(turn("PreRotation"), 0);
        let r_post_inv = transpose3(&rotate(turn("PostRotation"), 0));
        let rp = vec3(p, "RotationPivot", [0.0; 3]);
        let sp = vec3(p, "ScalingPivot", [0.0; 3]);
        let product = |ms: &[M4]| ms.iter().fold(IDENTITY, |acc, m| mul(&acc, m));
        Self {
            t: vec3(p, "Lcl Translation", [0.0; 3]),
            r: vec3(p, "Lcl Rotation", [0.0; 3]),
            s: vec3(p, "Lcl Scaling", [1.0; 3]),
            order,
            before: product(&[translate(vec3(p, "RotationOffset", [0.0; 3])), translate(rp), r_pre]),
            between: product(&[
                r_post_inv,
                translate(rp.map(|c| -c)),
                translate(vec3(p, "ScalingOffset", [0.0; 3])),
                translate(sp),
            ]),
            after: translate(sp.map(|c| -c)),
            ignore_parent_scale: choice(p, "InheritType", 0) == 2,
        }
    }

    /// The transform with translation, rotation and scale as given, or as the model holds them, under a parent
    /// scaled `parent` (its own scale, not its parents').
    pub(super) fn at(&self, t: Option<[f64; 3]>, r: Option<[f64; 3]>, s: Option<[f64; 3]>, parent: [f64; 3]) -> M4 {
        let mut m = translate(t.unwrap_or(self.t));
        if self.ignore_parent_scale && parent.iter().all(|c| c.abs() > 1e-12 && (1.0 / c).is_finite()) {
            m = mul(&m, &scale(parent.map(|c| 1.0 / c)));
        }
        let m = mul(&m, &self.before);
        let m = mul(&m, &rotate(r.unwrap_or(self.r), self.order));
        let m = mul(&m, &self.between);
        let m = mul(&m, &scale(s.unwrap_or(self.s)));
        mul(&m, &self.after)
    }
}

/// The transpose of a rotation (its inverse).
fn transpose3(r: &M4) -> M4 {
    let mut t = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = r[j][i];
        }
    }
    t
}

/// A model's geometric transform: applied to what it holds, not passed to its children.
fn geometric_of(p: &Props<'_>) -> M4 {
    mul(
        &translate(vec3(p, "GeometricTranslation", [0.0; 3])),
        &mul(
            &rotate(vec3(p, "GeometricRotation", [0.0; 3]), 0),
            &scale(vec3(p, "GeometricScaling", [1.0; 3])),
        ),
    )
}
/// A property's values as JSON: a number, a string, or a list of them.
fn prop_json(vals: &[A]) -> Option<Value> {
    let one = |v: &A| match v {
        A::String(s) => Some(json!(s.split('\0').next().unwrap_or(""))),
        A::Bool(b) => Some(json!(b)),
        A::I16(i) => Some(json!(i)),
        A::I32(i) => Some(json!(i)),
        A::I64(i) => Some(json!(i)),
        A::F32(f) if f.is_finite() => Some(json!(f)),
        A::F64(f) if f.is_finite() => Some(json!(f)),
        _ => None,
    };
    match vals {
        [] => None,
        [v] => one(v),
        vs => vs.iter().map(one).collect::<Option<Vec<_>>>().map(Value::Array),
    }
}

/// A model's user-defined properties (flagged `U`), for its node's extras.
fn user_props(node: NodeHandle<'_>) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(p70) = node.first_child_by_name("Properties70") {
        for p in p70.children_by_name("P") {
            let a = p.attributes();
            if let (Some(name), Some(flags)) = (a.first().and_then(string), a.get(3).and_then(string))
                && flags.contains('U')
                && let Some(v) = prop_json(a.get(4..).unwrap_or_default())
            {
                out.insert(name.to_owned(), v);
            }
        }
    }
    out
}

/// Sets a node's transform, unless it isn't finite (a broken file): the node stays where its parent is.
fn place(node: &mut Value, m: &M4) {
    if m.iter().flatten().all(|c| c.is_finite()) {
        set_transform(node, m);
    }
}

/// Marks a node hidden.
fn hide(node: &mut Value) {
    node["extensions"]["KHR_node_visibility"] = json!({ "visible": false });
}

impl Reader<'_, '_, '_> {
    /// A model's node: its name, local transform, visibility and user properties.
    pub(super) fn model_node(&mut self, m: i64) -> Value {
        let Some(node) = self.objs.node(m) else {
            return json!({});
        };
        let mut out = json!({});
        if let Some(name) = name_of(node) {
            out["name"] = json!(name);
        }
        place(&mut out, &self.rest_local(m));
        if hidden(&self.objs.props(m)) {
            self.hidden.insert(m);
        }
        if self.hidden.contains(&m) {
            hide(&mut out);
        }
        let extras = user_props(node);
        if !extras.is_empty() {
            out["extras"] = Value::Object(extras);
        }
        out
    }

    /// Model `m`'s local transform at rest.
    fn rest_local(&self, m: i64) -> M4 {
        let local = Local::new(&self.objs.props(m));
        local.at(None, None, None, self.parent_scale(&local, m))
    }

    /// The scale of model `m`'s parent, when `m` takes it off (one otherwise).
    pub(super) fn parent_scale(&self, local: &Local, m: i64) -> [f64; 3] {
        match self.objs.model_parent(m) {
            Some(p) if local.ignore_parent_scale && p != m => vec3(&self.objs.props(p), "Lcl Scaling", [1.0; 3]),
            _ => [1.0; 3],
        }
    }

    /// Whether model `m` or one of its parents is hidden.
    fn hidden_under(&self, m: i64) -> bool {
        let mut at = Some(m);
        for _ in 0..=self.node_of.len() {
            let Some(id) = at else { break };
            if self.hidden.contains(&id) {
                return true;
            }
            at = self.objs.model_parent(id).filter(|&p| p != id);
        }
        false
    }

    pub(super) fn add_child(&mut self, parent: usize, child: usize) {
        let kids = self.nodes[parent]
            .as_object_mut()
            .expect("a node is an object")
            .entry("children")
            .or_insert_with(|| json!([]));
        kids.as_array_mut().expect("children is a list").push(json!(child));
    }

    /// What model `m` holds: its geometry (with skin and morph targets), camera and light.
    pub(super) fn attach(&mut self, m: i64) -> Result<()> {
        let node = self.node_of[&m];
        let geometric = geometric_of(&self.objs.props(m));
        let geometries: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|&g| {
                self.objs
                    .node(g)
                    .is_some_and(|n| n.name() == "Geometry" && matches!(kind_of(n), "Mesh" | "Line" | "NurbsCurve"))
            })
            .collect();
        for g in geometries {
            let built = if self.objs.is_kind("Geometry", g, "Mesh") {
                self.mesh_for(g, m)?
            } else {
                match self.line_mesh(g, m) {
                    Some(built) => built,
                    None => continue,
                }
            };
            let skin = self
                .objs
                .children_of(g)
                .find(|d| self.objs.is_kind("Deformer", *d, "Skin"));
            let skinned = self.skinned_meshes.contains(&built);
            let target = if skinned && let Some(skin) = skin {
                // A skinned mesh's node transform is ignored, its joints place it (the geometric transform
                // goes into the inverse bind matrices), so it gets a root node of its own, as glTF asks; the
                // model's node stays where it is, a joint for vertices no bone weights. Out of its model's
                // tree, it's hidden when that is.
                let s = self.skin(skin, m, &geometric);
                let mut host = json!({ "skin": s });
                if let Some(name) = self.nodes[node].get("name") {
                    host["name"] = name.clone();
                }
                if self.hidden_under(m) {
                    hide(&mut host);
                }
                self.nodes.push(host);
                self.skinned_roots.push(self.nodes.len() - 1);
                self.nodes.len() - 1
            } else if geometric == IDENTITY && self.nodes[node].get("mesh").is_none() {
                node
            } else {
                self.child(node, Some(&geometric))
            };
            self.nodes[target]["mesh"] = json!(built);
            for (ch, first) in self.morphs.get(&built).cloned().unwrap_or_default() {
                self.channel_users.entry(ch).or_default().push((target, first));
            }
        }
        let attributes: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|a| self.objs.is("NodeAttribute", *a))
            .collect();
        for a in attributes {
            let Some(attr) = self.objs.node(a) else { continue };
            match kind_of(attr) {
                "Camera" if self.keep.cameras => {
                    let cam = camera(&self.objs.props(a), self.unit);
                    self.cameras.push(cam);
                    // FBX cameras look down +X, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        0.0,
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["camera"] = json!(self.cameras.len() - 1);
                }
                "Light" if self.keep.lights => {
                    let Some(light) = light(&self.objs.props(a)) else {
                        continue;
                    };
                    self.lights.push(light);
                    // FBX lights shine down -Y, glTF's down -Z.
                    let c = self.child(node, None);
                    self.nodes[c]["rotation"] = json!([
                        -std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2
                    ]);
                    self.nodes[c]["extensions"] = json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A new child of `parent`, with transform `m`; returns it.
    fn child(&mut self, parent: usize, m: Option<&M4>) -> usize {
        let mut n = json!({});
        if let Some(m) = m {
            place(&mut n, m);
        }
        self.nodes.push(n);
        let c = self.nodes.len() - 1;
        self.add_child(parent, c);
        c
    }

    /// Model `m`'s world transform in the file's space: its parents' times its own.
    pub(super) fn world_of(&self, m: i64) -> M4 {
        let mut world = IDENTITY;
        let mut at = Some(m);
        // At most one step per model, so a cycle in a broken file ends.
        for _ in 0..=self.node_of.len() {
            let Some(id) = at else { break };
            world = mul(&self.rest_local(id), &world);
            at = self.objs.model_parent(id).filter(|&p| p != id);
        }
        world
    }
}

/// A camera's glTF form. FBX gives the field of view horizontally unless its aperture mode says otherwise.
fn camera(p: &Props<'_>, unit: f64) -> Value {
    let aspect = match (scalar(p, "AspectWidth"), scalar(p, "AspectHeight")) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => w / h,
        _ => 16.0 / 9.0,
    };
    // Planes are in the file's units, but a node's scale doesn't reach a camera's planes: they're converted here.
    let near = scalar(p, "NearPlane").filter(|n| *n > 0.0).unwrap_or(10.0);
    let far = scalar(p, "FarPlane").filter(|f| *f > near).unwrap_or(near * 10_000.0);
    let (near, far) = (near * unit, far * unit);
    if scalar(p, "CameraProjectionType") == Some(1.0) {
        let zoom = scalar(p, "OrthoZoom").filter(|z| *z > 0.0).unwrap_or(1.0);
        return json!({ "type": "orthographic", "orthographic": { "xmag": zoom * aspect, "ymag": zoom, "znear": near, "zfar": far } });
    }
    #[allow(clippy::cast_possible_truncation, reason = "an enum value; `as` saturates bad ones")]
    let mode = scalar(p, "ApertureMode").map_or(1, |m| m as i32);
    let deg = scalar(p, "FieldOfView")
        .filter(|f| *f > 0.0 && *f < 180.0)
        .unwrap_or(40.0);
    let yfov = match mode {
        2 => deg.to_radians(),
        0 => scalar(p, "FieldOfViewY")
            .filter(|f| *f > 0.0 && *f < 180.0)
            .unwrap_or(deg)
            .to_radians(),
        3 => match (scalar(p, "FocalLength"), scalar(p, "FilmHeight")) {
            (Some(f), Some(h)) if f > 0.0 && h > 0.0 => 2.0 * (h * 25.4 / 2.0 / f).atan(),
            _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
        },
        _ => 2.0 * ((deg.to_radians() / 2.0).tan() / aspect).atan(),
    };
    json!({ "type": "perspective", "perspective": { "yfov": yfov, "aspectRatio": aspect, "znear": near, "zfar": far } })
}

/// A light's `KHR_lights_punctual` form: point, directional or spot. An area light, which shines from one side of
/// its shape, becomes a spot opening to a half sphere, and a volume light a point; their own kind is kept in the
/// light's extras, as is any falloff other than the inverse square.
fn light(p: &Props<'_>) -> Option<Value> {
    let kind = choice(p, "LightType", 0);
    let color = vec3(p, "Color", [1.0; 3]).map(|c| srgb_to_linear(f32_of(c.clamp(0.0, 1.0))));
    let intensity = scalar(p, "Intensity").unwrap_or(100.0).max(0.0) / 100.0;
    let mut out = json!({ "color": color, "intensity": intensity });
    match kind {
        0 => out["type"] = json!("point"),
        1 => out["type"] = json!("directional"),
        2 => {
            let half = std::f64::consts::FRAC_PI_2;
            let outer = (scalar(p, "OuterAngle").unwrap_or(45.0).to_radians() / 2.0).clamp(1e-4, half);
            let inner = (scalar(p, "InnerAngle").unwrap_or(0.0).to_radians() / 2.0).clamp(0.0, outer * 0.999);
            out["type"] = json!("spot");
            out["spot"] = json!({ "innerConeAngle": inner, "outerConeAngle": outer });
        }
        3 => {
            out["type"] = json!("spot");
            out["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": std::f64::consts::FRAC_PI_2 });
            let shape = if choice(p, "AreaLightShape", 0) == 1 {
                "sphere"
            } else {
                "rectangle"
            };
            out["extras"] = json!({ "lightType": "area", "areaLightShape": shape });
        }
        4 => {
            out["type"] = json!("point");
            out["extras"] = json!({ "lightType": "volume" });
        }
        _ => return None,
    }
    // FBX's falloff: none, linear, quadratic (glTF's own) or cubic.
    let decay = choice(p, "DecayType", 2);
    if decay != 2 && kind != 1 {
        let name = match decay {
            0 => "none",
            1 => "linear",
            3 => "cubic",
            _ => "unknown",
        };
        out["extras"]["decayType"] = json!(name);
    }
    Some(out)
}
