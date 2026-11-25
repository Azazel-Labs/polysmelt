//! Animations as animation stacks, each with a layer of curve nodes and curves.
//!
//! Translation, scale and morph weight keys keep glTF's own keys and interpolation: a step is a constant key, a
//! linear one a linear key, and a cubic spline a cubic key with the spline's tangents as its slopes. Rotation is
//! keyed as XYZ Euler angles, which don't run between keys as a quaternion does: keys are added between glTF's
//! until the Euler curve stays within [`TOLERANCE_DEGREES`] of the quaternion path, and each key takes whichever
//! angles for its rotation lie nearest the key before, so the curve never turns the long way.

use std::collections::{BTreeMap, HashSet};

use super::records::{Node, p_f64, p_time, props70};
use super::{KTIME, Model, Writer, euler, fbx_name, list};
use crate::Result;
use crate::quat;
use crate::rig::{Channel, Interpolation, Property};

/// How far, in degrees, a rotation curve may stray from the quaternion path it keys.
const TOLERANCE_DEGREES: f64 = 0.05;

/// The shortest gap, in seconds, rotation keys are added at.
const MIN_STEP: f64 = 1e-4;

/// The standard FBX time modes and their rates.
const MODES: [(i32, f64); 15] = [
    (6, 30.0),
    (11, 24.0),
    (10, 25.0),
    (3, 60.0),
    (4, 50.0),
    (5, 48.0),
    (1, 120.0),
    (2, 100.0),
    (12, 1000.0),
    (15, 96.0),
    (16, 72.0),
    (9, 30.0 / 1.001),
    (13, 24.0 / 1.001),
    (17, 60.0 / 1.001),
    (18, 120.0 / 1.001),
];

/// The FBX time mode for `fps` frames a second: a standard one within 0.1% of it, else custom (14) at `fps`.
pub(super) fn time_mode(fps: f64) -> (i32, f64) {
    MODES
        .iter()
        .find(|(_, r)| (fps / r - 1.0).abs() <= 1e-3)
        .map_or((14, fps), |&(m, r)| (m, r))
}

/// Seconds as FBX time.
pub(super) fn ktime(t: f64) -> i64 {
    #[allow(clippy::cast_possible_truncation, reason = "rounded, and `as` saturates")]
    let k = (t * KTIME).round() as i64;
    k
}

/// `KeyAttrFlags` of a constant key, a linear one, and a cubic one with slopes of its own either side.
const CONSTANT: i32 = 0x2;
const LINEAR: i32 = 0x4;
const CUBIC: i32 = 0x8 | 0x400 | 0x800;

/// A curve's keys: times in seconds, values, and per key its flags and slopes (out, and the next key's in, per
/// second).
#[derive(Default)]
struct Keys {
    times: Vec<f64>,
    values: Vec<f64>,
    flags: Vec<i32>,
    slopes: Vec<[f64; 2]>,
}

impl Keys {
    fn push(&mut self, t: f64, v: f64, flags: i32, slopes: [f64; 2]) {
        self.times.push(t);
        self.values.push(v);
        self.flags.push(flags);
        self.slopes.push(slopes);
    }
}

/// The flags of a glTF interpolation.
fn flags_of(i: Interpolation) -> i32 {
    match i {
        Interpolation::Step => CONSTANT,
        Interpolation::CubicSpline => CUBIC,
        _ => LINEAR,
    }
}

/// A channel's keys as `width` curves, each key's value (and a cubic key's tangents) through `map`, which takes and
/// gives a whole key element.
fn curves(c: &Channel, width: usize, map: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<Keys> {
    let n = c.times.len();
    let cubic = c.interpolation == Interpolation::CubicSpline;
    let stride = if cubic { width * 3 } else { width };
    if n == 0 || c.values.len() < n * stride {
        return Vec::new();
    }
    let flags = flags_of(c.interpolation);
    let at = |k: usize, part: usize| -> [f64; 3] {
        let base = k * stride + part * width;
        let v: [f64; 3] = std::array::from_fn(|j| if j < width { f64::from(c.values[base + j]) } else { 0.0 });
        if width == 3 { map(v) } else { v }
    };
    let mut out: Vec<Keys> = (0..width).map(|_| Keys::default()).collect();
    for (k, &t) in c.times.iter().enumerate() {
        let t = f64::from(t);
        let (value, out_slope, next_in) = if cubic {
            let next = if k + 1 < n { at(k + 1, 0) } else { [0.0; 3] };
            (at(k, 1), at(k, 2), next)
        } else {
            (at(k, 0), [0.0; 3], [0.0; 3])
        };
        for (j, keys) in out.iter_mut().enumerate() {
            keys.push(t, value[j], flags, [out_slope[j], next_in[j]]);
        }
    }
    out
}

/// The weights channel's keys as one curve per morph target, in percent.
fn weight_curves(c: &Channel, targets: usize) -> Vec<Keys> {
    let n = c.times.len();
    let cubic = c.interpolation == Interpolation::CubicSpline;
    if n == 0 || targets == 0 {
        return Vec::new();
    }
    let stride = if cubic { targets * 3 } else { targets };
    if c.values.len() < n * stride {
        return Vec::new();
    }
    let flags = flags_of(c.interpolation);
    let at = |k: usize, part: usize, j: usize| f64::from(c.values[k * stride + part * targets + j]) * 100.0;
    (0..targets)
        .map(|j| {
            let mut keys = Keys::default();
            for (k, &t) in c.times.iter().enumerate() {
                let t = f64::from(t);
                if cubic {
                    let next_in = if k + 1 < n { at(k + 1, 0, j) } else { 0.0 };
                    keys.push(t, at(k, 1, j), flags, [at(k, 2, j), next_in]);
                } else {
                    keys.push(t, at(k, 0, j), flags, [0.0; 2]);
                }
            }
            keys
        })
        .collect()
}

/// Euler angles in degrees for the same rotation as `e` lying nearest `prev`: `e` or its other set of three (X and
/// Z half a turn on, Y mirrored), each angle a whole number of turns from where it was.
fn nearest(e: [f64; 3], prev: [f64; 3]) -> [f64; 3] {
    let near = |v: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| v[k] + 360.0 * ((prev[k] - v[k]) / 360.0).round()) };
    let a = near(e);
    let b = near([e[0] + 180.0, 180.0 - e[1], e[2] + 180.0]);
    let apart = |v: &[f64; 3]| (0..3).map(|k| (v[k] - prev[k]).abs()).sum::<f64>();
    if apart(&b) < apart(&a) { b } else { a }
}

/// The rotation XYZ Euler angles in degrees make.
fn of_euler(e: [f64; 3]) -> [f64; 4] {
    quat::euler_xyz(e.map(f64::to_radians))
}

/// Rotation channel `c` as Euler curves keyed densely enough to follow it, each rotation through `map` first.
fn rotation_curves(c: &Channel, map: impl Fn([f64; 4]) -> [f64; 4]) -> Vec<Keys> {
    let n = c.times.len();
    let mut buf = Vec::new();
    let mut at = |t: f64| -> Option<[f64; 4]> {
        #[allow(clippy::cast_possible_truncation, reason = "a key time, which is f32 already")]
        let ok = c.sample_into(t as f32, &mut buf);
        (ok && buf.len() == 4).then(|| {
            let q = [buf[0], buf[1], buf[2], buf[3]].map(f64::from);
            map(quat::unit_above(q, 1e-12).unwrap_or(quat::IDENTITY))
        })
    };
    let mut keys: Vec<(f64, [f64; 3])> = Vec::with_capacity(n);
    let Some(first) = c.times.first().map(|&t| f64::from(t)) else {
        return Vec::new();
    };
    let Some(q0) = at(first) else {
        return Vec::new();
    };
    keys.push((first, euler(q0)));
    let flags = if c.interpolation == Interpolation::Step {
        CONSTANT
    } else {
        LINEAR
    };
    let tolerance = TOLERANCE_DEGREES.to_radians();
    for w in c.times.windows(2) {
        let end = f64::from(w[1]);
        let mut pending = vec![end];
        while let Some(&b) = pending.last() {
            let &(a, ea) = keys.last().expect("the first key is in");
            if b <= a {
                pending.pop();
                continue;
            }
            let Some(qb) = at(b) else { return Vec::new() };
            let eb = nearest(euler(qb), ea);
            // A step holds its rotation: only the keys themselves are written.
            let strays = flags == LINEAR
                && b - a > MIN_STEP
                && [0.25, 0.5, 0.75].into_iter().any(|f| {
                    let t = a + (b - a) * f;
                    let e: [f64; 3] = std::array::from_fn(|k| ea[k] + (eb[k] - ea[k]) * f);
                    at(t).is_some_and(|q| quat::angle(of_euler(e), q) > tolerance)
                });
            if strays {
                pending.push(a + (b - a) / 2.0);
            } else {
                keys.push((b, eb));
                pending.pop();
            }
        }
    }
    let mut out: Vec<Keys> = (0..3).map(|_| Keys::default()).collect();
    for (t, e) in keys {
        for (k, curve) in out.iter_mut().enumerate() {
            curve.push(t, e[k], flags, [0.0; 2]);
        }
    }
    out
}

/// An `AnimationCurve` for `keys`, with `default` as its value without keys.
fn curve_node(id: i64, keys: &Keys, default: f64) -> Node {
    let times: Vec<i64> = keys.times.iter().map(|&t| ktime(t)).collect();
    #[allow(clippy::cast_possible_truncation, reason = "FBX keeps key values as floats")]
    let values: Vec<f32> = keys.values.iter().map(|&v| v as f32).collect();
    // Runs of keys sharing flags and data: a third of the way along for every handle, as plain Bezier handles are.
    let third = f32::from_bits(0x0D05_0D05);
    let mut flags: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut counts: Vec<i32> = Vec::new();
    for (k, &f) in keys.flags.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, reason = "FBX keeps slopes as floats")]
        let d = if f == CUBIC {
            [keys.slopes[k][0] as f32, keys.slopes[k][1] as f32, third, 0.0]
        } else {
            [0.0, 0.0, third, 0.0]
        };
        let same = flags.last() == Some(&f) && data.len() >= 4 && data[data.len() - 4..] == d;
        if same {
            *counts.last_mut().expect("beside its flags") += 1;
        } else {
            flags.push(f);
            data.extend_from_slice(&d);
            counts.push(1);
        }
    }
    Node::new("AnimationCurve")
        .i64(id)
        .str(&fbx_name("", "AnimCurve"))
        .str("")
        .child("Default", |n| n.f64(default))
        .child("KeyVer", |n| n.i32(4008))
        .child("KeyTime", |n| n.arr(&times))
        .child("KeyValueFloat", |n| n.arr(&values))
        .child("KeyAttrFlags", |n| n.arr(&flags))
        .child("KeyAttrDataFloat", |n| n.arr(&data))
        .child("KeyAttrRefCount", |n| n.arr(&counts))
}

/// A model's value at rest of a transform property, as its `Lcl` property holds it.
fn rest(m: &Model, p: Property) -> [f64; 3] {
    let (t, r, s) = m.fbx_trs(m.t, m.r, m.s);
    match p {
        Property::Translation => t,
        Property::Rotation => euler(r),
        _ => s,
    }
}

impl Writer<'_> {
    /// Writes the document's animations, a stack each.
    pub(super) fn animations(&mut self) -> Result<()> {
        if list(self.json, "animations").is_empty() {
            return Ok(());
        }
        let rig = self.doc.rig()?;
        let fps = rig
            .animations
            .iter()
            .find_map(|a| a.frame_rate)
            .filter(|r| r.is_finite() && *r > 0.0)
            .unwrap_or(30.0);
        self.fps = time_mode(fps).1;
        let mut names = HashSet::new();
        let mut span: Option<(i64, i64)> = None;
        for (k, a) in rig.animations.iter().enumerate() {
            let mut name = a
                .name
                .clone()
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("animation_{k}"));
            if !names.insert(name.clone()) {
                name = format!("{name}_{k}");
                names.insert(name.clone());
            }
            let (start, stop) = a.span();
            let (start, stop) = (ktime(f64::from(start)), ktime(f64::from(stop)));
            span = Some(span.map_or((start, stop), |(a, b)| (a.min(start), b.max(stop))));
            let stack = self.id();
            let s = Node::new("AnimationStack")
                .i64(stack)
                .str(&fbx_name(&name, "AnimStack"))
                .str("")
                .with(props70(vec![
                    p_time("LocalStart", start),
                    p_time("LocalStop", stop),
                    p_time("ReferenceStart", start),
                    p_time("ReferenceStop", stop),
                ]))
                .closed();
            self.object("AnimationStack", s);
            let layer = self.id();
            let l = Node::new("AnimationLayer")
                .i64(layer)
                .str(&fbx_name("BaseLayer", "AnimLayer"))
                .str("")
                .closed();
            self.object("AnimationLayer", l);
            self.oo(layer, stack);
            // Channels by node, in node order, each property once (the first channel for it, as glTF allows one).
            let mut by_node: BTreeMap<usize, Vec<&Channel>> = BTreeMap::new();
            for c in &a.channels {
                let list = by_node.entry(c.node).or_default();
                if !list.iter().any(|o| o.property == c.property) {
                    list.push(c);
                }
            }
            for (node, channels) in by_node {
                for c in channels {
                    self.channel(node, c, layer);
                }
            }
            self.takes.push((name, start, stop));
        }
        self.span = span;
        Ok(())
    }

    /// Writes channel `c` of node `node` as a curve node on `layer`.
    fn channel(&mut self, node: usize, c: &Channel, layer: i64) {
        // Weights go to the node's blend shape channels, which a folded node has without a model of its own.
        if c.property == Property::Weights {
            let targets = self.channels.get(&node).cloned().unwrap_or_default();
            for (j, keys) in weight_curves(c, targets.len()).into_iter().enumerate() {
                let ch = targets[j];
                let default = keys.values.first().copied().unwrap_or(0.0);
                let cn = self.id();
                let n = Node::new("AnimationCurveNode")
                    .i64(cn)
                    .str(&fbx_name("DeformPercent", "AnimCurveNode"))
                    .str("")
                    .with(props70(vec![p_f64("d|DeformPercent", "Number", "A", default)]));
                self.object("AnimationCurveNode", n);
                self.oo(cn, layer);
                self.op(cn, ch, "DeformPercent");
                let id = self.id();
                self.object("AnimationCurve", curve_node(id, &keys, default));
                self.op(id, cn, "d|DeformPercent");
            }
            return;
        }
        let Some(m) = self.models.get(node).and_then(Option::as_ref) else {
            return;
        };
        let (prop, short) = match c.property {
            Property::Translation => ("Lcl Translation", "T"),
            Property::Rotation => ("Lcl Rotation", "R"),
            Property::Scale => ("Lcl Scaling", "S"),
            _ => return,
        };
        let rest = rest(m, c.property);
        let keys = match c.property {
            Property::Translation => {
                let back = m.back;
                curves(c, 3, |v| back.map_or(v, |b| quat::rotate(b, v)))
            }
            Property::Scale => {
                let perm = m.turn.map(|t| t.perm);
                curves(c, 3, |v| perm.map_or(v, |p| p.map(|i| v[i])))
            }
            _ => {
                let (back, turn) = (m.back, m.turn.map(|t| t.q));
                rotation_curves(c, |q| {
                    let q = back.map_or(q, |b| quat::mul(b, q));
                    turn.map_or(q, |t| quat::mul(q, t))
                })
            }
        };
        if keys.len() != 3 {
            return;
        }
        let model = m.id;
        let cn = self.id();
        let n = Node::new("AnimationCurveNode")
            .i64(cn)
            .str(&fbx_name(short, "AnimCurveNode"))
            .str("")
            .with(props70(vec![
                p_f64("d|X", "Number", "A", rest[0]),
                p_f64("d|Y", "Number", "A", rest[1]),
                p_f64("d|Z", "Number", "A", rest[2]),
            ]));
        self.object("AnimationCurveNode", n);
        self.oo(cn, layer);
        self.op(cn, model, prop);
        for (k, axis) in keys.iter().zip(["d|X", "d|Y", "d|Z"]) {
            let id = self.id();
            let default = k.values.first().copied().unwrap_or(0.0);
            self.object("AnimationCurve", curve_node(id, k, default));
            self.op(id, cn, axis);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_rates_have_their_modes() {
        assert_eq!(time_mode(30.0), (6, 30.0));
        assert_eq!(time_mode(24.0).0, 11);
        assert_eq!(time_mode(29.97).0, 9);
        assert_eq!(time_mode(15.0), (14, 15.0));
    }

    #[test]
    fn nearest_angles_keep_the_rotation() {
        // The same rotation as (350, 100, 150): nearer, once wrapped, than any turn of (170, 80, -30).
        let e = [170.0, 80.0, -30.0];
        let n = nearest(e, [-15.0, 95.0, 140.0]);
        assert!(quat::angle(of_euler(e), of_euler(n)) < 1e-9);
        assert!((0..3).all(|k| (n[k] - [-10.0, 100.0, 150.0][k]).abs() < 1e-9), "{n:?}");
        // Wrapped by whole turns.
        let n = nearest([10.0, 20.0, 30.0], [700.0, 20.0, -330.0]);
        assert!((0..3).all(|k| (n[k] - [730.0, 20.0, -330.0][k]).abs() < 1e-9), "{n:?}");
    }

    #[test]
    fn rotation_curves_follow_the_quaternions() {
        #[allow(clippy::cast_possible_truncation, reason = "test values fit")]
        let q = |deg: f64| quat::about_axis([0.6, 0.0, 0.8], deg.to_radians()).map(|x| x as f32);
        let c = Channel {
            node: 0,
            property: Property::Rotation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0, 2.0],
            values: [q(0.0), q(170.0), q(340.0)].concat(),
        };
        let keys = rotation_curves(&c, |q| q);
        assert!(keys[0].times.len() > 3);
        let mut buf = Vec::new();
        for i in 0..=200 {
            let t = f64::from(i) / 100.0;
            let k = keys[0]
                .times
                .partition_point(|&x| x <= t)
                .clamp(1, keys[0].times.len() - 1);
            let (t0, t1) = (keys[0].times[k - 1], keys[0].times[k]);
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let e: [f64; 3] =
                std::array::from_fn(|a| keys[a].values[k - 1] + (keys[a].values[k] - keys[a].values[k - 1]) * f);
            #[allow(clippy::cast_possible_truncation, reason = "a time within the clip")]
            let at = t as f32;
            assert!(c.sample_into(at, &mut buf));
            let want = [buf[0], buf[1], buf[2], buf[3]].map(f64::from);
            assert!(quat::angle(of_euler(e), want).to_degrees() < 0.06, "at {t}");
        }
        // Continuous: no key jumps more than a quarter turn.
        for a in &keys {
            assert!(a.values.windows(2).all(|w| (w[1] - w[0]).abs() < 90.0));
        }
    }
}
