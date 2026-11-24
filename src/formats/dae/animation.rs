//! Animations of node transforms, sampled into glTF translation, rotation and scale tracks, and of morph weights.

use std::collections::HashMap;
use std::rc::Rc;

use serde_json::{Value, json};

use super::{Transform, floats64, geometry, xml};
use crate::formats::xform::{IDENTITY, M4, decompose, mul};
use crate::num::f32_of;

/// How deeply animations nest.
const MAX_NESTING: usize = 32;

/// Points added inside each curved segment, so Bézier, Hermite, B-spline and cardinal curves keep their shape.
const CURVE_STEPS: usize = 8;

/// The largest turn between samples, in radians: longer ones are cut, as glTF blends rotations the short way.
const MAX_TURN: f64 = std::f64::consts::PI / 18.0;

/// The most steps one segment is cut into.
const MAX_CUTS: usize = 64;

/// The most samples a node's animation takes.
const MAX_SAMPLES: usize = 1 << 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Interp {
    Step,
    Linear,
    Bezier,
    Hermite,
    /// A uniform cubic B-spline over the keys' values, which it doesn't pass through.
    BSpline,
    /// A cardinal (Catmull-Rom) spline through the keys' values.
    Cardinal,
}

/// A sampler's curve: key times and the values at them, `width` numbers to a key.
struct Curve {
    times: Vec<f64>,
    values: Vec<f64>,
    width: usize,
    interp: Vec<Interp>,
    in_tan: geometry::Acc<f64>,
    out_tan: geometry::Acc<f64>,
}

/// A channel: which transform of which node, which of its numbers (all of them when `None`), and the curve. A
/// channel of morph weights has the `<morph>` for its node, and the target for its number.
struct Track {
    node: usize,
    transform: usize,
    member: Option<usize>,
    curve: Rc<Curve>,
    weights: bool,
}

impl Curve {
    fn value(&self, key: usize, c: usize) -> f64 {
        self.values.get(key * self.width + c).copied().unwrap_or(0.0)
    }

    /// A tangent of component `c` at `key`: its time (when it has one) and value.
    fn tangent(acc: &geometry::Acc<f64>, width: usize, key: usize, c: usize) -> Option<(Option<f64>, f64)> {
        let t = acc.get(u32::try_from(key).ok()?)?;
        if acc.width == 2 * width {
            Some((Some(*t.get(2 * c)?), *t.get(2 * c + 1)?))
        } else {
            Some((None, *t.get(c)?))
        }
    }

    fn curved(&self, key: usize) -> bool {
        matches!(
            self.interp.get(key),
            Some(Interp::Bezier | Interp::Hermite | Interp::BSpline | Interp::Cardinal)
        )
    }

    /// Component `c` at time `t`: held before the first key and after the last.
    fn eval(&self, t: f64, c: usize) -> f64 {
        let n = self.times.len().min(self.values.len() / self.width.max(1));
        if n == 0 {
            return 0.0;
        }
        if t <= self.times[0] || n == 1 {
            return self.value(0, c);
        }
        if t >= self.times[n - 1] {
            return self.value(n - 1, c);
        }
        let k = self.times[..n]
            .partition_point(|&x| x <= t)
            .saturating_sub(1)
            .min(n - 2);
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let (p0, p1) = (self.value(k, c), self.value(k + 1, c));
        let span = t1 - t0;
        let s = if span > 0.0 {
            ((t - t0) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lerp = p0 + (p1 - p0) * s;
        match self.interp.get(k).copied().unwrap_or(Interp::Linear) {
            Interp::Step => p0,
            Interp::Linear => lerp,
            Interp::Bezier => {
                let (Some(out), Some(inn)) = (
                    Self::tangent(&self.out_tan, self.width, k, c),
                    Self::tangent(&self.in_tan, self.width, k + 1, c),
                ) else {
                    return lerp;
                };
                let s = match (out.0, inn.0) {
                    // Two-dimensional tangents: find where the curve's time is `t`.
                    (Some(x0), Some(x1)) => {
                        let (mut lo, mut hi) = (0.0, 1.0);
                        for _ in 0..40 {
                            let mid = f64::midpoint(lo, hi);
                            if bezier(t0, x0, x1, t1, mid) < t {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        f64::midpoint(lo, hi)
                    }
                    _ => s,
                };
                bezier(p0, out.1, inn.1, p1, s)
            }
            Interp::Hermite => {
                let (Some(out), Some(inn)) = (
                    Self::tangent(&self.out_tan, self.width, k, c),
                    Self::tangent(&self.in_tan, self.width, k + 1, c),
                ) else {
                    return lerp;
                };
                hermite(p0, out.1, p1, inn.1, s)
            }
            Interp::BSpline => {
                let before = self.value(k.saturating_sub(1), c);
                let after = self.value((k + 2).min(n - 1), c);
                let (s2, s3) = (s * s, s * s * s);
                ((1.0 - s).powi(3) * before
                    + (3.0 * s3 - 6.0 * s2 + 4.0) * p0
                    + (-3.0 * s3 + 3.0 * s2 + 3.0 * s + 1.0) * p1
                    + s3 * after)
                    / 6.0
            }
            Interp::Cardinal => {
                let before = self.value(k.saturating_sub(1), c);
                let after = self.value((k + 2).min(n - 1), c);
                hermite(p0, (p1 - before) / 2.0, p1, (after - p0) / 2.0, s)
            }
        }
    }
}

/// The cubic Hermite curve from `p0` (tangent `t0`) to `p1` (tangent `t1`) at `s`.
fn hermite(p0: f64, t0: f64, p1: f64, t1: f64, s: f64) -> f64 {
    let (s2, s3) = (s * s, s * s * s);
    (2.0 * s3 - 3.0 * s2 + 1.0) * p0 + (s3 - 2.0 * s2 + s) * t0 + (-2.0 * s3 + 3.0 * s2) * p1 + (s3 - s2) * t1
}

/// The time just before `t` that glTF's single-precision times can still tell from it.
fn just_before(t: f64) -> Option<f64> {
    let f = f32_of(t);
    (f > 0.0 && f.is_finite()).then(|| f64::from(f32::from_bits(f.to_bits() - 1)).min(t))
}

fn bezier(p0: f64, c0: f64, c1: f64, p1: f64, s: f64) -> f64 {
    let u = 1.0 - s;
    u * u * u * p0 + 3.0 * u * u * s * c0 + 3.0 * u * s * s * c1 + s * s * s * p1
}

/// The quaternion of a sampled transform, on the same side as `prev`.
fn continuous(q: [f64; 4], prev: Option<[f64; 4]>) -> [f64; 4] {
    match prev {
        Some(p) if p.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() < 0.0 => q.map(|c| -c),
        _ => q,
    }
}

/// The angle between two unit quaternions.
fn turn(a: [f64; 4], b: [f64; 4]) -> f64 {
    let d = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>().abs().min(1.0);
    2.0 * d.acos()
}

impl super::Reader<'_> {
    /// The document's animations: one for each animation clip, or one of every animation when it has none.
    pub(super) fn animations(&mut self) -> Vec<Value> {
        let dom = self.dom;
        let Some(root) = dom.root() else {
            return Vec::new();
        };
        let mut curves = HashMap::new();
        let mut out = Vec::new();
        let clips: Vec<usize> = dom
            .kids(root, "library_animation_clips")
            .flat_map(|l| dom.kids(l, "animation_clip"))
            .collect();
        if clips.is_empty() {
            let mut tracks = Vec::new();
            for lib in dom.kids(root, "library_animations") {
                for a in dom.kids(lib, "animation") {
                    self.gather(a, 0, &mut curves, &mut tracks);
                }
            }
            if let Some(anim) = self.animation(&tracks, None, None) {
                out.push(anim);
            }
        } else {
            for clip in clips {
                let mut tracks = Vec::new();
                for i in dom.kids(clip, "instance_animation") {
                    if let Some(a) = dom.link(i, "url").filter(|&a| dom.is(a, "animation")) {
                        self.gather(a, 0, &mut curves, &mut tracks);
                    }
                }
                let name = dom.attr(clip, "name").or_else(|| dom.attr(clip, "id"));
                let start = dom.attr_num::<f64>(clip, "start").filter(|t| t.is_finite());
                let end = dom.attr_num::<f64>(clip, "end").filter(|t| t.is_finite());
                let range = start.map(|s| (s, end.filter(|&e| e > s).unwrap_or(f64::INFINITY)));
                if let Some(anim) = self.animation(&tracks, name, range) {
                    out.push(anim);
                }
            }
        }
        out
    }

    /// The channels of animation `el` and the animations in it.
    fn gather(&self, el: usize, depth: usize, curves: &mut HashMap<usize, Option<Rc<Curve>>>, out: &mut Vec<Track>) {
        let dom = self.dom;
        if depth > MAX_NESTING {
            return;
        }
        for c in dom.children(el) {
            match dom.name(c) {
                b"animation" => self.gather(c, depth + 1, curves, out),
                b"channel" => {
                    let Some(sampler) = dom.link(c, "source").filter(|&s| dom.is(s, "sampler")) else {
                        continue;
                    };
                    let Some(target) = dom.attr(c, "target") else {
                        continue;
                    };
                    let (node, transform, member, weights) = if let Some((n, t, m)) = self.target(c, target) {
                        (n, t, m, false)
                    } else if let Some((morph, m)) = self.weight_target(c, target) {
                        (morph, morph, m, true)
                    } else {
                        continue;
                    };
                    let curve = curves
                        .entry(sampler)
                        .or_insert_with(|| self.curve(sampler).map(Rc::new));
                    if let Some(curve) = curve {
                        out.push(Track {
                            node,
                            transform,
                            member,
                            curve: curve.clone(),
                            weights,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    /// A channel target, `node/sid` and a member (`.X`, `.ANGLE`, `(3)`, `(0)(3)`): the node, its transform, the
    /// number's index in it. Ids are looked up in the document of `at`.
    fn target(&self, at: usize, target: &str) -> Option<(usize, usize, Option<usize>)> {
        let dom = self.dom;
        let (id, rest) = target.split_once('/')?;
        let node = dom.by_id_at(at, id).filter(|&n| dom.is(n, "node"))?;
        let rest = rest.rsplit('/').next()?;
        let cut = rest.find(['.', '(']).unwrap_or(rest.len());
        let (sid, member) = rest.split_at(cut);
        let transform = dom
            .children(node)
            .find(|&c| dom.attr(c, "sid") == Some(sid) && Transform::of(dom.name(c)).is_some())?;
        let member = if member.is_empty() {
            None
        } else if let Some(m) = member.strip_prefix('.') {
            Some(match m {
                "X" | "x" | "S" | "R" => 0,
                "Y" | "y" | "T" | "G" => 1,
                "Z" | "z" | "P" | "B" => 2,
                "ANGLE" | "angle" | "W" | "w" | "A" | "Q" => 3,
                _ => return None,
            })
        } else {
            let numbers: Vec<usize> = member
                .split(['(', ')'])
                .filter(|s| !s.is_empty())
                .map(str::parse)
                .collect::<Result<_, _>>()
                .ok()?;
            match numbers[..] {
                [i] => Some(i),
                [i, j] if i < 4 && j < 4 => Some(i * 4 + j),
                _ => return None,
            }
        };
        let kind = Transform::of(dom.name(transform))?;
        if member.is_some_and(|m| m >= kind.len()) {
            return None;
        }
        Some((node, transform, member))
    }

    /// A channel target that is a morph's weights: the weights' `<source>` (`id(2)`), or the morph's controller
    /// (`id/weights(2)`), with the target it picks (all when `None`). Ids are looked up in the document of `at`.
    fn weight_target(&self, at: usize, target: &str) -> Option<(usize, Option<usize>)> {
        let dom = self.dom;
        let cut = target.find(['/', '(', '.']).unwrap_or(target.len());
        let el = dom.by_id_at(at, &target[..cut])?;
        let morph = match dom.name(el) {
            b"source" => dom.parent(el).filter(|&p| dom.is(p, "morph"))?,
            b"controller" => dom.child(el, "morph")?,
            b"morph" => el,
            _ => return None,
        };
        let member = match target.rfind('(') {
            Some(k) => Some(target[k + 1..].trim_end_matches(')').trim().parse().ok()?),
            None => None,
        };
        Some((morph, member))
    }

    /// A sampler's curve; `None` when it has no keys.
    fn curve(&self, sampler: usize) -> Option<Curve> {
        let dom = self.dom;
        let parse = |t: &[u8]| floats64(t);
        let times = geometry::read(dom, self.input(sampler, "INPUT")?, 1, parse);
        let values = geometry::read(dom, self.input(sampler, "OUTPUT")?, 1, parse);
        let width = values.width.max(1);
        let n = times.data.len().min(values.data.len() / width);
        if n == 0 || times.width != 1 {
            return None;
        }
        let interp = self
            .input(sampler, "INTERPOLATION")
            .map(|s| geometry::names(dom, s))
            .unwrap_or_default();
        let all = |name: &str| match name {
            "STEP" => Interp::Step,
            "BEZIER" => Interp::Bezier,
            "HERMITE" => Interp::Hermite,
            "BSPLINE" => Interp::BSpline,
            "CARDINAL" => Interp::Cardinal,
            _ => Interp::Linear,
        };
        let interp: Vec<Interp> = (0..n)
            .map(|k| interp.get(k).or(interp.first()).map_or(Interp::Linear, |s| all(s)))
            .collect();
        let tangents = |semantic: &str| {
            self.input(sampler, semantic)
                .map(|s| geometry::read(dom, s, width, parse))
                .unwrap_or_default()
        };
        let mut times = times.data;
        times.truncate(n);
        // Keys out of order are dropped.
        let mut keep = vec![true; n];
        let mut last = f64::NEG_INFINITY;
        for (k, t) in times.iter().enumerate() {
            if t.is_finite() && *t > last {
                last = *t;
            } else {
                keep[k] = false;
            }
        }
        let mut curve = Curve {
            times,
            values: values.data,
            width,
            interp,
            in_tan: tangents("IN_TANGENT"),
            out_tan: tangents("OUT_TANGENT"),
        };
        if keep.iter().any(|k| !k) {
            let pick = |v: &[f64], w: usize| -> Vec<f64> {
                (0..n)
                    .filter(|&k| keep[k])
                    .flat_map(|k| (0..w).map(move |c| v.get(k * w + c).copied().unwrap_or(0.0)))
                    .collect()
            };
            curve.values = pick(&curve.values, width);
            curve.in_tan.data = pick(&curve.in_tan.data, curve.in_tan.width);
            curve.out_tan.data = pick(&curve.out_tan.data, curve.out_tan.width);
            curve.times = pick(&curve.times, 1);
            curve.interp = (0..n).filter(|&k| keep[k]).map(|k| curve.interp[k]).collect();
        }
        Some(curve)
    }

    /// A node's local transform at time `t`, with its animated numbers from `tracks`.
    fn local_at(&self, node: usize, tracks: &[&Track], t: f64) -> M4 {
        let dom = self.dom;
        let mut m = IDENTITY;
        for c in dom.children(node) {
            let Some(kind) = Transform::of(dom.name(c)) else {
                continue;
            };
            let mut v = Vec::new();
            xml::floats(dom.text(c), &mut v);
            for track in tracks.iter().filter(|tr| tr.transform == c) {
                if let Some(i) = track.member {
                    if v.len() <= i {
                        v.resize(kind.len(), 0.0);
                    }
                    if let Some(x) = v.get_mut(i) {
                        *x = track.curve.eval(t, 0);
                    }
                } else {
                    v.resize(v.len().max(kind.len()).max(track.curve.width), 0.0);
                    for (k, x) in v.iter_mut().enumerate().take(track.curve.width) {
                        *x = track.curve.eval(t, k);
                    }
                }
            }
            m = mul(&m, &kind.matrix(&v));
        }
        m
    }

    /// The glTF animation of `tracks`, sampled within `range`; `None` when it moves nothing.
    fn animation(&mut self, tracks: &[Track], name: Option<&str>, range: Option<(f64, f64)>) -> Option<Value> {
        let dom = self.dom;
        let mut by_node: Vec<(usize, Vec<&Track>)> = Vec::new();
        for t in tracks.iter().filter(|t| !t.weights) {
            match by_node.iter_mut().find(|(n, _)| *n == t.node) {
                Some((_, list)) => list.push(t),
                None => by_node.push((t.node, vec![t])),
            }
        }
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for (node, list) in by_node {
            let Some(instances) = self.instances.get(&node).cloned() else {
                continue;
            };
            let mut times = sample_times(&list, range);
            if times.is_empty() {
                continue;
            }
            let mut quats: Vec<[f64; 4]> = times
                .iter()
                .map(|&t| decompose(&self.local_at(node, &list, t)).r)
                .collect();
            // Long turns cut into short ones.
            let mut cut = Vec::with_capacity(times.len());
            for k in 0..times.len() {
                cut.push(times[k]);
                if k + 1 < times.len() && cut.len() < MAX_SAMPLES {
                    let angle = turn(quats[k], quats[k + 1]);
                    if angle > MAX_TURN {
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            reason = "a small positive count"
                        )]
                        let steps = ((angle / MAX_TURN).ceil() as usize).min(MAX_CUTS);
                        #[allow(clippy::cast_precision_loss, reason = "a handful of steps")]
                        cut.extend((1..steps).map(|i| times[k] + (times[k + 1] - times[k]) * i as f64 / steps as f64));
                    }
                }
            }
            if cut.len() != times.len() {
                cut.truncate(MAX_SAMPLES);
                times = cut;
            }
            // Times as glTF holds them: not negative, and increasing.
            let mut input = Vec::with_capacity(times.len());
            let mut kept = Vec::with_capacity(times.len());
            for &t in &times {
                let f = f32_of(t.max(0.0));
                if input.last().is_none_or(|&l: &f32| f > l) {
                    input.push(f);
                    kept.push(t);
                }
            }
            let trs: Vec<_> = kept
                .iter()
                .map(|&t| decompose(&self.local_at(node, &list, t)))
                .collect();
            quats.clear();
            let mut prev = None;
            for d in &trs {
                let q = continuous(d.r, prev);
                prev = Some(q);
                quats.push(q);
            }
            let kinds: Vec<Transform> = list
                .iter()
                .filter_map(|tr| Transform::of(dom.name(tr.transform)))
                .collect();
            let has = |ks: &[Transform]| kinds.iter().any(|k| ks.contains(k));
            let input_acc = self.b.floats(input, "SCALAR", true);
            let mut outputs = Vec::new();
            if has(&[Transform::Translate, Transform::Matrix, Transform::Lookat]) {
                let v = trs.iter().flat_map(|d| d.t.map(f32_of)).collect();
                outputs.push(("translation", self.b.floats(v, "VEC3", false)));
            }
            if has(&[Transform::Rotate, Transform::Matrix, Transform::Lookat, Transform::Skew]) {
                let v = quats.iter().flat_map(|q| q.map(f32_of)).collect();
                outputs.push(("rotation", self.b.floats(v, "VEC4", false)));
            }
            if has(&[Transform::Scale, Transform::Matrix, Transform::Skew]) {
                let v = trs.iter().flat_map(|d| d.s.map(f32_of)).collect();
                outputs.push(("scale", self.b.floats(v, "VEC3", false)));
            }
            let static_trs = decompose(&self.local(node));
            for (path, output) in outputs {
                samplers.push(json!({ "input": input_acc, "output": output, "interpolation": "LINEAR" }));
                let s = samplers.len() - 1;
                for &n in &instances {
                    channels.push(json!({ "sampler": s, "target": { "node": n, "path": path } }));
                }
            }
            // glTF animates only nodes with TRS transforms.
            for &n in &instances {
                if let Some(o) = self.nodes[n].as_object_mut()
                    && o.remove("matrix").is_some()
                {
                    o.insert("translation".into(), json!(static_trs.t));
                    o.insert("rotation".into(), json!(static_trs.r));
                    o.insert("scale".into(), json!(static_trs.s));
                }
            }
        }
        self.weight_channels(tracks, range, &mut samplers, &mut channels);
        if channels.is_empty() {
            return None;
        }
        let mut anim = json!({ "channels": channels, "samplers": samplers });
        if let Some(name) = name {
            anim["name"] = json!(name);
        }
        Some(anim)
    }
}

impl super::Reader<'_> {
    /// The channels of the morph weights `tracks` animate, on every node that draws a mesh of the morph.
    fn weight_channels(
        &mut self,
        tracks: &[Track],
        range: Option<(f64, f64)>,
        samplers: &mut Vec<Value>,
        channels: &mut Vec<Value>,
    ) {
        let mut by_morph: Vec<(usize, Vec<&Track>)> = Vec::new();
        for t in tracks.iter().filter(|t| t.weights) {
            match by_morph.iter_mut().find(|(n, _)| *n == t.node) {
                Some((_, list)) => list.push(t),
                None => by_morph.push((t.node, vec![t])),
            }
        }
        for (morph, list) in by_morph {
            // The meshes of the morph that have their targets, and the nodes drawing them.
            let meshes: Vec<(usize, Vec<f64>)> = self
                .morphs
                .iter()
                .filter(|&&(_, m)| m == morph)
                .filter_map(|&(mesh, _)| {
                    let w = self.meshes[mesh].get("weights")?.as_array()?;
                    Some((mesh, w.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()))
                })
                .collect();
            let Some((_, defaults)) = meshes.first() else {
                continue;
            };
            let count = defaults.len();
            let nodes: Vec<usize> = (0..self.nodes.len())
                .filter(|&n| {
                    self.nodes[n]
                        .get("mesh")
                        .and_then(Value::as_u64)
                        .is_some_and(|m| meshes.iter().any(|(k, _)| *k as u64 == m))
                })
                .collect();
            let times = sample_times(&list, range);
            if nodes.is_empty() || times.is_empty() || count == 0 {
                continue;
            }
            let mut input = Vec::with_capacity(times.len());
            let mut output = Vec::with_capacity(times.len() * count);
            for &t in &times {
                let f = f32_of(t.max(0.0));
                if input.last().is_some_and(|&l: &f32| f <= l) {
                    continue;
                }
                input.push(f);
                for (k, &default) in defaults.iter().enumerate() {
                    let value = list
                        .iter()
                        .rev()
                        .find_map(|tr| match tr.member {
                            Some(m) if m == k => Some(tr.curve.eval(t, 0)),
                            None if k < tr.curve.width => Some(tr.curve.eval(t, k)),
                            _ => None,
                        })
                        .unwrap_or(default);
                    output.push(if value.is_finite() { f32_of(value) } else { 0.0 });
                }
            }
            let input_acc = self.b.floats(input, "SCALAR", true);
            let output_acc = self.b.floats(output, "SCALAR", false);
            samplers.push(json!({ "input": input_acc, "output": output_acc, "interpolation": "LINEAR" }));
            let s = samplers.len() - 1;
            for n in nodes {
                channels.push(json!({ "sampler": s, "target": { "node": n, "path": "weights" } }));
            }
        }
    }
}

/// The times a node's tracks are sampled at within `range`: every curve's keys, points within its curved segments,
/// and the last moment of each step, in order.
fn sample_times(list: &[&Track], range: Option<(f64, f64)>) -> Vec<f64> {
    let mut times: Vec<f64> = Vec::new();
    for tr in list {
        let c = &tr.curve;
        for (k, &t) in c.times.iter().enumerate() {
            times.push(t);
            if c.interp.get(k) == Some(&Interp::Step)
                && let Some(&next) = c.times.get(k + 1)
                && let Some(before) = just_before(next)
            {
                times.push(before);
            }
            if c.curved(k)
                && let Some(&next) = c.times.get(k + 1)
            {
                #[allow(clippy::cast_precision_loss, reason = "a handful of steps")]
                times.extend((1..CURVE_STEPS).map(|i| t + (next - t) * i as f64 / CURVE_STEPS as f64));
            }
        }
    }
    if let Some((start, end)) = range {
        times.retain(|&t| t >= start && t <= end);
    }
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    times.dedup();
    times.truncate(MAX_SAMPLES);
    times
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(interp: Interp, values: &[f64], tan: geometry::Acc<f64>) -> Curve {
        Curve {
            times: vec![0.0, 1.0],
            values: values.to_vec(),
            width: 1,
            interp: vec![interp; 2],
            in_tan: geometry::Acc {
                width: tan.width,
                data: tan.data.clone(),
            },
            out_tan: tan,
        }
    }

    #[test]
    fn interpolations() {
        let none = || geometry::Acc {
            width: 1,
            data: Vec::new(),
        };
        assert!((curve(Interp::Linear, &[0.0, 2.0], none()).eval(0.25, 0) - 0.5).abs() < 1e-12);
        assert!(curve(Interp::Step, &[0.0, 2.0], none()).eval(0.75, 0).abs() < 1e-12);
        // Held outside the keys.
        assert!((curve(Interp::Linear, &[0.0, 2.0], none()).eval(5.0, 0) - 2.0).abs() < 1e-12);
        // A two-dimensional Bezier whose handles lie on the line is the line.
        let line = geometry::Acc {
            width: 2,
            data: vec![-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
        };
        let c = Curve {
            in_tan: geometry::Acc {
                width: 2,
                data: vec![2.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0],
            },
            out_tan: line,
            ..curve(Interp::Bezier, &[0.0, 2.0], none())
        };
        assert!((c.eval(0.3, 0) - 0.6).abs() < 1e-6, "{}", c.eval(0.3, 0));
        // Ease-in-out: slow at the ends, half-way at the middle.
        let ease = Curve {
            in_tan: geometry::Acc {
                width: 2,
                data: vec![0.6, 1.0, 0.6, 1.0],
            },
            out_tan: geometry::Acc {
                width: 2,
                data: vec![0.4, 0.0, 0.4, 0.0],
            },
            ..curve(Interp::Bezier, &[0.0, 1.0], none())
        };
        assert!((ease.eval(0.5, 0) - 0.5).abs() < 1e-6);
        assert!(ease.eval(0.1, 0) < 0.05);
        // Hermite with zero tangents is smoothstep.
        let h = curve(
            Interp::Hermite,
            &[0.0, 1.0],
            geometry::Acc {
                width: 1,
                data: vec![0.0, 0.0],
            },
        );
        assert!((h.eval(0.25, 0) - 0.156_25).abs() < 1e-12);
    }
}
