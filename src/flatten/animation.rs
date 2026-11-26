//! The pose: each node's animated properties at a time, from samplers interpolated between their keys.

use serde_json::Value;

use super::{IDENTITY, M4, NodeView, Reader, at, floats, index, items, trs};
use crate::document::Pose;
use crate::num::f32_of;
use crate::quat;
use crate::{Context as _, ErrorKind, Result};

/// The pointer path, under a node, of its `KHR_node_visibility` flag.
pub(super) const VISIBLE: &str = "extensions/KHR_node_visibility/visible";

/// The node and property (`translation`, `rotation`, `scale` or `weights`) an animation
/// channel's target animates: named outright, or by a `KHR_animation_pointer` to one of them
/// or to the node's `KHR_node_visibility` flag ([`VISIBLE`]).
pub(crate) fn node_target(target: &Value) -> Option<(usize, &str)> {
    if let (Some(node), Some(path)) = (index(target, "node"), target.get("path").and_then(Value::as_str))
        && path != "pointer"
    {
        return Some((node, path));
    }
    let pointer = at(target, &["extensions", "KHR_animation_pointer", "pointer"]).and_then(Value::as_str)?;
    let (node, path) = pointer.strip_prefix("/nodes/")?.split_once('/')?;
    // An index the way JSON pointers write them: no sign, no leading zeros.
    if node.is_empty() || !node.bytes().all(|b| b.is_ascii_digit()) || (node.len() > 1 && node.starts_with('0')) {
        return None;
    }
    let path = ["translation", "rotation", "scale", "weights", VISIBLE]
        .into_iter()
        .find(|&p| p == path)?;
    Some((node.parse().ok()?, path))
}

impl Reader<'_> {
    /// When animation `a`'s first key is, and its last, over the channels that move nodes:
    /// `(0, 0)` for one without keys.
    pub(super) fn span(&mut self, a: usize) -> Result<(f32, f32)> {
        let anim = self.get("animations", a)?;
        let samplers = items(anim, "samplers");
        let (mut first, mut last) = (f32::INFINITY, f32::NEG_INFINITY);
        for ch in items(anim, "channels") {
            if ch.get("target").and_then(node_target).is_none() {
                continue;
            }
            let Some(input) = index(ch, "sampler")
                .and_then(|s| samplers.get(s))
                .and_then(|s| index(s, "input"))
            else {
                continue;
            };
            let times = self.read::<1>(input, [0.0])?;
            if let (Some([a]), Some([b])) = (times.first(), times.last()) {
                first = first.min(*a);
                last = last.max(*b);
            }
        }
        Ok(if first <= last { (first, last) } else { (0.0, 0.0) })
    }

    /// Sets each node's animated properties for `pose`.
    pub(super) fn pose(&mut self, pose: Pose) -> Result<()> {
        let Pose::Animation { index: a, time } = pose else {
            return Ok(());
        };
        let anim = self.get("animations", a)?;
        let samplers = items(anim, "samplers");
        for ch in items(anim, "channels") {
            let Some((node, path)) = ch.get("target").and_then(node_target) else {
                // A channel for another extension's target, or none.
                continue;
            };
            if node >= self.posed.len() {
                continue;
            }
            let s = index(ch, "sampler")
                .and_then(|s| samplers.get(s))
                .context(ErrorKind::Corrupt, "glTF animation channel names a missing sampler")?;
            let (Some(input), Some(output)) = (index(s, "input"), index(s, "output")) else {
                bail!(Corrupt, "glTF animation sampler has no input or output");
            };
            let interpolation = s.get("interpolation").and_then(Value::as_str).unwrap_or("LINEAR");
            let times: Vec<f32> = self.read::<1>(input, [0.0])?.into_flattened();
            let values: Vec<f32> = match path {
                "translation" | "scale" => self.read::<3>(output, [0.0; 3])?.into_flattened(),
                "rotation" => self.read::<4>(output, [0.0, 0.0, 0.0, 1.0])?.into_flattened(),
                "weights" | VISIBLE => self.read::<1>(output, [0.0])?.into_flattened(),
                _ => continue,
            };
            // A flag has nothing between its keys: it holds each until the next.
            let interpolation = if path == VISIBLE { "STEP" } else { interpolation };
            let Some(v) = sample(&times, &values, interpolation, time, path == "rotation") else {
                continue;
            };
            // A sampler with fewer outputs than keys samples narrower than its path: skipped,
            // as `Rig::pose` skips it, rather than read past its values.
            let p = &mut self.posed[node];
            match (path, v.as_slice()) {
                ("translation", &[x, y, z]) => p.translation = Some([x, y, z]),
                ("scale", &[x, y, z]) => p.scale = Some([x, y, z]),
                ("rotation", &[x, y, z, w]) => p.rotation = Some([x, y, z, w]),
                ("weights", _) => p.weights = Some(v.into_iter().map(f32_of).collect()),
                (VISIBLE, &[x]) => p.visible = Some(x != 0.0),
                _ => {}
            }
        }
        Ok(())
    }

    /// Node `i`'s own transform, as posed.
    pub(super) fn local(&self, i: usize, node: &NodeView<'_>) -> M4 {
        let p = &self.posed[i];
        // A node with a matrix can't be animated.
        if let Some(m) = node
            .matrix
            .and_then(Value::as_array)
            .filter(|m| m.len() == 16 && !p.given)
        {
            let mut out = IDENTITY;
            for (o, x) in out.iter_mut().zip(m) {
                *o = x.as_f64().unwrap_or(*o);
            }
            return out;
        }
        trs(
            p.translation.unwrap_or_else(|| floats(node.translation, [0.0; 3])),
            p.rotation
                .unwrap_or_else(|| floats(node.rotation, [0.0, 0.0, 0.0, 1.0])),
            p.scale.unwrap_or_else(|| floats(node.scale, [1.0; 3])),
        )
    }
}

/// An animation sampler's value at `t`: `values` holds one element per key (three, in-tangent,
/// value and out-tangent, for `CUBICSPLINE`), each `values.len() / keys` wide. Before the
/// first key it's the first value, after the last the last. `None` for a sampler without keys
/// or with values that don't fit them.
pub(crate) fn sample(times: &[f32], values: &[f32], interpolation: &str, t: f32, rotation: bool) -> Option<Vec<f64>> {
    let mut out = vec![0.0; width(times, values, interpolation)?];
    sample_into(times, values, interpolation, t, rotation, &mut out).then_some(out)
}

/// How wide each of a sampler's elements is: `None` for a sampler without keys or with values
/// that don't fit them.
pub(crate) fn width(times: &[f32], values: &[f32], interpolation: &str) -> Option<usize> {
    let keys = times.len() * if interpolation == "CUBICSPLINE" { 3 } else { 1 };
    (keys > 0 && !values.is_empty() && values.len().is_multiple_of(keys)).then(|| values.len() / keys)
}

/// [`sample`] written into `out`, which must be the sampler's [`width`]: `false`, leaving `out`
/// as it was, if it isn't or the sampler can't be read. Allocates nothing. A rotation comes out
/// unit length at every time, its first and last keys too.
pub(crate) fn sample_into(
    times: &[f32],
    values: &[f32],
    interpolation: &str,
    t: f32,
    rotation: bool,
    out: &mut [f64],
) -> bool {
    let k = times.partition_point(|&x| x <= t);
    sample_after(times, values, interpolation, t, rotation, k, out)
}

/// [`sample_into`] given `k`, how many keys are at or before `t`, as a caller walking through time in order can
/// know without searching.
pub(crate) fn sample_after(
    times: &[f32],
    values: &[f32],
    interpolation: &str,
    t: f32,
    rotation: bool,
    k: usize,
    out: &mut [f64],
) -> bool {
    let Some(w) = width(times, values, interpolation) else {
        return false;
    };
    if out.len() != w {
        return false;
    }
    let keys = times.len();
    let cubic = interpolation == "CUBICSPLINE";
    let per = if cubic { 3 } else { 1 };
    // The value of key k (its middle element for a cubic spline), and its tangents.
    let at = |k: usize, part: usize| -> &[f32] {
        let e = k * per + part;
        &values[e * w..(e + 1) * w]
    };
    let value = |k: usize| at(k, usize::from(cubic));
    let copy = |out: &mut [f64], v: &[f32]| {
        for (o, &x) in out.iter_mut().zip(v) {
            *o = f64::from(x);
        }
    };
    if k == 0 {
        copy(out, value(0));
    } else if k == keys {
        copy(out, value(keys - 1));
    } else {
        let (k, (t0, t1)) = (k - 1, (f64::from(times[k - 1]), f64::from(times[k])));
        let dt = t1 - t0;
        let s = if dt > 0.0 { (f64::from(t) - t0) / dt } else { 0.0 };
        match interpolation {
            // On a key every interpolation gives its value: sampling at the keys, as a clip baked frame by frame
            // does, needs no slerp or spline.
            _ if s == 0.0 => copy(out, value(k)),
            "STEP" => copy(out, value(k)),
            "CUBICSPLINE" => {
                // Hermite, per the glTF 2.0 specification's appendix C.
                let (v0, b0, a1, v1) = (at(k, 1), at(k, 2), at(k + 1, 0), at(k + 1, 1));
                let (s2, s3) = (s * s, s * s * s);
                let h = [
                    2.0 * s3 - 3.0 * s2 + 1.0,
                    (s3 - 2.0 * s2 + s) * dt,
                    -2.0 * s3 + 3.0 * s2,
                    (s3 - s2) * dt,
                ];
                for (c, o) in out.iter_mut().enumerate() {
                    let f = |v: &[f32]| f64::from(v[c]);
                    *o = h[0] * f(v0) + h[1] * f(b0) + h[2] * f(v1) + h[3] * f(a1);
                }
            }
            _ if rotation && w == 4 => {
                let q = |v: &[f32]| std::array::from_fn(|c| f64::from(v[c]));
                out.copy_from_slice(&quat::slerp(q(value(k)), q(value(k + 1)), s));
            }
            _ => {
                for ((o, &x), &y) in out.iter_mut().zip(value(k)).zip(value(k + 1)) {
                    let (x, y) = (f64::from(x), f64::from(y));
                    *o = x + (y - x) * s;
                }
            }
        }
    }
    if rotation && w == 4 {
        let len = out.iter().map(|c| c * c).sum::<f64>().sqrt();
        if len > 0.0 {
            for c in out.iter_mut() {
                *c /= len;
            }
        }
    }
    true
}
