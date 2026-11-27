//! The one finisher every reader's animations are written by, so an animation is the same kind of thing whatever
//! format held it.
//!
//! A reader gathers an animation's [`Track`]s (a node, what of it is animated, the key times in seconds and the
//! values at them) and hands them to a [`Writer`], which guarantees what glTF requires and what callers rely on:
//!
//! - **It starts at time 0.** The earliest key of all the animation's tracks is found, and that much is taken off
//!   every key of every track: the whole animation moves, never one node against another. What was taken off is
//!   kept in the animation's `extras` as [`TIME_OFFSET`], in seconds, so a time in the file is the animation's time
//!   plus the offset. An animation that already starts at 0 has no such key.
//! - **Times are finite and strictly increasing.** A key whose time or values aren't finite is dropped; a key at
//!   or before the one ahead of it moves to the next time after it that single precision can hold.
//! - **Rotations are unit quaternions, each on the same side as the one before**, so a blend between two keys goes
//!   the short way round.
//! - Tracks with the same times share one time accessor, and all the keys of a document are in one buffer view.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::document::{self, Accessor, Builder, Limits, Part, Sampler};
use crate::num::f32_of;
use crate::quat::normalize;

/// The key in an animation's `extras` holding the seconds taken off every key so that the animation starts at 0:
/// the time of its earliest key in the file. Absent when that was 0 already.
pub(crate) const TIME_OFFSET: &str = "timeOffset";

/// The frame rate a reader assumes when neither the file nor its format gives one: frames, keys or ticks a
/// second. A format with a convention of its own keeps it (MD2, MD5, `Blitz3D`, DirectX, ASF/AMC, USD), and its
/// reader cites where the convention comes from.
pub(crate) const ASSUMED_FRAME_RATE: f64 = 30.0;

/// The name of the one animation a motion file holds, where the file itself gives it none: the file's own name
/// without its extension, as a library of clips names them (`walk.bvh` is `walk`). `None` for a model read with no
/// file name. Only for the formats that are one clip to a file (BVH, AMC, C3D, TRC, HTR, VMD, `.md5anim`, SMD); an
/// animation inside a model file that the file doesn't name has no name.
pub(crate) fn clip_name(path: &std::path::Path) -> Option<String> {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
}

/// What an animation channel sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Path {
    Translation,
    Rotation,
    Scale,
    Weights,
}

impl Path {
    /// The path as glTF spells it.
    pub fn name(self) -> &'static str {
        match self {
            Path::Translation => "translation",
            Path::Rotation => "rotation",
            Path::Scale => "scale",
            Path::Weights => "weights",
        }
    }

    /// The path glTF spells `name`.
    pub fn of(name: &str) -> Option<Path> {
        Some(match name {
            "translation" => Path::Translation,
            "rotation" => Path::Rotation,
            "scale" => Path::Scale,
            "weights" => Path::Weights,
            _ => return None,
        })
    }
}

/// How values between keys are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Interp {
    /// Each key's value holds until the next.
    Step,
    Linear,
    /// glTF's cubic spline: each key is an in-tangent, a value and an out-tangent.
    Cubic,
}

/// Key times in seconds, in the precision the reader has them: double precision keeps a late start exact until
/// it is taken off.
#[derive(Debug, Clone)]
pub(crate) enum Times {
    Single(Vec<f32>),
    Double(Vec<f64>),
}

impl Times {
    fn len(&self) -> usize {
        match self {
            Times::Single(v) => v.len(),
            Times::Double(v) => v.len(),
        }
    }

    /// The earliest finite time.
    fn first(&self) -> Option<f64> {
        let least = |a: Option<f64>, t: f64| match a {
            _ if !t.is_finite() => a,
            Some(a) if a <= t => Some(a),
            _ => Some(t),
        };
        match self {
            Times::Single(v) => v.iter().map(|&t| f64::from(t)).fold(None, least),
            Times::Double(v) => v.iter().copied().fold(None, least),
        }
    }

    /// The times in single precision, `offset` taken off first.
    fn from(self, offset: f64) -> Vec<f32> {
        #[allow(clippy::float_cmp, reason = "exactly zero: nothing to take off")]
        match self {
            Times::Single(v) if offset == 0.0 => v,
            Times::Single(v) => v.into_iter().map(|t| f32_of(f64::from(t) - offset)).collect(),
            Times::Double(v) => v.into_iter().map(|t| f32_of(t - offset)).collect(),
        }
    }
}

impl From<Vec<f32>> for Times {
    fn from(v: Vec<f32>) -> Self {
        Times::Single(v)
    }
}

impl From<Vec<f64>> for Times {
    fn from(v: Vec<f64>) -> Self {
        Times::Double(v)
    }
}

/// One animated property of one node (or of several nodes that move alike).
#[derive(Debug, Clone)]
pub(crate) struct Track {
    pub node: usize,
    /// Other nodes the same keys drive: instances of one animated thing.
    pub also: Vec<usize>,
    pub path: Path,
    pub interp: Interp,
    /// Seconds.
    pub times: Times,
    /// `width` floats to a key, or three times that for [`Interp::Cubic`].
    pub values: Vec<f32>,
    /// Floats to a value: 3, 4 for a rotation, or the number of morph targets for weights.
    pub width: usize,
}

impl Track {
    /// A linear track of `path` on `node`, its width the path's (weights need [`Track::width`] set).
    pub fn linear(node: usize, path: Path, times: impl Into<Times>, values: Vec<f32>) -> Track {
        Track {
            node,
            also: Vec::new(),
            path,
            interp: Interp::Linear,
            times: times.into(),
            values,
            width: if path == Path::Rotation { 4 } else { 3 },
        }
    }
}

/// Writes a document's animations: see the module's notes for what it guarantees.
pub(crate) struct Writer {
    /// Whether to note the time taken off in `extras`.
    extras: bool,
    /// Each distinct set of times is written once, by a hash of their bits: channels usually share theirs.
    inputs: HashMap<u64, Vec<(Vec<f32>, usize)>>,
    /// Every key is written to one buffer view, each accessor at its offset into it: one part and one view, where
    /// a view per accessor would cost far more than the keys of a short channel.
    data: Vec<f32>,
    placed: Vec<usize>,
}

impl Writer {
    /// A writer; `extras` is whether the document keeps extras.
    pub fn new(extras: bool) -> Writer {
        Writer {
            extras,
            inputs: HashMap::new(),
            data: Vec::new(),
            placed: Vec::new(),
        }
    }

    fn place(&mut self, b: &mut Builder, v: &[f32], count: usize, kind: &'static str) -> usize {
        b.accessors.push(Accessor {
            byte_offset: Some(self.data.len() as u64 * 4),
            component_type: 5126,
            count,
            kind,
            ..Accessor::default()
        });
        self.data.extend_from_slice(v);
        self.placed.push(b.accessors.len() - 1);
        b.accessors.len() - 1
    }

    /// Adds the animation of `tracks` to `b`, starting at time 0, and returns the seconds taken off its keys to
    /// have it so; `None`, and nothing added, when no track has a key left. `extras` are the reader's own, which
    /// the time offset joins.
    pub fn animation(
        &mut self,
        b: &mut Builder,
        name: Option<String>,
        extras: Option<Value>,
        tracks: Vec<Track>,
    ) -> Option<f64> {
        let (offset, single) = start(tracks.iter().map(|t| &t.times));
        let mut channels = Vec::new();
        let mut samplers = Vec::new();
        for mut t in tracks {
            let (node, path, interp) = (t.node, t.path, t.interp);
            let also = std::mem::take(&mut t.also);
            let width = t.width.max(1);
            let Some((times, values)) = clean(t, offset) else {
                continue;
            };
            let count = times.len();
            let hash = times.iter().fold(times.len() as u64, |h, t| {
                (h.rotate_left(5) ^ u64::from(t.to_bits())).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            });
            let found = self.inputs.get(&hash).and_then(|list| {
                list.iter()
                    .find(|(t, _)| t.iter().map(|x| x.to_bits()).eq(times.iter().map(|x| x.to_bits())))
                    .map(|&(_, i)| i)
            });
            let input = if let Some(i) = found {
                i
            } else {
                let i = self.place(b, &times, count, "SCALAR");
                // Strictly increasing, so the first and last are the bounds.
                b.accessors[i].min = Some(Limits::floats(&times[..1]));
                b.accessors[i].max = Some(Limits::floats(&times[count - 1..]));
                self.inputs.entry(hash).or_default().push((times, i));
                i
            };
            let per = if interp == Interp::Cubic { 3 } else { 1 };
            let output = match path {
                Path::Weights => self.place(b, &values, count * per * width, "SCALAR"),
                Path::Rotation => self.place(b, &values, count * per, "VEC4"),
                Path::Translation | Path::Scale => self.place(b, &values, count * per, "VEC3"),
            };
            for node in std::iter::once(node).chain(also) {
                channels.push(document::Channel {
                    sampler: samplers.len(),
                    node,
                    path: path.name(),
                });
            }
            samplers.push(Sampler {
                input,
                output,
                interpolation: match interp {
                    Interp::Step => "STEP",
                    Interp::Linear => "LINEAR",
                    Interp::Cubic => "CUBICSPLINE",
                },
            });
        }
        if channels.is_empty() {
            return None;
        }
        let mut extras = extras.filter(|_| self.extras);
        if self.extras {
            note_offset(&mut extras, offset, single);
        }
        b.animations.push(document::Animation {
            channels,
            samplers,
            name: name.filter(|n| !n.is_empty()),
            extras,
        });
        Some(offset)
    }

    /// Writes the keys' buffer view, once every animation is added.
    pub fn finish(self, b: &mut Builder) {
        if !self.data.is_empty() {
            let view = b.view(Part::f32(self.data), None);
            for a in self.placed {
                b.accessors[a].buffer_view = Some(view);
            }
        }
    }
}

/// Where an animation whose tracks have `times` starts: its earliest key of all, and whether that is a
/// single-precision time.
fn start<'t>(times: impl IntoIterator<Item = &'t Times>) -> (f64, bool) {
    times
        .into_iter()
        .filter_map(|t| Some((t.first()?, matches!(t, Times::Single(_)))))
        .fold(None, |a: Option<(f64, bool)>, t| match a {
            Some(a) if a.0 <= t.0 => Some(a),
            _ => Some(t),
        })
        .unwrap_or((0.0, false))
}

/// Notes the `offset` seconds taken off an animation's keys in its `extras` as [`TIME_OFFSET`]; nothing when
/// nothing was taken off.
fn note_offset(extras: &mut Option<Value>, offset: f64, single: bool) {
    #[allow(clippy::float_cmp, reason = "exactly zero: nothing was taken off")]
    if offset == 0.0 {
        return;
    }
    let e = extras.get_or_insert_with(|| Value::Object(Map::new()));
    if let Some(o) = e.as_object_mut() {
        // A single-precision time is noted as the number it was written as (1.2, not 1.2000000476837158).
        let noted = if single {
            f32_of(offset).to_string().parse().unwrap_or(offset)
        } else {
            offset
        };
        o.insert(TIME_OFFSET.into(), json!(noted));
    }
}

/// What becomes of one sampler of a glTF animation held to the rules.
#[cfg(feature = "gltf")]
enum Kept {
    /// Its keys are clean and start where the animation does: left as the file has them.
    Same,
    /// Its keys as they now are: times, and values with how many floats make one of the output's elements.
    New(Vec<f32>, Vec<f32>, usize),
    /// No key is left.
    Gone,
}

/// A sampler of a glTF animation as [`conform`] rewrites it: `None` for one with no key left, else its new input
/// and output accessors, or `None` for one left as it is.
#[cfg(feature = "gltf")]
type Rewritten = Option<Option<(usize, usize)>>;

/// Holds the animations of a glTF document that a reader converts (glTF 1.0, VRM, 3D Tiles) to what [`Writer`]
/// guarantees of every reader's animations: each starts at 0 as a whole, what was taken off noted in its
/// `extras` as [`TIME_OFFSET`] when `extras` are kept, and its keys are clean. A sampler whose keys already are
/// so is left as the file has it, and only the keys that change are written, to a new buffer. A channel whose
/// sampler has no key left is dropped, and an animation with no channel left. An animation whose keys can't be
/// read is left as it is.
#[cfg(feature = "gltf")]
pub(crate) fn conform(doc: &mut document::Document, extras: bool) {
    let animations = doc
        .json()
        .get("animations")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let mut plans: Vec<Option<(Vec<Kept>, f64)>> = Vec::with_capacity(animations);
    for a in 0..animations {
        plans.push(plan(doc, a));
    }
    if plans.iter().flatten().all(|(kept, offset)| {
        #[allow(clippy::float_cmp, reason = "exactly zero: nothing to take off")]
        let at_zero = *offset == 0.0;
        at_zero && kept.iter().all(|k| matches!(k, Kept::Same))
    }) {
        return;
    }
    // The new keys, in a buffer of their own; times that are the same written once.
    let mut added = document::Added::new(doc);
    let mut inputs: HashMap<Vec<u32>, usize> = HashMap::new();
    // Each changed animation: its index, its samplers and the seconds taken off.
    let mut changed: Vec<(usize, Vec<Rewritten>, f64)> = Vec::new();
    for (a, plan) in plans.into_iter().enumerate() {
        let Some((kept, offset)) = plan else { continue };
        let mut at = Vec::with_capacity(kept.len());
        for k in kept {
            at.push(match k {
                Kept::Same => Some(None),
                Kept::Gone => None,
                Kept::New(times, values, n) => {
                    let key: Vec<u32> = times.iter().map(|t| t.to_bits()).collect();
                    let input = *inputs.entry(key).or_insert_with(|| {
                        // Strictly increasing, so the first and last are the bounds.
                        let bounds = (vec![times[0]], vec![times[times.len() - 1]]);
                        added.floats(&times, "SCALAR", Some(bounds))
                    });
                    let kind = match n {
                        3 => "VEC3",
                        4 => "VEC4",
                        _ => "SCALAR",
                    };
                    Some(Some((input, added.floats(&values, kind, None))))
                }
            });
        }
        changed.push((a, at, offset));
    }
    added.finish(doc);
    let json = doc.json_mut();
    let Some(Value::Array(list)) = json.get_mut("animations") else {
        return;
    };
    let mut emptied = Vec::new();
    for (a, at, offset) in changed {
        let Some(anim) = list.get_mut(a) else { continue };
        // The samplers kept, each at its new place.
        let mut renumber: Vec<Option<usize>> = Vec::with_capacity(at.len());
        let mut samplers = Vec::new();
        let old = anim
            .get("samplers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for (mut sampler, at) in old.into_iter().zip(at) {
            let Some(new) = at else {
                renumber.push(None);
                continue;
            };
            if let Some((input, output)) = new {
                sampler["input"] = json!(input);
                sampler["output"] = json!(output);
            }
            renumber.push(Some(samplers.len()));
            samplers.push(sampler);
        }
        let channels: Vec<Value> = anim
            .get("channels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|c| {
                let s = c.get("sampler")?.as_u64().and_then(|s| usize::try_from(s).ok())?;
                let to = (*renumber.get(s)?)?;
                let mut c = c.clone();
                c["sampler"] = json!(to);
                Some(c)
            })
            .collect();
        if channels.is_empty() {
            emptied.push(a);
            continue;
        }
        anim["samplers"] = Value::Array(samplers);
        anim["channels"] = Value::Array(channels);
        if extras {
            let mut e = anim.get("extras").filter(|e| e.is_object()).cloned();
            note_offset(&mut e, offset, true);
            if let Some(e) = e {
                anim["extras"] = e;
            }
        }
    }
    for a in emptied.into_iter().rev() {
        list.remove(a);
    }
    if list.is_empty() {
        json.remove("animations");
    }
}

/// What becomes of each of animation `a`'s samplers held to the rules, and the seconds taken off its keys;
/// `None` when a sampler's keys can't be read.
#[cfg(feature = "gltf")]
fn plan(doc: &document::Document, a: usize) -> Option<(Vec<Kept>, f64)> {
    let anim = doc.json().get("animations")?.get(a)?;
    let samplers = anim
        .get("samplers")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    // What each sampler animates: the path of the first channel it drives.
    let path_of = |s: usize| {
        anim.get("channels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|c| c.get("sampler").and_then(Value::as_u64) == Some(s as u64))
            .and_then(|c| c.pointer("/target/path")?.as_str())
            .and_then(Path::of)
    };
    let index = |s: &Value, k: &str| s.get(k)?.as_u64().and_then(|i| usize::try_from(i).ok());
    let mut tracks = Vec::with_capacity(samplers.len());
    for (i, s) in samplers.iter().enumerate() {
        let (_, times) = doc.read_floats(index(s, "input")?).ok()?;
        let (n, values) = doc.read_floats(index(s, "output")?).ok()?;
        let interp = match s.get("interpolation").and_then(Value::as_str) {
            Some("STEP") => Interp::Step,
            Some("CUBICSPLINE") => Interp::Cubic,
            _ => Interp::Linear,
        };
        let per = if interp == Interp::Cubic { 3 } else { 1 };
        // A path that isn't a node's own (a pointer) is cleaned as weights are: no turn to keep short.
        let path = path_of(i).unwrap_or(Path::Weights);
        let width = match path {
            Path::Rotation => 4,
            Path::Translation | Path::Scale => 3,
            Path::Weights => (values.len() / (times.len() * per).max(1)).max(1),
        };
        let track = Track {
            node: 0,
            also: Vec::new(),
            path,
            interp,
            times: Times::Single(times),
            values,
            width,
        };
        tracks.push((track, n));
    }
    let (offset, _) = start(tracks.iter().map(|(t, _)| &t.times));
    let kept = tracks
        .into_iter()
        .map(|(t, n)| {
            let Times::Single(was) = &t.times else {
                unreachable!("read as single precision")
            };
            let (was_t, was_v) = (was.clone(), t.values.clone());
            match clean(t, offset) {
                None => Kept::Gone,
                #[allow(clippy::float_cmp, reason = "exactly zero: nothing was taken off")]
                Some((times, values))
                    if offset == 0.0
                        && times.iter().map(|x| x.to_bits()).eq(was_t.iter().map(|x| x.to_bits()))
                        && values.iter().map(|x| x.to_bits()).eq(was_v.iter().map(|x| x.to_bits())) =>
                {
                    Kept::Same
                }
                Some((times, values)) => Kept::New(times, values, n),
            }
        })
        .collect();
    Some((kept, offset))
}

/// A track's keys as glTF wants them, `offset` seconds earlier: finite, times strictly increasing (a key at or
/// before the last moves just after it), rotations unit length and each on the same side as the last. `None` when
/// no key is left. Keys that are already in order are cleaned where they are.
fn clean(t: Track, offset: f64) -> Option<(Vec<f32>, Vec<f32>)> {
    let Track {
        times,
        mut values,
        path,
        interp,
        width,
        ..
    } = t;
    // A cubic key is three values: the tangent in, the value, the tangent out.
    let per = if interp == Interp::Cubic { 3 } else { 1 };
    let stride = width.max(1) * per;
    let keys = times.len().min(values.len() / stride);
    let mut times = times.from(offset);
    times.truncate(keys);
    values.truncate(keys * stride);
    let tidy = times.iter().all(|t| t.is_finite())
        && times.windows(2).all(|w| w[0] < w[1])
        && values.iter().all(|x| x.is_finite());
    if !tidy {
        let mut kept: Vec<f32> = Vec::with_capacity(keys);
        let mut out = Vec::with_capacity(keys * stride);
        for (&t, v) in times.iter().zip(values.chunks_exact(stride)) {
            if !t.is_finite() || v.iter().any(|x| !x.is_finite()) {
                continue;
            }
            kept.push(match kept.last() {
                Some(&p) if t <= p => p.next_up(),
                _ => t,
            });
            out.extend_from_slice(v);
        }
        times = kept;
        values = out;
    }
    if path == Path::Rotation && width == 4 {
        let mut last = [0.0f64, 0.0, 0.0, 1.0];
        for key in values.chunks_exact_mut(stride) {
            // The value: the middle of a cubic key's three.
            let at = if per == 3 { 4 } else { 0 };
            let mut q: [f64; 4] = std::array::from_fn(|i| f64::from(key[at + i]));
            // Already unit length to within half a float's step: normalizing wouldn't change it.
            let unit = (q.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() <= 1e-7;
            if !unit {
                q = normalize(q).map(f64::from);
            }
            let dot: f64 = (0..4).map(|i| q[i] * last[i]).sum();
            if dot < 0.0 {
                q = q.map(|x| -x);
            }
            last = q;
            if !unit || dot < 0.0 {
                for (o, x) in key[at..at + 4].iter_mut().zip(q) {
                    *o = f32_of(x);
                }
            }
            // A quaternion turned to the other side takes its tangents with it.
            if dot < 0.0 && per == 3 {
                for k in (0..4).chain(8..12) {
                    key[k] = -key[k];
                }
            }
        }
    }
    (!times.is_empty()).then_some((times, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(node: usize, path: Path, times: Times, values: Vec<f32>) -> Track {
        Track::linear(node, path, times, values)
    }

    fn times(b: &Builder, a: usize) -> (f64, f64) {
        let (Some(Limits::Float(lo)), Some(Limits::Float(hi))) = (&b.accessors[a].min, &b.accessors[a].max) else {
            panic!("a time accessor has bounds");
        };
        (lo[0], hi[0])
    }

    #[test]
    fn an_animation_starts_at_zero_as_a_whole() {
        // Two nodes whose keys start at 1 s and 1.5 s: both move back by 1 s, so the second still starts half a
        // second after the first.
        let mut b = Builder::plain();
        let mut w = Writer::new(true);
        let tracks = vec![
            track(0, Path::Translation, vec![1.0f64, 2.0].into(), vec![0.0; 6]),
            track(1, Path::Translation, vec![1.5f64, 3.0].into(), vec![0.0; 6]),
        ];
        assert_eq!(w.animation(&mut b, Some("a".into()), None, tracks), Some(1.0));
        w.finish(&mut b);
        let a = &b.animations[0];
        assert_eq!(times(&b, a.samplers[0].input), (0.0, 1.0));
        assert_eq!(times(&b, a.samplers[1].input), (0.5, 2.0));
        assert_eq!(a.extras, Some(json!({ "timeOffset": 1.0 })));
    }

    #[test]
    fn a_negative_start_moves_forward_and_zero_leaves_no_note() {
        let mut b = Builder::plain();
        let mut w = Writer::new(true);
        let early = vec![track(0, Path::Scale, vec![-0.5f32, 0.5].into(), vec![1.0; 6])];
        assert_eq!(w.animation(&mut b, None, Some(json!({ "own": 1 })), early), Some(-0.5));
        let at_zero = vec![track(0, Path::Scale, vec![0.0f32, 0.5].into(), vec![1.0; 6])];
        assert_eq!(w.animation(&mut b, None, None, at_zero), Some(0.0));
        assert_eq!(times(&b, b.animations[0].samplers[0].input), (0.0, 1.0));
        assert_eq!(b.animations[0].extras, Some(json!({ "own": 1, "timeOffset": -0.5 })));
        assert_eq!(b.animations[1].extras, None);
        assert_eq!(b.animations[1].name, None);
    }

    #[test]
    fn a_single_precision_start_is_noted_as_written() {
        let mut b = Builder::plain();
        let mut w = Writer::new(true);
        let t = vec![track(0, Path::Scale, vec![1.2f32, 2.2].into(), vec![1.0; 6])];
        w.animation(&mut b, None, None, t);
        assert_eq!(b.animations[0].extras, Some(json!({ "timeOffset": 1.2 })));
        assert_eq!(times(&b, b.animations[0].samplers[0].input).0, 0.0);
    }

    #[test]
    fn extras_are_left_out_when_not_kept() {
        let mut b = Builder::plain();
        let mut w = Writer::new(false);
        let t = vec![track(0, Path::Scale, vec![2.0f32, 3.0].into(), vec![1.0; 6])];
        assert_eq!(w.animation(&mut b, None, Some(json!({ "own": 1 })), t), Some(2.0));
        assert_eq!(b.animations[0].extras, None);
    }

    #[test]
    fn keys_are_finite_rising_and_rotations_take_the_short_way() {
        let t = Track::linear(
            0,
            Path::Rotation,
            vec![0.0f32, 1.0, 1.0, f32::NAN, 2.0],
            vec![
                0.0,
                0.0,
                0.0,
                2.0, // not unit
                0.0,
                0.0,
                0.0,
                -1.0, // the other side
                0.0,
                0.0,
                1.0,
                0.0, // at the same time as the last
                0.0,
                0.0,
                0.0,
                1.0, // no time
                f32::NAN,
                0.0,
                0.0,
                1.0, // no value
            ],
        );
        let (times, values) = clean(t, 0.0).unwrap();
        assert_eq!(times, [0.0, 1.0, 1.0f32.next_up()]);
        assert_eq!(values, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        // Nothing left is no track.
        let none = Track::linear(0, Path::Scale, vec![f32::NAN], vec![1.0; 3]);
        assert!(clean(none, 0.0).is_none());
    }

    #[test]
    fn a_cubic_rotation_turns_with_its_tangents() {
        let t = Track {
            interp: Interp::Cubic,
            ..Track::linear(
                0,
                Path::Rotation,
                vec![0.0f32, 1.0],
                vec![
                    0.0, 0.0, 0.0, 0.0, /* value */ 0.0, 0.0, 0.0, 1.0, /* out */ 0.5, 0.0, 0.0, 0.0, //
                    0.25, 0.0, 0.0, 0.0, /* value */ 0.0, 0.0, 0.0, -1.0, /* out */ 0.0, 0.5, 0.0, 0.0,
                ],
            )
        };
        let (_, v) = clean(t, 0.0).unwrap();
        assert_eq!(
            &v[12..],
            [-0.25, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 1.0, -0.0, -0.5, -0.0, -0.0]
        );
    }

    #[test]
    fn tracks_with_the_same_times_share_them_and_instances_share_a_sampler() {
        let mut b = Builder::plain();
        let mut w = Writer::new(true);
        let mut first = track(0, Path::Translation, vec![0.0f32, 1.0].into(), vec![0.0; 6]);
        first.also = vec![5];
        let second = track(1, Path::Scale, vec![0.0f64, 1.0].into(), vec![1.0; 6]);
        w.animation(&mut b, None, None, vec![first, second]);
        w.finish(&mut b);
        let a = &b.animations[0];
        assert_eq!(a.samplers[0].input, a.samplers[1].input);
        let nodes: Vec<_> = a.channels.iter().map(|c| (c.node, c.sampler)).collect();
        assert_eq!(nodes, [(0, 0), (5, 0), (1, 1)]);
        assert_eq!(b.views.len(), 1);
    }
}
