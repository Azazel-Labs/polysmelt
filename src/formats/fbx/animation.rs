//! Animation stacks: curves and their keys' interpolation, blended through a stack's layers and sampled into
//! translation, rotation, scale and weights channels.

use std::cell::Cell;
use std::collections::HashMap;

use serde_json::{Value, json};

use super::deform::split_weight;
use super::walker::{Local, rotate};
use super::{Props, Reader, child_attr, choice, flag, floats, ints, name_of, scalar};
use crate::document::{self, Sampler};
use crate::formats::fbx_tree::NodeHandle;
use crate::formats::xform::{IDENTITY, M4, MAX_STEP_DEGREES, Trs, decompose};
use crate::num::f32_of;

/// FBX time units per second.
const KTIME: f64 = 46_186_158_000.0;

/// How a key's value runs on to the next key's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Interp {
    /// Holds its value until the next key.
    Constant,
    /// Takes the next key's value straight after it.
    ConstantNext,
    Linear,
    /// A Bezier curve leaving the key along its slope out and reaching the next along that one's slope in.
    Cubic,
}

impl Interp {
    /// A key's interpolation, from its `KeyAttrFlags`; a key that names none is linear.
    fn of(flags: i64) -> Self {
        if flags & 0x2 != 0 {
            if flags & 0x100 != 0 {
                Self::ConstantNext
            } else {
                Self::Constant
            }
        } else if flags & 0x8 != 0 {
            Self::Cubic
        } else {
            Self::Linear
        }
    }
}

/// How far along its segment an unweighted Bezier handle reaches.
pub(super) const THIRD: f64 = 1.0 / 3.0;

/// The handles of a segment whose keys have no slopes: flat, a third of the way along.
pub(super) const FLAT: [f64; 4] = [0.0, THIRD, 0.0, THIRD];

/// An animation curve: keys in seconds, how each runs on to the next, and for a cubic one, its handles.
pub(super) struct Curve {
    pub(super) times: Vec<f64>,
    pub(super) values: Vec<f64>,
    pub(super) interp: Vec<Interp>,
    /// Per key, the handles of the segment after it: the key's slope out (value per second) and how far along the
    /// segment its handle reaches (a fraction of its length), then the same for the next key's way in.
    pub(super) handles: Vec<[f64; 4]>,
    /// Where the last lookup landed among the keys, tried first since times are mostly asked for in order.
    cursor: Cell<usize>,
}

impl Curve {
    /// A curve running straight from key to key.
    pub(super) fn linear(times: Vec<f64>, values: Vec<f64>) -> Self {
        let n = times.len();
        Self {
            times,
            values,
            interp: vec![Interp::Linear; n],
            handles: vec![FLAT; n],
            cursor: Cell::new(0),
        }
    }

    /// How many keys are at or before `t`.
    fn keys_until(&self, t: f64) -> usize {
        let n = self.times.len();
        let fits = |i: usize| (i == 0 || self.times[i - 1] <= t) && (i == n || self.times[i] > t);
        let c = self.cursor.get();
        if c <= n && fits(c) {
            return c;
        }
        if c < n && fits(c + 1) {
            self.cursor.set(c + 1);
            return c + 1;
        }
        let i = self.times.partition_point(|&k| k <= t);
        self.cursor.set(i);
        i
    }

    fn read(node: NodeHandle<'_>) -> Option<Self> {
        let times: Vec<f64> = child_attr(node, "KeyTime")
            .and_then(ints)?
            .into_iter()
            .map(|t| {
                #[allow(clippy::cast_precision_loss, reason = "FBX time to seconds, as the SDK does")]
                let t = t as f64;
                t / KTIME
            })
            .collect();
        let mut values = floats(child_attr(node, "KeyValueFloat")?)?;
        if times.is_empty() || times.len() != values.len() || times.windows(2).any(|w| w[1] < w[0]) {
            return None;
        }
        // A value that isn't a number takes the one before it (the first number, for the first).
        let mut last = values.iter().copied().find(|v| v.is_finite())?;
        for v in &mut values {
            if v.is_finite() {
                last = *v;
            } else {
                *v = last;
            }
        }
        // Flags and data are shared by runs of keys: run i covers the next `KeyAttrRefCount[i]` keys, with four
        // numbers of `KeyAttrDataFloat`: the slope out, the next key's slope in, and the handles' packed lengths
        // (or a TCB key's tension, continuity and bias).
        let flags = child_attr(node, "KeyAttrFlags").and_then(ints).unwrap_or_default();
        let counts = child_attr(node, "KeyAttrRefCount").and_then(ints).unwrap_or_default();
        let data = child_attr(node, "KeyAttrDataFloat")
            .and_then(floats)
            .unwrap_or_default();
        let n = times.len();
        let mut curve = Self::linear(times, values);
        let mut tcb = vec![None; n];
        let mut k = 0;
        for (run, (&f, &count)) in flags.iter().zip(&counts).enumerate() {
            let d: [f64; 4] = std::array::from_fn(|j| {
                let x = run
                    .checked_mul(4)
                    .and_then(|at| data.get(at + j))
                    .copied()
                    .unwrap_or(0.0);
                if x.is_finite() { x } else { 0.0 }
            });
            for _ in 0..usize::try_from(count).unwrap_or(0) {
                if k == n {
                    break;
                }
                curve.interp[k] = Interp::of(f);
                if curve.interp[k] == Interp::Cubic {
                    if f & 0x200 != 0 {
                        tcb[k] = Some([d[0], d[1], d[2]]);
                    } else {
                        let (out, next_in) = handle_lengths(d[2]);
                        curve.handles[k] = [
                            d[0],
                            if f & 0x0100_0000 != 0 { out } else { THIRD },
                            d[1],
                            if f & 0x0200_0000 != 0 { next_in } else { THIRD },
                        ];
                    }
                }
                k += 1;
            }
            if k == n {
                break;
            }
        }
        if tcb.iter().any(Option::is_some) {
            curve.tcb(&tcb);
        }
        Some(curve)
    }

    /// Slopes for the keys given as tension, continuity and bias, from the slopes of the segments either side, as
    /// a Kochanek-Bartels spline makes them. A key after one is reached along its own TCB slope in, plain (a
    /// Catmull-Rom slope) if it has no TCB values of its own.
    pub(super) fn tcb(&mut self, params: &[Option<[f64; 3]>]) {
        let n = self.times.len();
        let segment = |i: usize| {
            let dt = self.times[i + 1] - self.times[i];
            (dt > 0.0).then(|| (self.values[i + 1] - self.values[i]) / dt)
        };
        let slopes: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let [t, c, b] = params[i].unwrap_or([0.0; 3]);
                let before = if i > 0 { segment(i - 1) } else { None };
                let after = if i + 1 < n { segment(i) } else { None };
                let (before, after) = match (before, after) {
                    (Some(p), Some(q)) => (p, q),
                    (Some(p), None) => (p, p),
                    (None, Some(q)) => (q, q),
                    (None, None) => (0.0, 0.0),
                };
                let into = (1.0 - t) * ((1.0 - c) * (1.0 + b) * before + (1.0 + c) * (1.0 - b) * after) / 2.0;
                let out = (1.0 - t) * ((1.0 + c) * (1.0 + b) * before + (1.0 - c) * (1.0 - b) * after) / 2.0;
                (into, out)
            })
            .collect();
        for i in 0..n {
            if params[i].is_some() {
                let next_in = slopes.get(i + 1).map_or(0.0, |s| s.0);
                self.handles[i] = [slopes[i].1, THIRD, next_in, THIRD];
            }
        }
    }

    /// The value at `t`: the first or last key's outside them, else as the key before `t` runs on to the next.
    pub(super) fn at(&self, t: f64) -> f64 {
        let i = self.keys_until(t);
        if i == 0 {
            return self.values[0];
        }
        if i == self.times.len() {
            return self.values[i - 1];
        }
        self.segment(i - 1, t)
    }

    /// The value at `t` as key `i` runs on to the next, for `t` from that key up to the next.
    fn segment(&self, i: usize, t: f64) -> f64 {
        let (t0, t1, v0, v1) = (self.times[i], self.times[i + 1], self.values[i], self.values[i + 1]);
        let dt = t1 - t0;
        if t <= t0 || dt <= 0.0 {
            return v0;
        }
        let x = (t - t0) / dt;
        match self.interp[i] {
            Interp::Constant => v0,
            Interp::ConstantNext => v1,
            Interp::Linear => v0 + (v1 - v0) * x,
            Interp::Cubic => {
                let [s0, w0, s1, w1] = self.handles[i];
                let (w0, w1) = (w0.clamp(0.0, 1.0), w1.clamp(0.0, 1.0));
                // Handles a third of the way along make time run evenly along the curve; others are solved for.
                #[allow(
                    clippy::float_cmp_const,
                    reason = "a third is set exactly, as a marker, never computed"
                )]
                let u = if w0 == THIRD && w1 == THIRD {
                    x
                } else {
                    along(w0, w1, x)
                };
                bezier(v0, v0 + s0 * w0 * dt, v1 - s1 * w1 * dt, v1, u)
            }
        }
    }

    /// Euler angles in degrees turned the short way where a curve crosses ±180° between two keys. Exporters
    /// bake an angle going past 180 as its other side (179 then -179), which read literally is a whole
    /// turn back in one frame, not two degrees on. A turn over 180° is only a wrap when it dwarfs the
    /// curve's usual step: a coin keyed 0 then 360 still spins.
    pub(super) fn unwrap_degrees(&mut self) {
        let mut steps: Vec<f64> = self
            .values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut by = 0.0;
        for i in 1..self.values.len() {
            let raw = self.values[i] + by;
            let d = raw - self.values[i - 1];
            if d.abs() > 180.0 && d.abs() > usual * 4.0 {
                let turns = (d / 360.0).round() * 360.0;
                by -= turns;
            }
            self.values[i] += by;
        }
    }

    /// The times a linear sampler needs to follow the curve: its keys, a moment either side of each step, and
    /// along a cubic segment enough points (up to 32) that straight lines between them stray from it by no more
    /// than a thousandth of the curve's range where checked.
    fn samples(&self, out: &mut Vec<f64>) {
        out.extend_from_slice(&self.times);
        let (lo, hi) = self
            .values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let tolerance = 1e-3 * (hi - lo).max(1e-6);
        for i in 1..self.times.len() {
            let (t0, t1) = (self.times[i - 1], self.times[i]);
            let dt = t1 - t0;
            if dt <= 0.0 {
                continue;
            }
            match self.interp[i - 1] {
                Interp::Constant => out.push(t1 - (dt / 2.0).min(1e-3)),
                Interp::ConstantNext => out.push(t0 + (dt / 2.0).min(1e-3)),
                Interp::Linear => {}
                Interp::Cubic => {
                    let at = |k: u32, n: u32| t0 + dt * f64::from(k) / f64::from(n);
                    // Inside this segment it's evaluated directly; at its end (or past it, by rounding) looked up.
                    let value = |t: f64| {
                        if t0 <= t && t < t1 {
                            self.segment(i - 1, t)
                        } else {
                            self.at(t)
                        }
                    };
                    let mut ends = [0.0; 33];
                    let mut n = 1;
                    while n < 32 {
                        for k in 0..=n {
                            ends[k as usize] = value(at(k, n));
                        }
                        // Checked a quarter, half and three quarters of the way along each piece: an ease in and
                        // out is on the line at its middle.
                        let strays = (0..n).any(|k| {
                            let (a, b) = (at(k, n), at(k + 1, n));
                            let (va, vb) = (ends[k as usize], ends[k as usize + 1]);
                            [0.25, 0.5, 0.75]
                                .into_iter()
                                .any(|f| (value(a + (b - a) * f) - (va + (vb - va) * f)).abs() > tolerance)
                        });
                        if !strays {
                            break;
                        }
                        n *= 2;
                    }
                    out.extend((1..n).map(|k| at(k, n)));
                }
            }
        }
    }
}

/// How far along a segment's curve time has run a fraction `x` of the segment, for handles reaching `w0` and `w1`
/// of the way along, found by halving the range 48 times.
fn along(w0: f64, w1: f64, x: f64) -> f64 {
    let c = 1.0 - w1;
    // The time at `u`: `bezier(0, w0, c, 1, u)` without its first term, which is zero.
    let before = |u: f64| {
        let v = 1.0 - u;
        3.0 * w0 * v * v * u + 3.0 * c * v * u * u + u * u * u < x
    };
    let (mut lo, mut hi) = (0.0, 1.0);
    // Two halvings at a time: the midpoints of both halves are tried alongside the middle, so they don't wait on it.
    for _ in 0..24 {
        let mid = f64::midpoint(lo, hi);
        let (a, b) = (f64::midpoint(lo, mid), f64::midpoint(mid, hi));
        let (m, below_a, below_b) = (before(mid), before(a), before(b));
        if m {
            (lo, hi) = if below_b { (b, hi) } else { (mid, b) };
        } else {
            (lo, hi) = if below_a { (a, mid) } else { (lo, a) };
        }
    }
    f64::midpoint(lo, hi)
}

/// A key's packed handle lengths, in ten-thousandths of its segment: its own way out in the low 16 bits, the next
/// key's way in in the high 16. Binary files keep the bits in a float; text files write them as a whole number.
pub(super) fn handle_lengths(x: f64) -> (f64, f64) {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "packed bits; `as` saturates bad ones"
    )]
    let bits = if x >= 1.0 {
        x.round() as u32
    } else {
        (x as f32).to_bits()
    };
    let length = |b: u32| {
        let w = f64::from(b & 0xFFFF) / 9999.0;
        // A third written in ten-thousandths, or through a float, is a third.
        if (w - THIRD).abs() < 1e-3 {
            THIRD
        } else {
            w.clamp(0.0, 1.0)
        }
    };
    (length(bits), length(bits >> 16))
}

/// A cubic Bezier through `a`, `b`, `c` and `d` at `u` (0 to 1).
fn bezier(a: f64, b: f64, c: f64, d: f64, u: f64) -> f64 {
    let v = 1.0 - u;
    a * v * v * v + 3.0 * b * v * v * u + 3.0 * c * v * u * u + d * u * u * u
}

/// An animated vector property (`Lcl Translation`, ...): a curve or a fixed value per component.
pub(super) struct Animated {
    pub(super) curves: [Option<Curve>; 3],
    pub(super) fixed: [f64; 3],
}

impl Animated {
    fn at(&self, t: f64) -> [f64; 3] {
        std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.at(t)))
    }

    fn times(&self, out: &mut Vec<f64>) {
        for c in self.curves.iter().flatten() {
            c.samples(out);
        }
    }

    /// Euler rotation keys turned the short way, as Maya's Euler filter turns them. Exporters bake each
    /// frame's rotation as any of the angles that make it: an angle past 180 as its other side (179 then
    /// -179), or a limb raised past straight up as the other set of three (two axes half a turn on, the
    /// middle one mirrored). Read literally, either is a spin in one frame. Where a key jumps more than
    /// a quarter turn from the one before, and far more than the curve's usual step (so a coin keyed 0 then
    /// 360 still spins), it's given the angles making the same rotation that lie nearest the last key.
    pub(super) fn unwrap_rotation(&mut self, order: i32) {
        let Some(times) = self.curves.iter().flatten().next().map(|c| c.times.clone()) else {
            return;
        };
        let shared = self
            .curves
            .iter()
            .flatten()
            .all(|c| c.times.len() == times.len() && c.times.iter().zip(&times).all(|(a, b)| (a - b).abs() < 1e-9));
        if !shared {
            // The three can't be read together key by key: each turns the short way on its own.
            self.curves.iter_mut().flatten().for_each(Curve::unwrap_degrees);
            return;
        }
        let raw: Vec<[f64; 3]> = (0..times.len())
            .map(|i| std::array::from_fn(|k| self.curves[k].as_ref().map_or(self.fixed[k], |c| c.values[i])))
            .collect();
        let apart = |a: &[f64; 3], b: &[f64; 3]| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max);
        let mut steps: Vec<f64> = raw
            .windows(2)
            .map(|w| apart(&w[0], &w[1]))
            .filter(|d| *d > 1e-6)
            .collect();
        if steps.is_empty() {
            return;
        }
        steps.sort_by(f64::total_cmp);
        let usual = steps[steps.len() / 2];
        let mut out = raw.clone();
        // The axis mirrored at each key given the other set of three: its slopes turn the other way.
        let mut mirrored = vec![None; raw.len()];
        for i in 1..raw.len() {
            let (prev, key) = (out[i - 1], raw[i]);
            let jump = apart(&prev, &key);
            if jump <= 90.0 || jump <= usual * 4.0 {
                continue;
            }
            let nearest = |v: [f64; 3]| -> [f64; 3] {
                std::array::from_fn(|k| v[k] + 360.0 * ((prev[k] - v[k]) / 360.0).round())
            };
            let made = rotate(key, order);
            let mut best = nearest(key);
            // The other set of three: whichever axis is the middle one for this order is mirrored.
            for middle in 0..3 {
                let other: [f64; 3] =
                    std::array::from_fn(|k| if k == middle { 180.0 - key[k] } else { key[k] + 180.0 });
                let m = rotate(other, order);
                if (0..3).all(|r| (0..3).all(|c| (m[r][c] - made[r][c]).abs() < 1e-6)) {
                    let other = nearest(other);
                    if apart(&prev, &other) < apart(&prev, &best) {
                        best = other;
                        mirrored[i] = Some(middle);
                    }
                }
            }
            out[i] = best;
        }
        let interp: Vec<Interp> = self
            .curves
            .iter()
            .flatten()
            .next()
            .map(|c| {
                c.interp
                    .iter()
                    .map(|&i| if i == Interp::Cubic { Interp::Linear } else { i })
                    .collect()
            })
            .unwrap_or_default();
        for k in 0..3 {
            let values: Vec<f64> = out.iter().map(|v| v[k]).collect();
            match &mut self.curves[k] {
                Some(c) => {
                    c.values = values;
                    for (i, m) in mirrored.iter().enumerate() {
                        if *m == Some(k) {
                            c.handles[i][0] = -c.handles[i][0];
                            if i > 0 {
                                c.handles[i - 1][2] = -c.handles[i - 1][2];
                            }
                        }
                    }
                }
                // An axis with no curve of its own gets one when its angle had to change, keeping the others'
                // steps.
                None if values.iter().any(|v| (v - self.fixed[k]).abs() > 1e-9) => {
                    let mut c = Curve::linear(times.clone(), values);
                    c.interp.clone_from(&interp);
                    self.curves[k] = Some(c);
                }
                None => {}
            }
        }
    }
}

/// How an animation layer's value meets the value of the layers under it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Blend {
    /// Takes its place, by the layer's weight.
    Override,
    /// Adds to it, by the layer's weight.
    Add,
    /// Multiplies it, by the layer's weight (an additive layer's scale).
    Multiply,
}

/// An animation layer of a stack: its object, how it blends, and its weight (0 to 1).
#[derive(Clone, Copy)]
struct AnimLayer {
    id: i64,
    additive: bool,
    weight: f64,
    /// Whether an additive layer multiplies scale rather than adding to it.
    multiply_scale: bool,
}

/// A property animated through a stack's layers: its value at rest, then each layer's curves over it in order.
pub(super) struct Layered {
    pub(super) rest: [f64; 3],
    pub(super) layers: Vec<(Blend, f64, Animated)>,
}

impl Layered {
    pub(super) fn at(&self, t: f64) -> [f64; 3] {
        let mut v = self.rest;
        for (blend, w, a) in &self.layers {
            let x = a.at(t);
            for k in 0..3 {
                v[k] = match blend {
                    Blend::Override if *w >= 1.0 => x[k],
                    Blend::Override => v[k] + (x[k] - v[k]) * w,
                    Blend::Add => v[k] + x[k] * w,
                    Blend::Multiply if x[k] > 0.0 => v[k] * x[k].powf(*w),
                    Blend::Multiply => v[k] * (1.0 + (x[k] - 1.0) * w),
                };
            }
        }
        v
    }

    fn times(&self, out: &mut Vec<f64>) {
        for (_, _, a) in &self.layers {
            a.times(out);
        }
    }
}

/// Times closer than this are one key: exporters write the same frame to different curves a few
/// ticks apart (two float paths to one time), and a step that small blows up per-frame velocities.
const SAME_TIME: f64 = 1e-6;

/// Sorted times with keys a hair apart merged, each to the last of its run: a step curve keyed at
/// any of them already holds its new value there, where at the first it wouldn't yet.
pub(super) fn timeline(mut times: Vec<f64>) -> Vec<f64> {
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    let mut out: Vec<f64> = Vec::with_capacity(times.len());
    let mut start = f64::NEG_INFINITY;
    for t in times {
        match out.last_mut() {
            Some(last) if t - start < SAME_TIME => *last = t,
            _ => {
                start = t;
                out.push(t);
            }
        }
    }
    out
}

impl Reader<'_, '_, '_> {
    /// Every animation stack as a glTF animation, its curves sampled at their keys and blended through its layers.
    pub(super) fn animations(&mut self) -> Vec<document::Animation> {
        let stacks: Vec<i64> = self
            .objs
            .order
            .iter()
            .copied()
            .filter(|s| self.objs.is("AnimationStack", *s))
            .collect();
        let mut out = Vec::new();
        for stack in stacks {
            let layers = self.layers(stack);
            // Each curve node of the stack, by the layer it's on.
            let mut layer_of: HashMap<i64, usize> = HashMap::new();
            for (k, l) in layers.iter().enumerate() {
                for c in self.objs.children_of(l.id) {
                    if self.objs.is("AnimationCurveNode", c) {
                        layer_of.entry(c).or_insert(k);
                    }
                }
            }
            if layer_of.is_empty() {
                continue;
            }
            let stack_of = (layers.as_slice(), &layer_of);
            let mut channels = Vec::new();
            let mut samplers = Vec::new();
            let mut models: Vec<(i64, usize)> = self.node_of.iter().map(|(&m, &n)| (m, n)).collect();
            models.sort_by_key(|(_, n)| *n);
            for (m, node) in models {
                let local = Local::new(&self.objs.props(m));
                let rest = [local.t, local.r, local.s];
                let tracks: [Option<Layered>; 3] = std::array::from_fn(|k| {
                    let prop = ["Lcl Translation", "Lcl Rotation", "Lcl Scaling"][k];
                    self.layered(
                        m,
                        prop,
                        ["d|X", "d|Y", "d|Z"],
                        rest[k],
                        k == 2,
                        (k == 1).then_some(local.order),
                        stack_of,
                    )
                });
                // A model taking its parent's scale off moves with that scale's animation too.
                let parent_rest = self.parent_scale(&local, m);
                let parent = if local.ignore_parent_scale {
                    self.objs.model_parent(m).filter(|&p| p != m).and_then(|p| {
                        self.layered(
                            p,
                            "Lcl Scaling",
                            ["d|X", "d|Y", "d|Z"],
                            parent_rest,
                            true,
                            None,
                            stack_of,
                        )
                    })
                } else {
                    None
                };
                if tracks.iter().any(Option::is_some) || parent.is_some() {
                    let under = (parent.as_ref(), parent_rest);
                    self.transform_channels(node, &local, &tracks, under, &mut channels, &mut samplers);
                }
            }
            // Morph weights: every channel of a mesh at once, per node drawing it.
            let mut weight_nodes: HashMap<usize, Vec<(usize, Vec<f64>, Layered)>> = HashMap::new();
            let users: Vec<(i64, Vec<(usize, usize)>)> =
                self.channel_users.iter().map(|(c, u)| (*c, u.clone())).collect();
            for (ch, users) in users {
                let percent = scalar(&self.objs.props(ch), "DeformPercent").unwrap_or(0.0);
                let full = self.channel_full.get(&ch).cloned().unwrap_or_else(|| vec![1.0]);
                for (node, first) in users {
                    let rest = [percent, 0.0, 0.0];
                    if let Some(a) = self.layered(
                        ch,
                        "DeformPercent",
                        ["d|DeformPercent", "", ""],
                        rest,
                        false,
                        None,
                        stack_of,
                    ) {
                        weight_nodes.entry(node).or_default().push((first, full.clone(), a));
                    }
                }
            }
            let mut weight_nodes: Vec<_> = weight_nodes.into_iter().collect();
            weight_nodes.sort_by_key(|(n, _)| *n);
            for (node, tracks) in weight_nodes {
                self.weight_channel(node, &tracks, &mut channels, &mut samplers);
            }
            if channels.is_empty() {
                continue;
            }
            let mut anim = document::Animation {
                channels,
                samplers,
                name: self.objs.node(stack).and_then(name_of).map(Into::into),
                extras: None,
            };
            // The stack's own span, which may be wider or narrower than its keys.
            let p = self.objs.props(stack);
            if let (Some(start), Some(stop)) = (scalar(&p, "LocalStart"), scalar(&p, "LocalStop"))
                && stop > start
            {
                anim.extras = Some(json!({ "start": start / KTIME, "stop": stop / KTIME }));
            }
            out.push(anim);
        }
        out
    }

    /// A stack's animation layers that play, bottom first: muted ones are left out, and when any is soloed, all but
    /// those. The bottom one is the base, which always overrides.
    fn layers(&self, stack: i64) -> Vec<AnimLayer> {
        let all: Vec<(i64, Props<'_>)> = self
            .objs
            .children_of(stack)
            .filter(|l| self.objs.is("AnimationLayer", *l))
            .map(|l| (l, self.objs.props(l)))
            .collect();
        let solo = all.iter().any(|(_, p)| flag(p, "Solo", false));
        let mut out: Vec<AnimLayer> = Vec::new();
        for (id, p) in all {
            if flag(&p, "Mute", false) || (solo && !flag(&p, "Solo", false)) {
                continue;
            }
            out.push(AnimLayer {
                id,
                additive: !out.is_empty() && choice(&p, "BlendMode", 0) == 0,
                weight: (scalar(&p, "Weight").unwrap_or(100.0) / 100.0).clamp(0.0, 1.0),
                multiply_scale: choice(&p, "ScaleAccumulationMode", 0) == 0,
            });
        }
        out
    }

    /// Property `prop` of object `owner` as a stack animates it through its layers, from its value at rest; `None`
    /// when no layer of the stack animates it. A rotation (given its order) is turned the short way layer by layer.
    #[allow(clippy::too_many_arguments, reason = "one property's whole description")]
    fn layered(
        &self,
        owner: i64,
        prop: &str,
        components: [&str; 3],
        rest: [f64; 3],
        is_scale: bool,
        order: Option<i32>,
        (layers, layer_of): (&[AnimLayer], &HashMap<i64, usize>),
    ) -> Option<Layered> {
        let mut nodes: Vec<(usize, i64)> = self
            .objs
            .child_at(owner, prop)
            .filter_map(|cn| layer_of.get(&cn).map(|&k| (k, cn)))
            .collect();
        if nodes.is_empty() {
            return None;
        }
        nodes.sort_unstable();
        nodes.dedup_by_key(|(k, _)| *k);
        let mut out = Layered {
            rest,
            layers: Vec::new(),
        };
        for (k, cn) in nodes {
            let l = layers[k];
            let blend = match (l.additive, is_scale && l.multiply_scale) {
                (false, _) => Blend::Override,
                (true, true) => Blend::Multiply,
                (true, false) => Blend::Add,
            };
            // A component the curve node says nothing of stays as it is at rest: the rest value where the
            // layer sets the property, and what changes nothing where it adds to it or multiplies it.
            let unsaid = match blend {
                Blend::Override => rest,
                Blend::Add => [0.0; 3],
                Blend::Multiply => [1.0; 3],
            };
            let Some(mut a) = self.animated(cn, components, unsaid) else {
                continue;
            };
            if let Some(order) = order {
                a.unwrap_rotation(order);
            }
            out.layers.push((blend, l.weight, a));
        }
        (!out.layers.is_empty()).then_some(out)
    }

    /// A curve node's curves for its components (`d|X`, ...; "" for none), with its own values where a component
    /// has no curve, and `unsaid`'s where it gives no value either.
    fn animated(&self, cn: i64, components: [&str; 3], unsaid: [f64; 3]) -> Option<Animated> {
        let p = self.objs.props(cn);
        let curves = components.map(|c| {
            if c.is_empty() {
                return None;
            }
            let curve = self.objs.child_at(cn, c).find(|k| self.objs.is("AnimationCurve", *k))?;
            Curve::read(self.objs.node(curve)?)
        });
        if curves.iter().all(Option::is_none) {
            return None;
        }
        let fixed = std::array::from_fn(|k| scalar(&p, components[k]).unwrap_or(unsaid[k]));
        Some(Animated { curves, fixed })
    }

    /// Translation, rotation and scale channels for a node whose `Lcl` properties are animated (or whose parent's
    /// scale is, when it takes that off): its whole local transform is evaluated at each time (pivots and
    /// pre-rotations included) and taken apart again.
    fn transform_channels(
        &mut self,
        node: usize,
        local: &Local,
        tracks: &[Option<Layered>; 3],
        (parent, parent_rest): (Option<&Layered>, [f64; 3]),
        channels: &mut Vec<document::Channel>,
        samplers: &mut Vec<Sampler>,
    ) {
        let mut times = Vec::new();
        for t in tracks.iter().flatten().chain(parent) {
            t.times(&mut times);
        }
        let mut times = timeline(times);
        // Cut long turns into short steps.
        if let Some(r) = &tracks[1] {
            let mut fine = Vec::with_capacity(times.len());
            for w in times.windows(2) {
                fine.push(w[0]);
                let (a, b) = (r.at(w[0]), r.at(w[1] - 1e-9));
                let turn = (0..3).map(|k| (b[k] - a[k]).abs()).fold(0.0, f64::max);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a small step count"
                )]
                let n = ((turn / MAX_STEP_DEGREES).ceil() as usize).min(360);
                for k in 1..n {
                    #[allow(clippy::cast_precision_loss, reason = "a small step count")]
                    fine.push(w[0] + (w[1] - w[0]) * k as f64 / n as f64);
                }
            }
            fine.extend(times.last());
            times = fine;
        }
        if times.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let finite = |m: M4| m.iter().flatten().all(|c| c.is_finite()).then_some(m);
        let rest_m = finite(local.at(None, None, None, parent_rest)).unwrap_or(IDENTITY);
        let at = |k: usize, t: f64| tracks[k].as_ref().map(|a| a.at(t));
        let mut trs: Vec<Trs> = times
            .iter()
            .map(|&t| {
                let under = parent.map_or(parent_rest, |p| p.at(t));
                decompose(&finite(local.at(at(0, t), at(1, t), at(2, t), under)).unwrap_or(rest_m))
            })
            .collect();
        // Keep each quaternion on the same side as the last, so slerp takes the short way.
        for i in 1..trs.len() {
            let prev = trs[i - 1].r;
            let q = &mut trs[i].r;
            if (0..4).map(|k| prev[k] * q[k]).sum::<f64>() < 0.0 {
                *q = q.map(|c| -c);
            }
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let rest = decompose(&rest_m);
        let paths: [(&str, Vec<f64>, Vec<f64>); 3] = [
            ("translation", trs.iter().flat_map(|x| x.t).collect(), rest.t.to_vec()),
            ("rotation", trs.iter().flat_map(|x| x.r).collect(), rest.r.to_vec()),
            ("scale", trs.iter().flat_map(|x| x.s).collect(), rest.s.to_vec()),
        ];
        for (path, values, rest) in paths {
            let width = rest.len();
            let still = values
                .chunks_exact(width)
                .all(|v| v.iter().zip(&rest).all(|(a, b)| (a - b).abs() < 1e-7));
            if still {
                continue;
            }
            let output = self.b.floats(
                values.into_iter().map(f32_of).collect(),
                if width == 4 { "VEC4" } else { "VEC3" },
                false,
            );
            samplers.push(Sampler {
                input,
                output,
                interpolation: "LINEAR",
            });
            channels.push(document::Channel {
                sampler: samplers.len() - 1,
                node,
                path,
            });
        }
        // An animated node holds TRS, never a matrix.
        if self.nodes[node]
            .as_object_mut()
            .and_then(|n| n.remove("matrix"))
            .is_some()
            && [rest.t.as_slice(), &rest.r, &rest.s]
                .iter()
                .all(|v| v.iter().all(|c| c.is_finite()))
        {
            let n = &mut self.nodes[node];
            n["translation"] = json!(rest.t);
            n["rotation"] = json!(rest.r);
            n["scale"] = json!(rest.s);
        }
    }

    /// A weights channel for a node whose mesh's blend shape channels are animated: per channel, its first target,
    /// the weights its targets are reached at, and its weight in percent.
    fn weight_channel(
        &mut self,
        node: usize,
        tracks: &[(usize, Vec<f64>, Layered)],
        channels: &mut Vec<document::Channel>,
        samplers: &mut Vec<Sampler>,
    ) {
        let Some(mesh) = self.nodes[node]["mesh"].as_u64().and_then(|m| usize::try_from(m).ok()) else {
            return;
        };
        let rest: Vec<f64> = self.meshes[mesh]["weights"]
            .as_array()
            .map(|w| w.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let mut times = Vec::new();
        for (_, _, a) in tracks {
            a.times(&mut times);
        }
        let times = timeline(times);
        if times.is_empty() || rest.is_empty() {
            return;
        }
        let shift = times[0].min(0.0);
        let mut values = Vec::with_capacity(times.len() * rest.len());
        let mut w = rest.clone();
        for &t in &times {
            w.copy_from_slice(&rest);
            for (first, full, a) in tracks {
                if let Some(out) = w.get_mut(*first..first + full.len()) {
                    split_weight(a.at(t)[0] / 100.0, full, out);
                }
            }
            values.extend(w.iter().copied().map(f32_of));
        }
        let input = self
            .b
            .floats(times.iter().map(|&t| f32_of(t - shift)).collect(), "SCALAR", true);
        let output = self.b.floats(values, "SCALAR", false);
        samplers.push(Sampler {
            input,
            output,
            interpolation: "LINEAR",
        });
        channels.push(document::Channel {
            sampler: samplers.len() - 1,
            node,
            path: "weights",
        });
    }
}
