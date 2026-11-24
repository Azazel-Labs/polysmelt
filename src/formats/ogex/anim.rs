//! Animation tracks as glTF animations, one for each clip.
//!
//! A node's transforms are split into two groups, its own and its object's, each driving one
//! glTF node. A group that is translations, then rotations, then scales (at most one rotation
//! animated) is carried over channel by channel: the translations summed, the scales
//! multiplied, the rotation between the fixed ones before and after it. Linear and constant
//! tracks stay linear and stepped; Bézier and TCB tracks become cubic splines when their keys
//! line up. A lone matrix is taken apart at each key. Anything else is sampled.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::math::{mul, quat_axis, quat_mul, quat_unit, set_trs, trs};
use super::{Comp, Ctx, Kind, Morphs, Placed, Rot, Xf, matrix, rotation};
use crate::mesh::FastMap;
use crate::num::f32_of;

/// The most keys a track may have.
const MAX_KEYS: usize = 1 << 20;
/// The most keys a sampled channel may have.
const MAX_SAMPLES: usize = 1 << 16;
/// The most floats all animations may hold.
const MAX_FLOATS: usize = 1 << 22;
/// Samples taken between keys when a channel is sampled.
const SUBSAMPLES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Each key's value held until the next.
    Step,
    Linear,
    /// A cubic Hermite curve between keys (Bézier and TCB values).
    Cubic,
}

/// A track's keys.
struct Curve {
    /// Key times in seconds, strictly increasing.
    times: Vec<f64>,
    /// The incoming and outgoing control points of Bézier times, when they aren't linear.
    time_controls: Option<(Vec<f64>, Vec<f64>)>,
    width: usize,
    values: Vec<f64>,
    mode: Mode,
    /// A cubic curve's incoming and outgoing tangents at each key, per unit of the
    /// interpolation parameter of the segment they're in.
    tin: Vec<f64>,
    tout: Vec<f64>,
}

impl Curve {
    fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether time runs linearly through each segment.
    fn linear_time(&self) -> bool {
        self.time_controls.is_none()
    }

    /// The values at time `t` into `out`.
    fn eval(&self, t: f64, out: &mut [f64]) {
        let (n, w) = (self.len(), self.width);
        let key = |k: usize, out: &mut [f64]| out[..w].copy_from_slice(&self.values[k * w..k * w + w]);
        if n == 1 || t <= self.times[0] {
            return key(0, out);
        }
        if t >= self.times[n - 1] {
            return key(n - 1, out);
        }
        let k = self.times.partition_point(|&x| x <= t).saturating_sub(1).min(n - 2);
        if self.mode == Mode::Step {
            return key(k, out);
        }
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let s = match &self.time_controls {
            None => (t - t0) / (t1 - t0),
            Some((minus, plus)) => solve(t0, plus[k], minus[k + 1], t1, t),
        };
        let (a, b) = (&self.values[k * w..k * w + w], &self.values[(k + 1) * w..(k + 2) * w]);
        if self.mode == Mode::Linear {
            for j in 0..w {
                out[j] = a[j] + (b[j] - a[j]) * s;
            }
            return;
        }
        let (s2, s3) = (s * s, s * s * s);
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        for j in 0..w {
            out[j] = h00 * a[j] + h10 * self.tout[k * w + j] + h01 * b[j] + h11 * self.tin[(k + 1) * w + j];
        }
    }

    /// The outgoing (or incoming) tangent of component `j` at key `k`, per unit of its
    /// segment's parameter; 0 past the ends.
    fn tangent(&self, k: usize, j: usize, outgoing: bool) -> f64 {
        let (n, w) = (self.len(), self.width);
        match self.mode {
            Mode::Step => 0.0,
            Mode::Linear if outgoing => {
                if k + 1 < n {
                    self.values[(k + 1) * w + j] - self.values[k * w + j]
                } else {
                    0.0
                }
            }
            Mode::Linear => {
                if k > 0 {
                    self.values[k * w + j] - self.values[(k - 1) * w + j]
                } else {
                    0.0
                }
            }
            Mode::Cubic if outgoing => self.tout[k * w + j],
            Mode::Cubic => self.tin[k * w + j],
        }
    }
}

/// The parameter `s` at which the Bézier curve through times `t0` and `t1` with control
/// points `c0` and `c1` reaches time `t`: Newton's method from the linear guess, halving the
/// interval when that doesn't converge.
pub(super) fn solve(t0: f64, c0: f64, c1: f64, t1: f64, t: f64) -> f64 {
    let at = |s: f64| {
        let u = 1.0 - s;
        u * u * u * t0 + 3.0 * s * u * u * c0 + 3.0 * s * s * u * c1 + s * s * s * t1
    };
    let slope = |s: f64| {
        let u = 1.0 - s;
        3.0 * u * u * (c0 - t0) + 6.0 * s * u * (c1 - c0) + 3.0 * s * s * (t1 - c1)
    };
    let tolerance = 1e-12 * (t1 - t0).abs().max(1e-300);
    let mut s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    for _ in 0..8 {
        let f = at(s) - t;
        if f.abs() <= tolerance {
            return s;
        }
        let d = slope(s);
        if d.abs() < 1e-300 || !d.is_finite() {
            break;
        }
        s = (s - f / d).clamp(0.0, 1.0);
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = f64::midpoint(lo, hi);
        if at(mid) < t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// The data of the first `Key` of `kind` inside `s`.
fn key<'t>(c: &Ctx<'t>, s: u32, kind: &str) -> Option<&'t [f64]> {
    let t = c.t;
    t.kids(s)
        .filter(|&k| t.ident(k) == "Key")
        .find(|&k| {
            let got = t.prop_str(k, "kind").unwrap_or("value");
            // The specification writes the incoming control's sign as a minus sign.
            got == kind || (kind.starts_with('-') && got.strip_prefix('\u{2212}') == kind.strip_prefix('-'))
        })
        .map(|k| t.values(k).0)
}

/// A track's curve, its values `width` wide; `None` when it can't be read.
fn curve(c: &Ctx<'_>, track: u32, width: usize) -> Option<Curve> {
    let t = c.t;
    let time = t.kid(track, "Time")?;
    let value = t.kid(track, "Value")?;
    let scale = c.metric.time;
    let raw = key(c, time, "value")?;
    let n = raw.len();
    if n == 0 || n > MAX_KEYS {
        return None;
    }
    let times: Vec<f64> = raw.iter().map(|x| x * scale).collect();
    if times.iter().any(|x| !x.is_finite()) || times.windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    let time_controls = if t.prop_str(time, "curve") == Some("bezier") && n > 1 {
        let minus: Vec<f64> = key(c, time, "-control")?.iter().map(|x| x * scale).collect();
        let plus: Vec<f64> = key(c, time, "+control")?.iter().map(|x| x * scale).collect();
        if minus.len() != n || plus.len() != n || minus.iter().chain(&plus).any(|x| !x.is_finite()) {
            return None;
        }
        // Control points a third of the way along make time linear.
        let linear = (0..n - 1).all(|k| {
            let d = times[k + 1] - times[k];
            let eps = 1e-9 * d;
            (plus[k] - (times[k] + d / 3.0)).abs() <= eps && (minus[k + 1] - (times[k + 1] - d / 3.0)).abs() <= eps
        });
        (!linear).then_some((minus, plus))
    } else {
        None
    };
    let values = key(c, value, "value")?.to_vec();
    if values.len() != n * width || values.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mode = match t.prop_str(value, "curve").unwrap_or("linear") {
        _ if n == 1 => Mode::Step,
        "constant" => Mode::Step,
        "linear" => Mode::Linear,
        "bezier" | "tcb" => Mode::Cubic,
        _ => return None,
    };
    let (mut tin, mut tout) = (Vec::new(), Vec::new());
    if mode == Mode::Cubic {
        if t.prop_str(value, "curve") == Some("bezier") {
            let minus = key(c, value, "-control")?;
            let plus = key(c, value, "+control")?;
            if minus.len() != n * width || plus.len() != n * width || minus.iter().chain(plus).any(|x| !x.is_finite()) {
                return None;
            }
            tout = (0..n * width).map(|i| 3.0 * (plus[i] - values[i])).collect();
            tin = (0..n * width).map(|i| 3.0 * (values[i] - minus[i])).collect();
        } else {
            let param = |kind: &str| -> Option<Vec<f64>> {
                match key(c, value, kind) {
                    Some(v) if v.len() == n && v.iter().all(|x| x.is_finite()) => Some(v.to_vec()),
                    Some(_) => None,
                    None => Some(vec![0.0; n]),
                }
            };
            let (tension, continuity, bias) = (param("tension")?, param("continuity")?, param("bias")?);
            tin = vec![0.0; n * width];
            tout = vec![0.0; n * width];
            for k in 0..n {
                let (te, co, bi) = (tension[k], continuity[k], bias[k]);
                for j in 0..width {
                    let v = values[k * width + j];
                    // At the ends the missing neighbor is the key itself.
                    let prev = if k > 0 { values[(k - 1) * width + j] } else { v };
                    let next = if k + 1 < n { values[(k + 1) * width + j] } else { v };
                    let (d0, d1) = (v - prev, next - v);
                    let g = (1.0 - te) * 0.5;
                    tout[k * width + j] = g * ((1.0 + bi) * (1.0 + co) * d0 + (1.0 - bi) * (1.0 - co) * d1);
                    tin[k * width + j] = g * ((1.0 + bi) * (1.0 - co) * d0 + (1.0 - bi) * (1.0 + co) * d1);
                }
            }
        }
    }
    Some(Curve {
        times,
        time_controls,
        width,
        values,
        mode,
        tin,
        tout,
    })
}

/// An animation channel as built: its node and path, interpolation, key times and values (for
/// a cubic spline, each key's incoming tangent, value and outgoing tangent).
struct Out {
    node: usize,
    path: &'static str,
    interpolation: &'static str,
    times: Vec<f64>,
    values: Vec<f64>,
    /// Values in each key (and each tangent).
    width: usize,
}

/// What a channel component is made of: a constant, or a component of a curve.
#[derive(Clone, Copy)]
enum Term {
    Const(f64),
    Curve(usize, usize),
}

/// Times merged from several curves, without repeats.
fn union(curves: &[&Curve]) -> Vec<f64> {
    let mut all: Vec<f64> = curves.iter().flat_map(|c| c.times.iter().copied()).collect();
    all.sort_by(f64::total_cmp);
    all.dedup();
    all
}

/// `keys` with up to [`SUBSAMPLES`] evenly spaced times between each pair, fewer when there'd
/// be more than [`MAX_SAMPLES`]; `None` when even the keys are too many.
fn subsampled(keys: &[f64]) -> Option<Vec<f64>> {
    if keys.len() > MAX_SAMPLES {
        return None;
    }
    let gaps = keys.len().saturating_sub(1).max(1);
    let sub = SUBSAMPLES.min((MAX_SAMPLES - keys.len()) / gaps);
    let mut out = Vec::with_capacity(keys.len() + gaps * sub);
    for (i, &t) in keys.iter().enumerate() {
        out.push(t);
        if let Some(&next) = keys.get(i + 1) {
            for s in 1..=sub {
                #[allow(clippy::cast_precision_loss, reason = "a handful of samples")]
                let f = s as f64 / (sub + 1) as f64;
                out.push(t + (next - t) * f);
            }
        }
    }
    Some(out)
}

/// A channel of components `comps`, each the sum (or product) of its terms over `curves`:
/// exact when the curves allow it (all stepped, all linear, or cubic on the same keys), else
/// sampled. The second value says whether it was sampled.
fn combine(curves: &[&Curve], comps: &[Vec<Term>], product: bool) -> Option<(Out, bool)> {
    let w = comps.len();
    let used: Vec<usize> = (0..curves.len())
        .filter(|&i| {
            comps
                .iter()
                .flatten()
                .any(|t| matches!(t, Term::Curve(k, _) if *k == i))
        })
        .collect();
    if used.is_empty() {
        return None;
    }
    let cs: Vec<&Curve> = used.iter().map(|&i| curves[i]).collect();
    let mut scratch: Vec<Vec<f64>> = curves.iter().map(|c| vec![0.0; c.width]).collect();
    let mut value = |t: f64, out: &mut Vec<f64>| {
        for &i in &used {
            curves[i].eval(t, &mut scratch[i]);
        }
        for terms in comps {
            let mut v = if product { 1.0 } else { 0.0 };
            for term in terms {
                let x = match *term {
                    Term::Const(x) => x,
                    Term::Curve(i, j) => scratch[i][j],
                };
                if product { v *= x } else { v += x }
            }
            out.push(v);
        }
    };
    let one_per = comps
        .iter()
        .all(|terms| terms.iter().filter(|t| matches!(t, Term::Curve(..))).count() <= 1);
    let all_step = cs.iter().all(|c| c.mode == Mode::Step);
    let all_linear = cs.iter().all(|c| c.mode == Mode::Linear && c.linear_time());
    let same_keys = cs
        .iter()
        .all(|c| c.times == cs[0].times && c.linear_time() && c.mode != Mode::Step);
    let out = |interpolation, times, values| Out {
        node: 0,
        path: "",
        interpolation,
        times,
        values,
        width: w,
    };
    if all_step || (all_linear && (one_per || !product)) {
        let times = union(&cs);
        if times.len() > MAX_SAMPLES {
            return None;
        }
        let mut values = Vec::with_capacity(times.len() * w);
        for &t in &times {
            value(t, &mut values);
        }
        let interp = if all_step { "STEP" } else { "LINEAR" };
        return Some((out(interp, times, values), false));
    }
    if same_keys && (one_per || !product) {
        let times = cs[0].times.clone();
        let n = times.len();
        let mut values = Vec::with_capacity(n * w * 3);
        let mut keyed = Vec::with_capacity(w);
        for k in 0..n {
            keyed.clear();
            value(times[k], &mut keyed);
            for side in [false, true] {
                let span = if side {
                    times.get(k + 1).map(|t| t - times[k])
                } else {
                    (k > 0).then(|| times[k] - times[k - 1])
                };
                let mut tangents = Vec::with_capacity(w);
                for terms in comps {
                    let mut d = 0.0;
                    if let Some(span) = span {
                        // A product has one curve term; its constants scale the tangent.
                        let scale: f64 = if product {
                            terms
                                .iter()
                                .map(|t| if let Term::Const(x) = t { *x } else { 1.0 })
                                .product()
                        } else {
                            1.0
                        };
                        for term in terms {
                            if let Term::Curve(i, j) = *term {
                                d += curves[i].tangent(k, j, side) * scale / span;
                            }
                        }
                    }
                    tangents.push(d);
                }
                if side {
                    values.extend_from_slice(&keyed);
                    values.extend_from_slice(&tangents);
                } else {
                    values.extend_from_slice(&tangents);
                }
            }
        }
        return Some((out("CUBICSPLINE", times, values), false));
    }
    let times = subsampled(&union(&cs))?;
    let mut values = Vec::with_capacity(times.len() * w);
    for &t in &times {
        value(t, &mut values);
    }
    Some((out("LINEAR", times, values), true))
}

/// Flips quaternions where needed so each is on the same side as the one before.
fn continuous(values: &mut [f64]) {
    for k in 1..values.len() / 4 {
        let (a, b) = values.split_at_mut(k * 4);
        let prev = &a[(k - 1) * 4..];
        let d: f64 = (0..4).map(|i| prev[i] * b[i]).sum();
        if d < 0.0 {
            for v in &mut b[..4] {
                *v = -*v;
            }
        }
    }
}

/// A rotation channel: the rotation `x` driven by `cv`, between the fixed rotations `qa`
/// (before) and `qb` (after). Exact for stepped tracks, linear angles about a fixed axis (with
/// keys added so no step turns half a circle) and linear quaternions; sampled otherwise.
fn rotation_channel(x: &Xf, cv: &Curve, qa: [f64; 4], qb: [f64; 4], angle: f64) -> Option<(Out, bool)> {
    let Kind::Rotate(r) = x.kind else { return None };
    let w = cv.width;
    let q = |v: &[f64]| quat_unit(quat_mul(qa, quat_mul(rotation(r, v, angle), qb)));
    let mut times = Vec::new();
    let mut values = Vec::new();
    let mut sampled = false;
    let mut interpolation = "LINEAR";
    // The fixed axis of a linear rotation about one.
    let axis = match r {
        Rot::Axis(k) => {
            let mut a = [0.0; 3];
            a[k] = 1.0;
            Some(a)
        }
        Rot::AxisAngle => {
            let axes: Option<Vec<[f64; 3]>> = cv.values.chunks_exact(4).map(super::axis_of).collect();
            axes.filter(|a| a.iter().all(|x| (0..3).all(|i| (x[i] - a[0][i]).abs() < 1e-9)))
                .map(|a| a[0])
        }
        Rot::Quaternion => None,
    };
    if cv.mode == Mode::Step {
        interpolation = "STEP";
        times.clone_from(&cv.times);
        for v in cv.values.chunks_exact(w) {
            values.extend(q(v));
        }
    } else if cv.mode == Mode::Linear && cv.linear_time() && (axis.is_some() || r == Rot::Quaternion) {
        let n = cv.len();
        for k in 0..n {
            let v = &cv.values[k * w..k * w + w];
            times.push(cv.times[k]);
            values.extend(q(v));
            if let (Some(a), Some(next)) = (axis, cv.values.get((k + 1) * w..(k + 2) * w)) {
                // A step of half a turn or more would go the short way round: keys between.
                let d = (next[0] - v[0]) * angle;
                let pieces = (d.abs() / (0.9 * std::f64::consts::PI)).ceil();
                if pieces > 1.0 {
                    if pieces > 65536.0 {
                        return None;
                    }
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "checked above")]
                    let pieces = pieces as usize;
                    for p in 1..pieces {
                        #[allow(clippy::cast_precision_loss, reason = "a bounded count")]
                        let f = p as f64 / pieces as f64;
                        times.push(cv.times[k] + (cv.times[k + 1] - cv.times[k]) * f);
                        let theta = (v[0] + (next[0] - v[0]) * f) * angle;
                        values.extend(quat_unit(quat_mul(qa, quat_mul(quat_axis(a, theta), qb))));
                    }
                }
            }
            if times.len() > MAX_SAMPLES {
                return None;
            }
        }
    } else {
        sampled = true;
        times = subsampled(&cv.times)?;
        let mut v = vec![0.0; w];
        for &t in &times {
            cv.eval(t, &mut v);
            values.extend(q(&v));
        }
    }
    continuous(&mut values);
    Some((
        Out {
            node: 0,
            path: "rotation",
            interpolation,
            times,
            values,
            width: 4,
        },
        sampled,
    ))
}

/// Translation, rotation and scale channels taken from a matrix (or the product of a stack of
/// transforms) at `times`.
fn trs_channels(node: usize, times: &[f64], mut at: impl FnMut(f64) -> super::M4, step: bool) -> Vec<Out> {
    let n = times.len();
    let (mut tv, mut rv, mut sv) = (
        Vec::with_capacity(n * 3),
        Vec::with_capacity(n * 4),
        Vec::with_capacity(n * 3),
    );
    for &t in times {
        let (tr, q, s) = trs(&at(t));
        tv.extend(tr);
        rv.extend(q);
        sv.extend(s);
    }
    continuous(&mut rv);
    let interpolation = if step { "STEP" } else { "LINEAR" };
    [("translation", tv, 3), ("rotation", rv, 4), ("scale", sv, 3)]
        .into_iter()
        .map(|(path, values, width)| Out {
            node,
            path,
            interpolation,
            times: times.to_vec(),
            values,
            width,
        })
        .collect()
}

/// Whether a group of transforms is translations, then rotations, then scales, with at most
/// one rotation animated.
fn split(group: &[&Xf], animated: &dyn Fn(u32) -> bool) -> bool {
    let mut phase = 0;
    let mut rotations = 0;
    for x in group {
        let p = match x.kind {
            Kind::Translate(_) => 0,
            Kind::Rotate(_) => 1,
            Kind::Scale(_) => 2,
            Kind::Matrix(_) => return false,
        };
        if p < phase {
            return false;
        }
        phase = p;
        rotations += usize::from(p == 1 && animated(x.ddl));
    }
    rotations <= 1
}

/// The fixed translation, rotation and scale of a split group, at its unanimated values.
fn split_trs(group: &[&Xf], angle: f64) -> ([f64; 3], [f64; 4], [f64; 3]) {
    let (mut t, mut q, mut s) = ([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]);
    for x in group {
        match x.kind {
            Kind::Translate(Comp::One(k)) => t[k] += x.value[0],
            Kind::Translate(Comp::All) => (0..3).for_each(|k| t[k] += x.value[k]),
            Kind::Scale(Comp::One(k)) => s[k] *= x.value[0],
            Kind::Scale(Comp::All) => (0..3).for_each(|k| s[k] *= x.value[k]),
            Kind::Rotate(r) => q = quat_mul(q, rotation(r, &x.value, angle)),
            Kind::Matrix(_) => {}
        }
    }
    (t, quat_unit(q), s)
}

/// The channels of one group of transforms (driving glTF node `node`) in one clip, its
/// tracks `curves` by transform; how many tracks were sampled.
fn group_channels(
    c: &Ctx<'_>,
    group: &[&Xf],
    curves: &FastMap<u32, Curve>,
    node: usize,
    is_split: bool,
) -> Option<(Vec<Out>, usize)> {
    let angle = c.metric.angle;
    let driven: Vec<&Curve> = group.iter().filter_map(|x| curves.get(&x.ddl)).collect();
    if driven.is_empty() {
        return Some((Vec::new(), 0));
    }
    if is_split {
        let mut outs = Vec::new();
        let mut sampled = 0;
        // Each driven transform's curve, by its place in `driven`.
        let mut list: Vec<&Curve> = Vec::new();
        let mut slot = |x: &Xf| -> Option<usize> {
            let cv = curves.get(&x.ddl)?;
            list.push(cv);
            Some(list.len() - 1)
        };
        let mut tc: [Vec<Term>; 3] = Default::default();
        let mut sc: [Vec<Term>; 3] = Default::default();
        let mut rot: Option<(&Xf, &Curve)> = None;
        let (mut qa, mut qb) = ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
        for x in group {
            let s = slot(x);
            let term = |j: usize| s.map_or(Term::Const(x.value[j]), |i| Term::Curve(i, j));
            match x.kind {
                Kind::Translate(Comp::One(k)) => tc[k].push(term(0)),
                Kind::Translate(Comp::All) => (0..3).for_each(|k| tc[k].push(term(k))),
                Kind::Scale(Comp::One(k)) => sc[k].push(term(0)),
                Kind::Scale(Comp::All) => (0..3).for_each(|k| sc[k].push(term(k))),
                Kind::Rotate(r) => match curves.get(&x.ddl) {
                    Some(cv) => rot = Some((x, cv)),
                    None if rot.is_some() => qb = quat_mul(qb, rotation(r, &x.value, angle)),
                    None => qa = quat_mul(qa, rotation(r, &x.value, angle)),
                },
                Kind::Matrix(_) => return None,
            }
        }
        for (comps, product, path) in [(&tc, false, "translation"), (&sc, true, "scale")] {
            if let Some((mut o, s)) = combine(&list, comps, product) {
                o.node = node;
                o.path = path;
                sampled += usize::from(s);
                outs.push(o);
            }
        }
        if let Some((x, cv)) = rot {
            let (mut o, s) = rotation_channel(x, cv, qa, qb, angle)?;
            o.node = node;
            sampled += usize::from(s);
            outs.push(o);
        }
        return Some((outs, sampled));
    }
    // A lone matrix: taken apart at each key when its values are stepped or linear.
    if let [x] = group
        && let Kind::Matrix(w) = x.kind
        && let Some(cv) = curves.get(&x.ddl)
        && cv.mode != Mode::Cubic
        && cv.linear_time()
    {
        let mut v = vec![0.0; cv.width];
        let outs = trs_channels(
            node,
            &cv.times,
            |t| {
                cv.eval(t, &mut v);
                super::expand(&v, w)
            },
            cv.mode == Mode::Step,
        );
        return Some((outs, 0));
    }
    // Anything else is sampled: the whole stack multiplied out at each time.
    let step = driven.iter().all(|cv| cv.mode == Mode::Step);
    let keys = union(&driven);
    let times = if step {
        (keys.len() <= MAX_SAMPLES).then_some(keys)?
    } else {
        subsampled(&keys)?
    };
    let mut scratch = vec![0.0; 16];
    let outs = trs_channels(
        node,
        &times,
        |t| {
            let mut m = super::math::IDENTITY;
            for x in group {
                let v: &[f64] = match curves.get(&x.ddl) {
                    Some(cv) => {
                        cv.eval(t, &mut scratch);
                        &scratch[..cv.width]
                    }
                    None => &x.value,
                };
                m = mul(&m, &matrix(x.kind, v, angle));
            }
            m
        },
        step,
    );
    Some((outs, driven.len()))
}

/// A clip's channels and its extent.
#[derive(Default)]
struct Clip {
    outs: Vec<Out>,
    begin: Option<f64>,
    end: Option<f64>,
}

/// Builds the animations.
pub(super) fn animations(c: &mut Ctx<'_>) {
    let t = c.t;
    let placed = std::mem::take(&mut c.placed);
    let mut clips: BTreeMap<u32, Clip> = BTreeMap::new();
    for p in &placed {
        node_animations(c, p, &mut clips);
    }
    c.placed = placed;
    let mut floats = 0usize;
    for (index, clip) in clips {
        let info = t
            .kids(super::ROOT)
            .filter(|&k| t.ident(k) == "Clip")
            .find(|&k| t.prop_num(k, "index").map_or(0, super::geometry::index_of) == index);
        let name = info
            .and_then(|k| t.kid(k, "Name"))
            .and_then(|k| t.string(k))
            .map_or_else(|| format!("clip{index}"), str::to_owned);
        let mut extras = Map::new();
        extras.insert("clip".into(), json!(index));
        if let Some(rate) = info.and_then(|k| super::param(t, k, "rate")) {
            extras.insert("rate".into(), json!(rate / c.metric.time));
        }
        if let Some(b) = clip.begin {
            extras.insert("begin".into(), json!(b));
        }
        if let Some(e) = clip.end {
            extras.insert("end".into(), json!(e));
        }
        if let Some(a) = write(c, clip.outs, &mut floats, &mut extras) {
            let mut a = json!({ "name": name, "channels": a.0, "samplers": a.1 });
            if c.keep.extras {
                a["extras"] = Value::Object(extras);
            }
            c.animations.push(a);
        }
    }
}

/// The animation channels of node `p`, added to their clips.
fn node_animations(c: &mut Ctx<'_>, p: &Placed, clips: &mut BTreeMap<u32, Clip>) {
    let t = c.t;
    // Each clip's tracks: (target, track).
    let mut by_clip: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    for a in t.kids(p.ddl).filter(|&a| t.ident(a) == "Animation") {
        let clip = t.prop_num(a, "clip").map_or(0, super::geometry::index_of);
        let entry = clips.entry(clip).or_default();
        let time = c.metric.time;
        if let Some(b) = t.prop_num(a, "begin").filter(|v| v.is_finite()) {
            entry.begin = Some(entry.begin.map_or(b * time, |x: f64| x.min(b * time)));
        }
        if let Some(e) = t.prop_num(a, "end").filter(|v| v.is_finite()) {
            entry.end = Some(entry.end.map_or(e * time, |x: f64| x.max(e * time)));
        }
        let list = by_clip.entry(clip).or_default();
        for tr in t.kids(a).filter(|&k| t.ident(k) == "Track") {
            match t.prop_ref(tr, "target").and_then(|r| t.resolve(tr, r)) {
                Some(target) => list.push((target, tr)),
                None => c.stats.missing_targets += 1,
            }
        }
    }
    if by_clip.values().all(Vec::is_empty) {
        return;
    }
    let animated: Vec<u32> = by_clip.values().flatten().map(|&(target, _)| target).collect();
    let is_animated = |d: u32| animated.contains(&d);
    let groups: [(Vec<&Xf>, usize); 2] = [
        (p.xfs.iter().filter(|x| !x.object).collect(), p.node),
        (p.xfs.iter().filter(|x| x.object).collect(), p.holder),
    ];
    let mut splits = [false; 2];
    for (g, (group, node)) in groups.iter().enumerate() {
        // Object transforms drive the object's node, which a node without an object lacks.
        if g == 1 && *node == p.node {
            continue;
        }
        if group.iter().any(|x| is_animated(x.ddl)) && split(group, &is_animated) {
            splits[g] = true;
            let mut j = std::mem::take(&mut c.nodes[*node]);
            for key in ["translation", "rotation", "scale"] {
                if let Some(o) = j.as_object_mut() {
                    o.remove(key);
                }
            }
            set_trs(&mut j, split_trs(group, c.metric.angle));
            c.nodes[*node] = j;
        }
    }
    let morphs = c.morphs.get(&p.ddl).map(|m| Morphs {
        holder: m.holder,
        targets: m.targets.clone(),
        weights: m.weights.clone(),
    });
    for (clip, tracks) in by_clip {
        let mut curves: FastMap<u32, Curve> = FastMap::default();
        let mut weights: Vec<(u32, Curve)> = Vec::new();
        for (target, tr) in tracks {
            if let Some(x) = p.xfs.iter().find(|x| x.ddl == target) {
                match curve(c, tr, x.kind.width()) {
                    Some(cv) => {
                        curves.insert(target, cv);
                    }
                    None => c.stats.bad_tracks += 1,
                }
            } else if t.ident(target) == "MorphWeight" && t.nodes[target as usize].parent == p.ddl {
                let index = t.prop_num(target, "index").map_or(0, super::geometry::index_of);
                match curve(c, tr, 1) {
                    Some(cv) => weights.push((index, cv)),
                    None => c.stats.bad_tracks += 1,
                }
            } else {
                c.stats.missing_targets += 1;
            }
        }
        let entry = clips.entry(clip).or_default();
        for (g, (group, node)) in groups.iter().enumerate() {
            if g == 1 && *node == p.node {
                continue;
            }
            match group_channels(c, group, &curves, *node, splits[g]) {
                Some((outs, sampled)) => {
                    c.stats.sampled_tracks += sampled;
                    entry.outs.extend(outs);
                }
                None => c.stats.bad_tracks += 1,
            }
        }
        if !weights.is_empty() {
            match &morphs {
                Some(m) => {
                    let mut list = Vec::new();
                    let mut comps = vec![Vec::new(); m.targets.len()];
                    for (index, cv) in &weights {
                        match m.targets.iter().position(|&k| k == *index) {
                            Some(slot) => {
                                list.push(cv);
                                comps[slot] = vec![Term::Curve(list.len() - 1, 0)];
                            }
                            // The base's weight is what the others leave in glTF.
                            None => c.stats.morph_weights_off += 1,
                        }
                    }
                    for (slot, terms) in comps.iter_mut().enumerate() {
                        if terms.is_empty() {
                            terms.push(Term::Const(m.weights[slot]));
                        }
                    }
                    if let Some((mut o, s)) = combine(&list, &comps, false) {
                        o.node = m.holder;
                        o.path = "weights";
                        c.stats.sampled_tracks += usize::from(s);
                        entry.outs.push(o);
                    }
                }
                None => c.stats.missing_targets += weights.len(),
            }
        }
    }
}

/// A clip's channels and samplers; `None` when none is left. Times start at 0 (the clip's
/// earliest key moved there, noted in `extras`), and a channel whose times don't stay
/// strictly increasing in single precision, or that goes past the float budget, is left out.
fn write(
    c: &mut Ctx<'_>,
    mut outs: Vec<Out>,
    floats: &mut usize,
    extras: &mut Map<String, Value>,
) -> Option<(Vec<Value>, Vec<Value>)> {
    let first = outs
        .iter()
        .filter_map(|o| o.times.first().copied())
        .fold(f64::INFINITY, f64::min);
    if first.is_finite() && first < 0.0 {
        for o in &mut outs {
            for t in &mut o.times {
                *t -= first;
            }
        }
        extras.insert("timeOffset".into(), json!(first));
    }
    let mut inputs: FastMap<Vec<u32>, usize> = FastMap::default();
    let mut channels = Vec::new();
    let mut samplers = Vec::new();
    for o in outs {
        let times: Vec<f32> = o.times.iter().map(|&t| f32_of(t)).collect();
        let values: Vec<f32> = o.values.iter().map(|&v| f32_of(v)).collect();
        let ok = !times.is_empty()
            && times.windows(2).all(|w| w[1] > w[0])
            && times.iter().chain(&values).all(|v| v.is_finite());
        let size = times.len() + values.len();
        if !ok || *floats + size > MAX_FLOATS {
            c.stats.bad_tracks += 1;
            continue;
        }
        *floats += size;
        let bits: Vec<u32> = times.iter().map(|t| t.to_bits()).collect();
        let input = if let Some(&a) = inputs.get(&bits) {
            a
        } else {
            let a = super::floats(&mut c.b, times, "SCALAR", true);
            inputs.insert(bits, a);
            a
        };
        let kind = match (o.path, o.width) {
            ("weights", _) => "SCALAR",
            (_, 4) => "VEC4",
            _ => "VEC3",
        };
        let output = super::floats(&mut c.b, values, kind, false);
        channels.push(json!({ "sampler": samplers.len(), "target": { "node": o.node, "path": o.path } }));
        samplers.push(json!({ "input": input, "output": output, "interpolation": o.interpolation }));
    }
    (!channels.is_empty()).then_some((channels, samplers))
}
