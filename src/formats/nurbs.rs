//! NURBS curves drawn as polylines, for the readers that draw curves as lines.

#![cfg_attr(
    not(any(feature = "dxf", feature = "fbx")),
    allow(dead_code, reason = "USD samples its own curves and shares only the budget")
)]

/// The most points drawn curves add to a file, whichever reader draws them; past them, a curve is drawn through its
/// control points (an arc as its chord). Each control point is a few bytes of the file but up to 32 segments, so this
/// bounds what a file of curves costs: 4M points is 48 MiB of positions as `f32`.
pub(super) const MAX_CURVE_POINTS: usize = 1 << 22;
/// The highest degree a curve is drawn at; past it, a curve is its control points.
pub(super) const MAX_DEGREE: usize = 11;
/// The turn each segment of a curve stands for, at most: a sixteenth of a half turn.
const STEP: f64 = std::f64::consts::PI / 16.0;

/// Why a curve couldn't be drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Fail {
    /// Its degree or knots don't fit its control points.
    Unfit,
    /// It takes more points than the budget.
    Budget,
}

/// The points along a NURBS curve of `degree` through its control points `ctrl`, `knots` and
/// `weights` (left out, or not one for each control point, or not all above zero, the curve
/// isn't rational), into `out`: each knot span drawn with a segment for each sixteenth of a
/// half turn its control points turn, up to 32. No knots at all are taken as uniform, clamped
/// to the ends. The points it drew; or, when it's more than `budget` or the degree or knots
/// don't fit, why not.
pub(super) fn curve(
    degree: i64,
    ctrl: &[[f64; 3]],
    weights: &[f64],
    knots: &[f64],
    budget: usize,
    out: &mut Vec<[f64; 3]>,
) -> Result<usize, Fail> {
    let n = ctrl.len();
    let p = usize::try_from(degree)
        .ok()
        .filter(|&p| (1..=MAX_DEGREE).contains(&p) && p < n)
        .ok_or(Fail::Unfit)?;
    let uniform: Vec<f64>;
    let knots = if knots.is_empty() {
        #[allow(clippy::cast_precision_loss, reason = "a knot for each control point, read")]
        let at = |i: usize| i.clamp(p, n) as f64 - p as f64;
        uniform = (0..=n + p).map(at).collect();
        &uniform[..]
    } else {
        knots
    };
    if knots.len() != n + p + 1
        || knots.iter().any(|k| !k.is_finite())
        || knots.windows(2).any(|w| w[1] < w[0])
        || knots[p] >= knots[n]
    {
        return Err(Fail::Unfit);
    }
    let rational = weights.len() == n && weights.iter().all(|&w| w.is_finite() && w > 0.0);
    // The span from knot `k` to the next: the segments it's drawn with, from the angles between
    // the legs of its control points (the curve turns no more than they do).
    let segs = |k: usize| -> usize {
        if p == 1 {
            return 1;
        }
        let leg = |i: usize| -> [f64; 3] { std::array::from_fn(|c| ctrl[i + 1][c] - ctrl[i][c]) };
        let mut turn = 0.0;
        let mut last: Option<[f64; 3]> = None;
        for i in k - p..k {
            let d = leg(i);
            let len = d.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !(len > 0.0 && len.is_finite()) {
                continue;
            }
            let d = d.map(|x| x / len);
            if let Some(a) = last {
                turn += (a[0] * d[0] + a[1] * d[1] + a[2] * d[2]).clamp(-1.0, 1.0).acos();
            }
            last = Some(d);
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 1 to 32"
        )]
        let m = (turn / STEP - 1e-9).ceil().clamp(1.0, 32.0) as usize;
        m.max(1)
    };
    let spans = || (p..n).filter(|&k| knots[k] < knots[k + 1]);
    let total = spans().map(segs).sum::<usize>() + 1;
    if total > budget {
        return Err(Fail::Budget);
    }
    // De Boor's algorithm, in homogeneous coordinates: the point at `t` in the span from knot `k`.
    let at = |k: usize, t: f64| -> [f64; 3] {
        let mut d = [[0.0; 4]; MAX_DEGREE + 1];
        for (j, dj) in d.iter_mut().enumerate().take(p + 1) {
            let i = k - p + j;
            let w = if rational { weights[i] } else { 1.0 };
            let c = ctrl[i];
            *dj = [c[0] * w, c[1] * w, c[2] * w, w];
        }
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k - p + j;
                let span = knots[i + p + 1 - r] - knots[i];
                let a = if span > 0.0 { (t - knots[i]) / span } else { 0.0 };
                for c in 0..4 {
                    d[j][c] = (1.0 - a) * d[j - 1][c] + a * d[j][c];
                }
            }
        }
        let [x, y, z, w] = d[p];
        [x / w, y / w, z / w]
    };
    out.clear();
    let mut end = p;
    for k in spans() {
        let m = segs(k);
        for j in 0..m {
            #[allow(clippy::cast_precision_loss, reason = "at most 32 segments")]
            let t = knots[k] + (knots[k + 1] - knots[k]) * j as f64 / m as f64;
            out.push(at(k, t));
        }
        end = k;
    }
    out.push(at(end, knots[n]));
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nurbs_curves() {
        // A quarter circle, as a rational quadratic: every point is on the circle.
        let ctrl = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let weights = [1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0];
        let mut out = Vec::new();
        assert_eq!(curve(2, &ctrl, &weights, &knots, usize::MAX, &mut out), Ok(9));
        assert_eq!(out.len(), 9);
        assert!(out.iter().all(|p| (p[0].hypot(p[1]) - 1.0).abs() < 1e-12), "{out:?}");
        assert_eq!((out[0], out[8]), (ctrl[0], ctrl[2]));
        // Weights that aren't one for each control point, or aren't all above zero, aren't used.
        for weights in [&weights[..2], &[1.0, 0.0, 1.0], &[1.0, f64::NAN, 1.0]] {
            curve(2, &ctrl, weights, &knots, usize::MAX, &mut out).unwrap();
            assert!(out.iter().any(|p| (p[0].hypot(p[1]) - 1.0).abs() > 1e-3));
        }
        // A cubic with unclamped knots is drawn over its knots from the fourth to the fourth last.
        let line: Vec<[f64; 3]> = (0..6).map(|i| [f64::from(i), 0.0, 0.0]).collect();
        let uniform: Vec<f64> = (0..10).map(f64::from).collect();
        curve(3, &line, &[], &uniform, usize::MAX, &mut out).unwrap();
        let (first, last) = (out[0][0], out[out.len() - 1][0]);
        assert!((first - 1.0).abs() < 1e-12 && (last - 4.0).abs() < 1e-12, "{out:?}");
        // Degrees that don't fit, knots that go back, aren't numbers or leave no span; or more points
        // than the budget.
        let bad = Err(Fail::Unfit);
        assert_eq!(curve(0, &ctrl, &[], &knots, usize::MAX, &mut out), bad);
        assert_eq!(curve(3, &ctrl, &[], &[0.0; 7], usize::MAX, &mut out), bad);
        assert_eq!(curve(12, &[[0.0; 3]; 13], &[], &[], usize::MAX, &mut out), bad);
        assert_eq!(curve(i64::MIN, &ctrl, &[], &knots, usize::MAX, &mut out), bad);
        for knots in [
            [0.0, 0.0, 1.0, 0.5, 1.0, 1.0],
            [0.0, 0.0, 0.0, f64::NAN, 1.0, 1.0],
            [0.0; 6],
        ] {
            assert_eq!(curve(2, &ctrl, &[], &knots, usize::MAX, &mut out), bad, "{knots:?}");
        }
        assert_eq!(curve(2, &ctrl, &weights, &knots, 8, &mut out), Err(Fail::Budget));
    }
}
