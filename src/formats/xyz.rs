//! Point clouds as text: `.xyz`, `.pts` and `.asc` (a point per line), `.ptx` (scans, each with
//! its pose), and ESRI ASCII grids (`.asc` starting with `ncols`).
//!
//! A point per line, its values split by spaces, tabs, commas or semicolons (with semicolons, a
//! comma is a decimal point). The first three values are the position. What the rest are comes
//! from a line of names at the top when there is one (`//X Y Z R G B Nx Ny Nz`), and otherwise
//! from the values themselves: three whole numbers from 0 to 255 (or 65535) are a color, three
//! that make a unit vector are a normal, a lone fourth value is the intensity, and anything else
//! is a custom attribute named after its column (`_COLUMN_6`). Lines with fewer than three
//! numbers (a `.pts` file's point counts, comments) are skipped.
//!
//! A `.ptx` file holds scans, each a grid of points in the scanner's frame with the matrix that
//! places it; the points are placed, the empty ones (all zeros) left out, and each point's scan
//! is the `_SCAN` attribute when there are several. An ESRI grid is a point per cell with a
//! value, at the cell's center.
//!
//! The formats name no units or axes; these clouds come from surveys and scanners, so Z up is
//! assumed and turned Y-up. The units are unknown, and left as they are. Far from the origin, positions are measured from a
//! point near the cloud, which the node's translation carries back.

use serde_json::{Map, Value, json};

use super::Input;
use super::points::{self, Cloud, Column};
use super::stream::{f32_from, f64_from, lines, space};
use crate::document::{Document, Keep};
use crate::num::f32_of;
use crate::source::{Axis, Fact, SourceInfo};
use crate::{Context as _, ErrorKind, Result};

/// Bytes of text parsed per job.
const PIECE: usize = if cfg!(test) { 16 << 10 } else { 2 << 20 };
/// The most values read from a line.
const TOKENS: usize = 64;
/// Lines looked at to tell what the values are.
const SAMPLE: usize = 2000;

pub(crate) fn load(input: Input<'_>, ext: &str, keep: Keep) -> Result<Document> {
    let head = input
        .head(64 * 1024)
        .context(ErrorKind::Corrupt, "reading the point cloud")?;
    let head = head.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&head);
    if ext == "ptx" {
        let bytes = input.read().context(ErrorKind::Corrupt, "reading the PTX file")?;
        return ptx(&bytes, keep);
    }
    let first = head
        .split(|&b| b == b'\n')
        .map(trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default();
    if ext == "asc" && first.len() >= 5 && first[..5].eq_ignore_ascii_case(b"ncols") {
        let bytes = input.read().context(ErrorKind::Corrupt, "reading the grid")?;
        return grid(&bytes, keep);
    }
    let plan = Plan::sample(head, ext)?;
    let part = if let Some(b) = input.bytes() {
        let b = b.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(b);
        parse_all(b, &plan, keep)?
    } else {
        let mut blocks = input.blocks().context(ErrorKind::Corrupt, "reading the point cloud")?;
        let target = keep.thread_count().max(1) * PIECE;
        let mut part = Part::new(&plan);
        let mut batch = Vec::new();
        let mut first = true;
        loop {
            batch.clear();
            while batch.len() < target {
                match blocks
                    .text(lines)
                    .context(ErrorKind::Corrupt, "reading the point cloud")?
                {
                    Some(t) => batch.extend_from_slice(t),
                    None => break,
                }
            }
            if batch.is_empty() {
                break;
            }
            let text = if first {
                batch.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&batch)
            } else {
                &batch[..]
            };
            first = false;
            part.append(parse_all(text, &plan, keep)?, keep)?;
        }
        part
    };
    part.check(keep)?;
    let source = json!({ "columns": plan.described() });
    Ok(part.cloud(&plan, &ext.to_ascii_uppercase(), source))
}

/// `s` without spaces and control characters at its ends.
fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| !space(b)).unwrap_or(s.len());
    let end = s.iter().rposition(|&b| !space(b)).map_or(start, |e| e + 1);
    &s[start..end]
}

/// How a line's values are split.
#[derive(Clone, Copy)]
struct Split {
    /// Semicolons separate values and commas are decimal points.
    semicolons: bool,
}

impl Split {
    #[inline]
    fn sep(self, b: u8) -> bool {
        space(b) || b == b';' || (!self.semicolons && b == b',')
    }

    /// Up to [`TOKENS`] values of `line` into `out`, and how many.
    #[inline]
    fn tokens<'a>(self, line: &'a [u8], out: &mut [&'a [u8]; TOKENS]) -> usize {
        let mut n = 0;
        let mut i = 0;
        while i < line.len() && n < TOKENS {
            while i < line.len() && self.sep(line[i]) {
                i += 1;
            }
            let start = i;
            while i < line.len() && !self.sep(line[i]) {
                i += 1;
            }
            if i > start {
                out[n] = &line[start..i];
                n += 1;
            }
        }
        n
    }

    /// The number `t`, with a decimal comma where commas are that.
    #[inline]
    fn f64(self, t: &[u8]) -> Option<f64> {
        if self.semicolons && t.contains(&b',') {
            let mut buf = [0u8; 64];
            let s = buf.get_mut(..t.len())?;
            for (d, &c) in s.iter_mut().zip(t) {
                *d = if c == b',' { b'.' } else { c };
            }
            return f64_from(s);
        }
        f64_from(t)
    }

    #[inline]
    fn f32(self, t: &[u8]) -> Option<f32> {
        if self.semicolons && t.contains(&b',') {
            return self.f64(t).map(f32_of);
        }
        f32_from(t)
    }
}

/// What a line's values are.
struct Plan {
    split: Split,
    /// The color's columns, alpha's, and the scale that takes them to 0..=1.
    color: Option<([usize; 3], Option<usize>, f32)>,
    normal: Option<[usize; 3]>,
    /// The other values' columns and names.
    columns: Vec<(usize, String)>,
    /// The values of a line read.
    width: usize,
    /// Where positions are measured from, in the file's axes.
    origin: [f64; 3],
}

impl Plan {
    /// The plan for the text that starts `head`.
    fn sample(head: &[u8], ext: &str) -> Result<Plan> {
        let semicolons = head.contains(&b';');
        let split = Split { semicolons };
        let mut toks = [&b""[..]; TOKENS];
        let mut names: Option<Vec<String>> = None;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut counts = [0usize; TOKENS + 1];
        let body = match head.iter().rposition(|&b| b == b'\n') {
            // A head cut short ends mid-line; that line is left out.
            Some(end) if head.len() == 64 * 1024 => &head[..end],
            _ => head,
        };
        for line in body.split(|&b| b == b'\n') {
            if rows.len() >= SAMPLE {
                break;
            }
            let line = trim(line);
            let stripped = line
                .strip_prefix(b"//")
                .or_else(|| line.strip_prefix(b"#"))
                .unwrap_or(line);
            let n = split.tokens(stripped, &mut toks);
            if n == 0 {
                continue;
            }
            let values: Option<Vec<f64>> = toks[..n].iter().map(|t| split.f64(t)).collect();
            match values {
                Some(v) if n >= 3 && stripped.len() == line.len() => {
                    counts[n] += 1;
                    rows.push(v);
                }
                None if rows.is_empty() && names.is_none() && n >= 3 => {
                    names = Some(
                        toks[..n]
                            .iter()
                            .map(|t| String::from_utf8_lossy(t).into_owned())
                            .collect(),
                    );
                }
                _ => {}
            }
        }
        let width = (3..=TOKENS).max_by_key(|&n| (counts[n], n)).filter(|&n| counts[n] > 0);
        let width = width.with_context(ErrorKind::Empty, || {
            format!("no points in the {} file", ext.to_ascii_uppercase())
        })?;
        let rows: Vec<&Vec<f64>> = rows.iter().filter(|r| r.len() == width).collect();
        // Far away, measure from the middle of what was sampled.
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for r in &rows {
            for k in 0..3 {
                if r[k].is_finite() {
                    min[k] = min[k].min(r[k]);
                    max[k] = max[k].max(r[k]);
                }
            }
        }
        let origin = points::origin(min, max);
        let mut plan = Plan {
            split,
            color: None,
            normal: None,
            columns: Vec::new(),
            width,
            origin,
        };
        match names.filter(|n| n.len() == width) {
            Some(n) => plan.named(&n),
            None => plan.guess(&rows, ext == "pts"),
        }
        // Named colors: whole numbers to 255 (or 65535), or floats to 1.
        if let Some((c, a, 0.0)) = plan.color {
            let max = rows
                .iter()
                .flat_map(|r| c.iter().chain(&a).map(|&k| r[k]))
                .filter(|v| v.is_finite())
                .fold(0.0, f64::max);
            let scale = if max > 255.0 {
                1.0 / 65535.0
            } else if max > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            plan.color = Some((c, a, scale));
        }
        Ok(plan)
    }

    /// Columns found by their names.
    fn named(&mut self, names: &[String]) {
        let lower: Vec<String> = names.iter().map(|n| n.trim().to_ascii_lowercase()).collect();
        let find = |keys: &[&str]| lower.iter().position(|n| keys.contains(&n.as_str()));
        let rgb = [
            find(&["r", "red", "diffuse_red"]),
            find(&["g", "green", "diffuse_green"]),
            find(&["b", "blue", "diffuse_blue"]),
        ];
        if let [Some(r), Some(g), Some(b)] = rgb {
            let a = find(&["a", "alpha"]);
            self.color = Some(([r, g, b], a, 0.0));
        }
        let n = [
            find(&["nx", "normal_x", "normalx", "n_x"]),
            find(&["ny", "normal_y", "normaly", "n_y"]),
            find(&["nz", "normal_z", "normalz", "n_z"]),
        ];
        if let [Some(x), Some(y), Some(z)] = n {
            self.normal = Some([x, y, z]);
        }
        for (k, name) in names.iter().enumerate().skip(3) {
            if !self.used(k) {
                let name = if matches!(lower[k].as_str(), "i" | "int") {
                    "INTENSITY"
                } else {
                    name.as_str()
                };
                self.columns.push((k, points::attribute_name(name)));
            }
        }
    }

    fn used(&self, k: usize) -> bool {
        self.color.is_some_and(|(c, a, _)| c.contains(&k) || a == Some(k))
            || self.normal.is_some_and(|n| n.contains(&k))
    }

    /// Columns told apart by their values.
    fn guess(&mut self, rows: &[&Vec<f64>], pts: bool) {
        let whole = |k: usize, max: f64| rows.iter().all(|r| r[k].fract() == 0.0 && (0.0..=max).contains(&r[k]));
        let normal = |k: usize| {
            k + 3 <= self.width
                && !rows.is_empty()
                && rows.iter().all(|r| {
                    let l = r[k] * r[k] + r[k + 1] * r[k + 1] + r[k + 2] * r[k + 2];
                    (l - 1.0).abs() < 0.02
                })
        };
        let color = |k: usize| {
            if k + 3 > self.width || rows.is_empty() {
                return None;
            }
            if (k..k + 3).all(|c| whole(c, 255.0)) {
                Some(1.0 / 255.0)
            } else if (k..k + 3).all(|c| whole(c, 65535.0)) {
                Some(1.0 / 65535.0)
            } else if (k..k + 3).all(|c| rows.iter().all(|r| (0.0..=1.0).contains(&r[c]))) && !normal(k) {
                // Floats from 0 to 1, as some writers have colors.
                Some(1.0)
            } else {
                None
            }
        };
        let mut k = 3;
        while k < self.width {
            // The fourth value is intensity in a `.pts` file, and in others laid out alike (a
            // value and then a color).
            if k == 3 && (pts || (self.width == 7 && color(4).is_some())) {
                self.columns.push((3, "INTENSITY".into()));
                k = 4;
                continue;
            }
            if self.color.is_none()
                && let Some(s) = color(k)
            {
                self.color = Some(([k, k + 1, k + 2], None, s));
                k += 3;
            } else if self.normal.is_none() && normal(k) {
                self.normal = Some([k, k + 1, k + 2]);
                k += 3;
            } else {
                let name = if k == 3 && self.width == 4 {
                    "INTENSITY".to_string()
                } else {
                    format!("COLUMN_{k}")
                };
                self.columns.push((k, name));
                k += 1;
            }
        }
    }

    /// The columns, for the asset's extras.
    fn described(&self) -> Value {
        let mut names: Vec<String> = (0..self.width).map(|k| format!("_COLUMN_{k}")).collect();
        for (k, n) in ["x", "y", "z"].iter().enumerate() {
            names[k] = (*n).into();
        }
        if let Some((c, a, _)) = self.color {
            for (k, n) in c.iter().zip(["red", "green", "blue"]) {
                names[*k] = n.into();
            }
            if let Some(a) = a {
                names[a] = "alpha".into();
            }
        }
        if let Some(nm) = self.normal {
            for (k, n) in nm.iter().zip(["normal x", "normal y", "normal z"]) {
                names[*k] = n.into();
            }
        }
        for (k, n) in &self.columns {
            names[*k] = format!("_{n}");
        }
        json!(names)
    }
}

/// Points parsed.
struct Part {
    pos: Vec<[f32; 3]>,
    colors: Vec<[u16; 4]>,
    normals: Vec<[f32; 3]>,
    columns: Vec<Vec<f32>>,
}

impl Part {
    fn new(plan: &Plan) -> Part {
        Part {
            pos: Vec::new(),
            colors: Vec::new(),
            normals: Vec::new(),
            columns: vec![Vec::new(); plan.columns.len()],
        }
    }

    fn append(&mut self, mut other: Part, keep: Keep) -> Result<()> {
        other.check(keep)?;
        points::reserve(&mut self.pos, other.pos.len(), "text")?;
        self.pos.append(&mut other.pos);
        self.colors.append(&mut other.colors);
        self.normals.append(&mut other.normals);
        for (a, b) in self.columns.iter_mut().zip(&mut other.columns) {
            a.append(b);
        }
        self.check(keep)
    }

    fn check(&self, keep: Keep) -> Result<()> {
        points::check_count(u64::try_from(self.pos.len()).unwrap_or(u64::MAX), keep.points(), "text")
    }

    fn cloud(self, plan: &Plan, format: &str, source: Value) -> Document {
        let mut cloud = Cloud {
            positions: self.pos,
            colors: self.colors,
            normals: self.normals,
            origin: points::z_up(plan.origin),
            ..Cloud::default()
        };
        for ((_, name), values) in plan.columns.iter().zip(self.columns) {
            cloud.columns.push(Column::new(name, values));
        }
        document(cloud, format, source)
    }
}

/// `text`, in pieces on up to [`Keep::threads`] threads.
fn parse_all(text: &[u8], plan: &Plan, keep: Keep) -> Result<Part> {
    let mut cuts = vec![0];
    while let Some(&last) = cuts.last() {
        if last >= text.len() {
            break;
        }
        let want = last + PIECE;
        let cut = if want >= text.len() {
            text.len()
        } else {
            text[want..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(text.len(), |k| want + k + 1)
        };
        cuts.push(cut);
    }
    let pieces: Vec<&[u8]> = cuts.windows(2).map(|w| &text[w[0]..w[1]]).collect();
    // A piece holds at most a point per two bytes; refusing too many is up to the caller.
    let limit = keep.points().saturating_add(1);
    let parts = points::map(pieces, keep.thread_count(), |t| parse(t, plan, None, limit));
    let mut all = Part::new(plan);
    for p in parts {
        all.append(p, keep)?;
    }
    Ok(all)
}

/// A scan's pose: its matrix, which takes a point in the scanner's frame (as a row, `[x y z 1]`
/// times the matrix) to the file's frame.
type Pose = [[f64; 4]; 4];

/// The points in `text`, up to `limit` of them; `pose` places them (and leaves out the empty
/// ones) for a PTX scan.
fn parse(text: &[u8], plan: &Plan, pose: Option<&Pose>, limit: usize) -> Part {
    let mut part = Part::new(plan);
    let mut toks = [&b""[..]; TOKENS];
    let split = plan.split;
    let o = plan.origin;
    for line in text.split(|&b| b == b'\n') {
        if part.pos.len() >= limit {
            break;
        }
        let n = split.tokens(line, &mut toks);
        if n < 3 {
            continue;
        }
        let Some(x) = split.f64(toks[0]) else { continue };
        let y = split.f64(toks[1]).unwrap_or(f64::NAN);
        let z = split.f64(toks[2]).unwrap_or(f64::NAN);
        let mut p = [x, y, z];
        if let Some(m) = pose {
            if p.iter().all(|&c| c == 0.0) {
                continue;
            }
            p = std::array::from_fn(|j| x * m[0][j] + y * m[1][j] + z * m[2][j] + m[3][j]);
        }
        part.pos
            .push([f32_of(p[0] - o[0]), f32_of(p[2] - o[2]), f32_of(o[1] - p[1])]);
        let get = |k: usize| {
            if k < n {
                split.f32(toks[k]).unwrap_or(f32::NAN)
            } else {
                f32::NAN
            }
        };
        if let Some((c, a, s)) = plan.color {
            let rgb = c.map(&get);
            let alpha = a.map_or(1.0, |a| get(a) * s);
            let rgb = rgb.map(|v| if v.is_finite() { v * s } else { 0.0 });
            part.colors
                .push(points::linear_f(rgb, if alpha.is_finite() { alpha } else { 1.0 }));
        }
        if let Some(nm) = plan.normal {
            let v = nm.map(&get);
            let v = match pose {
                Some(m) => {
                    let v = v.map(f64::from);
                    std::array::from_fn(|j| f32_of(v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]))
                }
                None => v,
            };
            part.normals.push([v[0], v[2], -v[1]]);
        }
        for (out, (k, _)) in part.columns.iter_mut().zip(&plan.columns) {
            out.push(get(*k));
        }
    }
    part
}

/// The next line of `b` from `at`, trimmed, and where the one after starts.
fn line_at(b: &[u8], at: usize) -> Option<(&[u8], usize)> {
    if at >= b.len() {
        return None;
    }
    let end = b[at..].iter().position(|&c| c == b'\n').map_or(b.len(), |n| at + n);
    Some((trim(&b[at..end]), end + 1))
}

/// `n` numbers on the next line that has any.
fn numbers<const N: usize>(b: &[u8], at: &mut usize) -> Option<[f64; N]> {
    loop {
        let (line, next) = line_at(b, *at)?;
        *at = next;
        if line.is_empty() {
            continue;
        }
        let mut it = line.split(|&c| space(c)).filter(|t| !t.is_empty());
        let mut out = [0.0; N];
        for v in &mut out {
            *v = f64_from(it.next()?)?;
        }
        return Some(out);
    }
}

/// The scans of a PTX file.
fn ptx(bytes: &[u8], keep: Keep) -> Result<Document> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut at = 0;
    let mut scans = Vec::new();
    let mut all: Option<Part> = None;
    let mut plan: Option<Plan> = None;
    let mut scan_ids: Vec<f32> = Vec::new();
    loop {
        // Skip blank lines between scans.
        while let Some((line, next)) = line_at(bytes, at) {
            if !line.is_empty() {
                break;
            }
            at = next;
        }
        if at >= bytes.len() {
            break;
        }
        let head = (|| {
            let [cols] = numbers::<1>(bytes, &mut at)?;
            let [rows] = numbers::<1>(bytes, &mut at)?;
            let position = numbers::<3>(bytes, &mut at)?;
            let axes = [
                numbers::<3>(bytes, &mut at)?,
                numbers::<3>(bytes, &mut at)?,
                numbers::<3>(bytes, &mut at)?,
            ];
            let m = [
                numbers::<4>(bytes, &mut at)?,
                numbers::<4>(bytes, &mut at)?,
                numbers::<4>(bytes, &mut at)?,
                numbers::<4>(bytes, &mut at)?,
            ];
            Some((cols, rows, position, axes, m))
        })();
        let (cols, rows, position, axes, m) = head.context(ErrorKind::Corrupt, "a PTX scan's header is broken")?;
        ensure!(
            cols.fract() == 0.0 && rows.fract() == 0.0 && cols >= 0.0 && rows >= 0.0,
            Corrupt,
            "a PTX scan's size is broken"
        );
        ensure!(
            m.iter().flatten().all(|v| v.is_finite()),
            Corrupt,
            "a PTX scan's matrix is broken"
        );
        let count = cols * rows;
        #[allow(clippy::cast_precision_loss, reason = "a limit, rounded")]
        let most = keep.points() as f64;
        ensure!(
            count.le(&most),
            LimitExceeded,
            "{count} points in a PTX scan, more than the {} allowed (Keep::max_points)",
            keep.points()
        );
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked whole and in range"
        )]
        let count = count as usize;
        // The scan's lines.
        let start = at.min(bytes.len());
        let mut end = start;
        let mut found = 0;
        while found < count && end < bytes.len() {
            match bytes[end..].iter().position(|&c| c == b'\n') {
                Some(k) => end += k + 1,
                None => end = bytes.len(),
            }
            found += 1;
        }
        at = end;
        let body = &bytes[start..end];
        let plan = plan.get_or_insert_with(|| {
            let mut p =
                Plan::sample(body.get(..body.len().min(64 * 1024)).unwrap_or_default(), "ptx").unwrap_or(Plan {
                    split: Split { semicolons: false },
                    color: None,
                    normal: None,
                    columns: Vec::new(),
                    width: 4,
                    origin: [0.0; 3],
                });
            // The origin is near the first scanner; the intensity, the fourth value, from 0 to 1.
            p.split = Split { semicolons: false };
            p.origin = points::origin([m[3][0], m[3][1], m[3][2]], [m[3][0], m[3][1], m[3][2]]);
            p.columns.retain(|(k, _)| *k != 3);
            p.columns.insert(0, (3, "INTENSITY".into()));
            if p.color.is_some_and(|(c, _, _)| c[0] == 3) {
                p.color = None;
            }
            p
        });
        let pieces = split_lines(body);
        let limit = keep.points().saturating_add(1);
        let parts = points::map(pieces, keep.thread_count(), |t| parse(t, plan, Some(&m), limit));
        let index = scans.len();
        scans.push(json!({
            "columns": cols,
            "rows": rows,
            "points": count,
            "scannerPosition": position,
            "scannerAxes": axes,
            "matrix": m,
        }));
        for p in parts {
            let n = p.pos.len();
            if let Some(a) = &mut all {
                a.append(p, keep)?;
            } else {
                p.check(keep)?;
                all = Some(p);
            }
            #[allow(clippy::cast_precision_loss, reason = "scan counts are small")]
            scan_ids.extend(std::iter::repeat_n(index as f32, n));
        }
    }
    let plan = plan.context(ErrorKind::Empty, "no scans in the PTX file")?;
    let mut all = all.unwrap_or_else(|| Part::new(&plan));
    let mut plan = plan;
    if scans.len() > 1 {
        plan.columns.push((usize::MAX, "SCAN".into()));
        all.columns.push(scan_ids);
    }
    let source = json!({ "scans": scans, "columns": plan.described_ptx() });
    Ok(all.cloud(&plan, "PTX", source))
}

impl Plan {
    fn described_ptx(&self) -> Value {
        let mut p = Plan {
            split: self.split,
            color: self.color,
            normal: self.normal,
            columns: self.columns.iter().filter(|(k, _)| *k < self.width).cloned().collect(),
            width: self.width,
            origin: self.origin,
        };
        p.width = p.width.max(4);
        p.described()
    }
}

/// `text` cut into pieces of whole lines.
fn split_lines(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let want = start + PIECE;
        let cut = if want >= text.len() {
            text.len()
        } else {
            text[want..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(text.len(), |k| want + k + 1)
        };
        out.push(&text[start..cut]);
        start = cut;
    }
    out
}

/// An ESRI ASCII grid: a point at each cell's center that has a value.
fn grid(bytes: &[u8], keep: Keep) -> Result<Document> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut at = 0;
    let mut keys: Map<String, Value> = Map::new();
    let (mut cols, mut rows, mut x0, mut y0, mut dx, mut dy, mut nodata) = (None, None, None, None, None, None, None);
    let mut center = false;
    // The header: a key and a value a line, until the first line of numbers.
    while let Some((line, next)) = line_at(bytes, at) {
        let mut it = line.split(|&c| space(c)).filter(|t| !t.is_empty());
        let Some(key) = it.next() else {
            at = next;
            continue;
        };
        if f64_from(key).is_some() {
            break;
        }
        ensure!(keys.len() < 32, Corrupt, "the ESRI grid's header is broken");
        let key = String::from_utf8_lossy(key).to_ascii_lowercase();
        let v = it
            .next()
            .and_then(f64_from)
            .with_context(ErrorKind::Corrupt, || format!("the ESRI grid's {key} has no value"))?;
        keys.insert(key.clone(), json!(v));
        match key.as_str() {
            "ncols" => cols = Some(v),
            "nrows" => rows = Some(v),
            "xllcorner" => x0 = Some(v),
            "yllcorner" => y0 = Some(v),
            "xllcenter" => {
                x0 = Some(v);
                center = true;
            }
            "yllcenter" => {
                y0 = Some(v);
                center = true;
            }
            "cellsize" => {
                dx = Some(v);
                dy = Some(v);
            }
            "dx" | "xcellsize" => dx = Some(v),
            "dy" | "ycellsize" => dy = Some(v),
            "nodata_value" => nodata = Some(v),
            _ => {}
        }
        at = next;
    }
    let (Some(cols), Some(rows), Some(dx), Some(dy)) = (cols, rows, dx, dy) else {
        bail!(Corrupt, "the ESRI grid's header lacks its size or cell size");
    };
    ensure!(
        cols.fract() == 0.0 && rows.fract() == 0.0 && cols >= 1.0 && rows >= 1.0 && dx.is_finite() && dy.is_finite(),
        Corrupt,
        "the ESRI grid's size is broken"
    );
    let count = cols * rows;
    #[allow(clippy::cast_precision_loss, reason = "a limit, rounded")]
    let most = keep.points() as f64;
    ensure!(
        count.le(&most),
        LimitExceeded,
        "{count} cells in the ESRI grid, more than the {} allowed (Keep::max_points)",
        keep.points()
    );
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked whole and in range"
    )]
    let (cols, count) = (cols as usize, count as usize);
    let (x0, y0) = (x0.unwrap_or(0.0), y0.unwrap_or(0.0));
    let (cx, cy) = if center {
        (x0, y0)
    } else {
        (x0 + dx / 2.0, y0 + dy / 2.0)
    };
    // North row first.
    let top = cy + dy * (rows - 1.0);
    let mut z = Vec::new();
    points::reserve(&mut z, count.min(bytes.len() / 2 + 1), "grid")?;
    for t in bytes[at.min(bytes.len())..]
        .split(|&c| space(c))
        .filter(|t| !t.is_empty())
    {
        if z.len() >= count {
            break;
        }
        z.push(f64_from(t).unwrap_or(f64::NAN));
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &z {
        if v.is_finite() && Some(v) != nodata {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    #[allow(clippy::cast_precision_loss, reason = "cell counts are within the point limit")]
    let far = [cx + dx * (cols - 1) as f64, top, hi];
    let origin = points::origin([cx, cy, lo], far);
    let mut cloud = Cloud {
        origin: points::z_up(origin),
        ..Cloud::default()
    };
    points::reserve(&mut cloud.positions, z.len(), "grid")?;
    for (i, &v) in z.iter().enumerate() {
        if !v.is_finite() || Some(v) == nodata {
            continue;
        }
        #[allow(clippy::cast_precision_loss, reason = "cell counts are within the point limit")]
        let (c, r) = ((i % cols) as f64, (i / cols) as f64);
        let p = [cx + dx * c, top - dy * r, v];
        cloud.positions.push([
            f32_of(p[0] - origin[0]),
            f32_of(p[2] - origin[2]),
            f32_of(origin[1] - p[1]),
        ]);
    }
    let source = json!({ "header": keys, "cellsRead": z.len() });
    Ok(document(cloud, "ESRI ASCII grid", source))
}

/// The document of `cloud`, its source info `format`, with Z up assumed (and turned Y-up) and its
/// units unknown, over `source`'s details.
fn document(cloud: Cloud, format: &str, source: Value) -> Document {
    let mut info = SourceInfo {
        format: Some(format.into()),
        up_axis: Fact::Assumed(Axis::PosZ),
        ..SourceInfo::default()
    };
    info.converted.up_axis = true;
    let Value::Object(source) = source else { unreachable!() };
    cloud.document(&info, source, Map::new())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::format_collect,
    reason = "test data is small"
)]
mod tests {
    use super::*;

    fn load_text(text: &str, ext: &str, keep: Keep) -> Result<Document> {
        let name = format!("cloud.{ext}");
        load(Input::memory(text.as_bytes(), &name), ext, keep)
    }

    fn attrs(doc: &Document) -> Vec<String> {
        let mut v: Vec<String> = doc.json()["meshes"][0]["primitives"][0]["attributes"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        v.sort();
        v
    }

    fn values(doc: &Document, name: &str) -> Vec<f32> {
        let a = doc.json()["meshes"][0]["primitives"][0]["attributes"][name]
            .as_u64()
            .unwrap() as usize;
        let acc = &doc.json()["accessors"][a];
        let view = &doc.json()["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let buf = doc.buffer(view["buffer"].as_u64().unwrap_or(0) as usize).unwrap();
        let off = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let b = &buf[off..off + len];
        match acc["componentType"].as_u64() {
            Some(5126) => b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Some(5123) => b
                .chunks(2)
                .map(|c| f32::from(u16::from_le_bytes(c.try_into().unwrap())) / 65535.0)
                .collect(),
            t => panic!("{t:?}"),
        }
    }

    #[test]
    fn plain_xyz_turns_y_up() {
        let doc = load_text("1 2 3\n4 5 6\n\n# a comment\n7,8,9\n", "xyz", Keep::ALL).unwrap();
        assert_eq!(attrs(&doc), ["POSITION"]);
        assert_eq!(
            values(&doc, "POSITION"),
            [1.0, 3.0, -2.0, 4.0, 6.0, -5.0, 7.0, 9.0, -8.0]
        );
    }

    #[test]
    fn pts_with_counts_intensity_and_colors() {
        let text = "3\n0 0 0 -1200 255 0 0\n1 0 0 300 0 255 0\n0 1 0 7 0 0 255\n1\n2 2 2 5 10 10 10\n";
        let doc = load_text(text, "pts", Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 4);
        assert_eq!(attrs(&doc), ["COLOR_0", "POSITION", "_INTENSITY"]);
        assert_eq!(values(&doc, "_INTENSITY"), [-1200.0, 300.0, 7.0, 5.0]);
        assert_eq!(&values(&doc, "COLOR_0")[..8], &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn colors_and_normals_are_told_apart() {
        let text = "0 0 0 0 0 1 10 20 30\n1 0 0 0.6 0.8 0 40 50 60\n0 1 0 1 0 0 70 80 255\n";
        let doc = load_text(text, "xyz", Keep::ALL).unwrap();
        assert_eq!(attrs(&doc), ["COLOR_0", "NORMAL", "POSITION"]);
        let n = values(&doc, "NORMAL");
        assert_eq!(&n[..6], &[0.0, 1.0, 0.0, 0.6, 0.0, -0.8]);
        let cols = &doc.json()["asset"]["extras"]["source"]["columns"];
        assert_eq!(cols[3], "normal x");
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("XYZ"));
        assert_eq!(
            (info.meters_per_unit, info.up_axis),
            (Fact::Unknown, Fact::Assumed(Axis::PosZ))
        );
        assert!(info.converted.up_axis && !info.converted.units);
        assert_eq!(cols[8], "blue");
    }

    #[test]
    fn named_columns_and_decimal_commas() {
        let text = "//X;Y;Z;R;G;B;Intensity;Return\n1,5;2;3;1;0,5;0;0,25;1\n2,5;2;3;0;0;1;0,75;2\n";
        let doc = load_text(text, "asc", Keep::ALL).unwrap();
        assert_eq!(attrs(&doc), ["COLOR_0", "POSITION", "_INTENSITY", "_RETURN"]);
        assert_eq!(&values(&doc, "POSITION")[..3], &[1.5, 3.0, -2.0]);
        assert_eq!(values(&doc, "_INTENSITY"), [0.25, 0.75]);
        // Floats to 1: green 0.5 is sRGB, about 0.214 linear.
        let c = values(&doc, "COLOR_0");
        assert!((c[1] - 0.214).abs() < 0.001, "{c:?}");
    }

    #[test]
    fn far_points_keep_their_detail() {
        let text = "637012.24 849028.31 431.66\n637012.25 849028.33 431.67\n";
        let doc = load_text(text, "xyz", Keep::ALL).unwrap();
        assert_eq!(
            doc.json()["nodes"][0]["translation"],
            json!([637_012.0, 0.0, -849_028.0])
        );
        let p = values(&doc, "POSITION");
        assert!((p[3] - p[0] - 0.01).abs() < 1e-5);
        assert!((p[5] - p[2] + 0.02).abs() < 1e-5);
    }

    #[test]
    fn ptx_scans_are_placed() {
        // Two 2x1 scans; the second moved 10 along x and turned a quarter about z.
        let scan = |m: &str| format!("2\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n{m}0 0 0 0.5\n1 2 3 0.25\n");
        let text = scan("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n") + &scan("0 1 0 0\n-1 0 0 0\n0 0 1 0\n10 0 0 1\n");
        let doc = load_text(&text, "ptx", Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 2);
        // (1, 2, 3) turned to (-2, 1, 3) and moved to (8, 1, 3); Y-up, (8, 3, -1).
        assert_eq!(values(&doc, "POSITION"), [1.0, 3.0, -2.0, 8.0, 3.0, -1.0]);
        assert_eq!(values(&doc, "_SCAN"), [0.0, 1.0]);
        assert_eq!(doc.json()["meshes"][0]["extras"]["constant"]["_INTENSITY"], 0.25);
        assert_eq!(
            doc.json()["asset"]["extras"]["source"]["scans"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn esri_grids_are_points_at_cell_centers() {
        let text =
            "ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\ncellsize 10\nNODATA_value -9999\n1 2 3\n4 -9999 6\n";
        let doc = load_text(text, "asc", Keep::ALL).unwrap();
        assert_eq!(doc.facts["points"], 5);
        let p = values(&doc, "POSITION");
        // The first cell is the north-west one: (105, 215, 1), Y-up (105, 1, -215).
        assert_eq!(&p[..3], &[105.0, 1.0, -215.0]);
        assert_eq!(&p[12..15], &[125.0, 6.0, -205.0]);
    }

    #[test]
    fn threads_read_the_same_text() {
        let text: String = (0..20_000)
            .map(|i| format!("{} {} 1 {} {} 7 {}\n", i, i * 2, i % 256, (i / 3) % 256, i % 5))
            .collect();
        assert!(text.len() > 4 * PIECE);
        let one = load_text(&text, "xyz", Keep::ALL).unwrap();
        let many = load_text(&text, "xyz", Keep::ALL.with_threads(4)).unwrap();
        crate::keep::NO_THREADS.set(true);
        let none = load_text(&text, "xyz", Keep::ALL.with_threads(4)).unwrap();
        crate::keep::NO_THREADS.set(false);
        assert_eq!(one.facts["points"], 20_000);
        for d in [&many, &none] {
            assert_eq!(attrs(&one), attrs(d));
            for a in attrs(&one) {
                assert_eq!(values(&one, &a), values(d, &a), "{a}");
            }
        }
    }

    #[cfg(feature = "fs")]
    #[test]
    fn files_stream_in_batches() {
        let text: String = (0..30_000)
            .map(|i| {
                format!(
                    "{i} {} 2 {}
",
                    i % 100,
                    i % 9
                )
            })
            .collect();
        let path = std::env::temp_dir().join(format!("polysmelt-xyz-{}.xyz", std::process::id()));
        std::fs::write(&path, &text).unwrap();
        let file = load(Input::file(&path), "xyz", Keep::ALL.with_threads(2));
        let small = load(Input::file(&path), "xyz", Keep::ALL.with_max_points(20_000));
        std::fs::remove_file(&path).unwrap();
        let file = file.unwrap();
        let memory = load_text(&text, "xyz", Keep::ALL).unwrap();
        assert_eq!(file.facts["points"], 30_000);
        assert_eq!(values(&file, "POSITION"), values(&memory, "POSITION"));
        assert_eq!(values(&file, "_INTENSITY"), values(&memory, "_INTENSITY"));
        assert!(small.unwrap_err().to_string().contains("Keep::max_points"));
    }

    #[test]
    fn too_many_points_are_refused() {
        let text: String = (0..100).map(|i| format!("{i} 0 0\n")).collect();
        let err = load_text(&text, "xyz", Keep::ALL.with_max_points(10)).unwrap_err();
        assert!(err.to_string().contains("Keep::max_points"), "{err}");
        let err = load_text("ncols 100000\nnrows 100000\ncellsize 1\n1\n", "asc", Keep::ALL).unwrap_err();
        assert!(err.to_string().contains("Keep::max_points"), "{err}");
        let err = load_text(
            "100000\n100000\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
            "ptx",
            Keep::ALL,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Keep::max_points"), "{err}");
    }

    #[test]
    fn broken_files_fail_cleanly() {
        let mut rng = crate::test_rng::Pcg32::new(11);
        let ptx = "2\n2\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n5 5 5 1\n0 0 0 0.5\n1 2 3 0.1 10 20 30\n4 5 6 0.2 1 2 3\n7 8 9 0.3 4 5 6\n";
        let cases = [
            ("xyz", "1 2 3 0.5 0.5 0.7071 4\n5 6 7 0 1 0 2\n".repeat(20)),
            ("pts", "2\n1 2 3 4 5 6 7\n8 9 10 11 12 13 14\n".repeat(20)),
            ("ptx", ptx.repeat(3)),
            (
                "asc",
                "ncols 4\nnrows 3\nxllcenter 0\nyllcenter 0\ncellsize 2\n1 2 3 4\n5 6 7 8\n9 10 11 12\n".into(),
            ),
        ];
        for (ext, text) in cases {
            let good = text.into_bytes();
            assert!(load(Input::memory(&good, "c"), ext, Keep::ALL).is_ok(), "{ext}");
            for cut in 0..good.len() {
                let _ = load(Input::memory(&good[..cut], "c"), ext, Keep::ALL);
            }
            for _ in 0..300 {
                let mut b = good.clone();
                points::corrupt(&mut b, &mut rng, 3);
                let _ = load(Input::memory(&b, "c"), ext, Keep::ALL);
            }
        }
        assert!(load_text("", "xyz", Keep::ALL).is_err());
        assert!(load_text("a b c\n", "xyz", Keep::ALL).is_err());
    }
}
