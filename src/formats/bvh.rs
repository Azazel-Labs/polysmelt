//! BVH (Biovision Hierarchy) motion capture as a glTF document: the skeleton as nodes, the motion as one animation.
//!
//! The `HIERARCHY` section nests `ROOT` and `JOINT` blocks, each with an `OFFSET` from its parent, the `CHANNELS` its
//! motion columns animate, and its children. An `End Site` (or `End site`) is a leaf with only an offset; it becomes a
//! node named after its joint, `<joint>_end`. Nodes come in the order the file lists them, and each `ROOT` (a file may
//! have several) is a root of the scene. A joint's name is the rest of its line, spaces and all.
//!
//! The rest pose is the hierarchy's own: each node's translation is its `OFFSET` and its rotation the identity. That
//! is the pose the skeleton is defined in (mostly a T-pose), which retargeting measures poses against; the motion's
//! first frame is only its first key.
//!
//! The `MOTION` section's `Frames:` and `Frame Time:` lines (`=` may stand for the colon) are followed by a row of
//! numbers per frame, one for each channel in hierarchy order; frame k is at k times the frame time. The rows become
//! one linear animation: a translation, rotation or scale channel for each joint with position, rotation or scale
//! channels. A joint's rotation is the product of its axis rotations (in degrees) in the order its `CHANNELS` lists
//! them, applied to column vectors: `Zrotation Xrotation Yrotation` is Rz·Rx·Ry. Each frame's quaternion is kept on
//! the same side as the last frame's, so interpolation takes the short way. Position channels (`Xposition`, or
//! `Xtranslation`) set their components of the translation in place of the `OFFSET`; a component without a channel
//! keeps the `OFFSET`'s. Scale channels (`Xscale`, or `Xscaling`) set their factors, and an axis without one keeps a
//! factor of 1. Channel names are read in any case. Names the format doesn't define keep their column, which is
//! ignored.
//!
//! Values stay as written: BVH is Y up, as glTF is, and doesn't record its units (centimeters or inches, mostly), so
//! nothing is scaled. The source info has the up axis `+Y` assumed, the units unknown, and `frameRate` one over the
//! `Frame Time:`, declared, or assumed when the time is taken as a thirtieth.
//!
//! Rows are lines. A row with fewer numbers than there are channels, or with anything among them that isn't a finite
//! number (a damaged frame, or the last one cut short by the end of the file), is left out, and the frames around it
//! keep their times. Numbers past the channels are ignored, and so are rows past the `Frames:` count; a file with fewer
//! rows than it declares keeps the ones it has. A header without a `Frames:` line reads rows to the end of the file. A
//! `Frame Time:` that isn't a positive number of seconds, or is missing, is taken as a thirtieth of a second (some
//! writers put 0). A hierarchy that doesn't parse is an error. Nothing is allocated from a
//! count the file gives: the nodes are capped, and the rows take room as they're read. Nesting is followed with a
//! stack of open joints, not recursion, so any depth parses.

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, f32_from, f64_from};
use crate::Result;
use crate::document::{Builder, Document, Keep};
use crate::num::{f32_of, usize_of};
use crate::source::{Axis, Fact, SourceInfo};

/// Nodes (joints and end sites) a file may have: far past any skeleton.
const MAX_NODES: usize = 65_536;
/// Channels one joint may list. The format defines six.
const MAX_CHANNELS: usize = 64;

/// What a motion column sets: a joint's translation, rotation or scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Position,
    Rotation,
    Scale,
}

impl Kind {
    /// The kinds in the order their tracks are written.
    const ALL: [Self; 3] = [Self::Position, Self::Rotation, Self::Scale];
}

/// What one motion column animates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// A translation component, a rotation about an axis (in degrees) or a scale factor: the axis is 0, 1 or 2 for
    /// X, Y or Z.
    Axis(Kind, usize),
    /// A channel the format doesn't define: its column is skipped.
    Other,
}

impl Channel {
    fn parse(name: &[u8]) -> Self {
        let axis = match name.first().map(u8::to_ascii_lowercase) {
            Some(b'x') => 0,
            Some(b'y') => 1,
            Some(b'z') => 2,
            _ => return Self::Other,
        };
        let rest = &name[1..];
        let is = |k: &[u8]| rest.eq_ignore_ascii_case(k);
        let kind = if is(b"position") || is(b"translation") {
            Kind::Position
        } else if is(b"rotation") {
            Kind::Rotation
        } else if is(b"scale") || is(b"scaling") {
            Kind::Scale
        } else {
            return Self::Other;
        };
        Self::Axis(kind, axis)
    }
}

/// A `ROOT`, `JOINT` or `End Site`.
struct Joint {
    name: String,
    offset: [f32; 3],
    /// Its first column in a motion row.
    first: usize,
    channels: Vec<Channel>,
    children: Vec<usize>,
    /// Whether it's an end site.
    end: bool,
}

impl Joint {
    /// Its columns of one kind, each with its axis, in the order listed.
    fn columns(&self, kind: Kind) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (k, c) in self.channels.iter().enumerate() {
            if let Channel::Axis(of, a) = *c
                && of == kind
            {
                out.push((self.first + k, a));
            }
        }
        out
    }
}

/// A joint's translation, rotation or scale through the motion.
struct Track {
    node: usize,
    kind: Kind,
    /// Its columns in a row, each with its axis.
    columns: Vec<(usize, usize)>,
    /// A value per kept row: 3 floats, or 4 for a rotation.
    out: Vec<f32>,
    /// The last rotation, which the next is kept on the same side of.
    last: [f64; 4],
}

/// What the hierarchy's next token must be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Want {
    Keyword,
    /// One of the `OFFSET`'s numbers: this many are read.
    Offset(usize),
    /// The count after `CHANNELS`.
    ChannelCount,
    /// Channel names: this many are left.
    ChannelNames(usize),
}

/// The `HIERARCHY` section, read a line at a time.
struct Hierarchy {
    joints: Vec<Joint>,
    roots: Vec<usize>,
    /// The joints whose braces are open, innermost last.
    open: Vec<usize>,
    /// A joint whose `{` hasn't come yet.
    pending: Option<usize>,
    want: Want,
    /// Channels so far, across every joint: a motion row's width.
    columns: usize,
    /// Whether `HIERARCHY` has been read.
    started: bool,
}

/// Up to 40 characters of `s`, for an error message.
fn show(s: &[u8]) -> String {
    String::from_utf8_lossy(s).chars().take(40).collect()
}

fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| !stream::space(b)).unwrap_or(s.len());
    let end = s.iter().rposition(|&b| !stream::space(b)).map_or(start, |e| e + 1);
    &s[start..end]
}

/// The next token of `s` and what follows it. A brace is a token of its own, wherever it is.
fn token(s: &[u8]) -> Option<(&[u8], &[u8])> {
    let start = s.iter().position(|&b| !stream::space(b))?;
    let s = &s[start..];
    let len = if matches!(s[0], b'{' | b'}') {
        1
    } else {
        s.iter()
            .position(|&b| stream::space(b) || matches!(b, b'{' | b'}'))
            .unwrap_or(s.len())
    };
    Some(s.split_at(len))
}

fn finite(s: &[u8]) -> Option<f32> {
    f32_from(s).filter(|v| v.is_finite())
}

impl Hierarchy {
    fn new() -> Self {
        Self {
            joints: Vec::new(),
            roots: Vec::new(),
            open: Vec::new(),
            pending: None,
            want: Want::Keyword,
            columns: 0,
            started: false,
        }
    }

    /// The innermost open joint, which an `OFFSET` or `CHANNELS` belongs to.
    fn current(&self, what: &str, no: usize) -> Result<usize> {
        match self.open.last() {
            Some(&j) => Ok(j),
            None => bail!(Corrupt, "BVH line {no}: {what} outside any joint's braces"),
        }
    }

    /// Adds a joint under the innermost open one (or as a root), waiting for its `{`.
    fn add(&mut self, name: String, end: bool) -> Result<()> {
        ensure!(
            self.joints.len() < MAX_NODES,
            LimitExceeded,
            "BVH has more than {MAX_NODES} joints"
        );
        let i = self.joints.len();
        match self.open.last() {
            Some(&p) => self.joints[p].children.push(i),
            None => self.roots.push(i),
        }
        self.joints.push(Joint {
            name,
            offset: [0.0; 3],
            first: self.columns,
            channels: Vec::new(),
            children: Vec::new(),
            end,
        });
        self.pending = Some(i);
        Ok(())
    }

    /// Reads one line (`no` counting from 1); true at `MOTION`, where the hierarchy ends.
    fn line(&mut self, line: &[u8], no: usize) -> Result<bool> {
        let mut rest = line;
        while let Some((tok, after)) = token(rest) {
            rest = after;
            match self.want {
                Want::Keyword => {}
                Want::Offset(k) => {
                    let Some(v) = finite(tok) else {
                        bail!(
                            Corrupt,
                            "BVH line {no}: OFFSET needs three numbers, not `{}`",
                            show(tok)
                        );
                    };
                    let j = self.current("OFFSET", no)?;
                    self.joints[j].offset[k] = v;
                    self.want = if k == 2 { Want::Keyword } else { Want::Offset(k + 1) };
                    continue;
                }
                Want::ChannelCount => {
                    let n = std::str::from_utf8(tok).ok().and_then(|s| s.parse::<usize>().ok());
                    let Some(n) = n.filter(|&n| n <= MAX_CHANNELS) else {
                        bail!(
                            Corrupt,
                            "BVH line {no}: CHANNELS needs a count up to {MAX_CHANNELS}, not `{}`",
                            show(tok)
                        );
                    };
                    self.want = if n == 0 { Want::Keyword } else { Want::ChannelNames(n) };
                    continue;
                }
                Want::ChannelNames(n) => {
                    let j = self.current("CHANNELS", no)?;
                    self.joints[j].channels.push(Channel::parse(tok));
                    self.columns += 1;
                    self.want = if n == 1 {
                        Want::Keyword
                    } else {
                        Want::ChannelNames(n - 1)
                    };
                    continue;
                }
            }
            if !self.started {
                ensure!(
                    tok.eq_ignore_ascii_case(b"HIERARCHY"),
                    UnknownFormat,
                    "not a BVH file: it starts with `{}`, not HIERARCHY",
                    show(tok)
                );
                self.started = true;
                continue;
            }
            if let Some(j) = self.pending
                && tok != b"{"
            {
                bail!(
                    Corrupt,
                    "BVH line {no}: `{{` should follow joint `{}`, not `{}`",
                    self.joints[j].name,
                    show(tok)
                );
            }
            let is = |k: &[u8]| tok.eq_ignore_ascii_case(k);
            if tok == b"{" {
                let Some(j) = self.pending.take() else {
                    bail!(Corrupt, "BVH line {no}: `{{` opens no joint");
                };
                self.open.push(j);
            } else if tok == b"}" {
                ensure!(
                    self.open.pop().is_some(),
                    Corrupt,
                    "BVH line {no}: `}}` closes no joint"
                );
            } else if is(b"ROOT") || is(b"JOINT") {
                if is(b"ROOT") {
                    if let Some(&j) = self.open.last() {
                        bail!(
                            Corrupt,
                            "BVH line {no}: ROOT inside joint `{}` (a `}}` is missing)",
                            self.joints[j].name
                        );
                    }
                } else {
                    ensure!(
                        !self.open.is_empty(),
                        Corrupt,
                        "BVH line {no}: JOINT outside any joint's braces"
                    );
                }
                // The name is the rest of the line, up to a `{` on it.
                let cut = rest.iter().position(|&b| b == b'{').unwrap_or(rest.len());
                let name = trim(&rest[..cut]);
                rest = &rest[cut..];
                let name = if name.is_empty() {
                    format!("joint{}", self.joints.len())
                } else {
                    String::from_utf8_lossy(name).into_owned()
                };
                self.add(name, false)?;
            } else if is(b"End") {
                let site = token(rest).filter(|(t, _)| t.eq_ignore_ascii_case(b"Site"));
                let Some((_, after)) = site else {
                    bail!(Corrupt, "BVH line {no}: `End` should be followed by `Site`");
                };
                rest = after;
                let Some(&parent) = self.open.last() else {
                    bail!(Corrupt, "BVH line {no}: End Site outside any joint's braces");
                };
                let name = format!("{}_end", self.joints[parent].name);
                self.add(name, true)?;
            } else if is(b"OFFSET") {
                self.current("OFFSET", no)?;
                self.want = Want::Offset(0);
            } else if is(b"CHANNELS") {
                let j = self.current("CHANNELS", no)?;
                ensure!(
                    self.joints[j].channels.is_empty(),
                    Corrupt,
                    "BVH line {no}: joint `{}` lists CHANNELS twice",
                    self.joints[j].name
                );
                self.joints[j].first = self.columns;
                self.want = Want::ChannelCount;
            } else if is(b"MOTION") {
                self.finish()?;
                return Ok(true);
            } else {
                bail!(Corrupt, "BVH line {no}: `{}` isn't part of a BVH hierarchy", show(tok));
            }
        }
        Ok(false)
    }

    /// Checks the hierarchy is complete, at `MOTION` or the end of the file.
    fn finish(&self) -> Result<()> {
        ensure!(self.started, Empty, "not a BVH file: it's empty");
        if let Some(j) = self.pending.or(self.open.last().copied()) {
            bail!(
                Truncated,
                "BVH hierarchy ends inside joint `{}`: a brace is missing, or the file is cut short",
                self.joints[j].name
            );
        }
        ensure!(
            self.want == Want::Keyword,
            Truncated,
            "BVH hierarchy ends inside an OFFSET or CHANNELS"
        );
        ensure!(!self.roots.is_empty(), Corrupt, "BVH hierarchy has no ROOT");
        Ok(())
    }
}

/// The `MOTION` section, read a line at a time.
#[derive(Default)]
struct Motion {
    /// The `Frames:` count, as the file gives it.
    frames: Option<u64>,
    /// `Frame Time:`, in seconds.
    frame_time: Option<f64>,
    /// Rows read, kept or not.
    rows: u64,
    /// The kept rows' numbers, a row's width each (when animations are kept).
    values: Vec<f32>,
    /// Each kept row's place among the rows.
    kept: Vec<u64>,
    /// Rows left out for having too few numbers, or one that isn't.
    dropped: u64,
    /// Whether `Frame Time:` wasn't a positive number (or wasn't there), so the default was taken.
    guessed_time: bool,
    /// Whether the header is over and rows are being read.
    rows_begun: bool,
}

/// The frame time taken when the file's isn't a positive number of seconds.
const DEFAULT_FRAME_TIME: f64 = 1.0 / 30.0;

/// Whether `s` starts with `key`, in any case; what follows if so.
fn after_word<'s>(s: &'s [u8], key: &[u8]) -> Option<&'s [u8]> {
    (s.len() >= key.len() && s[..key.len()].eq_ignore_ascii_case(key)).then(|| &s[key.len()..])
}

/// A header line's value: what follows its `:` or `=` (or its name, without either).
fn value(rest: &[u8]) -> &[u8] {
    let rest = trim(rest);
    trim(
        rest.strip_prefix(b":")
            .or_else(|| rest.strip_prefix(b"="))
            .unwrap_or(rest),
    )
}

impl Motion {
    /// Reads one line of the section, not blank; true once no more rows are wanted. `columns` is a row's width,
    /// `room` a bound on the rows the file can hold, and `store` whether to keep the rows' numbers.
    fn line(&mut self, line: &[u8], no: usize, columns: usize, room: usize, store: bool) -> Result<bool> {
        if !self.rows_begun {
            let t = trim(line);
            if let Some(rest) = after_word(t, b"frames") {
                let v = value(rest);
                let Some(n) = std::str::from_utf8(v).ok().and_then(|s| s.parse::<u64>().ok()) else {
                    bail!(
                        Corrupt,
                        "BVH line {no}: the Frames count `{}` isn't a whole number",
                        show(v)
                    );
                };
                self.frames = Some(n);
            } else if let Some(rest) = after_word(t, b"frame").and_then(|r| after_word(trim(r), b"time")) {
                let s = f64_from(value(rest)).filter(|s| s.is_finite() && *s > 0.0);
                self.guessed_time = s.is_none();
                self.frame_time = Some(s.unwrap_or(DEFAULT_FRAME_TIME));
            } else if token(t).is_some_and(|(w, _)| finite(w).is_some()) {
                // A row where a header line was due: the header is missing a line, and the rows go on to the end
                // (or to the count, if there was one).
                self.begin(columns, room, store);
            } else {
                bail!(
                    Corrupt,
                    "BVH line {no}: MOTION should go on with `Frames:` and `Frame Time:`, not `{}`",
                    show(t)
                );
            }
            if !self.rows_begun {
                if self.frames.is_none() || self.frame_time.is_none() {
                    return Ok(false);
                }
                self.begin(columns, room, store);
                return Ok(self.frames == Some(0));
            }
        }
        let frames = self.frames.unwrap_or(u64::MAX);
        if self.rows >= frames {
            return Ok(true);
        }
        let row = self.rows;
        self.rows += 1;
        let start = self.values.len();
        let mut n = 0;
        let mut rest = line;
        while n < columns {
            let Some(at) = rest.iter().position(|&b| !stream::space(b)) else {
                break;
            };
            let len = rest[at..]
                .iter()
                .position(|&b| stream::space(b))
                .unwrap_or(rest.len() - at);
            let w = &rest[at..at + len];
            rest = &rest[at + len..];
            match finite(w) {
                Some(v) => {
                    if store {
                        self.values.push(v);
                    }
                    n += 1;
                }
                // Junk, or a number the end of the file cut short: the row is left out.
                None => break,
            }
        }
        if n < columns {
            self.values.truncate(start);
            self.dropped += 1;
        } else {
            self.kept.push(row);
        }
        Ok(self.rows >= frames)
    }

    /// Ends the header: a missing frame time is taken as the default, and room is made for the rows.
    fn begin(&mut self, columns: usize, room: usize, store: bool) {
        self.rows_begun = true;
        if self.frame_time.is_none() {
            self.guessed_time = true;
            self.frame_time = Some(DEFAULT_FRAME_TIME);
        }
        // Room for the rows the file can hold, not for any count it claims.
        let rows = usize_of(self.frames.unwrap_or(u64::MAX)).min(room);
        self.kept.reserve(rows);
        if store {
            self.values.reserve(rows.saturating_mul(columns));
        }
    }
}

/// The Hamilton product `a·b` of quaternions `[x, y, z, w]`: rotating by `b`, then by `a`.
fn mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// The rotation by `degrees` about axis `axis` (0, 1 or 2 for X, Y or Z).
fn about(axis: usize, degrees: f64) -> [f64; 4] {
    let (s, c) = (degrees.to_radians() * 0.5).sin_cos();
    let mut q = [0.0, 0.0, 0.0, c];
    q[axis] = s;
    q
}

/// Key times for the kept rows: row k at k frame times, each later than the last even where single precision
/// would round two together.
fn key_times(kept: &[u64], frame_time: f64) -> Vec<f32> {
    let mut times: Vec<f32> = Vec::with_capacity(kept.len());
    for &k in kept {
        #[allow(clippy::cast_precision_loss, reason = "frame numbers are far below 2^52")]
        let mut t = f32_of(k as f64 * frame_time);
        if let Some(&p) = times.last()
            && t <= p
        {
            t = p.next_up();
        }
        times.push(t);
    }
    times
}

pub(super) fn load(input: Input<'_>, keep: Keep) -> Result<Document<'static>> {
    let mut blocks = input.blocks()?;
    let size = blocks.size_hint();
    let mut h = Hierarchy::new();
    let mut motion: Option<Motion> = None;
    let mut no = 0;
    let mut first = true;
    'read: while let Some(chunk) = blocks.text(stream::lines)? {
        let mut chunk = chunk;
        if std::mem::take(&mut first) {
            chunk = chunk.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(chunk);
        }
        // Chunks end on a line break, except the file's last, which may stop mid-line.
        let terminated = chunk.last() == Some(&b'\n');
        let body = if terminated { &chunk[..chunk.len() - 1] } else { chunk };
        for line in body.split(|&b| b == b'\n') {
            no += 1;
            // A CR before the LF leaves an empty part; a CR alone (old Mac files) ends a line.
            for part in line.split(|&b| b == b'\r') {
                if part.iter().all(|&b| stream::space(b)) {
                    continue;
                }
                match &mut motion {
                    None => {
                        if h.line(part, no)? {
                            motion = Some(Motion::default());
                        }
                    }
                    Some(m) => {
                        // Each value takes two bytes at least: a digit and a space.
                        let room = size / (2 * h.columns).max(1);
                        if m.line(part, no, h.columns, room, keep.animations)? {
                            break 'read;
                        }
                    }
                }
            }
        }
    }
    if motion.is_none() {
        h.finish()?;
    }

    let joints = &h.joints;
    let mut b = Builder::default();
    let nodes: Vec<Value> = joints
        .iter()
        .map(|j| {
            let mut n = json!({ "name": j.name });
            if j.offset.iter().any(|&v| v != 0.0) {
                n["translation"] = json!(j.offset);
            }
            if !j.children.is_empty() {
                n["children"] = json!(j.children);
            }
            n
        })
        .collect();

    let mut animations = Vec::new();
    if keep.animations
        && let Some(m) = &motion
        && let Some(frame_time) = m.frame_time
        && !m.kept.is_empty()
        && h.columns > 0
    {
        let input = b.floats(key_times(&m.kept, frame_time), "SCALAR", true);
        let frames = m.kept.len();
        // The rows are read once, in order, for every track.
        let mut tracks = Vec::new();
        for (node, j) in joints.iter().enumerate() {
            for kind in Kind::ALL {
                let columns = j.columns(kind);
                if !columns.is_empty() {
                    let width = if kind == Kind::Rotation { 4 } else { 3 };
                    tracks.push(Track {
                        node,
                        kind,
                        columns,
                        out: Vec::with_capacity(frames * width),
                        last: [0.0, 0.0, 0.0, 1.0],
                    });
                }
            }
        }
        for row in m.values.chunks_exact(h.columns) {
            for track in &mut tracks {
                if track.kind != Kind::Rotation {
                    // A component without a channel keeps the rest pose's: the offset, or a scale of 1.
                    let mut t = match track.kind {
                        Kind::Position => joints[track.node].offset,
                        _ => [1.0; 3],
                    };
                    for &(col, axis) in &track.columns {
                        t[axis] = row[col];
                    }
                    track.out.extend_from_slice(&t);
                    continue;
                }
                let mut q = [0.0, 0.0, 0.0, 1.0];
                for &(col, axis) in &track.columns {
                    q = mul(q, about(axis, f64::from(row[col])));
                }
                let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
                // Each quaternion on the same side as the last, so slerp takes the short way.
                let dot: f64 = (0..4).map(|k| q[k] * track.last[k]).sum();
                let scale = if dot < 0.0 { -len.recip() } else { len.recip() };
                q = q.map(|c| c * scale);
                track.last = q;
                track.out.extend(q.map(f32_of));
            }
        }
        let (mut channels, mut samplers) = (Vec::new(), Vec::new());
        for track in tracks {
            let (path, kind) = match track.kind {
                Kind::Position => ("translation", "VEC3"),
                Kind::Rotation => ("rotation", "VEC4"),
                Kind::Scale => ("scale", "VEC3"),
            };
            let output = b.floats(track.out, kind, false);
            channels.push(json!({ "sampler": samplers.len(), "target": { "node": track.node, "path": path } }));
            samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
        }
        if !channels.is_empty() {
            animations.push(json!({ "name": "motion", "channels": channels, "samplers": samplers }));
        }
    }

    let mut facts = Map::new();
    facts.insert("joints".into(), json!(joints.iter().filter(|j| !j.end).count()));
    facts.insert("channels".into(), json!(h.columns));
    let mut info = SourceInfo {
        format: Some("BVH".into()),
        // Y up by the format's convention; the file doesn't say.
        up_axis: Fact::Assumed(Axis::PosY),
        ..SourceInfo::default()
    };
    let mut source = Map::new();
    if let Some(m) = &motion {
        facts.insert("frames".into(), json!(m.kept.len()));
        if let Some(t) = m.frame_time {
            info.frame_rate = if m.guessed_time {
                Fact::Assumed(1.0 / t)
            } else {
                Fact::Declared(1.0 / t)
            };
        }
        if m.dropped > 0 {
            facts.insert("dropped_frames".into(), json!(m.dropped));
        }
        if let Some(n) = m.frames {
            source.insert("frames".into(), json!(n));
        }
    }
    info.write(&mut source);
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": h.roots }]));
    json.insert("nodes".into(), Value::Array(nodes));
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    Ok(b.finish(json, facts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::rig::Path;

    fn read(text: &str) -> Result<Document<'static>> {
        load(Input::memory(text.as_bytes(), "t.bvh"), Keep::ALL)
    }

    fn error(text: &str) -> String {
        match read(text) {
            Ok(_) => panic!("read without error"),
            Err(e) => e.to_string(),
        }
    }

    const TWO: &str = "HIERARCHY
ROOT Hips
{
  OFFSET 0 90 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 5 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.5
1 2 3 0 0 0 0 0 0
4 5 6 90 0 0 0 0 90
";

    /// `v` rotated by unit quaternion `q`.
    fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        let [x, y, z, w] = q.map(f64::from);
        let [a, b, c] = v.map(f64::from);
        // t = 2 q.xyz × v, then v + w t + q.xyz × t.
        let t = [2.0 * (y * c - z * b), 2.0 * (z * a - x * c), 2.0 * (x * b - y * a)];
        let r = [
            a + w * t[0] + (y * t[2] - z * t[1]),
            b + w * t[1] + (z * t[0] - x * t[2]),
            c + w * t[2] + (x * t[1] - y * t[0]),
        ];
        r.map(f32_of)
    }

    fn near(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn two_joints() {
        let doc = read(TWO).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        let names: Vec<_> = nodes.iter().map(|n| n["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["Hips", "Spine", "Spine_end"]);
        assert_eq!(nodes[0]["translation"], json!([0.0, 90.0, 0.0]));
        assert_eq!(nodes[0]["children"], json!([1]));
        assert_eq!(doc.json["scenes"][0]["nodes"], json!([0]));
        assert_eq!(doc.facts["joints"], json!(2));
        assert_eq!(doc.facts["channels"], json!(9));
        assert_eq!(doc.facts["frames"], json!(2));
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("BVH"));
        assert_eq!(info.frame_rate, Fact::Declared(2.0));
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosY));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.converted, crate::source::Converted::default());

        let rig = doc.rig().unwrap();
        assert_eq!(rig.animations.len(), 1);
        let channels = &rig.animations[0].channels;
        let paths: Vec<_> = channels.iter().map(|c| (c.node, c.path)).collect();
        assert_eq!(
            paths,
            [(0, Path::Translation), (0, Path::Rotation), (1, Path::Rotation)]
        );
        assert_eq!(channels[0].times, [0.0, 0.5]);
        assert_eq!(channels[0].values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // The rest pose: the offsets, unrotated.
        let rest = rig.pose(None, 0.0);
        assert_eq!(rest[1].translation, [0.0, 10.0, 0.0]);
        assert_eq!(rest[1].rotation, [0.0, 0.0, 0.0, 1.0]);
        // Frame 1: Hips turned 90° about Z, Spine 90° about Y.
        let pose = rig.pose(Some(0), 0.5);
        assert!(near(&rotate(pose[0].rotation, [1.0, 0.0, 0.0]), &[0.0, 1.0, 0.0]));
        assert!(near(&rotate(pose[1].rotation, [0.0, 0.0, 1.0]), &[1.0, 0.0, 0.0]));
    }

    #[test]
    fn rotations_compose_in_listed_order() {
        // Rz(90)·Rx(90) takes X to Y and Y to Z; Rx(90)·Rz(90) would take X to Z.
        let text = |order: &str, values: &str| {
            format!(
                "HIERARCHY\nROOT a\n{{\nOFFSET 0 0 0\nCHANNELS 2 {order}\nEnd Site\n{{\nOFFSET 0 1 0\n}}\n}}\nMOTION\nFrames: 1\nFrame Time: 1\n{values}\n"
            )
        };
        let q = |doc: &Document| doc.rig().unwrap().pose(Some(0), 0.0)[0].rotation;
        let zx = q(&read(&text("Zrotation Xrotation", "90 90")).unwrap());
        assert!(near(&rotate(zx, [1.0, 0.0, 0.0]), &[0.0, 1.0, 0.0]));
        assert!(near(&rotate(zx, [0.0, 1.0, 0.0]), &[0.0, 0.0, 1.0]));
        let xz = q(&read(&text("Xrotation Zrotation", "90 90")).unwrap());
        assert!(near(&rotate(xz, [1.0, 0.0, 0.0]), &[0.0, 0.0, 1.0]));
    }

    #[test]
    fn quaternions_stay_on_one_side() {
        let rows: String = (0..=36).map(|k| (k * 10).to_string() + "\n").collect();
        let text = format!(
            "HIERARCHY\nROOT a\n{{\nOFFSET 0 0 0\nCHANNELS 1 Yrotation\n}}\nMOTION\nFrames: 37\nFrame Time: 0.1\n{rows}"
        );
        let rig = read(&text).unwrap().rig().unwrap();
        let v = &rig.animations[0].channels[0].values;
        assert_eq!(v.len(), 37 * 4);
        assert!(v[3] > 0.0);
        for (a, b) in v.chunks(4).zip(v.chunks(4).skip(1)) {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            assert!(dot > 0.0, "{a:?} then {b:?}");
        }
        // 360° is the identity, from the other side.
        assert!(near(&v[36 * 4..], &[0.0, 0.0, 0.0, -1.0]));
    }

    #[test]
    fn crlf_tabs_bom_and_cr_alone() {
        let crlf = format!("\u{feff}{}", TWO.replace('\n', "\r\n").replace("  ", "\t"));
        let a = read(&crlf).unwrap();
        let b = read(TWO).unwrap();
        assert_eq!(a.json["nodes"], b.json["nodes"]);
        assert_eq!(a.facts, b.facts);
        let cr = read(&TWO.replace('\n', "\r")).unwrap();
        assert_eq!(cr.json["nodes"], b.json["nodes"]);
        assert_eq!(cr.facts, b.facts);
    }

    #[test]
    fn spellings() {
        let text = "hierarchy\nroot My Hips (1){\noffset 1e1 -0 2.5E-1\nchannels 1 xROTATION\nend site{ offset 0 1 0 }}\nmotion\nframes:1\nframe time:\t0.0333\n-1.5e+1\n";
        let doc = read(text).unwrap();
        let nodes = doc.json["nodes"].as_array().unwrap();
        assert_eq!(nodes[0]["name"], "My Hips (1)");
        assert_eq!(nodes[1]["name"], "My Hips (1)_end");
        assert_eq!(nodes[0]["translation"], json!([10.0, -0.0, 0.25]));
        assert_eq!(doc.facts["frames"], json!(1));
        assert_eq!(doc.json["animations"][0]["channels"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn short_and_long_rows() {
        let text = TWO.replace("Frames: 2", "Frames: 5").replace(
            "4 5 6 90 0 0 0 0 90\n",
            "4 5 6 90 0 0 0 0 90 7 7\n1 2 3\n7 8 9 0 0 0 0 0 0\n7 8",
        );
        let doc = read(&text).unwrap();
        assert_eq!(doc.facts["frames"], json!(3));
        assert_eq!(doc.facts["dropped_frames"], json!(2));
        let rig = doc.rig().unwrap();
        let c = &rig.animations[0].channels[0];
        assert_eq!(c.times, [0.0, 0.5, 1.5]);
        assert_eq!(c.values[6..], [7.0, 8.0, 9.0]);
        // Rows past the count are ignored, junk or not.
        let text = TWO.replace("Frames: 2", "Frames: 1") + "junk\n";
        assert_eq!(read(&text).unwrap().facts["frames"], json!(1));
        // A number the end of the file cuts short only drops its row.
        let text = TWO.trim_end().strip_suffix(" 90").unwrap().to_owned() + " 1e";
        assert_eq!(read(&text).unwrap().facts["dropped_frames"], json!(1));
    }

    #[test]
    fn no_motion_and_no_frames() {
        let hierarchy = TWO.split("MOTION").next().unwrap();
        let doc = read(hierarchy).unwrap();
        assert!(doc.json.get("animations").is_none());
        assert!(doc.facts.get("frames").is_none());
        assert_eq!(doc.facts["channels"], json!(9));
        let zero = format!("{hierarchy}MOTION\nFrames: 0\nFrame Time: 0.1\n");
        let doc = read(&zero).unwrap();
        assert!(doc.json.get("animations").is_none());
        assert_eq!(doc.facts["frames"], json!(0));
        // Fewer rows than declared keeps what's there.
        let doc = read(&format!("{hierarchy}MOTION\nFrames: 9\nFrame Time: 0.1\n")).unwrap();
        assert_eq!(doc.facts["frames"], json!(0));
    }

    #[test]
    fn junk_rows_are_dropped_and_a_bad_frame_time_is_a_thirtieth() {
        for junk in ["4 5 6 nan", "4 5 x 90", "4 5 inf 90"] {
            let text = TWO.replace("Frames: 2", "Frames: 3").replace("4 5 6 90", junk) + "7 8 9 0 0 0 0 0 0\n";
            let doc = read(&text).unwrap();
            assert_eq!(
                (&doc.facts["frames"], &doc.facts["dropped_frames"]),
                (&json!(2), &json!(1)),
                "{junk}"
            );
            let rig = doc.rig().unwrap();
            // The frame after the junk keeps its time.
            assert_eq!(rig.animations[0].channels[0].times, [0.0, 1.0], "{junk}");
        }
        for time in ["0", "-1", "inf", "soon"] {
            let doc = read(&TWO.replace("Frame Time: 0.5", &format!("Frame Time: {time}"))).unwrap();
            assert_eq!(doc.source_info().frame_rate, Fact::Assumed(30.0), "{time}");
        }
        assert_eq!(read(TWO).unwrap().source_info().frame_rate, Fact::Declared(2.0));
    }

    #[test]
    fn bad_values_are_errors() {
        let e = error(&TWO.replace("OFFSET 0 10 0", "OFFSET 0 NaN 0"));
        assert!(e.contains("OFFSET"), "{e}");
        assert!(error(&TWO.replace("Frames: 2", "Frames: -2")).contains("Frames"));
        assert!(error(&TWO.replace("Frames: 2", "Frames: 99999999999999999999999")).contains("Frames"));
        assert!(error(&TWO.replace("Frames: 2", "Frame count 2")).contains("MOTION"));
    }

    #[test]
    fn scale_channels() {
        let text = "HIERARCHY\nROOT a\n{\nOFFSET 0 1 0\nCHANNELS 5 Xposition Yrotation Xscale YSCALE zScale\n\
            JOINT b\n{\nOFFSET 0 1 0\nCHANNELS 2 Yscale Zrotation\nEnd Site\n{\nOFFSET 0 1 0\n}\n}\n}\n\
            MOTION\nFrames: 2\nFrame Time: 0.5\n0 0 1 1 1 0 0\n5 90 2 3 0.5 4 45\n";
        let doc = read(text).unwrap();
        let rig = doc.rig().unwrap();
        let channels = &rig.animations[0].channels;
        let paths: Vec<_> = channels.iter().map(|c| (c.node, c.path)).collect();
        assert_eq!(
            paths,
            [
                (0, Path::Translation),
                (0, Path::Rotation),
                (0, Path::Scale),
                (1, Path::Rotation),
                (1, Path::Scale)
            ]
        );
        assert_eq!(channels[2].values, [1.0, 1.0, 1.0, 2.0, 3.0, 0.5]);
        // A scale axis without a channel stays 1.
        assert_eq!(channels[4].values, [1.0, 0.0, 1.0, 1.0, 4.0, 1.0]);
        let pose = rig.pose(Some(0), 0.5);
        assert_eq!(pose[0].translation, [5.0, 1.0, 0.0]);
        assert_eq!(pose[0].scale, [2.0, 3.0, 0.5]);
        // The rest pose is unscaled.
        assert_eq!(rig.pose(None, 0.0)[0].scale, [1.0, 1.0, 1.0]);
        // Other spellings exporters write.
        assert_eq!(Channel::parse(b"Xtranslation"), Channel::Axis(Kind::Position, 0));
        assert_eq!(Channel::parse(b"zScaling"), Channel::Axis(Kind::Scale, 2));
        assert_eq!(Channel::parse(b"Wscale"), Channel::Other);
        assert_eq!(Channel::parse(b""), Channel::Other);
    }

    #[test]
    fn header_variations() {
        let hierarchy = TWO.split("MOTION").next().unwrap();
        let rows = "1 2 3 0 0 0 0 0 0\n4 5 6 90 0 0 0 0 90\n";
        // `=` for `:`.
        let doc = read(&format!("{hierarchy}MOTION\nFrames = 2\nFrame Time = 0.5\n{rows}")).unwrap();
        assert_eq!(doc.facts, read(TWO).unwrap().facts);
        // No Frames line: the rows go on to the end of the file.
        let doc = read(&format!(
            "{hierarchy}MOTION\nFrame Time: 0.5\n{rows}5 5 5 0 0 0 0 0 0\n"
        ))
        .unwrap();
        assert_eq!(doc.facts["frames"], json!(3));
        assert!(doc.json["asset"]["extras"]["source"].get("frames").is_none());
        assert_eq!(doc.rig().unwrap().animations[0].channels[0].times, [0.0, 0.5, 1.0]);
        // No Frame Time line: a thirtieth of a second, noted.
        let doc = read(&format!("{hierarchy}MOTION\nFrames: 2\n{rows}")).unwrap();
        assert_eq!(doc.facts["frames"], json!(2));
        assert_eq!(doc.source_info().frame_rate, Fact::Assumed(30.0));
        // Neither: the rows alone.
        let doc = read(&format!("{hierarchy}MOTION\n{rows}")).unwrap();
        assert_eq!(doc.facts["frames"], json!(2));
        // A MOTION section with nothing in it has no frames.
        let doc = read(&format!("{hierarchy}MOTION\n")).unwrap();
        assert_eq!(doc.facts["frames"], json!(0));
        assert!(doc.json.get("animations").is_none());
    }

    #[test]
    fn bad_hierarchies_are_errors() {
        let cases = [
            ("", "empty"),
            ("ROOT a\n{\n}\n", "HIERARCHY"),
            ("HIERARCHY\n", "no ROOT"),
            ("HIERARCHY\nJOINT a\n{\n}\n", "JOINT outside"),
            ("HIERARCHY\nROOT a\n{\nOFFSET 0 0 0\n", "ends inside joint `a`"),
            ("HIERARCHY\nROOT a\nOFFSET 0 0 0\n", "`{` should follow"),
            ("HIERARCHY\nROOT a\n{\nOFFSET 0 0\n}\n", "OFFSET needs three numbers"),
            ("HIERARCHY\nROOT a\n{\n}\n}\n", "closes no joint"),
            ("HIERARCHY\nROOT a\n{\nROOT b\n{\n}\n}\n", "ROOT inside"),
            ("HIERARCHY\nROOT a\n{\nEnd\n{\n}\n}\n", "Site"),
            (
                "HIERARCHY\nROOT a\n{\nCHANNELS 65 Xrotation\n}\n",
                "CHANNELS needs a count",
            ),
            (
                "HIERARCHY\nROOT a\n{\nCHANNELS 1 Xrotation\nCHANNELS 1 Yrotation\n}\n",
                "twice",
            ),
            ("HIERARCHY\nROOT a\n{\nWIDGET 1\n}\n", "WIDGET"),
            ("HIERARCHY\nROOT a\n{\nMOTION\nFrames: 0\n", "ends inside joint"),
            ("HIERARCHY\nOFFSET 0 0 0\n", "outside"),
            ("HIERARCHY\nROOT a\n{\nCHANNELS 2 Xrotation\n}\n", "ends inside joint"),
        ];
        for (text, want) in cases {
            let e = error(text);
            assert!(e.contains(want), "{text:?}: {e}");
        }
        // A channel name the format doesn't define only takes up its column.
        let text = "HIERARCHY\nROOT a\n{\nCHANNELS 2 Xrotation Wiggle\n}\nMOTION\nFrames: 1\nFrame Time: 1\n90 7\n";
        let doc = read(text).unwrap();
        assert_eq!(doc.facts["channels"], json!(2));
        assert_eq!(doc.json["animations"][0]["channels"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn deep_nesting() {
        let depth = MAX_NODES - 1;
        let mut text = String::from("HIERARCHY\nROOT j\n{\nOFFSET 0 1 0\nCHANNELS 1 Zrotation\n");
        for _ in 1..depth {
            text.push_str("JOINT j\n{\nOFFSET 0 1 0\n");
        }
        text.push_str("End Site\n{\nOFFSET 0 1 0\n}\n");
        text.push_str(&"}\n".repeat(depth));
        text.push_str("MOTION\nFrames: 1\nFrame Time: 1\n45\n");
        let assets = Memory::new();
        let doc = crate::formats::load_bytes(text.as_bytes(), "t.bvh", "bvh", &assets, Keep::ALL).unwrap();
        assert_eq!(doc.json["nodes"].as_array().unwrap().len(), MAX_NODES);
        assert_eq!(doc.facts["joints"], json!(depth));
        // One more joint is too many.
        let text = text.replacen("JOINT j\n", "JOINT k\n{\n}\nJOINT j\n", 1);
        assert!(error(&text).contains("more than"));
    }

    #[test]
    fn counts_in_the_header_allocate_nothing() {
        let text = TWO.replace("Frames: 2", &format!("Frames: {}", u64::MAX));
        let doc = read(&text).unwrap();
        assert_eq!(doc.facts["frames"], json!(2));
        assert_eq!(doc.json["asset"]["extras"]["source"]["frames"], json!(u64::MAX));
    }

    #[test]
    fn keeping_no_animations_keeps_the_facts() {
        let all = read(TWO).unwrap();
        let keep = Keep {
            animations: false,
            ..Keep::ALL
        };
        let none = load(Input::memory(TWO.as_bytes(), "t.bvh"), keep).unwrap();
        assert!(none.json.get("animations").is_none());
        assert_eq!(all.facts, none.facts);
        assert_eq!(all.json["nodes"], none.json["nodes"]);
    }

    #[test]
    fn key_times_increase() {
        let times = key_times(&[0, 1, 2], 1e-30);
        assert!(times[0] < times[1] && times[1] < times[2]);
        assert_eq!(key_times(&[0, 3], 0.5), [0.0, 1.5]);
    }
}
