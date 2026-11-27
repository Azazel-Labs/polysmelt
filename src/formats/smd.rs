//! Valve SMD (Studiomdl Data), the text format the `GoldSrc` and Source model compilers read: a skeleton (`nodes`,
//! and its poses frame by frame under `skeleton`), triangles skinned to it (`triangles`), and in a `.vta` file
//! vertex animation (`vertexanimation`), the shapes a face's flexes are made of.
//!
//! A reference file becomes a mesh skinned to the skeleton as it stands at time 0: each vertex moves with its
//! parent bone, or with the bones its links weight, the parent taking whatever weight they leave. A file with more
//! than one frame becomes an animation at 30 frames a second. A `.vta` file becomes a point cloud, its first frame
//! the points and each later frame a morph target. Rotations are Euler angles in radians, applied X, then Y, then
//! Z. SMD is Z up and right-handed: the scene stays so under the document's `RootNode`, which turns it to Y up. In the source info, the up axis (`+Z`), handedness
//! and frame rate are assumed, as the format's convention; game units are kept unscaled, and their length unknown.

use std::collections::HashMap;

use serde_json::json;

use super::scene::{
    self, Animation, Channel, Interp, M4, Node, Path as Key, Primitive, Scene, SceneMesh, Skin, Target,
};
use super::stream::{self, f32_from, f64_from};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{FastMap, Material};
use crate::num::{f32_of, u32_of};
use crate::quat;
use crate::source::{Axis, Fact, Handedness};

/// Frames a second: the compilers' default (studiomdl plays a `$sequence` at 30 unless its `fps` option says
/// otherwise), which is the readers' shared assumption too.
const FPS: f64 = super::keys::ASSUMED_FRAME_RATE;

/// The most morph target values (targets times points) a `.vta` file may make, against a file whose frames and
/// points multiply past what memory holds.
const MAX_TARGET_VALUES: usize = 1 << 23;

/// A bone, as `nodes` declares it.
struct Bone {
    name: String,
    parent: i64,
}

/// One bone's pose in a frame: position, then Euler angles.
type Pose = [f32; 6];

/// A `skeleton` frame: its time and the poses of the bones it lists, by bone index.
struct Frame {
    time: f64,
    poses: Vec<(usize, Pose)>,
}

/// A vertex of `triangles`.
struct Vertex {
    parent: i64,
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    /// Bone ids and weights.
    links: Vec<(i64, f32)>,
}

/// A triangle: its material (an index into `Smd::materials`) and corners.
struct Triangle {
    material: usize,
    corners: [Vertex; 3],
}

/// A `vertexanimation` frame: its time, the name a comment gives it, and its vertices by id, position and normal.
struct Shape {
    time: f64,
    name: Option<String>,
    vertices: Vec<(i64, [f32; 6])>,
}

/// What a file holds.
#[derive(Default)]
struct Smd {
    version: i64,
    bones: Vec<Bone>,
    /// Bone ids to indices into `bones`.
    ids: HashMap<i64, usize>,
    frames: Vec<Frame>,
    materials: Vec<String>,
    triangles: Vec<Triangle>,
    shapes: Vec<Shape>,
    /// Triangles a cut-short file left without all their corners.
    dropped: usize,
}

/// Reads an `.smd` or `.vta` file (`ext`).
pub(crate) fn load(input: super::Input<'_>, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let text = scene::text(&bytes);
    let smd = parse(&text, keep.triangles())?;
    let stem = input
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let vta = ext.eq_ignore_ascii_case("vta") || (smd.triangles.is_empty() && !smd.shapes.is_empty());

    let mut scene = Scene::default();
    let frames = settle_frames(smd.frames);
    let rest = frames
        .iter()
        .find(|f| f.time == 0.0)
        .or_else(|| frames.first())
        .map(|f| f.poses.as_slice())
        .unwrap_or_default();
    for b in &smd.bones {
        let parent = smd.ids.get(&b.parent).copied();
        scene.add(Node::named(b.name.clone(), parent));
    }
    for &(bone, pose) in rest {
        let n = &mut scene.nodes[bone];
        n.translation = [pose[0], pose[1], pose[2]];
        n.rotation = euler(pose);
    }
    let bones = smd.bones.len();

    let mut facts = serde_json::Map::new();
    scene.info.format = Some((if vta { "VTA" } else { "SMD" }).into());
    scene.info.version = Some(smd.version.to_string());
    scene.info.up_axis = Fact::Assumed(Axis::PosZ);
    scene.info.handedness = Fact::Assumed(Handedness::Right);
    facts.insert("bones".into(), json!(bones));
    facts.insert("frames".into(), json!(frames.len()));

    if vta {
        let targets = vertex_animation(&mut scene, smd.shapes, &stem)?;
        facts.insert("morph_targets".into(), json!(targets));
    } else if !smd.triangles.is_empty() {
        let mut images = Images::new(!keep.image_bytes, input.dir());
        for name in &smd.materials {
            let mut m = Material::plain(name, [1.0; 4]);
            if keep.textures {
                m.color_image = images.find(assets, &[name], &[input.dir()]);
            }
            scene.materials.push(m);
        }
        scene.images = images.list;
        reference(&mut scene, &smd.ids, bones, &smd.triangles, &stem);
    }
    scene.dropped = smd.dropped;

    if !vta && frames.len() > 1 {
        let channels = animation(&frames, bones);
        if !channels.is_empty() {
            // The rate is the compiler script's to give; the file has none.
            scene.info.frame_rate = Fact::Assumed(FPS);
            scene.animations.push(Animation {
                name: stem,
                channels,
                extras: None,
            });
        }
    }
    ensure!(
        !scene.nodes.is_empty(),
        Empty,
        "SMD file has no skeleton, triangles or vertex animation"
    );
    scene.turn(scene::Z_UP_TO_Y_UP);
    scene.facts = facts;
    scene.finish(keep)
}

/// The rotation Euler angles `p[3..6]` (radians) make: about X, then Y, then Z.
fn euler(p: Pose) -> [f32; 4] {
    quat::normalize(quat::euler_xyz([p[3], p[4], p[5]].map(f64::from)))
}

/// Frames in time order, two frames at one time made one (the later poses winning), and each frame's poses in bone
/// order, a bone listed twice taking its last pose.
fn settle_frames(mut frames: Vec<Frame>) -> Vec<Frame> {
    frames.sort_by(|a, b| a.time.total_cmp(&b.time));
    let mut out: Vec<Frame> = Vec::with_capacity(frames.len());
    for f in frames {
        #[allow(clippy::float_cmp, reason = "frames at the very same time are one frame")]
        match out.last_mut() {
            Some(last) if last.time == f.time => last.poses.extend(f.poses),
            _ => out.push(f),
        }
    }
    for f in &mut out {
        f.poses.reverse();
        f.poses.sort_by_key(|p| p.0);
        f.poses.dedup_by_key(|p| p.0);
    }
    out
}

/// The skinned mesh of a reference file's triangles, one primitive per material, bound to the bones as they stand
/// at rest (nodes `0..bones`); without bones, a plain mesh.
fn reference(scene: &mut Scene, ids: &HashMap<i64, usize>, bones: usize, triangles: &[Triangle], name: &str) {
    struct Build {
        prim: Primitive,
        /// The first vertex with each position, normal and texture coordinate, by their bits.
        seen: FastMap<[u32; 8], u32>,
        /// The next vertex with the same position, normal and texture coordinate as each, differing in its links;
        /// `u32::MAX` after the last.
        next: Vec<u32>,
        /// Each vertex's run of `flat`, its joints and weights.
        links: Vec<(u32, u32)>,
        flat: Vec<(u32, f32)>,
    }
    let skinned = bones > 0;
    let mut builds: Vec<Build> = Vec::new();
    let mut prim_of: HashMap<usize, usize> = HashMap::new();
    let mut links: Vec<(u32, f32)> = Vec::new();
    for t in triangles {
        let b = *prim_of.entry(t.material).or_insert_with(|| {
            builds.push(Build {
                prim: Primitive {
                    material: Some(t.material),
                    uvs: vec![Vec::new()],
                    ..Primitive::default()
                },
                seen: FastMap::default(),
                next: Vec::new(),
                links: Vec::new(),
                flat: Vec::new(),
            });
            builds.len() - 1
        });
        let build = &mut builds[b];
        for v in &t.corners {
            links.clear();
            if skinned {
                influence(v, ids, &mut links);
            }
            let [x, y, z] = v.position.map(f32::to_bits);
            let [nx, ny, nz] = v.normal.map(f32::to_bits);
            let [s, t] = v.uv.map(f32::to_bits);
            let key = [x, y, z, nx, ny, nz, s, t];
            let next = u32_of(build.prim.positions.len());
            // A vertex is used again only when its links match bit for bit too.
            let same = |l: &[(u32, f32)]| {
                l.len() == links.len()
                    && l.iter()
                        .zip(&links)
                        .all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits())
            };
            let index = match build.seen.get(&key) {
                None => {
                    build.seen.insert(key, next);
                    next
                }
                Some(&first) => {
                    let mut at = first;
                    loop {
                        let (start, len) = build.links[at as usize];
                        if same(&build.flat[start as usize..][..len as usize]) {
                            break at;
                        }
                        let after = build.next[at as usize];
                        if after == u32::MAX {
                            build.next[at as usize] = next;
                            break next;
                        }
                        at = after;
                    }
                }
            };
            if index == next {
                build.next.push(u32::MAX);
                build.prim.positions.push(v.position);
                build.prim.normals.push(v.normal);
                build.prim.uvs[0].push([v.uv[0], 1.0 - v.uv[1]]);
                build.links.push((u32_of(build.flat.len()), u32_of(links.len())));
                build.flat.extend_from_slice(&links);
            }
            build.prim.indices.push(index);
        }
    }
    let mut prims = Vec::with_capacity(builds.len());
    for mut b in builds {
        if skinned {
            let width = b.links.iter().map(|l| l.1 as usize).max().unwrap_or(0).max(1);
            b.prim.influences = width;
            b.prim.joints = Vec::with_capacity(width * b.links.len());
            b.prim.weights = Vec::with_capacity(width * b.links.len());
            for &(start, len) in &b.links {
                let l = &b.flat[start as usize..][..len as usize];
                for k in 0..width {
                    let (j, w) = l.get(k).copied().unwrap_or((0, 0.0));
                    b.prim.joints.push(j);
                    b.prim.weights.push(w);
                }
            }
        }
        prims.push(b.prim);
    }
    scene.meshes.push(SceneMesh {
        name: name.to_owned(),
        prims,
        ..SceneMesh::default()
    });
    let mut node = Node::named(name, None);
    node.mesh = Some(scene.meshes.len() - 1);
    if skinned {
        let world = scene.world();
        let inverse_binds: Vec<M4> = world[..bones].iter().map(scene::invert).collect();
        scene.skins.push(Skin {
            name: name.to_owned(),
            joints: (0..bones).collect(),
            inverse_binds,
        });
        node.skin = Some(scene.skins.len() - 1);
    }
    scene.add(node);
}

/// A vertex's joints and weights: its links to known bones with weight, and its parent bone with whatever weight
/// they leave; a bone linked twice is one, and at most [`scene::MAX_INFLUENCES`] are kept, the heaviest.
fn influence(v: &Vertex, ids: &HashMap<i64, usize>, out: &mut Vec<(u32, f32)>) {
    let mut sum = 0.0f32;
    for &(bone, w) in &v.links {
        if let Some(&j) = ids.get(&bone)
            && w.is_finite()
            && w > 0.0
        {
            out.push((u32_of(j), w));
            sum += w;
        }
    }
    if let Some(&p) = ids.get(&v.parent)
        && sum < 1.0
    {
        out.push((u32_of(p), 1.0 - sum));
    }
    scene::strongest(out);
}

/// The bones' keys, frame by frame at the frames' own times (the shared writer has the animation start at 0, and
/// notes what it took off): a bone a frame leaves out holds its last pose.
fn animation(frames: &[Frame], bones: usize) -> Vec<Channel> {
    let time = |f: &Frame| f32_of(f.time / FPS);
    let mut keys: Vec<Vec<(usize, Pose)>> = (0..bones).map(|_| Vec::new()).collect();
    for (k, f) in frames.iter().enumerate() {
        for &(bone, pose) in &f.poses {
            keys[bone].push((k, pose));
        }
    }
    let mut channels = Vec::new();
    for (bone, list) in keys.into_iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let mut times = Vec::with_capacity(list.len());
        let mut moves = Vec::with_capacity(list.len() * 3);
        let mut turns = Vec::with_capacity(list.len() * 4);
        let mut push = |k: usize, p: Pose| {
            times.push(time(&frames[k]));
            moves.extend_from_slice(&p[..3]);
            turns.extend_from_slice(&euler(p));
        };
        let mut last: Option<(usize, Pose)> = None;
        for (k, pose) in list {
            if let Some((j, held)) = last
                && k > j + 1
            {
                push(k - 1, held);
            }
            push(k, pose);
            last = Some((k, pose));
        }
        channels.push(Channel {
            node: bone,
            path: Key::Translation,
            times: times.clone(),
            values: moves,
            interp: Interp::Linear,
        });
        channels.push(Channel {
            node: bone,
            path: Key::Rotation,
            times,
            values: turns,
            interp: Interp::Linear,
        });
    }
    channels
}

/// A `.vta` file's shapes as a point cloud and its morph targets; returns the number of targets.
fn vertex_animation(scene: &mut Scene, shapes: Vec<Shape>, name: &str) -> Result<usize> {
    let mut shapes = shapes;
    shapes.sort_by(|a, b| a.time.total_cmp(&b.time));
    let mut settled: Vec<Shape> = Vec::with_capacity(shapes.len());
    for s in shapes {
        #[allow(clippy::float_cmp, reason = "shapes at the very same time are one shape")]
        match settled.last_mut() {
            Some(last) if last.time == s.time => {
                last.vertices.extend(s.vertices);
                if last.name.is_none() {
                    last.name = s.name;
                }
            }
            _ => settled.push(s),
        }
    }
    let base_at = settled.iter().position(|s| s.time == 0.0).unwrap_or(0);
    let Some(base) = settled.get(base_at) else {
        bail!(Empty, "VTA file has no vertex animation");
    };
    let mut index: HashMap<i64, usize> = HashMap::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    for &(id, v) in &base.vertices {
        let next = positions.len();
        let i = *index.entry(id).or_insert(next);
        let (p, n) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
        if i == next {
            positions.push(p);
            normals.push(n);
        } else {
            positions[i] = p;
            normals[i] = n;
        }
    }
    ensure!(
        !positions.is_empty(),
        Empty,
        "VTA file has no vertices in its first frame"
    );
    let count = settled.len() - 1;
    ensure!(
        count.saturating_mul(positions.len()) <= MAX_TARGET_VALUES,
        LimitExceeded,
        "VTA file has too many frames of too many vertices to convert ({count} of {})",
        positions.len()
    );
    let mut targets = Vec::with_capacity(count);
    let mut names = Vec::with_capacity(count);
    for (k, s) in settled.iter().enumerate() {
        if k == base_at {
            continue;
        }
        let mut t = Target {
            positions: vec![[0.0; 3]; positions.len()],
            normals: vec![[0.0; 3]; positions.len()],
        };
        for &(id, v) in &s.vertices {
            if let Some(&i) = index.get(&id) {
                t.positions[i] = std::array::from_fn(|c| v[c] - positions[i][c]);
                t.normals[i] = std::array::from_fn(|c| v[c + 3] - normals[i][c]);
            }
        }
        // A shape that leaves the normals as they are has no normal offsets.
        if t.normals.iter().flatten().all(|&c| c == 0.0) {
            t.normals.clear();
        }
        targets.push(t);
        names.push(s.name.clone().unwrap_or_else(|| format!("frame{}", s.time)));
    }
    let n = positions.len();
    let prim = Primitive {
        positions,
        normals,
        indices: (0..u32_of(n)).collect(),
        points: true,
        targets,
        ..Primitive::default()
    };
    scene.meshes.push(SceneMesh {
        name: name.to_owned(),
        prims: vec![prim],
        target_names: names,
        ..SceneMesh::default()
    });
    let mut node = Node::named(name, None);
    node.mesh = Some(scene.meshes.len() - 1);
    scene.add(node);
    Ok(count)
}

/// The lines with something on them, by line number (from 1), trimmed; comment lines are left out.
fn lines(text: &str) -> Vec<(usize, &str)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut out = Vec::new();
    for (i, line) in text.split('\n').enumerate() {
        for part in line.split('\r') {
            let t = part.trim();
            if t.is_empty() || t.starts_with("//") || t.starts_with('#') || t.starts_with(';') {
                continue;
            }
            out.push((i + 1, t));
        }
    }
    out
}

/// A line's words, split where `str::split_whitespace` splits them: an ASCII line (nearly every one) by its bytes.
enum Words<'a> {
    Ascii(&'a str),
    Other(std::str::SplitWhitespace<'a>),
}

fn words_of(line: &str) -> Words<'_> {
    if line.is_ascii() {
        Words::Ascii(line)
    } else {
        Words::Other(line.split_whitespace())
    }
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self {
            Words::Ascii(rest) => {
                // The ASCII characters `char::is_whitespace` takes.
                let space = |c: &u8| matches!(c, b'\t'..=b'\r' | b' ');
                let b = rest.as_bytes();
                let Some(start) = b.iter().position(|c| !space(c)) else {
                    *rest = "";
                    return None;
                };
                let end = b[start..].iter().position(space).map_or(b.len(), |k| start + k);
                let word = &rest[start..end];
                *rest = &rest[end..];
                Some(word)
            }
            Words::Other(w) => w.next(),
        }
    }
}

/// A line without the comment that ends it.
fn code(line: &str) -> &str {
    let b = line.as_bytes();
    let mut from = 0;
    while let Some(k) = b[from..].iter().position(|&c| c == b'/') {
        let i = from + k;
        if b.get(i + 1) == Some(&b'/') {
            return line[..i].trim_end();
        }
        from = i + 1;
    }
    line
}

/// Whether the line ends a section.
fn is_end(line: &str) -> bool {
    // Lines come trimmed: most start with something other than an `e`, which settles it.
    line.as_bytes().first().is_some_and(|c| c.eq_ignore_ascii_case(&b'e')) && code(line).eq_ignore_ascii_case("end")
}

/// The finite numbers of `words`, as many as parse in a row.
fn numbers<'a>(words: impl Iterator<Item = &'a str>) -> Vec<f32> {
    words
        .map_while(|w| f32_from(w.as_bytes()).filter(|x| x.is_finite()))
        .collect()
}

/// An integer, written plainly or as a whole float (`3.000000`).
fn integer(w: &str) -> Option<i64> {
    stream::int(w.as_bytes()).or_else(|| {
        let f = f64_from(w.as_bytes())?;
        #[allow(clippy::cast_possible_truncation, reason = "checked to be a whole number in range")]
        (f.fract() == 0.0 && f.abs() < 9.0e15).then_some(f as i64)
    })
}

/// Reads the file's sections; more than `limit` triangles is an error.
fn parse(text: &str, limit: usize) -> Result<Smd> {
    let lines = lines(text);
    let mut s = Smd::default();
    let Some(&(n, first)) = lines.first() else {
        bail!(Empty, "SMD file is empty");
    };
    let mut words = words_of(code(first));
    ensure!(
        words.next().is_some_and(|w| w.eq_ignore_ascii_case("version")),
        UnknownFormat,
        "isn't an SMD file: it doesn't start with a version line"
    );
    s.version = match words.next().and_then(integer) {
        Some(v) => v,
        None => bail!(Corrupt, "SMD line {n}: the version isn't a number"),
    };
    let mut i = 1;
    while i < lines.len() {
        let (n, line) = lines[i];
        i += 1;
        let head = words_of(code(line)).next().unwrap_or("").to_ascii_lowercase();
        i = match head.as_str() {
            "nodes" => nodes(&lines, i, &mut s)?,
            "skeleton" => skeleton(&lines, i, &mut s)?,
            "triangles" => triangles(&lines, i, &mut s, limit)?,
            "vertexanimation" => vertex_frames(&lines, i, &mut s)?,
            "version" => bail!(Corrupt, "SMD line {n}: a second version line"),
            // A section this reader doesn't know: passed over to its end.
            _ => lines[i..]
                .iter()
                .position(|l| is_end(l.1))
                .map_or(lines.len(), |k| i + k + 1),
        };
    }
    Ok(s)
}

/// The `nodes` section from line `i`: `id "name" parent` each. Returns the line after its end.
fn nodes(lines: &[(usize, &str)], mut i: usize, s: &mut Smd) -> Result<usize> {
    while let Some(&(n, line)) = lines.get(i) {
        i += 1;
        if is_end(line) {
            break;
        }
        let Some((id, name, parent)) = node_line(code(line)) else {
            bail!(Corrupt, "SMD line {n}: a node needs an id, a name and a parent");
        };
        if s.ids.contains_key(&id) {
            continue;
        }
        s.ids.insert(id, s.bones.len());
        s.bones.push(Bone { name, parent });
    }
    Ok(i)
}

/// `id "name" parent`, the name quoted or a single word.
fn node_line(t: &str) -> Option<(i64, String, i64)> {
    let (id, rest) = t.split_once(char::is_whitespace)?;
    let id = integer(id)?;
    let rest = rest.trim_start();
    let (name, rest) = match rest.strip_prefix('"') {
        Some(r) => {
            let e = r.find('"')?;
            (&r[..e], &r[e + 1..])
        }
        None => rest.split_once(char::is_whitespace)?,
    };
    let parent = integer(words_of(rest).next()?)?;
    Some((id, name.to_owned(), parent))
}

/// The `skeleton` section from line `i`: `time n` lines, each followed by `bone x y z rx ry rz` lines. Returns the
/// line after its end.
fn skeleton(lines: &[(usize, &str)], mut i: usize, s: &mut Smd) -> Result<usize> {
    while let Some(&(n, line)) = lines.get(i) {
        i += 1;
        if is_end(line) {
            break;
        }
        let mut words = words_of(code(line));
        let first = words.next().unwrap_or("");
        if first.eq_ignore_ascii_case("time") {
            let Some(time) = words
                .next()
                .and_then(|w| f64_from(w.as_bytes()))
                .filter(|t| t.is_finite())
            else {
                bail!(Corrupt, "SMD line {n}: the time isn't a number");
            };
            s.frames.push(Frame {
                time,
                poses: Vec::new(),
            });
            continue;
        }
        let v = numbers(words);
        let (Some(bone), true) = (integer(first), v.len() >= 6) else {
            bail!(
                Corrupt,
                "SMD line {n}: a bone's pose needs its id, a position and a rotation"
            );
        };
        let Some(&b) = s.ids.get(&bone) else {
            continue;
        };
        if s.frames.is_empty() {
            // A pose before any time line is at time 0.
            s.frames.push(Frame {
                time: 0.0,
                poses: Vec::new(),
            });
        }
        if let Some(f) = s.frames.last_mut() {
            f.poses.push((b, [v[0], v[1], v[2], v[3], v[4], v[5]]));
        }
    }
    Ok(i)
}

/// The `triangles` section from line `i`: a material line and three vertex lines each. Returns the line after its
/// end.
fn triangles(lines: &[(usize, &str)], mut i: usize, s: &mut Smd, limit: usize) -> Result<usize> {
    let mut by_name: FastMap<String, usize> = s.materials.iter().enumerate().map(|(k, m)| (m.clone(), k)).collect();
    // Triangles run in long stretches of one material.
    let mut last: Option<(&str, usize)> = None;
    while let Some(&(_, line)) = lines.get(i) {
        i += 1;
        if is_end(line) {
            return Ok(i);
        }
        let mut corners: [Option<Vertex>; 3] = [None, None, None];
        for c in &mut corners {
            let Some(&(n, v)) = lines.get(i) else { break };
            if is_end(v) {
                break;
            }
            i += 1;
            match vertex(code(v)) {
                Some(v) => *c = Some(v),
                None => bail!(
                    Corrupt,
                    "SMD line {n}: a vertex needs its parent bone, position and normal"
                ),
            }
        }
        let [Some(a), Some(b), Some(c)] = corners else {
            s.dropped += 1;
            continue;
        };
        let corners = [a, b, c];
        ensure!(
            s.triangles.len() < limit,
            LimitExceeded,
            "too many triangles to convert (more than {limit})"
        );
        let material = if let Some((name, m)) = last
            && name == line
        {
            m
        } else if let Some(&m) = by_name.get(line) {
            m
        } else {
            s.materials.push(line.to_owned());
            by_name.insert(line.to_owned(), s.materials.len() - 1);
            s.materials.len() - 1
        };
        last = Some((line, material));
        s.triangles.push(Triangle { material, corners });
    }
    Ok(i)
}

/// `parent x y z nx ny nz u v [links (bone weight)...]`; the texture coordinates and links are optional.
fn vertex(t: &str) -> Option<Vertex> {
    let mut words = words_of(t);
    let parent = integer(words.next()?)?;
    let mut f = [0.0f32; 8];
    let mut got = 0;
    for slot in &mut f {
        match words
            .next()
            .and_then(|w| f32_from(w.as_bytes()))
            .filter(|x| x.is_finite())
        {
            Some(x) => {
                *slot = x;
                got += 1;
            }
            None => break,
        }
    }
    if got < 6 {
        return None;
    }
    let mut links = Vec::new();
    if got == 8
        && let Some(count) = words.next().and_then(integer)
    {
        for _ in 0..count.max(0) {
            let (Some(b), Some(w)) = (
                words.next().and_then(integer),
                words.next().and_then(|w| f32_from(w.as_bytes())),
            ) else {
                break;
            };
            links.push((b, w));
        }
    }
    Some(Vertex {
        parent,
        position: [f[0], f[1], f[2]],
        normal: [f[3], f[4], f[5]],
        uv: [f[6], f[7]],
        links,
    })
}

/// The `vertexanimation` section from line `i`: `time n` lines (a `#` comment after one names it), each followed by
/// `id x y z nx ny nz` lines. Returns the line after its end.
fn vertex_frames(lines: &[(usize, &str)], mut i: usize, s: &mut Smd) -> Result<usize> {
    while let Some(&(n, line)) = lines.get(i) {
        i += 1;
        if is_end(line) {
            break;
        }
        let mut words = words_of(code(line));
        let first = words.next().unwrap_or("");
        if first.eq_ignore_ascii_case("time") {
            let Some(time) = words
                .next()
                .and_then(|w| f64_from(w.as_bytes()))
                .filter(|t| t.is_finite())
            else {
                bail!(Corrupt, "SMD line {n}: the time isn't a number");
            };
            let name = line
                .find('#')
                .or_else(|| line.find("//").map(|k| k + 1))
                .map(|k| line[k + 1..].trim().to_owned())
                .filter(|x| !x.is_empty());
            s.shapes.push(Shape {
                time,
                name,
                vertices: Vec::new(),
            });
            continue;
        }
        let v = numbers(words);
        let (Some(id), true) = (integer(first), v.len() >= 3) else {
            bail!(Corrupt, "SMD line {n}: a vertex needs its id and position");
        };
        let normal = if v.len() >= 6 { [v[3], v[4], v[5]] } else { [0.0; 3] };
        if s.shapes.is_empty() {
            s.shapes.push(Shape {
                time: 0.0,
                name: None,
                vertices: Vec::new(),
            });
        }
        if let Some(shape) = s.shapes.last_mut() {
            shape
                .vertices
                .push((id, [v[0], v[1], v[2], normal[0], normal[1], normal[2]]));
        }
    }
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use serde_json::Value;

    fn read(text: &str, name: &str) -> Result<Document> {
        let ext = name.rsplit('.').next().unwrap_or("");
        load(
            super::super::Input::memory(text.as_bytes(), name),
            ext,
            &Memory::new(),
            Keep::ALL,
        )
    }

    fn error(text: &str) -> String {
        match read(text, "t.smd") {
            Ok(_) => panic!("read without error"),
            Err(e) => e.to_string(),
        }
    }

    const REFERENCE: &str = "version 1
nodes
0 \"root bone\" -1
1 \"child\" 0
end
skeleton
time 0
0 0 0 0 0 0 0
1 0 0 2 0 0 1.5707963
end
triangles
skin.bmp
0 0 0 0 0 0 1 0 0
1 1 0 0 0 0 1 1 0 1 0 0.25
1 0 1 0 0 0 1 0 1 2 0 0.5 1 0.5
end
";

    #[test]
    fn a_reference_file_is_a_skinned_mesh() {
        let doc = read(REFERENCE, "t.smd").unwrap();
        let j = &doc.json();
        assert_eq!(j["skins"].as_array().unwrap().len(), 1);
        assert_eq!(j["skins"][0]["joints"].as_array().unwrap().len(), 2);
        assert_eq!(j["nodes"][0]["name"], "root bone");
        assert_eq!(doc.facts["bones"], json!(2));
        assert_eq!(doc.facts["triangles"], json!(1));
        assert!(j.get("animations").is_none());
        // Turned Z up to Y up: the child's (0, 0, 2) is up.
        let mesh = doc.flatten(crate::document::Pose::Rest).unwrap().unwrap();
        let p = &mesh.prims[0];
        assert_eq!(p.positions.len(), 3);
        assert!(
            p.positions
                .iter()
                .any(|v| (v[0] - 1.0).abs() < 1e-5 && v[1].abs() < 1e-5)
        );
        assert!(
            p.positions
                .iter()
                .any(|v| v[1].abs() < 1e-5 && (v[2] + 1.0).abs() < 1e-5)
        );
        // UV v flipped: 0 at the bottom becomes 1.
        assert!(p.uvs.iter().any(|uv| (uv[1] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn links_take_their_weight_and_the_parent_the_rest() {
        let v = vertex("1 0 0 0 0 0 1 0 0 2 0 0.25 1 0.5").unwrap();
        let ids: HashMap<i64, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let mut l = Vec::new();
        influence(&v, &ids, &mut l);
        l.sort_by_key(|x| x.0);
        assert_eq!(l.len(), 2);
        assert!((l[0].1 - 0.25).abs() < 1e-6);
        assert!((l[1].1 - 0.75).abs() < 1e-6);
        // A link to a missing bone is dropped; its weight goes to the parent.
        let v = vertex("0 0 0 0 0 0 1 0 0 1 7 0.5").unwrap();
        l.clear();
        influence(&v, &ids, &mut l);
        assert_eq!(l, vec![(0, 1.0)]);
        // A huge count with nothing after it ends with the line.
        let v = vertex("0 0 0 0 0 0 1 0 0 999999999999 1 0.5").unwrap();
        assert_eq!(v.links.len(), 1);
    }

    #[test]
    fn frames_make_an_animation_at_thirty_a_second() {
        let text = "version 1
nodes
0 root -1
1 arm 0
end
skeleton
time 0
0 0 0 0 0 0 0
1 1 0 0 0 0 0
time 1
0 0 0 1 0 0 0
time 2
0 0 0 2 0 0 0
1 1 0 0 0 0 1
end
";
        let doc = read(text, "walk.smd").unwrap();
        assert!(doc.flatten(crate::document::Pose::Rest).unwrap().is_none());
        assert_eq!(doc.facts["frames"], json!(3));
        let rig = doc.rig().unwrap();
        let a = &rig.animations[0];
        let arm = rig.find("arm").unwrap();
        let c = a
            .channels
            .iter()
            .find(|c| c.node == arm && c.property == crate::rig::Property::Rotation)
            .unwrap();
        // Held at frame 1, where the file leaves it out.
        assert_eq!(c.times.len(), 3);
        assert!((c.times[2] - 2.0 / 30.0).abs() < 1e-6);
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("SMD"));
        assert_eq!(info.version.as_deref(), Some("1"));
        assert_eq!(info.frame_rate, crate::source::Fact::Assumed(30.0));
        assert_eq!(info.up_axis, crate::source::Fact::Assumed(crate::source::Axis::PosZ));
        assert_eq!(info.meters_per_unit, crate::source::Fact::Unknown);
        assert!(info.converted.up_axis);
    }

    #[test]
    fn a_vta_file_is_points_with_morph_targets() {
        let text = "version 1
nodes
0 \"root\" -1
end
skeleton
time 0
0 0 0 0 0 0 0
time 1
time 2
end
vertexanimation
time 0
0 0 0 0 0 0 1
1 1 0 0 0 0 1
time 1 # smile
1 1 0 1 0 0 1
time 2
0 0 0 -1 0 0 1
end
";
        let doc = read(text, "face.vta").unwrap();
        let j = &doc.json();
        let m = &j["meshes"][0];
        assert_eq!(m["primitives"][0]["mode"], json!(0));
        assert_eq!(m["primitives"][0]["targets"].as_array().unwrap().len(), 2);
        assert_eq!(m["extras"]["targetNames"], json!(["smile", "frame2"]));
        assert_eq!(doc.facts["morph_targets"], json!(2));
        assert!(j.get("animations").is_none());
    }

    #[test]
    fn cut_short_and_broken_files() {
        assert!(error("").contains("empty"));
        assert!(error("solid cube\nfacet normal 0 0 1\n").contains("isn't an SMD"));
        assert!(error("version x\n").contains("version"));
        assert!(error("version 1\nnodes\n0\nend\n").contains("node"));
        assert!(error("version 1\nnodes\n0 a -1\nend\nskeleton\ntime nan\n").contains("time"));
        assert!(error("version 1\nnodes\n0 a -1\nend\nskeleton\ntime 0\n0 1 2\n").contains("pose"));
        assert!(error("version 1\ntriangles\nm\n0 1 2 3\n").contains("vertex"));
        assert!(error("version 1\n").contains("no skeleton"));
        // A file cut off mid-triangle keeps what it finished.
        let cut = &REFERENCE[..REFERENCE.find("1 0 1 0 0 0 1").unwrap()];
        let doc = read(cut, "t.smd");
        let doc = doc.unwrap();
        assert_eq!(doc.json()["asset"]["extras"]["source"]["droppedGeometry"], json!(1));
        // Every prefix reads or fails without panicking.
        for k in 0..REFERENCE.len() {
            if REFERENCE.is_char_boundary(k) {
                let _ = read(&REFERENCE[..k], "t.smd");
            }
        }
    }

    #[test]
    fn parent_loops_and_unknown_bones_do_not_hang() {
        let text = "version 1
nodes
0 a 1
1 b 0
2 c 2
end
skeleton
time 0
0 1 0 0 0 0 0
9 1 0 0 0 0 0
end
triangles
m
5 0 0 0 0 0 1 0 0
0 1 0 0 0 0 1 0 0 1 2 0.5
1 0 1 0 0 0 1 0 0
end
";
        let doc = read(text, "t.smd").unwrap();
        assert_eq!(doc.facts["bones"], json!(3));
        assert!(
            doc.json()["skins"][0]["joints"]
                .as_array()
                // The three bones, and the mesh's own node for the vertex whose parent bone doesn't exist.
                .is_some_and(|j| j.len() == 4)
        );
    }

    #[test]
    fn crlf_tabs_comments_and_unknown_sections() {
        let text = "// exported\r\nversion 1\r\nnodes\r\n0\t\"root\"\t-1\r\nend\r\nflexcontrol\r\nx y z\r\nend\r\nskeleton\r\ntime 0 // rest\r\n0 0 0 0 0 0 0\r\nend\r\ntriangles\r\nmat\r\n0 0 0 0 0 0 1 0 0\r\n0 1 0 0 0 0 1 1 0\r\n0 0 1 0 0 0 1 0 1\r\nend\r\n";
        let doc = read(text, "t.smd").unwrap();
        assert_eq!(doc.facts["triangles"], json!(1));
        assert_eq!(doc.json()["materials"][0]["name"], Value::from("mat"));
    }

    #[test]
    fn numbers_read_as_the_standard_parser_reads_them() {
        let bits = |v: Option<f32>| v.map(f32::to_bits);
        for s in [
            "0",
            "1",
            "-1",
            "+1",
            "1.5",
            "-0",
            "+.5",
            ".5",
            "5.",
            "1e3",
            "1E-1",
            "-2.5E+2",
            "007",
            "0.000001",
            "3.000000",
            "-2.0",
            "1e2",
            "9223372036854775807",
            "-9223372036854775808",
            "9223372036854775808",
            "123456789012345678901234",
            "inf",
            "nan",
            "1.#INF",
            "",
            "x",
            "1,5",
            "0x10",
        ] {
            assert_eq!(bits(f32_from(s.as_bytes())), bits(s.parse().ok()), "{s:?}");
            assert_eq!(
                f64_from(s.as_bytes()).map(f64::to_bits),
                s.parse::<f64>().ok().map(f64::to_bits),
                "{s:?}"
            );
            let old = s.parse::<i64>().ok().or_else(|| {
                let f = s.parse::<f64>().ok()?;
                #[allow(clippy::cast_possible_truncation, reason = "checked to be a whole number in range")]
                (f.fract() == 0.0 && f.abs() < 9.0e15).then_some(f as i64)
            });
            assert_eq!(integer(s), old, "{s:?}");
        }
        // And in a file: a triangle whose numbers are spelled every which way.
        let text = "version 1
nodes
+0 \"root\" -1
end
skeleton
time 0.
0 +0 .0 0E0 0 0 0
end
triangles
mat
0 0 0 0 0 0 1 0 0
+0 1. +0 -0 0 0 1E+0 1.0 0 1 +0 1.000000
000 0 1e0 0 0 0 1 0 1
end
";
        let doc = read(text, "t.smd").unwrap();
        assert_eq!(doc.facts["triangles"], json!(1));
    }
}
