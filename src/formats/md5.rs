//! MD5 (the Doom 3 engine's text model formats) as a glTF document: `.md5mesh` skeletal meshes and `.md5anim`
//! skeletal animations.
//!
//! Both are text: `MD5Version 10`, then `keyword value` lines and `{ }` blocks, with `//` and `/* */` comments,
//! strings in double quotes and vectors in parentheses. A file is read by what its sections are, so an `.md5anim`
//! named `.md5mesh` still reads as an animation; a file that doesn't start with `MD5Version` isn't MD5.
//!
//! An `.md5mesh` gives its skeleton in object space: each joint of `joints { "name" parent ( x y z ) ( qx qy qz ) }`
//! has its position and orientation in the model's frame, not its parent's. The quaternion's `w` is left out and
//! taken as `-sqrt(1 - x² - y² - z²)` (0 when that's negative), the engine's convention. The joints become nodes
//! with their transforms made relative to their parents, a skin over all of them whose inverse bind matrices undo
//! each joint's object-space transform, and one mesh node with that skin.
//!
//! Each `mesh { }` block is a primitive with its own material. Its vertices don't hold positions: `vert i ( u v )
//! start count` names `count` weights from `start`, and each `weight i joint bias ( x y z )` is an offset in its
//! joint's frame. A vertex's position is the bias-weighted sum of its weights' offsets carried into object space by
//! their joints, and its joints and weights are its weights' joints and biases (repeated joints merged, the
//! sixteen strongest kept). Texture coordinates have their origin at the top left, as glTF's do, so they're kept
//! as written. The format has no normals: each vertex gets the area-weighted normal of the triangles around its
//! position (so a seam where texture coordinates split doesn't show). Triangles are wound clockwise seen from the
//! front, so each is turned around for glTF.
//!
//! A mesh's `shader` names a Doom 3 material, mostly a path without an extension (`models/monsters/imp/imp`): its
//! texture is looked for as the path itself when it has an extension, then with `.tga`, then as `<path>_d.tga`,
//! the engine's diffuse map name (the assets find the same name in other image formats, and by file name in other
//! folders). A mesh with no triangles is left out.
//!
//! An `.md5anim` gives the skeleton in each joint's parent's frame: `hierarchy { "name" parent flags start }` and
//! `baseframe { ( x y z ) ( qx qy qz ) }` are the joints and their base pose. Each `frame k { ... }` lists
//! `numAnimatedComponents` numbers: a joint's `flags` bits 1, 2, 4, 8, 16 and 32 say which of its position X, Y, Z
//! and quaternion X, Y, Z the frame replaces, in that order from its `start`; the rest keep the base frame's. The
//! nodes' rest pose is the first frame's, so the base frame gives only what no frame animates: the engine never
//! reads the rest, and some exporters write the base frame's quaternions with the other sign of `w` from their
//! frames'. Frame k is at k over `frameRate` seconds (24 when the rate isn't a positive number, as the engine
//! plays them, an assumed `frameRate` in the source info). The frames become one linear animation, named after the file: a translation
//! channel for each joint with a position bit, a rotation channel for each with a quaternion bit. Per-frame
//! `bounds` are dropped. A mesh and its animations are separate files, each read on its own; both share the joint
//! names that tie them together.
//!
//! The format is Z up by the engine's convention, so the scene's roots are turned for glTF's Y up (the source info's
//! `upAxis` is an assumed `+Z`, converted); game units are kept unscaled, and their length unknown. The exporter's
//! `commandline` is kept as written.
//!
//! A file cut short, a number that isn't one, or an index past what its block holds (a vertex's weights, a weight's
//! joint, a triangle's vertex, a joint's parent, a frame's components) is an error. Counts are checked against the
//! file's size before anything is allocated for them.

use std::collections::HashMap;

use serde_json::{Map, json};

use super::Input;
use super::scene::{
    self, Animation, Channel, Interp, M4, MAX_INFLUENCES, Node, Path, Primitive, Scene, SceneMesh, Skin, Z_UP_TO_Y_UP,
};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::Material;
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact};

/// The frame rate taken when a file gives none that's usable: the engine's.
const DEFAULT_RATE: f32 = 24.0;

/// Reads an `.md5mesh` or `.md5anim` model.
pub(crate) fn load(input: Input<'_>, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
    let data = input.read()?;
    let file = parse(&data)?;
    let stem = input
        .path
        .file_stem()
        .map_or_else(|| "md5".to_owned(), |s| s.to_string_lossy().into_owned());
    let anim = match (file.mesh.is_some(), file.anim.is_some()) {
        (true, false) => false,
        (false, true) => true,
        // Both kinds of section, or neither: the extension decides.
        (true, true) => ext.eq_ignore_ascii_case("md5anim"),
        (false, false) => bail!(Empty, "MD5 file has no joints, meshes or hierarchy"),
    };
    let mut scene = if anim {
        build_anim(&file.anim.unwrap_or_default(), &stem)?
    } else {
        build_mesh(&file.mesh.unwrap_or_default(), &stem, &input, assets, keep)?
    };
    scene.info.format = Some("MD5".into());
    scene.info.version = Some(file.version.to_string());
    scene.info.up_axis = Fact::Assumed(Axis::PosZ);
    if !file.commandline.is_empty() {
        scene.source.insert("commandline".into(), json!(file.commandline));
    }
    scene.rotate_roots(Z_UP_TO_Y_UP);
    scene.finish(keep)
}

/// A token: a bare word (a keyword or a number), a quoted string, or one of `{ } ( )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tok<'a> {
    Word(&'a [u8]),
    Str(&'a [u8]),
    Sym(u8),
}

/// Splits MD5 text into tokens, counting lines for errors.
struct Lex<'a> {
    b: &'a [u8],
    at: usize,
    line: usize,
}

impl<'a> Lex<'a> {
    fn new(b: &'a [u8]) -> Self {
        let b = b.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(b);
        Self { b, at: 0, line: 1 }
    }

    fn err(&self, what: &str) -> crate::Error {
        err!(Corrupt, "MD5 file, line {}: {what}", self.line)
    }

    /// Moves past `n` bytes, counting the line breaks among them.
    fn advance(&mut self, n: usize) {
        let end = (self.at + n).min(self.b.len());
        #[allow(clippy::naive_bytecount, reason = "short runs, and no dependency for it")]
        let breaks = self.b[self.at..end].iter().filter(|&&c| c == b'\n').count();
        self.line += breaks;
        self.at = end;
    }

    fn next(&mut self) -> Result<Option<Tok<'a>>> {
        loop {
            let rest = &self.b[self.at..];
            let space = rest.iter().take_while(|c| c.is_ascii_whitespace() || **c == 0).count();
            if space > 0 {
                self.advance(space);
                continue;
            }
            if rest.starts_with(b"//") {
                let n = rest.iter().position(|&c| c == b'\n').unwrap_or(rest.len());
                self.advance(n);
                continue;
            }
            if rest.starts_with(b"/*") {
                let n = rest
                    .windows(2)
                    .skip(2)
                    .position(|w| w == b"*/")
                    .map_or(rest.len(), |p| p + 4);
                self.advance(n);
                continue;
            }
            break;
        }
        let rest = &self.b[self.at..];
        let Some(&c) = rest.first() else {
            return Ok(None);
        };
        match c {
            b'{' | b'}' | b'(' | b')' => {
                self.advance(1);
                Ok(Some(Tok::Sym(c)))
            }
            b'"' => {
                let Some(end) = rest[1..].iter().position(|&c| c == b'"' || c == b'\n') else {
                    bail!(
                        Truncated,
                        "MD5 file, line {}: a string runs to the end of the file",
                        self.line
                    );
                };
                if rest[1 + end] != b'"' {
                    return Err(self.err("a string runs past the end of its line"));
                }
                self.advance(end + 2);
                Ok(Some(Tok::Str(&rest[1..=end])))
            }
            _ => {
                let mut n = 0;
                while n < rest.len() {
                    let c = rest[n];
                    if c.is_ascii_whitespace() || c == 0 || b"{}()\"".contains(&c) || rest[n..].starts_with(b"//") {
                        break;
                    }
                    n += 1;
                }
                self.advance(n);
                Ok(Some(Tok::Word(&rest[..n])))
            }
        }
    }

    fn peek(&mut self) -> Result<Option<Tok<'a>>> {
        let at = (self.at, self.line);
        let t = self.next();
        (self.at, self.line) = at;
        t
    }

    fn need(&mut self) -> Result<Tok<'a>> {
        self.next()?.ok_or_else(|| err!(Truncated, "MD5 file is cut short"))
    }

    fn sym(&mut self, s: u8) -> Result<()> {
        match self.need()? {
            Tok::Sym(c) if c == s => Ok(()),
            _ => Err(self.err(&format!("expected '{}'", char::from(s)))),
        }
    }

    fn word(&mut self) -> Result<&'a [u8]> {
        match self.need()? {
            Tok::Word(w) => Ok(w),
            _ => Err(self.err("expected a number or keyword")),
        }
    }

    fn num(&mut self) -> Result<f64> {
        let w = self.word()?;
        std::str::from_utf8(w)
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .ok_or_else(|| self.err(&format!("{:?} isn't a number", String::from_utf8_lossy(w))))
    }

    fn int(&mut self) -> Result<i64> {
        let w = self.word()?;
        std::str::from_utf8(w)
            .ok()
            .and_then(|s| s.parse::<i64>().ok())
            .ok_or_else(|| self.err(&format!("{:?} isn't a whole number", String::from_utf8_lossy(w))))
    }

    /// A count or index: a whole number that isn't negative.
    fn index(&mut self) -> Result<usize> {
        let v = self.int()?;
        usize::try_from(v).map_err(|_| self.err(&format!("{v} can't be a count or index")))
    }

    fn string(&mut self) -> Result<String> {
        match self.need()? {
            Tok::Str(s) => Ok(scene::text(s)),
            _ => Err(self.err("expected a quoted string")),
        }
    }

    /// `( a b c )`.
    fn vec3(&mut self) -> Result<[f64; 3]> {
        self.sym(b'(')?;
        let v = [self.num()?, self.num()?, self.num()?];
        self.sym(b')')?;
        Ok(v)
    }

    /// A count the file is about to list: refused when the rest of the file can't hold that many entries of at
    /// least `size` bytes each.
    fn room(&self, n: usize, size: usize) -> Result<usize> {
        if n.saturating_mul(size) > self.b.len() - self.at {
            bail!(
                Truncated,
                "MD5 file, line {}: a count of {n} is more than the file holds",
                self.line
            );
        }
        Ok(n)
    }

    /// Skips a block whose `{` was just read.
    fn skip_block(&mut self) -> Result<()> {
        let mut depth = 1usize;
        while depth > 0 {
            match self.need()? {
                Tok::Sym(b'{') => depth += 1,
                Tok::Sym(b'}') => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }
}

/// The quaternion `[x, y, z, w]` with `w` found from the rest, as the format leaves it out.
fn quat(xyz: [f64; 3]) -> [f64; 4] {
    let t = 1.0 - xyz[0] * xyz[0] - xyz[1] * xyz[1] - xyz[2] * xyz[2];
    let w = if t > 0.0 { -t.sqrt() } else { 0.0 };
    [xyz[0], xyz[1], xyz[2], w]
}

fn conj(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// What a file holds.
#[derive(Default)]
struct File {
    version: i64,
    commandline: String,
    mesh: Option<MeshFile>,
    anim: Option<AnimFile>,
}

#[derive(Default)]
struct MeshFile {
    joints: Vec<Joint>,
    meshes: Vec<MeshBlock>,
}

/// A joint of an `.md5mesh`, in object space.
struct Joint {
    name: String,
    parent: i64,
    pos: [f64; 3],
    rot: [f64; 4],
}

#[derive(Default)]
struct MeshBlock {
    shader: String,
    verts: Vec<Vert>,
    tris: Vec<[usize; 3]>,
    weights: Vec<Weight>,
}

#[derive(Clone, Copy, Default)]
struct Vert {
    uv: [f32; 2],
    start: usize,
    count: usize,
}

#[derive(Clone, Copy, Default)]
struct Weight {
    joint: usize,
    bias: f64,
    pos: [f64; 3],
}

#[derive(Default)]
struct AnimFile {
    frame_rate: Option<f64>,
    components: Option<usize>,
    /// Name, parent, flags, first component.
    hierarchy: Vec<(String, i64, u32, usize)>,
    /// Position and quaternion X, Y, Z.
    base: Vec<[f64; 6]>,
    frames: Vec<Vec<f32>>,
}

fn parse(data: &[u8]) -> Result<File> {
    let mut lex = Lex::new(data);
    match lex.next()? {
        Some(Tok::Word(b"MD5Version")) => {}
        _ => bail!(UnknownFormat, "not an MD5 file: it doesn't start with MD5Version"),
    }
    let mut file = File {
        version: lex.int()?,
        ..File::default()
    };
    while let Some(tok) = lex.next()? {
        let Tok::Word(key) = tok else {
            if tok == Tok::Sym(b'{') {
                lex.skip_block()?;
                continue;
            }
            return Err(lex.err("expected a keyword"));
        };
        match key {
            b"commandline" => file.commandline = lex.string()?,
            // The counts are checked against what the blocks hold, which is what's read.
            b"numJoints" | b"numMeshes" | b"numFrames" => {
                lex.int()?;
            }
            b"frameRate" => file.anim.get_or_insert_default().frame_rate = Some(lex.num()?),
            b"numAnimatedComponents" => {
                let n = lex.index()?;
                file.anim.get_or_insert_default().components = Some(n);
            }
            b"joints" => {
                lex.sym(b'{')?;
                let m = file.mesh.get_or_insert_default();
                loop {
                    match lex.need()? {
                        Tok::Sym(b'}') => break,
                        Tok::Str(s) => {
                            let name = scene::text(s);
                            let parent = lex.int()?;
                            let pos = lex.vec3()?;
                            let rot = quat(lex.vec3()?);
                            m.joints.push(Joint { name, parent, pos, rot });
                        }
                        _ => return Err(lex.err("expected a joint's name or '}'")),
                    }
                }
            }
            b"mesh" => {
                lex.sym(b'{')?;
                let block = mesh_block(&mut lex)?;
                file.mesh.get_or_insert_default().meshes.push(block);
            }
            b"hierarchy" => {
                lex.sym(b'{')?;
                let a = file.anim.get_or_insert_default();
                loop {
                    match lex.need()? {
                        Tok::Sym(b'}') => break,
                        Tok::Str(s) => {
                            let name = scene::text(s);
                            let parent = lex.int()?;
                            let flags = lex.int()?;
                            let start = lex.index()?;
                            let flags = u32::try_from(flags & 63).unwrap_or(0);
                            a.hierarchy.push((name, parent, flags, start));
                        }
                        _ => return Err(lex.err("expected a joint's name or '}'")),
                    }
                }
            }
            b"baseframe" => {
                lex.sym(b'{')?;
                let a = file.anim.get_or_insert_default();
                loop {
                    if lex.peek()? == Some(Tok::Sym(b'}')) {
                        lex.next()?;
                        break;
                    }
                    let p = lex.vec3()?;
                    let q = lex.vec3()?;
                    a.base.push([p[0], p[1], p[2], q[0], q[1], q[2]]);
                }
            }
            b"frame" => {
                lex.int()?;
                lex.sym(b'{')?;
                let mut values = Vec::new();
                loop {
                    match lex.peek()? {
                        Some(Tok::Sym(b'}')) => {
                            lex.next()?;
                            break;
                        }
                        Some(Tok::Word(_)) => values.push(f32_of(lex.num()?)),
                        Some(_) => return Err(lex.err("expected a number or '}'")),
                        None => bail!(Truncated, "MD5 file is cut short"),
                    }
                }
                file.anim.get_or_insert_default().frames.push(values);
            }
            b"bounds" => {
                lex.sym(b'{')?;
                lex.skip_block()?;
            }
            _ => {
                // A section this reader doesn't know: its block, or its one value, is skipped.
                match lex.peek()? {
                    Some(Tok::Sym(b'{')) => {
                        lex.next()?;
                        lex.skip_block()?;
                    }
                    Some(Tok::Word(_) | Tok::Str(_)) => {
                        lex.next()?;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(file)
}

/// A `mesh { }` block, its `{` read.
fn mesh_block(lex: &mut Lex<'_>) -> Result<MeshBlock> {
    let mut m = MeshBlock::default();
    loop {
        let tok = lex.need()?;
        let key = match tok {
            Tok::Sym(b'}') => break,
            Tok::Word(w) => w,
            Tok::Sym(b'{') => {
                lex.skip_block()?;
                continue;
            }
            _ => continue,
        };
        match key {
            b"shader" => m.shader = lex.string()?,
            b"numverts" => {
                let n = lex.index()?;
                m.verts = vec![Vert::default(); lex.room(n, 12)?];
            }
            b"vert" => {
                let i = lex.index()?;
                lex.sym(b'(')?;
                let uv = [f32_of(lex.num()?), f32_of(lex.num()?)];
                lex.sym(b')')?;
                let start = lex.index()?;
                let count = lex.index()?;
                let n = m.verts.len();
                let v = m
                    .verts
                    .get_mut(i)
                    .ok_or_else(|| lex.err(&format!("vertex {i} is past numverts {n}")))?;
                *v = Vert { uv, start, count };
            }
            b"numtris" => {
                let n = lex.index()?;
                m.tris = vec![[0; 3]; lex.room(n, 10)?];
            }
            b"tri" => {
                let i = lex.index()?;
                let t = [lex.index()?, lex.index()?, lex.index()?];
                let n = m.tris.len();
                *m.tris
                    .get_mut(i)
                    .ok_or_else(|| lex.err(&format!("triangle {i} is past numtris {n}")))? = t;
            }
            b"numweights" => {
                let n = lex.index()?;
                m.weights = vec![Weight::default(); lex.room(n, 16)?];
            }
            b"weight" => {
                let i = lex.index()?;
                let joint = lex.index()?;
                let bias = lex.num()?;
                let pos = lex.vec3()?;
                let n = m.weights.len();
                *m.weights
                    .get_mut(i)
                    .ok_or_else(|| lex.err(&format!("weight {i} is past numweights {n}")))? =
                    Weight { joint, bias, pos };
            }
            // Something this reader doesn't know: skipped, a word at a time.
            _ => {}
        }
    }
    Ok(m)
}

/// The texture a shader names, found by the engine's conventions.
fn texture(shader: &str, input: &Input<'_>, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    if shader.is_empty() {
        return None;
    }
    let dir = input.dir();
    let has_ext = std::path::Path::new(shader).extension().is_some_and(|e| e.len() <= 4);
    let mut tries = Vec::new();
    if has_ext {
        tries.push(shader.to_owned());
    }
    tries.push(format!("{shader}.tga"));
    tries.push(format!("{shader}_d.tga"));
    tries
        .iter()
        .find_map(|t| assets.find(t, dir).map(|p| (p, t)))
        .and_then(|(p, t)| images.file(assets, p, t))
}

fn build_mesh(m: &MeshFile, stem: &str, input: &Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Scene> {
    let mut scene = Scene::default();
    let n = m.joints.len();
    for j in &m.joints {
        ensure!(
            j.parent >= -1 && j.parent < i64::try_from(n).unwrap_or(i64::MAX),
            Corrupt,
            "MD5 joint {:?} has parent {}, but there are {n} joints",
            j.name,
            j.parent
        );
    }
    // Joints: object-space transforms made relative to their parents.
    let mut worlds: Vec<M4> = Vec::with_capacity(n);
    for (i, j) in m.joints.iter().enumerate() {
        let parent = usize::try_from(j.parent).ok().filter(|&p| p != i);
        let rot = scene::normalize(j.rot).map(f64::from);
        let mut node = Node::named(j.name.clone(), parent);
        let (t, r) = match parent {
            Some(p) => {
                let pj = &m.joints[p];
                let pr = conj(scene::normalize(pj.rot).map(f64::from));
                let d = [j.pos[0] - pj.pos[0], j.pos[1] - pj.pos[1], j.pos[2] - pj.pos[2]];
                (scene::rotate(pr, d), scene::quat_mul(pr, rot))
            }
            None => (j.pos, rot),
        };
        node.translation = t.map(f32_of);
        node.rotation = scene::normalize(r);
        scene.add(node);
        worlds.push(scene::trs(j.pos.map(f32_of), rot.map(f32_of), [1.0; 3]));
    }

    let mut images = Images::new(!keep.image_bytes, input.dir());
    let mut materials: HashMap<String, usize> = HashMap::new();
    let mut prims = Vec::new();
    for (mi, block) in m.meshes.iter().enumerate() {
        for (vi, v) in block.verts.iter().enumerate() {
            ensure!(
                v.start.checked_add(v.count).is_some_and(|e| e <= block.weights.len()),
                Corrupt,
                "MD5 mesh {mi}: vertex {vi}'s weights run past numweights {}",
                block.weights.len()
            );
        }
        for (wi, w) in block.weights.iter().enumerate() {
            ensure!(
                w.joint < n,
                Corrupt,
                "MD5 mesh {mi}: weight {wi} names joint {}, but there are {n}",
                w.joint
            );
        }
        for (ti, t) in block.tris.iter().enumerate() {
            ensure!(
                t.iter().all(|&i| i < block.verts.len()),
                Corrupt,
                "MD5 mesh {mi}: triangle {ti} names a vertex past numverts {}",
                block.verts.len()
            );
        }
        if block.tris.is_empty() || block.verts.is_empty() {
            continue;
        }
        let mut positions = Vec::with_capacity(block.verts.len());
        let mut sets: Vec<Vec<(u32, f32)>> = Vec::with_capacity(block.verts.len());
        for v in &block.verts {
            let mut p = [0.0f64; 3];
            let mut set: Vec<(u32, f32)> = Vec::new();
            for w in &block.weights[v.start..v.start + v.count] {
                let j = &m.joints[w.joint];
                let o = scene::rotate(scene::normalize(j.rot).map(f64::from), w.pos);
                for k in 0..3 {
                    p[k] += w.bias * (j.pos[k] + o[k]);
                }
                let joint = u32_of(w.joint);
                match set.iter_mut().find(|s| s.0 == joint) {
                    Some(s) => s.1 += f32_of(w.bias),
                    None => set.push((joint, f32_of(w.bias))),
                }
            }
            set.retain(|s| s.1 > 0.0 && s.1.is_finite());
            set.sort_by(|a, b| b.1.total_cmp(&a.1));
            set.truncate(MAX_INFLUENCES);
            positions.push(p.map(f32_of));
            sets.push(set);
        }
        let influences = sets.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let mut joints = Vec::with_capacity(sets.len() * influences);
        let mut weights = Vec::with_capacity(sets.len() * influences);
        for set in &sets {
            for k in 0..influences {
                let (j, w) = set.get(k).copied().unwrap_or((0, 0.0));
                joints.push(j);
                weights.push(w);
            }
        }
        // Clockwise front faces in the file; glTF's are counterclockwise.
        let indices: Vec<u32> = block.tris.iter().flat_map(|t| [t[0], t[2], t[1]].map(u32_of)).collect();
        let normals = smooth_normals(&positions, &indices);
        let material = if let Some(&i) = materials.get(&block.shader) {
            i
        } else {
            let name = if block.shader.is_empty() {
                format!("mesh{mi}")
            } else {
                block.shader.clone()
            };
            let mut mat = Material::plain(&name, [1.0, 1.0, 1.0, 1.0]);
            mat.color_image = texture(&block.shader, input, assets, &mut images);
            scene.materials.push(mat);
            materials.insert(block.shader.clone(), scene.materials.len() - 1);
            scene.materials.len() - 1
        };
        let skinned = n > 0;
        prims.push(Primitive {
            positions,
            normals,
            uvs: vec![block.verts.iter().map(|v| v.uv).collect()],
            joints: if skinned { joints } else { Vec::new() },
            weights: if skinned { weights } else { Vec::new() },
            influences: if skinned { influences } else { 0 },
            indices,
            material: Some(material),
            ..Primitive::default()
        });
    }
    scene.images = images.list;
    scene.facts.insert("joints".into(), json!(n));
    scene.facts.insert("meshes".into(), json!(m.meshes.len()));
    if !prims.is_empty() {
        let mut node = Node::named(stem, None);
        node.mesh = Some(0);
        if n > 0 {
            node.skin = Some(0);
            scene.skins.push(Skin {
                name: stem.to_owned(),
                joints: (0..n).collect(),
                inverse_binds: worlds.iter().map(scene::invert).collect(),
            });
        }
        scene.add(node);
        scene.meshes.push(SceneMesh {
            name: stem.to_owned(),
            prims,
            ..SceneMesh::default()
        });
    }
    Ok(scene)
}

/// Each vertex's normal: the area-weighted sum of the faces around its position, so vertices split only for their
/// texture coordinates share one. Zero where no face has an area, which the writer fills in.
fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut slot: HashMap<[u32; 3], usize> = HashMap::new();
    let of: Vec<usize> = positions
        .iter()
        .map(|p| {
            let n = slot.len();
            *slot.entry(p.map(f32::to_bits)).or_insert(n)
        })
        .collect();
    let mut sums = vec![[0.0f64; 3]; slot.len()];
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|k| positions[t[k] as usize].map(f64::from));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let f = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &i in t {
            let s = &mut sums[of[i as usize]];
            for k in 0..3 {
                s[k] += f[k];
            }
        }
    }
    of.iter()
        .map(|&s| {
            let v = sums[s];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0.0 && l.is_finite() {
                v.map(|c| f32_of(c / l))
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

fn build_anim(a: &AnimFile, stem: &str) -> Result<Scene> {
    let mut scene = Scene::default();
    let n = a.hierarchy.len();
    ensure!(n > 0, Empty, "MD5 animation has no joints");
    ensure!(
        a.base.len() == n,
        Corrupt,
        "MD5 animation's baseframe has {} joints, its hierarchy {n}",
        a.base.len()
    );
    let components = a.components.unwrap_or_else(|| a.frames.first().map_or(0, Vec::len));
    for (name, parent, flags, start) in &a.hierarchy {
        ensure!(
            *parent >= -1 && *parent < i64::try_from(n).unwrap_or(i64::MAX),
            Corrupt,
            "MD5 joint {name:?} has parent {parent}, but there are {n} joints"
        );
        ensure!(
            start
                .checked_add(flags.count_ones() as usize)
                .is_some_and(|e| e <= components),
            Corrupt,
            "MD5 joint {name:?}'s components run past numAnimatedComponents {components}"
        );
    }
    for (k, f) in a.frames.iter().enumerate() {
        ensure!(
            f.len() >= components,
            Corrupt,
            "MD5 frame {k} has {} values, expected numAnimatedComponents {components}",
            f.len()
        );
    }
    let (rate, guessed) = match a.frame_rate {
        Some(r) if r > 0.0 && r.is_finite() => (f32_of(r), false),
        _ => (DEFAULT_RATE, true),
    };
    // A joint's six components in a frame: the base frame's, with those its flags animate taken from the frame.
    let pose = |base: &[f64; 6], flags: u32, start: usize, frame: &[f32]| {
        let mut v = *base;
        let mut at = start;
        for (bit, c) in v.iter_mut().enumerate() {
            if flags & (1 << bit) != 0 {
                *c = f64::from(frame[at]);
                at += 1;
            }
        }
        v
    };
    for (i, ((name, parent, flags, start), base)) in a.hierarchy.iter().zip(&a.base).enumerate() {
        let parent = usize::try_from(*parent).ok().filter(|&p| p != i);
        let mut node = Node::named(name.clone(), parent);
        let v = match a.frames.first() {
            Some(f) => pose(base, *flags, *start, f),
            None => *base,
        };
        node.translation = [v[0], v[1], v[2]].map(f32_of);
        node.rotation = scene::normalize(quat([v[3], v[4], v[5]]));
        scene.add(node);
    }
    let times: Vec<f32> = (0..a.frames.len())
        .map(|k| f32_of(f64::from(u32_of(k))) / rate)
        .collect();
    let mut channels = Vec::new();
    for (i, ((_, _, flags, start), base)) in a.hierarchy.iter().zip(&a.base).enumerate() {
        if *flags == 0 || a.frames.is_empty() {
            continue;
        }
        let mut ts = Vec::with_capacity(a.frames.len() * 3);
        let mut rs = Vec::with_capacity(a.frames.len() * 4);
        for f in &a.frames {
            let v = pose(base, *flags, *start, f);
            ts.extend([v[0], v[1], v[2]].map(f32_of));
            rs.extend(scene::normalize(quat([v[3], v[4], v[5]])));
        }
        if flags & 0b111 != 0 {
            channels.push(Channel {
                node: i,
                path: Path::Translation,
                times: times.clone(),
                values: ts,
                interp: Interp::Linear,
            });
        }
        if flags & 0b11_1000 != 0 {
            channels.push(Channel {
                node: i,
                path: Path::Rotation,
                times: times.clone(),
                values: rs,
                interp: Interp::Linear,
            });
        }
    }
    if !channels.is_empty() {
        scene.animations.push(Animation {
            name: stem.to_owned(),
            channels,
            extras: None,
        });
    }
    let mut facts = Map::new();
    facts.insert("joints".into(), json!(n));
    facts.insert("frames".into(), json!(a.frames.len()));
    facts.insert("components".into(), json!(components));
    scene.facts = facts;
    scene.source.insert("frames".into(), json!(a.frames.len()));
    scene.info.frame_rate = match a.frame_rate {
        Some(r) if !guessed => Fact::Declared(r),
        _ => Fact::Assumed(f64::from(rate)),
    };
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Memory;
    use crate::document::Pose;

    const MESH: &str = r#"MD5Version 10
commandline "test"
numJoints 2
numMeshes 1
joints {
	"root" -1 ( 0 0 0 ) ( 0 0 0 ) // a comment
	"tip" 0 ( 0 0 2 ) ( 0 0 0 )
}
mesh {
	shader "skin"
	numverts 3
	vert 0 ( 0 0 ) 0 1
	vert 1 ( 1 0 ) 1 1
	vert 2 ( 0 1 ) 2 2
	numtris 1
	tri 0 0 2 1
	numweights 4
	weight 0 0 1 ( 0 0 0 )
	weight 1 0 1 ( 1 0 0 )
	weight 2 1 0.5 ( 0 0 0 )
	weight 3 0 0.5 ( 0 0 2 )
}
"#;

    const ANIM: &str = r#"MD5Version 10
commandline ""
numFrames 2
numJoints 2
frameRate 10
numAnimatedComponents 2
hierarchy {
	"root" -1 4 0
	"tip" 0 8 1
}
bounds {
	( 0 0 0 ) ( 1 1 1 )
	( 0 0 0 ) ( 1 1 1 )
}
baseframe {
	( 0 0 0 ) ( 0 0 0 )
	( 0 0 2 ) ( 0 0 0 )
}
frame 0 {
	0 0
}
frame 1 {
	1 0.5
}
"#;

    fn read(text: &[u8], name: &str) -> Result<Document<'static>> {
        let ext = name.rsplit('.').next().unwrap_or("");
        load(Input::memory(text, name), ext, &Memory::new(), Keep::ALL)
    }

    #[test]
    fn mesh_vertices_come_from_weights() {
        let doc = read(MESH.as_bytes(), "t.md5mesh").unwrap();
        assert_eq!(doc.json["skins"][0]["joints"], json!([0, 1]));
        let mesh = doc.flatten(Pose::Rest).unwrap().unwrap();
        // Z up turned to Y up: (x, y, z) becomes (x, z, -y).
        let mut got: Vec<[f32; 3]> = mesh.prims[0].positions.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let want = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]];
        for (g, w) in got.iter().zip(want) {
            assert!(g.iter().zip(w).all(|(a, b)| (a - b).abs() < 1e-5), "{got:?}");
        }
        assert_eq!(doc.facts["joints"], 2);
        assert_eq!(doc.json["asset"]["extras"]["source"]["commandline"], "test");
    }

    #[test]
    fn anim_keys_follow_flags() {
        let doc = read(ANIM.as_bytes(), "walk.md5anim").unwrap();
        let a = &doc.json["animations"][0];
        assert_eq!(a["name"], "walk");
        assert_eq!(a["channels"].as_array().unwrap().len(), 2);
        assert_eq!(doc.facts["frames"], 2);
        assert!(doc.flatten(Pose::Rest).unwrap().is_none());
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("MD5"));
        assert_eq!(info.version.as_deref(), Some("10"));
        assert_eq!(info.frame_rate, Fact::Declared(10.0));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert!(info.converted.up_axis);
        let guessed = ANIM.replace("frameRate 10", "frameRate 0");
        let doc = read(guessed.as_bytes(), "walk.md5anim").unwrap();
        assert_eq!(doc.source_info().frame_rate, Fact::Assumed(24.0));
    }

    #[test]
    fn w_is_negative() {
        let q = quat([0.6, 0.0, 0.0]);
        assert!((q[3] + 0.8).abs() < 1e-12);
        assert_eq!(quat([1.0, 1.0, 0.0])[3], 0.0);
    }

    #[test]
    fn hostile_files_fail_cleanly() {
        for cut in [10, 40, 120, 200, MESH.len() - 5] {
            assert!(read(&MESH.as_bytes()[..cut], "t.md5mesh").is_err(), "cut at {cut}");
        }
        for cut in (0..ANIM.len()).step_by(7) {
            let _ = read(&ANIM.as_bytes()[..cut], "t.md5anim");
        }
        let bad = MESH.replace("weight 2 1", "weight 2 7");
        assert!(
            read(bad.as_bytes(), "t.md5mesh")
                .unwrap_err()
                .to_string()
                .contains("joint 7")
        );
        let bad = MESH.replace("tri 0 0 2 1", "tri 0 0 2 9");
        assert!(read(bad.as_bytes(), "t.md5mesh").is_err());
        let bad = MESH.replace("vert 2 ( 0 1 ) 2 2", "vert 2 ( 0 1 ) 3 2");
        assert!(read(bad.as_bytes(), "t.md5mesh").is_err());
        let bad = MESH.replace("numverts 3", "numverts 99999999999");
        assert!(read(bad.as_bytes(), "t.md5mesh").is_err());
        let bad = MESH.replace("vert 1", "vert 5");
        assert!(read(bad.as_bytes(), "t.md5mesh").is_err());
        let bad = ANIM.replace("\"tip\" 0 8 1", "\"tip\" 0 8 2");
        assert!(read(bad.as_bytes(), "t.md5anim").is_err());
        let bad = ANIM.replace("1 0.5", "1");
        assert!(read(bad.as_bytes(), "t.md5anim").is_err());
        assert!(read(b"hello", "t.md5mesh").is_err());
        assert!(read(b"MD5Version 10 joints { \"a\" 0 ( 0 0 0 ) ( 0 0 0 ) }", "t.md5mesh").is_ok());
        assert!(read(b"MD5Version 10 joints { \"a\" 3 ( 0 0 0 ) ( 0 0 0 ) }", "t.md5mesh").is_err());
        assert!(read(b"MD5Version 10 joints { \"a", "t.md5mesh").is_err());
        assert!(read(b"MD5Version 10 /* never closed", "t.md5mesh").is_err());
        assert!(read(b"MD5Version 10 frame 0 { 1.#QNAN }", "t.md5anim").is_err());
    }

    /// Loads every prefix of `bytes`, and `bytes` with each byte overwritten by a few values, through `load`;
    /// any result but a panic passes.
    fn sweep(bytes: &[u8], load: impl Fn(&[u8])) {
        for end in 0..bytes.len() {
            load(&bytes[..end]);
        }
        let mut copy = bytes.to_vec();
        for at in 0..bytes.len() {
            for value in [0x00, 0x7F, 0x80, 0xFF] {
                copy[at] = value;
                load(&copy);
            }
            copy[at] = bytes[at];
        }
    }

    #[test]
    fn hostile_bytes_never_panic() {
        for (bytes, name) in [
            (
                &include_bytes!("../../tests/corpus/md5/skinned_quad/box.md5mesh")[..],
                "box.md5mesh",
            ),
            (
                &include_bytes!("../../tests/corpus/md5/anim_alone/wave.md5anim")[..],
                "wave.md5anim",
            ),
        ] {
            assert!(read(bytes, name).is_ok());
            sweep(bytes, |b| {
                let _ = read(b, name);
            });
        }
    }
}
