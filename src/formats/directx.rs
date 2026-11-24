//! DirectX `.x` models as a glTF document: the frame hierarchy as nodes, meshes with their materials and textures,
//! skins and animations.
//!
//! A file starts with a 16-byte header, `xof 0303txt 0032`: the version, the encoding (`txt `, `bin `, or `tzip`
//! and `bzip` for compressed text and binary) and the float size (`0032` or `0064`). The body is a tree of data
//! objects, each a template name, an optional name and GUID, and braces around its values and child objects. A
//! reference to a named object elsewhere is its name in braces, `{ Name }`. Text separates values with commas,
//! semicolons and white space, and comments start with `//` or `#`. Binary is a stream of tokens: names, strings,
//! integers, GUIDs, lists of integers and lists of floats, braces and punctuation. Compressed files hold MSZIP
//! blocks after the header and a total size: each block is a raw deflate stream that may refer back into the block
//! before it, inflated here, to the size the file declares, then read as text or binary. Template declarations are
//! skipped; what the standard templates mean is built in.
//!
//! - `Frame` is a node, its `FrameTransformMatrix` the node's transform. A mesh outside any frame hangs from a node
//!   of its own; a frame's reference to a named mesh instances it there.
//! - `Mesh` becomes a mesh with a primitive per material. Faces of more than three corners are cut into fans.
//!   `MeshNormals` keeps its own face list, so a corner's vertex is its position and its normal together.
//!   `MeshTextureCoords` (several sets), `MeshVertexColors` (sRGB, made linear), and the texture coordinates,
//!   normals and colors in `DeclData` that the mesh doesn't already give, are carried, as are extra texture
//!   coordinate sets in `FVFData`. Tangents and binormals in `DeclData` are dropped: glTF computes its own.
//! - `MeshMaterialList` gives each face a material, by an index into the materials it holds or refers to; a list
//!   shorter than the faces repeats its last index. `Material` is its face color (sRGB, made linear, alpha as
//!   opacity), its emissive color, and its specular power as roughness (`1 - sqrt(power / 1000)`, as for OBJ); the
//!   specular color is dropped. `TextureFilename` is the base color texture and `NormalmapFilename` the normal map.
//! - `SkinWeights` names a bone (a frame), the vertices it moves with their weights, and the offset matrix from the
//!   mesh into the bone, which is the joint's inverse bind matrix. A skinned mesh hangs from a root node of its own,
//!   since a skin places it. A bone no frame is named for becomes a root node where its offset puts it, and
//!   vertices no bone moves are bound to the mesh's frame, so they stay where it puts them.
//! - `AnimationSet` is an animation and each `Animation` in it animates the frame it refers to. `AnimationKey`s of
//!   type 0 (rotation, a quaternion stored w first), 1 (scale), 2 (translation) and 3 or 4 (a matrix, decomposed)
//!   become linear channels. Times are in ticks, `AnimTicksPerSecond` of them a second (4800 when the file doesn't
//!   say). `AnimationOptions` (whether the animation loops, and its position quality) go in the animation's
//!   `extras`.
//!
//! Matrices are stored for row vectors, which read in order are glTF's column-major ones. `.x` is left-handed: the
//! scene is mirrored through the XY plane, and faces (clockwise in front, as Direct3D draws them) keep facing out.
//! Texture coordinates start at the top left, as glTF's do.
//!
//! The source info gives the header's version, Direct3D's left-handed, Y up convention (assumed: the file doesn't
//! say), its handedness converted, and units of unknown length. Its `frameRate` is `AnimTicksPerSecond`, or an assumed
//! 4800 when the file has animations and doesn't say. The encoding and float size are kept as `encoding` and
//! `floatSize`.
//!
//! Nothing is allocated from a count the file gives without the data to back it, objects can nest only so deep,
//! and inflating stops at the size the file declares and at a gigabyte. A file that doesn't parse is an error.

use std::collections::{HashMap, VecDeque};
use std::path::Path as FsPath;

use serde_json::json;

use super::inflate::inflate;
use super::scene::{self, Animation, Bin, Channel, IDENTITY, Interp, M4, MAX_INFLUENCES, Node, Path, Primitive};
use super::scene::{Scene, SceneMesh, Skin};
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness};
use crate::{Context, Result};

/// How deep data objects may nest.
const MAX_DEPTH: usize = 256;
/// The most bytes a compressed file may inflate to.
const MAX_INFLATED: usize = 1 << 30;
/// Ticks per second when the file doesn't say.
const DEFAULT_TICKS: f64 = 4800.0;
/// The most texture coordinate sets a mesh keeps.
const MAX_UV_SETS: usize = 8;

/// Reads the model.
pub(crate) fn load(input: super::Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let file = parse_file(&data)?;
    let mut b = Build::new(assets, keep, input.dir());
    b.run(&file.objects)?;
    let Build {
        mut scene,
        images,
        stats,
        ticks,
        animated,
        ..
    } = b;
    ensure!(
        !scene.nodes.is_empty(),
        "the .x file is empty: it has no frames or meshes"
    );
    scene.images = images.list;
    scene.mirror_z();
    scene.info.format = Some("DirectX".into());
    scene.info.version = Some(file.version);
    scene.info.up_axis = Fact::Assumed(Axis::PosY);
    scene.info.handedness = Fact::Assumed(Handedness::Left);
    scene.info.frame_rate = match ticks {
        Some(t) => Fact::Declared(t),
        None if animated => Fact::Assumed(DEFAULT_TICKS),
        None => Fact::Unknown,
    };
    scene.source.insert("encoding".into(), json!(file.encoding));
    scene
        .source
        .insert("floatSize".into(), json!(if file.double { 64 } else { 32 }));
    scene.facts.insert("frames".into(), json!(stats.frames));
    scene.facts.insert("meshes".into(), json!(stats.meshes));
    if stats.skinned > 0 {
        scene.facts.insert("skinned_meshes".into(), json!(stats.skinned));
    }
    if stats.animations > 0 {
        scene.facts.insert("animations".into(), json!(stats.animations));
    }
    // Triangles as drawn: a mesh that several frames refer to counts once for each.
    let drawn: usize = scene
        .nodes
        .iter()
        .filter_map(|n| n.mesh)
        .filter_map(|m| scene.meshes.get(m))
        .flat_map(|m| &m.prims)
        .filter(|p| !p.points)
        .map(|p| p.indices.len() / 3)
        .sum();
    scene.facts.insert("triangles".into(), json!(drawn));
    if !keep.materials {
        // With no materials, no primitive names one.
        scene.materials.clear();
    }
    scene.finish(keep)
}

/// A parsed file: its header and its top-level objects.
struct File {
    version: String,
    encoding: &'static str,
    double: bool,
    objects: Vec<Obj>,
}

/// Reads the header, inflates a compressed body, and parses the objects.
fn parse_file(data: &[u8]) -> Result<File> {
    ensure!(
        data.len() >= 16 && data[..4] == *b"xof ",
        "not a DirectX .x file (no xof header)"
    );
    let major = std::str::from_utf8(&data[4..6])
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let minor = std::str::from_utf8(&data[6..8])
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let version = match (major, minor) {
        (Some(a), Some(b)) => format!("{a}.{b}"),
        _ => scene::text(&data[4..8]),
    };
    let format = data[8..12].to_ascii_lowercase();
    let double = &data[12..16] == b"0064";
    let (encoding, binary, compressed) = match &format[..] {
        b"txt " => ("text", false, false),
        b"bin " => ("binary", true, false),
        b"tzip" => ("compressed text", false, true),
        b"bzip" => ("compressed binary", true, true),
        _ => bail!("unknown .x encoding {:?}", scene::text(&data[8..12])),
    };
    let inflated;
    let body = if compressed {
        inflated = mszip(&data[16..])?;
        &inflated[..]
    } else {
        &data[16..]
    };
    let mut lex = Lexer {
        b: Bin::new(body, ".x"),
        binary,
        double,
        list: 0,
        list_float: false,
        ahead: VecDeque::new(),
    };
    let objects = parse(&mut lex)?;
    Ok(File {
        version,
        encoding,
        double,
        objects,
    })
}

// --- MSZIP ---------------------------------------------------------------------------------------------------------

/// Inflates a compressed body: the total size (counting the 16-byte header), then blocks of an uncompressed size,
/// a compressed size, `CK` and a raw deflate stream that may refer back into the blocks before it.
fn mszip(data: &[u8]) -> Result<Vec<u8>> {
    let mut b = Bin::new(data, "compressed .x");
    let total = crate::num::usize_of(u64::from(b.u32()?));
    let want = total.saturating_sub(16);
    ensure!(want <= MAX_INFLATED, "compressed .x file too big ({total} bytes)");
    let mut out = Vec::with_capacity(want.min(1 << 20));
    while out.len() < want && b.left() >= 4 {
        let size = usize::from(b.u16()?);
        let packed = usize::from(b.u16()?);
        ensure!(packed >= 2, "bad block in a compressed .x file");
        let block = b.bytes(packed)?;
        ensure!(block[..2] == *b"CK", "bad block in a compressed .x file (no CK)");
        let limit = size.min(want - out.len());
        inflate(&block[2..], &mut out, limit)?;
    }
    Ok(out)
}

// --- Tokens and objects --------------------------------------------------------------------------------------------

/// A token of either encoding. Text has words (names and numbers alike); binary has names and numbers apart.
#[derive(Clone, Copy, Debug)]
enum Tok<'a> {
    Word(&'a [u8]),
    Str(&'a [u8]),
    Num(f64),
    Open,
    Close,
    Guid,
    End,
}

/// Splits a body into tokens, with a few tokens of look-ahead.
struct Lexer<'a> {
    b: Bin<'a>,
    binary: bool,
    double: bool,
    /// Numbers left in the binary list being read.
    list: u32,
    list_float: bool,
    ahead: VecDeque<Tok<'a>>,
}

impl<'a> Lexer<'a> {
    fn next(&mut self) -> Result<Tok<'a>> {
        match self.ahead.pop_front() {
            Some(t) => Ok(t),
            None => self.lex(),
        }
    }

    /// The token `k` places ahead, without taking it.
    fn peek(&mut self, k: usize) -> Result<Tok<'a>> {
        while self.ahead.len() <= k {
            let t = self.lex()?;
            self.ahead.push_back(t);
        }
        Ok(self.ahead[k])
    }

    fn lex(&mut self) -> Result<Tok<'a>> {
        if self.binary {
            self.lex_binary()
        } else {
            Ok(self.lex_text())
        }
    }

    fn lex_text(&mut self) -> Tok<'a> {
        let d = self.b.data;
        let is_end = |c: u8| c <= b' ' || matches!(c, b',' | b';' | b'{' | b'}' | b'"' | b'<');
        loop {
            let at = self.b.at;
            let Some(&c) = d.get(at) else {
                return Tok::End;
            };
            match c {
                b'{' => {
                    self.b.at += 1;
                    return Tok::Open;
                }
                b'}' => {
                    self.b.at += 1;
                    return Tok::Close;
                }
                b'"' => {
                    let start = at + 1;
                    let end = d[start..]
                        .iter()
                        .position(|&x| x == b'"')
                        .map_or(d.len(), |p| start + p);
                    self.b.at = (end + 1).min(d.len());
                    return Tok::Str(&d[start..end]);
                }
                b'<' => {
                    let end = d[at..].iter().position(|&x| x == b'>').map_or(d.len(), |p| at + p);
                    self.b.at = (end + 1).min(d.len());
                    return Tok::Guid;
                }
                b'#' => self.skip_line(),
                b'/' if d.get(at + 1) == Some(&b'/') => self.skip_line(),
                _ if is_end(c) => self.b.at += 1,
                _ => {
                    let end = d[at..].iter().position(|&x| is_end(x)).map_or(d.len(), |p| at + p);
                    self.b.at = end;
                    return Tok::Word(&d[at..end]);
                }
            }
        }
    }

    fn skip_line(&mut self) {
        let d = self.b.data;
        self.b.at = d[self.b.at..]
            .iter()
            .position(|&x| x == b'\n')
            .map_or(d.len(), |p| self.b.at + p);
    }

    fn lex_binary(&mut self) -> Result<Tok<'a>> {
        loop {
            if self.list > 0 {
                self.list -= 1;
                let v = if !self.list_float {
                    f64::from(self.b.u32()?)
                } else if self.double {
                    self.b.f64()?
                } else {
                    f64::from(self.b.f32()?)
                };
                return Ok(Tok::Num(v));
            }
            if self.b.left() < 2 {
                return Ok(Tok::End);
            }
            let id = self.b.u16()?;
            match id {
                1 => {
                    let n = self.b.u32()?;
                    let n = self.b.count(u64::from(n), 1)?;
                    return Ok(Tok::Word(self.b.bytes(n)?));
                }
                2 => {
                    let n = self.b.u32()?;
                    let n = self.b.count(u64::from(n), 1)?;
                    let s = self.b.bytes(n)?;
                    // The string's terminator, a comma or semicolon token.
                    self.b.skip(self.b.left().min(2))?;
                    return Ok(Tok::Str(s));
                }
                3 => return Ok(Tok::Num(f64::from(self.b.u32()?))),
                5 => {
                    self.b.skip(16)?;
                    return Ok(Tok::Guid);
                }
                6 | 7 => {
                    let n = self.b.u32()?;
                    let size = if id == 7 && self.double { 8 } else { 4 };
                    self.b.count(u64::from(n), size)?;
                    self.list = n;
                    self.list_float = id == 7;
                }
                10 => return Ok(Tok::Open),
                11 => return Ok(Tok::Close),
                // Punctuation, and the type keywords of template declarations.
                12..=20 | 40..=53 => {}
                31 => return Ok(Tok::Word(b"template")),
                _ => bail!("unknown token {id} in a binary .x file"),
            }
        }
    }
}

/// A data object: its template, its name, its values (numbers and strings apart, each in order) and its children.
/// A reference to a named object is a child with no template and that name.
#[derive(Debug, Default)]
struct Obj {
    kind: String,
    name: String,
    nums: Vec<f64>,
    strs: Vec<String>,
    kids: Vec<Obj>,
}

impl Obj {
    fn is(&self, kind: &str) -> bool {
        self.kind.eq_ignore_ascii_case(kind)
    }

    /// Whether this is a reference to a named object.
    fn is_ref(&self) -> bool {
        self.kind.is_empty()
    }

    fn kid(&self, kind: &str) -> Option<&Obj> {
        self.kids.iter().find(|k| k.is(kind))
    }

    fn kids<'s>(&'s self, kind: &'s str) -> impl Iterator<Item = &'s Obj> {
        self.kids.iter().filter(move |k| k.is(kind))
    }
}

/// Parses the objects of a body. Nesting is followed with a stack, and objects left open at the end are closed.
fn parse(lex: &mut Lexer<'_>) -> Result<Vec<Obj>> {
    let mut stack = vec![Obj::default()];
    loop {
        let top = stack.len() - 1;
        match lex.next()? {
            Tok::End => break,
            Tok::Open => {
                // A reference: `{ Name }`, `{ Name <GUID> }` or `{ <GUID> }`.
                let mut name = String::new();
                loop {
                    match lex.next()? {
                        Tok::Close | Tok::End => break,
                        Tok::Word(w) | Tok::Str(w) => name = scene::text(w),
                        Tok::Open => bail!("an object inside a reference in a .x file"),
                        Tok::Num(_) | Tok::Guid => {}
                    }
                }
                stack[top].kids.push(Obj { name, ..Obj::default() });
            }
            Tok::Close => {
                if top > 0
                    && let Some(o) = stack.pop()
                {
                    stack[top - 1].kids.push(o);
                }
            }
            Tok::Word(w) if numeric(w) => word_value(&mut stack[top], w),
            Tok::Word(w) => {
                // An object's header: a template name, then an optional name and GUID, then `{`.
                let header = match lex.peek(0)? {
                    Tok::Open => Some((None, 1)),
                    Tok::Guid if matches!(lex.peek(1)?, Tok::Open) => Some((None, 2)),
                    Tok::Word(n) => match lex.peek(1)? {
                        Tok::Open => Some((Some(n), 2)),
                        Tok::Guid if matches!(lex.peek(2)?, Tok::Open) => Some((Some(n), 3)),
                        _ => None,
                    },
                    _ => None,
                };
                let Some((name, taken)) = header else {
                    word_value(&mut stack[top], w);
                    continue;
                };
                for _ in 0..taken {
                    lex.next()?;
                }
                if w.eq_ignore_ascii_case(b"template") {
                    skip_object(lex)?;
                    continue;
                }
                ensure!(stack.len() <= MAX_DEPTH, "objects nest too deeply in a .x file");
                stack.push(Obj {
                    kind: scene::text(w),
                    name: name.map(scene::text).unwrap_or_default(),
                    ..Obj::default()
                });
            }
            Tok::Str(s) => stack[top].strs.push(scene::text(s)),
            Tok::Num(x) => stack[top].nums.push(x),
            Tok::Guid => {}
        }
    }
    while stack.len() > 1 {
        if let Some(o) = stack.pop() {
            let top = stack.len() - 1;
            stack[top].kids.push(o);
        }
    }
    Ok(stack.pop().map(|o| o.kids).unwrap_or_default())
}

/// Skips the rest of an object whose `{` was just read.
fn skip_object(lex: &mut Lexer<'_>) -> Result<()> {
    let mut depth = 1usize;
    while depth > 0 {
        match lex.next()? {
            Tok::Open => depth += 1,
            Tok::Close => depth -= 1,
            Tok::End => break,
            _ => {}
        }
    }
    Ok(())
}

/// Whether a word starts as a number does, so it's a value and never a template name.
fn numeric(w: &[u8]) -> bool {
    w.first()
        .is_some_and(|&c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.'))
}

/// A bare word among an object's values: a number, or 0 for the forms some writers give a NaN or infinity
/// (`-1.#IND00`, `1.#QNAN0`), or else a string.
fn word_value(o: &mut Obj, w: &[u8]) {
    if numeric(w) {
        if let Ok(s) = std::str::from_utf8(w)
            && let Ok(x) = s.parse::<f64>()
        {
            o.nums.push(x);
            return;
        }
        if w.contains(&b'#') {
            o.nums.push(0.0);
            return;
        }
    }
    o.strs.push(scene::text(w));
}

/// A whole number from 0 to `u32::MAX`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is checked to be a whole number in range first"
)]
fn whole(x: f64) -> Option<u32> {
    (x >= 0.0 && x <= f64::from(u32::MAX) && x.fract() == 0.0).then_some(x as u32)
}

/// A value read as a 32-bit word: a whole number as it is (a negative one as its two's complement), anything else
/// as the bits of the float it is.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is checked to be a whole number in range first"
)]
fn dword(x: f64) -> u32 {
    match whole(x) {
        Some(v) => v,
        None if x.fract() == 0.0 && x >= f64::from(i32::MIN) && x < 0.0 => (x as i32) as u32,
        None => f32_of(x).to_bits(),
    }
}

/// Reads an object's numbers in order.
struct Nums<'a> {
    v: &'a [f64],
    at: usize,
    what: &'static str,
}

impl<'a> Nums<'a> {
    fn new(o: &'a Obj, what: &'static str) -> Self {
        Self {
            v: &o.nums,
            at: 0,
            what,
        }
    }

    fn left(&self) -> usize {
        self.v.len() - self.at
    }

    fn f64(&mut self) -> Result<f64> {
        let x = *self
            .v
            .get(self.at)
            .with_context(|| format!("{} in a .x file is cut short", self.what))?;
        self.at += 1;
        Ok(x)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32_of(self.f64()?))
    }

    fn vec3(&mut self) -> Result<[f32; 3]> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }

    /// An index; one that isn't a whole number is `u32::MAX`, out of every range.
    fn index(&mut self) -> Result<u32> {
        Ok(whole(self.f64()?).unwrap_or(u32::MAX))
    }

    /// A count of items `width` numbers each, refused when there aren't the numbers to hold them.
    fn count(&mut self, width: usize) -> Result<usize> {
        let n = whole(self.f64()?).with_context(|| format!("{} in a .x file has a bad count", self.what))?;
        let n = crate::num::usize_of(u64::from(n));
        ensure!(
            n.saturating_mul(width) <= self.left(),
            "{} in a .x file counts {n} items, more than it holds",
            self.what
        );
        Ok(n)
    }

    fn matrix(&mut self) -> Result<M4> {
        let mut m = IDENTITY;
        for x in &mut m {
            *x = self.f64()?;
        }
        Ok(m)
    }
}

// --- The scene -----------------------------------------------------------------------------------------------------

/// The address of an object, to know it again.
fn key(o: &Obj) -> usize {
    std::ptr::from_ref(o) as usize
}

/// What was found, for the facts.
#[derive(Default)]
struct Stats {
    frames: usize,
    meshes: usize,
    skinned: usize,
    animations: usize,
}

/// A bone of a mesh: the frame it names, the vertices it moves and the offset matrix.
struct Bone {
    name: String,
    weights: Vec<(u32, f32)>,
    offset: M4,
}

/// A mesh as read: the scene mesh, the position each primitive's vertices came from, and the bones.
struct Built {
    mesh: SceneMesh,
    sources: Vec<Vec<u32>>,
    vertices: usize,
    bones: Vec<Bone>,
}

/// A mesh's normals: none, one per position, or their own list indexed per corner.
enum Normals {
    None,
    PerVertex(Vec<[f32; 3]>),
    PerCorner(Vec<[f32; 3]>, Vec<u32>),
}

/// What a mesh holds per position, and per corner for normals.
struct Attributes {
    positions: Vec<[f32; 3]>,
    normals: Normals,
    uvs: Vec<Vec<[f32; 2]>>,
    colors: Option<Vec<[f32; 4]>>,
}

/// Builds the scene from the objects.
struct Build<'a> {
    scene: Scene,
    keep: Keep,
    assets: &'a dyn Assets,
    dir: &'a FsPath,
    images: Images,
    materials: HashMap<String, &'a Obj>,
    meshes: HashMap<String, &'a Obj>,
    material_index: HashMap<usize, Option<usize>>,
    mesh_index: HashMap<usize, usize>,
    frames: HashMap<String, usize>,
    pending: Vec<(&'a Obj, Option<usize>)>,
    referenced: Vec<usize>,
    triangles: usize,
    ticks: Option<f64>,
    /// Whether the file has animations, kept or not.
    animated: bool,
    stats: Stats,
}

impl<'a> Build<'a> {
    fn new(assets: &'a dyn Assets, keep: Keep, dir: &'a FsPath) -> Self {
        Self {
            scene: Scene::default(),
            keep,
            assets,
            dir,
            images: Images::new(!keep.image_bytes, dir),
            materials: HashMap::new(),
            meshes: HashMap::new(),
            material_index: HashMap::new(),
            mesh_index: HashMap::new(),
            frames: HashMap::new(),
            pending: Vec::new(),
            referenced: Vec::new(),
            triangles: 0,
            ticks: None,
            animated: false,
            stats: Stats::default(),
        }
    }

    fn run(&mut self, roots: &'a [Obj]) -> Result<()> {
        // Named materials and meshes anywhere, for references.
        let mut stack: Vec<&'a Obj> = roots.iter().collect();
        while let Some(o) = stack.pop() {
            if !o.name.is_empty() {
                if o.is("Material") {
                    self.materials.entry(o.name.clone()).or_insert(o);
                } else if o.is("Mesh") {
                    self.meshes.entry(o.name.clone()).or_insert(o);
                }
            }
            stack.extend(o.kids.iter());
        }
        let mut sets = Vec::new();
        let mut loose = Vec::new();
        let mut top_meshes = Vec::new();
        for o in roots {
            if o.is("Frame") {
                self.frames(o);
            } else if o.is("Mesh") {
                top_meshes.push(o);
            } else if o.is("AnimationSet") {
                sets.push(o);
            } else if o.is("Animation") {
                loose.push(o);
            } else if o.is("AnimTicksPerSecond") {
                self.ticks = o.nums.first().copied().filter(|t| *t > 0.0 && t.is_finite());
            }
        }
        for o in top_meshes {
            if !self.referenced.contains(&key(o)) {
                self.pending.push((o, None));
            }
        }
        for (m, frame) in std::mem::take(&mut self.pending) {
            self.place(m, frame)?;
        }
        self.animated = !sets.is_empty() || !loose.is_empty();
        if self.keep.animations {
            let ticks = self.ticks.unwrap_or(DEFAULT_TICKS);
            for s in sets {
                let anims: Vec<&Obj> = s.kids("Animation").collect();
                self.animation(&s.name, &anims, ticks)?;
            }
            if !loose.is_empty() {
                self.animation("", &loose, ticks)?;
            }
        }
        Ok(())
    }

    /// Adds a frame and the frames under it as nodes, and notes the meshes they hold.
    fn frames(&mut self, root: &'a Obj) {
        let mut stack = vec![(root, None)];
        while let Some((f, parent)) = stack.pop() {
            let mut node = Node::named(f.name.clone(), parent);
            if let Some(m) = f.kid("FrameTransformMatrix")
                && m.nums.len() >= 16
            {
                let mut r = Nums::new(m, "a frame's matrix");
                if let Ok(m) = r.matrix() {
                    (node.translation, node.rotation, node.scale) = scene::decompose(&m);
                }
            }
            let i = self.scene.add(node);
            self.stats.frames += 1;
            if !f.name.is_empty() {
                self.frames.entry(f.name.clone()).or_insert(i);
            }
            for k in &f.kids {
                if k.is("Mesh") {
                    self.pending.push((k, Some(i)));
                } else if k.is_ref()
                    && let Some(&m) = self.meshes.get(&k.name)
                {
                    self.pending.push((m, Some(i)));
                    self.referenced.push(key(m));
                }
            }
            stack.extend(f.kids.iter().rev().filter(|k| k.is("Frame")).map(|k| (k, Some(i))));
        }
    }

    /// Reads a mesh and hangs it from its frame, or, skinned, from a root node of its own.
    fn place(&mut self, m: &'a Obj, frame: Option<usize>) -> Result<()> {
        if let Some(&mi) = self.mesh_index.get(&key(m)) {
            self.attach(mi, &m.name, frame);
            return Ok(());
        }
        let built = self.mesh(m)?;
        self.stats.meshes += 1;
        if built.mesh.prims.is_empty() {
            return Ok(());
        }
        if !self.keep.skins || built.bones.is_empty() {
            let mi = self.scene.meshes.len();
            self.scene.meshes.push(built.mesh);
            self.mesh_index.insert(key(m), mi);
            self.attach(mi, &m.name, frame);
            return Ok(());
        }
        self.skin(m, frame, built);
        Ok(())
    }

    /// Hangs mesh `mi` from `frame`: on the frame's node when it has none yet, else on a child.
    fn attach(&mut self, mi: usize, name: &str, frame: Option<usize>) {
        match frame {
            Some(f) if self.scene.nodes[f].mesh.is_none() => self.scene.nodes[f].mesh = Some(mi),
            _ => {
                let mut n = Node::named(name, frame);
                n.mesh = Some(mi);
                self.scene.add(n);
            }
        }
    }

    /// Adds a skinned mesh: its joints are the frames its bones name, or new nodes where a bone's offset puts it.
    fn skin(&mut self, m: &Obj, frame: Option<usize>, built: Built) {
        let Built {
            mut mesh,
            sources,
            vertices,
            bones,
        } = built;
        let mut joints: Vec<usize> = Vec::new();
        let mut binds: Vec<M4> = Vec::new();
        let mut lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vertices];
        for bone in &bones {
            let node = if let Some(&n) = self.frames.get(&bone.name) {
                n
            } else {
                // A bone no frame is named for: a node where its offset puts it.
                let mut n = Node::named(bone.name.clone(), None);
                (n.translation, n.rotation, n.scale) = scene::decompose(&scene::invert(&bone.offset));
                let i = self.scene.add(n);
                self.frames.insert(bone.name.clone(), i);
                i
            };
            let j = if let Some(j) = joints.iter().position(|&x| x == node) {
                j
            } else {
                joints.push(node);
                binds.push(bone.offset);
                joints.len() - 1
            };
            let j = u32_of(j);
            for &(v, w) in &bone.weights {
                if let Some(l) = lists.get_mut(v as usize)
                    && w > 0.0
                    && w.is_finite()
                {
                    l.push((j, w));
                }
            }
        }
        for l in &mut lists {
            l.sort_by(|a, b| b.1.total_cmp(&a.1));
            l.truncate(MAX_INFLUENCES);
        }
        // Vertices no bone moves stay with the mesh's frame: a joint at the frame (or at a node under it, when the
        // frame is a bone already) with no offset.
        let used = |p: &u32| lists.get(*p as usize).is_some_and(Vec::is_empty);
        let mut rest = 0;
        if sources.iter().flatten().any(used) {
            let node = match frame {
                Some(f) if !joints.contains(&f) => f,
                _ => {
                    let name = format!("{}_rest", if m.name.is_empty() { "mesh" } else { &m.name });
                    self.scene.add(Node::named(name, frame))
                }
            };
            joints.push(node);
            binds.push(IDENTITY);
            rest = u32_of(joints.len() - 1);
        }
        for (p, src) in mesh.prims.iter_mut().zip(&sources) {
            let influences = src.iter().map(|&v| lists[v as usize].len()).max().unwrap_or(0).max(1);
            p.influences = influences;
            p.rest_joint = rest;
            p.joints = vec![0; src.len() * influences];
            p.weights = vec![0.0; src.len() * influences];
            for (i, &v) in src.iter().enumerate() {
                for (k, &(j, w)) in lists[v as usize].iter().enumerate() {
                    p.joints[i * influences + k] = j;
                    p.weights[i * influences + k] = w;
                }
            }
        }
        let si = self.scene.skins.len();
        self.scene.skins.push(Skin {
            name: m.name.clone(),
            joints,
            inverse_binds: binds,
        });
        let mi = self.scene.meshes.len();
        self.scene.meshes.push(mesh);
        let name = match frame {
            Some(f) if m.name.is_empty() => self.scene.nodes[f].name.clone(),
            _ => m.name.clone(),
        };
        let mut n = Node::named(name, None);
        n.mesh = Some(mi);
        n.skin = Some(si);
        self.scene.add(n);
        self.stats.skinned += 1;
    }

    /// Reads a mesh's geometry, attributes, materials and bones.
    fn mesh(&mut self, m: &Obj) -> Result<Built> {
        let mut r = Nums::new(m, "a mesh");
        let nv = r.count(3)?;
        let mut positions = Vec::with_capacity(nv);
        for _ in 0..nv {
            positions.push(r.vec3()?);
        }
        let nf = r.count(1)?;
        let mut faces = Vec::with_capacity(nf);
        let mut corners = Vec::new();
        for _ in 0..nf {
            let n = r.count(1)?;
            faces.push((corners.len(), n));
            for _ in 0..n {
                corners.push(r.index()?);
            }
        }
        let tris: usize = faces.iter().map(|f| f.1.saturating_sub(2)).sum();
        self.triangles = self.triangles.saturating_add(tris);
        let limit = self.keep.triangles();
        ensure!(
            self.triangles <= limit,
            "too many triangles to convert (more than {limit})"
        );

        let normals = mesh_normals(m, nv, &faces)?;
        let mut uvs = Vec::new();
        for k in m.kids("MeshTextureCoords") {
            let mut r = Nums::new(k, "texture coordinates");
            let n = r.count(2)?;
            if n == 0 || uvs.len() >= MAX_UV_SETS {
                continue;
            }
            let mut set = Vec::with_capacity(nv);
            for i in 0..n {
                let uv = [r.f32()?, r.f32()?];
                if i < nv {
                    set.push(uv);
                }
            }
            set.resize(nv, [0.0; 2]);
            uvs.push(set);
        }
        let mut colors = None;
        if let Some(k) = m.kid("MeshVertexColors") {
            let mut r = Nums::new(k, "vertex colors");
            let n = r.count(5)?;
            let mut set = vec![[1.0f32; 4]; nv];
            for _ in 0..n {
                let i = r.index()? as usize;
                let c = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
                if let Some(s) = set.get_mut(i) {
                    *s = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]];
                }
            }
            colors = Some(set);
        }
        let mut attrs = Attributes {
            positions,
            normals,
            uvs,
            colors,
        };
        if let Some(k) = m.kid("DeclData") {
            decl_data(k, nv, &mut attrs)?;
        }
        if let Some(k) = m.kid("FVFData") {
            fvf_data(k, nv, &mut attrs.uvs)?;
        }

        // Each face's material. Faces are split by material even when materials aren't kept, so the primitives are
        // the ones a full read makes; the materials themselves are dropped before the scene is written.
        let mut slots: Vec<Option<usize>> = Vec::new();
        if let Some(k) = m.kid("MeshMaterialList") {
            let mut r = Nums::new(k, "a material list");
            r.f64()?;
            let n = r.count(1)?;
            let mut idx = Vec::with_capacity(n);
            for _ in 0..n {
                idx.push(r.index()? as usize);
            }
            let mats: Vec<Option<usize>> = k
                .kids
                .iter()
                .filter(|o| o.is("Material") || o.is_ref())
                .map(|o| self.material(o))
                .collect();
            slots = (0..nf)
                .map(|f| idx.get(f).or(idx.last()).and_then(|&i| mats.get(i)).copied().flatten())
                .collect();
        }

        let mut prims: Vec<Primitive> = Vec::new();
        let mut sources: Vec<Vec<u32>> = Vec::new();
        let mut welds: Vec<HashMap<(u32, u32), u32>> = Vec::new();
        let mut by_slot: HashMap<Option<usize>, usize> = HashMap::new();
        for (f, &(start, n)) in faces.iter().enumerate() {
            let cs = &corners[start..start + n];
            if n < 3 || cs.iter().any(|&c| c as usize >= nv) {
                continue;
            }
            let slot = slots.get(f).copied().flatten();
            let pi = *by_slot.entry(slot).or_insert_with(|| {
                prims.push(Primitive {
                    material: slot,
                    ..Primitive::default()
                });
                sources.push(Vec::new());
                welds.push(HashMap::new());
                prims.len() - 1
            });
            let (p, src, weld) = (&mut prims[pi], &mut sources[pi], &mut welds[pi]);
            let mut tri = Vec::with_capacity(n);
            for (k, &c) in cs.iter().enumerate() {
                tri.push(corner(p, src, weld, &attrs, c, start + k));
            }
            for k in 1..n - 1 {
                p.indices.extend([tri[0], tri[k], tri[k + 1]]);
            }
        }

        let mut bones = Vec::new();
        for k in m.kids("SkinWeights") {
            let name = k.strs.first().cloned().unwrap_or_default();
            let mut r = Nums::new(k, "skin weights");
            let n = r.count(2)?;
            let mut idx = Vec::with_capacity(n);
            for _ in 0..n {
                idx.push(r.index()?);
            }
            let mut weights = Vec::with_capacity(n);
            for &i in &idx {
                let w = r.f32()?;
                if (i as usize) < nv {
                    weights.push((i, w));
                }
            }
            let offset = if r.left() >= 16 { r.matrix()? } else { IDENTITY };
            bones.push(Bone { name, weights, offset });
        }

        let mesh = SceneMesh {
            name: m.name.clone(),
            prims,
            ..SceneMesh::default()
        };
        Ok(Built {
            mesh,
            sources,
            vertices: nv,
            bones,
        })
    }

    /// The scene material for a `Material` object or a reference to one, made once.
    fn material(&mut self, o: &Obj) -> Option<usize> {
        let o: &Obj = if o.is_ref() {
            self.materials.get(&o.name).copied()?
        } else {
            o
        };
        if let Some(&i) = self.material_index.get(&key(o)) {
            return i;
        }
        let v = |k: usize, d: f32| o.nums.get(k).map_or(d, |&x| f32_of(x));
        let lin = |k: usize| srgb_to_linear(v(k, 1.0).clamp(0.0, 1.0));
        let mut mat = Material::plain(&o.name, [lin(0), lin(1), lin(2), v(3, 1.0).clamp(0.0, 1.0)]);
        let power = v(4, 0.0);
        let power = if power.is_finite() {
            power.clamp(0.0, 1000.0)
        } else {
            0.0
        };
        mat.roughness = 1.0 - (power / 1000.0).sqrt();
        mat.emissive = [8, 9, 10].map(|k| srgb_to_linear(v(k, 0.0).clamp(0.0, 1.0)));
        if self.keep.textures && self.keep.materials {
            if let Some(f) = o.kid("TextureFilename").and_then(|t| t.strs.first()) {
                mat.color_image = self.texture(f);
            }
            if let Some(f) = o.kid("NormalmapFilename").and_then(|t| t.strs.first()) {
                mat.normal_image = self.texture(f);
            }
        }
        let i = self.scene.materials.len();
        self.scene.materials.push(mat);
        self.material_index.insert(key(o), Some(i));
        Some(i)
    }

    /// The image a texture file name refers to, when it's found.
    fn texture(&mut self, name: &str) -> Option<usize> {
        let name = name.replace("\\\\", "\\");
        let path = self.assets.find(&name, self.dir)?;
        self.images.file(self.assets, path, &name)
    }

    /// Adds an animation set: a channel for each kind of key each `Animation` gives its frame.
    fn animation(&mut self, name: &str, anims: &[&Obj], ticks: f64) -> Result<()> {
        let mut channels: Vec<Channel> = Vec::new();
        let mut options = Vec::new();
        for a in anims {
            let target = a.kids.iter().find(|k| k.is_ref()).map_or(&a.name, |k| &k.name);
            let Some(&node) = self.frames.get(target) else {
                continue;
            };
            if let Some(o) = a.kid("AnimationOptions") {
                let mut opt = json!({ "node": target });
                if let Some(&x) = o.nums.first() {
                    opt["closed"] = json!(x != 0.0);
                }
                if let Some(&x) = o.nums.get(1) {
                    opt["splinePositions"] = json!(x != 0.0);
                }
                options.push(opt);
            }
            for k in a.kids("AnimationKey") {
                for c in keys(k, node, ticks)? {
                    if !channels.iter().any(|x| x.node == c.node && x.path == c.path) {
                        channels.push(c);
                    }
                }
            }
        }
        if channels.is_empty() {
            return Ok(());
        }
        let extras = (!options.is_empty()).then(|| json!({ "animationOptions": options }));
        self.scene.animations.push(Animation {
            name: name.to_owned(),
            channels,
            extras,
        });
        self.stats.animations += 1;
        Ok(())
    }
}

/// The vertex for a corner at position `p` (corner `c` of the mesh) in primitive `prim`, added the first time its
/// position and normal come together.
fn corner(
    prim: &mut Primitive,
    src: &mut Vec<u32>,
    weld: &mut HashMap<(u32, u32), u32>,
    a: &Attributes,
    p: u32,
    c: usize,
) -> u32 {
    let n = match &a.normals {
        Normals::PerCorner(_, idx) => idx.get(c).copied().unwrap_or(u32::MAX),
        _ => u32::MAX,
    };
    if let Some(&v) = weld.get(&(p, n)) {
        return v;
    }
    let v = u32_of(prim.positions.len());
    let pi = p as usize;
    prim.positions.push(a.positions[pi]);
    match &a.normals {
        Normals::None => {}
        Normals::PerVertex(list) => prim.normals.push(list.get(pi).copied().unwrap_or([0.0; 3])),
        Normals::PerCorner(list, _) => prim.normals.push(list.get(n as usize).copied().unwrap_or([0.0; 3])),
    }
    prim.uvs.resize(a.uvs.len(), Vec::new());
    for (set, from) in prim.uvs.iter_mut().zip(&a.uvs) {
        set.push(from.get(pi).copied().unwrap_or([0.0; 2]));
    }
    if let Some(colors) = &a.colors {
        prim.colors.resize(1, Vec::new());
        prim.colors[0].push(colors.get(pi).copied().unwrap_or([1.0; 4]));
    }
    src.push(p);
    weld.insert((p, n), v);
    v
}

/// A mesh's normals: by their own face list when it matches the mesh's faces, else one per position when there
/// are as many, else none.
fn mesh_normals(m: &Obj, nv: usize, faces: &[(usize, usize)]) -> Result<Normals> {
    let Some(k) = m.kid("MeshNormals") else {
        return Ok(Normals::None);
    };
    let mut r = Nums::new(k, "normals");
    let nn = r.count(3)?;
    let mut list = Vec::with_capacity(nn);
    for _ in 0..nn {
        list.push(r.vec3()?);
    }
    if r.left() > 0 {
        let nf = r.count(1)?;
        if nf == faces.len() {
            let mut idx = Vec::with_capacity(faces.iter().map(|f| f.1).sum());
            let mut fits = true;
            for &(_, n) in faces {
                let count = r.count(1)?;
                if count != n {
                    fits = false;
                    break;
                }
                for _ in 0..n {
                    idx.push(r.index()?);
                }
            }
            if fits {
                return Ok(Normals::PerCorner(list, idx));
            }
        }
    }
    Ok(if nn == nv {
        Normals::PerVertex(list)
    } else {
        Normals::None
    })
}

/// The size of a `DeclData` element type, in 32-bit words.
fn decl_size(ty: u32) -> Option<usize> {
    Some(match ty {
        0 | 4 | 5 | 6 | 8 | 9 | 11 | 13 | 14 | 15 => 1,
        1 | 7 | 10 | 12 | 16 => 2,
        2 => 3,
        3 => 4,
        _ => return None,
    })
}

/// Decodes a 16-bit float.
fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The components of a `DeclData` element of type `ty` from its words; `None` for the packed types not decoded.
fn decl_value(ty: u32, w: &[u32]) -> Option<[f32; 4]> {
    let mut out = [0.0, 0.0, 0.0, 1.0];
    let lo = |d: u32| (d & 0xffff) as u16;
    let hi = |d: u32| (d >> 16) as u16;
    let shorts: Vec<u16> = w.iter().flat_map(|&d| [lo(d), hi(d)]).collect();
    match ty {
        0..=3 => w.iter().zip(&mut out).for_each(|(&d, o)| *o = f32::from_bits(d)),
        4 => {
            let d = w[0];
            let byte = |s: u32| f32::from(((d >> s) & 0xff) as u8) / 255.0;
            out = [byte(16), byte(8), byte(0), byte(24)];
        }
        5 | 8 => {
            let scale = if ty == 8 { 255.0 } else { 1.0 };
            for (k, o) in out.iter_mut().enumerate() {
                *o = f32::from(((w[0] >> (k * 8)) & 0xff) as u8) / scale;
            }
        }
        // SHORT2 and SHORT4 as they are; SHORT2N and SHORT4N divided by 32767, -32768 clamped to -1.
        6 | 7 => out
            .iter_mut()
            .zip(&shorts)
            .for_each(|(o, &s)| *o = f32::from(s.cast_signed())),
        9 | 10 => out
            .iter_mut()
            .zip(&shorts)
            .for_each(|(o, &s)| *o = (f32::from(s.cast_signed()) / 32767.0).max(-1.0)),
        11 | 12 => out
            .iter_mut()
            .zip(&shorts)
            .for_each(|(o, &s)| *o = f32::from(s) / 65535.0),
        15 | 16 => out.iter_mut().zip(&shorts).for_each(|(o, &s)| *o = half(s)),
        _ => return None,
    }
    Some(out)
}

/// Takes what `DeclData` gives per vertex that the mesh doesn't already: normals, texture coordinates and colors.
fn decl_data(k: &Obj, nv: usize, a: &mut Attributes) -> Result<()> {
    let mut r = Nums::new(k, "vertex declaration data");
    let ne = r.count(4)?;
    let mut elements = Vec::with_capacity(ne);
    for _ in 0..ne {
        let ty = r.index()?;
        r.f64()?;
        let usage = r.index()?;
        r.f64()?;
        elements.push((ty, usage));
    }
    let nd = r.count(1)?;
    let mut data = Vec::with_capacity(nd);
    for _ in 0..nd {
        data.push(dword(r.f64()?));
    }
    let Some(sizes) = elements.iter().map(|e| decl_size(e.0)).collect::<Option<Vec<usize>>>() else {
        return Ok(());
    };
    let stride: usize = sizes.iter().sum();
    if stride == 0 || nv == 0 || data.len() < nv.saturating_mul(stride) {
        return Ok(());
    }
    let has_normals = !matches!(a.normals, Normals::None);
    let has_uvs = !a.uvs.is_empty();
    let has_colors = a.colors.is_some();
    let mut offset = 0;
    for (&(ty, usage), &size) in elements.iter().zip(&sizes) {
        let value = |v: usize| decl_value(ty, &data[v * stride + offset..v * stride + offset + size]);
        match usage {
            3 if !has_normals => {
                if let Some(list) = (0..nv).map(|v| value(v).map(|x| [x[0], x[1], x[2]])).collect() {
                    a.normals = Normals::PerVertex(list);
                }
            }
            5 if !has_uvs && a.uvs.len() < MAX_UV_SETS => {
                if let Some(set) = (0..nv).map(|v| value(v).map(|x| [x[0], x[1]])).collect() {
                    a.uvs.push(set);
                }
            }
            10 if !has_colors && a.colors.is_none() => {
                a.colors = (0..nv)
                    .map(|v| value(v).map(|x| [srgb_to_linear(x[0]), srgb_to_linear(x[1]), srgb_to_linear(x[2]), x[3]]))
                    .collect();
            }
            _ => {}
        }
        offset += size;
    }
    Ok(())
}

/// Takes the extra texture coordinate sets `FVFData` holds: as many as its flags' texture count, two floats each
/// per vertex, sets interleaved.
fn fvf_data(k: &Obj, nv: usize, uvs: &mut Vec<Vec<[f32; 2]>>) -> Result<()> {
    let mut r = Nums::new(k, "FVF data");
    let flags = dword(r.f64()?);
    let nd = r.count(1)?;
    let mut data = Vec::with_capacity(nd);
    for _ in 0..nd {
        data.push(f32::from_bits(dword(r.f64()?)));
    }
    let sets = ((flags >> 8) & 0xf) as usize;
    if sets == 0 || nv == 0 {
        return Ok(());
    }
    let per = (nd / (2 * sets)).min(nv);
    if per == 0 {
        return Ok(());
    }
    for s in 0..sets {
        if uvs.len() >= MAX_UV_SETS {
            break;
        }
        let mut set: Vec<[f32; 2]> = (0..per)
            .map(|v| {
                let at = (v * sets + s) * 2;
                [data[at], data[at + 1]]
            })
            .collect();
        set.resize(nv, [0.0; 2]);
        uvs.push(set);
    }
    Ok(())
}

/// The channels an `AnimationKey` gives frame `node`: one for rotation, scale or translation keys, three for
/// matrix keys.
fn keys(k: &Obj, node: usize, ticks: f64) -> Result<Vec<Channel>> {
    let mut r = Nums::new(k, "animation keys");
    let ty = r.index()?;
    let n = r.count(2)?;
    let channel = |path| Channel {
        node,
        path,
        times: Vec::new(),
        values: Vec::new(),
        interp: Interp::Linear,
    };
    let mut out = match ty {
        0 => vec![channel(Path::Rotation)],
        1 => vec![channel(Path::Scale)],
        2 => vec![channel(Path::Translation)],
        3 | 4 => vec![
            channel(Path::Translation),
            channel(Path::Rotation),
            channel(Path::Scale),
        ],
        _ => return Ok(Vec::new()),
    };
    for _ in 0..n {
        let t = f32_of(r.f64()? / ticks);
        let count = r.count(1)?;
        let at = r.at;
        r.at += count;
        let v = &r.v[at..at + count];
        let f = |i: usize| f32_of(v[i]);
        match ty {
            0 if count >= 4 => {
                out[0].times.push(t);
                out[0].values.extend(rotation_key([f(0), f(1), f(2), f(3)]));
            }
            1 | 2 if count >= 3 => {
                out[0].times.push(t);
                out[0].values.extend([f(0), f(1), f(2)]);
            }
            3 | 4 if count >= 16 => {
                let m: M4 = std::array::from_fn(|i| v[i]);
                let (tr, q, s) = scene::decompose(&m);
                for c in &mut out {
                    c.times.push(t);
                }
                out[0].values.extend(tr);
                out[1].values.extend(q);
                out[2].values.extend(s);
            }
            _ => {}
        }
    }
    out.retain(|c| !c.times.is_empty());
    Ok(out)
}

/// A rotation key as glTF's x, y, z, w. It's stored w, x, y, z, and turning the other way from the frame matrix
/// it stands for: the conjugate of the quaternion of that matrix, as the files exporters write show (a frame's
/// bind pose key against its `FrameTransformMatrix`).
fn rotation_key(q: [f32; 4]) -> [f32; 4] {
    [-q[1], -q[2], -q[3], q[0]]
}

#[cfg(test)]
mod tests;
