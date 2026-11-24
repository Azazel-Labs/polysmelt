//! `MikuMikuDance` models (`.pmx` 2.0 and 2.1, `.pmd`) and motions (`.vmd`).
//!
//! A model's vertices, faces and materials become one skinned mesh with a primitive per material. Its bones become
//! nodes under their parents, each placed by its offset from its parent (bones have no rest rotation), and every
//! bone is a joint of the skin, bound where it stands. Vertices keep their normal, texture coordinates and up to
//! four additional UVs (as `TEXCOORD_1` to `TEXCOORD_4`, their first two components), and their deform: BDEF1, BDEF2
//! and BDEF4 as they are, SDEF as the BDEF2 it falls back to (its C, R0 and R1 are dropped), and QDEF as BDEF4.
//! Vertex morphs become morph targets with their names, as do group morphs whose members move vertices (their
//! offsets summed by ratio). Bone, UV, material, flip and impulse morphs are kept in the mesh's `extras.morphs`.
//!
//! Materials take the diffuse color (as linear) with its alpha, the texture as the color image (cut out at alpha
//! 0.5 when it is a PNG, TGA or DDS, formats that carry alpha on purpose), double-sided drawing from the flag (in
//! PMD, for any material that isn't opaque), and a roughness from the specular power. The rest (specular and
//! ambient colors, edge, sphere map, toon) is kept in `asset.extras.source.materials`. Bones' IK, inherited
//! rotation and translation, and fixed and local axes go in their nodes' `extras`. Rigid bodies, joints, soft
//! bodies and display frames are counted, not converted: glTF has no physics.
//!
//! A motion has no skeleton of its own, so a `.vmd` becomes a node per bone it names, under one root, keyed as the
//! file keys them: a translation is the bone's move from where it rests, which a bare motion places at the origin,
//! and a rotation is the bone's turn. Keys are at 30 frames a second. Where a key's Bézier curve is the straight
//! line, the keys stay linear as they are; otherwise the curve is sampled at every frame between the keys (at most
//! 60 samples a span). Morph keys go in `asset.extras.source.morphKeys`; camera, light, shadow and IK keys are
//! counted.
//!
//! The format is left-handed and Y up with a unit of 8 cm, so the scene is mirrored through its XY plane and
//! scaled by 0.08 into meters. Text is UTF-16 or UTF-8 in PMX, as its header says, and Shift-JIS in PMD and VMD.
//!
//! The source info has the handedness as declared (the format defines it) and converted, the up axis `+Y` assumed,
//! and `metersPerUnit` an assumed 0.08 (the unit's length is the users' convention, not the format's), converted.
//! A model's name is its `title` and its comment its `description`; the English name and comment stay as
//! `englishName` and `englishComment`. A motion's `frameRate` is 30, the rate its frame numbers count at, and the
//! model it was made for is kept as `model`.
//!
//! A file cut short in its vertices, faces, materials, bones or morphs, a negative count, or a count past the end
//! of the file is an error. Display frames and physics after the morphs are read as far as they go.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::Input;
use super::scene::{
    self, Animation, Bin, Channel, Interp, Node, Path as Key, Primitive, Scene, SceneMesh, Skin, Target,
};
use crate::Result;
use crate::assets::Assets;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::{Material, srgb_to_linear};
use crate::num::{f32_of, u32_of};
use crate::source::{Axis, Fact, Handedness, SourceInfo};

/// Meters per MMD unit.
const UNIT: f32 = 0.08;
/// Frames per second of a motion's keys.
const FPS: f64 = 30.0;
/// The most morph offsets held for targets, over all primitives.
const MAX_MORPH_DATA: usize = 1 << 25;
/// The most target-vertex pairs written (targets times vertices, over all primitives).
const MAX_MORPH_WORK: usize = 1 << 27;
/// The most target-primitive pairs written.
const MAX_TARGET_PRIMS: usize = 1 << 18;
/// The most steps a curve is sampled at between two keys.
const MAX_SAMPLES: u32 = 60;
/// The deepest group morphs nest.
const MAX_GROUP_DEPTH: usize = 8;

/// Euler angle limits: the least angles about X, Y and Z, then the most.
type Limits = ([f32; 3], [f32; 3]);

/// Keys: their times in seconds, and their values one after another.
type Keys = (Vec<f32>, Vec<f32>);

/// A vertex: position, normal, texture coordinates, and up to four bones with their weights.
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    bones: [i32; 4],
    weights: [f32; 4],
    /// How many of `bones` the deform uses: 1, 2 or 4.
    links: u8,
}

/// A material's toon texture.
enum Toon {
    None,
    /// One of the ten shared toons, 0 for `toon01.bmp`.
    Shared(u8),
    Texture(String),
}

/// A material as the file has it.
struct Mat {
    name: String,
    english: String,
    diffuse: [f32; 4],
    specular: [f32; 3],
    power: f32,
    ambient: [f32; 3],
    flags: u8,
    edge_color: [f32; 4],
    edge_size: f32,
    texture: Option<String>,
    sphere: Option<String>,
    sphere_mode: u8,
    toon: Toon,
    memo: String,
    /// The vertex indices its faces take, in order after the materials before it.
    count: usize,
}

/// Where a bone's tail points.
enum Tail {
    Bone(i32),
    Offset([f32; 3]),
}

/// A bone's inverse kinematics: the bone it moves to its own place, the solver's loops and angle limit, and the
/// chain it turns, each link with optional Euler angle limits.
struct Ik {
    target: i32,
    loops: i32,
    limit: f32,
    links: Vec<(i32, Option<Limits>)>,
}

/// A bone as the file has it.
struct Bone {
    name: String,
    english: String,
    position: [f32; 3],
    parent: i32,
    layer: i32,
    flags: u16,
    tail: Tail,
    inherit: Option<(i32, f32)>,
    fixed_axis: Option<[f32; 3]>,
    local_axes: Option<([f32; 3], [f32; 3])>,
    external: Option<i32>,
    ik: Option<Ik>,
    /// A PMD bone's type.
    pmd_type: Option<u8>,
}

/// What a morph moves.
enum MorphData {
    Group(Vec<(i32, f32)>),
    Vertex(Vec<(u32, [f32; 3])>),
    Bone(Vec<(i32, [f32; 3], [f32; 4])>),
    /// UV morphs: how many offsets.
    Uv(usize),
    Material(Vec<Value>),
    Flip(Vec<(i32, f32)>),
    /// Impulse morphs: how many offsets.
    Impulse(usize),
}

/// A morph as the file has it.
struct Morph {
    name: String,
    english: String,
    panel: u8,
    kind: u8,
    data: MorphData,
}

/// What a model file holds.
struct Model {
    format: &'static str,
    version: f32,
    encoding: &'static str,
    name: String,
    english_name: String,
    comment: String,
    english_comment: String,
    vertices: Vec<Vertex>,
    /// Additional UVs, `add_uvs` per vertex.
    extra_uvs: Vec<[f32; 4]>,
    add_uvs: usize,
    indices: Vec<u32>,
    materials: Vec<Mat>,
    bones: Vec<Bone>,
    morphs: Vec<Morph>,
    display_frames: usize,
    rigid_bodies: usize,
    joints: usize,
    soft_bodies: usize,
    toon_textures: Vec<String>,
    sdef: usize,
    qdef: usize,
    /// Why the sections after the morphs weren't read to the end, if they weren't.
    unread: Option<String>,
}

impl Model {
    fn new(format: &'static str, version: f32, encoding: &'static str) -> Self {
        Self {
            format,
            version,
            encoding,
            name: String::new(),
            english_name: String::new(),
            comment: String::new(),
            english_comment: String::new(),
            vertices: Vec::new(),
            extra_uvs: Vec::new(),
            add_uvs: 0,
            indices: Vec::new(),
            materials: Vec::new(),
            bones: Vec::new(),
            morphs: Vec::new(),
            display_frames: 0,
            rigid_bodies: 0,
            joints: 0,
            soft_bodies: 0,
            toon_textures: Vec::new(),
            sdef: 0,
            qdef: 0,
            unread: None,
        }
    }
}

/// Reads a `.pmx`, `.pmd` or `.vmd` file, told apart by its signature.
pub(crate) fn load(input: Input<'_>, ext: &str, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let bytes = input.read()?;
    let stem = input
        .path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    if bytes.starts_with(b"Vocaloid Motion Data") {
        return motion(&bytes, &stem, keep);
    }
    let model = if bytes.starts_with(b"PMX ") {
        pmx(&bytes)?
    } else if bytes.starts_with(b"Pmd") {
        pmd(&bytes)?
    } else {
        bail!("isn't a {} file", ext.to_ascii_uppercase())
    };
    build(model, &input, &stem, assets, keep)
}

/// Shift-JIS text, without the broken character a field cut short leaves at its end.
fn sjis(b: &[u8]) -> String {
    scene::shift_jis(b).trim_end_matches('\u{FFFD}').to_owned()
}

/// A position as glTF has it: mirrored and in meters.
fn pt(v: [f32; 3]) -> [f32; 3] {
    [v[0] * UNIT, v[1] * UNIT, -v[2] * UNIT]
}

/// A direction as glTF has it: mirrored.
fn dir(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[1], -v[2]]
}

/// Euler angle limits as glTF has them: mirroring turns the angles about X and Y the other way.
fn limits(min: [f32; 3], max: [f32; 3]) -> Limits {
    ([-max[0], -max[1], min[2]], [-min[0], -min[1], max[2]])
}

// ---------------------------------------------------------------------------------------------------------------
// PMX.

/// A PMX file being read: its text encoding and index sizes.
struct Reader<'a> {
    b: Bin<'a>,
    utf8: bool,
    vertex: u8,
    texture: u8,
    material: u8,
    bone: u8,
    morph: u8,
    rigid: u8,
}

impl Reader<'_> {
    /// A text field: its length in bytes, then its bytes.
    fn text(&mut self) -> Result<String> {
        let n = self.b.i32()?;
        ensure!(n >= 0, "PMX text has a negative length");
        let n = self.b.count(u64::from(n.unsigned_abs()), 1)?;
        let b = self.b.bytes(n)?;
        Ok(if self.utf8 {
            String::from_utf8_lossy(b).into_owned()
        } else {
            let units: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&units)
        })
    }

    /// A count of items of at least `size` bytes each.
    fn count(&mut self, size: usize) -> Result<usize> {
        let n = self.b.i32()?;
        ensure!(n >= 0, "PMX file has a negative count");
        self.b.count(u64::from(n.unsigned_abs()), size)
    }

    /// An index of `size` bytes, signed (-1 for none).
    fn index(&mut self, size: u8) -> Result<i32> {
        Ok(match size {
            1 => i32::from(self.b.i8()?),
            2 => i32::from(self.b.i16()?),
            _ => self.b.i32()?,
        })
    }

    /// A vertex index: unsigned when 1 or 2 bytes, signed when 4 (a negative one is out of range).
    fn vertex_index(&mut self) -> Result<u32> {
        Ok(match self.vertex {
            1 => u32::from(self.b.u8()?),
            2 => u32::from(self.b.u16()?),
            _ => u32::try_from(self.b.i32()?).unwrap_or(u32::MAX),
        })
    }
}

/// Reads a PMX file.
fn pmx(data: &[u8]) -> Result<Model> {
    let mut b = Bin::new(data, "PMX");
    ensure!(b.bytes(4)? == b"PMX ", "isn't a PMX file");
    let version = b.f32()?;
    ensure!(
        (version - 2.0).abs() < 1e-3 || (version - 2.1).abs() < 1e-3,
        "PMX version {version} isn't 2.0 or 2.1"
    );
    let globals = usize::from(b.u8()?);
    ensure!(globals >= 8, "PMX header has {globals} settings, fewer than 8");
    let g = b.bytes(globals)?;
    ensure!(g[0] <= 1, "PMX text encoding {} isn't UTF-16 or UTF-8", g[0]);
    ensure!(g[1] <= 4, "PMX file has {} additional UVs, more than 4", g[1]);
    for &s in &g[2..8] {
        ensure!(matches!(s, 1 | 2 | 4), "PMX index size {s} isn't 1, 2 or 4");
    }
    let mut r = Reader {
        b,
        utf8: g[0] == 1,
        vertex: g[2],
        texture: g[3],
        material: g[4],
        bone: g[5],
        morph: g[6],
        rigid: g[7],
    };
    let mut m = Model::new("PMX", version, if r.utf8 { "UTF-8" } else { "UTF-16LE" });
    m.add_uvs = usize::from(g[1]);
    m.name = r.text()?;
    m.english_name = r.text()?;
    m.comment = r.text()?;
    m.english_comment = r.text()?;

    // Vertices.
    let bone = usize::from(r.bone);
    let n = r.count(32 + 16 * m.add_uvs + 1 + bone + 4)?;
    m.vertices.reserve(n);
    m.extra_uvs.reserve(n * m.add_uvs);
    for _ in 0..n {
        let position = r.b.vec3()?;
        let normal = r.b.vec3()?;
        let uv = r.b.vec2()?;
        for _ in 0..m.add_uvs {
            m.extra_uvs.push(r.b.vec4()?);
        }
        let mut v = Vertex {
            position,
            normal,
            uv,
            bones: [-1; 4],
            weights: [0.0; 4],
            links: 1,
        };
        match r.b.u8()? {
            0 => {
                v.bones[0] = r.index(r.bone)?;
                v.weights[0] = 1.0;
            }
            kind @ (1 | 3) => {
                v.bones[0] = r.index(r.bone)?;
                v.bones[1] = r.index(r.bone)?;
                let w = r.b.f32()?;
                v.weights = [w, 1.0 - w, 0.0, 0.0];
                v.links = 2;
                if kind == 3 {
                    // SDEF's C, R0 and R1: the spherical blend drops to the linear one.
                    r.b.skip(36)?;
                    m.sdef += 1;
                }
            }
            kind @ (2 | 4) => {
                for b in &mut v.bones {
                    *b = r.index(r.bone)?;
                }
                for w in &mut v.weights {
                    *w = r.b.f32()?;
                }
                v.links = 4;
                if kind == 4 {
                    m.qdef += 1;
                }
            }
            t => bail!("PMX vertex has an unknown deform type {t}"),
        }
        r.b.f32()?;
        m.vertices.push(v);
    }

    // Faces.
    let n = r.count(usize::from(r.vertex))?;
    m.indices.reserve(n);
    for _ in 0..n {
        m.indices.push(r.vertex_index()?);
    }

    // Textures.
    let n = r.count(4)?;
    let mut textures = Vec::with_capacity(n);
    for _ in 0..n {
        textures.push(r.text()?);
    }
    let texture = |i: i32| usize::try_from(i).ok().and_then(|i| textures.get(i)).cloned();

    // Materials.
    let tex = usize::from(r.texture);
    let n = r.count(8 + 16 + 12 + 4 + 12 + 1 + 16 + 4 + 2 * tex + 3 + 4 + 4)?;
    for _ in 0..n {
        let name = r.text()?;
        let english = r.text()?;
        let diffuse = r.b.vec4()?;
        let specular = r.b.vec3()?;
        let power = r.b.f32()?;
        let ambient = r.b.vec3()?;
        let flags = r.b.u8()?;
        let edge_color = r.b.vec4()?;
        let edge_size = r.b.f32()?;
        let tex_i = r.index(r.texture)?;
        let sphere_i = r.index(r.texture)?;
        let sphere_mode = r.b.u8()?;
        let toon = if r.b.u8()? == 0 {
            texture(r.index(r.texture)?).map_or(Toon::None, Toon::Texture)
        } else {
            Toon::Shared(r.b.u8()?)
        };
        let memo = r.text()?;
        let count = r.b.i32()?;
        ensure!(count >= 0, "PMX material has a negative face count");
        m.materials.push(Mat {
            name,
            english,
            diffuse,
            specular,
            power,
            ambient,
            flags,
            edge_color,
            edge_size,
            texture: texture(tex_i),
            sphere: if sphere_mode == 0 { None } else { texture(sphere_i) },
            sphere_mode,
            toon,
            memo,
            count: usize::try_from(count).unwrap_or(0),
        });
    }

    // Bones.
    let n = r.count(8 + 12 + bone + 4 + 2 + bone)?;
    for _ in 0..n {
        let name = r.text()?;
        let english = r.text()?;
        let position = r.b.vec3()?;
        let parent = r.index(r.bone)?;
        let layer = r.b.i32()?;
        let flags = r.b.u16()?;
        let tail = if flags & 0x0001 != 0 {
            Tail::Bone(r.index(r.bone)?)
        } else {
            Tail::Offset(r.b.vec3()?)
        };
        let inherit = if flags & 0x0300 != 0 {
            Some((r.index(r.bone)?, r.b.f32()?))
        } else {
            None
        };
        let fixed_axis = if flags & 0x0400 != 0 { Some(r.b.vec3()?) } else { None };
        let local_axes = if flags & 0x0800 != 0 {
            Some((r.b.vec3()?, r.b.vec3()?))
        } else {
            None
        };
        let external = if flags & 0x2000 != 0 { Some(r.b.i32()?) } else { None };
        let ik = if flags & 0x0020 != 0 {
            let target = r.index(r.bone)?;
            let loops = r.b.i32()?;
            let limit = r.b.f32()?;
            let count = r.count(bone + 1)?;
            let mut links = Vec::with_capacity(count);
            for _ in 0..count {
                let b = r.index(r.bone)?;
                let lim = if r.b.u8()? != 0 {
                    Some((r.b.vec3()?, r.b.vec3()?))
                } else {
                    None
                };
                links.push((b, lim));
            }
            Some(Ik {
                target,
                loops,
                limit,
                links,
            })
        } else {
            None
        };
        m.bones.push(Bone {
            name,
            english,
            position,
            parent,
            layer,
            flags,
            tail,
            inherit,
            fixed_axis,
            local_axes,
            external,
            ik,
            pmd_type: None,
        });
    }

    // Morphs.
    let n = r.count(8 + 2 + 4)?;
    for _ in 0..n {
        let name = r.text()?;
        let english = r.text()?;
        let panel = r.b.u8()?;
        let kind = r.b.u8()?;
        let size = match kind {
            0 | 9 => usize::from(r.morph) + 4,
            1 => usize::from(r.vertex) + 12,
            2 => bone + 28,
            3..=7 => usize::from(r.vertex) + 16,
            8 => usize::from(r.material) + 113,
            10 => usize::from(r.rigid) + 25,
            k => bail!("PMX morph has an unknown type {k}"),
        };
        let count = r.count(size)?;
        let data = match kind {
            0 | 9 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push((r.index(r.morph)?, r.b.f32()?));
                }
                if kind == 0 {
                    MorphData::Group(v)
                } else {
                    MorphData::Flip(v)
                }
            }
            1 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push((r.vertex_index()?, r.b.vec3()?));
                }
                MorphData::Vertex(v)
            }
            2 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    v.push((r.index(r.bone)?, r.b.vec3()?, r.b.vec4()?));
                }
                MorphData::Bone(v)
            }
            3..=7 => {
                r.b.skip(count * size)?;
                MorphData::Uv(count)
            }
            8 => {
                let mut v = Vec::with_capacity(count);
                for _ in 0..count {
                    let material = r.index(r.material)?;
                    let op = r.b.u8()?;
                    v.push(json!({
                        "material": material,
                        "operation": if op == 0 { "multiply" } else { "add" },
                        "diffuse": r.b.vec4()?,
                        "specular": r.b.vec3()?,
                        "specularPower": r.b.f32()?,
                        "ambient": r.b.vec3()?,
                        "edgeColor": r.b.vec4()?,
                        "edgeSize": r.b.f32()?,
                        "texture": r.b.vec4()?,
                        "sphere": r.b.vec4()?,
                        "toon": r.b.vec4()?,
                    }));
                }
                MorphData::Material(v)
            }
            _ => {
                r.b.skip(count * size)?;
                MorphData::Impulse(count)
            }
        };
        m.morphs.push(Morph {
            name,
            english,
            panel,
            kind,
            data,
        });
    }

    if let Err(e) = pmx_tail(&mut r, &mut m) {
        m.unread = Some(e.to_string());
    }
    Ok(m)
}

/// The sections after the morphs, counted: display frames, rigid bodies, joints and (in 2.1) soft bodies.
fn pmx_tail(r: &mut Reader<'_>, m: &mut Model) -> Result<()> {
    let n = r.count(8 + 1 + 4)?;
    for _ in 0..n {
        r.text()?;
        r.text()?;
        r.b.u8()?;
        let k = r.count(2)?;
        for _ in 0..k {
            let size = if r.b.u8()? == 0 { r.bone } else { r.morph };
            r.index(size)?;
        }
    }
    m.display_frames = n;
    let n = r.count(8 + usize::from(r.bone) + 61)?;
    for _ in 0..n {
        r.text()?;
        r.text()?;
        r.index(r.bone)?;
        r.b.skip(61)?;
    }
    m.rigid_bodies = n;
    let n = r.count(8 + 1 + 2 * usize::from(r.rigid) + 96)?;
    for _ in 0..n {
        r.text()?;
        r.text()?;
        r.b.u8()?;
        r.index(r.rigid)?;
        r.index(r.rigid)?;
        r.b.skip(96)?;
    }
    m.joints = n;
    if m.version > 2.05 && r.b.left() >= 4 {
        m.soft_bodies = r.count(1)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// PMD.

/// Reads a PMD file.
fn pmd(data: &[u8]) -> Result<Model> {
    let mut b = Bin::new(data, "PMD");
    ensure!(b.bytes(3)? == b"Pmd", "isn't a PMD file");
    let version = b.f32()?;
    let mut m = Model::new("PMD", version, "Shift-JIS");
    m.name = sjis(b.fixed_str(20)?);
    m.comment = sjis(b.fixed_str(256)?);

    let c = b.u32()?;
    let n = b.count(c.into(), 38)?;
    m.vertices.reserve(n);
    for _ in 0..n {
        let position = b.vec3()?;
        let normal = b.vec3()?;
        let uv = b.vec2()?;
        let b0 = b.u16()?;
        let b1 = b.u16()?;
        let w = f32::from(b.u8()?.min(100)) / 100.0;
        b.u8()?;
        let bone = |x: u16| if x == u16::MAX { -1 } else { i32::from(x) };
        m.vertices.push(Vertex {
            position,
            normal,
            uv,
            bones: [bone(b0), bone(b1), -1, -1],
            weights: [w, 1.0 - w, 0.0, 0.0],
            links: 2,
        });
    }

    let c = b.u32()?;
    let n = b.count(c.into(), 2)?;
    m.indices.reserve(n);
    for _ in 0..n {
        m.indices.push(u32::from(b.u16()?));
    }

    let c = b.u32()?;
    let n = b.count(c.into(), 70)?;
    for i in 0..n {
        let d = b.vec3()?;
        let alpha = b.f32()?;
        let power = b.f32()?;
        let specular = b.vec3()?;
        let ambient = b.vec3()?;
        let toon = b.u8()?;
        let edge = b.u8()?;
        let count = b.u32()?;
        let file = sjis(b.fixed_str(20)?);
        // "color.bmp*sphere.sph": a texture and a sphere map, either alone too.
        let (mut texture, mut sphere, mut sphere_mode) = (None, None, 0);
        for part in file.split('*').filter(|p| !p.is_empty()) {
            let ext = |e: &str| {
                std::path::Path::new(part)
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case(e))
            };
            if ext("sph") {
                sphere = Some(part.to_owned());
                sphere_mode = 1;
            } else if ext("spa") {
                sphere = Some(part.to_owned());
                sphere_mode = 2;
            } else {
                texture = Some(part.to_owned());
            }
        }
        // MMD draws a PMD material that isn't opaque from both sides.
        let mut flags = u8::from(alpha < 1.0);
        if edge != 0 {
            flags |= 0x10;
        }
        m.materials.push(Mat {
            name: format!("material {i}"),
            english: String::new(),
            diffuse: [d[0], d[1], d[2], alpha],
            specular,
            power,
            ambient,
            flags,
            edge_color: [0.0, 0.0, 0.0, 1.0],
            edge_size: if edge != 0 { 1.0 } else { 0.0 },
            texture,
            sphere,
            sphere_mode,
            toon: if toon == u8::MAX {
                Toon::None
            } else {
                Toon::Shared(toon)
            },
            memo: String::new(),
            count: usize::try_from(count).unwrap_or(usize::MAX),
        });
    }

    let c = b.u16()?;
    let n = b.count(c.into(), 39)?;
    for _ in 0..n {
        let name = sjis(b.fixed_str(20)?);
        let parent = b.u16()?;
        let tail = b.u16()?;
        let kind = b.u8()?;
        b.u16()?;
        let position = b.vec3()?;
        let index = |x: u16| if x == u16::MAX { -1 } else { i32::from(x) };
        m.bones.push(Bone {
            name,
            english: String::new(),
            position,
            parent: index(parent),
            layer: 0,
            flags: 0,
            tail: Tail::Bone(if tail == 0 { -1 } else { index(tail) }),
            inherit: None,
            fixed_axis: None,
            local_axes: None,
            external: None,
            ik: None,
            pmd_type: Some(kind),
        });
    }

    let c = b.u16()?;
    let n = b.count(c.into(), 11)?;
    for _ in 0..n {
        let bone = b.u16()?;
        let target = b.u16()?;
        let len = b.u8()?;
        let loops = b.u16()?;
        let limit = b.f32()?;
        let mut links = Vec::with_capacity(usize::from(len));
        for _ in 0..len {
            links.push((i32::from(b.u16()?), None));
        }
        if let Some(bn) = m.bones.get_mut(usize::from(bone)) {
            bn.ik = Some(Ik {
                target: i32::from(target),
                loops: i32::from(loops),
                limit,
                links,
            });
        }
    }

    // Morphs: the base lists the vertices the others move, by its own numbering.
    let c = b.u16()?;
    let n = b.count(c.into(), 25)?;
    let mut base: Vec<u32> = Vec::new();
    for _ in 0..n {
        let name = sjis(b.fixed_str(20)?);
        let c = b.u32()?;
        let kind = b.u8()?;
        let count = b.count(c.into(), 16)?;
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            offsets.push((b.u32()?, b.vec3()?));
        }
        if kind == 0 {
            base = offsets.iter().map(|o| o.0).collect();
            continue;
        }
        let offsets = offsets
            .into_iter()
            .map(|(i, d)| {
                (
                    usize::try_from(i)
                        .ok()
                        .and_then(|i| base.get(i))
                        .copied()
                        .unwrap_or(u32::MAX),
                    d,
                )
            })
            .collect();
        m.morphs.push(Morph {
            name,
            english: String::new(),
            panel: kind,
            kind: 1,
            data: MorphData::Vertex(offsets),
        });
    }

    if let Err(e) = pmd_tail(&mut b, &mut m) {
        m.unread = Some(e.to_string());
    }
    Ok(m)
}

/// The sections after the morphs: display lists, English names, toon textures, rigid bodies and joints.
fn pmd_tail(b: &mut Bin<'_>, m: &mut Model) -> Result<()> {
    if b.left() == 0 {
        return Ok(());
    }
    let n = usize::from(b.u8()?);
    b.skip(2 * n)?;
    let frames = usize::from(b.u8()?);
    b.skip(50 * frames)?;
    m.display_frames = frames;
    let c = b.u32()?;
    let n = b.count(c.into(), 3)?;
    b.skip(3 * n)?;
    if b.left() == 0 {
        return Ok(());
    }
    if b.u8()? != 0 {
        m.english_name = sjis(b.fixed_str(20)?);
        m.english_comment = sjis(b.fixed_str(256)?);
        for bone in &mut m.bones {
            bone.english = sjis(b.fixed_str(20)?);
        }
        for morph in &mut m.morphs {
            morph.english = sjis(b.fixed_str(20)?);
        }
        b.skip(50 * frames)?;
    }
    if b.left() < 1000 {
        return Ok(());
    }
    for _ in 0..10 {
        m.toon_textures.push(sjis(b.fixed_str(100)?));
    }
    if b.left() < 4 {
        return Ok(());
    }
    let c = b.u32()?;
    m.rigid_bodies = b.count(c.into(), 83)?;
    b.skip(83 * m.rigid_bodies)?;
    let c = b.u32()?;
    m.joints = b.count(c.into(), 124)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// The model as a scene.

/// A name, or what stands in for an empty one.
fn name_of(name: &str, i: usize, what: &str) -> String {
    if name.trim().is_empty() {
        format!("{what} {i}")
    } else {
        name.to_owned()
    }
}

/// Roughness from a Phong specular power, by the usual match of Blinn-Phong to GGX; fully rough without a
/// highlight.
fn roughness(power: f32, specular: [f32; 3]) -> f32 {
    if !power.is_finite() || power <= 0.0 || !specular.iter().any(|&c| c > 0.0) {
        return 1.0;
    }
    (2.0 / (power + 2.0)).powf(0.25).clamp(0.0, 1.0)
}

/// Whether a texture file is of a format whose alpha is there on purpose.
fn cuts_out(file: &str) -> bool {
    let lower = file.to_ascii_lowercase();
    [".png", ".tga", ".dds"].iter().any(|e| lower.ends_with(e))
}

/// A bone's `extras`: what glTF has no place for, with node indices for bone indices.
fn bone_extras(bone: &Bone, valid: &dyn Fn(i32) -> Option<usize>) -> Option<Value> {
    let mut e = Map::new();
    if !bone.english.is_empty() {
        e.insert("englishName".into(), json!(bone.english));
    }
    if let Some(t) = bone.pmd_type {
        e.insert("pmdType".into(), json!(t));
    } else {
        e.insert("flags".into(), json!(bone.flags));
        if bone.layer != 0 {
            e.insert("layer".into(), json!(bone.layer));
        }
    }
    match bone.tail {
        Tail::Bone(i) => {
            if let Some(i) = valid(i) {
                e.insert("tail".into(), json!({ "bone": i }));
            }
        }
        Tail::Offset(v) => {
            if v.iter().any(|&c| c != 0.0) {
                e.insert("tail".into(), json!({ "offset": pt(v) }));
            }
        }
    }
    if let Some((from, ratio)) = bone.inherit
        && let Some(from) = valid(from)
    {
        e.insert(
            "inherit".into(),
            json!({
                "bone": from,
                "ratio": ratio,
                "rotation": bone.flags & 0x0100 != 0,
                "translation": bone.flags & 0x0200 != 0,
                "local": bone.flags & 0x0080 != 0,
            }),
        );
    }
    if let Some(a) = bone.fixed_axis {
        e.insert("fixedAxis".into(), json!(dir(a)));
    }
    if let Some((x, z)) = bone.local_axes {
        e.insert("localAxes".into(), json!({ "x": dir(x), "z": dir(z) }));
    }
    if let Some(k) = bone.external {
        e.insert("externalParentKey".into(), json!(k));
    }
    if let Some(ik) = &bone.ik {
        let links: Vec<Value> = ik
            .links
            .iter()
            .filter_map(|(b, lim)| {
                let mut l = json!({ "bone": valid(*b)? });
                if let Some((min, max)) = lim {
                    let (min, max) = limits(*min, *max);
                    l["min"] = json!(min);
                    l["max"] = json!(max);
                }
                Some(l)
            })
            .collect();
        let mut j = json!({ "loops": ik.loops, "links": links });
        if let Some(t) = valid(ik.target) {
            j["target"] = json!(t);
        }
        // PMD's limit is a control weight, PMX's an angle in radians.
        let key = if bone.pmd_type.is_some() {
            "controlWeight"
        } else {
            "limitAngle"
        };
        j[key] = json!(ik.limit);
        e.insert("ik".into(), j);
    }
    (!e.is_empty()).then_some(Value::Object(e))
}

/// A morph's entry in the mesh's `extras.morphs`, for the morphs that aren't vertex morphs.
fn morph_extras(i: usize, morph: &Morph, bone: &dyn Fn(i32) -> Option<usize>) -> Option<Value> {
    let kind = match morph.kind {
        0 => "group",
        2 => "bone",
        3 => "uv",
        4..=7 => "additionalUv",
        8 => "material",
        9 => "flip",
        10 => "impulse",
        _ => return None,
    };
    let mut e = json!({ "index": i, "name": morph.name, "type": kind, "panel": morph.panel });
    if !morph.english.is_empty() {
        e["englishName"] = json!(morph.english);
    }
    if (4..=7).contains(&morph.kind) {
        e["uv"] = json!(morph.kind - 3);
    }
    match &morph.data {
        MorphData::Group(v) | MorphData::Flip(v) => {
            let members: Vec<Value> = v.iter().map(|(m, r)| json!({ "morph": m, "ratio": r })).collect();
            e["members"] = json!(members);
        }
        MorphData::Bone(v) => {
            let offsets: Vec<Value> = v
                .iter()
                .filter_map(|(b, t, q)| {
                    let q = scene::mirror_quat(scene::normalize(q.map(f64::from)));
                    Some(json!({ "bone": bone(*b)?, "translation": pt(*t), "rotation": q }))
                })
                .collect();
            e["offsets"] = json!(offsets);
        }
        MorphData::Material(v) => e["offsets"] = json!(v),
        MorphData::Uv(n) | MorphData::Impulse(n) => e["count"] = json!(n),
        MorphData::Vertex(_) => return None,
    }
    Some(e)
}

/// Adds the vertex offsets morph `i` makes, scaled by `ratio`, to `acc`: a vertex morph's own, a group's members'
/// in turn. `budget` is the offsets and steps still allowed.
fn expand(morphs: &[Morph], i: usize, ratio: f32, depth: usize, acc: &mut HashMap<u32, [f32; 3]>, budget: &mut usize) {
    if depth > MAX_GROUP_DEPTH || *budget == 0 {
        return;
    }
    *budget -= 1;
    match morphs.get(i).map(|m| &m.data) {
        Some(MorphData::Vertex(o)) => {
            for &(v, d) in o {
                if *budget == 0 {
                    return;
                }
                *budget -= 1;
                let e = acc.entry(v).or_insert([0.0; 3]);
                for k in 0..3 {
                    e[k] += d[k] * ratio;
                }
            }
        }
        Some(MorphData::Group(members)) => {
            for &(j, r) in members {
                if let Ok(j) = usize::try_from(j)
                    && r.is_finite()
                {
                    expand(morphs, j, ratio * r, depth + 1, acc, budget);
                }
            }
        }
        _ => {}
    }
}

/// The model as a glTF document.
fn build(m: Model, input: &Input<'_>, stem: &str, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let mut scene = Scene::default();
    let mut facts = Map::new();
    let nb = m.bones.len();
    let valid = |i: i32| usize::try_from(i).ok().filter(|&i| i < nb);

    // Bones, first, so a bone's index is its node's.
    for (i, bone) in m.bones.iter().enumerate() {
        let parent = valid(bone.parent).filter(|&p| p != i);
        let mut node = Node::named(name_of(&bone.name, i, "bone"), parent);
        let at = parent.map_or([0.0; 3], |p| m.bones[p].position);
        node.translation = std::array::from_fn(|k| bone.position[k] - at[k]);
        node.extras = bone_extras(bone, &valid);
        scene.add(node);
    }

    // Materials.
    let dir_ = input.dir();
    let mut images = Images::new(!keep.image_bytes, dir_);
    let mut materials_extra = Vec::new();
    for (i, mat) in m.materials.iter().enumerate() {
        let unit = |c: f32| if c.is_finite() { c.clamp(0.0, 1.0) } else { 1.0 };
        let d = mat.diffuse;
        let color = [
            srgb_to_linear(unit(d[0])),
            srgb_to_linear(unit(d[1])),
            srgb_to_linear(unit(d[2])),
            unit(d[3]),
        ];
        let mut out = Material::plain(&name_of(&mat.name, i, "material"), color);
        out.double_sided = mat.flags & 0x01 != 0;
        out.roughness = roughness(mat.power, mat.specular);
        if let Some(t) = &mat.texture
            && let Some(path) = assets.find(t, dir_)
        {
            // Decided by the file alone, so a read without textures cuts out as a full one does.
            if color[3] >= 1.0 && cuts_out(t) {
                out.alpha_cutoff = Some(0.5);
            }
            if keep.textures {
                out.color_image = images.file(assets, path, t);
            }
        }
        scene.materials.push(out);

        let mut e = json!({
            "name": mat.name,
            "diffuse": mat.diffuse,
            "specular": mat.specular,
            "specularPower": mat.power,
            "ambient": mat.ambient,
            "flags": mat.flags,
            "edgeColor": mat.edge_color,
            "edgeSize": mat.edge_size,
        });
        if !mat.english.is_empty() {
            e["englishName"] = json!(mat.english);
        }
        if let Some(t) = &mat.texture {
            e["texture"] = json!(t);
        }
        if let Some(s) = &mat.sphere {
            e["sphere"] = json!(s);
            e["sphereMode"] = json!(match mat.sphere_mode {
                1 => "multiply",
                2 => "add",
                _ => "subtexture",
            });
        }
        match &mat.toon {
            Toon::None => {}
            // A PMD lists its own ten toon files, which a material's index names; PMX's shared toons are fixed.
            Toon::Shared(k) => match m.toon_textures.get(usize::from(*k)).filter(|t| !t.is_empty()) {
                Some(t) => e["toon"] = json!(t),
                None => e["toon"] = json!(format!("toon{:02}.bmp", u32::from(*k) + 1)),
            },
            Toon::Texture(t) => e["toon"] = json!(t),
        }
        if !mat.memo.is_empty() {
            e["memo"] = json!(mat.memo);
        }
        materials_extra.push(e);
    }
    scene.images = images.list;

    // A primitive per material, over the faces in order.
    let nv = m.vertices.len();
    let mut ranges: Vec<(Option<usize>, usize, usize)> = Vec::new();
    let mut at = 0usize;
    let mut past_end = false;
    for (i, mat) in m.materials.iter().enumerate() {
        let want = at.saturating_add(mat.count);
        past_end |= want > m.indices.len();
        let end = want.min(m.indices.len());
        ranges.push((Some(i), at, end));
        at = end;
    }
    if at < m.indices.len() {
        facts.insert("faces_without_material".into(), json!((m.indices.len() - at) / 3));
        ranges.push((None, at, m.indices.len()));
    }
    if past_end {
        facts.insert("material_faces_past_end".into(), json!(true));
    }
    let colored = m.version > 2.05 && m.add_uvs > 0;
    let mut remap = vec![u32::MAX; nv];
    let mut prims: Vec<Primitive> = Vec::new();
    let mut globals: Vec<Vec<u32>> = Vec::new();
    let mut dropped = 0usize;
    let mut unbound = false;
    let mut zw = false;
    for &(material, start, end) in &ranges {
        let vertex_colors = colored && material.is_some_and(|i| m.materials[i].flags & 0x20 != 0);
        let mut p = Primitive {
            material,
            uvs: vec![Vec::new(); 1 + m.add_uvs],
            colors: if vertex_colors { vec![Vec::new()] } else { Vec::new() },
            ..Primitive::default()
        };
        let mut used: Vec<u32> = Vec::new();
        for tri in m.indices[start..end].chunks_exact(3) {
            if tri.iter().any(|&v| v as usize >= nv) {
                dropped += 1;
                continue;
            }
            for &v in tri {
                let slot = &mut remap[v as usize];
                if *slot == u32::MAX {
                    *slot = u32_of(used.len());
                    used.push(v);
                }
                p.indices.push(*slot);
            }
        }
        let width = used
            .iter()
            .map(|&v| usize::from(m.vertices[v as usize].links))
            .max()
            .unwrap_or(1);
        if nb > 0 {
            p.influences = width;
            p.rest_joint = u32_of(nb);
        }
        for &v in &used {
            let vx = &m.vertices[v as usize];
            p.positions.push(vx.position);
            p.normals.push(vx.normal);
            p.uvs[0].push(vx.uv);
            for k in 0..m.add_uvs {
                let a = m.extra_uvs[v as usize * m.add_uvs + k];
                zw |= a[2] != 0.0 || a[3] != 0.0;
                p.uvs[1 + k].push([a[0], a[1]]);
            }
            if vertex_colors {
                let a = m.extra_uvs[v as usize * m.add_uvs];
                p.colors[0].push(a.map(|c| if c.is_finite() { c.clamp(0.0, 1.0) } else { 1.0 }));
            }
            if nb > 0 {
                let mut any = false;
                for k in 0..width {
                    let (j, w) = (vx.bones[k], vx.weights[k]);
                    match valid(j) {
                        Some(j) if k < usize::from(vx.links) && w.is_finite() && w > 0.0 => {
                            any = true;
                            p.joints.push(u32_of(j));
                            p.weights.push(w);
                        }
                        _ => {
                            p.joints.push(0);
                            p.weights.push(0.0);
                        }
                    }
                }
                unbound |= !any;
            }
        }
        for &v in &used {
            remap[v as usize] = u32::MAX;
        }
        prims.push(p);
        globals.push(used);
    }
    drop(remap);
    if dropped > 0 {
        facts.insert("dropped_triangles".into(), json!(dropped));
    }
    if zw {
        facts.insert("additional_uv_zw_dropped".into(), json!(true));
    }

    // Morph targets: vertex morphs, and group morphs that move vertices.
    let mut target_names = Vec::new();
    let mut target_offsets: Vec<Vec<(u32, [f32; 3])>> = Vec::new();
    if keep.morph_targets && !prims.is_empty() {
        let mut budget = MAX_MORPH_DATA;
        for (i, morph) in m.morphs.iter().enumerate() {
            let offsets = match &morph.data {
                MorphData::Vertex(o) => o.clone(),
                MorphData::Group(_) => {
                    let mut acc = HashMap::new();
                    expand(&m.morphs, i, 1.0, 0, &mut acc, &mut budget);
                    let mut v: Vec<(u32, [f32; 3])> =
                        acc.into_iter().filter(|(_, d)| d.iter().any(|&c| c != 0.0)).collect();
                    v.sort_unstable_by_key(|o| o.0);
                    if v.is_empty() {
                        continue;
                    }
                    v
                }
                _ => continue,
            };
            target_names.push(name_of(&morph.name, i, "morph"));
            target_offsets.push(offsets);
        }
    }
    let total: usize = prims.iter().map(|p| p.positions.len()).sum();
    let mut most = target_offsets.len();
    if let Some(q) = MAX_MORPH_WORK.checked_div(total) {
        most = most.min(q);
    }
    most = most.min(MAX_TARGET_PRIMS / prims.len().max(1));
    if !target_offsets.is_empty() {
        // Where each vertex is in the primitives: (primitive, index there), vertex by vertex.
        let mut start = vec![0u32; nv + 1];
        for g in &globals {
            for &v in g {
                start[v as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut places = vec![(0u32, 0u32); total];
        for (pi, g) in globals.iter().enumerate() {
            for (li, &v) in g.iter().enumerate() {
                places[fill[v as usize] as usize] = (u32_of(pi), u32_of(li));
                fill[v as usize] += 1;
            }
        }
        let mut targets: Vec<Vec<Target>> = prims.iter().map(|_| Vec::new()).collect();
        let mut held = 0usize;
        'targets: for (t, offsets) in target_offsets.iter().enumerate().take(most) {
            for ts in &mut targets {
                ts.push(Target::default());
            }
            for &(v, d) in offsets {
                let Some(range) = start.get(v as usize..v as usize + 2) else {
                    continue;
                };
                if !d.iter().all(|c| c.is_finite()) {
                    continue;
                }
                for &(pi, li) in &places[range[0] as usize..range[1] as usize] {
                    let pi = pi as usize;
                    let tg = &mut targets[pi][t];
                    if tg.positions.is_empty() {
                        let len = prims[pi].positions.len();
                        if held + len > MAX_MORPH_DATA {
                            for ts in &mut targets {
                                ts.truncate(t);
                            }
                            most = t;
                            break 'targets;
                        }
                        held += len;
                        tg.positions = vec![[0.0; 3]; len];
                    }
                    let o = &mut tg.positions[li as usize];
                    for k in 0..3 {
                        o[k] += d[k];
                    }
                }
            }
        }
        for (p, ts) in prims.iter_mut().zip(targets) {
            p.targets = ts;
        }
        if most < target_offsets.len() {
            facts.insert("dropped_morph_targets".into(), json!(target_offsets.len() - most));
        }
        target_names.truncate(most);
    }

    // The mesh, skinned to every bone.
    let morphs_extra: Vec<Value> = m
        .morphs
        .iter()
        .enumerate()
        .filter_map(|(i, morph)| morph_extras(i, morph, &valid))
        .collect();
    let has_mesh = prims.iter().any(|p| !p.indices.is_empty());
    let model_name = name_of(&m.name, 0, "").trim().to_owned();
    let model_name = if model_name.is_empty() || model_name == "0" {
        stem.to_owned()
    } else {
        model_name
    };
    if has_mesh {
        let mut extras = Map::new();
        if !morphs_extra.is_empty() {
            extras.insert("morphs".into(), json!(morphs_extra));
        }
        scene.meshes.push(SceneMesh {
            name: model_name.clone(),
            prims,
            target_names,
            extras: (!extras.is_empty()).then_some(Value::Object(extras)),
            ..SceneMesh::default()
        });
        let mut node = Node::named(model_name.clone(), None);
        node.mesh = Some(0);
        if nb > 0 {
            let mut world = scene.world();
            world.truncate(nb);
            let mut skin = Skin {
                name: model_name.clone(),
                joints: (0..nb).collect(),
                inverse_binds: world.iter().map(scene::invert).collect(),
            };
            if unbound {
                let fixed = scene.add(Node::named("unbound", None));
                skin.joints.push(fixed);
                skin.inverse_binds.push(scene::IDENTITY);
            }
            scene.skins.push(skin);
            node.skin = Some(0);
        }
        scene.add(node);
    }
    ensure!(!scene.nodes.is_empty(), "{} file has no faces or bones", m.format);

    scene.mirror_z();
    scene.scale(UNIT);

    let mut source = Map::new();
    convention(&mut scene.info);
    scene.info.format = Some(m.format.into());
    // The version as written (2.1, not the nearest float to it, 2.0999999), with its tenths (2.0).
    let version = (f64::from(m.version) * 1000.0).round() / 1000.0;
    scene.info.version = Some(if version.fract() == 0.0 {
        format!("{version:.1}")
    } else {
        version.to_string()
    });
    let text = |s: &str| (!s.is_empty()).then(|| s.to_owned());
    scene.info.title = text(&m.name);
    scene.info.description = text(&m.comment);
    source.insert("encoding".into(), json!(m.encoding));
    for (key, v) in [("englishName", &m.english_name), ("englishComment", &m.english_comment)] {
        if !v.is_empty() {
            source.insert(key.into(), json!(v));
        }
    }
    if m.add_uvs > 0 {
        source.insert("additionalUvs".into(), json!(m.add_uvs));
    }
    if !materials_extra.is_empty() {
        source.insert("materials".into(), json!(materials_extra));
    }
    if !m.toon_textures.is_empty() {
        source.insert("toonTextures".into(), json!(m.toon_textures));
    }
    if !has_mesh && !morphs_extra.is_empty() {
        source.insert("morphs".into(), json!(morphs_extra));
    }
    scene.source = source;

    facts.insert("vertices".into(), json!(nv));
    facts.insert("bones".into(), json!(nb));
    facts.insert("morphs".into(), json!(m.morphs.len()));
    for (key, n) in [
        ("display_frames", m.display_frames),
        ("rigid_bodies", m.rigid_bodies),
        ("physics_joints", m.joints),
        ("soft_bodies", m.soft_bodies),
        ("sdef_vertices", m.sdef),
        ("qdef_vertices", m.qdef),
    ] {
        if n > 0 {
            facts.insert(key.into(), json!(n));
        }
    }
    if let Some(e) = m.unread {
        facts.insert("unread_tail".into(), json!(e));
    }
    scene.facts = facts;
    scene.finish(keep)
}

// ---------------------------------------------------------------------------------------------------------------
// VMD.

/// A bone key: frame, move, turn, and the Bézier curves into it (X, Y, Z and rotation).
struct BoneKey {
    frame: u32,
    position: [f32; 3],
    rotation: [f32; 4],
    curve: [u8; 64],
}

/// The value at `x` (0 to 1) of the easing curve through (0, 0), (x1, y1), (x2, y2) and (1, 1), in 127ths.
fn bezier(c: [u8; 4], x: f64) -> f64 {
    let [x1, y1, x2, y2] = c.map(|v| f64::from(v.min(127)) / 127.0);
    let at = |t: f64, a: f64, b: f64| 3.0 * (1.0 - t) * (1.0 - t) * t * a + 3.0 * (1.0 - t) * t * t * b + t * t * t;
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..40 {
        let mid = f64::midpoint(lo, hi);
        if at(mid, x1, x2) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    at(f64::midpoint(lo, hi), y1, y2)
}

/// Curve `axis` (0 to 3: X, Y, Z, rotation) of a key.
fn curve(k: &BoneKey, axis: usize) -> [u8; 4] {
    [k.curve[axis], k.curve[axis + 4], k.curve[axis + 8], k.curve[axis + 12]]
}

/// Whether a curve is the straight line.
fn straight(c: [u8; 4]) -> bool {
    c[0] == c[1] && c[2] == c[3]
}

/// Spherical interpolation, the short way round.
fn slerp(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
    let mut d: f64 = (0..4).map(|k| a[k] * b[k]).sum();
    let b = if d < 0.0 {
        d = -d;
        b.map(|c| -c)
    } else {
        b
    };
    if d > 0.9995 {
        return std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t);
    }
    let th = d.clamp(-1.0, 1.0).acos();
    let s = th.sin();
    let (wa, wb) = (((1.0 - t) * th).sin() / s, (t * th).sin() / s);
    std::array::from_fn(|k| a[k] * wa + b[k] * wb)
}

/// The steps a span of `gap` frames is sampled at when its curves bend: one a frame, at most [`MAX_SAMPLES`].
fn steps(gap: u32) -> u32 {
    gap.clamp(1, MAX_SAMPLES)
}

/// A bone's translation and rotation keys, in seconds, with bent spans sampled.
fn bone_keys(keys: &[BoneKey]) -> (Keys, Keys) {
    let (mut tt, mut tv, mut rt, mut rv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let quat = |k: &BoneKey| {
        let q = scene::normalize(k.rotation.map(f64::from));
        q.map(f64::from)
    };
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            let p = &keys[i - 1];
            let gap = k.frame - p.frame;
            let n = steps(gap);
            let bends = (0..3).any(|a| !straight(curve(k, a)));
            if bends && n > 1 {
                for s in 1..n {
                    let x = f64::from(s) / f64::from(n);
                    tt.push(f32_of((f64::from(p.frame) + f64::from(gap) * x) / FPS));
                    for a in 0..3 {
                        let e = bezier(curve(k, a), x);
                        let (v0, v1) = (f64::from(p.position[a]), f64::from(k.position[a]));
                        tv.push(f32_of(v0 + (v1 - v0) * e));
                    }
                }
            }
            if !straight(curve(k, 3)) && n > 1 {
                let (q0, q1) = (quat(p), quat(k));
                for s in 1..n {
                    let x = f64::from(s) / f64::from(n);
                    rt.push(f32_of((f64::from(p.frame) + f64::from(gap) * x) / FPS));
                    rv.extend(scene::normalize(slerp(q0, q1, bezier(curve(k, 3), x))));
                }
            }
        }
        let t = f32_of(f64::from(k.frame) / FPS);
        tt.push(t);
        tv.extend(k.position);
        rt.push(t);
        rv.extend(scene::normalize(quat(k)));
    }
    ((tt, tv), (rt, rv))
}

/// Reads a `.vmd` motion.
fn motion(data: &[u8], stem: &str, keep: Keep) -> Result<Document> {
    let mut b = Bin::new(data, "VMD");
    let head = b.bytes(30)?;
    let name_len = if head.starts_with(b"Vocaloid Motion Data 0002") {
        20
    } else if head.starts_with(b"Vocaloid Motion Data file") {
        10
    } else {
        bail!("isn't a VMD file")
    };
    let model = sjis(b.fixed_str(name_len)?);

    let c = b.u32()?;
    let n = b.count(c.into(), 111)?;
    let mut order: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, Vec<BoneKey>> = HashMap::new();
    for _ in 0..n {
        let name = sjis(b.fixed_str(15)?);
        let frame = b.u32()?;
        let position = b.vec3()?;
        let rotation = b.vec4()?;
        let curve = b.array::<64>()?;
        let keys = by_name.entry(name.clone()).or_insert_with(|| {
            order.push(name);
            Vec::new()
        });
        keys.push(BoneKey {
            frame,
            position,
            rotation,
            curve,
        });
    }
    let mut facts = Map::new();
    facts.insert("bone_keys".into(), json!(n));

    // The sections after the bone keys, as far as the file goes (older files stop early).
    let mut morph_keys: Vec<(String, u32, f32)> = Vec::new();
    let mut unread = None;
    let tail = (|| -> Result<()> {
        if b.left() < 4 {
            return Ok(());
        }
        let c = b.u32()?;
        let n = b.count(c.into(), 23)?;
        for _ in 0..n {
            let name = sjis(b.fixed_str(15)?);
            morph_keys.push((name, b.u32()?, b.f32()?));
        }
        for (key, size) in [("camera_keys", 61), ("light_keys", 28), ("shadow_keys", 9)] {
            if b.left() < 4 {
                return Ok(());
            }
            let c = b.u32()?;
            let n = b.count(c.into(), size)?;
            b.skip(n * size)?;
            if n > 0 {
                facts.insert(key.into(), json!(n));
            }
        }
        if b.left() < 4 {
            return Ok(());
        }
        let c = b.u32()?;
        let n = b.count(c.into(), 9)?;
        for _ in 0..n {
            b.skip(5)?;
            let c = b.u32()?;
            let k = b.count(c.into(), 21)?;
            b.skip(21 * k)?;
        }
        if n > 0 {
            facts.insert("ik_keys".into(), json!(n));
        }
        Ok(())
    })();
    if let Err(e) = tail {
        unread = Some(e.to_string());
    }

    let mut scene = Scene::default();
    let root_name = if model.is_empty() {
        stem.to_owned()
    } else {
        model.clone()
    };
    let root = scene.add(Node::named(root_name, None));
    let mut channels = Vec::new();
    for name in order {
        let Some(mut keys) = by_name.remove(&name) else {
            continue;
        };
        // In frame order; of keys on the same frame, the last.
        keys.sort_by_key(|k| k.frame);
        let mut kept: Vec<BoneKey> = Vec::with_capacity(keys.len());
        for k in keys {
            if kept.last().is_some_and(|l| l.frame == k.frame) {
                kept.pop();
            }
            kept.push(k);
        }
        let node = scene.add(Node::named(name, Some(root)));
        let moves = kept.iter().any(|k| k.position.iter().any(|&c| c != 0.0));
        let ((tt, tv), (rt, rv)) = bone_keys(&kept);
        if moves {
            channels.push(Channel {
                node,
                path: Key::Translation,
                times: tt,
                values: tv,
                interp: Interp::Linear,
            });
        }
        channels.push(Channel {
            node,
            path: Key::Rotation,
            times: rt,
            values: rv,
            interp: Interp::Linear,
        });
    }
    facts.insert("bones".into(), json!(scene.nodes.len() - 1));
    if !channels.is_empty() {
        scene.animations.push(Animation {
            name: if stem.is_empty() {
                "motion".into()
            } else {
                stem.to_owned()
            },
            channels,
            extras: None,
        });
    }

    let mut source = Map::new();
    convention(&mut scene.info);
    scene.info.format = Some("VMD".into());
    // Keys are frame numbers, which the format counts at 30 a second.
    scene.info.frame_rate = Fact::Declared(FPS);
    source.insert("model".into(), json!(model));
    if !morph_keys.is_empty() {
        facts.insert("morph_keys".into(), json!(morph_keys.len()));
        let mut names: Vec<String> = Vec::new();
        let mut per: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        for (name, frame, w) in morph_keys {
            per.entry(name.clone())
                .or_insert_with(|| {
                    names.push(name);
                    Vec::new()
                })
                .push((frame, w));
        }
        let mut out = Map::new();
        for name in names {
            let mut keys = per.remove(&name).unwrap_or_default();
            keys.sort_by_key(|k| k.0);
            let times: Vec<f32> = keys.iter().map(|k| f32_of(f64::from(k.0) / FPS)).collect();
            let weights: Vec<f32> = keys.iter().map(|k| if k.1.is_finite() { k.1 } else { 0.0 }).collect();
            out.insert(name, json!({ "times": times, "weights": weights }));
        }
        source.insert("morphKeys".into(), Value::Object(out));
    }
    if let Some(e) = unread {
        facts.insert("unread_tail".into(), json!(e));
    }
    scene.source = source;
    scene.facts = facts;
    scene.mirror_z();
    scene.scale(UNIT);
    scene.finish(keep)
}

/// The source info the formats share: left-handed and Y up, as the format defines, in units taken as 8 cm.
fn convention(info: &mut SourceInfo) {
    // 0.08 as written, not the f32 widened (0.0799999982).
    info.meters_per_unit = Fact::Assumed(0.08);
    info.up_axis = Fact::Assumed(Axis::PosY);
    info.handedness = Fact::Declared(Handedness::Left);
}

#[cfg(test)]
mod tests;
