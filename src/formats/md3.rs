//! MD3 (the Quake III Arena engine's model format) as a glTF document: a mesh per surface whose animation frames
//! are morph targets, and the tags weapons and heads attach to as nodes.
//!
//! The file is little-endian binary: a 108-byte header (`IDP3`, version 15, the model's path in the game, then the
//! counts and offsets of its frames, tags and surfaces), the frames (bounds, origin, radius and a name, 56 bytes
//! each; only the names are kept), the tags (a name, an origin and three axes, 112 bytes, for every tag in every
//! frame), and the surfaces one after another. Each surface is its own block, with offsets from its start: its
//! shaders (texture paths), triangles, texture coordinates, and, for every frame, every vertex's position (three
//! 16-bit integers in 64ths of a unit) and normal (latitude and longitude bytes, 256ths of a turn).
//!
//! Each surface is a node with a mesh of one primitive, named after the surface. Its first frame is the mesh;
//! every later frame is a morph target (position and normal offsets from the first) named after its frame (or
//! `frame1`, `frame2`, ... when the names don't tell the frames apart). Texture coordinates have their origin at
//! the top left, as glTF's do, and are kept as written. Triangles are wound clockwise seen from the front, so each
//! is turned around for glTF. A surface with no triangles is left out.
//!
//! Each tag is a root node named after it, at its place in the first frame (its axes are the columns of its
//! rotation). With more than one frame, the frames become one linear animation, named after the file: a weights
//! channel per surface with a key per frame that turns that frame's target fully on (the first frame's key turns
//! them all off), and translation and rotation channels per tag. The format has no frame rate; frames are taken at
//! 10 a second (a model's frame rates are in the game's animation scripts).
//!
//! A surface's first shader name is its material's texture, looked for as written and, when it has no extension,
//! as a `.tga` (the assets find the same name in other image formats, and by file name in other folders); its
//! material is named after it. A surface with no shader name takes its texture from `<model>_default.skin` beside
//! the model when there is one (`surface,texture` lines, as player models name theirs). Shader scripts, which can
//! name a different texture for a shader, aren't read; frame bounds are dropped.
//!
//! The format is Z up by the engine's convention, so the scene's roots are turned for glTF's Y up (the source info's
//! `upAxis` is an assumed `+Z`, converted); game units are kept unscaled, and their length unknown. With more than
//! one frame the assumed frame rate is recorded. The header's name (the model's path in the game's folders) is kept
//! as `name`.
//!
//! A file cut short, an offset past its end, a negative count, a surface that doesn't start with `IDP3` or has no
//! frames, or a triangle naming a vertex past its surface's is an error. Counts are checked against the file's size
//! before anything is allocated for them, and a file whose frames would make more morph data than a model of the
//! engine can hold is refused.

use std::collections::{HashMap, HashSet};
use std::f64::consts::TAU;

use serde_json::json;

use super::Input;
use super::scene::{self, Animation, Channel, Interp, Node, Path, Primitive, Scene, SceneMesh, Target, Z_UP_TO_Y_UP};
use crate::Result;
use crate::assets::Assets;
use crate::bytes::Bin;
use crate::document::{Document, Keep};
use crate::images::Images;
use crate::mesh::Material;
use crate::num::{f32_of, u32_of};
use crate::quat;
use crate::source::{Axis, Fact};

/// Frames a second, as the format stores none.
const RATE: f32 = 10.0;

/// The most frames read; the engine holds 1024.
const MAX_FRAMES: usize = 1024;

/// The most morph data read, in frame-vertices (frames times vertices) plus target weights (frames squared) over
/// all surfaces; the engine's largest model has about four million.
const MAX_MORPH_DATA: usize = 1 << 24;

/// The size of a surface's header.
const SURFACE_HEADER: usize = 108;

/// Reads an `.md3` model.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document> {
    let data = input.read()?;
    let file = parse(&data, keep.morph_targets)?;
    let stem = input
        .path
        .file_stem()
        .map_or_else(|| "md3".to_owned(), |s| s.to_string_lossy().into_owned());

    let mut scene = Scene::default();
    let mut images = Images::new(!keep.image_bytes, input.dir());
    let skin = if file.surfaces.iter().any(|s| s.shader.is_empty()) {
        skin_file(&input, &stem, assets)
    } else {
        HashMap::new()
    };
    let mut materials: HashMap<String, usize> = HashMap::new();
    let mut weights = Vec::new();
    let n_frames = file.frame_names.len();
    let target_names = target_names(&file.frame_names);
    for s in file.surfaces {
        let texture = if s.shader.is_empty() {
            skin.get(&s.name.to_ascii_lowercase()).cloned().unwrap_or_default()
        } else {
            s.shader.clone()
        };
        let key = if texture.is_empty() {
            format!("\0{}", s.name)
        } else {
            texture.clone()
        };
        let material = if let Some(&i) = materials.get(&key) {
            i
        } else {
            let name = if texture.is_empty() {
                s.name.clone()
            } else {
                texture.clone()
            };
            let mut m = Material::plain(&name, [1.0, 1.0, 1.0, 1.0]);
            m.color_image = find_texture(&texture, &input, assets, &mut images);
            scene.materials.push(m);
            materials.insert(key, scene.materials.len() - 1);
            scene.materials.len() - 1
        };
        let mut prim = s.prim;
        prim.material = Some(material);
        let targets = prim.targets.len();
        let mut node = Node::named(s.name.clone(), None);
        node.mesh = Some(scene.meshes.len());
        let at = scene.add(node);
        // Each target's frame is past the first, and no surface keeps more frames than the header has.
        let names = s
            .frames
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_some())
            .map(|(f, _)| target_names.get(f - 1).cloned().unwrap_or_default())
            .collect();
        scene.meshes.push(SceneMesh {
            name: s.name,
            prims: vec![prim],
            target_names: names,
            ..SceneMesh::default()
        });
        if targets > 0 {
            let mut values = vec![0.0; s.frames.len() * targets];
            for (k, t) in s.frames.iter().enumerate() {
                if let Some(t) = t {
                    values[k * targets + t] = 1.0;
                }
            }
            weights.push(Channel {
                node: at,
                path: Path::Weights,
                times: times(s.frames.len()),
                values,
                interp: Interp::Linear,
            });
        }
    }
    scene.images = images.list;

    let mut channels = weights;
    let n_tags = file.tags.len();
    for tag in file.tags {
        let Some(&(t0, r0)) = tag.frames.first() else {
            continue;
        };
        let mut node = Node::named(tag.name, None);
        node.translation = t0;
        node.rotation = r0;
        let at = scene.add(node);
        if tag.frames.len() > 1 {
            channels.push(Channel {
                node: at,
                path: Path::Translation,
                times: times(tag.frames.len()),
                values: tag.frames.iter().flat_map(|f| f.0).collect(),
                interp: Interp::Linear,
            });
            channels.push(Channel {
                node: at,
                path: Path::Rotation,
                times: times(tag.frames.len()),
                values: tag.frames.iter().flat_map(|f| f.1).collect(),
                interp: Interp::Linear,
            });
        }
    }
    if !channels.is_empty() {
        scene.animations.push(Animation {
            name: stem,
            channels,
            extras: None,
        });
    }

    scene.facts.insert("frames".into(), json!(n_frames));
    scene.facts.insert("surfaces".into(), json!(file.surface_count));
    scene.facts.insert("tags".into(), json!(n_tags));
    scene.info.format = Some("MD3".into());
    scene.info.version = Some(file.version.to_string());
    if !file.name.is_empty() {
        // The model's path in the game's folders, not a title.
        scene.source.insert("name".into(), json!(file.name));
    }
    scene.info.up_axis = Fact::Assumed(Axis::PosZ);
    if n_frames > 1 {
        scene.info.frame_rate = Fact::Assumed(f64::from(RATE));
    }
    scene.rotate_roots(Z_UP_TO_Y_UP);
    scene.finish(keep)
}

/// The times of `n` frames, in seconds.
fn times(n: usize) -> Vec<f32> {
    (0..n).map(|k| f32_of(f64::from(u32_of(k))) / RATE).collect()
}

/// The morph targets' names, one per frame after the first: the frames' names when they tell the frames apart,
/// `frame1`, `frame2`, ... otherwise.
fn target_names(frames: &[String]) -> Vec<String> {
    let rest = frames.get(1..).unwrap_or_default();
    let mut seen = HashSet::new();
    if rest.iter().all(|n| !n.is_empty() && seen.insert(n.as_str())) {
        rest.to_vec()
    } else {
        (1..frames.len()).map(|k| format!("frame{k}")).collect()
    }
}

/// The texture a shader or skin names, by the engine's conventions: as written, then as a `.tga` when it has no
/// extension.
fn find_texture(name: &str, input: &Input<'_>, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    if name.is_empty() {
        return None;
    }
    let mut tries = vec![name.to_owned()];
    if std::path::Path::new(name).extension().is_none() {
        tries.push(format!("{name}.tga"));
    }
    tries
        .iter()
        .find_map(|t| assets.find(t, input.dir()).map(|p| (p, t)))
        .and_then(|(p, t)| images.file(assets, p, t))
}

/// The textures `<stem>_default.skin` beside the model gives its surfaces, by lowercase surface name; empty when
/// there's no such file.
fn skin_file(input: &Input<'_>, stem: &str, assets: &dyn Assets) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Some(path) = assets.find(&format!("{stem}_default.skin"), input.dir()) else {
        return out;
    };
    let Some(bytes) = assets.read(&path) else {
        return out;
    };
    for line in scene::text(&bytes).lines() {
        if let Some((surface, texture)) = line.split_once(',') {
            let (surface, texture) = (surface.trim(), texture.trim());
            if !surface.is_empty() && !texture.is_empty() {
                out.insert(surface.to_ascii_lowercase(), texture.to_owned());
            }
        }
    }
    out
}

/// What a file holds, ready for the scene.
struct File {
    version: i32,
    name: String,
    frame_names: Vec<String>,
    tags: Vec<Tag>,
    /// The surfaces with triangles.
    surfaces: Vec<Surface>,
    /// All the surfaces, with triangles or not.
    surface_count: usize,
}

/// A tag: its name and its place (translation and rotation) in each frame.
struct Tag {
    name: String,
    frames: Vec<([f32; 3], [f32; 4])>,
}

/// A surface: its name, first shader name, and mesh (the first frame, with the later frames as targets when
/// they're kept).
struct Surface {
    name: String,
    shader: String,
    prim: Primitive,
    /// Each kept frame's target; `None` for the first frame and any frame that doesn't move from it.
    frames: Vec<Option<usize>>,
}

/// A header count or offset, which can't be negative.
fn size(v: i32, what: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| err!(Corrupt, "MD3 {what} is negative ({v})"))
}

/// Reads the file; `morphs` says whether frames after the first are kept as targets.
fn parse(data: &[u8], morphs: bool) -> Result<File> {
    ensure!(
        data.starts_with(b"IDP3"),
        UnknownFormat,
        "not an MD3 file: no IDP3 signature"
    );
    let mut b = Bin::new(data, "MD3");
    b.skip(4)?;
    let version = b.i32()?;
    let name = scene::text(b.fixed_str(64)?);
    let _flags = b.i32()?;
    let n_frames = size(b.i32()?, "frame count")?;
    let n_tags = size(b.i32()?, "tag count")?;
    let n_surfaces = size(b.i32()?, "surface count")?;
    let _n_skins = b.i32()?;
    let ofs_frames = size(b.i32()?, "frame offset")?;
    let ofs_tags = size(b.i32()?, "tag offset")?;
    let ofs_surfaces = size(b.i32()?, "surface offset")?;
    ensure!(n_frames > 0, Empty, "MD3 file has no frames");
    ensure!(
        n_frames <= MAX_FRAMES,
        LimitExceeded,
        "MD3 file has {n_frames} frames, more than {MAX_FRAMES}"
    );

    b.seek(ofs_frames)?;
    let mut frame_names = Vec::with_capacity(b.count(n_frames as u64, 56)?);
    for _ in 0..n_frames {
        b.skip(40)?;
        frame_names.push(scene::text(b.fixed_str(16)?));
    }

    let mut tags: Vec<Tag> = Vec::new();
    if n_tags > 0 {
        b.seek(ofs_tags)?;
        let total = n_frames
            .checked_mul(n_tags)
            .ok_or_else(|| err!(LimitExceeded, "MD3 file claims too many tags"))?;
        b.count(total as u64, 112)?;
        for f in 0..n_frames {
            for t in 0..n_tags {
                let name = scene::text(b.fixed_str(64)?);
                let origin = b.vec3()?;
                let axes = [b.vec3()?, b.vec3()?, b.vec3()?];
                // The axes are the rotation's columns, which is how the matrix is laid out.
                let m: [f64; 9] = std::array::from_fn(|i| f64::from(axes[i / 3][i % 3]));
                let q = quat::normalize(quat::from_matrix(&m));
                if f == 0 {
                    tags.push(Tag {
                        name,
                        frames: Vec::with_capacity(n_frames),
                    });
                }
                tags[t].frames.push((origin, q));
            }
        }
    }

    let mut surfaces = Vec::new();
    let mut budget = MAX_MORPH_DATA;
    let mut at = ofs_surfaces;
    if n_surfaces > 0 {
        b.seek(at)?;
        b.count(n_surfaces as u64, SURFACE_HEADER)?;
    }
    for s in 0..n_surfaces {
        let kept = if morphs { n_frames } else { 1 };
        let (surface, end) = surface(data, at, s, kept, &mut budget)?;
        surfaces.extend(surface);
        at = at
            .checked_add(end)
            .ok_or_else(|| err!(Truncated, "MD3 surface {s} runs past the end of the file"))?;
    }
    Ok(File {
        version,
        name,
        frame_names,
        tags,
        surfaces,
        surface_count: n_surfaces,
    })
}

/// The surface `s` at `start`, unless it has no triangles, and its size; it keeps at most `keep` frames, and
/// `budget` is the morph data still allowed.
fn surface(data: &[u8], start: usize, s: usize, keep: usize, budget: &mut usize) -> Result<(Option<Surface>, usize)> {
    let mut b = Bin::new(data, "MD3");
    b.seek(start)?;
    ensure!(
        b.bytes(4)? == b"IDP3",
        Corrupt,
        "MD3 surface {s} doesn't start with IDP3"
    );
    let name = scene::text(b.fixed_str(64)?);
    let _flags = b.i32()?;
    let n_frames = size(b.i32()?, "surface frame count")?;
    let n_shaders = size(b.i32()?, "shader count")?;
    let n_verts = size(b.i32()?, "vertex count")?;
    let n_tris = size(b.i32()?, "triangle count")?;
    let ofs = |b: &mut Bin<'_>, what: &str| -> Result<usize> {
        start
            .checked_add(size(b.i32()?, what)?)
            .ok_or_else(|| err!(Truncated, "MD3 surface {s}'s {what} runs past the end of the file"))
    };
    let ofs_tris = ofs(&mut b, "triangle offset")?;
    let ofs_shaders = ofs(&mut b, "shader offset")?;
    let ofs_st = ofs(&mut b, "texture coordinate offset")?;
    let ofs_xyz = ofs(&mut b, "vertex offset")?;
    let end = size(b.i32()?, "surface size")?;
    // Each surface follows the last, so the next can't start inside this one's header.
    ensure!(
        end >= SURFACE_HEADER,
        Corrupt,
        "MD3 surface {s} is {end} bytes, less than its header"
    );
    ensure!(n_frames > 0, Corrupt, "MD3 surface {s} has no frames");
    ensure!(
        n_frames <= MAX_FRAMES,
        LimitExceeded,
        "MD3 surface {s} has {n_frames} frames, more than {MAX_FRAMES}"
    );

    let mut shader = String::new();
    if n_shaders > 0 {
        b.seek(ofs_shaders)?;
        b.count(n_shaders as u64, 68)?;
        shader = scene::text(b.fixed_str(64)?);
    }
    if n_tris == 0 || n_verts == 0 {
        return Ok((None, end));
    }
    b.seek(ofs_tris)?;
    let mut indices = Vec::with_capacity(b.count(n_tris as u64, 12)?.saturating_mul(3));
    for t in 0..n_tris {
        let v = [b.i32()?, b.i32()?, b.i32()?];
        let v = v.map(|i| usize::try_from(i).ok().filter(|&i| i < n_verts));
        let [Some(a), Some(c1), Some(c2)] = v else {
            bail!(
                Corrupt,
                "MD3 surface {s}'s triangle {t} names a vertex past its {n_verts}"
            );
        };
        // Clockwise front faces in the file; glTF's are counterclockwise.
        indices.extend([a, c2, c1].map(u32_of));
    }
    b.seek(ofs_st)?;
    let mut uvs = Vec::with_capacity(b.count(n_verts as u64, 8)?);
    for _ in 0..n_verts {
        uvs.push(b.vec2()?);
    }

    let kept = n_frames.min(keep);
    let cost = kept
        .checked_mul(n_verts)
        .and_then(|n| n.checked_add(kept * kept))
        .filter(|&n| n <= *budget)
        .ok_or_else(|| err!(LimitExceeded, "MD3 file's frames hold more vertices than a model can"))?;
    *budget -= cost;
    b.seek(ofs_xyz)?;
    b.count((n_frames as u64).saturating_mul(n_verts as u64), 8)?;
    let mut prim = Primitive {
        indices,
        uvs: vec![uvs],
        ..Primitive::default()
    };
    let mut frames = Vec::with_capacity(kept);
    for f in 0..kept {
        let mut positions = Vec::with_capacity(n_verts);
        let mut normals = Vec::with_capacity(n_verts);
        for _ in 0..n_verts {
            let p = [b.i16()?, b.i16()?, b.i16()?];
            positions.push(p.map(|c| f32::from(c) / 64.0));
            normals.push(normal(b.u16()?));
        }
        if f == 0 {
            prim.positions = positions;
            prim.normals = normals;
            frames.push(None);
        } else {
            frames.push(target(&mut prim, &positions, &normals));
        }
    }
    Ok((
        Some(Surface {
            name,
            shader,
            prim,
            frames,
        }),
        end,
    ))
}

/// A normal from its packed latitude (high byte) and longitude (low byte), each in 256ths of a turn.
fn normal(n: u16) -> [f32; 3] {
    let [lng, lat] = n.to_le_bytes().map(|c| f64::from(c) * TAU / 256.0);
    [lat.cos() * lng.sin(), lat.sin() * lng.sin(), lng.cos()].map(f32_of)
}

/// Adds a frame as a target of `prim` (holding the first frame) and gives its index, or `None` when the frame
/// doesn't move a vertex: then it's the first frame again, with no target of its own.
fn target(prim: &mut Primitive, positions: &[[f32; 3]], normals: &[[f32; 3]]) -> Option<usize> {
    let zero = |d: &[[f32; 3]]| d.iter().flatten().all(|&c| c == 0.0);
    let positions = minus(positions, &prim.positions);
    if zero(&positions) {
        return None;
    }
    let normals = minus(normals, &prim.normals);
    prim.targets.push(Target {
        positions,
        // A target needn't carry what it leaves alone.
        normals: if zero(&normals) { Vec::new() } else { normals },
    });
    Some(prim.targets.len() - 1)
}

/// `a - b`, vector by vector.
fn minus(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<[f32; 3]> {
    a.iter()
        .zip(b)
        .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_unpack_from_latitude_and_longitude() {
        // Longitude 0 is straight up; a quarter turn of each points along +Y.
        let up = normal(0);
        assert!((up[2] - 1.0).abs() < 1e-6);
        let y = normal(0x4040);
        assert!((y[1] - 1.0).abs() < 1e-6, "{y:?}");
    }

    #[test]
    fn target_names_fall_back_to_numbers() {
        let s = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(target_names(&s(&["a", "b", "c"])), s(&["b", "c"]));
        assert_eq!(target_names(&s(&["a", "b", "b"])), s(&["frame1", "frame2"]));
        assert_eq!(target_names(&s(&["a", ""])), s(&["frame1"]));
        assert_eq!(target_names(&s(&["a"])), Vec::<String>::new());
    }

    #[test]
    fn hostile_headers_are_errors() {
        assert!(parse(b"IDP", true).is_err());
        assert!(parse(b"IDP2xxxx", true).is_err());
        let mut h = b"IDP3".to_vec();
        h.extend_from_slice(&15i32.to_le_bytes());
        h.extend_from_slice(&[0; 68]);
        // One frame, no tags, one surface whose header runs past the end.
        for v in [1i32, 0, 1, 0, 108, 164, 164, 0] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(&[0; 56]);
        assert!(parse(&h, true).is_err());
        // A negative count.
        h[76..80].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(parse(&h, true).is_err());
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
        let bytes = include_bytes!("../../tests/corpus/md3/surfaces_tag_and_skin/thing.md3");
        assert!(
            load(
                Input::memory(bytes, "thing.md3"),
                &crate::assets::Memory::new(),
                Keep::ALL
            )
            .is_ok()
        );
        sweep(bytes, |b| {
            let _ = load(Input::memory(b, "thing.md3"), &crate::assets::Memory::new(), Keep::ALL);
        });
    }

    #[test]
    fn source_info_is_game_units_z_up_at_ten_frames_a_second() {
        let bytes = include_bytes!("../../tests/corpus/md3/surfaces_tag_and_skin/thing.md3");
        let doc = load(
            Input::memory(bytes, "thing.md3"),
            &crate::assets::Memory::new(),
            Keep::ALL,
        )
        .unwrap();
        let info = doc.source_info();
        assert_eq!(info.format.as_deref(), Some("MD3"));
        assert_eq!(info.version.as_deref(), Some("15"));
        assert_eq!(info.meters_per_unit, Fact::Unknown);
        assert_eq!(info.up_axis, Fact::Assumed(Axis::PosZ));
        assert_eq!(info.frame_rate, Fact::Assumed(10.0));
        assert!(info.converted.up_axis && !info.converted.units);
        assert_eq!(
            doc.json()["asset"]["extras"]["source"]["name"],
            "models/thing/thing.md3"
        );
    }
}
