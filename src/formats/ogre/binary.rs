//! The binary `.mesh` and `.skeleton` layouts.
//!
//! Both start with a 16-bit header id (0x1000, whose byte order says the file's) and a version line, then hold
//! chunks: a 16-bit id and a 32-bit length that counts the six header bytes. Old writers got many lengths wrong,
//! so chunks are read by what they hold, as OGRE reads them, and lengths only skip what isn't read (bounding
//! boxes' extreme points, a skeleton's unknown chunks). A chunk the reader doesn't expect where it is ends the
//! list it was in.

use super::{
    Assign, Bone, BoneTrack, Geometry, Key, Keys, LodLevel, MeshAnimation, MeshFile, MorphKey, Pose, PoseKey,
    PoseOffset, SkeletonAnimation, SkeletonFile, SubMesh, Track,
};
use crate::Result;
use crate::bytes::{self, Bin, Endian, Order as _};
use crate::num::usize_of;

/// Mesh chunk ids.
const M_MESH: u16 = 0x3000;
const M_SUBMESH: u16 = 0x4000;
const M_SUBMESH_OPERATION: u16 = 0x4010;
const M_SUBMESH_BONE_ASSIGNMENT: u16 = 0x4100;
const M_SUBMESH_TEXTURE_ALIAS: u16 = 0x4200;
const M_GEOMETRY: u16 = 0x5000;
const M_GEOMETRY_VERTEX_DECLARATION: u16 = 0x5100;
const M_GEOMETRY_VERTEX_ELEMENT: u16 = 0x5110;
const M_GEOMETRY_VERTEX_BUFFER: u16 = 0x5200;
const M_GEOMETRY_VERTEX_BUFFER_DATA: u16 = 0x5210;
/// The geometry chunks before version 1.30, in place of the declaration and buffers.
const M_GEOMETRY_NORMALS: u16 = 0x5100;
const M_GEOMETRY_COLOURS: u16 = 0x5200;
const M_GEOMETRY_TEXCOORDS: u16 = 0x5300;
const M_MESH_SKELETON_LINK: u16 = 0x6000;
const M_MESH_BONE_ASSIGNMENT: u16 = 0x7000;
const M_MESH_LOD_LEVEL: u16 = 0x8000;
const M_MESH_LOD_USAGE: u16 = 0x8100;
const M_MESH_LOD_MANUAL: u16 = 0x8110;
const M_MESH_LOD_GENERATED: u16 = 0x8120;
const M_MESH_BOUNDS: u16 = 0x9000;
const M_SUBMESH_NAME_TABLE: u16 = 0xA000;
const M_SUBMESH_NAME_TABLE_ELEMENT: u16 = 0xA100;
const M_EDGE_LISTS: u16 = 0xB000;
const M_EDGE_LIST_LOD: u16 = 0xB100;
const M_EDGE_GROUP: u16 = 0xB110;
const M_POSES: u16 = 0xC000;
const M_POSE: u16 = 0xC100;
const M_POSE_VERTEX: u16 = 0xC111;
const M_ANIMATIONS: u16 = 0xD000;
const M_ANIMATION: u16 = 0xD100;
const M_ANIMATION_BASEINFO: u16 = 0xD105;
const M_ANIMATION_TRACK: u16 = 0xD110;
const M_ANIMATION_MORPH_KEYFRAME: u16 = 0xD111;
const M_ANIMATION_POSE_KEYFRAME: u16 = 0xD112;
const M_ANIMATION_POSE_REF: u16 = 0xD113;
const M_TABLE_EXTREMES: u16 = 0xE000;

/// Skeleton chunk ids.
const SKELETON_BLENDMODE: u16 = 0x1010;
const SKELETON_BONE: u16 = 0x2000;
const SKELETON_BONE_PARENT: u16 = 0x3000;
const SKELETON_ANIMATION: u16 = 0x4000;
const SKELETON_ANIMATION_BASEINFO: u16 = 0x4010;
const SKELETON_ANIMATION_TRACK: u16 = 0x4100;
const SKELETON_ANIMATION_TRACK_KEYFRAME: u16 = 0x4110;
const SKELETON_ANIMATION_LINK: u16 = 0x5000;

/// A chunk header's size: id and length.
const HEADER: usize = 6;
/// A bone chunk's length without its scale: the length leaves the name out.
const BONE_WITHOUT_SCALE: usize = HEADER + 2 + 12 + 16;
/// A keyframe chunk's length without its scale.
const KEY_WITHOUT_SCALE: usize = HEADER + 4 + 16 + 12;

/// Vertex element semantics.
const VES_POSITION: u16 = 1;
const VES_BLEND_WEIGHTS: u16 = 2;
const VES_BLEND_INDICES: u16 = 3;
const VES_NORMAL: u16 = 4;
const VES_DIFFUSE: u16 = 5;
const VES_SPECULAR: u16 = 6;
const VES_TEXTURE_COORDINATES: u16 = 7;
const VES_BINORMAL: u16 = 8;
const VES_TANGENT: u16 = 9;

use crate::formats::scene::MAX_UV_SETS;

/// The mesh serializer versions, oldest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Version {
    V110,
    V120,
    V130,
    V140,
    V141,
    V18,
    V100,
}

/// Reads a binary file's header: its byte order (true for big-endian) and version line, or `None` when it isn't
/// one.
pub(super) fn header(data: &[u8]) -> Option<(bool, String)> {
    let be = match data.get(..2)? {
        [0x00, 0x10] => false,
        [0x10, 0x00] => true,
        _ => return None,
    };
    let line = reader(data, be).string();
    line.starts_with('[').then_some((be, line))
}

/// A binary file being read, in its byte order.
type Reader<'a> = Bin<'a, Endian>;

/// A reader just past a file's 16-bit header id, which `header` has seen.
fn reader(data: &[u8], be: bool) -> Reader<'_> {
    let mut r = Bin::with_order(data, "OGRE", Endian { big: be });
    r.at = 2;
    r
}

/// The strings, chunks and counts both layouts share.
trait Chunks {
    /// A string: the bytes up to a line feed (or the end), without a carriage return before it.
    fn string(&mut self) -> String;
    /// The next chunk's id and length, without reading them.
    fn peek(&self) -> Option<(u16, usize)>;
    /// Reads the next chunk's header when its id is one of `ids`: its id, where it starts and its length.
    fn chunk(&mut self, ids: &[u16]) -> Option<(u16, usize, usize)>;
    /// The next chunk's header, which must have id `id`.
    fn expect(&mut self, id: u16, what: &str) -> Result<(usize, usize)>;
    /// Moves past a chunk that starts at `start` and is `len` long.
    fn skip_chunk(&mut self, start: usize, len: usize) -> Result<()>;
    /// `n` items of `size` bytes each, checked against what's left.
    fn items(&self, n: u32, size: usize) -> Result<usize>;
    /// `n` indices, 16 or 32 bits each.
    fn index_list(&mut self, n: u32, idx32: bool) -> Result<Vec<u32>>;
}

impl Chunks for Reader<'_> {
    fn string(&mut self) -> String {
        let line = self.until(b'\n');
        crate::formats::scene::text(line.strip_suffix(b"\r").unwrap_or(line))
    }

    fn peek(&self) -> Option<(u16, usize)> {
        let id = self.u16_at(self.at)?;
        let len = self.u32_at(self.at.checked_add(2)?)?;
        Some((id, usize_of(u64::from(len))))
    }

    fn chunk(&mut self, ids: &[u16]) -> Option<(u16, usize, usize)> {
        let (id, len) = self.peek()?;
        if !ids.contains(&id) {
            return None;
        }
        let start = self.at;
        // `peek` saw all six header bytes.
        self.at += HEADER;
        Some((id, start, len))
    }

    fn expect(&mut self, id: u16, what: &str) -> Result<(usize, usize)> {
        match self.chunk(&[id]) {
            Some((_, start, len)) => Ok((start, len)),
            None => bail!(Corrupt, "OGRE {what} chunk missing at byte {}", self.at),
        }
    }

    fn skip_chunk(&mut self, start: usize, len: usize) -> Result<()> {
        let end = start.saturating_add(len);
        ensure!(
            len >= HEADER && end <= self.data.len() && end >= self.at,
            Corrupt,
            "OGRE chunk at byte {start} has a bad length"
        );
        self.at = end;
        Ok(())
    }

    fn items(&self, n: u32, size: usize) -> Result<usize> {
        let n = usize_of(u64::from(n));
        ensure!(
            n.checked_mul(size).is_some_and(|b| b <= self.left()),
            Truncated,
            "OGRE count {n} at byte {} runs past the end of the file",
            self.at
        );
        Ok(n)
    }

    fn index_list(&mut self, n: u32, idx32: bool) -> Result<Vec<u32>> {
        let size = if idx32 { 4 } else { 2 };
        let n = self.items(n, size)?;
        self.indices(n, size)
    }
}

/// The version a mesh's header line names.
fn mesh_version(line: &str) -> Result<Version> {
    Ok(match line {
        "[MeshSerializer_v1.10]" => Version::V110,
        "[MeshSerializer_v1.20]" => Version::V120,
        "[MeshSerializer_v1.30]" => Version::V130,
        "[MeshSerializer_v1.40]" => Version::V140,
        "[MeshSerializer_v1.41]" => Version::V141,
        "[MeshSerializer_v1.8]" => Version::V18,
        "[MeshSerializer_v1.100]" | "[MeshSerializer_v2.1 R0 LEGACYV1]" => Version::V100,
        l if l.starts_with("[MeshSerializer_v2") => {
            bail!(
                Unsupported,
                "OGRE mesh {l} is the Ogre-Next v2 layout, which isn't read"
            )
        }
        l => bail!(Unsupported, "unknown OGRE mesh version {l}"),
    })
}

/// Reads a binary `.mesh`.
pub(super) fn mesh(data: &[u8]) -> Result<MeshFile> {
    let (be, line) = header(data).ok_or_else(|| err!(UnknownFormat, "not an OGRE mesh"))?;
    let version = mesh_version(&line)?;
    let mut r = reader(data, be);
    r.string();
    let mut file = MeshFile {
        version: line.trim_matches(['[', ']']).to_owned(),
        big_endian: be,
        ..MeshFile::default()
    };
    r.expect(M_MESH, "mesh")?;
    file.skeletally_animated = r.bool()?;
    let ids = [
        M_GEOMETRY,
        M_SUBMESH,
        M_MESH_SKELETON_LINK,
        M_MESH_BONE_ASSIGNMENT,
        M_MESH_LOD_LEVEL,
        M_MESH_BOUNDS,
        M_SUBMESH_NAME_TABLE,
        M_EDGE_LISTS,
        M_POSES,
        M_ANIMATIONS,
        M_TABLE_EXTREMES,
    ];
    while let Some((id, start, len)) = r.chunk(&ids) {
        match id {
            M_GEOMETRY => {
                let g = geometry(&mut r, version, &mut file)?;
                file.shared = Some(g);
            }
            M_SUBMESH => {
                let s = submesh(&mut r, version, &mut file)?;
                file.submeshes.push(s);
            }
            M_MESH_SKELETON_LINK => file.skeleton = Some(r.string()),
            M_MESH_BONE_ASSIGNMENT => {
                let a = assignment(&mut r)?;
                file.shared_bones.push(a);
            }
            M_MESH_LOD_LEVEL => lod(&mut r, version, &mut file)?,
            M_MESH_BOUNDS => file.bounds = Some((r.vec3()?, r.vec3()?, r.f32()?)),
            M_SUBMESH_NAME_TABLE => {
                while r.chunk(&[M_SUBMESH_NAME_TABLE_ELEMENT]).is_some() {
                    let index = usize::from(r.u16()?);
                    let name = r.string();
                    if let Some(s) = file.submeshes.get_mut(index) {
                        s.name = Some(name);
                    }
                }
            }
            M_EDGE_LISTS => edge_lists(&mut r, version, &mut file)?,
            M_POSES => poses(&mut r, version, &mut file)?,
            M_ANIMATIONS => animations(&mut r, version, &mut file)?,
            _ => {
                // M_TABLE_EXTREMES: a submesh index and points, skipped.
                r.skip_chunk(start, len)?;
                file.extremes += len.saturating_sub(HEADER + 2) / 12;
            }
        }
    }
    Ok(file)
}

fn assignment(r: &mut Reader<'_>) -> Result<Assign> {
    Ok(Assign {
        vertex: r.u32()?,
        bone: r.u16()?,
        weight: r.f32()?,
    })
}

fn submesh(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<SubMesh> {
    let mut s = SubMesh {
        material: r.string(),
        shared: r.bool()?,
        operation: 4,
        ..SubMesh::default()
    };
    let count = r.u32()?;
    let idx32 = r.bool()?;
    if count > 0 {
        s.indices = Some(r.index_list(count, idx32)?);
    }
    if !s.shared {
        r.expect(M_GEOMETRY, "submesh geometry")?;
        s.geometry = Some(geometry(r, version, file)?);
    }
    while let Some((id, _, _)) = r.chunk(&[M_SUBMESH_OPERATION, M_SUBMESH_BONE_ASSIGNMENT, M_SUBMESH_TEXTURE_ALIAS]) {
        match id {
            M_SUBMESH_OPERATION => s.operation = r.u16()?,
            M_SUBMESH_BONE_ASSIGNMENT => s.bones.push(assignment(r)?),
            _ => s.aliases.push((r.string(), r.string())),
        }
    }
    Ok(s)
}

/// One vertex element of a declaration.
#[derive(Debug, Clone, Copy)]
struct Element {
    source: u16,
    ty: u16,
    semantic: u16,
    offset: u16,
    index: u16,
}

/// A geometry chunk's vertices, the chunk's header already read.
fn geometry(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<Geometry> {
    if version < Version::V130 {
        return old_geometry(r, version);
    }
    let count = r.u32()?;
    let mut elements = Vec::new();
    let mut buffers: Vec<(u16, usize, &[u8])> = Vec::new();
    while let Some((id, _, _)) = r.chunk(&[M_GEOMETRY_VERTEX_DECLARATION, M_GEOMETRY_VERTEX_BUFFER]) {
        if id == M_GEOMETRY_VERTEX_DECLARATION {
            while r.chunk(&[M_GEOMETRY_VERTEX_ELEMENT]).is_some() {
                elements.push(Element {
                    source: r.u16()?,
                    ty: r.u16()?,
                    semantic: r.u16()?,
                    offset: r.u16()?,
                    index: r.u16()?,
                });
            }
        } else {
            let bind = r.u16()?;
            let size = usize::from(r.u16()?);
            r.expect(M_GEOMETRY_VERTEX_BUFFER_DATA, "vertex buffer data")?;
            let n = r.items(count, size)?;
            buffers.push((bind, size, r.bytes(n * size)?));
        }
    }
    let n = usize_of(u64::from(count));
    decode(n, &elements, &buffers, r.order.big, file)
}

/// The bytes and component count of a vertex element type; `None` for a type there isn't.
fn element_size(ty: u16) -> Option<(usize, usize)> {
    let n = |first: u16| usize::from(ty - first + 1);
    Some(match ty {
        0..=3 => (4 * n(0), n(0)),
        4 | 9..=11 | 28..=30 | 35 => (4, 4),
        5..=8 => (2 * n(5), n(5)),
        12..=15 => (8 * n(12), n(12)),
        16..=19 => (2 * n(16), n(16)),
        20..=23 => (4 * n(20), n(20)),
        24..=27 => (4 * n(24), n(24)),
        31 | 33 => (4, 2),
        32 | 34 => (8, 4),
        36..=39 => (2 * n(36), n(36)),
        _ => return None,
    })
}

/// A half float.
fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let e = (h >> 10) & 0x1f;
    let m = f32::from(h & 0x3ff);
    match e {
        0 => sign * m * (2.0f32).powi(-24),
        31 if m == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m / 1024.0) * (2.0f32).powi(i32::from(e) - 15),
    }
}

/// One vertex's element as up to four floats (missing ones 0, 0, 0, 1). Integer types are taken as they are,
/// normalized types as fractions of their range, packed colors as bytes in R, G, B, A order (not yet divided by
/// 255). `short_unsigned` reads 16-bit signed integers as unsigned fractions, as texture coordinates hold them.
fn element(b: &[u8], ty: u16, be: bool, short_unsigned: bool) -> [f32; 4] {
    let mut out = [0.0, 0.0, 0.0, 1.0];
    let Some((_, n)) = element_size(ty) else {
        return out;
    };
    // `b` holds the element's bytes, as `element_size` says, so no read falls past it.
    let order = Endian { big: be };
    let w32 = |i: usize| bytes::array_at(b, 4 * i).map_or(0, |w| order.u32(w));
    let w16 = |i: usize| bytes::array_at(b, 2 * i).map_or(0, |w| order.u16(w));
    let b4: [u8; 4] = bytes::array_at(b, 0).unwrap_or_default();
    match ty {
        0..=3 => (0..n).for_each(|i| out[i] = f32::from_bits(w32(i))),
        4 | 10 | 11 => {
            // Packed colors are 32-bit words: their bytes swapped in a big-endian file.
            let mut c = b4;
            if be {
                c.reverse();
            }
            if ty == 10 {
                // A8R8G8B8 as a word, so B, G, R, A in memory.
                c.swap(0, 2);
            }
            out = c.map(f32::from);
        }
        5..=8 if short_unsigned => (0..n).for_each(|i| out[i] = f32::from(w16(i)) / 65535.0),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are a signed integer")]
        5..=8 => (0..n).for_each(|i| out[i] = f32::from(w16(i) as i16)),
        9 => out = b4.map(f32::from),
        12..=15 => (0..n).for_each(|i| {
            let v = bytes::array_at(b, 8 * i).map_or(0, |w| order.u64(w));
            out[i] = crate::num::f32_of(f64::from_bits(v));
        }),
        16..=19 => (0..n).for_each(|i| out[i] = f32::from(w16(i))),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are a signed integer")]
        20..=23 => (0..n).for_each(|i| out[i] = crate::num::f32_of(f64::from(w32(i) as i32))),
        24..=27 => (0..n).for_each(|i| out[i] = crate::num::f32_of(f64::from(w32(i)))),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are signed bytes")]
        28 => out = b4.map(|x| f32::from(x as i8)),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are signed bytes")]
        29 => out = b4.map(|x| (f32::from(x as i8) / 127.0).max(-1.0)),
        30 => out = b4.map(|x| f32::from(x) / 255.0),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are a signed integer")]
        31 | 32 => (0..n).for_each(|i| out[i] = (f32::from(w16(i) as i16) / 32767.0).max(-1.0)),
        33 | 34 => (0..n).for_each(|i| out[i] = f32::from(w16(i)) / 65535.0),
        35 => {
            let v = w32(0);
            for (i, o) in out.iter_mut().take(3).enumerate() {
                // A signed 10-bit field.
                let f = (v >> (10 * i)) & 0x3ff;
                let s = if f & 0x200 == 0 {
                    f32::from(u16::try_from(f).unwrap_or(0))
                } else {
                    f32::from(u16::try_from(f).unwrap_or(0)) - 1024.0
                };
                *o = (s / 511.0).max(-1.0);
            }
            let w = v >> 30;
            out[3] = if w & 2 == 0 {
                f32::from(u8::try_from(w).unwrap_or(0))
            } else {
                -1.0
            };
        }
        36..=39 => (0..n).for_each(|i| out[i] = half(w16(i))),
        _ => {}
    }
    out
}

/// Notes a dropped vertex element in the file's list (once each).
fn drop_element(file: &mut MeshFile, what: String) {
    if !file.dropped_elements.contains(&what) {
        file.dropped_elements.push(what);
    }
}

/// Decodes a declaration's elements from the vertex buffers they read.
fn decode(
    count: usize,
    elements: &[Element],
    buffers: &[(u16, usize, &[u8])],
    be: bool,
    file: &mut MeshFile,
) -> Result<Geometry> {
    let mut g = Geometry::default();
    let mut uv_sets: Vec<(u16, Vec<[f32; 2]>)> = Vec::new();
    let mut seen: Vec<(u16, u16)> = Vec::new();
    for e in elements {
        let Some((size, comps)) = element_size(e.ty) else {
            bail!(Corrupt, "OGRE vertex element type {} is unknown", e.ty);
        };
        let Some(&(_, stride, data)) = buffers.iter().find(|b| b.0 == e.source) else {
            bail!(
                Corrupt,
                "OGRE vertex element reads buffer {}, which isn't there",
                e.source
            );
        };
        let offset = usize::from(e.offset);
        ensure!(
            offset + size <= stride,
            Corrupt,
            "OGRE vertex element at offset {offset} runs past the end of its {stride}-byte vertex"
        );
        let wanted = match e.semantic {
            VES_POSITION | VES_NORMAL | VES_TANGENT | VES_BINORMAL | VES_DIFFUSE | VES_SPECULAR => e.index == 0,
            VES_TEXTURE_COORDINATES => true,
            _ => false,
        };
        if !wanted || seen.contains(&(e.semantic, e.index)) {
            let name = match e.semantic {
                VES_BLEND_WEIGHTS => "blend_weights".to_owned(),
                VES_BLEND_INDICES => "blend_indices".to_owned(),
                s @ (VES_POSITION | VES_NORMAL | VES_TANGENT | VES_BINORMAL | VES_DIFFUSE | VES_SPECULAR) => {
                    format!("semantic_{s}_index_{}", e.index)
                }
                s => format!("semantic_{s}"),
            };
            drop_element(file, name);
            continue;
        }
        if e.semantic == VES_TEXTURE_COORDINATES && uv_sets.len() >= MAX_UV_SETS {
            drop_element(file, format!("texcoord_{}", e.index));
            continue;
        }
        seen.push((e.semantic, e.index));
        let texcoord = e.semantic == VES_TEXTURE_COORDINATES;
        // Each buffer holds `count` vertices of `stride` bytes, and the element fits in one.
        let vertices = data
            .chunks_exact(stride)
            .take(count)
            .map(|v| element(v.get(offset..offset + size).unwrap_or_default(), e.ty, be, texcoord));
        let bytes = matches!(e.ty, 4 | 9..=11);
        match e.semantic {
            VES_POSITION => g.positions = vertices.map(|v| [v[0], v[1], v[2]]).collect(),
            VES_NORMAL => g.normals = vertices.map(|v| [v[0], v[1], v[2]]).collect(),
            VES_BINORMAL => g.binormals = vertices.map(|v| [v[0], v[1], v[2]]).collect(),
            VES_TANGENT => {
                g.tangents = vertices
                    .map(|v| [v[0], v[1], v[2], if comps == 4 { v[3] } else { f32::NAN }])
                    .collect();
            }
            VES_DIFFUSE | VES_SPECULAR => {
                let colors = vertices.map(|v| if bytes { v.map(|c| c / 255.0) } else { v }).collect();
                if e.semantic == VES_DIFFUSE {
                    g.diffuse = colors;
                } else {
                    g.specular = colors;
                }
            }
            _ => {
                if comps > 2 {
                    file.texcoord_w += 1;
                }
                let set = vertices.map(|v| [v[0], if comps == 1 { 0.0 } else { v[1] }]).collect();
                uv_sets.push((e.index, set));
            }
        }
    }
    uv_sets.sort_by_key(|s| s.0);
    g.uvs = uv_sets.into_iter().map(|s| s.1).collect();
    Ok(g)
}

/// A geometry chunk from before version 1.30: positions, then normal, color and texture coordinate chunks.
fn old_geometry(r: &mut Reader<'_>, version: Version) -> Result<Geometry> {
    let count = r.u32()?;
    let n = r.items(count, 12)?;
    let mut g = Geometry {
        positions: r.vec3s(n)?,
        ..Geometry::default()
    };
    while let Some((id, _, _)) = r.chunk(&[M_GEOMETRY_NORMALS, M_GEOMETRY_COLOURS, M_GEOMETRY_TEXCOORDS]) {
        match id {
            M_GEOMETRY_NORMALS => {
                r.items(count, 12)?;
                g.normals = r.vec3s(n)?;
            }
            M_GEOMETRY_COLOURS => {
                r.items(count, 4)?;
                let mut colors = Vec::with_capacity(n);
                for _ in 0..n {
                    colors.push(r.u32()?.to_le_bytes().map(|c| f32::from(c) / 255.0));
                }
                g.diffuse = colors;
            }
            _ => {
                let dim = usize::from(r.u16()?);
                ensure!(
                    (1..=4).contains(&dim),
                    Corrupt,
                    "OGRE texture coordinates have {dim} dimensions"
                );
                r.items(count, 4 * dim)?;
                let all = r.f32s(n * dim)?;
                let flip = version == Version::V110 && dim == 2;
                let set = all
                    .chunks_exact(dim)
                    .map(|c| {
                        let v = if dim > 1 { c[1] } else { 0.0 };
                        [c[0], if flip { 1.0 - v } else { v }]
                    })
                    .collect();
                if g.uvs.len() < MAX_UV_SETS {
                    g.uvs.push(set);
                }
            }
        }
    }
    Ok(g)
}

/// The level of detail chunk: the strategy, each level's value and manual mesh; generated levels' indices are
/// skipped.
fn lod(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<()> {
    let subs = file.submeshes.len();
    if version >= Version::V141 {
        file.lod_strategy = Some(r.string());
    }
    let levels = r.u16()?;
    if version == Version::V100 {
        for _ in 1..levels {
            let Some((id, _, _)) = r.chunk(&[M_MESH_LOD_MANUAL, M_MESH_LOD_GENERATED]) else {
                bail!(Corrupt, "OGRE level of detail chunk missing at byte {}", r.at);
            };
            let usage = r.f32()?;
            let mut manual = None;
            if id == M_MESH_LOD_MANUAL {
                manual = Some(r.string());
            } else {
                for _ in 0..subs {
                    let _count = r.u32()?;
                    let _offset = r.u32()?;
                    if r.u32()? == u32::MAX {
                        let idx32 = r.bool()?;
                        let n = r.u32()?;
                        let size = if idx32 { 4 } else { 2 };
                        let n = r.items(n, size)?;
                        r.bytes(n * size)?;
                    }
                }
            }
            file.lods.push(LodLevel { usage, manual });
        }
        return Ok(());
    }
    let is_manual = r.bool()?;
    for _ in 1..levels {
        r.expect(M_MESH_LOD_USAGE, "level of detail usage")?;
        let usage = r.f32()?;
        let mut manual = None;
        if is_manual {
            r.expect(M_MESH_LOD_MANUAL, "manual level of detail")?;
            manual = Some(r.string());
        } else {
            for _ in 0..subs {
                r.expect(M_MESH_LOD_GENERATED, "generated level of detail")?;
                let n = r.u32()?;
                let idx32 = r.bool()?;
                let size = if idx32 { 4 } else { 2 };
                let n = r.items(n, size)?;
                r.bytes(n * size)?;
            }
        }
        file.lods.push(LodLevel { usage, manual });
    }
    Ok(())
}

/// The edge lists, which glTF has no place for: read past.
fn edge_lists(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<()> {
    while r.chunk(&[M_EDGE_LIST_LOD]).is_some() {
        let _lod = r.u16()?;
        let manual = r.bool()?;
        file.edge_lists += 1;
        if manual {
            continue;
        }
        if version >= Version::V140 {
            let _closed = r.bool()?;
        }
        let triangles = r.u32()?;
        let groups = r.u32()?;
        let n = r.items(triangles, 48)?;
        r.bytes(n * 48)?;
        let group_header = if version >= Version::V140 { 12 } else { 4 };
        for _ in 0..groups {
            r.expect(M_EDGE_GROUP, "edge group")?;
            r.bytes(group_header)?;
            let edges = r.u32()?;
            let n = r.items(edges, 25)?;
            r.bytes(n * 25)?;
        }
    }
    Ok(())
}

fn poses(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<()> {
    while r.chunk(&[M_POSE]).is_some() {
        let name = r.string();
        let target = r.u16()?;
        let normals = version >= Version::V18 && r.bool()?;
        let mut offsets = Vec::new();
        while r.chunk(&[M_POSE_VERTEX]).is_some() {
            offsets.push(PoseOffset {
                index: r.u32()?,
                offset: r.vec3()?,
                normal: if normals { Some(r.vec3()?) } else { None },
            });
        }
        file.poses.push(Pose { name, target, offsets });
    }
    Ok(())
}

/// The number of vertices a pose or morph track's target holds: 0 the shared geometry, `k` submesh `k - 1`'s.
pub(super) fn target_vertices(file: &MeshFile, target: u16) -> Option<usize> {
    let g = if target == 0 {
        file.shared.as_ref()
    } else {
        file.submeshes.get(usize::from(target) - 1)?.geometry.as_ref()
    };
    g.map(|g| g.positions.len())
}

fn animations(r: &mut Reader<'_>, version: Version, file: &mut MeshFile) -> Result<()> {
    while r.chunk(&[M_ANIMATION]).is_some() {
        let mut a = MeshAnimation {
            name: r.string(),
            length: r.f32()?,
            base: None,
            tracks: Vec::new(),
        };
        while let Some((id, _, _)) = r.chunk(&[M_ANIMATION_BASEINFO, M_ANIMATION_TRACK]) {
            if id == M_ANIMATION_BASEINFO {
                a.base = Some((r.string(), r.f32()?));
                continue;
            }
            let kind = r.u16()?;
            let target = r.u16()?;
            let keys = match kind {
                1 => {
                    let vertices = target_vertices(file, target);
                    let mut keys = Vec::new();
                    while r.chunk(&[M_ANIMATION_MORPH_KEYFRAME]).is_some() {
                        let time = r.f32()?;
                        let normals = version >= Version::V18 && r.bool()?;
                        let Some(n) = vertices else {
                            bail!(Corrupt, "OGRE morph track targets geometry {target}, which isn't there");
                        };
                        let width = if normals { 24 } else { 12 };
                        let n = r.items(u32::try_from(n).unwrap_or(u32::MAX), width)?;
                        let all = r.vec3s(n * width / 12)?;
                        let (positions, normals) = if normals {
                            all.chunks_exact(2).map(|p| (p[0], p[1])).unzip()
                        } else {
                            (all, Vec::new())
                        };
                        keys.push(MorphKey {
                            time,
                            positions,
                            normals,
                        });
                    }
                    Keys::Morph(keys)
                }
                2 => {
                    let mut keys = Vec::new();
                    while r.chunk(&[M_ANIMATION_POSE_KEYFRAME]).is_some() {
                        let time = r.f32()?;
                        let mut refs = Vec::new();
                        while r.chunk(&[M_ANIMATION_POSE_REF]).is_some() {
                            refs.push((r.u16()?, r.f32()?));
                        }
                        keys.push(PoseKey { time, refs });
                    }
                    Keys::Pose(keys)
                }
                k => bail!(Corrupt, "OGRE animation track type {k} is unknown"),
            };
            a.tracks.push(Track { target, keys });
        }
        file.animations.push(a);
    }
    Ok(())
}

/// Reads a binary `.skeleton`.
pub(super) fn skeleton(data: &[u8]) -> Result<SkeletonFile> {
    let (be, line) = header(data).ok_or_else(|| err!(UnknownFormat, "not an OGRE skeleton"))?;
    ensure!(
        line.starts_with("[Serializer_v1."),
        Unsupported,
        "unknown OGRE skeleton version {line}"
    );
    let mut r = reader(data, be);
    r.string();
    let mut file = SkeletonFile {
        version: line.trim_matches(['[', ']']).to_owned(),
        big_endian: be,
        ..SkeletonFile::default()
    };
    while let Some((id, len)) = r.peek() {
        let start = r.at;
        r.at += HEADER;
        match id {
            SKELETON_BLENDMODE => {
                file.blend_mode = Some(if r.u16()? == 1 { "cumulative" } else { "average" });
            }
            SKELETON_BONE => {
                let name = r.string();
                let handle = r.u16()?;
                let position = r.vec3()?;
                let rotation = r.vec4()?;
                let scale = if len > BONE_WITHOUT_SCALE { r.vec3()? } else { [1.0; 3] };
                file.bones.push(Bone {
                    name,
                    handle,
                    position,
                    rotation,
                    scale,
                    parent: None,
                });
            }
            SKELETON_BONE_PARENT => {
                let child = r.u16()?;
                let parent = r.u16()?;
                if let Some(b) = file.bones.iter_mut().find(|b| b.handle == child) {
                    b.parent = Some(parent);
                }
            }
            SKELETON_ANIMATION => {
                let mut a = SkeletonAnimation {
                    name: r.string(),
                    length: r.f32()?,
                    base: None,
                    tracks: Vec::new(),
                };
                if r.chunk(&[SKELETON_ANIMATION_BASEINFO]).is_some() {
                    a.base = Some((r.string(), r.f32()?));
                }
                while r.chunk(&[SKELETON_ANIMATION_TRACK]).is_some() {
                    let bone = r.u16()?;
                    let mut keys = Vec::new();
                    while let Some((_, _, len)) = r.chunk(&[SKELETON_ANIMATION_TRACK_KEYFRAME]) {
                        let time = r.f32()?;
                        let rotation = r.vec4()?;
                        let translation = r.vec3()?;
                        let scale = if len > KEY_WITHOUT_SCALE { r.vec3()? } else { [1.0; 3] };
                        keys.push(Key {
                            time,
                            rotation,
                            translation,
                            scale,
                        });
                    }
                    a.tracks.push(BoneTrack { bone, keys });
                }
                file.animations.push(a);
            }
            SKELETON_ANIMATION_LINK => file.links.push((r.string(), r.f32()?)),
            _ => r.skip_chunk(start, len)?,
        }
    }
    Ok(file)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn halves() {
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(0x3800), 0.5);
        assert_eq!(half(0x0001), 2.0f32.powi(-24));
        assert!(half(0x7c00).is_infinite());
        assert!(half(0x7e00).is_nan());
    }

    #[test]
    fn packed_types() {
        // Colors: R, G, B, A in memory for 4 and 11; B, G, R, A for 10; reversed words in big-endian files.
        assert_eq!(element(&[1, 2, 3, 4], 11, false, false), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(element(&[1, 2, 3, 4], 10, false, false), [3.0, 2.0, 1.0, 4.0]);
        assert_eq!(element(&[4, 3, 2, 1], 4, true, false), [1.0, 2.0, 3.0, 4.0]);
        // 10-bit fields: x = 511 (1), y = -511 (-1), z = 0, w = 1.
        let v: u32 = 511 | ((1024 - 511) << 10) | (1 << 30);
        assert_eq!(element(&v.to_le_bytes(), 35, false, false), [1.0, -1.0, 0.0, 1.0]);
        assert_eq!(
            element(&[0x80, 0x7f, 0, 1], 29, false, false),
            [-1.0, 1.0, 0.0, 1.0 / 127.0]
        );
        assert_eq!(element(&0xffffu16.to_le_bytes(), 5, false, true), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(element(&0xffffu16.to_be_bytes(), 5, true, false), [-1.0, 0.0, 0.0, 1.0]);
        assert_eq!(element(&[0, 0x3c], 36, false, false), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn short_elements_read_as_zero_not_a_panic() {
        for ty in 0..=40 {
            for len in 0..8 {
                let _ = element(&[0x7f; 8][..len], ty, len % 2 == 0, len % 3 == 0);
            }
        }
        assert_eq!(element(&[1, 0], 2, false, false), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn chunks_and_counts_stop_at_the_end() {
        // The header, a line, then a chunk header cut short.
        let mut data = vec![0x00, 0x10];
        data.extend(b"[MeshSerializer_v1.100]\n");
        let line_end = data.len();
        data.extend([0x00, 0x30, 0xFF, 0xFF]);
        let mut r = reader(&data, false);
        assert_eq!(r.string(), "[MeshSerializer_v1.100]");
        assert_eq!(r.peek(), None);
        assert!(r.chunk(&[0x3000]).is_none());
        assert!(r.expect(0x3000, "mesh").is_err());
        assert!(r.skip_chunk(line_end, usize::MAX).is_err());
        let e = r.items(u32::MAX, 4).unwrap_err();
        assert_eq!(e.kind(), crate::ErrorKind::Truncated);
        assert!(e.to_string().contains("runs past the end"), "{e}");
        assert!(r.index_list(u32::MAX, true).is_err());
        assert!(r.index_list(3, false).is_err());
        assert_eq!(r.index_list(2, false).unwrap(), [0x3000, 0xFFFF]);
        let e = r.u32().unwrap_err();
        assert!(e.to_string().contains("cut short"), "{e}");
        assert_eq!(r.string(), "");
    }
}
