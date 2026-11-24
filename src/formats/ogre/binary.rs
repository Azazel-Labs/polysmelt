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

/// The most texture coordinate sets a geometry keeps.
const MAX_UV_SETS: usize = 8;

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
    let mut r = Reader { data, at: 2, be };
    let line = r.string();
    line.starts_with('[').then_some((be, line))
}

/// A binary file being read: its bytes, where the reader is, and its byte order.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    be: bool,
}

impl<'a> Reader<'a> {
    fn left(&self) -> usize {
        self.data.len() - self.at
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(n <= self.left(), "OGRE file cut short at byte {}", self.at);
        let b = &self.data[self.at..self.at + n];
        self.at += n;
        Ok(b)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0; N];
        a.copy_from_slice(self.bytes(N)?);
        Ok(a)
    }

    fn bool(&mut self) -> Result<bool> {
        Ok(self.array::<1>()?[0] != 0)
    }

    fn u16(&mut self) -> Result<u16> {
        let a = self.array()?;
        Ok(if self.be {
            u16::from_be_bytes(a)
        } else {
            u16::from_le_bytes(a)
        })
    }

    fn u32(&mut self) -> Result<u32> {
        let a = self.array()?;
        Ok(if self.be {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        })
    }

    fn f32(&mut self) -> Result<f32> {
        self.u32().map(f32::from_bits)
    }

    fn vec3(&mut self) -> Result<[f32; 3]> {
        Ok([self.f32()?, self.f32()?, self.f32()?])
    }

    fn vec4(&mut self) -> Result<[f32; 4]> {
        Ok([self.f32()?, self.f32()?, self.f32()?, self.f32()?])
    }

    /// A string: the bytes up to a line feed (or the end), without a carriage return before it.
    fn string(&mut self) -> String {
        let rest = &self.data[self.at..];
        let end = rest.iter().position(|&b| b == b'\n');
        let line = &rest[..end.unwrap_or(rest.len())];
        self.at += end.map_or(rest.len(), |e| e + 1);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        crate::formats::scene::text(line)
    }

    /// The next chunk's id and length, without reading them.
    fn peek(&self) -> Option<(u16, usize)> {
        let mut r = Reader {
            data: self.data,
            at: self.at,
            be: self.be,
        };
        Some((r.u16().ok()?, usize_of(u64::from(r.u32().ok()?))))
    }

    /// Reads the next chunk's header when its id is one of `ids`: its id, where it starts and its length.
    fn chunk(&mut self, ids: &[u16]) -> Option<(u16, usize, usize)> {
        let (id, len) = self.peek()?;
        if !ids.contains(&id) {
            return None;
        }
        let start = self.at;
        self.at += HEADER;
        Some((id, start, len))
    }

    /// The next chunk's header, which must have id `id`.
    fn expect(&mut self, id: u16, what: &str) -> Result<(usize, usize)> {
        match self.chunk(&[id]) {
            Some((_, start, len)) => Ok((start, len)),
            None => bail!("OGRE {what} chunk missing at byte {}", self.at),
        }
    }

    /// Moves past a chunk that starts at `start` and is `len` long.
    fn skip_chunk(&mut self, start: usize, len: usize) -> Result<()> {
        let end = start.saturating_add(len);
        ensure!(
            len >= HEADER && end <= self.data.len() && end >= self.at,
            "OGRE chunk at byte {start} has a bad length"
        );
        self.at = end;
        Ok(())
    }

    /// `n` items of `size` bytes each, checked against what's left.
    fn count(&self, n: u32, size: usize) -> Result<usize> {
        let n = usize_of(u64::from(n));
        ensure!(
            n.checked_mul(size).is_some_and(|b| b <= self.left()),
            "OGRE count {n} at byte {} runs past the end of the file",
            self.at
        );
        Ok(n)
    }

    /// `n` vectors of three floats.
    fn vec3s(&mut self, n: usize) -> Result<Vec<[f32; 3]>> {
        let b = self.bytes(n.saturating_mul(12))?;
        let be = self.be;
        Ok(b.chunks_exact(12)
            .map(|v| [0, 4, 8].map(|o| f32::from_bits(word(&v[o..o + 4], be))))
            .collect())
    }

    /// `n` indices, 16 or 32 bits each.
    fn indices(&mut self, n: u32, idx32: bool) -> Result<Vec<u32>> {
        let size = if idx32 { 4 } else { 2 };
        let n = self.count(n, size)?;
        let b = self.bytes(n * size)?;
        let be = self.be;
        Ok(if idx32 {
            b.chunks_exact(4).map(|c| word(c, be)).collect()
        } else {
            b.chunks_exact(2).map(|c| u32::from(half_word(c, be))).collect()
        })
    }
}

/// A 32-bit word from four bytes.
fn word(b: &[u8], be: bool) -> u32 {
    let a = [b[0], b[1], b[2], b[3]];
    if be {
        u32::from_be_bytes(a)
    } else {
        u32::from_le_bytes(a)
    }
}

/// A 16-bit word from two bytes.
fn half_word(b: &[u8], be: bool) -> u16 {
    let a = [b[0], b[1]];
    if be {
        u16::from_be_bytes(a)
    } else {
        u16::from_le_bytes(a)
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
            bail!("OGRE mesh {l} is the Ogre-Next v2 layout, which isn't read")
        }
        l => bail!("unknown OGRE mesh version {l}"),
    })
}

/// Reads a binary `.mesh`.
pub(super) fn mesh(data: &[u8]) -> Result<MeshFile> {
    let (be, line) = header(data).ok_or_else(|| invalid!("not an OGRE mesh"))?;
    let version = mesh_version(&line)?;
    let mut r = Reader { data, at: 2, be };
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
        s.indices = Some(r.indices(count, idx32)?);
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
            let n = r.count(count, size)?;
            buffers.push((bind, size, r.bytes(n * size)?));
        }
    }
    let n = usize_of(u64::from(count));
    decode(n, &elements, &buffers, r.be, file)
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
    let w32 = |i: usize| word(&b[4 * i..4 * i + 4], be);
    let w16 = |i: usize| half_word(&b[2 * i..2 * i + 2], be);
    match ty {
        0..=3 => (0..n).for_each(|i| out[i] = f32::from_bits(w32(i))),
        4 | 10 | 11 => {
            // Packed colors are 32-bit words: their bytes swapped in a big-endian file.
            let mut c = [b[0], b[1], b[2], b[3]];
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
        9 => out = [b[0], b[1], b[2], b[3]].map(f32::from),
        12..=15 => (0..n).for_each(|i| {
            let mut a = [0; 8];
            a.copy_from_slice(&b[8 * i..8 * i + 8]);
            let v = if be {
                f64::from_be_bytes(a)
            } else {
                f64::from_le_bytes(a)
            };
            out[i] = crate::num::f32_of(v);
        }),
        16..=19 => (0..n).for_each(|i| out[i] = f32::from(w16(i))),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are a signed integer")]
        20..=23 => (0..n).for_each(|i| out[i] = crate::num::f32_of(f64::from(w32(i) as i32))),
        24..=27 => (0..n).for_each(|i| out[i] = crate::num::f32_of(f64::from(w32(i)))),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are signed bytes")]
        28 => out = [b[0], b[1], b[2], b[3]].map(|x| f32::from(x as i8)),
        #[allow(clippy::cast_possible_wrap, reason = "the bits are signed bytes")]
        29 => out = [b[0], b[1], b[2], b[3]].map(|x| (f32::from(x as i8) / 127.0).max(-1.0)),
        30 => out = [b[0], b[1], b[2], b[3]].map(|x| f32::from(x) / 255.0),
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
            bail!("OGRE vertex element type {} is unknown", e.ty);
        };
        let Some(&(_, stride, data)) = buffers.iter().find(|b| b.0 == e.source) else {
            bail!("OGRE vertex element reads buffer {}, which isn't there", e.source);
        };
        let offset = usize::from(e.offset);
        ensure!(
            offset + size <= stride,
            "OGRE vertex element at offset {offset} runs past its {stride}-byte vertex"
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
        let vertices = (0..count).map(|v| {
            let at = v * stride + offset;
            element(&data[at..at + size], e.ty, be, texcoord)
        });
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
    let n = r.count(count, 12)?;
    let mut g = Geometry {
        positions: r.vec3s(n)?,
        ..Geometry::default()
    };
    while let Some((id, _, _)) = r.chunk(&[M_GEOMETRY_NORMALS, M_GEOMETRY_COLOURS, M_GEOMETRY_TEXCOORDS]) {
        match id {
            M_GEOMETRY_NORMALS => {
                r.count(count, 12)?;
                g.normals = r.vec3s(n)?;
            }
            M_GEOMETRY_COLOURS => {
                r.count(count, 4)?;
                let mut colors = Vec::with_capacity(n);
                for _ in 0..n {
                    colors.push(r.u32()?.to_le_bytes().map(|c| f32::from(c) / 255.0));
                }
                g.diffuse = colors;
            }
            _ => {
                let dim = usize::from(r.u16()?);
                ensure!((1..=4).contains(&dim), "OGRE texture coordinates have {dim} dimensions");
                r.count(count, 4 * dim)?;
                let b = r.bytes(n * 4 * dim)?;
                let be = r.be;
                let flip = version == Version::V110 && dim == 2;
                let set = b
                    .chunks_exact(4 * dim)
                    .map(|c| {
                        let u = f32::from_bits(word(c, be));
                        let v = if dim > 1 {
                            f32::from_bits(word(&c[4..], be))
                        } else {
                            0.0
                        };
                        [u, if flip { 1.0 - v } else { v }]
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
                bail!("OGRE level of detail chunk missing at byte {}", r.at);
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
                        let n = r.count(n, size)?;
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
                let n = r.count(n, size)?;
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
        let n = r.count(triangles, 48)?;
        r.bytes(n * 48)?;
        let group_header = if version >= Version::V140 { 12 } else { 4 };
        for _ in 0..groups {
            r.expect(M_EDGE_GROUP, "edge group")?;
            r.bytes(group_header)?;
            let edges = r.u32()?;
            let n = r.count(edges, 25)?;
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
                            bail!("OGRE morph track targets geometry {target}, which isn't there");
                        };
                        let width = if normals { 24 } else { 12 };
                        let n = r.count(u32::try_from(n).unwrap_or(u32::MAX), width)?;
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
                k => bail!("OGRE animation track type {k} is unknown"),
            };
            a.tracks.push(Track { target, keys });
        }
        file.animations.push(a);
    }
    Ok(())
}

/// Reads a binary `.skeleton`.
pub(super) fn skeleton(data: &[u8]) -> Result<SkeletonFile> {
    let (be, line) = header(data).ok_or_else(|| invalid!("not an OGRE skeleton"))?;
    ensure!(
        line.starts_with("[Serializer_v1."),
        "unknown OGRE skeleton version {line}"
    );
    let mut r = Reader { data, at: 2, be };
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
}
