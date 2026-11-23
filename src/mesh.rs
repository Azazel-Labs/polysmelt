//! A model as plain triangle lists: what the format readers produce, written out as a GLB.

use crate::Result;
use crate::document::Document;
use crate::num::u32_of;

/// The most triangles a reader accepts, against hostile or runaway files.
pub const MAX_TRIANGLES: usize = 4_000_000;

/// One material's worth of triangles.
#[derive(Debug, Default, Clone)]
pub struct Prim {
    pub positions: Vec<[f32; 3]>,
    /// Unit vertex normals; empty for a faceted model (STL), which glTF clients shade flat.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates, glTF convention (origin top-left); empty when the model has none.
    pub uvs: Vec<[f32; 2]>,
    /// Vertex colors, linear RGBA; empty when the model has none.
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material: usize,
}

/// A material: a base color, optionally textured, an optional normal map, and the metallic-roughness scalars.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub color: [f32; 4],
    /// Index into `Mesh::images`.
    pub color_image: Option<usize>,
    pub normal_image: Option<usize>,
    /// Linear RGB light given off.
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    /// Seen from both sides rather than culled from behind.
    pub double_sided: bool,
}

impl Material {
    /// A material of one color, without textures.
    #[must_use]
    pub fn plain(name: &str, color: [f32; 4]) -> Self {
        Self {
            name: name.to_owned(),
            color,
            color_image: None,
            normal_image: None,
            emissive: [0.0; 3],
            metallic: 0.0,
            roughness: 0.85,
            double_sided: false,
        }
    }
}

/// An image ready to embed: PNG or JPEG bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The encoded image.
    pub bytes: Vec<u8>,
    /// `image/png` or `image/jpeg`.
    pub mime: &'static str,
}

/// A model as triangles grouped by material, ready to write as glTF.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    pub prims: Vec<Prim>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
}

impl Mesh {
    /// The number of triangles across all primitives.
    #[must_use]
    pub fn triangles(&self) -> usize {
        self.prims.iter().map(|p| p.indices.len() / 3).sum()
    }
}

/// The color of a material a format leaves unspecified.
pub const GRAY: [f32; 4] = [0.72, 0.72, 0.74, 1.0];

impl Image {
    /// `bytes` as an image when they are PNG or JPEG, the formats glTF embeds without an
    /// extension.
    #[must_use]
    pub fn png_or_jpeg(bytes: &[u8]) -> Option<Self> {
        Self::mime_of(bytes)?;
        Self::from_png_or_jpeg(bytes.to_vec())
    }

    /// [`Image::png_or_jpeg`], taking the bytes without a copy.
    #[must_use]
    pub fn from_png_or_jpeg(bytes: Vec<u8>) -> Option<Self> {
        let mime = Self::mime_of(&bytes)?;
        Some(Self { bytes, mime })
    }

    /// `bytes` as an image: PNG and JPEG as they are, and with the `webp` feature, WebP
    /// re-encoded as PNG.
    #[must_use]
    pub fn decode(bytes: Vec<u8>) -> Option<Self> {
        if Self::mime_of(&bytes).is_some() {
            Self::from_png_or_jpeg(bytes)
        } else {
            Self::webp(&bytes)
        }
    }

    /// WebP `bytes` re-encoded as PNG, with the `webp` feature.
    pub(crate) fn webp(bytes: &[u8]) -> Option<Self> {
        if !(bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP") {
            return None;
        }
        #[cfg(feature = "webp")]
        return Some(Self {
            bytes: crate::webp::to_png(bytes)?,
            mime: "image/png",
        });
        #[cfg(not(feature = "webp"))]
        None
    }

    fn mime_of(bytes: &[u8]) -> Option<&'static str> {
        if bytes.starts_with(b"\x89PNG") {
            Some("image/png")
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some("image/jpeg")
        } else {
            None
        }
    }
}

/// An sRGB-encoded color channel in linear light.
#[must_use]
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Smooth normals from the triangles, for models that don't carry any.
#[must_use]
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut nrm = vec![[0.0f32; 3]; positions.len()];
    for t in indices.as_chunks::<3>().0 {
        let n = face_normal(
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        for &i in t {
            for (a, b) in nrm[i as usize].iter_mut().zip(n) {
                *a += b;
            }
        }
    }
    for n in &mut nrm {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        *n = if len > 0.0 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    nrm
}

/// One gray primitive.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn single(p: Prim) -> Mesh {
    Mesh {
        prims: vec![p],
        materials: vec![Material::plain("default", GRAY)],
        images: Vec::new(),
    }
}

pub(crate) fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Builds a primitive from triangles given as corner positions, storing each distinct point
/// once (by bit pattern; `-0.0` counts as `0.0`) and the triangles as indices into them. No
/// normals: glTF clients shade a primitive without normals flat, which is what a triangle soup
/// (STL) means.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) struct Welder {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    /// Open addressing, linear probing: 0 for an empty slot, otherwise a position's index + 1.
    table: Vec<u32>,
    /// 64 minus log2 of the table's size: a hash's top bits pick the slot.
    shift: u32,
}

#[allow(dead_code)]
impl Welder {
    /// Room for about `tris` triangles of a closed mesh (half as many points).
    pub fn with_capacity(tris: usize) -> Self {
        let size = tris.max(16).next_power_of_two();
        Self {
            positions: Vec::with_capacity(tris / 2 + 16),
            indices: Vec::with_capacity(tris * 3),
            table: vec![0; size],
            shift: 64 - size.trailing_zeros(),
        }
    }

    /// Room for `tris` triangles in all, once a reader can estimate the total better.
    pub fn expect(&mut self, tris: usize) {
        self.indices
            .reserve_exact((tris * 3).saturating_sub(self.indices.len()));
        self.positions
            .reserve_exact((tris / 2 + 16).saturating_sub(self.positions.len()));
        while self.table.len() < tris.next_power_of_two() {
            self.grow();
        }
    }

    #[inline]
    fn slot(&self, key: [u32; 3]) -> usize {
        let h = (u64::from(key[0]) | u64::from(key[1]) << 32).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(key[2]).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "`shift` leaves only as many bits as the table has slots"
        )]
        let slot = (h >> self.shift) as usize;
        slot
    }

    /// The index of point `p`, added if it's new.
    #[inline]
    pub fn index(&mut self, p: [f32; 3]) -> u32 {
        let p = p.map(|c| c + 0.0);
        let key = p.map(f32::to_bits);
        let mask = self.table.len() - 1;
        let mut i = self.slot(key);
        loop {
            match self.table[i] {
                0 => break,
                e => {
                    if self.positions[(e - 1) as usize].map(f32::to_bits) == key {
                        return e - 1;
                    }
                }
            }
            i = (i + 1) & mask;
        }
        let at = u32_of(self.positions.len());
        self.positions.push(p);
        self.table[i] = at + 1;
        // Keep the table at most 5/8 full.
        if self.positions.len() * 8 > self.table.len() * 5 {
            self.grow();
        }
        at
    }

    fn grow(&mut self) {
        let size = self.table.len() * 2;
        self.table = vec![0; size];
        self.shift -= 1;
        let mask = size - 1;
        for (at, p) in self.positions.iter().enumerate() {
            let mut i = self.slot(p.map(f32::to_bits));
            while self.table[i] != 0 {
                i = (i + 1) & mask;
            }
            self.table[i] = u32_of(at) + 1;
        }
    }

    /// One triangle, by its corners.
    #[inline]
    pub fn push(&mut self, t: [[f32; 3]; 3]) {
        let i = t.map(|p| self.index(p));
        self.indices.extend_from_slice(&i);
    }

    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// The primitive, its buffers trimmed to size.
    pub fn finish(mut self) -> Prim {
        drop(std::mem::take(&mut self.table));
        let trim = |len: usize, cap: usize| cap > len + len / 16 + 16;
        if trim(self.positions.len(), self.positions.capacity()) {
            self.positions.shrink_to_fit();
        }
        if trim(self.indices.len(), self.indices.capacity()) {
            self.indices.shrink_to_fit();
        }
        Prim {
            positions: self.positions,
            indices: self.indices,
            ..Prim::default()
        }
    }
}

/// A GLB with one mesh: a primitive per `Prim`, its materials, and the images embedded.
///
/// # Errors
///
/// When the mesh has nothing to draw.
pub fn to_glb(mesh: &Mesh) -> Result<Vec<u8>> {
    ensure!(mesh.prims.iter().any(|p| !p.indices.is_empty()), "nothing to draw");
    Document::from_mesh(mesh.clone(), serde_json::Map::new()).to_glb()
}

/// Triangulates one polygon face, its corners in order, appending triangles as indices into
/// `corners` that keep the polygon's winding. Convex polygons (nearly every face) are fanned;
/// concave ones are ear-clipped in the polygon's plane, so no triangle falls outside the
/// outline.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn triangulate(corners: &[[f32; 3]], out: &mut Vec<[u32; 3]>) {
    let n = corners.len();
    let fan = |out: &mut Vec<[u32; 3]>, idx: &[u32]| {
        for k in 1..idx.len().saturating_sub(1) {
            out.push([idx[0], idx[k], idx[k + 1]]);
        }
    };
    if n < 3 {
        return;
    }
    if n == 3 {
        out.push([0, 1, 2]);
        return;
    }
    // The plane's normal (Newell), then 2D coordinates along its two other axes.
    let mut nrm = [0f64; 3];
    for (i, a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % n];
        let (a, b) = (a.map(f64::from), b.map(f64::from));
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let axis = (0..3)
        .max_by(|&i, &j| nrm[i].abs().total_cmp(&nrm[j].abs()))
        .unwrap_or(2);
    if nrm[axis] == 0.0 || n > 4096 {
        out.extend((1..u32_of(n) - 1).map(|k| [0, k, k + 1]));
        return;
    }
    // Project so the polygon winds counter-clockwise in 2D.
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let flip = nrm[axis] < 0.0;
    let to2 = |c: &[f32; 3]| {
        let (x, y) = (f64::from(c[u]), f64::from(c[v]));
        if flip { [y, x] } else { [x, y] }
    };
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if n == 4 {
        // Quads, most faces that aren't triangles: no allocation. A convex one is fanned; one
        // with a single reflex corner is cut along the diagonal from that corner.
        let q = [to2(&corners[0]), to2(&corners[1]), to2(&corners[2]), to2(&corners[3])];
        let turn = |k: usize| cross(q[(k + 3) % 4], q[k], q[(k + 1) % 4]);
        let mut r = None;
        let mut count = 0;
        for k in 0..4 {
            if turn(k) < 0.0 {
                r = Some(u32_of(k));
                count += 1;
            }
        }
        match (count, r) {
            (0, _) => {
                out.extend([[0, 1, 2], [0, 2, 3]]);
                return;
            }
            (1, Some(r)) => {
                let c = |k: u32| (r + k) % 4;
                out.extend([[r, c(1), c(2)], [r, c(2), c(3)]]);
                return;
            }
            _ => {}
        }
    }
    let all: Vec<u32> = (0..u32_of(n)).collect();
    let p: Vec<[f64; 2]> = corners.iter().map(to2).collect();
    if (0..n).all(|i| cross(p[i], p[(i + 1) % n], p[(i + 2) % n]) >= 0.0) {
        fan(out, &all);
        return;
    }
    // Ear clipping: cut off a convex corner whose triangle holds no other corner.
    let mut idx = all;
    while idx.len() > 3 {
        let m = idx.len();
        let ear = (0..m).find(|&i| {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (pa, pb, pc) = (p[a as usize], p[b as usize], p[c as usize]);
            if cross(pa, pb, pc) <= 0.0 {
                return false;
            }
            !idx.iter().any(|&q| {
                if q == a || q == b || q == c {
                    return false;
                }
                let pq = p[q as usize];
                cross(pa, pb, pq) >= 0.0 && cross(pb, pc, pq) >= 0.0 && cross(pc, pa, pq) >= 0.0
            })
        });
        let Some(i) = ear else {
            // Self-intersecting or degenerate: nothing better to do than fan what's left.
            break;
        };
        out.push([idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]]);
        idx.remove(i);
    }
    fan(out, &idx);
}

#[cfg(test)]
mod triangulate_tests {
    use super::triangulate;

    fn area(c: &[[f32; 3]], t: &[[u32; 3]]) -> f32 {
        t.iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| c[i as usize]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
            })
            .sum()
    }

    #[test]
    fn concave_and_convex() {
        let ell = [
            [2., 1., 0.],
            [1., 1., 0.],
            [1., 2., 0.],
            [0., 2., 0.],
            [0., 0., 0.],
            [2., 0., 0.],
        ];
        let mut t = Vec::new();
        triangulate(&ell, &mut t);
        assert_eq!(t.len(), 4);
        // Signed: every triangle counter-clockwise like the polygon, and they tile it exactly.
        assert!(t.iter().all(|t| area(&ell, &[*t]) > 0.0));
        assert_eq!(area(&ell, &t), 3.0);
        // Clockwise (seen from +z) stays clockwise.
        let rev: Vec<[f32; 3]> = ell.iter().rev().copied().collect();
        t.clear();
        triangulate(&rev, &mut t);
        assert_eq!(area(&rev, &t), -3.0);
        let quad = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        t.clear();
        triangulate(&quad, &mut t);
        assert_eq!(t, [[0, 1, 2], [0, 2, 3]]);
    }
}

/// A multiply-rotate hasher (the Fx hash) for small integer keys: much faster than the
/// default `SipHash`, and keys here come from the file being read, not from an attacker who can
/// see the hash.
#[cfg(any(feature = "ply", feature = "3ds", feature = "fbx"))]
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

#[cfg(any(feature = "ply", feature = "3ds", feature = "fbx"))]
impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A `HashMap` with [`FxHasher`].
#[cfg(any(feature = "ply", feature = "3ds", feature = "fbx"))]
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;
