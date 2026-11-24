//! A model as plain triangle lists (and line and point lists): what the format readers
//! produce, written out as a GLB.

use crate::Result;
use crate::document::Document;
use crate::num::u32_of;

/// The most triangles a load accepts by default, against hostile or runaway files:
/// [`Keep::max_triangles`](crate::document::Keep::max_triangles) raises or lowers it.
pub const MAX_TRIANGLES: usize = 64_000_000;

/// The most triangles any load accepts, whatever
/// [`Keep::max_triangles`](crate::document::Keep::max_triangles) says: three vertices to each
/// still fit a 32-bit index.
pub const TRIANGLE_CEILING: usize = 1 << 30;

/// The most points and line segments a flattened model draws by default, against huge point
/// clouds: [`Keep::max_points`](crate::document::Keep::max_points) raises or lowers it.
pub const MAX_POINTS: usize = 16_000_000;

/// One material's worth of triangles, and of the lines and points drawn on the same vertices.
#[derive(Debug, Default, Clone)]
pub struct Prim {
    pub positions: Vec<[f32; 3]>,
    /// Unit vertex normals; empty for a faceted model (STL), which glTF clients shade flat.
    pub normals: Vec<[f32; 3]>,
    /// Unit tangents for normal mapping, as glTF stores them: `xyz` the direction, `w` ±1 the
    /// bitangent's handedness (`bitangent = cross(normal, xyz) * w`). Empty when the model has
    /// none, or has no normals.
    pub tangents: Vec<[f32; 4]>,
    /// The texture coordinates the color texture reads, glTF convention (origin top-left), with
    /// its texture transform applied; empty when the model has none.
    pub uvs: Vec<[f32; 2]>,
    /// Every other texture coordinate set, by number (glTF's `TEXCOORD_n`): lightmaps, masks,
    /// detail maps. The slot of the set `uvs` holds is left empty, as is any set a model lacks,
    /// so nothing is stored twice; empty when the model has only the one set.
    pub uv_sets: Vec<Vec<[f32; 2]>>,
    /// Vertex colors, linear RGBA (glTF's `COLOR_0`); empty when the model has none.
    pub colors: Vec<[f32; 4]>,
    /// Every other vertex color set, by number (glTF's `COLOR_n`), linear RGBA. Slot 0, which
    /// `colors` holds, is left empty, as is any set a model lacks; empty when the model has
    /// only the one set.
    pub color_sets: Vec<Vec<[f32; 4]>>,
    /// Triangles, three vertex indices each.
    pub indices: Vec<u32>,
    /// Line segments, two vertex indices each (glTF's `LINES`; line strips and loops are
    /// stored as their segments).
    pub lines: Vec<u32>,
    /// Points, a vertex index each (glTF's `POINTS`): a point cloud is a primitive of these
    /// alone.
    pub points: Vec<u32>,
    pub material: usize,
    /// Each vertex's most-weighted joint, as a node index (the order of `Rig::nodes`), when the primitive is skinned;
    /// empty otherwise. Tells which bone each part of the surface moves with, for fitting a body to its skeleton.
    /// `u32::MAX` for a vertex no joint moves.
    pub joints: Vec<u32>,
    /// Each vertex's four strongest joints, as indices into [`Mesh::skeleton`], strongest first, when the
    /// primitive is skinned; empty otherwise. A file may weight a vertex to more than four joints: they're all
    /// used to pose it, and the four strongest kept here.
    pub skin_joints: Vec<[u32; 4]>,
    /// The weights of `skin_joints`, summing to 1, or all 0 for a vertex no joint moves.
    pub skin_weights: Vec<[f32; 4]>,
}

impl Prim {
    /// Whether it draws nothing: no triangles, lines or points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }

    /// Whether every vertex has joints and weights, so the primitive can be bound to [`Mesh::skeleton`].
    #[must_use]
    pub fn skinned(&self) -> bool {
        !self.positions.is_empty()
            && self.skin_joints.len() == self.positions.len()
            && self.skin_weights.len() == self.positions.len()
    }
}

/// A joint skinned primitives are bound to.
#[derive(Debug, Clone, PartialEq)]
pub struct Bone {
    pub name: Option<String>,
    /// The bone it hangs from, an index into [`Mesh::skeleton`], always before this one.
    pub parent: Option<usize>,
    /// The document node it was made from.
    pub node: usize,
    /// Where the bone is in the pose the mesh was drawn in: its world transform, column-major.
    pub matrix: [f32; 16],
}

pub use crate::material::{
    Anisotropy, CLAMP, Clearcoat, Factor, Iridescence, LINEAR, LINEAR_MIPMAP_LINEAR, MIRROR, REPEAT, Sheen, Specular,
    UvTransform, Volume,
};

/// How a material samples a texture: the coordinates it reads, a transform on them, and the
/// sampler's wrap and filters. The default reads [`Prim::uvs`] as they are, repeating, filtered
/// linearly between mipmaps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// The coordinates read: `None` is [`Prim::uvs`], `Some(n)` is [`Prim::uv_sets`]`[n]`.
    pub uv_set: Option<usize>,
    /// Moves those coordinates before sampling (glTF's `KHR_texture_transform`).
    pub transform: Option<UvTransform>,
    /// S and T wrap modes, glTF's numbers: [`REPEAT`], [`CLAMP`] or [`MIRROR`].
    pub wrap: [u32; 2],
    /// Magnification and minification filters, glTF's numbers (9728 nearest, 9729 linear,
    /// 9984 to 9987 between mipmaps); `None` leaves the choice to the viewer.
    pub filter: [Option<u32>; 2],
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            uv_set: None,
            transform: None,
            wrap: [REPEAT; 2],
            filter: [Some(LINEAR), Some(LINEAR_MIPMAP_LINEAR)],
        }
    }
}

/// How each of a material's core textures is sampled.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Samplings {
    pub color: Sampling,
    pub normal: Sampling,
    pub metallic_roughness: Sampling,
    pub occlusion: Sampling,
    pub emissive: Sampling,
}

/// A texture an extension uses: an image in [`Mesh::images`], and how it's sampled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Texture {
    pub image: usize,
    pub sampling: Sampling,
}

/// The `KHR_materials_*` extensions to glTF's metallic-roughness model: index of refraction,
/// specular, transmission, clearcoat, sheen, volume, anisotropy, iridescence and dispersion.
/// A `None` is the extension left out.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extensions {
    /// `KHR_materials_ior`.
    pub ior: Option<f32>,
    pub specular: Option<Specular<Texture>>,
    /// `KHR_materials_transmission`: how much light passes through, its texture's red scaling it.
    pub transmission: Option<Factor<Texture>>,
    pub clearcoat: Option<Clearcoat<Texture>>,
    pub sheen: Option<Sheen<Texture>>,
    pub volume: Option<Volume<Texture>>,
    pub anisotropy: Option<Anisotropy<Texture>>,
    pub iridescence: Option<Iridescence<Texture>>,
    /// `KHR_materials_dispersion`: how much the index of refraction varies with wavelength.
    pub dispersion: Option<f32>,
}

/// A material, glTF's metallic-roughness model: a base color, the metallic and roughness
/// scalars, and the textures that vary them. The textures read [`Prim::uvs`] unless their
/// [`Sampling`] says otherwise; the images are indices into [`Mesh::images`].
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    /// Linear RGBA. Its alpha is opacity: below 1 the material is blended, unless
    /// `alpha_cutoff` makes it cut out instead.
    pub color: [f32; 4],
    pub color_image: Option<usize>,
    /// A tangent-space normal map.
    pub normal_image: Option<usize>,
    /// Roughness in the green channel, metalness in the blue, as glTF packs them; they scale
    /// `roughness` and `metallic`.
    pub metallic_roughness_image: Option<usize>,
    /// Ambient occlusion in the red channel.
    pub occlusion_image: Option<usize>,
    /// Linear RGB light given off.
    pub emissive: [f32; 3],
    /// Light given off, scaling `emissive`.
    pub emissive_image: Option<usize>,
    pub metallic: f32,
    pub roughness: f32,
    /// Seen from both sides rather than culled from behind.
    pub double_sided: bool,
    /// Cut out where the color's alpha is below this, rather than blended (foliage, fences,
    /// hair cards); `None` blends when the alpha is below 1.
    pub alpha_cutoff: Option<f32>,
    /// Drawn in its color as it is, without lighting (glTF's `KHR_materials_unlit`).
    pub unlit: bool,
    /// How much the normal map bends the normals: its X and Y scale.
    pub normal_scale: f32,
    /// How much the occlusion map darkens, 0 to 1.
    pub occlusion_strength: f32,
    /// How each texture above is sampled.
    pub sampling: Samplings,
    /// The `KHR_materials_*` extensions the source used, if any.
    pub extensions: Option<Box<Extensions>>,
}

impl Default for Material {
    /// [`Material::plain`] in [`GRAY`], unnamed.
    fn default() -> Self {
        Self::plain("", GRAY)
    }
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
            metallic_roughness_image: None,
            occlusion_image: None,
            emissive: [0.0; 3],
            emissive_image: None,
            metallic: 0.0,
            roughness: 0.85,
            double_sided: false,
            alpha_cutoff: None,
            unlit: false,
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            sampling: Samplings::default(),
            extensions: None,
        }
    }
}

/// An image ready to embed: PNG or JPEG bytes, or a WebP image a glTF document holds, which is
/// re-encoded as PNG only when it's written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The encoded image. Shared: an image read from a document's buffer points into it, and
    /// clones point at the same bytes.
    pub bytes: crate::Bytes,
    /// `image/png` or `image/jpeg`; or `image/webp` for a WebP image [`Document::flatten`]
    /// took from a glTF document as it is, which [`to_glb`] writes as PNG and
    /// [`Image::into_png_or_jpeg`] turns into PNG.
    ///
    /// [`Document::flatten`]: crate::document::Document::flatten
    pub mime: &'static str,
}

/// A model as triangles (and lines and points) grouped by material, ready to write as glTF.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    pub prims: Vec<Prim>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    /// The joints skinned primitives are bound to, parents first; empty when nothing is skinned. Written out, a
    /// skin binds the vertices where they are, so the GLB loads in the pose the mesh was drawn in, ready to move.
    pub skeleton: Vec<Bone>,
}

impl Mesh {
    /// The number of triangles across all primitives.
    #[must_use]
    pub fn triangles(&self) -> usize {
        self.prims.iter().map(|p| p.indices.len() / 3).sum()
    }

    /// The number of line segments across all primitives.
    #[must_use]
    pub fn lines(&self) -> usize {
        self.prims.iter().map(|p| p.lines.len() / 2).sum()
    }

    /// The number of points across all primitives.
    #[must_use]
    pub fn points(&self) -> usize {
        self.prims.iter().map(|p| p.points.len()).sum()
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
        Self::from_png_or_jpeg(crate::Bytes::copy_from_slice(bytes))
    }

    /// [`Image::png_or_jpeg`], taking the bytes (a `Vec<u8>` or [`Bytes`](crate::Bytes))
    /// without a copy.
    #[must_use]
    pub fn from_png_or_jpeg(bytes: impl Into<crate::Bytes>) -> Option<Self> {
        let bytes = bytes.into();
        let mime = Self::mime_of(&bytes)?;
        Some(Self { bytes, mime })
    }

    /// `bytes` as an image: PNG and JPEG as they are, and WebP, BMP, GIF, TIFF, PSD and DDS
    /// re-encoded as PNG, each with the feature of its name. TGA has no signature, so only one
    /// with a version 2 footer is found here; [`Image::decode_named`] also goes by its name.
    #[must_use]
    pub fn decode(bytes: impl Into<crate::Bytes>) -> Option<Self> {
        Self::decode_named(bytes, std::path::Path::new(""))
    }

    /// `bytes`, a file named `name` (a path, or just its extension's worth, as `x.tga`), as an
    /// image, as [`Image::decode`] makes one; the name finds TGA files without a footer. PNG and
    /// JPEG bytes are kept as they are, without a copy.
    #[must_use]
    pub fn decode_named(bytes: impl Into<crate::Bytes>, name: &std::path::Path) -> Option<Self> {
        let bytes = bytes.into();
        if Self::mime_of(&bytes).is_some() {
            return Self::from_png_or_jpeg(bytes);
        }
        let image = Self::webp(&bytes);
        #[cfg(any(
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        ))]
        let image = image.or_else(|| crate::raster::decode(&bytes, name));
        #[cfg(not(any(
            feature = "tga",
            feature = "bmp",
            feature = "gif",
            feature = "tiff",
            feature = "psd",
            feature = "dds"
        )))]
        let _ = name;
        image
    }

    /// [`Image::decode_named`], except that a WebP image whose header reads is kept as it is,
    /// as `image/webp`: decoding and re-encoding it waits until it's written.
    pub(crate) fn kept_named(bytes: crate::Bytes, name: &std::path::Path) -> Option<Self> {
        #[cfg(feature = "webp")]
        if is_webp(&bytes) && crate::webp::header_reads(&bytes) {
            return Some(Self {
                bytes,
                mime: "image/webp",
            });
        }
        Self::decode_named(bytes, name)
    }

    /// The image as PNG or JPEG: PNG and JPEG as they are, and a WebP image (see
    /// [`Image::mime`]) re-encoded as PNG. `None` for a WebP image that doesn't decode.
    #[must_use]
    pub fn into_png_or_jpeg(self) -> Option<Self> {
        if self.mime == "image/webp" {
            Self::webp(&self.bytes)
        } else {
            Some(self)
        }
    }

    /// WebP `bytes` re-encoded as PNG, with the `webp` feature.
    pub(crate) fn webp(bytes: &[u8]) -> Option<Self> {
        if !is_webp(bytes) {
            return None;
        }
        #[cfg(feature = "webp")]
        return Some(Self {
            bytes: crate::webp::to_png(bytes)?.into(),
            mime: "image/png",
        });
        #[cfg(not(feature = "webp"))]
        None
    }

    pub(crate) fn mime_of(bytes: &[u8]) -> Option<&'static str> {
        if bytes.starts_with(b"\x89PNG") {
            Some("image/png")
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some("image/jpeg")
        } else {
            None
        }
    }
}

/// Whether `bytes` start as a WebP file does.
fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP"
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
        skeleton: Vec::new(),
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

/// A GLB with one mesh: a primitive per `Prim` (and one more for its lines, and one for its
/// points, on the same vertices), its materials, and the images embedded.
///
/// # Errors
///
/// When the mesh has nothing to draw.
pub fn to_glb(mesh: &Mesh) -> Result<Vec<u8>> {
    ensure!(mesh.prims.iter().any(|p| !p.is_empty()), "nothing to draw");
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
    if let Some(t) = <&[[f32; 3]; 4]>::try_from(corners).ok().and_then(quad) {
        out.extend(t);
        return;
    }
    // The plane's normal (Newell), then 2D coordinates along its two other axes.
    let mut nrm = [0f64; 3];
    for (i, a) in corners.iter().enumerate() {
        let b = corners[if i + 1 == n { 0 } else { i + 1 }];
        newell(&mut nrm, a.map(f64::from), b.map(f64::from));
    }
    let axis = plane_axis(nrm);
    if nrm[axis] == 0.0 || n > 4096 {
        out.extend((1..u32_of(n) - 1).map(|k| [0, k, k + 1]));
        return;
    }
    let to2 = projection(nrm, axis);
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
                // A corner on top of one of the ear's (where a hole's bridge comes back) doesn't block it.
                let bits = |v: [f64; 2]| v.map(f64::to_bits);
                if [pa, pb, pc].iter().any(|&v| bits(v) == bits(pq)) {
                    return false;
                }
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

/// A step of Newell's normal: the edge from `a` to `b`.
#[inline]
fn newell(nrm: &mut [f64; 3], a: [f64; 3], b: [f64; 3]) {
    nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
    nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
    nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
}

/// The axis a polygon with normal `nrm` faces most: the last of the largest.
#[inline]
fn plane_axis(nrm: [f64; 3]) -> usize {
    (0..3)
        .max_by(|&i, &j| nrm[i].abs().total_cmp(&nrm[j].abs()))
        .unwrap_or(2)
}

/// 2D coordinates along the two axes other than `axis`, so the polygon winds counter-clockwise.
#[inline]
fn projection(nrm: [f64; 3], axis: usize) -> impl Fn(&[f32; 3]) -> [f64; 2] {
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let flip = nrm[axis] < 0.0;
    move |c: &[f32; 3]| {
        let (x, y) = (f64::from(c[u]), f64::from(c[v]));
        if flip { [y, x] } else { [x, y] }
    }
}

/// How far `c` turns left of the line from `a` through `b`.
#[inline]
fn cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// A quad's two triangles as [`triangulate`] makes them, without allocating: fanned when
/// convex, cut along the diagonal from the one reflex corner when there is one. `None` for a
/// quad with more, which [`triangulate`] ear-clips.
#[inline]
pub(crate) fn quad(c: &[[f32; 3]; 4]) -> Option<[[u32; 3]; 2]> {
    const FAN: [[u32; 3]; 2] = [[0, 1, 2], [0, 2, 3]];
    let p = [
        c[0].map(f64::from),
        c[1].map(f64::from),
        c[2].map(f64::from),
        c[3].map(f64::from),
    ];
    let mut nrm = [0f64; 3];
    newell(&mut nrm, p[0], p[1]);
    newell(&mut nrm, p[1], p[2]);
    newell(&mut nrm, p[2], p[3]);
    newell(&mut nrm, p[3], p[0]);
    let axis = plane_axis(nrm);
    if nrm[axis] == 0.0 {
        return Some(FAN);
    }
    // As `projection` does it: swapping the axes is the flip.
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let (u, v) = if nrm[axis] < 0.0 { (v, u) } else { (u, v) };
    let q = [
        [p[0][u], p[0][v]],
        [p[1][u], p[1][v]],
        [p[2][u], p[2][v]],
        [p[3][u], p[3][v]],
    ];
    let right = [
        cross(q[3], q[0], q[1]) < 0.0,
        cross(q[0], q[1], q[2]) < 0.0,
        cross(q[1], q[2], q[3]) < 0.0,
        cross(q[2], q[3], q[0]) < 0.0,
    ];
    match right {
        [false | true, false, false, false] => Some(FAN),
        [false, true, false, false] => Some([[1, 2, 3], [1, 3, 0]]),
        [false, false, true, false] => Some([[2, 3, 0], [2, 0, 1]]),
        [false, false, false, true] => Some([[3, 0, 1], [3, 1, 2]]),
        _ => None,
    }
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

    /// Quads the long way, every step written out: Newell's normal in order, the last of the
    /// largest axes, and the corners that turn right.
    fn quad_by_hand(c: &[[f32; 3]; 4]) -> Option<Vec<[u32; 3]>> {
        let p = c.map(|c| c.map(f64::from));
        let mut nrm = [0.0f64; 3];
        for i in 0..4 {
            let (a, b) = (p[i], p[(i + 1) % 4]);
            nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
            nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
            nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        let mut axis = 0;
        for i in 1..3 {
            if nrm[i].abs().total_cmp(&nrm[axis].abs()).is_ge() {
                axis = i;
            }
        }
        if nrm[axis] == 0.0 {
            return Some(vec![[0, 1, 2], [0, 2, 3]]);
        }
        let (u, v) = [(1, 2), (2, 0), (0, 1)][axis];
        let q = p.map(|c| if nrm[axis] < 0.0 { [c[v], c[u]] } else { [c[u], c[v]] });
        let right: Vec<u32> = (0..4u32)
            .filter(|&k| {
                let (a, b, c) = (q[(k as usize + 3) % 4], q[k as usize], q[(k as usize + 1) % 4]);
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.0
            })
            .collect();
        match right[..] {
            [] => Some(vec![[0, 1, 2], [0, 2, 3]]),
            [r] => Some(vec![[r, (r + 1) % 4, (r + 2) % 4], [r, (r + 2) % 4, (r + 3) % 4]]),
            _ => None,
        }
    }

    #[test]
    fn quads_cut_like_the_rules_say() {
        let mut rng = crate::test_rng::Pcg32::new(7);
        // Mostly small grids of values, so corners repeat, line up and cancel; some wild ones.
        let pick = |rng: &mut crate::test_rng::Pcg32| match rng.below(10) {
            0 => f32::NAN,
            1 => -0.0,
            2 => rng.f32() * 2e30 - 1e30,
            3 => rng.f32() * 2e-30,
            4..=6 => rng.f32() * 2.0 - 1.0,
            _ => [-2.0, -1.0, 0.0, 1.0, 2.0][rng.below(5) as usize],
        };
        let mut t = Vec::new();
        let mut cut = 0;
        for k in 0..200_000 {
            let flat = k % 3 == 0;
            let c: [[f32; 3]; 4] = std::array::from_fn(|_| {
                let z = if flat { 0.5 } else { pick(&mut rng) };
                [pick(&mut rng), pick(&mut rng), z]
            });
            let want = quad_by_hand(&c);
            assert_eq!(super::quad(&c).map(Vec::from), want, "{c:?}");
            t.clear();
            triangulate(&c, &mut t);
            match want {
                Some(w) => {
                    cut += usize::from(w[0][0] != 0);
                    assert_eq!(t, w, "{c:?}");
                }
                // Twisted: ear-clipped, still two triangles over the four corners.
                None => assert_eq!(t.len(), 2, "{c:?}"),
            }
        }
        assert!(cut > 1000, "few quads had a reflex corner: {cut}");
    }
}

/// A multiply-rotate hasher (the Fx hash) for small integer keys: much faster than the
/// default `SipHash`, and keys here come from the file being read, not from an attacker who can
/// see the hash.
#[cfg(any(
    feature = "ply",
    feature = "3ds",
    feature = "lwo",
    feature = "ac3d",
    feature = "fbx",
    feature = "vox",
    feature = "3mf",
    feature = "off",
    feature = "x3d",
    feature = "dxf",
    feature = "md2",
    feature = "smd",
    feature = "flt",
    feature = "ogex"
))]
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

#[cfg(any(
    feature = "ply",
    feature = "3ds",
    feature = "lwo",
    feature = "ac3d",
    feature = "fbx",
    feature = "vox",
    feature = "3mf",
    feature = "off",
    feature = "x3d",
    feature = "dxf",
    feature = "md2",
    feature = "smd",
    feature = "flt",
    feature = "ogex"
))]
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
#[cfg(any(
    feature = "ply",
    feature = "3ds",
    feature = "lwo",
    feature = "ac3d",
    feature = "fbx",
    feature = "off",
    feature = "x3d",
    feature = "dxf",
    feature = "md2",
    feature = "smd",
    feature = "flt",
    feature = "ogex"
))]
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

#[cfg(all(test, feature = "tga"))]
mod image_tests {
    use super::Image;
    use std::path::Path;

    /// A 2 by 1 uncompressed true-color TGA, red then blue, with no footer.
    fn tga() -> Vec<u8> {
        let mut b = vec![0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 24, 0];
        b.extend([0, 0, 255, 255, 0, 0]);
        b
    }

    #[test]
    fn a_tga_is_found_by_its_name() {
        let image = Image::decode_named(tga(), Path::new("textures/albedo.TGA")).unwrap();
        assert_eq!(image.mime, "image/png");
        assert!(image.bytes.starts_with(b"\x89PNG"));
        // Without a name or a footer, nothing says it's a TGA.
        assert!(Image::decode(tga()).is_none());
    }
}
