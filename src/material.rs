//! A glTF material as the readers describe it: every core slot, each texture's coordinate set, sampler wrap,
//! filters and transform (`KHR_texture_transform`), and the `KHR_materials_*` extensions. [`Builder::materials`]
//! writes them into the document; [`Material`] (the flat view's material) converts into one. The extensions'
//! types are generic over their texture, so the flat view's material carries them too.
//!
//! [`Builder::materials`]: crate::document::Builder::materials

// Each reader uses its share; a build without all of them leaves some unused.
#![cfg_attr(
    not(all(
        feature = "obj",
        feature = "3ds",
        feature = "3mf",
        feature = "usd",
        feature = "fbx",
        feature = "vox"
    )),
    allow(dead_code, reason = "each reader uses its share")
)]
// Defaults are compared exactly, to leave them unwritten.
#![allow(clippy::float_cmp, reason = "defaults are compared exactly")]

use serde_json::{Map, Value, json};

use crate::mesh::{Extensions, Material, Sampling, Texture};

/// glTF's repeating wrap.
pub const REPEAT: u32 = 10497;
/// glTF's wrap that clamps to the edge.
pub const CLAMP: u32 = 33071;
/// glTF's mirrored repeating wrap.
pub const MIRROR: u32 = 33648;
/// glTF's linear filter, and its linear filter between linearly filtered mipmaps.
pub const LINEAR: u32 = 9729;
pub const LINEAR_MIPMAP_LINEAR: u32 = 9987;

/// A texture: an image, the texture coordinates it reads and how.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tex {
    /// Index into the document's images.
    pub image: usize,
    /// The `TEXCOORD_n` set.
    pub uv: u32,
    pub transform: Option<UvTransform>,
    /// S and T wrap modes.
    pub wrap: [u32; 2],
    /// A normal texture's scale or an occlusion texture's strength.
    pub scale: f32,
    /// Magnification and minification filters; `None` leaves the choice to the viewer.
    pub filter: [Option<u32>; 2],
}

impl Tex {
    pub(crate) fn new(image: usize) -> Self {
        Self {
            image,
            uv: 0,
            transform: None,
            wrap: [REPEAT; 2],
            scale: 1.0,
            filter: [Some(LINEAR), Some(LINEAR_MIPMAP_LINEAR)],
        }
    }
}

/// `KHR_texture_transform`: `uv' = translate(offset) * rotate(rotation) * scale(scale) * uv`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    pub offset: [f32; 2],
    /// Radians, counter-clockwise.
    pub rotation: f32,
    pub scale: [f32; 2],
}

impl UvTransform {
    /// No change.
    pub const IDENTITY: Self = Self {
        offset: [0.0; 2],
        rotation: 0.0,
        scale: [1.0; 2],
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Alpha {
    Opaque,
    /// Cut out below this alpha.
    Mask(f32),
    Blend,
}

/// A factor and the texture that scales it. `T` is the texture: a [`Texture`] in the flat view's material.
#[derive(Clone, Debug, PartialEq)]
pub struct Factor<T> {
    pub factor: f32,
    pub tex: Option<T>,
}

impl<T> Factor<T> {
    /// The factor, without a texture.
    #[must_use]
    pub fn of(factor: f32) -> Self {
        Self { factor, tex: None }
    }

    fn map<U>(&self, f: &mut impl FnMut(&T) -> Option<U>) -> Factor<U> {
        Factor {
            factor: self.factor,
            tex: self.tex.as_ref().and_then(f),
        }
    }
}

/// `KHR_materials_specular`: the strength and color of the reflection at normal incidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Specular<T> {
    /// Its texture's alpha scales it.
    pub factor: Factor<T>,
    /// Linear RGB.
    pub color: [f32; 3],
    /// sRGB, scaling `color`.
    pub color_tex: Option<T>,
}

/// `KHR_materials_clearcoat`: a clear lacquer layer over the material.
#[derive(Clone, Debug, PartialEq)]
pub struct Clearcoat<T> {
    /// Its texture's red scales it.
    pub factor: Factor<T>,
    /// Its texture's green scales it.
    pub roughness: Factor<T>,
    /// The layer's own normal map.
    pub normal: Option<T>,
}

/// `KHR_materials_sheen`: the soft back-scattering of cloth.
#[derive(Clone, Debug, PartialEq)]
pub struct Sheen<T> {
    /// Linear RGB.
    pub color: [f32; 3],
    /// sRGB, scaling `color`.
    pub color_tex: Option<T>,
    /// Its texture's alpha scales it.
    pub roughness: Factor<T>,
}

/// `KHR_materials_volume`: the thickness of a transmissive material, and how it tints light passing through.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume<T> {
    /// In the mesh's units; its texture's green scales it.
    pub thickness: Factor<T>,
    /// How far light goes before it is tinted `attenuation_color`; `None` is forever.
    pub attenuation_distance: Option<f32>,
    pub attenuation_color: [f32; 3],
}

/// `KHR_materials_anisotropy`: highlights stretched along a direction (brushed metal, hair).
#[derive(Clone, Debug, PartialEq)]
pub struct Anisotropy<T> {
    /// The texture holds a direction in red and green, and a strength in blue.
    pub strength: Factor<T>,
    /// Radians, counter-clockwise from the tangent.
    pub rotation: f32,
}

/// `KHR_materials_iridescence`: the color shift of a thin film (soap bubbles, oil).
#[derive(Clone, Debug, PartialEq)]
pub struct Iridescence<T> {
    /// Its texture's red scales it.
    pub factor: Factor<T>,
    pub ior: f32,
    /// The thin film's thickness range in nanometers.
    pub thickness: [f32; 2],
    /// Green picks a thickness in that range.
    pub thickness_tex: Option<T>,
}

/// A glTF material.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Full {
    pub name: String,
    /// Linear RGBA.
    pub base_color: [f32; 4],
    pub base_color_tex: Option<Tex>,
    pub metallic: f32,
    pub roughness: f32,
    /// Roughness in green, metalness in blue.
    pub metallic_roughness_tex: Option<Tex>,
    pub normal_tex: Option<Tex>,
    /// Occlusion in red.
    pub occlusion_tex: Option<Tex>,
    /// Linear RGB, brighter than one when it must be (`KHR_materials_emissive_strength`).
    pub emissive: [f32; 3],
    pub emissive_tex: Option<Tex>,
    pub alpha: Alpha,
    pub double_sided: bool,
    /// `KHR_materials_unlit`.
    pub unlit: bool,
    /// `KHR_materials_ior`.
    pub ior: Option<f32>,
    pub specular: Option<Specular<Tex>>,
    /// `KHR_materials_transmission`.
    pub transmission: Option<Factor<Tex>>,
    pub clearcoat: Option<Clearcoat<Tex>>,
    pub sheen: Option<Sheen<Tex>>,
    pub volume: Option<Volume<Tex>>,
    pub anisotropy: Option<Anisotropy<Tex>>,
    pub iridescence: Option<Iridescence<Tex>>,
    /// `KHR_materials_dispersion`.
    pub dispersion: Option<f32>,
    pub extras: Option<Value>,
}

impl Default for Full {
    /// glTF's default material.
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            base_color_tex: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_tex: None,
            normal_tex: None,
            occlusion_tex: None,
            emissive: [0.0; 3],
            emissive_tex: None,
            alpha: Alpha::Opaque,
            double_sided: false,
            unlit: false,
            ior: None,
            specular: None,
            transmission: None,
            clearcoat: None,
            sheen: None,
            volume: None,
            anisotropy: None,
            iridescence: None,
            dispersion: None,
            extras: None,
        }
    }
}

impl Full {
    /// The glTF material for a flat view's material, its textures reading the coordinates `texcoord` numbers:
    /// `texcoord(None)` for [`Prim::uvs`](crate::mesh::Prim::uvs), `texcoord(Some(n))` for `uv_sets[n]`. A
    /// texture whose coordinates aren't there (`None`) is left out.
    pub(crate) fn of(m: &Material, texcoord: impl Fn(Option<usize>) -> Option<u32>) -> Self {
        let tex = |image: usize, s: &Sampling, scale: f32| {
            Some(Tex {
                uv: texcoord(s.uv_set)?,
                transform: s.transform,
                wrap: s.wrap,
                scale,
                filter: s.filter,
                ..Tex::new(image)
            })
        };
        let slot = |image: Option<usize>, s: &Sampling, scale: f32| image.and_then(|i| tex(i, s, scale));
        let mut t = |t: &Texture| tex(t.image, &t.sampling, 1.0);
        let s = &m.sampling;
        let mut full = Self {
            name: m.name.clone(),
            base_color: m.color,
            base_color_tex: slot(m.color_image, &s.color, 1.0),
            metallic: m.metallic,
            roughness: m.roughness,
            metallic_roughness_tex: slot(m.metallic_roughness_image, &s.metallic_roughness, 1.0),
            normal_tex: slot(m.normal_image, &s.normal, m.normal_scale),
            occlusion_tex: slot(m.occlusion_image, &s.occlusion, m.occlusion_strength),
            emissive: m.emissive,
            emissive_tex: slot(m.emissive_image, &s.emissive, 1.0),
            alpha: match m.alpha_cutoff {
                Some(c) => Alpha::Mask(c),
                None if m.color[3] < 1.0 => Alpha::Blend,
                None => Alpha::Opaque,
            },
            double_sided: m.double_sided,
            unlit: m.unlit,
            ..Self::default()
        };
        if let Some(x) = m.extensions.as_deref() {
            let Extensions {
                ior,
                specular,
                transmission,
                clearcoat,
                sheen,
                volume,
                anisotropy,
                iridescence,
                dispersion,
            } = x;
            full.ior = *ior;
            full.specular = specular.as_ref().map(|s| Specular {
                factor: s.factor.map(&mut t),
                color: s.color,
                color_tex: s.color_tex.as_ref().and_then(&mut t),
            });
            full.transmission = transmission.as_ref().map(|f| f.map(&mut t));
            full.clearcoat = clearcoat.as_ref().map(|c| Clearcoat {
                factor: c.factor.map(&mut t),
                roughness: c.roughness.map(&mut t),
                normal: c.normal.as_ref().and_then(&mut t),
            });
            full.sheen = sheen.as_ref().map(|s| Sheen {
                color: s.color,
                color_tex: s.color_tex.as_ref().and_then(&mut t),
                roughness: s.roughness.map(&mut t),
            });
            full.volume = volume.as_ref().map(|v| Volume {
                thickness: v.thickness.map(&mut t),
                attenuation_distance: v.attenuation_distance,
                attenuation_color: v.attenuation_color,
            });
            full.anisotropy = anisotropy.as_ref().map(|a| Anisotropy {
                strength: a.strength.map(&mut t),
                rotation: a.rotation,
            });
            full.iridescence = iridescence.as_ref().map(|i| Iridescence {
                factor: i.factor.map(&mut t),
                ior: i.ior,
                thickness: i.thickness,
                thickness_tex: i.thickness_tex.as_ref().and_then(&mut t),
            });
            full.dispersion = *dispersion;
        }
        full
    }
}

impl From<&Material> for Full {
    /// The glTF material, its textures reading `TEXCOORD_0`; one reading another of a primitive's sets
    /// ([`Sampling::uv_set`]) is left out, as which number that set gets depends on the primitive.
    fn from(m: &Material) -> Self {
        Self::of(m, |set| set.is_none().then_some(0))
    }
}

impl From<Material> for Full {
    fn from(m: Material) -> Self {
        Self::from(&m)
    }
}

impl From<&Full> for Full {
    fn from(m: &Full) -> Self {
        m.clone()
    }
}

fn unit(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

fn positive(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

/// A sampler: S and T wrap, magnification and minification filters.
type Sampler = ([u32; 2], [Option<u32>; 2]);

/// Writes materials: the textures they name (one per image and sampler), their samplers, and the extensions used.
pub(crate) struct Writer {
    /// Each image's index among those written: `None` for one left out.
    images: Vec<Option<usize>>,
    /// (image, sampler) → texture, in order.
    textures: Vec<(usize, Sampler)>,
    used: Vec<&'static str>,
}

impl Writer {
    /// A writer over images written at the indices `images` gives (`None` for one left out, whose textures go
    /// too). Textures are made as materials use them, one for each image and sampler.
    pub(crate) fn new(images: Vec<Option<usize>>) -> Self {
        Self {
            images,
            textures: Vec::new(),
            used: Vec::new(),
        }
    }

    fn uses(&mut self, ext: &'static str) {
        if !self.used.contains(&ext) {
            self.used.push(ext);
        }
    }

    /// A texture info, or `None` when its image isn't there.
    fn tex(&mut self, t: Option<&Tex>, scale_key: Option<&str>) -> Option<Value> {
        let t = t?;
        let image = (*self.images.get(t.image)?)?;
        let key = (image, (t.wrap, t.filter));
        let index = self.textures.iter().position(|&k| k == key).unwrap_or_else(|| {
            self.textures.push(key);
            self.textures.len() - 1
        });
        let mut info = json!({ "index": index });
        if t.uv != 0 {
            info["texCoord"] = json!(t.uv);
        }
        #[allow(clippy::float_cmp, reason = "exactly one is the default")]
        if let Some(k) = scale_key
            && t.scale != 1.0
            && t.scale.is_finite()
        {
            info[k] = json!(if k == "strength" { unit(t.scale) } else { t.scale });
        }
        if let Some(x) = t.transform.filter(|x| *x != UvTransform::IDENTITY) {
            let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
            let mut tt = Map::new();
            if x.offset != [0.0; 2] {
                tt.insert("offset".into(), json!(x.offset.map(finite)));
            }
            if x.rotation != 0.0 {
                tt.insert("rotation".into(), json!(finite(x.rotation)));
            }
            if x.scale != [1.0; 2] {
                tt.insert("scale".into(), json!(x.scale.map(finite)));
            }
            info["extensions"] = json!({ "KHR_texture_transform": tt });
            self.uses("KHR_texture_transform");
        }
        Some(info)
    }

    fn factor(&mut self, out: &mut Map<String, Value>, f: &Factor<Tex>, key: &str, tex_key: &str, default: f32) {
        #[allow(clippy::float_cmp, reason = "exactly the default is left out")]
        if unit(f.factor) != default {
            out.insert(key.into(), json!(unit(f.factor)));
        }
        if let Some(t) = self.tex(f.tex.as_ref(), None) {
            out.insert(tex_key.into(), t);
        }
    }

    #[allow(clippy::too_many_lines, reason = "one field after another")]
    pub(crate) fn material(&mut self, m: &Full) -> Value {
        let mut pbr = Map::new();
        pbr.insert("baseColorFactor".into(), json!(m.base_color.map(unit)));
        pbr.insert("metallicFactor".into(), json!(unit(m.metallic)));
        pbr.insert("roughnessFactor".into(), json!(unit(m.roughness)));
        if let Some(t) = self.tex(m.base_color_tex.as_ref(), None) {
            pbr.insert("baseColorTexture".into(), t);
        }
        if let Some(t) = self.tex(m.metallic_roughness_tex.as_ref(), None) {
            pbr.insert("metallicRoughnessTexture".into(), t);
        }
        let mut mat = json!({ "name": m.name, "pbrMetallicRoughness": pbr });
        if let Some(t) = self.tex(m.normal_tex.as_ref(), Some("scale")) {
            mat["normalTexture"] = t;
        }
        if let Some(t) = self.tex(m.occlusion_tex.as_ref(), Some("strength")) {
            mat["occlusionTexture"] = t;
        }
        let mut ext = Map::new();
        let emissive = m.emissive.map(positive);
        if emissive.iter().any(|&e| e > 0.0) {
            // Brighter than one: the factor scaled down, and the strength that scales it back.
            let strength = emissive.iter().copied().fold(1.0, f32::max);
            mat["emissiveFactor"] = json!(emissive.map(|e| unit(e / strength)));
            if strength > 1.0 && strength.is_finite() {
                ext.insert(
                    "KHR_materials_emissive_strength".into(),
                    json!({ "emissiveStrength": strength }),
                );
            }
        }
        if let Some(t) = self.tex(m.emissive_tex.as_ref(), None) {
            mat["emissiveTexture"] = t;
            if emissive.iter().all(|&e| e <= 0.0) {
                // A texture alone glows as painted.
                mat["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
            }
        }
        match m.alpha {
            Alpha::Opaque => {}
            Alpha::Mask(cutoff) => {
                mat["alphaMode"] = json!("MASK");
                #[allow(clippy::float_cmp, reason = "exactly the default is left out")]
                if positive(cutoff) != 0.5 {
                    mat["alphaCutoff"] = json!(positive(cutoff));
                }
            }
            Alpha::Blend => mat["alphaMode"] = json!("BLEND"),
        }
        if m.double_sided {
            mat["doubleSided"] = json!(true);
        }
        if m.unlit {
            ext.insert("KHR_materials_unlit".into(), json!({}));
        }
        if let Some(ior) = m.ior.filter(|i| i.is_finite() && (*i == 0.0 || *i >= 1.0)) {
            ext.insert("KHR_materials_ior".into(), json!({ "ior": ior }));
        }
        if let Some(s) = &m.specular {
            let mut o = Map::new();
            self.factor(&mut o, &s.factor, "specularFactor", "specularTexture", 1.0);
            let color = s.color.map(positive);
            if color != [1.0; 3] {
                o.insert("specularColorFactor".into(), json!(color));
            }
            if let Some(t) = self.tex(s.color_tex.as_ref(), None) {
                o.insert("specularColorTexture".into(), t);
            }
            ext.insert("KHR_materials_specular".into(), Value::Object(o));
        }
        if let Some(t) = &m.transmission {
            let mut o = Map::new();
            self.factor(&mut o, t, "transmissionFactor", "transmissionTexture", 0.0);
            ext.insert("KHR_materials_transmission".into(), Value::Object(o));
        }
        if let Some(c) = &m.clearcoat {
            let mut o = Map::new();
            self.factor(&mut o, &c.factor, "clearcoatFactor", "clearcoatTexture", 0.0);
            self.factor(
                &mut o,
                &c.roughness,
                "clearcoatRoughnessFactor",
                "clearcoatRoughnessTexture",
                0.0,
            );
            if let Some(t) = self.tex(c.normal.as_ref(), Some("scale")) {
                o.insert("clearcoatNormalTexture".into(), t);
            }
            ext.insert("KHR_materials_clearcoat".into(), Value::Object(o));
        }
        if let Some(s) = &m.sheen {
            let mut o = Map::new();
            let color = s.color.map(unit);
            if color != [0.0; 3] {
                o.insert("sheenColorFactor".into(), json!(color));
            }
            if let Some(t) = self.tex(s.color_tex.as_ref(), None) {
                o.insert("sheenColorTexture".into(), t);
            }
            self.factor(
                &mut o,
                &s.roughness,
                "sheenRoughnessFactor",
                "sheenRoughnessTexture",
                0.0,
            );
            ext.insert("KHR_materials_sheen".into(), Value::Object(o));
        }
        if let Some(v) = &m.volume {
            let mut o = Map::new();
            let thickness = positive(v.thickness.factor);
            if thickness != 0.0 {
                o.insert("thicknessFactor".into(), json!(thickness));
            }
            if let Some(t) = self.tex(v.thickness.tex.as_ref(), None) {
                o.insert("thicknessTexture".into(), t);
            }
            if let Some(d) = v.attenuation_distance.filter(|d| d.is_finite() && *d > 0.0) {
                o.insert("attenuationDistance".into(), json!(d));
            }
            let color = v.attenuation_color.map(unit);
            if color != [1.0; 3] {
                o.insert("attenuationColor".into(), json!(color));
            }
            ext.insert("KHR_materials_volume".into(), Value::Object(o));
        }
        if let Some(a) = &m.anisotropy {
            let mut o = Map::new();
            self.factor(&mut o, &a.strength, "anisotropyStrength", "anisotropyTexture", 0.0);
            if a.rotation != 0.0 && a.rotation.is_finite() {
                o.insert("anisotropyRotation".into(), json!(a.rotation));
            }
            ext.insert("KHR_materials_anisotropy".into(), Value::Object(o));
        }
        if let Some(i) = &m.iridescence {
            let mut o = Map::new();
            self.factor(&mut o, &i.factor, "iridescenceFactor", "iridescenceTexture", 0.0);
            #[allow(clippy::float_cmp, reason = "exactly the default is left out")]
            if i.ior != 1.3 && i.ior.is_finite() && i.ior >= 1.0 {
                o.insert("iridescenceIor".into(), json!(i.ior));
            }
            let [lo, hi] = i.thickness.map(positive);
            #[allow(clippy::float_cmp, reason = "exactly the default is left out")]
            if lo != 100.0 {
                o.insert("iridescenceThicknessMinimum".into(), json!(lo));
            }
            #[allow(clippy::float_cmp, reason = "exactly the default is left out")]
            if hi != 400.0 {
                o.insert("iridescenceThicknessMaximum".into(), json!(hi));
            }
            if let Some(t) = self.tex(i.thickness_tex.as_ref(), None) {
                o.insert("iridescenceThicknessTexture".into(), t);
            }
            ext.insert("KHR_materials_iridescence".into(), Value::Object(o));
        }
        if let Some(d) = m.dispersion.map(positive).filter(|&d| d > 0.0) {
            ext.insert("KHR_materials_dispersion".into(), json!({ "dispersion": d }));
        }
        for name in ext.keys() {
            let known = [
                "KHR_materials_emissive_strength",
                "KHR_materials_unlit",
                "KHR_materials_ior",
                "KHR_materials_specular",
                "KHR_materials_transmission",
                "KHR_materials_clearcoat",
                "KHR_materials_sheen",
                "KHR_materials_volume",
                "KHR_materials_anisotropy",
                "KHR_materials_iridescence",
                "KHR_materials_dispersion",
            ];
            if let Some(&k) = known.iter().find(|&&k| k == name)
                && !self.used.contains(&k)
            {
                self.used.push(k);
            }
        }
        if !ext.is_empty() {
            mat["extensions"] = Value::Object(ext);
        }
        if let Some(e) = &m.extras {
            mat["extras"] = e.clone();
        }
        mat
    }

    /// The textures (each `{sampler, source}`), the samplers they use, and the extensions the materials used.
    pub(crate) fn finish(self) -> (Vec<Value>, Vec<Value>, Vec<&'static str>) {
        let mut samplers: Vec<Sampler> = Vec::new();
        let textures = self
            .textures
            .iter()
            .map(|&(image, sampler)| {
                let s = samplers.iter().position(|&w| w == sampler).unwrap_or_else(|| {
                    samplers.push(sampler);
                    samplers.len() - 1
                });
                json!({ "sampler": s, "source": image })
            })
            .collect();
        let samplers = samplers
            .iter()
            .map(|(w, [mag, min])| {
                let mut s = json!({ "wrapS": w[0], "wrapT": w[1] });
                // Only glTF's filter values: anything else is left to the viewer.
                if let Some(f) = mag.filter(|f| matches!(f, 9728 | 9729)) {
                    s["magFilter"] = json!(f);
                }
                if let Some(f) = min.filter(|f| matches!(f, 9728 | 9729 | 9984..=9987)) {
                    s["minFilter"] = json!(f);
                }
                s
            })
            .collect();
        (textures, samplers, self.used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_material_writes_only_the_core() {
        let mut w = Writer::new(Vec::new());
        let m = w.material(&Full::default());
        assert!(m.get("extensions").is_none());
        assert_eq!(m["pbrMetallicRoughness"]["metallicFactor"], json!(1.0));
        let (textures, samplers, used) = w.finish();
        assert!(textures.is_empty() && samplers.is_empty() && used.is_empty());
    }

    #[test]
    fn bright_emission_takes_a_strength() {
        let mut w = Writer::new(Vec::new());
        let m = w.material(&Full {
            emissive: [4.0, 2.0, 0.0],
            ..Full::default()
        });
        assert_eq!(m["emissiveFactor"], json!([1.0, 0.5, 0.0]));
        assert_eq!(
            m["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"],
            json!(4.0)
        );
        assert_eq!(w.finish().2, ["KHR_materials_emissive_strength"]);
    }

    #[test]
    fn textures_are_shared_per_image_and_wrap() {
        let mut w = Writer::new(vec![Some(0), Some(1)]);
        let clamped = Tex {
            wrap: [CLAMP; 2],
            transform: Some(UvTransform {
                scale: [2.0, 2.0],
                ..UvTransform::IDENTITY
            }),
            ..Tex::new(0)
        };
        let m = w.material(&Full {
            base_color_tex: Some(Tex::new(0)),
            emissive_tex: Some(clamped),
            normal_tex: Some(Tex::new(0)),
            // An image that isn't there is left out.
            occlusion_tex: Some(Tex::new(5)),
            ..Full::default()
        });
        assert_eq!(m["pbrMetallicRoughness"]["baseColorTexture"]["index"], json!(0));
        assert_eq!(m["normalTexture"]["index"], json!(0));
        assert_eq!(m["emissiveTexture"]["index"], json!(1));
        assert_eq!(
            m["emissiveTexture"]["extensions"]["KHR_texture_transform"],
            json!({ "scale": [2.0, 2.0] })
        );
        // A texture alone glows as painted.
        assert_eq!(m["emissiveFactor"], json!([1.0, 1.0, 1.0]));
        assert!(m.get("occlusionTexture").is_none());
        let (textures, samplers, used) = w.finish();
        assert_eq!(textures.len(), 2);
        assert_eq!(samplers.len(), 2);
        assert_eq!(samplers[1]["wrapS"], json!(CLAMP));
        assert_eq!(used, ["KHR_texture_transform"]);
    }

    #[test]
    fn out_of_range_values_are_clamped_or_dropped() {
        let mut w = Writer::new(Vec::new());
        let m = w.material(&Full {
            metallic: 2.0,
            roughness: -1.0,
            ior: Some(0.5),
            transmission: Some(Factor::of(1.5)),
            ..Full::default()
        });
        assert_eq!(m["pbrMetallicRoughness"]["metallicFactor"], json!(1.0));
        assert_eq!(m["pbrMetallicRoughness"]["roughnessFactor"], json!(0.0));
        assert!(m["extensions"].get("KHR_materials_ior").is_none());
        assert_eq!(
            m["extensions"]["KHR_materials_transmission"]["transmissionFactor"],
            json!(1.0)
        );
    }
}
