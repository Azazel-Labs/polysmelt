//! Materials (classic, PBR shaders and `glTF|` properties) and their textures, embedded or found in the pack.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use super::{Objects, Props, Reader, child_attr, num, scalar, string, vec3};
use crate::assets::Assets;
use crate::formats::fbx_tree::Attr as A;
use crate::images::{Images, Location};
use crate::legacy_material::{self, Phong, Rgb, Transparency};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Sheen, Tex, UvTransform};
use crate::mesh::{GRAY, Material, srgb_to_linear};
use crate::num::f32_of;

impl Reader<'_, '_, '_> {
    /// Model `m`'s materials, in connection order: a geometry's material layer indexes into this list.
    pub(super) fn model_materials(&mut self, m: i64) -> Vec<usize> {
        let ids: Vec<i64> = self
            .objs
            .children_of(m)
            .filter(|c| self.objs.is("Material", *c))
            .collect();
        if !self.keep.materials {
            return Vec::new();
        }
        ids.into_iter()
            .map(|mid| {
                if let Some(&i) = self.material_index.get(&mid) {
                    return i;
                }
                let (mat, color_set, sets) = read_material(
                    mid,
                    self.objs,
                    self.linear,
                    self.dir,
                    self.keep.textures.then_some(self.assets),
                    &mut self.images,
                    &mut self.image_index,
                );
                let i = self.materials.len();
                self.materials.push(mat);
                self.material_index.insert(mid, i);
                if let Some(set) = color_set {
                    self.color_uv.insert(i, set);
                }
                if !sets.is_empty() {
                    self.material_uvs.insert(i, sets);
                }
                i
            })
            .collect()
    }

    pub(super) fn default_material(&mut self) -> usize {
        *self.default_material.get_or_insert_with(|| {
            self.materials.push(legacy_material::default_material(false).into());
            self.materials.len() - 1
        })
    }

    /// An unlit material for lines of sRGB `color`, one for each color.
    pub(super) fn line_material(&mut self, color: [f64; 3]) -> usize {
        let rgb = color.map(|c| srgb_to_linear(f32_of(c).clamp(0.0, 1.0)));
        let key = rgb.map(f32::to_bits);
        if let Some(&i) = self.line_materials.get(&key) {
            return i;
        }
        let mut m = Full::from(&Material::plain("", GRAY));
        m.unlit = true;
        m.metallic = 0.0;
        m.base_color = [rgb[0], rgb[1], rgb[2], 1.0];
        self.materials.push(m);
        self.line_materials.insert(key, self.materials.len() - 1);
        self.materials.len() - 1
    }
}

/// The material slot a texture connects to, from the property it connects as: the classic slots (`DiffuseColor`,
/// `NormalMap`, ...) and PBR shaders' (`Maya|TEX_color_map`, `Maya|baseColor`, `3dsMax|main|base_color_map`,
/// `3dsMax|Parameters|base_color_map`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Color,
    Normal,
    Emissive,
    Occlusion,
    Metallic,
    Roughness,
    /// Roughness in green and metalness in blue, glTF's own arrangement.
    MetalRough,
    /// Occlusion, roughness and metalness in red, green and blue.
    OcclusionMetalRough,
    Opacity,
    Transmission,
    Coat,
    CoatRoughness,
    CoatNormal,
    SheenColor,
    SheenRoughness,
}

fn slot(prop: &str) -> Option<Slot> {
    let s: String = prop
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| s.contains(w));
    Some(if has(&["occlusionroughnessmetal", "aoroughnessmetal"]) {
        Slot::OcclusionMetalRough
    } else if has(&["metallicroughness", "metalroughness", "roughnessmetal"]) {
        Slot::MetalRough
    } else if has(&["coatnormal", "coatbump"]) {
        Slot::CoatNormal
    } else if has(&["coatroughness"]) {
        Slot::CoatRoughness
    } else if has(&["sheenroughness", "fuzzroughness"]) {
        Slot::SheenRoughness
    } else if has(&["sheencolor", "fuzzcolor"]) {
        Slot::SheenColor
    } else if has(&["coatcolor", "coatior", "coataffect", "transmissioncolor", "transcolor"]) {
        return None;
    } else if s.ends_with("coat") || s.ends_with("coatweight") || has(&["coatmap", "coatweightmap"]) {
        Slot::Coat
    } else if s.ends_with("transmission") || has(&["transmissionweight", "transweight", "transmissionmap"]) {
        Slot::Transmission
    } else if has(&["emissive", "emitcolor", "emission"]) {
        Slot::Emissive
    } else if s.contains("normal") || s == "bump" || s.ends_with("bumpmap") || s.ends_with("normmap") {
        Slot::Normal
    } else if has(&["aomap", "ambientocclusion"]) {
        Slot::Occlusion
    } else if has(&["metallic", "metalness"]) {
        Slot::Metallic
    } else if s.contains("roughness") {
        Slot::Roughness
    } else if has(&["transparentcolor", "transparencyfactor", "opacity", "cutout"]) {
        Slot::Opacity
    } else if s.contains("diffusecolor") || s == "diffuse" || s.contains("basecolor") || s.ends_with("texcolormap") {
        Slot::Color
    } else {
        return None;
    })
}

/// A PBR shader's parameters, as tools namespace them on a material (`Maya|baseColor`, `3dsMax|main|basecolor`,
/// `3dsMax|Parameters|base_color`, ...), found by their last part's name without case or underscores.
struct Pbr<'p> {
    by: HashMap<String, &'p [A<'p>]>,
}

impl<'p> Pbr<'p> {
    fn of(p: &'p Props<'_>) -> Self {
        let by = p
            .own
            .iter()
            .filter_map(|(k, v)| {
                let (_, last) = k.rsplit_once('|')?;
                let name: String = last
                    .chars()
                    .filter(char::is_ascii_alphanumeric)
                    .collect::<String>()
                    .to_lowercase();
                Some((name, v.as_slice()))
            })
            .collect();
        Self { by }
    }

    fn get(&self, names: &[&str]) -> Option<&'p [A<'p>]> {
        names.iter().find_map(|n| self.by.get(*n).copied())
    }

    /// A number: the first of the names given that's there and finite.
    fn f(&self, names: &[&str]) -> Option<f32> {
        names
            .iter()
            .find_map(|n| self.by.get(*n)?.first().and_then(num).filter(|v| v.is_finite()))
            .map(f32_of)
    }

    /// A color (or a number, as a gray).
    fn color(&self, names: &[&str]) -> Option<[f64; 3]> {
        let v = self.get(names)?;
        let c: Vec<f64> = v.iter().filter_map(num).filter(|x| x.is_finite()).collect();
        match c.as_slice() {
            [r, g, b, ..] => Some([*r, *g, *b]),
            [x] => Some([*x; 3]),
            _ => None,
        }
    }
}

/// A material's PBR shader, when it has one: standard surfaces, physical materials, metal/rough and spec/gloss
/// PBR materials, glTF materials and `OpenPBR`. They name their parameters alike, so one reading covers them;
/// each name list holds the synonyms the shaders use. Gives the base color's weight.
fn read_pbr(mat: &mut Full, p: &Props<'_>, to_linear: &dyn Fn([f64; 3]) -> [f32; 3]) -> Option<f32> {
    let pbr = Pbr::of(p);
    let base = pbr.color(&["basecolor", "basecolour"])?;
    let weight = pbr.f(&["baseweight", "base"]).unwrap_or(1.0).max(0.0);
    let c = to_linear(base);
    mat.base_color[..3].copy_from_slice(&c.map(|c| c * weight));
    // Spec/gloss has no metalness: its specular color stands for it.
    mat.metallic = pbr
        .f(&["metalness", "metallic", "basemetalness"])
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let glossy = pbr.f(&["roughnessinv", "useglossiness"]).is_some_and(|i| i > 0.5);
    mat.roughness = match (pbr.f(&["roughness", "specularroughness"]), pbr.f(&["glossiness"])) {
        (Some(r), _) if glossy => 1.0 - r,
        (Some(r), _) => r,
        (None, Some(g)) => 1.0 - g,
        (None, None) => 0.5,
    }
    .clamp(0.0, 1.0);
    let emit = pbr
        .color(&["emissive", "emissioncolor", "emitcolor"])
        .map_or([0.0; 3], to_linear);
    let strength = pbr
        .f(&["emissiveintensity", "emission", "emissionweight", "emissionluminance"])
        .unwrap_or(1.0)
        .max(0.0);
    mat.emissive = emit.map(|c| c * strength);
    if let Some(t) = pbr
        .f(&["transmission", "transmissionweight", "transweight"])
        .filter(|t| *t > 0.0)
    {
        mat.transmission = Some(Factor::of(t.min(1.0)));
    }
    mat.ior = pbr
        .f(&["specularior", "ior", "transior", "trans_ior"])
        .filter(|i| *i >= 1.0);
    if let Some(c) = pbr.f(&["coat", "coatweight"]).filter(|c| *c > 0.0) {
        mat.clearcoat = Some(Clearcoat {
            factor: Factor::of(c.min(1.0)),
            roughness: Factor::of(pbr.f(&["coatroughness"]).unwrap_or(0.0).clamp(0.0, 1.0)),
            normal: None,
        });
    }
    if let Some(s) = pbr.f(&["sheen", "sheenweight", "fuzzweight"]).filter(|s| *s > 0.0) {
        let color = pbr.color(&["sheencolor", "fuzzcolor"]).map_or([1.0; 3], to_linear);
        mat.sheen = Some(Sheen {
            color: color.map(|c| c * s),
            color_tex: None,
            roughness: Factor::of(
                pbr.f(&["sheenroughness", "fuzzroughness"])
                    .unwrap_or(0.3)
                    .clamp(0.0, 1.0),
            ),
        });
    }
    // A specular color without metalness (spec/gloss): the tint of its reflections.
    if pbr.f(&["metalness", "metallic", "basemetalness"]).is_none()
        && let Some(s) = pbr.color(&["specular", "specularcolor"])
    {
        mat.specular = Some(material::Specular {
            factor: Factor::of(1.0),
            color: to_linear(s),
            color_tex: None,
        });
    }
    if let Some(o) = pbr.color(&["opacity", "geometryopacity"]) {
        let a = f32_of((o[0] + o[1] + o[2]) / 3.0).clamp(0.0, 1.0);
        mat.base_color[3] = a;
        if a < 1.0 {
            mat.alpha = Alpha::Blend;
        }
    }
    if pbr.f(&["doublesided"]).is_some_and(|d| d > 0.5) {
        mat.double_sided = true;
    }
    if pbr.f(&["unlit"]).is_some_and(|u| u > 0.5) {
        mat.unlit = true;
    }
    Some(weight)
}

/// A material's glTF parameters, kept under `glTF|` properties as glTF holds them (linear colors), which take the
/// place of what the classic and PBR properties gave. Gives whether the material has them.
fn read_gltf(mat: &mut Full, p: &Props<'_>) -> bool {
    if !p.own.keys().any(|k| k.starts_with("glTF|")) {
        return false;
    }
    let f = |k: &str| p.own_scalar(&format!("glTF|{k}")).map(f32_of);
    let c = |k: &str| {
        p.own.get(&format!("glTF|{k}")).and_then(|v| {
            let c: Vec<f32> = v.iter().filter_map(num).filter(|x| x.is_finite()).map(f32_of).collect();
            (c.len() >= 3).then(|| [c[0], c[1], c[2]])
        })
    };
    if let Some(b) = c("baseColor") {
        mat.base_color[..3].copy_from_slice(&b);
    }
    if let Some(a) = f("opacity") {
        mat.base_color[3] = a;
    }
    mat.metallic = f("metalness").unwrap_or(mat.metallic);
    mat.roughness = f("roughness").unwrap_or(mat.roughness);
    if let Some(e) = c("emissive") {
        let k = f("emissiveIntensity").unwrap_or(1.0);
        mat.emissive = e.map(|e| e * k);
    }
    mat.alpha = match f("alphaMode") {
        Some(m) if (m - 1.0).abs() < 0.5 => Alpha::Mask(f("alphaCutoff").unwrap_or(0.5)),
        Some(m) if (m - 2.0).abs() < 0.5 => Alpha::Blend,
        _ => Alpha::Opaque,
    };
    mat.double_sided = f("doubleSided").is_some_and(|d| d != 0.0);
    mat.unlit = f("unlit").is_some_and(|u| u != 0.0);
    mat.ior = f("ior");
    mat.transmission = f("transmission").map(Factor::of);
    mat.clearcoat = f("coat").map(|c| Clearcoat {
        factor: Factor::of(c),
        roughness: Factor::of(f("coatRoughness").unwrap_or(0.0)),
        normal: None,
    });
    mat.sheen = c("sheenColor").map(|color| Sheen {
        color,
        color_tex: None,
        roughness: Factor::of(f("sheenRoughness").unwrap_or(0.0)),
    });
    mat.specular = f("specular").map(|s| material::Specular {
        factor: Factor::of(s),
        color: c("specularColor").unwrap_or([1.0; 3]),
        color_tex: None,
    });
    mat.volume = f("thickness").map(|t| material::Volume {
        thickness: Factor::of(t),
        attenuation_distance: f("attenuationDistance"),
        attenuation_color: c("attenuationColor").unwrap_or([1.0; 3]),
    });
    mat.anisotropy = f("anisotropy").map(|a| material::Anisotropy {
        strength: Factor::of(a),
        rotation: f("anisotropyRotation").unwrap_or(0.0),
    });
    mat.iridescence = f("iridescence").map(|i| material::Iridescence {
        factor: Factor::of(i),
        ior: f("iridescenceIor").unwrap_or(1.3),
        thickness: [
            f("iridescenceThicknessMin").unwrap_or(100.0),
            f("iridescenceThicknessMax").unwrap_or(400.0),
        ],
        thickness_tex: None,
    });
    mat.dispersion = f("dispersion");
    true
}

/// The texture field a `glTF|` property names that no classic slot covers.
fn gltf_field(prop: &str) -> Option<TexField> {
    Some(match prop {
        "glTF|specularMap" => |m| m.specular.as_mut()?.factor.tex.as_mut(),
        "glTF|specularColorMap" => |m| m.specular.as_mut()?.color_tex.as_mut(),
        "glTF|thicknessMap" => |m| m.volume.as_mut()?.thickness.tex.as_mut(),
        "glTF|anisotropyMap" => |m| m.anisotropy.as_mut()?.strength.tex.as_mut(),
        "glTF|iridescenceMap" => |m| m.iridescence.as_mut()?.factor.tex.as_mut(),
        "glTF|iridescenceThicknessMap" => |m| m.iridescence.as_mut()?.thickness_tex.as_mut(),
        _ => return None,
    })
}

/// Puts texture `t` where `glTF|` property `prop` says, when the material has what holds it and it's empty.
fn put_gltf_texture(mat: &mut Full, prop: &str, t: Tex) -> bool {
    let place = match prop {
        "glTF|specularMap" => mat.specular.as_mut().map(|s| &mut s.factor.tex),
        "glTF|specularColorMap" => mat.specular.as_mut().map(|s| &mut s.color_tex),
        "glTF|thicknessMap" => mat.volume.as_mut().map(|v| &mut v.thickness.tex),
        "glTF|anisotropyMap" => mat.anisotropy.as_mut().map(|a| &mut a.strength.tex),
        "glTF|iridescenceMap" => mat.iridescence.as_mut().map(|i| &mut i.factor.tex),
        "glTF|iridescenceThicknessMap" => mat.iridescence.as_mut().map(|i| &mut i.thickness_tex),
        _ => None,
    };
    match place {
        Some(slot) if slot.is_none() => {
            *slot = Some(t);
            true
        }
        _ => false,
    }
}

/// The texture a material's slot takes from object `id`: the object itself, or for a layered texture, its first
/// layer.
fn texture_of(id: i64, objs: &Objects<'_>) -> i64 {
    if objs.is("LayeredTexture", id) {
        objs.children_of(id).find(|t| objs.is("Texture", *t)).unwrap_or(id)
    } else {
        id
    }
}

/// The UV set a texture reads, when it names one other than the first.
fn texture_uv_set(id: i64, objs: &Objects<'_>) -> Option<String> {
    let p = objs.props(id);
    let name = p.text("UVSet")?.split('\0').next()?;
    (!name.is_empty() && name != "default").then(|| name.to_owned())
}

/// A texture field of a material.
pub(super) type TexField = fn(&mut Full) -> Option<&mut Tex>;

/// A material, the UV set its color texture reads by name, and the sets its other textures read, where they name
/// one.
type Read = (Full, Option<String>, Vec<(TexField, String)>);

/// How transparent a material without an `Opacity` is, from 0 to 1. The FBX SDK's Lambert surface gives
/// transparency as `TransparentColor` scaled by `TransparencyFactor` (the color is what is transparent, the
/// factor how much of it counts), so the two are multiplied, the color as the mean of its channels: Maya
/// writes the color with a factor of 1, 3ds Max a white color with the factor. A factor without a color is the
/// transparency by itself, as the writers that give only it mean, except a factor of 1: the color it scales is
/// black unless given, so that material is opaque, not invisible. This is the reading Blender's FBX importer
/// and three.js's `FBXLoader` settle on too (a factor of exactly 1 or 0 decides nothing; the color does).
fn transparency(p: &Props<'_>) -> f64 {
    let factor = p.own_scalar("TransparencyFactor").map(|f| f.clamp(0.0, 1.0));
    let color = p.own.get("TransparentColor").filter(|v| v.len() >= 3).map(|v| {
        let sum: f64 = v[..3]
            .iter()
            .map(|c| num(c).filter(|x| x.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0))
            .sum();
        sum / 3.0
    });
    match (factor, color) {
        (Some(f), Some(c)) => f * c,
        (None, Some(c)) => c,
        (Some(f), None) if f < 1.0 => f,
        _ => 0.0,
    }
}

#[allow(clippy::too_many_lines, reason = "one property after another")]
fn read_material(
    id: i64,
    objs: &Objects<'_>,
    linear: bool,
    dir: &Path,
    // None when textures aren't kept: they aren't read.
    assets: Option<&dyn Assets>,
    images: &mut Images,
    image_index: &mut HashMap<i64, Option<usize>>,
) -> Read {
    let (node, _) = objs.nodes[&id];
    let p = objs.props_of(node);
    let name = node
        .attributes()
        .get(1)
        .and_then(string)
        .unwrap_or("material")
        .split('\0')
        .next()
        .unwrap_or("material")
        .to_owned();
    let to_linear = |c: [f64; 3]| {
        c.map(|v| {
            let v = f32_of(v.clamp(0.0, 1.0));
            if linear { v } else { srgb_to_linear(v) }
        })
    };
    let rgb = |c: [f64; 3]| Rgb::of(c.map(f32_of), linear);
    let mut phong = Phong::new(rgb(vec3(&p, "DiffuseColor", vec3(&p, "Diffuse", [0.8, 0.8, 0.8]))));
    // Opacity and reflection are read only where the material sets them: the file's defaults for them are no
    // statement about this one.
    phong.opacity = f32_of(p.own_scalar("Opacity").unwrap_or_else(|| 1.0 - transparency(&p)));
    phong.emissive = rgb(vec3(&p, "EmissiveColor", vec3(&p, "Emissive", [0.0; 3])));
    let shininess = scalar(&p, "Shininess").or_else(|| scalar(&p, "ShininessExponent"));
    if linear {
        // Blender's exporter keeps its roughness in the exponent, as ((1 - roughness) * 10)^2, and writes no
        // classic highlight: read back as the exponent of that roughness.
        phong.exponent = shininess.map(|n| {
            f32_of(legacy_material::exponent_of_roughness(
                1.0 - n.clamp(0.0, 100.0).sqrt() / 10.0,
            ))
        });
    } else {
        // `FbxSurfacePhong`: the exponent of the highlight, its color and the factor scaling it;
        // `FbxSurfaceLambert` (a shading model of "lambert") has no highlight.
        phong.exponent = shininess.map(|n| f32_of(n.max(0.0)));
        if p.get("SpecularColor").is_some() || p.get("Specular").is_some() {
            phong.specular = Some(rgb(vec3(&p, "SpecularColor", vec3(&p, "Specular", [0.2; 3]))));
            phong.strength = f32_of(scalar(&p, "SpecularFactor").unwrap_or(1.0));
        }
        let model = child_attr(node, "ShadingModel")
            .and_then(string)
            .or_else(|| p.text("ShadingModel"));
        phong.highlight = !model.is_some_and(|m| m.split('\0').next().unwrap_or(m).eq_ignore_ascii_case("lambert"));
    }
    let mut mat = Full::from(Material::plain(&name, [1.0; 4]));
    phong.pbr().set_full(&mut mat);
    // The emissive color is a color (sRGB unless Blender's); its factor is a linear intensity.
    let factor = f32_of(scalar(&p, "EmissiveFactor").unwrap_or(1.0).max(0.0));
    mat.emissive = mat.emissive.map(|c| c * factor);
    // Blender keeps its metallic in ReflectionFactor; other tools' reflectivity is a mirror blend, not metal.
    if linear && let Some(m) = p.own_scalar("ReflectionFactor") {
        mat.metallic = f32_of(m.clamp(0.0, 1.0));
    }
    // A PBR shader's own specular takes the classic highlight's place.
    let classic = mat.specular.take();
    let pbr = read_pbr(&mut mat, &p, &to_linear);
    if pbr.is_none() {
        mat.specular = classic;
    }
    let gltf = read_gltf(&mut mat, &p);
    let mut uv_sets: Vec<(TexField, String)> = Vec::new();
    let mut color_set = None;
    let (mut metallic, mut roughness, mut opacity) = (None, None, None);
    for (link, prop) in objs.children.get(&id).cloned().unwrap_or_default() {
        let Some(slot) = slot(&prop) else {
            if gltf
                && let Some(field) = gltf_field(&prop)
                && let Some(assets) = assets
            {
                let tex = texture_of(link, objs);
                let img = *image_index
                    .entry(tex)
                    .or_insert_with(|| texture_image(tex, objs, dir, assets, images));
                if let Some(t) = img.map(|i| texture_tex(tex, objs, i))
                    && put_gltf_texture(&mut mat, &prop, t)
                    && let Some(set) = texture_uv_set(tex, objs)
                {
                    uv_sets.push((field, set));
                }
            }
            continue;
        };
        let Some(assets) = assets else {
            continue;
        };
        let tex = texture_of(link, objs);
        let img = *image_index
            .entry(tex)
            .or_insert_with(|| texture_image(tex, objs, dir, assets, images));
        let Some(t) = img.map(|i| texture_tex(tex, objs, i)) else {
            continue;
        };
        let field: Option<TexField> = match slot {
            Slot::Color if mat.base_color_tex.is_none() => {
                // A texture takes the place of the color it's connected to (a classic material's diffuse factor is
                // usually a leftover): it's shown as painted, scaled only by a PBR shader's base weight. glTF's factor
                // does scale it.
                if !gltf {
                    mat.base_color[..3].fill(pbr.unwrap_or(1.0));
                }
                mat.base_color_tex = Some(t);
                Some(|m| m.base_color_tex.as_mut())
            }
            Slot::Normal if mat.normal_tex.is_none() => {
                mat.normal_tex = Some(t);
                Some(|m| m.normal_tex.as_mut())
            }
            Slot::Emissive if mat.emissive_tex.is_none() => {
                // An emissive map with no emissive color would show nothing: it's shown as painted.
                if !gltf && mat.emissive.iter().all(|c| *c == 0.0) {
                    mat.emissive = [1.0; 3];
                }
                mat.emissive_tex = Some(t);
                Some(|m| m.emissive_tex.as_mut())
            }
            Slot::Occlusion if mat.occlusion_tex.is_none() => {
                mat.occlusion_tex = Some(t);
                Some(|m| m.occlusion_tex.as_mut())
            }
            Slot::MetalRough | Slot::OcclusionMetalRough if mat.metallic_roughness_tex.is_none() => {
                if slot == Slot::OcclusionMetalRough && mat.occlusion_tex.is_none() {
                    mat.occlusion_tex = Some(t.clone());
                }
                mat.metallic_roughness_tex = Some(t);
                if !gltf {
                    mat.metallic = 1.0;
                    mat.roughness = 1.0;
                }
                Some(|m| m.metallic_roughness_tex.as_mut())
            }
            Slot::Metallic => {
                metallic = Some(t);
                None
            }
            Slot::Roughness => {
                roughness = Some(t);
                None
            }
            Slot::Opacity => {
                opacity = Some(t);
                None
            }
            Slot::Transmission => {
                let tr = mat.transmission.get_or_insert_with(|| Factor::of(1.0));
                tr.tex.get_or_insert(t);
                Some(|m| m.transmission.as_mut()?.tex.as_mut())
            }
            Slot::Coat => {
                let coat = mat.clearcoat.get_or_insert_with(|| Clearcoat {
                    factor: Factor::of(1.0),
                    roughness: Factor::of(0.0),
                    normal: None,
                });
                coat.factor.tex.get_or_insert(t);
                Some(|m| m.clearcoat.as_mut()?.factor.tex.as_mut())
            }
            Slot::CoatRoughness => {
                if let Some(coat) = &mut mat.clearcoat {
                    coat.roughness.tex.get_or_insert(t);
                }
                Some(|m| m.clearcoat.as_mut()?.roughness.tex.as_mut())
            }
            Slot::CoatNormal => {
                if let Some(coat) = &mut mat.clearcoat {
                    coat.normal.get_or_insert(t);
                }
                Some(|m| m.clearcoat.as_mut()?.normal.as_mut())
            }
            Slot::SheenColor => {
                if let Some(sheen) = &mut mat.sheen {
                    sheen.color_tex.get_or_insert(t);
                }
                Some(|m| m.sheen.as_mut()?.color_tex.as_mut())
            }
            Slot::SheenRoughness => {
                if let Some(sheen) = &mut mat.sheen {
                    sheen.roughness.tex.get_or_insert(t);
                }
                Some(|m| m.sheen.as_mut()?.roughness.tex.as_mut())
            }
            _ => None,
        };
        if let (Some(field), Some(set)) = (field, texture_uv_set(tex, objs)) {
            if slot == Slot::Color {
                color_set = Some(set);
            } else {
                uv_sets.push((field, set));
            }
        }
    }
    // glTF reads roughness and metalness from one image's green and blue.
    if mat.metallic_roughness_tex.is_none()
        && let (Some(m), Some(r)) = (&metallic, &roughness)
        && m.image == r.image
    {
        mat.metallic_roughness_tex.clone_from(&roughness);
        mat.metallic = 1.0;
        mat.roughness = 1.0;
    }
    // Opacity from the color texture is its alpha, glTF's own arrangement.
    if let (Some(o), Some(c)) = (&opacity, &mat.base_color_tex)
        && o.image == c.image
        && mat.alpha == Alpha::Opaque
        && !gltf
    {
        let transparency = Transparency {
            texture_alpha: true,
            ..Transparency::of(mat.base_color[3])
        };
        mat.alpha = transparency.mode();
    }
    if gltf {
        if let (Some(t), Some(s)) = (&mut mat.normal_tex, p.own_scalar("glTF|normalScale")) {
            t.scale = f32_of(s);
        }
        if let (Some(t), Some(s)) = (&mut mat.occlusion_tex, p.own_scalar("glTF|occlusionStrength")) {
            t.scale = f32_of(s);
        }
    }
    (mat, color_set, uv_sets)
}

/// A texture of `image` as texture `id` lays it out: its wrap modes, and its translation, rotation and scaling (on
/// UVs with V up, as `uv' = rotate * scale * uv + translate`) as a transform of the document's V-down UVs.
fn texture_tex(id: i64, objs: &Objects<'_>, image: usize) -> Tex {
    let mut t = Tex::new(image);
    let p = objs.props(id);
    let wrap = |k: &str| match scalar(&p, k) {
        Some(w) if w != 0.0 => material::CLAMP,
        _ => material::REPEAT,
    };
    t.wrap = [wrap("WrapModeU"), wrap("WrapModeV")];
    // glTF's own sampler, where the texture keeps it.
    let gl = |k: &str| {
        let v = scalar(&p, k).filter(|v| *v >= 0.0 && v.fract() == 0.0 && *v <= f64::from(u32::MAX))?;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a whole number in range"
        )]
        Some(v as u32)
    };
    for (w, k) in t.wrap.iter_mut().zip(["glTF|wrapS", "glTF|wrapT"]) {
        if let Some(v) = gl(k) {
            *w = v;
        }
    }
    for (f, k) in t.filter.iter_mut().zip(["glTF|magFilter", "glTF|minFilter"]) {
        if let Some(v) = gl(k) {
            *f = Some(v);
        }
    }
    let [tu, tv, _] = vec3(&p, "Translation", [0.0; 3]).map(f32_of);
    let [_, _, angle] = vec3(&p, "Rotation", [0.0; 3]).map(f32_of);
    let [su, sv, _] = vec3(&p, "Scaling", [1.0; 3]).map(f32_of);
    let (su, sv) = (if su == 0.0 { 1.0 } else { su }, if sv == 0.0 { 1.0 } else { sv });
    let angle = angle.to_radians();
    if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
        let (sin, cos) = angle.sin_cos();
        t.transform = Some(UvTransform {
            offset: [tu - sv * sin, 1.0 - tv - sv * cos],
            rotation: angle,
            scale: [su, sv],
        });
    }
    t
}

/// A texture's pixels: embedded in a connected `Video`'s `Content`, or the file it names, found in the pack. One
/// that gives neither is counted once as a texture the document doesn't have, and the file listed when it wasn't
/// found.
fn texture_image(id: i64, objs: &Objects<'_>, dir: &Path, assets: &dyn Assets, images: &mut Images) -> Option<usize> {
    let (node, _) = objs.nodes.get(&id)?;
    let names: Vec<String> = ["RelativeFilename", "FileName"]
        .iter()
        .filter_map(|k| child_attr(*node, k).and_then(string).map(str::to_owned))
        .filter(|s| !s.is_empty())
        .collect();
    let mut embedded = false;
    for (vid, _) in objs.children.get(&id).cloned().unwrap_or_default() {
        if let Some((video, _)) = objs.nodes.get(&vid)
            && video.name() == "Video"
            && let Some(A::Binary(bytes)) = child_attr(*video, "Content")
            && !bytes.is_empty()
        {
            let hint = names.first().map_or_else(
                || Path::new("t.png").to_path_buf(),
                |n| Path::new(&n.replace('\\', "/")).to_path_buf(),
            );
            let name = hint.to_string_lossy().into_owned();
            let convert = |b: &[u8]| match objs.file.as_ref().and_then(|f| shared_part(f, b)) {
                Some(part) => assets.image_shared(part, &hint),
                None => assets.image(b, &hint),
            };
            // Not an image: the file it was made from may be.
            if let Some(i) = images.try_bytes(Cow::Borrowed(&bytes[..]), &name, Location::Embedded, convert) {
                return Some(i);
            }
            embedded = true;
        }
    }
    if names.is_empty() {
        // Embedded, not an image, and naming no file to look for in its place.
        if embedded {
            images.drop("");
        }
        return None;
    }
    // Counted here when it isn't found, or isn't an image.
    images.find(assets, &names, &[dir])
}

/// `part` of `file`, sharing its bytes, when it lies inside them and is at least a quarter of them: a small texture
/// is copied instead, so that it doesn't keep a big file's bytes alive.
fn shared_part(file: &crate::Bytes, part: &[u8]) -> Option<crate::Bytes> {
    let (start, at) = (file.as_ptr().addr(), part.as_ptr().addr());
    let inside = at >= start && at - start + part.len() <= file.len();
    (inside && part.len() >= file.len() / 4).then(|| file.slice_ref(part))
}
