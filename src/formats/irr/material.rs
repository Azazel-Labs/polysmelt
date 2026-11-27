//! Irrlicht's attribute lists: the typed `<kind name="..." value="..."/>` elements that hold a node's settings,
//! an animator's or a material's, and the materials made from them.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::assets::Assets;
use crate::document::Keep;
use crate::formats::xml::El;
use crate::images::Images;
use crate::legacy_material::{Phong, Rgb, Transparency};
use crate::material::Alpha;
use crate::mesh::{CLAMP, MIRROR, Material, REPEAT, srgb_to_linear};

/// glTF's filter numbers.
const NEAREST: u32 = 9728;
const LINEAR: u32 = 9729;
const NEAREST_MIPMAP_NEAREST: u32 = 9984;
const LINEAR_MIPMAP_NEAREST: u32 = 9985;
const LINEAR_MIPMAP_LINEAR: u32 = 9987;

/// An attribute list: each attribute's kind (`float`, `vector3d`...), name and value, in order.
pub(super) struct Attrs {
    list: Vec<(String, String, String)>,
}

impl Attrs {
    /// The attributes `el` holds as its children.
    pub fn of(el: El<'_, '_>) -> Self {
        let list = el
            .children()
            .filter_map(|c| {
                let name = c.attr("name")?.into_owned();
                let value = c.attr("value").map(std::borrow::Cow::into_owned).unwrap_or_default();
                Some((c.name().to_owned(), name, value))
            })
            .collect();
        Self { list }
    }

    /// An attribute's value, its name in any case.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.list
            .iter()
            .find(|(_, n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, _, v)| v.as_str())
    }

    /// The attributes in order: kind, name and value.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.list.iter().map(|(k, n, v)| (k.as_str(), n.as_str(), v.as_str()))
    }

    /// A number, or `default` when it's missing or isn't one.
    pub fn f32(&self, name: &str, default: f32) -> f32 {
        self.get(name)
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(default)
    }

    /// A flag, or `default` when it's missing.
    pub fn bool(&self, name: &str, default: bool) -> bool {
        self.get(name)
            .map_or(default, |v| v.trim().eq_ignore_ascii_case("true"))
    }

    /// A vector (`"x, y, z"`), or `default` when it's missing.
    pub fn vec3(&self, name: &str, default: [f32; 3]) -> [f32; 3] {
        self.get(name).map_or(default, floats)
    }

    /// A floating-point color (`"r, g, b, a"`), or `default` when it's missing.
    pub fn colorf(&self, name: &str, default: [f32; 4]) -> [f32; 4] {
        self.get(name).map_or(default, floats)
    }

    /// An 8-bit color (`"aarrggbb"`), or `default` when it's missing.
    pub fn color(&self, name: &str, default: u32) -> u32 {
        self.get(name).map_or(default, argb)
    }
}

/// The numbers of a comma- or space-separated list, 0 for each missing one.
pub(super) fn floats<const N: usize>(v: &str) -> [f32; N] {
    let mut out = [0.0; N];
    for (o, t) in out
        .iter_mut()
        .zip(v.split([',', ' ', '\t', '\n', '\r']).filter(|t| !t.is_empty()))
    {
        *o = t.parse::<f32>().ok().filter(|x| x.is_finite()).unwrap_or(0.0);
    }
    out
}

/// A hexadecimal `aarrggbb` color; white when it isn't one.
pub(super) fn argb(v: &str) -> u32 {
    u32::from_str_radix(v.trim(), 16).unwrap_or(0xFFFF_FFFF)
}

/// An `aarrggbb` color as linear RGBA.
pub(super) fn linear(c: u32) -> [f32; 4] {
    let [a, r, g, b] = c.to_be_bytes().map(|x| f32::from(x) / 255.0);
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
}

/// A material as Irrlicht keeps it.
#[derive(Debug, Clone)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Irrlicht's own flags, and whether the file has the material"
)]
pub(super) struct IrrMaterial {
    /// Its type: `solid`, `trans_vertex_alpha`, `lightmap_m4`...
    pub kind: String,
    pub diffuse: u32,
    pub emissive: u32,
    pub specular: u32,
    pub shininess: f32,
    /// `Texture1` to `Texture4`.
    pub textures: [String; 4],
    /// Each texture's S and T wrap modes, glTF's numbers.
    pub wrap: [[u32; 2]; 4],
    /// Each texture's bilinear and trilinear filtering.
    pub filter: [[bool; 2]; 4],
    pub lighting: bool,
    pub backface_culling: bool,
    pub wireframe: bool,
    /// The buffer has no `<material>`: this is Irrlicht's own default, standing in.
    pub missing: bool,
}

impl Default for IrrMaterial {
    fn default() -> Self {
        Self {
            kind: "solid".into(),
            diffuse: 0xFFFF_FFFF,
            emissive: 0,
            specular: 0xFFFF_FFFF,
            shininess: 0.0,
            textures: Default::default(),
            wrap: [[REPEAT; 2]; 4],
            filter: [[true, false]; 4],
            lighting: true,
            backface_culling: true,
            wireframe: false,
            missing: false,
        }
    }
}

/// A wrap mode's name as a glTF wrap mode.
fn wrap(v: &str) -> u32 {
    let v = v.trim().to_ascii_lowercase();
    let v = v.strip_prefix("texture_clamp_").unwrap_or(&v);
    match v {
        "repeat" => REPEAT,
        "mirror" => MIRROR,
        _ => CLAMP,
    }
}

impl IrrMaterial {
    /// The material an attribute list describes.
    pub fn parse(a: &Attrs) -> Self {
        let mut m = Self::default();
        for (_, name, value) in a.iter() {
            let n = name.to_ascii_lowercase();
            // A texture slot's number, 1 to 4, at the end of its name.
            let slot = n
                .bytes()
                .last()
                .and_then(|d| d.checked_sub(b'1'))
                .map(usize::from)
                .filter(|&s| s < 4);
            match (n.as_str(), slot) {
                ("type", _) => m.kind = value.trim().to_ascii_lowercase(),
                ("diffuse", _) => m.diffuse = argb(value),
                ("emissive", _) => m.emissive = argb(value),
                ("specular", _) => m.specular = argb(value),
                ("shininess", _) => m.shininess = value.trim().parse().unwrap_or(0.0),
                ("lighting", _) => m.lighting = value.trim().eq_ignore_ascii_case("true"),
                ("backfaceculling", _) => m.backface_culling = value.trim().eq_ignore_ascii_case("true"),
                ("wireframe", _) => m.wireframe = value.trim().eq_ignore_ascii_case("true"),
                (t, Some(s)) if t.starts_with("texture") && t.len() == 8 => value.trim().clone_into(&mut m.textures[s]),
                (t, Some(s)) if t.starts_with("texturewrapu") => m.wrap[s][0] = wrap(value),
                (t, Some(s)) if t.starts_with("texturewrapv") => m.wrap[s][1] = wrap(value),
                (t, Some(s)) if t.starts_with("texturewrap") => m.wrap[s] = [wrap(value); 2],
                (t, Some(s)) if t.starts_with("bilinearfilter") => {
                    m.filter[s][0] = value.trim().eq_ignore_ascii_case("true");
                }
                (t, Some(s)) if t.starts_with("trilinearfilter") => {
                    m.filter[s][1] = value.trim().eq_ignore_ascii_case("true");
                }
                _ => {}
            }
        }
        m
    }

    /// Whether vertex alpha blends it.
    pub fn vertex_alpha(&self) -> bool {
        self.kind.ends_with("trans_vertex_alpha")
    }

    /// Whether its second texture is a lightmap.
    fn lightmap(&self) -> bool {
        self.kind.starts_with("lightmap")
    }

    /// Whether its second texture is a normal map.
    fn normal_map(&self) -> bool {
        self.kind.starts_with("normalmap") || self.kind.starts_with("parallaxmap")
    }
}

/// What a glTF material needs beyond what [`Material`] holds, written into the document afterwards.
#[derive(Debug, Clone, Default)]
pub(super) struct Patch {
    /// Blended by its vertices' alpha.
    pub blend: bool,
    /// A lightmap on the second coordinate set (it's written as the occlusion texture, which reads the first).
    pub second_set: bool,
    /// The Irrlicht type, when it's one glTF can't show.
    pub kind: Option<String>,
}

/// The images textures name, each read once.
pub(super) struct Textures<'a> {
    assets: &'a dyn Assets,
    pub images: Images,
    root: PathBuf,
    keep: bool,
}

impl<'a> Textures<'a> {
    pub fn new(assets: &'a dyn Assets, keep: Keep, root: &Path) -> Self {
        Self {
            assets,
            images: Images::new(!keep.image_bytes, root),
            root: root.to_path_buf(),
            keep: keep.textures,
        }
    }

    /// The image file `name` names, from the folder of the file that names it or else the scene's.
    pub fn get(&mut self, name: &str, dir: &Path) -> Option<usize> {
        if !self.keep || name.trim().is_empty() {
            return None;
        }
        let dirs: &[&Path] = if dir == self.root { &[dir] } else { &[dir, &self.root] };
        self.images.find(self.assets, &[name], dirs)
    }
}

/// The glTF material for an Irrlicht one, named `name`, its textures found from `dir`; `sets` is how many
/// coordinate sets its vertices have.
pub(super) fn convert(
    m: &IrrMaterial,
    name: &str,
    tex: &mut Textures<'_>,
    dir: &Path,
    sets: usize,
) -> (Material, Patch) {
    let srgb = |c: u32| {
        let [_, r, g, b] = c.to_be_bytes().map(|x| f32::from(x) / 255.0);
        Rgb::Srgb([r, g, b])
    };
    // `SMaterial`: Shininess is the highlight's exponent (0.5 to 128), and at 0 there is no highlight at all,
    // whatever the specular color.
    let mut phong = Phong::new(srgb(m.diffuse));
    phong.emissive = srgb(m.emissive);
    phong.specular = Some(srgb(m.specular));
    phong.exponent = Some(m.shininess.clamp(0.0, 128.0));
    phong.highlight = m.shininess > 0.0;
    let mut out = phong.pbr().material(name);
    out.unlit = !m.lighting;
    if out.unlit {
        // Nothing reflects, and nothing needs saying so.
        out.extensions = None;
    }
    out.double_sided = !m.backface_culling;
    let sampling = |s: usize| {
        let [bi, tri] = m.filter[s];
        crate::mesh::Sampling {
            wrap: m.wrap[s],
            filter: if tri {
                [Some(LINEAR), Some(LINEAR_MIPMAP_LINEAR)]
            } else if bi {
                [Some(LINEAR), Some(LINEAR_MIPMAP_NEAREST)]
            } else {
                [Some(NEAREST), Some(NEAREST_MIPMAP_NEAREST)]
            },
            ..Default::default()
        }
    };
    out.color_image = tex.get(&m.textures[0], dir);
    out.sampling.color = sampling(0);
    let mut patch = Patch::default();
    if m.normal_map() {
        out.normal_image = tex.get(&m.textures[1], dir);
        out.sampling.normal = sampling(1);
    } else if m.lightmap() {
        out.occlusion_image = tex.get(&m.textures[1], dir);
        out.sampling.occlusion = sampling(1);
        patch.second_set = sets > 1 && out.occlusion_image.is_some();
    }
    // `trans_alphach_ref` tests the texture's alpha against a half; `trans_alphach` blends by it.
    let transparency = Transparency {
        texture_alpha: m.kind == "trans_alphach",
        cutout: (m.kind == "trans_alphach_ref").then_some(0.5),
        ..Transparency::of(1.0)
    };
    match transparency.mode() {
        Alpha::Mask(c) => out.alpha_cutoff = Some(c),
        Alpha::Blend => patch.blend = true,
        Alpha::Opaque => {}
    }
    match m.kind.as_str() {
        "trans_alphach_ref" | "trans_alphach" => {}
        k if m.vertex_alpha() || k.is_empty() => {}
        "solid" | "lightmap" | "lightmap_m2" | "lightmap_m4" | "lightmap_light" | "lightmap_light_m2"
        | "lightmap_light_m4" | "normalmap_solid" | "parallaxmap_solid" => {}
        k => patch.kind = Some(k.to_owned()),
    }
    if m.wireframe {
        patch.kind = Some(format!("{} wireframe", m.kind));
    }
    (out, patch)
}

/// Writes what the patches add to the document's materials: blending, lightmaps' coordinate sets and, with
/// `extras`, the Irrlicht types glTF can't show.
pub(super) fn patch(json: &mut Map<String, Value>, patches: &[Patch], extras: bool) {
    let Some(Value::Array(materials)) = json.get_mut("materials") else {
        return;
    };
    for (m, p) in materials.iter_mut().zip(patches) {
        let Some(o) = m.as_object_mut() else { continue };
        if p.blend {
            o.insert("alphaMode".into(), json!("BLEND"));
        }
        if p.second_set
            && let Some(Value::Object(t)) = o.get_mut("occlusionTexture")
        {
            t.insert("texCoord".into(), json!(1));
        }
        if extras && let Some(k) = &p.kind {
            o.insert("extras".into(), json!({ "irrlichtType": k }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_materials() {
        let x = crate::formats::xml::Xml::parse(
            "<material><enum name='Type' value='lightmap_m4'/><color name='Diffuse' value='ff808080'/>\
             <texture name='Texture2' value='lm.png'/><bool name='BackfaceCulling' value='false'/>\
             <enum name='TextureWrap1' value='texture_clamp_mirror'/><enum name='TextureWrapV2' value='texture_clamp_clamp'/>\
             <bool name='Lighting' value='false'/></material>",
        )
        .unwrap();
        let m = IrrMaterial::parse(&Attrs::of(x.root()));
        assert_eq!(m.kind, "lightmap_m4");
        assert_eq!(m.diffuse, 0xFF80_8080);
        assert_eq!(m.textures[1], "lm.png");
        assert!(!m.backface_culling && !m.lighting && m.lightmap());
        assert_eq!(m.wrap[0], [MIRROR; 2]);
        assert_eq!(m.wrap[1], [REPEAT, CLAMP]);
        assert_eq!(floats::<3>("1, -2.5,3"), [1.0, -2.5, 3.0]);
        assert_eq!(linear(0x80FF_0000), [1.0, 0.0, 0.0, 128.0 / 255.0]);
    }
}
