//! One reading of the classic material (Phong, Blinn or Lambert: a diffuse color, a specular color, a highlight
//! exponent) as glTF's metallic-roughness one, shared by every reader, so the same material comes out the same
//! whatever format held it. A reader only brings its format's numbers to the scales [`Phong`] uses (a 3DS
//! percentage, a `LightWave` glossiness, an X3D shininess of 0 to 1 all become a Blinn-Phong exponent) and
//! [`Phong::pbr`] does the rest. Here too: the material of faces a file gives none ([`default_material`]), and
//! the one rule for how a material's transparency is drawn ([`Transparency::mode`]).
//!
//! # The rules
//!
//! - **Metalness** is 0: a classic material is a dielectric.
//! - **Roughness** is `(2 / (n + 2))^(1/4)` for a Blinn-Phong exponent `n`. A Blinn-Phong lobe of exponent `n`
//!   is closest to a GGX lobe of `alpha^2 = 2 / (n + 2)`, and glTF's roughness is `sqrt(alpha)`. An exponent of
//!   20 is a roughness of 0.549.
//! - **No highlight** (a Lambert material, a black specular color or a strength of zero) is a roughness of 1
//!   and a `KHR_materials_specular` factor of 0, which takes glTF's own reflection away as well.
//! - **Specular color**: the file's color scaled by its strength (the classic model multiplies the two, in
//!   the values it stores), made linear, is how much of glTF's dielectric reflection is left. The largest
//!   channel (at most 1) is `specularFactor` and the color over it is `specularColorFactor`, so its largest
//!   channel is 1. A classic white specular of full strength is glTF's default dielectric, and no extension
//!   is written for it: a file asking for all of the highlight gets all glTF has, and 0.5 gray (or white at
//!   half strength) gets its linear value, 0.214, of it. The classic model has no Fresnel term to measure
//!   against, so the full highlight is the fixed point that can be matched.
//! - **Transparency** is one decision, [`Transparency::mode`]: cut out (`MASK`) where the format asks for an
//!   alpha test, blended where the material, its color texture's alpha or its vertices' colors are less than
//!   opaque, opaque otherwise. A reader passes a cutout as a hint, never decides the mode itself. The hints:
//!   a `MilkShape` 3D model's alpha reference, an OGRE `alpha_rejection`, an Irrlicht `trans_alphach_ref`
//!   and a `Blitz3D` masked texture (each the format's own alpha test); and, at a half, three conventions
//!   the formats' tools draw as cutouts: an MTL `map_d` naming the `map_Kd` image, a `SketchUp` texture
//!   with an alpha channel, and a `MikuMikuDance` texture in a format that carries alpha (PNG, TGA, DDS).
//! - **Nothing said** (no exponent in the file, and no default in the format) is [`ROUGHNESS`], as for the
//!   material of faces with none: no information, not no highlight.

// Each reader uses its share; a build without all of them leaves some unused.
#![allow(dead_code, reason = "each reader uses its share")]
// Defaults are compared exactly, to leave them unwritten.
#![allow(clippy::float_cmp, reason = "defaults are compared exactly")]

use crate::material::{Alpha, Factor, Full, Specular};
use crate::mesh::{Extensions, GRAY, Material, srgb_to_linear};

/// The roughness of a material whose file says nothing of its highlight ([`Material::plain`]'s).
pub(crate) const ROUGHNESS: f32 = 0.85;

/// The name of the material given to faces that have none.
pub(crate) const DEFAULT_NAME: &str = "default";

/// A color, with the transfer function its numbers are in: the two can't be passed for one another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Rgb {
    /// Display values, as most classic formats store them.
    Srgb([f32; 3]),
    /// Linear values.
    Linear([f32; 3]),
}

impl Rgb {
    pub(crate) const BLACK: Self = Self::Linear([0.0; 3]);
    pub(crate) const WHITE: Self = Self::Linear([1.0; 3]);

    /// `c`, sRGB or (when `linear`) linear.
    pub(crate) fn of(c: [f32; 3], linear: bool) -> Self {
        if linear { Self::Linear(c) } else { Self::Srgb(c) }
    }

    /// Linear values, none negative or not a number; display values are clamped to 0..1 first.
    pub(crate) fn linear(self) -> [f32; 3] {
        self.scaled(1.0)
    }

    /// Linear values of a light given off, which may be brighter than 1: none negative or not a number.
    fn glow(self) -> [f32; 3] {
        let clean = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        match self {
            Self::Srgb(c) => c.map(|v| srgb_to_linear(clean(v))),
            Self::Linear(c) => c.map(clean),
        }
    }

    /// The color times `k`, in the values it is stored in, as linear values.
    fn scaled(self, k: f32) -> [f32; 3] {
        let clean = |v: f32| if v.is_finite() { (v * k).max(0.0) } else { 0.0 };
        match self {
            Self::Srgb(c) => c.map(|v| srgb_to_linear(clean(v).min(1.0))),
            Self::Linear(c) => c.map(clean),
        }
    }
}

/// A classic material as a file gives it, on the scales every reader converts its own to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Phong {
    /// The diffuse color.
    pub diffuse: Rgb,
    /// 1 is opaque.
    pub opacity: f32,
    /// The light given off.
    pub emissive: Rgb,
    /// The highlight's color. `None` when the file gives none and its format sets no default: the highlight
    /// is then glTF's own.
    pub specular: Option<Rgb>,
    /// Scales `specular` (a "specular level" or "shininess strength"), 0 to 1.
    pub strength: f32,
    /// The Blinn-Phong exponent, 0 up. `None` when the file gives none and its format sets no default.
    pub exponent: Option<f32>,
    /// Whether the shading model has a highlight at all: false for a Lambert material.
    pub highlight: bool,
}

impl Phong {
    /// An opaque material of a diffuse color that says nothing more.
    pub(crate) fn new(diffuse: Rgb) -> Self {
        Self {
            diffuse,
            opacity: 1.0,
            emissive: Rgb::BLACK,
            specular: None,
            strength: 1.0,
            exponent: None,
            highlight: true,
        }
    }

    /// The glTF material's numbers.
    pub(crate) fn pbr(&self) -> Pbr {
        let [r, g, b] = self.diffuse.linear().map(|v| v.min(1.0));
        let opacity = if self.opacity.is_finite() {
            self.opacity.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let strength = if self.strength.is_finite() {
            self.strength.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let level = self.specular.map(|s| s.scaled(strength));
        let top = level.map(|s| s[0].max(s[1]).max(s[2]));
        let dark = !self.highlight || top.is_some_and(|t| t <= 0.0) || strength <= 0.0;
        let (roughness, specular) = if dark {
            (1.0, Some(Highlight::NONE))
        } else {
            let roughness = self.exponent.map_or(ROUGHNESS, roughness_of_exponent);
            let specular = level.zip(top).and_then(|(s, top)| {
                let h = Highlight {
                    factor: top.min(1.0),
                    color: s.map(|v| v / top),
                };
                // Within a step of an 8-bit channel of the default is the default.
                let plain = h.factor >= 0.998 && h.color.iter().all(|&c| c >= 0.998);
                (!plain).then_some(h)
            });
            (roughness, specular)
        };
        Pbr {
            base_color: [r, g, b, opacity],
            roughness,
            emissive: self.emissive.glow(),
            specular,
        }
    }
}

/// A Blinn-Phong exponent as glTF roughness: `(2 / (n + 2))^(1/4)`.
pub(crate) fn roughness_of_exponent(n: f32) -> f32 {
    let n = if n.is_nan() { 0.0 } else { n.max(0.0) };
    (2.0 / (n + 2.0)).sqrt().sqrt().clamp(0.0, 1.0)
}

/// The Blinn-Phong exponent [`roughness_of_exponent`] reads back as `roughness`: `2 / r^4 - 2`. The smoothest
/// surfaces stop at the exponent of a roughness of 0.03, some 2.5 million.
pub(crate) fn exponent_of_roughness(roughness: f64) -> f64 {
    let r = if roughness.is_nan() {
        1.0
    } else {
        roughness.clamp(0.03, 1.0)
    };
    (2.0 / r.powi(4) - 2.0).max(0.0)
}

/// `KHR_materials_specular`'s two numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Highlight {
    /// `specularFactor`.
    pub factor: f32,
    /// `specularColorFactor`, linear.
    pub color: [f32; 3],
}

impl Highlight {
    /// No reflection at all.
    pub(crate) const NONE: Self = Self {
        factor: 0.0,
        color: [1.0; 3],
    };
}

/// What [`Phong::pbr`] works out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pbr {
    /// Linear RGB and opacity.
    pub base_color: [f32; 4],
    pub roughness: f32,
    /// Linear RGB.
    pub emissive: [f32; 3],
    /// `None` is glTF's default, written as no extension.
    pub specular: Option<Highlight>,
}

impl Pbr {
    /// Sets `m`'s color, emission, metalness, roughness and specular extension; the rest is left as it is.
    pub(crate) fn set(&self, m: &mut Material) {
        m.color = self.base_color;
        m.emissive = self.emissive;
        m.metallic = 0.0;
        m.roughness = self.roughness;
        match self.specular {
            Some(h) => {
                m.extensions.get_or_insert_with(Box::<Extensions>::default).specular = Some(Specular {
                    factor: Factor::of(h.factor),
                    color: h.color,
                    color_tex: None,
                });
            }
            None => {
                if let Some(x) = m.extensions.as_deref_mut() {
                    x.specular = None;
                }
            }
        }
    }

    /// A material named `name` of these numbers.
    pub(crate) fn material(&self, name: &str) -> Material {
        let mut m = Material::plain(name, self.base_color);
        self.set(&mut m);
        m
    }

    /// Sets `m`'s color, emission, metalness, roughness and specular extension, and its alpha mode by its
    /// opacity alone; the rest is left as it is.
    pub(crate) fn set_full(&self, m: &mut Full) {
        m.base_color = self.base_color;
        m.alpha = Transparency::of(self.base_color[3]).mode();
        m.emissive = self.emissive;
        m.metallic = 0.0;
        m.roughness = self.roughness;
        m.specular = self.specular.map(|h| Specular {
            factor: Factor::of(h.factor),
            color: h.color,
            color_tex: None,
        });
    }

    /// A material named `name` of these numbers.
    pub(crate) fn full(&self, name: &str) -> Full {
        let mut m = Full::from(Material::plain(name, self.base_color));
        self.set_full(&mut m);
        m
    }
}

/// The material of faces their file gives none: named [`DEFAULT_NAME`], [`GRAY`] (white when `colored`: under
/// vertex colors or a texture, which then show as they are), not metallic, and of [`ROUGHNESS`].
pub(crate) fn default_material(colored: bool) -> Material {
    Material::plain(DEFAULT_NAME, if colored { [1.0; 4] } else { GRAY })
}

/// What decides how a material's transparency is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Transparency {
    /// The material's own opacity: 1 is opaque.
    pub opacity: f32,
    /// The color texture's alpha is the material's opacity: the format says so, or the texture is known to
    /// have an alpha channel in a format that always uses it. Never whether a file was found.
    pub texture_alpha: bool,
    /// Vertex colors with an alpha below 1.
    pub vertex_alpha: bool,
    /// The format says the alpha is tested against this threshold, not blended.
    pub cutout: Option<f32>,
}

impl Transparency {
    /// A material of `opacity` and nothing else.
    pub(crate) fn of(opacity: f32) -> Self {
        Self {
            opacity,
            texture_alpha: false,
            vertex_alpha: false,
            cutout: None,
        }
    }

    /// glTF's alpha mode: cut out where the format asks for an alpha test; else blended where anything is less
    /// than opaque (the material, its texture's alpha or its vertices' colors); else opaque.
    pub(crate) fn mode(self) -> Alpha {
        if let Some(c) = self.cutout {
            return Alpha::Mask(if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.5 });
        }
        if self.opacity < 1.0 || self.texture_alpha || self.vertex_alpha {
            Alpha::Blend
        } else {
            Alpha::Opaque
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp_const, reason = "defaults are compared exactly")]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn an_exponent_is_the_ggx_roughness_of_its_lobe() {
        // (2 / (n + 2))^(1/4), worked by hand: n = 0 is 1, n = 20 is (1/11)^0.25, n = 30 is 0.5.
        assert_eq!(roughness_of_exponent(0.0), 1.0);
        assert!(close(roughness_of_exponent(20.0), 0.549_100));
        assert!(close(roughness_of_exponent(30.0), 0.5));
        assert!(close(roughness_of_exponent(1000.0), 0.211_368));
        assert_eq!(roughness_of_exponent(-5.0), 1.0);
        assert_eq!(roughness_of_exponent(f32::NAN), 1.0);
        for r in [0.05f32, 0.2, 0.5, 0.85, 1.0] {
            #[allow(clippy::cast_possible_truncation, reason = "a test value")]
            let n = exponent_of_roughness(f64::from(r)) as f32;
            assert!(close(roughness_of_exponent(n), r), "{r}");
        }
    }

    #[test]
    fn a_phong_material_becomes_a_rough_dielectric() {
        let mut p = Phong::new(Rgb::Srgb([0.8, 0.2, 0.1]));
        p.specular = Some(Rgb::Srgb([0.5; 3]));
        p.exponent = Some(20.0);
        let pbr = p.pbr();
        // sRGB 0.8, 0.2, 0.1 and 0.5 as linear values, by the sRGB formula.
        for (got, want) in pbr.base_color.iter().zip([0.603_827, 0.033_105, 0.010_023, 1.0]) {
            assert!(close(*got, want), "{:?}", pbr.base_color);
        }
        assert!(close(pbr.roughness, 0.549_100));
        let h = pbr.specular.unwrap();
        assert!(close(h.factor, 0.214_041), "{h:?}");
        assert_eq!(h.color, [1.0; 3]);
        let m = pbr.material("m");
        assert_eq!((m.metallic, m.roughness), (0.0, pbr.roughness));
        assert!(m.extensions.unwrap().specular.is_some());
        let f = pbr.full("m");
        assert_eq!((f.metallic, f.roughness), (0.0, pbr.roughness));
    }

    #[test]
    fn a_full_white_highlight_is_gltfs_default_and_none_is_no_reflection() {
        let mut p = Phong::new(Rgb::Srgb([0.5; 3]));
        p.exponent = Some(30.0);
        p.specular = Some(Rgb::Srgb([1.0; 3]));
        assert_eq!(p.pbr().specular, None);
        // A tinted highlight: the color over its largest channel, which is the factor.
        p.specular = Some(Rgb::Linear([0.5, 0.25, 0.0]));
        let h = p.pbr().specular.unwrap();
        assert!(close(h.factor, 0.5));
        assert_eq!(h.color, [1.0, 0.5, 0.0]);
        // Its strength scales it; a display color's in display values, so white at half strength is 0.5 gray.
        p.strength = 0.5;
        assert!(close(p.pbr().specular.unwrap().factor, 0.25));
        p.specular = Some(Rgb::Srgb([1.0; 3]));
        assert!(close(p.pbr().specular.unwrap().factor, 0.214_041));
        p.specular = Some(Rgb::Linear([0.5, 0.25, 0.0]));
        // Black, no strength and Lambert are all no highlight.
        p.strength = 0.0;
        assert_eq!((p.pbr().roughness, p.pbr().specular), (1.0, Some(Highlight::NONE)));
        p.strength = 1.0;
        p.specular = Some(Rgb::Srgb([0.0; 3]));
        assert_eq!((p.pbr().roughness, p.pbr().specular), (1.0, Some(Highlight::NONE)));
        p.specular = Some(Rgb::Srgb([1.0; 3]));
        p.highlight = false;
        assert_eq!((p.pbr().roughness, p.pbr().specular), (1.0, Some(Highlight::NONE)));
        // An emission brighter than 1 stays so; a diffuse color doesn't.
        let mut bright = Phong::new(Rgb::Linear([2.0, 0.5, 0.0]));
        bright.emissive = Rgb::Linear([4.0, 0.0, 0.0]);
        assert_eq!(
            (bright.pbr().base_color, bright.pbr().emissive),
            ([1.0, 0.5, 0.0, 1.0], [4.0, 0.0, 0.0])
        );
        // Nothing said of the highlight: no information.
        let quiet = Phong::new(Rgb::Srgb([0.5; 3])).pbr();
        assert_eq!((quiet.roughness, quiet.specular), (ROUGHNESS, None));
    }

    #[test]
    fn the_default_material_is_one_gray() {
        let m = default_material(false);
        assert_eq!(
            (m.name.as_str(), m.color, m.roughness, m.metallic),
            ("default", GRAY, 0.85, 0.0)
        );
        assert_eq!(default_material(true).color, [1.0; 4]);
        assert_eq!(ROUGHNESS, Material::plain("", GRAY).roughness);
    }

    #[test]
    fn transparency_is_cut_out_when_asked_and_blended_otherwise() {
        let opaque = Transparency::of(1.0);
        assert_eq!(opaque.mode(), Alpha::Opaque);
        assert_eq!(Transparency::of(0.5).mode(), Alpha::Blend);
        let texture = Transparency {
            texture_alpha: true,
            ..opaque
        };
        assert_eq!(texture.mode(), Alpha::Blend);
        let vertex = Transparency {
            vertex_alpha: true,
            ..opaque
        };
        assert_eq!(vertex.mode(), Alpha::Blend);
        let cut = Transparency {
            cutout: Some(0.3),
            ..Transparency::of(0.5)
        };
        assert_eq!(cut.mode(), Alpha::Mask(0.3));
    }
}
