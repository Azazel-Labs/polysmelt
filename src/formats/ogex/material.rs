//! Materials: an `OpenGEX` material's colors, parameters and textures as a glTF material.

use serde_json::{Map, Value, json};

use super::{Ctx, color, param, product, transforms};
use crate::material::{Alpha, CLAMP, Clearcoat, Factor, Full, MIRROR, REPEAT, Sheen, Specular, Tex, UvTransform};
use crate::num::{f32_of, u32_of};

/// A texture as read: the glTF texture and the channel the swizzle puts first (0 to 3, or
/// `None` for a constant).
struct Texture {
    tex: Tex,
    channel: Option<usize>,
    /// Whether the swizzle is left as stored, so a one-channel (gray) image is read from
    /// whichever channel glTF reads.
    identity: bool,
}

/// The material structure `m` as a glTF material; its index. `None` when it isn't one.
pub(super) fn material(c: &mut Ctx<'_>, m: u32) -> Option<usize> {
    if let Some(&k) = c.material_of.get(&m) {
        return k;
    }
    let out = (c.t.ident(m) == "Material").then(|| build(c, m));
    c.material_of.insert(m, out);
    out
}

/// The first channel swizzle `s` picks: 0 to 3, `None` for a constant.
fn channel(s: &str) -> Option<usize> {
    match s.as_bytes().first() {
        Some(b'r' | b'x' | b'i') | None => Some(0),
        Some(b'g' | b'y') => Some(1),
        Some(b'b' | b'z') => Some(2),
        Some(b'a' | b'w') => Some(3),
        _ => None,
    }
}

/// An addressing mode as a glTF wrap: repeating, mirrored repeating, or clamped for the rest.
fn wrap(mode: Option<&str>) -> u32 {
    match mode {
        None | Some("repeat") => REPEAT,
        Some("mirror_repeat") => MIRROR,
        Some(_) => CLAMP,
    }
}

/// A texture coordinate transform (column-major, on `OpenGEX`'s bottom-left coordinates) as
/// `KHR_texture_transform`'s, on glTF's top-left ones; whether it had a shear glTF can't hold.
fn uv_transform(m: &super::M4) -> (Option<UvTransform>, bool) {
    // glTF coordinates are flipped, (u, 1 - v): the transform seen through the flip.
    let apply = |u: f64, v: f64| {
        let (x, y) = (u, 1.0 - v);
        let tx = m[0] * x + m[4] * y + m[12];
        let ty = m[1] * x + m[5] * y + m[13];
        (tx, 1.0 - ty)
    };
    let o = apply(0.0, 0.0);
    let ex = apply(1.0, 0.0);
    let ey = apply(0.0, 1.0);
    let (a, b) = (ex.0 - o.0, ex.1 - o.1);
    let (cc, d) = (ey.0 - o.0, ey.1 - o.1);
    // T·R·S with R = [[cos, sin], [-sin, cos]]: the first column is sx·(cos, -sin), the second
    // sy·(sin, cos).
    let sx = a.hypot(b);
    let r = (-b).atan2(a);
    let (sin, cos) = r.sin_cos();
    let sy = cc * sin + d * cos;
    let shear = (cc * cos - d * sin).abs() > 1e-6 * (sx.abs() + sy.abs()).max(1e-12);
    let t = UvTransform {
        offset: [f32_of(o.0), f32_of(o.1)],
        rotation: f32_of(r),
        scale: [f32_of(sx), f32_of(sy)],
    };
    let all_finite = [o.0, o.1, r, sx, sy].iter().all(|x| x.is_finite());
    let identity = t == UvTransform::IDENTITY;
    ((all_finite && !identity).then_some(t), shear)
}

/// The image of texture file `file`: `None` when it's missing or textures aren't kept. A missing
/// file is counted whether or not textures are kept, as a full read counts it.
fn image(c: &mut Ctx<'_>, file: String) -> Option<usize> {
    if let Some(&i) = c.image_of.get(&file) {
        return i;
    }
    let i = match c.assets.find(&file, &c.dir) {
        Some(path) if c.keep.textures => c.images.file(c.assets, path, &file),
        Some(_) => None,
        None => {
            c.stats.missing_textures += 1;
            None
        }
    };
    c.image_of.insert(file, i);
    i
}

/// Counts the missing texture files of material structure `m` when materials aren't kept, as
/// a full read counts them.
pub(super) fn count_missing(c: &mut Ctx<'_>, m: u32) {
    let t = c.t;
    if t.ident(m) != "Material" || !c.counted.insert(m) {
        return;
    }
    for k in t.kids(m).filter(|&k| t.ident(k) == "Texture") {
        if let Some(file) = t.string(k) {
            image(c, file.to_owned());
        }
    }
}

/// The texture structure `k`; `None` when its image is missing or textures aren't kept.
fn texture(c: &mut Ctx<'_>, k: u32, uvs: &mut usize, record: &mut Vec<Value>) -> Option<Texture> {
    let t = c.t;
    let file = t.string(k)?.to_owned();
    let texcoord = t
        .prop_num(k, "texcoord")
        .map_or(0.0, |v| if (0.0..=7.0).contains(&v) { v.trunc() } else { 0.0 });
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0 to 7")]
    let texcoord = texcoord as usize;
    let swizzle = t.prop_str(k, "swizzle").unwrap_or("i");
    let xs = transforms(t, k, &mut c.stats.bad_transforms);
    let animated = t.kids(k).any(|a| t.ident(a) == "Animation");
    if animated {
        c.stats.animated_textures += 1;
    }
    let m = product(xs.iter(), c.metric.angle);
    let (transform, shear) = uv_transform(&m);
    if shear {
        c.stats.sheared_textures += 1;
    }
    let mut e = json!({ "attrib": t.prop_str(k, "attrib").unwrap_or(""), "file": file, "texcoord": texcoord });
    for key in ["swizzle", "x_address", "y_address", "border"] {
        if let Some(v) = t.prop_str(k, key) {
            e[key] = json!(v);
        }
    }
    record.push(e);
    let image = image(c, file);
    if !c.keep.textures {
        return None;
    }
    let mut tex = Tex::new(image?);
    tex.uv = u32_of(texcoord);
    tex.transform = transform;
    tex.wrap = [wrap(t.prop_str(k, "x_address")), wrap(t.prop_str(k, "y_address"))];
    *uvs = (*uvs).max(texcoord + 1);
    Some(Texture {
        tex,
        channel: channel(swizzle),
        identity: swizzle == "i",
    })
}

/// The material's glTF form.
#[allow(clippy::too_many_lines, reason = "one attribute after another")]
fn build(c: &mut Ctx<'_>, m: u32) -> usize {
    let t = c.t;
    let mut uvs = 0;
    let mut record = Vec::new();
    let mut texs: Vec<(String, Texture)> = Vec::new();
    for k in t.kids(m).filter(|&k| t.ident(k) == "Texture") {
        let attrib = t.prop_str(k, "attrib").unwrap_or("").to_owned();
        if let Some(x) = texture(c, k, &mut uvs, &mut record) {
            texs.push((attrib, x));
        }
    }
    c.stats.spectra += t.kids(m).filter(|&k| t.ident(k) == "Spectrum").count();
    // A texture of `attrib` read from `want` (the channel glTF reads it from), or any channel
    // when `want` is `None`; one read from the wrong channel is left out.
    let wrong = std::cell::Cell::new(0usize);
    let left_out = || wrong.set(wrong.get() + 1);
    let tex = |attrib: &str, want: Option<usize>| -> Option<Tex> {
        let (_, x) = texs.iter().find(|(a, _)| a == attrib)?;
        if want.is_some() && x.channel != want && !x.identity {
            left_out();
            return None;
        }
        Some(x.tex.clone())
    };
    let col = |attrib: &str| color(c, m, attrib);
    let rgb = |v: [f64; 4]| [f32_of(v[0]), f32_of(v[1]), f32_of(v[2])];
    let diffuse = col("diffuse").unwrap_or([1.0; 4]);
    let opacity = param(t, m, "opacity").unwrap_or(1.0).clamp(0.0, 1.0);
    let alpha = diffuse[3].clamp(0.0, 1.0) * opacity;
    let base_tex = tex("diffuse", None);
    // Opacity from a texture: glTF reads it from the base color texture's alpha, so only an
    // opacity texture that is that same image's alpha fits.
    let opacity_tex = texs.iter().find(|(a, _)| a == "opacity").map(|(_, x)| x);
    let opacity_fits =
        opacity_tex.is_none_or(|x| x.channel == Some(3) && base_tex.as_ref().is_some_and(|b| b.image == x.tex.image));
    if !opacity_fits {
        left_out();
    }
    let blend = alpha < 1.0 || opacity_tex.is_some();
    let pbr = param(t, m, "roughness").is_some()
        || param(t, m, "metalness").is_some()
        || texs.iter().any(|(a, _)| a == "roughness" || a == "metalness");
    let specular = col("specular");
    let power = param(t, m, "specular_power").unwrap_or(1.0).max(0.0);
    let mut full = Full {
        name: t
            .kid(m, "Name")
            .and_then(|k| t.string(k))
            .or_else(|| t.name(m))
            .unwrap_or("")
            .to_owned(),
        base_color: [
            f32_of(diffuse[0]),
            f32_of(diffuse[1]),
            f32_of(diffuse[2]),
            f32_of(alpha),
        ],
        base_color_tex: base_tex,
        alpha: if blend { Alpha::Blend } else { Alpha::Opaque },
        double_sided: t.prop_bool(m, "two_sided").unwrap_or(false),
        ..Full::default()
    };
    if pbr {
        full.metallic = f32_of(param(t, m, "metalness").unwrap_or(0.0).clamp(0.0, 1.0));
        full.roughness = f32_of(param(t, m, "roughness").unwrap_or(0.0).clamp(0.0, 1.0));
        // glTF reads roughness from green and metalness from blue of one texture.
        let rough = tex("roughness", Some(1));
        let metal = tex("metalness", Some(2));
        full.metallic_roughness_tex = match (rough, metal) {
            (Some(r), Some(mt)) if r.image == mt.image && r.uv == mt.uv => Some(r),
            (Some(r), Some(_)) => {
                left_out();
                Some(r)
            }
            (r, mt) => r.or(mt),
        };
        if let Some(s) = specular {
            full.specular = Some(Specular {
                factor: Factor::of(1.0),
                color: rgb(s).map(|v| v.clamp(0.0, 1.0)),
                color_tex: tex("specular", None),
            });
        }
    } else {
        // A specular color and power alone: glossy plastic, its roughness from the power
        // (Blinn-Phong's exponent to a microfacet roughness), its highlights' color the
        // specular extension's.
        full.metallic = 0.0;
        full.roughness = f32_of((2.0 / (power + 2.0)).powf(0.25).clamp(0.0, 1.0));
        let s = specular.unwrap_or([0.0; 4]);
        let top = s[..3].iter().copied().fold(0.0f64, f64::max);
        let spec_tex = tex("specular", None);
        full.specular = Some(if top > 0.0 || spec_tex.is_some() {
            Specular {
                factor: Factor::of(1.0),
                color: rgb(s).map(|v| v.clamp(0.0, 1.0)),
                color_tex: spec_tex,
            }
        } else {
            Specular {
                factor: Factor::of(0.0),
                color: [1.0; 3],
                color_tex: None,
            }
        });
    }
    let emission = col("emission").unwrap_or([0.0; 4]);
    full.emissive = rgb(emission).map(|v| v.clamp(0.0, 1.0));
    full.emissive_tex = tex("emission", None);
    if full.emissive_tex.is_some() && col("emission").is_none() {
        // A texture alone gives the emission; the color it's multiplied by is white.
        full.emissive = [1.0; 3];
    }
    full.normal_tex = tex("normal", None);
    full.occlusion_tex = tex("occlusion", Some(0));
    if let Some(tr) = col("transparency") {
        let f = (tr[0] + tr[1] + tr[2]) / 3.0;
        if f > 0.0 {
            full.transmission = Some(Factor {
                factor: f32_of(f.clamp(0.0, 1.0)),
                tex: tex("transparency", Some(0)),
            });
        }
    }
    if let Some(cc) = col("clearcoat") {
        let f = cc[..3].iter().copied().fold(0.0f64, f64::max);
        if f > 0.0 {
            full.clearcoat = Some(Clearcoat {
                factor: Factor {
                    factor: f32_of(f.clamp(0.0, 1.0)),
                    tex: tex("clearcoat", Some(0)),
                },
                roughness: Factor {
                    factor: f32_of(param(t, m, "clearcoat_roughness").unwrap_or(0.0).clamp(0.0, 1.0)),
                    tex: tex("clearcoat_roughness", Some(1)),
                },
                normal: tex("clearcoat_normal", None),
            });
        }
    }
    if let Some(sh) = col("sheen")
        && sh[..3].iter().any(|&v| v > 0.0)
    {
        full.sheen = Some(Sheen {
            color: rgb(sh).map(|v| v.clamp(0.0, 1.0)),
            color_tex: tex("sheen", None),
            roughness: Factor {
                factor: f32_of(param(t, m, "sheen_roughness").unwrap_or(0.0).clamp(0.0, 1.0)),
                tex: tex("sheen_roughness", Some(3)),
            },
        });
    }
    // glTF's index of refraction is 1 or more; left out, glTF takes 1.5.
    if let Some(ior) = param(t, m, "ior").filter(|&v| v >= 1.0) {
        full.ior = Some(f32_of(ior));
    }
    if c.keep.extras {
        let mut e = Map::new();
        let mut colors = Map::new();
        let mut params = Map::new();
        for k in t.kids(m) {
            let Some(attrib) = t.prop_str(k, "attrib") else {
                continue;
            };
            match t.ident(k) {
                "Color" => {
                    colors.insert(attrib.into(), json!(t.values(k).0));
                }
                "Param" => {
                    if let Some(&v) = t.values(k).0.first() {
                        params.insert(attrib.into(), json!(v));
                    }
                }
                _ => {}
            }
        }
        if !colors.is_empty() {
            e.insert("colors".into(), Value::Object(colors));
        }
        if !params.is_empty() {
            e.insert("params".into(), Value::Object(params));
        }
        if !record.is_empty() {
            e.insert("textures".into(), json!(record));
        }
        if wrong.get() > 0 {
            e.insert("texturesLeftOut".into(), json!(wrong.get()));
        }
        if !e.is_empty() {
            full.extras = Some(json!({ "opengex": e }));
        }
    }
    c.materials.push(full);
    c.material_uvs.push(uvs);
    c.materials.len() - 1
}
