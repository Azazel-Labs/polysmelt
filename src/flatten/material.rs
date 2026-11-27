//! Materials and their textures, with each texture's coordinate set and transform.

use serde_json::Value;

use super::{Reader, at, f32s, index};
use crate::gltf as container;
#[cfg(doc)]
use crate::mesh::Prim;
use crate::mesh::{
    Anisotropy, Clearcoat, Extensions, Factor, Image, Iridescence, Material, Mesh, Sampling, Sheen, Specular, Texture,
    UvTransform, Volume,
};
use crate::num::f32_of;

impl Reader<'_> {
    /// The texture coordinate set a material's color texture reads, and its
    /// `KHR_texture_transform` as a 2x3 matrix (row-major).
    pub(super) fn uv_set(&self, material: Option<usize>) -> (u64, Option<[f32; 6]>) {
        material
            .and_then(|m| self.materials_json.get(m))
            .and_then(color_texture)
            .map_or((0, None), mapping)
    }

    /// The mesh material for glTF material `i` (or glTF's default material), where its mesh is drawn `scale`
    /// times its size: a volume's thickness is in the mesh's space, so each scale makes a material of its own.
    pub(super) fn material(&mut self, i: Option<usize>, scale: f32, mesh: &mut Mesh) -> usize {
        #[allow(clippy::float_cmp, reason = "a scale within rounding of 1 was made exactly 1")]
        let scaled = i.filter(|&i| scale != 1.0 && self.materials_json.get(i).is_some_and(thick));
        let slot = match (scaled, i) {
            (Some(i), _) => self.scaled_materials.get(&(i, scale.to_bits())).copied(),
            (None, Some(i)) if i < self.materials.len() => self.materials[i],
            _ => self.default_material,
        };
        if let Some(s) = slot {
            return s;
        }
        let mut m = Material {
            color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            ..Material::plain("", [1.0; 4])
        };
        if let Some(d) = i.and_then(|i| self.materials_json.get(i)) {
            d.get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .clone_into(&mut m.name);
            let ext = d.get("extensions");
            let sg = ext.and_then(|e| e.get("KHR_materials_pbrSpecularGlossiness"));
            match (d.get("pbrMetallicRoughness"), sg) {
                (None, Some(sg)) => {
                    m.color = f32s(sg.get("diffuseFactor"), [1.0; 4]);
                    m.metallic = 0.0;
                    m.roughness = 1.0 - f32_of(sg.get("glossinessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
                (pbr, _) => {
                    let pbr = pbr.unwrap_or(&Value::Null);
                    m.color = f32s(pbr.get("baseColorFactor"), [1.0; 4]);
                    m.metallic = f32_of(pbr.get("metallicFactor").and_then(Value::as_f64).unwrap_or(1.0));
                    m.roughness = f32_of(pbr.get("roughnessFactor").and_then(Value::as_f64).unwrap_or(1.0));
                }
            }
            let strength = ext
                .and_then(|e| at(e, &["KHR_materials_emissive_strength", "emissiveStrength"]))
                .and_then(Value::as_f64)
                .map_or(1.0, f32_of);
            m.emissive = f32s(d.get("emissiveFactor"), [0.0; 3]).map(|e| e * strength);
            m.double_sided = d.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
            match d.get("alphaMode").and_then(Value::as_str) {
                Some("BLEND") => {}
                Some("MASK") => {
                    let cutoff = d.get("alphaCutoff").and_then(Value::as_f64).map_or(0.5, f32_of);
                    m.alpha_cutoff = Some(if cutoff.is_finite() { cutoff.max(0.0) } else { 0.5 });
                }
                _ => m.color[3] = 1.0,
            }
            m.unlit = ext.and_then(|e| e.get("KHR_materials_unlit")).is_some();
            // `Prim::uvs` are the color texture's coordinates with its transform baked in: the
            // other textures read them through what's left of their own transform, or read
            // another set.
            let color = color_texture(d);
            let uv = color.map_or((0, None), mapping);
            let mut take = |info: Option<&Value>| self.texture_info(info?, uv, mesh);
            let mut slot = |info: Option<&Value>, image: &mut Option<usize>, sampling: &mut Sampling| {
                if let Some(t) = take(info) {
                    *image = Some(t.image);
                    *sampling = t.sampling;
                }
            };
            let s = &mut m.sampling;
            slot(color, &mut m.color_image, &mut s.color);
            slot(d.get("normalTexture"), &mut m.normal_image, &mut s.normal);
            slot(d.get("occlusionTexture"), &mut m.occlusion_image, &mut s.occlusion);
            slot(d.get("emissiveTexture"), &mut m.emissive_image, &mut s.emissive);
            slot(
                at(d, &["pbrMetallicRoughness", "metallicRoughnessTexture"]),
                &mut m.metallic_roughness_image,
                &mut s.metallic_roughness,
            );
            if m.normal_image.is_some() {
                m.normal_scale = at(d, &["normalTexture", "scale"])
                    .and_then(Value::as_f64)
                    .map_or(1.0, f32_of);
            }
            if m.occlusion_image.is_some() {
                m.occlusion_strength = at(d, &["occlusionTexture", "strength"])
                    .and_then(Value::as_f64)
                    .map_or(1.0, f32_of);
            }
            if let Some(ext) = ext {
                m.extensions = self.extensions(ext, uv, mesh).map(Box::new);
            }
        }
        if scaled.is_some()
            && let Some(volume) = m.extensions.as_mut().and_then(|e| e.volume.as_mut())
        {
            volume.thickness.factor *= scale;
        }
        mesh.materials.push(m);
        let s = mesh.materials.len() - 1;
        match (scaled, i) {
            (Some(i), _) => {
                self.scaled_materials.insert((i, scale.to_bits()), s);
            }
            (None, Some(i)) if i < self.materials.len() => self.materials[i] = Some(s),
            _ => self.default_material = Some(s),
        }
        s
    }

    /// A texture info's image and how it's sampled, given the color texture's coordinate set and
    /// transform `color` (which [`Prim::uvs`] have baked in); `None` when the image can't be
    /// made, or the texture reads the color's coordinates through a transform that can't be
    /// written relative to the color's (a shear).
    fn texture_info(&mut self, info: &Value, color: (u64, Option<[f32; 6]>), mesh: &mut Mesh) -> Option<Texture> {
        let t = index(info, "index")?;
        let (set, own) = mapping(info);
        let (uv_set, transform) = if set != color.0 {
            (Some(usize::try_from(set).ok()?), uv_transform(info))
        } else if color.1.is_none() {
            (None, uv_transform(info))
        } else if own == color.1 {
            (None, None)
        } else {
            let relative = relative(own.unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), color.1?)?;
            (None, Some(uv_decompose(relative)?))
        };
        let image = self.texture(t, mesh)?;
        let sampler = self
            .textures
            .get(t)
            .and_then(|tex| index(tex, "sampler"))
            .and_then(|s| self.samplers.get(s));
        let num = |k: &str| {
            sampler
                .and_then(|s| s.get(k))
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
        };
        Some(Texture {
            image,
            sampling: Sampling {
                uv_set,
                transform: transform.filter(|x| *x != UvTransform::IDENTITY),
                wrap: [num("wrapS").unwrap_or(10497), num("wrapT").unwrap_or(10497)],
                filter: [num("magFilter"), num("minFilter")],
            },
        })
    }

    /// The `KHR_materials_*` extensions in a material's `extensions`, with their textures;
    /// `None` when it has none of them.
    fn extensions(&mut self, ext: &Value, uv: (u64, Option<[f32; 6]>), mesh: &mut Mesh) -> Option<Extensions> {
        let num = |e: &Value, k: &str, default: f32| e.get(k).and_then(Value::as_f64).map_or(default, f32_of);
        let mut tex = |e: &Value, k: &str| e.get(k).and_then(|info| self.texture_info(info, uv, mesh));
        let get = |name: &str| ext.get(name).filter(|e| e.is_object());
        let mut out = Extensions {
            ior: get("KHR_materials_ior").map(|e| num(e, "ior", 1.5)),
            dispersion: get("KHR_materials_dispersion").map(|e| num(e, "dispersion", 0.0)),
            ..Extensions::default()
        };
        if let Some(e) = get("KHR_materials_specular") {
            out.specular = Some(Specular {
                factor: Factor {
                    factor: num(e, "specularFactor", 1.0),
                    tex: tex(e, "specularTexture"),
                },
                color: f32s(e.get("specularColorFactor"), [1.0; 3]),
                color_tex: tex(e, "specularColorTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_transmission") {
            out.transmission = Some(Factor {
                factor: num(e, "transmissionFactor", 0.0),
                tex: tex(e, "transmissionTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_clearcoat") {
            out.clearcoat = Some(Clearcoat {
                factor: Factor {
                    factor: num(e, "clearcoatFactor", 0.0),
                    tex: tex(e, "clearcoatTexture"),
                },
                roughness: Factor {
                    factor: num(e, "clearcoatRoughnessFactor", 0.0),
                    tex: tex(e, "clearcoatRoughnessTexture"),
                },
                normal: tex(e, "clearcoatNormalTexture"),
            });
        }
        if let Some(e) = get("KHR_materials_sheen") {
            out.sheen = Some(Sheen {
                color: f32s(e.get("sheenColorFactor"), [0.0; 3]),
                color_tex: tex(e, "sheenColorTexture"),
                roughness: Factor {
                    factor: num(e, "sheenRoughnessFactor", 0.0),
                    tex: tex(e, "sheenRoughnessTexture"),
                },
            });
        }
        if let Some(e) = get("KHR_materials_volume") {
            out.volume = Some(Volume {
                thickness: Factor {
                    factor: num(e, "thicknessFactor", 0.0),
                    tex: tex(e, "thicknessTexture"),
                },
                attenuation_distance: e
                    .get("attenuationDistance")
                    .and_then(Value::as_f64)
                    .map(f32_of)
                    .filter(|d| d.is_finite() && *d > 0.0),
                attenuation_color: f32s(e.get("attenuationColor"), [1.0; 3]),
            });
        }
        if let Some(e) = get("KHR_materials_anisotropy") {
            out.anisotropy = Some(Anisotropy {
                strength: Factor {
                    factor: num(e, "anisotropyStrength", 0.0),
                    tex: tex(e, "anisotropyTexture"),
                },
                rotation: num(e, "anisotropyRotation", 0.0),
            });
        }
        if let Some(e) = get("KHR_materials_iridescence") {
            out.iridescence = Some(Iridescence {
                factor: Factor {
                    factor: num(e, "iridescenceFactor", 0.0),
                    tex: tex(e, "iridescenceTexture"),
                },
                ior: num(e, "iridescenceIor", 1.3),
                thickness: [
                    num(e, "iridescenceThicknessMinimum", 100.0),
                    num(e, "iridescenceThicknessMaximum", 400.0),
                ],
                thickness_tex: tex(e, "iridescenceThicknessTexture"),
            });
        }
        (out != Extensions::default()).then_some(out)
    }

    /// The mesh image for glTF texture `t`: its source, or failing that one an extension
    /// names (`EXT_texture_webp`, `KHR_texture_basisu`, ...), if an image can be made of it.
    fn texture(&mut self, t: usize, mesh: &mut Mesh) -> Option<usize> {
        let tex = self.textures.get(t)?;
        let mut sources: Vec<usize> = index(tex, "source").into_iter().collect();
        if let Some(ext) = tex.get("extensions").and_then(Value::as_object) {
            sources.extend(ext.values().filter_map(|e| index(e, "source")));
        }
        sources.into_iter().find_map(|s| self.image(s, mesh))
    }

    /// The mesh image for glTF image `s`, when it's PNG or JPEG, or a format a feature
    /// re-encodes (WebP, TGA, BMP, GIF, TIFF, PSD, DDS). A WebP image is kept as it is, and
    /// re-encoded only when the mesh is written.
    fn image(&mut self, s: usize, mesh: &mut Mesh) -> Option<usize> {
        if let Some(done) = *self.images.get(s)? {
            return done;
        }
        let img = self.images_json.get(s)?;
        // TGA has no signature: its MIME type, or a file name, says what it is.
        let name = match img.get("mimeType").and_then(Value::as_str) {
            Some("image/x-tga" | "image/tga" | "image/x-targa" | "image/targa") => "image.tga",
            _ => img
                .get("uri")
                .and_then(Value::as_str)
                .filter(|u| !u.starts_with("data:"))
                .unwrap_or(""),
        };
        let name = std::path::PathBuf::from(name);
        let bytes = if let Some(v) = index(img, "bufferView") {
            self.view_shared(v).ok()
        } else {
            // A file the loader didn't find stays a URI, and isn't drawn.
            img.get("uri")
                .and_then(Value::as_str)
                .filter(|u| u.starts_with("data:"))
                .and_then(|u| container::data_uri(u).ok())
                .map(|(_, b)| b.into())
        };
        let out = bytes.and_then(|b: crate::Bytes| Image::kept_named(b, &name)).map(|i| {
            mesh.images.push(i);
            mesh.images.len() - 1
        });
        self.images[s] = Some(out);
        out
    }
}

/// Whether a material is a volume with a thickness (`KHR_materials_volume`), which its mesh's node scales.
pub(super) fn thick(material: &Value) -> bool {
    at(material, &["extensions", "KHR_materials_volume", "thicknessFactor"])
        .and_then(Value::as_f64)
        .is_some_and(|t| t > 0.0)
}

/// The texture coordinate set a texture info reads, and its `KHR_texture_transform` as a 2x3
/// matrix (row-major).
fn mapping(info: &Value) -> (u64, Option<[f32; 6]>) {
    let mut set = info.get("texCoord").and_then(Value::as_u64).unwrap_or(0);
    let Some(t) = info.get("extensions").and_then(|e| e.get("KHR_texture_transform")) else {
        return (set, None);
    };
    if let Some(s) = t.get("texCoord").and_then(Value::as_u64) {
        set = s;
    }
    let [ox, oy] = f32s(t.get("offset"), [0.0; 2]);
    let [sx, sy] = f32s(t.get("scale"), [1.0; 2]);
    let r = f32_of(t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0));
    let (sin, cos) = r.sin_cos();
    // translation * rotation * scale, as the extension defines them.
    (set, Some([cos * sx, sin * sy, ox, -sin * sx, cos * sy, oy]))
}

/// A texture info's `KHR_texture_transform` as it's written; `None` without one.
fn uv_transform(info: &Value) -> Option<UvTransform> {
    let t = info.get("extensions")?.get("KHR_texture_transform")?;
    Some(UvTransform {
        offset: f32s(t.get("offset"), [0.0; 2]),
        rotation: f32_of(t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0)),
        scale: f32s(t.get("scale"), [1.0; 2]),
    })
}

/// The 2x3 transform (row-major) that takes coordinates `color` has moved to where `own` moves
/// them: `own * color^-1`; `None` when `color` collapses them.
fn relative(own: [f32; 6], color: [f32; 6]) -> Option<[f32; 6]> {
    let [a, b, e, c, d, f] = color.map(f64::from);
    let det = a * d - b * c;
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    // The inverse: its linear part, then the translation it undoes.
    let (ia, ib, ic, id) = (d / det, -b / det, -c / det, a / det);
    let (ie, if_) = (-(ia * e + ib * f), -(ic * e + id * f));
    let [p, q, s, r, t, u] = own.map(f64::from);
    let out = [
        p * ia + q * ic,
        p * ib + q * id,
        p * ie + q * if_ + s,
        r * ia + t * ic,
        r * ib + t * id,
        r * ie + t * if_ + u,
    ];
    #[allow(clippy::cast_possible_truncation, reason = "back to the f32s the matrices came in")]
    Some(out.map(|v| v as f32))
}

/// A 2x3 transform (row-major) as `KHR_texture_transform` writes one, `translate * rotate *
/// scale`; `None` when it shears, which that can't write.
fn uv_decompose(m: [f32; 6]) -> Option<UvTransform> {
    let [a, b, e, c, d, f] = m;
    // The first column is (cos, -sin) * scale x; the second (sin, cos) * scale y.
    let sx = a.hypot(c);
    if sx == 0.0 || !sx.is_finite() {
        return None;
    }
    let rotation = (-c).atan2(a);
    let (sin, cos) = rotation.sin_cos();
    let sy = b * sin + d * cos;
    let tolerance = 1e-4 * sy.abs().max(1.0);
    if (b - sin * sy).abs() > tolerance || (d - cos * sy).abs() > tolerance {
        return None;
    }
    Some(UvTransform {
        offset: [e, f],
        rotation,
        scale: [sx, sy],
    })
}

/// A material's color texture info: `baseColorTexture`, or the specular-glossiness diffuse.
fn color_texture(m: &Value) -> Option<&Value> {
    m.get("pbrMetallicRoughness")
        .and_then(|p| p.get("baseColorTexture"))
        .or_else(|| {
            m.get("extensions")?
                .get("KHR_materials_pbrSpecularGlossiness")?
                .get("diffuseTexture")
        })
}
