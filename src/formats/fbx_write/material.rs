//! Materials as Phong materials with glTF's metal/rough model alongside, under `glTF|` properties holding the
//! glTF values as they are (linear colors); textures as file textures with their images embedded in videos.

use serde_json::Value;

use super::records::{Node, p, p_f64, p_int, p_str, p_vec, props70};
use super::{Writer, fbx_name, index, list, number, numbers, to_srgb};

/// The glTF sampler constants for repeating and clamping.
const REPEAT: i64 = 10497;
const CLAMP: i64 = 33071;

/// A material extension's object, when the material has it.
fn ext<'a>(mat: &'a Value, name: &str) -> Option<&'a Value> {
    mat.get("extensions")?.get(name)
}

/// A texture reference's texture and UV set (as `KHR_texture_transform` may change it).
fn tex_ref(info: &Value) -> Option<(usize, usize)> {
    let t = index(info, "index")?;
    let uv = info
        .pointer("/extensions/KHR_texture_transform/texCoord")
        .and_then(Value::as_u64)
        .or_else(|| info.get("texCoord").and_then(Value::as_u64))
        .and_then(|u| usize::try_from(u).ok())
        .unwrap_or(0);
    Some((t, uv))
}

/// A file name for image `i`: its name (or `image_i`) with the extension its type takes.
fn image_file(name: Option<&str>, i: usize, mime: Option<&str>) -> String {
    let stem: String = name
        .filter(|n| !n.is_empty())
        .map_or_else(|| format!("image_{i}"), |n| n.replace(['/', '\\', ':', '\0'], "_"));
    let ext = match mime {
        Some("image/jpeg") => "jpg",
        Some("image/webp") => "webp",
        Some("image/ktx2") => "ktx2",
        _ => "png",
    };
    let lower = stem.to_ascii_lowercase();
    if [".png", ".jpg", ".jpeg", ".webp", ".ktx2"]
        .iter()
        .any(|e| lower.ends_with(e))
    {
        stem
    } else {
        format!("{stem}.{ext}")
    }
}

impl Writer<'_> {
    /// Material `m`'s object, written the first time it's asked for. Primitives without a material (`None`) get
    /// glTF's default one, white and fully metallic and rough, rather than whatever default a reader has.
    pub(super) fn material(&mut self, m: Option<usize>) -> i64 {
        if let Some(&id) = self.materials.get(&m) {
            return id;
        }
        let mat = m.map_or(&Value::Null, |m| &list(self.json, "materials")[m]);
        let id = self.id();
        self.materials.insert(m, id);
        let name = mat
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .map_or_else(
                || m.map_or_else(|| "default".to_owned(), |m| format!("material_{m}")),
                str::to_owned,
            );
        let pbr = mat.get("pbrMetallicRoughness").unwrap_or(&Value::Null);
        // A specular/glossiness material, without metal/rough values of its own, gives its diffuse as the base
        // color, no metal, and roughness from its glossiness.
        let gloss = ext(mat, "KHR_materials_pbrSpecularGlossiness").filter(|_| pbr.is_null());
        let (base, metallic, roughness, color_texture) = match gloss {
            Some(sg) => (
                numbers(sg, "diffuseFactor", [1.0; 4]),
                0.0,
                1.0 - number(sg, "glossinessFactor").unwrap_or(1.0),
                sg.get("diffuseTexture"),
            ),
            None => (
                numbers(pbr, "baseColorFactor", [1.0; 4]),
                number(pbr, "metallicFactor").unwrap_or(1.0),
                number(pbr, "roughnessFactor").unwrap_or(1.0),
                pbr.get("baseColorTexture"),
            ),
        };
        let specular = gloss.map_or([0.2 + 0.8 * metallic.clamp(0.0, 1.0); 3], |sg| {
            numbers(sg, "specularFactor", [1.0; 3]).map(to_srgb)
        });
        let emissive = numbers(mat, "emissiveFactor", [0.0; 3]);
        let strength = ext(mat, "KHR_materials_emissive_strength")
            .and_then(|e| number(e, "emissiveStrength"))
            .unwrap_or(1.0);
        let mode = match mat.get("alphaMode").and_then(Value::as_str) {
            Some("MASK") => 1,
            Some("BLEND") => 2,
            _ => 0,
        };
        let srgb = |c: [f64; 3]| c.map(to_srgb);
        let rgb = [base[0], base[1], base[2]];
        let alpha = base[3].clamp(0.0, 1.0);
        let shininess = ((1.0 - roughness.clamp(0.0, 1.0)) * 10.0).powi(2);
        let mut props = vec![
            // The classic model, for tools that know only it.
            p_vec("DiffuseColor", "Color", "", "A", srgb(rgb)),
            p_f64("DiffuseFactor", "Number", "A", 1.0),
            p_vec("EmissiveColor", "Color", "", "A", srgb(emissive)),
            p_f64("EmissiveFactor", "Number", "A", strength),
            p_vec("SpecularColor", "Color", "", "A", specular),
            p_f64("Shininess", "double", "", shininess),
            p_f64("ShininessExponent", "double", "", shininess),
            p_f64("TransparencyFactor", "Number", "A", 1.0 - alpha),
            p_f64("Opacity", "double", "", alpha),
            // glTF's own model.
            p_vec("glTF|baseColor", "ColorRGB", "Color", "", rgb),
            p_f64("glTF|baseWeight", "double", "", 1.0),
            p_f64("glTF|opacity", "double", "", base[3]),
            p_f64("glTF|metalness", "double", "", metallic),
            p_f64("glTF|roughness", "double", "", roughness),
            p_vec("glTF|emissive", "ColorRGB", "Color", "", emissive),
            p_f64("glTF|emissiveIntensity", "double", "", strength),
            p_int("glTF|alphaMode", "enum", mode),
            p_f64(
                "glTF|alphaCutoff",
                "double",
                "",
                number(mat, "alphaCutoff").unwrap_or(0.5),
            ),
            p_int(
                "glTF|doubleSided",
                "bool",
                i32::from(mat.get("doubleSided").and_then(Value::as_bool) == Some(true)),
            ),
            p_int(
                "glTF|unlit",
                "bool",
                i32::from(ext(mat, "KHR_materials_unlit").is_some()),
            ),
        ];
        if let Some(n) = mat.get("normalTexture").and_then(|t| number(t, "scale")) {
            props.push(p_f64("glTF|normalScale", "double", "", n));
        }
        if let Some(o) = mat.get("occlusionTexture").and_then(|t| number(t, "strength")) {
            props.push(p_f64("glTF|occlusionStrength", "double", "", o));
        }
        if let Some(e) = ext(mat, "KHR_materials_ior") {
            props.push(p_f64("glTF|ior", "double", "", number(e, "ior").unwrap_or(1.5)));
        }
        if let Some(e) = ext(mat, "KHR_materials_transmission") {
            props.push(p_f64(
                "glTF|transmission",
                "double",
                "",
                number(e, "transmissionFactor").unwrap_or(0.0),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_clearcoat") {
            props.push(p_f64(
                "glTF|coat",
                "double",
                "",
                number(e, "clearcoatFactor").unwrap_or(0.0),
            ));
            props.push(p_f64(
                "glTF|coatRoughness",
                "double",
                "",
                number(e, "clearcoatRoughnessFactor").unwrap_or(0.0),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_sheen") {
            props.push(p_f64("glTF|sheen", "double", "", 1.0));
            props.push(p_vec(
                "glTF|sheenColor",
                "ColorRGB",
                "Color",
                "",
                numbers(e, "sheenColorFactor", [0.0; 3]),
            ));
            props.push(p_f64(
                "glTF|sheenRoughness",
                "double",
                "",
                number(e, "sheenRoughnessFactor").unwrap_or(0.0),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_specular") {
            props.push(p_f64(
                "glTF|specular",
                "double",
                "",
                number(e, "specularFactor").unwrap_or(1.0),
            ));
            props.push(p_vec(
                "glTF|specularColor",
                "ColorRGB",
                "Color",
                "",
                numbers(e, "specularColorFactor", [1.0; 3]),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_volume") {
            props.push(p_f64(
                "glTF|thickness",
                "double",
                "",
                number(e, "thicknessFactor").unwrap_or(0.0),
            ));
            if let Some(d) = number(e, "attenuationDistance").filter(|d| d.is_finite()) {
                props.push(p_f64("glTF|attenuationDistance", "double", "", d));
            }
            props.push(p_vec(
                "glTF|attenuationColor",
                "ColorRGB",
                "Color",
                "",
                numbers(e, "attenuationColor", [1.0; 3]),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_anisotropy") {
            props.push(p_f64(
                "glTF|anisotropy",
                "double",
                "",
                number(e, "anisotropyStrength").unwrap_or(0.0),
            ));
            props.push(p_f64(
                "glTF|anisotropyRotation",
                "double",
                "",
                number(e, "anisotropyRotation").unwrap_or(0.0),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_iridescence") {
            props.push(p_f64(
                "glTF|iridescence",
                "double",
                "",
                number(e, "iridescenceFactor").unwrap_or(0.0),
            ));
            props.push(p_f64(
                "glTF|iridescenceIor",
                "double",
                "",
                number(e, "iridescenceIor").unwrap_or(1.3),
            ));
            props.push(p_f64(
                "glTF|iridescenceThicknessMin",
                "double",
                "",
                number(e, "iridescenceThicknessMinimum").unwrap_or(100.0),
            ));
            props.push(p_f64(
                "glTF|iridescenceThicknessMax",
                "double",
                "",
                number(e, "iridescenceThicknessMaximum").unwrap_or(400.0),
            ));
        }
        if let Some(e) = ext(mat, "KHR_materials_dispersion") {
            props.push(p_f64(
                "glTF|dispersion",
                "double",
                "",
                number(e, "dispersion").unwrap_or(0.0),
            ));
        }
        let node = Node::new("Material")
            .i64(id)
            .str(&fbx_name(&name, "Material"))
            .str("")
            .child("Version", |n| n.i32(102))
            .child("ShadingModel", |n| n.str("phong"))
            .child("MultiLayer", |n| n.i32(0))
            .with(props70(props));
        self.object("Material", node);

        // Textures, each connected to the property it drives; the first of a property is what a reader takes.
        let mut links: Vec<(&Value, &'static str)> = Vec::new();
        if let Some(t) = color_texture {
            links.push((t, "DiffuseColor"));
            if mode != 0 {
                links.push((t, "TransparencyFactor"));
            }
        }
        for (key, prop) in [
            ("normalTexture", "NormalMap"),
            ("emissiveTexture", "EmissiveColor"),
            ("occlusionTexture", "glTF|ambientOcclusion"),
        ] {
            if let Some(t) = mat.get(key) {
                links.push((t, prop));
            }
        }
        if let Some(t) = pbr.get("metallicRoughnessTexture") {
            links.push((t, "glTF|metallicRoughness"));
        }
        for (e, key, prop) in [
            (
                "KHR_materials_transmission",
                "transmissionTexture",
                "glTF|transmissionMap",
            ),
            ("KHR_materials_clearcoat", "clearcoatTexture", "glTF|coatMap"),
            (
                "KHR_materials_clearcoat",
                "clearcoatRoughnessTexture",
                "glTF|coatRoughnessMap",
            ),
            (
                "KHR_materials_clearcoat",
                "clearcoatNormalTexture",
                "glTF|coatNormalMap",
            ),
            ("KHR_materials_sheen", "sheenColorTexture", "glTF|sheenColorMap"),
            ("KHR_materials_sheen", "sheenRoughnessTexture", "glTF|sheenRoughnessMap"),
            ("KHR_materials_specular", "specularTexture", "glTF|specularMap"),
            (
                "KHR_materials_specular",
                "specularColorTexture",
                "glTF|specularColorMap",
            ),
            ("KHR_materials_volume", "thicknessTexture", "glTF|thicknessMap"),
            ("KHR_materials_anisotropy", "anisotropyTexture", "glTF|anisotropyMap"),
            ("KHR_materials_iridescence", "iridescenceTexture", "glTF|iridescenceMap"),
            (
                "KHR_materials_iridescence",
                "iridescenceThicknessTexture",
                "glTF|iridescenceThicknessMap",
            ),
        ] {
            if let Some(t) = ext(mat, e).and_then(|e| e.get(key)) {
                links.push((t, prop));
            }
        }
        for (info, prop) in links {
            if let Some(t) = self.texture(info) {
                self.op(t, id, prop);
            }
        }
        id
    }

    /// The texture object a material's texture reference makes, written the first time it's asked for; `None` when
    /// it names no image.
    fn texture(&mut self, info: &Value) -> Option<i64> {
        let (t, uv) = tex_ref(info)?;
        let texture = list(self.json, "textures").get(t)?;
        let source = index(texture, "source")
            .or_else(|| {
                texture
                    .pointer("/extensions/EXT_texture_webp/source")
                    .and_then(as_index)
            })
            .or_else(|| {
                texture
                    .pointer("/extensions/KHR_texture_basisu/source")
                    .and_then(as_index)
            })
            .filter(|&s| s < list(self.json, "images").len());
        let image = source?;
        let sampler = index(texture, "sampler").and_then(|s| list(self.json, "samplers").get(s));
        let gl = |k: &str| sampler.and_then(|s| s.get(k)).and_then(Value::as_i64);
        let (wrap_s, wrap_t) = (gl("wrapS").unwrap_or(REPEAT), gl("wrapT").unwrap_or(REPEAT));
        let tt = info.pointer("/extensions/KHR_texture_transform");
        let offset = tt.map_or([0.0; 2], |t| numbers(t, "offset", [0.0; 2]));
        let rotation = tt.and_then(|t| number(t, "rotation")).unwrap_or(0.0);
        let scale = tt.map_or([1.0; 2], |t| numbers(t, "scale", [1.0; 2]));
        let key = format!(
            "{image} {uv} {wrap_s} {wrap_t} {:?} {:?} {offset:?} {rotation} {scale:?}",
            gl("magFilter"),
            gl("minFilter")
        );
        if let Some(&id) = self.textures.get(&key) {
            return Some(id);
        }
        let video = self.video(image);
        let id = self.id();
        self.textures.insert(key, id);
        let img = &list(self.json, "images")[image];
        let handle = self.doc.images().nth(image);
        let file = image_file(
            img.get("name").and_then(Value::as_str),
            image,
            handle.as_ref().and_then(crate::images::ImageHandle::mime),
        );
        let file = handle
            .as_ref()
            .filter(|h| h.is_pending())
            .and_then(crate::images::ImageHandle::reference)
            .map_or(file, str::to_owned);
        let name = texture
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .map_or_else(|| format!("texture_{t}"), str::to_owned);
        // KHR_texture_transform maps UVs (V down) as `T * R * S`; FBX's texture translation is on V-up UVs.
        let (sin, cos) = rotation.sin_cos();
        let tu = offset[0] + scale[1] * sin;
        let tv = 1.0 - offset[1] - scale[1] * cos;
        let transformed = tt.is_some() && (offset, rotation, scale) != ([0.0; 2], 0.0, [1.0; 2]);
        let (tu, tv) = if transformed { (tu, tv) } else { (0.0, 0.0) };
        let mut props = vec![
            p_str("UVSet", &format!("TEXCOORD_{uv}")),
            p_int("UseMaterial", "bool", 1),
            p_int("WrapModeU", "enum", i32::from(wrap_s == CLAMP)),
            p_int("WrapModeV", "enum", i32::from(wrap_t == CLAMP)),
        ];
        if transformed {
            props.push(p_vec("Translation", "Vector", "", "A", [tu, tv, 0.0]));
            props.push(p_vec("Rotation", "Vector", "", "A", [0.0, 0.0, rotation.to_degrees()]));
            props.push(p_vec("Scaling", "Vector", "", "A", [scale[0], scale[1], 1.0]));
        }
        // What FBX's wrap modes and textures can't say (mirroring, filters), as glTF says it.
        for (k, v) in [("glTF|wrapS", gl("wrapS")), ("glTF|wrapT", gl("wrapT"))]
            .into_iter()
            .chain([("glTF|magFilter", gl("magFilter")), ("glTF|minFilter", gl("minFilter"))])
        {
            if let Some(v) = v.and_then(|v| i32::try_from(v).ok()) {
                props.push(p(k, "int", "Integer", "").i32(v));
            }
        }
        let video_name = fbx_name(
            &handle
                .and_then(|h| h.name())
                .map_or_else(|| format!("image_{image}"), str::to_owned),
            "Video",
        );
        let node = Node::new("Texture")
            .i64(id)
            .str(&fbx_name(&name, "Texture"))
            .str("")
            .child("Type", |n| n.str("TextureVideoClip"))
            .child("Version", |n| n.i32(202))
            .child("TextureName", |n| n.str(&fbx_name(&name, "Texture")))
            .with(props70(props))
            .child("Media", |n| n.str(&video_name))
            .child("FileName", |n| n.str(&file))
            .child("RelativeFilename", |n| n.str(&file))
            .child("ModelUVTranslation", |n| n.f64(0.0).f64(0.0))
            .child("ModelUVScaling", |n| n.f64(1.0).f64(1.0))
            .child("Texture_Alpha_Source", |n| n.str("None"))
            .child("Cropping", |n| n.i32(0).i32(0).i32(0).i32(0));
        self.object("Texture", node);
        if let Some(v) = video {
            self.oo(v, id);
        }
        Some(id)
    }

    /// The video holding image `i`, written the first time it's asked for: its bytes embedded, or for an image
    /// not yet read, only its file name. `None` when its bytes can't be read.
    fn video(&mut self, i: usize) -> Option<i64> {
        if let Some(&v) = self.videos.get(&i) {
            return v;
        }
        let Some(handle) = self.doc.images().nth(i) else {
            self.videos.insert(i, None);
            return None;
        };
        // An image whose bytes can't be had keeps its file name, as a pending one does.
        let bytes = if handle.is_pending() {
            None
        } else {
            handle.load(&crate::assets::Memory::default()).ok().map(|i| i.bytes)
        };
        let file = image_file(handle.name(), i, handle.mime());
        let file = handle.reference().map_or(file, str::to_owned);
        let name = handle.name().map_or_else(|| format!("image_{i}"), str::to_owned);
        let id = self.id();
        let mut node = Node::new("Video")
            .i64(id)
            .str(&fbx_name(&name, "Video"))
            .str("Clip")
            .child("Type", |n| n.str("Clip"))
            .with(props70(vec![p("Path", "KString", "XRefUrl", "").str(&file)]))
            .child("UseMipMap", |n| n.i32(0))
            .child("Filename", |n| n.str(&file))
            .child("RelativeFilename", |n| n.str(&file));
        if let Some(b) = bytes {
            node.push(Node::new("Content").raw(&b));
        }
        self.object("Video", node);
        self.videos.insert(i, Some(id));
        Some(id)
    }
}

/// A JSON index.
fn as_index(v: &Value) -> Option<usize> {
    v.as_u64().and_then(|i| usize::try_from(i).ok())
}
