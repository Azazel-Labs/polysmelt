//! Materials: `profile_COMMON` effects as metallic-roughness materials, and the images their samplers read.

use std::borrow::Cow;
use std::io::Read as _;
use std::path::Path;

use super::xml;
use crate::images::Location;
use crate::material::{Alpha, CLAMP, Factor, Full, MIRROR, REPEAT, Specular, Tex, UvTransform};
use crate::mesh::linear_rgba as linear;
use crate::num::f32_of;

/// What a shading channel (`diffuse`, `emission`, ...) holds.
#[derive(Clone)]
enum Channel {
    Color([f32; 4]),
    Float(f32),
    /// A sampler (or image), the texture coordinate semantic it reads, and the `<texture>` element.
    Texture(String, String, usize),
}

/// The share of red, green and blue in luminance.
fn luminance(c: [f32; 4]) -> f32 {
    0.212_671 * c[0] + 0.715_160 * c[1] + 0.072_169 * c[2]
}

impl super::Reader<'_> {
    /// The material of `<material>` `mat`; `uv` maps texture coordinate semantics to the `TEXCOORD_n` they read.
    pub(super) fn material(&mut self, mat: usize, uv: &[(String, u32)]) -> Full {
        let dom = self.dom;
        let mut full = Full {
            name: dom
                .attr(mat, "name")
                .or_else(|| dom.attr(mat, "id"))
                .unwrap_or("")
                .to_owned(),
            metallic: 0.0,
            ..Full::default()
        };
        let instance = dom.child(mat, "instance_effect");
        let Some(effect) = instance
            .and_then(|i| dom.link(i, "url"))
            .filter(|&e| dom.is(e, "effect"))
        else {
            return full;
        };
        full.double_sided = [effect, mat].iter().any(|&e| {
            dom.descendant(e, "double_sided", 8)
                .is_some_and(|d| matches!(dom.string(d), "1" | "true"))
        });
        let profile = dom.child(effect, "profile_COMMON");
        let technique = profile.and_then(|p| dom.child(p, "technique"));
        let holders: Vec<usize> = [Some(effect), profile, technique, Some(mat)]
            .into_iter()
            .flatten()
            .collect();
        full.extras = self.extras_of(&holders);
        // Parameters the material sets first, then those the effect declares, innermost first.
        let scopes: Vec<usize> = [instance, technique, profile, Some(effect)]
            .into_iter()
            .flatten()
            .collect();
        let Some(shader) = technique.and_then(|t| {
            dom.children(t)
                .find(|&c| matches!(dom.name(c), b"phong" | b"blinn" | b"lambert" | b"constant"))
        }) else {
            return full;
        };
        let constant = dom.is(shader, "constant");
        let channels: Vec<(&[u8], Option<Channel>)> = dom
            .children(shader)
            .map(|c| (dom.name(c), self.channel(c, &scopes)))
            .collect();
        let get = |name: &str| {
            channels
                .iter()
                .find(|(n, _)| *n == name.as_bytes())
                .and_then(|(_, c)| c.clone())
        };

        // The color: diffuse, or emission for the unlit constant shader.
        let color = if constant { get("emission") } else { get("diffuse") };
        let mut color_image = None;
        match color {
            Some(Channel::Color(c)) => full.base_color = linear(c),
            Some(Channel::Texture(sampler, set, el)) => {
                if let Some(t) = self.texture(&sampler, &set, el, &scopes, uv) {
                    color_image = Some(t.image);
                    full.base_color_tex = Some(t);
                }
            }
            _ => {}
        }
        full.unlit = constant;
        if !constant {
            match get("emission") {
                Some(Channel::Color(c)) => {
                    let c = linear(c);
                    full.emissive = [c[0], c[1], c[2]].map(|v| v.max(0.0));
                }
                Some(Channel::Texture(sampler, set, el)) => {
                    if let Some(t) = self.texture(&sampler, &set, el, &scopes, uv) {
                        full.emissive_tex = Some(t);
                        full.emissive = [1.0; 3];
                    }
                }
                _ => {}
            }
        }
        let glossy = matches!(dom.name(shader), b"phong" | b"blinn");
        full.roughness = match get("shininess") {
            Some(Channel::Float(s)) if glossy && s.is_finite() => (2.0 / (s.max(0.0) + 2.0)).sqrt().clamp(0.05, 1.0),
            _ => 1.0,
        };
        // The highlight's color, where half is glTF's usual.
        if glossy {
            match get("specular") {
                Some(Channel::Color(c)) => {
                    let c = linear(c);
                    full.specular = Some(Specular {
                        factor: Factor::of(1.0),
                        color: [c[0], c[1], c[2]].map(|v| if v.is_finite() { v.max(0.0) * 2.0 } else { 1.0 }),
                        color_tex: None,
                    });
                }
                Some(Channel::Texture(sampler, set, el)) => {
                    if let Some(t) = self.texture(&sampler, &set, el, &scopes, uv) {
                        full.specular = Some(Specular {
                            factor: Factor::of(1.0),
                            color: [1.0; 3],
                            color_tex: Some(t),
                        });
                    }
                }
                _ => {}
            }
        }

        // Transparency, in whichever of its modes says what is opaque.
        if let Some(t) = dom.child(shader, "transparent") {
            let factor = match get("transparency") {
                Some(Channel::Float(f)) if f.is_finite() => f,
                _ => 1.0,
            };
            let mode = dom.attr(t, "opaque").unwrap_or("A_ONE");
            match self.channel(t, &scopes) {
                Some(Channel::Color(c)) => {
                    let alpha = match mode {
                        "RGB_ZERO" => 1.0 - luminance(c) * factor,
                        "A_ZERO" => 1.0 - c[3] * factor,
                        "RGB_ONE" => luminance(c) * factor,
                        _ => c[3] * factor,
                    };
                    if alpha > 0.0 && alpha < 0.999 {
                        full.alpha = Alpha::Blend;
                        full.base_color[3] *= alpha;
                    }
                }
                Some(Channel::Texture(sampler, set, el)) if mode == "A_ONE" => {
                    // The color texture's own alpha.
                    if self.keep.textures
                        && let Some(image) = color_image
                        && self
                            .texture(&sampler, &set, el, &scopes, uv)
                            .is_some_and(|t| t.image == image)
                    {
                        full.alpha = Alpha::Blend;
                    }
                }
                _ => {}
            }
        }
        if let Some(Channel::Float(ior)) = get("index_of_refraction")
            && ior > 1.0
            && ior <= 10.0
        {
            full.ior = Some(ior);
        }
        // The bump map the common extras name, unless they say it holds heights rather than normals.
        if let Some(bump) = dom.descendant(effect, "bump", 8)
            && !dom
                .attr(bump, "bumptype")
                .is_some_and(|t| t.eq_ignore_ascii_case("HEIGHTFIELD"))
            && let Some(Channel::Texture(sampler, set, el)) = self.channel(bump, &scopes)
        {
            full.normal_tex = self.texture(&sampler, &set, el, &scopes, uv);
        }
        full
    }

    /// The value of a channel element: a color, a float, a texture, or a parameter holding one.
    fn channel(&self, el: usize, scopes: &[usize]) -> Option<Channel> {
        let dom = self.dom;
        let value = dom.children(el).next()?;
        let numbers = |e: usize| -> Option<Channel> {
            let v = xml::floats32(dom.text(e));
            match v.len() {
                0 => None,
                1 | 2 => Some(Channel::Float(v[0])),
                3 => Some(Channel::Color([v[0], v[1], v[2], 1.0])),
                _ => Some(Channel::Color([v[0], v[1], v[2], v[3]])),
            }
        };
        match dom.name(value) {
            b"color" => numbers(value).map(|c| match c {
                Channel::Float(f) => Channel::Color([f, f, f, 1.0]),
                c => c,
            }),
            b"float" => numbers(value).map(|c| match c {
                Channel::Color(c) => Channel::Float(c[0]),
                c => c,
            }),
            b"texture" => Some(Channel::Texture(
                dom.attr(value, "texture")?.to_owned(),
                dom.attr(value, "texcoord").unwrap_or("").to_owned(),
                value,
            )),
            b"param" => {
                let r = dom.attr(value, "ref").or_else(|| dom.attr(value, "sid"))?;
                let param = self.newparam(r, scopes)?;
                let v = dom.children(param).find(|&c| dom.name(c).starts_with(b"float"))?;
                numbers(v)
            }
            _ => None,
        }
    }

    /// The `<newparam>` with `sid` (or the material's `<setparam>` of it), looked for in each scope in turn.
    fn newparam(&self, sid: &str, scopes: &[usize]) -> Option<usize> {
        let dom = self.dom;
        scopes.iter().find_map(|&s| {
            dom.children(s).find(|&p| match dom.name(p) {
                b"newparam" => dom.attr(p, "sid") == Some(sid),
                b"setparam" => dom.attr(p, "ref") == Some(sid),
                _ => false,
            })
        })
    }

    /// The texture a channel's `texture` attribute names (a sampler parameter, or an image in a loose document),
    /// reading the coordinates `uv` binds `set` to, placed as the `<texture>` element `el`'s extras say.
    fn texture(&mut self, sampler: &str, set: &str, el: usize, scopes: &[usize], uv: &[(String, u32)]) -> Option<Tex> {
        if !self.keep.textures {
            return None;
        }
        let dom = self.dom;
        let mut wrap = [REPEAT; 2];
        let mut image = None;
        if let Some(s) = self
            .newparam(sampler, scopes)
            .and_then(|p| dom.children(p).find(|&c| dom.name(c).starts_with(b"sampler")))
        {
            for (k, name) in ["wrap_s", "wrap_t"].into_iter().enumerate() {
                if let Some(w) = dom.child_string(s, name) {
                    wrap[k] = match w {
                        "MIRROR" | "MIRROR_ONCE" => MIRROR,
                        "CLAMP" | "BORDER" | "NONE" => CLAMP,
                        _ => REPEAT,
                    };
                }
            }
            image = dom
                .child(s, "instance_image")
                .and_then(|i| dom.link(i, "url"))
                .or_else(|| {
                    // A sampler of a surface, which is initialized from an image.
                    let surface = self
                        .newparam(dom.child_string(s, "source")?, scopes)
                        .and_then(|p| dom.child(p, "surface"))?;
                    let init = dom.child(surface, "init_from")?;
                    dom.url_at(init, dom.string(init))
                })
                .filter(|&i| dom.is(i, "image"));
        }
        let image = image.or_else(|| dom.url_at(el, sampler).filter(|&i| dom.is(i, "image")))?;
        let index = self.image(image)?;
        let mut tex = Tex::new(index);
        tex.wrap = wrap;
        tex.uv = uv.iter().find(|(s, _)| s == set).map_or(0, |&(_, k)| k);
        self.placement(el, &mut tex);
        Some(tex)
    }

    /// The repeat, offset, rotation and wrap a texture's `MAYA` extras give it. Its coordinates are
    /// `uv * repeat`, rotated about the origin, `+ offset`, in COLLADA's bottom-left frame, carried into
    /// glTF's top-left one.
    fn placement(&self, el: usize, tex: &mut Tex) {
        let dom = self.dom;
        let Some(t) = dom
            .kids(el, "extra")
            .flat_map(|x| dom.kids(x, "technique"))
            .find(|&t| dom.attr(t, "profile") == Some("MAYA"))
        else {
            return;
        };
        let num = |name: &str| {
            dom.child(t, name)
                .map(|c| xml::float(dom.text(c)))
                .filter(|v| v.is_finite())
                .map(f32_of)
        };
        for (k, axis) in ["U", "V"].into_iter().enumerate() {
            if num(&format!("mirror{axis}")).is_some_and(|m| m != 0.0) {
                tex.wrap[k] = MIRROR;
            } else if num(&format!("wrap{axis}")).is_some_and(|w| w == 0.0) {
                tex.wrap[k] = CLAMP;
            }
        }
        let scale = [num("repeatU").unwrap_or(1.0), num("repeatV").unwrap_or(1.0)];
        let offset = [num("offsetU").unwrap_or(0.0), num("offsetV").unwrap_or(0.0)];
        let rotation = num("rotateUV").unwrap_or(0.0);
        let (sin, cos) = rotation.sin_cos();
        let transform = UvTransform {
            offset: [offset[0] - scale[1] * sin, 1.0 - offset[1] - scale[1] * cos],
            rotation,
            scale,
        };
        if transform != UvTransform::IDENTITY {
            tex.transform = Some(transform);
        }
    }

    /// The document's image for `<image>` `el`, read once.
    fn image(&mut self, el: usize) -> Option<usize> {
        if let Some(&i) = self.image_of.get(&el) {
            return i;
        }
        let i = self.read_image(el);
        self.image_of.insert(el, i);
        i
    }

    fn read_image(&mut self, el: usize) -> Option<usize> {
        let dom = self.dom;
        let name = dom.attr(el, "name").or_else(|| dom.attr(el, "id")).unwrap_or("image");
        let init = dom.child(el, "init_from");
        if let Some(hex) = init.and_then(|i| dom.child(i, "hex")).or_else(|| dom.child(el, "data")) {
            let bytes = xml::from_hex(dom.text(hex))?;
            let format = dom.attr(hex, "format").unwrap_or("png").to_ascii_lowercase();
            let file = format!("{name}.{format}");
            let assets = self.assets;
            return self.images.bytes(Cow::Owned(bytes), &file, Location::Embedded, |b| {
                assets.image(b, Path::new(&file))
            });
        }
        let init = init?;
        let uri = dom
            .child(init, "ref")
            .map_or_else(|| dom.string(init), |r| dom.string(r))
            .trim();
        if uri.is_empty() {
            return None;
        }
        // A relative path in another document is from that document's folder.
        let dir = dom.dir_of(el);
        if !dir.is_empty() && !uri.contains(':') && !uri.starts_with(['/', '\\']) {
            return self.uri_image(&format!("{dir}{uri}"));
        }
        self.uri_image(uri)
    }

    /// The image at a URI: a `data:` URI, an entry of the archive, or a file beside the model.
    fn uri_image(&mut self, uri: &str) -> Option<usize> {
        let assets = self.assets;
        if uri.starts_with("data:") {
            let (mime, bytes) = crate::gltf::data_uri(uri).ok()?;
            let ext = mime.as_deref().and_then(|m| m.strip_prefix("image/")).unwrap_or("png");
            let file = format!("image.{ext}");
            return self.images.bytes(Cow::Owned(bytes), &file, Location::Embedded, |b| {
                assets.image(b, Path::new(&file))
            });
        }
        let decoded = xml::percent_decode(uri).replace('\\', "/");
        let path = if let Some(rest) = decoded.strip_prefix("file:///") {
            if rest.as_bytes().get(1) == Some(&b':') {
                rest.to_owned()
            } else {
                format!("/{rest}")
            }
        } else if let Some(rest) = decoded.strip_prefix("file://") {
            rest.strip_prefix("localhost").unwrap_or(rest).to_owned()
        } else {
            decoded
        };
        let path = path.trim_start_matches("./").to_owned();
        if path.is_empty() {
            return None;
        }
        if self.archive.is_some() {
            let entry = normalize(&format!(
                "{}{path}",
                self.archive.as_ref().map_or("", |a| a.dir.as_str())
            ));
            if let Some(found) = self.archive_image(&entry) {
                return found;
            }
            if let Some(found) = self.archive_image(&normalize(&path)) {
                return found;
            }
        }
        let relative = !path.starts_with('/') && path.as_bytes().get(1) != Some(&b':');
        let found = assets.find(&path, self.dir).or_else(|| {
            let base = self.base.as_deref()?;
            let base = base.strip_prefix("file:///").unwrap_or(base);
            let dir = base.rsplit_once('/').map_or("", |(d, _)| d);
            relative
                .then(|| assets.find(&format!("{dir}/{path}"), self.dir))
                .flatten()
        })?;
        self.images.file(assets, found, &path)
    }

    /// The image in archive entry `entry`: `None` when the archive has no such entry.
    #[allow(clippy::option_option, reason = "no entry and an entry that isn't an image differ")]
    fn archive_image(&mut self, entry: &str) -> Option<Option<usize>> {
        if let Some(&i) = self.archive_images.get(entry) {
            return Some(i);
        }
        let archive = self.archive.as_mut()?;
        let names: Vec<String> = archive
            .zip
            .file_names()
            .filter_map(Result::ok)
            .map(Cow::into_owned)
            .collect();
        let name = names
            .iter()
            .find(|n| *n == entry)
            .or_else(|| names.iter().find(|n| n.eq_ignore_ascii_case(entry)))?
            .clone();
        let mut bytes = Vec::new();
        let read = archive
            .zip
            .by_name(&name)
            .ok()
            .and_then(|f| f.take(super::MAX_ENTRY).read_to_end(&mut bytes).ok());
        let assets = self.assets;
        let i = read.and_then(|_| {
            self.images
                .bytes(Cow::Owned(bytes), &name, Location::Archive(name.clone()), |b| {
                    assets.image(b, Path::new(&name))
                })
        });
        self.archive_images.insert(entry.to_owned(), i);
        Some(i)
    }
}

/// A path with its `.` and `..` parts resolved and no leading `/`.
pub(super) fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for p in path.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}
