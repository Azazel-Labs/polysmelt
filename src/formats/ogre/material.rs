//! OGRE material scripts (`.material`): the objects they define, inheritance between them, variables, and what a
//! material's first usable technique says, as a glTF material.
//!
//! A script is a tree of objects (`material`, `technique`, `pass`, `texture_unit`, ...), each a line of words and
//! a block in braces, holding property lines. `material Child : Parent` starts from the parent's tree and lays the
//! child's over it: objects of a kind are matched by name, or by position when unnamed, and the child's
//! properties come after the parent's, so they win. `abstract` objects are only inherited from. `set $name value`
//! defines a variable that `$name` in the object and the objects inside it stands for. `import` reads another
//! script's objects.
//!
//! The technique used is the first in the default scheme at level of detail 0. Its first pass gives the colors
//! and render state: `diffuse` the base color (its alpha only when the pass blends), `emissive`, `specular`'s
//! shininess a roughness, `scene_blend` alpha blending, `alpha_rejection` a cutoff, `cull_hardware none` both
//! sides, `lighting off` unlit (whose color is then white, as unlit passes ignore `diffuse`). The first texture
//! unit of any pass holding a plain image becomes the color texture, with its coordinate set, address mode,
//! filtering and `scale`, `scroll` and `rotate` as a texture transform; a unit whose name or file says it's a
//! normal map becomes the normal texture. Everything else is kept as the material's extras.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::material::{CLAMP, LINEAR, LINEAR_MIPMAP_LINEAR, MIRROR, REPEAT, UvTransform};
use crate::mesh::{Material, Sampling, srgb_to_linear};

/// The deepest objects nest; deeper ones are skipped.
const MAX_DEPTH: usize = 32;
/// The longest chain of inheritance followed.
const MAX_INHERITANCE: usize = 16;
/// The most scripts read, imports included.
pub(super) const MAX_SCRIPTS: usize = 64;
/// The most words a variable's substitution may add to a line.
const MAX_LINE_WORDS: usize = 256;

/// glTF's nearest filters.
const NEAREST: u32 = 9728;
const NEAREST_MIPMAP_NEAREST: u32 = 9984;
const LINEAR_MIPMAP_NEAREST: u32 = 9985;
const NEAREST_MIPMAP_LINEAR: u32 = 9986;

/// A line of a script: its words, and its block when it's an object.
#[derive(Clone, Debug, Default)]
pub(super) struct Item {
    pub words: Vec<String>,
    pub body: Option<Vec<Item>>,
}

impl Item {
    fn key(&self) -> &str {
        self.words.first().map_or("", String::as_str)
    }

    /// An object's name and the object it inherits from: `pass name : parent`.
    fn name_and_parent(&self) -> (Option<String>, Option<String>) {
        let rest = &self.words[1.min(self.words.len())..];
        // `abstract pass name`: the kind is the second word.
        let rest = if self.key() == "abstract" {
            &rest[1.min(rest.len())..]
        } else {
            rest
        };
        if let Some(colon) = rest.iter().position(|w| w == ":") {
            let name = (colon > 0).then(|| rest[..colon].join(" "));
            return (name, rest.get(colon + 1).cloned());
        }
        match rest {
            [] => (None, None),
            [n] => match n.split_once(':') {
                Some((a, b)) if !a.is_empty() && !b.is_empty() && !n.contains("::") => {
                    (Some(a.to_owned()), Some(b.to_owned()))
                }
                _ => (Some(n.trim_end_matches(':').to_owned()), None),
            },
            [n, p, ..] if n.ends_with(':') => (Some(n.trim_end_matches(':').to_owned()), Some(p.clone())),
            [n, p, ..] if p.starts_with(':') && p.len() > 1 => (Some(n.clone()), Some(p[1..].to_owned())),
            [n, ..] => (Some(n.clone()), None),
        }
    }

    /// The kind of object: `pass` for `abstract pass name` too.
    fn kind(&self) -> &str {
        if self.key() == "abstract" {
            self.words.get(1).map_or("", String::as_str)
        } else {
            self.key()
        }
    }
}

enum Token {
    Word(String, usize),
    Open,
    Close,
}

/// Splits a script into words, quoted strings and braces; comments are dropped.
fn tokens(src: &str) -> Vec<Token> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut line = 0usize;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                if b[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else if c == b'{' {
            out.push(Token::Open);
            i += 1;
        } else if c == b'}' {
            out.push(Token::Close);
            i += 1;
        } else if c == b'"' {
            let start = i + 1;
            i = start;
            while i < b.len() && b[i] != b'"' && b[i] != b'\n' {
                i += 1;
            }
            out.push(Token::Word(src[start..i].to_owned(), line));
            if i < b.len() && b[i] == b'"' {
                i += 1;
            }
        } else {
            let start = i;
            while i < b.len() {
                let c = b[i];
                if c.is_ascii_whitespace() || c == b'"' || b[i..].starts_with(b"//") || b[i..].starts_with(b"/*") {
                    break;
                }
                // A brace ends a word, except in `${name}`.
                if (c == b'{' && !(i > start && b[i - 1] == b'$')) || (c == b'}' && !src[start..i].contains("${")) {
                    break;
                }
                i += 1;
            }
            out.push(Token::Word(src[start..i].to_owned(), line));
        }
    }
    out
}

/// Parses a script into its top-level items. A stray closing brace is ignored, and a block left open is closed
/// at the end.
pub(super) fn parse(src: &str) -> Vec<Item> {
    let mut stack: Vec<Vec<Item>> = vec![Vec::new()];
    // Items opened deeper than the limit: their braces are counted and their contents dropped.
    let mut skipped = 0usize;
    let mut last_line = usize::MAX;
    for t in tokens(src) {
        match t {
            Token::Word(w, line) => {
                if skipped > 0 {
                    continue;
                }
                let items = stack.last_mut().expect("the top level is never popped");
                match items.last_mut() {
                    Some(item) if line == last_line && item.body.is_none() => item.words.push(w),
                    _ => items.push(Item {
                        words: vec![w],
                        body: None,
                    }),
                }
                last_line = line;
            }
            Token::Open => {
                if skipped > 0 || stack.len() > MAX_DEPTH {
                    skipped += 1;
                    continue;
                }
                let items = stack.last_mut().expect("the top level is never popped");
                if items.last().is_none_or(|i| i.body.is_some()) {
                    // A block without a header: an unnamed object of no kind.
                    items.push(Item::default());
                }
                stack.push(Vec::new());
                last_line = usize::MAX;
            }
            Token::Close => {
                if skipped > 0 {
                    skipped -= 1;
                    continue;
                }
                if stack.len() > 1 {
                    close(&mut stack);
                }
                last_line = usize::MAX;
            }
        }
    }
    while stack.len() > 1 {
        close(&mut stack);
    }
    stack.pop().unwrap_or_default()
}

fn close(stack: &mut Vec<Vec<Item>>) {
    let body = stack.pop().unwrap_or_default();
    if let Some(item) = stack.last_mut().and_then(|items| items.last_mut()) {
        item.body = Some(body);
    }
}

/// The objects of every script read, by kind and name; the first definition of each wins.
#[derive(Default)]
pub(super) struct Library {
    objects: HashMap<(String, String), (Item, usize)>,
    /// The folder each script was in, for its textures.
    pub dirs: Vec<PathBuf>,
    /// The scripts read.
    pub scripts: Vec<PathBuf>,
}

impl Library {
    /// Adds a script's objects, read from a file in `dir`; returns the files it imports.
    pub(super) fn add(&mut self, path: PathBuf, src: &str) -> Vec<String> {
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let file = self.dirs.len();
        self.dirs.push(dir);
        self.scripts.push(path);
        let mut imports = Vec::new();
        for item in parse(src) {
            if item.key() == "import" {
                if let Some(from) = item.words.iter().position(|w| w == "from")
                    && let Some(f) = item.words.get(from + 1)
                {
                    imports.push(f.clone());
                }
                continue;
            }
            if item.body.is_none() {
                continue;
            }
            let (Some(name), _) = item.name_and_parent() else {
                continue;
            };
            let kind = item.kind().to_owned();
            self.objects.entry((kind, name)).or_insert((item, file));
        }
        imports
    }

    /// Whether a material named `name` is defined.
    pub(super) fn has(&self, name: &str) -> bool {
        self.objects.contains_key(&("material".to_owned(), name.to_owned()))
    }

    /// An object's tree with what it inherits laid under it, and the script it's in.
    fn resolve(&self, kind: &str, name: &str, depth: usize) -> Option<(Vec<Item>, usize)> {
        let (item, file) = self.objects.get(&(kind.to_owned(), name.to_owned()))?;
        let body = item.body.clone().unwrap_or_default();
        let (_, parent) = item.name_and_parent();
        let body = self.inherit(kind, parent.as_deref(), body, depth);
        Some((body, *file))
    }

    /// `body` laid over the tree of the `kind` object named `parent`, if there is one.
    fn inherit(&self, kind: &str, parent: Option<&str>, body: Vec<Item>, depth: usize) -> Vec<Item> {
        let parent = parent
            .filter(|_| depth < MAX_INHERITANCE)
            .and_then(|p| self.resolve(kind, p, depth + 1));
        let body = match parent {
            Some((base, _)) => overlay(base, body),
            None => body,
        };
        // Objects inside that inherit from abstract ones.
        body.into_iter()
            .map(|mut item| {
                if let Some(inner) = item.body.take() {
                    let (_, p) = item.name_and_parent();
                    let kind = item.kind().to_owned();
                    item.body = Some(self.inherit(&kind, p.as_deref(), inner, depth + 1));
                }
                item
            })
            .collect()
    }
}

/// `child` laid over `base`: objects matched by kind and name (by position among the unnamed), properties
/// appended.
fn overlay(mut base: Vec<Item>, child: Vec<Item>) -> Vec<Item> {
    let mut unnamed_seen: HashMap<String, usize> = HashMap::new();
    for item in child {
        if item.body.is_none() {
            base.push(item);
            continue;
        }
        let kind = item.kind().to_owned();
        let (name, _) = item.name_and_parent();
        let target = if let Some(n) = &name {
            base.iter()
                .position(|b| b.body.is_some() && b.kind() == kind && b.name_and_parent().0.as_deref() == Some(n))
        } else {
            let k = unnamed_seen.entry(kind.clone()).or_insert(0);
            let at = base
                .iter()
                .enumerate()
                .filter(|(_, b)| b.body.is_some() && b.kind() == kind && b.name_and_parent().0.is_none())
                .nth(*k)
                .map(|(i, _)| i);
            *k += 1;
            at
        };
        match target {
            Some(t) => {
                let inner = base[t].body.take().unwrap_or_default();
                base[t].body = Some(overlay(inner, item.body.unwrap_or_default()));
            }
            None => base.push(item),
        }
    }
    base
}

/// A texture a material uses: the file named, the folder of the script naming it, and how it's sampled.
#[derive(Debug, Clone)]
pub(super) struct TextureRef {
    pub file: String,
    pub dir: PathBuf,
    pub sampling: Sampling,
    /// The `TEXCOORD_n` set it reads.
    pub uv_set: u32,
}

/// A material as glTF can hold it, the textures still to find, and the rest as extras.
#[derive(Debug, Clone)]
pub(super) struct Mapped {
    pub material: Material,
    pub color: Option<TextureRef>,
    pub normal: Option<TextureRef>,
    /// Blended by the color's or texture's alpha (glTF's `BLEND`) though the color may be opaque.
    pub blend: bool,
    pub extras: Map<String, Value>,
}

/// Variables in scope, innermost last.
struct Scopes(Vec<HashMap<String, Vec<String>>>);

impl Scopes {
    fn enter(&mut self, body: &[Item]) {
        let mut vars = HashMap::new();
        for item in body {
            if item.key() == "set"
                && item.body.is_none()
                && let Some(name) = item.words.get(1).and_then(|n| n.strip_prefix('$'))
            {
                vars.insert(name.to_owned(), item.words[2.min(item.words.len())..].to_vec());
            }
        }
        self.0.push(vars);
    }

    fn leave(&mut self) {
        self.0.pop();
    }

    /// A line's words with its variables replaced.
    fn words(&self, item: &Item) -> Vec<String> {
        let mut out = Vec::with_capacity(item.words.len());
        for w in &item.words {
            let name = w
                .strip_prefix("${")
                .and_then(|n| n.strip_suffix('}'))
                .or_else(|| w.strip_prefix('$'));
            match name.and_then(|n| self.0.iter().rev().find_map(|s| s.get(n))) {
                Some(v) if out.len() + v.len() <= MAX_LINE_WORDS => out.extend(v.iter().cloned()),
                _ => out.push(w.clone()),
            }
        }
        out
    }
}

fn num(w: Option<&String>) -> Option<f32> {
    w.and_then(|w| w.parse::<f32>().ok()).filter(|v| v.is_finite())
}

/// Numbers from a line's words after its key.
fn numbers(words: &[String]) -> Vec<f32> {
    words[1.min(words.len())..]
        .iter()
        .filter_map(|w| w.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

fn rest(words: &[String]) -> Value {
    json!(words[1.min(words.len())..].join(" "))
}

fn is_on(w: Option<&String>) -> bool {
    w.is_some_and(|w| super::xml::truthy(w))
}

/// Whether a texture unit's name, alias or file says it holds a normal map: "normal" anywhere, or a word (split
/// at anything but letters and digits) such as "bump", "nm" or "nrm", or a name ending in "bump" or "bumpmap"
/// (`RockBump`); a file stem also by suffixes such as "_n" and "_ddn".
fn is_normal_map(unit_name: Option<&str>, alias: Option<&str>, file: &str) -> bool {
    let says = |s: &str| {
        let s = s.to_ascii_lowercase();
        s.contains("normal")
            || s.ends_with("bump")
            || s.ends_with("bumpmap")
            || s.split(|c: char| !c.is_ascii_alphanumeric())
                .any(|w| matches!(w, "bump" | "bumpmap" | "nmap" | "nm" | "nrm" | "norm" | "ddn"))
    };
    if unit_name.is_some_and(says) || alias.is_some_and(says) {
        return true;
    }
    let stem = Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    says(&stem) || ["_n", "-n", "_local"].iter().any(|s| stem.ends_with(s))
}

/// One texture unit, read.
struct Unit {
    pass: usize,
    name: Option<String>,
    file: Option<String>,
    alias: Option<String>,
    /// An environment, cube, shadow or rendered map, not a plain image.
    special: bool,
    sampling: Sampling,
    uv_set: u32,
    /// Its properties not otherwise used.
    other: Map<String, Value>,
}

fn wrap(w: &str) -> Option<u32> {
    match w {
        "wrap" => Some(REPEAT),
        "clamp" | "border" => Some(CLAMP),
        "mirror" => Some(MIRROR),
        _ => None,
    }
}

/// `filtering`'s words as glTF's magnification and minification filters.
fn filtering(words: &[String]) -> Option<[Option<u32>; 2]> {
    match words.get(1).map(String::as_str)? {
        "none" => Some([Some(NEAREST), Some(NEAREST)]),
        "bilinear" => Some([Some(LINEAR), Some(LINEAR_MIPMAP_NEAREST)]),
        "trilinear" | "anisotropic" => Some([Some(LINEAR), Some(LINEAR_MIPMAP_LINEAR)]),
        min => {
            let mag = words.get(2)?;
            let mip = words.get(3)?;
            let linear = |w: &str| w == "linear" || w == "anisotropic";
            let mag = if linear(mag) { LINEAR } else { NEAREST };
            let min = match (linear(min), mip.as_str()) {
                (true, "none") => LINEAR,
                (false, "none") => NEAREST,
                (true, "point") => LINEAR_MIPMAP_NEAREST,
                (false, "point") => NEAREST_MIPMAP_NEAREST,
                (true, _) => LINEAR_MIPMAP_LINEAR,
                (false, _) => NEAREST_MIPMAP_LINEAR,
            };
            Some([Some(mag), Some(min)])
        }
    }
}

/// OGRE's `scale`, `scroll` and `rotate` (degrees, counter-clockwise, both about the texture's center) as
/// `KHR_texture_transform`, whose rotation turns the other way.
pub(super) fn transform(scale: [f32; 2], scroll: [f32; 2], degrees: f32) -> Option<UvTransform> {
    #[allow(clippy::float_cmp, reason = "only OGRE's exact defaults leave the coordinates alone")]
    if scale == [1.0, 1.0] && scroll == [0.0, 0.0] && degrees == 0.0 {
        return None;
    }
    let s = scale.map(|v| if v == 0.0 { 1.0 } else { 1.0 / v });
    let d = [scroll[0] - 0.5 * s[0], scroll[1] - 0.5 * s[1]];
    let theta = degrees.to_radians();
    let (sin, cos) = theta.sin_cos();
    let offset = [0.5 + cos * d[0] - sin * d[1], 0.5 + sin * d[0] + cos * d[1]];
    let t = UvTransform {
        offset,
        rotation: -theta,
        scale: s,
    };
    (t.offset.iter().chain(&t.scale).all(|v| v.is_finite()) && t.rotation.is_finite()).then_some(t)
}

fn unit(pass: usize, item: &Item, scopes: &mut Scopes) -> Unit {
    let body = item.body.as_deref().unwrap_or_default();
    scopes.enter(body);
    let (name, _) = item.name_and_parent();
    let mut u = Unit {
        pass,
        name,
        file: None,
        alias: None,
        special: false,
        sampling: Sampling::default(),
        uv_set: 0,
        other: Map::new(),
    };
    let (mut scale, mut scroll, mut rotate) = ([1.0f32; 2], [0.0f32; 2], 0.0f32);
    for line in body {
        if line.body.is_some() {
            u.other.insert(line.key().to_owned(), json!("{...}"));
            continue;
        }
        let w = scopes.words(line);
        let key = w.first().map_or("", String::as_str);
        match key {
            "set" => {}
            "texture" => u.file = w.get(1).cloned(),
            "anim_texture" => {
                // The first frame: named, or `name_0.ext` when given as a name and a count.
                let first = match (w.get(1), w.get(2)) {
                    (Some(n), Some(c)) if w.len() == 4 && c.parse::<u32>().is_ok() => {
                        let p = Path::new(n);
                        let stem = p.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
                        Some(match p.extension() {
                            Some(e) => format!("{stem}_0.{}", e.to_string_lossy()),
                            None => format!("{stem}_0"),
                        })
                    }
                    (Some(n), _) => Some(n.clone()),
                    _ => None,
                };
                u.file = u.file.take().or(first);
                u.other.insert(key.to_owned(), rest(&w));
            }
            "cubic_texture" => {
                u.file = w.get(1).cloned();
                u.special = true;
                u.other.insert(key.to_owned(), rest(&w));
            }
            "env_map" => {
                if w.get(1).is_some_and(|m| m != "off") {
                    u.special = true;
                }
                u.other.insert(key.to_owned(), rest(&w));
            }
            "content_type" => {
                if w.get(1).is_some_and(|c| c != "named") {
                    u.special = true;
                }
                u.other.insert(key.to_owned(), rest(&w));
            }
            "texture_alias" => u.alias = w.get(1).cloned(),
            "tex_coord_set" => u.uv_set = w.get(1).and_then(|n| n.parse().ok()).unwrap_or(0).min(7),
            "tex_address_mode" => {
                let uw = w.get(1).and_then(|m| wrap(m));
                let vw = w.get(2).and_then(|m| wrap(m)).or(uw);
                if let (Some(a), Some(b)) = (uw, vw) {
                    u.sampling.wrap = [a, b];
                } else {
                    u.other.insert(key.to_owned(), rest(&w));
                }
            }
            "filtering" => match filtering(&w) {
                Some(f) => u.sampling.filter = f,
                None => {
                    u.other.insert(key.to_owned(), rest(&w));
                }
            },
            "scale" => {
                if let (Some(a), Some(b)) = (num(w.get(1)), num(w.get(2))) {
                    scale = [a, b];
                }
            }
            "scroll" => {
                if let (Some(a), Some(b)) = (num(w.get(1)), num(w.get(2))) {
                    scroll = [a, b];
                }
            }
            "rotate" => rotate = num(w.get(1)).unwrap_or(0.0),
            _ => {
                u.other.insert(key.to_owned(), rest(&w));
            }
        }
    }
    scopes.leave();
    // In later OGRE versions a unit's name is its alias too.
    if u.alias.is_none() {
        u.alias.clone_from(&u.name);
    }
    u.sampling.transform = transform(scale, scroll, rotate);
    u
}

/// The technique a material draws with: the first in the default scheme at level of detail 0, else the first.
fn technique<'b>(body: &'b [Item], scopes: &Scopes) -> Option<(usize, &'b Item)> {
    let techniques: Vec<&Item> = body
        .iter()
        .filter(|i| i.body.is_some() && i.kind() == "technique")
        .collect();
    let usable = techniques.iter().position(|t| {
        t.body.as_deref().unwrap_or_default().iter().all(|p| {
            let w = scopes.words(p);
            match w.first().map(String::as_str) {
                Some("scheme") => w.get(1).is_none_or(|s| s.eq_ignore_ascii_case("default")),
                Some("lod_index") => w.get(1).is_none_or(|l| l == "0"),
                _ => true,
            }
        })
    });
    let k = usable.or((!techniques.is_empty()).then_some(0))?;
    Some((k, techniques[k]))
}

/// Maps the material named `name` (with a submesh's texture aliases) to glTF, or `None` when no script defines it.
pub(super) fn map(lib: &Library, name: &str, aliases: &[(String, String)]) -> Option<Mapped> {
    let (body, file) = lib.resolve("material", name, 0)?;
    let dir = lib.dirs.get(file).cloned().unwrap_or_default();
    let mut scopes = Scopes(Vec::new());
    scopes.enter(&body);
    let mut extras = Map::new();
    let mut props = Map::new();
    let mut set_aliases: Vec<(String, String)> = Vec::new();
    for line in &body {
        if line.body.is_some() {
            continue;
        }
        let w = scopes.words(line);
        match w.first().map(String::as_str) {
            Some("set") | None => {}
            Some("set_texture_alias") => {
                if let (Some(a), Some(f)) = (w.get(1), w.get(2)) {
                    set_aliases.push((a.clone(), f.clone()));
                }
            }
            Some(k) => {
                props.insert(k.to_owned(), rest(&w));
            }
        }
    }
    let mut mat = Material::plain(name, [1.0; 4]);
    let mut blend = false;
    let mut alpha = 1.0;
    let mut units = Vec::new();
    let mut programs = Vec::new();
    let techniques = body
        .iter()
        .filter(|i| i.body.is_some() && i.kind() == "technique")
        .count();
    if techniques > 1 {
        extras.insert("techniques".into(), json!(techniques));
    }
    if let Some((k, tech)) = technique(&body, &scopes) {
        let tb = tech.body.as_deref().unwrap_or_default();
        scopes.enter(tb);
        if k > 0 || tech.name_and_parent().0.is_some() {
            extras.insert(
                "technique".into(),
                tech.name_and_parent().0.map_or_else(|| json!(k), |n| json!(n)),
            );
        }
        let passes: Vec<&Item> = tb.iter().filter(|i| i.body.is_some() && i.kind() == "pass").collect();
        if passes.len() > 1 {
            extras.insert("passes".into(), json!(passes.len()));
        }
        for (pi, pass) in passes.iter().enumerate() {
            let pb = pass.body.as_deref().unwrap_or_default();
            scopes.enter(pb);
            for line in pb {
                if line.body.is_some() {
                    match line.kind() {
                        "texture_unit" => units.push(unit(pi, line, &mut scopes)),
                        k if k.ends_with("_program_ref") => {
                            programs.push(json!(format!(
                                "{} {}",
                                k,
                                line.words[1.min(line.words.len())..].join(" ")
                            )));
                        }
                        k if pi == 0 => {
                            props.insert(k.to_owned(), json!("{...}"));
                        }
                        _ => {}
                    }
                    continue;
                }
                if pi > 0 {
                    continue;
                }
                let w = scopes.words(line);
                let key = w.first().map_or("", String::as_str);
                let n = numbers(&w);
                match key {
                    "set" => {}
                    "diffuse" if n.len() >= 3 => {
                        mat.color = [srgb_to_linear(n[0]), srgb_to_linear(n[1]), srgb_to_linear(n[2]), 1.0];
                        alpha = n.get(3).copied().unwrap_or(1.0);
                        if alpha < 1.0 {
                            props.insert(key.to_owned(), rest(&w));
                        }
                    }
                    "emissive" if n.len() >= 3 => {
                        mat.emissive = [srgb_to_linear(n[0]), srgb_to_linear(n[1]), srgb_to_linear(n[2])];
                    }
                    "specular" => {
                        // `r g b shininess` or `r g b a shininess`: Phong's exponent as a roughness.
                        if n.len() >= 4 && n[..3].iter().any(|&c| c > 0.0) {
                            let s = n[n.len() - 1].max(0.0);
                            mat.roughness = (2.0 / (s + 2.0)).sqrt().clamp(0.0, 1.0);
                        }
                        props.insert(key.to_owned(), rest(&w));
                    }
                    "scene_blend" => {
                        let mode: Vec<&str> = w[1..].iter().map(String::as_str).collect();
                        if matches!(mode.as_slice(), ["alpha_blend"] | ["src_alpha", "one_minus_src_alpha"]) {
                            blend = true;
                        } else {
                            props.insert(key.to_owned(), rest(&w));
                        }
                    }
                    "alpha_rejection" => match (w.get(1).map(String::as_str), num(w.get(2))) {
                        (Some("greater_equal"), Some(v)) if v > 0.0 => mat.alpha_cutoff = Some((v / 255.0).min(1.0)),
                        (Some("greater"), Some(v)) => mat.alpha_cutoff = Some(((v + 1.0) / 255.0).min(1.0)),
                        (Some("always_pass"), _) => {}
                        _ => {
                            props.insert(key.to_owned(), rest(&w));
                        }
                    },
                    "cull_hardware" => {
                        if w.get(1).is_some_and(|c| c == "none") {
                            mat.double_sided = true;
                        } else {
                            props.insert(key.to_owned(), rest(&w));
                        }
                    }
                    "lighting" => {
                        if is_on(w.get(1)) {
                            continue;
                        }
                        mat.unlit = true;
                    }
                    _ => {
                        props.insert(key.to_owned(), rest(&w));
                    }
                }
            }
            scopes.leave();
        }
        scopes.leave();
    }
    if mat.unlit {
        // Without lighting a pass draws its texture and vertex colors, not its diffuse color.
        mat.color = [1.0; 4];
        props.remove("diffuse");
    }
    if blend {
        mat.color[3] = alpha.clamp(0.0, 1.0);
    }
    if mat.alpha_cutoff.is_some() && blend {
        // glTF cuts out or blends, not both: the cutoff is kept.
        props.insert("scene_blend".into(), json!("alpha_blend"));
        blend = false;
        mat.color[3] = 1.0;
    }
    if blend {
        // The alpha went into the color.
        props.remove("diffuse");
    }

    // Texture aliases: the submesh's, then the material's.
    for u in &mut units {
        let Some(a) = u.alias.as_deref() else {
            continue;
        };
        if let Some((_, f)) = aliases.iter().chain(&set_aliases).find(|(n, _)| n == a) {
            u.file = Some(f.clone());
        }
    }
    let usable = |u: &Unit| u.file.as_ref().is_some_and(|f| !f.is_empty()) && !u.special;
    let normal_of = |u: &Unit| is_normal_map(u.name.as_deref(), u.alias.as_deref(), u.file.as_deref().unwrap_or(""));
    let color = units
        .iter()
        .position(|u| usable(u) && !normal_of(u) && u.pass == 0)
        .or_else(|| units.iter().position(|u| usable(u) && !normal_of(u)));
    let normal = units.iter().position(|u| usable(u) && normal_of(u));
    let tex = |i: Option<usize>| {
        i.map(|i| {
            let u = &units[i];
            TextureRef {
                file: u.file.clone().unwrap_or_default(),
                dir: dir.clone(),
                sampling: u.sampling,
                uv_set: u.uv_set,
            }
        })
    };
    let color_ref = tex(color);
    let normal_ref = tex(normal);
    let unit_extras: Vec<Value> = units
        .iter()
        .enumerate()
        .filter_map(|(i, u)| {
            let mapped = Some(i) == color || Some(i) == normal;
            if mapped && u.other.is_empty() {
                return None;
            }
            let mut o = u.other.clone();
            if !mapped {
                if let Some(f) = &u.file {
                    o.insert("texture".into(), json!(f));
                }
                if u.uv_set > 0 {
                    o.insert("tex_coord_set".into(), json!(u.uv_set));
                }
            }
            if let Some(n) = &u.name {
                o.insert("name".into(), json!(n));
            }
            o.insert("pass".into(), json!(u.pass));
            if Some(i) == color {
                o.insert("use".into(), json!("baseColor"));
            } else if Some(i) == normal {
                o.insert("use".into(), json!("normal"));
            }
            Some(Value::Object(o))
        })
        .collect();
    if !props.is_empty() {
        extras.insert("properties".into(), Value::Object(props));
    }
    if !unit_extras.is_empty() {
        extras.insert("textureUnits".into(), json!(unit_extras));
    }
    if !programs.is_empty() {
        extras.insert("programs".into(), json!(programs));
    }
    Some(Mapped {
        material: mat,
        color: color_ref,
        normal: normal_ref,
        blend,
        extras,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib(src: &str) -> Library {
        let mut l = Library::default();
        l.add(PathBuf::from("m/a.material"), src);
        l
    }

    #[test]
    fn parses_objects_and_comments() {
        let items =
            parse("// c\nmaterial \"A B\" /* x\n y */\n{\n technique { pass\n { diffuse 1 0 0 } }\n}\n}}\nmaterial C{");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].words, ["material", "A B"]);
        let tech = &items[0].body.as_ref().unwrap()[0];
        assert_eq!(tech.words, ["technique"]);
        let pass = &tech.body.as_ref().unwrap()[0];
        assert_eq!(pass.body.as_ref().unwrap()[0].words, ["diffuse", "1", "0", "0"]);
        assert!(items[1].body.as_ref().unwrap().is_empty());
        // Deep nesting is cut off, not followed.
        let deep = format!("material D {}{}", "a {".repeat(1000), "}".repeat(1000));
        assert_eq!(parse(&deep).len(), 1);
    }

    #[test]
    fn maps_a_pass() {
        let l = lib(
            "material M\n{\n technique\n {\n  pass\n  {\n   diffuse 1 0.5 0 0.5\n   specular 1 1 1 50\n   \
             scene_blend alpha_blend\n   cull_hardware none\n   ambient 0.2 0.2 0.2\n   texture_unit\n   {\n    \
             texture rock.png\n    tex_address_mode clamp mirror\n    filtering none\n    scale 0.5 0.5\n    \
             colour_op modulate\n   }\n   texture_unit\n   {\n    texture rock_normal.png\n   }\n  }\n }\n}\n",
        );
        let m = map(&l, "M", &[]).unwrap();
        assert!(m.blend);
        assert!((m.material.color[3] - 0.5).abs() < 1e-6);
        assert!(m.material.double_sided);
        assert!((m.material.roughness - (2.0f32 / 52.0).sqrt()).abs() < 1e-6);
        let c = m.color.unwrap();
        assert_eq!(c.file, "rock.png");
        assert_eq!(c.sampling.wrap, [CLAMP, MIRROR]);
        assert_eq!(c.sampling.filter, [Some(NEAREST), Some(NEAREST)]);
        let t = c.sampling.transform.unwrap();
        assert_eq!(t.scale, [2.0, 2.0]);
        assert_eq!(t.offset, [-0.5, -0.5]);
        assert_eq!(m.normal.unwrap().file, "rock_normal.png");
        assert_eq!(m.extras["properties"]["ambient"], "0.2 0.2 0.2");
        assert_eq!(m.extras["textureUnits"][0]["colour_op"], "modulate");
    }

    #[test]
    fn inherits_and_substitutes() {
        let l = lib(
            "abstract material Base\n{\n technique\n {\n  pass\n  {\n   diffuse $col\n   texture_unit\n   \
             {\n    texture $tex\n   }\n  }\n }\n}\nmaterial Child : Base\n{\n set $col \"1 0 0\"\n set $tex \
             red.png\n technique { pass { lighting off } }\n}\nmaterial Alias\n{\n set_texture_alias Diffuse b.png\n \
             technique { pass { texture_unit Diffuse { texture a.png } } }\n}\n",
        );
        let m = map(&l, "Child", &[]).unwrap();
        assert!(m.material.unlit);
        assert_eq!(m.color.unwrap().file, "red.png");
        let a = map(&l, "Alias", &[]).unwrap();
        assert_eq!(a.color.unwrap().file, "b.png");
        let a = map(&l, "Alias", &[("Diffuse".into(), "c.png".into())]).unwrap();
        assert_eq!(a.color.unwrap().file, "c.png");
        assert!(map(&l, "Nothing", &[]).is_none());
    }

    #[test]
    fn inheritance_loops_end() {
        let l = lib("material A : B { technique { pass { diffuse 1 0 0 } } }\nmaterial B : A { }\n");
        assert!(map(&l, "A", &[]).is_some());
    }

    #[test]
    fn texture_transform_matches_ogre() {
        // OGRE's matrix: rotate about the center after scroll after scale about the center.
        let (su, sv, mu, mv, deg) = (0.5f32, 0.25f32, 0.1f32, 0.2f32, 30.0f32);
        let t = transform([su, sv], [mu, mv], deg).unwrap();
        let (s, c) = deg.to_radians().sin_cos();
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.3, 0.7]] {
            let p = [uv[0] / su + 0.5 - 0.5 / su + mu, uv[1] / sv + 0.5 - 0.5 / sv + mv];
            let q = [
                c * (p[0] - 0.5) - s * (p[1] - 0.5) + 0.5,
                s * (p[0] - 0.5) + c * (p[1] - 0.5) + 0.5,
            ];
            // KHR_texture_transform: translate * rotate(r) * scale, rotate's rows [cos r, sin r; -sin r, cos r].
            let (rs, rc) = t.rotation.sin_cos();
            let a = [uv[0] * t.scale[0], uv[1] * t.scale[1]];
            let k = [
                rc * a[0] + rs * a[1] + t.offset[0],
                -rs * a[0] + rc * a[1] + t.offset[1],
            ];
            assert!((k[0] - q[0]).abs() < 1e-5 && (k[1] - q[1]).abs() < 1e-5, "{k:?} {q:?}");
        }
        assert!(transform([1.0, 1.0], [0.0, 0.0], 0.0).is_none());
    }
}
