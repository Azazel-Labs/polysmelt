//! A structural GLB validator and decoder: what `mesh::to_glb` writes must be a GLB any viewer accepts, so every
//! conformance case is checked against the glTF 2.0 rules that matter for a mesh (container layout, buffer and
//! accessor bounds, index ranges, attribute counts, unit normals, finite values, valid references, embedded images
//! that are what their MIME type says), then decoded so its expectations can be checked on what was written.

use serde_json::Value;

/// One decoded primitive.
#[derive(Debug, Default)]
pub struct Prim {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    /// Triangles, three indices each.
    pub indices: Vec<u32>,
    /// Line segments (mode 1), two indices each, from a primitive on the same vertices.
    pub lines: Vec<u32>,
    /// Points (mode 0), from a primitive on the same vertices.
    pub points: Vec<u32>,
    pub material: Option<usize>,
}

/// One decoded material.
#[derive(Debug, Default)]
pub struct Material {
    pub name: String,
    pub color: [f32; 4],
    pub color_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha_mode: String,
    pub double_sided: bool,
}

/// A decoded GLB.
#[derive(Debug, Default)]
pub struct Glb {
    pub prims: Vec<Prim>,
    pub materials: Vec<Material>,
    /// Embedded images: (MIME type, bytes).
    pub images: Vec<(String, Vec<u8>)>,
}

impl Glb {
    pub fn triangles(&self) -> usize {
        self.prims.iter().map(|p| p.indices.len() / 3).sum()
    }
    pub fn vertices(&self) -> usize {
        self.prims.iter().map(|p| p.positions.len()).sum()
    }
    pub fn lines(&self) -> usize {
        self.prims.iter().map(|p| p.lines.len() / 2).sum()
    }
    pub fn points(&self) -> usize {
        self.prims.iter().map(|p| p.points.len()).sum()
    }
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| format!("truncated at byte {at}"))
}

fn idx(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("{what} is not an index: {v}"))
}

/// Validates `bytes` as a GLB and decodes it.
pub fn validate(bytes: &[u8]) -> Result<Glb, String> {
    // Container: 12-byte header, then a JSON chunk and an optional BIN chunk, each 4-byte aligned.
    if bytes.get(..4) != Some(b"glTF") {
        return Err("no glTF magic".into());
    }
    if u32_at(bytes, 4)? != 2 {
        return Err("GLB version is not 2".into());
    }
    if u32_at(bytes, 8)? as usize != bytes.len() {
        return Err(format!(
            "header length {} != file length {}",
            u32_at(bytes, 8)?,
            bytes.len()
        ));
    }
    let json_len = u32_at(bytes, 12)? as usize;
    if &bytes[16..20] != b"JSON" {
        return Err("first chunk is not JSON".into());
    }
    if !json_len.is_multiple_of(4) {
        return Err("JSON chunk is not 4-byte aligned".into());
    }
    let json_bytes = bytes.get(20..20 + json_len).ok_or("JSON chunk runs past the end")?;
    let json: Value = serde_json::from_slice(json_bytes).map_err(|e| format!("JSON chunk doesn't parse: {e}"))?;
    let mut bin: &[u8] = &[];
    let mut at = 20 + json_len;
    if at < bytes.len() {
        let len = u32_at(bytes, at)? as usize;
        if &bytes[at + 4..at + 8] != b"BIN\0" {
            return Err("second chunk is not BIN".into());
        }
        if !len.is_multiple_of(4) {
            return Err("BIN chunk is not 4-byte aligned".into());
        }
        bin = bytes.get(at + 8..at + 8 + len).ok_or("BIN chunk runs past the end")?;
        at += 8 + len;
    }
    if at != bytes.len() {
        return Err("bytes after the last chunk".into());
    }

    if json["asset"]["version"] != "2.0" {
        return Err("asset.version is not 2.0".into());
    }
    let arr = |k: &str| json[k].as_array().cloned().unwrap_or_default();
    let buffers = arr("buffers");
    if buffers.len() > 1 {
        return Err("more than one buffer in a GLB".into());
    }
    if let Some(b) = buffers.first() {
        let len = idx(&b["byteLength"], "buffer byteLength")?;
        if b.get("uri").is_some() {
            return Err("the GLB buffer has a uri".into());
        }
        if len > bin.len() || bin.len() - len > 3 {
            return Err(format!("buffer byteLength {len} doesn't match BIN chunk {}", bin.len()));
        }
    }
    let views = arr("bufferViews");
    let view_bytes = |v: usize| -> Result<&[u8], String> {
        let view = views.get(v).ok_or(format!("bufferView {v} doesn't exist"))?;
        if idx(&view["buffer"], "bufferView.buffer")? != 0 {
            return Err("bufferView points at a missing buffer".into());
        }
        let off = view.get("byteOffset").map_or(Ok(0), |o| idx(o, "byteOffset"))?;
        let len = idx(&view["byteLength"], "byteLength")?;
        bin.get(off..off + len)
            .ok_or_else(|| format!("bufferView {v} runs past the buffer"))
    };
    let accessors = arr("accessors");

    // Accessors: valid types, inside their views, aligned, finite.
    let read_accessor = |a: usize| -> Result<(Vec<f32>, usize, &Value), String> {
        let acc = accessors.get(a).ok_or(format!("accessor {a} doesn't exist"))?;
        let count = idx(&acc["count"], "accessor count")?;
        let width = match acc["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            t => return Err(format!("accessor {a} has type {t:?}")),
        };
        let ct = acc["componentType"].as_u64().unwrap_or(0);
        let size = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err(format!("accessor {a} has componentType {ct}")),
        };
        let view = views
            .get(idx(&acc["bufferView"], "accessor bufferView")?)
            .ok_or("accessor bufferView missing")?;
        let data = view_bytes(idx(&acc["bufferView"], "")?)?;
        let off = acc.get("byteOffset").map_or(Ok(0), |o| idx(o, "accessor byteOffset"))?;
        let view_off = view.get("byteOffset").map_or(0, |o| o.as_u64().unwrap_or(0) as usize);
        if (off + view_off) % size != 0 {
            return Err(format!("accessor {a} is not aligned to its component size"));
        }
        let stride = view
            .get("byteStride")
            .map_or(width * size, |s| s.as_u64().unwrap_or(0) as usize);
        if count > 0 && off + stride * (count - 1) + width * size > data.len() {
            return Err(format!("accessor {a} runs past its bufferView"));
        }
        let normalized = acc["normalized"].as_bool().unwrap_or(false);
        let mut out = Vec::with_capacity(count * width);
        for i in 0..count {
            for k in 0..width {
                let p = off + i * stride + k * size;
                let b = &data[p..p + size];
                let v = match ct {
                    5120 => {
                        let v = f32::from(b[0].cast_signed());
                        if normalized { (v / 127.0).max(-1.0) } else { v }
                    }
                    5121 => {
                        let v = f32::from(b[0]);
                        if normalized { v / 255.0 } else { v }
                    }
                    5122 => {
                        let v = f32::from(i16::from_le_bytes([b[0], b[1]]));
                        if normalized { (v / 32767.0).max(-1.0) } else { v }
                    }
                    5123 => {
                        let v = f32::from(u16::from_le_bytes([b[0], b[1]]));
                        if normalized { v / 65535.0 } else { v }
                    }
                    5125 => u32::from_le_bytes(b.try_into().unwrap()) as f32,
                    _ => f32::from_le_bytes(b.try_into().unwrap()),
                };
                if !v.is_finite() {
                    return Err(format!("accessor {a} holds a non-finite value"));
                }
                out.push(v);
            }
        }
        Ok((out, width, acc))
    };
    let read_u32s = |a: usize| -> Result<Vec<u32>, String> {
        let acc = accessors.get(a).ok_or(format!("accessor {a} doesn't exist"))?;
        let ct = acc["componentType"].as_u64().unwrap_or(0);
        if !matches!(ct, 5121 | 5123 | 5125) || acc["type"] != "SCALAR" {
            return Err(format!("index accessor {a} is not an unsigned scalar"));
        }
        if acc["normalized"].as_bool() == Some(true) {
            return Err(format!("index accessor {a} is normalized"));
        }
        let (v, _, _) = read_accessor(a)?;
        Ok(v.into_iter().map(|x| x as u32).collect())
    };

    // Images: in the buffer, with a MIME type their bytes agree with.
    let mut images = Vec::new();
    for (i, img) in arr("images").iter().enumerate() {
        let mime = img["mimeType"].as_str().unwrap_or_default().to_owned();
        let data = view_bytes(idx(&img["bufferView"], "image bufferView")?)?.to_vec();
        let ok = match mime.as_str() {
            "image/png" => data.starts_with(b"\x89PNG\r\n\x1a\n"),
            "image/jpeg" => data.starts_with(&[0xFF, 0xD8, 0xFF]),
            _ => false,
        };
        if !ok {
            return Err(format!("image {i} isn't the {mime:?} it claims to be"));
        }
        images.push((mime, data));
    }
    let textures = arr("textures");
    let samplers = arr("samplers");
    for (i, t) in textures.iter().enumerate() {
        if idx(&t["source"], "texture source")? >= images.len() {
            return Err(format!("texture {i} points at a missing image"));
        }
        if let Some(s) = t.get("sampler")
            && idx(s, "texture sampler")? >= samplers.len()
        {
            return Err(format!("texture {i} points at a missing sampler"));
        }
    }
    let texture_image = |info: &Value, what: &str| -> Result<Option<usize>, String> {
        if info.is_null() {
            return Ok(None);
        }
        let t = idx(&info["index"], what)?;
        let tex = textures.get(t).ok_or(format!("{what} points at missing texture {t}"))?;
        Ok(Some(idx(&tex["source"], "texture source")?))
    };

    let mut materials = Vec::new();
    for m in arr("materials") {
        let pbr = &m["pbrMetallicRoughness"];
        let f = |v: &Value, d: f32| v.as_f64().map_or(d, |x| x as f32);
        let mut color = [1.0f32; 4];
        if let Some(c) = pbr["baseColorFactor"].as_array() {
            if c.len() != 4 {
                return Err("baseColorFactor is not 4 numbers".into());
            }
            for (k, v) in c.iter().enumerate() {
                color[k] = f(v, 1.0);
            }
        }
        if color.iter().any(|c| !(0.0..=1.0).contains(c)) {
            return Err(format!("baseColorFactor out of range: {color:?}"));
        }
        let mut emissive = [0.0f32; 3];
        if let Some(c) = m["emissiveFactor"].as_array() {
            for (k, v) in c.iter().enumerate().take(3) {
                emissive[k] = f(v, 0.0);
            }
        }
        let metallic = f(&pbr["metallicFactor"], 1.0);
        let roughness = f(&pbr["roughnessFactor"], 1.0);
        if !(0.0..=1.0).contains(&metallic) || !(0.0..=1.0).contains(&roughness) {
            return Err("metallic or roughness factor out of range".into());
        }
        let alpha_mode = m["alphaMode"].as_str().unwrap_or("OPAQUE").to_owned();
        if !matches!(alpha_mode.as_str(), "OPAQUE" | "MASK" | "BLEND") {
            return Err(format!("alphaMode {alpha_mode}"));
        }
        materials.push(Material {
            name: m["name"].as_str().unwrap_or_default().to_owned(),
            color,
            color_image: texture_image(&pbr["baseColorTexture"], "baseColorTexture")?,
            normal_image: texture_image(&m["normalTexture"], "normalTexture")?,
            emissive,
            metallic,
            roughness,
            alpha_mode,
            double_sided: m["doubleSided"].as_bool().unwrap_or(false),
        });
    }

    // Scene graph: a scene of nodes that reach every mesh.
    let meshes = arr("meshes");
    let nodes = arr("nodes");
    if let Some(s) = json.get("scene")
        && idx(s, "scene")? >= arr("scenes").len()
    {
        return Err("scene points at a missing scene".into());
    }
    for scene in arr("scenes") {
        for n in scene["nodes"].as_array().cloned().unwrap_or_default() {
            if idx(&n, "scene node")? >= nodes.len() {
                return Err("scene points at a missing node".into());
            }
        }
    }
    for n in &nodes {
        if let Some(m) = n.get("mesh")
            && idx(m, "node mesh")? >= meshes.len()
        {
            return Err("node points at a missing mesh".into());
        }
    }

    // Indices a primitive in `mode` draws on `n` vertices: every one in range, a whole number of
    // triangles or segments.
    let draws = |p: &Value, mode: u64, n: usize| -> Result<Vec<u32>, String> {
        let indices = match p.get("indices") {
            Some(i) => read_u32s(idx(i, "indices")?)?,
            None => (0..n as u32).collect(),
        };
        let per = match mode {
            4 => 3,
            1 => 2,
            _ => 1,
        };
        if !indices.len().is_multiple_of(per) {
            return Err(format!("mode {mode} index count is not a multiple of {per}"));
        }
        if let Some(&bad) = indices.iter().find(|&&i| i as usize >= n) {
            return Err(format!("index {bad} >= vertex count {n}"));
        }
        Ok(indices)
    };

    let mut prims: Vec<Prim> = Vec::new();
    for mesh in &meshes {
        // The POSITION accessor and material of the primitive decoded last in this mesh: lines and
        // points on the same vertices join it.
        let mut last: Option<(usize, Option<Value>)> = None;
        for p in mesh["primitives"].as_array().cloned().unwrap_or_default() {
            let mode = p["mode"].as_u64().unwrap_or(4);
            if !matches!(mode, 0 | 1 | 4) {
                return Err(format!("primitive mode {mode}, not triangles, lines or points"));
            }
            let attrs = &p["attributes"];
            let pos_a = idx(&attrs["POSITION"], "POSITION")?;
            if mode != 4
                && let Some((a, material)) = &last
                && *a == pos_a
                && material.as_ref() == p.get("material")
                && let Some(prim) = prims.last_mut()
            {
                let d = draws(&p, mode, prim.positions.len())?;
                if mode == 1 { &mut prim.lines } else { &mut prim.points }.extend(d);
                continue;
            }
            let (pos, w, acc) = read_accessor(pos_a)?;
            if w != 3 || acc["componentType"] != 5126 {
                return Err("POSITION is not float VEC3".into());
            }
            let positions: Vec<[f32; 3]> = pos.as_chunks::<3>().0.to_vec();
            // POSITION must carry min and max, and they must be the real bounds.
            let (Some(min), Some(max)) = (acc["min"].as_array(), acc["max"].as_array()) else {
                return Err("POSITION has no min/max".into());
            };
            for k in 0..3 {
                let lo = positions.iter().map(|p| p[k]).fold(f32::MAX, f32::min);
                let hi = positions.iter().map(|p| p[k]).fold(f32::MIN, f32::max);
                let (m0, m1) = (
                    min[k].as_f64().unwrap_or(f64::NAN) as f32,
                    max[k].as_f64().unwrap_or(f64::NAN) as f32,
                );
                #[allow(clippy::float_cmp, reason = "the spec says min/max are the data's own values")]
                if m0 != lo || m1 != hi {
                    return Err(format!(
                        "POSITION min/max [{m0}, {m1}] != actual [{lo}, {hi}] on axis {k}"
                    ));
                }
            }
            let n = positions.len();
            let mut prim = Prim {
                positions,
                ..Prim::default()
            };
            let attr = |name: &str, width: usize| -> Result<Option<Vec<f32>>, String> {
                let Some(a) = attrs.get(name) else { return Ok(None) };
                let (v, w, _) = read_accessor(idx(a, name)?)?;
                if w != width {
                    return Err(format!("{name} has width {w}, not {width}"));
                }
                if v.len() / w != n {
                    return Err(format!("{name} count {} != POSITION count {n}", v.len() / w));
                }
                Ok(Some(v))
            };
            if let Some(v) = attr("NORMAL", 3)? {
                prim.normals = v.as_chunks::<3>().0.to_vec();
                for nm in &prim.normals {
                    let len = (nm[0] * nm[0] + nm[1] * nm[1] + nm[2] * nm[2]).sqrt();
                    if (len - 1.0).abs() > 1e-3 {
                        return Err(format!("NORMAL {nm:?} is not unit length"));
                    }
                }
            }
            if let Some(v) = attr("TEXCOORD_0", 2)? {
                prim.uvs = v.as_chunks::<2>().0.to_vec();
            }
            if let Some(a) = attrs.get("COLOR_0") {
                let (v, w, _) = read_accessor(idx(a, "COLOR_0")?)?;
                if v.len() / w != n {
                    return Err("COLOR_0 count != POSITION count".into());
                }
                if v.iter().any(|c| !(0.0..=1.0).contains(c)) {
                    return Err("COLOR_0 outside [0, 1]".into());
                }
                prim.colors = v
                    .chunks(w)
                    .map(|c| [c[0], c[1], c[2], if w == 4 { c[3] } else { 1.0 }])
                    .collect();
            }
            let d = draws(&p, mode, n)?;
            *match mode {
                4 => &mut prim.indices,
                1 => &mut prim.lines,
                _ => &mut prim.points,
            } = d;
            if let Some(m) = p.get("material") {
                let m = idx(m, "material")?;
                if m >= materials.len() {
                    return Err(format!("primitive material {m} doesn't exist"));
                }
                prim.material = Some(m);
            }
            prims.push(prim);
            last = Some((pos_a, p.get("material").cloned()));
        }
    }
    if prims.is_empty() {
        return Err("no primitives".into());
    }
    Ok(Glb {
        prims,
        materials,
        images,
    })
}
