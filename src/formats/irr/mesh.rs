//! Irrlicht's mesh files: `<buffer>`s, each a material, a vertex list and an index list.

use crate::Result;
use crate::formats::scene::Primitive;
use crate::formats::stream::f32_from;
use crate::formats::xml::{El, Xml};

use super::material::{Attrs, IrrMaterial, argb, linear};

/// A buffer: its material and its triangles, in Irrlicht's coordinates.
pub(super) struct Buffer {
    pub material: IrrMaterial,
    pub prim: Primitive,
    /// The color every vertex has, when they all have the same one (`aarrggbb`).
    pub uniform: Option<u32>,
    /// How many coordinate sets the vertices have.
    pub sets: usize,
}

/// A mesh file's buffers.
#[derive(Default)]
pub(super) struct Mesh {
    pub buffers: Vec<Buffer>,
    /// The root's `version`.
    pub version: Option<String>,
    /// Triangles dropped for an index past the vertices, or a last one cut short.
    pub dropped: usize,
}

/// Reads a mesh file's text, stopping with an error past `limit` triangles.
pub(super) fn parse(text: &str, limit: usize) -> Result<Mesh> {
    let xml = Xml::parse(text)?;
    let root = xml.root();
    ensure!(root.is("mesh"), "not an Irrlicht mesh: the root is <{}>", root.name());
    // Other formats' roots are `<mesh>` too: an Irrlicht mesh names its namespace, or at least has buffers.
    let named = root
        .attr("xmlns")
        .is_some_and(|n| n.to_ascii_lowercase().contains("irrlicht"));
    ensure!(
        named || root.children().any(|c| c.is("buffer")),
        "not an Irrlicht mesh: <mesh> has neither Irrlicht's namespace nor buffers"
    );
    let mut mesh = Mesh {
        version: root.attr("version").map(std::borrow::Cow::into_owned),
        ..Mesh::default()
    };
    let mut total = 0usize;
    for b in root.children().filter(|c| c.is("buffer")) {
        let buffer = read_buffer(b, &mut mesh.dropped);
        total += buffer.prim.indices.len() / 3;
        ensure!(total <= limit, "too many triangles to convert (more than {limit})");
        mesh.buffers.push(buffer);
    }
    Ok(mesh)
}

/// The kinds of vertex: how many numbers each has, past its color, and how many coordinate sets.
fn layout(kind: &str) -> (usize, usize) {
    match kind.trim().to_ascii_lowercase().as_str() {
        "2tcoords" => (4, 2),
        "tangents" => (8, 1),
        _ => (2, 1),
    }
}

fn read_buffer(b: El<'_, '_>, dropped: &mut usize) -> Buffer {
    let material = b
        .children()
        .find(|c| c.is("material"))
        .map_or_else(IrrMaterial::default, |m| IrrMaterial::parse(&Attrs::of(m)));
    let mut prim = Primitive::default();
    let mut colors: Vec<u32> = Vec::new();
    let mut sets = 1;
    if let Some(v) = b.children().find(|c| c.is("vertices")) {
        let kind = v.attr("type").unwrap_or_default();
        let (rest, s) = layout(&kind);
        sets = s;
        let tangents = rest == 8;
        let capacity = v.u32("vertexCount").map_or(0, |n| (n as usize).min(1 << 20));
        prim.positions.reserve(capacity);
        prim.normals.reserve(capacity);
        colors.reserve(capacity);
        let mut uvs: Vec<Vec<[f32; 2]>> = (0..sets).map(|_| Vec::with_capacity(capacity)).collect();
        let mut tokens = v.raw().split_ascii_whitespace();
        let mut num = [0f32; 6];
        let mut tail = [0f32; 8];
        'vertices: loop {
            for n in &mut num {
                match tokens.next() {
                    Some(t) => *n = f32_from(t.as_bytes()).unwrap_or(0.0),
                    None => break 'vertices,
                }
            }
            let Some(color) = tokens.next() else { break };
            for n in tail.iter_mut().take(rest) {
                match tokens.next() {
                    Some(t) => *n = f32_from(t.as_bytes()).unwrap_or(0.0),
                    None => break 'vertices,
                }
            }
            prim.positions.push([num[0], num[1], num[2]]);
            prim.normals.push([num[3], num[4], num[5]]);
            colors.push(argb(color));
            uvs[0].push([tail[0], tail[1]]);
            if sets > 1 {
                uvs[1].push([tail[2], tail[3]]);
            }
            if tangents {
                let n = [num[3], num[4], num[5]];
                let t = [tail[2], tail[3], tail[4]];
                let bi = [tail[5], tail[6], tail[7]];
                let c = [
                    n[1] * t[2] - n[2] * t[1],
                    n[2] * t[0] - n[0] * t[2],
                    n[0] * t[1] - n[1] * t[0],
                ];
                let w = if c[0] * bi[0] + c[1] * bi[1] + c[2] * bi[2] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                prim.tangents.push([t[0], t[1], t[2], w]);
            }
        }
        prim.uvs = uvs;
    }
    let count = prim.positions.len();
    if let Some(i) = b.children().find(|c| c.is("indices")) {
        let mut tri = [0u32; 3];
        let mut at = 0;
        for t in i.raw().split_ascii_whitespace() {
            // A bad index spoils its triangle.
            tri[at] = t
                .parse::<u32>()
                .ok()
                .filter(|&x| (x as usize) < count)
                .unwrap_or(u32::MAX);
            at += 1;
            if at == 3 {
                if tri.contains(&u32::MAX) {
                    *dropped += 1;
                } else {
                    prim.indices.extend_from_slice(&tri);
                }
                at = 0;
            }
        }
        if at > 0 {
            *dropped += 1;
        }
    }
    let uniform = match colors.first() {
        Some(&c) if colors.iter().all(|&x| x == c) => Some(c),
        Some(_) => None,
        None => Some(0xFFFF_FFFF),
    };
    if uniform.is_none() {
        let alpha = material.vertex_alpha();
        prim.colors = vec![
            colors
                .iter()
                .map(|&c| {
                    let mut l = linear(c);
                    if !alpha {
                        l[3] = 1.0;
                    }
                    l
                })
                .collect(),
        ];
    }
    Buffer {
        material,
        prim,
        uniform,
        sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_buffers() {
        let text = r#"<?xml version="1.0"?>
<mesh xmlns="http://irrlicht.sourceforge.net/IRRMESH_09_2007" version="1.0">
<buffer><material><enum name="Type" value="lightmap"/></material>
<vertices type="2tcoords" vertexCount="3">
0 0 0 0 0 -1 ff00ff00 0 0 0.5 0.5
1 0 0 0 0 -1 ff00ff00 1 0 1 0.5
0 1 0 0 0 -1 ff00ff00 0 1 0.5 1
</vertices><indices indexCount="7">0 1 2 0 9 1 2</indices></buffer>
<buffer><vertices type="tangents" vertexCount="1">0 0 0 0 0 1 ffffffff 0 0 1 0 0 0 -1 0</vertices></buffer>
</mesh>"#;
        let m = parse(text, 100).unwrap();
        assert_eq!(m.version.as_deref(), Some("1.0"));
        assert_eq!(m.buffers.len(), 2);
        let b = &m.buffers[0];
        assert_eq!(b.material.kind, "lightmap");
        assert_eq!(b.prim.indices, [0, 1, 2]);
        assert_eq!(m.dropped, 2);
        assert_eq!(b.sets, 2);
        assert_eq!(b.prim.uvs[1][1], [1.0, 0.5]);
        assert_eq!(b.uniform, Some(0xFF00_FF00));
        // n × t = (0,0,1) × (1,0,0) = (0,1,0), against a binormal of (0,-1,0).
        assert_eq!(m.buffers[1].prim.tangents, [[1.0, 0.0, 0.0, -1.0]]);
        assert!(parse(text, 0).is_err());
    }
}
