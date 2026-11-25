//! Meshes as FBX geometry: the triangles as polygons on control points (the primitives' vertices, one after
//! another), with normals, tangents, UV and color sets as layer elements and a material per polygon; lines as
//! `Line` geometry.

use serde_json::Value;

use super::records::{Node, props70};
use super::{Writer, fbx_name, index, list, to_srgb};
use crate::Result;

/// A mesh as written.
#[derive(Clone, Debug)]
pub(super) struct Geom {
    /// The triangles' geometry, when there are any.
    pub(super) id: Option<i64>,
    /// The material of each slot the polygons name; `None` for primitives without one (always the last slot).
    pub(super) slots: Vec<Option<usize>>,
    /// The lines' geometry and its material.
    pub(super) lines: Option<(i64, Option<usize>)>,
    /// By primitive, where its vertices start among the control points (triangle primitives only).
    pub(super) bases: Vec<Option<usize>>,
}

/// A primitive's attribute accessor.
fn attribute(prim: &Value, name: &str) -> Option<usize> {
    prim.get("attributes").and_then(|a| index(a, name))
}

/// How many sets of `prefix` (`TEXCOORD_`, `COLOR_`) a primitive has, numbered from 0 without a gap.
fn sets(prim: &Value, prefix: &str) -> usize {
    (0..)
        .take_while(|k| attribute(prim, &format!("{prefix}{k}")).is_some())
        .count()
}

/// A primitive's mode: 4 (triangles) unless it says.
fn mode(prim: &Value) -> u64 {
    prim.get("mode").and_then(Value::as_u64).unwrap_or(4)
}

/// The triangles of a primitive drawing `mode` with `idx`, as vertex indices.
fn triangles(mode: u64, idx: &[u32], out: &mut Vec<[u32; 3]>) {
    match mode {
        5 => {
            for i in 0..idx.len().saturating_sub(2) {
                out.push(if i % 2 == 0 {
                    [idx[i], idx[i + 1], idx[i + 2]]
                } else {
                    [idx[i], idx[i + 2], idx[i + 1]]
                });
            }
        }
        6 => {
            for i in 1..idx.len().saturating_sub(1) {
                out.push([idx[i], idx[i + 1], idx[0]]);
            }
        }
        _ => out.extend(idx.chunks_exact(3).map(|t| [t[0], t[1], t[2]])),
    }
}

/// A layer element of values indexed by polygon vertex.
fn element(
    kind: &'static str,
    n: i32,
    name: &str,
    values: (&'static str, &[f64]),
    index: (&'static str, &[i32]),
) -> Node {
    Node::new(kind)
        .i32(n)
        .child("Version", |v| v.i32(101))
        .child("Name", |v| v.str(name))
        .child("MappingInformationType", |v| v.str("ByPolygonVertex"))
        .child("ReferenceInformationType", |v| v.str("IndexToDirect"))
        .child(values.0, |v| v.arr(values.1))
        .child(index.0, |v| v.arr(index.1))
}

/// A `Layer` record naming the elements it holds.
fn layer(n: i32, elements: &[&str]) -> Node {
    let mut l = Node::new("Layer").i32(n).child("Version", |v| v.i32(100));
    for e in elements {
        l.push(
            Node::new("LayerElement")
                .child("Type", |v| v.str(e))
                .child("TypedIndex", |v| v.i32(n)),
        );
    }
    l
}

/// A count as an FBX index, or an error when it's past what one holds.
fn fbx_index(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| err!(LimitExceeded, "a mesh with {n} vertices is more than FBX indexes"))
}

impl Writer<'_> {
    /// Writes mesh `mesh`'s geometry; `None` when it draws nothing FBX holds.
    #[allow(clippy::too_many_lines, reason = "one layer after another")]
    pub(super) fn geometry(&mut self, mesh: usize) -> Result<Option<Geom>> {
        let doc = self.doc;
        let mesh_json = &list(self.json, "meshes")[mesh];
        let name = mesh_json
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .map_or_else(|| format!("mesh_{mesh}"), str::to_owned);
        let prims: &[Value] = mesh_json
            .get("primitives")
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice);
        let tri_prims: Vec<usize> = (0..prims.len())
            .filter(|&p| matches!(mode(&prims[p]), 4..=6) && attribute(&prims[p], "POSITION").is_some())
            .collect();
        let materials = list(self.json, "materials").len();
        let material_of = |p: &Value| index(p, "material").filter(|&m| m < materials);

        // Positions and triangles first: they size everything else.
        let mut verts: Vec<f64> = Vec::new();
        let mut polys: Vec<i32> = Vec::new();
        let mut poly_slot: Vec<i32> = Vec::new();
        let mut bases = vec![None; prims.len()];
        let mut slots: Vec<Option<usize>> = Vec::new();
        // By triangle primitive: (base, vertex count, triangles from, triangles to).
        let mut spans: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut tris = Vec::new();
        for &p in &tri_prims {
            let prim = &prims[p];
            let (w, pos) = doc.read_floats(attribute(prim, "POSITION").unwrap_or(0))?;
            if w != 3 {
                return Err(err!(Corrupt, "mesh {mesh} primitive {p} has positions that aren't 3D"));
            }
            let n = pos.len() / 3;
            let base = verts.len() / 3;
            fbx_index(base + n)?;
            tris.clear();
            match index(prim, "indices") {
                Some(a) => {
                    let idx = doc.read_indices(a)?;
                    if let Some(bad) = idx.iter().find(|&&i| i as usize >= n) {
                        return Err(err!(Corrupt, "mesh {mesh} primitive {p} names vertex {bad} of {n}"));
                    }
                    triangles(mode(prim), &idx, &mut tris);
                }
                None => {
                    let idx: Vec<u32> = (0..u32::try_from(n).unwrap_or(u32::MAX)).collect();
                    triangles(mode(prim), &idx, &mut tris);
                }
            }
            if tris.is_empty() {
                continue;
            }
            bases[p] = Some(base);
            verts.extend(pos.iter().map(|&c| f64::from(c)));
            let m = material_of(prim);
            let slot = slots.iter().position(|s| *s == m).unwrap_or_else(|| {
                slots.push(m);
                slots.len() - 1
            });
            let from = polys.len() / 3;
            polys.reserve(tris.len() * 3);
            // Within i32: the base and count are checked above.
            #[allow(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "checked to fit"
            )]
            for t in &tris {
                let b = base as i32;
                polys.extend_from_slice(&[b + t[0] as i32, b + t[1] as i32, !(b + t[2] as i32)]);
            }
            #[allow(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "a slot per primitive"
            )]
            poly_slot.extend(std::iter::repeat_n(slot as i32, tris.len()));
            spans.push((base, n, from, polys.len() / 3));
        }
        // The material-less slot goes last, after the materials primitives name.
        if let Some(none) = slots.iter().position(Option::is_none)
            && none + 1 != slots.len()
        {
            let last = i32::try_from(slots.len() - 1).unwrap_or(0);
            let none_i = i32::try_from(none).unwrap_or(0);
            for s in &mut poly_slot {
                *s = match *s {
                    s if s == none_i => last,
                    s if s > none_i => s - 1,
                    s => s,
                };
            }
            slots.remove(none);
            slots.push(None);
        }
        let id = if polys.is_empty() {
            None
        } else {
            Some(self.triangle_geometry(&name, prims, &tri_prims, &bases, verts, &polys, &poly_slot, slots.len())?)
        };

        let lines = self.line_geometry(mesh, &name, prims)?;
        if id.is_none() && lines.is_none() {
            return Ok(None);
        }
        Ok(Some(Geom {
            id,
            slots,
            lines,
            bases,
        }))
    }

    /// Writes the triangles' geometry with its layers; returns its id.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the pieces geometry() gathered"
    )]
    fn triangle_geometry(
        &mut self,
        name: &str,
        prims: &[Value],
        tri_prims: &[usize],
        bases: &[Option<usize>],
        verts: Vec<f64>,
        polys: &[i32],
        poly_slot: &[i32],
        nslots: usize,
    ) -> Result<i64> {
        let doc = self.doc;
        let points = verts.len() / 3;
        let used: Vec<usize> = tri_prims.iter().copied().filter(|&p| bases[p].is_some()).collect();
        // Each polygon vertex's control point, which indexes every layer.
        let pv_cp: Vec<i32> = polys.iter().map(|&i| if i < 0 { !i } else { i }).collect();

        // Normals: a primitive's own, or flat ones made from its triangles.
        let mut normals = vec![0.0f64; points * 3];
        let mut normal_index: Option<Vec<i32>> = None;
        let mut has_normals = vec![false; prims.len()];
        for &p in &used {
            let base = bases[p].unwrap_or(0);
            if let Some(a) = attribute(&prims[p], "NORMAL") {
                let (w, n) = doc.read_floats(a)?;
                if w == 3 {
                    let to = (base * 3 + n.len()).min(normals.len());
                    for (d, s) in normals[base * 3..to].iter_mut().zip(&n) {
                        *d = f64::from(*s);
                    }
                    has_normals[p] = true;
                }
            }
        }
        if used.iter().any(|&p| !has_normals[p]) {
            let mut ix = pv_cp.clone();
            for &p in used.iter().filter(|&&p| !has_normals[p]) {
                let base = bases[p].unwrap_or(0);
                let end = base + self.vertex_count(&prims[p])?;
                // This primitive's triangles are the polygons whose corners are its control points.
                for (t, tri) in pv_cp.chunks_exact(3).enumerate() {
                    let c = usize::try_from(tri[0]).unwrap_or(0);
                    if c < base || c >= end {
                        continue;
                    }
                    let at = |k: usize| {
                        let v = usize::try_from(tri[k]).unwrap_or(0) * 3;
                        [verts[v], verts[v + 1], verts[v + 2]]
                    };
                    let (a, b, c) = (at(0), at(1), at(2));
                    let n = crate::vec3::unit_or(
                        crate::vec3::cross(crate::vec3::sub(b, a), crate::vec3::sub(c, a)),
                        [0.0, 0.0, 1.0],
                    );
                    let k = fbx_index(normals.len() / 3)?;
                    normals.extend_from_slice(&n);
                    ix[t * 3..t * 3 + 3].fill(k);
                }
            }
            normal_index = Some(ix);
        }

        // Tangents, and binormals for their handedness, when every primitive has them.
        let tangents = if used
            .iter()
            .all(|&p| has_normals[p] && attribute(&prims[p], "TANGENT").is_some())
        {
            let mut t = vec![0.0f64; points * 3];
            let mut b = vec![0.0f64; points * 3];
            for &p in &used {
                let base = bases[p].unwrap_or(0);
                let (w, tan) = doc.read_floats(attribute(&prims[p], "TANGENT").unwrap_or(0))?;
                if w != 4 {
                    continue;
                }
                for (i, q) in tan.chunks_exact(4).enumerate() {
                    let v = base + i;
                    if v >= points {
                        break;
                    }
                    let tv = [q[0], q[1], q[2]].map(f64::from);
                    let n = [normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]];
                    let bv = crate::vec3::cross(n, tv).map(|c| c * f64::from(q[3].signum()));
                    t[v * 3..v * 3 + 3].copy_from_slice(&tv);
                    b[v * 3..v * 3 + 3].copy_from_slice(&bv);
                }
            }
            Some((t, b))
        } else {
            None
        };

        // UV and color sets, as many as the primitive with the most has; the others fill theirs with zeros (UVs)
        // or white (colors).
        let uv_sets = used.iter().map(|&p| sets(&prims[p], "TEXCOORD_")).max().unwrap_or(0);
        let color_sets = used.iter().map(|&p| sets(&prims[p], "COLOR_")).max().unwrap_or(0);
        let mut uvs: Vec<Vec<f64>> = Vec::with_capacity(uv_sets);
        for k in 0..uv_sets {
            let mut v = vec![0.0f64; points * 2];
            for &p in &used {
                let base = bases[p].unwrap_or(0);
                let Some(a) = attribute(&prims[p], &format!("TEXCOORD_{k}")) else {
                    continue;
                };
                let (w, uv) = doc.read_floats(a)?;
                if w != 2 {
                    continue;
                }
                let to = (base * 2 + uv.len()).min(v.len());
                // FBX's V runs up, glTF's down.
                for (d, s) in v[base * 2..to].chunks_exact_mut(2).zip(uv.chunks_exact(2)) {
                    d[0] = f64::from(s[0]);
                    d[1] = 1.0 - f64::from(s[1]);
                }
            }
            uvs.push(v);
        }
        let mut colors: Vec<Vec<f64>> = Vec::with_capacity(color_sets);
        for k in 0..color_sets {
            let mut v = vec![1.0f64; points * 4];
            for &p in &used {
                let base = bases[p].unwrap_or(0);
                let Some(a) = attribute(&prims[p], &format!("COLOR_{k}")) else {
                    continue;
                };
                let (w, c) = doc.read_floats(a)?;
                if w != 3 && w != 4 {
                    continue;
                }
                // FBX colors are sRGB-encoded, glTF's linear.
                for (i, s) in c.chunks_exact(w).enumerate() {
                    let at = (base + i) * 4;
                    if at + 4 > v.len() {
                        break;
                    }
                    for ch in 0..3 {
                        v[at + ch] = to_srgb(f64::from(s[ch]));
                    }
                    v[at + 3] = if w == 4 { f64::from(s[3]) } else { 1.0 };
                }
            }
            colors.push(v);
        }

        let id = self.id();
        let mut g = Node::new("Geometry")
            .i64(id)
            .str(&fbx_name(name, "Geometry"))
            .str("Mesh")
            .with(props70(Vec::new()))
            .child("GeometryVersion", |v| v.i32(124))
            .child("Vertices", |v| v.arr(&verts))
            .child("PolygonVertexIndex", |v| v.arr(polys));
        drop(verts);
        g.push(element(
            "LayerElementNormal",
            0,
            "",
            ("Normals", &normals),
            ("NormalsIndex", normal_index.as_deref().unwrap_or(&pv_cp)),
        ));
        if let Some((t, b)) = &tangents {
            g.push(element(
                "LayerElementBinormal",
                0,
                "",
                ("Binormals", b),
                ("BinormalsIndex", &pv_cp),
            ));
            g.push(element(
                "LayerElementTangent",
                0,
                "",
                ("Tangents", t),
                ("TangentsIndex", &pv_cp),
            ));
        }
        for (k, v) in uvs.iter().enumerate() {
            let n = fbx_index(k)?;
            g.push(element(
                "LayerElementUV",
                n,
                &format!("TEXCOORD_{k}"),
                ("UV", v),
                ("UVIndex", &pv_cp),
            ));
        }
        for (k, v) in colors.iter().enumerate() {
            let n = fbx_index(k)?;
            g.push(element(
                "LayerElementColor",
                n,
                &format!("COLOR_{k}"),
                ("Colors", v),
                ("ColorIndex", &pv_cp),
            ));
        }
        let (mapping, mats): (&str, &[i32]) = if nslots <= 1 {
            ("AllSame", &[0])
        } else {
            ("ByPolygon", poly_slot)
        };
        g.push(
            Node::new("LayerElementMaterial")
                .i32(0)
                .child("Version", |v| v.i32(101))
                .child("Name", |v| v.str(""))
                .child("MappingInformationType", |v| v.str(mapping))
                .child("ReferenceInformationType", |v| v.str("IndexToDirect"))
                .child("Materials", |v| v.arr(mats)),
        );
        let mut first = vec!["LayerElementNormal"];
        if tangents.is_some() {
            first.extend(["LayerElementBinormal", "LayerElementTangent"]);
        }
        first.push("LayerElementMaterial");
        if !colors.is_empty() {
            first.push("LayerElementColor");
        }
        if !uvs.is_empty() {
            first.push("LayerElementUV");
        }
        g.push(layer(0, &first));
        for k in 1..uv_sets.max(color_sets) {
            let mut more = Vec::new();
            if k < color_sets {
                more.push("LayerElementColor");
            }
            if k < uv_sets {
                more.push("LayerElementUV");
            }
            g.push(layer(fbx_index(k)?, &more));
        }
        self.object("Geometry", g);
        Ok(id)
    }

    /// How many vertices a primitive has.
    fn vertex_count(&self, prim: &Value) -> Result<usize> {
        let a = attribute(prim, "POSITION").unwrap_or(0);
        let acc = list(self.json, "accessors")
            .get(a)
            .ok_or_else(|| err!(Corrupt, "accessor {a} doesn't exist"))?;
        Ok(index(acc, "count").unwrap_or(0))
    }

    /// Writes a mesh's lines as one `Line` geometry, with the first line primitive's material.
    fn line_geometry(&mut self, mesh: usize, name: &str, prims: &[Value]) -> Result<Option<(i64, Option<usize>)>> {
        let doc = self.doc;
        let materials = list(self.json, "materials").len();
        let mut points: Vec<f64> = Vec::new();
        let mut index_list: Vec<i32> = Vec::new();
        let mut material = None;
        let mut any = false;
        for (p, prim) in prims.iter().enumerate() {
            let m = mode(prim);
            if !matches!(m, 1..=3) {
                continue;
            }
            let Some(pos) = attribute(prim, "POSITION") else {
                continue;
            };
            let (w, pos) = doc.read_floats(pos)?;
            if w != 3 {
                continue;
            }
            let n = pos.len() / 3;
            let base = points.len() / 3;
            fbx_index(base + n)?;
            let idx: Vec<u32> = match index(prim, "indices") {
                Some(a) => doc.read_indices(a)?,
                None => (0..u32::try_from(n).unwrap_or(u32::MAX)).collect(),
            };
            if let Some(bad) = idx.iter().find(|&&i| i as usize >= n) {
                return Err(err!(Corrupt, "mesh {mesh} primitive {p} names vertex {bad} of {n}"));
            }
            #[allow(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "checked to fit"
            )]
            let at = |i: u32| (base + i as usize) as i32;
            let before = index_list.len();
            match m {
                1 => {
                    for s in idx.chunks_exact(2) {
                        index_list.extend_from_slice(&[at(s[0]), !at(s[1])]);
                    }
                }
                _ if idx.len() >= 2 => {
                    index_list.extend(idx.iter().map(|&i| at(i)));
                    if m == 2 {
                        index_list.push(at(idx[0]));
                    }
                    if let Some(last) = index_list.last_mut() {
                        *last = !*last;
                    }
                }
                _ => {}
            }
            if index_list.len() == before {
                continue;
            }
            points.extend(pos.iter().map(|&c| f64::from(c)));
            if !any {
                material = index(prim, "material").filter(|&m| m < materials);
                any = true;
            }
        }
        if !any {
            return Ok(None);
        }
        let id = self.id();
        let g = Node::new("Geometry")
            .i64(id)
            .str(&fbx_name(name, "Geometry"))
            .str("Line")
            .with(props70(Vec::new()))
            .child("GeometryVersion", |v| v.i32(124))
            .child("Type", |v| v.str("Line"))
            .child("LineVersion", |v| v.i32(100))
            .child("Points", |v| v.arr(&points))
            .child("PointsIndex", |v| v.arr(&index_list));
        self.object("Geometry", g);
        Ok(Some((id, material)))
    }
}
