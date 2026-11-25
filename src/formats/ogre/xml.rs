//! The XML `.mesh.xml` and `.skeleton.xml` layouts, as OGRE's converter writes and reads them, read with the
//! shared XML reader.

use std::borrow::Cow;

use super::{
    Assign, Bone, BoneTrack, Geometry, Key, Keys, LodLevel, MeshAnimation, MeshFile, MorphKey, Pose, PoseKey,
    PoseOffset, SkeletonAnimation, SkeletonFile, SubMesh, Track,
};
use crate::Result;
use crate::formats::stream::f32_from;
pub(super) use crate::formats::xml::{El, Xml};

use crate::formats::scene::MAX_UV_SETS;

/// What OGRE reads from an element's attributes beyond the shared reader's.
trait Attrs {
    /// An attribute as a yes or no, as OGRE reads them.
    fn bool(&self, name: &str) -> bool;
    /// The `x`, `y` and `z` attributes.
    fn xyz(&self) -> [f32; 3];
}

impl Attrs for El<'_, '_> {
    fn bool(&self, name: &str) -> bool {
        self.attr(name).is_some_and(|v| truthy(&v))
    }

    fn xyz(&self) -> [f32; 3] {
        [self.f32("x"), self.f32("y"), self.f32("z")]
    }
}

/// A yes or no as OGRE reads one.
pub(super) fn truthy(v: &str) -> bool {
    let v = v.trim();
    ["true", "yes", "1", "on"].iter().any(|t| v.eq_ignore_ascii_case(t))
}

/// A color value: three or four numbers, alpha 1 when left out.
fn color(v: &str) -> [f32; 4] {
    let mut c = [0.0, 0.0, 0.0, 1.0];
    for (o, t) in c.iter_mut().zip(v.split_ascii_whitespace()) {
        *o = f32_from(t.as_bytes()).unwrap_or(0.0);
    }
    c
}

/// Reads a `.mesh.xml` from its parsed document.
pub(super) fn mesh(doc: &Xml<'_>) -> Result<MeshFile> {
    let root = doc.root();
    ensure!(
        root.name() == "mesh",
        UnknownFormat,
        "not an OGRE mesh: its XML is a <{}>",
        root.name()
    );
    let mut file = MeshFile {
        version: "xml".to_owned(),
        ..MeshFile::default()
    };
    if let Some(g) = root.child("sharedgeometry") {
        file.shared = Some(geometry(g, &mut file));
    }
    if let Some(subs) = root.child("submeshes") {
        for s in subs.named("submesh") {
            let sub = submesh(s, &mut file);
            file.submeshes.push(sub);
        }
    }
    if let Some(link) = root.child("skeletonlink") {
        file.skeleton = link.attr("name").map(Cow::into_owned);
    }
    if let Some(b) = root.child("boneassignments") {
        file.shared_bones = assignments(b);
    }
    if let Some(names) = root.child("submeshnames") {
        for n in names.named("submeshname") {
            let index = n.u32("index").map(|i| i as usize);
            if let (Some(s), Some(name)) = (index.and_then(|i| file.submeshes.get_mut(i)), n.attr("name")) {
                s.name = Some(name.into_owned());
            }
        }
    }
    if let Some(lod) = root.child("levelofdetail") {
        file.lod_strategy = lod.attr("strategy").map(Cow::into_owned);
        for l in lod.children() {
            let usage = l
                .opt_f32("value")
                .or_else(|| l.opt_f32("fromdepthsquared"))
                .unwrap_or(0.0);
            match l.name() {
                "lodmanual" => file.lods.push(LodLevel {
                    usage,
                    manual: Some(l.attr("meshname").unwrap_or_default().into_owned()),
                }),
                "lodgenerated" => file.lods.push(LodLevel { usage, manual: None }),
                _ => {}
            }
        }
    }
    if let Some(x) = root.child("extremes") {
        file.extremes = x.named("submesh_extremes").map(|s| s.named("position").count()).sum();
    }
    if let Some(poses) = root.child("poses") {
        for p in poses.named("pose") {
            let target = target(p);
            let offsets = p
                .named("poseoffset")
                .map(|o| PoseOffset {
                    index: o.u32("index").unwrap_or(0),
                    offset: o.xyz(),
                    normal: (o.attr("nx").is_some() && o.attr("ny").is_some() && o.attr("nz").is_some())
                        .then(|| [o.f32("nx"), o.f32("ny"), o.f32("nz")]),
                })
                .collect();
            file.poses.push(Pose {
                name: p.attr("name").unwrap_or_default().into_owned(),
                target,
                offsets,
            });
        }
    }
    if let Some(anims) = root.child("animations") {
        for a in anims.named("animation") {
            let animation = mesh_animation(a, &file)?;
            file.animations.push(animation);
        }
    }
    Ok(file)
}

/// A pose's or track's target: 0 the shared geometry, `k` submesh `k - 1`.
fn target(e: El<'_, '_>) -> u16 {
    if e.attr("target").as_deref() == Some("mesh") {
        0
    } else {
        e.u32("index")
            .and_then(|i| u16::try_from(i).ok())
            .map_or(u16::MAX, |i| i.saturating_add(1))
    }
}

fn mesh_animation(a: El<'_, '_>, file: &MeshFile) -> Result<MeshAnimation> {
    let mut out = MeshAnimation {
        name: a.attr("name").unwrap_or_default().into_owned(),
        length: a.f32("length"),
        base: a.child("baseinfo").map(|b| {
            (
                b.attr("baseanimationname").unwrap_or_default().into_owned(),
                b.f32("basekeyframetime"),
            )
        }),
        tracks: Vec::new(),
    };
    let Some(tracks) = a.child("tracks") else {
        return Ok(out);
    };
    for t in tracks.named("track") {
        let target = target(t);
        let frames = t.child("keyframes");
        let frames = frames.iter().flat_map(|k| k.named("keyframe"));
        let keys = match t.attr("type").as_deref() {
            Some("morph") => {
                let n = super::binary::target_vertices(file, target).unwrap_or(0);
                Keys::Morph(
                    frames
                        .map(|k| {
                            let mut positions = Vec::with_capacity(n.min(1 << 16));
                            let mut normals = Vec::new();
                            let mut p = k.child("position");
                            while let Some(e) = p {
                                if positions.len() >= n {
                                    break;
                                }
                                positions.push(e.xyz());
                                p = e.next_named("position");
                            }
                            let mut q = k.child("normal");
                            while let Some(e) = q {
                                if normals.len() >= n {
                                    break;
                                }
                                normals.push(e.xyz());
                                q = e.next_named("normal");
                            }
                            MorphKey {
                                time: k.f32("time"),
                                positions,
                                normals,
                            }
                        })
                        .collect(),
                )
            }
            Some("pose") => Keys::Pose(
                frames
                    .map(|k| PoseKey {
                        time: k.f32("time"),
                        refs: k
                            .named("poseref")
                            .map(|r| {
                                (
                                    r.u32("poseindex")
                                        .and_then(|i| u16::try_from(i).ok())
                                        .unwrap_or(u16::MAX),
                                    r.opt_f32("influence").unwrap_or(1.0),
                                )
                            })
                            .collect(),
                    })
                    .collect(),
            ),
            other => bail!(
                Corrupt,
                "OGRE animation track type {:?} is unknown",
                other.unwrap_or("")
            ),
        };
        out.tracks.push(Track { target, keys });
    }
    Ok(out)
}

fn assignments(e: El<'_, '_>) -> Vec<Assign> {
    e.named("vertexboneassignment")
        .map(|a| Assign {
            vertex: a.u32("vertexindex").unwrap_or(u32::MAX),
            bone: a
                .u32("boneindex")
                .and_then(|b| u16::try_from(b).ok())
                .unwrap_or(u16::MAX),
            weight: a.f32("weight"),
        })
        .collect()
}

fn submesh(s: El<'_, '_>, file: &mut MeshFile) -> SubMesh {
    let operation = match s.attr("operationtype").as_deref() {
        Some("point_list") => 1,
        Some("line_list") => 2,
        Some("line_strip") => 3,
        Some("triangle_strip") => 5,
        Some("triangle_fan") => 6,
        _ => 4,
    };
    let mut out = SubMesh {
        material: s.attr("material").unwrap_or_default().into_owned(),
        shared: s.bool("usesharedvertices"),
        operation,
        ..SubMesh::default()
    };
    if let Some(faces) = s.child("faces") {
        let mut indices = Vec::new();
        let get = |f: &El<'_, '_>, n: &str| f.u32(n).unwrap_or(0);
        for (i, f) in faces.named("face").enumerate() {
            indices.push(get(&f, "v1"));
            match operation {
                2 => indices.push(get(&f, "v2")),
                4 => {
                    indices.push(get(&f, "v2"));
                    indices.push(get(&f, "v3"));
                }
                5 | 6 if i == 0 => {
                    indices.push(get(&f, "v2"));
                    indices.push(get(&f, "v3"));
                }
                _ => {}
            }
        }
        if !indices.is_empty() {
            out.indices = Some(indices);
        }
    }
    if !out.shared
        && let Some(g) = s.child("geometry")
    {
        out.geometry = Some(geometry(g, file));
    }
    if let Some(b) = s.child("boneassignments") {
        out.bones = assignments(b);
    }
    if let Some(t) = s.child("textures") {
        for a in t.named("texture") {
            if let (Some(alias), Some(name)) = (a.attr("alias"), a.attr("name")) {
                out.aliases.push((alias.into_owned(), name.into_owned()));
            }
        }
    }
    out
}

/// What a texture coordinate dimension attribute gives: the number of values read.
fn texcoord_dims(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some("1" | "float1" | "short1" | "half1") => 1,
        Some("3" | "float3" | "short3" | "half3") => 3,
        Some("4" | "float4" | "short4" | "half4" | "ubyte4") => 4,
        _ => 2,
    }
}

/// A geometry element: its vertex buffers, each giving some of every vertex's parts. The vertex count is how many
/// vertices the first buffer holds, whatever the element claims.
fn geometry(g: El<'_, '_>, file: &mut MeshFile) -> Geometry {
    let mut out = Geometry::default();
    for vb in g.named("vertexbuffer") {
        let positions = vb.bool("positions");
        let normals = vb.bool("normals");
        let tangents = vb.bool("tangents");
        let tangent_w = vb.u32("tangent_dimensions") == Some(4);
        let binormals = vb.bool("binormals");
        let diffuse = vb.bool("colours_diffuse");
        let specular = vb.bool("colours_specular");
        let sets = vb.u32("texture_coords").unwrap_or(0) as usize;
        let dims: Vec<usize> = (0..sets.min(MAX_UV_SETS))
            .map(|i| texcoord_dims(vb.attr(&format!("texture_coord_dimensions_{i}")).as_deref()))
            .collect();
        if sets > MAX_UV_SETS && !file.dropped_elements.iter().any(|d| d == "texcoord_sets_over_8") {
            file.dropped_elements.push("texcoord_sets_over_8".to_owned());
        }
        if dims.iter().any(|&d| d > 2) {
            file.texcoord_w += dims.iter().filter(|&&d| d > 2).count();
        }
        let first_uv = out.uvs.len();
        let room = MAX_UV_SETS.saturating_sub(first_uv);
        let dims = &dims[..dims.len().min(room)];
        out.uvs.extend(dims.iter().map(|_| Vec::new()));
        for v in vb.named("vertex") {
            if positions {
                out.positions.push(v.child("position").map_or([0.0; 3], |p| p.xyz()));
            }
            if normals {
                out.normals.push(v.child("normal").map_or([0.0; 3], |p| p.xyz()));
            }
            if tangents {
                out.tangents
                    .push(v.child("tangent").map_or([0.0, 0.0, 0.0, f32::NAN], |p| {
                        let [x, y, z] = p.xyz();
                        [x, y, z, if tangent_w { p.f32("w") } else { f32::NAN }]
                    }));
            }
            if binormals {
                out.binormals.push(v.child("binormal").map_or([0.0; 3], |p| p.xyz()));
            }
            if diffuse {
                let c = v.child("colour_diffuse").and_then(|c| c.attr("value"));
                out.diffuse.push(c.map_or([1.0; 4], |c| color(&c)));
            }
            if specular {
                let c = v.child("colour_specular").and_then(|c| c.attr("value"));
                out.specular.push(c.map_or([1.0; 4], |c| color(&c)));
            }
            let mut t = v.child("texcoord");
            for (k, &d) in dims.iter().enumerate() {
                let uv = t.map_or([0.0; 2], |e| [e.f32("u"), if d > 1 { e.f32("v") } else { 0.0 }]);
                out.uvs[first_uv + k].push(uv);
                t = t.and_then(|e| e.next_named("texcoord"));
            }
        }
    }
    // Every part as long as the positions, or left out.
    let n = out.positions.len();
    let fit = |len: usize| len == n;
    if !fit(out.normals.len()) {
        out.normals.clear();
    }
    if !fit(out.tangents.len()) {
        out.tangents.clear();
    }
    if !fit(out.binormals.len()) {
        out.binormals.clear();
    }
    if !fit(out.diffuse.len()) {
        out.diffuse.clear();
    }
    if !fit(out.specular.len()) {
        out.specular.clear();
    }
    out.uvs.retain(|s| s.len() == n);
    out
}

/// A rotation as an angle in radians about an axis, as a quaternion (x, y, z, w); none for a zero axis.
fn angle_axis(angle: f32, axis: [f32; 3]) -> [f32; 4] {
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if !(len > 0.0 && len.is_finite() && angle.is_finite()) {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let (s, c) = (angle / 2.0).sin_cos();
    [axis[0] / len * s, axis[1] / len * s, axis[2] / len * s, c]
}

/// A rotation element: its `angle` attribute and `axis` child.
fn rotation(e: Option<El<'_, '_>>) -> [f32; 4] {
    e.map_or([0.0, 0.0, 0.0, 1.0], |r| {
        angle_axis(r.f32("angle"), r.child("axis").map_or([0.0; 3], |a| a.xyz()))
    })
}

/// A scale element: a `factor`, or `x`, `y` and `z` each 1 when left out.
fn scale(e: Option<El<'_, '_>>) -> [f32; 3] {
    e.map_or([1.0; 3], |s| match s.opt_f32("factor") {
        Some(f) => [f; 3],
        None => ["x", "y", "z"].map(|k| s.opt_f32(k).unwrap_or(1.0)),
    })
}

/// Reads a `.skeleton.xml` from its parsed document.
pub(super) fn skeleton(doc: &Xml<'_>) -> Result<SkeletonFile> {
    let root = doc.root();
    ensure!(
        root.name() == "skeleton",
        UnknownFormat,
        "not an OGRE skeleton: its XML is a <{}>",
        root.name()
    );
    let mut file = SkeletonFile {
        version: "xml".to_owned(),
        blend_mode: root.attr("blendmode").map(|m| {
            if m.as_ref() == "cumulative" {
                "cumulative"
            } else {
                "average"
            }
        }),
        ..SkeletonFile::default()
    };
    if let Some(bones) = root.child("bones") {
        for (k, b) in bones.named("bone").enumerate() {
            let handle = b
                .u32("id")
                .and_then(|i| u16::try_from(i).ok())
                .or_else(|| u16::try_from(k).ok())
                .unwrap_or(u16::MAX);
            file.bones.push(Bone {
                name: b.attr("name").unwrap_or_default().into_owned(),
                handle,
                position: b.child("position").map_or([0.0; 3], |p| p.xyz()),
                rotation: rotation(b.child("rotation")),
                scale: scale(b.child("scale")),
                parent: None,
            });
        }
    }
    let by_name = |file: &SkeletonFile, name: &str| file.bones.iter().find(|b| b.name == name).map(|b| b.handle);
    if let Some(h) = root.child("bonehierarchy") {
        for p in h.named("boneparent") {
            let (Some(child), Some(parent)) = (p.attr("bone"), p.attr("parent")) else {
                continue;
            };
            let parent = by_name(&file, &parent);
            if let Some(b) = file.bones.iter_mut().find(|b| b.name == child.as_ref()) {
                b.parent = parent;
            }
        }
    }
    if let Some(anims) = root.child("animations") {
        for a in anims.named("animation") {
            let mut out = SkeletonAnimation {
                name: a.attr("name").unwrap_or_default().into_owned(),
                length: a.f32("length"),
                base: a.child("baseinfo").map(|b| {
                    (
                        b.attr("baseanimationname").unwrap_or_default().into_owned(),
                        b.f32("basekeyframetime"),
                    )
                }),
                tracks: Vec::new(),
            };
            for t in a.child("tracks").iter().flat_map(|t| t.named("track")) {
                let Some(bone) = t.attr("bone").and_then(|n| by_name(&file, &n)) else {
                    continue;
                };
                let keys = t
                    .child("keyframes")
                    .iter()
                    .flat_map(|k| k.named("keyframe"))
                    .map(|k| Key {
                        time: k.f32("time"),
                        rotation: rotation(k.child("rotate")),
                        translation: k.child("translate").map_or([0.0; 3], |p| p.xyz()),
                        scale: scale(k.child("scale")),
                    })
                    .collect();
                out.tracks.push(BoneTrack { bone, keys });
            }
            file.animations.push(out);
        }
    }
    if let Some(links) = root.child("animationlinks") {
        for l in links.named("animationlink") {
            if let Some(name) = l.attr("skeletonName") {
                file.links.push((name.into_owned(), l.opt_f32("scale").unwrap_or(1.0)));
            }
        }
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_axis_rotation() {
        let q = angle_axis(std::f32::consts::PI, [0.0, 2.0, 0.0]);
        assert!((q[1] - 1.0).abs() < 1e-6 && q[3].abs() < 1e-6);
        assert_eq!(angle_axis(1.0, [0.0; 3]), [0.0, 0.0, 0.0, 1.0]);
    }
}
