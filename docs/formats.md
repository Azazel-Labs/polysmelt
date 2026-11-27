# Formats

`polysmelt::open` and `polysmelt::read` pick the format by the file's extension, or else by its
first bytes, and return a `Document`. `Format` names each format and its extensions. Every format
is a [cargo feature](features.md), all on by default.

Every reader follows the same rules for axes, units, normals, materials, animation and errors;
see [consistency.md](consistency.md). Each one is tested against files built from its
specification and, for most, files that real tools wrote; see [testing.md](testing.md).

## Scenes and interchange

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| FBX | `.fbx` | `fbx` | Binary and ASCII FBX 7.x (older files are refused as `Unsupported`), every axis system. The node tree, meshes, materials and textures, skins, blend shapes, cameras and lights; animation with Bezier and TCB curves, Euler rotations, pivots and layers, sampled into linear keys; line geometry and NURBS curves as lines. |
| USD | `.usda`, `.usdc`, `.usdz` | `usd` | All three encodings and composition. Meshes, materials, `UsdSkel` skeletons and skins, transform stacks and time samples; basis, NURBS and Hermite curves and points. |
| glTF 2.0 | `.gltf`, `.glb` | `gltf` | Passed through untouched, Draco and meshopt compression included. |
| glTF 1.0 | `.gltf`, `.glb` | `gltf` | With `KHR_binary_glTF` and `KHR_materials_common`, upgraded to 2.0 as it's read. |
| COLLADA | `.dae`, `.zae`, `.kmz` | `dae` | The node tree, instanced geometry, materials and effects, skins, cameras and lights, lines and line strips; animation with Bezier, Hermite, B-spline and cardinal curves, stacks of transforms, and animation clips. |
| 3DS | `.3ds` | `3ds` | Objects, materials and texture maps, the keyframer's hierarchy and animation (its cubic splines kept as glTF cubic splines), cameras and lights. The master scale is applied. |
| X3D and VRML97 | `.x3d`, `.x3dv`, `.x3dz`, `.wrl`, `.wrz` | `x3d` | XML and classic encodings, gzipped or not. Shapes, line and point sets, Extrusion, the 2D nodes, PROTO and EXTERNPROTO instances, materials, textures, lights and viewpoints. |
| OpenGEX | `.ogex` | `ogex` | Nodes and their transforms, meshes, materials and textures, skins, morph targets, lights, cameras and animation tracks (cubic splines kept). |

## Meshes and 3D printing

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| OBJ | `.obj` + `.mtl` | `obj` | Every face form and the MTL dialects exporters write, materials and textures, groups, smoothing groups, lines and points. Large files can be read on several threads (`Keep::threads`). |
| STL | `.stl` | `stl` | Binary and ASCII, several solids in one file. |
| PLY | `.ply` | `ply` | ASCII and binary in both byte orders, every property type, vertex colors and custom properties, faces, edges and point clouds. |
| OFF | `.off` | `off` | Text and binary, with vertex and face colors. |
| 3MF | `.3mf` | `3mf` | Objects, components and build items, base materials, color groups, textures and units. |
| AMF | `.amf` | `amf` | Plain or zipped: objects, volumes and their materials, vertex and triangle colors, textures and constellations. Curved triangles stay flat. |

## CAD and modeling

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| SketchUp | `.skp` | `skp` | SketchUp 3 to the latest, without the SketchUp SDK. Groups and components, faces, edges, materials and textures, soft edges as smoothing. |
| DXF | `.dxf` | `dxf` | Text or binary. 3D faces, polyface and polygon meshes, MESH entities, lines, points, 2D and 3D polylines with their arcs, circles, arcs, ellipses and splines, and block inserts. |
| LightWave | `.lwo` | `lwo` | LWOB, LWO2 and LWO3: layers and pivots, surfaces and their image maps, UV and color maps, smoothing angles and morph targets. |
| AC3D | `.ac` | `ac3d` | The object tree, materials, textures, crease angles, lines and triangle strips. |
| XGL | `.xgl`, `.zgl` | `xgl` | Objects and their transforms, meshes placed by reference, materials, lines and points, a directional light. |

## Games and characters

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| VRM | `.vrm`, `.vrma` | `gltf` | VRM 0.x and 1.0 avatars and VRM animations, through the glTF reader. |
| MikuMikuDance | `.pmx`, `.pmd`, `.vmd` | `mmd` | PMX 2.0 and 2.1 and PMD models, skinned, with morph targets, IK and materials; VMD motions with their Bezier eases. |
| DirectX | `.x` | `directx` | Text, binary and MSZIP-compressed, 32- and 64-bit floats: frames, meshes, materials, skins and animation sets. |
| Quake II | `.md2` | `md2` | Frames as morph targets, and their animations. |
| Quake III | `.md3` | `md3` | Frames as morph targets, tags as nodes. |
| Doom 3 | `.md5mesh`, `.md5anim` | `md5` | Skinned meshes and their animations. |
| Inter-Quake Model | `.iqm` | `iqm` | Skinned meshes, joints and animations. |
| Valve SMD | `.smd`, `.vta` | `smd` | Reference meshes, animations, and vertex animation as morph targets. |
| MilkShape 3D | `.ms3d` | `ms3d` | Skinned and animated meshes and their materials. |
| Blitz3D | `.b3d` | `b3d` | Skinned and animated meshes, materials and texture flags. |
| OGRE | `.mesh`, `.skeleton`, `.material` | `ogre` | Binary (v1.10 to v1.100, either byte order) and XML: meshes, skeletons and animations, poses and morphs as morph targets, and `.material` scripts as materials. |
| Irrlicht | `.irrmesh`, `.irr` | `irr` | Meshes with buffers of each vertex type and their materials; scenes with the node tree, cubes, spheres, sky boxes and billboards, mesh nodes naming other files, lights, cameras, and animators as animations. |
| MagicaVoxel | `.vox` | `vox` | Models, the scene graph, the palette and its materials, and animation frames. |

## Motion capture

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| BVH | `.bvh` | `bvh` | The skeleton as nodes and the motion as one animation, in every rotation order. |
| Acclaim | `.asf`, `.amc` | `asf` | Skeletons and their motions. |
| C3D | `.c3d` | `c3d` | Marker tracks, each marker an animated node, in the file's units. |
| TRC | `.trc` | `trc` | Marker tracks, each marker an animated node. |
| HTR | `.htr` | `htr` | Skeletons and their motion, in the file's units and axes. |

## Simulation

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| VTK | `.vtk`, `.vtp`, `.vtu` | `vtk` | Legacy (ASCII and binary) and XML (zlib-compressed or not): poly data, unstructured and structured grids as triangles, lines and points, volume cells as their boundary faces, point and cell data as normals, texture coordinates, colors and custom attributes. |
| Gmsh | `.msh` | `msh` | Versions 1.0 to 4.1, ASCII and binary in either byte order: every element type by its corner nodes, each physical group a mesh of its own, node and element data as custom attributes. |
| Medit | `.mesh`, `.meshb` | `medit` | Text and binary, versions 1 to 4: vertices, edges, surface and volume elements, each reference a mesh of its own, normals and tangents. |
| OpenFlight | `.flt` | `flt` | The node tree with levels of detail, switches, instances, replication and references to other files, faces and meshes, color, material and texture palettes with SGI images, light sources and animated groups. |

## Scans and geospatial

Points come out as point primitives with their colors. Other per-point fields (intensity,
classification, return numbers, GPS time and so on) become vertex attributes, and the coordinate
system goes in `extras`. Positions far from the origin keep their precision in a node translation.

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| LAS | `.las` | `las` | LAS 1.0 to 1.4, point formats 0 to 10. |
| LAZ | `.laz` | `laz` | Compressed LAS. |
| E57 | `.e57` | `e57` | Every scan, placed by its pose. |
| PCD | `.pcd` | `pcd` | ASCII, binary and binary_compressed, with normals and custom fields. |
| Text clouds | `.xyz`, `.pts`, `.ptx`, `.asc` | `xyz` | Including `.ptx` scanner transforms and ESRI ASCII grids. |
| OGC 3D Tiles | `.b3dm`, `.i3dm`, `.pnts`, `.cmpt`, `tileset.json` | `tiles` | Batched models with their batch tables, instanced models as `EXT_mesh_gpu_instancing`, point clouds (quantized, oct-encoded and Draco-compressed), composites, and a tileset of them or of glTF at its finest level of detail, each tile under its transform. |

## Gaussian splats

Splats come out as `KHR_gaussian_splatting`, which the glTF reader reads too. `to_glb` keeps every
splat's spherical harmonics, scale and rotation, and `flatten` draws them as points in their base
color.

| Format | Extensions | Feature | What's read |
| --- | --- | --- | --- |
| 3D gaussian splatting | `.ply` | `ply` | The PLY files 3DGS training writes, and SuperSplat's compressed ones. |
| `.splat` | `.splat` | `splat` | The 32-byte layout. |
| SPZ | `.spz` | `spz` | Gzipped, versions 1 to 3. |
| SOG | `.sog`, `meta.json` | `sog` | PlayCanvas's format, zipped or as `meta.json` beside its WebP images. |

## Textures

PNG and JPEG are embedded as they are. The others are re-encoded as PNG, each behind its own
feature. To convert other formats, override `Assets::image`.

| Format | Feature | What's read |
| --- | --- | --- |
| WebP | `webp` | `EXT_texture_webp` and `.webp` files. |
| TGA | `tga` | Color-mapped, true-color and gray, raw or run-length. |
| BMP | `bmp` | 1 to 32 bits, RLE4 and RLE8, bit-field masks. |
| GIF | `gif` | The first frame. |
| TIFF | `tiff` | Strips or tiles; uncompressed, LZW, Deflate or PackBits; 1 to 16 bits. |
| PSD | `psd` | `.psd` and `.psb`: the merged image, raw or RLE. |
| DDS | `dds` | BC1 to BC5 and uncompressed. |
| SGI | `flt` | OpenFlight's textures. |

The XML formats read UTF-16 files too.

## Writing

| Format | How | What's written |
| --- | --- | --- |
| GLB | `Document::to_glb`, `Document::write_glb` | The whole document, images embedded. glTF input goes back out as it came, compression included. |
| GLB, compressed | `gltf::compress`, `gltf::compress_with` | `EXT_meshopt_compression` or `KHR_meshopt_compression`, losslessly. See [gltf-tools.md](gltf-tools.md). |
| FBX | `Document::to_fbx`, `Document::write_fbx` | Binary FBX 7.4 (7.5 past 4 GiB): the node tree with instancing, meshes with every UV and color set, PBR materials with embedded textures, skins and bind poses, blend shapes, each animation as a take, cameras, lights, and extras as user properties. Points have no FBX form and are left out. The same input gives the same bytes every time. |
| Flat mesh | `Document::flatten`, `mesh::to_glb` | Posed triangles, lines and points with their materials, as a `mesh::Mesh` or a plain GLB. See [guide.md](guide.md). |
