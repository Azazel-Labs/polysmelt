# Cargo features

Every reader is a cargo feature, and every one of them is on by default, along with `fs`, `webp`
and the six other texture decoders. Turn off the defaults to build only what you need:

```toml
polysmelt = { version = "0.20", default-features = false, features = ["obj", "gltf", "fs"] }
```

Asking `open` or `read` for a format whose feature is off returns an error that names the
feature, so a trimmed build fails clearly rather than silently.

## The features

| Feature | Format | Extra dependencies |
| --- | --- | --- |
| `stl` | STL | none |
| `ply` | PLY, 3D gaussian splatting's files and SuperSplat's compressed ones included | none |
| `vox` | MagicaVoxel | none |
| `obj` | OBJ + MTL | none |
| `3ds` | 3DS | none |
| `gltf` | glTF 2.0 + GLB, glTF 1.0, and VRM `.vrm`/`.vrma` | none |
| `bvh` | BVH motion capture | none |
| `off` | OFF | none |
| `dxf` | DXF meshes, lines and points | none |
| `splat` | Gaussian splats in the 32-byte `.splat` layout | none |
| `spz` | Gaussian splats in SPZ, versions 1 to 3 (gzip) | `flate2` |
| `sog` | Gaussian splats in SOG, versions 1 to 3 (`.sog`, or `meta.json` and its WebP images) | `zip`, `image-webp` |
| `mmd` | MikuMikuDance `.pmx` (2.0 and 2.1) and `.pmd` models, `.vmd` motions | none |
| `directx` | DirectX `.x` (text, binary, MSZIP-compressed; 32- and 64-bit floats) | none |
| `md2` | Quake II `.md2`, frames as morph targets | none |
| `md3` | Quake III `.md3`, frames as morph targets, tags as nodes | none |
| `md5` | Doom 3 `.md5mesh` and `.md5anim` | none |
| `iqm` | Inter-Quake Model `.iqm` | none |
| `smd` | Valve `.smd` (reference and animation) and `.vta` | none |
| `ms3d` | MilkShape 3D `.ms3d` | none |
| `b3d` | Blitz3D `.b3d` | none |
| `asf` | Acclaim `.asf` skeletons and `.amc` motions | none |
| `c3d` | C3D motion capture markers | none |
| `trc` | TRC motion capture markers | none |
| `htr` | HTR motion capture | none |
| `fbx` | FBX | `flate2` |
| `3mf` | 3MF | `zip`, `quick-xml` |
| `usd` | USD | `openusd`, `openusd-schemas`, `zip` |
| `dae` | COLLADA `.dae`, `.zae` and `.kmz` | `zip`, `quick-xml` |
| `x3d` | X3D and VRML97 (`.x3d`, `.x3dv`, `.wrl`, gzipped `.x3dz`, `.wrz`) | `quick-xml`, `flate2` |
| `skp` | SketchUp `.skp` (through glTF, so it turns on `gltf`) | `flate2` |
| `draco` | Draco-compressed glTF (KHR_draco_mesh_compression) | `draco-core` |
| `webp` | WebP textures (EXT_texture_webp, `.webp` files), re-encoded as PNG for the flat mesh | `image-webp`, `flate2` |
| `tga` | TGA textures (color-mapped, true-color and gray, raw or run-length), re-encoded as PNG | `flate2` |
| `bmp` | BMP textures (1 to 32 bits, RLE4 and RLE8, bit-field masks), re-encoded as PNG | `flate2` |
| `gif` | GIF textures (the first frame), re-encoded as PNG | `flate2` |
| `tiff` | TIFF textures (strips or tiles; uncompressed, LZW, Deflate or PackBits; 1 to 16 bits), re-encoded as PNG | `flate2` |
| `psd` | Photoshop `.psd` and `.psb` textures (the merged image, raw or RLE), re-encoded as PNG | `flate2` |
| `dds` | DDS textures (BC1 to BC5 and uncompressed), re-encoded as PNG | `flate2` |
| `las` | LAS lidar point clouds, 1.0 to 1.4 | none |
| `laz` | LAZ, compressed LAS (turns on `las`) | `laz` |
| `pcd` | PCD point clouds | none |
| `xyz` | Text point clouds: `.xyz`, `.pts`, `.ptx`, `.asc` | none |
| `e57` | E57 scans | `e57` |
| `amf` | AMF, plain or zipped | `quick-xml`, `zip`, `flate2` |
| `lwo` | LightWave objects (`.lwo`) | none |
| `ac3d` | AC3D (`.ac`) | none |
| `ogre` | OGRE `.mesh` and `.skeleton`, binary (v1.10 to v1.100) and XML, with `.material` scripts | none |
| `flt` | OpenFlight (`.flt`), with SGI image textures | none |
| `ogex` | OpenGEX (`.ogex`) | none |
| `irr` | Irrlicht meshes (`.irrmesh`) and scenes (`.irr`) | none |
| `xgl` | XGL (`.xgl`) and zipped XGL (`.zgl`) | none |
| `vtk` | VTK: legacy `.vtk` (ASCII and binary), XML `.vtp` and `.vtu` (zlib-compressed or not) | `flate2` |
| `tiles` | 3D Tiles: `.b3dm`, `.i3dm`, `.pnts` and `.cmpt` tiles, and a `tileset.json` of them or of glTF (needs `gltf`; Draco point clouds with `draco`) | none |
| `msh` | Gmsh meshes (`.msh`), versions 1.0 to 4.1, ASCII and binary | none |
| `medit` | Medit (INRIA) meshes: `.mesh` as text, `.meshb` as binary | none |
| `fs` | Reading files: `open`, `Loader::open`, `assets::Folder`, `gltf::files_in` | none |

## `minimal`

`minimal` is every reader that has no dependencies (thirty-six of them) plus `fs`:

```toml
polysmelt = { version = "0.20", default-features = false, features = ["minimal"] }
```

## Pure Rust

Every dependency in the table is pure Rust, `flate2` included (it uses the `zlib-rs` backend).
Nothing needs a C compiler, a system library or a vendor SDK, which is what lets every feature
build for WebAssembly.

## WebAssembly and other targets without a file system

Without `fs` the file APIs (`open`, `Loader::open`, `assets::Folder`, `gltf::files_in`) don't
exist, so a build for a target with no file system, such as `wasm32-unknown-unknown`, fails to
compile instead of failing when it runs. Models load from memory instead, with the files they
refer to in an `assets::Memory`:

```toml
polysmelt = { version = "0.20", default-features = false, features = ["stl", "ply", "obj", "gltf", "usd"] }
```

```rust ignore
use polysmelt::{Loader, assets::Memory, document::Pose};

fn show(gltf: Vec<u8>, bin: Vec<u8>, png: Vec<u8>) -> polysmelt::Result<()> {
    // The bytes are shared, not copied: the document keeps what it needs of them.
    let assets = Memory::new().with("scene.bin", bin).with("albedo.png", png);
    let mut doc = Loader::new().assets(assets).read(gltf, "scene.gltf")?;
    let glb = doc.to_glb()?; // the scene, losslessly
    let mesh = doc.flatten(Pose::Rest)?; // or flat triangles
    doc.decode()?; // or plain glTF, for a reader without Draco, meshopt or quantization
    Ok(())
}
```

CI builds each feature on its own, and every reader together for `wasm32-unknown-unknown`.
The [live demo](https://azazel-labs.github.io/polysmelt/) is polysmelt compiled to WebAssembly
with nearly every reader.

## Minimum Rust

Rust 1.96, which `openusd` (the `usd` feature, on by default) needs. CI builds everything on
1.96.
