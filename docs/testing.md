# How polysmelt is tested

A reader is only as good as the files it has been checked against. polysmelt checks each reader
three ways: against files built from the format's specification, against files real tools
wrote, and against what other libraries read from the same files.

## The conformance corpus

`tests/corpus` holds about 1,900 cases. Each is a model file, whatever it refers to (textures,
material libraries, buffers, other layers), and an `expect.json` saying exactly what it must
become: counts, bounds, node names, materials, animation spans, facts, and for broken files the
error kind. The `conformance` test loads every case, writes it as GLB, validates that GLB as
glTF 2.0, decodes it and checks it against the expectations.

```
cargo test --test conformance          # every format
cargo test --test conformance usd      # one format
```

A failing run lists every failed case and what was wrong with it. Every extension polysmelt
reads has cases; a test fails if one doesn't.

### Where the cases come from

- **Generated from the specification.** Most cases are written by Python generators in
  `tools/corpus/`, one per format, which build each file from the format's specification and
  write its expectations from the same knowledge, never from polysmelt's output. They cover the
  variants exporters really write: OBJ's face forms and MTL dialects, both PLY byte orders, FBX
  in both encodings and every axis system, USD's three encodings and its composition, BVH's
  rotation orders, damaged rows, truncated files, wrong magic bytes. `tools/corpus/check.py`
  reruns every generator and fails if what it writes differs from what is checked in.
- **Written by real tools.** Blender 5.2 exports for the formats Blender writes, files SketchUp
  itself saved from SketchUp 3 to 2026, Khronos sample models, and real files from VTK, meshio,
  PCL, PDAL, Cesium and other projects' test data, each with its license beside it.
- **Checked against other libraries.** 75 `ref_*` cases across 21 formats are files from other
  projects, measured by an independent reader: assimp for 3DS, COLLADA, DirectX, LightWave,
  AC3D, AMF, XGL, Irrlicht, OpenGEX, OGRE, OFF, MD2, MD5, SMD, MilkShape 3D, Blitz3D, DXF and
  BVH; lib3mf for 3MF; py-vox-io for MagicaVoxel; ezdxf for DXF lines and points; ezc3d for C3D.
  Each compares what is drawn (triangles, bounds, area and the set of positions) and, where the
  file has them, node names, bone positions at rest and at a key, and animation spans. Where the
  other tool is wrong about a file, the case says so and leaves that part out.

[tests/corpus/README.md](../tests/corpus/README.md) lists every format's cases and sources.

## The glTF validator

Every GLB the conformance tests write is run through the Khronos glTF validator
(`tools/validate`) in CI. The validator predates `KHR_gaussian_splatting` and calls that
extension's attribute names invalid; that one error is let through and no other.

## Cross-format tests

Beyond the per-format corpus, a set of tests holds the readers to the same rules
([consistency.md](consistency.md)):

| Test | What it checks |
| --- | --- |
| `tests/uniform.rs` | One box written in seven formats reads back as the same document; axes, units and root nodes follow one rule. |
| `tests/geometry.rs` | Bad faces are dropped and counted the same way everywhere; normals and the crease angle behave the same in every reader. |
| `tests/materials.rs` | Classic materials convert by one formula; the default material and alpha rule are the same everywhere. |
| `tests/motion.rs` | Animations start at 0, keys are clean, skins follow one rule. |
| `tests/kinds.rs`, `tests/errors.rs` | The same failure is the same error kind in every reader, and every error names its file. |
| `tests/robust.rs` | Damaged, truncated and hostile input fails cleanly or loads what it can, never panics. |
| `tests/fbx_write.rs` | FBX written and read back keeps the document; repeated round trips are stable. |
| `tests/rig.rs` | Blender's armature, exported as FBX and as USD, gives the same rig and the same poses. |

## Running it all

```
cargo test --release                                            # everything
cargo test --release --no-default-features --features minimal   # the dependency-free build
python -I tools/corpus/check.py                                 # generators match the corpus
```

CI runs the tests in the default and minimal builds, the glTF validator, clippy on every feature
on its own and on the WebAssembly build, and a build on the minimum supported Rust.
