# Changelog

Breaking changes are allowed in minor releases until 1.0.

## 0.17.0

This release reshapes the public API ahead of 1.0. Most code needs only the small changes
listed under Migration.

### Added

- `polysmelt::open(path)` and `polysmelt::read(bytes, name)` read any supported model into a
  `Document`.
- `Loader` sets the `Keep` options, the `Assets` source and the `Format` before a read, as in
  `Loader::new().keep(keep).assets(assets).open(path)`.
- `Format` lists the supported formats, with `from_extension`, `detect`, `extensions` and
  `name`.
- `Error` has `kind()`, `path()` and `format()`, and `source()` returns the underlying zip,
  XML, JSON or decoder error. `ErrorKind` has `UnknownFormat`, `FeatureDisabled`,
  `Unsupported`, `Corrupt`, `Truncated`, `Empty`, `MissingAsset`, `LimitExceeded`, `Io` and
  `Invalid`.
- `Document::to_glb_with(&CompressOptions)`, `Document::write_glb(writer)`,
  `Document::animation_bounds`, `Document::json()` and `json_mut()`, plus `Clone`, `Default`
  and `From<Mesh>`.
- `rig::Channel::sample_into` samples into a reused buffer without allocating.
- `CompressOptions` setters that chain: `CompressOptions::KHR.level(3).color_filter(true)`.
- `Keep` setters that chain: `Keep::ALL.with_animations(false)`.
- `meshopt::Filter`, `source::ParseAxisError`, and `FromStr`/`Display` for `Axis`.
- `PartialEq` on `Mesh`, `Prim` and `Material`, and `Default` on `UvTransform`.
- `examples/convert.rs` and `examples/inspect.rs`.

### Changed

- Rust 1.96 is now the minimum, because the `usd` feature (on by default) needs it.
- A `Document` owns its data and has no lifetime. Reads are eager, so the source file is not
  held open until the GLB is written.
- Error messages name the file that failed. Truncated data reads "is cut short" or "runs past
  the end of" in every reader.
- A file that is well formed but has nothing to convert is `ErrorKind::Empty`, not an error
  about corrupt data.
- `Rig::world` no longer panics on a bad parent index.

### Removed

- The `formats` module is private. Use `open`, `read`, `Loader` and `Format`.
- `Assets::reads_files`.
- The `From` impls on `Error` for `ParseIntError`, `ParseFloatError`, `TryFromIntError`,
  `TryFromSliceError`, `Utf8Error`, `serde_json::Error`, `zip::result::ZipError` and
  `quick_xml::Error`. `From<std::io::Error>` stays.
- The `thiserror` dependency.

### Migration

| 0.16 | 0.17 |
|---|---|
| `formats::open(p, keep)` | `polysmelt::open(p)`, or `Loader::new().keep(keep).open(p)` |
| `formats::load(path, ext, &assets, keep)` | `Loader::new().assets(assets).keep(keep).open(path)`, adding `.format(Format::X)` when the extension is wrong |
| `formats::load_bytes(bytes, name, ext, &assets, keep)` | `Loader::new().assets(assets).keep(keep).read(bytes, name)` |
| `formats::detect(b)` | `Format::detect(b)` |
| `Document<'a>`, `.into_owned()` | `Document` |
| `doc.json` | `doc.json()` / `doc.json_mut()` |
| `Document::empty(facts)` / `Document::from_mesh(m, facts)` | `Document::default()` / `Document::from(m)` |
| `Keep { animations: false, ..Keep::ALL }` | `Keep::ALL.with_animations(false)` |
| `Memory::get` returns `Option<&[u8]>` | returns `Option<&Bytes>`; `insert` and `with` take `impl Into<Bytes>` |
| `match err { Error::InvalidInput(msg) => .. }` | `match err.kind() { ErrorKind::Corrupt => .., _ => .. }`, with `err.to_string()` for the text |
| `Error::Io(e)` (a short read or invalid data is now `Truncated` or `Corrupt`) | `err.kind() == ErrorKind::Io`, and `err.source()` for the `io::Error` |
| `rig::Path`, `channel.path` | `rig::Property`, `channel.property` |
| `Axis::parse(s)` | `s.parse::<Axis>().ok()` |
| `MeshoptExtension::name()` | `as_str()` or `to_string()` |
| `mesh::to_glb(&m)` | `m.to_glb()` |
| `Prim::skinned()` | `Prim::is_skinned()` |
| `meshopt::apply_filter("OCTAHEDRAL", ..)` | `apply_filter(Filter::Octahedral, ..)` |
| `CompressOptions { level: 3, ..CompressOptions::KHR }` | `CompressOptions::KHR.level(3)` |

Outside the crate, a struct literal no longer compiles for `SourceInfo`, `Converted`,
`Extensions`, `Material` and `CompressOptions`. Start from `default()`, a constant or a
constructor such as `Material::plain`, then assign the fields. A `match` on `ErrorKind`,
`Property`, `Interpolation`, `MeshoptExtension`, `ExpMode`, `Filter` or `Location` needs a
`_ =>` arm.
