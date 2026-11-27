# glTF tools

Besides reading other formats into glTF, polysmelt has tools for glTF files themselves:
compressing and decompressing them with meshopt, and packing a `.gltf` and its files into one
GLB. They work on the GLB bytes directly and don't need the file decoded first.

## Compression (`gltf`)

- `gltf::compress` compresses a GLB losslessly with `EXT_meshopt_compression`, keeping any
  quantization it has.
- `gltf::compress_with` writes `KHR_meshopt_compression` instead: the version 1 attribute codec
  at a level you choose, and optionally the `COLOR` filter for 8- and 16-bit vertex colors
  (within 1 per channel).
- `gltf::decompress` turns a meshopt-compressed GLB back into a plain one. Quantization is kept;
  use `Document::decode` to turn quantized attributes into floats too.

## Packing (`gltf`)

- `gltf::pack_glb` turns a `.gltf` and the files it refers to into one GLB, and
  `gltf::pack_glb_decompressed` does the same while decompressing meshopt data.
- `gltf::files_in` is a fetch for those that stays inside the model's folder, so a `.gltf` can't
  read files outside it (it needs the `fs` feature).

## The meshopt bitstreams (`meshopt`)

The `meshopt` module is a Rust port of meshoptimizer's codecs for `EXT_meshopt_compression` and
`KHR_meshopt_compression`, byte for byte meshoptimizer's own output on its test vectors:

- the attribute encoder (codec versions 0 and 1, levels 0 to 3), the triangle and index-sequence
  encoders, and their size bounds;
- the filter encoders: octahedral, quaternion, exponential (all four exponent modes) and color;
- decoders for every mode and filter.

`cargo run --example meshopt -- sample out.glb` writes a small compressed test file.
