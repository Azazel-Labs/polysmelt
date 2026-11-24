#!/usr/bin/env bash
# Builds the demo page into demo/dist: the wasm module and its bindings, the page, the samples and the social image.
#
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli at the version Cargo.lock has. wasm-opt (binaryen) is
# used when it's on PATH.
set -euo pipefail

demo="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$demo/.." && pwd)"
dist="$demo/dist"
cd "$demo"

cargo build --release --lib --target wasm32-unknown-unknown

rm -rf "$dist"
mkdir -p "$dist/pkg" "$dist/samples"
wasm-bindgen --target web --no-typescript --out-dir "$dist/pkg" target/wasm32-unknown-unknown/release/polysmelt_demo.wasm
if command -v wasm-opt > /dev/null; then
    wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
        "$dist/pkg/polysmelt_demo_bg.wasm" -o "$dist/pkg/polysmelt_demo_bg.wasm"
else
    echo "wasm-opt isn't on PATH: the module is left as wasm-bindgen wrote it"
fi

cp web/* "$dist/"
cp "$root/media/social.png" "$dist/social.png"

# The samples app.js offers, from the test corpus, with the files of each listed in samples/index.json.
samples=(
    gltf/skin_animated
    fbx/blender_armature_animated
    fbx/blender_shape_key_animated
    md2/morph_frame_groups
    bvh/humanoid
    dae/skin_twenty_influences
    obj/blender_suzanne_smooth
    pcd/pcl_car6_compressed
)
entries=""
for dir in "${samples[@]}"; do
    mkdir -p "$dist/samples/$dir"
    names=""
    for file in "$root/tests/corpus/$dir"/*; do
        name="$(basename "$file")"
        [ "$name" = expect.json ] && continue
        cp "$file" "$dist/samples/$dir/"
        names="$names${names:+,}\"$name\""
    done
    entries="$entries${entries:+,}\"$dir\":[$names]"
done
echo "{${entries}}" > "$dist/samples/index.json"

echo "built $dist: $(du -h "$dist/pkg/polysmelt_demo_bg.wasm" | cut -f1) of wasm"
