#!/usr/bin/env bash
# Remakes hero.png (the README's) and social.png (the repository's and the demo page's link preview).
#
# The models are drawn by the demo's own renderer (demo/src/bin/media.rs, on the native GPU), read from the test
# corpus by polysmelt: the same monkey from STL, PLY and OBJ. hero.svg and social.svg lay the wordmark over those
# stages; resvg renders them (with the system's Segoe UI and Consolas, or what stands in for them), oxipng shrinks
# the result when it's on PATH.
set -euo pipefail

media="$(cd "$(dirname "$0")" && pwd)"
# A native path on Windows too, where the model specs are passed to the binary unconverted.
corpus="$(cd "$media/../tests/corpus" && (pwd -W 2> /dev/null || pwd))"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

(cd "$media/../demo" && cargo build --profile media --features media --bin media)
render="$media/../demo/target/media/media"

stl="$corpus/stl/blender_ascii_two_objects/scene.stl#steel:0"
ply="$corpus/ply/blender_ascii_smooth_ngons/scene.ply#copper:180"
obj="$corpus/obj/blender_suzanne_smooth/suzanne_smooth.obj#gold:-20"
"$render" "$work/stage-social.png" 1280 640 --zoom 1.33 --shift 0.62 --lift 0.02 --yaw 24 --pitch 9 "$ply" "$obj"
"$render" "$work/stage-hero.png" 1600 560 --zoom 1.44 --shift 0.72 --lift 0 --yaw 16 --pitch 8 "$stl" "$ply" "$obj"

for name in social hero; do
    cp "$media/$name.svg" "$work/"
    resvg "$work/$name.svg" "$media/$name.png"
    if command -v oxipng > /dev/null; then
        oxipng -q -o 4 --strip safe "$media/$name.png"
    fi
done
ls -l "$media"/*.png
