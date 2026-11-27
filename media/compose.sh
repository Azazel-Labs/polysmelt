#!/usr/bin/env bash
# Remakes hero.png (the README's) and social.png (the repository's and the demo page's link preview).
#
# The models are drawn by the demo's own renderer (demo/src/bin/media.rs, on the native GPU), read from the test
# corpus by polysmelt. hero.svg and social.svg lay the wordmark over those
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

# Four kinds of data from four formats: a MagicaVoxel character, a VTK surface, a PCL lidar scan drawn as points
# and a Blitz3D model, all read from the corpus (their licenses are beside them).
knight="$corpus/vox/ref_voxel_model_chr_knight/chr_knight.vox:-25"
cow="$corpus/vtk/real_vtk_cow/cow.vtp#copper:-35"
bunny="$corpus/pcd/pcl_bun0_ascii/bun0.pcd#ember:0"
wuson="$corpus/b3d/ref_assimp_wuson/WusonBlitz.b3d#gold:130"
"$render" "$work/stage-social.png" 1280 640 --zoom 1.12 --shift 1.02 --lift 0 --yaw 20 --pitch 10 "$cow" "$knight" "$bunny"
"$render" "$work/stage-hero.png" 1600 560 --zoom 1.85 --shift 0.70 --lift 0 --yaw 18 --pitch 9 "$knight" "$cow" "$bunny" "$wuson"

for name in social hero; do
    cp "$media/$name.svg" "$work/"
    resvg "$work/$name.svg" "$media/$name.png"
    if command -v oxipng > /dev/null; then
        oxipng -q -o 4 --strip safe "$media/$name.png"
    fi
done
ls -l "$media"/*.png
