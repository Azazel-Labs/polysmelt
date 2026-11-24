#!/usr/bin/env bash
# Fetches the evidence corpora, each pinned to a commit or a checksum, into one folder per corpus:
#
#   tools/evidence/fetch_corpora.sh <dir>
#
# Nothing here is committed to this repository: every model keeps its own license, fetched from
# where its authors publish it. Git sources are sparse, shallow, blob-filtered checkouts of a pinned
# commit; archives are checked against a SHA-256 before they're opened.
set -euo pipefail
out=$(realpath -m "$1")
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# git_sparse <corpus> <repo> <commit> <pattern>...  (non-cone patterns)
git_sparse() {
  local name=$1 repo=$2 rev=$3
  shift 3
  local dest="$out/$name"
  if [ -f "$dest/.fetched-$rev" ]; then echo "$name: cached"; return; fi
  rm -rf "$dest" "$tmp/$name"
  git init -q "$tmp/$name"
  git -C "$tmp/$name" remote add origin "https://github.com/$repo.git"
  git -C "$tmp/$name" config core.sparseCheckout true
  git -C "$tmp/$name" sparse-checkout set --no-cone "$@"
  git -C "$tmp/$name" fetch -q --depth 1 --filter=blob:none origin "$rev"
  git -C "$tmp/$name" -c advice.detachedHead=false checkout -q FETCH_HEAD
  rm -rf "$tmp/$name/.git"
  mv "$tmp/$name" "$dest"
  touch "$dest/.fetched-$rev"
  echo "$name: $(find "$dest" -type f | wc -l) files, $(du -sh "$dest" | cut -f1)"
}

# fetch <url> <sha256> <file>
fetch() {
  curl -fsSL --retry 3 --retry-delay 5 -o "$3" "$1"
  echo "$2  $3" | sha256sum -c --quiet -
}

# glTF 2.0: the Khronos sample models, every variant (embedded, binary, Draco, meshopt, KTX).
git_sparse khronos KhronosGroup/glTF-Sample-Assets e124e8d8d7330a5372bc2b4d00d39a4a9d7997ea \
  '/Models/' '!/Models/*/screenshot/'
# The COLLADA sources the older Khronos samples were converted from.
git_sparse khronos-dae KhronosGroup/glTF-Sample-Models d7a3cc8e51d7c573771ae77a57f16b0662a905c6 \
  '/sourceModels/'
# Assimp's own test models (v6.0.6): every format it reads, including its regression cases.
git_sparse assimp assimp/assimp a79ff6bbce54d3e01186a918dc6374ed55317426 \
  '/test/models/' '/test/models-nonbsd/'
# three.js r187's example models.
git_sparse threejs mrdoob/three.js 7e89746a687bb88271253698fa108b1dd59825d0 \
  '/examples/models/'
# The 3MF Consortium's samples.
git_sparse 3mf 3MFConsortium/3mf-samples 665e20dc4d7777fd4c9702bca86a2d4028440337 '/*'
# MagicaVoxel's own sample models.
git_sparse vox ephtracy/voxel-model 3bff8feaa7d94a6237f2c07dc4ca3c3d268ef06c '/*'

# The Stanford scans (PLY), the reconstructions only.
if [ ! -f "$out/stanford/.fetched-1" ]; then
  rm -rf "$out/stanford"
  mkdir -p "$out/stanford" "$tmp/st"
  base=http://graphics.stanford.edu/pub/3Dscanrep
  fetch "$base/bunny.tar.gz" a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84 "$tmp/st/bunny.tgz"
  fetch "$base/dragon/dragon_recon.tar.gz" 74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f "$tmp/st/dragon.tgz"
  fetch "$base/happy/happy_recon.tar.gz" 409cd294efbfd8244e15a382b95a9423f153b7776e736c9b09f19ec9d3c10ed0 "$tmp/st/happy.tgz"
  for a in bunny dragon happy; do
    mkdir -p "$tmp/st/$a"
    tar -xzf "$tmp/st/$a.tgz" -C "$tmp/st/$a" --no-same-owner
  done
  find "$tmp/st" -name '*.ply' -path '*recon*' -exec cp {} "$out/stanford/" \;
  touch "$out/stanford/.fetched-1"
  echo "stanford: $(ls "$out/stanford" | wc -l) files, $(du -sh "$out/stanford" | cut -f1)"
fi

# Morgan McGuire's Computer Graphics Archive: big real-world OBJ scenes.
if [ ! -f "$out/mcguire/.fetched-1" ]; then
  rm -rf "$out/mcguire"
  mkdir -p "$out/mcguire" "$tmp/mc"
  base=https://casual-effects.com/g3d/data10
  while read -r name path sum; do
    fetch "$base/$path" "$sum" "$tmp/mc/$name.zip"
    mkdir -p "$out/mcguire/$name"
    unzip -q -o "$tmp/mc/$name.zip" -d "$out/mcguire/$name"
    rm "$tmp/mc/$name.zip"
  done <<'LIST'
sponza common/model/crytek_sponza/sponza.zip da005cbee0be2df2abc8513f3ceb61bcb6f69aac112babcd9c00169a27c2770c
rungholt research/model/rungholt/rungholt.zip dd927901828e19e042e02448f3814e18592c9c076b39bff226bb8dc7a5ad6f4a
bmw research/model/bmw/bmw.zip 644d48ad8381c4e1e96be34a9900574b4df08d348e35a3394870e23c3baeffa6
conference research/model/conference/conference.zip 3dfd86325416ec344bfb2d19290b2e3305cc862ee504a06a52ce0a0e895453f0
LIST
  touch "$out/mcguire/.fetched-1"
  echo "mcguire: $(find "$out/mcguire" -type f | wc -l) files, $(du -sh "$out/mcguire" | cut -f1)"
fi
du -sh "$out"/* | sort -h
