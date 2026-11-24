"""Picks the real files the resilience run damages, and prints shard k of n of them, one path a line.

    python3 tools/evidence/pick_resilience.py <corpora-dir> <corpora.txt> <k> <n> [--per-format 24] [--max-kb 1024]

Up to `--per-format` files of each format Polysmelt reads, of at most `--max-kb`, spread over the
corpora by a hash of their names (the same pick every run).
"""

import hashlib
import os
import sys

EXTS = {"stl", "ply", "vox", "obj", "fbx", "3ds", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb", "skp", "bvh",
        "dae", "zae", "kmz", "off", "vrm", "vrma", "x3d", "x3dv", "x3dz", "wrl", "wrz", "dxf", "splat", "spz"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
root, listing, k, n = args[0], args[1], int(args[2]), int(args[3])
per = int(opts.get("per-format", 24))
max_bytes = int(opts.get("max-kb", 1024)) * 1024
by_ext = {}
for line in open(listing, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    _, d = line.split("=", 1)
    for dirpath, dirs, files in os.walk(os.path.join(root, d)):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for f in files:
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            p = os.path.join(dirpath, f)
            if ext in EXTS and 0 < os.path.getsize(p) <= max_bytes:
                by_ext.setdefault(ext, []).append(p)
picked = []
for ext, ps in sorted(by_ext.items()):
    ps.sort(key=lambda p: hashlib.sha256(os.path.relpath(p, root).encode()).hexdigest())
    picked += ps[:per]
picked.sort(key=lambda p: hashlib.sha256(p.encode()).hexdigest())
for p in picked[k::n]:
    print(p)
