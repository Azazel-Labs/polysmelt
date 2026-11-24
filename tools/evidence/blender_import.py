"""Runs inside Blender: imports each file of a list with Blender's own importer and appends what it
found to a JSON-lines log, a `start` line before each file and a result line after, so the driver
(blender_sweep.py) can tell which file a crash or a hang was in.

    blender -b --factory-startup --python tools/evidence/blender_import.py -- <list.json> <log.jsonl> <first>
"""

import json
import os
import sys
import time

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
items = json.load(open(args[0], encoding="utf-8"))
log = open(args[1], "a", encoding="utf-8")
first = int(args[2])


def importer(ext):
    """The operator and its keyword arguments for an extension, or None where this Blender has none."""
    ops = {
        "gltf": ("import_scene", "gltf", {}),
        "glb": ("import_scene", "gltf", {}),
        "vrm": ("import_scene", "gltf", {}),
        "obj": ("wm", "obj_import", {}),
        "fbx": ("import_scene", "fbx", {}),
        "ply": ("wm", "ply_import", {}),
        "stl": ("wm", "stl_import", {}),
        "dae": ("wm", "collada_import", {}),
        "usd": ("wm", "usd_import", {}),
        "usda": ("wm", "usd_import", {}),
        "usdc": ("wm", "usd_import", {}),
        "usdz": ("wm", "usd_import", {}),
        "bvh": ("import_anim", "bvh", {}),
        "abc": ("wm", "alembic_import", {}),
        "svg": ("import_curve", "svg", {}),
    }
    if ext not in ops:
        return None
    mod, name, kw = ops[ext]
    group = getattr(bpy.ops, mod)
    if not hasattr(group, name):
        return None
    op = getattr(group, name)
    try:
        op.get_rna_type()
    except Exception:
        return None
    return op, kw


def stats():
    tris = verts = 0
    for me in bpy.data.meshes:
        verts += len(me.vertices)
        n = len(me.polygons)
        loops = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("loop_total", loops)
        tris += int(loops.sum()) - 2 * n
    bones = sum(len(a.bones) for a in bpy.data.armatures)
    return {
        "meshes": len(bpy.data.meshes),
        "vertices": verts,
        "triangles": tris,
        "materials": len(bpy.data.materials),
        "nodes": len(bpy.data.objects),
        "joints": bones,
        "animations": len(bpy.data.actions),
        "images": len(bpy.data.images),
    }


def write(rec):
    log.write(json.dumps(rec) + "\n")
    log.flush()


for i in range(first, len(items)):
    it = items[i]
    write({"start": i})
    rec = {"i": i, "corpus": it["corpus"], "file": it["file"], "ext": it["ext"], "bytes": it["bytes"], "loader": "blender"}
    imp = importer(it["ext"])
    if imp is None:
        rec["status"] = "unsupported"
        write(rec)
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    op, kw = imp
    t = time.perf_counter()
    try:
        result = op(filepath=os.path.abspath(it["path"]), **kw)
        rec["ms"] = (time.perf_counter() - t) * 1e3
        if "FINISHED" in result:
            s = stats()
            rec["stats"] = s
            rec["status"] = "ok" if s["nodes"] or s["meshes"] or s["animations"] else "empty"
        else:
            rec["status"] = "err"
            rec["error"] = f"operator returned {sorted(result)}"
    except Exception as e:  # an importer's failure on this file, not the sweep's
        rec["ms"] = (time.perf_counter() - t) * 1e3
        rec["status"] = "err"
        rec["error"] = str(e).replace("\n", " ")[:400]
    write(rec)
write({"end": True})
