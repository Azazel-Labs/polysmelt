"""Real models in every format Blender writes, for the `files` benchmark and the resilience run.

Imports a glTF model (a Khronos sample, say) and exports it with Blender's own exporters as OBJ (with
its MTL), binary FBX, binary and ASCII PLY, binary and ASCII STL, COLLADA and USD: real exporter output of real
geometry, in formats where only generated tori were benchmarked before.

    blender -b --factory-startup --python-exit-code 1 --python bench/export_real.py -- <model.gltf|glb> <out-dir> <name>
"""

import os
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
src, out, name = args[0], args[1], args[2]
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(src))


def path(ext, suffix=""):
    return os.path.join(out, f"{name}{suffix}.{ext}")


bpy.ops.wm.obj_export(filepath=path("obj"), export_materials=True, path_mode="COPY")
bpy.ops.export_scene.fbx(filepath=path("fbx"), path_mode="COPY", embed_textures=True, bake_anim=True)
bpy.ops.wm.ply_export(filepath=path("ply", "_binary"), ascii_format=False)
bpy.ops.wm.ply_export(filepath=path("ply", "_ascii"), ascii_format=True)
bpy.ops.wm.stl_export(filepath=path("stl", "_binary"), ascii_format=False)
bpy.ops.wm.stl_export(filepath=path("stl", "_ascii"), ascii_format=True)
# COLLADA (gone from Blender 5) and USD, where this Blender has them.
for op, kw, ext in (
    ("collada_export", {}, "dae"),
    ("usd_export", {}, "usdc"),
):
    if hasattr(bpy.ops.wm, op):
        try:
            getattr(bpy.ops.wm, op)(filepath=path(ext), **kw)
        except Exception as e:  # an exporter that fails shouldn't lose the others
            print("export failed:", ext, e)
print("exported", name)
