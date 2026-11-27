"""Conformance cases exported by Blender: real exporter output, not hand-written files.

Run from the repository root with Blender (4.2 or later):

    blender -b --factory-startup --python-exit-code 1 --python tools/corpus/blender_export.py -- obj fbx

Each scene is built from scratch, exported, and its expectations (triangle count, surface area,
bounds, materials) are taken from Blender's own data, so a case checks the reader against what
the exporter meant to write. Cases are named `blender_<scene>` and replace earlier ones; the
other generators keep them.
"""

import math
import os
import shutil
import sys

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AXES, ROOT, case, png, r  # noqa: E402

VERSION = bpy.app.version_string


# --- scene building -------------------------------------------------------------------------

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, emission=None, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    if image:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(image)
        m.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def add(mesh_op, **kw):
    mesh_op(**kw)
    return bpy.context.active_object


def from_polygon(name, pts):
    """An object holding one polygon."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in pts]
    bm.faces.new(verts)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


# --- expectations from Blender's data ----------------------------------------------------------

def y_up(v):
    """Blender's Z-up to the Y-up the exporters write (forward -Z, up Y): (x, y, z) -> (x, z, -y)."""
    return (v[0], v[2], -v[1])


def stats(axes=None):
    """Triangles, surface area and bounds (Y-up unless `axes` says otherwise) of every mesh in the scene, modifiers applied."""
    deps = bpy.context.evaluated_depsgraph_get()
    tris, area, tri_area = 0, 0.0, 0.0
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(deps)
        me = ev.to_mesh()
        me.transform(ob.matrix_world)
        for p in me.polygons:
            tris += len(p.vertices) - 2
            area += p.area
        me.calc_loop_triangles()
        tri_area += sum(t.area for t in me.loop_triangles)
        for v in me.vertices:
            for k, c in enumerate((axes or y_up)(v.co)):
                lo[k], hi[k] = min(lo[k], c), max(hi[k], c)
        ev.to_mesh_clear()
    out = {"triangles": tris, "bounds": {"min": [r(c, 4) for c in lo], "max": [r(c, 4) for c in hi]}}
    # The area is only well defined when every polygon is planar: a bent quad's area depends on
    # the diagonal chosen to split it.
    if abs(area - tri_area) <= 1e-5 * max(area, 1.0):
        out["area"] = r(area, 4)
    return out


def animated(frames=24):
    """An animation running from frame 0 (time 0) to `frames`, at 24 fps."""
    sc = bpy.context.scene
    sc.render.fps, sc.render.fps_base = 24, 1.0
    sc.frame_start, sc.frame_end = 0, frames
    sc.frame_set(0)


def poses(frames, axes=None):
    """The bounds at each frame, Blender's own evaluation (modifiers, armatures and shape keys applied), as the
    conformance test's `poses`; the scene is left at frame 0."""
    sc = bpy.context.scene
    out = []
    for f in frames:
        sc.frame_set(f)
        out.append({"animation": 0, "time": r(f / sc.render.fps, 6), "bounds": stats(axes)["bounds"]})
    sc.frame_set(0)
    return out


def mat_expect(m, **extra):
    bsdf = m.node_tree.nodes["Principled BSDF"]
    c = bsdf.inputs["Base Color"].default_value
    out = {"name": m.name, "color": [r(c[0], 4), r(c[1], 4), r(c[2], 4), 1],
           "roughness": r(bsdf.inputs["Roughness"].default_value, 4)}
    out.update(extra)
    return out


def armature_scene():
    """A column skinned to two bones, the upper one bending, the lower one turning."""
    reset()
    animated()
    body = add(bpy.ops.mesh.primitive_cylinder_add, vertices=8, radius=0.25, depth=2, location=(0, 0, 1))
    bm = bmesh.new()
    bm.from_mesh(body.data)
    tall = [e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 1e-6]
    bmesh.ops.subdivide_edges(bm, edges=tall, cuts=5, use_grid_fill=True)
    bm.to_mesh(body.data)
    bm.free()
    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    lower = arm_data.edit_bones.new("Lower")
    lower.head, lower.tail = (0, 0, 0), (0, 0, 1)
    upper = arm_data.edit_bones.new("Upper")
    upper.head, upper.tail = (0, 0, 1), (0, 0, 2)
    upper.parent, upper.use_connect = lower, True
    bpy.ops.object.mode_set(mode="OBJECT")
    groups = {n: body.vertex_groups.new(name=n) for n in ("Lower", "Upper")}
    for v in body.data.vertices:
        t = min(max(v.co.z + 1 - 0.5, 0.0), 1.0)  # world height 0.5 to 1.5 blends lower into upper
        if t < 1:
            groups["Lower"].add([v.index], 1 - t, "REPLACE")
        if t > 0:
            groups["Upper"].add([v.index], t, "REPLACE")
    body.parent = arm
    body.modifiers.new("Armature", "ARMATURE").object = arm
    for name, rot in (("Upper", (math.radians(90), 0, 0)), ("Lower", (0, math.radians(30), 0))):
        pb = arm.pose.bones[name]
        pb.rotation_mode = "XYZ"
        pb.keyframe_insert("rotation_euler", frame=0)
        pb.rotation_euler = rot
        pb.keyframe_insert("rotation_euler", frame=24)
    bpy.context.scene.frame_set(0)


def shape_key_scene():
    """A cube whose shape key's value is animated from 0 to 1."""
    reset()
    animated()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    ob.shape_key_add(name="Basis")
    key = ob.shape_key_add(name="Tall")
    for v in key.data:
        if v.co.z > 0:
            v.co.z += 2
    key.value = 0.0
    key.keyframe_insert("value", frame=0)
    key.value = 1.0
    key.keyframe_insert("value", frame=24)
    bpy.context.scene.frame_set(0)


def object_scene():
    """An object moving, turning half a turn and growing, under a still parent."""
    reset()
    animated()
    parent = add(bpy.ops.object.empty_add, location=(0, 0, 1), rotation=(0, 0, 0.5))
    ob = add(bpy.ops.mesh.primitive_monkey_add)
    ob.parent = parent
    ob.keyframe_insert("location", frame=0)
    ob.keyframe_insert("rotation_euler", frame=0)
    ob.keyframe_insert("scale", frame=0)
    ob.location, ob.rotation_euler, ob.scale = (3, 0, 1), (0, 0, math.pi), (1, 2, 1)
    ob.keyframe_insert("location", frame=24)
    ob.keyframe_insert("rotation_euler", frame=24)
    ob.keyframe_insert("scale", frame=24)
    bpy.context.scene.frame_set(0)


def cameras_lights_scene():
    """A cube with a camera and point, sun and spot lights."""
    reset()
    add(bpy.ops.mesh.primitive_cube_add)
    add(bpy.ops.object.camera_add, location=(0, -8, 2), rotation=(math.radians(80), 0, 0))
    for kind, loc in (("POINT", (2, 0, 3)), ("SUN", (0, 0, 5)), ("SPOT", (-2, 0, 3))):
        add(bpy.ops.object.light_add, type=kind, location=loc)


# --- writing cases ----------------------------------------------------------------------------

def clear_blender_cases(fmt):
    d = os.path.join(ROOT, fmt)
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(d):
        if name.startswith("blender_"):
            shutil.rmtree(os.path.join(d, name))


def case_dir(fmt, name):
    d = os.path.join(ROOT, fmt, "blender_" + name)
    os.makedirs(d, exist_ok=True)
    return d


def write_case(fmt, name, spec, op, expect):
    expect = {"spec": spec, "source": f"exported by Blender {VERSION} ({op}), tools/corpus/blender_export.py",
              **expect}
    case(fmt, "blender_" + name, {}, expect)


# --- OBJ --------------------------------------------------------------------------------------

OBJ_SPEC = "Wavefront OBJ + MTL as Blender's exporter writes them"
OBJ_OP = "bpy.ops.wm.obj_export"


def obj_export(name, **kw):
    d = case_dir("obj", name)
    args = dict(filepath=os.path.join(d, name + ".obj"), export_selected_objects=False,
                export_materials=True, path_mode="STRIP")
    args.update(kw)
    bpy.ops.wm.obj_export(**args)
    return d


def obj_cases():
    clear_blender_cases("obj")

    # The default cube with one material: quads, UVs, flat normals, Kd and Ns (roughness).
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Painted", color=(0.8, 0.1, 0.05), roughness=0.4)
    ob.data.materials.append(m)
    obj_export("cube")
    write_case("obj", "cube", OBJ_SPEC + ": quads, UVs, flat normals, one material", OBJ_OP,
               {**stats(), "uvs": True, "normals_in": AXES, "winding_matches_normals": True,
                "material_count": 1, "materials": [mat_expect(m)]})

    # Suzanne, smooth shaded: shared smooth normals, triangles and quads mixed.
    reset()
    ob = add(bpy.ops.mesh.primitive_monkey_add)
    bpy.ops.object.shade_smooth()
    obj_export("suzanne_smooth", export_materials=False)
    write_case("obj", "suzanne_smooth", OBJ_SPEC + ": smooth normals, mixed triangles and quads, no MTL",
               OBJ_OP, {**stats(), "uvs": True})

    # A cylinder with n-gon caps.
    reset()
    add(bpy.ops.mesh.primitive_cylinder_add, vertices=32, end_fill_type="NGON")
    obj_export("cylinder_ngon_caps", export_materials=False)
    write_case("obj", "cylinder_ngon_caps", OBJ_SPEC + ": 32-sided n-gon caps", OBJ_OP,
               {**stats(), "winding_matches_normals": True})

    # A concave star: triangulating it as a fan would cover area outside the outline.
    reset()
    star = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rad = 1.0 if i % 2 == 0 else 0.4
        star.append((rad * math.cos(a), rad * math.sin(a), 0.0))
    from_polygon("Star", star)
    obj_export("concave_star", export_materials=False, export_uv=False)
    write_case("obj", "concave_star", OBJ_SPEC + ": one concave 10-sided polygon", OBJ_OP,
               {**stats(), "normals_in": [[0, 1, 0]], "winding_matches_normals": True})

    # Two materials on one mesh: one primitive per material.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    gray = material("Gray", color=(0.5, 0.5, 0.5), roughness=0.9)
    red = material("Red", color=(0.9, 0.0, 0.0), roughness=0.2)
    ob.data.materials.append(gray)
    ob.data.materials.append(red)
    top = max(ob.data.polygons, key=lambda p: p.center.z)
    top.material_index = 1
    obj_export("two_materials")
    write_case("obj", "two_materials", OBJ_SPEC + ": usemtl switching within one object", OBJ_OP,
               {**stats(), "primitives": 2, "material_count": 2, "materials": [mat_expect(gray), mat_expect(red)]})

    # Image texture: map_Kd naming a PNG beside the model.
    reset()
    d = case_dir("obj", "textured")
    tex = os.path.join(d, "checker_red.png")
    with open(tex, "wb") as f:
        f.write(png((255, 0, 0), 4))
    ob = add(bpy.ops.mesh.primitive_plane_add)
    m = material("Textured", image=tex)
    ob.data.materials.append(m)
    obj_export("textured")
    write_case("obj", "textured", OBJ_SPEC + ": map_Kd image texture", OBJ_OP,
               {**stats(), "uvs": True, "materials": [{"name": "Textured", "textured": True, "pixel": [255, 0, 0]}]})

    # Vertex colors (`v x y z r g b`), which Blender writes as sRGB.
    reset()
    ob = add(bpy.ops.mesh.primitive_plane_add)
    attr = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.214, 0.214, 0.214)]
    for i, c in enumerate(linear):
        attr.data[i].color = (*c, 1.0)
    obj_export("vertex_colors", export_colors=True, export_materials=False)
    write_case("obj", "vertex_colors", OBJ_SPEC + ": vertex colors after positions", OBJ_OP,
               {**stats(), "colors": True, "tolerance": 2e-3,
                "colors_include": [[*c, 1] for c in linear]})

    # The PBR extension keys: Pr, Pm, Ke.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Metal", color=(0.9, 0.6, 0.2), roughness=0.25, metallic=1.0, emission=(1.0, 0.5, 0.0))
    ob.data.materials.append(m)
    obj_export("pbr_extensions", export_pbr_extensions=True)
    write_case("obj", "pbr_extensions", OBJ_SPEC + ": PBR extension keys (Pr, Pm, Ke)", OBJ_OP,
               {**stats(), "materials": [mat_expect(m, metallic=1, emissive=[1, 0.5, 0])]})

    # Every grouping option, triangulated: o/g lines, smooth groups, two transformed objects.
    reset()
    a = add(bpy.ops.mesh.primitive_cube_add, location=(3, 0, 0), rotation=(0.3, 0.2, 0.1))
    b = add(bpy.ops.mesh.primitive_uv_sphere_add, location=(-2, 1, 0.5), scale=(1, 1, 2))
    bpy.ops.object.shade_smooth()
    m = material("Shared", color=(0.2, 0.4, 0.8), roughness=0.6)
    a.data.materials.append(m)
    b.data.materials.append(m)
    obj_export("groups_triangulated", export_triangulated_mesh=True, export_smooth_groups=True,
               export_object_groups=True, export_material_groups=True)
    write_case("obj", "groups_triangulated", OBJ_SPEC + ": o/g/s statements, triangulated, two transformed objects",
               OBJ_OP, {**stats(), "material_count": 1, "materials": [mat_expect(m)],
                        "document": {"nodes": 3, "node_names": ["Cube_Shared", "Sphere_Shared", "RootNode"],
                                     "source": {"format": "OBJ", "linearColors": True}}})

    print("blender obj cases written")


# --- STL and PLY ------------------------------------------------------------------------------

def z_up(v):
    """Blender's own axes, which its STL and PLY exporters keep by default."""
    return tuple(v)


def stl_cases():
    clear_blender_cases("stl")
    op = "bpy.ops.wm.stl_export"
    spec = "STL as Blender's exporter writes it"
    for ascii_format in (False, True):
        kind = "ascii" if ascii_format else "binary"
        reset()
        add(bpy.ops.mesh.primitive_monkey_add)
        add(bpy.ops.mesh.primitive_cube_add, location=(3, 0, 0), rotation=(0.4, 0, 0.7))
        d = case_dir("stl", f"{kind}_two_objects")
        bpy.ops.wm.stl_export(filepath=os.path.join(d, "scene.stl"), ascii_format=ascii_format)
        write_case("stl", f"{kind}_two_objects", f"{spec}: {kind}, two objects in one solid", op,
                   {**stats(z_up), "winding_matches_normals": True})
    print("blender stl cases written")


def ply_cases():
    clear_blender_cases("ply")
    op = "bpy.ops.wm.ply_export"
    spec = "PLY as Blender's exporter writes it"
    for ascii_format in (False, True):
        kind = "ascii" if ascii_format else "binary"
        # UVs, normals and sRGB vertex colors.
        reset()
        ob = add(bpy.ops.mesh.primitive_plane_add)
        attr = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.214, 0.214, 0.214)]
        for i, c in enumerate(linear):
            attr.data[i].color = (*c, 1.0)
        d = case_dir("ply", f"{kind}_colors_uvs")
        bpy.ops.wm.ply_export(filepath=os.path.join(d, "plane.ply"), ascii_format=ascii_format,
                              export_uv=True, export_normals=True, export_colors="SRGB")
        # Blender truncates when it quantizes (1.0 is written as 254), so allow one 8-bit step.
        write_case("ply", f"{kind}_colors_uvs", f"{spec}: {kind}, normals, UVs, sRGB vertex colors", op,
                   {**stats(z_up), "uvs": True, "colors": True, "normals_in": [[0, 0, 1]], "tolerance": 1e-2,
                    "colors_include": [[*c, 1] for c in linear]})
        # Smooth Suzanne with n-gons kept.
        reset()
        add(bpy.ops.mesh.primitive_monkey_add)
        bpy.ops.object.shade_smooth()
        add(bpy.ops.mesh.primitive_cylinder_add, vertices=12, end_fill_type="NGON", location=(3, 0, 0))
        d = case_dir("ply", f"{kind}_smooth_ngons")
        bpy.ops.wm.ply_export(filepath=os.path.join(d, "scene.ply"), ascii_format=ascii_format,
                              export_uv=False, export_normals=True, export_colors="NONE")
        write_case("ply", f"{kind}_smooth_ngons", f"{spec}: {kind}, smooth normals, n-gon caps", op,
                   {k: v for k, v in stats(z_up).items() if k != "area"})
    print("blender ply cases written")


# --- FBX --------------------------------------------------------------------------------------

FBX_SPEC = "binary FBX 7.4 as Blender's exporter writes it"
FBX_OP = "bpy.ops.export_scene.fbx"


def fbx_export(name, **kw):
    d = case_dir("fbx", name)
    args = dict(filepath=os.path.join(d, name + ".fbx"), use_selection=False, path_mode="COPY",
                embed_textures=True, axis_forward="-Z", axis_up="Y")
    args.update(kw)
    bpy.ops.export_scene.fbx(**args)
    return d


def fbx_mat(m, **extra):
    """Blender writes its base color linear, its roughness as Shininess ((1 - r) * 10)^2 and metallic as
    ReflectionFactor."""
    bsdf = m.node_tree.nodes["Principled BSDF"]
    return mat_expect(m, metallic=r(bsdf.inputs["Metallic"].default_value, 4), **extra)


def fbx_cases():
    clear_blender_cases("fbx")

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Painted", color=(0.8, 0.1, 0.05), roughness=0.4, metallic=0.3)
    ob.data.materials.append(m)
    fbx_export("cube")
    write_case("fbx", "cube", FBX_SPEC + ": quads, UVs, flat normals, one material", FBX_OP,
               {**stats(), "uvs": True, "normals_in": AXES, "winding_matches_normals": True,
                "faces_outward": True, "material_count": 1, "materials": [fbx_mat(m)]})

    reset()
    add(bpy.ops.mesh.primitive_monkey_add)
    bpy.ops.object.shade_smooth()
    fbx_export("suzanne_smooth")
    write_case("fbx", "suzanne_smooth", FBX_SPEC + ": smooth normals, mixed triangles and quads", FBX_OP,
               {**stats(), "uvs": True})

    reset()
    star = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rad = 1.0 if i % 2 == 0 else 0.4
        star.append((rad * math.cos(a), rad * math.sin(a), 0.0))
    from_polygon("Star", star)
    fbx_export("concave_star")
    write_case("fbx", "concave_star", FBX_SPEC + ": one concave 10-sided polygon", FBX_OP,
               {**stats(), "normals_in": [[0, 1, 0]], "winding_matches_normals": True})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    gray = material("Gray", color=(0.5, 0.5, 0.5), roughness=0.9)
    red = material("Red", color=(0.9, 0.0, 0.0), roughness=0.2)
    ob.data.materials.append(gray)
    ob.data.materials.append(red)
    max(ob.data.polygons, key=lambda p: p.center.z).material_index = 1
    fbx_export("two_materials")
    write_case("fbx", "two_materials", FBX_SPEC + ": a material layer choosing per polygon", FBX_OP,
               {**stats(), "primitives": 2, "material_count": 2, "materials": [fbx_mat(gray), fbx_mat(red)]})

    reset()
    d = case_dir("fbx", "textured_embedded")
    tex = os.path.join(d, "..", "red_source.png")
    with open(tex, "wb") as f:
        f.write(png((255, 0, 0), 4))
    ob = add(bpy.ops.mesh.primitive_plane_add)
    m = material("Textured", image=tex)
    ob.data.materials.append(m)
    fbx_export("textured_embedded")
    os.remove(tex)
    write_case("fbx", "textured_embedded", FBX_SPEC + ": an image texture embedded in the file", FBX_OP,
               {**stats(), "uvs": True, "image_count": 1,
                "materials": [{"name": "Textured", "textured": True, "pixel": [255, 0, 0]}]})

    reset()
    d = case_dir("fbx", "textured_external")
    tex = os.path.join(d, "green.png")
    with open(tex, "wb") as f:
        f.write(png((0, 255, 0), 4))
    ob = add(bpy.ops.mesh.primitive_plane_add)
    m = material("Leaf", image=tex)
    ob.data.materials.append(m)
    fbx_export("textured_external", path_mode="RELATIVE", embed_textures=False)
    write_case("fbx", "textured_external", FBX_SPEC + ": an image texture beside the file", FBX_OP,
               {**stats(), "uvs": True, "materials": [{"name": "Leaf", "textured": True, "pixel": [0, 255, 0]}]})

    reset()
    ob = add(bpy.ops.mesh.primitive_plane_add)
    attr = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.214, 0.214, 0.214)]
    for i, c in enumerate(linear):
        attr.data[i].color = (*c, 1.0)
    fbx_export("vertex_colors", colors_type="SRGB")
    write_case("fbx", "vertex_colors", FBX_SPEC + ": a color layer, written as sRGB", FBX_OP,
               {**stats(), "colors": True, "tolerance": 5e-3, "colors_include": [[*c, 1] for c in linear]})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Lamp", color=(1, 1, 1), emission=(1.0, 0.5, 0.0))
    m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.5
    ob.data.materials.append(m)
    fbx_export("emission")
    write_case("fbx", "emission", FBX_SPEC + ": emission color and strength", FBX_OP,
               {**stats(), "materials": [{"name": "Lamp", "emissive": [0.5, 0.25, 0]}]})

    # A hierarchy of transformed objects, one of them mirrored.
    reset()
    parent = add(bpy.ops.mesh.primitive_cube_add, location=(2, 0, 1), rotation=(0.3, 0.5, 0.2), scale=(1, 2, 0.5))
    child = add(bpy.ops.mesh.primitive_cone_add, location=(0, 3, 0), rotation=(0, 0, 0.7))
    child.parent = parent
    mirrored = add(bpy.ops.mesh.primitive_monkey_add, location=(-3, 0, 0), scale=(-1, 1, 1))
    fbx_export("hierarchy")
    write_case("fbx", "hierarchy", FBX_SPEC + ": parented, rotated, non-uniformly scaled and mirrored objects", FBX_OP,
               {**stats(), "winding_matches_normals": True})

    # Other axis and unit choices: the reader's result is the same Y-up metres.
    reset()
    add(bpy.ops.mesh.primitive_cube_add, location=(1, 2, 3), rotation=(0.2, 0.4, 0.6), scale=(1, 2, 3))
    for name, kw, what in [
        ("z_up", dict(axis_forward="Y", axis_up="Z"), "Z up, Y forward"),
        ("x_forward", dict(axis_forward="X", axis_up="Y"), "Y up, X forward"),
        ("scale_units", dict(apply_scale_options="FBX_SCALE_UNITS"), "units applied to the scale"),
        ("scale_all", dict(apply_scale_options="FBX_SCALE_ALL"), "every scaling applied"),
        ("bake_space_transform", dict(axis_forward="Y", axis_up="Z", bake_space_transform=True),
         "Z up with the axis change baked into the geometry"),
    ]:
        fbx_export("axes_" + name, **kw)
        write_case("fbx", "axes_" + name, FBX_SPEC + ": " + what, FBX_OP,
                   {**stats(), "winding_matches_normals": True, "faces_outward": True})

    # One take: the scene's animation, every frame keyed.
    take = dict(bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0)

    armature_scene()
    fbx_export("armature_animated", **take)
    write_case("fbx", "armature_animated", FBX_SPEC + ": a column skinned to two bones, both animated", FBX_OP,
               {**stats(), "tolerance": 1e-3,
                "document": {"skins": 1, "animations": 1, "node_names": ["Armature", "Lower", "Upper"]},
                "poses": poses([0, 6, 12, 24])})

    shape_key_scene()
    fbx_export("shape_key_animated", **take)
    write_case("fbx", "shape_key_animated", FBX_SPEC + ": a shape key whose value is animated", FBX_OP,
               {**stats(), "tolerance": 1e-3, "document": {"morph_targets": 1, "animations": 1},
                "poses": poses([0, 8, 16, 24])})

    object_scene()
    fbx_export("object_animated", **take)
    write_case("fbx", "object_animated", FBX_SPEC + ": an object's location, rotation and scale animated", FBX_OP,
               {**stats(), "tolerance": 1e-3, "document": {"animations": 1},
                "poses": poses([0, 5, 12, 19, 24])})

    cameras_lights_scene()
    fbx_export("cameras_lights")
    write_case("fbx", "cameras_lights", FBX_SPEC + ": a camera and point, sun and spot lights", FBX_OP,
               {**stats(), "document": {"cameras": 1, "lights": 3}})

    print("blender fbx cases written")


# --- USD --------------------------------------------------------------------------------------

USD_SPEC = "OpenUSD as Blender's exporter writes it"
USD_OP = "bpy.ops.wm.usd_export"


def usd_export(name, ext="usdc", **kw):
    d = case_dir("usd", name)
    args = dict(filepath=os.path.join(d, name + "." + ext), export_lights=False, export_cameras=False)
    args.update(kw)
    bpy.ops.wm.usd_export(**args)
    return d


def usd_mat(m, **extra):
    """Blender's UsdPreviewSurface carries the base color (linear), roughness and metallic as they are."""
    bsdf = m.node_tree.nodes["Principled BSDF"]
    return mat_expect(m, metallic=r(bsdf.inputs["Metallic"].default_value, 4), **extra)


def usd_cases():
    clear_blender_cases("usd")

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Painted", color=(0.8, 0.1, 0.05), roughness=0.4, metallic=0.3)
    ob.data.materials.append(m)
    for ext in ["usdc", "usda"]:
        usd_export("cube_" + ext, ext)
        write_case("usd", "cube_" + ext, USD_SPEC + f": .{ext}, Z up, quads, UVs, normals, one material", USD_OP,
                   {**stats(), "uvs": True, "normals_in": AXES, "winding_matches_normals": True,
                    "faces_outward": True, "material_count": 1, "materials": [usd_mat(m)]})

    reset()
    add(bpy.ops.mesh.primitive_monkey_add)
    bpy.ops.object.shade_smooth()
    usd_export("suzanne_smooth")
    write_case("usd", "suzanne_smooth", USD_SPEC + ": smooth normals, mixed triangles and quads", USD_OP,
               {**stats(), "uvs": True})

    # A subdivision modifier is written as the base cage with subdivisionScheme catmullClark.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 1
    ob.modifiers["Subdivision"].show_viewport = False
    ob.modifiers["Subdivision"].show_render = False
    cage = stats()
    ob.modifiers["Subdivision"].show_render = True
    usd_export("subdivision_cage", export_normals=False)
    write_case("usd", "subdivision_cage", USD_SPEC + ": a subdivision modifier written as a catmullClark cage",
               USD_OP, {**cage, "faces_outward": True})
    ob.modifiers["Subdivision"].show_viewport = True
    usd_export("subdivision_tessellated", export_subdivision="TESSELLATE")
    write_case("usd", "subdivision_tessellated", USD_SPEC + ": a subdivision modifier applied before writing",
               USD_OP, {**stats(), "faces_outward": True})

    reset()
    star = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rad = 1.0 if i % 2 == 0 else 0.4
        star.append((rad * math.cos(a), rad * math.sin(a), 0.0))
    from_polygon("Star", star)
    usd_export("concave_star")
    write_case("usd", "concave_star", USD_SPEC + ": one concave 10-sided polygon", USD_OP,
               {**stats(), "normals_in": [[0, 1, 0]], "winding_matches_normals": True})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    gray = material("Gray", color=(0.5, 0.5, 0.5), roughness=0.9)
    red = material("Red", color=(0.9, 0.0, 0.0), roughness=0.2)
    ob.data.materials.append(gray)
    ob.data.materials.append(red)
    max(ob.data.polygons, key=lambda p: p.center.z).material_index = 1
    usd_export("two_materials")
    write_case("usd", "two_materials", USD_SPEC + ": GeomSubsets choosing materials per face", USD_OP,
               {**stats(), "primitives": 2, "material_count": 2, "materials": [usd_mat(gray), usd_mat(red)]})

    reset()
    d = case_dir("usd", "textured_usdz")
    tex = os.path.join(d, "..", "red_source.png")
    with open(tex, "wb") as f:
        f.write(png((255, 0, 0), 4))
    ob = add(bpy.ops.mesh.primitive_plane_add)
    m = material("Textured", image=tex)
    ob.data.materials.append(m)
    usd_export("textured_usdz", "usdz")
    os.remove(tex)
    write_case("usd", "textured_usdz", USD_SPEC + ": a .usdz with its texture packed in", USD_OP,
               {**stats(), "uvs": True, "image_count": 1,
                "materials": [{"name": "Textured", "textured": True, "pixel": [255, 0, 0]}]})

    reset()
    d = case_dir("usd", "textured_external")
    tex = os.path.join(d, "..", "green_source.png")
    with open(tex, "wb") as f:
        f.write(png((0, 255, 0), 4))
    ob = add(bpy.ops.mesh.primitive_plane_add)
    m = material("Leaf", image=tex)
    ob.data.materials.append(m)
    usd_export("textured_external", "usda", export_textures_mode="NEW")
    os.remove(tex)
    write_case("usd", "textured_external", USD_SPEC + ": a texture copied to a textures folder beside the file",
               USD_OP, {**stats(), "uvs": True, "materials": [{"name": "Leaf", "textured": True, "pixel": [0, 255, 0]}],
                        "file": "textured_external.usda"})

    reset()
    ob = add(bpy.ops.mesh.primitive_plane_add)
    attr = ob.data.color_attributes.new("displayColor", "FLOAT_COLOR", "CORNER")
    linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.214, 0.214, 0.214)]
    for i, c in enumerate(linear):
        attr.data[i].color = (*c, 1.0)
    usd_export("vertex_colors", "usda", export_materials=False)
    write_case("usd", "vertex_colors", USD_SPEC + ": a corner color attribute named displayColor, no material",
               USD_OP, {**stats(), "colors": True, "colors_include": [[*c, 1] for c in linear]})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Lamp", color=(1, 1, 1), emission=(1.0, 0.5, 0.0))
    m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.5
    ob.data.materials.append(m)
    usd_export("emission")
    write_case("usd", "emission", USD_SPEC + ": emission color and strength", USD_OP,
               {**stats(), "materials": [{"name": "Lamp", "emissive": [0.5, 0.25, 0]}]})

    reset()
    parent = add(bpy.ops.mesh.primitive_cube_add, location=(2, 0, 1), rotation=(0.3, 0.5, 0.2), scale=(1, 2, 0.5))
    child = add(bpy.ops.mesh.primitive_cone_add, location=(0, 3, 0), rotation=(0, 0, 0.7))
    child.parent = parent
    add(bpy.ops.mesh.primitive_monkey_add, location=(-3, 0, 0), scale=(-1, 1, 1))
    for mode in ["TRS", "TOS", "MAT"]:
        usd_export("hierarchy_" + mode.lower(), xform_op_mode=mode)
        write_case("usd", "hierarchy_" + mode.lower(),
                   USD_SPEC + f": parented, rotated, scaled and mirrored objects, xformOps as {mode}", USD_OP,
                   {**stats(), "winding_matches_normals": True})

    # Linked duplicates written as instances of one prototype.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    for x in (3, 6):
        dup = ob.copy()
        dup.location = (x, 0, 0)
        bpy.context.collection.objects.link(dup)
    usd_export("instancing", use_instancing=True)
    write_case("usd", "instancing", USD_SPEC + ": linked duplicates written as instances", USD_OP,
               {**stats(), "faces_outward": False, "winding_matches_normals": True})

    reset()
    add(bpy.ops.mesh.primitive_cube_add, location=(1, 2, 3), rotation=(0.2, 0.4, 0.6), scale=(1, 2, 3))
    for name, kw, what in [
        ("y_up", dict(convert_orientation=True, export_global_forward_selection="NEGATIVE_Z",
                      export_global_up_selection="Y"), "converted to Y up"),
        ("centimetres", dict(convert_scene_units="CENTIMETERS"), "in centimetres"),
        ("millimetres_y_up", dict(convert_scene_units="MILLIMETERS", convert_orientation=True), "Y up, in millimetres"),
        ("triangulated", dict(triangulate_meshes=True), "triangulated on export"),
    ]:
        usd_export("axes_" + name, **kw)
        write_case("usd", "axes_" + name, USD_SPEC + ": " + what, USD_OP,
                   {**stats(), "winding_matches_normals": True, "faces_outward": True})

    # An animated object: the reader shows the first frame.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 10
    ob.location = (0, 0, 0)
    ob.keyframe_insert("location", frame=1)
    ob.location = (5, 0, 0)
    ob.keyframe_insert("location", frame=10)
    scene.frame_set(5)
    usd_export("animated", export_animation=True)
    scene.frame_set(1)
    write_case("usd", "animated", USD_SPEC + ": an animated transform; the first frame shows", USD_OP,
               {**stats(), "faces_outward": True})

    # Animation: an armature, a shape key and an object, each keyed from frame 0 to 24.
    anim = dict(export_animation=True)
    armature_scene()
    usd_export("armature_animated", **anim)
    write_case("usd", "armature_animated", USD_SPEC + ": a column skinned to two bones, both animated", USD_OP,
               {**stats(), "tolerance": 1e-3,
                "document": {"skins": 1, "animations": 1, "node_names": ["Lower", "Upper"]},
                "poses": poses([0, 6, 12, 24])})

    shape_key_scene()
    usd_export("shape_key_animated", **anim)
    write_case("usd", "shape_key_animated", USD_SPEC + ": a shape key whose value is animated", USD_OP,
               {**stats(), "tolerance": 1e-3, "document": {"morph_targets": 1, "animations": 1},
                "poses": poses([0, 8, 16, 24])})

    object_scene()
    usd_export("object_animated", **anim)
    write_case("usd", "object_animated", USD_SPEC + ": an object's location, rotation and scale animated", USD_OP,
               {**stats(), "tolerance": 1e-3, "document": {"animations": 1},
                "poses": poses([0, 5, 12, 19, 24])})

    cameras_lights_scene()
    usd_export("cameras_lights", export_cameras=True, export_lights=True)
    write_case("usd", "cameras_lights", USD_SPEC + ": a camera and point, sun and spot lights", USD_OP,
               {**stats(), "document": {"cameras": 1, "lights": 3}})

    print("blender usd cases written")


# --- glTF -------------------------------------------------------------------------------------

GLTF_SPEC = "glTF 2.0 as Blender's exporter writes it"
GLTF_OP = "bpy.ops.export_scene.gltf"


def gltf_export(name, fmt="GLB", **kw):
    d = case_dir("gltf", name)
    ext = "glb" if fmt == "GLB" else "gltf"
    args = dict(filepath=os.path.join(d, name + "." + ext), export_format=fmt, export_cameras=False,
                export_lights=False, export_apply=True)
    args.update(kw)
    bpy.ops.export_scene.gltf(**args)
    return d


def gltf_mat(m, **extra):
    """Blender's glTF materials carry the base color (linear), roughness and metallic as they are."""
    bsdf = m.node_tree.nodes["Principled BSDF"]
    return mat_expect(m, metallic=r(bsdf.inputs["Metallic"].default_value, 4), **extra)


def gltf_cases():
    clear_blender_cases("gltf")

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Painted", color=(0.8, 0.1, 0.05), roughness=0.4, metallic=0.3)
    ob.data.materials.append(m)
    for name, fmt in [("cube_glb", "GLB"), ("cube_separate", "GLTF_SEPARATE")]:
        gltf_export(name, fmt)
        write_case("gltf", name, GLTF_SPEC + f": {fmt}, UVs, flat normals, one material", GLTF_OP,
                   {**stats(), "uvs": True, "normals_in": AXES, "winding_matches_normals": True,
                    "faces_outward": True, "material_count": 1, "materials": [gltf_mat(m)]})

    reset()
    add(bpy.ops.mesh.primitive_monkey_add)
    bpy.ops.object.shade_smooth()
    gltf_export("suzanne_smooth")
    write_case("gltf", "suzanne_smooth", GLTF_SPEC + ": smooth normals", GLTF_OP, {**stats(), "uvs": True})

    reset()
    star = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rad = 1.0 if i % 2 == 0 else 0.4
        star.append((rad * math.cos(a), rad * math.sin(a), 0.0))
    from_polygon("Star", star)
    gltf_export("concave_star")
    write_case("gltf", "concave_star", GLTF_SPEC + ": a concave polygon, triangulated by the exporter", GLTF_OP,
               {**stats(), "normals_in": [[0, 1, 0]], "winding_matches_normals": True})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    gray = material("Gray", color=(0.5, 0.5, 0.5), roughness=0.9)
    red = material("Red", color=(0.9, 0.0, 0.0), roughness=0.2, metallic=1.0)
    ob.data.materials.append(gray)
    ob.data.materials.append(red)
    max(ob.data.polygons, key=lambda p: p.center.z).material_index = 1
    gltf_export("two_materials")
    write_case("gltf", "two_materials", GLTF_SPEC + ": one primitive per material", GLTF_OP,
               {**stats(), "primitives": 2, "material_count": 2, "materials": [gltf_mat(gray), gltf_mat(red)]})

    for name, fmt, color in [("textured_embedded", "GLB", (255, 0, 0)), ("textured_external", "GLTF_SEPARATE", (0, 255, 0))]:
        reset()
        d = case_dir("gltf", name)
        tex = os.path.join(d, "..", name + "_source.png")
        with open(tex, "wb") as f:
            f.write(png(color, 4))
        ob = add(bpy.ops.mesh.primitive_plane_add)
        m = material("Textured", image=tex)
        ob.data.materials.append(m)
        gltf_export(name, fmt)
        os.remove(tex)
        write_case("gltf", name, GLTF_SPEC + f": an image texture, {fmt}", GLTF_OP,
                   {**stats(), "uvs": True, "image_count": 1,
                    "materials": [{"name": "Textured", "textured": True, "pixel": list(color)}]})

    reset()
    ob = add(bpy.ops.mesh.primitive_plane_add)
    attr = ob.data.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.214, 0.214, 0.214)]
    for i, c in enumerate(linear):
        attr.data[i].color = (*c, 1.0)
    ob.data.color_attributes.active_color = attr
    gltf_export("vertex_colors", export_vertex_color="ACTIVE")
    write_case("gltf", "vertex_colors", GLTF_SPEC + ": COLOR_0, linear", GLTF_OP,
               {**stats(), "colors": True, "tolerance": 5e-3, "colors_include": [[*c, 1] for c in linear]})

    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    m = material("Lamp", color=(1, 1, 1), emission=(1.0, 0.5, 0.0))
    m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.5
    ob.data.materials.append(m)
    gltf_export("emission")
    write_case("gltf", "emission", GLTF_SPEC + ": emission color times strength", GLTF_OP,
               {**stats(), "materials": [{"name": "Lamp", "emissive": [0.5, 0.25, 0]}]})

    # A hierarchy of transformed objects, one of them mirrored.
    reset()
    parent = add(bpy.ops.mesh.primitive_cube_add, location=(2, 0, 1), rotation=(0.3, 0.5, 0.2), scale=(1, 2, 0.5))
    child = add(bpy.ops.mesh.primitive_cone_add, location=(0, 3, 0), rotation=(0, 0, 0.7))
    child.parent = parent
    add(bpy.ops.mesh.primitive_monkey_add, location=(-3, 0, 0), scale=(-1, 1, 1))
    gltf_export("hierarchy")
    write_case("gltf", "hierarchy", GLTF_SPEC + ": parented, rotated, non-uniformly scaled and mirrored objects",
               GLTF_OP, {**stats(), "winding_matches_normals": True})

    # One mesh shared by two objects: the exporter writes it once, used by two nodes.
    reset()
    a = add(bpy.ops.mesh.primitive_cube_add, location=(0, 0, 0))
    b = add(bpy.ops.mesh.primitive_cube_add, location=(4, 0, 0), rotation=(0, 0, 0.5))
    b.data = a.data
    gltf_export("shared_mesh")
    write_case("gltf", "shared_mesh", GLTF_SPEC + ": one mesh drawn by two nodes", GLTF_OP,
               {**stats(), "winding_matches_normals": True})

    # A shape key at 0.5: the exporter writes a morph target and the weight as the default.
    reset()
    ob = add(bpy.ops.mesh.primitive_cube_add)
    ob.shape_key_add(name="Basis")
    key = ob.shape_key_add(name="Tall")
    for v in key.data:
        if v.co.z > 0:
            v.co.z += 2
    key.value = 0.5
    gltf_export("shape_key", export_apply=False)
    write_case("gltf", "shape_key", GLTF_SPEC + ": a morph target at its default weight", GLTF_OP, stats())

    # Draco (KHR_draco_mesh_compression) at three levels. Blender's exporter writes EdgeBreaker
    # connectivity at the default and at 10 (the most compression), and sequential connectivity
    # at 0. Positions are quantized to 14 bits, so bounds hold to about a ten-thousandth of the
    # model's size.
    for name, level, how in (("draco", 6, "EdgeBreaker"), ("draco_sequential", 0, "sequential connectivity"),
                             ("draco_level_10", 10, "EdgeBreaker at the most compression")):
        reset()
        add(bpy.ops.mesh.primitive_monkey_add)
        bpy.ops.object.shade_smooth()
        gltf_export(name, export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=level)
        write_case("gltf", name, GLTF_SPEC + f": smooth normals, Draco-compressed ({how}, level {level})",
                   GLTF_OP, {**stats(), "uvs": True, "tolerance": 5e-4, "needs": "draco"})

    print("blender gltf cases written")


FORMATS = {"obj": obj_cases, "stl": stl_cases, "ply": ply_cases, "fbx": fbx_cases, "usd": usd_cases, "gltf": gltf_cases}

if __name__ == "__main__":
    wanted = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else list(FORMATS)
    for fmt in wanted:
        FORMATS[fmt]()
