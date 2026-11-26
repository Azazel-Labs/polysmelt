"""Khronos sample models for the glTF corpus: tests/corpus/gltf/khronos_*.

The models come from https://github.com/KhronosGroup/glTF-Sample-Assets at COMMIT, each under
its own license (CC0-1.0 or CC-BY-4.0 only; tests/corpus/gltf/LICENSE-khronos-samples.md, which
this writes, lists them). Where a model ships the same asset several ways, the compressed or
quantized variant's case says `same_as` the plain one: the flat meshes must match triangle for
triangle within the tolerance its quantization allows.

Expectations come from the plain variant, measured by the small reader here (accessors of any
component type, node transforms, EXT_mesh_gpu_instancing, KHR_node_visibility), never from
Polysmelt. Counts are the ones each variant's own JSON declares.

Textures larger than 64 pixels are scaled down to 64 (needs Pillow), so the corpus stays small:
nothing here tests their pixels. Everything else is copied as it is, except the one derived
variant, `khronos_meshopt_cube_fallback`: the sample's JSON with its meshopt extension objects
taken out, so its views read the uncompressed fallback buffer.

    python tools/corpus/khronos.py            # writes the cases, fetching what they need
    python tools/corpus/khronos.py --large    # only fetches the models `khronos_large_samples` reads

Files are fetched once into a cache folder: `POLYSMELT_KHRONOS_SAMPLES`, or
`target/khronos-samples`. It's laid out as the repository's `Models` folder is, so a checkout's
`Models` folder serves as it is. The `khronos_large_samples` test (tests/conformance) reads
models too large for the corpus from the same folder, and passes over those that aren't there.
"""

import io
import json
import math
import os
import shutil
import struct
import sys
import urllib.parse
import urllib.request

from common import ROOT, case

FMT = "gltf"
COMMIT = "e124e8d8d7330a5372bc2b4d00d39a4a9d7997ea"
REPO = "KhronosGroup/glTF-Sample-Assets"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Models/"
CACHE = os.environ.get("POLYSMELT_KHRONOS_SAMPLES") or os.path.normpath(
    os.path.join(ROOT, "..", "..", "target", "khronos-samples"))
TEXTURE_SIZE = 64

# Each model's license and whom it credits, from its LICENSE.md and metadata.json at COMMIT.
MODELS = {
    "Avocado": ("CC0-1.0", "Microsoft"),
    "Lantern": ("CC0-1.0", "Microsoft; Draco compression by Frank Galligan"),
    "WaterBottle": ("CC0-1.0", "Microsoft"),
    "MeshoptCubeTest": ("CC0-1.0", "Arseny Kapoulkine"),
    "SimpleInstancing": ("CC0-1.0", "Marco Hutter"),
    "NodeVisibilityTest": ("CC0-1.0", "Marco Hutter"),
    "CubeVisibility": ("CC0-1.0", "Aaron Franke, Khronos Group"),
    "TextureTransformTest": ("CC0-1.0", "Microsoft"),
    "TextureTransformMultiTest": ("CC-BY-4.0", "Ed Mackey, Analytical Graphics, Inc."),
    "UnlitTest": ("CC-BY-4.0", "Ed Mackey, Analytical Graphics, Inc."),
    "SpecGlossVsMetalRough": ("CC-BY-4.0", "Microsoft"),
}

# Models too large for the corpus, which the `khronos_large_samples` test reads from the cache.
# DragonAttenuation's dragon is under the Stanford Graphics Library's terms, not CC0 or CC-BY,
# so it's only ever fetched, never copied into the corpus.
LARGE = {
    "Corset": ["glTF/Corset.gltf", "glTF-Draco/Corset.gltf"],
    "DragonAttenuation": ["glTF/DragonAttenuation.gltf", "glTF-Meshopt/DragonAttenuation.gltf",
                          "glTF-Meshopt-EXT/DragonAttenuation.gltf"],
    "AnimationPointerUVs": ["glTF/AnimationPointerUVs.gltf"],
    "PotOfCoalsAnimationPointer": ["glTF/PotOfCoalsAnimationPointer.gltf"],
}


# --- fetching ---------------------------------------------------------------------------------

def fetch(rel):
    """The cached copy of `Models/<rel>`, fetched if it isn't there."""
    path = os.path.join(CACHE, *rel.split("/"))
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        request = urllib.request.Request(RAW + urllib.parse.quote(rel), headers={"User-Agent": "corpus-fetch"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        print("fetched", rel)
    return path


def uris(j):
    """The files a glTF's JSON names."""
    names = [x["uri"] for x in j.get("buffers", []) + j.get("images", []) if "uri" in x]
    assert not any(n.startswith("data:") for n in names)
    return [urllib.parse.unquote(n) for n in names]


def model(rel, textures=True):
    """A model's JSON and its files' bytes by name, fetched."""
    folder = rel.rsplit("/", 1)[0]
    with open(fetch(rel), "rb") as f:
        text = f.read()
    j = json.loads(text)
    files = {rel.rsplit("/", 1)[1]: text}
    images = {urllib.parse.unquote(i["uri"]) for i in j.get("images", []) if "uri" in i}
    for name in uris(j):
        if name in images and not textures:
            continue
        with open(fetch(folder + "/" + name), "rb") as f:
            files[name] = f.read()
    return j, files


def shrink(data, name):
    """An image no larger than TEXTURE_SIZE pixels a side, in the format it was."""
    from PIL import Image
    image = Image.open(io.BytesIO(data))
    if max(image.size) <= TEXTURE_SIZE:
        return data, False
    image = image.resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "PNG" if name.lower().endswith(".png") else "JPEG", optimize=True)
    return out.getvalue(), True


# --- a small reader ---------------------------------------------------------------------------

PACK = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
NORMALIZER = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
IDENT = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def accessor(j, files, i):
    """Accessor `i`'s elements as tuples, normalized integers mapped as glTF 2.0 says (3.6.2.2)."""
    a = j["accessors"][i]
    assert "sparse" not in a
    view = j["bufferViews"][a["bufferView"]]
    assert not view.get("extensions"), "a compressed view"
    data = files[urllib.parse.unquote(j["buffers"][view["buffer"]]["uri"])]
    n = COMPONENTS[a["type"]]
    fmt = "<" + PACK[a["componentType"]] * n
    stride = view.get("byteStride") or struct.calcsize(fmt)
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    out = [struct.unpack_from(fmt, data, start + k * stride) for k in range(a["count"])]
    if a.get("normalized"):
        scale = NORMALIZER[a["componentType"]]
        out = [tuple(max(x / scale, -1.0) for x in e) for e in out]
    return out


def mat_mul(a, b):
    """Column-major 4x4 product."""
    return [sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4)]


def trs(t=(0, 0, 0), q=(0, 0, 0, 1), s=(1, 1, 1)):
    x, y, z, w = q
    rot = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
           2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
           2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]
    m = []
    for c in range(3):
        m += [rot[c * 3 + r] * s[c] for r in range(3)] + [0]
    return m + [t[0], t[1], t[2], 1]


def local(node):
    if "matrix" in node:
        return node["matrix"]
    return trs(node.get("translation", (0, 0, 0)), node.get("rotation", (0, 0, 0, 1)), node.get("scale", (1, 1, 1)))


def apply(m, p):
    return tuple(m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] for r in range(3))


def drawn(j, files, visible=None):
    """What the default scene draws: (node, world matrix, mesh) for every mesh, once per
    EXT_mesh_gpu_instancing instance, leaving out what KHR_node_visibility hides. `visible`
    maps a node to the flag an animation gives it."""
    out = []

    def walk(i, parent):
        node = j["nodes"][i]
        ext = node.get("extensions", {})
        shown = ext.get("KHR_node_visibility", {}).get("visible", True)
        if not (visible or {}).get(i, shown):
            return
        world = mat_mul(parent, local(node))
        if "mesh" in node:
            attrs = ext.get("EXT_mesh_gpu_instancing", {}).get("attributes")
            if attrs is None:
                out.append((i, world, node["mesh"]))
            else:
                parts = {k: accessor(j, files, v) for k, v in attrs.items()}
                for k in range(min(len(v) for v in parts.values())):
                    inst = trs(parts.get("TRANSLATION", [(0, 0, 0)] * (k + 1))[k],
                               parts.get("ROTATION", [(0, 0, 0, 1)] * (k + 1))[k],
                               parts.get("SCALE", [(1, 1, 1)] * (k + 1))[k])
                    out.append((i, mat_mul(world, inst), node["mesh"]))
        for c in node.get("children", []):
            walk(c, world)

    for root in j["scenes"][j.get("scene", 0)]["nodes"]:
        walk(root, IDENT)
    return out


def measure(j, files, visible=None):
    """Triangles, vertices, bounds and area of what's drawn, and the materials drawn with."""
    tris = verts = 0
    area = 0.0
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    used = []
    for _, world, mesh in drawn(j, files, visible):
        for prim in j["meshes"][mesh]["primitives"]:
            assert prim.get("mode", 4) == 4
            pos = [apply(world, p) for p in accessor(j, files, prim["attributes"]["POSITION"])]
            idx = [e[0] for e in accessor(j, files, prim["indices"])] if "indices" in prim else list(range(len(pos)))
            verts += len(pos)
            tris += len(idx) // 3
            if prim.get("material") not in used:
                used.append(prim.get("material"))
            for p in pos:
                for k in range(3):
                    lo[k], hi[k] = min(lo[k], p[k]), max(hi[k], p[k])
            for t in range(0, len(idx) - 2, 3):
                a, b, c = (pos[idx[t + k]] for k in range(3))
                u = [b[k] - a[k] for k in range(3)]
                v = [c[k] - a[k] for k in range(3)]
                cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
                area += 0.5 * math.sqrt(sum(x * x for x in cross))
    return {"triangles": tris, "vertices": verts, "area": round(area, 6),
            "bounds": {"min": [round(x, 6) for x in lo], "max": [round(x, 6) for x in hi]},
            "material_count": len(used)}, used


def declared(j):
    """Triangles and vertices as a JSON declares them, each mesh once, with no instancing: all
    a compressed variant can be counted by."""
    tris = verts = 0
    nodes = [n for n in j["nodes"] if "mesh" in n]
    assert not any("EXT_mesh_gpu_instancing" in n.get("extensions", {}) for n in nodes)
    for n in nodes:
        for prim in j["meshes"][n["mesh"]]["primitives"]:
            count = j["accessors"][prim["attributes"]["POSITION"]]["count"]
            verts += count
            tris += (j["accessors"][prim["indices"]]["count"] if "indices" in prim else count) // 3
    return tris, verts


def materials(j, used):
    """Checks of the materials drawn with, by name, from the JSON; none unless every one has a
    name of its own."""
    names = [j["materials"][m].get("name") for m in used if m is not None]
    if None in used or None in names or "" in names or len(set(names)) != len(names):
        return []
    out = []
    for m in used:
        mat = j["materials"][m]
        ext = mat.get("extensions", {})
        pbr = mat.get("pbrMetallicRoughness")
        check = {"name": mat["name"], "double_sided": mat.get("doubleSided", False),
                 "alpha_mode": mat.get("alphaMode", "OPAQUE"), "normal_map": "normalTexture" in mat,
                 "emissive": mat.get("emissiveFactor", [0, 0, 0])}
        gloss = ext.get("KHR_materials_pbrSpecularGlossiness")
        if pbr is None and gloss is not None:
            color = gloss.get("diffuseFactor", [1, 1, 1, 1])
            check["textured"] = "diffuseTexture" in gloss
        else:
            pbr = pbr or {}
            color = pbr.get("baseColorFactor", [1, 1, 1, 1])
            check["textured"] = "baseColorTexture" in pbr
            check["metallic"] = pbr.get("metallicFactor", 1)
            check["roughness"] = pbr.get("roughnessFactor", 1)
        # An opaque material's alpha isn't drawn.
        check["color"] = color if check["alpha_mode"] != "OPAQUE" else color[:3]
        out.append(check)
    return out


# --- the cases --------------------------------------------------------------------------------

NOTES = []


def source(name, variant, changes):
    license_, credit = MODELS[name]
    text = f"{REPO}@{COMMIT[:7]}, Models/{name}/{variant} ({license_}, {credit})"
    return text + ("; " + "; ".join(changes) if changes else "")


def sample(case_name, name, variant, spec, reference=None, same=None, needs=None, edit=None, extra=None):
    """Writes one case from `Models/<name>/<variant>/<name>.gltf`. Its expectations are measured
    from `reference` (a (json, files) pair of the plain variant) or, without one, from itself."""
    j, files = model(f"{name}/{variant}/{name}.gltf")
    changes = []
    if edit:
        changes.append(edit(j))
        files[f"{name}.gltf"] = json.dumps(j, indent=1).encode()
    shrunk = False
    for image in {urllib.parse.unquote(i["uri"]) for i in j.get("images", [])}:
        files[image], did = shrink(files[image], image)
        shrunk |= did
    if shrunk:
        changes.append(f"textures scaled down to {TEXTURE_SIZE} pixels")
    measured, used = measure(*(reference or (j, files)))
    expect = {"spec": spec, "source": source(name, variant, changes), "file": f"{name}.gltf", **measured}
    if reference:
        expect["triangles"], expect["vertices"] = declared(j)
    checks = materials(reference[0] if reference else j, used)
    if checks:
        expect["materials"] = checks
    if needs:
        expect["needs"] = needs
    if same:
        expect["same_as"] = same
        # The reference's bounds and area, as far off as the variant's quantization puts them.
        expect["tolerance"] = max(1e-3, same.get("positions", 0))
    expect.update(extra or {})
    d = os.path.join(ROOT, FMT, case_name)
    shutil.rmtree(d, ignore_errors=True)
    case(FMT, case_name, files, expect)
    NOTES.append((case_name, name, variant, changes))
    return j, files


def quantization(j):
    """How far a gltfpack-quantized variant's values may be from the plain one's: half a step
    of each attribute's integers, in the units the node's scale and the texture transform give
    them, and a little for the float arithmetic on the way."""
    node = next(n for n in j["nodes"] if "mesh" in n)
    prim = j["meshes"][node["mesh"]]["primitives"][0]

    def step(semantic):
        a = j["accessors"][prim["attributes"][semantic]]
        assert a["componentType"] != 5126
        return 1 / NORMALIZER[a["componentType"]] if a.get("normalized") else 1.0

    info = j["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]
    uv_scale = info["extensions"]["KHR_texture_transform"]["scale"]
    return {"positions": float("%.3g" % (0.51 * step("POSITION") * max(node["scale"]))),
            "normals": float("%.3g" % (0.51 * step("NORMAL") * 2)),
            "uvs": float("%.3g" % (0.51 * step("TEXCOORD_0") * max(uv_scale)))}


def strip_meshopt(j):
    """Takes the meshopt extension objects out, leaving views of the fallback buffer."""
    for x in j["bufferViews"] + j["buffers"]:
        x.get("extensions", {}).pop("KHR_meshopt_compression", None)
        if x.get("extensions") == {}:
            del x["extensions"]
    j["extensionsUsed"].remove("KHR_meshopt_compression")
    return "the JSON's KHR_meshopt_compression objects taken out, so it reads the fallback buffer"


def large():
    for name, variants in LARGE.items():
        for variant in variants:
            model(f"{name}/{variant}", textures=False)
    print("large samples are in", CACHE)


def main():
    plain = "the same as its uncompressed variant"

    # Draco's quantization is in its streams, not the JSON. These hold 11-bit positions, 10-bit
    # texture coordinates and 8-bit normals, as their largest errors show: half a step of each,
    # positions over the mesh's largest extent (the encoder quantizes in a cube), texture
    # coordinates over the unit square they fill, normals over an octahedron.
    def draco(j):
        box = [(a["min"], a["max"]) for a in j["accessors"] if "min" in a and a["type"] == "VEC3"]
        extent = max(hi[k] - lo[k] for lo, hi in box for k in range(3))
        return {"positions": float("%.3g" % (0.51 * extent / (2 ** 11 - 1))),
                "normals": 0.015, "uvs": float("%.3g" % (0.51 / (2 ** 10 - 1)))}

    ref = sample("khronos_avocado", "Avocado", "glTF", "glTF 2.0 (Khronos, 2021): Khronos sample Avocado")
    j, _ = model("Avocado/glTF-Quantized/Avocado.gltf", textures=False)
    sample("khronos_avocado_quantized", "Avocado", "glTF-Quantized",
           "KHR_mesh_quantization: Khronos sample Avocado (gltfpack), " + plain, ref,
           {"case": "khronos_avocado", **quantization(j)})
    sample("khronos_avocado_draco", "Avocado", "glTF-Draco",
           "KHR_draco_mesh_compression: Khronos sample Avocado, " + plain, ref,
           {"case": "khronos_avocado", **draco(ref[0])}, needs="draco")

    ref = sample("khronos_lantern", "Lantern", "glTF", "glTF 2.0 (Khronos, 2021): Khronos sample Lantern, three meshes")
    j, _ = model("Lantern/glTF-Quantized/Lantern.gltf", textures=False)
    sample("khronos_lantern_quantized", "Lantern", "glTF-Quantized",
           "KHR_mesh_quantization: Khronos sample Lantern (gltfpack, its meshes merged), " + plain, ref,
           {"case": "khronos_lantern", **quantization(j)})
    sample("khronos_lantern_draco", "Lantern", "glTF-Draco",
           "KHR_draco_mesh_compression: Khronos sample Lantern, " + plain, ref,
           {"case": "khronos_lantern", **draco(ref[0])}, needs="draco")

    ref = sample("khronos_water_bottle", "WaterBottle", "glTF", "glTF 2.0 (Khronos, 2021): Khronos sample WaterBottle")
    sample("khronos_water_bottle_draco", "WaterBottle", "glTF-Draco",
           "KHR_draco_mesh_compression: Khronos sample WaterBottle, " + plain, ref,
           {"case": "khronos_water_bottle", **draco(ref[0])}, needs="draco")

    # The fallback buffer holds what the compressed views decode to, the lossy filters of rows 4
    # and 5 included, so the two must match exactly.
    ref = sample("khronos_meshopt_cube_fallback", "MeshoptCubeTest", "glTF",
                 "KHR_mesh_quantization: Khronos sample MeshoptCubeTest read from its uncompressed fallback buffer",
                 edit=strip_meshopt)
    for case_name, variant, what in [
        ("khronos_meshopt_cube", "glTF", "with its fallback buffer beside it"),
        ("khronos_meshopt_cube_required", "glTF-Meshopt", "required, with no fallback buffer"),
    ]:
        sample(case_name, "MeshoptCubeTest", variant,
               f"KHR_meshopt_compression: Khronos sample MeshoptCubeTest {what} (every mode, filter and version), "
               "the same as its fallback data", ref, {"case": "khronos_meshopt_cube_fallback"})

    sample("khronos_simple_instancing", "SimpleInstancing", "glTF",
           "EXT_mesh_gpu_instancing: Khronos sample SimpleInstancing, 125 cubes placed by translation, rotation "
           "and scale")

    sample("khronos_node_visibility", "NodeVisibilityTest", "glTF",
           "KHR_node_visibility: Khronos sample NodeVisibilityTest, hidden nodes and what's under them")

    # The blue cube blinks: an animation pointer to its node's visibility, held between keys.
    j, files = model("CubeVisibility/glTF/CubeVisibility.gltf")
    (channel,) = j["animations"][0]["channels"]
    pointer = channel["target"]["extensions"]["KHR_animation_pointer"]["pointer"]
    node = int(pointer.split("/")[2])
    assert pointer == f"/nodes/{node}/extensions/KHR_node_visibility/visible"
    sampler = j["animations"][0]["samplers"][channel["sampler"]]
    assert sampler["interpolation"] == "STEP"
    times = [t[0] for t in accessor(j, files, sampler["input"])]
    flags = [bool(v[0]) for v in accessor(j, files, sampler["output"])]
    poses = []
    for k in (flags.index(False), flags.index(True, flags.index(False))):
        # Between this key and the next, its value holds.
        time = (times[k] + times[k + 1]) / 2
        poses.append({"animation": 0, "time": time, "bounds": measure(j, files, {node: flags[k]})[0]["bounds"]})
    assert poses[0]["bounds"] != poses[1]["bounds"]
    sample("khronos_cube_visibility", "CubeVisibility", "glTF",
           "KHR_node_visibility: Khronos sample CubeVisibility, a hidden node's descendants, and a node an "
           "animation pointer shows and hides", extra={"poses": poses})

    sample("khronos_texture_transform", "TextureTransformTest", "glTF",
           "KHR_texture_transform: Khronos sample TextureTransformTest")
    sample("khronos_texture_transform_multi", "TextureTransformMultiTest", "glTF",
           "KHR_texture_transform: Khronos sample TextureTransformMultiTest, a transform on every texture slot")
    sample("khronos_unlit", "UnlitTest", "glTF", "KHR_materials_unlit: Khronos sample UnlitTest",
           extra={"flat_json": {"/materials/0/extensions/KHR_materials_unlit": {},
                                "/materials/1/extensions/KHR_materials_unlit": {}}})
    sample("khronos_spec_gloss", "SpecGlossVsMetalRough", "glTF",
           "KHR_materials_pbrSpecularGlossiness: Khronos sample SpecGlossVsMetalRough, one bottle in each workflow")

    lines = [
        "# Khronos glTF sample models",
        "",
        f"The `khronos_*` cases are models from https://github.com/{REPO}, at commit",
        f"`{COMMIT}`, copied by `tools/corpus/khronos.py`. Each is under the license its",
        "model's `LICENSE.md` there gives:",
        "",
        "| Case | Model and variant | License | Credit | Changes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case_name, name, variant, changes in NOTES:
        license_, credit = MODELS[name]
        lines.append(f"| `{case_name}` | `Models/{name}/{variant}` | {license_} | {credit} | "
                     f"{'; '.join(changes) or 'none'} |")
    lines += [
        "",
        "CC0-1.0: https://creativecommons.org/publicdomain/zero/1.0/legalcode",
        "CC-BY-4.0: https://creativecommons.org/licenses/by/4.0/legalcode",
        "",
    ]
    with open(os.path.join(ROOT, FMT, "LICENSE-khronos-samples.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("khronos cases written")


if __name__ == "__main__":
    large() if "--large" in sys.argv[1:] else main()
