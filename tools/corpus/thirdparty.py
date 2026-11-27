"""Real files from other projects, measured by other tools: the `ref_*` cases of many formats.

Most of the corpus is written by this project's own generators, so its expectations share
their author's reading of each specification. These cases don't: each file comes from another
project's repository at a pinned commit, and its expectations come from an independent
reader, never from Polysmelt and never from a parser written here:

- assimp (the Open Asset Import Library, through the `assimp-py` wheel, which bundles assimp
  6.0.5) for most formats,
- lib3mf (the 3MF Consortium's reference implementation) for 3MF,
- ezdxf for the DXF files that hold lines and points, which assimp-py can't return,
- py-vox-io for MagicaVoxel models,
- ezc3d for C3D. No C3D sample files have clear terms, so ezc3d writes these cases' files
  too: another project's writer, and its reader for the expectations.

What is compared: the triangles, line segments and points drawn, their bounds, the surface
area, the set of distinct positions (`position_set`, a file of little-endian f32 triples the
conformance test holds against what's drawn, in any order and however either side welds),
and, where a file has them, node names, bones, where each bone's origin is in the scene at
rest and at a key of an animation, the animations' spans and the bounds of the skinned mesh
at that key.

The two sides don't share a space: Polysmelt turns every file to glTF's (Y up, meters, right
handed), and each reference tool has its own habits. Every case states how the reference's
numbers map to glTF space (`axes` and `scale` below) and why. A mapping is only ever the
format's own convention (its up axis and its unit), never a fit to make a case pass. Where a
reference tool is known to be wrong about a file, the case's `source` says so and leaves that
fact out.

    pip install assimp-py lib3mf ezdxf ezc3d py-vox-io numpy
    python tools/corpus/thirdparty.py            # every case
    python tools/corpus/thirdparty.py off 3ds    # only these formats

Files are fetched once into a cache folder: `POLYSMELT_THIRDPARTY`, or `target/thirdparty`,
laid out as `<source>/<path in its repository>`. A generator of a format's own cases leaves
the `ref_*` folders alone (`common.fresh`).
"""

import json
import math
import os
import shutil
import sys
import urllib.parse
import urllib.request

import numpy as np

# The shared modules are beside this script, which `python -I` leaves off the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

CACHE = os.environ.get("POLYSMELT_THIRDPARTY") or os.path.normpath(
    os.path.join(ROOT, "..", "..", "target", "thirdparty"))

# Where the files come from. `license` is the repository's own, read from `license_file` at
# `commit`; a file with its own terms says so in its case (`terms`).
SOURCES = {
    "assimp": {
        "repo": "assimp/assimp",
        "commit": "711b4e9bbde25bdadfc1821e9a66ab80f02f3fe3",
        "root": "test/models/",
        "license": "BSD-3-Clause",
        "license_file": "LICENSE",
        "what": "the Open Asset Import Library's test models (`test/models`, never `test/models-nonbsd`)",
    },
    "3mf-samples": {
        "repo": "3MFConsortium/3mf-samples",
        "commit": "665e20dc4d7777fd4c9702bca86a2d4028440337",
        "license": "BSD-2-Clause",
        "license_file": "LICENSE",
        "what": "the 3MF Consortium's sample files",
    },
    "voxel-model": {
        "repo": "ephtracy/voxel-model",
        "commit": "3bff8feaa7d94a6237f2c07dc4ca3c3d268ef06c",
        "license": "MIT",
        "license_file": "LICENSE",
        "what": "MagicaVoxel's author's voxel models",
    },
}

# glTF space from the reference tool's, as the rows of a matrix.
AXES = {
    "SAME": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # Z up, right handed: x stays, z is up, y points away from the viewer.
    "Z_UP": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    # Left handed, Y up, as stored: glTF's is its mirror image through the XY plane.
    "MIRROR_Z": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
}


# --- fetching ---------------------------------------------------------------------------------

def fetch(source, rel, root=True):
    """The cached copy of a source's file (under its `root`, or the repository's), fetched if
    it isn't there."""
    s = SOURCES[source]
    path = os.path.join(CACHE, source, *rel.split("/"))
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://raw.githubusercontent.com/{s['repo']}/{s['commit']}/" + urllib.parse.quote(
            (s.get("root", "") if root else "") + rel)
        request = urllib.request.Request(url, headers={"User-Agent": "corpus-fetch"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
        print("fetched", source, rel)
    return path


# --- measuring with assimp --------------------------------------------------------------------

def quat_matrix(q):
    """A rotation matrix from a quaternion as assimp-py gives it (x, y, z, w)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(a, b, f):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9999:
        q = a + (b - a) * f
    else:
        angle = math.acos(d)
        q = (math.sin((1 - f) * angle) * a + math.sin(f * angle) * b) / math.sin(angle)
    return q / np.linalg.norm(q)


def key_at(times, values, t, width, turn=False):
    """A channel's value at `t` ticks: its keys joined by straight lines (the shorter way
    round for rotations), held before the first and after the last, as assimp's own viewer
    plays them."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1, width)
    i = int(np.searchsorted(times, t, side="right"))
    if i == 0:
        return values[0]
    if i == len(times):
        return values[-1]
    f = (t - times[i - 1]) / (times[i] - times[i - 1])
    if turn:
        return slerp(values[i - 1], values[i], f)
    return values[i - 1] + (values[i] - values[i - 1]) * f


class Assimp:
    """A file as assimp reads it, triangulated, each mesh holding one kind of primitive."""

    def __init__(self, path, flags=0):
        import assimp_py
        self.scene = assimp_py.import_file(path, assimp_py.Process_Triangulate | assimp_py.Process_SortByPType | flags)
        self.nodes = []      # (node, parent index)
        self.index = {}

        def walk(n, parent):
            i = len(self.nodes)
            self.nodes.append((n, parent))
            self.index.setdefault(n.name, i)
            for c in n.children:
                walk(c, i)

        walk(self.scene.root_node, None)
        self.rest = [np.array(n.transformation, dtype=np.float64).reshape(4, 4) for n, _ in self.nodes]

    def world(self, local=None):
        """Every node's world matrix, from each one's matrix in its parent."""
        local = local or self.rest
        out = []
        for i, (_, parent) in enumerate(self.nodes):
            out.append(local[i] if parent is None else out[parent] @ local[i])
        return out

    def posed(self, animation, ticks):
        """Each node's matrix in its parent at `ticks` of an animation. A channel replaces its
        node's whole transform, as assimp documents."""
        local = list(self.rest)
        for ch in self.scene.animations[animation].channels:
            i = self.index.get(ch.node_name)
            if i is None:
                continue
            m = np.eye(4)
            r = quat_matrix(key_at(ch.rotation_key_times, ch.rotation_key_values, ticks, 4, True)) \
                if ch.num_rotation_keys else np.eye(3)
            s = key_at(ch.scaling_key_times, ch.scaling_key_values, ticks, 3) if ch.num_scaling_keys else np.ones(3)
            m[:3, :3] = r * s
            if ch.num_position_keys:
                m[:3, 3] = key_at(ch.position_key_times, ch.position_key_values, ticks, 3)
            local[i] = m
        return local

    def drawn(self, local=None):
        """World-space triangles (n, 3, 3), line segments (n, 2, 3) and points (n, 1, 3), at
        rest or in a pose. A mesh with bones is drawn where they put it, as glTF draws a skin,
        not where its node is: each vertex is the weighted sum of its bones' world matrices
        times their offset matrices."""
        world = self.world(local)
        kinds = {1: [], 2: [], 3: []}
        for i, (n, _) in enumerate(self.nodes):
            for mi in n.mesh_indices:
                m = self.scene.meshes[mi]
                if m.num_faces == 0:
                    continue
                v = np.asarray(m.vertices, dtype=np.float64).reshape(-1, 3)
                placed = v @ world[i][:3, :3].T + world[i][:3, 3]
                if m.bones:
                    total = np.zeros(len(v))
                    skinned = np.zeros_like(v)
                    for b in m.bones:
                        j = self.index[b.name]
                        mat = world[j] @ np.array(b.offset_matrix, dtype=np.float64).reshape(4, 4)
                        ids = np.asarray(b.weight_vertex_ids, dtype=np.int64)
                        w = np.asarray(b.weights, dtype=np.float64)
                        skinned[ids] += w[:, None] * (v[ids] @ mat[:3, :3].T + mat[:3, 3])
                        total[ids] += w
                    bound = total > 1e-6
                    placed[bound] = skinned[bound] / total[bound][:, None]
                idx = np.asarray(m.indices, dtype=np.int64)
                k = len(idx) // m.num_faces
                kinds[k].append(placed[idx.reshape(-1, k)])
        return [np.concatenate(kinds[k]) if kinds[k] else np.zeros((0, k, 3)) for k in (3, 2, 1)]

    def bones(self):
        """The names of the bones meshes are bound to, in node order."""
        names = {b.name for m in self.scene.meshes for b in m.bones}
        return [n.name for n, _ in self.nodes if n.name in names]


# --- expectations -----------------------------------------------------------------------------

def rounded(v, digits=6):
    return [round(float(x), digits) + 0.0 for x in v]


def geometry(expect, files, tris, lines, points, axes, scale, skip=(), twice=False):
    """The drawn geometry's facts, in glTF space, but those named in `skip`.

    Triangles with two corners in one place, which draw nothing, are counted as the reference
    counts them: a reader keeps the triangles a file lists. `twice` halves the triangles and
    their area: assimp draws a two-sided surface twice, facing each way, where glTF has a
    material flag."""
    m = np.array(axes, dtype=np.float64) * scale
    sides = 2 if twice else 1
    tris, lines, points = (a @ m.T for a in (tris, lines, points))
    every = np.concatenate([tris.reshape(-1, 3), lines.reshape(-1, 3), points.reshape(-1, 3)])
    if not len(every):
        expect["no_mesh"] = True
        return
    if "triangles" not in skip:
        expect["triangles"] = len(tris) // sides
        if len(lines):
            expect["lines"] = len(lines)
        if len(points):
            expect["points"] = len(points)
    if "bounds" not in skip:
        expect["bounds"] = {"min": rounded(every.min(0)), "max": rounded(every.max(0))}
    if len(tris) and "area" not in skip:
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        expect["area"] = round(float(0.5 * np.linalg.norm(cross, axis=1).sum()) / sides, 6)
    if "positions" not in skip:
        distinct = np.unique(every.astype("<f4"), axis=0)
        expect["position_set"] = "positions.f32"
        files["positions.f32"] = distinct.astype("<f4").tobytes()


def bounds_of(parts, axes, scale):
    m = np.array(axes, dtype=np.float64) * scale
    every = np.concatenate([a.reshape(-1, 3) for a in parts]) @ m.T
    return {"min": rounded(every.min(0), 5), "max": rounded(every.max(0), 5)}


def collada_names(path):
    """Each node's name, by its id."""
    import xml.etree.ElementTree as ET
    return {n.get("id"): n.get("name") for n in ET.parse(path).iter() if n.tag.endswith("}node") and n.get("name")}


def measure_assimp(c, path, expect, files):
    a = Assimp(path)
    axes, scale = AXES[c.get("axes", "SAME")], c.get("scale", 1.0)
    m = np.array(axes, dtype=np.float64) * scale
    if c.get("geometry") is False:
        pass
    elif c.get("static") == "first_key":
        # assimp's scene stands as the animation's first key has it, not as the file's own
        # vertices do: its numbers are a pose's, not the rest's.
        tris, lines, points = a.drawn()
        expect["triangles"] = len(tris)
        first = min(float(t[0]) for ch in a.scene.animations[0].channels for t in (
            ch.position_key_times, ch.rotation_key_times, ch.scaling_key_times) if len(t))
        rate = a.scene.animations[0].ticks_per_second or c["ticks_per_second"]
        expect["poses"] = [{"animation": 0, "time": round(first / rate, 6),
                            "bounds": bounds_of([tris, lines, points], axes, scale)}]
    else:
        geometry(expect, files, *a.drawn(), axes, scale, c.get("skip", ()), c.get("twice", False))
    document = {}
    names = c.get("node_names")
    if names == "all":
        names = [n.name for n, _ in a.nodes if n.name and not n.name.startswith(("<", "EndSite_"))]
    if names:
        document["node_names"] = names
    # assimp calls a COLLADA node by its id, Polysmelt by its name.
    called = collada_names(path) if c["fmt"] == "dae" else {}
    rig = []
    # A BVH End Site has no name of its own; assimp makes one up.
    bones = [b for b in a.bones() if not b.startswith("EndSite_")]
    for name in c.get("bones", []) if c.get("bones") != "all" else bones:
        i = a.index[name]
        parent = a.nodes[i][1]
        entry = {"node": called.get(name, name), "world": rounded(m @ a.world()[i][:3, 3], 5)}
        if c.get("parents", True) and parent is not None and not a.nodes[parent][0].name.startswith("<"):
            entry["parent"] = called.get(a.nodes[parent][0].name, a.nodes[parent][0].name)
        rig.append(entry)
    anims = a.scene.animations
    if c.get("animations", True) and anims:
        document["animations"] = len(anims)
        spans = []
        for k, anim in enumerate(anims):
            rate = anim.ticks_per_second or c["ticks_per_second"]
            times = np.concatenate([np.asarray(t, dtype=np.float64) for ch in anim.channels for t in (
                ch.position_key_times, ch.rotation_key_times, ch.scaling_key_times)])
            # Every reader but glTF's has an animation start at 0: the time of its earliest key (or the clip's
            # own start, where the case gives the clips' `starts`) is taken off every key.
            start = 0.0 if c["fmt"] == "gltf" else times.min() / rate
            if c.get("starts"):
                start = c["starts"][k]
            spans.append(rounded([times.min() / rate - start, times.max() / rate - start], 5))
            # Posed at a key in the middle of the animation.
            keys = np.unique(times)
            ticks = float(c.get("pose_tick", keys[len(keys) // 2]))
            local = a.posed(k, ticks)
            world = a.world(local)
            for name in c.get("posed", []) if c.get("posed") != "all" else bones:
                rig.append({"node": called.get(name, name), "animation": k, "time": round(ticks / rate - start, 6),
                            "world": rounded(m @ world[a.index[name]][:3, 3], 5)})
            if c.get("pose_bounds", False):
                expect.setdefault("poses", []).append({
                    "animation": k, "time": round(ticks / rate - start, 6),
                    "bounds": bounds_of(a.drawn(local), axes, scale)})
        if c.get("spans", True):
            expect["animation_spans"] = spans
    if c.get("joints"):
        document["joints"] = len(a.bones())
    if c.get("material_count"):
        expect["material_count"] = a.scene.num_materials
    if document:
        expect["document"] = document
    if rig:
        expect["rig"] = rig


def measure_lib3mf(c, path, expect, files):
    """A 3MF package's build as lib3mf reads it: every build item's object, a mesh or
    components of other objects, each placed by its transform. 3MF's matrices move row
    vectors; its space is Z up, in the model's unit."""
    import lib3mf
    model = lib3mf.get_wrapper().CreateModel()
    model.QueryReader("3mf").ReadFromFile(path)
    units = lib3mf.ModelUnit
    scale = {units.MicroMeter: 1e-6, units.MilliMeter: 1e-3, units.CentiMeter: 1e-2, units.Inch: 0.0254,
             units.Foot: 0.3048, units.Meter: 1.0}[model.GetUnit()]

    def matrix(t):
        rows = np.array([list(r) for r in t.Fields], dtype=np.float64)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = rows[:3].T, rows[3]
        return m

    tris = []

    def draw(obj, m):
        if obj.IsMeshObject():
            mesh = model.GetMeshObjectByID(obj.GetUniqueResourceID())
            v = np.array([list(p.Coordinates) for p in mesh.GetVertices()], dtype=np.float64)
            t = np.array([list(t.Indices) for t in mesh.GetTriangleIndices()], dtype=np.int64)
            tris.append((v @ m[:3, :3].T + m[:3, 3])[t])
        elif obj.IsComponentsObject():
            parts = model.GetComponentsObjectByID(obj.GetUniqueResourceID())
            for k in range(parts.GetComponentCount()):
                part = parts.GetComponent(k)
                draw(part.GetObjectResource(), m @ matrix(part.GetTransform()) if part.HasTransform() else m)

    items = model.GetBuildItems()
    while items.MoveNext():
        item = items.GetCurrent()
        draw(item.GetObjectResource(), matrix(item.GetObjectTransform()) if item.HasObjectTransform() else np.eye(4))
    none = np.zeros((0, 2, 3)), np.zeros((0, 1, 3))
    geometry(expect, files, np.concatenate(tris), *none, AXES["Z_UP"], scale, c.get("skip", ()))


def measure_ezdxf(c, path, expect, files):
    """A drawing's model space as ezdxf reads it, for the entities that need no working out:
    a 3DFACE's three or four corners (a face whose last two corners are one is a triangle), a
    LINE's two ends and a POINT. DXF is Z up; these drawings give no unit."""
    import ezdxf
    tris, lines, points = [], [], []
    for e in ezdxf.readfile(path).modelspace():
        kind = e.dxftype()
        if kind == "3DFACE":
            a, b, c3, d = (tuple(v) for v in (e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3))
            tris.append([a, b, c3])
            if d != c3:
                tris.append([a, c3, d])
        elif kind == "LINE":
            lines.append([tuple(e.dxf.start), tuple(e.dxf.end)])
        elif kind == "POINT":
            points.append([tuple(e.dxf.location)])
        else:
            raise ValueError(f"{kind} isn't measured here")
    shaped = [np.array(a, dtype=np.float64).reshape(-1, k, 3) for a, k in ((tris, 3), (lines, 2), (points, 1))]
    geometry(expect, files, *shaped, AXES["Z_UP"], 1.0, c.get("skip", ()))


def measure_pyvox(c, path, expect, files):
    """A MagicaVoxel model as py-vox-io reads it: its voxels. What's drawn is every face
    between a voxel and empty space, a unit square each, so their count is the surface area;
    the triangles depend on how a reader merges faces and aren't compared. A file without a
    scene graph keeps its voxels' own coordinates, Z up."""
    from pyvox.parser import VoxParser
    vox = VoxParser(path).parse()
    assert len(vox.models) == 1
    v = np.array([(p.x, p.y, p.z) for p in vox.models[0].voxels], dtype=np.int64)
    full = np.zeros(v.max(0) + 3, dtype=bool)
    full[tuple((v + 1).T)] = True
    faces = sum(int((full & ~np.roll(full, step, axis)).sum()) for axis in range(3) for step in (1, -1))
    corners = np.array([v.min(0), v.max(0) + 1], dtype=np.float64) @ np.array(AXES["Z_UP"], dtype=np.float64).T
    expect["area"] = faces
    expect["bounds"] = {"min": rounded(corners.min(0)), "max": rounded(corners.max(0))}
    size = vox.models[0].size
    expect["facts"] = {"voxels": len(v), "voxel_size": f"{size.x}x{size.y}x{size.z}"}
    expect["material_count"] = len({p.c for p in vox.models[0].voxels})


def make_c3d(c, path):
    """Writes a C3D file with ezc3d: markers on made-up paths, in the units and at the rate
    the case gives."""
    import ezc3d
    labels, frames = c["labels"], c["frames"]
    out = ezc3d.c3d()
    out["parameters"]["POINT"]["RATE"]["value"] = [c["rate"]]
    out["parameters"]["POINT"]["LABELS"]["value"] = tuple(labels)
    if c.get("units"):
        out["parameters"]["POINT"]["UNITS"]["value"] = [c["units"]]
    for name, value in c.get("screen", {}).items():
        out.add_parameter("POINT", {"name": name, "type": -1, "dimension": [2], "value": [value]})
    k, f = np.meshgrid(np.arange(len(labels)), np.arange(frames), indexing="ij")
    points = np.zeros((4, len(labels), frames))
    points[0] = 100.0 * k + 10.0 * np.sin(0.7 * f + k)
    points[1] = -50.0 * k + 3.5 * f
    points[2] = 900.0 + 25.0 * np.cos(0.3 * f * (k + 1))
    for marker, frame in c.get("missing", []):
        points[:3, marker, frame] = np.nan
    out["data"]["points"] = points
    if c.get("analogs"):
        out["parameters"]["ANALOG"]["RATE"]["value"] = [c["rate"] * c["analogs"]]
        out["parameters"]["ANALOG"]["LABELS"]["value"] = ("Fz",)
        out["data"]["analogs"] = np.arange(frames * c["analogs"], dtype=np.float64).reshape(1, 1, -1)
    out.write(path)


def measure_ezc3d(c, path, expect, files):
    """A C3D capture as ezc3d reads it back: its markers' labels and where each is at each
    frame, in the file's unit and axes. Frame k is at k / rate seconds. A sample ezc3d reads
    as NaN is missing."""
    import ezc3d
    read = ezc3d.c3d(path)
    labels = list(read["parameters"]["POINT"]["LABELS"]["value"])
    rate = float(read["header"]["points"]["frame_rate"])
    points = np.asarray(read["data"]["points"])[:3]
    frames = points.shape[2]
    m = np.array(AXES[c.get("axes", "Z_UP")], dtype=np.float64) * c.get("scale", 0.001)
    missing = np.isnan(points).any(0)
    expect["no_mesh"] = True
    expect["facts"] = {"markers": len(labels), "frames": frames}
    if missing.any():
        expect["facts"]["missing_samples"] = int(missing.sum())
    expect["document"] = {"animations": 1, "node_names": labels}
    expect["animation_spans"] = [rounded([0.0, (frames - 1) / rate], 5)]
    rig = []
    for k, label in enumerate(labels):
        present = [f for f in range(frames) if not missing[k, f]]
        rig.append({"node": label, "keys": {"translation": len(present)}})
        for f in sorted({present[0], present[len(present) // 2], present[-1]}):
            rig.append({"node": label, "time": round(f / rate, 6), "world": rounded(m @ points[:, k, f], 5)})
    expect["rig"] = rig


def measure_assimp_motion(c, path, expect, files):
    """A motion capture file's skeleton and motion as assimp reads them. The bones assimp
    draws for it are its own picture of the skeleton, not the file's, and aren't compared."""
    measure_assimp({**c, "geometry": False}, path, expect, files)
    expect["no_mesh"] = True


MEASURES = {
    "assimp": (measure_assimp, "assimp 6.0.5 (assimp-py 1.2.0)"),
    "assimp_motion": (measure_assimp_motion, "assimp 6.0.5 (assimp-py 1.2.0)"),
    "lib3mf": (measure_lib3mf, "lib3mf 2.5.0"),
    "ezdxf": (measure_ezdxf, "ezdxf 1.4.5"),
    "pyvox": (measure_pyvox, "py-vox-io 0.1"),
    "ezc3d": (measure_ezc3d, "ezc3d 1.7.2"),
}

# --- the cases --------------------------------------------------------------------------------
# Each: the format's corpus folder, the case's name (after `ref_`), the source and its files (the
# model first; `as` renames one), what it is, and how the reference's space maps to glTF's.

SPEC = {
    "off": "OFF (Geomview OFF(5) manual page)",
    "3ds": "3DS (Autodesk 3D Studio chunk format, as lib3ds documents it)",
    "md2": "Quake II MD2 (qfiles.h, version 8)",
    "md5": "Doom 3 MD5 (version 10)",

    "smd": "Valve SMD (Studiomdl Data version 1)",
    "ms3d": "MilkShape 3D (MS3D version 4)",
    "b3d": "Blitz3D (B3D version 1)",
    "dxf": "DXF (DXF Reference, 2013 edition)",
    "dae": "COLLADA 1.4.1",
    "x": "DirectX .x (X File Format Reference, DirectX 9 SDK)",
    "lwo": "LightWave object (LWO2 and LWOB file formats)",
    "ac": "AC3D (the AC3D file format)",
    "amf": "AMF (ISO/ASTM 52915)",
    "xgl": "XGL 1.0 (the XGL file format specification)",
    "irr": "Irrlicht mesh and scene formats (the Irrlicht Engine's XML formats)",
    "ogex": "OpenGEX 3.0 (the OpenGEX specification)",
    "ogre": "OGRE mesh and skeleton serializers (OgreMeshSerializerImpl, OgreSkeletonSerializer) and material scripts",
    "bvh": "BVH (Biovision Hierarchy motion capture)",
    "3mf": "3MF (3D Manufacturing Format core specification)",
    "vox": "MagicaVoxel .vox (ephtracy/voxel-model, MagicaVoxel-file-format-vox.txt)",
    "c3d": "C3D (biomechanics motion capture, c3d.org)",
}


AMF_VOLUMES = ("assimp 6.0.5 returns one of each mesh's two volumes (half the triangles the file lists), so only the "
               "bounds are compared")


def assimp(fmt, name, path, what, **more):
    """A case from assimp's test models, measured with assimp."""
    files = [path] if isinstance(path, str) else path
    return {"fmt": fmt, "name": "assimp_" + name, "source": "assimp", "files": files,
            "spec": f"{SPEC[fmt]}: a real file, {what}", **more}


def lib3mf(name, path, what):
    return {"fmt": "3mf", "name": "3mf_samples_" + name, "source": "3mf-samples", "files": [path], "tool": "lib3mf",
            # Meters make these models small: a tolerance to match.
            "tolerance": 1e-6,
            "spec": f"{SPEC['3mf']}: a real file, {what}"}


def pyvox(name, path, what):
    return {"fmt": "vox", "name": "voxel_model_" + name, "source": "voxel-model", "files": [path], "tool": "pyvox",
            "spec": f"{SPEC['vox']}: a real file, {what}"}


def c3d(name, what, **more):
    return {"fmt": "c3d", "name": "ezc3d_" + name, "tool": "ezc3d", "make": make_c3d, "file": name + ".c3d",
            "spec": f"{SPEC['c3d']}: a file ezc3d wrote, {what}", **more}


CASES = [
    assimp("off", "cube", "OFF/Cube.off", "a cube of quads"),
    assimp("off", "wuson", "OFF/Wuson.off", "a creature of 3732 triangles"),

    assimp("3ds", "test1", "3DS/test1.3ds", "nine boxes with three materials"),
    assimp("3ds", "cubes_with_alpha", "3DS/cubes_with_alpha.3DS", "cubes with transparent materials"),
    assimp("3ds", "fels", "3DS/fels.3ds", "a rock"),
    assimp("3ds", "cube_with_diffuse_texture", "3DS/cube_with_diffuse_texture.3DS", "a cube naming a texture that is not there"),
    # 3DS keys are frame numbers and the file has no frame rate: assimp leaves it unset, and
    # 30 a second is 3D Studio's own.
    assimp("3ds", "rotating_cube", "3DS/RotatingCube.3DS", "a cube turning through 182 rotation keys, saved on frame 180",
           ticks_per_second=30, static="first_key", pose_bounds=True, pose_tick=40, node_names=["Box01"],
           note="assimp multiplies each rotation key onto the wrong side of the turns before it: the file's mesh matrix, "
                "saved on frame 180, is where the keys put the cube only when each turns the ones before it. So the pose is "
                "compared on frame 40, while every key still turns about X"),
    assimp("3ds", "camera_roll", "3DS/CameraRollAnim.3ds", "a box and a camera with a roll track",
           ticks_per_second=30, animations=False),

    assimp("md2", "sydney", "MD2/sydney.md2", "a character of 198 frames, drawn in its first",
           terms="the Irrlicht Engine license, in sidney-source.txt beside it", extra=["MD2/sidney-source.txt"]),

    assimp("md5", "simple_cube", "MD5/SimpleCube.md5mesh", "a cube bound to two joints",
           bones="all", node_names="all"),



    # SMD is Z up (Valve's tools); assimp leaves the file's axes as they are.
    assimp("smd", "triangle", "SMD/triangle.smd", "one triangle on one bone", axes="Z_UP",
           animations=False, bones="all", joints=True),
    assimp("smd", "holy_grail", "SMD/holy_grailref.smd", "a reference mesh of 896 triangles on one bone", axes="Z_UP",
           animations=False, bones="all", joints=True),

    assimp("ms3d", "twospheres", "MS3D/twospheres.ms3d", "two spheres in two groups"),
    assimp("ms3d", "twospheres_withmats", "MS3D/twospheres_withmats.ms3d", "two spheres with materials"),

    assimp("b3d", "wuson", "B3D/WusonBlitz.b3d", "a creature of 3732 triangles"),

    assimp("dxf", "pink_egg", "DXF/PinkEggFromLW.dxf", "3DFACE entities LightWave exported"),
    # $INSUNITS is 1, inches, which assimp doesn't read.
    assimp("dxf", "issue_2229", "DXF/issue_2229.dxf", "polyface meshes in inches", scale=0.0254),

    assimp("dxf", "line_test", "DXF/lineTest.dxf", "two 3DFACE entities and a LINE", tool="ezdxf"),
    assimp("dxf", "points", "DXF/points.dxf", "two POINT entities and nothing else", tool="ezdxf"),

    assimp("dae", "cube_tristrips", "Collada/cube_tristrips.dae", "a cube of <tristrips>"),
    assimp("dae", "cube_empty_tags", "Collada/cube_emptyTags.dae", "a cube with empty elements"),
    assimp("dae", "cube_with_2uvs", "Collada/cube_with_2UVs.DAE", "a cube in inches with two UV sets", tolerance=1e-6),
    assimp("dae", "cube_utf16le", "Collada/cube_UTF16LE.dae", "a cube in UTF-16"),
    assimp("dae", "cube_xml_special_chars", "Collada/cube_xmlspecialchars.dae", "a cube with character references in its names"),
    assimp("dae", "concave_polygon", "Collada/ConcavePolygon.dae", "a concave polygon of 66 corners"),
    assimp("dae", "regr01", "Collada/regr01.dae", "six meshes sharing seven materials", tolerance=1e-6),
    assimp("dae", "sphere", "Collada/sphere.dae", "a sphere of polygons", tolerance=1e-6),
    assimp("dae", "teapot_instance_nodes", "Collada/teapot_instancenodes.DAE", "teapots placed by <instance_node>"),
    assimp("dae", "cinema4d", "Collada/Cinema4D.dae", "three objects Cinema 4D exported"),
    assimp("dae", "box_nested_animation", "Collada/box_nested_animation.dae", "a skinned box with animations nested in one another",
           bones="all", posed="all", pose_bounds=True),
    assimp("dae", "library_animation_clips", "Collada/library_animation_clips.dae", "a skinned mesh with two animation clips",
           bones="all", posed="all", spans=False, starts=[0.041666666666666664, 0.0],
           note="assimp 6.0.5 loses the first vertex's weight (the file binds it wholly to the first joint), so the skinned "
                "mesh isn't compared in a pose, only where the joints are. It also keeps the keys before a clip's start, "
                "which COLLADA leaves out of the clip, so the animations' spans aren't compared"),
    assimp("dae", "kwxport_vcolors", "Collada/kwxport_test_vcolors.dae", "a cube with vertex colors kW Xport exported",
           terms="MIT, kW Xport's notice in kwxport_test_vcolors.dae.source.txt beside it",
           extra=["Collada/kwxport_test_vcolors.dae.source.txt"]),

    assimp("x", "test_cube_text", "X/test_cube_text.x", "a cube in text"),
    assimp("x", "test_cube_binary", "X/test_cube_binary.x", "a cube in binary"),
    assimp("x", "test_cube_compressed", "X/test_cube_compressed.x", "a cube in compressed binary"),
    assimp("x", "test", "X/test.x", "a textured cube Maya exported"),
    assimp("x", "kwxport_cube", "X/kwxport_test_cubewithvcolors.x", "a cube with vertex colors kW Xport exported",
           terms="MIT, kW Xport's notice in kwxport_test_cubewithvcolors.source.txt beside it",
           extra=["X/kwxport_test_cubewithvcolors.source.txt"]),

    assimp("lwo", "lwo2_boxuv", "LWO/LWO2/boxuv.lwo", "an LWO2 box with a UV map"),
    assimp("lwo", "lwo2_hierarchy", "LWO/LWO2/hierarchy.lwo", "LWO2 layers in a hierarchy"),
    assimp("lwo", "lwo2_concave_polygon", "LWO/LWO2/concave_polygon.lwo", "an LWO2 concave polygon of 66 corners"),

    assimp("lwo", "lwo2_uvtest", "LWO/LWO2/uvtest.lwo", "an LWO2 plane of sixteen surfaces"),
    assimp("lwo", "lwo2_box_2uv", "LWO/LWO2/box_2uv_1unused.lwo", "an LWO2 box with two UV maps"),
    assimp("lwo", "lwo2_subdivision", "LWO/LWO2/Subdivision.lwo", "LWO2 subdivision patches, drawn as their cage"),
    assimp("lwo", "lwo2_transparency", "LWO/LWO2/transparency.lwo", "an LWO2 transparent sphere"),
    assimp("lwo", "lwob_sphere", "LWO/LWOB/sphere_with_mat_gloss_10pc.lwo", "an LWOB sphere with a glossy surface"),
    assimp("lwo", "lwob_concave_polygon", "LWO/LWOB/ConcavePolygon.lwo", "an LWOB concave polygon of 66 corners"),

    assimp("ac", "double_sided_face", "AC/doubleSidedFace.ac", "a two-sided quad", twice=True,
           expect={"materials": [{"double_sided": True}]}),

    # AMF is Z up, in millimeters unless its `unit` says otherwise; assimp leaves both alone.
    assimp("amf", "test1", "AMF/test1.amf", "a tetrahedron", axes="Z_UP", scale=0.001),
    assimp("amf", "test5", "AMF/test5.amf", "a tetrahedron with colors on its object, volume, triangles and vertices",
           axes="Z_UP", scale=0.001),

    assimp("amf", "test_with_mat", "AMF/test_with_mat.amf", "a pyramid of two volumes in inches", axes="Z_UP", scale=0.0254,
           skip=["triangles", "area", "positions"], note=AMF_VOLUMES),

    assimp("xgl", "cubes_with_alpha", "XGL/cubes_with_alpha.zgl", "five transparent cubes, compressed"),
    assimp("xgl", "sample_official", "XGL/sample_official.xgl", "the specification's sample", skip=["bounds", "positions"],
           note="assimp 6.0.5 leaves both objects where the mesh is, without their <TRANSFORM>'s <POSITION>, so only "
                "what moving them doesn't change is compared"),
    assimp("xgl", "sphere_gloss", "XGL/sphere_with_mat_gloss_10pc.zgl", "a glossy sphere, compressed"),
    assimp("xgl", "spider", "XGL/Spider_ascii.zgl", "a spider of 1312 triangles, compressed"),

    # Irrlicht is left handed; assimp returns an IRRMesh's coordinates as they are stored.
    assimp("irr", "cellar", "IRRMesh/cellar.irrmesh", "a mesh of two buffers", axes="MIRROR_Z"),
    assimp("irr", "spider", "IRRMesh/spider.irrmesh", "a mesh of four buffers", axes="MIRROR_Z"),

    # Both say `Metric (key = "up") {string {"z"}}`, which assimp doesn't apply.
    assimp("ogex", "example", "OpenGEX/Example.ogex", "the specification's example", axes="Z_UP"),
    assimp("ogex", "camera", "OpenGEX/camera.ogex", "a cube, a light and a camera", axes="Z_UP"),

    assimp("ogre", "the_thing", "Ogre/TheThing/Mesh.mesh.xml", "an XML mesh of 2656 triangles"),

    # BVH's frames are numbered from 0 and assimp counts its ticks in frames.
    assimp("bvh", "cmu_01_01", "BVH/01_01.bvh", "a CMU capture of 31 joints, cut to its first 24 frames",
           tool="assimp_motion", frames=24, bones="all", posed="all", node_names="all",
           terms="the CMU Graphics Lab Motion Capture Database's and its BVH conversion's terms, free to use, "
                 "in 01_nn.bvh.source.txt beside it",
           extra=["BVH/01_nn.bvh.source.txt"]),

    lib3mf("box", "examples/core/box.3mf", "a box"),
    lib3mf("cylinder", "examples/core/cylinder.3mf", "a cylinder"),
    lib3mf("multiple_cylinders", "examples/core/multiple_cylinders.3mf", "one cylinder in six build items"),
    lib3mf("torus", "examples/core/torus.3mf", "a torus"),
    lib3mf("sphere", "examples/core/sphere.3mf", "a sphere"),
    lib3mf("pyramid_vertexcolor", "examples/material/pyramid_vertexcolor.3mf", "a pyramid with a color at each vertex"),
    lib3mf("rhombicuboctahedron_color", "examples/material/rhombicuboctahedron_color.3mf",
           "a rhombicuboctahedron with a color on each face"),
    lib3mf("box_prod", "examples/production/box_prod.3mf", "a box in the production extension's own part"),

    pyvox("chr_knight", "vox/character/chr_knight.vox", "a knight of 398 voxels"),
    pyvox("ff3", "vox/procedure/ff3.vox", "a procedural shape of 529 voxels"),
    pyvox("snow", "vox/procedure/snow.vox", "a snowflake in the default palette"),
    pyvox("monu0", "vox/monument/monu0.vox", "a monument of 12717 voxels"),

    c3d("markers", "three markers over 20 frames, in millimeters", labels=["LASI", "RASI", "SACR"], frames=20,
        rate=100.0, units="mm"),
    c3d("missing_samples", "markers with samples missing", labels=["LASI", "RASI", "SACR", "LKNE"], frames=12,
        rate=50.0, units="mm", missing=[(1, 0), (1, 1), (2, 5), (3, 11)]),
    c3d("meters_and_analogs", "markers in meters beside an analog channel", labels=["HEAD", "C7"], frames=10,
        rate=120.0, units="m", scale=1.0, analogs=10),
]


def keep_frames(path, frames):
    """Cuts a BVH file to its first `frames` frames, so it fits the corpus."""
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    at = next(i for i, line in enumerate(lines) if line.startswith(b"Frames:"))
    lines[at] = b"Frames: %d" % frames + (b"\r" if lines[at].endswith(b"\r") else b"")
    with open(path, "wb") as f:
        f.write(b"\n".join(lines[:at + 2 + frames]) + b"\n")


def finish(c, d, model, measure, expect, name):
    """Measures a case's model and writes its expect.json and what it names."""
    files = {}
    measure(c, model, expect, files)
    expect.update(c.get("expect", {}))
    for fname, data in files.items():
        with open(os.path.join(d, fname), "wb") as f:
            f.write(data)
    with open(os.path.join(d, "expect.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(expect, f, indent=2)
        f.write("\n")
    return name


def write(c):
    name = "ref_" + c["name"]
    d = os.path.join(ROOT, c["fmt"], name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    measure, tool = MEASURES[c.get("tool", "assimp")]
    if "make" in c:
        # Written here by another project's library, and read back with it.
        model = os.path.join(d, c["file"])
        c["make"](c, model)
        expect = {"spec": c["spec"],
                  "source": f"written with {tool} by tools/corpus/thirdparty.py; expectations are what it reads back"}
        return finish(c, d, model, measure, expect, name)
    source = SOURCES[c["source"]]
    model = None
    for rel in c["files"] + c.get("extra", []):
        rel, to = rel if isinstance(rel, tuple) else (rel, rel.rsplit("/", 1)[1])
        os.makedirs(os.path.dirname(os.path.join(d, to)), exist_ok=True)
        shutil.copyfile(fetch(c["source"], rel), os.path.join(d, to))
        model = model or os.path.join(d, to)
    if "frames" in c:
        keep_frames(model, c["frames"])
    license = c.get("terms", f"{source['license']}, see ../LICENSE-{c['source']}")
    expect = {
        "spec": c["spec"],
        "source": f"{source['repo']} at {source['commit'][:12]}, {source.get('root', '')}{c['files'][0]}"
                  f" ({license}); expectations measured with {tool}" + (". " + c["note"] if c.get("note") else ""),
    }
    if len(c["files"]) > 1:
        expect["file"] = os.path.basename(model)
    if "tolerance" in c:
        expect["tolerance"] = c["tolerance"]
    finish(c, d, model, measure, expect, name)
    license_path = os.path.join(ROOT, c["fmt"], "LICENSE-" + c["source"])
    if not os.path.isfile(license_path):
        with open(fetch(c["source"], source["license_file"], root=False), encoding="utf-8") as f:
            text = f.read()
        # Only the repository's own license: what follows a line of asterisks is about files
        # that aren't copied here (assimp's `models-nonbsd`, its bundled libraries).
        text = text.split("\n*****")[0].rstrip() + "\n"
        with open(license_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return name


def main():
    only = set(sys.argv[1:])
    for c in CASES:
        if not only or c["fmt"] in only:
            print(c["fmt"], write(c))


if __name__ == "__main__":
    main()
