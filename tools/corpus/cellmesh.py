"""What a finite-element mesh draws, worked out with VTK: shared by msh.py and medit.py.

A generator reads a file with something other than Polysmelt (Gmsh, meshio, or what it wrote by
hand) and hands the result here as groups of elements, each a mesh of its own. An element draws
by its corner nodes as VTK's linear cell of the same shape: a volume element's faces are the ones
vtkCell.GetFace gives, and a group's boundary is the faces only one of its elements has;
`check_surface` holds that count against vtkDataSetSurfaceFilter's.

`measure` returns the expectations for expect.json and the reference for reference.json, which
tests/fem.rs compares with what Polysmelt reads. Each mesh's reference holds a 64-bit FNV-1a hash
of its soup: every triangle, line and point drawn, as the positions of its corners with each
attribute's values there, so it's the same however vertices are shared. Where nothing splits
vertices (no element data), it also holds a hash of the points drawn, in the file's order.
"""

import numpy as np
import vtk

from common import r

# A shape's corner nodes and the VTK cell they draw as.
SHAPES = {
    "vertex": (1, vtk.VTK_VERTEX),
    "line": (2, vtk.VTK_LINE),
    "triangle": (3, vtk.VTK_TRIANGLE),
    "quad": (4, vtk.VTK_QUAD),
    "tetra": (4, vtk.VTK_TETRA),
    "hexahedron": (8, vtk.VTK_HEXAHEDRON),
    "wedge": (6, vtk.VTK_WEDGE),
    "pyramid": (5, vtk.VTK_PYRAMID),
}


def shape_of(meshio_type):
    """The shape a meshio cell type's corners make: `tetra10` is a `tetra`."""
    return meshio_type.rstrip("0123456789")


def fnv(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def f32(values):
    return np.asarray(values, dtype="<f4").tobytes()


def attribute_name(name):
    return "_" + ("".join(c.upper() if c.isascii() and c.isalnum() else "_" for c in name.strip()) or "VALUE")


def custom(name, values):
    """An array as the custom attributes it becomes: up to four components are one attribute, up
    to nine an attribute each."""
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(len(v), -1)
    v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
    key = attribute_name(name)
    if v.shape[1] <= 4:
        return {key: v}
    assert v.shape[1] <= 9, name
    return {"%s_%d" % (key, k): v[:, k:k + 1] for k in range(v.shape[1])}


class Group:
    """Elements drawn as one mesh. `elements` are (shape, corner point indices); `cell_data` maps
    a name to a value for each element, and `corner_data` to a value for each element's corners."""

    def __init__(self, name=None, extras=None):
        self.name = name
        self.extras = extras
        self.elements = []
        self.cell_data = {}
        self.corner_data = {}


def grid(points, elements):
    g = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetDataType(vtk.VTK_FLOAT)
    for p in points:
        pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    g.SetPoints(pts)
    for shape, ids in elements:
        corners, kind = SHAPES[shape]
        g.InsertNextCell(kind, corners, [int(i) for i in ids[:corners]])
    return g


def drawn(ds):
    """What a grid draws: triangles, lines and points as point ids, each with its cell, and how
    many faces its boundary has."""
    tris, lines, verts = [], [], []
    faces = {}
    for c in range(ds.GetNumberOfCells()):
        cell = ds.GetCell(c)
        ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        if cell.GetCellDimension() == 0:
            verts.append(((ids[0],), c))
        elif cell.GetCellDimension() == 1:
            lines.append(((ids[0], ids[1]), c))
        elif cell.GetCellDimension() == 2:
            for k in range(1, len(ids) - 1):
                tris.append(((ids[0], ids[k], ids[k + 1]), c))
        else:
            for j in range(cell.GetNumberOfFaces()):
                face = cell.GetFace(j)
                face = [face.GetPointId(i) for i in range(face.GetNumberOfPoints())]
                faces.setdefault(frozenset(face), []).append((face, c))
    boundary = [found[0] for found in faces.values() if len(found) == 1]
    for face, c in boundary:
        for k in range(1, len(face) - 1):
            tris.append(((face[0], face[k], face[k + 1]), c))
    return tris, lines, verts, len(boundary)


def check_surface(ds, tris):
    """vtkDataSetSurfaceFilter draws as many triangles."""
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(ds)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surface.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    n = tri.GetOutput().GetNumberOfPolys()
    assert n == len(tris), (n, len(tris))
    return n


def measure(points, groups, point_data=None, normals=None, ordered=True):
    """The expectations and the reference for a mesh of `points` (in the file's order) drawn as
    `groups`. `point_data` maps names to a value for each point; `normals` is a normal for each
    point, or None. `ordered` says the points' order is the file's, so the vertices' is known."""
    points = np.asarray(points, dtype=np.float32)
    point_attrs = {}
    for name, values in (point_data or {}).items():
        point_attrs.update(custom(name, values))
    expect = {"triangles": 0, "lines": 0, "points": 0}
    meshes = []
    used_all = set()
    area = 0.0
    names = set(point_attrs)
    vertices = 0
    exact_vertices = True
    for group in groups:
        ds = grid(points, group.elements)
        tris, lines, verts, boundary = drawn(ds)
        surface = check_surface(ds, tris)
        cell_attrs, corner_attrs = {}, {}
        for name, values in group.cell_data.items():
            cell_attrs.update(custom(name, values))
        for name, values in group.corner_data.items():
            # A value for each corner of each element: (elements, corners, components), ragged.
            comps = np.asarray(values[0], dtype=np.float64).reshape(len(values[0]), -1).shape[1]
            keys = list(custom(name, np.zeros((1, comps))))
            per = [custom(name, v) for v in values]
            for key in keys:
                corner_attrs[key] = [p[key] for p in per]
        split = bool(cell_attrs or corner_attrs)
        soup_names = sorted(set(point_attrs) | set(cell_attrs) | set(corner_attrs))
        names |= set(soup_names)
        records = []
        for items in (tris, lines, verts):
            for ids, c in items:
                corners = []
                element = [int(i) for i in group.elements[c][1]]
                for i in ids:
                    record = f32(points[i])
                    for key in soup_names:
                        if key in corner_attrs:
                            record += f32(corner_attrs[key][c][element.index(i)])
                        elif key in cell_attrs:
                            record += f32(cell_attrs[key][c])
                        else:
                            record += f32(point_attrs[key][i])
                    corners.append(record)
                k = corners.index(min(corners))
                records.append(bytes([len(ids)]) + b"".join(corners[k:] + corners[:k]))
        used = sorted({i for t, _ in tris for i in t} | {i for l, _ in lines for i in l} | {i for v, _ in verts for i in v})
        used_all |= set(used)
        for t, _ in tris:
            a, b, c = (points[i].astype(np.float64) for i in t)
            area += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        expect["triangles"] += len(tris)
        expect["lines"] += len(lines)
        expect["points"] += len(verts)
        ref = {"name": group.name, "extras": group.extras, "points": None, "vertices": None,
               "soup": fnv(b"".join(sorted(records))), "soup_attributes": soup_names, "approx": {},
               "triangles": len(tris), "lines": len(lines), "verts": len(verts),
               "boundary_faces": boundary, "surface_filter_triangles": surface}
        if not split and ordered:
            # Nothing splits a vertex: the vertices are the points used, in the file's order.
            ref["points"] = fnv(f32(points[used]))
            ref["vertices"] = len(used)
            vertices += len(used)
            if normals is not None:
                n = np.asarray(normals, dtype=np.float64)[used]
                length = np.linalg.norm(n, axis=1, keepdims=True)
                n = np.where(length > 1e-12, n / np.maximum(length, 1e-30), [0.0, 1.0, 0.0])
                ref["approx"]["NORMAL"] = {"width": 3, "values": [r(float(v), 7) for v in n.flat]}
        else:
            exact_vertices = False
        if used:
            meshes.append(ref)
    if normals is not None:
        names.add("NORMAL")
    for key in ("lines", "points"):
        if not expect[key]:
            del expect[key]
    if used_all:
        drawn_points = points[sorted(used_all)]
        expect["bounds"] = {"min": [float(v) for v in drawn_points.min(axis=0)],
                            "max": [float(v) for v in drawn_points.max(axis=0)]}
        if expect["triangles"]:
            expect["area"] = r(area)
        expect["facts"] = {"triangles": expect["triangles"]}
        # Without normals in the file, its triangles get the default ones, which add vertices where an edge is
        # hard: copies of the file's points, after them.
        if exact_vertices and (normals is not None or not expect["triangles"]):
            expect["vertices"] = vertices
        document = {"nodes": len(meshes)}
        if names:
            document["attributes"] = sorted(names)
        node_names = [m["name"] for m in meshes if m["name"]]
        if node_names:
            document["node_names"] = node_names
        expect["document"] = document
        expect["colors"] = False
        expect["uvs"] = False
    return expect, {"meshes": meshes}
