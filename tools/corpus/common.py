"""Shared helpers for the conformance corpus generators.

Each generator writes cases under tests/corpus/<format>/<case>/: the model file, whatever it
refers to, and an expect.json. Expectations come from what the generator wrote (the geometry
it chose, the colors it set), never from reading the result back with Polysmelt, so a case
checks the reader against the standard rather than against itself.

Run from the repository root:  python tools/corpus/<format>.py
"""

import json
import math
import os
import shutil
import struct
import zlib

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests", "corpus"))


def fresh(fmt, keep_prefix=None):
    """Empties tests/corpus/<fmt>, except case folders starting with `keep_prefix` (made by
    other tools)."""
    d = os.path.join(ROOT, fmt)
    if os.path.isdir(d):
        for name in os.listdir(d):
            if not (keep_prefix and name.startswith(keep_prefix)):
                shutil.rmtree(os.path.join(d, name))
    os.makedirs(d, exist_ok=True)
    return d


def case(fmt, name, files, expect):
    """Writes one case: `files` maps file names to bytes or str; `expect` is expect.json."""
    d = os.path.join(ROOT, fmt, name)
    os.makedirs(d, exist_ok=True)
    for fname, data in files.items():
        p = os.path.join(d, fname)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(p, "wb") as f:
            f.write(data)
    assert "spec" in expect and "source" in expect, name
    with open(os.path.join(d, "expect.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(expect, f, indent=2)
        f.write("\n")


def png(rgb, size=2):
    """A solid-color RGB PNG."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind, data):
        c = kind + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def webp(rgb, size=2):
    """A solid-color lossless WebP (needs Pillow)."""
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (size, size), tuple(rgb)).save(out, "WEBP", lossless=True)
    return out.getvalue()


def pil_image(rgb, fmt, size=2, mode="RGB", **options):
    """A solid-color image in a format Pillow writes (TGA, BMP, GIF, TIFF, DDS), with its save options."""
    import io

    from PIL import Image
    out = io.BytesIO()
    image = Image.new("RGB", (size, size), tuple(rgb))
    image = image.convert(mode) if mode != "RGB" else image
    image.save(out, fmt, **options)
    return out.getvalue()


def psd(rgb, size=2):
    """A solid-color RGB Photoshop document: its merged image as raw planes, no layers."""
    head = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, size, size, 8, 3)
    sections = struct.pack(">III", 0, 0, 0)
    planes = b"".join(bytes([c]) * (size * size) for c in rgb)
    return head + sections + struct.pack(">H", 0) + planes


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def r(x, n=6):
    return round(x, n)


AXES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

# A unit cube, [0,1]^3: 8 corners and 6 counter-clockwise quads (seen from outside), each with
# its outward normal.
CUBE_V = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_QUADS = [
    ((0, 2, 3, 1), (0, 0, -1)),
    ((4, 5, 7, 6), (0, 0, 1)),
    ((0, 1, 5, 4), (0, -1, 0)),
    ((2, 6, 7, 3), (0, 1, 0)),
    ((0, 4, 6, 2), (-1, 0, 0)),
    ((1, 3, 7, 5), (1, 0, 0)),
]


def check_cube():
    """The quads really wind counter-clockwise around their normals."""
    for q, n in CUBE_QUADS:
        a, b, c = (CUBE_V[i] for i in q[:3])
        u = [b[k] - a[k] for k in range(3)]
        v = [c[k] - a[k] for k in range(3)]
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        assert sum(cross[k] * n[k] for k in range(3)) > 0, q


check_cube()
CUBE_BOUNDS = {"min": [0, 0, 0], "max": [1, 1, 1]}


def polygon_area(pts):
    """Area of a planar polygon in 3D (Newell's method)."""
    nx = ny = nz = 0.0
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
