"""Shared helpers for the conformance corpus generators.

Each generator writes cases under tests/corpus/<format>/<case>/: the model file, whatever it
refers to, and an expect.json. Expectations come from what the generator wrote (the geometry
it chose, the colors it set), never from reading the result back with Polysmelt, so a case
checks the reader against the standard rather than against itself.

Run from the repository root:  python tools/corpus/<format>.py

With POLYSMELT_CORPUS set, the cases are written under that folder instead of tests/corpus:
tools/corpus/check.py does so to compare what the generators write with what is checked in.
"""

import json
import math
import os
import shutil
import struct
import zlib

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.path.normpath(os.environ.get("POLYSMELT_CORPUS") or os.path.join(REPO, "tests", "corpus"))


def fresh(fmt, keep_prefix=None):
    """Empties tests/corpus/<fmt>, except case folders starting with `keep_prefix` (made by
    other tools), the `ref_*` folders thirdparty.py writes, and license files."""
    d = os.path.join(ROOT, fmt)
    if os.path.isdir(d):
        for name in os.listdir(d):
            kept = name.startswith("ref_") or (keep_prefix and name.startswith(keep_prefix))
            if not kept and os.path.isdir(os.path.join(d, name)):
                shutil.rmtree(os.path.join(d, name))
    os.makedirs(d, exist_ok=True)
    return d


# In an expectation's "json", stands for the model file's name without its extension: what a file that is one
# clip (BVH, AMC, C3D, TRC, HTR, an .md5anim, SMD) names its animation, as the file itself gives it none.
CLIP = "\0the model file's stem"


def case(fmt, name, files, expect):
    """Writes one case: `files` maps file names to bytes or str (the model first); `expect` is expect.json."""
    stem = os.path.splitext(os.path.basename(next(iter(files))))[0] if files else name
    checks = expect.get("json")
    if isinstance(checks, dict):
        for key, value in checks.items():
            if value == CLIP:
                checks[key] = stem
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


# The usual name of a unit by its length in meters: what `asset.extras.source.unit` is, whatever the file calls it.
UNIT_NAMES = {1e-9: "nanometer", 1e-6: "micrometer", 0.001: "millimeter", 0.01: "centimeter", 0.1: "decimeter",
              1.0: "meter", 1000.0: "kilometer", 0.0254: "inch", 0.3048: "foot", 0.9144: "yard", 1609.344: "mile"}


def unit_source(meters, written=None, prefix="/asset/extras/source/"):
    """What `asset.extras.source` says of a unit `meters` long (None when its length is unknown) that the file calls
    `written`: `unit`, its usual name (absent when the length is unknown or no common unit's), and `unitAsWritten`,
    the file's own word when that isn't the usual name. Keyed by JSON pointer under `prefix`, or bare when `prefix`
    is empty; None means the key must be absent."""
    usual = UNIT_NAMES.get(meters)
    out = {prefix + "unit": usual}
    out[prefix + "unitAsWritten"] = written if written is not None and written != usual else None
    return out


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def r(x, n=6):
    return round(x, n)


# The classic material (Phong, Blinn, Lambert) as glTF's, by the rule every reader shares. Worked out here from
# the rule itself, apart from the readers' code.

# The material of faces their file gives none, and its roughness: a material that says nothing of its highlight.
GRAY = [0.72, 0.72, 0.74, 1.0]
ROUGHNESS = 0.85
DEFAULT_MATERIAL = {"name": "default", "color": GRAY, "roughness": ROUGHNESS, "metallic": 0.0}
# The default material under vertex colors or a texture, which show as they are.
DEFAULT_WHITE = {"name": "default", "color": [1.0, 1.0, 1.0, 1.0], "roughness": ROUGHNESS, "metallic": 0.0}
# KHR_materials_specular for no highlight at all.
NO_HIGHLIGHT = {"specularFactor": 0.0}


def phong_roughness(n):
    """glTF roughness for a Blinn-Phong exponent `n`: a lobe of exponent n is closest to a GGX lobe of
    alpha^2 = 2 / (n + 2), and roughness is sqrt(alpha)."""
    return (2.0 / (max(n, 0.0) + 2.0)) ** 0.25


def phong_specular(color, strength=1.0, linear=False):
    """KHR_materials_specular for a classic specular `color` (sRGB unless `linear`) scaled by `strength`: `None`
    when it is glTF's default (full white), else its members. The color times the strength (in the values the
    file stores, as the classic model multiplies them), made linear, is how much of glTF's highlight is left: its
    largest channel is the factor, and the color over it the color factor. Black is no highlight."""
    scaled = [max(c * min(max(strength, 0.0), 1.0), 0.0) for c in color]
    lin = [c if linear else srgb_to_linear(min(c, 1.0)) for c in scaled]
    top = max(lin)
    if top <= 0.0:
        return dict(NO_HIGHLIGHT)
    factor, tint = min(top, 1.0), [c / top for c in lin]
    if factor >= 0.998 and min(tint) >= 0.998:
        return None
    out = {}
    if factor < 1.0:
        out["specularFactor"] = r(factor)
    if tint != [1.0, 1.0, 1.0]:
        out["specularColorFactor"] = [r(c) for c in tint]
    return out


def phong(specular, exponent, strength=1.0, linear=False):
    """(roughness, KHR_materials_specular or None) for a classic highlight: a specular color (`None` for a
    Lambert material), its strength and its Blinn-Phong exponent. No highlight is fully rough."""
    spec = dict(NO_HIGHLIGHT) if specular is None else phong_specular(specular, strength, linear)
    if spec == NO_HIGHLIGHT:
        return 1.0, spec
    return r(phong_roughness(exponent)), spec


# The one classic material written in every format that can hold it (the `same_phong` case of each generator, which
# tests/materials.rs reads back and holds against each other): diffuse, a gray specular color, a Blinn-Phong
# exponent, opaque.
SAME_PHONG = {"diffuse": (0.8, 0.2, 0.1), "specular": (0.5, 0.5, 0.5), "exponent": 20.0}


def same_phong_expect(index=0, name=None):
    """The expectations of the `same_phong` material, as material `index` of the document."""
    roughness, spec = phong(SAME_PHONG["specular"], SAME_PHONG["exponent"])
    m = {"color": [r(srgb_to_linear(c)) for c in SAME_PHONG["diffuse"]] + [1.0], "roughness": roughness,
         "metallic": 0.0, "alpha_mode": "OPAQUE"}
    if name is not None:
        m["name"] = name
    return {"materials": [m], "json": {f"/materials/{index}/extensions/KHR_materials_specular": spec}}


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
