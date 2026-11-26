"""E57 conformance cases: ASTM E2807, the E57 3D imaging data file format.

E57's writers are big libraries, so the cases are real files: small ones from the e57 crate's
test data (MIT, see LICENSE-e57 beside the cases), made with libE57Format. They are kept in
their case folders (or read from the folder given as the first argument). Their expectations
come from reading them here, by the standard: the file is 1024-byte pages, each ending in a
CRC; the XML section describes each scan (data3D), its pose and its points' prototype; the
points are a compressed vector whose data packets hold a bytestream per field, floats as they
are and integers bit-packed from their minimum. Spherical coordinates turn Cartesian as
x = r cos(elevation) cos(azimuth), y = r cos(elevation) sin(azimuth), z = r sin(elevation).
Positions with an invalid state other than 0 are left out. E57 is Z up, in meters, so glTF's
position is (x, z, -y).
"""

import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

# The shared modules are beside this script, which `python -I` leaves off the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, case, fresh, r, srgb_to_linear  # noqa: E402

FMT = "e57"
SPEC = "E57 (ASTM E2807)"
SRC = "e57 crate test data (testdata/), MIT license, see ../LICENSE-e57"
REAL = {"tinyCartesianFloatRgb.e57": "cartesian_float_rgb", "tiny_spherical.e57": "spherical"}
NS = "{http://www.astm.org/COMMIT/E57/2010-e57-v1.0}"


def keep_real():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    out = {}
    for name, folder in [*REAL.items(), ("LICENSE-e57", None)]:
        p = os.path.join(src or os.path.join(ROOT, FMT, *([folder] if folder else [])), name)
        with open(p, "rb") as f:
            out[name] = f.read()
    return out


real = keep_real()
if os.path.exists(os.path.join(ROOT, FMT, "LICENSE-e57")):
    os.remove(os.path.join(ROOT, FMT, "LICENSE-e57"))
fresh(FMT)
with open(os.path.join(ROOT, FMT, "LICENSE-e57"), "wb") as f:
    f.write(real["LICENSE-e57"])


def read_e57(b):
    """Each scan's points: a list of dicts of field values, and the scan's XML element."""
    page = struct.unpack_from("<Q", b, 40)[0]
    logical = b"".join(b[p:p + page - 4] for p in range(0, len(b), page))

    def at(physical):
        return physical - 4 * (physical // page)

    xml_off, xml_len = struct.unpack_from("<QQ", b, 24)
    root = ET.fromstring(logical[at(xml_off):at(xml_off) + xml_len])
    scans = []
    for scan in root.find(NS + "data3D"):
        pts = scan.find(NS + "points")
        count = int(pts.get("recordCount"))
        fields = list(pts.find(NS + "prototype"))
        start = at(int(pts.get("fileOffset")))
        assert logical[start] == 1, "not a compressed vector section"
        length, data = struct.unpack_from("<QQ", logical, start + 8)
        streams = [bytearray() for _ in fields]
        p = at(data)
        while p < start + length:
            kind, _, size = struct.unpack_from("<BBH", logical, p)
            size += 1
            if kind == 1:
                n, = struct.unpack_from("<H", logical, p + 4)
                lens = struct.unpack_from(f"<{n}H", logical, p + 6)
                q = p + 6 + 2 * n
                for k, m in enumerate(lens):
                    streams[k] += logical[q:q + m]
                    q += m
            p += size
        cols = []
        for f, s in zip(fields, streams):
            kind = f.get("type")
            if kind == "Float":
                single = f.get("precision") == "single"
                cols.append(struct.unpack_from(f"<{count}{'f' if single else 'd'}", s))
            else:
                lo, hi = int(f.get("minimum", 0)), int(f.get("maximum", 0))
                bits = (hi - lo).bit_length()
                v = int.from_bytes(s, "little")
                scale = float(f.get("scale", 1)) if kind == "ScaledInteger" else 1
                cols.append([lo + ((v >> (bits * i)) & ((1 << bits) - 1)) * scale if bits else lo for i in range(count)])
        names = [f.tag[len(NS):] for f in fields]
        scans.append(([dict(zip(names, row)) for row in zip(*cols)], scan))
    return scans


def position(p):
    if "cartesianX" in p:
        if p.get("cartesianInvalidState", 0) != 0:
            return None
        return [p["cartesianX"], p["cartesianY"], p["cartesianZ"]]
    if p.get("sphericalInvalidState", 0) != 0:
        return None
    rr, az, el = p["sphericalRange"], p["sphericalAzimuth"], p["sphericalElevation"]
    return [rr * math.cos(el) * math.cos(az), rr * math.cos(el) * math.sin(az), rr * math.sin(el)]


def y_up(p):
    return [r(p[0]), r(p[2]), r(-p[1])]


for name, folder in REAL.items():
    (pts, scan), = read_e57(real[name])
    assert scan.find(NS + "pose") is None, "the cases' scans have no pose"
    placed = [q for q in map(position, pts) if q is not None]
    g = [y_up(q) for q in placed]
    expect = {"points": len(placed), "triangles": 0,
              "bounds": {"min": [min(q[k] for q in g) for k in range(3)], "max": [max(q[k] for q in g) for k in range(3)]},
              "positions_include": [g[0], g[len(g) // 2], g[-1]], "colors": "colorRed" in pts[0]}
    if "colorRed" in pts[0]:
        pick = [pts[0], pts[len(pts) // 2], pts[-1]]
        expect["colors_include"] = [[r(srgb_to_linear(p[c] / 255)) for c in ("colorRed", "colorGreen", "colorBlue")]
                                    + [1.0] for p in pick]
    kind = "Cartesian single-precision floats and 8-bit colors" if "cartesianX" in pts[0] else \
        "spherical doubles with an invalid state"
    case(FMT, folder, {name: real[name]},
         {"spec": f"{SPEC}: a real file, one scan, {kind}", "source": SRC, "tolerance": 2e-4, **expect})

case(FMT, "not_e57", {"scan.e57": b"ASTM-E58" + bytes(1016)},
     {"spec": f"{SPEC}: a file without the ASTM-E57 signature", "source": "generated by tools/corpus/e57.py",
      "error": "e57", "error_kind": "Corrupt"})
