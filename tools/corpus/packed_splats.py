"""Writers for PlayCanvas's two packed gaussian splat containers, for the generators that use
them (`sog.py`, `ply.py`): SOG (a meta.json and lossless WebP images, versions 1 to 3) and
SuperSplat's compressed PLY. Standard library only.

SOG follows its specification (developer.playcanvas.com, "The SOG Format", versions 3 and 2);
version 1 and the compressed PLY, which have none, follow splat-transform's reader and writer
(read-sog-v1.ts, compressed-chunk.ts, write-compressed-ply.ts).

A splat is a dict in the file's own axes (right, down, forward):
  pos (x, y, z), rot (w, x, y, z), scale (three logarithms), dc (three coefficients),
  opacity (0 to 1, or to 1.5625 in SOG version 3), sh (a list of (r, g, b), a coefficient each)

Every writer also returns the splats as the file holds them: the same dicts with each value
replaced by what its quantized form decodes to, by the format's own rule. A case's expectations
come from those.
"""

import io
import json
import math
import struct
import zipfile

SH_C0 = 0.28209479177387814
COEFFS = [3, 8, 15, 24]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


# --- lossless WebP ------------------------------------------------------------------------------

REVERSED = [int(f"{b:08b}"[::-1], 2) for b in range(256)]


def webp(width, height, pixels):
    """A lossless WebP (VP8L) of `pixels`, (r, g, b, a) in row order: no transform, no color cache
    and no backward reference, each byte under a prefix code of 8 bits."""
    assert len(pixels) == width * height and 0 < width <= 16384 and 0 < height <= 16384
    acc, n = 0, 0

    def put(value, bits):
        nonlocal acc, n
        acc |= value << n
        n += bits

    put(0x2F, 8)
    put(width - 1, 14)
    put(height - 1, 14)
    put(1 if any(p[3] != 255 for p in pixels) else 0, 1)
    put(0, 3)  # version
    put(0, 1)  # no transform
    put(0, 1)  # no color cache
    put(0, 1)  # one group of prefix codes
    for symbols in (280, 256, 256, 256):  # green (with the length codes), red, blue, alpha
        put(0, 1)  # a normal code: its lengths, themselves coded
        # The code length code: 1 bit each for lengths 0 and 8, listed in the order
        # 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8.
        put(12 - 4, 4)
        for k in range(12):
            put(1 if k in (2, 11) else 0, 3)
        put(0, 1)  # a length for every symbol
        for s in range(symbols):
            put(1 if s < 256 else 0, 1)  # the byte values take 8 bits, the rest none
    for bit in (1, 0, 0, 0):  # distances: a simple code of the one symbol 0
        put(bit, 1)
    # A code's bits go most significant first.
    for r, g, b, a in pixels:
        put(REVERSED[g] | REVERSED[r] << 8 | REVERSED[b] << 16 | REVERSED[a] << 24, 32)
    data = acc.to_bytes((n + 7) // 8, "little")
    pad = b"\0" * (len(data) & 1)
    return b"RIFF" + struct.pack("<I", 12 + len(data) + len(pad)) + b"WEBPVP8L" + struct.pack("<I", len(data)) + data + pad


def layout(n):
    """An image's width and height for `n` splats, multiples of four as splat-transform makes them."""
    width = max(4, math.ceil(math.sqrt(n) / 4) * 4)
    return width, max(4, math.ceil(n / width / 4) * 4)


def image(pixels, size=None, fill=(0, 0, 0, 255)):
    width, height = size or layout(len(pixels))
    return webp(width, height, list(pixels) + [fill] * (width * height - len(pixels)))


# --- SOG ----------------------------------------------------------------------------------------

def nearest(codebook, v):
    return min(range(len(codebook)), key=lambda k: abs(codebook[k] - v))


def spread(values, steps):
    """`values` as steps from their least to their greatest: (min, max, codes)."""
    lo, hi = min(values), max(values)
    return lo, hi, [round((v - lo) / (hi - lo) * steps) if hi > lo else 0 for v in values]


def linear(lo, hi, steps=256):
    return [f32(lo + (hi - lo) * k / (steps - 1)) for k in range(steps)]


def smallest_three(q):
    """Versions 1 and 2: three components in (w, x, y, z) order as bytes, and 252 plus the index
    of the largest, left out. Returns the pixel and the rotation it decodes to."""
    ln = math.sqrt(sum(c * c for c in q))
    q = [c / ln for c in q]
    largest = max(range(4), key=lambda k: abs(q[k]))
    if q[largest] < 0:
        q = [-c for c in q]
    rest = [round((q[k] * math.sqrt(2) * 0.5 + 0.5) * 255) for k in range(4) if k != largest]
    back = [(c / 255 - 0.5) * 2 / math.sqrt(2) for c in rest]
    back.insert(largest, math.sqrt(max(0.0, 1 - sum(c * c for c in back))))
    return (*rest, 252 + largest), back


def exp_map(q):
    """Version 3: the rotation's exponential map, a byte a component. Returns the pixel and the
    rotation it decodes to."""
    ln = math.sqrt(sum(c * c for c in q))
    w, x, y, z = [c / ln * (-1 if q[0] < 0 else 1) for c in q]
    v = math.sqrt(x * x + y * y + z * z)
    k = math.atan2(v, w) / v / (math.pi / 2) if v > 0 else 2 / math.pi
    code = [max(0, min(255, round(c * k * 127 + 128))) for c in (x, y, z)]
    p = [(c - 128) / 127 for c in code]
    rr = math.sqrt(sum(c * c for c in p))
    theta = math.pi / 2 * rr
    kk = math.sin(theta) / rr if rr > 0 else 0.0
    return (*code, 255), [math.cos(theta), p[0] * kk, p[1] * kk, p[2] * kk]


BLIND_BYTES = {"u8": 1, "u16": 2, "u24": 3, "u32": 4, "f16": 2, "f32": 4, "unorm8": 1, "unorm16": 2}


def sog(splats, version=2, bands=0, palette=None, labels=None, blind=None, meta_extra=None,
        scale_codebook=None, dc_codebook=None, sh_codebook=None, state_bands=True):
    """The files of a SOG and the splats as they decode: ({name: bytes}, meta, [splat]).

    `palette` is the spherical-harmonic palette, each entry a list of (r, g, b) coefficients, and
    `labels` each splat's entry; by default every splat's own `sh` is an entry. `blind` is a list
    of textures, each a list of (field, [value a splat]) with field the meta's description.
    """
    n = len(splats)
    held = [dict(s) for s in splats]
    files = {}
    meta = {} if version == 1 else {"version": version, "asset": {"generator": "tools/corpus/sog.py"}, "count": n}

    # Positions: 16 bits an axis of sign(x) ln(|x| + 1) between its bounds.
    logs = [[math.copysign(math.log(abs(c) + 1), c) for c in s["pos"]] for s in splats]
    axes = [spread([p[k] for p in logs], 65535) for k in range(3)] if n else [(0, 0, [])] * 3
    mins, maxs = [a[0] for a in axes], [a[1] for a in axes]
    files["means_l.webp"] = image([tuple(axes[k][2][i] & 255 for k in range(3)) + (255,) for i in range(n)])
    files["means_u.webp"] = image([tuple(axes[k][2][i] >> 8 for k in range(3)) + (255,) for i in range(n)])
    for i, s in enumerate(held):
        back = [mins[k] + (maxs[k] - mins[k]) * (axes[k][2][i] / 65535) for k in range(3)]
        s["pos"] = [math.copysign(math.exp(abs(v)) - 1, v) for v in back]
    meta["means"] = {"mins": mins, "maxs": maxs, "files": ["means_l.webp", "means_u.webp"]}
    if version == 1:
        meta["means"]["shape"] = [n, 3]

    # Rotations.
    pixels = []
    for s in held:
        pixel, s["rot"] = exp_map(s["rot"]) if version == 3 else smallest_three(s["rot"])
        pixels.append(pixel)
    files["quats.webp"] = image(pixels)
    meta["quats"] = {"files": ["quats.webp"]}

    # Scales and base color: bytes between bounds in version 1, codebook indices after. Opacity
    # is a logit between bounds in version 1, a byte in version 2 and a square root's in 3.
    if version == 1:
        axes = [spread([s["scale"][k] for s in splats], 255) for k in range(3)]
        files["scales.webp"] = image([tuple(axes[k][2][i] for k in range(3)) + (255,) for i in range(n)])
        for i, s in enumerate(held):
            s["scale"] = [axes[k][0] + (axes[k][1] - axes[k][0]) * (axes[k][2][i] / 255) for k in range(3)]
        meta["scales"] = {"mins": [a[0] for a in axes], "maxs": [a[1] for a in axes], "files": ["scales.webp"]}
        logit = [math.log(s["opacity"] / (1 - s["opacity"])) for s in splats]
        axes = [spread([s["dc"][k] for s in splats], 255) for k in range(3)] + [spread(logit, 255)]
        files["sh0.webp"] = image([tuple(axes[k][2][i] for k in range(4)) for i in range(n)])
        for i, s in enumerate(held):
            back = [axes[k][0] + (axes[k][1] - axes[k][0]) * (axes[k][2][i] / 255) for k in range(4)]
            s["dc"], s["opacity"] = back[:3], 1 / (1 + math.exp(-back[3]))
        meta["sh0"] = {"mins": [a[0] for a in axes], "maxs": [a[1] for a in axes], "files": ["sh0.webp"]}
    else:
        book = scale_codebook or linear(-8.0, 2.0)
        codes = [[nearest(book, c) for c in s["scale"]] for s in splats]
        files["scales.webp"] = image([(*c, 255) for c in codes])
        meta["scales"] = {"codebook": book, "files": ["scales.webp"]}
        colors = dc_codebook or linear(-3.0, 3.0)
        pixels = []
        for s, c in zip(held, codes):
            s["scale"] = [book[k] for k in c]
            dc = [nearest(colors, v) for v in s["dc"]]
            if version == 2:
                a = round(max(0.0, min(1.0, s["opacity"])) * 255)
                s["opacity"] = a / 255
            else:
                a = min(255, round(204 * math.sqrt(s["opacity"]))) if s["opacity"] > 0 else 0
                s["opacity"] = (a / 204) ** 2
            s["dc"] = [colors[k] for k in dc]
            pixels.append((*dc, a))
        files["sh0.webp"] = image(pixels)
        meta["sh0"] = {"codebook": colors, "files": ["sh0.webp"]}
        top = max((s["opacity"] for s in held), default=0)
        if version == 3 and top > 1:
            meta["opacity"] = {"max": top}

    # The higher harmonics: a palette of entries, 64 a row, a pixel a coefficient, and a label
    # a splat (16 bits, 24 in version 3).
    if bands:
        coeffs = COEFFS[bands - 1]
        if palette is None:
            palette, labels = [s["sh"] for s in splats], list(range(n))
        rows = math.ceil(len(palette) / 64)
        pixels = [(0, 0, 0, 255)] * (64 * coeffs * rows)
        if version == 1:
            lo, hi, _ = spread([c for entry in palette for rgb in entry for c in rgb], 255)
            book = [lo + (hi - lo) * (b / 255) for b in range(256)]
            section = {"mins": lo, "maxs": hi}
        else:
            book = sh_codebook or linear(-2.0, 2.0)
            section = {"count": len(palette), "bands": bands, "codebook": book} if state_bands else {"codebook": book}
        decoded = []
        for e, entry in enumerate(palette):
            assert len(entry) == coeffs
            row = []
            for c, rgb in enumerate(entry):
                code = [nearest(book, v) for v in rgb]
                pixels[(e // 64) * 64 * coeffs + (e % 64) * coeffs + c] = (*code, 255)
                row.append([book[k] for k in code])
            decoded.append(row)
        files["shN_centroids.webp"] = image(pixels, (64 * coeffs, rows))
        files["shN_labels.webp"] = image([(l & 255, (l >> 8) & 255, l >> 16, 255) for l in labels])
        for s, l in zip(held, labels):
            s["sh"] = decoded[l] if l < len(palette) else [[0.0] * 3] * coeffs
        meta["shN"] = {**section, "files": ["shN_centroids.webp", "shN_labels.webp"]}
    else:
        for s in held:
            s["sh"] = []

    # Blind data: a texture's fields packed in order into a pixel's bytes, low byte first.
    for t, texture in enumerate(blind or []):
        pixels = [[0, 0, 0, 255] for _ in range(n)]
        at = 0
        for field, values in texture:
            size = BLIND_BYTES[field["type"]]
            lo, hi = field.get("range", [0, 1])
            back = []
            for i, v in enumerate(values):
                kind = field["type"]
                if kind == "f16":
                    code = struct.unpack("<H", struct.pack("<e", v))[0]
                    back.append(struct.unpack("<e", struct.pack("<e", v))[0])
                elif kind == "f32":
                    code = struct.unpack("<I", struct.pack("<f", v))[0]
                    back.append(f32(v))
                elif kind.startswith("unorm"):
                    top = 255 if size == 1 else 65535
                    code = round((v - lo) / (hi - lo) * top)
                    back.append(lo + code / top * (hi - lo))
                else:
                    code = v
                    back.append(float(v))
                pixels[i][at:at + size] = code.to_bytes(size, "little")
            for s, v in zip(held, back):
                s.setdefault("blind", {})[field["name"]] = v
            at += size
        assert at <= 4
        files[f"blind_{t}.webp"] = image([tuple(p) for p in pixels])
        meta.setdefault("blind", []).append({"files": [f"blind_{t}.webp"], "fields": [f for f, _ in texture]})

    meta.update(meta_extra or {})
    files["meta.json"] = json.dumps(meta).encode()
    return files, meta, held


def zipped(files, order=None):
    """The files as a .sog: a zip of stored entries, meta.json last as splat-transform writes it."""
    out = io.BytesIO()
    names = order or [n for n in files if n != "meta.json"] + ["meta.json"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for name in names:
            z.writestr(zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)), files[name])
    return out.getvalue()


# --- compressed PLY -----------------------------------------------------------------------------

CHUNK_PROPS = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z", "min_scale_x", "min_scale_y", "min_scale_z",
               "max_scale_x", "max_scale_y", "max_scale_z", "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"]
VERTEX_PROPS = ["packed_position", "packed_rotation", "packed_scale", "packed_color"]


def pack_unorm(value, bits):
    t = (1 << bits) - 1
    return max(0, min(t, math.floor(value * t + 0.5)))


def unorm(value, bits):
    t = (1 << bits) - 1
    return (value & t) / t


def normalize(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return 0.0 if hi - lo < 0.00001 else (x - lo) / (hi - lo)


def lerp(a, b, t):
    return a * (1 - t) + b * t


def compressed_ply(splats, bands=0, chunk_colors=True, comments=("Generated by tools/corpus/ply.py",)):
    """A compressed PLY and the splats as it holds them: (bytes, [splat])."""
    n = len(splats)
    coeffs = COEFFS[bands - 1] if bands else 0
    chunk_props = CHUNK_PROPS if chunk_colors else CHUNK_PROPS[:12]
    head = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in comments),
            f"element chunk {math.ceil(n / 256)}", *(f"property float {p}" for p in chunk_props),
            f"element vertex {n}", *(f"property uint {p}" for p in VERTEX_PROPS)]
    if bands:
        head += [f"element sh {n}", *(f"property uchar f_rest_{k}" for k in range(coeffs * 3))]
    chunks, vertices, sh, held = b"", b"", b"", []
    for start in range(0, n, 256):
        part = splats[start:start + 256]
        color = [[f32(f32(s["dc"][k]) * SH_C0 + 0.5) for k in range(3)] for s in part]
        scale = [[max(-20.0, min(20.0, f32(s["scale"][k]))) for k in range(3)] for s in part]
        pos = [[f32(c) for c in s["pos"]] for s in part]
        bounds = []
        for values in (pos, scale, color) if chunk_colors else (pos, scale):
            lo = [f32(min(v[k] for v in values)) for k in range(3)]
            hi = [f32(max(v[k] for v in values)) for k in range(3)]
            bounds += [lo, hi]
        if not chunk_colors:
            bounds += [[0.0] * 3, [1.0] * 3]
        chunks += struct.pack(f"<{len(chunk_props)}f", *(c for b in bounds[:len(chunk_props) // 3] for c in b))
        for s, p, e, c in zip(part, pos, scale, color):
            t = [[normalize(v[k], bounds[2 * j][k], bounds[2 * j + 1][k]) for k in range(3)]
                 for j, v in enumerate((p, e, c))]
            words = [pack_unorm(t[0][0], 11) << 21 | pack_unorm(t[0][1], 10) << 11 | pack_unorm(t[0][2], 11)]
            # The rotation: the index of the largest of (w, x, y, z), then the others, 10 bits each.
            ln = math.sqrt(sum(v * v for v in s["rot"]))
            q = [v / ln for v in s["rot"]]
            largest = max(range(4), key=lambda k: abs(q[k]))
            if q[largest] < 0:
                q = [-v for v in q]
            word = largest
            for k in range(4):
                if k != largest:
                    word = word << 10 | pack_unorm(q[k] * math.sqrt(2) * 0.5 + 0.5, 10)
            words.append(word)
            words.append(pack_unorm(t[1][0], 11) << 21 | pack_unorm(t[1][1], 10) << 11 | pack_unorm(t[1][2], 11))
            words.append(pack_unorm(t[2][0], 8) << 24 | pack_unorm(t[2][1], 8) << 16 | pack_unorm(t[2][2], 8) << 8
                         | pack_unorm(s["opacity"], 8))
            vertices += struct.pack("<4I", *words)
            # What the words decode to.
            back = dict(s)
            u = [unorm(words[0] >> 21, 11), unorm(words[0] >> 11, 10), unorm(words[0], 11)]
            back["pos"] = [lerp(bounds[0][k], bounds[1][k], u[k]) for k in range(3)]
            u = [unorm(words[2] >> 21, 11), unorm(words[2] >> 11, 10), unorm(words[2], 11)]
            back["scale"] = [lerp(bounds[2][k], bounds[3][k], u[k]) for k in range(3)]
            u = [unorm(words[3] >> 24, 8), unorm(words[3] >> 16, 8), unorm(words[3] >> 8, 8)]
            back["dc"] = [(lerp(bounds[4][k], bounds[5][k], u[k]) - 0.5) / SH_C0 for k in range(3)]
            back["opacity"] = unorm(words[3], 8)
            rest = [(unorm(word >> shift, 10) - 0.5) * math.sqrt(2) for shift in (20, 10, 0)]
            rest.insert(largest, math.sqrt(max(0.0, 1 - sum(v * v for v in rest))))
            back["rot"] = rest
            # The harmonics, a color channel at a time: a byte's step of the range from -4 to 4.
            codes = [max(0, min(255, math.trunc((s["sh"][j][c] / 8 + 0.5) * 256)))
                     for c in range(3) for j in range(coeffs)]
            sh += bytes(codes)
            value = [((0 if b == 0 else 1 if b == 255 else (b + 0.5) / 256) - 0.5) * 8 for b in codes]
            back["sh"] = [[value[c * coeffs + j] for c in range(3)] for j in range(coeffs)]
            held.append(back)
    return "\n".join(head + ["end_header\n"]).encode() + chunks + vertices + sh, held
