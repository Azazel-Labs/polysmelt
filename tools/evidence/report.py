"""Builds EVIDENCE.md, the run summary and CSVs from one evidence.yml run's raw results.

    python3 tools/evidence/report.py --in <results-dir> --out <report-dir> [--run-url URL]

`<results-dir>` holds what the jobs uploaded (any job may be missing; its sections say so):

    env/*.json                     machine, versions, corpus pins
    speed/sweep.jsonl              every (file, loader): status, counts, peak RSS, heap
    speed/time.json                pinned, rotating timing of the loaders that read each file
    speed/time-mt.json             OBJ with Polysmelt's threads on, unpinned
    speed/torus.md                 polysmelt-bench (generated tori)
    quality/parity.json            glTF deep parity against the gltf crate
    quality/glb-sweep.jsonl        the conversions that wrote GLBs
    quality/validate-*.json        Khronos validator reports
    resilience/resilience-*.json   damaged-file shards
    blender/blender.jsonl          Blender's importers
    profile/*.svg                  flamegraphs
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict

P = "polysmelt"
PG = "polysmelt → GLB"
A = "assimp"
AG = "assimp → GLB"
B = "blender"
FAIL = ("err", "panic", "crash", "oom", "timeout", "empty")
HARD = ("triangles",)
SOFT = ("vertices", "materials", "nodes", "joints", "animations")



# Competitors whose result is less than what Polysmelt's load makes, so their times aren't like for like.
PARTIAL = {
    "dot_vox": "only parses the voxels; Polysmelt also meshes them into triangles",
    "fbxcel-dom": "only parses the node tree; Polysmelt also decodes the geometry, materials, skins and animation",
    "bvh_anim": "only parses the joints and frames; Polysmelt also builds the glTF skeleton and animation",
    "ply-rs": "parses into a generic map per element; Polysmelt builds an indexed mesh",
}

def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return out


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def table(head, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(out) + "\n"


def code(s):
    return f"`{s}`"


def short(s, n=110):
    s = str(s or "").replace("\n", " ").replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def fmt_ratio(x):
    return "–" if x is None else f"{x:.2f}×"


def fmt_ms(x):
    if x is None:
        return "–"
    return f"{x:.3f}" if x < 1 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def mb(b):
    return f"{b / 1048576:.1f}"


def write_csv(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def is_crate(loader):
    return loader not in (P, PG, A, AG, B) and not loader.startswith("polysmelt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-url", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = a.inp
    md = []
    losses = []  # (rank, kind, file, detail, magnitude)

    env = {}
    for p in sorted(glob.glob(os.path.join(R, "env", "*.json"))):
        env.update(load_json(p, {}))
    sweep = load_jsonl(os.path.join(R, "speed", "sweep.jsonl"))
    blender = load_jsonl(os.path.join(R, "blender", "blender.jsonl"))
    timing = load_json(os.path.join(R, "speed", "time.json"), {})
    timing_mt = load_json(os.path.join(R, "speed", "time-mt.json"), {})
    parity = load_json(os.path.join(R, "quality", "parity.json"), [])
    glb_sweep = load_jsonl(os.path.join(R, "quality", "glb-sweep.jsonl"))
    val_p = load_json(os.path.join(R, "quality", "validate-polysmelt.json"), None)
    val_a = load_json(os.path.join(R, "quality", "validate-assimp.json"), None)
    res = [load_json(p, {}) for p in sorted(glob.glob(os.path.join(R, "resilience", "resilience-*.json")))]
    svgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, "profile", "*.svg")))

    # (corpus/file) -> loader -> record
    by_file = defaultdict(dict)
    ext_of = {}
    size_of = {}
    for r in sweep + blender:
        k = f"{r['corpus']}/{r['file']}"
        by_file[k][r["loader"]] = r
        ext_of[k] = r["ext"]
        size_of[k] = r.get("bytes", 0)
    loaders_seen = sorted({l for v in by_file.values() for l in v})
    crates = [l for l in loaders_seen if is_crate(l)]

    md.append("# Evidence: Polysmelt against assimp, Blender and the Rust crates\n")
    md.append(
        "Regenerated by `tools/evidence/report.py` from one run of `.github/workflows/evidence.yml`"
        + (f" ([run]({a.run_url}))" if a.run_url else "")
        + ". Every number below is from that run; the raw JSON and CSV are in its `evidence` artifact.\n"
    )
    if env:
        rows = [[k, short(v, 160)] for k, v in sorted(env.items())]
        md.append("## Machine and versions\n")
        md.append(table(["", ""], rows, ["---", "---"]))
        md.append(
            "\nAll timed comparisons in one table come from one job on one machine. Timing runs pinned to one core "
            "(`taskset`), every library in the same process, the loaders for a file taking turns in a rotating "
            "order; the figure is the median of the rounds. Polysmelt runs with `threads: 1` except in the table "
            "that says otherwise.\n"
        )

    # --- Completeness ---------------------------------------------------------------------------
    md.append("## 1. Completeness: which files each library reads\n")
    if not by_file:
        md.append("_No sweep results in this run._\n")
    else:
        md.append(
            "Every file of the corpora whose extension any library here reads, each (file, library) pair in a process "
            "of its own with a 60 s timeout and a memory ceiling. `ok` is a clean read; a failure is an error the "
            "library returned, a panic, a crash (signal), running out of memory (`oom`) or the timeout. Polysmelt's "
            "column is the load; `→ GLB` is load plus writing the GLB. The crate column is the best of the Rust "
            "crates for the format (`gltf` with or without image decoding). Blender is its own importers, headless "
            "(`empty`: the importer finished but made nothing).\n"
        )
        per_ext = defaultdict(list)
        for k, ls in by_file.items():
            per_ext[ext_of[k]].append(k)
        rows, csv_rows, only_others = [], [], []

        def stat(ls, name):
            r = ls.get(name)
            return r["status"] if r else None

        for ext in sorted(per_ext, key=lambda e: (-len(per_ext[e]), e)):
            ks = per_ext[ext]
            pc = Counter(stat(by_file[k], P) for k in ks)
            if pc.get("unsupported", 0) == len(ks) or pc.get(None, 0) + pc.get("unsupported", 0) == len(ks):
                ac = Counter(stat(by_file[k], A) for k in ks)
                bc = Counter(stat(by_file[k], B) for k in ks)
                only_others.append([ext, len(ks), f"{ac.get('ok', 0)}" if ac.get(None, 0) < len(ks) else "–",
                                    f"{bc.get('ok', 0)}" if bc.get(None, 0) + bc.get("unsupported", 0) < len(ks) else "–"])
                continue
            gc = Counter(stat(by_file[k], PG) for k in ks)
            crate_names = sorted({l for k in ks for l in by_file[k] if is_crate(l)})
            def best_crate(k):
                ss = [by_file[k][l]["status"] for l in crate_names if l in by_file[k]]
                return "ok" if "ok" in ss else (ss[0] if ss else None)
            cc = Counter(best_crate(k) for k in ks)
            ac = Counter(stat(by_file[k], A) for k in ks)
            bc = Counter(stat(by_file[k], B) for k in ks)
            n = len(ks)
            def col(c):
                if c.get(None, 0) + c.get("unsupported", 0) == n:
                    return "–"
                return f"{c.get('ok', 0)}"
            rows.append([
                ext, n, pc.get("ok", 0), pc.get("err", 0), pc.get("panic", 0), pc.get("crash", 0) + pc.get("oom", 0),
                pc.get("timeout", 0), col(gc), ", ".join(code(c) for c in crate_names) or "–", col(cc), col(ac),
                col(bc),
            ])
            csv_rows.append([ext, n, pc.get("ok", 0), pc.get("err", 0), pc.get("panic", 0), pc.get("crash", 0),
                             pc.get("oom", 0), pc.get("timeout", 0), gc.get("ok", 0), "/".join(crate_names),
                             cc.get("ok", 0), ac.get("ok", 0), bc.get("ok", 0), ac.get("panic", 0) + ac.get("crash", 0),
                             ac.get("timeout", 0), bc.get("crash", 0), bc.get("timeout", 0)])
        md.append(table(
            ["Format", "Files", "Polysmelt ok", "error", "panic", "crash/oom", "timeout", "→ GLB ok", "Crate",
             "crate ok", "assimp ok", "Blender ok"],
            rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:", "---:", "---:"]))
        write_csv(os.path.join(a.out, "completeness.csv"),
                  ["format", "files", "polysmelt_ok", "polysmelt_err", "polysmelt_panic", "polysmelt_crash",
                   "polysmelt_oom", "polysmelt_timeout", "polysmelt_glb_ok", "crates", "crate_ok", "assimp_ok",
                   "blender_ok", "assimp_panic_crash", "assimp_timeout", "blender_crash", "blender_timeout"], csv_rows)

        # Failure modes of the others, for the record.
        fm = defaultdict(Counter)
        for k, ls in by_file.items():
            for l, r in ls.items():
                fm[l][r["status"]] += 1
        md.append("\nEvery library's outcomes over all the files it was given:\n\n")
        statuses = ["ok", "err", "empty", "panic", "crash", "oom", "timeout"]
        md.append(table(["Library"] + statuses, [[code(l)] + [fm[l].get(s, 0) for s in statuses] for l in sorted(fm) if l != P or True]))

        if only_others:
            md.append("\nFormats Polysmelt doesn't read at all (files in the corpora; files each reads):\n\n")
            md.append(table(["Format", "Files", "assimp ok", "Blender ok"], only_others))
            for ext, nf, aok, bok in only_others:
                if (aok not in ("–", "0")) or (bok not in ("–", "0")):
                    losses.append((6, "format not read", f"*.{ext}", f"{nf} files; assimp reads {aok}, Blender {bok}", nf))

        # Files someone reads and Polysmelt doesn't.
        md.append("\n### Files another library reads and Polysmelt doesn't\n\n")
        lost = []
        for k, ls in sorted(by_file.items()):
            p = ls.get(P)
            if not p or p["status"] in ("ok", "unsupported"):
                continue
            readers = [l for l, r in ls.items() if l not in (P, PG) and r["status"] == "ok"]
            if readers:
                lost.append([k, p["status"], ", ".join(code(r) for r in sorted(readers)), short(p.get("error"), 140)])
                rank = 0 if p["status"] in ("panic", "crash", "oom", "timeout") else 1
                losses.append((rank, f"fails a file others read ({p['status']})", k,
                               f"read by {', '.join(sorted(readers))}: {short(p.get('error'), 160)}", size_of.get(k, 0)))
        md.append(table(["File", "Polysmelt", "Read by", "Polysmelt's error"], lost, ["---", "---", "---", "---"])
                  if lost else "None.\n")
        write_csv(os.path.join(a.out, "polysmelt-fails-others-read.csv"), ["file", "status", "readers", "error"],
                  [[r[0], r[1], r[2].replace("`", ""), r[3]] for r in lost])

        # Polysmelt failures nobody else reads (for honesty, they're listed too).
        alone = []
        for k, ls in sorted(by_file.items()):
            p = ls.get(P)
            if p and p["status"] not in ("ok", "unsupported") and not any(
                    r["status"] == "ok" for l, r in ls.items() if l not in (P, PG)):
                alone.append([k, p["status"], short(p.get("error"), 140)])
                if p["status"] in ("panic", "crash", "oom", "timeout"):
                    losses.append((0, f"{p['status']} (no one reads it)", k, short(p.get("error"), 160), 0))
        md.append(f"\nFiles no library here reads, Polysmelt included: {len(alone)}"
                  + (" (Polysmelt's outcome and error):\n\n" + table(["File", "Status", "Error"], alone[:80],
                                                                     ["---", "---", "---"]) if alone else ".\n"))
        if len(alone) > 80:
            md.append(f"\n… and {len(alone) - 80} more in `polysmelt-fails-alone.csv`.\n")
        write_csv(os.path.join(a.out, "polysmelt-fails-alone.csv"), ["file", "status", "error"], alone)

        # Conversions that fail after a clean load.
        conv = [[k, short(ls[PG].get("error"), 140)] for k, ls in sorted(by_file.items())
                if ls.get(P, {}).get("status") == "ok" and PG in ls and ls[PG]["status"] != "ok"]
        if conv:
            md.append("\nFiles Polysmelt loads but fails to write as GLB:\n\n" + table(["File", "Error"], conv, ["---", "---"]))
            for k, e in conv:
                losses.append((1, "loads but GLB write fails", k, e, size_of.get(k, 0)))

        # Others reading files Polysmelt reads and they don't: counts only.
        md.append("\n### Files Polysmelt reads and another library doesn't\n\n")
        wins = defaultdict(lambda: [0, 0])
        for k, ls in by_file.items():
            if ls.get(P, {}).get("status") != "ok":
                continue
            for l, r in ls.items():
                if l in (P, PG) or r["status"] == "unsupported":
                    continue
                wins[l][1] += 1
                if r["status"] != "ok":
                    wins[l][0] += 1
        md.append(table(["Library", "Files Polysmelt reads that it was given", "of those, it fails"],
                        [[code(l), v[1], v[0]] for l, v in sorted(wins.items())]))

    # --- Count parity ---------------------------------------------------------------------------
    md.append("\n## 2. Count parity: do they find the same things?\n")
    declared = {}
    for r in parity or []:
        if r.get("declared"):
            declared[f"khronos/{r['file']}"] = r["declared"]
    if by_file:
        md.append(
            "Where Polysmelt and another library both read a file, their counts compared. Libraries count differently "
            "by design, so only triangles are a hard check; vertices (assimp and tobj split vertices per face corner "
            "or per material, Polysmelt keeps the file's), materials (assimp adds a default), nodes (assimp adds a "
            "root), joints and animations are soft and only tallied. Counts are per mesh, each mesh once however many "
            "nodes draw it. A disagreement is not a verdict: it's flagged for a human. For glTF files the count the "
            "JSON declares is the referee where available.\n"
        )
        agree = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # loader -> key -> [agree, total]
        dis = []
        for k, ls in sorted(by_file.items()):
            p = ls.get(P)
            if not p or p["status"] != "ok":
                continue
            ps = p.get("stats", {})
            for l, r in ls.items():
                if l in (P, PG) or r["status"] != "ok" or not r.get("stats"):
                    continue
                cs = r["stats"]
                for key in HARD + SOFT:
                    if key not in cs or key not in ps:
                        continue
                    pv, cv = ps[key], cs[key]
                    if l == B and key == "nodes":
                        continue  # Blender objects and glTF nodes aren't the same thing
                    agree[l][key][1] += 1
                    if pv == cv:
                        agree[l][key][0] += 1
                    else:
                        d = declared.get(k, {}).get(key)
                        rel = abs(pv - cv) / max(pv, cv, 1)
                        dis.append([k, l, key, pv, cv, "" if d is None else d, rel])
        keys = list(HARD + SOFT)
        rows = []
        for l in sorted(agree):
            rows.append([code(l)] + [f"{agree[l][kk][0]}/{agree[l][kk][1]}" if agree[l][kk][1] else "–" for kk in keys])
        md.append(table(["Against", "triangles agree"] + [f"{kk} agree" for kk in SOFT], rows))
        hard = sorted([d for d in dis if d[2] in HARD], key=lambda d: (-d[6], d[0]))
        write_csv(os.path.join(a.out, "disagreements.csv"),
                  ["file", "other", "count", "polysmelt", "other_value", "declared", "relative_difference"], dis)
        md.append(f"\n### Triangle disagreements ({len(hard)})\n\n")
        if hard:
            # Who's in the majority on each file: the others agreeing among themselves against Polysmelt
            # weighs more.
            per_file = defaultdict(list)
            for d in hard:
                per_file[d[0]].append(d)
            shown = []
            for d in hard[:150]:
                others = [x for x in by_file[d[0]].items() if x[0] not in (P, PG) and x[1]["status"] == "ok"
                          and "triangles" in (x[1].get("stats") or {})]
                agree_p = sum(1 for _, r in others if r["stats"]["triangles"] == d[3])
                shown.append([d[0], code(d[1]), d[3], d[4], d[5] if d[5] != "" else "–", f"{d[6] * 100:.1f}%",
                              f"{agree_p}/{len(others)}"])
            md.append(table(["File", "Other", "Polysmelt", "Other's", "Declared", "Difference",
                             "Others agreeing with Polysmelt"], shown,
                            ["---", "---", "---:", "---:", "---:", "---:", "---:"]))
            if len(hard) > 150:
                md.append(f"\n… {len(hard) - 150} more in `disagreements.csv`.\n")
            if any(d[0].endswith(".dae") for d in hard):
                md.append(
                    "\nKnown causes on COLLADA, checked by hand against the files' `<polylist>` index lists: "
                    "Polysmelt makes one glTF mesh per (geometry, material binding) pair, so a geometry that two "
                    "instances bind to different materials counts twice (GearboxAssy.dae: 32 geometries, 41 pairs, "
                    "378 674 triangles exactly); and it drops triangles whose three corners are not three distinct "
                    "vertices, i.e. repeat a full (position, normal, texcoord) index tuple (2CylinderEngine.dae "
                    "75 730 less 10 413 = 65 317; Buggy.dae 308 306 less 28 457 = 279 849). The `collada` crate "
                    "counts every geometry once and every triangle the file lists.\n")
            for f, ds in per_file.items():
                others = [x for x in by_file[f].items() if x[0] not in (P, PG) and x[1]["status"] == "ok"
                          and "triangles" in (x[1].get("stats") or {})]
                agree_p = sum(1 for _, r in others if r["stats"]["triangles"] == ds[0][3])
                decl = ds[0][5]
                if decl != "" and decl == ds[0][3]:
                    continue  # the file's own JSON agrees with Polysmelt
                if agree_p == 0:
                    losses.append((2, "triangle count disagrees with every other reader", f,
                                   "; ".join(f"{d[1]} {d[4]} vs polysmelt {d[3]}" for d in ds), max(d[6] for d in ds)))
        else:
            md.append("None.\n")

    if parity:
        md.append("\n### glTF deep parity with the `gltf` crate (values, not just counts)\n\n")
        md.append(
            "Each Khronos sample read by both: every accessor through each library's readers, the sum of every "
            "position component and of every index compared, and every count.\n\n"
        )
        outcome = Counter()
        mism = []
        for r in parity:
            g, p = r.get("gltf", {}), r.get("polysmelt", {})
            gs, ps_ = g.get("status"), p.get("status")
            outcome[(gs, ps_)] += 1
            if gs == "ok" and ps_ == "ok" and g.get("stats") and p.get("stats"):
                a_, b_ = g["stats"], p["stats"]
                bad = []
                for kk in ("meshes", "primitives", "vertices", "indices", "triangles", "attrs", "morph_targets", "skins",
                           "joints", "animations", "channels", "keys", "materials", "textures", "images", "cameras",
                           "lights", "scenes", "nodes"):
                    if a_.get(kk) != b_.get(kk):
                        bad.append(f"{kk} {b_.get(kk)} vs {a_.get(kk)}")
                for kk in ("pos_sum", "idx_sum"):
                    x, y = a_.get(kk) or 0, b_.get(kk) or 0
                    if abs(x - y) > 1e-6 * max(abs(x), abs(y), 1):
                        bad.append(f"{kk} {y:.6g} vs {x:.6g}")
                if bad:
                    mism.append([r["file"], "; ".join(bad)])
            if ps_ != "ok" and gs == "ok":
                losses.append((1, "fails a glTF the gltf crate reads (deep parity)", f"khronos/{r['file']}",
                               short(p.get("error"), 160), r.get("bytes", 0)))
        md.append(table(["gltf crate", "Polysmelt", "Files"],
                        [[s1 or "–", s2 or "–", n] for (s1, s2), n in sorted(outcome.items(), key=lambda x: -x[1])],
                        ["---", "---", "---:"]))
        md.append(f"\nFiles both read with any count or value different: {len(mism)}\n\n")
        if mism:
            md.append(table(["File", "Polysmelt vs gltf"], mism[:60], ["---", "---"]))
            for f, b in mism:
                losses.append((2, "deep glTF parity mismatch", f"khronos/{f}", b, 0))

    # --- Quality --------------------------------------------------------------------------------
    md.append("\n## 3. Output quality: the Khronos validator on every GLB written\n")
    if val_p is None and val_a is None:
        md.append("_No validator results in this run._\n")
    else:
        md.append(
            "Every file each library converts (Polysmelt: load and `to_glb`; assimp: its `glb2` exporter, as "
            "`assimp export x.glb` does) written as a GLB and run through the Khronos glTF validator "
            "(`gltf-validator` 2.0.0-dev.3.10). Errors are spec violations; warnings are advisory. GLBs over 200 MB "
            "are skipped by the validator run and counted as such.\n\n"
        )
        rows = []
        codes = {}
        for name, rep in ((P, val_p), (A, val_a)):
            if rep is None:
                continue
            files = len(rep)
            crashed = sum(1 for v in rep.values() if "crash" in v)
            skipped = sum(1 for v in rep.values() if "skipped" in v)
            with_err = sum(1 for v in rep.values() if v.get("error_count", 0) > 0)
            errs = sum(v.get("error_count", 0) for v in rep.values())
            warns = sum(v.get("warnings", 0) for v in rep.values())
            with_warn = sum(1 for v in rep.values() if v.get("warnings", 0) > 0)
            c = Counter()
            for v in rep.values():
                c.update(v.get("errors", {}))
            codes[name] = c
            rows.append([code(name), files, files - with_err - crashed - skipped, with_err, errs, with_warn, warns,
                         crashed, skipped])
        md.append(table(["Writer", "GLBs", "valid (0 errors)", "with errors", "errors", "with warnings", "warnings",
                         "validator crashed", "skipped (size)"], rows))
        for name, c in codes.items():
            if c:
                md.append(f"\nMost common errors in {code(name)}'s GLBs: "
                          + ", ".join(f"{code(k)} ×{v}" for k, v in c.most_common(12)) + "\n")
        # Per-format, both writers, files in common.
        def src_key(k):
            return k[:-4] if k.endswith(".glb") else k
        vp = {src_key(k): v for k, v in (val_p or {}).items()}
        va = {src_key(k): v for k, v in (val_a or {}).items()}
        fmt = defaultdict(lambda: [0, 0, 0, 0, 0])
        for k in set(vp) | set(va):
            ext = k.rsplit(".", 1)[-1].lower()
            f = fmt[ext]
            if k in vp:
                f[0] += 1
                f[1] += vp[k].get("error_count", 0) > 0 or "crash" in vp[k]
            if k in va:
                f[2] += 1
                f[3] += va[k].get("error_count", 0) > 0 or "crash" in va[k]
        md.append("\nBy source format (GLBs written, of which with errors):\n\n")
        md.append(table(["Source format", "Polysmelt GLBs", "with errors", "assimp GLBs", "with errors"],
                        [[e, v[0], v[1], v[2], v[3]] for e, v in sorted(fmt.items(), key=lambda x: -x[1][0])]))
        bad = sorted(k for k, v in vp.items() if v.get("error_count", 0) > 0 or "crash" in v)
        if bad:
            md.append(f"\nPolysmelt GLBs with validator errors ({len(bad)}):\n\n")
            md.append(table(["Source file", "Errors", "assimp's GLB of it"],
                            [[k, ", ".join(f"{c} ×{n}" for c, n in vp[k].get("errors", {}).items()) or vp[k].get("crash"),
                              ("–" if k not in va else f"{va[k].get('error_count', 'crash')} errors")] for k in bad[:100]],
                            ["---", "---", "---"]))
            for k in bad:
                clean_a = k in va and va[k].get("error_count", 1) == 0
                losses.append((3, "GLB fails validation" + (" (assimp's is clean)" if clean_a else ""), k,
                               ", ".join(f"{c} ×{n}" for c, n in vp[k].get("errors", {}).items()), 0))
        write_csv(os.path.join(a.out, "validator.csv"), ["writer", "source", "errors", "warnings", "codes"],
                  [[w, k, v.get("error_count", ""), v.get("warnings", ""), json.dumps(v.get("errors", v))]
                   for w, rep in ((P, vp), (A, va)) for k, v in sorted(rep.items())])
        conv_ok = Counter(r["loader"] for r in glb_sweep if r.get("status") == "ok")
        conv_all = Counter(r["loader"] for r in glb_sweep if r.get("status") != "unsupported")
        if conv_all:
            md.append("\nConversions attempted and written in the validator job: "
                      + ", ".join(f"{code(l)} {conv_ok[l]}/{conv_all[l]}" for l in sorted(conv_all)) + "\n")

    # --- Resilience -----------------------------------------------------------------------------
    md.append("\n## 4. Resilience: damaged files\n")
    if not res:
        md.append("_No resilience results in this run._\n")
    else:
        tab = defaultdict(Counter)
        notable = []
        for r in res:
            for t in r.get("table", []):
                tab[(t["format"], t["loader"])].update(t["counts"])
            notable += r.get("notable", [])
        r0 = res[0]
        md.append(
            f"Real files truncated at many offsets and with bytes flipped ({r0.get('flips')} flip cases per file, a fixed "
            f"seed), every damaged file through Polysmelt's whole conversion (load, both GLBs, the rig) and each other "
            f"library that reads the format. Each case runs in a worker process on a thread with an 8 MiB stack, a "
            f"{r0.get('timeout_s')} s deadline and a {r0.get('limit_mb')} MB heap ceiling (address space for C++). "
            f"`Error` is the right answer to a damaged file; panic, abort, crash and hang are not.\n\n"
        )
        rows, csv_rows = [], []
        for (f, l), c in sorted(tab.items()):
            orig = sum(v for k, v in c.items() if k.startswith("orig-"))
            total = sum(c.values())
            row = [f, code(l), total, c.get("ok", 0), c.get("err", 0), c.get("panic", 0), c.get("abort-alloc", 0),
                   c.get("crash", 0), c.get("hang", 0), orig]
            rows.append(row)
            csv_rows.append([f, l] + row[2:])
        md.append(table(["Format", "Library", "Cases", "Read", "Error", "Panic", "Abort (alloc)", "Crash", "Hang",
                         "Undamaged file failed"], rows, ["---", "---"] + ["---:"] * 8))
        write_csv(os.path.join(a.out, "resilience.csv"),
                  ["format", "library", "cases", "read", "error", "panic", "abort_alloc", "crash", "hang", "orig_failed"],
                  csv_rows)
        tot = defaultdict(Counter)
        for (f, l), c in tab.items():
            tot[l].update(c)
        md.append("\nTotals per library:\n\n")
        md.append(table(["Library", "Cases", "Panic", "Abort (alloc)", "Crash", "Hang"],
                        [[code(l), sum(c.values()), c.get("panic", 0), c.get("abort-alloc", 0), c.get("crash", 0),
                          c.get("hang", 0)] for l, c in sorted(tot.items())]))
        mine = [n for n in notable if n.get("loader") == P and not str(n.get("status", "")).startswith("orig-")]
        md.append(f"\nPolysmelt's own panics, aborts, crashes and hangs: {len(mine)}\n\n")
        if mine:
            md.append(table(["File", "Damage", "Outcome", "Detail"],
                            [[n["file"], n["mutation"], n["status"], short(n.get("detail"), 140)] for n in mine[:80]],
                            ["---", "---", "---", "---"]))
            for n in mine:
                losses.append((0, f"resilience: {n['status']} on a damaged file", n["file"],
                               f"{n['mutation']}: {short(n.get('detail'), 140)}", 0))

    # --- Speed ----------------------------------------------------------------------------------
    md.append("\n## 5. Speed\n")
    tfiles = (timing or {}).get("files", [])
    speed_loss = []
    if not tfiles:
        md.append("_No timing results in this run._\n")
    else:
        md.append(
            f"Pinned to CPU {timing.get('cpus_allowed')}, all libraries in one process, rotating turns, at least 5 "
            f"rounds (3 for files whose round takes over 4 s, 1 over 20 s) and {timing.get('budget_s')} s per file. "
            f"Only files every library in the pair read cleanly in the sweep. `×` is the other's median time over "
            f"Polysmelt's: above 1, Polysmelt is faster. Load is the library's read into its in-memory result "
            f"(Polysmelt's glTF load includes decoding it into a flat mesh); conversion is read plus writing a GLB in "
            f"memory. Threads alive after the run: {timing.get('threads_after')}.\n\n"
        )
        md.append("† Not like for like: " + "; ".join(f"{code(o)} {w}" for o, w in PARTIAL.items()) + ".\n\n")
        pairs = defaultdict(lambda: defaultdict(list))  # (polysmelt side, other) -> ext -> [(file, p, o)]
        for f in tfiles:
            m = {l["loader"]: l for l in f["loaders"] if l.get("median_ms") is not None}
            k = f"{f['corpus']}/{f['file']}"
            for o, lm in m.items():
                if o.startswith("polysmelt"):
                    continue
                side = PG if o == AG else P
                if side in m:
                    pairs[(side, o)][f["ext"]].append((k, m[side]["median_ms"], lm["median_ms"]))
        rows, csv_rows = [], []
        for (side, o), exts in sorted(pairs.items(), key=lambda x: (x[0][1] != A, x[0][1] != AG, x[0][1])):
            for ext, xs in sorted(exts.items(), key=lambda x: -len(x[1])):
                ratios = [oo / pp for _, pp, oo in xs if pp > 0]
                w = sum(1 for r in ratios if r > 1)
                tp, to = sum(x[1] for x in xs), sum(x[2] for x in xs)
                rows.append([code(o) + (" †" if o in PARTIAL else ""), ext, len(xs), fmt_ratio(geomean(ratios)),
                             fmt_ratio(min(ratios) if ratios else None),
                             fmt_ratio(max(ratios) if ratios else None), f"{w}/{len(xs)}", fmt_ms(tp), fmt_ms(to)])
                for k, pp, oo in xs:
                    csv_rows.append([side, o, ext, k, pp, oo, oo / pp if pp else ""])
                    if oo < pp:
                        speed_loss.append((k, side, o, pp, oo))
        md.append(table(["Against", "Format", "Files", "× geomean", "× worst", "× best", "Polysmelt faster",
                         "Polysmelt total ms", "Other total ms"], rows,
                        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        write_csv(os.path.join(a.out, "speed.csv"),
                  ["polysmelt_side", "other", "format", "file", "polysmelt_ms", "other_ms", "ratio"], csv_rows)
        big = [f for f in tfiles if f.get("bytes", 0) >= 4 << 20]
        if big:
            md.append("\nThe biggest files (4 MB and up), every library's median:\n\n")
            rows = []
            for f in sorted(big, key=lambda f: -f["bytes"])[:40]:
                m = {l["loader"]: l.get("median_ms") for l in f["loaders"]}
                rows.append([f"{f['corpus']}/{f['file']}", mb(f["bytes"]),
                             "; ".join(f"{l} {fmt_ms(v)}" for l, v in m.items() if v is not None)])
            md.append(table(["File", "MB", "Median ms"], rows, ["---", "---:", "---"]))
    if (timing_mt or {}).get("files"):
        md.append("\n### OBJ with Polysmelt's threads on (unpinned)\n\n")
        rows = []
        for f in sorted(timing_mt["files"], key=lambda f: -f["bytes"])[:30]:
            m = {l["loader"]: l.get("median_ms") for l in f["loaders"]}
            rows.append([f"{f['corpus']}/{f['file']}", mb(f["bytes"])] +
                        [fmt_ms(m.get(l)) for l in (P, "polysmelt (threads: 0, all cores)", "tobj", A)])
        md.append(table(["File", "MB", "polysmelt (1 thread) ms", "polysmelt (all cores) ms", "tobj ms", "assimp ms"],
                        rows))
        md.append(f"\nCPUs: {timing_mt.get('cpus_allowed')}.\n")
    torus = os.path.join(R, "speed", "torus.md")
    if os.path.exists(torus):
        md.append("\n### Generated tori (`polysmelt-bench`)\n\n")
        md.append(open(torus, encoding="utf-8").read() + "\n")

    # --- Memory ---------------------------------------------------------------------------------
    md.append("\n## 6. Memory\n")
    mem_loss = []
    if by_file:
        md.append(
            "From the sweep: each (file, library) in a fresh process, so the peak resident memory the load added "
            "(`VmHWM` after, less `VmRSS` before, with the peak reset first) counts every heap, assimp's C++ one "
            "included. Only files where one side needs 8 MB or more, where the code pages a load first touches "
            "stop mattering. `×` is the other's peak over Polysmelt's: above 1, Polysmelt needs less. Polysmelt "
            "keeps the bytes of image files a model names (`Keep::ALL`); assimp keeps only their paths.\n\n"
        )
        pairs = defaultdict(lambda: defaultdict(list))
        for k, ls in by_file.items():
            for o, r in ls.items():
                if o.startswith("polysmelt") or o == B or r["status"] != "ok" or "rss_kb" not in r:
                    continue
                side = PG if o == AG else P
                p = ls.get(side)
                if not p or p["status"] != "ok" or "rss_kb" not in p:
                    continue
                pr, orr = p["rss_kb"], r["rss_kb"]
                if max(pr, orr) < 8 * 1024:
                    continue
                pairs[(side, o)][ext_of[k]].append((k, pr, orr, p.get("heap_peak"), r.get("heap_peak")))
        rows, csv_rows = [], []
        for (side, o), exts in sorted(pairs.items(), key=lambda x: (x[0][1] != A, x[0][1] != AG, x[0][1])):
            for ext, xs in sorted(exts.items(), key=lambda x: -len(x[1])):
                ratios = [orr / max(pr, 1) for _, pr, orr, _, _ in xs]
                w = sum(1 for r in ratios if r > 1)
                rows.append([code(o), ext, len(xs), fmt_ratio(geomean(ratios)), fmt_ratio(min(ratios)),
                             fmt_ratio(max(ratios)), f"{w}/{len(xs)}", f"{sum(x[1] for x in xs) / 1024:.0f}",
                             f"{sum(x[2] for x in xs) / 1024:.0f}"])
                for k, pr, orr, ph, oh in xs:
                    csv_rows.append([side, o, ext, k, pr, orr, ph, oh])
                    if orr < pr:
                        mem_loss.append((k, side, o, pr, orr))
        md.append(table(["Against", "Format", "Files", "× geomean", "× worst", "× best", "Polysmelt leaner",
                         "Polysmelt total MB", "Other total MB"], rows,
                        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        write_csv(os.path.join(a.out, "memory.csv"),
                  ["polysmelt_side", "other", "format", "file", "polysmelt_rss_kb", "other_rss_kb",
                   "polysmelt_heap_peak", "other_heap_peak"], csv_rows)

    # --- Profiles -------------------------------------------------------------------------------
    if svgs:
        md.append("\n## 7. Profiles\n\n`perf record` flamegraphs in the run's `flamegraphs` artifact: "
                  + ", ".join(code(s) for s in svgs) + "\n")

    # --- Losses ---------------------------------------------------------------------------------
    for k, side, o, pp, oo in speed_loss:
        losses.append((4, f"slower than {o}" + (" (conversion)" if side == PG else ""), k,
                       f"{fmt_ms(pp)} ms vs {fmt_ms(oo)} ms ({pp / oo:.2f}× slower)"
                       + (f"; not like for like: {o} {PARTIAL[o]}" if o in PARTIAL else ""), pp / oo))
    for k, side, o, pr, orr in mem_loss:
        losses.append((5, f"more memory than {o}" + (" (conversion)" if side == PG else ""), k,
                       f"{pr / 1024:.1f} MB vs {orr / 1024:.1f} MB ({pr / max(orr, 1):.2f}×)", pr / max(orr, 1)))
    losses.sort(key=lambda x: (x[0], -(x[4] or 0), x[2]))
    names = {0: "Panics, crashes, out-of-memory and hangs", 1: "Fails a file another library reads",
             2: "Counts disagree with every other reader", 3: "Output the validator rejects",
             4: "Slower", 5: "More memory", 6: "Formats Polysmelt doesn't read"}
    md.append("\n## Where Polysmelt loses, ranked\n\n"
              "Worst first: robustness failures, then files it can't read that something else can, then counts no "
              "other reader agrees with, invalid output, speed and memory (each sorted by how big the loss is), and "
              "formats it doesn't read at all. The full list is `losses.csv`.\n")
    by_rank = defaultdict(list)
    for l in losses:
        by_rank[l[0]].append(l)
    for rank in sorted(names):
        xs = by_rank.get(rank, [])
        md.append(f"\n### {rank + 1}. {names[rank]} ({len(xs)})\n\n")
        if not xs:
            md.append("None.\n")
            continue
        md.append(table(["What", "File", "Detail"], [[x[1], x[2], short(x[3], 160)] for x in xs[:60]],
                        ["---", "---", "---"]))
        if len(xs) > 60:
            md.append(f"\n… {len(xs) - 60} more in `losses.csv`.\n")
    write_csv(os.path.join(a.out, "losses.csv"), ["rank", "kind", "file", "detail", "magnitude"],
              [[x[0] + 1, x[1], x[2], x[3], x[4]] for x in losses])

    md.append("\n## Reproduce\n\nRun the `evidence` workflow (Actions → evidence → Run workflow), or push to a "
              "`cloud-bench/**` branch. Locally on Linux: `tools/evidence/fetch_corpora.sh corpora`, build assimp "
              "6.0.6 and `cargo build --release --manifest-path bench/Cargo.toml --features assimp` with `ASSIMP_DIR` "
              "set, then the steps of `.github/workflows/evidence.yml`.\n")
    text = "\n".join(md)
    with open(os.path.join(a.out, "EVIDENCE.md"), "w", encoding="utf-8") as f:
        f.write(text)
    # The run summary: everything up to the ranked losses' long tails, within GitHub's 1 MiB.
    summary = text if len(text) < 900_000 else text[:900_000] + "\n\n… (truncated: see EVIDENCE.md in the artifact)\n"
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"EVIDENCE.md: {len(text)} chars, {len(losses)} losses")


if __name__ == "__main__":
    main()
