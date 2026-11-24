"""Every file of the corpora that Blender has an importer for, through Blender, headless.

    python3 tools/evidence/blender_sweep.py --blender <blender> --out blender.jsonl [--jobs 4]
        [--timeout 60] <name>=<dir>...

Each of `--jobs` shards runs one Blender process over its files in turn (blender_import.py). A file
that crashes Blender, or runs past the timeout, is recorded as a `crash` or a `timeout`, and a new
Blender goes on from the next file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

EXTS = {"gltf", "glb", "vrm", "obj", "fbx", "ply", "stl", "dae", "usd", "usda", "usdc", "usdz", "bvh"}
HERE = os.path.dirname(os.path.abspath(__file__))


def files(corpora):
    out = []
    for spec in corpora:
        name, root = spec.split("=", 1)
        for d, dirs, fs in os.walk(root):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for f in sorted(fs):
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                if ext in EXTS:
                    p = os.path.join(d, f)
                    out.append({
                        "corpus": name,
                        "file": os.path.relpath(p, root).replace(os.sep, "/"),
                        "path": p,
                        "ext": ext,
                        "bytes": os.path.getsize(p),
                    })
    return out


def shard(blender, items, timeout, results, lock, tmp, k):
    lst = os.path.join(tmp, f"list{k}.json")
    log = os.path.join(tmp, f"log{k}.jsonl")
    with open(lst, "w", encoding="utf-8") as f:
        json.dump(items, f)
    open(log, "w").close()
    i = 0
    seen = 0  # lines of the log already read
    stalls = 0
    while i < len(items):
        start_i = i
        proc = subprocess.Popen(
            [blender, "-b", "--factory-startup", "--python", os.path.join(HERE, "blender_import.py"), "--", lst, log, str(i)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        current, since = None, time.monotonic()
        while True:
            with open(log, encoding="utf-8") as f:
                lines = f.read().splitlines()
            for line in lines[seen:]:
                try:
                    rec = json.loads(line)
                except ValueError:
                    break  # a line still being written
                seen += 1
                if "start" in rec:
                    current, since = rec["start"], time.monotonic()
                elif "i" in rec:
                    rec.pop("i")
                    with lock:
                        results.append(rec)
                    i = current + 1
                    current = None
                elif "end" in rec:
                    i = len(items)
            code = proc.poll()
            if code is not None or (current is not None and time.monotonic() - since > timeout):
                if code is None:
                    proc.kill()
                    proc.wait()
                    status = "timeout"
                else:
                    status = "crash"
                if current is not None and i <= current:
                    it = dict(items[current])
                    it.pop("path")
                    it.update(loader="blender", status=status, ms=(time.monotonic() - since) * 1e3,
                              error=f"exit {code}" if code is not None else f"no result in {timeout}s")
                    with lock:
                        results.append(it)
                    i = current + 1
                # Drop any half-written line, so the next Blender's lines parse.
                with open(log, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                seen = len(lines)
                break
            time.sleep(0.05)
        stalls = stalls + 1 if i == start_i else 0
        if stalls >= 3:
            print(f"  shard {k}: Blender exits without progress at {i}; giving up", file=sys.stderr)
            return
        if i < len(items):
            print(f"  shard {k}: restarting at {i}/{len(items)}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("corpora", nargs="+")
    a = ap.parse_args()
    items = files(a.corpora)
    # Big files spread over the shards.
    items.sort(key=lambda x: -x["bytes"])
    shards = [items[k::a.jobs] for k in range(a.jobs)]
    results, lock = [], threading.Lock()
    t = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        threads = [threading.Thread(target=shard, args=(a.blender, s, a.timeout, results, lock, tmp, k)) for k, s in enumerate(shards)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{len(results)}/{len(items)} files in {time.monotonic() - t:.0f}s: {counts}")


if __name__ == "__main__":
    main()
