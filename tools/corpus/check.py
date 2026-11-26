"""Checks that the generators still write the corpus that is checked in.

Each generator is run on a copy of tests/corpus in a scratch folder (through POLYSMELT_CORPUS, which
common.py reads), and every file it leaves different from the checked-in one, missing or new is
listed. Nothing under tests/corpus is written.

Run from the repository root:

    python tools/corpus/check.py                 # every generator
    python tools/corpus/check.py obj gltf        # only these
    python tools/corpus/check.py --path <dir>    # with packages installed by `pip install --target <dir>`

A generator is passed over, with the reason, when what it needs isn't there: a Python package
(numpy, Pillow, usd-core, vtk, meshio), Blender (`--blender <blender.exe>`, or `blender` on the
path) or the network (`--network`: khronos.py fetches the sample models it doesn't have in
target/khronos-samples). The exit code is 1 when a generator fails or writes something else, and 0
otherwise, passed over or not.

Generators are run as `python -I`, so nothing in the environment or the user's site packages
changes what they write; `--path` puts the packages they need in reach.
"""

import argparse
import ast
import filecmp
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
CORPUS = os.path.join(REPO, "tests", "corpus")

# Modules the generators share, and this script.
NOT_GENERATORS = {"common.py", "packed_splats.py", "check.py"}
# Run by Blender's own Python, once for every format it exports.
BLENDER = "blender_export.py"
# Fetches what its cache doesn't have.
NETWORK = {"khronos.py"}
# The packages to install for a module that can't be imported.
PACKAGES = {"PIL": "Pillow", "pxr": "usd-core"}


def imports(tree, inside_try=False):
    """The modules a syntax tree imports, each with whether it's only tried (inside a `try`)."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], inside_try
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], inside_try
        yield from imports(node, inside_try or isinstance(node, ast.Try))


def parse(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return ast.parse(f.read(), name)


def needs(name):
    """The modules generator `name` can't run without, the shared modules' included: of those, what the
    functions it takes from them import."""
    tree = parse(name)
    out = {module for module, tried in imports(tree) if not tried}
    for node in ast.walk(tree):
        shared = node.module + ".py" if isinstance(node, ast.ImportFrom) and node.module else None
        if shared in NOT_GENERATORS:
            taken = {alias.name for alias in node.names}
            for part in parse(shared).body:
                # A function's imports are needed only when it is called; the rest on import.
                if not isinstance(part, ast.FunctionDef) or part.name in taken:
                    out |= {module for module, tried in imports(ast.Module([part], [])) if not tried}
    return {m for m in out if m + ".py" not in NOT_GENERATORS}


def missing(name, paths):
    """The modules generator `name` needs that aren't installed."""
    saved = list(sys.path)
    # As `python -I` finds them, with `paths` in reach.
    sys.path[:] = paths + [p for p in saved if os.path.abspath(p or ".") != HERE]
    try:
        out = []
        for module in sorted(needs(name)):
            try:
                found = module in sys.builtin_module_names or importlib.util.find_spec(module)
            except (ImportError, ValueError):
                found = False
            if not found:
                out.append(PACKAGES.get(module, module))
        return out
    finally:
        sys.path[:] = saved


def files(root):
    """Every file under `root`, with its size and when it was written."""
    out = {}
    for folder, _, names in os.walk(root):
        for n in names:
            path = os.path.join(folder, n)
            stat = os.stat(path)
            out[os.path.relpath(path, root).replace(os.sep, "/")] = (stat.st_size, stat.st_mtime_ns)
    return out


def differences(scratch, before):
    """Every file a generator left different from the checked-in one, with how. `before` is the scratch
    corpus's files as they were, the same as the checked-in ones: only those written since are read."""
    after = files(scratch)
    out = [(f, "isn't written") for f in sorted(before.keys() - after.keys())]
    out += [(f, "is new") for f in sorted(after.keys() - before.keys())]
    for f in sorted(before.keys() & after.keys()):
        if before[f] != after[f] and not filecmp.cmp(os.path.join(CORPUS, f), os.path.join(scratch, f), shallow=False):
            out.append((f, "differs"))
    return out


def restore(scratch, changed):
    """Puts the scratch corpus back as the checked-in one is, so each generator starts from it."""
    for f, how in changed:
        to = os.path.join(scratch, *f.split("/"))
        if how == "is new":
            os.remove(to)
        else:
            os.makedirs(os.path.dirname(to), exist_ok=True)
            shutil.copyfile(os.path.join(CORPUS, *f.split("/")), to)


def command(name, args):
    script = os.path.join(HERE, name)
    if name == BLENDER:
        return [args.blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", script]
    if not args.path:
        return [sys.executable, "-I", script]
    # `-I` leaves PYTHONPATH out, so the packages' folders are put on the path by hand.
    start = ("import runpy, sys; sys.path[:0] = {!r}; sys.argv = [{!r}]; "
             "runpy.run_path({!r}, run_name='__main__')").format(args.path, script, script)
    return [sys.executable, "-I", "-c", start]


def main():
    parser = argparse.ArgumentParser(description="Check that the corpus generators write the checked-in corpus.")
    parser.add_argument("generators", nargs="*", help="generators to run (obj, or obj.py); all when none is given")
    parser.add_argument("--path", action="append", default=[], help="a folder of Python packages; may be repeated")
    parser.add_argument("--blender", default=shutil.which("blender"), help="the Blender to run blender_export.py")
    parser.add_argument("--network", action="store_true", help="run the generators that fetch files")
    parser.add_argument("--keep", action="store_true", help="leave the scratch corpus, and say where it is")
    args = parser.parse_args()
    args.path = [os.path.abspath(p) for p in args.path]

    every = sorted(n for n in os.listdir(HERE) if n.endswith(".py") and n not in NOT_GENERATORS)
    wanted = [g if g.endswith(".py") else g + ".py" for g in args.generators] or every
    unknown = [g for g in wanted if g not in every]
    if unknown:
        parser.error("no such generator: " + ", ".join(unknown))

    top = tempfile.mkdtemp(prefix="polysmelt-corpus-")
    scratch = os.path.join(top, "corpus")
    shutil.copytree(CORPUS, scratch)
    env = dict(os.environ, POLYSMELT_CORPUS=scratch)
    same, passed_over, wrong = [], [], []
    try:
        for name in wanted:
            if name == BLENDER and not args.blender:
                why = "needs Blender (--blender <blender.exe>)"
            elif name in NETWORK and not args.network:
                why = "needs the network (--network)"
            elif name != BLENDER and missing(name, args.path):
                why = "needs " + ", ".join(missing(name, args.path)) + " (pip install, or --path <dir>)"
            else:
                why = None
            if why:
                print(f"{name}: passed over, {why}", flush=True)
                passed_over.append(name)
                continue
            before = files(scratch)
            run = subprocess.run(command(name, args), cwd=REPO, env=env, capture_output=True, text=True,
                                 errors="replace")
            changed = differences(scratch, before)
            if run.returncode != 0:
                print(f"{name}: failed with exit code {run.returncode}")
                print("    " + "\n    ".join((run.stderr or run.stdout).strip().splitlines()[-12:]))
            for f, how in changed:
                print(f"{name}: {f} {how}")
            if run.returncode != 0 or changed:
                wrong.append(name)
            else:
                print(f"{name}: writes what is checked in")
                same.append(name)
            sys.stdout.flush()
            restore(scratch, changed)
    finally:
        if args.keep:
            print("the scratch corpus is in", scratch)
        else:
            shutil.rmtree(top, ignore_errors=True)
    print(f"{len(same)} the same, {len(wrong)} not, {len(passed_over)} passed over")
    if wrong:
        print("not the same:", ", ".join(wrong))
    if passed_over:
        print("passed over:", ", ".join(passed_over))
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
