#!/usr/bin/env python3
"""Idempotence on real files: apply each rule set to a scratch COPY, apply it again, compare.

    python3 tools/sandbox.py

`sandbox/orig/` holds copies of eight files of the repository (never the repository itself).
For every rule set the copy `sandbox/<ruleset>/once` gets one `--apply`, `sandbox/<ruleset>/twice`
gets two; the two trees must be byte-identical and the second run must report no rewrite.
`sandbox/all/` does the same with all seven rule sets applied in the order of the blocks.
"""
import contextlib
import filecmp
import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import codemod  # noqa: E402

ORIG = os.path.join(ROOT, "sandbox", "orig")
ORDER = ["numbers", "paths", "errors", "truth", "subs", "domains", "core-renames"]  # N3, S1, E2a/R1, T2, V2b, ST/LA, NM1


def files(d):
    out = []
    for dp, _dn, fn in os.walk(d):
        for f in sorted(fn):
            out.append(os.path.relpath(os.path.join(dp, f), d))
    return sorted(out)


def apply(ruleset, tree):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        codemod.main([ruleset, "--repo", tree, "--apply", "--quiet"])
    total = [ln for ln in buf.getvalue().splitlines() if ln.startswith("TOTAL")][0].split()
    return int(total[2]), int(total[3])  # rewrite, manual


def same(a, b):
    fa, fb = files(a), files(b)
    if fa != fb:
        return False
    return all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False) for f in fa)


def changed(a, b):
    return sum(1 for f in files(a) if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False))


def main():
    ok = True
    print("%-14s %8s %8s %14s %16s %10s" % ("rule set", "rewrite", "manual", "files changed", "2nd run rewrite", "identical"))
    for name in ORDER + ["all"]:
        once = os.path.join(ROOT, "sandbox", name, "once")
        twice = os.path.join(ROOT, "sandbox", name, "twice")
        for d in (once, twice):
            shutil.copytree(ORIG, d, dirs_exist_ok=True)
        sets = ORDER if name == "all" else [name]
        rw = man = 0
        for rs in sets:
            r, m = apply(rs, once)
            rw += r
            man = m if name != "all" else man + m
        shutil.copytree(once, twice, dirs_exist_ok=True)
        rw2 = 0
        for rs in sets:
            rw2 += apply(rs, twice)[0]
        identical = same(once, twice)
        ok = ok and identical and rw2 == 0
        print("%-14s %8d %8d %14d %16d %10s" % (name, rw, man, changed(ORIG, once), rw2, "yes" if identical else "NO"))
    print("orig files: %d" % len(files(ORIG)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
