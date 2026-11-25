#!/usr/bin/env python3
"""Compare a reviewer's regular-expression count with the tool's sites, line by line.

    python3 tools/compare_regex.py NAME REGEX SITES.tsv RULE_PREFIX[,RULE_PREFIX..] [--samples N] [--status S]

Re-runs REGEX the way the reviewers' scripts did (per line of text, no tokenizing) over
tests/, examples/, book/src/, README.md and the doc-comment lines of src/, and matches its
hits to the tool's sites of the given rules by (file, line).  Prints both counts per corpus and
samples of the lines only one side has.
"""
import collections
import csv
import os
import random
import re
import sys

REPO = os.environ.get("SYMPLEX_REPO", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
CORPORA = [("tests", "tests", (".rs",)), ("examples", "examples", (".rs",)), ("book", "book/src", (".md",)),
           ("readme", "README.md", (".md",)), ("src-doc", "src", (".rs",))]
SKIP = ("target", ".git", "node_modules")


def files(root, exts):
    p = os.path.join(REPO, root)
    if os.path.isfile(p):
        yield p
        return
    for d, dirs, fs in os.walk(p):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in sorted(fs):
            if f.endswith(exts):
                yield os.path.join(d, f)


def main():
    name, regex, sites_path, prefixes = sys.argv[1:5]
    samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 10
    status = sys.argv[sys.argv.index("--status") + 1].split(",") if "--status" in sys.argv else None
    rx = re.compile(regex)
    is_doc = re.compile(r"^\s*//[/!]")
    theirs = collections.Counter()
    text_of = {}
    for corpus, root, exts in CORPORA:
        for path in files(root, exts):
            rel = os.path.relpath(path, REPO)
            with open(path, errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if corpus == "src-doc" and not is_doc.match(line):
                        continue
                    k = len(rx.findall(line))
                    if k:
                        theirs[(corpus, rel, n)] += k
                        text_of[(rel, n)] = line.strip()
    mine = collections.Counter()
    mine_rw = collections.Counter()
    pre = tuple(prefixes.split(","))
    for r in csv.DictReader(open(sites_path), delimiter="\t", quoting=csv.QUOTE_NONE):
        if not r["rule"].startswith(pre) or (status and r["status"] not in status):
            continue
        if r["corpus"] not in ("tests", "examples", "book", "readme", "src-doc"):
            continue
        key = (r["corpus"], r["file"], int(r["line"]))
        mine[key] += 1
        if r["status"] == "rewrite":
            mine_rw[key] += 1
        text_of.setdefault((r["file"], int(r["line"])), r["source"])
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    only_t, only_m = [], []
    for key in set(theirs) | set(mine):
        c = key[0]
        t, m = theirs.get(key, 0), mine.get(key, 0)
        per[c][0] += t
        per[c][1] += m
        per[c][2] += mine_rw.get(key, 0)
        if t > m:
            per[c][3] += t - m
            only_t.append(key)
        if m > t:
            per[c][4] += m - t
            only_m.append(key)
    print("== %s   regex: %s   rules: %s" % (name, regex, prefixes))
    print("   %-9s %7s %7s %9s %12s %11s" % ("corpus", "regex", "tool", "rewrite", "regex only", "tool only"))
    tot = [0, 0, 0, 0, 0]
    for c, _r, _e in CORPORA:
        v = per[c]
        print("   %-9s %7d %7d %9d %12d %11d" % (c, v[0], v[1], v[2], v[3], v[4]))
        tot = [a + b for a, b in zip(tot, v)]
    print("   %-9s %7d %7d %9d %12d %11d" % ("total", tot[0], tot[1], tot[2], tot[3], tot[4]))
    random.seed(5)
    for title, rows in (("regex only", only_t), ("tool only", only_m)):
        if rows and samples:
            print("   sample, %s:" % title)
            for key in random.sample(sorted(rows), min(samples, len(rows))):
                print("      %s:%d: %s" % (key[1], key[2], text_of.get((key[1], key[2]), "")[:130]))
    print()


if __name__ == "__main__":
    main()
