#!/usr/bin/env python3
"""Whole-tree checks that need no compiler, all in memory (nothing is written anywhere):

  1. fixpoint   for every file a rule set changes, running the rule set on its OUTPUT changes nothing more;
  2. tokenizes  every changed Rust file still tokenizes with balanced brackets (and every changed
                doctest / Markdown block that tokenized before still does).

    python3 tools/fixpoint_check.py [--repo PATH] [RULESET ...]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cm import engine  # noqa: E402
from cm.rulesets import RULESETS  # noqa: E402
from cm.views import FileUnit  # noqa: E402


def bad_views(rel, text, corpus):
    u = FileUnit("/x/" + rel, rel, text, corpus)
    kinds = ["code", "doc"] if u.is_rust else ["md"]
    return sum(1 for v in u.views(kinds) if not v.lx.ok)


def main():
    args = sys.argv[1:]
    repo = os.environ.get("SYMPLEX_REPO", "")
    if "--repo" in args:
        k = args.index("--repo")
        repo = args[k + 1]
        del args[k:k + 2]
    names = args or list(RULESETS)
    roots = [r for r, _c in engine.DEFAULT_ROOTS if os.path.exists(os.path.join(repo, r))]
    failed = False
    print("%-14s %8s %10s %12s %14s" % ("rule set", "files", "rewrites", "not fixpoint", "broke a lexer"))
    for name in names:
        rs = RULESETS[name]
        cx = engine.Cx()
        results = engine.run(repo, roots, rs, cx)
        changed = [r for r in results if r.new != r.old]
        unstable = []
        broken = []
        for r in changed:
            again, rw, _man, _notes = engine.transform(r.new, os.path.join(repo, r.rel), r.rel, r.corpus, rs, cx)
            if again != r.new or rw:
                unstable.append(r.rel)
            if bad_views(r.rel, r.new, r.corpus) > bad_views(r.rel, r.old, r.corpus):
                broken.append(r.rel)
        print("%-14s %8d %10d %12d %14d" % (name, len(changed), sum(len(r.rewritten) for r in results),
                                           len(unstable), len(broken)))
        for f in unstable[:8]:
            print("    not a fixpoint: %s" % f)
        for f in broken[:8]:
            print("    no longer tokenizes: %s" % f)
        failed = failed or bool(unstable or broken)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
