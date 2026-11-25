#!/usr/bin/env python3
"""codemod.py — the rewrite toolkit of the symplex 1.0 API work (package WP-05).

    python3 codemod.py list                         the rule sets and their rules
    python3 codemod.py RULESET [ROOT ...]           dry run (the default): report, change nothing
    python3 codemod.py RULESET --diff out.diff      ... and write the unified diff of what would change
    python3 codemod.py RULESET --apply [ROOT ...]   rewrite the files in place

RULESET is one of: truth, errors, numbers, core-renames, subs, paths, domains.
ROOT is a file or directory, relative to --repo; without one the default roots are used
(tests, examples, benches, book/src, README.md, src, fuzz and the side crates; in `src`
only the doctests in doc comments are touched unless --src-code is given).

Python 3, standard library only.  See README.md for how a package uses the tool and
for what a text-level rewrite can and cannot do.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cm import engine  # noqa: E402
from cm.engine import REWRITE, MANUAL, INFO  # noqa: E402
from cm.rulesets import RULESETS  # noqa: E402

DEFAULT_REPO = os.environ.get("SYMPLEX_REPO", "")


def find_repo(start):
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "Cargo.toml")) and os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent


def summarize(results, ruleset):
    """{rule id: Counter(status)} and the same per corpus."""
    per_rule = collections.OrderedDict((r.id, collections.Counter()) for r in ruleset.rules)
    per_corpus = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for fr in results:
        for s in fr.rewritten:
            per_rule.setdefault(s.rule, collections.Counter())[REWRITE] += 1
            per_corpus[s.rule][_corpus(fr, s)][REWRITE] += 1
        for s in fr.manual:
            per_rule.setdefault(s.rule, collections.Counter())[s.status] += 1
            per_corpus[s.rule][_corpus(fr, s)][s.status] += 1
    return per_rule, per_corpus


def _corpus(fr, s):
    if fr.corpus == "src":
        return "src-doc" if s.kind in ("doc", "text") else "src-code"
    return fr.corpus


def print_report(results, ruleset, out, verbose_manual=True, max_manual=0):
    per_rule, per_corpus = summarize(results, ruleset)
    w = out.write
    w("rule set `%s` — %s\n" % (ruleset.name, ruleset.description))
    w("%-34s %7s %9s %8s %10s\n" % ("rule", "found", "rewrite", "manual", "unchanged"))
    tot = collections.Counter()
    for rid, c in per_rule.items():
        found = c[REWRITE] + c[MANUAL] + c[INFO]
        w("%-34s %7d %9d %8d %10d\n" % (rid, found, c[REWRITE], c[MANUAL], c[INFO]))
        tot.update(c)
    w("%-34s %7d %9d %8d %10d\n" % ("TOTAL", sum(tot.values()), tot[REWRITE], tot[MANUAL], tot[INFO]))
    w("  found = rewrite + manual + unchanged.  manual: a person edits it (listed below with the reason).\n")
    w("  unchanged: counted, but nothing has to be edited for the site to compile.\n\n")
    corpora = sorted({c for d in per_corpus.values() for c in d})
    if corpora:
        w("by corpus (found / rewrite):\n")
        w("%-34s" % "rule" + "".join("%14s" % c for c in corpora) + "\n")
        for rid in per_rule:
            d = per_corpus.get(rid, {})
            if not d:
                continue
            w("%-34s" % rid + "".join(
                "%14s" % ("%d/%d" % (sum(d[c].values()), d[c][REWRITE]) if c in d else "·") for c in corpora) + "\n")
        w("\n")
    files = sum(1 for fr in results if fr.new != fr.old)
    w("files that would change: %d\n" % files)
    notes = [n for fr in results for n in fr.notes]
    if notes:
        w("\nnotes (%d):\n" % len(notes))
        for n in notes[:200]:
            w("  " + n + "\n")
        if len(notes) > 200:
            w("  ... %d more\n" % (len(notes) - 200))
    if verbose_manual:
        man = [(fr.rel, s) for fr in results for s in fr.manual if s.status == MANUAL]
        w("\nNOT rewritten — for a person (%d sites):\n" % len(man))
        by_reason = collections.Counter((s.rule, s.reason) for _rel, s in man)
        for (rule, reason), n in by_reason.most_common():
            w("  %5d  %-26s %s\n" % (n, rule, reason))
        w("\n")
        shown = 0
        for rel, s in sorted(man, key=lambda x: (x[1].rule, x[0], x[1].line)):
            if max_manual and shown >= max_manual:
                w("  ... %d more (use --sites FILE for the full list)\n" % (len(man) - shown))
                break
            w("  %s:%d: [%s%s] %s\n      %s\n" % (rel, s.line, s.rule, "~" if s.later_pass else "", s.reason, s.snippet))
            shown += 1


def write_sites(results, path):
    """One line per site: status, rule, class, name, file, line, reason, source line."""
    with open(path, "w") as f:
        f.write("status\trule\tclass\tname\tcorpus\tfile\tline\tkind\treason\tsource\n")
        for fr in results:
            for s in fr.rewritten + fr.manual:
                f.write("\t".join([s.status, s.rule, s.cls, s.name, _corpus(fr, s), fr.rel, str(s.line), s.kind,
                                   " ".join(s.reason.split()), " ".join(s.snippet.split())]) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ruleset", help="a rule set, or `list`")
    ap.add_argument("roots", nargs="*", help="files or directories (default: the repository's user-side roots)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="the symplex checkout (default: $SYMPLEX_REPO, or the "
                                                         "repository that contains this script)")
    ap.add_argument("--apply", action="store_true", help="write the rewritten files (default: dry run)")
    ap.add_argument("--diff", metavar="FILE", help="write the unified diff to FILE")
    ap.add_argument("--sites", metavar="FILE", help="write every site (TSV) to FILE")
    ap.add_argument("--src-code", action="store_true", help="in src/, also rewrite the library's own code "
                                                            "(only rules made for it)")
    ap.add_argument("--only", metavar="KINDS", help="comma-separated view kinds: code,doc,md")
    ap.add_argument("--corpus", metavar="NAME", help="treat every file as this corpus (for files outside a repository)")
    ap.add_argument("--rules", metavar="IDS", help="only the rules whose id starts with one of these "
                                                   "comma-separated prefixes")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="a rule option (README)")
    ap.add_argument("--quiet", action="store_true", help="do not list the sites left for a person")
    ap.add_argument("--max-manual", type=int, default=0, help="list at most this many manual sites")
    a = ap.parse_args(argv)

    if a.ruleset == "list":
        for name, rs in RULESETS.items():
            print("%s  (%d rules; package %s) — %s" % (name, len(rs.rules), rs.package, rs.description))
            for r in rs.rules:
                print("    %-30s %s" % (r.id, r.description))
            print("    needs: %s\n" % rs.needs)
        return 0
    if a.ruleset not in RULESETS:
        ap.error("unknown rule set %r (try `list`)" % a.ruleset)
    ruleset = RULESETS[a.ruleset]
    if a.rules:
        prefixes = tuple(a.rules.split(","))
        chosen = [r for r in ruleset.rules if r.id.startswith(prefixes)]
        if not chosen:
            ap.error("no rule of `%s` starts with %s" % (a.ruleset, prefixes))
        ruleset = engine.RuleSet(ruleset.name, ruleset.description, chosen, ruleset.needs, ruleset.package)
    repo = a.repo or find_repo(os.path.dirname(os.path.abspath(__file__))) or os.getcwd()
    roots = a.roots or [r for r, _c in engine.DEFAULT_ROOTS if os.path.exists(os.path.join(repo, r))]
    options = dict(kv.split("=", 1) for kv in a.opt)
    cx = engine.Cx(options)
    only = set(a.only.split(",")) if a.only else None
    results = engine.run(repo, roots, ruleset, cx, src_code=a.src_code, only=only, corpus_override=a.corpus)
    print_report(results, ruleset, sys.stdout, verbose_manual=not a.quiet, max_manual=a.max_manual)
    if a.diff:
        with open(a.diff, "w") as f:
            f.write(engine.unified_diff(results))
        print("diff written to %s" % a.diff)
    if a.sites:
        write_sites(results, a.sites)
        print("sites written to %s" % a.sites)
    if a.apply:
        n = 0
        for fr in results:
            if fr.new != fr.old:
                with open(os.path.join(repo, fr.rel), "w", encoding="utf-8") as f:
                    f.write(fr.new)
                n += 1
        print("APPLIED: %d files rewritten under %s" % (n, repo))
    else:
        print("dry run: nothing was written (use --apply to rewrite)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
