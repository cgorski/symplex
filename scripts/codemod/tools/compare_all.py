#!/usr/bin/env python3
"""The remaining count comparisons of DRYRUN.md (reviews C, D, E, F, A), from the site lists in out/.

    python3 tools/compare_all.py > out/compare_all.txt
"""
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
API = os.path.dirname(ROOT)
REPO = os.environ.get("SYMPLEX_REPO", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
USER = ("tests", "examples", "book", "readme")
csv.field_size_limit(sys.maxsize)


def tsv(path):
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))


def sites(name):
    return [r for r in tsv(os.path.join(ROOT, "out", name + ".sites.tsv")) if r.get("rule") and r.get("corpus")]


def corpus_lines():
    """{corpus: [(rel, line_no, text)]} for the user corpora and the doc-comment lines of src/."""
    out = collections.defaultdict(list)
    is_doc = re.compile(r"^\s*//[/!]")
    for corpus, root, exts in (("tests", "tests", (".rs",)), ("examples", "examples", (".rs",)),
                               ("book", "book/src", (".md",)), ("readme", "README.md", (".md",)),
                               ("src-doc", "src", (".rs",))):
        p = os.path.join(REPO, root)
        paths = [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _dn, fs in os.walk(p) for f in sorted(fs)
                                              if f.endswith(exts) and "/target" not in d]
        for path in sorted(paths):
            rel = os.path.relpath(path, REPO)
            with open(path, errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if corpus == "src-doc" and not is_doc.match(line):
                        continue
                    out[corpus].append((rel, n, line))
    return out


LINES = corpus_lines()
ALL5 = ("tests", "examples", "book", "readme", "src-doc")


def regex_count(rx, corpora=ALL5):
    r = re.compile(rx)
    return sum(len(r.findall(t)) for c in corpora for (_f, _n, t) in LINES[c])


def mine(rows, rule, corpora=ALL5):
    c = collections.Counter()
    for r in rows:
        if r["rule"] == rule and r["corpus"] in corpora:
            c[r["status"]] += 1
    return c


def table_renames(title, table, site_rows):
    print("== %s: per rule, the reviewers' kind of count (regex `.old(` / `::old(` on text lines) and the tool's" % title)
    print("   %-34s %-28s %6s %6s %8s %7s" % ("rule", "old name", "regex", "found", "rewrite", "manual"))
    tot = [0, 0, 0, 0]
    for row in tsv(os.path.join(ROOT, "tables", table + ".tsv")):
        old = row["old"]
        if row["shape"] in ("type", "manual-type"):
            rx = r"\b%s\b" % re.escape(old)
        elif row["shape"] in ("free", "manual-free"):
            rx = r"(?<![A-Za-z0-9_.])%s\(" % re.escape(old)
        else:
            rx = r"(?:\.|::)%s\(" % re.escape(old)
        n = regex_count(rx)
        m = mine(site_rows, row["id"])
        found = sum(m.values())
        print("   %-34s %-28s %6d %6d %8d %7d" % (row["id"], old, n, found, m["rewrite"], m["manual"]))
        tot = [tot[0] + n, tot[1] + found, tot[2] + m["rewrite"], tot[3] + m["manual"]]
    print("   %-34s %-28s %6d %6d %8d %7d\n" % ("total", "", tot[0], tot[1], tot[2], tot[3]))


def errors():
    rows = sites("errors")
    c_rows = tsv(os.path.join(API, "review_C_refusals", "data", "error_users.tsv"))
    print("== C: occurrences of SymplexError::<Variant> (review C data/error_users.tsv) against the tool")
    where_map = {"tests": "tests", "examples": "examples", "book": "book", "README": "readme", "src-doc": "src-doc",
                 "fuzz": "fuzz"}
    theirs = collections.defaultdict(collections.Counter)
    their_keys = collections.Counter()
    for r in c_rows:
        w = where_map.get(r["where"])
        if w is None:
            continue
        theirs[w][r["use"]] += 1
        their_keys[(r["file"], int(r["line"]), r["variant"])] += 1
    ours = collections.defaultdict(collections.Counter)
    our_keys = collections.Counter()
    for r in rows:
        if r["rule"] in ("errors.matches", "errors.match-arm", "errors.other", "errors.construct"):
            kind = {"errors.matches": "tests-only", "errors.match-arm": "tests-only"}.get(r["rule"], r["class"])
            ours[r["corpus"]][(kind, r["status"])] += 1
            our_keys[(r["file"], int(r["line"]), r["name"])] += 1
    print("   %-9s | C: %-13s %-13s %-12s | tool: %-9s %-9s %-13s %-12s %-12s" % (
        "corpus", "pattern-tests", "pattern-binds", "construction", "rewrite", "by hand", "binds (hand)",
        "construction", "bare pattern"))
    for w in ("tests", "examples", "book", "readme", "fuzz", "src-doc"):
        t, o = theirs[w], ours[w]
        print("   %-9s |    %-13d %-13d %-12d |       %-9d %-9d %-13d %-12d %-12d" % (
            w, t["pattern-tests"], t["pattern-binds"], t["construction"], o[("tests-only", "rewrite")],
            o[("tests-only", "manual")], o[("pattern-binds", "manual")], o[("construction", "manual")],
            o[("pattern-bare", "manual")] + o[("pattern", "manual")]))
    only_c = sum((their_keys - our_keys).values())
    print("   by (file, line, variant): C has %d occurrences; the tool has a site on the same line for all but %d "
          "(one site of the tool can hold several variants of one pattern)\n" % (sum(their_keys.values()), only_c))
    for (f, l, v), n in list((their_keys - our_keys).items())[:12]:
        print("      C only: %s:%d %s" % (f, l, v))
    # the pairs
    pairs = tsv(os.path.join(API, "review_C_refusals", "data", "try_pairs_060.tsv"))

    def num(s):
        m = re.search(r"\d+", s or "")
        return int(m.group()) if m else 0
    by = collections.Counter()
    for r in pairs:
        handled = r["class"] in ("b", "cV", "cP") and r["container"] != "Distribution" and r["try"] not in (
            "try_symbol", "try_replace")
        key = "handled by errors.*" if handled else ("class a/cT/cI, Distribution, try_symbol, try_replace")
        by[(key, "try")] += num(r["try_calls_user"])
        if not r["plain_calls_user"].startswith("unchanged"):
            by[(key, "plain")] += num(r["plain_calls_user"])
    print("\n== C: the pairs (data/try_pairs_060.tsv, user-side calls counted by C's regular expressions)")
    for k in sorted(by):
        print("   %-52s %-6s %5d" % (k[0], k[1], by[k]))
    all5 = ALL5 + ("fuzz", "benches", "probes", "wasm", "build", "rubi")
    for rid in ("errors.try-rename", "errors.plain-unwrap", "errors.doc-link"):
        m = mine(rows, rid, all5)
        print("   tool %-20s found %5d  rewrite %5d  manual %4d" % (rid, sum(m.values()), m["rewrite"], m["manual"]))
    print("   C: mentions of a variant in doc comments: 825 (review C §3.1); the tool's rule takes only the "
          "intra-doc link form [`SymplexError::V`]\n")


def numbers():
    rows = sites("numbers")
    print("== D: the number types on the user side (tests, examples, benches, book/src, README, fuzz)")
    user = ("tests", "examples", "benches", "book", "readme", "fuzz")
    files = set(r["file"] for r in rows if r["corpus"] in user)
    changed = set(r["file"] for r in rows if r["corpus"] in user and r["status"] == "rewrite")
    print("   files with a site: %d (changed: %d)        D: 227 files name a num type or `Q`" % (len(files), len(changed)))
    for label, pred, d in (
        ("`Ratio::new` / `Q::new` / `BigRational::new` listed", lambda r: r["rule"] == "numbers.ratio-new" and r["name"] == "new", "88"),
        ("`parse_bytes`", lambda r: r["name"] == "parse_bytes", "7"),
        ("`.to_integer()`", lambda r: r["name"] == "to_integer", "4"),
        ("`modpow`, `extended_gcd`, `magnitude`, `to_biguint`", lambda r: r["name"] in ("modpow", "extended_gcd", "magnitude", "to_biguint"), "8"),
        ("`as_ratio_parts`", lambda r: r["name"] == "as_ratio_parts", "4"),
        ("`Q` as a type name (tokens)", lambda r: r["rule"] == "numbers.rational" and r["name"] == "Q", "about 662 in 84 files"),
        ("`BigInt` tokens renamed", lambda r: r["rule"] == "numbers.bigint" and r["name"] == "BigInt", "552 `BigInt::from` + 43 + 15 + types"),
        ("`use` items of the num crates rewritten", lambda r: r["class"] == "import", "92 + 54 + 53 + 15 + 6 files import a num crate"),
    ):
        n = sum(1 for r in rows if r["corpus"] in user and pred(r))
        print("   %-58s tool %5d    D: %s" % (label, n, d))
    fi = regex_count(r"\b(?:Ratio|Q|BigRational)::from_integer\(", ("tests", "examples", "book", "readme"))
    print("   (`from_integer` by D's regex on tests+examples+book+README today: %d)" % fi)
    for rid in ("numbers.bigint", "numbers.rational", "numbers.trait-imports", "numbers.complex-import",
                "numbers.ratio-new", "numbers.by-hand", "numbers.renames"):
        m = mine(rows, rid, user)
        print("   tool %-24s user side: found %5d  rewrite %5d  manual %4d" % (rid, sum(m.values()), m["rewrite"], m["manual"]))
    print()


def paths():
    rows = sites("paths")
    print("== A: paths.  A's counts: 743 written paths outside src/ that move or go (tests 545, examples 44, book and")
    print("   README 104, sibling crates 50), 456 in doc comments of src/; prelude: 417 imports to add in 247 of 570 test")
    print("   files, 85 in 23 examples, 17 in 6 sibling files, 208 (upper bound, prose included) in book and README")
    sib = ("benches", "fuzz", "wasm", "build", "rubi", "probes", "macros")
    for label, cs in (("tests", ("tests",)), ("examples", ("examples",)), ("book + README", ("book", "readme")),
                      ("sibling crates", sib), ("src doc comments", ("src-doc",))):
        w = mine(rows, "paths.written", cs)
        u = mine(rows, "paths.use", cs)
        p = mine(rows, "paths.prelude-names", cs)
        files = len(set(r["file"] for r in rows if r["rule"] == "paths.prelude-names" and r["corpus"] in cs))
        leaves = sum(int(r["name"]) for r in rows if r["rule"] == "paths.use" and r["status"] == "rewrite"
                     and r["corpus"] in cs and r["name"].isdigit())
        print("   %-17s written: found %4d rewrite %4d | use items: rewrite %4d (%4d paths in them), listed %4d | "
              "prelude names: %4d imports (%3d rewrite) in %3d files" % (
                  label, sum(w.values()), w["rewrite"], u["rewrite"], leaves, u["manual"], sum(p.values()),
                  p["rewrite"], files))
    print()


def main():
    errors()
    numbers()
    table_renames("E core renames", "core_renames", sites("core-renames") + sites("subs"))
    table_renames("F domain renames", "domain_renames", sites("domains"))
    paths()


if __name__ == "__main__":
    main()
