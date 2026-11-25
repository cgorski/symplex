#!/usr/bin/env python3
"""Compare a `truth` dry run (out/truth.sites.tsv) with review B's site list (data/callsites.tsv).

    python3 tools/compare_B.py out/truth.sites.tsv ../review_B_truth/data/callsites.tsv [--samples N]

Sites are matched by (file, line, query name).  Prints, per corpus group, a table of B's reading
class against the tool's rule, and lists the disagreements by kind.
"""
import collections
import csv
import sys

MAP = {  # B's reading -> the rule that should hold it
    "assert_eq_some_true": "truth.assert-eq-true", "assert_eq_some_false": "truth.assert-eq-false",
    "assert_eq_none": "truth.assert-eq-none", "assert_ne_some_true": "truth.assert-ne",
    "assert_ne_some_false": "truth.assert-ne", "assert_ne_none": "truth.assert-ne",
    "eq_some_true": "truth.cmp", "eq_some_false": "truth.cmp", "ne_some_true": "truth.cmp",
    "ne_some_false": "truth.cmp", "eq_none": "truth.cmp", "unwrap_or_false": "truth.unwrap-or",
    "is_none": "truth.is-none", "is_some": "truth.is-none", "match": "truth.match-arms",
    "matches_other": "truth.matches",
}
SEVEN = ("truth.assert-eq-true", "truth.assert-eq-false", "truth.assert-eq-none", "truth.assert-ne", "truth.cmp",
         "truth.unwrap-or", "truth.is-none")
USER = ("tests", "examples", "book", "README")


def main():
    mine_path, b_path = sys.argv[1], sys.argv[2]
    samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 10
    mine = collections.defaultdict(list)
    for r in csv.DictReader(open(mine_path), delimiter="\t", quoting=csv.QUOTE_NONE):
        corpus = {"readme": "README"}.get(r["corpus"], r["corpus"])
        mine[(r["file"], int(r["line"]), r["name"])].append((r["rule"], r["class"], r["status"], corpus, r["source"]))
    theirs = collections.defaultdict(list)
    for r in csv.DictReader(open(b_path), delimiter="\t", quoting=csv.QUOTE_NONE):
        theirs[(r["file"], int(r["line"]), r["query"])].append((r["reading"], r["corpus"]))

    both = b_only = m_only = 0
    agree = collections.Counter()
    b_only_rows = []
    m_only_rows = []
    differ = []
    for key in set(mine) | set(theirs):
        ms = list(mine.get(key, []))
        ts = list(theirs.get(key, []))
        ms = [m for m in ms if m[0] not in ("truth.rename", "truth.bool-query", "truth.printed-comment")]
        ts = [t for t in ts if t[1] not in ("src-code", "other")]  # the tool does not scan the library's own code
        while ms and ts:
            t = ts.pop(0)
            want0 = MAP.get(t[0])
            pick = next((k for k, m in enumerate(ms) if m[0] == want0), 0)
            m = ms.pop(pick)
            both += 1
            want = MAP.get(t[0])
            if want == m[0] or (want is None and m[0] in ("truth.other-reading", "truth.mention", "truth.printed-comment",
                                                          "truth.bool-query", "truth.rename")):
                agree[(t[1], "same")] += 1
            else:
                agree[(t[1], "class differs")] += 1
                differ.append((key, t, m))
        for t in ts:
            b_only += 1
            b_only_rows.append((key, t))
        for m in ms:
            if m[0] in ("truth.rename", "truth.bool-query", "truth.printed-comment"):
                continue  # rules outside B's locator (renames, bool queries, comments)
            m_only += 1
            m_only_rows.append((key, m))
    print("matched by (file, line, name): %d   only in B's list: %d   only in the tool's: %d" % (both, b_only, m_only))
    print("of the matched: %s" % dict(agree))

    def group(rows, f):
        c = collections.Counter(f(r) for r in rows)
        for k, n in c.most_common():
            print("   %5d  %s" % (n, k))

    print("\nonly in B's list, by corpus and B's reading:")
    group(b_only_rows, lambda r: (r[1][1], r[1][0]))
    print("\nonly in the tool's list, by corpus, rule and class:")
    group(m_only_rows, lambda r: (r[1][3], r[1][0], r[1][1]))
    print("\nmatched with another class (B's reading -> the tool's rule/class):")
    group(differ, lambda r: (r[1][1], r[1][0], r[2][0], r[2][1]))

    # the seven shapes on the user side
    user_b = collections.Counter()
    for key, ts in theirs.items():
        for t in ts:
            if t[1] in USER and MAP.get(t[0]) in SEVEN:
                user_b[MAP[t[0]]] += 1
    user_m = collections.Counter()
    user_m_rw = collections.Counter()
    for key, ms in mine.items():
        for m in ms:
            if m[3] in USER and m[0] in SEVEN:
                user_m[m[0]] += 1
                if m[2] == "rewrite":
                    user_m_rw[m[0]] += 1
    print("\nthe seven shapes, user side (tests + examples + book + README):")
    print("   %-24s %6s %6s %8s" % ("rule", "B", "tool", "rewrite"))
    for r in SEVEN:
        print("   %-24s %6d %6d %8d" % (r, user_b[r], user_m[r], user_m_rw[r]))
    print("   %-24s %6d %6d %8d" % ("total", sum(user_b.values()), sum(user_m.values()), sum(user_m_rw.values())))

    import random
    random.seed(17)
    for title, rows in (("only in B's list", b_only_rows), ("only in the tool's list", m_only_rows),
                        ("class differs", differ)):
        print("\nsample — %s:" % title)
        for r in random.sample(rows, min(samples, len(rows))):
            print("   %s:%d %s  %s" % (r[0][0], r[0][1], r[0][2], r[1:]))


if __name__ == "__main__":
    main()
