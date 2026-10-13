#!/usr/bin/env python3
"""Compare two golden TSVs (standard library only).

    golden_diff.py base.tsv new.tsv [-n 10] [--ops simplify,diff]
                   [--changed-out changed.tsv] [--allow-missing]
                   [--rerun-base BIN] [--rerun-new BIN] [--file F]

Per op: identical / changed / only-in-base / only-in-new / timeout-involved.
A pair where either side is TIMEOUT or OOM is counted as timeout-involved
and never as a change.  Prints the first N changed lines per op (case id,
input, base, new).  Exit status 0 iff nothing changed (and, unless
--allow-missing, no (case, op) is present in only one file); 1 otherwise.

Both files must be sorted by case id (the driver writes them so); they are
merged in one streaming pass.

--rerun-base / --rerun-new BIN re-run every changed (case, op) singly
(`BIN --seeds c..c+1 --ops op`, --rerun-times times) and report whether the
recorded line is reproduced: a change that does not reproduce is
nondeterminism, not a behaviour change.  Pass --file F if the TSVs came from
`--file` mode.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESOURCE = ("TIMEOUT", "OOM")


def groups(path):
    """Yield (case_id, {op: (status, result)}) in file order."""
    cur, items, prev = None, {}, -1
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) < 3:
                continue
            while len(parts) < 4:
                parts.append("")
            c = int(parts[0])
            if c != cur:
                if c < prev:
                    sys.exit(f"{path}:{n}: not sorted by case id ({c} after {prev})")
                if cur is not None:
                    yield cur, items
                cur, items, prev = c, {}, c
            items[parts[1]] = (parts[2], parts[3])
    if cur is not None:
        yield cur, items


def merged(base, new):
    """Yield (case_id, base_items or None, new_items or None)."""
    a, b = groups(base), groups(new)
    x, y = next(a, None), next(b, None)
    while x is not None or y is not None:
        if y is None or (x is not None and x[0] < y[0]):
            yield x[0], x[1], None
            x = next(a, None)
        elif x is None or y[0] < x[0]:
            yield y[0], None, y[1]
            y = next(b, None)
        else:
            yield x[0], x[1], y[1]
            x, y = next(a, None), next(b, None)


def show_meta(path):
    p = Path(str(path) + ".meta.json")
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    print(
        f"{path}: symplex {m.get('symplex_version')} commit {str(m.get('git_commit'))[:10]} "
        f"tree {m.get('tree')} gen v{m.get('gen_version')} harness {m.get('harness_fingerprint')} "
        f"timeout {m.get('timeout_ms')} ms"
    )
    return m


def clip(s, n=300):
    return s if len(s) <= n else s[:n] + "…"


def rerun(bin_path, case, op, file, timeout_ms):
    cmd = [bin_path, "--seeds", f"{case}..{case + 1}", "--ops", op, "--timeout-ms", str(timeout_ms)]
    if file:
        cmd += ["--file", file]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4 and parts[1] == op:
            return parts[2], parts[3]
    return "CRASH", f"exit {p.returncode}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("-n", type=int, default=10, help="changed lines to print per op (default 10)")
    ap.add_argument("--ops", help="only these ops (comma-separated)")
    ap.add_argument("--changed-out", help="write every changed pair here as TSV")
    ap.add_argument("--allow-missing", action="store_true", help="only-in-one pairs do not fail")
    ap.add_argument("--rerun-base", metavar="BIN")
    ap.add_argument("--rerun-new", metavar="BIN")
    ap.add_argument("--rerun-times", type=int, default=2)
    ap.add_argument("--rerun-max", type=int, default=500)
    ap.add_argument("--rerun-timeout-ms", type=int, default=30000)
    ap.add_argument("--file", help="the --file input, for reruns")
    ap.add_argument("-j", "--jobs", type=int, default=32)
    args = ap.parse_args()

    mb, mn = show_meta(args.base), show_meta(args.new)
    if mb and mn:
        if mb.get("gen_version") != mn.get("gen_version"):
            print("WARNING: different generator versions; the inputs differ", file=sys.stderr)
        if mb.get("harness_fingerprint") != mn.get("harness_fingerprint"):
            print("note: different harness sources (check `expr` lines)", file=sys.stderr)
    only = set(args.ops.split(",")) if args.ops else None

    keys = ["identical", "changed", "only_base", "only_new", "timeout"]
    stats, order, changed = {}, [], []
    expr_of = {}
    for case, bi, ni in merged(args.base, args.new):
        ops = list((bi or {}).keys()) + [o for o in (ni or {}) if o not in (bi or {})]
        ex = ((bi or {}).get("expr") or (ni or {}).get("expr") or ("", ""))[1]
        for op in ops:
            if only and op not in only:
                continue
            if op not in stats:
                stats[op] = dict.fromkeys(keys, 0)
                order.append(op)
            s = stats[op]
            b = bi.get(op) if bi else None
            n = ni.get(op) if ni else None
            if b is None:
                s["only_new"] += 1
            elif n is None:
                s["only_base"] += 1
            elif b[0] in RESOURCE or n[0] in RESOURCE:
                s["timeout"] += 1
            elif b == n:
                s["identical"] += 1
            else:
                s["changed"] += 1
                changed.append((case, op, b, n))
                expr_of[case] = ex

    print(f"\n{'op':<14}" + "".join(f"{k:>11}" for k in keys))
    tot = dict.fromkeys(keys, 0)
    for op in order:
        print(f"{op:<14}" + "".join(f"{stats[op][k]:>11}" for k in keys))
        for k in keys:
            tot[k] += stats[op][k]
    print(f"{'TOTAL':<14}" + "".join(f"{tot[k]:>11}" for k in keys))
    if stats.get("expr", {}).get("changed"):
        print("\nWARNING: `expr` lines differ: the two runs used different inputs", file=sys.stderr)

    by_op = {}
    for c in changed:
        by_op.setdefault(c[1], []).append(c)
    for op in order:
        lst = by_op.get(op, [])
        if not lst:
            continue
        print(f"\n== {op}: {len(lst)} changed" + (f" (first {args.n})" if len(lst) > args.n else "") + " ==")
        for case, _, b, n in lst[: args.n]:
            print(f"  case {case}  input: {clip(expr_of.get(case, ''), 200)}")
            print(f"    base {b[0]:<5} {clip(b[1])}")
            print(f"    new  {n[0]:<5} {clip(n[1])}")

    if args.changed_out:
        with open(args.changed_out, "w", encoding="utf-8") as f:
            f.write("case_id\top\tbase_status\tbase_result\tnew_status\tnew_result\tinput\n")
            for case, op, b, n in changed:
                f.write(f"{case}\t{op}\t{b[0]}\t{b[1]}\t{n[0]}\t{n[1]}\t{expr_of.get(case, '')}\n")

    for label, bin_path, side in (("base", args.rerun_base, 2), ("new", args.rerun_new, 3)):
        if not bin_path or not changed:
            continue
        todo = changed[: args.rerun_max]
        jobs = [(c, t) for c in todo for t in range(args.rerun_times)]
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            got = list(
                ex.map(lambda j: rerun(bin_path, j[0][0], j[0][1], args.file, args.rerun_timeout_ms), jobs)
            )
        flaky = []
        for (c, _), g in zip(jobs, got):
            rec = c[side]
            if g != rec and g[0] not in RESOURCE and rec[0] not in RESOURCE:
                flaky.append((c[0], c[1], rec, g))
        print(
            f"\nrerun {label} ({bin_path}): {len(todo)} changed pairs x {args.rerun_times}: "
            f"{len(flaky)} runs did NOT reproduce the recorded {label} line"
        )
        for case, op, rec, g in flaky[: args.n]:
            print(f"  case {case} {op}: recorded {rec[0]} {clip(rec[1], 150)} | rerun {g[0]} {clip(g[1], 150)}")

    failed = tot["changed"] > 0 or (not args.allow_missing and (tot["only_base"] or tot["only_new"]))
    print("\nRESULT: " + ("DIFFERENCES" if failed else "no changes"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
