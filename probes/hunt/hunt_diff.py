#!/usr/bin/env python3
"""Compare two hunt.sh outputs (standard library only).

    hunt_diff.py base.tsv new.tsv [--factor 2] [--min-ms 50] [-n 10]
                 [--ops a,b,c] [--changed-out changed.tsv]

Every (case, op) present in both files is put into one kind:

    same                 status and result are identical
    value -> value       OK on both sides, another result
    value -> refusal     OK, now ERR
    refusal -> value     ERR, now OK
    refusal -> refusal   ERR on both sides, another message
    -> PANIC/CRASH       now a panic or a crash (whatever it was)
    PANIC/CRASH ->       was a panic or a crash, now a value or a refusal
    -> TIMEOUT/OOM       now over the time or memory limit
    TIMEOUT/OOM ->       was over a limit, now a value or a refusal

and, independently, a pair is `slower` when both sides have a CPU time, the
new one is more than --factor times the base one and at least --min-ms (the
floor keeps timer noise out).  The table also gives the CPU seconds per op
on each side, over the pairs timed on both.

Prints the table, then the first -n pairs of every kind per op.  Exit status
0 iff every pair is `same` (slower pairs and pairs present on one side only
are reported and do not change it).  Works on golden TSVs too (no CPU
column: no `slower`).
"""

import argparse
import sys

KINDS = [
    "value -> value",
    "value -> refusal",
    "refusal -> value",
    "refusal -> refusal",
    "-> PANIC/CRASH",
    "PANIC/CRASH ->",
    "-> TIMEOUT/OOM",
    "TIMEOUT/OOM ->",
]
SHORT = ["val->val", "val->ref", "ref->val", "ref->ref", "->PANIC", "PANIC->", "->TIMEOUT", "TIMEOUT->"]
BROKEN = ("PANIC", "CRASH")
RESOURCE = ("TIMEOUT", "OOM")


def load(path):
    """{(case, op): (status, result, cpu_ms or None)} in file order, and {case: input}."""
    rows, exprs = {}, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            p += [""] * (5 - len(p))
            try:
                cpu = float(p[4])
            except ValueError:
                cpu = None
            case = int(p[0])
            if p[1] == "expr":
                exprs[case] = p[3]
            rows[(case, p[1])] = (p[2], p[3], cpu)
    return rows, exprs


def kind_of(b, n):
    if b[:2] == n[:2]:
        return None
    if n[0] in BROKEN:
        return "-> PANIC/CRASH"
    if n[0] in RESOURCE:
        return "-> TIMEOUT/OOM"
    if b[0] in BROKEN:
        return "PANIC/CRASH ->"
    if b[0] in RESOURCE:
        return "TIMEOUT/OOM ->"
    if b[0] == "OK":
        return "value -> value" if n[0] == "OK" else "value -> refusal"
    return "refusal -> value" if n[0] == "OK" else "refusal -> refusal"


def clip(s, n=160):
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--factor", type=float, default=2.0, help="slower: new CPU > FACTOR x base CPU (default 2)")
    ap.add_argument("--min-ms", type=float, default=50.0, help="slower: and new CPU >= this (default 50 ms)")
    ap.add_argument("-n", type=int, default=10, help="pairs to print per op and kind (default 10)")
    ap.add_argument("--ops", help="only these ops (comma-separated)")
    ap.add_argument("--changed-out", help="write every changed or slower pair here as TSV")
    args = ap.parse_args()

    base, exprs = load(args.base)
    new, exprs_new = load(args.new)
    only = set(args.ops.split(",")) if args.ops else None
    ops, stats, found = [], {}, []
    missing = [0, 0]
    for key in list(base) + [k for k in new if k not in base]:
        case, op = key
        if only and op not in only:
            continue
        if op not in stats:
            ops.append(op)
            stats[op] = {"pairs": 0, "same": 0, "slower": 0, "cpu_b": 0.0, "cpu_n": 0.0, **dict.fromkeys(KINDS, 0)}
        b, n = base.get(key), new.get(key)
        if b is None or n is None:
            missing[0 if n is None else 1] += 1
            continue
        s = stats[op]
        s["pairs"] += 1
        kind = kind_of(b, n)
        s[kind or "same"] += 1
        slower = None
        if b[2] is not None and n[2] is not None:
            s["cpu_b"] += b[2]
            s["cpu_n"] += n[2]
            if n[2] >= args.min_ms and n[2] > args.factor * b[2]:
                slower = f"{b[2]:.1f} ms -> {n[2]:.1f} ms (x{n[2] / max(b[2], 1e-3):.1f})"
                s["slower"] += 1
        if kind or slower:
            found.append((case, op, kind, slower, b, n))

    if exprs != exprs_new:
        print("WARNING: the `expr` lines differ: the two runs did not have the same inputs", file=sys.stderr)
    head = f"{'op':<13}{'pairs':>7}{'same':>7}" + "".join(f"{k:>10}" for k in SHORT) + f"{'slower':>8}{'cpu s base':>12}{'cpu s new':>11}"
    print(head)
    tot = None
    for op in ops:
        s = stats[op]
        if op == "expr" and s["same"] == s["pairs"]:
            continue
        print(
            f"{op:<13}{s['pairs']:>7}{s['same']:>7}" + "".join(f"{s[k]:>10}" for k in KINDS)
            + f"{s['slower']:>8}{s['cpu_b'] / 1000:>12.2f}{s['cpu_n'] / 1000:>11.2f}"
        )
        tot = {k: (tot[k] if tot else 0) + v for k, v in s.items()}
    if tot:
        print(
            f"{'TOTAL':<13}{tot['pairs']:>7}{tot['same']:>7}" + "".join(f"{tot[k]:>10}" for k in KINDS)
            + f"{tot['slower']:>8}{tot['cpu_b'] / 1000:>12.2f}{tot['cpu_n'] / 1000:>11.2f}"
        )
    if missing[0] or missing[1]:
        print(f"only in base: {missing[0]} pairs; only in new: {missing[1]} pairs")

    for op in ops:
        for kind in KINDS + ["slower"]:
            lst = [c for c in found if c[1] == op and (c[3] if kind == "slower" else c[2] == kind)]
            if not lst:
                continue
            more = f" (first {args.n})" if len(lst) > args.n else ""
            print(f"\n== {op}: {kind}: {len(lst)}{more} ==")
            for case, _, _, slower, b, n in lst[: args.n]:
                print(f"  case {case}  input: {clip(exprs.get(case, ''))}")
                if kind == "slower":
                    print(f"    cpu  {slower}")
                else:
                    print(f"    base {b[0]:<5} {clip(b[1])}")
                    print(f"    new  {n[0]:<5} {clip(n[1])}")

    if args.changed_out:
        with open(args.changed_out, "w", encoding="utf-8") as f:
            f.write("case_id\top\tkind\tslower\tbase_status\tbase_result\tnew_status\tnew_result\tinput\n")
            for case, op, kind, slower, b, n in found:
                f.write(f"{case}\t{op}\t{kind or ''}\t{slower or ''}\t{b[0]}\t{b[1]}\t{n[0]}\t{n[1]}\t{exprs.get(case, '')}\n")

    changed = sum(1 for c in found if c[2])
    print(f"\nRESULT: {changed} changed pairs, {sum(1 for c in found if c[3])} slower" if found else "\nRESULT: no changes")
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
