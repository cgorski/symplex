#!/usr/bin/env python3
"""Parallel driver for the `golden` harness (standard library only).

Splits the case range into slices, runs up to --jobs `golden` processes at a
time, and restarts a process after it stops early:

* exit 3 after a TIMEOUT line: resume with `--after <op>` (the next op of the
  same case);
* abort above the allocation cap (stderr `golden: OOM`): record the running
  (case, op) as OOM and resume after it;
* any other abnormal exit (signal, stack overflow, ...): record CRASH;
* no output for --stall-s seconds: kill, record TIMEOUT ("driver-kill").

The running (case, op) of a crashed process is the successor of its last
complete output line in the canonical op order (`golden --list-ops`).

Output: one TSV sorted by case id (op order within a case), plus
`<out>.meta.json` (tree, version, harness fingerprint, settings, runtime,
status counts), `<out>.summary.txt` (per-op status table) and
`<out>.panics.txt` (the PANIC reports with source locations, from stderr).
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SEEDS = "0..200000"
RESOURCE = ("TIMEOUT", "OOM")


def parse_range(s):
    a, b = s.split("..")
    return int(a), int(b)


def case_ids(args):
    if args.file:
        lo, hi = parse_range(args.seeds) if args.seeds else (0, 1 << 62)
        ids = []
        with open(args.file, encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                t = line.strip()
                if lo <= n < hi and t and not t.startswith("#"):
                    ids.append(n)
        return ids
    lo, hi = parse_range(args.seeds or DEFAULT_SEEDS)
    return list(range(lo, hi))


def run_proc(cmd, stall_s):
    """Run `cmd`; return (returncode, complete stdout lines, stderr, killed)."""
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL)
        last_size, last_change, killed = 0, time.monotonic(), False
        while True:
            try:
                rc = p.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            size = os.fstat(fo.fileno()).st_size
            now = time.monotonic()
            if size != last_size:
                last_size, last_change = size, now
            elif now - last_change > stall_s:
                p.kill()
                rc = p.wait()
                killed = True
                break
        fo.seek(0)
        out = fo.read().decode("utf-8", "replace")
        fe.seek(0)
        err = fe.read().decode("utf-8", "replace")
    lines = out.split("\n")[:-1]  # drop a partial last line
    return rc, lines, err, killed


def describe_exit(rc, err):
    lines = [l for l in err.splitlines() if l.strip() and not l.startswith("golden: PANIC")]
    if any("overflowed its stack" in l for l in lines):
        return "stack overflow"
    sig = ""
    if rc < 0:
        try:
            sig = signal.Signals(-rc).name
        except ValueError:
            sig = f"signal {-rc}"
    else:
        sig = f"exit {rc}"
    tail = lines[-1][:200] if lines else ""
    return f"{sig}: {tail}" if tail else sig


class Runner:
    def __init__(self, args, ops):
        self.args = args
        self.ops = ops
        self.stall_s = args.stall_s or (2 * args.timeout_ms / 1000 + 30)
        self.lock = threading.Lock()
        self.restarts = 0
        self.panics = []

    def base_cmd(self):
        a = self.args
        cmd = [a.bin, "--ops", a.ops, "--timeout-ms", str(a.timeout_ms), "--mem-mb", str(a.mem_mb)]
        if a.file:
            cmd += ["--file", a.file]
        return cmd

    def successor(self, ids, pos, after, last):
        """The (case, op) a process was running when it died."""
        if last is None:
            k = self.ops.index(after) + 1 if after else 0
            if k < len(self.ops):
                return ids[pos], self.ops[k]
            pos, k = pos + 1, 0
            return (ids[pos], self.ops[0]) if pos < len(ids) else (None, None)
        c, op = last.split("\t")[:2]
        k = self.ops.index(op) + 1
        if k < len(self.ops):
            return int(c), self.ops[k]
        p = ids.index(int(c), pos) + 1
        return (ids[p], self.ops[0]) if p < len(ids) else (None, None)

    def run_slice(self, ids):
        out, pos, after = [], 0, None
        hi = ids[-1] + 1
        while pos < len(ids):
            cmd = self.base_cmd() + ["--seeds", f"{ids[pos]}..{hi}"]
            if after:
                cmd += ["--after", after]
            rc, lines, err, killed = run_proc(cmd, self.stall_s)
            out.extend(lines)
            panics = [l for l in err.splitlines() if l.startswith("golden: PANIC")]
            if panics:
                with self.lock:
                    self.panics.extend(panics)
            if rc == 0 and not killed:
                break
            last = lines[-1] if lines else None
            if rc == 3 and not killed and last and last.split("\t")[2] == "TIMEOUT":
                c, op = last.split("\t")[:2]
                c = int(c)
            else:
                c, op = self.successor(ids, pos, after, last)
                if c is None:
                    break
                if killed:
                    status, desc = "TIMEOUT", f"driver-kill (no output for {self.stall_s:.0f} s)"
                elif "golden: OOM" in err:
                    status, desc = "OOM", f">{self.args.mem_mb} MiB"
                else:
                    status, desc = "CRASH", describe_exit(rc, err)
                out.append(f"{c}\t{op}\t{status}\t{desc}")
            with self.lock:
                self.restarts += 1
            pos = ids.index(c, pos)
            after = op
            if op == self.ops[-1]:
                pos, after = pos + 1, None
        return out


def file_digest(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def tree_info(bin_path):
    # <tree>/probes/golden/target/release/golden
    harness = Path(bin_path).resolve().parents[2]
    tree = harness.parents[1]
    version = None
    try:
        for line in (tree / "Cargo.toml").read_text().splitlines():
            if line.startswith("version"):
                version = line.split("=", 1)[1].strip().strip('"')
                break
    except OSError:
        pass
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    srcs = sorted((harness / "src").glob("*.rs")) + [harness / "Cargo.toml"]
    return {
        "tree": str(tree),
        "symplex_version": version,
        "git_commit": commit,
        "harness_fingerprint": file_digest([p for p in srcs if p.exists()]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(HERE / "target" / "release" / "golden"))
    ap.add_argument("--seeds", help=f"case range A..B (default {DEFAULT_SEEDS}; with --file: line numbers)")
    ap.add_argument("--file", help="one expression per line instead of generated cases")
    ap.add_argument("--ops", default="all")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--mem-mb", type=int, default=4096)
    ap.add_argument("-j", "--jobs", type=int, default=48)
    ap.add_argument("--slice", type=int, default=100, help="cases per slice (default 100)")
    ap.add_argument("--stall-s", type=float, help="kill a silent process after this long")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ops = subprocess.run([args.bin, "--list-ops", "--ops", args.ops], capture_output=True, text=True, check=True)
    ops = ops.stdout.split()
    info = dict(
        l.split("\t", 1)
        for l in subprocess.run([args.bin, "--info"], capture_output=True, text=True, check=True).stdout.splitlines()
    )
    fncheck = subprocess.run([args.bin, "--list-fns"], capture_output=True, text=True, check=True).stdout
    fnproblems = [l for l in fncheck.splitlines() if l.split("\t")[0] in ("UNCOVERED", "ARITY", "GONE")]
    for l in fnproblems:
        print(f"warning: function table: {l}", file=sys.stderr)

    ids = case_ids(args)
    if not ids:
        sys.exit("no cases")
    slices = [ids[i : i + args.slice] for i in range(0, len(ids), args.slice)]
    runner = Runner(args, ops)
    results = [None] * len(slices)
    t0 = time.monotonic()
    last_report = t0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(runner.run_slice, s): i for i, s in enumerate(slices)}
        done = 0
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            now = time.monotonic()
            if now - last_report > 30 or done == len(slices):
                last_report = now
                print(
                    f"[{now - t0:6.0f}s] {done}/{len(slices)} slices, {runner.restarts} restarts",
                    file=sys.stderr,
                    flush=True,
                )
    wall = time.monotonic() - t0

    lines = [l for r in results for l in r]
    lines.sort(key=lambda l: int(l.split("\t", 1)[0]))  # stable: op order kept
    with open(args.out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    counts = {}
    for l in lines:
        _, op, st = l.split("\t")[:3]
        counts.setdefault(op, {}).setdefault(st, 0)
        counts[op][st] += 1
    statuses = ["OK", "ERR", "PANIC", "TIMEOUT", "OOM", "CRASH"]
    table = [f"{'op':<14}" + "".join(f"{s:>9}" for s in statuses)]
    for op in ops:
        c = counts.get(op, {})
        table.append(f"{op:<14}" + "".join(f"{c.get(s, 0):>9}" for s in statuses))
    tot = {s: sum(c.get(s, 0) for c in counts.values()) for s in statuses}
    table.append(f"{'TOTAL':<14}" + "".join(f"{tot[s]:>9}" for s in statuses))
    summary = (
        f"{len(ids)} cases, {len(lines)} lines, {len(ops)} ops, wall {wall:.0f} s, "
        f"{args.jobs} jobs x {args.slice}-case slices, timeout {args.timeout_ms} ms, "
        f"{runner.restarts} restarts\n" + "\n".join(table) + "\n"
    )
    Path(args.out + ".summary.txt").write_text(summary)
    Path(args.out + ".panics.txt").write_text("".join(p + "\n" for p in sorted(set(runner.panics))))
    meta = tree_info(args.bin)
    meta.update(
        {
            "gen_version": info.get("gen_version"),
            "seeds": args.seeds or (None if args.file else DEFAULT_SEEDS),
            "file": args.file,
            "ops": ops,
            "timeout_ms": args.timeout_ms,
            "mem_mb": args.mem_mb,
            "jobs": args.jobs,
            "slice": args.slice,
            "cases": len(ids),
            "lines": len(lines),
            "wall_s": round(wall, 1),
            "restarts": runner.restarts,
            "status_totals": tot,
            "function_table_problems": fnproblems,
        }
    )
    Path(args.out + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(summary, end="")


if __name__ == "__main__":
    main()
