#!/usr/bin/env python3
"""Run symplex's value dump for the stats oracle in parallel shards.

    python3 scripts/stats_oracle/run_dump.py --bin target/release/deps/v36-<hash> \
        --cases scripts/stats_oracle/cases --out target/stats_oracle/dump.tsv [--jobs 4]

`--bin` is the `v36` integration-test binary (built with `cargo test --release
--test v36 --no-run`; `--bin auto` asks cargo for its path).  Each shard runs
the test `v36_s15l5_numerics::oracle::stats_oracle_dump` over every `jobs`-th
row; a row that hangs (the dumper writes `HANG` and exits 3) or crashes the
process (an abort, an allocation failure) is recorded and the shard resumes
after it.  Standard library only.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

TEST = "v36_s15l5_numerics::oracle::stats_oracle_dump"


def find_binary(root):
    """The release `v36` test binary, from cargo's JSON messages."""
    out = subprocess.run(
        ["cargo", "test", "--release", "--test", "v36", "--no-run", "--message-format=json"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") and \
                msg.get("target", {}).get("name") == "v36":
            return msg["executable"]
    sys.exit("run_dump.py: cargo did not report the v36 test binary")


def last_row(path):
    """(index, id) of the last `#ROW` line, and whether any output followed it."""
    last, after = None, False
    if not path.exists():
        return None, False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#ROW\t"):
                _, idx, rid = line.rstrip("\n").split("\t", 2)
                last, after = (int(idx), rid), False
            elif line.strip():
                after = True
    return last, after


def limit_memory(gb):
    def apply():
        if gb > 0:
            lim = gb << 30
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    return apply


def run_shard(args, shard, part, log):
    start = 0
    restarts = 0
    while True:
        env = dict(os.environ,
                   STATS_ORACLE_CASES=str(args.cases), STATS_ORACLE_OUT=str(part),
                   STATS_ORACLE_SHARD=f"{shard}/{args.jobs}", STATS_ORACLE_START=str(start),
                   STATS_ORACLE_TIMEOUT=str(args.timeout))
        proc = subprocess.run(
            [args.bin, TEST, "--exact", "--nocapture", "--test-threads=1"],
            env=env, capture_output=True, text=True, preexec_fn=limit_memory(args.mem_gb),
        )
        if proc.returncode == 0:
            return
        row, answered = last_row(part)
        if row is None:
            log.append(f"shard {shard}: the dump failed before its first row (rc={proc.returncode}):\n"
                       f"{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
            return
        idx, rid = row
        if not answered:
            # The process died inside the row (abort, out of memory): record it.
            with open(part, "a", encoding="utf-8") as fh:
                fh.write(f"{rid}\t*\tCRASH:rc={proc.returncode}\t0\n")
        restarts += 1
        if restarts > args.max_restarts:
            log.append(f"shard {shard}: more than {args.max_restarts} restarts; giving up after row {idx}")
            return
        start = idx + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--timeout", type=int, default=60, help="seconds per row before it counts as a hang")
    ap.add_argument("--mem-gb", type=int, default=6, help="address-space limit per shard (0: none)")
    ap.add_argument("--max-restarts", type=int, default=50)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.bin == "auto":
        args.bin = find_binary(root)
    args.cases = args.cases.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parts = [args.out.with_name(f"{args.out.name}.part{i}") for i in range(args.jobs)]
    for p in parts:
        p.unlink(missing_ok=True)
    log = []
    t0 = time.time()
    threads = [threading.Thread(target=run_shard, args=(args, i, parts[i], log)) for i in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(args.out, "w", encoding="utf-8") as out:
        for p in parts:
            if p.exists():
                with open(p, encoding="utf-8") as fh:
                    out.writelines(line for line in fh if not line.startswith("#ROW\t"))
                p.unlink()
    print(f"run_dump.py: {args.jobs} shards in {time.time() - t0:.0f} s -> {args.out}")
    for msg in log:
        print(msg, file=sys.stderr)
    sys.exit(1 if log else 0)


if __name__ == "__main__":
    main()
