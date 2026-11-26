#!/bin/sh
# Run the golden harness of THIS tree on your own list of expressions.
#
#   probes/hunt/hunt.sh <cases-file> <out.tsv> [--ops a,b,c] [--jobs N] [--timeout S]
#
# Builds `probes/golden` of the tree this script stands in (release) and runs
# it in `--file` mode: one expression per line of <cases-file> (case id = line
# number; blank lines and lines starting with # are skipped).  <out.tsv> gets
# one line per (case, op):
#
#   case_id <TAB> op <TAB> status <TAB> result <TAB> cpu_ms
#
# status is OK (result = the value), ERR (a refusal: result = the error text),
# PANIC, TIMEOUT, OOM or CRASH; cpu_ms is the CPU time of the op's thread
# ('-' where there is none).  Results are never shortened to a hash here.
# Next to <out.tsv>: .summary.txt (statuses per op), .meta.json, .panics.txt.
#
#   --ops a,b,c    only these ops (default all; see `golden --list-ops`)
#   --jobs N       parallel processes (default 8)
#   --timeout S    per-op limit in seconds (default 20)
#   --mem-mb M     per-process heap cap in MiB (default 4096)
#
# PYTHON overrides the interpreter (default python3; standard library only).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
golden=$here/../golden
usage() {
    echo "usage: $0 <cases-file> <out.tsv> [--ops a,b,c] [--jobs N] [--timeout S] [--mem-mb M]" >&2
    exit 2
}
[ $# -ge 2 ] || usage
cases=$1
out=$2
shift 2
ops=all
jobs=8
timeout=20
mem=4096
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || usage
    case "$1" in
    --ops) ops=$2 ;;
    --jobs) jobs=$2 ;;
    --timeout) timeout=$2 ;;
    --mem-mb) mem=$2 ;;
    *) usage ;;
    esac
    shift 2
done
case "$cases" in /*) ;; *) cases="$PWD/$cases" ;; esac
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
[ -r "$cases" ] || { echo "$0: cannot read $cases" >&2; exit 2; }
[ -f "$golden/Cargo.toml" ] || { echo "$0: no harness at $golden" >&2; exit 2; }
n=$(grep -cv -e '^[[:space:]]*$' -e '^[[:space:]]*#' "$cases" || true)
[ "$n" -gt 0 ] || { echo "$0: no cases in $cases" >&2; exit 2; }
# Spread the cases over the processes (the driver's slices).
slice=$(((n + jobs - 1) / jobs))
[ "$slice" -le 100 ] || slice=100
timeout_ms=$(awk -v s="$timeout" 'BEGIN { printf "%d", s * 1000 }')
cd "$golden"
cargo build --release --quiet
exec "${PYTHON:-python3}" "$golden/driver.py" --bin "$golden/target/release/golden" --out "$out" \
    --file "$cases" --ops "$ops" --timeout-ms "$timeout_ms" --mem-mb "$mem" \
    -j "$jobs" --slice "$slice" --cpu --full
