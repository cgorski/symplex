#!/bin/sh
# The nightly statistics oracle, end to end (scripts/stats_oracle/README.md):
# check the generated seed set, build the release `v36` test binary, dump
# symplex's values, compare them with scipy / statsmodels / mpmath.
#
#   scripts/stats_oracle/run.sh [OUT_DIR]        (default target/stats_oracle)
#
# PYTHON overrides the interpreter (it needs requirements.txt installed);
# JOBS the parallelism (default: the number of CPUs).  Exit status 1 when the
# comparison found a failure that is not on the allowlist.
set -eu
cd "$(dirname "$0")/../.."
OUT=${1:-target/stats_oracle}
PY=${PYTHON:-python3}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}
mkdir -p "$OUT"
start=$(date +%s)
"$PY" scripts/stats_oracle/gen_cases.py --check
cargo test --release --test v36 --no-run --message-format=json-render-diagnostics > "$OUT/build.json"
BIN=$("$PY" -c 'import json, sys
for line in open(sys.argv[1]):
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "v36":
        print(m["executable"])' "$OUT/build.json")
built=$(date +%s)
"$PY" scripts/stats_oracle/run_dump.py --bin "$BIN" --cases scripts/stats_oracle/cases \
    --out "$OUT/dump.tsv" --jobs "$JOBS"
dumped=$(date +%s)
rc=0
"$PY" scripts/stats_oracle/compare.py --cases scripts/stats_oracle/cases --dump "$OUT/dump.tsv" \
    --allowlist scripts/stats_oracle/allowlist.tsv --report "$OUT/report.md" \
    --verdicts "$OUT/verdicts.tsv" --timings "$OUT/timings.tsv" --jobs "$JOBS" || rc=$?
done_=$(date +%s)
echo "timing: build $((built - start)) s, dump $((dumped - built)) s, compare $((done_ - dumped)) s" \
    | tee "$OUT/timing.txt"
exit $rc
