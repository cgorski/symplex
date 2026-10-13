#!/bin/sh
# Build the golden harness against the tree it sits in (`../..`) and run the
# parallel driver.
#
#   probes/golden/run_golden.sh OUT.tsv [driver.py options: --seeds A..B -j 48 ...]
#
# PYTHON overrides the interpreter (default python3; the driver needs only the
# standard library).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
if [ $# -lt 1 ]; then
    echo "usage: $0 OUT.tsv [driver.py options]" >&2
    exit 2
fi
out=$1
shift
case "$out" in
/*) ;;
*) out="$PWD/$out" ;;
esac
cd "$here"
cargo build --release --quiet
exec "${PYTHON:-python3}" "$here/driver.py" --bin "$here/target/release/golden" --out "$out" "$@"
