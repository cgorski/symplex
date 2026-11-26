#!/bin/sh
# Compare two hunt.sh outputs: the changed (case, op) pairs by op and by kind.
#
#   probes/hunt/hunt_diff.sh <base.tsv> <new.tsv> [--factor F] [--min-ms M]
#                            [-n N] [--ops a,b,c] [--changed-out FILE]
#
# See hunt_diff.py (`--help`).  `golden_diff.py` is not enough here: it knows
# neither the kinds nor the CPU column.  PYTHON overrides the interpreter.
here=$(cd "$(dirname "$0")" && pwd)
exec "${PYTHON:-python3}" "$here/hunt_diff.py" "$@"
