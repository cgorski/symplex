#!/bin/sh
# The API snapshot in api/: one line per public item of the crate.
#
#   scripts/api_snapshot/api_snapshot.sh            rewrite api/*.txt from the code
#   scripts/api_snapshot/api_snapshot.sh --check    change nothing; exit 1 and print
#                                                   the diff if api/ is out of date
#
# Needs the nightly that .github/workflows/ci.yml pins as API_TOOLCHAIN
# (`rustup toolchain install <that name>`); set API_TOOLCHAIN in the
# environment to use another one.  Further arguments are passed to
# api_snapshot.py (--tiers, --out, --verbose).  README.md beside this file
# explains the output.
set -eu

here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH='' cd -- "$here/../.." && pwd)
cd "$root"

if [ -z "${API_TOOLCHAIN:-}" ]; then
  API_TOOLCHAIN=$(sed -n 's/^ *API_TOOLCHAIN: *"\([^"]*\)".*$/\1/p' .github/workflows/ci.yml | head -n 1)
fi
if [ -z "$API_TOOLCHAIN" ]; then
  echo "api_snapshot: API_TOOLCHAIN is not set and .github/workflows/ci.yml does not pin one" >&2
  exit 2
fi

# rustdoc's JSON output is nightly-only (`-Z unstable-options`).
#   --document-hidden-items  `#[doc(hidden)]` items are public to the compiler:
#                            they are listed (and marked) instead of left out.
#   --cap-lints allow        this is not a lint job; the Docs job is.
cargo "+$API_TOOLCHAIN" rustdoc --lib -- \
  -Z unstable-options --output-format json --document-hidden-items --cap-lints allow

target=$(cargo "+$API_TOOLCHAIN" metadata --format-version 1 --no-deps |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')

exec python3 "$here/api_snapshot.py" --json "$target/doc/symplex.json" "$@"
