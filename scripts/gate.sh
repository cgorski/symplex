#!/bin/sh
# The release gate, one bounded stage at a time.
#
# Every stage runs under `timeout`, writes its full output to
# $GATE_LOG_DIR/<stage>.log, and prints one line: the stage, its exit code,
# its wall time and the test-result summary.  A failure is therefore named by
# `grep -E "FAILED|panicked" $GATE_LOG_DIR/<stage>.log` — never by re-running
# the stage.  Stages are independent; pass their names as arguments to run a
# subset (default: all, in order).
#
#   scripts/gate.sh                 # everything
#   scripts/gate.sh fmt clippy      # just those
#   GATE_LOG_DIR=/tmp/gate scripts/gate.sh nextest
#
# Budgets are per stage (seconds); a stage that exceeds its budget is killed
# and reported as `rc=124`.  The budgets are deliberately tight: the merged
# doctest binary runs all ~1,300 doctests in about a second, so a doctest
# stage that takes minutes means rustdoc fell back to compiling each doctest
# standalone — which it does silently when *one* doc example fails to
# compile.  Run `cargo test --doc -- <one doctest name>` and look for
# `finished in 0.00s`: anything slower means the merged build is broken.
#
# Known environmental stall: right after a full rebuild (e.g. a version
# bump) `cargo nextest` can sit for minutes between "Finished" and
# "Starting" with no CPU — the host's security agent scanning the freshly
# built test binaries on first execution.  `cargo nextest list` shows the
# same wait; each binary's own `--list` takes milliseconds.  Rerun once the
# binaries are warm; do not chase it in the code.

set -u
cd "$(dirname "$0")/.." || exit 2
LOG_DIR="${GATE_LOG_DIR:-/tmp/symplex-gate}"
mkdir -p "$LOG_DIR"

run_stage() {
    name="$1"; budget="$2"; shift 2
    log="$LOG_DIR/$name.log"
    start=$(date +%s)
    # `-k 5`: SIGKILL five seconds after SIGTERM; `--foreground` is NOT used so
    # the whole process group (cargo, nextest, the test binaries) is signalled
    # and a killed stage leaves no orphan holding the target-dir lock.
    timeout -k 5 "$budget" "$@" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        # Belt and braces: reap anything still running from this crate's tests.
        pkill -f "$(pwd)/target/debug/deps/" 2>/dev/null
    fi
    end=$(date +%s)
    summary=$(grep -E "^\s*(Summary|test result:)" "$log" | tail -1 | sed 's/^ *//')
    printf '%-14s rc=%-3s %4ss  %s\n' "$name" "$rc" "$((end - start))" "$summary"
    if [ "$rc" -ne 0 ]; then
        grep -E "FAILED|panicked at|^error" "$log" | head -5 | sed 's/^/    /'
        FAILED_STAGES="$FAILED_STAGES $name"
    fi
}

FAILED_STAGES=""
STAGES="${*:-fmt clippy deny nextest doctest doc ui benches subcrates examples book}"

for stage in $STAGES; do
    case "$stage" in
        # All four crates, as CI checks them (0.25.0 failed CI on
        # symplex-wasm, which `cargo fmt --all` here does not reach).
        fmt)      run_stage fmt      60   sh -c 'cargo fmt --all -- --check && cargo fmt --all --manifest-path symplex-macros/Cargo.toml -- --check && cargo fmt --all --manifest-path symplex-build/Cargo.toml -- --check && cargo fmt --all --manifest-path symplex-wasm/Cargo.toml -- --check' ;;
        clippy)   run_stage clippy   600  cargo clippy --all-targets -- -D warnings ;;
        deny)     run_stage deny     120  cargo deny check ;;
        nextest)  run_stage nextest  900  cargo nextest run --no-fail-fast ;;
        doctest)  run_stage doctest  120  cargo test --doc -q ;;
        doc)      run_stage doc      300  env RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items ;;
        ui)       run_stage ui       120  cargo test --test ui_tests ;;
        # CI runs the criterion benches once each in test mode; a bench that is
        # slow in debug (0.14: qmatrix/hnf/40, two hours) shows up here first.
        benches)  run_stage benches  300  cargo test --benches ;;
        subcrates)
            run_stage macros   120  sh -c 'cd symplex-macros && cargo test'
            run_stage build    120  sh -c 'cd symplex-build && cargo test'
            run_stage wasm     300  sh -c 'cd symplex-wasm && cargo build'
            # CI also checks the 32-bit wasm target: `usize` is 32 bits there (0.13.0
            # shipped `1 << 40` as a `usize` const and CI was red for three releases).
            run_stage wasm32   600  cargo check --manifest-path symplex-wasm/Cargo.toml --target wasm32-unknown-unknown
            run_stage fuzz     300  cargo check --manifest-path fuzz/Cargo.toml
            ;;
        examples)
            for e in certificates_to_lean readme_snippets polynomials exact_matrices polyhedron_certificates control_system dynamics; do
                run_stage "ex_$e" 120 cargo run -q --example "$e"
            done
            ;;
        book)     run_stage book     120  mdbook build book ;;
        *) echo "unknown stage: $stage" >&2; exit 2 ;;
    esac
done

if [ -n "$FAILED_STAGES" ]; then
    echo "FAILED:$FAILED_STAGES  (logs in $LOG_DIR)"
    exit 1
fi
echo "all stages passed  (logs in $LOG_DIR)"
