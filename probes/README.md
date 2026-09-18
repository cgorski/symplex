# Probes

Developer diagnostic scripts for measuring feature coverage and spotting
regressions. These are **not** user-facing examples — they are internal
testing tools that print pass/fail tallies rather than asserting.

They live in `probes/` (not `examples/`), so Cargo does not compile them by
default and they do not run in CI. To run one, copy it into `examples/`
temporarily:

```sh
cp probes/probe_integration.rs examples/probe_integration.rs
cargo run --release --example probe_integration
rm examples/probe_integration.rs
```

All five probes compile against the 0.2 API (checked at the 0.2.0 release).

## Probes

| File | What it measures |
|------|------------------|
| `probe_integration.rs` | ~60 indefinite integrals across every strategy family (basic, by-parts, u-substitution, trig powers, rational, radicals, cyclic IBP, inverse trig, parametric, hyperbolic, completing the square, linear substitution, special/non-elementary); counts closed forms found |
| `probe_calculus.rs` | Limits, series expansions, ODE solving and classification |
| `probe_correctness.rs` | For each integral that returns a closed form, checks `F(hi) − F(lo)` against numerical quadrature; also verifies simplification identities, series accuracy and ODE solutions numerically. Reports PASS / WRONG / SKIP |
| `safety_test.rs` | Dimensional-analysis type safety: every legal ordering of typed products compiles and tracks the dimension |
| `expr_units_test.rs` | `expr!` + `from_ex`, named-type arithmetic, and `diff_wrt` / `integrate_wrt` workflows in the units module |

The user-facing counterparts of these checks are the integration test suites
in `tests/` (`test_correctness_audit.rs`, `test_sympy_cross_validation.rs`,
`v02_integration_*.rs`, …), which *assert* rather than tally.

## Adding a new probe

1. Create `probes/probe_YOURNAME.rs` (a complete program with `fn main`).
2. Add a row to the table above.
3. Use the `try_*` / pass-fail counting pattern from the existing probes; never
   `unwrap()` a result you are probing — count it.
4. Print a summary line at the end: `=== Summary: N passed, M failed, T total ===`.
5. Before committing, copy it into `examples/` and confirm it builds and runs;
   then remove the copy.
