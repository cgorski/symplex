# Probes

Developer diagnostic scripts for testing feature coverage and finding regressions.
These are NOT user-facing examples — they are internal testing tools.

Run a probe: `cargo run --example probe_NAME`

They live in `examples/` with a `probe_` prefix so Cargo auto-discovers them,
but they are conceptually separate from the user-facing examples.

## Probes

- `probe_integration.rs` — Tests ~60 integrals across all categories (basic, by-parts, u-sub, trig powers, rational, sqrt, cyclic IBP, inverse trig, parametric, hyperbolic, completing the square, linear substitution, special/non-elementary)
- `probe_calculus.rs` — Tests limits, series expansions, ODE solving and classification
- `probe_apart.rs` — Tests partial fraction decomposition edge cases *(planned)*
- `probe_api_ergonomics.rs` — Tests public API patterns, type safety, error handling *(planned)*

## Adding a new probe

1. Create `examples/probe_YOURNAME.rs`
2. Add an entry to this README
3. Use the `try_*` / pass-fail counting pattern from the existing probes
4. Print a summary line at the end: `=== Summary: N passed, M failed, T total ===`
