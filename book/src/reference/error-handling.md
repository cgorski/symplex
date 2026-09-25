# Error Handling

symplex has one error type, `SymplexError` (in the prelude; `#[non_exhaustive]`, implements `std::error::Error` via `thiserror`). The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. `Matrix::zeros(0, n)`, `Context::symbol("")`) are documented under `# Panics` on each item and counted by the same ratchet. Every other failure is either an unevaluated node (see [API Patterns](./api-patterns.md)) or one of the variants below.

## Variants

| Variant | Fields | Raised by |
|---------|--------|-----------|
| `FreeSymbol { name }` | the unbound symbol | `eval_f64`, `compile`, `to_rust_fn`, `to_c_fn`, `integrate_numeric`, `eval_f64_with` when a symbol is not supplied |
| `Unevaluable { reason }` | | numeric evaluation of a node with no finite value (`oo`, `zoo`, a set, …), a non-real integration bound |
| `PrecisionExhausted { requested, achieved }` | | `eval_decimal`, `eval_f64`, `eval_complex64` when the requested digits cannot be certified: evaluation tracks an error bound and re-evaluates at a higher precision (up to twice the initial one plus 256 bits), so this means a division by a quantity that cancels to 0, `sign`/`floor` of such a quantity, or a precision above `EvalConfig::max_evalf_precision` |
| `NotImplemented(String)` | names the node | `compile`/codegen on a node without numerical meaning (unevaluated `Integral`, `Apply`, Bessel with symbolic order, …) |
| `ComputationFailed { operation, reason }` | which operation, why | every `try_*` method when the result is unevaluated; `fourier_transform`/`mellin_transform`/`z_transform` when no rule applies; `solve` when no method applies; `solve_ode_ivp` when constants cannot be fitted; `integrate_numeric` when quadrature does not converge |
| `Divergent { operation, reason }` | | `try_integrate_definite` when the integral is *proven* divergent; `laplace_final_value` for a pole in the closed right half-plane |
| `NoSolution { operation, reason }` | | `solve` / `solve_general` on a contradiction or range violation (`sin x = 2`); `solve_ode_ivp` with contradictory initial conditions |
| `InfiniteSolutions { operation, reason }` | | `solve` on an identity; `polysys::solve_system_ex` on a positive-dimensional system |
| `InvalidArgument { operation, reason }` | | malformed input: a non-symbol variable, duplicate parameter names, wrong shapes (`cholesky` on a non-symmetric matrix, `norm_p` on a non-vector), `linsolve` with a non-linear equation, `Rule::try_new` with an unbound wildcard, `CompiledFn::try_call` with the wrong arity |
| `ContradictoryAssumptions { symbol, a, b }` | the two clashing facts | `Context::symbol_with`, `Ex::assume`, `Ex::refine_with` when the assumptions contradict each other once their consequences are drawn (`Positive` with `Negative`, `Integer` with `Irrational`, …); nothing is declared |

Because the enum is `#[non_exhaustive]`, always include a wildcard arm when matching.

## Matching on outcomes

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for eq in [&x.powi(2) - 4, &x - &x, &x.sin() - 2, &x.exp().ln().exp() - &x.exp()] {
        match eq.solve(&x) {
            Ok(roots) => println!("{eq} = 0 → {roots:?}"),
            Err(SymplexError::InfiniteSolutions { reason, .. }) => println!("{eq} = 0 → identity: {reason}"),
            Err(SymplexError::NoSolution { reason, .. }) => println!("{eq} = 0 → no solution: {reason}"),
            Err(SymplexError::ComputationFailed { reason, .. }) => println!("{eq} = 0 → could not solve: {reason}"),
            Err(e) => println!("{eq} = 0 → {e}"),
        }
    }

    match x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &ctx.int(1)) {
        Err(SymplexError::Divergent { reason, .. }) => println!("divergent: {reason}"),
        Err(SymplexError::ComputationFailed { .. }) => println!("undecided"),
        other => println!("{other:?}"),
    }
}
```

## Unevaluated nodes vs. errors

The base methods (`integrate`, `limit`, `summation`, …) return an unevaluated node and never fail; the `try_` twin turns that into `Err(ComputationFailed)`. Choose based on the caller:

- **Interactive / exploratory code**: use the base method and print the result; an `Integral(…)` node is informative.
- **Pipelines and code generation**: use `try_` so that a missing closed form stops the pipeline instead of producing a function that calls `Integral`.

## `Option` is not an error

Three-valued queries (`is_positive`, `equals`, `SetEx::contains`, `Matrix::is_symmetric`, …) return `None` for "cannot decide". That is an answer, not a failure — typically it means a symbol needs an assumption (`ctx.symbol_with("a", &[Assumption::Positive])`).

## Panics

The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. `Matrix::zeros(0, n)`, `Context::symbol("")`) are documented under `# Panics` on each item and counted by the same ratchet. The programming errors that panic by design:

1. **Cross-context mixing** — combining expressions from different `Context`s. The message names the operation.
2. **Empty `Sum`/`Product` iterators** — `iter.sum::<Ex>()` on an empty iterator has no context to build `0` in. Use `ctx.sum(iter)` / `ctx.product(iter)`, or collect into `Option<Ex>` (which yields `None`).

If you find a panic elsewhere, it is a bug — please report it with the expression that triggered it.

## Configuration limits

`EvalConfig { max_pow_exponent, max_result_digits, max_evalf_precision }` (via `Context::with_config`) caps the size of intermediate results. Exceeding a cap leaves the expression unevaluated (`2^5000` stays a `Pow` node) rather than consuming unbounded memory. Simplification, rewriting and eigenvalue computations have their own internal budgets (`MAX_REWRITE_OPS`, `MATCH_BUDGET`, `EXPRESSION_BUDGET`, the Gruntz work budget) that make them return the best result so far instead of hanging.
