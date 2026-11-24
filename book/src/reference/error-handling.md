# Error Handling

symplex has one error type, `SymplexError` (in the prelude; `#[non_exhaustive]`, implements `std::error::Error` via `thiserror`). The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. an out-of-range `m[(i, j)]`, `Context::symbol("")`) are documented under `# Panics` on each item and counted by the same ratchet. Every other failure is either an unevaluated node (see [API Patterns](./api-patterns.md)) or one of the variants below.

## Variants

The enum has fifteen variants (`src/base/errors.rs`); the two newest are `Infinite` (0.58) and `DegenerateData` (0.57).

| Variant | Fields | Raised by |
|---------|--------|-----------|
| `FreeSymbol { name }` | the unbound symbol | `eval_f64`, `compile`, `to_rust_fn`, `to_c_fn`, `integrate_numeric`, `eval_f64_with` when a symbol is not supplied |
| `Unevaluable { reason }` | | numeric evaluation of something that has no number to print: `nan`, an unevaluated derivative or integral, a user function; a value beyond the exponent range of the arbitrary-precision floats (about `2^(±2.1·10⁹)`) — an overflow (`exp(10¹⁶)`), a value certainly not 0 that underflows (`exp(−10¹⁰)`, since 0.32; `eval_f64` returns its correctly rounded `0.0`), and since 0.59 a real or imaginary part certainly not 0 that underflows next to a part within the range (`1 + i·exp(−10¹⁰)`, which printed `1`); a non-real integration bound |
| `Infinite { reason }` | which value is infinite | numeric evaluation (`eval_decimal`, `eval_f64`, `eval_complex64`) of a value that is infinite: `∞`, `−∞`, `zoo`, or a function at a pole or singularity of exact arguments (`elliptic_pi(π, 1)`, `expint(1, 0)`, `elliptic_k(1)`), in the expression or inside it (`1/elliptic_pi(π, 1)`).  New in 0.58: up to 0.57 these were `Unevaluable` (the message is unchanged), and `solve` could claim such a value as a root.  An argument that only comes close to the singular point has a value (`elliptic_k(1 − 10⁻⁶⁰)` is `70.46…`) |
| `PrecisionExhausted { requested, achieved }` | | `eval_decimal`, `eval_f64`, `eval_complex64` when the requested digits cannot be certified: evaluation tracks an error bound and re-evaluates at a higher precision (up to twice the initial one plus 256 bits), so this means a division by a quantity that cancels to 0, `sign`/`floor` of such a quantity, a precision above `EvalConfig::max_evalf_precision`, or a value that is zero to the precision reached after losing a term certainly not 0 that only a precision beyond that maximum would show (`sin²1 + cos²1 − 1 + sin(10⁻⁵⁰⁰⁰)`) |
| `NotImplemented(String)` | names the node | `compile`/codegen on a node without numerical meaning (unevaluated `Integral`, `Apply`, Bessel with symbolic order, …), and on a constant subexpression that is not real (`asin(2)`, `Chi(−60)`); `to_python`/`to_numpy`/`to_julia` on a library function such as `besselj` (they emit `math`/`numpy`/base Julia only; the message names the SciPy routine); numerical evaluation outside the implemented parameter range (`jacobi(9999, …)`, `polylog(−10¹², 1/3)`) |
| `ComputationFailed { operation, reason }` | which operation, why | every `try_*` method when the result is unevaluated; `fourier_transform`/`mellin_transform`/`z_transform` when no rule applies; `solve` when no method applies, and (since 0.49) when only some roots were found — the reason names them, `solve_as_set` returns them ∪ a `ConditionSet`; `linsolve` when it cannot decide whether a right-hand side left over by elimination is 0; `solve_ode_ivp` when constants cannot be fitted; `integrate_numeric` when quadrature does not converge; `eval_f64` when the value is not real (its imaginary part certified nonzero, however small next to the real part; since 0.53) |
| `BudgetExhausted { operation, kind }` | which deterministic work budget (`BudgetKind`) | an engine that stopped at its work limit, e.g. arbitrary-precision quadrature after 30,000 integrand evaluations (`kind: QuadratureEvaluations`).  Budgets count work, never wall-clock time, so the same call fails or succeeds the same way on every machine |
| `ExpressionSwell { operation, nodes }` | the budget that was exceeded, in expression nodes (100,000 for the matrix operations) | matrix operations (`det`, `inv`, `exp`, eigenvalues, …) whose symbolic entries grew past the expression budget |
| `NotDiagonalizable { operation, reason }` | | `diagonalize` of a defective matrix |
| `Divergent { operation, reason }` | | `try_integrate_definite` when the integral is *proven* divergent; `integrate_numeric` when a comparison test proves the divergence; `laplace_final_value` for a pole in the closed right half-plane |
| `NoSolution { operation, reason }` | | `solve` / `solve_general` on a contradiction or range violation proved in the unknown's domain (`sin x = 2` for a real `x`, `|x| = −1`); `solve_ode_ivp` with contradictory initial conditions.  A range restriction is used only where it holds: for a complex unknown `erf(x) = 2` is refused (`ComputationFailed`), not `NoSolution` — it has the root `0.8694 + 1.3406i` |
| `InfiniteSolutions { operation, reason }` | | `solve` on an identity; `polysys::solve_system_ex` on a positive-dimensional system |
| `InvalidArgument { operation, reason }` | | malformed input: a non-symbol variable, duplicate parameter names, wrong shapes (`cholesky` on a non-symmetric matrix, `norm_p` on a non-vector), `linsolve` with a non-linear equation, `Rule::try_new` with an unbound wildcard, `CompiledFn::try_call` with the wrong arity |
| `ContradictoryAssumptions { symbol, a, b }` | the two clashing facts | `Context::symbol_with`, `Ex::assume`, `Ex::refine_with` when the assumptions contradict each other once their consequences are drawn (`Positive` with `Negative`, `Integer` with `Irrational`, …); nothing is declared |
| `DegenerateData { operation, kind, reason }` | `kind` is a `Degeneracy` (below); `reason` names the column, group or coefficient | the statistics on data, when the estimate or statistic asked for does not exist for these data (it is infinite or `0/0`): the regression, ANOVA, survival and Cox fits since 0.57 (they were `InvalidArgument` or `ComputationFailed`, with the same reason text), and since 0.58 the hypothesis tests, effect sizes and reliability coefficients on a zero variance (they were `InvalidArgument`).  A wrong shape (different lengths, no data) stays `InvalidArgument` |

Because the enum is `#[non_exhaustive]`, always include a wildcard arm when matching.  `SymplexError` is `Clone`.  Match on variants, not on the text of `reason` — symplex itself never branches on error messages.

## One call for each variant

Each call below is taken from the test suite (the file is named in the comment) and returns the variant shown.

```rust
use symplex::errors::Degeneracy;
use symplex::linprog::qi;
use symplex::matrix::Matrix;
use symplex::prelude::*;
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::{t_test_one_sample, Alternative};

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let parse = |s: &str| ctx.parse(s).unwrap();

    // ContradictoryAssumptions, InvalidArgument          tests/v28/v28_perf.rs
    assert!(matches!(
        ctx.symbol_with("u", &[Assumption::Positive, Assumption::Negative]),
        Err(SymplexError::ContradictoryAssumptions { .. })
    ));
    assert!(matches!(
        ctx.symbol_with("", &[Assumption::Real]),
        Err(SymplexError::InvalidArgument { .. })
    ));

    // FreeSymbol                                         tests/v02/v02_basefix_evalf.rs
    assert!(matches!(x.eval_f64(), Err(SymplexError::FreeSymbol { name }) if name == "x"));

    // Unevaluable: beyond the exponent range             tests/v30/v30_evalf.rs, tests/v31/v31_values.rs
    assert!(matches!(parse("exp(10^16)").eval_decimal(16), Err(SymplexError::Unevaluable { .. })));
    assert!(matches!(
        parse("1 + I*exp(-10^10)").eval_decimal(16),
        Err(SymplexError::Unevaluable { .. })
    ));

    // Infinite: a pole of exact arguments                tests/v36/v36_s15p1_evalf.rs
    assert!(matches!(parse("elliptic_pi(pi, 1)").eval_decimal(15), Err(SymplexError::Infinite { .. })));
    assert!(matches!(parse("1/elliptic_pi(pi, 1)").eval_decimal(15), Err(SymplexError::Infinite { .. })));
    assert!(parse("elliptic_k(1 - 10^-60)").eval_decimal(15).is_ok());

    // PrecisionExhausted: the lost term needs more than  tests/v34/v34_evalf.rs
    // the configured maximum precision to show
    assert!(matches!(
        parse("sin(1)^2+cos(1)^2-1+sin(10^(-5000))").eval_decimal(16),
        Err(SymplexError::PrecisionExhausted { .. })
    ));

    // NotImplemented: `to_python` emits `math` only      tests/v20/v20_libfn.rs
    assert!(matches!(x.bessel_j(&ctx.int(0)).to_python(), Err(SymplexError::NotImplemented(_))));

    // ComputationFailed: no closed form was found        tests/v02/v02_defint_node.rs
    assert!(matches!(
        x.pow(&x).try_integrate_definite(&x, &ctx.int(0), &ctx.int(1)),
        Err(SymplexError::ComputationFailed { .. })
    ));

    // Divergent: proved, not guessed                     tests/v02/v02_integration_definite.rs
    assert!(matches!(
        x.powi(-2).try_integrate_definite(&x, &ctx.int(-1), &ctx.int(1)),
        Err(SymplexError::Divergent { .. })
    ));

    // NoSolution (for a real unknown), InfiniteSolutions tests/v30/v30_hunts.rs
    let xr = ctx.symbol_with("xr", &[Assumption::Real]).unwrap();
    assert!(matches!(parse("exp(xr) + 1").solve(&xr), Err(SymplexError::NoSolution { .. })));
    assert!(matches!(
        parse("sqrt(2)*sqrt(3) - sqrt(6)").solve(&x),
        Err(SymplexError::InfiniteSolutions { .. })
    ));

    // NotDiagonalizable, ExpressionSwell                 tests/v36/v36_w9_budgets.rs
    let jordan = Matrix::from_i64(&ctx, &[&[1, 1], &[0, 1]]).unwrap();
    assert!(matches!(jordan.diagonalize(), Err(SymplexError::NotDiagonalizable { .. })));
    let mut e = x.clone();
    for _ in 0..16 {
        e = &e.sin() + &e.cos();
    }
    let swollen = Matrix::new(vec![vec![e.clone(), ctx.int(1)], vec![ctx.int(1), e]]).unwrap();
    assert!(matches!(
        swollen.inv(),
        Err(SymplexError::ExpressionSwell { operation: "inv", nodes: 100_000 })
    ));

    // DegenerateData: a constant sample has no t         tests/v13/v13_hypothesis.rs
    let err = t_test_one_sample(&ctx, &from_i64(&[3, 3, 3]), &qi(0), Alternative::TwoSided).unwrap_err();
    assert!(matches!(err, SymplexError::DegenerateData { kind: Degeneracy::ZeroVariance, .. }));
    assert_eq!(err.degeneracy(), Some(Degeneracy::ZeroVariance));
}
```

`BudgetExhausted` is the one variant the test suite builds by hand instead of reaching it through a public call.  A call that returns it (found by trying; it takes about 7 s in a debug build, so the block is compiled but not run):

```rust,no_run
use symplex::prelude::*;
use symplex::BudgetKind;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // No closed form: stays `Integral(sin(x^4), x, 0, 200)`.
    let f = ctx.parse("sin(x^4)").unwrap().integrate_definite(&x, &ctx.int(0), &ctx.int(200));
    // The quadrature stops after 30,000 evaluations of the integrand.
    assert!(matches!(
        f.eval_decimal(15),
        Err(SymplexError::BudgetExhausted { operation: "evalf", kind: BudgetKind::QuadratureEvaluations })
    ));
}
```

## Degenerate data

`DegenerateData` carries a `Degeneracy` (`symplex::errors::Degeneracy`, `#[non_exhaustive]`) so that a pipeline can react — drop an aliased regressor, switch to a penalised fit, skip a constant group — without reading the message.  `err.degeneracy()` returns it, and `None` for every other variant.

| `kind` | Meaning | A call that returns it (from the tests) |
|--------|---------|------------------------------------------|
| `Collinear` | the design is rank deficient: a regressor is a linear combination of the others, so its coefficient is not identified | `ols` with a duplicated column; `hat_matrix`, `vif`, `polyfit` with too few distinct abscissae; `cox_ph` with a constant covariate |
| `Separation` | the likelihood has no finite maximiser, so some coefficient is infinite | `logit` when every `x > 0` is a success; a monotone partial likelihood in `cox_ph` |
| `ConstantResponse` | the response takes a single value | `ols` with a constant `y`; `anova_two_way_with` |
| `ZeroVariance` | a variance the statistic divides by is zero | `t_test_one_sample` of `[3, 3, 3]`; `anova_one_way` of `[1, 1]`, `[2, 2]`; `t_statistics` of a perfect fit; `cohens_d`, `spearman_test`, `kruskal_wallis` on constant samples |
| `NoEvents` | every subject is censored | `log_rank_test`, `mean_event_time`, `cox_ph` on all-censored observations |

The exact methods (least squares and ANOVA over ℚ, Kaplan–Meier, the log-rank test) decide these cases exactly.  The `f64` maximum-likelihood fits (logistic and Cox regression) decide `Collinear` numerically, to the threshold each documents, and `Separation` by the signature of a diverging Newton iteration.

```rust
use symplex::errors::Degeneracy;
use symplex::prelude::*;
use symplex::stats::regression::{logit, LogitOpts};

fn main() {
    // Every x > 0 is a success: the likelihood has no finite maximiser.
    let y = [false, false, false, true, true, true];
    let x: Vec<Vec<f64>> = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0].iter().map(|&v| vec![v]).collect();
    match logit(&y, &x, true, &LogitOpts::default()) {
        Err(e) if e.degeneracy() == Some(Degeneracy::Separation) => println!("separated: {e}"),
        Err(e) => println!("failed: {e}"),
        Ok(_) => println!("fitted"),
    }
}
```

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

## Why a call is refused

A refusal — an unevaluated node, `Err(ComputationFailed)`, `None` from a query, `Unknown` from a prover — means that the engine could not *establish* an answer, not that none exists. The usual reasons, which the `reason` text of the `try_` twin's error names:

- **No method applies**: no rule, table entry or algorithm covers the input.
- **Something the answer depends on is undecided**: a sign or a zero test that neither the assumptions nor an error-bounded evaluation settles, a parameter that may be complex. Declaring what you know (`Positive`, `Real`, `Integer`) is usually the way forward.
- **The answer would hold only in part**: `solve` found some of the roots, a root exists only for some parameter values, a rewrite is valid on one branch only.
- **The expression has no value there.** `expint(1, 0)` is infinite, so `x^2*expint(1, 0)` is not a number at any `x` and has no limit at 0 — although `0·c = 0` for every number `c`:

```rust
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.parse("x^2*expint(1, 0)").unwrap();
    println!("{}", e.limit(&x, &ctx.int(0)));           // Limit(x^2*expint(1, 0), x, 0)
    match e.try_limit(&x, &ctx.int(0)) {
        Err(SymplexError::ComputationFailed { operation, reason }) => println!("{operation}: {reason}"),
        other => println!("{other:?}"),
    }
    // limit: a part of the expression has no value at any point
}
```

- **A work budget ran out** (`BudgetExhausted`, or a `reason` that names the budget). Budgets count work, not time, so the same call is refused on every machine.

The cases that are refused today and are known to have an answer are listed under [Known Limitations](./known-limitations.md).

## `Option` is not an error

Three-valued queries (`is_positive`, `equals`, `SetEx::contains`, `Matrix::is_symmetric`, …) return `None` for "cannot decide". That is an answer, not a failure — typically it means a symbol needs an assumption (`ctx.symbol_with("a", &[Assumption::Positive])`).

## Panics

The library never calls `unwrap`/`expect`/`panic!`/`unreachable!` on user data (ratchet `tests/unit/test_no_panics.rs`); the remaining `assert!`s on caller-supplied *shapes* (e.g. an out-of-range `m[(i, j)]`, `Context::symbol("")`) are documented under `# Panics` on each item and counted by the same ratchet. The programming errors that panic by design (the ratchet's allowlist is the complete list):

1. **Cross-context mixing** — combining expressions from different `Context`s. The message names the operation.
2. **Empty `Sum`/`Product` iterators** — `iter.sum::<Ex>()` on an empty iterator has no context to build `0` in. Use `ctx.sum(iter)` / `ctx.product(iter)`, or collect into `Option<Ex>` (which yields `None`).
3. **Operators and indexing, which cannot return a `Result`** — `&a + &b` on matrices of different shapes and `m[(i, j)]`, `row`, `col` out of range (`add`/`sub`/`matmul` and `try_get`/`try_row`/`try_col` return one); `+`/`-`/`*` and `pow` of `MultiPoly`s over different variable counts or past the exponent range (`try_add`, `try_mul`, `try_pow`); a zero denominator to `RationalFn::new` or its division (`try_new`, `try_div`).  The matrix constructors (`Matrix::zeros`, `identity`, …) return `Result` since 0.29.
4. **`Context::symbol("")`** (`try_symbol` returns an error) and **`Polytope::with_halfspace`** with the wrong number of coefficients (`try_with_halfspace`).
5. **More than 2³² distinct nodes, numbers or symbols in one context** — the index space of the arena.

The ratchet is a scan of the source for panicking constructs.  It does not see an allocation that fails: several functions that build a list as long as a count they are given (`Matrix::identity`, `symbols_indexed`, `equispaced_grid`, …) still abort the process for a count like `2⁴⁰` (and panic with "capacity overflow" for `usize::MAX`); 0.58 and 0.59 bounded the others (see `CHANGELOG.md`).

If you find a panic elsewhere, it is a bug — please report it with the expression that triggered it.

## Configuration limits

`EvalConfig { max_pow_exponent, max_result_digits, max_evalf_precision }` (via `Context::with_config`) caps the size of intermediate results. Exceeding a cap leaves the expression unevaluated (`2^5000` stays a `Pow` node) rather than consuming unbounded memory. Simplification, rewriting and eigenvalue computations have their own internal budgets (`MAX_REWRITE_OPS`, `MATCH_BUDGET`, `EXPRESSION_BUDGET`, the Gruntz work budget) that make them return the best result so far instead of hanging.
