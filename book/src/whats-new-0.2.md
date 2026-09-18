# What's New in 0.2

symplex 0.2.0 is a large release. This page is the tour; the complete list — including every breaking change and its replacement — is in the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md), and the [migration guide](./reference/migrating-0.2.md) walks through the code changes you will need.

The theme of the release is **never silently wrong**. Several 0.1 operations guessed (`∫₋₁¹ dx/x² = −2`, `re(z) = z`, `solve(x − x) = []`); in 0.2 they return an `Err`, an unevaluated node, or `None`.

## By the numbers

Measured on the release commit:

| | 0.1 | 0.2 |
|--|-----|-----|
| `ExprNode` variants | ~80 | 91 |
| `#[test]` functions | ~6,000 | 10,177 |
| Lines in `src/` | ~103K | ~163K |
| ODE classes | 13 | 16 |
| Code generation targets | Rust | Rust, C99, compiled closures |

## Complex analysis

New `Re`, `Im`, `Conjugate`, `Arg` nodes and `Ex::{re, im, conjugate, arg, as_real_imag, expand_complex, polar, abs_squared, is_real_valued}`. Conjugation distributes over sums, products and integer powers and commutes with real-analytic functions; realness is decided by the assumption system, never assumed. New constants `EulerGamma`, `Catalan`, `GoldenRatio`, and complex infinity (`zoo`, the value of `1/0`).

→ [Complex Analysis and Special Functions](./guide/complex-analysis.md)

## Definite, improper and numeric integration

`integrate_definite` / `try_integrate_definite` detect interior singularities, handle infinite bounds and endpoint singularities via one-sided limits, exploit symmetry, integrate `Abs`/`Heaviside`/`DiracDelta`/`Piecewise`, and consult a ~30-entry table of improper integrals (with symbolic parameters under assumptions). `integrate_numeric` is adaptive Gauss–Kronrod. Residues work at poles of any order and at infinity.

→ [Definite Integration and Quadrature](./guide/definite-integration.md)

## Summation, products and series

`summation`, `product_over`, `hypergeometric_ratio`, `is_convergent`, `series_at_infinity`, and an `Ex`-based `FormalPowerSeries` with lazy exact coefficients, closed-form general terms, arithmetic, composition, inversion and reversion.

→ [Summation and Series](./guide/summation.md)

## Solving

`solve` reports identities as `Err(InfiniteSolutions)` and contradictions/range violations as `Err(NoSolution)`. `solve_general` returns periodic families with an integer parameter. `linsolve` returns `Unique`/`Parametric`/`Inconsistent`. Polynomial systems return algebraic solutions. `solve_numeric_system` is a damped Newton method. ODEs gained nth-order constant-coefficient, Clairaut and Riccati classes, initial-value problems (`solve_ode_ivp`) and system IVPs. `rsolve_linear` solves linear recurrences.

→ [Solving Equations](./guide/solving.md)

## Sets and logic

`SetEx` has a normal form (`simplify`), set algebra (`difference`, `symmetric_difference`, `absolute_complement`), three-valued queries (`contains`, `is_subset`, `is_disjoint`, `is_empty`), topology (`boundary`, `closure`, `interior`), and conversion to conditions. `BoolEx` has `to_nnf`/`to_cnf`/`to_dnf`, `is_tautology`, `satisfiable`, `truth_table`, and `eval` folds relations through assumptions. `reduce_inequalities` turns a conjunction of conditions into a set.

→ [Sets and Logic](./guide/sets-and-logic.md)

## The rule engine is public

`Rule`, `RuleSet`, `Bindings`, `RewriteOpts`, `Step`; `Ex::{rewrite, rewrite_once, rewrite_traced, rewrite_with, simplify_with_rules, simplify_traced}`; AC matching with `rest__` sequence wildcards; `rule!` macro rules via `RuleSet::from_macro_rules`. Plus new targeted simplifiers: `sqrtdenest`, `signsimp`, `powdenest(force)`, `expand_with(ExpandOpts)`, `nsimplify`, `rcollect`, `subs_algebraic`.

→ [The Rule Engine](./guide/rule-engine.md)

## Matrices

The eigen family (`eigenvals`, `eigenvects`, `diagonalize`, `jordan_form`, `matrix_exp`) no longer takes a dummy variable. Irreducible cubic/quartic characteristic polynomials give exact, evaluable `RootOf` eigenvalues instead of Cardano swell. New: `qr`, `ldl`, `gram_schmidt`, `matrix_exp_t`, `matrix_pow_symbolic`, `matrix_sqrt`, `hessian`, `wronskian`, structure tests, norms, least squares, `Index`/`IndexMut`, scalar operators on both sides.

→ [Matrices](./guide/matrices.md)

## Transforms

One-sided limits (`limit_left`/`limit_right`/`limit_dir`), Fourier transforms in three conventions, Mellin transforms with their fundamental strip, Laplace table/inverse extensions and initial/final-value theorems, Fourier series on arbitrary intervals with exact coefficients, Z-transform extensions.

→ [Transforms](./guide/transforms.md)

## Number theory and factoring

Berlekamp–Zassenhaus factoring for any degree, multivariate factoring, a polynomial-algebra API on `Ex` (`resultant`, `discriminant`, `sqf_list`, `poly_div`, `poly_gcdex`, `nroots`, `real_roots_isolate`, …), `factorint` with Pollard–Brent rho + ECM, BPSW `isprime`, `sqrt_mod`, `discrete_log`, `primitive_root`, `primepi`, continued fractions, Egyptian fractions, Pell equations, sums of squares, Pythagorean triples, and integer sequences.

→ [Number Theory and Combinatorics](./guide/number-theory.md)

## Code generation

`compile()` returns `Result<CompiledFn>` and covers every numerically evaluable node; `compile_many` shares CSE across a gradient. `to_rust_fn` embeds only the special-function helpers it needs. New **C99 backend** (`to_c_fn`). `CodegenOptions::{use_mul_add, checked_domain, emit_runtime}` and `runtime_module()`/`c_runtime()` for multi-function files.

→ [Code Generation](./guide/code-generation.md)

## Ergonomics

`Context::{from_f64, from_f64_approx, from_f64_nice, from_bigint, from_ratio, from_i128, rational_str, decimal_str, complex, symbols, symbols_indexed, apply, sum, product}`; operators with `f64`/`i32`/`u64`/`i128`/`BigInt`/`Ratio` and compound assignment; `Ex::{as_rational, as_bigint, as_i64, compare_numeric, is_less_than, probably_equal, eval_at}`; `Equation` arithmetic and `solve_for`; `Debug for Ex` prints the expression.

## Companion crates

- `symplex-wasm`: a persistent `Session` (define names, evaluate, differentiate, solve) plus a full stateless API including `integrate_definite` and `to_c_fn`.
- `symplex-build`: exact DH parameters (`0.3` → `3/10`), `generate_fk_matrix`, `"fk_matrix"` in TOML configs.
