# What's New in 0.3

symplex 0.3 is an additive release. Nothing in the public API changed signature; the [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the complete list, and the [behaviour changes](#behaviour-changes) at the end of this page are the only things an upgrading program might notice.

The theme of the release is **polynomials and certificates as data**. 0.2 could *tell* you that `(x + 1)² = 2(x + 1) + (x² − 1)`; 0.3 lets you *ask* for the multipliers — exactly, over ℚ, with non-negativity constraints if you need them — and hand back a proof a reader can check by expanding. Around that core sit four new public modules: `poly_ex` (the `Poly` view), `linprog` (exact simplex), `normalforms` (Hermite and Smith forms over ℤ) and `optimize` (deterministic `f64` root finding, minimisation and fitting).

## By the numbers

Measured while preparing this page (`grep -c '#[test]'` over `src/` and `tests/`; `wc -l` over `src/**/*.rs`):

| | 0.2 | 0.3 |
|--|-----|-----|
| `#[test]` functions | 10,177 | 11,021 |
| Lines in `src/` | ~163K | ~174K |
| New public modules | — | `poly_ex`, `linprog`, `normalforms`, `optimize` |
| Signature-breaking changes | many (see the [0.2 migration guide](./reference/migrating-0.2.md)) | 0 |

## Polynomial views

`Ex::as_poly(&[&x, &y])` (or `Poly::new`) views an expression as a sparse polynomial in explicit generators, with coefficients that are `Ex` values — exact rationals *or* symbolic parameters such as `a + 1`. Terms come back in lex-descending order (SymPy's `Poly.terms()`), and you get `coeff_monomial`, `total_degree`, `degree_in`, `degree_list`, `leading_coeff`, `all_coeffs`, exact `eval` at any point, partial evaluation `eval_gen`, `add`/`sub`/`mul`/`pow`/`derivative`/`scale`, `content_and_primitive`, `monic`, `nroots`, and round-trips to the Gröbner-basis representation (`to_multipoly` / `from_multipoly`). `Poly::monomial_basis` and `Poly::coefficient_matrix` turn a family of polynomials into a `Matrix` — one row per monomial, one column per polynomial — so "find λ with `goal = Σ λᵢhᵢ`" becomes a call to `linsolve_matrix` or to the LP solver.

→ [Polynomials as Data](./guide/polynomials.md)

## Rational normal forms and `solve` output

`Ex::ratsimp` is a canonical form for rational expressions in all variables at once: one fraction, common factors cancelled by a multivariate GCD, integer-primitive numerator and denominator, positive leading coefficient in the denominator. Non-rational subexpressions (`sin x`, `π`, `√x`) are opaque indeterminates, as in SymPy's `cancel`. `simplify_rational` now *is* `ratsimp`, and `solve` applies it to solutions with symbolic coefficients, so `((3r − 1)/(j + 1) − (r + 1)/(2j)).solve(&r)` returns `(3*j + 1)/(5*j - 1)` instead of a fraction of fractions. `degree`, `coeffs`, `coeff`, `leading_coeff` and `is_polynomial` accept parameter coefficients. `poly_is_nonnegative_on(&x, &lo, &hi)` and `poly_is_positive_on` decide the sign of a univariate polynomial on an interval exactly (square-free decomposition + Sturm count), with endpoints that may be `±∞`.

→ [Algebra: `ratsimp`](./guide/algebra.md#rational-normal-form-ratsimp) · [Solving: symbolic coefficients](./guide/solving.md#symbolic-coefficients-are-returned-in-rational-normal-form) · [Sign on an interval](./guide/polynomials.md#sign-of-a-polynomial-on-an-interval)

## Exact linear programming

`symplex::linprog` is a two-phase dense simplex over `Ratio<BigInt>` with Bland's rule (no cycling), a builder (`LpProblem::maximize(c).le(row, rhs).ge(…).eq(…).bounds(j, lo, hi).free(j).solve()`), a SciPy-shaped `linprog`, a `Matrix` front end `linprog_matrix`, and `feasible_nonneg` for the question "is there an `x ≥ 0` with `A·x = b`?". `LpStatus` is `Optimal` / `Infeasible` / `Unbounded` — none of them an error. An optimal solution carries exact **shadow prices** (`duals`, with complementary slackness and strong duality holding as identities); an infeasible one carries an exact **Farkas certificate** (`farkas`) proving that no feasible point exists. Sizes up to about 100 rows × 200 variables solve in seconds in release builds.

→ [Exact Linear Programming](./guide/exact-lp.md) · [Cookbook: Polynomial Inequality Certificates](./cookbook/polynomial-certificates.md)

## Integer lattices and normal forms

`symplex::normalforms` computes, over `BigInt`, the row-style Hermite normal form `H = U·A` (unique; `hermite_normal_form_with_transform` returns the unimodular `U`), the column-style `H = A·V` in SymPy's convention (`column_hermite_normal_form`), the Smith normal form `S = U·A·V` with its invariant factors (`smith_normal_form[_with_transforms]`), a ℤ-basis of the integer kernel (`integer_nullspace` — strictly more than the rational nullspace scaled up), `is_unimodular` and `lattice_determinant` (the index of a column lattice in ℤᵐ). `Matrix` gained `hermite_normal_form`, `smith_normal_form` and `integer_nullspace` methods. Non-integer entries are an `InvalidArgument`, never a rounding. `ntheory` gained `gcd_many`, `lcm_many`, `igcd`, `ilcm` and `rational_lcm_of_denominators`.

→ [Integer Lattices and Normal Forms](./guide/integer-lattices.md)

## Matrix ergonomics

`extract(&rows, &cols)`, `select_rows`, `select_cols`, `delete_row`, `delete_col`; the three-valued `is_integer_matrix` (next to the existing `is_zero`); `nnz`; `subs_map` (simultaneous substitution); and exact conversions to and from the `num` types — `to_rational_rows`, `to_bigint_rows`, `Matrix::from_ratio`, `Matrix::from_bigint`, `Matrix::from_f64_rows` (each float becomes the exact dyadic rational it denotes). These are the glue between `Matrix` and the LP / normal-form modules.

→ [Matrices: selecting sub-matrices](./guide/matrices.md#selecting-sub-matrices-and-exact-conversion) · [Integer normal forms](./guide/matrices.md#integer-normal-forms)

## Numerical optimisation and fitting

`symplex::optimize`: `brent_root` (Brent–Dekker), `bisect`, `newton_root`; `nelder_mead` (dimension-adaptive coefficients, `NaN` treated as `+∞`); `minimize_scalar` (Brent) and `golden_section`; `differential_evolution` (DE/rand/1/bin with a Nelder–Mead polish, seeded SplitMix64 — bit-identical results for the same seed); `poly_fit` (Householder QR, **ascending** coefficients — NumPy's `polyfit` is highest-degree first), `poly_fit_exact` over ℚ, `linear_fit`, `eval_poly`, `trapezoid`. On `Ex`: `find_root_bracket`, `minimize_numeric`, `minimize_scalar_numeric`, `minimize_global_numeric`, `poly_fit_points` — each compiles the expression first, so a stray free symbol is a `FreeSymbol` error rather than a `NaN`. Every routine is bounded by an explicit iteration budget and never panics; the minimisers report an exhausted budget through `MinimizeResult::converged` so the best point is never discarded.

→ [Numerical Optimisation](./guide/numerical-optimization.md)

## Behaviour changes

There are **no signature-breaking changes** in 0.3. Two behaviours changed in ways an existing program could observe:

- `degree`, `coeffs`, `coeff`, `leading_coeff` and `is_polynomial` now **succeed** for polynomials whose coefficients are symbolic parameters (`a·x² + (a + 1)·x + 3` in `x`). In 0.2 they returned `None` / `false`. Code that used `None` from these methods to mean "has parameters" should test `Poly::has_rational_coeffs` or `free_symbols` instead.
- `simplify_rational` and the results of `solve` with symbolic coefficients are now put into rational normal form with `ratsimp`. The values are mathematically equal to the 0.2 results; only the **printed form** differs (a single cancelled fraction, e.g. `(3*j + 1)/(5*j - 1)`). Tests that compare against a string may need updating; tests that compare with `equals` or by substitution do not.

Everything else on this page is new API.
