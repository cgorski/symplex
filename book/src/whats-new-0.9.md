# What's New in 0.9

symplex 0.9.0 is a **comprehensiveness** release — additive over 0.8.2 — driven by a module-by-module survey of the crate against SymPy 1.14. Four areas got a first thorough pass; every new function is tested against reference values from that SymPy. The [CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md) has the full list.

## Algebraic numbers and polynomial algebra

`Ex::minimal_polynomial(&x)` gives the minimal polynomial over ℚ of an algebraic-number expression (`√2 + √3` → `x⁴ − 10x² + 1`), `gcd_all`/`lcm_all` are the variable-free multivariate gcd and lcm, `Ex::groebner(&polys, &vars, MonomialOrder::Lex)` and `reduce_modulo` expose Gröbner bases and normal forms without the `MultiPoly` dance, `real_roots(&x)` returns the real roots of a rational polynomial as exact `RootOf`s in increasing order, `factor_mod(&x, p)` factors over GF(p), and `resultant_symbolic`/`discriminant_symbolic` handle polynomials whose other coefficients are symbols (`disc(ax² + bx + c) = b² − 4ac`).

## Special functions

Twenty-four functions that appear as integration results or in physics: `erfi`, `erfinv`, `erfcinv`, `expint`/`E1`, `Shi`, `Chi`, `fresnels`, `fresnelc`, `lowergamma`, `uppergamma`, `polylog`, `dirichlet_eta`, the Airy functions and their derivatives, complete and incomplete elliptic integrals (`elliptic_k/e/f/pi`), and the Gegenbauer, Jacobi, associated Legendre and associated Laguerre polynomials. Each has exact special values, a derivative rule, arbitrary-precision `evalf` (checked at 40 digits against mpmath on every branch — series, asymptotic, continued fraction, reflection), `Display`, LaTeX and `parse` support. `integrate` now reaches `∫e^{x²} = (√π/2)·erfi(x)`, `∫sinh(x)/x = Shi(x)`, `∫cosh(x)/x = Chi(x)`.

## Analysing a function

SymPy's `calculus.util` on `Ex`: `singularities`, `stationary_points`, `maximum`/`minimum` on unions of intervals (with one-sided limits at open or infinite endpoints), `is_increasing`/`is_decreasing`/`is_monotonic`/`is_convex` (exact for polynomial and rational derivatives via Sturm sequences; three-valued, never a guess), `periodicity` and `function_range`.

## Matrices

`singular_values`, `condition_number`, a `pinv` that is now defined for every matrix (rank-deficient inputs go through the full-rank factorisation), `rank_decomposition`, `hessenberg`, `companion`, `jordan_block`, `permanent`, row/column insertion, deletion and permutation, `inv_mod`, `matrix_log`, `casoratian`, and an exact `lll` lattice reduction on `ZMatrix` with rational Gram–Schmidt.
