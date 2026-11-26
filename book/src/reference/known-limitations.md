# Known Limitations

What is not in the library yet, and what is in it and known to be imperfect, at 0.60. Every example on
this page was run against that code. The `CHANGELOG.md` entry of each release has a *Known issues*
section with more detail; if you meet something that is not listed here or there, it is a bug worth
reporting.

## Not there

- **Hypergeometric and Meijer G functions.** There is no `pFq`, no Meijer G, no Tricomi `U`, no public
  Hurwitz zeta, no inverse incomplete gamma or beta; elliptic `E` and `Π` are the complete integrals
  only. The integrals, sums and transforms whose answers need them stay unevaluated: of the 72,254
  integrands of the Rubi suite, 41,237 are left as an `Integral` (30,768 are answered and verified).
- **Systems of nonlinear equations.** `polysys::solve_system_ex` solves polynomial systems with
  finitely many solutions. A system with infinitely many is `Err(InfiniteSolutions)` (no parametric
  description), and a non-polynomial system has no symbolic solver; `solve_numeric_system` is a
  Newton iteration from a starting point.
- **Geometry, tensor algebra, PDEs, group theory, quantum mechanics.** Not implemented.
- **Series.** No `O()` term; the two-sided `series` refuses fractional powers (`fps` gives Puiseux
  series). **Summation:** no Zeilberger algorithm. **ODEs:** no power-series solutions.
  **Recurrences:** linear with constant coefficients, and first order.
- **Input.** The parser reads symplex's own notation; there is no LaTeX or Mathematica input.
- **Logic.** `satisfiable` answers yes / no / unknown without an assignment, for at most 24 atoms, and
  does not reason about arithmetic: `x > y ∧ x > 0` is `None`.
- **Linear algebra.** Singular values but no full SVD; no sparse matrices; no matrix expressions.
- **Statistics.** 22 distribution families and finite tables (`sympy.stats` has about three times
  as many); joint distributions only as independent variables or the multivariate normal.
- **Code generation.** `expint`, `polylog`, `zeta`, `polygamma`, the Airy functions, elliptic `F` and
  `Π` and `dirichlet_eta` are not compiled by any back end (`Err(NotImplemented)`), and neither are
  the incomplete gamma and beta functions and the Gegenbauer, Jacobi and associated Legendre /
  Laguerre polynomials with a symbolic parameter; with literal parameters (`lowergamma(5/2, x)`,
  `gegenbauer(3, a, x)`) `eval` expands them and `compile` takes the expansion, while `to_rust_fn`
  and `to_c_fn` still refuse.
- **Lean export.** Elementary expressions and the certificates; special functions are
  `Err(NotImplemented)`.

## Known to be imperfect

**Numerical evaluation**

- **A numeric integral can miss a narrow part of its integrand and certify the rest.** The density of
  `1/2·Gamma(5/2, 10⁻²⁰) + 1/2·Normal(0, 1)` over the real line integrates to `0.5`:
  `integrate_numeric` returns `Ok(0.5)` with a claimed error of `1.2e-11`, and
  `integrate_numeric_decimal` prints `0.5` with 12 and with 30 certified digits. The integral is 1.
  The gamma half lies within `10⁻²⁰` of 0, behind the values of the normal half; no node of the
  quadrature reaches it. With a scale of `10⁻⁶` only `integrate_numeric` is wrong. Expect this
  wherever the parts of a sum live on scales that differ by many powers of ten: of 1,452 integrals
  of such mixtures (scales `10³` to `10³⁰⁰` apart) 473 values of `integrate_numeric` and 378 at 12
  certified digits are wrong in 0.60.1. The closed forms of the statistics module (`cdf`,
  `characteristic_function` where the law has them) do not go through the quadrature.
- `integrate_numeric` returns `0.0` for a density whose mass lies between two nodes of the
  quadrature: `LogNormal(-300, 1)` over `[0, ∞)`, whose integral is 1.
- Since 0.60.1 `integrate_numeric` refuses an estimate that is not larger than its own error
  estimate (it returned `6.13e-66` for the density of `Normal(0, 10⁸⁰)`, which integrates to 1),
  and an integral of which a piece next to a far singular point was seen by no node. Such a
  refusal can take up to 4 s, and some refused integrals have a value: for `Normal(0, 10⁸⁰)`,
  `integrate_numeric_decimal` at 12 digits answers `1`.
- `eval_decimal(n)` prints digits inside a tracked error bound or returns an error. The bound is
  proved for the elementary functions and most special-function kernels; for some kernels — among
  them Bessel functions of large order, the orthogonal polynomials and `ζ` left of 0 — its
  derivative part is *estimated* (by re-evaluation at a nudged argument).
- Next to a zero of the function, the bound of a value computed by a three-term recurrence (an
  orthogonal polynomial of a large degree) is an estimate with a margin, not a proof, and refusals are
  not monotone in the digits: `besselj(1000, 200273630423430773792713/10^20)` prints at 16 and 30
  digits and is `PrecisionExhausted` at 50.
- The quadratures (`integrate_numeric`, `integrate_numeric_decimal`, `eval_decimal` of an
  unevaluated integral) accept a value when two rules or two levels agree. A spike much narrower
  than the node spacing, or a breakpoint at a huge offset that rounding moves, can be missed
  without an error.
- A part of a value that is not 0 but lies below the exponent range is refused in most cases and
  still dropped in some: `gamma(pi - I*exp(-3*10^9)) + sqrt(2)` prints a real number.
- Refused although a value exists: `elliptic_pi(1/3, 10^600 + I)` and `ln(-2 + I*exp(-10^10))`
  (`PrecisionExhausted`).
- Refused although a value exists, also: the incomplete gamma functions of a shape above about
  `3·10⁹` next to the mode (the distribution function of `Poisson(2.8·10¹⁰)` at its median);
  `uppergamma(1/10, 4/3)` and `expint(2, 4/3)` at 600 digits; `besselj(8171/4, 9805/4)` from 71
  digits on.
- `eval_f64` of an expression that holds an unevaluated definite integral is certified to 8 digits
  (a relative error below `2⁻³⁰`), not rounded to the nearest `f64`: of 416 integrals tried, 149
  came out as the nearest `f64`. Parse `eval_decimal(20)` when you need the nearest one.
- An integral whose mass lies in a thin layer at an end of its range is refused, not evaluated: the
  moment generating function of `FDistribution(11, 2.3·10⁻²⁵³)` at `-1/8` has the value
  `6.5973·10⁻²⁵¹` and is `PrecisionExhausted`.
- The error bound of `binomial(n, k)` and `beta(a, b)` formed as a quotient of three Gamma values is
  not proved (no wrong digit in 153,920 values checked).
- Compiled code (`compile`, `to_rust_fn`, `to_c_fn`) is accurate to about a unit in the last place,
  not correctly rounded: a product of constants is rounded more than once (`expint(0, sqrt(2))/x` at
  `x = 0.7` is 1.1 units off).

**Symbolic results**

- Two folds are applied too widely: `polylog(s, 1)` becomes `zeta(s)` for every `s` (true for
  `Re s > 1`; `polylog(0, 1)` gives `-1/2` although `Li₀` has a pole at 1), and `uppergamma(y, 0)`
  becomes `Gamma(y)` for a symbol without assumptions (true for `Re y > 0`).
- Powers of 0 and step functions are generic in places: `0*0^y` is `0`, `Heaviside(y)/Heaviside(y)`
  is `1`, `zoo*stirling2(0, 8)` is `zoo` (the second factor is 0).
- A set expression stays unevaluated where a member is real or not real but not proved so: `{p + I} \
  ℝ` for a positive `p` stays `{p + I} \ (-oo, oo)` (it is `{p + I}`), and `reals.contains(p + I)` is
  `None`.
- An inequality with a side that is not real through a constant without a literal `I`, next to the
  variable, still gets a set: `t + acos(2) > 2*t + acos(2)` for a real `t` is `(-oo, 0)` (it holds
  nowhere; with a literal `I` the result is `EmptySet`).
- `eigenvals` is an error where a factor of the characteristic polynomial has no exact roots in the
  library: an irreducible factor of degree 5 or more with symbolic coefficients, or of degree 3 or
  more with `π`, `e` or `ln 2` in it. Only a matrix of rational numbers always has its eigenvalues (as
  `RootOf`).
- Symbolic matrices: an entry that the zero test can neither prove zero nor prove nonzero is a
  refusal in the functions that return `Result` (`solve`, `lu`, `inv`, `pinv`, `qr`,
  `rank_decomposition`, `eigenvects`, `diagonalize` …: `Err(ComputationFailed)` naming the entry).
  `rref`, `rank`, `nullspace`, `columnspace`, `rowspace` and `left_nullspace` return plain values
  and count such an entry as nonzero: `Matrix [[gamma(x + 1) - x*gamma(x)]]` has `rank()` 1 although
  the entry is 0. The zero test does not know four identities — `atan(x) + atan(1/x) - pi/2` for
  `x > 0`, `gamma(x + 1) - x*gamma(x)`, `asin(x) + acos(x) - pi/2`, `besselj(1/2, x) -
  sqrt(2/(pi*x))*sin(x)` — nor a sum of two different constant identities. Where every pivot is
  decided, the result still holds for most values of the symbols, not for every value.
- Matrix verdicts with symbols: `is_diagonalizable` is `Some(true)` only when the matrix is
  diagonalizable for every value of its symbols, but `Some(false)` is the answer for most values
  (`[[1, x], [0, 1]]` is `Some(false)` and is the identity matrix at `x = 0`).
  `StateSpace::is_controllable` and `is_observable` answer for most values too (`A = diag(1, x)`, `B
  = (1, 1)ᵀ` is `true` and is not controllable at `x = 1`), and their `false` also means "not
  proved".
- `eigenvects`, `diagonalize` and `jordan_form` refuse a matrix whose eigenvalues they cannot prove
  different, or whose zero tests exceed their work budget, although an answer exists.
- `series` keeps a function at a pole as a coefficient (`x^2*expint(1, 0)` expands to itself) where
  `limit` refuses; `limit(gamma(floor(y) - 3 + x), x, 0)` is `Gamma(floor(y) - 3)`, a pole for `y < 4`.
- `isprime` beyond `3.3·10²⁴` is the Baillie–PSW test: no counterexample is known, but it is not a
  proof. `is_prime_value()` reports that verdict; `is_prime()` answers only on a proof and is `None`
  for a probable prime beyond the bound (`2^89 - 1`).
- `inverse_mellin_transform(&s, &x)` cannot be told the strip of the transform and takes the one to
  the right of the pole: for `-1/s`, the transform of `H(x - 1)` on `re(s) < 0`, it returns `-H(1 -
  x)`. `inverse_mellin_transform_in_strip(&s, &x, &lower, &upper)` inverts on the strip you give.
- `check_ode_solution` returns `false` for a correct solution that it cannot prove (8 of 253 tried):
  `false` means "not proved", not "wrong".
- `probably_equal` answers `None` when it cannot draw a value of the declared kind (a symbol
  declared irrational), and `Equation::is_identity` tries six fixed points: an equation that fails
  only elsewhere stays `None`.
- An order relation with a side proved non-real (`I < 1`) is `None` in `is_less_than` and
  `is_greater_than`, not `Some(false)`.
- The ASCII pretty printer writes `lgamma(x)` and `psi(x)`, which the parser does not read back.

**Waiting for the 1.0 interface**

Each of these is confirmed by a test; the fix changes a public signature and comes with the
interface work for 1.0.

- Declaring a symbol again with other assumptions (`symbol_with` on an existing name) leaves the
  expressions built before: for an integer `n`, `cos(pi*n)` is built as `(-1)^n` and stays so after
  `n` is declared merely real.
- A symbol named `i`, `e`, `pi`, `oo`, `nan` … prints as the constant of that name, and `parse`
  reads the text back as the constant (JSON keeps the symbol). `expr!` reads `E` and `I` as the
  constants even when variables of those names are in scope.
- `Distribution::kurtosis` is `E[(X - μ)⁴]/σ⁴`; `stats::data::kurtosis` is that minus 3.
- The plain distribution constructors do not check their parameters (`Distribution::normal(0, -1)`
  has a negative density); the `try_` constructors do. `logit` returns a fit that did not converge
  as `Ok` with `converged == false`.
- `solve` reports "no root in the declared domain" as `Ok([])` for a polynomial and as
  `Err(NoSolution)` for any other equation.
- `Assumption::NotCommutative` and five more flags are accepted and read by nothing: `A*B - B*A` is
  `0`.
- An `f64` NaN or infinity as an operand (`&x * f64::NAN`) silently becomes the `nan` or `oo` node,
  and a number where a variable belongs is not always refused (`sin(x).limit(&zero, &x)` builds
  `Limit(sin(x), 0, x)`).
- `dim!(ctx, Dimensionless: sin(len))` compiles for a `Length`; `sym!` works only in a function that
  returns `Result`.
- `ntheory::mod_pow` with a modulus `≤ 0` or a negative exponent of a base without an inverse,
  `totient(0)`, `mobius(0)`, `divisors(0)`, `multiplicity(p, 0)` and `is_quad_residue(a, 0)` return
  a documented stand-in instead of an error.

**Refusals that have an answer**

- `try_` twins, and the Fourier and Mellin transforms, refuse a result that merely carries an
  unevaluated node over from the input (`Sum(sin(k^2), k, 1, m)*exp(-t)` has a `laplace`, no
  `try_laplace`).
- `try_integrate_definite(sin(x), 0, oo)` is `ComputationFailed`, not `Divergent`; the engines have
  no expansion of `lowergamma(s, x)` at 0 (`limit(lowergamma(-1/2, x), x, 0⁺)` is refused).
- Extreme parameters in statistics: the lower tail of a gamma or χ² law with a shape from about
  `4.5·10⁹` (`chi_squared(34000000000).cdf(33999990340.5)`), densities with both shapes huge
  (`F(3e128, 3e128)`, `Beta(1e128, 1e128)`), a windowed sum beyond 65,536 terms (a binomial with
  `n·p·(1 − p)` near `4·10⁶`), the exact Clopper–Pearson interval beyond 120 trials.
- The envelope of the statistics module. The library aims to answer for finite `f64` arguments,
  distribution parameters between `2⁻⁵³` and `2⁵³` in magnitude, and probabilities and levels down
  to `10⁻³⁰⁰`; beyond that a refusal is the intended result where an answer would be slow. Since
  0.60.1 the incomplete beta function with `a + b` above `2⁵⁴` has a budget of work and refuses
  some values it computed in 0.35 to 2.3 s before (a first shape between `10¹⁶⁰` and `10³⁰⁰`, a
  second between `10⁸` and `10¹⁴`, an argument within `10⁻¹⁴⁰` of 1).
- `1/3·Geometric(1/3) + 2/3·Geometric(10⁻¹²)`: `quantile_f64` is refused ("precision exhausted")
  wherever the answer lies in the second component's range.

**Size and time**

- Some functions that build a list as long as a count they are given still abort or do not return
  for a huge count: `Matrix::identity`/`zeros`, `symbols_indexed`, `equispaced_grid`,
  `finite_diff_weights`, `pell_solutions`, and the sequences `bell`, `partition_count`, `stirling2`,
  `divisor_sigma`, `diff_n`, `Distribution::moment`. Within the caps some calls are slow
  (`coefficients(16385)` of `sin`: more than 20 s).
- A Gröbner basis stops at a work budget of about five seconds, which cannot be raised: the cyclic
  system in six variables is refused (`BudgetExhausted`). The plain `groebner::groebner_basis` then
  returns the polynomials it was given; test the result with `is_groebner_basis`, or call
  `try_groebner_basis`.
- A table from `transformed` of a lattice with unequal masses (`Binomial`, `Hypergeometric`) sums
  its masses as expressions at every query: `√X` for `X ~ Binomial(9000, 1/3)`, one `cdf` and one
  quantile, takes 35 s.
- A characteristic or moment generating function that has no closed form is left as an integral,
  and its value, like a moment computed by quadrature, takes seconds or is refused after seconds:
  `FDistribution(11/8·e, 31/13).characteristic_function(-1/8).eval_complex64()` is
  `PrecisionExhausted` after 2.95 s; the variance of `StudentT(7/2)` truncated to `[-6.1·10⁸⁵,
  4.4·10²⁸]` takes 1.4 s.
- Quantiles of discrete laws that are still slow: a mixture with a large hypergeometric component
  (`1/2·Hypergeometric(90000, 36562, 78750) + 1/2·Poisson(30000)`: up to 5.1 s per quantile);
  beyond the envelope, `Hypergeometric(10³⁰, 10²⁹, 1000)` (up to 25 s per call).
  `NegativeBinomial(r, p).cdf(k)` is a sum of its terms: at `k = 1000` it takes 0.3 to 1.9 s, or
  is refused after 5.6 to 7.2 s where the value underflows (`NegativeBinomial(10¹⁵, 1/100)`).
- `order_statistic` of a law on a finite range of integers returns the `OrderStatistic` family
  instead of a table where the table's fractions would be too long (`Binomial(843, p)` with a
  14-digit `p`); a quantile of that family takes 0.47 to 8.1 s and its `mean()` and `variance()`
  do not return in 20 s. With a parameter that is not a fraction (`Binomial(30, 1/π)`) building
  the table takes 11 to 15 s for five draws.
- `integrate` is slow on some products of a high power and an exponential that it does not find:
  the integrand of the moment generating function of `FDistribution(d₁, 1)` takes 0.55 s at `d₁ =
  801` and 9.7 s at 1999; `x^(n/2)·e^(-x/2)` times a polynomial for odd `n` from 13 to 61 takes 0.5
  to 2.1 s.
- The library bounds its work on the common paths, not on every path. For input you do not control,
  see `SECURITY.md`, "Running symplex on untrusted input".

## How it is tested, and how stable it is

- **Platform.** The test suite runs on Linux x86-64 only. `wasm32-unknown-unknown` is type-checked;
  macOS, Windows, 32-bit and big-endian targets are untested.
- **Oracles.** The symbolic engines are tested against SymPy, mpmath, scipy and statsmodels, not
  proved. Two comparisons run continuously: the Rubi suite (0 wrong answers) and the nightly
  statistics oracle. Since 0.60.1 the nightly statistics fuzzer draws distribution parameters
  inside the envelope `2⁻⁵³ … 2⁵³` (wider draws are run before a release). It is not known to be
  green yet: the last long run, made before the statistics fixes of 0.60.1 were merged, still
  saved about 4 inputs a minute — slow calls of the kinds listed under "Size and time", and one
  failed check, the missed narrow part of a numeric integral. It skips five slow cases
  (`fuzz/README.md`): a lattice with unequal masses of more than 512 atoms is not transformed into a table, a lattice of
  more than 4,096 atoms whose tails must be summed mass by mass is not checked, without a finite
  variance the first moment is checked only while `|mean| ≤ 10⁸`, an integral whose integrand
  leaves the range of `f64` at a sample point is not evaluated, and a characteristic function is
  compared with its quadrature only for tails of order 4 and more.
- **Less tested.** The certificate provers, the exact LP and the Lean export are covered by their
  own exact re-verification and by byte-identity tests, but have not been hunted for wrong answers
  the way calculus and statistics have; neither have the code generators for the functions added
  most recently, units, the WebAssembly bindings, the matrix decompositions and plotting. Lean text
  is compared with fixtures compiled at Lean 4.30.0; CI does not run Lean.
- **Speed** has not been measured against other libraries.
- **API stability.** Before 1.0 a minor release may break the API or change a result; the CHANGELOG
  lists such changes first. Only `LeanOpts`, `PolyhedronOpts`, `SosOpts` and `Budget` are
  `#[non_exhaustive]`: adding a field to any other option struct, or a variant to most public
  enums, is a breaking change. There is no promise yet about which printed forms (`Display`,
  LaTeX) stay stable, and types of the `num-*` crates (themselves before 1.0) appear in public
  signatures.
