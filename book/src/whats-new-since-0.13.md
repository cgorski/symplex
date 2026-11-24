# What's New since 0.13

One page for the releases 0.14 to 0.59.  For each release it names the two or three changes a user
must know, changes of behaviour first, in the words of that release's entry in the
[CHANGELOG](https://github.com/cgorski/symplex/blob/main/CHANGELOG.md).  The CHANGELOG has the rest:
every breaking change under `### Breaking`, every fix with the oracle that confirmed it, and the
measurements.  The releases up to 0.13 have pages of their own
([0.13 / 0.14](./whats-new-0.13.md) and earlier).

From 0.23 on, almost every release changes *results* rather than signatures: an answer that was
wrong is corrected, or a claim that could not be proved is refused (an `Err`, an unevaluated node, a
`ConditionSet`).  A program that upgrades across several releases should read the parts marked
**Behaviour** or **Breaking**.

## 0.59 (this release)

- **Behaviour — powers of 0.**  `0^y` with a symbolic exponent is 0 for `re y > 0`, 1 at `y = 0` and
  `zoo` for `re y < 0`, and results keep that case distinction: `0^y·0^(−y)` was `1` and is
  `Piecewise((1, y = 0), (nan, True))`; `0/H(y)`, `0/floor(y)` and `zoo·0^y` are `Piecewise`s (they
  were `0`, `0` and `zoo`); `limit(x/0^y, x, 0)` was 0 and is `Piecewise((0, re(y) < 0 ∨ y = 0))`.
- **Behaviour — folds and limits.**  `lowergamma(s, 0)` folds to 0 only where `re s > 0` is known
  (declare the symbol positive for the old value); `limit(KroneckerDelta(1, x + 1), x, 0)` is 0 (was
  1); an infinite limit whose direction depends on a parameter keeps its finite part, and the
  two-sided limit is refused where the two sides differ in it.
- **Behaviour — sizes are bounded.**  Calls that aborted the process or did not return are refused:
  `FormalPowerSeries::coefficients(n)` returns an empty list beyond `n = 2¹⁴ + 1`, and
  `ntheory::fibonacci`/`lucas` return 0 beyond `|n| = 2²⁷`.  Each such function has a new `try_`
  twin that reports the refusal (`try_coefficients`, `try_fibonacci`, `try_lucas`, `try_primorial`,
  `try_primorial_up_to`, `discrete::try_convolution_cyclic`).  `Context::from_tree` no longer
  overflows the stack on a deeply nested tree.
- **Behaviour — numerical evaluation.**  A real or imaginary part that is certainly not 0 is never
  dropped, also below the exponent range: `1 + i·exp(−10¹⁰)` printed `1` and is now refused
  (`SymplexError::Unevaluable`).
- New values: `evalf` of long sums whose terms peak far from their ends (the entropy of
  `Binomial(10⁵, 13/100)`), densities of `F` and Student t with huge degrees of freedom, integrals
  of rational functions of `c^x` for a complex `c`, `singularities` of a solution family with a
  complex step.
- The repository has a `SECURITY.md` (how to report an input that crashes or hangs the parsers or
  the JSON import), and all four crates are `#![forbid(unsafe_code)]`.

## 0.50 – 0.58

- **0.58.**  *Behaviour:* a value that is infinite is the new error variant
  `SymplexError::Infinite` (`eval_decimal` of `expint(1, 0)`, `elliptic_k(1)` was `Unevaluable`),
  and `solve` no longer claims roots at an infinite value; `Max`/`Min` of a non-real argument is
  `nan`; zero-variance errors in the hypothesis tests, effect sizes and reliability coefficients are
  `DegenerateData { kind: ZeroVariance }` (they were `InvalidArgument`).  Lengths read from the
  input are bounded before they size an allocation.
- **0.57.**  *Behaviour:* degenerate data in the statistical models (rank-deficient designs,
  separation, constant responses, zero-variance groups, no events) returns the new `DegenerateData`
  (it was `InvalidArgument` or `ComputationFailed`); the `try_` constructors of the distributions
  reject non-finite, non-real and irrational invalid parameters.  Every family, test and model was
  compared with scipy, statsmodels and mpmath, and a nightly workflow keeps comparing them.
- **0.56.**  *Behaviour:* reversed limits of a finite product follow Karr's convention, like sums;
  `Max`/`Min` read their arguments as real; a fold written for `±∞` no longer fires at `zoo`.
  `Distribution::transformed` answers the monotone maps it refused (`ln X`, `√X`, `atan X`).
- **0.55.**  *Behaviour:* a finite sum's closed form holds for every integer value of a symbolic
  bound or comes with its condition (`Σ_{k=0}^{n} C(n, k) = 2ⁿ` was `1/2` at `n = −1`); reversed
  limits follow Karr's convention, as SymPy's do.
- **0.54.**  *Behaviour:* a symbol whose assumptions pin it to one value is that value (even and
  prime: 2); infinite sums with a symbolic ratio return their closed form under its exact
  convergence condition; `compile` refuses a constant now known not to be real.
- **0.53.**  *Behaviour:* an answer is a claim that must hold for every value it covers.  A root
  that holds only for some values of a parameter comes with its exact condition (`solve_as_set`)
  instead of being returned as if it always held; a quotient whose denominator vanishes identically
  is `nan`; a non-real value is never printed or returned as its real part.
- **0.52.**  *Behaviour:* the poles of `tan` are not roots; an equation `c = 0` for a constant `c`
  not proved nonzero is refused instead of answered "no solution"; unions of real intervals are
  ordered ascending.
- **0.51.**  *Behaviour:* inverting a principal power needs the right-hand side in its range (roots
  outside it were returned); `DiracDelta` of a non-real argument has no value.
- **0.50.**  *Behaviour:* the solvers work over the declared domain of the unknown everywhere
  (`solve_general` gives complex families; parameters may be complex); the Bernoulli numbers use
  `B₁ = +1/2`; `Context::apply` with a built-in name builds the built-in function.  Rust and C emit
  the exponential and trigonometric integrals; `compile` handles the complete elliptic integrals.

## 0.40 – 0.49

- **0.49.**  *Breaking:* expression identifiers have no order (`ExprId` no longer implements
  `Ord`); `SymplexError` gains `BudgetExhausted`, `ExpressionSwell` and `NotDiagonalizable`, so
  errors are told apart by variant (same messages as before).  *Behaviour:* `solve` reports partial
  answers as failures — `ComputationFailed` naming the roots found, where it returned them as if
  they were all; `solve_as_set` returns them ∪ a `ConditionSet`.
- **0.48.**  *Behaviour:* `solve` returns the complex roots of an unknown with no assumptions
  (`eˣ + 1 = 0` → `[iπ]`; it said "no real solutions").  New: formal power series of `√x·eˣ`-type
  functions, infinite sums that are polylogarithms, `compile()` of `Si`, `Ci`, `Ei`, `li`, `Shi`,
  `Chi`, `erfi` and the Fresnel integrals.
- **0.47.**  *Behaviour:* decimal exponents in literals — `1e-05`, `2.5E3` parse as the numbers
  they denote (`1e-05` was read as `1·e − 5`); numbers for divergent integrals are gone
  (`∫_{−∞}^{∞} |x|/((x² + 1)π) dx` was 225.93).
- **0.46.**  *Behaviour:* `eval`, `equals`, the `is_*` queries, `sign` and `solve` give the same
  answer about the same constant; mixtures with overlapping supports no longer count the overlap
  twice, and divergent expectations no longer come out finite.  Improper integrals of rational
  functions times `cos`, `sin` or `ln` by residues.
- **0.45.**  *Behaviour:* wrong roots removed — roots on the boundary of a declared domain (a root
  equal to 0 for a positive unknown), non-real members of real solution sets.  `evalf` evaluates
  the exponential integrals, incomplete gamma, Bessel, Airy and Fresnel functions at complex
  arguments with proven error bounds.
- **0.44.**  *Behaviour:* `integrate_numeric` no longer returns a value beyond its tolerance with a
  tiny error estimate (`∫₀¹ dx/√(1 − x)` was 1.99999999); `(−1)^q` and `i^q` reduce their exponent
  when built.
- **0.43.**  *Behaviour:* solvers answer over the declared domain of the unknown: for a positive
  `x`, `x² − 4 = 0` gave `[2, −2]`; for an integer `m`, `2m − 3 = 0` gave `[3/2]`.  Trigonometric
  values at rational multiples of π are radicals (`cos(π/5) = (1 + √5)/4`).  Definite integrals
  evaluate to any number of digits by tanh-sinh quadrature.
- **0.42.**  *Behaviour:* limits of complex-valued functions keep finite imaginary parts
  (`limit(ln(−x), x, oo)` was `oo`, it is `oo + iπ`); functions and powers whose value follows from
  the assumptions on their symbols fold when built (`sin(πn)` for an integer `n`), so such a `0/0`
  no longer comes out `0`.
- **0.41.**  *Behaviour:* `Ex::diff` with respect to something other than a symbol no longer
  returns `0`; `oo + I` is no longer `oo`; `brent_root` and `bisect` refuse a sign change through a
  pole.
- **0.40.**  *Behaviour:* `solve_as_set` returns the whole solution set, periodic families as image
  sets (`sin x = 0` was `{0, π}`, now `{nπ : n ∈ ℤ}`); moments and generating functions of
  distributions are the true values (`Exponential(1).mgf(2)` was `−1`); number theory refuses
  instead of hanging or aborting.  The public `ExprTree` enum gained a variant.

## 0.30 – 0.39

- **0.39.**  *Behaviour:* `solve` no longer claims "no solution" where another branch solves the
  equation (`tan(x)/x`) and no longer returns poles of rational equations; `solve_as_set` returns a
  `ConditionSet` instead of an empty set when it merely found nothing.
- **0.38.**  *Behaviour:* antiderivatives are continuous where the integrand is: answers through
  `tan(x/2)` and `tan x` carry a `floor` term (`∫ dx/(2 + cos x)` dropped by `2π/√3` at every odd
  multiple of π); `ldl` and `cholesky` refuse a matrix whose symmetry cannot be decided.
- **0.37.**  *Behaviour:* `binomial`, `rising_factorial`, `falling_factorial` and `factorial2` fold
  at negative and rational arguments as SymPy 1.14 does (`binomial(−1, 3) = −1`).  The integrator
  finds 65% more antiderivatives of the Rubi suite.
- **0.36.**  *Behaviour:* `solve` no longer returns part of the roots of a product; `cholesky` and
  `ldl` accept Hermitian matrices.
- **0.35.**  *Behaviour:* a value that cancelled to zero at the working precision is no longer
  printed `0` when a term that is not zero was lost in the cancellation
  (`sin²1 + cos²1 − 1 + e^(−10¹⁰)`).
- **0.34.**  *Behaviour:* numbers too small for the exponent range are no longer printed as `0`
  when they are differences of such numbers; more expressions that are undefined everywhere stay
  undefined.
- **0.33.**  *Behaviour:* `Matrix::norm()` sums `|aᵢⱼ|²` (right for complex entries); expressions
  that are undefined everywhere stay undefined under `expand`, `simplify` and canonicalisation.
- **0.32.**  *Behaviour:* `eval_decimal` refuses a nonzero value below the exponent range instead of
  printing `0`; `qr` and `gram_schmidt` use the Hermitian inner product.
- **0.31.**  *Behaviour:* in integration the variable is real and a parameter is what its
  assumptions say, so an unassumed one may be complex: `∫ dx/(a·x + 1)` is `ln(a·x + 1)/a` (it was
  `ln|a·x + 1|/a`); `solve` rejects any candidate with a certified nonzero residual.
- **0.30.**  *Breaking:* `ode::OdeResult` has a new public field `branches`.  *Behaviour:* compiled
  and emitted code have a single rule — evalf's value, or `NaN` where that value is not real
  (`x^(p/q)` of a negative `x` is `NaN`; the real root is `real_root`); `series` is the real
  asymptotic expansion; singular constants fold when they are built.

## 0.23 – 0.29

- **0.29.**  *Breaking:* the `Matrix` and `ExactMatrix` constructors (`zeros`, `identity`,
  `from_fn`, …) return `Result` where they panicked; `Context::symbol_with`, `Ex::assume` and
  `Ex::refine_with` return `Result`; `StateSpace`'s fields are private.
- **0.28.**  *Behaviour:* far tails of the symbolic distributions keep their digits (a tail built as
  `1 − F` cancelled to `0`).  No signature changed.
- **0.27.**  *Behaviour:* `bonferroni` and `holm` decide rejection by `p ≤ α/m`, as statsmodels
  does; the first differential tests of `stats::hypothesis` and `numdist` against scipy, statsmodels
  and mpmath.
- **0.26.**  *Behaviour:* `eval_decimal`, `eval_f64` and `eval_complex64` return
  `PrecisionExhausted` when the digits cannot be certified; such a value used to be noise.  Every
  sub-expression carries an error bound.
- **0.25.**  *Behaviour:* `expand` leaves an expansion of more than 1,000,000 terms unexpanded; the
  logarithmic part of a rational integral is the full Lazard–Rioboo–Trager algorithm, so no
  integrand of the Rubi suite exhausts memory.
- **0.24.**  `rubi-harness/` runs `integrate` on the 72,254 integrands of the Rubi suite and judges
  every answer by differentiation; the 76 wrong answers it found in 0.23 are fixed.
- **0.23.**  *Behaviour — one domain model:* a symbol without assumptions may be complex and every
  function takes its principal branch, now everywhere: `expand_log` and `log_combine` apply only
  identities that hold, `simplify` no longer combines `ln x + ln y` into `ln(xy)` without
  assumptions, and `eval` takes the principal root of a negative number.  See
  [The Domain Model](./getting-started/key-concepts.md#the-domain-model).

## 0.14 – 0.22

- **0.22** (0.22.0 – 0.22.3).  *Breaking:* the 0.18 transitional re-exports in `stats` are removed;
  use the names' homes.  The Rust blocks of `README.md` and of this book compile and run as
  doctests (0.22.0); five cargo-fuzz targets check properties nightly (0.22.3).
- **0.21.**  *Breaking:* `MultiPoly::mul`/`pow` assert on `u32` exponent overflow.  Duplicated
  kernels are merged into shared ones, with measured speed-ups from 6× to 1 000×.
- **0.20.**  Multinomial and ordinal logistic regression, Bayesian rater models (MAP Dawid–Skene,
  MACE).
- **0.19.**  *Breaking:* `Sprt` decisions are sticky.  New: Cox proportional hazards (`stats::cox`).
- **0.18.**  The statistics module gets one rule for what lives where; every moved item keeps a
  re-export at its old path for this release (removed in 0.22).
- **0.17** (0.17.0 – 0.17.2).  *Breaking:* complex `f64` values are `num_complex::Complex64`
  instead of `(re, im)` pairs.  New: factorial and repeated-measures ANOVA with post-hoc tests.
- **0.16.**  *Breaking:* matrix decompositions return structs that state their identity
  (`Qr { q, r }`, `Ldl { l, d }`), the extended-gcd routines share `ExtendedGcd { gcd, x, y }`,
  `solve_ode_ivp` takes `InitialCondition { order, x, value }`.
- **0.15.**  *Breaking:* tuples whose positions share a type are replaced on the public surface by
  structs with named fields: `Interval<T>` with its kind (`[a, b]`, `(a, b)`, …), `Bounds<Q>` for
  LP variable bounds.
- **0.14.**  Additive over 0.13: reliability and item analysis, exact least squares and logistic
  regression, SPRT, information measures, survival analysis, the multivariate normal, PCA and order
  statistics (see [What's New in 0.13 / 0.14](./whats-new-0.13.md)).
