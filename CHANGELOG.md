# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor releases may contain breaking changes; they are listed first.

## [0.28.0] - 2026-09-24

Far tails of the symbolic distributions keep their digits.  0.27.0 listed
this as a known issue: a tail built as `1 − F`, or from the classic
closed forms, cancels in a far tail and evaluates to `0`.  No signature
changed; `Family` gains two provided methods.

### Breaking (behaviour; no signature changed)

- **`eval` no longer folds `lowergamma(s, x)` into `Γ(s) − Γ(s, x)` where
  that difference cancels.** For an integer or half-integer `s ≤ 64` and a
  rational `x` well below `s`, the closed form would lose more than 64
  bits: `lowergamma(5, 10⁻³⁰)` was an exact expression that evaluates to
  `0`, and now stays `lowergamma(5, 1/10^30)`, which `evalf` computes by
  its power series (`2e-151`).  Everywhere else the closed form stays, as
  SymPy writes it: `lowergamma(3, 1)` is still `2 − 5e⁻¹`.
- **A far tail of a symbolic distribution is written in a
  non-cancelling form**, so its exact expression changes.  This applies
  where the classic CDF puts a numeric point below `2⁻³²` or above
  `1 − 2⁻³²`.  For example, `P(N > 20)` is `½ erfc(10√2)`, not
  `½ − ½ erf(10√2)`.  Near the median the classic forms stay.

### Added

- `Distribution::sf` and `RandomVariable::sf`: the survival function
  `P(X > x)` on the whole line (scipy's `sf`).
- `Family::sf` and `Family::cdf_lower`: the non-cancelling survival
  function and lower-tail CDF a family may supply.  Both are provided
  methods, so existing families compile unchanged.  Every built-in
  continuous family has an `sf`; Normal, LogNormal, Cauchy, Exponential
  and Weibull also have a `cdf_lower`.  Of the discrete families, Poisson,
  Geometric and NegativeBinomial have an `sf`.

### Fixed

- **The symbolic distributions' far tails keep their digits.** Each of
  the following evaluated to `0` (or was an error) in 0.27.0, and now
  matches mpmath to 1e-14 or better:
  - standard normal: `P(X > 20) = 2.75e-89`;
  - `Normal(2, 3).cdf(−100) = 1.11e-253`;
  - log-normal: `P(X > 10²⁰) = 2.63e-463`;
  - Cauchy: `P(X > 10¹⁰⁰) = 3.18e-101`;
  - `Gamma(⅓)`: `P(X > 800)`;
  - `Gamma(5).cdf(10⁻³⁰) = 8.33e-153`;
  - `Exponential(1).cdf(10⁻¹⁰⁰)`;
  - Poisson(1): `P(X > 60) = 7.37e-85`;
  - NegativeBinomial(3, ½): `P(X > 2000)`, which was an unevaluable
    infinite sum;
  - the same tails through a truncation, a decreasing affine map, a
    mixture and an order statistic.

  Changes that make this work:
  - `probability_of` measures a region that reaches the support's upper
    end as `P(X ≥ lo)`, and an interval inside the far upper tail as
    `S(lo) − S(hi)`.
  - The wrappers transport both forms.

### Measured

- `tests/v27/`: `v27_tails` (8 tests).
- Rubi harness: unchanged (6,032 verified, 0 wrong).  Self-test: 55,044
  verified, 0 wrong, 1,173 undecided.
- Byte identity of the LP paths and the s42 Lean certificates: identical.

## [0.27.0] - 2026-09-24

The first round of a bug hunt in `symplex::stats`.  `hypothesis` was
differentially tested against scipy 1.18.1 and statsmodels 0.15.0 (about
19,700 value checks on random datasets, and 2,625 degenerate cases), and
`numdist` (the `f64`
reference distributions) against mpmath at 50 digits, far tails and extreme
shapes included.  Every bug found is pinned in `tests/v26/` against the
oracle call that produced the reference value.  No signature changed.

### Breaking (behaviour; no signature changed)

- **`mcnemar_test` with the continuity correction at `b = c`** gives
  statistic `0` and `p = 1`.  It used to give `1/(b + c)`: `b = c = 5` gave
  `χ² = 1/10`, `p = 0.7518`, more significant than the exact test's
  `p = 1`.  The correction now moves `|b − c|` towards 0 and stops there,
  as R's `mcnemar.test` and scipy's Yates correction do.  statsmodels
  squares the `−1`, so symplex now differs from it here (documented).
- **`bonferroni` and `holm` decide rejection by `p ≤ α/m`**, as
  statsmodels does, instead of `m·p ≤ α`.  The two disagree in the last
  bit: `p = 0.05/11` with `m = 11` is now rejected at `α = 0.05`.
- **`sample_size_for_proportion` returns `ComputationFailed`** when the
  size doesn't fit in `usize`.  A margin of `1e-200` used to return
  `18446744073709551615`.
- **Counts that overflow `usize` are an `InvalidArgument`** in
  `mcnemar_test` (`b + c`) and `relative_risk` (a row total), instead of an
  overflow.  `z_test_two_proportions` and `phi_coefficient` compute their
  sums without overflowing.
- **`cohens_h` returns `2·atan((p₁ − p₂)/(√(p₁q₁) + √(p₂q₂)))`**, one
  arctangent, instead of the difference of two arcsines (same value, no
  cancellation; see Fixed).
- **`numdist`:**
  - A quantile beyond the positive floats is `5e-324`, `+∞` or `1`.
    `f::isf(0.3, 1e-300, 1e-100)` and `beta::isf(0.1, 2, 1e20)` used to
    return `Ok(NaN)`.
  - Beta quantiles return `InvalidArgument`, and the beta tails `NaN`,
    when `α + β` overflows a double.
  - Many quantiles change in their last ulps: each is now the float where
    the level is crossed.

### Fixed

- **`hypothesis`:**
  - `p_value_ln` and `p_value_log10` are finite for every χ² tail.
    Goodness of fit on `[301, 1, 0]` gave `PrecisionExhausted`; it now
    gives `ln p = −299.00993377483445` (mpmath
    `log(gammainc(1, Fraction(90301, 151)/2, inf, regularized=True))`).
    This helper also serves `anova` and `regression`.
  - The exact binomial, sign and McNemar tests stream integer weights over
    one denominator instead of adding reduced rationals:
    `binomial_test(0, 1500, ½)` took 2.1 s and now takes 1.1 ms (debug
    build), with the same exact p-value.
  - `p_value_f64`, `p_value_ln` and `p_value_decimal` of an exact rational
    no longer go through the evaluator, which is slow on the
    `10⁵`-bit rationals of the large exact tests.  `p_value_decimal`
    rounds exactly, ties to even, in `eval_decimal`'s format.
  - `cohens_h(½, ½ + 10⁻¹⁰⁰)` evaluated to `0`, and
    `cohens_h(1 − 10⁻³⁰, 1)` to `PrecisionExhausted`.  They now give
    `−2e-100` and `−2e-15`.  `power_two_proportions` uses the same form.
  - `power_t_test_two_sample`:
    - it no longer overflows `2n − 2`;
    - its χ² density no longer cancels terms of size `ν ln ν`: at
      `d = 0` and `n = 10⁹` the power was `α − 1.65e-6` (now within
      `1e-13` of `α`).
  - `relative_risk` forms its variance exactly, as `b/(a n₁) + d/(c n₂)`.
  - The docs spell out the remaining deliberate differences from scipy:
    - the two-sided binomial p compares the pmf exactly (scipy allows a
      `1e-7` slack: `binomtest(198, 950, 0.25)` is `0.00305` in scipy and
      `0.0027180105491057661` exactly);
    - `kendall_test` with `n = 2` is asymptotic (scipy raises
      `ZeroDivisionError`);
    - scipy's `method='auto'` rules are listed on `RankMethod`.
- **`numdist`:**
  - Every tail carries its logarithm (a mantissa and a separate exponent),
    so quantiles at subnormal levels are right.
    `t::ppf(4.4e-323, 111961533.34)` was `−38.410885323407236` (1.2e-5
    off); it is now `−38.41040928038728`.
  - The quantile solver returns its best evaluated point plus a Newton
    correction, snapped to the floats where the level is crossed.
  - Discrete quantiles compare the smaller tail:
    - `poisson::isf(1 − 2⁻⁵², 1229036.68)` was `1220000`; it is now
      `1220039`;
    - binomial quantiles near 1 and at subnormal levels are fixed.
  - The incomplete gamma for a shape below 1 uses the Taylor route of
    DiDonato & Morris (1986).  `gammaincc(1e-300, 0.9)` was
    `−4.64e-14`; it is now `2.6018393932599960e-301`.
  - The incomplete beta with a tiny second shape (`bgrat`) and with two
    tiny shapes is fixed.
  - F tails beyond the normal range of the beta arguments are fixed.
  - `gamma::ppf` for shapes of `1e10` and above uses Cornish–Fisher plus
    Newton: `gamma::ppf(0.3, 1e30)` was `7.389e30` and is now
    `9.999999999999995e29`, and `gamma::ppf(0.5, 1e300)` converges.
  - `gamma::sf(f64::MAX, 1e300)` was `2.2e-159`; it is now `0`.
  - `t::ppf(1e-300, 1e100)` converges.
  - `fuzz_numdist` covers wider shape ranges and checks the discrete
    `isf`.

### Known issues (reproduced; next round)

- The symbolic distributions build tails as `½ + ½ erf(…)` or
  `1 − Σ pmf`, which cancel in a far tail, so the result evaluates to `0`:
  - `Distribution::normal(2, 3).cdf(−100)` gives `0.0` (Φ(−34) is
    1.11e-253);
  - the standard normal `P(X > 20)`, the log-normal `P(X > 10²⁰)` and
    Poisson(1) `P(X > 60)` do the same.
- `Distribution::quantile_f64` for Binomial and Poisson compares against
  `p − 10⁻¹²`:
  - `poisson(10⁶).quantile_f64(10⁻¹³)` gives `962716`; the answer is
    `992660`;
  - `poisson(7/3)` at `1 − 2⁻⁵³` gives `20`; the answer is `24`.
- The evaluator's error bound for `ln(exp(x))` is too pessimistic below
  `x ≈ −250`: `exp(−2601/10).ln().eval_f64()` is `PrecisionExhausted`.
  `stats` works around it.

### Measured

- `tests/v26/`: `v26_hypothesis`, `v26_numdist` (33 tests).

## [0.26.0] - 2026-09-23

Numerical evaluation knows how many of its digits are right.  Until now an
expression was evaluated once, bottom-up, at a fixed working precision,
and the result was returned whatever its accuracy: catastrophic
cancellation silently lost digits, and a quotient by a difference that
cancels to zero came out as noise presented as a number (the two false
"wrong" verdicts of the Rubi harness self-test).  Every sub-expression now
carries an error bound, propagated to first order (`src/transforms/evalf/
accuracy.rs`); when the bound does not cover the requested digits the
expression is re-evaluated at a higher precision, as SymPy's `evalf` does
for a cancelling sum, and otherwise the evaluation is refused.

### Breaking (behaviour; no signature changed)

- **`eval_decimal`, `eval_f64` and `eval_complex64` return
  `PrecisionExhausted` when the digits cannot be certified** within twice
  the initial working precision plus 256 bits: `1/(sin²1 + cos²1 − 1)`,
  `sign(sin²1 + cos²1 − 1)`, Rubi's `∫ cot(x)/ln(e^sin x)` answer at
  `x = 13/4` (which evaluated to `−64` at every precision).  Such a value
  used to be noise.
- **Digits lost to cancellation are recovered:** `exp(10⁻³⁰) − 1` is
  `1e-30` (was `0`), `sqrt(10⁴⁰ + 1) − 10²⁰` is `5e-21`.  A value that is
  zero to the precision reached is `0` (was a rounding residue such as
  `1e-40`).
- **`n^(a/b)` with `a > b` splits off its integer part**, as SymPy writes
  it: `2^(3/2)` is `2*sqrt(2)`, `15^(3/2)` is `15*sqrt(15)`, `(29/15)^(-3/2)`
  is `15/841*sqrt(435)`, `2^(5/3)` is `2*2^(2/3)`.  Before, `15^(3/2)` and
  `15*sqrt(15)` were two canonical forms of one number and did not
  cancel; 0.25's flattening of `(√b)ⁿ` produced more of the first kind,
  and 161 of Rubi's own answers lost their numeric value in the self-test
  (54,993 verified at 0.24, 54,832 at 0.25) because `polylog` terms that
  should cancel no longer did.

### Fixed

- LaTeX and the pretty printer render `b^(-1/2)` as `\frac{1}{\sqrt{b}}`
  and `1/√(b)` again: 0.25 flattened `(√b)^{-1}` to `b^(-1/2)` and they
  showed `(x + 1)^{1/2}` in the denominator.  The pretty printer also
  showed a denominator of several factors with their negative exponents
  (`x⁻²·y` for `x²·y`) and without parentheses around sums.

### Measured

- Rubi harness: unchanged (6,032 verified, 0 wrong).  Self-test on Rubi's
  own answers: 55,044 verified (0.25: 54,832; 0.24: 54,993), **0 wrong**
  (0.24 and 0.25: 2 false alarms), 1,173 undecided (0.25: 1,383).
- `tests/v25/`: `v25_evalf`, `v25_radicals` (6 tests).

## [0.25.0] - 2026-09-23

No integrand can exhaust memory.  The three Rubi "timeouts" of 0.24 were
`∫ x³/(x⁸+1)`-shaped integrands that passed 8 GB within seconds.  The root
cause was in the rational integrator: the Rothstein–Trager resultant of
`x³/(x⁸+1)` is `(64t²+1)⁴`, and a residue of multiplicity `i` needs a log
argument of degree `i`, but only the degree-1 member of the remainder
sequence was ever used.  The integrand therefore fell through to partial
fractions over the eight complex roots, whose common-denominator expansion
headed for ~10⁸ terms.  The logarithmic part is now the full
Lazard–Rioboo–Trager algorithm, quadratic factors get exact real forms,
and budgets on `expand`, on the integrator's arena growth and on dense
polynomial degree bound what is left.  On the Rubi suite: **6,032 verified**
(0.24: 5,524), 77 real_verified (73), **0 wrong, 0 panics, 0 timeouts, 0
undecided** (3 timeouts and 2 undecided in 0.24), 66,012 unevaluated.
Each change is pinned in `tests/v24/`.

### Breaking (behaviour; no signature changed)

- **`(a^b)^c = a^(b·c)` for every base when `c` is an integer and `b` a
  rational.**  `z^c` is single-valued for integer `c`, so the rule holds on
  the principal branch for all complex `a` (SymPy's `Pow._eval_power`
  does the same).  Before, only integer `b` or a positive rational `a`
  flattened, so `(√y)²` stayed `(√y)²` instead of `y`, and
  `((x²+1)^{1/2})^{-1}` stayed nested instead of `(x²+1)^{-1/2}`.
  Structural results change accordingly: `∫ atan(√x − x⁹) dx` now
  begins `x·atan(…)` instead of `(√x)²·atan(…)`.
- **Display: `b^(-1/2)` prints as `1/sqrt(b)`** (`a/sqrt(x + 1)`,
  `1/(sqrt(x + 1)*sqrt(y + 1))`).  That was the printed form of the old
  nested `(√b)^{-1}`; it is kept for the flattened power.  A power that
  prints as a quotient is now parenthesised where it binds: `x^(1/y)`
  printed as `x^1/y`, which re-parses as `x/y`.
- **`expand` leaves an expansion of more than 1,000,000 terms unexpanded**
  ([`EXPAND_TERM_LIMIT`](https://docs.rs/symplex/0.25.0/symplex/macros/constant.EXPAND_TERM_LIMIT.html)):
  a product of 21 binomials, or `(a + b + c + d + 1)^100`, stays as it is
  (it is still equal to its expansion) instead of exhausting memory.
- **Rational-function antiderivatives take the Lazard–Rioboo–Trager
  forms.**  `∫ x/(x⁴ + 1) dx` is `atan(x²)/2` (was a sum of two linear
  `atan`s), `∫ 1/(x² − 2) dx` is `√2/4·(ln|x − √2| − ln|x + √2|)` (was a
  `RootSum`), `∫ x/(x⁶ + 1) dx` has `atan(√3/3·(2x² − 1))`, and
  integrands `x^(k−1)·F(x^k)` are integrated through `u = x^k`.  The
  answers agree with SymPy's where it has one.
- `integrate` evaluates the integrand first (`ln 1`, `atan 0` are exact
  zeros): `∫ ln(x)/(x² + ln 1) dx` was built on `atan(x/√(ln 1))`, whose
  derivative is 0 (found by `fuzz_integrate` during this release).
- The conversion to a dense `ℚ`-polynomial behind `collect`, `cancel` and
  the rational integrators refuses a power of degree above 10,000
  (`x^1000000000` asked for a billion coefficients); `∫ 1/(x^1000000000 +
  1) dx` is unevaluated at once and `collect` leaves such a power alone.

### Fixed — memory and time

- `∫ x³/(x⁸+1)`, `x⁴/(16+x¹⁰)`, `x⁵/(9+x¹²)`, `x⁷/(x¹⁶+1)`: 5.8–10 GB
  resident when killed at 6 s → `atan(x⁴)/4`, `atan(x⁵/4)/20`,
  `atan(x⁶/3)/18`, `atan(x⁸)/8` in about 1 ms, as SymPy.
- One top-level `integrate` may create at most 2,000,000 arena nodes;
  past that the remaining sub-integrals stay unevaluated.  The rational
  integrator refuses anything that is not in `ℚ(x)` before expanding it.
- The integrator remembers the outcome of each sub-integrand for the rest
  of the call: `∫ x^(3/2)/(sin(−1) − 2x) dx` made 918 u-substitution
  attempts on 53 distinct integrands (1.3 s, 850,000 nodes) to give up;
  it now has a closed form, as does `∫ atan(√x·cos 4) dx` (0.8 s → a few
  ms; both SymPy's): polynomials over quadratics with transcendental
  coefficients are divided out over the field of constants.

### Fixed — closed forms that were missing

- A real irrational root of the resultant (`∫ 1/(x³ − 2) dx`, root `∛2/6`)
  lost its `α·ln|x − root|` term, so the derivative check rejected every
  such answer: `1/(x³ ± 2)`, `x/(x³ − 2)`, `1/(x⁵ − 2)`, `1/(x⁶ − 2)` now
  integrate.
- Quadratic factors of any multiplicity (`x/(x⁸+1)`, `x/(x⁶+1)`,
  `x/(x⁸+x⁴+1)`, `x³/(x¹²+1)`, `x²/(x⁹+1)`, `(x⁴+1)/(x⁶+1)`) were
  unevaluated or `RootSum`s.
- `∫ sech(6x)·sinh(x) dx` and `∫ cosh(x)·sech(6x) dx` (the two Rubi
  "undecided" cases, whose derivative evaluated to NaN) are verified.

### Fixed — panics reachable from user input

- `symplex-wasm` passed JavaScript strings to `Context::symbol`, which
  panics on an empty name; every entry point now returns an error, as does
  an empty DH configuration.
- `RandomVariable::try_new` panicked on an empty name; it returns
  `InvalidArgument`.
- `Ex::refine_with` left the temporary assumptions of earlier symbols in
  place when a later one was contradictory (the panic corrupted the
  `Context`); the stored assumptions are now restored before the panic
  continues.
- Three `_ => unreachable!()` arms (the Weierstrass substitution, the
  Fourier transform of `sin`/`cos` products, `i^n` in `canon`) are gone;
  the no-panics ratchet now counts `panic!`/`unreachable!` anywhere on a
  line (it only saw them at the start), with the matrix and polytope
  operators that panic on a shape mismatch allowlisted and documented.

### Added

- `Context::try_symbol` (an error, not a panic, for an empty name) and
  `symplex::macros::EXPAND_TERM_LIMIT`.
- Stage tracing for debugging (`src/base/stage.rs`, CONTRIBUTING.md →
  "Finding where a slow or memory-hungry call spends its time"): the
  expensive steps of the integrator and the Risch rational path are
  `DEBUG` spans that log their time and arena growth under the
  `symplex::stage` target (`RUST_LOG=symplex::stage=debug` lists the hot
  ones).  Nothing is computed unless a subscriber enables them.
  `rubi-harness --probe` logs under `RUST_LOG`.
- `tests/v24/`: `v24_canon`, `v24_integrate`, `v24_panics` (11 tests).

## [0.24.0] - 2026-09-23

The Rubi release.  `rubi-harness/` runs `integrate` on the Rubi integration
test suite (72,254 integrands, MIT, vendored) and judges every answer by
differentiation at five real points (both signs) with fixed rational
parameters.  Against 0.23.0 it found 76 answers whose derivative is not the
integrand; all are fixed, and the suite now reports **0 wrong, 0 panics**:
5,524 answers verified, 73 more verified wherever the integrand is real (the
real-variable convention: `ln|u|` where the integrand is complex), 66,519
unevaluated, 3 timeouts.  A ratchet (`rubi-harness/ratchet.tsv`, `--check`)
keeps each file's counts from regressing.  Each fix is pinned in
`tests/v23/`.

### Breaking (behaviour; no signature changed)

- `d/dx acosh(u)` is `u′/(√(u−1)·√(u+1))` (SymPy's form), the derivative
  of the principal `acosh` everywhere; `u′/√(u²−1)` had the wrong sign for
  `u < −1`.
- Antiderivatives of `1/√(x² − a²)`, `√(x² − a²)` and
  `1/√(a x² + b x + c)` with a gap in the real domain are written with
  `ln|x + √(x² − a²)|` instead of `acosh(x/a)` (which is right only on
  `x > a`); `∫ acosh(g) dx` uses `√(g−1)·√(g+1)`.
- `eval` folds `exp(ln w) = w` and `exp(w + i·k·π)` for half-integer `k`
  (`exp(ln 2 − 3πi) = −2`); the `exp_ln` pattern rule therefore no longer
  appears in `simplify_traced` steps.

### Fixed — wrong antiderivatives (Rubi suite)

- `∫ (a·x + b)/(x² + 1) dx` was `0` (and 55 more of the linear-over-
  quadratic shape): a coefficient containing a parameter has no numeric
  value, and was dropped as if it were zero.
- `∫ dx/√((3x − 2)²)` was `ln|x − 2/3|/3`: `√(a·u²)` is `√a·|u|`, so the
  antiderivative needs `sign(u)`.
- `∫ cosh x/(i + sinh x) dx` was `ln|i + sinh x|`, whose derivative is the
  conjugate: `ln|u|` is the real-variable antiderivative only for real `u`;
  a `u` containing `i` now gives `ln u`.
- `acosh`-based answers (above) were wrong for `x < −a`.

### Fixed — found by `fuzz_integrate` / `fuzz_simplify`

- `∫ |√x|² dx` was `x²/2`: `|g|ⁿ → gⁿ` for even `n` needs `g` real on the
  real line.
- `√((e^{ln(−2)})^{−3})` evaluated to `−(√2/4)i` at 30 digits (`+` at 50):
  the exponential of `−3 ln 2 − 3πi` carried a rounding residue that chose
  the branch (fixed by the `eval` folds above).

### Added

- `rubi-harness/` (own workspace, not published): Maxima-syntax reader and
  translator, per-integrand time and memory limits in worker processes, the
  verdicts `verified` / `real_verified` / `wrong` / `unevaluated` /
  `undecided` / `timeout` / `panic` / `unsupported`, per-file summary, a
  `--selftest` that judges Rubi's own answers (54,993 verified, 2 false
  alarms, both Rubi answers dividing by `x − log(e^x) ≡ 0`), and the ratchet.
- `tests/v23/`: `v23_eval`, `v23_integrate` (9 tests).

## [0.23.0] - 2026-09-23

One domain model.  The documentation has always promised a CAS over ℂ — a
symbol without assumptions may be complex, every function takes its
principal branch — and the code now keeps it everywhere: in the exact
evaluator, in numerical evaluation and in every rewrite.  `fuzz_simplify`
now compares *complex* values at real and complex points (it used to skip
every point where either side was not real) and found each of the defects
below; each is pinned by a test in `tests/v22/` quoting the point where the
old answer was wrong and the mpmath / SymPy value.  The new model is
documented in one place, [Key Concepts → The Domain
Model](book/src/getting-started/key-concepts.md#the-domain-model).

### Breaking (behaviour; no signature changed)

- **`expand_log` and `log_combine` apply only identities that hold** (the
  forced forms were the default).  `expand_log` splits off factors known
  positive (`ln(2x) = ln 2 + ln x`), a known-negative factor as `ln(−f)`,
  and `ln(a^c) = c·ln a` for `a > 0` with real `c`, or `−1 < c ≤ 1`
  (`ln √x = ½ ln x` for every `x`); `ln(xy)`, `ln(x²)`, `ln(1/x)` stay.
  `log_combine` joins the logarithms of known-positive arguments with each
  other *and with at most one other logarithm* (`ln 2 + ln x = ln(2x)` for
  every complex `x`, since `arg 2 = 0` — SymPy's `logcombine` refuses it),
  and `c·ln a → ln(a^c)` under the same condition.  The old behaviour is
  `expand_log_with(true)` / `log_combine_with(true)`.  Symbols that are
  positive should be declared so (`Assumption::Positive`).
- **`simplify` no longer combines `ln x + ln y` into `ln(xy)`** for
  arguments of unknown sign (strategy 6 used the forced `log_combine`):
  at `x = y = −1` the two differ by `2πi`, and even at real negative
  points `−ln x → ln(1/x)` was off by `2πi`.
- **Real-analysis identities need a known-real argument.**
  `ln(e^w) → w` and `√(w²) → |w|` (the simplify rules and
  `powdenest(false)`) fired for every argument not known *non*-real;
  `asinh(sinh w) → w`, `atanh(tanh w) → w` and `acosh(cosh w) → |w|` had
  no condition at all.  All five now require `w` known real (for real
  arguments they still fire — SymPy 1.14 leaves the three hyperbolic ones
  alone even then).  `√(i²) = i`, `ln(e^{1/3+4i}) = 1/3 + (4 − 2π)i`,
  `acosh(cosh(1 + 2i)) = 1 + 2i`.
- **`eval` takes the principal root of a negative number**, as `evalf`
  always did: `(−8)^(1/3)` is `2·(−1)^(1/3) = 1 + √3·i` (SymPy's
  `2*(-1)**(1/3)`), not `−2`.  Before, the two evaluators disagreed, so
  `simplify(∛x²) = x^(2/3)` looked wrong at `x = −2`.  The real root is the
  new [`Ex::real_root`].  `compile()` and the Rust / C / Python / NumPy /
  Julia back ends are unchanged: generated `f64` code takes the real odd
  root (the documented exception since 0.11.1 — an `f64` cannot hold the
  principal value).
- **`eval` merges `(e^f)^g` into `e^(f·g)` only on the principal branch**:
  for integer `g`, or `f` known real (`√(e^{4i}) = −e^{2i}`, not `e^{2i}`).
  This also changes `eval_decimal`/`eval_f64` of such expressions, which
  run `eval` first and so returned the wrong branch.  Limits are
  unaffected: the limit variable is now a positive dummy inside Gruntz, as
  in SymPy.
- **`evalf` (`eval_f64`, `eval_complex64`, `eval_decimal`) returns
  `PrecisionExhausted` for `sin`/`cos`/`tan` of an inexact argument above
  `2⁶⁴`** (e.g. `sin(exp(100))`), whose reduction modulo 2π the working
  precision cannot determine — it used to print digits the argument did not
  determine.  Exact rational arguments are reduced exactly at any size up to
  `EvalConfig::max_evalf_precision` bits (`sin(10¹⁰⁰)` is right); beyond
  that, or for a huge imaginary part of `exp`/`sinh`/`cosh`/`tanh`, the
  same error instead of a minutes-long computation.

### Fixed — the evaluator (found by `fuzz_simplify`: the oracle itself)

- The sign of a zero imaginary part chose the side of a branch cut:
  `c_neg` turns `+0` into `−0`, so `ln(−sin(1/3))` evaluated to `… − πi`
  and `√(−tan(1/3))`, `(−sin(1/3))^(−3/2)` to the conjugate of the
  principal value.  A negative real now has argument `+π` however it was
  computed; `(−r)^(n/2)` is exactly `r^(n/2)·iⁿ`; and `sin`/`cos` of the
  working-precision constants `0, ±π/2, ±π` are exact, so
  `e^{ln(−5/7)}` has no `1e-50·i` residue to push the next function onto
  the wrong side.
- `acosh` was the other branch for `Re z < 0` (`acosh(−2 + i/2)` came out
  negated); `atanh` and `atan` were wrong on their cuts (`atanh(2)`,
  `atan(2i)`); now `ln(z + √(z+1)·√(z−1))`, `½(ln(1+z) − ln(1−z))`,
  `(i/2)(ln(1−iz) − ln(1+iz))`, which agree with mpmath across the plane.
  `asin` and `asinh` (and so `acos`) of large `|z|` in half the plane had no
  correct digit — the two terms of the logarithm's argument cancelled:
  `acos(cosh(x^(9/2)))` at `x = 1/3 + 4i` was `2.64 + 346.7i`, mpmath
  `0.548 − 476.7i`.  The sum is now formed without cancellation.
- Real arguments outside the real domain gave NaN (an error): `asin(2)`,
  `acos(−2)`, `acosh(1/3)`, `acosh(−2)`, `atanh(2)` now evaluate to their
  principal complex values.

### Fixed — rewrites

- `refine`'s `√(r²) → |r|` returned `|r|` for the *whole* expression:
  `simplify(cos(√(|x|²)))` was `|x|`, `refine(√(r²) + 1)` stayed
  `√(r²) + 1` (the rewrite of an inner node was dropped).
- The assumption system called `asin`, `acos` and `atanh` of any real
  real, and `acosh` of any positive real real and non-negative:
  `asin(4)`, `atanh(−2)`, `acosh(1/64)` are not, and `refine` dropped
  `|acosh(e⁻³)|`.  Now only on `[−1, 1]`, `(−1, 1)` and `[1, ∞)`.
- Integration flattened `(g^m)^n → g^(m·n)` whenever both exponents were
  rational, so `∫ √x·√(1/x) dx` was `x` (the integrand is `−1` for every
  `x < 0`), and `∫ cos(√x)/√(x⁻²) dx` lost `|x|` through `x = s²` (found by
  `fuzz_integrate`).  The flattening now keeps the value, and
  radical-substitution results are verified by differentiation like the
  other risky routes; these integrals are unevaluated.
- `powdenest(true)` denests the factors it distributes (`√(4x²) → 2x`
  in one pass), which is what keeps `matrix_exp_t([[0, −ω], [ω, 0]])` in
  `cos(ωt)` form without a realness assumption on `ω`.

### Added

- `Ex::real_root(n)` — SymPy's `real_root`: the real odd root of a real
  (`real_root(−8, 3) = −2`), the principal root for even `n`, and SymPy's
  `Piecewise` on `im(x) = 0` for a symbol not known real.
- `fuzz_simplify` compares complex values at real and complex points, and
  its grammar (`Grammar::Full`) adds `sinh`/`cosh`/`tanh`, their inverses,
  `asin`/`acos` and rational powers; points where either side is not
  continuous (on a branch cut) are skipped.
- `tests/v22/`: `v22_evalf`, `v22_simplify`, `v22_integrate` (34 tests).
  39 existing tests and 3 doctests pinned the old behaviour (a sum of logs
  of unassumed symbols combined, `√(x²) = |x|` and `ln(e^x) = x` for
  unassumed `x`, `∛(−8) = −2`, …); they now declare their symbols `Real`
  or `Positive`, or assert the value is kept.
- `THIRD-PARTY-NOTICES.md`: how SymPy, mpmath, SciPy and statsmodels are
  used (as oracles; a few modules follow SymPy's implementation of a
  published algorithm), with SymPy's BSD-3-Clause notice.  CONTRIBUTING
  gains the provenance policy (mathematics is free; restrictively licensed
  code is not read, and a clean-room description is used instead).

### Documentation

- The domain model — ℂ by default, principal branch, which identity needs
  which assumption, the `force` variants, the real-variable conventions of
  limits and integration, and the generated-code exception — in Key
  Concepts, README principle 7, the complex-analysis guide and the SymPy
  migration table (`expand_log(force=True)`, `logcombine(force=True)`,
  `cbrt`, `real_root`).  Stale lines fixed: `ExprNode` has 92 variants,
  "Patterns 2–5", `stats` is no longer listed as missing.

## [0.22.3] - 2026-09-22

The fuzzing release.  Five new cargo-fuzz targets check *properties* —
exact linear-algebra and polynomial identities, probability invariants of
the `f64` distributions, value preservation of the rewriting transforms,
`F′ = f` for every closed-form antiderivative — and run nightly in CI.
A few hours of local fuzzing found the defects below — wrong closed forms,
wrong quantiles, a stack overflow and two searches of tens of seconds;
each is fixed at its cause and pinned by a regression test in `tests/v21/`.  No
API change.

### Fixed — found by `fuzz_integrate`

- `∫ |√x| dx` was `⅔ x^(3/2)·sign(√x)` and `∫ |c/cos x| dx` was
  `|c|·ln|sec x + tan x|` — wrong wherever `√x` is imaginary or `cos x < 0`.
  The `|g|`/`sign(g)` route now requires `g` real *and continuous* (a pole
  changes sign without a root), and the `x = t²` substitution refuses
  integrands whose value depends on `t` being real.
- `∫ x^(−5/2)·atan(√x) dx` contained `−⅔ ln|√x|`, wrong for `x < 0`: after
  `x = t²` the analytic `ln t` is kept instead of the real-variable
  `ln|t|`.
- `∫ (x + ln x − ¾)·ln x dx` lost its `∫ x·ln x` part: the Risch tower's
  extraction of a polynomial in `θ = ln x` kept only the first coefficient
  when two terms had the same power.  Tower results are now also verified
  by differentiation, like the other risky routes.
- `∫ x²(x + 1/x) dx` stayed unevaluated (a reciprocal inside a sum was not
  recognised as a rational function) and `∫ x²(x − x⁻³ − ¼)·|x| dx`
  returned `nan` (the `|g|` route shifted by an antiderivative singular at
  the root).
- `∫ ln(√x + sin(−2)) dx` searched for 12.5 s and gave up: nothing handled
  `∫ N(x)/(a·x + b) dx` for a constant `b` that is not rational (`π`, `√2`,
  `sin 2`, a symbol).  New exact route; 2 ms, SymPy's answer.  A failing
  search is also capped at 20 000 `integrate_node` calls per `integrate`
  (successful integrations in the suite use ≤ ~600).
- `∫ atan(√x − x³) dx` took 36 s and returned a 32 KB sum; now 0.2 s and
  700 characters.  The Lazard–Rioboo–Trager step computed the whole
  Euclidean remainder sequence over ℚ(t) though it needs only the degree-1
  member (`GenPoly::euclidean_prs_until`), each remainder's coefficients
  grew in `t` (now made primitive over ℚ[t] at every step,
  `euclidean_prs_normalized`), and `RootSum` was expanded over `RootOf`
  placeholders (now kept compact).
- `diff` of a `RootOf` was the unevaluated `Derivative(RootOf(…), x)` —
  even when `x` is the polynomial's bound variable — so no antiderivative
  containing a root could be differentiated or evaluated.  A node in which
  the variable is not free differentiates to 0.

### Fixed — found by `fuzz_numdist`

- Quantiles whose true value is below the smallest positive float (tiny
  shapes put their mass within e⁻⁷⁰⁰ of 0): `beta`, `gamma`, `chi2` and `f`
  returned values whose cdf was far from the level (`beta::ppf(0.2055,
  10⁻³, 10⁻³) = 5.6e-309` with cdf 0.246, or 0); they return the smallest
  float with `F(x) ≥ p`, and `F` quantiles beyond `f64::MAX` are `+∞`.
- Subnormal quantiles: the beta logit `1/(1 + e^{−v})` stopped at 5.6e-309,
  the gamma solve scaled after exponentiating, and the F solve formed
  `d₁·x` — all now reach the subnormals (the solvers work in logarithms and
  use the tail's log-linear leading term below e⁻⁶⁹⁰).
- `gamma::cdf`/`chi2::cdf` at subnormal `x` were 0 (`bd0` overflowed `k/m`;
  `x/θ` underflowed); `f::cdf` at subnormal or huge `x` lost the tail
  (`d₁·x` under- or overflowed); the beta deep-tail Newton start underflowed
  to `ln 0 = −∞`.

### Fixed — found by `fuzz_parser`

- `7.4**77.4**74` overflowed the stack: the radical normal form rewrote
  `5^(−a/b)` as `5^(−(k+1))·5^((b−s)/b)` even when `5^(k+1)` (k ≈ 10¹⁴⁰)
  cannot fold to a number, and `mul` added the exponents back — an endless
  recursion.  The arena also hashes rationals iteratively now (`Ratio`'s
  `Hash` recurses once per continued-fraction partial quotient).

### Added

- `fuzz/`: `fuzz_numdist`, `fuzz_exact_matrix`, `fuzz_poly`, `fuzz_simplify`,
  `fuzz_integrate` (and `print_expr`, which decodes an input), a shared
  expression generator, and `fuzz/README.md` with the replay loop (build
  once, replay an artifact in ~1 s, 30 s bursts).
- `.github/workflows/fuzz.yml`: nightly (and on-demand) fuzzing, one job per
  target, ten minutes each, the corpus cached between nights, a failing
  input uploaded with its decoded expression.
- `GenPoly::euclidean_prs_until`, `GenPoly::euclidean_prs_normalized`.

## [0.22.2] - 2026-09-22

### Changed

- `Poly::extended_gcd` (and `Ex::poly_gcdex`, the partial-fraction and
  Hermite-reduction steps that call it) runs through the primitive PRS in
  `ℤ[x]` via a new `Field::poly_extended_gcd` hook, as `gcd` has since 0.21:
  remainders are made primitive and the cofactors kept over one integer
  denominator per step, instead of Euclid over ℚ paying a gcd per rational
  operation.  Degree 11 × 12 with 40-digit rational coefficients: 5.15 s →
  0.18 s (release).  The results are the same three polynomials as
  Euclid's (pinned in-crate against the Euclidean reference on 250 random
  and edge-case pairs, and against SymPy `gcdex` in
  `tests/v21/v21_poly_gcdex.rs`); the LP pivot-path baseline and the
  Mathlib-compiled Lean certificate are byte-identical.

## [0.22.1] - 2026-09-22

Fixes the seven known issues 0.22.0 shipped with reproducers for, and a
packaging gap.  No API change.

### Fixed

- `eval_f64` / `evalf` of a factorial with a non-integer argument: `x!` is
  now `Γ(x + 1)` (`(−1/2)! = √π`); it used to be `Unevaluable`.
- One-sided limits of `x!` at its poles: `limit_right(x!, x → −1) = ∞`,
  `limit_left = −∞`, and the signs alternate at `−2, −3, …` (gruntz
  rewrites `x!` to `Γ(x + 1)`); they used to stay unevaluated.
- `from_tree(to_tree(e))` is the identity for `Factorial` and `Binomial`
  (they came back as opaque `factorial(…)` / `binomial(…)` applications
  that no longer evaluated).
- A `physical_constant` whose value is a composite expression (ħ = h/2π)
  evaluates numerically; the value was never visited because the constant
  is an atom to every tree walk.
- `fourier_series`' fallback no longer returns the literal `nan`:
  coefficients go through the definite integrator (one-sided limits at
  `±π`, where tan-half-angle antiderivatives are `zoo`) and, without a
  closed form, stay `DefiniteIntegral` nodes that report
  `has_unevaluated()` and evaluate by quadrature.
- `evalf` decides `=` / `≠` between complex values (equality needs no
  order), so a `Piecewise` guarded by `a ≠ (−4)^(−1/2)` evaluates at a
  numeric `a`; this made the `exp(a·x)` Fourier fallback evaluable.
- The published package includes the book's Markdown sources, so
  `cargo test --doc` works from the crates.io tarball (`src/doctests.rs`
  includes them); only the rendered HTML is excluded.

## [0.22.0] - 2026-09-22

The audit release.  Four independent reviews of 0.21 — a documentation-vs-code
audit, an instrumented coverage run (89.9 % of lines, 93.4 % of functions),
and differential tests of the symbolic core, the polynomial / exact-kernel
layer and the statistics layer against SymPy, SciPy, statsmodels and mpmath
— found wrong answers in `simplify`, `series`, `integrate` and
`partial_fractions`, five defects in the `f64` reference distributions, a
degraded `solve`, and ~100 stale sentences.  All are fixed here, each with a
regression test citing its oracle (`tests/v21/`, 231 tests).  The poly /
exact-kernel layer (863 comparisons) and the data-statistics layer (381
comparisons: tests, intervals, ANOVA, Cox, logit / mnlogit) had no wrong
answers.  The Rust blocks of `README.md` and the book now compile and run as
doctests.  The LP pivot-path baseline and the Mathlib-compiled Lean
certificate are byte-identical to 0.14.0.

### Breaking

- The 0.18 transitional re-exports are removed; use the names' homes
  (unchanged since 0.18):
  - `stats::aggregation::{IntervalMethod, proportion_interval,
    proportion_interval_exact, proportion_interval_symbolic,
    z_for_confidence}` → `stats::estimation`;
  - `stats::hypothesis::{AnovaResult, anova_one_way}` → `stats::anova`,
    `stats::hypothesis::confidence_interval_mean` → `stats::estimation`;
  - `stats::reliability::{KappaCi, cochrans_q, cohen_kappa_ci,
    cohen_kappa_maximum, kappa_ci_from_confusion,
    kappa_maximum_from_confusion, kappa_test, kappa_test_from_confusion}` →
    `stats::agreement`; `{ConcordanceCounts, Dependent, concordance_counts,
    goodman_kruskal_gamma, kendall_tau_c, somers_d}` → `stats::data`;
    `{fisher_z, pearson_ci}` → `stats::estimation`; `{adjusted_residuals,
    chi2_contributions, compare_two_correlations, expected_counts,
    pearson_t_statistic, pearson_test, standardized_residuals}` →
    `stats::hypothesis`.
- The deprecated `stats::hypothesis::two_proportion_z_test` (0.18) is
  removed; the name is `z_test_two_proportions`.
- `numdist::binom::ppf` and `numdist::poisson::ppf` return
  `InvalidArgument` for `n` / `rate` above 2⁵³, where integer quantiles are
  no longer representable (they used to loop forever).

### Fixed — wrong answers

- **`simplify`: `(x^a)^b → x^(a·b)` fired whenever *either* exponent was
  an integer.**  An integer *inner* exponent is exactly the unsafe case:
  `(x²)^(3/2) → x³` (8 vs −8 at x = −2), `sqrt(1/x²) → 1/x`,
  `(x²)^(1/4) → sqrt(x)`, `cbrt(x³) → x`, and even the real constant
  `cbrt(|sin 6|²) → sin(6)^(2/3)` (complex).  The rule now fires iff the
  outer exponent is an integer, the base is known non-negative, or the inner
  exponent is a rational in (−1, 1] — SymPy's condition.  Consequently
  `(x^(1/2))^(1/3)` now simplifies to `x^(1/6)` and `sqrt(x³)` stays as
  written (it used to become `x^(3/2)`, false for x < 0).
- **`series` treated `|g(x)|` as the constant `|g(0)|`**: `series(|x²|)` was
  `0`, `series(cos|x|)` was `1`, `series(e^|x|)` was `1`.  `|g|` with an
  even-order leading term `c·xᵏ` of known sign expands as `±g`; an odd-order
  one has no two-sided expansion and returns the unevaluated `Series` unless
  both one-sided expansions of the whole expression agree.  `series(x^(5/2),
  x, 0, 3)` returned `0`; Puiseux exponents now return unevaluated at every
  order.
- **`integrate`**: `∫ sqrt(x²) dx` was `x²/2` (now `x·|x|/2`); the
  tan-half-angle path applied a constant factor twice (`∫ 3(cos x − 4) sin x`
  had derivative `3f`); `∫ cos x·R(sin x)` and `∫ sin x·R(cos x)` with a
  coefficient on `sin²`/`cos²` gave wrong `atan(3·tan(x/2))` forms (now the
  substitution `u = sin x` / `cos x`: `∫ cos x/(sin²x + 1) = atan(sin x)`);
  `∫ 1/(sin²x + 4)` was wrong; `∫ 1/((x²−4)²+1)` and `∫ ln((x²−4)²+1)`
  emitted `re(…)`/`im(…)` garbage.  Candidate antiderivatives from the
  Risch-rational, partial-fraction and Weierstrass routes are now verified
  (`F′ − f` at up to six rational points, 30 digits) and rejected on
  mismatch; a new exact route integrates biquadratic denominators
  `x⁴ + px² + q`.  `∫ 1/|x|` and `∫ |sin x|` are now unevaluated (they
  were wrong).  A latent infinite recursion in the piecewise-parameter path
  (reachable once a wrong result was no longer accepted) is fixed.
- **`partial_fractions`** returned float-rounded poles as "exact" rationals
  for irreducible cubics (`1/(−91/64·x³ + 3x − 1)` → denominators of
  5·10¹¹, relative error 2.6e-9).  Every `nsimplify`d root, residue and
  quadratic factor is now verified exactly against the denominator, or the
  exact symbolic route is used.
- **`Ex::to_latex` of `Sum` / `Product`** dropped the brace closing the
  subscript (`\sum_{n=1^{\infty} …`).
- **`stats::numdist`**:
  - `t` tails were exactly 0 once `x²` overflowed (|x| > 1.3e154), so for
    `df < 2` every far quantile was clamped to ±1.34e154 (`t::ppf(1e-200, 1)`
    is −3.18e199).  The tail is now computed from `ln(ν/x²)`; quantiles
    beyond `f64::MAX` are `±∞`.
  - Temme's incomplete gamma (a ≥ 10⁶) returned a *negative* tail when
    `erfc` underflowed (`gamma::sf(5.027e7, 5e7, 1) = −6.4e-321`), so
    `chi2::isf`/`gamma::isf` failed below ~1e-306.  Both terms now share one
    exponential scaling (`erfcx`), and the tail is clamped to [0, 1].
  - `norm::cdf(x)` was exactly 0 for x ∈ (−38.47, −37.5); `erfc` now
    continues through the subnormal range.
  - `poisson::cdf(+∞, μ)` was NaN (now 1).
  - `binom::ppf` / `poisson::ppf` hung for `n`, `rate` ≥ 2⁵³ (see Breaking).

### Fixed — degraded answers

- **`Ex::solve` on polynomials** depended on a constant factor: `p` gave
  `[5/2, 7/3, −7/3, 9]` but `4p` gave seven `RootOf`s (the rational-root
  search silently gave up on constant terms above 10⁶).  Polynomials are now
  factored exactly over ℤ first; each irreducible factor is solved once
  (radicals through degree 4, `RootOf` only over an irreducible factor of
  degree ≥ 5); each distinct root is returned once.
- `poly::modpoly`: `PolyIn::roots` over 𝔽₂ returned `[]` for `x² + x`
  (Cantor–Zassenhaus cannot split at p = 2); p = 2 is now evaluated
  directly.

### Added

- `numdist::binom::isf`, `numdist::poisson::isf` (the smallest `k` with
  `sf(k) ≤ q`), completing `{cdf, sf, ppf, isf}` for all eight families.
- `src/doctests.rs`: `README.md` and every book chapter are included as
  doctests (`cargo test --doc -- doctests::`, 233 blocks run); fragments
  were completed with hidden `# ` setup lines (invisible in the rendered
  book), stale examples corrected, and the 10 before/after migration
  listings stay `ignore`.
- `tests/unit/test_no_panics.rs` also ratchets runtime `assert!` /
  `assert_eq!` / `assert_ne!` in library code (`ASSERT_ALLOWLIST`, 82 sites
  in 19 files, each with its reason; the list may only shrink).
- `tests/v21/`: the audit regressions (`apart`, `integrate`, `numdist`,
  `series`, `simplify`, `solve`), every unit-conversion constructor pinned
  to its SI / NIST SP 811 definition (`units`), and tests for public
  functions no test reached before (`coverage`: `Matrix::lll`,
  `QMatrix::{set, get_mut, as_slice}`, every `Outcome` method, `fourier`,
  LaTeX / MathML / tree / compaction of ~45 node kinds).
- `tests/unit/test_size_assertions.rs` pins `SymplexError` at 72 bytes (the
  figure CONTRIBUTING.md quotes).

### Documentation

- README, CONTRIBUTING, `tests/README.md`, the crate docs and the book
  brought up to date: version banners (0.11 / 0.16 / 0.3 → 0.22), test and
  line counts, the 21 test binaries, the architecture tree (the whole
  `domains/stats/` directory and ~20 newer files were missing), the
  no-panics policy stated truthfully (the `assert!` category), fifteen
  continuous families (`FDistribution` was missing), Beta / Student-t CDFs
  are closed-form, polytope volume works in any dimension, five certificate
  provers, the `stats` data modules listed in the README, `cox` and
  `numdist` in the `stats` placement table, the `numdist` algorithm table
  describes the Temme dispatch and a realistic far-tail accuracy (≈ 1e-11
  for shapes ≥ 10⁴).
- `Ex::solve` documents the distinct-roots / factor-first convention.

### Known issues (reproducers in `tests/v21`, `#[ignore = "BUG: …"]`; all fixed in 0.22.1)

- `fourier_series(ln(2 + cos x), x, 2)` returns the literal `nan` without an
  unevaluated marker; `fourier_series` of `exp(a·x)` leaves a `Piecewise`
  that `eval` cannot resolve at a numeric `a`.
- `eval_f64` of the literal `(−1/2)!` is `Unevaluable` (expected √π);
  `limit_right(x!, x → −1)` stays unevaluated (expected ∞).
- `from_tree(to_tree((n+1)!))` yields an opaque `factorial` application.
- A `physical_constant` with a composite value (e.g. ħ = h/2π) is
  `Unevaluable`.
- Not fixed in this release, found by the same audit: three evaluators
  disagree on rational powers of negative numbers (`eval` takes the real
  root for `(−n)^(1/k)`, `evalf` the principal branch, `compile` the real
  root); `ln(exp(x)) → x` and `sqrt(x²) → |x|` fire without a realness
  assumption; `Poly::extended_gcd` does not use the fast gcd path.

## [0.21.0] - 2026-09-22

The consolidation release.  Three surveys catalogued every type, trait and
algorithm the crate had written more than once; this release merges them
into single shared versions — one `f64` linear-algebra kernel, one
polynomial gcd path, one RNG, one budget, one binomial, one special-function
registry, one numeric reference-distribution kernel — and gives the
symbolic `Matrix` the exact fast tier `Poly` already had.  Fifteen
measured speed-ups, from 6× to 1 000×; no numeric result changed; the LP
pivot-path baseline and the Mathlib-compiled Lean certificate are
byte-identical to 0.14.0.  ~10 900 lines added, ~5 800 removed.

### Breaking

- `exact_matrix::ExactScalar` (sealed; implemented only by `BigInt` and `Q`)
  is now a thin supertrait `Sealed + Ring + IntegralCoeff + Ord + Hash +
  Display + Signed`; its `add_ref`/`sub_ref`/`mul_ref`/`from_i64` are the
  `poly::traits::Ring`/`IntegralCoeff` methods of the same meaning.
- `stats::common`'s numeric Student-t helpers lost their unused `ctx`
  (`t_two_sided(op, df, confidence)`, `student_t_quantile_f64(op, df, p)`);
  they are `pub(crate)`, listed for completeness.
- `Matrix::as_qmatrix()` (`pub(crate)`) returns `Option<&QMatrix>` from a
  cache instead of a fresh `Option<QMatrix>`.
- `MultiPoly::mul`/`pow` assert on `u32` exponent overflow (previously a
  debug-only overflow); `try_mul`/`try_pow` return `None`.
- The homogeneous-tuple ratchet counts *any* repeated element type;
  `stats::Rng` is now a re-export of `base::rng::SplitMix64` (same path,
  same methods, bit-identical streams).

### Added — shared kernels (each replaces the copies named)

- **`base::dense_f64`** (crate-private): flat row-major `f64` Cholesky
  (`cholesky`, `cholesky_solve`, `spd_inverse`, `quadratic_form`,
  `congruence_inverse`, `solve_spd_regularised`), cyclic Jacobi `sym_eigen`
  with `EigenTol::{RelativeFrobenius, Absolute}` and `OnExhaust::{Error,
  Accept}`, `solve_partial_pivot`, `lstsq_householder`.  Replaces four
  Cholesky copies (`regression`, `cox`, `multivariate`, `sos`), two Jacobi
  copies, four dense solvers (`heurisch` moved from normal equations to
  Householder QR — no result changed).  `stats::common` gains `norm_cdf`/
  `norm_sf`/`norm_pdf`/`normal_two_sided`/`wald_summary` (five normal-CDF
  wrappers and two inlined Wald blocks gone).
- **`Field::poly_gcd` hook** in `poly::traits`, implemented for `Q` by the
  ℤ[x] primitive PRS (`poly::zpoly::gcd_via_z`): every `Poly::gcd`, `square_free_part`,
  `squarefree_factors`, `is_squarefree` — 17 call sites in `apart`, `gosper`,
  Risch, canonicalisation, `polybridge`, `ratfn` — is on the fast path with
  no call-site change.  Degree-30 gcd with a degree-10 common factor:
  349 ms → 20 ms.  `sturm::square_free_part` deleted.
- **`poly::zpoly`**: `ZPoly<O>` (common-denominator integer form, moved from
  `poly_ex` and made generic over the monomial order) now backs
  `MultiPoly::mul` (11.2 ms → 2.4 ms on 35×35 terms), new `MultiPoly::pow`
  (repeated squaring) and `eval` (14 ms → 0.8 ms); the integer-scaling
  helpers of `dense`, `multipoly`, `sturm`, `poly_ex` are one set.
- **`poly::interp`**: Newton divided-difference interpolation (`O(n²)`)
  replacing four `O(n³)` Lagrange builders; `Ex::poly_interpolate` through
  31 points 627 ms → 52 ms.  One `cauchy_bound`; `bigint_to_bigfloat`/
  `ratio_to_bigfloat` in `base::numeric` replace three identical copies;
  `ratio_to_f64` replaces `to_string().parse()` in `apart`, `evalf`,
  `algebraic`.
- **`base::rng`**: `SplitMix64` and `XorShift64Star` replace three and two
  copies (`stats::sample`, `optimize`, `expr_ops`; `ntheory`,
  `factor_zassenhaus`); the first ten outputs from seed 42 are pinned so the
  streams can never drift.  `shuffle` replaces two Fisher–Yates copies.
- **`base::budget`**: `Budget`/`BudgetHit`/`deadline_from`/`deadline_passed`
  moved from `linprog` (re-exported there and at the crate root); `Budget`
  gains `time_limit`, `with_max_steps`, `unlimited`, `exhausted`;
  `SosOpts`/`PolyhedronOpts` route their deadlines through it and expose
  `budget()`/`with_budget()`.  `EXPRESSION_BUDGET` lives in `base::config`
  (removing an allowlisted upward layering edge).
- **`base::combinatorics`**: `factorial` (binary splitting), `binomial`,
  `multinomial` replace fourteen copies (`fu::binomial_coeff` was `i64` and
  overflowed at `n ≈ 66`); `eval`'s negative-`n` guard preserved.
- **`Extended<T>` adopted** by `sets::Pos`/`Piece`, `limit::Ext`,
  `definite::LimVal`; `Ratio<BigInt>` is spelled `Q` throughout.
- **`regression::LikelihoodFit`** (`log_likelihood`, `null_log_likelihood`,
  `n_params`, `nobs`, `df_model` → provided `aic`, `bic`, `llr`,
  `pseudo_r_squared`, `llr_test`) and **`WaldFit`** (`coefficients`,
  `standard_errors` → `z_values`, `p_values`, `conf_int`) for `Logit`,
  `MnLogit`, `OrderedLogit`, `CoxModel`: sixteen method bodies deleted,
  `CoxModel::bic` added (`−2ℓ + p·ln n_events`, R's `BIC(coxph)`).  One
  `family_boilerplate!` macro; `Support::accumulate`,
  `Distribution::mass_below` centralise the continuous/lattice decisions.
- **`evalf_f64`/`evalf_complex64`**: `Ex::eval_f64` rounds the 128-bit
  result directly (the old decimal-string route was *less* accurate: up to
  4 ulp) and a per-thread constants cache stops recomputing π/ln 2 per call
  — `sin(1) + exp(2)` × 10 000: 7.3 s → 1.2 s.  No test value changed.
  `optimize::grow_bracket` and `partition_point_by` replace five
  bracket-doubling loops, two fixed 200-step bisections and three hand-rolled
  integer searches.  `base::graph::strongly_connected_components` (Tarjan)
  replaces `MarkovChain`'s Warshall reachability, whose classes are now
  cached (seven methods recomputed them).
- **`base::libfn::LibFn`**: the 50 special functions as an enum with
  `name`, `from_name`, `arity`; `eval`, `evalf`, `diff`, `latex`, `mathml`,
  `codegen`, `codegen_c`, `lambdify`, `codegen_py` and `parse` match on it
  exhaustively, so adding a function is a compile error in every printer
  instead of a runtime `unsupported`.  This surfaced that **C code
  generation lacked `erfinv`/`erfcinv`** (added, verified against
  `scipy.special.erfinv`), that `mathml` rendered only the Bessel family by
  name, and that `codegen_py` refused every special function (its refusal
  now cites the `scipy.special` routine).  The parser accepts every
  registry function at its arity.  `SymbolId` numbering is unchanged
  (names are resolved from text, not pre-interned).
- **`Matrix` exact tier, stage 1**: `QMatrix::char_poly_coeffs` (Berkowitz
  on the escalating kernel), `QMatrix::lu` (fraction-free), fraction-free
  `QMatrix::matmul`; `Matrix::{char_poly_coeffs, matmul, trace, lu}` take
  the exact path when every entry is rational and the decision is cached
  (`OnceLock`) instead of rescanned per call.  Rational 8×8 `char_poly`
  9.8 ms → 1.0 ms, 20×20 `matmul` 43 ms → 3.4 ms, 15×15 `lu` 25 ms → 2.7 ms;
  results are the identical arena nodes.  (Stage 2 — storing the tier
  instead of caching it — is blocked on `get_mut`/`IndexMut` needing an
  unreachable arm; documented in the source.)
- **ℤ[x] and 𝔽ₚ[x] on the generic polynomial**: `Ring + EuclideanDomain +
  IntegralCoeff for BigInt`, so `GenPoly<BigInt>` replaces the hand-rolled
  `Vec<BigInt>` helpers in the Zassenhaus factoriser; **`poly::modpoly`**
  (`RingOps` value-level ring, `Fp64`, `PolyIn<R>` with `div_rem`, `gcd`,
  `extended_gcd`, `powmod`, `roots`, …) replaces both hand-rolled 𝔽ₚ[x]
  copies (`factor_zassenhaus`, `ntheory`).  Factoring timings unchanged to
  within noise (debug and release, 16 polynomials up to `x¹⁰⁵ − 1`).
- **`stats::numdist`** — the `f64` reference-distribution kernel:
  `betainc_regularized_f64` (the full TOMS 708 dispatch: `bpser`, `bup`,
  `bgrat`, `bfrac`, `basym`), `gammainc_{lower,upper}_regularized_f64`
  (series / Lentz continued fraction, and **Temme's uniform asymptotic
  expansion** for `a ≥ 10⁶` within 40σ of the mean — DLMF 8.12.3–8.12.10
  with the `c₀, c₁` Taylor series of 8.12.12–8.12.14; verified against
  50-digit quadrature of the density, worst relative error `9·10⁻¹³` at
  `a = 10⁹`), and `norm`/`t`/`chi2`/`f`/`beta`/`gamma`/`binom`/`poisson`
  `{cdf, sf, ppf, isf}`.  Numeric quantile routes switched:
  `student_t(10).quantile_f64(0.975)` 670 ms → well under a millisecond,
  `confidence_interval_mean` on 20 points 407 ms → microseconds; the exact
  `Ex` CDFs are untouched and remain what `cdf()`/`p_value` return.  While
  writing the reference grid: **scipy 1.18's `gammaincinv(5·10⁷, 10⁻¹⁰)` is
  18 % off** (the mass below its answer is `1.18·10⁻¹⁰`); the crate's value
  carries `9.99999999995·10⁻¹¹`.

### Fixed

- `DivRem` over 𝔽ₚ: a `debug_assert!` contradicted the documented `(0, self)`
  contract for a zero divisor; removed.
- Two test oracles had used the composite `7 000 021 = 7 · 1 000 003` as a
  "prime" (`sympy.polynomial_congruence` happily works modulo composites);
  replaced by the prime `7 000 009`.

### Changed

- `scripts/gate.sh` honours `GATE_LOG_DIR` (the sandbox may forbid `/tmp`).

## [0.20.0] - 2026-09-21

Multinomial and ordinal logistic regression, Bayesian rater models (MAP
Dawid–Skene, MACE), and the exact linear-algebra kernel shared between the
matrices and the LP solver — with width escalation (`i64 → i128 → 256-bit
→ BigInt`) now available to `QMatrix`: `rref` on a 40×48 matrix is 4.7×
faster, results identical.  Additive except the `i128` verification below.
LP pivot paths and Lean certificates are byte-identical.

### Added

- **`regression::mnlogit`** — multinomial logit (reference category 0,
  Newton–Raphson on the full `(k−1)·p` vector with the exact block Hessian,
  step-halving, separation named by category and covariate).  `MnLogit`
  with `coefficients`/`standard_errors`/`z_values`/`p_values` as
  `[category−1][param]`, `cov_params`, `log_likelihood`,
  `null_log_likelihood`, `pseudo_r_squared`, `predict_proba`, `predict`,
  `relative_risk_ratios`, `conf_int`, `llr`/`llr_test`, `aic`, `bic`.
- **`regression::ologit`** — proportional-odds (cumulative-link) logit
  `P(Y ≤ j | x) = σ(θ_j − xβ)` with the *actual* thresholds reported
  (statsmodels' `OrderedModel` reports `θ₁` and log-differences; the
  transform is verified in the tests and standard errors are moved by the
  delta method), `OrderedLogit` with `thresholds`, `coefficients`,
  `cumulative_proba`, `predict_proba`, `predict`, `odds_ratios`,
  `conf_int`, `llr_test`, `aic`, `bic`.  34 tests against `statsmodels`
  (Newton fits at `tol = 1e-12`) and a 40-digit mpmath Hessian.
- **`aggregation::dawid_skene_map`** with `DawidSkenePriors` (Dirichlet
  priors on class prevalence and on the shared confusion rows; posterior-mode
  M-step; all-ones priors reproduce `dawid_skene` bit for bit) and
  **`aggregation::mace`** (Hovy et al. 2013: each rater copies the truth
  with competence `θ_r` or spams from `ξ_r`; `MaceOpts`, `MaceInit`,
  `Mace { posteriors, competence, spam_distribution, log_likelihood, … }`).
  On a 200-item synthetic table MACE labels 199/200 correctly against
  majority vote's 164/200.  Also `rater_confusion_from_gold` (exact
  per-rater confusion against gold) and `posterior_entropy` (bits, to rank
  items by uncertainty).
- **`markov`**: `impl Display for MarkovChain` (rows of exact rationals,
  labelled when labels exist) and `limiting_distribution()` (`Some(π)` iff
  the chain is irreducible and aperiodic).
- **`domains::exact_kernel`** (crate-private): the `Cell` trait (`i64`,
  `i128`, 256-bit, `BigInt`), the single-pivot fraction-free row update,
  a resumable Gauss–Jordan `Elimination` and `Bareiss` determinant shared
  by `exact_matrix`, `polytope` and `linprog`'s tableau.  `QMatrix::{rref,
  rank, det, inv, solve, nullspace, …}` start in `i64` cells and escalate
  on overflow, converting the pre-overflow state instead of restarting:
  `qmatrix/rref/40` 6.05 ms → 1.28 ms, `det/40` 2.1 ms → 0.26 ms,
  `inv/40` 12.5 ms → 4.6 ms.

### Changed

- The fraction-free kernel's exactness check runs in **release** builds
  (was `debug_assert!`): every exact division is verified (`BigInt`: the
  remainder of the `div_rem` already computed; fixed-width cells: a
  checking multiply).  For `i128` cells that adds a 256-bit verification
  multiply: `+3 %` on `rref/20`, `+29 %` on an `i128`-bound LP
  (`linprog/maximize/20x50`), `+3–4 %` on the certificate tree build.  An
  inexact division is now `ComputationFailed("internal invariant
violated…")` on the fallible operations and a logged fall-back to plain
  rational Gauss–Jordan on the infallible ones — never a wrong number.

## [0.19.0] - 2026-09-21

Cox proportional hazards (`stats::cox`), an exact fast path inside `Poly`
(5–50× on rational-coefficient arithmetic, byte-identical output), and the
last items from the verification passes: `fisher_exact` on large tables,
`usize` overflow in rank statistics, sticky sequential decisions.  LP pivot
paths and Lean certificates are byte-identical.

### Breaking

- **`Sprt` decisions are sticky**: once `AcceptH0`/`AcceptH1` is reached,
  `update`/`observe`/`decision` keep returning it (observations are still
  recorded); `reset()` clears.  New `is_decided()`, `stopped_at()`.
  Previously a later observation could pull the test back to `Continue`.
- **`fisher_exact` above `FISHER_EXACT_NUMERIC_THRESHOLD` (2 000 support
  points)** returns a numeric p-value (`ctx.from_f64`, dyadic-exact;
  `exp(ln p)` below `10⁻³⁰⁸` so `p_value_log10` stays informative) computed
  by an exact-ratio walk of the hypergeometric pmf from its mode; below the
  threshold the p-value is the exact rational as before.  Tables with cells
  in the `10⁶–10⁹` range that used to take minutes take `< 2 ms`; agreement
  with `scipy.stats.fisher_exact` is `≤ 10⁻⁹` relative (mpmath agrees with
  us to `10⁻¹³`; scipy's Boost CDF is the one `2.6·10⁻¹⁰` off on the
  million-count table).
- `mann_whitney_u`/`wilcoxon_signed_rank` exact frequency tables return
  `InvalidArgument` instead of overflowing when `n₁n₂ + 1` does not fit
  (`> 10¹⁸` observations — unreachable in practice, but now an error).

### Added

- **`stats::cox`** — `cox_ph(obs, x, &CoxOpts)` and
  `cox_ph_stratified(obs, x, strata, &CoxOpts)` (Newton–Raphson on the
  partial likelihood, `Ties::{Efron, Breslow}`, step-halving, monotone
  likelihood detected and named); `CoxModel` with `coefficients`,
  `hazard_ratios`, `standard_errors`, `z_values`, `p_values`, `conf_int` /
  `hazard_ratio_conf_int`, `log_likelihood`, `null_log_likelihood`,
  `llr_test`/`wald_test`/`score_test` (χ² `TestResult`s), `aic`,
  **`concordance`** (Harrell's C, exact rational over the usable pairs),
  `baseline_hazard` (Breslow), `predict_partial_hazard`,
  `linear_predictors`, `schoenfeld_residuals`, `martingale_residuals`,
  `cov_params`.  39 tests against `statsmodels.duration.PHReg` (Efron and
  Breslow, ties, censoring, strata, the Gehan–Freireich data); the score
  test of a single binary covariate equals the log-rank statistic.  Book:
  "Survival regression" in the statistics chapter.
- `hypothesis::tie_term`, `FISHER_EXACT_NUMERIC_THRESHOLD`;
  `KaplanMeier::{censoring_times, censored_at, quantile_strict}`;
  `Observation::{try_from_i64, try_from_q}` (the infallible `from_*` zip to
  the shorter slice, now documented).
- `Distribution::quantile_f64` on a lattice family without a closed CDF
  walks the pmf (`negative_binomial(3/2, 1/3).quantile_f64(0.9)`: 4 s →
  3 ms).

### Changed / performance

- **`Poly` exact fast path.**  When every coefficient is a rational
  literal, `Poly` now stores a `MultiPoly<Lex>` and runs `add`/`sub`/`mul`/
  `pow`/`derivative`/`eval`/`content_and_primitive`/`monic` on it through
  a common-denominator integer form, materialising the `Ex` view lazily;
  `Poly::new` reads expanded input and simple trees without the arena.
  Ordering, `Display` and every `terms()` result are unchanged (the
  Lex-not-GrevLex fact is pinned as a test).  Debug timings:
  `(x+y+z+1)^12` 45 ms → 7 ms, dense degree-30 product 43 ms → 3 ms,
  `eval` 30 ms → 0.6 ms, 1 000 `Poly::new` 50 ms → 4 ms.
- `fisher_exact`'s exact path stops recomputing two `O(x)` binomials per
  support point (2 000 points: 10 s → 25 ms, values identical).
- Tie terms (`t³ − t`, `n(n+1)(2n+1)`), `n₁n₂` products and
  `kappa_from_confusion`'s `row·col` are computed in `Q`, not `usize`.

## [0.18.0] - 2026-09-21

The statistics module gets one rule per sub-module for what lives where,
one rule for when a function takes a `Context`, one type for counts, one
wording for every level check — answering a user's "inconsistency in where
functions live and what types they expect".  Every moved item keeps a
documented re-export at its old path for this release.  Also: samplers
for the eight families that had none, and exact `RootOf` roots at degree
60 in seconds instead of `None`.  LP pivot paths and Lean certificates are
byte-identical.

### Breaking

*Where things live* (old path still works via a `pub use` marked "moved
in 0.18; kept for one release"):

| moved | to |
|---|---|
| `hypothesis::{anova_one_way, AnovaResult}` | `anova` |
| `hypothesis::confidence_interval_mean` | `estimation` |
| `reliability::{cohen_kappa_ci, kappa_ci_from_confusion, KappaCi, kappa_test, kappa_test_from_confusion, cohen_kappa_maximum, kappa_maximum_from_confusion, cochrans_q}` | `agreement` |
| `reliability::{pearson_test, pearson_t_statistic, compare_two_correlations, expected_counts, chi2_contributions, standardized_residuals, adjusted_residuals}` | `hypothesis` |
| `reliability::{pearson_ci, fisher_z}` | `estimation` |
| `reliability::{concordance_counts, ConcordanceCounts, goodman_kruskal_gamma, somers_d, Dependent, kendall_tau_c}` | `data` |
| `aggregation::{proportion_interval, IntervalMethod, proportion_interval_symbolic, proportion_interval_exact, z_for_confidence}` | `estimation` |

The rules, now in each module's docs: **`data`** describes a sample or the
association of two without inference; **`estimation`** estimates a
parameter (point or interval); **`hypothesis`** tests, effect sizes,
multiplicity, resampling, power, contingency-table tools; **`anova`** every
ANOVA and post-hoc; **`agreement`** every inter-rater statistic including
its inference; **`reliability`** scale reliability and item analysis only;
**`aggregation`** combines raters' labels or scores a rater.

*`ctx` iff the result contains an `Ex`* — dropped from
`confidence_interval_mean`, `confidence_interval_mean_z`,
`Ols::{conf_int, confidence_interval_mean_response, prediction_interval}`
(all return `Interval<f64>`) and from `reliability::spearman_brown(r, k)`
(uses `r.context()`).

*Counts are `usize`* — `estimation` used `u64`: `fit_binomial_p(ctx, n:
usize, …)`, `FamilyKind::Binomial { n: usize }`, `beta_binomial_posterior(
ctx, prior_alpha, prior_beta, successes: usize, failures: usize)`,
`posterior_predictive_beta_binomial(ctx, prior_alpha, prior_beta, n:
usize)`, `gamma_poisson_posterior`/`dirichlet_posterior_alphas`/
`dirichlet_multinomial_posterior(…, &[usize])`.  (The Beta shapes are named
`prior_alpha`/`prior_beta` so `alpha` means a level everywhere.)

*Parameters are borrowed* — `Sprt::bernoulli(p0: &Q, p1: &Q, …)`,
`Sprt::normal_mean(mu0: &Q, mu1: &Q, sigma: &Q, …)`, `Sprt::observe(x: &Q)`.

*Names* — `two_proportion_z_test` → `z_test_two_proportions` (deprecated
forwarding function kept).

*Errors* — `estimation`, `information`, `multivariate`, `sequential`,
`survival`, `data`, `markov` errors name the raising function (was a
module-wide `"stats::estimation"`); every level check says
"`confidence`/`alpha` must lie strictly between 0 and 1, got …".

*`RootOf` may now be reducible-squarefree*: when the ℤ-factorisation is
not certified within budget, `Ex::real_roots`/`root_of` name the root on
the square-free factor instead of returning `None`; evaluation is exact,
but two `RootOf`s of the same number may compare unequal (documented on
`real_roots`).

### Added

- **Samplers** for `Gamma` (Marsaglia–Tsang, with the `k < 1` boost),
  `ChiSquared`, `Beta` (gamma ratio), `StudentT`, `FDistribution`,
  `Poisson` (Knuth below `λ = 30`, Hörmann's PTRS above), `Geometric`,
  `NegativeBinomial` (Poisson–Gamma mixture, any rational `r`).  18 tests
  (`tests/v18/v18_samplers.rs`): moments, support, pmf at the mode, KS
  distance, reproducibility; a 400 000-draw χ² goodness-of-fit sweep found
  no bias.
- `hypothesis::counts_usize(&[Vec<usize>]) -> Vec<Vec<Q>>` so
  `confusion_matrix → chi_square_independence` composes.
- `stats::common` (crate-private): one copy each of `χ²`/`F`/`t`/normal
  tails, level checks, exact conversions — replacing three copies of
  `f_sf`, `chi_squared_sf`, `norm_isf`, `check_unit_open` across modules.

### Fixed / performance

- **`root_of` at degree ≥ 40 returned `None`** after 9 s: the Aberth
  iteration that indexes a `RootOf` started every guess on the Cauchy circle
  (radius `6·10⁷` for a binomial-tail polynomial whose roots lie in
  `|z| < 1.5`) and had not converged after 200 steps.  Starts are now
  Bini's Newton-polygon circles (MPSolve's initialisation); the Sturm chain
  and its sign evaluations run on integer-scaled polynomials (primitive
  PRS, integer Horner); Yun's square-free decomposition uses a gcd over
  ℤ.  Degree 40: `real_roots_isolate` 16.9 s → 0.1 s, `root_of` `None`
  → 1.1 s; degree 60: ~160 s → 3.4 s; random degree 30 `root_of` 6.0 s →
  0.25 s.  Values agree with `scipy.stats.beta.ppf` to 1e-16.
- `nroots` snaps a real part below the iteration's own tolerance to exactly
  `0.0` (`x² + 1` gives `±i`, not `−7.7·10⁻⁹³ + i`); genuinely tiny roots
  (`x² − 10⁻⁴⁰`) are untouched.
- `bigint_to_bigfloat` was lossy beyond `i128`; now exact.

## [0.17.2] - 2026-09-21

Two additions prompted by user feedback on the statistics API.  Additive.

### Added

- **Tiny p-values.**  A `χ²` of 12 800 has `p ≈ 2.3·10⁻²⁷⁸²`; `p_value_f64()`
  returned `0.0` (below `f64::MIN_POSITIVE`) although the exact expression
  `uppergamma(1/2, 6400)/Γ(1/2)` retained it.  New trait
  **`stats::PValue`** — `p_value_ex`, `p_value_f64`, **`p_value_log10`**
  (`−2781.636…` for that case), `p_value_ln`, `p_value_decimal(digits)`
  (`"2.3100265595063985852e-2782"`) — implemented for `TestResult`,
  `ChiSquareResult`, `AnovaResult`, `Mauchly`, `RepeatedMeasuresAnova`
  (plus `p_value_gg_log10`/`p_value_hf_log10`), with inherent methods on
  each and on `AnovaRow`; `Ols::p_values_log10`.  Evaluation runs in
  arbitrary precision internally, so the logarithm never underflows.  Book:
  "Tiny p-values" in the statistics chapter.
- **Exact intervals.**  `aggregation::z_for_confidence(ctx, &Q) -> Ex`
  (`√2·erfinv(c)`), `proportion_interval_symbolic(ctx, k, n, z: &Ex,
  method) -> Interval<Ex>` (Wald / Wilson / Agresti–Coull as closed forms
  in any `z`, unclipped) and **`proportion_interval_exact(ctx, k, n,
  confidence: &Q, method) -> Interval<Ex>`** — for Clopper–Pearson the
  endpoints are the roots in `(0, 1)` of the degree-`n` binomial-tail
  polynomials, certified unique by Sturm counting and returned as `RootOf`
  (a rational when it is one), evaluable to any precision
  (`eval_decimal(30)` agrees with mpmath to 25+ digits).  Documented limit:
  exact but `O(n)`-degree root isolation; practical up to `n ≈ 30`.
  `estimation::confidence_interval_mean_z_symbolic` / `_exact` likewise
  (`x̄ ± z·σ/√n` exact).  No exact Student-t quantile exists in the crate, so
  the t-based interval has no exact twin yet.  Book: "Exact intervals" in
  the response-analysis chapter.

## [0.17.1] - 2026-09-21

An independent verification pass over the whole of `symplex::stats`: 200 new
tests on fresh data with every reference value from scipy, statsmodels,
pingouin, `krippendorff` or `Fraction` arithmetic (`tests/v17/v17_audit_*.rs`),
re-running the oracle behind ~200 quoted doc numbers, and probing every
documented edge case.  Twenty bugs found and fixed; no API changed.

### Fixed

*Wrong values (highest severity first):*

- **`Truncated` of a lattice family** (Die, Poisson, Geometric, Bernoulli
  with a closed CDF) excluded the atom at the lower end: `die(6)` truncated
  to `[3, ∞)` had `cdf(3) = 0`, `quantile(1/4) = 4`, and its **sampler never
  drew a 3** (sample mean 5.49 vs 4.5); zero-truncated Poisson `cdf(2)` was
  half its value.  `mass_below` now subtracts `F(lo − 1)`.
- **`Transformed` by an even power** summed the `f_X(−√y)` branch even when
  `−√y` lies outside the support: `exponential(2)` squared had
  `P(Y < 1) = e² − e⁻² > 1`; `uniform(1, 3)²` had a doubled density.  One
  branch on a one-sided support, both on a symmetric one, `NotImplemented`
  for an asymmetric straddling support.
- **Decreasing `Affine` of a lattice family**: `cdf` used `1 − F(x − 1)`
  instead of `1 − F(⌈x⌉ − 1)` (`die(6).affine(−1, 7).cdf(7/2)` was `2/3`, not
  `1/2`); the reflected `quantile` was off by one lattice step where `F`
  hits `1 − p` exactly (now `None`, so the numeric route decides).
- **`Support::intersect` of a `Finite` table** lattice-normalised the region
  before testing listed values: `P(1 < X ≤ 5/2)` on `{1, 5/2, 3}` was `0`,
  not `1/3`.  Points are now tested against the raw piece (openness
  semantics unchanged).
- **Yates' continuity correction** (`chi_square_independence(…, true)`)
  shifted `|O − E|` by ½ even when `|O − E| < ½`, so `[[3, 3], [3, 4]]` gave
  `13/144` where scipy ≥ 1.7 and R give `0`; the shift is now
  `min(½, |O − E|)` and the doc no longer claims otherwise.
- **`operating_characteristic_bernoulli` / `expected_sample_size_bernoulli`
  at `p ∈ {0, 1}`** returned `NaN` (Wald's `h` root is at `±∞` there; the
  bracketing loop overflowed).  Both now return the limits, continuous with
  `p = 1 − 10⁻⁹`.
- **`studentized_range_cdf` for huge `df`** lost accuracy from `df ≈ 10⁸`
  (`1.2·10⁻⁷`) to garbage at `10¹⁵` (`1.0`) through catastrophic
  cancellation of three `O(ν)` terms; they are now cancelled analytically.
  Worst error over a 270-point scipy grid: `5·10⁻¹¹`.  Tukey p-values with
  very large `N` were affected.
- **Clopper–Pearson upper limit** at small `α` (`0.999` confidence) was off
  by `4·10⁻¹²`: `P(X ≤ k)` was formed as `1 − P(X ≥ k+1)`; the lower tail is
  now summed directly.
- **`Triangular` with the mode at an endpoint** (`c = a` or `c = b`,
  documented as allowed) divided by zero in the density, CDF, quantile and
  MGF; degenerate-mode branches added, `raw_moment` falls back to exact
  integration.
- **`log_normal(0, 1).cdf(0)`** (and `P(X ≤ 0)`, and `Truncated` CDFs at the
  support's own lower end) returned an unevaluable `erf(√2·ln 0)`; the
  closed forms are no longer called *at* the lower end.
- **`Mixture` quantiles**: two normals gave "unsupported support shape"
  (`ℝ ∪ ℝ`); two tables sharing an atom double-counted it
  (`quantile_f64(0.9) = 2`, want `3`).
- **`quantile_f64` on a lattice with `F(lo) ≥ p`** (`poisson(2).quantile_f64
  (0.1)`) failed Brent's sign test; returns `lo`.
- **`data::to_f64`** returned `NaN` when numerator and denominator each
  exceeded `f64::MAX` (`inf/inf`); uses `Ratio::to_f64`.
- **`maximum_of(exponential(λ), 2).mean()`** stayed an unevaluated
  `Integral`; `integrate_over` retries with the expanded integrand.

*Panics in library code (the crate promises none):*

- **`phi_coefficient`** and **`odds_ratio` / `fisher_exact`** multiplied
  table margins in `usize` — overflow (debug panic, release wrap) from
  margins around `2·10⁵` / cells around `5·10⁹`.  Products are now `Q`.

*Validation and small fixes:*

- `binomial(5, 1/3).affine(1, 1/2)` was accepted (support `ℤ + ½`); a
  non-integer numeric intercept on a lattice family is `InvalidArgument`.
- `die(6).transformed(x, 2x)` refused the affine route without trying the
  finite enumeration; it now falls through.
- `transformed(x, x³)` (documented as recognised) was refused because
  `solve` listed the complex cube roots; branches containing `I` are dropped.
- Doc citations corrected: `quantile_f64` quoted `t.ppf(0.975, 5)` with a
  wrong 11th digit; `KaplanMeier::quantile` cited statsmodels' strict-`<`
  quantile while implementing R's `≤` (now stated).

### Known limits found (not changed)

- `fisher_exact` enumerates the hypergeometric support with `BigInt`
  binomials: cells in the `10⁶–10⁹` range do not error, they take minutes.
- `kruskal_wallis`/`friedman`/`wilcoxon` tie terms and `kappa_from_confusion`
  products are still `usize` — safe below ~2.6·10⁶ observations.
- `Sprt` decisions are not sticky: `update` after `AcceptH1` may return
  `Continue`.  `LifeTableRow.censored` records only censorings that
  coincide with an event time.  `quantile_f64` on `NegativeBinomial` with a
  rational `r` takes ~4 s (Brent on a symbolic `Sum`).  No samplers for
  Gamma/Beta/χ²/t/F/Poisson/Geometric/NegBin.  "Clopper–Pearson contains
  Wilson" is not an invariant at 99 %.

## [0.17.0] - 2026-09-21

Factorial and repeated-measures ANOVA with post-hoc tests (`stats::anova`),
complex `f64` values as `num_complex::Complex64` instead of `(re, im)`
pairs, one `Extended<T>` for "a value or `±∞`", and two `simplify` fixes
the new work exposed: a multiple-angle expansion that was exponential in
`n`, and a Fu transform that produced a *wrong value* on a rotation-matrix
entry.  LP pivot paths and Lean certificates are byte-identical.

### Breaking

- **Complex `f64` values are `Complex64`** (`num_complex`, re-exported at
  the crate root and as `prelude::Complex64`): `Ex::eval_complex64 ->
  Result<Complex64, _>`, `Ex::nroots` and `Poly::nroots -> Result<Vec<Complex64>,
  _>` (were `(f64, f64)` pairs).  `.re`/`.im` replace `.0`/`.1`; you also
  get `.norm()`, `.arg()` and arithmetic.  `Ex::polar -> Polar { modulus,
  argument }` (was `(Ex, Ex)`).  `as_real_imag -> (Ex, Ex)` is unchanged
  (symbolic parts).
- New dependency `num-complex 0.4` (pure Rust, `MIT OR Apache-2.0`, part of
  the `num` family already used; builds for `wasm32`).

### Added

- **`stats::anova`** — `anova_two_way` / `anova_two_way_with(ss_type)` on a
  `TwoWayData` (`from_cells`, `from_i64`, `from_long(&[Observation { a, b,
  y }])`): rows for factor A, factor B, interaction, residual and total
  (`AnovaRow { source, ss, df, ms, f, p_value, eta_squared,
  partial_eta_squared }`), exact sums of squares for balanced and
  unbalanced designs with **Type I, II and III** (sum-to-zero contrasts)
  sums of squares matching `statsmodels.anova_lm(typ=…)`;
  `anova_repeated_measures(subjects_by_condition)` with exact
  `F`, **Greenhouse–Geisser and Huynh–Feldt ε** (exact rationals from the
  double-centred covariance), corrected p-values and **Mauchly's W** (exact,
  with Box's χ² approximation); `studentized_range_cdf/sf/quantile`
  (numerical, to `scipy.stats.studentized_range` at ~1e–14);
  **`tukey_hsd`** (`PairwiseComparison { i, j, diff, se, statistic, p_adj,
  ci: Interval<f64> }`) and `pairwise_t_tests` with `Adjustment::{Bonferroni,
  Holm}`.  36 oracle-cited tests (`tests/v17/`); the book's statistics
  chapter gained both examples.
- **`Extended<T>`** (`base::extended`, crate root and prelude): `NegInf |
  Finite(T) | PosInf` with the right `PartialOrd`/`Ord`, `is_finite`,
  `finite`, `into_finite`, `map`, `as_ref`, `Neg`, `from_f64`, `Display`
  (`-∞`/`∞`).  `Interval<Extended<T>>` is the honest spelling of an
  unbounded interval (`Interval::right_open(Finite(0), PosInf).contains(&…)`
  is correct where `Interval<Option<T>>` was not); `is_bounded()` on it.
  Replaces the private `Endpoint`, `Bound` and `EndVal` enums.
- `expr_complex::Polar { modulus, argument }`.

### Fixed

- **`simplify` returned a wrong value.**  Fu's TR10i (the inverse addition
  formulas) picked two trig factors out of a *longer* product and dropped
  the rest: `sin(c)·cos(a)·cos(b) + sin(a)·sin(b)` became `cos(a − b)`.  A
  numeric Euler rotation's `RᵀR` entry simplified to `0.78`.  The transform
  now requires the term to be exactly `coeff·trig·trig`.  (Latent since Fu
  was added; it surfaced because the measure fix below made that candidate
  win.)
- **`simplify(sin(n·x))` was exponential in `n`.**  `expand_trig` expanded
  `sin(nx)` by the addition formula recursively without collecting — `2ⁿ`
  terms, a 143 KB expression for `sin(14x)` and minutes for `sin(20x)`; the
  `simplify_idempotent` property test found it.  It now uses the De Moivre
  closed form (`⌈n/2⌉` terms): `simplify(sin(20x))` takes 44 ms instead
  of minutes.
- Fu's `L` measure counted trig *nodes* in the shared-DAG arena rather than
  *occurrences*, so `16cos⁷x − 24cos⁵x + 10cos³x − cos x` scored as one trig
  function; it now counts with multiplicity (SymPy's definition) and Morrie's
  law wins again.

### Changed

- `CONTRIBUTING.md`: `(re, im)` stays a convention for *symbolic* parts
  only; `f64` complex values are `Complex64`.  The interval table lists
  `Extended<T>`.

## [0.16.0] - 2026-09-21

The second half of "named endpoints everywhere": the algebra surfaces that
0.15.0 deferred.  Matrix decompositions return structs that state their
identity (`Qr { q, r }`: `A = Q·R`), the three extended-gcd routines that
disagreed on where the gcd sat now share `num_integer::ExtendedGcd`,
Denavit–Hartenberg links and generalised coordinates are structs, and
`Interval<T>` gains the set operations and `f64` corner-case handling that
a general interval type needs.  Also fixes the `wasm32` build (red in CI
since 0.13.0) and one silent coefficient-order inconsistency.  No numeric
result changed; the LP pivot-path and Lean-certificate baselines are
byte-identical.

### Breaking

*Matrix decompositions* — new module **`decompositions`** (crate root and
prelude) with generic result structs, each documenting the identity it
satisfies:

- `Matrix::qr -> Qr<Matrix> { q, r }` (`A = Q·R`), `ldl -> Ldl { l, d }`
  (`A = L·D·Lᵀ`), `lu -> Lu { l, u, perm }` (`P·A = L·U`),
  `diagonalize -> Diagonalization { p, d }` (`A = P·D·P⁻¹`),
  `jordan_form -> JordanForm { p, j }`, `hessenberg -> Hessenberg { h, p }`
  (`H = P⁻¹·A·P`), `rank_decomposition -> RankDecomposition { c, f }`
  (`A = C·F`); `QMatrix::{rank_decomposition, hessenberg}` likewise.
- `ZMatrix::hermite_normal_form_with_transform -> HermiteNormalForm { h, u }`
  (`H = U·A`), `smith_normal_form_with_transforms -> SmithNormalForm { s, u,
  v }` (`S = U·A·V`), `lll_with_transform -> LllReduction { reduced,
  transform }`; the `normalforms::*` functions on `Matrix` likewise.
- LLL's `delta: (i64, i64)` is `delta: num_rational::Rational64` in
  `Matrix::lll`, `ZMatrix::lll`, `ZMatrix::lll_with_transform`,
  `normalforms::{lll, lll_with_transform}`; `LLL_DEFAULT_DELTA` is
  `Ratio::new_raw(3, 4)`.

*Extended gcd* — one type, one field order:

- `ntheory::gcdex -> num_integer::ExtendedGcd<BigInt> { gcd, x, y }`
  (`a·x + b·y = gcd`; was `(g, x, y)`); it now delegates to
  `Integer::extended_gcd`, verified identical on `[−40, 40]²`.
- `GenPoly::extended_gcd -> ExtendedGcd<Self>` and
  `Ex::poly_gcdex -> Option<ExtendedGcd<Ex>>` with `x = s`, `y = t`
  (were `(s, t, g)` — the gcd *last*).

*Number theory, ODEs:*

- `diophantine::linear_diophantine -> Option<LinearDiophantine { x, y,
  x_step, y_step }>` (general solution `(x + k·x_step, y + k·y_step)`).
- `ntheory::continued_fraction_periodic -> Option<PeriodicContinuedFraction
  { pre_period, period }>`; `continued_fraction_reduce_periodic ->
  Option<QuadraticSurd { p, q, d }>` (the value `(p + √d)/q`).
- `Ex::solve_ode_ivp(…, ics: &[InitialCondition { order, x, value }])`
  (`y^{(order)}(x) = value`; was `&[(usize, Ex, Ex)]`).  `InitialCondition`
  is in the prelude.

*Robotics and dynamics:*

- `robotics::DhParams<'a>` is a struct `{ theta, d, a, alpha }` of unit-typed
  references (was a 4-tuple alias with two `Length`s adjacent);
  `fk_chain`, `fk_position`, `fk_rotation` take `&[DhLink<'_> { theta, d, a,
  alpha }]` of `&Ex` (was `&[(&Ex, &Ex, &Ex, &Ex)]`).  `From<DhParams> for
  DhLink` erases the units.
- `dynamics::total_time_derivative(expr, coords: &[GeneralizedCoordinate {
  q, q_dot, q_ddot }]) -> Ex` and `euler_lagrange(t, v, coords) -> Vec<Ex>`:
  the `(q, q̇)` pairs and the parallel `accels` slice are one struct per
  coordinate, and the only error (`coords.len() != accels.len()`) is
  unrepresentable, so both are **infallible** (no `Result`).
- `dynamics::manipulator_equation -> ManipulatorEquation { mass, coriolis,
  gravity }` (`M(q)·q̈ + C(q, q̇)·q̇ + G(q) = τ`; was `(M, C, G)`).

*Coefficient order:*

- `stats::regression::polyfit` returns coefficients **ascending**
  (`c[i]` multiplies `xⁱ`), the convention of `optimize::poly_fit`,
  `eval_poly`, `Ex::coeffs` and `control::from_coeffs`; it was the only
  routine returning numpy's highest-first order, so `eval_poly(&polyfit(..))`
  silently evaluated the reversed polynomial.  (`Poly::all_coeffs` stays
  highest-first: it is SymPy's `all_coeffs` by name.)

### Added

- `Interval<T>`: `intersect` (open wins at a shared endpoint), `hull`
  (closed wins), `contains_interval`, `clamp_to_closure`; `is_point` now
  needs only `PartialEq`, so it works for `Interval<Ex>`.
  `Interval<f64>::{is_finite, with_infinite_ends_open}` — `±∞` is not a
  real number, so an infinite end should be open, as `stats::Support` and
  the set layer already insist.  `Interval<Ex>::{contains, is_empty} ->
  Option<bool>` through the set layer.  The module docs list the corner
  cases per endpoint type (NaN, `±∞`, `-0.0`, discrete emptiness, `Ex`).
- `IntervalKind::{with_lower_open, with_upper_open, reversed}`.

### Fixed

- **`wasm32` build** (CI red since 0.13.0): `smallest_n_with_power` used
  `1 << 40` as a `usize` constant, which overflows on 32-bit targets; the
  cap is now `usize::MAX / 2` there.  `scripts/gate.sh subcrates` now runs
  the `wasm32-unknown-unknown` check and the fuzz-target check that CI runs.
- `proportion_interval` clips through `Interval::clamp_to_closure` instead
  of a hand `clamp`.
- **CI's `Test` job took two hours** because `cargo test --benches` runs
  every criterion bench once in debug mode and `qmatrix/hnf/40` (a 40×40
  random-entry Hermite normal form, exponential coefficient growth) took
  6,996 s of it.  The bench now stops at 20×20 for HNF; `scripts/gate.sh`
  gained a `benches` stage (23 s) that mirrors CI.

### Changed

- README: design principle 9 ("named positions, not tuples") and an
  "Intervals" paragraph in the API model; the SymPy comparison column is
  labelled 0.16.
- The homogeneous-tuple ratchet flags *any repeated element type* in a
  tuple (`(&Angle, &Length, &Length, &Angle)`), not only fully homogeneous
  ones; its allowlist now holds only justified conventions.

## [0.15.0] - 2026-09-21

Named endpoints everywhere.  A tuple whose positions share a type —
`(f64, f64)`, `(&Q, &Q)`, `(Ex, Ex, bool, bool)` — lets a `(lower, upper)`
become an `(upper, lower)`, a `(shape, scale)` a `(shape, rate)`, without a
compiler word; this release replaces every such tuple on the public surface
with a struct whose fields say what they are, at no runtime cost, and adds a
ratchet so none come back.  It also gives the crate one interval
vocabulary: `Interval<T>` carries its *kind* (`[a, b]`, `(a, b)`, `(a, b]`,
`[a, b)`), so a root isolator can return `(lo, hi]` cells beside an exact
`[r, r]` hit, a sequential test's continuation region is the open `(A, B)`
it actually is, and the symbolic-set and distribution-support layers speak
the same type.  No numeric result, LP pivot path or Lean certificate text
changed (the 4,000-LP path baseline and the Mathlib-compiled certificate
reference are byte-identical to 0.14.0).

### Breaking

New shared types (crate root and prelude): **`Interval<T> { lower, upper,
kind }`** with `IntervalKind::{Closed, Open, LeftOpen, RightOpen}` and
**`Bounds<T> { lower: Option<T>, upper: Option<T> }`** (a closed constraint
side that may be absent — `Interval<Option<T>>` is *not* the way to spell
an unbounded end, since `None < Some` makes its `contains` wrong).

*Intervals and bounds (`(lower, upper)` → `Interval<f64>` unless noted):*

- `stats`: `proportion_interval`, `credible_interval`,
  `confidence_interval_mean`, `confidence_interval_mean_z`, `bootstrap_ci`,
  `pearson_ci`, `KaplanMeier::confidence_interval`,
  `Ols::{confidence_interval_mean_response, prediction_interval}` return
  `Interval<f64>`; `Ols::conf_int` and `Logit::conf_int` return
  `Vec<Interval<f64>>`; `RatioEstimate.ci` and the new `KappaCi.ci` (replacing
  its `lower`/`upper` fields) are `Interval<f64>`; `min_max` returns
  `Interval<Q>`; `wald_boundaries` and `Sprt::boundaries` return the **open**
  continuation region `Interval::open(A, B)` (`contains(&llr)` is exactly
  "continue"; previously a closed pair).
- `optimize`: `differential_evolution` and `Ex::minimize_global_numeric`
  take `bounds: &[Interval<f64>]` (closed; another kind is
  `InvalidArgument`).
- Root isolation: `Ex::real_roots_isolate` and `Poly::real_roots_isolate`
  return `Vec<Interval<Ex>>` whose kinds are honest — `(lo, hi]` cells are
  `LeftOpen`, an exact hit is `Interval::point(r)`; test `kind ==
  IntervalKind::Closed` where you compared `lo == hi`.
- `SetEx::as_intervals` returns `Option<Vec<Interval<Ex>>>` (points as
  `Interval::point(p)`); `Context::interval(start, end, left_open,
  right_open)` is `Context::interval(start, end, kind: IntervalKind)`;
  `Interval<Ex>::to_set()` goes the other way.
- `stats::Support`: `Piece::Interval { lo, hi, lo_open, hi_open }` is
  `Piece::Interval(Interval<Ex>)`; `Support::as_interval` returns
  `Option<&Interval<Ex>>`.  `Support::from_pieces` now normalises an infinite
  end to open, as `Support::interval`/`integers` always did.
- `linprog`: `LpProblem::bounds(var, lo, hi)` is `bounds(var, Bounds<Q>)`
  (`Bounds::closed(lo, hi)`, `at_least`, `at_most`, `free`); `linprog(…,
  bounds: &[Bounds<Q>])`.  `polytope::BoundingBox` is `Vec<Bounds<Q>>`.

*Other pairs → structs:*

- `optimize::{minimize_scalar, golden_section}` and
  `Ex::minimize_scalar_numeric` return **`ScalarMinimum { x, value }`**;
  `linear_fit` returns **`LinearFit { slope, intercept }`**.
- `definite::quadrature` and `Ex::integrate_numeric_with` return
  **`QuadResult { value, error }`** (prelude, beside `QuadOpts`).
- `stats::information::marginals` returns **`Marginals { rows, cols }`**;
  `stats::aggregation::wins_matrix` takes `&[PairwiseOutcome { winner,
  loser }]`.
- Conjugate priors take named parameters instead of a `(&Q, &Q)` pair:
  `beta_binomial_posterior(ctx, alpha, beta, …)`,
  `gamma_poisson_posterior(ctx, shape, scale, …)` (scale, i.e. `1/rate`),
  `normal_known_variance_posterior(ctx, prior_mean, prior_sd, sigma, data)`.
- `polytope::Polytope::split` returns **`Split { nonnegative, nonpositive }`**
  (the former `.0`/`.1`, in that order).
- `certificates`: `prove_nonnegative_on_box`, `is_nonnegative_on_box` and
  `Ex::prove_nonnegative_on_box` take `&[BoxBound { var, lo, hi }]` (the
  struct existed) instead of `&[(Ex, Ex, Ex)]`; the polyhedron parameter
  `Option<(&Ex, &Ex)>` is `Option<&ParamBound { var, lower }>` in
  `PolyhedronProver::new`, `prove_nonnegative_on_polyhedron`,
  `prove_polyhedron_empty`, and `parameter()` returns
  `Option<ParamBound>`; every certificate's `identity()` returns an
  **`Equation`** (`lhs = rhs`) instead of `(Ex, Ex)`.
- **Certificate JSON (wire format):** `BoxCertificateData.bounds` entries are
  objects `{var, lo, hi}` (`BoxBoundTree`), its `terms` are
  `{lower_powers, upper_powers, weight}` (`HandelmanTermData`);
  `PolyhedronCertificateData.param` is `{var, lower}` (`ParamBoundTree`) and
  its `terms` are `{hyps, var_power, shift_power, weight}`
  (`PolyhedronTermData`).  Files written by 0.14 do not deserialise.

### Added

- **`Interval<T>`** (`src/base/interval.rs`): `closed/open/left_open/
  right_open/point` constructors, `contains`, `is_empty`, `is_point`,
  `is_ordered`, `width`, `map`, `as_ref`/`cloned`, `with_kind`, `reversed`
  (what a decreasing map does to an interval), `into_pair`; `From<a..=b>`
  (closed) and `From<a..b>` (`[a, b)`); `RangeBounds<T>` so it works with
  `BTreeMap::range` and friends; `Display` prints `[a, b]`, `(a, b]`, …
  `IntervalKind::{from_open_ends, lower_open, upper_open, with_lower_open,
  with_upper_open, reversed}`.
- **`Bounds<T>`**: `closed/at_least/at_most/free`, `is_bounded`, `contains`,
  `is_empty`, `map`, `into_interval`, `From<Interval<T>>` (the closure),
  `Display` (`[0, ∞)`).
- `Interval<Ex>::to_set() -> SetEx`.

### Deferred to 0.16.0 (allowlisted in the ratchet with that note)

`poly_gcdex` / `ntheory::gcdex` / `GenPoly::extended_gcd` (which today
disagree on the position of the gcd), the matrix decompositions
(`qr`, `ldl`, `diagonalize`, `jordan_form`, `hessenberg`,
`rank_decomposition`, `lu`, Hermite/Smith/LLL `*_with_transform`), LLL's
`delta: (i64, i64)`, `linear_diophantine`, the continued-fraction pairs,
robotics DH parameters and `fk_*`, `dynamics` coordinate pairs and
`manipulator_equation`, `solve_ode_ivp` initial conditions.

### Changed

- **Policy and ratchet.** `CONTRIBUTING.md` §"Tuples versus structs" states
  the rule and the conventions that stay (`(x, y)` points, `(re, im)`,
  `(numer, denom)`, `(quotient, remainder)`, key → value pairs, `shape() ->
  (rows, cols)`, symmetric pairs, enum tuple variants).
  `tests/unit/test_homogeneous_tuples.rs` parses `src/` with `syn` and fails
  when a file's public surface gains a tuple with a repeated element type
  beyond its allowlisted count, or loses one without the allowlist being
  tightened; each entry names its kept sites.
- Root-isolation documentation now says one thing: cells are `(lo, hi]`,
  exact hits are `[r, r]` (previously described four different ways).
- `SetEx` and `Support` print intervals identically to `Interval`'s
  `Display`.
- `CONTRIBUTING.md` §"Timeouts and granularity": sub-agents run in parallel
  only when spawned in one tool block.

### Fixed

- `Support::from_pieces` accepted a closed infinite end that `Support::
  interval` would have opened; it now normalises, and `Truncated`'s
  decreasing-map transform uses `Interval::reversed` instead of a hand swap.

## [0.14.0] - 2026-09-20

The second chunk of statistics on data: the questionnaire itself
(reliability and item analysis), models (exact least squares, logistic
regression), screening as answers arrive (SPRT), how label distributions
differ (information measures), time to completion or attrition (survival
analysis), and several measures at once (multivariate normal, PCA, order
statistics).  Additive over 0.13.0; 160 new oracle-cited tests
(`tests/v14/`).

### Added

- **`stats::reliability`** — exact `cronbach_alpha` / `cronbach_alpha_complete`,
  `standardized_alpha`, `kr20`, `guttman_lambda2`, `alpha_if_deleted`,
  `average_inter_item_correlation`, `split_half` / `split_half_correlation`
  (`SplitHalf::{OddEven, FirstLast, Custom}`) with `spearman_brown`; item
  analysis `total_scores`, `item_difficulty`, `item_discrimination_index`,
  `point_biserial`, `item_total_correlation`,
  `corrected_item_total_correlation`, `item_response_summary`
  (`ItemSummary`); agreement inference `cohen_kappa_ci` (`KappaCi`,
  Fleiss–Cohen–Everitt variance, exact), `kappa_test`,
  `cohen_kappa_maximum`, `cochrans_q`; ordinal association
  `goodman_kruskal_gamma`, `somers_d` (`Dependent::{Y, X, Symmetric}`),
  `kendall_tau_c`, `concordance_counts`; contingency diagnostics
  `expected_counts`, `chi2_contributions`, `standardized_residuals`,
  `adjusted_residuals`; correlation inference `pearson_test`,
  `pearson_t_statistic`, `pearson_ci`, `fisher_z`,
  `compare_two_correlations`.
- **`stats::regression`** — `ols(y, x, add_intercept)`, `wls`,
  `simple_linear_regression`, `polyfit`, the `Design` builder; `Ols` with
  exact `coefficients`, `fitted`, `residuals`, `ssr`/`ess`/`tss`,
  `r_squared`, `adjusted_r_squared`, `mse_resid`, `cov_params`, and
  `standard_errors`, `t_statistics`, `p_values`, `coefficient_tests`,
  `f_statistic` / `f_test`, `anova_table`, `conf_int`, `predict`,
  `confidence_interval_mean_response`, `prediction_interval`,
  `hat_matrix`, `leverage`, `cooks_distance`, `durbin_watson`,
  `log_likelihood` / `aic` / `bic`, `residual_standard_error`; `vif`;
  `logit` (IRLS, `LogitOpts`) with `Logit { coefficients, standard_errors,
  z_values, p_values, log_likelihood, null_log_likelihood,
  pseudo_r_squared, deviance, iterations, converged, … }`,
  `predict_proba`, `odds_ratios`, `conf_int`, `llr`, `aic`, `bic` —
  perfect separation is an error; `r_squared_from_correlation`,
  `slope_from_correlation`.
- **`stats::survival`** — `Observation`, `KaplanMeier::fit` with an exact
  life table (`LifeTableRow { time, at_risk, events, censored, survival,
  variance, cumulative_hazard }`), `survival_at`, `variance_at`
  (Greenwood), `cumulative_hazard_at` (Nelson–Aalen), `quantile` /
  `median`, `confidence_interval` (`CiMethod::{Linear, LogLog}`),
  `restricted_mean`; `log_rank_test` for `k` groups (exact rational
  statistic, `χ²(k − 1)` p-value); `exponential_rate`, `mean_event_time`,
  `survival_function`, `hazard_function`.
- **`stats::sequential`** — Wald's SPRT: `Sprt::bernoulli`,
  `Sprt::normal_mean`, `update` → `Decision::{Continue, AcceptH0,
  AcceptH1}`, `observe`, exact `log_likelihood_ratio`, `boundaries` /
  `wald_boundaries`, `operating_characteristic_bernoulli`,
  `expected_sample_size_bernoulli`.
- **`stats::information`** — exact `entropy` (`Base::{Nats, Bits}`),
  `perplexity`, `probability_vector`, `kl_divergence`, `cross_entropy`,
  `js_divergence`, `total_variation`, `bhattacharyya_coefficient` /
  `_distance`, `hellinger`, `joint_from_counts`, `marginals`,
  `joint_entropy`, `mutual_information` / `information_gain`,
  `conditional_entropy` (`Given::{Row, Column}`),
  `normalized_mutual_information` (`Norm::{Arithmetic, Geometric, Min,
  Max}`).  Logarithms are kept canonical over prime factors, so
  `H(½, ¼, ¼) = 3/2` bits exactly.
- **`stats::multivariate`** — `MultivariateNormal { mean, cov }` with
  `try_new`, `density`, `mahalanobis`, `entropy`, `marginal`,
  `marginal_1d`, `conditional` (Schur complement, exact), `affine`,
  `sample`; `covariance_matrix` / `correlation_matrix` from data; `pca`
  (exact, through `Matrix::eigenvects`; `RootOf` eigenvalues where the
  characteristic polynomial does not factor) and `pca_f64` (Jacobi).
- **`stats::order`** — `OrderStatistic` as a `Family`: `order_statistic(dist,
  n, k)`, `minimum_of`, `maximum_of` — density `k·C(n,k)·F^{k−1}(1−F)^{n−k}·f`,
  CDF through `betainc_regularized`, exact `Finite` tables for finite
  parents, sampling.

### Infrastructure

- `symplex` and `symplex-build` at 0.14.0; `symplex-macros` unchanged at
  0.3.3.  New test group `tests/v14/`.  Book: the response-analysis guide
  gains sections on reliability, regression, screening, information,
  survival and multivariate analysis.

## [0.13.0] - 2026-09-20

Statistics **on data**: the methods used to analyse many raters' answers
to many items — agreement, aggregation, rater quality, group comparisons,
screening under multiple comparisons — exact wherever the quantity is a
rational function of the data.  Additive over 0.12.0.  Every reference
value in `tests/v13/` (183 tests) is cited from statsmodels 0.15, scipy
1.18, the `krippendorff` package, SymPy 1.14 or Python `Fraction`
arithmetic; rationals are asserted exactly.  Book: "Analysing Rater and
Response Data" (a walk-through that is itself a test).

### Added

- **`stats::data`** — exact descriptive statistics on `&[Q]`
  (`Q = Ratio<BigInt>`): `mean`, `variance` / `std` / `covariance` with
  `Ddof::{Population, Sample}`, `pearson`, `median`, `quantile` /
  `quantiles` / `iqr` with `QuantileMethod::{Exclusive, Inclusive}`,
  `min_max`, `modes`, `frequencies`, `ranks` (average ranks) and
  `tie_sizes`, `spearman`, `kendall_tau` (τ-b), `skewness`, `kurtosis`,
  `central_moment`, `median_abs_deviation`, `zscores`, `geometric_mean`,
  `harmonic_mean`, `trimmed_mean`, `iqr_outliers`, `mad_outliers`,
  `from_i64` / `from_f64` (exact) / `to_f64`, `binomial_q`.
- **`stats::agreement`** — `RatingTable` (items × raters, missing cells)
  and, all exact rationals: `percent_agreement`,
  `pairwise_percent_agreement`, `cohen_kappa` (`KappaResult { kappa,
  observed, expected }`), `weighted_kappa` (`Weights::{Unweighted, Linear,
  Quadratic, Custom}`), `kappa_from_confusion`, `confusion_matrix`,
  `category_frequencies`, `scott_pi`, `fleiss_kappa` /
  `fleiss_kappa_ratings`, `krippendorff_alpha` (`Level::{Nominal, Ordinal,
  Interval, Ratio}`, missing data), `gwet_ac1`, `icc` / `icc_anova`
  (`IccForm::{Icc1, Icc1Average, Icc2Single, Icc2Average, Icc3Single,
  Icc3Average}`, Shrout–Fleiss), `kendall_w` (tie-corrected).
- **`stats::aggregation`** — `LabelTable`; exact `majority_vote` /
  `majority_votes` (`Vote { winner, tied, counts }`), `plurality`,
  `weighted_vote`, `worker_accuracy` (`Accuracy`), `category_metrics`
  (precision / recall / F₁), `gold_screening`; `dawid_skene` /
  `dawid_skene_counts` (EM, `DawidSkeneOpts`, posteriors, per-rater
  confusion matrices, priors, convergence), `bradley_terry` (Hunter's MM)
  and `wins_matrix`; `proportion_interval` with
  `IntervalMethod::{Wilson, ClopperPearson, AgrestiCoull, Wald}`.
- **`stats::hypothesis`** — `TestResult { statistic: Ex, p_value: Ex, df,
  alternative }` with `p_value_f64`, `statistic_f64`, `p_value_exact`;
  `Alternative::{TwoSided, Less, Greater}`.  Exact discrete tests
  (`binomial_test`, `fisher_exact`, `mcnemar_test`, `sign_test`;
  p-values as rationals); `chi_square_independence` (Yates optional),
  `chi_square_goodness_of_fit`, `g_test`; `t_test_one_sample`,
  `t_test_two_sample` (Student / Welch with exact Welch–Satterthwaite df),
  `t_test_paired`, `z_test_proportion`, `two_proportion_z_test`,
  `anova_one_way` (`AnovaResult`), `confidence_interval_mean`;
  `mann_whitney_u` (`RankMethod::{Exact, Asymptotic { continuity }}` — the
  exact null distribution of `U` by counting), `wilcoxon_signed_rank`,
  `kruskal_wallis`, `friedman`, `spearman_test`, `kendall_test`,
  `ks_one_sample`; effect sizes `cohens_d`, `hedges_g`, `glass_delta`,
  `rank_biserial`, `eta_squared`, `cliffs_delta`, `cramers_v`,
  `phi_coefficient`, `odds_ratio`, `relative_risk`, `cohens_h`;
  `bonferroni`, `holm`, `benjamini_hochberg`, `benjamini_yekutieli`
  (`Adjusted { p_adjusted, reject }`); `bootstrap_ci`
  (`BootstrapMethod::{Percentile, Basic}`) and `permutation_test` on
  `stats::Rng`; `sample_size_for_proportion`, `power_two_proportions`,
  `sample_size_two_proportions`, `power_t_test_two_sample` (noncentral t
  by quadrature), `sample_size_t_test_two_sample`.  Statistics are exact
  expressions; p-values are exact expressions through the symbolic
  StudentT / χ² / F CDFs (`betainc_regularized`, `uppergamma`, `erfc`).
- **`stats::estimation`** — maximum likelihood `fit_normal`,
  `fit_exponential`, `fit_poisson`, `fit_bernoulli`, `fit_binomial_p`,
  `fit_geometric`, `fit_uniform`, `fit_log_normal`; method of moments
  `fit_gamma_moments`, `fit_beta_moments`, `fit_negative_binomial_moments`,
  `fit_uniform_moments`, `fit_log_normal_moments`; `fit(FamilyKind, data)`
  and `method_of_moments`; `log_likelihood`, `aic`, `bic`; conjugate
  posteriors `beta_binomial_posterior`, `gamma_poisson_posterior`,
  `normal_known_variance_posterior`, `dirichlet_posterior_alphas`,
  `dirichlet_multinomial_posterior`, `credible_interval`,
  `posterior_predictive_beta_binomial` (exact `Finite` table);
  `standard_error_mean`, `confidence_interval_mean_z`.  Fits return
  `Distribution`s.
- **`stats::markov`** — `MarkovChain` on an exact `QMatrix`: `n_step`,
  `distribution_after`, `communication_classes` / `closed_classes` /
  `transient_states`, `is_irreducible`, `period_of` / `is_aperiodic`,
  `is_ergodic`, `is_regular`, `stationary_distributions` /
  `stationary_distribution`, `absorbing_states`, `is_absorbing_chain`,
  `fundamental_matrix`, `absorption_probabilities`,
  `expected_steps_to_absorption`, `hitting_probability`,
  `expected_hitting_time`, `fundamental_matrix_ergodic`,
  `mean_first_passage_times`, `mean_recurrence_times`, `sample_path`.
- `Distribution::quantile_f64(p)`: the numeric inverse CDF for every
  family — the closed form when there is one, else Brent's method on the
  CDF (compiled when possible, evaluated exactly otherwise), the smallest
  lattice point with `F(x) ≥ p` for a discrete family.

### Infrastructure

- `symplex` and `symplex-build` at 0.13.0; `symplex-macros` unchanged at
  0.3.3.  New test group `tests/v13/`.  The Python oracle environment
  gains scipy, statsmodels and `krippendorff`.

## [0.12.0] - 2026-09-20

### Breaking (all in `symplex::stats`; see `book/src/reference/migrating-0.12.md`)

- **`Distribution` is an opaque struct, not an enum.**  `ContinuousFamily`
  and `DiscreteFamily` are gone; each family is a struct (`Normal`,
  `Binomial`, `Finite`, …) implementing the new **`Family` trait**, and a
  `Distribution` is a shared handle to one.  `Distribution::downcast_ref::<Normal>()`
  recovers the struct; `Distribution::family()` exposes the closed forms
  (`mean`, `variance`, `raw_moment`, `cdf`, `mgf`, `quantile`, `entropy`
  as `Option<Ex>`, no `&ctx` argument), while `Distribution::{mean,
  variance, moment, cdf, mgf, entropy, …}` return `Ex` through the generic
  route when a closed form is missing.  `Distribution::from_family` admits
  user-defined families.
- **`Support` is a typed region**: a struct with a `Kind` (`Continuous` /
  `Discrete`) and `Piece`s (`Interval { lo, hi, lo_open, hi_open }` with
  `±∞` as expressions, `Point`), built with `Support::{reals, interval,
  half_line, integers, points, from_pieces}` and read with `as_interval`,
  `as_points`, `pieces`, `contains`, `intersect`, `to_set` / `from_set`.
  The variants `Continuous { lo, hi }`, `Discrete { lo, hi }` and
  `Finite(values)` no longer exist.
- `Distribution::finite` / `try_finite` take the context first.
- **`cdf` is clamped to the support** (SymPy's `cdf(X)(x)`): `0` below,
  the closed form on it, `1` above — `Uniform(0, 1).cdf(3)` is `1` (was
  `3`), `Geometric(p).cdf(k)` is a `Piecewise`.  The closed form on the
  support is `distribution().family().cdf(&x)`.

### Added

- **Conditioning, transformations and mixtures of distributions.**
  `RandomVariable::given(&event)` / `Distribution::truncated(&region)`
  (SymPy `given`): `E[N | N > 0] = √(2/π)`, `E[B | B ≥ 2] = 325/131` for
  `Binomial(5, ⅓)`, with the CDF and quantile transported when the inner
  family has them.  `RandomVariable::transform(name, &g)` /
  `Distribution::transformed(&x, &g)`: an affine `aX + b` transports every
  closed form exactly (`Affine`: `2N + 1 ~ Normal(1, 2)` with its mgf,
  quantile, moments, entropy); a strictly monotone `g` on the support
  (`eˣ`, `1/x`, `e^{−x}`, …) and the even shapes `X²`, `|X|`, `X^{2k}` go
  through the change-of-variables formula (`Transformed`: `N²` has the
  χ²(1) density `e^{−y/2}/√(2πy)`; `E[eᴺ] = √e` by LOTUS); a finite table
  or finite integer range has its values mapped and merged (`Die²`).
  `Distribution::mixture(&[(w, F), …])` (`Mixture`): every query is the
  weighted sum of the components'.  All sample (`Sampler`).
- **Events through the set machinery**: with numeric bounds any boolean
  combination of relations in the variable is accepted (`P(N² < 1)`,
  `P(N < −1 ∨ N > 1)`, `E[X | X² > 1]`), through `reduce_inequalities`;
  symbolic bounds keep the linear fast path (`P(X > a)`, `P(a < X < b)`).
  `RandomVariable::event_region` exposes the region.
- `FDistribution(d₁, d₂)` with mean, variance, raw moments and CDF.
- `betainc(a, b, x₁, x₂)` and `betainc_regularized(a, b, x₁, x₂)` (SymPy's
  4-argument form; methods `x2.betainc(&a, &b, &x1)`): exact polynomial
  folds for integer `a, b` (`cdf(Beta(2, 3)) = 3x⁴ − 8x³ + 6x²`), `∂x₁`/`∂x₂`
  derivatives, arbitrary-precision evaluation by Lentz's continued
  fraction (50 digits against mpmath), LaTeX / parse / `expr!`.  The
  `Beta` and `StudentT` CDFs are closed forms through it.
- `erfinv` / `erfcinv` in the `f64` runtime (≤ 3·10⁻¹⁶ relative), in
  `Ex::compile` and in Rust codegen — so `Normal` / `LogNormal` sample.
- Summation closes `Σ C(k+c, k) xᵏ` (and with `k`, `k²` factors), the
  binomial theorem with symbolic `n` **and** `p`, and finite hypergeometric
  sums with `C(n(k), m(k))` through Gosper; `C(n, k) = 0` for `0 ≤ n < k`
  in `eval` and `evalf`.  `NegativeBinomial` uses its true pmf.
- `RandomVariable::try_new` reports a distribution from another context
  as an error (`new` raises the crate's cross-context guard at
  construction instead of deep inside a query).

### Fixed

- `P(X > 1 ∧ X ≥ 2)` for a discrete variable took the *strict* bound's
  strictness with the *larger* bound (`17/81` for `Binomial(5, ⅓)`; correct
  `131/243`).  `P(X = 3 ∧ X > 5)` ignored the inequality (`40/243`; correct
  `0`).  `P(X = ½)`, `P(X = −1)` for integer-valued variables returned
  `C(5, ½)`-style expressions and Γ poles instead of `0`.
- Internal bound variables (`_t_cdf`, `_sample_p`, …) were interned by
  fixed name and could collide with a user's symbol; they are now chosen
  fresh against the expressions in play.

### Infrastructure

- `symplex` and `symplex-build` at 0.12.0; **`symplex-macros` at 0.3.3**
  (`expr!` knows `betainc` / `betainc_regularized`).  New test group
  `tests/v12/`.  Book: "What's New in 0.12", "Migrating from 0.11 to
  0.12", the statistics guide rewritten for the trait design.

## [0.11.2] - 2026-09-20

A **structure** release: the second pass of the review, consolidating the
plumbing the first pass left in place.  Additive over 0.11.1; byte-identical
on the pinned fixtures, the 4,000-LP pivot-path check and the downstream
generator's Mathlib-compiled output.

### Added

- `SosUnknown::budget_exhausted: Option<BudgetHit>` — the typed budget leaf
  `PolyhedronUnknown` already had, so all four provers report a spent
  budget the same way (`reason` still starts `budget exhausted: deadline`).
- `SymplexError::invalid_argument(operation, reason)` and
  `SymplexError::computation_failed(operation, reason)`: the two
  constructors seventeen private per-module helpers were re-implementing.
- `numeric::{Q, q, qi}`: the exact rational type and its literal
  constructors now live in `base::numeric` and are re-exported from
  `linprog` (their documented paths and the prelude's `Q` are unchanged).
  The exact-matrix, polytope and certificate modules no longer import
  `linprog` for a type alias.
- `GenPoly::try_div_rem` (internal) observes a zero divisor.

### Changed

- **One deadline rule, one LP meter.**  `linprog::{deadline_from,
  deadline_passed, LpMeter, Stop}` (crate-internal) replace the three
  copies of "earlier of the absolute deadline and now + time limit" and
  "has the deadline passed" in `linprog`, `polyhedron` and `sos`; the
  polyhedron prover's per-call pivot meter is the shared `LpMeter`.
- **`BigInt` tableau cells check their divisions.**  The fraction-free
  update is exact by Sylvester's identity; the fixed-width cells verify
  each quotient (Jebelean's multiplication, or the high-half bound for
  `i128`) and now the `BigInt` cells — whose division computes the
  remainder anyway — return `None` on a non-zero remainder, so a violated
  invariant surfaces as `ComputationFailed` in release builds instead of a
  truncated tableau.  Pivot paths are unchanged.
- **Layering made honest and enforced.**  CONTRIBUTING's dependency
  diagram now shows the two hubs (`base::{node, arena}` and
  `api::{expr, context}`), the ordered algorithm layers, and `output`
  *below* `domains`; `tests/unit/test_layering.rs` is a ratchet over the
  upward `crate::<layer>` edges of every file (each allowlisted with its
  reason) that fails when a file gains a new one.  Two moves that the
  ratchet made obvious: the `Sum`/`Product` evaluation shim
  (`sum_eval`) lives in `calculus` next to the engine it wraps, and the
  arbitrary-precision complex field operations (`base::bigcomplex`) are
  shared by `evalf`, the root finders and algebraic-number verification
  instead of imported upward from `evalf`.
- Dedup: `Arena::num_ratio` replaces eight identical
  `rational_to_expr`/`num_expr` helpers; one `describe()` serves the
  MathML and Python printers; the Rust printer's named constants come from
  the shared numeric runtime.

### Infrastructure

- **`scripts/gate.sh`**: the release gate one bounded stage at a time,
  every stage under `timeout` with its full output in a log and a one-line
  summary, so a failing test is named by `grep` on a log that already
  exists.  CONTRIBUTING: how to localise a slow test (smallest unit, 60 s
  cap, `--test-threads=1`), and the signature of rustdoc's silent fallback
  from the merged doctest binary (one broken doc example turns a 1-second
  stage into twenty minutes of standalone compiles).
- `.cargo/config.toml` caps every property-test case at 5 s
  (`PROPTEST_TIMEOUT`) and shrinking at 30 s: a rare expensive draw fails
  with its shrunk expression instead of stretching one test to minutes.
- `symplex` and `symplex-build` at 0.11.2; `symplex-macros` unchanged at
  0.3.2.

## [0.11.1] - 2026-09-20

A **correctness** release: the first pass of a line-by-line review of the
0.9–0.11 modules.  Additive over 0.11.0 (no signature changes); every
fix below was reproduced before it was made, and every new reference value
is cited from SymPy 1.14 / mpmath 1.3.  Byte-identical on the pinned Lean
fixtures, the 4,000-LP pivot-path check and a downstream generator's
Mathlib-compiled output.

### Fixed

- **Function analysis** (`calculus.util` on `Ex`): `is_increasing` /
  `is_decreasing` / `is_strictly_*` / `is_monotonic` / `is_convex` ignored
  poles strictly inside the domain on the assumption and inequality-solver
  routes — `(1/x).is_decreasing(&x, [−1, 1])` and `tan(x).is_increasing(&x,
  [0, π])` were `Some(true)` for a `Real` symbol.  An interior singularity
  with an infinite one-sided limit now refutes monotonicity (`Some(false)`;
  a monotone function has finite one-sided limits), a removable one lets
  the derivative analysis proceed, and an undecidable family is `None`.
  The documented `abs` support in `maximum` / `minimum` / `function_range`
  / `stationary_points` was unreachable (`|u|.maximum` errored on the zeros
  of `sign(u)`); kinks are now candidates and `f′` is solved with each
  `sign(gᵢ)` fixed to ±1 over all patterns (up to four factors), keeping a
  solution only where the sign agrees — `|u|.maximum(&u, [−1, 2]) = 2`,
  `(|u − 1| + u²).minimum = 3/4`, `function_range(|x|, [−1, 2]) = [0, 2]`,
  `stationary_points(|x| + x) = [−1, 0)`, all as SymPy.  "Could not
  decide" is reported as `ComputationFailed { operation, reason }` (was
  `NotImplemented`, the missing-feature variant).  Docs: `periodicity`
  returns *a* period (`sin²x·cos²x → π`), not necessarily the fundamental
  one; the three numeric steps in the extremum search are named; the
  `singularities` error contract is stated as it is.
- **Algebraic numbers**: `minimal_polynomial` returned an unverified
  factor — chosen by an `f64` evaluation, or the *smallest-degree* factor
  when that failed, and possibly reducible when Zassenhaus' recombination
  budget ran out.  The factorisation now reports completeness, the
  candidate is verified at 320 bits against a 2⁻¹²⁸ residual bound, and
  the result is `None` unless exactly one factor verifies.  `real_roots` /
  `root_of` derived the `RootOf` index from an `f64` sort while the
  evaluator uses a BigFloat sort; one shared definition
  (`poly::roots::rootof_roots`) now serves both, and the emitted index is
  checked against the Sturm isolating interval.  `minimal_polynomial`,
  `gcd_all` / `lcm_all`, `groebner`, `reduce_modulo`, `real_roots` and
  `factor_mod` no longer hold the arena write lock while they compute.
- **Special functions**: `uppergamma(s, 0)` and `lowergamma(s, 0)` folded
  to `Γ(s)` / `0` for numeric `s ≤ 0`, where the integrals diverge; they
  stay unevaluated.  `jacobi`, `gegenbauer` and `assoc_laguerre` with
  *symbolic* parameters expanded through a 1000-degree bound with a full
  `expand` per step (`jacobi(14, a, b, x).eval()` > 60 s); the bound is 16
  for symbolic parameters and the Jacobi sum is built incrementally
  (`jacobi(14, a, b, x)` in 67 ms release; higher degrees stay
  unevaluated).  `dirichlet_eta(n)` for large negative integers allocated
  `|n|` bits (`η(−10⁶)` 100 s → instant).
- **Arbitrary-precision evaluation**: `erf` / `erfc` used a Taylor series
  with 32 guard bits where the terms peak near `e^{x²}` — `erfc(7)` at 16
  digits was `−3.7·10⁻¹⁵` (mpmath `4.18·10⁻²³`), `erf(10)` at 50 digits
  was `−7.7·10¹⁶`, and through the `Γ(½ + k, x)` fold
  `uppergamma(1/2, 49).eval_f64()` was **negative**.  The Taylor branch
  carries `x²·log₂e + 8` guard bits, the asymptotic series is used when it
  applies, and the `Erfc` node evaluates through `arb_erfc`.
  `polylog(s, z)` near `z = 1` tested the wrong term for convergence
  (`polylog(2, 3/4)` wrong from digit ~150 of 200; now all 200 agree with
  mpmath) and for `s < 0` could return a partial sum silently; every series
  loop that can exhaust its term budget now returns `ComputationFailed`
  instead of a partial sum.  `dirichlet_eta` and `polylog(s, −1)` go
  through a native Borwein `η` (`η(1 + 10⁻²⁰)` no longer reports "zeta(1)
  is a pole"; `Li_s(−1) = −η(s)` for all real `s`).  `airyaiprime` /
  `airybiprime` at exactly `0` no longer divide by zero on the direct
  numeric path.
- **Code generation**: the Rust printer emitted a negative receiver before
  a method call — `(−2)^x` → `-2_f64.powf(x)` (= `−(2ˣ)`), `exp(−xy)` →
  `-(x * y).exp()`, `−2xy + z` → `-2_f64.mul_add(…)`, `min(−2, x)` →
  `-2_f64.min(x)` — all silently wrong; every `Std` method emission now
  parenthesises a receiver that starts with `-`.  The Python / NumPy /
  Julia printers emitted `x**(1/3)` and `x**(3/5)`, which are complex (or
  `NaN`) for negative `x` while Rust, C and `compile()` take the real root;
  they now emit `math.copysign(abs(x)**(1/3), x)`, `numpy.cbrt(x)`,
  `cbrt(x)` and the `copysign` idiom for odd denominators.  Python's
  `Factorial` is `math.gamma(x + 1)` (`math.factorial` rejects floats),
  empty `Min` / `Max` are `math.inf` / `-math.inf` like the other targets,
  and a lone `Mul` factor keeps its precedence as a `Pow` base.
- **Lean proofs** (`lean::{Block, Tactic, Decl}`): `Tactic::apply`'s
  arguments could still be split inside by the width-wrapping pass;
  Apply lines are now exempt.  An empty `by` block (in a `have` or a
  `Decl` body) rendered no tactic — a parse error; it renders `skip`.  A
  doc comment containing `-/` or `/-` is escaped.  The `Tactic::raw`
  docstring describes the actual placement of continuation lines.
- **MathML**: `to_mathml` kept every node's markup in its cache, so a
  5,000-deep expression took 1.0 GB (20,000: 15.5 GB); cached strings are
  released when their last reader has consumed them (14 MB / 137 MB).
  `E` and `I` render as `ⅇ` / `ⅈ` (SymPy's entities) and a `Piecewise`
  default branch as "otherwise".
- **Sums of squares**: a `SosOpts` time limit was measured only *after*
  the Newton-polytope pruning, whose one-LP-per-monomial stage ran
  unbudgeted; the deadline is now fixed at the start of `prove_sos`, the
  pruning LPs carry it, and exhaustion during pruning or before the SDP is
  assembled is `Unknown` with a reason saying where.  (The cheap
  refutation search stays unbudgeted: a counterexample is decisive.)
- **NTT**: `ntt` / `intt` / `convolution_ntt` validated the prime and
  re-derived the primitive root (three factorisations) once *per
  transform*; a convolution now plans once.  `convolution_ntt` with an
  empty operand validates its modulus like `ntt` does.
- **No-panics ratchet**: `tests/unit/test_no_panics.rs` stopped scanning a
  file at its first `#[cfg(test)]` *line*, so a test-only helper `fn` in
  the middle of a file hid everything after it — twelve library panic
  sites in `codegen.rs`, `gruntz.rs` and `factor_zassenhaus.rs`.  The
  ratchet now stops only at the `#[cfg(test)] mod` and all twelve sites
  are gone (non-panicking `let … else` paths and `ComputationFailed`
  invariants).  `GenPoly::div_rem` no longer `assert!`s on a zero divisor
  (`try_div_rem` observes it).

### Infrastructure

- `symplex` and `symplex-build` at 0.11.1; `symplex-macros` unchanged at
  0.3.2.  New test group `tests/v11/` (one module per review track, 44
  tests).  README refreshed for 0.11.

## [0.11.0] - 2026-09-19

### Added

- **`symplex::stats` — symbolic probability and statistics** (the
  counterpart of `sympy.stats`).  A `RandomVariable` is a symbol with a
  `Distribution`; the queries are exact wherever the parameters are:
  `mean`, `variance`, `std`, `moment(n)`, `central_moment`, `skewness`,
  `kurtosis`, `expectation(&g)` (polynomial `g` through closed-form raw
  moments — `E[X² + 3X] = 1` for a standard normal — else exact
  integration / summation over the support), `probability(&event)` for
  relations and their conjunctions (`P(Y > 2) = 17/81` for
  `Binomial(5, 1/3)`; the closed-form CDF is used before integration),
  `density`, `cdf`, `mgf`, `characteristic_function`, `quantile`,
  `median`, `entropy`, and seeded `sample` (`stats::Rng`, SplitMix64;
  inverse transform or cumulative sums).
  - Continuous families (each with support, density, closed-form
    moments where they exist, CDF, MGF, quantile and entropy, textbook
    formulas cited): `Normal`, `Uniform`, `Exponential`, `Gamma`,
    `ChiSquared`, `Beta`, `Cauchy` (moments honestly divergent),
    `Laplace`, `Logistic`, `LogNormal`, `StudentT`, `Weibull`, `Pareto`,
    `Triangular`.
  - Discrete families: `Bernoulli`, `Binomial` (all raw moments via
    Stirling numbers), `Poisson` (Touchard polynomials, CDF through the
    upper incomplete gamma), `Geometric`, `NegativeBinomial`,
    `Hypergeometric`, `DiscreteUniform`, `Die`, and **`Finite`** tables
    with arbitrary (non-integer) values (SymPy `FiniteRV`).
  - Several variables under independence: `stats::{expectation,
    variance, covariance, correlation}` over polynomials in the symbols,
    `stats::probability` on rectangles and on `X < Y` for two normals
    (`1/2` exactly), `sum_distribution` (Normal, Binomial, Poisson,
    NegativeBinomial, Gamma/Exponential/χ² closures),
    `conditional_expectation` / `conditional_probability`
    (`E[X | X > 0] = √(2/π)`), `entropy`.
  - Every constructor has a `try_` twin validating numeric parameters;
    `Support` and the family enums are `#[non_exhaustive]`.  All
    reference values in `tests/v10/` come from SymPy 1.14 (~130 tests).
- `Context::bool_true()` / `bool_false()` (SymPy `S.true`/`S.false`): the
  catch-all branch of a `piecewise`.

### Infrastructure

- `symplex` and `symplex-build` at 0.11.0; `symplex-macros` unchanged at
  0.3.2.  New test group `tests/v10/`.  Book: "Probability and
  Statistics" guide page.

## [0.10.1] - 2026-09-19

### Added

- **`Polytope::clip(&h) -> Clip`** (a downstream generator's request):
  the vertex sets of `P ∩ {h ≥ 0}` and `P ∩ {h ≤ 0}` from the cached
  vertices and their **tight sets**, without re-enumerating — crossings
  are computed on the edges, and adjacency is decided by the exact rank
  of the shared tight normals, which is correct for degenerate vertices
  too (verified against full enumeration on random 2–4-D cells including
  cubes cut through vertices, a pyramid with a degenerate apex and the
  4-D cross-polytope).  `Clip::{pos, neg, on}`, `pos_is_full_dimensional`
  / `neg_is_full_dimensional` (exact rank, no LP), and
  `pos_polytope` / `neg_polytope` returning the halves with their vertex
  cache **pre-filled** so a following `volume()` enumerates nothing.
  Also `Polytope::vertices_with_tight()` (`TightVertex = (point, tight
  indices)`) and `is_full_dimensional_from_vertices()`.  Per-candidate cut
  geometry in the generator: 0.372 s → 0.034 s (11×), identical output.

### Changed

- **The exact simplex is faster again, with unchanged pivot paths.**
  Profiling the certificate pipeline showed arithmetic *width*, not pivot
  count, was the cost: 97% of certificate LPs outgrow `i64` and the 4% that
  outgrew `i128` (peaks of 130–190 bits) took half the simplex time on
  `BigInt`.  Now: Jebelean exact division for `i64` cells (inverse
  computed once per pivot, verified by one exact multiply), hoisted
  divisor inverses for `i128`, a new **256-bit fixed-width cell** (`W256`,
  4×64-bit limbs with 512-bit intermediates) in the chain `i64 → i128 →
  W256 → BigInt`, heap-free scaling of small rationals, and lazily built
  stage bases in `PolyhedronProver` (651 ms → 45 ms per prover; only the
  stages a goal reaches are built).  Certificate pipeline on the n = 5
  floor generator: 3.83 s → 1.66 s; `poly_cert_bench` 2.3–7.4 ms → 0.3–0.5
  ms per goal.  Byte-identical on 4,000 random LPs and on the generator's
  Mathlib-compiled output.  A warm-started (dual simplex) design was
  measured and set aside: it could only serve the `λ = 1` stages and
  would land on different degenerate-optimal vertices, changing
  certificates.
- `tracing::debug!` events `linprog attempt` (cell type, overflow, µs),
  `stage basis built`, `refutation`, `certificate re-verified`, and
  `build_micros` on `polyhedron stage LP`.

## [0.10.0] - 2026-09-19

### Breaking

- `linprog::LpStatus` gained the variant **`BudgetExhausted`** (an LP
  stopped by a [`Budget`](#budget) — deadline or pivot cap — reports it as
  a status, not an error); exhaustive matches need an arm.
- `lean::Tactic` gained the variant **`Apply { head, args }`**
  (`Tactic::apply(head, args)`: an application whose argument list the
  renderer wraps, never splitting an argument); exhaustive matches need an
  arm.
- `lean::Decl` gained the field **`preamble: Vec<String>`** (lines such
  as `set_option maxHeartbeats 400000 in` and a comment emitted before
  the doc comment); 0.8 struct literals need the field — or use the new
  `Decl::new(kind, name, statement, body)` with `with_binders`,
  `with_doc`, `with_preamble`.

### Added

- **A budget on a single prover call** (a downstream generator's request:
  a deep cell at the degree-3 pairwise stage could run for minutes with
  nothing able to stop it).  `linprog::Budget { deadline, max_pivots }`
  (`#[non_exhaustive]`; `Budget::within(Duration)`, `::deadline(Instant)`,
  `::max_pivots(n)`), `LpProblem::with_budget`; the deadline is checked at
  every pivot and the pivot count spans all stages and the `i64 → i128 →
  BigInt` fallback chain.  `PolyhedronOpts::{with_time_limit(Duration),
  with_deadline(Instant), with_max_pivots(n)}` — a time limit is converted
  to a deadline when each `prove`/`prove_empty` starts, so one prover can
  be reused with a fresh per-call budget; on exhaustion the outcome is
  `Unknown(PolyhedronUnknown { budget_exhausted: Some(BudgetHit::Deadline
  | MaxPivots), .. })` and its `Display` says so.  `SosOpts::{with_time_limit,
  with_deadline}` for the interior-point and facial-reduction loops.
  Measured: a 50 ms limit returns at 50.0 ms; without a budget every pivot
  path is byte-identical (4,000-LP check).
- `PolyhedronProver::prove_poly` accepts a goal whose generator list has
  **unused** extra generators (exponent 0 in every term — a parameter
  carried on a row where it does not occur); only a foreign generator
  that actually occurs is an `InvalidArgument`.
- **Number theory** (`ntheory`): `nthroot_mod` for **any** modulus
  (Johnston's generalised root algorithm per prime, Hensel lifting, CRT),
  `quadratic_residues`, `is_nthpow_residue`, `polynomial_congruence`
  (roots of an integer polynomial mod `m`), `multiplicity`, `primenu`,
  `primeomega`, `primorial` / `primorial_up_to`,
  `continued_fraction_reduce` (+ `_periodic`, `_periodic_ex` → the
  quadratic surd), `is_carmichael`, `is_amicable`,
  `binomial_coefficients` / `_list`.
- **Discrete transforms** (`discrete`, exact over `Ratio<BigInt>`):
  `convolution` (linear, `_cyclic`, `_subset`, `_ex` over `Ex`), `ntt` /
  `intt` / `convolution_ntt` (number-theoretic transform for a prime with
  `len | p − 1`), `fwht` / `ifwht`, `mobius_transform` /
  `inverse_mobius_transform` (+ superset variants).  No floating-point
  FFT, by design.
- **Parsing**: `Context::parse_bool` (relations `< <= > >= == !=`,
  `and`/`&`, `or`/`|`, `not`/`~`, `True`/`False`, SymPy's `Eq(…)`/`And(…)`
  forms) and `Context::parse_implicit` (`sin x`, `2 sin x`, `sin 2x`,
  `x(x + 1)`; the ambiguity rules are in the docs).
- **Interchange and code generation**: `Ex::to_mathml` (Presentation
  MathML mirroring the LaTeX printer's parenthesisation; byte-identical to
  SymPy's on the checked cases), `Ex::to_srepr` and `Ex::to_dot`
  (SymPy `srepr` / `dotprint`, total, derived from `ExprTree`),
  `Ex::{to_python, to_numpy, to_julia}` and `{to_python_fn, to_numpy_fn,
  to_julia_fn}` (one table-driven printer with shared CSE; generated
  Python is executed against `eval_f64` in the tests; functions a target
  lacks are `NotImplemented`, never a guess).

### Infrastructure

- `symplex` and `symplex-build` at 0.10.0; `symplex-macros` unchanged at
  0.3.2.  `tests/v09/` complete (`ntheory_discrete`, `output`).

## [0.9.0] - 2026-09-19

A **comprehensiveness** release: the first pass over the gaps a SymPy user
hits in the first hour, taken from a module-by-module survey against
SymPy 1.14 (reference values from that SymPy are cited in the tests).
Additive over 0.8.2.

### Added

- **Algebraic numbers and polynomial algebra on `Ex`**
  (`api::expr_algebraic_ext`): `minimal_polynomial(&var)` (`√2 + √3` →
  `x⁴ − 10x² + 1`; `∛2`, `(1 + √2)⁻¹`, `√(3 + 2√2)`, the golden ratio),
  `gcd_all` / `lcm_all` (multivariate over ℚ, no variable named),
  `Ex::groebner(&polys, &vars, order)` and `reduce_modulo` with
  `MonomialOrder::{Lex, GrevLex}` (in the prelude), `real_roots(&var)` /
  `root_of(&var, k)` (rational roots exact, the others as ordered
  `RootOf` nodes), `factor_mod(&var, p)` (factorisation over GF(p), odd
  prime `p`), and `resultant_symbolic` / `discriminant_symbolic` for
  polynomials whose other coefficients are symbolic (Sylvester
  determinant: `disc(ax² + bx + c) = b² − 4ac`).
- **24 special functions**, each with constructor, exact special values,
  derivative, arbitrary-precision `evalf` (verified at 40 digits against
  mpmath on every branch), `Display`, LaTeX and `parse` support:
  `erfi`, `erfinv`, `erfcinv`, `expint(n, x)` / `E1`, `Shi`, `Chi`,
  `fresnels`, `fresnelc`, `lowergamma`, `uppergamma`, `polylog` (with
  `Li₂(½)` and the `s ≤ 0` rational values), `dirichlet_eta`, `airyai`,
  `airybi`, `airyaiprime`, `airybiprime`, `elliptic_k`, `elliptic_e`,
  `elliptic_f`, `elliptic_pi` (Carlson forms), `gegenbauer`, `jacobi`,
  `assoc_legendre`, `assoc_laguerre` (polynomial expansion for integer
  degree).  `integrate` now reaches `∫e^{ax²+bx+c}` for `a > 0` (`erfi`),
  `∫sinh(x)/x = Shi(x)` and `∫cosh(x)/x = Chi(x)`.  `expr!` accepts all
  of them (multi-argument ones in SymPy's `f(params…, x)` order).  Lean
  rendering of these is `NotImplemented` (no Mathlib spelling); codegen
  likewise.
- **Function analysis on `Ex`** (`api::expr_calculus_util_ext`, SymPy's
  `calculus.util`): `singularities`, `stationary_points`, `maximum` /
  `minimum` on a union of intervals (stationary points, endpoints,
  one-sided limits; `±∞` allowed), `is_increasing` / `is_decreasing` /
  `is_strictly_*` / `is_monotonic` / `is_convex` (exact Sturm route for
  polynomial and rational derivatives; three-valued, never a guess),
  `periodicity` (`sin(2x) + cos(3x)` → `2π`), `function_range`.
- **Matrices**: `singular_values`, `condition_number`, `rank_decomposition`
  (`A = C·F`), `hessenberg` (`(H, P)` by Gaussian similarity, exact over
  ℚ), `companion`, `jordan_block`, `permanent` (Ryser / subset DP),
  `row_insert` / `col_insert` / `row_del` / `col_del` / `permute_rows` /
  `permute_cols`, `inv_mod`, `matrix_log` (Jordan form, defective blocks
  included), `casoratian`; `ZMatrix::{lll, lll_default,
  lll_with_transform}` with exact rational Gram–Schmidt
  (`normalforms::lll`), `QMatrix::{rank_decomposition, pinv, hessenberg}`,
  `ExactMatrix::permanent`.

### Changed

- `Matrix::pinv` is defined for **every** matrix: a rank-deficient input
  goes through the full-rank factorisation `Fᵀ(FFᵀ)⁻¹(CᵀC)⁻¹Cᵀ` (SymPy:
  `[[1,2],[2,4]].pinv() = (1/25)·[[1,2],[2,4]]`) instead of returning
  `ComputationFailed`.
- `∫e^{x²}` and friends are no longer unevaluated `Integral` nodes (see
  `erfi` above).

### Fixed

- A random-LP property test boxed the variables at a fixed `±50`, which
  one seed showed can exclude the whole feasible region (`x₃ ≥ 52`); the
  box is now placed around a feasible vertex.  The solver's verdicts were
  correct.

### Infrastructure

- `symplex` and `symplex-build` at 0.9.0; `symplex-macros` at 0.3.2
  (new function names for `expr!`; the unreachable arm is a compile
  error).  New test group `tests/v09/` (algebraic, calculus_util, matrix,
  special; `ntheory_discrete` and `output` reserved for the next pass).

## [0.8.2] - 2026-09-19

### Changed

- `prove_sos` prunes its Gram basis by the **Newton polytope**: a monomial
  is kept only if its doubled exponent lies in the convex hull of the
  goal's support (one small exact LP per candidate), which is exactly the
  set of monomials that can occur in a sum-of-squares decomposition
  (Reznick) — so nothing provable is lost, and sparse goals fit the
  `max_basis` budget: `x⁸ + y⁸ + 1` needs 6 basis monomials instead of 45.
  Dense goals are unchanged (the eight pinned Mathlib shapes are
  byte-identical; the 120-case random stress is still 119 proved, the
  same one unknown).

## [0.8.1] - 2026-09-19

### Changed

- **`symplex-macros` 0.3.1.**  `syn` is now depended on with an explicit,
  minimal feature set (`parsing`, `printing`, `proc-macro`, `derive`,
  `full`; `default-features = false`): `full` — which the `rule!` macro's
  `if <closure>` condition needs — was previously enabled only
  transitively through `tracing-attributes`, and `extra-traits` /
  `clone-impls` were unused.  `matrix!` rejects an empty literal at
  compile time (a ragged one already was) and its expansion no longer
  contains an `.expect(...)`: the shape is checked by the parser, so the
  matrix is built infallibly.  The doc examples of `expr!`, `matrix!` and
  `eq!` show the actual `ctx,` first argument.  Two new `trybuild`
  snapshots (`tests/ui/matrix_{empty,ragged}.rs`).

## [0.8.0] - 2026-09-19

### Added

- **Structured Lean proofs: `lean::{Block, Tactic, Proof, Decl,
  DeclKind}`.**  A small model of a tactic proof — `Tactic::have(name,
  ty, Proof::term(…) | Proof::by(block))`, `Tactic::bullet(block)`,
  `Tactic::raw(text)` — whose `Block::render(indent)` places every line
  from its *tactic column* (a `· ` bullet moves it by two; a `by` block
  sits two further in) and wraps long lines past it, so a generator that
  stitches many certificates into one lemma no longer counts spaces or
  risks a tactic silently joining the wrong block.  `Decl` renders a
  `theorem`/`lemma`/`example` header in Mathlib's style (binders packed,
  ` :` closing the binder lines, statement on its own line, ` := by`) with
  an optional doc comment.  `PolyhedronLeanSteps::block()` returns a
  certificate's closing steps as a `Block` (its `have hg : … := by` with
  the `linarith` inside the `by`), and `to_block` is now that block
  rendered — byte-identical to before.  The renderer's output for a
  downstream generator's leaf shape (a `refine … ?_` call, bullets of
  wrapped `have hg` certificates) is pinned to text that compiled against
  Mathlib.
- `lean::lean_ident` is public: a plain identifier when the name is one,
  `«…»`-quoted otherwise.
- `Tactic::introduced_names()`: the `have` names a tactic introduces,
  recursively.

### Infrastructure

- `symplex` and `symplex-build` at 0.8.0; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.7.2.

## [0.7.2] - 2026-09-19

### Changed

- `PolyhedronProver` runs its cheapest LP stage **before** the exact
  refutation (ten sampled parameter values, one small LP each), so the
  majority of goals — true ones certified at the first stage — never pay
  for refutation; a certified goal has no counterexample, so outcomes are
  exactly those of "refute first".  Refutation itself now evaluates the
  goal and hypotheses through exact `MultiPoly` arithmetic instead of the
  expression arena.  n = 5 floor generator: 25 s → 21 s, output identical
  to the Mathlib-compiled 0.7.1 file (cumulative since 0.6.0: 103 s →
  21 s).

## [0.7.1] - 2026-09-19

### Changed

- **Hybrid arithmetic in the exact simplex.**  The fraction-free tableau
  is generic over its cell type and runs on `i64` cells first, then `i128`
  cells with exact 256-bit intermediates (a hand-rolled 128×128→256
  multiply, Jebelean exact division by the modular inverse, 256-bit
  comparison), and only on the first value that does not fit does it
  solve the problem again on `BigInt` cells.  Every decision is a sign
  test or an exact comparison of products, so all three take the same
  pivot path and give the same answer — verified byte-for-byte on 4,000
  random LPs and on the pinned Mathlib fixtures.  Measured on a
  downstream generator: the entries of its final tableaux have median
  67 bits and p90 99, so `i64` alone fit 38% of its 18,000 LPs and `i128`
  fits 99%; certificate LP time 8.3 s → 2.3 s, the whole run 33 s → 25 s.
- **Anti-cycling policy.**  Bland's rule used to take over permanently
  after the *first* degenerate pivot; the certificate LPs are degenerate
  from the start (zero right-hand sides), so they walked Bland's slow
  path throughout.  Dantzig's rule now stays in force until twelve
  consecutive degenerate pivots, then Bland's rule runs until the next
  improving pivot — still provably finite.  Handelman degree 8 on a
  2-variable box: 34,763 pivots / 92 s → 3,683 pivots / 11 s.  Optimal
  objectives and statuses are unchanged on the 4,000-LP check; 24 of them
  now report a different (equally optimal) vertex or a different (equally
  valid) Farkas vector, and one facet of the n = 5 floor generator's
  output uses a different hypothesis set — the generated file compiles
  against Mathlib.
- `PolyhedronProver` emits a `tracing::debug!` event per stage LP
  (`symplex::certificates::polyhedron`: degree, rows, cols, status,
  microseconds); `linprog` reports fallbacks to `BigInt` at `debug` and
  final tableau growth at `trace` under `symplex::linprog::growth`.

## [0.7.0] - 2026-09-19

### Breaking

- **One `certificates::Outcome<C, U>` for every prover.**  `BoxOutcome`,
  `HalfLineOutcome`, `PolyhedronOutcome` and `SosOutcome` are now type
  aliases of it (`Proved(C)` / `Refuted { point, value, param_value }` /
  `Unknown(U)`), so `PolyhedronOutcome::Proved(c)` still reads as before.
  What changes: `Refuted.point` is `Vec<(Ex, Q)>` everywhere (was
  `Vec<Q>` for boxes and `Q` for half-lines) and the variant is
  `#[non_exhaustive]` — patterns need `..`; the `Unknown` payloads are the
  structs `BoxUnknown { farkas, degree }`, `HalfLineUnknown {
  max_polya_power }`, `PolyhedronUnknown { degree, lambda_degree, pairwise
  }`, `SosUnknown { reason }` (all `#[non_exhaustive]`, all `Display`) —
  patterns become `Unknown(u)`.
- The Handelman box certificate struct `certificates::Certificate` is
  renamed **`BoxCertificate`** (`CertificateData` → `BoxCertificateData`);
  `certificates::Certificate` is now the **trait** (`goal`, `verify`,
  `to_lean`, `to_lean_with`, `to_json`, `from_json`) implemented by all
  five certificate types.  Inherent methods are unchanged.
- `lean::LeanOpts`, `certificates::PolyhedronOpts` and
  `certificates::SosOpts` are `#[non_exhaustive]`: struct literals
  (including `..Default::default()`) no longer compile outside the crate;
  use `::default()` with the `with_*` builders or assign fields on a `mut`
  default.  Adding an option is no longer a breaking change.
- Functions that could panic on their arguments now return `Result`
  (found by the panic audit below): `StateSpace::{controllability_matrix,
  observability_matrix, discretize_zoh}` (were infallible),
  `StateSpace::{riccati_residual, ackermann}` (were `Option`),
  `robotics::homogeneous`, and `dynamics::{total_time_derivative,
  euler_lagrange, mass_matrix, christoffel_symbols, coriolis_matrix,
  manipulator_equation}`.  `StateSpace::char_poly` and
  `matrix_decomp::wronskian` return NaN instead of panicking on an
  ill-shaped model / empty list (new `try_char_poly`, `try_wronskian`
  return the error); `is_controllable` / `is_observable` are `false` for an
  ill-shaped model; `ode::solve_ode_system{,_nonhomogeneous}` return `None`
  where they could panic.

See `book/src/reference/migrating-0.7.md` for the one-line fix to each.

### Added

- `Outcome::{is_refuted, is_unknown, into_certificate, refutation, unknown,
  map_certificate}`; `Display` for outcomes.
- `RealLineCertificate::{goal, to_data, from_data, to_json, from_json}`,
  `RealLineCertificateData`, and `Display`.
- Builders `PolyhedronOpts::{with_max_degree, with_max_lambda_degree,
  with_pairwise, with_staged}` and `SosOpts::{with_max_basis,
  with_max_iterations, with_rounding_digits, with_max_facial_reductions}`.
- **No-panic policy and ratchet.**  `CONTRIBUTING.md` spells out the
  practical policy (validate at the boundary, `Result` for failure, `Option`
  for absence, `debug_assert!` for invariants, `std`-style `try_` siblings
  for indexing; error plumbing measured at zero cost);
  `tests/unit/test_no_panics.rs` counts `unwrap`/`expect`/`panic!`/
  `unreachable!` in library code and fails on any increase — or on an
  allowlist that is no longer tight.  108 sites removed; the allowlist is
  the two documented logic errors, the arena's `u32` index conversion and
  the compile-time `const_assert_dim!`.

### Fixed

- `matrix_decomp::wronskian` panicked when the derivatives exceeded the
  expression budget (reachable from user input); it now returns NaN and
  `try_wronskian` reports the error.

### Infrastructure

- `symplex` and `symplex-build` at 0.7.0; `symplex-macros` unchanged at
  0.3.0.  Pinned Mathlib-compiled fixtures byte-identical to 0.6.1.

## [0.6.1] - 2026-09-19

### Added

- `PolyhedronCertificate::used_hyps()`: the indices of the hypotheses the
  identity actually uses, so a generated lemma can list exactly those in
  its signature (previously recoverable only by scanning the emitted Lean
  for hypothesis names).
- `PolyhedronProver::prove_poly(&Poly)`: prove a goal that is already an
  exact polynomial, skipping the expression round trip; its generators may
  be any subset of the prover's in any order (a tool's `(j, r, t)` against
  the prover's sorted `(r, t, j)`).
- `MultiPoly::{as_constant, affine_form, eval_var, to_ex}`: the value of a
  constant polynomial; a degree-≤ 1 polynomial as `(coefficients,
  constant)`; substitution of a value for one variable that **keeps** the
  variable count (unlike `substitute`, which drops the variable and shifts
  the indices — the natural operation for instantiating a parameter); and
  the bridge to an `Ex` over named symbols.  `MultiPoly` and the rational
  type `Q` are re-exported from the prelude.
- `Polytope::is_full_dimensional()` and `Polytope::interior_point()`: one
  exact LP (the largest common slack), instead of testing `volume() > 0`;
  defined for unbounded polyhedra too.  `HalfSpace::{value_sign, is_tight,
  is_trivial, normalized, same_hyperplane}` — gcd-free sign tests and the
  canonical hyperplane key that identifies a cut with its flip and its
  rescalings.

### Changed

- **`Polytope::vertices` is 10× faster and cached.**  The enumeration runs
  in integer arithmetic throughout: half-spaces are scaled to integers
  once, only *distinct* hyperplanes are combined, each `n × n` system is
  solved by the fraction-free kernel (which yields the point as `X / D`
  directly) and containment is the sign of `a·X + b·D` — no rational
  reduction until the accepted vertices are returned.  The vertex list is
  cached on the polytope (`Clone` carries it; `PartialEq`/`Debug` ignore
  it).  `Polytope::volume` enumerates vertices once and hands each facet
  its own vertices (those on its hyperplane, projected) instead of
  re-enumerating at every level of the recursion; `is_bounded` recognises a
  description with axis-parallel bounds on every coordinate without LPs.
  `HalfSpace::contains` / `Polytope::contains` use the gcd-free sign test.
  Profiled on a downstream decision-tree generator (three free
  coordinates, 49 leaves): 103 s → 34 s with byte-identical output; the
  remaining time is the certificate LPs.
- `ParametricPolytope::at` instantiates through exact `MultiPoly`
  arithmetic rather than the expression arena (identical results; verified
  against the symbolic route on random families).

### Fixed

- `lean::wrap_lean` measures its continuation indent from the line's
  *tactic column* (past `· ` / `. ` bullets), not from the leading spaces.
  A bullet's tactics sit two columns right of the `·`, so the old `+2`
  put a wrapped `· have … := by tac` continuation at the same column as
  the following tactic — Lean then swallowed that tactic into the inner
  `by` block (`expected '{' or indented tactic sequence`), or rejected a
  wrapped application argument (`unknown tactic`).  Both shapes were
  compiled against Mathlib before and after; plain (non-bullet) lines and
  the pinned certificate fixtures are unchanged.

### Infrastructure

- `symplex` and `symplex-build` at 0.6.1; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.6.0 (`cargo semver-checks`: no semver update
  required).  End-to-end check: a downstream Lean generator built against
  this tree reproduces its 0.6.0 output byte-for-byte apart from the
  `wrap_lean` bullet fix, and the generated file compiles against Mathlib.
- `CONTRIBUTING.md`: no Cargo feature flags by design; the explicit
  context argument of `expr!` and friends is deliberate.

## [0.6.0] - 2026-09-18

### Added

- **Sums-of-squares certificates** (`certificates::prove_sos(goal, &vars,
  &SosOpts)`, `is_sos`): prove `g ≥ 0` on all of ℝⁿ by an exact
  decomposition `g = Σₖ dₖ·pₖ²` with rational `dₖ > 0` and
  rational-coefficient `pₖ` — the class of goals the box, half-line and
  polyhedron certificates could not reach (`(x − 1)² + (y − 1)²`, the
  AM–GM form `x⁴ + y⁴ + z⁴ + 1 − 4xyz`, …).  Outcomes
  `Proved(SosCertificate)` / `Refuted { point, value }` (exact rational
  point, found by a grid and a rationalised numerical minimiser) /
  `Unknown { reason }` (Motzkin's polynomial, odd degree, or a search that
  did not converge — never a wrong `Proved`).
  - The pipeline is Peyrl–Parrilo made exact: the Gram SDP `g = mᵀQm`,
    `Q ⪰ 0` is solved numerically by a small dense primal–dual
    interior-point method (HKM direction, Mehrotra predictor–corrector,
    exact-to-the-boundary steps; no external solver) whose zero objective
    makes it converge to the analytic centre; the solution is rounded,
    projected back onto the coefficient constraints exactly (rational
    least-norm correction) and tested for positive semidefiniteness with
    the rational `QMatrix::ldl_psd`, whose factorisation *is* the
    decomposition.
  - Goals with real zeros have only singular Gram matrices; the search then
    performs **facial reduction**: the numerical kernel is made exact
    either directly (rational kernel) or through its integer relations
    (LLL on the kernel lattice, with Newton-refined zeros of the goal
    providing a double-precision kernel), the problem is restricted to the
    face `Q = B Q' Bᵀ` and re-solved, up to three times.  Sums of two or
    three random squares with irrational common zeros are recovered
    exactly (119 of 120 random cases through degree 6 in two and three
    variables).
  - `SosCertificate::{goal, vars, basis, gram, squares, rank, identity,
    verify, lean_hints, to_lean, to_lean_with, to_data, from_data,
    to_json, from_json}` (`from_*` re-verify), `Display` as
    `goal = d₁·(p₁)² + …`.
  - Lean export: `have h : goal = d₁ * (p₁) ^ 2 + … := by ring` then
    `rw [h]; positivity` — two deterministic steps, no search.  Eight
    shapes (squares with a common zero, positive definite quadratics and
    quartics, a perfect square, a product of squares, univariate, three
    variables, AM–GM) compile against Mathlib (Lean 4.30.0) with
    `linter.style.longLine` on; the emitted text is pinned to that file
    (`tests/fixtures/sos_certificates.lean`).
- `lean::wrap_lean` never breaks between `^` and its exponent.
- `Poly::new` docs point to `try_new` for the failure reason.

### Infrastructure

- `symplex` and `symplex-build` at 0.6.0; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.5.0.

## [0.5.0] - 2026-09-18

### Breaking

- `certificates::PolyhedronOutcome::Refuted` gained the field
  `param_value: Option<Q>` (the sampled parameter value at which the
  counterexample was found).  Patterns must add `..` or bind it.
- `lean::LeanOpts` gained the fields `single_fraction` and `symbol_text`
  (as announced in 0.4.0: use `..Default::default()` or the `with_*`
  builders; literals naming every field break).

### Added

- **`certificates::PolyhedronProver`**: `PolyhedronProver::new(&hyps,
  param, &opts)?` parses the hypotheses and builds every stage's product
  basis once; `.prove(&goal)` / `.prove_empty()` then certify any number
  of goals against them (accessors `hyps`, `gens`, `parameter`, `opts`).
  `prove_nonnegative_on_polyhedron` and `prove_polyhedron_empty` are now
  one-line wrappers over it.  A goal mentioning a symbol absent from the
  hypotheses is an `InvalidArgument`.
- `LeanOpts::symbol_text` (+ `with_symbol_text(name, text)`): render a
  symbol as given Lean text everywhere it occurs — `("J", "(j : ℝ)")` for
  a parameter that is a cast natural in the surrounding proof.  Applied
  by `Ex::to_lean_with` and therefore by every certificate emitter
  (goals, hypotheses, `λ`, `hg`); hypothesis *names* such as `e1J` are
  untouched.  In a full theorem the binder keeps the plain identifier.
- `LeanOpts::single_fraction` (+ `with_single_fraction`): combine over a
  common denominator before rendering, so an `expand`ed rational function
  prints as `(-(8 * j) - 2) / (7 * j + 4)` instead of
  `-(8 * j / (7 * j + 4)) - 2 / (7 * j + 4)`.
- `PolyhedronLeanSteps::to_block(indent)` now re-flows its lines to
  Mathlib's width (indent included) and `to_block_width(indent, width)`
  takes an explicit width; `lean::wrap_lean` never starts a continuation
  line with `:=`, so `have hg : … := by` keeps its `:= by`.
- **`polytope::ParametricPolytope`**: a family `{x : hₖ(j, x) ≥ 0}` with
  half-spaces affine in `x` and polynomial in one parameter.  `at(&j)`
  instantiates exactly; `polytope_at` / `vertices_at` / `volume_at` /
  `is_empty_at` / `contains_at` cache per sample; `clear_cache`.
- `Polytope::volume` works in **any dimension** (was `≤ 3`): exact facet
  decomposition around the vertex centroid, recursing on each facet's
  exact `(n − 1)`-dimensional H-representation; duplicate or rescaled
  facets are counted once.  Verified on hypercubes and simplices up to
  dimension 5 and the 4-D cross-polytope.
- `QMatrix::ldl_psd()` (exact `L·D·Lᵀ` of a PSD matrix, `None` if not
  PSD), `QMatrix::is_positive_semidefinite()`, `QMatrix::is_symmetric()`.

### Infrastructure

- `symplex` and `symplex-build` at 0.5.0; `symplex-macros` unchanged at
  0.3.0.  The three new `lean_steps` skeleton shapes (cast parameter with
  long names and a wrapped `have hg`, emptiness with `K` chains,
  `prefer_subtraction`) compile against Mathlib.

## [0.4.0] - 2026-09-18

### Breaking

- `Ex::roots_count_real` (the 0.3 alias) is removed; call
  `count_real_roots_in` on `Ex` or `Poly`.
- `lean::LeanOpts` gained the field `prefer_subtraction`.  Struct literals
  must add `..Default::default()` (`LeanOpts { real_type: "ℚ".into(),
  ..Default::default() }`) or use the new builders
  `with_real_type` / `with_ascribe_integers` / `with_prefer_subtraction`.
  Further fields may be added in minor releases.

### Added

- **Certificates on a parametric polyhedron**
  (`certificates::prove_nonnegative_on_polyhedron(goal, hyps, Some((&j,
  &j0)), &PolyhedronOpts)` and `prove_polyhedron_empty`): prove `g ≥ 0`
  on `{hₖ(j, x) ≥ 0}` for every real `j ≥ j₀` — or that the set is empty
  — by the exact identity `λ(j)·g = Σ μ·jᵃ(j − j₀)ᵇ·hₖ + Σ μ·jᵃ(j − j₀)ᵇ
  + μ₀` (optionally `+ Σ μ·hₖhₗ`), `λ(j) = 1 + Σ νₐ jᵃ`, all `μ, ν ≥ 0`.
  The polynomial multiplier `λ` on the goal is what makes `j`-dependent
  facets certifiable (their Farkas multipliers are rational functions of
  `j`).  Outcomes `Proved(PolyhedronCertificate)` / `Refuted { point,
  value }` (an exact point of the set, found by sampling `j` and
  minimising an affine goal with the exact LP) / `Unknown { degree,
  lambda_degree, pairwise }`.  The search is **staged** (degree-1
  multipliers with `λ = 1` first, then higher degrees, pairwise products
  last; `PolyhedronOpts::single` for one LP) and every certificate is
  re-verified with exact polynomial arithmetic.  `param = None` gives a
  plain Farkas / pairwise certificate on a fixed polyhedron.
  - `PolyhedronCertificate::{goal, hyps, parameter, terms, lambda,
    lambda_coeffs, lambda_is_one, degree, uses_pairwise,
    proves_emptiness, product_expr, identity, verify}`, `Display` with
    the hypotheses abbreviated `hₖ`.
  - Lean export in the shape a hand-written proof uses: `to_lean(name)`
    emits a theorem with `hj : j₀ ≤ j`, `h0 : 0 ≤ h₀`, …, derives `hJ0`
    / `hK0` when needed, one `have … := mul_nonneg …` per product
    (`h0K`, `h0JK`, `h0xh1`, `pJJ`), and closes with `linarith only […]`
    — through `have hg : 0 ≤ λ * g` and `nonneg_of_mul_nonneg_right`
    when `λ ≠ 1`; emptiness certificates conclude `False`.
    `lean_steps(&PolyhedronLeanNames { hyps, param_nonneg, shift_nonneg },
    &opts)` returns the same `have` lines, hint names and closing block
    (`PolyhedronLeanSteps`, `to_block(indent)`) for an existing proof
    skeleton.  Fifteen shapes (λ = 1; λ of degree 1 and 2 on `j` or on
    `j − j₀`; `j₀ > 0`, `j₀ = 0`, `j₀ < 0`; mixed `J`/`K` chains; pairwise
    products with and without a parameter; emptiness with and without `λ`;
    pure parameter powers `pJJ`, `pJK`, `pKK`; no parameter) compile
    against Mathlib (Lean 4.30.0) with
    `linter.style.longLine` on; the emitted text is pinned to that compiled
    file (`tests/fixtures/polyhedron_certificates.lean`).
- **`symplex::polytope`**: exact convex polyhedra in ℚⁿ from half-spaces
  `a·x + b ≥ 0`.  `Polytope::{new, from_rows, from_exprs, to_exprs,
  halfspaces, contains, with_halfspace, split, is_empty, any_point,
  bounding_box, is_bounded, vertices, irredundant, vertex_centroid,
  volume}` (vertices by `QMatrix::solve` over `n`-subsets, emptiness and
  bounds by the exact LP, volume for `n ≤ 3`); `HalfSpace::{value,
  contains, flipped}`.  Meant for the geometry around certificate
  searches (cells of a decision tree, cuts, containment), not for large
  polyhedra.
- `Poly::try_new(expr, gens) -> Result<Poly, SymplexError>`: `Poly::new`
  with the reason for failure — which generator sits inside a function,
  under a negative power (a rational function), under a fractional or
  symbolic power, or in an exponent.
- `Poly::terms_iter()` (borrowed `(&[u32], &Ex)` pairs, no allocation) and
  `Poly::coeffs_rational() -> Option<Vec<Ratio<BigInt>>>`.
- `LeanOpts::prefer_subtraction`: a sum with exactly one negated term is
  printed as a subtraction with that term last (`(1 / 2 : ℝ) - r` instead
  of `-r + (1 / 2 : ℝ)`), so generated hypotheses match hand-written ones
  syntactically; `with_real_type` / `with_ascribe_integers` /
  `with_prefer_subtraction` builders.
- `HalfLineCertificate::lean_hints(hk, &opts)`: the hint list of the Lean
  proof (`hk`, `pow_nonneg hk n`, `mul_nonneg (sq_nonneg g) (…)`) for a
  caller's proof skeleton, with the caller's name for `0 ≤ k`.
- Certificates cross trust boundaries: `Certificate`,
  `HalfLineCertificate` and `PolyhedronCertificate` have `to_data()` /
  `from_data(&ctx, &data)` (plain serde-derived structs
  `CertificateData`, `HalfLineCertificateData`,
  `PolyhedronCertificateData` built from `ExprTree`s and `"p/q"`
  rationals) and `to_json()` / `from_json(&ctx, json)`.  `from_*`
  **re-verifies** the identity with exact polynomial arithmetic and
  rejects data that does not hold, so a checker can accept a certificate
  produced elsewhere without trusting the producer.
- `linsolve` docs state explicitly that an over-determined but consistent
  system is `Unique` (with an example), a contradictory one
  `Inconsistent`.
- `examples/polyhedron_certificates.rs`; test group `tests/v04.rs`
  (`v04_polyhedron`, `v04_polytope`); book: "What's New in 0.4",
  "Migrating from 0.3 to 0.4", a parametric-polyhedra section in the
  certificates cookbook and a polytope section in the LP guide.

### Infrastructure

- `symplex` and `symplex-build` at 0.4.0; `symplex-macros` unchanged at
  0.3.0.  `cargo-semver-checks --release-type minor` against 0.3.5 reports
  exactly the two breaking changes listed above.

## [0.3.5] - 2026-09-18

### Added

- **Exact matrix core.**  `matrix::{QMatrix, ZMatrix}` (aliases of
  `ExactMatrix<Ratio<BigInt>>` / `ExactMatrix<BigInt>`, both in the
  prelude): dense row-major matrices with no expression arena behind
  them.  Construction (`new`, `from_i64`, `from_fn`, `from_flat`, `zeros`,
  `identity`, `diag`, `row_vector`, `col_vector`), access (`get`,
  `try_get`, `row`, `col`, `diagonal`, `rows`, `iter`, `as_slice`,
  `to_rows`, `into_flat`, indexing), shape ops (`transpose`, `submatrix`,
  `hstack`, `vstack`, `map`), arithmetic (`add`, `sub`, `neg`, `scale`,
  `matmul`, `trace`, operators `+ − *`), `is_zero`, `is_identity`, and
  `Display`/`Debug` in the `Matrix` layout.
  - `QMatrix`: `rref`, `rank`, `nullspace`, `columnspace`, `rowspace`,
    `det`, `inv`, `solve` (square, multiple right-hand sides),
    `clear_denominators`, `to_zmatrix`, `is_integer`, `to_matrix`.  Every
    elimination is **fraction-free** (Bareiss Gauss–Jordan on the
    row-wise integerised matrix): intermediate entries are minors of the
    input, all divisions are exact, and no gcd runs in the inner loop.
  - `ZMatrix`: Bareiss `det`, `rank`, `content`,
    `hermite_normal_form[_with_transform]`, `column_hermite_normal_form`,
    `smith_normal_form[_with_transforms]`, `integer_nullspace`,
    `is_unimodular`, `lattice_determinant`, `to_qmatrix`, `to_matrix`.
  - Conversions: `TryFrom<&Matrix>` for both (constant arithmetic is
    folded first; a symbolic entry is `InvalidArgument`), `From<ZMatrix>
    for QMatrix`.
- `examples/exact_matrices.rs`, `benches/exact_matrix.rs`; book: a
  "0.3.5: the exact matrix core" section on the What's New page, a new
  section in the Matrices guide, performance notes in the LP and lattice
  guides.

### Changed

- `Matrix::{rref, rank, nullspace, columnspace, rowspace, left_nullspace,
  det, inv, solve, solve_least_squares, pinv}`, `linsolve`,
  `linsolve_matrix` and every function in `normalforms` now detect
  all-rational input and run on `QMatrix`/`ZMatrix`, converting back at
  the end.  Results are unchanged (the RREF is unique; parametric
  `linsolve` solutions go through the same `tidy` step and print
  identically); a 30×30 rational `inv` drops from 470 ms to 10 ms, `rref`
  of a 30×36 from 208 ms to 4 ms, `linsolve_matrix` 30×30 from 135 ms to
  1.5 ms.  The `Ex`-based Bareiss determinant that only served numeric
  matrices is gone; symbolic matrices take the same paths as before.
- `linprog`: the simplex tableau pivots on **integers with a common
  denominator** (Bareiss/Edmonds integer pivoting).  Each constraint row
  is scaled once to clear denominators (its artificial gets phase-1 cost
  `1/sᵢ` and the scale is divided back out of the duals and Farkas
  vectors); every pivot keeps the tableau integral; ratio and sign tests
  are cross-multiplied integer comparisons with no gcd in the loop.  The
  entering/leaving choices are made on the same rational values as
  before, so the pivot sequence is the same: on 4,000 random LPs with
  fractional data, degenerate rows, all three relations, free and
  two-sided-bounded variables, `x`, objective, duals and Farkas vectors
  are byte-identical to 0.3.4.  A 40-row × 100-variable program goes from
  1.1 s to 40 ms, 60 × 160 from 2.8 s to 80 ms, and Handelman certificate
  searches run 4–7× faster.
- `normalforms` is now a thin wrapper over `ZMatrix`; error messages and
  conventions are unchanged.

### Infrastructure

- Decision recorded after benchmarking `num-bigint 0.4` against `dashu
  0.6` on the exact-linear-algebra kernels: dashu is ~9× faster on
  Gauss–Jordan over `Ratio` but only 1.2–2× on integer kernels — the gap
  is `Ratio`'s per-operation gcd, not bignum speed.  Fraction-free
  elimination on `num-bigint` beats dashu's rational elimination by 5×
  and the previous code by 40×, so `num-bigint` stays and the public
  `Ratio<BigInt>` types are untouched.
- `tests/v03/v03_exact_matrix.rs`: the exact core against textbook
  Gauss–Jordan, the `Matrix` fast paths against the core, the numeric
  `linsolve` route against the symbolic one, random fractional LPs against
  the exact KKT conditions, Farkas certificates on fractional data, and a
  40×40 rational inverse.  A white-box `linprog` unit test covers the
  tableau's negative common denominator after an artificial is driven out
  on a negative pivot.
- `symplex` and `symplex-build` at 0.3.5; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.4.

## [0.3.4] - 2026-09-18

### Added

- Sign facts for univariate polynomials with rational coefficients in a
  real-assumed symbol are now decided exactly (square-free factoring +
  Sturm sequences) when the structural assumption rules are silent:
  `(3u² + 2u + 1).is_positive()` is `Some(true)` for `u ≥ 0`,
  `(x² − 2x + 2).is_positive()` is `Some(true)` for real `x`,
  `(x − 1)²` is `is_nonnegative() == Some(true)` but `is_positive() ==
  None`, `x² − 1` stays `None`.  The symbol's own sign assumptions
  (`Positive`, `NonNegative`, `Negative`, `NonPositive`) restrict the
  domain; an excluded endpoint may be a root (`p² + p > 0` for `p > 0`).
  Everything downstream benefits: `simplify` drops `abs(·)` and folds
  `sqrt(p²)`/`sign(p)`, `ln(p).is_real()`, `compare_numeric`, `BoolEx::eval`
  of `p > 0`.  Guarded to degree ≤ 24; symbols without a `Real` assumption
  (possibly complex) are untouched.
- `lean::wrap_lean(text, width)` and `lean::MATHLIB_LINE_WIDTH`: re-flow
  Lean source at spaces, preferring breaks after commas (hint lists) and
  between binders (signatures), with Lean-compatible continuation
  indentation.  All certificate emitters (`Certificate`,
  `HalfLineCertificate`, `RealLineCertificate`) now produce output that
  passes Mathlib's `linter.style.longLine` — verified by compiling the
  wrapped output with the linter enabled.  `Ex::to_lean` stays single-line
  for embedding.

### Infrastructure

- `symplex` and `symplex-build` at 0.3.4; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.3.

## [0.3.3] - 2026-09-18

### Added

- `certificates::prove_nonnegative_on_halfline(goal, var, a, Ray::{AtLeast,
  AtMost}, max_polya_power)`: exact certificates for univariate `goal ≥ 0`
  on `x ≥ a` / `x ≤ a`.  With `k = x − a` the identity is
  `(1 + k)^N · goal = g² · Σ cᵢ kⁱ` with `cᵢ ≥ 0`: `N = 0` is the
  shift-and-read-off-coefficients certificate, `N > 0` a Pólya multiplier
  (always exists for a strictly positive goal), and `g` collects
  even-multiplicity zeros.  Outcomes: `Proved(HalfLineCertificate)`,
  `Refuted { point, value }` (exact, found via root isolation), or
  `Unknown { max_polya_power }`.  `HalfLineCertificate::{verify, identity,
  coefficients, polya_power, square, to_lean}`.
- `certificates::prove_nonnegative_on_reals(goal, var, split, max_polya)`
  → `RealLineCertificate` (two half-lines; Lean proof by
  `rcases le_total split x`).
- Box certificates now handle interior even-multiplicity zeros: when the
  plain Handelman search fails, the goal is factored as `g²·h` (exact
  factoring over ℤ) and `h` is certified; `Certificate::square()` exposes
  `g` and the Lean hints become `mul_nonneg (sq_nonneg g) (…)`.
  `(x − 1/2)²·(1 − xy)` on the unit square is now `Proved`.
- Every Lean theorem emitted by `examples/certificates_to_lean.rs`
  (9 theorems: boxes, half-lines both directions, Pólya exponents 1 and
  12, square factors, the real line) was compiled against Mathlib
  (Lean 4.30.0) with no errors or warnings; the texts are pinned in
  `tests/v03/v03_certificates.rs`.

### Fixed

- `(c·m)^n` with a numeric coefficient `c` and `|n| > 10` was left as an
  opaque power by canonicalisation (the product-power distribution has a
  swell guard at 10), so `(-x)^11` did not become `-x^11`,
  `expand((2 - x)^11)` lost its leading term and `Poly::new` rejected the
  result.  A numeric coefficient is now always pulled out (`(-x)^11 →
  -x^11`, `(2x)^13 → 8192*x^13`); products of symbols keep the guard
  (`(x*y)^12` stays as is) but the polynomial view now recognises such
  powers as monomials (`Poly::new((x*y)^12 + 1)` works).

### Infrastructure

- `symplex` and `symplex-build` at 0.3.3; `symplex-macros` unchanged at
  0.3.0.

## [0.3.2] - 2026-09-18

### Added

- `symplex::certificates` — exact, machine-checkable non-negativity
  certificates on a box.  `prove_nonnegative_on_box(goal, &[(var, lo, hi)],
  degree)` searches for a Handelman certificate
  `goal = Σ λₖ · Π (xᵢ − lᵢ)^a (uᵢ − xᵢ)^b`, `λ ≥ 0`, by exact LP (weighted to
  prefer few, low-degree products), **re-verifies the identity with exact
  `Poly` arithmetic** before returning, and otherwise either refutes the
  claim with an exact counterexample (`BoxOutcome::Refuted { point, value }`)
  or reports `BoxOutcome::Unknown { farkas, degree }` (e.g. when the goal
  touches zero inside the box, where no Handelman certificate exists).
  `Certificate::{terms, product, product_expr, identity, verify, degree}`,
  `is_nonnegative_on_box` (three-valued convenience),
  `Poly::express_as_nonneg_combination(basis)` (the LP step alone), and
  `Ex::prove_nonnegative_on_box`.
- `Certificate::to_lean(name)` / `to_lean_with`: a Lean 4 / Mathlib theorem
  `theorem name (x y : ℝ) (h_x_lo : …) … : 0 ≤ goal := by nlinarith […]`
  whose hints are exactly the certificate's products (`mul_nonneg
  (sub_nonneg.mpr h_x_lo) (sub_nonneg.mpr h_y_hi)`, …); linear certificates
  use `linarith`; unused bounds are underscored for the linter.  Every
  theorem produced by `examples/certificates_to_lean.rs` was compiled
  against Mathlib (Lean 4.30.0) with no errors or warnings.

### Infrastructure

- `symplex` and `symplex-build` at 0.3.2; `symplex-macros` unchanged at
  0.3.0.  Additive over 0.3.1 (`cargo-semver-checks`: 223/223).

## [0.3.1] - 2026-09-18

Driven by field notes from downstream tools built on 0.3.0.  Additive only
(verified with `cargo-semver-checks` against 0.3.0).

### Added

- The exact-arithmetic crates are re-exported — `symplex::num_bigint`,
  `num_rational`, `num_integer`, `num_traits` — so a downstream crate can
  name `Ratio<BigInt>` (from `as_rational`, `linprog::Q`, `Matrix::from_ratio`,
  …) without adding and version-matching those crates itself.
- `Ex::as_ratio_parts() -> Option<(BigInt, BigInt)>` and
  `Ex::as_ratio_i128() -> Option<(i128, i128)>`: numerator and denominator of
  a rational literal (SymPy's `Rational.p` / `.q`).
- `Poly::{count_real_roots, count_real_roots_in(lo, hi), real_roots_isolate,
  is_nonnegative_on(lo, hi), is_positive_on(lo, hi)}` — the Sturm-based
  primitives that were only reachable through `Ex` — and `Poly::shift(gen, a)`
  (Taylor shift; "all coefficients of `p(k + a)` are `≥ 0`" is the
  certificate-style sufficient condition for `p ≥ 0` on `[a, ∞)`).
- `Ex::count_real_roots_in(var, lo, hi)`; the old name `roots_count_real`
  stays as an alias (no deprecation warning in a patch release) and is
  removed in 0.4.
- `Ex::to_lean()` / `to_lean_with(&LeanOpts)` (`symplex::lean`): Lean 4 /
  Mathlib rendering with Mathlib spacing (`2 * j + 1`, `j ^ 2`), ascribed
  rational literals (`(3 / 31 : ℝ)`), single-fraction division
  (`(j - 1) / (2 * j)`), `⁻¹` for negative powers, `Real.sin`/`Real.exp`/
  `Real.sqrt`/`Real.pi`/`|x|`/`⌊x⌋`, relations and connectives for `BoolEx`
  (`0 < x ∧ x < 1`), `if … then … else` for `Piecewise`.  Nodes without a
  standard Mathlib spelling are `Err(NotImplemented)`.

### Behaviour changes

- `Ex::as_numer_denom` follows SymPy: a rational literal splits into
  integers (`3/31` → `(3, 31)`), a rational coefficient splits
  (`2/3·x` → `(2*x, 3)`), and sums are combined over a common denominator at
  every depth (`x/2 + 1/3` → `(3*x + 2, 6)`, `1/x + 1/y` → `(x + y, x*y)`).
  Previously a rational literal was an atom (`(3/31, 1)`) and a sum was
  returned whole.  Still no cancellation (`ratsimp` does that).
- `Ex::together` is deep: fractions nested inside numerators, denominators,
  products and integer powers are flattened into one quotient.  Previously
  only a top-level sum was combined, so `Poly::new` on the numerator of a
  `together()` result could silently see a rational function.

### Fixed

- `eval` was not idempotent on `exp(f)^g`: `(1/exp(-1)).eval()` gave
  `exp(1)`, and only a second `eval` gave `E`.  The rewritten exponent is
  now evaluated in the same pass.
- `laplace_final_value` located poles of `s·F(s)` without cancelling the
  factor `s`, which with the deep `together` would have reported a spurious
  pole at the origin for `F(s) = 3/s − 2/(s + 1)`; it now uses `ratsimp`.

### Infrastructure

- The ~275 integration-test source files are compiled into nine test
  binaries (`tests/{v03,v03_oracle,v02,v02_oracle,unit,legacy,proptests,
  perf}.rs` plus `ui_tests`); every test keeps its name as
  `<module>::<test>`.  Linking dropped from ~6.5 min / 13 GB to seconds, and
  the CI disk-space workaround is gone.  See `tests/README.md` for the
  layout and the `cargo test --test <group> <module>::` / `cargo nextest run
  -E …` invocations.
- `.config/nextest.toml`: `cargo nextest run` executes one process per test
  with per-test wall-clock limits (`default` and `ci` profiles).
- `deny.toml` + a `cargo deny check` CI job enforce the pure-Rust dependency
  policy (no C/C++ or system libraries), a licence allow-list, advisories and
  registry sources.
- Removed the assertion-free `tests/zz_probe_tmp.rs` left over from 0.2.
- `symplex-wasm`: dropped the unused `web-sys` dependency.
- `symplex` and `symplex-build` at 0.3.1; `symplex-macros` is unchanged at
  0.3.0.

## [0.3.0] - 2026-09-18

**Polynomials as data, exact certificates.**  This release makes the
polynomial structure of an expression a first-class object (`Poly`: sparse
terms over explicit generators with symbolic coefficients), gives rational
functions a real normal form (`ratsimp`), and adds three exact
certificate-producing domains: linear programming over ℚ with shadow prices
and Farkas infeasibility vectors, Hermite/Smith normal forms of integer
matrices with unimodular transforms and ℤ-bases of integer kernels, and
Sturm-verified polynomial signs on intervals.  A deterministic `f64`
optimisation toolbox (Brent, Nelder–Mead, differential evolution,
least-squares fits) rounds out the numeric side.  There are no
signature-breaking changes; the behaviour changes below alter the *form* of
some results, never their value.

Measured at release: 92 `ExprNode` variants (unchanged), ~11,000 `#[test]`
functions (~154K lines of tests, 273 integration-test files), ~174K lines in
`src/`, ~600 doctests in the main crate, and a SymPy 1.14 oracle extended
with 455 fixtures for the 0.3 features
(`tests/fixtures/v03_cross_validation.json`).

### Behaviour changes

Not breaking — no signature changed — but results may print differently.

- `Ex::{degree, coeffs, coeff, leading_coeff, is_polynomial}` accept
  symbolic, variable-free coefficients: `(a·x² + x).degree(&x)` is `Some(2)`
  and `coeffs` is `[0, 1, a]` where 0.2 returned `None`.  Rational-coefficient
  results are byte-for-byte unchanged.
- `Ex::solve` on linear and quadratic equations with parametric coefficients
  puts the root (and the quadratic discriminant) into rational normal form:
  `((3r−1)/(j+1) − (r+1)/(2j)).solve(&r)` is `(3*j + 1)/(5*j - 1)` instead of
  a fraction of fractions.
- `Ex::simplify_rational` is now `ratsimp` (one fraction over all variables
  at once, integer-primitive numerator and denominator, positive leading
  denominator coefficient) instead of `together` followed by a per-symbol
  `cancel`.  Same value; nested fractions that 0.2 left uncancelled are
  now cancelled.
- `Display`: a product with a rational coefficient *and* inverse factors is
  printed as one fraction.  `1/2*1/j*(j - 1)` is now `(j - 1)/(2*j)`,
  `4/3*1/pi*sin(3*x)` is `4*sin(3*x)/(3*pi)`, `-1/2*1/(x + 1)` is
  `-1/(2*(x + 1))`, and `x - 3*y*1/z` is `x - 3*y/z`.  Products without an
  inverse factor are unchanged (`1/2*x`), as is `x^(-2)`.

### Added

**Polynomials**

- `poly_ex::Poly` (also `prelude::Poly`) and `Ex::as_poly(&[&gens])`: an
  expression as a sparse polynomial in explicit generators with exact
  rational *or* symbolic coefficients; terms in SymPy's lex-descending
  order.  `Poly::{new, from_terms, zero, one, constant, from_multipoly}`;
  queries `gens`, `num_gens`, `is_zero/is_ground/is_univariate/is_linear/
  is_homogeneous`, `has_rational_coeffs`, `num_terms`, `terms`, `monoms`,
  `coeffs`, `coeff_monomial`, `total_degree`, `degree_in`, `degree_list`,
  `leading_term/leading_coeff/leading_monomial`, `all_coeffs`, `equals`;
  conversion `to_ex`, `to_multipoly`; evaluation `eval` (all generators)
  and `eval_gen` (partial, generator removed); arithmetic `add`, `sub`,
  `mul`, `neg`, `scale`, `pow`, `derivative`, `content_and_primitive`,
  `monic`; `nroots`; `Poly::monomial_basis` and `Poly::coefficient_matrix`
  for turning "goal = Σ λᵢ pᵢ" into an exact linear system; `Display` as
  `Poly(expr, gens…)`.
- `Ex::poly_is_nonnegative_on` / `poly_is_positive_on(var, lo, hi)`: exact
  three-valued sign of a rational-coefficient polynomial on a closed
  interval (square-free part isolates odd-multiplicity roots, Sturm count,
  one sign sample); endpoints may be `±∞`.
- `MultiPoly::{gcd, lcm}` (heuristic GCD, GCDHEU, verified by exact
  division), `integer_content`, `clear_denominators`, `from_terms`,
  `coeff`, `map_coeffs`.

**Simplification**

- `Ex::ratsimp`: rational-function normal form.  `P/Q` over the free
  symbols with every maximal non-rational subexpression (`sin x`, `π`,
  `√x`) treated as an independent indeterminate, `gcd(P, Q)` divided out,
  denominators cleared to integer-primitive parts, leading coefficient of
  `Q` positive.  Idempotent; expressions with `±∞`, `NaN` or unevaluated
  nodes are returned unchanged.

**Linear programming** (`symplex::linprog`)

- `LpProblem` builder (`minimize`/`maximize`, `le`/`ge`/`eq` rows,
  per-variable `bounds`, `free`) and `LpProblem::solve` → `LpSolution
  { status, x, objective, duals, farkas }` with `LpStatus::{Optimal,
  Infeasible, Unbounded}`; `LpSolution::{is_optimal, x_ex}`.  Two-phase
  dense simplex over `Ratio<BigInt>`, Dantzig pivots switching to Bland's
  rule at the first degenerate step (no cycling), pivot cap reported as
  `ComputationFailed`.
- Exact **shadow prices** (`duals`, one per constraint in insertion order,
  `yᵢ = ∂ optimum / ∂ bᵢ`) and exact **Farkas certificates** (`farkas`,
  `Aᵀy`-based inequality proving infeasibility; `None` only when the bounds
  alone are contradictory).  Sign conventions documented in the module docs.
- `linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)` (SciPy-shaped),
  `feasible_nonneg(A, b)` ("is there `x ≥ 0` with `Ax = b`?", exactly),
  `feasible_nonneg_certified` and the column-oriented `nonneg_combination
  (vectors, target)` returning `Feasibility::{Feasible(x), Infeasible {
  farkas }}`, `linprog_matrix(Objective, &c, A_ub, b_ub, A_eq, b_eq)` on
  `Matrix` data with numeric-literal entries, `LpSolution::duals_ex`, a
  one-line `Display` for `LpSolution`, and the literal helpers `q(n, d)`,
  `qi(n)`.
- `prelude` re-exports `LpProblem`, `LpSolution`, `LpStatus`.

**Integer normal forms** (`symplex::normalforms`)

- `hermite_normal_form` (row style, `H = U·A`, unique: positive pivots,
  entries above pivots reduced into `[0, pivot)`, zero rows last),
  `hermite_normal_form_with_transform` → `(H, U)`,
  `column_hermite_normal_form` (`H = A·V`, SymPy / Cohen 2.4.5 convention,
  leading zero columns kept).
- `smith_normal_form` (`diag(d₁, …, dᵣ, 0, …)`, `dᵢ | dᵢ₊₁`) and
  `smith_normal_form_with_transforms` → `(S, U, V)`.
- `integer_nullspace` (a ℤ-basis of `{x ∈ ℤⁿ : Ax = 0}` — generates every
  integer solution, unlike a scaled rational nullspace), `is_unimodular`,
  `lattice_determinant` (index of the column lattice in `ℤᵐ`).
- `Matrix::{hermite_normal_form, smith_normal_form, integer_nullspace}`
  method forms.  Non-integer entries are `InvalidArgument`.

**Matrices**

- Selection: `extract(rows, cols)`, `select_rows`, `select_cols`,
  `delete_row`, `delete_col` (index lists may repeat or reorder; empty or
  out-of-range is `InvalidArgument`).
- Three-valued structure test `is_integer_matrix` (alongside the existing
  `is_zero`).
- Exact numeric conversion: `to_rational_rows`, `to_bigint_rows`,
  `Matrix::from_ratio`, `Matrix::from_bigint`, `Matrix::from_f64_rows`
  (exact dyadic; `NaN`/`∞` rejected).
- `subs_map(&[(&from, &to)])` (simultaneous substitution in every entry),
  `nnz`.

**Number theory**

- `ntheory::{gcd_many, lcm_many}` on `&[BigInt]` (empty list: `0` / `1`;
  early exit at gcd 1; any zero makes the lcm 0), generic `igcd` / `ilcm`
  for any `Into<BigInt>`, and `rational_lcm_of_denominators`.

**Numerical optimisation** (`symplex::optimize`)

- Root finding: `brent_root` (Brent–Dekker), `bisect`, `newton_root`, with
  `RootOpts { xtol, rtol, max_iter }`.
- Minimisation: `nelder_mead`, `minimize_scalar` (Brent), `golden_section`,
  `differential_evolution` (DE/rand/1/bin, Latin-hypercube start,
  Nelder–Mead polish, SplitMix64 seeded by `DeOpts::seed` — fully
  deterministic), with `MinimizeOpts`, `DeOpts` and `MinimizeResult { x,
  fun, iterations, evaluations, converged }`.  An exhausted budget is
  reported through `converged = false`, never by discarding the best point.
- Fitting and helpers: `poly_fit` (column-scaled Householder QR, ascending
  coefficients), `poly_fit_exact` (rational normal equations),
  `linear_fit`, `trapezoid`, `eval_poly`.
- On `Ex`, compiling first: `find_root_bracket[_with]`,
  `minimize_numeric[_with]`, `minimize_scalar_numeric`,
  `minimize_global_numeric`, and `Ex::poly_fit_points` (exact least-squares
  polynomial through rational points).  A stray free symbol is
  `FreeSymbol`, not a silent `NaN`.
- `prelude` re-exports `RootOpts`, `MinimizeOpts`, `MinimizeResult`.

**Examples and tests**

- New examples `polynomials`, `exact_lp`, `integer_lattices`,
  `numeric_optimization`; `readme_snippets` mirrors the new README sections.
- The crate docs list the new modules (`poly_ex`, `linprog`,
  `normalforms`, `optimize`) in the module map.

### Fixed

- `simplify_rational` could not cancel common factors that only appear
  after combining (`(x² − y²)/(x − y)` was returned unchanged) or
  fractions nested inside fractions (`1/(x + 1/y) + 1/(1/x + y)` became
  `(x + 1/x + y + 1/y)/((x + 1/y)*(1/x + y))`); it now returns `x + y` and
  `(x + y)/(x*y + 1)`.
- `solve` returned unsimplified nested fractions for linear and quadratic
  equations whose coefficients are themselves parametric fractions
  (`(1/2*1/j + 1/(j + 1))/(-1/2*1/j + 3/(j + 1))` for the example above).
- `degree`/`coeffs`/`coeff`/`leading_coeff`/`is_polynomial` reported
  "not a polynomial" (`None`/`false`) for polynomials with symbolic
  parameter coefficients such as `a·x² + (a + 1)·x + 3`.

### Infrastructure

- `tests/v03_*` integration suites, one concept per test: `v03_poly_view`,
  `v03_poly_symbolic_coeffs`, `v03_ratsimp`, `v03_linprog` (every `Optimal`
  result checked against the full KKT conditions exactly, every
  `Infeasible` result's Farkas vector verified), `v03_normalforms` (defining
  invariants `H = U·A`, `|det U| = 1`, `S = U·A·V`, divisibility chain,
  `A·k = 0` checked rather than pinned answers), `v03_matrix_ergonomics`,
  `v03_optimize`.
- SymPy 1.14 oracle extended to the 0.3 features (`Poly.terms`/`coeffs`,
  `cancel`/`ratsimp`, `sympy.solvers.simplex`, `hermite_normal_form`,
  `smith_normal_form`) via `tests/fixtures/v03_cross_validation.json`;
  as before, comparisons are numeric or structural, never by printed form.
- Crate, `symplex-macros`, `symplex-build` and `symplex-wasm` at 0.3.0.

## [0.2.0] - 2026-09-18

A large release: complex analysis, definite/improper/numeric integration,
symbolic summation, a public rewrite-rule engine, sets and logic, a C99
backend, linear-system and general solvers, ODE initial-value problems,
recurrences, Berlekamp–Zassenhaus factoring, and a new `factorint`.  The
guiding rule for every change below is *never silently wrong*: operations
that used to guess now return `Err`, an unevaluated node, or `None`.

Measured at release: 91 `ExprNode` variants, ~10,600 `#[test]` functions
(~148K lines of tests), ~167K lines in `src/`, 520 doctests, and a
1,400-fixture SymPy 1.14 oracle with zero numerical disagreements.

### Breaking

**Core / API model**

- `Ex::compile(&[&str])` returns `Result<CompiledFn, SymplexError>` instead of
  `Option`.  `CompiledFn` is `Clone + Send + Sync`, callable as a closure,
  and has `arity()` and `try_call()` (arity-checked).
- `Ex::definite_integral` is replaced by `Ex::integrate_definite` (returns a
  bounded, unevaluated `DefiniteIntegral` node — displayed
  `Integral(f, x, a, b)` — when it cannot decide) and
  `Ex::try_integrate_definite` (`Err(Divergent)` / `Err(ComputationFailed)`).
  The old method could return a wrong finite number for `∫₋₁¹ dx/x²`.
- `Ex::solve` semantics: identities (`x − x = 0`) return
  `Err(InfiniteSolutions)`; contradictions and range violations (`sin x = 2`,
  `eˣ = −1`, `|x| = −1`) return `Err(NoSolution)`.  Results are `eval`'d
  (`asin(1/2)` → `π/6`).
- `Context::solve_system` returns `Result<LinearSolution>` with
  `Unique` / `Parametric { solution, free }` / `Inconsistent` instead of a
  flat vector.
- `polysys::solve_system_ex` returns algebraic (radical) solutions, not only
  rational ones, and `Err(InfiniteSolutions)` for positive-dimensional systems.
- `has_unevaluated()` no longer counts `RootOf` / `RootSum`: they are complete
  algebraic answers.  `try_*` methods therefore succeed on degree ≥ 5 roots.
- `Ex::equals` may now return `Some(false)` (previously only `Some(true)`/`None`).
- `Debug for Ex` prints the expression (`Ex(x^2 + 1)`) instead of internal ids.
- `re`, `im`, `conjugate`, `arg` of a symbol without a realness assumption
  return unevaluated `re(z)`, `im(z)`, `conjugate(z)`, `arg(z)` nodes.  0.1
  silently assumed every symbol real.
- `Digamma(n)` folds for positive integers and half-integers
  (`ψ(1) = −γ`); `d/dx digamma(x)` is `polygamma(1, x)` (was a formal
  derivative).
- `expand()` no longer splits `(x·y)^a` for symbols of unknown sign (unsound
  over ℂ).  Use `expand_power_base(force)` to opt in.
- `Ex::differentiate_finite(var, points, order)` replaces the previous
  finite-difference signature; `finite_diff::{finite_diff_weights,
  apply_finite_diff, …}` are `Ex`-based.
- `FormalPowerSeries` is `Ex`-based (`from_coefficients`, `coefficient`,
  `general_term`, arithmetic) instead of `Ratio`-based.
- `std::iter::Sum` / `Product` for `Ex` **panic on an empty iterator** (there
  is no context to build `0`/`1` in).  Use `Context::sum` / `Context::product`
  or collect into `Option<Ex>`.
- Plotting methods (`plot_data`, `textplot`, `to_svg`, `to_tikz`, `eval_table`)
  return `Result` instead of panicking or producing empty output.
- `SimplifyOpts::trace` is honoured; use `Ex::simplify_traced` to obtain the
  steps.
- `Assumption` gained `ExtendedReal` and the negated variants (`NotPositive`,
  `NotZero`, …); `match` statements on it need updating.

**Matrices**

- `Matrix::{eigenvals, eigenvects, diagonalize, jordan_form, matrix_exp}` take
  no dummy variable; the eigenvalue symbol is internal.  Use `char_poly(&λ)`
  when you want a named variable.
- `Matrix::is_diagonalizable` and `Matrix::is_symmetric` return `Option<bool>`.
- `Matrix::cholesky` and `Matrix::lu` return `Result` (`InvalidArgument` for a
  non-symmetric / non-square matrix, `ComputationFailed` when not positive
  definite).
- `Matrix::minor(i, j)` returns `Result<Ex>` (the determinant of the minor);
  the sub-matrix is `Matrix::minor_matrix(i, j)`.
- `Matrix::from_i64(ctx, rows)` returns `Result` (ragged rows are an error).
- Removed: `add_elementwise`, `sub_elementwise`, `try_identity`, `try_zeros`
  (use `add`/`sub`/`identity`/`zeros`).
- `StateSpace::poles()` takes no variable.
- `vector::{is_conservative, is_irrotational, is_solenoidal}` return
  `Option<bool>` (three-valued) instead of `bool`.

**Sets & logic**

- `SetEx::contains(&elem)` is set membership returning `Option<bool>`
  (previously structural containment).
- `BoolEx::eval` folds relations through the assumption system
  (`pos > 0` → `True` for a positive-assumed symbol).

**Fixed behaviour that may change results**

- `fourier_series` (and the new `fourier_series_on`) returns correct closed
  forms for `|x|`, `sign(x)` and piecewise inputs (coefficients are exact
  definite integrals).
- One-sided limits fall back to a two-sided `Limit` node when unevaluated.
- Factoring is no longer limited by `MAX_KRONECKER_DEGREE`: `factor` uses
  Berlekamp–Zassenhaus and handles any degree.
- `OdeType` gained `NthOrderLinearConstCoeff`, `Clairaut`, `Riccati`,
  `HomogeneousCoefficient`, `IntegratingFactor` (exhaustive matches break).

### Added

**Core nodes and constants**

- Complex analysis: `Re`, `Im`, `Conjugate`, `Arg` nodes with
  `Ex::{re, im, conjugate, arg, as_real_imag, expand_complex, polar,
  abs_squared, is_real_valued}`; conjugation distributes over `Add`/`Mul`/
  integer powers and commutes with real-analytic functions at construction.
- Constants `EulerGamma`, `Catalan`, `GoldenRatio`
  (`Context::{euler_gamma, catalan, golden_ratio}`) and
  `Context::complex_infinity()` (`zoo`; `1/0` evaluates to it).
- Special functions `si`, `ci`, `ei`, `li`, `zeta`, `polygamma(n, x)`,
  `kronecker_delta(i, j)` with exact values (`ζ(2m)`, `ζ(0)`, `ζ(−n)`,
  `ψ⁽ⁿ⁾(1)`, `Si(∞)`), derivative rules and arbitrary-precision `evalf`.
- `evalf` for `besseli` / `besselk` and orthogonal polynomials of any degree;
  derivative rules for Bessel functions and orthogonal polynomials.
- The parser accepts the new names (`re`, `im`, `conjugate`/`conj`, `arg`,
  `si`, `ci`, `ei`, `li`, `zeta`, `polygamma`, `kronecker`, `zoo`).
- `ExprNode::DefiniteIntegral(body, var, lo, hi)`: a bounded unevaluated
  integral (`Integral(f, x, a, b)`, LaTeX `\int_a^b f\,dx`).  It round-trips
  through Display/parse/JSON, binds its variable for `free_symbols`/`subs`,
  differentiates by the Leibniz rule, evaluates numerically via Gauss–Kronrod
  quadrature in `eval_f64`, and is resolved innermost-out by
  `integrate_definite` / `Ex::eval_integrals`.  Also `Ex::definite_integral_node`,
  `Ex::is_definite_integral`.

**Numeric backends**

- `compile()` covers every numerically evaluable node: Γ, lnΓ, ψ, erf/erfc,
  Lambert W, Beta, factorials and binomials, Bessel J/Y/I/K, orthogonal
  polynomials, integer sequences (`fibonacci`, `lucas`, `harmonic`, …),
  `min`/`max`/`floor`/`ceiling`/`sign`/`heaviside`/`atan2`, piecewise and
  boolean conditions.
- `Ex::compile_many` → `CompiledFnVec` (shared CSE across outputs;
  `call`, `call_vec`, `try_call`, `arity`, `len`).
- `to_rust_fn` embeds a self-contained `mod symplex_rt` runtime with only the
  special-function helpers the expression uses.
- `CodegenOptions::{use_mul_add, checked_domain, emit_runtime}` and
  `CodegenOptions::{runtime_module, c_runtime}` for multi-function files.
- **C99 backend**: `Ex::to_c_fn` / `to_c_fn_with_options`: `#include <math.h>`,
  `static inline symplex_*` helpers, `fma`, `float` precision with
  `f`-suffixed calls, `assert` domain checks, piecewise → ternary chains.
- Deterministic CSE (post-order numbering, cheap-node threshold, no boolean
  temporaries) and `Ex::cse_many`.

**Integration**

- `Ex::integrate_definite` / `try_integrate_definite`: interior
  singularities, infinite bounds, endpoint singularities via one-sided
  limits, symmetry shortcuts, `Piecewise` / `Abs` / `Sign` / `Heaviside` /
  `DiracDelta` integrands, and a ~30-entry table of classical improper
  integrals (Gaussian, Dirichlet, Fresnel, `x/(eˣ−1)`, `ln x`, Γ, …) with
  symbolic parameters under assumptions.
- `Ex::integrate_numeric` / `integrate_numeric_with` (adaptive Gauss–Kronrod
  G7/K15, `QuadOpts`, infinite bounds); `definite::quadrature` for plain
  `Fn(f64) -> f64`.
- Residues at poles of any order; `Ex::residue_at_infinity`.
- ~20 new indefinite-integration families.

**Summation and series**

- `Ex::summation` / `try_summation`, `product_over` / `try_product_over`,
  `hypergeometric_ratio`: Faulhaber sums of any degree, telescoping,
  binomial sums (`Σ P(k)·C(n,k)·xᵏ`), p-series (`ζ(2m)` exact, `zeta(p)` for
  odd `p`, Catalan's constant), power-series recognition (`Σ xᵏ/k! = eˣ`),
  Gosper with a polynomial-time normal form, infinite products.
- `Ex::is_convergent` / `is_absolutely_convergent` (decisive answers only).
- `Ex::series_at_infinity` / `series_at_neg_infinity`.
- `FormalPowerSeries`: lazy exact coefficients, `general_term`, `add`, `mul`,
  `compose`, `inverse`, `reversion`, `derivative`, `integral`.
- `Ex`-based finite differences (`finite_diff_weights`, `apply_finite_diff`,
  `equispaced_grid`, `Ex::differentiate_finite`).

**Solving**

- `Ex::solve_general` → `GeneralSolution` (periodic families with a fresh
  integer parameter, `instance(k)`).
- `linsolve`, `linsolve_matrix`, `LinearSolution`, `ZeroForm` (accepts `Ex`
  or `Equation`), symbolic coefficients, parametric solutions.
- Algebraic solutions in `polysys::solve_system_ex`.
- `solve_numeric_system` / `solve_numeric_system_with` (damped Newton,
  `NewtonOpts`).
- `Ex::solve_ode_ivp`, nth-order constant-coefficient ODEs, Clairaut,
  Riccati (`Ex::solve_riccati`), `ode::solve_ode_system_ivp`.
- `rsolve::rsolve_linear` / `rsolve_first_order` for recurrences.
- Inequalities with absolute values (`|x − 1| < 2`).

**Sets and logic**

- `SetEx::{simplify, eval, difference, symmetric_difference,
  absolute_complement, contains, is_subset, is_superset, is_disjoint,
  is_empty, inf, sup, measure, boundary, closure, interior, is_open,
  is_closed, as_intervals, as_finite_set, to_condition}`; `Ex::is_in`.
- `reduce_inequalities(&[BoolEx], &x) -> Result<SetEx>` and
  `BoolEx::solve_for`.
- `BoolEx::{simplify, to_nnf, to_cnf, to_dnf, is_tautology,
  is_contradiction, satisfiable, atoms, truth_table}` (DPLL with unit
  propagation; declared assumptions respected).
- `Ex::piecewise_simplify`.
- `Props::EXTENDED_REAL`, `Assumptions::implies`, `Assumption::negate`,
  `Display for Assumptions`.

**Matrices, vectors, quaternions, control**

- Eigen family without a dummy variable; `eigenvals_with_multiplicity`,
  `char_poly_coeffs`, `matrix_exp_t`, `matrix_pow_symbolic`, `matrix_sqrt`.
- `RootOf` eigenvalues for irreducible cubics/quartics without a compact
  radical form (exact, numerically evaluable, no Cardano swell);
  `EXPRESSION_BUDGET` swell guard.
- `qr`, `gram_schmidt`, `ldl`, `hessian`, `wronskian`, `adjoint`,
  `is_hermitian`, `is_orthogonal`, `is_unitary`, `is_positive_definite`,
  `is_positive_semidefinite`, `is_nilpotent`, `is_skew_symmetric`,
  `is_upper_triangular`, `is_lower_triangular`, `is_diagonal`, `is_identity`,
  `is_zero`, `norm_1`, `norm_inf`, `norm_p`, `solve_least_squares`,
  `rowspace`, `left_nullspace`.
- Ergonomics: `Index<(usize, usize)>` / `IndexMut`, `TryFrom<Vec<Vec<Ex>>>`,
  scalar operators on both sides (`2 * &m`, `&m * 2`, `m / 2`), `Neg`,
  `col`, `diagonal`, `submatrix`, `set`, `iter`, `to_vec`, `vec`, `eval_f64`,
  `equals`, `map_indexed`, `block_diag`, `hadamard`.
- `Quaternion`: arithmetic operators, `slerp`, `exp`/`ln`/`pow`,
  `rotate_vector`, `to_euler`/`from_euler`, `from_rotation_matrix`,
  `to_axis_angle`.
- `vector::CoordinateSystem` with cylindrical/spherical `gradient_in`,
  `divergence_in`, `curl_in`, `laplacian_in`; `directional_derivative`,
  `line_integral_scalar`, `line_integral_vector`, `scalar_potential`.
- `TransferFunction::to_state_space`, `StateSpace::to_transfer_function`.

**Number theory and polynomials**

- Berlekamp–Zassenhaus `factor` for any degree; multivariate `factor_all`
  (Kronecker substitution); `factor_list`, `factor_list_all`.
- `Ex` polynomial algebra: `resultant`, `discriminant`, `sqf_list`,
  `square_free_part`, `is_squarefree`, `is_irreducible`, `poly_div`,
  `poly_quo`, `poly_rem`, `poly_gcdex`, `decompose`, `content_primitive`,
  `leading_coeff`, `monic`, `poly_compose`, `poly_shift`, `poly_reverse`,
  `poly_interpolate`, `count_real_roots`, `roots_count_real`,
  `real_roots_isolate`, `nroots`.
- `ntheory::factorint` (Pollard–Brent rho with Montgomery `u128` arithmetic
  + ECM for `BigInt`), BPSW `isprime`, `is_probable_prime`,
  `jacobi_symbol`, `kronecker_symbol`, `is_quad_residue`, `sqrt_mod`,
  `sqrt_mod_all`, `discrete_log`, `n_order`/`multiplicative_order`,
  `primitive_root`, `is_primitive_root`, `primepi`, `prime`, `primerange`,
  `carmichael_lambda`, `perfect_power`, `is_mersenne_prime`,
  `continued_fraction`, `continued_fraction_periodic`,
  `continued_fraction_convergents`, `egyptian_fraction`, `digits`,
  `is_palindromic`.
- Sequences: `fibonacci`, `lucas`, `bernoulli`, `euler_number`, `harmonic`
  (ntheory); `bell`, `catalan`, `derangements`, `partitions` iterator
  (combinatorics); symbolic `Ex::{fibonacci, lucas, bell, catalan_number,
  bernoulli_number, euler_number, harmonic, partition_count}`.
- `diophantine::{linear_diophantine, linear_diophantine_n, pell,
  pell_solutions, pell_negative, sum_of_two_squares, sum_of_four_squares,
  pythagorean_triples, frobenius_number}`.

**Simplification and rules**

- Public rewrite-rule engine: `Rule` (template, guarded, closure RHS),
  `RuleSet`, `Bindings`, `RewriteOpts`, `RewriteStrategy`, `Step`;
  `Ex::{rewrite, rewrite_once, rewrite_traced, rewrite_with,
  rewrite_with_traced, simplify_with_rules, simplify_traced}`;
  `RuleSet::standard(&ctx)`.
- AC matching for `Add`/`Mul` with `rest__` sequence wildcards and a
  bounded backtracking budget; `rule!` macro rules usable via
  `Rule::from_macro_rule` / `RuleSet::from_macro_rules`.
- `sqrtdenest`, `signsimp`, `powdenest(force)`, `expand_with(ExpandOpts)`,
  `expand_power_base`, `expand_power_exp`, `expand_multinomial`,
  `log_combine_with`, `expand_log_with`, `nsimplify`,
  `nsimplify_with_constants`, `rcollect`, `collect_const`,
  `separate_vars_additive`, `separate_vars_dict`, `subs_algebraic`.
- 15 trig/hyperbolic identity rules; `vars!` macro (alias of `syms!`).

**Transforms and limits**

- `Ex::{limit_dir, limit_left, limit_right}` + `try_` twins and
  `Direction`; Gruntz work budget; many limits fixed or newly solved.
- `fourier_transform` / `fourier_transform_with` /
  `inverse_fourier_transform[_with]` with `FourierConvention`
  (non-unitary angular, unitary angular, ordinary).
- `mellin_transform` (returns the fundamental strip as a `BoolEx`) and
  `inverse_mellin_transform`.
- Laplace table extensions (`f(t)/t`, Bessel, `t^n e^{−at}`, …), inverse
  extensions (`1/√s`, shifted `e^{−as}F(s)`), `laplace_initial_value`,
  `laplace_final_value` (`Err(Divergent)` for unstable poles).
- `FourierSeries` with `fourier_series_on(var, lower, upper, n)`,
  `coefficient_a/b/c`, `truncate`, `omega0`.
- Z-transform table and inverse extensions.

**Ergonomics**

- `Context::{from_f64 (exact dyadic), from_f64_approx, from_f64_nice,
  from_bigint, from_ratio, from_i128, from_u64, rational_str, decimal_str,
  complex, symbols, symbols_indexed, apply, sum, product}`.
- Operators with `f64`, `i32`, `u32`, `u64`, `i128`, `BigInt`, `Ratio`;
  compound assignment (`+=`, `*=`, …); `ToEx` and `Scalar` traits.
- `Ex::{as_rational, as_bigint, as_i64, compare_numeric, is_less_than,
  is_greater_than, probably_equal, eval_at, subs_map_with}`.
- `Equation` accessors (`lhs`, `rhs`, `swap`, `to_zero_equation`, `to_expr`),
  arithmetic with scalars and equations, `solve` / `solve_for` /
  `solve_or_empty`, `subs`, `is_satisfied`, `is_identity`, `apply`.
- `base::numeric::{f64_to_ratio_exact, f64_to_ratio_approx}`.
- `symplex-wasm`: `Session` (persistent context with `define`) and a full
  stateless API (`integrate_definite`, `to_c_fn`, `eval_decimal`, …).
- `symplex-build`: exact DH parameters via `from_f64_approx`,
  `RobotArmBuilder::generate_fk_matrix`, `"fk_matrix"` in TOML configs.

### Fixed

**Found by the new SymPy oracle and fixed before release**

- Inequality solver: poles are now sign-change points, both-negative
  branches are kept, and the natural domain is intersected in
  (`1/x > 2` → `(0, 1/2)`, `(x−1)/(x+1) ≥ 0` → `(−∞,−1) ∪ [1,∞)`,
  `√x < 2` → `[0, 4)`).  Undecidable cases return `ConditionSet`, never a
  guess.
- `solve_system_ex` returned non-solutions (Cardano emitted `cbrt` of a
  negative radicand, evaluated on the principal branch) and `Ok([])` for
  biquadratic eliminants (Ferrari `0/0`).  Every returned tuple is now
  verified against all equations at 30 digits.
- `rsolve_linear` hung on irrational cubic characteristic roots; roots are
  now `RootOf` values and constant fitting is budgeted.
- `series_at_infinity(atan x)` returned the garbage `atan(zoo)`; constant
  terms are now limits, and unevaluable results are formal `Series` nodes.
- `evalf` Bessel `J`/`Y` were wrong for `x ≳ 12` (doubled leading Hankel
  term, sign error in the recurrence, premature series→asymptotic switch);
  now 25+ digits at any `x`.
- `eval()` of `Piecewise` selected a later `True` branch over an earlier
  undecided one.
- `0 · oo` / `0 · zoo` were order-dependent (`nan` vs `0`).
- Debug-build panic (nested `Mul`) when multiplying numeric radicals such as
  `(√6/3)·(√3/3)`.
- Display of rational/negative bases: `(2/3)^x` printed as `2/3^x`.
- `free_symbols` counted bound index variables of `Sum`/`Product`/`RootOf`/
  `RootSum`/`ConditionSet`/`DefiniteIntegral` as free.
- `eval_decimal` truncated instead of rounding the last digit.
- `eval_f64` on compound expressions with free symbols reported a cache
  miss instead of `FreeSymbol { name }`.
- Assumption lattice: `oo` is positive, extended-real and infinite but not
  real/finite; queries are order-independent; contradictory declarations
  panic with a clear message.
- `nroots` missed real roots of odd/even polynomials (mirror-symmetric Aberth
  start points); real roots are snapped only after an exact Sturm count.
- Expression construction was proportional to tree size (sort keys
  concatenated whole subtrees); keys are now bounded and hashed, and the
  debug canonical-form verifier is iterative (deep expressions no longer
  overflow the stack).
- `sqrt(<large integer>).eval()` trial-divided to `√n` (14 s); square factors
  are now found via bounded `factorint`.  Radical normal form unified:
  `√(1/2) = 1/√2 = √2/2`, `√(4/9) = 2/3`, `∛54 = 3∛2`.
- `Context::rational(p, 0)` panicked; it now returns `zoo` (`nan` for
  `0/0`).
- `abs(3 + 4i)` folds to `5`.
- Generated `no_std` code called `libm::abs` (does not exist); now `fabs`.
  `symplex-build` emitted the `symplex_rt` runtime once per function.
- `expr_type()` reported `RootOf` as unevaluated; `piecewise_simplify`
  ignored assumption-decided conditions; `BoolEx::simplify` gained
  consensus.
- Parser: `binomial`, `beta`, `bessel{j,y,i,k}`, `cot/sec/csc/coth/sech/csch`,
  `min`/`max`, `polygamma`, `Sum`/`Product`, `Integral(f, x[, a, b])`, `n!`.
- `expr!(ctx, 2^10)` (purely numeric bodies) now compiles.

**Other**

- `∫₋₁¹ dx/x²` and other integrals across interior poles no longer return a
  finite value.
- `fourier_series` coefficients for `|x|`, `sign(x)` and piecewise inputs.
- Sign error in shifted alternating half-integer p-series.
- Gosper: dispersion via a bounded gcd scan instead of resultant
  interpolation (`Σ k⁸·2ᵏ` from 23 s to 26 ms); certificate degree cap.
- Gruntz limits: wrong answers for several `exp`/`ln` towers; work budget
  prevents hangs.
- Binomial series for large `|n|`; series at hidden valuations (`1/x` at
  order 1).
- `matrix_exp` for numeric complex eigenvalues (`sin(−1)` parity);
  Jordan chains for repeated eigenvalues; nilpotent blocks.
- Real-root parity in the polynomial root counter.
- `factor_zassenhaus` on non-square-free input.
- Log-to-real exactness guard; polar-form complex powers.
- Definite integrator rejects leaked limit-engine dummies; assumption-decided
  `Piecewise` branches.
- `eval_f64_with` reports `FreeSymbol` for unbound symbols.

### Infrastructure

- CI rewritten: fmt / clippy / test / UI compile-fail (pinned toolchain
  `1.95.0`, `TRYBUILD=overwrite` to refresh snapshots) / sub-crates
  (`symplex-macros`, `symplex-build`, `symplex-wasm` native + `wasm32`, fuzz
  build) / docs (`cargo doc -D warnings` + `mdbook build`) / MSRV `1.93.0` /
  every non-interactive example run.
- GitHub Pages deployment of the mdBook.
- `tests/v02_*` integration suites per area, one concept per test and
  each under a few seconds; SymPy 1.14 oracle (`tests/fixtures/*.json`,
  ~1,400 fixtures, one `#[test]` per subcategory, strict-xfail known-bug
  tables); `tests/README.md` documents the layout and how to regenerate
  fixtures.
- Crate, `symplex-macros`, `symplex-build` and `symplex-wasm` at 0.2.0.

## [0.1.0]

Initial public release: exact arithmetic on `Ratio<BigInt>`, hash-consed
expression arena, differentiation, indefinite integration (Risch,
Rothstein–Trager, Lazard–Rioboo–Trager, heuristics), Gruntz limits,
series, Laplace and Z-transforms, polynomial solving through quartic with
`RootOf`/`RootSum`, Gröbner bases, 13 ODE classes, symbolic matrices with
eigenvalues/Jordan form/matrix exponential, algebraic number fields ℚ(α),
Rust code generation with CSE, compile-time dimensional analysis, and the
`symplex-macros`, `symplex-build` and `symplex-wasm` companion crates.

[0.3.0]: https://github.com/cgorski/symplex/releases/tag/v0.3.0
[0.2.0]: https://github.com/cgorski/symplex/releases/tag/v0.2.0
[0.1.0]: https://github.com/cgorski/symplex/releases/tag/v0.1.0
