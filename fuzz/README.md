# Fuzzing symplex

Coverage-guided fuzz targets for [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
(libFuzzer; needs a nightly toolchain: `cargo install cargo-fuzz`).  They run
nightly in CI (`.github/workflows/fuzz.yml`, ten minutes per target, one job
per target, the corpus cached between nights).

Every target checks a **property**, not just "no panic" — so a failure is a
wrong answer, a hang or a crash, never noise:

| Target | Checks |
|---|---|
| `fuzz_numdist` | `stats::numdist`: `cdf`/`sf` in `[0, 1]`, `cdf + sf = 1`, `cdf` monotone, `ppf`/`isf` terminate and invert (`cdf(ppf(p)) ≈ p`, or the neighbouring floats bracket `p` where the quantile is not representable), for every family over the whole `f64` range |
| `fuzz_stats` | the symbolic distributions (`stats::Distribution`: the 23 families and finite tables, with exact, extreme and irrational parameters, sometimes wrapped by `affine`/`truncated`/an order statistic/a mixture/`transformed`): `try_` constructors reject invalid parameters; the density is real, `≥ 0` and integrates (sums) to 1; `cdf`/`sf` in `[0, 1]`, monotone, complementary, `0`/`1` beyond the ends; `cdf′ = pdf` (lattice: `cdf(k) − cdf(k − 1) = pmf(k)`); closed quantiles invert the CDF and `quantile_f64`/`isf_f64` meet their level (lattice: `cdf(k) ≥ p > cdf(k − 1)`, decided exactly); moments that exist match the integrals and the others are `±∞`/NaN; `mgf`/`cf` against their integrals and domains; the eight `numdist` families agree with the symbolic `cdf`/`sf`; order statistics of tables against brute-force enumeration; closures of `sum_distribution`; `transformed` by monotone maps.  Claims are decided exactly or from certified digits |
| `fuzz_exact_matrix` | `QMatrix`/`Matrix` exact identities: `det(AB) = det A·det B`, `A·A⁻¹ = I`, rank–nullity, `rref` idempotent, Cayley–Hamilton, `P·A = L·U`, the `Matrix` tier agrees with `QMatrix` |
| `fuzz_poly` | ℚ[x]: `gcd` divides both inputs and is divisible by their planted common factor, Bézout for `poly_gcdex`, `a = q·b + r`, `factor_list` reproduces the input, every `solve` root makes `a` vanish and `c·a` has the same roots |
| `fuzz_simplify` | `simplify`, `expand`, `factor`, `together`, `cancel`, `ratsimp`, `simplify_trig` preserve the **complex** value at real and complex sample points (second quadrant, just below the negative real axis, `Im x = 4 > π`), wherever both sides evaluate and are continuous (30-digit evaluation) — a symbol without assumptions may be complex; `simplify` is idempotent |
| `fuzz_integrate` | when `integrate` returns a closed form `F`, `F′ = f` at the sample points |
| `fuzz_parser` | parsing never panics; what parses displays, re-parses and prints as LaTeX |
| `fuzz_refine`, `fuzz_eigenvects`, `fuzz_lambertw` | `refine` under assumptions, `A·v = λv` for eigenvectors, `W(x)·e^{W(x)} = x` |
| `fuzz_evalf` | evalf self-consistency on constant expressions (special functions, sums, `RootOf`, complex arguments, cancellation, huge/tiny scales): every digit `eval_decimal` certifies at 16 and 30 digits agrees with the 60-digit value to one unit in the last place, a part printed `0` is negligible, `eval_f64`/`eval_complex64` are within one ulp and `eval_f64` refuses a non-negligible imaginary part, no non-refusal error at one precision while another evaluates |
| `fuzz_calculus` | the calculus routines against independent numerical oracles (`calc/mod.rs`): `integrate_definite` vs adaptive Gauss–Legendre quadrature (or `Divergent` where the quadrature converges), `limit` vs the function at `p ± 10⁻ᵏ`, `series` residual `O(hⁿ)`, `diff` vs central differences, every `solve` solution satisfies the equation and no real root is missed, infinite sums vs Richardson-extrapolated partial sums and finite sums vs the partial sums, `dsolve` by substitution |
| `fuzz_roundtrip` | API-built expressions (special functions, `Piecewise`, `Derivative`, `Integral`, `Sum`/`Product`, `Limit`, `Subs`, `RootOf`, huge/tiny rationals, `I`, `oo`, `zoo`, `nan`) display as text that `parse` reads back to the **same tree** (relations through `parse_bool`); `to_latex`, `pretty`, `pretty_ascii`, `to_mathml` do not panic |

`fuzz_simplify`, `fuzz_integrate` and `fuzz_poly` decode their input with
`fuzz_targets/common/mod.rs`: elementary expressions of depth ≤ 4 in one
symbol for `fuzz_integrate`/`fuzz_poly` (`Grammar::Elementary`), and for
`fuzz_simplify` also `sinh`/`cosh`/`tanh`, their inverses, `asin`/`acos` and
rational powers `x^(p/q)`, `q ≤ 5` (`Grammar::Full`).  `fuzz_integrate`
compares real values at real points: the integrator's antiderivatives are
the real-variable ones (`∫ dx/x = ln|x|`).  `print_expr` is not a target:
it prints the expression an input decodes to (`depth 4:` is the
`fuzz_simplify` tree, `depth 3:` the `fuzz_integrate` one).

`fuzz_evalf`, `fuzz_calculus` and `fuzz_roundtrip` are the in-house
differential hunters of 0.30 made permanent.  Their generators decode the
input through `fuzz_targets/common/choose.rs` (`Src`: one decision per
byte, or a few for a wide range; zero bytes past the end pick leaves), so
a mutated byte changes one choice.  A failure panics with the case and
both values; `FUZZ_SHOW=1 $B/<target> <input>` prints the decoded case and
its verdict (`ok`, or why it was skipped) instead of needing `print_expr`,
and `-runs=0` over a corpus directory shows the whole corpus that way.
`fuzz_roundtrip` also takes `FUZZ_TRACE=1`: every subexpression is
printed as it is built, which locates the construction step that left a
non-canonical form (and `FUZZ_SHOW=1` prints a relation's tree).
They check self-consistency (`fuzz_evalf`) or agreement with an oracle,
which cannot see an error every route shares: evalf's `erfcinv` tail was
wrong at every precision (0.30), found by `compile()`, not by `fuzz_evalf`.
`fuzz_calculus` skips what its oracles cannot decide (non-convergent
quadrature, domain problems, unevaluated results) — a failure is a wrong
answer.  Typical rates: `fuzz_evalf` ~500 exec/s, `fuzz_calculus` ~100,
`fuzz_roundtrip` several thousand.

To keep fuzzing past the first finding and collect every failing input,
run in fork mode: `$B/fuzz_simplify -fork=6 -ignore_crashes=1
-ignore_timeouts=1 -max_total_time=300 -artifact_prefix=… fuzz/corpus/fuzz_simplify`,
then replay each artifact to read its message.

The nightly job uploads a failing input before decoding it and prints it
as base64 in the log (`base64 -d` gives the file back); every replay there
is bounded, so an input that hangs is reported with libFuzzer's stack
instead of cancelling the job.

Every job also uploads its corpus as the artifact `fuzz-corpus-<target>`
(kept seven days, on success as well as on failure), so a fix can be run
against what CI has collected:

```sh
gh run list --workflow fuzz.yml -L 5                    # the run id
gh run download <run-id> -n fuzz-corpus-fuzz_integrate -D fuzz/corpus/fuzz_integrate
$B/fuzz_integrate -runs=0 fuzz/corpus/fuzz_integrate    # replay all of it ($B: see below)
```

## Known slow inputs and the five skips of `fuzz_stats`

`fuzz_stats` has five temporary skips at 0.60.1 (three at 0.60).  Each
keeps a known slow path of the library out of the nightly run, so that the
run reports new findings; remove each skip with its fix.  The harness
(`fuzz_targets/fuzz_stats.rs`) is the reference for what is skipped; the
times below are those of a release build unless the fuzz build is named.

- It transforms a lattice with unequal masses into a table only up to 512
  atoms (`MAX_TRANSFORMED_UNEQUAL`): every query of such a table sums its
  masses as expressions (`√X` for `X ~ Binomial(9000, 1/3)`: one `cdf` and
  one quantile take 35 s, in 0.60.0 and now; for `Binomial(2000, 1/3)`
  1.1 s).  A lattice of equal masses (`Die`, `DiscreteUniform`) is
  transformed whatever its length since 0.60.1.  Up to 0.60 the cap was
  20,000 atoms for every lattice (`MAX_TRANSFORMED_ATOMS`): for unequal
  masses the harness now checks less than it did, for equal masses more.
  Removed by: a table that keeps the lattice it came from and asks it for
  its tails (the tails of the image of a monotone map are the tails of the
  lattice).
- It skips a lattice of more than 4,096 atoms (`MAX_SUMMED_ATOMS`) whose
  tails the library can only form as sums of its masses
  (`tails_are_long_sums`: the family gives neither a `cdf` nor an `sf`
  there; today a hypergeometric law whose `C(N, n)` and `C(N, m)` both
  exceed `2¹⁶` bits, the bound of the exact tails).  Such a law is not
  drawn into a mixture either.  Unchanged in 0.60.1.  Measured there: a
  tail of such a law is a sum that evaluates in 5 to 50 ms, but every step
  of a quantile search is one such sum (`quantile_f64(0.3)`: 0.23 s and
  0.55 s for `Hypergeometric(2000000, 1875000, 1875000)` and
  `Hypergeometric(10000000, 5000000, 300000)`, about four times that in
  the fuzz build).  Removed by: search steps decided in integers from one
  anchored term.
- Without a finite variance to standardise by, the first moment of a
  continuous law is checked against the mean, to an absolute `10⁻⁷`, only
  while the mean is at most `10⁸` in absolute value (`check_moments`): for
  `LogNormal(0, 8π²/3)` (mean `2.6·10¹⁵⁰`) the quadrature was asked for
  `10¹⁵⁰ − 10¹⁵⁰` to `10⁻⁷`, which it refuses, but only after 5 s.
  Unchanged in 0.60.1: the slow refusal is the harness's own call of
  `integrate_numeric`, nothing of the statistics code runs in it.
- It does not start a quadrature check, and does not evaluate a moment
  generating or characteristic function that the library left as an
  integral, when the integrand compiled to `f64` is not finite at one of
  the case's sample points (`leaves_the_doubles`, new in 0.60.1): the
  constants of such an integrand have hundreds of digits, and the
  quadratures work in arbitrary precision.  It was added for the red
  nightly after 0.60.0, the moment generating function of
  `FDistribution(11, 2.3·10⁻²⁵³)` at `−1/8`, whose integral was refused only
  after the quadrature's whole budget of evaluations (1.45 s; 5.5 s in the
  fuzz build).  0.60.1 also changed the quadrature: that integral is now
  refused at once (0.09 s), as is every such F law with `d₂` below about
  `10⁻¹⁷⁰`; with `d₂` between about `10⁻⁴⁰` and `10⁻¹⁶⁵` the evaluation still
  takes 1.2 to 4.4 s for a value or 1.5 to 1.9 s for a refusal (an input
  of 5 to 20 s in the fuzz build).  Removed by: a quadrature piece in
  `ln x`, which would evaluate these integrals instead of refusing them;
  whether the skip can go before that for the inputs now refused at once
  has not been measured on the merged code.
- It checks a characteristic function against the quadrature of
  `cos(sx)·f` and `sin(sx)·f` only for tails of order 4 and more (up to
  0.60: 1 and more): for a Student t with `ν ≤ 3`, `integrate_numeric`
  refuses only after 0.8 to 3.7 s, and eight inputs of the saved corpus
  took 6.3 to 7.9 s on it in the fuzz build.  An existing skip made wider,
  not a fix.  Removed by: a fast refusal of an oscillating integrand with a
  polynomial tail.

Each input stops starting new steps after 2 s (`BUDGET_SECONDS`; 5 s up to
0.60).  That is a clock, so a slower machine checks less of an input: under
the sanitizer in CI the input `AAoAAADOzs7Ozis=` spent its budget on four
quantiles and ended as a slow unit, while the plain build went on to nine
more, of a minute each.  An input takes at most the 2 s plus its last
step, which is why every single step must be fast.

The nightly runs on the tree before 0.58 found no wrong answer, only inputs
that are right and slow: seconds in a fuzz build without the sanitizer, and
CI (AddressSanitizer on a shared runner, several times slower) stops an
input after 30 s.  These are the ones 0.58 left open, and what 0.59, 0.60
and 0.60.1 did.  The times are those of the fuzz build without the
sanitizer at 0.58, then of a release build (for `fuzz_evalf`, the five
evaluations of its check):

| Input | Fuzz build, 0.58 | Release build | State at 0.60.1 |
|---|---|---|---|
| `fuzz_calculus` `bc61d72d0ea7`: `limit(abs((x − 1)^tanh(x − 1))^(x − 1), x, 1, −)`, a refusal | 5.6 s | 1.2 s → 0.05 s | **fixed** in 0.59 (the same refusal, at once) |
| `fuzz_evalf`: `polylog(−3/2, −1 + 10⁻⁷⁰)` | 5.1 s | 0.91 s → 0.07 s | **fixed** in 0.59 |
| `fuzz_evalf`: `floor(erfc(√2) − 1 + erf(√2))` | 7 s | refused after 1.0 s → `0` in 0.01 s | **fixed** in 0.59 |
| `fuzz_evalf`: `sinh(W(−1/e + 3·10⁻³⁰⁰))` | 3.8 s | 0.42 s → 0.01 s | **fixed** in 0.59 |
| `fuzz_evalf`: `zeta(−2 + 10⁻³⁰⁰)` | 2.7 s | 0.44 s → 0.005 s | **fixed** in 0.59 |
| `fuzz_calculus` `c9feedf1241f`: `series(atanh(π^x + x^(3/2)), x, 1, 4)`, right | 6.7 s | 1.83 s → 0.001 s | **fixed** in 0.60 (the same expansion; built as CI builds the target, with AddressSanitizer, 57 s → 0.5 s) |
| `fuzz_stats`: `√X` of a 100,000-sided die, the transform into a table | 5.5 s | 1.4 s → 0.33 s | **fixed** in 0.60.1 (no image is evaluated; the input takes 1.9 s in the fuzz build, where the 2 s limit ends it after the first point) |
| `fuzz_stats`: `Hypergeometric(2000000, 1875000, 1875000)`, whose tails are sums of 125,001 masses | a timeout: seconds per tail, minutes per quantile | — | open; kept out by `tails_are_long_sums` |
| `fuzz_stats`: `integrate_numeric((x − μ)·f(x), 0, ∞)` for `LogNormal(0, 8π²/3)`, a refusal | 5.1 s | — | open; no longer asked (the mean is above `10⁸`) |
| `fuzz_evalf` (three inputs, one expression): `li(sin(1 + 10⁻³⁰⁰) − sin(1))` | 3.9 s (at 0.60) | 0.145 s → 0.002 s per evaluation | **fixed** in 0.60.1 (`Ei` left of zero by a continued fraction) |

## Finding and fixing a failure — the fast loop

Build once per change to `src/` (≈ 2 min, the rest is ~1 s each).  `-s none`
turns AddressSanitizer off, which is faster to build and to run; keep it on
(`cargo +nightly fuzz build <target>`) when chasing a crash without a Rust
panic message — it turned a bare "Segmentation fault" into a stack-overflow
report with the recursing function.

```sh
cargo +nightly fuzz build -s none fuzz_integrate            # ~2 min after a src/ change
B=fuzz/target/aarch64-apple-darwin/release                   # x86_64-unknown-linux-gnu on CI
$B/fuzz_integrate fuzz/artifacts/fuzz_integrate/crash-…      # replay one input, ~1 s
$B/print_expr    fuzz/artifacts/fuzz_integrate/crash-…       # the expression it decodes to
$B/fuzz_integrate -max_total_time=30 -timeout=20 \
    -artifact_prefix=fuzz/artifacts/fuzz_integrate/ fuzz/corpus/fuzz_integrate   # a 30 s burst
```

Then: reproduce the finding as a regression test in `tests/` (with the oracle
value cited, as everywhere else), fix the root cause, rebuild, replay the
artifact, and only then escalate the bursts (30 s → 2 min → 5 min).

### Slow inputs found by the Fuzz run on the 0.60.0 release commit

All three took the same time in 0.59.0; none was a wrong value.  Times are
for the plain fuzz build (`-s none -O -a`) on a 64-core x86-64 machine; CI
builds with AddressSanitizer on a shared runner (about 7× slower, 30 s
limit).  The parameters are those of the replayed inputs (the list written
after 0.60.0 gave `d₂ ≈ 2.2·10⁻²⁴⁹` and `ν ≈ 6.6·10²³⁸`).

| Input (base64) | Case | 0.60.0 | 0.60.1 |
|---|---|---|---|
| `KgsBCjDsDjVADouL` (the timeout) | `FDistribution(11, d₂ ≈ 2.3·10⁻²⁵³)`, `mgf(−1/8)` | 5.5 s | 0.28 s: **not asked** (`leaves_the_doubles`).  The evaluation itself is a refusal after 0.09 s in a release build instead of 1.45 s; the integral has the value `6.5973·10⁻²⁵¹`, which the library does not compute |
| `uymICuU9PQ==` | `StudentT(ν ≈ 6.6·10²³⁶)` against its `numdist` kernel | 5.4 s, 4 values compared | **fixed**: 0.32 s, all 9 compared |
| `AAoAAADOzs7Ozis=` | `StudentT(1)` truncated to `[−3.2·10²⁹⁹, 3.08]`, density checks | more than 120 s | **fixed** (each quantile 12 s → 0.12 s in a release build); the input takes 2.5 s |

### Slow inputs that remain after 0.60.1

Measured while the fixes were made, each on the tree of its fix; no run on
the merged code is recorded here.

- A fifteen-minute run of `fuzz_stats` (64 forks, the saved corpus of 3,932
  inputs, plain fuzz build) on the statistics fixes: 878,048 executions, no
  crash, no timeout, no out-of-memory; 80 inputs reported slow under the
  load, 5 of them above 3 s when replayed alone (3.3 to 3.7 s): `Poisson(λ)`
  for `λ` between `1.3·10⁹` and `1.8·10⁹` against its `numdist` kernel,
  one evaluation of the incomplete gamma function at a shape of `10⁹`
  next to its mode.  With the incomplete-gamma fix of 0.60.1 (392 ms →
  43 ms per value in a release build) those inputs took 0.81 to 0.90 s.
- `ChiSquared(≈ 10⁹)` against its kernel: 2.85 to 2.94 s in that run; not
  measured again after the incomplete-gamma fix.
- The replay of the whole saved corpus, each input once, 48 at a time:
  690 s with 38 inputs above 3 s at 0.60.0; 408 s with 2 above 3 s (a
  `Poisson` and a `ChiSquared` case as above) and none above 10 s on the
  statistics fixes.
- `Poisson` between 2 and 9 standard deviations above its mean at a rate of
  `2·10⁹`: 55 to 150 ms per value in a release build, an input of about
  1.3 s in the fuzz build.
