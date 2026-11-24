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

## Known slow inputs and the three skips of `fuzz_stats`

`fuzz_stats` has three temporary skips.  Each keeps a known slow path of
the library out of the nightly run, so that the run reports new findings;
remove each skip with its fix.  All three are still in the harness at 0.60:

- It does not transform a lattice of more than 20,000 atoms into a table
  (`MAX_TRANSFORMED_ATOMS`): the table costs one certified evaluation per
  atom to merge equal images, about 1.4 s for 100,000 atoms in a release
  build and past the nightly timeout under the sanitizer.  Larger tables
  are covered by the timed tests in `tests/v36/`.
- It skips a lattice of more than 4,096 atoms (`MAX_SUMMED_ATOMS`) whose
  tails the library can only form as sums of its masses
  (`tails_are_long_sums`: the family gives neither a `cdf` nor an `sf`
  there; today a hypergeometric law whose `C(N, n)` and `C(N, m)` both
  exceed `2¹⁶` bits).  Such a law is not drawn into a mixture either.
- Without a finite variance to standardise by, the first moment of a
  continuous law is checked against the mean, to an absolute `10⁻⁷`, only
  while the mean is at most `10⁸` in absolute value (`check_moments`): for
  `LogNormal(0, 8π²/3)` (mean `2.6·10¹⁵⁰`) the quadrature was asked for
  `10¹⁵⁰ − 10¹⁵⁰` to `10⁻⁷`, which it refuses, but only after 5 s.

The nightly runs on the tree before 0.58 found no wrong answer, only inputs
that are right and slow: seconds in a fuzz build without the sanitizer, and
CI (AddressSanitizer on a shared runner, several times slower) stops an
input after 30 s.  These are the ones 0.58 left open, and what 0.59 and
0.60 did.  The times are those of the fuzz build without the sanitizer at
0.58, then of a release build (for `fuzz_evalf`, the five evaluations of
its check):

| Input | Fuzz build, 0.58 | Release build | State at 0.60 |
|---|---|---|---|
| `fuzz_calculus` `bc61d72d0ea7`: `limit(abs((x − 1)^tanh(x − 1))^(x − 1), x, 1, −)`, a refusal | 5.6 s | 1.2 s → 0.05 s | **fixed** in 0.59 (the same refusal, at once) |
| `fuzz_evalf`: `polylog(−3/2, −1 + 10⁻⁷⁰)` | 5.1 s | 0.91 s → 0.07 s | **fixed** in 0.59 |
| `fuzz_evalf`: `floor(erfc(√2) − 1 + erf(√2))` | 7 s | refused after 1.0 s → `0` in 0.01 s | **fixed** in 0.59 |
| `fuzz_evalf`: `sinh(W(−1/e + 3·10⁻³⁰⁰))` | 3.8 s | 0.42 s → 0.01 s | **fixed** in 0.59 |
| `fuzz_evalf`: `zeta(−2 + 10⁻³⁰⁰)` | 2.7 s | 0.44 s → 0.005 s | **fixed** in 0.59 |
| `fuzz_calculus` `c9feedf1241f`: `series(atanh(π^x + x^(3/2)), x, 1, 4)`, right | 6.7 s | 1.83 s → 0.001 s | **fixed** in 0.60 (the same expansion; built as CI builds the target, with AddressSanitizer, 57 s → 0.5 s) |
| `fuzz_stats`: `√X` of a 100,000-sided die, the transform into a table | 5.5 s | 1.4 s | open; kept out by `MAX_TRANSFORMED_ATOMS` |
| `fuzz_stats`: `Hypergeometric(2000000, 1875000, 1875000)`, whose tails are sums of 125,001 masses | a timeout: seconds per tail, minutes per quantile | — | open; kept out by `tails_are_long_sums` |
| `fuzz_stats`: `integrate_numeric((x − μ)·f(x), 0, ∞)` for `LogNormal(0, 8π²/3)`, a refusal | 5.1 s | — | open; no longer asked (the mean is above `10⁸`) |

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
