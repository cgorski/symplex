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
