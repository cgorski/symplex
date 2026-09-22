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
| `fuzz_simplify` | `simplify`, `expand`, `factor`, `together`, `cancel`, `ratsimp`, `simplify_trig` preserve the value at every sample point where both sides are finite reals (30-digit evaluation); `simplify` is idempotent |
| `fuzz_integrate` | when `integrate` returns a closed form `F`, `F′ = f` at the sample points |
| `fuzz_parser` | parsing never panics; what parses displays, re-parses and prints as LaTeX |
| `fuzz_refine`, `fuzz_eigenvects`, `fuzz_lambertw` | `refine` under assumptions, `A·v = λv` for eigenvectors, `W(x)·e^{W(x)} = x` |

`fuzz_simplify`, `fuzz_integrate` and `fuzz_poly` decode their input with
`fuzz_targets/common/mod.rs` (elementary expressions of depth ≤ 4 in one
symbol; odd roots other than `√` are left out while the three evaluators
disagree on odd roots of negative numbers).  `print_expr` is not a target:
it prints the expression an input decodes to.

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
