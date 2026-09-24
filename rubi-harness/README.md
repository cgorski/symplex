# rubi-harness

Runs symplex's `Ex::integrate` on the [Rubi](https://rulebasedintegration.org/)
integration test suite (72,254 integrands, vendored in [`suite/`](suite/)) and
judges every answer **by differentiation**, under symplex's real-variable
convention.  It reports per-file pass counts and, most importantly, a list
of **wrong antiderivatives**.  Each one is a correctness bug with a
reproducer.

This crate is its own Cargo workspace (`publish = false`).  It is excluded
from the published `symplex` crate (`exclude` in the top-level `Cargo.toml`).

## Running

```sh
cd rubi-harness
timeout 3600 cargo run --release -- --jobs 12            # run and report
timeout 3600 cargo run --release -- --jobs 12 --check    # fail on regression vs ratchet.tsv
timeout 3600 cargo run --release -- --jobs 12 --update   # rewrite ratchet.tsv
cargo run --release -- --only "4.1.0 (a sin)" --only Wester   # a subset of files
```

A full run of symplex 0.23.0 takes about 12–16 s of wall time with 12 jobs.
Most answers come back fast (usually unevaluated).

| option | meaning |
|---|---|
| `--jobs N` | worker processes in parallel (default: CPUs − 2) |
| `--only SUBSTR` | only files whose path under `suite/` contains `SUBSTR` (repeatable) |
| `--check` | exit 1 if any file's `verified + real_verified` count fell below `ratchet.tsv`, or its `wrong + panic` count rose above it |
| `--update` | rewrite `ratchet.tsv` from this run (rows of files not run are kept) |
| `--timeout SECS` | wall-clock limit for parsing + integrating one entry (default 10) |
| `--check-timeout SECS` | limit for verifying one entry (default 30) |
| `--mem-limit-mb MB` | resident memory limit per worker process (default 4000) |
| `--chunk N` | entries per worker process (default 25) |
| `--out DIR` | output directory (default `results/`, `results/only/` with `--only`, `results/selftest/` with `--selftest`) |
| `--selftest` | judge Rubi's *optimal* antiderivatives instead of symplex's (validates the translation and the checker) |
| `--scan` | translate and parse every integrand (no integration) and print statistics |
| `--probe EXPR [VAR]` | run one Maxima expression through the whole pipeline, verbosely; with `RUST_LOG` set it logs to stderr (`RUST_LOG=symplex::stage=debug` lists the steps that took ≥ 250 ms or created ≥ 200,000 nodes, see CONTRIBUTING.md) |

## Pipeline

1. **Reading** (`src/mac.rs`).  Maxima comments are blanked (they nest).  The
   `lst: '[ ... ]` list is split into entries by bracket nesting, and each
   entry into fields at its depth-0 commas:
   `[integrand, variable, rubi_steps, optimal_antiderivative (, alternative)]`.
2. **Translation** (`src/translate.rs`).  The Maxima text is parsed into a
   small AST and printed in the syntax `symplex::parse` accepts:
   - `%e`, `%pi`, `%i` → `E`, `pi`, `I`.  A bare identifier is a symbol.
     Names symplex reads as constants get a trailing `_`: the Rubi parameter
     `e` becomes `e_` (not Euler's number), and `i` becomes `i_`.
   - `log` → `ln`.  `asec(u)`, `acsc(u)`, `acoth(u)`, `asech(u)`,
     `acsch(u)` → `acos(1/u)`, `asin(1/u)`, `atanh(1/u)`, `acosh(1/u)`,
     `asinh(1/u)`.  symplex has no nodes for them, and these are the
     Mathematica/Maxima definitions.
   - The suite's Mathematica-flavoured special functions are mapped where
     the conventions match: `Si Ci Shi Chi Ei`, `Li` → `li`,
     `Ei(n, z)` → `expint(n, z)`, `FresnelS/C` → `fresnels/c`,
     `ProductLog` → `lambertw`, `GAMMA(a, z)` → `uppergamma`,
     `polylog(n, z)`, `elliptic_f(φ, m)`, two-argument `atan(x, y)` →
     `atan2(y, x)`, …
   - Everything else is **unsupported**, with a reason:
     hypergeometric/Appell functions, incomplete `elliptic_e(φ, m)` and
     `elliptic_pi(n, φ, m)`, Hurwitz `Zeta(s, a)`, abstract functions
     `f(x)`/`F(x)`, formal `Derivative(1)(f)(x)`.  If symplex's parser
     rejects a translation, that is also *unsupported*.
3. **Integration**.  A fresh `Context` for every entry, then
   `f.integrate(&x)` inside `catch_unwind`.
4. **Check** (`src/check.rs`).  Every free symbol other than `x` gets a fixed
   generic rational: `a=6/5, b=3/4, c=5/3, d=2/7, e=11/9, f=13/6, g=5/8,
   h=9/7, i=8/11, m=4/3, n=5/7, p=3/11, q=2/13, A=19/10, B=23/12, …` (see
   `table_value`).  No value is a sample point.  Then `F′` (symplex's
   `diff`) and `f` are evaluated at `x ∈ {1/3, 7/5, 13/4, −5/7, −13/4}`
   with `eval_decimal(30)`, as complex numbers with principal branches.
   They are compared with relative tolerance `1e-8`.  A mismatch is
   re-evaluated at 80 digits.  It is dropped if it then agrees, or if the
   difference shrank by 20 orders of magnitude (rounding noise around a
   common zero).  A point where either side does not evaluate is skipped.
   The integrand counts as *real* at a point if `|Im f| ≤ 1e-12·|f|`.
5. **Verdict** (`check::verdict`).  symplex promises that `F′ = f` on
   every real interval where `f` is real and continuous.  Where `f` is
   complex for real `x` (e.g. `1/sqrt(x²−1)` on `(−1, 1)`), `ln|u|`-style
   answers legitimately differ from `f`.  So only mismatches at points
   where `f` is real count against an answer.  The exception is an
   integrand containing `%i`: it is complex by construction, and there
   `ln|u|` is a genuine bug, so any mismatch counts.

### Classification

| status | meaning |
|---|---|
| `unsupported` | integrand not translatable/parsable, or uses functions symplex lacks |
| `verified` | `F′ = f` at every point where both evaluate, and at least one point evaluated |
| `real_verified` | `F′ = f` at every point where `f` is real; `F′ ≠ f` only at points where `f` is complex; the integrand has no `%i` |
| `unevaluated` | the result `has_unevaluated()` (contains a formal `Integral`, …) |
| `wrong` | `F′ ≠ f` at a point where `f` is real, or (integrand contains `%i`) at any point |
| `undecided` | no point evaluated on both sides, or the check exceeded `--check-timeout` |
| `timeout` | parsing + integration exceeded `--timeout`, or the worker exceeded `--mem-limit-mb` |
| `panic` | symplex panicked (caught), or the worker process died (e.g. stack overflow) |

`real_verified` is vacuously true when `f` is complex at *every*
evaluated point.  For example, `1/(a+x*sqrt(-a))` with `a = 6/5 > 0` has
no real sample point, and symplex's `ln|…|` answer is only testable for
`a < 0`.  Such cases are tagged "no real sample point evaluated" in
`real_verified.txt` and counted separately in its header.

Every `wrong` entry in `results/wrong.txt` carries two triage aids:

- **classification**: whether a mismatch happened at a point where the
  integrand is real, or only where it is complex in an integrand
  containing `%i`.
- **Rubi cross-check**: Rubi's optimal antiderivative is differentiated and
  compared at the same points.  If Rubi's answer also "fails", suspect the
  harness (translation, branches, evaluation) rather than symplex.

### Process model

The driver splits files into chunks and runs each chunk in a child process
(`rubi-harness --worker FILE START END`).  The children stream one record
per line (`src/protocol.rs`).  The driver enforces the per-entry limits.
When an entry exceeds a limit, the driver kills the child, classifies the
entry, and starts a new child at the next entry.  A crash (stack overflow,
abort) is handled the same way.  So a hanging or runaway integration costs
at most one timeout and can never wedge the run.  Details (the answer, the
points) are streamed before the final record, so a `wrong` or
`real_verified` verdict survives even if printing or cross-checking then
times out.  For testing the driver,
`RUBI_HARNESS_FAULT=panic:N` (or `abort:N`, `hang:N`) injects a fault at
entry `N` (0-based) of every file.

## Outputs

| file | committed | content |
|---|---|---|
| `ratchet.tsv` | yes | per file: `min_verified` (counts `verified + real_verified`), `max_wrong_panic` (`wrong + panic`) |
| `results/summary.tsv` | yes | per file: total, unsupported, verified, real_verified, unevaluated, wrong, undecided, timeout, panic, seconds; then `TOTAL <chapter>` rows and `TOTAL` |
| `results/wrong.txt` | no | every wrong/panic case: file, entry, line, Maxima integrand, translation, Rubi's answer, symplex's `F`, parameter values, per-point values, cross-check |
| `results/real_verified.txt` | no | every real_verified case, same format (no cross-check) |
| `results/undecided.txt` | no | every undecided case with the per-point reasons |
| `results/entries.tsv` | no | one line per entry: status, seconds, reason |
| `results/reasons.tsv` | no | reason counts for unsupported/undecided/timeout/panic |

Timeouts depend on machine load.  An entry close to the limit can flip
between `verified` and `timeout`, which `--check` would report.  If that
happens, rerun with fewer `--jobs` or a larger `--timeout`.

## Validation of the harness itself

`--selftest` runs the same checker, with the same five points and verdicts,
on Rubi's optimal antiderivatives, which are known to be correct.  On the
full suite:

- 55,044 verified and 0 real_verified (0.26).  Rubi's answers are valid
  for complex `x` too.
- 1,173 undecided: symplex cannot evaluate the derivative numerically,
  often `elliptic_f`/`polylog` at complex arguments.
- 16,037 unsupported, mostly answers with hypergeometric, incomplete
  elliptic or Appell functions, or `Unintegrable`.
- 0 wrong.  Until 0.25 there were 2 false alarms: two entries whose Rubi
  answer divides by `x − log(%e^x)`, which is identically zero, and the
  evaluator returned noise instead of an error.  Since 0.26 it tracks an
  error bound and refuses (`PrecisionExhausted`), so those points are
  skipped.

An earlier false alarm came from symplex's own `acosh` derivative
(`u′/sqrt(u²−1)`, wrong sign for `u < −1`); it is now fixed in the library.
This shows the check trusts symplex's own `diff` and `eval_decimal`.  A bug
there can produce a false `wrong` or, less likely, mask a real one.  The
Rubi cross-check in `wrong.txt` guards against the first kind.

## Licence

The test suite in `suite/` is © 2018 Rule-based Integration, MIT licence
(`suite/LICENSE`), vendored unmodified from
<https://github.com/RuleBasedIntegration/MaximaSyntaxTestSuite> at commit
`60295e21c571ca210ecfbb695f4af99947454adf` (see `suite/README.md`).  The
harness code is licensed like symplex (MIT OR Apache-2.0).
