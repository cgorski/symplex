# The nightly statistics oracle

`.github/workflows/stats-oracle.yml` runs every night (and on demand from the
Actions tab): symplex evaluates a fixed seed set of statistics cases in a
release build, and every value it claims is checked — certified decimals to
their last digit, `f64` routes to their documented tolerance, exact rationals
exactly — against mpmath at two precisions, with scipy and statsmodels (the
libraries people would otherwise use) as references judged the same way.  A
disagreement that is not on the allowlist turns the run red; the job summary
lists it with the three values.

```sh
python3 -m venv .venv && .venv/bin/pip install -r scripts/stats_oracle/requirements.txt
PYTHON=.venv/bin/python scripts/stats_oracle/run.sh          # → target/stats_oracle/report.md
```

| file | role |
|---|---|
| `cases/*.tsv` | the seed set: one case per row (format below); every `.tsv` here is read |
| `gen_cases.py` | writes `cases/{families,kernels,tests,models}.tsv` from a fixed seed (`--check`: are the committed files current?) |
| `../../tests/v36/v36_s15l5_oracle.rs` | the dumper: the test `v36_s15l5_numerics::oracle::stats_oracle_dump`, active when `STATS_ORACLE_CASES` and `STATS_ORACLE_OUT` are set (otherwise it checks a few built-in rows) |
| `run_dump.py` | runs the dumper in parallel shards, resumes after a row that hangs or crashes |
| `compare.py`, `refs.py` | the comparison: scipy / statsmodels mappings and the mpmath formulas |
| `allowlist.tsv` | known disagreements, each with its mpmath value and a reason |
| `requirements.txt` | the pinned references (a scipy release changes the oracle only through a reviewed bump) |
| `run.sh` | all of it, in order; `PYTHON` and `JOBS` override the interpreter and the parallelism |

## Case rows

Tab-separated, `#` starts a comment line:

```
id    kind    target    params    method    arg
```

* `id` — unique across all case files and stable: the allowlist refers to it.
  Generated ids are `dist.<family>.<set>.<method>[.<n>]`, `kernel.<name>.<n>`,
  `test.<name>.<n>`, `model.<name>.<n>`; a hand-written or hunter-exported file
  should use its own prefix (`l2.…`, `l3.…`).
* `kind` and the other columns:

| kind | target | params | method | arg |
|---|---|---|---|---|
| `dist` | a family: `normal`, `uniform`, `exponential`, `gamma`, `chi_squared`, `beta`, `cauchy`, `laplace`, `logistic`, `log_normal`, `student_t`, `f`, `weibull`, `pareto`, `triangular`, `bernoulli`, `binomial`, `poisson`, `geometric`, `negative_binomial`, `hypergeometric`, `discrete_uniform`, `finite` | the constructor's parameters in its order (`Distribution::gamma(shape, scale)`, `weibull(scale, shape)`, `hypergeometric(population, successes, draws)`, …) as exact rationals `N` or `N/D`; `finite`: `value:prob,…` | `pdf` (the mass function of a discrete family), `cdf`, `sf`, `quantile`, `median`, `mean`, `variance`, `skewness`, `kurtosis` (not excess), `entropy`, `mgf`, `cf`, `quantile_f64`, `isf_f64` | the point `x`, level `p` or `t` as an exact rational; for `quantile_f64` / `isf_f64` the level as an `f64` literal; `-` for none |
| `kernel` | a `stats::numdist` function: `betainc_regularized`, `gammainc_lower_regularized`, `gammainc_upper_regularized`, `<family>.<fn>` with family `norm`, `t`, `gamma`, `chi2`, `beta`, `f`, `binom`, `poisson` and fn `cdf`, `sf`, `pdf` (norm, t), `ppf`, `isf` | its arguments in order (`t.cdf`: `x,df`; `gamma.ppf`: `p,shape,scale`) as `f64` literals | `-` | `-` |
| `test` | `t_test_one_sample`, `t_test_two_sample`, `t_test_paired`, `binomial_test`, `fisher_exact`, `chi_square_independence`, `chi_square_goodness_of_fit`, `z_test_proportion`, `pearson_test`, `spearman_test`, `mann_whitney_u`, `wilcoxon_signed_rank`, `kruskal_wallis`, `anova_one_way` | `key=value;…`: samples `x`, `y` (comma-separated rationals), `groups` and `table` (rows separated by `:`), `mu0`, `p0`, `k`, `n`, `alt` (`two-sided`, `less`, `greater`), `equal_var` and `correction` (`true`/`false`), `method` (`exact`, `asymptotic`, `asymptotic_cc` = with continuity correction), `observed` | `-` | `-` |
| `model` | `ols` | `y=…;x1=…;x2=…;intercept=true` | `-` | `-` |

To add cases from a hunter (the families, tests and models hunters of wave 12),
export its interesting draws as rows in a new file `cases/<prefix>_<topic>.tsv`;
nothing else changes.  A new family, method or test needs a line in the
dumper's `match` and its references in `refs.py` / `compare.py`.

## Dump rows

`id <TAB> output <TAB> value <TAB> ms` — a `dist` row gives `d30`
(`eval_decimal(30)` of the exact expression) and `f64` (`eval_f64`), `cf`
gives `d30` and `c64` (`eval_complex64`), `quantile_f64` / `isf_f64` and the
kernels give `f64`; a test gives `statistic`, `pvalue`, `pvalue_f64` and `df`;
an OLS fit `coef.i` (exact), `se.i`, `t.i`, `p.i`, `rsquared`,
`rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic`.  Values:
`D:<digits>` (certified; complex as `a+b*i`), `F:<f64>`, `C:<re>,<im>`,
`Q:<n>/<d>`, `INF`, `-INF`, `NAN` (the exact answer is that symbol), `NONE`
(no answer by design), `OFF` (a density asked off the support: `density` is
the formula on the support, the density is 0 there by definition),
`ERR:<variant>:<message>`, and the row-level `*`
outputs `PANIC:…`, `HANG` (over 60 s), `CRASH:rc=…` (the process died),
`BADROW:…` (the dumper could not read the row).

## Verdicts

The truth is mpmath at 40 and 60 digits (refs.py), trusted where the two agree
to 33 digits.  The incomplete beta and gamma functions take each tail directly
where it is the small one (mpmath's hypergeometric `betainc` loses digits next
to 1, and gives up for huge shapes; refs.py then uses the continued fractions
and the series), so a far tail is never `1 −` a number next to 1.  symplex is judged at its exact input, scipy at the double it
was given (`x` and `p` rounded to `f64`), so the rounding of an argument is
never scipy's error.

| class | meaning | fails the run |
|---|---|---|
| `AGREE` | within its claim of the truth (and scipy within `1e-12`, or no scipy value) | no |
| `AGREE_SCIPY` | no mpmath formula (or no agreement at two precisions): within `1e-10` of scipy | no |
| `NONFINITE` | ±∞ / NaN, or a refusal, where the value is not finite | no |
| `OFFSUPPORT` | a density off the support (`OFF`), not compared | no |
| `OUT_OF_RANGE` | `eval_decimal` refuses a value beyond the arbitrary-precision exponent range, as documented | no |
| `SCIPY_INACCURATE` | symplex right, scipy off by `1e-12` … `1e-9` | no (listed) |
| `UNCHECKED` | no reference at all | no |
| `SYMPLEX_WRONG` | outside its claim of the truth | yes |
| `SCIPY_WRONG` | symplex right, scipy off by more than `1e-9` (or not finite) | unless allowlisted |
| `UNRESOLVED` | symplex and scipy disagree and mpmath cannot settle it | unless allowlisted |
| `REFUSED` | symplex refuses (`ERR`, `NONE`) where a finite value exists | unless allowlisted |
| `PANIC`, `HANG`, `CRASH`, `BADROW`, `MISSING` | as named | yes |

Claims: a `D:` value to one unit of its 30th significant digit; `eval_f64`
to `1e-15`; a `numdist` kernel to `max(5e-14, 10ε·|ln P|)` relative (its
documented accuracy); a continuous `quantile_f64` to `1e-13·κ` with `κ =
|d ln x / d ln P|`; a discrete quantile exactly (`F(k) ≥ p > F(k − 1)`, or
`S(k) ≤ q < S(k − 1)`); `pvalue_f64` to `1e-14`.

## Allowlist

`allowlist.tsv`: `id-glob <TAB> output-glob <TAB> class <TAB> mpmath value
<TAB> reason` (fnmatch globs; the class must match).  Allowlist a scipy error
with the mpmath value and, where symplex is the one that is right, keep a
test that records it ("scipy 1.18.1 gives X").  An entry that matches nothing
any more is reported as stale — delete it.
