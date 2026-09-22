# Analysing rater and response data

Many studies produce the same shape of data: several people (raters,
annotators, respondents, workers) each answer some of a set of items, and
the analyst wants to know how much they agree, what the "true" answer to
each item is, how reliable each rater is, whether two groups differ, and
which of many comparisons survive correction.  `symplex::stats` answers
those questions **exactly**: every coefficient that is a rational function
of the counts is an exact rational, every test statistic is an exact
expression, and every p-value is an exact expression (or an exact rational
for the discrete exact tests) that is only rounded when you ask with
`.eval_f64()`.  The oracles the tests cite are `statsmodels`, `scipy`,
the `krippendorff` package and Python's `statistics` module on `Fraction`s.

The walk-through below is `tests/v13/v13_walkthrough.rs`; every number in
it is produced by the library.

## The data

Eight items rated by five raters into three categories `0, 1, 2`, with two
missing ratings:

```rust,ignore
use symplex::stats::agreement::*;
let table = RatingTable::from_i64_missing(&[
    &[Some(0), Some(0), Some(0), Some(0), Some(1)],
    &[Some(1), Some(1), Some(1), Some(2), Some(1)],
    &[Some(2), Some(2), Some(2), Some(2), Some(2)],
    &[Some(0), Some(1), Some(0), None,    Some(0)],
    &[Some(1), Some(1), Some(2), Some(1), Some(1)],
    &[Some(2), Some(1), Some(2), Some(2), None   ],
    &[Some(0), Some(0), Some(1), Some(0), Some(0)],
    &[Some(1), Some(2), Some(1), Some(1), Some(1)],
])?;
```

`RatingTable` is items × raters with `Option<Q>` cells (`Q` is the exact
rational `Ratio<BigInt>`; `symplex::linprog::{q, qi}` build literals).
`from_i64`, `from_rows`, and the raters-first `from_raters_i64` (the
`krippendorff` package's layout) are the other constructors.

## How much do the raters agree?

| Question | Function | Result on the data |
|---|---|---|
| Two raters, raw agreement | `percent_agreement(&a, &b)` | `5/8` |
| Two raters, chance-corrected (nominal) | `cohen_kappa(&a, &b)` → `KappaResult { kappa, observed, expected }` | κ = `3/7`, observed `5/8`, expected `11/32` |
| Two raters, ordinal categories | `weighted_kappa(&a, &b, &Weights::Linear)` (or `Quadratic`, `Custom`) | — |
| Many raters, nominal, complete rows | `fleiss_kappa_ratings(&complete)` | `23/48` |
| Many raters, **missing data**, any scale | `krippendorff_alpha(&table, Level::Nominal)` | `214/473` |
| … treating the categories as ordered | `krippendorff_alpha(&table, Level::Ordinal)` | `577/836` |
| Interval-scale ratings, reliability of one rater / the mean of `k` | `icc(&complete, IccForm::Icc2Single)` (also `Icc1`, `Icc3Single`, `*Average`) | `133/183` |
| Rankings of items by several judges | `kendall_w(&complete)` | `577/745` |
| Alternatives to κ | `scott_pi`, `gwet_ac1` | — |

Every one of these is an exact rational: the `krippendorff` package
reports `0.4524312896405921` for the nominal α, which *is* `214/473`.  The
coefficients that need complete tables (Fleiss, ICC, Kendall's W) say so
in their errors; Krippendorff's α is the one to reach for when cells are
missing.  `confusion_matrix`, `category_frequencies` and
`RatingTable::count_table` expose the counts the coefficients are built
from.

## What is the answer to each item?

```rust,ignore
use symplex::stats::aggregation::*;
let labels = LabelTable::from_rows(&rows, 3)?;      // items × raters, Option<usize>
let votes = majority_votes(&labels);                // Vote { winner, tied, counts }
let ds = dawid_skene(&labels, &DawidSkeneOpts::default())?;
ds.labels();          // one label per item (argmax posterior)
ds.confusion[j];      // rater j's estimated confusion matrix
ds.priors;            // class prevalences
```

`majority_vote` / `majority_votes` / `plurality(labels, &threshold)` /
`weighted_vote(labels, &weights)` are exact.  `dawid_skene` is the
Dawid–Skene (1979) EM algorithm in `f64` — deterministic, with `max_iter`,
`tol`, `smoothing` and an initialisation choice — returning per-item
posteriors, per-rater confusion matrices, class priors and convergence
information; `dawid_skene_counts` takes replicate ratings.  For pairwise
judgements ("is A better than B?") `bradley_terry(&wins, &opts)` fits the
Bradley–Terry model by Hunter's MM algorithm.

## How good is each rater?

```rust,ignore
use symplex::stats::estimation::{proportion_interval, IntervalMethod};
let acc = worker_accuracy(&rater_labels, &gold)?;   // Accuracy { correct: 5, answered: 8, accuracy: Some(5/8) }
category_metrics(&rater_labels, &gold, 3)?;         // precision / recall / F₁ per category, exact
let ci = proportion_interval(5, 8, 0.95, IntervalMethod::ClopperPearson)?;   // Interval<f64>
(ci.lower, ci.upper);                                // (0.2449, 0.9148)
proportion_interval(5, 8, 0.95, IntervalMethod::Wilson)?;
gold_screening(&labels, &gold, &q(2, 3))?;          // pass / fail per rater on the gold items
```

The Clopper–Pearson interval is the exact one (statsmodels
`proportion_confint(method='beta')`); Wilson, Agresti–Coull and Wald are
the usual approximations.  The proportion intervals are interval
*estimates*, so since 0.18 they live in `stats::estimation` beside the
mean intervals (the `stats::aggregation` paths still re-export them for
one release).  A rater's accuracy against chance is an exact binomial
test (below).

### Exact intervals

The `f64` function rounds; the same intervals exist with exact endpoints.
Wald, Wilson and Agresti–Coull are closed algebraic forms in the normal
quantile `z`, so they take any expression for `z` — a symbol for the
textbook formula, or `z_for_confidence(&ctx, &q(95, 100))` for the exact
`√2·erfinv(19/20)` (`1.959963984540054`).  Clopper–Pearson's endpoints
are the roots in `(0, 1)` of the two binomial-tail polynomials
`Σ_{j≥k} C(n,j) pʲ(1−p)ⁿ⁻ʲ − α/2` and `Σ_{j≤k} … − α/2`, which
`proportion_interval_exact` returns as `RootOf` algebraic numbers:

```rust,ignore
let z = ctx.symbol("z");
let ci = proportion_interval_symbolic(&ctx, 5, 8, &z, IntervalMethod::Wilson)?;   // Interval<Ex> in z, contains z^2

let ci = proportion_interval_exact(&ctx, 5, 8, &q(95, 100), IntervalMethod::ClopperPearson)?;
ci.lower;                          // RootOf(…degree-8 polynomial in _p…, k)
ci.lower.eval_f64()?;              // 0.2448632163665516   = scipy beta.ppf(0.025, 5, 4)
ci.upper.eval_f64()?;              // 0.9147665858627464   = scipy beta.ppf(0.975, 6, 3)
ci.lower.eval_decimal(30)?;        // as many digits as you like

proportion_interval_exact(&ctx, 1, 1, &q(9, 10), IntervalMethod::ClopperPearson)?;   // [1/20, 1]: linear tail, rational root
```

The exact closed forms are **not** clipped to `[0, 1]` (Agresti–Coull at
`k = 0` has a negative lower end that the `f64` function clamps), and the
Clopper–Pearson roots need a Sturm isolation and a factorisation of a
degree-`n` polynomial — fine for tens of trials, not thousands.  The
known-`σ` mean interval has the same pair,
`confidence_interval_mean_z_symbolic` / `_exact`, in the same module.

## Do two groups differ?

Response times of two groups, in seconds:

```rust,ignore
use symplex::stats::{data, hypothesis::*};
let fast = data::from_i64(&[12, 15, 11, 14, 13, 16, 10, 17]);
let slow = data::from_i64(&[18, 22, 19, 25, 20, 21, 23, 24]);
data::mean(&fast)?;                                          // 27/2
data::variance(&fast, data::Ddof::Sample)?;                  // 6
data::quantile(&slow, &q(3, 4), data::QuantileMethod::Inclusive)?;   // 93/4

let t = t_test_two_sample(&ctx, &fast, &slow, false, Alternative::TwoSided)?;   // Welch
t.statistic;          // exact: −(…)·√(…)   → −6.531972647421809
t.p_value;            // exact expression through betainc_regularized → 1.3298737271301488e-05
cohens_d(&ctx, &fast, &slow, true)?;

let u = mann_whitney_u(&ctx, &fast, &slow, Alternative::TwoSided, RankMethod::Exact)?;
u.statistic;          // 0
u.p_value;            // 1/6435 — the exact null distribution of U, as a rational
```

`TestResult { statistic, p_value, df, alternative }` carries exact
expressions; `p_value_f64()` / `statistic_f64()` round them, and
`p_value_exact()` returns the rational when there is one (the discrete
exact tests).  The family:

| Data | Tests |
|---|---|
| Two means | `t_test_one_sample`, `t_test_two_sample` (Student or Welch), `t_test_paired`; the t interval for a mean is `estimation::confidence_interval_mean(&x, 0.95)` |
| Several means | `anova::anova_one_way` → `AnovaResult { f, df_between, df_within, p_value, ss_between, ss_within, eta_squared }` (in `stats::anova` with the factorial and repeated-measures designs since 0.18) |
| Two proportions / one proportion | `z_test_proportion`, `z_test_two_proportions` (renamed from `two_proportion_z_test` in 0.18), `binomial_test` (exact) |
| Ranks / ordinal scores | `mann_whitney_u` (exact or asymptotic with tie correction), `wilcoxon_signed_rank`, `kruskal_wallis`, `friedman`, `spearman_test`, `kendall_test` |
| Categorical tables | `chi_square_independence` (with Yates), `chi_square_goodness_of_fit`, `g_test`, `fisher_exact` (exact up to a 2 000-point support, numeric above), `mcnemar_test` (exact or χ²), `sign_test`; `counts(&[&[i64]])` / `counts_usize(&[Vec<usize>])` build the table (the latter from a `confusion_matrix`) |
| Which cells drive a χ²? | `expected_counts`, `chi2_contributions`, `standardized_residuals`, `adjusted_residuals` (Haberman) |
| Correlation inference | `pearson_test`, `pearson_t_statistic`, `compare_two_correlations`; the Fisher-z interval is `estimation::pearson_ci` |
| Distribution fit | `ks_one_sample(x, &Distribution)` |
| Effect sizes | `cohens_d`, `hedges_g`, `glass_delta`, `rank_biserial`, `cliffs_delta`, `eta_squared`, `cramers_v`, `phi_coefficient`, `odds_ratio`, `relative_risk`, `cohens_h` |

Approval counts by group, `[[30, 10], [18, 22]]`: `fisher_exact` gives
`p = 0.01150621201656047` exactly as a rational, and
`chi_square_independence(…, true)` `p = 0.01205961617749023` as a
χ²-tail expression — both matching scipy to the last digit.  Fisher's
p-value is an exact rational while the hypergeometric support has at most
2 000 points; for larger tables (cells in the `10⁴`–`10¹¹` range) it is
numeric — the pmf walked from its mode in floating point, returned as
`ctx.from_f64(p)` and agreeing with scipy to `1e-9` — and a `[[10⁶, 10⁶ +
7], [10⁶ − 3, 10⁶]]` table takes milliseconds.  Below `1e-308` the
expression is `exp(ln p)`, so `p_value_log10` still reads the tail.

## Screening many raters at once

A rater who answered 14 of 20 gold questions correctly, against a chance
rate of ⅓: `binomial_test(&ctx, 14, 20, &q(1, 3), Alternative::Greater)`
has the **exact** p-value `1021403/1162261467` (≈ `8.79·10⁻⁴`).  Testing
many raters multiplies the false positives; `bonferroni`, `holm`,
`benjamini_hochberg` and `benjamini_yekutieli` return adjusted p-values
and reject flags (matching `statsmodels.multipletests`):

```rust,ignore
let adj = benjamini_hochberg(&[0.001, 0.02, 0.03, 0.2, 0.8], 0.05)?;
adj.reject;   // [true, true, true, false, false]
```

`bootstrap_ci` and `permutation_test` (seeded through `stats::Rng`, so
reproducible) cover statistics without a known null distribution;
`sample_size_for_proportion`, `sample_size_two_proportions`,
`power_t_test_two_sample` and `sample_size_t_test_two_sample` plan a study
(the t-power uses the noncentral t by quadrature and matches statsmodels).

## Estimating and modelling

`stats::estimation` fits distributions to data — `fit_normal`,
`fit_exponential`, `fit_poisson`, `fit_bernoulli`, `fit_geometric`,
`fit_uniform`, `fit_log_normal` by maximum likelihood (closed forms, exact
where the estimate is rational), `fit_gamma_moments`, `fit_beta_moments`,
`fit_negative_binomial_moments` by moments — and returns `Distribution`s,
so everything from the previous chapter (probabilities, quantiles,
sampling) applies to the fitted model.  `log_likelihood`, `aic`, `bic`
compare fits.  Conjugate Bayesian updating returns distributions too:
`beta_binomial_posterior(&ctx, &α, &β, successes, failures)` is the
posterior for a rater's accuracy after `s` right and `f` wrong answers,
`credible_interval(&post, 0.95)` its equal-tailed `Interval<f64>` (fields
`lower`, `upper`), and
`posterior_predictive_beta_binomial` the exact predictive table for the
next `n` answers.

Sequences of states — a respondent moving between "engaged", "guessing"
and "gone", say — are `stats::markov::MarkovChain` on an exact `QMatrix`:
`stationary_distribution()` is one exact linear solve (`[2/7, 3/7, 2/7]`
for the textbook 3-state chain), `absorption_probabilities`,
`expected_steps_to_absorption`, `expected_hitting_time`,
`communication_classes`, periods, and `sample_path`.

## Descriptive statistics, exactly

`stats::data` has the everyday summaries on exact observations: `mean`,
`variance(Ddof::{Population, Sample})`, `std`, `median`, `quantile` /
`quantiles` (`QuantileMethod::{Exclusive, Inclusive}` — Python's
`statistics.quantiles` default and `method='inclusive'`), `iqr`,
`modes`, `frequencies`, `ranks` (average ranks, as `scipy.stats.rankdata`),
`covariance`, `pearson`, `spearman`, `kendall_tau`, `skewness`, `kurtosis`,
`median_abs_deviation`, `zscores`, `geometric_mean`, `harmonic_mean`,
`trimmed_mean`, and the outlier screens `iqr_outliers` (Tukey's fences) and
`mad_outliers` (modified z-scores) for response times.  The ordinal
association measures live here too: `goodman_kruskal_gamma`,
`somers_d(x, y, Dependent::{Y, X, Symmetric})`, `kendall_tau_c` and the
`concordance_counts` they are built from.  `from_f64` converts floats
*exactly* (every `f64` is a dyadic rational), `to_f64` rounds back.

## Is the questionnaire itself reliable?

When each item is meant to measure the same thing, `stats::reliability`
asks whether they do — exactly.  With respondents as rows and items as
columns of a `RatingTable`:

| Question | Function |
|---|---|
| Internal consistency | `cronbach_alpha` (and `cronbach_alpha_complete` with list-wise deletion), `standardized_alpha`, `kr20` for right/wrong items, `guttman_lambda2`, `alpha_if_deleted` |
| Split-half reliability | `split_half(&table, &SplitHalf::{OddEven, FirstLast, Custom})` with the Spearman–Brown prophecy (`spearman_brown(&r, k)`, in the context of `r`) |
| Item quality | `item_difficulty`, `item_discrimination_index` (upper vs lower third), `point_biserial`, `item_total_correlation`, `corrected_item_total_correlation`, all bundled by `item_response_summary` |

Every coefficient that is a rational function of the scores is an exact
rational (Cronbach's α, KR-20); the ones with roots are exact
expressions.  Since 0.18 `stats::reliability` holds *only* scale
reliability and item analysis; its former neighbours moved to the module
their rule names (the old paths re-export them for one release):

| Question | Function (0.18 home) |
|---|---|
| Agreement inference | `agreement::{cohen_kappa_ci}` (Fleiss–Cohen–Everitt variance, exact), `kappa_test` (H₀: κ = 0), `cohen_kappa_maximum` (the κ the marginals allow), `cochrans_q` (many raters, binary items) |
| Ordinal association | `data::{goodman_kruskal_gamma, somers_d, kendall_tau_c, concordance_counts}` |
| Which cells drive a χ²? | `hypothesis::{expected_counts, chi2_contributions, standardized_residuals, adjusted_residuals}` |
| Correlation inference | `hypothesis::{pearson_test, pearson_t_statistic, compare_two_correlations}`; `estimation::{pearson_ci, fisher_z}` |

## Explaining accuracy or time by features

`stats::regression` fits models to the data you have about respondents:

```rust,ignore
use symplex::stats::regression::{ols, logit, LogitOpts, Design};
// Response time explained by two features, with an intercept — exactly.
let fit = ols(&y, &rows, true)?;                     // or Design::new().intercept().column(&x1).column(&x2).fit(&y)?
fit.coefficients;                                     // Vec<Q>, exact (XᵀX)⁻¹Xᵀy
fit.r_squared; fit.adjusted_r_squared;                // exact rationals
fit.standard_errors(&ctx)?;                           // exact expressions (√ of σ̂²(XᵀX)⁻¹)
fit.coefficient_tests(&ctx)?;                         // TestResult per coefficient, p through StudentT
fit.f_test(&ctx)?; fit.anova_table();                 // overall F, exact
fit.conf_int(0.95)?; fit.prediction_interval(&x_new, 0.95)?;   // Vec<Interval<f64>>, Interval<f64> — no ctx: the limits are f64
fit.leverage(); fit.cooks_distance(); fit.durbin_watson(); vif(&rows)?;
// Correct / incorrect explained by features: logistic regression (IRLS, f64).
let lg = logit(&correct, &features, true, &LogitOpts::default())?;
lg.coefficients; lg.odds_ratios(); lg.p_values; lg.pseudo_r_squared; lg.predict_proba(&x_new);
```

`ols`, `wls`, `simple_linear_regression` and `polyfit` are exact and match
statsmodels `OLS` attribute for attribute; `logit` matches `Logit` to
1e-6 and reports perfect separation as an error rather than as enormous
coefficients.

## Screening while the answers arrive

`stats::sequential::Sprt::bernoulli(&p0, &p1, α, β)` is Wald's sequential
probability ratio test: feed each gold-question outcome to `update`, and
the moment the exact log-likelihood ratio (`log_likelihood_ratio(&ctx)`)
crosses a boundary the decision is `AcceptH0` (the rater performs at the
chance rate `p0`) or `AcceptH1` (at the competent rate `p1`); until then
`Continue`.  The decision is sticky — once reached, every later `update`
returns it, `is_decided()` is `true` and `stopped_at()` gives the sample
size at which the test stopped; `reset()` starts over.
`expected_sample_size_bernoulli` says how many questions that takes on
average.

## Comparing label distributions

`stats::information` measures, exactly, how two raters' (or two
populations') label distributions differ: `kl_divergence`,
`js_divergence`, `total_variation`, `hellinger`, `bhattacharyya_distance`,
`cross_entropy`; and from a joint table how much one variable tells about
another: `mutual_information`, `conditional_entropy`,
`normalized_mutual_information` (`Norm::{Arithmetic, Geometric, Min,
Max}`), `joint_from_counts`.  Logs are kept as exact `ln` expressions
over prime factors, so `H(½, ¼, ¼)` is exactly `3/2` bits.

## Time to completion, time to attrition

`stats::survival` handles right-censored durations — how long until a
respondent finishes (or is still working when the study ends), how long a
worker stays active:

```rust,ignore
use symplex::stats::survival::{KaplanMeier, Observation, CiMethod, log_rank_test};
let obs = Observation::from_i64(&[3, 5, 6, 7, 8, 10, 12, 12], &[true, false, true, true, false, true, true, false]);
let km = KaplanMeier::fit(&obs)?;
km.survival_at(&q(6, 1));               // 35/48 — an exact step function
km.variance_at(&q(6, 1));               // Greenwood, 1505/55296
km.cumulative_hazard_at(&q(6, 1));      // Nelson–Aalen, 7/24
km.median();                            // Some(10)
km.confidence_interval(&q(7, 1), 0.95, CiMethod::LogLog)?;   // Interval<f64>
km.restricted_mean(&q(12, 1));          // 1279/144
let r = log_rank_test(&ctx, &all_obs, &groups)?;   // exact χ² statistic 149059681/48496587 ≈ 3.0736, p ≈ 0.0796
```

The Kaplan–Meier steps, Greenwood variances and Nelson–Aalen hazards are
exact rationals matching statsmodels' `SurvfuncRight`; the log-rank
statistic is an exact rational matching scipy's `logrank` and statsmodels'
`survdiff`.  `exponential_rate` is the censored MLE of a constant hazard,
and `survival_function` / `hazard_function` give `S(t)` and `h(t)` of any
`Distribution` as expressions.

## Several measures at once, and extremes

`stats::multivariate::MultivariateNormal` (symbolic mean vector and
covariance `Matrix`) has an exact density, marginals, conditionals by the
Schur complement, Mahalanobis distance and sampling; `covariance_matrix`
and `correlation_matrix` summarise several columns of ratings at once
(exact), and `pca` finds their principal components exactly
(`pca_f64` for larger matrices).  `stats::order::{order_statistic,
minimum_of, maximum_of}` give the distribution of the k-th smallest of `n`
independent draws as a `Distribution` in its own right — the fastest of
`n` responses, the worst of `n` ratings — exactly for both continuous
families and finite tables.
