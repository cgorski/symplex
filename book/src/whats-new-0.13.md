# What's New in 0.13 / 0.14

**0.14** continues on data: `stats::reliability` (Cronbach's α, KR-20,
split-half, item analysis, κ confidence intervals and tests, Cochran's Q,
Somers' D / Goodman–Kruskal γ, χ² residuals, Pearson inference),
`stats::regression` (exact OLS/WLS with the full inference table, logistic
regression by IRLS), `stats::survival` (Kaplan–Meier with Greenwood
variances, Nelson–Aalen, log-rank — exact), `stats::sequential` (Wald's
SPRT for screening as answers arrive), `stats::information` (KL, JS,
mutual information, exactly), `stats::multivariate` (multivariate normal,
covariance/correlation matrices, PCA) and `stats::order` (order statistics
as distributions).  See the guide's later sections.


**Statistics on data.**  0.12 made distributions compose; 0.13 turns to
the data people actually collect — many raters answering many items — and
answers the questions asked of it exactly.  See
[Analysing Rater and Response Data](guide/response-analysis.md) for the
walk-through.

- **Inter-rater agreement** (`stats::agreement`): percent agreement,
  Cohen's κ (plain and weighted), Scott's π, Fleiss' κ, Gwet's AC1,
  Krippendorff's α (nominal / ordinal / interval / ratio, with missing
  data), the six ICC forms, Kendall's W — every one an exact rational,
  matching statsmodels / the `krippendorff` package to the last digit.
- **Label aggregation** (`stats::aggregation`): majority and weighted
  votes, Dawid–Skene EM, Bradley–Terry, per-rater accuracy /
  precision / recall / F₁, exact Clopper–Pearson and Wilson intervals,
  gold-question screening.
- **Hypothesis tests** (`stats::hypothesis`): exact binomial, Fisher,
  McNemar and sign tests (p-values as rationals); t (Student, Welch,
  paired), z, one-way ANOVA; Mann–Whitney (exact null distribution or
  asymptotic), Wilcoxon, Kruskal–Wallis, Friedman, Spearman, Kendall,
  KS; χ² and G tests; effect sizes; Bonferroni / Holm / BH / BY;
  bootstrap and permutation; power and sample size.  Statistics are exact
  expressions and p-values exact expressions through the symbolic
  StudentT / χ² / F CDFs.
- **Estimation** (`stats::estimation`): MLE and method of moments for the
  standard families, log-likelihood / AIC / BIC, conjugate Bayesian
  posteriors (Beta–Binomial, Gamma–Poisson, Normal, Dirichlet) with
  credible intervals and exact predictive tables.
- **Markov chains** (`stats::markov`) on exact `QMatrix` transition
  matrices: stationary distributions, classes and periods, absorption
  probabilities and times, hitting probabilities and times.
- **Descriptive statistics** (`stats::data`), exact: moments, quantiles
  (both conventions), ranks, rank correlations, robust summaries and
  outlier screens.
- `Distribution::quantile_f64`: numeric inverse CDF for every family,
  through the exact CDF when it cannot be compiled.
