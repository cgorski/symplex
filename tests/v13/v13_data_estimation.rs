//! symplex 0.13 — data_estimation.  Reference values cite scipy 1.18.1 /
//! numpy 2.5.3 / SymPy 1.14.0 / Python `statistics` on `Fraction`s
//! (`symplex/.venv/bin/python`).  Every rational quantity is asserted
//! exactly; roots and logarithms are checked as expressions and to 1e-12.

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::data::{self, Ddof, Q, QuantileMethod};
use symplex::stats::estimation::{
    FamilyKind, aic, beta_binomial_posterior, bic, confidence_interval_mean_z, credible_interval,
    dirichlet_multinomial_posterior, dirichlet_posterior_alphas, fit, fit_bernoulli,
    fit_beta_moments, fit_binomial_p, fit_exponential, fit_gamma_moments, fit_geometric,
    fit_log_normal, fit_negative_binomial_moments, fit_normal, fit_poisson, fit_uniform,
    gamma_poisson_posterior, log_likelihood, method_of_moments, normal_known_variance_posterior,
    posterior_predictive_beta_binomial, standard_error_mean,
};
use symplex::stats::markov::MarkovChain;
use symplex::stats::{
    Bernoulli, Beta, Binomial, Distribution, Exponential, Gamma, Geometric, LogNormal,
    NegativeBinomial, Normal, Poisson, Rng, Uniform,
};

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "{label}: {actual} vs {expected}"
    );
}

fn ex_close(actual: &Ex, expected: f64, label: &str) {
    let v = actual
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: `{actual}` did not evaluate: {e}"));
    close(v, expected, label);
}

fn ints(v: &[i64]) -> Vec<Q> {
    data::from_i64(v)
}

/// `[1/2, 3/2, 2, 7/2, 5, 5]`: the descriptive-statistics sample.
fn sample_d() -> Vec<Q> {
    vec![q(1, 2), q(3, 2), qi(2), q(7, 2), qi(5), qi(5)]
}

/// `[2, 4, 4, 4, 5, 5, 7, 9]`: the classic textbook sample.
fn sample_d2() -> Vec<Q> {
    ints(&[2, 4, 4, 4, 5, 5, 7, 9])
}

fn qmatrix(rows: Vec<Vec<Q>>) -> QMatrix {
    QMatrix::new(rows).expect("well-formed matrix")
}

/// The 3-state regular chain `[[1/2, 1/2, 0], [1/3, 0, 2/3], [0, 1, 0]]`.
fn chain_three() -> MarkovChain {
    MarkovChain::new(qmatrix(vec![
        vec![q(1, 2), q(1, 2), qi(0)],
        vec![q(1, 3), qi(0), q(2, 3)],
        vec![qi(0), qi(1), qi(0)],
    ]))
    .expect("valid chain")
}

/// Gambler's ruin on `0..=4`, winning a round with probability `p`.
fn gambler(p: Q) -> MarkovChain {
    let lose = Q::from_integer(1.into()) - &p;
    let mut rows = vec![vec![qi(0); 5]; 5];
    rows[0][0] = qi(1);
    rows[4][4] = qi(1);
    for i in 1..4 {
        rows[i][i - 1] = lose.clone();
        rows[i][i + 1] = p.clone();
    }
    MarkovChain::new(qmatrix(rows)).expect("valid chain")
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::data — descriptive statistics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn data_mean_and_variances_are_exact() -> Result<(), SymplexError> {
    // statistics.mean/pvariance/variance on Fractions: 35/12, 425/144, 85/24
    let d = sample_d();
    assert_eq!(data::mean(&d)?, q(35, 12));
    assert_eq!(data::variance(&d, Ddof::Population)?, q(425, 144));
    assert_eq!(data::variance(&d, Ddof::Sample)?, q(85, 24));
    // statistics.pvariance([2,4,4,4,5,5,7,9]) = 4, variance = 32/7
    let d2 = sample_d2();
    assert_eq!(data::mean(&d2)?, qi(5));
    assert_eq!(data::variance(&d2, Ddof::Population)?, qi(4));
    assert_eq!(data::variance(&d2, Ddof::Sample)?, q(32, 7));
    let ctx = Context::new();
    assert_eq!(data::std(&ctx, &d2, Ddof::Population)?, ctx.int(2));
    Ok(())
}

#[test]
fn data_median_even_and_odd() -> Result<(), SymplexError> {
    // statistics.median([1/2, 3/2, 2, 7/2, 5, 5]) = 11/4; median([3, 1, 2]) = 2
    assert_eq!(data::median(&sample_d())?, q(11, 4));
    assert_eq!(data::median(&ints(&[3, 1, 2]))?, qi(2));
    assert_eq!(data::median(&[q(7, 3)])?, q(7, 3));
    Ok(())
}

#[test]
fn data_quantiles_both_methods_match_statistics() -> Result<(), SymplexError> {
    // statistics.quantiles(d, n=4) = [5/4, 11/4, 5]; method='inclusive' = [13/8, 11/4, 37/8]
    let d = sample_d();
    assert_eq!(
        data::quantiles(&d, 4, QuantileMethod::Exclusive)?,
        vec![q(5, 4), q(11, 4), qi(5)]
    );
    assert_eq!(
        data::quantiles(&d, 4, QuantileMethod::Inclusive)?,
        vec![q(13, 8), q(11, 4), q(37, 8)]
    );
    // numpy.quantile(d2, 0.75, method='weibull') = 6.5, method='linear' = 5.5
    let d2 = sample_d2();
    assert_eq!(
        data::quantile(&d2, &q(3, 4), QuantileMethod::Exclusive)?,
        q(13, 2)
    );
    assert_eq!(
        data::quantile(&d2, &q(3, 4), QuantileMethod::Inclusive)?,
        q(11, 2)
    );
    assert_eq!(data::iqr(&d2, QuantileMethod::Exclusive)?, q(5, 2));
    assert_eq!(data::iqr(&d2, QuantileMethod::Inclusive)?, q(3, 2));
    Ok(())
}

#[test]
fn data_quantile_extremes_and_singleton() -> Result<(), SymplexError> {
    let d2 = sample_d2();
    // numpy.quantile(d2, [0, 1], method='weibull'|'linear') = 2, 9
    for m in [QuantileMethod::Exclusive, QuantileMethod::Inclusive] {
        assert_eq!(data::quantile(&d2, &qi(0), m)?, qi(2));
        assert_eq!(data::quantile(&d2, &qi(1), m)?, qi(9));
        // numpy.quantile([7], 0.3, method=…) = 7
        assert_eq!(data::quantile(&[qi(7)], &q(3, 10), m)?, qi(7));
    }
    // numpy 'weibull' clamps at the extremes: quantile(d2, 0.1) = 2, quantile(d2, 0.9) = 9
    // (statistics.quantiles(n=10) extrapolates to 9/5 and 46/5 instead — documented).
    assert_eq!(
        data::quantile(&d2, &q(1, 10), QuantileMethod::Exclusive)?,
        qi(2)
    );
    assert_eq!(
        data::quantile(&d2, &q(9, 10), QuantileMethod::Exclusive)?,
        qi(9)
    );
    // statistics.quantiles(d2, n=10, method='inclusive')[0] = 17/5, [-1] = 38/5
    assert_eq!(
        data::quantile(&d2, &q(1, 10), QuantileMethod::Inclusive)?,
        q(17, 5)
    );
    assert_eq!(
        data::quantile(&d2, &q(9, 10), QuantileMethod::Inclusive)?,
        q(38, 5)
    );
    assert!(data::quantile(&d2, &q(3, 2), QuantileMethod::Inclusive).is_err());
    assert!(data::quantile(&[], &q(1, 2), QuantileMethod::Inclusive).is_err());
    Ok(())
}

#[test]
fn data_ranks_with_ties_match_rankdata() {
    // scipy.stats.rankdata([3,1,4,1,5,9,2,6]) = [4, 1.5, 5, 1.5, 6, 8, 3, 7]
    let x = ints(&[3, 1, 4, 1, 5, 9, 2, 6]);
    assert_eq!(
        data::ranks(&x),
        vec![qi(4), q(3, 2), qi(5), q(3, 2), qi(6), qi(8), qi(3), qi(7)]
    );
    // rankdata([2,7,1,8,2,8,1,8]) = [3.5, 5, 1.5, 7, 3.5, 7, 1.5, 7]
    let y = ints(&[2, 7, 1, 8, 2, 8, 1, 8]);
    assert_eq!(
        data::ranks(&y),
        vec![
            q(7, 2),
            qi(5),
            q(3, 2),
            qi(7),
            q(7, 2),
            qi(7),
            q(3, 2),
            qi(7)
        ]
    );
    assert_eq!(data::tie_sizes(&x), vec![2]);
    assert_eq!(data::tie_sizes(&y), vec![2, 2, 3]);
}

#[test]
fn data_spearman_without_and_with_ties() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.spearmanr([3,1,4,2,5], [2,4,1,5,3]).statistic = -0.6 = -3/5 (no ties: rational)
    let rho = data::spearman(&ctx, &ints(&[3, 1, 4, 2, 5]), &ints(&[2, 4, 1, 5, 3]))?;
    assert_eq!(rho, ctx.rational(-3, 5));
    // With ties: Pearson of the ranks; sxy = 8, sxx = 83/2, syy = 39 ⇒ ρ = 8/√(3237/2) = 8√6474/3237
    // scipy.stats.spearmanr(x, y).statistic = 0.19885368120992467
    let x = ints(&[3, 1, 4, 1, 5, 9, 2, 6]);
    let y = ints(&[2, 7, 1, 8, 2, 8, 1, 8]);
    let rho = data::spearman(&ctx, &x, &y)?;
    ex_close(&rho, 0.198_853_681_209_924_67, "spearman with ties");
    assert_eq!(rho.powi(2).simplify(), ctx.rational(128, 3237));
    Ok(())
}

#[test]
fn data_kendall_tau_b_without_and_with_ties() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.kendalltau([3,1,4,2,5], [2,4,1,5,3]).statistic = -0.4 = -2/5
    let tau = data::kendall_tau(&ctx, &ints(&[3, 1, 4, 2, 5]), &ints(&[2, 4, 1, 5, 3]))?;
    assert_eq!(tau, ctx.rational(-2, 5));
    // Ties: C = 13, D = 9, n0 = 28, n1 = 1, n2 = 5 ⇒ τ_b = 4/√(27·23) = 4/√621
    // scipy.stats.kendalltau(x, y).statistic = 0.16051447078102563
    let x = ints(&[3, 1, 4, 1, 5, 9, 2, 6]);
    let y = ints(&[2, 7, 1, 8, 2, 8, 1, 8]);
    let tau = data::kendall_tau(&ctx, &x, &y)?;
    ex_close(&tau, 0.160_514_470_781_025_63, "kendall tau-b with ties");
    assert_eq!(tau.powi(2).simplify(), ctx.rational(16, 621));
    Ok(())
}

#[test]
fn data_skewness_and_kurtosis_biased() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = sample_d();
    // m2 = 425/144, m3 = 5/54, m4 = 88283/6912 (Fraction arithmetic)
    assert_eq!(data::central_moment(&d, 2)?, q(425, 144));
    assert_eq!(data::central_moment(&d, 3)?, q(5, 54));
    assert_eq!(data::central_moment(&d, 4)?, q(88283, 6912));
    // scipy.stats.skew(d, bias=True) = 0.01826150588508888; skew² = 1024/3070625 exactly
    let skew = data::skewness(&ctx, &d)?;
    ex_close(&skew, 0.018_261_505_885_088_88, "skewness");
    assert_eq!(skew.powi(2).simplify(), ctx.rational(1024, 3_070_625));
    // scipy.stats.kurtosis(d, fisher=True, bias=True) = -1.5337079584775093 = -277026/180625
    let kurt = data::kurtosis(&d)?;
    assert_eq!(kurt, q(-277_026, 180_625));
    close(
        data::to_f64(&[kurt])[0],
        -1.533_707_958_477_509_3,
        "kurtosis",
    );
    Ok(())
}

#[test]
fn data_median_abs_deviation_and_zscores() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = sample_d();
    // scipy.stats.median_abs_deviation(d) = 1.75 = 7/4
    assert_eq!(data::median_abs_deviation(&d)?, q(7, 4));
    // scipy.stats.zscore(d, ddof=0)[0] = -1.4067066252107312 (z₀² = 841/425);
    // ddof=1: -1.284141584033138 (z₀² = 841/510)
    let z0 = data::zscores(&ctx, &d, Ddof::Population)?;
    let z1 = data::zscores(&ctx, &d, Ddof::Sample)?;
    assert_eq!(z0.len(), 6);
    ex_close(&z0[0], -1.406_706_625_210_731_2, "z[0] population");
    ex_close(&z0[3], 0.339_549_875_050_866_26, "z[3] population");
    ex_close(&z1[0], -1.284_141_584_033_138, "z[0] sample");
    ex_close(&z1[5], 1.107_018_606_925_119, "z[5] sample");
    assert_eq!(z0[0].powi(2).simplify(), ctx.rational(841, 425));
    assert_eq!(z1[0].powi(2).simplify(), ctx.rational(841, 510));
    Ok(())
}

#[test]
fn data_geometric_harmonic_and_trimmed_means() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = sample_d();
    // scipy.stats.gmean(d) = 2.254325128297661; Π = 525/4 so gmean⁶ = 525/4
    let g = data::geometric_mean(&ctx, &d)?;
    ex_close(&g, 2.254_325_128_297_661, "geometric mean");
    assert_eq!(g.powi(6).simplify(), ctx.rational(525, 4));
    // statistics.harmonic_mean(d) = 1260/809; scipy.stats.hmean(d) = 1.557478368355995
    let h = data::harmonic_mean(&d)?;
    assert_eq!(h, q(1260, 809));
    close(
        data::to_f64(&[h])[0],
        1.557_478_368_355_995,
        "harmonic mean",
    );
    // scipy.stats.trim_mean([1..9, 100], 0.1) = 5.5, (…, 0.2) = 5.5
    let t = ints(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
    assert_eq!(data::trimmed_mean(&t, &q(1, 10))?, q(11, 2));
    assert_eq!(data::trimmed_mean(&t, &q(1, 5))?, q(11, 2));
    // scipy.stats.trim_mean([1,3,4,8,15,27,50,100], 0.25) = 13.5; (…, 0.1) = 26 (⌊0.8⌋ = 0 cut)
    let t2 = ints(&[1, 3, 4, 8, 15, 27, 50, 100]);
    assert_eq!(data::trimmed_mean(&t2, &q(1, 4))?, q(27, 2));
    assert_eq!(data::trimmed_mean(&t2, &q(1, 10))?, qi(26));
    assert!(data::trimmed_mean(&t2, &q(1, 2)).is_err());
    Ok(())
}

#[test]
fn data_outlier_detectors_flag_the_planted_point() -> Result<(), SymplexError> {
    let o = ints(&[10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 10, 100]);
    // numpy weibull Q1 = 10, Q3 = 13.5 ⇒ fences 4.75, 18.75; linear Q1 = 10, Q3 = 13 ⇒ 5.5, 17.5
    assert_eq!(
        data::iqr_outliers(&o, &q(3, 2), QuantileMethod::Exclusive)?,
        vec![12]
    );
    assert_eq!(
        data::iqr_outliers(&o, &q(3, 2), QuantileMethod::Inclusive)?,
        vec![12]
    );
    // median 12, MAD 2: modified z of 100 is 0.6745·88/2 = 29.678; every other |z| ≤ 1.01175
    assert_eq!(data::median(&o)?, qi(12));
    assert_eq!(data::median_abs_deviation(&o)?, qi(2));
    assert_eq!(data::mad_outliers(&o, &q(7, 2))?, vec![12]);
    // A tighter threshold of 1 also catches 15 (z = 1.01175), nothing else.
    assert_eq!(data::mad_outliers(&o, &qi(1))?, vec![8, 12]);
    // Constant data: MAD = 0 is rejected.
    assert!(data::mad_outliers(&ints(&[3, 3, 3]), &q(7, 2)).is_err());
    Ok(())
}

#[test]
fn data_from_f64_is_exact_and_to_f64_round_trips() -> Result<(), SymplexError> {
    // Fraction(0.1) = 3602879701896397/36028797018963968; Fraction(0.25) = 1/4; Fraction(-1.5) = -3/2
    let v = data::from_f64(&[0.1, 0.25, -1.5])?;
    assert_eq!(
        v[0],
        Q::new(
            "3602879701896397".parse().expect("literal"),
            "36028797018963968".parse().expect("literal")
        )
    );
    assert_eq!(v[1], q(1, 4));
    assert_eq!(v[2], q(-3, 2));
    assert_eq!(data::to_f64(&v), vec![0.1, 0.25, -1.5]);
    assert!(data::from_f64(&[f64::NAN]).is_err());
    assert!(data::from_f64(&[f64::INFINITY]).is_err());
    assert_eq!(data::from_i64(&[-2, 7]), vec![qi(-2), qi(7)]);
    Ok(())
}

#[test]
fn data_modes_frequencies_and_min_max() -> Result<(), SymplexError> {
    // statistics.multimode([1,2,2,3,3]) = [2, 3]
    let d = ints(&[1, 2, 2, 3, 3]);
    assert_eq!(data::modes(&d), vec![qi(2), qi(3)]);
    assert_eq!(
        data::frequencies(&d),
        vec![(qi(1), 1), (qi(2), 2), (qi(3), 2)]
    );
    assert_eq!(data::min_max(&sample_d())?, (q(1, 2), qi(5)));
    assert!(data::modes(&[]).is_empty());
    Ok(())
}

#[test]
fn data_error_cases() {
    let ctx = Context::new();
    let empty: Vec<Q> = vec![];
    let constant = ints(&[4, 4, 4]);
    assert!(data::mean(&empty).is_err());
    assert!(data::median(&empty).is_err());
    assert!(data::variance(&empty, Ddof::Population).is_err());
    assert!(data::variance(&[qi(1)], Ddof::Sample).is_err());
    assert!(data::variance(&[qi(1)], Ddof::Population).is_ok());
    assert!(data::covariance(&ints(&[1, 2]), &ints(&[1, 2, 3]), Ddof::Sample).is_err());
    assert!(data::pearson(&ctx, &constant, &ints(&[1, 2, 3])).is_err());
    assert!(data::spearman(&ctx, &constant, &ints(&[1, 2, 3])).is_err());
    assert!(data::kendall_tau(&ctx, &constant, &ints(&[1, 2, 3])).is_err());
    assert!(data::kendall_tau(&ctx, &[qi(1)], &[qi(2)]).is_err());
    assert!(data::skewness(&ctx, &constant).is_err());
    assert!(data::kurtosis(&constant).is_err());
    assert!(data::zscores(&ctx, &constant, Ddof::Population).is_err());
    assert!(data::geometric_mean(&ctx, &ints(&[1, 0, 2])).is_err());
    assert!(data::harmonic_mean(&empty).is_err());
    assert!(data::harmonic_mean(&ints(&[1, -1])).is_err());
    assert!(data::quantiles(&sample_d(), 1, QuantileMethod::Inclusive).is_err());
    assert!(data::min_max(&empty).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::estimation — maximum likelihood
// ═══════════════════════════════════════════════════════════════════════════

/// `[2, 3.5, 4, 5.5, 7]`: the normal sample.
fn normal_data() -> Vec<Q> {
    vec![qi(2), q(7, 2), qi(4), q(11, 2), qi(7)]
}

#[test]
fn mle_normal_matches_scipy_norm_fit() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.norm.fit([2, 3.5, 4, 5.5, 7]) = (4.4, 1.7146428199482247); σ̂² = 2.94 = 147/50
    let d = fit_normal(&ctx, &normal_data())?;
    let n = d.downcast_ref::<Normal>().expect("Normal");
    assert_eq!(n.mean, ctx.rational(22, 5));
    assert_eq!(n.std.powi(2).simplify(), ctx.rational(147, 50));
    ex_close(&n.std, 1.714_642_819_948_224_7, "σ̂");
    assert_eq!(d.variance().simplify(), ctx.rational(147, 50));
    Ok(())
}

#[test]
fn mle_exponential_poisson_bernoulli() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.expon.fit([1,2,3,6], floc=0) = (0, 3.0) ⇒ λ̂ = 1/3
    let e = fit_exponential(&ctx, &ints(&[1, 2, 3, 6]))?;
    assert_eq!(
        e.downcast_ref::<Exponential>().expect("Exponential").rate,
        ctx.rational(1, 3)
    );
    assert_eq!(e.mean(), ctx.int(3));
    // λ̂ = 12/6 = 2
    let p = fit_poisson(&ctx, &ints(&[0, 1, 1, 2, 3, 5]))?;
    assert_eq!(
        p.downcast_ref::<Poisson>().expect("Poisson").rate,
        ctx.int(2)
    );
    // p̂ = 5/7
    let b = fit_bernoulli(&ctx, &ints(&[1, 0, 1, 1, 0, 1, 1]))?;
    assert_eq!(
        b.downcast_ref::<Bernoulli>().expect("Bernoulli").p,
        ctx.rational(5, 7)
    );
    Ok(())
}

#[test]
fn mle_binomial_geometric_uniform() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // n = 10 trials, counts [3, 5, 7, 4]: p̂ = 19/40
    let b = fit_binomial_p(&ctx, 10, &ints(&[3, 5, 7, 4]))?;
    let bin = b.downcast_ref::<Binomial>().expect("Binomial");
    assert_eq!(
        (bin.n.clone(), bin.p.clone()),
        (ctx.int(10), ctx.rational(19, 40))
    );
    // Geometric (trials to first success) [1, 3, 2, 6]: p̂ = 4/12 = 1/3
    let g = fit_geometric(&ctx, &ints(&[1, 3, 2, 6]))?;
    assert_eq!(
        g.downcast_ref::<Geometric>().expect("Geometric").p,
        ctx.rational(1, 3)
    );
    // scipy.stats.uniform.fit([2.5, 1, 4, 3]) = (loc 1.0, scale 3.0) ⇒ [1, 4]
    let u = fit_uniform(&ctx, &[q(5, 2), qi(1), qi(4), qi(3)])?;
    let uni = u.downcast_ref::<Uniform>().expect("Uniform");
    assert_eq!((uni.lo.clone(), uni.hi.clone()), (ctx.int(1), ctx.int(4)));
    Ok(())
}

#[test]
fn mle_log_normal_is_the_normal_fit_of_the_logs() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // ln [1,2,4,8] = [0,1,2,3]·ln 2 ⇒ μ̂ = (3/2) ln 2, σ̂² = (5/4) ln² 2
    // scipy.stats.lognorm.fit([1,2,4,8], floc=0) = (s=0.7749621070721792, 0, scale=2.82842712474619),
    // ln(scale) = 1.0397207708399179
    let d = fit_log_normal(&ctx, &ints(&[1, 2, 4, 8]))?;
    let l = d.downcast_ref::<LogNormal>().expect("LogNormal");
    ex_close(&l.mu, 1.039_720_770_839_917_9, "μ̂");
    ex_close(&l.sigma, 0.774_962_107_072_179_2, "σ̂");
    let ln2 = ctx.int(2).ln();
    assert_eq!(l.mu, (ctx.rational(3, 2) * &ln2).simplify());
    assert_eq!(
        l.sigma
            .powi(2)
            .simplify()
            .equals(&(ctx.rational(5, 4) * ln2.powi(2))),
        Some(true)
    );
    Ok(())
}

#[test]
fn mle_log_normal_expands_logs_over_prime_factors() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let ln2 = ctx.int(2).ln();
    // scipy.stats.lognorm.fit([0.5, 2], floc=0) = (s = 0.6931471805599453 = ln 2, 0, scale = 1 ⇒ μ̂ = 0)
    let d = fit_log_normal(&ctx, &[q(1, 2), qi(2)])?;
    let l = d.downcast_ref::<LogNormal>().expect("LogNormal");
    assert_eq!(l.mu, ctx.int(0));
    assert_eq!(l.sigma, ln2);
    // [3/4, 3, 12]: ln = (ln 3 − 2 ln 2, ln 3, ln 3 + 2 ln 2) ⇒ μ̂ = ln 3, σ̂² = (8/3) ln² 2
    // scipy.stats.lognorm.fit([0.75, 3, 12], floc=0) = (1.1319046060137772, 0, 3.0000000000000004)
    let d = fit_log_normal(&ctx, &[q(3, 4), qi(3), qi(12)])?;
    let l = d.downcast_ref::<LogNormal>().expect("LogNormal");
    assert_eq!(l.mu, ctx.int(3).ln());
    ex_close(&l.sigma, 1.131_904_606_013_777_2, "σ̂");
    assert_eq!(
        l.sigma
            .powi(2)
            .simplify()
            .equals(&(ctx.rational(8, 3) * ln2.powi(2))),
        Some(true)
    );
    Ok(())
}

#[test]
fn method_of_moments_gamma_beta_negative_binomial() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // [1,2,3,6]: x̄ = 3, s² = 7/2 ⇒ k̂ = 18/7, θ̂ = 7/6 (Fraction arithmetic)
    let g = fit_gamma_moments(&ctx, &ints(&[1, 2, 3, 6]))?;
    let gam = g.downcast_ref::<Gamma>().expect("Gamma");
    assert_eq!(
        (gam.shape.clone(), gam.scale.clone()),
        (ctx.rational(18, 7), ctx.rational(7, 6))
    );
    // The fitted law reproduces the data's moments exactly.
    assert_eq!(g.mean(), ctx.int(3));
    assert_eq!(g.variance().simplify(), ctx.rational(7, 2));
    // [1/4,1/2,1/2,3/4]: x̄ = 1/2, s² = 1/32 ⇒ α̂ = β̂ = 7/2
    let b = fit_beta_moments(&ctx, &[q(1, 4), q(1, 2), q(1, 2), q(3, 4)])?;
    let bet = b.downcast_ref::<Beta>().expect("Beta");
    assert_eq!(
        (bet.alpha.clone(), bet.beta.clone()),
        (ctx.rational(7, 2), ctx.rational(7, 2))
    );
    assert_eq!(b.variance().simplify(), ctx.rational(1, 32));
    // [0,1,2,5,12]: x̄ = 4, s² = 94/5 ⇒ r̂ = 40/37, p̂ = 10/47
    let nb = fit_negative_binomial_moments(&ctx, &ints(&[0, 1, 2, 5, 12]))?;
    let neg = nb
        .downcast_ref::<NegativeBinomial>()
        .expect("NegativeBinomial");
    assert_eq!(
        (neg.r.clone(), neg.p.clone()),
        (ctx.rational(40, 37), ctx.rational(10, 47))
    );
    assert_eq!(nb.mean(), ctx.int(4));
    assert_eq!(nb.variance().simplify(), ctx.rational(94, 5));
    Ok(())
}

#[test]
fn method_of_moments_uniform_and_log_normal() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Uniform on [1,2,3,6]: x̄ = 3, 3s² = 21/2 ⇒ a, b = 3 ∓ √(21/2) = -0.2403703492039302, 6.24037034920393
    let u = method_of_moments(&ctx, FamilyKind::Uniform, &ints(&[1, 2, 3, 6]))?;
    let uni = u.downcast_ref::<Uniform>().expect("Uniform");
    ex_close(&uni.lo, -0.240_370_349_203_930_2, "a");
    ex_close(&uni.hi, 6.240_370_349_203_93, "b");
    assert_eq!(u.mean().simplify(), ctx.int(3));
    assert_eq!(u.variance().simplify(), ctx.rational(7, 2));
    // LogNormal on [1,2,4,8]: x̄ = 15/4, s² = 115/16, 1 + s²/x̄² = 68/45
    // σ² = ln(68/45) = 0.4128452154057869, σ = 0.6425303225574548, μ = ln(15/4) − σ²/2 = 1.115333232279426
    let l = method_of_moments(&ctx, FamilyKind::LogNormal, &ints(&[1, 2, 4, 8]))?;
    let ln = l.downcast_ref::<LogNormal>().expect("LogNormal");
    ex_close(&ln.sigma, 0.642_530_322_557_454_8, "σ");
    ex_close(&ln.mu, 1.115_333_232_279_426, "μ");
    assert_eq!(
        ln.sigma
            .powi(2)
            .simplify()
            .equals(&ctx.rational(68, 45).ln()),
        Some(true)
    );
    Ok(())
}

#[test]
fn fit_dispatches_by_family_and_rejects_closed_form_less_mles() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = normal_data();
    assert_eq!(fit(&ctx, FamilyKind::Normal, &d)?, fit_normal(&ctx, &d)?);
    assert_eq!(
        method_of_moments(&ctx, FamilyKind::Normal, &d)?,
        fit_normal(&ctx, &d)?
    );
    let counts = ints(&[0, 1, 1, 2, 3, 5]);
    assert_eq!(
        fit(&ctx, FamilyKind::Poisson, &counts)?,
        fit_poisson(&ctx, &counts)?
    );
    assert_eq!(
        fit(&ctx, FamilyKind::Binomial { n: 10 }, &ints(&[3, 5, 7, 4]))?,
        fit_binomial_p(&ctx, 10, &ints(&[3, 5, 7, 4]))?
    );
    for family in [
        FamilyKind::Gamma,
        FamilyKind::Beta,
        FamilyKind::NegativeBinomial,
    ] {
        let err = fit(&ctx, family, &d).expect_err("no closed form");
        assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
        assert!(err.to_string().contains("method_of_moments"), "{err}");
    }
    assert_eq!(
        method_of_moments(&ctx, FamilyKind::Gamma, &ints(&[1, 2, 3, 6]))?,
        fit_gamma_moments(&ctx, &ints(&[1, 2, 3, 6]))?
    );
    Ok(())
}

#[test]
fn fit_error_cases() {
    let ctx = Context::new();
    let empty: Vec<Q> = vec![];
    assert!(fit_normal(&ctx, &empty).is_err());
    assert!(fit_normal(&ctx, &ints(&[3, 3, 3])).is_err());
    assert!(fit_exponential(&ctx, &ints(&[1, -2])).is_err());
    assert!(fit_exponential(&ctx, &ints(&[0, 2])).is_err());
    assert!(fit_poisson(&ctx, &[q(1, 2)]).is_err());
    assert!(fit_poisson(&ctx, &ints(&[0, 0])).is_err());
    assert!(fit_bernoulli(&ctx, &ints(&[0, 1, 2])).is_err());
    assert!(fit_binomial_p(&ctx, 3, &ints(&[1, 4])).is_err());
    assert!(fit_binomial_p(&ctx, 0, &ints(&[0])).is_err());
    assert!(fit_geometric(&ctx, &ints(&[0, 2])).is_err());
    assert!(fit_uniform(&ctx, &ints(&[5, 5])).is_err());
    assert!(fit_log_normal(&ctx, &ints(&[1, 0])).is_err());
    assert!(fit_log_normal(&ctx, &ints(&[2, 2])).is_err());
    assert!(fit_gamma_moments(&ctx, &ints(&[2, 2])).is_err());
    assert!(fit_beta_moments(&ctx, &[q(1, 2), qi(1)]).is_err());
    assert!(fit_beta_moments(&ctx, &[q(1, 2), q(1, 2)]).is_err());
    // Data strictly inside (0, 1) always satisfy s² < x̄(1 − x̄) (Bhatia–Davis), so even the
    // extreme pair fits: Fraction arithmetic gives c = 99/2401, α̂ = β̂ = 99/4802.
    let extreme = fit_beta_moments(&ctx, &[q(1, 100), q(99, 100)]).expect("fits");
    let b = extreme.downcast_ref::<Beta>().expect("Beta");
    assert_eq!(
        (b.alpha.clone(), b.beta.clone()),
        (ctx.rational(99, 4802), ctx.rational(99, 4802))
    );
    // Under-dispersed counts: s² ≤ x̄.
    assert!(fit_negative_binomial_moments(&ctx, &ints(&[2, 2, 3, 3])).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::estimation — likelihood, information criteria
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_likelihood_numeric_matches_scipy() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // scipy.stats.norm.logpdf(data, 4.4, 1.7146428199482247).sum() = -9.79071661939984
    let d = normal_data();
    let normal = fit_normal(&ctx, &d)?;
    ex_close(
        &log_likelihood(&normal, &d),
        -9.790_716_619_399_84,
        "normal ℓ",
    );
    // scipy.stats.expon.logpdf([1,2,3,6], scale=3).sum() = -8.39444915467244 = 4 ln(1/3) − 4
    let ell = log_likelihood(
        &Distribution::exponential(ctx.rational(1, 3)),
        &ints(&[1, 2, 3, 6]),
    );
    ex_close(&ell, -8.394_449_154_672_44, "exponential ℓ");
    assert_eq!(
        ell.equals(&(ctx.int(4) * ctx.rational(1, 3).ln() - ctx.int(4)).simplify()),
        Some(true)
    );
    // scipy.stats.poisson.logpmf([0,1,1,2,3,5], 2).sum() = -10.954632225850702
    let ell = log_likelihood(
        &Distribution::poisson(ctx.int(2)),
        &ints(&[0, 1, 1, 2, 3, 5]),
    );
    ex_close(&ell, -10.954_632_225_850_702, "poisson ℓ");
    Ok(())
}

#[test]
fn log_likelihood_symbolic_score_vanishes_at_the_mle() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let d = normal_data();
    let mu = ctx.symbol("mu");
    let sigma_hat = ctx.rational(147, 50).sqrt();
    let ell = log_likelihood(&Distribution::normal(mu.clone(), sigma_hat), &d);
    let score = ell.diff(&mu).simplify();
    // dℓ/dμ = Σ (xᵢ − μ)/σ² vanishes at μ̂ = 22/5 …
    assert_eq!(score.subs(&mu, &ctx.rational(22, 5)).simplify(), ctx.int(0));
    // … and solving the score equation recovers the MLE.
    let roots = score.solve(&mu)?;
    assert_eq!(roots, vec![ctx.rational(22, 5)]);
    // Second derivative is −n/σ² < 0: a maximum.
    assert_eq!(
        ell.diff(&mu).diff(&mu).simplify(),
        (ctx.int(-5) / ctx.rational(147, 50)).simplify()
    );
    Ok(())
}

#[test]
fn aic_and_bic_from_the_log_likelihood() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // ℓ = -9.79071661939984, k = 2, n = 5: AIC = 4 − 2ℓ = 23.58143323879968,
    // BIC = 2 ln 5 − 2ℓ = 22.80030906366788
    let d = normal_data();
    let ell = log_likelihood(&fit_normal(&ctx, &d)?, &d);
    ex_close(&aic(&ell, 2), 23.581_433_238_799_68, "AIC");
    ex_close(&bic(&ell, 2, 5), 22.800_309_063_667_88, "BIC");
    // Exact structure: AIC − BIC = 2k − k ln n.
    assert_eq!(
        (aic(&ell, 2) - bic(&ell, 2, 5)).simplify(),
        (ctx.int(4) - ctx.int(2) * ctx.int(5).ln()).simplify()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::estimation — Bayesian conjugate updating
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn beta_binomial_posterior_and_credible_interval() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Beta(2, 3) + 7 successes, 3 failures = Beta(9, 6); mean 9/15 = 3/5
    let post = beta_binomial_posterior(&ctx, (&qi(2), &qi(3)), 7, 3)?;
    let b = post.downcast_ref::<Beta>().expect("Beta");
    assert_eq!((b.alpha.clone(), b.beta.clone()), (ctx.int(9), ctx.int(6)));
    assert_eq!(post.mean(), ctx.rational(3, 5));
    // scipy.stats.beta.ppf([0.025, 0.975], 9, 6) = (0.3513801106159917, 0.8233889100178821)
    let (lo, hi) = credible_interval(&post, 0.95)?;
    assert!((lo - 0.351_380_110_615_991_7).abs() < 1e-9, "{lo}");
    assert!((hi - 0.823_388_910_017_882_1).abs() < 1e-9, "{hi}");
    assert!(credible_interval(&post, 1.0).is_err());
    assert!(credible_interval(&post, 0.0).is_err());
    // Fractional prior, no data: unchanged.
    let same = beta_binomial_posterior(&ctx, (&q(1, 2), &q(1, 2)), 0, 0)?;
    assert_eq!(
        same,
        Distribution::beta(ctx.rational(1, 2), ctx.rational(1, 2))
    );
    assert!(beta_binomial_posterior(&ctx, (&qi(0), &qi(1)), 1, 1).is_err());
    Ok(())
}

#[test]
fn gamma_poisson_posterior_in_shape_scale_form() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Gamma(shape 2, scale 1/2) + counts [3, 5, 4]: shape 2 + 12 = 14, scale (1/2)/(1 + 3/2) = 1/5
    let post = gamma_poisson_posterior(&ctx, (&qi(2), &q(1, 2)), &[3, 5, 4])?;
    let g = post.downcast_ref::<Gamma>().expect("Gamma");
    assert_eq!(
        (g.shape.clone(), g.scale.clone()),
        (ctx.int(14), ctx.rational(1, 5))
    );
    assert_eq!(post.mean(), ctx.rational(14, 5));
    // No counts: the prior is returned unchanged.
    assert_eq!(
        gamma_poisson_posterior(&ctx, (&qi(2), &q(1, 2)), &[])?,
        Distribution::gamma(ctx.int(2), ctx.rational(1, 2))
    );
    assert!(gamma_poisson_posterior(&ctx, (&qi(2), &qi(0)), &[1]).is_err());
    Ok(())
}

#[test]
fn normal_known_variance_posterior_is_precision_weighted() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Prior N(0, 1), σ = 2, data [1, 2, 3]: τ = 1 + 3/4 = 7/4, mean (0 + 6/4)/(7/4) = 6/7, var 4/7
    let post = normal_known_variance_posterior(&ctx, (&qi(0), &qi(1)), &qi(2), &ints(&[1, 2, 3]))?;
    let n = post.downcast_ref::<Normal>().expect("Normal");
    assert_eq!(n.mean, ctx.rational(6, 7));
    assert_eq!(n.std.powi(2).simplify(), ctx.rational(4, 7));
    assert_eq!(post.variance().simplify(), ctx.rational(4, 7));
    assert!(normal_known_variance_posterior(&ctx, (&qi(0), &qi(1)), &qi(2), &[]).is_err());
    assert!(normal_known_variance_posterior(&ctx, (&qi(0), &qi(0)), &qi(2), &[qi(1)]).is_err());
    Ok(())
}

#[test]
fn dirichlet_multinomial_posterior_mean() -> Result<(), SymplexError> {
    // Flat prior (1,1,1) + counts (3,2,5) = (4,3,6); mean (4/13, 3/13, 6/13)
    assert_eq!(
        dirichlet_posterior_alphas(&[qi(1), qi(1), qi(1)], &[3, 2, 5])?,
        vec![qi(4), qi(3), qi(6)]
    );
    let m = dirichlet_multinomial_posterior(&[qi(1), qi(1), qi(1)], &[3, 2, 5])?;
    assert_eq!(m, vec![q(4, 13), q(3, 13), q(6, 13)]);
    assert_eq!(data::sum(&m), qi(1));
    // Fractional prior (1/2, 1/2) + (0, 3): (1/2, 7/2)/4 = (1/8, 7/8)
    assert_eq!(
        dirichlet_multinomial_posterior(&[q(1, 2), q(1, 2)], &[0, 3])?,
        vec![q(1, 8), q(7, 8)]
    );
    assert!(dirichlet_multinomial_posterior(&[], &[]).is_err());
    assert!(dirichlet_multinomial_posterior(&[qi(1)], &[1, 2]).is_err());
    assert!(dirichlet_multinomial_posterior(&[qi(0), qi(1)], &[1, 2]).is_err());
    Ok(())
}

#[test]
fn posterior_predictive_beta_binomial_masses_are_exact() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // sympy: binomial(4,k)·B(k+2, 4−k+3)/B(2,3) = [3/14, 2/7, 9/35, 6/35, 1/14];
    // scipy.stats.betabinom.pmf(range(5), 4, 2, 3) = [0.2142857…, 0.2857142…, 0.2571428…, 0.1714285…, 0.0714285…]
    let pred = posterior_predictive_beta_binomial(&ctx, &qi(2), &qi(3), 4)?;
    assert_eq!(pred.name(), "Finite");
    let expected = [q(3, 14), q(2, 7), q(9, 35), q(6, 35), q(1, 14)];
    for (k, e) in expected.iter().enumerate() {
        assert_eq!(
            pred.density(&ctx.int(k as i64)).simplify(),
            ctx.from_ratio(e.clone())
        );
    }
    close(
        data::to_f64(&expected[2..3])[0],
        0.257_142_857_142_857_34,
        "pmf(2)",
    );
    // Beta-binomial mean nα/(α+β) = 8/5 and variance nαβ(α+β+n)/((α+β)²(α+β+1)) = 4·6·9/(25·6) = 36/25
    assert_eq!(pred.mean(), ctx.rational(8, 5));
    assert_eq!(pred.variance(), ctx.rational(36, 25));
    // n = 0: the point mass at 0.
    let none = posterior_predictive_beta_binomial(&ctx, &q(1, 2), &q(3, 2), 0)?;
    assert_eq!(none.density(&ctx.int(0)).simplify(), ctx.int(1));
    assert!(posterior_predictive_beta_binomial(&ctx, &qi(0), &qi(1), 2).is_err());
    Ok(())
}

#[test]
fn standard_error_and_z_confidence_interval() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // σ = 2, n = 3: se = 2/√3 = 1.1547005383792517
    let se = standard_error_mean(&ctx, &qi(2), 3)?;
    ex_close(&se, 1.154_700_538_379_251_7, "se");
    assert_eq!(se.powi(2).simplify(), ctx.rational(4, 3));
    // scipy.stats.norm.interval(0.95, loc=2, scale=2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
    let (lo, hi) = confidence_interval_mean_z(&ctx, &ints(&[1, 2, 3]), &qi(2), 0.95)?;
    assert!((lo + 0.263_171_468_152_343_8).abs() < 1e-9, "{lo}");
    assert!((hi - 4.263_171_468_152_343).abs() < 1e-9, "{hi}");
    assert!(standard_error_mean(&ctx, &qi(2), 0).is_err());
    assert!(standard_error_mean(&ctx, &qi(0), 3).is_err());
    assert!(confidence_interval_mean_z(&ctx, &[], &qi(2), 0.95).is_err());
    assert!(confidence_interval_mean_z(&ctx, &ints(&[1]), &qi(2), 1.0).is_err());
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// stats::markov — discrete-time Markov chains
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn markov_construction_validates_the_matrix_and_labels() {
    let not_square = qmatrix(vec![vec![q(1, 2), q(1, 2)]]);
    assert!(MarkovChain::new(not_square).is_err());
    let negative = qmatrix(vec![vec![q(3, 2), q(-1, 2)], vec![qi(0), qi(1)]]);
    assert!(MarkovChain::new(negative).is_err());
    let bad_sum = qmatrix(vec![vec![q(1, 2), q(1, 3)], vec![qi(0), qi(1)]]);
    let err = MarkovChain::new(bad_sum).expect_err("rows must sum to 1");
    assert!(err.to_string().contains("sums to 5/6"), "{err}");
    let ok = qmatrix(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]]);
    assert!(MarkovChain::with_labels(ok.clone(), vec!["a".into()]).is_err());
    let chain = MarkovChain::with_labels(ok.clone(), vec!["rain".into(), "sun".into()])
        .expect("valid chain");
    assert_eq!(chain.n_states(), 2);
    assert_eq!(chain.state_index("sun"), Some(1));
    assert_eq!(chain.state_index("snow"), None);
    assert_eq!(chain.labels().map(<[String]>::len), Some(2));
    assert_eq!(chain.transition_matrix(), &ok);
    assert!(MarkovChain::new(ok).expect("valid").labels().is_none());
}

#[test]
fn markov_n_step_and_distribution_after() -> Result<(), SymplexError> {
    let chain = chain_three();
    // SymPy: T**2 = [[5/12, 1/4, 1/3], [1/6, 5/6, 0], [1/3, 0, 2/3]]
    let p2 = chain.n_step(2);
    assert_eq!(p2.row(0), &[q(5, 12), q(1, 4), q(1, 3)]);
    assert_eq!(p2.row(1), &[q(1, 6), q(5, 6), qi(0)]);
    assert_eq!(p2.row(2), &[q(1, 3), qi(0), q(2, 3)]);
    // (T**3)[0, :] = [7/24, 13/24, 1/6]
    assert_eq!(chain.n_step(3).row(0), &[q(7, 24), q(13, 24), q(1, 6)]);
    // T**4 = [[47/144, 5/16, 13/36], [5/24, 53/72, 1/18], [13/36, 1/12, 5/9]]
    let p4 = chain.n_step(4);
    assert_eq!(p4.row(0), &[q(47, 144), q(5, 16), q(13, 36)]);
    assert_eq!(p4.row(1), &[q(5, 24), q(53, 72), q(1, 18)]);
    assert_eq!(p4.row(2), &[q(13, 36), q(1, 12), q(5, 9)]);
    assert!(chain.n_step(0).is_identity());
    assert_eq!(chain.n_step(1), *chain.transition_matrix());
    // (T**10)[0, :] = [74189/248832, 1159/3072, 20191/62208]; [2, :] = [20191/62208, 5551/20736, 6341/15552]
    let p10 = chain.n_step(10);
    assert_eq!(
        p10.row(0),
        &[q(74_189, 248_832), q(1159, 3072), q(20_191, 62_208)]
    );
    assert_eq!(
        p10.row(2),
        &[q(20_191, 62_208), q(5551, 20_736), q(6341, 15_552)]
    );
    // Repeated squaring agrees with the plain product chain.
    let mut plain = chain.n_step(1);
    for _ in 1..10 {
        plain = plain.matmul(chain.transition_matrix())?;
    }
    assert_eq!(p10, plain);
    // [1, 0, 0]·T² = [5/12, 1/4, 1/3]; [1/2, 1/2, 0]·T³ = [47/144, 5/16, 13/36]
    assert_eq!(
        chain.distribution_after(&[qi(1), qi(0), qi(0)], 2)?,
        vec![q(5, 12), q(1, 4), q(1, 3)]
    );
    assert_eq!(
        chain.distribution_after(&[q(1, 2), q(1, 2), qi(0)], 3)?,
        vec![q(47, 144), q(5, 16), q(13, 36)]
    );
    assert!(chain.distribution_after(&[qi(1), qi(0)], 1).is_err());
    assert!(
        chain
            .distribution_after(&[q(1, 2), q(1, 2), q(1, 2)], 1)
            .is_err()
    );
    assert!(
        chain
            .distribution_after(&[qi(2), qi(-1), qi(0)], 1)
            .is_err()
    );
    Ok(())
}

#[test]
fn markov_stationary_distribution_of_the_three_state_chain() -> Result<(), SymplexError> {
    // SymPy: DiscreteMarkovChain('Y', [0,1,2], T).stationary_distribution() = [2/7, 3/7, 2/7]
    let chain = chain_three();
    let pi = chain.stationary_distribution()?;
    assert_eq!(pi, vec![q(2, 7), q(3, 7), q(2, 7)]);
    assert_eq!(chain.stationary_distributions(), vec![pi.clone()]);
    // πP = π, exactly.
    assert_eq!(chain.distribution_after(&pi, 1)?, pi);
    assert_eq!(chain.distribution_after(&pi, 7)?, pi);
    Ok(())
}

#[test]
fn markov_structure_of_the_three_state_chain() -> Result<(), SymplexError> {
    // SymPy: communication_classes() = [([0, 1, 2], True, 1)], is_regular() = is_ergodic() = True
    let chain = chain_three();
    assert_eq!(chain.communication_classes(), vec![vec![0, 1, 2]]);
    assert_eq!(chain.closed_classes(), vec![vec![0, 1, 2]]);
    assert!(chain.transient_states().is_empty());
    assert!(chain.is_irreducible());
    assert!(chain.is_ergodic());
    assert!(chain.is_aperiodic());
    assert!(chain.is_regular());
    for s in 0..3 {
        assert_eq!(chain.period_of(s)?, Some(1));
    }
    assert!(chain.period_of(3).is_err());
    assert!(chain.absorbing_states().is_empty());
    assert!(!chain.is_absorbing_chain());
    assert!(chain.fundamental_matrix().is_err());
    assert!(chain.absorption_probabilities().is_err());
    Ok(())
}

#[test]
fn markov_fair_gamblers_ruin_is_absorbing() -> Result<(), SymplexError> {
    // SymPy on the fair gambler's ruin (5 states):
    // communication_classes() = [([0], True, 1), ([4], True, 1), ([1, 2, 3], False, 2)]
    // fundamental_matrix() = [[3/2, 1, 1/2], [1, 2, 1], [1/2, 1, 3/2]]
    // absorbing_probabilities() = [[3/4, 1/4], [1/2, 1/2], [1/4, 3/4]]; N·1 = [3, 4, 3]
    let chain = gambler(q(1, 2));
    assert_eq!(
        chain.communication_classes(),
        vec![vec![0], vec![1, 2, 3], vec![4]]
    );
    assert_eq!(chain.closed_classes(), vec![vec![0], vec![4]]);
    assert_eq!(chain.transient_states(), vec![1, 2, 3]);
    assert_eq!(chain.absorbing_states(), vec![0, 4]);
    assert!(chain.is_absorbing_chain());
    assert!(!chain.is_irreducible() && !chain.is_regular() && !chain.is_ergodic());
    assert_eq!(chain.period_of(2)?, Some(2));
    assert_eq!(chain.period_of(0)?, Some(1));
    let n = chain.fundamental_matrix()?;
    assert_eq!(n.row(0), &[q(3, 2), qi(1), q(1, 2)]);
    assert_eq!(n.row(1), &[qi(1), qi(2), qi(1)]);
    assert_eq!(n.row(2), &[q(1, 2), qi(1), q(3, 2)]);
    let b = chain.absorption_probabilities()?;
    assert_eq!(b.row(0), &[q(3, 4), q(1, 4)]);
    assert_eq!(b.row(1), &[q(1, 2), q(1, 2)]);
    assert_eq!(b.row(2), &[q(1, 4), q(3, 4)]);
    assert_eq!(
        chain.expected_steps_to_absorption()?,
        vec![qi(3), qi(4), qi(3)]
    );
    // Two closed classes ⇒ two extreme stationary distributions, no unique one.
    assert_eq!(
        chain.stationary_distributions(),
        vec![
            vec![qi(1), qi(0), qi(0), qi(0), qi(0)],
            vec![qi(0), qi(0), qi(0), qi(0), qi(1)],
        ]
    );
    assert!(chain.stationary_distribution().is_err());
    assert!(chain.fundamental_matrix_ergodic().is_err());
    Ok(())
}

#[test]
fn markov_biased_gamblers_ruin_has_rational_absorption_data() -> Result<(), SymplexError> {
    // SymPy, p = 2/3: fundamental_matrix() = [[7/5, 6/5, 4/5], [3/5, 9/5, 6/5], [1/5, 3/5, 7/5]],
    // absorbing_probabilities() = [[7/15, 8/15], [1/5, 4/5], [1/15, 14/15]], N·1 = [17/5, 18/5, 11/5]
    let chain = gambler(q(2, 3));
    let n = chain.fundamental_matrix()?;
    assert_eq!(n.row(0), &[q(7, 5), q(6, 5), q(4, 5)]);
    assert_eq!(n.row(1), &[q(3, 5), q(9, 5), q(6, 5)]);
    assert_eq!(n.row(2), &[q(1, 5), q(3, 5), q(7, 5)]);
    let b = chain.absorption_probabilities()?;
    assert_eq!(b.row(0), &[q(7, 15), q(8, 15)]);
    assert_eq!(b.row(1), &[q(1, 5), q(4, 5)]);
    assert_eq!(b.row(2), &[q(1, 15), q(14, 15)]);
    assert_eq!(
        chain.expected_steps_to_absorption()?,
        vec![q(17, 5), q(18, 5), q(11, 5)]
    );
    // Absorption probabilities coincide with the hitting probabilities of each absorbing state.
    let ruin = chain.hitting_probability(&[0])?;
    assert_eq!(ruin, vec![qi(1), q(7, 15), q(1, 5), q(1, 15), qi(0)]);
    Ok(())
}

#[test]
fn markov_hitting_probabilities_and_times() -> Result<(), SymplexError> {
    // Fair gambler's ruin: sympy.solve of h1 = 1/2 + h2/2, h2 = h1/2 + h3/2, h3 = h2/2
    // gives (3/4, 1/2, 1/4); expected time to {0, 4}: (3, 4, 3).
    let chain = gambler(q(1, 2));
    assert_eq!(
        chain.hitting_probability(&[0])?,
        vec![qi(1), q(3, 4), q(1, 2), q(1, 4), qi(0)]
    );
    assert_eq!(
        chain.hitting_probability(&[0, 4])?,
        vec![qi(1), qi(1), qi(1), qi(1), qi(1)]
    );
    assert_eq!(
        chain.expected_hitting_time(&[4, 0])?,
        vec![qi(0), qi(3), qi(4), qi(3), qi(0)]
    );
    // Hitting {0} alone happens with probability < 1 from 1..=4: infinite expectation.
    let err = chain.expected_hitting_time(&[0]).expect_err("infinite");
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err}"
    );
    // Three-state chain: sympy.solve(k0 = 1 + k0/2 + k1/2, k1 = 1 + k0/3) = (9/2, 5/2).
    let three = chain_three();
    assert_eq!(three.hitting_probability(&[2])?, vec![qi(1), qi(1), qi(1)]);
    assert_eq!(
        three.expected_hitting_time(&[2])?,
        vec![q(9, 2), q(5, 2), qi(0)]
    );
    assert!(three.hitting_probability(&[]).is_err());
    assert!(three.hitting_probability(&[3]).is_err());
    Ok(())
}

#[test]
fn markov_periodic_cycle() -> Result<(), SymplexError> {
    // SymPy on the 3-cycle: communication_classes() = [([0, 1, 2], True, 3)],
    // is_regular() = False, is_ergodic() = True, stationary_distribution() = [1/3, 1/3, 1/3]
    let chain = MarkovChain::new(qmatrix(vec![
        vec![qi(0), qi(1), qi(0)],
        vec![qi(0), qi(0), qi(1)],
        vec![qi(1), qi(0), qi(0)],
    ]))?;
    assert_eq!(chain.communication_classes(), vec![vec![0, 1, 2]]);
    assert_eq!(chain.period_of(0)?, Some(3));
    assert!(!chain.is_aperiodic());
    assert!(chain.is_irreducible() && chain.is_ergodic());
    assert!(!chain.is_regular());
    assert_eq!(
        chain.stationary_distribution()?,
        vec![q(1, 3), q(1, 3), q(1, 3)]
    );
    assert!(chain.n_step(3).is_identity());
    assert!(!chain.n_step(2).is_identity());
    Ok(())
}

#[test]
fn markov_bipartite_chain_has_period_two() -> Result<(), SymplexError> {
    // SymPy: communication_classes() = [([0, 1, 2, 3], True, 2)], is_regular() = False,
    // is_ergodic() = True, stationary_distribution() = [5/14, 3/14, 1/7, 2/7]
    let chain = MarkovChain::new(qmatrix(vec![
        vec![qi(0), q(1, 2), qi(0), q(1, 2)],
        vec![q(1, 3), qi(0), q(2, 3), qi(0)],
        vec![qi(0), q(1, 4), qi(0), q(3, 4)],
        vec![qi(1), qi(0), qi(0), qi(0)],
    ]))?;
    assert!(chain.is_irreducible());
    for s in 0..4 {
        assert_eq!(chain.period_of(s)?, Some(2));
    }
    assert!(!chain.is_aperiodic() && !chain.is_regular() && chain.is_ergodic());
    let pi = chain.stationary_distribution()?;
    assert_eq!(pi, vec![q(5, 14), q(3, 14), q(1, 7), q(2, 7)]);
    assert_eq!(chain.distribution_after(&pi, 1)?, pi);
    // Mean recurrence times 1/πᵢ.
    assert_eq!(
        chain.mean_recurrence_times()?,
        vec![q(14, 5), q(14, 3), qi(7), q(7, 2)]
    );
    Ok(())
}

#[test]
fn markov_reducible_chain_with_two_closed_classes() -> Result<(), SymplexError> {
    // SymPy: communication_classes() = [([0, 1], True, 1), ([2], True, 1), ([3], False, 1)],
    // stationary_distribution() = [1/3 − τ/3, 2/3 − 2τ/3, τ, 0]; the {0,1} sub-chain has [1/3, 2/3].
    let chain = MarkovChain::new(qmatrix(vec![
        vec![q(1, 2), q(1, 2), qi(0), qi(0)],
        vec![q(1, 4), q(3, 4), qi(0), qi(0)],
        vec![qi(0), qi(0), qi(1), qi(0)],
        vec![q(1, 3), qi(0), q(1, 3), q(1, 3)],
    ]))?;
    assert_eq!(
        chain.communication_classes(),
        vec![vec![0, 1], vec![2], vec![3]]
    );
    assert_eq!(chain.closed_classes(), vec![vec![0, 1], vec![2]]);
    assert_eq!(chain.transient_states(), vec![3]);
    assert_eq!(chain.absorbing_states(), vec![2]);
    // States 0 and 1 never reach the absorbing state 2: not an absorbing chain.
    assert!(!chain.is_absorbing_chain());
    assert!(chain.fundamental_matrix().is_err());
    assert_eq!(
        chain.stationary_distributions(),
        vec![
            vec![q(1, 3), q(2, 3), qi(0), qi(0)],
            vec![qi(0), qi(0), qi(1), qi(0)],
        ]
    );
    for pi in chain.stationary_distributions() {
        assert_eq!(chain.distribution_after(&pi, 1)?, pi);
    }
    assert!(chain.stationary_distribution().is_err());
    // From the transient state 3: reach {0,1} w.p. 1/2, get absorbed at 2 w.p. 1/2.
    assert_eq!(
        chain.hitting_probability(&[2])?,
        vec![qi(0), qi(0), qi(1), q(1, 2)]
    );
    assert_eq!(
        chain.hitting_probability(&[0, 1])?,
        vec![qi(1), qi(1), qi(0), q(1, 2)]
    );
    assert_eq!(chain.period_of(3)?, Some(1));
    assert!(chain.is_aperiodic());
    Ok(())
}

#[test]
fn markov_transient_state_without_a_cycle_has_no_period() -> Result<(), SymplexError> {
    // State 0 leaves for good and never returns: no cycle, period undefined.
    let chain = MarkovChain::new(qmatrix(vec![
        vec![qi(0), q(1, 2), q(1, 2)],
        vec![qi(0), qi(1), qi(0)],
        vec![qi(0), qi(0), qi(1)],
    ]))?;
    assert_eq!(chain.period_of(0)?, None);
    assert_eq!(chain.period_of(1)?, Some(1));
    assert!(chain.is_aperiodic());
    assert!(chain.is_absorbing_chain());
    assert_eq!(chain.fundamental_matrix()?.row(0), &[qi(1)]);
    assert_eq!(
        chain.absorption_probabilities()?.row(0),
        &[q(1, 2), q(1, 2)]
    );
    assert_eq!(chain.expected_steps_to_absorption()?, vec![qi(1)]);
    Ok(())
}

#[test]
fn markov_ergodic_fundamental_matrix_and_mean_first_passage() -> Result<(), SymplexError> {
    // SymPy: DiscreteMarkovChain(..., T).fundamental_matrix() (regular chain ⇒ Z = (I − P + W)⁻¹)
    // = [[68/49, -3/49, -16/49], [-2/49, 39/49, 12/49], [-16/49, 18/49, 47/49]]
    let chain = chain_three();
    let z = chain.fundamental_matrix_ergodic()?;
    assert_eq!(z.row(0), &[q(68, 49), q(-3, 49), q(-16, 49)]);
    assert_eq!(z.row(1), &[q(-2, 49), q(39, 49), q(12, 49)]);
    assert_eq!(z.row(2), &[q(-16, 49), q(18, 49), q(47, 49)]);
    // M[i][j] = (Z[j][j] − Z[i][j])/π_j (sympy arithmetic) = [[0, 2, 9/2], [5, 0, 5/2], [6, 1, 0]]
    let m = chain.mean_first_passage_times()?;
    assert_eq!(m.row(0), &[qi(0), qi(2), q(9, 2)]);
    assert_eq!(m.row(1), &[qi(5), qi(0), q(5, 2)]);
    assert_eq!(m.row(2), &[qi(6), qi(1), qi(0)]);
    // Consistency with the direct hitting-time solve of each singleton.
    for j in 0..3 {
        let k = chain.expected_hitting_time(&[j])?;
        for (i, ki) in k.iter().enumerate() {
            assert_eq!(m.get(i, j), ki, "M[{i}][{j}]");
        }
    }
    // Mean recurrence times 1/π = [7/2, 7/3, 7/2].
    assert_eq!(
        chain.mean_recurrence_times()?,
        vec![q(7, 2), q(7, 3), q(7, 2)]
    );
    Ok(())
}

#[test]
fn markov_sample_path_is_deterministic_and_feasible() -> Result<(), SymplexError> {
    let chain = chain_three();
    let a = chain.sample_path(0, 50, &mut Rng::new(7))?;
    let b = chain.sample_path(0, 50, &mut Rng::new(7))?;
    assert_eq!(a, b);
    assert_eq!(a.len(), 51);
    assert_eq!(a[0], 0);
    // Every transition taken has positive probability.
    for w in a.windows(2) {
        assert!(
            *chain.transition_matrix().get(w[0], w[1]) > qi(0),
            "impossible step {} → {}",
            w[0],
            w[1]
        );
    }
    // Different seeds disagree somewhere over 50 steps.
    assert_ne!(a, chain.sample_path(0, 50, &mut Rng::new(8))?);
    // A deterministic cycle walks 0, 1, 2, 0, 1, 2, …
    let cycle = MarkovChain::new(qmatrix(vec![
        vec![qi(0), qi(1), qi(0)],
        vec![qi(0), qi(0), qi(1)],
        vec![qi(1), qi(0), qi(0)],
    ]))?;
    assert_eq!(
        cycle.sample_path(1, 5, &mut Rng::new(1))?,
        vec![1, 2, 0, 1, 2, 0]
    );
    assert!(chain.sample_path(3, 1, &mut Rng::new(1)).is_err());
    assert_eq!(chain.sample_path(2, 0, &mut Rng::new(1))?, vec![2]);
    Ok(())
}

#[test]
fn markov_two_state_weather_chain_end_to_end() -> Result<(), SymplexError> {
    // P = [[1/2, 1/2], [1, 0]]: SymPy T**2 = [[3/4, 1/4], [1/2, 1/2]], stationary [2/3, 1/3].
    let chain = MarkovChain::with_labels(
        qmatrix(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]]),
        vec!["dry".into(), "wet".into()],
    )?;
    assert_eq!(chain.n_step(2).row(0), &[q(3, 4), q(1, 4)]);
    assert_eq!(chain.stationary_distribution()?, vec![q(2, 3), q(1, 3)]);
    assert!(chain.is_regular());
    let wet = chain.state_index("wet").expect("labelled");
    // From dry, the expected time to the first wet day is 2; from wet back to wet is 1/π = 3.
    assert_eq!(chain.expected_hitting_time(&[wet])?, vec![qi(2), qi(0)]);
    assert_eq!(chain.mean_recurrence_times()?[wet], qi(3));
    Ok(())
}
