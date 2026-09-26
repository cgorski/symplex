//! Audit round 3b-i of `symplex::stats`: `estimation`, `data`,
//! `information`.
//!
//! 1. Every normal / Student-t critical value of a two-sided interval was
//!    the quantile at the rounded level `1 − α/2` (`proportion_interval`'s
//!    Wald, Wilson and Agresti–Coull, `confidence_interval_mean`, the upper
//!    end of `credible_interval`): a relative error of up to `2⁻⁵⁴/(α/2)` in
//!    the tail — `3.3e-3` in the Wilson lower end of `(5, 5)` at
//!    `1 − 10⁻¹⁵`, `1.1e-4` in the t half-width at `1 − 10⁻¹²` — and at
//!    `c = 1 − 2⁻⁵³` the level rounded to `1`, an error.  Each is now solved
//!    on the upper tail (`isf`).
//! 2. Clopper–Pearson bisected a binomial sum over all `n + 1` terms per
//!    step: hours at `n = 10⁹`, forever at `usize::MAX`, and a relative
//!    error near `10⁻¹²` at `n = 10⁵` from the accumulated `ln C(n, i)`.
//!    It is now the Beta quantiles (statsmodels' `'beta'`), `O(1)` in `n`;
//!    the same route gives the new Jeffreys interval.
//! 3. The Wilson lower end was `centre − half`, which cancels (to
//!    `5.6e-17` at `k = 0`, where it is exactly `0`).
//! 4. `aic` / `bic` cast counts to `i64`: `bic(ℓ, 1, usize::MAX)` was
//!    `iπ + 6`, `aic(ℓ, usize::MAX)` was `4 − 2ℓ`.
//! 5. `joint_from_counts` summed in `usize` (a panic past `usize::MAX`) and
//!    `probability_vector` subtracted the support ends in `i64` (a panic).
//! 6. `kendall_tau` multiplied its two denominator factors in `i64`, which
//!    overflows from about 78 000 untied pairs on, after an `O(n²)` count;
//!    the pairs are now counted in `O(n log n)` (Knight 1966).
//! 7. `normalized_mutual_information` with `Min` / `Max` chose the
//!    normaliser by summing `Σ c_q ln q` in `f64`, whose terms cancel.
//! 8. `posterior_predictive_beta_binomial` did `O(n²)` big-rational work
//!    and pairwise table checks (22 s at `n = 300`).
//! 9. `QuantileMethod::Exclusive` was documented as `statistics.quantiles`,
//!    which extrapolates beyond the extreme levels where it clamps.
//!
//! Reference values: statsmodels 0.15.0 `proportion_confint`, scipy
//! 1.18.1 by the call quoted, and mpmath 1.3 at 60 digits where the
//! library itself is less accurate than the fix (Clopper–Pearson by Newton
//! on the binomial tail summed outward from `k`; Wilson's lower end as the
//! product of the roots), scripts `target/scratch/oracle_3b1.py` and
//! `prop_oracle.py` of this session.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::data::{
    ConcordanceCounts, QuantileMethod, concordance_counts, from_i64, kendall_tau, quantile,
    quantiles,
};
use symplex::stats::estimation::{
    IntervalMethod, aic, bic, confidence_interval_mean, credible_interval, proportion_interval,
    proportion_interval_exact, proportion_interval_symbolic,
};
use symplex::stats::information::{
    Norm, joint_from_counts, mutual_information, normalized_mutual_information, probability_vector,
};

type R = Result<(), SymplexError>;

/// `1 − 2⁻⁵³`, the largest double below `1`.
const TOP: f64 = 1.0 - f64::EPSILON / 2.0;

/// `actual` within `rel` of `expected`, relatively (exactly for `0`).
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e} (rel {:.1e})",
        (actual / expected - 1.0).abs()
    );
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::InvalidArgument { .. }) => {}
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

/// The closed-form intervals are a few roundings of `z = Φ̄⁻¹(α/2)`, which
/// `numdist::norm::isf` gives to about `1e-16`: a few ulps.
const Z_REL: f64 = 1e-14;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Critical values from the upper tail
// ═══════════════════════════════════════════════════════════════════════════

/// Wald, Wilson and Agresti–Coull took `z` at the rounded level
/// `1 − α/2`: the Wilson lower end of `(5, 5)` at `1 − 10⁻¹⁵` was
/// `0.071772965491` (truly `0.072012864295`), Wald at `(591, 1000)`
/// `0.465977822754` (truly `0.466202371707`).
#[test]
fn proportion_z_from_the_upper_tail() -> R {
    let c = 1.0 - 1e-15;
    let ci = proportion_interval(5, 5, c, IntervalMethod::Wilson)?;
    // mpmath: Wilson (5, 5, 1 - 1e-15) lower = 0.072012864294626556625
    close(ci.lower, 0.072012864294626556625, Z_REL, "Wilson lo");
    assert_eq!(ci.upper, 1.0);
    let ci = proportion_interval(591, 1000, c, IntervalMethod::Wald)?;
    // statsmodels: proportion_confint(591, 1000, alpha=1-c, method='normal')
    //   = (0.46620237170668827, 0.7157976282933116)
    close(ci.lower, 0.46620237170668827, Z_REL, "Wald lo");
    close(ci.upper, 0.7157976282933116, Z_REL, "Wald hi");
    let ci = proportion_interval(591, 1000, c, IntervalMethod::AgrestiCoull)?;
    // statsmodels: method='agresti_coull' = (0.46428703868489457, 0.7066961648799113)
    close(ci.lower, 0.46428703868489457, Z_REL, "AC lo");
    close(ci.upper, 0.7066961648799113, Z_REL, "AC hi");
    Ok(())
}

/// At `confidence = 1 − 2⁻⁵³` the level `1 − α/2` rounded to `1` and every
/// closed-form interval, the t interval and the credible interval were
/// errors ("p must lie strictly between 0 and 1, got 1").
#[test]
fn intervals_at_the_largest_confidence_below_one() -> R {
    let ci = proportion_interval(3, 10, TOP, IntervalMethod::Wilson)?;
    // mpmath: lower 0.012194665778964795903, upper 0.93702022964904016276
    close(ci.lower, 0.012194665778964795903, Z_REL, "Wilson lo");
    close(ci.upper, 0.93702022964904016276, Z_REL, "Wilson hi");
    let ci = proportion_interval(3, 10, TOP, IntervalMethod::AgrestiCoull)?;
    // statsmodels: method='agresti_coull' = (0.008027989103221056, 0.941186906324784)
    close(ci.lower, 0.008027989103221056, Z_REL, "AC lo");
    close(ci.upper, 0.941186906324784, Z_REL, "AC hi");
    let ci = proportion_interval(3, 10, TOP, IntervalMethod::Wald)?;
    // statsmodels: method='normal' = (0.0, 1.0) (clipped)
    assert_eq!((ci.lower, ci.upper), (0.0, 1.0));

    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    let ci = confidence_interval_mean(&x, TOP)?;
    // scipy: mean(x) -+ t.isf(2**-54, 5) * sem(x) = (-2764.3666728794624, 2781.3666728794624)
    // (t.interval itself returns (-2764.37, inf): its upper end is ppf(1.0))
    close(ci.lower, -2764.3666728794624, 1e-14, "t lo");
    close(ci.upper, 2781.3666728794624, 1e-14, "t hi");

    let ctx = Context::new();
    let ci = credible_interval(&Distribution::gamma(ctx.int(14), ctx.rational(1, 5)), TOP)?;
    // scipy: gamma.ppf(2**-54, 14, scale=0.2) = 0.08585987388322851,
    //        gamma.isf(2**-54, 14, scale=0.2) = 14.0757635394569
    close(ci.lower, 0.08585987388322851, 1e-14, "gamma lo");
    close(ci.upper, 14.0757635394569, 1e-14, "gamma hi");
    Ok(())
}

/// The t interval's half-width at `c = 1 − 10⁻¹²` was `1.1e-5` relatively
/// too small (`t.ppf` at the rounded `1 − 5·10⁻¹³`).
#[test]
fn t_interval_critical_value_from_the_upper_tail() -> R {
    let x = from_i64(&[5, 7, 8, 9, 10, 12]);
    let ci = confidence_interval_mean(&x, 1.0 - 1e-12)?;
    // scipy: c = 1 - 1e-12; mean(x) -+ t.isf((1 - c)/2, 5) * sem(x)
    //   = (-440.2542041420432, 457.2542041420432); 0.28: (-440.2442400318308, 457.2442400318308)
    // The `f64` critical value is good to ~1e-15 but is multiplied by sem
    // and cancels against the mean at the lower end (|lo| ≈ 440 from 8.5 − 448.7).
    close(ci.lower, -440.2542041420432, 1e-13, "lo");
    close(ci.upper, 457.2542041420432, 1e-13, "hi");
    Ok(())
}

/// The upper end of `credible_interval` was the quantile at the rounded
/// level `1 − t`: `Beta(9, 6)` at `c = 1 − 10⁻¹²` gave `0.99764814506449`.
/// Kernel families use their `isf`, other continuous laws the reflection
/// `−X` (checked on the exponential).
#[test]
fn credible_interval_upper_end_from_the_upper_tail() -> R {
    let ctx = Context::new();
    let c = 1.0 - 1e-12;
    let ci = credible_interval(&Distribution::beta(ctx.int(9), ctx.int(6)), c)?;
    // scipy: beta.ppf((1-c)/2, 9, 6) = 0.01864100141687073, beta.isf((1-c)/2, 9, 6) = 0.9976481886981728
    close(ci.lower, 0.01864100141687073, 1e-14, "beta lo");
    close(ci.upper, 0.9976481886981728, 1e-15, "beta hi");
    let ci = credible_interval(&Distribution::exponential(ctx.int(2)), c)?;
    // scipy: expon.isf((1-c)/2, scale=0.5) = 14.162095209226653
    close(ci.upper, 14.162095209226653, 1e-14, "exponential hi");
    // A finite table keeps the quantile at 1 − t (its tails are far from t).
    let die = Distribution::die(ctx.int(6));
    let ci = credible_interval(&die, 0.95)?;
    assert_eq!((ci.lower, ci.upper), (1.0, 6.0));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Clopper–Pearson and Jeffreys from Beta quantiles
// ═══════════════════════════════════════════════════════════════════════════

/// Clopper–Pearson was a bisection over an `O(n)` binomial sum: `n = 10⁶`
/// took 2 s (debug) and was `3.5e-11` off at `k = 1`; `n = 3·10⁹` never
/// finished.  The Beta quantiles of `numdist` are good to about `1e-15`
/// (`1e-13` for shapes up to `10⁶`, its documented bound), so `2e-14` —
/// where statsmodels' scipy values are themselves up to `1e-12` off.
#[test]
fn clopper_pearson_from_beta_quantiles() -> R {
    let cases: [(usize, usize, f64, f64); 4] = [
        // mpmath: Newton on P(Bin(n, p) >= k) = a/2 and P(Bin(n, p) <= k) = a/2
        (
            1,
            1_000_000,
            2.5317807663794200318e-8,
            5.5716306551722432843e-6,
        ),
        (
            45162,
            100_000,
            0.4485320065710186853,
            0.45471079993930159907,
        ),
        (
            2,
            3_000_000_000,
            8.073642619151855901e-11,
            2.4082292204776122965e-9,
        ),
        (
            7,
            usize::MAX,
            1.5256692673103865136e-19,
            7.8185479801054382485e-19,
        ),
    ];
    for (k, n, lo, hi) in cases {
        let ci = proportion_interval(k, n, 0.95, IntervalMethod::ClopperPearson)?;
        close(ci.lower, lo, 2e-14, &format!("CP({k}, {n}) lo"));
        close(ci.upper, hi, 2e-14, &format!("CP({k}, {n}) hi"));
    }
    // The ends: 0 at k = 0, 1 at k = n (also at usize::MAX).
    let ci = proportion_interval(0, 10, 0.95, IntervalMethod::ClopperPearson)?;
    assert_eq!(ci.lower, 0.0);
    let ci = proportion_interval(usize::MAX, usize::MAX, 0.95, IntervalMethod::ClopperPearson)?;
    assert_eq!((ci.lower, ci.upper), (1.0, 1.0));
    // mpmath: CP(3, 10) at 1 - 2^-53: (7.7339302412191314718e-7, 0.99759489074611191419)
    let ci = proportion_interval(3, 10, TOP, IntervalMethod::ClopperPearson)?;
    close(ci.lower, 7.7339302412191314718e-7, 2e-14, "CP top lo");
    close(ci.upper, 0.99759489074611191419, 2e-14, "CP top hi");
    Ok(())
}

/// New: the Jeffreys interval (statsmodels `method='jeffreys'`).
#[test]
fn jeffreys_interval() -> R {
    let cases: [(usize, usize, f64, f64, f64); 4] = [
        // statsmodels: proportion_confint(3, 10, alpha=0.05, method='jeffreys')
        //   (mpmath: 0.092694593938153189839, 0.60581831814867124425)
        (3, 10, 0.95, 0.09269459393815319, 0.6058183181486713),
        // statsmodels: proportion_confint(0, 20, alpha=0.01, method='jeffreys')
        (0, 20, 0.99, 9.69565720539231e-07, 0.17675409743668993),
        // statsmodels: proportion_confint(20, 20, alpha=0.1, method='jeffreys')
        (20, 20, 0.9, 0.9095235734621276, 0.9999029222414176),
        // statsmodels: proportion_confint(123, 4567, alpha=1e-9, method='jeffreys')
        (
            123,
            4567,
            1.0 - 1e-9,
            0.014805284097062235,
            0.04425326893343241,
        ),
    ];
    for (k, n, c, lo, hi) in cases {
        let ci = proportion_interval(k, n, c, IntervalMethod::Jeffreys)?;
        close(ci.lower, lo, 1e-13, &format!("Jeffreys({k}, {n}) lo"));
        close(ci.upper, hi, 1e-13, &format!("Jeffreys({k}, {n}) hi"));
    }
    let ctx = Context::new();
    is_invalid(proportion_interval_symbolic(
        &ctx,
        3,
        10,
        &ctx.symbol("z"),
        IntervalMethod::Jeffreys,
    ));
    is_invalid(proportion_interval_exact(
        &ctx,
        3,
        10,
        &q(95, 100),
        IntervalMethod::Jeffreys,
    ));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Wilson's lower end without cancellation
// ═══════════════════════════════════════════════════════════════════════════

/// The lower end was `centre − half`: `5.55e-17` at `(0, 7)` (statsmodels
/// has the same noise), and a few digits lost when `k ≪ z²`.
#[test]
fn wilson_lower_end_without_cancellation() -> R {
    let ci = proportion_interval(0, 7, 0.95, IntervalMethod::Wilson)?;
    assert_eq!(ci.lower, 0.0);
    // mpmath: upper 0.35433043506668736708
    close(ci.upper, 0.35433043506668736708, Z_REL, "(0, 7) hi");
    let ci = proportion_interval(1, 1_000_000, 1.0 - 1e-15, IntervalMethod::Wilson)?;
    // mpmath: (1.5056390335356392666e-8, 6.6412702503078524932e-5)
    close(ci.lower, 1.5056390335356392666e-8, Z_REL, "(1, 1e6) lo");
    close(ci.upper, 6.6412702503078524932e-5, Z_REL, "(1, 1e6) hi");
    let ci = proportion_interval(1, 100, 1.0 - 1e-10, IntervalMethod::Wilson)?;
    // mpmath: lower 0.00022836749617745435941 (statsmodels 0.00022836749617749508)
    close(ci.lower, 0.00022836749617745435941, Z_REL, "(1, 100) lo");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4–5. Counts beyond the integer types
// ═══════════════════════════════════════════════════════════════════════════

/// `aic` formed `2 * k as i64` and `bic` took `k as i64`, `n as i64`.
#[test]
fn aic_bic_counts_beyond_i64() -> R {
    let ctx = Context::new();
    let l = ctx.int(-3);
    // 2 (2^64 - 1) + 6 (0.28: 4)
    assert_eq!(aic(&l, usize::MAX).to_string(), "36893488147419103236");
    // (2^64 - 1) ln 10 + 6 (0.28: ln(1/10) + 6, k read as -1)
    let b = bic(&l, usize::MAX, 10);
    let want = ctx.from_bigint(usize::MAX.into()) * ctx.int(10).ln() + ctx.int(6);
    assert_eq!((&b - &want).simplify(), ctx.zero());
    // mpmath: log(2**64 - 1) + 6 = 50.36141955583649980264865 (0.28: iπ + 6)
    let b = bic(&l, 1, usize::MAX);
    close(b.eval_f64()?, 50.36141955583649980264865, 1e-15, "bic");
    Ok(())
}

/// `joint_from_counts` summed in `usize`: `[[usize::MAX, 1]]` panicked
/// ("attempt to add with overflow").
#[test]
fn joint_from_counts_beyond_usize() -> R {
    let ctx = Context::new();
    let m = usize::MAX;
    let j = joint_from_counts(&[vec![m, 1]])?;
    // Fraction(2**64 - 1, 2**64), Fraction(1, 2**64)
    assert_eq!(
        j[0][0],
        "18446744073709551615/18446744073709551616"
            .parse::<Q>()
            .unwrap()
    );
    assert_eq!(j[0][1], "1/18446744073709551616".parse::<Q>().unwrap());
    // Independent margins: I = 0 exactly (a table of four counts past usize::MAX in total).
    let j = joint_from_counts(&[vec![m, m], vec![m, m]])?;
    assert_eq!(j[0][0], q(1, 4));
    assert_eq!(
        mutual_information(&ctx, &j, symplex::stats::information::Base::Nats)?,
        ctx.zero()
    );
    Ok(())
}

/// `probability_vector` computed `hi − lo` in `i64`: a support from
/// `i64::MIN` to `i64::MAX` panicked.
#[test]
fn probability_vector_of_a_support_wider_than_i64() {
    let ctx = Context::new();
    let d = Distribution::discrete_uniform(ctx.int(i64::MIN), ctx.int(i64::MAX));
    is_invalid(probability_vector(&d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Kendall's τ-b and the concordance counts
// ═══════════════════════════════════════════════════════════════════════════

/// `(n₀ − n₁)(n₀ − n₂)` was an `i64` product: at `n = 10⁵` it is
/// `2.4999·10¹⁹ > i64::MAX` (a panic in debug; a release build wrapped it
/// to `6.55·10¹⁸` and returned `τ ≈ 3.99e-4`), after an `O(n²)` count.
#[test]
fn kendall_tau_denominator_beyond_i64() -> R {
    let ctx = Context::new();
    let n = 100_000i64;
    let x = from_i64(&(0..n).map(|i| i / 2).collect::<Vec<_>>());
    let y = from_i64(&(0..n).map(|i| (i * 7919) % n).collect::<Vec<_>>());
    let tau = kendall_tau(&ctx, &x, &y)?;
    // scipy: kendalltau(i // 2, 7919 i mod n).statistic = 0.00020414626219648484
    close(tau.eval_f64()?, 0.00020414626219648484, 1e-14, "tau-b");
    Ok(())
}

/// The `O(n log n)` counts against the pair-by-pair definition on random
/// samples with many ties (a xorshift stream, so the test is reproducible).
#[test]
fn concordance_counts_match_the_pairwise_definition() -> R {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |m: u64| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m) as i64
    };
    for trial in 0..400 {
        let n = 2 + next(30) as usize;
        let k = 1 + next(if trial % 3 == 0 { 3 } else { 40 }) as u64;
        let x: Vec<i64> = (0..n).map(|_| next(k)).collect();
        let y: Vec<i64> = (0..n).map(|_| next(k)).collect();
        let mut want = ConcordanceCounts {
            concordant: 0,
            discordant: 0,
            ties_x: 0,
            ties_y: 0,
            ties_both: 0,
        };
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = (x[i].cmp(&x[j]), y[i].cmp(&y[j]));
                use std::cmp::Ordering::Equal;
                match (a, b) {
                    (Equal, Equal) => want.ties_both += 1,
                    (Equal, _) => want.ties_x += 1,
                    (_, Equal) => want.ties_y += 1,
                    _ if a == b => want.concordant += 1,
                    _ => want.discordant += 1,
                }
            }
        }
        assert_eq!(
            concordance_counts(&from_i64(&x), &from_i64(&y))?,
            want,
            "x = {x:?}, y = {y:?}"
        );
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Normalised mutual information
// ═══════════════════════════════════════════════════════════════════════════

/// `Min` / `Max` compared `Σ c_q ln q` in `f64`: for margins
/// `(1 − 10⁻³⁰, 10⁻³⁰)` and `(1 − 2·10⁻³⁰, 2·10⁻³⁰)` those sums are noise
/// around `±69`, and `Min` divided by the larger entropy (`0.495` for
/// `0.980`), `Max` by the smaller.
#[test]
fn nmi_min_max_decided_on_the_exact_entropies() -> R {
    let ctx = Context::new();
    let e: Q = "1/1000000000000000000000000000000".parse().unwrap();
    let joint = vec![vec![qi(1) - qi(2) * &e, e.clone()], vec![qi(0), e.clone()]];
    // mpmath (80 dps): I/min(H) = 0.98021771157908860078, I/max(H) = 0.49500502184551809379
    let min = normalized_mutual_information(&ctx, &joint, Norm::Min)?.eval_f64()?;
    close(min, 0.98021771157908860078, 1e-14, "Min");
    let max = normalized_mutual_information(&ctx, &joint, Norm::Max)?.eval_f64()?;
    close(max, 0.49500502184551809379, 1e-14, "Max");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. The Beta-Binomial predictive table
// ═══════════════════════════════════════════════════════════════════════════

/// `posterior_predictive_beta_binomial` formed three rising factorials per
/// mass and re-validated the table pairwise: `O(n²)` big-rational work and
/// symbolic comparisons, 22 s at `n = 300` (debug).  The masses are now one
/// exact ratio recursion.
#[test]
fn beta_binomial_predictive_masses_by_recursion() -> R {
    let ctx = Context::new();
    let pred = symplex::stats::estimation::posterior_predictive_beta_binomial(
        &ctx,
        &q(7, 3),
        &q(5, 2),
        40,
    )?;
    // Fraction: comb(40, 3) rf(7/3, 3) rf(5/2, 37) / rf(29/6, 40)
    //   = 315706917416061504278206830610972295164800/29193441429839812936906622332787739527542933
    //   (scipy: betabinom.pmf(3, 40, 7/3, 5/2) = 0.0108143097200374)
    let want: Q =
        "315706917416061504278206830610972295164800/29193441429839812936906622332787739527542933"
            .parse()
            .unwrap();
    assert_eq!(pred.density(&ctx.int(3)).simplify(), ctx.from_ratio(want));
    // n α / (α + β) = 40 (7/3) / (29/6) = 560/29
    assert_eq!(pred.mean(), ctx.rational(560, 29));
    let big = symplex::stats::estimation::posterior_predictive_beta_binomial(
        &ctx,
        &q(7, 3),
        &q(5, 2),
        300,
    )?;
    // scipy: betabinom.pmf(3, 300, 7/3, 5/2) = 0.00012627829702840322
    close(
        big.density(&ctx.int(3)).eval_f64()?,
        0.00012627829702840322,
        1e-13,
        "pmf(3), n = 300",
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Quantile conventions
// ═══════════════════════════════════════════════════════════════════════════

/// `Exclusive` is Hyndman & Fan's definition 6 clamped to the sample range
/// (numpy `'weibull'`); it was documented as `statistics.quantiles`, which
/// agrees only between the levels `1/(n+1)` and `n/(n+1)`.
#[test]
fn exclusive_quantile_clamps_like_numpy_weibull() -> R {
    let d = from_i64(&[1, 2, 3]);
    // numpy: percentile([1, 2, 3], 1, method='weibull') = 1.0
    // (statistics.quantiles([1, 2, 3], n=100)[0] = Fraction(1, 25))
    assert_eq!(quantile(&d, &q(1, 100), QuantileMethod::Exclusive)?, qi(1));
    assert_eq!(quantiles(&d, 100, QuantileMethod::Exclusive)?[0], qi(1));
    // Inside the levels both agree: statistics.quantiles([2,4,4,4,5,5,7,9], n=4) = [4, 9/2, 13/2]
    let d = from_i64(&[2, 4, 4, 4, 5, 5, 7, 9]);
    assert_eq!(
        quantiles(&d, 4, QuantileMethod::Exclusive)?,
        vec![qi(4), q(9, 2), q(13, 2)]
    );
    Ok(())
}
