//! Tests of session 15 (s15p4): statistics leftovers and the oracle's seed set.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::errors::Degeneracy;
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::agreement::RatingTable;
use symplex::stats::anova::{Adjustment, pairwise_t_tests};
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::{
    Alternative, fisher_exact, kruskal_wallis, pearson_test, t_test_one_sample, t_test_paired,
    t_test_two_sample,
};
use symplex::stats::regression::{LogitOpts, logit};
use symplex::stats::reliability::{cronbach_alpha, kr20};

use crate::perf_assert::CpuTimer;

/// `got` agrees with `want` in its first `n` significant digits (decimal
/// strings with an optional exponent).
fn assert_digits(got: &str, want: &str, n: usize) {
    let split = |s: &str| -> (String, i64) {
        let s = s.trim_start_matches('-');
        let (mant, exp) = s
            .split_once('e')
            .map_or((s, 0), |(m, e)| (m, e.parse().unwrap_or(0)));
        let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
        let all = format!("{int}{frac}");
        let lead = all.len() - all.trim_start_matches('0').len();
        let sig = all.trim_start_matches('0').to_string();
        (sig, exp + int.len() as i64 - 1 - lead as i64)
    };
    let ((g, ge), (w, we)) = (split(got), split(want));
    assert!(
        got.starts_with('-') == want.starts_with('-')
            && ge == we
            && g.len() >= n
            && w.len() >= n
            && g[..n] == w[..n],
        "{got} does not agree with {want} to {n} digits"
    );
}

#[track_caller]
fn zero_variance<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::DegenerateData {
            kind: Degeneracy::ZeroVariance,
            ..
        }) => {}
        other => panic!("expected DegenerateData(ZeroVariance), got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Fisher's numeric route says it is an approximation
// ═══════════════════════════════════════════════════════════════════════════

/// Beyond 2 000 support points `fisher_exact` walks the pmf in `f64`, and up
/// to 0.58 it returned the p-value as `ctx.from_f64(p)`: a dyadic rational
/// that `p_value_exact()` handed out as if it were the exact one
/// (`Some(562500784653773/562949953421312)` for the two-sided p of this
/// table, whose exact value is a rational with a 600 000-digit
/// denominator).  Now `p_value_exact()` is `None` on that route, and the
/// values are unchanged: mpmath (40 digits, the pmf walked from the mode
/// and normalised by its total; also `math.comb`) two-sided
/// `0.99920211598773724586`, less `0.49960106298063621942`, greater
/// `0.50119681943588244659`.
#[test]
fn fisher_numeric_route_is_not_an_exact_rational() {
    let ctx = Context::new();
    let table = [[1_000_000, 1_000_007], [999_997, 1_000_000]];
    for (alt, want) in [
        (Alternative::TwoSided, 0.999_202_115_987_737_2),
        (Alternative::Less, 0.499_601_062_980_636_2),
        (Alternative::Greater, 0.501_196_819_435_882_5),
    ] {
        let r = fisher_exact(&ctx, table, alt).unwrap();
        assert_eq!(r.p_value_exact(), None, "{alt:?}");
        let p = r.p_value_f64().unwrap();
        assert!((p / want - 1.0).abs() < 1e-13, "{alt:?}: {p}");
        // The logarithm is exact: the expression is exp(ln p).
        assert!((r.p_value_ln().unwrap() - want.ln()).abs() < 1e-13);
    }
    // The exact route below the threshold is untouched (scipy 5/143).
    let small = fisher_exact(&ctx, [[8, 2], [1, 5]], Alternative::TwoSided).unwrap();
    assert_eq!(small.p_value_exact(), Some(q(5, 143)));
}

/// `P(X ≤ 5000)` of `[[5000, 3000], [2000, 6000]]` lies within `10⁻⁵¹¹` of
/// 1 (mpmath: `log10 P(X ≥ 5001) = −511.8`); up to 0.58 the numeric route
/// rounded its logarithm to 0 and returned the exact rational `1`.  A tail
/// that is the whole support is exactly 1 and stays so: `P(X ≤ 3000)` of
/// `[[3000, 0], [0, 3000]]` (3 001 support points, `a` at its upper end).
#[test]
fn fisher_numeric_route_claims_one_only_for_the_whole_support() {
    let ctx = Context::new();
    let near_one = fisher_exact(&ctx, [[5000, 3000], [2000, 6000]], Alternative::Less).unwrap();
    assert_eq!(near_one.p_value_exact(), None);
    assert_eq!(near_one.p_value_f64().unwrap(), 1.0);
    let whole = fisher_exact(&ctx, [[3000, 0], [0, 3000]], Alternative::Less).unwrap();
    assert_eq!(whole.p_value_exact(), Some(Q::from_integer(1.into())));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Zero variance is degenerate data, as in the models
// ═══════════════════════════════════════════════════════════════════════════

/// The t-tests (behind `anova::pairwise_t_tests`), the rank and correlation
/// tests and the reliability coefficients reported a zero variance as
/// `InvalidArgument` up to 0.58, while one-way ANOVA, Tukey's HSD and the
/// regressions said `DegenerateData { kind: ZeroVariance }` since 0.57: a
/// pipeline could not skip a constant group the same way everywhere.  (The
/// statistics are undefined, `0/0`: scipy returns NaN for each.)
#[test]
fn zero_variance_is_degenerate_data_in_tests_and_reliability() {
    let ctx = Context::new();
    let alt = Alternative::TwoSided;
    let constant = from_i64(&[3, 3, 3]);
    zero_variance(t_test_one_sample(&ctx, &constant, &qi(0), alt));
    zero_variance(t_test_two_sample(
        &ctx,
        &from_i64(&[1, 1]),
        &from_i64(&[2, 2]),
        false,
        alt,
    ));
    zero_variance(t_test_paired(
        &ctx,
        &from_i64(&[1, 2, 3]),
        &from_i64(&[2, 3, 4]),
        alt,
    ));
    zero_variance(pearson_test(&ctx, &constant, &from_i64(&[1, 2, 3]), alt));
    zero_variance(kruskal_wallis(
        &ctx,
        &[from_i64(&[1, 1]), from_i64(&[1, 1])],
    ));
    zero_variance(pairwise_t_tests(
        &ctx,
        &[
            from_i64(&[1, 1, 1]),
            from_i64(&[2, 2, 2]),
            from_i64(&[1, 2, 3]),
        ],
        Adjustment::Holm,
        0.05,
    ));
    // Every total score is 5: α and KR-20 divide by its zero variance.
    let anti = RatingTable::from_i64(&[&[1, 4], &[2, 3], &[3, 2], &[4, 1]]).unwrap();
    zero_variance(cronbach_alpha(&anti));
    zero_variance(kr20(
        &RatingTable::from_i64(&[&[1, 0], &[0, 1], &[1, 0]]).unwrap(),
    ));
    // Shape errors stay InvalidArgument.
    assert!(matches!(
        t_test_one_sample(&ctx, &from_i64(&[3]), &qi(0), alt),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Refusals and slow paths
// ═══════════════════════════════════════════════════════════════════════════

/// `betainc_regularized(10¹⁵, 6·10¹¹, 0, 5000/5003)` — the CDF of a beta
/// with huge shapes next to its median — was `PrecisionExhausted`: the
/// sensitivity of `I_x` to its rounded limit `x` was bounded with `f64`
/// values of `ln B(a, b)` (each `ln Γ` known to `10⁻¹⁴·a ln a`, about 700
/// nats here) and of `a·log₂ x`, so the bound came out `2^1000` times too
/// large and never certified; for `B(1.21·10¹⁷, 7.388·10¹³)` (the nightly
/// oracle's case) the bound was `2^268610` and the refusal took 64 s.  The
/// integrand's logarithm is now taken in arbitrary precision beyond shapes
/// of `2³²`.  mpmath (the continued fraction of DLMF 8.17.22 with the
/// prefactor from `loggamma`, at 60 and 75 digits):
/// `0.49999982847703250648285780117669885216638886`.
#[test]
fn incomplete_beta_with_huge_shapes_near_the_median_is_evaluated() {
    let ctx = Context::new();
    let x = ctx.rational(5000, 5003);
    let i = x.betainc_regularized(
        &ctx.parse("1000000000000000").unwrap(),
        &ctx.parse("600000000000").unwrap(),
        &ctx.zero(),
    );
    let t0 = CpuTimer::start();
    let v = i.eval_f64().unwrap();
    t0.assert_within(Duration::from_secs(60), "I_x(1e15, 6e11) at its median");
    assert!((v / 0.499_999_828_477_032_5 - 1.0).abs() < 1e-15, "{v}");
}

/// `ln Γ` at its zeros 1 and 2: `loggamma(1 + 10⁻⁷⁰)` was
/// `PrecisionExhausted` — the argument rounds to 1 at the first
/// precision, the Stirling sum was measured as having lost every bit
/// against a value of 0, and the evaluation stopped before the argument's
/// precision was raised.  `ln Γ(1 + ε) = −γε + ζ(2)ε²/2 − …` and `ln Γ(2 + ε)
/// = (1 − γ)ε + …` (DLMF 5.7.3): `−5.77215664901532860606512090082e-71`,
/// `4.22784335098467139393487909918e-71` (mpmath `loggamma(1 + mpf(10)**-70)`
/// at 120 digits).
#[test]
fn log_gamma_next_to_its_zeros_is_evaluated() {
    let ctx = Context::new();
    let at = |s: &str| ctx.parse(s).unwrap().log_gamma().eval_decimal(30).unwrap();
    assert_digits(
        &at("1 + 10^-70"),
        "-5.77215664901532860606512090082e-71",
        29,
    );
    assert_digits(&at("2 + 10^-70"), "4.22784335098467139393487909918e-71", 29);
    assert_digits(&at("1 - 10^-80"), "5.77215664901532860606512090082e-81", 29);
}

/// `ln(beta(10¹², 10¹²))` was refused ("Beta: the value lies beyond the
/// arbitrary-precision exponent range"): `B ≈ 2^(−2·10¹²)` is below the
/// range, its logarithm is not.  mpmath at 50 digits,
/// `loggamma(a) + loggamma(b) − loggamma(a + b)`:
/// `−1386294361132.44061726894374662` and, for `a = 10¹² + ⅓`,
/// `−1386294361132.67166632913045062`.  `B` itself is refused by
/// `eval_decimal` as below the range and is `0.0` in `eval_f64` (the
/// documented underflow rule; it was an error).
#[test]
fn logarithm_of_a_beta_below_the_range_is_evaluated() {
    let ctx = Context::new();
    let big = ctx.parse("10^12").unwrap();
    let ln_b = big.beta(&big).ln();
    assert_digits(
        &ln_b.eval_decimal(30).unwrap(),
        "-1386294361132.44061726894374662",
        29,
    );
    let third = ctx.parse("10^12 + 1/3").unwrap().beta(&big).ln();
    assert_digits(
        &third.eval_decimal(30).unwrap(),
        "-1386294361132.67166632913045062",
        29,
    );
    assert!(big.beta(&big).eval_decimal(20).is_err());
    assert_eq!(big.beta(&big).eval_f64().unwrap(), 0.0);
}

/// The F distribution's upper quantile with a huge `d₁`: `f::isf(1e-20, 3e300,
/// 42/11)` was `41499494283.40515`, `1.6·10⁻¹³` off — the small-argument
/// series of the tail (its argument `d₂/(d₁x + d₂) ≈ 3·10⁻³¹¹` is subnormal)
/// took its prefactor as `p·ln w − ln B(p, q)`, two exponents near 1300 that
/// cancel to 46.  For `d₁ → ∞`, `S(x) = P(d₂/2, (d₁/2)·d₂/(d₁x + d₂))`
/// to `O(1/d₁)`, so `x = d₂/(2z) − d₂/d₁` with `P(d₂/2, z) = 10⁻²⁰` (mpmath,
/// 50 digits): `41499494283.41195519167` at the double `d₂`, and
/// `41499494283.411996769539` at `d₂ = 42/11` (the distribution's `isf_f64`).
#[test]
fn f_upper_quantile_with_a_huge_numerator_shape_is_accurate() {
    let x = symplex::stats::numdist::f::isf(1e-20, 3e300, 3.818_181_818_181_818_3).unwrap();
    assert!((x / 41_499_494_283.411_96 - 1.0).abs() < 2e-15, "{x}");
    let ctx = Context::new();
    let d1 = ctx.parse("3*10^300").unwrap();
    let f = symplex::stats::Distribution::f_distribution(d1, ctx.rational(42, 11));
    let x = f.isf_f64(1e-20).unwrap();
    assert!((x / 41_499_494_283.412 - 1.0).abs() < 2e-15, "{x}");
}

/// A logistic regression whose regressor varies only on a subnormal scale
/// (`|x| ≈ 10⁻³¹⁰`) was refused as a collinear design: the
/// standardisation skipped the column (`1/scale` overflows), its squares
/// underflowed in the information matrix, and the factorisation failed.
/// The coefficient and standard error are of order `10³¹⁰`, beyond the
/// `f64` range — that is the refusal now, as an `InvalidArgument` that
/// says so (L4's reproducer).
#[test]
fn logit_with_subnormal_regressors_says_why_it_refuses() {
    let y = [0u8, 1, 0, 1, 1, 0];
    let x: Vec<Vec<f64>> = [0.0, 1e-310, 2e-310, -1e-310, 3e-310, 0.5e-310]
        .iter()
        .map(|&v| vec![v])
        .collect();
    match logit(&y, &x, true, &LogitOpts::default()) {
        Err(SymplexError::InvalidArgument { reason, .. }) => {
            assert!(reason.contains("below 2^-1024"), "{reason}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    // The same data scaled by 2^1100 (exactly, in two steps) fit: the scale
    // is the only issue.
    let scaled: Vec<Vec<f64>> = x
        .iter()
        .map(|r| vec![r[0] * 2f64.powi(550) * 2f64.powi(550)])
        .collect();
    assert!(logit(&y, &scaled, true, &LogitOpts::default()).is_ok());
}
