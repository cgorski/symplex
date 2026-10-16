//! Tests of session 15 (s15l4): regression, ANOVA, multivariate, survival, Cox.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::linprog::q;
use symplex::prelude::*;
use symplex::stats::anova::tukey_hsd;
use symplex::stats::cox::{CoxOpts, cox_ph};
use symplex::stats::data::from_i64;
use symplex::stats::regression::{Ols, ols};
use symplex::stats::survival::Observation;

fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: {actual} vs {expected}"
    );
}

/// `2^e` as an exact rational.
fn pow2(e: i32) -> Q {
    let one = Q::from_integer(1.into());
    let base = Q::from_integer(2.into());
    (0..e.unsigned_abs()).fold(one, |acc, _| if e >= 0 { acc * &base } else { acc / &base })
}

/// The documented example fit, its response scaled by `s`.
fn scaled_fit(s: &Q) -> Result<Ols, SymplexError> {
    let x: Vec<Vec<Q>> = from_i64(&[1, 2, 3, 4, 5, 6, 7])
        .into_iter()
        .map(|v| vec![v])
        .collect();
    let y: Vec<Q> = from_i64(&[2, 3, 5, 4, 6, 8, 9])
        .into_iter()
        .map(|v| v * s)
        .collect();
    ols(&y, &x, true)
}

/// Least squares is equivariant under scaling the response: `y → s·y`
/// scales the coefficients, standard errors and every limit by `s`
/// exactly, and for `s = 2^±560` the `f64` limits are the unscaled ones
/// times `s` (a power of two, exact in `f64`).  Up to 0.56 the interval
/// methods converted the exact variance `σ̂²·[(XᵀX)⁻¹]ⱼⱼ` (of order `s²`) to
/// `f64` before the square root: for `s = 2^−560 ≈ 2.6·10⁻¹⁶⁹` it
/// underflowed to `0` and every interval was a single point
/// (`conf_int()[1] = [b, b]` with `b = 2.98·10⁻¹⁶⁹`), and for `s = 2^560` it
/// overflowed and `conf_int` was refused ("does not fit in an f64")
/// although the limits are `≈ 10¹⁶⁸`.  Oracle: statsmodels 0.15
/// `OLS(y, add_constant(x)).fit().conf_int(0.05)` = `[[-0.9279987789168567,
/// 2.3565702074882813], [0.7756311663376696, 1.5100831193766169]]`,
/// `get_prediction([1, 4]).summary_frame(alpha=0.05)`: mean CI
/// `[4.551262332675338, 6.020166238753231]`, obs CI
/// `[3.2083704599161127, 7.363058111512457]`; the scaled limits by the
/// exact identity.
#[test]
fn ols_intervals_keep_their_width_for_data_near_the_ends_of_the_f64_range()
-> Result<(), SymplexError> {
    let reference = [
        [-0.927_998_778_916_856_7, 2.356_570_207_488_281_3],
        [0.775_631_166_337_669_6, 1.510_083_119_376_616_9],
    ];
    let mean_ci = [4.551_262_332_675_338, 6.020_166_238_753_231];
    let obs_ci = [3.208_370_459_916_112_7, 7.363_058_111_512_457];
    for e in [0, -560, 560] {
        let s = pow2(e);
        let sf = 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
        let fit = scaled_fit(&s)?;
        for (iv, r) in fit.conf_int(0.95)?.iter().zip(&reference) {
            close(iv.lower, r[0] * sf, 1e-13, "conf_int lower");
            close(iv.upper, r[1] * sf, 1e-13, "conf_int upper");
            assert!(iv.lower < iv.upper, "2^{e}: a point interval");
        }
        let x0 = [q(4, 1)];
        let m = fit.confidence_interval_mean_response(&x0, 0.95)?;
        close(m.lower, mean_ci[0] * sf, 1e-13, "mean lower");
        close(m.upper, mean_ci[1] * sf, 1e-13, "mean upper");
        let o = fit.prediction_interval(&x0, 0.95)?;
        close(o.lower, obs_ci[0] * sf, 1e-13, "obs lower");
        close(o.upper, obs_ci[1] * sf, 1e-13, "obs upper");
    }
    Ok(())
}

/// A limit beyond the `f64` range is refused, never infinite: the response
/// scaled by `2^1100` has limits near `10³³¹`.
#[test]
fn ols_interval_beyond_the_f64_range_is_refused() -> Result<(), SymplexError> {
    let fit = scaled_fit(&pow2(1100))?;
    assert!(matches!(
        fit.conf_int(0.95),
        Err(SymplexError::ComputationFailed { .. })
    ));
    Ok(())
}

/// Tukey's HSD intervals of groups scaled by `2^−560`: up to 0.56 the
/// variance of each difference underflowed before its square root and
/// every interval was the point `[d, d]`.  Oracle: scipy 1.18.1
/// `tukey_hsd(*g).confidence_interval(0.95)` for groups 0 and 1:
/// `[-7.1929989958799485, -0.8070010041200515]`, times `2^−560` exactly.
#[test]
fn tukey_intervals_keep_their_width_for_tiny_data() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let groups = |s: &Q| -> Vec<Vec<Q>> {
        [
            &[6, 8, 4, 5, 3, 4][..],
            &[8, 12, 9, 11, 6, 8],
            &[13, 9, 11, 8, 7, 12],
        ]
        .iter()
        .map(|g| from_i64(g).into_iter().map(|v| v * s).collect())
        .collect()
    };
    let sf = 2f64.powi(-280) * 2f64.powi(-280);
    let plain = tukey_hsd(&ctx, &groups(&pow2(0)), 0.95)?;
    let tiny = tukey_hsd(&ctx, &groups(&pow2(-560)), 0.95)?;
    close(
        plain[0].ci.lower,
        -7.192_998_995_879_948_5,
        1e-9,
        "scipy lower",
    );
    close(
        plain[0].ci.upper,
        -0.807_001_004_120_051_5,
        1e-9,
        "scipy upper",
    );
    for (a, b) in plain.iter().zip(&tiny) {
        close(b.ci.lower, a.ci.lower * sf, 1e-13, "scaled lower");
        close(b.ci.upper, a.ci.upper * sf, 1e-13, "scaled upper");
        assert_eq!(a.p_adj, b.p_adj, "the p-value is scale-free");
    }
    Ok(())
}

/// `CoxOpts { tol: ∞ }` declared convergence after the second Newton step:
/// up to 0.56 the fit returned `β̂ = 0.5412739218906902` (`converged =
/// true`), wrong in the fourth digit.  A tolerance must be finite (as for
/// `LogitOpts`).  Oracle for the default fit: Newton's method at 40
/// digits on the Efron partial likelihood, `β̂ = 0.54135098088723158177`;
/// statsmodels `PHReg(...).fit().params = [0.5413509808872312]`.
#[test]
fn cox_rejects_an_infinite_tolerance() -> Result<(), SymplexError> {
    let obs = Observation::from_i64(&[1, 2, 3, 4, 5, 6], &[true, true, false, true, true, false]);
    let x: Vec<Vec<f64>> = [0.5, 1.0, 0.0, 2.0, 1.5, 0.3]
        .iter()
        .map(|&v| vec![v])
        .collect();
    let loose = CoxOpts {
        tol: f64::INFINITY,
        ..CoxOpts::default()
    };
    assert!(matches!(
        cox_ph(&obs, &x, &loose),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let fit = cox_ph(&obs, &x, &CoxOpts::default())?;
    close(fit.coefficients[0], 0.541_350_980_887_231_5, 1e-9, "beta");
    Ok(())
}

/// Least squares stays exact for a nearly collinear design (`x₂ = x₁ ±
/// 2⁻⁴⁰`, condition number of `XᵀX` about `10²⁶`): the exact coefficients
/// are `[621/4120, −54150947667601/206, 27075473833984/103]`.  Oracle: exact
/// Fraction normal equations; statsmodels 0.15 (pinv) gives `[0.15081065160596863,
/// -262882442565.81396, 262882442567.59546]`, relative error `5·10⁻⁵`.
#[test]
fn ols_is_exact_for_a_nearly_collinear_design_where_statsmodels_is_not() -> Result<(), SymplexError>
{
    let s = [1, -1, -1, 1, 1, -1, 1, -1, -1, 1];
    let eps = pow2(-40);
    let x: Vec<Vec<Q>> = (0..10)
        .map(|i| {
            let x1 = q(i + 1, 2);
            let x2 = &x1 + &eps * Q::from_integer(s[i as usize].into());
            vec![x1, x2]
        })
        .collect();
    let y: Vec<Q> = [2, 5, 4, 9, 8, 11, 14, 13, 16, 19]
        .iter()
        .map(|&v| q(v, 2))
        .collect();
    let fit = ols(&y, &x, true)?;
    assert_eq!(
        fit.coefficients,
        vec![
            q(621, 4120),
            q(-54_150_947_667_601, 206),
            q(27_075_473_833_984, 103)
        ]
    );
    Ok(())
}

/// The studentized range of two groups is `√2·|T_ν|`, so its upper tail
/// is `I_{ν/(ν+q²/2)}(ν/2, ½)`: at `q = 19.503304907072128`, `ν = 10` it is
/// `7.8187283512318594677e-8` (mpmath at 30 and 60 digits).  scipy 1.18.1
/// `studentized_range.sf(q, 2, 10)` gives `7.818728131603336e-08`
/// (relative error `2.8·10⁻⁸`).
#[test]
fn studentized_range_far_tail_is_better_than_scipy() -> Result<(), SymplexError> {
    let sf = symplex::stats::anova::studentized_range_sf(19.503_304_907_072_128, 2, 10.0)?;
    close(sf, 7.818_728_351_231_859e-8, 1e-13, "sf");
    Ok(())
}

/// The design `x₂ = x₁ ± 2⁻ᵏ` (twelve observations) for the `f64`
/// maximum-likelihood fits.
fn nearly_collinear(k: i32) -> Vec<Vec<f64>> {
    let s = [
        1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    ];
    (0..12)
        .map(|i| {
            let x1 = 0.5 * f64::from(i + 1);
            vec![x1, x1 + s[i as usize] * 2f64.powi(-k)]
        })
        .collect()
}

/// Logistic regression on a nearly collinear design.  At `k = 17`
/// (condition number of the standardised design about `2·10⁵`) the
/// estimate is `[−2.7439494082218644, 67475.526061484270, −67474.507691957014]`
/// with standard errors `[1.7765751712067759, 102312.97781236483,
/// 102312.86677249704]` (Newton's method at 40 and 60 digits); symplex's
/// square-root information keeps about ten digits, while statsmodels 0.15
/// `Logit(...).fit(method="newton")` gives `67475.01427545313` and standard
/// error `102312.041225038` (relative errors `8·10⁻⁶`, `9·10⁻⁶`).  At `k =
/// 20` the design is refused as collinear (a documented numerical
/// threshold), where statsmodels returns `96974.03653584441` for the
/// coefficient `539797.07990518337` — a factor 5.6 off.
#[test]
fn logit_on_a_nearly_collinear_design_is_accurate_or_refused() -> Result<(), SymplexError> {
    use symplex::errors::Degeneracy;
    use symplex::stats::regression::{LogitOpts, logit};
    let y = [0u8, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1];
    let fit = logit(&y, &nearly_collinear(17), true, &LogitOpts::default())?;
    let b = [
        -2.743_949_408_221_864_5,
        67_475.526_061_484_27,
        -67_474.507_691_957_01,
    ];
    let se = [
        1.776_575_171_206_775_9,
        102_312.977_812_364_83,
        102_312.866_772_497_04,
    ];
    for j in 0..3 {
        close(fit.coefficients[j], b[j], 1e-9, "beta");
        close(fit.standard_errors[j], se[j], 1e-9, "se");
    }
    let refused = logit(&y, &nearly_collinear(20), true, &LogitOpts::default());
    assert_eq!(
        refused.err().and_then(|e| e.degeneracy()),
        Some(Degeneracy::Collinear)
    );
    Ok(())
}

/// The Cox fit on the same design (`k = 17`): the coefficients are
/// accurate, the standard errors carry the conditioning of the formed
/// information matrix (`κ ≈ 10¹¹`, documented: relative error up to about
/// `κ·10⁻¹⁶`).  Oracle: Newton's method at 40 and 60 digits on the Efron
/// partial likelihood, `β̂ = [45002.605651786745, −45003.311101765787]`,
/// standard errors `[50669.914747198221, 50669.968124333846]`; statsmodels
/// 0.15 `PHReg` gives standard errors `[50669.37772416778, ...]` (relative
/// error `1.1·10⁻⁵`).
#[test]
fn cox_on_a_nearly_collinear_design_keeps_its_documented_accuracy() -> Result<(), SymplexError> {
    let obs = Observation::from_i64(
        &[3, 5, 1, 8, 2, 9, 4, 7, 6, 12, 10, 11],
        &[
            true, true, false, true, true, true, false, true, true, false, true, true,
        ],
    );
    let fit = cox_ph(&obs, &nearly_collinear(17), &CoxOpts::default())?;
    let b = [45_002.605_651_786_75, -45_003.311_101_765_79];
    let se = [50_669.914_747_198_22, 50_669.968_124_333_85];
    for j in 0..2 {
        close(fit.coefficients[j], b[j], 1e-9, "beta");
        close(fit.standard_errors[j], se[j], 1e-4, "se");
    }
    Ok(())
}
