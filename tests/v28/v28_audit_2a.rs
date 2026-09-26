//! Audit round 2a of `symplex::stats`: `regression` (OLS/WLS, `Logit`,
//! `MnLogit`, `OrderedLogit`, the `LikelihoodFit` / `WaldFit` summaries).
//!
//! 1. Ordered-logit category probabilities (`predict_proba`,
//!    `fitted_probabilities`, `predict`) were `σ(u) − σ(l)` and `1 − σ(l)`:
//!    `0` for a true `2.6e-22`, relative errors of `2e-3` at `3e-14`, and
//!    `3e-7` between nearly tied thresholds.  They are now the product
//!    `σ(u) σ(−l) (−expm1(l − u))`; the likelihood kernel (`ln P`, score,
//!    information) no longer divides by `P` or subtracts densities.
//! 2. The standard errors of `logit` / `mnlogit` / `ologit` lost digits
//!    like `cond(XᵀWX) = cond(X)²`: `2.9e-6` relative for a regressor with
//!    offset `1e6` (reported `converged`), `4e-8` for a polynomial design,
//!    and an offset of `1e8` — or units of `1e±300` — was refused as
//!    "rank deficient".  The iterations now run on a standardised design
//!    (an exact affine reparametrisation) and the covariance comes from
//!    the information in square-root form.
//! 3. Panics on caller input: category labels near `usize::MAX`
//!    (`max + 1` overflow; a label of `2⁴⁰` allocated `2⁴⁰` counters) and
//!    `polyfit(…, usize::MAX)`.
//! 4. `LogitOpts { tol: ∞ }` declared convergence after one Newton step.
//!
//! OLS/WLS (exact over ℚ) were differentially tested against exact
//! `Fraction` arithmetic and mpmath (168 random and degenerate designs,
//! p-values down to `1e-120`) with no discrepancy; test 6 is a guard.
//!
//! Reference values: statsmodels 0.15.0 by the call quoted, mpmath 1.3 at
//! 50–60 digits (`target/scratch/logit_mp.py`: Newton's method on the
//! logistic likelihood at 60 digits with the same IEEE inputs;
//! `target/scratch/ologit_tail_mp.py`: `σ(θ_j − xβ) − σ(θ_{j−1} − xβ)` at 50
//! digits), exact rationals from `fractions.Fraction`.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::linprog::q;
use symplex::prelude::*;
use symplex::stats::data::from_i64;
use symplex::stats::regression::{
    LogitOpts, OrderedLogit, logit, mnlogit, ologit, polyfit, simple_linear_regression,
};

/// `actual` within `rel` of `expected`, relatively.
fn close(actual: f64, expected: f64, rel: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "{label}: got {actual:e}, expected {expected:e} (rel {:.1e})",
        (actual / expected - 1.0).abs()
    );
}

fn invalid_reason<T: std::fmt::Debug>(r: Result<T, SymplexError>) -> String {
    match r {
        Err(SymplexError::InvalidArgument { reason, .. }) => reason,
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

/// An ordered logit with given cut points and one slope (only the fields
/// `predict_proba` / `cumulative_proba` read matter).
fn ordered(thresholds: Vec<f64>, beta: f64) -> OrderedLogit {
    OrderedLogit {
        n_categories: thresholds.len() + 1,
        thresholds,
        coefficients: vec![beta],
        standard_errors: vec![],
        z_values: vec![],
        p_values: vec![],
        log_likelihood: 0.0,
        null_log_likelihood: 0.0,
        pseudo_r_squared: 0.0,
        iterations: 0,
        converged: true,
        cov_params: vec![],
        fitted_probabilities: vec![],
        nobs: 0,
        df_model: 1,
        df_resid: 0,
    }
}

/// The binary fixture of tests 2–3: `yᵢ = [(7i + 3) mod 11 < ⌊i/3⌋]`,
/// `i = 0, …, 29`, not separated.
fn binary30() -> Vec<u8> {
    (0..30)
        .map(|i| u8::from((i * 7 + 3) % 11 < (i / 3)))
        .collect()
}

/// Three ordered/unordered categories on `i = 0, …, 39`.
fn three40() -> Vec<usize> {
    (0..40).map(|i| ((i * 7 + 3) % 11 + i / 4) % 3).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Ordered-logit probabilities in the tails
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `predict_proba` subtracted cumulative probabilities, so at
/// `x = −400` (`θ − xβ ≈ 50`) the two upper categories were exactly `0`,
/// and at `x = −240` the top one was `2.6867e-14` (true `2.6810e-14`).
/// Now each factor of `σ(u) σ(−l)(1 − e^{l−u})` is accurate to a few ulps,
/// hence `1e-14`.
#[test]
fn ologit_category_probabilities_in_the_far_tails() -> Result<(), SymplexError> {
    let fit = ordered(vec![-0.5, 1.25], 0.125);
    // mpmath (ologit_tail_mp.py): θ = (−1/2, 5/4), β = 1/8
    //   x = −400: [1.0, 2.6273748168127246932e-22, 5.5259608338502480501e-23]
    //   x = −240: [0.99999999999984571888, 1.2747108164134766719e-13, 2.6810038677817313443e-14]
    //   x = 400:  [1.1698459177061964686e-22, 5.5621525308402612395e-22, 1.0]
    let far = fit.predict_proba(&[-400.0])?;
    assert_eq!(far[0], 1.0);
    close(far[1], 2.627_374_816_812_724_693_2e-22, 1e-14, "P1(-400)");
    close(far[2], 5.525_960_833_850_248_050_1e-23, 1e-14, "P2(-400)");
    let mid = fit.predict_proba(&[-240.0])?;
    close(mid[0], 0.999_999_999_999_845_718_88, 1e-15, "P0(-240)");
    close(mid[1], 1.274_710_816_413_476_671_9e-13, 1e-14, "P1(-240)");
    close(mid[2], 2.681_003_867_781_731_344_3e-14, 1e-14, "P2(-240)");
    let high = fit.predict_proba(&[400.0])?;
    close(high[0], 1.169_845_917_706_196_468_6e-22, 1e-14, "P0(400)");
    close(high[1], 5.562_152_530_840_261_239_5e-22, 1e-14, "P1(400)");
    assert_eq!(high[2], 1.0);
    // The cumulative curve was never the problem (mpmath: σ(θ_j + 30) =
    //   0.99999999999984571888, 0.99999999999997318996).
    let cum = fit.cumulative_proba(&[-240.0])?;
    close(cum[0], 0.999_999_999_999_845_718_88, 1e-15, "F0(-240)");
    close(cum[1], 0.999_999_999_999_973_189_96, 1e-15, "F1(-240)");
    assert_eq!(fit.predict(&[-400.0])?, 0);
    Ok(())
}

/// Before: between cut points `1` and `1 + 2⁻³⁰` the middle category was
/// `1.83109083e-10` (relative error `2.6e-7`: `σ(u) − σ(l)` cancels);
/// now `−expm1(l − u)` carries the gap exactly.
#[test]
fn ologit_probability_between_nearly_tied_thresholds() -> Result<(), SymplexError> {
    let fit = ordered(vec![1.0, 1.0 + 2f64.powi(-30)], 0.125);
    // mpmath (ologit_tail_mp.py): θ = (1, 1 + 2^-30), x = 0:
    //   [0.73105857863000487925, 1.8310913182718019633e-10, 0.26894142118688598892]
    let pr = fit.predict_proba(&[0.0])?;
    close(pr[0], 0.731_058_578_630_004_879_25, 1e-15, "P0");
    close(pr[1], 1.831_091_318_271_801_963_3e-10, 1e-14, "P1");
    close(pr[2], 0.268_941_421_186_885_988_92, 1e-15, "P2");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Conditioning: offsets, units, polynomial designs
// ═══════════════════════════════════════════════════════════════════════════

/// Before: with the regressor `10⁶ + i` the slope's standard error was
/// `0.05941339203346359` (relative error `2.9e-6`, `cond(XᵀWX)·ε`) while the
/// fit reported `converged`, and with `10⁸ + i` the fit was refused as
/// "rank deficient".  (statsmodels' `Logit(...).fit(method='newton')` stops
/// with a ConvergenceWarning and wrong values from `10⁶` on.)  Now the
/// fit runs on the centred, scaled regressor and maps back exactly; the
/// Newton fixed point and the covariance are good to about `1e-15` here,
/// and `1e-11` leaves room for libm differences.
#[test]
fn logit_standard_errors_of_an_offset_regressor() -> Result<(), SymplexError> {
    let y = binary30();
    let opts = LogitOpts::default();
    // mpmath (logit_mp.py offset1e6): beta [-151303.84642201270213, 0.15130108903223019675],
    //   se [59414.537765600603636, 0.059413561490384587627], llf -15.72830759496838017470324
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![1e6 + f64::from(i)]).collect();
    let fit = logit(&y, &x, true, &opts)?;
    assert!(fit.converged);
    close(
        fit.coefficients[0],
        -151_303.846_422_012_702_13,
        1e-11,
        "b0(1e6)",
    );
    close(
        fit.coefficients[1],
        0.151_301_089_032_230_196_75,
        1e-11,
        "b1(1e6)",
    );
    close(
        fit.standard_errors[0],
        59_414.537_765_600_603_636,
        1e-11,
        "se0(1e6)",
    );
    close(
        fit.standard_errors[1],
        0.059_413_561_490_384_587_627,
        1e-11,
        "se1(1e6)",
    );
    close(
        fit.log_likelihood,
        -15.728_307_594_968_380_174_7,
        1e-13,
        "llf(1e6)",
    );
    // mpmath (logit_mp.py offset1e8): beta [-15130111.66061280218, 0.15130108903223019675],
    //   se [5941357.125312080945, 0.059413561490384587627]
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![1e8 + f64::from(i)]).collect();
    let fit = logit(&y, &x, true, &opts)?;
    assert!(fit.converged);
    close(
        fit.coefficients[0],
        -15_130_111.660_612_802_18,
        1e-11,
        "b0(1e8)",
    );
    close(
        fit.coefficients[1],
        0.151_301_089_032_230_196_75,
        1e-11,
        "b1(1e8)",
    );
    close(
        fit.standard_errors[0],
        5_941_357.125_312_080_945,
        1e-11,
        "se0(1e8)",
    );
    close(
        fit.standard_errors[1],
        0.059_413_561_490_384_587_627,
        1e-11,
        "se1(1e8)",
    );
    // The Wald p-value and interval follow the accurate standard error.
    let ci = fit.conf_int(0.95)?;
    assert!(ci[1].lower < fit.coefficients[1] && fit.coefficients[1] < ci[1].upper);
    Ok(())
}

/// Before: a regressor in units of `10³⁰⁰` or `10⁻³⁰⁰` was refused as
/// "rank deficient" (`XᵀWX` overflowed or underflowed).  The fit is
/// equivariant under a change of units: the slope and its standard error
/// scale by `10∓³⁰⁰`.  Oracle: the unit-scale fit, `1e-11` as above.
#[test]
fn logit_is_equivariant_under_extreme_units() -> Result<(), SymplexError> {
    let y = binary30();
    // mpmath (logit_mp.py offset0, x = i): beta [-2.7573897825053844751, 0.15130108903223019675],
    //   se [1.069774294579879468, 0.059413561490384587627]
    for (scale, b1, se1) in [
        (
            1e300,
            0.151_301_089_032_230_196_75e-300,
            0.059_413_561_490_384_587_627e-300,
        ),
        (
            1e-300,
            0.151_301_089_032_230_196_75e300,
            0.059_413_561_490_384_587_627e300,
        ),
    ] {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![scale * f64::from(i)]).collect();
        let fit = logit(&y, &x, true, &LogitOpts::default())?;
        close(
            fit.coefficients[0],
            -2.757_389_782_505_384_475_1,
            1e-11,
            "b0",
        );
        close(fit.coefficients[1], b1, 1e-11, "b1");
        close(
            fit.standard_errors[0],
            1.069_774_294_579_879_468,
            1e-11,
            "se0",
        );
        close(fit.standard_errors[1], se1, 1e-11, "se1");
    }
    Ok(())
}

/// Before: for the columns `t, t², …, t⁸` (`t = i/40`) the standard errors
/// were off by up to `7.4e-8` relative (inverting the formed `XᵀWX`
/// squares the condition number).  Now the information is accumulated as
/// Givens rotations of its square root; what remains is the conditioning
/// of the standardised monomial design itself (about `1e5`: its MLE and
/// covariance are good to `~1e-11`, observed `5e-12`), hence `1e-9`.
#[test]
fn logit_standard_errors_of_a_polynomial_design() -> Result<(), SymplexError> {
    let y: Vec<u8> = (0..41).map(|i| u8::from((i * 7 + 3) % 11 < 5)).collect();
    let x: Vec<Vec<f64>> = (0..41)
        .map(|i| {
            let t = f64::from(i) / 40.0;
            let mut v = 1.0;
            (1..=8)
                .map(|_| {
                    v *= t;
                    v
                })
                .collect()
        })
        .collect();
    let fit = logit(&y, &x, true, &LogitOpts::default())?;
    assert!(fit.converged);
    // mpmath (logit_mp.py poly8, the same IEEE products t·t·…):
    //   beta [1.62891715322866464, -139.3239800249717422, 2156.3547894847059253, -14046.139193545362168,
    //         47675.52183993811234, -91350.652524486536263, 99598.80271028926544, -57633.275484898558517,
    //         13736.385386669617587]
    //   se [2.419834700760672203, 117.41299647122236433, 1806.9310260759114728, 12354.428825204480142,
    //       44415.993105975173592, 89982.727345513136324, 103248.81694351435276, 62562.018447448168728,
    //       15543.783208439856676]
    //   llf -27.04092395730332508488431
    let beta = [
        1.628_917_153_228_664_64,
        -139.323_980_024_971_742_2,
        2_156.354_789_484_705_925_3,
        -14_046.139_193_545_362_168,
        47_675.521_839_938_112_34,
        -91_350.652_524_486_536_263,
        99_598.802_710_289_265_44,
        -57_633.275_484_898_558_517,
        13_736.385_386_669_617_587,
    ];
    let se = [
        2.419_834_700_760_672_203,
        117.412_996_471_222_364_33,
        1_806.931_026_075_911_472_8,
        12_354.428_825_204_480_142,
        44_415.993_105_975_173_592,
        89_982.727_345_513_136_324,
        103_248.816_943_514_352_76,
        62_562.018_447_448_168_728,
        15_543.783_208_439_856_676,
    ];
    for j in 0..9 {
        close(fit.coefficients[j], beta[j], 1e-9, &format!("beta[{j}]"));
        close(fit.standard_errors[j], se[j], 1e-9, &format!("se[{j}]"));
    }
    // ℓ sums ηᵢ = Σⱼ βⱼ xᵢⱼ over coefficients of 10⁵ with alternating signs:
    // its rounding is ~1e-13 relative here.
    close(
        fit.log_likelihood,
        -27.040_923_957_303_325_084_884_31,
        1e-11,
        "llf",
    );
    Ok(())
}

/// Before: `mnlogit` and `ologit` with the regressor `10⁸ + i` were refused
/// as "rank deficient" (statsmodels' `MNLogit(..., x + 1e8)` stops
/// unconverged at `llf −39.31` instead of `−38.646`).  A shift of the
/// regressor changes only the intercepts (`β₀ − 10⁸β₁`) or the cut points
/// (`θ_j + 10⁸β`); the slopes and their standard errors must not move.
/// The offset-0 fits are pinned to statsmodels first.
#[test]
fn mnlogit_and_ologit_are_invariant_under_a_regressor_offset() -> Result<(), SymplexError> {
    let y = three40();
    let opts = LogitOpts::default();
    let x0: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
    let x8: Vec<Vec<f64>> = (0..40).map(|i| vec![1e8 + f64::from(i)]).collect();

    // statsmodels: MNLogit(y, add_constant(i)).fit(method='newton', tol=1e-12):
    //   params.T [[-1.2295368450123052, -0.0029221633976426443], [-0.6908615334131918, 0.03190167155322667]]
    //   bse.T [[0.9278858334569905, 0.04450676186418017], [0.6924412619674059, 0.030154942145344685]]
    //   llf -38.64648782269525
    let m0 = mnlogit(&y, &x0, true, &opts)?;
    close(
        m0.coefficients[0][1],
        -0.002_922_163_397_642_644_3,
        1e-9,
        "mn b1",
    );
    close(
        m0.coefficients[1][1],
        0.031_901_671_553_226_67,
        1e-9,
        "mn b2",
    );
    close(
        m0.standard_errors[0][1],
        0.044_506_761_864_180_17,
        1e-9,
        "mn se1",
    );
    close(
        m0.standard_errors[1][1],
        0.030_154_942_145_344_685,
        1e-9,
        "mn se2",
    );
    close(m0.log_likelihood, -38.646_487_822_695_25, 1e-12, "mn llf");
    let m8 = mnlogit(&y, &x8, true, &opts)?;
    assert!(m8.converged);
    close(
        m8.log_likelihood,
        m0.log_likelihood,
        1e-13,
        "mn llf shifted",
    );
    for j in 0..2 {
        close(
            m8.coefficients[j][1],
            m0.coefficients[j][1],
            1e-12,
            "mn slope shifted",
        );
        close(
            m8.standard_errors[j][1],
            m0.standard_errors[j][1],
            1e-12,
            "mn se shifted",
        );
        let b0 = m0.coefficients[j][0] - 1e8 * m0.coefficients[j][1];
        close(m8.coefficients[j][0], b0, 1e-12, "mn intercept shifted");
        // var(β₀ − 10⁸β₁) from the offset-0 covariance (flat index (j − 1)p + a).
        let c = &m0.cov_params;
        let (a, s) = (2 * j, 2 * j + 1);
        let var = c[a][a] - 2e8 * c[a][s] + 1e16 * c[s][s];
        close(
            m8.standard_errors[j][0],
            var.sqrt(),
            1e-9,
            "mn intercept se shifted",
        );
    }

    // statsmodels: OrderedModel(y, i, distr='logit').fit(method='newton', tol=1e-14):
    //   params [0.025801283426135094, 0.29127318819446435, -0.6637768533593212]
    //   transform_threshold_params(params)[1:-1] [0.29127318819446435, 0.8061761326949473]
    //   bse [0.0254768167401985, ...] (a numerical Hessian: good to ~1e-7), llf -38.79501723866032
    let o0 = ologit(&y, &x0, &opts)?;
    close(
        o0.coefficients[0],
        0.025_801_283_426_135_094,
        1e-8,
        "ol beta",
    );
    close(
        o0.thresholds[0],
        0.291_273_188_194_464_35,
        1e-8,
        "ol theta0",
    );
    close(o0.thresholds[1], 0.806_176_132_694_947_3, 1e-8, "ol theta1");
    close(
        o0.standard_errors[0],
        0.025_476_816_740_198_5,
        1e-6,
        "ol se",
    );
    close(o0.log_likelihood, -38.795_017_238_660_32, 1e-12, "ol llf");
    let o8 = ologit(&y, &x8, &opts)?;
    assert!(o8.converged);
    close(
        o8.log_likelihood,
        o0.log_likelihood,
        1e-13,
        "ol llf shifted",
    );
    close(
        o8.coefficients[0],
        o0.coefficients[0],
        1e-12,
        "ol slope shifted",
    );
    close(
        o8.standard_errors[0],
        o0.standard_errors[0],
        1e-12,
        "ol se shifted",
    );
    for j in 0..2 {
        let theta = o0.thresholds[j] + 1e8 * o0.coefficients[0];
        close(o8.thresholds[j], theta, 1e-12, "ol theta shifted");
        let c = &o0.cov_params;
        let var = c[1 + j][1 + j] + 2e8 * c[0][1 + j] + 1e16 * c[0][0];
        close(
            o8.standard_errors[1 + j],
            var.sqrt(),
            1e-9,
            "ol theta se shifted",
        );
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Panics on caller input, 4. an infinite tolerance
// ═══════════════════════════════════════════════════════════════════════════

/// Before: `mnlogit(&[0, 1, usize::MAX], …)` panicked on `max + 1`
/// (debug) and a label of `2⁴⁰` allocated `2⁴⁰` counters.  With `n`
/// observations a label `≥ n` leaves some label below `n` unused, which is
/// now reported without allocating.  `polyfit(…, usize::MAX)` panicked on
/// `degree + 1`.
#[test]
fn huge_category_labels_and_degrees_are_refused() {
    let x: Vec<Vec<f64>> = (0..3).map(|i| vec![f64::from(i)]).collect();
    let opts = LogitOpts::default();
    let reason = invalid_reason(mnlogit(&[0, 1, usize::MAX], &x, true, &opts));
    assert!(
        reason.contains("category 2 has no observations"),
        "{reason}"
    );
    let reason = invalid_reason(mnlogit(&[0, 1, 1 << 40], &x, true, &opts));
    assert!(
        reason.contains("category 2 has no observations"),
        "{reason}"
    );
    let reason = invalid_reason(ologit(&[usize::MAX, 0, 1], &x, &opts));
    assert!(
        reason.contains("category 2 has no observations"),
        "{reason}"
    );
    let reason = invalid_reason(ologit(&[1, 2, 3], &x, &opts));
    assert!(
        reason.contains("category 0 has no observations"),
        "{reason}"
    );
    let reason = invalid_reason(polyfit(
        &from_i64(&[1, 2, 3]),
        &from_i64(&[2, 4, 7]),
        usize::MAX,
    ));
    assert!(reason.contains("needs more than"), "{reason}");
}

/// Before: `tol = ∞` passed the `tol > 0` check and every step met
/// `|Δβ| ≤ ∞`, so `logit` returned `converged: true` after one Newton step
/// with `β = [−1.345, 0.388]` (the MLE is `[−1.762, 0.528]`).
#[test]
fn an_infinite_tolerance_is_refused() {
    let y = [0u8, 0, 1, 0, 1, 1, 0, 1, 1, 1];
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
    let opts = LogitOpts {
        max_iter: 100,
        tol: f64::INFINITY,
    };
    let reason = invalid_reason(logit(&y, &x, true, &opts));
    assert!(reason.contains("finite"), "{reason}");
    let yc: Vec<usize> = y.iter().map(|&v| usize::from(v)).collect();
    invalid_reason(mnlogit(&yc, &x, true, &opts));
    invalid_reason(ologit(&yc, &x, &opts));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. The likelihood-ratio test's degenerate case
// ═══════════════════════════════════════════════════════════════════════════

/// A single-regressor logit without a constant has statsmodels'
/// `df_model = p − 1 = 0`; the refusal used to say "the model has no
/// regressors besides the constant", which is false there.
#[test]
fn llr_test_with_zero_degrees_of_freedom_names_the_real_reason() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let y = binary30();
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![f64::from(i) - 14.5]).collect();
    let fit = logit(&y, &x, false, &LogitOpts::default())?;
    assert_eq!(fit.df_model, 0);
    let reason = invalid_reason(fit.llr_test(&ctx));
    assert!(reason.contains("df_model = 0"), "{reason}");
    assert!(!reason.contains("constant"), "{reason}");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Guard: exact least squares in the far tail (no bug found)
// ═══════════════════════════════════════════════════════════════════════════

/// Regression guard for the exact OLS tails: `y = 10¹²x ± 1` on five
/// points, `t = 2.5·10¹²` with 3 degrees of freedom.  statsmodels' float
/// fit gives `params [-0.20166015625, 1000000000000.0002]`,
/// `pvalues[1] 1.4110599135479283e-37` (`2.4e-4` off); the exact fit
/// gives the true values.
#[test]
fn ols_slope_p_value_with_a_huge_t_and_three_degrees_of_freedom() -> Result<(), SymplexError> {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5]);
    let y = from_i64(&[
        999_999_999_999,
        2_000_000_000_001,
        2_999_999_999_999,
        4_000_000_000_001,
        4_999_999_999_999,
    ]);
    let fit = simple_linear_regression(&x, &y)?;
    // Fraction: params [-1/5, 1000000000000]; mpmath: t = 2500000000000.0, bse[1] = 0.4,
    //   betainc(3/2, 1/2, 0, 3/(3 + t²), regularized=True) = 1.4114019722797876467e-37,
    //   pvalues[0] = 0.88973463688659554043, f_pvalue = 1.4114019722797876467e-37
    assert_eq!(fit.coefficients, vec![q(-1, 5), q(1_000_000_000_000, 1)]);
    let p = fit.p_values(&ctx)?;
    close(
        p[1].eval_f64()?,
        1.411_401_972_279_787_646_7e-37,
        1e-13,
        "p1",
    );
    close(p[0].eval_f64()?, 0.889_734_636_886_595_540_43, 1e-13, "p0");
    close(
        fit.f_test(&ctx)?.p_value.eval_f64()?,
        1.411_401_972_279_787_646_7e-37,
        1e-13,
        "f_pvalue",
    );
    close(
        fit.p_values_log10(&ctx)?[1],
        (1.411_401_972_279_787_646_7e-37_f64).log10(),
        1e-13,
        "log10 p1",
    );
    Ok(())
}
