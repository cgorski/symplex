//! symplex 0.17 — the `stats::PValue` trait and the inherent
//! `p_value_log10` / `p_value_ln` / `p_value_decimal` accessors: reading a
//! p-value that is below `f64::MIN_POSITIVE` without losing it.
//!
//! Oracles: `mpmath` 1.3 (`symplex/.venv/bin/python`, `mp.dps = 40`) for
//! the tiny values — the exact statistics were derived with
//! `fractions.Fraction` on the integer data and the tails evaluated with
//! `mpmath.gammainc(…, regularized=True)`, `mpmath.erfc` and
//! `mpmath.betainc(…, regularized=True)`; scipy 1.18 / statsmodels 0.15 /
//! pingouin 0.6 for the ordinary ones (the values already pinned in the
//! doc examples of the functions used).  The reference p-value for the χ²
//! example is
//! `mpmath.log10(mpmath.gammainc(0.5, 6400, regularized=True))`
//! `= -2781.6363830267828509`.
//!
//! The numbers in the "Tiny p-values" subsection of
//! `book/src/guide/statistics.md` are asserted in [`chi_square_book_example`].

use std::f64::consts::LN_10;

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::PValue;
use symplex::stats::agreement::RatingTable;
use symplex::stats::anova::{
    Mauchly, RepeatedMeasuresAnova, TwoWayData, anova_repeated_measures, anova_two_way,
};
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::{
    Alternative, AnovaResult, ChiSquareResult, TestResult, anova_one_way, binomial_test,
    chi_square_independence, t_test_one_sample, z_test_proportion,
};
use symplex::stats::regression::ols;
use symplex::stats::reliability::{cochrans_q, pearson_test};
use symplex::stats::survival::{Observation, log_rank_test};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

/// Split `"m.mmm…e-XXXX"` (the `eval_decimal` form) into its mantissa and
/// decimal exponent; the exponent is `0` when the string has none.
fn mantissa_exponent(s: &str) -> (f64, i32) {
    match s.split_once('e') {
        Some((m, e)) => (m.parse().unwrap(), e.parse().unwrap()),
        None => (s.parse().unwrap(), 0),
    }
}

/// `log10 p` read off the decimal string — the second, independent route
/// to a tiny p-value's magnitude.
fn log10_from_decimal(s: &str) -> f64 {
    let (m, e) = mantissa_exponent(s);
    m.log10() + f64::from(e)
}

/// The mantissa of a decimal string starts with `prefix` and the exponent
/// is `exponent` (the last digit of a 20-digit mantissa is rounding-
/// sensitive; the first fifteen are not).
fn assert_decimal(s: &str, prefix: &str, exponent: i32) {
    let (mantissa, e) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().unwrap()),
        None => (s, 0),
    };
    assert!(
        mantissa.starts_with(prefix),
        "decimal {s}: mantissa does not start with {prefix}"
    );
    assert_eq!(e, exponent, "decimal {s}: exponent");
}

fn is_invalid_argument(e: &SymplexError) -> bool {
    matches!(e, SymplexError::InvalidArgument { .. })
}

// ── Data ─────────────────────────────────────────────────────────────────

/// `[[9000, 1000], [1000, 9000]]`: χ² = 12800 on 1 df without correction.
fn extreme_table() -> Vec<Vec<Q>> {
    vec![from_i64(&[9000, 1000]), from_i64(&[1000, 9000])]
}

/// Three groups of 50, `y = 1000 i + (j mod 2)`: `F = 196 000 000` on
/// `(2, 147)`.  Python: `groups = [[1000*i + (j % 2) for j in range(50)] for i in range(3)]`.
fn extreme_groups() -> Vec<Vec<Q>> {
    (0..3)
        .map(|i| (0..50).map(|j| qi(1000 * i + (j % 2))).collect())
        .collect()
}

/// 30 subjects × 3 conditions, `y[i][j] = 10⁶ j + noise[i][j] + i` with
/// `noise = [i % 3, i² % 5, 7i % 4]`: `F = 1740000870000435/97` on `(2, 58)`.
fn extreme_repeated() -> Vec<Vec<Q>> {
    (0..30i64)
        .map(|i| {
            let noise = [i % 3, (i * i) % 5, (7 * i) % 4];
            (0..3i64)
                .map(|j| qi(1_000_000 * j + noise[j as usize] + i))
                .collect()
        })
        .collect()
}

/// `x = 0..80`, `y = 1000 x + (i mod 2)`: slope `2133001/2133`, intercept
/// `13/27`, `t² = 13649079798003/82` on 78 df for the slope.
fn extreme_regression() -> (Vec<Q>, Vec<Vec<Q>>) {
    let x: Vec<Vec<Q>> = (0..80i64).map(|i| vec![qi(i)]).collect();
    let y: Vec<Q> = (0..80i64).map(|i| qi(1000 * i + (i % 2))).collect();
    (y, x)
}

/// The RM3 dataset of `tests/v17/v17_anova.rs` (5 subjects × 3 conditions).
fn rm3() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[5, 7, 9]),
        from_i64(&[4, 5, 8]),
        from_i64(&[6, 8, 10]),
        from_i64(&[3, 6, 4]),
        from_i64(&[7, 9, 13]),
    ]
}

// ── ChiSquareResult: the reported case ───────────────────────────────────

/// Every number quoted in the book's "Tiny p-values" subsection.
#[test]
fn chi_square_book_example() {
    let ctx = Context::new();
    let r = chi_square_independence(&ctx, &extreme_table(), false).unwrap();
    assert_eq!(r.statistic, qi(12800));
    assert_eq!(r.df, 1);
    assert_eq!(r.p_value.to_string(), "uppergamma(1/2, 6400)/Gamma(1/2)");
    // f64 underflows …
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    // … the expression does not.  mpmath (dps 40):
    //   mpmath.gammainc(0.5, 6400, regularized=True)              = 2.3100265595063985852e-2782
    //   mpmath.log10(mpmath.gammainc(0.5, 6400, regularized=True)) = -2781.6363830267828509
    //   mpmath.log(mpmath.gammainc(0.5, 6400, regularized=True))   = -6404.9544696873456703
    close(r.p_value_log10().unwrap(), -2_781.636_383_026_783, 1e-9);
    close(r.p_value_ln().unwrap(), -6_404.954_469_687_346, 1e-9);
    assert_eq!(
        r.p_value_decimal(20).unwrap(),
        "2.3100265595063985852e-2782"
    );
    // Trailing zeros are dropped: 2.3100… → "2.31".
    assert_eq!(r.p_value_decimal(5).unwrap(), "2.31e-2782");
}

/// The trait methods and the inherent ones are the same functions.
#[test]
fn chi_square_inherent_equals_trait() {
    let ctx = Context::new();
    let r = chi_square_independence(&ctx, &extreme_table(), false).unwrap();
    assert_eq!(<ChiSquareResult as PValue>::p_value_ex(&r), &r.p_value);
    assert_eq!(
        r.p_value_log10().unwrap(),
        PValue::p_value_log10(&r).unwrap()
    );
    assert_eq!(r.p_value_ln().unwrap(), PValue::p_value_ln(&r).unwrap());
    assert_eq!(
        r.p_value_decimal(12).unwrap(),
        PValue::p_value_decimal(&r, 12).unwrap()
    );
    assert_eq!(r.p_value_f64().unwrap(), PValue::p_value_f64(&r).unwrap());
}

/// `log10` through the logarithm expression agrees with `log10` read off
/// the decimal string (mantissa and exponent) — the two independent routes.
#[test]
fn log10_agrees_with_decimal_exponent_path() {
    let ctx = Context::new();
    let chi = chi_square_independence(&ctx, &extreme_table(), false).unwrap();
    let via_ln = chi.p_value_log10().unwrap();
    let via_decimal = log10_from_decimal(&chi.p_value_decimal(20).unwrap());
    close(via_ln, via_decimal, 1e-9);
    // And `ln = log10 · ln 10` to the last bit or so.
    close(chi.p_value_ln().unwrap(), via_ln * LN_10, 1e-9);

    let z = z_test_proportion(&ctx, 9000, 10000, &q(1, 2), Alternative::TwoSided).unwrap();
    close(
        z.p_value_log10().unwrap(),
        log10_from_decimal(&z.p_value_decimal(20).unwrap()),
        1e-9,
    );

    // An ordinary p-value goes through the same code and has no exponent.
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let t = t_test_one_sample(&ctx, &x, &qi(5), Alternative::TwoSided).unwrap();
    let s = t.p_value_decimal(20).unwrap();
    assert!(!s.contains('e'), "{s}");
    close(t.p_value_log10().unwrap(), log10_from_decimal(&s), 1e-12);
}

// ── TestResult ───────────────────────────────────────────────────────────

/// A discrete exact test: `p = 2⁻²⁰⁰⁰` is an exact rational below `f64`.
#[test]
fn binomial_exact_rational_tiny_p() {
    let ctx = Context::new();
    let r = binomial_test(&ctx, 0, 2000, &q(1, 2), Alternative::Less).unwrap();
    assert!(r.p_value_exact().is_some(), "p should be an exact rational");
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    // mpmath: mpf(2)**-2000 = 8.7098098162172166756e-603
    //   log10 = -602.05999132796239043,  ln = -1386.2943611198906188 (= -2000 ln 2)
    close(r.p_value_log10().unwrap(), -602.059_991_327_962_4, 1e-9);
    close(r.p_value_ln().unwrap(), -1_386.294_361_119_890_6, 1e-9);
    close(
        r.p_value_ln().unwrap(),
        -2000.0 * std::f64::consts::LN_2,
        1e-9,
    );
    assert_decimal(&r.p_value_decimal(20).unwrap(), "8.70980981621721", -603);
}

/// A normal-tail test: `p = erfc(40√2)`, a transcendental expression.
#[test]
fn z_test_erfc_tiny_p() {
    let ctx = Context::new();
    let r = z_test_proportion(&ctx, 9000, 10000, &q(1, 2), Alternative::TwoSided).unwrap();
    close(r.statistic_f64().unwrap(), 80.0, 1e-12);
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    // mpmath: erfc(80/sqrt(2)) = 1.8048460032409472178e-1392
    //   log10 = -1391.7435598479388441,  ln = -3204.6079741763304482
    close(r.p_value_log10().unwrap(), -1_391.743_559_847_938_8, 1e-9);
    close(r.p_value_ln().unwrap(), -3_204.607_974_176_330_4, 1e-9);
    assert_decimal(&r.p_value_decimal(20).unwrap(), "1.80484600324094", -1392);
}

/// An ordinary p-value: the robust path must agree with `f64::log10` of
/// the `f64` p-value to well below the tolerance of the `f64` itself.
#[test]
fn ordinary_p_matches_f64_logarithms() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let r = t_test_one_sample(&ctx, &x, &qi(5), Alternative::TwoSided).unwrap();
    // scipy: ttest_1samp(range(1, 11), 5).pvalue = 0.6141172548083939
    let p = r.p_value_f64().unwrap();
    close(p, 0.614_117_254_808_393_9, 1e-12);
    close(r.p_value_log10().unwrap(), p.log10(), 1e-12);
    close(r.p_value_log10().unwrap(), -0.211_748_700_105_543_6, 1e-12);
    close(r.p_value_ln().unwrap(), p.ln(), 1e-12);
    assert_eq!(r.p_value_decimal(5).unwrap(), "0.61412");
    assert_eq!(r.p_value_decimal(20).unwrap(), "0.61411725480839390773");

    // A χ² result of ordinary size, likewise.
    // scipy: chi2_contingency([[10, 20], [30, 25]], correction=False)
    //   → statistic 3.505892255892255, pvalue 0.06115089757606777
    let table = vec![from_i64(&[10, 20]), from_i64(&[30, 25])];
    let c = chi_square_independence(&ctx, &table, false).unwrap();
    let pc = c.p_value_f64().unwrap();
    close(pc, 0.061_150_897_576_067_77, 1e-12);
    close(c.p_value_log10().unwrap(), pc.log10(), 1e-12);
    close(c.p_value_log10().unwrap(), -1.213_597_163_983_828_8, 1e-12);
    close(c.p_value_ln().unwrap(), pc.ln(), 1e-12);
}

/// An exact `0` (a perfectly correlated sample: `|r| = 1`, `t = ∞`) gives
/// `log10 p = ln p = −∞` rather than an evaluation error.
#[test]
fn exact_zero_p_value() {
    let ctx = Context::new();
    let x = from_i64(&[1, 2, 3, 4]);
    let y = from_i64(&[2, 4, 6, 8]);
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    assert_eq!(r.p_value_ex(), &ctx.zero());
    assert_eq!(r.p_value_exact(), Some(qi(0)));
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    assert_eq!(r.p_value_log10().unwrap(), f64::NEG_INFINITY);
    assert_eq!(r.p_value_ln().unwrap(), f64::NEG_INFINITY);
    assert_eq!(r.p_value_decimal(10).unwrap(), "0");

    // The same through a hand-built result (any exact-zero expression).
    let built = TestResult {
        statistic: ctx.int(1),
        p_value: ctx.zero(),
        df: None,
        alternative: Alternative::Greater,
    };
    assert_eq!(PValue::p_value_log10(&built).unwrap(), f64::NEG_INFINITY);
}

/// An exact `1` gives `log10 p = ln p = 0`.
#[test]
fn exact_one_p_value() {
    let ctx = Context::new();
    // r = 1 against `Less`: the observed correlation is as far from the
    // alternative as possible.
    let x = from_i64(&[1, 2, 3, 4]);
    let y = from_i64(&[2, 4, 6, 8]);
    let r = pearson_test(&ctx, &x, &y, Alternative::Less).unwrap();
    assert_eq!(r.p_value_ex(), &ctx.one());
    assert_eq!(r.p_value_log10().unwrap(), 0.0);
    assert_eq!(r.p_value_ln().unwrap(), 0.0);
    assert_eq!(r.p_value_decimal(10).unwrap(), "1");

    // scipy: binomtest(5, 10, 0.5).pvalue = 1.0 — a discrete exact test at
    // its null value.
    let b = binomial_test(&ctx, 5, 10, &q(1, 2), Alternative::TwoSided).unwrap();
    assert_eq!(b.p_value_exact(), Some(qi(1)));
    assert_eq!(b.p_value_f64().unwrap(), 1.0);
    assert_eq!(b.p_value_log10().unwrap(), 0.0);
    assert_eq!(b.p_value_ln().unwrap(), 0.0);
}

/// `TestResult`s built by other modules (survival, reliability) carry the
/// accessors too.
#[test]
fn test_results_from_other_modules() {
    let ctx = Context::new();
    // statsmodels survdiff: p 0.07957250154977413 → log10 -1.0992369886501683
    let obs = Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 12, 4, 9, 11, 13, 15, 16, 18, 20],
        &[
            true, false, true, true, false, true, true, false, true, true, false, true, true, true,
            false, true,
        ],
    );
    let groups = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    let lr = log_rank_test(&ctx, &obs, &groups).unwrap();
    close(lr.p_value_f64().unwrap(), 0.079_572_501_549_774_13, 1e-12);
    close(lr.p_value_log10().unwrap(), -1.099_236_988_650_168_3, 1e-12);
    close(
        lr.p_value_ln().unwrap(),
        lr.p_value_f64().unwrap().ln(),
        1e-12,
    );

    // statsmodels cochrans_q: pvalue 0.003095586852365 → log10 -2.5092570065521553
    let t = RatingTable::from_i64(&[
        &[1, 1, 0],
        &[1, 1, 0],
        &[1, 0, 0],
        &[1, 1, 1],
        &[0, 1, 0],
        &[1, 0, 0],
        &[1, 1, 0],
        &[1, 1, 0],
        &[0, 0, 0],
        &[1, 1, 1],
        &[1, 0, 0],
        &[1, 1, 0],
    ])
    .unwrap();
    let cq = cochrans_q(&ctx, &t).unwrap();
    close(cq.p_value_log10().unwrap(), -2.509_257_006_552_155, 1e-11);
    assert_eq!(cq.p_value_decimal(6).unwrap(), "0.00309559");
}

// ── AnovaResult ──────────────────────────────────────────────────────────

#[test]
fn anova_one_way_tiny_p() {
    let ctx = Context::new();
    let r = anova_one_way(&ctx, &extreme_groups()).unwrap();
    assert_eq!(r.f, qi(196_000_000));
    assert_eq!((r.df_between, r.df_within), (2, 147));
    assert_eq!(r.ss_between, qi(100_000_000));
    assert_eq!(r.ss_within, q(75, 2));
    // scipy: f_oneway(*groups).pvalue = 0.0 — underflow.
    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    // mpmath: betainc(147/2, 1, 0, 147/(147 + 2·196000000), regularized=True)
    //   = 4.9123149975417296828e-473;  log10 -472.30871379225207744,  ln -1087.5310036692308571
    close(r.p_value_log10().unwrap(), -472.308_713_792_252_1, 1e-9);
    close(r.p_value_ln().unwrap(), -1_087.531_003_669_230_9, 1e-9);
    assert_decimal(&r.p_value_decimal(20).unwrap(), "4.91231499754172", -473);
    assert_eq!(
        <AnovaResult as PValue>::p_value_log10(&r).unwrap(),
        r.p_value_log10().unwrap()
    );
}

// ── RepeatedMeasuresAnova, AnovaRow ──────────────────────────────────────

/// The uncorrected p-value underflows; the Greenhouse–Geisser and
/// Huynh–Feldt corrected ones (fewer effective degrees of freedom) do not,
/// so their `log10` can be checked against `f64::log10` as well as mpmath.
#[test]
fn repeated_measures_tiny_p() {
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &extreme_repeated()).unwrap();
    assert_eq!(r.f, q(1_740_000_870_000_435, 97));
    assert_eq!(r.conditions.ss, qi(60_000_030_000_015));
    assert_eq!(r.error.ss, qi(97));
    assert_eq!(r.epsilon_gg, q(9409, 11810));
    assert_eq!(r.epsilon_hf, Some(q(67615, 80918)));

    assert_eq!(r.p_value_f64().unwrap(), 0.0);
    // mpmath: F tail on (2, 58): p = 1.1219867683790263155e-342
    //   log10 -341.95001226469648772,  ln -787.36900078982122436
    close(r.p_value_log10().unwrap(), -341.950_012_264_696_5, 1e-9);
    close(r.p_value_ln().unwrap(), -787.369_000_789_821_2, 1e-9);
    assert_decimal(&r.p_value_decimal(20).unwrap(), "1.12198676837902", -342);
    // The conditions row carries the same p-value.
    assert_eq!(
        r.conditions.p_value_log10().unwrap(),
        r.p_value_log10().unwrap()
    );
    assert_eq!(r.conditions.p_value_ln().unwrap(), r.p_value_ln().unwrap());
    assert_eq!(
        r.conditions.p_value_decimal(15).unwrap(),
        r.p_value_decimal(15).unwrap()
    );

    // mpmath: F tail on (2ε̂, 58ε̂), ε̂ = 9409/11810: p_gg = 1.6711735780206128696e-273
    //   log10 -272.77697843935766648
    let gg = r.p_value_gg_f64().unwrap();
    assert!(gg > 0.0, "the GG-corrected p is representable: {gg}");
    close(r.p_value_gg_log10().unwrap(), -272.776_978_439_357_66, 1e-9);
    close(r.p_value_gg_log10().unwrap(), gg.log10(), 1e-9);
    // mpmath: with min(ε̃, 1) = 67615/80918: p_hf = 9.6906272405088761605e-287
    //   log10 -286.01364811167207674
    let hf = r.p_value_hf_log10().unwrap().unwrap();
    close(hf, -286.013_648_111_672_08, 1e-9);
    close(hf, r.p_value_hf_f64().unwrap().unwrap().log10(), 1e-9);
    assert_eq!(
        <RepeatedMeasuresAnova as PValue>::p_value_ex(&r),
        &r.p_value
    );
}

/// Mauchly's test and the ordinary-sized repeated-measures p-values of the
/// RM3 dataset (pingouin `rm_anova(correction=True)`, `sphericity`).
#[test]
fn mauchly_and_repeated_measures_ordinary_p() {
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    // pingouin: p-unc 0.003735511033474317 → log10 -2.427649976489498
    close(r.p_value_log10().unwrap(), -2.427_649_976_489_498, 1e-11);
    close(
        r.p_value_log10().unwrap(),
        r.p_value_f64().unwrap().log10(),
        1e-12,
    );
    // pingouin: p-GG-corr 0.021264365858261566 → log10 -1.6723475642015089
    close(
        r.p_value_gg_log10().unwrap(),
        -1.672_347_564_201_508_9,
        1e-11,
    );
    let hf = r.p_value_hf_log10().unwrap().unwrap();
    close(hf, r.p_value_hf_f64().unwrap().unwrap().log10(), 1e-12);

    // pingouin sphericity: W 1245/7921, χ² 5.551145791696415, p 0.0623137671632364
    //   → log10 -1.2054159927871055
    let m = r.mauchly.as_ref().unwrap();
    assert_eq!(m.w, q(1245, 7921));
    close(m.p_value_f64().unwrap(), 0.062_313_767_163_236_4, 1e-12);
    close(m.p_value_log10().unwrap(), -1.205_415_992_787_105_5, 1e-11);
    close(
        m.p_value_ln().unwrap(),
        m.p_value_f64().unwrap().ln(),
        1e-12,
    );
    assert_eq!(m.p_value_decimal(4).unwrap(), "0.06231");
    assert_eq!(<Mauchly as PValue>::p_value_ex(m), &m.p_value);
    assert_eq!(
        <Mauchly as PValue>::p_value_log10(m).unwrap(),
        m.p_value_log10().unwrap()
    );
}

/// `AnovaRow`: effect rows have the accessors; the residual and total rows
/// report `InvalidArgument`, as `p_value_f64` does.
#[test]
fn anova_row_accessors() {
    let ctx = Context::new();
    let data = TwoWayData::from_i64(&[
        &[&[4, 5, 6], &[6, 7, 8], &[9, 10, 12]],
        &[&[5, 5, 7], &[8, 9, 11], &[13, 14, 16]],
    ])
    .unwrap();
    let r = anova_two_way(&ctx, &data).unwrap();
    // statsmodels anova_lm(typ=2): C(B) PR(>F) 3.291990727040258e-06 → log10 -5.48254139678933
    close(
        r.factor_b.p_value_log10().unwrap(),
        -5.482_541_396_789_33,
        1e-10,
    );
    close(
        r.factor_b.p_value_ln().unwrap(),
        r.factor_b.p_value_f64().unwrap().ln(),
        1e-12,
    );
    assert_decimal(&r.factor_b.p_value_decimal(6).unwrap(), "3.29199", -6);

    for row in [&r.residual, &r.total] {
        assert!(row.p_value.is_none());
        assert!(is_invalid_argument(&row.p_value_f64().unwrap_err()));
        assert!(is_invalid_argument(&row.p_value_log10().unwrap_err()));
        assert!(is_invalid_argument(&row.p_value_ln().unwrap_err()));
        assert!(is_invalid_argument(&row.p_value_decimal(10).unwrap_err()));
    }
    let msg = r.residual.p_value_log10().unwrap_err().to_string();
    assert!(msg.contains("Residual row has no F test"), "{msg}");
}

// ── Ols ──────────────────────────────────────────────────────────────────

#[test]
fn ols_p_values_log10() {
    let ctx = Context::new();
    let (y, x) = extreme_regression();
    let fit = ols(&y, &x, true).unwrap();
    assert_eq!(fit.coefficients, vec![q(13, 27), q(2_133_001, 2133)]);
    assert_eq!(fit.ssr, q(42640, 2133));
    let ps = fit.p_values(&ctx).unwrap();
    // scipy: linregress(x, y).pvalue = 0.0 — the slope's p underflows.
    assert_eq!(ps[1].eval_f64().unwrap(), 0.0);
    let logs = fit.p_values_log10(&ctx).unwrap();
    assert_eq!(logs.len(), 2);
    // mpmath (t² = 40053/2173 and 13649079798003/82 on 78 df):
    //   intercept p = 0.000050152632833810660566 → log10 -4.2997062631384315
    //   slope     p = 1.3058672441188327659e-365 → log10 -364.88410097166521304
    close(logs[0], -4.299_706_263_138_431_5, 1e-11);
    close(logs[0], ps[0].eval_f64().unwrap().log10(), 1e-11);
    close(logs[1], -364.884_100_971_665_2, 1e-9);
    // Each coefficient test is a `TestResult` with the same p-value.
    let tests = fit.coefficient_tests(&ctx).unwrap();
    close(tests[1].p_value_log10().unwrap(), logs[1], 1e-12);
    assert_decimal(
        &tests[1].p_value_decimal(20).unwrap(),
        "1.30586724411883",
        -365,
    );

    // The module's ordinary example: statsmodels pvalues[1] 0.00049290666057244
    //   → log10 -3.3072353132506056
    let x: Vec<Vec<Q>> = from_i64(&[1, 2, 3, 4, 5, 6, 7])
        .into_iter()
        .map(|v| vec![v])
        .collect();
    let y = from_i64(&[2, 3, 5, 4, 6, 8, 9]);
    let fit = ols(&y, &x, true).unwrap();
    let logs = fit.p_values_log10(&ctx).unwrap();
    close(logs[1], -3.307_235_313_250_605_6, 1e-11);
    // The overall F test is a `TestResult` too.
    let f = fit.f_test(&ctx).unwrap();
    close(f.p_value_log10().unwrap(), logs[1], 1e-11); // one regressor: F = t²
}

// ── Generic use ──────────────────────────────────────────────────────────

/// `−log10 p` of any result, generically.
fn neg_log10<T: PValue + ?Sized>(r: &T) -> f64 {
    -r.p_value_log10().unwrap()
}

/// The trait is implemented for every result type with an exact p-value
/// and is object-safe.
#[test]
fn trait_is_generic_and_object_safe() {
    let ctx = Context::new();
    let chi = chi_square_independence(&ctx, &extreme_table(), false).unwrap();
    let t = binomial_test(&ctx, 0, 2000, &q(1, 2), Alternative::Less).unwrap();
    let a = anova_one_way(&ctx, &extreme_groups()).unwrap();
    let rm = anova_repeated_measures(&ctx, &rm3()).unwrap();
    let m = rm.mauchly.clone().unwrap();

    close(neg_log10(&chi), 2_781.636_383_026_783, 1e-9);
    close(neg_log10(&t), 602.059_991_327_962_4, 1e-9);
    close(neg_log10(&a), 472.308_713_792_252_1, 1e-9);
    close(neg_log10(&rm), 2.427_649_976_489_498, 1e-11);
    close(neg_log10(&m), 1.205_415_992_787_105_5, 1e-11);

    let all: Vec<&dyn PValue> = vec![&chi, &t, &a, &rm, &m];
    let order: Vec<f64> = all.iter().map(|r| neg_log10(*r)).collect();
    assert!(order.iter().all(|v| v.is_finite() && *v > 0.0));
    // Sorting by evidence works across underflowing and ordinary values.
    let mut sorted = order.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(
        sorted,
        vec![order[4], order[3], order[2], order[1], order[0]]
    );
}

/// The contract in one place: below `f64::MIN_POSITIVE`, `p_value_f64` is
/// `0.0` while `p_value_log10` is finite and below `log10(MIN_POSITIVE)`.
#[test]
fn underflow_contract() {
    let ctx = Context::new();
    let floor = f64::MIN_POSITIVE.log10(); // ≈ -307.65
    let tiny: Vec<Box<dyn PValue>> = vec![
        Box::new(chi_square_independence(&ctx, &extreme_table(), false).unwrap()),
        Box::new(binomial_test(&ctx, 0, 2000, &q(1, 2), Alternative::Less).unwrap()),
        Box::new(z_test_proportion(&ctx, 9000, 10000, &q(1, 2), Alternative::TwoSided).unwrap()),
        Box::new(anova_one_way(&ctx, &extreme_groups()).unwrap()),
        Box::new(anova_repeated_measures(&ctx, &extreme_repeated()).unwrap()),
    ];
    for r in &tiny {
        assert_eq!(r.p_value_f64().unwrap(), 0.0);
        let l = r.p_value_log10().unwrap();
        assert!(l.is_finite() && l < floor, "log10 p = {l}");
        let s = r.p_value_decimal(3).unwrap();
        assert!(s.contains("e-"), "{s}");
    }
}
