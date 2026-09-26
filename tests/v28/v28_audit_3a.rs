//! Audit round 3a of `symplex::stats`: `anova`, `agreement`,
//! `reliability`, `aggregation`.
//!
//! 1. The studentized range upper tail was `1 − P(Q ≤ q)`
//!    (`studentized_range_sf`, every `tukey_hsd` p-value): rounding noise
//!    (`4.1e-15` for a true `9.8e-49`) or `0` past `10⁻¹⁵`, and the old
//!    quadrature was only good to about `1e-9` absolute, so `p_adj` for a
//!    clear separation carried a relative error up to `1e-3`.  The
//!    quantile bracketed `cdf − p`, so `p → 1` levels were off by orders of
//!    magnitude and `p → 0` returned `0`.  Both tails are now integrated
//!    directly (the upper one from `1 − (1 − r)^{k−1}` as `expm1`/`ln1p`),
//!    and the quantile solves on the logarithm of the smaller tail.
//! 2. Mauchly's p-value could exceed `1` (Box's truncated series).
//! 3. Dawid–Skene / MACE with finite inputs whose sums overflow returned
//!    rows of zeros as "probabilities" (`converged: true`).
//! 4. Panics on counts or labels near `usize::MAX` (votes, Bradley–Terry,
//!    Fleiss' and Cohen's κ).
//! 5. `item_response_summary` dropped every item's α-if-deleted when one
//!    was undefined; `posterior_entropy` called a NaN or overflowing row
//!    certain.
//!
//! Reference values: scipy 1.18.1, statsmodels 0.15.0, pingouin 0.6.1 by
//! the call quoted; mpmath 1.3 where scipy is itself `1 − cdf` (scripts
//! `target/scratch/sr_mp.py` — the double integral with the range tail in
//! the non-cancelling form `min`, cross-checked by the subtraction form
//! `sub` at 40–110 digits — `k2_oracle.py`, `mauchly_oracle.py`,
//! `box_omega2.py`, `probe_before_values.py` of this session); exact
//! rationals from `fractions.Fraction` on the integer data.

// Reference values are quoted at the digits mpmath printed them.
#![allow(clippy::excessive_precision)]

use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::aggregation::{
    BradleyTerryOpts, DawidSkeneInit, DawidSkeneOpts, DawidSkenePriors, LabelTable, MaceOpts,
    bradley_terry, category_metrics, dawid_skene, dawid_skene_map, mace, majority_vote, plurality,
    posterior_entropy, weighted_vote, wins_matrix,
};
use symplex::stats::agreement::{
    RatingTable, Weights, fleiss_kappa, kappa_ci_from_confusion, kappa_from_confusion,
    kappa_test_from_confusion,
};
use symplex::stats::anova::{
    Observation, TwoWayData, anova_repeated_measures, studentized_range_cdf,
    studentized_range_quantile, studentized_range_sf, tukey_hsd,
};
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::Alternative;
use symplex::stats::reliability::item_response_summary;

type R = Result<(), SymplexError>;

/// `actual` within `rel` of `expected`, relatively.
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

/// The relative accuracy of the studentized range tails: the quadrature is
/// good to about `10⁻¹⁵` (checked against mpmath at 25–110 digits and
/// against the exact `k = 2` case), and `erfc` in the far tail to a few
/// `10⁻¹⁴` (`2 Φ̄(20/√2)` is `3.6e-14` off), so `1e-13`.
const SR_REL: f64 = 1e-13;

// ═══════════════════════════════════════════════════════════════════════════
// 1. The studentized range distribution and Tukey's HSD
// ═══════════════════════════════════════════════════════════════════════════

/// The upper tail was `1 − cdf`: `sf(40, 4, 100)` was `4.1e-15` (true
/// `9.8e-49`; scipy's `studentized_range.sf`, also `1 − cdf`, gives
/// `7.77e-16`), `sf(25, 5, 20)` `1.0925e-12` (true `1.0687e-12`),
/// `sf(12, 10, ∞)` `3.2e-15` (true `9.7e-16`).
#[test]
fn studentized_range_sf_far_tail_is_integrated_directly() -> R {
    // mpmath: sr_mp.py <dps> min q,k,df  (and `sub` where quoted)
    for (qv, k, df, want) in [
        // dps 25 (min); scipy sf 7.771561172376096e-16
        (40.0, 4, 100.0, 9.815_794_721_810_433_995_9e-49),
        // dps 25 (min); scipy 1.0720313525780512e-12
        (25.0, 5, 20.0, 1.068_673_678_259_347_002_3e-12),
        // dps 25 (min); scipy 3.312444318837038e-9
        (30.0, 3, 10.0, 3.312_445_333_427_213_296_6e-9),
        // dps 25 (min); scipy 2.139277643919968e-11
        (50.0, 3, 10.0, 2.139_430_935_679_972_675_2e-11),
        // dps 20 (min) = dps 40 (sub) to all 20 digits
        (10.0, 3, 10.0, 9.176_336_455_094_431_999_7e-5),
        // dps 25 (min); scipy 0.08809832202351708 (1.5e-12 off)
        (3.0, 3, 200.0, 0.088_098_322_023_384_821_472),
        // dps 20 (min); scipy 0.005741314344716364 (2.7e-10 off)
        (8.0, 1000, 1e4, 0.005_741_314_346_266_784_987_2),
    ] {
        close(
            studentized_range_sf(qv, k, df)?,
            want,
            SR_REL,
            &format!("sf({qv}, {k}, {df})"),
        );
        // The two tails are computed separately and still add up.
        let total = studentized_range_sf(qv, k, df)? + studentized_range_cdf(qv, k, df)?;
        assert!((total - 1.0).abs() < 1e-14, "cdf + sf = {total}");
    }
    // df ≥ 1e30 is the range of k normals: mpmath rbar_min(12, 10) at dps 30
    // = rbar_sub(12, 10) at dps 110 = 9.6838259462524857221e-16; scipy
    // studentized_range.sf(12, 10, inf) = 1.1102230246251565e-15.
    close(
        studentized_range_sf(12.0, 10, 1e35)?,
        9.683_825_946_252_485_722_1e-16,
        SR_REL,
        "sf(12, 10, ∞)",
    );
    Ok(())
}

/// `Q_{2,ν} = √2 |T_ν|`, so both tails of `k = 2` are Student-t tails.  The
/// upper tail was `1 − cdf` (`sf(1e4, 2, 5)` = `5.8e-15`, true `1.07e-18`;
/// `sf(1e6, 2, 2)` = `5.3e-15`, true `2.0e-12`); the lower tail was `1e-9`
/// off relatively at `q = 1e-8` and `0` at `q = 1e-300`.
#[test]
fn studentized_range_k2_is_the_t_distribution() -> R {
    // mpmath (k2_oracle.py, dps 50): P(Q > q) = betainc(ν/2, ½, 0, ν/(ν + q²/2), regularized)
    // (scipy 2·t.sf(q/√2, ν) agrees: 1.0736896281526943e-18, 0.000900315715946949,
    // 1.9999999999940005e-12, 5.167197745839186e-26)
    for (qv, df, want) in [
        (1e4, 5.0, 1.073_689_628_152_694_345_2e-18),
        (1e3, 1.0, 0.000_900_315_715_946_948_883_51),
        (1e6, 2.0, 1.999_999_999_994e-12),
        (50.0, 30.0, 5.167_197_745_839_160_180_6e-26),
    ] {
        close(
            studentized_range_sf(qv, 2, df)?,
            want,
            SR_REL,
            &format!("sf({qv}, 2, {df})"),
        );
    }
    // mpmath: P(Q ≤ q) = betainc(½, ν/2, 0, (q²/2)/(ν + q²/2), regularized)
    close(
        studentized_range_cdf(1e-8, 2, 5.0)?,
        5.368_449_291_145_283_795_8e-9,
        SR_REL,
        "cdf(1e-8, 2, 5)",
    );
    close(
        studentized_range_cdf(1e-300, 2, 5.0)?,
        5.368_449_291_145_283_871_7e-301,
        SR_REL,
        "cdf(1e-300, 2, 5)",
    );
    Ok(())
}

/// The quantile solved `cdf(q) = p` on a CDF good to `1e-9` absolute:
/// `ppf(1 − 2⁻⁴⁰, 2, 1)` was `66454.6` (true `9.9e11`), `ppf(1 − 2⁻⁴⁰, 2, 5)`
/// `653.07` (true `652.25`), `ppf(1e-300, 2, 5)` `0`.  It now solves on the
/// logarithm of the smaller tail, in `ln q`.  Tolerance: the tails' `1e-13`
/// divided by `|d ln tail / d ln q|` (`≥ 1` for these), with margin.
#[test]
fn studentized_range_quantile_uses_the_smaller_tail() -> R {
    // mpmath (k2_oracle.py): the root of ln P(Q ≷ q) = ln(level), k = 2 exactly;
    // scipy √2·t.isf(level/2, ν) quoted beside.
    let top = 1.0 - f64::EPSILON / 2.0; // 1 − 2⁻⁵³; 1 − top is exact
    for (p, df, want) in [
        // scipy 652.2459039909513
        (1.0 - 2f64.powi(-40), 5.0, 652.245_903_990_951_243_91),
        // scipy 989908258291.1917
        (1.0 - 2f64.powi(-40), 1.0, 989_908_258_291.191_543_49),
        // scipy 3954.5184417619244
        (top, 5.0, 3_954.518_441_761_924_647_3),
        // scipy 3.6353516951468037
        (0.95, 5.0, 3.635_351_695_146_803_656),
        (1e-300, 5.0, 1.862_735_299_836_769_028_1e-300),
    ] {
        close(
            studentized_range_quantile(p, 2, df)?,
            want,
            1e-12,
            &format!("ppf({p}, 2, {df})"),
        );
    }
    // k = 3: the round trip on the smaller tail (no closed form).
    let qv = studentized_range_quantile(1e-10, 3, 10.0)?;
    close(
        studentized_range_cdf(qv, 3, 10.0)?,
        1e-10,
        1e-12,
        "cdf(ppf(1e-10))",
    );
    let qv = studentized_range_quantile(top, 3, 10.0)?;
    close(
        studentized_range_sf(qv, 3, 10.0)?,
        1.0 - top,
        1e-12,
        "sf(ppf(1 − 2⁻⁵³))",
    );
    Ok(())
}

/// `sf(1e4, 3, 100)` was `0` (`1 − cdf`) for a true `2.69e-286`; a tail
/// below the smallest double is an exact `0` (no error), and an infinite
/// `q` a certain event.
#[test]
fn studentized_range_underflow_and_infinite_q() -> R {
    assert_eq!(studentized_range_cdf((-511f64).exp(), 3, 10.0)?, 0.0);
    assert_eq!(studentized_range_sf(f64::INFINITY, 3, 10.0)?, 0.0);
    assert_eq!(studentized_range_cdf(f64::INFINITY, 3, 10.0)?, 1.0);
    assert_eq!(studentized_range_sf(-1.0, 3, 10.0)?, 1.0);
    // Q > 1e4 needs S below about 1e-3, and P(S < s) ∝ s^ν: about 1e-3000 at
    // ν = 1000, an exact 0 in f64 (not noise); at ν = 100 still a double.
    assert_eq!(studentized_range_sf(1e4, 3, 1000.0)?, 0.0);
    // mpmath: sr_mp.py 25 min 10000,3,100 = 2.6880190660736353237e-286 (old: 1 − cdf = 0)
    close(
        studentized_range_sf(1e4, 3, 100.0)?,
        2.688_019_066_073_635_323_7e-286,
        SR_REL,
        "sf(1e4, 3, 100)",
    );
    Ok(())
}

/// Tukey's HSD p-values were `1 − cdf` of the old quadrature: for three
/// well-separated groups `1.0004271250e-6`, `4.97169e-12`, `7.146707e-10`
/// against the true `1.0004271193e-6`, `4.96636e-12`, `7.146650e-10` (a
/// relative `1e-3` at the smallest).
#[test]
fn tukey_hsd_far_separation_p_values() -> R {
    let ctx = Context::new();
    let g = [
        from_i64(&[1, 2, 3, 4, 5]),
        from_i64(&[11, 12, 13, 14, 15]),
        from_i64(&[30, 31, 33, 32, 34]),
    ];
    // Exact statistics q² = 200, 1682, 722 (Fraction), k = 3, ν = 12.
    // mpmath: sr_mp.py 25 min s200,3,12 s1682,3,12 s722,3,12
    // scipy: tukey_hsd(*g).pvalue = 1.0004271195906966e-06, 4.9666937229631e-12, 7.146653269174408e-10
    let want = [
        1.000_427_119_328_797_938_5e-6,
        4.966_355_603_641_435_572_2e-12,
        7.146_650_498_382_700_185e-10,
    ];
    let pairs = tukey_hsd(&ctx, &g, 0.95)?;
    for (p, w) in pairs.iter().zip(want) {
        close(p.p_adj, w, SR_REL, &format!("p_adj({}, {})", p.i, p.j));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Mauchly's test
// ═══════════════════════════════════════════════════════════════════════════

/// Box's second-order p-value `P₁ + ω₂(P₂ − P₁)` is a truncated series; with
/// `ω₂ = 17766/9409 > 1` (11 subjects, 11 conditions) it was `1.00626`.
/// Now capped at `1`.  The statistic is unchanged.
#[test]
fn mauchly_p_value_is_capped_at_one() -> R {
    let ctx = Context::new();
    let y: Vec<Vec<Q>> = [
        [6, 4, 6, 8, 14, 18, 20, 20, 17, 21, 22],
        [4, 9, 4, 11, 17, 19, 14, 19, 17, 24, 24],
        [1, 5, 4, 8, 11, 10, 19, 18, 21, 20, 22],
        [4, 6, 8, 15, 11, 18, 17, 23, 24, 27, 26],
        [9, 2, 13, 14, 16, 19, 21, 21, 24, 26, 27],
        [7, 2, 5, 9, 14, 11, 17, 22, 16, 27, 25],
        [0, 8, 13, 9, 14, 10, 14, 22, 24, 23, 20],
        [7, 3, 11, 12, 12, 10, 18, 14, 18, 25, 25],
        [5, 11, 9, 10, 13, 10, 13, 15, 23, 24, 22],
        [2, 9, 8, 8, 13, 12, 15, 15, 23, 26, 26],
        [8, 3, 6, 10, 10, 13, 12, 20, 22, 23, 29],
    ]
    .iter()
    .map(|r| from_i64(r))
    .collect();
    let r = anova_repeated_measures(&ctx, &y)?;
    let Some(m) = r.mauchly else {
        panic!("Mauchly's test should be defined for 11 subjects");
    };
    // pingouin: sphericity(...) → W 0.002947789286464691, chi2 37.67932526977302, dof 54,
    //   pval 1.0063206128960565 (> 1 too); mauchly_oracle.py (mpmath, exact W): chi2
    //   37.679325269773103009, Box p = 1.0062624873222218988.
    close(q_f64(&m.w), 0.002_947_789_286_464_691, 1e-12, "W");
    close(
        m.chi_squared_f64()?,
        37.679_325_269_773_103_009,
        1e-12,
        "chi2",
    );
    assert_eq!(m.df, 54);
    assert_eq!(m.p_value_f64()?, 1.0);
    assert_eq!(m.p_value_log10()?, 0.0);
    Ok(())
}

/// Not a bug here — pinned because the oracle disagrees: pingouin's
/// `sphericity` writes `3k` for the `3d` of Box's `ω₂` (`box_omega2.py`
/// derives `ω₂` from the Bernoulli-polynomial expansion of `E[W^h]` in
/// mpmath and matches the `3d` form), so for `k ≥ 4` its p-value differs.
#[test]
fn mauchly_box_omega2_for_four_conditions() -> R {
    let ctx = Context::new();
    let y: Vec<Vec<Q>> = [
        [16, 10, 6, 5],
        [17, 4, 11, 7],
        [11, 19, 14, 3],
        [13, 9, 12, 9],
        [23, 6, 11, 10],
        [9, 8, 15, 8],
    ]
    .iter()
    .map(|r| from_i64(r))
    .collect();
    let r = anova_repeated_measures(&ctx, &y)?;
    let Some(m) = r.mauchly else {
        panic!("Mauchly's test should be defined for 6 subjects × 4 conditions");
    };
    // pingouin: sphericity → W 0.14839903626268872, chi2 7.101443313588746, pval 0.2272469411437794
    // mauchly_oracle.py (mpmath, Box's ω₂ with 3d): p = 0.22690163545751724
    close(q_f64(&m.w), 0.148_399_036_262_688_72, 1e-12, "W");
    close(m.chi_squared_f64()?, 7.101_443_313_588_746, 1e-12, "chi2");
    close(m.p_value_f64()?, 0.226_901_635_457_517_24, 1e-12, "p");
    Ok(())
}

fn q_f64(x: &Q) -> f64 {
    use symplex::num_traits::ToPrimitive;
    x.to_f64().unwrap_or(f64::NAN)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. The EM models: sums that overflow
// ═══════════════════════════════════════════════════════════════════════════

fn small_table() -> LabelTable {
    LabelTable::complete(&[&[0, 1], &[1, 1], &[0, 0], &[1, 0]], 2).unwrap()
}

fn rows_are_distributions(rows: &[Vec<f64>], label: &str) {
    for r in rows {
        let s: f64 = r.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12 && r.iter().all(|v| (0.0..=1.0).contains(v)),
            "{label}: {r:?} is not a distribution"
        );
    }
}

/// Starting posteriors `[f64::MAX, f64::MAX]` were divided by their sum `∞`:
/// the row became `[0, 0]`, and `dawid_skene` returned priors `[½, 0]`, a
/// posterior row `[0, 0]`, log-likelihood `−∞` and `converged: true`.  The
/// documented normalisation is scale-free, so the run equals the one started
/// from `[1, 1]` (an identity, no oracle needed).
#[test]
fn dawid_skene_huge_starting_posteriors_are_normalised() -> R {
    let t = small_table();
    let start = |row: Vec<f64>| DawidSkeneOpts {
        init: DawidSkeneInit::Posteriors(vec![row, vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
        ..DawidSkeneOpts::default()
    };
    let huge = dawid_skene(&t, &start(vec![f64::MAX, f64::MAX]))?;
    let unit = dawid_skene(&t, &start(vec![1.0, 1.0]))?;
    assert_eq!(huge, unit);
    rows_are_distributions(&huge.posteriors, "posteriors");
    rows_are_distributions(std::slice::from_ref(&huge.priors), "priors");
    assert!(huge.log_likelihood.is_finite());
    let flat = DawidSkenePriors::symmetric(2, 1.0, 1.0, 1.0);
    let huge_map = dawid_skene_map(&t, &flat, &start(vec![f64::MAX, f64::MAX]))?;
    assert_eq!(huge_map, unit);
    Ok(())
}

/// A huge finite prior `α = β = f64::MAX` overflowed the M-step denominator
/// `I + Σ(α − 1)`: priors `[0, 0]`.  The posterior mode
/// `(Σ T + α − 1)/(I + Σ(α − 1))` is `½` to within `10⁻³⁰⁰` by symmetry.
#[test]
fn dawid_skene_map_huge_prior_is_the_prior_mode() -> R {
    let priors = DawidSkenePriors::symmetric(2, f64::MAX, f64::MAX, f64::MAX);
    let ds = dawid_skene_map(&small_table(), &priors, &DawidSkeneOpts::default())?;
    assert_eq!(ds.priors, vec![0.5, 0.5]);
    for rater in &ds.confusion {
        assert_eq!(rater, &vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    }
    rows_are_distributions(&ds.posteriors, "posteriors");
    Ok(())
}

/// A huge finite smoothing `δ = f64::MAX` overflowed `Σ_l (count + δ)` in the
/// Dawid–Skene M-step and `n_r + 2δ`, `Σ ρ + Kδ` in MACE's: every confusion
/// row, competence and spam distribution became `0`.  By the documented
/// formulas they are `(c + δ)/(n + 2δ) = ½` and uniform, to `10⁻³⁰⁰`.
#[test]
fn em_models_huge_smoothing_keep_distributions() -> R {
    let t = small_table();
    let ds = dawid_skene(
        &t,
        &DawidSkeneOpts {
            smoothing: f64::MAX,
            ..DawidSkeneOpts::default()
        },
    )?;
    for rater in &ds.confusion {
        rows_are_distributions(rater, "confusion");
    }
    let m = mace(
        &t,
        &MaceOpts {
            smoothing: f64::MAX,
            ..MaceOpts::default()
        },
    )?;
    assert_eq!(m.competence, vec![0.5, 0.5]);
    for row in &m.spam_distribution {
        assert_eq!(row, &vec![0.5, 0.5]);
    }
    rows_are_distributions(&m.posteriors, "mace posteriors");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Counts and labels near usize::MAX
// ═══════════════════════════════════════════════════════════════════════════

/// `wins[i][j] + wins[j][i]` and the row totals were `usize` sums: a
/// panic (debug) for `M = 2⁶⁴ − 1` wins.  Two players have the closed-form
/// maximum-likelihood strengths `w₀₁/(w₀₁ + w₁₀)`: `M/(M + ⌊M/3⌋) = ¾` to
/// `1e-19` (Fraction).
#[test]
fn bradley_terry_counts_beyond_a_usize_sum() -> R {
    let m = usize::MAX;
    let bt = bradley_terry(&[vec![0, m], vec![m / 3, 0]], &BradleyTerryOpts::default())?;
    // The MM iteration stops at a change below 1e-12.
    assert!((bt.strengths[0] - 0.75).abs() < 1e-11, "{:?}", bt.strengths);
    assert!((bt.strengths[1] - 0.25).abs() < 1e-11, "{:?}", bt.strengths);
    let even = bradley_terry(&[vec![0, m], vec![m, 0]], &BradleyTerryOpts::default())?;
    assert_eq!(even.strengths, vec![0.5, 0.5]);
    Ok(())
}

/// A label of `usize::MAX` overflowed `max + 1` (a panic) and one past
/// `isize::MAX` bytes of counts was a capacity-overflow panic.
/// `majority_vote` (infallible) now tallies such labels sparsely — exact
/// winner and ties, no count vector — and the fallible entry points refuse.
#[test]
fn votes_with_labels_too_large_for_a_count_vector() {
    let m = usize::MAX;
    let v = majority_vote(&[Some(m), Some(3), Some(m), None]);
    assert_eq!(
        (v.winner, v.tied.clone(), v.counts),
        (Some(m), vec![m], vec![])
    );
    let tie = majority_vote(&[Some(m), Some(3)]);
    assert_eq!((tie.winner, tie.tied), (None, vec![3, m]));
    let big = majority_vote(&[Some(1 << 61), Some(7)]);
    assert_eq!((big.winner, big.tied), (None, vec![7, 1 << 61]));
    is_invalid(weighted_vote(&[Some(m)], &[qi(1)]));
    is_invalid(plurality(&[Some(m)], &q(1, 2)));
    is_invalid(LabelTable::complete(&[&[0]], m));
    is_invalid(wins_matrix(&[], m));
    is_invalid(category_metrics(&[Some(0)], &[0], m));
    // Ordinary labels are unchanged.
    let v = majority_vote(&[Some(2), Some(0), Some(2)]);
    assert_eq!((v.winner, v.counts), (Some(2), vec![1, 0, 2]));
}

/// `TwoWayData::from_long` took `max level + 1` (a panic at `usize::MAX`)
/// and allocated the `a × b` table before finding its empty cells (a level
/// index of `2⁴⁰` asked for terabytes).  A design with more combinations
/// of levels than observations is refused first.
#[test]
fn two_way_long_form_with_huge_level_indices() {
    let obs = |a, b, y| Observation { a, b, y: qi(y) };
    is_invalid(TwoWayData::from_long(&[
        obs(usize::MAX, 0, 1),
        obs(0, 1, 2),
    ]));
    is_invalid(TwoWayData::from_long(&[
        obs(1 << 40, 0, 1),
        obs(0, 1, 2),
        obs(0, 0, 3),
    ]));
    // An ordinary long form is unchanged.
    let rows = [
        obs(0, 0, 4),
        obs(0, 1, 6),
        obs(1, 0, 5),
        obs(1, 1, 8),
        obs(1, 1, 9),
    ];
    let data = TwoWayData::from_long(&rows).unwrap();
    assert_eq!((data.a_levels(), data.b_levels(), data.n_obs()), (2, 2, 5));
}

/// Fleiss' κ summed and squared counts in `usize`: `n(n − 1)` with
/// `n = 2³³` raters per item panicked (debug).  Exact now.
#[test]
fn fleiss_kappa_counts_beyond_usize() -> R {
    let (a, b) = (1usize << 32, 1usize << 33);
    // Fraction (Fleiss 1971) = 8589934589/25769803773; statsmodels fleiss_kappa → 0.3333333332557231
    assert_eq!(
        fleiss_kappa(&[vec![a, a], vec![b, 0]])?,
        q(8_589_934_589, 25_769_803_773)
    );
    let (c, d) = (1usize << 62, 1usize << 63);
    // Fraction: 127605887595351923778012890703989833721/191408831393027885767323506956780437506;
    //   statsmodels fleiss_kappa → 0.6666666666666667
    let got = fleiss_kappa(&[vec![c, c, 1], vec![d, 0, 1], vec![1, d, 0]])?;
    let want: Q = "127605887595351923778012890703989833721/191408831393027885767323506956780437506"
        .parse()
        .unwrap();
    assert_eq!(got, want);
    Ok(())
}

/// Cohen's κ summed the confusion matrix in `usize`: `n` overflowed for a
/// cell of `2⁶⁴ − 1` (a panic).  Exact now — where statsmodels' float
/// arithmetic cancels to NaN.
#[test]
fn cohen_kappa_counts_beyond_usize() -> R {
    let ctx = Context::new();
    let m = usize::MAX;
    let t = [vec![m, 1], vec![1, 1]];
    // Fraction (Cohen 1960) = 9223372036854775807/18446744073709551616;
    // statsmodels cohens_kappa(t).kappa = nan
    let want: Q = "9223372036854775807/18446744073709551616".parse().unwrap();
    assert_eq!(kappa_from_confusion(&t, &Weights::Unweighted)?.kappa, want);
    assert_eq!(kappa_ci_from_confusion(&ctx, &t, 0.95)?.kappa, want);
    // Fraction = 105/181; statsmodels 0.580110497237569
    let t = [vec![m, m / 3], vec![m / 5, m]];
    assert_eq!(
        kappa_from_confusion(&t, &Weights::Unweighted)?.kappa,
        q(105, 181)
    );
    Ok(())
}

/// Confirmed correct, pinned: the one-sided κ test builds `1 − ½ erfc(…)`
/// only when κ is on the other side, where the p-value is at least `½` —
/// no cancellation.
#[test]
fn kappa_test_one_sided_against_the_tail() -> R {
    let ctx = Context::new();
    let r = kappa_test_from_confusion(&ctx, &[vec![2, 8], vec![9, 3]], Alternative::Greater)?;
    // statsmodels: cohens_kappa([[2, 8], [9, 3]], return_results=True) → kappa −0.5454545454545454,
    //   z_value −2.5690465157330253, pvalue_one_sided 0.9949010616118799
    close(r.statistic_f64()?, -2.569_046_515_733_025_3, 1e-14, "z");
    close(r.p_value_f64()?, 0.994_901_061_611_879_9, 1e-14, "p");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Item analysis and posterior entropy
// ═══════════════════════════════════════════════════════════════════════════

/// `item_response_summary` took α-if-deleted from `alpha_if_deleted(table)
/// .ok()`: one item whose removal leaves a constant total (items 1 and 2
/// sum to 4 here) blanked every item's value, although the field is
/// documented per item.
#[test]
fn item_response_summary_alpha_if_deleted_per_item() -> R {
    let ctx = Context::new();
    let t = RatingTable::from_i64(&[&[1, 1, 3], &[2, 2, 2], &[4, 3, 1], &[3, 0, 4], &[5, 2, 2]])?;
    // Fraction: without item 0 undefined (constant total), without item 1 −30/23, without item 2 30/53;
    // pingouin cronbach_alpha(data.drop(columns=j)) → −inf, −1.3043478260869565, 0.5660377358490567
    let s = item_response_summary(&ctx, &t)?;
    let got: Vec<Option<Q>> = s.into_iter().map(|i| i.alpha_if_deleted).collect();
    assert_eq!(got, vec![None, Some(q(-30, 23)), Some(q(30, 53))]);
    Ok(())
}

/// `posterior_entropy` returned `0` ("certain") for a row whose positive
/// mass overflowed (`[f64::MAX, f64::MAX]`, entropy `1` bit) and for rows
/// with a NaN or `+∞` entry.
#[test]
fn posterior_entropy_overflowing_and_nan_rows() {
    let h = posterior_entropy(&[
        vec![f64::MAX, f64::MAX],
        vec![f64::NAN, 0.5],
        vec![f64::INFINITY, 1.0],
        vec![0.25, 0.25, 0.5],
    ]);
    assert_eq!(h[0], 1.0);
    assert!(h[1].is_nan() && h[2].is_nan(), "{h:?}");
    assert_eq!(h[3], 1.5);
}
