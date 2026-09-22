//! 0.18.1 track: the "known limits" left by the three verification passes
//! over `symplex::stats` (CHANGELOG 0.17.1 §"Known limits found").
//!
//! 1. `fisher_exact` on large tables: exact `BigInt` weights up to a
//!    support of 2 000 points, a numeric log-space route above it.
//! 2. `usize` overflow in rank-test tie terms, `kappa_from_confusion` and
//!    the `n₁n₂`-type products — now `Q`.
//! 3. `Sprt` decisions are sticky (`is_decided`, `stopped_at`).
//! 4. `KaplanMeier::censoring_times` / `censored_at` see censorings at
//!    times without an event.
//! 5. `quantile_f64` on a lattice family without a closed CDF walks the
//!    pmf instead of running Brent on a symbolic `Sum`.
//! 6. `KaplanMeier::quantile_strict` (statsmodels' `<`).
//! 7. `weighted_vote` with zero weights: documented.
//! 8. `Observation::try_from_i64` / `try_from_q` report a length mismatch.
//!
//! Oracles: `scipy 1.18.1` (`symplex/.venv/bin/python`), `mpmath` at 40
//! digits for the tails scipy cannot represent, Python `fractions.Fraction`
//! / `math.comb` for exact rationals, and hand counts for the life tables.

use std::time::Instant;

use num_bigint::BigInt;
use num_traits::{One, Zero};
use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::aggregation::{majority_vote, weighted_vote};
use symplex::stats::agreement::{Weights, kappa_from_confusion};
use symplex::stats::data::from_i64;
use symplex::stats::hypothesis::{
    Alternative, FISHER_EXACT_NUMERIC_THRESHOLD, RankMethod, TestResult, fisher_exact, friedman,
    kruskal_wallis, mann_whitney_u, rank_biserial, tie_term, wilcoxon_signed_rank,
};
use symplex::stats::sequential::{Decision, Sprt};
use symplex::stats::survival::{KaplanMeier, Observation};

type R = Result<(), SymplexError>;

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual} (|Δ| = {:e})",
        (actual - expected).abs()
    );
}

fn rel_close(actual: f64, expected: f64, rel: f64) {
    assert!(
        (actual - expected).abs() <= rel * expected.abs(),
        "expected {expected}, got {actual} (relative Δ = {:e})",
        (actual - expected).abs() / expected.abs()
    );
}

fn big(s: &str) -> BigInt {
    s.parse().unwrap_or_else(|_| panic!("not an integer: {s}"))
}

fn is_invalid<T: std::fmt::Debug>(r: Result<T, SymplexError>) {
    match r {
        Err(SymplexError::InvalidArgument { .. }) => {}
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

fn fisher(table: [[usize; 2]; 2], alt: Alternative) -> TestResult {
    let ctx = Context::new();
    fisher_exact(&ctx, table, alt).unwrap_or_else(|e| panic!("fisher_exact({table:?}): {e}"))
}

/// The numeric route's p-value is `ctx.from_f64(p)`: a dyadic rational.
fn is_dyadic(r: &TestResult) -> bool {
    match r.p_value_exact() {
        Some(p) => {
            let d = p.denom();
            d.is_one() || (d & (d - BigInt::one())).is_zero()
        }
        None => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. fisher_exact on large tables
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fisher_million_cell_table_matches_scipy_to_1e9() {
    // scipy.stats.fisher_exact([[10**6, 10**6 + 7], [10**6 - 3, 10**6]], alternative=alt):
    //   two-sided 0.9992021157304142, less 0.49960106287695355, greater 0.501196819571527
    // mpmath (40 digits, pmf walked from the mode, normalised by its total):
    //   two-sided 0.99920211598773724586, less 0.49960106298063621942, greater 0.50119681943588244659
    //   pmf(a) = 0.00079788241651866600629 — also math.comb exactly (64 s) — so scipy's Boost CDF
    //   is the one 2.6e-10 off here; both bounds are asserted.
    let table = [[1_000_000, 1_000_007], [999_997, 1_000_000]];
    let t0 = Instant::now();
    let two = fisher(table, Alternative::TwoSided);
    let less = fisher(table, Alternative::Less);
    let greater = fisher(table, Alternative::Greater);
    let elapsed = t0.elapsed();
    assert!(elapsed.as_secs_f64() < 1.0, "took {elapsed:?}");
    let (p2, pl, pg) = (
        two.p_value_f64().unwrap(),
        less.p_value_f64().unwrap(),
        greater.p_value_f64().unwrap(),
    );
    rel_close(p2, 0.999_202_115_730_414_2, 1e-9);
    rel_close(pl, 0.499_601_062_876_953_55, 1e-9);
    rel_close(pg, 0.501_196_819_571_527, 1e-9);
    rel_close(p2, 0.999_202_115_987_737_2, 1e-13);
    rel_close(pl, 0.499_601_062_980_636_2, 1e-13);
    rel_close(pg, 0.501_196_819_435_882_5, 1e-13);
    // P(X ≤ a) + P(X ≥ a) − 1 = pmf(a).
    rel_close(pl + pg - 1.0, 0.000_797_882_416_518_666, 1e-9);
    assert!(is_dyadic(&two) && is_dyadic(&less) && is_dyadic(&greater));
    // The odds ratio stays exact: 10¹² / ((10⁶ + 7)(10⁶ − 3)).
    let stat = two.statistic_exact().unwrap();
    assert_eq!(stat, Q::new(big("1000000000000"), big("1000003999979")));
}

#[test]
fn fisher_large_symmetric_table_underflows_like_scipy_but_keeps_the_logarithm() {
    // scipy.stats.fisher_exact([[5000, 3000], [2000, 6000]]): two-sided 0.0, less 1.0, greater 0.0
    // mpmath (40 digits, mp.binomial): log10 P(X ≥ 5000) = -511.52156068351288907,
    //   log10 two-sided = -511.22053068784890788 (n₁ = n₂: the mirror point is 2000, p = 2·P(X ≥ 5000))
    let table = [[5000, 3000], [2000, 6000]];
    let t0 = Instant::now();
    let two = fisher(table, Alternative::TwoSided);
    let less = fisher(table, Alternative::Less);
    let greater = fisher(table, Alternative::Greater);
    assert!(t0.elapsed().as_secs_f64() < 1.0);
    assert_eq!(two.p_value_f64().unwrap(), 0.0);
    assert_eq!(less.p_value_f64().unwrap(), 1.0);
    assert_eq!(greater.p_value_f64().unwrap(), 0.0);
    close(two.p_value_log10().unwrap(), -511.220_530_687_848_9, 1e-9);
    close(
        greater.p_value_log10().unwrap(),
        -511.521_560_683_512_9,
        1e-9,
    );
    // Below 1e-308 the expression is `exp(ln p)`, not a rational — but it still prints.
    assert!(two.p_value_exact().is_none());
    assert!(
        two.p_value_decimal(6).unwrap().ends_with("e-512"),
        "{}",
        two.p_value_decimal(6).unwrap()
    );
    assert_eq!(less.p_value_exact(), Some(Q::one()));
}

#[test]
fn fisher_one_sided_large_tables_match_scipy() {
    // scipy.stats.fisher_exact([[1200, 900], [1000, 1150]], alternative=...):
    //   less 0.9999999999984381, greater 2.4158828010763813e-12
    // scipy.stats.fisher_exact([[3000, 2900], [2800, 3100]], alternative=...):
    //   less 0.999892866963136, greater 0.00012379717886806704
    // mpmath: 0.99999999999843778998 / 2.4158828010763862201e-12 / 0.99989286696313542882 / 0.00012379717886806690641
    let a = [[1200, 900], [1000, 1150]];
    let b = [[3000, 2900], [2800, 3100]];
    let t0 = Instant::now();
    let al = fisher(a, Alternative::Less).p_value_f64().unwrap();
    let ag = fisher(a, Alternative::Greater).p_value_f64().unwrap();
    let bl = fisher(b, Alternative::Less).p_value_f64().unwrap();
    let bg = fisher(b, Alternative::Greater).p_value_f64().unwrap();
    assert!(t0.elapsed().as_secs_f64() < 1.0);
    rel_close(al, 0.999_999_999_998_438_1, 1e-9);
    rel_close(ag, 2.415_882_801_076_381_3e-12, 1e-9);
    rel_close(bl, 0.999_892_866_963_136, 1e-9);
    rel_close(bg, 0.000_123_797_178_868_067_04, 1e-9);
    rel_close(al, 0.999_999_999_998_437_8, 1e-13);
    rel_close(ag, 2.415_882_801_076_386e-12, 1e-13);
    rel_close(bl, 0.999_892_866_963_135_4, 1e-13);
    rel_close(bg, 0.000_123_797_178_868_066_9, 1e-13);
}

#[test]
fn fisher_two_sided_cutoff_on_the_far_side_matches_scipy() {
    // Asymmetric margins (the cutoff is searched): scipy 4.717954425131835e-12, mpmath 4.7179544251318393105e-12
    let asym = fisher([[1200, 900], [1000, 1150]], Alternative::TwoSided)
        .p_value_f64()
        .unwrap();
    rel_close(asym, 4.717_954_425_131_835e-12, 1e-9);
    rel_close(asym, 4.717_954_425_131_839_3e-12, 1e-13);
    // n₁ = n₂ (mirrored exactly): scipy 0.0002475943577361341 = 2 × greater; mpmath 0.00024759435773613381283
    let sym = fisher([[3000, 2900], [2800, 3100]], Alternative::TwoSided)
        .p_value_f64()
        .unwrap();
    rel_close(sym, 0.000_247_594_357_736_134_1, 1e-9);
    rel_close(sym, 0.000_247_594_357_736_133_8, 1e-13);
}

#[test]
fn fisher_numeric_route_agrees_with_exact_arithmetic_just_above_the_threshold() {
    // 2 001 support points → numeric.  Python Fraction / math.comb:
    //   [[1050, 950], [950, 1050]]: less 0.99929966542030733828, greater 0.00087038091170614920282,
    //                              two-sided 0.0017407618234122984056
    //   [[1000, 1500], [1000, 1500]]: less = greater = 0.51151464835212214196, two-sided 1 (a is the mode)
    let t = [[1050, 950], [950, 1050]];
    let r = fisher(t, Alternative::Less);
    assert!(is_dyadic(&r), "numeric route gives a dyadic p");
    rel_close(r.p_value_f64().unwrap(), 0.999_299_665_420_307_3, 1e-13);
    rel_close(
        fisher(t, Alternative::Greater).p_value_f64().unwrap(),
        0.000_870_380_911_706_149_2,
        1e-13,
    );
    rel_close(
        fisher(t, Alternative::TwoSided).p_value_f64().unwrap(),
        0.001_740_761_823_412_298_4,
        1e-13,
    );
    let u = [[1000, 1500], [1000, 1500]];
    rel_close(
        fisher(u, Alternative::Less).p_value_f64().unwrap(),
        0.511_514_648_352_122_1,
        1e-13,
    );
    rel_close(
        fisher(u, Alternative::Greater).p_value_f64().unwrap(),
        0.511_514_648_352_122_1,
        1e-13,
    );
    assert_eq!(
        fisher(u, Alternative::TwoSided).p_value_exact(),
        Some(Q::one())
    );
}

#[test]
fn fisher_exact_path_is_still_exact_up_to_the_threshold_and_fast() {
    assert_eq!(FISHER_EXACT_NUMERIC_THRESHOLD, 2_000);
    // [[1000, 999], [999, 1000]]: n = n₁ = n₂ = 1999, support 0..=1999 = 2 000 points → exact rationals.
    // P(X ≥ 1000) = 1/2 exactly (symmetric about 999.5); two-sided 1; less (Fraction) 0.52522028814176322083.
    let t = [[1000, 999], [999, 1000]];
    let t0 = Instant::now();
    let greater = fisher(t, Alternative::Greater);
    let two = fisher(t, Alternative::TwoSided);
    let less = fisher(t, Alternative::Less);
    assert!(
        t0.elapsed().as_secs_f64() < 1.0,
        "exact path at 2 000 points"
    );
    assert_eq!(greater.p_value_exact(), Some(q(1, 2)));
    assert_eq!(two.p_value_exact(), Some(Q::one()));
    let pl = less.p_value_exact().unwrap();
    assert!(pl.denom().bits() > 1000, "a genuine rational, not a dyadic");
    rel_close(less.p_value_f64().unwrap(), 0.525_220_288_141_763_2, 1e-14);
    // [[700, 1300], [900, 1100]] (1 601 points): Fraction less 6.4218066873860735919e-11,
    // greater 0.99999999995813932276, two-sided 1.2843613374772147184e-10.
    let s = [[700, 1300], [900, 1100]];
    let sl = fisher(s, Alternative::Less);
    assert!(sl.p_value_exact().unwrap().denom().bits() > 1000);
    rel_close(sl.p_value_f64().unwrap(), 6.421_806_687_386_074e-11, 1e-14);
    rel_close(
        fisher(s, Alternative::Greater).p_value_f64().unwrap(),
        0.999_999_999_958_139_4,
        1e-14,
    );
    rel_close(
        fisher(s, Alternative::TwoSided).p_value_f64().unwrap(),
        1.284_361_337_477_214_7e-10,
        1e-14,
    );
    // The doc example is untouched: scipy fisher_exact([[8, 2], [1, 5]]) → 20, 5/143.
    let d = fisher([[8, 2], [1, 5]], Alternative::TwoSided);
    assert_eq!(d.statistic_exact(), Some(qi(20)));
    assert_eq!(d.p_value_exact(), Some(q(5, 143)));
}

#[test]
fn fisher_far_tail_of_a_million_cell_table_keeps_a_finite_logarithm() {
    // [[2, 10⁶], [10⁶, 3]]: a = 2 lies 5·10⁵ lattice steps below the mode.
    // mpmath (30 digits): log10 P(X ≤ 2) = -602029.32707963603468.
    let t = [[2, 1_000_000], [1_000_000, 3]];
    let t0 = Instant::now();
    let less = fisher(t, Alternative::Less);
    let greater = fisher(t, Alternative::Greater);
    let two = fisher(t, Alternative::TwoSided);
    assert!(t0.elapsed().as_secs_f64() < 1.0);
    assert_eq!(less.p_value_f64().unwrap(), 0.0);
    assert_eq!(greater.p_value_exact(), Some(Q::one()));
    close(less.p_value_log10().unwrap(), -602_029.327_079_636, 1e-6);
    assert!(less.p_value_decimal(8).unwrap().ends_with("e-602030"));
    // The far-side cutoff adds a comparable mass: ln p₂ ≥ ln p_less, and p₂ ≤ 2·p_less-ish.
    let (l2, ll) = (two.p_value_log10().unwrap(), less.p_value_log10().unwrap());
    assert!(l2 >= ll && l2 - ll < 1.0, "{l2} vs {ll}");
}

#[test]
fn fisher_billion_cell_table_runs_in_milliseconds() {
    // Beyond every oracle (p ≈ 10^(-6·10⁸)); the point is that it returns at all, with the
    // tails decided (scipy gives 1.0 / 0.0 / 0.0 as well).
    let t = [[1_000_000_000, 1], [1, 1_000_000_000]];
    let t0 = Instant::now();
    let less = fisher(t, Alternative::Less);
    let greater = fisher(t, Alternative::Greater);
    let two = fisher(t, Alternative::TwoSided);
    assert!(t0.elapsed().as_secs_f64() < 1.0);
    assert_eq!(less.p_value_exact(), Some(Q::one()));
    assert_eq!(greater.p_value_f64().unwrap(), 0.0);
    assert_eq!(two.p_value_f64().unwrap(), 0.0);
    let lg = greater.p_value_log10().unwrap();
    assert!(lg.is_finite() && lg < -1e8, "log10 p = {lg}");
    assert!(two.p_value_log10().unwrap() >= lg);
}

#[test]
fn fisher_still_rejects_an_empty_margin_and_reports_an_overflowing_one() {
    let ctx = Context::new();
    is_invalid(fisher_exact(&ctx, [[0, 0], [5, 7]], Alternative::TwoSided));
    is_invalid(fisher_exact(
        &ctx,
        [[usize::MAX, 1], [1, 1]],
        Alternative::TwoSided,
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. usize products → Q
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tie_term_is_exact_beyond_usize() {
    // Python: 3_000_000**3 - 3_000_000 = 26999999999997000000 > 2**64 = 18446744073709551616
    assert_eq!(
        tie_term(&[3_000_000]),
        Q::from_integer(big("26999999999997000000"))
    );
    // 2_600_000³ − 2_600_000 = 17575999999997400000 (the old "safe below ~2.6·10⁶" edge)
    assert_eq!(
        tie_term(&[2_600_000, 2]),
        Q::from_integer(big("17575999999997400006"))
    );
    assert_eq!(tie_term(&[2, 2, 3]), qi(36));
    assert_eq!(tie_term(&[]), Q::zero());
    assert_eq!(tie_term(&[1, 0]), Q::zero());
}

#[test]
fn rank_tests_with_ties_keep_their_scipy_values() -> R {
    let ctx = Context::new();
    // scipy.stats.kruskal([1,1,2,2,3],[2,3,3,4,4],[5,5,6,6,7]) → 11.424817518248176 (= 7826/685 by Fraction),
    //   pvalue 0.0033047027172332945
    let g = [
        from_i64(&[1, 1, 2, 2, 3]),
        from_i64(&[2, 3, 3, 4, 4]),
        from_i64(&[5, 5, 6, 6, 7]),
    ];
    let r = kruskal_wallis(&ctx, &g)?;
    assert_eq!(r.statistic_exact(), Some(q(7826, 685)));
    close(r.p_value_f64()?, 0.003_304_702_717_233_294_5, 1e-12);
    // scipy.stats.friedmanchisquare over the columns of blocks [[1,2,2],[3,1,1],[2,2,3],[1,3,2],[2,1,3]]
    //   → 1.5294117647058874 (= 26/17), pvalue 0.4654708140240603
    let blocks = [
        from_i64(&[1, 2, 2]),
        from_i64(&[3, 1, 1]),
        from_i64(&[2, 2, 3]),
        from_i64(&[1, 3, 2]),
        from_i64(&[2, 1, 3]),
    ];
    let r = friedman(&ctx, &blocks)?;
    assert_eq!(r.statistic_exact(), Some(q(26, 17)));
    close(r.p_value_f64()?, 0.465_470_814_024_060_3, 1e-12);
    // scipy.stats.wilcoxon(x, y, method='asymptotic', correction=True) → (8.0, 0.17252964335868337);
    //   correction=False → 0.15102615479062909   (ties in |d|)
    let x = from_i64(&[3, 5, 2, 8, 7, 6, 4, 9, 10, 12]);
    let y = from_i64(&[1, 4, 4, 6, 9, 3, 4, 7, 10, 8]);
    let r = wilcoxon_signed_rank(
        &ctx,
        &x,
        Some(&y),
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )?;
    assert_eq!(r.statistic_exact(), Some(qi(8)));
    close(r.p_value_f64()?, 0.172_529_643_358_683_37, 1e-12);
    let r = wilcoxon_signed_rank(
        &ctx,
        &x,
        Some(&y),
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: false },
    )?;
    close(r.p_value_f64()?, 0.151_026_154_790_629_09, 1e-12);
    // scipy.stats.mannwhitneyu([1,2,2,3,5,5,6],[2,3,4,4,6,7], method='asymptotic') → (15.0, 0.4269081980442815)
    let a = from_i64(&[1, 2, 2, 3, 5, 5, 6]);
    let b = from_i64(&[2, 3, 4, 4, 6, 7]);
    let r = mann_whitney_u(
        &ctx,
        &a,
        &b,
        Alternative::TwoSided,
        RankMethod::Asymptotic { continuity: true },
    )?;
    assert_eq!(r.statistic_exact(), Some(qi(15)));
    close(r.p_value_f64()?, 0.426_908_198_044_281_5, 1e-12);
    Ok(())
}

#[test]
fn kappa_from_confusion_with_billion_counts_does_not_overflow() -> R {
    // Fraction: table [[3e9, 1e9], [5e8, 2.5e9]], n = 7e9: p_o = 11/14, p_e = 1/2, κ = 4/7
    // (row · column products reach 3.6·10¹⁹ > usize::MAX).
    let table = vec![
        vec![3_000_000_000, 1_000_000_000],
        vec![500_000_000, 2_500_000_000],
    ];
    let k = kappa_from_confusion(&table, &Weights::Unweighted)?;
    assert_eq!(k.observed, q(11, 14));
    assert_eq!(k.expected, q(1, 2));
    assert_eq!(k.kappa, q(4, 7));
    Ok(())
}

#[test]
fn rank_biserial_with_five_billion_per_group_does_not_overflow() -> R {
    // Fraction: 2·(1.5·10¹⁹)/(5·10⁹ · 5·10⁹) − 1 = 1/5   (n₁n₂ = 2.5·10¹⁹ > usize::MAX)
    let u1 = Q::from_integer(big("15000000000000000000"));
    assert_eq!(rank_biserial(&u1, 5_000_000_000, 5_000_000_000)?, q(1, 5));
    // U above n₁n₂ is still refused.
    let too_big = Q::from_integer(big("25000000000000000001"));
    is_invalid(rank_biserial(&too_big, 5_000_000_000, 5_000_000_000));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Sticky SPRT decisions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sprt_accept_h1_is_sticky_through_contrary_evidence() -> R {
    // Wald's rule: 12 successes reach B = ln(0.9/0.05) (12·ln(9/7) = 3.0158 > 2.8904).
    let mut test = Sprt::bernoulli(&q(7, 10), &q(9, 10), 0.05, 0.10)?;
    let mut d = Decision::Continue;
    while d == Decision::Continue {
        assert!(!test.is_decided());
        assert_eq!(test.stopped_at(), None);
        d = test.update(true);
    }
    assert_eq!(d, Decision::AcceptH1);
    assert!(test.is_decided());
    assert_eq!(test.stopped_at(), Some(12));
    // Ten failures would take Λ to 12·ln(9/7) + 10·ln(1/3) = −7.97 < A: without stickiness the
    // decision would flip to AcceptH0.
    for _ in 0..10 {
        assert_eq!(test.update(false), Decision::AcceptH1);
    }
    assert_eq!(test.decision(), Decision::AcceptH1);
    assert!(test.log_likelihood_ratio_f64() < test.boundaries().lower);
    // The data are still recorded.
    assert_eq!(test.observations(), 22);
    assert_eq!((test.successes(), test.failures()), (12, 10));
    assert_eq!(test.stopped_at(), Some(12));
    Ok(())
}

#[test]
fn sprt_accept_h0_is_sticky_and_reset_clears_it() -> R {
    // Three failures reach A = ln(0.1/0.95) (3·ln(1/3) = −3.296 < −2.2513).
    let mut test = Sprt::bernoulli(&q(7, 10), &q(9, 10), 0.05, 0.10)?;
    assert_eq!(test.update(false), Decision::Continue);
    assert_eq!(test.update(false), Decision::Continue);
    assert_eq!(test.update(false), Decision::AcceptH0);
    assert_eq!(test.stopped_at(), Some(3));
    // 30 successes would push Λ to 30·0.2513 − 3.296 = 4.24 > B.
    for _ in 0..30 {
        assert_eq!(test.observe(&qi(1))?, Decision::AcceptH0);
    }
    assert!(test.log_likelihood_ratio_f64() > test.boundaries().upper);
    assert_eq!(test.decision(), Decision::AcceptH0);
    test.reset();
    assert!(!test.is_decided());
    assert_eq!(test.stopped_at(), None);
    assert_eq!(test.decision(), Decision::Continue);
    assert_eq!(test.observations(), 0);
    // A fresh run decides afresh.
    for _ in 0..11 {
        assert_eq!(test.update(true), Decision::Continue);
    }
    assert_eq!(test.update(true), Decision::AcceptH1);
    assert_eq!(test.stopped_at(), Some(12));
    Ok(())
}

#[test]
fn sprt_normal_mean_decision_is_sticky_too() -> R {
    // μ₀ = 0, μ₁ = 1, σ = 1: Λ = Σ(xᵢ − ½); 1/2, 3/2, 4 give Λ = 4 > B = 2.8904.
    let mut test = Sprt::normal_mean(&qi(0), &qi(1), &qi(1), 0.05, 0.10)?;
    assert_eq!(test.observe(&q(1, 2))?, Decision::Continue);
    assert_eq!(test.observe(&q(3, 2))?, Decision::Continue);
    assert_eq!(test.observe(&qi(4))?, Decision::AcceptH1);
    assert_eq!(test.stopped_at(), Some(3));
    assert_eq!(test.observe(&qi(-100))?, Decision::AcceptH1);
    assert_eq!(test.decision(), Decision::AcceptH1);
    assert_eq!(*test.sum(), qi(-94));
    // A clone made after stopping carries the decision; one made before does not.
    let after = test.clone();
    assert!(after.is_decided());
    let mut before = Sprt::normal_mean(&qi(0), &qi(1), &qi(1), 0.05, 0.10)?;
    before.observe(&q(1, 2))?;
    assert!(!before.is_decided());
    Ok(())
}

#[test]
fn sprt_replay_of_the_python_sequence_stops_at_the_seventh_observation() -> R {
    // T T F T T F F: Continue ×6, then AcceptH0 (v14's Python replay of Wald's rule); afterwards
    // T T T T T T T T T T T T (twelve successes) would cross B, but the decision holds.
    let mut test = Sprt::bernoulli(&q(7, 10), &q(9, 10), 0.05, 0.10)?;
    let seq = [true, true, false, true, true, false, false];
    let decisions: Vec<Decision> = seq.iter().map(|&s| test.update(s)).collect();
    assert_eq!(&decisions[..6], &[Decision::Continue; 6]);
    assert_eq!(decisions[6], Decision::AcceptH0);
    assert_eq!(test.stopped_at(), Some(7));
    let later: Vec<Decision> = (0..12).map(|_| test.update(true)).collect();
    assert_eq!(later, vec![Decision::AcceptH0; 12]);
    assert_eq!((test.successes(), test.failures()), (16, 3));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Censorings at times without an event
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn km_censoring_times_include_times_without_an_event() -> R {
    // Hand count: times 3 5 6 7 8 10 12 12, events T F T T F T T F → censored at 5, 8, 12;
    // the life table (event times 3, 6, 7, 10, 12) records only the one at 12.
    let obs = Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 12],
        &[true, false, true, true, false, true, true, false],
    );
    let km = KaplanMeier::fit(&obs)?;
    assert_eq!(km.censoring_times(), vec![qi(5), qi(8), qi(12)]);
    assert_eq!(km.censored_at(&qi(5)), 1);
    assert_eq!(km.censored_at(&qi(8)), 1);
    assert_eq!(km.censored_at(&qi(12)), 1);
    assert_eq!(km.censored_at(&qi(6)), 0);
    assert_eq!(km.censored_at(&qi(100)), 0);
    let in_table: usize = km.table().iter().map(|r| r.censored).sum();
    assert_eq!(in_table, 1);
    let total: usize = km.censoring_times().iter().map(|t| km.censored_at(t)).sum();
    assert_eq!(total, 3);
    assert_eq!(km.event_times(), vec![qi(3), qi(6), qi(7), qi(10), qi(12)]);
    Ok(())
}

#[test]
fn km_censored_at_counts_several_censorings_at_one_time() -> R {
    // Hand count: times 1 2 2 2 3 3, events T F F F T F → 3 censored at 2 (no event there), 1 at 3.
    // Risk set at 3: 6 − 1 − 3 = 2, one event → S(3) = (5/6)(1/2) = 5/12.
    let obs = Observation::from_i64(
        &[1, 2, 2, 2, 3, 3],
        &[true, false, false, false, true, false],
    );
    let km = KaplanMeier::fit(&obs)?;
    assert_eq!(km.censoring_times(), vec![qi(2), qi(3)]);
    assert_eq!(km.censored_at(&qi(2)), 3);
    assert_eq!(km.censored_at(&qi(3)), 1);
    assert_eq!(km.censored_at(&qi(1)), 0);
    let rows = km.table();
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (rows[0].at_risk, rows[0].events, rows[0].censored),
        (6, 1, 0)
    );
    assert_eq!(
        (rows[1].at_risk, rows[1].events, rows[1].censored),
        (2, 1, 1)
    );
    assert_eq!(km.survival_at(&qi(3)), q(5, 12));
    Ok(())
}

#[test]
fn km_censoring_times_are_empty_without_censoring_and_complete_without_events() -> R {
    let all_events = KaplanMeier::fit(&Observation::from_i64(&[1, 2, 3, 4], &[true; 4]))?;
    assert!(all_events.censoring_times().is_empty());
    assert_eq!(all_events.censored_at(&qi(2)), 0);
    // Every subject censored: no life-table row, yet every censoring time is known.
    let none = KaplanMeier::fit(&Observation::from_i64(&[4, 1, 4, 6], &[false; 4]))?;
    assert!(none.table().is_empty());
    assert_eq!(none.censoring_times(), vec![qi(1), qi(4), qi(6)]);
    assert_eq!(none.censored_at(&qi(4)), 2);
    assert_eq!(none.survival_at(&qi(10)), Q::one());
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Lattice walk for quantile_f64 without a closed CDF
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn negative_binomial_with_rational_r_quantile_matches_scipy_and_is_fast() -> R {
    // The crate's NegativeBinomial(r, p) counts failures before the r-th success with success
    // probability p — scipy's nbinom(n, p) parametrisation.
    // scipy.stats.nbinom.ppf([0.9, 0.999, 0.05, 0.5], 1.5, 1/3) = [7, 19, 0, 2]
    let ctx = Context::new();
    let nb = Distribution::negative_binomial(ctx.rational(3, 2), ctx.rational(1, 3));
    let t0 = Instant::now();
    let q90 = nb.quantile_f64(0.9)?;
    let elapsed = t0.elapsed();
    assert_eq!(q90, 7.0);
    assert!(elapsed.as_secs_f64() < 0.5, "took {elapsed:?} (was ~4 s)");
    assert_eq!(nb.quantile_f64(0.999)?, 19.0);
    assert_eq!(nb.quantile_f64(0.05)?, 0.0);
    assert_eq!(nb.quantile_f64(0.5)?, 2.0);
    // scipy.stats.nbinom.ppf([0.1, 0.9, 0.99], 7/3, 1/5) = [2, 18, 31]
    let nb2 = Distribution::negative_binomial(ctx.rational(7, 3), ctx.rational(1, 5));
    assert_eq!(nb2.quantile_f64(0.1)?, 2.0);
    assert_eq!(nb2.quantile_f64(0.9)?, 18.0);
    assert_eq!(nb2.quantile_f64(0.99)?, 31.0);
    // The pmf itself is untouched: P(X = 7) for nb(3/2, 1/3) evaluates (scipy nbinom.pmf(7, 1.5, 1/3)
    // = 0.03539141310618851).
    let p7 = nb.density(&ctx.int(7)).eval_f64()?;
    close(p7, 0.035_391_413_106_188_51, 1e-12);
    Ok(())
}

#[test]
fn lattice_walk_leaves_families_with_a_closed_cdf_alone() -> R {
    // scipy: poisson.ppf(0.1, 2) = 0, binom.ppf(0.5, 10, 1/3) = 3, randint.ppf(0.5, 1, 7) = 3
    let ctx = Context::new();
    assert_eq!(Distribution::poisson(ctx.int(2)).quantile_f64(0.1)?, 0.0);
    assert_eq!(
        Distribution::binomial(ctx.int(10), ctx.rational(1, 3)).quantile_f64(0.5)?,
        3.0
    );
    assert_eq!(Distribution::die(ctx.int(6)).quantile_f64(0.5)?, 3.0);
    // An integer r keeps whatever route it had and agrees with scipy: nbinom.ppf(0.9, 3, 0.4) = 9
    let nb3 = Distribution::negative_binomial(ctx.int(3), ctx.rational(2, 5));
    assert_eq!(nb3.quantile_f64(0.9)?, 9.0);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. quantile_strict
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn km_quantile_strict_follows_statsmodels_where_the_curve_lands_on_the_level() -> R {
    // Four events at 1, 2, 3, 4: S(2) = 1/2 exactly.  R survfit: median 2; statsmodels
    // SurvfuncRight.quantile(0.5): 3 (strict <).
    let km = KaplanMeier::fit(&Observation::from_i64(&[1, 2, 3, 4], &[true; 4]))?;
    assert_eq!(km.quantile(&q(1, 2)), Some(qi(2)));
    assert_eq!(km.quantile_strict(&q(1, 2)), Some(qi(3)));
    assert_eq!(km.quantile_strict(&q(1, 4)), Some(qi(2)));
    assert_eq!(km.quantile_strict(&q(99, 100)), Some(qi(4)));
    // Where the curve does not land on the level the two agree (doc data: median 10 both ways;
    // statsmodels SurvfuncRight quantile(0.5) = 10).
    let obs = Observation::from_i64(
        &[3, 5, 6, 7, 8, 10, 12, 12],
        &[true, false, true, true, false, true, true, false],
    );
    let km = KaplanMeier::fit(&obs)?;
    assert_eq!(km.median(), Some(qi(10)));
    assert_eq!(km.quantile_strict(&q(1, 2)), Some(qi(10)));
    // Never falling far enough: None under both conventions.
    let heavy = KaplanMeier::fit(&Observation::from_i64(
        &[1, 2, 3, 4],
        &[true, false, false, false],
    ))?;
    assert_eq!(heavy.quantile(&q(1, 2)), None);
    assert_eq!(heavy.quantile_strict(&q(1, 2)), None);
    // S(1) = 3/4 lands on 1 − 1/4: `≤` takes it, `<` does not.
    assert_eq!(heavy.quantile(&q(1, 4)), Some(qi(1)));
    assert_eq!(heavy.quantile_strict(&q(1, 4)), None);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. weighted_vote with zero weights
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn weighted_vote_with_zero_weights_elects_nobody_like_an_uncast_majority_vote() -> R {
    // One label cast with weight 0: no winner, no tie (documented), scores run to the label.
    let v = weighted_vote(&[Some(2)], &[qi(0)])?;
    assert_eq!(v.winner, None);
    assert!(v.tied.is_empty());
    assert_eq!(v.scores, vec![qi(0), qi(0), qi(0)]);
    // The same label counted once by majority_vote does win — a weight of 0 is a vote not cast,
    // which is majority_vote's `None`.
    assert_eq!(majority_vote(&[Some(2)]).winner, Some(2));
    assert_eq!(majority_vote(&[None]).winner, None);
    // Any positive weight elects it.
    assert_eq!(weighted_vote(&[Some(2)], &[q(1, 7)])?.winner, Some(2));
    // Mixed: zero-weight raters do not break a tie among the others.
    let v = weighted_vote(&[Some(0), Some(1), Some(0)], &[qi(1), qi(1), qi(0)])?;
    assert_eq!((v.winner, v.tied), (None, vec![0, 1]));
    // The v17 case is unchanged: two labels, both weightless → (None, []).
    let v = weighted_vote(&[Some(0), Some(1)], &[qi(0), qi(0)])?;
    assert_eq!((v.winner, v.tied), (None, vec![]));
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Observation::try_from_*
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn observation_try_from_reports_a_length_mismatch_while_from_zips() -> R {
    let ok = Observation::try_from_i64(&[1, 2, 3], &[true, false, true])?;
    assert_eq!(ok, Observation::from_i64(&[1, 2, 3], &[true, false, true]));
    assert_eq!(ok.len(), 3);
    is_invalid(Observation::try_from_i64(&[1, 2, 3], &[true]));
    is_invalid(Observation::try_from_i64(&[1], &[true, false]));
    let okq = Observation::try_from_q(&[q(1, 2), qi(3)], &[false, true])?;
    assert_eq!(okq, Observation::from_q(&[q(1, 2), qi(3)], &[false, true]));
    is_invalid(Observation::try_from_q(&[q(1, 2)], &[]));
    // The zip is still the documented behaviour of the infallible constructors.
    assert_eq!(Observation::from_i64(&[1, 2, 3], &[true]).len(), 1);
    assert_eq!(Observation::from_q(&[qi(1)], &[true, false]).len(), 1);
    // And an empty pair is fine for both.
    assert!(Observation::try_from_i64(&[], &[])?.is_empty());
    Ok(())
}
