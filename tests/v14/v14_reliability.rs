//! symplex 0.14 — reliability.  Reference values cite scipy 1.18 / statsmodels 0.15 /
//! numpy (`symplex/.venv/bin/python`).
//!
//! Every exact value below comes from a `fractions.Fraction`
//! re-implementation of the published formula in Python (never from a
//! float rounded to a fraction); every float from the cited
//! `scipy.stats` / `statsmodels` / `numpy` call, printed to 15 decimals.
//! Data sets are named once here and reused across tests.

use symplex::linprog::{Q, q, qi};
use symplex::prelude::*;
use symplex::stats::agreement::RatingTable;
use symplex::stats::data::{from_i64, to_f64};
use symplex::stats::hypothesis::{Alternative, chi_square_independence, counts};
use symplex::stats::reliability::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn f(x: &Q) -> f64 {
    to_f64(std::slice::from_ref(x))[0]
}

// ── Data sets ────────────────────────────────────────────────────────

/// A: eight respondents × four Likert items (1–5).
fn table_a() -> RatingTable {
    RatingTable::from_i64(&[
        &[4, 3, 5, 4],
        &[2, 2, 3, 1],
        &[5, 4, 4, 5],
        &[3, 3, 2, 3],
        &[1, 2, 1, 2],
        &[4, 4, 5, 3],
        &[2, 1, 2, 2],
        &[5, 5, 4, 4],
    ])
    .unwrap()
}

/// B: ten respondents × five dichotomous items.
fn table_b() -> RatingTable {
    RatingTable::from_i64(&[
        &[1, 1, 1, 1, 1],
        &[1, 1, 1, 0, 1],
        &[1, 1, 0, 1, 0],
        &[1, 0, 1, 0, 0],
        &[0, 1, 0, 1, 1],
        &[1, 0, 0, 0, 0],
        &[0, 0, 1, 0, 0],
        &[1, 1, 1, 1, 0],
        &[0, 0, 0, 0, 1],
        &[1, 1, 0, 1, 1],
    ])
    .unwrap()
}

fn b_rows() -> Vec<Vec<i64>> {
    vec![
        vec![1, 1, 1, 1, 1],
        vec![1, 1, 1, 0, 1],
        vec![1, 1, 0, 1, 0],
        vec![1, 0, 1, 0, 0],
        vec![0, 1, 0, 1, 1],
        vec![1, 0, 0, 0, 0],
        vec![0, 0, 1, 0, 0],
        vec![1, 1, 1, 1, 0],
        vec![0, 0, 0, 0, 1],
        vec![1, 1, 0, 1, 1],
    ]
}

/// A with a ninth, incomplete respondent appended.
fn table_a_missing() -> RatingTable {
    let mut rows: Vec<Vec<Option<Q>>> = table_a().rows().to_vec();
    rows.push(vec![Some(qi(3)), None, Some(qi(4)), Some(qi(2))]);
    RatingTable::new(rows).unwrap()
}

/// D1 (v13): two raters, 20 items, categories 1..3; confusion
/// [[5, 2, 0], [0, 5, 2], [1, 0, 5]].
fn d1() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 2, 2, 3, 1]),
        from_i64(&[1, 2, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 3, 2]),
    )
}

fn t2x2() -> Vec<Vec<usize>> {
    vec![vec![20, 5], vec![10, 15]]
}

fn t3() -> Vec<Vec<usize>> {
    vec![vec![10, 5, 0], vec![2, 8, 3], vec![0, 4, 12]]
}

/// Ordinal pair with ties: C = 44, D = 3, T_x = 8, T_y = 9, T_xy = 2.
fn ordinal1() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 4]),
        from_i64(&[1, 1, 2, 2, 3, 2, 4, 3, 5, 2, 3, 4]),
    )
}

/// Ordinal pair whose x and y ties balance: C = 25, D = 7, T_x = T_y = 15.
fn ordinal2() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3]),
        from_i64(&[1, 1, 2, 3, 1, 2, 2, 3, 3, 2, 3, 3]),
    )
}

fn cont1() -> Vec<Vec<Q>> {
    counts(&[&[10, 20, 30], &[6, 9, 17]])
}

fn cont2() -> Vec<Vec<Q>> {
    counts(&[&[20, 15, 5], &[10, 25, 10], &[5, 10, 30]])
}

fn xy10() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        from_i64(&[2, 1, 4, 3, 7, 8, 5, 6, 10, 9]),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Internal consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cronbach_alpha_a_exact() {
    // Fraction: item variances 4/3 + 4/3 + 5/2 + 11/6 ... Σ = 55/7 with the
    // total variance 180/7 → 4/3 · (1 − 55/180) = 25/27; numpy 0.9259259259259258.
    let alpha = cronbach_alpha(&table_a()).unwrap();
    assert_eq!(alpha, q(25, 27));
    assert!(close(f(&alpha), 0.925_925_925_925_925_8));
}

#[test]
fn cronbach_alpha_b_exact() {
    // Fraction re-implementation: 95/196 (0.4846938775510204).
    assert_eq!(cronbach_alpha(&table_b()).unwrap(), q(95, 196));
}

#[test]
fn cronbach_alpha_rejects_missing_and_complete_drops() {
    let t = table_a_missing();
    assert!(cronbach_alpha(&t).is_err());
    // List-wise deletion leaves exactly A.
    assert_eq!(cronbach_alpha_complete(&t).unwrap(), q(25, 27));
    assert_eq!(cronbach_alpha_complete(&table_a()).unwrap(), q(25, 27));
}

#[test]
fn cronbach_alpha_degenerate_inputs_error() {
    // One item.
    let one = RatingTable::from_i64(&[&[1], &[2], &[3]]).unwrap();
    assert!(cronbach_alpha(&one).is_err());
    // One respondent.
    let single = RatingTable::from_i64(&[&[1, 2, 3]]).unwrap();
    assert!(cronbach_alpha(&single).is_err());
    // Constant totals (zero total variance).
    let flat = RatingTable::from_i64(&[&[1, 0], &[0, 1], &[1, 0]]).unwrap();
    assert!(cronbach_alpha(&flat).is_err());
}

#[test]
fn kr20_equals_alpha_on_dichotomous_items() {
    // Fraction: 5/4 · (1 − (21/100 + 6/25 + 1/4 + 1/4 + 1/4) / (9/5)) = 95/196.
    let t = table_b();
    assert_eq!(kr20(&t).unwrap(), q(95, 196));
    assert_eq!(kr20(&t).unwrap(), cronbach_alpha(&t).unwrap());
    // Polytomous items are rejected.
    assert!(kr20(&table_a()).is_err());
}

#[test]
fn standardized_alpha_and_average_inter_item_correlation() {
    let ctx = Context::new();
    // numpy: mean of the upper triangle of corrcoef(A.T) = 0.760452911806828,
    // 4r̄/(1 + 3r̄) = 0.926997592306080.
    let r = average_inter_item_correlation(&ctx, &table_a()).unwrap();
    assert!(close(r.eval_f64().unwrap(), 0.760_452_911_806_828));
    let a = standardized_alpha(&ctx, &table_a()).unwrap();
    assert!(close(a.eval_f64().unwrap(), 0.926_997_592_306_080));
    // B: r̄ = 0.159931108417748, α_std = 0.487676786213695.
    let r = average_inter_item_correlation(&ctx, &table_b()).unwrap();
    assert!(close(r.eval_f64().unwrap(), 0.159_931_108_417_748));
    let a = standardized_alpha(&ctx, &table_b()).unwrap();
    assert!(close(a.eval_f64().unwrap(), 0.487_676_786_213_695));
}

#[test]
fn split_half_odd_even_a() {
    let ctx = Context::new();
    // numpy: corrcoef(items {0, 2}, items {1, 3}) = 0.845405751313380; 2r/(1+r) = 0.916227502500957.
    // Fraction: cov = 41/7, variances 8 and 6 → r² = 1681/2352.
    let r = split_half_correlation(&ctx, &table_a(), &SplitHalf::OddEven).unwrap();
    assert!(close(r.eval_f64().unwrap(), 0.845_405_751_313_380));
    assert_eq!((&r * &r).simplify(), ctx.from_ratio(q(1681, 2352)));
    let sb = split_half(&ctx, &table_a(), &SplitHalf::OddEven).unwrap();
    assert!(close(sb.eval_f64().unwrap(), 0.916_227_502_500_957));
    // First-last: r = 0.897926289553223, SB = 0.946218295721692.
    let sb = split_half(&ctx, &table_a(), &SplitHalf::FirstLast).unwrap();
    assert!(close(sb.eval_f64().unwrap(), 0.946_218_295_721_692));
}

#[test]
fn split_half_first_last_and_custom_b() {
    let ctx = Context::new();
    // numpy: first ⌊5/2⌋ = 2 items vs last 3: r = 0.555835714703748, SB = 0.714517232700995.
    let r = split_half_correlation(&ctx, &table_b(), &SplitHalf::FirstLast).unwrap();
    assert!(close(r.eval_f64().unwrap(), 0.555_835_714_703_748));
    let sb = split_half(&ctx, &table_b(), &SplitHalf::FirstLast).unwrap();
    assert!(close(sb.eval_f64().unwrap(), 0.714_517_232_700_995));
    // Odd-even: r = 0.312153288970447, SB = 0.475787839110431.
    let sb = split_half(&ctx, &table_b(), &SplitHalf::OddEven).unwrap();
    assert!(close(sb.eval_f64().unwrap(), 0.475_787_839_110_431));
    // Custom {0, 3, 4} vs {1, 2}: r = 0.523809523809524 = 11/21 exactly, SB = 11/16.
    let split = SplitHalf::Custom(vec![true, false, false, true, true]);
    let r = split_half_correlation(&ctx, &table_b(), &split).unwrap();
    assert_eq!(r.simplify(), ctx.from_ratio(q(11, 21)));
    assert_eq!(
        split_half(&ctx, &table_b(), &split).unwrap().simplify(),
        ctx.from_ratio(q(11, 16))
    );
}

#[test]
fn split_half_rejects_bad_splits() {
    let ctx = Context::new();
    let t = table_b();
    assert!(split_half(&ctx, &t, &SplitHalf::Custom(vec![true, false])).is_err());
    assert!(split_half(&ctx, &t, &SplitHalf::Custom(vec![true; 5])).is_err());
    assert!(split_half(&ctx, &t, &SplitHalf::Custom(vec![false; 5])).is_err());
    // A single item cannot be split.
    let one = RatingTable::from_i64(&[&[1], &[2]]).unwrap();
    assert!(split_half(&ctx, &one, &SplitHalf::OddEven).is_err());
}

#[test]
fn spearman_brown_prophecy_formula() {
    let ctx = Context::new();
    // 2 · 0.6 / (1 + 0.6) = 3/4; tripling: 3 · 0.6 / (1 + 2 · 0.6) = 9/11.
    assert_eq!(
        spearman_brown(&ctx, &ctx.rational(3, 5), 2).simplify(),
        ctx.rational(3, 4)
    );
    assert_eq!(
        spearman_brown(&ctx, &ctx.rational(3, 5), 3).simplify(),
        ctx.rational(9, 11)
    );
    // k = 1 is the identity.
    assert_eq!(
        spearman_brown(&ctx, &ctx.rational(3, 5), 1).simplify(),
        ctx.rational(3, 5)
    );
}

#[test]
fn alpha_if_deleted_a() {
    // Fraction re-implementation on the four 3-item sub-scales.
    assert_eq!(
        alpha_if_deleted(&table_a()).unwrap(),
        vec![q(52, 61), q(65, 72), q(198, 211), q(201, 220)]
    );
}

#[test]
fn alpha_if_deleted_b_has_an_exact_zero() {
    // Fraction: dropping item 1 leaves Σ s²ᵢ = s²_X exactly, so α = 0.
    assert_eq!(
        alpha_if_deleted(&table_b()).unwrap(),
        vec![q(200, 447), qi(0), q(344, 543), q(104, 363), q(88, 161)]
    );
    // Needs at least three items.
    let two = RatingTable::from_i64(&[&[1, 0], &[1, 1], &[0, 0]]).unwrap();
    assert!(alpha_if_deleted(&two).is_err());
}

#[test]
fn guttman_lambda2_is_at_least_alpha() {
    let ctx = Context::new();
    // numpy: A → 0.929315917921957 (= (180/7 − 55/7 + √(766/21)) / (180/7)); B → 0.603312552798751.
    let l2a = guttman_lambda2(&ctx, &table_a())
        .unwrap()
        .eval_f64()
        .unwrap();
    assert!(close(l2a, 0.929_315_917_921_957));
    assert!(l2a >= f(&cronbach_alpha(&table_a()).unwrap()));
    let l2b = guttman_lambda2(&ctx, &table_b())
        .unwrap()
        .eval_f64()
        .unwrap();
    assert!(close(l2b, 0.603_312_552_798_751));
    assert!(l2b >= f(&cronbach_alpha(&table_b()).unwrap()));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Item analysis
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn total_scores_and_difficulty_a() {
    assert_eq!(
        total_scores(&table_a()).unwrap(),
        from_i64(&[16, 8, 18, 11, 6, 16, 7, 18])
    );
    // Column means: 26/8, 24/8, 26/8, 24/8.
    assert_eq!(
        item_difficulty(&table_a()).unwrap(),
        vec![q(13, 4), qi(3), q(13, 4), qi(3)]
    );
    assert!(total_scores(&table_a_missing()).is_err());
}

#[test]
fn item_difficulty_b_is_the_proportion_correct() {
    assert_eq!(
        item_difficulty(&table_b()).unwrap(),
        vec![q(7, 10), q(3, 5), q(1, 2), q(1, 2), q(1, 2)]
    );
}

#[test]
fn discrimination_index_a() {
    // Fraction re-implementation of the thirds rule: g = ⌊8/3⌋ = 2;
    // upper = rows {2, 7} (totals 18, 18), lower = rows {4, 6} (totals 6, 7).
    assert_eq!(
        item_discrimination_index(&table_a()).unwrap(),
        vec![q(7, 2), qi(3), q(5, 2), q(5, 2)]
    );
}

#[test]
fn discrimination_index_b_and_minimum_size() {
    // g = 3; upper = rows {0, 1, 7} (5, 4, 4 — row 9 also totals 4 but comes
    // later), lower = rows {5, 6, 8} (all 1).
    assert_eq!(
        item_discrimination_index(&table_b()).unwrap(),
        vec![q(2, 3), qi(1), q(2, 3), q(2, 3), q(1, 3)]
    );
    let two = RatingTable::from_i64(&[&[1, 0], &[0, 1]]).unwrap();
    assert!(item_discrimination_index(&two).is_err());
}

#[test]
fn point_biserial_matches_scipy() {
    let ctx = Context::new();
    let rows = b_rows();
    let total = total_scores(&table_b()).unwrap();
    // scipy: pointbiserialr(B[:, j], B.sum(1)).statistic for j = 0..5
    let expected = [
        0.529_957_733_430_267,
        0.903_978_357_455_697,
        0.285_714_285_714_286,
        0.714_285_714_285_714,
        0.428_571_428_571_429,
    ];
    for (j, want) in expected.iter().enumerate() {
        let item: Vec<Q> = rows.iter().map(|r| qi(r[j])).collect();
        let r = point_biserial(&ctx, &item, &total).unwrap();
        assert!(close(r.eval_f64().unwrap(), *want), "item {j}");
    }
    // Items 2, 3, 4 have rational correlations: 2/7, 5/7, 3/7.
    for (j, want) in [(2, q(2, 7)), (3, q(5, 7)), (4, q(3, 7))] {
        let item: Vec<Q> = rows.iter().map(|r| qi(r[j])).collect();
        assert_eq!(
            point_biserial(&ctx, &item, &total).unwrap(),
            ctx.from_ratio(want)
        );
    }
}

#[test]
fn point_biserial_rejects_bad_input() {
    let ctx = Context::new();
    let total = from_i64(&[5, 4, 3, 2]);
    assert!(point_biserial(&ctx, &from_i64(&[1, 2, 0, 1]), &total).is_err());
    assert!(point_biserial(&ctx, &from_i64(&[1, 1, 1, 1]), &total).is_err());
    assert!(point_biserial(&ctx, &from_i64(&[1, 0, 1]), &total).is_err());
    assert!(point_biserial(&ctx, &from_i64(&[1, 0, 1, 0]), &from_i64(&[2, 2, 2, 2])).is_err());
}

#[test]
fn item_total_correlation_a() {
    let ctx = Context::new();
    // numpy: corrcoef(A[:, j], A.sum(1))[0, 1]
    let want = [
        0.984_467_179_361_579,
        0.903_696_114_115_064,
        0.851_942_751_370_597,
        0.882_179_539_969_467,
    ];
    let got = item_total_correlation(&ctx, &table_a()).unwrap();
    assert_eq!(got.len(), 4);
    for (g, w) in got.iter().zip(want) {
        assert!(close(g.eval_f64().unwrap(), w));
    }
}

#[test]
fn corrected_item_total_correlation_a_and_b() {
    let ctx = Context::new();
    // numpy: corrcoef(A[:, j], A.sum(1) − A[:, j])[0, 1]
    let want_a = [
        0.969_206_835_331_211,
        0.833_333_333_333_333,
        0.729_507_795_438_098,
        0.798_198_729_716_611,
    ];
    let got = corrected_item_total_correlation(&ctx, &table_a()).unwrap();
    for (g, w) in got.iter().zip(want_a) {
        assert!(close(g.eval_f64().unwrap(), w));
    }
    assert_eq!(got[1], ctx.from_ratio(q(5, 6)));
    // Fraction: for item 0, r² = (73/2)² / ((31/2)(183/2)) = 5329/5673.
    assert_eq!(
        (&got[0] * &got[0]).simplify(),
        ctx.from_ratio(q(5329, 5673))
    );
    // B: 0.232402379702539, 19/24, −0.074329414624717, 5/11, 0.078811040623910.
    let got = corrected_item_total_correlation(&ctx, &table_b()).unwrap();
    assert!(close(got[0].eval_f64().unwrap(), 0.232_402_379_702_539));
    assert_eq!(got[1], ctx.from_ratio(q(19, 24)));
    assert!(close(got[2].eval_f64().unwrap(), -0.074_329_414_624_717));
    assert_eq!(got[3], ctx.from_ratio(q(5, 11)));
    assert!(close(got[4].eval_f64().unwrap(), 0.078_811_040_623_910));
}

#[test]
fn item_response_summary_bundles_the_per_item_statistics() {
    let ctx = Context::new();
    let t = table_b();
    let s = item_response_summary(&ctx, &t).unwrap();
    assert_eq!(s.len(), 5);
    let difficulty: Vec<Q> = s.iter().map(|i| i.difficulty.clone()).collect();
    assert_eq!(difficulty, item_difficulty(&t).unwrap());
    let disc: Vec<Q> = s.iter().map(|i| i.discrimination.clone()).collect();
    assert_eq!(disc, item_discrimination_index(&t).unwrap());
    let deleted: Vec<Q> = s
        .iter()
        .map(|i| i.alpha_if_deleted.clone().unwrap())
        .collect();
    assert_eq!(deleted, alpha_if_deleted(&t).unwrap());
    let corrected = corrected_item_total_correlation(&ctx, &t).unwrap();
    for (i, c) in s.iter().zip(&corrected) {
        assert_eq!(i.corrected_item_total.as_ref().unwrap(), c);
    }
    // scipy pointbiserialr(item 1, total) = 0.903978357455697
    assert!(close(
        s[1].item_total.as_ref().unwrap().eval_f64().unwrap(),
        0.903_978_357_455_697
    ));
}

#[test]
fn item_response_summary_uses_none_where_undefined() {
    let ctx = Context::new();
    // Two items: α-if-deleted needs three, so it is None; item 1 is constant.
    let t = RatingTable::from_i64(&[&[1, 1], &[0, 1], &[1, 1], &[0, 1]]).unwrap();
    let s = item_response_summary(&ctx, &t).unwrap();
    assert_eq!(s[0].alpha_if_deleted, None);
    assert!(s[1].item_total.is_none());
    assert!(s[1].corrected_item_total.is_none());
    assert_eq!(s[1].difficulty, qi(1));
    assert_eq!(s[1].discrimination, qi(0));
    // Item 0's corrected correlation is with a constant remainder → None,
    // but its item–total correlation exists.
    assert!(s[0].corrected_item_total.is_none());
    assert!(s[0].item_total.is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Agreement extras
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn kappa_ci_d1_matches_statsmodels() {
    let ctx = Context::new();
    let (a, b) = d1();
    // statsmodels: cohens_kappa([[5,2,0],[0,5,2],[1,0,5]], return_results=True)
    //   kappa 0.625468164794008, var_kappa 0.020944304373709,
    //   kappa_low 0.341819292454126, kappa_upp 0.909117037133889
    // Fraction: κ = 167/267, Var = 35480500/1694040507.
    let ci = cohen_kappa_ci(&ctx, &a, &b, 0.95).unwrap();
    assert_eq!(ci.kappa, q(167, 267));
    assert_eq!(ci.variance, q(35_480_500, 1_694_040_507));
    assert!(close(ci.lower, 0.341_819_292_454_126));
    assert!(close(ci.upper, 0.909_117_037_133_889));
    assert_eq!(ci.confidence, 0.95);
    assert!(close(
        ci.se.eval_f64().unwrap(),
        0.020_944_304_373_709_f64.sqrt()
    ));
    assert_eq!(
        (&ci.se * &ci.se).simplify(),
        ctx.from_ratio(ci.variance.clone())
    );
}

#[test]
fn kappa_ci_2x2_exact_variance_and_other_levels() {
    let ctx = Context::new();
    // statsmodels: kappa 0.4, var_kappa 0.016128, kappa_low 0.151092290476661, kappa_upp 0.648907709523339
    let ci = kappa_ci_from_confusion(&ctx, &t2x2(), 0.95).unwrap();
    assert_eq!(ci.kappa, q(2, 5));
    assert_eq!(ci.variance, q(252, 15625));
    assert!(close(ci.lower, 0.151_092_290_476_661));
    assert!(close(ci.upper, 0.648_907_709_523_339));
    // 90 %: kappa ∓ norm.isf(0.05)·√var = (0.191110065279222, 0.608889934720778)
    let ci = kappa_ci_from_confusion(&ctx, &t2x2(), 0.90).unwrap();
    assert!(close(ci.lower, 0.191_110_065_279_222));
    assert!(close(ci.upper, 0.608_889_934_720_778));
}

#[test]
fn kappa_ci_3x3_unbalanced() {
    let ctx = Context::new();
    // statsmodels on [[10,5,0],[2,8,3],[0,4,12]]: kappa 0.524324324324324,
    // var_kappa 0.010855178884997, kappa_low 0.320119224784704, kappa_upp 0.728529423863944.
    // Fraction: 97/185, 89006544/8199454375.
    let ci = kappa_ci_from_confusion(&ctx, &t3(), 0.95).unwrap();
    assert_eq!(ci.kappa, q(97, 185));
    assert_eq!(ci.variance, q(89_006_544, 8_199_454_375));
    assert!(close(ci.lower, 0.320_119_224_784_704));
    assert!(close(ci.upper, 0.728_529_423_863_944));
}

#[test]
fn kappa_test_d1_matches_statsmodels() {
    let ctx = Context::new();
    let (a, b) = d1();
    // statsmodels: var_kappa0 0.024778717614218 (= 35329/1425780), z_value 3.973432105774423,
    //   pvalue_one_sided 0.000035422179259, pvalue_two_sided 0.000070844358517
    let r = kappa_test(&ctx, &a, &b, Alternative::TwoSided).unwrap();
    assert!(close(r.statistic_f64().unwrap(), 3.973_432_105_774_423));
    assert!(close(r.p_value_f64().unwrap(), 0.000_070_844_358_517));
    assert_eq!(r.df, None);
    // z² = κ² / Var₀ exactly.
    let z2 = (&r.statistic * &r.statistic).simplify();
    assert_eq!(
        z2,
        ctx.from_ratio(q(167, 267) * q(167, 267) / q(35_329, 1_425_780))
    );
    let one = kappa_test(&ctx, &a, &b, Alternative::Greater).unwrap();
    assert!(close(one.p_value_f64().unwrap(), 0.000_035_422_179_259));
    let less = kappa_test(&ctx, &a, &b, Alternative::Less).unwrap();
    assert!(close(
        less.p_value_f64().unwrap(),
        1.0 - 0.000_035_422_179_259
    ));
}

#[test]
fn kappa_test_from_confusion_2x2_and_3x3() {
    let ctx = Context::new();
    // statsmodels 2×2: z 2.886751345948128, p1 0.001946208561389, p2 0.003892417122779
    let r = kappa_test_from_confusion(&ctx, &t2x2(), Alternative::TwoSided).unwrap();
    assert!(close(r.statistic_f64().unwrap(), 2.886_751_345_948_128));
    assert!(close(r.p_value_f64().unwrap(), 0.003_892_417_122_779));
    let g = kappa_test_from_confusion(&ctx, &t2x2(), Alternative::Greater).unwrap();
    assert!(close(g.p_value_f64().unwrap(), 0.001_946_208_561_389));
    // Var₀ = 12/625 → z = (2/5)/√(12/625) = 5/√3.
    assert_eq!(
        (&r.statistic * &r.statistic).simplify(),
        ctx.rational(25, 3)
    );
    // 3×3: z 4.977036960541850, p2 0.000000645649967
    let r = kappa_test_from_confusion(&ctx, &t3(), Alternative::TwoSided).unwrap();
    assert!(close(r.statistic_f64().unwrap(), 4.977_036_960_541_85));
    assert!(close(r.p_value_f64().unwrap(), 0.000_000_645_649_967));
}

#[test]
fn kappa_maximum_from_marginals() {
    let (a, b) = d1();
    // statsmodels kappa_max: D1 0.925093632958801 (= 247/267), 2×2 0.8, 3×3 0.864092664092664 (= 1119/1295).
    assert_eq!(cohen_kappa_maximum(&a, &b).unwrap(), q(247, 267));
    assert_eq!(kappa_maximum_from_confusion(&t2x2()).unwrap(), q(4, 5));
    assert_eq!(kappa_maximum_from_confusion(&t3()).unwrap(), q(1119, 1295));
    // Equal marginals allow κ = 1.
    assert_eq!(
        cohen_kappa_maximum(&from_i64(&[0, 0, 1, 1]), &from_i64(&[1, 0, 0, 1])).unwrap(),
        qi(1)
    );
}

#[test]
fn kappa_inference_rejects_bad_input() {
    let ctx = Context::new();
    let (a, b) = d1();
    assert!(cohen_kappa_ci(&ctx, &a, &b, 1.0).is_err());
    assert!(cohen_kappa_ci(&ctx, &a, &b, 0.0).is_err());
    assert!(cohen_kappa_ci(&ctx, &a, &b[..5], 0.95).is_err());
    assert!(kappa_ci_from_confusion(&ctx, &[vec![1, 2], vec![3]], 0.95).is_err());
    // A single category: p_e = 1.
    assert!(
        kappa_test(
            &ctx,
            &from_i64(&[1, 1, 1]),
            &from_i64(&[1, 1, 1]),
            Alternative::TwoSided
        )
        .is_err()
    );
    assert!(cohen_kappa_maximum(&from_i64(&[2, 2]), &from_i64(&[2, 2])).is_err());
}

#[test]
fn cochrans_q_matches_statsmodels() {
    let ctx = Context::new();
    // statsmodels: cochrans_q(B) → statistic 1.523809523809524 (= 32/21), pvalue 0.822415705767248, df 4
    let r = cochrans_q(&ctx, &table_b()).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(32, 21)));
    assert_eq!(r.df, Some(ctx.int(4)));
    assert!(close(r.p_value_f64().unwrap(), 0.822_415_705_767_248));
    // 12 subjects × 3 treatments: statistic 11.555555555555555 (= 104/9), pvalue 0.003095586852365, df 2
    let c12 = RatingTable::from_i64(&[
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
    let r = cochrans_q(&ctx, &c12).unwrap();
    assert_eq!(r.statistic_exact(), Some(q(104, 9)));
    assert_eq!(r.df, Some(ctx.int(2)));
    assert!(close(r.p_value_f64().unwrap(), 0.003_095_586_852_365));
}

#[test]
fn cochrans_q_rejects_bad_input() {
    let ctx = Context::new();
    assert!(cochrans_q(&ctx, &table_a()).is_err());
    // Every subject constant: zero denominator.
    let flat = RatingTable::from_i64(&[&[1, 1, 1], &[0, 0, 0], &[1, 1, 1]]).unwrap();
    assert!(cochrans_q(&ctx, &flat).is_err());
    let one = RatingTable::from_i64(&[&[1], &[0]]).unwrap();
    assert!(cochrans_q(&ctx, &one).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Ordinal association
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn concordance_counts_with_ties() {
    let (x, y) = ordinal1();
    let c = concordance_counts(&x, &y).unwrap();
    assert_eq!(
        c,
        ConcordanceCounts {
            concordant: 44,
            discordant: 3,
            ties_x: 8,
            ties_y: 9,
            ties_both: 2,
        }
    );
    assert_eq!(c.pairs(), 66);
    let (x, y) = ordinal2();
    let c = concordance_counts(&x, &y).unwrap();
    assert_eq!(
        (c.concordant, c.discordant, c.ties_x, c.ties_y, c.ties_both),
        (25, 7, 15, 15, 4)
    );
}

#[test]
fn gamma_somers_d_and_tau_c_with_ties() {
    let (x, y) = ordinal1();
    // Fraction: γ = 41/47.
    assert_eq!(goodman_kruskal_gamma(&x, &y).unwrap(), q(41, 47));
    // scipy: somersd(x, y).statistic = 0.732142857142857 (= 41/56), somersd(y, x) = 0.745454545454545 (= 41/55);
    // symmetric 2(C−D)/(2(C+D)+Tx+Ty) = 82/111.
    assert_eq!(somers_d(&x, &y, Dependent::Y).unwrap(), q(41, 56));
    assert_eq!(somers_d(&x, &y, Dependent::X).unwrap(), q(41, 55));
    assert_eq!(somers_d(&y, &x, Dependent::Y).unwrap(), q(41, 55));
    assert_eq!(somers_d(&x, &y, Dependent::Symmetric).unwrap(), q(82, 111));
    // scipy: kendalltau(x, y, variant='c').statistic = 0.711805555555556 (= 205/288)
    assert_eq!(kendall_tau_c(&x, &y).unwrap(), q(205, 288));
    assert!(close(
        f(&kendall_tau_c(&x, &y).unwrap()),
        0.711_805_555_555_556
    ));
}

#[test]
fn somers_d_coincides_when_ties_balance() {
    let (x, y) = ordinal2();
    // scipy: somersd(x2, y2) = somersd(y2, x2) = 0.382978723404255 (= 18/47); γ = 9/16; τ-c = 3/8.
    assert_eq!(somers_d(&x, &y, Dependent::Y).unwrap(), q(18, 47));
    assert_eq!(somers_d(&x, &y, Dependent::X).unwrap(), q(18, 47));
    assert_eq!(somers_d(&x, &y, Dependent::Symmetric).unwrap(), q(18, 47));
    assert_eq!(goodman_kruskal_gamma(&x, &y).unwrap(), q(9, 16));
    assert_eq!(kendall_tau_c(&x, &y).unwrap(), q(3, 8));
}

#[test]
fn perfect_concordance_gives_one() {
    let x = from_i64(&[1, 2, 3, 4, 5]);
    let y = from_i64(&[2, 3, 5, 7, 11]);
    // scipy: somersd(x, y).statistic = 1.0; no ties at all.
    let c = concordance_counts(&x, &y).unwrap();
    assert_eq!((c.concordant, c.discordant, c.pairs()), (10, 0, 10));
    assert_eq!(goodman_kruskal_gamma(&x, &y).unwrap(), qi(1));
    assert_eq!(somers_d(&x, &y, Dependent::Y).unwrap(), qi(1));
    assert_eq!(somers_d(&x, &y, Dependent::Symmetric).unwrap(), qi(1));
    // τ-c with m = n = 5: 2·5·10/(25·4) = 1.
    assert_eq!(kendall_tau_c(&x, &y).unwrap(), qi(1));
    // Reversal flips the sign.
    let yr = from_i64(&[11, 7, 5, 3, 2]);
    assert_eq!(goodman_kruskal_gamma(&x, &yr).unwrap(), qi(-1));
}

#[test]
fn ordinal_measures_reject_degenerate_input() {
    let x = from_i64(&[1, 2, 3]);
    assert!(concordance_counts(&x, &from_i64(&[1, 2])).is_err());
    assert!(concordance_counts(&from_i64(&[1]), &from_i64(&[1])).is_err());
    let flat = from_i64(&[2, 2, 2]);
    // Every pair tied on x: γ undefined, D_{Y|X} undefined, τ-c undefined.
    assert!(goodman_kruskal_gamma(&flat, &x).is_err());
    assert!(somers_d(&flat, &x, Dependent::Y).is_err());
    assert!(kendall_tau_c(&flat, &x).is_err());
    // …but D_{X|Y} (y = x the independent variable, untied) is defined: 0/(0+0+3) = 0.
    assert_eq!(somers_d(&flat, &x, Dependent::X).unwrap(), qi(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Contingency diagnostics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expected_counts_exact() {
    // statsmodels Table(t).fittedvalues: [[10.434782608695652, 18.91304347826087, 30.65217391304348],
    //   [5.565217391304348, 10.08695652173913, 16.347826086956523]]
    assert_eq!(
        expected_counts(&cont1()).unwrap(),
        vec![
            vec![q(240, 23), q(435, 23), q(705, 23)],
            vec![q(128, 23), q(232, 23), q(376, 23)],
        ]
    );
    let e = expected_counts(&cont2()).unwrap();
    // 10.76923076923077 = 140/13; 12.115384615384615 = 315/26
    assert_eq!(e[0][0], q(140, 13));
    assert_eq!(e[1][0], q(315, 26));
    assert_eq!(e[2][2], q(405, 26));
}

#[test]
fn chi2_contributions_exact_and_sum_to_the_statistic() {
    let ctx = Context::new();
    // statsmodels Table(t).chi2_contribs: [[0.018115942028985522, 0.062468765617191224, 0.013876040703052817],
    //   [0.03396739130434785, 0.11712893553223394, 0.026017576318223736]]
    let c = chi2_contributions(&cont1()).unwrap();
    assert_eq!(
        c,
        vec![
            vec![q(5, 276), q(125, 2001), q(15, 1081)],
            vec![q(25, 736), q(625, 5336), q(225, 8648)],
        ]
    );
    let total: Q = c.iter().flatten().fold(qi(0), |acc, x| acc + x);
    assert_eq!(
        total,
        chi_square_independence(&ctx, &cont1(), false)
            .unwrap()
            .statistic
    );
    let c = chi2_contributions(&cont2()).unwrap();
    // 7.9120879120879115 = 720/91; 13.35470085470086 = 3125/234
    assert_eq!(c[0][0], q(720, 91));
    assert_eq!(c[2][2], q(3125, 234));
}

#[test]
fn standardized_residuals_match_resid_pearson() {
    let ctx = Context::new();
    // statsmodels Table(t2).resid_pearson
    let want = [
        [
            2.812_843_385_630_972_5,
            -0.098_058_067_569_092_12,
            -2.377_332_412_959_829,
        ],
        [
            -0.607_744_082_120_812_8,
            1.849_000_654_084_096_2,
            -1.413_037_919_994_657_7,
        ],
        [
            -2.044_230_094_406_370_5,
            -1.756_550_621_379_892_5,
            3.654_408_413_779_289,
        ],
    ];
    let r = standardized_residuals(&ctx, &cont2()).unwrap();
    let contrib = chi2_contributions(&cont2()).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(close(r[i][j].eval_f64().unwrap(), want[i][j]), "({i}, {j})");
            // Squares are the χ² contributions, exactly.
            assert_eq!(
                (&r[i][j] * &r[i][j]).simplify(),
                ctx.from_ratio(contrib[i][j].clone())
            );
        }
    }
}

#[test]
fn adjusted_residuals_match_standardized_resids() {
    let ctx = Context::new();
    // statsmodels Table(t1).standardized_resids
    let want1 = [
        [
            -0.251_094_097_131_006_1,
            0.512_122_698_990_566_4,
            -0.285_587_611_229_800_7,
        ],
        [
            0.251_094_097_131_006_1,
            -0.512_122_698_990_567_2,
            0.285_587_611_229_799_1,
        ],
    ];
    let r = adjusted_residuals(&ctx, &cont1()).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                close(r[i][j].eval_f64().unwrap(), want1[i][j]),
                "({i}, {j})"
            );
        }
    }
    // A 2 × c table's adjusted residuals are antisymmetric in the rows, exactly.
    for (top, bottom) in r[0].iter().zip(&r[1]) {
        assert_eq!((top + bottom).simplify(), ctx.zero());
    }
    // statsmodels Table(t2).standardized_resids
    let want2 = [
        [
            3.954_629_912_439_455_7,
            -0.150_231_303_144_333_03,
            -3.533_479_258_712_322_5,
        ],
        [
            -0.879_210_417_800_808_6,
            2.914_915_440_650_686,
            -2.161_116_818_815_359,
        ],
        [
            -2.957_344_132_602_72,
            -2.769_169_668_618_153,
            5.589_095_221_074_207,
        ],
    ];
    let r = adjusted_residuals(&ctx, &cont2()).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                close(r[i][j].eval_f64().unwrap(), want2[i][j]),
                "({i}, {j})"
            );
        }
    }
}

#[test]
fn contingency_diagnostics_reject_bad_tables() {
    let ctx = Context::new();
    assert!(expected_counts(&counts(&[&[1, 2], &[0, 0]])).is_err());
    assert!(expected_counts(&counts(&[&[1, 0], &[2, 0]])).is_err());
    assert!(chi2_contributions(&counts(&[&[1, 2], &[3]])).is_err());
    assert!(standardized_residuals(&ctx, &counts(&[&[1, -2], &[3, 4]])).is_err());
    assert!(expected_counts(&Vec::<Vec<Q>>::new()).is_err());
    // A single row has expected counts (the row itself) but no adjustment factor.
    assert_eq!(
        expected_counts(&counts(&[&[3, 5]])).unwrap(),
        counts(&[&[3, 5]])
    );
    assert!(adjusted_residuals(&ctx, &counts(&[&[3, 5]])).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Correlation inference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fisher_z_is_atanh() {
    let ctx = Context::new();
    // math.atanh(0.8) = 1.098612288668110 (= ln 3)
    let z = fisher_z(&ctx.rational(4, 5));
    assert!(close(z.eval_f64().unwrap(), 1.098_612_288_668_11));
    assert!(close(
        fisher_z(&ctx.rational(-1, 2)).eval_f64().unwrap(),
        -0.5_f64.atanh()
    ));
    assert_eq!(fisher_z(&ctx.zero()).simplify(), ctx.zero());
}

#[test]
fn pearson_ci_matches_scipy() {
    // scipy: pearsonr(x, y).confidence_interval(0.95) with r = 13/15, n = 10
    //   → (0.521743144851242, 0.9680507713838036); 0.90 → (0.6029901323842667, 0.9596310194877663)
    let ci = pearson_ci(13.0 / 15.0, 10, 0.95).unwrap();
    assert!(close(ci.lower, 0.521_743_144_851_242));
    assert!(close(ci.upper, 0.968_050_771_383_803_6));
    let ci = pearson_ci(13.0 / 15.0, 10, 0.90).unwrap();
    assert!(close(ci.lower, 0.602_990_132_384_266_7));
    assert!(close(ci.upper, 0.959_631_019_487_766_3));
    // Hand Fisher z with scipy.stats.norm.isf: (0.5, 30, 0.99) → (0.053536328090218556, 0.7798645389691239)
    let ci = pearson_ci(0.5, 30, 0.99).unwrap();
    assert!(close(ci.lower, 0.053_536_328_090_218_556));
    assert!(close(ci.upper, 0.779_864_538_969_123_9));
    // (0.8, 10, 0.95) → (0.34328844799174646, 0.9507383861974117)
    let ci = pearson_ci(0.8, 10, 0.95).unwrap();
    assert!(close(ci.lower, 0.343_288_447_991_746_46));
    assert!(close(ci.upper, 0.950_738_386_197_411_7));
}

#[test]
fn pearson_ci_rejects_bad_input() {
    assert!(pearson_ci(1.0, 10, 0.95).is_err());
    assert!(pearson_ci(-1.0, 10, 0.95).is_err());
    assert!(pearson_ci(f64::NAN, 10, 0.95).is_err());
    assert!(pearson_ci(0.5, 3, 0.95).is_err());
    assert!(pearson_ci(0.5, 10, 1.0).is_err());
    assert!(pearson_ci(0.5, 10, 0.0).is_err());
}

#[test]
fn pearson_test_two_sided_and_one_sided() {
    let ctx = Context::new();
    let (x, y) = xy10();
    // scipy: pearsonr(x, y) → statistic 0.866666666666666 (= 13/15), pvalue 0.001173538180155;
    //   alternative='less' 0.999413230909922, 'greater' 0.000586769090078
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(13, 15)));
    assert_eq!(r.df, Some(ctx.int(8)));
    assert_eq!(r.alternative, Alternative::TwoSided);
    assert!(close(r.p_value_f64().unwrap(), 0.001_173_538_180_155));
    let l = pearson_test(&ctx, &x, &y, Alternative::Less).unwrap();
    assert!(close(l.p_value_f64().unwrap(), 0.999_413_230_909_922));
    let g = pearson_test(&ctx, &x, &y, Alternative::Greater).unwrap();
    assert!(close(g.p_value_f64().unwrap(), 0.000_586_769_090_078));
}

#[test]
fn pearson_test_rational_r_and_perfect_correlation() {
    let ctx = Context::new();
    // scipy: pearsonr([1..5], [1, 3, 2, 5, 4]) → statistic 0.8, pvalue 0.104088038661828
    let x = from_i64(&[1, 2, 3, 4, 5]);
    let y = from_i64(&[1, 3, 2, 5, 4]);
    let r = pearson_test(&ctx, &x, &y, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.from_ratio(q(4, 5)));
    assert!(close(r.p_value_f64().unwrap(), 0.104_088_038_661_828));
    // |r| = 1: p = 0 in the direction of r, 1 against it.
    let y2 = from_i64(&[3, 5, 7, 9, 11]);
    let r = pearson_test(&ctx, &x, &y2, Alternative::TwoSided).unwrap();
    assert_eq!(r.statistic, ctx.one());
    assert_eq!(r.p_value_exact(), Some(qi(0)));
    let r = pearson_test(&ctx, &x, &y2, Alternative::Less).unwrap();
    assert_eq!(r.p_value_exact(), Some(qi(1)));
    // Errors: unequal lengths, too few pairs, constant sample.
    assert!(pearson_test(&ctx, &x, &from_i64(&[1, 2]), Alternative::TwoSided).is_err());
    assert!(
        pearson_test(
            &ctx,
            &from_i64(&[1, 2]),
            &from_i64(&[2, 1]),
            Alternative::TwoSided
        )
        .is_err()
    );
    assert!(pearson_test(&ctx, &x, &from_i64(&[1, 1, 1, 1, 1]), Alternative::TwoSided).is_err());
}

#[test]
fn pearson_t_statistic_has_a_rational_square() {
    let ctx = Context::new();
    let (x, y) = xy10();
    // Fraction: sxy = 143/2, sxx = syy = 165/2 (population sums), t² = r²(n−2)/(1−r²) = 169/7;
    // t = 4.913538149119947.
    let t = pearson_t_statistic(&ctx, &x, &y).unwrap();
    assert!(close(t.eval_f64().unwrap(), 4.913_538_149_119_947));
    assert_eq!((&t * &t).simplify(), ctx.rational(169, 7));
    // |r| = 1 is infinite.
    assert!(pearson_t_statistic(&ctx, &x, &x).is_err());
}

#[test]
fn compare_two_correlations_by_fisher_z() {
    let ctx = Context::new();
    // scipy.stats.norm on z = (atanh .7 − atanh .4)/√(1/47 + 1/57) = 2.251706268343729:
    //   two-sided 0.024340840282246236, greater 0.012170420141123118, less 0.9878295798588769
    let r = compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::TwoSided).unwrap();
    assert!(close(r.statistic_f64().unwrap(), 2.251_706_268_343_729));
    assert!(close(r.p_value_f64().unwrap(), 0.024_340_840_282_246_236));
    assert_eq!(r.df, None);
    let g = compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::Greater).unwrap();
    assert!(close(g.p_value_f64().unwrap(), 0.012_170_420_141_123_118));
    let l = compare_two_correlations(&ctx, 0.7, 50, 0.4, 60, Alternative::Less).unwrap();
    assert!(close(l.p_value_f64().unwrap(), 0.987_829_579_858_876_9));
    // Negative z: (0.3, 40, 0.5, 40) → z −1.0313609064325715, two-sided 0.3023716063689633,
    //   greater 0.8488141968155183, less 0.15118580318448166
    let r = compare_two_correlations(&ctx, 0.3, 40, 0.5, 40, Alternative::TwoSided).unwrap();
    assert!(close(r.statistic_f64().unwrap(), -1.031_360_906_432_571_5));
    assert!(close(r.p_value_f64().unwrap(), 0.302_371_606_368_963_3));
    let g = compare_two_correlations(&ctx, 0.3, 40, 0.5, 40, Alternative::Greater).unwrap();
    assert!(close(g.p_value_f64().unwrap(), 0.848_814_196_815_518_3));
    let l = compare_two_correlations(&ctx, 0.3, 40, 0.5, 40, Alternative::Less).unwrap();
    assert!(close(l.p_value_f64().unwrap(), 0.151_185_803_184_481_66));
}

#[test]
fn compare_two_correlations_rejects_bad_input() {
    let ctx = Context::new();
    assert!(compare_two_correlations(&ctx, 1.0, 50, 0.4, 60, Alternative::TwoSided).is_err());
    assert!(compare_two_correlations(&ctx, 0.7, 3, 0.4, 60, Alternative::TwoSided).is_err());
    assert!(
        compare_two_correlations(&ctx, 0.7, 50, f64::INFINITY, 60, Alternative::TwoSided).is_err()
    );
    assert!(compare_two_correlations(&ctx, 0.7, 50, 0.4, 2, Alternative::TwoSided).is_err());
}
