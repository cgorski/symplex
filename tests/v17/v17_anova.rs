//! symplex 0.17 — `stats::anova`: two-way and repeated-measures ANOVA, the
//! studentized range distribution and post-hoc comparisons.
//!
//! Reference values cite statsmodels 0.15 / scipy 1.18 / pingouin 0.6
//! (`symplex/.venv/bin/python`).  Exact rationals were derived with
//! `fractions.Fraction` arithmetic on the integer data in the same scripts
//! (Gauss–Jordan on the normal equations of the dummy-coded designs for the
//! two-way sums of squares, the double-centred covariance for `ε`, the sum
//! of principal minors for Mauchly's `W`) and `float(frac)` was checked
//! against the statsmodels / pingouin float to ≤ 1e-9 before being written
//! here.
//!
//! Datasets:
//! * `BAL`: balanced 2 × 3 design, 3 replicates per cell (N = 18).
//! * `UNB`: unbalanced 2 × 3 design, cell sizes 4, 2, 3 / 2, 4, 3 (N = 18).
//! * `RM3`: 5 subjects × 3 conditions; `RM4`: 6 subjects × 4 conditions.
//! * `G3`: the three groups of the `anova_one_way` doc example (n = 6 each);
//!   `G3U`: three groups of sizes 4, 3, 5.

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::anova::{
    Adjustment, Observation, Source, SsType, TwoWayData, anova_one_way, anova_repeated_measures,
    anova_two_way, anova_two_way_with, pairwise_t_tests, studentized_range_cdf,
    studentized_range_quantile, studentized_range_sf, tukey_hsd,
};
use symplex::stats::data::from_i64;

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn ev(e: &Ex) -> f64 {
    e.eval_f64().unwrap()
}

fn qf(x: &Q) -> f64 {
    symplex::stats::data::to_f64(std::slice::from_ref(x))[0]
}

// Long-form frame used for every statsmodels call on BAL:
//   df = pd.DataFrame({'A': [0]*9 + [1]*9,
//                      'B': [0,0,0,1,1,1,2,2,2]*2,
//                      'y': [4,5,6,6,7,8,9,10,12, 5,5,7,8,9,11,13,14,16]})
//   m = ols('y ~ C(A) * C(B)', df).fit()
fn bal() -> TwoWayData {
    TwoWayData::from_i64(&[
        &[&[4, 5, 6], &[6, 7, 8], &[9, 10, 12]],
        &[&[5, 5, 7], &[8, 9, 11], &[13, 14, 16]],
    ])
    .unwrap()
}

// UNB long form:
//   A: [0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1]
//   B: [0,0,0,0,1,1,2,2,2, 0,0,1,1,1,1,2,2,2]
//   y: [4,5,6,7,6,8,9,10,12, 5,7,8,9,11,10,13,14,16]
fn unb() -> TwoWayData {
    TwoWayData::from_i64(&[
        &[&[4, 5, 6, 7], &[6, 8], &[9, 10, 12]],
        &[&[5, 7], &[8, 9, 11, 10], &[13, 14, 16]],
    ])
    .unwrap()
}

// RM3 long form: subject i, cond j, y = RM3[i][j];  wide = pd.DataFrame(RM3, columns=['c0','c1','c2'])
fn rm3() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[5, 7, 9]),
        from_i64(&[4, 5, 8]),
        from_i64(&[6, 8, 10]),
        from_i64(&[3, 6, 4]),
        from_i64(&[7, 9, 13]),
    ]
}

fn rm4() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[3, 5, 6, 8]),
        from_i64(&[2, 4, 4, 7]),
        from_i64(&[5, 6, 8, 9]),
        from_i64(&[4, 4, 7, 10]),
        from_i64(&[3, 6, 5, 9]),
        from_i64(&[6, 7, 9, 12]),
    ]
}

fn g3() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[6, 8, 4, 5, 3, 4]),
        from_i64(&[8, 12, 9, 11, 6, 8]),
        from_i64(&[13, 9, 11, 8, 7, 12]),
    ]
}

fn g3u() -> Vec<Vec<Q>> {
    vec![
        from_i64(&[4, 5, 6, 7]),
        from_i64(&[6, 8, 9]),
        from_i64(&[9, 10, 12, 11, 13]),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-way ANOVA — balanced (BAL)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_way_balanced_sums_of_squares_and_df_exact() {
    // statsmodels: anova_lm(m, typ=2)
    //   C(A)       sum_sq 24.49999999999974   df 1
    //   C(B)       sum_sq 148.77777777777786  df 2
    //   C(A):C(B)  sum_sq 8.333333333333293   df 2
    //   Residual   sum_sq 20.666666666666668  df 12
    // Fraction: SS_total = 3641/18 = 202.27777777777777
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    assert_eq!(r.ss_type, SsType::TypeII);
    assert_eq!((r.factor_a.ss.clone(), r.factor_a.df), (q(49, 2), 1));
    assert_eq!((r.factor_b.ss.clone(), r.factor_b.df), (q(1339, 9), 2));
    assert_eq!((r.interaction.ss.clone(), r.interaction.df), (q(25, 3), 2));
    assert_eq!((r.residual.ss.clone(), r.residual.df), (q(62, 3), 12));
    assert_eq!((r.total.ss.clone(), r.total.df), (q(3641, 18), 17));
    close(qf(&r.factor_b.ss), 148.777_777_777_777_86, 1e-12);
}

#[test]
fn two_way_balanced_f_statistics_and_p_values() {
    // statsmodels: anova_lm(m, typ=2)
    //   C(A)       F 14.225806451612754  PR(>F) 0.00266347768868363
    //   C(B)       F 43.19354838709679   PR(>F) 3.29199072704e-06
    //   C(A):C(B)  F 2.4193548387096655  PR(>F) 0.13098893805732365
    // scipy: f.sf(441/31, 1, 12) = 0.0026634776886835334, f.sf(1339/31, 2, 12) = 3.291990727040258e-06,
    //        f.sf(75/31, 2, 12) = 0.13098893805732253
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    assert_eq!(r.factor_a.f, Some(q(441, 31)));
    assert_eq!(r.factor_b.f, Some(q(1339, 31)));
    assert_eq!(r.interaction.f, Some(q(75, 31)));
    close(
        r.factor_a.p_value_f64().unwrap(),
        0.002_663_477_688_683_533_4,
        1e-12,
    );
    close(
        r.factor_b.p_value_f64().unwrap(),
        3.291_990_727_040_258e-6,
        1e-12,
    );
    close(
        r.interaction.p_value_f64().unwrap(),
        0.130_988_938_057_322_53,
        1e-12,
    );
    close(
        qf(r.factor_a.f.as_ref().unwrap()),
        14.225_806_451_612_904,
        1e-12,
    );
}

#[test]
fn two_way_balanced_all_ss_types_coincide() {
    // statsmodels: anova_lm(m, typ=1), typ=2 and anova_lm(ols('y ~ C(A, Sum) * C(B, Sum)').fit(), typ=3)
    // all print sum_sq 24.5 / 148.777… / 8.333… for BAL.
    let ctx = Context::new();
    let t2 = anova_two_way_with(&ctx, &bal(), SsType::TypeII).unwrap();
    for t in [SsType::TypeI, SsType::TypeIII] {
        let r = anova_two_way_with(&ctx, &bal(), t).unwrap();
        assert_eq!(r.ss_type, t);
        assert_eq!(r.factor_a.ss, t2.factor_a.ss);
        assert_eq!(r.factor_b.ss, t2.factor_b.ss);
        assert_eq!(r.interaction.ss, t2.interaction.ss);
        assert_eq!(r.factor_a.f, t2.factor_a.f);
        assert_eq!(r.factor_b.f, t2.factor_b.f);
        assert_eq!(r.residual, t2.residual);
    }
}

#[test]
fn two_way_balanced_effect_sizes_exact() {
    // Fraction: partial η² = SS/(SS + SS_resid): A 147/271 (0.5424354243542435),
    //   B 1339/1525 (0.8780327868852459), A×B 25/87 (0.28735632183908044);
    //   η² = SS/SS_total: A 441/3641, B 2678/3641, A×B 150/3641.
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    assert_eq!(r.factor_a.partial_eta_squared, Some(q(147, 271)));
    assert_eq!(r.factor_b.partial_eta_squared, Some(q(1339, 1525)));
    assert_eq!(r.interaction.partial_eta_squared, Some(q(25, 87)));
    assert_eq!(r.factor_a.eta_squared, Some(q(441, 3641)));
    assert_eq!(r.factor_b.eta_squared, Some(q(2678, 3641)));
    assert_eq!(r.interaction.eta_squared, Some(q(150, 3641)));
    close(
        qf(r.factor_b.partial_eta_squared.as_ref().unwrap()),
        0.878_032_786_885_245_9,
        1e-15,
    );
    // The residual and total rows carry no effect size.
    assert_eq!(r.residual.partial_eta_squared, None);
    assert_eq!(r.total.eta_squared, None);
}

#[test]
fn two_way_balanced_means_exact() {
    // Fraction: grand mean 155/18; cell means [[5, 7, 31/3], [17/3, 28/3, 43/3]]
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    assert_eq!(r.grand_mean, q(155, 18));
    assert_eq!(
        r.cell_means,
        vec![
            vec![qi(5), qi(7), q(31, 3)],
            vec![q(17, 3), q(28, 3), q(43, 3)]
        ]
    );
}

#[test]
fn two_way_rows_in_table_order_and_untested_rows() {
    // Fraction: MS_resid = (62/3)/12 = 31/18 (statsmodels mse_resid 1.7222222222222223)
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    let sources: Vec<Source> = r.rows().iter().map(|row| row.source).collect();
    assert_eq!(
        sources,
        vec![
            Source::FactorA,
            Source::FactorB,
            Source::Interaction,
            Source::Residual,
            Source::Total
        ]
    );
    assert_eq!(r.residual.ms, Some(q(31, 18)));
    assert_eq!(
        (r.residual.f.clone(), r.residual.p_value.clone()),
        (None, None)
    );
    assert_eq!((r.total.ms.clone(), r.total.f.clone()), (None, None));
    assert!(matches!(
        r.residual.p_value_f64(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert_eq!(r.factor_a.ms, Some(q(49, 2)));
    assert_eq!(r.factor_b.ms, Some(q(1339, 18)));
}

#[test]
fn two_way_balanced_sums_of_squares_decompose_total() {
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &bal()).unwrap();
    let sum = &r.factor_a.ss + &r.factor_b.ss + &r.interaction.ss + &r.residual.ss;
    assert_eq!(sum, r.total.ss);
    assert_eq!(
        r.factor_a.df + r.factor_b.df + r.interaction.df + r.residual.df,
        r.total.df
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-way ANOVA — unbalanced (UNB)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_way_unbalanced_type1_sequential() {
    // statsmodels: anova_lm(m, typ=1)
    //   C(A)       sum_sq 37.555555555555636  F 19.314285714285752  PR(>F) 0.00087331758917192
    //   C(B)       sum_sq 120.22222222222233  F 30.91428571428574   PR(>F) 1.843914104950e-05
    //   C(A):C(B)  sum_sq 8.666666666666679   F 2.228571428571431   PR(>F) 0.15030064414394545
    //   Residual   sum_sq 23.333333333333336  df 12
    // scipy: f.sf(676/35, 1, 12) = 0.0008733175891719245, f.sf(1082/35, 2, 12) = 1.84391410494999e-05,
    //        f.sf(78/35, 2, 12) = 0.15030064414394567
    let ctx = Context::new();
    let r = anova_two_way_with(&ctx, &unb(), SsType::TypeI).unwrap();
    assert!(!unb().is_balanced());
    assert_eq!(r.factor_a.ss, q(338, 9));
    assert_eq!(r.factor_a.f, Some(q(676, 35)));
    assert_eq!(r.factor_b.ss, q(1082, 9));
    assert_eq!(r.factor_b.f, Some(q(1082, 35)));
    assert_eq!(r.interaction.ss, q(26, 3));
    assert_eq!(r.interaction.f, Some(q(78, 35)));
    assert_eq!((r.residual.ss.clone(), r.residual.df), (q(70, 3), 12));
    close(
        r.factor_a.p_value_f64().unwrap(),
        0.000_873_317_589_171_924_5,
        1e-12,
    );
    close(
        r.factor_b.p_value_f64().unwrap(),
        1.843_914_104_949_99e-5,
        1e-12,
    );
    close(
        r.interaction.p_value_f64().unwrap(),
        0.150_300_644_143_945_67,
        1e-12,
    );
}

#[test]
fn two_way_unbalanced_type2_hierarchical() {
    // statsmodels: anova_lm(m, typ=2)
    //   C(A)       sum_sq 24.000000000000096  F 12.34285714285719   PR(>F) 0.00427631711567724
    //   C(B)       sum_sq 120.22222222222244  F 30.914285714285768  PR(>F) 1.843914104950e-05
    //   C(A):C(B)  sum_sq 8.666666666666684   F 2.2285714285714326  PR(>F) 0.15030064414394528
    // scipy: f.sf(432/35, 1, 12) = 0.004276317115677296
    // Fraction: partial η²(A) = 24/(24 + 70/3) = 36/71
    let ctx = Context::new();
    let r = anova_two_way(&ctx, &unb()).unwrap();
    assert_eq!(r.factor_a.ss, qi(24));
    assert_eq!(r.factor_a.f, Some(q(432, 35)));
    close(
        r.factor_a.p_value_f64().unwrap(),
        0.004_276_317_115_677_296,
        1e-12,
    );
    // B adjusted for A is the same under Type I and II (B comes second in the sequence).
    assert_eq!(r.factor_b.ss, q(1082, 9));
    assert_eq!(r.interaction.ss, q(26, 3));
    assert_eq!(r.factor_a.partial_eta_squared, Some(q(36, 71)));
    assert_eq!(r.factor_a.eta_squared, Some(q(54, 427)));
}

#[test]
fn two_way_unbalanced_type3_sum_to_zero_contrasts() {
    // statsmodels: anova_lm(ols('y ~ C(A, Sum) * C(B, Sum)', df).fit(), typ=3)
    //   C(A, Sum)            sum_sq 22.61538461538463   F 11.630769230769239  PR(>F) 0.00516950674476044
    //   C(B, Sum)            sum_sq 125.89333333333337  F 32.372571428571433  PR(>F) 1.461443679018e-05
    //   C(A, Sum):C(B, Sum)  sum_sq 8.666666666666663   F 2.2285714285714273  PR(>F) 0.15030064414394587
    // Fraction (drop-term RSS differences under sum coding): A 294/13, B 9442/75, AB 26/3
    // scipy: f.sf(756/65, 1, 12) = 0.005169506744760451, f.sf(28326/875, 2, 12) = 1.4614436790178848e-05
    let ctx = Context::new();
    let r = anova_two_way_with(&ctx, &unb(), SsType::TypeIII).unwrap();
    assert_eq!(r.factor_a.ss, q(294, 13));
    assert_eq!(r.factor_a.f, Some(q(756, 65)));
    assert_eq!(r.factor_b.ss, q(9442, 75));
    assert_eq!(r.factor_b.f, Some(q(28326, 875)));
    assert_eq!(r.interaction.ss, q(26, 3));
    assert_eq!(r.interaction.f, Some(q(78, 35)));
    close(
        r.factor_a.p_value_f64().unwrap(),
        0.005_169_506_744_760_451,
        1e-12,
    );
    close(
        r.factor_b.p_value_f64().unwrap(),
        1.461_443_679_017_884_8e-5,
        1e-12,
    );
    close(qf(&r.factor_b.ss), 125.893_333_333_333_37, 1e-12);
    assert_eq!(r.factor_a.partial_eta_squared, Some(q(63, 128)));
}

#[test]
fn two_way_unbalanced_type1_a_is_the_one_way_anova_on_a() {
    // Type I SS(A) is the between-groups SS of the one-way ANOVA on A alone.
    // scipy: f_oneway(A0, A1) on the pooled levels — ss_between = 338/9 by Fraction.
    let ctx = Context::new();
    let r = anova_two_way_with(&ctx, &unb(), SsType::TypeI).unwrap();
    let data = unb();
    let level = |a: usize| -> Vec<Q> {
        (0..data.b_levels())
            .flat_map(|b| data.cell(a, b).unwrap().to_vec())
            .collect()
    };
    let one_way = anova_one_way(&ctx, &[level(0), level(1)]).unwrap();
    assert_eq!(one_way.ss_between, r.factor_a.ss);
    assert_eq!(one_way.ss_between, q(338, 9));
}

#[test]
fn two_way_unbalanced_only_type1_decomposes_total() {
    // Fraction: 338/9 + 1082/9 + 26/3 + 70/3 = 1708/9 = SS_total;
    //           Type II: 24 + 1082/9 + 26/3 + 70/3 = 1586/9 ≠ 1708/9.
    let ctx = Context::new();
    let t1 = anova_two_way_with(&ctx, &unb(), SsType::TypeI).unwrap();
    let t2 = anova_two_way_with(&ctx, &unb(), SsType::TypeII).unwrap();
    let sum = |r: &symplex::stats::anova::TwoWayAnova| {
        &r.factor_a.ss + &r.factor_b.ss + &r.interaction.ss + &r.residual.ss
    };
    assert_eq!(t1.total.ss, q(1708, 9));
    assert_eq!(sum(&t1), t1.total.ss);
    assert_eq!(sum(&t2), q(1586, 9));
    assert_ne!(sum(&t2), t2.total.ss);
}

#[test]
fn two_way_unbalanced_residual_and_means() {
    // Fraction: residual = pooled within-cell SS = 70/3 (statsmodels 23.333333333333336), MS 35/18;
    //   grand mean 80/9; cell means [[11/2, 7, 31/3], [6, 19/2, 43/3]]; N = 18
    let ctx = Context::new();
    let data = unb();
    assert_eq!(data.n_obs(), 18);
    assert_eq!((data.a_levels(), data.b_levels()), (2, 3));
    let r = anova_two_way(&ctx, &data).unwrap();
    assert_eq!(r.residual.ss, q(70, 3));
    assert_eq!(r.residual.ms, Some(q(35, 18)));
    assert_eq!(r.grand_mean, q(80, 9));
    assert_eq!(
        r.cell_means,
        vec![
            vec![q(11, 2), qi(7), q(31, 3)],
            vec![qi(6), q(19, 2), q(43, 3)]
        ]
    );
    assert_eq!(r.total.df, 17);
}

#[test]
fn two_way_from_long_matches_from_cells() {
    let ctx = Context::new();
    let mut rows = Vec::new();
    for (a, row) in unb().cells().iter().enumerate() {
        for (b, cell) in row.iter().enumerate() {
            for y in cell {
                rows.push(Observation { a, b, y: y.clone() });
            }
        }
    }
    // Interleave the two A levels: long form need not be grouped by cell
    // (the order of replicates within a cell is preserved).
    let (first, second) = rows.split_at(9);
    let interleaved: Vec<Observation> = first
        .iter()
        .zip(second)
        .flat_map(|(x, y)| [x.clone(), y.clone()])
        .collect();
    let long = TwoWayData::from_long(&interleaved).unwrap();
    assert_eq!(long, unb());
    assert_eq!(long.cell(0, 1), Some(&[qi(6), qi(8)][..]));
    assert_eq!(long.cell(2, 0), None);
    assert_eq!(long.cell(0, 3), None);
    assert_eq!(
        anova_two_way(&ctx, &long).unwrap(),
        anova_two_way(&ctx, &unb()).unwrap()
    );
}

#[test]
fn two_way_invalid_inputs() {
    let ctx = Context::new();
    let invalid = |r: Result<TwoWayData, SymplexError>| {
        assert!(matches!(r, Err(SymplexError::InvalidArgument { .. })));
    };
    // One level of A.
    invalid(TwoWayData::from_i64(&[&[&[1, 2], &[3, 4]]]));
    // One level of B.
    invalid(TwoWayData::from_i64(&[&[&[1, 2]], &[&[3, 4]]]));
    // Ragged rows.
    invalid(TwoWayData::from_i64(&[&[&[1, 2], &[3, 4]], &[&[5, 6]]]));
    // Empty cell.
    invalid(TwoWayData::from_i64(&[&[&[1, 2], &[]], &[&[5, 6], &[7]]]));
    // Missing combination in long form.
    invalid(TwoWayData::from_long(&[
        Observation {
            a: 0,
            b: 0,
            y: qi(1),
        },
        Observation {
            a: 1,
            b: 1,
            y: qi(2),
        },
    ]));
    invalid(TwoWayData::from_long(&[]));
    // One observation per cell: no residual df.
    let saturated = TwoWayData::from_i64(&[&[&[1], &[2]], &[&[3], &[5]]]).unwrap();
    assert!(matches!(
        anova_two_way(&ctx, &saturated),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Constant response.
    let constant = TwoWayData::from_i64(&[&[&[2, 2], &[2, 2]], &[&[2, 2], &[2, 2]]]).unwrap();
    assert!(matches!(
        anova_two_way(&ctx, &constant),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Cells constant but different: zero within-cell variance.
    let zero_resid = TwoWayData::from_i64(&[&[&[1, 1], &[2, 2]], &[&[3, 3], &[5, 5]]]).unwrap();
    assert!(matches!(
        anova_two_way(&ctx, &zero_resid),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Repeated measures (RM3, RM4)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rm_sums_of_squares_and_df_exact() {
    // pingouin: rm_anova(data=long, dv='y', within='cond', subject='subject', detailed=True)
    //   cond  SS 36.13333333333335  DF 2  MS 18.066666666666674
    //   Error SS 11.866666666666674 DF 8  MS 1.4833333333333343
    // Fraction: SS_cond 542/15, SS_subj 764/15, SS_err 178/15, SS_total 1484/15; MS 271/15, 191/15, 89/60
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    assert_eq!((r.n_subjects, r.n_conditions), (5, 3));
    assert_eq!((r.conditions.ss.clone(), r.conditions.df), (q(542, 15), 2));
    assert_eq!((r.subjects.ss.clone(), r.subjects.df), (q(764, 15), 4));
    assert_eq!((r.error.ss.clone(), r.error.df), (q(178, 15), 8));
    assert_eq!((r.total.ss.clone(), r.total.df), (q(1484, 15), 14));
    assert_eq!(r.conditions.ms, Some(q(271, 15)));
    assert_eq!(r.subjects.ms, Some(q(191, 15)));
    assert_eq!(r.error.ms, Some(q(89, 60)));
    assert_eq!(&r.conditions.ss + &r.subjects.ss + &r.error.ss, r.total.ss);
    let sources: Vec<Source> = r.rows().iter().map(|row| row.source).collect();
    assert_eq!(
        sources,
        vec![
            Source::Conditions,
            Source::Subjects,
            Source::Residual,
            Source::Total
        ]
    );
}

#[test]
fn rm_f_statistic_and_p_value() {
    // statsmodels: AnovaRM(long, 'y', 'subject', within=['cond']).fit().anova_table
    //   F Value 12.179775280898882  Num DF 2.0  Den DF 8.0  Pr > F 0.0037355110334743136
    // Fraction: F = (271/15)/(89/60) = 1084/89; scipy f.sf(1084/89, 2, 8) = 0.003735511033474317
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    assert_eq!(r.f, q(1084, 89));
    assert_eq!(r.conditions.f, Some(q(1084, 89)));
    close(qf(&r.f), 12.179_775_280_898_882, 1e-12);
    close(r.p_value_f64().unwrap(), 0.003_735_511_033_474_317, 1e-12);
    close(
        r.conditions.p_value_f64().unwrap(),
        0.003_735_511_033_474_313_6,
        1e-12,
    );
    assert_eq!((r.subjects.f.clone(), r.error.f.clone()), (None, None));
}

#[test]
fn rm_effect_sizes_and_means_exact() {
    // Fraction: partial η² = (542/15)/(542/15 + 178/15) = 271/360 (0.7527777777777778);
    //   SS_cond/SS_total = 271/742 — pingouin ng2 0.36522911051212953;
    //   grand mean 104/15; condition means [5, 7, 44/5]; subject means [7, 17/3, 8, 13/3, 29/3]
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    assert_eq!(r.conditions.partial_eta_squared, Some(q(271, 360)));
    assert_eq!(r.conditions.eta_squared, Some(q(271, 742)));
    close(
        qf(r.conditions.eta_squared.as_ref().unwrap()),
        0.365_229_110_512_129_53,
        1e-15,
    );
    assert_eq!(r.grand_mean, q(104, 15));
    assert_eq!(r.condition_means, vec![qi(5), qi(7), q(44, 5)]);
    assert_eq!(
        r.subject_means,
        vec![qi(7), q(17, 3), qi(8), q(13, 3), q(29, 3)]
    );
}

#[test]
fn rm_greenhouse_geisser_epsilon_exact() {
    // pingouin: epsilon(wide, correction='gg') = 0.5426457491265326
    // numpy: eigvalsh of the double-centred np.cov → (Σλ)²/(2 Σλ²) = 0.5426457491265329
    // Fraction: (tr S̃)² / (2 tr S̃²) = 7921/14597
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    assert_eq!(r.epsilon_gg, q(7921, 14597));
    close(qf(&r.epsilon_gg), 0.542_645_749_126_532_6, 1e-12);
}

#[test]
fn rm_huynh_feldt_epsilon_exact() {
    // pingouin: epsilon(wide, correction='hf') = 0.5877873360597936
    // Fraction: (5·2·ε̂ − 2) / (2·(4 − 2ε̂)) with ε̂ = 7921/14597 → 4168/7091
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    assert_eq!(r.epsilon_hf, Some(q(4168, 7091)));
    close(
        qf(r.epsilon_hf.as_ref().unwrap()),
        0.587_787_336_059_793_6,
        1e-12,
    );
}

#[test]
fn rm_sphericity_corrected_p_values() {
    // pingouin: rm_anova(..., correction=True).p_GG_corr = 0.02126436585826144
    // scipy: f.sf(1084/89, 2·7921/14597, 8·7921/14597) = 0.021264365858261566
    //        f.sf(1084/89, 2·4168/7091, 8·4168/7091) = 0.01784225302651239   (Huynh–Feldt, ε̃ < 1 so uncapped)
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    close(
        r.p_value_gg_f64().unwrap(),
        0.021_264_365_858_261_566,
        1e-12,
    );
    close(r.p_value_gg_f64().unwrap(), 0.021_264_365_858_261_44, 1e-12);
    close(
        r.p_value_hf_f64().unwrap().unwrap(),
        0.017_842_253_026_512_39,
        1e-12,
    );
    // The correction can only make the test more conservative.
    assert!(r.p_value_gg_f64().unwrap() > r.p_value_f64().unwrap());
}

#[test]
fn rm_mauchly_three_conditions() {
    // pingouin: sphericity(wide) → W 0.1571771241004926, chi2 5.551145791696411, dof 2, pval 0.062313767163236534
    // Fraction: W = (product of the two non-zero eigenvalues of S̃ = 83/240) / (tr S̃ / 2)² = 1245/7921
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm3()).unwrap();
    let m = r.mauchly.as_ref().unwrap();
    assert_eq!(m.w, q(1245, 7921));
    assert_eq!(m.df, 2);
    close(m.chi_squared_f64().unwrap(), 5.551_145_791_696_411, 1e-12);
    close(m.p_value_f64().unwrap(), 0.062_313_767_163_236_534, 1e-12);
}

#[test]
fn rm_four_conditions_epsilons_mauchly_and_corrected_p() {
    // statsmodels: AnovaRM → F Value 52.0142180094787, Num DF 3, Den DF 15, Pr > F 3.6786531334646144e-08
    // Fraction: SS_cond 2195/24, SS_subj 1001/24, SS_err 211/24, F = 10975/211
    // pingouin: epsilon(wide, 'gg') = 0.6115942028985543 (Fraction 211/345),
    //           epsilon(wide, 'hf') = 0.9487179487179586 (Fraction 37/39),
    //           rm_anova(correction=True).p_GG_corr = 1.182595561866e-05,
    //           sphericity(wide) → W 0.10429882867992099 (Fraction 979776/9393931), chi2 8.414065270641222, dof 5
    // scipy: f.sf(10975/211, 3·211/345, 15·211/345) = 1.1825955618663876e-05,
    //        f.sf(10975/211, 3·37/39, 15·37/39) = 7.849523323673606e-08
    // Mauchly p with Box's second-order term as in R's mauchly.test
    //   (ω₂ = (d+2)(d−1)(d−2)(2d³+6d²+3d+2)/(288((n−1)dρ)²), d = 3, n = 6, ρ = 67/90):
    //   scipy chi2.sf(8.414065270641217, 5) = 0.13484392499616613, chi2.sf(·, 9) → p = 0.1467125103668793.
    //   (pingouin 0.6.1 prints 0.14701171840143473: its ω₂ has `3k + 2` where Box and R have `3d + 2`.)
    let ctx = Context::new();
    let r = anova_repeated_measures(&ctx, &rm4()).unwrap();
    assert_eq!((r.n_subjects, r.n_conditions), (6, 4));
    assert_eq!(r.conditions.ss, q(2195, 24));
    assert_eq!(r.subjects.ss, q(1001, 24));
    assert_eq!(r.error.ss, q(211, 24));
    assert_eq!((r.conditions.df, r.error.df), (3, 15));
    assert_eq!(r.f, q(10975, 211));
    close(r.p_value_f64().unwrap(), 3.678_653_133_464_614_4e-8, 1e-15);
    assert_eq!(r.epsilon_gg, q(211, 345));
    assert_eq!(r.epsilon_hf, Some(q(37, 39)));
    close(
        r.p_value_gg_f64().unwrap(),
        1.182_595_561_866_387_6e-5,
        1e-13,
    );
    close(
        r.p_value_hf_f64().unwrap().unwrap(),
        7.849_523_323_673_606e-8,
        1e-15,
    );
    let m = r.mauchly.as_ref().unwrap();
    assert_eq!(m.w, q(979_776, 9_393_931));
    assert_eq!(m.df, 5);
    close(m.chi_squared_f64().unwrap(), 8.414_065_270_641_222, 1e-12);
    close(m.p_value_f64().unwrap(), 0.146_712_510_366_879_3, 1e-12);
}

#[test]
fn rm_two_conditions_is_the_squared_paired_t() {
    // statsmodels: AnovaRM on [[5,7],[4,5],[6,8],[3,6],[7,9]] → F Value 40.0, Num DF 1, Den DF 4,
    //   Pr > F 0.0031982021523353056
    // scipy: ttest_rel(c0, c1) → statistic -6.324555320336758 (t² = 40), pvalue 0.0031982021523353065
    // Fraction: SS_cond 10, SS_subj 19, SS_err 1
    let ctx = Context::new();
    let y = vec![
        from_i64(&[5, 7]),
        from_i64(&[4, 5]),
        from_i64(&[6, 8]),
        from_i64(&[3, 6]),
        from_i64(&[7, 9]),
    ];
    let r = anova_repeated_measures(&ctx, &y).unwrap();
    assert_eq!(r.f, qi(40));
    assert_eq!(
        (
            r.conditions.ss.clone(),
            r.subjects.ss.clone(),
            r.error.ss.clone()
        ),
        (qi(10), qi(19), qi(1))
    );
    close(r.p_value_f64().unwrap(), 0.003_198_202_152_335_305_6, 1e-12);
    // With two conditions sphericity is automatic: ε = 1 and no Mauchly test.
    assert_eq!(r.epsilon_gg, qi(1));
    assert!(r.mauchly.is_none());
    close(r.p_value_gg_f64().unwrap(), r.p_value_f64().unwrap(), 1e-15);
}

#[test]
fn rm_two_subjects_three_conditions_degenerate_corrections() {
    // statsmodels: AnovaRM on [[1,2,4],[2,5,3]] → F Value 1.333333333333333, Num DF 2, Den DF 2,
    //   Pr > F 0.4285714285714286
    // Fraction: SS_cond 16/3, SS_subj 3/2, SS_err 4, F = 4/3; P(F_{2,2} ≥ f) = 1/(1 + f) = 3/7
    // pingouin: epsilon(wide, 'gg') = 0.5 (the lower bound 1/(k−1): the covariance has rank 1),
    //           epsilon(wide, 'hf') = nan (zero denominator)
    let ctx = Context::new();
    let y = vec![from_i64(&[1, 2, 4]), from_i64(&[2, 5, 3])];
    let r = anova_repeated_measures(&ctx, &y).unwrap();
    assert_eq!(r.f, q(4, 3));
    assert_eq!((r.conditions.df, r.error.df), (2, 2));
    close(r.p_value_f64().unwrap(), 0.428_571_428_571_428_6, 1e-12);
    assert_eq!(r.epsilon_gg, q(1, 2));
    assert_eq!(r.epsilon_hf, None);
    assert_eq!(r.p_value_hf, None);
    assert_eq!(r.p_value_hf_f64().unwrap(), None);
    // n − 1 < k − 1: the covariance is singular, W = 0, Mauchly's test is undefined.
    assert!(r.mauchly.is_none());
}

#[test]
fn rm_invalid_inputs() {
    let ctx = Context::new();
    let invalid = |rows: Vec<Vec<Q>>| {
        assert!(matches!(
            anova_repeated_measures(&ctx, &rows),
            Err(SymplexError::InvalidArgument { .. })
        ));
    };
    invalid(vec![from_i64(&[1, 2, 3])]);
    invalid(vec![from_i64(&[1]), from_i64(&[2])]);
    invalid(vec![from_i64(&[1, 2]), from_i64(&[2, 3, 4])]);
    // Every subject's profile is a shift of the same curve: SS_error = 0.
    invalid(vec![
        from_i64(&[1, 3, 4]),
        from_i64(&[2, 4, 5]),
        from_i64(&[5, 7, 8]),
    ]);
    invalid(vec![]);
}

// ═══════════════════════════════════════════════════════════════════════════
// The studentized range distribution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn studentized_range_sf_matches_scipy() {
    // scipy: studentized_range.sf(q, k, df)
    for (qv, k, df, want) in [
        (3.0, 3, 12.0, 0.127_032_591_355_744_18),
        (3.5, 3, 15.0, 0.062_915_354_643_710_23),
        (1.0, 4, 10.0, 0.892_014_018_618_922_4),
        (5.0, 4, 10.0, 0.023_445_996_365_997_646),
        (2.0, 2, 5.0, 0.216_437_229_269_685_34),
        (4.2, 5, 30.0, 0.042_726_538_220_137_61),
        (3.0, 3, 200.0, 0.088_098_322_023_517_08),
    ] {
        close(studentized_range_sf(qv, k, df).unwrap(), want, 1e-9);
        close(studentized_range_cdf(qv, k, df).unwrap(), 1.0 - want, 1e-9);
    }
}

#[test]
fn studentized_range_quantile_matches_scipy() {
    // scipy: studentized_range.ppf(p, k, df)
    for (p, k, df, want) in [
        (0.95, 3, 15.0, 3.673_377_658_897_097_7),
        (0.95, 4, 10.0, 4.326_582_115_731_219),
        (0.99, 3, 12.0, 5.045_934_725_166_239),
        (0.9, 5, 20.0, 3.736_402_824_922_501),
    ] {
        let got = studentized_range_quantile(p, k, df).unwrap();
        close(got, want, 1e-7);
        // Round trip.
        close(studentized_range_cdf(got, k, df).unwrap(), p, 1e-9);
    }
}

#[test]
fn studentized_range_edge_cases_and_invalid_arguments() {
    assert_eq!(studentized_range_cdf(0.0, 3, 10.0).unwrap(), 0.0);
    assert_eq!(studentized_range_sf(-1.0, 3, 10.0).unwrap(), 1.0);
    let invalid = |r: Result<f64, SymplexError>| {
        assert!(matches!(r, Err(SymplexError::InvalidArgument { .. })));
    };
    invalid(studentized_range_cdf(3.0, 1, 10.0));
    invalid(studentized_range_cdf(3.0, 3, 0.5));
    invalid(studentized_range_cdf(f64::NAN, 3, 10.0));
    invalid(studentized_range_quantile(1.0, 3, 10.0));
    invalid(studentized_range_quantile(0.5, 3, f64::INFINITY));
    // Monotone in q.
    let a = studentized_range_cdf(2.0, 4, 8.0).unwrap();
    let b = studentized_range_cdf(3.0, 4, 8.0).unwrap();
    assert!(0.0 < a && a < b && b < 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Post-hoc comparisons
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tukey_hsd_balanced_matches_scipy() {
    // scipy: r = tukey_hsd(*G3)
    //   statistic[0,1] -4.0, [0,2] -5.0, [1,2] -1.0
    //   pvalue[0,1] 0.013913287267276697, [0,2] 0.0027321219673729358, [1,2] 0.700659938532346
    //   confidence_interval(0.95): low[0,1] -7.192998995879951 high[0,1] -0.8070010041200488,
    //                              low[1,2] -4.192998995879951 high[1,2] 2.192998995879951
    // Fraction: means [5, 9, 10], MSE = (68/15), se² = MSE/2·(1/6 + 1/6) = 34/45 (se 0.8692269873603532),
    //   q statistics 4.6017899330842225, 5.752237416355278, 1.1504474832710556; df = 15
    let ctx = Context::new();
    let pairs = tukey_hsd(&ctx, &g3(), 0.95).unwrap();
    assert_eq!(pairs.len(), 3);
    let idx: Vec<(usize, usize)> = pairs.iter().map(|p| (p.i, p.j)).collect();
    assert_eq!(idx, vec![(0, 1), (0, 2), (1, 2)]);
    assert_eq!(pairs[0].diff, qi(-4));
    assert_eq!(pairs[1].diff, qi(-5));
    assert_eq!(pairs[2].diff, qi(-1));
    let se = ctx.from_ratio(q(34, 45)).sqrt();
    for p in &pairs {
        assert_eq!(p.se, se);
    }
    close(ev(&pairs[0].se), 0.869_226_987_360_353_2, 1e-12);
    close(ev(&pairs[0].statistic), 4.601_789_933_084_222_5, 1e-12);
    close(ev(&pairs[1].statistic), 5.752_237_416_355_278, 1e-12);
    close(pairs[0].p_adj, 0.013_913_287_267_276_697, 1e-9);
    close(pairs[1].p_adj, 0.002_732_121_967_372_935_8, 1e-9);
    close(pairs[2].p_adj, 0.700_659_938_532_346, 1e-9);
    close(pairs[0].ci.lower, -7.192_998_995_879_951, 1e-8);
    close(pairs[0].ci.upper, -0.807_001_004_120_048_8, 1e-8);
    close(pairs[2].ci.lower, -4.192_998_995_879_951, 1e-8);
    close(pairs[2].ci.upper, 2.192_998_995_879_951, 1e-8);
    // The interval excludes 0 exactly when p_adj < 0.05.
    for p in &pairs {
        assert_eq!(p.ci.lower > 0.0 || p.ci.upper < 0.0, p.p_adj < 0.05);
    }
}

#[test]
fn tukey_hsd_unbalanced_tukey_kramer_matches_scipy() {
    // scipy: r = tukey_hsd(*G3U)
    //   pvalue[0,1] 0.18881130471561636, [0,2] 0.0009337796925320552, [1,2] 0.031533822027608016
    //   confidence_interval(0.95): [0,1] (-5.3189032700382155, 0.9855699367048816),
    //     [0,2] (-8.268641138063197, -2.7313588619368026), [1,2] (-6.347448030725932, -0.3192186359407341)
    //   confidence_interval(0.99): [0,1] (-6.500086708915748, 2.166753375582414)
    // Fraction: means [11/2, 23/3, 11], MSE 59/27, df 9; diffs -13/6, -11/2, -10/3;
    //   se² = MSE/2·(1/nᵢ + 1/nⱼ) = 413/648, 59/120, 236/405
    let ctx = Context::new();
    let pairs = tukey_hsd(&ctx, &g3u(), 0.95).unwrap();
    assert_eq!(pairs[0].diff, q(-13, 6));
    assert_eq!(pairs[1].diff, q(-11, 2));
    assert_eq!(pairs[2].diff, q(-10, 3));
    assert_eq!(pairs[0].se, ctx.from_ratio(q(413, 648)).sqrt());
    assert_eq!(pairs[1].se, ctx.from_ratio(q(59, 120)).sqrt());
    assert_eq!(pairs[2].se, ctx.from_ratio(q(236, 405)).sqrt());
    close(pairs[0].p_adj, 0.188_811_304_715_616_36, 1e-9);
    close(pairs[1].p_adj, 0.000_933_779_692_532_055_2, 1e-9);
    close(pairs[2].p_adj, 0.031_533_822_027_608_016, 1e-9);
    close(pairs[0].ci.lower, -5.318_903_270_038_215_5, 1e-8);
    close(pairs[0].ci.upper, 0.985_569_936_704_881_6, 1e-8);
    close(pairs[1].ci.lower, -8.268_641_138_063_197, 1e-8);
    close(pairs[1].ci.upper, -2.731_358_861_936_802_6, 1e-8);
    close(pairs[2].ci.lower, -6.347_448_030_725_932, 1e-8);
    close(pairs[2].ci.upper, -0.319_218_635_940_734_1, 1e-8);
    let wide = tukey_hsd(&ctx, &g3u(), 0.99).unwrap();
    close(wide[0].ci.lower, -6.500_086_708_915_748, 1e-8);
    close(wide[0].ci.upper, 2.166_753_375_582_414, 1e-8);
    // Same differences and p-values, wider intervals.
    assert_eq!(wide[0].diff, pairs[0].diff);
    close(wide[0].p_adj, pairs[0].p_adj, 1e-15);
}

#[test]
fn tukey_hsd_invalid_inputs() {
    let ctx = Context::new();
    let invalid = |groups: Vec<Vec<Q>>, confidence: f64| {
        assert!(matches!(
            tukey_hsd(&ctx, &groups, confidence),
            Err(SymplexError::InvalidArgument { .. })
        ));
    };
    invalid(vec![from_i64(&[1, 2, 3])], 0.95);
    invalid(vec![from_i64(&[1, 2, 3]), vec![]], 0.95);
    invalid(vec![from_i64(&[1]), from_i64(&[2])], 0.95);
    invalid(vec![from_i64(&[2, 2]), from_i64(&[3, 3])], 0.95);
    invalid(g3(), 1.0);
    invalid(g3(), 0.0);
}

#[test]
fn pairwise_welch_t_tests_holm_matches_scipy_and_statsmodels() {
    // scipy: ttest_ind(G3[i], G3[j], equal_var=False)
    //   (0,1) statistic -3.464101615137755  pvalue 0.00644386616395533   df 9.615384615384615
    //   (0,2) statistic -4.128614119223852  pvalue 0.002386749612426776  df 9.307692307692308
    //   (1,2) statistic -0.7595545253127499 pvalue 0.46515103975349575  df 9.941176470588234
    // Fraction: Welch df 125/13, 121/13, 169/17
    // statsmodels: multipletests(p, alpha=0.05, method='holm')
    //   → [0.01288773232791066, 0.007160248837280328, 0.46515103975349575], reject [True, True, False]
    let ctx = Context::new();
    let t = pairwise_t_tests(&ctx, &g3(), Adjustment::Holm, 0.05).unwrap();
    assert_eq!(t.len(), 3);
    assert_eq!((t[0].i, t[0].j, t[2].i, t[2].j), (0, 1, 1, 2));
    assert_eq!(t[0].diff, qi(-4));
    assert_eq!(t[1].diff, qi(-5));
    assert_eq!(t[2].diff, qi(-1));
    assert_eq!(t[0].test.df, Some(ctx.from_ratio(q(125, 13))));
    assert_eq!(t[1].test.df, Some(ctx.from_ratio(q(121, 13))));
    assert_eq!(t[2].test.df, Some(ctx.from_ratio(q(169, 17))));
    close(
        t[0].test.statistic_f64().unwrap(),
        -3.464_101_615_137_755,
        1e-12,
    );
    close(
        t[0].test.p_value_f64().unwrap(),
        0.006_443_866_163_955_33,
        1e-12,
    );
    close(
        t[1].test.p_value_f64().unwrap(),
        0.002_386_749_612_426_776,
        1e-12,
    );
    close(
        t[2].test.p_value_f64().unwrap(),
        0.465_151_039_753_495_75,
        1e-12,
    );
    close(t[0].p_adj, 0.012_887_732_327_910_66, 1e-12);
    close(t[1].p_adj, 0.007_160_248_837_280_328, 1e-12);
    close(t[2].p_adj, 0.465_151_039_753_495_75, 1e-12);
    assert_eq!(
        t.iter().map(|p| p.reject).collect::<Vec<_>>(),
        vec![true, true, false]
    );
}

#[test]
fn pairwise_welch_t_tests_bonferroni_unbalanced() {
    // scipy: ttest_ind(G3U[i], G3U[j], equal_var=False).pvalue
    //   (0,1) 0.11919070661877638 (df 3.9593147751605993), (0,2) 0.0007093070760374697 (df 6.980769230769231),
    //   (1,2) 0.03658848773182927 (df 4.473572938689218)
    // Fraction: Welch df 1849/467, 363/52, 2116/473
    // statsmodels: multipletests(p, alpha=0.05, method='bonferroni')
    //   → [0.35757211985632914, 0.002127921228112409, 0.1097654631954878], reject [False, True, False]
    //   multipletests(p, alpha=0.05, method='holm') → [0.11919070661877638, 0.002127921228112409, 0.07317697546365853]
    let ctx = Context::new();
    let t = pairwise_t_tests(&ctx, &g3u(), Adjustment::Bonferroni, 0.05).unwrap();
    assert_eq!(t[0].test.df, Some(ctx.from_ratio(q(1849, 467))));
    assert_eq!(t[1].test.df, Some(ctx.from_ratio(q(363, 52))));
    assert_eq!(t[2].test.df, Some(ctx.from_ratio(q(2116, 473))));
    close(t[0].p_adj, 0.357_572_119_856_329_14, 1e-12);
    close(t[1].p_adj, 0.002_127_921_228_112_409, 1e-12);
    close(t[2].p_adj, 0.109_765_463_195_487_8, 1e-12);
    assert_eq!(
        t.iter().map(|p| p.reject).collect::<Vec<_>>(),
        vec![false, true, false]
    );
    let h = pairwise_t_tests(&ctx, &g3u(), Adjustment::Holm, 0.05).unwrap();
    close(h[0].p_adj, 0.119_190_706_618_776_38, 1e-12);
    close(h[2].p_adj, 0.073_176_975_463_658_53, 1e-12);
    // Invalid: one group, a singleton group, alpha out of range.
    for (groups, alpha) in [
        (vec![from_i64(&[1, 2])], 0.05),
        (vec![from_i64(&[1, 2]), from_i64(&[3])], 0.05),
        (g3u(), 1.5),
    ] {
        assert!(matches!(
            pairwise_t_tests(&ctx, &groups, Adjustment::Holm, alpha),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The numbers printed in book/src/guide/statistics.md
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn book_two_way_example() {
    // book/src/guide/statistics.md, "Analysis of variance on data": every printed number.
    // statsmodels: anova_lm(ols('y ~ C(A) * C(B)', df).fit(), typ=2) on BAL (see above);
    // scipy: f.sf(441/31, 1, 12) = 0.0026634776886835334, f.sf(1339/31, 2, 12) = 3.291990727040258e-06,
    //        f.sf(75/31, 2, 12) = 0.13098893805732253
    let ctx = Context::new();
    let data = TwoWayData::from_i64(&[
        &[&[4, 5, 6], &[6, 7, 8], &[9, 10, 12]],
        &[&[5, 5, 7], &[8, 9, 11], &[13, 14, 16]],
    ])
    .unwrap();
    let r = anova_two_way(&ctx, &data).unwrap();
    assert_eq!(format!("{}", r.factor_a.ss), "49/2");
    assert_eq!(r.factor_a.df, 1);
    assert_eq!(format!("{}", r.factor_a.f.clone().unwrap()), "441/31");
    close(
        r.factor_a.p_value_f64().unwrap(),
        0.002_663_477_688_683_533_4,
        1e-12,
    );
    assert_eq!(format!("{}", r.factor_b.ss), "1339/9");
    assert_eq!(r.factor_b.df, 2);
    assert_eq!(format!("{}", r.factor_b.f.clone().unwrap()), "1339/31");
    close(
        r.factor_b.p_value_f64().unwrap(),
        3.291_990_727_040_258e-6,
        1e-12,
    );
    assert_eq!(format!("{}", r.interaction.ss), "25/3");
    assert_eq!(r.interaction.df, 2);
    assert_eq!(format!("{}", r.interaction.f.clone().unwrap()), "75/31");
    close(
        r.interaction.p_value_f64().unwrap(),
        0.130_988_938_057_322_53,
        1e-12,
    );
    assert_eq!(format!("{}", r.residual.ss), "62/3");
    assert_eq!(r.residual.df, 12);
    assert_eq!(format!("{}", r.residual.ms.clone().unwrap()), "31/18");
    assert_eq!(format!("{}", r.total.ss), "3641/18");
    assert_eq!(r.total.df, 17);
    assert_eq!(
        format!("{}", r.factor_b.partial_eta_squared.clone().unwrap()),
        "1339/1525"
    );
    // Unbalanced UNB, Type II versus Type I for A (statsmodels typ=2 sum_sq 24.000000000000096, typ=1 37.555555555555636).
    let unb = TwoWayData::from_i64(&[
        &[&[4, 5, 6, 7], &[6, 8], &[9, 10, 12]],
        &[&[5, 7], &[8, 9, 11, 10], &[13, 14, 16]],
    ])
    .unwrap();
    assert_eq!(
        format!("{}", anova_two_way(&ctx, &unb).unwrap().factor_a.ss),
        "24"
    );
    assert_eq!(
        format!(
            "{}",
            anova_two_way_with(&ctx, &unb, SsType::TypeI)
                .unwrap()
                .factor_a
                .ss
        ),
        "338/9"
    );
}

#[test]
fn book_repeated_measures_example() {
    // book/src/guide/statistics.md, "Analysis of variance on data": every printed number.
    // statsmodels: AnovaRM → F Value 12.179775280898882, Pr > F 0.0037355110334743136
    // pingouin: epsilon 'gg' 0.5426457491265326, 'hf' 0.5877873360597936, p_GG_corr 0.02126436585826144,
    //           sphericity → W 0.1571771241004926, chi2 5.551145791696411, dof 2, pval 0.062313767163236534
    let ctx = Context::new();
    let y = [
        from_i64(&[5, 7, 9]),
        from_i64(&[4, 5, 8]),
        from_i64(&[6, 8, 10]),
        from_i64(&[3, 6, 4]),
        from_i64(&[7, 9, 13]),
    ];
    let r = anova_repeated_measures(&ctx, &y).unwrap();
    assert_eq!(
        (format!("{}", r.conditions.ss), r.conditions.df),
        ("542/15".into(), 2)
    );
    assert_eq!(
        (format!("{}", r.subjects.ss), r.subjects.df),
        ("764/15".into(), 4)
    );
    assert_eq!(
        (format!("{}", r.error.ss), r.error.df),
        ("178/15".into(), 8)
    );
    assert_eq!(format!("{}", r.f), "1084/89");
    close(qf(&r.f), 12.179_775_280_898_877, 1e-12);
    close(r.p_value_f64().unwrap(), 0.003_735_511_033_474_317, 1e-12);
    assert_eq!(format!("{}", r.epsilon_gg), "7921/14597");
    close(qf(&r.epsilon_gg), 0.542_645_749_126_532_9, 1e-12);
    assert_eq!(format!("{}", r.epsilon_hf.clone().unwrap()), "4168/7091");
    close(
        qf(r.epsilon_hf.as_ref().unwrap()),
        0.587_787_336_059_794,
        1e-12,
    );
    close(
        r.p_value_gg_f64().unwrap(),
        0.021_264_365_858_261_566,
        1e-12,
    );
    let m = r.mauchly.as_ref().unwrap();
    assert_eq!(format!("{}", m.w), "1245/7921");
    close(m.chi_squared_f64().unwrap(), 5.551_145_791_696_415, 1e-12);
    assert_eq!(m.df, 2);
    close(m.p_value_f64().unwrap(), 0.062_313_767_163_236_4, 1e-12);
}
