//! symplex 0.13 — agreement.  Reference values cite scipy 1.18 / statsmodels 0.15 /
//! krippendorff 0.8 / SymPy 1.14 (`symplex/.venv/bin/python`).
//!
//! Every exact value below was produced by the cited oracle call and
//! converted with `Fraction(x).limit_denominator(10**6)` (checked to
//! agree with the float to 1e-12), or — where noted — by an independent
//! `fractions.Fraction` re-implementation of the published formula in
//! Python.  Data sets are named once here and reused across tests.

use symplex::Interval;
use symplex::linprog::{Q, q, qi};
use symplex::stats::aggregation::*;
use symplex::stats::agreement::*;
use symplex::stats::data::{from_i64, to_f64};

// ── Data sets ────────────────────────────────────────────────────────

/// D1: two raters, 20 items, nominal categories 1..3.
/// Confusion matrix (statsmodels layout): [[5, 2, 0], [0, 5, 2], [1, 0, 5]].
fn d1() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 2, 2, 3, 1]),
        from_i64(&[1, 2, 3, 1, 3, 3, 1, 2, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 3, 2]),
    )
}

/// D2: two raters, 15 items, ordinal categories 1..5.
/// Confusion matrix: [[2,0,0,0,0],[0,2,1,0,0],[0,1,2,1,0],[0,0,1,1,1],[0,0,0,1,2]].
fn d2() -> (Vec<Q>, Vec<Q>) {
    (
        from_i64(&[1, 2, 3, 4, 5, 3, 2, 4, 5, 1, 3, 3, 2, 4, 5]),
        from_i64(&[1, 3, 3, 4, 4, 2, 2, 5, 5, 1, 3, 4, 2, 3, 5]),
    )
}

/// Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability",
/// the worked example: 4 observers × 12 units, values 1..5, missing
/// cells; unit 12 has a single rating and drops out.
fn k2011() -> RatingTable {
    RatingTable::from_raters_i64(&[
        &[
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(2),
            Some(1),
            Some(4),
            Some(1),
            Some(2),
            None,
            None,
            None,
        ],
        &[
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(2),
            Some(2),
            Some(4),
            Some(1),
            Some(2),
            Some(5),
            None,
            Some(3),
        ],
        &[
            None,
            Some(3),
            Some(3),
            Some(3),
            Some(2),
            Some(3),
            Some(4),
            Some(2),
            Some(2),
            Some(5),
            Some(1),
            None,
        ],
        &[
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(2),
            Some(4),
            Some(4),
            Some(1),
            Some(2),
            Some(5),
            Some(1),
            None,
        ],
    ])
    .unwrap()
}

/// K2: complete, 3 raters × 8 items, values 0..2.
fn k2() -> RatingTable {
    RatingTable::from_raters_i64(&[
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
        ],
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(2),
            Some(2),
            Some(0),
            Some(1),
        ],
        &[
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(1),
            Some(2),
            Some(1),
            Some(1),
        ],
    ])
    .unwrap()
}

/// K3: 4 raters × 7 items with missing cells, values 1..4.
fn k3() -> RatingTable {
    RatingTable::from_raters_i64(&[
        &[Some(1), Some(2), None, Some(4), Some(3), Some(2), Some(1)],
        &[Some(1), Some(3), Some(3), Some(4), Some(3), None, Some(1)],
        &[None, Some(2), Some(3), Some(4), Some(4), Some(2), Some(2)],
        &[Some(1), Some(2), Some(3), None, Some(3), Some(2), Some(1)],
    ])
    .unwrap()
}

/// K4: 3 raters × 8 items with missing cells, values 0..3 (a zero value
/// exercises the ratio metric's `c + k = 0` rule).
fn k4() -> RatingTable {
    RatingTable::from_raters_i64(&[
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(3),
            Some(1),
            None,
            Some(2),
        ],
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(1),
            Some(3),
            Some(1),
            Some(2),
            Some(2),
        ],
        &[
            Some(0),
            Some(2),
            Some(2),
            Some(0),
            Some(3),
            None,
            Some(2),
            Some(1),
        ],
    ])
    .unwrap()
}

/// Shrout & Fleiss (1979), Table 2: 6 targets × 4 judges.
fn shrout_fleiss() -> RatingTable {
    RatingTable::from_i64(&[
        &[9, 2, 5, 8],
        &[6, 1, 3, 2],
        &[8, 4, 6, 8],
        &[7, 1, 2, 6],
        &[10, 5, 6, 9],
        &[6, 2, 4, 7],
    ])
    .unwrap()
}

/// A rational as a float, for comparisons against the oracle's float.
fn f(x: &Q) -> f64 {
    to_f64(std::slice::from_ref(x))[0]
}

// ── Percent agreement ────────────────────────────────────────────────

#[test]
fn percent_agreement_two_raters() {
    let (a, b) = d1();
    // 15 of 20 items agree (trace of the confusion matrix = 5 + 5 + 5).
    assert_eq!(percent_agreement(&a, &b).unwrap(), q(3, 4));
    assert!(percent_agreement(&a[..5], &b).is_err());
    assert!(percent_agreement(&[], &[]).is_err());
}

#[test]
fn pairwise_percent_agreement_skips_pairs_without_common_items() {
    // Python (Fractions): pairs (0,1) 1/2 over items 1,2; (0,2) 1/1 over item 0; (1,2) no common item → mean 3/4.
    let t = RatingTable::from_i64_missing(&[
        &[Some(1), None, Some(1)],
        &[Some(2), Some(2), None],
        &[Some(1), Some(2), None],
    ])
    .unwrap();
    assert_eq!(pairwise_percent_agreement(&t).unwrap(), q(3, 4));
    // Complete K2: pairs (0,1) 7/8, (0,2) 6/8, (1,2) 5/8 → 3/4.
    assert_eq!(pairwise_percent_agreement(&k2()).unwrap(), q(3, 4));
    let one = RatingTable::from_i64(&[&[1], &[2]]).unwrap();
    assert!(pairwise_percent_agreement(&one).is_err());
}

// ── Cohen's κ, Scott's π ─────────────────────────────────────────────

#[test]
fn cohen_kappa_nominal_matches_statsmodels() {
    let (a, b) = d1();
    // statsmodels: cohens_kappa([[5,2,0],[0,5,2],[1,0,5]], return_results=True).kappa
    //   = 0.6254681647940076 = 167/267;  p_o = 15/20 = 3/4;
    //   p_e = Σ row_i·col_i / n² = (7·6 + 7·7 + 6·7)/400 = 133/400.
    let k = cohen_kappa(&a, &b).unwrap();
    assert_eq!(k.kappa, q(167, 267));
    assert_eq!(k.observed, q(3, 4));
    assert_eq!(k.expected, q(133, 400));
    assert!((f(&k.kappa) - 0.6254681647940076).abs() < 1e-12);
}

#[test]
fn confusion_matrix_and_kappa_from_confusion() {
    let (a, b) = d1();
    let cats = from_i64(&[1, 2, 3]);
    let m = confusion_matrix(&a, &b, &cats).unwrap();
    assert_eq!(m, vec![vec![5, 2, 0], vec![0, 5, 2], vec![1, 0, 5]]);
    assert_eq!(
        kappa_from_confusion(&m, &Weights::Unweighted)
            .unwrap()
            .kappa,
        q(167, 267)
    );
    // A rating outside the category list, a repeated category.
    assert!(confusion_matrix(&a, &b, &from_i64(&[1, 2])).is_err());
    assert!(confusion_matrix(&a, &b, &from_i64(&[1, 2, 2, 3])).is_err());
    // Degenerate: both raters constant → p_e = 1, κ undefined.
    assert!(kappa_from_confusion(&[vec![7]], &Weights::Unweighted).is_err());
    assert!(kappa_from_confusion(&[vec![1, 2], vec![3]], &Weights::Unweighted).is_err());
}

#[test]
fn weighted_kappa_linear_quadratic_custom_match_statsmodels() {
    let (a, b) = d2();
    // statsmodels, table = [[2,0,0,0,0],[0,2,1,0,0],[0,1,2,1,0],[0,0,1,1,1],[0,0,0,1,2]]:
    //   cohens_kappa(table, return_results=False)                 = 0.4943820224719101 = 44/89
    //   cohens_kappa(table, wt='linear', return_results=False)    = 0.7289156626506024 = 121/166
    //   cohens_kappa(table, wt='quadratic', return_results=False) = 0.883419689119171  = 341/386
    assert_eq!(cohen_kappa(&a, &b).unwrap().kappa, q(44, 89));
    assert_eq!(
        weighted_kappa(&a, &b, &Weights::Linear).unwrap().kappa,
        q(121, 166)
    );
    assert_eq!(
        weighted_kappa(&a, &b, &Weights::Quadratic).unwrap().kappa,
        q(341, 386)
    );
    // Custom disagreement weights: 1 off the diagonal except 1/2 for adjacent categories.
    //   W = np.ones((5,5)); W[i,i] = 0; W[i,i+1] = W[i+1,i] = 0.5
    //   cohens_kappa(table, weights=W, return_results=False) = 0.6762589928057554 = 94/139
    let w: Vec<Vec<Q>> = (0..5usize)
        .map(|i| {
            (0..5usize)
                .map(|j| match i.abs_diff(j) {
                    0 => qi(0),
                    1 => q(1, 2),
                    _ => qi(1),
                })
                .collect()
        })
        .collect();
    assert_eq!(
        weighted_kappa(&a, &b, &Weights::Custom(w)).unwrap().kappa,
        q(94, 139)
    );
    // A custom matrix of the wrong shape.
    assert!(weighted_kappa(&a, &b, &Weights::Custom(vec![vec![qi(0)]])).is_err());
}

#[test]
fn weighted_kappa_observed_and_expected_are_weighted_agreements() {
    let (a, b) = d2();
    // For any weights κ = (p_o − p_e)/(1 − p_e) with p_o = 1 − Σ w p_ij, p_e = 1 − Σ w p_i· p_·j.
    for w in [Weights::Unweighted, Weights::Linear, Weights::Quadratic] {
        let k = weighted_kappa(&a, &b, &w).unwrap();
        let one = qi(1);
        assert_eq!(k.kappa, (&k.observed - &k.expected) / (&one - &k.expected));
    }
}

#[test]
fn scott_pi_matches_fleiss_kappa_for_two_raters() {
    let (a, b) = d1();
    // Fractions: p_o = 3/4; pooled marginals (13, 14, 13)/40 → p_e = (169 + 196 + 169)/1600 = 267/800;
    //   π = (3/4 − 267/800)/(1 − 267/800) = 333/533 = 0.624765478424015.
    // statsmodels: fleiss_kappa(aggregate_raters(np.array([a1, b1]).T)[0]) = 0.624765478424015 = 333/533.
    assert_eq!(scott_pi(&a, &b).unwrap(), q(333, 533));
    let table = RatingTable::new(
        a.iter()
            .zip(&b)
            .map(|(x, y)| vec![Some(x.clone()), Some(y.clone())])
            .collect(),
    )
    .unwrap();
    assert_eq!(fleiss_kappa_ratings(&table).unwrap(), q(333, 533));
    assert!(scott_pi(&from_i64(&[1, 1]), &from_i64(&[1, 1])).is_err());
}

// ── Fleiss' κ ────────────────────────────────────────────────────────

#[test]
fn fleiss_kappa_wikipedia_table_matches_statsmodels() {
    // The Wikipedia "Fleiss' kappa" worked example: 10 subjects, 14 raters, 5 categories.
    // statsmodels: fleiss_kappa(np.array(F)) = 0.20993070442195522 = 4211/20059.
    let table = vec![
        vec![0, 0, 0, 0, 14],
        vec![0, 2, 6, 4, 2],
        vec![0, 0, 3, 5, 6],
        vec![0, 3, 9, 2, 0],
        vec![2, 2, 8, 1, 1],
        vec![7, 7, 0, 0, 0],
        vec![3, 2, 6, 3, 0],
        vec![2, 5, 3, 2, 2],
        vec![6, 5, 2, 1, 0],
        vec![0, 2, 2, 3, 7],
    ];
    assert_eq!(fleiss_kappa(&table).unwrap(), q(4211, 20059));
    assert!((f(&fleiss_kappa(&table).unwrap()) - 0.20993070442195522).abs() < 1e-12);
}

#[test]
fn fleiss_kappa_from_ratings_aggregates_like_statsmodels() {
    // R3 = [[0,0,1],[1,1,1],[2,2,0],[0,0,0],[1,2,1],[2,2,2]] (items × raters).
    // statsmodels: aggregate_raters(np.array(R3))[0]
    //   = [[2,1,0],[0,3,0],[1,0,2],[3,0,0],[0,2,1],[0,0,3]];  fleiss_kappa(that) = 0.5 = 1/2.
    let t = RatingTable::from_i64(&[
        &[0, 0, 1],
        &[1, 1, 1],
        &[2, 2, 0],
        &[0, 0, 0],
        &[1, 2, 1],
        &[2, 2, 2],
    ])
    .unwrap();
    assert_eq!(
        t.count_table(&t.categories()).unwrap(),
        vec![
            vec![2, 1, 0],
            vec![0, 3, 0],
            vec![1, 0, 2],
            vec![3, 0, 0],
            vec![0, 2, 1],
            vec![0, 0, 3]
        ]
    );
    assert_eq!(fleiss_kappa_ratings(&t).unwrap(), q(1, 2));
    // Validation: ragged rows, unequal rater counts, a single rater, a missing cell.
    assert!(fleiss_kappa(&[vec![1, 2], vec![3]]).is_err());
    assert!(fleiss_kappa(&[vec![1, 2], vec![2, 2]]).is_err());
    assert!(fleiss_kappa(&[vec![1, 0], vec![0, 1]]).is_err());
    assert!(fleiss_kappa_ratings(&k3()).is_err());
}

// ── Krippendorff's α ─────────────────────────────────────────────────

#[test]
fn krippendorff_alpha_2011_worked_example_all_levels() {
    // krippendorff.alpha(reliability_data=K, level_of_measurement=…) with K the 4 × 12 matrix
    // of Krippendorff (2011) (np.nan for '.'):
    //   nominal  0.743421052631579  = 113/152   (paper: .743)
    //   ordinal  0.8153875037548814 = 108577/133160   (paper: .815)
    //   interval 0.8491071428571428 = 951/1120  (paper: .849)
    //   ratio    0.7974027747116121 = 18222619/22852465   (paper: .797)
    // The ratio value's exact fraction was confirmed by an independent Fraction re-implementation
    // of the coincidence-matrix formula (limit_denominator(10**6) is too coarse for it).
    let t = k2011();
    assert_eq!(t.n_items(), 12);
    assert_eq!(t.n_raters(), 4);
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(113, 152));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(108577, 133160)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Interval).unwrap(),
        q(951, 1120)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Ratio).unwrap(),
        q(18222619, 22852465)
    );
    assert!((f(&krippendorff_alpha(&t, Level::Ratio).unwrap()) - 0.7974027747116121).abs() < 1e-12);
}

#[test]
fn krippendorff_alpha_complete_table() {
    // krippendorff.alpha(reliability_data=K2, level_of_measurement=…):
    //   nominal 0.6329787234042552 = 119/188, ordinal 0.8012979497354498 = 19385/24192,
    //   interval 0.7921686746987951 = 263/332, ratio 0.7912541254125413 = 959/1212.
    let t = k2();
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(119, 188));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(19385, 24192)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Interval).unwrap(),
        q(263, 332)
    );
    assert_eq!(krippendorff_alpha(&t, Level::Ratio).unwrap(), q(959, 1212));
}

#[test]
fn krippendorff_alpha_with_missing_cells() {
    // krippendorff.alpha(reliability_data=K3, level_of_measurement=…):
    //   nominal 0.676056338028169 = 48/71, ordinal 0.8913322445170322 = 15281/17144,
    //   interval 0.8878048780487805 = 182/205, ratio 0.8660424575155347 = 562369/649355
    //   (ratio fraction confirmed exactly by the Fraction re-implementation).
    let t = k3();
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(48, 71));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(15281, 17144)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Interval).unwrap(),
        q(182, 205)
    );
    assert_eq!(
        krippendorff_alpha(&t, Level::Ratio).unwrap(),
        q(562369, 649355)
    );
}

#[test]
fn krippendorff_alpha_with_zero_values_and_two_raters() {
    // krippendorff.alpha(reliability_data=K4, level_of_measurement=…):
    //   nominal 0.64 = 16/25, ordinal 0.8496063211972302 = 60859/71632,
    //   interval 0.8656716417910448 = 58/67, ratio 0.7320620780847658 = 10519/14369.
    let t = k4();
    assert_eq!(krippendorff_alpha(&t, Level::Nominal).unwrap(), q(16, 25));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ordinal).unwrap(),
        q(60859, 71632)
    );
    assert_eq!(krippendorff_alpha(&t, Level::Interval).unwrap(), q(58, 67));
    assert_eq!(
        krippendorff_alpha(&t, Level::Ratio).unwrap(),
        q(10519, 14369)
    );
    // Two raters, D1: krippendorff.alpha(reliability_data=[a1, b1], level_of_measurement='nominal')
    //   = 0.6341463414634146 = 26/41.
    let (a, b) = d1();
    let pair = RatingTable::new(
        a.iter()
            .zip(&b)
            .map(|(x, y)| vec![Some(x.clone()), Some(y.clone())])
            .collect(),
    )
    .unwrap();
    assert_eq!(
        krippendorff_alpha(&pair, Level::Nominal).unwrap(),
        q(26, 41)
    );
}

#[test]
fn krippendorff_alpha_degenerate_inputs() {
    // A single value → error (the package raises "There has to be more than one value").
    let constant = RatingTable::from_i64(&[&[1, 1], &[1, 1]]).unwrap();
    assert!(krippendorff_alpha(&constant, Level::Nominal).is_err());
    // No unit with two ratings.
    let singles = RatingTable::from_i64_missing(&[&[Some(1), None], &[None, Some(2)]]).unwrap();
    assert!(krippendorff_alpha(&singles, Level::Nominal).is_err());
    // Perfect agreement → α = 1.
    let perfect = RatingTable::from_i64(&[&[1, 1], &[2, 2], &[3, 3]]).unwrap();
    assert_eq!(
        krippendorff_alpha(&perfect, Level::Interval).unwrap(),
        qi(1)
    );
}

// ── Gwet's AC₁ ───────────────────────────────────────────────────────

#[test]
fn gwet_ac1_with_missing_ratings() {
    // G (items × raters, None missing), categories {1, 2, 3}.  Reference by Fractions in Python from
    // Gwet (2008): p_a = mean over items with r_i ≥ 2 of Σ_k r_ik(r_ik−1)/(r_i(r_i−1)),
    // π_k = mean_i r_ik/r_i, p_e = Σ π_k(1−π_k)/(K−1), AC1 = (p_a−p_e)/(1−p_e) = 191/271 = 0.7047970479704797.
    let g = RatingTable::from_i64_missing(&[
        &[Some(1), Some(1), Some(1), Some(1)],
        &[Some(2), Some(2), Some(2), Some(1)],
        &[Some(1), Some(1), None, Some(1)],
        &[Some(3), Some(3), Some(3), Some(3)],
        &[Some(2), Some(2), Some(1), Some(2)],
        &[Some(1), None, Some(1), Some(1)],
        &[Some(3), Some(3), Some(3), Some(2)],
        &[Some(2), Some(2), Some(2), Some(2)],
        &[Some(1), Some(1), Some(1), Some(1)],
        &[Some(3), Some(2), Some(3), Some(3)],
    ])
    .unwrap();
    assert_eq!(gwet_ac1(&g, None).unwrap(), q(191, 271));
    // Same formula on D1 (two raters, three categories): 667/1067 = 0.6251171508903468.
    let (a, b) = d1();
    let pair = RatingTable::new(
        a.iter()
            .zip(&b)
            .map(|(x, y)| vec![Some(x.clone()), Some(y.clone())])
            .collect(),
    )
    .unwrap();
    assert_eq!(gwet_ac1(&pair, None).unwrap(), q(667, 1067));
    // Passing the categories explicitly changes K: with a fourth, unused category
    //   p_e = Σ π(1−π)/3 instead of /2, so AC1 rises.  Fractions: with K = 4 on D1, 1267/1867.
    assert_eq!(
        gwet_ac1(&pair, Some(&from_i64(&[1, 2, 3, 4]))).unwrap(),
        q(1267, 1867)
    );
    assert!(gwet_ac1(&pair, Some(&from_i64(&[1, 2]))).is_err());
    assert!(gwet_ac1(&RatingTable::from_i64(&[&[1, 1]]).unwrap(), None).is_err());
}

// ── Intraclass correlation ───────────────────────────────────────────

#[test]
fn icc_anova_matches_statsmodels_two_way_anova() {
    // statsmodels: anova_lm(ols('y ~ C(t) + C(j)', df).fit(), typ=2) on Shrout & Fleiss' table:
    //   C(t) sum_sq 56.208333 df 5;  C(j) 97.458333 df 3;  Residual 15.291667 df 15
    //   → MSR = 1349/120 = 11.2417, MSC = 2339/72 = 32.4861, MSE = 367/360 = 1.01944;
    //   MSW = (SSC + SSE)/(n(k−1)) = 451/72 = 6.26389 (Fractions).
    let a = icc_anova(&shrout_fleiss()).unwrap();
    assert_eq!(a.msr, q(1349, 120));
    assert_eq!(a.msc, q(2339, 72));
    assert_eq!(a.mse, q(367, 360));
    assert_eq!(a.msw, q(451, 72));
    assert!((f(&a.msr) * 5.0 - 56.208333).abs() < 1e-5);
    assert!((f(&a.msc) * 3.0 - 97.458333).abs() < 1e-5);
    assert!((f(&a.mse) * 15.0 - 15.291667).abs() < 1e-5);
}

#[test]
fn icc_six_forms_match_shrout_fleiss_1979() {
    // Fractions in Python from the mean squares above (Shrout & Fleiss 1979, Table 4 formulas;
    // the paper reports .17, .29, .71, .44, .62, .91 for the same table):
    //   ICC(1)   = 448/2703  = 0.1657417684054754
    //   ICC(2,1) = 184/635   = 0.28976377952755905
    //   ICC(3,1) = 920/1287  = 0.7148407148407149
    //   ICC(1,k) = 1792/4047 = 0.4427971336792686
    //   ICC(2,k) = 736/1187  = 0.620050547598989
    //   ICC(3,k) = 3680/4047 = 0.9093155423770695
    let t = shrout_fleiss();
    assert_eq!(icc(&t, IccForm::Icc1).unwrap(), q(448, 2703));
    assert_eq!(icc(&t, IccForm::Icc2Single).unwrap(), q(184, 635));
    assert_eq!(icc(&t, IccForm::Icc3Single).unwrap(), q(920, 1287));
    assert_eq!(icc(&t, IccForm::Icc1Average).unwrap(), q(1792, 4047));
    assert_eq!(icc(&t, IccForm::Icc2Average).unwrap(), q(736, 1187));
    assert_eq!(icc(&t, IccForm::Icc3Average).unwrap(), q(3680, 4047));
    assert!((f(&icc(&t, IccForm::Icc2Single).unwrap()) - 0.28976377952755905).abs() < 1e-12);
}

#[test]
fn icc_rejects_incomplete_and_degenerate_tables() {
    assert!(icc(&k3(), IccForm::Icc1).is_err());
    assert!(icc(&RatingTable::from_i64(&[&[1, 2]]).unwrap(), IccForm::Icc1).is_err());
    assert!(
        icc(
            &RatingTable::from_i64(&[&[1], &[2]]).unwrap(),
            IccForm::Icc1
        )
        .is_err()
    );
    // No variance at all: every mean square is zero → denominator zero.
    let flat = RatingTable::from_i64(&[&[3, 3], &[3, 3]]).unwrap();
    assert!(icc(&flat, IccForm::Icc3Average).is_err());
    // Perfect agreement between raters: ICC(3,1) = ICC(3,k) = 1.
    let perfect = RatingTable::from_i64(&[&[1, 1], &[4, 4], &[2, 2]]).unwrap();
    assert_eq!(icc(&perfect, IccForm::Icc3Single).unwrap(), qi(1));
    assert_eq!(icc(&perfect, IccForm::Icc2Single).unwrap(), qi(1));
}

// ── Kendall's W ──────────────────────────────────────────────────────

#[test]
fn kendall_w_without_ties() {
    // KW (items × raters, rankings): [[1,1,2],[2,3,1],[3,2,3],[4,4,4],[5,6,5],[6,5,6]].
    // Python: W = 12 S / (m²(n³−n) − m ΣT) with scipy.stats.rankdata per column = 0.9111111111111111 = 41/45;
    //   scipy.stats.friedmanchisquare(*rows).statistic / (m(n−1)) = 0.9111111111111104.
    let t = RatingTable::from_i64(&[
        &[1, 1, 2],
        &[2, 3, 1],
        &[3, 2, 3],
        &[4, 4, 4],
        &[5, 6, 5],
        &[6, 5, 6],
    ])
    .unwrap();
    assert_eq!(kendall_w(&t).unwrap(), q(41, 45));
}

#[test]
fn kendall_w_with_ties() {
    // KW2 (scores with ties): [[3,4,2],[1,1,1],[4,4,3],[2,2,2],[5,5,5],[3,3,4]].
    // Python (rankdata + tie correction, and exact Fractions): W = 277/306 = 0.9052287581699346;
    //   scipy.stats.friedmanchisquare(*rows).statistic / (m(n−1)) = 0.9052287581699346.
    let t = RatingTable::from_i64(&[
        &[3, 4, 2],
        &[1, 1, 1],
        &[4, 4, 3],
        &[2, 2, 2],
        &[5, 5, 5],
        &[3, 3, 4],
    ])
    .unwrap();
    assert_eq!(kendall_w(&t).unwrap(), q(277, 306));
    // Degenerate: every rater ties everything; incomplete table; one item.
    assert!(kendall_w(&RatingTable::from_i64(&[&[1, 1], &[1, 1]]).unwrap()).is_err());
    assert!(kendall_w(&k3()).is_err());
    assert!(kendall_w(&RatingTable::from_i64(&[&[1, 2]]).unwrap()).is_err());
    // Identical rankings → W = 1.
    let same = RatingTable::from_i64(&[&[1, 1], &[2, 2], &[3, 3]]).unwrap();
    assert_eq!(kendall_w(&same).unwrap(), qi(1));
}

// ── Rating tables ────────────────────────────────────────────────────

#[test]
fn rating_table_construction_and_queries() {
    let t = k2011();
    assert!(!t.is_complete());
    assert!(k2().is_complete());
    assert_eq!(t.categories(), from_i64(&[1, 2, 3, 4, 5]));
    assert_eq!(t.get(0, 2), None);
    assert_eq!(t.get(0, 0), Some(&qi(1)));
    assert_eq!(t.rater(2).unwrap()[0], None);
    assert_eq!(t.item(11).unwrap(), &[None, Some(qi(3)), None, None]);
    // Observers B and D share units 1–10 (12 items minus unit 11 (B missing) and unit 12 (D missing)).
    let (b, d) = t.paired_ratings(1, 3).unwrap();
    assert_eq!(b.len(), 10);
    assert_eq!(percent_agreement(&b, &d).unwrap(), q(9, 10));
    // Frequencies over all present cells (Python Counter): 1×9, 2×13, 3×11, 4×5, 5×3 (41 ratings).
    assert_eq!(
        category_frequencies(&t),
        vec![(qi(1), 9), (qi(2), 13), (qi(3), 11), (qi(4), 5), (qi(5), 3)]
    );
    // Validation.
    assert!(RatingTable::new(vec![]).is_err());
    assert!(RatingTable::new(vec![vec![]]).is_err());
    assert!(RatingTable::from_i64(&[&[1, 2], &[1]]).is_err());
    assert!(RatingTable::from_raters_i64(&[&[Some(1), Some(2)], &[Some(1)]]).is_err());
    assert!(t.paired_ratings(0, 4).is_err());
    assert!(t.complete_rows().is_err());
    assert_eq!(k2().complete_rows().unwrap().len(), 8);
}

// ── Votes ────────────────────────────────────────────────────────────

#[test]
fn majority_plurality_and_weighted_votes() {
    let labels = [Some(2), Some(0), Some(2), None, Some(1), Some(2)];
    let v = majority_vote(&labels);
    assert_eq!(
        (v.winner, v.tied.clone(), v.counts.clone()),
        (Some(2), vec![2], vec![1, 1, 3])
    );
    let tie = majority_vote(&[Some(0), Some(1), None]);
    assert_eq!((tie.winner, tie.tied.clone()), (None, vec![0, 1]));
    let none = majority_vote(&[None, None]);
    assert_eq!(
        (none.winner, none.tied.clone(), none.counts.clone()),
        (None, vec![], vec![])
    );
    // Plurality: 3 of 5 cast ≥ 3/5 passes, > 3/5 fails; the threshold must lie in [0, 1].
    assert_eq!(plurality(&labels, &q(3, 5)).unwrap().winner, Some(2));
    assert_eq!(plurality(&labels, &q(2, 3)).unwrap().winner, None);
    assert!(plurality(&labels, &q(3, 2)).is_err());
    // Weighted: a rater of weight 5/2 outvotes two of weight 1.
    let w = weighted_vote(&[Some(0), Some(1), Some(1)], &[q(5, 2), qi(1), qi(1)]).unwrap();
    assert_eq!(
        (w.winner, w.scores.clone()),
        (Some(0), vec![q(5, 2), qi(2)])
    );
    let w = weighted_vote(&[Some(0), Some(1), Some(1)], &[qi(2), qi(1), qi(1)]).unwrap();
    assert_eq!((w.winner, w.tied.clone()), (None, vec![0, 1]));
    assert!(weighted_vote(&[Some(0)], &[qi(1), qi(1)]).is_err());
    assert!(weighted_vote(&[Some(0)], &[qi(-1)]).is_err());
}

#[test]
fn majority_votes_over_a_label_table() {
    let t = LabelTable::from_rows(
        &[
            &[Some(0), Some(0), Some(1)],
            &[Some(1), None, Some(2)],
            &[None, None, None],
        ],
        3,
    )
    .unwrap();
    let v = majority_votes(&t);
    assert_eq!(v[0].winner, Some(0));
    assert_eq!(
        (v[1].winner, v[1].tied.clone(), v[1].counts.clone()),
        (None, vec![1, 2], vec![0, 1, 1])
    );
    assert_eq!((v[2].winner, v[2].counts.clone()), (None, vec![0, 0, 0]));
    assert_eq!(t.rater(1).unwrap(), vec![Some(0), None, None]);
    assert!(LabelTable::from_rows(&[&[Some(3)]], 3).is_err());
    assert!(LabelTable::from_rows(&[&[Some(0)], &[]], 3).is_err());
    assert!(LabelTable::new(vec![], 3).is_err());
    assert!(LabelTable::complete(&[&[0, 1]], 0).is_err());
}

// ── Dawid–Skene ──────────────────────────────────────────────────────

/// The M-step / E-step of Dawid & Skene (1979) written out independently
/// of the library, to check the returned estimates are a fixed point.
fn ds_fixed_point_residual(counts: &[Vec<Vec<usize>>], ds: &DawidSkene, smoothing: f64) -> f64 {
    let n_items = counts.len();
    let n_raters = counts[0].len();
    let j = ds.priors.len();
    let mut worst = 0.0f64;
    // p_j = Σ_i T_ij / I
    for c in 0..j {
        let p: f64 = ds.posteriors.iter().map(|r| r[c]).sum::<f64>() / n_items as f64;
        worst = worst.max((p - ds.priors[c]).abs());
    }
    // π^(k)_jl = (Σ_i T_ij n_ikl + s) / Σ_l (…)
    for k in 0..n_raters {
        for c in 0..j {
            let num: Vec<f64> = (0..j)
                .map(|l| {
                    counts
                        .iter()
                        .zip(&ds.posteriors)
                        .map(|(it, t)| t[c] * it[k][l] as f64)
                        .sum::<f64>()
                        + smoothing
                })
                .collect();
            let den: f64 = num.iter().sum();
            for (l, &n_l) in num.iter().enumerate() {
                let expect = if den > 0.0 { n_l / den } else { 1.0 / j as f64 };
                worst = worst.max((expect - ds.confusion[k][c][l]).abs());
            }
        }
    }
    // T_ij ∝ p_j Π_k Π_l π^(k)_jl^{n_ikl}
    for (item, t) in counts.iter().zip(&ds.posteriors) {
        let mut w: Vec<f64> = (0..j)
            .map(|c| {
                let mut v = ds.priors[c];
                for (k, labels) in item.iter().enumerate() {
                    for (l, &n) in labels.iter().enumerate() {
                        v *= ds.confusion[k][c][l].powi(n as i32);
                    }
                }
                v
            })
            .collect();
        let s: f64 = w.iter().sum();
        for v in &mut w {
            *v /= s;
        }
        for (a, b) in w.iter().zip(t) {
            worst = worst.max((a - b).abs());
        }
    }
    worst
}

/// Synthetic table: 14 items, 5 raters, 3 classes; rater 0 is perfect,
/// item 12 is tied 2–2 under majority vote (rater 3 abstains).
fn synthetic_ds() -> LabelTable {
    let raters: [[Option<usize>; 14]; 5] = [
        [
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
        ],
        [
            Some(0),
            Some(1),
            Some(2),
            Some(1),
            Some(1),
            Some(2),
            Some(0),
            Some(2),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(1),
            Some(1),
        ],
        [
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(1),
            Some(1),
        ],
        [
            Some(0),
            Some(2),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(2),
            Some(1),
            Some(2),
            None,
            Some(1),
        ],
        [
            Some(0),
            Some(1),
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            Some(1),
        ],
    ];
    let rows: Vec<Vec<Option<usize>>> = (0..14)
        .map(|i| raters.iter().map(|r| r[i]).collect())
        .collect();
    LabelTable::new(rows, 3).unwrap()
}

#[test]
fn dawid_skene_recovers_the_perfect_rater_and_breaks_the_tie() {
    // Independent numpy EM (majority-vote init, tol 1e-12) on the same table:
    //   iterations 6, converged, argmax = truth = [0,1,2,0,1,2,0,1,2,0,1,2,0,1],
    //   T[12] = [1, 0, 0] (majority vote is tied 2–2 there), priors = [5/14, 5/14, 4/14],
    //   π^(0) = I₃,  π^(1) = [[0.6,0.4,0],[0,0.8,0.2],[0,0,1]],  π^(3) = [[0.75,0,0.25],[0,0.8,0.2],[0,0,1]].
    let t = synthetic_ds();
    let truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let mv = majority_votes(&t);
    assert_eq!((mv[12].winner, mv[12].tied.clone()), (None, vec![0, 1]));
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene(&t, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 6);
    assert_eq!(ds.labels(), truth.to_vec());
    assert!((ds.posteriors[12][0] - 1.0).abs() < 1e-9);
    for (p, e) in ds.priors.iter().zip([5.0 / 14.0, 5.0 / 14.0, 4.0 / 14.0]) {
        assert!((p - e).abs() < 1e-9);
    }
    for c in 0..3 {
        for l in 0..3 {
            let expect = if c == l { 1.0 } else { 0.0 };
            assert!((ds.confusion[0][c][l] - expect).abs() < 1e-9);
        }
    }
    let pi1 = [[0.6, 0.4, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]];
    let pi3 = [[0.75, 0.0, 0.25], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]];
    for c in 0..3 {
        for l in 0..3 {
            assert!(
                (ds.confusion[1][c][l] - pi1[c][l]).abs() < 1e-9,
                "π^(1)[{c}][{l}] = {}",
                ds.confusion[1][c][l]
            );
            assert!(
                (ds.confusion[3][c][l] - pi3[c][l]).abs() < 1e-9,
                "π^(3)[{c}][{l}] = {}",
                ds.confusion[3][c][l]
            );
        }
    }
    assert!(ds_fixed_point_residual(&t.to_counts(), &ds, 0.0) < 1e-9);
    assert!(ds.log_likelihood.is_finite() && ds.log_likelihood <= 0.0);
}

#[test]
fn dawid_skene_with_smoothing_matches_the_reference_and_is_a_fixed_point() {
    // numpy EM with smoothing 0.01 (pseudo-count per confusion cell), tol 1e-12:
    //   iterations 9, converged, same argmax, T[12] = [0.999956368498, 0.000039740842, 0.000003890661],
    //   priors = [0.357139776163, 0.35714557332, 0.285714650516],
    //   π^(1) = [[0.598414669199, 0.399597238728, 0.001988092073],
    //            [0.001988056546, 0.797218084953, 0.2007938585],
    //            [0.002481392823, 0.002482836207, 0.99503577097]].
    let t = synthetic_ds();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        smoothing: 0.01,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene(&t, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 9);
    assert_eq!(ds.labels(), vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]);
    for (v, e) in ds.posteriors[12]
        .iter()
        .zip([0.999956368498, 0.000039740842, 0.000003890661])
    {
        assert!((v - e).abs() < 1e-9);
    }
    for (v, e) in ds
        .priors
        .iter()
        .zip([0.357139776163, 0.35714557332, 0.285714650516])
    {
        assert!((v - e).abs() < 1e-9);
    }
    let pi1 = [
        [0.598414669199, 0.399597238728, 0.001988092073],
        [0.001988056546, 0.797218084953, 0.2007938585],
        [0.002481392823, 0.002482836207, 0.99503577097],
    ];
    for (row, expect) in ds.confusion[1].iter().zip(&pi1) {
        for (v, e) in row.iter().zip(expect) {
            assert!((v - e).abs() < 1e-9);
        }
    }
    assert!(ds_fixed_point_residual(&t.to_counts(), &ds, 0.01) < 1e-9);
    // Deterministic: the same input gives bit-identical output.
    assert_eq!(dawid_skene(&t, &opts).unwrap(), ds);
}

/// Dawid & Skene (1979), Table 1 as transcribed here: 45 patients, observer 1
/// rating three times and observers 2–5 once, 4 categories.  The
/// transcription could not be verified against the paper (the paper's
/// fitted prevalences are .40/.42/.11/.07; this transcription gives
/// .400/.467/.111/.022), so the table serves as a realistic replicate-count
/// input checked against the independent numpy EM, not against the paper.
fn anesthetist_counts() -> Vec<Vec<Vec<usize>>> {
    const ROWS: [[usize; 7]; 45] = [
        [1, 1, 1, 1, 1, 1, 1],
        [3, 3, 3, 4, 3, 3, 4],
        [1, 1, 2, 2, 1, 2, 2],
        [2, 2, 2, 3, 1, 2, 1],
        [2, 2, 2, 3, 2, 2, 2],
        [2, 2, 2, 3, 3, 2, 2],
        [1, 2, 2, 2, 1, 1, 1],
        [3, 3, 3, 3, 4, 3, 3],
        [2, 2, 2, 2, 2, 2, 3],
        [2, 3, 2, 2, 2, 2, 3],
        [4, 4, 4, 4, 4, 4, 4],
        [2, 2, 2, 3, 3, 4, 3],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 3, 2, 1, 2],
        [1, 2, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 1],
        [2, 2, 2, 1, 3, 2, 2],
        [2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 1],
        [2, 2, 2, 3, 2, 2, 2],
        [2, 2, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 3, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1, 2, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [3, 3, 3, 3, 2, 3, 3],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 3, 2, 3, 2],
        [4, 3, 3, 4, 3, 4, 3],
        [2, 2, 1, 2, 2, 3, 2],
        [2, 3, 2, 3, 2, 3, 3],
        [3, 3, 3, 3, 4, 3, 2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1, 1, 1],
        [2, 3, 2, 2, 2, 2, 2],
        [1, 2, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2],
    ];
    ROWS.iter()
        .map(|r| {
            let mut c = vec![vec![0usize; 4]; 5];
            for &l in &r[..3] {
                c[0][l - 1] += 1;
            }
            for k in 1..5 {
                c[k][r[2 + k] - 1] += 1;
            }
            c
        })
        .collect()
}

#[test]
fn dawid_skene_replicate_counts_anesthetist_table() {
    // numpy EM on the same counts (majority-vote init, tol 1e-12): iterations 15, converged,
    //   priors = [0.399710904985, 0.466959777836, 0.111107305723, 0.022222011455],
    //   π^(0) row 0 = [0.889279076456, 0.110720923544, 0, 0], row 1 = [0.063668525989, 0.872870663914, 0.063460810097, 0],
    //   π^(1) row 1 = [0.047613479587, 0.571664663049, 0.380721857365, 0],
    //   T[6] = [0.987616710008, 0.012383289992, 0, 0]; patient 38 (index 37) moves from majority class 3 to 2.
    let counts = anesthetist_counts();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene_counts(&counts, 4, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 15);
    for (v, e) in ds.priors.iter().zip([
        0.399710904985,
        0.466959777836,
        0.111107305723,
        0.022222011455,
    ]) {
        assert!((v - e).abs() < 1e-9, "prior {v} vs {e}");
    }
    for (v, e) in ds.confusion[0][0]
        .iter()
        .zip([0.889279076456, 0.110720923544, 0.0, 0.0])
    {
        assert!((v - e).abs() < 1e-9);
    }
    for (v, e) in
        ds.confusion[0][1]
            .iter()
            .zip([0.063668525989, 0.872870663914, 0.063460810097, 0.0])
    {
        assert!((v - e).abs() < 1e-9);
    }
    for (v, e) in
        ds.confusion[1][1]
            .iter()
            .zip([0.047613479587, 0.571664663049, 0.380721857365, 0.0])
    {
        assert!((v - e).abs() < 1e-9);
    }
    for (v, e) in ds.posteriors[6]
        .iter()
        .zip([0.987616710008, 0.012383289992, 0.0, 0.0])
    {
        assert!((v - e).abs() < 1e-9);
    }
    assert_eq!(ds.labels()[37], 1);
    assert!(ds_fixed_point_residual(&counts, &ds, 0.0) < 1e-9);
    assert_eq!(ds.confusion.len(), 5);
    assert_eq!(ds.posteriors.len(), 45);
}

#[test]
fn dawid_skene_validation_and_explicit_init() {
    let t = synthetic_ds();
    let bad_iter = DawidSkeneOpts {
        max_iter: 0,
        ..DawidSkeneOpts::default()
    };
    assert!(dawid_skene(&t, &bad_iter).is_err());
    let bad_tol = DawidSkeneOpts {
        tol: 0.0,
        ..DawidSkeneOpts::default()
    };
    assert!(dawid_skene(&t, &bad_tol).is_err());
    let bad_smooth = DawidSkeneOpts {
        smoothing: -1.0,
        ..DawidSkeneOpts::default()
    };
    assert!(dawid_skene(&t, &bad_smooth).is_err());
    assert!(
        dawid_skene_counts(&[vec![vec![1, 0], vec![1]]], 2, &DawidSkeneOpts::default()).is_err()
    );
    assert!(dawid_skene_counts(&t.to_counts(), 1, &DawidSkeneOpts::default()).is_err());
    // Starting from the truth as hard posteriors converges to the same fixed point.
    let truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let init: Vec<Vec<f64>> = truth
        .iter()
        .map(|&c| (0..3).map(|j| if j == c { 1.0 } else { 0.0 }).collect())
        .collect();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        init: DawidSkeneInit::Posteriors(init),
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene(&t, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.labels(), truth.to_vec());
    let wrong_shape = DawidSkeneOpts {
        init: DawidSkeneInit::Posteriors(vec![vec![1.0, 0.0, 0.0]]),
        ..DawidSkeneOpts::default()
    };
    assert!(dawid_skene(&t, &wrong_shape).is_err());
}

// ── Bradley–Terry ────────────────────────────────────────────────────

#[test]
fn bradley_terry_matches_scipy_mle() {
    // W1 = [[0,3,2,4],[1,0,3,2],[2,1,0,3],[0,2,1,0]].
    // scipy: minimize(negative BT log-likelihood, method='BFGS', gtol=1e-12), normalised to sum 1:
    //   [0.492234862913, 0.209359648702, 0.209359657411, 0.089045830974];
    // Hunter's MM iterated to 1e-15 (66 iterations): [0.492234868958, 0.209359649947, 0.209359649947, 0.089045831147].
    let w = vec![
        vec![0, 3, 2, 4],
        vec![1, 0, 3, 2],
        vec![2, 1, 0, 3],
        vec![0, 2, 1, 0],
    ];
    let bt = bradley_terry(&w, &BradleyTerryOpts::default()).unwrap();
    assert!(bt.converged);
    let mm = [
        0.492234868958,
        0.209359649947,
        0.209359649947,
        0.089045831147,
    ];
    for (p, e) in bt.strengths.iter().zip(mm) {
        assert!((p - e).abs() < 1e-9, "{p} vs {e}");
    }
    let mle = [
        0.492234862913,
        0.209359648702,
        0.209359657411,
        0.089045830974,
    ];
    for (p, e) in bt.strengths.iter().zip(mle) {
        assert!((p - e).abs() < 1e-7);
    }
    assert!((bt.strengths.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // Players 1 and 2 have identical records (4 wins; symmetric results) → equal strengths.
    assert!((bt.strengths[1] - bt.strengths[2]).abs() < 1e-12);
}

#[test]
fn bradley_terry_transitive_symmetric_and_fixed_point() {
    // Transitive: i beats j 3–1 for i < j.  MM to 1e-15 (67 iterations):
    //   [0.487202869863, 0.27069225348, 0.15563392227, 0.086470954387]; scipy MLE agrees to 1e-8.
    let w: Vec<Vec<usize>> = (0..4)
        .map(|i| {
            (0..4)
                .map(|j| {
                    if i == j {
                        0
                    } else if i < j {
                        3
                    } else {
                        1
                    }
                })
                .collect()
        })
        .collect();
    let bt = bradley_terry(&w, &BradleyTerryOpts::default()).unwrap();
    assert!(bt.converged);
    for (p, e) in
        bt.strengths
            .iter()
            .zip([0.487202869863, 0.27069225348, 0.15563392227, 0.086470954387])
    {
        assert!((p - e).abs() < 1e-9);
    }
    assert!(bt.strengths.windows(2).all(|pair| pair[0] > pair[1]));
    // MM fixed point: p_i = W_i / Σ_{j≠i} N_ij / (p_i + p_j), up to the common normalisation.
    let p = &bt.strengths;
    let ratios: Vec<f64> = (0..4)
        .map(|i| {
            let wins: f64 = w[i].iter().sum::<usize>() as f64;
            let denom: f64 = (0..4)
                .filter(|&j| j != i)
                .map(|j| (w[i][j] + w[j][i]) as f64 / (p[i] + p[j]))
                .sum();
            wins / denom / p[i]
        })
        .collect();
    assert!(ratios.iter().all(|r| (r - ratios[0]).abs() < 1e-9));
    // Symmetric: every pair split 2–2 → uniform in one step.
    let s: Vec<Vec<usize>> = (0..3)
        .map(|i| (0..3).map(|j| if i == j { 0 } else { 2 }).collect())
        .collect();
    let bt = bradley_terry(&s, &BradleyTerryOpts::default()).unwrap();
    assert!(bt.converged && bt.iterations == 1);
    assert!(bt.strengths.iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-12));
}

#[test]
fn bradley_terry_validation_and_wins_matrix() {
    let beat = |winner, loser| PairwiseOutcome { winner, loser };
    assert_eq!(
        wins_matrix(&[beat(0, 1), beat(0, 1), beat(1, 2), beat(2, 0)], 3).unwrap(),
        vec![vec![0, 2, 0], vec![0, 0, 1], vec![1, 0, 0]]
    );
    assert!(wins_matrix(&[beat(0, 3)], 3).is_err());
    assert!(wins_matrix(&[beat(1, 1)], 3).is_err());
    // An undefeated player: the beat graph is not strongly connected → no finite MLE.
    let undefeated = wins_matrix(&[beat(0, 1), beat(0, 2), beat(1, 2), beat(2, 1)], 3).unwrap();
    assert!(bradley_terry(&undefeated, &BradleyTerryOpts::default()).is_err());
    let cycle = wins_matrix(&[beat(0, 1), beat(1, 2), beat(2, 0)], 3).unwrap();
    let bt = bradley_terry(&cycle, &BradleyTerryOpts::default()).unwrap();
    assert!(bt.strengths.iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-12));
    assert!(bradley_terry(&[vec![1, 1], vec![1, 0]], &BradleyTerryOpts::default()).is_err());
    assert!(bradley_terry(&[vec![0]], &BradleyTerryOpts::default()).is_err());
    let few = BradleyTerryOpts {
        max_iter: 2,
        tol: 1e-15,
    };
    let w = vec![
        vec![0, 3, 2, 4],
        vec![1, 0, 3, 2],
        vec![2, 1, 0, 3],
        vec![0, 2, 1, 0],
    ];
    let bt = bradley_terry(&w, &few).unwrap();
    assert!(!bt.converged && bt.iterations == 2);
}

// ── Worker quality ───────────────────────────────────────────────────

#[test]
fn worker_accuracy_and_category_metrics() {
    let labels = [
        Some(0),
        Some(1),
        Some(1),
        None,
        Some(2),
        Some(0),
        Some(2),
        Some(1),
    ];
    let gold = [0, 1, 0, 2, 2, 1, 2, 1];
    let acc = worker_accuracy(&labels, &gold).unwrap();
    assert_eq!(
        (acc.correct, acc.answered, acc.accuracy),
        (5, 7, Some(q(5, 7)))
    );
    let silent = worker_accuracy(&[None, None], &[0, 1]).unwrap();
    assert_eq!((silent.answered, silent.accuracy), (0, None));
    assert!(worker_accuracy(&labels, &gold[..3]).is_err());
    // Per category over the 7 answered items (gold 0,1,0,2,1,2,1 vs labels 0,1,1,2,0,2,1):
    //   class 0: tp 1 fp 1 fn 1 → P 1/2 R 1/2 F1 1/2;  class 1: tp 2 fp 1 fn 1 → P 2/3 R 2/3 F1 2/3;
    //   class 2: tp 2 fp 0 fn 0 → P 1 R 1 F1 1.
    let m = category_metrics(&labels, &gold, 3).unwrap();
    assert_eq!(
        (
            m[0].true_positives,
            m[0].false_positives,
            m[0].false_negatives
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (m[0].precision.clone(), m[0].recall.clone(), m[0].f1.clone()),
        (Some(q(1, 2)), Some(q(1, 2)), Some(q(1, 2)))
    );
    assert_eq!(
        (m[1].precision.clone(), m[1].recall.clone(), m[1].f1.clone()),
        (Some(q(2, 3)), Some(q(2, 3)), Some(q(2, 3)))
    );
    assert_eq!(
        (m[2].precision.clone(), m[2].recall.clone(), m[2].f1.clone()),
        (Some(qi(1)), Some(qi(1)), Some(qi(1)))
    );
    // A category never predicted nor present: all None.
    let m = category_metrics(&[Some(0), Some(0)], &[0, 0], 2).unwrap();
    assert_eq!(
        (m[1].precision.clone(), m[1].recall.clone(), m[1].f1.clone()),
        (None, None, None)
    );
    assert!(category_metrics(&[Some(0)], &[2], 2).is_err());
}

#[test]
fn gold_question_screening() {
    // Items 0, 1, 3 are gold; rater 0 gets 3/3, rater 1 gets 2/3, rater 2 answered no gold item.
    let t = LabelTable::from_rows(
        &[
            &[Some(0), Some(0), None],
            &[Some(1), Some(1), None],
            &[Some(0), Some(1), Some(1)],
            &[Some(2), Some(0), None],
        ],
        3,
    )
    .unwrap();
    let gold = [Some(0), Some(1), None, Some(2)];
    assert_eq!(
        gold_screening(&t, &gold, &q(2, 3)).unwrap(),
        vec![Some(true), Some(true), None]
    );
    assert_eq!(
        gold_screening(&t, &gold, &q(9, 10)).unwrap(),
        vec![Some(true), Some(false), None]
    );
    assert!(gold_screening(&t, &gold[..2], &q(1, 2)).is_err());
    assert!(gold_screening(&t, &[Some(7), None, None, None], &q(1, 2)).is_err());
    assert!(gold_screening(&t, &gold, &q(3, 2)).is_err());
}

// ── Confidence intervals for a proportion ────────────────────────────

/// `ci` agrees with the oracle's `(lower, upper)` pair to 1e-9.
fn close(ci: Interval<f64>, expected: (f64, f64)) -> bool {
    (ci.lower - expected.0).abs() < 1e-9 && (ci.upper - expected.1).abs() < 1e-9
}

#[test]
fn proportion_interval_matches_statsmodels() {
    // statsmodels.stats.proportion.proportion_confint(k, n, alpha=0.05, method=…):
    //   (3, 10)   wilson (0.10779126740630104, 0.6032218525388546)  beta (0.06673951117773447, 0.6524528500599973)
    //             agresti_coull (0.10333841792242526, 0.6076747020227304)  normal (0.015974234910674567, 0.5840257650893255)
    //   (7, 10)   wilson (0.39677814746114537, 0.8922087325936989)  beta (0.3475471499400027, 0.9332604888222655)
    //             agresti_coull (0.39232529797726956, 0.8966615820775747)  normal (0.41597423491067453, 0.9840257650893254)
    //   (45, 120) wilson (0.293524297864468, 0.464230493740208)  beta (0.2883136705701342, 0.4680118617139824)
    //             agresti_coull (0.29343900175432563, 0.4643157898503502)  normal (0.2883810109781451, 0.4616189890218549)
    use IntervalMethod::*;
    let ci = |k, n, m| proportion_interval(k, n, 0.95, m).unwrap();
    assert!(close(
        ci(3, 10, Wilson),
        (0.10779126740630104, 0.6032218525388546)
    ));
    assert!(close(
        ci(3, 10, ClopperPearson),
        (0.06673951117773447, 0.6524528500599973)
    ));
    assert!(close(
        ci(3, 10, AgrestiCoull),
        (0.10333841792242526, 0.6076747020227304)
    ));
    assert!(close(
        ci(3, 10, Wald),
        (0.015974234910674567, 0.5840257650893255)
    ));
    assert!(close(
        ci(7, 10, Wilson),
        (0.39677814746114537, 0.8922087325936989)
    ));
    assert!(close(
        ci(7, 10, ClopperPearson),
        (0.3475471499400027, 0.9332604888222655)
    ));
    assert!(close(
        ci(7, 10, AgrestiCoull),
        (0.39232529797726956, 0.8966615820775747)
    ));
    assert!(close(
        ci(7, 10, Wald),
        (0.41597423491067453, 0.9840257650893254)
    ));
    assert!(close(
        ci(45, 120, Wilson),
        (0.293524297864468, 0.464230493740208)
    ));
    assert!(close(
        ci(45, 120, ClopperPearson),
        (0.2883136705701342, 0.4680118617139824)
    ));
    assert!(close(
        ci(45, 120, AgrestiCoull),
        (0.29343900175432563, 0.4643157898503502)
    ));
    assert!(close(
        ci(45, 120, Wald),
        (0.2883810109781451, 0.4616189890218549)
    ));
}

#[test]
fn proportion_interval_edge_cases_match_statsmodels() {
    // proportion_confint(0, 10, 0.05, method=…): wilson (0.0, 0.27753279986288926)  beta (0.0, 0.30849710781876083)
    //   agresti_coull (0.0, 0.3208873057505458)  normal (0.0, 0.0)
    // proportion_confint(10, 10, 0.05, method=…): wilson (0.7224672001371106, 1.0)  beta (0.6915028921812392, 1.0)
    //   agresti_coull (0.6791126942494543, 1.0)  normal (1.0, 1.0)
    use IntervalMethod::*;
    let ci = |k, n, m| proportion_interval(k, n, 0.95, m).unwrap();
    assert!(close(ci(0, 10, Wilson), (0.0, 0.27753279986288926)));
    assert!(close(ci(0, 10, ClopperPearson), (0.0, 0.30849710781876083)));
    assert!(close(ci(0, 10, AgrestiCoull), (0.0, 0.3208873057505458)));
    assert!(close(ci(0, 10, Wald), (0.0, 0.0)));
    assert!(close(ci(10, 10, Wilson), (0.7224672001371106, 1.0)));
    assert!(close(ci(10, 10, ClopperPearson), (0.6915028921812392, 1.0)));
    assert!(close(ci(10, 10, AgrestiCoull), (0.6791126942494543, 1.0)));
    assert!(close(ci(10, 10, Wald), (1.0, 1.0)));
    assert!(proportion_interval(3, 0, 0.95, Wilson).is_err());
    assert!(proportion_interval(11, 10, 0.95, Wilson).is_err());
    assert!(proportion_interval(3, 10, 1.0, Wilson).is_err());
    assert!(proportion_interval(3, 10, 0.0, Wilson).is_err());
}

// ── Cross-module: agreement of the synthetic Dawid–Skene raters ──────

#[test]
fn label_table_and_rating_table_agree_on_the_same_data() {
    // The five synthetic raters as a RatingTable: Fleiss' κ needs a complete table (rater 3 abstains once),
    // Krippendorff's α does not.  krippendorff.alpha(reliability_data=R, level_of_measurement='nominal')
    // on the same 5 × 14 matrix (np.nan for the abstention) = 0.6386054421768708 = 751/1176.
    let t = synthetic_ds();
    let rows: Vec<Vec<Option<Q>>> = t
        .rows()
        .iter()
        .map(|r| r.iter().map(|l| l.map(|l| qi(l as i64))).collect())
        .collect();
    let rt = RatingTable::new(rows).unwrap();
    assert!(fleiss_kappa_ratings(&rt).is_err());
    assert_eq!(
        krippendorff_alpha(&rt, Level::Nominal).unwrap(),
        q(751, 1176)
    );
    let (r0, r1) = rt.paired_ratings(0, 1).unwrap();
    assert_eq!(percent_agreement(&r0, &r1).unwrap(), q(11, 14));
}
