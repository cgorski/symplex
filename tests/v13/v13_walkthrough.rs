//! symplex 0.13 — the book's "Analysing rater and response data" walk-through
//! as a test, so every number printed in `book/src/guide/response-analysis.md`
//! is produced by the library (and re-checked against the oracles cited in
//! the per-module test files).

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::aggregation::{
    DawidSkeneOpts, LabelTable, dawid_skene, majority_votes, worker_accuracy,
};
use symplex::stats::agreement::{
    IccForm, Level, RatingTable, cohen_kappa, fleiss_kappa_ratings, icc, kendall_w,
    krippendorff_alpha, percent_agreement,
};
use symplex::stats::data::{self, Ddof, Q, QuantileMethod};
use symplex::stats::estimation::{IntervalMethod, proportion_interval};
use symplex::stats::hypothesis::{
    Alternative, RankMethod, benjamini_hochberg, binomial_test, chi_square_independence, cohens_d,
    counts, fisher_exact, mann_whitney_u, t_test_two_sample,
};

/// Five raters × eight items, categories 0..3, two missing cells.
fn ratings() -> RatingTable {
    RatingTable::from_i64_missing(&[
        &[Some(0), Some(0), Some(0), Some(0), Some(1)],
        &[Some(1), Some(1), Some(1), Some(2), Some(1)],
        &[Some(2), Some(2), Some(2), Some(2), Some(2)],
        &[Some(0), Some(1), Some(0), None, Some(0)],
        &[Some(1), Some(1), Some(2), Some(1), Some(1)],
        &[Some(2), Some(1), Some(2), Some(2), None],
        &[Some(0), Some(0), Some(1), Some(0), Some(0)],
        &[Some(1), Some(2), Some(1), Some(1), Some(1)],
    ])
    .unwrap()
}

#[test]
fn agreement_walkthrough() {
    let table = ratings();
    // Two raters (columns 0 and 1) on the items both rated.
    let a: Vec<Q> = data::from_i64(&[0, 1, 2, 0, 1, 2, 0, 1]);
    let b: Vec<Q> = data::from_i64(&[0, 1, 2, 1, 1, 1, 0, 2]);
    assert_eq!(percent_agreement(&a, &b).unwrap(), q(5, 8));
    let kappa = cohen_kappa(&a, &b).unwrap();
    // statsmodels cohens_kappa(confusion) — see v13_agreement.rs for the
    // oracle protocol; observed 5/8, expected (3·2 + 3·4 + 2·2)/64 = 22/64.
    assert_eq!(kappa.observed, q(5, 8));
    assert_eq!(kappa.expected, q(11, 32));
    assert_eq!(kappa.kappa, q(3, 7));
    // All five raters, with the missing cells: krippendorff.alpha(rel, 'nominal')
    // = 0.4524312896405921 = 214/473; 'ordinal' = 0.6901913875598087 = 577/836
    // (the near-misses between adjacent categories count for less).
    let alpha = krippendorff_alpha(&table, Level::Nominal).unwrap();
    assert_eq!(alpha, q(214, 473));
    let alpha_ord = krippendorff_alpha(&table, Level::Ordinal).unwrap();
    assert_eq!(alpha_ord, q(577, 836));
    // Fleiss' κ needs complete rows: the six complete items.
    let complete = RatingTable::from_i64(&[
        &[0, 0, 0, 0, 1],
        &[1, 1, 1, 2, 1],
        &[2, 2, 2, 2, 2],
        &[1, 1, 2, 1, 1],
        &[0, 0, 1, 0, 0],
        &[1, 2, 1, 1, 1],
    ])
    .unwrap();
    // statsmodels fleiss_kappa(aggregate_raters(complete)[0]) = 0.4791666… = 23/48
    let fleiss = fleiss_kappa_ratings(&complete).unwrap();
    assert_eq!(fleiss, q(23, 48));
    // Treat the categories as a scale: ICC(2,1) and Kendall's W.
    let icc21 = icc(&complete, IccForm::Icc2Single).unwrap();
    assert!(icc21 > q(1, 2) && icc21 < qi(1), "{icc21}");
    let w = kendall_w(&complete).unwrap();
    assert!(w > q(1, 2) && w <= qi(1), "{w}");
    println!(
        "agreement: pct {} κ {} α_nom {} α_ord {} fleiss {} icc21 {} W {}",
        percent_agreement(&a, &b).unwrap(),
        kappa.kappa,
        alpha,
        alpha_ord,
        fleiss,
        icc21,
        w
    );
}

#[test]
fn aggregation_walkthrough() {
    let labels = LabelTable::from_rows(
        &[
            &[Some(0), Some(0), Some(0), Some(0), Some(1)],
            &[Some(1), Some(1), Some(1), Some(2), Some(1)],
            &[Some(2), Some(2), Some(2), Some(2), Some(2)],
            &[Some(0), Some(1), Some(0), None, Some(0)],
            &[Some(1), Some(1), Some(2), Some(1), Some(1)],
            &[Some(2), Some(1), Some(2), Some(2), None],
            &[Some(0), Some(0), Some(1), Some(0), Some(0)],
            &[Some(1), Some(2), Some(1), Some(1), Some(1)],
        ],
        3,
    )
    .unwrap();
    let votes = majority_votes(&labels);
    let winners: Vec<Option<usize>> = votes.iter().map(|v| v.winner).collect();
    assert_eq!(
        winners,
        [
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(1)
        ]
    );
    // Dawid–Skene agrees with the majority here and reports the raters'
    // confusion matrices; rater 3 (column 3) is the least reliable one.
    let ds = dawid_skene(&labels, &DawidSkeneOpts::default()).unwrap();
    assert!(ds.converged);
    assert_eq!(
        ds.labels(),
        winners.iter().map(|w| w.unwrap()).collect::<Vec<_>>()
    );
    // Worker quality against the majority as gold: rater 1 got 5 of 8
    // (Python: sum(x == y for x, y in zip(majority, rater1)) = 5).
    let gold: Vec<usize> = winners.iter().map(|w| w.unwrap()).collect();
    let rater1: Vec<Option<usize>> = labels.rater(1).unwrap();
    let acc = worker_accuracy(&rater1, &gold).unwrap();
    assert_eq!(acc.correct, 5);
    assert_eq!(acc.answered, 8);
    assert_eq!(acc.accuracy, Some(q(5, 8)));
    // An exact 95% Clopper–Pearson interval for 5/8: statsmodels
    // proportion_confint(5, 8, alpha=0.05, method='beta') =
    // (0.2448632163665516, 0.9147665858627465).
    let ci = proportion_interval(5, 8, 0.95, IntervalMethod::ClopperPearson).unwrap();
    assert!(
        (ci.lower - 0.244_863_216_366_551_6).abs() < 1e-9
            && (ci.upper - 0.914_766_585_862_746_5).abs() < 1e-9,
        "{ci}"
    );
    let wilson = proportion_interval(5, 8, 0.95, IntervalMethod::Wilson).unwrap();
    assert!(wilson.lower < 0.625 && wilson.upper > 0.625);
}

#[test]
fn comparison_walkthrough() -> Result<(), SymplexError> {
    let ctx = Context::new();
    // Response times (seconds) of two groups of respondents (no ties, so the
    // exact rank test applies).
    let fast = data::from_i64(&[12, 15, 11, 14, 13, 16, 10, 17]);
    let slow = data::from_i64(&[18, 22, 19, 25, 20, 21, 23, 24]);
    // statistics.mean/variance/median on Fractions: 27/2, 6, 43/2; numpy.quantile(slow, 0.75) = 93/4
    assert_eq!(data::mean(&fast)?, q(27, 2));
    assert_eq!(data::variance(&fast, Ddof::Sample)?, qi(6));
    assert_eq!(data::median(&slow)?, q(43, 2));
    assert_eq!(
        data::quantile(&slow, &q(3, 4), QuantileMethod::Inclusive)?,
        q(93, 4)
    );
    // Welch's t: statistic exact (rational × √rational), p an exact expression.
    // scipy: ttest_ind(fast, slow, equal_var=False) → t = -6.531972647421809,
    // p = 1.3298737271301488e-05.
    let t = t_test_two_sample(&ctx, &fast, &slow, false, Alternative::TwoSided)?;
    assert!((t.statistic_f64()? - -6.531_972_647_421_809).abs() < 1e-12);
    assert!(
        (t.p_value_f64()? - 1.329_873_727_130_148_8e-5).abs() < 1e-15,
        "{}",
        t.p_value
    );
    let d = cohens_d(&ctx, &fast, &slow, true)?;
    assert!(d.eval_f64()? < -3.0, "{d}");
    // Mann–Whitney with the exact null distribution: the samples do not
    // overlap, so U = 0 and scipy's mannwhitneyu(fast, slow, method='exact')
    // gives p = 0.0001554001554001554 = 1/6435 = 2/C(16, 8).
    let u = mann_whitney_u(&ctx, &fast, &slow, Alternative::TwoSided, RankMethod::Exact)?;
    assert_eq!(u.statistic, ctx.int(0));
    assert_eq!(u.p_value, ctx.rational(1, 6435));
    // Approval counts by group: 2×2 exact (scipy fisher_exact p =
    // 0.01150621201656047) and χ² with Yates (chi2_contingency p = 0.01205961617749023).
    let approved = [[30usize, 10], [18, 22]];
    let f = fisher_exact(&ctx, approved, Alternative::TwoSided)?;
    assert!(
        (f.p_value_f64()? - 0.011_506_212_016_560_47).abs() < 1e-12,
        "{}",
        f.p_value
    );
    let table = counts(&[&[30, 10], &[18, 22]]);
    let chi = chi_square_independence(&ctx, &table, true)?;
    assert_eq!(chi.df, 1);
    assert!((chi.p_value.eval_f64()? - 0.012_059_616_177_490_23).abs() < 1e-12);
    // A worker who got 14 of 20 gold questions right, against chance 1/3:
    // scipy binomtest(14, 20, 1/3, alternative='greater').pvalue =
    // 0.0008788065585934114; exactly, Σ_{i≥14} C(20,i)(1/3)^i(2/3)^(20−i) =
    // 1021403/1162261467 (Python Fractions — note that
    // Fraction(float).limit_denominator() gives a *different* rational with
    // the same leading digits; exact tests need exact oracles).
    let bt = binomial_test(&ctx, 14, 20, &q(1, 3), Alternative::Greater)?;
    assert_eq!(bt.p_value_exact(), Some(q(1_021_403, 1_162_261_467)));
    // Many workers screened at once: control the false-discovery rate.
    let adj = benjamini_hochberg(&[0.001, 0.02, 0.03, 0.2, 0.8], 0.05)?;
    assert_eq!(adj.reject, vec![true, true, true, false, false]);
    Ok(())
}
