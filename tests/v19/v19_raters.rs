//! symplex 0.20 — Bayesian rater-quality models (`stats::aggregation`:
//! MAP Dawid–Skene, MACE, confusion against gold, posterior entropy) and
//! `stats::markov` (`Display`, `limiting_distribution`).
//!
//! Reference values come from two ~40-line numpy EM scripts written
//! independently of the library with the same rules (majority-vote start,
//! posterior-mode M-step / add-δ M-step, log-space E-step, stop when the
//! posteriors move by less than `tol`), run under `symplex/.venv/bin/python`
//! (numpy 2.5.3) and rounded to 12 digits.  The MACE script's E- and M-step
//! were checked against one iteration of the published `mace.py`
//! (github.com/dirkhovy/MACE, `--em` mode) on the same table: the two agree
//! to 2e-16 in the posteriors, θ and ξ.

// Index loops over parallel tables mirror the E/M equations; the indices are the point.
#![allow(clippy::needless_range_loop)]

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::Rng;
use symplex::stats::aggregation::*;
use symplex::stats::markov::MarkovChain;

// ── Data sets ────────────────────────────────────────────────────────

/// The synthetic Dawid–Skene table of `tests/v13/v13_agreement.rs`:
/// 14 items, 5 raters, 3 classes; rater 0 is perfect, item 12 is tied
/// 2–2 under majority vote (rater 3 abstains).
fn synthetic_ds() -> LabelTable {
    let raters: [[Option<usize>; 14]; 5] = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1].map(Some),
        [0, 1, 2, 1, 1, 2, 0, 2, 2, 0, 1, 2, 1, 1].map(Some),
        [0, 1, 2, 0, 1, 1, 0, 1, 2, 0, 1, 2, 1, 1].map(Some),
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
        [0, 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 1, 0, 1].map(Some),
    ];
    let rows: Vec<Vec<Option<usize>>> = (0..14)
        .map(|i| raters.iter().map(|r| r[i]).collect())
        .collect();
    LabelTable::new(rows, 3).unwrap()
}

/// Three perfect raters and one who always answers `0`, on items of
/// classes `0, 0, 1, 1, 2, 2`.
fn constant_rater() -> LabelTable {
    LabelTable::complete(
        &[
            &[0, 0, 0, 0],
            &[0, 0, 0, 0],
            &[1, 1, 1, 0],
            &[1, 1, 1, 0],
            &[2, 2, 2, 0],
            &[2, 2, 2, 0],
        ],
        3,
    )
    .unwrap()
}

/// The MACE reference table: 8 items, 4 raters, 3 labels; rater 3 mostly
/// answers `2`, item 6 has no labels, item 7 a single one.
fn mace_table() -> LabelTable {
    LabelTable::from_rows(
        &[
            &[Some(0), Some(0), Some(1), Some(2)],
            &[Some(1), Some(1), Some(1), Some(2)],
            &[Some(2), Some(2), None, Some(2)],
            &[Some(0), Some(0), Some(0), Some(2)],
            &[Some(1), Some(2), Some(1), Some(1)],
            &[Some(2), Some(2), Some(2), Some(0)],
            &[None, None, None, None],
            &[None, Some(0), None, None],
        ],
        3,
    )
    .unwrap()
}

/// A data set drawn from the MACE generative model with the crate's
/// `Rng`: 200 items, 20 raters, 3 labels, each rater labelling each item
/// with probability 0.6.  Raters 0–5 have `θ = 0.9`, 6–11 `θ = 0.6`
/// (both spam uniformly), 12–19 `θ = 0.1` and spam `(0.9, 0.05, 0.05)`.
struct MaceSample {
    table: LabelTable,
    truth: Vec<usize>,
    theta: Vec<f64>,
}

fn synthetic_mace() -> MaceSample {
    let mut rng = Rng::new(20_260_921);
    let (n_items, n_raters, k) = (200, 20, 3);
    let theta: Vec<f64> = (0..n_raters)
        .map(|r| {
            if r < 6 {
                0.9
            } else if r < 12 {
                0.6
            } else {
                0.1
            }
        })
        .collect();
    let spam = |r: usize| -> [f64; 3] {
        if r < 12 {
            [1.0 / 3.0; 3]
        } else {
            [0.9, 0.05, 0.05]
        }
    };
    let mut truth = Vec::with_capacity(n_items);
    let mut rows = Vec::with_capacity(n_items);
    for _ in 0..n_items {
        let t = rng.below(k);
        truth.push(t);
        let row: Vec<Option<usize>> = (0..n_raters)
            .map(|r| {
                if rng.next_f64() >= 0.6 {
                    return None;
                }
                if rng.next_f64() < theta[r] {
                    return Some(t);
                }
                let u = rng.next_f64();
                let xi = spam(r);
                let mut acc = 0.0;
                for (l, p) in xi.iter().enumerate() {
                    acc += p;
                    if u < acc {
                        return Some(l);
                    }
                }
                Some(k - 1)
            })
            .collect();
        rows.push(row);
    }
    MaceSample {
        table: LabelTable::new(rows, k).unwrap(),
        truth,
        theta,
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn assert_rows_close(got: &[Vec<f64>], want: &[Vec<f64>], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{what}[{i}]: length");
        for (j, (a, b)) in g.iter().zip(w).enumerate() {
            assert!(close(*a, *b, tol), "{what}[{i}][{j}] = {a}, want {b}");
        }
    }
}

// ── MAP Dawid–Skene: the M-step and E-step written out ───────────────

/// The MAP M-step `(count + α − 1) / Σ (count + α − 1)` and the E-step,
/// independently of the library, as a residual against the returned
/// estimates.
fn ds_map_residual(table: &LabelTable, priors: &DawidSkenePriors, ds: &DawidSkene) -> f64 {
    let counts = table.to_counts();
    let n_items = counts.len();
    let n_raters = table.n_raters();
    let j = table.n_categories();
    let mut worst = 0.0f64;
    // p_j = (Σ_i T_ij + α_j − 1) / (I + Σ_j (α_j − 1))
    let class_den: f64 = n_items as f64
        + priors
            .class_prior_alpha
            .iter()
            .map(|a| a - 1.0)
            .sum::<f64>();
    for c in 0..j {
        let p = (ds.posteriors.iter().map(|r| r[c]).sum::<f64>() + priors.class_prior_alpha[c]
            - 1.0)
            / class_den;
        worst = worst.max((p - ds.priors[c]).abs());
    }
    // π^(k)_jl = (Σ_i T_ij n_ikl + β_jl − 1) / Σ_l (…)
    for k in 0..n_raters {
        for c in 0..j {
            let num: Vec<f64> = (0..j)
                .map(|l| {
                    counts
                        .iter()
                        .zip(&ds.posteriors)
                        .map(|(it, t)| t[c] * it[k][l] as f64)
                        .sum::<f64>()
                        + priors.confusion_alpha[c][l]
                        - 1.0
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

#[test]
fn symmetric_priors_have_the_documented_shape() {
    let p = DawidSkenePriors::symmetric(3, 2.0, 5.0, 1.5);
    assert_eq!(p.class_prior_alpha, vec![2.0; 3]);
    assert_eq!(
        p.confusion_alpha,
        vec![
            vec![5.0, 1.5, 1.5],
            vec![1.5, 5.0, 1.5],
            vec![1.5, 1.5, 5.0]
        ]
    );
}

#[test]
fn map_with_flat_priors_is_bit_identical_to_the_mle() {
    let t = synthetic_ds();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        ..DawidSkeneOpts::default()
    };
    let mle = dawid_skene(&t, &opts).unwrap();
    let map = dawid_skene_map(&t, &DawidSkenePriors::symmetric(3, 1.0, 1.0, 1.0), &opts).unwrap();
    assert_eq!(map, mle);
    assert_eq!(map.iterations, 6);
}

#[test]
fn map_with_flat_priors_and_smoothing_is_bit_identical_to_the_smoothed_mle() {
    let t = synthetic_ds();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        smoothing: 0.01,
        ..DawidSkeneOpts::default()
    };
    let mle = dawid_skene(&t, &opts).unwrap();
    let map = dawid_skene_map(&t, &DawidSkenePriors::symmetric(3, 1.0, 1.0, 1.0), &opts).unwrap();
    assert_eq!(map, mle);
    assert_eq!(map.iterations, 9);
}

#[test]
fn map_prior_alpha_is_smoothing_plus_one() {
    // β = 1.01 everywhere is a pseudo-count of 0.01 per cell (up to the rounding of 1.01 − 1).
    let t = synthetic_ds();
    let smoothed = dawid_skene(
        &t,
        &DawidSkeneOpts {
            tol: 1e-12,
            smoothing: 0.01,
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    let map = dawid_skene_map(
        &t,
        &DawidSkenePriors::symmetric(3, 1.0, 1.01, 1.01),
        &DawidSkeneOpts {
            tol: 1e-12,
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    assert_eq!(map.iterations, smoothed.iterations);
    assert_rows_close(&map.posteriors, &smoothed.posteriors, 1e-12, "posteriors");
    for (a, b) in map.confusion.iter().zip(&smoothed.confusion) {
        assert_rows_close(a, b, 1e-12, "confusion");
    }
}

#[test]
fn map_matches_the_numpy_reference() {
    // ref_map_ds.py case A: symmetric(3, 2, 3, 1.5), majority-vote start, tol 1e-12:
    //   iterations 25, converged, log-likelihood −42.052650560256.
    let t = synthetic_ds();
    let priors = DawidSkenePriors::symmetric(3, 2.0, 3.0, 1.5);
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        ..DawidSkeneOpts::default()
    };
    let ds = dawid_skene_map(&t, &priors, &opts).unwrap();
    assert!(ds.converged);
    assert_eq!(ds.iterations, 25);
    assert_eq!(ds.labels(), vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]);
    for (v, e) in ds
        .priors
        .iter()
        .zip([0.348257614626, 0.357344932604, 0.294397452771])
    {
        assert!(close(*v, e, 1e-9), "prior {v} vs {e}");
    }
    for (v, e) in ds.posteriors[12]
        .iter()
        .zip([0.919928045832, 0.075594111294, 0.004477842874])
    {
        assert!(close(*v, e, 1e-9), "T[12] {v} vs {e}");
    }
    for (v, e) in ds.posteriors[3]
        .iter()
        .zip([0.999841215183, 0.000120453733, 3.8331084e-05])
    {
        assert!(close(*v, e, 1e-9), "T[3] {v} vs {e}");
    }
    let pi1 = vec![
        vec![0.631178439466, 0.305589457931, 0.063232102603],
        vec![0.061927682276, 0.752107453728, 0.185964863997],
        vec![0.07148994292, 0.072300904508, 0.856209152572],
    ];
    let pi3 = vec![
        vec![0.714210358339, 0.071446496778, 0.214343144883],
        vec![0.062521964488, 0.749806151349, 0.187671884163],
        vec![0.071436445131, 0.071702945308, 0.856860609561],
    ];
    assert_rows_close(&ds.confusion[1], &pi1, 1e-9, "π^(1)");
    assert_rows_close(&ds.confusion[3], &pi3, 1e-9, "π^(3)");
    assert!(close(ds.log_likelihood, -42.052650560256, 1e-9));
    // Deterministic.
    assert_eq!(dawid_skene_map(&t, &priors, &opts).unwrap(), ds);
}

#[test]
fn map_is_a_fixed_point_of_one_e_step_and_m_step() {
    let t = synthetic_ds();
    let priors = DawidSkenePriors::symmetric(3, 2.0, 3.0, 1.5);
    let ds = dawid_skene_map(
        &t,
        &priors,
        &DawidSkeneOpts {
            tol: 1e-14,
            max_iter: 5000,
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    assert!(ds.converged);
    assert!(ds_map_residual(&t, &priors, &ds) < 1e-12);
}

#[test]
fn map_single_label_rater_gets_the_smoothed_row_and_the_mle_a_degenerate_one() {
    // ref_map_ds.py case C.  MLE: rater 3's rows are all (1, 0, 0) — it only ever said 0.
    let t = constant_rater();
    let opts = DawidSkeneOpts {
        tol: 1e-12,
        ..DawidSkeneOpts::default()
    };
    let mle = dawid_skene(&t, &opts).unwrap();
    for row in &mle.confusion[3] {
        assert_eq!(row, &vec![1.0, 0.0, 0.0]);
    }
    // MAP with symmetric(3, 1, 3, 2): pseudo-counts 2 on the diagonal, 1 off it.
    // Row c of rater 3 is (n_c [c = 0] + β_c0 − 1, β_c1 − 1, β_c2 − 1) / (n_c + Σ_l (β_cl − 1)),
    // n_c = Σ_i T_ic the posterior mass of class c: with n_0 = 2.048409416977 the reference
    // gives row 0 = (2.0484 + 2, 1, 1) / (2.0484 + 4) = (0.669334553579, 0.16533272321, 0.16533272321)
    // and rows 1, 2 = (n_c + 1, 2, 1) / (n_c + 4) with n_1 = n_2 = 1.975795291512.
    let priors = DawidSkenePriors::symmetric(3, 1.0, 3.0, 2.0);
    let map = dawid_skene_map(&t, &priors, &opts).unwrap();
    assert!(map.converged);
    assert_eq!(map.iterations, 16);
    let n: Vec<f64> = (0..3)
        .map(|c| map.posteriors.iter().map(|r| r[c]).sum::<f64>())
        .collect();
    assert!(close(n[0], 2.048409416977, 1e-9));
    assert!(close(n[1], 1.975795291512, 1e-9));
    let by_hand = vec![
        vec![
            (n[0] + 2.0) / (n[0] + 4.0),
            1.0 / (n[0] + 4.0),
            1.0 / (n[0] + 4.0),
        ],
        vec![
            (n[1] + 1.0) / (n[1] + 4.0),
            2.0 / (n[1] + 4.0),
            1.0 / (n[1] + 4.0),
        ],
        vec![
            (n[2] + 1.0) / (n[2] + 4.0),
            1.0 / (n[2] + 4.0),
            2.0 / (n[2] + 4.0),
        ],
    ];
    assert_rows_close(&map.confusion[3], &by_hand, 1e-12, "rater 3 by hand");
    let reference = vec![
        vec![0.669334553579, 0.16533272321, 0.16533272321],
        vec![0.497974770946, 0.334683486036, 0.167341743018],
        vec![0.497974770946, 0.167341743018, 0.334683486036],
    ];
    assert_rows_close(&map.confusion[3], &reference, 1e-9, "rater 3 vs numpy");
    let perfect = vec![
        vec![0.652936205416, 0.173531897292, 0.173531897292],
        vec![0.171590090225, 0.655139757455, 0.17327015232],
        vec![0.171590090225, 0.17327015232, 0.655139757455],
    ];
    assert_rows_close(&map.confusion[0], &perfect, 1e-9, "rater 0 vs numpy");
    assert!(close(map.log_likelihood, -17.591659167645, 1e-9));
    assert_eq!(map.labels(), vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn map_strong_priors_pull_the_confusion_toward_the_prior_mean() {
    // With β − 1 = (1000, 500, 500) on every row the six counts barely matter:
    // every row of every rater is within 3e-3 of (0.5, 0.25, 0.25).
    let t = constant_rater();
    let map = dawid_skene_map(
        &t,
        &DawidSkenePriors::symmetric(3, 1.0, 1001.0, 501.0),
        &DawidSkeneOpts::default(),
    )
    .unwrap();
    for rater in &map.confusion {
        for (c, row) in rater.iter().enumerate() {
            for (l, v) in row.iter().enumerate() {
                let prior_mean = if c == l { 0.5 } else { 0.25 };
                assert!(close(*v, prior_mean, 3e-3), "π[{c}][{l}] = {v}");
            }
        }
    }
    // …whereas the class prior with α = (1, 1, 1) is the plain posterior average.
    let s: f64 = map.priors.iter().sum();
    assert!(close(s, 1.0, 1e-12));
}

#[test]
fn map_class_prior_follows_the_formula() {
    // α = (4, 1, 1): p_j = (Σ_i T_ij + α_j − 1) / (I + 3).
    let t = synthetic_ds();
    let priors = DawidSkenePriors {
        class_prior_alpha: vec![4.0, 1.0, 1.0],
        confusion_alpha: vec![vec![1.0; 3]; 3],
    };
    let map = dawid_skene_map(&t, &priors, &DawidSkeneOpts::default()).unwrap();
    for c in 0..3 {
        let mass: f64 = map.posteriors.iter().map(|r| r[c]).sum();
        let expect = (mass + priors.class_prior_alpha[c] - 1.0) / 17.0;
        assert!(close(map.priors[c], expect, 1e-12));
    }
    assert!(map.priors[0] > map.priors[1]);
}

#[test]
fn map_rejects_alpha_below_one_and_bad_shapes() {
    let t = synthetic_ds();
    let opts = DawidSkeneOpts::default();
    assert!(dawid_skene_map(&t, &DawidSkenePriors::symmetric(3, 0.5, 1.0, 1.0), &opts).is_err());
    assert!(dawid_skene_map(&t, &DawidSkenePriors::symmetric(3, 1.0, 1.0, 0.999), &opts).is_err());
    assert!(
        dawid_skene_map(
            &t,
            &DawidSkenePriors::symmetric(3, 1.0, f64::NAN, 1.0),
            &opts
        )
        .is_err()
    );
    assert!(dawid_skene_map(&t, &DawidSkenePriors::symmetric(2, 1.0, 1.0, 1.0), &opts).is_err());
    let ragged = DawidSkenePriors {
        class_prior_alpha: vec![1.0; 3],
        confusion_alpha: vec![vec![1.0; 3], vec![1.0; 2], vec![1.0; 3]],
    };
    assert!(dawid_skene_map(&t, &ragged, &opts).is_err());
    let one_class = LabelTable::complete(&[&[0, 0]], 1).unwrap();
    assert!(
        dawid_skene_map(
            &one_class,
            &DawidSkenePriors::symmetric(1, 1.0, 1.0, 1.0),
            &opts
        )
        .is_err()
    );
}

#[test]
fn map_validates_options_and_explicit_initial_posteriors() {
    let t = synthetic_ds();
    let priors = DawidSkenePriors::symmetric(3, 1.0, 2.0, 1.0);
    let bad = |opts: DawidSkeneOpts| dawid_skene_map(&t, &priors, &opts).is_err();
    assert!(bad(DawidSkeneOpts {
        max_iter: 0,
        ..DawidSkeneOpts::default()
    }));
    assert!(bad(DawidSkeneOpts {
        tol: 0.0,
        ..DawidSkeneOpts::default()
    }));
    assert!(bad(DawidSkeneOpts {
        smoothing: -1.0,
        ..DawidSkeneOpts::default()
    }));
    assert!(bad(DawidSkeneOpts {
        init: DawidSkeneInit::Posteriors(vec![vec![1.0; 3]; 13]),
        ..DawidSkeneOpts::default()
    }));
    assert!(bad(DawidSkeneOpts {
        init: DawidSkeneInit::Posteriors(vec![vec![0.0; 3]; 14]),
        ..DawidSkeneOpts::default()
    }));
    let err = dawid_skene_map(
        &t,
        &priors,
        &DawidSkeneOpts {
            init: DawidSkeneInit::Posteriors(vec![vec![-1.0, 1.0, 1.0]; 14]),
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("dawid_skene_map"), "{err}");
    // Starting from the truth as hard posteriors reaches the same fixed point.
    let truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let init: Vec<Vec<f64>> = truth
        .iter()
        .map(|&c| (0..3).map(|l| if l == c { 5.0 } else { 0.0 }).collect())
        .collect();
    let from_truth = dawid_skene_map(
        &t,
        &priors,
        &DawidSkeneOpts {
            tol: 1e-13,
            init: DawidSkeneInit::Posteriors(init),
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    let from_votes = dawid_skene_map(
        &t,
        &priors,
        &DawidSkeneOpts {
            tol: 1e-13,
            ..DawidSkeneOpts::default()
        },
    )
    .unwrap();
    assert_rows_close(
        &from_truth.posteriors,
        &from_votes.posteriors,
        1e-9,
        "posteriors",
    );
}

// ── MACE: the E-step and M-step written out ──────────────────────────

/// One E-step from the returned `(θ, ξ)` followed by one M-step,
/// independently of the library: the largest change in the posteriors,
/// `θ` and `ξ`.
fn mace_residual(table: &LabelTable, m: &Mace, smoothing: f64) -> f64 {
    let k = table.n_categories();
    let n_raters = table.n_raters();
    let mut worst = 0.0f64;
    let mut copied = vec![0.0f64; n_raters];
    let mut spammed = vec![0.0f64; n_raters];
    let mut spam_counts = vec![vec![0.0f64; k]; n_raters];
    for (row, t) in table.rows().iter().zip(&m.posteriors) {
        // T_it ∝ Π_r (θ_r [a_ir = t] + (1 − θ_r) ξ_r(a_ir))
        let mut w = vec![1.0f64; k];
        for (r, a) in row.iter().enumerate() {
            let Some(a) = a else { continue };
            let spam = (1.0 - m.competence[r]) * m.spam_distribution[r][*a];
            for (c, v) in w.iter_mut().enumerate() {
                *v *= if c == *a {
                    m.competence[r] + spam
                } else {
                    spam
                };
            }
        }
        let s: f64 = w.iter().sum();
        let gamma: Vec<f64> = if s > 0.0 {
            w.iter().map(|v| v / s).collect()
        } else {
            vec![1.0 / k as f64; k]
        };
        for (a, b) in gamma.iter().zip(t) {
            worst = worst.max((a - b).abs());
        }
        // ρ_ir = Σ_t T_it P(S_ir = 1 | T_i = t, a_ir): 1 off the label, the spam odds on it.
        for (r, a) in row.iter().enumerate() {
            let Some(a) = a else { continue };
            let theta = m.competence[r];
            let spam = (1.0 - theta) * m.spam_distribution[r][*a];
            let rho = (1.0 - gamma[*a]) + gamma[*a] * spam / (theta + spam);
            copied[r] += 1.0 - rho;
            spammed[r] += rho;
            spam_counts[r][*a] += rho;
        }
    }
    for r in 0..n_raters {
        let theta = (copied[r] + smoothing) / (copied[r] + spammed[r] + 2.0 * smoothing);
        worst = worst.max((theta - m.competence[r]).abs());
        for l in 0..k {
            let xi = (spam_counts[r][l] + smoothing) / (spammed[r] + k as f64 * smoothing);
            worst = worst.max((xi - m.spam_distribution[r][l]).abs());
        }
    }
    worst
}

#[test]
fn mace_matches_the_numpy_reference_and_the_published_e_m_steps() {
    // ref_mace.py: smoothing 0.1, majority-vote start, tol 1e-12: iterations 76, converged,
    //   θ = [0.973819343735, 0.749512992596, 0.654224973357, 0.178774263214],
    //   log-likelihood −17.526476814226; mace.py's e_step/m_step agree to 2e-16.
    let t = mace_table();
    let opts = MaceOpts {
        tol: 1e-12,
        ..MaceOpts::default()
    };
    let m = mace(&t, &opts).unwrap();
    assert!(m.converged);
    assert_eq!(m.iterations, 76);
    for (v, e) in m.competence.iter().zip([
        0.973819343735,
        0.749512992596,
        0.654224973357,
        0.178774263214,
    ]) {
        assert!(close(*v, e, 1e-9), "θ {v} vs {e}");
    }
    let xi = vec![
        vec![0.331350220796, 0.335703933253, 0.332945845951],
        vec![0.191460271252, 0.05974510191, 0.748794626839],
        vec![0.067247556622, 0.865412465579, 0.067339977799],
        vec![0.207864575608, 0.060576622563, 0.731558801829],
    ];
    assert_rows_close(&m.spam_distribution, &xi, 1e-9, "ξ");
    let posteriors = vec![
        vec![0.997624760809, 0.001687877212, 0.00068736198],
        vec![5.4946109e-05, 0.999873757315, 7.1296576e-05],
        vec![0.001364771072, 0.001364771072, 0.997270457857],
        vec![0.999958129041, 1.8223997e-05, 2.3646962e-05],
        vec![0.000608866217, 0.996349205881, 0.003041927902],
        vec![0.000124918242, 6.1016748e-05, 0.99981406501],
        vec![0.333333333333, 0.333333333333, 0.333333333333],
        vec![0.892637209195, 0.053681395403, 0.053681395403],
    ];
    assert_rows_close(&m.posteriors, &posteriors, 1e-9, "T");
    assert!(close(m.log_likelihood, -17.526476814226, 1e-9));
    assert_eq!(m.labels(), vec![0, 1, 2, 0, 1, 2, 0, 0]);
    assert_eq!(mace(&t, &opts).unwrap(), m);
}

#[test]
fn mace_is_a_fixed_point_of_one_e_step_and_m_step() {
    let t = mace_table();
    let m = mace(
        &t,
        &MaceOpts {
            tol: 1e-14,
            max_iter: 5000,
            ..MaceOpts::default()
        },
    )
    .unwrap();
    assert!(m.converged);
    assert!(mace_residual(&t, &m, 0.1) < 1e-12);
}

#[test]
fn mace_uniform_start_reaches_the_same_fixed_point() {
    // ref_mace.py uniform start: 83 iterations, the same θ to 1e-11.
    let t = mace_table();
    let from_votes = mace(
        &t,
        &MaceOpts {
            tol: 1e-12,
            ..MaceOpts::default()
        },
    )
    .unwrap();
    let from_uniform = mace(
        &t,
        &MaceOpts {
            tol: 1e-12,
            init: MaceInit::Uniform,
            ..MaceOpts::default()
        },
    )
    .unwrap();
    assert!(from_uniform.converged);
    assert_eq!(from_uniform.iterations, 83);
    for (a, b) in from_uniform.competence.iter().zip(&from_votes.competence) {
        assert!(close(*a, *b, 1e-9));
    }
    assert_rows_close(&from_uniform.posteriors, &from_votes.posteriors, 1e-9, "T");
    assert!(close(
        from_uniform.log_likelihood,
        from_votes.log_likelihood,
        1e-9
    ));
}

#[test]
fn mace_item_without_labels_has_a_uniform_posterior() {
    let t = mace_table();
    let m = mace(&t, &MaceOpts::default()).unwrap();
    assert_eq!(m.posteriors[6], vec![1.0 / 3.0; 3]);
    assert_eq!(m.labels()[6], 0);
    assert_eq!(posterior_entropy(&m.posteriors)[6], 3f64.log2());
}

#[test]
fn mace_one_rater() {
    // A single rater: nothing can contradict it, so its label is the most probable one
    // for every item, but θ is not identifiable — the marginal of a label is the mixture
    // θ/K + (1 − θ) ξ(a), so every (θ, ξ) matching the empirical label frequencies
    // (1/2, 1/4, 1/4) is a maximum, and the EM settles where the smoothing puts it.
    let t = LabelTable::complete(&[&[0], &[1], &[2], &[0]], 3).unwrap();
    let m = mace(&t, &MaceOpts::default()).unwrap();
    assert!(m.converged);
    assert_eq!(m.labels(), vec![0, 1, 2, 0]);
    assert_eq!(m.competence.len(), 1);
    assert!(m.competence[0] > 0.0 && m.competence[0] < 1.0);
    // At least as likely as the uniform model (θ = 1: 1/K per item) and at most the
    // saturated multinomial 2 ln(1/2) + 2 ln(1/4).
    assert!(m.log_likelihood >= -4.0 * 3f64.ln() - 1e-12);
    assert!(m.log_likelihood <= 2.0 * 0.5f64.ln() + 2.0 * 0.25f64.ln() + 1e-12);
    let mixture: Vec<f64> = (0..3)
        .map(|l| m.competence[0] / 3.0 + (1.0 - m.competence[0]) * m.spam_distribution[0][l])
        .collect();
    assert!(mixture[0] > mixture[1] && close(mixture[1], mixture[2], 1e-12));
    for (row, &l) in m.posteriors.iter().zip(&[0usize, 1, 2, 0]) {
        let s: f64 = row.iter().sum();
        assert!(close(s, 1.0, 1e-12));
        assert!(row.iter().all(|v| *v <= row[l]));
    }
    assert!(mace_residual(&t, &m, 0.1) < 1e-9);
    // Without smoothing θ = 1 is a fixed point from the majority-vote start: the rater is
    // taken at its word and the posteriors are one-hot.
    let mle = mace(
        &t,
        &MaceOpts {
            smoothing: 0.0,
            ..MaceOpts::default()
        },
    )
    .unwrap();
    assert_eq!(mle.competence, vec![1.0]);
    assert_eq!(mle.posteriors[1], vec![0.0, 1.0, 0.0]);
}

#[test]
fn mace_unanimous_raters_are_fully_competent() {
    let t = LabelTable::complete(
        &[
            &[0, 0, 0],
            &[1, 1, 1],
            &[2, 2, 2],
            &[1, 1, 1],
            &[0, 0, 0],
            &[2, 2, 2],
        ],
        3,
    )
    .unwrap();
    // Maximum likelihood (no smoothing): θ_r = 1 exactly, one-hot posteriors, in one step.
    let mle = mace(
        &t,
        &MaceOpts {
            smoothing: 0.0,
            ..MaceOpts::default()
        },
    )
    .unwrap();
    assert!(mle.converged);
    assert_eq!(mle.competence, vec![1.0; 3]);
    assert_eq!(mle.labels(), vec![0, 1, 2, 1, 0, 2]);
    for (row, &l) in mle.posteriors.iter().zip(&[0usize, 1, 2, 1, 0, 2]) {
        assert_eq!(row[l], 1.0);
    }
    assert!(close(mle.log_likelihood, -6.0 * 3f64.ln(), 1e-12));
    // Smoothed: θ_r → (6 + δ)/(6 + 2δ) from above as the spam responsibilities vanish.
    let smoothed = mace(&t, &MaceOpts::default()).unwrap();
    assert!(smoothed.converged);
    for theta in &smoothed.competence {
        assert!(*theta > 0.95 && *theta < 6.1 / 6.2 + 1e-12, "θ = {theta}");
    }
}

#[test]
fn mace_synthetic_recovers_competence_of_reliable_and_spammy_raters() {
    let MaceSample { table, theta, .. } = synthetic_mace();
    let m = mace(&table, &MaceOpts::default()).unwrap();
    assert!(m.converged);
    for r in (0..6).chain(12..20) {
        assert!(
            close(m.competence[r], theta[r], 0.1),
            "rater {r}: θ̂ = {} vs θ = {}",
            m.competence[r],
            theta[r]
        );
    }
    // The spammers' strategy is recovered too: ξ_r(0) ≈ 0.9.
    for r in 12..20 {
        assert!(
            m.spam_distribution[r][0] > 0.75,
            "rater {r}: ξ(0) = {}",
            m.spam_distribution[r][0]
        );
    }
    // And the middling raters land between the two groups.
    for r in 6..12 {
        assert!(
            m.competence[r] > 0.35 && m.competence[r] < 0.85,
            "rater {r}: θ̂ = {}",
            m.competence[r]
        );
    }
}

#[test]
fn mace_synthetic_beats_majority_vote() {
    let MaceSample { table, truth, .. } = synthetic_mace();
    let m = mace(&table, &MaceOpts::default()).unwrap();
    let mace_correct = m
        .labels()
        .iter()
        .zip(&truth)
        .filter(|(a, b)| a == b)
        .count();
    let mv_correct = majority_votes(&table)
        .iter()
        .zip(&truth)
        .filter(|(v, t)| v.winner.or(v.tied.first().copied()) == Some(**t))
        .count();
    assert!(
        mace_correct > mv_correct,
        "MACE {mace_correct}/200 vs majority vote {mv_correct}/200"
    );
    assert!(mace_correct >= 190, "MACE {mace_correct}/200");
}

#[test]
fn mace_validates_options_and_initial_posteriors() {
    let t = mace_table();
    let bad = |opts: MaceOpts| mace(&t, &opts).is_err();
    assert!(bad(MaceOpts {
        max_iter: 0,
        ..MaceOpts::default()
    }));
    assert!(bad(MaceOpts {
        tol: f64::INFINITY,
        ..MaceOpts::default()
    }));
    assert!(bad(MaceOpts {
        smoothing: -0.1,
        ..MaceOpts::default()
    }));
    assert!(bad(MaceOpts {
        init: MaceInit::Posteriors(vec![vec![1.0; 2]; 8]),
        ..MaceOpts::default()
    }));
    assert!(bad(MaceOpts {
        init: MaceInit::Posteriors(vec![vec![0.0; 3]; 8]),
        ..MaceOpts::default()
    }));
    let err = mace(
        &t,
        &MaceOpts {
            init: MaceInit::Posteriors(vec![vec![f64::NAN; 3]; 8]),
            ..MaceOpts::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("mace"), "{err}");
    let one_class = LabelTable::complete(&[&[0, 0]], 1).unwrap();
    assert!(mace(&one_class, &MaceOpts::default()).is_err());
    // A valid explicit start (unnormalised rows are normalised) runs and converges.
    let ok = mace(
        &t,
        &MaceOpts {
            init: MaceInit::Posteriors(vec![vec![2.0, 1.0, 1.0]; 8]),
            ..MaceOpts::default()
        },
    )
    .unwrap();
    assert!(ok.converged);
    assert_eq!(ok.labels()[..6], [0, 1, 2, 0, 1, 2]);
}

// ── Confusion against gold, posterior entropy ────────────────────────

#[test]
fn rater_confusion_from_gold_by_hand() {
    // Rater 0: gold 0 → (0, 0, 1): 2/3 right; gold 1 → (1): wrong once; gold 2: never answered.
    let t = LabelTable::from_rows(
        &[
            &[Some(0), Some(0)],
            &[Some(0), Some(1)],
            &[Some(1), Some(0)],
            &[Some(0), Some(1)],
            &[None, Some(2)],
            &[Some(2), Some(2)],
        ],
        3,
    )
    .unwrap();
    let gold = [Some(0), Some(0), Some(0), Some(1), Some(2), None];
    let c = rater_confusion_from_gold(&t, &gold).unwrap();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0][0], vec![q(2, 3), q(1, 3), qi(0)]);
    assert_eq!(c[0][1], vec![qi(1), qi(0), qi(0)]);
    assert_eq!(c[0][2], vec![q(1, 3), q(1, 3), q(1, 3)]);
    assert_eq!(c[1][0], vec![q(2, 3), q(1, 3), qi(0)]);
    assert_eq!(c[1][1], vec![qi(0), qi(1), qi(0)]);
    assert_eq!(c[1][2], vec![qi(0), qi(0), qi(1)]);
    for rater in &c {
        for row in rater {
            let s = row.iter().fold(qi(0), |acc, v| acc + v);
            assert_eq!(s, qi(1));
        }
    }
}

#[test]
fn rater_confusion_from_gold_agrees_with_worker_accuracy_on_the_diagonal() {
    let t = synthetic_ds();
    let truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1];
    let gold: Vec<Option<usize>> = truth.iter().map(|&g| Some(g)).collect();
    let c = rater_confusion_from_gold(&t, &gold).unwrap();
    // Rater 0 is perfect: identity.
    for i in 0..3 {
        for l in 0..3 {
            assert_eq!(c[0][i][l], if i == l { qi(1) } else { qi(0) });
        }
    }
    // Rater 1: class 0 → (0, 1, 0, 0, 1): 3/5 right, 2/5 said 1.
    assert_eq!(c[1][0], vec![q(3, 5), q(2, 5), qi(0)]);
    // The class-weighted diagonal is the rater's accuracy.
    for r in 0..5 {
        let acc = worker_accuracy(&t.rater(r).unwrap(), &truth).unwrap();
        let answered_per_class: Vec<usize> = (0..3)
            .map(|g| {
                t.rater(r)
                    .unwrap()
                    .iter()
                    .zip(&truth)
                    .filter(|(a, tr)| a.is_some() && **tr == g)
                    .count()
            })
            .collect();
        let weighted = (0..3).fold(qi(0), |acc, g| {
            acc + &c[r][g][g] * qi(answered_per_class[g] as i64)
        });
        assert_eq!(weighted, qi(acc.correct as i64));
    }
}

#[test]
fn rater_confusion_from_gold_errors() {
    let t = LabelTable::complete(&[&[0, 1], &[1, 1]], 2).unwrap();
    assert!(rater_confusion_from_gold(&t, &[Some(0)]).is_err());
    assert!(rater_confusion_from_gold(&t, &[Some(0), Some(2)]).is_err());
    // No gold at all: every row uniform.
    let c = rater_confusion_from_gold(&t, &[None, None]).unwrap();
    assert_eq!(c[0], vec![vec![q(1, 2), q(1, 2)], vec![q(1, 2), q(1, 2)]]);
}

#[test]
fn posterior_entropy_by_hand() {
    let h = posterior_entropy(&[
        vec![1.0, 0.0],
        vec![0.5, 0.5],
        vec![0.25; 4],
        vec![0.5, 0.25, 0.25],
        vec![0.5, 0.25, 0.125, 0.125],
    ]);
    assert_eq!(h, vec![0.0, 1.0, 2.0, 1.5, 1.75]);
    assert!(h[0].is_sign_positive());
}

#[test]
fn posterior_entropy_normalises_rows_and_ranks_items() {
    // Unnormalised and negative / zero entries.
    let h = posterior_entropy(&[vec![2.0, 2.0], vec![3.0, 0.0, -1.0], vec![0.0, 0.0], vec![]]);
    assert_eq!(h, vec![1.0, 0.0, 0.0, 0.0]);
    // The tied item of the synthetic table is the least certain under MAP Dawid–Skene.
    let ds = dawid_skene_map(
        &synthetic_ds(),
        &DawidSkenePriors::symmetric(3, 2.0, 3.0, 1.5),
        &DawidSkeneOpts::default(),
    )
    .unwrap();
    let h = posterior_entropy(&ds.posteriors);
    let most_uncertain = (0..14)
        .max_by(|&a, &b| h[a].partial_cmp(&h[b]).unwrap())
        .unwrap();
    assert_eq!(most_uncertain, 12);
    assert!(h.iter().all(|v| *v >= 0.0 && *v <= 3f64.log2()));
}

// ── Markov chains: Display and the limiting distribution ─────────────

#[test]
fn markov_display_of_a_two_state_chain_is_the_matrix_layout() {
    let p = QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]]).unwrap();
    let chain = MarkovChain::new(p.clone()).unwrap();
    assert_eq!(chain.to_string(), "[\n  [1/2, 1/2],\n  [  1,   0]\n]");
    assert_eq!(chain.to_string(), p.to_string());
}

#[test]
fn markov_display_with_labels_prefixes_the_rows() {
    let p = QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]]).unwrap();
    let chain = MarkovChain::with_labels(p, vec!["sunny".into(), "rain".into()]).unwrap();
    assert_eq!(
        chain.to_string(),
        "[\n  sunny: [1/2, 1/2],\n   rain: [  1,   0]\n]"
    );
    let single =
        MarkovChain::with_labels(QMatrix::new(vec![vec![qi(1)]]).unwrap(), vec!["a".into()])
            .unwrap();
    assert_eq!(single.to_string(), "[\n  a: [1]\n]");
    assert_eq!(
        MarkovChain::new(QMatrix::new(vec![vec![qi(1)]]).unwrap())
            .unwrap()
            .to_string(),
        "[[1]]"
    );
}

#[test]
fn limiting_distribution_of_an_ergodic_chain_is_the_stationary_one() {
    // Regular (second eigenvalue 1/4): every row of P⁵⁰ is within 1e-12 of π = (4/11, 3/11, 4/11),
    // checked exactly (SymPy: max |P⁵⁰ − π| = 3.9e-31).
    let p = QMatrix::new(vec![
        vec![q(1, 2), q(1, 4), q(1, 4)],
        vec![q(1, 3), q(1, 3), q(1, 3)],
        vec![q(1, 4), q(1, 4), q(1, 2)],
    ])
    .unwrap();
    let chain = MarkovChain::new(p).unwrap();
    assert!(chain.is_regular());
    let pi = chain.limiting_distribution().unwrap().unwrap();
    assert_eq!(pi, vec![q(4, 11), q(3, 11), q(4, 11)]);
    assert_eq!(pi, chain.stationary_distribution().unwrap());
    let p50 = chain.n_step(50);
    let bound = q(1, 1_000_000_000_000);
    for i in 0..3 {
        for j in 0..3 {
            let diff = p50.get(i, j) - &pi[j];
            let abs = if diff < qi(0) { -diff } else { diff };
            assert!(abs < bound, "P⁵⁰[{i}][{j}] is {} from π", abs);
        }
    }
    // And the two-state chain of the Display test: π = (2/3, 1/3).
    let two =
        MarkovChain::new(QMatrix::new(vec![vec![q(1, 2), q(1, 2)], vec![qi(1), qi(0)]]).unwrap())
            .unwrap();
    assert_eq!(
        two.limiting_distribution().unwrap(),
        Some(vec![q(2, 3), q(1, 3)])
    );
}

#[test]
fn limiting_distribution_of_a_periodic_chain_is_none() {
    // Period 2: π = (1/2, 1/2) is stationary but Pⁿ alternates between I and P.
    let p = QMatrix::new(vec![vec![qi(0), qi(1)], vec![qi(1), qi(0)]]).unwrap();
    let chain = MarkovChain::new(p.clone()).unwrap();
    assert!(chain.is_irreducible() && !chain.is_aperiodic());
    assert_eq!(chain.period_of(0).unwrap(), Some(2));
    assert_eq!(
        chain.stationary_distribution().unwrap(),
        vec![q(1, 2), q(1, 2)]
    );
    assert_eq!(chain.limiting_distribution().unwrap(), None);
    assert_eq!(chain.n_step(50), QMatrix::identity(2).unwrap());
    assert_eq!(chain.n_step(51), p);
}

#[test]
fn limiting_distribution_of_a_reducible_chain_is_none() {
    // Fair gambler's ruin on 0..=3: Pⁿ converges, but to a start-dependent row.
    let h = q(1, 2);
    let p = QMatrix::new(vec![
        vec![qi(1), qi(0), qi(0), qi(0)],
        vec![h.clone(), qi(0), h.clone(), qi(0)],
        vec![qi(0), h.clone(), qi(0), h.clone()],
        vec![qi(0), qi(0), qi(0), qi(1)],
    ])
    .unwrap();
    let chain = MarkovChain::new(p).unwrap();
    assert!(!chain.is_irreducible());
    assert_eq!(chain.limiting_distribution().unwrap(), None);
    assert_eq!(chain.stationary_distributions().len(), 2);
    // The start-dependent limit is the absorption matrix: from 1, ruin with probability 2/3.
    let b = chain.absorption_probabilities().unwrap();
    assert_eq!(b.row(0), &[q(2, 3), q(1, 3)]);
    let p60 = chain.n_step(60);
    let bound = q(1, 1_000_000_000_000);
    let d = p60.get(1, 0) - q(2, 3);
    assert!((if d < qi(0) { -d } else { d }) < bound);
    assert_ne!(p60.row(1), p60.row(2));
}
