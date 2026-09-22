//! symplex 0.17 — exact (symbolic) confidence intervals:
//! `stats::estimation::{z_for_confidence, proportion_interval_symbolic,
//! proportion_interval_exact, confidence_interval_mean_z_symbolic,
//! confidence_interval_mean_z_exact}` (the proportion intervals moved from
//! `aggregation` to `estimation` in 0.18).
//!
//! Oracles (`symplex/.venv/bin/python`, scipy 1.18.1, statsmodels 0.15.0,
//! mpmath 1.3.0):
//! * `scipy.stats.norm.ppf(1 - alpha/2)` for `z`;
//! * `statsmodels.stats.proportion.proportion_confint(k, n, alpha,
//!   method='normal' | 'wilson' | 'agresti_coull')` for the closed forms
//!   (statsmodels clips to `[0, 1]`; the exact forms do not, so the
//!   comparisons below clamp the exact value first);
//! * `scipy.stats.beta.ppf(alpha/2, k, n-k+1)` and
//!   `beta.ppf(1-alpha/2, k+1, n-k)` for Clopper–Pearson;
//! * `mpmath` at 50 dps, `findroot` (bisection) on the binomial tail
//!   `Σ C(n,j) p^j (1-p)^(n-j) - α/2` — cross-checked against
//!   `mpmath.betainc(..., regularized=True)` — for the 25-digit values.
//!
//! Confidence levels are `fractions.Fraction`s (`Fraction(95, 100)` etc.),
//! never floats converted with `limit_denominator`.

use symplex::linprog::{q, qi};
use symplex::prelude::*;
use symplex::stats::estimation::{
    IntervalMethod, confidence_interval_mean_z, confidence_interval_mean_z_exact,
    confidence_interval_mean_z_symbolic, proportion_interval, proportion_interval_exact,
    proportion_interval_symbolic, z_for_confidence,
};

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "got {actual}, expected {expected} (tol {tol})"
    );
}

fn ev(e: &Ex) -> f64 {
    e.eval_f64().unwrap()
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

/// `(k, n, confidence)` triples used throughout, with the `f64` level the
/// existing function takes.
struct Triple {
    k: usize,
    n: usize,
    conf: Q,
    conf_f64: f64,
}

fn triples() -> Vec<Triple> {
    let t = |k, n, conf: Q, conf_f64| Triple {
        k,
        n,
        conf,
        conf_f64,
    };
    vec![
        t(3, 10, q(95, 100), 0.95),
        t(0, 10, q(95, 100), 0.95),
        t(10, 10, q(95, 100), 0.95),
        t(1, 1, q(9, 10), 0.9),
        t(0, 1, q(95, 100), 0.95),
        t(1, 5, q(95, 100), 0.95),
        t(5, 8, q(95, 100), 0.95),
        t(1, 20, q(99, 100), 0.99),
        t(12, 30, q(999, 1000), 0.999),
    ]
}

const CLOSED_FORMS: [IntervalMethod; 3] = [
    IntervalMethod::Wald,
    IntervalMethod::Wilson,
    IntervalMethod::AgrestiCoull,
];

// ── z_for_confidence ─────────────────────────────────────────────────

#[test]
fn z_for_confidence_95_matches_scipy() {
    let ctx = Context::new();
    let z = z_for_confidence(&ctx, &q(95, 100)).unwrap();
    assert_eq!(format!("{z}"), "sqrt(2)*erfinv(19/20)");
    // scipy: norm.ppf(0.975)
    close(ev(&z), 1.959963984540054, 1e-14);
}

#[test]
fn z_for_confidence_other_levels_match_scipy() {
    let ctx = Context::new();
    // norm.ppf(0.95), norm.ppf(0.995); norm.isf(0.0005) for the last (ppf(0.9995)
    // loses 3e-14 to the rounding of 0.9995 — mpmath: 3.290526731491894793221627).
    close(
        ev(&z_for_confidence(&ctx, &q(9, 10)).unwrap()),
        1.644853626951472,
        1e-14,
    );
    close(
        ev(&z_for_confidence(&ctx, &q(99, 100)).unwrap()),
        2.5758293035489004,
        1e-14,
    );
    close(
        ev(&z_for_confidence(&ctx, &q(999, 1000)).unwrap()),
        3.2905267314918945,
        1e-14,
    );
}

#[test]
fn z_for_confidence_evaluates_to_high_precision() {
    let ctx = Context::new();
    let z = z_for_confidence(&ctx, &q(95, 100)).unwrap();
    // mpmath: sqrt(2)*erfinv(mpf(95)/100) = 1.9599639845400542355245944305205515
    let s = z.eval_decimal(30).unwrap();
    assert!(s.starts_with("1.9599639845400542355245944305"), "got {s}");
}

#[test]
fn z_for_confidence_rejects_levels_outside_unit_interval() {
    let ctx = Context::new();
    for bad in [qi(0), qi(1), q(-1, 2), q(3, 2), qi(2)] {
        let err = z_for_confidence(&ctx, &bad).unwrap_err();
        assert!(
            matches!(err, SymplexError::InvalidArgument { .. }),
            "{bad}: {err}"
        );
    }
}

// ── proportion_interval_symbolic: shape ──────────────────────────────

#[test]
fn wilson_symbolic_display_contains_z_squared() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::Wilson).unwrap();
    assert_eq!(ci.kind, IntervalKind::Closed);
    let (lo, hi) = (format!("{}", ci.lower), format!("{}", ci.upper));
    assert!(lo.contains("z^2"), "lower: {lo}");
    assert!(hi.contains("z^2"), "upper: {hi}");
    assert!(hi.contains("sqrt("), "upper: {hi}");
    assert_eq!(ci.upper.free_symbols(), vec![z.clone()]);
    assert_eq!(ci.lower.free_symbols(), vec![z]);
}

#[test]
fn wald_symbolic_is_the_textbook_formula() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::Wald).unwrap();
    // p̂ ± z √(p̂(1−p̂)/n) = 3/10 ± z √(21/1000) = 3/10 ± z √210 / 100
    assert_eq!(format!("{}", ci.upper), "1/100*z*sqrt(210) + 3/10");
    assert_eq!(format!("{}", ci.lower), "-1/100*z*sqrt(210) + 3/10");
    assert_eq!(
        (&ci.upper + &ci.lower).simplify(),
        ctx.rational(3, 5),
        "the endpoints are symmetric about p̂"
    );
}

#[test]
fn symbolic_intervals_collapse_to_p_hat_at_z_zero() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for m in CLOSED_FORMS {
        let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, m).unwrap();
        assert_eq!(
            ci.lower.subs_i64(&z, 0).simplify(),
            ctx.rational(3, 10),
            "{m:?} lower"
        );
        assert_eq!(
            ci.upper.subs_i64(&z, 0).simplify(),
            ctx.rational(3, 10),
            "{m:?} upper"
        );
    }
}

#[test]
fn agresti_coull_symbolic_centre_is_shrunk_towards_one_half() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let ci = proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::AgrestiCoull).unwrap();
    // centre p̃ = (3 + z²/2)/(10 + z²); at z = 2: 5/14
    let centre = ((&ci.lower + &ci.upper) / ctx.int(2))
        .subs_i64(&z, 2)
        .simplify();
    assert_eq!(centre, ctx.rational(5, 14));
}

#[test]
fn symbolic_z_with_exact_quantile_matches_f64_on_all_triples() {
    let ctx = Context::new();
    for t in triples() {
        let z = z_for_confidence(&ctx, &t.conf).unwrap();
        for m in CLOSED_FORMS {
            let exact = proportion_interval_symbolic(&ctx, t.k, t.n, &z, m).unwrap();
            let f = proportion_interval(t.k, t.n, t.conf_f64, m).unwrap();
            let what = format!("{m:?} k={} n={} c={}", t.k, t.n, t.conf);
            assert!(
                (clamp01(ev(&exact.lower)) - f.lower).abs() < 1e-12,
                "{what} lower"
            );
            assert!(
                (clamp01(ev(&exact.upper)) - f.upper).abs() < 1e-12,
                "{what} upper"
            );
        }
    }
}

// ── proportion_interval_exact: closed forms ──────────────────────────

#[test]
fn exact_closed_forms_agree_with_f64_on_all_triples() {
    let ctx = Context::new();
    for t in triples() {
        for m in CLOSED_FORMS {
            let exact = proportion_interval_exact(&ctx, t.k, t.n, &t.conf, m).unwrap();
            let f = proportion_interval(t.k, t.n, t.conf_f64, m).unwrap();
            let what = format!("{m:?} k={} n={} c={}", t.k, t.n, t.conf);
            assert_eq!(exact.kind, IntervalKind::Closed);
            assert!(
                (clamp01(ev(&exact.lower)) - f.lower).abs() < 1e-12,
                "{what} lower"
            );
            assert!(
                (clamp01(ev(&exact.upper)) - f.upper).abs() < 1e-12,
                "{what} upper"
            );
        }
    }
}

#[test]
fn exact_wilson_matches_statsmodels() {
    let ctx = Context::new();
    // proportion_confint(3, 10, 0.05, 'wilson'); (12, 30, 0.001); (5, 8, 0.05)
    let cases = [
        (3, 10, q(95, 100), 0.10779126740630104, 0.6032218525388546),
        (12, 30, q(999, 1000), 0.1728435010901418, 0.6801969673296636),
        (5, 8, q(95, 100), 0.3057423946026273, 0.8631557141764027),
    ];
    for (k, n, c, lo, hi) in cases {
        let ci = proportion_interval_exact(&ctx, k, n, &c, IntervalMethod::Wilson).unwrap();
        close(ev(&ci.lower), lo, 1e-12);
        close(ev(&ci.upper), hi, 1e-12);
    }
}

#[test]
fn exact_agresti_coull_matches_statsmodels() {
    let ctx = Context::new();
    // proportion_confint(…, method='agresti_coull')
    let cases = [
        (3, 10, q(95, 100), 0.10333841792242526, 0.6076747020227304),
        (
            12,
            30,
            q(999, 1000),
            0.17182691884989748,
            0.6812135495699079,
        ),
        (5, 8, q(95, 100), 0.3037564603462702, 0.8651416484327599),
    ];
    for (k, n, c, lo, hi) in cases {
        let ci = proportion_interval_exact(&ctx, k, n, &c, IntervalMethod::AgrestiCoull).unwrap();
        close(ev(&ci.lower), lo, 1e-12);
        close(ev(&ci.upper), hi, 1e-12);
    }
}

#[test]
fn exact_wald_matches_statsmodels() {
    let ctx = Context::new();
    // proportion_confint(…, method='normal')
    let cases = [
        (3, 10, q(95, 100), 0.015974234910674567, 0.5840257650893255),
        (12, 30, q(999, 1000), 0.10568634186415704, 0.694313658135843),
        (5, 8, q(95, 100), 0.289526098053033, 0.960473901946967),
    ];
    for (k, n, c, lo, hi) in cases {
        let ci = proportion_interval_exact(&ctx, k, n, &c, IntervalMethod::Wald).unwrap();
        close(ev(&ci.lower), lo, 1e-12);
        close(ev(&ci.upper), hi, 1e-12);
    }
}

#[test]
fn exact_closed_forms_are_not_clipped_to_unit_interval() {
    let ctx = Context::new();
    // Wald, k = 1, n = 5: 1/5 − 1.96·√(4/125) ≈ −0.15; statsmodels gives 0.0.
    let wald = proportion_interval_exact(&ctx, 1, 5, &q(95, 100), IntervalMethod::Wald).unwrap();
    assert!(ev(&wald.lower) < -0.15, "{}", ev(&wald.lower));
    assert_eq!(
        proportion_interval(1, 5, 0.95, IntervalMethod::Wald)
            .unwrap()
            .lower,
        0.0
    );
    // Agresti–Coull, k = 0, n = 10: statsmodels (0.0, 0.3208873057505458).
    let ac =
        proportion_interval_exact(&ctx, 0, 10, &q(95, 100), IntervalMethod::AgrestiCoull).unwrap();
    assert!(ev(&ac.lower) < -0.04, "{}", ev(&ac.lower));
    close(ev(&ac.upper), 0.3208873057505458, 1e-12);
    // Wilson always lies inside [0, 1]: k = 0 gives exactly 0.
    let w = proportion_interval_exact(&ctx, 0, 10, &q(95, 100), IntervalMethod::Wilson).unwrap();
    close(ev(&w.lower), 0.0, 1e-15);
    close(ev(&w.upper), 0.27753279986288926, 1e-12);
}

// ── proportion_interval_exact: Clopper–Pearson ───────────────────────

#[test]
fn clopper_pearson_exact_matches_scipy_beta_ppf_small_n() {
    let ctx = Context::new();
    // beta.ppf(alpha/2, k, n-k+1), beta.ppf(1-alpha/2, k+1, n-k); 0 / 1 at k = 0 / k = n.
    let cases = [
        (3, 10, q(95, 100), 0.06673951117773447, 0.6524528500599973),
        (0, 10, q(95, 100), 0.0, 0.3084971078187608),
        (10, 10, q(95, 100), 0.6915028921812392, 1.0),
        (1, 1, q(9, 10), 0.05, 1.0),
        (0, 1, q(95, 100), 0.0, 0.975),
        (1, 5, q(95, 100), 0.0050507633794680575, 0.7164179361180895),
        (5, 8, q(95, 100), 0.2448632163665516, 0.9147665858627464),
    ];
    for (k, n, c, lo, hi) in cases {
        let ci = proportion_interval_exact(&ctx, k, n, &c, IntervalMethod::ClopperPearson).unwrap();
        assert_eq!(ci.kind, IntervalKind::Closed);
        close(ev(&ci.lower), lo, 1e-12);
        close(ev(&ci.upper), hi, 1e-12);
    }
}

#[test]
fn clopper_pearson_exact_matches_scipy_beta_ppf_k1_n12_99() {
    let ctx = Context::new();
    // beta.ppf(0.005, 1, 12) = 0.00041762458919299063, beta.ppf(0.995, 2, 11) = 0.4770262943621243
    let ci = proportion_interval_exact(&ctx, 1, 12, &q(99, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    close(ev(&ci.lower), 0.00041762458919299063, 1e-12);
    close(ev(&ci.upper), 0.4770262943621243, 1e-12);
    assert!(format!("{}", ci.lower).contains("_p^12"));
}

#[test]
fn clopper_pearson_exact_n12_999_matches_scipy_and_mpmath() {
    // Degree-`n` `RootOf` isolation is exact but slow in debug builds
    // (n = 30 took ~40 s, n = 18 ~10 s); n = 12 keeps the test fast and the
    // exactness claim is the same.
    let ctx = Context::new();
    // beta.ppf(0.0005, 5, 8) = 0.06196208751526386, beta.ppf(0.9995, 6, 7) = 0.856910011395322
    let ci = proportion_interval_exact(&ctx, 5, 12, &q(999, 1000), IntervalMethod::ClopperPearson)
        .unwrap();
    close(ev(&ci.lower), 0.06196208751526386, 1e-12);
    close(ev(&ci.upper), 0.856910011395322, 1e-12);
    // mpmath (50 dps) bisection on the binomial tails:
    //   0.061962087515263850883318548277740807
    //   0.85691001139531943944656747330288747
    let lo = ci.lower.eval_decimal(30).unwrap();
    let hi = ci.upper.eval_decimal(30).unwrap();
    assert!(lo.starts_with("0.0619620875152638508833185"), "got {lo}");
    assert!(hi.starts_with("0.8569100113953194394465674"), "got {hi}");
}

#[test]
fn clopper_pearson_exact_agrees_with_f64_bisection() {
    let ctx = Context::new();
    for t in triples().into_iter().filter(|t| t.n <= 12) {
        let exact =
            proportion_interval_exact(&ctx, t.k, t.n, &t.conf, IntervalMethod::ClopperPearson)
                .unwrap();
        let f = proportion_interval(t.k, t.n, t.conf_f64, IntervalMethod::ClopperPearson).unwrap();
        let what = format!("k={} n={} c={}", t.k, t.n, t.conf);
        close(ev(&exact.lower), f.lower, 1e-12);
        close(ev(&exact.upper), f.upper, 1e-12);
        assert!(ev(&exact.lower) <= ev(&exact.upper), "{what}");
    }
}

#[test]
fn clopper_pearson_k_zero_lower_end_is_exact_zero() {
    let ctx = Context::new();
    let ci = proportion_interval_exact(&ctx, 0, 10, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    assert_eq!(ci.lower, ctx.int(0));
    assert!(format!("{}", ci.upper).starts_with("RootOf("));
    // (1 − p)^10 = 1/40 ⇒ p = 1 − 40^(−1/10): beta.ppf(0.975, 1, 10)
    close(ev(&ci.upper), 0.3084971078187608, 1e-12);
    close(ev(&ci.upper), 1.0 - 40f64.powf(-0.1), 1e-14);
}

#[test]
fn clopper_pearson_k_equals_n_upper_end_is_exact_one() {
    let ctx = Context::new();
    let ci = proportion_interval_exact(&ctx, 10, 10, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    assert_eq!(ci.upper, ctx.int(1));
    assert!(format!("{}", ci.lower).starts_with("RootOf("));
    // p^10 = 1/40: beta.ppf(0.025, 10, 1)
    close(ev(&ci.lower), 0.6915028921812392, 1e-12);
    close(ev(&ci.lower), 40f64.powf(-0.1), 1e-14);
}

#[test]
fn clopper_pearson_n_one_endpoints_are_rational() {
    let ctx = Context::new();
    // k = n = 1 at 90 %: p_L solves p = 1/20.
    let ci =
        proportion_interval_exact(&ctx, 1, 1, &q(9, 10), IntervalMethod::ClopperPearson).unwrap();
    assert_eq!(ci.lower, ctx.rational(1, 20));
    assert_eq!(ci.upper, ctx.int(1));
    assert_eq!(format!("{ci}"), "[1/20, 1]");
    // k = 0, n = 1 at 95 %: p_U solves 1 − p = 1/40.
    let ci =
        proportion_interval_exact(&ctx, 0, 1, &q(95, 100), IntervalMethod::ClopperPearson).unwrap();
    assert_eq!(ci.lower, ctx.int(0));
    assert_eq!(ci.upper, ctx.rational(39, 40));
}

#[test]
fn clopper_pearson_endpoints_are_root_of_nodes_of_degree_n() {
    let ctx = Context::new();
    let ci = proportion_interval_exact(&ctx, 3, 10, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    let (lo, hi) = (format!("{}", ci.lower), format!("{}", ci.upper));
    assert!(lo.starts_with("RootOf(") && lo.contains("_p^10"), "{lo}");
    assert!(hi.starts_with("RootOf(") && hi.contains("_p^10"), "{hi}");
    // The bound variable is not free in the result.
    assert!(ci.lower.free_symbols().is_empty());
    assert!(ci.upper.free_symbols().is_empty());
    assert_ne!(ci.lower, ci.upper);
}

#[test]
fn clopper_pearson_roots_satisfy_the_tail_equations() {
    let ctx = Context::new();
    let (k, n) = (3usize, 10usize);
    let ci =
        proportion_interval_exact(&ctx, k, n, &q(95, 100), IntervalMethod::ClopperPearson).unwrap();
    let (p_l, p_u) = (ev(&ci.lower), ev(&ci.upper));
    let choose = |j: usize| (0..j).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
    let pmf = |p: f64, j: usize| choose(j) * p.powi(j as i32) * (1.0 - p).powi((n - j) as i32);
    let upper_tail: f64 = (k..=n).map(|j| pmf(p_l, j)).sum(); // P(Bin(n, p_L) ≥ k)
    let lower_tail: f64 = (0..=k).map(|j| pmf(p_u, j)).sum(); // P(Bin(n, p_U) ≤ k)
    close(upper_tail, 0.025, 1e-14);
    close(lower_tail, 0.025, 1e-14);
    assert!(
        p_l < 0.3 && 0.3 < p_u,
        "p̂ = 3/10 lies inside [{p_l}, {p_u}]"
    );
}

#[test]
fn clopper_pearson_eval_decimal_30_matches_mpmath_to_25_digits() {
    let ctx = Context::new();
    let ci = proportion_interval_exact(&ctx, 3, 10, &q(95, 100), IntervalMethod::ClopperPearson)
        .unwrap();
    // mpmath (50 dps): 0.066739511177734467114648056291648899,
    //                  0.65245285005999729503830202810655414
    let lo = ci.lower.eval_decimal(30).unwrap();
    let hi = ci.upper.eval_decimal(30).unwrap();
    assert!(lo.starts_with("0.0667395111777344671146480"), "got {lo}");
    assert!(hi.starts_with("0.6524528500599972950383020"), "got {hi}");
    // All 30 requested digits of the lower end are right (…62916|48899 rounds down).
    assert!(
        lo.starts_with("0.0667395111777344671146480562916"),
        "got {lo}"
    );
}

#[test]
fn clopper_pearson_higher_confidence_gives_wider_interval() {
    let ctx = Context::new();
    // Two different levels for the same counts: the 99 % interval contains the 95 % one.
    let ci95 =
        proportion_interval_exact(&ctx, 5, 8, &q(95, 100), IntervalMethod::ClopperPearson).unwrap();
    let ci99 =
        proportion_interval_exact(&ctx, 5, 8, &q(99, 100), IntervalMethod::ClopperPearson).unwrap();
    assert!(ev(&ci99.lower) < ev(&ci95.lower));
    assert!(ev(&ci95.upper) < ev(&ci99.upper));
}

// ── error paths ──────────────────────────────────────────────────────

#[test]
fn proportion_interval_symbolic_rejects_clopper_pearson() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let err =
        proportion_interval_symbolic(&ctx, 3, 10, &z, IntervalMethod::ClopperPearson).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    assert!(
        err.to_string().contains("proportion_interval_exact"),
        "{err}"
    );
}

#[test]
fn proportion_interval_symbolic_rejects_bad_counts() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    for m in CLOSED_FORMS {
        let err = proportion_interval_symbolic(&ctx, 0, 0, &z, m).unwrap_err();
        assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
        let err = proportion_interval_symbolic(&ctx, 11, 10, &z, m).unwrap_err();
        assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    }
}

#[test]
fn proportion_interval_exact_rejects_bad_input() {
    let ctx = Context::new();
    let all = [
        IntervalMethod::Wald,
        IntervalMethod::Wilson,
        IntervalMethod::AgrestiCoull,
        IntervalMethod::ClopperPearson,
    ];
    for m in all {
        for (k, n, c) in [
            (0, 0, q(95, 100)),
            (11, 10, q(95, 100)),
            (3, 10, qi(0)),
            (3, 10, qi(1)),
            (3, 10, q(-1, 2)),
            (3, 10, q(101, 100)),
        ] {
            let err = proportion_interval_exact(&ctx, k, n, &c, m).unwrap_err();
            assert!(
                matches!(err, SymplexError::InvalidArgument { .. }),
                "{m:?} k={k} n={n} c={c}: {err}"
            );
        }
    }
}

// ── estimation: mean with known σ ────────────────────────────────────

#[test]
fn mean_z_symbolic_has_textbook_shape() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    // x̄ = 2, σ/√n = 2/√3
    let ci = confidence_interval_mean_z_symbolic(&ctx, &[1, 2, 3].map(qi), &qi(2), &z).unwrap();
    assert_eq!(ci.kind, IntervalKind::Closed);
    assert_eq!(format!("{ci}"), "[-2/3*z*sqrt(3) + 2, 2/3*z*sqrt(3) + 2]");
    assert_eq!(
        format!("{}", (&ci.upper - &ci.lower).simplify()),
        "4/3*z*sqrt(3)"
    );
    assert_eq!(ci.lower.subs_i64(&z, 0).simplify(), ctx.int(2));
    assert_eq!(ci.upper.free_symbols(), vec![z]);
}

#[test]
fn mean_z_exact_matches_scipy_norm_interval() {
    let ctx = Context::new();
    // scipy: norm.interval(0.95, 2, 2/sqrt(3)) = (-0.2631714681523438, 4.263171468152343)
    let ci =
        confidence_interval_mean_z_exact(&ctx, &[1, 2, 3].map(qi), &qi(2), &q(95, 100)).unwrap();
    close(ev(&ci.lower), -0.2631714681523438, 1e-12);
    close(ev(&ci.upper), 4.263171468152343, 1e-12);
    assert!(format!("{}", ci.upper).contains("erfinv(19/20)"));
}

#[test]
fn mean_z_exact_matches_f64_function() {
    let ctx = Context::new();
    let cases: [(&[i64], Q, Q, f64); 4] = [
        (&[1, 2, 3], qi(2), q(95, 100), 0.95),
        (&[5, 7, 8, 9, 10, 12], q(3, 2), q(99, 100), 0.99),
        (&[-4, 0, 4, 8], q(1, 3), q(9, 10), 0.9),
        (&[7], qi(1), q(999, 1000), 0.999),
    ];
    for (data, sigma, conf, conf_f64) in cases {
        let data: Vec<Q> = data.iter().map(|&x| qi(x)).collect();
        let exact = confidence_interval_mean_z_exact(&ctx, &data, &sigma, &conf).unwrap();
        let f = confidence_interval_mean_z(&data, &sigma, conf_f64).unwrap();
        close(ev(&exact.lower), f.lower, 1e-12);
        close(ev(&exact.upper), f.upper, 1e-12);
    }
}

#[test]
fn mean_z_exact_and_symbolic_reject_bad_input() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let data = [1, 2, 3].map(qi);
    let is_invalid = |r: Result<Interval<Ex>, SymplexError>| {
        let err = r.unwrap_err();
        assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    };
    is_invalid(confidence_interval_mean_z_symbolic(&ctx, &[], &qi(2), &z));
    is_invalid(confidence_interval_mean_z_symbolic(&ctx, &data, &qi(0), &z));
    is_invalid(confidence_interval_mean_z_symbolic(
        &ctx,
        &data,
        &qi(-1),
        &z,
    ));
    is_invalid(confidence_interval_mean_z_exact(
        &ctx,
        &[],
        &qi(2),
        &q(95, 100),
    ));
    is_invalid(confidence_interval_mean_z_exact(
        &ctx,
        &data,
        &qi(0),
        &q(95, 100),
    ));
    is_invalid(confidence_interval_mean_z_exact(
        &ctx,
        &data,
        &qi(2),
        &qi(1),
    ));
    is_invalid(confidence_interval_mean_z_exact(
        &ctx,
        &data,
        &qi(2),
        &qi(0),
    ));
}
