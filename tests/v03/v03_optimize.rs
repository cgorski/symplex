//! symplex 0.3 — `symplex::optimize`: Brent/bisection/Newton root finding,
//! Nelder–Mead, Brent/golden-section scalar minimisation, differential
//! evolution, least-squares polynomial fitting (float and exact) and the
//! `Ex` conveniences built on `compile`.

use num_bigint::BigInt;
use num_rational::Ratio;
use std::cell::Cell;
use std::f64::consts::PI;
use symplex::optimize::{
    DeOpts, MinimizeOpts, RootOpts, bisect, brent_root, differential_evolution, eval_poly,
    golden_section, linear_fit, minimize_scalar, nelder_mead, newton_root, poly_fit,
    poly_fit_exact, trapezoid,
};
use symplex::prelude::*;

/// Wall-clock hang guard.  Two seconds on a developer machine; scaled up on
/// shared CI runners (`CI` is set), which are several times slower and noisy.
fn time_budget(secs: u64) -> std::time::Duration {
    let mult = if std::env::var_os("CI").is_some() {
        5
    } else {
        1
    };
    std::time::Duration::from_secs(secs * mult)
}

fn q(p: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(d))
}

fn rosenbrock(p: &[f64]) -> f64 {
    (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2)
}

fn rastrigin(p: &[f64]) -> f64 {
    10.0 * p.len() as f64
        + p.iter()
            .map(|x| x * x - 10.0 * (2.0 * PI * x).cos())
            .sum::<f64>()
}

// ═══════════════════════════════════════════════════════════════════════════
// brent_root
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn brent_sqrt2_to_1e12() {
    let r = brent_root(|x| x * x - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-12, "{r}");
}

#[test]
fn brent_cos_x_equals_x() {
    let r = brent_root(|x| x.cos() - x, 0.0, 1.0, &RootOpts::default()).unwrap();
    assert!((r - 0.739_085_133_215_160_6).abs() < 1e-12, "{r}");
    assert!((r.cos() - r).abs() < 1e-12);
}

#[test]
fn brent_transcendental_family_in_n() {
    // 2n·c − 2 − sin(2πc)/π = 0 has exactly one root in (0, 1) for n ≥ 1.
    for n in 1..=5 {
        let nf = n as f64;
        let f = |c: f64| 2.0 * nf * c - 2.0 - (2.0 * PI * c).sin() / PI;
        let root = brent_root(f, 1e-9, 1.0, &RootOpts::default()).unwrap();
        assert!(root > 0.0 && root <= 1.0, "n = {n}: {root}");
        assert!(f(root).abs() < 1e-10, "n = {n}: residual {}", f(root));
    }
}

#[test]
fn brent_exact_zero_at_left_endpoint() {
    let r = brent_root(|x| x * (x - 1.0), 0.0, 0.5, &RootOpts::default()).unwrap();
    assert_eq!(r, 0.0);
}

#[test]
fn brent_exact_zero_at_right_endpoint() {
    let r = brent_root(|x| x * (x - 1.0), 0.5, 1.0, &RootOpts::default()).unwrap();
    assert_eq!(r, 1.0);
}

#[test]
fn brent_invalid_bracket_is_invalid_argument() {
    let e = brent_root(|x| x * x + 1.0, -1.0, 1.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn brent_non_finite_endpoint_value_is_error() {
    let e = brent_root(|x| 1.0 / x, 0.0, 1.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
    let e = brent_root(|x| x, f64::NAN, 1.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
    let e = brent_root(|x| x, f64::NEG_INFINITY, 1.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn brent_non_finite_interior_value_is_computation_failed() {
    // Sign change across a pole: f(0.5) = NaN.
    let f = |x: f64| {
        if (x - 0.5).abs() < 1e-3 {
            f64::NAN
        } else {
            x - 0.5
        }
    };
    let e = brent_root(f, 0.0, 1.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn brent_iteration_cap_is_error_not_hang() {
    let opts = RootOpts {
        max_iter: 2,
        ..RootOpts::default()
    };
    let e = brent_root(|x| x * x - 2.0, 0.0, 2.0, &opts);
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn brent_negative_tolerance_is_invalid_argument() {
    let opts = RootOpts {
        xtol: -1.0,
        ..RootOpts::default()
    };
    assert!(matches!(
        brent_root(|x| x, -1.0, 1.0, &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn brent_handles_reversed_bracket() {
    let r = brent_root(|x| x * x - 2.0, 2.0, 0.0, &RootOpts::default()).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-12);
}

#[test]
fn brent_agrees_with_bisect() {
    let f = |x: f64| x.exp() - 3.0 * x;
    let b = brent_root(f, 0.0, 1.0, &RootOpts::default()).unwrap();
    let s = bisect(f, 0.0, 1.0, &RootOpts::default()).unwrap();
    assert!((b - s).abs() < 1e-10, "brent {b} vs bisect {s}");
}

#[test]
fn brent_uses_fewer_evaluations_than_bisect() {
    let count = Cell::new(0usize);
    let f = |x: f64| {
        count.set(count.get() + 1);
        x * x * x - x - 1.0
    };
    brent_root(f, 1.0, 2.0, &RootOpts::default()).unwrap();
    let brent_evals = count.get();
    count.set(0);
    bisect(f, 1.0, 2.0, &RootOpts::default()).unwrap();
    let bisect_evals = count.get();
    assert!(
        brent_evals < bisect_evals,
        "brent {brent_evals} vs bisect {bisect_evals}"
    );
}

#[test]
fn brent_tight_bracket_on_steep_function() {
    let r = brent_root(|x| 1e10 * (x - 0.3), 0.0, 1.0, &RootOpts::default()).unwrap();
    assert!((r - 0.3).abs() < 1e-12);
}

// ═══════════════════════════════════════════════════════════════════════════
// bisect
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bisect_sqrt2() {
    let r = bisect(|x| x * x - 2.0, 0.0, 2.0, &RootOpts::default()).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-11, "{r}");
}

#[test]
fn bisect_exact_midpoint_zero() {
    let r = bisect(|x| x - 0.5, 0.0, 1.0, &RootOpts::default()).unwrap();
    assert_eq!(r, 0.5);
}

#[test]
fn bisect_invalid_bracket_is_error() {
    assert!(matches!(
        bisect(|x| x * x + 1.0, -1.0, 1.0, &RootOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn bisect_iteration_cap_is_error() {
    let opts = RootOpts {
        max_iter: 3,
        ..RootOpts::default()
    };
    assert!(matches!(
        bisect(|x| x * x - 2.0, 0.0, 2.0, &opts),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn bisect_respects_loose_tolerance() {
    let opts = RootOpts {
        xtol: 1e-3,
        rtol: 0.0,
        max_iter: 100,
    };
    let r = bisect(|x| x * x - 2.0, 0.0, 2.0, &opts).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-3);
}

// ═══════════════════════════════════════════════════════════════════════════
// newton_root
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn newton_cube_root_of_two() {
    let r = newton_root(
        |x| x * x * x - 2.0,
        |x| 3.0 * x * x,
        1.0,
        &RootOpts::default(),
    )
    .unwrap();
    assert!((r - 2f64.cbrt()).abs() < 1e-12, "{r}");
}

#[test]
fn newton_diverging_start_is_error_not_hang() {
    let start = std::time::Instant::now();
    let e = newton_root(
        f64::atan,
        |x| 1.0 / (1.0 + x * x),
        2.0,
        &RootOpts::default(),
    );
    assert!(e.is_err(), "{e:?}");
    assert!(start.elapsed() < time_budget(1));
}

#[test]
fn newton_cycle_hits_iteration_cap() {
    // x³ − 2x + 2 from x₀ = 0 cycles 0 → 1 → 0 → …
    let e = newton_root(
        |x| x * x * x - 2.0 * x + 2.0,
        |x| 3.0 * x * x - 2.0,
        0.0,
        &RootOpts::default(),
    );
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn newton_zero_derivative_is_error() {
    let e = newton_root(|x| x * x - 1.0, |x| 2.0 * x, 0.0, &RootOpts::default());
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn newton_exact_root_start_returns_immediately() {
    let r = newton_root(|x| x * x - 4.0, |x| 2.0 * x, 2.0, &RootOpts::default()).unwrap();
    assert_eq!(r, 2.0);
}

#[test]
fn newton_non_finite_start_is_invalid_argument() {
    assert!(matches!(
        newton_root(|x| x, |_| 1.0, f64::NAN, &RootOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn newton_matches_brent_on_transcendental() {
    let f = |x: f64| x.exp() - 3.0 * x;
    let n = newton_root(f, |x| x.exp() - 3.0, 0.5, &RootOpts::default()).unwrap();
    let b = brent_root(f, 0.0, 1.0, &RootOpts::default()).unwrap();
    assert!((n - b).abs() < 1e-11, "newton {n} vs brent {b}");
}

// ═══════════════════════════════════════════════════════════════════════════
// nelder_mead
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nelder_mead_rosenbrock_from_classic_start() {
    let opts = MinimizeOpts {
        max_iter: 2000,
        ..MinimizeOpts::default()
    };
    let r = nelder_mead(rosenbrock, &[-1.2, 1.0], &opts).unwrap();
    assert!(r.converged, "{r:?}");
    assert!((r.x[0] - 1.0).abs() < 1e-4, "{:?}", r.x);
    assert!((r.x[1] - 1.0).abs() < 1e-4, "{:?}", r.x);
    assert!(r.fun < 1e-8);
}

#[test]
fn nelder_mead_quadratic_bowl_5d() {
    let target = [1.0, -2.0, 3.0, -4.0, 5.0];
    let bowl = |p: &[f64]| {
        p.iter()
            .zip(&target)
            .map(|(x, t)| (x - t) * (x - t))
            .sum::<f64>()
    };
    let r = nelder_mead(bowl, &[0.0; 5], &MinimizeOpts::default()).unwrap();
    assert!(r.converged, "{r:?}");
    for (x, t) in r.x.iter().zip(&target) {
        assert!((x - t).abs() < 1e-5, "{:?}", r.x);
    }
    assert!(r.fun < 1e-10);
}

#[test]
fn nelder_mead_one_dimensional() {
    let r = nelder_mead(
        |p: &[f64]| (p[0] - 3.0).powi(2) + 1.0,
        &[0.0],
        &MinimizeOpts::default(),
    )
    .unwrap();
    assert!(r.converged);
    assert!((r.x[0] - 3.0).abs() < 1e-6, "{:?}", r.x);
    assert!((r.fun - 1.0).abs() < 1e-12);
}

#[test]
fn nelder_mead_counts_evaluations() {
    let count = Cell::new(0usize);
    let f = |p: &[f64]| {
        count.set(count.get() + 1);
        p[0] * p[0] + p[1] * p[1]
    };
    let r = nelder_mead(f, &[1.0, 1.0], &MinimizeOpts::default()).unwrap();
    assert_eq!(r.evaluations, count.get());
    assert!(r.evaluations >= 3, "at least the initial simplex");
    assert!(r.iterations > 0);
}

#[test]
fn nelder_mead_empty_start_is_invalid_argument() {
    let e = nelder_mead(|_: &[f64]| 0.0, &[], &MinimizeOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn nelder_mead_non_finite_start_value_is_invalid_argument() {
    let e = nelder_mead(|p: &[f64]| p[0].ln(), &[-1.0], &MinimizeOpts::default());
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn nelder_mead_budget_exhaustion_reports_not_converged() {
    let opts = MinimizeOpts {
        max_iter: 3,
        ..MinimizeOpts::default()
    };
    let r = nelder_mead(rosenbrock, &[-1.2, 1.0], &opts).unwrap();
    assert!(!r.converged);
    assert_eq!(r.iterations, 3);
    assert!(
        r.fun <= rosenbrock(&[-1.2, 1.0]),
        "never worse than the start"
    );
}

#[test]
fn nelder_mead_explicit_initial_step() {
    let opts = MinimizeOpts {
        initial_step: 1.0,
        ..MinimizeOpts::default()
    };
    let r = nelder_mead(|p: &[f64]| (p[0] - 10.0).powi(2), &[0.0], &opts).unwrap();
    assert!((r.x[0] - 10.0).abs() < 1e-6);
}

#[test]
fn nelder_mead_treats_nan_as_worse() {
    // NaN outside the unit disc; minimum at the origin.
    let f = |p: &[f64]| {
        let r2 = p[0] * p[0] + p[1] * p[1];
        if r2 > 1.0 { f64::NAN } else { r2 }
    };
    let r = nelder_mead(f, &[0.5, 0.5], &MinimizeOpts::default()).unwrap();
    assert!(r.fun < 1e-10, "{r:?}");
}

#[test]
fn nelder_mead_unbounded_below_is_computation_failed() {
    // Finite inside |x| ≤ 0.5 and decreasing outward, −∞ beyond.
    let f = |p: &[f64]| {
        if p[0].abs() > 0.5 {
            f64::NEG_INFINITY
        } else {
            -p[0] * p[0]
        }
    };
    let e = nelder_mead(f, &[0.4], &MinimizeOpts::default());
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn nelder_mead_negative_tolerance_is_invalid_argument() {
    let opts = MinimizeOpts {
        xtol: -1.0,
        ..MinimizeOpts::default()
    };
    assert!(matches!(
        nelder_mead(|p: &[f64]| p[0] * p[0], &[1.0], &opts),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// minimize_scalar / golden_section
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn brent_min_shifted_parabola() {
    let (x, fx) = minimize_scalar(
        |x| (x - 1.0).powi(2) + 3.0,
        -5.0,
        5.0,
        &MinimizeOpts::default(),
    )
    .unwrap();
    assert!((x - 1.0).abs() < 1e-6, "{x}");
    assert!((fx - 3.0).abs() < 1e-12, "{fx}");
}

#[test]
fn golden_min_shifted_parabola() {
    let (x, fx) = golden_section(
        |x| (x - 1.0).powi(2) + 3.0,
        -5.0,
        5.0,
        &MinimizeOpts::default(),
    )
    .unwrap();
    assert!((x - 1.0).abs() < 1e-6, "{x}");
    assert!((fx - 3.0).abs() < 1e-12, "{fx}");
}

#[test]
fn brent_min_x_ln_x() {
    let (x, fx) = minimize_scalar(|x| x * x.ln(), 0.1, 2.0, &MinimizeOpts::default()).unwrap();
    let e_inv = (-1.0f64).exp();
    assert!((x - e_inv).abs() < 1e-6, "{x}");
    assert!((fx + e_inv).abs() < 1e-12, "{fx}");
}

#[test]
fn golden_min_x_ln_x() {
    let (x, fx) = golden_section(|x| x * x.ln(), 0.1, 2.0, &MinimizeOpts::default()).unwrap();
    let e_inv = (-1.0f64).exp();
    assert!((x - e_inv).abs() < 1e-6, "{x}");
    assert!((fx + e_inv).abs() < 1e-12, "{fx}");
}

#[test]
fn brent_min_bracket_selects_one_of_two_minima() {
    // f = (x² − 1)² has minima at ±1.
    let f = |x: f64| (x * x - 1.0).powi(2);
    let (xl, _) = minimize_scalar(f, -2.0, 0.0, &MinimizeOpts::default()).unwrap();
    let (xr, _) = minimize_scalar(f, 0.0, 2.0, &MinimizeOpts::default()).unwrap();
    assert!((xl + 1.0).abs() < 1e-6, "{xl}");
    assert!((xr - 1.0).abs() < 1e-6, "{xr}");
}

#[test]
fn golden_min_bracket_selects_one_of_two_minima() {
    let f = |x: f64| (x * x - 1.0).powi(2);
    let (xl, _) = golden_section(f, -2.0, 0.0, &MinimizeOpts::default()).unwrap();
    let (xr, _) = golden_section(f, 0.0, 2.0, &MinimizeOpts::default()).unwrap();
    assert!((xl + 1.0).abs() < 1e-6, "{xl}");
    assert!((xr - 1.0).abs() < 1e-6, "{xr}");
}

#[test]
fn brent_min_uses_fewer_evaluations_than_golden_on_smooth_function() {
    let count = Cell::new(0usize);
    let f = |x: f64| {
        count.set(count.get() + 1);
        (x - 0.3).powi(2) + x.sin()
    };
    minimize_scalar(f, -2.0, 2.0, &MinimizeOpts::default()).unwrap();
    let brent = count.get();
    count.set(0);
    golden_section(f, -2.0, 2.0, &MinimizeOpts::default()).unwrap();
    let golden = count.get();
    assert!(brent < golden, "brent {brent} vs golden {golden}");
}

#[test]
fn scalar_minimisers_accept_reversed_interval() {
    let (x, _) =
        minimize_scalar(|x| (x - 1.0).powi(2), 5.0, -5.0, &MinimizeOpts::default()).unwrap();
    assert!((x - 1.0).abs() < 1e-6);
    let (x, _) =
        golden_section(|x| (x - 1.0).powi(2), 5.0, -5.0, &MinimizeOpts::default()).unwrap();
    assert!((x - 1.0).abs() < 1e-6);
}

#[test]
fn scalar_minimisers_reject_degenerate_interval() {
    assert!(matches!(
        minimize_scalar(|x| x, 1.0, 1.0, &MinimizeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        golden_section(|x| x, 1.0, 1.0, &MinimizeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        minimize_scalar(|x| x, 0.0, f64::INFINITY, &MinimizeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn scalar_minimisers_report_non_finite_values() {
    let e = minimize_scalar(|x| x.ln(), -1.0, 1.0, &MinimizeOpts::default());
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
    let e = golden_section(|x| x.ln(), -1.0, 1.0, &MinimizeOpts::default());
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn scalar_minimiser_iteration_cap_is_error() {
    let opts = MinimizeOpts {
        max_iter: 2,
        ..MinimizeOpts::default()
    };
    assert!(matches!(
        golden_section(|x| (x - 1.0).powi(2), -5.0, 5.0, &opts),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn brent_min_monotone_function_returns_endpoint_region() {
    let (x, fx) = minimize_scalar(|x| x, 0.0, 1.0, &MinimizeOpts::default()).unwrap();
    assert!(x < 1e-6, "{x}");
    assert!(fx < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// differential_evolution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn de_rastrigin_2d_default_seed() {
    let start = std::time::Instant::now();
    let bounds = [(-5.12, 5.12), (-5.12, 5.12)];
    let r = differential_evolution(rastrigin, &bounds, &DeOpts::default()).unwrap();
    assert!(r.fun < 1e-6, "{r:?}");
    assert!(r.x.iter().all(|x| x.abs() < 1e-3), "{:?}", r.x);
    assert!(start.elapsed() < time_budget(2), "{:?}", start.elapsed());
}

#[test]
fn de_is_deterministic_for_a_seed() {
    let bounds = [(-5.12, 5.12), (-5.12, 5.12)];
    let opts = DeOpts {
        seed: 12345,
        ..DeOpts::default()
    };
    let a = differential_evolution(rastrigin, &bounds, &opts).unwrap();
    let b = differential_evolution(rastrigin, &bounds, &opts).unwrap();
    assert_eq!(a, b);
}

#[test]
fn de_different_seeds_all_converge() {
    let start = std::time::Instant::now();
    let bounds = [(-5.12, 5.12), (-5.12, 5.12)];
    let mut results = Vec::new();
    for seed in [1u64, 7, 42, 2024] {
        let opts = DeOpts {
            seed,
            ..DeOpts::default()
        };
        let r = differential_evolution(rastrigin, &bounds, &opts).unwrap();
        assert!(r.fun < 1e-6, "seed {seed}: {r:?}");
        results.push(r);
    }
    // Different seeds explore differently: the evaluation counts should not all agree.
    let first = results[0].evaluations;
    assert!(results.iter().any(|r| r.evaluations != first));
    assert!(start.elapsed() < time_budget(4), "{:?}", start.elapsed());
}

#[test]
fn de_respects_bounds_on_every_evaluation() {
    let bounds = [(-1.0, 2.0), (0.5, 3.0), (-4.0, -2.0)];
    let violations = Cell::new(0usize);
    let f = |p: &[f64]| {
        for (x, &(lo, hi)) in p.iter().zip(&bounds) {
            if *x < lo || *x > hi {
                violations.set(violations.get() + 1);
            }
        }
        p.iter().map(|x| x * x).sum::<f64>()
    };
    let r = differential_evolution(f, &bounds, &DeOpts::default()).unwrap();
    assert_eq!(violations.get(), 0);
    for (x, &(lo, hi)) in r.x.iter().zip(&bounds) {
        assert!(*x >= lo && *x <= hi, "{:?}", r.x);
    }
    // Minimum of Σx² over the box: (0, 0.5, −2).
    assert!(
        (r.x[0]).abs() < 1e-6 && (r.x[1] - 0.5).abs() < 1e-6 && (r.x[2] + 2.0).abs() < 1e-6,
        "{:?}",
        r.x
    );
}

#[test]
fn de_counts_evaluations() {
    let count = Cell::new(0usize);
    let f = |p: &[f64]| {
        count.set(count.get() + 1);
        (p[0] - 0.5).powi(2)
    };
    let r = differential_evolution(f, &[(-1.0, 1.0)], &DeOpts::default()).unwrap();
    assert_eq!(r.evaluations, count.get());
    assert!(r.evaluations >= 8, "initial population is evaluated");
}

#[test]
fn de_converged_flag_on_easy_problem() {
    let r = differential_evolution(
        |p: &[f64]| (p[0] - 0.25).powi(2) + (p[1] + 0.75).powi(2),
        &[(-2.0, 2.0), (-2.0, 2.0)],
        &DeOpts::default(),
    )
    .unwrap();
    assert!(r.converged, "{r:?}");
    assert!(r.iterations < 300);
    assert!((r.x[0] - 0.25).abs() < 1e-6 && (r.x[1] + 0.75).abs() < 1e-6);
}

#[test]
fn de_generation_cap_reports_not_converged() {
    let opts = DeOpts {
        max_generations: 2,
        ..DeOpts::default()
    };
    let r = differential_evolution(rastrigin, &[(-5.12, 5.12), (-5.12, 5.12)], &opts).unwrap();
    assert!(!r.converged);
    assert_eq!(r.iterations, 2);
    assert!(r.fun.is_finite());
}

#[test]
fn de_rejects_bad_inputs() {
    let f = |p: &[f64]| p[0];
    assert!(matches!(
        differential_evolution(f, &[], &DeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        differential_evolution(f, &[(1.0, 0.0)], &DeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        differential_evolution(f, &[(0.0, f64::INFINITY)], &DeOpts::default()),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let bad_pop = DeOpts {
        population: 3,
        ..DeOpts::default()
    };
    assert!(matches!(
        differential_evolution(f, &[(0.0, 1.0)], &bad_pop),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let bad_cr = DeOpts {
        crossover: 1.5,
        ..DeOpts::default()
    };
    assert!(matches!(
        differential_evolution(f, &[(0.0, 1.0)], &bad_cr),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let bad_f = DeOpts {
        differential_weight: 0.0,
        ..DeOpts::default()
    };
    assert!(matches!(
        differential_evolution(f, &[(0.0, 1.0)], &bad_f),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn de_all_nan_objective_is_computation_failed() {
    let opts = DeOpts {
        max_generations: 5,
        ..DeOpts::default()
    };
    let e = differential_evolution(|_: &[f64]| f64::NAN, &[(0.0, 1.0)], &opts);
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn de_finds_global_minimum_among_local_ones() {
    // f(x) = x⁴ − 4x² + x has a local minimum near x ≈ 1.37 and the global one near x ≈ −1.47.
    let f = |p: &[f64]| p[0].powi(4) - 4.0 * p[0].powi(2) + p[0];
    let r = differential_evolution(f, &[(-3.0, 3.0)], &DeOpts::default()).unwrap();
    assert!(r.x[0] < -1.0, "{:?}", r.x);
    let (local, _) = minimize_scalar(|x| f(&[x]), 0.5, 2.5, &MinimizeOpts::default()).unwrap();
    assert!(r.fun < f(&[local]) - 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// poly_fit / linear_fit / eval_poly
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poly_fit_recovers_quadratic_from_six_points() {
    let xs: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 0.5 - 1.5 * x + 2.25 * x * x).collect();
    let c = poly_fit(&xs, &ys, 2).unwrap();
    assert_eq!(c.len(), 3);
    assert!((c[0] - 0.5).abs() < 1e-9, "{c:?}");
    assert!((c[1] + 1.5).abs() < 1e-9, "{c:?}");
    assert!((c[2] - 2.25).abs() < 1e-9, "{c:?}");
}

#[test]
fn poly_fit_coefficients_are_ascending() {
    // y = x³ exactly: only the cubic coefficient is non-zero.
    let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let ys: Vec<f64> = xs.iter().map(|x| x * x * x).collect();
    let c = poly_fit(&xs, &ys, 3).unwrap();
    assert!(
        c[0].abs() < 1e-9 && c[1].abs() < 1e-9 && c[2].abs() < 1e-9,
        "{c:?}"
    );
    assert!((c[3] - 1.0).abs() < 1e-9, "{c:?}");
}

#[test]
fn poly_fit_noisy_line_slope() {
    // Deterministic "noise" that sums to zero over a period.
    let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
    let noise = [0.01, -0.02, 0.015, -0.005];
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, x)| 2.0 * x + 1.0 + noise[i % 4])
        .collect();
    let c = poly_fit(&xs, &ys, 1).unwrap();
    assert!((c[1] - 2.0).abs() < 1e-2, "{c:?}");
    assert!((c[0] - 1.0).abs() < 2e-2, "{c:?}");
}

#[test]
fn poly_fit_interpolates_when_degree_plus_one_equals_len() {
    let xs = [1.0, 2.0, 4.0];
    let ys = [3.0, -1.0, 7.0];
    let c = poly_fit(&xs, &ys, 2).unwrap();
    for (x, y) in xs.iter().zip(&ys) {
        assert!((eval_poly(&c, *x) - y).abs() < 1e-9);
    }
}

#[test]
fn poly_fit_least_squares_residual_is_orthogonal_to_columns() {
    let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
    let ys = [1.0, 0.0, 4.0, 2.0, 5.0];
    let c = poly_fit(&xs, &ys, 1).unwrap();
    let residual: Vec<f64> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| y - eval_poly(&c, *x))
        .collect();
    // Residual ⟂ span{1, x}.
    let s0: f64 = residual.iter().sum();
    let s1: f64 = residual.iter().zip(&xs).map(|(r, x)| r * x).sum();
    assert!(s0.abs() < 1e-10 && s1.abs() < 1e-10, "{s0} {s1}");
}

#[test]
fn linear_fit_exact_line() {
    let xs: Vec<f64> = (0..10).map(f64::from).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x + 1.0).collect();
    let (slope, intercept) = linear_fit(&xs, &ys).unwrap();
    assert!((slope - 3.0).abs() < 1e-12, "{slope}");
    assert!((intercept - 1.0).abs() < 1e-12, "{intercept}");
}

#[test]
fn linear_fit_matches_closed_form() {
    let xs = [1.0, 2.0, 3.0, 5.0, 8.0];
    let ys = [2.0, 2.5, 3.9, 6.1, 9.8];
    let (slope, intercept) = linear_fit(&xs, &ys).unwrap();
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let m = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let b = (sy - m * sx) / n;
    assert!((slope - m).abs() < 1e-12 && (intercept - b).abs() < 1e-12);
}

#[test]
fn poly_fit_ill_conditioned_still_finite_and_accurate() {
    // Abscissae far from the origin: raw Vandermonde is badly conditioned.
    let xs: Vec<f64> = (0..12).map(|i| 1000.0 + i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| 2.0 + 0.5 * x - 0.001 * x * x).collect();
    let c = poly_fit(&xs, &ys, 2).unwrap();
    assert!(c.iter().all(|v| v.is_finite()), "{c:?}");
    for (x, y) in xs.iter().zip(&ys) {
        assert!(
            (eval_poly(&c, *x) - y).abs() < 1e-6 * y.abs().max(1.0),
            "{c:?}"
        );
    }
}

#[test]
fn poly_fit_degree_too_high_is_invalid_argument() {
    let e = poly_fit(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], 3);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
    let e = poly_fit(&[0.0, 1.0], &[1.0, 2.0], 2);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn poly_fit_length_mismatch_is_invalid_argument() {
    let e = poly_fit(&[0.0, 1.0, 2.0], &[1.0, 2.0], 1);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn poly_fit_non_finite_sample_is_invalid_argument() {
    let e = poly_fit(&[0.0, 1.0, f64::NAN], &[1.0, 2.0, 3.0], 1);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn poly_fit_repeated_abscissae_is_rank_deficient() {
    let e = poly_fit(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 1);
    assert!(
        matches!(e, Err(SymplexError::ComputationFailed { .. })),
        "{e:?}"
    );
}

#[test]
fn poly_fit_degree_zero_is_the_mean() {
    let c = poly_fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0], 0).unwrap();
    assert_eq!(c.len(), 1);
    assert!((c[0] - 4.0).abs() < 1e-12);
}

#[test]
fn eval_poly_horner() {
    assert_eq!(eval_poly(&[1.0, 2.0, 3.0], 2.0), 17.0);
    assert_eq!(eval_poly(&[5.0], 100.0), 5.0);
    assert_eq!(eval_poly(&[], 100.0), 0.0);
    assert_eq!(eval_poly(&[0.0, 0.0, 1.0], -3.0), 9.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// poly_fit_exact
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn poly_fit_exact_recovers_rational_quadratic() {
    // y = x²/3 − x/2 + 1/7 at x = −2, −1, 0, 1, 2, 3.
    let pts: Vec<(Ratio<BigInt>, Ratio<BigInt>)> = (-2..=3)
        .map(|i| {
            let x = q(i, 1);
            let y = &x * &x * q(1, 3) - &x * q(1, 2) + q(1, 7);
            (x, y)
        })
        .collect();
    let c = poly_fit_exact(&pts, 2).unwrap();
    assert_eq!(c, vec![q(1, 7), q(-1, 2), q(1, 3)]);
}

#[test]
fn poly_fit_exact_with_rational_abscissae() {
    let pts = [(q(1, 2), q(3, 4)), (q(-1, 3), q(-1, 9)), (q(5, 7), q(2, 1))];
    let c = poly_fit_exact(&pts, 2).unwrap();
    // Interpolation: reproduces every point exactly.
    for (x, y) in &pts {
        let v = &c[0] + &c[1] * x + &c[2] * x * x;
        assert_eq!(&v, y);
    }
}

#[test]
fn poly_fit_exact_overdetermined_consistent_is_exact() {
    // Twenty points on y = 3x − 5/2, fitting a line.
    let pts: Vec<_> = (0..20).map(|i| (q(i, 4), q(3 * i, 4) - q(5, 2))).collect();
    let c = poly_fit_exact(&pts, 1).unwrap();
    assert_eq!(c, vec![q(-5, 2), q(3, 1)]);
}

#[test]
fn poly_fit_exact_inconsistent_data_is_exact_projection() {
    // Data not on a line; compare with the normal equations solved by hand.
    let pts = [
        (q(0, 1), q(1, 1)),
        (q(1, 1), q(0, 1)),
        (q(2, 1), q(4, 1)),
        (q(3, 1), q(2, 1)),
    ];
    let c = poly_fit_exact(&pts, 1).unwrap();
    // Normal equations for a line: [n Σx; Σx Σx²]·[c0; c1] = [Σy; Σxy].
    let n = q(4, 1);
    let sx = q(6, 1);
    let sxx = q(14, 1);
    let sy = q(7, 1);
    let sxy = q(14, 1);
    let det = &n * &sxx - &sx * &sx;
    let c0 = (&sy * &sxx - &sx * &sxy) / &det;
    let c1 = (&n * &sxy - &sx * &sy) / &det;
    assert_eq!(c, vec![c0, c1]);
    // Residual is orthogonal to the columns 1 and x — exactly.
    let r: Vec<Ratio<BigInt>> = pts.iter().map(|(x, y)| y - (&c[0] + &c[1] * x)).collect();
    assert!(
        r.iter().sum::<Ratio<BigInt>>().is_integer() && r.iter().sum::<Ratio<BigInt>>() == q(0, 1)
    );
    let s1: Ratio<BigInt> = r.iter().zip(&pts).map(|(ri, (x, _))| ri * x).sum();
    assert_eq!(s1, q(0, 1));
}

#[test]
fn poly_fit_exact_degree_too_high_is_invalid_argument() {
    let pts = [(q(0, 1), q(1, 1)), (q(1, 1), q(2, 1))];
    assert!(matches!(
        poly_fit_exact(&pts, 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        poly_fit_exact(&[], 0),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn poly_fit_exact_repeated_abscissae_is_singular() {
    let pts = [(q(1, 1), q(1, 1)), (q(1, 1), q(2, 1)), (q(1, 1), q(3, 1))];
    assert!(matches!(
        poly_fit_exact(&pts, 1),
        Err(SymplexError::ComputationFailed { .. })
    ));
}

#[test]
fn poly_fit_exact_agrees_with_float_fit() {
    let pts: Vec<_> = (0..7)
        .map(|i| (q(i, 1), q(i * i * i - 2 * i + 1, 3)))
        .collect();
    let exact = poly_fit_exact(&pts, 3).unwrap();
    let xs: Vec<f64> = (0..7).map(f64::from).collect();
    let ys: Vec<f64> = (0..7)
        .map(|i| f64::from(i * i * i - 2 * i + 1) / 3.0)
        .collect();
    let approx = poly_fit(&xs, &ys, 3).unwrap();
    for (e, a) in exact.iter().zip(&approx) {
        let ef = e.numer().to_string().parse::<f64>().unwrap()
            / e.denom().to_string().parse::<f64>().unwrap();
        assert!((ef - a).abs() < 1e-9, "{ef} vs {a}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// trapezoid
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trapezoid_x_squared_on_unit_interval() {
    let xs: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
    let area = trapezoid(&ys, &xs).unwrap();
    assert!((area - 1.0 / 3.0).abs() < 1e-6, "{area}");
}

#[test]
fn trapezoid_is_exact_for_linear_data_on_uneven_grid() {
    let xs = [0.0, 0.1, 0.5, 0.7, 2.0];
    let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x + 1.0).collect();
    // ∫₀² (3x + 1) dx = 6 + 2 = 8.
    assert!((trapezoid(&ys, &xs).unwrap() - 8.0).abs() < 1e-12);
}

#[test]
fn trapezoid_degenerate_inputs() {
    assert_eq!(trapezoid(&[], &[]).unwrap(), 0.0);
    assert_eq!(trapezoid(&[5.0], &[1.0]).unwrap(), 0.0);
}

#[test]
fn trapezoid_length_mismatch_is_invalid_argument() {
    assert!(matches!(
        trapezoid(&[1.0, 2.0], &[0.0]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex conveniences
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ex_find_root_bracket_sqrt2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) - 2).find_root_bracket(&x, 0.0, 2.0).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-12, "{r}");
}

#[test]
fn ex_find_root_bracket_transcendental() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x·eˣ = 1  ⇒  x = W(1) = Ω ≈ 0.567143…
    let r = (&x * x.exp() - 1).find_root_bracket(&x, 0.0, 1.0).unwrap();
    assert!((r - 0.567_143_290_409_784).abs() < 1e-12, "{r}");
}

#[test]
fn ex_find_root_bracket_with_options() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let opts = RootOpts {
        xtol: 1e-4,
        rtol: 0.0,
        max_iter: 100,
    };
    let r = (x.sin() - ctx.rational(1, 2))
        .find_root_bracket_with(&x, 0.0, 1.0, &opts)
        .unwrap();
    assert!((r - PI / 6.0).abs() < 1e-4, "{r}");
}

#[test]
fn ex_find_root_bracket_extra_free_symbol_is_free_symbol_error() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let e = (&x.powi(2) - &a).find_root_bracket(&x, 0.0, 2.0);
    match e {
        Err(SymplexError::FreeSymbol { name }) => assert_eq!(name, "a"),
        other => panic!("expected FreeSymbol, got {other:?}"),
    }
}

#[test]
fn ex_find_root_bracket_non_symbol_var_is_invalid_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x.powi(2) - 2).find_root_bracket(&x.powi(2), 0.0, 2.0);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn ex_find_root_bracket_no_sign_change_is_invalid_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = (&x.powi(2) + 1).find_root_bracket(&x, -1.0, 1.0);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn ex_minimize_numeric_bowl() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let bowl = (&x - 1).powi(2) + (&y + 2).powi(2);
    let r = bowl.minimize_numeric(&[&x, &y], &[0.0, 0.0]).unwrap();
    assert!(r.converged, "{r:?}");
    assert!(
        (r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] + 2.0).abs() < 1e-6,
        "{:?}",
        r.x
    );
    assert!(r.fun < 1e-12);
}

#[test]
fn ex_minimize_numeric_variable_order_matters() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let bowl = (&x - 1).powi(2) + (&y + 2).powi(2);
    let r = bowl.minimize_numeric(&[&y, &x], &[0.0, 0.0]).unwrap();
    assert!(
        (r.x[0] + 2.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6,
        "{:?}",
        r.x
    );
}

#[test]
fn ex_minimize_numeric_with_rosenbrock() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let rosen = (1 - &x).powi(2) + 100 * (&y - &x.powi(2)).powi(2);
    let opts = MinimizeOpts {
        max_iter: 2000,
        ..MinimizeOpts::default()
    };
    let r = rosen
        .minimize_numeric_with(&[&x, &y], &[-1.2, 1.0], &opts)
        .unwrap();
    assert!(
        (r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4,
        "{:?}",
        r.x
    );
}

#[test]
fn ex_minimize_numeric_extra_free_symbol_is_error() {
    let ctx = Context::new();
    let (x, y, c) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("c"));
    let e = (&x.powi(2) + &y.powi(2) + &c).minimize_numeric(&[&x, &y], &[0.0, 0.0]);
    assert!(matches!(e, Err(SymplexError::FreeSymbol { .. })), "{e:?}");
}

#[test]
fn ex_minimize_numeric_length_mismatch_is_invalid_argument() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x.powi(2) + &y.powi(2)).minimize_numeric(&[&x, &y], &[0.0]);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
    let e = (&x.powi(2) + &y.powi(2)).minimize_numeric(&[], &[]);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn ex_minimize_numeric_with_special_function() {
    // Γ(x) has its minimum on (0, ∞) at x ≈ 1.4616321…
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = x.gamma().minimize_numeric(&[&x], &[1.0]).unwrap();
    assert!((r.x[0] - 1.461_632_144_968_362_3).abs() < 1e-5, "{:?}", r.x);
}

#[test]
fn ex_minimize_scalar_numeric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (xm, fm) = ((&x - 1).powi(2) + 3)
        .minimize_scalar_numeric(&x, -5.0, 5.0)
        .unwrap();
    assert!((xm - 1.0).abs() < 1e-6, "{xm}");
    assert!((fm - 3.0).abs() < 1e-12, "{fm}");
}

#[test]
fn ex_minimize_scalar_numeric_x_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (xm, fm) = (&x * x.ln()).minimize_scalar_numeric(&x, 0.1, 2.0).unwrap();
    let e_inv = (-1.0f64).exp();
    assert!((xm - e_inv).abs() < 1e-6 && (fm + e_inv).abs() < 1e-12);
}

#[test]
fn ex_minimize_scalar_numeric_extra_symbol_is_error() {
    let ctx = Context::new();
    let (x, k) = (ctx.symbol("x"), ctx.symbol("k"));
    let e = (&x.powi(2) + &k).minimize_scalar_numeric(&x, -1.0, 1.0);
    assert!(matches!(e, Err(SymplexError::FreeSymbol { .. })), "{e:?}");
}

#[test]
fn ex_minimize_global_numeric_rastrigin() {
    let start = std::time::Instant::now();
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let two_pi = ctx.pi() * 2;
    let ras = 20 + &x.powi(2) + &y.powi(2) - 10 * (&two_pi * &x).cos() - 10 * (&two_pi * &y).cos();
    let r = ras
        .minimize_global_numeric(
            &[&x, &y],
            &[(-5.12, 5.12), (-5.12, 5.12)],
            &DeOpts::default(),
        )
        .unwrap();
    assert!(r.fun < 1e-6, "{r:?}");
    assert!(r.x.iter().all(|v| v.abs() < 1e-3), "{:?}", r.x);
    assert!(start.elapsed() < time_budget(2), "{:?}", start.elapsed());
}

#[test]
fn ex_minimize_global_numeric_bounds_mismatch_is_invalid_argument() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let e = (&x.powi(2) + &y.powi(2)).minimize_global_numeric(
        &[&x, &y],
        &[(-1.0, 1.0)],
        &DeOpts::default(),
    );
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn ex_poly_fit_points_exact_rational_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts: Vec<(Ex, Ex)> = (-2..=3)
        .map(|i| {
            let xi = ctx.int(i);
            let yi =
                &xi.powi(2) * ctx.rational(1, 3) - &xi * ctx.rational(1, 2) + ctx.rational(1, 7);
            (xi, yi.eval())
        })
        .collect();
    let p = Ex::poly_fit_points(&ctx, &pts, &x, 2).unwrap();
    let expected = &x.powi(2) * ctx.rational(1, 3) - &x * ctx.rational(1, 2) + ctx.rational(1, 7);
    assert!((&p - &expected).expand().is_zero_structural(), "{p}");
    assert_eq!(p.degree(&x), Some(2));
}

#[test]
fn ex_poly_fit_points_constant_folds_coordinates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sqrt(4) and 2^3 are rational after eval().
    let pts = [
        (ctx.int(4).sqrt(), ctx.int(2).powi(3)),
        (ctx.int(0), ctx.int(0)),
    ];
    let p = Ex::poly_fit_points(&ctx, &pts, &x, 1).unwrap();
    assert!((&p - &x * 4).expand().is_zero_structural(), "{p}");
}

#[test]
fn ex_poly_fit_points_least_squares_line() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [
        (ctx.int(0), ctx.int(1)),
        (ctx.int(1), ctx.int(0)),
        (ctx.int(2), ctx.int(4)),
        (ctx.int(3), ctx.int(2)),
    ];
    let p = Ex::poly_fit_points(&ctx, &pts, &x, 1).unwrap();
    // Normal-equation solution computed independently: c1 = 7/10, c0 = 7/10.
    let expected = &x * ctx.rational(7, 10) + ctx.rational(7, 10);
    assert!((&p - &expected).expand().is_zero_structural(), "{p}");
}

#[test]
fn ex_poly_fit_points_symbolic_coordinate_is_invalid_argument() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
    let pts = [(ctx.int(0), a.clone()), (ctx.int(1), ctx.int(1))];
    let e = Ex::poly_fit_points(&ctx, &pts, &x, 1);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
    let pts = [(ctx.pi(), ctx.int(0)), (ctx.int(1), ctx.int(1))];
    let e = Ex::poly_fit_points(&ctx, &pts, &x, 1);
    assert!(
        matches!(e, Err(SymplexError::InvalidArgument { .. })),
        "{e:?}"
    );
}

#[test]
fn ex_poly_fit_points_degree_too_high_is_invalid_argument() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [(ctx.int(0), ctx.int(1)), (ctx.int(1), ctx.int(1))];
    assert!(matches!(
        Ex::poly_fit_points(&ctx, &pts, &x, 2),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn ex_poly_fit_points_zero_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [
        (ctx.int(0), ctx.int(0)),
        (ctx.int(1), ctx.int(0)),
        (ctx.int(2), ctx.int(0)),
    ];
    let p = Ex::poly_fit_points(&ctx, &pts, &x, 1).unwrap();
    assert!(p.is_zero_structural(), "{p}");
}

#[test]
fn ex_root_and_min_agree_with_calculus() {
    // The minimum of f is a root of f'.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(4) - 3 * &x.powi(2) + &x;
    let (xm, _) = f.minimize_scalar_numeric(&x, -3.0, -0.5).unwrap();
    let root = f.diff(&x).find_root_bracket(&x, -3.0, -0.5).unwrap();
    assert!((xm - root).abs() < 1e-6, "{xm} vs {root}");
}
