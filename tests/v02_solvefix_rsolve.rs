//! Regression tests for the 0.2 "silently wrong solver results" campaign:
//! `rsolve_linear` with irrational characteristic roots.
//!
//! Root cause fixed: an irreducible cubic / quartic characteristic factor
//! was solved by Cardano / Ferrari radicals; the complex-pair detection
//! (`re`/`im` of nested cube roots) then failed and the constants were
//! fitted through a simplification of nested radicals that never
//! terminated.  Such factors now yield `RootOf` roots (which evaluate
//! numerically to any precision), and the closed form is bounded by a
//! node budget — "expression swell" is an error, never a spin.

use std::time::{Duration, Instant};
use symplex::prelude::*;
use symplex::rsolve::rsolve_linear;

/// `a(k)` from the closed form.
fn seq_at(sol: &Ex, n: &Ex, k: i64) -> f64 {
    sol.subs_i64(n, k)
        .eval_f64()
        .unwrap_or_else(|e| panic!("a({k}) of {sol} does not evaluate: {e}"))
}

/// `a(k)` by iterating the recurrence `Σ cᵢ a(n+i) = 0` from `ics`.
fn iterate(coeffs: &[f64], ics: &[f64], upto: usize) -> Vec<f64> {
    let order = coeffs.len() - 1;
    let mut a: Vec<f64> = ics.to_vec();
    while a.len() <= upto {
        let n = a.len() - order;
        let mut s = 0.0;
        for (i, c) in coeffs.iter().enumerate().take(order) {
            s += c * a[n + i];
        }
        a.push(-s / coeffs[order]);
    }
    a
}

fn check_closed_form(coeffs_i: &[i64], ics_i: &[i64], upto: usize, budget: Duration) -> Ex {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs: Vec<Ex> = coeffs_i.iter().map(|&c| ctx.int(c)).collect();
    let ics: Vec<Ex> = ics_i.iter().map(|&c| ctx.int(c)).collect();
    let t = Instant::now();
    let sol = rsolve_linear(&coeffs, None, &n, &ics).expect("closed form");
    assert!(t.elapsed() < budget, "rsolve took {:?}", t.elapsed());
    let want = iterate(
        &coeffs_i.iter().map(|&c| c as f64).collect::<Vec<_>>(),
        &ics_i.iter().map(|&c| c as f64).collect::<Vec<_>>(),
        upto,
    );
    for (k, w) in want.iter().enumerate() {
        let got = seq_at(&sol, &n, k as i64);
        assert!(
            (got - w).abs() < 1e-7 * w.abs().max(1.0),
            "a({k}) = {got}, expected {w} (closed form {sol})"
        );
    }
    sol
}

#[test]
fn irreducible_cubic_terminates_quickly_and_matches_recurrence() {
    // r³ − r² + 5r − 6: one real irrational root and a complex pair.
    let sol = check_closed_form(&[-6, 5, -1, 1], &[1, 0, 0], 10, Duration::from_secs(2));
    assert!(format!("{sol}").contains("RootOf"), "{sol}");
}

#[test]
fn irreducible_cubic_with_three_real_roots() {
    // r³ − 3r + 1 (casus irreducibilis).
    check_closed_form(&[1, -3, 0, 1], &[1, 2, 3], 10, Duration::from_secs(2));
}

#[test]
fn irreducible_quartic_characteristic_polynomial() {
    // r⁴ + r + 1 has no rational roots and is irreducible over ℚ.
    check_closed_form(&[1, 1, 0, 0, 1], &[1, 0, 0, 0], 10, Duration::from_secs(2));
}

#[test]
fn general_solution_of_irreducible_cubic_uses_rootof_and_three_constants() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(-6), ctx.int(5), ctx.int(-1), ctx.int(1)];
    let t = Instant::now();
    let g = rsolve_linear(&coeffs, None, &n, &[]).unwrap();
    assert!(t.elapsed() < Duration::from_secs(2));
    for k in 1..=3 {
        assert!(g.contains(&ctx.symbol(&format!("C{k}"))), "{g}");
    }
    assert!(!g.contains(&ctx.symbol("C4")), "{g}");
    let s = format!("{g}");
    assert!(s.contains("RootOf(r^3 - r^2 + 5*r - 6, 0)"), "{s}");
}

#[test]
fn quadratic_factors_keep_radicals() {
    // (r² − r − 1)(r − 2) = r³ − 3r² + r + 2: Binet radicals for the
    // quadratic factor.
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(2), ctx.int(1), ctx.int(-3), ctx.int(1)];
    let sol = rsolve_linear(&coeffs, None, &n, &[ctx.int(0), ctx.int(1), ctx.int(1)]).unwrap();
    let s = format!("{sol}");
    assert!(s.contains("sqrt(5)") && !s.contains("RootOf"), "{s}");
    let want = iterate(&[2.0, 1.0, -3.0, 1.0], &[0.0, 1.0, 1.0], 8);
    for (k, w) in want.iter().enumerate() {
        assert!((seq_at(&sol, &n, k as i64) - w).abs() < 1e-8, "a({k})");
    }
}

#[test]
fn fibonacci_binet_unchanged() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let coeffs = [ctx.int(-1), ctx.int(-1), ctx.int(1)];
    let f = rsolve_linear(&coeffs, None, &n, &[ctx.int(0), ctx.int(1)]).unwrap();
    assert!(format!("{f}").contains("sqrt(5)"), "{f}");
    let fib = [0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0];
    for (k, e) in fib.iter().enumerate() {
        assert!((seq_at(&f, &n, k as i64) - e).abs() < 1e-8, "F({k})");
    }
}

#[test]
fn index_symbol_named_r_does_not_clash_with_rootof_variable() {
    let ctx = Context::new();
    let r = ctx.symbol("r");
    let coeffs = [ctx.int(-6), ctx.int(5), ctx.int(-1), ctx.int(1)];
    let sol = rsolve_linear(&coeffs, None, &r, &[ctx.int(1), ctx.int(0), ctx.int(0)]).unwrap();
    assert!(format!("{sol}").contains("RootOf(_r^3"), "{sol}");
    assert!((seq_at(&sol, &r, 3) - 6.0).abs() < 1e-9);
    assert!((seq_at(&sol, &r, 4) - 6.0).abs() < 1e-9);
}
