//! Integration tests for RootOf support in the equation solver.
//!
//! Verifies that irreducible polynomials of degree ≥ 5 return `RootOf`
//! objects and that those objects can be numerically evaluated.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Task 1 — solver emits RootOf for degree ≥ 5
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_quintic_returns_rootof() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ − x − 1 = 0  (irreducible over ℚ, no rational roots)
    let poly = &x.powi(5) - &x - 1;
    let roots = poly.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "quintic x⁵ − x − 1 should return RootOf objects, got empty"
    );
    // All solutions should display as RootOf(…) since there are no
    // rational roots and no radical formula for degree 5.
    for root in &roots {
        let s = format!("{root}");
        assert!(
            s.contains("RootOf"),
            "expected RootOf in display, got: {s}"
        );
    }
}

#[test]
fn solve_quintic_returns_five_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(5) - &x - 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        5,
        "degree-5 polynomial should yield 5 RootOf objects, got {}",
        roots.len()
    );
}

#[test]
fn solve_sextic_returns_rootof() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁶ + x + 1 = 0  (no rational roots)
    let poly = &x.powi(6) + &x + 1;
    let roots = poly.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "sextic x⁶ + x + 1 should return RootOf objects"
    );
    assert_eq!(
        roots.len(),
        6,
        "degree-6 polynomial should yield 6 RootOf objects, got {}",
        roots.len()
    );
}

#[test]
fn solve_quintic_with_rational_root_mixed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x − 1)(x⁵ − x − 1) = x⁶ − x⁵ − x² + x − x + 1
    // Actually let's just build the product directly.
    let factor1 = &x - 1;
    let factor2 = &x.powi(5) - &x - 1;
    let poly = &factor1 * &factor2;
    let roots = poly.solve_or_empty(&x);
    // Should contain x = 1 (exact) plus RootOf entries
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    let has_one = strs.iter().any(|s| s == "1");
    let has_rootof = strs.iter().any(|s| s.contains("RootOf"));
    assert!(
        has_one,
        "should find rational root x = 1 among: {strs:?}"
    );
    assert!(
        has_rootof,
        "should also have RootOf entries for the quintic factor: {strs:?}"
    );
}

#[test]
fn solve_degree_7_returns_rootof() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁷ − 2x − 5 = 0 (classic irreducible)
    let poly = &x.powi(7) - &(&x * 2) - 5;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(
        roots.len(),
        7,
        "degree-7 polynomial should yield 7 RootOf objects, got {}",
        roots.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 2 — numerical evaluation of RootOf via Sturm + bisection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rootof_eval_f64_quintic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ − x − 1 has exactly one real root ≈ 1.1673
    let poly = &x.powi(5) - &x - 1;
    let roots = poly.solve_or_empty(&x);
    assert!(!roots.is_empty());

    let mut real_count = 0;
    for root in &roots {
        if let Ok(v) = root.eval_f64() {
            real_count += 1;
            // Substitute back: should be ≈ 0
            let residual = poly.subs(&x, root).eval();
            if let Ok(r) = residual.eval_f64() {
                assert!(
                    r.abs() < 1e-6,
                    "RootOf should satisfy the equation, residual = {r}"
                );
            }
            // Known approximate value
            assert!(
                (v - 1.1673).abs() < 0.01,
                "real root of x⁵−x−1 ≈ 1.1673, got {v}"
            );
        }
    }
    assert!(
        real_count >= 1,
        "at least one real root should be numerically evaluable"
    );
}

#[test]
fn rootof_eval_decimal_quintic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(5) - &x - 1;
    let roots = poly.solve_or_empty(&x);
    // Among all roots (real + complex), at least one should be the
    // real root ≈ 1.1673.  With Aberth, complex roots also evaluate
    // to decimal strings like "-0.764... - 0.352...*i".
    let mut found_real = false;
    for root in &roots {
        if let Ok(dec) = root.eval_decimal(15) {
            if dec.starts_with("1.167") {
                found_real = true;
            }
        }
    }
    assert!(found_real, "should find the real root ≈ 1.167 among the RootOf objects");
}

#[test]
fn rootof_eval_f64_degree7() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁷ − 2x − 5 has one real root ≈ 1.3267
    let poly = &x.powi(7) - &(&x * 2) - 5;
    let roots = poly.solve_or_empty(&x);

    let mut found_real = false;
    for root in &roots {
        if let Ok(v) = root.eval_f64() {
            found_real = true;
            // Verify it's actually a root
            let residual = poly.subs(&x, root).eval();
            if let Ok(r) = residual.eval_f64() {
                assert!(
                    r.abs() < 1e-6,
                    "residual should be ~0, got {r} for root value {v}"
                );
            }
        }
    }
    assert!(
        found_real,
        "degree-7 polynomial should have at least one evaluable real root"
    );
}

#[test]
fn rootof_complex_roots_are_unevaluable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ − x − 1 has 1 real root and 4 complex roots.
    // RootOf indices for the complex roots should fail eval_f64.
    let poly = &x.powi(5) - &x - 1;
    let roots = poly.solve_or_empty(&x);
    let mut eval_ok = 0;
    let mut eval_err = 0;
    for root in &roots {
        match root.eval_f64() {
            Ok(_) => eval_ok += 1,
            Err(_) => eval_err += 1,
        }
    }
    assert_eq!(eval_ok, 1, "exactly 1 real root should be evaluable");
    assert_eq!(eval_err, 4, "4 complex roots should fail eval_f64");
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 3 — solve() API returns Ok for quintics (not Err)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_api_returns_ok_for_quintic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(5) - &x - 1;
    let result = poly.solve(&x);
    assert!(
        result.is_ok(),
        "solve() should return Ok for polynomial expressions, got: {result:?}"
    );
    let roots = result.unwrap();
    assert_eq!(roots.len(), 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: lower-degree polynomials still work
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_quadratic_still_works() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let poly = &x.powi(2) - 1;
    let roots = poly.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x²−1 should have 2 roots");
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert_eq!(strs, vec!["-1", "1"]);
}

#[test]
fn solve_quartic_still_works() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x−1)(x−2)(x−3)(x−4) = x⁴ − 10x³ + 35x² − 50x + 24
    let poly = &(&(&(&x - 1) * &(&x - 2)) * &(&x - 3)) * &(&x - 4);
    let roots = poly.solve_or_empty(&x);
    assert!(
        roots.len() >= 4,
        "(x-1)(x-2)(x-3)(x-4) should have 4 roots, got {}",
        roots.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge case: polynomial with multiple real roots at degree 5
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rootof_eval_multiple_real_roots() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x⁵ − 5x³ + 4x = x(x²−1)(x²−4) = x(x−1)(x+1)(x−2)(x+2)
    // This factors completely over ℚ, so the solver should find
    // rational roots (not RootOf). Verifies no regression.
    let poly = &x.powi(5) - &(&x.powi(3) * 5) + &(&x * 4);
    let roots = poly.solve_or_empty(&x);
    assert!(
        roots.len() >= 5,
        "x⁵ − 5x³ + 4x should have 5 rational roots, got {}",
        roots.len()
    );
    // None should be RootOf since all roots are rational
    for root in &roots {
        let s = format!("{root}");
        assert!(
            !s.contains("RootOf"),
            "fully factorable polynomial should not produce RootOf: {s}"
        );
    }
}
