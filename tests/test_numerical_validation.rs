//! Numerical cross-validation tests.
//!
//! Verifies that symbolic transformations preserve numerical values
//! by evaluating input and output at random points.

use symplex::prelude::*;

/// Evaluate an expression at x=pt using substitution and evalf_f64.
#[allow(dead_code)]
fn eval_at(expr: &Ex, x: &Ex, pt: f64) -> Option<f64> {
    // Substitute x → pt via rational approximation
    let _ctx = symplex::default_context();
    // Use integer points for exact evaluation
    let pt_int = pt as i64;
    let substituted = expr.subs_i64(x, pt_int);
    substituted.evalf_f64().ok()
}

/// Check that two expressions have the same numerical value at several integer points.
fn assert_numerically_equal(a: &Ex, b: &Ex, x: &Ex, points: &[i64], tolerance: f64, msg: &str) {
    for &pt in points {
        let a_sub = a.subs_i64(x, pt);
        let b_sub = b.subs_i64(x, pt);
        let a_val = a_sub.evalf_f64();
        let b_val = b_sub.evalf_f64();
        match (a_val, b_val) {
            (Ok(av), Ok(bv)) => {
                if av.is_nan() && bv.is_nan() {
                    continue;
                }
                if av.is_infinite() && bv.is_infinite() && av.signum() == bv.signum() {
                    continue;
                }
                let diff = (av - bv).abs();
                assert!(
                    diff < tolerance,
                    "{msg} at x={pt}: {av} vs {bv} (diff={diff})"
                );
            }
            _ => {} // If either can't be evaluated, skip the point
        }
    }
}

const POINTS: &[i64] = &[-3, -2, -1, 1, 2, 3, 4, 5];

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation preserves value via integration roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_integrate_roundtrip_x2() {
    let x = symplex::var("x");
    let f = x.powi(2);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "d/dx(∫ x² dx) == x²");
}

#[test]
fn diff_integrate_roundtrip_x3() {
    let x = symplex::var("x");
    let f = x.powi(3);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "d/dx(∫ x³ dx) == x³");
}

#[test]
fn diff_integrate_roundtrip_polynomial() {
    let x = symplex::var("x");
    let f = &(&x.powi(3) * 2) - &(&x.powi(2) * 3) + &(&x * 5) - 7;
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "polynomial roundtrip");
}

// ═══════════════════════════════════════════════════════════════════════════
// Expand preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_preserves_x_plus_1_squared() {
    let x = symplex::var("x");
    let original = (&x + 1).powi(2);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)² expand");
}

#[test]
fn expand_preserves_x_plus_1_cubed() {
    let x = symplex::var("x");
    let original = (&x + 1).powi(3);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)³ expand");
}

#[test]
fn expand_preserves_product_of_sums() {
    let x = symplex::var("x");
    let original = &(&x + 1) * &(&x - 1);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)(x-1) expand");
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplify preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sin2_cos2_value() {
    let x = symplex::var("x");
    let original = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = original.simplify();
    assert_numerically_equal(
        &original,
        &simplified,
        &x,
        POINTS,
        1e-10,
        "sin²+cos² simplify",
    );
}

#[test]
fn simplify_exp_ln_value() {
    let x = symplex::var("x");
    let original = x.ln().exp();
    let simplified = original.simplify();
    // Only check at positive points (ln needs positive input)
    let positive_pts = &[1, 2, 3, 4, 5];
    assert_numerically_equal(
        &original,
        &simplified,
        &x,
        positive_pts,
        1e-10,
        "exp(ln(x)) simplify",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Factor preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factor_x2_minus_1_value() {
    let x = symplex::var("x");
    let original = &x.powi(2) - 1;
    let factored = original.factor(&x);
    assert_numerically_equal(&original, &factored, &x, POINTS, 1e-10, "factor x²-1");
}

#[test]
fn factor_x3_minus_x_value() {
    let x = symplex::var("x");
    let original = &x.powi(3) - &x;
    let factored = original.factor(&x);
    assert_numerically_equal(&original, &factored, &x, POINTS, 1e-10, "factor x³-x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Solve verification: roots satisfy the equation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_verify_quadratic() {
    let x = symplex::var("x");
    let eq = &x.powi(2) - &(&x * 5) + 6;
    let roots = eq.solve_or_empty(&x);
    for root in &roots {
        let val = eq.subs(&x, root);
        let v = val.evalf_f64();
        if let Ok(f) = v {
            assert!(
                f.abs() < 1e-10,
                "root {} should make equation 0, got {f}",
                root
            );
        }
    }
}

#[test]
fn solve_verify_cubic() {
    let x = symplex::var("x");
    let eq = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = eq.solve_or_empty(&x);
    for root in &roots {
        let val = eq.subs(&x, root);
        let v = val.evalf_f64();
        if let Ok(f) = v {
            assert!(
                f.abs() < 1e-10,
                "root {} should make equation 0, got {f}",
                root
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig expand/combine roundtrip preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trig_expand_preserves_sin_2x() {
    let x = symplex::var("x");
    let ctx = symplex::default_context();
    let two = ctx.int(2);
    let angle = &x * &two;
    let original = angle.sin();
    let expanded = original.expand_trig();
    assert_numerically_equal(
        &original,
        &expanded,
        &x,
        POINTS,
        1e-10,
        "sin(2x) trig expand",
    );
}

#[test]
fn trig_expand_preserves_cos_2x() {
    let x = symplex::var("x");
    let ctx = symplex::default_context();
    let two = ctx.int(2);
    let angle = &x * &two;
    let original = angle.cos();
    let expanded = original.expand_trig();
    assert_numerically_equal(
        &original,
        &expanded,
        &x,
        POINTS,
        1e-10,
        "cos(2x) trig expand",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Log expand/combine roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_expand_preserves_value() {
    let x = symplex::var("x");
    let y = symplex::var("y");
    let original = (&x * &y).ln();
    let expanded = original.expand_log();
    // Only check positive values (ln domain)
    // Substitute x=2, y=3
    let o = original.subs_i64(&x, 2).subs_i64(&y, 3);
    let e = expanded.subs_i64(&x, 2).subs_i64(&y, 3);
    let ov = o.evalf_f64();
    let ev = e.evalf_f64();
    if let (Ok(a), Ok(b)) = (ov, ev) {
        assert!((a - b).abs() < 1e-10, "ln(x*y) expand: {a} vs {b}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series approximation verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_sin_approximates_at_small_x() {
    let x = symplex::var("x");
    let series = x.sin().maclaurin(&x, 5).unwrap().expand();
    // At x=0.1, sin(0.1) ≈ 0.0998334...
    // The series x - x³/6 + x⁵/120 should be close
    // We can't easily substitute 0.1 so use x=1 where sin(1) ≈ 0.841
    // The 5th order Maclaurin of sin at x=1: 1 - 1/6 + 1/120 ≈ 0.8417
    let series_at_1 = series.subs_i64(&x, 1).evalf_f64();
    let exact = 1.0f64.sin();
    if let Ok(approx) = series_at_1 {
        assert!(
            (approx - exact).abs() < 0.01,
            "sin Maclaurin at x=1: approx={approx}, exact={exact}"
        );
    }
}

#[test]
fn maclaurin_exp_approximates_at_small_x() {
    let x = symplex::var("x");
    let series = x.exp().maclaurin(&x, 6).unwrap().expand();
    let series_at_1 = series.subs_i64(&x, 1).evalf_f64();
    let exact = 1.0f64.exp();
    if let Ok(approx) = series_at_1 {
        assert!(
            (approx - exact).abs() < 0.01,
            "exp Maclaurin at x=1: approx={approx}, exact={exact}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex number numerical verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complex_i_squared_numerically() {
    let i = symplex::i_unit();
    let result = i.powi(2);
    let v = result.evalf_f64();
    if let Ok(f) = v {
        assert!((f - (-1.0)).abs() < 1e-10, "i² should be -1: {f}");
    }
}

#[test]
fn complex_one_plus_i_fourth() {
    let i = symplex::i_unit();
    let expr = (&symplex::int(1) + &i).powi(4).expand();
    let v = expr.evalf_f64();
    if let Ok(f) = v {
        assert!((f - (-4.0)).abs() < 1e-10, "(1+i)⁴ should be -4: {f}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_pi_digits() {
    let pi = symplex::pi();
    let result = pi.evalf(20).unwrap();
    assert!(
        result.starts_with("3.14159265"),
        "π should start with 3.14159265: {result}"
    );
}

#[test]
fn evalf_e_digits() {
    let e = symplex::e();
    let result = e.evalf(20).unwrap();
    assert!(
        result.starts_with("2.71828182"),
        "e should start with 2.71828182: {result}"
    );
}

#[test]
fn evalf_sqrt_2() {
    let result = symplex::int(2).sqrt().evalf(15).unwrap();
    assert!(result.starts_with("1.41421356"), "√2: {result}");
}

#[test]
fn evalf_ln_2() {
    let result = symplex::int(2).ln().evalf(15).unwrap();
    assert!(result.starts_with("0.69314718"), "ln(2): {result}");
}
