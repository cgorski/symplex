//! Numerical cross-validation tests.
//!
//! Verifies that symbolic transformations preserve numerical values
//! by evaluating input and output at random points.

use symplex::prelude::*;

/// Check that two expressions have the same numerical value at several integer points.
fn assert_numerically_equal(a: &Ex, b: &Ex, x: &Ex, points: &[i64], tolerance: f64, msg: &str) {
    for &pt in points {
        let a_sub = a.subs_i64(x, pt);
        let b_sub = b.subs_i64(x, pt);
        let a_val = a_sub.eval_f64();
        let b_val = b_sub.eval_f64();
        // If either can't be evaluated, skip the point
        if let (Ok(av), Ok(bv)) = (a_val, b_val) {
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
    }
}

const POINTS: &[i64] = &[-3, -2, -1, 1, 2, 3, 4, 5];

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation preserves value via integration roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_integrate_roundtrip_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "d/dx(∫ x² dx) == x²");
}

#[test]
fn diff_integrate_roundtrip_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "d/dx(∫ x³ dx) == x³");
}

#[test]
fn diff_integrate_roundtrip_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &(&x.powi(3) * 2) - &(&x.powi(2) * 3) + &(&x * 5) - 7;
    let roundtrip = f.integrate(&x).diff(&x);
    assert_numerically_equal(&f, &roundtrip, &x, POINTS, 1e-10, "polynomial roundtrip");
}

// ═══════════════════════════════════════════════════════════════════════════
// Expand preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_preserves_x_plus_1_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = (&x + 1).powi(2);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)² expand");
}

#[test]
fn expand_preserves_x_plus_1_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = (&x + 1).powi(3);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)³ expand");
}

#[test]
fn expand_preserves_product_of_sums() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = &(&x + 1) * &(&x - 1);
    let expanded = original.expand();
    assert_numerically_equal(&original, &expanded, &x, POINTS, 1e-10, "(x+1)(x-1) expand");
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplify preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_sin2_cos2_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = &x.powi(2) - 1;
    let factored = original.factor(&x);
    assert_numerically_equal(&original, &factored, &x, POINTS, 1e-10, "factor x²-1");
}

#[test]
fn factor_x3_minus_x_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = &x.powi(3) - &x;
    let factored = original.factor(&x);
    assert_numerically_equal(&original, &factored, &x, POINTS, 1e-10, "factor x³-x");
}

// ═══════════════════════════════════════════════════════════════════════════
// Solve verification: roots satisfy the equation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_verify_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) - &(&x * 5) + 6;
    let roots = eq.solve_or_empty(&x);
    assert!(!roots.is_empty(), "quadratic should have roots");
    for root in &roots {
        let val = eq.subs(&x, root);
        let f = val.eval_f64().expect("root evaluation should succeed");
        assert!(
            f.abs() < 1e-10,
            "root {} should make equation 0, got {f}",
            root
        );
    }
}

#[test]
fn solve_verify_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6;
    let roots = eq.solve_or_empty(&x);
    assert!(!roots.is_empty(), "cubic should have roots");
    for root in &roots {
        let val = eq.subs(&x, root);
        let f = val.eval_f64().expect("root evaluation should succeed");
        assert!(
            f.abs() < 1e-10,
            "root {} should make equation 0, got {f}",
            root
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig expand/combine roundtrip preserves value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trig_expand_preserves_sin_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
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
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let original = (&x * &y).ln();
    let expanded = original.expand_log();
    // Only check positive values (ln domain)
    // Substitute x=2, y=3
    let o = original.subs_i64(&x, 2).subs_i64(&y, 3);
    let e = expanded.subs_i64(&x, 2).subs_i64(&y, 3);
    let ov = o.eval_f64();
    let ev = e.eval_f64();
    if let (Ok(a), Ok(b)) = (ov, ev) {
        assert!((a - b).abs() < 1e-10, "ln(x*y) expand: {a} vs {b}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Series approximation verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn maclaurin_sin_approximates_at_small_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 5).expand();
    // At x=0.1, sin(0.1) ≈ 0.0998334...
    // The series x - x³/6 + x⁵/120 should be close
    // We can't easily substitute 0.1 so use x=1 where sin(1) ≈ 0.841
    // The 5th order Maclaurin of sin at x=1: 1 - 1/6 + 1/120 ≈ 0.8417
    let approx = series
        .subs_i64(&x, 1)
        .eval_f64()
        .expect("Maclaurin sin evaluation should succeed");
    let exact = 1.0f64.sin();
    assert!(
        (approx - exact).abs() < 0.01,
        "sin Maclaurin at x=1: approx={approx}, exact={exact}"
    );
}

#[test]
fn maclaurin_exp_approximates_at_small_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().maclaurin(&x, 6).expand();
    let approx = series
        .subs_i64(&x, 1)
        .eval_f64()
        .expect("Maclaurin exp evaluation should succeed");
    let exact = 1.0f64.exp();
    assert!(
        (approx - exact).abs() < 0.01,
        "exp Maclaurin at x=1: approx={approx}, exact={exact}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex number numerical verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complex_i_squared_numerically() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2);
    let f = result.eval_f64().expect("i² evaluation should succeed");
    assert!((f - (-1.0)).abs() < 1e-10, "i² should be -1: {f}");
}

#[test]
fn complex_one_plus_i_fourth() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(4).expand();
    let f = expr.eval_f64().expect("(1+i)⁴ evaluation should succeed");
    assert!((f - (-4.0)).abs() < 1e-10, "(1+i)⁴ should be -4: {f}");
}

// ═══════════════════════════════════════════════════════════════════════════
// evalf consistency
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_pi_digits() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let result = pi.eval_decimal(20).unwrap();
    assert!(
        result.starts_with("3.14159265"),
        "π should start with 3.14159265: {result}"
    );
}

#[test]
fn evalf_e_digits() {
    let ctx = Context::new();
    let e = ctx.e();
    let result = e.eval_decimal(20).unwrap();
    assert!(
        result.starts_with("2.71828182"),
        "e should start with 2.71828182: {result}"
    );
}

#[test]
fn evalf_sqrt_2() {
    let ctx = Context::new();
    let result = ctx.int(2).sqrt().eval_decimal(15).unwrap();
    assert!(result.starts_with("1.41421356"), "√2: {result}");
}

#[test]
fn evalf_ln_2() {
    let ctx = Context::new();
    let result = ctx.int(2).ln().eval_decimal(15).unwrap();
    assert!(result.starts_with("0.69314718"), "ln(2): {result}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-rule numerical validation
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify simplify preserves value at a point.
fn check_simplify_value(
    expr: &symplex::prelude::Ex,
    var: &symplex::prelude::Ex,
    point_num: i64,
    point_den: i64,
) {
    let ctx = expr.context();
    let point = ctx.rational(point_num, point_den);
    let simplified = expr.simplify();
    let v1 = expr.subs(var, &point).eval_f64().unwrap();
    let v2 = simplified.subs(var, &point).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-8,
        "simplify changed value at {point_num}/{point_den}: {v1} vs {v2} for '{expr}' → '{simplified}'"
    );
}

#[test]
fn value_rule_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&(&x.sin().powi(2) + &x.cos().powi(2)), &x, 7, 10);
}

#[test]
fn value_rule_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.ln().exp(), &x, 3, 1);
}

#[test]
fn value_rule_ln_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.exp().ln(), &x, 1, 2);
}

#[test]
fn value_rule_abs_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.abs().abs(), &x, -3, 1);
}

#[test]
fn value_rule_sqrt_sq() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    check_simplify_value(&x.powi(2).pow(&half), &x, -5, 2);
}

#[test]
fn value_rule_cosh_sinh_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&(&x.cosh().powi(2) - &x.sinh().powi(2)), &x, 3, 2);
}

#[test]
fn value_rule_sin_div_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&(&x.sin() / &x.cos()), &x, 1, 3);
}

#[test]
fn value_rule_sinh_div_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&(&x.sinh() / &x.cosh()), &x, 1, 2);
}

#[test]
fn value_rule_exp_mul() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let point_x = ctx.int(1);
    let point_y = ctx.int(2);
    let expr = &x.exp() * &y.exp();
    let simplified = expr.simplify();
    let v1 = expr
        .subs(&x, &point_x)
        .subs(&y, &point_y)
        .eval_f64()
        .unwrap();
    let v2 = simplified
        .subs(&x, &point_x)
        .subs(&y, &point_y)
        .eval_f64()
        .unwrap();
    assert!(
        (v1 - v2).abs() < 1e-8,
        "exp(x)*exp(y) value mismatch: {v1} vs {v2}"
    );
}

#[test]
fn value_rule_sin_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let _ctx = Context::new();
    let _point = _ctx.rational(1, 2);
    check_simplify_value(&x.asin().sin(), &x, 1, 2);
}

#[test]
fn value_rule_cos_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.acos().cos(), &x, 1, 2);
}

#[test]
fn value_rule_tan_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.atan().tan(), &x, 3, 2);
}

#[test]
fn value_rule_acosh_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_simplify_value(&x.cosh().acosh(), &x, -2, 1);
}

#[test]
fn value_rule_pow_pow_integers() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).powi(3); // (x^2)^3 = x^6
    check_simplify_value(&expr, &x, 3, 2);
}
