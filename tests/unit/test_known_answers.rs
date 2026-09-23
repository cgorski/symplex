//! Known-answer test corpus — verified against standard textbook results.
//!
//! Each test verifies a specific symbolic computation against its
//! expected result. These tests serve as regression guards and as
//! documentation of what the CAS can compute correctly.

use symplex::prelude::*;

// Helper to check display output matches expected string
fn check(expr: &Ex, expected: &str) {
    let s = format!("{expr}");
    assert_eq!(s, expected, "expected '{expected}', got '{s}'");
}

// Helper for soft checks (contains substring)
fn check_contains(expr: &Ex, substrings: &[&str], msg: &str) {
    let s = format!("{expr}");
    for sub in substrings {
        assert!(
            s.contains(sub),
            "{msg}: expected to contain '{sub}', got '{s}'"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — power rule (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.diff(&x), "1");
}

#[test]
fn diff_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).diff(&x), "2*x");
}

#[test]
fn diff_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(3).diff(&x), "3*x^2");
}

#[test]
fn diff_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(4).diff(&x), "4*x^3");
}

#[test]
fn diff_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff(&x), "5*x^4");
}

#[test]
fn diff_const_7() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(7).diff(&x), "0");
}

#[test]
fn diff_const_0() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(0).diff(&x), "0");
}

#[test]
fn diff_const_neg3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(-3).diff(&x), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().diff(&x), "cos(x)");
}

#[test]
fn diff_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cos().diff(&x), "-sin(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — exp/ln (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.exp().diff(&x), "exp(x)");
}

#[test]
fn diff_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.ln().diff(&x), "1/x");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — chain rule (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_sin_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).sin().diff(&x), "2*x*cos(x^2)");
}

#[test]
fn diff_exp_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).exp().diff(&x), "2*x*exp(x^2)");
}

#[test]
fn diff_ln_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).ln().diff(&x), "2/x");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — product rule (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_x_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * &x.sin()).diff(&x), "x*cos(x) + sin(x)");
}

#[test]
fn diff_x_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * &x.exp()).diff(&x), "x*exp(x) + exp(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — inverse trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asin().diff(&x), "1/sqrt(-x^2 + 1)");
}

#[test]
fn diff_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acos().diff(&x), "-1/sqrt(-x^2 + 1)");
}

#[test]
fn diff_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atan().diff(&x), "1/(x^2 + 1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — hyperbolic (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sinh().diff(&x), "cosh(x)");
}

#[test]
fn diff_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cosh().diff(&x), "sinh(x)");
}

#[test]
fn diff_tanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tanh().diff(&x), "-tanh(x)^2 + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — inverse hyperbolic (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_asinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asinh().diff(&x), "1/sqrt(x^2 + 1)");
}

#[test]
fn diff_acosh() {
    // SymPy: diff(acosh(x), x) = 1/(sqrt(x - 1)*sqrt(x + 1)) — the derivative of
    // the principal acosh everywhere; 1/sqrt(x^2 - 1) has the wrong sign for
    // x < -1 (mpmath: diff(acosh, -2) = -0.57735026918962576).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acosh().diff(&x), "1/(sqrt(x - 1)*sqrt(x + 1))");
}

#[test]
fn diff_atanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atanh().diff(&x), "1/(-x^2 + 1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — higher order (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff2_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(3).diff_n(&x, 2), "6*x");
}

#[test]
fn diff3_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(4).diff_n(&x, 3), "24*x");
}

#[test]
fn diff4_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(4).diff_n(&x, 4), "24");
}

#[test]
fn diff5_x4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(4).diff_n(&x, 5), "0");
}

#[test]
fn diff0_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).diff_n(&x, 0), "x^2");
}

#[test]
fn diff2_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff_n(&x, 2), "20*x^3");
}

#[test]
fn diff3_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff_n(&x, 3), "60*x^2");
}

#[test]
fn diff4_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff_n(&x, 4), "120*x");
}

#[test]
fn diff5_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff_n(&x, 5), "120");
}

#[test]
fn diff6_x5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(5).diff_n(&x, 6), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — multi-variable (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_xy_wrt_x() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    check(&(&x * &y).diff(&x), "y");
}

#[test]
fn diff_xy_wrt_y() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    check(&(&x * &y).diff(&y), "x");
}

#[test]
fn diff_y_wrt_x_is_zero() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    check(&y.diff(&x), "0");
}

#[test]
fn diff_x2y_wrt_x() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let r = (&x.powi(2) * &y).diff(&x);
    let s = format!("{r}");
    assert!(
        s.contains("2") && s.contains("x") && s.contains("y"),
        "d/dx(x²y) should be 2xy: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// DIFFERENTIATION — linearity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_3x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.powi(2) * 3).diff(&x), "6*x");
}

#[test]
fn diff_sum_x_plus_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x + &x.powi(2)).diff(&x);
    let s = format!("{r}");
    assert!(
        s.contains("1") && s.contains("2") && s.contains("x"),
        "d/dx(x+x²) should be 1+2x: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — power rule (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(1).integrate(&x), "x");
}

#[test]
fn int_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.integrate(&x), "1/2*x^2");
}

#[test]
fn int_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).integrate(&x), "1/3*x^3");
}

#[test]
fn int_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(3).integrate(&x), "1/4*x^4");
}

#[test]
fn int_x_inv() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(-1).integrate(&x), "ln(abs(x))");
}

#[test]
fn int_const_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(5).integrate(&x), "5*x");
}

#[test]
fn int_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&ctx.int(0).integrate(&x), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().integrate(&x), "-cos(x)");
}

#[test]
fn int_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cos().integrate(&x), "sin(x)");
}

#[test]
fn int_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tan().integrate(&x), "-ln(abs(cos(x)))");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — exp/ln (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.exp().integrate(&x), "exp(x)");
}

#[test]
fn int_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.ln().integrate(&x), "-x + x*ln(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — hyperbolic (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sinh().integrate(&x), "cosh(x)");
}

#[test]
fn int_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cosh().integrate(&x), "sinh(x)");
}

#[test]
fn int_tanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tanh().integrate(&x), "ln(cosh(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — inverse trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asin().integrate(&x), "x*asin(x) + sqrt(-x^2 + 1)");
}

#[test]
fn int_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acos().integrate(&x), "x*acos(x) - sqrt(-x^2 + 1)");
}

#[test]
fn int_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atan().integrate(&x), "x*atan(x) - 1/2*ln(x^2 + 1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — standard forms (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_one_over_x2_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.powi(2) + 1).powi(-1).integrate(&x), "atan(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — u-substitution (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sin_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * 2).sin().integrate(&x), "-1/2*cos(2*x)");
}

#[test]
fn int_exp_3x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * 3).exp().integrate(&x), "1/3*exp(3*x)");
}

#[test]
fn int_cos_5x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * 5).cos().integrate(&x), "1/5*sin(5*x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — by parts (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_x_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * &x.sin()).integrate(&x);
    let s = format!("{r}");
    // -(-x*cos(x)) + sin(x)
    assert!(
        s.contains("sin(x)") && s.contains("cos(x)"),
        "∫ x·sin(x) dx should involve sin(x) and cos(x): {s}"
    );
}

#[test]
fn int_x_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * &x.exp()).integrate(&x), "x*exp(x) - exp(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — trig powers (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sin2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().powi(2).integrate(&x), "1/2*x - 1/2*sin(x)*cos(x)");
}

#[test]
fn int_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cos().powi(2).integrate(&x), "1/2*x + 1/2*sin(x)*cos(x)");
}

#[test]
fn int_sin3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(
        &x.sin().powi(3).integrate(&x),
        "-1/3*sin(x)^2*cos(x) - 2/3*cos(x)",
    );
}

#[test]
fn int_cos3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(
        &x.cos().powi(3).integrate(&x),
        "1/3*cos(x)^2*sin(x) + 2/3*sin(x)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — linear substitution power (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_linear_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * 2 + 1).powi(3).integrate(&x), "1/8*(2*x + 1)^4");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — linearity (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x + &x.powi(2)).integrate(&x), "1/3*x^3 + 1/2*x^2");
}

#[test]
fn int_const_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.powi(2) * 5).integrate(&x), "5/3*x^3");
}

#[test]
fn int_const_times_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x * 3).integrate(&x), "3/2*x^2");
}

#[test]
fn int_other_symbol_treated_as_const() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    check(&y.integrate(&x), "x*y");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — roundtrip verification (diff ∘ integrate = id)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_diff_roundtrip_x3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let roundtrip = f.integrate(&x).diff(&x);
    let v1 = format!("{}", f.subs_i64(&x, 2));
    let v2 = format!("{}", roundtrip.subs_i64(&x, 2));
    assert_eq!(v1, v2, "∫→d/dx roundtrip at x=2");
}

#[test]
fn int_diff_roundtrip_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().integrate(&x).diff(&x), "sin(x)");
}

#[test]
fn int_diff_roundtrip_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.exp().integrate(&x).diff(&x), "exp(x)");
}

#[test]
fn int_diff_roundtrip_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).integrate(&x).diff(&x), "x^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION — hyperbolic u-substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn int_sinh_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * 2).sinh().integrate(&x);
    check_contains(&r, &["cosh"], "∫ sinh(2x) dx");
}

#[test]
fn int_cosh_3x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * 3).cosh().integrate(&x);
    check_contains(&r, &["sinh"], "∫ cosh(3x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — linear (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x_eq_0() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = x.solve_or_empty(&x);
    assert_eq!(r.len(), 1);
    check(&r[0], "0");
}

#[test]
fn solve_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x - 1).solve_or_empty(&x);
    assert_eq!(r.len(), 1);
    check(&r[0], "1");
}

#[test]
fn solve_2x_minus_6() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * 2 - 6).solve_or_empty(&x);
    assert_eq!(r.len(), 1);
    check(&r[0], "3");
}

#[test]
fn solve_x_plus_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x + 3).solve_or_empty(&x);
    assert_eq!(r.len(), 1);
    check(&r[0], "-3");
}

#[test]
fn solve_3x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * 3 - 1).solve_or_empty(&x);
    assert_eq!(r.len(), 1);
    check(&r[0], "1/3");
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — quadratic (real roots)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x2_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) - 1).solve_or_empty(&x);
    assert_eq!(r.len(), 2);
    let strs: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(",");
    assert!(
        joined.contains("1") && joined.contains("-1"),
        "roots: {joined}"
    );
}

#[test]
fn solve_x2_minus_5x_plus_6() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) - &(&x * 5) + 6).solve_or_empty(&x);
    assert_eq!(r.len(), 2);
    let strs: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(",");
    assert!(
        joined.contains("2") && joined.contains("3"),
        "roots: {joined}"
    );
}

#[test]
fn solve_x2_minus_4x_plus_4_double_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) - &(&x * 4) + 4).solve_or_empty(&x);
    assert!(!r.is_empty(), "should find at least 1 root");
    for root in &r {
        check(root, "2");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — quadratic (complex roots)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x2_plus_1_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) + 1).solve_or_empty(&x);
    assert_eq!(r.len(), 2, "x²+1=0 should have 2 complex roots");
    let s: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
    assert!(s.join(",").contains("I"), "roots should contain I");
}

#[test]
fn solve_x2_plus_4_complex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.powi(2) + 4).solve_or_empty(&x);
    if r.len() == 2 {
        let s: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
        assert!(s.join(",").contains("I"), "roots should contain I");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — cubic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_cubic_factored() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
    let r = (&x.powi(3) - &(&x.powi(2) * 6) + &(&x * 11) - 6).solve_or_empty(&x);
    assert_eq!(r.len(), 3, "should have 3 roots");
    let strs: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(",");
    assert!(joined.contains("1"), "should have root 1: {joined}");
    assert!(joined.contains("2"), "should have root 2: {joined}");
    assert!(joined.contains("3"), "should have root 3: {joined}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — already-factored polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_factored() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * &(&x - 1) * &(&x + 2)).solve_or_empty(&x);
    assert_eq!(
        r.len(),
        3,
        "x(x-1)(x+2) should have 3 roots, got {}",
        r.len()
    );
    let strs: Vec<String> = r.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(",");
    assert!(joined.contains("0"), "should have root 0: {joined}");
    assert!(joined.contains("1"), "should have root 1: {joined}");
    assert!(joined.contains("-2"), "should have root -2: {joined}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — transcendental (solver returns empty)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_exp_minus_1_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.exp() - 1).solve_or_empty(&x);
    // exp(x)-1=0 → x=ln(1)=0, the internal solver handles this via inversion peeling.
    assert!(
        !r.is_empty(),
        "exp(x)-1 should be solvable now via inversion peeling"
    );
    let val = r[0].eval_f64().expect("root should evaluate");
    assert!(val.abs() < 1e-9, "root should be 0, got {val}");
}

#[test]
fn solve_sqrt_x_minus_2_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x.sqrt() - 2).solve_or_empty(&x);
    // sqrt(x)-2=0 → x=4, the internal solver handles this via inversion peeling.
    assert!(
        !r.is_empty(),
        "sqrt(x)-2 should be solvable now via inversion peeling"
    );
    let val = r[0].eval_f64().expect("root should evaluate");
    assert!((val - 4.0).abs() < 1e-9, "root should be 4, got {val}");
}

#[test]
fn solve_sin_non_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)=0 is now handled by the internal solver via inversion peeling.
    let result = x.sin().solve(&x);
    assert!(
        result.is_ok(),
        "sin(x) should be solvable via inversion peeling, got: {:?}",
        result.err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — constant equations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_constant_nonzero_no_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx.int(5).solve_or_empty(&x);
    assert!(r.is_empty(), "5=0 has no solutions");
}

#[test]
fn solve_constant_zero_no_solutions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = ctx.int(0).solve_or_empty(&x);
    assert!(r.is_empty(), "0=0 has infinite solutions, returns empty");
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLVING — substitution verification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_verify_roots_sub_back() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) - &(&x * 5) + 6;
    let roots = eq.solve_or_empty(&x);
    assert_eq!(roots.len(), 2);
    for root in &roots {
        let val = eq.subs(&x, root);
        assert!(
            val.is_zero_structural(),
            "substituting x={root} should give 0, got {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — trig at 0 (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_0() {
    let ctx = Context::new();
    check(&ctx.int(0).sin().eval(), "0");
}

#[test]
fn eval_cos_0() {
    let ctx = Context::new();
    check(&ctx.int(0).cos().eval(), "1");
}

#[test]
fn eval_tan_0() {
    let ctx = Context::new();
    check(&ctx.int(0).tan().eval(), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — trig at π (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_pi() {
    let ctx = Context::new();
    check(&ctx.pi().sin().eval(), "0");
}

#[test]
fn eval_cos_pi() {
    let ctx = Context::new();
    check(&ctx.pi().cos().eval(), "-1");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — trig special angles (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_pi_over_6() {
    let ctx = Context::new();
    check(&(&ctx.pi() / 6).sin().eval(), "1/2");
}

#[test]
fn eval_cos_pi_over_3() {
    let ctx = Context::new();
    check(&(&ctx.pi() / 3).cos().eval(), "1/2");
}

#[test]
fn eval_tan_pi_over_4() {
    let ctx = Context::new();
    check(&(&ctx.pi() / 4).tan().eval(), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — exp/ln (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_exp_0() {
    let ctx = Context::new();
    check(&ctx.int(0).exp().eval(), "1");
}

#[test]
fn eval_ln_1() {
    let ctx = Context::new();
    check(&ctx.int(1).ln().eval(), "0");
}

#[test]
fn eval_ln_e() {
    let ctx = Context::new();
    check(&ctx.e().ln().eval(), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — complex powers of i (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_i_squared() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(2), "-1");
}

#[test]
fn eval_i_cubed() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(3), "-I");
}

#[test]
fn eval_i_fourth() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(4), "1");
}

#[test]
fn eval_i_to_100() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(100), "1");
}

#[test]
fn eval_i_to_neg_1() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(-1), "-I");
}

#[test]
fn eval_i_to_neg_2() {
    let ctx = Context::new();
    check(&ctx.i_unit().powi(-2), "-1");
}

#[test]
fn eval_sqrt_neg_1() {
    let ctx = Context::new();
    check(&ctx.int(-1).sqrt(), "I");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — Euler's formula (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_exp_i_pi() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    check(&(&i * &ctx.pi()).exp().eval(), "-1");
}

#[test]
fn eval_exp_i_pi_over_2() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let angle = &ctx.rational(1, 2) * &ctx.pi();
    check(&(&i * &angle).exp().eval(), "I");
}

#[test]
fn eval_euler_identity_zero() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let euler = &(&i * &ctx.pi()).exp().eval() + 1;
    check(&euler, "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — sqrt simplification (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sqrt_4() {
    let ctx = Context::new();
    check(&ctx.int(4).sqrt().eval(), "2");
}

#[test]
fn eval_sqrt_9() {
    let ctx = Context::new();
    check(&ctx.int(9).sqrt().eval(), "3");
}

#[test]
fn eval_sqrt_16() {
    let ctx = Context::new();
    check(&ctx.int(16).sqrt().eval(), "4");
}

#[test]
fn eval_sqrt_25() {
    let ctx = Context::new();
    check(&ctx.int(25).sqrt().eval(), "5");
}

#[test]
fn eval_sqrt_8() {
    let ctx = Context::new();
    check(&ctx.int(8).sqrt().eval(), "2*sqrt(2)");
}

#[test]
fn eval_sqrt_nine_fourths() {
    let ctx = Context::new();
    check(&ctx.rational(9, 4).sqrt().eval(), "3/2");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — inverse trig special values (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_asin_0() {
    let ctx = Context::new();
    check(&ctx.int(0).asin().eval(), "0");
}

#[test]
fn eval_asin_1() {
    let ctx = Context::new();
    let r = ctx.int(1).asin().eval();
    check_contains(&r, &["pi"], "asin(1) should be π/2");
}

#[test]
fn eval_acos_1() {
    let ctx = Context::new();
    check(&ctx.int(1).acos().eval(), "0");
}

#[test]
fn eval_atan_0() {
    let ctx = Context::new();
    check(&ctx.int(0).atan().eval(), "0");
}

#[test]
fn eval_asin_half() {
    let ctx = Context::new();
    check(&ctx.rational(1, 2).asin().eval(), "1/6*pi");
}

#[test]
fn eval_acos_half() {
    let ctx = Context::new();
    check(&ctx.rational(1, 2).acos().eval(), "1/3*pi");
}

#[test]
fn eval_asin_neg_half() {
    let ctx = Context::new();
    let r = ctx.rational(-1, 2).asin().eval();
    check_contains(&r, &["pi"], "asin(-1/2) should involve pi");
}

#[test]
fn eval_acos_neg_half() {
    let ctx = Context::new();
    let r = ctx.rational(-1, 2).acos().eval();
    check_contains(&r, &["pi"], "acos(-1/2) should involve pi");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — hyperbolic at 0 (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sinh_0() {
    let ctx = Context::new();
    check(&ctx.int(0).sinh().eval(), "0");
}

#[test]
fn eval_cosh_0() {
    let ctx = Context::new();
    check(&ctx.int(0).cosh().eval(), "1");
}

#[test]
fn eval_tanh_0() {
    let ctx = Context::new();
    check(&ctx.int(0).tanh().eval(), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — inverse hyperbolic at 0/1 (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_asinh_0() {
    let ctx = Context::new();
    check(&ctx.int(0).asinh().eval(), "0");
}

#[test]
fn eval_acosh_1() {
    let ctx = Context::new();
    check(&ctx.int(1).acosh().eval(), "0");
}

#[test]
fn eval_atanh_0() {
    let ctx = Context::new();
    check(&ctx.int(0).atanh().eval(), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — ln of negative (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_ln_neg_1() {
    let ctx = Context::new();
    check(&ctx.int(-1).ln().eval(), "pi*I");
}

#[test]
fn eval_ln_neg_2() {
    let ctx = Context::new();
    let r = ctx.int(-2).ln().eval();
    let s = format!("{r}");
    assert!(
        s.contains("ln") && s.contains("I"),
        "ln(-2) should be ln(2)+iπ, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — abs (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_abs_3() {
    let ctx = Context::new();
    check(&ctx.int(3).abs().eval(), "3");
}

#[test]
fn eval_abs_neg_3() {
    let ctx = Context::new();
    check(&ctx.int(-3).abs().eval(), "3");
}

#[test]
fn eval_abs_0() {
    let ctx = Context::new();
    check(&ctx.int(0).abs().eval(), "0");
}

#[test]
fn eval_abs_i() {
    let ctx = Context::new();
    check(&ctx.i_unit().abs().eval(), "1");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — odd/even trig parity (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_sin_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(-&x).sin().eval(), "-sin(x)");
}

#[test]
fn eval_cos_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(-&x).cos().eval(), "cos(x)");
}

#[test]
fn eval_tan_neg_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(-&x).tan().eval(), "-tan(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// EVALUATION — complex algebra (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_one_plus_i_squared() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    check(&(&ctx.int(1) + &i).powi(2).expand(), "2*I");
}

#[test]
fn eval_complex_addition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z1 = &ctx.int(2) + &(&ctx.int(3) * &i);
    let z2 = &ctx.int(4) + &(&ctx.int(5) * &i);
    check(&(&z1 + &z2), "8*I + 6");
}

#[test]
fn eval_i_times_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    check(&(&i * &i), "-1");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — Pythagorean identity (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_sin2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.sin().powi(2) + &x.cos().powi(2)).simplify(), "1");
}

#[test]
fn simp_sin2_cos2_plus_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.sin().powi(2) + &x.cos().powi(2) + 3).simplify(), "4");
}

#[test]
fn simp_sin2_cos2_plus_y() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    check(
        &(&y + &x.sin().powi(2) + &x.cos().powi(2)).simplify(),
        "y + 1",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — hyperbolic Pythagorean (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_cosh2_minus_sinh2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.cosh().powi(2) - &x.sinh().powi(2)).simplify(), "1");
}

#[test]
fn simp_cosh2_minus_sinh2_plus_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.cosh().powi(2) - &x.sinh().powi(2) + 5).simplify(), "6");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — inverse pairs (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.ln().exp().simplify(), "x");
}

#[test]
fn simp_ln_exp() {
    let ctx = Context::new();
    use symplex::prelude::Assumption;
    let x = ctx.symbol("x").assume(Assumption::Real);
    check(&x.exp().ln().simplify(), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — forward inverse trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_sin_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asin().sin().simplify(), "x");
}

#[test]
fn simp_cos_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acos().cos().simplify(), "x");
}

#[test]
fn simp_tan_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atan().tan().simplify(), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — reverse inverse trig (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_asin_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().asin().simplify(), "asin(sin(x))");
}

#[test]
fn simp_acos_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cos().acos().simplify(), "acos(cos(x))");
}

#[test]
fn simp_atan_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tan().atan().simplify(), "atan(tan(x))");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — forward inverse hyperbolic (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_sinh_asinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asinh().sinh().simplify(), "x");
}

#[test]
fn simp_cosh_acosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acosh().cosh().simplify(), "x");
}

#[test]
fn simp_tanh_atanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atanh().tanh().simplify(), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — trig ratio (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_sin_div_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.sin() / &x.cos()).simplify(), "tan(x)");
}

#[test]
fn simp_sinh_div_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.sinh() / &x.cosh()).simplify(), "tanh(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — exp combine (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_exp_a_times_exp_b() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    check(&(&x.exp() * &y.exp()).simplify(), "exp(x + y)");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — abs (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_abs_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.abs().abs().simplify(), "abs(x)");
}

#[test]
fn simp_abs_positive() {
    let ctx = Context::new();
    check(&ctx.int(5).abs().simplify(), "5");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — power of power (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_pow_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).powi(3).simplify(), "x^6");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — sqrt of square (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simp_sqrt_of_square() {
    let ctx = Context::new();
    // 0.23: the identity needs a real argument (a symbol without assumptions may be complex).
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    check(&x.powi(2).sqrt().simplify(), "abs(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMPLIFICATION — full_simplify (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simp_expand_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^2 - x^2 - 2*x = 1
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &x * 2;
    check(&expr.simplify(), "1");
}

#[test]
fn full_simp_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.sin().powi(2) + &x.cos().powi(2)).simplify(), "1");
}

#[test]
fn full_simp_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.ln().exp().simplify(), "x");
}

#[test]
fn full_simp_sin_0_plus_cos_0() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    check(&(&zero.sin() + &zero.cos()).simplify(), "1");
}

#[test]
fn full_simp_already_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x + 1).simplify(), "x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTION AND DISPLAY — constants (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_pi() {
    let ctx = Context::new();
    check(&ctx.pi(), "pi");
}

#[test]
fn display_e() {
    let ctx = Context::new();
    check(&ctx.e(), "E");
}

#[test]
fn display_i() {
    let ctx = Context::new();
    check(&ctx.i_unit(), "I");
}

#[test]
fn display_int_42() {
    let ctx = Context::new();
    check(&ctx.int(42), "42");
}

#[test]
fn display_rational_half() {
    let ctx = Context::new();
    check(&ctx.rational(1, 2), "1/2");
}

#[test]
fn display_rational_third() {
    let ctx = Context::new();
    check(&ctx.rational(1, 3), "1/3");
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTION AND DISPLAY — functions (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin(), "sin(x)");
}

#[test]
fn display_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cos(), "cos(x)");
}

#[test]
fn display_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tan(), "tan(x)");
}

#[test]
fn display_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.exp(), "exp(x)");
}

#[test]
fn display_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.ln(), "ln(x)");
}

#[test]
fn display_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asin(), "asin(x)");
}

#[test]
fn display_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acos(), "acos(x)");
}

#[test]
fn display_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atan(), "atan(x)");
}

#[test]
fn display_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sinh(), "sinh(x)");
}

#[test]
fn display_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.cosh(), "cosh(x)");
}

#[test]
fn display_tanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.tanh(), "tanh(x)");
}

#[test]
fn display_asinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.asinh(), "asinh(x)");
}

#[test]
fn display_acosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.acosh(), "acosh(x)");
}

#[test]
fn display_atanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.atanh(), "atanh(x)");
}

#[test]
fn display_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.abs(), "abs(x)");
}

#[test]
fn display_sqrt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sqrt(), "sqrt(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTION AND DISPLAY — powers (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2), "x^2");
}

#[test]
fn display_x_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(3), "x^3");
}

#[test]
fn display_x_inv() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(-1), "1/x");
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRUCTION AND DISPLAY — canonical ordering (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_x_plus_1_canonical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x + 1), "x + 1");
}

#[test]
fn display_1_plus_x_canonical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(1 + &x), "x + 1");
}

#[test]
fn display_x2_plus_x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x.powi(2) + &x + 1), "x^2 + x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// ASSUMPTIONS — basic queries (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assume_positive_int() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_positive(), Some(true));
}

#[test]
fn assume_negative_int() {
    let ctx = Context::new();
    assert_eq!(ctx.int(-3).is_negative(), Some(true));
}

#[test]
fn assume_zero_is_not_positive() {
    let ctx = Context::new();
    assert_eq!(ctx.int(0).is_positive(), Some(false));
}

#[test]
fn assume_integer_on_integer() {
    let ctx = Context::new();
    assert_eq!(ctx.int(7).is_integer(), Some(true));
}

#[test]
fn assume_real_on_integer() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_real(), Some(true));
}

#[test]
fn assume_real_on_pi() {
    let ctx = Context::new();
    assert_eq!(ctx.pi().is_real(), Some(true));
}

#[test]
fn assume_not_real_on_i() {
    let ctx = Context::new();
    assert_eq!(ctx.i_unit().is_real(), Some(false));
}

#[test]
fn assume_imaginary_on_i() {
    let ctx = Context::new();
    assert_eq!(ctx.i_unit().is_imaginary(), Some(true));
}

#[test]
fn assume_finite_on_int() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_finite(), Some(true));
}

#[test]
fn assume_not_finite_on_inf() {
    let ctx = Context::new();
    assert_eq!(ctx.infinity().is_finite(), Some(false));
}

#[test]
fn assume_nonzero_on_5() {
    let ctx = Context::new();
    assert_eq!(ctx.int(5).is_nonzero(), Some(true));
}

#[test]
fn assume_not_nonzero_on_0() {
    let ctx = Context::new();
    assert_eq!(ctx.int(0).is_nonzero(), Some(false));
}

#[test]
fn assume_symbol_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.is_positive(), None);
    assert_eq!(x.is_negative(), None);
}

#[test]
fn assume_rational_is_not_integer() {
    let ctx = Context::new();
    assert_eq!(ctx.rational(1, 2).is_integer(), Some(false));
}

#[test]
fn assume_symbol_with_assumption() {
    let ctx = Context::new();
    let t = ctx.symbol("t").assume(Assumption::Positive);
    assert_eq!(t.is_positive(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// NUMERICAL EVALUATION — evalf_f64 (approximate)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn evalf_pi() {
    let ctx = Context::new();
    let val = ctx.pi().eval_f64().unwrap();
    assert!((val - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn evalf_e() {
    let ctx = Context::new();
    let val = ctx.e().eval_f64().unwrap();
    assert!((val - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn evalf_integer() {
    let ctx = Context::new();
    let val = ctx.int(7).eval_f64().unwrap();
    assert!((val - 7.0).abs() < 1e-10);
}

#[test]
fn evalf_rational() {
    let ctx = Context::new();
    let val = ctx.rational(1, 3).eval_f64().unwrap();
    assert!((val - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn evalf_sinh_zero() {
    let ctx = Context::new();
    let val = ctx.int(0).sinh().eval_f64().unwrap();
    assert!((val - 0.0).abs() < 1e-10);
}

#[test]
fn evalf_cosh_zero() {
    let ctx = Context::new();
    let val = ctx.int(0).cosh().eval_f64().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn evalf_atan_one() {
    let ctx = Context::new();
    let val = ctx.int(1).atan().eval_f64().unwrap();
    assert!((val - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn evalf_free_symbol_errors() {
    let ctx = Context::new();
    assert!(ctx.symbol("x").eval_f64().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBSTITUTION — subs_i64 (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_x2_at_3() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).subs_i64(&x, 3), "9");
}

#[test]
fn subs_x2_at_neg2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(2).subs_i64(&x, -2), "4");
}

#[test]
fn subs_x3_at_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.powi(3).subs_i64(&x, 2), "8");
}

#[test]
fn subs_x_plus_1_at_5() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&(&x + 1).subs_i64(&x, 5), "6");
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPAND — algebraic expansion (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_x_plus_1_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x + 1).powi(2).expand();
    let s = format!("{r}");
    assert!(
        s.contains("x^2") && s.contains("2*x") && s.contains("1"),
        "(x+1)^2 expanded: {s}"
    );
}

#[test]
fn expand_x_times_x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let r = (&x * &(&x + 1)).expand();
    let s = format!("{r}");
    assert!(s.contains("x^2") && s.contains("x"), "x(x+1) expanded: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// TRIG EXPANSION (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_trig_sin_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let r = (&x + &y).sin().expand_trig();
    let s = format!("{r}");
    assert!(
        s.contains("sin(x)") && s.contains("cos(y)"),
        "sin(x+y) should expand: {s}"
    );
}

#[test]
fn expand_trig_cos_sum() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let r = (&x + &y).cos().expand_trig();
    let s = format!("{r}");
    assert!(
        s.contains("cos(x)") && s.contains("sin(x)"),
        "cos(x+y) should expand: {s}"
    );
}

#[test]
fn expand_trig_bare_sin_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&x.sin().expand_trig(), "sin(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// CANCEL — common factors (exact)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cancel_x2_minus_1_over_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(&((x.powi(2) - 1) / (&x - 1)).cancel(&x), "x + 1");
}

#[test]
fn cancel_no_common_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1) / (&x + 2);
    let orig = format!("{expr}");
    let cancelled = format!("{}", expr.cancel(&x));
    assert_eq!(orig, cancelled, "(x+1)/(x+2) should be unchanged");
}

// ═══════════════════════════════════════════════════════════════════════════
// FREE SYMBOLS — structural queries
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_symbols_single() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let syms = x.free_symbols();
    assert_eq!(syms.len(), 1);
    check(&syms[0], "x");
}

#[test]
fn free_symbols_two() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let syms = (&x + &y).free_symbols();
    assert_eq!(syms.len(), 2);
}

#[test]
fn free_symbols_constant_empty() {
    let ctx = Context::new();
    assert!(ctx.pi().free_symbols().is_empty());
}

#[test]
fn free_symbols_no_duplicates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let syms = (&x + &x).free_symbols();
    assert_eq!(syms.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTAINS — structural check
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn contains_self() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.contains(&x));
}

#[test]
fn contains_child() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert!((&x + &y).contains(&x));
    assert!((&x + &y).contains(&y));
}

#[test]
fn contains_deep() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.powi(2).sin().contains(&x));
}

#[test]
fn contains_not() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    assert!(!(&x + &y).contains(&z));
}

// ═══════════════════════════════════════════════════════════════════════════
// EQUALS — algebraic equality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equals_identical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.equals(&x), Some(true));
}

#[test]
fn equals_canonical_same() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).equals(&(1 + &x)), Some(true));
}

#[test]
fn equals_expand_needed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = (&x + 1).powi(2);
    let b = &x.powi(2) + &x * 2 + 1;
    assert_eq!(a.equals(&b), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// NSOLVE — numerical root finding (approximate)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsolve_x_minus_cos_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let root = (&x - &x.cos()).solve_numeric(&x, 1.0, 50, 1e-12).unwrap();
    assert!((root - 0.7390851332).abs() < 1e-6, "got: {root}");
}

#[test]
fn nsolve_x2_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let root = (&x.powi(2) - 2).solve_numeric(&x, 1.5, 50, 1e-12).unwrap();
    assert!(
        (root - std::f64::consts::SQRT_2).abs() < 1e-8,
        "got: {root}"
    );
}

#[test]
fn nsolve_exp_minus_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let root = (&x.exp() - 2).solve_numeric(&x, 1.0, 50, 1e-12).unwrap();
    assert!((root - 2.0_f64.ln()).abs() < 1e-8, "got: {root}");
}
