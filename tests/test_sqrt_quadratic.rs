//! Tests for ∫ poly(x)/√(ax²+bx+c) dx and sec·tan / csc·cot integration forms.

use symplex::prelude::*;

/// Fundamental Theorem of Calculus check: differentiate the antiderivative
/// and compare numerically against the original integrand at a test point.
fn assert_ftc(integrand: &Ex, var: &Ex, label: &str) {
    let anti = integrand.integrate(var);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral"),
        "{label}: got unevaluated integral: {s}"
    );
    let deriv = anti.diff(var);
    let test_point = symplex::default_context().rational(7, 10);
    if let (Ok(o), Ok(d)) = (
        integrand.subs(var, &test_point).eval_f64(),
        deriv.subs(var, &test_point).eval_f64(),
    )
        && o.is_finite() && d.is_finite()
    {
        assert!(
            (o - d).abs() < 1e-6 * o.abs().max(1.0),
            "{label}: FTC failed — integrand={o}, deriv={d}, diff={}",
            (o - d).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 1: sec·tan and csc·cot
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sec_tan() {
    // ∫ sin(x)/cos²(x) dx = sec(x)·tan(x) → 1/cos(x)
    let x = symplex::default_context().symbol("x");
    let integrand = &x.sin() / &x.cos().powi(2);
    assert_ftc(&integrand, &x, "∫ sec(x)tan(x) dx");
}

#[test]
fn integrate_csc_cot() {
    // ∫ cos(x)/sin²(x) dx = csc(x)·cot(x) → -1/sin(x)
    let x = symplex::default_context().symbol("x");
    let integrand = &x.cos() / &x.sin().powi(2);
    assert_ftc(&integrand, &x, "∫ csc(x)cot(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 2: ∫ 1/√(ax²+bx+c) dx — base cases (I₀)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_sqrt_1_plus_x2() {
    // ∫ 1/√(1+x²) dx = asinh(x)
    let x = symplex::default_context().symbol("x");
    let integrand = &symplex::default_context().int(1) / &(&x.powi(2) + 1).sqrt();
    assert_ftc(&integrand, &x, "∫ 1/√(1+x²) dx");
}

#[test]
fn integrate_1_over_sqrt_4x2_plus_1() {
    // ∫ 1/√(4x²+1) dx — coefficient a=4, b=0, c=1
    // Should use completing-the-square path with a>0, d>0
    let x = symplex::default_context().symbol("x");
    let integrand = &symplex::default_context().int(1) / &(&(&x.powi(2) * &symplex::default_context().int(4)) + 1).sqrt();
    assert_ftc(&integrand, &x, "∫ 1/√(4x²+1) dx");
}

#[test]
fn integrate_1_over_sqrt_quadratic_with_linear_term() {
    // ∫ 1/√(x²+2x+5) dx — completing the square yields (x+1)² + 4
    let x = symplex::default_context().symbol("x");
    let quad = &(&x.powi(2) + &(&symplex::default_context().int(2) * &x)) + 5;
    let integrand = &symplex::default_context().int(1) / &quad.sqrt();
    assert_ftc(&integrand, &x, "∫ 1/√(x²+2x+5) dx");
}

#[test]
fn integrate_1_over_sqrt_quadratic_neg_a() {
    // ∫ 1/√(3-2x²) dx — a=-2, b=0, c=3 → asin form
    // Using test point x=0.7 gives 3-2*(0.49) = 2.02 > 0, good.
    let x = symplex::default_context().symbol("x");
    let quad = &symplex::default_context().int(3) - &(&x.powi(2) * &symplex::default_context().int(2));
    let integrand = &symplex::default_context().int(1) / &quad.sqrt();
    assert_ftc(&integrand, &x, "∫ 1/√(3-2x²) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 3: ∫ x/√(ax²+bx+c) dx (I₁)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_over_sqrt_1_plus_x2() {
    // ∫ x/√(1+x²) dx = √(1+x²)   (b=0, so just √R/a)
    let x = symplex::default_context().symbol("x");
    let integrand = &x / &(&x.powi(2) + 1).sqrt();
    assert_ftc(&integrand, &x, "∫ x/√(1+x²) dx");
}

#[test]
fn integrate_x_over_sqrt_quadratic_with_linear_term() {
    // ∫ x/√(x²+2x+5) dx = √(x²+2x+5) - I₀ term
    let x = symplex::default_context().symbol("x");
    let quad = &(&x.powi(2) + &(&symplex::default_context().int(2) * &x)) + 5;
    let integrand = &x / &quad.sqrt();
    assert_ftc(&integrand, &x, "∫ x/√(x²+2x+5) dx");
}
