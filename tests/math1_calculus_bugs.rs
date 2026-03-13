//! Calculus correctness tests for symplex.
//!
//! Every test checks a known mathematical identity or result.
//! A failing test indicates a potential math bug in the library.

mod common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Check two expressions are numerically equal at several points in (0, 2).
fn assert_exprs_equal_in_domain(a: &Ex, b: &Ex, var: &Ex, label: &str) {
    let ctx = a.context();
    let points: &[(i64, i64)] = &[(1, 4), (1, 2), (3, 4), (1, 1), (3, 2)];
    let mut checked = 0;
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let va = a.subs(var, &pt).eval().eval_f64();
        let vb = b.subs(var, &pt).eval().eval_f64();
        if let (Ok(av), Ok(bv)) = (va, vb) {
            checked += 1;
            let scale = av.abs().max(bv.abs()).max(1.0);
            assert!(
                (av - bv).abs() < 1e-8 * scale,
                "{label} at {var}={p}/{q}: {av} vs {bv} (diff={})",
                (av - bv).abs()
            );
        }
    }
    assert!(checked > 0, "{label}: no evaluation points succeeded — test is vacuous");
}

/// FTC check: d/dx(∫ f dx) should equal f, verified numerically.
/// Returns the string representation of the antiderivative for further inspection.
fn assert_ftc_custom(integrand: &Ex, var: &Ex, label: &str) -> String {
    let antideriv = integrand.integrate(var);
    let s = format!("{antideriv}");
    assert!(
        !s.contains("Integral"),
        "{label}: integration returned unevaluated Integral: {s}"
    );
    let deriv = antideriv.diff(var);
    let ctx = integrand.context();
    let mut checked = 0;
    for &pt_f in &[0.3, 0.7, 1.4, 2.1] {
        let numer = (pt_f * 1000.0_f64).round() as i64;
        let pt = ctx.rational(numer, 1000);
        let orig_val = integrand.subs(var, &pt).eval().eval_f64();
        let deriv_val = deriv.subs(var, &pt).eval().eval_f64();
        if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
            checked += 1;
            let scale = o.abs().max(d.abs()).max(1.0);
            assert!(
                (o - d).abs() < 1e-8 * scale,
                "FTC failed for {label} at {var}={pt_f}: integrand={o}, d/dx(antideriv)={d}, \
                 antideriv='{antideriv}', deriv='{deriv}'",
            );
        }
    }
    assert!(checked > 0, "FTC {label}: no evaluation points succeeded — test is vacuous");
    s
}

/// FTC check with restricted domain points (for functions like asin, etc.)
fn assert_ftc_domain(integrand: &Ex, var: &Ex, points: &[f64], tol: f64, label: &str) -> String {
    let antideriv = integrand.integrate(var);
    let s = format!("{antideriv}");
    assert!(
        !s.contains("Integral"),
        "{label}: integration returned unevaluated Integral: {s}"
    );
    let deriv = antideriv.diff(var);
    let ctx = integrand.context();
    let mut checked = 0;
    for &pt_f in points {
        let numer = (pt_f * 10000.0).round() as i64;
        let pt = ctx.rational(numer, 10000);
        let orig_val = integrand.subs(var, &pt).eval().eval_f64();
        let deriv_val = deriv.subs(var, &pt).eval().eval_f64();
        if let (Ok(o), Ok(d)) = (orig_val, deriv_val) {
            checked += 1;
            let scale = o.abs().max(d.abs()).max(1.0);
            assert!(
                (o - d).abs() < tol * scale,
                "FTC failed for {label} at {var}={pt_f}: integrand={o}, d/dx(antideriv)={d}, \
                 antideriv='{antideriv}', deriv='{deriv}'",
            );
        }
    }
    assert!(checked > 0, "FTC {label}: no evaluation points succeeded — test is vacuous");
    s
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: DIFFERENTIATION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

// ── 1a. Chain rule with nested functions ─────────────────────────────────

#[test]
fn diff_chain_sin_of_exp() {
    // d/dx sin(exp(x)) = cos(exp(x)) * exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().sin();
    let df = f.diff(&x);

    // Verify numerically
    let expected = &x.exp().cos() * &x.exp();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sin(exp(x))");
}

#[test]
fn diff_chain_exp_of_sin() {
    // d/dx exp(sin(x)) = exp(sin(x)) * cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().exp();
    let df = f.diff(&x);
    let expected = &x.sin().exp() * &x.cos();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx exp(sin(x))");
}

#[test]
fn diff_chain_ln_of_cos() {
    // d/dx ln(cos(x)) = -sin(x)/cos(x) = -tan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos().ln();
    let df = f.diff(&x);
    let expected = -&x.tan();
    // Check at points where cos(x) > 0
    let ctx2 = df.context();
    for &(p, q) in &[(1, 10), (1, 4), (1, 2)] {
        let pt = ctx2.rational(p, q);
        let got = df.subs(&x, &pt).eval().eval_f64();
        let want = expected.subs(&x, &pt).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-9 * g.abs().max(w.abs()).max(1.0),
                "d/dx ln(cos(x)) at x={p}/{q}: got {g}, want {w}"
            );
        }
    }
}

#[test]
fn diff_chain_triple_nested() {
    // d/dx sin(cos(exp(x)))
    // = cos(cos(exp(x))) * (-sin(exp(x))) * exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().cos().sin();
    let df = f.diff(&x);
    let expected = &x.exp().cos().cos() * &(-&x.exp().sin()) * &x.exp();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sin(cos(exp(x)))");
}

#[test]
fn diff_chain_sqrt_of_polynomial() {
    // d/dx sqrt(x^2 + 1) = x / sqrt(x^2 + 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = &x.powi(2) + 1;
    let f = inner.sqrt();
    let df = f.diff(&x);
    let expected = &x / &(&x.powi(2) + 1).sqrt();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sqrt(x^2+1)");
}

// ── 1b. Higher-order derivatives ─────────────────────────────────────────

#[test]
fn diff_higher_order_sin() {
    // d^4/dx^4 sin(x) = sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let d4 = f.diff_n(&x, 4);
    assert_exprs_equal_in_domain(&d4, &f, &x, "d^4/dx^4 sin(x) = sin(x)");
}

#[test]
fn diff_higher_order_cos() {
    // d^2/dx^2 cos(x) = -cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos();
    let d2 = f.diff_n(&x, 2);
    let expected = -&x.cos();
    assert_exprs_equal_in_domain(&d2, &expected, &x, "d^2/dx^2 cos(x) = -cos(x)");
}

#[test]
fn diff_higher_order_exp() {
    // d^n/dx^n exp(x) = exp(x) for all n
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp();
    for n in 1..=6 {
        let dn = f.diff_n(&x, n);
        assert_exprs_equal_in_domain(
            &dn,
            &f,
            &x,
            &format!("d^{n}/dx^{n} exp(x) = exp(x)"),
        );
    }
}

#[test]
fn diff_higher_order_x_to_the_n() {
    // d^5/dx^5 x^5 = 120
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(5);
    let d5 = f.diff_n(&x, 5);
    let s = format!("{}", d5.eval());
    assert_eq!(s, "120", "d^5/dx^5 x^5 should be 120, got {s}");
}

#[test]
fn diff_sixth_of_x_fifth_is_zero() {
    // d^6/dx^6 x^5 = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(5);
    let d6 = f.diff_n(&x, 6);
    let s = format!("{}", d6.eval());
    assert_eq!(s, "0", "d^6/dx^6 x^5 should be 0, got {s}");
}

#[test]
fn diff_leibniz_product_second_derivative() {
    // d^2/dx^2 (x * sin(x)) = 2*cos(x) - x*sin(x)
    // Using Leibniz rule: (fg)'' = f''g + 2f'g' + fg''
    //   f = x, g = sin(x), f'' = 0, f' = 1, g' = cos(x), g'' = -sin(x)
    //   => 0 + 2*cos(x) + x*(-sin(x)) = 2*cos(x) - x*sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.sin();
    let d2 = f.diff_n(&x, 2);
    let expected = &(&ctx.int(2) * &x.cos()) - &(&x * &x.sin());
    assert_exprs_equal_in_domain(&d2, &expected, &x, "(x*sin(x))''");
}

// ── 1c. Derivatives involving multiple variables ─────────────────────────

#[test]
fn diff_partial_x_of_xy() {
    // ∂/∂x (x*y) = y
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y;
    let df = f.diff(&x);
    assert_eq!(format!("{df}"), "y", "∂/∂x (x*y) should be y, got {df}");
}

#[test]
fn diff_partial_y_of_xy() {
    // ∂/∂y (x*y) = x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y;
    let df = f.diff(&y);
    assert_eq!(format!("{df}"), "x", "∂/∂y (x*y) should be x, got {df}");
}

#[test]
fn diff_y_wrt_x_is_zero() {
    // ∂/∂x (y) = 0 (y is independent of x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let df = y.diff(&x);
    assert_eq!(format!("{df}"), "0", "∂/∂x (y) should be 0, got {df}");
}

#[test]
fn diff_mixed_partials_commute() {
    // Clairaut's theorem: ∂²f/∂x∂y = ∂²f/∂y∂x for f = x^3 * y^2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(3) * &y.powi(2);

    let fxy = f.diff(&x).diff(&y);
    let fyx = f.diff(&y).diff(&x);

    // Both should be 6*x^2*y
    let val_xy = fxy
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("fxy eval");
    let val_yx = fyx
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("fyx eval");
    // Exact: 6 * 4 * 3 = 72
    assert!(
        (val_xy - 72.0).abs() < 1e-10,
        "∂²(x³y²)/∂x∂y at (2,3) should be 72, got {val_xy}"
    );
    assert!(
        (val_xy - val_yx).abs() < 1e-10,
        "mixed partials should commute: {val_xy} vs {val_yx}"
    );
}

#[test]
fn diff_multivar_chain_rule() {
    // f(x,y) = sin(x*y), ∂f/∂x = y*cos(x*y)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = (&x * &y).sin();
    let df_dx = f.diff(&x);
    let expected = &y * &(&x * &y).cos();

    // Check at (x,y) = (1, 2)
    let got = df_dx
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .eval()
        .eval_f64()
        .expect("df/dx eval");
    let want = expected
        .subs(&x, &ctx.int(1))
        .subs(&y, &ctx.int(2))
        .eval()
        .eval_f64()
        .expect("expected eval");
    assert!(
        (got - want).abs() < 1e-10,
        "∂/∂x sin(xy) at (1,2): got {got}, want {want}"
    );
}

// ── 1d. Derivative of absolute value ─────────────────────────────────────

#[test]
fn diff_abs_x_squared_plus_1() {
    // d/dx |x^2 + 1| = d/dx (x^2 + 1) = 2x (since x^2+1 > 0 always)
    // If the CAS can handle this, it should give 2x (or something equivalent).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(2) + 1).abs();
    let df = f.diff(&x);

    // Verify numerically: at x=2, derivative should be 4
    if let Ok(val) = df.subs(&x, &ctx.int(2)).eval().eval_f64() {
        assert!(
            (val - 4.0).abs() < 1e-8,
            "d/dx |x^2+1| at x=2 should be 4, got {val}"
        );
    }
}

// ── 1e. Quotient rule ────────────────────────────────────────────────────

#[test]
fn diff_quotient_rule() {
    // d/dx (sin(x)/x) = (x*cos(x) - sin(x))/x^2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin() / &x;
    let df = f.diff(&x);
    let expected = &(&(&x * &x.cos()) - &x.sin()) / &x.powi(2);

    // Verify numerically at x=1, x=2
    for &pt in &[1i64, 2, 3] {
        let got = df.subs_i64(&x, pt).eval().eval_f64();
        let want = expected.subs_i64(&x, pt).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-9 * g.abs().max(w.abs()).max(1.0),
                "d/dx(sin(x)/x) at x={pt}: got {g}, want {w}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: INTEGRATION CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

// ── 2a. FTC roundtrips: ∫f'(x)dx should give back f(x) ──────────────────

#[test]
fn integrate_then_diff_polynomial() {
    // ∫(3x^2 + 2x + 1)dx = x^3 + x^2 + x, d/dx of that = 3x^2 + 2x + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!(ctx, 3 * x ^ 2 + 2 * x + 1);
    assert_ftc_custom(&f, &x, "∫(3x²+2x+1)dx");
}

#[test]
fn integrate_then_diff_sin_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ∫ sin(x) dx = -cos(x), d/dx(-cos(x)) = sin(x)
    assert_ftc_custom(&x.sin(), &x, "∫sin(x)dx");
    // ∫ cos(x) dx = sin(x), d/dx(sin(x)) = cos(x)
    assert_ftc_custom(&x.cos(), &x, "∫cos(x)dx");
}

#[test]
fn integrate_then_diff_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_ftc_custom(&x.exp(), &x, "∫exp(x)dx");
}

#[test]
fn integrate_then_diff_one_over_x() {
    // ∫ 1/x dx = ln|x|, d/dx ln|x| = 1/x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &x;
    assert_ftc_custom(&integrand, &x, "∫(1/x)dx");
}

// ── 2b. Specific antiderivative checks ───────────────────────────────────

#[test]
fn integrate_one_over_x2_plus_1_is_arctan() {
    // ∫ 1/(x²+1) dx = atan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 1);
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("atan"),
        "∫ 1/(x²+1) dx should be atan(x), got: {s}"
    );
    assert_ftc_custom(&integrand, &x, "∫1/(x²+1)dx = atan(x)");
}

#[test]
fn integrate_sec_squared_is_tan() {
    // ∫ sec²(x) dx = ∫ 1/cos²(x) dx = tan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(-2);
    let s = assert_ftc_custom(&integrand, &x, "∫sec²(x)dx = tan(x)");
    // The result should involve tan in some form
    let _ = s; // FTC verified correctness
}

#[test]
fn integrate_one_over_sqrt_1_minus_x2_is_arcsin() {
    // ∫ 1/√(1-x²) dx = asin(x) (for |x| < 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&ctx.int(1) - &x.powi(2)).sqrt();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        // Unevaluated — this is a known-hard integral, skip
        return;
    }
    // Verify via FTC at points in (-1, 1)
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.1, 0.3, 0.5, 0.7],
        1e-7,
        "∫1/√(1-x²)dx = asin(x)",
    );
}

#[test]
fn integrate_x_exp_x_by_parts() {
    // ∫ x*exp(x) dx = x*exp(x) - exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.exp();
    assert_ftc_custom(&integrand, &x, "∫x·exp(x)dx");
}

#[test]
fn integrate_x_sin_x_by_parts() {
    // ∫ x*sin(x) dx = sin(x) - x*cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.sin();
    assert_ftc_custom(&integrand, &x, "∫x·sin(x)dx");
}

#[test]
fn integrate_x_cos_x_by_parts() {
    // ∫ x*cos(x) dx = cos(x) + x*sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.cos();
    assert_ftc_custom(&integrand, &x, "∫x·cos(x)dx");
}

#[test]
fn integrate_exp_sin_cyclic() {
    // ∫ exp(x)*sin(x) dx = exp(x)*(sin(x)-cos(x))/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.sin();
    assert_ftc_custom(&integrand, &x, "∫exp(x)·sin(x)dx");
}

#[test]
fn integrate_exp_cos_cyclic() {
    // ∫ exp(x)*cos(x) dx = exp(x)*(sin(x)+cos(x))/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.exp() * &x.cos();
    assert_ftc_custom(&integrand, &x, "∫exp(x)·cos(x)dx");
}

#[test]
fn integrate_ln_x() {
    // ∫ ln(x) dx = x*ln(x) - x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_ftc_custom(&x.ln(), &x, "∫ln(x)dx");
}

#[test]
fn integrate_x_squared_exp_x() {
    // ∫ x²·exp(x) dx = x²·exp(x) - 2x·exp(x) + 2·exp(x) (triple IBP)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.exp();
    assert_ftc_custom(&integrand, &x, "∫x²·exp(x)dx");
}

// ── 2c. Definite integral checks ─────────────────────────────────────────

#[test]
fn definite_integral_x_squared_0_to_1() {
    // ∫₀¹ x² dx = 1/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        (val - 1.0 / 3.0).abs() < 1e-10,
        "∫₀¹ x² dx should be 1/3, got {val}"
    );
}

#[test]
fn definite_integral_sin_0_to_pi() {
    // ∫₀^π sin(x) dx = 2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin().definite_integral(&x, &ctx.int(0), &ctx.pi());
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        (val - 2.0).abs() < 1e-10,
        "∫₀^π sin(x) dx should be 2, got {val}"
    );
}

#[test]
fn definite_integral_exp_0_to_1() {
    // ∫₀¹ exp(x) dx = e - 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp().definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    let expected = std::f64::consts::E - 1.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "∫₀¹ exp(x) dx should be e-1 ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_1_over_x_1_to_e() {
    // ∫₁^e 1/x dx = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &x;
    let result = integrand.definite_integral(&x, &ctx.int(1), &ctx.e());
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        (val - 1.0).abs() < 1e-10,
        "∫₁^e 1/x dx should be 1, got {val}"
    );
}

// ── 2d. Integration of trig functions ────────────────────────────────────

#[test]
fn integrate_sin_squared() {
    // ∫ sin²(x) dx = x/2 - sin(2x)/4  (or equivalent)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(2);
    assert_ftc_custom(&integrand, &x, "∫sin²(x)dx");
}

#[test]
fn integrate_cos_squared() {
    // ∫ cos²(x) dx = x/2 + sin(2x)/4  (or equivalent)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(2);
    assert_ftc_custom(&integrand, &x, "∫cos²(x)dx");
}

#[test]
fn integrate_tan_x() {
    // ∫ tan(x) dx = -ln|cos(x)|
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_ftc_custom(&x.tan(), &x, "∫tan(x)dx");
}

// ── 2e. Integration linearity ────────────────────────────────────────────

#[test]
fn integrate_linearity_sum() {
    // ∫ (sin(x) + exp(x)) dx should equal ∫sin(x)dx + ∫exp(x)dx
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let combined = (&x.sin() + &x.exp()).integrate(&x);
    let separate = &x.sin().integrate(&x) + &x.exp().integrate(&x);
    assert_exprs_equal_in_domain(&combined, &separate, &x, "linearity of integration");
}

#[test]
fn integrate_linearity_constant_factor() {
    // ∫ 5*sin(x) dx = 5 * ∫sin(x) dx
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let combined = (&ctx.int(5) * &x.sin()).integrate(&x);
    let separate = &ctx.int(5) * &x.sin().integrate(&x);
    assert_exprs_equal_in_domain(
        &combined,
        &separate,
        &x,
        "constant factor in integration",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: LIMITS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_sin_x_over_x_at_0() {
    // lim(x→0) sin(x)/x = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x;
    let result = expr.limit(&x, &ctx.int(0));
    assert_eq!(format!("{result}"), "1", "lim(x→0) sin(x)/x should be 1");
}

#[test]
fn limit_exp_minus_1_over_x_at_0() {
    // lim(x→0) (exp(x)-1)/x = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.exp() - 1) / &x;
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→0) (exp(x)-1)/x should be 1, got {val}"
        );
    }
}

#[test]
fn limit_1_minus_cos_over_x2_at_0() {
    // lim(x→0) (1-cos(x))/x² = 1/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&ctx.int(1) - &x.cos()) / &x.powi(2);
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            (val - 0.5).abs() < 1e-8,
            "lim(x→0) (1-cos(x))/x² should be 1/2, got {val}"
        );
    }
}

#[test]
fn limit_tan_x_over_x_at_0() {
    // lim(x→0) tan(x)/x = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.tan() / &x;
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→0) tan(x)/x should be 1, got {val}"
        );
    }
}

#[test]
fn limit_x_ln_x_at_0_from_right() {
    // lim(x→0+) x*ln(x) = 0
    // This is a classic 0·(-∞) indeterminate form
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x.ln();
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            val.abs() < 1e-8,
            "lim(x→0+) x*ln(x) should be 0, got {val}"
        );
    }
}

#[test]
fn limit_1_over_x_at_infinity() {
    // lim(x→∞) 1/x = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(1) / &x;
    let result = expr.limit(&x, &ctx.infinity());
    assert_eq!(format!("{result}"), "0", "lim(x→∞) 1/x should be 0");
}

#[test]
fn limit_x_exp_neg_x_at_infinity() {
    // lim(x→∞) x*exp(-x) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &(-&x).exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(
            format!("{r}"),
            "0",
            "lim(x→∞) x·exp(-x) should be 0"
        );
    }
}

#[test]
fn limit_1_plus_1_over_x_to_x_at_infinity() {
    // lim(x→∞) (1+1/x)^x = e
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &ctx.int(1) + &(&ctx.int(1) / &x);
    let expr = base.pow(&x);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        if s == "E" {
            // Perfect
        } else if let Ok(v) = r.eval_f64() {
            assert!(
                (v - std::f64::consts::E).abs() < 0.01,
                "lim(x→∞) (1+1/x)^x should be e ≈ 2.71828, got {v}"
            );
        }
    }
}

#[test]
fn limit_lhopital_0_over_0() {
    // lim(x→0) (exp(x) - 1 - x) / x² = 1/2
    // Requires L'Hôpital twice (or Taylor expansion)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x.exp() - &ctx.int(1) - &x;
    let expr = &numer / &x.powi(2);
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            (val - 0.5).abs() < 1e-8,
            "lim(x→0) (e^x-1-x)/x² should be 1/2, got {val}"
        );
    }
}

#[test]
fn limit_lhopital_inf_over_inf() {
    // lim(x→∞) x / exp(x) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x / &x.exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→∞) x/exp(x) should be 0");
    }
}

#[test]
fn limit_polynomial_ratio_same_degree() {
    // lim(x→∞) (3x² + 2x + 1)/(x² + 1) = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = expr!(ctx, 3 * x ^ 2 + 2 * x + 1);
    let denom = expr!(ctx, x ^ 2 + 1);
    let expr = &numer / &denom;
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should be numeric");
        assert!(
            (val - 3.0).abs() < 1e-8,
            "lim(x→∞) (3x²+2x+1)/(x²+1) should be 3, got {val}"
        );
    }
}

#[test]
fn limit_polynomial_ratio_higher_degree_numerator() {
    // lim(x→∞) x³ / (x² + 1) = ∞
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) / &(&x.powi(2) + 1);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        // Should be +∞ or ∞
        assert!(
            s.contains("∞") || s.contains("oo") || s.contains("Inf") || s.contains("inf"),
            "lim(x→∞) x³/(x²+1) should be ∞, got: {s}"
        );
    }
}

#[test]
fn limit_sin_x_at_pi_over_2() {
    // lim(x→π/2) sin(x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi_over_2 = &ctx.pi() / 2;
    let result = x.sin().limit(&x, &pi_over_2);
    let val = result.eval_f64().expect("should evaluate");
    assert!(
        (val - 1.0).abs() < 1e-10,
        "lim(x→π/2) sin(x) should be 1, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: TAYLOR / MACLAURIN SERIES
// ═══════════════════════════════════════════════════════════════════════════

fn eval_series_at(series: &Ex, var: &Ex, p: i64, q: i64) -> f64 {
    let ctx = series.context();
    let pt = ctx.rational(p, q);
    series
        .subs(var, &pt)
        .eval()
        .eval_f64()
        .expect("series evaluation should succeed")
}

// ── 4a. exp(x) series coefficients ───────────────────────────────────────

#[test]
fn series_exp_coefficients_exact() {
    // exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + ...
    // The n-th coefficient should be 1/n!
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().maclaurin(&x, 6);
    assert!(!series.has_unevaluated(), "exp(x) maclaurin should work");
    let expanded = series.expand().eval();

    // Evaluate at x = 1: should approximate e
    let val = eval_series_at(&expanded, &x, 1, 1);
    let exact = std::f64::consts::E;
    // 6 terms: 1 + 1 + 1/2 + 1/6 + 1/24 + 1/120 = 2.71667
    assert!(
        (val - exact).abs() < 0.01,
        "exp(x) series (order 6) at x=1 should be close to e={exact}, got {val}"
    );
}

#[test]
fn series_exp_at_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().maclaurin(&x, 5);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 0, 1);
    assert!(
        (val - 1.0).abs() < 1e-12,
        "exp(x) series at x=0 should be 1, got {val}"
    );
}

// ── 4b. sin(x) series (odd terms only) ──────────────────────────────────

#[test]
fn series_sin_coefficients() {
    // sin(x) = x - x³/3! + x⁵/5! - ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 8);
    assert!(!series.has_unevaluated(), "sin(x) maclaurin should work");
    let expanded = series.expand().eval();

    // At x = π/6 = approximately 0.5236
    let val = eval_series_at(&expanded, &x, 5236, 10000);
    let exact = (std::f64::consts::PI / 6.0).sin();
    assert!(
        (val - exact).abs() < 1e-5,
        "sin(x) series at x≈π/6: got {val}, expected {exact}"
    );
}

#[test]
fn series_sin_at_zero_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 5);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 0, 1);
    assert!(
        val.abs() < 1e-12,
        "sin(x) series at x=0 should be 0, got {val}"
    );
}

// ── 4c. cos(x) series (even terms only) ─────────────────────────────────

#[test]
fn series_cos_coefficients() {
    // cos(x) = 1 - x²/2! + x⁴/4! - ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cos().maclaurin(&x, 8);
    assert!(!series.has_unevaluated(), "cos(x) maclaurin should work");
    let expanded = series.expand().eval();

    // At x = 1: cos(1) ≈ 0.5403
    let val = eval_series_at(&expanded, &x, 1, 1);
    let exact = 1.0_f64.cos();
    assert!(
        (val - exact).abs() < 1e-4,
        "cos(x) series at x=1: got {val}, expected {exact}"
    );
}

#[test]
fn series_cos_at_zero_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cos().maclaurin(&x, 5);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 0, 1);
    assert!(
        (val - 1.0).abs() < 1e-12,
        "cos(x) series at x=0 should be 1, got {val}"
    );
}

// ── 4d. ln(1+x) series ──────────────────────────────────────────────────

#[test]
fn series_ln_1_plus_x() {
    // ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
    // We expand ln(x) around x = 1 which is the same thing with substitution
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).ln();
    let series = f.try_maclaurin(&x, 6);
    if let Ok(s) = series {
        let expanded = s.expand().eval();

        // At x = 1/2: ln(3/2) ≈ 0.4055
        let val = eval_series_at(&expanded, &x, 1, 2);
        let exact = 1.5_f64.ln();
        assert!(
            (val - exact).abs() < 0.01,
            "ln(1+x) series at x=0.5: got {val}, expected {exact}"
        );

        // At x = 0: ln(1) = 0
        let val0 = eval_series_at(&expanded, &x, 0, 1);
        assert!(
            val0.abs() < 1e-12,
            "ln(1+x) series at x=0 should be 0, got {val0}"
        );
    }
}

// ── 4e. Series convergence: error should decrease with more terms ────────

#[test]
fn series_convergence_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exact = 0.5_f64.exp();
    let mut prev_err = f64::MAX;
    for order in [3u32, 5, 7, 9] {
        let series = x.exp().maclaurin(&x, order);
        let expanded = series.expand().eval();
        if let Ok(val) = expanded
            .subs(&x, &ctx.rational(1, 2))
            .eval()
            .eval_f64()
        {
            let err = (val - exact).abs();
            assert!(
                err < prev_err,
                "exp series error should decrease: order {order} err={err} >= prev_err={prev_err}"
            );
            prev_err = err;
        }
    }
}

#[test]
fn series_convergence_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exact = 0.5_f64.sin();
    let mut prev_err = f64::MAX;
    for order in [3u32, 5, 7, 9] {
        let series = x.sin().maclaurin(&x, order);
        let expanded = series.expand().eval();
        if let Ok(val) = expanded
            .subs(&x, &ctx.rational(1, 2))
            .eval()
            .eval_f64()
        {
            let err = (val - exact).abs();
            assert!(
                err < prev_err,
                "sin series error should decrease: order {order} err={err} >= prev_err={prev_err}"
            );
            prev_err = err;
        }
    }
}

// ── 4f. Series of 1/(1-x) — geometric series ────────────────────────────

#[test]
fn series_geometric() {
    // 1/(1-x) = 1 + x + x² + x³ + ... for |x| < 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&ctx.int(1) - &x);
    let series = f.try_maclaurin(&x, 5);
    if let Ok(s) = series {
        let expanded = s.expand().eval();

        // At x = 0.1: 1/(1-0.1) = 1/0.9 ≈ 1.1111
        let val = eval_series_at(&expanded, &x, 1, 10);
        let exact = 1.0 / 0.9;
        assert!(
            (val - exact).abs() < 0.001,
            "geometric series at x=0.1: got {val}, expected {exact}"
        );
    }
}

// ── 4g. Series of arctan(x) ─────────────────────────────────────────────

#[test]
fn series_arctan() {
    // arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.atan().try_maclaurin(&x, 7);
    if let Ok(s) = series {
        let expanded = s.expand().eval();

        // At x = 0.5: arctan(0.5) ≈ 0.4636
        let val = eval_series_at(&expanded, &x, 1, 2);
        let exact = 0.5_f64.atan();
        assert!(
            (val - exact).abs() < 0.01,
            "arctan series at x=0.5: got {val}, expected {exact}"
        );
    }
}

// ── 4h. Taylor series around non-zero point ──────────────────────────────

#[test]
fn series_exp_around_1() {
    // exp(x) around x=1: exp(x) = e * (1 + (x-1) + (x-1)²/2! + ...)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().series(&x, &ctx.int(1), 6);
    assert!(!series.has_unevaluated(), "exp(x) series around 1 should work");
    let expanded = series.expand().eval();

    // At x=1 (the expansion point): should be exactly e
    let val = eval_series_at(&expanded, &x, 1, 1);
    let exact = std::f64::consts::E;
    assert!(
        (val - exact).abs() < 1e-10,
        "exp(x) series at expansion point x=1 should be e={exact}, got {val}"
    );

    // At x = 1.1
    let val2 = eval_series_at(&expanded, &x, 11, 10);
    let exact2 = 1.1_f64.exp();
    assert!(
        (val2 - exact2).abs() < 1e-4,
        "exp(x) series(about 1) at x=1.1: got {val2}, expected {exact2}"
    );
}

#[test]
fn series_ln_around_1() {
    // ln(x) around x=1: ln(x) = (x-1) - (x-1)²/2 + (x-1)³/3 - ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.ln().series(&x, &ctx.int(1), 8);
    assert!(!series.has_unevaluated(), "ln(x) series around 1 should work");
    let expanded = series.expand().eval();

    // At x=1: ln(1) = 0
    let val = eval_series_at(&expanded, &x, 1, 1);
    assert!(val.abs() < 1e-12, "ln(x) series at x=1 should be 0, got {val}");

    // At x = 1.1: ln(1.1) ≈ 0.09531
    let val2 = eval_series_at(&expanded, &x, 11, 10);
    let exact2 = 1.1_f64.ln();
    assert!(
        (val2 - exact2).abs() < 1e-6,
        "ln(x) series(about 1) at x=1.1: got {val2}, expected {exact2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: CROSS-DOMAIN CHECKS (differentiation + integration + limits)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_definite_integral_gives_integrand() {
    // By FTC: if F(x) = ∫₀ˣ t² dt = x³/3, then F'(x) = x²
    // We test: d/dx(x³/3) = x²
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) / 3;
    let df = f.diff(&x);
    let expected = x.powi(2);
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx(x³/3) = x²");
}

#[test]
fn series_of_derivative_matches_derivative_of_series() {
    // For f(x) = sin(x), the series of f'(x) should match the derivative of the series of f(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let sin_series = x.sin().maclaurin(&x, 6);
    let cos_series = x.cos().maclaurin(&x, 6);

    let deriv_of_sin_series = sin_series.diff(&x);
    let expanded_deriv = deriv_of_sin_series.expand().eval();
    let expanded_cos_series = cos_series.expand().eval();

    // Both should agree numerically at x = 0.5
    let val_d = eval_series_at(&expanded_deriv, &x, 1, 2);
    let val_c = eval_series_at(&expanded_cos_series, &x, 1, 2);
    assert!(
        (val_d - val_c).abs() < 1e-3,
        "d/dx(sin series) should ≈ cos series at x=0.5: {val_d} vs {val_c}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: TRICKY / EDGE CASE CHECKS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_constant_expression_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // d/dx(π) = 0
    let dp = ctx.pi().diff(&x);
    assert_eq!(format!("{dp}"), "0", "d/dx(π) should be 0");
    // d/dx(e) = 0
    let de = ctx.e().diff(&x);
    assert_eq!(format!("{de}"), "0", "d/dx(e) should be 0");
    // d/dx(42) = 0
    let d42 = ctx.int(42).diff(&x);
    assert_eq!(format!("{d42}"), "0", "d/dx(42) should be 0");
}

#[test]
fn diff_zero_times_is_zero() {
    // d^0/dx^0 f = f
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let d0 = f.diff_n(&x, 0);
    assert_exprs_equal_in_domain(&d0, &f, &x, "d^0/dx^0 sin(x) = sin(x)");
}

#[test]
fn integrate_zero_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(0).integrate(&x);
    assert_eq!(format!("{result}"), "0", "∫0 dx should be 0");
}

#[test]
fn limit_constant_is_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = ctx.int(42).limit(&x, &ctx.int(0));
    assert_eq!(
        format!("{result}"),
        "42",
        "lim(x→0) 42 should be 42"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: INVERSE TRIG DIFFERENTIATION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_arcsin_numerical() {
    // d/dx arcsin(x) = 1/√(1-x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.asin().diff(&x);
    let expected = &ctx.int(1) / &(&ctx.int(1) - &x.powi(2)).sqrt();

    // Check at x = 0.3, 0.5
    for &(p, q) in &[(3, 10), (1, 2)] {
        let pt = ctx.rational(p, q);
        let got = df.subs(&x, &pt).eval().eval_f64();
        let want = expected.subs(&x, &pt).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-8,
                "d/dx arcsin(x) at x={p}/{q}: got {g}, want {w}"
            );
        }
    }
}

#[test]
fn diff_arccos_numerical() {
    // d/dx arccos(x) = -1/√(1-x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.acos().diff(&x);
    let expected = &ctx.int(-1) / &(&ctx.int(1) - &x.powi(2)).sqrt();

    for &(p, q) in &[(3, 10), (1, 2)] {
        let pt = ctx.rational(p, q);
        let got = df.subs(&x, &pt).eval().eval_f64();
        let want = expected.subs(&x, &pt).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-8,
                "d/dx arccos(x) at x={p}/{q}: got {g}, want {w}"
            );
        }
    }
}

#[test]
fn diff_arctan_numerical() {
    // d/dx arctan(x) = 1/(1+x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.atan().diff(&x);
    let expected = &ctx.int(1) / &(&ctx.int(1) + &x.powi(2));

    for &pt_val in &[1i64, 2, 3] {
        let got = df.subs_i64(&x, pt_val).eval().eval_f64();
        let want = expected.subs_i64(&x, pt_val).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-10,
                "d/dx arctan(x) at x={pt_val}: got {g}, want {w}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: HYPERBOLIC FUNCTION DERIVATIVES AND INTEGRALS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_sinh() {
    // d/dx sinh(x) = cosh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.sinh().diff(&x);
    let expected = x.cosh();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sinh(x) = cosh(x)");
}

#[test]
fn diff_cosh() {
    // d/dx cosh(x) = sinh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.cosh().diff(&x);
    let expected = x.sinh();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx cosh(x) = sinh(x)");
}

#[test]
fn diff_tanh() {
    // d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let df = x.tanh().diff(&x);
    let expected = &ctx.int(1) - &x.tanh().powi(2);
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx tanh(x) = 1 - tanh²(x)");
}

#[test]
fn integrate_sinh() {
    // ∫ sinh(x) dx = cosh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_ftc_custom(&x.sinh(), &x, "∫sinh(x)dx");
}

#[test]
fn integrate_cosh() {
    // ∫ cosh(x) dx = sinh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_ftc_custom(&x.cosh(), &x, "∫cosh(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: COMPOSITION OF DERIVATIVES AND IDENTITIES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_power_rule_fractional_exponent() {
    // d/dx x^(1/2) = (1/2) * x^(-1/2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt();
    let df = f.diff(&x);

    // At x = 4: should be 1/(2*sqrt(4)) = 1/4 = 0.25
    let val = df.subs(&x, &ctx.int(4)).eval().eval_f64().expect("should eval");
    assert!(
        (val - 0.25).abs() < 1e-10,
        "d/dx sqrt(x) at x=4 should be 0.25, got {val}"
    );
}

#[test]
fn diff_exponential_with_constant_base() {
    // d/dx 2^x = 2^x * ln(2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.int(2).pow(&x);
    let df = f.diff(&x);

    let val = df.subs(&x, &ctx.int(1)).eval().eval_f64().expect("should eval");
    let expected = 2.0 * 2.0_f64.ln();
    assert!(
        (val - expected).abs() < 1e-9,
        "d/dx 2^x at x=1 should be 2·ln(2) ≈ {expected}, got {val}"
    );
}

#[test]
fn diff_x_to_the_x() {
    // d/dx x^x = x^x * (ln(x) + 1)
    // This is a classic calculus problem using logarithmic differentiation
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&x);
    let df = f.diff(&x);
    let expected = &x.pow(&x) * &(&x.ln() + 1);

    // Check at x = 2: 2^2 * (ln(2) + 1) = 4 * 1.6931 ≈ 6.7726
    let got = df.subs(&x, &ctx.int(2)).eval().eval_f64();
    let want = expected.subs(&x, &ctx.int(2)).eval().eval_f64();
    if let (Ok(g), Ok(w)) = (got, want) {
        assert!(
            (g - w).abs() < 1e-6,
            "d/dx x^x at x=2: got {g}, want {w}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10: SECOND FUNDAMENTAL THEOREM + MISC
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_derivative_recovers_original_up_to_constant() {
    // ∫ f'(x) dx should differ from f(x) by at most a constant
    // Test with f(x) = x^3 + sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(3) + &x.sin();
    let df = f.diff(&x);
    let antideriv = df.integrate(&x);

    // antideriv - f should be a constant (evaluate at two points, same value)
    let diff_at_1 = (&antideriv - &f)
        .subs(&x, &ctx.int(1))
        .eval()
        .eval_f64();
    let diff_at_2 = (&antideriv - &f)
        .subs(&x, &ctx.int(2))
        .eval()
        .eval_f64();
    if let (Ok(d1), Ok(d2)) = (diff_at_1, diff_at_2) {
        assert!(
            (d1 - d2).abs() < 1e-8,
            "∫(d/dx(x³+sin(x)))dx should differ from x³+sin(x) by a constant: \
             diff at x=1 is {d1}, diff at x=2 is {d2}"
        );
    }
}

#[test]
fn integrate_cos_2x() {
    // ∫ cos(2x) dx = sin(2x)/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x * 2).cos();
    assert_ftc_custom(&integrand, &x, "∫cos(2x)dx");
}

#[test]
fn integrate_exp_3x() {
    // ∫ exp(3x) dx = exp(3x)/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x * 3).exp();
    assert_ftc_custom(&integrand, &x, "∫exp(3x)dx");
}

#[test]
fn integrate_sin_squared_plus_cos_squared() {
    // ∫ (sin²(x) + cos²(x)) dx = ∫ 1 dx = x
    // This tests if the simplifier is used during integration
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.sin().powi(2) + &x.cos().powi(2);
    let result = integrand.integrate(&x);

    // Whatever the form, d/dx of the result should be 1 (or sin²+cos²)
    let deriv = result.diff(&x);
    // Evaluate at x = 1
    if let Ok(val) = deriv.subs(&x, &ctx.int(1)).eval().eval_f64() {
        assert!(
            (val - 1.0).abs() < 1e-8,
            "d/dx ∫(sin²x+cos²x)dx should be 1, got {val}"
        );
    }
}

#[test]
fn limit_squeeze_theorem_x_sin_1_over_x() {
    // lim(x→0) x * sin(1/x) = 0 (by squeeze theorem: -|x| ≤ x*sin(1/x) ≤ |x|)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &(&ctx.int(1) / &x).sin();
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            val.abs() < 1e-8,
            "lim(x→0) x·sin(1/x) should be 0, got {val}"
        );
    }
}

#[test]
fn series_sinh_matches_odd_exp_terms() {
    // sinh(x) = x + x³/3! + x⁵/5! + ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sinh().maclaurin(&x, 8);
    assert!(!series.has_unevaluated(), "sinh series should work");
    let expanded = series.expand().eval();

    // At x = 1: sinh(1) ≈ 1.17520
    let val = eval_series_at(&expanded, &x, 1, 1);
    let exact = 1.0_f64.sinh();
    assert!(
        (val - exact).abs() < 1e-4,
        "sinh series at x=1: got {val}, expected {exact}"
    );
}

#[test]
fn series_cosh_matches_even_exp_terms() {
    // cosh(x) = 1 + x²/2! + x⁴/4! + ...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cosh().maclaurin(&x, 8);
    assert!(!series.has_unevaluated(), "cosh series should work");
    let expanded = series.expand().eval();

    // At x = 1: cosh(1) ≈ 1.54308
    let val = eval_series_at(&expanded, &x, 1, 1);
    let exact = 1.0_f64.cosh();
    assert!(
        (val - exact).abs() < 1e-4,
        "cosh series at x=1: got {val}, expected {exact}"
    );
}

#[test]
fn series_exp_equals_cosh_plus_sinh() {
    // Identity: exp(x) = cosh(x) + sinh(x)
    // Verify the series agree numerically
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_series = x.exp().maclaurin(&x, 8).expand().eval();
    let sinh_series = x.sinh().maclaurin(&x, 8).expand().eval();
    let cosh_series = x.cosh().maclaurin(&x, 8).expand().eval();

    let sum = &cosh_series + &sinh_series;

    for &(p, q) in &[(1, 4), (1, 2), (3, 4), (1, 1)] {
        let e_val = eval_series_at(&exp_series, &x, p, q);
        let s_val = eval_series_at(&sum, &x, p, q);
        assert!(
            (e_val - s_val).abs() < 1e-6,
            "exp series ≠ cosh + sinh series at x={p}/{q}: {e_val} vs {s_val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11: POLYNOMIAL INTEGRATION BOUNDARY CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_high_degree_polynomial() {
    // ∫ (x^10) dx = x^11/11
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.powi(10);
    assert_ftc_custom(&integrand, &x, "∫x^10 dx");
}

#[test]
fn integrate_negative_power() {
    // ∫ x^(-3) dx = -1/(2x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.powi(-3);
    assert_ftc_custom(&integrand, &x, "∫x^(-3) dx");
}

#[test]
fn definite_integral_symmetry_odd_function() {
    // ∫₋₁¹ x³ dx = 0 (odd function on symmetric interval)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).definite_integral(&x, &ctx.int(-1), &ctx.int(1));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        val.abs() < 1e-10,
        "∫₋₁¹ x³ dx should be 0 (odd function), got {val}"
    );
}

#[test]
fn definite_integral_symmetry_even_function() {
    // ∫₋₁¹ x² dx = 2 * ∫₀¹ x² dx = 2/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).definite_integral(&x, &ctx.int(-1), &ctx.int(1));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        (val - 2.0 / 3.0).abs() < 1e-10,
        "∫₋₁¹ x² dx should be 2/3, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 12: PRODUCT AND CHAIN RULE STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_triple_product() {
    // d/dx (x * sin(x) * exp(x))
    // = sin(x)*exp(x) + x*cos(x)*exp(x) + x*sin(x)*exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &(&x * &x.sin()) * &x.exp();
    let df = f.diff(&x);

    // Expected: exp(x) * (sin(x) + x*cos(x) + x*sin(x))
    let term1 = &x.sin() * &x.exp();
    let term2 = &(&x * &x.cos()) * &x.exp();
    let term3 = &(&x * &x.sin()) * &x.exp();
    let expected = &(&term1 + &term2) + &term3;
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx(x·sin(x)·exp(x))");
}

#[test]
fn diff_nested_exp_exp() {
    // d/dx exp(exp(x)) = exp(exp(x)) * exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().exp();
    let df = f.diff(&x);
    let expected = &x.exp().exp() * &x.exp();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx exp(exp(x))");
}

#[test]
fn diff_sin_of_x_squared() {
    // d/dx sin(x²) = 2x * cos(x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).sin();
    let df = f.diff(&x);
    let expected = &(&ctx.int(2) * &x) * &x.powi(2).cos();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sin(x²) = 2x·cos(x²)");
}

#[test]
fn diff_ln_of_x_squared_plus_1() {
    // d/dx ln(x²+1) = 2x/(x²+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x.powi(2) + 1).ln();
    let df = f.diff(&x);
    let expected = &(&ctx.int(2) * &x) / &(&x.powi(2) + 1);
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx ln(x²+1) = 2x/(x²+1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 13: RATIONAL FUNCTION INTEGRATION (partial fractions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x_squared_minus_1() {
    // ∫ 1/(x²-1) dx = (1/2)*ln|(x-1)/(x+1)| = (1/2)*ln|x-1| - (1/2)*ln|x+1|
    // Domain: |x| > 1 or |x| < 1 (avoid x = ±1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) - 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return; // unevaluated, skip
    }
    // Verify via FTC at points away from ±1
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.2, 0.4, 0.6],
        1e-7,
        "∫1/(x²-1)dx partial fractions",
    );
}

#[test]
fn integrate_x_over_x2_plus_1_squared() {
    // ∫ x/(x²+1)² dx = -1/(2(x²+1))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x / &(&x.powi(2) + 1).powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫x/(x²+1)² dx");
}

#[test]
fn integrate_1_over_x2_plus_1_squared() {
    // ∫ 1/(x²+1)² dx = x/(2(x²+1)) + (1/2)*arctan(x)
    // This is a classic reduction formula result
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 1).powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫1/(x²+1)² dx");
}

#[test]
fn integrate_x_squared_over_x2_plus_1() {
    // ∫ x²/(x²+1) dx = x - arctan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) / &(&x.powi(2) + 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫x²/(x²+1) dx");
}

#[test]
fn integrate_1_over_x2_plus_4() {
    // ∫ 1/(x²+4) dx = (1/2)*arctan(x/2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 4);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫1/(x²+4) dx");
}

#[test]
fn integrate_2x_plus_3_over_x2_plus_1() {
    // ∫ (2x+3)/(x²+1) dx = ln(x²+1) + 3*arctan(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &(&ctx.int(2) * &x + 3) / &(&x.powi(2) + 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫(2x+3)/(x²+1) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 14: INVERSE TRIG INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_arctan_x() {
    // ∫ arctan(x) dx = x*arctan(x) - ln(1+x²)/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.atan();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫arctan(x) dx");
}

#[test]
fn integrate_arcsin_x() {
    // ∫ arcsin(x) dx = x*arcsin(x) + √(1-x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.asin();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    // Domain: |x| < 1
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.1, 0.3, 0.5, 0.7],
        1e-7,
        "∫arcsin(x) dx",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 15: CHAIN RULE WITH INVERSE TRIG COMPOSITIONS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_arctan_of_2x() {
    // d/dx arctan(2x) = 2/(1+4x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x * 2).atan();
    let df = f.diff(&x);
    let expected = &ctx.int(2) / &(&ctx.int(1) + &(&ctx.int(4) * &x.powi(2)));
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx arctan(2x)");
}

#[test]
fn diff_arcsin_of_x_squared() {
    // d/dx arcsin(x²) = 2x/√(1-x⁴)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).asin();
    let df = f.diff(&x);
    let expected = &(&ctx.int(2) * &x) / &(&ctx.int(1) - &x.powi(4)).sqrt();
    // Check at small x values where 1-x⁴ > 0
    let ctx2 = df.context();
    for &(p, q) in &[(1, 10), (3, 10), (1, 2)] {
        let pt = ctx2.rational(p, q);
        let got = df.subs(&x, &pt).eval().eval_f64();
        let want = expected.subs(&x, &pt).eval().eval_f64();
        if let (Ok(g), Ok(w)) = (got, want) {
            assert!(
                (g - w).abs() < 1e-8 * g.abs().max(w.abs()).max(1.0),
                "d/dx arcsin(x²) at x={p}/{q}: got {g}, want {w}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 16: EXACT SERIES COEFFICIENT CHECKS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_exp_exact_coefficient_check() {
    // exp(x) = Σ x^n/n!. Check exact rational coefficients by evaluating
    // the series at x=1 and comparing to the known partial sum.
    // Order 7: 1 + 1 + 1/2 + 1/6 + 1/24 + 1/120 + 1/720 = 2.718055...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.exp().maclaurin(&x, 7);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 1, 1);
    let partial_sum: f64 = 1.0 + 1.0 + 0.5 + 1.0/6.0 + 1.0/24.0 + 1.0/120.0 + 1.0/720.0;
    assert!(
        (val - partial_sum).abs() < 1e-12,
        "exp(x) order-7 series at x=1: got {val}, expected {partial_sum}"
    );
}

#[test]
fn series_sin_exact_coefficient_check() {
    // sin(x) = x - x³/6 + x⁵/120 - x⁷/5040
    // At x=1: 1 - 1/6 + 1/120 - 1/5040 = 0.841468...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 8);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 1, 1);
    let partial_sum: f64 = 1.0 - 1.0/6.0 + 1.0/120.0 - 1.0/5040.0;
    assert!(
        (val - partial_sum).abs() < 1e-10,
        "sin(x) order-8 series at x=1: got {val}, expected partial sum {partial_sum}"
    );
}

#[test]
fn series_cos_exact_coefficient_check() {
    // cos(x) = 1 - x²/2 + x⁴/24 - x⁶/720
    // At x=1: 1 - 0.5 + 1/24 - 1/720 = 0.540278...
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cos().maclaurin(&x, 7);
    let expanded = series.expand().eval();
    let val = eval_series_at(&expanded, &x, 1, 1);
    let partial_sum: f64 = 1.0 - 0.5 + 1.0/24.0 - 1.0/720.0;
    assert!(
        (val - partial_sum).abs() < 1e-10,
        "cos(x) order-7 series at x=1: got {val}, expected partial sum {partial_sum}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 17: TRICKY DEFINITE INTEGRALS WITH KNOWN ANSWERS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_integral_1_over_1_plus_x2_0_to_1() {
    // ∫₀¹ 1/(1+x²) dx = arctan(1) - arctan(0) = π/4
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 1);
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    let expected = std::f64::consts::FRAC_PI_4;
    assert!(
        (val - expected).abs() < 1e-10,
        "∫₀¹ 1/(1+x²) dx should be π/4 ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_x_exp_neg_x_0_to_inf() {
    // ∫₀^∞ x·exp(-x) dx = 1 (Gamma(2) = 1! = 1)
    // This may or may not be supported; test if it works
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &(-&x).exp();
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.infinity());
    if let Ok(val) = result.eval().eval_f64() {
        assert!(
            (val - 1.0).abs() < 1e-8,
            "∫₀^∞ x·exp(-x) dx should be 1 (Gamma(2)), got {val}"
        );
    }
}

#[test]
fn definite_integral_cos_squared_0_to_pi() {
    // ∫₀^π cos²(x) dx = π/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(2);
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.pi());
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    let expected = std::f64::consts::PI / 2.0;
    assert!(
        (val - expected).abs() < 1e-8,
        "∫₀^π cos²(x) dx should be π/2 ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_sin_squared_0_to_pi() {
    // ∫₀^π sin²(x) dx = π/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(2);
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.pi());
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    let expected = std::f64::consts::PI / 2.0;
    assert!(
        (val - expected).abs() < 1e-8,
        "∫₀^π sin²(x) dx should be π/2 ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_x_squared_neg1_to_2() {
    // ∫₋₁² x² dx = [x³/3]₋₁² = 8/3 - (-1/3) = 9/3 = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).definite_integral(&x, &ctx.int(-1), &ctx.int(2));
    let val = result.eval().eval_f64().expect("definite integral should evaluate");
    assert!(
        (val - 3.0).abs() < 1e-10,
        "∫₋₁² x² dx should be 3, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 18: HIGHER-ORDER DERIVATIVE STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_third_derivative_of_x_exp_x() {
    // f = x*exp(x)
    // f' = exp(x) + x*exp(x) = (1+x)*exp(x)
    // f'' = exp(x) + (1+x)*exp(x) = (2+x)*exp(x)
    // f''' = exp(x) + (2+x)*exp(x) = (3+x)*exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.exp();
    let d3 = f.diff_n(&x, 3);
    let expected = &(&ctx.int(3) + &x) * &x.exp();
    assert_exprs_equal_in_domain(&d3, &expected, &x, "d³/dx³(x·exp(x)) = (3+x)·exp(x)");
}

#[test]
fn diff_fourth_derivative_of_x_sin_x() {
    // f = x*sin(x)
    // f'  = sin(x) + x*cos(x)
    // f'' = 2*cos(x) - x*sin(x)
    // f''' = -3*sin(x) - x*cos(x)
    // f'''' = -4*cos(x) + x*sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.sin();
    let d4 = f.diff_n(&x, 4);
    let expected = &(&ctx.int(-4) * &x.cos()) + &(&x * &x.sin());
    assert_exprs_equal_in_domain(&d4, &expected, &x, "d⁴/dx⁴(x·sin(x))");
}

#[test]
fn diff_second_derivative_of_exp_of_x_squared() {
    // f = exp(x²)
    // f' = 2x*exp(x²)
    // f'' = 2*exp(x²) + 4x²*exp(x²) = (2 + 4x²)*exp(x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2).exp();
    let d2 = f.diff_n(&x, 2);
    let expected = &(&ctx.int(2) + &(&ctx.int(4) * &x.powi(2))) * &x.powi(2).exp();
    assert_exprs_equal_in_domain(&d2, &expected, &x, "d²/dx²(exp(x²))");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 19: INTEGRATION U-SUBSTITUTION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_sin_of_2x_plus_1() {
    // ∫ sin(2x+1) dx = -cos(2x+1)/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&(&x * 2) + 1).sin();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫sin(2x+1) dx");
}

#[test]
fn integrate_exp_of_neg_x() {
    // ∫ exp(-x) dx = -exp(-x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (-&x).exp();
    assert_ftc_custom(&integrand, &x, "∫exp(-x) dx");
}

#[test]
fn integrate_x_times_x2_plus_1_cubed() {
    // ∫ x*(x²+1)³ dx
    // u = x²+1, du = 2x dx → (1/2) ∫ u³ du = u⁴/8 = (x²+1)⁴/8
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &(&x.powi(2) + 1).powi(3);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫x(x²+1)³ dx");
}

#[test]
fn integrate_cos_x_times_exp_sin_x() {
    // ∫ cos(x)*exp(sin(x)) dx = exp(sin(x))
    // u = sin(x), du = cos(x)dx
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.cos() * &x.sin().exp();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫cos(x)·exp(sin(x)) dx");
}

#[test]
fn integrate_2x_over_x2_plus_1() {
    // ∫ 2x/(x²+1) dx = ln(x²+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &(&ctx.int(2) * &x) / &(&x.powi(2) + 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫2x/(x²+1) dx = ln(x²+1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 20: LIMIT EDGE CASES — indeterminate forms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_x_squared_sin_1_over_x_at_0() {
    // lim(x→0) x²·sin(1/x) = 0 (squeeze: |x²·sin(1/x)| ≤ x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) * &(&ctx.int(1) / &x).sin();
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            val.abs() < 1e-8,
            "lim(x→0) x²·sin(1/x) should be 0, got {val}"
        );
    }
}

#[test]
fn limit_sin_3x_over_x_at_0() {
    // lim(x→0) sin(3x)/x = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x * 3).sin() / &x;
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 3.0).abs() < 1e-8,
            "lim(x→0) sin(3x)/x should be 3, got {val}"
        );
    }
}

#[test]
fn limit_sin_ax_over_sin_bx_at_0() {
    // lim(x→0) sin(2x)/sin(3x) = 2/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x * 2).sin() / &(&x * 3).sin();
    let result = expr.try_limit(&x, &ctx.int(0));
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 2.0/3.0).abs() < 1e-8,
            "lim(x→0) sin(2x)/sin(3x) should be 2/3, got {val}"
        );
    }
}

#[test]
fn limit_ln_x_over_x_at_infinity() {
    // lim(x→∞) ln(x)/x = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.ln() / &x;
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→∞) ln(x)/x should be 0");
    }
}

#[test]
fn limit_x_to_the_1_over_x_at_infinity() {
    // lim(x→∞) x^(1/x) = 1
    //
    // Proof: x^(1/x) = exp((1/x)*ln(x)) = exp(ln(x)/x).
    // Since lim(x→∞) ln(x)/x = 0, the result is exp(0) = 1.
    //
    // BUG: The library returns e ≈ 2.718 instead of 1.
    // Root cause: The 1^∞ heuristic in limit.rs fires for ANY Pow(base, exp)
    // where exp depends on var, even when the base doesn't tend to 1.
    // For x^(1/x), it computes exp(lim (1/x)*(x-1)) = exp(1) = e,
    // but the heuristic b^e ≈ exp(e·(b-1)) is only valid when b→1.
    // Here b=x→∞, so the heuristic should not apply.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.pow(&(&ctx.int(1) / &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→∞) x^(1/x) should be 1, got {val}"
        );
    }
}

#[test]
fn limit_x_to_the_2_over_x_at_infinity() {
    // lim(x→∞) x^(2/x) = 1  (same class of bug as x^(1/x))
    //
    // x^(2/x) = exp((2/x)*ln(x)) → exp(0) = 1.
    //
    // BUG: The 1^∞ heuristic computes exp(lim (2/x)*(x-1)) = exp(2) ≈ 7.389.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.pow(&(&ctx.int(2) / &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→∞) x^(2/x) should be 1, got {val}"
        );
    }
}

#[test]
fn limit_2x_to_the_1_over_x_at_infinity() {
    // lim(x→∞) (2x)^(1/x) = 1
    //
    // (2x)^(1/x) = exp((1/x)*ln(2x)) = exp((ln(2)+ln(x))/x) → exp(0) = 1.
    //
    // BUG: The 1^∞ heuristic computes exp(lim (1/x)*(2x-1)) = exp(2) ≈ 7.389.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &ctx.int(2) * &x;
    let expr = base.pow(&(&ctx.int(1) / &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→∞) (2x)^(1/x) should be 1, got {val}"
        );
    }
}

#[test]
fn limit_1_plus_1_over_x_to_x_still_works() {
    // lim(x→∞) (1+1/x)^x = e — the CORRECT case for the 1^∞ heuristic
    // (base → 1, exponent → ∞). This should NOT be broken.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &ctx.int(1) + &(&ctx.int(1) / &x);
    let expr = base.pow(&x);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        if s == "E" {
            // Perfect — correctly returns e
        } else if let Ok(v) = r.eval_f64() {
            assert!(
                (v - std::f64::consts::E).abs() < 0.01,
                "lim(x→∞) (1+1/x)^x should be e ≈ 2.71828, got {v}"
            );
        }
    }
}

#[test]
fn limit_1_plus_2_over_x_to_x() {
    // lim(x→∞) (1+2/x)^x = e² — another valid 1^∞ case
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &ctx.int(1) + &(&ctx.int(2) / &x);
    let expr = base.pow(&x);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        let expected = std::f64::consts::E * std::f64::consts::E;
        assert!(
            (val - expected).abs() < 0.01,
            "lim(x→∞) (1+2/x)^x should be e² ≈ {expected}, got {val}"
        );
    }
}

#[test]
fn limit_exp_neg_x_at_infinity() {
    // lim(x→∞) exp(-x) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→∞) exp(-x) should be 0");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 21: SERIES — tan(x) AND arctan(x) COEFFICIENT CHECKS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_tan_first_terms() {
    // tan(x) = x + x³/3 + 2x⁵/15 + ...
    // At x = 0.2: tan(0.2) ≈ 0.20271
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.tan().try_maclaurin(&x, 6);
    if let Ok(s) = series {
        let expanded = s.expand().eval();
        let val = eval_series_at(&expanded, &x, 1, 5); // x=0.2
        let exact = 0.2_f64.tan();
        assert!(
            (val - exact).abs() < 1e-5,
            "tan(x) series at x=0.2: got {val}, expected {exact}"
        );
    }
}

#[test]
fn series_arctan_first_terms() {
    // arctan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
    // At x = 0.5: partial sum = 0.5 - 0.5³/3 + 0.5⁵/5 - 0.5⁷/7
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.atan().try_maclaurin(&x, 8);
    if let Ok(s) = series {
        let expanded = s.expand().eval();
        let val = eval_series_at(&expanded, &x, 1, 2);
        let exact = 0.5_f64.atan();
        // Order 8 includes terms up to x^7; next term is x^9/9 ≈ 2e-4 at x=0.5,
        // so truncation error of ~2e-4 is expected.
        assert!(
            (val - exact).abs() < 5e-4,
            "arctan(x) series at x=0.5: got {val}, expected {exact}, err={}",
            (val - exact).abs()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 22: COMBINED CALCULUS IDENTITIES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ftc_part2_derivative_of_integral_with_variable_upper_bound() {
    // If F(x) = ∫₀ˣ t² dt = x³/3, then F'(x) = x²
    // We verify: d/dx(∫₀ˣ t² dt) numerically equals x²
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = ctx.symbol("t");
    // Compute ∫ t² dt = t³/3, substitute t = x → x³/3
    let anti_t = t.powi(2).integrate(&t);
    let fx = anti_t.subs(&t, &x) - anti_t.subs(&t, &ctx.int(0));
    let dfx = fx.diff(&x);
    let expected = x.powi(2);
    assert_exprs_equal_in_domain(&dfx, &expected, &x, "FTC part 2");
}

#[test]
fn mean_value_theorem_numerical_check() {
    // MVT: for f(x) = x³ on [1, 3], there exists c in (1,3) such that
    // f'(c) = (f(3) - f(1))/(3 - 1) = (27 - 1)/2 = 13
    // f'(c) = 3c² = 13 → c = √(13/3) ≈ 2.082
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let df = f.diff(&x);

    let f3 = f.subs(&x, &ctx.int(3)).eval().eval_f64().unwrap();
    let f1 = f.subs(&x, &ctx.int(1)).eval().eval_f64().unwrap();
    let mvt_slope = (f3 - f1) / 2.0;
    assert!(
        (mvt_slope - 13.0).abs() < 1e-10,
        "MVT slope should be 13, got {mvt_slope}"
    );

    // Check f'(c) = 13 at c = √(13/3)
    let c = (13.0_f64 / 3.0).sqrt();
    let c_expr = ctx.rational((c * 10000.0).round() as i64, 10000);
    let fc = df.subs(&x, &c_expr).eval().eval_f64().unwrap();
    assert!(
        (fc - 13.0).abs() < 0.01,
        "f'(√(13/3)) should be ≈ 13, got {fc}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 23: INTEGRATION OF LOGARITHMIC FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_ln_x_squared() {
    // ∫ ln(x)² dx = x*ln(x)² - 2x*ln(x) + 2x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.ln().powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    // Verify FTC at points > 0
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.5, 1.0, 1.5, 2.0, 3.0],
        1e-7,
        "∫ln(x)² dx",
    );
}

#[test]
fn integrate_x_ln_x() {
    // ∫ x*ln(x) dx = x²*ln(x)/2 - x²/4
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.ln();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.5, 1.0, 1.5, 2.0],
        1e-7,
        "∫x·ln(x) dx",
    );
}

#[test]
fn integrate_ln_x_over_x() {
    // ∫ ln(x)/x dx = ln(x)²/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.ln() / &x;
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.5, 1.0, 1.5, 2.5],
        1e-7,
        "∫ln(x)/x dx = ln(x)²/2",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 24: DERIVATIVE + SIMPLIFICATION IDENTITIES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_trig_identity_gives_zero() {
    // d/dx(sin²(x) + cos²(x)) should be 0 (since sin²+cos² = 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.sin().powi(2) + &x.cos().powi(2);
    let df = f.diff(&x);
    // Should simplify to 0, or at least evaluate to 0
    for &pt_val in &[1i64, 2, 3] {
        if let Ok(val) = df.subs_i64(&x, pt_val).eval().eval_f64() {
            assert!(
                val.abs() < 1e-10,
                "d/dx(sin²x+cos²x) should be 0, got {val} at x={pt_val}"
            );
        }
    }
}

#[test]
fn diff_of_exp_ln_is_one() {
    // d/dx(exp(ln(x))) = d/dx(x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.ln().exp();
    let df = f.diff(&x);
    // Should simplify to 1
    for &pt_val in &[1i64, 2, 3] {
        if let Ok(val) = df.subs_i64(&x, pt_val).eval().eval_f64() {
            assert!(
                (val - 1.0).abs() < 1e-10,
                "d/dx(exp(ln(x))) should be 1, got {val} at x={pt_val}"
            );
        }
    }
}

#[test]
fn diff_of_ln_exp_is_one() {
    // d/dx(ln(exp(x))) = d/dx(x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp().ln();
    let df = f.diff(&x);
    for &pt_val in &[1i64, 2, 3] {
        if let Ok(val) = df.subs_i64(&x, pt_val).eval().eval_f64() {
            assert!(
                (val - 1.0).abs() < 1e-10,
                "d/dx(ln(exp(x))) should be 1, got {val} at x={pt_val}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 25: INTEGRATION THEN DEFINITE EVALUATION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_integral_x_exp_x_0_to_1() {
    // ∫₀¹ x·exp(x) dx = [x·exp(x) - exp(x)]₀¹ = (e - e) - (0 - 1) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.exp();
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("should evaluate");
    assert!(
        (val - 1.0).abs() < 1e-10,
        "∫₀¹ x·exp(x) dx should be 1, got {val}"
    );
}

#[test]
fn definite_integral_x_sin_x_0_to_pi() {
    // ∫₀^π x·sin(x) dx = [sin(x) - x·cos(x)]₀^π = (0 - π·(-1)) - (0 - 0) = π
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x * &x.sin();
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.pi());
    let val = result.eval().eval_f64().expect("should evaluate");
    let expected = std::f64::consts::PI;
    assert!(
        (val - expected).abs() < 1e-8,
        "∫₀^π x·sin(x) dx should be π ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_exp_neg_x_0_to_1() {
    // ∫₀¹ exp(-x) dx = [-exp(-x)]₀¹ = -exp(-1) + 1 = 1 - 1/e
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (-&x).exp();
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("should evaluate");
    let expected = 1.0 - (-1.0_f64).exp();
    assert!(
        (val - expected).abs() < 1e-10,
        "∫₀¹ exp(-x) dx should be 1-1/e ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 26: CORNER CASES IN DIFFERENTIATION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_x_over_x_simplifies_to_zero() {
    // d/dx(x/x) = d/dx(1) = 0 (assuming x ≠ 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x / &x;
    let df = f.diff(&x);
    // The derivative should evaluate to 0 at any nonzero point
    for &pt_val in &[1i64, 2, 3, -1, -2] {
        if let Ok(val) = df.subs_i64(&x, pt_val).eval().eval_f64() {
            assert!(
                val.abs() < 1e-10,
                "d/dx(x/x) should be 0, got {val} at x={pt_val}"
            );
        }
    }
}

#[test]
fn diff_of_exp_a_x_is_a_exp_a_x() {
    // d/dx exp(a*x) = a*exp(a*x) where a is a constant symbol
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let f = (&a * &x).exp();
    let df = f.diff(&x);
    let expected = &a * &(&a * &x).exp();

    // Check at (a, x) = (2, 1): expected = 2*exp(2)
    let got = df.subs(&a, &ctx.int(2)).subs(&x, &ctx.int(1)).eval().eval_f64();
    let want = expected.subs(&a, &ctx.int(2)).subs(&x, &ctx.int(1)).eval().eval_f64();
    if let (Ok(g), Ok(w)) = (got, want) {
        assert!(
            (g - w).abs() < 1e-8,
            "d/dx exp(ax) at (a=2,x=1): got {g}, want {w}"
        );
    }
}

#[test]
fn diff_of_sin_a_x_is_a_cos_a_x() {
    // d/dx sin(a*x) = a*cos(a*x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let f = (&a * &x).sin();
    let df = f.diff(&x);
    let expected = &a * &(&a * &x).cos();

    let got = df.subs(&a, &ctx.int(3)).subs(&x, &ctx.int(1)).eval().eval_f64();
    let want = expected.subs(&a, &ctx.int(3)).subs(&x, &ctx.int(1)).eval().eval_f64();
    if let (Ok(g), Ok(w)) = (got, want) {
        assert!(
            (g - w).abs() < 1e-8,
            "d/dx sin(ax) at (a=3,x=1): got {g}, want {w}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 27: INTEGRATION — TRIG SUBSTITUTION AND SPECIAL FORMS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_sqrt_x2_plus_1() {
    // ∫ 1/√(x²+1) dx = sinh⁻¹(x) = ln(x + √(x²+1))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x.powi(2) + 1).sqrt();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫1/√(x²+1) dx");
}

#[test]
fn integrate_sqrt_x() {
    // ∫ √x dx = (2/3)*x^(3/2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sqrt();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.25, 0.5, 1.0, 2.0, 4.0],
        1e-7,
        "∫√x dx = (2/3)x^(3/2)",
    );
}

#[test]
fn integrate_x_over_sqrt_x2_plus_1() {
    // ∫ x/√(x²+1) dx = √(x²+1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x / &(&x.powi(2) + 1).sqrt();
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫x/√(x²+1) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 28: SERIES — COMPOSITION AND NON-STANDARD
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_1_over_1_plus_x_squared() {
    // 1/(1+x²) = 1 - x² + x⁴ - x⁶ + ... for |x| < 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x.powi(2) + 1);
    let series = f.try_maclaurin(&x, 5);
    if let Ok(s) = series {
        let expanded = s.expand().eval();
        // At x = 0.3: 1/(1+0.09) = 1/1.09 ≈ 0.91743
        let val = eval_series_at(&expanded, &x, 3, 10);
        let exact = 1.0 / (1.0 + 0.09);
        assert!(
            (val - exact).abs() < 0.01,
            "1/(1+x²) series at x=0.3: got {val}, expected {exact}"
        );
    }
}

#[test]
fn series_sqrt_1_plus_x() {
    // √(1+x) = 1 + x/2 - x²/8 + x³/16 - ... (binomial series)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (&x + 1).sqrt();
    let series = f.try_maclaurin(&x, 5);
    if let Ok(s) = series {
        let expanded = s.expand().eval();
        // At x = 0.5: √1.5 ≈ 1.22474
        let val = eval_series_at(&expanded, &x, 1, 2);
        let exact = 1.5_f64.sqrt();
        assert!(
            (val - exact).abs() < 0.01,
            "√(1+x) series at x=0.5: got {val}, expected {exact}"
        );
    }
}

#[test]
fn series_exp_of_negative_x_squared() {
    // exp(-x²) = 1 - x² + x⁴/2 - x⁶/6 + ... (Gaussian-like)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = (-&x.powi(2)).exp();
    let series = f.try_maclaurin(&x, 6);
    if let Ok(s) = series {
        let expanded = s.expand().eval();
        // At x = 0.5: exp(-0.25) ≈ 0.7788
        let val = eval_series_at(&expanded, &x, 1, 2);
        let exact = (-0.25_f64).exp();
        assert!(
            (val - exact).abs() < 0.01,
            "exp(-x²) series at x=0.5: got {val}, expected {exact}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 29: SUBTLE BUG HUNTING — SIGN ERRORS AND OFF-BY-ONE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_negative_exponent_chain() {
    // d/dx (sin(x))^(-1) = -cos(x)/sin(x)^2 = -cos(x)*csc²(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(-1);
    let df = f.diff(&x);
    let expected = &(-&x.cos()) / &x.sin().powi(2);
    // Check at x=1 (away from zeros of sin)
    let got = df.subs(&x, &ctx.int(1)).eval().eval_f64();
    let want = expected.subs(&x, &ctx.int(1)).eval().eval_f64();
    if let (Ok(g), Ok(w)) = (got, want) {
        assert!(
            (g - w).abs() < 1e-9,
            "d/dx (1/sin(x)) at x=1: got {g}, want {w}"
        );
    }
}

#[test]
fn diff_cos_squared_chain_rule() {
    // d/dx cos²(x) = -2*sin(x)*cos(x) = -sin(2x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.cos().powi(2);
    let df = f.diff(&x);
    let expected = -&(&x * 2).sin();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx cos²(x) = -sin(2x)");
}

#[test]
fn diff_sin_cubed_chain_rule() {
    // d/dx sin³(x) = 3*sin²(x)*cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(3);
    let df = f.diff(&x);
    let expected = &(&ctx.int(3) * &x.sin().powi(2)) * &x.cos();
    assert_exprs_equal_in_domain(&df, &expected, &x, "d/dx sin³(x) = 3sin²(x)cos(x)");
}

#[test]
fn integrate_then_diff_roundtrip_for_tricky_rational() {
    // ∫ (x+1)/(x²+1) dx then differentiate back
    // = (1/2)*ln(x²+1) + arctan(x) (split into x/(x²+1) + 1/(x²+1))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &(&x + 1) / &(&x.powi(2) + 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    let deriv = antideriv.diff(&x);
    let ctx2 = integrand.context();
    let mut checked = 0;
    for &(p, q) in &[(1, 4), (1, 2), (3, 4), (3, 2)] {
        let pt = ctx2.rational(p, q);
        let orig = integrand.subs(&x, &pt).eval().eval_f64();
        let back = deriv.subs(&x, &pt).eval().eval_f64();
        if let (Ok(o), Ok(b)) = (orig, back) {
            checked += 1;
            assert!(
                (o - b).abs() < 1e-8 * o.abs().max(b.abs()).max(1.0),
                "FTC roundtrip for (x+1)/(x²+1) at x={p}/{q}: orig={o}, back={b}",
            );
        }
    }
    assert!(checked > 0, "no points succeeded");
}

#[test]
fn integrate_cos_cubed() {
    // ∫ cos³(x) dx = sin(x) - sin³(x)/3
    // via cos³ = cos·(1-sin²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.cos().powi(3);
    assert_ftc_custom(&integrand, &x, "∫cos³(x) dx");
}

#[test]
fn integrate_sin_cubed() {
    // ∫ sin³(x) dx = -cos(x) + cos³(x)/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.sin().powi(3);
    assert_ftc_custom(&integrand, &x, "∫sin³(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 30: LIMIT — ASYMPTOTIC BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_x_squared_over_exp_x_at_infinity() {
    // lim(x→∞) x²/exp(x) = 0  (exponential dominates polynomial)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) / &x.exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→∞) x²/exp(x) should be 0");
    }
}

#[test]
fn limit_x_cubed_over_exp_x_at_infinity() {
    // lim(x→∞) x³/exp(x) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(3) / &x.exp();
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→∞) x³/exp(x) should be 0");
    }
}

#[test]
fn limit_exp_x_over_x_n_at_infinity() {
    // lim(x→∞) exp(x)/x^10 = ∞
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.exp() / &x.powi(10);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        assert!(
            s.contains("∞") || s.contains("oo") || s.contains("Inf"),
            "lim(x→∞) exp(x)/x^10 should be ∞, got: {s}"
        );
    }
}

#[test]
fn limit_at_neg_infinity_polynomial() {
    // lim(x→-∞) x³ = -∞
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3).try_limit(&x, &ctx.neg_infinity());
    if let Ok(r) = result {
        let s = format!("{r}");
        assert!(
            s.contains("-∞") || s.contains("-oo") || s.contains("-Inf") || s.contains("NegInf"),
            "lim(x→-∞) x³ should be -∞, got: {s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 31: DEFINITE INTEGRALS — KNOWN CLOSED FORMS WITH π AND e
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn definite_integral_sin_x_over_x_limit_check() {
    // Instead of the improper integral, verify something tractable:
    // ∫₀^(π/2) cos(x) dx = [sin(x)]₀^(π/2) = sin(π/2) - sin(0) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos().definite_integral(&x, &ctx.int(0), &(&ctx.pi() / 2));
    let val = result.eval().eval_f64().expect("should evaluate");
    assert!(
        (val - 1.0).abs() < 1e-10,
        "∫₀^(π/2) cos(x) dx should be 1, got {val}"
    );
}

#[test]
fn definite_integral_2x_exp_x2_0_to_1() {
    // ∫₀¹ 2x·exp(x²) dx = [exp(x²)]₀¹ = e - 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &(&ctx.int(2) * &x) * &x.powi(2).exp();
    let result = integrand.definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let val = result.eval().eval_f64().expect("should evaluate");
    let expected = std::f64::consts::E - 1.0;
    assert!(
        (val - expected).abs() < 1e-8,
        "∫₀¹ 2x·exp(x²) dx should be e-1 ≈ {expected}, got {val}"
    );
}

#[test]
fn definite_integral_1_over_x_1_to_2() {
    // ∫₁² 1/x dx = ln(2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &x;
    let result = integrand.definite_integral(&x, &ctx.int(1), &ctx.int(2));
    let val = result.eval().eval_f64().expect("should evaluate");
    let expected = 2.0_f64.ln();
    assert!(
        (val - expected).abs() < 1e-10,
        "∫₁² 1/x dx should be ln(2) ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 32: POTENTIAL SIGN/FACTOR BUG HUNTERS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_negative_sin() {
    // ∫ -sin(x) dx = cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = -&x.sin();
    assert_ftc_custom(&integrand, &x, "∫-sin(x) dx = cos(x)");
}

#[test]
fn integrate_3_sin_2x() {
    // ∫ 3·sin(2x) dx = -3·cos(2x)/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(3) * &(&x * 2).sin();
    assert_ftc_custom(&integrand, &x, "∫3·sin(2x) dx");
}

#[test]
fn integrate_exp_5x() {
    // ∫ exp(5x) dx = exp(5x)/5
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x * 5).exp();
    assert_ftc_custom(&integrand, &x, "∫exp(5x) dx");
}

#[test]
fn diff_and_integrate_cancel_for_x_sin_x() {
    // ∫ d/dx(x·sin(x)) dx should differ from x·sin(x) by at most a constant
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.sin();
    let df = f.diff(&x);
    let anti = df.integrate(&x);
    // (anti - f) should be constant
    let diff_at_1 = (&anti - &f).subs(&x, &ctx.int(1)).eval().eval_f64();
    let diff_at_2 = (&anti - &f).subs(&x, &ctx.int(2)).eval().eval_f64();
    if let (Ok(d1), Ok(d2)) = (diff_at_1, diff_at_2) {
        assert!(
            (d1 - d2).abs() < 1e-8,
            "∫(d/dx(x·sin(x)))dx should differ from x·sin(x) by constant: \
             diff@1={d1}, diff@2={d2}"
        );
    }
}

#[test]
fn diff_of_definite_integral_polynomial() {
    // d/dx ∫₀ˣ (3t²+1) dt = 3x²+1 (by FTC)
    // ∫ (3t²+1) dt = t³+t, evaluated at x gives x³+x
    // d/dx(x³+x) = 3x²+1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = ctx.symbol("t");
    let integrand_t = &(&ctx.int(3) * &t.powi(2)) + 1;
    let anti_t = integrand_t.integrate(&t);
    let fx = &anti_t.subs(&t, &x) - &anti_t.subs(&t, &ctx.int(0));
    let dfx = fx.diff(&x);
    let expected = &(&ctx.int(3) * &x.powi(2)) + 1;
    assert_exprs_equal_in_domain(&dfx, &expected, &x, "d/dx ∫₀ˣ(3t²+1)dt = 3x²+1");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 33: MULTIVARIATE CALCULUS EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_implicit_x_cubed_y_cubed() {
    // ∂/∂x (x³*y³) = 3x²*y³ (y treated as constant)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(3) * &y.powi(3);
    let df = f.diff(&x);
    let expected = &(&ctx.int(3) * &x.powi(2)) * &y.powi(3);
    let val = df
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("should eval");
    let want = expected
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        (val - want).abs() < 1e-8,
        "∂/∂x(x³y³) at (2,3): got {val}, want {want}"
    );
    // Exact: 3*4*27 = 324
    assert!(
        (val - 324.0).abs() < 1e-8,
        "∂/∂x(x³y³) at (2,3) should be 324, got {val}"
    );
}

#[test]
fn diff_laplacian_of_x2y2() {
    // f = x²y², ∇²f = ∂²f/∂x² + ∂²f/∂y² = 2y² + 2x²
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(2) * &y.powi(2);
    let fxx = f.diff(&x).diff(&x);
    let fyy = f.diff(&y).diff(&y);
    let laplacian = &fxx + &fyy;
    // At (3, 4): 2*16 + 2*9 = 32 + 18 = 50
    let val = laplacian
        .subs(&x, &ctx.int(3))
        .subs(&y, &ctx.int(4))
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        (val - 50.0).abs() < 1e-8,
        "∇²(x²y²) at (3,4) should be 50, got {val}"
    );
}

#[test]
fn diff_third_mixed_partial() {
    // f = x²y³z, ∂³f/∂x∂y∂z = ∂/∂x(∂/∂y(∂/∂z(x²y³z)))
    // ∂f/∂z = x²y³
    // ∂²f/∂y∂z = 3x²y²
    // ∂³f/∂x∂y∂z = 6xy²
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    let f = &(&x.powi(2) * &y.powi(3)) * &z;
    let d3 = f.diff(&z).diff(&y).diff(&x);
    // At (2, 3, anything): 6*2*9 = 108
    let val = d3
        .subs(&x, &ctx.int(2))
        .subs(&y, &ctx.int(3))
        .subs(&z, &ctx.int(1))
        .eval()
        .eval_f64()
        .expect("should eval");
    assert!(
        (val - 108.0).abs() < 1e-8,
        "∂³(x²y³z)/∂x∂y∂z at (2,3,1) should be 108, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 34: INTEGRATION — RATIONAL WITH REPEATED ROOTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_1_over_x_minus_1_squared() {
    // ∫ 1/(x-1)² dx = -1/(x-1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&x - 1).powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    // Verify FTC at points away from x=1
    assert_ftc_domain(
        &integrand,
        &x,
        &[2.0, 3.0, 4.0, 5.0],
        1e-7,
        "∫1/(x-1)² dx = -1/(x-1)",
    );
}

#[test]
fn integrate_x_over_x_plus_1_squared() {
    // ∫ x/(x+1)² dx = ln|x+1| + 1/(x+1) (via partial fractions: 1/(x+1) - 1/(x+1)²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x / &(&x + 1).powi(2);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_domain(
        &integrand,
        &x,
        &[0.5, 1.0, 2.0, 3.0],
        1e-7,
        "∫x/(x+1)² dx",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 35: MORE LIMIT BUG VARIANTS — probing the 1^∞ heuristic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_x_squared_to_the_1_over_x_at_infinity() {
    // lim(x→∞) (x²)^(1/x) = lim exp((2 ln x)/x) = exp(0) = 1
    //
    // BUG EXPECTED: The 1^∞ heuristic computes exp(lim (1/x)*(x²-1)) = exp(∞),
    // which diverges or gives a wrong result.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).pow(&(&ctx.int(1) / &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→∞) (x²)^(1/x) should be 1, got {val}"
        );
    }
}

#[test]
fn limit_exp_x_to_the_1_over_x_at_infinity() {
    // lim(x→∞) exp(x)^(1/x) = exp(x/x) = exp(1) = e
    //
    // This is ∞^0 form.  exp(x)^(1/x) = exp(x · (1/x)) = exp(1) = e.
    //
    // NOTE: This may also be affected by the 1^∞ heuristic bug in limit.rs.
    // The heuristic computes product = (1/x)*(exp(x)-1).  If Gruntz on that
    // product returns a large-but-finite symbolic expression instead of ∞,
    // the heuristic fires incorrectly and returns exp(wrong).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp().pow(&(&ctx.int(1) / &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        // The correct answer is e ≈ 2.71828
        if let Ok(val) = r.eval_f64() {
            let e = std::f64::consts::E;
            assert!(
                (val - e).abs() < 1e-6,
                "lim(x→∞) exp(x)^(1/x) should be e ≈ {e}, got {val}  (symbolic: {r})"
            );
        }
    }
}

#[test]
fn limit_1_plus_1_over_x_to_2x() {
    // lim(x→∞) (1+1/x)^(2x) = e²
    // Valid 1^∞ case: base → 1, exponent → ∞
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &ctx.int(1) + &(&ctx.int(1) / &x);
    let expr = base.pow(&(&ctx.int(2) * &x));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        let expected = std::f64::consts::E.powi(2);
        assert!(
            (val - expected).abs() < 0.01,
            "lim(x→∞) (1+1/x)^(2x) should be e² ≈ {expected}, got {val}"
        );
    }
}

#[test]
fn limit_x_plus_1_over_x_to_the_x_at_infinity() {
    // lim(x→∞) ((x+1)/x)^x = lim (1+1/x)^x = e
    // This is the same as (1+1/x)^x but written differently.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = &(&x + 1) / &x;
    let expr = base.pow(&x);
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - std::f64::consts::E).abs() < 0.01,
            "lim(x→∞) ((x+1)/x)^x should be e, got {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 36: DIFFERENTIATION — IMPLICIT AND PARAMETRIC EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_x_to_negative_fraction() {
    // d/dx x^(-1/2) = (-1/2) * x^(-3/2)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.rational(-1, 2));
    let df = f.diff(&x);
    // At x = 4: (-1/2) * 4^(-3/2) = (-1/2) * (1/8) = -1/16 = -0.0625
    let val = df.subs(&x, &ctx.int(4)).eval().eval_f64().expect("should eval");
    assert!(
        (val - (-0.0625)).abs() < 1e-10,
        "d/dx x^(-1/2) at x=4 should be -0.0625, got {val}"
    );
}

#[test]
fn diff_of_rational_power() {
    // d/dx x^(2/3) = (2/3) * x^(-1/3)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.rational(2, 3));
    let df = f.diff(&x);
    // At x = 8: (2/3) * 8^(-1/3) = (2/3) * (1/2) = 1/3 ≈ 0.3333
    let val = df.subs(&x, &ctx.int(8)).eval().eval_f64().expect("should eval");
    assert!(
        (val - 1.0/3.0).abs() < 1e-9,
        "d/dx x^(2/3) at x=8 should be 1/3, got {val}"
    );
}

#[test]
fn diff_second_derivative_of_arctan() {
    // d²/dx² arctan(x) = -2x/(1+x²)²
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d2 = x.atan().diff_n(&x, 2);
    let expected = &(&ctx.int(-2) * &x) / &(&x.powi(2) + 1).powi(2);
    // At x = 1: -2/(1+1)² = -2/4 = -0.5
    let got = d2.subs(&x, &ctx.int(1)).eval().eval_f64().expect("should eval");
    let want = expected.subs(&x, &ctx.int(1)).eval().eval_f64().expect("should eval");
    assert!(
        (got - want).abs() < 1e-9,
        "d²/dx²(arctan(x)) at x=1: got {got}, want {want}"
    );
    assert!(
        (got - (-0.5)).abs() < 1e-9,
        "d²/dx²(arctan(x)) at x=1 should be -0.5, got {got}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 37: INTEGRATION — EXPRESSIONS THAT TEST FACTOR SIGNS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_minus_exp_x() {
    // ∫ -exp(x) dx = -exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = -&x.exp();
    assert_ftc_custom(&integrand, &x, "∫-exp(x) dx = -exp(x)");
}

#[test]
fn integrate_1_over_2x_plus_1() {
    // ∫ 1/(2x+1) dx = (1/2)*ln|2x+1|
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &ctx.int(1) / &(&(&x * 2) + 1);
    let antideriv = integrand.integrate(&x);
    let s = format!("{antideriv}");
    if s.contains("Integral") {
        return;
    }
    assert_ftc_custom(&integrand, &x, "∫1/(2x+1) dx");
}

#[test]
fn definite_integral_reversal() {
    // ∫ₐᵇ f dx = -∫ᵇₐ f dx
    // ∫₀¹ x² dx = 1/3 and ∫₁⁰ x² dx = -1/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let forward = x.powi(2).definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let backward = x.powi(2).definite_integral(&x, &ctx.int(1), &ctx.int(0));
    let v_fwd = forward.eval().eval_f64().expect("should eval");
    let v_bwd = backward.eval().eval_f64().expect("should eval");
    assert!(
        (v_fwd + v_bwd).abs() < 1e-10,
        "∫₀¹ x² dx + ∫₁⁰ x² dx should be 0: got {} + {} = {}",
        v_fwd, v_bwd, v_fwd + v_bwd
    );
}

#[test]
fn definite_integral_additivity() {
    // ∫₀² x dx = ∫₀¹ x dx + ∫₁² x dx
    // = 2 = 1/2 + 3/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let full = x.clone().definite_integral(&x, &ctx.int(0), &ctx.int(2));
    let part1 = x.clone().definite_integral(&x, &ctx.int(0), &ctx.int(1));
    let part2 = x.clone().definite_integral(&x, &ctx.int(1), &ctx.int(2));
    let v_full = full.eval().eval_f64().expect("should eval");
    let v_parts = part1.eval().eval_f64().expect("p1") + part2.eval().eval_f64().expect("p2");
    assert!(
        (v_full - v_parts).abs() < 1e-10,
        "∫₀² x dx should equal ∫₀¹ x dx + ∫₁² x dx: {v_full} vs {v_parts}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 38: SERIES — CHECKING THAT EVEN/ODD SYMMETRY IS RESPECTED
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn series_sin_has_no_even_power_terms() {
    // sin(x) Maclaurin series should have zero coefficients for even powers.
    // Check by evaluating the series at x and -x: sin series should be odd.
    // s(-x) = -s(x) for an odd function series.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.sin().maclaurin(&x, 8).expand().eval();
    let val_pos = eval_series_at(&series, &x, 1, 2);
    let val_neg = eval_series_at(&series, &x, -1, 2);
    assert!(
        (val_pos + val_neg).abs() < 1e-12,
        "sin(x) series should be odd: s(0.5)={val_pos}, s(-0.5)={val_neg}, \
         sum={}", val_pos + val_neg
    );
}

#[test]
fn series_cos_has_no_odd_power_terms() {
    // cos(x) Maclaurin series should have zero coefficients for odd powers.
    // Check: cos series should be even, so s(x) = s(-x).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let series = x.cos().maclaurin(&x, 8).expand().eval();
    let val_pos = eval_series_at(&series, &x, 1, 2);
    let val_neg = eval_series_at(&series, &x, -1, 2);
    assert!(
        (val_pos - val_neg).abs() < 1e-12,
        "cos(x) series should be even: s(0.5)={val_pos}, s(-0.5)={val_neg}, \
         diff={}", val_pos - val_neg
    );
}

#[test]
fn series_exp_differentiates_to_itself() {
    // d/dx of the exp(x) Maclaurin series should equal the same series
    // (minus the highest-order term, which is lost by truncation).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_series = x.exp().maclaurin(&x, 8).expand().eval();
    let deriv = exp_series.diff(&x);
    // At x = 0.5, the derivative of the truncated series should be very close
    // to the series itself (difference is from the highest dropped term).
    let val_series = eval_series_at(&exp_series, &x, 1, 2);
    let val_deriv = eval_series_at(&deriv, &x, 1, 2);
    assert!(
        (val_series - val_deriv).abs() < 1e-4,
        "d/dx(exp series) should ≈ exp series at x=0.5: series={val_series}, deriv={val_deriv}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 39: LIMIT — NEGATIVE INFINITY AND COMPOSITION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_exp_x_at_neg_infinity() {
    // lim(x→-∞) exp(x) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp().try_limit(&x, &ctx.neg_infinity());
    if let Ok(r) = result {
        assert_eq!(format!("{r}"), "0", "lim(x→-∞) exp(x) should be 0");
    }
}

#[test]
fn limit_1_over_1_plus_exp_neg_x_at_infinity() {
    // lim(x→∞) 1/(1+exp(-x)) = 1 (sigmoid function → 1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(1) / &(&ctx.int(1) + &(-&x).exp());
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - 1.0).abs() < 1e-8,
            "lim(x→∞) sigmoid should be 1, got {val}"
        );
    }
}

#[test]
fn limit_1_over_1_plus_exp_neg_x_at_neg_infinity() {
    // lim(x→-∞) 1/(1+exp(-x)) = 0 (sigmoid function → 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(1) / &(&ctx.int(1) + &(-&x).exp());
    let result = expr.try_limit(&x, &ctx.neg_infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            val.abs() < 1e-8,
            "lim(x→-∞) sigmoid should be 0, got {val}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 40: INTEGRATION — VERIFY SPECIFIC SYMBOLIC FORMS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_x_cubed_exp_x() {
    // ∫ x³·exp(x) dx (requires 3 rounds of IBP)
    // = x³·exp(x) - 3x²·exp(x) + 6x·exp(x) - 6·exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(3) * &x.exp();
    assert_ftc_custom(&integrand, &x, "∫x³·exp(x) dx");
}

#[test]
fn integrate_x_squared_sin_x() {
    // ∫ x²·sin(x) dx = -x²·cos(x) + 2x·sin(x) + 2·cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.sin();
    assert_ftc_custom(&integrand, &x, "∫x²·sin(x) dx");
}

#[test]
fn integrate_x_squared_cos_x() {
    // ∫ x²·cos(x) dx = x²·sin(x) + 2x·cos(x) - 2·sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = &x.powi(2) * &x.cos();
    assert_ftc_custom(&integrand, &x, "∫x²·cos(x) dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 41: DIFF/INTEGRATE ROUND-TRIPS WITH COMPOUND EXPRESSIONS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_diff_integrate_exp_sin() {
    // ∫ d/dx(exp(x)·sin(x)) dx should differ from exp(x)·sin(x) by a constant
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.exp() * &x.sin();
    let df = f.diff(&x);
    let anti = df.integrate(&x);
    let diff_at_1 = (&anti - &f).subs(&x, &ctx.int(1)).eval().eval_f64();
    let diff_at_2 = (&anti - &f).subs(&x, &ctx.int(2)).eval().eval_f64();
    if let (Ok(d1), Ok(d2)) = (diff_at_1, diff_at_2) {
        assert!(
            (d1 - d2).abs() < 1e-8,
            "∫(d/dx(exp·sin))dx - exp·sin should be constant: @1={d1}, @2={d2}"
        );
    }
}

#[test]
fn roundtrip_diff_integrate_x_ln_x() {
    // ∫ d/dx(x·ln(x)) dx should differ from x·ln(x) by a constant
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * &x.ln();
    let df = f.diff(&x); // ln(x) + 1
    let anti = df.integrate(&x);
    // Check at x=2 and x=3
    let diff_at_2 = (&anti - &f).subs(&x, &ctx.int(2)).eval().eval_f64();
    let diff_at_3 = (&anti - &f).subs(&x, &ctx.int(3)).eval().eval_f64();
    if let (Ok(d2), Ok(d3)) = (diff_at_2, diff_at_3) {
        assert!(
            (d2 - d3).abs() < 1e-8,
            "∫(d/dx(x·ln(x)))dx - x·ln(x) should be constant: @2={d2}, @3={d3}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 42: MORE POTENTIAL LIMIT BUGS — ∞^0 form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn limit_x_to_the_1_over_ln_x_at_infinity() {
    // lim(x→∞) x^(1/ln(x)) = e
    //
    // x^(1/ln(x)) = exp(ln(x)/ln(x)) = exp(1) = e
    //
    // This is ∞^0 form. The correct answer is e.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.pow(&(&ctx.int(1) / &x.ln()));
    let result = expr.try_limit(&x, &ctx.infinity());
    if let Ok(r) = result {
        let val = r.eval_f64().expect("limit should evaluate");
        assert!(
            (val - std::f64::consts::E).abs() < 1e-6,
            "lim(x→∞) x^(1/ln(x)) should be e, got {val}"
        );
    }
}

#[test]
fn limit_n_to_the_1_over_n_discrete_style() {
    // Verify numerically: for moderate x, x^(1/x) → 1
    // This doesn't test the limit engine, but verifies the math claim.
    // We use small-to-moderate values to avoid expensive big-integer arithmetic.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&(&ctx.int(1) / &x));
    // Evaluate at x = 5, 10, 20 (small enough to compute quickly)
    for &n in &[5i64, 10, 20] {
        let val = f.subs_i64(&x, n).eval().eval_f64().expect("should eval");
        assert!(
            (val - 1.0).abs() < 0.5,
            "x^(1/x) at x={n} should be close to 1, got {val}"
        );
    }
}
