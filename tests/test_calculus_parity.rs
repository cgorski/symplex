//! Calculus feature-parity tests: trig identity integration, ln(x)² by-parts,
//! and ODE improvements (trig homogeneous form, non-homogeneous, integrating factor).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify FTC numerically: d/dx(∫ f dx) ≈ f at a test point.
/// Also asserts the antiderivative is not unevaluated.
fn assert_ftc(integrand: &Ex, var: &Ex, label: &str) {
    let anti = integrand.integrate(var);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral"),
        "{label}: integration returned unevaluated: {s}"
    );

    let deriv = anti.diff(var);
    let test_point = symplex::rational(7, 10);
    if let (Ok(o), Ok(d)) = (
        integrand.subs(var, &test_point).eval_f64(),
        deriv.subs(var, &test_point).eval_f64(),
    ) && o.is_finite() && d.is_finite()
    {
        let diff = (o - d).abs();
        let tol = 1e-7 * o.abs().max(1.0);
        assert!(
            diff < tol,
            "{label}: FTC failed — integrand={o:.12}, deriv={d:.12}, diff={diff:.2e}",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trig identity integrals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_tan_squared() {
    // ∫ tan²(x) dx = tan(x) − x  (via sec²(x) − 1)
    let x = symplex::var("x");
    let integrand = x.tan().powi(2);
    assert_ftc(&integrand, &x, "∫tan²(x)dx");
}

#[test]
fn integrate_tan_squared_numeric() {
    // Verify numerically at x = 0.7
    let x = symplex::var("x");
    let anti = x.tan().powi(2).integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "tan² should be integrated: {s}");

    // d/dx[tan(x) - x] = sec²(x) - 1 = tan²(x)
    let deriv = anti.diff(&x);
    let pt = symplex::rational(7, 10);
    let integrand_val = x.tan().powi(2).subs(&x, &pt).eval_f64().unwrap();
    let deriv_val = deriv.subs(&x, &pt).eval_f64().unwrap();
    let err = (integrand_val - deriv_val).abs();
    assert!(err < 1e-8, "FTC for tan²: {integrand_val} vs {deriv_val}");
}

#[test]
fn integrate_sech_squared() {
    // ∫ sech²(x) dx = ∫ cosh(x)^(-2) dx = tanh(x)
    let x = symplex::var("x");
    let integrand = x.cosh().powi(-2);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "sech² should be integrated: {s}");

    // Numerical check
    let pt = symplex::rational(7, 10);
    let anti_val = anti.subs(&x, &pt).eval_f64().unwrap();
    let tanh_val = (0.7_f64).tanh();
    assert!(
        (anti_val - tanh_val).abs() < 1e-10,
        "∫sech²(x)dx at x=0.7: got {anti_val}, expected tanh(0.7)={tanh_val}"
    );
}

#[test]
fn integrate_sech_squared_ftc() {
    let x = symplex::var("x");
    let integrand = x.cosh().powi(-2);
    assert_ftc(&integrand, &x, "∫sech²(x)dx");
}

#[test]
fn integrate_sinh_squared() {
    // ∫ sinh²(x) dx = sinh(2x)/4 − x/2
    let x = symplex::var("x");
    let integrand = x.sinh().powi(2);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "sinh² should be integrated: {s}");

    assert_ftc(&integrand, &x, "∫sinh²(x)dx");
}

#[test]
fn integrate_sinh_squared_numeric() {
    let x = symplex::var("x");
    let anti = x.sinh().powi(2).integrate(&x);

    // At x = 0.7: expected = sinh(1.4)/4 - 0.7/2
    let val = anti.subs(&x, &symplex::rational(7, 10)).eval_f64().unwrap();
    let expected = (1.4_f64).sinh() / 4.0 - 0.7 / 2.0;
    assert!(
        (val - expected).abs() < 1e-10,
        "∫sinh²(x)dx at x=0.7: got {val}, expected {expected}"
    );
}

#[test]
fn integrate_cosh_squared() {
    // ∫ cosh²(x) dx = sinh(2x)/4 + x/2
    let x = symplex::var("x");
    let integrand = x.cosh().powi(2);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "cosh² should be integrated: {s}");

    assert_ftc(&integrand, &x, "∫cosh²(x)dx");
}

#[test]
fn integrate_tanh_squared() {
    // ∫ tanh²(x) dx = x − tanh(x) (via 1 − sech²)
    let x = symplex::var("x");
    let integrand = x.tanh().powi(2);
    assert_ftc(&integrand, &x, "∫tanh²(x)dx");
}

// ═══════════════════════════════════════════════════════════════════════════
// ln(x)^n by parts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_ln_x_squared() {
    // ∫ ln(x)² dx = x·ln(x)² − 2x·ln(x) + 2x
    let x = symplex::var("x");
    let integrand = x.ln().powi(2);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "ln(x)² should be integrated: {s}");

    // FTC: d/dx(result) should equal ln(x)²
    let deriv = anti.diff(&x);
    // Test at x = 2
    let pt = symplex::int(2);
    let orig_val = integrand.subs(&x, &pt).eval_f64().unwrap();
    let deriv_val = deriv.subs(&x, &pt).eval_f64().unwrap();
    let err = (orig_val - deriv_val).abs();
    assert!(
        err < 1e-8,
        "FTC for ∫ln(x)²dx: integrand={orig_val}, deriv={deriv_val}, err={err}"
    );
}

#[test]
fn integrate_ln_x_squared_numeric() {
    let x = symplex::var("x");
    let anti = x.ln().powi(2).integrate(&x);

    // At x=e: ln(e)² = 1, antideriv = e·1 − 2e·1 + 2e = e
    // Actually: x·ln(x)² − 2(x·ln(x) − x) = x·ln²(x) − 2x·ln(x) + 2x
    // At x=e: e·1 − 2e + 2e = e
    let e_val = std::f64::consts::E;
    let e_expr = symplex::var("__e_placeholder");
    // We use a numerical approach: substitute x=2 and check
    let val = anti.subs(&x, &symplex::int(2)).eval_f64().unwrap();
    let ln2 = 2.0_f64.ln();
    let expected = 2.0 * ln2 * ln2 - 2.0 * 2.0 * ln2 + 2.0 * 2.0;
    let _ = e_val;
    let _ = e_expr;
    assert!(
        (val - expected).abs() < 1e-8,
        "∫ln(x)²dx at x=2: got {val}, expected {expected}"
    );
}

#[test]
fn integrate_ln_x_cubed() {
    // ∫ ln(x)³ dx should also work by recursive by-parts
    let x = symplex::var("x");
    let integrand = x.ln().powi(3);
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("Integral"), "ln(x)³ should be integrated: {s}");

    // FTC check at x=2
    let deriv = anti.diff(&x);
    let pt = symplex::int(2);
    let orig_val = integrand.subs(&x, &pt).eval_f64().unwrap();
    let deriv_val = deriv.subs(&x, &pt).eval_f64().unwrap();
    let err = (orig_val - deriv_val).abs();
    assert!(
        err < 1e-7,
        "FTC for ∫ln(x)³dx: integrand={orig_val}, deriv={deriv_val}, err={err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE improvements
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ode_y_double_prime_plus_y_eq_0_uses_trig() {
    // y'' + y = 0 → y = C1·cos(x) + C2·sin(x) (not complex exponentials)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y; // y'' + y = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("should solve y'' + y = 0");
    let s = format!("{sol}");

    // Must have two constants
    assert_eq!(constants.len(), 2, "should have 2 constants: {s}");
    assert!(s.contains("C1"), "should contain C1: {s}");
    assert!(s.contains("C2"), "should contain C2: {s}");

    // Must use trig, not complex exp
    assert!(
        s.contains("cos") && s.contains("sin"),
        "should use cos and sin (not complex exp): {s}"
    );
}

#[test]
fn ode_y_double_prime_plus_y_eq_0_trig_solution_correct() {
    // Verify the solution y = C1*cos(x) + C2*sin(x) satisfies the ODE
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y + &y;

    let (sol, _) = ode.solve_ode(&y, &x).expect("should solve");

    // Substitute C1=1, C2=0: y = cos(x) → y'' + y = -cos(x) + cos(x) = 0
    let c1 = symplex::var("C1");
    let c2 = symplex::var("C2");
    let sol_c = sol
        .subs(&c1, &symplex::int(1))
        .subs(&c2, &symplex::int(0));

    // Compute y'' + y numerically
    let sol_dd = sol_c.diff(&x).diff(&x);
    let check = &sol_dd + &sol_c;

    let pt = symplex::rational(7, 10);
    if let Ok(val) = check.subs(&x, &pt).eval_f64() {
        assert!(
            val.abs() < 1e-8,
            "y'' + y should be 0 for y=cos(x): got {val}"
        );
    }
}

#[test]
fn ode_y_double_prime_plus_y_eq_sin_x() {
    // y'' + y = sin(x) — resonance case
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let sin_x = x.sin();
    let ode = &(&d2y + &y) - &sin_x; // y'' + y - sin(x) = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y'' + y = sin(x)");

    let (sol, _) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("C1") && s.contains("C2"),
        "should have two constants: {s}"
    );
    // Trig form in homogeneous part
    assert!(
        s.contains("cos") && s.contains("sin"),
        "should use trig form: {s}"
    );
}

#[test]
fn ode_y_double_prime_plus_y_eq_sin_x_verifies() {
    // Verify the solution numerically
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let sin_x = x.sin();
    let ode = &(&d2y + &y) - &sin_x;

    if let Some((sol, _)) = ode.solve_ode(&y, &x) {
        let c1 = symplex::var("C1");
        let c2 = symplex::var("C2");
        let sol_specific = sol
            .subs(&c1, &symplex::int(0))
            .subs(&c2, &symplex::int(0));

        // Check y'' + y − sin(x) ≈ 0
        let sol_dd = sol_specific.diff(&x).diff(&x);
        let check = &(&sol_dd + &sol_specific) - &sin_x;

        let pt = symplex::rational(7, 10);
        if let Ok(val) = check.subs(&x, &pt).eval_f64() {
            assert!(
                val.abs() < 1e-6,
                "y'' + y - sin(x) should be ≈0 for particular solution: got {val}"
            );
        }
    }
}

#[test]
fn ode_first_order_linear_exp_rhs() {
    // y' + 2y = exp(-x) → integrating factor solution
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let two_y = &y * 2;
    let neg_x = (&x * -1).exp();
    let ode = &(&dy + &two_y) - &neg_x; // y' + 2y - exp(-x) = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y' + 2y = exp(-x)");

    let (sol, _) = result.unwrap();
    let s = format!("{sol}");
    // The solution should not contain unevaluated Integral
    assert!(
        !s.contains("Integral"),
        "solution should not have unevaluated Integral: {s}"
    );
    assert!(s.contains("C1"), "should have constant C1: {s}");
}

#[test]
fn ode_y_double_prime_minus_y_eq_0_real_exp() {
    // y'' - y = 0 → y = C1*exp(x) + C2*exp(-x) (real roots, no trig)
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &d2y - &y; // y'' - y = 0

    let (sol, constants) = ode
        .solve_ode(&y, &x)
        .expect("should solve y'' - y = 0");
    let s = format!("{sol}");
    assert_eq!(constants.len(), 2, "should have 2 constants: {s}");
    assert!(s.contains("exp"), "should use exp: {s}");
}

#[test]
fn ode_damped_oscillator() {
    // y'' + 2y' + 5y = 0 → complex roots -1 ± 2i
    // Solution: exp(-x)·(C1·cos(2x) + C2·sin(2x))
    let x = symplex::var("x");
    let y = symplex::var("y");
    let dy = y.formal_diff(&x);
    let d2y = dy.formal_diff(&x);
    let ode = &(&d2y + &(&dy * 2)) + &(&y * 5); // y'' + 2y' + 5y = 0

    let result = ode.solve_ode(&y, &x);
    assert!(result.is_some(), "should solve y'' + 2y' + 5y = 0");

    let (sol, _) = result.unwrap();
    let s = format!("{sol}");
    assert!(
        s.contains("cos") && s.contains("sin"),
        "damped oscillator should use trig form: {s}"
    );
    assert!(s.contains("exp"), "should have exponential decay: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cosmetic: exp(0) artifacts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn no_exp_zero_artifact() {
    // Integration results should not contain exp(0)
    let x = symplex::var("x");
    let integrand = x.sin();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("exp(0)"), "should not contain exp(0): {s}");
}

#[test]
fn no_exp_zero_in_linear_sub() {
    // ∫ sin(2x) dx — should not produce exp(0) artifacts
    let x = symplex::var("x");
    let two_x = &x * 2;
    let integrand = two_x.sin();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(!s.contains("exp(0)"), "should not contain exp(0): {s}");
}
