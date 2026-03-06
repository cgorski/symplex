//! Tests for Gaussian integral (erf) and Laplace derivative rule.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Gaussian integral: ∫ exp(a·x² + b·x + c) dx  →  erf  (when a < 0)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integrate_exp_neg_x2_contains_erf() {
    // ∫ exp(-x²) dx = √π/2 · erf(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) * -1).exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral") && !s.contains('∫'),
        "should not be unevaluated: {s}"
    );
    assert!(
        s.contains("erf") || s.contains("Erf"),
        "should contain erf: {s}"
    );
}

#[test]
fn integrate_exp_neg_x2_has_sqrt_pi() {
    // The antiderivative √π/2 · erf(x) should mention π
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) * -1).exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        s.contains('π') || s.to_lowercase().contains("pi"),
        "should contain π: {s}"
    );
}

#[test]
fn integrate_exp_neg_x2_ftc() {
    // Fundamental Theorem of Calculus:
    //   d/dx [ ∫ exp(-x²) dx ] == exp(-x²)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) * -1).exp();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    // Evaluate both at x = 7/10
    let pt = ctx.rational(7, 10);
    let orig = integrand.subs(&x, &pt).eval_f64().unwrap();
    let d = deriv.subs(&x, &pt).eval_f64().unwrap();
    assert!(
        (orig - d).abs() < 1e-8,
        "FTC violated: integrand={orig}, derivative={d}"
    );
}

#[test]
fn integrate_exp_neg_2x2() {
    // ∫ exp(-2x²) dx  (a = -2, b = 0, c = 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) * -2).exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral") && !s.contains('∫'),
        "should not be unevaluated: {s}"
    );
    assert!(
        s.contains("erf") || s.contains("Erf"),
        "should contain erf: {s}"
    );
}

#[test]
fn integrate_exp_neg_2x2_ftc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = (&x.powi(2) * -2).exp();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    let pt = ctx.rational(3, 10);
    let orig = integrand.subs(&x, &pt).eval_f64().unwrap();
    let d = deriv.subs(&x, &pt).eval_f64().unwrap();
    assert!(
        (orig - d).abs() < 1e-8,
        "FTC violated: integrand={orig}, derivative={d}"
    );
}

#[test]
fn integrate_exp_neg_x2_plus_2x() {
    // ∫ exp(-x² + 2x) dx  (a = -1, b = 2, c = 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let arg = &(&x.powi(2) * -1) + &(&x * 2);
    let integrand = arg.exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral") && !s.contains('∫'),
        "should not be unevaluated: {s}"
    );
    assert!(
        s.contains("erf") || s.contains("Erf"),
        "should contain erf: {s}"
    );
}

#[test]
fn integrate_exp_neg_x2_plus_2x_ftc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let arg = &(&x.powi(2) * -1) + &(&x * 2);
    let integrand = arg.exp();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    let pt = ctx.rational(1, 2);
    let orig = integrand.subs(&x, &pt).eval_f64().unwrap();
    let d = deriv.subs(&x, &pt).eval_f64().unwrap();
    assert!(
        (orig - d).abs() < 1e-6,
        "FTC violated: integrand={orig}, derivative={d}"
    );
}

#[test]
fn integrate_exp_full_quadratic() {
    // ∫ exp(-x² + 2x - 1) dx  (a = -1, b = 2, c = -1)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let arg = &(&(&x.powi(2) * -1) + &(&x * 2)) + &ctx.int(-1);
    let integrand = arg.exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    assert!(
        !s.contains("Integral") && !s.contains('∫'),
        "should not be unevaluated: {s}"
    );
    assert!(
        s.contains("erf") || s.contains("Erf"),
        "should contain erf: {s}"
    );
}

#[test]
fn integrate_exp_full_quadratic_ftc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let arg = &(&(&x.powi(2) * -1) + &(&x * 2)) + &ctx.int(-1);
    let integrand = arg.exp();
    let anti = integrand.integrate(&x);
    let deriv = anti.diff(&x);

    let pt = ctx.rational(4, 10);
    let orig = integrand.subs(&x, &pt).eval_f64().unwrap();
    let d = deriv.subs(&x, &pt).eval_f64().unwrap();
    assert!(
        (orig - d).abs() < 1e-6,
        "FTC violated: integrand={orig}, derivative={d}"
    );
}

#[test]
fn integrate_exp_pos_x2_stays_unevaluated() {
    // ∫ exp(x²) dx has a > 0 — should remain unevaluated (no erfi support)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let integrand = x.powi(2).exp();
    let anti = integrand.integrate(&x);
    let s = format!("{anti}");
    // Positive leading coefficient: must NOT produce erf
    assert!(
        !s.contains("erf") && !s.contains("Erf"),
        "positive a should not produce erf: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace derivative rule: L{f'(t)} = s · F(s) − f(0)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_derivative_of_sin() {
    // Derivative(sin(t), t) represents d/dt sin(t) = cos(t)
    // L{cos(t)} = s/(s²+1)
    // Via the rule: L{Derivative(sin(t), t)} = s·L{sin(t)} − sin(0)
    //   = s · 1/(s²+1) − 0  =  s/(s²+1)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.sin().formal_diff(&t);
    let result = df.laplace(&t, &s).unwrap();

    // Compare numerically with L{cos(t)} = s/(s²+1)
    let cos_transform = t.cos().laplace(&t, &s).unwrap();
    let test_s = ctx.int(3);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = cos_transform.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{sin'(t)}} should equal L{{cos(t)}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_derivative_of_exp() {
    // Derivative(exp(2t), t) represents d/dt exp(2t) = 2·exp(2t)
    // L{2·exp(2t)} = 2/(s−2)
    // Via the rule: s·L{exp(2t)} − exp(0) = s/(s−2) − 1 = 2/(s−2)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let f = (&t * 2).exp();
    let df = f.formal_diff(&t);
    let result = df.laplace(&t, &s).unwrap();

    // Direct: L{2·exp(2t)} = 2/(s−2)
    let direct = (&f * 2).laplace(&t, &s).unwrap();

    let test_s = ctx.int(5);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = direct.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{d/dt exp(2t)}} should equal L{{2·exp(2t)}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_derivative_of_t() {
    // Derivative(t, t) represents d/dt(t) = 1
    // L{1} = 1/s
    // Via the rule: s·L{t} − t(0) = s·1/s² − 0 = 1/s
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.formal_diff(&t);
    let result = df.laplace(&t, &s).unwrap();

    let one_transform = ctx.int(1).laplace(&t, &s).unwrap();

    let test_s = ctx.int(4);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = one_transform.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{t'}} should equal L{{1}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_constant_times_derivative() {
    // L{3 · Derivative(sin(t), t)} = 3 · L{cos(t)} = 3s/(s²+1)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.sin().formal_diff(&t);
    let three_df = &df * 3;
    let result = three_df.laplace(&t, &s).unwrap();

    let expected = &t.cos().laplace(&t, &s).unwrap() * 3;

    let test_s = ctx.int(2);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = expected.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{3·sin'(t)}} should equal 3·L{{cos(t)}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_derivative_of_t_squared() {
    // Derivative(t², t) represents d/dt(t²) = 2t
    // L{2t} = 2/s²
    // Via the rule: s·L{t²} − t²(0) = s·2/s³ − 0 = 2/s²
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.powi(2).formal_diff(&t);
    let result = df.laplace(&t, &s).unwrap();

    // Direct: L{2t} = 2·L{t} = 2/s²
    let direct = (&t * 2).laplace(&t, &s).unwrap();

    let test_s = ctx.int(3);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = direct.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{d/dt t²}} should equal L{{2t}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_derivative_of_cos() {
    // Derivative(cos(t), t) = -sin(t)
    // L{-sin(t)} = -1/(s²+1)
    // Via the rule: s·L{cos(t)} − cos(0) = s·s/(s²+1) − 1
    //   = s²/(s²+1) − 1 = -1/(s²+1)
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.cos().formal_diff(&t);
    let result = df.laplace(&t, &s).unwrap();

    // Direct: L{-sin(t)} = -1/(s²+1)
    let neg_sin_transform = (&t.sin() * -1).laplace(&t, &s).unwrap();

    let test_s = ctx.int(2);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = neg_sin_transform.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{cos'(t)}} should equal L{{-sin(t)}}: {v1} vs {v2}"
    );
}

#[test]
fn laplace_neg_derivative() {
    // L{-Derivative(sin(t), t)} = -L{cos(t)}
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    let df = t.sin().formal_diff(&t);
    let neg_df = &df * -1;
    let result = neg_df.laplace(&t, &s).unwrap();

    let expected = &t.cos().laplace(&t, &s).unwrap() * -1;

    let test_s = ctx.int(5);
    let v1 = result.subs(&s, &test_s).eval_f64().unwrap();
    let v2 = expected.subs(&s, &test_s).eval_f64().unwrap();
    assert!(
        (v1 - v2).abs() < 1e-10,
        "L{{-sin'(t)}} should equal -L{{cos(t)}}: {v1} vs {v2}"
    );
}
