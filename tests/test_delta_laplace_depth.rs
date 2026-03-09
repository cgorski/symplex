//! Tests for DiracDelta sifting property (Wave δ) and Laplace depth (Wave ε).

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// DiracDelta sifting property: ∫ f(x)·δ(g(x)) dx = f(root)·H(g(x))
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sifting_x_times_delta_x() {
    // ∫ x·δ(x) dx = f(0)·H(x) = 0·H(x) = 0
    let x = symplex::default_context().symbol("x");
    let integrand = &x * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert_eq!(s, "0", "∫x·δ(x)dx = f(0)·H(x) = 0, got: {s}");
}

#[test]
fn sifting_const_times_delta_x() {
    // ∫ 5·δ(x) dx = 5·H(x)
    let x = symplex::default_context().symbol("x");
    let five = symplex::default_context().int(5);
    let integrand = &five * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("H") || s.contains("eaviside"),
        "∫5·δ(x)dx should be 5·H(x), got: {s}"
    );
}

#[test]
fn sifting_x_squared_times_delta_x() {
    // ∫ x²·δ(x) dx = f(0)·H(x) = 0·H(x) = 0
    let x = symplex::default_context().symbol("x");
    let x_sq = x.pow(&symplex::default_context().int(2));
    let integrand = &x_sq * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert_eq!(s, "0", "∫x²·δ(x)dx = 0, got: {s}");
}

#[test]
fn sifting_delta_x_minus_a() {
    // ∫ x·δ(x-3) dx = f(3)·H(x-3) = 3·H(x-3)
    let x = symplex::default_context().symbol("x");
    let three = symplex::default_context().int(3);
    let delta_arg = &x - &three;
    let integrand = &x * &delta_arg.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{result}");
    // Result should contain 3 and H/Heaviside
    assert!(
        s.contains("3") && (s.contains("H") || s.contains("eaviside")),
        "∫x·δ(x-3)dx should be 3·H(x-3), got: {s}"
    );
}

#[test]
fn sifting_exp_times_delta_x() {
    // ∫ exp(x)·δ(x) dx = exp(0)·H(x) = 1·H(x) = H(x)
    let x = symplex::default_context().symbol("x");
    let integrand = &x.exp() * &x.dirac_delta();
    let result = integrand.integrate(&x);
    let s = format!("{}", result.eval());
    assert!(
        s.contains("H") || s.contains("eaviside"),
        "∫exp(x)·δ(x)dx should be H(x), got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace time-shift: L{H(t-a)·f(t)} = exp(-a·s)·L{f(t+a)}
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: numerically evaluate a Laplace result at a given s value (expressed
/// as integer numerator / denominator) and check it is close to the expected value.
fn verify_laplace_numerically(
    result: &Ex,
    s_var: &Ex,
    s_num: i64,
    s_den: i64,
    expected: f64,
    label: &str,
) {
    let s_val = symplex::default_context().rational(s_num, s_den);
    let at_s = result.subs(s_var, &s_val);
    let val = at_s.eval_f64().unwrap_or_else(|_| panic!(
        "{label}: should evaluate numerically at s={s_num}/{s_den}"
    ));
    assert!(
        (val - expected).abs() < 1e-2,
        "{label}: at s={s_num}/{s_den}, expected {expected}, got {val}"
    );
}

#[test]
fn laplace_time_shift_heaviside_exp() {
    // L{H(t-2)·exp(t)} should apply time-shift and produce a result
    let t = symplex::default_context().symbol("t");
    let s = symplex::default_context().symbol("s");
    let two = symplex::default_context().int(2);
    let h = (&t - &two).heaviside();
    let f = t.exp();
    let integrand = &h * &f;
    let result = integrand.laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "time-shift for H(t-2)*exp(t) should succeed, got: {result}"
    );
    let d = format!("{result}");
    // Should contain exp (for the exp(-2s) factor)
    assert!(
        d.contains("exp") || d.contains("e"),
        "time-shift result should contain exponential: {d}"
    );
}

#[test]
fn laplace_time_shift_heaviside_t() {
    // L{H(t-1)·t} — time-shift with f(t) = t
    // = exp(-s)·L{(t+1)} = exp(-s)·(1/s² + 1/s)
    let t = symplex::default_context().symbol("t");
    let s = symplex::default_context().symbol("s");
    let one = symplex::default_context().int(1);
    let h = (&t - &one).heaviside();
    let integrand = &h * &t;
    let result = integrand.laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "time-shift for H(t-1)*t should succeed, got: {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace frequency differentiation: L{t^n·f(t)} = (-1)^n d^n/ds^n L{f(t)}
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_t_times_exp_t() {
    // L{t·exp(t)} = -d/ds[1/(s-1)] = 1/(s-1)²
    let t = symplex::default_context().symbol("t");
    let s = symplex::default_context().symbol("s");
    let integrand = &t * &t.exp();
    let result = integrand.laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "L{{t*exp(t)}} should succeed, got: {result}"
    );
    let d = format!("{result}");
    // Should involve (s-1) in some form
    assert!(d.contains("s"), "result should be a function of s: {d}");
    // Numerical verification: at s=3, 1/(3-1)² = 1/4 = 0.25
    verify_laplace_numerically(&result, &s, 3, 1, 0.25, "L{t*exp(t)}");
}

#[test]
fn laplace_t_times_sin_t() {
    // L{t·sin(t)} = -d/ds[1/(s²+1)] = 2s/(s²+1)²
    let t = symplex::default_context().symbol("t");
    let s = symplex::default_context().symbol("s");
    let integrand = &t * &t.sin();
    let result = integrand.laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "L{{t*sin(t)}} should succeed, got: {result}"
    );
    let d = format!("{result}");
    assert!(d.contains("s"), "result should be a function of s: {d}");
    // Numerical verification: at s=2, 2*2/(4+1)² = 4/25 = 0.16
    verify_laplace_numerically(&result, &s, 2, 1, 0.16, "L{t*sin(t)}");
}

#[test]
fn laplace_t_times_cos_t() {
    // L{t·cos(t)} = -d/ds[s/(s²+1)] = (s²-1)/(s²+1)²
    let t = symplex::default_context().symbol("t");
    let s = symplex::default_context().symbol("s");
    let integrand = &t * &t.cos();
    let result = integrand.laplace(&t, &s);
    assert!(
        !result.has_unevaluated(),
        "L{{t*cos(t)}} should succeed, got: {result}"
    );
    let d = format!("{result}");
    assert!(d.contains("s"), "result should be a function of s: {d}");
    // Numerical verification: at s=2, (4-1)/(4+1)² = 3/25 = 0.12
    verify_laplace_numerically(&result, &s, 2, 1, 0.12, "L{t*cos(t)}");
}
