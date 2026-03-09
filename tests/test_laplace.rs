//! Public API tests for Laplace and inverse Laplace transforms.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: numerically verify a Laplace transform result at a given s value
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate the Laplace result expression at a specific numeric s value
/// (given as integer numerator/denominator) and compare against the expected
/// numeric value.
fn verify_laplace_numerically(
    result: &Ex,
    s: &Ex,
    s_num: i64,
    s_den: i64,
    expected: f64,
    label: &str,
) {
    let __ctx = result.context();
    let s_val = __ctx.rational(s_num, s_den);
    let at_s = result.subs(s, &s_val);
    let val = at_s.eval_f64().unwrap_or_else(|_| panic!(
        "{label}: should evaluate numerically at s={s_num}/{s_den}"
    ));
    assert!(
        (val - expected).abs() < 1e-3,
        "{label}: at s={s_num}/{s_den}, expected {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward Laplace transforms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplace_constant() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    let f = __ctx.int(5);
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    // L{5} = 5/s — displayed as 5*s^(-1) or similar
    assert!(
        d.contains("5") && d.contains("s"),
        "L{{5}} should be 5/s, got: {d}"
    );
    // Numerical: at s=3, 5/3 ≈ 1.6667
    verify_laplace_numerically(&result, &s, 3, 1, 5.0 / 3.0, "L{5}");
}

#[test]
fn laplace_one() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    let f = __ctx.int(1);
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    // L{1} = 1/s
    assert!(d.contains("s"), "L{{1}} should be 1/s, got: {d}");
    // Numerical: at s=4, 1/4 = 0.25
    verify_laplace_numerically(&result, &s, 4, 1, 0.25, "L{1}");
}

#[test]
fn laplace_exp() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{exp(2t)} = 1/(s-2)
    let f = (&t * 2).exp();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    // Should contain (s - 2) in denominator
    assert!(
        d.contains("s") && d.contains("2"),
        "L{{exp(2t)}} should involve s and 2, got: {d}"
    );
    // Numerical: at s=5, 1/(5-2) = 1/3 ≈ 0.3333
    verify_laplace_numerically(&result, &s, 5, 1, 1.0 / 3.0, "L{exp(2t)}");
}

#[test]
fn laplace_exp_negative() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{exp(-3t)} = 1/(s+3)
    let f = (&t * -3).exp();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(
        d.contains("s") && d.contains("3"),
        "L{{exp(-3t)}} should involve s and 3, got: {d}"
    );
    // Numerical: at s=2, 1/(2+3) = 0.2
    verify_laplace_numerically(&result, &s, 2, 1, 0.2, "L{exp(-3t)}");
}

#[test]
fn laplace_sin() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{sin(3t)} = 3/(s²+9)
    let f = (&t * 3).sin();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(
        d.contains("3") && d.contains("s"),
        "L{{sin(3t)}} should involve 3 and s, got: {d}"
    );
    // Numerical: at s=4, 3/(16+9) = 3/25 = 0.12
    verify_laplace_numerically(&result, &s, 4, 1, 3.0 / 25.0, "L{sin(3t)}");
}

#[test]
fn laplace_cos() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{cos(t)} = s/(s²+1)
    let f = t.cos();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(d.contains("s"), "L{{cos(t)}} should involve s, got: {d}");
    // Numerical: at s=2, 2/(4+1) = 2/5 = 0.4
    verify_laplace_numerically(&result, &s, 2, 1, 2.0 / 5.0, "L{cos(t)}");
}

#[test]
fn laplace_t_squared() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{t²} = 2/s³ = 2*s^(-3)
    let f = t.powi(2);
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(
        d.contains("2") && d.contains("s"),
        "L{{t²}} should be 2/s³, got: {d}"
    );
    // Numerical: at s=3, 2/27 ≈ 0.07407
    verify_laplace_numerically(&result, &s, 3, 1, 2.0 / 27.0, "L{t²}");
}

#[test]
fn laplace_linearity() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{3*exp(t) + 2*sin(t)} should succeed (linearity)
    let term1 = &t.exp() * 3;
    let term2 = &t.sin() * 2;
    let f = &term1 + &term2;
    let r = f.laplace(&t, &s);
    let d = format!("{r}");
    assert!(d.contains("s"), "result should contain s, got: {d}");
    // Numerical: at s=4, 3/(4-1) + 2/(16+1) = 1 + 2/17 ≈ 1.1176
    verify_laplace_numerically(
        &r,
        &s,
        4,
        1,
        3.0 / 3.0 + 2.0 / 17.0,
        "L{3*exp(t)+2*sin(t)}",
    );
}

#[test]
fn laplace_freq_shift() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{exp(2t)*sin(3t)} = 3/((s-2)²+9) via frequency shift
    let f = &((&t * 2).exp()) * &((&t * 3).sin());
    let r = f.laplace(&t, &s);
    let d = format!("{r}");
    assert!(
        d.contains("s") && d.contains("3"),
        "L{{exp(2t)*sin(3t)}} should involve s and 3, got: {d}"
    );
    // Numerical: at s=5, 3/((5-2)²+9) = 3/(9+9) = 3/18 ≈ 0.1667
    verify_laplace_numerically(&r, &s, 5, 1, 3.0 / 18.0, "L{exp(2t)*sin(3t)}");
}

#[test]
fn laplace_sinh() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{sinh(2t)} = 2/(s²-4)
    let f = (&t * 2).sinh();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(
        d.contains("s") && d.contains("2"),
        "L{{sinh(2t)}} should involve s and 2, got: {d}"
    );
    // Numerical: at s=3, 2/(9-4) = 2/5 = 0.4
    verify_laplace_numerically(&result, &s, 3, 1, 2.0 / 5.0, "L{sinh(2t)}");
}

#[test]
fn laplace_cosh() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L{cosh(t)} = s/(s²-1)
    let f = t.cosh();
    let result = f.laplace(&t, &s);
    let d = format!("{result}");
    assert!(d.contains("s"), "L{{cosh(t)}} should involve s, got: {d}");
    // Numerical: at s=3, 3/(9-1) = 3/8 = 0.375
    verify_laplace_numerically(&result, &s, 3, 1, 3.0 / 8.0, "L{cosh(t)}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Laplace transforms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_laplace_1_over_s() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L⁻¹{1/s} = 1
    let f = &__ctx.int(1) / &s;
    let result = f.inverse_laplace(&s, &t);
    let d = format!("{result}");
    // The result should be 1 (no t dependence)
    assert!(
        d == "1" || !d.contains("s"),
        "L⁻¹{{1/s}} should be 1 (constant), got: {d}"
    );
}

#[test]
fn inverse_laplace_1_over_s_minus_a() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L⁻¹{1/(s-2)} = exp(2t)
    let f = &__ctx.int(1) / &(&s - 2);
    let r = f.inverse_laplace(&s, &t);
    let d = format!("{r}");
    assert!(
        d.contains("exp"),
        "L⁻¹{{1/(s-2)}} should contain exp: {d}"
    );
}

#[test]
fn inverse_laplace_constant_over_s() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let s = __ctx.symbol("s");
    // L⁻¹{5/s} = 5
    let f = &__ctx.int(5) / &s;
    let result = f.inverse_laplace(&s, &t);
    let d = format!("{result}");
    assert!(d.contains("5"), "L⁻¹{{5/s}} should be 5, got: {d}");
}

#[test]
fn laplace_rejects_non_symbol_t() {
    let __ctx = Context::new();
    let s = __ctx.symbol("s");
    let f = __ctx.int(1);
    let bad_t = __ctx.int(42); // not a symbol
    let result = f.laplace(&bad_t, &s);
    assert!(result.has_unevaluated(), "should produce unevaluated node for non-symbol t");
}

#[test]
fn laplace_rejects_non_symbol_s() {
    let __ctx = Context::new();
    let t = __ctx.symbol("t");
    let f = __ctx.int(1);
    let bad_s = __ctx.int(42); // not a symbol
    let result = f.laplace(&t, &bad_s);
    assert!(result.has_unevaluated(), "should produce unevaluated node for non-symbol s");
}
