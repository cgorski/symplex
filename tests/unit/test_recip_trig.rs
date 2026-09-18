//! Tests for reciprocal trig/hyperbolic convenience methods and sinc.

// ═══════════════════════════════════════════════════════════════════════════
// Structural equality — each convenience method is sugar over existing ops
// ═══════════════════════════════════════════════════════════════════════════

use symplex::prelude::*;
#[test]
fn sec_is_one_over_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sec = x.sec();
    let expected = &ctx.int(1) / &x.cos();
    assert_eq!(sec, expected);
}

#[test]
fn csc_is_one_over_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let csc = x.csc();
    let expected = &ctx.int(1) / &x.sin();
    assert_eq!(csc, expected);
}

#[test]
fn cot_is_cos_over_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cot = x.cot();
    let expected = &x.cos() / &x.sin();
    assert_eq!(cot, expected);
}

#[test]
fn acot_is_atan_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.acot(), (&ctx.int(1) / &x).atan());
}

#[test]
fn asec_is_acos_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.asec(), (&ctx.int(1) / &x).acos());
}

#[test]
fn acsc_is_asin_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.acsc(), (&ctx.int(1) / &x).asin());
}

// ── Reciprocal hyperbolic ──────────────────────────────────────────────

#[test]
fn coth_is_cosh_over_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.coth(), &x.cosh() / &x.sinh());
}

#[test]
fn sech_is_one_over_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.sech(), &ctx.int(1) / &x.cosh());
}

#[test]
fn csch_is_one_over_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.csch(), &ctx.int(1) / &x.sinh());
}

#[test]
fn acoth_is_atanh_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.acoth(), (&ctx.int(1) / &x).atanh());
}

// ── sinc ───────────────────────────────────────────────────────────────

#[test]
fn sinc_is_sin_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.sinc(), &x.sin() / &x);
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval at known points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sec_at_zero_is_one() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.sec().eval();
    assert_eq!(format!("{result}"), "1", "sec(0) = 1/cos(0) = 1/1 = 1");
}

#[test]
fn csc_at_pi_over_2() {
    let ctx = Context::new();
    let pi_half = &ctx.pi() / &ctx.int(2);
    let result = pi_half.csc().eval();
    assert_eq!(format!("{result}"), "1", "csc(π/2) = 1/sin(π/2) = 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical checks via evalf_f64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sec_numerical() {
    let ctx = Context::new();
    let val = ctx.rational(7, 10);
    let result = val.sec().eval_f64().unwrap();
    let expected = 1.0 / (0.7_f64).cos();
    assert!(
        (result - expected).abs() < 1e-10,
        "sec(0.7) ≈ {expected}, got {result}"
    );
}

#[test]
fn cot_numerical() {
    let ctx = Context::new();
    let val = ctx.rational(7, 10);
    let result = val.cot().eval_f64().unwrap();
    let expected = (0.7_f64).cos() / (0.7_f64).sin();
    assert!(
        (result - expected).abs() < 1e-10,
        "cot(0.7) ≈ {expected}, got {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_sec() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sec = x.sec();
    let d = sec.diff(&x);
    // d/dx(1/cos(x)) = sin(x)/cos²(x) = sec(x)·tan(x)
    // Check numerically at x = 0.7
    let val = ctx.rational(7, 10);
    let d_val = d.subs(&x, &val).eval_f64().unwrap();
    let expected = (0.7_f64).sin() / (0.7_f64).cos().powi(2);
    assert!(
        (d_val - expected).abs() < 1e-10,
        "d/dx sec(x) at 0.7: expected {expected}, got {d_val}"
    );
}
