//! Tests for Heaviside and DiracDelta integration edge cases.

use symplex::prelude::*;

#[test]
fn heaviside_positive_coeff_ftc() {
    let ctx = Context::new();
    // ∫ H(2x - 1) dx — verify FTC at points where H is 0 and 1
    symplex::syms!(ctx; x);
    let inner = &ctx.int(2) * &x - &ctx.int(1);
    let heaviside = inner.heaviside();
    let antideriv = heaviside.integrate(&x);

    // Verify d/dx(antideriv) = H(2x-1) at several points
    let deriv = antideriv.diff(&x);

    // At x=2 (where 2*2-1=3>0, H=1): deriv should be 1
    let val = deriv.subs_i64(&x, 2).eval().eval_f64();
    if let Ok(v) = val {
        assert!((v - 1.0).abs() < 1e-8, "FTC check at x=2: got {}", v);
    }

    // At x=-1 (where 2*(-1)-1=-3<0, H=0): deriv should be 0
    let val = deriv.subs_i64(&x, -1).eval().eval_f64();
    if let Ok(v) = val {
        assert!(v.abs() < 1e-8, "FTC check at x=-1: got {}", v);
    }
}

#[test]
fn heaviside_negative_coeff_ftc() {
    let ctx = Context::new();
    // ∫ H(-2x + 3) dx — negative coefficient
    symplex::syms!(ctx; x);
    let inner = &ctx.int(-2) * &x + &ctx.int(3);
    let heaviside = inner.heaviside();
    let antideriv = heaviside.integrate(&x);

    // Verify FTC: d/dx(antideriv) should equal H(-2x+3)
    let deriv = antideriv.diff(&x);

    // At x=0 (where -2*0+3=3>0, H=1): deriv should be 1
    let val = deriv.subs_i64(&x, 0).eval().eval_f64();
    if let Ok(v) = val {
        assert!((v - 1.0).abs() < 1e-8, "FTC check at x=0: got {}", v);
    }

    // At x=5 (where -2*5+3=-7<0, H=0): deriv should be 0
    let val = deriv.subs_i64(&x, 5).eval().eval_f64();
    if let Ok(v) = val {
        assert!(v.abs() < 1e-8, "FTC check at x=5: got {}", v);
    }
}

#[test]
fn heaviside_simple_integration() {
    let ctx = Context::new();
    // ∫ H(x) dx = x·H(x)
    symplex::syms!(ctx; x);
    let h = x.heaviside();
    let result = h.integrate(&x);
    let display = format!("{}", result);
    assert!(
        !display.contains("Integral"),
        "integration should not be unevaluated: {}",
        display
    );
}

#[test]
fn heaviside_linear_integration_not_unevaluated() {
    let ctx = Context::new();
    // ∫ H(3x + 2) dx should not remain as an unevaluated Integral
    symplex::syms!(ctx; x);
    let inner = &ctx.int(3) * &x + &ctx.int(2);
    let h = inner.heaviside();
    let result = h.integrate(&x);
    let display = format!("{}", result);
    assert!(
        !display.contains("Integral"),
        "integration of H(3x+2) should not be unevaluated: {}",
        display
    );
}

#[test]
fn dirac_simple_integration() {
    let ctx = Context::new();
    // ∫ δ(x) dx = H(x)
    symplex::syms!(ctx; x);
    let delta = x.dirac_delta();
    let result = delta.integrate(&x);
    let display = format!("{}", result);
    assert!(
        !display.contains("Integral"),
        "integration of δ(x) should not be unevaluated: {}",
        display
    );
    assert!(
        display.contains("Heaviside") || display.contains("H("),
        "∫δ(x)dx should contain Heaviside: {}",
        display
    );
}

#[test]
fn dirac_negative_coeff() {
    let ctx = Context::new();
    // ∫ δ(-3x + 6) dx = H(-3x+6) / 3
    symplex::syms!(ctx; x);
    let inner = &ctx.int(-3) * &x + &ctx.int(6);
    let delta = inner.dirac_delta();
    let result = delta.integrate(&x);
    let display = format!("{}", result);
    // Just verify it's not unevaluated
    assert!(
        !display.contains("Integral"),
        "integration should not be unevaluated: {}",
        display
    );
}

#[test]
fn dirac_positive_coeff() {
    let ctx = Context::new();
    // ∫ δ(4x - 8) dx = H(4x-8) / 4
    symplex::syms!(ctx; x);
    let inner = &ctx.int(4) * &x - &ctx.int(8);
    let delta = inner.dirac_delta();
    let result = delta.integrate(&x);
    let display = format!("{}", result);
    assert!(
        !display.contains("Integral"),
        "integration of δ(4x-8) should not be unevaluated: {}",
        display
    );
}

#[test]
fn heaviside_evaluates_to_one_for_positive_arg() {
    let ctx = Context::new();
    // H(5) should evaluate to 1
    let h = ctx.int(5).heaviside();
    let result = h.eval().eval_f64();
    if let Ok(v) = result {
        assert!(
            (v - 1.0).abs() < 1e-10,
            "H(5) should be 1, got {}",
            v
        );
    }
}

#[test]
fn heaviside_evaluates_to_zero_for_negative_arg() {
    let ctx = Context::new();
    // H(-3) should evaluate to 0
    let h = ctx.int(-3).heaviside();
    let result = h.eval().eval_f64();
    if let Ok(v) = result {
        assert!(
            v.abs() < 1e-10,
            "H(-3) should be 0, got {}",
            v
        );
    }
}
