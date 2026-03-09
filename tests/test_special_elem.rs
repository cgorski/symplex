//! Tests for Wave S special elementary functions:
//! Heaviside step function, Dirac delta distribution, Lambert W function.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Heaviside step function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heaviside_positive() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(5).heaviside().eval()), "1");
}

#[test]
fn heaviside_negative() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(-3).heaviside().eval()), "0");
}

#[test]
fn heaviside_zero() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(0).heaviside().eval()), "1/2");
}

#[test]
fn heaviside_positive_rational() {
    let __ctx = Context::new();
    // Heaviside(3/7) should be 1 (positive argument)
    assert_eq!(
        format!("{}", __ctx.rational(3, 7).heaviside().eval()),
        "1"
    );
}

#[test]
fn heaviside_negative_rational() {
    let __ctx = Context::new();
    // Heaviside(-2/5) should be 0 (negative argument)
    assert_eq!(
        format!("{}", __ctx.rational(-2, 5).heaviside().eval()),
        "0"
    );
}

#[test]
fn heaviside_symbolic_stays() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let h = x.heaviside();
    let s = format!("{h}");
    assert!(
        s.contains("H") || s.contains("heaviside") || s.contains("Heaviside"),
        "symbolic heaviside should stay unevaluated: {s}"
    );
}

#[test]
fn heaviside_via_macro() {
    let __ctx = Context::new();
    let n = __ctx.int(5);
    let result = expr!(heaviside(n));
    assert_eq!(format!("{}", result.eval()), "1");
}

#[test]
fn heaviside_via_macro_symbolic() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(heaviside(x));
    assert_eq!(result, x.heaviside());
}

// ═══════════════════════════════════════════════════════════════════════════
// Dirac delta distribution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dirac_delta_nonzero() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(5).dirac_delta().eval()), "0");
}

#[test]
fn dirac_delta_negative_nonzero() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(-7).dirac_delta().eval()), "0");
}

#[test]
fn dirac_delta_rational_nonzero() {
    let __ctx = Context::new();
    assert_eq!(
        format!("{}", __ctx.rational(1, 3).dirac_delta().eval()),
        "0"
    );
}

#[test]
fn dirac_delta_at_zero_stays() {
    let __ctx = Context::new();
    let result = __ctx.int(0).dirac_delta().eval();
    let s = format!("{result}");
    // Should stay unevaluated (not 0, not infinity)
    assert!(
        s.contains("DiracDelta") || s.contains("dirac_delta") || s.contains("delta"),
        "should be unevaluated at 0: {s}"
    );
}

#[test]
fn dirac_delta_symbolic_stays() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let d = x.dirac_delta();
    let s = format!("{d}");
    assert!(
        s.contains("DiracDelta") || s.contains("dirac_delta") || s.contains("delta"),
        "symbolic dirac_delta should stay unevaluated: {s}"
    );
}

#[test]
fn dirac_delta_via_macro() {
    let __ctx = Context::new();
    let n = __ctx.int(3);
    let result = expr!(dirac_delta(n));
    assert_eq!(format!("{}", result.eval()), "0");
}

#[test]
fn dirac_delta_via_macro_symbolic() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(dirac_delta(x));
    assert_eq!(result, x.dirac_delta());
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambertw_at_zero() {
    let __ctx = Context::new();
    assert_eq!(format!("{}", __ctx.int(0).lambertw().eval()), "0");
}

#[test]
fn lambertw_at_e() {
    let __ctx = Context::new();
    let result = __ctx.e().lambertw().eval();
    assert_eq!(format!("{result}"), "1", "W(e) = 1");
}

#[test]
fn lambertw_symbolic() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let w = expr!(lambertw(x));
    let s = format!("{w}");
    assert!(s.contains("W("), "should display as W(x): {s}");
}

#[test]
fn lambertw_symbolic_stays() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; y);
    let w = y.lambertw();
    let s = format!("{w}");
    assert!(
        s.contains("W("),
        "symbolic lambertw should stay unevaluated: {s}"
    );
}

#[test]
fn lambertw_integer_nonzero_stays() {
    let __ctx = Context::new();
    // W(2) has no closed form — should remain unevaluated
    let result = __ctx.int(2).lambertw().eval();
    let s = format!("{result}");
    assert!(s.contains("W("), "W(2) should stay unevaluated: {s}");
}

#[test]
fn lambertw_via_macro() {
    let __ctx = Context::new();
    let n = __ctx.int(0);
    let result = expr!(lambertw(n));
    assert_eq!(format!("{}", result.eval()), "0");
}

#[test]
fn lambertw_via_macro_symbolic() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let result = expr!(lambertw(x));
    assert_eq!(result, x.lambertw());
}
