//! Tests for Wave S special elementary functions:
//! Heaviside step function, Dirac delta distribution, Lambert W function.

use symplex::prelude::*;
use symplex::vars;

// ═══════════════════════════════════════════════════════════════════════════
// Heaviside step function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn heaviside_positive() {
    assert_eq!(format!("{}", symplex::int(5).heaviside().eval()), "1");
}

#[test]
fn heaviside_negative() {
    assert_eq!(format!("{}", symplex::int(-3).heaviside().eval()), "0");
}

#[test]
fn heaviside_zero() {
    assert_eq!(format!("{}", symplex::int(0).heaviside().eval()), "1/2");
}

#[test]
fn heaviside_positive_rational() {
    // Heaviside(3/7) should be 1 (positive argument)
    assert_eq!(
        format!("{}", symplex::rational(3, 7).heaviside().eval()),
        "1"
    );
}

#[test]
fn heaviside_negative_rational() {
    // Heaviside(-2/5) should be 0 (negative argument)
    assert_eq!(
        format!("{}", symplex::rational(-2, 5).heaviside().eval()),
        "0"
    );
}

#[test]
fn heaviside_symbolic_stays() {
    vars!(x);
    let h = x.heaviside();
    let s = format!("{h}");
    assert!(
        s.contains("H") || s.contains("heaviside") || s.contains("Heaviside"),
        "symbolic heaviside should stay unevaluated: {s}"
    );
}

#[test]
fn heaviside_via_macro() {
    let n = symplex::int(5);
    let result = expr!(heaviside(n));
    assert_eq!(format!("{}", result.eval()), "1");
}

#[test]
fn heaviside_via_macro_symbolic() {
    vars!(x);
    let result = expr!(heaviside(x));
    assert_eq!(result, x.heaviside());
}

// ═══════════════════════════════════════════════════════════════════════════
// Dirac delta distribution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dirac_delta_nonzero() {
    assert_eq!(format!("{}", symplex::int(5).dirac_delta().eval()), "0");
}

#[test]
fn dirac_delta_negative_nonzero() {
    assert_eq!(format!("{}", symplex::int(-7).dirac_delta().eval()), "0");
}

#[test]
fn dirac_delta_rational_nonzero() {
    assert_eq!(
        format!("{}", symplex::rational(1, 3).dirac_delta().eval()),
        "0"
    );
}

#[test]
fn dirac_delta_at_zero_stays() {
    let result = symplex::int(0).dirac_delta().eval();
    let s = format!("{result}");
    // Should stay unevaluated (not 0, not infinity)
    assert!(
        s.contains("DiracDelta") || s.contains("dirac_delta") || s.contains("delta"),
        "should be unevaluated at 0: {s}"
    );
}

#[test]
fn dirac_delta_symbolic_stays() {
    vars!(x);
    let d = x.dirac_delta();
    let s = format!("{d}");
    assert!(
        s.contains("DiracDelta") || s.contains("dirac_delta") || s.contains("delta"),
        "symbolic dirac_delta should stay unevaluated: {s}"
    );
}

#[test]
fn dirac_delta_via_macro() {
    let n = symplex::int(3);
    let result = expr!(dirac_delta(n));
    assert_eq!(format!("{}", result.eval()), "0");
}

#[test]
fn dirac_delta_via_macro_symbolic() {
    vars!(x);
    let result = expr!(dirac_delta(x));
    assert_eq!(result, x.dirac_delta());
}

// ═══════════════════════════════════════════════════════════════════════════
// Lambert W function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lambertw_at_zero() {
    assert_eq!(format!("{}", symplex::int(0).lambertw().eval()), "0");
}

#[test]
fn lambertw_at_e() {
    let result = symplex::e().lambertw().eval();
    assert_eq!(format!("{result}"), "1", "W(e) = 1");
}

#[test]
fn lambertw_symbolic() {
    vars!(x);
    let w = expr!(lambertw(x));
    let s = format!("{w}");
    assert!(s.contains("W("), "should display as W(x): {s}");
}

#[test]
fn lambertw_symbolic_stays() {
    vars!(y);
    let w = y.lambertw();
    let s = format!("{w}");
    assert!(
        s.contains("W("),
        "symbolic lambertw should stay unevaluated: {s}"
    );
}

#[test]
fn lambertw_integer_nonzero_stays() {
    // W(2) has no closed form — should remain unevaluated
    let result = symplex::int(2).lambertw().eval();
    let s = format!("{result}");
    assert!(s.contains("W("), "W(2) should stay unevaluated: {s}");
}

#[test]
fn lambertw_via_macro() {
    let n = symplex::int(0);
    let result = expr!(lambertw(n));
    assert_eq!(format!("{}", result.eval()), "0");
}

#[test]
fn lambertw_via_macro_symbolic() {
    vars!(x);
    let result = expr!(lambertw(x));
    assert_eq!(result, x.lambertw());
}
