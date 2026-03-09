//! Wave α tests: Gamma half-integers, new macro functions, #[must_use] smoke tests.

use symplex::prelude::*;

// ── Gamma half-integer evaluation ──────────────────────────────────────

#[test]
fn gamma_half_is_sqrt_pi() {
    let __ctx = Context::new();
    let half = __ctx.rational(1, 2);
    let result = half.gamma().eval();
    let expected = __ctx.pi().sqrt();
    assert_eq!(
        format!("{result}"),
        format!("{expected}"),
        "Gamma(1/2) = √π"
    );
}

#[test]
fn gamma_three_halves() {
    let __ctx = Context::new();
    // Gamma(3/2) = (1/2) · √π
    let arg = __ctx.rational(3, 2);
    let result = arg.gamma().eval();
    let expected = &__ctx.rational(1, 2) * &__ctx.pi().sqrt();
    assert_eq!(
        format!("{result}"),
        format!("{expected}"),
        "Gamma(3/2) = (1/2)·√π"
    );
}

#[test]
fn gamma_five_halves() {
    let __ctx = Context::new();
    // Gamma(5/2) = (3/4) · √π
    let arg = __ctx.rational(5, 2);
    let result = arg.gamma().eval();
    let expected = &__ctx.rational(3, 4) * &__ctx.pi().sqrt();
    assert_eq!(
        format!("{result}"),
        format!("{expected}"),
        "Gamma(5/2) = (3/4)·√π"
    );
}

#[test]
fn gamma_seven_halves() {
    let __ctx = Context::new();
    // Gamma(7/2) = (15/8) · √π
    let arg = __ctx.rational(7, 2);
    let result = arg.gamma().eval();
    let expected = &__ctx.rational(15, 8) * &__ctx.pi().sqrt();
    assert_eq!(
        format!("{result}"),
        format!("{expected}"),
        "Gamma(7/2) = (15/8)·√π"
    );
}

// ── expr! macro: floor / ceiling ───────────────────────────────────────

#[test]
fn macro_floor() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let e = expr!(floor(x));
    let s = format!("{e}");
    assert!(
        s.contains("floor"),
        "floor(x) should display with floor: {s}"
    );
}

#[test]
fn macro_ceiling() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let e = expr!(ceiling(x));
    let s = format!("{e}");
    assert!(
        s.contains("ceiling") || s.contains("ceil"),
        "ceiling(x) should display with ceiling: {s}"
    );
}

// ── expr! macro: min / max ─────────────────────────────────────────────

#[test]
fn macro_min() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let e = expr!(min(x, y));
    let s = format!("{e}");
    assert!(
        s.contains("min") || s.contains("Min"),
        "min(x,y) should display with min: {s}"
    );
}

#[test]
fn macro_max() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let y = __ctx.symbol("y");
    let e = expr!(max(x, y));
    let s = format!("{e}");
    assert!(
        s.contains("max") || s.contains("Max"),
        "max(x,y) should display with max: {s}"
    );
}

// ── expr! macro: heaviside / dirac_delta / lambertw ────────────────────

#[test]
fn macro_heaviside_eval_positive() {
    let __ctx = Context::new();
    let result = expr!(heaviside(5)).eval();
    assert_eq!(format!("{result}"), "1", "heaviside(5) should eval to 1");
}

#[test]
fn macro_heaviside_eval_negative() {
    let __ctx = Context::new();
    let result = expr!(heaviside(-3)).eval();
    assert_eq!(format!("{result}"), "0", "heaviside(-3) should eval to 0");
}

#[test]
fn macro_dirac_delta_symbolic() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let e = expr!(dirac_delta(x));
    let s = format!("{e}");
    assert!(
        s.contains("dirac") || s.contains("Dirac") || s.contains("delta") || s.contains("δ"),
        "dirac_delta(x) should display appropriately: {s}"
    );
}

#[test]
fn macro_lambertw_symbolic() {
    let __ctx = Context::new();
    let x = __ctx.symbol("x");
    let e = expr!(lambertw(x));
    let s = format!("{e}");
    assert!(
        s.contains("lambert") || s.contains("Lambert") || s.contains("W("),
        "lambertw(x) should display appropriately: {s}"
    );
}
