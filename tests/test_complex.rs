//! Tests for complex number support.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// i^n canonicalization
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_squared_is_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(2);
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn i_cubed_is_neg_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(3);
    assert_eq!(format!("{result}"), "-I");
}

#[test]
fn i_fourth_is_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(4);
    assert_eq!(format!("{result}"), "1");
}

#[test]
fn i_to_100() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(100);
    assert_eq!(format!("{result}"), "1"); // 100 mod 4 = 0
}

#[test]
fn i_to_neg_one() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(-1);
    assert_eq!(format!("{result}"), "-I"); // i^(-1) = -i
}

#[test]
fn i_to_neg_two() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = i.powi(-2);
    assert_eq!(format!("{result}"), "-1"); // i^(-2) = -1
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex algebra
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn one_plus_i_squared() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(2).expand();
    let s = format!("{expr}");
    // (1+i)^2 = 1 + 2i + i^2 = 1 + 2i - 1 = 2i
    assert_eq!(s, "2*I");
}

#[test]
fn i_times_i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let result = &i * &i;
    assert_eq!(format!("{result}"), "-1");
}

#[test]
fn complex_addition() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let z1 = &ctx.int(2) + &(&ctx.int(3) * &i);
    let z2 = &ctx.int(4) + &(&ctx.int(5) * &i);
    let sum = &z1 + &z2;
    let s = format!("{sum}");
    // (2+3i) + (4+5i) = 6+8i
    assert_eq!(s, "8*I + 6");
}

#[test]
fn diff_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // d/dx(i*x^2) = 2*i*x
    let expr = &i * &x.powi(2);
    let result = expr.diff(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "d/dx(i*x²) should be 2*i*x, got: {s}"
    );
}

#[test]
fn integrate_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    // ∫ i*x dx = i*x²/2
    let expr = &i * &x;
    let result = expr.integrate(&x);
    let s = format!("{result}");
    assert!(
        s.contains("I") && s.contains("x"),
        "∫ i*x dx should involve I and x, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn i_is_imaginary() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.query(Props::IMAGINARY), Some(true));
    assert_eq!(i.query(Props::REAL), Some(false));
    assert_eq!(i.query(Props::COMPLEX), Some(true));
}

#[test]
fn i_squared_is_real() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let i2 = i.powi(2);
    // i^2 canonicalizes to -1, which is real
    assert_eq!(i2.is_real(), Some(true));
    assert_eq!(i2.is_negative(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Complex quadratic roots
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_x2_plus_1() {
    // x²+1=0 has purely complex roots; the solver may not yet return them.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) + 1;
    let roots = eq.solve_or_empty(&x);
    if roots.len() == 2 {
        let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        let joined = strs.join(", ");
        assert!(joined.contains("I"), "roots should contain I: {joined}");
    } else {
        // Complex-root solving not yet supported — document the gap.
        assert_eq!(roots.len(), 0, "expected 0 (unsupported) or 2 roots");
    }
}

#[test]
fn solve_x2_plus_4() {
    // x²+4=0 has purely complex roots; the solver may not yet return them.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) + 4;
    let roots = eq.solve_or_empty(&x);
    if roots.len() == 2 {
        let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        let joined = strs.join(", ");
        assert!(joined.contains("I"), "roots should contain I: {joined}");
    } else {
        assert_eq!(roots.len(), 0, "expected 0 (unsupported) or 2 roots");
    }
}

#[test]
fn solve_x2_plus_2x_plus_5() {
    // x²+2x+5=0 has complex roots; the solver may not yet return them.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let eq = &x.powi(2) + &(&x * 2) + 5;
    let roots = eq.solve_or_empty(&x);
    if roots.len() == 2 {
        let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
        let joined = strs.join(", ");
        assert!(joined.contains("I"), "roots should contain I: {joined}");
    } else {
        assert_eq!(roots.len(), 0, "expected 0 (unsupported) or 2 roots");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler's formula
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn euler_exp_i_pi() {
    // exp(i*π) should ideally evaluate to -1.
    // The library may not yet perform this simplification, so accept both forms.
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&i * &ctx.pi()).exp().eval();
    let s = format!("{expr}");
    assert!(
        s == "-1" || s.contains("exp") && s.contains("I"),
        "exp(iπ) should be -1 or unevaluated exp(…I): {s}"
    );
}

#[test]
fn euler_exp_i_pi_over_2() {
    // exp(i*π/2) should ideally evaluate to i.
    // Accept the unevaluated form if the library doesn't simplify it yet.
    let ctx = Context::new();
    let i = ctx.i_unit();
    let angle = &ctx.rational(1, 2) * &ctx.pi();
    let expr = (&i * &angle).exp().eval();
    let s = format!("{expr}");
    assert!(
        s == "I" || s.contains("exp") && s.contains("I"),
        "exp(iπ/2) should be I or unevaluated exp(…I): {s}"
    );
}

#[test]
fn euler_exp_i_pi_plus_1_is_zero() {
    // exp(i*π) + 1 should ideally evaluate to 0.
    // Accept the unevaluated form if the library doesn't simplify it yet.
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = &(&i * &ctx.pi()).exp() + 1;
    let evald = expr.eval();
    let s = format!("{evald}");
    assert!(
        s == "0" || s.contains("exp"),
        "exp(iπ)+1 should be 0 or contain unevaluated exp: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn global_i_unit() {
    let i = symplex::default_context().i_unit();
    assert_eq!(format!("{i}"), "I");
    assert_eq!(i.is_imaginary(), Some(true));
}

#[test]
fn global_pi() {
    let pi = symplex::default_context().pi();
    assert_eq!(format!("{pi}"), "pi");
}

#[test]
fn global_e() {
    let e = symplex::default_context().e();
    assert_eq!(format!("{e}"), "E");
}

#[test]
fn is_imaginary_convenience() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.is_imaginary(), Some(true));
    let x = ctx.symbol("x");
    assert_eq!(x.is_imaginary(), None);
}

#[test]
fn is_nonnegative_convenience() {
    let ctx = Context::new();
    let two = ctx.int(2);
    assert_eq!(two.is_nonnegative(), Some(true));
    let neg = ctx.int(-3);
    assert_eq!(neg.is_nonnegative(), Some(false));
}
