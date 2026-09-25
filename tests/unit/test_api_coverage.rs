//! Integration tests for public Ex methods that were under-tested.
//! Covers: convenience query methods, free_symbols, contains, equals.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Convenience query methods
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_negative_on_negative_int() {
    let ctx = Context::new();
    let neg3 = ctx.int(-3);
    assert_eq!(neg3.is_negative(), Some(true));
}

#[test]
fn is_negative_on_positive_int() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_eq!(five.is_negative(), Some(false));
}

#[test]
fn is_negative_on_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // No assumptions → unknown
    assert_eq!(x.is_negative(), None);
}

#[test]
fn is_negative_on_positive_symbol() {
    let ctx = Context::new();
    let t = ctx.symbol_with("t", &[Assumption::Positive]).unwrap();
    assert_eq!(t.is_negative(), Some(false));
}

#[test]
fn is_real_on_integer() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_eq!(five.is_real(), Some(true));
}

#[test]
fn is_real_on_pi() {
    let ctx = Context::new();
    let pi = ctx.pi();
    assert_eq!(pi.is_real(), Some(true));
}

#[test]
fn is_real_on_imaginary_unit() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.is_real(), Some(false));
}

#[test]
fn is_integer_on_integer() {
    let ctx = Context::new();
    let seven = ctx.int(7);
    assert_eq!(seven.is_integer(), Some(true));
}

#[test]
fn is_integer_on_rational() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    assert_eq!(half.is_integer(), Some(false));
}

#[test]
fn is_integer_on_symbol_with_assumption() {
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]).unwrap();
    assert_eq!(n.is_integer(), Some(true));
}

#[test]
fn is_nonzero_on_nonzero() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_eq!(five.is_nonzero(), Some(true));
}

#[test]
fn is_nonzero_on_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    assert_eq!(zero.is_nonzero(), Some(false));
}

#[test]
fn is_finite_on_integer() {
    let ctx = Context::new();
    let five = ctx.int(5);
    assert_eq!(five.is_finite(), Some(true));
}

#[test]
fn is_finite_on_infinity() {
    let ctx = Context::new();
    let inf = ctx.infinity();
    assert_eq!(inf.is_finite(), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// free_symbols
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_symbols_single_var() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let syms = x.free_symbols();
    assert_eq!(syms.len(), 1);
    assert_eq!(format!("{}", syms[0]), "x");
}

#[test]
fn free_symbols_expression() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &x + &y;
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 2);
    let names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(names.contains(&"x".to_string()));
    assert!(names.contains(&"y".to_string()));
}

#[test]
fn free_symbols_constant_has_none() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let syms = pi.free_symbols();
    assert!(syms.is_empty());
}

#[test]
fn free_symbols_no_duplicates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + x = 2*x — should only have x once
    let expr = &x + &x;
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1);
}

#[test]
fn free_symbols_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).sin();
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1);
    assert_eq!(format!("{}", syms[0]), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// contains
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn contains_self() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.contains(&x));
}

#[test]
fn contains_child() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let expr = &x + &y;
    assert!(expr.contains(&x));
    assert!(expr.contains(&y));
}

#[test]
fn contains_does_not_contain() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    let expr = &x + &y;
    assert!(!expr.contains(&z));
}

#[test]
fn contains_deep() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).sin();
    assert!(expr.contains(&x));
}

// ═══════════════════════════════════════════════════════════════════════════
// equals (improved with expand fallback)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equals_identical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.equals(&x), Some(true));
}

#[test]
fn equals_canonical_same() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x + 1) and (1 + x) should be canonically identical
    let a = &x + 1;
    let b = 1 + &x;
    assert_eq!(a.equals(&b), Some(true));
}

#[test]
fn equals_expand_needed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^2 vs x^2 + 2x + 1 — structurally different, algebraically equal
    let a = (&x + 1).powi(2);
    let b = &x.powi(2) + &x * 2 + 1;
    assert_eq!(a.equals(&b), Some(true));
}

#[test]
fn equals_different() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // x and y are definitely not equal (they're different symbols)
    // But equals() can only confirm equality, not disprove it for symbolic exprs
    // So this may return None
    let result = x.equals(&y);
    assert!(result != Some(true), "x should not equal y");
}
