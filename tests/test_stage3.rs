//! Stage 3 integration tests for symplex.
//!
//! Tests the user-facing API: `Context`, `Ex`, operator overloading,
//! i64 coercion, method chaining, `Display`, and macros.

use symplex::prelude::*;
use symplex::{sym, syms};

// ─── Context Construction ─────────────────────────────────────────────────

#[test]
fn context_new_has_nonzero_node_count() {
    let ctx = Context::new();
    assert!(ctx.node_count() >= 9, "pre-interned constants expected");
}

#[test]
fn context_clone_shares_arena() {
    let ctx1 = Context::new();
    let x = ctx1.symbol("x");
    let ctx2 = ctx1.clone();
    assert_eq!(ctx2.display(&x), "x");
}

#[test]
fn context_display_formats_correctly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(ctx.display(&x), "x");
}

// ─── Ex basics ────────────────────────────────────────────────────────────

#[test]
fn ex_equality_same_context() {
    let ctx = Context::new();
    let x1 = ctx.symbol("x");
    let x2 = ctx.symbol("x");
    assert_eq!(x1, x2);
}

#[test]
fn ex_inequality_different_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert_ne!(x, y);
}

#[test]
fn ex_inequality_different_contexts() {
    let ctx1 = Context::new();
    let ctx2 = Context::new();
    let x1 = ctx1.symbol("x");
    let x2 = ctx2.symbol("x");
    // Different contexts → different CtxId → not equal.
    assert_ne!(x1, x2);
}

#[test]
fn ex_display_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{x}"), "x");
}

#[test]
fn ex_display_integer() {
    let ctx = Context::new();
    let n = ctx.int(42);
    assert_eq!(format!("{n}"), "42");
}

#[test]
fn ex_display_rational() {
    let ctx = Context::new();
    let r = ctx.rational(3, 7);
    assert_eq!(format!("{r}"), "3/7");
}

#[test]
fn ex_display_constants() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.pi()), "pi");
    assert_eq!(format!("{}", ctx.e()), "E");
    assert_eq!(format!("{}", ctx.i_unit()), "I");
    assert_eq!(format!("{}", ctx.infinity()), "oo");
    assert_eq!(format!("{}", ctx.nan()), "nan");
}

#[test]
fn ex_debug_format() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let dbg = format!("{x:?}");
    assert!(
        dbg.starts_with("Ex("),
        "debug should start with Ex(, got: {dbg}"
    );
}

// ─── Operator overloading: Ex op Ex ───────────────────────────────────────

#[test]
fn add_ex_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let sum = &x + &y;
    assert_eq!(format!("{sum}"), "x + y");
}

#[test]
fn mul_ex_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let prod = &x * &y;
    assert_eq!(format!("{prod}"), "x*y");
}

#[test]
fn sub_ex_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let diff = &x - &y;
    assert_eq!(format!("{diff}"), "x - y");
}

#[test]
fn div_ex_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let quot = &x / &y;
    assert_eq!(format!("{quot}"), "x*y**(-1)");
}

#[test]
fn neg_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_x = -&x;
    assert_eq!(format!("{neg_x}"), "-x");
}

#[test]
fn add_owned_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let sum = x + y; // consumes both
    assert_eq!(format!("{sum}"), "x + y");
}

// ─── Operator overloading: Ex op i64 ─────────────────────────────────────

#[test]
fn add_ex_i64() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x + 1;
    assert_eq!(format!("{result}"), "1 + x");
}

#[test]
fn add_i64_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = 1 + &x;
    assert_eq!(format!("{result}"), "1 + x");
}

#[test]
fn mul_ex_i64() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x * 3;
    assert_eq!(format!("{result}"), "3*x");
}

#[test]
fn mul_i64_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = 3 * &x;
    assert_eq!(format!("{result}"), "3*x");
}

#[test]
fn sub_ex_i64() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x - 3;
    assert_eq!(format!("{result}"), "-3 + x");
}

#[test]
fn sub_i64_ex() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = 3 - &x;
    assert_eq!(format!("{result}"), "3 - x");
}

#[test]
fn div_ex_i64() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x / 2;
    assert_eq!(format!("{result}"), "1/2*x");
}

#[test]
fn neg_owned() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = -x;
    assert_eq!(format!("{result}"), "-x");
}

// ─── Canonicalization through operators ───────────────────────────────────

#[test]
fn like_term_collection_via_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x + &x;
    assert_eq!(format!("{result}"), "2*x");
}

#[test]
fn like_term_collection_with_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &x * 2;
    let three_x = &x * 3;
    let result = &two_x + &three_x;
    assert_eq!(format!("{result}"), "5*x");
}

#[test]
fn subtraction_cancellation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x - &x;
    assert!(result.is_zero_structural());
}

#[test]
fn numeric_evaluation_in_operators() {
    let ctx = Context::new();
    let two = ctx.int(2);
    let three = ctx.int(3);
    let result = &two + &three;
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn polynomial_via_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let x_sq = x.powi(2);
    let expr = &x_sq + &x * 2 + 1;
    assert_eq!(format!("{expr}"), "1 + x**2 + 2*x");
}

// ─── Method chaining ─────────────────────────────────────────────────────

#[test]
fn pow_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(3);
    assert_eq!(format!("{result}"), "x**3");
}

#[test]
fn sin_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sin();
    assert_eq!(format!("{result}"), "sin(x)");
}

#[test]
fn cos_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.cos();
    assert_eq!(format!("{result}"), "cos(x)");
}

#[test]
fn chained_sin_of_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).sin();
    assert_eq!(format!("{result}"), "sin(x**2)");
}

#[test]
fn chained_expression_building() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 3*sin(x^2) + 1
    let result = &x.powi(2).sin() * 3 + 1;
    assert_eq!(format!("{result}"), "1 + 3*sin(x**2)");
}

#[test]
fn sqrt_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.sqrt();
    assert_eq!(format!("{result}"), "sqrt(x)");
}

#[test]
fn ln_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.ln();
    assert_eq!(format!("{result}"), "ln(x)");
}

#[test]
fn exp_fn_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.exp_fn();
    assert_eq!(format!("{result}"), "exp(x)");
}

#[test]
fn abs_method() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.abs();
    assert_eq!(format!("{result}"), "abs(x)");
}

// ─── Non-auto-evaluation through operator API ─────────────────────────────

#[test]
fn no_auto_expand_via_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sum = &x + 1;
    let result = sum.powi(2);
    // Should stay as (1 + x)**2, NOT expand.
    assert_eq!(format!("{result}"), "(1 + x)**2");
}

#[test]
fn no_auto_distribute_symbolic_via_operators() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let sum = &x + &y;
    // Symbolic * Add does NOT distribute (that's .expand()).
    let result = &z * &sum;
    assert_eq!(format!("{result}"), "z*(x + y)");
    // But numeric * Add DOES distribute (design choice for correct
    // cancellation in a - a = 0).
    let result2 = &ctx.int(2) * &sum;
    assert_eq!(format!("{result2}"), "2*x + 2*y");
}

#[test]
fn no_auto_eval_sin_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let result = zero.sin();
    assert_eq!(format!("{result}"), "sin(0)");
}

// ─── syms! macro ──────────────────────────────────────────────────────────

#[test]
fn syms_macro_declares_symbols() {
    let ctx = Context::new();
    syms!(ctx; a, b, c);
    let result = &a + &b + &c;
    assert_eq!(format!("{result}"), "a + b + c");
}

#[test]
fn syms_macro_trailing_comma() {
    let ctx = Context::new();
    syms!(ctx; x, y,);
    let result = &x * &y;
    assert_eq!(format!("{result}"), "x*y");
}

// ─── sym! macro ───────────────────────────────────────────────────────────

#[test]
fn sym_macro_creates_symbol() {
    let ctx = Context::new();
    sym!(ctx; t, Positive, Real);
    assert_eq!(format!("{t}"), "t");
    assert_eq!(ctx.query(&t, Props::POSITIVE), Some(true));
    assert_eq!(ctx.query(&t, Props::REAL), Some(true));
}

// ─── Complex expressions ──────────────────────────────────────────────────

#[test]
fn quadratic_expression() {
    let ctx = Context::new();
    syms!(ctx; x);
    let expr = &x.powi(2) + &x * 5 + 6;
    assert_eq!(format!("{expr}"), "6 + x**2 + 5*x");
}

#[test]
fn product_of_symbols() {
    let ctx = Context::new();
    syms!(ctx; a, b, c);
    let result = &(&a * &b) * &c * 2;
    assert_eq!(format!("{result}"), "2*a*b*c");
}

#[test]
fn nested_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let three = ctx.int(3);
    let result = x.pow(&two).pow(&three);
    // (x^2)^3 stays as-is — no power-of-power simplification yet.
    assert_eq!(format!("{result}"), "(x**2)**3");
}

#[test]
fn fraction_expression() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = &(&x + &y) / &(&x - &y);
    let s = format!("{result}");
    // (x + y) / (x - y) = (x + y) * (x - y)^(-1)
    assert!(s.contains("(x + y)"), "should contain (x + y), got: {s}");
}

// ─── Hash and Eq ──────────────────────────────────────────────────────────

#[test]
fn ex_hashable() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let ctx = Context::new();
    let x1 = ctx.symbol("x");
    let x2 = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Same symbol → same hash.
    let hash = |ex: &Ex| {
        let mut h = DefaultHasher::new();
        ex.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash(&x1), hash(&x2));
    // Different symbols → (almost certainly) different hash.
    assert_ne!(hash(&x1), hash(&y));
    // Equality matches hash.
    assert_eq!(x1, x2);
    assert_ne!(x1, y);
}

// ─── Send + Sync ──────────────────────────────────────────────────────────

#[test]
fn ex_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Ex>();
    assert_sync::<Ex>();
}

#[test]
fn context_is_send_and_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Context>();
    assert_sync::<Context>();
}

// ─── Structural predicates ────────────────────────────────────────────────

#[test]
fn is_zero_structural_on_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    assert!(zero.is_zero_structural());
}

#[test]
fn is_zero_structural_on_nonzero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(!x.is_zero_structural());
}

#[test]
fn is_one_structural() {
    let ctx = Context::new();
    let one = ctx.int(1);
    assert!(one.is_one_structural());
    let two = ctx.int(2);
    assert!(!two.is_one_structural());
}

#[test]
fn cancellation_produces_structural_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x - &x;
    assert!(result.is_zero_structural());
}

// ─── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn multiply_by_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &x * &zero;
    assert!(result.is_zero_structural());
}

#[test]
fn add_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x + 0;
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn multiply_by_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x * 1;
    assert_eq!(format!("{result}"), "x");
}

#[test]
#[should_panic(expected = "symbol name cannot be empty")]
fn empty_symbol_name_panics() {
    let ctx = Context::new();
    let _x = ctx.symbol("");
}
