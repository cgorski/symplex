//! Stage 1 integration tests for symplex.
//!
//! These tests verify the foundational layer through the public API:
//! expression construction, deduplication (hash-consing), display
//! formatting, and the Stage 1 milestone.

use symplex::prelude::*;
use symplex::syms;

// ─── Context Construction ─────────────────────────────────────────────────

#[test]
fn context_has_pre_interned_constants() {
    let ctx = Context::new();
    // Pre-interned constants contribute to node count.
    assert!(
        ctx.node_count() >= 9,
        "expected at least 9 pre-interned nodes, got {}",
        ctx.node_count()
    );
}

#[test]
fn constants_are_distinct() {
    let ctx = Context::new();
    let constants = [
        ctx.int(0),
        ctx.int(1),
        ctx.int(-1),
        ctx.pi(),
        ctx.e(),
        ctx.i_unit(),
        ctx.infinity(),
        ctx.nan(),
    ];
    for (i, a) in constants.iter().enumerate() {
        for (j, b) in constants.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "constants at index {i} and {j} should differ");
            }
        }
    }
}

// ─── Interning / Hash-Consing ────────────────────────────────────────────

#[test]
fn interning_same_integer_returns_same_id() {
    let ctx = Context::new();
    let a = ctx.int(42);
    let b = ctx.int(42);
    assert_eq!(a, b, "same integer should produce equal Ex");
}

#[test]
fn interning_different_integers_returns_different() {
    let ctx = Context::new();
    let a = ctx.int(1);
    let b = ctx.int(2);
    assert_ne!(a, b);
}

#[test]
fn interning_same_symbol_returns_same_id() {
    let ctx = Context::new();
    let a = ctx.symbol("x");
    let b = ctx.symbol("x");
    assert_eq!(a, b, "same symbol name should produce equal Ex");
}

#[test]
fn interning_different_symbols_returns_different() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert_ne!(x, y);
}

#[test]
fn interning_same_compound_expression_deduplicates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);

    let pow1 = x.pow(&two);
    let pow2 = x.pow(&two);
    assert_eq!(pow1, pow2, "structurally identical Pow should deduplicate");

    let one = ctx.int(1);
    let add1 = &pow1 + &one;
    let add2 = &pow2 + &one;
    assert_eq!(add1, add2, "structurally identical Add should deduplicate");
}

// ─── Number Handling ────────────────────────────────────────────────────

#[test]
fn rational_reduces_to_lowest_terms() {
    let ctx = Context::new();
    let r = ctx.rational(6, 4);
    // 6/4 should display as 3/2 (reduced).
    assert_eq!(format!("{r}"), "3/2");
}

#[test]
fn integer_displays_without_denominator() {
    let ctx = Context::new();
    let n = ctx.int(7);
    assert_eq!(format!("{n}"), "7");
}

#[test]
fn negative_integer_displays_with_minus() {
    let ctx = Context::new();
    let n = ctx.int(-3);
    assert_eq!(format!("{n}"), "-3");
}

#[test]
fn rational_displays_as_fraction() {
    let ctx = Context::new();
    let r = ctx.rational(1, 2);
    assert_eq!(format!("{r}"), "1/2");
}

// ─── Edge Cases in Construction ─────────────────────────────────────────

#[test]
fn add_with_zero_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &x + &zero;
    assert_eq!(result, x);
}

#[test]
fn mul_with_one_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x * 1;
    assert_eq!(format!("{result}"), "x");
}

#[test]
fn mul_with_zero_gives_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let result = &x * &zero;
    assert!(result.is_zero_structural());
}

// ─── Display ────────────────────────────────────────────────────────────

#[test]
fn display_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(format!("{x}"), "x");
}

#[test]
fn display_constants() {
    let ctx = Context::new();
    assert_eq!(format!("{}", ctx.pi()), "pi");
    assert_eq!(format!("{}", ctx.e()), "E");
    assert_eq!(format!("{}", ctx.i_unit()), "I");
    assert_eq!(format!("{}", ctx.infinity()), "oo");
    assert_eq!(format!("{}", ctx.nan()), "nan");
}

#[test]
fn display_simple_add() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let sum = &x + &y;
    assert_eq!(format!("{sum}"), "x + y");
}

#[test]
fn display_simple_mul() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let prod = &x * &y;
    assert_eq!(format!("{prod}"), "x*y");
}

#[test]
fn display_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2);
    assert_eq!(format!("{result}"), "x**2");
}

#[test]
fn display_neg() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_x = -&x;
    assert_eq!(format!("{neg_x}"), "-x");
}

#[test]
fn display_function_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.sin();
    assert_eq!(format!("{s}"), "sin(x)");
}

#[test]
fn display_nested_pow_in_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let x_sq = x.powi(2);
    let sum = &x_sq + &x;
    assert_eq!(format!("{sum}"), "x + x**2");
}

#[test]
fn display_add_base_in_pow_gets_parens() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sum = &x + 1;
    let result = sum.powi(2);
    assert_eq!(format!("{result}"), "(1 + x)**2");
}

#[test]
fn display_subtraction_rendering() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x - 3;
    assert_eq!(format!("{result}"), "-3 + x");
}

// ─── The Stage 1 Milestone ──────────────────────────────────────────────

#[test]
fn milestone_x_squared_plus_2x_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.powi(2) + &x * 2 + 1;
    assert_eq!(format!("{expr}"), "1 + x**2 + 2*x");
}

#[test]
fn milestone_nested_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x_sq = x.powi(2).sin();
    let expr = &sin_x_sq * 3 + 1;
    assert_eq!(format!("{expr}"), "1 + 3*sin(x**2)");
}

// ─── Ex Properties ──────────────────────────────────────────────────────

#[test]
fn is_zero_structural_works() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    assert!(zero.is_zero_structural());
    let x = ctx.symbol("x");
    assert!(!x.is_zero_structural());
}

#[test]
fn is_one_structural_works() {
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

// ─── Canonicalization basics ────────────────────────────────────────────

#[test]
fn like_term_collection() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x + &x;
    assert_eq!(format!("{result}"), "2*x");
}

#[test]
fn numeric_evaluation_in_add() {
    let ctx = Context::new();
    let two = ctx.int(2);
    let three = ctx.int(3);
    let result = &two + &three;
    assert_eq!(format!("{result}"), "5");
}

#[test]
fn power_combination_in_mul() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let x2 = x.powi(2);
    let x3 = x.powi(3);
    let result = &x2 * &x3;
    assert_eq!(format!("{result}"), "x**5");
}

#[test]
fn numeric_pow_evaluation() {
    let ctx = Context::new();
    let two = ctx.int(2);
    let result = two.powi(10);
    assert_eq!(format!("{result}"), "1024");
}

#[test]
fn nan_propagation_in_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let nan = ctx.nan();
    let result = &x + &nan;
    assert_eq!(format!("{result}"), "nan");
}

// ─── Sort key ordering ──────────────────────────────────────────────────

#[test]
fn numbers_display_before_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = &x + 3;
    let s = format!("{result}");
    // Number should come first in canonical ordering.
    assert!(s.starts_with('3'), "expected number first, got: {s}");
}

#[test]
fn symbols_display_alphabetically() {
    let ctx = Context::new();
    syms!(ctx; z, a, m);
    let result = &z + &a + &m;
    assert_eq!(format!("{result}"), "a + m + z");
}
