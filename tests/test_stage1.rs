//! Stage 1 integration tests for symplex.
//!
//! These tests verify the foundational layer: arena interning, expression
//! construction, deduplication (hash-consing), sort key ordering, display
//! formatting, and the Stage 1 milestone.

use symplex::prelude::*;

// ─── Arena Construction ───────────────────────────────────────────────────

#[test]
fn arena_new_has_pre_interned_constants() {
    let arena = Arena::new();
    // All pre-interned constants should be distinct ExprIds
    let ids = [
        arena.zero,
        arena.one,
        arena.neg_one,
        arena.pi,
        arena.e_const,
        arena.i_unit,
        arena.infinity,
        arena.neg_infinity,
        arena.nan,
    ];
    for (i, &a) in ids.iter().enumerate() {
        for (j, &b) in ids.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "constants at index {i} and {j} should differ");
            }
        }
    }
}

#[test]
fn arena_node_count_starts_nonzero() {
    let arena = Arena::new();
    // Pre-interned constants contribute to node count
    assert!(
        arena.node_count() >= 9,
        "expected at least 9 pre-interned nodes"
    );
}

// ─── Interning / Hash-Consing ────────────────────────────────────────────

#[test]
fn interning_same_integer_returns_same_id() {
    let mut arena = Arena::new();
    let a = arena.int(42);
    let b = arena.int(42);
    assert_eq!(a, b, "same integer should produce same ExprId");
}

#[test]
fn interning_different_integers_returns_different_ids() {
    let mut arena = Arena::new();
    let a = arena.int(1);
    let b = arena.int(2);
    assert_ne!(a, b);
}

#[test]
fn interning_same_symbol_returns_same_id() {
    let mut arena = Arena::new();
    let a = arena.symbol("x");
    let b = arena.symbol("x");
    assert_eq!(a, b, "same symbol name should produce same ExprId");
}

#[test]
fn interning_different_symbols_returns_different_ids() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let y = arena.symbol("y");
    assert_ne!(x, y);
}

#[test]
fn interning_same_compound_expression_deduplicates() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let two = arena.int(2);

    let pow1 = arena.pow(x, two);
    let pow2 = arena.pow(x, two);
    assert_eq!(pow1, pow2, "structurally identical Pow should deduplicate");

    let add1 = arena.add(&[pow1, arena.one]);
    let add2 = arena.add(&[pow2, arena.one]);
    assert_eq!(add1, add2, "structurally identical Add should deduplicate");
}

#[test]
fn interning_preserves_node_identity() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");

    match arena.node(x) {
        ExprNode::Symbol(sid) => {
            assert_eq!(arena.symbol_name(*sid), "x");
        }
        other => panic!("expected Symbol, got {other:?}"),
    }
}

// ─── Number Side Table ──────────────────────────────────────────────────

#[test]
fn rational_reduces_to_lowest_terms() {
    let mut arena = Arena::new();
    let r = arena.rational(6, 4);
    match arena.node(r) {
        ExprNode::Num(nid) => {
            let val = arena.num(*nid);
            assert_eq!(*val.numer(), 3.into());
            assert_eq!(*val.denom(), 2.into());
        }
        other => panic!("expected Num, got {other:?}"),
    }
}

#[test]
fn integer_has_denominator_one() {
    let mut arena = Arena::new();
    let n = arena.int(7);
    match arena.node(n) {
        ExprNode::Num(nid) => {
            let val = arena.num(*nid);
            assert!(val.is_integer(), "7 should have denominator 1");
        }
        other => panic!("expected Num, got {other:?}"),
    }
}

// ─── Edge Cases in Construction ─────────────────────────────────────────

#[test]
fn add_zero_args_returns_zero() {
    let mut arena = Arena::new();
    let result = arena.add(&[]);
    assert_eq!(result, arena.zero);
}

#[test]
fn add_one_arg_returns_that_arg() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let result = arena.add(&[x]);
    assert_eq!(result, x);
}

#[test]
fn mul_zero_args_returns_one() {
    let mut arena = Arena::new();
    let result = arena.mul(&[]);
    assert_eq!(result, arena.one);
}

#[test]
fn mul_one_arg_returns_that_arg() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let result = arena.mul(&[x]);
    assert_eq!(result, x);
}

// ─── Sort Key Ordering ──────────────────────────────────────────────────

#[test]
fn sort_key_numbers_before_symbols() {
    let mut arena = Arena::new();
    let n = arena.int(5);
    let x = arena.symbol("x");
    assert!(
        arena.sort_key(n) < arena.sort_key(x),
        "numbers should sort before symbols"
    );
}

#[test]
fn sort_key_symbols_alphabetical() {
    let mut arena = Arena::new();
    let a = arena.symbol("a");
    let b = arena.symbol("b");
    let z = arena.symbol("z");
    assert!(arena.sort_key(a) < arena.sort_key(b));
    assert!(arena.sort_key(b) < arena.sort_key(z));
}

// ─── Display ────────────────────────────────────────────────────────────

#[test]
fn display_integer() {
    let mut arena = Arena::new();
    let n = arena.int(42);
    assert_eq!(arena.display(n).to_string(), "42");
}

#[test]
fn display_negative_integer() {
    let mut arena = Arena::new();
    let n = arena.int(-3);
    assert_eq!(arena.display(n).to_string(), "-3");
}

#[test]
fn display_rational() {
    let mut arena = Arena::new();
    let r = arena.rational(1, 2);
    assert_eq!(arena.display(r).to_string(), "1/2");
}

#[test]
fn display_symbol() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    assert_eq!(arena.display(x).to_string(), "x");
}

#[test]
fn display_constants() {
    let arena = Arena::new();
    assert_eq!(arena.display(arena.pi).to_string(), "pi");
    assert_eq!(arena.display(arena.e_const).to_string(), "E");
    assert_eq!(arena.display(arena.i_unit).to_string(), "I");
    assert_eq!(arena.display(arena.infinity).to_string(), "oo");
    assert_eq!(arena.display(arena.neg_infinity).to_string(), "-oo");
    assert_eq!(arena.display(arena.nan).to_string(), "nan");
}

#[test]
fn display_simple_add() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let y = arena.symbol("y");
    let sum = arena.add(&[x, y]);
    assert_eq!(arena.display(sum).to_string(), "x + y");
}

#[test]
fn display_simple_mul() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let y = arena.symbol("y");
    let prod = arena.mul(&[x, y]);
    assert_eq!(arena.display(prod).to_string(), "x*y");
}

#[test]
fn display_pow() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let two = arena.int(2);
    let p = arena.pow(x, two);
    assert_eq!(arena.display(p).to_string(), "x**2");
}

#[test]
fn display_neg() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let neg_x = arena.neg(x);
    assert_eq!(arena.display(neg_x).to_string(), "-x");
}

#[test]
fn display_function_sin() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let s = arena.sin(x);
    assert_eq!(arena.display(s).to_string(), "sin(x)");
}

#[test]
fn display_nested_pow_in_add() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let two = arena.int(2);
    let x_sq = arena.pow(x, two);
    let sum = arena.add(&[x_sq, x]);
    assert_eq!(arena.display(sum).to_string(), "x + x**2");
}

#[test]
fn display_add_base_in_pow_gets_parens() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let one = arena.one;
    let sum = arena.add(&[x, one]);
    let two = arena.int(2);
    let p = arena.pow(sum, two);
    assert_eq!(arena.display(p).to_string(), "(1 + x)**2");
}

// ─── The Stage 1 Milestone ──────────────────────────────────────────────

#[test]
fn milestone_x_squared_plus_2x_plus_1() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let two = arena.int(2);

    let x_sq = arena.pow(x, two);
    let two_x = arena.mul(&[two, x]);
    let expr = arena.add(&[x_sq, two_x, arena.one]);

    assert_eq!(arena.display(expr).to_string(), "1 + x**2 + 2*x");
}

#[test]
fn milestone_nested_expression() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let two = arena.int(2);
    let three = arena.int(3);

    // 3*sin(x**2) + 1
    let x_sq = arena.pow(x, two);
    let sin_x_sq = arena.sin(x_sq);
    let three_sin = arena.mul(&[three, sin_x_sq]);
    let expr = arena.add(&[three_sin, arena.one]);

    assert_eq!(arena.display(expr).to_string(), "1 + 3*sin(x**2)");
}

#[test]
fn milestone_neg_in_add_displays_as_minus() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let three = arena.int(3);
    let neg_three = arena.neg(three);
    let expr = arena.add(&[x, neg_three]);

    assert_eq!(arena.display(expr).to_string(), "-3 + x");
}

// ─── ExprNode Properties ────────────────────────────────────────────────

#[test]
fn atoms_are_atoms() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let n = arena.int(5);

    assert!(arena.node(x).is_atom());
    assert!(arena.node(n).is_atom());
    assert!(arena.node(arena.pi).is_atom());
    assert!(arena.node(arena.nan).is_atom());
}

#[test]
fn composites_are_not_atoms() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let y = arena.symbol("y");

    let sum = arena.add(&[x, y]);
    let prod = arena.mul(&[x, y]);
    let two = arena.int(2);
    let p = arena.pow(x, two);
    let s = arena.sin(x);
    let neg = arena.neg(x);

    assert!(!arena.node(sum).is_atom());
    assert!(!arena.node(prod).is_atom());
    assert!(!arena.node(p).is_atom());
    assert!(!arena.node(s).is_atom());
    assert!(!arena.node(neg).is_atom());
}

#[test]
fn children_returns_correct_ids() {
    let mut arena = Arena::new();
    let x = arena.symbol("x");
    let y = arena.symbol("y");
    let sum = arena.add(&[x, y]);

    let children = arena.children(sum);
    assert_eq!(children.len(), 2);
    assert_eq!(children[0], x);
    assert_eq!(children[1], y);
}
