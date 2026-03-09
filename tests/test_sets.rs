//! Tests for Wave ζ — the Set type system.
//!
//! Covers: SetValued sort, EmptySet, UniversalSet, Interval, FiniteSet,
//! SetUnion, SetIntersection, SetComplement, canonical constructors,
//! display formatting, and public API on SetEx / Ex / Context.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Basic atoms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn empty_set_display() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    assert_eq!(format!("{e}"), "EmptySet");
}

#[test]
fn empty_set_expr_type() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    assert_eq!(e.as_ex().expr_type(), ExprType::Set);
}

#[test]
fn empty_set_is_eq_to_itself() {
    let ctx = Context::new();
    let a = ctx.empty_set();
    let b = ctx.empty_set();
    assert_eq!(a, b);
}

#[test]
fn universal_set_display() {
    let ctx = Context::new();
    let u = ctx.universal_set();
    assert_eq!(format!("{u}"), "UniversalSet");
}

#[test]
fn universal_set_expr_type() {
    let ctx = Context::new();
    let u = ctx.universal_set();
    assert_eq!(u.as_ex().expr_type(), ExprType::Set);
}

// ═══════════════════════════════════════════════════════════════════════════
// Intervals — display
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_closed_display() {
    let ctx = Context::new();
    let a = ctx.int(0);
    let b = ctx.int(1);
    let i = ctx.interval(&a, &b, false, false);
    let s = format!("{i}");
    assert!(
        s.contains('[') && s.contains(']'),
        "closed interval should use square brackets: {s}"
    );
    assert!(s.contains('0'), "should contain start: {s}");
    assert!(s.contains('1'), "should contain end: {s}");
}

#[test]
fn interval_open_display() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), true, true);
    let s = format!("{i}");
    assert!(
        s.contains('(') && s.contains(')'),
        "open interval should use round brackets: {s}"
    );
}

#[test]
fn interval_half_open_left_display() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), true, false);
    let s = format!("{i}");
    assert!(
        s.contains('(') && s.contains(']'),
        "half-open-left interval: {s}"
    );
}

#[test]
fn interval_half_open_right_display() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, true);
    let s = format!("{i}");
    assert!(
        s.contains('[') && s.contains(')'),
        "half-open-right interval: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Intervals — degenerate cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn degenerate_interval_start_gt_end_is_empty() {
    let ctx = Context::new();
    // Interval(3, 1) should be EmptySet (start > end)
    let i = ctx.interval(&ctx.int(3), &ctx.int(1), false, false);
    assert_eq!(format!("{i}"), "EmptySet");
}

#[test]
fn degenerate_interval_start_gt_end_open_is_empty() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(5), &ctx.int(2), true, true);
    assert_eq!(format!("{i}"), "EmptySet");
}

#[test]
fn point_interval_closed_becomes_finite_set() {
    let ctx = Context::new();
    // Interval(2, 2, closed, closed) → FiniteSet({2})
    let i = ctx.interval(&ctx.int(2), &ctx.int(2), false, false);
    let s = format!("{i}");
    assert!(
        s.contains('{') && s.contains('}'),
        "point interval should become a finite set: {s}"
    );
    assert!(s.contains('2'), "should contain the point value: {s}");
}

#[test]
fn point_interval_open_becomes_empty() {
    let ctx = Context::new();
    // Interval(2, 2, open, open) → EmptySet
    let i = ctx.interval(&ctx.int(2), &ctx.int(2), true, true);
    assert_eq!(format!("{i}"), "EmptySet");
}

#[test]
fn point_interval_half_open_left_becomes_empty() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(2), &ctx.int(2), true, false);
    assert_eq!(format!("{i}"), "EmptySet");
}

#[test]
fn point_interval_half_open_right_becomes_empty() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(2), &ctx.int(2), false, true);
    assert_eq!(format!("{i}"), "EmptySet");
}

#[test]
fn interval_with_symbolic_endpoints() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let i = ctx.interval(&x, &y, false, false);
    let s = format!("{i}");
    assert!(
        s.contains('[') && s.contains(']'),
        "symbolic interval should display as closed: {s}"
    );
    assert!(s.contains('x') && s.contains('y'), "symbolic interval: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Finite sets
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn finite_set_display() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
    let display = format!("{s}");
    assert!(
        display.contains('{') && display.contains('}'),
        "finite set: {display}"
    );
    assert!(display.contains('1'), "contains 1: {display}");
    assert!(display.contains('2'), "contains 2: {display}");
    assert!(display.contains('3'), "contains 3: {display}");
}

#[test]
fn finite_set_deduplicates() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(1), ctx.int(2)]);
    let display = format!("{s}");
    // Count the commas — should be exactly 1 for {1, 2}
    let comma_count = display.matches(',').count();
    assert_eq!(comma_count, 1, "duplicates should be removed: {display}");
}

#[test]
fn finite_set_sorts_elements() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(3), ctx.int(1), ctx.int(2)]);
    let display = format!("{s}");
    // Elements should be sorted by sort key (numbers sorted numerically)
    let pos_1 = display.find('1').unwrap();
    let pos_2 = display.find('2').unwrap();
    let pos_3 = display.find('3').unwrap();
    assert!(pos_1 < pos_2, "1 before 2: {display}");
    assert!(pos_2 < pos_3, "2 before 3: {display}");
}

#[test]
fn finite_set_empty_is_empty_set() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[]);
    assert_eq!(format!("{s}"), "EmptySet");
}

#[test]
fn finite_set_single_element() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(42)]);
    let display = format!("{s}");
    assert!(display.contains("42"), "single element: {display}");
    assert!(display.contains('{'), "has braces: {display}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Union
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn union_of_intervals() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let u = a.union(&b);
    let s = format!("{u}");
    // Should show union with ∪
    assert!(s.contains('∪'), "union display should contain ∪: {s}");
}

#[test]
fn union_with_empty_set_is_identity() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let e = ctx.empty_set();
    let result = a.union(&e);
    // Union of A and EmptySet should be A
    assert_eq!(
        format!("{result}"),
        format!("{a}"),
        "union with empty set is identity"
    );
}

#[test]
fn union_of_empty_sets_is_empty() {
    let ctx = Context::new();
    let e1 = ctx.empty_set();
    let e2 = ctx.empty_set();
    let result = e1.union(&e2);
    assert_eq!(format!("{result}"), "EmptySet");
}

#[test]
fn union_with_universal_set() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let univ = ctx.universal_set();
    let result = a.union(&univ);
    assert_eq!(
        format!("{result}"),
        "UniversalSet",
        "union with universal set is universal set"
    );
}

#[test]
fn union_flattens_nested() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let c = ctx.interval(&ctx.int(4), &ctx.int(5), false, false);
    let ab = a.union(&b);
    let abc = ab.union(&c);
    let s = format!("{abc}");
    // Should be flattened, showing all three intervals joined by ∪
    let union_count = s.matches('∪').count();
    assert_eq!(
        union_count, 2,
        "nested union should be flattened to 3 children: {s}"
    );
}

#[test]
fn union_deduplicates() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = a.union(&a);
    // Union of A with itself should be just A
    let s = format!("{result}");
    assert!(
        !s.contains('∪'),
        "union of A with itself should collapse: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Intersection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn intersection_with_empty() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let e = ctx.empty_set();
    let result = a.intersection(&e);
    assert_eq!(format!("{result}"), "EmptySet");
}

#[test]
fn intersection_with_universal_set_is_identity() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let univ = ctx.universal_set();
    let result = a.intersection(&univ);
    assert_eq!(
        format!("{result}"),
        format!("{a}"),
        "intersection with universal set is identity"
    );
}

#[test]
fn intersection_of_intervals() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let result = a.intersection(&b);
    let s = format!("{result}");
    // Should show intersection with ∩
    assert!(
        s.contains('∩'),
        "intersection display should contain ∩: {s}"
    );
}

#[test]
fn intersection_deduplicates() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = a.intersection(&a);
    // Intersection of A with itself should be just A
    let s = format!("{result}");
    assert!(
        !s.contains('∩'),
        "intersection of A with itself should collapse: {s}"
    );
}

#[test]
fn intersection_flattens_nested() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let c = ctx.interval(&ctx.int(4), &ctx.int(5), false, false);
    let ab = a.intersection(&b);
    let abc = ab.intersection(&c);
    let s = format!("{abc}");
    let inter_count = s.matches('∩').count();
    assert_eq!(
        inter_count, 2,
        "nested intersection should be flattened to 3 children: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Complement
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complement_display() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let result = a.complement(&b);
    let s = format!("{result}");
    assert!(
        s.contains('\\'),
        "complement display should contain backslash: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Reals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reals_display() {
    let ctx = Context::new();
    let r = ctx.reals();
    let s = format!("{r}");
    assert!(s.contains("-oo"), "reals should show -oo: {s}");
    assert!(s.contains("oo"), "reals should show oo: {s}");
    assert!(
        s.contains('(') && s.contains(')'),
        "reals should be open interval: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Ex methods: closed_interval, open_interval
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ex_closed_interval() {
    let ctx = Context::new();
    let i = ctx.int(0).closed_interval(&ctx.int(10));
    let s = format!("{i}");
    assert!(s.contains('[') && s.contains(']'), "closed: {s}");
    assert!(s.contains('0') && s.contains("10"), "endpoints: {s}");
}

#[test]
fn ex_open_interval() {
    let ctx = Context::new();
    let i = ctx.int(-1).open_interval(&ctx.int(1));
    let s = format!("{i}");
    assert!(s.contains('(') && s.contains(')'), "open: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// SetEx into_ex / as_ex escape hatches
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn set_ex_into_ex() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let display_before = format!("{i}");
    let as_numeric: Ex = i.into_ex();
    // Display should be the same
    assert_eq!(format!("{as_numeric}"), display_before);
}

#[test]
fn set_ex_as_ex() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let borrowed: Ex = i.as_ex();
    assert_eq!(format!("{borrowed}"), format!("{i}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Expr type classification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_expr_type() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    assert_eq!(i.as_ex().expr_type(), ExprType::Set);
}

#[test]
fn finite_set_expr_type() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2)]);
    assert_eq!(s.as_ex().expr_type(), ExprType::Set);
}

#[test]
fn union_expr_type() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let u = a.union(&b);
    assert_eq!(u.as_ex().expr_type(), ExprType::Set);
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization roundtrip (ExprTree)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tree_roundtrip_empty_set() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    let tree = e.as_ex().to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), "EmptySet");
}

#[test]
fn tree_roundtrip_interval() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(5), true, false);
    let original = format!("{i}");
    let tree = i.as_ex().to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), original);
}

#[test]
fn tree_roundtrip_finite_set() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
    let original = format!("{s}");
    let tree = s.as_ex().to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), original);
}

#[test]
fn tree_roundtrip_union() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let u = a.union(&b);
    let original = format!("{u}");
    let tree = u.as_ex().to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), original);
}

#[test]
fn tree_roundtrip_universal_set() {
    let ctx = Context::new();
    let u = ctx.universal_set();
    let tree = u.as_ex().to_tree();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), "UniversalSet");
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn json_roundtrip_interval() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let original = format!("{i}");
    let json = i.as_ex().to_json().unwrap();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(format!("{back}"), original);
}

#[test]
fn json_roundtrip_empty_set() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    let json = e.as_ex().to_json().unwrap();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(format!("{back}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// Interaction with existing systems
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_of_set_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    // Differentiating a set-valued expression should give zero
    let d = i.as_ex().diff(&x);
    assert_eq!(format!("{d}"), "0");
}

#[test]
fn diff_of_empty_set_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = ctx.empty_set();
    let d = e.as_ex().diff(&x);
    assert_eq!(format!("{d}"), "0");
}

#[test]
fn evalf_of_set_errors() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = i.as_ex().eval_decimal(15);
    assert!(result.is_err(), "evalf on a set should error");
}

#[test]
fn evalf_of_empty_set_errors() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    let result = e.as_ex().eval_decimal(15);
    assert!(result.is_err(), "evalf on empty set should error");
}

#[test]
fn expand_of_set_is_identity() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let u = a.union(&b);
    let expanded = u.as_ex().expand();
    assert_eq!(
        format!("{expanded}"),
        format!("{u}"),
        "expand on a set should be identity"
    );
}

#[test]
fn eval_of_set_is_identity() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let evaled = i.as_ex().eval();
    assert_eq!(
        format!("{evaled}"),
        format!("{i}"),
        "eval on a set should be identity"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// args / count_ops / free_symbols
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn empty_set_args_is_empty() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    assert_eq!(e.as_ex().args().len(), 0);
}

#[test]
fn interval_args_has_two_children() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let args = i.as_ex().args();
    assert_eq!(args.len(), 2, "interval has 2 children");
}

#[test]
fn finite_set_args_count() {
    let ctx = Context::new();
    let s = ctx.finite_set(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
    let args = s.as_ex().args();
    assert_eq!(args.len(), 3, "finite set with 3 elements has 3 children");
}

#[test]
fn interval_with_symbol_has_free_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.interval(&x, &ctx.int(1), false, false);
    let free = i.as_ex().free_symbols();
    let free_names: Vec<String> = free.iter().map(|e| format!("{e}")).collect();
    assert!(
        free_names.contains(&"x".to_string()),
        "x is a free symbol: {:?}",
        free_names
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Substitution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_in_interval_endpoint() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.interval(&x, &ctx.int(10), false, false);
    let result = i.as_ex().subs(&x, &ctx.int(0));
    let s = format!("{result}");
    assert!(s.contains('0') && s.contains("10"), "substituted: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Rational interval endpoints
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_rational_endpoints() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let three_halves = ctx.rational(3, 2);
    let i = ctx.interval(&half, &three_halves, false, true);
    let s = format!("{i}");
    assert!(s.contains("1/2"), "start: {s}");
    assert!(s.contains("3/2"), "end: {s}");
    assert!(s.contains('['), "closed left: {s}");
    assert!(s.contains(')'), "open right: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge: negative endpoint ordering
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_negative_endpoints() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(-3), &ctx.int(-1), false, false);
    let s = format!("{i}");
    assert!(s.contains('['), "has bracket: {s}");
    assert!(s.contains("-3"), "has -3: {s}");
    assert!(s.contains("-1"), "has -1: {s}");
}

#[test]
fn interval_negative_reversed_is_empty() {
    let ctx = Context::new();
    // -1 > -3 is true, so Interval(-1, -3) is start > end → empty
    let i = ctx.interval(&ctx.int(-1), &ctx.int(-3), false, false);
    assert_eq!(format!("{i}"), "EmptySet");
}

// ═══════════════════════════════════════════════════════════════════════════
// Pre-interned constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn empty_set_is_pre_interned() {
    let ctx = Context::new();
    let e1 = ctx.empty_set();
    let e2 = ctx.empty_set();
    // Should be the same ExprId (pre-interned)
    assert_eq!(e1, e2);
}

#[test]
fn universal_set_is_pre_interned() {
    let ctx = Context::new();
    let u1 = ctx.universal_set();
    let u2 = ctx.universal_set();
    assert_eq!(u1, u2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiple contexts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sets_in_separate_contexts() {
    let ctx1 = Context::new();
    let ctx2 = Context::new();
    let i1 = ctx1.interval(&ctx1.int(0), &ctx1.int(1), false, false);
    let i2 = ctx2.interval(&ctx2.int(0), &ctx2.int(1), false, false);
    // Different contexts, but same display
    assert_eq!(format!("{i1}"), format!("{i2}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Stress: larger finite set
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn finite_set_many_elements() {
    let ctx = Context::new();
    let elems: Vec<Ex> = (0..20).map(|n| ctx.int(n)).collect();
    let s = ctx.finite_set(&elems);
    let display = format!("{s}");
    assert!(display.starts_with('{'), "starts with brace: {display}");
    assert!(display.ends_with('}'), "ends with brace: {display}");
    // Should have 19 commas for 20 elements
    assert_eq!(display.matches(',').count(), 19, "20 elements: {display}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Union/Intersection lattice properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn union_identity_element_is_empty_set() {
    // union(A, EmptySet) = A for any set A
    let ctx = Context::new();
    let a = ctx.finite_set(&[ctx.int(1), ctx.int(2)]);
    let e = ctx.empty_set();
    let result = a.union(&e);
    assert_eq!(format!("{result}"), format!("{a}"));
}

#[test]
fn union_absorbing_element_is_universal() {
    // union(A, UniversalSet) = UniversalSet for any set A
    let ctx = Context::new();
    let a = ctx.finite_set(&[ctx.int(1), ctx.int(2)]);
    let u = ctx.universal_set();
    let result = a.union(&u);
    assert_eq!(format!("{result}"), "UniversalSet");
}

#[test]
fn intersection_identity_element_is_universal() {
    // intersection(A, UniversalSet) = A for any set A
    let ctx = Context::new();
    let a = ctx.finite_set(&[ctx.int(1), ctx.int(2)]);
    let u = ctx.universal_set();
    let result = a.intersection(&u);
    assert_eq!(format!("{result}"), format!("{a}"));
}

#[test]
fn intersection_absorbing_element_is_empty() {
    // intersection(A, EmptySet) = EmptySet for any set A
    let ctx = Context::new();
    let a = ctx.finite_set(&[ctx.int(1), ctx.int(2)]);
    let e = ctx.empty_set();
    let result = a.intersection(&e);
    assert_eq!(format!("{result}"), "EmptySet");
}

#[test]
fn union_idempotent() {
    // union(A, A) = A
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = a.union(&a);
    assert_eq!(format!("{result}"), format!("{a}"));
}

#[test]
fn intersection_idempotent() {
    // intersection(A, A) = A
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = a.intersection(&a);
    assert_eq!(format!("{result}"), format!("{a}"));
}

#[test]
fn union_commutative_display() {
    // union(A, B) should have the same display regardless of order
    // (because we canonically sort children)
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let ab = a.union(&b);
    let ba = b.union(&a);
    assert_eq!(format!("{ab}"), format!("{ba}"), "union is commutative");
}

#[test]
fn intersection_commutative_display() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let ab = a.intersection(&b);
    let ba = b.intersection(&a);
    assert_eq!(
        format!("{ab}"),
        format!("{ba}"),
        "intersection is commutative"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Complement specifics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complement_of_different_sets() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(10), false, false);
    let b = ctx.interval(&ctx.int(3), &ctx.int(7), true, true);
    let result = a.complement(&b);
    let s = format!("{result}");
    // Should show "A \ B" style
    assert!(s.contains('\\'), "complement: {s}");
}

#[test]
fn complement_expr_type() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let result = a.complement(&b);
    assert_eq!(result.as_ex().expr_type(), ExprType::Set);
}

// ═══════════════════════════════════════════════════════════════════════════
// Mixed: union of finite sets and intervals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn union_of_finite_set_and_interval() {
    let ctx = Context::new();
    let fs = ctx.finite_set(&[ctx.int(5)]);
    let iv = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let u = fs.union(&iv);
    let s = format!("{u}");
    assert!(s.contains('∪'), "mixed union: {s}");
    assert!(s.contains('{'), "contains finite set: {s}");
    assert!(s.contains('['), "contains interval: {s}");
}

#[test]
fn intersection_of_finite_set_and_interval() {
    let ctx = Context::new();
    let fs = ctx.finite_set(&[ctx.int(5)]);
    let iv = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let result = fs.intersection(&iv);
    let s = format!("{result}");
    assert!(s.contains('∩'), "mixed intersection: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Nested set operations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn complement_inside_union() {
    let ctx = Context::new();
    let a = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let b = ctx.interval(&ctx.int(2), &ctx.int(3), false, false);
    let c = ctx.interval(&ctx.int(4), &ctx.int(5), false, false);
    let comp = a.complement(&b);
    let u = comp.union(&c);
    let s = format!("{u}");
    assert!(s.contains('∪'), "outer union: {s}");
    assert!(s.contains('\\'), "inner complement: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// contains check on Ex
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interval_contains_its_endpoints() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let one = ctx.int(1);
    let i = ctx.interval(&zero, &one, false, false);
    // The interval should "contain" the subexpression 0 and 1
    // (structural containment, not set membership)
    let ex = i.as_ex();
    assert!(ex.contains(&zero), "interval contains endpoint 0");
    assert!(ex.contains(&one), "interval contains endpoint 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// count_ops
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn count_ops_of_empty_set() {
    let ctx = Context::new();
    let e = ctx.empty_set();
    let ops = e.as_ex().count_ops();
    assert_eq!(ops, 0, "empty set is an atom with 0 ops");
}

#[test]
fn count_ops_of_interval() {
    let ctx = Context::new();
    let i = ctx.interval(&ctx.int(0), &ctx.int(1), false, false);
    let ops = i.as_ex().count_ops();
    assert!(ops >= 1, "interval should count as at least 1 op: {ops}");
}
