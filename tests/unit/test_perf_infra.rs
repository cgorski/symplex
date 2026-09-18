//! Comprehensive tests for performance and infrastructure changes.
//!
//! Covers:
//!   1. `for_each_child` correctness — verified indirectly via walk infrastructure
//!      (free_symbols, contains) since `for_each_child` is `pub(crate)`.
//!   2. `free_symbols` correctness for various node types.
//!   3. Gruntz counter isolation — sequential limits are reproducible and
//!      don't interfere with each other.
//!   4. Substitution correctness after walk changes.
//!

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. for_each_child correctness (indirect via walk infrastructure)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn walk_finds_all_symbols_in_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let expr = &x + &y + &z;
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        3,
        "x + y + z should have 3 free symbols, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn walk_finds_symbols_in_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let expr = &x * &y * &z;
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        3,
        "x * y * z should have 3 free symbols, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn walk_finds_symbols_in_nested_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().exp().ln();
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "sin(exp(ln(x))) should have 1 free symbol, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn walk_finds_symbols_in_trig_chain() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().cos();
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1, "cos(sin(x)) should have 1 free symbol");
    assert_eq!(format!("{}", syms[0]), "x");
}

#[test]
fn walk_finds_symbols_in_piecewise() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let cond = x.gt(&ctx.int(0));
    let pw = Ex::piecewise(&[(&y, &cond)]);
    let syms = pw.free_symbols();
    assert!(
        !syms.is_empty(),
        "should find at least y or x in piecewise, got {} symbols: {:?}",
        syms.len(),
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
    // y is certainly present as a value branch
    let sym_names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(
        sym_names.contains(&"y".to_string()),
        "piecewise should contain y, got: {sym_names:?}"
    );
}

#[test]
fn walk_contains_finds_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let deep = x.sin().cos().exp().ln().abs();
    assert!(
        deep.contains(&x),
        "should find x in abs(ln(exp(cos(sin(x)))))"
    );
}

#[test]
fn walk_contains_returns_false_for_absent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.sin();
    assert!(!expr.contains(&y), "y should not be in sin(x)");
}

#[test]
fn walk_contains_finds_in_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x + &y;
    assert!(expr.contains(&x), "x + y should contain x");
    assert!(expr.contains(&y), "x + y should contain y");
}

#[test]
fn walk_contains_finds_in_mul() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &y;
    assert!(expr.contains(&x), "x * y should contain x");
    assert!(expr.contains(&y), "x * y should contain y");
}

#[test]
fn walk_contains_finds_in_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.pow(&y);
    assert!(expr.contains(&x), "x^y should contain x");
    assert!(expr.contains(&y), "x^y should contain y");
}

#[test]
fn walk_contains_self_always_true() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.contains(&x), "x should contain itself");

    let expr = x.sin();
    assert!(expr.contains(&expr), "sin(x) should contain itself");
}

#[test]
fn walk_contains_constant_not_in_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    assert!(!x.contains(&five), "x should not contain 5");
}

#[test]
fn walk_finds_symbols_across_mixed_ops() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    // (a + b) * sin(c)
    let expr = &(&a + &b) * &c.sin();
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        3,
        "(a + b) * sin(c) should have 3 free symbols, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn walk_handles_neg_correctly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = -&x;
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1, "-x should have 1 free symbol");
    assert!(expr.contains(&x), "-x should contain x");
}

#[test]
fn walk_handles_abs_correctly() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs();
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1, "|x| should have 1 free symbol");
    assert!(expr.contains(&x), "|x| should contain x");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. free_symbols correctness for various node types
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn free_symbols_empty_for_integer() {
    let ctx = Context::new();
    assert!(
        ctx.int(5).free_symbols().is_empty(),
        "integer 5 should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_pi() {
    let ctx = Context::new();
    assert!(
        ctx.pi().free_symbols().is_empty(),
        "pi should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_e() {
    let ctx = Context::new();
    assert!(
        ctx.e().free_symbols().is_empty(),
        "E should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_zero() {
    let ctx = Context::new();
    assert!(
        ctx.int(0).free_symbols().is_empty(),
        "0 should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_negative_integer() {
    let ctx = Context::new();
    assert!(
        ctx.int(-42).free_symbols().is_empty(),
        "-42 should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_rational() {
    let ctx = Context::new();
    assert!(
        ctx.rational(1, 2).free_symbols().is_empty(),
        "1/2 should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_infinity() {
    let ctx = Context::new();
    assert!(
        ctx.infinity().free_symbols().is_empty(),
        "oo should have no free symbols"
    );
}

#[test]
fn free_symbols_empty_for_neg_infinity() {
    let ctx = Context::new();
    assert!(
        ctx.neg_infinity().free_symbols().is_empty(),
        "-oo should have no free symbols"
    );
}

#[test]
fn free_symbols_single_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let syms = x.free_symbols();
    assert_eq!(syms.len(), 1);
    assert_eq!(format!("{}", syms[0]), "x");
}

#[test]
fn free_symbols_in_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.pow(&y);
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        2,
        "x^y should have 2 free symbols, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_in_powi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3);
    let syms = expr.free_symbols();
    assert_eq!(syms.len(), 1, "x^3 should have 1 free symbol");
}

#[test]
fn free_symbols_in_derivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3).diff(&x);
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "d/dx(x^3) = 3x^2 should have 1 free symbol, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_deduplicates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x + &x; // x^2 + x — only one distinct symbol
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "x^2 + x should have 1 unique free symbol, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_deduplicates_across_branches() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x) + cos(x) + x^2 — x appears in three branches
    let expr = &x.sin() + &x.cos() + &x.powi(2);
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "sin(x) + cos(x) + x^2 should have 1 unique free symbol, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_in_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // (x + y)^2 * sin(z) / ln(x)
    let expr = &(&x + &y).powi(2) * &z.sin() / &x.ln();
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        3,
        "complex expression with x, y, z should have 3 free symbols, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_with_constant_mult() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 5 * x — only x is a free symbol, not 5
    let expr = &ctx.int(5) * &x;
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "5*x should have 1 free symbol, got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn free_symbols_with_pi_and_e() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // pi * x + e — only x is free; pi and e are constants
    let expr = &ctx.pi() * &x + &ctx.e();
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "pi*x + e should have 1 free symbol (x), got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Gruntz counter isolation (P3-13)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gruntz_limits_are_reproducible() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();

    // lim(x→∞) 1/x = 0 — compute the same limit twice
    let expr = &ctx.int(1) / &x;
    let r1 = expr.limit(&x, &oo);
    let r2 = expr.limit(&x, &oo);

    // Both should give the same result
    assert_eq!(
        format!("{r1}"),
        format!("{r2}"),
        "same limit computed twice should be reproducible"
    );
}

#[test]
fn gruntz_limit_reproducible_rational_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();

    // lim(x→∞) (3x^2 + 1) / (x^2 - x) = 3
    let numer = &x.powi(2) * 3 + 1;
    let denom = &x.powi(2) - &x;
    let expr = &numer / &denom;

    let r1 = expr.limit(&x, &oo);
    let r2 = expr.limit(&x, &oo);

    assert_eq!(
        format!("{r1}"),
        format!("{r2}"),
        "rational function limit should be reproducible"
    );
}

#[test]
fn sequential_limits_dont_interfere() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();

    // First limit: lim(x→∞) exp(-x) = 0
    let expr1 = (-&x).exp();
    let r1 = expr1.limit(&x, &oo);

    // Second limit: lim(x→∞) 1/x = 0
    let expr2 = &ctx.int(1) / &x;
    let r2 = expr2.limit(&x, &oo);

    // Each should give the correct result independently
    assert_eq!(format!("{r1}"), "0", "exp(-x) as x→∞ should be 0");
    assert_eq!(format!("{r2}"), "0", "1/x as x→∞ should be 0");
}

#[test]
fn sequential_different_limits_no_cross_contamination() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();

    // First: lim(x→∞) 5/x = 0
    let expr1 = &ctx.int(5) / &x;
    let r1 = expr1.limit(&x, &oo);

    // Second: lim(x→∞) constant = 5
    let r2 = ctx.int(5).limit(&x, &oo);

    // Third: lim(x→∞) x/(x+1) = 1
    let expr3 = &x / &(&x + 1);
    let r3 = expr3.limit(&x, &oo);

    assert_eq!(format!("{r1}"), "0", "5/x as x→∞ should be 0");
    assert_eq!(format!("{r2}"), "5", "constant 5 as x→∞ should be 5");
    assert_eq!(format!("{r3}"), "1", "x/(x+1) as x→∞ should be 1");
}

#[test]
fn gruntz_limit_finite_then_infinite_no_interference() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // First: finite limit — lim(x→2) x^2 = 4
    let v1 = x.powi(2).limit(&x, &ctx.int(2));
    assert_eq!(format!("{v1}"), "4", "x^2 as x→2 should be 4");

    // Second: infinite limit — lim(x→∞) 1/x = 0
    let oo = ctx.infinity();
    let expr = &ctx.int(1) / &x;
    let v2 = expr.limit(&x, &oo);
    assert_eq!(format!("{v2}"), "0", "1/x as x→∞ should be 0");

    // Third: finite again — lim(x→0) sin(x)/x = 1
    let sinc = &x.sin() / &x;
    let v3 = sinc.limit(&x, &ctx.int(0));
    assert_eq!(format!("{v3}"), "1", "sin(x)/x as x→0 should be 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Substitution correctness after walk changes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn subs_in_complex_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &(&x.sin() + &x.cos()) * &x.exp();
    let result = expr.subs(&x, &y);
    let syms = result.free_symbols();
    let sym_names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(
        sym_names.contains(&"y".to_string()),
        "substitution should replace x with y, got symbols: {sym_names:?}"
    );
    assert!(
        !sym_names.contains(&"x".to_string()),
        "no x should remain after substitution, got symbols: {sym_names:?}"
    );
}

#[test]
fn subs_symbol_for_symbol() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let result = x.subs(&x, &y);
    assert_eq!(format!("{result}"), "y", "x[x→y] should be y");
}

#[test]
fn subs_no_match_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let expr = x.sin();
    let result = expr.subs(&y, &z);
    let s = format!("{result}");
    assert!(s.contains('x'), "sin(x) with y→z should remain sin(x): {s}");
    assert!(
        !s.contains('z'),
        "z should not appear after non-matching subs: {s}"
    );
}

#[test]
fn subs_in_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x + &ctx.int(1);
    let result = expr.subs(&x, &y);
    let syms = result.free_symbols();
    let sym_names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(
        sym_names.contains(&"y".to_string()),
        "after x→y in x+1, should contain y: {sym_names:?}"
    );
    assert!(
        !sym_names.contains(&"x".to_string()),
        "after x→y in x+1, should not contain x: {sym_names:?}"
    );
}

#[test]
fn subs_in_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &ctx.int(3);
    let result = expr.subs(&x, &y);
    let syms = result.free_symbols();
    let sym_names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(
        sym_names.contains(&"y".to_string()),
        "after x→y in 3*x, should contain y: {sym_names:?}"
    );
}

#[test]
fn subs_in_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.powi(2);
    let result = expr.subs(&x, &y);
    let s = format!("{result}");
    assert!(s.contains('y'), "x^2[x→y] should contain y: {s}");
    assert!(!s.contains('x'), "x^2[x→y] should not contain x: {s}");
}

#[test]
fn subs_in_nested_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = x.sin().exp().ln();
    let result = expr.subs(&x, &y);
    let syms = result.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "after substitution, should have 1 free symbol"
    );
    assert_eq!(
        format!("{}", syms[0]),
        "y",
        "the remaining free symbol should be y"
    );
}

#[test]
fn subs_with_constant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &x + &x; // x^2 + x
    let result = expr.subs(&x, &ctx.int(3));
    let s = format!("{result}");
    // After evaluating: 3^2 + 3 = 12
    // It may or may not be evaluated; at minimum x should be gone
    assert!(!s.contains('x'), "after x→3, no x should remain: {s}");
}

#[test]
fn subs_preserves_free_symbols_count() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // (x + y) → substitute x with z, result should have y and z
    let expr = &x + &y;
    let result = expr.subs(&x, &z);
    let syms = result.free_symbols();
    assert_eq!(
        syms.len(),
        2,
        "(x+y)[x→z] should have 2 free symbols (y,z), got: {:?}",
        syms.iter().map(|s| format!("{s}")).collect::<Vec<_>>()
    );
}

#[test]
fn subs_multiple_occurrences() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x*sin(x) + x^2 — x appears multiple times
    let expr = &(&x * &x.sin()) + &x.powi(2);
    let result = expr.subs(&x, &y);
    let syms = result.free_symbols();
    let sym_names: Vec<String> = syms.iter().map(|s| format!("{s}")).collect();
    assert!(
        sym_names.contains(&"y".to_string()),
        "all x occurrences should be replaced with y: {sym_names:?}"
    );
    assert!(
        !sym_names.contains(&"x".to_string()),
        "no x should remain: {sym_names:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. count_ops sanity — verifies walk-based counting works
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn count_ops_atom_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.count_ops(), 0, "a single symbol should have 0 ops");

    let five = ctx.int(5);
    assert_eq!(five.count_ops(), 0, "an integer should have 0 ops");
}

#[test]
fn count_ops_unary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.sin().count_ops(), 1, "sin(x) should have 1 op");
}

#[test]
fn count_ops_binary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).count_ops(), 1, "x + 1 should have 1 op");
}

#[test]
fn count_ops_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2);
    assert_eq!(
        expr.count_ops(),
        2,
        "sin(x)^2 should have 2 ops (Sin + Pow)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Walk + contains edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn contains_integer_in_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let expr = &x + &two;
    // The exact integer node must be found inside the sum
    assert!(expr.contains(&two), "x + 2 should contain the integer 2");
}

#[test]
fn walk_handles_large_sum() {
    let ctx = Context::new();
    // Stress test: sum of many distinct variables
    let vars: Vec<Ex> = (0..50).map(|i| ctx.symbol(&format!("v{i}"))).collect();
    let mut expr = vars[0].clone();
    for v in &vars[1..] {
        expr = &expr + v;
    }
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        50,
        "sum of 50 distinct variables should have 50 free symbols, got {}",
        syms.len()
    );
}

#[test]
fn walk_handles_deep_nesting() {
    let ctx = Context::new();
    // Deep nesting: sin(sin(sin(...sin(x)...)))
    let x = ctx.symbol("x");
    let mut expr = x.clone();
    for _ in 0..100 {
        expr = expr.sin();
    }
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "deeply nested sin should still find 1 free symbol"
    );
    assert!(expr.contains(&x), "deeply nested sin should contain x");
}
