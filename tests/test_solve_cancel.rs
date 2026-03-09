//! Comprehensive public API tests for `Ex::solve()` and `Ex::cancel()`.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// solve() tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_linear() {
    // 2*x - 6 = 0 → x = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x * 2 - 6);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "expected 1 root, got {}", roots.len());
    assert_eq!(format!("{}", roots[0]), "3");
}

#[test]
fn solve_linear_negative() {
    // x + 3 = 0 → x = -3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x + 3);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "expected 1 root, got {}", roots.len());
    assert_eq!(format!("{}", roots[0]), "-3");
}

#[test]
fn solve_linear_rational() {
    // 3*x - 1 = 0 → x = 1/3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x * 3 - 1);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "expected 1 root, got {}", roots.len());
    assert_eq!(format!("{}", roots[0]), "1/3");
}

#[test]
fn solve_quadratic_two_roots() {
    // x^2 - 5*x + 6 = 0 → x = 2, x = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x ^ 2 - 5 * x + 6);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {}", roots.len());
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        vals.contains(&"2".to_string()),
        "should have root 2: {vals:?}"
    );
    assert!(
        vals.contains(&"3".to_string()),
        "should have root 3: {vals:?}"
    );
}

#[test]
fn solve_quadratic_double_root() {
    // x^2 - 4*x + 4 = 0 → x = 2 (double root, may appear once or twice)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x ^ 2 - 4 * x + 4);
    let roots = expr.solve(&x).unwrap();
    assert!(
        !roots.is_empty(),
        "expected at least 1 root for double root equation"
    );
    for root in &roots {
        assert_eq!(format!("{root}"), "2", "all roots should be 2, got {root}");
    }
}

#[test]
fn solve_quadratic_complex_roots() {
    // x^2 + 1 = 0 → complex roots ±i
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x ^ 2 + 1);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(
        roots.len(),
        2,
        "expected 2 complex roots for x^2 + 1, got {}",
        roots.len()
    );
    let strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    let joined = strs.join(", ");
    assert!(
        joined.contains("I"),
        "roots of x^2+1 should contain I: {joined}"
    );
}

#[test]
fn solve_cubic_rational_roots() {
    // x^3 - 6*x^2 + 11*x - 6 = 0 → x = 1, 2, 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x ^ 3 - 6 * x ^ 2 + 11 * x - 6);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 3, "expected 3 roots, got {}", roots.len());
    let vals: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        vals.contains(&"1".to_string()),
        "should have root 1: {vals:?}"
    );
    assert!(
        vals.contains(&"2".to_string()),
        "should have root 2: {vals:?}"
    );
    assert!(
        vals.contains(&"3".to_string()),
        "should have root 3: {vals:?}"
    );
}

#[test]
fn solve_non_polynomial_returns_empty() {
    // sin(x) = 0 → now handled by inversion peeling: x = asin(0) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let result = expr.solve(&x);
    assert!(result.is_ok(), "sin(x) should be solvable via inversion peeling, got: {:?}", result.err());
}

#[test]
fn solve_constant_nonzero_no_solutions() {
    // 5 = 0 → no solutions
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let five = ctx.int(5);
    let roots = five.solve(&x).unwrap();
    assert!(
        roots.is_empty(),
        "expected no solutions for constant 5, got {} root(s)",
        roots.len()
    );
}

#[test]
fn solve_constant_zero_no_solutions() {
    // 0 = 0 → infinite solutions, returns empty
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.int(0);
    let roots = zero.solve(&x).unwrap();
    assert!(
        roots.is_empty(),
        "expected empty for 0=0 (infinite solutions), got {} root(s)",
        roots.len()
    );
}

#[test]
fn solve_verify_quadratic_roots() {
    // Solve x^2 - 5*x + 6 = 0, substitute each root back, verify it is zero
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = expr!(x ^ 2 - 5 * x + 6);
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2, "expected 2 roots, got {}", roots.len());
    for root in &roots {
        let substituted = expr.subs(&x, root);
        assert!(
            substituted.is_zero_structural(),
            "substituting x={root} into x^2 - 5x + 6 should give 0, got {substituted}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// cancel() tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cancel_common_factor() {
    // (x^2 - 1) / (x - 1) → 1 + x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (x.powi(2) - 1) / (&x - 1);
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "x + 1",
        "cancelling (x^2-1)/(x-1) should give 1 + x"
    );
}

#[test]
fn cancel_perfect_square() {
    // (x^2 + 2*x + 1) / (x + 1) → 1 + x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (x.powi(2) + &x * 2 + 1) / (&x + 1);
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "x + 1",
        "cancelling (x^2+2x+1)/(x+1) should give 1 + x"
    );
}

#[test]
fn cancel_no_common_factor() {
    // (x + 1) / (x + 2) → stays unchanged
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1) / (&x + 2);
    let original_str = format!("{expr}");
    let cancelled = expr.cancel(&x);
    let cancelled_str = format!("{cancelled}");
    assert_eq!(
        cancelled_str, original_str,
        "cancelling (x+1)/(x+2) should stay unchanged"
    );
}

#[test]
fn cancel_already_simple() {
    // x.cancel(&x) → stays as x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cancelled = x.cancel(&x);
    assert_eq!(format!("{cancelled}"), "x", "cancelling x should stay as x");
}

#[test]
fn cancel_non_polynomial_unchanged() {
    // sin(x).cancel(&x) → stays as sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let cancelled = expr.cancel(&x);
    assert_eq!(
        format!("{cancelled}"),
        "sin(x)",
        "cancelling sin(x) should stay as sin(x)"
    );
}
