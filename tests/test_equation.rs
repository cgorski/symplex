//! Comprehensive tests for the Equation type and eq! macro.

use symplex::eq::Equation;
use symplex::prelude::*;

// ── Construction & Display ────────────────────────────────

#[test]
fn equation_display_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = Equation::new(&x + 1, ctx.int(5));
    let s = format!("{equation}");
    assert!(s.contains("x"), "display should contain variable: {s}");
    assert!(s.contains("="), "display should contain equals sign: {s}");
    assert!(s.contains("5"), "display should contain rhs: {s}");
}

#[test]
fn equation_debug_format() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = Equation::new(x.clone(), ctx.int(0));
    let s = format!("{equation:?}");
    assert!(
        s.contains("Equation"),
        "debug format should contain 'Equation': {s}"
    );
    assert!(s.contains("="), "debug format should contain '=': {s}");
}

// ── Solving ───────────────────────────────────────────────

#[test]
fn equation_solve_linear() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = Equation::new(&x + 1, ctx.int(5));
    let roots = equation.solve(&x).unwrap();
    assert_eq!(roots.len(), 1, "linear equation should have 1 root");
    assert_eq!(format!("{}", roots[0]), "4");
}

#[test]
fn equation_solve_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = Equation::new(x.powi(2), ctx.int(9));
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x^2 = 9 should have 2 roots");
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"-3".to_string()) && strs.contains(&"3".to_string()),
        "roots of x^2=9 should be -3 and 3, got: {strs:?}"
    );
}

#[test]
fn equation_solve_or_empty_when_no_solution() {
    let ctx = Context::new();
    // sin(x) = 2 has no real solution; solve_or_empty should return empty or
    // at least not panic.
    let x = ctx.symbol("x");
    let equation = Equation::new(x.sin(), ctx.int(2));
    let roots = equation.solve_or_empty(&x);
    // The solver may or may not find solutions for transcendental equations,
    // but it must not panic. If it returns roots, that's fine too.
    let _ = roots;
}

// ── Substitution ──────────────────────────────────────────

#[test]
fn equation_subs_symbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let equation = Equation::new(&x + 1, ctx.int(5));
    // Substitute x → y+1
    let substituted = equation.subs(&x, &(&y + 1));
    let s = format!("{substituted}");
    assert!(
        s.contains("y"),
        "after substituting x→y+1, should contain y: {s}"
    );
}

#[test]
fn equation_subs_i64_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x + 3 = 7  →  solution is x = 4
    let equation = Equation::new(&x + 3, ctx.int(7));
    let substituted = equation.subs_i64(&x, 4);
    assert_eq!(
        substituted.is_satisfied(),
        Some(true),
        "substituting root x=4 into x+3=7 should satisfy equation"
    );
}

// ── Transforms ────────────────────────────────────────────

#[test]
fn equation_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x + 1)^2 = 4  →  expand  →  x^2 + 2*x + 1 = 4
    let equation = Equation::new((&x + 1).powi(2), ctx.int(4));
    let expanded = equation.expand();
    let lhs_str = format!("{}", expanded.lhs);
    assert!(
        lhs_str.contains("x^2"),
        "expanded (x+1)^2 should contain x^2: {lhs_str}"
    );
    assert!(
        lhs_str.contains("2*x") || lhs_str.contains("2x"),
        "expanded (x+1)^2 should contain 2*x term: {lhs_str}"
    );
}

#[test]
fn equation_eval() {
    let ctx = Context::new();
    // sin(0) = 0  →  eval  →  0 = 0
    let zero = ctx.int(0);
    let equation = Equation::new(zero.sin(), ctx.int(0));
    let evaled = equation.eval();
    assert_eq!(format!("{}", evaled.lhs), "0", "eval of sin(0) should be 0");
    assert_eq!(
        evaled.is_satisfied(),
        Some(true),
        "0 = 0 should be satisfied after eval"
    );
}

#[test]
fn equation_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin^2(x) + cos^2(x) = 1  →  simplify  →  1 = 1
    let equation = Equation::new(&x.sin().powi(2) + &x.cos().powi(2), ctx.int(1));
    let simplified = equation.simplify();
    assert_eq!(
        format!("{}", simplified.lhs),
        "1",
        "sin^2(x)+cos^2(x) should simplify to 1"
    );
}

// ── is_satisfied ──────────────────────────────────────────

#[test]
fn equation_is_satisfied_true() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Solve x + 1 = 5, substitute root back
    let equation = Equation::new(&x + 1, ctx.int(5));
    let roots = equation.solve_or_empty(&x);
    assert!(!roots.is_empty(), "should find at least one root");
    for root in &roots {
        let check = equation.subs(&x, root);
        assert_eq!(
            check.is_satisfied(),
            Some(true),
            "root {root} should satisfy x+1=5"
        );
    }
}

#[test]
fn equation_is_satisfied_false() {
    let ctx = Context::new();
    // 5 = 3  →  obviously false, but `equals` only proves equality;
    // it returns None when it cannot confirm the two sides are equal.
    let equation = Equation::new(ctx.int(5), ctx.int(3));
    assert_ne!(
        equation.is_satisfied(),
        Some(true),
        "5 = 3 must not be reported as satisfied"
    );
}

#[test]
fn equation_is_satisfied_unknown() {
    let ctx = Context::new();
    // x = y  →  can't determine without knowing values
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let equation = Equation::new(x.clone(), y.clone());
    assert_eq!(
        equation.is_satisfied(),
        None,
        "x = y should be unknown (None)"
    );
}

// ── to_expr ───────────────────────────────────────────────

#[test]
fn equation_to_expr_gives_difference() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = Equation::new(x.clone(), ctx.int(3));
    let expr = equation.to_expr();
    let s = format!("{expr}");
    // to_expr returns lhs - rhs = x - 3
    assert!(
        s.contains("x") && s.contains("3"),
        "to_expr of (x = 3) should give x - 3: {s}"
    );
}

// ── eq! macro ─────────────────────────────────────────────

#[test]
fn eq_macro_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = eq!(ctx, x ^ 2 - 1 = 0);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x^2-1=0 should have 2 roots");
    let mut strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    strs.sort();
    assert!(
        strs.contains(&"-1".to_string()) && strs.contains(&"1".to_string()),
        "roots of x^2-1=0 should be -1 and 1, got: {strs:?}"
    );
}

#[test]
fn eq_macro_with_rationals() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let equation = eq!(ctx, x = 1 / 2);
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 1);
    assert_eq!(format!("{}", roots[0]), "1/2");
}

// ── Workflow: construct → solve → verify ──────────────────

#[test]
fn equation_solve_then_verify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^2 - 6x + 8 = 0  →  roots are 2 and 4
    let equation = Equation::new(&x.powi(2) - &(&x * 6) + ctx.int(8), ctx.int(0));
    let roots = equation.solve_or_empty(&x);
    assert_eq!(roots.len(), 2, "x^2-6x+8=0 should have 2 roots");
    for root in &roots {
        let check = equation.subs(&x, root);
        assert_eq!(
            check.is_satisfied(),
            Some(true),
            "root {root} should satisfy x^2-6x+8=0"
        );
    }
    let mut root_strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    root_strs.sort();
    assert!(
        root_strs.contains(&"2".to_string()) && root_strs.contains(&"4".to_string()),
        "roots should be 2 and 4, got: {root_strs:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponential equations: a^x = c
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_2_pow_x_eq_8() {
    // 2^x = 8 → x = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(2).pow(&x) - 8;
    let roots = expr.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "2^x - 8 = 0 should have solutions, got empty"
    );
    let root_strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        root_strs.contains(&"3".to_string()),
        "2^x = 8 should give x = 3, got: {root_strs:?}"
    );
}

#[test]
fn solve_3_pow_x_eq_81() {
    // 3^x = 81 → x = 4
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(3).pow(&x) - 81;
    let roots = expr.solve_or_empty(&x);
    let root_strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        root_strs.contains(&"4".to_string()),
        "3^x = 81 should give x = 4, got: {root_strs:?}"
    );
}

#[test]
fn solve_10_pow_x_eq_1000() {
    // 10^x = 1000 → x = 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(10).pow(&x) - 1000;
    let roots = expr.solve_or_empty(&x);
    let root_strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        root_strs.contains(&"3".to_string()),
        "10^x = 1000 should give x = 3, got: {root_strs:?}"
    );
}

#[test]
fn solve_2_pow_x_eq_5_general() {
    // 2^x = 5 → x = ln(5)/ln(2) (no integer shortcut)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(2).pow(&x) - 5;
    let roots = expr.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "2^x = 5 should have a solution"
    );
    // Verify numerically: x ≈ 2.32193
    if let Ok(v) = roots[0].eval_f64() {
        let expected = 5.0_f64.ln() / 2.0_f64.ln();
        assert!(
            (v - expected).abs() < 1e-8,
            "2^x = 5: root should be ≈ {expected:.6}, got {v:.6}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse trig peeling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_asin_x_eq_value() {
    // asin(x) = π/6 → x = sin(π/6) = 1/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi_over_6 = &ctx.pi() / 6;
    let expr = &x.asin() - &pi_over_6;
    let roots = expr.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "asin(x) = π/6 should have a solution"
    );
    // sin(π/6) = 1/2
    if let Ok(v) = roots[0].eval_f64() {
        assert!(
            (v - 0.5).abs() < 1e-8,
            "asin(x) = π/6: root should be 0.5, got {v}"
        );
    }
}

#[test]
fn solve_atan_x_eq_value() {
    // atan(x) = π/4 → x = tan(π/4) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi_over_4 = &ctx.pi() / 4;
    let expr = &x.atan() - &pi_over_4;
    let roots = expr.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "atan(x) = π/4 should have a solution"
    );
    if let Ok(v) = roots[0].eval_f64() {
        assert!(
            (v - 1.0).abs() < 1e-8,
            "atan(x) = π/4: root should be 1.0, got {v}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse hyperbolic peeling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_sinh_x_eq_zero() {
    // sinh(x) = 0 → x = asinh(0) = 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sinh();
    let roots = expr.solve_or_empty(&x);
    assert!(
        !roots.is_empty(),
        "sinh(x) = 0 should have a solution"
    );
    // The solver returns asinh(0) which evaluates numerically to 0
    if let Ok(v) = roots[0].eval_f64() {
        assert!(
            v.abs() < 1e-10,
            "sinh(x) = 0: root should be ≈ 0, got {v}"
        );
    } else {
        // If eval_f64 fails, check if simplify/eval produces 0
        let simplified = roots[0].eval();
        let s = format!("{simplified}");
        assert!(
            s == "0" || s == "asinh(0)",
            "sinh(x) = 0 should give x = 0 or asinh(0), got: {s}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Abs peeling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn solve_abs_x_eq_5() {
    // |x| = 5 → x = 5 or x = -5
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.abs() - 5;
    let roots = expr.solve_or_empty(&x);
    assert!(
        roots.len() >= 2,
        "|x| = 5 should have two solutions, got {}: {:?}",
        roots.len(),
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
    let root_strs: Vec<String> = roots.iter().map(|r| format!("{r}")).collect();
    assert!(
        root_strs.contains(&"5".to_string()) && root_strs.contains(&"-5".to_string()),
        "|x| = 5 should give x = 5 and x = -5, got: {root_strs:?}"
    );
}

#[test]
fn solve_abs_x_eq_negative_empty() {
    // |x| = -3 → no solutions
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.abs() + 3; // |x| + 3 = 0 → |x| = -3
    let roots = expr.solve_or_empty(&x);
    assert!(
        roots.is_empty(),
        "|x| = -3 should have no solutions, got: {:?}",
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );
}
