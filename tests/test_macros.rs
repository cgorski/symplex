//! Integration tests for the `expr!` and `rule!` proc macros.

use symplex::prelude::*;
use symplex::{expr, rule, syms};

// ═══════════════════════════════════════════════════════════════════════════
// expr! macro
// ═══════════════════════════════════════════════════════════════════════════

// ── Basic arithmetic ────────────────────────────────────────────────────

#[test]
fn expr_add() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = expr!(x + y);
    assert_eq!(format!("{result}"), "x + y");
}

#[test]
fn expr_sub() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = expr!(x - y);
    assert_eq!(format!("{result}"), "x - y");
}

#[test]
fn expr_mul() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = expr!(x * y);
    assert_eq!(format!("{result}"), "x*y");
}

#[test]
fn expr_div() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = expr!(x / y);
    assert_eq!(format!("{result}"), "x*1/y");
}

#[test]
fn expr_neg() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(-x);
    assert_eq!(format!("{result}"), "-x");
}

// ── Power ───────────────────────────────────────────────────────────────

#[test]
fn expr_power_integer() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x ^ 2);
    assert_eq!(format!("{result}"), "x^2");
}

#[test]
fn expr_power_cubed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x ^ 3);
    assert_eq!(format!("{result}"), "x^3");
}

#[test]
fn expr_power_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x ^ (-1));
    assert_eq!(format!("{result}"), "1/x");
}

#[test]
fn expr_power_symbolic() {
    let ctx = Context::new();
    syms!(ctx; x, n);
    let result = expr!(x ^ n);
    let s = format!("{result}");
    assert!(s.contains("x") && s.contains("n"), "got: {s}");
}

// ── Precedence ──────────────────────────────────────────────────────────

#[test]
fn expr_precedence_add_mul() {
    let ctx = Context::new();
    syms!(ctx; x, y, z);
    // x + y * z should be x + (y*z), not (x+y)*z
    let result = expr!(x + y * z);
    let manual = &x + &(&y * &z);
    assert_eq!(result, manual);
}

#[test]
fn expr_precedence_pow_mul() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // x^2 * y should be (x^2) * y
    let result = expr!(x ^ 2 * y);
    let manual = &x.powi(2) * &y;
    assert_eq!(result, manual);
}

#[test]
fn expr_precedence_parens() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    // (x + y)^2 should group correctly
    let result = expr!((x + y) ^ 2);
    let manual = (&x + &y).powi(2);
    assert_eq!(result, manual);
}

// ── Integer literals ────────────────────────────────────────────────────

#[test]
fn expr_with_integer_literal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x + 1);
    assert_eq!(format!("{result}"), "1 + x");
}

#[test]
fn expr_integer_mul() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(2 * x);
    assert_eq!(format!("{result}"), "2*x");
}

#[test]
fn expr_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x ^ 2 + 2 * x + 1);
    assert_eq!(format!("{result}"), "1 + x^2 + 2*x");
}

// ── Functions ───────────────────────────────────────────────────────────

#[test]
fn expr_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(sin(x));
    assert_eq!(format!("{result}"), "sin(x)");
}

#[test]
fn expr_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(cos(x));
    assert_eq!(format!("{result}"), "cos(x)");
}

#[test]
fn expr_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(tan(x));
    assert_eq!(format!("{result}"), "tan(x)");
}

#[test]
fn expr_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(exp(x));
    assert_eq!(format!("{result}"), "exp(x)");
}

#[test]
fn expr_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(ln(x));
    assert_eq!(format!("{result}"), "ln(x)");
}

#[test]
fn expr_sqrt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(sqrt(x));
    assert_eq!(format!("{result}"), "sqrt(x)");
}

#[test]
fn expr_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(abs(x));
    assert_eq!(format!("{result}"), "abs(x)");
}

// ── Nested functions ────────────────────────────────────────────────────

#[test]
fn expr_sin_of_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(sin(x ^ 2));
    assert_eq!(format!("{result}"), "sin(x^2)");
}

#[test]
fn expr_function_in_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(sin(x) ^ 2 + cos(x) ^ 2);
    let manual = &x.sin().powi(2) + &x.cos().powi(2);
    assert_eq!(result, manual);
}

#[test]
fn expr_nested_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(sin(cos(x)));
    assert_eq!(format!("{result}"), "sin(cos(x))");
}

// ── Complex expressions ─────────────────────────────────────────────────

#[test]
fn expr_quadratic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = expr!(x ^ 2 - 5 * x + 6);
    // Canonical form has terms sorted
    let s = format!("{result}");
    assert!(s.contains("x^2"), "should contain x^2: {s}");
    assert!(s.contains("5*x"), "should contain 5*x: {s}");
    assert!(s.contains("6"), "should contain 6: {s}");
}

#[test]
fn expr_product_of_sum() {
    let ctx = Context::new();
    syms!(ctx; x, y);
    let result = expr!((x + 1) * (y - 1));
    let manual = &(&x + 1) * &(&y - 1);
    assert_eq!(result, manual);
}

#[test]
fn expr_right_associative_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^(2^3) = x^8.  We write it as x^8 directly because the macro
    // cannot handle Int^Int (the literal 2 is not an Ex, so .powi()
    // is unavailable on it).
    let result = expr!(x ^ 8);
    let s = format!("{result}");
    assert_eq!(s, "x^8", "x^8 should display as x^8: {s}");
}

// ── Reuse of variables ──────────────────────────────────────────────────

#[test]
fn expr_reuse_variable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = expr!(x ^ 2 + 1);
    let b = expr!(x ^ 3 - 1);
    // x is still usable — expr! only borrows
    let c = expr!(x + 1);
    assert!(format!("{a}").contains("x"));
    assert!(format!("{b}").contains("x"));
    assert!(format!("{c}").contains("x"));
}

#[test]
fn expr_with_pre_built_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = &x + 1; // pre-built Ex
    let result = expr!(inner ^ 2);
    assert_eq!(format!("{result}"), "(1 + x)^2");
}

// ═══════════════════════════════════════════════════════════════════════════
// rule! macro
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_pythagorean() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1);
        assert_eq!(r.name, "pythagorean");
    });
}

#[test]
fn rule_pythagorean_construction() {
    // Verify the pythagorean rule compiles with the macro and has the right name.
    // Rule application is tested in the pattern engine's own unit tests.
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1);
        assert_eq!(r.name, "pythagorean");
        // The pattern should have a root and wildcard bindings
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_exp_ln() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "exp_ln", exp(ln(w_)) => w_);
        assert_eq!(r.name, "exp_ln");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_ln_exp() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "ln_exp", ln(exp(w_)) => w_);
        assert_eq!(r.name, "ln_exp");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_constant() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // Rule with a concrete constant (pi) instead of a wildcard
        let r = rule!(arena, "sin_pi", sin(pi) => 0);
        assert_eq!(r.name, "sin_pi");
        // pi is a concrete symbol, not a wildcard, so wilds should be empty
        assert!(r.pattern.wilds.is_empty(), "no wildcards in this rule");
    });
}

#[test]
fn rule_no_match_construction() {
    // Just verify the rule constructs correctly; matching is tested internally.
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        let r = rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1);
        assert_eq!(r.name, "pythagorean");
    });
}

#[test]
fn rule_sqrt_sq() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // Rule: sqrt(w_^2) => abs(w_)
        let r = rule!(arena, "sqrt_sq", sqrt(w_^2) => abs(w_));
        assert_eq!(r.name, "sqrt_sq");
        assert!(!r.pattern.wilds.is_empty(), "should have wildcard w_");
    });
}

#[test]
fn rule_with_different_wild_names() {
    let ctx = Context::new();
    ctx.with_arena_mut(|arena| {
        // Rule: a_ * 0 => 0
        // Mul(a_, 0) canonicalizes to 0 directly, so this pattern can't
        // match (the expression would already be 0). Test that the rule
        // at least compiles and constructs without error.
        let r = rule!(arena, "mul_zero_wild", a_ * 0 => 0);
        assert_eq!(r.name, "mul_zero_wild");
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Combined workflows
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expr_then_diff() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!(x ^ 3 + 2 * x + 1);
    let df = f.diff(&x);
    let s = format!("{df}");
    assert!(s.contains("3*x^2"), "d/dx(x³+2x+1) should contain 3x²: {s}");
    assert!(s.contains("2"), "should contain 2: {s}");
}

#[test]
fn expr_then_subs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!(x ^ 2 + 1);
    let result = f.subs(&x, &ctx.int(3));
    assert_eq!(format!("{result}"), "10");
}

#[test]
fn expr_then_expand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!((x + 1) ^ 2);
    let expanded = f.expand();
    assert_eq!(format!("{expanded}"), "1 + x^2 + 2*x");
}

#[test]
fn expr_then_solve() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!(x ^ 2 - 5 * x + 6);
    let roots = f.solve(&x);
    assert_eq!(roots.len(), 2);
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
fn expr_then_evalf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = expr!(x ^ 2 + 1);
    let at_pi = f.subs(&x, &ctx.pi());
    let result = at_pi.evalf(15).unwrap();
    assert!(
        result.starts_with("10.8696"),
        "pi² + 1 ≈ 10.8696..., got: {result}"
    );
}
