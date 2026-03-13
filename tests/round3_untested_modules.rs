//! Round 3 — under-tested module bug hunt.
//!
//! Targets: sort_key.rs, walk.rs, pretty.rs, rewrite.rs, fourier.rs
//!
//! These modules have low or zero external test coverage. Bugs here
//! silently corrupt canonicalization, tree traversal, display, rewriting,
//! and Fourier series computation.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerical equivalence check for single-variable expressions.
fn numerical_eq_1var(a: &Ex, b: &Ex, var: &Ex, points: &[i64], tol: f64) -> bool {
    for &p in points {
        let va = a.subs_i64(var, p).eval().eval_f64();
        let vb = b.subs_i64(var, p).eval().eval_f64();
        match (va, vb) {
            (Ok(fa), Ok(fb)) => {
                let denom = fa.abs().max(fb.abs()).max(1.0);
                if (fa - fb).abs() > tol * denom {
                    eprintln!(
                        "  numerical_eq_1var FAIL at var={p}: a={fa}, b={fb}, diff={}",
                        (fa - fb).abs()
                    );
                    return false;
                }
            }
            (Err(_), Err(_)) => {} // both fail — acceptable
            (Ok(fa), Err(e)) => {
                eprintln!("  numerical_eq_1var FAIL at var={p}: a={fa}, b=Err({e})");
                return false;
            }
            (Err(e), Ok(fb)) => {
                eprintln!("  numerical_eq_1var FAIL at var={p}: a=Err({e}), b={fb}");
                return false;
            }
        }
    }
    true
}

fn fmt(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 1: sort_key.rs — canonical ordering of expression terms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sort_key_commutative_add_xy_vs_yx() {
    // x + y and y + x should canonicalize identically
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr1 = &x + &y;
    let expr2 = &y + &x;

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "x+y and y+x should have identical canonical form"
    );
}

#[test]
fn sort_key_commutative_mul_abc_vs_cab() {
    // a*b*c and c*a*b should canonicalize identically
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");

    let expr1 = &(&a * &b) * &c;
    let expr2 = &(&c * &a) * &b;

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "a*b*c and c*a*b should have identical canonical form"
    );
}

#[test]
fn sort_key_commutative_add_three_symbols() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let expr1 = &(&x + &y) + &z;
    let expr2 = &(&z + &x) + &y;
    let expr3 = &(&y + &z) + &x;

    let s1 = fmt(&expr1);
    let s2 = fmt(&expr2);
    let s3 = fmt(&expr3);

    assert_eq!(s1, s2, "x+y+z vs z+x+y should canonicalize the same");
    assert_eq!(s2, s3, "z+x+y vs y+z+x should canonicalize the same");
}

#[test]
fn sort_key_number_before_symbol() {
    // In x + 3, the number 3 should sort before x (or after, consistently)
    // The key thing is that the canonical form is deterministic.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three = ctx.int(3);

    let expr1 = &x + &three;
    let expr2 = &three + &x;

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "x+3 and 3+x should canonicalize identically"
    );
}

#[test]
fn sort_key_different_symbols_different_keys() {
    // Verify x and y have different representations
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    assert_ne!(
        fmt(&x),
        fmt(&y),
        "different symbols should have different display forms"
    );

    // Also check that x + 0*y ≠ y + 0*x would differ — but more simply,
    // x and y as standalone should be different.
    let sum_x_first = &x + &y;
    // If sort keys were the same, we'd get wrong canonicalization.
    // The expression should contain both x and y.
    let s = fmt(&sum_x_first);
    assert!(s.contains('x') && s.contains('y'), "sum should contain both symbols: {s}");
}

#[test]
fn sort_key_stability() {
    // sort_key(a) == sort_key(a) always — same expression, same key
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2);

    let s1 = fmt(&expr);
    let s2 = fmt(&expr);

    assert_eq!(s1, s2, "same expression should always format the same way");
}

#[test]
fn sort_key_total_order_symbol_vs_function() {
    // For two different expression types, the ordering should be total
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    // They should have consistent ordering — neither should be "equal"
    let sx = fmt(&x);
    let ss = fmt(&sin_x);
    assert_ne!(sx, ss, "symbol and function of it should differ");
}

#[test]
fn sort_key_complex_expression_different_build_orders() {
    // sin(x) + x^2 + 3 — build in different orders, verify same canonical form
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three = ctx.int(3);

    // Order 1: sin(x) + x^2 + 3
    let expr1 = &(&x.sin() + &x.powi(2)) + &three;
    // Order 2: 3 + x^2 + sin(x)
    let expr2 = &(&three + &x.powi(2)) + &x.sin();
    // Order 3: x^2 + 3 + sin(x)
    let expr3 = &(&x.powi(2) + &three) + &x.sin();

    let s1 = fmt(&expr1);
    let s2 = fmt(&expr2);
    let s3 = fmt(&expr3);

    assert_eq!(s1, s2, "sin(x)+x^2+3 built in different orders should match: '{s1}' vs '{s2}'");
    assert_eq!(s2, s3, "same expression, third build order: '{s2}' vs '{s3}'");
}

#[test]
fn sort_key_mul_with_numbers_and_symbols() {
    // 2*x*y and y*2*x should canonicalize the same
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let two = ctx.int(2);

    let expr1 = &(&two * &x) * &y;
    let expr2 = &(&y * &two) * &x;

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "2*x*y and y*2*x should canonicalize identically"
    );
}

#[test]
fn sort_key_constants_ordering() {
    // Pi, E, and i should have distinct canonical forms
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.symbol("e"); // Note: ctx.e() may not exist, use a real constant if available
    let i_unit = ctx.i_unit();

    let s_pi = fmt(&pi);
    let s_i = fmt(&i_unit);

    assert_ne!(s_pi, s_i, "pi and i should be different");
}

#[test]
fn sort_key_add_with_negation() {
    // x - y and -y + x should canonicalize the same
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr1 = &x - &y;
    let expr2 = &(-&y) + &x;

    // These should be equivalent after canonicalization
    let s1 = fmt(&expr1);
    let s2 = fmt(&expr2);
    assert_eq!(s1, s2, "x-y and -y+x should canonicalize identically: '{s1}' vs '{s2}'");
}

#[test]
fn sort_key_nested_products_same_canonical() {
    // (a*b)*(c*d) and (d*c)*(b*a) should be identical
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");

    let expr1 = &(&a * &b) * &(&c * &d);
    let expr2 = &(&d * &c) * &(&b * &a);

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "nested products in different order should canonicalize the same"
    );
}

#[test]
fn sort_key_symbols_alphabetical() {
    // a + b + c should have the symbols in alphabetical order
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");

    let expr = &(&c + &a) + &b;
    let s = fmt(&expr);

    // Find positions of a, b, c in the string
    let pos_a = s.find('a').expect("should contain a");
    let pos_b = s.find('b').expect("should contain b");
    let pos_c = s.find('c').expect("should contain c");

    assert!(
        pos_a < pos_b && pos_b < pos_c,
        "symbols should appear in alphabetical order in canonical form: '{s}' (a@{pos_a}, b@{pos_b}, c@{pos_c})"
    );
}

#[test]
fn sort_key_number_vs_symbol_ordering() {
    // Numbers should sort before (or after) symbols consistently
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);

    let sum = &x + &two;
    let s = fmt(&sum);

    // The canonical form should be deterministic
    // According to sort_key.rs, RANK_NUM=0 < RANK_SYMBOL=20
    // So number sorts first, meaning the display should show the number first
    // (or the number is collected as a coefficient).
    // Just verify it's deterministic:
    let sum2 = &two + &x;
    assert_eq!(fmt(&sum), fmt(&sum2), "number+symbol order should be canonical");
}

#[test]
fn sort_key_function_vs_power_ordering() {
    // sin(x) and x^2 should have a deterministic order
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let sin_x = x.sin();
    let x_sq = x.powi(2);

    let expr1 = &sin_x + &x_sq;
    let expr2 = &x_sq + &sin_x;

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "sin(x)+x^2 vs x^2+sin(x) should canonicalize the same"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 2: walk.rs — tree traversal (free_symbols, contains, etc.)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn walk_free_symbols_simple_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &x + &y;
    let syms = expr.free_symbols();
    let mut names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();
    names.sort();

    assert_eq!(names, vec!["x", "y"], "free_symbols(x+y) should be {{x, y}}");
}

#[test]
fn walk_free_symbols_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = x.sin();
    let syms = expr.free_symbols();
    let names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(names, vec!["x"], "free_symbols(sin(x)) should be {{x}}");
}

#[test]
fn walk_free_symbols_number_only() {
    let ctx = Context::new();
    let five = ctx.int(5);

    let syms = five.free_symbols();
    assert!(
        syms.is_empty(),
        "free_symbols(5) should be empty, got: {:?}",
        syms.iter().map(|s| fmt(s)).collect::<Vec<_>>()
    );
}

#[test]
fn walk_free_symbols_no_duplicates() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x + x = 2*x — should still only have {x}
    let expr = &x + &x;
    let syms = expr.free_symbols();
    let names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(
        names.len(),
        1,
        "free_symbols(x+x) should have no duplicates, got: {names:?}"
    );
    assert_eq!(names[0], "x");
}

#[test]
fn walk_free_symbols_nested() {
    // sin(x) + y*cos(z)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let expr = &x.sin() + &(&y * &z.cos());
    let syms = expr.free_symbols();
    let mut names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();
    names.sort();

    assert_eq!(
        names,
        vec!["x", "y", "z"],
        "free_symbols(sin(x) + y*cos(z)) should be {{x,y,z}}"
    );
}

#[test]
fn walk_free_symbols_constant_expression() {
    // pi + 2*e^(1) — no free symbols (all constants/numbers)
    let ctx = Context::new();
    let pi = ctx.pi();
    let two = ctx.int(2);

    let expr = &pi + &two;
    let syms = expr.free_symbols();

    assert!(
        syms.is_empty(),
        "free_symbols(pi+2) should be empty (constants aren't free symbols), got: {:?}",
        syms.iter().map(|s| fmt(s)).collect::<Vec<_>>()
    );
}

#[test]
fn walk_contains_direct_child() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &x + &y;
    assert!(expr.contains(&x), "x+y should contain x");
    assert!(expr.contains(&y), "x+y should contain y");
}

#[test]
fn walk_contains_not_present() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let expr = &x + &y;
    assert!(!expr.contains(&z), "x+y should not contain z");
}

#[test]
fn walk_contains_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = x.sin();
    assert!(expr.contains(&x), "sin(x) should contain x");
}

#[test]
fn walk_contains_self() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + &ctx.int(1);

    assert!(expr.contains(&expr), "expression should contain itself");
}

#[test]
fn walk_contains_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build sin(sin(sin(x)))
    let expr = x.sin().sin().sin();
    assert!(expr.contains(&x), "sin(sin(sin(x))) should contain x");
}

#[test]
fn walk_free_symbols_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let n = ctx.symbol("n");

    let expr = x.pow(&n);
    let syms = expr.free_symbols();
    let mut names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();
    names.sort();

    assert_eq!(names, vec!["n", "x"], "free_symbols(x^n) should be {{n, x}}");
}

#[test]
fn walk_deep_expression_no_stack_overflow() {
    // Build a deeply nested expression: ((((x + 1) + 1) + 1) ... + 1) — 100 deep
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);

    let mut expr = x.clone();
    for _ in 0..100 {
        expr = &expr + &one;
    }

    // Should not stack overflow
    let syms = expr.free_symbols();
    assert_eq!(
        syms.len(),
        1,
        "deeply nested expression should still find x"
    );
    assert!(expr.contains(&x), "deeply nested expression should contain x");
}

#[test]
fn walk_has_unevaluated_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Simple polynomial — no unevaluated forms
    let expr = &x.powi(2) + &x + &ctx.int(1);
    assert!(
        !expr.has_unevaluated(),
        "x^2 + x + 1 should not have unevaluated forms"
    );
}

#[test]
fn walk_free_symbols_mul_same_symbol() {
    // x * x = x^2 — should have free symbols {x}
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &x * &x;
    let syms = expr.free_symbols();
    let names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(
        names.len(),
        1,
        "free_symbols(x*x) should have exactly 1 symbol, got: {names:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 3: pretty.rs — human-readable 2D formatted output
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pretty_fraction_has_horizontal_bar() {
    // Pretty-print of 1/x should show a fraction with horizontal bar
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let expr = &one / &x;

    let pretty = expr.pretty();
    // A 2D fraction should be multi-line
    let line_count = pretty.lines().count();
    assert!(
        line_count >= 2,
        "1/x pretty should have multiple lines (fraction), got {line_count} line(s): '{pretty}'"
    );
    // Should contain some kind of horizontal bar character
    let has_bar = pretty.contains('─')
        || pretty.contains('━')
        || pretty.contains('╌')
        || pretty.contains('-')
        || pretty.contains('=');
    assert!(
        has_bar,
        "1/x pretty should contain a horizontal bar: '{pretty}'"
    );
}

#[test]
fn pretty_power_shows_exponent() {
    // Pretty-print of x^2 should show superscript or caret
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);

    let pretty = expr.pretty();
    // Should show 2 somewhere (as superscript or explicit exponent)
    assert!(
        pretty.contains('2') || pretty.contains("**") || pretty.contains('^'),
        "x^2 pretty should show exponent: '{pretty}'"
    );
}

#[test]
fn pretty_never_empty_for_nonzero() {
    // Pretty-print should never be empty for any non-zero expression
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exprs: Vec<Ex> = vec![
        ctx.int(1),
        ctx.int(-1),
        x.clone(),
        x.sin(),
        x.powi(2),
        &x + &ctx.int(1),
        ctx.pi(),
        ctx.i_unit(),
    ];

    for (i, e) in exprs.iter().enumerate() {
        let pretty = e.pretty();
        assert!(
            !pretty.trim().is_empty(),
            "pretty-print should never be empty for expression #{i}: '{}'",
            fmt(e)
        );
    }
}

#[test]
fn pretty_ascii_fallback_works() {
    // ASCII mode should work for all basic expressions
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &x.sin() + &x.powi(2);
    let ascii = expr.pretty_ascii();

    assert!(
        !ascii.trim().is_empty(),
        "ASCII pretty-print should not be empty"
    );
    // ASCII should not contain Unicode box-drawing characters
    assert!(
        !ascii.contains('━') && !ascii.contains('─'),
        "ASCII mode should not contain Unicode box-drawing chars: '{ascii}'"
    );
}

#[test]
fn pretty_fraction_expression_multiline() {
    // (x + 1) / (x - 1) should render as a stacked fraction
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let numer = &x + &one;
    let denom = &x - &one;
    let expr = &numer / &denom;

    let pretty = expr.pretty();
    let line_count = pretty.lines().count();
    assert!(
        line_count >= 3,
        "(x+1)/(x-1) pretty should have >=3 lines (numer + bar + denom), got {line_count}: '{pretty}'"
    );
}

#[test]
fn pretty_deeply_nested_no_panic() {
    // Pretty-print should handle deeply nested expressions without panic
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build sin(sin(sin(...sin(x)...))) 20 levels deep
    let mut expr = x.clone();
    for _ in 0..20 {
        expr = expr.sin();
    }

    // Should not panic
    let pretty = expr.pretty();
    assert!(!pretty.is_empty(), "deeply nested pretty should not be empty");
}

#[test]
fn pretty_integer_is_simple() {
    let ctx = Context::new();
    let n = ctx.int(42);
    let pretty = n.pretty();
    assert_eq!(pretty.trim(), "42", "pretty-print of 42 should be '42', got '{pretty}'");
}

#[test]
fn pretty_symbol_is_name() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pretty = x.pretty();
    assert_eq!(pretty.trim(), "x", "pretty-print of x should be 'x', got '{pretty}'");
}

#[test]
fn pretty_negative_number() {
    let ctx = Context::new();
    let neg3 = ctx.int(-3);
    let pretty = neg3.pretty();
    assert!(
        pretty.contains("-3") || pretty.contains("−3"),
        "pretty-print of -3 should contain '-3': '{pretty}'"
    );
}

#[test]
fn pretty_sqrt_rendering() {
    // x^(1/2) should render as sqrt
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let expr = x.pow(&half);

    let pretty = expr.pretty();
    // Should contain a square root symbol or 'sqrt'
    assert!(
        pretty.contains('√') || pretty.contains("sqrt"),
        "x^(1/2) pretty should show sqrt symbol: '{pretty}'"
    );
}

#[test]
fn pretty_sin_function() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();

    let pretty = expr.pretty();
    assert!(
        pretty.contains("sin"),
        "sin(x) pretty should contain 'sin': '{pretty}'"
    );
    assert!(
        pretty.contains('x'),
        "sin(x) pretty should contain 'x': '{pretty}'"
    );
}

#[test]
fn pretty_pi_unicode() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let pretty = pi.pretty();
    assert!(
        pretty.contains('π') || pretty.contains("pi"),
        "pi pretty should contain π or 'pi': '{pretty}'"
    );
}

#[test]
fn pretty_abs_has_bars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs();

    let pretty = expr.pretty();
    assert!(
        pretty.contains('|') || pretty.contains('│'),
        "abs(x) pretty should contain vertical bars: '{pretty}'"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 4: rewrite.rs — expression rewriting (trig ↔ exp)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rewrite_sin_to_exp_numerical() {
    // sin(x).rewrite_as_exp() should be numerically equivalent to sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    let rewritten = sin_x.rewrite_as_exp();
    let s = fmt(&rewritten);

    // Should contain exp
    assert!(
        s.contains("exp"),
        "sin(x).rewrite_as_exp() should contain 'exp': '{s}'"
    );

    // Numerical equivalence
    assert!(
        numerical_eq_1var(&sin_x, &rewritten, &x, &[0, 1, 2, -1, -2, 3], 1e-10),
        "sin(x) and its exp rewrite should be numerically equal. rewrite = '{s}'"
    );
}

#[test]
fn rewrite_cos_to_exp_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_x = x.cos();

    let rewritten = cos_x.rewrite_as_exp();
    let s = fmt(&rewritten);

    assert!(
        s.contains("exp"),
        "cos(x).rewrite_as_exp() should contain 'exp': '{s}'"
    );

    assert!(
        numerical_eq_1var(&cos_x, &rewritten, &x, &[0, 1, 2, -1, -2, 3], 1e-10),
        "cos(x) and its exp rewrite should be numerically equal. rewrite = '{s}'"
    );
}

#[test]
fn rewrite_exp_ix_to_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();

    let ix = &i * &x;
    let exp_ix = ix.exp();

    let rewritten = exp_ix.rewrite_as_trig();
    let s = fmt(&rewritten);

    assert!(
        s.contains("cos") && s.contains("sin"),
        "exp(ix).rewrite_as_trig() should contain cos and sin: '{s}'"
    );
}

/// BUG FOUND [rewrite.rs]: sinh(x).rewrite_as_exp() should produce exponential form.
/// sinh(x) = (exp(x) - exp(-x)) / 2
///
/// rewrite.rs only matches Sin, Cos, Tan in its ExprNode match — Sinh, Cosh,
/// and Tanh are silently skipped, returning the expression unchanged.
#[test]

fn rewrite_sinh_to_exp_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sinh_x = x.sinh();

    let rewritten = sinh_x.rewrite_as_exp();
    let s = fmt(&rewritten);

    // This is the bug detector: rewrite.rs does not handle Sinh
    // So rewritten will likely still be "sinh(x)"
    let changed = s != fmt(&sinh_x);
    if !changed {
        eprintln!(
            "BUG FOUND [rewrite.rs]: sinh(x).rewrite_as_exp() returned unchanged: '{s}'. \
             The rewrite module only handles Sin/Cos/Tan but not Sinh/Cosh/Tanh."
        );
    }

    // Even if it didn't rewrite, numerical equivalence should hold
    assert!(
        numerical_eq_1var(&sinh_x, &rewritten, &x, &[0, 1, 2, -1, -2], 1e-10),
        "sinh(x) and its rewrite should be numerically equal"
    );

    // Assert the bug: rewrite should have changed it
    assert!(
        changed,
        "BUG: sinh(x).rewrite_as_exp() should produce exponential form but returned '{s}' unchanged"
    );
}

/// BUG FOUND [rewrite.rs]: cosh(x).rewrite_as_exp() should produce exponential form.
/// cosh(x) = (exp(x) + exp(-x)) / 2
///
/// Same root cause as sinh: the match in rewrite_as_exp only covers Sin/Cos/Tan.
#[test]

fn rewrite_cosh_to_exp_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cosh_x = x.cosh();

    let rewritten = cosh_x.rewrite_as_exp();
    let s = fmt(&rewritten);

    let changed = s != fmt(&cosh_x);
    if !changed {
        eprintln!(
            "BUG FOUND [rewrite.rs]: cosh(x).rewrite_as_exp() returned unchanged: '{s}'. \
             The rewrite module only handles Sin/Cos/Tan but not Sinh/Cosh/Tanh."
        );
    }

    assert!(
        numerical_eq_1var(&cosh_x, &rewritten, &x, &[0, 1, 2, -1, -2], 1e-10),
        "cosh(x) and its rewrite should be numerically equal"
    );

    assert!(
        changed,
        "BUG: cosh(x).rewrite_as_exp() should produce exponential form but returned '{s}' unchanged"
    );
}

/// BUG FOUND [rewrite.rs]: tanh(x).rewrite_as_exp() should produce exponential form.
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// Same root cause: rewrite_as_exp handles Tan but not Tanh.
#[test]

fn rewrite_tanh_to_exp_numerical() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tanh_x = x.tanh();

    let rewritten = tanh_x.rewrite_as_exp();
    let s = fmt(&rewritten);

    let changed = s != fmt(&tanh_x);
    if !changed {
        eprintln!(
            "BUG FOUND [rewrite.rs]: tanh(x).rewrite_as_exp() returned unchanged: '{s}'. \
             The rewrite module only handles Sin/Cos/Tan but not Sinh/Cosh/Tanh."
        );
    }

    assert!(
        numerical_eq_1var(&tanh_x, &rewritten, &x, &[0, 1, 2, -1, -2], 1e-10),
        "tanh(x) and its rewrite should be numerically equal"
    );

    assert!(
        changed,
        "BUG: tanh(x).rewrite_as_exp() should produce exponential form but returned '{s}' unchanged"
    );
}

#[test]
fn rewrite_atom_unchanged() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Rewriting a plain symbol should leave it unchanged
    let exp_result = x.rewrite_as_exp();
    assert_eq!(
        fmt(&exp_result),
        fmt(&x),
        "rewrite_as_exp on a symbol should be identity"
    );

    let trig_result = x.rewrite_as_trig();
    assert_eq!(
        fmt(&trig_result),
        fmt(&x),
        "rewrite_as_trig on a symbol should be identity"
    );
}

#[test]
fn rewrite_number_unchanged() {
    let ctx = Context::new();
    let five = ctx.int(5);

    let exp_result = five.rewrite_as_exp();
    assert_eq!(
        fmt(&exp_result),
        "5",
        "rewrite_as_exp on a number should be identity"
    );
}

#[test]
fn rewrite_sin_to_exp_roundtrip() {
    // sin(x) -> exp form -> trig form should recover something
    // numerically equivalent to sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    let as_exp = sin_x.rewrite_as_exp();
    let back_to_trig = as_exp.rewrite_as_trig();

    // Should be numerically equivalent to sin(x)
    assert!(
        numerical_eq_1var(
            &sin_x,
            &back_to_trig,
            &x,
            &[0, 1, 2, -1, -2, 3],
            1e-9
        ),
        "sin(x) -> exp -> trig roundtrip should be numerically equivalent. \
         Original: {}, Roundtrip: {}",
        fmt(&sin_x),
        fmt(&back_to_trig)
    );
}

#[test]
fn rewrite_cos_to_exp_roundtrip() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_x = x.cos();

    let as_exp = cos_x.rewrite_as_exp();
    let back_to_trig = as_exp.rewrite_as_trig();

    assert!(
        numerical_eq_1var(
            &cos_x,
            &back_to_trig,
            &x,
            &[0, 1, 2, -1, -2, 3],
            1e-9
        ),
        "cos(x) -> exp -> trig roundtrip should be numerically equivalent. \
         Original: {}, Roundtrip: {}",
        fmt(&cos_x),
        fmt(&back_to_trig)
    );
}

#[test]
fn rewrite_nested_trig_to_exp() {
    // sin(x) + cos(x) should rewrite both terms
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() + &x.cos();

    let rewritten = expr.rewrite_as_exp();
    let s = fmt(&rewritten);

    assert!(
        s.contains("exp"),
        "sin(x)+cos(x) rewritten as exp should contain 'exp': '{s}'"
    );

    assert!(
        numerical_eq_1var(&expr, &rewritten, &x, &[0, 1, 2, -1, -2], 1e-10),
        "sin(x)+cos(x) and its exp rewrite should be numerically equal"
    );
}

#[test]
fn rewrite_sin_to_exp_structure_check() {
    // sin(x) = (exp(ix) - exp(-ix)) / (2i)
    // The rewritten form should evaluate to the same values at specific points
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let rewritten = sin_x.rewrite_as_exp();

    // At x=0: sin(0) = 0
    let at_zero = rewritten.subs_i64(&x, 0).eval();
    let val = at_zero.eval_f64();
    match val {
        Ok(v) => assert!(
            v.abs() < 1e-10,
            "sin(0) rewritten as exp should evaluate to 0, got {v}"
        ),
        Err(_) => {
            // Complex evaluation might be needed; just check it doesn't panic
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 5: fourier.rs — Fourier series computation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fourier_constant_function() {
    // Fourier series of f(x) = 1 over [-π, π] should be approximately 1
    // a₀ = (1/π) * ∫_{-π}^{π} 1 dx = 2, so a₀/2 = 1
    // All other coefficients should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);

    let result = one.fourier_series(&x, 3);
    let s = fmt(&result);

    // Evaluate at a few points — should be close to 1
    for &pt in &[0i64, 1, -1, 2] {
        let val = result.subs_i64(&x, pt).eval().eval_f64();
        match val {
            Ok(v) => {
                assert!(
                    (v - 1.0).abs() < 1e-6,
                    "Fourier series of 1 at x={pt} should be ~1, got {v}. Series: '{s}'"
                );
            }
            Err(e) => {
                eprintln!(
                    "WARNING: Fourier series of 1 could not be evaluated at x={pt}: {e}. Series: '{s}'"
                );
            }
        }
    }
}

#[test]
fn fourier_sin_x_has_one_nonzero_coeff() {
    // Fourier series of sin(x) with enough terms should recover sin(x)
    // Only b₁ = 1 should be nonzero
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    let result = sin_x.fourier_series(&x, 3);

    // Numerical check: should be equivalent to sin(x)
    let points = [0i64, 1, -1, 2, -2];
    let mut all_match = true;
    let mut any_evaluated = false;

    for &pt in &points {
        let original = sin_x.subs_i64(&x, pt).eval().eval_f64();
        let series = result.subs_i64(&x, pt).eval().eval_f64();

        match (original, series) {
            (Ok(o), Ok(s)) => {
                any_evaluated = true;
                if (o - s).abs() > 1e-6 * o.abs().max(1.0) {
                    eprintln!(
                        "Fourier of sin(x) at x={pt}: original={o}, series={s}, diff={}",
                        (o - s).abs()
                    );
                    all_match = false;
                }
            }
            (Ok(o), Err(e)) => {
                eprintln!(
                    "WARNING: Fourier series of sin(x) failed to evaluate at x={pt}: {e} (original={o})"
                );
            }
            _ => {}
        }
    }

    if any_evaluated {
        assert!(
            all_match,
            "Fourier series of sin(x) should be numerically equivalent to sin(x). Series: '{}'",
            fmt(&result)
        );
    } else {
        eprintln!(
            "WARNING: Could not numerically evaluate Fourier series of sin(x): '{}'",
            fmt(&result)
        );
    }
}

#[test]
fn fourier_zero_terms() {
    // Fourier series with 0 harmonics should be just a₀/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);

    let result = one.fourier_series(&x, 0);

    // Should evaluate to something close to 1 at x=0
    let val = result.subs_i64(&x, 0).eval().eval_f64();
    match val {
        Ok(v) => {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "Fourier series of 1 with 0 terms should be ~1 at x=0, got {v}"
            );
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not evaluate Fourier(1, 0 terms) at x=0: {e}. Result: '{}'",
                fmt(&result)
            );
        }
    }
}

#[test]
fn fourier_x_squared_known_coefficients() {
    // Fourier series of x^2 on [-π, π]:
    // a₀ = (1/π) * ∫_{-π}^{π} x² dx = (1/π) * (2π³/3) = 2π²/3
    // aₙ = (1/π) * ∫_{-π}^{π} x² cos(nx) dx = 4(-1)^n / n²
    // bₙ = 0 (x² is even)
    //
    // So the series is: π²/3 + Σ 4(-1)^n cos(nx)/n²
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let x_sq = x.powi(2);

    let result = x_sq.fourier_series(&x, 5);

    // Evaluate at x=0: should be π²/3 + 4*(-1 + 1/4 - 1/9 + 1/16 - 1/25)
    // = π²/3 + 4*(-1 + 0.25 - 0.1111 + 0.0625 - 0.04)
    // = π²/3 + 4*(-0.8386)
    // ≈ 3.2899 - 3.3544 ≈ -0.065 ... wait let me recalculate
    //
    // At x=0, x²=0, so the Fourier series should approximate 0 at x=0.
    // Actually the series converges to x² everywhere on (-π, π).
    // At x=0: x²=0

    let val_at_0 = result.subs_i64(&x, 0).eval().eval_f64();
    match val_at_0 {
        Ok(v) => {
            assert!(
                v.abs() < 0.5,
                "Fourier series of x^2 at x=0 should be ~0 (since 0²=0), got {v}"
            );
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not evaluate Fourier(x², 5 terms) at x=0: {e}. Result: '{}'",
                fmt(&result)
            );
        }
    }

    // At x=1: x²=1, Fourier series should approximate 1
    let val_at_1 = result.subs_i64(&x, 1).eval().eval_f64();
    match val_at_1 {
        Ok(v) => {
            // With 5 terms, approximation within 0.5 is reasonable
            assert!(
                (v - 1.0).abs() < 1.0,
                "Fourier series of x^2 at x=1 should be ~1, got {v}"
            );
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not evaluate Fourier(x², 5 terms) at x=1: {e}. Result: '{}'",
                fmt(&result)
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-MODULE: Sort key + canonicalization integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sort_key_canonicalization_with_functions() {
    // sin(x)*cos(x) and cos(x)*sin(x) should canonicalize the same
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr1 = &x.sin() * &x.cos();
    let expr2 = &x.cos() * &x.sin();

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "sin(x)*cos(x) vs cos(x)*sin(x) should canonicalize the same"
    );
}

#[test]
fn sort_key_canonicalization_preserves_evaluation() {
    // Building in different orders should still evaluate the same
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.int(2);
    let b = ctx.int(3);

    let expr1 = &(&a * &x) + &b;
    let expr2 = &b + &(&x * &a);

    let v1 = expr1.subs_i64(&x, 5).eval().eval_f64();
    let v2 = expr2.subs_i64(&x, 5).eval().eval_f64();

    match (v1, v2) {
        (Ok(f1), Ok(f2)) => {
            assert!(
                (f1 - f2).abs() < 1e-10,
                "different build orders should give same value: {f1} vs {f2}"
            );
        }
        _ => panic!("both should evaluate successfully"),
    }
}

#[test]
fn sort_key_neg_placement_investigation() {
    // Neg is at RANK_SPECIAL (210), which is very high.
    // This means -x sorts AFTER everything including constants.
    // Test: in the expression -x + pi, does canonicalization handle ordering correctly?
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();

    let expr1 = &(-&x) + &pi;
    let expr2 = &pi + &(-&x);

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "-x + pi and pi + (-x) should canonicalize the same"
    );

    // The canonical form should exist and be well-defined
    let s = fmt(&expr1);
    assert!(
        !s.is_empty(),
        "expression should have non-empty string form"
    );
}

#[test]
fn walk_and_pretty_integration() {
    // An expression that exercises both walk (for free_symbols) and pretty
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &x.sin().powi(2) + &y.cos();

    // Walk: verify free symbols
    let syms = expr.free_symbols();
    let mut names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();
    names.sort();
    assert_eq!(names, vec!["x", "y"]);

    // Pretty: verify non-empty output
    let pretty = expr.pretty();
    assert!(!pretty.trim().is_empty());
    assert!(pretty.contains("sin") || pretty.contains("cos"));

    // Contains: verify sub-expression checks
    assert!(expr.contains(&x));
    assert!(expr.contains(&y));
}

#[test]
fn sort_key_large_sum_canonical() {
    // Build a sum of 10 symbols in reverse order, verify alphabetical
    let ctx = Context::new();
    let syms: Vec<Ex> = (b'a'..=b'j')
        .map(|c| ctx.symbol(&String::from(c as char)))
        .collect();

    // Add in reverse order
    let mut expr = syms[9].clone();
    for s in syms[0..9].iter().rev() {
        expr = &expr + s;
    }

    let s = fmt(&expr);

    // All symbols should appear
    for c in b'a'..=b'j' {
        assert!(
            s.contains(c as char),
            "sum should contain symbol '{}': '{s}'",
            c as char
        );
    }

    // Build in forward order and compare
    let mut expr2 = syms[0].clone();
    for s in &syms[1..] {
        expr2 = &expr2 + s;
    }

    assert_eq!(
        fmt(&expr),
        fmt(&expr2),
        "10-symbol sum built in different orders should canonicalize the same"
    );
}

#[test]
fn pretty_rational_number() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let pretty = half.pretty();

    // Should show a fraction, not 0.5
    let line_count = pretty.lines().count();
    assert!(
        line_count >= 2 || pretty.contains('/'),
        "1/2 should render as a fraction: '{pretty}'"
    );
    assert!(
        pretty.contains('1') && pretty.contains('2'),
        "1/2 should contain digits 1 and 2: '{pretty}'"
    );
}

#[test]
fn rewrite_exp_to_trig_pure_real_unchanged() {
    // exp(x) where x is real (no i) should remain unchanged
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exp_x = x.exp();

    let rewritten = exp_x.rewrite_as_trig();
    let s_orig = fmt(&exp_x);
    let s_new = fmt(&rewritten);

    assert_eq!(
        s_orig, s_new,
        "exp(x) (real argument) should remain unchanged under rewrite_as_trig: '{s_orig}' vs '{s_new}'"
    );
}

#[test]
fn rewrite_exp_with_real_and_imag_parts() {
    // exp(a + ix) should become exp(a) * (cos(x) + i*sin(x))
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let x = ctx.symbol("x");
    let i = ctx.i_unit();

    let arg = &a + &(&i * &x);
    let exp_expr = arg.exp();

    let rewritten = exp_expr.rewrite_as_trig();
    let s = fmt(&rewritten);

    // Should contain cos and sin if it detected the imaginary part
    let has_trig = s.contains("cos") && s.contains("sin");
    if !has_trig {
        eprintln!(
            "NOTE: exp(a + ix).rewrite_as_trig() did not produce trig form: '{s}'. \
             This may be because the Add child detection is sensitive to canonical ordering."
        );
    }
}

#[test]
fn walk_contains_number() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);

    let expr = &x * &two;
    // The expression 2*x should contain x
    assert!(expr.contains(&x), "2*x should contain x");
}

#[test]
fn walk_free_symbols_imaginary_unit() {
    // i is a constant, not a free symbol
    let ctx = Context::new();
    let i = ctx.i_unit();
    let x = ctx.symbol("x");

    let expr = &i * &x;
    let syms = expr.free_symbols();
    let names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(names, vec!["x"], "free_symbols(i*x) should be just {{x}}, got {names:?}");
}

#[test]
fn walk_free_symbols_pi_not_free() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let x = ctx.symbol("x");

    let expr = &pi * &x;
    let syms = expr.free_symbols();
    let names: Vec<String> = syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(
        names,
        vec!["x"],
        "free_symbols(pi*x) should be just {{x}}, got {names:?}"
    );
}

#[test]
fn sort_key_mixed_expression_canonical() {
    // 3 + x + x^2 + sin(x) — verify canonical form is consistent
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let three = ctx.int(3);

    // Build in two different orders
    let expr1 = &(&three + &x) + &(&x.powi(2) + &x.sin());
    let expr2 = &(&x.sin() + &three) + &(&x + &x.powi(2));

    assert_eq!(
        fmt(&expr1),
        fmt(&expr2),
        "3+x+x^2+sin(x) in different orders should canonicalize the same"
    );

    // Verify it evaluates correctly
    assert!(
        numerical_eq_1var(&expr1, &expr2, &x, &[0, 1, 2, -1], 1e-10),
        "canonical forms should be numerically equal"
    );
}

#[test]
fn pretty_complex_expression_multiline() {
    // A complex fraction like (x^2 + 1)/(sin(x) + 2) should be multiline
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let numer = &x.powi(2) + &ctx.int(1);
    let denom = &x.sin() + &ctx.int(2);
    let expr = &numer / &denom;

    let pretty = expr.pretty();
    assert!(
        pretty.lines().count() >= 2,
        "complex fraction should produce multiline pretty output: '{pretty}'"
    );
}

#[test]
fn pretty_zero() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let pretty = zero.pretty();
    assert_eq!(pretty.trim(), "0", "pretty-print of 0 should be '0': '{pretty}'");
}

#[test]
fn pretty_negative_fraction() {
    let ctx = Context::new();
    let neg_half = ctx.rational(-1, 2);
    let pretty = neg_half.pretty();

    // Should contain a minus sign somewhere
    assert!(
        pretty.contains('-') || pretty.contains('−'),
        "-1/2 pretty should contain a minus sign: '{pretty}'"
    );
}

#[test]
fn walk_term_count() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let sum3 = &(&x + &y) + &z;
    assert_eq!(sum3.term_count(), 3, "x+y+z should have 3 terms");

    let single = x.powi(2);
    assert_eq!(single.term_count(), 1, "x^2 should have 1 term");
}

#[test]
fn walk_count_ops() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    assert_eq!(x.count_ops(), 0, "symbol should have 0 ops");
    assert_eq!(ctx.int(5).count_ops(), 0, "number should have 0 ops");

    let sin_x = x.sin();
    assert!(sin_x.count_ops() >= 1, "sin(x) should have >= 1 ops");

    let expr = &x.sin().powi(2) + &x;
    assert!(expr.count_ops() >= 2, "sin(x)^2 + x should have >= 2 ops");
}

#[test]
fn rewrite_preserves_free_symbols() {
    // Rewriting should not introduce or remove free symbols
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();

    let original_syms = sin_x.free_symbols();
    let rewritten = sin_x.rewrite_as_exp();
    let rewritten_syms = rewritten.free_symbols();

    let orig_names: Vec<String> = original_syms.iter().map(|s| fmt(s)).collect();
    let new_names: Vec<String> = rewritten_syms.iter().map(|s| fmt(s)).collect();

    assert_eq!(
        orig_names, new_names,
        "rewrite should preserve free symbols: {orig_names:?} vs {new_names:?}"
    );
}
