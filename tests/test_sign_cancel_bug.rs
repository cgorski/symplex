//! Reproduction tests for the differentiation linearity bug with abs/sign.
//!
//! The proptest found that `d/dx(a+b) - (d/dx(a) + d/dx(b))` can give
//! `-sign(x)` instead of `0`. This file contains minimal reproduction
//! cases to identify the root cause.

use symplex::prelude::*;

/// Helper: check that diff linearity holds symbolically for given a, b.
/// Returns the string representation of the difference if it's not "0".
fn check_linearity(a: &Ex, b: &Ex, x: &Ex) -> Option<String> {
    let sum = a + b;
    let diff_sum = sum.diff(x);
    let diff_a = a.diff(x);
    let diff_b = b.diff(x);
    let sum_diffs = &diff_a + &diff_b;
    let difference = &diff_sum - &sum_diffs;
    let s = format!("{difference}");
    if s == "0" { None } else { Some(s) }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: Minimal reproduction cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn repro_abs_x_plus_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();
    let zero = ctx.int(0);

    let d_abs = abs_x.diff(&x);
    let sum = &abs_x + &zero;
    let d_sum = sum.diff(&x);

    println!("abs(x) = {abs_x}");
    println!("abs(x) + 0 = {sum}");
    println!("d/dx(|x|) = {d_abs}");
    println!("d/dx(|x|+0) = {d_sum}");
    println!("difference = {}", &d_sum - &d_abs);

    assert_eq!(
        format!("{d_abs}"),
        format!("{d_sum}"),
        "d/dx(|x|) should equal d/dx(|x|+0)"
    );
}

#[test]
fn repro_abs_x_plus_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    if let Some(diff) = check_linearity(&abs_x, &x, &x) {
        panic!("Linearity failed for abs(x) + x: difference = {diff}");
    }
}

#[test]
fn repro_abs_x_plus_neg_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();
    let neg_abs_x = -&abs_x;

    let d_abs = abs_x.diff(&x);
    let d_neg_abs = neg_abs_x.diff(&x);
    let sum_diffs = &d_abs + &d_neg_abs;

    println!("d/dx(|x|) = {d_abs}");
    println!("d/dx(-|x|) = {d_neg_abs}");
    println!("d/dx(|x|) + d/dx(-|x|) = {sum_diffs}");

    // abs(x) + (-abs(x)) should canonicalize to 0
    let sum = &abs_x + &neg_abs_x;
    println!("|x| + (-|x|) = {sum}");
    let d_sum = sum.diff(&x);
    println!("d/dx(|x| + (-|x|)) = {d_sum}");

    assert_eq!(
        format!("{sum_diffs}"),
        "0",
        "sign(x) + (-sign(x)) should cancel"
    );
    assert_eq!(format!("{d_sum}"), "0");
}

#[test]
fn repro_abs_x_plus_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();
    let sin_x = x.sin();

    if let Some(diff) = check_linearity(&abs_x, &sin_x, &x) {
        panic!("Linearity failed for abs(x) + sin(x): difference = {diff}");
    }
}

#[test]
fn repro_abs_x_plus_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // abs(x) + abs(x) canonicalizes to 2*abs(x)
    // d/dx(2*abs(x)) should equal d/dx(abs(x)) + d/dx(abs(x)) = 2*sign(x)
    if let Some(diff) = check_linearity(&abs_x, &abs_x, &x) {
        panic!("Linearity failed for abs(x) + abs(x): difference = {diff}");
    }
}

#[test]
fn repro_nested_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();
    let abs_abs_x = abs_x.abs();

    println!("abs(abs(x)) = {abs_abs_x}");
    println!("d/dx(abs(abs(x))) = {}", abs_abs_x.diff(&x));
    println!("d/dx(abs(x)) = {}", abs_x.diff(&x));

    // Test linearity with nested abs
    if let Some(diff) = check_linearity(&abs_abs_x, &x, &x) {
        panic!("Linearity failed for abs(abs(x)) + x: difference = {diff}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: Test sign node deduplication in arena
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sign_nodes_are_deduplicated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Differentiate abs(x) twice — both should produce the same sign(x)
    let d1 = x.abs().diff(&x);
    let d2 = x.abs().diff(&x);

    let s1 = format!("{d1}");
    let s2 = format!("{d2}");
    println!("d1 = {s1}");
    println!("d2 = {s2}");
    assert_eq!(
        s1, s2,
        "Two diff(abs(x)) calls should produce identical results"
    );

    // Now check they cancel
    let diff = &d1 - &d2;
    assert_eq!(
        format!("{diff}"),
        "0",
        "sign(x) - sign(x) should be 0, got: {diff}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: Expressions that combine abs with arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn repro_abs_of_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // abs(x + y) differentiated
    let sum_xy = &x + &y;
    let abs_sum = sum_xy.abs();
    let d = abs_sum.diff(&x);
    println!("d/dx(|x+y|) = {d}");

    if let Some(diff) = check_linearity(&abs_sum, &x, &x) {
        panic!("Linearity failed for |x+y| + x: difference = {diff}");
    }
}

#[test]
fn repro_abs_of_product() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // abs(2x) vs 2*abs(x) — these are equal but may be represented differently
    let two_x = &x * 2;
    let abs_2x = two_x.abs();
    let d = abs_2x.diff(&x);
    println!("d/dx(|2x|) = {d}");

    if let Some(diff) = check_linearity(&abs_2x, &x, &x) {
        panic!("Linearity failed for |2x| + x: difference = {diff}");
    }
}

#[test]
fn repro_mul_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // x * abs(x) — product rule should give abs(x) + x*sign(x)
    let x_times_abs = &x * &abs_x;
    let d = x_times_abs.diff(&x);
    println!("d/dx(x*|x|) = {d}");

    if let Some(diff) = check_linearity(&x_times_abs, &x, &x) {
        panic!("Linearity failed for x*|x| + x: difference = {diff}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 4: Expressions from the proptest generator shapes
// (depth 2 expressions combining abs with other operations)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn repro_neg_abs_x_plus_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();
    let neg_abs = -&abs_x;

    // -abs(x) + abs(x) should be 0 => d/dx(0) = 0
    // But d/dx(-abs(x)) + d/dx(abs(x)) = -sign(x) + sign(x) = 0
    if let Some(diff) = check_linearity(&neg_abs, &abs_x, &x) {
        panic!("Linearity failed for -|x| + |x|: difference = {diff}");
    }
}

#[test]
fn repro_abs_x_minus_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // Subtraction: abs(x) - abs(x) = 0
    let diff_expr = &abs_x - &abs_x;
    println!("|x| - |x| = {diff_expr}");
    let d = diff_expr.diff(&x);
    println!("d/dx(|x| - |x|) = {d}");

    // Individually
    let d_abs = abs_x.diff(&x);
    let d_neg_abs = (-&abs_x).diff(&x);
    let sum = &d_abs + &d_neg_abs;
    println!("d/dx(|x|) + d/dx(-|x|) = {sum}");

    assert_eq!(format!("{d}"), "0");
    assert_eq!(format!("{sum}"), "0");
}

#[test]
fn repro_abs_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let abs_sin = sin_x.abs();

    let d = abs_sin.diff(&x);
    println!("d/dx(|sin(x)|) = {d}");

    if let Some(diff) = check_linearity(&abs_sin, &x, &x) {
        panic!("Linearity failed for |sin(x)| + x: difference = {diff}");
    }
}

#[test]
fn repro_abs_x_powi_2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // abs(x)^2 = x^2, so d/dx should be 2x
    let abs_sq = abs_x.powi(2);
    let d = abs_sq.diff(&x);
    println!("d/dx(|x|^2) = {d}");

    if let Some(diff) = check_linearity(&abs_sq, &x, &x) {
        panic!("Linearity failed for |x|^2 + x: difference = {diff}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 5: Systematic sweep of binary operations with abs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn systematic_linearity_sweep_with_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Build a bunch of expressions involving abs
    let exprs: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("y", y.clone()),
        ("|x|", x.abs()),
        ("-|x|", -&x.abs()),
        ("2*|x|", &x.abs() * 2),
        ("|x| + x", &x.abs() + &x),
        ("x*|x|", &x * &x.abs()),
        ("|sin(x)|", x.sin().abs()),
        ("sin(|x|)", x.abs().sin()),
        ("|x|^2", x.abs().powi(2)),
        ("||x||", x.abs().abs()),
        ("x^2", x.powi(2)),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("-x", -&x),
        ("1", ctx.int(1)),
        ("0", ctx.int(0)),
        ("-1", ctx.int(-1)),
        ("3", ctx.int(3)),
    ];

    let mut failures = Vec::new();

    for (name_a, a) in &exprs {
        for (name_b, b) in &exprs {
            if let Some(diff) = check_linearity(a, b, &x) {
                failures.push(format!(
                    "  a={name_a}, b={name_b}: d/dx(a+b) - (d/dx(a)+d/dx(b)) = {diff}"
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Diff linearity failures found ({} total):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 6: Direct arena-level investigation
// (check if sign(x) created via different paths has same representation)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sign_from_diff_vs_direct_creation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // Get sign(x) via differentiation
    let sign_via_diff = abs_x.diff(&x);

    // Get sign(x) via direct creation
    let sign_direct = x.sign();

    let s1 = format!("{sign_via_diff}");
    let s2 = format!("{sign_direct}");
    println!("sign via diff(|x|) = {s1}");
    println!("sign direct = {s2}");

    assert_eq!(
        s1, s2,
        "sign(x) from diff(abs(x)) should match directly-created sign(x)"
    );

    // They should cancel
    let diff = &sign_via_diff - &sign_direct;
    assert_eq!(
        format!("{diff}"),
        "0",
        "sign(x) from two sources should cancel, got: {diff}"
    );
}

#[test]
fn sign_cancellation_after_add() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let sign_x = x.sign();
    let neg_sign_x = -&sign_x;

    // Direct cancellation
    let sum = &sign_x + &neg_sign_x;
    println!("sign(x) + (-sign(x)) = {sum}");
    assert_eq!(format!("{sum}"), "0", "sign(x) + (-sign(x)) should be 0");

    // Via Mul(-1, sign(x))
    let neg_sign_via_mul = &sign_x * &ctx.int(-1);
    let sum2 = &sign_x + &neg_sign_via_mul;
    println!("sign(x) + sign(x)*(-1) = {sum2}");
    assert_eq!(format!("{sum2}"), "0", "sign(x) + (-1)*sign(x) should be 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 7: The exact proptest scenario — arb_expr(2) style expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn repro_proptest_abs_combined_with_subtraction() {
    // The proptest uses arb_expr(2) which can generate depth-2 trees.
    // A common pattern that might fail:
    // a = abs(x - y), b = x - y
    // or a = abs(some_expr), b = -some_expr
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Try: a = abs(x), b = -x (this creates abs(x) + (-x) = abs(x) - x)
    let a = x.abs();
    let b = -&x;
    if let Some(diff) = check_linearity(&a, &b, &x) {
        panic!("Linearity failed for |x| + (-x): difference = {diff}");
    }

    // Try: a = abs(-x), b = x
    let a2 = (-&x).abs();
    let b2 = x.clone();
    println!("|(-x)| = {a2}");
    println!("d/dx(|(-x)|) = {}", a2.diff(&x));
    if let Some(diff) = check_linearity(&a2, &b2, &x) {
        panic!("Linearity failed for |(-x)| + x: difference = {diff}");
    }

    // Try: a = abs(x) * abs(x), b = -x^2
    let a3 = &x.abs() * &x.abs();
    let b3 = -&x.powi(2);
    if let Some(diff) = check_linearity(&a3, &b3, &x) {
        panic!("Linearity failed for |x|*|x| + (-x^2): difference = {diff}");
    }
}

#[test]
fn repro_abs_of_neg_x() {
    // abs(-x) should ideally equal abs(x), but if not, diff paths diverge
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let neg_x = -&x;
    let abs_neg_x = neg_x.abs();
    let abs_x = x.abs();

    println!("|x| = {abs_x}");
    println!("|(-x)| = {abs_neg_x}");
    println!("d/dx(|x|) = {}", abs_x.diff(&x));
    println!("d/dx(|(-x)|) = {}", abs_neg_x.diff(&x));

    // Even if they're structurally different, the derivatives should be equal
    let d1 = abs_x.diff(&x);
    let d2 = abs_neg_x.diff(&x);
    let diff = &d1 - &d2;
    println!("d/dx(|x|) - d/dx(|(-x)|) = {diff}");

    // This is the key test — if this fails, the root cause is that
    // sign(-x) * (-1) doesn't simplify to sign(x)
    assert_eq!(
        format!("{diff}"),
        "0",
        "d/dx(|x|) and d/dx(|-x|) should be equal, got difference: {diff}"
    );
}

#[test]
fn repro_depth2_abs_of_subtraction() {
    // arb_expr(2) can generate: abs(a - b) where a, b are leaves
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exprs_to_try: Vec<(&str, Ex)> = vec![
        ("x - 1", &x - 1),
        ("x + 1", &x + 1),
        ("2*x", &x * 2),
        ("-x", -&x),
        ("x - x", &x - &x),
    ];

    for (name, inner) in &exprs_to_try {
        let abs_inner = inner.abs();
        let d = abs_inner.diff(&x);
        println!("d/dx(|{name}|) = {d}");

        // Check linearity with a constant
        if let Some(diff) = check_linearity(&abs_inner, &ctx.int(1), &x) {
            println!("  FAIL: |{name}| + 1 linearity: {diff}");
        }
        // Check linearity with x
        if let Some(diff) = check_linearity(&abs_inner, &x, &x) {
            println!("  FAIL: |{name}| + x linearity: {diff}");
        }
        // Check linearity with another abs
        if let Some(diff) = check_linearity(&abs_inner, &x.abs(), &x) {
            println!("  FAIL: |{name}| + |x| linearity: {diff}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 8: Exhaustive depth-2 combinations (the proptest shape)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exhaustive_depth2_linearity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Depth-1 expressions
    let d1: Vec<(&str, Ex)> = vec![
        ("x", x.clone()),
        ("-x", -&x),
        ("|x|", x.abs()),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
        ("x^2", x.powi(2)),
        ("x^(-1)", x.powi(-1)),
        ("1", ctx.int(1)),
        ("0", ctx.int(0)),
        ("-1", ctx.int(-1)),
        ("2", ctx.int(2)),
    ];

    // Depth-2: unary operations applied to depth-1
    let mut d2: Vec<(String, Ex)> = Vec::new();
    for (name, e) in &d1 {
        d2.push((format!("-({name})"), -e));
        d2.push((format!("|{name}|"), e.abs()));
        d2.push((format!("sin({name})"), e.sin()));
    }
    // Also add some binary combos
    for (na, a) in &d1 {
        for (nb, b) in &d1 {
            d2.push((format!("{na}+{nb}"), a + b));
            d2.push((format!("{na}*{nb}"), a * b));
            d2.push((format!("{na}-{nb}"), a - b));
        }
    }

    let mut failures = Vec::new();

    // Test linearity for all depth-2 pairs
    for (na, a) in &d2 {
        for (nb, b) in &d2 {
            if let Some(diff) = check_linearity(a, b, &x) {
                let msg = format!("  a={na}, b={nb}: difference = {diff}");
                if !failures.contains(&msg) {
                    failures.push(msg);
                }
            }
        }
    }

    if !failures.is_empty() {
        // Print first 20 failures for diagnosis
        let show = failures.len().min(20);
        panic!(
            "Diff linearity failures found ({} total, showing first {show}):\n{}",
            failures.len(),
            failures[..show].join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 9: Detailed trace of the diff computation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn detailed_trace_abs_diff_linearity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let abs_x = x.abs();

    // Direct diff of abs(x)
    let d1 = abs_x.diff(&x);
    println!("d/dx(|x|) = {d1}");

    // diff(abs(x) + 0) — since abs(x)+0 = abs(x), this should be identical
    let sum = &abs_x + &ctx.int(0);
    println!("|x| + 0 = {sum}");
    let d_sum = sum.diff(&x);
    println!("d/dx(|x| + 0) = {d_sum}");

    // diff(abs(x)) + diff(0)
    let d_abs = abs_x.diff(&x);
    let d_zero = ctx.int(0).diff(&x);
    let reconstructed = &d_abs + &d_zero;
    println!("d/dx(|x|) + d/dx(0) = {reconstructed}");
    println!("difference = {}", &d_sum - &reconstructed);

    // Now try with a non-trivial partner
    let sin_x = x.sin();
    let sum2 = &abs_x + &sin_x;
    println!("\n|x| + sin(x) = {sum2}");

    let d_sum2 = sum2.diff(&x);
    let d_abs2 = abs_x.diff(&x);
    let d_sin = sin_x.diff(&x);
    let reconstructed2 = &d_abs2 + &d_sin;

    println!("d/dx(|x| + sin(x)) = {d_sum2}");
    println!("d/dx(|x|) + d/dx(sin(x)) = {reconstructed2}");
    println!("difference = {}", &d_sum2 - &reconstructed2);

    // Verify the string representations
    assert_eq!(
        format!("{d_sum2}"),
        format!("{reconstructed2}"),
        "Linearity should hold for |x| + sin(x)"
    );
}
