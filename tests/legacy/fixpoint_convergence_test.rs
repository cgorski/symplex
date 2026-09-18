//! Fixpoint convergence tests for iterated simplification.
//!
//! The proposal is to make `.simplify()` iterate `smart_simplify` up to 10
//! times until the expression stops changing.  These tests verify whether
//! that is safe — i.e., whether the iteration converges or can oscillate
//! between two (or more) forms indefinitely.
//!
//! Key risks:
//!   - `expand` distributes: `a*(b+c)` → `a*b + a*c`
//!   - `factor_terms` extracts: `a*b + a*c` → `a*(b+c)`
//!   - These are inverses. If both strategies tie on op count, the winner
//!     could alternate between iterations.
//!   - `fu()` may rewrite `sin²(x)` ↔ `(1 - cos(2x))/2`
//!   - `expand_trig` expands `sin(a+b)` into products; `fu` may recombine.
//!   - The 1.7× bloat guard compares against the *current* input to
//!     `smart_simplify`, not the original expression from iteration 0.
//!     So 10 iterations could compound up to 1.7^10 ≈ 202× bloat.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Iterate `.simplify()` up to `max_iters` times, recording the string form
/// at each step.  Returns `(forms, converged_at)` where `converged_at` is
/// `Some(i)` if `forms[i] == forms[i-1]` (fixpoint reached), or `None` if
/// it never stabilized.
fn iterate_simplify(expr: &Ex, max_iters: usize) -> (Vec<String>, Option<usize>) {
    let mut current = expr.clone();
    let mut forms: Vec<String> = vec![format!("{current}")];

    for i in 1..=max_iters {
        let next = current.simplify();
        let s = format!("{next}");
        if s == forms[i - 1] {
            forms.push(s);
            return (forms, Some(i));
        }
        forms.push(s);
        current = next;
    }
    (forms, None)
}

/// Same as above but uses `.simplify()` directly (the engine-level
/// function that tries 15 strategies).
fn iterate_smart_simplify(expr: &Ex, max_iters: usize) -> (Vec<String>, Option<usize>) {
    let mut current = expr.clone();
    let mut forms: Vec<String> = vec![format!("{current}")];

    for i in 1..=max_iters {
        let next = current.simplify();
        let s = format!("{next}");
        if s == forms[i - 1] {
            forms.push(s);
            return (forms, Some(i));
        }
        forms.push(s);
        current = next;
    }
    (forms, None)
}

/// Same as above but uses `.simplify()` (the existing fixpoint loop).
fn iterate_full_simplify(expr: &Ex, max_iters: usize) -> (Vec<String>, Option<usize>) {
    let mut current = expr.clone();
    let mut forms: Vec<String> = vec![format!("{current}")];

    for i in 1..=max_iters {
        let next = current.simplify();
        let s = format!("{next}");
        if s == forms[i - 1] {
            forms.push(s);
            return (forms, Some(i));
        }
        forms.push(s);
        current = next;
    }
    (forms, None)
}

/// Detect a cycle in the forms list.  Returns `Some((cycle_start, cycle_len))`
/// if `forms[i] == forms[j]` for some `j > i`, meaning the expression
/// oscillated back to a previous form.
fn detect_cycle(forms: &[String]) -> Option<(usize, usize)> {
    for i in 0..forms.len() {
        for j in (i + 1)..forms.len() {
            if forms[i] == forms[j] {
                // If i+1 < j, we have a non-trivial cycle (not just fixpoint)
                if j - i > 1 {
                    return Some((i, j - i));
                }
            }
        }
    }
    None
}

/// Print the iteration trace for debugging.
fn print_trace(label: &str, forms: &[String], converged_at: Option<usize>) {
    println!("\n{}", "=".repeat(60));
    println!("  {label}");
    println!("{}", "=".repeat(60));
    for (i, form) in forms.iter().enumerate() {
        let marker = match converged_at {
            Some(c) if i == c => " ← CONVERGED",
            _ => "",
        };
        println!("  iter {i:2}: {form}{marker}");
    }
    if let Some(cycle) = detect_cycle(forms) {
        println!("  ⚠ CYCLE detected: start={}, length={}", cycle.0, cycle.1);
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: expand vs factor_terms oscillation
//
// `2*(x + 1)` could expand to `2*x + 2`, which factor_terms pulls back to
// `2*(x + 1)`.  If both forms have the same op count, smart_simplify might
// alternate.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_expand_vs_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // 2*(x + 1) — classic expand/factor tug-of-war
    let expr = &(&x + 1) * 2;
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("2*(x + 1) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in 2*(x+1): cycle of length {} starting at iteration {}.\n\
                 Forms: {:?}",
            cycle.1,
            cycle.0,
            &forms[cycle.0..cycle.0 + cycle.1 + 1]
        );
    }
    // Even if it doesn't oscillate, it should converge within 20 iterations
    assert!(
        converged.is_some(),
        "2*(x+1) did not converge in 20 iterations of .simplify().\nForms: {forms:?}"
    );
}

#[test]
fn convergence_expand_vs_factor_larger() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // 3*(x + y + 1) — more terms to distribute/collect
    let expr = &(&x + &y + 1) * 3;
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("3*(x + y + 1) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in 3*(x+y+1): cycle length {} at iteration {}",
            cycle.1, cycle.0
        );
    }
    assert!(
        converged.is_some(),
        "3*(x+y+1) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: trig identity oscillation
//
// sin²(x) could be rewritten as (1 - cos(2x))/2 by fu (TRpower), but then
// expand_trig or pattern rules might rewrite it back.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_sin_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = x.sin().powi(2);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("sin²(x) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in sin²(x): cycle length {} at iteration {}.\nForms: {:?}",
            cycle.1,
            cycle.0,
            &forms[cycle.0..std::cmp::min(forms.len(), cycle.0 + cycle.1 + 1)]
        );
    }
    assert!(
        converged.is_some(),
        "sin²(x) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_half_angle_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (1 - cos(2x))/2 — the other form of sin²(x).  Does simplify flip it back?
    let two_x = &x * 2;
    let expr = &(&ctx.int(1) - &two_x.cos()) / 2;
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("(1 - cos(2x))/2 via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in (1-cos(2x))/2: cycle length {} at iteration {}",
            cycle.1, cycle.0
        );
    }
    assert!(
        converged.is_some(),
        "(1-cos(2x))/2 did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin²(x) + cos²(x) — should simplify to 1 and stay there
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("sin²(x) + cos²(x) via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "sin²+cos² did not converge in 20 iterations.\nForms: {forms:?}"
    );
    // Should converge to "1"
    let final_form = forms.last().unwrap();
    assert_eq!(final_form, "1", "sin²+cos² should simplify to 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: exp/ln oscillation
//
// exp(ln(x) + ln(y)) could be simplified to x*y, but could a later
// iteration try to log-combine or do something weird?
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_exp_ln_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // exp(ln(x) + ln(y))
    let expr = (&x.ln() + &y.ln()).exp();
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("exp(ln(x) + ln(y)) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in exp(ln(x)+ln(y)): cycle length {} at iteration {}",
            cycle.1, cycle.0
        );
    }
    assert!(
        converged.is_some(),
        "exp(ln(x)+ln(y)) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_ln_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // ln(exp(x)) — inverse pair
    let expr = x.exp().ln();
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("ln(exp(x)) via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "ln(exp(x)) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: smart_simplify specifically (the 15-strategy engine)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_smart_simplify_expand_factor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x + 1) * 2;
    let (forms, converged) = iterate_smart_simplify(&expr, 20);
    print_trace("2*(x+1) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION via smart_simplify on 2*(x+1): cycle length {}",
            cycle.1
        );
    }
    assert!(
        converged.is_some(),
        "smart_simplify on 2*(x+1) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_smart_simplify_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = x.sin().powi(2);
    let (forms, converged) = iterate_smart_simplify(&expr, 20);
    print_trace("sin²(x) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION via smart_simplify on sin²(x): cycle length {}",
            cycle.1
        );
    }
    assert!(
        converged.is_some(),
        "smart_simplify on sin²(x) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: full_simplify idempotence
//
// full_simplify already does a fixpoint loop internally (up to 10 iters).
// Calling it AGAIN on its own output should be a no-op.  If it isn't,
// that means the internal loop's convergence check is broken.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simplify_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let exprs: Vec<(&str, Ex)> = vec![
        ("2*(x+1)", &(&x + 1) * 2),
        ("sin²(x)", x.sin().powi(2)),
        ("sin²+cos²", &x.sin().powi(2) + &x.cos().powi(2)),
        ("exp(ln(x)+ln(y))", (&x.ln() + &y.ln()).exp()),
        ("(x²-1)/(x-1)", &(&x.powi(2) - 1) / &(&x - 1)),
        ("(x+1)^3", (&x + 1).powi(3)),
    ];

    for (label, expr) in &exprs {
        let once = expr.simplify();
        let twice = once.simplify();
        let s1 = format!("{once}");
        let s2 = format!("{twice}");
        println!("full_simplify idempotence: {label}");
        println!("  once:  {s1}");
        println!("  twice: {s2}");
        assert_eq!(
            s1, s2,
            "full_simplify is NOT idempotent on {label}: first={s1}, second={s2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: bloat guard compounding analysis
//
// The 1.7× bloat guard in smart_simplify compares against the input to
// THAT call, not the original expression.  So if each call returns
// something ≤ 1.7× its input, after N iterations we could have
// 1.7^N × original ops.  For N=10: 1.7^10 ≈ 202×.
//
// This test checks whether op counts actually grow across iterations.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bloat_guard_does_not_compound() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // Construct a moderately complex expression
    let expr = &(&x.sin().powi(2) + &y.cos().powi(2)) * &(&x + &y);

    let mut current = expr.clone();
    let initial_display = format!("{current}");
    let mut op_counts: Vec<String> = Vec::new();

    println!("\nBloat guard compounding test on: {initial_display}");
    for i in 0..10 {
        let display = format!("{current}");
        let len = display.len(); // proxy for complexity
        op_counts.push(format!("iter {i}: len={len} form={display}"));
        let next = current.simplify();
        if format!("{next}") == display {
            println!("  Converged at iteration {i}");
            break;
        }
        current = next;
    }

    for line in &op_counts {
        println!("  {line}");
    }

    // The final form should not be more than 3× the length of the initial form
    // (a very generous bound — the real concern is 202× blowup)
    let final_len = format!("{current}").len();
    let initial_len = initial_display.len();
    if initial_len > 0 {
        let ratio = final_len as f64 / initial_len as f64;
        println!(
            "  Length ratio (final/initial): {:.2}× ({final_len}/{initial_len})",
            ratio
        );
        assert!(
            ratio < 5.0,
            "Expression bloated {ratio:.1}× over {initial_len} chars after iterated smart_simplify. \
             The 1.7× per-call bloat guard is compounding!"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: count_ops determinism
//
// count_ops counts non-atom nodes.  Two equivalent expressions could have
// the same op count but different forms.  If smart_simplify breaks ties
// non-deterministically (e.g., by ExprId ordering that depends on arena
// allocation), repeated calls could return different results.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn smart_simplify_deterministic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exprs: Vec<(&str, Ex)> = vec![
        ("2*(x+1)", &(&x + 1) * 2),
        ("sin²(x)", x.sin().powi(2)),
        ("x + x", &x + &x),
        ("(x+1)^2", (&x + 1).powi(2)),
    ];

    for (label, expr) in &exprs {
        let results: Vec<String> = (0..5).map(|_| format!("{}", expr.simplify())).collect();

        let all_same = results.iter().all(|r| r == &results[0]);
        println!("Determinism check: {label}");
        for (i, r) in results.iter().enumerate() {
            println!("  call {i}: {r}");
        }
        assert!(
            all_same,
            "smart_simplify is NON-DETERMINISTIC on {label}: {results:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: apply_until_stable with .simplify()
//
// This is exactly the proposed pattern — use the existing
// apply_until_stable infrastructure with simplify.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn apply_until_stable_simplify_converges() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let test_cases: Vec<(&str, Ex)> = vec![
        ("2*(x+1)", &(&x + 1) * 2),
        ("sin²(x) + cos²(x)", &x.sin().powi(2) + &x.cos().powi(2)),
        ("exp(ln(x)+ln(y))", (&x.ln() + &y.ln()).exp()),
        ("sin²(x)", x.sin().powi(2)),
        ("(x²-1)/(x-1)", &(&x.powi(2) - 1) / &(&x - 1)),
    ];

    for (label, expr) in &test_cases {
        let (result, iters) = expr.apply_until_stable(10, |e| e.simplify());
        let result_str = format!("{result}");
        println!(
            "apply_until_stable(.simplify(), 10) on {label}: converged in {iters} iters → {result_str}"
        );

        // Should converge before hitting max
        assert!(
            iters < 10,
            "{label} hit max iterations (10) in apply_until_stable with .simplify(). \
             This means fixpoint iteration is NOT converging! Final form: {result_str}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: harder expressions that stress expand/factor interplay
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_polynomial_expand_factor_stress() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // (x+y)^2 - x^2 - 2*x*y - y^2 should simplify to 0
    let expr = &(&x + &y).powi(2) - &x.powi(2) - &(&x * &y) * 2 - &y.powi(2);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace(
        "(x+y)^2 - x^2 - 2xy - y^2 via .simplify()",
        &forms,
        converged,
    );

    assert!(
        converged.is_some(),
        "polynomial identity did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_nested_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin(2x) — could be expanded to 2*sin(x)*cos(x) and maybe back
    let expr = (&x * 2).sin();
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("sin(2x) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in sin(2x): cycle length {} at iteration {}.\n\
                 This likely means expand_trig and fu are fighting.",
            cycle.1, cycle.0
        );
    }
    assert!(
        converged.is_some(),
        "sin(2x) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_cos_double_angle() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // cos(2x) — has multiple equivalent forms:
    //   cos²(x) - sin²(x)
    //   2cos²(x) - 1
    //   1 - 2sin²(x)
    let expr = (&x * 2).cos();
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("cos(2x) via .simplify()", &forms, converged);

    if let Some(cycle) = detect_cycle(&forms)
        && cycle.1 > 1
    {
        panic!(
            "OSCILLATION in cos(2x): cycle length {} at iteration {}",
            cycle.1, cycle.0
        );
    }
    assert!(
        converged.is_some(),
        "cos(2x) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: mixed trig + polynomial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_trig_times_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (sin²(x) + cos²(x)) * (x + 1) should simplify to x + 1
    let expr = &(&x.sin().powi(2) + &x.cos().powi(2)) * &(&x + 1);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("(sin²+cos²)*(x+1) via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "(sin²+cos²)*(x+1) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: rational expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_rational_cancel() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (x^3 - x) / (x^2 - 1) = x*(x-1)*(x+1) / ((x-1)*(x+1)) = x
    let expr = &(&x.powi(3) - &x) / &(&x.powi(2) - 1);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("(x³-x)/(x²-1) via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "(x³-x)/(x²-1) did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Test: power expressions (powsimp / powdenest interplay)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convergence_power_simplification() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x^2 * x^3 → x^5  (should be stable)
    let expr = &x.powi(2) * &x.powi(3);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("x^2 * x^3 via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "x^2*x^3 did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

#[test]
fn convergence_sqrt_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sqrt(x)^2  — powsimp vs powdenest
    let expr = x.sqrt().powi(2);
    let (forms, converged) = iterate_simplify(&expr, 20);
    print_trace("sqrt(x)^2 via .simplify()", &forms, converged);

    assert!(
        converged.is_some(),
        "sqrt(x)^2 did not converge in 20 iterations.\nForms: {forms:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Comprehensive: run full_simplify iterated on ALL candidate oscillators
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_simplify_iterated_convergence_suite() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let test_cases: Vec<(&str, Ex)> = vec![
        ("2*(x+1)", &(&x + 1) * 2),
        ("3*(x+y+1)", &(&x + &y + 1) * 3),
        ("sin²(x)", x.sin().powi(2)),
        ("cos²(x)", x.cos().powi(2)),
        ("sin²+cos²", &x.sin().powi(2) + &x.cos().powi(2)),
        ("sin(2x)", (&x * 2).sin()),
        ("cos(2x)", (&x * 2).cos()),
        ("exp(ln(x))", x.ln().exp()),
        ("ln(exp(x))", x.exp().ln()),
        ("exp(ln(x)+ln(y))", (&x.ln() + &y.ln()).exp()),
        ("(x²-1)/(x-1)", &(&x.powi(2) - 1) / &(&x - 1)),
        ("(x+1)^3", (&x + 1).powi(3)),
        ("x^2*x^3", &x.powi(2) * &x.powi(3)),
        ("sqrt(x)^2", x.sqrt().powi(2)),
    ];

    let mut any_failed = false;

    for (label, expr) in &test_cases {
        let (forms, converged) = iterate_full_simplify(expr, 5);

        let has_cycle = detect_cycle(&forms).map(|c| c.1 > 1).unwrap_or(false);

        let status = if has_cycle {
            any_failed = true;
            "⚠ OSCILLATING"
        } else if converged.is_some() {
            "✓ converged"
        } else {
            any_failed = true;
            "✗ not converged"
        };

        let final_form = forms.last().unwrap();
        let n_iters = converged.unwrap_or(forms.len() - 1);
        println!("  {status:18} {label:30} → {final_form:30} (iters: {n_iters})");
    }

    if any_failed {
        panic!("Some expressions did not converge under iterated full_simplify. See output above.");
    }
}
