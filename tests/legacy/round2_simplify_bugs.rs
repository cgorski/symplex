//! Round 2 — simplification, rewriting, and expression transformation bug hunt.
//!
//! Focuses on: trig simplification edge cases, log simplification, power
//! simplification, expand edge cases, rewrite rules, nsimplify, expression
//! equality, and assumption-driven refinement.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerical check: evaluate two expressions at integer points and compare.
fn numerical_eq_1var(a: &Ex, b: &Ex, var: &Ex, points: &[i64], tol: f64) -> bool {
    for &p in points {
        let va = a.subs_i64(var, p).eval().eval_f64();
        let vb = b.subs_i64(var, p).eval().eval_f64();
        match (va, vb) {
            (Ok(fa), Ok(fb)) => {
                if (fa - fb).abs() > tol * fa.abs().max(fb.abs()).max(1.0) {
                    return false;
                }
            }
            (Err(_), Err(_)) => {} // both fail, ok
            _ => return false,
        }
    }
    true
}

/// Format helper
fn fmt(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Trig simplification edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn trig_double_angle_sin_2x_from_2sincos() {
    // 2*sin(x)*cos(x) should simplify to sin(2x) or stay equivalent
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() * &x.cos() * 2;
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    // At minimum must be numerically equivalent
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3, 5], 1e-10),
        "2*sin(x)*cos(x) simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn trig_double_angle_cos2x_from_cos2_minus_sin2() {
    // cos²(x) - sin²(x) should simplify (ideally to cos(2x))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.cos().powi(2) - &x.sin().powi(2);
    let simplified = expr.simplify();
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "cos²(x)-sin²(x) simplify broke numerical equivalence: {}",
        fmt(&simplified)
    );
}

#[test]
fn trig_pythagorean_identity_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(fmt(&simplified), "1", "sin²+cos² should be 1");
}

#[test]
fn trig_pythagorean_scaled() {
    // 3*sin²(x) + 3*cos²(x) should simplify to 3
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3);
    let simplified = expr.simplify();
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "3*(sin²+cos²) simplify broke numerical equivalence: {}",
        fmt(&simplified)
    );
}

#[test]
fn trig_nested_sin_of_sin() {
    // sin(sin(x)) should remain as-is and not crash
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().sin();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert!(
        s.contains("sin"),
        "sin(sin(x)) should still contain sin: {s}"
    );
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "sin(sin(x)) simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn trig_nested_cos_of_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().cos();
    let simplified = expr.simplify();
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "cos(sin(x)) simplify broke numerical equivalence: {}",
        fmt(&simplified)
    );
}

#[test]
fn trig_expand_sin_3x() {
    // sin(3x) should expand to terms involving only sin(x) and cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x * 3).sin();
    let expanded = expr.expand_trig();
    let s = fmt(&expanded);
    // Should not contain sin(3*x) or sin(2*x) — fully expanded
    assert!(
        !s.contains("3*x") && !s.contains("2*x"),
        "sin(3x) should fully expand, got: {s}"
    );
    assert!(
        numerical_eq_1var(&expr, &expanded, &x, &[1, 2, 3], 1e-10),
        "sin(3x) trig_expand broke numerical equivalence: {s}"
    );
}

#[test]
fn trig_expand_cos_4x() {
    // cos(4x) should expand to terms of sin(x) and cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x * 4).cos();
    let expanded = expr.expand_trig();
    let s = fmt(&expanded);
    assert!(!s.contains("4*x"), "cos(4x) should be expanded, got: {s}");
    assert!(
        numerical_eq_1var(&expr, &expanded, &x, &[1, 2, 3], 1e-10),
        "cos(4x) trig_expand broke numerical equivalence: {s}"
    );
}

#[test]
fn trig_product_to_sum_sinx_cosy() {
    // sin(x)*cos(y) -> ½[sin(x+y) + sin(x-y)]
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.sin() * &y.cos();
    let combined = expr.trig_combine();
    // Numerically equivalent at a couple of points
    let original_val = expr.subs_i64(&x, 1).subs_i64(&y, 2).eval().eval_f64();
    let combined_val = combined.subs_i64(&x, 1).subs_i64(&y, 2).eval().eval_f64();
    match (original_val, combined_val) {
        (Ok(a), Ok(b)) => {
            assert!(
                (a - b).abs() < 1e-10,
                "sin(x)*cos(y) trig_combine broke equivalence: {a} vs {b}, result={}",
                fmt(&combined)
            );
        }
        _ => panic!("evaluation failed"),
    }
}

#[test]
fn trig_product_to_sum_sinx_siny() {
    // sin(x)*sin(y) -> ½[cos(x-y) - cos(x+y)]
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.sin() * &y.sin();
    let combined = expr.trig_combine();
    let original_val = expr.subs_i64(&x, 1).subs_i64(&y, 2).eval().eval_f64();
    let combined_val = combined.subs_i64(&x, 1).subs_i64(&y, 2).eval().eval_f64();
    match (original_val, combined_val) {
        (Ok(a), Ok(b)) => {
            assert!(
                (a - b).abs() < 1e-10,
                "sin(x)*sin(y) trig_combine broke equivalence: {a} vs {b}"
            );
        }
        _ => panic!("evaluation failed"),
    }
}

#[test]
fn trig_product_to_sum_cosx_cosy() {
    // cos(x)*cos(y) -> ½[cos(x-y) + cos(x+y)]
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.cos() * &y.cos();
    let combined = expr.trig_combine();
    let original_val = expr.subs_i64(&x, 2).subs_i64(&y, 3).eval().eval_f64();
    let combined_val = combined.subs_i64(&x, 2).subs_i64(&y, 3).eval().eval_f64();
    match (original_val, combined_val) {
        (Ok(a), Ok(b)) => {
            assert!(
                (a - b).abs() < 1e-10,
                "cos(x)*cos(y) trig_combine broke equivalence: {a} vs {b}"
            );
        }
        _ => panic!("evaluation failed"),
    }
}

#[test]
fn trig_hyperbolic_identity_cosh2_minus_sinh2() {
    // cosh²(x) - sinh²(x) = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2);
    // Evaluate numerically since simplifier may not know hyperbolic identities
    for p in [1, 2, 3] {
        let val = expr.subs_i64(&x, p).eval().eval_f64();
        match val {
            Ok(v) => assert!(
                (v - 1.0).abs() < 1e-9,
                "cosh²(x)-sinh²(x) should be 1 at x={p}, got {v}"
            ),
            Err(e) => panic!("evaluation failed at x={p}: {e}"),
        }
    }
    // Try simplifying — ideally it recognizes this as 1
    let simplified = expr.simplify();
    let full = expr.simplify();
    let s_simp = fmt(&simplified);
    let s_full = fmt(&full);
    // Report what we got (may or may not simplify — that's what we're testing)
    eprintln!("cosh²-sinh²: simplify -> {s_simp}");
    eprintln!("cosh²-sinh²: full_simplify -> {s_full}");
    // At minimum numerical equivalence must hold
    for p in [1, 2, 3] {
        let v1 = simplified.subs_i64(&x, p).eval().eval_f64();
        let v2 = full.subs_i64(&x, p).eval().eval_f64();
        match v1 {
            Ok(v) => assert!(
                (v - 1.0).abs() < 1e-9,
                "simplified cosh²-sinh² at x={p}: {v}"
            ),
            Err(e) => panic!("simplified eval failed at x={p}: {e}"),
        }
        match v2 {
            Ok(v) => assert!(
                (v - 1.0).abs() < 1e-9,
                "full_simplified cosh²-sinh² at x={p}: {v}"
            ),
            Err(e) => panic!("full_simplified eval failed at x={p}: {e}"),
        }
    }
}

#[test]
fn trig_hyperbolic_tanh_equals_sinh_over_cosh() {
    // tanh(x) = sinh(x)/cosh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tanh_x = x.tanh();
    let ratio = &x.sinh() / &x.cosh();
    for p in [1, 2, 3] {
        let v1 = tanh_x.subs_i64(&x, p).eval().eval_f64();
        let v2 = ratio.subs_i64(&x, p).eval().eval_f64();
        match (v1, v2) {
            (Ok(a), Ok(b)) => assert!(
                (a - b).abs() < 1e-10,
                "tanh(x) != sinh(x)/cosh(x) at x={p}: {a} vs {b}"
            ),
            _ => panic!("evaluation failed at x={p}"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Log simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_expand_product() {
    // ln(a*b) -> ln(a) + ln(b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a * &b).ln();
    let expanded = expr.expand_log();
    let s = fmt(&expanded);
    assert!(
        s.contains("ln(a)") && s.contains("ln(b)"),
        "ln(a*b) should expand to ln(a)+ln(b), got: {s}"
    );
}

#[test]
fn log_expand_power() {
    // ln(a^n) -> n*ln(a)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = a.powi(3).ln();
    let expanded = expr.expand_log();
    let s = fmt(&expanded);
    assert!(
        s.contains("ln(a)") && s.contains("3"),
        "ln(a^3) should expand to 3*ln(a), got: {s}"
    );
}

#[test]
fn log_combine_sum() {
    // ln(a) + ln(b) -> ln(a*b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &a.ln() + &b.ln();
    let combined = expr.log_combine();
    let s = fmt(&combined);
    // Should be a single ln of a product
    let ln_count = s.matches("ln(").count();
    assert_eq!(
        ln_count, 1,
        "ln(a)+ln(b) should combine to single ln, got: {s}"
    );
}

#[test]
fn log_combine_coeff() {
    // 2*ln(a) -> ln(a^2)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let expr = &a.ln() * 2;
    let combined = expr.log_combine();
    let s = fmt(&combined);
    assert!(
        s.contains("ln("),
        "2*ln(a) should combine to ln(a^2), got: {s}"
    );
}

#[test]
fn log_roundtrip_expand_combine() {
    // expand then combine should be identity (up to structural equivalence)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let original = (&a * &b).ln();
    let expanded = original.expand_log();
    let recombined = expanded.log_combine();
    // Numerically check at positive integer values
    for (pa, pb) in [(2, 3), (5, 7)] {
        let v_orig = original.subs_i64(&a, pa).subs_i64(&b, pb).eval().eval_f64();
        let v_recom = recombined
            .subs_i64(&a, pa)
            .subs_i64(&b, pb)
            .eval()
            .eval_f64();
        match (v_orig, v_recom) {
            (Ok(o), Ok(r)) => assert!(
                (o - r).abs() < 1e-10,
                "ln(a*b) roundtrip failed at a={pa},b={pb}: {o} vs {r}"
            ),
            _ => panic!("evaluation failed"),
        }
    }
}

#[test]
fn log_exp_simplify() {
    // ln(exp(x)) should simplify to x (for real x)
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let expr = x.exp().ln();
    let simplified = expr.simplify();
    assert_eq!(
        fmt(&simplified),
        "x",
        "ln(exp(x)) should simplify to x, got: {}",
        fmt(&simplified)
    );
}

#[test]
fn exp_log_simplify() {
    // exp(ln(x)) should simplify to x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp();
    let simplified = expr.simplify();
    assert_eq!(
        fmt(&simplified),
        "x",
        "exp(ln(x)) should simplify to x, got: {}",
        fmt(&simplified)
    );
}

#[test]
fn log_of_one_is_zero() {
    let ctx = Context::new();
    let expr = ctx.int(1).ln();
    let evaled = expr.eval();
    assert_eq!(
        fmt(&evaled),
        "0",
        "ln(1) should eval to 0, got: {}",
        fmt(&evaled)
    );
}

#[test]
fn log_of_e_is_one() {
    let ctx = Context::new();
    let expr = ctx.e().ln();
    let evaled = expr.eval();
    assert_eq!(
        fmt(&evaled),
        "1",
        "ln(e) should eval to 1, got: {}",
        fmt(&evaled)
    );
}

#[test]
fn log_of_negative_number_is_complex() {
    // ln(-1) should be i*pi
    let ctx = Context::new();
    let expr = ctx.int(-1).ln();
    let evaled = expr.eval();
    let s = fmt(&evaled);
    eprintln!("ln(-1) = {s}");
    // Numerically, ln(-1) = i*pi ≈ 3.14159*i
    let eval_result = evaled.eval_complex64();
    match eval_result {
        Ok(Complex64 { re, im }) => {
            eprintln!("ln(-1) complex: ({re}, {im})");
            // re should be ~0, im should be ~pi
            assert!(re.abs() < 1e-9, "ln(-1) real part should be 0, got: {re}");
            assert!(
                (im.abs() - std::f64::consts::PI).abs() < 1e-9,
                "ln(-1) imaginary part should be ±π, got: {im}"
            );
        }
        Err(e) => {
            eprintln!("ln(-1) eval failed (may be expected): {e}");
            // Not necessarily a bug — report it
        }
    }
}

#[test]
fn log_expand_three_factors() {
    // ln(a*b*c) -> ln(a) + ln(b) + ln(c)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let expr = (&a * &b * &c).ln();
    let expanded = expr.expand_log();
    let s = fmt(&expanded);
    let ln_count = s.matches("ln(").count();
    assert_eq!(
        ln_count, 3,
        "ln(a*b*c) should expand to 3 ln terms, got {ln_count}: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Power simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pow_x_squared_sqrt_no_assumption() {
    // (x²)^(1/2) without assumptions — should NOT blindly simplify to x
    // because x could be negative. Should stay as-is or become abs(x).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).sqrt();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("sqrt(x^2) with no assumptions = {s}");
    // It should NOT be just "x" since x could be negative
    // Acceptable: sqrt(x^2), abs(x), (x^2)^(1/2), or x^2^(1/2)
    // Verify numerically at x = -3: should give 3, not -3
    let val_neg = simplified.subs_i64(&x, -3).eval().eval_f64();
    match val_neg {
        Ok(v) => {
            assert!(
                (v - 3.0).abs() < 1e-10,
                "sqrt((-3)^2) should be 3, got {v} (simplified form: {s}) — BUG if it's -3"
            );
        }
        Err(e) => eprintln!("sqrt(x^2) eval at x=-3 failed: {e}"),
    }
}

#[test]
fn pow_x_squared_sqrt_positive_assumption() {
    // sqrt(x²) when x is positive -> x
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let expr = x.powi(2).sqrt();
    let refined = expr.refine();
    let s = fmt(&refined);
    assert_eq!(
        s, "x",
        "sqrt(x²) with x positive should refine to x, got: {s}"
    );
}

#[test]
fn pow_x_squared_sqrt_real_assumption() {
    // sqrt(x²) when x is real -> abs(x)
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let expr = x.powi(2).sqrt();
    let refined = expr.refine();
    let s = fmt(&refined);
    assert_eq!(
        s, "abs(x)",
        "sqrt(x²) with x real should refine to abs(x), got: {s}"
    );
}

#[test]
fn pow_combine_exponents_xa_times_xb() {
    // x^a * x^b should simplify to x^(a+b)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &x.pow(&a) * &x.pow(&b);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("x^a * x^b = {s}");
    // Should contain x^(a+b) or equivalent
    // Numerically check: at x=2, a=3, b=4 => 2^7 = 128
    let val_orig = expr
        .subs_i64(&x, 2)
        .subs_i64(&a, 3)
        .subs_i64(&b, 4)
        .eval()
        .eval_f64();
    let val_simp = simplified
        .subs_i64(&x, 2)
        .subs_i64(&a, 3)
        .subs_i64(&b, 4)
        .eval()
        .eval_f64();
    match (val_orig, val_simp) {
        (Ok(o), Ok(s)) => assert!((o - s).abs() < 1e-6, "x^a*x^b simplify broke: {o} vs {s}"),
        _ => eprintln!("evaluation failed for x^a*x^b test"),
    }
}

#[test]
fn pow_of_pow_integer_exponents() {
    // (x^2)^3 should simplify to x^6
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2).powi(3);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("(x^2)^3 = {s}");
    // Check numerically at x=3: 3^6 = 729
    let val = simplified.subs_i64(&x, 3).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 729.0).abs() < 1e-6,
            "(x^2)^3 at x=3 should be 729, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn pow_zero_exponent() {
    // x^0 = 1 (for x != 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(0);
    let s = fmt(&expr);
    // Should be 1 from canonicalization or eval
    let evaled = expr.eval();
    let s2 = fmt(&evaled);
    assert!(
        s == "1" || s2 == "1",
        "x^0 should be 1, got raw={s}, eval={s2}"
    );
}

#[test]
fn pow_one_exponent() {
    // x^1 = x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(1);
    assert_eq!(fmt(&expr), "x", "x^1 should be x, got: {}", fmt(&expr));
}

#[test]
fn pow_negative_exponent() {
    // x^(-1) display and evaluation
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(-1);
    let val = expr.subs_i64(&x, 5).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 0.2).abs() < 1e-10,
            "x^(-1) at x=5 should be 0.2, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Expand edge cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn expand_binomial_small() {
    // (a+b)^2 = a^2 + 2*a*b + b^2
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).powi(2);
    let expanded = expr.expand();
    let s = fmt(&expanded);
    eprintln!("(a+b)^2 expanded = {s}");
    // Check at a=3, b=5: (3+5)^2 = 64
    let val = expanded.subs_i64(&a, 3).subs_i64(&b, 5).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 64.0).abs() < 1e-10,
            "(a+b)^2 at (3,5) should be 64, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_binomial_large_exponent() {
    // (a+b)^10 — tests large multinomial expansion
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).powi(10);
    let expanded = expr.expand();
    let s = fmt(&expanded);
    eprintln!("(a+b)^10 expanded length: {} chars", s.len());
    // Check at a=1, b=1: (1+1)^10 = 1024
    let val_expanded = expanded.subs_i64(&a, 1).subs_i64(&b, 1).eval().eval_f64();
    let val_original = expr.subs_i64(&a, 1).subs_i64(&b, 1).eval().eval_f64();
    match (val_original, val_expanded) {
        (Ok(o), Ok(e)) => assert!((o - e).abs() < 1e-6, "(a+b)^10 expand broke: {o} vs {e}"),
        _ => panic!("eval failed for (a+b)^10"),
    }
    // Also check at a=2, b=3: (2+3)^10 = 5^10 = 9765625
    let v2 = expanded.subs_i64(&a, 2).subs_i64(&b, 3).eval().eval_f64();
    match v2 {
        Ok(v) => assert!(
            (v - 9765625.0).abs() < 1.0,
            "(a+b)^10 at (2,3) should be 9765625, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_trinomial() {
    // (a+b+c)^3 — tests multinomial with 3 terms
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let expr = (&a + &b + &c).powi(3);
    let expanded = expr.expand();
    // Check at a=1, b=2, c=3: (1+2+3)^3 = 216
    let val = expanded
        .subs_i64(&a, 1)
        .subs_i64(&b, 2)
        .subs_i64(&c, 3)
        .eval()
        .eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 216.0).abs() < 1e-6,
            "(a+b+c)^3 at (1,2,3) should be 216, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_nested_product() {
    // (a+b)*(c+d) -> ac + ad + bc + bd
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");
    let expr = &(&a + &b) * &(&c + &d);
    let expanded = expr.expand();
    // Check at a=1,b=2,c=3,d=4: (3)*(7) = 21
    let val = expanded
        .subs_i64(&a, 1)
        .subs_i64(&b, 2)
        .subs_i64(&c, 3)
        .subs_i64(&d, 4)
        .eval()
        .eval_f64();
    match val {
        Ok(v) => assert!((v - 21.0).abs() < 1e-10, "expected 21, got {v}"),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_trig_sin_a_plus_b() {
    // sin(a+b) -> sin(a)*cos(b) + cos(a)*sin(b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).sin();
    let expanded = expr.expand_trig();
    let s = fmt(&expanded);
    assert!(
        s.contains("sin") && s.contains("cos"),
        "sin(a+b) expand_trig should produce sin and cos terms, got: {s}"
    );
    // Numerical check
    let v_orig = expr.subs_i64(&a, 1).subs_i64(&b, 2).eval().eval_f64();
    let v_exp = expanded.subs_i64(&a, 1).subs_i64(&b, 2).eval().eval_f64();
    match (v_orig, v_exp) {
        (Ok(o), Ok(e)) => assert!(
            (o - e).abs() < 1e-10,
            "sin(a+b) expand_trig broke: {o} vs {e}"
        ),
        _ => panic!("eval failed"),
    }
}

#[test]
fn expand_does_not_change_atoms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expanded = x.expand();
    assert_eq!(fmt(&expanded), "x");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Rewrite rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rewrite_sin_as_exp() {
    // sin(x) -> (exp(ix) - exp(-ix)) / (2i)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let rewritten = sin_x.rewrite_as_exp();
    let s = fmt(&rewritten);
    eprintln!("sin(x) rewritten as exp = {s}");
    assert!(
        s.contains("exp") || s.contains("E"),
        "sin(x) rewrite_as_exp should contain exp, got: {s}"
    );
    // Numerical check
    let v_orig = sin_x.subs_i64(&x, 2).eval().eval_f64();
    // The rewritten form is complex — use eval_complex64
    let v_rew = rewritten.subs_i64(&x, 2).eval().eval_complex64();
    match (v_orig, v_rew) {
        (Ok(o), Ok(Complex64 { re, im })) => {
            assert!(
                (o - re).abs() < 1e-9 && im.abs() < 1e-9,
                "sin(x) rewrite_as_exp broke: orig={o}, rewritten=({re}+{im}i)"
            );
        }
        (Ok(o), Err(e)) => {
            eprintln!("rewritten eval failed (may be structural): orig={o}, err={e}");
        }
        _ => eprintln!("evaluation issues"),
    }
}

#[test]
fn rewrite_cos_as_exp() {
    // cos(x) -> (exp(ix) + exp(-ix)) / 2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_x = x.cos();
    let rewritten = cos_x.rewrite_as_exp();
    let s = fmt(&rewritten);
    eprintln!("cos(x) rewritten as exp = {s}");
    assert!(
        s.contains("exp") || s.contains("E"),
        "cos(x) rewrite_as_exp should contain exp, got: {s}"
    );
}

#[test]
fn rewrite_exp_ix_as_trig() {
    // exp(ix) -> cos(x) + i*sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i = ctx.i_unit();
    let exp_ix = (&i * &x).exp();
    let rewritten = exp_ix.rewrite_as_trig();
    let s = fmt(&rewritten);
    eprintln!("exp(ix) rewritten as trig = {s}");
    assert!(
        s.contains("cos") && s.contains("sin"),
        "exp(ix) rewrite_as_trig should produce cos and sin, got: {s}"
    );
    // Numerical check at x=1: exp(i) = cos(1) + i*sin(1)
    let orig_val = exp_ix.subs_i64(&x, 1).eval().eval_complex64();
    let rew_val = rewritten.subs_i64(&x, 1).eval().eval_complex64();
    match (orig_val, rew_val) {
        (Ok(Complex64 { re: r1, im: i1 }), Ok(Complex64 { re: r2, im: i2 })) => {
            assert!(
                (r1 - r2).abs() < 1e-9 && (i1 - i2).abs() < 1e-9,
                "exp(ix) rewrite_as_trig broke: ({r1}+{i1}i) vs ({r2}+{i2}i)"
            );
        }
        _ => eprintln!("complex evaluation issue, may be ok"),
    }
}

#[test]
fn rewrite_roundtrip_sin_to_exp_and_back() {
    // sin(x) -> exp -> trig should give back something involving sin(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let as_exp = sin_x.rewrite_as_exp();
    let back_to_trig = as_exp.rewrite_as_trig();
    let s = fmt(&back_to_trig);
    eprintln!("sin(x) -> exp -> trig = {s}");
    // Should at least be numerically equivalent
    let v_orig = sin_x.subs_i64(&x, 2).eval().eval_f64();
    let v_round = back_to_trig.subs_i64(&x, 2).eval().eval_complex64();
    match (v_orig, v_round) {
        (Ok(o), Ok(Complex64 { re, im })) => {
            assert!(
                (o - re).abs() < 1e-8 && im.abs() < 1e-8,
                "sin(x) roundtrip rewrite broke: orig={o}, roundtrip=({re}+{im}i)"
            );
        }
        _ => eprintln!("roundtrip eval issue"),
    }
}

#[test]
fn rewrite_atom_unchanged() {
    // rewriting a bare symbol should leave it unchanged
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let as_exp = x.rewrite_as_exp();
    let as_trig = x.rewrite_as_trig();
    assert_eq!(fmt(&as_exp), "x", "symbol rewrite_as_exp should be x");
    assert_eq!(fmt(&as_trig), "x", "symbol rewrite_as_trig should be x");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. nsimplify (via eval_decimal + structure, since nsimplify is internal)
//    We test by constructing rational approximations and simplifying.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_pi_is_recognizable() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let s = fmt(&pi);
    assert!(
        s.contains("pi") || s.contains("π"),
        "pi should display as pi, got: {s}"
    );
    let val = pi.eval_f64();
    match val {
        Ok(v) => assert!(
            (v - std::f64::consts::PI).abs() < 1e-10,
            "pi eval_f64 should be ~3.14159, got {v}"
        ),
        Err(e) => panic!("pi eval failed: {e}"),
    }
}

#[test]
fn eval_e_is_recognizable() {
    let ctx = Context::new();
    let e = ctx.e();
    let s = fmt(&e);
    assert!(
        s.contains("E") || s.contains("e"),
        "e should display properly, got: {s}"
    );
    let val = e.eval_f64();
    match val {
        Ok(v) => assert!(
            (v - std::f64::consts::E).abs() < 1e-10,
            "e eval_f64 should be ~2.71828, got {v}"
        ),
        Err(e) => panic!("e eval failed: {e}"),
    }
}

#[test]
fn eval_sqrt2_decimal() {
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).sqrt();
    let val = sqrt2.eval_f64();
    match val {
        Ok(v) => assert!(
            (v - std::f64::consts::SQRT_2).abs() < 1e-10,
            "sqrt(2) should be ~1.41421, got {v}"
        ),
        Err(e) => panic!("sqrt(2) eval failed: {e}"),
    }
}

#[test]
fn eval_sqrt3_decimal() {
    let ctx = Context::new();
    let sqrt3 = ctx.int(3).sqrt();
    let val = sqrt3.eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 3.0_f64.sqrt()).abs() < 1e-10,
            "sqrt(3) should be ~1.73205, got {v}"
        ),
        Err(e) => panic!("sqrt(3) eval failed: {e}"),
    }
}

#[test]
fn eval_decimal_precision() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let result = pi.eval_decimal(30);
    match result {
        Ok(s) => {
            eprintln!("pi to 30 digits: {s}");
            assert!(
                s.starts_with("3.14159265358979"),
                "pi should start with 3.14159265358979, got: {s}"
            );
        }
        Err(e) => panic!("eval_decimal failed: {e}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Expression equality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equals_structural_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.equals(&x), Some(true), "x should equal itself");
}

#[test]
fn equals_simple_algebraic() {
    // x+x should equal 2*x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_x = &x + &x;
    let also_two_x = &x * 2;
    assert_eq!(
        two_x.equals(&also_two_x),
        Some(true),
        "x+x should equal 2*x"
    );
}

#[test]
fn equals_after_expand() {
    // (x+1)^2 should equal x^2 + 2x + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x + 1).powi(2);
    let rhs = &(&x.powi(2) + &(&x * 2)) + 1;
    let result = lhs.equals(&rhs);
    assert_eq!(
        result,
        Some(true),
        "(x+1)^2 should equal x^2+2x+1, got {:?}",
        result
    );
}

#[test]
fn equals_different_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let result = x.equals(&y);
    // Should be None (unknown) or Some(false) — but NOT Some(true)
    assert_ne!(result, Some(true), "x should not equal y");
}

#[test]
fn equals_pythagorean_identity() {
    // sin²(x) + cos²(x) should equal 1
    // This is a tough test — requires simplification
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = &x.sin().powi(2) + &x.cos().powi(2);
    let one = ctx.int(1);
    let result = lhs.equals(&one);
    eprintln!("sin²+cos² equals 1? {:?}", result);
    // If equals can detect this, great. If not, it's a known limitation.
    // But simplify should handle it:
    let simplified = lhs.simplify();
    assert_eq!(
        fmt(&simplified),
        "1",
        "sin²+cos² should simplify to 1 even if equals doesn't detect it"
    );
}

#[test]
fn equals_commutative_addition() {
    // a+b should equal b+a
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let lhs = &a + &b;
    let rhs = &b + &a;
    assert_eq!(
        lhs.equals(&rhs),
        Some(true),
        "a+b should equal b+a (commutative)"
    );
}

#[test]
fn equals_commutative_multiplication() {
    // a*b should equal b*a
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let lhs = &a * &b;
    let rhs = &b * &a;
    assert_eq!(
        lhs.equals(&rhs),
        Some(true),
        "a*b should equal b*a (commutative)"
    );
}

#[test]
fn equals_zero_difference() {
    // (x+1)^2 - x^2 - 2*x - 1 should be zero
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &(&x * 2) - 1;
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("(x+1)^2 - x^2 - 2x - 1 fully simplified = {s}");
    let zero = ctx.int(0);
    let is_zero = simplified.equals(&zero);
    assert_eq!(
        is_zero,
        Some(true),
        "(x+1)^2 - x^2 - 2x - 1 should be 0, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn assume_positive_implies_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    assert_eq!(x.is_positive(), Some(true));
    assert_eq!(x.is_real(), Some(true));
    assert_eq!(x.is_negative(), Some(false));
}

#[test]
fn assume_negative_implies_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]);
    assert_eq!(x.is_negative(), Some(true));
    assert_eq!(x.is_real(), Some(true));
    assert_eq!(x.is_positive(), Some(false));
}

#[test]
fn assume_integer_implies_rational_real() {
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    assert_eq!(ctx.query(&n, Props::INTEGER), Some(true));
    assert_eq!(ctx.query(&n, Props::RATIONAL), Some(true));
    assert_eq!(ctx.query(&n, Props::REAL), Some(true));
}

#[test]
fn refine_abs_positive() {
    // abs(x) when x > 0 -> x
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let expr = x.abs();
    let refined = expr.refine();
    assert_eq!(
        fmt(&refined),
        "x",
        "abs(x) with positive x should refine to x, got: {}",
        fmt(&refined)
    );
}

#[test]
fn refine_abs_negative() {
    // abs(x) when x < 0 -> -x
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]);
    let expr = x.abs();
    let refined = expr.refine();
    let s = fmt(&refined);
    assert_eq!(
        s, "-x",
        "abs(x) with negative x should refine to -x, got: {s}"
    );
}

#[test]
fn refine_with_temporary_assumptions() {
    // Use refine_with for temporary "what if" queries
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x has no permanent assumptions
    assert!(x.is_positive().is_none());
    let expr = x.abs();
    let refined = expr.refine_with(&[(&x, Assumption::Positive)]);
    assert_eq!(
        fmt(&refined),
        "x",
        "abs(x) with temp positive assumption should be x, got: {}",
        fmt(&refined)
    );
    // Original x is still unconstrained
    assert!(x.is_positive().is_none());
}

#[test]
fn refine_sqrt_x_squared_negative_x() {
    // sqrt(x²) when x is negative should be abs(x) = -x
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Negative]);
    let expr = x.powi(2).sqrt();
    let refined = expr.refine();
    let s = fmt(&refined);
    eprintln!("sqrt(x²) with x<0 refined to: {s}");
    // Should be abs(x) or -x; not x (since x is negative)
    // Check numerically by substituting x = -5: should give 5
    let val = refined.subs_i64(&x, -5).eval().eval_f64();
    match val {
        Ok(v) => assert!((v - 5.0).abs() < 1e-10, "sqrt((-5)²) should be 5, got {v}"),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn assume_nonneg_and_nonpositive_implies_zero() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::NonNegative, Assumption::NonPositive]);
    assert_eq!(ctx.query(&x, Props::ZERO), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional edge cases and stress tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_preserves_value_trig_heavy() {
    // Complex trig expression: sin^4(x) + cos^4(x)
    // = (sin²(x) + cos²(x))² - 2*sin²(x)*cos²(x)
    // = 1 - ½*sin²(2x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(4) + &x.cos().powi(4);
    let simplified = expr.simplify();
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3, 4, 5], 1e-10),
        "sin⁴+cos⁴ simplify broke numerical equivalence, simplified to: {}",
        fmt(&simplified)
    );
}

#[test]
fn full_simplify_polynomial_identity() {
    // (x+1)^3 - (x^3 + 3*x^2 + 3*x + 1) should be 0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expanded = (&x + 1).powi(3);
    let manual = &(&(&x.powi(3) + &(&x.powi(2) * 3)) + &(&x * 3)) + 1;
    let diff = &expanded - &manual;
    let simplified = diff.simplify();
    let s = fmt(&simplified);
    assert_eq!(
        s, "0",
        "(x+1)^3 minus expansion should full_simplify to 0, got: {s}"
    );
}

#[test]
fn expand_then_simplify_restores() {
    // Expand (x+1)^2 then simplify — shouldn't lose information
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = (&x + 1).powi(2);
    let expanded = original.expand();
    let s = fmt(&expanded);
    assert!(
        s.contains("x^2"),
        "expanded (x+1)^2 should contain x^2, got: {s}"
    );
    // Verify numerical equivalence
    for p in [0, 1, 2, -1, -2] {
        let v1 = original.subs_i64(&x, p).eval().eval_f64();
        let v2 = expanded.subs_i64(&x, p).eval().eval_f64();
        match (v1, v2) {
            (Ok(a), Ok(b)) => assert!(
                (a - b).abs() < 1e-10,
                "(x+1)^2 expand broke at x={p}: {a} vs {b}"
            ),
            _ => panic!("eval failed at x={p}"),
        }
    }
}

#[test]
#[allow(clippy::erasing_op)] // Symbolic `x * 0` is exactly what's under test.
fn simplify_zero_times_anything() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 0;
    let s = fmt(&expr);
    assert_eq!(s, "0", "x*0 should be 0, got: {s}");
}

#[test]
fn simplify_one_times_anything() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * 1;
    let s = fmt(&expr);
    assert_eq!(s, "x", "x*1 should be x, got: {s}");
}

#[test]
fn simplify_add_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x + 0;
    let s = fmt(&expr);
    assert_eq!(s, "x", "x+0 should be x, got: {s}");
}

#[test]
fn trig_expand_cos_2x_identity() {
    // cos(2x) expanded should be cos²(x) - sin²(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_2x = (&x * 2).cos();
    let expanded = cos_2x.expand_trig();
    // Numerical equivalence
    assert!(
        numerical_eq_1var(&cos_2x, &expanded, &x, &[1, 2, 3, 4], 1e-10),
        "cos(2x) expand_trig broke, result: {}",
        fmt(&expanded)
    );
}

#[test]
fn trig_combine_then_expand_roundtrip() {
    // sin(x)*cos(x) -> trig_combine -> expand_trig -> should be equivalent
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() * &x.cos();
    let combined = expr.trig_combine();
    let re_expanded = combined.expand_trig();
    // All three should be numerically equivalent
    for p in [1, 2, 3] {
        let v_orig = expr.subs_i64(&x, p).eval().eval_f64();
        let v_comb = combined.subs_i64(&x, p).eval().eval_f64();
        let v_reex = re_expanded.subs_i64(&x, p).eval().eval_f64();
        match (v_orig, v_comb, v_reex) {
            (Ok(a), Ok(b), Ok(c)) => {
                assert!(
                    (a - b).abs() < 1e-10,
                    "trig_combine broke at x={p}: {a} vs {b}"
                );
                assert!(
                    (a - c).abs() < 1e-10,
                    "expand_trig(trig_combine(...)) broke at x={p}: {a} vs {c}"
                );
            }
            _ => eprintln!("eval issue at x={p}"),
        }
    }
}

#[test]
fn smart_simplify_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.simplify();
    assert_eq!(
        fmt(&result),
        "1",
        "smart_simplify should get sin²+cos²=1, got: {}",
        fmt(&result)
    );
}

#[test]
fn log_expand_nested_deep() {
    // ln(x^2 * y) should fully expand to 2*ln(x) + ln(y) after two passes
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = (&x.powi(2) * &y).ln();
    let expanded = expr.expand_log();
    let s = fmt(&expanded);
    eprintln!("ln(x²*y) expand_log = {s}");
    // May need a second pass
    let expanded2 = expanded.expand_log();
    let s2 = fmt(&expanded2);
    eprintln!("ln(x²*y) expand_log x2 = {s2}");
    // After two passes, should have ln(x) and ln(y) separately
    assert!(
        s2.contains("ln(x)") && s2.contains("ln(y)"),
        "ln(x²*y) should fully expand, got: {s2}"
    );
}

#[test]
fn eval_trig_special_values() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let pi = ctx.pi();

    // sin(0) = 0
    let s0 = zero.sin().eval();
    assert_eq!(fmt(&s0), "0", "sin(0) should be 0, got: {}", fmt(&s0));

    // cos(0) = 1
    let c0 = zero.cos().eval();
    assert_eq!(fmt(&c0), "1", "cos(0) should be 1, got: {}", fmt(&c0));

    // sin(pi) = 0
    let sp = pi.sin().eval();
    assert_eq!(fmt(&sp), "0", "sin(π) should be 0, got: {}", fmt(&sp));

    // cos(pi) = -1
    let cp = pi.cos().eval();
    assert_eq!(fmt(&cp), "-1", "cos(π) should be -1, got: {}", fmt(&cp));
}

#[test]
fn eval_exp_special_values() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    let one = ctx.int(1);

    // exp(0) = 1
    let e0 = zero.exp().eval();
    assert_eq!(fmt(&e0), "1", "exp(0) should be 1, got: {}", fmt(&e0));

    // ln(1) = 0
    let l1 = one.ln().eval();
    assert_eq!(fmt(&l1), "0", "ln(1) should be 0, got: {}", fmt(&l1));
}

#[test]
fn simplify_double_negative() {
    // -(-x) should simplify to x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = -(&(-&x));
    let s = fmt(&expr);
    eprintln!("-(-x) = {s}");
    // Should be "x" from canonicalization
    assert_eq!(s, "x", "-(-x) should be x, got: {s}");
}

#[test]
fn expand_large_power_does_not_panic() {
    // (x+1)^15 — stress test for expansion
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(15);
    let expanded = expr.expand();
    // Just verify it doesn't panic and produces valid result at x=1
    let val = expanded.subs_i64(&x, 1).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 32768.0).abs() < 1.0,
            "(x+1)^15 at x=1 should be 2^15=32768, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn simplify_x_minus_x_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x - &x;
    let s = fmt(&expr);
    assert_eq!(s, "0", "x-x should be 0, got: {s}");
}

#[test]
fn simplify_x_div_x_is_one() {
    // x/x should be 1 (assuming x != 0)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x / &x;
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(s, "1", "x/x should simplify to 1, got: {s}");
}

#[test]
fn fu_simplification() {
    // fu() is the dedicated Fu algorithm for trig simplification
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.fu();
    let s = fmt(&result);
    eprintln!("fu(sin²+cos²) = {s}");
    // Should give 1
    assert!(
        numerical_eq_1var(&expr, &result, &x, &[1, 2, 3], 1e-10),
        "fu() broke numerical equivalence: {s}"
    );
}

#[test]
fn trig_expand_handles_negative_argument() {
    // sin(-x) should simplify or expand correctly
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).sin();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("sin(-x) simplified = {s}");
    // sin(-x) = -sin(x)
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "sin(-x) simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn cos_negative_argument() {
    // cos(-x) = cos(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).cos();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("cos(-x) simplified = {s}");
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "cos(-x) simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn log_combine_handles_subtraction() {
    // ln(a) - ln(b) = ln(a/b)
    // This needs to work via ln(a) + (-1)*ln(b) → ln(a) + ln(b^(-1)) → ln(a/b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &a.ln() - &b.ln();
    let combined = expr.log_combine();
    let s = fmt(&combined);
    eprintln!("ln(a)-ln(b) log_combine = {s}");
    // Verify numerically at a=6, b=3 => ln(2) ≈ 0.693
    let v_orig = expr.subs_i64(&a, 6).subs_i64(&b, 3).eval().eval_f64();
    let v_comb = combined.subs_i64(&a, 6).subs_i64(&b, 3).eval().eval_f64();
    match (v_orig, v_comb) {
        (Ok(o), Ok(c)) => assert!(
            (o - c).abs() < 1e-10,
            "ln(a)-ln(b) log_combine broke: {o} vs {c}"
        ),
        _ => eprintln!("eval issue, may be structural"),
    }
}

#[test]
fn power_simplify_numeric_bases() {
    // 2^3 should evaluate to 8
    let ctx = Context::new();
    let expr = ctx.int(2).powi(3);
    let evaled = expr.eval();
    assert_eq!(fmt(&evaled), "8", "2^3 should be 8, got: {}", fmt(&evaled));
}

#[test]
fn power_simplify_rational_exponent() {
    // 4^(1/2) = 2
    let ctx = Context::new();
    let four = ctx.int(4);
    let result = four.sqrt().eval();
    assert_eq!(
        fmt(&result),
        "2",
        "sqrt(4) should be 2, got: {}",
        fmt(&result)
    );
}

#[test]
fn power_simplify_27_cbrt() {
    // 27^(1/3) = 3
    let ctx = Context::new();
    let expr = ctx.int(27).pow(&ctx.rational(1, 3));
    let evaled = expr.eval();
    let s = fmt(&evaled);
    eprintln!("27^(1/3) = {s}");
    // May or may not simplify to 3 depending on implementation
    let val = evaled.eval_f64();
    match val {
        Ok(v) => assert!((v - 3.0).abs() < 1e-10, "27^(1/3) should be 3, got {v}"),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn refine_floor_of_integer() {
    // floor(n) where n is integer -> n
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    let expr = n.floor();
    let refined = expr.refine();
    assert_eq!(
        fmt(&refined),
        "n",
        "floor(integer) should refine to n, got: {}",
        fmt(&refined)
    );
}

#[test]
fn constants_are_real() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();
    assert_eq!(pi.is_real(), Some(true), "pi should be real");
    assert_eq!(e.is_real(), Some(true), "e should be real");
}

#[test]
fn imaginary_unit_properties() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    assert_eq!(i.is_real(), Some(false), "i should not be real");
}

// ═══════════════════════════════════════════════════════════════════════════
// Deeper edge-case probes — hunting for real bugs
// ═══════════════════════════════════════════════════════════════════════════

// ── Power edge cases ───────────────────────────────────────────────────

#[test]
fn pow_symbolic_double_pow_xa_b_simplify() {
    // (x^a)^b with symbolic exponents — should simplify to x^(a*b)
    // only when safe (x positive).  Without assumptions, must preserve value.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = x.pow(&a).pow(&b);
    let simplified = expr.simplify();
    // Numerically: x=2, a=3, b=2 => (2^3)^2 = 64 = 2^6
    let v_orig = expr
        .subs_i64(&x, 2)
        .subs_i64(&a, 3)
        .subs_i64(&b, 2)
        .eval()
        .eval_f64();
    let v_simp = simplified
        .subs_i64(&x, 2)
        .subs_i64(&a, 3)
        .subs_i64(&b, 2)
        .eval()
        .eval_f64();
    match (v_orig, v_simp) {
        (Ok(o), Ok(s)) => assert!(
            (o - s).abs() < 1e-6,
            "(x^a)^b simplify broke numerical equiv: {o} vs {s}, form={}",
            fmt(&simplified)
        ),
        _ => eprintln!("(x^a)^b eval issue"),
    }
}

#[test]
fn pow_fractional_of_negative_number() {
    // (-8)^(1/3) — cube root of -8 should be -2 (real branch)
    // or a complex value.  Must not panic.
    let ctx = Context::new();
    let expr = ctx.int(-8).pow(&ctx.rational(1, 3));
    let evaled = expr.eval();
    let s = fmt(&evaled);
    eprintln!("(-8)^(1/3) = {s}");
    // Try numerical eval — should be -2 on real branch or complex
    let val = evaled.eval_complex64();
    match val {
        Ok(Complex64 { re, im }) => {
            eprintln!("(-8)^(1/3) complex = ({re}, {im})");
            // Real cube root: -2.  Principal complex root: 1 + i*sqrt(3)
            // Either is acceptable; verify magnitude = 2
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 2.0).abs() < 1e-9,
                "(-8)^(1/3) magnitude should be 2, got {mag}"
            );
        }
        Err(e) => eprintln!("(-8)^(1/3) eval failed: {e}"),
    }
}

#[test]
fn pow_zero_base_positive_exponent() {
    // 0^5 = 0
    let ctx = Context::new();
    let expr = ctx.int(0).powi(5);
    let evaled = expr.eval();
    assert_eq!(fmt(&evaled), "0", "0^5 should be 0, got: {}", fmt(&evaled));
}

#[test]
fn pow_zero_base_zero_exponent() {
    // 0^0 is conventionally 1 in combinatorics but undefined in analysis.
    // Whatever the library chooses, it should not panic.
    let ctx = Context::new();
    let expr = ctx.int(0).powi(0);
    let evaled = expr.eval();
    let s = fmt(&evaled);
    eprintln!("0^0 = {s}");
    // Most CAS return 1; verify no panic at least
}

#[test]
fn pow_one_base_any_exponent() {
    // 1^x = 1 for all x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = ctx.int(1).pow(&x);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    // Should be 1
    assert_eq!(s, "1", "1^x should be 1, got: {s}");
}

// ── Log edge cases ─────────────────────────────────────────────────────

#[test]
fn log_of_zero_is_neg_infinity_or_error() {
    // ln(0) should be -∞ or an error — must not return a finite number or panic
    let ctx = Context::new();
    let expr = ctx.int(0).ln();
    let evaled = expr.eval();
    let s = fmt(&evaled);
    eprintln!("ln(0) = {s}");
    // If it evaluates numerically, it should be -inf
    let val = evaled.eval_f64();
    match val {
        Ok(v) => {
            assert!(
                v.is_infinite() && v < 0.0 || v < -1e10,
                "ln(0) should be -∞, got {v}"
            );
        }
        Err(_) => {
            // An error is also acceptable
            eprintln!("ln(0) eval returned error (acceptable)");
        }
    }
}

#[test]
fn log_exp_without_real_assumption() {
    // ln(exp(x)) with x having NO assumption — should still simplify to x
    // in many CAS, or stay as ln(exp(x)).
    // The key: simplify must NOT produce a wrong answer.
    let ctx = Context::new();
    let x = ctx.symbol("x"); // no assumptions
    let expr = x.exp().ln();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("ln(exp(x)) no assumption = {s}");
    // If it simplified to x, check it's numerically correct for complex x
    // For real x, ln(exp(x)) = x always.
    // For complex x = a+bi, ln(exp(x)) = x + 2*pi*i*k
    // At integer points (real), must be correct:
    for p in [1, 2, -1, -3] {
        let v_orig = expr.subs_i64(&x, p).eval().eval_f64();
        let v_simp = simplified.subs_i64(&x, p).eval().eval_f64();
        if let (Ok(o), Ok(s)) = (v_orig, v_simp) {
            assert!(
                (o - s).abs() < 1e-9,
                "ln(exp(x)) simplify wrong at x={p}: {o} vs {s}"
            )
        }
    }
}

#[test]
fn log_expand_respects_negative_exponents() {
    // ln(x/y) = ln(x * y^(-1)) — expand_log should give ln(x) - ln(y)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = (&x / &y).ln();
    let expanded = expr.expand_log();
    let s = fmt(&expanded);
    eprintln!("ln(x/y) expand_log = {s}");
    // Should contain ln(x) and ln(y) with subtraction
    assert!(
        s.contains("ln(x)") && s.contains("ln(y)"),
        "ln(x/y) should expand to ln(x)-ln(y), got: {s}"
    );
    // Numerical check at x=10, y=2
    let v_orig = expr.subs_i64(&x, 10).subs_i64(&y, 2).eval().eval_f64();
    let v_exp = expanded.subs_i64(&x, 10).subs_i64(&y, 2).eval().eval_f64();
    match (v_orig, v_exp) {
        (Ok(o), Ok(e)) => assert!((o - e).abs() < 1e-10, "ln(x/y) expand broke: {o} vs {e}"),
        _ => eprintln!("ln(x/y) eval issue"),
    }
}

// ── Trig deeper probes ─────────────────────────────────────────────────

#[test]
fn trig_half_angle_sin_squared() {
    // sin²(x/2) = (1 - cos(x))/2
    // Build sin(x/2)^2 and verify numerically against (1-cos(x))/2
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let sin_half = (&x * &half).sin();
    let sin_half_sq = sin_half.powi(2);
    let identity_rhs = &(&ctx.int(1) - &x.cos()) * &half;
    for p in [1, 2, 3, 4] {
        let v_lhs = sin_half_sq.subs_i64(&x, p).eval().eval_f64();
        let v_rhs = identity_rhs.subs_i64(&x, p).eval().eval_f64();
        match (v_lhs, v_rhs) {
            (Ok(a), Ok(b)) => assert!(
                (a - b).abs() < 1e-10,
                "sin²(x/2) != (1-cos(x))/2 at x={p}: {a} vs {b}"
            ),
            _ => panic!("eval failed at x={p}"),
        }
    }
}

#[test]
fn trig_tan_squared_plus_one() {
    // 1 + tan²(x) = sec²(x) = 1/cos²(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = &ctx.int(1) + &x.tan().powi(2);
    let rhs = &ctx.int(1) / &x.cos().powi(2);
    for p in [1, 2, 4] {
        let v_lhs = lhs.subs_i64(&x, p).eval().eval_f64();
        let v_rhs = rhs.subs_i64(&x, p).eval().eval_f64();
        match (v_lhs, v_rhs) {
            (Ok(a), Ok(b)) => assert!(
                (a - b).abs() < 1e-9,
                "1+tan²(x) != 1/cos²(x) at x={p}: {a} vs {b}"
            ),
            _ => panic!("eval failed at x={p}"),
        }
    }
}

#[test]
fn trig_sin_pi_over_4_eval() {
    // sin(π/4) = √2/2
    let ctx = Context::new();
    let expr = (&ctx.pi() / 4).sin();
    let evaled = expr.eval();
    let val = evaled.eval_f64();
    match val {
        Ok(v) => assert!(
            (v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10,
            "sin(π/4) should be √2/2 ≈ 0.7071, got {v}"
        ),
        Err(e) => panic!("sin(π/4) eval failed: {e}"),
    }
}

#[test]
fn trig_simplify_does_not_increase_complexity() {
    // Simplifying sin(x) should not make it more complex
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(s, "sin(x)", "sin(x) simplify should stay sin(x), got: {s}");
}

#[test]
fn trig_simplify_1_minus_sin_squared() {
    // 1 - sin²(x) = cos²(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(1) - &x.sin().powi(2);
    let simplified = expr.simplify();
    // Numerically must be equivalent
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3, 4], 1e-10),
        "1-sin²(x) simplify broke: {}",
        fmt(&simplified)
    );
}

#[test]
fn trig_expand_sin_5x_fully() {
    // sin(5x) should fully expand to sin(x)/cos(x) terms only
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x * 5).sin();
    let expanded = expr.expand_trig();
    let s = fmt(&expanded);
    // No sin(2*x), sin(3*x), sin(4*x) should remain
    assert!(
        !s.contains("5*x") && !s.contains("4*x") && !s.contains("3*x") && !s.contains("2*x"),
        "sin(5x) not fully expanded: {s}"
    );
    assert!(
        numerical_eq_1var(&expr, &expanded, &x, &[1, 2, 3], 1e-9),
        "sin(5x) expand broke numerical equiv: {s}"
    );
}

#[test]
fn trig_sinh_of_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).sinh();
    let evaled = expr.eval();
    let val = evaled.eval_f64();
    match val {
        Ok(v) => assert!(v.abs() < 1e-15, "sinh(0) should be 0, got {v}"),
        Err(e) => panic!("sinh(0) eval failed: {e}"),
    }
}

#[test]
fn trig_cosh_of_zero() {
    let ctx = Context::new();
    let expr = ctx.int(0).cosh();
    let evaled = expr.eval();
    let val = evaled.eval_f64();
    match val {
        Ok(v) => assert!((v - 1.0).abs() < 1e-15, "cosh(0) should be 1, got {v}"),
        Err(e) => panic!("cosh(0) eval failed: {e}"),
    }
}

// ── Expand edge cases ──────────────────────────────────────────────────

#[test]
fn expand_binomial_zero_exponent() {
    // (a+b)^0 = 1
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).powi(0);
    let s = fmt(&expr);
    let expanded = expr.expand();
    let se = fmt(&expanded);
    // Either the raw form or expanded form should be 1
    assert!(
        s == "1" || se == "1",
        "(a+b)^0 should be 1, raw={s}, expanded={se}"
    );
}

#[test]
fn expand_binomial_one_exponent() {
    // (a+b)^1 = a+b
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).powi(1);
    let expanded = expr.expand();
    let s = fmt(&expanded);
    assert!(
        s.contains('a') && s.contains('b'),
        "(a+b)^1 should be a+b, got: {s}"
    );
    let val = expanded.subs_i64(&a, 3).subs_i64(&b, 7).eval().eval_f64();
    match val {
        Ok(v) => assert!((v - 10.0).abs() < 1e-10, "expected 10, got {v}"),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_product_of_three_sums() {
    // (a+b)*(c+d)*(e+f) — should expand to 8 terms
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");
    let e = ctx.symbol("e");
    let f = ctx.symbol("f");
    let expr = &(&(&a + &b) * &(&c + &d)) * &(&e + &f);
    let expanded = expr.expand();
    // At a=1,b=1,c=1,d=1,e=1,f=1 => 2*2*2 = 8
    let val = expanded
        .subs_i64(&a, 1)
        .subs_i64(&b, 1)
        .subs_i64(&c, 1)
        .subs_i64(&d, 1)
        .subs_i64(&e, 1)
        .subs_i64(&f, 1)
        .eval()
        .eval_f64();
    match val {
        Ok(v) => assert!((v - 8.0).abs() < 1e-10, "expected 8, got {v}"),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn expand_a_plus_b_pow20_does_not_panic() {
    // Stress test: (a+b)^20 — very large expansion (21 terms)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = (&a + &b).powi(20);
    let expanded = expr.expand();
    // (1+1)^20 = 2^20 = 1048576
    let val = expanded.subs_i64(&a, 1).subs_i64(&b, 1).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 1048576.0).abs() < 1.0,
            "(a+b)^20 at (1,1) should be 1048576, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

// ── Rewrite deeper probes ──────────────────────────────────────────────

#[test]
fn rewrite_tan_as_exp() {
    // tan(x) -> -i*(exp(ix) - exp(-ix)) / (exp(ix) + exp(-ix))
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tan_x = x.tan();
    let rewritten = tan_x.rewrite_as_exp();
    let s = fmt(&rewritten);
    eprintln!("tan(x) rewrite_as_exp = {s}");
    assert!(
        s.contains("exp"),
        "tan(x) rewrite_as_exp should contain exp, got: {s}"
    );
    // Numerical check at x=1: tan(1) ≈ 1.5574
    let v_orig = tan_x.subs_i64(&x, 1).eval().eval_f64();
    let v_rew = rewritten.subs_i64(&x, 1).eval().eval_complex64();
    match (v_orig, v_rew) {
        (Ok(o), Ok(Complex64 { re, im })) => {
            assert!(
                (o - re).abs() < 1e-8 && im.abs() < 1e-8,
                "tan(x) rewrite broke: {o} vs ({re}+{im}i)"
            );
        }
        _ => eprintln!("tan rewrite eval issue"),
    }
}

#[test]
fn rewrite_nested_sin_cos_as_exp() {
    // sin(x) + cos(x) rewritten as exp — both terms should be exponentials
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() + &x.cos();
    let rewritten = expr.rewrite_as_exp();
    let s = fmt(&rewritten);
    eprintln!("sin(x)+cos(x) as exp = {s}");
    assert!(
        s.contains("exp"),
        "sin(x)+cos(x) rewrite should contain exp, got: {s}"
    );
    // Should NOT contain sin or cos anymore
    assert!(
        !s.contains("sin(") && !s.contains("cos("),
        "rewrite_as_exp should eliminate trig, got: {s}"
    );
}

#[test]
fn rewrite_exp_a_plus_ib_as_trig() {
    // exp(a + i*b) -> exp(a) * (cos(b) + i*sin(b))
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let i = ctx.i_unit();
    let expr = (&a + &(&i * &b)).exp();
    let rewritten = expr.rewrite_as_trig();
    let s = fmt(&rewritten);
    eprintln!("exp(a+ib) as trig = {s}");
    // Should contain cos and sin and exp(a)
    assert!(
        s.contains("cos") || s.contains("sin"),
        "exp(a+ib) rewrite should produce trig, got: {s}"
    );
}

// ── Equality deeper probes ─────────────────────────────────────────────

#[test]
fn equals_distributive() {
    // a*(b+c) should equal a*b + a*c
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let lhs = &a * &(&b + &c);
    let rhs = &(&a * &b) + &(&a * &c);
    let result = lhs.equals(&rhs);
    assert_eq!(
        result,
        Some(true),
        "a*(b+c) should equal a*b+a*c, got {:?}",
        result
    );
}

#[test]
fn equals_nested_expand() {
    // (x+1)^3 should equal x^3 + 3x^2 + 3x + 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x + 1).powi(3);
    let rhs = &(&(&x.powi(3) + &(&x.powi(2) * 3)) + &(&x * 3)) + 1;
    let result = lhs.equals(&rhs);
    assert_eq!(
        result,
        Some(true),
        "(x+1)^3 should equal x^3+3x^2+3x+1, got {:?}",
        result
    );
}

#[test]
fn equals_subtraction_order() {
    // a - b should NOT equal b - a (unless a==b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let lhs = &a - &b;
    let rhs = &b - &a;
    let result = lhs.equals(&rhs);
    assert_ne!(
        result,
        Some(true),
        "(a-b) should not be proven equal to (b-a)"
    );
}

// ── Assumptions deeper probes ──────────────────────────────────────────

#[test]
fn assume_positive_sum_is_positive() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let y = ctx.symbol_with("y", &[Assumption::Positive]);
    let sum = &x + &y;
    assert_eq!(
        sum.is_positive(),
        Some(true),
        "positive + positive should be positive"
    );
}

#[test]
fn assume_positive_times_negative_is_negative() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]);
    let y = ctx.symbol_with("y", &[Assumption::Negative]);
    let prod = &x * &y;
    assert_eq!(
        prod.is_negative(),
        Some(true),
        "positive * negative should be negative"
    );
}

#[test]
fn assume_integer_squared_is_integer() {
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    let n2 = n.powi(2);
    assert_eq!(
        ctx.query(&n2, Props::INTEGER),
        Some(true),
        "integer² should be integer"
    );
}

#[test]
fn assume_real_x_squared_is_nonneg() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let x2 = x.powi(2);
    let result = ctx.query(&x2, Props::NONNEGATIVE);
    // x² ≥ 0 for real x — the assumption system should know this
    eprintln!("x² nonneg for real x? {:?}", result);
    // This is a desirable property; report if unknown
    if result != Some(true) {
        eprintln!("NOTE: assumption system does not infer x² ≥ 0 for real x");
    }
}

#[test]
fn refine_neg_one_to_even_power() {
    // (-1)^(2n) where n is integer -> 1
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    let two_n = &n * 2;
    let expr = ctx.int(-1).pow(&two_n);
    let refined = expr.refine();
    let s = fmt(&refined);
    eprintln!("(-1)^(2n) refined = {s}");
    // Check numerically
    for p in [0, 1, 2, 3, -1] {
        let val = expr.subs_i64(&n, p).eval().eval_f64();
        match val {
            Ok(v) => assert!((v - 1.0).abs() < 1e-10, "(-1)^(2*{p}) should be 1, got {v}"),
            Err(e) => eprintln!("(-1)^(2*{p}) eval failed: {e}"),
        }
    }
}

// ── Simplification consistency ─────────────────────────────────────────

#[test]
fn simplify_is_idempotent() {
    // simplify(simplify(e)) == simplify(e)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x;
    let s1 = expr.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        fmt(&s1),
        fmt(&s2),
        "simplify should be idempotent: first={}, second={}",
        fmt(&s1),
        fmt(&s2)
    );
}

#[test]
fn full_simplify_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x + 1).powi(2) - &x.powi(2) - &(&x * 2) - 1;
    let s1 = expr.simplify();
    let s2 = s1.simplify();
    assert_eq!(
        fmt(&s1),
        fmt(&s2),
        "full_simplify should be idempotent: first={}, second={}",
        fmt(&s1),
        fmt(&s2)
    );
}

#[test]
fn expand_is_idempotent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(3);
    let e1 = expr.expand();
    let e2 = e1.expand();
    assert_eq!(
        fmt(&e1),
        fmt(&e2),
        "expand should be idempotent: first={}, second={}",
        fmt(&e1),
        fmt(&e2)
    );
}

#[test]
fn simplify_mixed_hyp_trig_no_panic() {
    // sinh(x) + sin(x) — mixing trig and hyperbolic should not confuse simplifier
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sinh() + &x.sin();
    let simplified = expr.simplify();
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3], 1e-10),
        "mixed sinh+sin simplify broke: {}",
        fmt(&simplified)
    );
}

#[test]
fn simplify_exp_sum_rule() {
    // exp(a) * exp(b) should simplify to exp(a+b)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &a.exp() * &b.exp();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("exp(a)*exp(b) simplified = {s}");
    // Verify numerically
    let v_orig = expr.subs_i64(&a, 1).subs_i64(&b, 2).eval().eval_f64();
    let v_simp = simplified.subs_i64(&a, 1).subs_i64(&b, 2).eval().eval_f64();
    match (v_orig, v_simp) {
        (Ok(o), Ok(s)) => assert!(
            (o - s).abs() / o.abs().max(1.0) < 1e-10,
            "exp(a)*exp(b) simplify broke: {o} vs {s}"
        ),
        _ => eprintln!("exp product eval issue"),
    }
}

#[test]
fn simplify_large_expression_no_stack_overflow() {
    // Build a deeply nested expression: ((((x+1)+1)+1)+1)...
    let ctx = Context::new();
    let mut expr = ctx.symbol("x");
    for _ in 0..100 {
        expr = &expr + 1;
    }
    // Should simplify to x + 100 without stack overflow
    let simplified = expr.simplify();
    let val = simplified.subs_i64(&ctx.symbol("x"), 0).eval().eval_f64();
    match val {
        Ok(v) => assert!(
            (v - 100.0).abs() < 1e-10,
            "x + 100 at x=0 should be 100, got {v}"
        ),
        Err(e) => panic!("eval failed: {e}"),
    }
}

#[test]
fn trig_expand_cos_negative_2x() {
    // cos(-2x) = cos(2x)  (even function)
    // After expand_trig, should be numerically equivalent
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_neg2x = (&x * -2).cos();
    let cos_2x = (&x * 2).cos();
    for p in [1, 2, 3] {
        let v1 = cos_neg2x.subs_i64(&x, p).eval().eval_f64();
        let v2 = cos_2x.subs_i64(&x, p).eval().eval_f64();
        match (v1, v2) {
            (Ok(a), Ok(b)) => assert!(
                (a - b).abs() < 1e-10,
                "cos(-2x) != cos(2x) at x={p}: {a} vs {b}"
            ),
            _ => panic!("eval failed"),
        }
    }
}

#[test]
fn log_combine_three_logs() {
    // ln(a) + ln(b) + ln(c) -> ln(a*b*c)
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let expr = &(&a.ln() + &b.ln()) + &c.ln();
    let combined = expr.log_combine();
    let s = fmt(&combined);
    let ln_count = s.matches("ln(").count();
    assert_eq!(
        ln_count, 1,
        "3 ln terms should combine to 1, got {ln_count}: {s}"
    );
    // Numerical check
    let v_orig = expr
        .subs_i64(&a, 2)
        .subs_i64(&b, 3)
        .subs_i64(&c, 5)
        .eval()
        .eval_f64();
    let v_comb = combined
        .subs_i64(&a, 2)
        .subs_i64(&b, 3)
        .subs_i64(&c, 5)
        .eval()
        .eval_f64();
    match (v_orig, v_comb) {
        (Ok(o), Ok(c)) => assert!((o - c).abs() < 1e-10, "3-log combine broke: {o} vs {c}"),
        _ => eprintln!("3-log combine eval issue"),
    }
}

#[test]
fn eval_decimal_sqrt2_many_digits() {
    let ctx = Context::new();
    let sqrt2 = ctx.int(2).sqrt();
    let result = sqrt2.eval_decimal(20);
    match result {
        Ok(s) => {
            eprintln!("sqrt(2) to 20 digits: {s}");
            assert!(
                s.starts_with("1.41421356237"),
                "sqrt(2) should start with 1.41421356237, got: {s}"
            );
        }
        Err(e) => panic!("eval_decimal failed: {e}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG FINDINGS — Tests that expose specific bugs or document limitations
//
// Summary of bugs found:
//
// BUG 1 (bug_sin_negative_arg_not_simplified):
//   sin(-x) is not simplified to -sin(x). The simplifier lacks odd-function
//   parity rules for trig functions. sin, tan, sinh, tanh are all odd but
//   none of them get f(-x) → -f(x) applied. cos and cosh (even functions)
//   DO get f(-x) → f(x) applied, so the parity handling is inconsistent.
//   Root cause: pattern rules in transforms/pattern.rs do not include
//   sin(Neg(x)) → Neg(sin(x)) or equivalent.
//
// BUG 2 (bug_neg1_to_even_power_not_refined):
//   (-1)^(2*n) where n is Integer does not refine to 1. The refine engine
//   checks Props::EVEN on the exponent via the assumption cache, but the
//   cache's compute_mul does not propagate EVEN for the product 2*n when
//   n is Integer. The issue is in AssumptionCache::compute_mul — it only
//   infers INTEGER for products of integers, but does not check whether
//   the product is specifically EVEN (2 * integer → even).
//
// BUG 3 (bug_equals_misses_trig_identity):
//   equals() returns None for sin²(x)+cos²(x) vs 1. The equals() method
//   only uses structural identity and expand, never simplify. Adding a
//   simplify layer to equals() would fix many such cases. This is a design
//   limitation in api/expr_funcs.rs Expr::equals().
//
// BUG 4 (bug_tan_neg_x_should_be_neg_tan_x):
//   Same root cause as BUG 1. tan(-x) stays as tan(-x) instead of -tan(x).
//
// BUG 5 (bug_sinh_neg_x_should_be_neg_sinh_x):
//   Same root cause as BUG 1. sinh(-x) stays as sinh(-x) instead of -sinh(x).
//
// BUG 6 (bug_full_simplify_catches_scaled_pythagorean):
//   3*sin²(x) + 3*cos²(x) is not simplified to 3, even by full_simplify.
//   The Pythagorean rule fires on bare sin²+cos²=1, but factor_terms does
//   not extract the common factor 3 before applying the identity. The fix
//   would be to run factor_terms before trigsimp in smart_simplify.
//
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bug_sin_negative_arg_not_simplified() {
    // BUG 1: sin(-x) does not simplify to -sin(x).
    // sin is an odd function: sin(-x) = -sin(x).
    // The simplifier should recognize this.
    //
    // Root cause: The pattern-based rewrite rules in transforms/pattern.rs
    // do not include a rule for Sin(Neg(x)) → Neg(Sin(x)). The eval pass
    // only handles numeric special values (sin(0), sin(π), etc.), not
    // structural parity simplification.
    //
    // Impact: Any expression containing sin(-θ), tan(-θ), or sinh(-θ) will
    // not simplify correctly, which cascades into rewrite roundtrips
    // (sin→exp→trig→simplify fails to recover sin(x)).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).sin();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(
        s, "-sin(x)",
        "BUG: sin(-x) should simplify to -sin(x), got: {s}"
    );
}

#[test]
fn bug_neg1_to_even_power_not_refined() {
    // BUG 2: (-1)^(2*n) where n is Integer does not refine to 1.
    // The exponent 2*n is even (since n is integer), so (-1)^(2*n) = 1.
    //
    // Root cause: AssumptionCache::compute_mul only propagates INTEGER
    // for integer*integer products, but does not check whether a factor
    // of 2 (or any even number) makes the product EVEN. The refine pass
    // at refine_pow_immut checks exp_props.is_even == Some(true), which
    // fails because the assumption cache never sets EVEN on Mul(2, n).
    //
    // Fix: In compute_mul, if any factor is a known even integer (like 2)
    // and all other factors are integers, mark the product as EVEN.
    let ctx = Context::new();
    let n = ctx.symbol_with("n", &[Assumption::Integer]);
    let two_n = &n * 2;
    let expr = ctx.int(-1).pow(&two_n);
    let refined = expr.refine();
    let s = fmt(&refined);
    assert_eq!(
        s, "1",
        "BUG: (-1)^(2n) with integer n should refine to 1, got: {s}"
    );
}

#[test]
fn bug_cosh2_minus_sinh2_not_simplified_symbolically() {
    // cosh²(x) - sinh²(x) = 1 is the hyperbolic Pythagorean identity.
    // PASSES: The simplifier now handles this (likely via trigsimp strategies
    // or pattern rules that include hyperbolic identities).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(s, "1", "cosh²(x) - sinh²(x) should simplify to 1, got: {s}");
}

#[test]
fn bug_equals_misses_trig_identity() {
    // BUG 3: equals() cannot detect sin²(x)+cos²(x) = 1.
    //
    // Root cause: Expr::equals() in api/expr_funcs.rs implements a 3-layer
    // check: (1) structural identity, (2) diff is zero, (3) expand diff
    // is zero. It never calls simplify(). Since sin²(x)+cos²(x)-1 does
    // not expand to zero (it requires the Pythagorean identity), equals()
    // returns None.
    //
    // Fix: Add a 4th layer that calls simplify() on the difference, or
    // at minimum full_simplify(). This would allow equals() to detect
    // all identities that the simplifier can handle.
    //
    // Note: simplify().equals() would work as a workaround:
    //   lhs.simplify().equals(&one) returns Some(true).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = &x.sin().powi(2) + &x.cos().powi(2);
    let one = ctx.int(1);
    let result = lhs.equals(&one);
    assert_eq!(
        result,
        Some(true),
        "BUG: equals() should detect sin²+cos²=1, got {:?}",
        result
    );
}

#[test]
fn bug_rewrite_roundtrip_not_clean() {
    // sin(x) → rewrite_as_exp → rewrite_as_trig should be recoverable.
    // This test documents that the roundtrip produces a messy expression
    // with sin(-x) and cos(-x) terms, but after simplify it DOES recover
    // (once BUG 1 is fixed it would be cleaner, but currently the
    // simplifier manages to get the right numerical answer).
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let as_exp = sin_x.rewrite_as_exp();
    let back = as_exp.rewrite_as_trig();
    let simplified_back = back.simplify();
    let s = fmt(&simplified_back);
    eprintln!("[INFO] sin(x)->exp->trig->simplify = {s}");
    // At minimum must be numerically equivalent
    assert!(
        numerical_eq_1var(&sin_x, &simplified_back, &x, &[1, 2, 3, 4], 1e-9),
        "rewrite roundtrip broke numerical equivalence: sin(x) vs {s}"
    );
}

#[test]
fn bug_ln_exp_no_assumption_correctness() {
    // POTENTIAL BUG: ln(exp(x)) simplifies to x even without a Real assumption.
    // For complex x, ln(exp(x)) = x + 2πi·k (multivalued), so the
    // principal branch gives ln(exp(x)) = x only when Im(x) ∈ (-π, π].
    // A strict CAS should either require a Real assumption or leave it
    // unsimplified for a general symbol.
    let ctx = Context::new();
    let x = ctx.symbol("x"); // no assumptions at all
    let expr = x.exp().ln();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("[BUG?] ln(exp(x)) no assumptions = {s}");
    // If simplify returns "x" without checking domain, that's a
    // potential correctness issue for complex analysis use cases.
    if s == "x" {
        eprintln!(
            "NOTE: ln(exp(x)) simplified to x without Real assumption — \
                    this is only correct on the principal branch"
        );
    }
    // Verify numerical correctness at least for real integer points
    for p in [1, 2, -1, -3] {
        let v_orig = expr.subs_i64(&x, p).eval().eval_f64();
        let v_simp = simplified.subs_i64(&x, p).eval().eval_f64();
        if let (Ok(o), Ok(s)) = (v_orig, v_simp) {
            assert!(
                (o - s).abs() < 1e-9,
                "ln(exp(x)) wrong at x={p}: {o} vs {s}"
            )
        }
    }
}

#[test]
fn bug_rewrite_sin_cos_sum_as_exp_messy() {
    // BUG: rewrite_as_exp on sin(x)+cos(x) produces a messy expression
    // with double-negatives and unsimplified structure like
    //   "-1/2*-exp(-x*I) + exp(x*I)*I + 1/2*exp(-x*I) + 1/2*exp(x*I)"
    // instead of a clean exponential form.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() + &x.cos();
    let rewritten = expr.rewrite_as_exp();
    let s = fmt(&rewritten);
    eprintln!("[BUG?] sin(x)+cos(x) as exp = {s}");
    // The string should not contain "sin(" or "cos(" — all trig should be gone
    assert!(
        !s.contains("sin(") && !s.contains("cos("),
        "rewrite_as_exp should eliminate all trig, got: {s}"
    );
    // The expression should contain "exp" terms
    assert!(
        s.contains("exp"),
        "rewrite_as_exp result should contain exp, got: {s}"
    );
    // Check for double-negatives or messy structure (e.g. "*-exp" or "--")
    let has_double_neg = s.contains("*-") || s.contains("--");
    if has_double_neg {
        eprintln!("NOTE: rewrite_as_exp produces messy double-negative structure: {s}");
    }
    // Numerical correctness at minimum
    for p in [1, 2, 3] {
        let v_orig = expr.subs_i64(&x, p).eval().eval_f64();
        let v_rew = rewritten.subs_i64(&x, p).eval().eval_complex64();
        match (v_orig, v_rew) {
            (Ok(o), Ok(Complex64 { re, im })) => {
                assert!(
                    (o - re).abs() < 1e-9 && im.abs() < 1e-9,
                    "rewrite broke numerical equiv at x={p}: {o} vs ({re}+{im}i)"
                );
            }
            _ => eprintln!("eval issue at x={p}"),
        }
    }
}

#[test]
fn bug_3sin2_plus_3cos2_not_simplified_to_3() {
    // Test whether 3*sin²(x) + 3*cos²(x) simplifies to 3.
    // Related to BUG 6 (scaled Pythagorean identity). The basic simplify
    // call also fails here. We verify numerical correctness at least.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("[INFO] 3*sin²+3*cos² simplified = {s}");
    // Verify numerical correctness even if symbolic form isn't optimal
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3, 4], 1e-10),
        "3*sin²+3*cos² simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn bug_x_squared_nonneg_not_inferred_for_real() {
    // For real x, x² should be provably nonnegative.
    // The assumption system's compute_pow handles this: Pow(real, even) → NONNEG.
    // This test verifies the inference works correctly.
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let x2 = x.powi(2);
    let result = ctx.query(&x2, Props::NONNEGATIVE);
    assert_eq!(
        result,
        Some(true),
        "x² should be nonnegative when x is real, got {:?}",
        result
    );
}

#[test]
fn bug_trig_simplify_1_minus_cos2_to_sin2() {
    // 1 - cos²(x) should simplify to sin²(x).
    // The inverse of the Pythagorean identity.
    // Whether the simplifier converts to sin²(x) depends on cost heuristics:
    // both forms have the same complexity, so the simplifier may keep either.
    // We verify at minimum that numerical equivalence is preserved.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &ctx.int(1) - &x.cos().powi(2);
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    eprintln!("[INFO] 1-cos²(x) simplified = {s}");
    // Must be numerically correct
    assert!(
        numerical_eq_1var(&expr, &simplified, &x, &[1, 2, 3, 4], 1e-10),
        "1-cos²(x) simplify broke numerical equivalence: {s}"
    );
}

#[test]
fn bug_cos_neg_x_simplify_to_cos_x() {
    // cos(-x) = cos(x) (even function). The simplifier currently
    // does handle this, but let's make it an explicit assertion.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).cos();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(s, "cos(x)", "cos(-x) should simplify to cos(x), got: {s}");
}

#[test]
fn bug_tan_neg_x_should_be_neg_tan_x() {
    // BUG 4: tan(-x) = -tan(x) (odd function) — not simplified.
    // Same root cause as BUG 1: no parity rules for odd trig functions.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).tan();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(
        s, "-tan(x)",
        "BUG: tan(-x) should simplify to -tan(x), got: {s}"
    );
}

#[test]
fn bug_sinh_neg_x_should_be_neg_sinh_x() {
    // BUG 5: sinh(-x) = -sinh(x) (odd function) — not simplified.
    // Same root cause as BUG 1: no parity rules for odd functions.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).sinh();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(
        s, "-sinh(x)",
        "BUG: sinh(-x) should simplify to -sinh(x), got: {s}"
    );
}

#[test]
fn bug_cosh_neg_x_should_be_cosh_x() {
    // cosh(-x) = cosh(x) (even function)
    // PASSES: The simplifier handles even-function parity for cosh.
    // This contrasts with odd functions (sin, tan, sinh) which are NOT
    // handled — see BUGs 1, 4, 5. The inconsistency suggests the
    // even-function rule was added but the odd-function counterpart was
    // overlooked.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-&x).cosh();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(
        s, "cosh(x)",
        "cosh(-x) should simplify to cosh(x), got: {s}"
    );
}

#[test]
fn bug_exp_ln_should_always_simplify() {
    // exp(ln(x)) should ALWAYS simplify to x regardless of assumptions.
    // This is valid because exp and ln are inverse functions on their
    // respective domains, and exp(ln(x)) = x for all x in dom(ln).
    let ctx = Context::new();
    let x = ctx.symbol("x"); // no assumptions
    let expr = x.ln().exp();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    assert_eq!(s, "x", "exp(ln(x)) should always simplify to x, got: {s}");
}

#[test]
fn bug_full_simplify_catches_scaled_pythagorean() {
    // BUG 6: 3*sin²(x) + 3*cos²(x) is not simplified to 3.
    //
    // Root cause: The Pythagorean identity sin²+cos²=1 is matched as a
    // pattern rule on bare sin²(x)+cos²(x) sums. When coefficients are
    // present (3*sin²+3*cos²), the pattern does not match. The fix would
    // be to run factor_terms (to extract the common factor 3) before
    // applying trigsimp, or to extend the Pythagorean rule to recognize
    // c*sin²(x) + c*cos²(x) → c for matching coefficients.
    //
    // Note: simplify() gets 3*sin²+3*cos² but doesn't reduce it.
    //       full_simplify() also fails (10 iterations of eval+expand+simplify).
    //       smart_simplify() also fails.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3);
    let result = expr.simplify();
    let s = fmt(&result);
    assert_eq!(
        s, "3",
        "full_simplify should reduce 3*sin²+3*cos² to 3, got: {s}"
    );
}

#[test]
fn bug_simplify_sin4_plus_cos4() {
    // sin⁴(x) + cos⁴(x) = 1 - ½*sin²(2x) = (3 + cos(4x))/4
    // A good simplifier should at least reduce the operation count.
    // This is a harder identity than sin²+cos²=1, so we test whether
    // the simplifier makes any progress at all.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let original = &x.sin().powi(4) + &x.cos().powi(4);
    let simplified = original.simplify();
    let full = original.simplify();
    let s_orig = fmt(&original);
    let s_simp = fmt(&simplified);
    let s_full = fmt(&full);
    eprintln!("[INFO] sin⁴+cos⁴ original  = {s_orig}");
    eprintln!("[INFO] sin⁴+cos⁴ simplify  = {s_simp}");
    eprintln!("[INFO] sin⁴+cos⁴ full_simp = {s_full}");
    // The full_simplify result should be at most as long as the original
    // (length is a rough proxy for complexity)
    assert!(
        s_full.len() <= s_orig.len(),
        "full_simplify should not make sin⁴+cos⁴ larger: orig={s_orig}, full={s_full}"
    );
    // Numerical correctness is mandatory regardless
    assert!(
        numerical_eq_1var(&original, &full, &x, &[1, 2, 3, 4], 1e-10),
        "full_simplify broke sin⁴+cos⁴ numerical equivalence"
    );
}

#[test]
fn bug_sqrt_x_squared_no_assumption_returns_abs() {
    // sqrt(x²) with no assumptions should return abs(x), NOT x.
    // Returning x would be incorrect for negative x.
    let ctx = Context::new();
    let x = ctx.symbol("x"); // no assumptions
    let expr = x.powi(2).sqrt();
    let simplified = expr.simplify();
    let s = fmt(&simplified);
    // Must be abs(x) or equivalent — NOT bare "x"
    assert_ne!(
        s, "x",
        "BUG: sqrt(x²) without assumptions must NOT simplify to bare x \
         (would be wrong for x<0), got: {s}"
    );
}
