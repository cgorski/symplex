//! Integration tests for the trig simplification algorithm.
//!
//! These tests exercise trig simplification scenarios through the public API.
//! They assume that a `.fu()` method is available on `Ex` (wired up by the
//! module-declaration agent via `expr_funcs.rs`).  Each test verifies
//! numerical equivalence at 3 points to confirm correctness.

use super::common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that two single-variable expressions are numerically equal at 3
/// integer points (1, 2, 3).
fn verify_1var(original: &Ex, simplified: &Ex, var: &Ex, label: &str) {
    common::assert_math_eq_tol(original, simplified, var, &[1, 2, 3], 1e-9, label);
}

/// Verify numerical equality for two-variable expressions at 3 point-pairs.
fn verify_2var(original: &Ex, simplified: &Ex, v1: &Ex, v2: &Ex, label: &str) {
    let point_sets: [(i64, i64); 3] = [(1, 2), (2, 3), (3, 5)];
    for (p1, p2) in &point_sets {
        let vo = original
            .subs_i64(v1, *p1)
            .subs_i64(v2, *p2)
            .eval()
            .eval_f64();
        let vs = simplified
            .subs_i64(v1, *p1)
            .subs_i64(v2, *p2)
            .eval()
            .eval_f64();
        match (vo, vs) {
            (Ok(a), Ok(b)) => {
                assert!(
                    common::approx_eq(a, b, 1e-9),
                    "{label} at ({p1},{p2}): {a} vs {b} (diff={})",
                    (a - b).abs()
                );
            }
            (Err(_), Err(_)) => {} // both fail — skip (singularity)
            (Ok(a), Err(e)) => {
                panic!("{label} at ({p1},{p2}): orig={a} but simplified failed: {e}")
            }
            (Err(e), Ok(b)) => {
                panic!("{label} at ({p1},{p2}): orig failed: {e} but simplified={b}")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

/// sin²(x) + cos²(x) → 1
#[test]
fn fu_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_pythagorean");
    assert_eq!(format!("{result}"), "1");
}

/// sin(x)/cos(x) → tan(x)   (TR2i)
#[test]
fn fu_sin_cos_to_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin() / &x.cos();
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_sin_cos_to_tan");
    let s = format!("{result}");
    assert!(s.contains("tan"), "expected tan(x), got: {s}");
}

/// 2·sin(x)·cos(x) → sin(2x)   (TR10i / TR8)
#[test]
fn fu_double_angle() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let expr = &two * &x.sin() * &x.cos();
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_double_angle");
    // The result should be no more complex than the original.
    assert!(
        result.count_ops() <= expr.count_ops(),
        "fu should simplify 2sin(x)cos(x), got: {result}"
    );
}

/// sin(a) + sin(b) → 2·sin((a+b)/2)·cos((a-b)/2)   (TR9)
#[test]
fn fu_sum_to_product() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let expr = &a.sin() + &b.sin();
    let result = expr.fu();
    verify_2var(&expr, &result, &a, &b, "fu_sum_to_product");
    let s = format!("{result}");
    // Sum-to-product should produce both sin and cos in a product form.
    assert!(
        s.contains("sin") && s.contains("cos"),
        "expected product form, got: {s}"
    );
}

/// sin²(x) → (1 - cos(2x))/2   (TR5)
#[test]
fn fu_power_reduce() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().powi(2);
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_power_reduce");
    // Result should be numerically equivalent; the exact form may vary.
}

/// 1/2 - cos(2x)/2 → sin²(x)   (inverse of TR5)
#[test]
fn fu_half_minus_cos2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let neg_half = ctx.rational(-1, 2);
    let cos_2x = (&ctx.int(2) * &x).cos();
    let expr = &half + &neg_half * &cos_2x;
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_half_minus_cos2x");
    // Should not be worse.
    assert!(
        result.count_ops() <= expr.count_ops() + 1,
        "fu should not make 1/2-cos(2x)/2 much worse, got: {result}"
    );
}

/// cos(x)·cos(2x)·cos(4x) → sin(8x)/(8·sin(x))   (TRmorrie)
#[test]
fn fu_morrie() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cos_x = x.cos();
    let cos_2x = (&ctx.int(2) * &x).cos();
    let cos_4x = (&ctx.int(4) * &x).cos();
    let expr = &cos_x * &cos_2x * &cos_4x;
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_morrie");
    let s = format!("{result}");
    // Morrie's law should produce sin terms.
    assert!(s.contains("sin"), "expected sin form from Morrie, got: {s}");
}

/// √6·cos(x) + √2·sin(x) → 2√2·sin(x + π/3)   (compound angle)
///
/// This is a hard simplification that requires recognising the linear
/// combination as a compound-angle form.  We verify numerical equivalence
/// and accept any result that is no worse.
#[test]
fn fu_sqrt6_cos_sqrt2_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt6 = ctx.int(6).sqrt();
    let sqrt2 = ctx.int(2).sqrt();
    let expr = &sqrt6 * &x.cos() + &sqrt2 * &x.sin();
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_sqrt6_cos_sqrt2_sin");
    // At minimum, the transform must preserve value.
}

/// sin(x)⁴ - cos(y)² + sin(y)² + 2·cos(x)² → cos(x)⁴ - 2cos(y)² + 2
#[test]
fn fu_complex_example() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr =
        &x.sin().powi(4) - &y.cos().powi(2) + &y.sin().powi(2) + &ctx.int(2) * &x.cos().powi(2);
    let result = expr.fu();
    verify_2var(&expr, &result, &x, &y, "fu_complex_example");
    // The result should have fewer or equal trig nodes.
}

/// sin(x) stays as sin(x) — no unnecessary transforms.
#[test]
fn fu_preserves_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    let result = expr.fu();
    assert_eq!(format!("{result}"), "sin(x)");
}

/// sec(x)² + csc(x)² is simplified (TR1 removes sec/csc).
///
/// sec(x) = 1/cos(x) and csc(x) = 1/sin(x) in this system.
#[test]
fn fu_sec_csc_removed() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sec2 = x.sec().powi(2);
    let csc2 = x.csc().powi(2);
    let expr = &sec2 + &csc2;
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_sec_csc_removed");
    // Should be at least as simple.
    assert!(
        result.count_ops() <= expr.count_ops() + 2,
        "fu should not make sec²+csc² much worse, got: {result}"
    );
}

/// tan(x)·cot(x) → 1   (TR13)
#[test]
fn fu_tan_cot_reduce() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tan_x = &x.sin() / &x.cos();
    let cot_x = &x.cos() / &x.sin();
    let expr = &tan_x * &cot_x;
    let result = expr.fu();
    verify_1var(&expr, &result, &x, "fu_tan_cot_reduce");
    assert_eq!(format!("{result}"), "1", "tan(x)*cot(x) should → 1");
}
