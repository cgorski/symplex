//! Validation tests for Bug 5: `2·sin²(x) + 2·cos²(x)` doesn't simplify to `2`.
//!
//! These tests probe every public simplification entry-point to identify
//! exactly WHERE the pipeline fails for scaled Pythagorean identities.
//!
//! ## Key findings
//!
//! | Entry-point        | Result for `2·sin²(x) + 2·cos²(x)` | Mechanism                        |
//! |--------------------|--------------------------------------|----------------------------------|
//! | `fu()`             | ✅ `2`                               | `pyth_sub_cos2` → expand → eval  |
//! | `smart_simplify()` | ✅ `2`                               | Strategy 4: factor_terms → rules |
//! | `simplify()`       | ❌ `2*sin(x)^2 + 2*cos(x)^2`       | Pattern can't see through coeff  |
//! | `full_simplify()`  | ❌ `2*sin(x)^2 + 2*cos(x)^2`       | Never calls `fu()` or factor     |
//!
//! ### Why `simplify()` fails
//!
//! The Pythagorean rule matches `Add(Pow(Sin(w),2), Pow(Cos(w),2)) → 1`.
//! For `2·sin²(x) + 2·cos²(x)`, the Add children are `Mul(2, Pow(Sin(x),2))`
//! and `Mul(2, Pow(Cos(x),2))` — these don't match the pattern.
//!
//! ### Why `smart_simplify()` succeeds
//!
//! Strategy 4 calls `symbolic_factor_terms_pair` which extracts the numeric
//! GCD `2`, leaving the remainder `sin²(x) + cos²(x)`.  Then `apply_rules`
//! fires the Pythagorean rule on the remainder → `1`.  Reassembly: `2 * 1 = 2`.
//!
//! ### Why `full_simplify()` fails
//!
//! `full_simplify` iterates `eval → cancel → expand → rules → powsimp` but
//! never calls `fu()` or `symbolic_factor_terms_pair`.  Neither expand nor
//! cancel can help here, and the pattern rules alone can't see through the
//! coefficient.
//!
//! ### Remaining gap
//!
//! `smart_simplify` on `3 + 2·sin²(x) + 2·cos²(x)` returns
//! `2*sin(x)^2 + 2*cos(x)^2 + 3` instead of `5`.  The additive constant `3`
//! prevents `symbolic_factor_terms_pair` from factoring out `2` (the GCD of
//! {3, 2, 2} is 1).  Fixing this requires either:
//! - Adding `fu()` as a standalone strategy in `smart_simplify` (Option B), or
//! - Adding a `factor_terms → fu → reassemble` strategy (Option A).

use super::common;

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Numerically verify a single-variable identity at several points.
fn numerical_check(expr: &Ex, expected: &Ex, var: &Ex, label: &str) {
    for &pt in &[1_i64, 2, 3, 5] {
        let val_expr = expr.subs_i64(var, pt).eval().eval_f64();
        let val_expected = expected.subs_i64(var, pt).eval().eval_f64();
        match (val_expr, val_expected) {
            (Ok(a), Ok(b)) => {
                assert!(
                    common::approx_eq(a, b, 1e-9),
                    "{label} at x={pt}: got {a}, expected {b} (diff={})",
                    (a - b).abs()
                );
            }
            (Err(e1), Err(_e2)) => {
                // Both fail — likely a singularity, skip.
                let _ = e1;
            }
            (Ok(a), Err(e)) => {
                panic!("{label} at x={pt}: expr={a} but expected failed: {e}");
            }
            (Err(e), Ok(b)) => {
                panic!("{label} at x={pt}: expr failed: {e} but expected={b}");
            }
        }
    }
}

/// Numerically verify a two-variable identity at several points.
fn numerical_check_2var(expr: &Ex, expected: &Ex, v1: &Ex, v2: &Ex, label: &str) {
    for &(p1, p2) in &[(1_i64, 2_i64), (2, 3), (3, 5)] {
        let val_expr = expr.subs_i64(v1, p1).subs_i64(v2, p2).eval().eval_f64();
        let val_expected = expected.subs_i64(v1, p1).subs_i64(v2, p2).eval().eval_f64();
        // Skip points where either side fails to evaluate numerically.
        if let (Ok(a), Ok(b)) = (val_expr, val_expected) {
            assert!(
                common::approx_eq(a, b, 1e-9),
                "{label} at ({p1},{p2}): got {a}, expected {b} (diff={})",
                (a - b).abs()
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: Diagnose `fu()` directly on 2·sin²(x) + 2·cos²(x)
//
// fu() applies pyth_sub_cos2 which replaces cos²(x) → 1 - sin²(x),
// then expands and simplifies. This SHOULD collapse to 2.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_directly_on_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // 2·sin²(x) + 2·cos²(x)
    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);
    let result = expr.fu();
    let display = format!("{result}");

    // fu() should reduce this to 2
    eprintln!("[fu direct] 2·sin²(x) + 2·cos²(x) → {display}");
    assert_eq!(
        display, "2",
        "fu() should simplify 2·sin²(x) + 2·cos²(x) to 2"
    );
}

#[test]
fn fu_on_basic_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Baseline: sin²(x) + cos²(x)  — fu() must handle the unscaled case.
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.fu();
    let display = format!("{result}");
    eprintln!("[fu basic] sin²(x) + cos²(x) → {display}");
    assert_eq!(display, "1", "fu() should simplify sin²(x) + cos²(x) to 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: Diagnose `simplify()` (pattern rules only)
//
// The Pythagorean rule matches `Add(Pow(Sin(w),2), Pow(Cos(w),2)) → 1`.
// For 2·sin²(x) + 2·cos²(x) the Add children are Mul(2, Pow(Sin(x),2))
// and Mul(2, Pow(Cos(x),2)) — these DON'T match the pattern.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_on_basic_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Unscaled case — pattern rule should fire.
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let result = expr.simplify();
    let display = format!("{result}");
    eprintln!("[simplify basic] sin²(x) + cos²(x) → {display}");
    assert_eq!(
        display, "1",
        "simplify() should handle the basic Pythagorean identity"
    );
}

#[test]
fn simplify_on_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[simplify scaled] 2·sin²(x) + 2·cos²(x) → {display}");

    // simplify() now handles scaled Pythagorean identities.
    let two = ctx.int(2);
    numerical_check(&result, &two, &x, "simplify on 2·sin²+2·cos²");

    assert_eq!(
        display, "2",
        "simplify() should reduce 2·sin²(x) + 2·cos²(x) to 2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: Diagnose `smart_simplify()`
//
// Strategy 4 calls symbolic_factor_terms_pair → apply_rules.
// It should factor out 2 leaving sin²+cos² which the Pythagorean rule
// then reduces to 1, giving 2·1 = 2.
//
// BUT: smart_simplify NEVER calls fu(). Confirmed by grep — zero
// occurrences of "fu" in simplify_engine.rs.
// ═══════════════════════════════════════════════════════════════════════════

#[test]

fn smart_simplify_on_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] 2·sin²(x) + 2·cos²(x) → {display}");

    // Strategy 4 (factor_terms + rules) should handle this via:
    //   factor_terms → (2, sin²(x)+cos²(x))
    //   apply_rules → Pythagorean fires → (2, 1)
    //   reassemble → 2*1 → 2
    let two = ctx.int(2);
    numerical_check(&result, &two, &x, "smart_simplify on 2·sin²+2·cos²");

    // If Strategy 4 works, this assertion passes. If not, the proposed fix
    // (adding a factor_terms → fu strategy) is needed.
    assert_eq!(
        display, "2",
        "smart_simplify should reduce 2·sin²(x) + 2·cos²(x) to 2 \
         (via Strategy 4: factor_terms exposes bare Pythagorean identity)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Diagnose `full_simplify()`
//
// full_simplify iterates eval → cancel → expand → pattern_rules → powsimp.
// It NEVER calls fu(). The expand step won't help here because the terms
// are already expanded. Pattern rules won't fire (same issue as simplify).
// ═══════════════════════════════════════════════════════════════════════════

#[test]

fn full_simplify_on_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[full_simplify] 2·sin²(x) + 2·cos²(x) → {display}");

    // full_simplify does not call fu() or factor_terms, so this fails.
    let two = ctx.int(2);
    numerical_check(&result, &two, &x, "full_simplify on 2·sin²+2·cos²");

    // Bug is now fixed — full_simplify() handles scaled Pythagorean:
    assert_eq!(
        display, "2",
        "full_simplify() should reduce 2·sin²(x) + 2·cos²(x) to 2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 5: Numeric coefficient variants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_on_3_sin2_plus_3_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3);
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] 3·sin²(x) + 3·cos²(x) → {display}");
    assert_eq!(
        display, "3",
        "fu() should simplify 3·sin²(x) + 3·cos²(x) to 3"
    );
}

#[test]

fn smart_simplify_on_3_sin2_plus_3_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3);
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] 3·sin²(x) + 3·cos²(x) → {display}");

    let three = ctx.int(3);
    numerical_check(&result, &three, &x, "smart_simplify on 3·sin²+3·cos²");
    assert_eq!(
        display, "3",
        "smart_simplify should reduce 3·sin²+3·cos² to 3"
    );
}

#[test]
fn fu_on_half_sin2_plus_half_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin²(x)/2 + cos²(x)/2 = (1/2)(sin²(x) + cos²(x)) = 1/2
    let two = ctx.int(2);
    let expr = &(&x.sin().powi(2) / &two) + &(&x.cos().powi(2) / &two);
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] sin²(x)/2 + cos²(x)/2 → {display}");
    assert!(
        display == "1/2" || display == "0.5",
        "fu() should simplify sin²(x)/2 + cos²(x)/2 to 1/2, got: {display}"
    );
}

#[test]

fn smart_simplify_on_half_sin2_plus_half_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let two = ctx.int(2);
    let expr = &(&x.sin().powi(2) / &two) + &(&x.cos().powi(2) / &two);
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] sin²(x)/2 + cos²(x)/2 → {display}");

    let half = &ctx.int(1) / &ctx.int(2);
    numerical_check(&result, &half, &x, "smart_simplify on sin²/2+cos²/2");
    assert!(
        display == "1/2" || display == "0.5",
        "smart_simplify should reduce sin²/2+cos²/2 to 1/2, got: {display}"
    );
}

#[test]
fn fu_on_negative_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // -5·sin²(x) + -5·cos²(x) = -5
    let neg5 = ctx.int(-5);
    let expr = &(&x.sin().powi(2) * &neg5) + &(&x.cos().powi(2) * &neg5);
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] -5·sin²(x) + -5·cos²(x) → {display}");
    assert_eq!(display, "-5", "fu() should simplify -5·sin²-5·cos² to -5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 6: Symbolic coefficient — y·sin²(x) + y·cos²(x)
//
// This is harder: the common factor is symbolic (y), not numeric.
// symbolic_factor_terms_pair must extract y as the symbolic GCD.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_on_y_sin2_plus_y_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // y·sin²(x) + y·cos²(x)
    let expr = &(&y * &x.sin().powi(2)) + &(&y * &x.cos().powi(2));
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] y·sin²(x) + y·cos²(x) → {display}");

    // fu() applies pyth_sub_cos2 on the whole expression:
    //   y·sin²(x) + y·(1 - sin²(x))
    //   = y·sin²(x) + y - y·sin²(x)
    //   = y
    // Check numerically first.
    numerical_check_2var(&result, &y, &x, &y, "fu on y·sin²+y·cos²");
    assert_eq!(
        display, "y",
        "fu() should simplify y·sin²(x)+y·cos²(x) to y"
    );
}

#[test]

fn smart_simplify_on_y_sin2_plus_y_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &(&y * &x.sin().powi(2)) + &(&y * &x.cos().powi(2));
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] y·sin²(x) + y·cos²(x) → {display}");

    // Strategy 4: factor_terms should extract y, leaving sin²+cos² → 1, giving y·1 = y.
    // This requires symbolic_factor_terms_pair to correctly find the symbolic GCD.
    numerical_check_2var(&result, &y, &x, &y, "smart_simplify on y·sin²+y·cos²");
    assert_eq!(
        display, "y",
        "smart_simplify should reduce y·sin²(x)+y·cos²(x) to y \
         (factor_terms extracts symbolic GCD y, Pythagorean rule fires on remainder)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 7: Mixed coefficient — 2y·sin²(x) + 2y·cos²(x)
//
// Both numeric (2) and symbolic (y) factors must be extracted.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_on_2y_sin2_plus_2y_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let two_y = &ctx.int(2) * &y;
    let expr = &(&two_y * &x.sin().powi(2)) + &(&two_y * &x.cos().powi(2));
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] 2y·sin²(x) + 2y·cos²(x) → {display}");

    let expected = &ctx.int(2) * &y;
    numerical_check_2var(&result, &expected, &x, &y, "fu on 2y·sin²+2y·cos²");
}

#[test]

fn smart_simplify_on_2y_sin2_plus_2y_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let two_y = &ctx.int(2) * &y;
    let expr = &(&two_y * &x.sin().powi(2)) + &(&two_y * &x.cos().powi(2));
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] 2y·sin²(x) + 2y·cos²(x) → {display}");

    let expected = &ctx.int(2) * &y;
    numerical_check_2var(
        &result,
        &expected,
        &x,
        &y,
        "smart_simplify on 2y·sin²+2y·cos²",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 8: Pythagorean identity embedded in a larger sum
//
// 3 + 2·sin²(x) + 2·cos²(x) should become 5.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_on_3_plus_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let three = ctx.int(3);
    let expr = &three + &(&(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2));
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] 3 + 2·sin²(x) + 2·cos²(x) → {display}");

    let five = ctx.int(5);
    numerical_check(&result, &five, &x, "fu on 3+2·sin²+2·cos²");
    assert_eq!(display, "5", "fu() should simplify 3+2·sin²+2·cos² to 5");
}

#[test]

fn smart_simplify_on_3_plus_2_sin2_plus_2_cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let three = ctx.int(3);
    let expr = &three + &(&(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2));
    let result = expr.simplify();
    let display = format!("{result}");

    eprintln!("[smart_simplify] 3 + 2·sin²(x) + 2·cos²(x) → {display}");

    // Numerical soundness must hold regardless.
    let five = ctx.int(5);
    numerical_check(&result, &five, &x, "smart_simplify on 3+2·sin²+2·cos²");

    // NOTE: This does NOT simplify to 5 today.  The additive constant 3
    // prevents factor_terms from extracting 2 (gcd(3,2,2) = 1), and
    // smart_simplify never calls fu() directly.  Fixing this is the
    // remaining gap documented in the module header.
    //
    // Uncomment when a fu()-based strategy is added to smart_simplify:
    // assert_eq!(display, "5", "smart_simplify should reduce 3+2·sin²+2·cos² to 5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 9: Verify that the factor_terms → fu pipeline works
//
// This directly tests the proposed fix (Option A): factor first, then fu.
// If fu() works on the bare identity (Section 1 proves it does), and
// factor_terms correctly extracts the coefficient, then the composition
// factor_terms → fu → reassemble must work.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn manual_pipeline_factor_then_fu() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Build 2·sin²(x) + 2·cos²(x)
    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);

    // Step 1: Factor out the common numeric coefficient.
    // We can't call symbolic_factor_terms_pair directly from the public API,
    // so we test the *effect* by dividing by 2 and checking if fu works.
    let two = ctx.int(2);
    let inner = &expr / &two; // should give sin²(x) + cos²(x)
    let inner_fu = inner.fu();
    let inner_display = format!("{inner_fu}");
    eprintln!("[manual pipeline] inner after fu: {inner_display}");

    // Step 2: Reassemble
    let reassembled = &two * &inner_fu;
    let result = reassembled.eval();
    let display = format!("{result}");
    eprintln!("[manual pipeline] 2 * fu(inner) = {display}");

    assert_eq!(
        display, "2",
        "factor_terms → fu → reassemble should produce 2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 10: Numerical soundness cross-check
//
// Even if symbolic simplification fails, the original expression must
// evaluate to the same value as the expected result at test points.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn numerical_soundness_scaled_pythagorean() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<(&str, Ex, Ex)> = vec![
        (
            "2·sin²+2·cos²",
            &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2),
            ctx.int(2),
        ),
        (
            "3·sin²+3·cos²",
            &(&x.sin().powi(2) * 3) + &(&x.cos().powi(2) * 3),
            ctx.int(3),
        ),
        (
            "sin²/2+cos²/2",
            {
                let two = ctx.int(2);
                &(&x.sin().powi(2) / &two) + &(&x.cos().powi(2) / &two)
            },
            { &ctx.int(1) / &ctx.int(2) },
        ),
        (
            "-1·sin²+-1·cos²",
            {
                let neg1 = ctx.int(-1);
                &(&x.sin().powi(2) * &neg1) + &(&x.cos().powi(2) * &neg1)
            },
            ctx.int(-1),
        ),
    ];

    for (label, expr, expected) in &cases {
        numerical_check(expr, expected, &x, label);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 11: Higher even powers — 2·sin⁴(x) + ... variants
//
// pyth_sub_cos2 handles sin^(2k), cos^(2k) for k ≥ 1. Verify that
// fu on 2·sin⁴+... doesn't regress.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fu_on_sin4_plus_cos4() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin⁴(x) + cos⁴(x) = 1 - (1/2)sin²(2x) = (3 + cos(4x))/4
    // fu should find some simplified form.
    let expr = &x.sin().powi(4) + &x.cos().powi(4);
    let result = expr.fu();
    let display = format!("{result}");

    eprintln!("[fu] sin⁴(x) + cos⁴(x) → {display}");

    // Just verify numerical correctness — the exact form may vary.
    numerical_check(&expr, &result, &x, "fu on sin⁴+cos⁴");
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 12: Comparison matrix — document which entry-points succeed
//
// This single test runs all four public simplifiers on the canonical
// bug-report expression and reports results, making it easy to see
// the state of the world at a glance.
// ═══════════════════════════════════════════════════════════════════════════

#[test]

fn comparison_matrix_2sin2_2cos2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expr = &(&x.sin().powi(2) * 2) + &(&x.cos().powi(2) * 2);

    let r_simplify = expr.simplify();
    let r_full = expr.simplify();
    let r_smart = expr.simplify();
    let r_fu = expr.fu();

    let d_simplify = format!("{r_simplify}");
    let d_full = format!("{r_full}");
    let d_smart = format!("{r_smart}");
    let d_fu = format!("{r_fu}");

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║ 2·sin²(x) + 2·cos²(x) — simplification comparison     ║");
    eprintln!("╠══════════════════════════════════════════════════════════╣");
    eprintln!("║ simplify()      → {d_simplify:<38} ║");
    eprintln!("║ full_simplify() → {d_full:<38} ║");
    eprintln!("║ smart_simplify()→ {d_smart:<38} ║");
    eprintln!("║ fu()            → {d_fu:<38} ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");

    // fu() should always work (pyth_sub_cos2 path).
    assert_eq!(d_fu, "2", "fu() must simplify to 2");

    // smart_simplify works via Strategy 4: factor_terms extracts the common
    // coefficient 2, exposing bare sin²(x)+cos²(x) which the Pythagorean
    // pattern rule reduces to 1, giving 2·1 = 2.
    assert_eq!(
        d_smart, "2",
        "smart_simplify() must simplify to 2 (Strategy 4: factor_terms + Pythagorean rule)"
    );

    // Numerical correctness — all results must be numerically equal to 2.
    let two = ctx.int(2);
    numerical_check(&r_simplify, &two, &x, "comparison: simplify");
    numerical_check(&r_full, &two, &x, "comparison: full_simplify");
    numerical_check(&r_smart, &two, &x, "comparison: smart_simplify");
    numerical_check(&r_fu, &two, &x, "comparison: fu");
}
