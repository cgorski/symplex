//! Validation tests for the 1^∞ / ∞^0 limit heuristic in `limit.rs`.
//!
//! Bug 3: The `b^e → exp(e·(b-1))` heuristic fires unconditionally for any
//! `Pow(base, exponent)` where the exponent depends on `var`, but the
//! approximation `ln(b) ≈ b - 1` is only valid when `b → 1`. When the base
//! does NOT tend to 1 (e.g. `x^(1/x)`, `2^(1/x)`), the heuristic produces
//! garbage.
//!
//! These tests exercise every edge case from the mathematical analysis to
//! establish the full scope of the bug and validate any proposed fix.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Group 1: Cases where base → 1 (heuristic SHOULD apply)
// ═══════════════════════════════════════════════════════════════════════════

/// Classic: lim(x→∞) (1 + 1/x)^x = e
#[test]
fn heuristic_valid_1_plus_1_over_x_to_the_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + 1/x)^x
    let base = &x.powi(-1) + 1;
    let expr = base.pow(&x);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "lim (1+1/x)^x should be e ≈ {expected}, got {val}"
    );
}

/// lim(x→∞) (1 - 1/x)^x = 1/e
#[test]
fn heuristic_valid_1_minus_1_over_x_to_the_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 - 1/x)^x
    let base = 1 - &x.powi(-1);
    let expr = base.pow(&x);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = 1.0 / std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "lim (1-1/x)^x should be 1/e ≈ {expected}, got {val}"
    );
}

/// lim(x→∞) (1 + 2/x)^(3x) = exp(6)
/// Generalised form with constants a=2, b=3 → exp(ab)=exp(6).
#[test]
fn heuristic_valid_1_plus_a_over_x_to_bx() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + 2/x)^(3x)
    let base = 1 + &(2 / &x);
    let exponent = &x * 3;
    let expr = base.pow(&exponent);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = (6.0_f64).exp(); // e^6 ≈ 403.43
    assert!(
        (val - expected).abs() / expected < 1e-6,
        "lim (1+2/x)^(3x) should be exp(6) ≈ {expected}, got {val}"
    );
}

/// lim(x→∞) (1 + 1/x²)^x = exp(0) = 1
/// Base → 1, heuristic gives lim x·(1/x²) = lim 1/x = 0, so exp(0) = 1.
#[test]
fn heuristic_valid_1_plus_1_over_x2_to_the_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + 1/x²)^x
    let base = 1 + &x.powi(-2);
    let expr = base.pow(&x);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim (1+1/x²)^x should be 1, got {val}"
    );
}

/// lim(x→∞) (1 + 1/x)^(x²) = ∞
/// Base → 1, but the inner product x²·(1/x) = x → ∞, so exp(∞) = ∞.
/// The heuristic should detect the divergence and fall through to Gruntz.
#[test]
fn heuristic_divergent_1_plus_1_over_x_to_the_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + 1/x)^(x²)
    let base = 1 + &x.powi(-1);
    let exponent = x.powi(2);
    let expr = base.pow(&exponent);
    let result = expr.limit(&x, &ctx.infinity());
    let s = format!("{result}");
    assert!(
        s.contains("∞") || s.contains("oo"),
        "lim (1+1/x)^(x²) should be ∞, got {s}"
    );
}

/// lim(x→∞) (1 + 3/x)^x = e^3
#[test]
fn heuristic_valid_1_plus_3_over_x_to_the_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = 1 + &(3 / &x);
    let expr = base.pow(&x);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = (3.0_f64).exp();
    assert!(
        (val - expected).abs() / expected < 1e-6,
        "lim (1+3/x)^x should be exp(3) ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 2: Cases where base → ∞ or base → constant ≠ 1 (heuristic MUST NOT
//          apply — these are the bug triggers)
// ═══════════════════════════════════════════════════════════════════════════

/// BUG CASE: lim(x→∞) x^(1/x) = 1
/// ∞^0 indeterminate form.  base = x → ∞, NOT 1.
/// The heuristic would compute exp((1/x)·(x - 1)) = exp((x-1)/x) → exp(1) = e.
/// That is WRONG; the correct answer is 1.
#[test]
fn bug_inf_pow_0_x_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^(1/x)
    let expr = x.pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim x^(1/x) should be 1, got {val}"
    );
}

/// BUG CASE: lim(x→∞) 2^(1/x) = 1
/// Constant base = 2 ≠ 1, exponent → 0.
/// The heuristic would compute exp((1/x)·(2 - 1)) = exp(1/x) → exp(0) = 1.
/// In this case, the heuristic gives the right answer accidentally, but only
/// because (b-1) = 1 is a constant. We still want to verify correctness.
#[test]
fn const_base_2_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 2^(1/x)
    let expr = ctx.int(2).pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim 2^(1/x) should be 1, got {val}"
    );
}

/// BUG CASE: lim(x→∞) x^(1/x²) = 1
/// ∞^0 form. base = x → ∞.
/// Correct: exp(ln(x)/x²) → exp(0) = 1 (since ln(x)/x² → 0).
/// Heuristic gives: exp((1/x²)·(x - 1)) = exp((x-1)/x²) → exp(0) = 1.
/// Accidentally correct, but via wrong reasoning.
#[test]
fn inf_pow_0_x_to_the_1_over_x2() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^(1/x²)
    let expr = x.pow(&x.powi(-2));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim x^(1/x²) should be 1, got {val}"
    );
}

/// BUG CASE: lim(x→∞) (x²)^(1/x) = 1
/// ∞^0 form. base = x² → ∞.
/// Correct: exp(2·ln(x)/x) → exp(0) = 1.
/// Heuristic gives: exp((1/x)·(x² - 1)) = exp((x² - 1)/x) → exp(∞) = ∞. WRONG.
#[test]
fn bug_inf_pow_0_x2_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x²)^(1/x)
    let expr = x.powi(2).pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim (x²)^(1/x) should be 1, got {val}"
    );
}

/// BUG CASE: lim(x→∞) (exp(x))^(1/x) = e^1 = e?  No!
/// Actually exp(x)^(1/x) = exp(x/x) = exp(1) = e ... wait, let's be careful.
/// exp(x)^(1/x) = exp(x · (1/x)) = exp(1) = e. That's correct.
/// But the heuristic computes exp((1/x)·(exp(x) - 1)). Since exp(x) → ∞,
/// exp(x)-1 → ∞, so (1/x)·(exp(x)-1) → ∞, and exp(∞) = ∞. WRONG.
#[test]
fn bug_exp_x_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x)^(1/x) = e
    let expr = x.exp().pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "lim exp(x)^(1/x) should be e ≈ {expected}, got {val}"
    );
}

/// BUG CASE: lim(x→∞) (1+x)^(1/x) = 1
/// base = 1+x → ∞, NOT 1.
/// Correct: exp(ln(1+x)/x) → exp(0) = 1 (since ln(1+x)/x → 0).
/// Heuristic gives: exp((1/x)·((1+x) - 1)) = exp(x/x) = exp(1) = e. WRONG.
#[test]
fn bug_1_plus_x_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + x)^(1/x)
    let base = &x + 1;
    let expr = base.pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim (1+x)^(1/x) should be 1, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 3: Cases where base → 0 (heuristic also invalid)
// ═══════════════════════════════════════════════════════════════════════════

/// lim(x→∞) (1/x)^(1/x) = 1
/// 0^0 form. base = 1/x → 0.
/// Correct: exp(ln(1/x)/x) = exp(-ln(x)/x) → exp(0) = 1.
/// Heuristic gives: exp((1/x)·(1/x - 1)) = exp((1/x² - 1/x)) → exp(0) = 1.
/// Accidentally correct since both terms → 0, but reasoning is wrong.
#[test]
fn zero_pow_zero_1_over_x_to_the_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1/x)^(1/x)
    let expr = x.powi(-1).pow(&x.powi(-1));
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim (1/x)^(1/x) should be 1, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 4: Exponent does NOT depend on var (heuristic should not fire at all)
// ═══════════════════════════════════════════════════════════════════════════

/// lim(x→∞) x^2 — exponent is constant, heuristic guard `contains(exp, var)`
/// should prevent it from firing.
#[test]
fn constant_exponent_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let result = x.powi(2).limit(&x, &ctx.infinity());
    let s = format!("{result}");
    assert!(
        s.contains("∞") || s.contains("oo"),
        "lim x^2 at ∞ should be ∞, got {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 5: The exp(e·ln(b)) alternative — would Gruntz handle these
// if we rewrote b^e → exp(e·ln(b)) instead of using the heuristic?
// ═══════════════════════════════════════════════════════════════════════════

/// Verify that Gruntz can handle exp(x·ln(1+1/x)) as x→∞.
/// This is the exact rewrite of (1+1/x)^x = exp(x·ln(1+1/x)).
/// The limit should be e.
///
/// This tests whether the exp(e·ln(b)) alternative would work for the
/// classic 1^∞ case — i.e., whether Gruntz can handle it WITHOUT the heuristic.
#[test]
fn gruntz_handles_exp_x_ln_1_plus_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x · ln(1 + 1/x))
    let inner = (&x * (1 + &x.powi(-1)).ln());
    let expr = inner.exp();
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "lim exp(x·ln(1+1/x)) should be e ≈ {expected}, got {val}"
    );
}

/// Verify Gruntz handles exp(ln(x)/x) as x→∞ (the exact rewrite of x^(1/x)).
/// Should give exp(0) = 1.
#[test]
fn gruntz_handles_exp_ln_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(ln(x)/x)
    let inner = &x.ln() / &x;
    let expr = inner.exp();
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim exp(ln(x)/x) should be 1, got {val}"
    );
}

/// Verify Gruntz handles exp(2·ln(x)/x) as x→∞ (exact rewrite of (x²)^(1/x)).
/// Should give exp(0) = 1.
#[test]
fn gruntz_handles_exp_2_ln_x_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(2·ln(x)/x)
    let inner = &(&x.ln() * 2) / &x;
    let expr = inner.exp();
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    assert!(
        (val - 1.0).abs() < 1e-6,
        "lim exp(2·ln(x)/x) should be 1, got {val}"
    );
}

/// Verify Gruntz handles exp(ln(1+1/x)·x²) as x→∞.
/// This is the exact rewrite of (1+1/x)^(x²).
/// ln(1+1/x) ≈ 1/x - 1/(2x²) + ..., so x²·ln(1+1/x) ≈ x - 1/2 + ... → ∞.
/// Result should be ∞.
#[test]
fn gruntz_handles_exp_x2_ln_1_plus_1_over_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x² · ln(1 + 1/x))
    let inner = &x.powi(2) * (1 + &x.powi(-1)).ln();
    let expr = inner.exp();
    let result = expr.limit(&x, &ctx.infinity());
    let s = format!("{result}");
    assert!(
        s.contains("∞") || s.contains("oo"),
        "lim exp(x²·ln(1+1/x)) should be ∞, got {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 6: Negative-infinity limits (ensure the heuristic guard works there too)
// ═══════════════════════════════════════════════════════════════════════════

/// lim(x→-∞) (1 + 1/x)^x = e
/// Same identity, but approaching from -∞.
#[test]
fn heuristic_valid_1_plus_1_over_x_to_x_neg_inf() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = 1 + &x.powi(-1);
    let expr = base.pow(&x);
    let neg_inf = -ctx.infinity();
    let result = expr.limit(&x, &neg_inf);
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = std::f64::consts::E;
    assert!(
        (val - expected).abs() < 1e-6,
        "lim(x→-∞) (1+1/x)^x should be e ≈ {expected}, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Group 7: Additional sanity checks on power limits
// ═══════════════════════════════════════════════════════════════════════════

/// lim(x→∞) (1 + 5/(3x))^x = exp(5/3)
#[test]
fn heuristic_valid_rational_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (1 + 5/(3x))^x
    let base = 1 + &(&ctx.int(5) / &(&ctx.int(3) * &x));
    let expr = base.pow(&x);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = (5.0_f64 / 3.0).exp();
    assert!(
        (val - expected).abs() / expected < 1e-4,
        "lim (1+5/(3x))^x should be exp(5/3) ≈ {expected}, got {val}"
    );
}

/// lim(x→∞) (1 + 1/x)^(2x) = e²
#[test]
fn heuristic_valid_1_plus_1_over_x_to_2x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let base = 1 + &x.powi(-1);
    let exponent = &x * 2;
    let expr = base.pow(&exponent);
    let result = expr.limit(&x, &ctx.infinity());
    let val = result
        .eval_f64()
        .expect("limit should evaluate to f64");
    let expected = std::f64::consts::E.powi(2);
    assert!(
        (val - expected).abs() / expected < 1e-6,
        "lim (1+1/x)^(2x) should be e² ≈ {expected}, got {val}"
    );
}
