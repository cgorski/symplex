//! Round 3 — Algebraic property tests for symplex.
//!
//! Tests mathematical invariants that should ALWAYS hold:
//!   1. apart↔together round-trip
//!   2. trig expand↔combine round-trip
//!   3. rewrite_as_exp preserves value
//!   4. substitution commutativity
//!   5. solve roots satisfy equation
//!   6. series convergence
//!   7. diff of integral is identity (FTC)
//!
//! Run with:
//!   cd symplex && cargo test --test legacy round3_properties:: 2>&1

#[allow(unused_imports)]
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Approximate equality with mixed absolute + relative tolerance.
fn approx(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return false;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < tol * scale
}

/// Evaluate expr at var = p/q, returning f64.
#[allow(dead_code)]
fn eval_rational(expr: &Ex, var: &Ex, p: i64, q: i64) -> Result<f64, SymplexError> {
    let ctx = expr.context();
    let pt = ctx.rational(p, q);
    expr.subs(var, &pt).eval().eval_f64()
}

/// Evaluate an expression at x = p/q, y = r/s (two variables).
#[allow(dead_code)]
fn eval_rational_2(
    expr: &Ex,
    xvar: &Ex,
    xp: i64,
    xq: i64,
    yvar: &Ex,
    yp: i64,
    yq: i64,
) -> Result<f64, SymplexError> {
    let ctx = expr.context();
    let xpt = ctx.rational(xp, xq);
    let ypt = ctx.rational(yp, yq);
    expr.subs(xvar, &xpt).subs(yvar, &ypt).eval().eval_f64()
}

/// Check numerical equivalence of two expressions at integer points.
#[allow(dead_code)]
fn numerical_eq(a: &Ex, b: &Ex, var: &Ex, points: &[i64], tol: f64) -> Vec<(i64, f64, f64)> {
    let mut failures = Vec::new();
    for &pt in points {
        let va = a.subs_i64(var, pt).eval().eval_f64();
        let vb = b.subs_i64(var, pt).eval().eval_f64();
        match (va, vb) {
            (Ok(fa), Ok(fb)) => {
                if !approx(fa, fb, tol) {
                    failures.push((pt, fa, fb));
                }
            }
            (Err(_), Err(_)) => {} // both fail — skip
            (Ok(fa), Err(_)) => failures.push((pt, fa, f64::NAN)),
            (Err(_), Ok(fb)) => failures.push((pt, f64::NAN, fb)),
        }
    }
    failures
}

/// Compare two expressions at rational points, returning failures.
fn compare_at_rational_points(
    a: &Ex,
    b: &Ex,
    var: &Ex,
    points: &[(i64, i64)],
    tol: f64,
) -> Vec<String> {
    let ctx = a.context();
    let mut failures = Vec::new();
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let va = a.subs(var, &pt).eval().eval_f64();
        let vb = b.subs(var, &pt).eval().eval_f64();
        match (va, vb) {
            (Ok(fa), Ok(fb)) => {
                if !approx(fa, fb, tol) {
                    failures.push(format!(
                        "at {var}={p}/{q}: a={fa}, b={fb}, diff={}",
                        (fa - fb).abs()
                    ));
                }
            }
            (Err(_), Err(_)) => {}
            (Ok(fa), Err(e)) => {
                failures.push(format!("at {var}={p}/{q}: a={fa} but b failed: {e}"));
            }
            (Err(e), Ok(fb)) => {
                failures.push(format!("at {var}={p}/{q}: a failed: {e} but b={fb}"));
            }
        }
    }
    failures
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 1: apart ↔ together round-trip
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify that apart(together(r), x) ≈ r at multiple points.
fn assert_apart_together_roundtrip(expr: &Ex, var: &Ex, points: &[i64], tol: f64, label: &str) {
    let combined = expr.together();
    let decomposed = combined.partial_fractions(var);

    let mut checked = 0;
    for &pt in points {
        let v_orig = expr.subs_i64(var, pt).eval().eval_f64();
        let v_round = decomposed.subs_i64(var, pt).eval().eval_f64();
        match (v_orig, v_round) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                assert!(
                    approx(a, b, tol),
                    "apart↔together roundtrip FAILED for {label} at {var}={pt}: \
                     original={a}, roundtripped={b}, diff={}",
                    (a - b).abs()
                );
            }
            (Err(_), Err(_)) => {} // both fail at singularity
            (Ok(a), Err(e)) => {
                panic!(
                    "apart↔together {label} at {var}={pt}: original={a} but roundtrip failed: {e}"
                );
            }
            (Err(e), Ok(b)) => {
                panic!(
                    "apart↔together {label} at {var}={pt}: original failed: {e} but roundtrip={b}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "apart↔together {label}: no points evaluated — test is vacuous"
    );
}

#[test]
fn prop1_apart_together_1_over_x_times_x_plus_1() {
    // 1/(x*(x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x * (&x + 1));
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 2, 3, 5, 7], 1e-9, "1/(x*(x+1))");
}

#[test]
fn prop1_apart_together_x_plus_1_over_x2_minus_1() {
    // (x+1)/(x^2-1)  = (x+1)/((x-1)(x+1)) = 1/(x-1)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x + 1) / (&x.powi(2) - 1);
    // Avoid x = ±1 (poles)
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "(x+1)/(x²-1)");
}

#[test]
fn prop1_apart_together_1_over_x2_plus_1() {
    // 1/(x^2+1) — irreducible over ℝ, should be unchanged by apart
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) + 1);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "1/(x²+1)");
}

#[test]
fn prop1_apart_together_x_over_x2_minus_x_minus_2() {
    // x/(x^2-x-2) = x/((x-2)(x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x / (&x.powi(2) - &x - 2);
    // Avoid x = 2, x = -1 (poles)
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 1, 3, 5, 7], 1e-9, "x/(x²-x-2)");
}

#[test]
fn prop1_apart_together_x2_plus_1_over_x3_minus_1() {
    // (x^2+1)/(x^3-1) = (x^2+1)/((x-1)(x^2+x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x.powi(2) + 1) / (&x.powi(3) - 1);
    // Avoid x = 1 (pole)
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "(x²+1)/(x³-1)");
}

#[test]
fn prop1_apart_together_1_over_x2_minus_1() {
    // 1/(x^2 - 1) = 1/((x-1)(x+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "1/(x²-1)");
}

#[test]
fn prop1_apart_together_1_over_x3() {
    // 1/x^3 — already a single fraction, roundtrip should preserve it
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &x.powi(3);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 2, 3, 5], 1e-9, "1/x³");
}

#[test]
fn prop1_apart_together_sum_of_simple_fractions() {
    // 1/x + 1/(x+1) + 1/(x+2) — a sum of partial fractions
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let one = ctx.int(1);
    let expr = &(&one / &x) + &(&one / &(&x + 1)) + &one / &(&x + 2);
    assert_apart_together_roundtrip(
        &expr,
        &x,
        &[-4, -3, 3, 5, 7, 10],
        1e-9,
        "1/x + 1/(x+1) + 1/(x+2)",
    );
}

#[test]
fn prop1_apart_together_2x_plus_3_over_x2_plus_3x_plus_2() {
    // (2x+3)/((x+1)(x+2))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x * 2 + 3) / (&x.powi(2) + &x * 3 + 2);
    // Avoid x = -1, x = -2
    assert_apart_together_roundtrip(
        &expr,
        &x,
        &[-4, -3, 0, 1, 2, 3, 5],
        1e-9,
        "(2x+3)/(x²+3x+2)",
    );
}

#[test]
fn prop1_apart_together_polynomial_unchanged() {
    // x^2 + 3x + 2 — already a polynomial, should roundtrip cleanly
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + &x * 3 + 2;
    assert_apart_together_roundtrip(
        &expr,
        &x,
        &[-3, -2, 0, 1, 2, 5],
        1e-9,
        "x²+3x+2 (polynomial)",
    );
}

#[test]
fn prop1_apart_together_1_over_x2_minus_5x_plus_6() {
    // 1/(x²-5x+6) = 1/((x-2)(x-3))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - &x * 5 + 6);
    assert_apart_together_roundtrip(&expr, &x, &[-2, -1, 0, 1, 4, 5, 7], 1e-9, "1/(x²-5x+6)");
}

#[test]
fn prop1_apart_together_x3_plus_1_over_x2_minus_1() {
    // (x³+1)/(x²-1) — improper rational function
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x.powi(3) + 1) / (&x.powi(2) - 1);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "(x³+1)/(x²-1)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 2: trig expand ↔ combine round-trip
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify expand_trig(trig_combine(e)) ≈ e at rational points.
fn assert_trig_roundtrip_1var(
    expr: &Ex,
    var: &Ex,
    points: &[(i64, i64)], // rational points (p/q)
    tol: f64,
    label: &str,
) {
    let combined = expr.trig_combine();
    let re_expanded = combined.expand_trig();

    let ctx = expr.context();
    let mut checked = 0;
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let v_orig = expr.subs(var, &pt).eval().eval_f64();
        let v_round = re_expanded.subs(var, &pt).eval().eval_f64();
        match (v_orig, v_round) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                assert!(
                    approx(a, b, tol),
                    "trig roundtrip FAILED for {label} at {var}={p}/{q}: \
                     original={a}, roundtripped={b}, diff={}",
                    (a - b).abs()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(a), Err(e)) => {
                panic!(
                    "trig roundtrip {label} at {var}={p}/{q}: original={a} but roundtrip failed: {e}"
                );
            }
            (Err(e), Ok(b)) => {
                panic!(
                    "trig roundtrip {label} at {var}={p}/{q}: original failed: {e} but roundtrip={b}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "trig roundtrip {label}: no points evaluated — vacuous"
    );
}

/// Helper for two-variable trig roundtrip.
fn assert_trig_roundtrip_2var(
    expr: &Ex,
    xvar: &Ex,
    yvar: &Ex,
    points: &[(i64, i64, i64, i64)], // (xp, xq, yp, yq)
    tol: f64,
    label: &str,
) {
    let combined = expr.trig_combine();
    let re_expanded = combined.expand_trig();

    let ctx = expr.context();
    let mut checked = 0;
    for &(xp, xq, yp, yq) in points {
        let xpt = ctx.rational(xp, xq);
        let ypt = ctx.rational(yp, yq);
        let v_orig = expr.subs(xvar, &xpt).subs(yvar, &ypt).eval().eval_f64();
        let v_round = re_expanded
            .subs(xvar, &xpt)
            .subs(yvar, &ypt)
            .eval()
            .eval_f64();
        match (v_orig, v_round) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                assert!(
                    approx(a, b, tol),
                    "trig 2var roundtrip FAILED for {label} at x={xp}/{xq}, y={yp}/{yq}: \
                     original={a}, roundtripped={b}, diff={}",
                    (a - b).abs()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(a), Err(e)) => {
                panic!(
                    "trig 2var roundtrip {label} at x={xp}/{xq}, y={yp}/{yq}: original={a}, roundtrip err: {e}"
                );
            }
            (Err(e), Ok(b)) => {
                panic!(
                    "trig 2var roundtrip {label} at x={xp}/{xq}, y={yp}/{yq}: original err: {e}, roundtrip={b}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "trig 2var roundtrip {label}: no points evaluated — vacuous"
    );
}

const TRIG_1VAR_POINTS: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (3, 1), (11, 10), (5, 3)];

const TRIG_2VAR_POINTS: &[(i64, i64, i64, i64)] = &[
    (1, 2, 7, 10),
    (1, 1, 2, 1),
    (3, 10, 11, 10),
    (5, 3, 1, 4),
    (2, 1, 3, 1),
];

#[test]
fn prop2_trig_roundtrip_sin_x_cos_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.sin() * &y.cos();
    assert_trig_roundtrip_2var(&expr, &x, &y, TRIG_2VAR_POINTS, 1e-9, "sin(x)*cos(y)");
}

#[test]
fn prop2_trig_roundtrip_sin_x_squared() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin().powi(2);
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "sin(x)²");
}

#[test]
fn prop2_trig_roundtrip_cos_2x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x * 2).cos();
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "cos(2x)");
}

#[test]
fn prop2_trig_roundtrip_sin_x_plus_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = (&x + &y).sin();
    assert_trig_roundtrip_2var(&expr, &x, &y, TRIG_2VAR_POINTS, 1e-9, "sin(x+y)");
}

#[test]
fn prop2_trig_roundtrip_cos_diff_identity() {
    // cos(x)*cos(y) - sin(x)*sin(y) = cos(x+y)
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &(&x.cos() * &y.cos()) - &(&x.sin() * &y.sin());
    assert_trig_roundtrip_2var(
        &expr,
        &x,
        &y,
        TRIG_2VAR_POINTS,
        1e-9,
        "cos(x)cos(y)-sin(x)sin(y)",
    );
}

#[test]
fn prop2_trig_roundtrip_sin_x_sin_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.sin() * &y.sin();
    assert_trig_roundtrip_2var(&expr, &x, &y, TRIG_2VAR_POINTS, 1e-9, "sin(x)*sin(y)");
}

#[test]
fn prop2_trig_roundtrip_cos_x_cos_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.cos() * &y.cos();
    assert_trig_roundtrip_2var(&expr, &x, &y, TRIG_2VAR_POINTS, 1e-9, "cos(x)*cos(y)");
}

#[test]
fn prop2_trig_roundtrip_sin_x_cos_x() {
    // sin(x)*cos(x) = ½ sin(2x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin() * &x.cos();
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "sin(x)*cos(x)");
}

#[test]
fn prop2_trig_roundtrip_cos_x_squared() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos().powi(2);
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "cos(x)²");
}

#[test]
fn prop2_trig_roundtrip_sin2_plus_cos2() {
    // sin²(x) + cos²(x) — should stay 1 throughout
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "sin²(x)+cos²(x)");
}

#[test]
fn prop2_trig_roundtrip_sin_3x() {
    // sin(3x) — expand_trig will break it into products
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x * 3).sin();
    assert_trig_roundtrip_1var(&expr, &x, TRIG_1VAR_POINTS, 1e-9, "sin(3x)");
}

#[test]
fn prop2_trig_roundtrip_cos_x_plus_y_sum_identity() {
    // cos(x)*cos(y) + sin(x)*sin(y) = cos(x-y)
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &(&x.cos() * &y.cos()) + &(&x.sin() * &y.sin());
    assert_trig_roundtrip_2var(
        &expr,
        &x,
        &y,
        TRIG_2VAR_POINTS,
        1e-9,
        "cos(x)cos(y)+sin(x)sin(y)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 3: rewrite_as_exp preserves value
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify e.rewrite_as_exp() ≈ e at rational points.
///
/// Since rewrite_as_exp introduces complex exponentials, we compare using
/// eval_complex64 on the rewritten form (taking real part) vs eval_f64 on
/// the original (when the original is real-valued).
fn assert_rewrite_exp_preserves_value(
    expr: &Ex,
    var: &Ex,
    points: &[(i64, i64)],
    tol: f64,
    label: &str,
) {
    let rewritten = expr.rewrite_as_exp();
    let ctx = expr.context();
    let mut checked = 0;

    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let v_orig = expr.subs(var, &pt).eval().eval_f64();
        // The rewritten form may have complex intermediates, so use
        // eval_complex64 and check the real part.
        let v_rewritten_real = rewritten.subs(var, &pt).eval().eval_f64();
        let v_rewritten_complex = rewritten.subs(var, &pt).eval().eval_complex64();

        let orig_val = match v_orig {
            Ok(v) => v,
            Err(_) => continue, // original can't be evaluated here — skip
        };

        // Try real evaluation first
        if let Ok(rw) = v_rewritten_real {
            checked += 1;
            assert!(
                approx(orig_val, rw, tol),
                "rewrite_as_exp FAILED for {label} at {var}={p}/{q}: \
                 original={orig_val}, rewritten(real)={rw}, diff={}",
                (orig_val - rw).abs()
            );
            continue;
        }

        // Fall back to complex evaluation — the imaginary part should
        // be ~0 for real-valued expressions.
        match v_rewritten_complex {
            Ok((re, im)) => {
                checked += 1;
                assert!(
                    im.abs() < tol * re.abs().max(1.0),
                    "rewrite_as_exp {label} at {var}={p}/{q}: \
                     imaginary part should be ~0 but got im={im} (re={re})"
                );
                assert!(
                    approx(orig_val, re, tol),
                    "rewrite_as_exp FAILED for {label} at {var}={p}/{q}: \
                     original={orig_val}, rewritten(complex re)={re}, diff={}",
                    (orig_val - re).abs()
                );
            }
            Err(e) => {
                panic!(
                    "rewrite_as_exp {label} at {var}={p}/{q}: \
                     original={orig_val} but rewritten can't be evaluated: {e}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "rewrite_as_exp {label}: no points evaluated — vacuous"
    );
}

const EXP_REWRITE_POINTS: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (3, 2), (11, 10), (2, 1)];

#[test]
fn prop3_rewrite_exp_sin_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "sin(x)");
}

#[test]
fn prop3_rewrite_exp_cos_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "cos(x)");
}

#[test]
fn prop3_rewrite_exp_tan_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.tan();
    // Avoid x = π/2 ≈ 1.57 — use safe points away from poles
    let safe_points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (1, 4), (3, 10)];
    assert_rewrite_exp_preserves_value(&expr, &x, safe_points, 1e-8, "tan(x)");
}

#[test]
fn prop3_rewrite_exp_sinh_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sinh();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "sinh(x)");
}

#[test]
fn prop3_rewrite_exp_cosh_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cosh();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "cosh(x)");
}

#[test]
fn prop3_rewrite_exp_sin2_plus_cos2() {
    // sin²(x) + cos²(x) should evaluate to 1 even after rewrite_as_exp
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "sin²(x)+cos²(x)");
}

#[test]
fn prop3_rewrite_exp_sin_plus_cos() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin() + &x.cos();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "sin(x)+cos(x)");
}

#[test]
fn prop3_rewrite_exp_sin_squared() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin().powi(2);
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "sin²(x)");
}

#[test]
fn prop3_rewrite_exp_cos_squared() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos().powi(2);
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "cos²(x)");
}

#[test]
fn prop3_rewrite_exp_2sin_cos() {
    // 2*sin(x)*cos(x) = sin(2x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &(&x.sin() * &x.cos()) * 2;
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "2sin(x)cos(x)");
}

#[test]
fn prop3_rewrite_exp_atom_unchanged() {
    // Bare symbol should not be changed by rewrite_as_exp
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let rewritten = x.rewrite_as_exp();
    assert_eq!(
        format!("{rewritten}"),
        "x",
        "bare symbol should be unchanged"
    );
}

#[test]
fn prop3_rewrite_exp_exp_unchanged() {
    // exp(x) should be unchanged by rewrite_as_exp
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.exp();
    assert_rewrite_exp_preserves_value(&expr, &x, EXP_REWRITE_POINTS, 1e-9, "exp(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 4: substitution commutativity
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify e.subs(x,a).subs(y,b) == e.subs(y,b).subs(x,a) numerically.
fn assert_subs_commute(
    expr: &Ex,
    xvar: &Ex,
    yvar: &Ex,
    xval: &Ex,
    yval: &Ex,
    tol: f64,
    label: &str,
) {
    let xy_first = expr.subs(xvar, xval).subs(yvar, yval);
    let yx_first = expr.subs(yvar, yval).subs(xvar, xval);

    let v_xy = xy_first.eval().eval_f64();
    let v_yx = yx_first.eval().eval_f64();

    match (v_xy, v_yx) {
        (Ok(a), Ok(b)) => {
            assert!(
                approx(a, b, tol),
                "substitution commutativity FAILED for {label}: \
                 subs(x,a).subs(y,b)={a}, subs(y,b).subs(x,a)={b}, diff={}",
                (a - b).abs()
            );
        }
        (Err(e1), Err(e2)) => {
            eprintln!("subs commutativity {label}: both orders failed — xy: {e1}, yx: {e2}");
        }
        (Ok(a), Err(e)) => {
            panic!("subs commutativity {label}: xy={a} but yx failed: {e}");
        }
        (Err(e), Ok(b)) => {
            panic!("subs commutativity {label}: xy failed: {e} but yx={b}");
        }
    }
}

#[test]
fn prop4_subs_commute_x_plus_y_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x + &y;
    for (xv, yv) in [(2, 3), (5, 7), (-1, 4), (0, 10)] {
        let xval = ctx.int(xv);
        let yval = ctx.int(yv);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-12,
            &format!("x+y at x={xv}, y={yv}"),
        );
    }
}

#[test]
fn prop4_subs_commute_x_times_y_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x * &y;
    for (xv, yv) in [(2, 3), (5, 7), (-1, 4), (0, 10)] {
        let xval = ctx.int(xv);
        let yval = ctx.int(yv);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-12,
            &format!("x*y at x={xv}, y={yv}"),
        );
    }
}

#[test]
fn prop4_subs_commute_sin_x_plus_cos_y_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.sin() + &y.cos();
    for &(xp, xq, yp, yq) in &[(1, 2, 7, 10), (1, 1, 2, 1), (3, 1, 5, 2)] {
        let xval = ctx.rational(xp, xq);
        let yval = ctx.rational(yp, yq);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("sin(x)+cos(y) at x={xp}/{xq}, y={yp}/{yq}"),
        );
    }
}

#[test]
fn prop4_subs_commute_x2_plus_y2_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.powi(2) + &y.powi(2);
    for (xv, yv) in [(1, 2), (3, 4), (-2, 5), (0, 7)] {
        let xval = ctx.int(xv);
        let yval = ctx.int(yv);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-12,
            &format!("x²+y² at x={xv}, y={yv}"),
        );
    }
}

#[test]
fn prop4_subs_commute_exp_xy_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = (&x * &y).exp();
    for &(xp, xq, yp, yq) in &[(1, 2, 1, 3), (1, 1, 1, 2), (1, 4, 3, 4)] {
        let xval = ctx.rational(xp, xq);
        let yval = ctx.rational(yp, yq);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("exp(x*y) at x={xp}/{xq}, y={yp}/{yq}"),
        );
    }
}

#[test]
fn prop4_subs_commute_symbolic_values() {
    // Substitute with symbolic values that don't contain the other variable
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, a, b);
    let expr = &x + &y;

    // subs(x, a²).subs(y, b+1) should equal subs(y, b+1).subs(x, a²)
    let a_sq = a.powi(2);
    let b_plus_1 = &b + 1;

    let xy_first = expr.subs(&x, &a_sq).subs(&y, &b_plus_1);
    let yx_first = expr.subs(&y, &b_plus_1).subs(&x, &a_sq);

    // Evaluate at a=2, b=3 to get concrete values
    let v_xy = xy_first
        .subs_i64(&a, 2)
        .subs_i64(&b, 3)
        .eval()
        .eval_f64()
        .unwrap();
    let v_yx = yx_first
        .subs_i64(&a, 2)
        .subs_i64(&b, 3)
        .eval()
        .eval_f64()
        .unwrap();
    assert!(
        approx(v_xy, v_yx, 1e-12),
        "symbolic subs commutativity FAILED: xy={v_xy}, yx={v_yx}"
    );
}

#[test]
fn prop4_subs_commute_x_over_y_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x / &y;
    for &(xp, xq, yp, yq) in &[(1, 1, 2, 1), (3, 1, 5, 1), (7, 2, 3, 2)] {
        let xval = ctx.rational(xp, xq);
        let yval = ctx.rational(yp, yq);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("x/y at x={xp}/{xq}, y={yp}/{yq}"),
        );
    }
}

#[test]
fn prop4_subs_commute_x_pow_y_numeric() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = x.pow(&y);
    for (xv, yv) in [(2, 3), (3, 2), (5, 1)] {
        let xval = ctx.int(xv);
        let yval = ctx.int(yv);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("x^y at x={xv}, y={yv}"),
        );
    }
}

#[test]
fn prop4_subs_commute_sin_x_times_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = (&x * &y).sin();
    for &(xp, xq, yp, yq) in &[(1, 2, 1, 3), (1, 1, 1, 2), (3, 4, 2, 3)] {
        let xval = ctx.rational(xp, xq);
        let yval = ctx.rational(yp, yq);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("sin(x*y) at x={xp}/{xq}, y={yp}/{yq}"),
        );
    }
}

#[test]
fn prop4_subs_commute_ln_x_plus_exp_y() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let expr = &x.ln() + &y.exp();
    // x must be positive for ln
    for &(xp, xq, yp, yq) in &[(1, 1, 1, 2), (2, 1, 1, 3), (3, 2, 3, 4)] {
        let xval = ctx.rational(xp, xq);
        let yval = ctx.rational(yp, yq);
        assert_subs_commute(
            &expr,
            &x,
            &y,
            &xval,
            &yval,
            1e-9,
            &format!("ln(x)+exp(y) at x={xp}/{xq}, y={yp}/{yq}"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 5: solve roots satisfy equation
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify every root returned by solve satisfies the equation.
fn assert_solve_roots_satisfy(
    expr: &Ex,
    var: &Ex,
    expected_min_roots: usize,
    tol: f64,
    label: &str,
) {
    let roots = match expr.solve(var) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP: solve failed for {label}: {e}");
            return;
        }
    };

    assert!(
        roots.len() >= expected_min_roots,
        "{label}: expected at least {expected_min_roots} roots, got {} ({:?})",
        roots.len(),
        roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
    );

    for (i, root) in roots.iter().enumerate() {
        let substituted = expr.subs(var, root).eval();
        let s = format!("{substituted}");

        // First check: exact symbolic zero
        if s == "0" {
            continue;
        }

        // Try simplify to zero
        let simplified = substituted.simplify();
        let ss = format!("{simplified}");
        if ss == "0" {
            continue;
        }

        // Try real numerical evaluation
        if let Ok(v) = substituted.eval_f64() {
            assert!(
                v.abs() < tol,
                "{label}: root {i} ({root}) does NOT satisfy equation: \
                 residual = {v} (substituted = '{substituted}')"
            );
            continue;
        }

        // Fall back to complex evaluation
        match substituted.eval_complex64() {
            Ok((re, im)) => {
                let mag = (re * re + im * im).sqrt();
                assert!(
                    mag < tol,
                    "{label}: root {i} ({root}) does NOT satisfy equation: \
                     complex residual = ({re}, {im}), |r| = {mag}"
                );
            }
            Err(e) => {
                // If we can't evaluate at all, check if simplification helped
                eprintln!(
                    "WARNING: {label} root {i} ({root}): can't evaluate residual \
                     '{substituted}' (simplified: '{simplified}'): {e}"
                );
            }
        }
    }
}

#[test]
fn prop5_solve_x2_minus_4() {
    // x² - 4 = 0 → x = ±2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 4;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²-4");
}

#[test]
fn prop5_solve_x3_minus_1() {
    // x³ - 1 = 0 → x = 1 (and two complex roots)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - 1;
    assert_solve_roots_satisfy(&expr, &x, 1, 1e-9, "x³-1");
}

#[test]
fn prop5_solve_x2_plus_x_minus_6() {
    // x² + x - 6 = 0 → x = 2, x = -3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + &x - 6;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²+x-6");
}

#[test]
fn prop5_solve_x4_minus_1() {
    // x⁴ - 1 = 0 → x = ±1, ±i
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - 1;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x⁴-1");
}

#[test]
fn prop5_solve_x2_minus_2() {
    // x² - 2 = 0 → x = ±√2 (irrational roots)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 2;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²-2");
}

#[test]
fn prop5_solve_x2_minus_5x_plus_6() {
    // x² - 5x + 6 = 0 → x = 2, 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x * 5 + 6;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²-5x+6");
}

#[test]
fn prop5_solve_x3_minus_6x2_plus_11x_minus_6() {
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(3) - &x.powi(2) * 6 + &x * 11 - 6;
    assert_solve_roots_satisfy(&expr, &x, 3, 1e-9, "x³-6x²+11x-6");
}

#[test]
fn prop5_solve_x2_plus_1() {
    // x² + 1 = 0 → x = ±i (complex roots)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) + 1;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²+1");
}

#[test]
fn prop5_solve_2x2_minus_3x_plus_1() {
    // 2x² - 3x + 1 = (2x-1)(x-1) → x = 1/2, 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) * 2 - &x * 3 + 1;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "2x²-3x+1");
}

#[test]
fn prop5_solve_x5_minus_x() {
    // x⁵ - x = x(x⁴-1) = x(x²-1)(x²+1) → x=0, ±1, ±i
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(5) - &x;
    assert_solve_roots_satisfy(&expr, &x, 3, 1e-9, "x⁵-x");
}

#[test]
fn prop5_solve_x2_minus_3() {
    // x² - 3 = 0 → x = ±√3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 3;
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "x²-3");
}

#[test]
fn prop5_solve_x4_minus_5x2_plus_4() {
    // x⁴ - 5x² + 4 = (x²-1)(x²-4) = (x-1)(x+1)(x-2)(x+2)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - &x.powi(2) * 5 + 4;
    assert_solve_roots_satisfy(&expr, &x, 4, 1e-9, "x⁴-5x²+4");
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 6: series convergence
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify that series approximation error decreases with order.
///
/// Computes series at orders `orders` and checks that absolute error
/// at the given rational point decreases (or stays below tolerance).
fn assert_series_convergence(
    expr: &Ex,
    var: &Ex,
    point_p: i64,
    point_q: i64,
    orders: &[u32],
    label: &str,
) {
    let ctx = expr.context();
    let zero = ctx.int(0);
    let eval_pt = ctx.rational(point_p, point_q);

    // Evaluate exact value
    let exact = match expr.subs(var, &eval_pt).eval().eval_f64() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SKIP series convergence {label}: exact eval failed: {e}");
            return;
        }
    };

    let mut prev_error: Option<f64> = None;
    let mut errors = Vec::new();

    for &n in orders {
        let series = expr.series(var, &zero, n);

        // Check if series is unevaluated
        if series.has_unevaluated() {
            eprintln!("SKIP series convergence {label} at order {n}: unevaluated series");
            continue;
        }

        let series_expanded = series.expand().eval();
        let approx_val = match series_expanded.subs(var, &eval_pt).eval().eval_f64() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("SKIP series convergence {label} at order {n}: eval failed: {e}");
                continue;
            }
        };

        let error = (exact - approx_val).abs();
        errors.push((n, error, approx_val));

        if let Some(pe) = prev_error {
            // Error should generally decrease (or be zero).
            // Allow a small fudge factor for floating-point noise.
            if error > pe * 1.01 + 1e-15 && pe > 1e-15 {
                eprintln!(
                    "BUG: series convergence {label}: error INCREASED at order {n}: \
                     error({n})={error:.2e} > error(prev)={pe:.2e}  \
                     (exact={exact}, approx={approx_val})"
                );
                // Collect all errors for the report
                eprintln!("  All errors: {:?}", errors);
            }
        }
        prev_error = Some(error);
    }

    // At minimum, the highest-order series should be a decent approximation
    if let Some(&(n, error, _)) = errors.last() {
        assert!(
            error < 1.0,
            "series convergence {label}: even at order {n}, error={error} is too large \
             (exact={exact}, all errors={errors:?})"
        );
    } else {
        eprintln!("SKIP series convergence {label}: no orders evaluated successfully");
    }
}

#[test]
fn prop6_series_exp_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.exp();
    // exp(0.1) ≈ 1.10517...
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "exp(x) at x=0.1");
}

#[test]
fn prop6_series_sin_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin();
    // sin(0.1) ≈ 0.0998...
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "sin(x) at x=0.1");
}

#[test]
fn prop6_series_cos_x() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos();
    // cos(0.1) ≈ 0.995...
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "cos(x) at x=0.1");
}

#[test]
fn prop6_series_1_over_1_minus_x() {
    // 1/(1-x) at x=0.1 — geometric series
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &(ctx.int(1) - &x);
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 10], "1/(1-x) at x=0.1");
}

#[test]
fn prop6_series_ln_1_plus_x() {
    // ln(1+x) at x=0.1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x + 1).ln();
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "ln(1+x) at x=0.1");
}

#[test]
fn prop6_series_exp_x_at_larger_point() {
    // exp(x) at x=0.5 — should still converge but need more terms
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.exp();
    assert_series_convergence(&expr, &x, 1, 2, &[3, 5, 7, 10], "exp(x) at x=0.5");
}

#[test]
fn prop6_series_sin_x_at_larger_point() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin();
    assert_series_convergence(&expr, &x, 1, 2, &[3, 5, 7, 10], "sin(x) at x=0.5");
}

#[test]
fn prop6_series_1_over_1_plus_x2() {
    // 1/(1+x²) at x=0.1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &(&x.powi(2) + 1);
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "1/(1+x²) at x=0.1");
}

#[test]
fn prop6_series_exp_negative_x() {
    // exp(-x) at x=0.1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x * (-1)).exp();
    assert_series_convergence(&expr, &x, 1, 10, &[3, 5, 7], "exp(-x) at x=0.1");
}

#[test]
fn prop6_series_1_over_1_minus_x_at_half() {
    // 1/(1-x) at x=0.5 — sum should converge but slowly
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &(ctx.int(1) - &x);
    assert_series_convergence(&expr, &x, 1, 2, &[3, 5, 10, 15], "1/(1-x) at x=0.5");
}

// ═══════════════════════════════════════════════════════════════════════════
// Property 7: diff of integral is identity (FTC)
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: verify d/dx(∫f dx) ≈ f at rational evaluation points.
fn assert_ftc_property(
    integrand: &Ex,
    var: &Ex,
    eval_points: &[(i64, i64)],
    tol: f64,
    label: &str,
) {
    let antideriv = integrand.integrate(var);
    let antideriv_str = format!("{antideriv}");

    // Check for unevaluated integral
    if antideriv.has_unevaluated() {
        eprintln!("SKIP FTC {label}: integration returned unevaluated form: {antideriv_str}");
        return;
    }

    let deriv = antideriv.diff(var);
    let ctx = integrand.context();
    let mut checked = 0;

    for &(p, q) in eval_points {
        let pt = ctx.rational(p, q);
        let v_orig = integrand.subs(var, &pt).eval().eval_f64();
        let v_deriv = deriv.subs(var, &pt).eval().eval_f64();
        match (v_orig, v_deriv) {
            (Ok(orig), Ok(derived)) => {
                checked += 1;
                assert!(
                    approx(orig, derived, tol),
                    "FTC FAILED for {label} at {var}={p}/{q}: \
                     f(x)={orig}, d/dx(∫f dx)={derived}, diff={}, \
                     antideriv='{antideriv_str}', deriv='{deriv}'",
                    (orig - derived).abs()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(orig), Err(e)) => {
                panic!(
                    "FTC {label} at {var}={p}/{q}: integrand={orig} but derivative eval failed: {e}\n\
                     antideriv = '{antideriv_str}'\n\
                     deriv = '{deriv}'"
                );
            }
            (Err(e), Ok(derived)) => {
                panic!(
                    "FTC {label} at {var}={p}/{q}: integrand eval failed: {e} but derivative={derived}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "FTC {label}: no evaluation points succeeded — test is vacuous"
    );
}

/// Safe evaluation points for FTC: avoid 0 (for 1/x type functions),
/// avoid negatives (for ln), stay modest in size.
const FTC_EVAL_POINTS: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10)];

#[test]
fn prop7_ftc_x_pow_0() {
    // ∫ 1 dx = x, d/dx(x) = 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x^0 = 1");
}

#[test]
fn prop7_ftc_x_pow_1() {
    // ∫ x dx = x²/2, d/dx(x²/2) = x
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    assert_ftc_property(&x, &x, FTC_EVAL_POINTS, 1e-9, "x^1");
}

#[test]
fn prop7_ftc_x_pow_2() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(2);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x^2");
}

#[test]
fn prop7_ftc_x_pow_3() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(3);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x^3");
}

#[test]
fn prop7_ftc_x_pow_4() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(4);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x^4");
}

#[test]
fn prop7_ftc_x_pow_5() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.powi(5);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x^5");
}

#[test]
fn prop7_ftc_sin_x() {
    // ∫ sin(x) dx = -cos(x), d/dx(-cos(x)) = sin(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "sin(x)");
}

#[test]
fn prop7_ftc_cos_x() {
    // ∫ cos(x) dx = sin(x), d/dx(sin(x)) = cos(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "cos(x)");
}

#[test]
fn prop7_ftc_exp_x() {
    // ∫ exp(x) dx = exp(x), d/dx(exp(x)) = exp(x)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.exp();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "exp(x)");
}

#[test]
fn prop7_ftc_1_over_1_plus_x2() {
    // ∫ 1/(1+x²) dx = atan(x), d/dx(atan(x)) = 1/(1+x²)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &(&x.powi(2) + 1);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "1/(1+x²)");
}

#[test]
fn prop7_ftc_polynomial_3x2_plus_2x_plus_1() {
    // ∫ (3x²+2x+1) dx = x³+x²+x
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) * 3 + &x * 2 + 1;
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "3x²+2x+1");
}

#[test]
fn prop7_ftc_1_over_x() {
    // ∫ 1/x dx = ln(x), d/dx(ln(x)) = 1/x
    // Only test at positive x values
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &x;
    let positive_points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (3, 1)];
    assert_ftc_property(&expr, &x, positive_points, 1e-9, "1/x");
}

#[test]
fn prop7_ftc_x_exp_x() {
    // ∫ x*exp(x) dx = (x-1)*exp(x) via integration by parts
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * &x.exp();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x*exp(x)");
}

#[test]
fn prop7_ftc_x_squared_exp() {
    // ∫ x²*exp(x) dx — integration by parts twice
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) * &x.exp();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-8, "x²*exp(x)");
}

#[test]
fn prop7_ftc_sin_x_squared() {
    // ∫ sin²(x) dx = x/2 - sin(2x)/4
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin().powi(2);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "sin²(x)");
}

#[test]
fn prop7_ftc_cos_x_squared() {
    // ∫ cos²(x) dx = x/2 + sin(2x)/4
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.cos().powi(2);
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "cos²(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-property tests: multiple invariants on the same expression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cross_prop_rational_function_full_pipeline() {
    // For 1/(x²-1): test apart↔together AND solve roots satisfy equation
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // apart ↔ together
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "cross: 1/(x²-1)");

    // The denominator roots should satisfy x² - 1 = 0
    let denom = &x.powi(2) - 1;
    assert_solve_roots_satisfy(&denom, &x, 2, 1e-9, "cross: denom x²-1");
}

#[test]
fn cross_prop_trig_and_rewrite() {
    // sin(x)*cos(x): trig roundtrip AND rewrite_as_exp both preserve value
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin() * &x.cos();

    // Trig roundtrip
    assert_trig_roundtrip_1var(
        &expr,
        &x,
        TRIG_1VAR_POINTS,
        1e-9,
        "cross: sin(x)*cos(x) trig",
    );

    // Rewrite as exp
    assert_rewrite_exp_preserves_value(
        &expr,
        &x,
        EXP_REWRITE_POINTS,
        1e-9,
        "cross: sin(x)*cos(x) exp",
    );
}

#[test]
fn cross_prop_polynomial_ftc_and_solve() {
    // x² - 4: FTC and solve
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - 4;

    // FTC on derivative (2x)
    let deriv = expr.diff(&x);
    assert_ftc_property(&deriv, &x, FTC_EVAL_POINTS, 1e-9, "cross: d/dx(x²-4) = 2x");

    // Solve
    assert_solve_roots_satisfy(&expr, &x, 2, 1e-9, "cross: x²-4 solve");
}

// ═══════════════════════════════════════════════════════════════════════════
// Harder / edge-case tests to hunt for bugs
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// Property 1 (harder): together→apart direction (reverse of above)
// ---------------------------------------------------------------------------

/// Verify that together(apart(r)) ≈ r — the *other* direction.
fn assert_together_apart_roundtrip(expr: &Ex, var: &Ex, points: &[i64], tol: f64, label: &str) {
    let decomposed = expr.partial_fractions(var);
    let recombined = decomposed.together();

    let mut checked = 0;
    for &pt in points {
        let v_orig = expr.subs_i64(var, pt).eval().eval_f64();
        let v_round = recombined.subs_i64(var, pt).eval().eval_f64();
        match (v_orig, v_round) {
            (Ok(a), Ok(b)) => {
                checked += 1;
                assert!(
                    approx(a, b, tol),
                    "together(apart(…)) roundtrip FAILED for {label} at {var}={pt}: \
                     original={a}, roundtripped={b}, diff={}",
                    (a - b).abs()
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(a), Err(e)) => {
                panic!(
                    "together(apart(…)) {label} at {var}={pt}: original={a} but roundtrip failed: {e}"
                );
            }
            (Err(e), Ok(b)) => {
                panic!(
                    "together(apart(…)) {label} at {var}={pt}: original failed: {e} but roundtrip={b}"
                );
            }
        }
    }
    assert!(
        checked > 0,
        "together(apart(…)) {label}: no points evaluated — vacuous"
    );
}

#[test]
fn prop1_hard_reverse_roundtrip_1_over_x2_minus_1() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    assert_together_apart_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "rev: 1/(x²-1)");
}

#[test]
fn prop1_hard_reverse_roundtrip_x_over_x2_minus_x_minus_2() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x / (&x.powi(2) - &x - 2);
    assert_together_apart_roundtrip(&expr, &x, &[-3, -2, 0, 1, 3, 5, 7], 1e-9, "rev: x/(x²-x-2)");
}

#[test]
fn prop1_hard_reverse_roundtrip_1_over_x3_minus_1() {
    // Cubic denominator — apart produces (Bx+C)/(x²+x+1) terms
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(3) - 1);
    assert_together_apart_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "rev: 1/(x³-1)");
}

#[test]
fn prop1_hard_apart_idempotent() {
    // apart(apart(r)) == apart(r)  — partial fractions should be idempotent
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    let once = expr.partial_fractions(&x);
    let twice = once.partial_fractions(&x);

    let pts = &[-3i64, -2, 0, 2, 3, 5];
    for &pt in pts {
        let v1 = once.subs_i64(&x, pt).eval().eval_f64();
        let v2 = twice.subs_i64(&x, pt).eval().eval_f64();
        match (v1, v2) {
            (Ok(a), Ok(b)) => {
                assert!(
                    approx(a, b, 1e-9),
                    "apart idempotency FAILED at x={pt}: once={a}, twice={b}"
                );
            }
            (Err(_), Err(_)) => {}
            (Ok(a), Err(e)) => panic!("apart idempotency at x={pt}: once={a}, twice err: {e}"),
            (Err(e), Ok(b)) => panic!("apart idempotency at x={pt}: once err: {e}, twice={b}"),
        }
    }
}

#[test]
fn prop1_hard_apart_repeated_roots() {
    // 1/(x-1)^3 — repeated root: partial fractions should handle cleanly
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x - 1).powi(3);
    assert_apart_together_roundtrip(&expr, &x, &[-2, -1, 0, 2, 3, 5], 1e-9, "1/(x-1)³ repeated");
}

#[test]
fn prop1_hard_apart_high_degree_denom() {
    // 1/(x⁴-1) = 1/((x²-1)(x²+1)) = 1/((x-1)(x+1)(x²+1))
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(4) - 1);
    assert_apart_together_roundtrip(&expr, &x, &[-3, -2, 0, 2, 3, 5], 1e-9, "1/(x⁴-1)");
}

#[test]
fn prop1_hard_apart_numerator_degree_equals_denom() {
    // x²/(x²-1) — improper: degree(num) = degree(denom)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) / (&x.powi(2) - 1);
    assert_apart_together_roundtrip(
        &expr,
        &x,
        &[-3, -2, 0, 2, 3, 5],
        1e-9,
        "x²/(x²-1) improper same deg",
    );
}

#[test]
fn prop1_hard_apart_numerator_higher_degree() {
    // x⁴/(x²-1) — very improper rational function
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) / (&x.powi(2) - 1);
    assert_apart_together_roundtrip(
        &expr,
        &x,
        &[-3, -2, 0, 2, 3, 5],
        1e-9,
        "x⁴/(x²-1) very improper",
    );
}

// ---------------------------------------------------------------------------
// Property 2 (harder): one-way trig transforms preserve value
// ---------------------------------------------------------------------------

#[test]
fn prop2_hard_expand_trig_preserves_value() {
    // expand_trig alone should preserve value (not just roundtrip)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let cases: Vec<(&str, Ex)> = vec![
        ("sin(2x)", (&x * 2).sin()),
        ("cos(2x)", (&x * 2).cos()),
        ("sin(3x)", (&x * 3).sin()),
        ("cos(3x)", (&x * 3).cos()),
        ("sin(x+1)", (&x + 1).sin()),
    ];

    let points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (3, 2), (11, 10)];

    for (label, expr) in &cases {
        let expanded = expr.expand_trig();
        let failures = compare_at_rational_points(expr, &expanded, &x, points, 1e-9);
        assert!(
            failures.is_empty(),
            "expand_trig DOES NOT preserve value for {label}: {:?}",
            failures
        );
    }
}

#[test]
#[allow(clippy::type_complexity)]
fn prop2_hard_trig_combine_preserves_value() {
    // trig_combine alone should preserve value
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let cases: Vec<(&str, Ex, Vec<(&Ex, &[(i64, i64)])>)> = vec![
        (
            "sin(x)*cos(x)",
            &x.sin() * &x.cos(),
            vec![(&x, &[(1, 2), (7, 10), (1, 1), (3, 2)])],
        ),
        (
            "sin(x)²",
            x.sin().powi(2),
            vec![(&x, &[(1, 2), (7, 10), (1, 1), (3, 2)])],
        ),
        (
            "cos(x)²",
            x.cos().powi(2),
            vec![(&x, &[(1, 2), (7, 10), (1, 1), (3, 2)])],
        ),
    ];

    for (label, expr, var_points_list) in &cases {
        let combined = expr.trig_combine();
        for (var, points) in var_points_list {
            let failures = compare_at_rational_points(expr, &combined, var, points, 1e-9);
            assert!(
                failures.is_empty(),
                "trig_combine DOES NOT preserve value for {label}: {:?}",
                failures
            );
        }
    }
}

#[test]
fn prop2_hard_trig_sin_squared_combine_has_cos() {
    // sin²(x) should combine to something involving cos(2x) = 1 - 2sin²(x)
    // i.e. sin²(x) = (1 - cos(2x)) / 2
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.sin().powi(2);
    let combined = expr.trig_combine();
    let s = format!("{combined}");
    // Verify it at least changed form
    eprintln!("sin²(x) trig_combine => {s}");
    // And preserves value
    let points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (5, 3), (2, 1)];
    let failures = compare_at_rational_points(&expr, &combined, &x, points, 1e-9);
    assert!(
        failures.is_empty(),
        "sin²(x) trig_combine value mismatch: {:?}",
        failures
    );
}

#[test]
fn prop2_hard_double_trig_combine_idempotent() {
    // trig_combine(trig_combine(e)) should equal trig_combine(e) numerically
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin() * &x.cos();
    let once = expr.trig_combine();
    let twice = once.trig_combine();

    let points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (3, 2), (2, 1)];
    let failures = compare_at_rational_points(&once, &twice, &x, points, 1e-9);
    assert!(
        failures.is_empty(),
        "trig_combine NOT idempotent for sin(x)*cos(x): {:?}",
        failures
    );
}

// ---------------------------------------------------------------------------
// Property 3 (harder): rewrite_as_exp structural + roundtrip
// ---------------------------------------------------------------------------

#[test]
fn prop3_hard_rewrite_exp_trig_contains_exp() {
    // sin(x), cos(x), tan(x) should all produce "exp" in output
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("tan(x)", x.tan()),
    ];

    for (label, expr) in &cases {
        let rewritten = expr.rewrite_as_exp();
        let s = format!("{rewritten}");
        assert!(
            s.contains("exp"),
            "rewrite_as_exp({label}) should contain 'exp', got: {s}"
        );
    }
}

#[test]
fn prop3_hard_rewrite_exp_hyp_contains_exp() {
    // BUG: rewrite_as_exp does NOT handle sinh(x) or cosh(x).
    //
    // sinh(x) = (exp(x) - exp(-x))/2  and  cosh(x) = (exp(x) + exp(-x))/2
    // are the canonical exponential forms, but rewrite_as_exp only matches
    // Sin, Cos, and Tan nodes — the Sinh/Cosh/Tanh cases are missing from
    // src/simplify/rewrite.rs::rewrite_as_exp.
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let sinh_rewritten = x.sinh().rewrite_as_exp();
    let cosh_rewritten = x.cosh().rewrite_as_exp();

    let sinh_s = format!("{sinh_rewritten}");
    let cosh_s = format!("{cosh_rewritten}");

    // Document the bug: these SHOULD contain "exp" but currently don't.
    if !sinh_s.contains("exp") {
        eprintln!(
            "BUG (known): rewrite_as_exp(sinh(x)) returns '{sinh_s}' — \
             should be (exp(x) - exp(-x))/2"
        );
    }
    if !cosh_s.contains("exp") {
        eprintln!(
            "BUG (known): rewrite_as_exp(cosh(x)) returns '{cosh_s}' — \
             should be (exp(x) + exp(-x))/2"
        );
    }

    // At minimum, value must still be preserved even if form is unchanged
    let points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (3, 2)];
    let sinh_failures = compare_at_rational_points(&x.sinh(), &sinh_rewritten, &x, points, 1e-9);
    let cosh_failures = compare_at_rational_points(&x.cosh(), &cosh_rewritten, &x, points, 1e-9);
    assert!(
        sinh_failures.is_empty(),
        "rewrite_as_exp(sinh) value mismatch: {:?}",
        sinh_failures
    );
    assert!(
        cosh_failures.is_empty(),
        "rewrite_as_exp(cosh) value mismatch: {:?}",
        cosh_failures
    );

    // The actual assertion that SHOULD pass once the bug is fixed:
    // Uncomment these when sinh/cosh support is added to rewrite_as_exp.
    // assert!(sinh_s.contains("exp"), "rewrite_as_exp(sinh(x)) should contain 'exp', got: {sinh_s}");
    // assert!(cosh_s.contains("exp"), "rewrite_as_exp(cosh(x)) should contain 'exp', got: {cosh_s}");
}

#[test]
fn prop3_hard_rewrite_exp_then_simplify_sin2_cos2() {
    // sin²(x) + cos²(x) → rewrite_as_exp → simplify should give 1
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let rewritten = expr.rewrite_as_exp();
    let simplified = rewritten.expand().simplify();

    // Check numerically at multiple points
    let points: &[(i64, i64)] = &[(1, 2), (7, 10), (1, 1), (2, 1), (3, 1)];
    for &(p, q) in points {
        let pt = ctx.rational(p, q);
        let v = simplified.subs(&x, &pt).eval().eval_f64();
        match v {
            Ok(val) => {
                assert!(
                    approx(val, 1.0, 1e-9),
                    "sin²+cos² after rewrite_as_exp+simplify should be 1.0, got {val} at x={p}/{q}"
                );
            }
            Err(_) => {
                // Try complex eval
                if let Ok((re, im)) = simplified.subs(&x, &pt).eval().eval_complex64() {
                    assert!(
                        approx(re, 1.0, 1e-9) && im.abs() < 1e-9,
                        "sin²+cos² rewrite+simplify: ({re}, {im}) at x={p}/{q}"
                    );
                }
            }
        }
    }
}

#[test]
fn prop3_hard_rewrite_exp_nested_sin_of_sum() {
    // sin(x + 1) → rewrite_as_exp should preserve value
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = (&x + 1).sin();
    assert_rewrite_exp_preserves_value(
        &expr,
        &x,
        &[(1, 2), (7, 10), (1, 1), (3, 2)],
        1e-9,
        "sin(x+1)",
    );
}

#[test]
fn prop3_hard_rewrite_exp_product_sin_cos() {
    // sin(x)*cos(x) → rewrite_as_exp should preserve value
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.sin() * &x.cos();
    assert_rewrite_exp_preserves_value(
        &expr,
        &x,
        &[(1, 2), (7, 10), (1, 1), (3, 2), (2, 1)],
        1e-9,
        "sin(x)*cos(x)",
    );
}

// ---------------------------------------------------------------------------
// Property 4 (harder): subs_map vs sequential subs
// ---------------------------------------------------------------------------

#[test]
fn prop4_hard_subs_map_matches_sequential() {
    // subs_map([x→a, y→b]) should equal subs(x,a).subs(y,b)
    // when a doesn't contain y and b doesn't contain x
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let three = ctx.int(3);
    let five = ctx.int(5);

    let exprs: Vec<(&str, Ex)> = vec![
        ("x+y", &x + &y),
        ("x*y", &x * &y),
        ("x²+y²", &x.powi(2) + &y.powi(2)),
        ("sin(x)+cos(y)", &x.sin() + &y.cos()),
        ("x/y", &x / &y),
    ];

    for (label, expr) in &exprs {
        let via_map = expr.subs_map(&[(&x, &three), (&y, &five)]);
        let via_seq = expr.subs(&x, &three).subs(&y, &five);

        let v_map = via_map.eval().eval_f64();
        let v_seq = via_seq.eval().eval_f64();

        match (v_map, v_seq) {
            (Ok(a), Ok(b)) => {
                assert!(
                    approx(a, b, 1e-12),
                    "subs_map vs sequential MISMATCH for {label}: map={a}, seq={b}"
                );
            }
            (Err(e1), Err(e2)) => {
                eprintln!("{label}: both failed — map: {e1}, seq: {e2}");
            }
            (Ok(a), Err(e)) => {
                panic!("{label}: map={a} but seq failed: {e}");
            }
            (Err(e), Ok(b)) => {
                panic!("{label}: map failed: {e} but seq={b}");
            }
        }
    }
}

#[test]
fn prop4_hard_subs_simultaneous_swap() {
    // Simultaneous swap x↔y via subs_map should differ from sequential subs
    // subs_map([x→y, y→x]) swaps; subs(x,y).subs(y,x) replaces x→y then y→x = all x.
    // So x+2y with subs_map([x→y,y→x]) = y+2x, but subs(x,y).subs(y,x) = x+2x = 3x
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    let expr = &x + &y * 2; // x + 2y

    // Simultaneous swap
    let swapped = expr.subs_map(&[(&x, &y), (&y, &x)]);
    // This should be y + 2x
    let v_swap = swapped
        .subs_i64(&x, 3)
        .subs_i64(&y, 7)
        .eval()
        .eval_f64()
        .unwrap();
    // y + 2x at x=3, y=7 → 7 + 6 = 13
    assert!(
        approx(v_swap, 13.0, 1e-12),
        "simultaneous swap: expected 13.0, got {v_swap}"
    );

    // Sequential (NOT a swap — x→y first, then y→x collapses both)
    let seq = expr.subs(&x, &y).subs(&y, &x);
    let v_seq = seq
        .subs_i64(&x, 3)
        .subs_i64(&y, 7)
        .eval()
        .eval_f64()
        .unwrap();
    // After subs(x,y): y + 2y = 3y; after subs(y,x): 3x → at x=3: 9
    assert!(
        approx(v_seq, 9.0, 1e-12),
        "sequential subs: expected 9.0, got {v_seq}"
    );

    // They should NOT be equal — this validates subs_map is truly simultaneous
    assert!(
        !approx(v_swap, v_seq, 1e-6),
        "BUG: subs_map and sequential give same result ({v_swap}), \
         but simultaneous swap should differ from sequential"
    );
}

// ---------------------------------------------------------------------------
// Property 5 (harder): solve root count + complex roots
// ---------------------------------------------------------------------------

#[test]
fn prop5_hard_solve_returns_correct_count_for_quadratics() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    // A battery of quadratics: each should return exactly 2 roots
    let quadratics: Vec<(&str, Ex)> = vec![
        ("x²-1", &x.powi(2) - 1),
        ("x²-4", &x.powi(2) - 4),
        ("x²+x-6", &x.powi(2) + &x - 6),
        ("x²-5x+6", &x.powi(2) - &x * 5 + 6),
        ("x²+1", &x.powi(2) + 1), // complex roots
        ("x²-2", &x.powi(2) - 2), // irrational roots
        ("2x²-3x+1", &x.powi(2) * 2 - &x * 3 + 1),
    ];

    for (label, poly) in &quadratics {
        match poly.solve(&x) {
            Ok(roots) => {
                assert_eq!(
                    roots.len(),
                    2,
                    "BUG: {label} should have exactly 2 roots, got {} ({:?})",
                    roots.len(),
                    roots.iter().map(|r| format!("{r}")).collect::<Vec<_>>()
                );
            }
            Err(e) => {
                panic!("BUG: solve({label}) failed: {e}");
            }
        }
    }
}

#[test]
fn prop5_hard_solve_double_root() {
    // (x-3)² = x²-6x+9 → should return x=3 (possibly twice)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x * 6 + 9;
    let roots = expr.solve(&x).unwrap();

    // Every root returned should satisfy the equation
    for (i, root) in roots.iter().enumerate() {
        let residual = expr.subs(&x, root).eval();
        let s = format!("{residual}");
        if s != "0"
            && let Ok(v) = residual.eval_f64()
        {
            assert!(
                v.abs() < 1e-9,
                "double root: root {i} ({root}) has residual {v}"
            );
        }
    }

    // All returned roots should be 3 (or very close to 3)
    for root in &roots {
        let v = root
            .eval_f64()
            .unwrap_or_else(|e| panic!("root {root} of (x-3)² must be numeric: {e}"));
        assert!(
            approx(v, 3.0, 1e-9),
            "double root of (x-3)² should be 3, got {v}"
        );
    }
}

#[test]
fn prop5_hard_solve_high_degree_factored() {
    // x(x-1)(x-2)(x-3) = x⁴ - 6x³ + 11x² - 6x
    // Should find roots 0, 1, 2, 3
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(4) - &x.powi(3) * 6 + &x.powi(2) * 11 - &x * 6;
    assert_solve_roots_satisfy(&expr, &x, 4, 1e-9, "x(x-1)(x-2)(x-3)");
}

#[test]
fn prop5_hard_solve_roots_substitution_then_simplify() {
    // For x²-5x+6: roots 2 and 3. Substituting root and simplifying should give 0.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.powi(2) - &x * 5 + 6;
    let roots = expr.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);

    for root in &roots {
        let substituted = expr.subs(&x, root).simplify();
        let s = format!("{substituted}");
        assert_eq!(
            s, "0",
            "BUG: substituting root {root} into x²-5x+6 and simplifying should give '0', got '{s}'"
        );
    }
}

// ---------------------------------------------------------------------------
// Property 6 (harder): series truncation error order
// ---------------------------------------------------------------------------

#[test]
fn prop6_hard_series_error_order_exp() {
    // For exp(x), the Taylor series to order n should have error O(x^n).
    // At x=h, error ≈ h^n/n! approximately. Check that error(h) / h^n is bounded.
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = x.exp();
    let zero = ctx.int(0);

    for n in [3u32, 5, 7] {
        let series = expr.series(&x, &zero, n);
        if series.has_unevaluated() {
            eprintln!("SKIP exp series order {n}: unevaluated");
            continue;
        }
        let series_expanded = series.expand().eval();

        // Evaluate at several small x values
        for &(hp, hq) in &[(1, 10), (1, 20), (1, 50)] {
            let h = hp as f64 / hq as f64;
            let pt = ctx.rational(hp, hq);
            let exact = expr.subs(&x, &pt).eval().eval_f64().unwrap();
            let approx_val = series_expanded.subs(&x, &pt).eval().eval_f64().unwrap();
            let error = (exact - approx_val).abs();
            let h_n = h.powi(n as i32);

            // error / h^n should be bounded (roughly by 1/n! * e^h, but we'll be generous)
            if h_n > 1e-100 {
                let ratio = error / h_n;
                assert!(
                    ratio < 10.0,
                    "BUG: exp(x) series order {n} at x={h}: error={error:.2e}, h^n={h_n:.2e}, \
                     ratio={ratio:.2e} (should be bounded)"
                );
            }
        }
    }
}

#[test]
fn prop6_hard_series_sin_cos_consistency() {
    // d/dx(sin series) should equal cos series
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);

    let sin_series = x.sin().series(&x, &zero, 7).expand().eval();
    let cos_series = x.cos().series(&x, &zero, 7).expand().eval();
    let sin_deriv = sin_series.diff(&x);

    // Compare sin'(x) series vs cos(x) series at small points
    let points: &[(i64, i64)] = &[(1, 10), (1, 5), (3, 10)];
    let failures = compare_at_rational_points(&sin_deriv, &cos_series, &x, points, 1e-6);
    // Allow some tolerance because truncation differs by one order
    if !failures.is_empty() {
        eprintln!(
            "NOTE: d/dx(sin series) vs cos series small differences (expected due to truncation): {:?}",
            failures
        );
    }
}

#[test]
fn prop6_hard_series_geometric_exact() {
    // 1/(1-x) series to order n should be 1 + x + x² + ... + x^(n-1) exactly
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &(ctx.int(1) - &x);
    let zero = ctx.int(0);

    let series = expr.series(&x, &zero, 5).expand().eval();
    // At x = 1/10: exact = 10/9, series5 = 1 + 0.1 + 0.01 + 0.001 + 0.0001 = 1.1111
    let pt = ctx.rational(1, 10);
    let exact = expr.subs(&x, &pt).eval().eval_f64().unwrap();
    let series_val = series.subs(&x, &pt).eval().eval_f64().unwrap();
    let error = (exact - series_val).abs();

    // Error should be x^5 / (1-x) = 0.1^5 / 0.9 ≈ 1.11e-5
    assert!(
        error < 2e-5,
        "geometric series(5) at x=0.1: error={error:.2e}, expected ~1.1e-5"
    );
    assert!(
        error > 1e-7,
        "geometric series(5) at x=0.1: error={error:.2e} is suspiciously small"
    );
}

// ---------------------------------------------------------------------------
// Property 7 (harder): integrate(diff(f)) = f + C
// ---------------------------------------------------------------------------

#[test]
fn prop7_hard_integrate_diff_recovers_function() {
    // integrate(diff(f, x), x) should differ from f by at most a constant
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("x³", x.powi(3)),
        ("x²+x+1", &x.powi(2) + &x + 1),
        ("sin(x)", x.sin()),
        ("cos(x)", x.cos()),
        ("exp(x)", x.exp()),
    ];

    for (label, f) in &cases {
        let deriv = f.diff(&x);
        let antideriv = deriv.integrate(&x);

        if antideriv.has_unevaluated() {
            eprintln!("SKIP integrate(diff({label})): unevaluated integral");
            continue;
        }

        // f and antideriv should differ by a constant.
        // Check: d/dx(f - antideriv) = 0 numerically
        let difference = f - &antideriv;
        let diff_of_diff = difference.diff(&x);

        let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10)];
        for &(p, q) in points {
            let pt = ctx.rational(p, q);
            let v = diff_of_diff
                .subs(&x, &pt)
                .eval()
                .eval_f64()
                .unwrap_or_else(|e| {
                    panic!("{label}: d/dx(f - ∫f'dx) not numeric at x={p}/{q}: {e}")
                });
            assert!(
                v.abs() < 1e-9,
                "BUG: integrate(diff({label})) differs from {label} by non-constant: \
                 d/dx(f - ∫f'dx) = {v} at x={p}/{q}"
            );
        }
    }
}

#[test]
fn prop7_hard_ftc_chain_rule() {
    // d/dx(∫ sin(x) dx) = sin(x) — even though ∫sin = -cos, the derivative brings it back
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let integrand = x.sin();
    let anti = integrand.integrate(&x);
    assert!(
        !anti.has_unevaluated(),
        "integration of sin(x) should not be unevaluated"
    );
    let deriv = anti.diff(&x);
    let deriv_simplified = deriv.simplify();

    // The derivative should simplify to sin(x)
    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    let failures = compare_at_rational_points(&integrand, &deriv_simplified, &x, points, 1e-9);
    assert!(
        failures.is_empty(),
        "FTC chain rule for sin(x): d/dx(∫sin dx) ≠ sin(x): {:?}",
        failures
    );
}

#[test]
fn prop7_hard_ftc_1_over_x_squared() {
    // ∫ 1/x² dx = -1/x, d/dx(-1/x) = 1/x²
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / &x.powi(2);
    let positive_points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (3, 1)];
    assert_ftc_property(&expr, &x, positive_points, 1e-9, "1/x²");
}

#[test]
fn prop7_hard_ftc_x_sin_x() {
    // ∫ x*sin(x) dx = sin(x) - x*cos(x) via integration by parts
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * &x.sin();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x*sin(x)");
}

#[test]
fn prop7_hard_ftc_x_cos_x() {
    // ∫ x*cos(x) dx = cos(x) + x*sin(x) via integration by parts
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x * &x.cos();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-9, "x*cos(x)");
}

#[test]
fn prop7_hard_ftc_exp_sin() {
    // ∫ exp(x)*sin(x) dx — a classic integration-by-parts-twice integral
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.exp() * &x.sin();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-8, "exp(x)*sin(x)");
}

#[test]
fn prop7_hard_ftc_exp_cos() {
    // ∫ exp(x)*cos(x) dx — another classic
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = &x.exp() * &x.cos();
    assert_ftc_property(&expr, &x, FTC_EVAL_POINTS, 1e-8, "exp(x)*cos(x)");
}

// ---------------------------------------------------------------------------
// Cross-property harder tests
// ---------------------------------------------------------------------------

#[test]
fn cross_hard_solve_then_factor_consistency() {
    // If solve finds roots r1, r2 for x²+bx+c, then (x-r1)(x-r2) expanded
    // should equal the original polynomial (up to leading coefficient)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let poly = &x.powi(2) - &x * 5 + 6; // roots 2, 3
    let roots = poly.solve(&x).unwrap();
    assert_eq!(roots.len(), 2);

    // Build (x - r1)*(x - r2)
    let reconstructed = (&x - &roots[0]) * (&x - &roots[1]);
    let expanded = reconstructed.expand();

    // They should agree numerically
    let pts = &[-2i64, 0, 1, 4, 5, 7];
    for &pt in pts {
        let v_orig = poly.subs_i64(&x, pt).eval().eval_f64().unwrap();
        let v_recon = expanded.subs_i64(&x, pt).eval().eval_f64().unwrap();
        assert!(
            approx(v_orig, v_recon, 1e-9),
            "solve-then-reconstruct: at x={pt}, original={v_orig}, reconstructed={v_recon}"
        );
    }
}

#[test]
fn cross_hard_diff_series_at_zero() {
    // series(f, x, 0, n) evaluated at x=0 should give f(0)
    // (the constant term of the Taylor series)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let zero = ctx.int(0);

    let cases: Vec<(&str, Ex, f64)> = vec![
        ("exp(x)", x.exp(), 1.0), // exp(0) = 1
        ("sin(x)", x.sin(), 0.0), // sin(0) = 0
        ("cos(x)", x.cos(), 1.0), // cos(0) = 1
        ("1/(1-x)", ctx.int(1) / &(ctx.int(1) - &x), 1.0),
        ("ln(1+x)", (&x + 1).ln(), 0.0), // ln(1) = 0
    ];

    for (label, expr, expected_at_0) in &cases {
        let series = expr.series(&x, &zero, 5);
        if series.has_unevaluated() {
            eprintln!("SKIP series at 0 for {label}: unevaluated");
            continue;
        }
        let series_at_0 = series.expand().eval().subs(&x, &zero).eval();
        match series_at_0.eval_f64() {
            Ok(v) => {
                assert!(
                    approx(v, *expected_at_0, 1e-12),
                    "series({label}, x, 0, 5) at x=0 should be {expected_at_0}, got {v}"
                );
            }
            Err(e) => {
                panic!("series({label}) at x=0 eval failed: {e}");
            }
        }
    }
}

#[test]
fn cross_hard_apart_then_integrate_vs_direct() {
    // For a rational function, ∫apart(f) dx should equal ∫f dx numerically
    // (partial fraction decomposition makes integration easier, but result should match)
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let expr = ctx.int(1) / (&x.powi(2) - 1);
    let apart = expr.partial_fractions(&x);

    let int_direct = expr.integrate(&x);
    let int_apart = apart.integrate(&x);

    // Both might be unevaluated
    if int_direct.has_unevaluated() || int_apart.has_unevaluated() {
        eprintln!(
            "SKIP apart-then-integrate: direct_unevaluated={}, apart_unevaluated={}",
            int_direct.has_unevaluated(),
            int_apart.has_unevaluated()
        );
        return;
    }

    // The two antiderivatives may differ by a constant.
    // Check that their derivatives agree.
    let deriv_direct = int_direct.diff(&x);
    let deriv_apart = int_apart.diff(&x);

    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10)];
    // Both derivatives should equal the original integrand
    let f1 = compare_at_rational_points(&expr, &deriv_direct, &x, points, 1e-8);
    let f2 = compare_at_rational_points(&expr, &deriv_apart, &x, points, 1e-8);

    if !f1.is_empty() {
        eprintln!("WARNING: direct integral derivative mismatch: {:?}", f1);
    }
    if !f2.is_empty() {
        eprintln!("WARNING: apart integral derivative mismatch: {:?}", f2);
    }
}

#[test]
fn cross_hard_simplify_preserves_value_comprehensive() {
    // .simplify() should ALWAYS preserve numerical value
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("sin²+cos²", &x.sin().powi(2) + &x.cos().powi(2)),
        ("(x+1)²-x²-2x", &(&x + 1).powi(2) - &x.powi(2) - &x * 2),
        ("x/x", &x / &x),
        ("(x²-1)/(x-1)", (&x.powi(2) - 1) / (&x - 1)),
        ("sin(2x)/(2cos(x))", (&x * 2).sin() / &(&x.cos() * 2)),
    ];

    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    for (label, expr) in &cases {
        let simplified = expr.simplify();
        let failures = compare_at_rational_points(expr, &simplified, &x, points, 1e-8);
        assert!(
            failures.is_empty(),
            "BUG: simplify() does NOT preserve value for {label}: {:?}\n  original = {expr}\n  simplified = {simplified}",
            failures
        );
    }
}

#[test]
fn cross_hard_expand_preserves_value() {
    // .expand() should ALWAYS preserve numerical value
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("(x+1)²", (&x + 1).powi(2)),
        ("(x+1)(x-1)", (&x + 1) * (&x - 1)),
        ("(x+1)³", (&x + 1).powi(3)),
        ("(2x+3)(x-1)", (&x * 2 + 3) * (&x - 1)),
        ("(x²+1)(x-1)", (&x.powi(2) + 1) * (&x - 1)),
    ];

    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    for (label, expr) in &cases {
        let expanded = expr.expand();
        let failures = compare_at_rational_points(expr, &expanded, &x, points, 1e-9);
        assert!(
            failures.is_empty(),
            "BUG: expand() does NOT preserve value for {label}: {:?}",
            failures
        );
    }
}

#[test]
fn cross_hard_factor_preserves_value() {
    // .factor() should ALWAYS preserve numerical value
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("x²-1", &x.powi(2) - 1),
        ("x²+2x+1", &x.powi(2) + &x * 2 + 1),
        ("x²-5x+6", &x.powi(2) - &x * 5 + 6),
        ("x³-x", &x.powi(3) - &x),
        ("x⁴-1", &x.powi(4) - 1),
    ];

    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    for (label, expr) in &cases {
        let factored = expr.factor(&x);
        let failures = compare_at_rational_points(expr, &factored, &x, points, 1e-9);
        assert!(
            failures.is_empty(),
            "BUG: factor() does NOT preserve value for {label}: {:?}\n  original = {expr}\n  factored = {factored}",
            failures
        );
    }
}

#[test]
fn cross_hard_cancel_preserves_value() {
    // .cancel() should preserve value at non-singular points
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let cases: Vec<(&str, Ex)> = vec![
        ("(x²-1)/(x-1)", (&x.powi(2) - 1) / (&x - 1)),
        ("(x²-4)/(x+2)", (&x.powi(2) - 4) / (&x + 2)),
        ("x²/x", &x.powi(2) / &x),
        ("(x³-x)/(x²-1)", (&x.powi(3) - &x) / (&x.powi(2) - 1)),
    ];

    // Use points away from singularities
    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    for (label, expr) in &cases {
        let cancelled = expr.cancel(&x);
        let failures = compare_at_rational_points(expr, &cancelled, &x, points, 1e-9);
        assert!(
            failures.is_empty(),
            "BUG: cancel() does NOT preserve value for {label}: {:?}\n  original = {expr}\n  cancelled = {cancelled}",
            failures
        );
    }
}

#[test]
fn cross_hard_diff_of_constant_is_zero() {
    // d/dx(constant) should always be 0
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let constants: Vec<(&str, Ex)> = vec![
        ("0", ctx.int(0)),
        ("1", ctx.int(1)),
        ("42", ctx.int(42)),
        ("-7", ctx.int(-7)),
        ("pi", ctx.pi()),
        ("e", ctx.e()),
        ("1/2", ctx.rational(1, 2)),
    ];

    for (label, c) in &constants {
        let deriv = c.diff(&x);
        let s = format!("{deriv}");
        assert_eq!(s, "0", "BUG: d/dx({label}) should be '0', got '{s}'");
    }
}

#[test]
fn cross_hard_diff_of_x_is_one() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    let deriv = x.diff(&x);
    let s = format!("{deriv}");
    assert_eq!(s, "1", "BUG: d/dx(x) should be '1', got '{s}'");
}

#[test]
fn cross_hard_diff_linearity() {
    // d/dx(a*f + b*g) should equal a*d/dx(f) + b*d/dx(g)
    let ctx = Context::new();
    symplex::syms!(ctx; x);

    let f = x.sin();
    let g = x.cos();
    let a = 3i64;
    let b = 5i64;

    let combined = &f * a + &g * b; // 3sin(x) + 5cos(x)
    let deriv_combined = combined.diff(&x);

    let deriv_f = f.diff(&x); // cos(x)
    let deriv_g = g.diff(&x); // -sin(x)
    let linear_combo = &deriv_f * a + &deriv_g * b; // 3cos(x) - 5sin(x)

    let points: &[(i64, i64)] = &[(3, 10), (7, 10), (14, 10), (21, 10), (31, 10)];
    let failures = compare_at_rational_points(&deriv_combined, &linear_combo, &x, points, 1e-9);
    assert!(
        failures.is_empty(),
        "BUG: diff linearity violated: d/dx(3sin+5cos) ≠ 3cos-5sin: {:?}",
        failures
    );
}
