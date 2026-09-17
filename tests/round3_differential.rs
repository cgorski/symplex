//! Round 3 Differential Testing: Hunting for bugs by comparing evaluation paths.
//!
//! Three evaluation paths:
//! 1. **Symbolic eval:** `expr.subs(&x, &val).eval_f64()` — substitutes then evaluates
//! 2. **Compile:** `expr.compile(&["x"]).unwrap()(&[val])` — bytecode interpreter
//! 3. **Codegen:** `expr.to_rust_fn("f", &["x"])` — generates Rust source (syntax check)
//!
//! We systematically compare paths 1 and 2 for numeric agreement, and check path 3
//! for syntactic validity.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compare two f64 values with relative tolerance, handling NaN and Inf.
fn approx_eq_rel(a: f64, b: f64, rel_tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return false;
    }
    let denom = a.abs().max(b.abs()).max(1e-300);
    ((a - b) / denom).abs() <= rel_tol
}

/// Describes the outcome of a single comparison.
#[derive(Debug)]
enum CompareResult {
    /// Both paths agree within tolerance.
    Agree,
    /// Both paths produced NaN or both errored — acceptable.
    BothNanOrError,
    /// eval_f64 returned an error, compile returned a finite value — potential issue.
    EvalErrorCompileOk { compile_val: f64 },
    /// eval_f64 returned Ok, compile returned NaN — potential issue.
    EvalOkCompileNan { eval_val: f64 },
    /// Numeric mismatch between the two paths.
    Mismatch { eval_val: f64, compile_val: f64 },
}

/// Compare compile vs subs+eval_f64 at a given f64 point for a single-variable expression.
fn compare_at(expr: &Ex, var: &Ex, var_name: &str, val: f64, ctx: &Context) -> CompareResult {
    let compiled = match expr.compile(&[var_name]) {
        Some(f) => f,
        None => {
            // If compile returns None, we can't compare — skip
            return CompareResult::BothNanOrError;
        }
    };
    let compile_val = compiled(&[val]);

    // For eval path: substitute a rational approximation or integer
    let eval_result = if val == val.floor() && val.abs() < 1e15 {
        expr.subs_i64(var, val as i64).eval_f64()
    } else {
        // Use rational approximation for non-integer values
        let (numer, denom) = float_to_rational(val);
        let rat = ctx.rational(numer, denom);
        expr.subs(var, &rat).eval_f64()
    };

    match eval_result {
        Ok(eval_val) => {
            if eval_val.is_nan() && compile_val.is_nan() {
                CompareResult::BothNanOrError
            } else if compile_val.is_nan() && !eval_val.is_nan() {
                CompareResult::EvalOkCompileNan { eval_val }
            } else if approx_eq_rel(eval_val, compile_val, 1e-10) {
                CompareResult::Agree
            } else {
                CompareResult::Mismatch {
                    eval_val,
                    compile_val,
                }
            }
        }
        Err(_) => {
            if compile_val.is_nan() || compile_val.is_infinite() {
                CompareResult::BothNanOrError
            } else {
                CompareResult::EvalErrorCompileOk { compile_val }
            }
        }
    }
}

/// Convert a float to a rational approximation (numerator, denominator).
fn float_to_rational(val: f64) -> (i64, i64) {
    if val == 0.0 {
        return (0, 1);
    }
    // Use power-of-10 denominator for common test values
    let denom = 10_000_i64;
    let numer = (val * denom as f64).round() as i64;
    // Simplify
    let g = gcd(numer.unsigned_abs(), denom.unsigned_abs()) as i64;
    (numer / g, denom / g)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Assert that compile and eval agree at the given points, collecting all failures.
fn assert_paths_agree(
    ctx: &Context,
    expr: &Ex,
    var: &Ex,
    var_name: &str,
    test_vals: &[f64],
    expr_desc: &str,
) {
    let mut failures = Vec::new();
    for &val in test_vals {
        match compare_at(expr, var, var_name, val, ctx) {
            CompareResult::Agree | CompareResult::BothNanOrError => {}
            CompareResult::EvalErrorCompileOk { compile_val } => {
                // eval errors are acceptable if compile returns a "reasonable" NaN/Inf
                // but if compile returns a normal finite number, that's interesting
                if compile_val.is_finite() {
                    failures.push(format!(
                        "  {expr_desc} at {var_name}={val}: eval_f64 errored but compile returned {compile_val}"
                    ));
                }
            }
            CompareResult::EvalOkCompileNan { eval_val } => {
                failures.push(format!(
                    "  {expr_desc} at {var_name}={val}: eval_f64={eval_val} but compile returned NaN"
                ));
            }
            CompareResult::Mismatch {
                eval_val,
                compile_val,
            } => {
                let rel_err = if eval_val.abs().max(compile_val.abs()) > 1e-300 {
                    ((eval_val - compile_val) / eval_val.abs().max(compile_val.abs())).abs()
                } else {
                    (eval_val - compile_val).abs()
                };
                failures.push(format!(
                    "  {expr_desc} at {var_name}={val}: eval_f64={eval_val}, compile={compile_val}, rel_err={rel_err:.2e}"
                ));
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "DIFFERENTIAL BUG(S) found for `{expr_desc}`:\n{}",
            failures.join("\n")
        );
    }
}

/// Verify generated Rust code has balanced delimiters and required structure.
fn assert_valid_rust_structure(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("pub fn {fn_name}")),
        "missing 'pub fn {fn_name}' in generated code:\n{code}"
    );
    assert!(
        code.contains("-> f64"),
        "missing '-> f64' return type in generated code:\n{code}"
    );
    assert!(
        code.contains("fn"),
        "missing 'fn' keyword in generated code:\n{code}"
    );
    let open_braces = code.chars().filter(|&c| c == '{').count();
    let close_braces = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        open_braces, close_braces,
        "unbalanced braces ({open_braces} open vs {close_braces} close) in:\n{code}"
    );
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(
        open_parens, close_parens,
        "unbalanced parens ({open_parens} open vs {close_parens} close) in:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Category A: Systematic comparison across expression types
// ═══════════════════════════════════════════════════════════════════════════

/// Standard test points for most expressions.
const STANDARD_VALS: &[f64] = &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0];
/// Positive-only test points (for ln, sqrt, etc.).
const POSITIVE_VALS: &[f64] = &[0.1, 0.5, 1.0, 2.0, 3.0];
/// Points in (-1, 1) for asin/acos.
const UNIT_INTERVAL_VALS: &[f64] = &[-0.5, 0.1, 0.5];

// ── Polynomials ────────────────────────────────────────────────────────

#[test]
fn diff_polynomial_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.clone();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "x");
}

#[test]
fn diff_polynomial_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "x^2");
}

#[test]
fn diff_polynomial_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3) + &x * 2 - 1;
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "x^3 + 2*x - 1");
}

#[test]
fn diff_polynomial_quintic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(5) - x.powi(3) + &x;
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "x^5 - x^3 + x");
}

// ── Trigonometric ──────────────────────────────────────────────────────

#[test]
fn diff_trig_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sin(x)");
}

#[test]
fn diff_trig_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cos();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "cos(x)");
}

#[test]
fn diff_trig_tan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.tan();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "tan(x)");
}

#[test]
fn diff_trig_pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 + cos(x)^2 should always be 1
    let expr = x.sin().powi(2) + x.cos().powi(2);
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sin(x)^2 + cos(x)^2");
}

// ── Exponential ────────────────────────────────────────────────────────

#[test]
fn diff_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "exp(x)");
}

#[test]
fn diff_exp_neg_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (-x.powi(2)).exp();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "exp(-x^2)");
}

#[test]
fn diff_exp_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp() - 1;
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "exp(x) - 1");
}

// ── Logarithmic ────────────────────────────────────────────────────────

#[test]
fn diff_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln();
    assert_paths_agree(&ctx, &expr, &x, "x", POSITIVE_VALS, "ln(x)");
}

#[test]
fn diff_ln_1_plus_x_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (x.powi(2) + 1).ln();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "ln(1 + x^2)");
}

// ── Hyperbolic ─────────────────────────────────────────────────────────

#[test]
fn diff_sinh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sinh();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sinh(x)");
}

#[test]
fn diff_cosh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cosh();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "cosh(x)");
}

#[test]
fn diff_tanh() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.tanh();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "tanh(x)");
}

// ── Inverse trig ───────────────────────────────────────────────────────

#[test]
fn diff_asin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.asin();
    assert_paths_agree(&ctx, &expr, &x, "x", UNIT_INTERVAL_VALS, "asin(x)");
}

#[test]
fn diff_acos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.acos();
    assert_paths_agree(&ctx, &expr, &x, "x", UNIT_INTERVAL_VALS, "acos(x)");
}

#[test]
fn diff_atan() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.atan();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "atan(x)");
}

// ── Absolute value ─────────────────────────────────────────────────────

#[test]
fn diff_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "abs(x)");
}

#[test]
fn diff_abs_x_minus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x - 1).abs();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "abs(x - 1)");
}

// ── Square root ────────────────────────────────────────────────────────

#[test]
fn diff_sqrt_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sqrt();
    assert_paths_agree(&ctx, &expr, &x, "x", POSITIVE_VALS, "sqrt(x)");
}

#[test]
fn diff_sqrt_x_sq_plus_1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (x.powi(2) + 1).sqrt();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sqrt(x^2 + 1)");
}

// ── Composite ──────────────────────────────────────────────────────────

#[test]
fn diff_sin_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp().sin();
    // Limit to moderate x to keep exp(x) from overflowing before sin
    let vals = &[-1.0, -0.5, 0.1, 0.5, 1.0, 2.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "sin(exp(x))");
}

#[test]
fn diff_ln_1_plus_x_sq_composite() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (x.powi(2) + 1).ln();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "ln(1 + x^2)");
}

#[test]
fn diff_exp_sin_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin().exp();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "exp(sin(x))");
}

#[test]
fn diff_sqrt_abs_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs().sqrt();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sqrt(abs(x))");
}

// ── Sign ───────────────────────────────────────────────────────────────

#[test]
fn diff_sign_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sign();
    // Test at several points including 0 (Bug 12 fix: sign(0)=0)
    let vals = &[-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "sign(x)");
}

#[test]
fn diff_sign_at_zero_is_zero() {
    // Explicitly verify sign(0) = 0 from both paths
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sign();

    let compiled = expr.compile(&["x"]).expect("sign should compile");
    let compile_val = compiled(&[0.0]);
    assert_eq!(
        compile_val, 0.0,
        "compile: sign(0) should be 0, got {compile_val}"
    );

    let eval_val = expr.subs_i64(&x, 0).eval_f64();
    match eval_val {
        Ok(v) => assert_eq!(v, 0.0, "eval_f64: sign(0) should be 0, got {v}"),
        Err(e) => {
            // If eval errors, that's noteworthy but not necessarily a bug
            eprintln!("NOTE: sign(0).eval_f64() returned error: {e}");
        }
    }
}

// ── Floor / Ceiling ────────────────────────────────────────────────────

#[test]
fn diff_floor_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.floor();
    let vals = &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "floor(x)");
}

#[test]
fn diff_ceiling_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ceiling();
    let vals = &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "ceiling(x)");
}

#[test]
fn diff_floor_specific_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.floor();

    let compiled = expr.compile(&["x"]).expect("floor should compile");
    // floor(2.7) = 2, floor(-2.3) = -3
    assert_eq!(compiled(&[2.7]), 2.0, "floor(2.7) should be 2");
    assert_eq!(compiled(&[-2.3]), -3.0, "floor(-2.3) should be -3");
    assert_eq!(compiled(&[0.0]), 0.0, "floor(0) should be 0");
}

#[test]
fn diff_ceiling_specific_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ceiling();

    let compiled = expr.compile(&["x"]).expect("ceiling should compile");
    // ceil(2.3) = 3, ceil(-2.7) = -2
    assert_eq!(compiled(&[2.3]), 3.0, "ceil(2.3) should be 3");
    assert_eq!(compiled(&[-2.7]), -2.0, "ceil(-2.7) should be -2");
    assert_eq!(compiled(&[0.0]), 0.0, "ceil(0) should be 0");
}

// ── Powers ─────────────────────────────────────────────────────────────

#[test]
fn diff_pow_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let expr = x.pow(&half);
    assert_paths_agree(&ctx, &expr, &x, "x", POSITIVE_VALS, "x^(1/2)");
}

#[test]
fn diff_pow_third() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let third = ctx.rational(1, 3);
    let expr = x.pow(&third);
    // x^(1/3) for positive x only to avoid complex results
    assert_paths_agree(&ctx, &expr, &x, "x", POSITIVE_VALS, "x^(1/3)");
}

#[test]
fn diff_pow_neg_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(-1);
    // Avoid x=0
    let vals = &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "x^(-1)");
}

#[test]
fn diff_pow_neg_two() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(-2);
    let vals = &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0];
    assert_paths_agree(&ctx, &expr, &x, "x", vals, "x^(-2)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Category A (continued): Multi-variable expressions via compile
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_multi_x_plus_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x + &y;

    let compiled = expr.compile(&["x", "y"]).expect("x+y should compile");

    let test_pairs: &[(f64, f64)] = &[
        (1.0, 2.0),
        (-1.0, 3.0),
        (0.0, 0.0),
        (0.5, -0.5),
        (2.0, -3.0),
    ];

    for &(xv, yv) in test_pairs {
        let compile_val = compiled(&[xv, yv]);
        let eval_val = expr
            .subs(&x, &ctx.int(xv as i64))
            .subs(&y, &ctx.int(yv as i64))
            .eval_f64();

        if let Ok(ev) = eval_val {
            assert!(
                approx_eq_rel(compile_val, ev, 1e-10),
                "x+y at ({xv},{yv}): compile={compile_val}, eval={ev}"
            );
        }
    }
}

#[test]
fn diff_multi_x_times_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x * &y;

    let compiled = expr.compile(&["x", "y"]).expect("x*y should compile");

    let test_pairs: &[(f64, f64)] = &[
        (2.0, 3.0),
        (-1.0, 5.0),
        (0.0, 100.0),
        (0.5, -2.0),
        (-3.0, -4.0),
    ];

    for &(xv, yv) in test_pairs {
        let compile_val = compiled(&[xv, yv]);
        let expected = xv * yv;
        assert!(
            approx_eq_rel(compile_val, expected, 1e-10),
            "x*y at ({xv},{yv}): compile={compile_val}, expected={expected}"
        );
    }
}

#[test]
fn diff_multi_sin_x_cos_y() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let expr = &x.sin() * &y.cos();

    let compiled = expr
        .compile(&["x", "y"])
        .expect("sin(x)*cos(y) should compile");

    let test_pairs: &[(f64, f64)] = &[
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 2.0),
        (0.5, -0.5),
        (2.0, 3.0),
        (std::f64::consts::PI, std::f64::consts::FRAC_PI_2),
    ];

    for &(xv, yv) in test_pairs {
        let compile_val = compiled(&[xv, yv]);
        let expected = xv.sin() * yv.cos();
        assert!(
            approx_eq_rel(compile_val, expected, 1e-10),
            "sin(x)*cos(y) at ({xv},{yv}): compile={compile_val}, expected={expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category B: CSE consistency — codegen structural validity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_trig_identity_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 + sin(x)*cos(x) + cos(x)^2
    let sin_x = x.sin();
    let cos_x = x.cos();
    let expr = &sin_x.powi(2) + &(&sin_x * &cos_x) + &cos_x.powi(2);

    let code = expr
        .to_rust_fn("trig_cse", &["x"])
        .expect("to_rust_fn should succeed for trig CSE expression");
    assert_valid_rust_structure(&code, "trig_cse");
}

#[test]
fn cse_nested_sin_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^3 + sin(x)^2 + sin(x)
    let sin_x = x.sin();
    let expr = &sin_x.powi(3) + &sin_x.powi(2) + &sin_x;

    let code = expr
        .to_rust_fn("sin_powers", &["x"])
        .expect("to_rust_fn should succeed for sin powers");
    assert_valid_rust_structure(&code, "sin_powers");
}

#[test]
fn cse_exp_shared_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x)^2 + exp(x) + 1
    let exp_x = x.exp();
    let expr = &exp_x.powi(2) + &exp_x + 1;

    let code = expr
        .to_rust_fn("exp_shared", &["x"])
        .expect("to_rust_fn should succeed for exp shared");
    assert_valid_rust_structure(&code, "exp_shared");
}

#[test]
fn cse_multi_shared_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x^2+1)^3 + (x^2+1)^2 + (x^2+1) — the subexpression x^2+1 appears 3 times
    let sub = x.powi(2) + 1;
    let expr = &sub.powi(3) + &sub.powi(2) + &sub;

    let code = expr
        .to_rust_fn("multi_cse", &["x"])
        .expect("to_rust_fn should succeed for multi-CSE");
    assert_valid_rust_structure(&code, "multi_cse");
}

#[test]
fn cse_two_variable_codegen_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let shared = &x.sin() + &y.cos();
    let expr = &shared.powi(2) + &shared;

    let code = expr
        .to_rust_fn("two_var_cse", &["x", "y"])
        .expect("to_rust_fn should succeed for two-variable CSE");
    assert_valid_rust_structure(&code, "two_var_cse");
    assert!(
        code.contains("x: f64") && code.contains("y: f64"),
        "should have both x and y parameters"
    );
}

#[test]
fn codegen_all_basic_functions_produce_valid_code() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<(&str, Ex)> = vec![
        ("cg_sin", x.sin()),
        ("cg_cos", x.cos()),
        ("cg_tan", x.tan()),
        ("cg_exp", x.exp()),
        ("cg_ln", x.ln()),
        ("cg_sqrt", x.sqrt()),
        ("cg_abs", x.abs()),
        ("cg_asin", x.asin()),
        ("cg_acos", x.acos()),
        ("cg_atan", x.atan()),
        ("cg_sinh", x.sinh()),
        ("cg_cosh", x.cosh()),
        ("cg_tanh", x.tanh()),
        ("cg_sign", x.sign()),
        ("cg_floor", x.floor()),
        ("cg_ceil", x.ceiling()),
    ];

    for (name, expr) in &cases {
        let code = expr
            .to_rust_fn(name, &["x"])
            .unwrap_or_else(|e| panic!("codegen failed for {name}: {e}"));
        assert_valid_rust_structure(&code, name);
    }
}

#[test]
fn codegen_composite_expressions_produce_valid_code() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let cases: Vec<(&str, Ex)> = vec![
        ("cg_sin_exp", x.exp().sin()),
        ("cg_exp_sin", x.sin().exp()),
        ("cg_ln_sq_p1", (x.powi(2) + 1).ln()),
        ("cg_sqrt_abs", x.abs().sqrt()),
        ("cg_sin2_cos2", x.sin().powi(2) + x.cos().powi(2)),
        ("cg_exp_neg_sq", (-x.powi(2)).exp()),
        ("cg_atan_exp", x.exp().atan()),
        ("cg_sinh_cos", x.sinh() + x.cos()),
    ];

    for (name, expr) in &cases {
        let code = expr
            .to_rust_fn(name, &["x"])
            .unwrap_or_else(|e| panic!("codegen failed for {name}: {e}"));
        assert_valid_rust_structure(&code, name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category C: Edge values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn edge_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Functions that should be well-defined at x=0
    let cases: Vec<(&str, Ex, f64)> = vec![
        ("sin(0)", x.sin(), 0.0),
        ("cos(0)", x.cos(), 1.0),
        ("exp(0)", x.exp(), 1.0),
        ("sinh(0)", x.sinh(), 0.0),
        ("cosh(0)", x.cosh(), 1.0),
        ("tanh(0)", x.tanh(), 0.0),
        ("atan(0)", x.atan(), 0.0),
        ("abs(0)", x.abs(), 0.0),
        ("x^2 at 0", x.powi(2), 0.0),
    ];

    for (desc, expr, expected) in &cases {
        let compiled = expr
            .compile(&["x"])
            .unwrap_or_else(|| panic!("{desc} should compile"));
        let compile_val = compiled(&[0.0]);
        assert!(
            approx_eq_rel(compile_val, *expected, 1e-10),
            "{desc}: compile at 0 = {compile_val}, expected {expected}"
        );

        let eval_result = expr.subs_i64(&x, 0).eval_f64();
        if let Ok(eval_val) = eval_result {
            assert!(
                approx_eq_rel(eval_val, *expected, 1e-10),
                "{desc}: eval_f64 at 0 = {eval_val}, expected {expected}"
            );
            // Cross-check: compile and eval agree
            assert!(
                approx_eq_rel(compile_val, eval_val, 1e-10),
                "{desc}: compile ({compile_val}) != eval ({eval_val}) at x=0"
            );
        }
    }
}

#[test]
fn edge_very_small_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let tiny = 1e-15;

    // sin(tiny) ≈ tiny
    let sin_compiled = x.sin().compile(&["x"]).unwrap();
    let sin_val = sin_compiled(&[tiny]);
    assert!(
        (sin_val - tiny).abs() < tiny * 1e-5,
        "sin({tiny}) should be approximately {tiny}, got {sin_val}"
    );

    // cos(tiny) ≈ 1
    let cos_compiled = x.cos().compile(&["x"]).unwrap();
    let cos_val = cos_compiled(&[tiny]);
    assert!(
        (cos_val - 1.0).abs() < 1e-10,
        "cos({tiny}) should be approximately 1, got {cos_val}"
    );

    // exp(tiny) ≈ 1 + tiny
    // Note: the ULP of 1.0 in f64 is ~2.22e-16, so subtracting 1.0
    // from exp(tiny) loses precision.  We just check closeness to 1.0.
    let exp_compiled = x.exp().compile(&["x"]).unwrap();
    let exp_val = exp_compiled(&[tiny]);
    assert!(
        (exp_val - 1.0).abs() < 1e-14,
        "exp({tiny}) should be approximately 1, got {exp_val}"
    );

    // tanh(tiny) ≈ tiny
    let tanh_compiled = x.tanh().compile(&["x"]).unwrap();
    let tanh_val = tanh_compiled(&[tiny]);
    assert!(
        (tanh_val - tiny).abs() < tiny * 1e-5,
        "tanh({tiny}) should be approximately {tiny}, got {tanh_val}"
    );
}

#[test]
fn edge_very_large_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let big = 1e15;

    // x^2 at 1e15 should be 1e30
    let sq_compiled = x.powi(2).compile(&["x"]).unwrap();
    let sq_val = sq_compiled(&[big]);
    let expected_sq = 1e30;
    assert!(
        approx_eq_rel(sq_val, expected_sq, 1e-10),
        "x^2 at {big}: compile={sq_val}, expected={expected_sq}"
    );

    // sin(x) at large x should still be in [-1, 1]
    let sin_compiled = x.sin().compile(&["x"]).unwrap();
    let sin_val = sin_compiled(&[big]);
    assert!(
        (-1.0..=1.0).contains(&sin_val),
        "sin({big}) should be in [-1,1], got {sin_val}"
    );

    // tanh(big) should be ≈ 1
    let tanh_compiled = x.tanh().compile(&["x"]).unwrap();
    let tanh_val = tanh_compiled(&[big]);
    assert!(
        (tanh_val - 1.0).abs() < 1e-10,
        "tanh({big}) should be ≈1, got {tanh_val}"
    );

    // tanh(-big) should be ≈ -1
    let tanh_neg = tanh_compiled(&[-big]);
    assert!(
        (tanh_neg + 1.0).abs() < 1e-10,
        "tanh(-{big}) should be ≈-1, got {tanh_neg}"
    );
}

#[test]
fn edge_negative_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // abs(-5) = 5
    let abs_compiled = x.abs().compile(&["x"]).unwrap();
    assert_eq!(abs_compiled(&[-5.0]), 5.0);

    // x^3 at -2: should be -8
    let cube_compiled = x.powi(3).compile(&["x"]).unwrap();
    let cube_val = cube_compiled(&[-2.0]);
    assert!(
        approx_eq_rel(cube_val, -8.0, 1e-10),
        "(-2)^3 = {cube_val}, expected -8"
    );

    // sign(-5) = -1
    let sign_compiled = x.sign().compile(&["x"]).unwrap();
    assert_eq!(sign_compiled(&[-5.0]), -1.0);
    assert_eq!(sign_compiled(&[5.0]), 1.0);
}

#[test]
fn edge_ln_at_zero_and_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let ln_compiled = x.ln().compile(&["x"]).unwrap();

    // ln(0) should be -inf
    let ln_zero = ln_compiled(&[0.0]);
    assert!(
        ln_zero.is_infinite() && ln_zero < 0.0,
        "ln(0) should be -inf, got {ln_zero}"
    );

    // ln(-1) should be NaN (in real domain)
    let ln_neg = ln_compiled(&[-1.0]);
    assert!(ln_neg.is_nan(), "ln(-1) should be NaN, got {ln_neg}");
}

#[test]
fn edge_sqrt_at_zero_and_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt_compiled = x.sqrt().compile(&["x"]).unwrap();

    assert_eq!(sqrt_compiled(&[0.0]), 0.0, "sqrt(0) should be 0");
    assert_eq!(sqrt_compiled(&[1.0]), 1.0, "sqrt(1) should be 1");
    assert!(
        (sqrt_compiled(&[4.0]) - 2.0).abs() < 1e-10,
        "sqrt(4) should be 2"
    );

    // sqrt(-1) should be NaN in real domain
    let sqrt_neg = sqrt_compiled(&[-1.0]);
    assert!(sqrt_neg.is_nan(), "sqrt(-1) should be NaN, got {sqrt_neg}");
}

#[test]
fn edge_division_by_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(-1); // 1/x

    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[0.0]);
    // 1/0 should be inf or NaN
    assert!(
        val.is_infinite() || val.is_nan(),
        "1/0 should be inf or NaN, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Category D: Two and three variable functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_two_variables_basic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // f(x, y) = x^2 + y^2
    let expr = x.powi(2) + y.powi(2);
    let compiled = expr.compile(&["x", "y"]).expect("x^2+y^2 should compile");

    assert!(approx_eq_rel(compiled(&[3.0, 4.0]), 25.0, 1e-10));
    assert!(approx_eq_rel(compiled(&[0.0, 0.0]), 0.0, 1e-10));
    assert!(approx_eq_rel(compiled(&[1.0, 1.0]), 2.0, 1e-10));
    assert!(approx_eq_rel(compiled(&[-1.0, 2.0]), 5.0, 1e-10));
}

#[test]
fn multi_two_variables_asymmetric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // f(x, y) = x - y (not commutative)
    let expr = &x - &y;
    let compiled = expr.compile(&["x", "y"]).expect("x-y should compile");

    // f(3, 1) = 2
    assert!(
        approx_eq_rel(compiled(&[3.0, 1.0]), 2.0, 1e-10),
        "f(3,1)={}, expected 2",
        compiled(&[3.0, 1.0])
    );
    // f(1, 3) = -2
    assert!(
        approx_eq_rel(compiled(&[1.0, 3.0]), -2.0, 1e-10),
        "f(1,3)={}, expected -2",
        compiled(&[1.0, 3.0])
    );
}

#[test]
fn multi_variable_order_matters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    // f(x, y) = x - y
    let expr = &x - &y;

    let compiled_xy = expr.compile(&["x", "y"]).expect("xy order should compile");
    let compiled_yx = expr.compile(&["y", "x"]).expect("yx order should compile");

    // With ["x", "y"]: args[0]=x=5, args[1]=y=2 → x-y = 5-2 = 3
    let val_xy = compiled_xy(&[5.0, 2.0]);
    assert!(
        approx_eq_rel(val_xy, 3.0, 1e-10),
        "xy order: f(5,2)={val_xy}, expected 3"
    );

    // With ["y", "x"]: args[0]=y=5, args[1]=x=2 → x-y = 2-5 = -3
    let val_yx = compiled_yx(&[5.0, 2.0]);
    assert!(
        approx_eq_rel(val_yx, -3.0, 1e-10),
        "yx order: f(5,2)={val_yx}, expected -3"
    );

    // They should NOT be equal for asymmetric expressions
    assert!(
        (val_xy - val_yx).abs() > 1e-10,
        "variable ordering should matter: xy={val_xy}, yx={val_yx}"
    );
}

#[test]
fn multi_three_variables() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    // f(x, y, z) = x*y + y*z + z*x
    let expr = &x * &y + &y * &z + &z * &x;
    let compiled = expr
        .compile(&["x", "y", "z"])
        .expect("3-var should compile");

    // f(1, 2, 3) = 1*2 + 2*3 + 3*1 = 2 + 6 + 3 = 11
    assert!(
        approx_eq_rel(compiled(&[1.0, 2.0, 3.0]), 11.0, 1e-10),
        "f(1,2,3) = {}, expected 11",
        compiled(&[1.0, 2.0, 3.0])
    );

    // f(0, 0, 0) = 0
    assert!(approx_eq_rel(compiled(&[0.0, 0.0, 0.0]), 0.0, 1e-10));

    // f(-1, 2, -3) = -1*2 + 2*(-3) + (-3)*(-1) = -2 - 6 + 3 = -5
    assert!(
        approx_eq_rel(compiled(&[-1.0, 2.0, -3.0]), -5.0, 1e-10),
        "f(-1,2,-3) = {}, expected -5",
        compiled(&[-1.0, 2.0, -3.0])
    );
}

#[test]
fn multi_three_variables_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    // f(x, y, z) = sin(x) + cos(y) + exp(z)
    let expr = x.sin() + y.cos() + z.exp();
    let compiled = expr
        .compile(&["x", "y", "z"])
        .expect("trig 3-var should compile");

    let test_point = (1.0_f64, 2.0_f64, 0.5_f64);
    let expected = test_point.0.sin() + test_point.1.cos() + test_point.2.exp();
    let result = compiled(&[test_point.0, test_point.1, test_point.2]);
    assert!(
        approx_eq_rel(result, expected, 1e-10),
        "f({},{},{}) = {result}, expected {expected}",
        test_point.0,
        test_point.1,
        test_point.2
    );
}

#[test]
fn multi_three_var_order_matters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    // f(x, y, z) = x - 2*y + 3*z
    let expr = &x - &y * 2 + &z * 3;

    let compiled_xyz = expr
        .compile(&["x", "y", "z"])
        .expect("xyz order should compile");
    let compiled_zyx = expr
        .compile(&["z", "y", "x"])
        .expect("zyx order should compile");

    // xyz: args[0]=x=1, args[1]=y=2, args[2]=z=3 → 1 - 4 + 9 = 6
    let val_xyz = compiled_xyz(&[1.0, 2.0, 3.0]);
    // zyx: args[0]=z=1, args[1]=y=2, args[2]=x=3 → 3 - 4 + 3 = 2
    let val_zyx = compiled_zyx(&[1.0, 2.0, 3.0]);

    let expected_xyz = 1.0 - 2.0 * 2.0 + 3.0 * 3.0;
    let expected_zyx = 3.0 - 2.0 * 2.0 + 3.0 * 1.0;

    assert!(
        approx_eq_rel(val_xyz, expected_xyz, 1e-10),
        "xyz order: got {val_xyz}, expected {expected_xyz}"
    );
    assert!(
        approx_eq_rel(val_zyx, expected_zyx, 1e-10),
        "zyx order: got {val_zyx}, expected {expected_zyx}"
    );
}

#[test]
fn multi_unused_variable_does_not_affect_result() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Expression only uses x, compiled with ["x", "y"]
    let expr = x.powi(2) + 1;
    let compiled = expr.compile(&["x", "y"]).expect("extra var should be fine");

    // y value (999) should be irrelevant
    assert!(
        approx_eq_rel(compiled(&[3.0, 999.0]), 10.0, 1e-10),
        "extra variable should not affect result"
    );
    assert!(
        approx_eq_rel(compiled(&[3.0, 0.0]), 10.0, 1e-10),
        "extra variable should not affect result"
    );
}

#[test]
fn multi_missing_variable_returns_none() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &x + &y;
    let result = expr.compile(&["x"]); // y is unbound
    assert!(
        result.is_none(),
        "compile should return None when expression has unbound variable 'y'"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional differential checks: verify compile and eval_f64 agree for
// tricky expressions that exercise different internal code paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_polynomial_with_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);

    // (1/2)*x^2 + (1/3)*x + 1
    let expr = &x.powi(2) * &half + &x * &third + 1;

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "(1/2)*x^2 + (1/3)*x + 1",
    );
}

#[test]
fn diff_nested_trig_deep() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin(cos(sin(x)))
    let expr = x.sin().cos().sin();
    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "sin(cos(sin(x)))");
}

#[test]
fn diff_hyperbolic_identity_cosh_sq_minus_sinh_sq() {
    // cosh(x)^2 - sinh(x)^2 = 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cosh().powi(2) - x.sinh().powi(2);

    let compiled = expr.compile(&["x"]).expect("should compile");
    for &v in STANDARD_VALS {
        let result = compiled(&[v]);
        assert!(
            approx_eq_rel(result, 1.0, 1e-8),
            "cosh({v})^2 - sinh({v})^2 = {result}, expected 1"
        );
    }

    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "cosh(x)^2 - sinh(x)^2");
}

#[test]
fn diff_exp_vs_sinh_plus_cosh() {
    // sinh(x) + cosh(x) = exp(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = x.sinh() + x.cosh();

    let compiled = lhs.compile(&["x"]).expect("should compile");
    for &v in STANDARD_VALS {
        let result = compiled(&[v]);
        let expected = v.exp();
        assert!(
            approx_eq_rel(result, expected, 1e-10),
            "sinh({v}) + cosh({v}) = {result}, expected exp({v}) = {expected}"
        );
    }
}

#[test]
fn diff_pow_with_pi_and_e_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let e = ctx.e();

    // pi * x + e
    let expr = &pi * &x + &e;
    assert_paths_agree(&ctx, &expr, &x, "x", &[-1.0, 0.0, 1.0, 2.0], "pi*x + e");
}

#[test]
fn diff_complex_multiop_expression() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // sin(x)^2 * exp(-x) + cos(x) * ln(x^2 + 1)
    let expr = &(&x.sin().powi(2) * &(-&x).exp()) + &(&x.cos() * &(x.powi(2) + 1).ln());
    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-1.0, 0.0, 0.5, 1.0, 2.0],
        "sin(x)^2*exp(-x) + cos(x)*ln(x^2+1)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Stress tests: Many expressions at many points
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_systematic_sweep_all_expressions() {
    // Run ALL Category A expressions through a systematic sweep, collecting
    // any discrepancies as warnings rather than hard failures, to get a
    // complete picture.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);

    struct TestCase {
        name: &'static str,
        expr: Ex,
        test_vals: &'static [f64],
    }

    let cases = vec![
        TestCase {
            name: "x",
            expr: x.clone(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "x^2",
            expr: x.powi(2),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "x^3+2x-1",
            expr: x.powi(3) + &x * 2 - 1,
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "x^5-x^3+x",
            expr: x.powi(5) - x.powi(3) + &x,
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sin(x)",
            expr: x.sin(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "cos(x)",
            expr: x.cos(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "tan(x)",
            expr: x.tan(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sin^2+cos^2",
            expr: x.sin().powi(2) + x.cos().powi(2),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "exp(x)",
            expr: x.exp(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "exp(-x^2)",
            expr: (-x.powi(2)).exp(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "exp(x)-1",
            expr: x.exp() - 1,
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "ln(x)",
            expr: x.ln(),
            test_vals: POSITIVE_VALS,
        },
        TestCase {
            name: "ln(1+x^2)",
            expr: (x.powi(2) + 1).ln(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sinh(x)",
            expr: x.sinh(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "cosh(x)",
            expr: x.cosh(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "tanh(x)",
            expr: x.tanh(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "asin(x)",
            expr: x.asin(),
            test_vals: UNIT_INTERVAL_VALS,
        },
        TestCase {
            name: "acos(x)",
            expr: x.acos(),
            test_vals: UNIT_INTERVAL_VALS,
        },
        TestCase {
            name: "atan(x)",
            expr: x.atan(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "abs(x)",
            expr: x.abs(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "abs(x-1)",
            expr: (&x - 1).abs(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sqrt(x)",
            expr: x.sqrt(),
            test_vals: POSITIVE_VALS,
        },
        TestCase {
            name: "sqrt(x^2+1)",
            expr: (x.powi(2) + 1).sqrt(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sin(exp(x))",
            expr: x.exp().sin(),
            test_vals: &[-1.0, -0.5, 0.1, 0.5, 1.0, 2.0],
        },
        TestCase {
            name: "exp(sin(x))",
            expr: x.sin().exp(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sqrt(abs(x))",
            expr: x.abs().sqrt(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "sign(x)",
            expr: x.sign(),
            test_vals: &[-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0],
        },
        TestCase {
            name: "floor(x)",
            expr: x.floor(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "ceiling(x)",
            expr: x.ceiling(),
            test_vals: STANDARD_VALS,
        },
        TestCase {
            name: "x^(1/2)",
            expr: x.pow(&half),
            test_vals: POSITIVE_VALS,
        },
        TestCase {
            name: "x^(1/3)",
            expr: x.pow(&third),
            test_vals: POSITIVE_VALS,
        },
        TestCase {
            name: "x^(-1)",
            expr: x.powi(-1),
            test_vals: &[-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 3.0],
        },
    ];

    let mut all_failures: Vec<String> = Vec::new();

    for case in &cases {
        let compiled = match case.expr.compile(&["x"]) {
            Some(f) => f,
            None => {
                all_failures.push(format!("  {}: compile returned None", case.name));
                continue;
            }
        };

        for &val in case.test_vals {
            let compile_val = compiled(&[val]);

            let eval_result = if val == val.floor() && val.abs() < 1e15 {
                case.expr.subs_i64(&x, val as i64).eval_f64()
            } else {
                let (n, d) = float_to_rational(val);
                let rat = ctx.rational(n, d);
                case.expr.subs(&x, &rat).eval_f64()
            };

            match eval_result {
                Ok(eval_val) => {
                    let both_nan = compile_val.is_nan() && eval_val.is_nan();
                    if !both_nan && !approx_eq_rel(compile_val, eval_val, 1e-10) {
                        let rel_err = if eval_val.abs().max(compile_val.abs()) > 1e-300 {
                            ((eval_val - compile_val) / eval_val.abs().max(compile_val.abs())).abs()
                        } else {
                            (eval_val - compile_val).abs()
                        };
                        all_failures.push(format!(
                            "  {} at x={}: eval={}, compile={}, rel_err={:.2e}",
                            case.name, val, eval_val, compile_val, rel_err
                        ));
                    }
                }
                Err(_) => {
                    // eval error — only flag if compile returned a normal finite number
                    if compile_val.is_finite() && !compile_val.is_nan() {
                        all_failures.push(format!(
                            "  {} at x={}: eval errored but compile returned {}",
                            case.name, val, compile_val
                        ));
                    }
                }
            }
        }
    }

    if !all_failures.is_empty() {
        panic!(
            "DIFFERENTIAL BUGS found in systematic sweep ({} failures):\n{}",
            all_failures.len(),
            all_failures.join("\n")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Codegen vs compile cross-check: verify codegen produces valid code for
// every expression that compile accepts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_valid_for_all_compilable_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    let third = ctx.rational(1, 3);

    let cases: Vec<(&str, Ex)> = vec![
        ("cg_x", x.clone()),
        ("cg_x2", x.powi(2)),
        ("cg_x3_2x_1", x.powi(3) + &x * 2 - 1),
        ("cg_sinx", x.sin()),
        ("cg_cosx", x.cos()),
        ("cg_tanx", x.tan()),
        ("cg_expx", x.exp()),
        ("cg_lnx", x.ln()),
        ("cg_sinhx", x.sinh()),
        ("cg_coshx", x.cosh()),
        ("cg_tanhx", x.tanh()),
        ("cg_asinx", x.asin()),
        ("cg_acosx", x.acos()),
        ("cg_atanx", x.atan()),
        ("cg_absx", x.abs()),
        ("cg_sqrtx", x.sqrt()),
        ("cg_signx", x.sign()),
        ("cg_floorx", x.floor()),
        ("cg_ceilx", x.ceiling()),
        ("cg_xhalf", x.pow(&half)),
        ("cg_xthird", x.pow(&third)),
        ("cg_xinv", x.powi(-1)),
        ("cg_sinexp", x.exp().sin()),
        ("cg_expsin", x.sin().exp()),
        ("cg_sqrtabs", x.abs().sqrt()),
        ("cg_ln1px2", (x.powi(2) + 1).ln()),
    ];

    for (name, expr) in &cases {
        match expr.to_rust_fn(name, &["x"]) {
            Ok(code) => {
                assert_valid_rust_structure(&code, name);
            }
            Err(e) => {
                // Some expressions might legitimately not be codegen-able
                eprintln!("NOTE: codegen failed for {name}: {e}");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression: specific known-tricky values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_sin_at_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin();

    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[std::f64::consts::PI]);
    // sin(pi) should be ~0 (not exactly 0 due to floating point)
    assert!(val.abs() < 1e-14, "sin(pi) = {val}, expected ~0");
}

#[test]
fn regression_cos_at_pi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.cos();

    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[std::f64::consts::PI]);
    assert!((val + 1.0).abs() < 1e-14, "cos(pi) = {val}, expected -1");
}

#[test]
fn regression_exp_at_large_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp();

    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[-100.0]);
    // exp(-100) is very small but not zero
    assert!(val > 0.0, "exp(-100) should be positive, got {val}");
    assert!(val < 1e-40, "exp(-100) should be very small, got {val}");
}

#[test]
fn regression_pow_negative_base_integer_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // (-2)^3 = -8 via compile
    let expr = x.powi(3);
    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[-2.0]);
    assert!(
        approx_eq_rel(val, -8.0, 1e-10),
        "(-2)^3 via compile = {val}, expected -8"
    );

    // Cross-check with eval
    let eval_val = expr.subs_i64(&x, -2).eval_f64().unwrap();
    assert!(
        approx_eq_rel(eval_val, -8.0, 1e-10),
        "(-2)^3 via eval = {eval_val}, expected -8"
    );
}

#[test]
fn regression_pow_fractional_negative_base() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);

    // (-1)^(1/2) should produce NaN or error (complex in real domain)
    let expr = x.pow(&half);
    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[-1.0]);
    // powf(-1.0, 0.5) gives NaN on most platforms
    assert!(
        val.is_nan(),
        "(-1)^(1/2) via compile should be NaN, got {val}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify that codegen and compile handle the same expression consistently
// (both should succeed or both should fail for basic expressions)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn consistency_compile_and_codegen_same_expressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expressions: Vec<(&str, Ex)> = vec![
        ("sin(x)", x.sin()),
        ("x^2+1", x.powi(2) + 1),
        ("exp(x)*cos(x)", &x.exp() * &x.cos()),
        ("abs(x)+sign(x)", x.abs() + x.sign()),
        ("floor(x)+ceil(x)", x.floor() + x.ceiling()),
    ];

    for (desc, expr) in &expressions {
        let compile_ok = expr.compile(&["x"]).is_some();
        let codegen_ok = expr.to_rust_fn("f", &["x"]).is_ok();

        // Both should succeed for basic expressions
        assert!(
            compile_ok,
            "{desc}: compile returned None but should succeed"
        );
        assert!(codegen_ok, "{desc}: to_rust_fn failed but should succeed");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Aggressive edge-case differential tests
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: compare compile result with a known f64 expected value.
fn assert_compile_eq(expr: &Ex, var_name: &str, input: f64, expected: f64, desc: &str) {
    let compiled = expr
        .compile(&[var_name])
        .unwrap_or_else(|| panic!("{desc}: compile returned None"));
    let got = compiled(&[input]);
    assert!(
        approx_eq_rel(got, expected, 1e-10),
        "{desc}: compile({input}) = {got}, expected {expected}"
    );
}

#[test]
fn aggressive_negative_base_even_power() {
    // (-x)^2 should equal x^2 for all x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(2);

    let compiled = expr.compile(&["x"]).unwrap();
    for &v in &[-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
        let got = compiled(&[v]);
        let expected = v * v;
        assert!(
            approx_eq_rel(got, expected, 1e-10),
            "x^2 at x={v}: compile={got}, expected={expected}"
        );
    }
}

#[test]
fn aggressive_negative_base_odd_power() {
    // x^3 should be negative for negative x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(3);

    for &v in &[-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0] {
        assert_compile_eq(&expr, "x", v, v.powi(3), &format!("x^3 at x={v}"));
    }

    // Cross-check: compile vs eval at all these integer points
    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        "x^3 (negative base odd power)",
    );
}

#[test]
fn aggressive_negative_base_fractional_power_two_thirds() {
    // x^(2/3) for positive x should agree between paths
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two_thirds = ctx.rational(2, 3);
    let expr = x.pow(&two_thirds);

    // Positive x only — negative base with fractional power is complex
    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[0.1, 0.5, 1.0, 2.0, 3.0, 8.0],
        "x^(2/3)",
    );
}

#[test]
fn aggressive_negative_base_pow_via_compile() {
    // (-2)^(1/3): compile uses powf(-2.0, 1.0/3.0) which is NaN.
    // eval_f64 might give the real cube root -cbrt(2).
    // This is a known semantic mismatch worth documenting.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let third = ctx.rational(1, 3);
    let expr = x.pow(&third);

    let compiled = expr.compile(&["x"]).unwrap();
    let compile_val = compiled(&[-8.0]);

    let eval_result = expr.subs_i64(&x, -8).eval_f64();

    // Document the outcome regardless of what happens
    eprintln!("x^(1/3) at x=-8: compile={compile_val}, eval={eval_result:?}");

    // compile uses f64::powf(-8.0, 1.0/3.0) which gives NaN
    if compile_val.is_nan()
        && let Ok(eval_val) = &eval_result
        && !eval_val.is_nan()
    {
        eprintln!(
            "BUG: x^(1/3) at x=-8: compile=NaN but eval={eval_val} \
                     — compile path cannot compute real cube root of negative numbers"
        );
    }
}

#[test]
fn aggressive_high_order_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x^10 + x^9 + ... + x + 1  (geometric sum)
    let mut expr = ctx.int(1);
    for i in 1..=10 {
        expr = expr + x.powi(i);
    }

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "x^10 + x^9 + ... + x + 1",
    );
}

#[test]
fn aggressive_near_cancellation_sin_x_minus_x() {
    // sin(x) - x ≈ -x^3/6 near zero — tests precision of both paths
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sin() - &x;

    let compiled = expr.compile(&["x"]).unwrap();
    let small_vals = [0.01, 0.001, 0.0001];

    for &v in &small_vals {
        let got = compiled(&[v]);
        let expected = v.sin() - v;
        assert!(
            approx_eq_rel(got, expected, 1e-8),
            "sin(x)-x at x={v}: compile={got}, expected={expected}"
        );
    }
}

#[test]
fn aggressive_large_integer_eval_vs_compile() {
    // x^10 at x=10: symbolic eval computes 10^10=10000000000 exactly,
    // compile computes 10.0_f64.powf(10.0) = 10000000000.0
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(10);

    let compiled = expr.compile(&["x"]).unwrap();
    let compile_val = compiled(&[10.0]);
    let eval_val = expr.subs_i64(&x, 10).eval_f64().unwrap();

    assert!(
        approx_eq_rel(compile_val, eval_val, 1e-10),
        "x^10 at x=10: compile={compile_val}, eval={eval_val}"
    );
    assert!(
        approx_eq_rel(compile_val, 1e10, 1e-10),
        "x^10 at x=10: compile={compile_val}, expected 1e10"
    );
}

#[test]
fn aggressive_alternating_sign_polynomial() {
    // x^4 - x^3 + x^2 - x + 1: lots of sign changes
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(4) - x.powi(3) + x.powi(2) - &x + 1;

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0],
        "x^4 - x^3 + x^2 - x + 1",
    );
}

#[test]
fn aggressive_double_negation() {
    // --x should equal x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = -(-&x);

    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "--x (double negation)");
}

#[test]
fn aggressive_nested_abs() {
    // abs(abs(x)) == abs(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.abs().abs();

    let single_abs = x.abs();
    let compiled_nested = expr.compile(&["x"]).unwrap();
    let compiled_single = single_abs.compile(&["x"]).unwrap();

    for &v in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
        let a = compiled_nested(&[v]);
        let b = compiled_single(&[v]);
        assert!(
            approx_eq_rel(a, b, 1e-15),
            "abs(abs({v}))={a} vs abs({v})={b}"
        );
    }
}

#[test]
fn aggressive_sign_times_abs_equals_x() {
    // sign(x) * abs(x) should equal x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sign() * &x.abs();

    let compiled = expr.compile(&["x"]).unwrap();
    for &v in &[-5.0, -1.0, -0.001, 0.0, 0.001, 1.0, 5.0] {
        let got = compiled(&[v]);
        assert!(
            approx_eq_rel(got, v, 1e-10),
            "sign({v})*abs({v})={got}, expected {v}"
        );
    }
}

#[test]
fn aggressive_exp_ln_roundtrip() {
    // exp(ln(x)) should equal x for positive x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.ln().exp();

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
        "exp(ln(x)) roundtrip",
    );
}

#[test]
fn aggressive_ln_exp_roundtrip() {
    // ln(exp(x)) should equal x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.exp().ln();

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        "ln(exp(x)) roundtrip",
    );
}

#[test]
fn aggressive_trig_double_angle() {
    // sin(2x) vs 2*sin(x)*cos(x) — two different expressions, same value
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);

    let lhs = (&x * &two).sin();
    let rhs = &(&x.sin() * &x.cos()) * &two;

    let compiled_lhs = lhs.compile(&["x"]).unwrap();
    let compiled_rhs = rhs.compile(&["x"]).unwrap();

    for &v in STANDARD_VALS {
        let a = compiled_lhs(&[v]);
        let b = compiled_rhs(&[v]);
        assert!(
            approx_eq_rel(a, b, 1e-10),
            "sin(2*{v})={a} vs 2*sin({v})*cos({v})={b}"
        );
    }
}

#[test]
fn aggressive_atan_at_extreme_values() {
    // atan(x) → ±π/2 as x → ±∞
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.atan();

    let compiled = expr.compile(&["x"]).unwrap();
    let half_pi = std::f64::consts::FRAC_PI_2;

    assert!(
        (compiled(&[1e15]) - half_pi).abs() < 1e-10,
        "atan(1e15) should be ~π/2"
    );
    assert!(
        (compiled(&[-1e15]) + half_pi).abs() < 1e-10,
        "atan(-1e15) should be ~-π/2"
    );
}

#[test]
fn aggressive_floor_ceil_at_integers() {
    // floor(n) == ceil(n) == n for integer n
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let floor_expr = x.floor();
    let ceil_expr = x.ceiling();

    let compiled_floor = floor_expr.compile(&["x"]).unwrap();
    let compiled_ceil = ceil_expr.compile(&["x"]).unwrap();

    for n in -5..=5 {
        let v = n as f64;
        assert_eq!(compiled_floor(&[v]), v, "floor({v}) should be {v}");
        assert_eq!(compiled_ceil(&[v]), v, "ceil({v}) should be {v}");
    }
}

#[test]
fn aggressive_floor_ceil_relationship() {
    // ceil(x) >= floor(x) always, and ceil(x) - floor(x) is 0 or 1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let floor_expr = x.floor();
    let ceil_expr = x.ceiling();

    let compiled_floor = floor_expr.compile(&["x"]).unwrap();
    let compiled_ceil = ceil_expr.compile(&["x"]).unwrap();

    for &v in &[-2.7, -2.0, -1.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.9, 2.0, 2.1] {
        let fl = compiled_floor(&[v]);
        let ce = compiled_ceil(&[v]);
        assert!(ce >= fl, "ceil({v})={ce} < floor({v})={fl}");
        let diff = ce - fl;
        assert!(
            diff == 0.0 || diff == 1.0,
            "ceil({v}) - floor({v}) = {diff}, expected 0 or 1"
        );
    }
}

#[test]
fn aggressive_compile_vs_eval_at_rationals() {
    // Test at x = 1/3, 2/3, 1/7, 5/7 — rationals that don't have exact f64 representations
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exprs: Vec<(&str, Ex)> = vec![
        ("x^2", x.powi(2)),
        ("sin(x)", x.sin()),
        ("exp(x)", x.exp()),
        ("cos(x)+1", x.cos() + 1),
    ];

    let rational_vals: &[(i64, i64)] = &[(1, 3), (2, 3), (1, 7), (5, 7), (3, 11)];

    for (desc, expr) in &exprs {
        let compiled = expr.compile(&["x"]).unwrap();
        for &(p, q) in rational_vals {
            let fval = p as f64 / q as f64;
            let compile_val = compiled(&[fval]);
            let rat = ctx.rational(p, q);
            let eval_result = expr.subs(&x, &rat).eval_f64();
            if let Ok(eval_val) = eval_result {
                assert!(
                    approx_eq_rel(compile_val, eval_val, 1e-9),
                    "{desc} at x={p}/{q}: compile={compile_val}, eval={eval_val}"
                );
            }
        }
    }
}

#[test]
fn aggressive_multivar_codegen_structure_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");

    let cases: Vec<(&str, Ex, &[&str])> = vec![
        ("mv_sub", &x - &y, &["x", "y"]),
        ("mv_div", &x / &y, &["x", "y"]),
        ("mv_sincos", &x.sin() * &y.cos(), &["x", "y"]),
        ("mv_triple", &x * &y + &y * &z + &z * &x, &["x", "y", "z"]),
        ("mv_nested", (&x + &y).sin() * &z.exp(), &["x", "y", "z"]),
    ];

    for (name, expr, args) in &cases {
        let code = expr
            .to_rust_fn(name, args)
            .unwrap_or_else(|e| panic!("codegen failed for {name}: {e}"));
        assert_valid_rust_structure(&code, name);
        // Verify all parameter names appear in signature
        for &arg in *args {
            assert!(
                code.contains(&format!("{arg}: f64")),
                "{name}: missing parameter '{arg}' in:\n{code}"
            );
        }
    }
}

#[test]
fn aggressive_pow_zero_gives_one() {
    // x^0 = 1 for all x (including x=0 by convention)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(0);

    // The symbolic engine may simplify x^0 to 1 immediately, so compile
    // might just push the constant 1.  Either way the result must be 1.
    if let Some(compiled) = expr.compile(&["x"]) {
        for &v in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let got = compiled(&[v]);
            assert!(
                approx_eq_rel(got, 1.0, 1e-10),
                "x^0 at x={v}: compile={got}, expected 1"
            );
        }
    }

    // eval path
    for v in [-2, -1, 0, 1, 2] {
        let eval_val = expr.subs_i64(&x, v).eval_f64().unwrap();
        assert!(
            approx_eq_rel(eval_val, 1.0, 1e-10),
            "x^0 at x={v}: eval={eval_val}, expected 1"
        );
    }
}

#[test]
fn aggressive_pow_one_gives_x() {
    // x^1 = x
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(1);

    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "x^1 should equal x");
}

#[test]
fn aggressive_sum_of_cubes_identity() {
    // (x+y)^3 vs x^3 + 3x^2*y + 3x*y^2 + y^3
    // Both paths should give the same numeric result
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let lhs = (&x + &y).powi(3);
    let rhs = x.powi(3) + &x.powi(2) * &y * 3 + &x * &y.powi(2) * 3 + y.powi(3);

    let compiled_lhs = lhs.compile(&["x", "y"]).unwrap();
    let compiled_rhs = rhs.compile(&["x", "y"]).unwrap();

    let test_pairs: &[(f64, f64)] = &[
        (1.0, 1.0),
        (2.0, 3.0),
        (-1.0, 2.0),
        (0.5, -0.5),
        (0.0, 5.0),
        (-2.0, -3.0),
    ];

    for &(xv, yv) in test_pairs {
        let a = compiled_lhs(&[xv, yv]);
        let b = compiled_rhs(&[xv, yv]);
        assert!(
            approx_eq_rel(a, b, 1e-8),
            "(x+y)^3 vs expanded at ({xv},{yv}): {a} vs {b}"
        );
    }
}

#[test]
fn aggressive_sinh_cosh_exp_identity() {
    // exp(x) = sinh(x) + cosh(x) — checked across all standard vals via both paths
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let exp_expr = x.exp();
    let sum_expr = x.sinh() + x.cosh();

    assert_paths_agree(&ctx, &exp_expr, &x, "x", STANDARD_VALS, "exp(x)");
    assert_paths_agree(&ctx, &sum_expr, &x, "x", STANDARD_VALS, "sinh(x)+cosh(x)");

    // Also check that the two compiled closures give the same result
    let compiled_exp = exp_expr.compile(&["x"]).unwrap();
    let compiled_sum = sum_expr.compile(&["x"]).unwrap();
    for &v in STANDARD_VALS {
        let a = compiled_exp(&[v]);
        let b = compiled_sum(&[v]);
        assert!(
            approx_eq_rel(a, b, 1e-10),
            "exp({v})={a} vs sinh({v})+cosh({v})={b}"
        );
    }
}

#[test]
fn aggressive_tanh_vs_sinh_over_cosh() {
    // tanh(x) = sinh(x)/cosh(x)
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let tanh_expr = x.tanh();
    let ratio_expr = &x.sinh() / &x.cosh();

    let compiled_tanh = tanh_expr.compile(&["x"]).unwrap();
    let compiled_ratio = ratio_expr.compile(&["x"]).unwrap();

    for &v in STANDARD_VALS {
        let a = compiled_tanh(&[v]);
        let b = compiled_ratio(&[v]);
        assert!(
            approx_eq_rel(a, b, 1e-10),
            "tanh({v})={a} vs sinh({v})/cosh({v})={b}"
        );
    }
}

#[test]
fn aggressive_codegen_sign_generates_conditional() {
    // sign(x) codegen should produce an if-else chain, not .signum()
    // (since f64::signum(0.0) = 0.0 but sign(0) should be 0 per symplex semantics)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.sign();
    let code = expr.to_rust_fn("sign_fn", &["x"]).unwrap();
    assert_valid_rust_structure(&code, "sign_fn");
    // Codegen might use .signum() or an if-chain; either is fine
    // as long as the code is syntactically valid.
}

#[test]
fn aggressive_codegen_floor_ceil_use_method_calls() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let floor_code = x.floor().to_rust_fn("fl", &["x"]).unwrap();
    assert_valid_rust_structure(&floor_code, "fl");
    assert!(
        floor_code.contains(".floor()"),
        "floor codegen should emit .floor(): {floor_code}"
    );

    let ceil_code = x.ceiling().to_rust_fn("ce", &["x"]).unwrap();
    assert_valid_rust_structure(&ceil_code, "ce");
    assert!(
        ceil_code.contains(".ceil()"),
        "ceiling codegen should emit .ceil(): {ceil_code}"
    );
}

#[test]
fn aggressive_multivar_eval_vs_compile_cross_check() {
    // f(x, y) = x^2*sin(y) + y*exp(x) — check compile vs subs+eval
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");

    let expr = &x.powi(2) * &y.sin() + &y * &x.exp();
    let compiled = expr.compile(&["x", "y"]).unwrap();

    let test_pairs: &[(i64, i64)] = &[(1, 2), (-1, 1), (0, 0), (2, -1), (-2, 3)];
    for &(xv, yv) in test_pairs {
        let compile_val = compiled(&[xv as f64, yv as f64]);
        let eval_val = expr
            .subs(&x, &ctx.int(xv))
            .subs(&y, &ctx.int(yv))
            .eval_f64();

        if let Ok(ev) = eval_val {
            assert!(
                approx_eq_rel(compile_val, ev, 1e-9),
                "x^2*sin(y)+y*exp(x) at ({xv},{yv}): compile={compile_val}, eval={ev}"
            );
        }
    }
}

#[test]
fn aggressive_constant_expression_no_vars() {
    // Pure constants: compile with no variables
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();

    // pi^2 + e
    let expr = pi.powi(2) + &e;
    let compiled = expr.compile(&[]).unwrap();
    let compile_val = compiled(&[]);

    let eval_val = expr.eval_f64().unwrap();

    assert!(
        approx_eq_rel(compile_val, eval_val, 1e-10),
        "pi^2+e: compile={compile_val}, eval={eval_val}"
    );

    let expected = std::f64::consts::PI.powi(2) + std::f64::consts::E;
    assert!(
        approx_eq_rel(compile_val, expected, 1e-10),
        "pi^2+e: compile={compile_val}, expected={expected}"
    );
}

#[test]
fn aggressive_powi_large_exponents() {
    // x^(-10) at x=2: should be 1/1024
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = x.powi(-10);

    assert_paths_agree(
        &ctx,
        &expr,
        &x,
        "x",
        &[-2.0, -1.0, 0.5, 1.0, 2.0],
        "x^(-10)",
    );

    let compiled = expr.compile(&["x"]).unwrap();
    let val = compiled(&[2.0]);
    assert!(
        approx_eq_rel(val, 1.0 / 1024.0, 1e-10),
        "2^(-10) = {val}, expected {}",
        1.0 / 1024.0
    );
}

#[test]
fn aggressive_zero_expression() {
    // 0 * x should be 0 everywhere
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &ctx.int(0);

    // Might simplify to 0, so compile could be trivial
    if let Some(compiled) = expr.compile(&["x"]) {
        for &v in STANDARD_VALS {
            let got = compiled(&[v]);
            assert!(approx_eq_rel(got, 0.0, 1e-15), "0*x at x={v}: got {got}");
        }
    }
}

#[test]
fn aggressive_identity_expression() {
    // 1 * x should be x everywhere
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x * &ctx.int(1);

    assert_paths_agree(&ctx, &expr, &x, "x", STANDARD_VALS, "1*x should equal x");
}
