//! Round 2 Bug-Hunting Tests: Code Generation & Compile/Eval Pipeline
//!
//! Focus areas:
//! 1. `to_rust_fn` correctness — structural validity, powi/mul_add usage
//! 2. `compile` correctness — cross-checked against subs+eval_f64
//! 3. CSE (Common Subexpression Elimination) — equivalence after extraction
//! 4. LaTeX output — balanced braces, known outputs
//! 5. Display/parse round-trip — parse(display(expr)) == expr
//! 6. Multi-variable compile — variable ordering, missing variables
//! 7. Numerical precision — edge cases near zero, overflow, cancellation

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Verify generated Rust code has balanced delimiters and required structure.
fn assert_valid_rust(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("pub fn {fn_name}")),
        "missing 'pub fn {fn_name}' in:\n{code}"
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

/// Verify LaTeX has balanced braces and \\left/\\right pairing.
fn assert_balanced_latex(latex: &str) {
    let open = latex.chars().filter(|&c| c == '{').count();
    let close = latex.chars().filter(|&c| c == '}').count();
    assert_eq!(
        open, close,
        "unbalanced LaTeX braces ({open} open vs {close} close) in: {latex}"
    );
    let lefts = latex.matches(r"\left").count();
    let rights = latex.matches(r"\right").count();
    assert_eq!(
        lefts, rights,
        "unbalanced \\left/\\right ({lefts} vs {rights}) in: {latex}"
    );
}

/// Compare f64 values with tolerance, handling NaN and Inf.
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= tol
}

/// Test compile vs subs+eval_f64 for a single-variable expression at integer points.
fn check_compile_vs_eval(expr: &Ex, var: &Ex, var_name: &str, test_vals: &[i64]) {
    let compiled = expr
        .compile(&[var_name])
        .unwrap_or_else(|_| panic!("expression `{expr}` should compile"));
    for &v in test_vals {
        let from_compile = compiled(&[v as f64]);
        let from_eval = expr.subs_i64(var, v).eval_f64();
        match from_eval {
            Ok(expected) => {
                assert!(
                    approx_eq(from_compile, expected, 1e-9),
                    "MISMATCH for `{expr}` at {var_name}={v}: compile={from_compile}, eval={expected}"
                );
            }
            Err(_) => {
                // eval may fail on domain issues (e.g. ln(0)); compile returning
                // NaN/Inf in those cases is acceptable.
            }
        }
    }
}

/// Test compile vs subs+eval_f64 for a single-variable expression at rational points.
fn check_compile_vs_eval_rational(
    ctx: &Context,
    expr: &Ex,
    var: &Ex,
    var_name: &str,
    test_vals: &[(i64, i64)],
) {
    let compiled = expr
        .compile(&[var_name])
        .unwrap_or_else(|_| panic!("expression `{expr}` should compile"));
    for &(p, q) in test_vals {
        let fval = p as f64 / q as f64;
        let from_compile = compiled(&[fval]);
        let subst = ctx.rational(p, q);
        let from_eval = expr.subs(var, &subst).eval_f64();
        if let Ok(expected) = from_eval {
            assert!(
                approx_eq(from_compile, expected, 1e-8),
                "MISMATCH for `{expr}` at {var_name}={p}/{q}: compile={from_compile}, eval={expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. to_rust_fn CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_polynomial_structure() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x^3 + 2x^2 - 5x + 7
    let f = x.powi(3) + &x.powi(2) * 2 - &x * 5 + 7;
    let code = f.to_rust_fn("poly3", &["x"]).unwrap();
    assert_valid_rust(&code, "poly3");
    // x^3 should either expand to multiplication chain or use powi(3)
    assert!(
        code.contains("powi(3)") || (code.contains("x * x") && code.contains("* x")),
        "expected powi(3) or expanded mul for x^3 in:\n{code}"
    );
}

#[test]
fn codegen_trig_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = sin(cos(x))
    let f = x.cos().sin();
    let code = f.to_rust_fn("trig_nested", &["x"]).unwrap();
    assert_valid_rust(&code, "trig_nested");
    assert!(code.contains(".cos()"), "missing cos in:\n{code}");
    assert!(code.contains(".sin()"), "missing sin in:\n{code}");
}

#[test]
fn codegen_abs_floor_ceil() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let f_abs = x.abs();
    let code_abs = f_abs.to_rust_fn("f_abs", &["x"]).unwrap();
    assert_valid_rust(&code_abs, "f_abs");
    assert!(code_abs.contains(".abs()"), "missing abs in:\n{code_abs}");

    let f_floor = symplex::parse::parse(&ctx, "floor(x)").unwrap();
    let code_floor = f_floor.to_rust_fn("f_floor", &["x"]).unwrap();
    assert_valid_rust(&code_floor, "f_floor");
    assert!(
        code_floor.contains(".floor()"),
        "missing floor in:\n{code_floor}"
    );

    let f_ceil = symplex::parse::parse(&ctx, "ceil(x)").unwrap();
    let code_ceil = f_ceil.to_rust_fn("f_ceil", &["x"]).unwrap();
    assert_valid_rust(&code_ceil, "f_ceil");
    assert!(
        code_ceil.contains(".ceil()"),
        "missing ceil in:\n{code_ceil}"
    );
}

#[test]
fn codegen_piecewise_generates_if_else() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.zero();
    let cond_pos = x.gt(&zero);
    let cond_neg = x.le(&zero);
    let neg_x = -&x;
    let f = Ex::piecewise(&[(&x, &cond_pos), (&neg_x, &cond_neg)]);
    let code = f.to_rust_fn("pw", &["x"]).unwrap();
    assert_valid_rust(&code, "pw");
    assert!(
        code.contains("if") && code.contains("else"),
        "piecewise should generate if/else chain in:\n{code}"
    );
}

#[test]
fn codegen_powi_uses_correct_form() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // x^2 → powi(2)
    let f2 = x.powi(2);
    let code2 = f2.to_rust_fn("sq", &["x"]).unwrap();
    assert_valid_rust(&code2, "sq");
    assert!(
        code2.contains("powi(2)"),
        "x^2 should use powi(2) in:\n{code2}"
    );

    // x^(-1) → powi(-1)
    let f_inv = x.powi(-1);
    let code_inv = f_inv.to_rust_fn("inv", &["x"]).unwrap();
    assert_valid_rust(&code_inv, "inv");
    assert!(
        code_inv.contains("powi(-1)"),
        "x^(-1) should use powi(-1) in:\n{code_inv}"
    );

    // x^(1/2) → sqrt
    let f_sqrt = x.sqrt();
    let code_sqrt = f_sqrt.to_rust_fn("sr", &["x"]).unwrap();
    assert_valid_rust(&code_sqrt, "sr");
    assert!(
        code_sqrt.contains(".sqrt()"),
        "x^(1/2) should use sqrt in:\n{code_sqrt}"
    );
}

#[test]
fn codegen_mul_add_for_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = 3*x^2 + 2*x + 1 — the Add of Mul children should trigger FMA
    let f = &x.powi(2) * 3 + &x * 2 + 1;
    let code = f.to_rust_fn("fma_poly", &["x"]).unwrap();
    assert_valid_rust(&code, "fma_poly");
    assert!(
        code.contains("mul_add"),
        "polynomial 3x²+2x+1 should use mul_add (FMA) in:\n{code}"
    );
}

#[test]
fn codegen_exp_minus_one_optimization() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(x) - 1 → exp_m1
    let f = x.exp() - 1;
    let code = f.to_rust_fn("expm1_test", &["x"]).unwrap();
    assert_valid_rust(&code, "expm1_test");
    assert!(
        code.contains("exp_m1()"),
        "exp(x) - 1 should use exp_m1() in:\n{code}"
    );
}

#[test]
fn codegen_ln_one_plus_optimization() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ln(1 + x) → ln_1p
    let f = (&x + 1).ln();
    let code = f.to_rust_fn("ln1p_test", &["x"]).unwrap();
    assert_valid_rust(&code, "ln1p_test");
    assert!(
        code.contains("ln_1p()"),
        "ln(1 + x) should use ln_1p() in:\n{code}"
    );
}

#[test]
fn codegen_negative_exponent_powi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-2);
    let code = f.to_rust_fn("neg_pow", &["x"]).unwrap();
    assert_valid_rust(&code, "neg_pow");
    assert!(
        code.contains("powi(-2)"),
        "x^(-2) should use powi(-2) in:\n{code}"
    );
}

#[test]
fn codegen_hyperbolic_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for (expr, name, expected) in [
        (x.sinh(), "f_sinh", ".sinh()"),
        (x.cosh(), "f_cosh", ".cosh()"),
        (x.tanh(), "f_tanh", ".tanh()"),
    ] {
        let code = expr.to_rust_fn(name, &["x"]).unwrap();
        assert_valid_rust(&code, name);
        assert!(
            code.contains(expected),
            "missing {expected} in {name}:\n{code}"
        );
    }
}

#[test]
fn codegen_inverse_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for (expr, name, expected) in [
        (x.asin(), "f_asin", ".asin()"),
        (x.acos(), "f_acos", ".acos()"),
        (x.atan(), "f_atan", ".atan()"),
    ] {
        let code = expr.to_rust_fn(name, &["x"]).unwrap();
        assert_valid_rust(&code, name);
        assert!(
            code.contains(expected),
            "missing {expected} in {name}:\n{code}"
        );
    }
}

#[test]
fn codegen_two_variable_fn() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y + x.powi(2) + y.sin();
    let code = f.to_rust_fn("two_var", &["x", "y"]).unwrap();
    assert_valid_rust(&code, "two_var");
    assert!(
        code.contains("x: f64") && code.contains("y: f64"),
        "should have both params in:\n{code}"
    );
}

#[test]
fn codegen_constant_expression() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let code = pi.to_rust_fn("get_pi", &[]).unwrap();
    assert_valid_rust(&code, "get_pi");
    assert!(
        code.contains("PI"),
        "pi should emit std::f64::consts::PI in:\n{code}"
    );
}

#[test]
fn codegen_deeply_nested() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(cos(exp(ln(abs(x)))))
    let f = x.abs().ln().exp().cos().sin();
    let code = f.to_rust_fn("deep", &["x"]).unwrap();
    assert_valid_rust(&code, "deep");
}

#[test]
fn codegen_powi_expansion_small_exponents() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Exponents 3–6 should be expanded to multiplication chains
    for exp in 3..=6i64 {
        let f = x.powi(exp);
        let fn_name = format!("pow{exp}");
        let code = f.to_rust_fn(&fn_name, &["x"]).unwrap();
        assert_valid_rust(&code, &fn_name);
        // The expanded form uses inline lets or direct multiplication,
        // it should NOT contain the raw powi call for these exponents.
        assert!(
            !code.contains(&format!("powi({exp})")),
            "x^{exp} should be expanded, not use powi({exp}) in:\n{code}"
        );
    }
}

#[test]
fn codegen_powi_7_uses_powi_call() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x^7 is outside the 3..=6 expansion range → should use powi(7)
    let f = x.powi(7);
    let code = f.to_rust_fn("pow7", &["x"]).unwrap();
    assert_valid_rust(&code, "pow7");
    assert!(
        code.contains("powi(7)"),
        "x^7 should use powi(7), not expansion, in:\n{code}"
    );
}

#[test]
fn codegen_free_symbol_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x + &y;
    // Only bind x — y is free → should error
    let result = f.to_rust_fn("bad", &["x"]);
    assert!(
        result.is_err(),
        "codegen with unbound variable y should error"
    );
}

#[test]
fn codegen_imaginary_unit_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let i_expr = symplex::parse::parse(&ctx, "I").unwrap();
    let f = &x + &i_expr;
    let result = f.to_rust_fn("has_i", &["x"]);
    assert!(result.is_err(), "codegen with imaginary unit should error");
}

#[test]
fn codegen_handles_constant_only_expression() {
    let ctx = Context::new();
    let f = ctx.int(42);
    let code = f.to_rust_fn("const_fn", &[]).unwrap();
    assert_valid_rust(&code, "const_fn");
    assert!(
        code.contains("42"),
        "constant fn should contain 42 in:\n{code}"
    );
}

#[test]
fn codegen_many_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Build x + x^2 + x^3 + … + x^10
    let mut f = x.clone();
    for i in 2..=10 {
        f = &f + &x.powi(i);
    }
    let code = f.to_rust_fn("many_terms", &["x"]).unwrap();
    assert_valid_rust(&code, "many_terms");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. compile CORRECTNESS — compile vs subs + eval_f64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_polynomial_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3) - &x.powi(2) * 2 + &x * 5 - 3;
    check_compile_vs_eval(&f, &x, "x", &[-5, -1, 0, 1, 2, 5, 10]);
}

#[test]
fn compile_trig_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() + x.cos();
    check_compile_vs_eval(&f, &x, "x", &[-3, -1, 0, 1, 2, 3]);
}

#[test]
fn compile_exp_ln_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() + (&x + 1).ln();
    // Only non-negative x so ln(x+1) is defined
    check_compile_vs_eval(&f, &x, "x", &[0, 1, 2, 3, 5]);
}

#[test]
fn compile_sqrt_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt() + 1;
    check_compile_vs_eval(&f, &x, "x", &[0, 1, 4, 9, 16, 100]);
}

#[test]
fn compile_abs_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.abs();
    check_compile_vs_eval(&f, &x, "x", &[-10, -1, 0, 1, 10]);
}

#[test]
fn compile_floor_ceil_correctness() {
    let ctx = Context::new();

    let f_floor = symplex::parse::parse(&ctx, "floor(x)").unwrap();
    let f_ceil = symplex::parse::parse(&ctx, "ceil(x)").unwrap();

    let compiled_floor = f_floor.compile(&["x"]).expect("floor should compile");
    let compiled_ceil = f_ceil.compile(&["x"]).expect("ceil should compile");

    assert_eq!(compiled_floor(&[2.7]), 2.0);
    assert_eq!(compiled_floor(&[-2.3]), -3.0);
    assert_eq!(compiled_ceil(&[2.3]), 3.0);
    assert_eq!(compiled_ceil(&[-2.7]), -2.0);
    assert_eq!(compiled_floor(&[0.0]), 0.0);
    assert_eq!(compiled_ceil(&[0.0]), 0.0);
}

#[test]
fn compile_hyperbolic_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sinh(x) + cosh(x) = exp(x)
    let f = x.sinh() + x.cosh();
    let compiled = f.compile(&["x"]).expect("should compile");
    for v in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let result = compiled(&[v]);
        let expected = v.exp();
        assert!(
            approx_eq(result, expected, 1e-10),
            "sinh({v}) + cosh({v}) = {result}, expected exp({v}) = {expected}"
        );
    }
}

#[test]
fn compile_nested_expression_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = sin(x^2) * exp(-x)
    let f = &x.powi(2).sin() * &(-&x).exp();
    let compiled = f.compile(&["x"]).expect("should compile");
    for v in [0.0_f64, 0.5, 1.0, -1.0, 2.0] {
        let expected = (v * v).sin() * (-v).exp();
        let result = compiled(&[v]);
        assert!(
            approx_eq(result, expected, 1e-10),
            "f({v}) = {result}, expected {expected}"
        );
    }
}

#[test]
fn compile_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let c_sin = x.sin().compile(&["x"]).unwrap();
    assert!(c_sin(&[0.0]).abs() < 1e-15, "sin(0) should be 0");

    let c_cos = x.cos().compile(&["x"]).unwrap();
    assert!((c_cos(&[0.0]) - 1.0).abs() < 1e-15, "cos(0) should be 1");

    let c_exp = x.exp().compile(&["x"]).unwrap();
    assert!((c_exp(&[0.0]) - 1.0).abs() < 1e-15, "exp(0) should be 1");
}

#[test]
fn compile_at_large_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let c_sq = x.powi(2).compile(&["x"]).unwrap();
    assert!(
        (c_sq(&[1e6]) - 1e12).abs() / 1e12 < 1e-10,
        "x^2 at 1e6 should be 1e12"
    );

    // sin is bounded [-1, 1] even at large arguments
    let c_sin = x.sin().compile(&["x"]).unwrap();
    let v = c_sin(&[1e8]);
    assert!(
        (-1.0..=1.0).contains(&v),
        "sin at large value should be in [-1,1], got {v}"
    );
}

#[test]
fn compile_at_negative_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3) + 1;
    let compiled = f.compile(&["x"]).unwrap();
    assert!((compiled(&[-2.0]) - (-7.0)).abs() < 1e-10, "(-2)^3+1 = -7");
    assert!(compiled(&[-1.0]).abs() < 1e-10, "(-1)^3+1 = 0");
}

#[test]
fn compile_rational_coefficients() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let half = ctx.rational(1, 2);
    // f(x) = x/2 + 1
    let f = &x * &half + 1;
    let compiled = f.compile(&["x"]).unwrap();
    assert!((compiled(&[4.0]) - 3.0).abs() < 1e-10);
    assert!((compiled(&[0.0]) - 1.0).abs() < 1e-10);
    assert!((compiled(&[-2.0]) - 0.0).abs() < 1e-10);
}

#[test]
fn compile_rational_coefficients_via_eval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + &x * 3 - 7;
    check_compile_vs_eval_rational(&ctx, &f, &x, "x", &[(1, 2), (3, 4), (-5, 3), (7, 1)]);
}

#[test]
fn compile_pi_and_e_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let e = ctx.e();
    let f = &x * &pi + &e;
    let compiled = f.compile(&["x"]).unwrap();
    let expected = std::f64::consts::PI + std::f64::consts::E;
    assert!(
        (compiled(&[1.0]) - expected).abs() < 1e-10,
        "pi*1+e = {}, expected {expected}",
        compiled(&[1.0])
    );
}

#[test]
fn compile_many_terms_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut f = x.clone();
    for i in 2..=10 {
        f = &f + &x.powi(i);
    }
    let compiled = f.compile(&["x"]).unwrap();
    // f(2) = 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 + 512 + 1024 = 2046
    let result = compiled(&[2.0]);
    assert!(
        approx_eq(result, 2046.0, 1e-6),
        "f(2) = {result}, expected 2046"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CSE (Common Subexpression Elimination) CORRECTNESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cse_basic_shared_subexpr() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    // sin(x) appears in both sin(x)^2 and sin(x)
    let f = &sin_x.powi(2) + &sin_x;
    let (bindings, result) = f.cse();

    // sin(x) should be extracted (appears 2+ times)
    if !bindings.is_empty() {
        let binding_names: Vec<String> = bindings.iter().map(|(n, _)| format!("{n}")).collect();
        assert!(
            binding_names.iter().any(|n| n.contains("__cse_")),
            "CSE bindings should use __cse_ prefix, got: {binding_names:?}"
        );
    }
    // The result and bindings together should still print without panicking
    let _ = format!("{result}");
    for (name, val) in &bindings {
        let _ = format!("{name} = {val}");
    }
}

#[test]
fn cse_no_extraction_for_atoms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x + 1;
    let (bindings, result) = f.cse();
    // Atoms (symbols, numbers) should never be extracted
    assert!(
        bindings.is_empty(),
        "CSE should not extract atoms, got {} bindings for `{f}`",
        bindings.len()
    );
    assert_eq!(
        format!("{result}"),
        format!("{f}"),
        "CSE result should match original for simple expression"
    );
}

#[test]
fn cse_deeply_nested_shared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let inner = x.sin().exp(); // exp(sin(x)) used twice
    let f = &inner.powi(2) + &inner * 3;

    let (bindings, _result) = f.cse();
    assert!(
        !bindings.is_empty(),
        "CSE should extract exp(sin(x)) as a shared subexpression"
    );
}

#[test]
fn cse_preserves_evaluation_via_codegen() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let shared = &x * &x + 1; // x^2 + 1
    let f = &shared.sin() + &shared.cos();

    // Compile the original
    let compiled_orig = f.compile(&["x"]).unwrap();

    // CSE should not change semantics — verify via to_rust_fn (which uses CSE internally)
    let code = f.to_rust_fn("cse_verify", &["x"]).unwrap();
    assert_valid_rust(&code, "cse_verify");

    // The compiled version must still give correct values
    for v in [0.0, 1.0, 2.0, -1.0] {
        let orig = compiled_orig(&[v]);
        let x_sq_plus_1 = v * v + 1.0;
        let expected = x_sq_plus_1.sin() + x_sq_plus_1.cos();
        assert!(
            approx_eq(orig, expected, 1e-10),
            "at x={v}: compiled={orig}, manual={expected}"
        );
    }
}

#[test]
fn cse_multiple_shared_subexpressions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = x.sin();
    let b = x.cos();
    // f = sin(x)*cos(x) + sin(x)^2 + cos(x)^2
    // sin^2 + cos^2 = 1, so f = sin*cos + 1
    let f = &a * &b + &a.powi(2) + &b.powi(2);

    let compiled = f.compile(&["x"]).unwrap();
    let val = compiled(&[1.0]);
    let expected = 1.0_f64.sin() * 1.0_f64.cos() + 1.0;
    assert!(
        approx_eq(val, expected, 1e-10),
        "f(1) = {val}, expected {expected}"
    );

    let (bindings, _) = f.cse();
    // sin(x) and/or cos(x) should be extracted (each appears 2+ times)
    assert!(
        !bindings.is_empty(),
        "expected at least one CSE binding, got {}",
        bindings.len()
    );
}

#[test]
fn cse_codegen_integration() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sin_x = x.sin();
    let f = &sin_x.powi(2) + &sin_x * 3 + &sin_x.exp();
    // to_rust_fn enables CSE by default
    let code = f.to_rust_fn("with_cse", &["x"]).unwrap();
    assert_valid_rust(&code, "with_cse");
    // sin(x) appears 3 times; CSE should extract it as a let binding
    assert!(
        code.contains("let t"),
        "CSE should extract sin(x) into a let-binding in:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. LaTeX OUTPUT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn latex_simple_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + &x * 2 + 1;
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("x^{2}") || latex.contains("x^2"),
        "latex should contain x^2: {latex}"
    );
}

#[test]
fn latex_fraction() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    let latex = half.to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("frac"),
        "1/2 should render as \\frac: {latex}"
    );
}

#[test]
fn latex_trig_functions_known_output() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    assert_eq!(
        x.sin().to_latex(),
        r"\sin\left(x\right)",
        "sin(x) latex mismatch"
    );
    assert_eq!(
        x.cos().to_latex(),
        r"\cos\left(x\right)",
        "cos(x) latex mismatch"
    );
    assert_eq!(
        x.tan().to_latex(),
        r"\tan\left(x\right)",
        "tan(x) latex mismatch"
    );
}

#[test]
fn latex_sqrt() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let latex = x.sqrt().to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("\\sqrt"),
        "sqrt should render as \\sqrt: {latex}"
    );
}

#[test]
fn latex_nested_deep_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (sin(x) + cos(x))^2 / (exp(x) + 1)
    let num = (&x.sin() + &x.cos()).powi(2);
    let denom = x.exp() + 1;
    let f = &num / &denom;
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
}

#[test]
fn latex_pi_and_e() {
    let ctx = Context::new();
    let pi_latex = ctx.pi().to_latex();
    assert!(
        pi_latex.contains("\\pi"),
        "pi should render as \\pi: {pi_latex}"
    );
    let e_latex = ctx.e().to_latex();
    assert!(e_latex.contains('e'), "e should contain 'e': {e_latex}");
}

#[test]
fn latex_negative_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x - 1;
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
    // Should show subtraction
    assert!(
        latex.contains('-'),
        "x - 1 should contain minus sign: {latex}"
    );
}

#[test]
fn latex_power_negative_exponent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-1);
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("frac") || latex.contains("^{-1}"),
        "x^(-1) should be \\frac{{1}}{{x}} or x^{{-1}}: {latex}"
    );
}

#[test]
fn latex_inline_and_display() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let inline = x.to_latex_inline();
    assert!(
        inline.starts_with('$') && inline.ends_with('$'),
        "inline latex should be wrapped in $: {inline}"
    );
    assert!(
        !inline.starts_with("$$"),
        "inline should use single $, not $$: {inline}"
    );

    let display = x.to_latex_display();
    assert!(
        display.starts_with("$$") && display.ends_with("$$"),
        "display latex should be wrapped in $$: {display}"
    );
}

#[test]
fn latex_abs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let latex = x.abs().to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains('|') || latex.contains("\\lvert") || latex.contains("\\left|"),
        "abs should render with vertical bars: {latex}"
    );
}

#[test]
fn latex_exp_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let latex = x.exp().to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("e^{") || latex.contains("\\exp"),
        "exp(x) should contain e^{{}} or \\exp: {latex}"
    );
}

#[test]
fn latex_ln_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let latex = x.ln().to_latex();
    assert_balanced_latex(&latex);
    assert!(
        latex.contains("\\ln") || latex.contains("\\log"),
        "ln(x) should render as \\ln or \\log: {latex}"
    );
}

#[test]
fn latex_balanced_braces_stress() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let expressions: Vec<Ex> = vec![
        x.sin(),
        x.cos().exp(),
        x.powi(2) + &x + 1,
        x.sqrt(),
        x.powi(-1),
        x.powi(-2),
        x.abs(),
        x.ln(),
        (&x.sin() + &x.cos()).powi(3),
        x.sin().powi(2),
        (&x.powi(2) + 1).sqrt(),
    ];

    for (i, expr) in expressions.iter().enumerate() {
        let latex = expr.to_latex();
        assert_balanced_latex(&latex);
        assert!(!latex.is_empty(), "expression {i} produced empty LaTeX");
    }
}

#[test]
fn latex_complex_multi_term() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x.powi(3).sin() * &y.cos() + (&x * &y).exp() - y.sqrt();
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. DISPLAY / PARSE ROUND-TRIP
// ═══════════════════════════════════════════════════════════════════════════

/// Check that parse(display(expr)) displays the same as the original.
fn assert_roundtrip(ctx: &Context, expr: &Ex, label: &str) {
    let displayed = format!("{expr}");
    match symplex::parse::parse(ctx, &displayed) {
        Ok(reparsed) => {
            let redisplayed = format!("{reparsed}");
            assert_eq!(
                displayed, redisplayed,
                "round-trip failure for {label}: '{displayed}' -> '{redisplayed}'"
            );
        }
        Err(e) => {
            panic!("parse failed for {label}: display='{displayed}', error: {e}");
        }
    }
}

#[test]
fn roundtrip_integer() {
    let ctx = Context::new();
    assert_roundtrip(&ctx, &ctx.int(42), "42");
    assert_roundtrip(&ctx, &ctx.int(0), "0");
    assert_roundtrip(&ctx, &ctx.int(-7), "-7");
}

#[test]
fn roundtrip_symbol() {
    let ctx = Context::new();
    assert_roundtrip(&ctx, &ctx.symbol("x"), "x");
    assert_roundtrip(&ctx, &ctx.symbol("abc"), "abc");
}

#[test]
fn roundtrip_addition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    assert_roundtrip(&ctx, &(&x + &y), "x + y");
}

#[test]
fn roundtrip_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + &x * 2 + 1;
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "failed to parse polynomial '{displayed}': {:?}",
        reparsed.err()
    );
    if let Ok(r) = reparsed {
        assert_eq!(format!("{r}"), displayed, "polynomial round-trip mismatch");
    }
}

#[test]
fn roundtrip_sin() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin();
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse sin display '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_negation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = -&x;
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse negation '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_pi() {
    let ctx = Context::new();
    assert_roundtrip(&ctx, &ctx.pi(), "pi");
}

#[test]
fn roundtrip_power() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse power '{displayed}': {:?}",
        reparsed.err()
    );
    if let Ok(r) = reparsed {
        assert_eq!(format!("{r}"), displayed, "power round-trip mismatch");
    }
}

#[test]
fn roundtrip_nested_functions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().exp();
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse nested fns '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_subtraction() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x - &y;
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse subtraction '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_multiplication() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x * 3;
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse multiplication '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_division_representation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x / &y;
    let displayed = format!("{f}");
    // Division may display as x*y^(-1) — the parser must handle it
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse division display '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_exp() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp();
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse exp display '{displayed}': {:?}",
        reparsed.err()
    );
}

#[test]
fn roundtrip_multiple_operations() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + &x * 3 + 1;
    let displayed = format!("{f}");
    let reparsed = symplex::parse::parse(&ctx, &displayed);
    assert!(
        reparsed.is_ok(),
        "should parse '{displayed}': {:?}",
        reparsed.err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. MULTI-VARIABLE COMPILE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_two_variables() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x * &y + x.powi(2) - y.sin();
    let compiled = f.compile(&["x", "y"]).expect("should compile 2-var");

    // f(2, 3) = 2*3 + 4 - sin(3)
    let expected = 6.0 + 4.0 - 3.0_f64.sin();
    let result = compiled(&[2.0, 3.0]);
    assert!(
        approx_eq(result, expected, 1e-10),
        "f(2,3) = {result}, expected {expected}"
    );
}

#[test]
fn compile_variable_order_matters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // f(x, y) = x - y  (not commutative)
    let f = &x - &y;

    let compiled_xy = f.compile(&["x", "y"]).unwrap();
    let compiled_yx = f.compile(&["y", "x"]).unwrap();

    // ["x", "y"]: args[0]=x=3, args[1]=y=1 → 3 - 1 = 2
    assert!(
        approx_eq(compiled_xy(&[3.0, 1.0]), 2.0, 1e-10),
        "xy order: f(3,1) should be 2, got {}",
        compiled_xy(&[3.0, 1.0])
    );

    // ["y", "x"]: args[0]=y=3, args[1]=x=1 → 1 - 3 = -2
    assert!(
        approx_eq(compiled_yx(&[3.0, 1.0]), -2.0, 1e-10),
        "yx order: f(3,1) should be -2, got {}",
        compiled_yx(&[3.0, 1.0])
    );
}

#[test]
fn compile_three_variables() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let f = &x * &y + &y * &z + &z * &x;
    let compiled = f.compile(&["x", "y", "z"]).expect("3-var should compile");
    // f(1, 2, 3) = 1*2 + 2*3 + 3*1 = 11
    assert!(
        approx_eq(compiled(&[1.0, 2.0, 3.0]), 11.0, 1e-10),
        "f(1,2,3) = {}, expected 11",
        compiled(&[1.0, 2.0, 3.0])
    );
}

#[test]
fn compile_unused_variable_in_declaration() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + 1;
    // Expression only uses x, but we compile with ["x", "y"]
    let compiled = f
        .compile(&["x", "y"])
        .expect("should compile with extra var");
    assert!(
        approx_eq(compiled(&[3.0, 999.0]), 10.0, 1e-10),
        "unused y should not affect result"
    );
}

#[test]
fn compile_missing_variable_returns_none() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x + &y;
    let compiled = f.compile(&["x"]); // y is unbound
    assert!(
        compiled.is_err(),
        "compile should return None when expression has unbound variable"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. NUMERICAL PRECISION EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn precision_near_zero_exp_m1() {
    // FIXED: compile() now detects exp(x)-1 and emits ExpM1 instruction,
    // matching the codegen path's exp_m1() optimization.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() - 1;
    let compiled = f.compile(&["x"]).unwrap();

    let tiny = 1e-15;
    let result = compiled(&[tiny]);

    // The correct answer is ~1e-15 (approximately equal to tiny itself).
    // With the ExpM1 instruction, we get full precision.
    let relative_error = ((result - tiny) / tiny).abs();
    assert!(
        relative_error < 1e-5,
        "compile() should use exp_m1 for good precision near zero, \
         but got relative error {relative_error}"
    );

    // codegen also uses exp_m1:
    let code = f.to_rust_fn("expm1_check", &["x"]).unwrap();
    assert!(
        code.contains("exp_m1()"),
        "codegen should use exp_m1 optimization"
    );
}

#[test]
fn precision_sin_near_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let compiled = x.sin().compile(&["x"]).unwrap();

    let tiny = 1e-10;
    let result = compiled(&[tiny]);
    assert!(
        (result - tiny).abs() < tiny * 1e-5,
        "sin({tiny}) = {result}, expected approximately {tiny}"
    );
}

#[test]
fn precision_large_cancellation() {
    // PARTIALLY FIXED: compile() now runs eval() as a pre-pass, which catches
    // exact-zero terms and special values but does NOT do full algebraic
    // simplification. The expression (x+1)^2 - x^2 - 2x is algebraically 1,
    // but eval() alone can't prove that — it would need expand() + cancel().
    //
    // At moderate values the result is close to 1 (floating-point is adequate).
    // At very large values, catastrophic cancellation still occurs because
    // the eval() pre-pass doesn't expand/simplify polynomial expressions.
    // Users should call .simplify() before .compile() for such cases.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x+1)^2 - x^2 - 2x = 1 exactly (algebraically)
    let f = (&x + 1).powi(2) - x.powi(2) - &x * 2;
    let compiled = f.compile(&["x"]).unwrap();

    // At moderate values, the result is close to 1
    assert!(
        approx_eq(compiled(&[100.0]), 1.0, 1e-6),
        "at x=100, should be close to 1.0, got {}",
        compiled(&[100.0])
    );

    // Workaround: simplify before compiling → exact result
    let f_simplified = f.simplify();
    let compiled_simplified = f_simplified.compile(&["x"]).unwrap();
    assert!(
        approx_eq(compiled_simplified(&[1e10]), 1.0, 1e-10),
        "simplify() before compile() should give exact result at x=1e10, got {}",
        compiled_simplified(&[1e10])
    );
}

#[test]
fn precision_overflow_boundary() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let compiled = x.exp().compile(&["x"]).unwrap();

    // exp(710) overflows f64
    assert!(
        compiled(&[710.0]).is_infinite(),
        "exp(710) should overflow to inf"
    );

    // exp(-750) underflows to 0
    assert_eq!(compiled(&[-750.0]), 0.0, "exp(-750) should underflow to 0");
}

#[test]
fn precision_sqrt_of_square() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sqrt(x^2) = |x|
    let f = x.powi(2).sqrt();
    let compiled = f.compile(&["x"]).unwrap();

    for v in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        let result = compiled(&[v]);
        assert!(
            approx_eq(result, v.abs(), 1e-10),
            "sqrt(({v})^2) = {result}, expected {}",
            v.abs()
        );
    }
}

#[test]
fn precision_sin_squared_plus_cos_squared() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(2) + x.cos().powi(2);
    let compiled = f.compile(&["x"]).unwrap();
    // sin²(x) + cos²(x) = 1 for all x
    for v in [0.0, 0.1, 1.0, 2.0, std::f64::consts::PI, 100.0, -5.0] {
        let result = compiled(&[v]);
        assert!(
            (result - 1.0).abs() < 1e-12,
            "sin^2({v}) + cos^2({v}) = {result}, expected 1.0"
        );
    }
}

#[test]
fn precision_tan_vs_sin_over_cos() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let c_tan = x.tan().compile(&["x"]).unwrap();
    let c_ratio = (&x.sin() / &x.cos()).compile(&["x"]).unwrap();

    for v in [0.1, 0.5, 1.0, -0.5, 2.0] {
        let t = c_tan(&[v]);
        let r = c_ratio(&[v]);
        assert!(approx_eq(t, r, 1e-10), "tan({v})={t} vs sin/cos={r}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-cutting: compile vs codegen agree
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_and_compile_agree_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4) - &x.powi(3) * 2 + &x * 7 - 3;

    let code = f.to_rust_fn("p4", &["x"]).unwrap();
    assert_valid_rust(&code, "p4");

    let compiled = f.compile(&["x"]).unwrap();
    for v in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let result = compiled(&[v]);
        let expected = v.powi(4) - 2.0 * v.powi(3) + 7.0 * v - 3.0;
        assert!(
            approx_eq(result, expected, 1e-8),
            "at x={v}: compiled={result}, expected={expected}"
        );
    }
}

#[test]
fn codegen_and_compile_agree_trig_composition() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x.powi(2).sin() + x.cos().exp();

    let code = f.to_rust_fn("trig_comp", &["x"]).unwrap();
    assert_valid_rust(&code, "trig_comp");

    let compiled = f.compile(&["x"]).unwrap();
    for v in [0.0, 0.5, 1.0, -1.0, 2.0] {
        let result = compiled(&[v]);
        let expected = (v * v).sin() + v.cos().exp();
        assert!(
            approx_eq(result, expected, 1e-9),
            "at x={v}: compiled={result}, expected={expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge: compile and codegen agree on piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_piecewise_matches_codegen() {
    // Historically compile() rejected Piecewise while to_rust_fn supported
    // it.  Since 0.2 both back-ends handle it; compile() evaluates the
    // conditions in order and picks the first true branch.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let zero = ctx.zero();
    let cond = x.gt(&zero);
    let neg_x = -&x;
    let f = Ex::piecewise(&[(&x, &cond), (&neg_x, &x.le(&zero))]);

    let compiled = f
        .compile(&["x"])
        .expect("compile() should support piecewise");
    assert_eq!(compiled(&[3.0]), 3.0, "x > 0 branch");
    assert_eq!(compiled(&[-2.0]), 2.0, "x <= 0 branch");
    assert_eq!(compiled(&[0.0]), 0.0, "boundary");

    // And codegen should succeed
    let code = f.to_rust_fn("abs_pw", &["x"]);
    assert!(
        code.is_ok(),
        "to_rust_fn should handle piecewise, got: {:?}",
        code.err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: sign(0) semantics — Rust signum vs mathematical sign
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sign_function_semantics() {
    // Bug 12 fixed: sign(0) now returns 0.0 (mathematical convention)
    // instead of 1.0 (Rust's f64::signum / IEEE 754 semantics).
    let ctx = Context::new();
    let f = symplex::parse::parse(&ctx, "sign(x)").unwrap();
    let compiled = f.compile(&["x"]);

    if let Ok(func) = compiled {
        assert_eq!(func(&[5.0]), 1.0, "sign(5) should be 1");
        assert_eq!(func(&[-3.0]), -1.0, "sign(-3) should be -1");
        assert_eq!(
            func(&[0.0]),
            0.0,
            "sign(0) should be 0 (mathematical convention)"
        );
    }
}

#[test]
fn heaviside_function_at_zero() {
    let ctx = Context::new();
    let f = symplex::parse::parse(&ctx, "Heaviside(x)").unwrap();
    let compiled = f.compile(&["x"]);

    if let Ok(func) = compiled {
        assert_eq!(func(&[1.0]), 1.0, "heaviside(1) should be 1");
        assert_eq!(func(&[-1.0]), 0.0, "heaviside(-1) should be 0");
        // Heaviside(0) = 0.5 by convention in the implementation
        assert_eq!(func(&[0.0]), 0.5, "heaviside(0) should be 0.5");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: compile x^0 and x^1 (canonicalization)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_powi_zero_gives_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(0);
    // x^0 should canonicalize to 1
    let compiled = f.compile(&["x"]).unwrap();
    assert!(
        approx_eq(compiled(&[42.0]), 1.0, 1e-15),
        "x^0 should be 1, got {}",
        compiled(&[42.0])
    );
    assert!(
        approx_eq(compiled(&[0.0]), 1.0, 1e-15),
        "0^0 should be 1 (convention), got {}",
        compiled(&[0.0])
    );
}

#[test]
fn compile_powi_one_gives_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(1);
    let compiled = f.compile(&["x"]).unwrap();
    assert!(
        approx_eq(compiled(&[42.0]), 42.0, 1e-15),
        "x^1 should be x, got {}",
        compiled(&[42.0])
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Compile: complex expression comprehensive test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_complex_expression_comprehensive() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f(x) = x^3 * sin(x) + exp(-x^2) + |x| + ln(x^2 + 1)
    let f = &(&x.powi(3) * &x.sin()) + &(-&x.powi(2)).exp() + &x.abs() + &(&x.powi(2) + 1).ln();

    let compiled = f.compile(&["x"]).unwrap();
    for v in [0.5_f64, 1.0, 2.0, -1.0, -2.0] {
        let expected = v.powi(3) * v.sin() + (-v * v).exp() + v.abs() + (v * v + 1.0).ln();
        let result = compiled(&[v]);
        assert!(
            approx_eq(result, expected, 1e-8),
            "complex f({v}) = {result}, expected {expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compile: constants in expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_pure_constants() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let e = ctx.e();

    let c_pi = pi.compile(&[]).unwrap();
    assert!(
        (c_pi(&[]) - std::f64::consts::PI).abs() < 1e-15,
        "compiled pi = {}",
        c_pi(&[])
    );

    let c_e = e.compile(&[]).unwrap();
    assert!(
        (c_e(&[]) - std::f64::consts::E).abs() < 1e-15,
        "compiled e = {}",
        c_e(&[])
    );
}

#[test]
fn compile_sin_pi_x() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let f = (&pi * &x).sin();
    let compiled = f.compile(&["x"]).unwrap();

    // sin(pi * 0) = 0
    assert!(compiled(&[0.0]).abs() < 1e-10);
    // sin(pi * 0.5) = sin(pi/2) = 1
    assert!((compiled(&[0.5]) - 1.0).abs() < 1e-10);
    // sin(pi * 1) ≈ 0
    assert!(compiled(&[1.0]).abs() < 1e-10);
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: expand_powi correctness at each expansion level
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_powi_expansion_numerical_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    // Test every exponent from -6 to 10 at several values
    for exp in -6..=10i64 {
        let f = x.powi(exp);
        let compiled = f.compile(&["x"]).unwrap();
        for v in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0] {
            // Skip 0^negative which is Inf
            if v == 0.0 && exp < 0 {
                continue;
            }
            let result = compiled(&[v]);
            let expected = v.powi(exp as i32);
            assert!(
                approx_eq(result, expected, 1e-8),
                "x^{exp} at x={v}: compiled={result}, expected={expected}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: codegen for zero coefficients and identity simplification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[allow(clippy::erasing_op)] // Symbolic `x * 0` is exactly what's under test.
fn codegen_zero_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 0 * x + 1 should simplify to 1
    let f = &x * 0 + 1;
    let code = f.to_rust_fn("zero_coeff", &["x"]).unwrap();
    assert_valid_rust(&code, "zero_coeff");
    let compiled = f.compile(&["x"]).unwrap();
    assert!(
        approx_eq(compiled(&[999.0]), 1.0, 1e-15),
        "0*x+1 should be 1, got {}",
        compiled(&[999.0])
    );
}

#[test]
fn codegen_one_coefficient() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // 1 * x should simplify to x
    let f = &x * 1;
    let compiled = f.compile(&["x"]).unwrap();
    assert!(
        approx_eq(compiled(&[42.0]), 42.0, 1e-15),
        "1*x should be x, got {}",
        compiled(&[42.0])
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: codegen for negative terms in sums
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_subtraction_detection() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x - 1: should generate subtraction, not addition of -1
    let f = &x - 1;
    let code = f.to_rust_fn("sub_test", &["x"]).unwrap();
    assert_valid_rust(&code, "sub_test");
    // Verify numerically
    let compiled = f.compile(&["x"]).unwrap();
    assert!(approx_eq(compiled(&[5.0]), 4.0, 1e-15));
}

#[test]
fn codegen_multiple_subtractions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x - y - 1
    let f = &x - &y - 1;
    let code = f.to_rust_fn("multi_sub", &["x", "y"]).unwrap();
    assert_valid_rust(&code, "multi_sub");
    let compiled = f.compile(&["x", "y"]).unwrap();
    assert!(
        approx_eq(compiled(&[10.0, 3.0]), 6.0, 1e-15),
        "10 - 3 - 1 = {}",
        compiled(&[10.0, 3.0])
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: inverse hyperbolic functions in codegen
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_inverse_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    for (expr, name, expected_str) in [
        (x.asinh(), "f_asinh", ".asinh()"),
        (x.acosh(), "f_acosh", ".acosh()"),
        (x.atanh(), "f_atanh", ".atanh()"),
    ] {
        let code = expr.to_rust_fn(name, &["x"]).unwrap();
        assert_valid_rust(&code, name);
        assert!(
            code.contains(expected_str),
            "missing {expected_str} in {name}:\n{code}"
        );
    }
}

#[test]
fn compile_inverse_hyperbolic_correctness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");

    let c_asinh = x.asinh().compile(&["x"]).unwrap();
    assert!(approx_eq(c_asinh(&[0.0]), 0.0, 1e-15), "asinh(0)=0");

    let c_acosh = x.acosh().compile(&["x"]).unwrap();
    assert!(approx_eq(c_acosh(&[1.0]), 0.0, 1e-15), "acosh(1)=0");

    let c_atanh = x.atanh().compile(&["x"]).unwrap();
    assert!(approx_eq(c_atanh(&[0.0]), 0.0, 1e-15), "atanh(0)=0");
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: NaN and Infinity in expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_ln_at_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let compiled = x.ln().compile(&["x"]).unwrap();
    let result = compiled(&[0.0]);
    assert!(
        result.is_infinite() && result < 0.0,
        "ln(0) should be -inf, got {result}"
    );
}

#[test]
fn compile_division_by_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(-1);
    let compiled = f.compile(&["x"]).unwrap();
    let result = compiled(&[0.0]);
    assert!(
        result.is_infinite() || result.is_nan(),
        "1/0 should be inf or nan, got {result}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// BUG HUNTING: codegen Heaviside emits valid if/else
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_heaviside_generates_valid_code() {
    let ctx = Context::new();
    let f = symplex::parse::parse(&ctx, "Heaviside(x)").unwrap();
    let code = f.to_rust_fn("heaviside_fn", &["x"]).unwrap();
    assert_valid_rust(&code, "heaviside_fn");
    // Should contain conditional logic
    assert!(
        code.contains("if") || code.contains("0.5"),
        "heaviside codegen should have conditional or 0.5 in:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LaTeX: verify no double-negation artifacts
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn latex_no_double_negation() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = -(-&x); // double negation should simplify
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
    // Should not contain "--"
    assert!(
        !latex.contains("--"),
        "double negation should simplify, got: {latex}"
    );
}

#[test]
fn latex_subtraction_of_negative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &x - &(-&ctx.int(1)); // x - (-1) should be x + 1
    let latex = f.to_latex();
    assert_balanced_latex(&latex);
}
