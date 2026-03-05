//! Integration tests for the `to_rust_fn` codegen feature.

use symplex::prelude::*;

/// Helper: verify generated code has balanced braces and required structure.
fn assert_valid_rust_structure(code: &str, fn_name: &str) {
    assert!(
        code.contains(&format!("pub fn {fn_name}")),
        "missing function signature 'pub fn {fn_name}' in:\n{code}"
    );
    assert!(
        code.contains("-> f64"),
        "missing return type '-> f64' in:\n{code}"
    );
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        opens, closes,
        "unbalanced braces ({opens} open vs {closes} close) in:\n{code}"
    );
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(
        open_parens, close_parens,
        "unbalanced parentheses ({open_parens} open vs {close_parens} close) in:\n{code}"
    );
}

#[test]
fn codegen_simple_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2) + &x * 2 + 1;
    let code = f.to_rust_fn("poly", &["x"]).unwrap();
    assert!(
        code.contains("pub fn poly(x: f64) -> f64"),
        "missing function signature in:\n{code}"
    );
    // x^2 should use powi(2) for efficiency
    assert!(
        code.contains("powi(2)"),
        "expected powi(2) for x^2 in:\n{code}"
    );
    // Should end with a closing brace
    assert!(code.trim_end().ends_with('}'), "missing closing brace");
}

#[test]
fn codegen_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() + x.cos();
    let code = f.to_rust_fn("trig", &["x"]).unwrap();
    assert!(code.contains(".sin()"), "missing sin() in:\n{code}");
    assert!(code.contains(".cos()"), "missing cos() in:\n{code}");
    assert!(code.contains("pub fn trig(x: f64) -> f64"));
}

#[test]
fn codegen_two_vars() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = x.powi(2) + y.powi(2);
    let code = f.to_rust_fn("sum_sq", &["x", "y"]).unwrap();
    assert!(
        code.contains("x: f64, y: f64"),
        "missing two-parameter signature in:\n{code}"
    );
    assert!(code.contains("pub fn sum_sq"));
}

#[test]
fn codegen_constants() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = ctx.pi();
    let e = ctx.e();
    let f = &pi * &x + e;
    let code = f.to_rust_fn("with_consts", &["x"]).unwrap();
    assert!(
        code.contains("PI") || code.contains("std::f64::consts"),
        "missing PI constant in:\n{code}"
    );
    assert!(
        code.contains("E") || code.contains("std::f64::consts::E"),
        "missing E constant in:\n{code}"
    );
}

#[test]
fn codegen_free_symbol_error() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let f = &x + &y;
    // y is not listed as an argument, so it should be a free symbol error
    let result = f.to_rust_fn("partial", &["x"]);
    assert!(result.is_err(), "expected error for free symbol y");
}

#[test]
fn codegen_exp_ln() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() + x.ln();
    let code = f.to_rust_fn("expln", &["x"]).unwrap();
    assert!(code.contains(".exp()"), "missing exp() in:\n{code}");
    assert!(code.contains(".ln()"), "missing ln() in:\n{code}");
}

#[test]
fn codegen_sqrt_via_pow_half() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sqrt();
    let code = f.to_rust_fn("my_sqrt", &["x"]).unwrap();
    // sqrt is Pow(x, 1/2) internally; codegen should emit .sqrt()
    assert!(
        code.contains(".sqrt()"),
        "expected .sqrt() for x^(1/2) in:\n{code}"
    );
}

#[test]
fn codegen_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sinh() + x.cosh() + x.tanh();
    let code = f.to_rust_fn("hyp", &["x"]).unwrap();
    assert!(code.contains(".sinh()"), "missing sinh() in:\n{code}");
    assert!(code.contains(".cosh()"), "missing cosh() in:\n{code}");
    assert!(code.contains(".tanh()"), "missing tanh() in:\n{code}");
}

#[test]
fn codegen_abs_and_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.abs() + x.sign();
    let code = f.to_rust_fn("abs_sign", &["x"]).unwrap();
    assert!(code.contains(".abs()"), "missing abs() in:\n{code}");
    assert!(code.contains(".signum()"), "missing signum() in:\n{code}");
}

#[test]
fn codegen_negative_exponent_uses_powi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(-2));
    let code = f.to_rust_fn("inv_sq", &["x"]).unwrap();
    assert!(
        code.contains("powi(-2)"),
        "expected powi(-2) for x^(-2) in:\n{code}"
    );
}

#[test]
fn codegen_no_args_constant_expr() {
    let ctx = Context::new();
    let pi = ctx.pi();
    let f = pi.powi(2);
    let code = f.to_rust_fn("pi_squared", &[]).unwrap();
    assert!(
        code.contains("pub fn pi_squared() -> f64"),
        "expected zero-arg function in:\n{code}"
    );
    assert!(code.contains("PI"), "expected PI constant in:\n{code}");
}

#[test]
fn codegen_inverse_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.asin() + x.acos() + x.atan();
    let code = f.to_rust_fn("inv_trig", &["x"]).unwrap();
    assert!(code.contains(".asin()"), "missing asin() in:\n{code}");
    assert!(code.contains(".acos()"), "missing acos() in:\n{code}");
    assert!(code.contains(".atan()"), "missing atan() in:\n{code}");
}

#[test]
fn codegen_complex_expression() {
    // A more realistic expression: sin(x)^2 + cos(x)^2 (which may or may not simplify)
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin().powi(2) + x.cos().powi(2);
    // Whether it simplifies to 1 or keeps the form, codegen should succeed
    let code = f.to_rust_fn("trig_identity", &["x"]).unwrap();
    assert_valid_rust_structure(&code, "trig_identity");
}

// ═══════════════════════════════════════════════════════════════════════════
// Wave 8: Verify generated code syntax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_output_is_valid_rust_syntax() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let code = x.powi(2).to_rust_fn("square", &["x"]).unwrap();
    // Verify it contains required elements
    assert!(code.contains("fn square"), "missing fn square in:\n{code}");
    assert!(code.contains("-> f64"), "missing -> f64 in:\n{code}");
    // Verify balanced braces
    let opens = code.chars().filter(|&c| c == '{').count();
    let closes = code.chars().filter(|&c| c == '}').count();
    assert_eq!(opens, closes, "unbalanced braces in: {code}");
    // Verify balanced parentheses
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(open_parens, close_parens, "unbalanced parentheses in: {code}");
}

#[test]
fn codegen_trig_structure_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)^2 + cos(x) + tan(x)
    let f = x.sin().powi(2) + x.cos() + x.tan();
    let code = f.to_rust_fn("trig_combo", &["x"]).unwrap();
    assert_valid_rust_structure(&code, "trig_combo");
    assert!(code.contains(".sin()"), "missing sin() in:\n{code}");
    assert!(code.contains(".cos()"), "missing cos() in:\n{code}");
    assert!(code.contains(".tan()"), "missing tan() in:\n{code}");
}

#[test]
fn codegen_multi_variable_structure_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let f = &x.powi(2) * &y + z.sin();
    let code = f.to_rust_fn("multi_var", &["x", "y", "z"]).unwrap();
    assert_valid_rust_structure(&code, "multi_var");
    assert!(
        code.contains("x: f64") && code.contains("y: f64") && code.contains("z: f64"),
        "missing three-parameter signature in:\n{code}"
    );
}

#[test]
fn codegen_nested_expression_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(sin(x^2)) + ln(cos(x) + 2)
    let f = x.powi(2).sin().exp() + (&x.cos() + 2).ln();
    let code = f.to_rust_fn("nested", &["x"]).unwrap();
    assert_valid_rust_structure(&code, "nested");
    assert!(code.contains(".exp()"), "missing exp() in:\n{code}");
    assert!(code.contains(".sin()"), "missing sin() in:\n{code}");
    assert!(code.contains(".ln()"), "missing ln() in:\n{code}");
}

#[test]
fn codegen_all_basic_functions_valid() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Chain several functions to stress-test codegen
    for (name, expr) in [
        ("gen_sqrt", x.sqrt()),
        ("gen_abs", x.abs()),
        ("gen_exp", x.exp()),
        ("gen_ln", x.ln()),
        ("gen_sin", x.sin()),
        ("gen_cos", x.cos()),
        ("gen_asin", x.asin()),
        ("gen_acos", x.acos()),
        ("gen_atan", x.atan()),
        ("gen_sinh", x.sinh()),
        ("gen_cosh", x.cosh()),
        ("gen_tanh", x.tanh()),
    ] {
        let code = expr
            .to_rust_fn(name, &["x"])
            .unwrap_or_else(|e| panic!("codegen failed for {name}: {e}"));
        assert_valid_rust_structure(&code, name);
    }
}
