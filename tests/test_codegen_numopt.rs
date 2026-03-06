//! Integration tests for numerical optimization rewrites in code generation.
//!
//! These tests verify that the codegen pipeline detects patterns like
//! `exp(x) - 1`, `ln(1 + x)`, `ln(x)/ln(2)`, and `2^x` and emits
//! numerically precise Rust intrinsics (`exp_m1`, `ln_1p`, `log2`, `exp2`).

use symplex::matrix::{CodegenOptions, MathBackend};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 1: exp(x) - 1 → exp_m1
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_expm1_pattern() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() - 1;
    let code = f.to_rust_fn("expm1_fn", &["x"]).unwrap();
    assert!(
        code.contains(".exp_m1()"),
        "expected .exp_m1() for exp(x) - 1, got:\n{code}"
    );
    assert!(
        !code.contains(".exp()"),
        "should NOT contain bare .exp() when exp_m1 is used, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 2: ln(1 + x) → ln_1p
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_log1p_pattern() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    let f = (&x + &one).ln();
    let code = f.to_rust_fn("log1p_fn", &["x"]).unwrap();
    assert!(
        code.contains(".ln_1p()"),
        "expected .ln_1p() for ln(1 + x), got:\n{code}"
    );
    assert!(
        !code.contains(".ln()"),
        "should NOT contain bare .ln() when ln_1p is used, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 3: ln(x) / ln(2) → log2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_log2_pattern() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    // x.log(&two) computes ln(x) / ln(2)
    let f = x.log(&two);
    let code = f.to_rust_fn("log2_fn", &["x"]).unwrap();
    assert!(
        code.contains(".log2()"),
        "expected .log2() for ln(x)/ln(2), got:\n{code}"
    );
    assert!(
        !code.contains(".ln()"),
        "should NOT contain bare .ln() when log2 is used, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Pattern 4: 2^x → exp2
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_exp2_pattern() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let f = two.pow(&x);
    let code = f.to_rust_fn("exp2_fn", &["x"]).unwrap();
    assert!(
        code.contains(".exp2()"),
        "expected .exp2() for 2^x, got:\n{code}"
    );
    assert!(
        !code.contains(".powf(") && !code.contains(".powi("),
        "should NOT contain powf/powi when exp2 is used, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Negative tests — must NOT trigger false positives
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_no_false_positive_exp() {
    // exp(x) - 2 should NOT use exp_m1
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() - 2;
    let code = f.to_rust_fn("no_expm1", &["x"]).unwrap();
    assert!(
        !code.contains("exp_m1"),
        "exp(x) - 2 should NOT use exp_m1, got:\n{code}"
    );
}

#[test]
fn codegen_no_false_positive_ln() {
    // ln(2 + x) should NOT use ln_1p
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let two = ctx.int(2);
    let f = (&x + &two).ln();
    let code = f.to_rust_fn("no_log1p", &["x"]).unwrap();
    assert!(
        !code.contains("ln_1p"),
        "ln(2 + x) should NOT use ln_1p, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Subexpression inside a larger expression
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_numopt_in_larger_expr() {
    // sin(x) + (exp(x) - 1) should use exp_m1 for the subexpression
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.sin() + x.exp() - 1;
    let code = f.to_rust_fn("mixed_fn", &["x"]).unwrap();
    assert!(
        code.contains("exp_m1"),
        "expected exp_m1 for the exp(x)-1 subexpression in sin(x)+exp(x)-1, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Backend: Libm
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn codegen_libm_backend_expm1() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.exp() - 1;
    let opts = CodegenOptions {
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code = f.to_rust_fn_with_options("libm_expm1", &["x"], &opts).unwrap();
    assert!(
        code.contains("libm::expm1("),
        "expected libm::expm1 with Libm backend, got:\n{code}"
    );
}
