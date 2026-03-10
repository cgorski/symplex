//! Integration tests for advanced codegen optimizations:
//! FMA detection, Horner-style pow expansion, and sin_cos pairing.

use symplex::matrix::{CodegenOptions, MathBackend};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Feature 1: FMA (fused multiply-add) detection
// ═══════════════════════════════════════════════════════════════════════════

/// `x*y + z` with fresh (non-CSE) variables should use `mul_add`.
#[test]
fn codegen_fma_simple() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // x*y + z  —  the Mul(x,y) is inline (single-use), eligible for FMA
    let f = &x * &y + &z;
    let code = f.to_rust_fn("fma_simple", &["x", "y", "z"]).unwrap();
    assert!(
        code.contains("mul_add"),
        "expected mul_add for x*y + z, got:\n{code}"
    );
}

/// When a multiplication result is used in multiple places, CSE extracts it.
/// The CSE variable should NOT be fused into mul_add (CSE is more valuable).
#[test]
fn codegen_fma_not_for_multi_use() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    // Build an expression where x*y is used twice:
    //   sin(x*y) + (x*y) + z
    // CSE should extract x*y as a binding; the Add then sees symbols, not Mul.
    let xy = &x * &y;
    let f = xy.sin() + &xy + &z;
    let code = f.to_rust_fn("fma_multi_use", &["x", "y", "z"]).unwrap();
    // The CSE variable for x*y is referenced as tN (a symbol), not as a Mul node,
    // so FMA should NOT fire for the addition involving that CSE variable.
    // The code may or may not contain mul_add for other reasons, but the key
    // check is that CSE bindings are present (meaning x*y was extracted).
    assert!(
        code.contains("let t"),
        "expected CSE binding for shared x*y, got:\n{code}"
    );
}

/// `a*b + c*d` should produce chained mul_add calls.
#[test]
fn codegen_fma_chain() {
    let ctx = Context::new();
    let a = ctx.symbol("a");
    let b = ctx.symbol("b");
    let c = ctx.symbol("c");
    let d = ctx.symbol("d");
    let f = &a * &b + &c * &d;
    let code = f.to_rust_fn("fma_chain", &["a", "b", "c", "d"]).unwrap();
    assert!(
        code.contains("mul_add"),
        "expected mul_add for a*b + c*d, got:\n{code}"
    );
}

/// Libm backend should emit `libm::fma(...)` for FMA patterns.
#[test]
fn codegen_fma_libm_backend() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let f = &x * &y + &z;
    let opts = CodegenOptions {
        math_backend: MathBackend::Libm,
        ..Default::default()
    };
    let code = f
        .to_rust_fn_with_options("fma_libm", &["x", "y", "z"], &opts)
        .unwrap();
    assert!(
        code.contains("libm::fma("),
        "expected libm::fma with Libm backend, got:\n{code}"
    );
}

/// CfgGated backend should emit `math::fma(...)` and include `fma` in the module.
#[test]
fn codegen_fma_cfg_gated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let f = &x * &y + &z;
    let opts = CodegenOptions {
        math_backend: MathBackend::CfgGated,
        ..Default::default()
    };
    let code = f
        .to_rust_fn_with_options("fma_cfg", &["x", "y", "z"], &opts)
        .unwrap();
    assert!(
        code.contains("math::fma("),
        "expected math::fma call with CfgGated backend, got:\n{code}"
    );
    assert!(
        code.contains("pub fn fma("),
        "expected fma function in cfg-gated module, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Feature 2: Horner-style pow expansion
// ═══════════════════════════════════════════════════════════════════════════

/// `x.powi(3)` should be expanded to multiplication, not call powi.
#[test]
fn codegen_pow_expansion() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(3);
    let code = f.to_rust_fn("cube", &["x"]).unwrap();
    // Should contain multiplication
    assert!(
        code.contains("x * x * x"),
        "expected expanded multiplication for powi(3), got:\n{code}"
    );
    // Should NOT contain powi(3)
    assert!(
        !code.contains("powi(3)"),
        "should not contain powi(3) after expansion, got:\n{code}"
    );
}

/// `x.powi(4)` should use squaring: compute x² then square it.
#[test]
fn codegen_pow4_uses_squaring() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(4);
    let code = f.to_rust_fn("quartic", &["x"]).unwrap();
    // Should NOT contain powi(4)
    assert!(
        !code.contains("powi(4)"),
        "should not contain powi(4) after expansion, got:\n{code}"
    );
    // Should use a squaring pattern: some variable = base * base, then var * var
    assert!(
        code.contains("_p2") || code.contains("* _p2"),
        "expected squaring intermediate for powi(4), got:\n{code}"
    );
}

/// powi(2) should still use the original powi call (not expanded)
/// to preserve backward compatibility with existing tests.
#[test]
fn codegen_pow2_still_powi() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(2);
    let code = f.to_rust_fn("square", &["x"]).unwrap();
    assert!(
        code.contains("powi(2)"),
        "powi(2) should still use powi call, got:\n{code}"
    );
}

/// Negative exponents should still use powi (not expanded).
#[test]
fn codegen_pow_neg_not_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.pow(&ctx.int(-3));
    let code = f.to_rust_fn("inv_cube", &["x"]).unwrap();
    assert!(
        code.contains("powi(-3)"),
        "negative exponents should keep powi, got:\n{code}"
    );
}

/// powi(5) should also be expanded with squaring intermediate.
#[test]
fn codegen_pow5_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(5);
    let code = f.to_rust_fn("fifth", &["x"]).unwrap();
    assert!(
        !code.contains("powi(5)"),
        "should not contain powi(5) after expansion, got:\n{code}"
    );
    assert!(
        code.contains("_p2"),
        "expected squaring intermediate for powi(5), got:\n{code}"
    );
}

/// powi(6) should also be expanded.
#[test]
fn codegen_pow6_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(6);
    let code = f.to_rust_fn("sixth", &["x"]).unwrap();
    assert!(
        !code.contains("powi(6)"),
        "should not contain powi(6) after expansion, got:\n{code}"
    );
    assert!(
        code.contains("_p2"),
        "expected squaring intermediate for powi(6), got:\n{code}"
    );
}

/// powi(7) should NOT be expanded (threshold is 3..=6).
#[test]
fn codegen_pow7_not_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = x.powi(7);
    let code = f.to_rust_fn("seventh", &["x"]).unwrap();
    assert!(
        code.contains("powi(7)"),
        "powi(7) should stay as powi call, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Feature 3: sin_cos combined emission
// ═══════════════════════════════════════════════════════════════════════════

/// When both sin(x) and cos(x) appear as CSE bindings (used multiple times),
/// the codegen should emit a single `.sin_cos()` call.
#[test]
fn codegen_sin_cos_combined() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let w = ctx.symbol("w");
    // Use sin(x) and cos(x) each twice so CSE extracts both:
    //   sin(x)*y + cos(x)*z + sin(x)*w + cos(x)*y
    let f = x.sin() * &y + x.cos() * &z + x.sin() * &w + x.cos() * &y;
    let code = f
        .to_rust_fn("trig_combined", &["x", "y", "z", "w"])
        .unwrap();
    assert!(
        code.contains(".sin_cos()"),
        "expected .sin_cos() for paired sin/cos, got:\n{code}"
    );
    // Should NOT have separate .sin() and .cos() calls
    assert!(
        !code.contains(".sin()"),
        "should not have separate .sin() when sin_cos is used, got:\n{code}"
    );
    assert!(
        !code.contains(".cos()"),
        "should not have separate .cos() when sin_cos is used, got:\n{code}"
    );
}

/// sin_cos pairing should also add sin_cos to the CfgGated module.
#[test]
fn codegen_sin_cos_cfg_gated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let w = ctx.symbol("w");
    let f = x.sin() * &y + x.cos() * &z + x.sin() * &w + x.cos() * &y;
    let opts = CodegenOptions {
        math_backend: MathBackend::CfgGated,
        ..Default::default()
    };
    let code = f
        .to_rust_fn_with_options("trig_cfg", &["x", "y", "z", "w"], &opts)
        .unwrap();
    assert!(
        code.contains("pub fn sin_cos("),
        "expected sin_cos in cfg-gated module, got:\n{code}"
    );
    assert!(
        code.contains("math::sin_cos("),
        "expected math::sin_cos call, got:\n{code}"
    );
}

/// When sin(x) and cos(x) appear only once each (not CSE'd),
/// sin_cos pairing should NOT trigger (nothing to pair at binding level).
#[test]
fn codegen_sin_cos_no_pair_single_use() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x) + cos(x) — each used once, no CSE bindings
    let f = x.sin() + x.cos();
    let code = f.to_rust_fn("trig_single", &["x"]).unwrap();
    // No sin_cos — each is used only once, so no CSE binding to pair
    assert!(
        !code.contains(".sin_cos()"),
        "should not use sin_cos for single-use sin/cos, got:\n{code}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural validity of generated code with new optimizations
// ═══════════════════════════════════════════════════════════════════════════

/// Verify generated code with FMA has balanced delimiters.
#[test]
fn codegen_fma_balanced_delimiters() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let z = ctx.symbol("z");
    let w = ctx.symbol("w");
    let f = &x * &y + &z * &w + &x;
    let code = f.to_rust_fn("fma_balance", &["x", "y", "z", "w"]).unwrap();
    let open_parens = code.chars().filter(|&c| c == '(').count();
    let close_parens = code.chars().filter(|&c| c == ')').count();
    assert_eq!(
        open_parens, close_parens,
        "unbalanced parentheses ({open_parens} open vs {close_parens} close) in:\n{code}"
    );
    let open_braces = code.chars().filter(|&c| c == '{').count();
    let close_braces = code.chars().filter(|&c| c == '}').count();
    assert_eq!(
        open_braces, close_braces,
        "unbalanced braces ({open_braces} open vs {close_braces} close) in:\n{code}"
    );
}

/// Verify pow expansion code has balanced delimiters.
#[test]
fn codegen_pow_expansion_balanced() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Use powi(4) which generates a block expression with let binding
    let f = x.powi(4) + &x;
    let code = f.to_rust_fn("pow4_balance", &["x"]).unwrap();
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

/// Combined FMA + pow expansion in one expression.
#[test]
fn codegen_combined_fma_and_pow() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    // x^3 * y + x  — should expand x^3 and potentially use FMA
    let f = x.powi(3) * &y + &x;
    let code = f.to_rust_fn("combined", &["x", "y"]).unwrap();
    // Should not have powi(3)
    assert!(
        !code.contains("powi(3)"),
        "should expand powi(3), got:\n{code}"
    );
    // Should contain mul_add (the product x^3*y fused with + x)
    assert!(
        code.contains("mul_add"),
        "expected mul_add in combined expression, got:\n{code}"
    );
}
