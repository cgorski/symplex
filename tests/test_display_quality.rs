//! Display quality tests — verify clean subtraction rendering and LaTeX output.
//!
//! These tests ensure that the Display impl never produces ugly patterns like
//! `+ -` and that negative terms are rendered with proper subtraction syntax.

use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. x + (-3) displays as "x - 3"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_sub_neg_int() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let expr = &x + __ctx.int(-3);
    let s = format!("{expr}");
    assert!(
        s.contains("- 3"),
        "expected 'x - 3' pattern, got: {s}"
    );
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. x + (-1/2) displays as "x - 1/2"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_sub_neg_frac() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let expr = &x + __ctx.rational(-1, 2);
    let s = format!("{expr}");
    assert!(
        s.contains("- 1/2"),
        "expected '- 1/2' pattern, got: {s}"
    );
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. x - y displays as "x - y" (no "+ -")
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_sub_neg_symbol() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let expr = &x - &y;
    let s = format!("{expr}");
    assert!(
        s.contains("- y"),
        "expected '- y' pattern, got: {s}"
    );
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. -3 + x displays correctly (canonical ordering should give "x - 3")
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_leading_neg_int() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let expr = __ctx.int(-3) + &x;
    let s = format!("{expr}");
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
    // The display should have a subtraction for the negative constant
    assert!(
        s.contains("- 3"),
        "expected '- 3' somewhere in output, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. (-x) + (-y) displays as "-x - y"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_double_neg() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let expr = -&x + (-&y);
    let s = format!("{expr}");
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
    // Should have a leading minus and a subtraction
    assert!(
        s.contains('-'),
        "expected at least one minus sign, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. x + (-2*y) displays as "x - 2*y"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_neg_mul_coeff() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let expr = &x + __ctx.int(-2) * &y;
    let s = format!("{expr}");
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
    assert!(
        s.contains("- 2*y") || s.contains("- 2y"),
        "expected subtraction of 2*y term, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Build 10 expressions with negative terms — NONE contain "+ -"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_no_plus_minus_anywhere() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);

    let exprs: Vec<Ex> = vec![
        // 1: x + (-3)
        &x + __ctx.int(-3),
        // 2: x - y
        &x - &y,
        // 3: x + (-1/2)
        &x + __ctx.rational(-1, 2),
        // 4: (-x) + (-y)
        -&x + (-&y),
        // 5: x + (-2*y)
        &x + __ctx.int(-2) * &y,
        // 6: (-3) + x + (-y)
        __ctx.int(-3) + &x + (-&y),
        // 7: x + y + (-z)
        &x + &y - &z,
        // 8: (-x) + y
        -&x + &y,
        // 9: x + (-5/3)
        &x + __ctx.rational(-5, 3),
        // 10: x + (-1)*y + (-2)*z
        &x + __ctx.int(-1) * &y + __ctx.int(-2) * &z,
    ];

    for (i, e) in exprs.iter().enumerate() {
        let s = format!("{e}");
        assert!(
            !s.contains("+ -"),
            "expression {} contains '+ -': {s}",
            i + 1
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. -1 * x displays as "-x", not "-1*x"
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_mul_neg_one_invisible() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    let expr = __ctx.int(-1) * &x;
    let s = format!("{expr}");
    assert_eq!(s, "-x", "expected '-x', got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. (-3/4)*x + y displays cleanly (no "+ -")
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_neg_fraction_coeff() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let expr = __ctx.rational(-3, 4) * &x + &y;
    let s = format!("{expr}");
    assert!(
        !s.contains("+ -"),
        "must not contain '+ -', got: {s}"
    );
    // y should appear without a negative prefix, and the fraction term
    // should use subtraction syntax
    assert!(
        s.contains('y'),
        "expected 'y' in output, got: {s}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Matrix::to_latex() produces valid LaTeX with bmatrix environment
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_to_latex_works() {
    let __ctx = Context::new();
    let m = matrix![[1, 2], [3, 4]];
    let latex = m.to_latex();
    assert!(
        latex.contains("\\begin{bmatrix}"),
        "expected \\begin{{bmatrix}} in LaTeX output, got: {latex}"
    );
    assert!(
        latex.contains("\\end{bmatrix}"),
        "expected \\end{{bmatrix}} in LaTeX output, got: {latex}"
    );
}
