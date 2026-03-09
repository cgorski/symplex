//! Integration tests for Wave M: vector calculus operations.

use symplex::matrix::Matrix;
use symplex::prelude::*;
use symplex::vector::*;

// ═══════════════════════════════════════════════════════════════════════════
// Gradient
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gradient_of_x2_plus_y2() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let f = expr!(x ^ 2 + y ^ 2);
    let grad = gradient(&f, &[&x, &y]);
    // ∇(x² + y²) = [2x, 2y]
    assert_eq!(grad.nrows(), 2);
    assert_eq!(grad.ncols(), 1);
    assert_eq!(format!("{}", grad.get(0, 0)), "2*x");
    assert_eq!(format!("{}", grad.get(1, 0)), "2*y");
}

#[test]
fn gradient_of_xyz() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // f = x*y*z
    let f = expr!(x * y * z);
    let grad = gradient(&f, &[&x, &y, &z]);
    // ∇(xyz) = [yz, xz, xy]
    assert_eq!(grad.nrows(), 3);
    assert_eq!(format!("{}", grad.get(0, 0)), "y*z");
    assert_eq!(format!("{}", grad.get(1, 0)), "x*z");
    assert_eq!(format!("{}", grad.get(2, 0)), "x*y");
}

#[test]
fn gradient_of_constant_is_zero() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let f = __ctx.int(5);
    let grad = gradient(&f, &[&x, &y]);
    assert!(grad.get(0, 0).is_zero_structural());
    assert!(grad.get(1, 0).is_zero_structural());
}

// ═══════════════════════════════════════════════════════════════════════════
// Divergence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn divergence_of_position_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // F = [x, y, z], ∇·F = 3
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]);
    let div = divergence(&field, &[&x, &y, &z]);
    assert_eq!(format!("{div}"), "3");
}

#[test]
fn divergence_of_quadratic_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    // F = [x², y²], ∇·F = 2x + 2y
    let field = Matrix::col_vector(vec![expr!(x ^ 2), expr!(y ^ 2)]);
    let div = divergence(&field, &[&x, &y]);
    let simplified = div.eval().simplify();
    let s = format!("{simplified}");
    assert!(s == "2*x + 2*y" || s == "2*y + 2*x", "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Curl
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn curl_of_position_is_zero() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]);
    let c = curl(&field, &[&x, &y, &z]);
    assert!(c.get(0, 0).eval().simplify().is_zero_structural());
    assert!(c.get(1, 0).eval().simplify().is_zero_structural());
    assert!(c.get(2, 0).eval().simplify().is_zero_structural());
    // Also verify via is_conservative
    assert!(is_conservative(&field, &[&x, &y, &z]));
}

#[test]
fn curl_of_rotation_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // F = [-y, x, 0], curl = [0, 0, 2]
    let field = Matrix::col_vector(vec![-&y, x.clone(), __ctx.int(0)]);
    let c = curl(&field, &[&x, &y, &z]);
    let c0 = c.get(0, 0).eval().simplify();
    let c1 = c.get(1, 0).eval().simplify();
    let c2 = c.get(2, 0).eval().simplify();
    assert!(c0.is_zero_structural(), "c0 = {c0}");
    assert!(c1.is_zero_structural(), "c1 = {c1}");
    assert_eq!(format!("{c2}"), "2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplacian
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn laplacian_of_x2_y2_z2() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    let f = expr!(x ^ 2 + y ^ 2 + z ^ 2);
    let lap = laplacian(&f, &[&x, &y, &z]);
    assert_eq!(format!("{lap}"), "6");
}

#[test]
fn laplacian_of_linear_is_zero() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // f = 3x + 2y + z  →  ∇²f = 0
    let f = &(&__ctx.int(3) * &x) + &(&(&__ctx.int(2) * &y) + &z);
    let lap = laplacian(&f, &[&x, &y, &z]);
    assert!(
        lap.eval().simplify().is_zero_structural(),
        "got: {}",
        lap.eval().simplify()
    );
}

#[test]
fn laplacian_of_x4() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x);
    // f = x^4, ∇²f = 12x²
    let f = expr!(x ^ 4);
    let lap = laplacian(&f, &[&x]);
    let simplified = lap.eval().simplify();
    let s = format!("{simplified}");
    assert!(s == "12*x^2", "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Conservative / solenoidal predicates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn conservative_gradient_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // Any gradient field is conservative: F = ∇(x²+y²+z²) = [2x,2y,2z]
    let f = expr!(x ^ 2 + y ^ 2 + z ^ 2);
    let field = gradient(&f, &[&x, &y, &z]);
    assert!(is_conservative(&field, &[&x, &y, &z]));
}

#[test]
fn divergence_free_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // F = [y*z, x*z, x*y] is solenoidal (div = 0)
    let field = Matrix::col_vector(vec![&y * &z, &x * &z, &x * &y]);
    assert!(is_solenoidal(&field, &[&x, &y, &z]));
}

#[test]
fn non_conservative_rotation_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // F = [-y, x, 0] has nonzero curl, so not conservative
    let field = Matrix::col_vector(vec![-&y, x.clone(), __ctx.int(0)]);
    assert!(!is_conservative(&field, &[&x, &y, &z]));
}

#[test]
fn non_solenoidal_position_field() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y, z);
    // F = [x, y, z], div = 3, not solenoidal
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]);
    assert!(!is_solenoidal(&field, &[&x, &y, &z]));
}

// ═══════════════════════════════════════════════════════════════════════════
// Dimension mismatches (should panic)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected = "field dimension")]
fn divergence_dimension_mismatch_panics() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let field = Matrix::col_vector(vec![x.clone(), __ctx.int(1), __ctx.int(2)]);
    let _ = divergence(&field, &[&x, &y]);
}

#[test]
#[should_panic(expected = "curl requires 3D")]
fn curl_non_3d_panics() {
    let __ctx = Context::new();
    let __vars_ctx = __ctx.clone(); symplex::syms!(__vars_ctx; x, y);
    let field = Matrix::col_vector(vec![x.clone(), y.clone()]);
    let _ = curl(&field, &[&x, &y]);
}
