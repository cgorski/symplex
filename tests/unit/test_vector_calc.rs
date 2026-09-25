//! Integration tests for Wave M: vector calculus operations.

use symplex::matrix::Matrix;
use symplex::prelude::*;
use symplex::vector::*;

// ═══════════════════════════════════════════════════════════════════════════
// Gradient
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gradient_of_x2_plus_y2() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f = expr!(ctx, x ^ 2 + y ^ 2);
    let grad = gradient(&f, &[&x, &y]).unwrap();
    // ∇(x² + y²) = [2x, 2y]
    assert_eq!(grad.nrows(), 2);
    assert_eq!(grad.ncols(), 1);
    assert_eq!(format!("{}", grad.get(0, 0)), "2*x");
    assert_eq!(format!("{}", grad.get(1, 0)), "2*y");
}

#[test]
fn gradient_of_xyz() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // f = x*y*z
    let f = expr!(ctx, x * y * z);
    let grad = gradient(&f, &[&x, &y, &z]).unwrap();
    // ∇(xyz) = [yz, xz, xy]
    assert_eq!(grad.nrows(), 3);
    assert_eq!(format!("{}", grad.get(0, 0)), "y*z");
    assert_eq!(format!("{}", grad.get(1, 0)), "x*z");
    assert_eq!(format!("{}", grad.get(2, 0)), "x*y");
}

#[test]
fn gradient_of_constant_is_zero() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let f = ctx.int(5);
    let grad = gradient(&f, &[&x, &y]).unwrap();
    assert!(grad.get(0, 0).is_zero_structural());
    assert!(grad.get(1, 0).is_zero_structural());
}

// ═══════════════════════════════════════════════════════════════════════════
// Divergence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn divergence_of_position_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // F = [x, y, z], ∇·F = 3
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]).unwrap();
    let div = divergence(&field, &[&x, &y, &z]).unwrap();
    assert_eq!(format!("{div}"), "3");
}

#[test]
fn divergence_of_quadratic_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    // F = [x², y²], ∇·F = 2x + 2y
    let field = Matrix::col_vector(vec![expr!(ctx, x ^ 2), expr!(ctx, y ^ 2)]).unwrap();
    let div = divergence(&field, &[&x, &y]).unwrap();
    let simplified = div.eval().simplify();
    let s = format!("{simplified}");
    assert!(s == "2*x + 2*y" || s == "2*y + 2*x", "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Curl
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn curl_of_position_is_zero() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]).unwrap();
    let c = curl(&field, &[&x, &y, &z]).unwrap();
    assert!(c.get(0, 0).eval().simplify().is_zero_structural());
    assert!(c.get(1, 0).eval().simplify().is_zero_structural());
    assert!(c.get(2, 0).eval().simplify().is_zero_structural());
    // Also verify via is_conservative
    assert_eq!(is_conservative(&field, &[&x, &y, &z]), Some(true));
}

#[test]
fn curl_of_rotation_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // F = [-y, x, 0], curl = [0, 0, 2]
    let field = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]).unwrap();
    let c = curl(&field, &[&x, &y, &z]).unwrap();
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
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    let f = expr!(ctx, x ^ 2 + y ^ 2 + z ^ 2);
    let lap = laplacian(&f, &[&x, &y, &z]).unwrap();
    assert_eq!(format!("{lap}"), "6");
}

#[test]
fn laplacian_of_linear_is_zero() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // f = 3x + 2y + z  →  ∇²f = 0
    let f = &(&ctx.int(3) * &x) + &(&(&ctx.int(2) * &y) + &z);
    let lap = laplacian(&f, &[&x, &y, &z]).unwrap();
    assert!(
        lap.eval().simplify().is_zero_structural(),
        "got: {}",
        lap.eval().simplify()
    );
}

#[test]
fn laplacian_of_x4() {
    let ctx = Context::new();
    symplex::syms!(ctx; x);
    // f = x^4, ∇²f = 12x²
    let f = expr!(ctx, x ^ 4);
    let lap = laplacian(&f, &[&x]).unwrap();
    let simplified = lap.eval().simplify();
    let s = format!("{simplified}");
    assert!(s == "12*x^2", "got: {s}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Conservative / solenoidal predicates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn conservative_gradient_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // Any gradient field is conservative: F = ∇(x²+y²+z²) = [2x,2y,2z]
    let f = expr!(ctx, x ^ 2 + y ^ 2 + z ^ 2);
    let field = gradient(&f, &[&x, &y, &z]).unwrap();
    assert_eq!(is_conservative(&field, &[&x, &y, &z]), Some(true));
}

#[test]
fn divergence_free_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // F = [y*z, x*z, x*y] is solenoidal (div = 0)
    let field = Matrix::col_vector(vec![&y * &z, &x * &z, &x * &y]).unwrap();
    assert_eq!(is_solenoidal(&field, &[&x, &y, &z]), Some(true));
}

#[test]
fn non_conservative_rotation_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // F = [-y, x, 0] has nonzero curl, so not conservative
    let field = Matrix::col_vector(vec![-&y, x.clone(), ctx.int(0)]).unwrap();
    assert_eq!(is_conservative(&field, &[&x, &y, &z]), Some(false));
}

#[test]
fn non_solenoidal_position_field() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y, z);
    // F = [x, y, z], div = 3, not solenoidal
    let field = Matrix::col_vector(vec![x.clone(), y.clone(), z.clone()]).unwrap();
    assert_eq!(is_solenoidal(&field, &[&x, &y, &z]), Some(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// Dimension mismatches (errors; they panicked before 0.29)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn divergence_dimension_mismatch_is_an_error() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let field = Matrix::col_vector(vec![x.clone(), ctx.int(1), ctx.int(2)]).unwrap();
    assert!(matches!(
        divergence(&field, &[&x, &y]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn curl_non_3d_is_an_error() {
    let ctx = Context::new();
    symplex::syms!(ctx; x, y);
    let field = Matrix::col_vector(vec![x.clone(), y.clone()]).unwrap();
    assert!(matches!(
        curl(&field, &[&x, &y]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}
