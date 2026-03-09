//! Vector calculus operations: gradient, divergence, curl, laplacian.
//!
//! These operate on symbolic scalar and vector fields expressed as
//! `Ex` (scalar) or `Matrix` (column vectors of `Ex`).

use crate::api::expr::Ex;
use crate::domains::matrix::Matrix;

/// Gradient of a scalar field: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
///
/// Returns an n×1 column vector of partial derivatives.
pub fn gradient(f: &Ex, vars: &[&Ex]) -> Matrix {
    let partials: Vec<Ex> = vars.iter().map(|v| f.diff(v)).collect();
    Matrix::col_vector(partials)
}

/// Divergence of a vector field: ∇·F = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ... + ∂Fₙ/∂xₙ
///
/// `field` is an n×1 column vector, `vars` has n variables.
pub fn divergence(field: &Matrix, vars: &[&Ex]) -> Ex {
    assert_eq!(
        field.nrows(),
        vars.len(),
        "field dimension ({}) must match variable count ({})",
        field.nrows(),
        vars.len()
    );
    assert_eq!(field.ncols(), 1, "field must be a column vector");

    let mut sum = vars[0].context().zero();
    for (i, var) in vars.iter().enumerate() {
        let component = field.get(i, 0);
        sum = &sum + &component.diff(var);
    }
    sum
}

/// Curl of a 3D vector field: ∇×F
///
/// `field` is a 3×1 column vector, `vars` is [x, y, z].
/// Returns a 3×1 column vector.
pub fn curl(field: &Matrix, vars: &[&Ex]) -> Matrix {
    assert_eq!(field.nrows(), 3, "curl requires 3D vector field");
    assert_eq!(vars.len(), 3, "curl requires 3 variables");
    assert_eq!(field.ncols(), 1, "field must be a column vector");

    let f1 = field.get(0, 0);
    let f2 = field.get(1, 0);
    let f3 = field.get(2, 0);
    let (x, y, z) = (vars[0], vars[1], vars[2]);

    // curl = [∂F3/∂y - ∂F2/∂z, ∂F1/∂z - ∂F3/∂x, ∂F2/∂x - ∂F1/∂y]
    let c1 = &f3.diff(y) - &f2.diff(z);
    let c2 = &f1.diff(z) - &f3.diff(x);
    let c3 = &f2.diff(x) - &f1.diff(y);

    Matrix::col_vector(vec![c1, c2, c3])
}

/// Laplacian of a scalar field: ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ...
pub fn laplacian(f: &Ex, vars: &[&Ex]) -> Ex {
    let grad = gradient(f, vars);
    divergence(&grad, vars)
}

/// Test if a vector field is conservative (curl-free).
/// Only defined for 3D fields.
pub fn is_conservative(field: &Matrix, vars: &[&Ex]) -> bool {
    if field.nrows() != 3 || vars.len() != 3 || field.ncols() != 1 {
        return false;
    }
    let c = curl(field, vars);
    // Check if all components simplify to zero
    for i in 0..3 {
        let comp = c.get(i, 0);
        if !comp.eval().simplify().is_zero_structural() {
            return false;
        }
    }
    true
}

/// Test if a vector field is solenoidal (divergence-free).
pub fn is_solenoidal(field: &Matrix, vars: &[&Ex]) -> bool {
    let div = divergence(field, vars);
    div.eval().simplify().is_zero_structural()
}
