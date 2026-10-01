//! Matrix and vector norms over ℂ: `|a|²`, not `a²`.

use symplex::matrix::dot;
use symplex::prelude::*;

fn close(z: Complex64, want: f64) -> bool {
    (z - Complex64::new(want, 0.0)).norm() <= 1e-12 * want.max(1.0)
}

/// `Matrix::norm()` (the Frobenius norm) squared every entry: for
/// `[[i, 0], [0, 2]]` it summed `i² + 2² = 3` and returned `√3`, and for a
/// complex matrix in general a complex "norm".  Now each term is `|a|²`.
///
/// SymPy: `Matrix([[I, 0], [0, 2]]).norm()` → `sqrt(5)`;
/// `Matrix([[1 + I, 2], [3, 4 - I]]).norm()` → `4*sqrt(2)`;
/// `Matrix([1 + 2*I, 2 - I, 3]).norm()` → `sqrt(19)`.
/// numpy: `np.linalg.norm(np.array([[1+1j, 2], [3, 4-1j]]))` →
/// `5.656854249492381`.
#[test]
fn frobenius_norm_of_complex_matrices() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let m = Matrix::new(vec![
        vec![i.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    assert_eq!(m.norm(), ctx.int(5).sqrt());
    assert_eq!(m.norm_frobenius(), ctx.int(5).sqrt());

    let m = Matrix::new(vec![
        vec![&ctx.int(1) + &i, ctx.int(2)],
        vec![ctx.int(3), &ctx.int(4) - &i],
    ])
    .unwrap();
    assert_eq!(m.norm(), &ctx.int(2).sqrt() * 4);
    assert!(close(m.norm().eval_complex64().unwrap(), 5.656854249492381));

    let v =
        Matrix::col_vector(vec![&ctx.int(1) + &(&i * 2), &ctx.int(2) - &i, ctx.int(3)]).unwrap();
    assert_eq!(v.norm(), ctx.int(19).sqrt());
}

/// A symbol without assumptions may be complex: the norm of `[[a, b]]`
/// was `√(a² + b²)`, which at `a = 1 + 2i`, `b = 3` is `√6` instead of
/// `√14`.  Real symbols keep `a²` (the form real matrices always had).
///
/// SymPy: `Matrix([[a, b]]).norm()` → `sqrt(Abs(a)**2 + Abs(b)**2)`;
/// with `real=True` symbols → `sqrt(a**2 + b**2)`;
/// `Matrix([[x + I*y]]).norm()` for real `x`, `y` → `sqrt(x**2 + y**2)`.
#[test]
fn frobenius_norm_of_symbols_follows_sympy() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let v = Matrix::new(vec![vec![a.clone(), b.clone()]]).unwrap();
    let n = v.norm();
    assert_eq!(n, (&a.abs().powi(2) + &b.abs().powi(2)).sqrt());
    let at = n
        .subs(&a, &(&ctx.int(1) + &(&ctx.i_unit() * 2)))
        .subs(&b, &ctx.int(3));
    assert!(close(at.eval_complex64().unwrap(), 14f64.sqrt()));

    let x = ctx.symbol_with("x", &[Assumption::Real]).unwrap();
    let y = ctx.symbol_with("y", &[Assumption::Real]).unwrap();
    let v = Matrix::new(vec![vec![x.clone(), y.clone()]]).unwrap();
    assert_eq!(v.norm(), (&x.powi(2) + &y.powi(2)).sqrt());
    let z = Matrix::new(vec![vec![&x + &(&ctx.i_unit() * &y)]]).unwrap();
    assert_eq!(z.norm(), (&x.powi(2) + &y.powi(2)).sqrt());
}

/// Real matrices keep their exact form (the byte-identity contract):
/// `‖[[1, 2], [3, 4]]‖_F = √30`, `‖[3, 4]‖ = 5`, `‖[√2, π]‖ = √(π² + 2)`.
///
/// SymPy: `Matrix([[1, 2], [3, 4]]).norm()` → `sqrt(30)`;
/// `Matrix([[sqrt(2), pi]]).norm()` → `sqrt(2 + pi**2)`.
#[test]
fn frobenius_norm_of_real_matrices_unchanged() {
    let ctx = Context::new();
    assert_eq!(matrix![ctx, [1, 2], [3, 4]].norm(), ctx.int(30).sqrt());
    assert_eq!(matrix![ctx, [3, 4]].norm(), ctx.int(5));
    let v = Matrix::new(vec![vec![ctx.int(2).sqrt(), ctx.pi()]]).unwrap();
    assert_eq!(v.norm(), (&ctx.pi().powi(2) + 2).sqrt());
}

/// The other norms already took `|a|` (audited with the Frobenius fix):
/// the 1- and ∞-norms, the vector p-norm and the 2-norm condition number
/// (Hermitian Gram matrix since 0.32).
///
/// numpy: `np.linalg.norm(A, 1)` → `6.123105625617661`,
/// `np.linalg.norm(A, np.inf)` → `7.123105625617661`,
/// `np.linalg.cond(A)` → `10.01948296332431` for `A = [[1+1j, 2], [3, 4-1j]]`;
/// `np.linalg.cond([[1j, 0], [0, 2]])` → `2.0`;
/// `np.linalg.norm([1+1j, 2-3j], 3)` → `3.676663290298976`.
#[test]
fn other_norms_of_complex_matrices() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let a = Matrix::new(vec![
        vec![&ctx.int(1) + &i, ctx.int(2)],
        vec![ctx.int(3), &ctx.int(4) - &i],
    ])
    .unwrap();
    assert!(close(
        a.norm_1().eval_complex64().unwrap(),
        6.123105625617661
    ));
    assert!(close(
        a.norm_inf().eval_complex64().unwrap(),
        7.123105625617661
    ));
    let k = a.condition_number().unwrap().eval_complex64().unwrap();
    assert!((k - Complex64::new(10.01948296332431, 0.0)).norm() < 1e-12);
    let d = Matrix::new(vec![
        vec![i.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(2)],
    ])
    .unwrap();
    assert_eq!(d.condition_number().unwrap(), ctx.int(2));
    let v = Matrix::col_vector(vec![&ctx.int(1) + &i, &ctx.int(2) - &(&i * 3)]).unwrap();
    let p3 = v.norm_p(&ctx.int(3)).unwrap().eval_complex64().unwrap();
    assert!(close(p3, 3.676663290298976));
}

/// `dot` stays bilinear, as SymPy's default — it is documented now, since
/// `dot(v, v)` is not `‖v‖²` for a complex `v`.
///
/// SymPy: `Matrix([1, I]).dot(Matrix([1, I]))` → `0`;
/// `Matrix([1, I]).dot(Matrix([1, I]), hermitian=True,
/// conjugate_convention='left')` → `2`.
#[test]
fn dot_is_bilinear() {
    let ctx = Context::new();
    let v = Matrix::col_vector(vec![ctx.int(1), ctx.i_unit()]).unwrap();
    assert_eq!(dot(&v, &v).unwrap(), ctx.int(0));
    assert_eq!(dot(&v.map(Ex::conjugate), &v).unwrap(), ctx.int(2));
    assert_eq!(v.norm(), ctx.int(2).sqrt());
}
