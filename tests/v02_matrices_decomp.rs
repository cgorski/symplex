//! symplex 0.2 — decompositions and structure tests (QR, Gram–Schmidt,
//! Cholesky, LDLᵀ, definiteness, norms, matrix functions, subspaces).
//!
//! Every decomposition is verified by reconstruction both symbolically
//! (`simplify`) and numerically.

mod common;

use symplex::matrix::{Matrix, dot};
use symplex::matrix_decomp::{gram_schmidt, hessian, wronskian};
use symplex::prelude::*;

fn mi(ctx: &Context, rows: &[&[i64]]) -> Matrix {
    Matrix::from_i64(ctx, rows).unwrap()
}

fn assert_close_matrix(a: &Matrix, b: &Matrix, tol: f64, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let av = a.get(i, j).simplify().eval_f64().unwrap();
            let bv = b.get(i, j).simplify().eval_f64().unwrap();
            assert!(
                common::approx_eq(av, bv, tol),
                "{label}: ({i},{j}) {av} vs {bv}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QR / Gram–Schmidt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qr_square_symbolic_reconstruction() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 1, 0], &[1, 0, 1], &[0, 1, 1]]);
    let (q, r) = a.qr().unwrap();
    // Exact reconstruction after simplification
    assert_eq!((&q * &r).simplify(), a);
    assert_eq!((&q.transpose() * &q).simplify(), Matrix::identity(&ctx, 3));
    assert_eq!(r.is_upper_triangular(), Some(true));
    // R has positive diagonal
    for i in 0..3 {
        assert!(r.get(i, i).eval_f64().unwrap() > 0.0);
    }
    // Numerical check too
    assert_close_matrix(&(&q * &r), &a, 1e-12, "QR numeric");
    // |det A| = Π r_ii
    let prod: f64 = (0..3).map(|i| r.get(i, i).eval_f64().unwrap()).product();
    assert!(common::approx_eq(
        prod,
        a.det().unwrap().eval_f64().unwrap().abs(),
        1e-12
    ));
}

#[test]
fn qr_rectangular_and_radicals() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2], &[3, 4], &[5, 6]]);
    let (q, r) = a.qr().unwrap();
    assert_eq!(q.shape(), (3, 2));
    assert_eq!(r.shape(), (2, 2));
    assert_eq!((&q * &r).simplify(), a);
    assert_eq!((&q.transpose() * &q).simplify(), Matrix::identity(&ctx, 2));
    // r_00 = ‖(1,3,5)‖ = √35 exactly
    assert_eq!(r.get(0, 0).eval(), ctx.int(35).sqrt().eval());
    // Simple radical case: [[1, 1], [1, -1]] → Q = (1/√2)[[1, 1], [1, -1]], R = √2 I
    let h = mi(&ctx, &[&[1, 1], &[1, -1]]);
    let (q, r) = h.qr().unwrap();
    assert_eq!(q.get(0, 0).powi(2).simplify(), ctx.rational(1, 2));
    assert_eq!(r.get(0, 0).eval(), ctx.int(2).sqrt().eval());
    assert!(r.get(0, 1).is_zero_structural());
}

#[test]
fn qr_symbolic_entries() {
    let ctx = Context::new();
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let a = Matrix::new(vec![
        vec![p.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    let (q, r) = a.qr().unwrap();
    assert_eq!(q.simplify(), Matrix::identity(&ctx, 2));
    assert_eq!(r.simplify(), a);
}

#[test]
fn qr_dependent_columns_err() {
    let ctx = Context::new();
    let err = mi(&ctx, &[&[1, 2], &[2, 4]]).qr().unwrap_err();
    assert!(matches!(err, SymplexError::ComputationFailed { .. }));
    assert!(err.to_string().contains("dependent"));
}

#[test]
fn gram_schmidt_orthogonal_and_normalized() {
    let ctx = Context::new();
    let vs = vec![
        matrix![ctx, [1], [1], [1]],
        matrix![ctx, [1], [0], [0]],
        matrix![ctx, [0], [1], [0]],
    ];
    let q = gram_schmidt(&vs, true).unwrap();
    assert_eq!(q.len(), 3);
    for i in 0..3 {
        assert_eq!(dot(&q[i], &q[i]).simplify(), ctx.int(1), "‖q{i}‖ = 1");
        for j in (i + 1)..3 {
            assert_eq!(dot(&q[i], &q[j]).simplify(), ctx.int(0), "q{i}·q{j} = 0");
        }
    }
    let u = gram_schmidt(&vs, false).unwrap();
    assert_eq!(u[0], vs[0]);
    assert_eq!(dot(&u[0], &u[1]).simplify(), ctx.int(0));
    // Dependent → Err; shape errors → Err
    let dep = vec![vs[0].clone(), &vs[0] * 3];
    assert!(gram_schmidt(&dep, true).is_err());
    assert!(gram_schmidt(&[vs[0].clone(), matrix![ctx, [1], [2]]], true).is_err());
    assert!(gram_schmidt(&[matrix![ctx, [1, 2]]], true).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Cholesky / LDLᵀ / LU
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cholesky_result_semantics() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[25, 15, -5], &[15, 18, 0], &[-5, 0, 11]]);
    let l = a.cholesky().unwrap();
    assert_eq!(l, mi(&ctx, &[&[5, 0, 0], &[3, 3, 0], &[-1, 1, 3]]));
    assert_eq!((&l * &l.transpose()).eval(), a);
    // Indefinite / non-symmetric / undecidable
    assert!(mi(&ctx, &[&[1, 2], &[2, 1]]).cholesky().is_err());
    assert!(mi(&ctx, &[&[1, 2], &[0, 1]]).cholesky().is_err());
    let x = ctx.symbol("x");
    let undecidable = Matrix::new(vec![
        vec![x.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    let err = undecidable.cholesky().unwrap_err();
    assert!(err.to_string().contains("cannot decide"), "{err}");
    // Assumptions unlock symbolic Cholesky
    let a_pos = ctx.symbol_with("a", &[Assumption::Positive]);
    let sym = Matrix::new(vec![
        vec![a_pos.clone(), ctx.int(0)],
        vec![ctx.int(0), &a_pos * 4],
    ])
    .unwrap();
    let l = sym.cholesky().unwrap();
    assert_eq!((&l * &l.transpose()).simplify(), sym);
}

#[test]
fn ldl_symbolic_and_numeric() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[4, 12, -16], &[12, 37, -43], &[-16, -43, 98]]);
    let (l, d) = a.ldl().unwrap();
    assert_eq!(l.is_lower_triangular(), Some(true));
    assert_eq!(d.is_diagonal(), Some(true));
    assert_eq!((&(&l * &d) * &l.transpose()).eval(), a);
    // Indefinite works (no square roots needed)
    let ind = mi(&ctx, &[&[1, 2], &[2, 1]]);
    let (l, d) = ind.ldl().unwrap();
    assert_eq!(d, mi(&ctx, &[&[1, 0], &[0, -3]]));
    assert_eq!((&(&l * &d) * &l.transpose()).eval(), ind);
    // Fully symbolic symmetric
    let (a, b, c) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
    let s = Matrix::new(vec![vec![a.clone(), b.clone()], vec![b.clone(), c.clone()]]).unwrap();
    let (l, d) = s.ldl().unwrap();
    assert_eq!(l.get(1, 0), &(&b / &a));
    let back = (&(&l * &d) * &l.transpose()).simplify();
    assert_eq!(back.equals(&s), Some(true));
}

#[test]
fn lu_reconstruction_with_permutation() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[0, 2, 1], &[1, 1, 1], &[2, 1, 3]]);
    let (l, u, perm) = a.lu().unwrap();
    assert_ne!(perm[0], 0, "first pivot needs a row swap");
    let pa = Matrix::new(perm.iter().map(|&i| a.row(i).to_vec()).collect()).unwrap();
    assert_eq!((&l * &u).eval(), pa);
    assert_eq!(l.is_lower_triangular(), Some(true));
    assert_eq!(u.is_upper_triangular(), Some(true));
    for i in 0..3 {
        assert!(l.get(i, i).is_one_structural());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Structure predicates (three-valued)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn predicates_three_valued() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sym = mi(&ctx, &[&[1, 2], &[2, 3]]);
    assert_eq!(sym.is_symmetric(), Some(true));
    assert_eq!(sym.is_hermitian(), Some(true));
    assert_eq!(mi(&ctx, &[&[1, 2], &[3, 4]]).is_symmetric(), Some(false));
    let unknown = Matrix::new(vec![
        vec![ctx.int(1), x.clone()],
        vec![ctx.int(2), ctx.int(1)],
    ])
    .unwrap();
    assert_eq!(unknown.is_symmetric(), None);
    let trig = Matrix::new(vec![
        vec![ctx.int(0), &x.sin().powi(2) + &x.cos().powi(2)],
        vec![ctx.int(1), ctx.int(0)],
    ])
    .unwrap();
    assert_eq!(trig.is_symmetric(), Some(true));

    let i = ctx.i_unit();
    let herm = Matrix::new(vec![
        vec![ctx.int(2), &ctx.int(3) * &i],
        vec![&ctx.int(-3) * &i, ctx.int(1)],
    ])
    .unwrap();
    assert_eq!(herm.is_hermitian(), Some(true));
    assert_eq!(herm.is_symmetric(), Some(false));
    assert_eq!(herm.adjoint(), herm);

    // Orthogonality needs no realness; unitarity conjugates the entries, so
    // for a possibly-complex `x` it is undecidable and for real `x` it holds.
    let rot = Matrix::new(vec![vec![x.cos(), -&x.sin()], vec![x.sin(), x.cos()]]).unwrap();
    assert_eq!(rot.is_orthogonal(), Some(true));
    assert_eq!(rot.is_unitary(), None);
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    let rot_real = Matrix::new(vec![vec![r.cos(), -&r.sin()], vec![r.sin(), r.cos()]]).unwrap();
    assert_eq!(rot_real.is_orthogonal(), Some(true));
    assert_eq!(rot_real.is_unitary(), Some(true));
    let phase = Matrix::new(vec![vec![i.clone(), ctx.int(0)], vec![ctx.int(0), -&i]]).unwrap();
    assert_eq!(phase.is_unitary(), Some(true));
    assert_eq!(phase.is_orthogonal(), Some(false));

    let skew = mi(&ctx, &[&[0, 2, -1], &[-2, 0, 3], &[1, -3, 0]]);
    assert_eq!(skew.is_skew_symmetric(), Some(true));
    assert_eq!(sym.is_skew_symmetric(), Some(false));

    let up = mi(&ctx, &[&[1, 2, 3], &[0, 4, 5], &[0, 0, 6]]);
    assert_eq!(up.is_upper_triangular(), Some(true));
    assert_eq!(up.is_lower_triangular(), Some(false));
    assert_eq!(up.is_diagonal(), Some(false));
    assert_eq!(
        Matrix::diag(&[x.clone(), ctx.int(2)]).is_diagonal(),
        Some(true)
    );
    assert_eq!(Matrix::identity(&ctx, 4).is_identity(), Some(true));
    assert_eq!(Matrix::zeros(&ctx, 3, 2).is_zero(), Some(true));
    assert_eq!(Matrix::zeros(&ctx, 3, 2).is_identity(), Some(false));
    let nil = mi(&ctx, &[&[0, 1, 2], &[0, 0, 3], &[0, 0, 0]]);
    assert_eq!(nil.is_nilpotent(), Some(true));
    assert_eq!(up.is_nilpotent(), Some(false));
    assert!(nil.matrix_exp().is_ok());
}

#[test]
fn definiteness_sylvester() {
    let ctx = Context::new();
    assert_eq!(
        mi(&ctx, &[&[2, -1, 0], &[-1, 2, -1], &[0, -1, 2]]).is_positive_definite(),
        Some(true)
    );
    assert_eq!(
        mi(&ctx, &[&[1, 2], &[2, 1]]).is_positive_definite(),
        Some(false)
    );
    assert_eq!(
        mi(&ctx, &[&[1, 2], &[3, 4]]).is_positive_definite(),
        Some(false)
    ); // not symmetric
    assert_eq!(
        mi(&ctx, &[&[1, 1], &[1, 1]]).is_positive_definite(),
        Some(false)
    );
    assert_eq!(
        mi(&ctx, &[&[1, 1], &[1, 1]]).is_positive_semidefinite(),
        Some(true)
    );
    assert_eq!(
        mi(&ctx, &[&[0, 0], &[0, -1]]).is_positive_semidefinite(),
        Some(false)
    );
    assert_eq!(
        mi(&ctx, &[&[0, 0], &[0, -1]]).is_positive_definite(),
        Some(false)
    );
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![
        vec![x.clone(), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    assert_eq!(m.is_positive_definite(), None);
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let m = Matrix::new(vec![vec![p.clone(), ctx.int(0)], vec![ctx.int(0), &p + 1]]).unwrap();
    assert_eq!(m.is_positive_definite(), Some(true));
    // PD ⇒ Cholesky succeeds and ⇒ PSD
    let spd = mi(&ctx, &[&[4, 1], &[1, 3]]);
    assert_eq!(spd.is_positive_definite(), Some(true));
    assert_eq!(spd.is_positive_semidefinite(), Some(true));
    assert!(spd.cholesky().is_ok());
}

// ═══════════════════════════════════════════════════════════════════════════
// Norms
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn norms_numeric_and_symbolic() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, -2], &[-3, 4]]);
    assert_eq!(a.norm_1(), ctx.int(6));
    assert_eq!(a.norm_inf(), ctx.int(7));
    assert_eq!(a.norm_frobenius().eval(), ctx.int(30).sqrt().eval());
    assert_eq!(a.norm(), a.norm_frobenius());
    let v = matrix![ctx, [1], [2], [2]];
    assert_eq!(v.norm_p(&ctx.int(2)).unwrap().eval(), ctx.int(3));
    assert_eq!(v.norm_p(&ctx.int(1)).unwrap().eval(), ctx.int(5));
    let p3 = v.norm_p(&ctx.int(3)).unwrap().eval_f64().unwrap();
    assert!(common::approx_eq(p3, 17f64.cbrt(), 1e-12));
    assert!(a.norm_p(&ctx.int(2)).is_err());
    let x = ctx.symbol("x");
    let sv = Matrix::new(vec![vec![x.clone()], vec![ctx.int(1)]]).unwrap();
    let n1 = sv.norm_1();
    assert!(n1.contains(&x));
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix functions & least squares & subspaces
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbolic_power_matches_integer_powers() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    let a = mi(&ctx, &[&[1, 2], &[2, 1]]); // eigenvalues 3, −1
    let an = a.matrix_pow_symbolic(&n).unwrap();
    for k in 0..5u32 {
        let direct = a.powi(k).unwrap();
        let via = an.subs(&n, &ctx.int(k as i64)).eval().simplify();
        assert_eq!(via, direct, "A^{k}");
    }
    assert!(
        mi(&ctx, &[&[1, 1], &[0, 1]])
            .matrix_pow_symbolic(&n)
            .is_err()
    );
}

#[test]
fn matrix_sqrt_squares_back() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[5, 4], &[4, 5]]); // eigenvalues 9, 1 → sqrt = [[2,1],[1,2]]
    let s = a.matrix_sqrt().unwrap();
    assert_eq!(s, mi(&ctx, &[&[2, 1], &[1, 2]]));
    let b = mi(&ctx, &[&[2, 1], &[1, 2]]);
    let sb = b.matrix_sqrt().unwrap();
    assert_close_matrix(&(&sb * &sb), &b, 1e-12, "√B·√B = B");
}

#[test]
fn least_squares_normal_equations() {
    let ctx = Context::new();
    // Overdetermined: fit a line through (0,1), (1,3), (2,4), (3,6): slope 8/5, intercept 11/10
    let a = mi(&ctx, &[&[1, 0], &[1, 1], &[1, 2], &[1, 3]]);
    let b = mi(&ctx, &[&[1], &[3], &[4], &[6]]);
    let x = a.solve_least_squares(&b).unwrap();
    assert_eq!(
        x,
        Matrix::new(vec![vec![ctx.rational(11, 10)], vec![ctx.rational(8, 5)]]).unwrap()
    );
    // Residual is orthogonal to the column space
    let res = &b - &(&a * &x);
    let at_res = (&a.transpose() * &res).eval();
    assert_eq!(at_res.is_zero(), Some(true));
    // Agrees with pinv
    let via_pinv = (&a.pinv().unwrap() * &b).eval();
    assert_eq!(via_pinv, x);
    assert!(a.solve_least_squares(&matrix![ctx, [1], [2]]).is_err());
}

#[test]
fn four_fundamental_subspaces() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2, 3], &[2, 4, 6], &[1, 1, 1]]);
    assert_eq!(a.rank(), 2);
    assert_eq!(a.columnspace().len(), 2);
    assert_eq!(a.rowspace().len(), 2);
    assert_eq!(a.nullspace().len(), 1);
    assert_eq!(a.left_nullspace().len(), 1);
    for v in a.nullspace() {
        assert_eq!((&a * &v).eval().is_zero(), Some(true));
    }
    for y in a.left_nullspace() {
        assert_eq!((&y.transpose() * &a).eval().is_zero(), Some(true));
    }
    for r in a.rowspace() {
        assert_eq!(r.shape(), (1, 3));
    }
}

#[test]
fn hessian_wronskian_and_jacobian() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &(&x.powi(2) * &y.powi(2)) + &x.exp();
    let h = hessian(&f, &[&x, &y]);
    assert_eq!(h.is_symmetric(), Some(true));
    assert_eq!(h.get(0, 1), &(&(&x * &y) * 4));
    assert_eq!(h.get(1, 1), &(&x.powi(2) * 2));
    // Hessian of a quadratic form ½ xᵀ Q x is Q
    let q = mi(&ctx, &[&[2, 1], &[1, 3]]);
    let v = Matrix::new(vec![vec![x.clone()], vec![y.clone()]]).unwrap();
    let quad = (&(&v.transpose() * &q) * &v)[(0, 0)].clone() / 2;
    assert_eq!(hessian(&quad, &[&x, &y]).expand(), q);

    // Wronskian of e^x, e^{2x}, e^{3x} = 2 e^{6x}
    let w = wronskian(&[&x.exp(), &(&x * 2).exp(), &(&x * 3).exp()], &x).simplify();
    assert_eq!(w, &(&x * 6).exp() * 2);
    // sin, cos → −1
    assert_eq!(wronskian(&[&x.sin(), &x.cos()], &x).simplify(), ctx.int(-1));
    // linearly dependent → 0
    assert!(
        wronskian(&[&x, &(&x * 5)], &x)
            .simplify()
            .is_zero_structural()
    );
}
