//! Linear algebra found by the differential matrix hunter (random
//! integer, rational, ℚ(√2, √3, i), symbolic and structured matrices; every
//! result checked by its defining identity with mpmath at 50 digits).
//! Each test says what was wrong before and cites the oracle call.

use symplex::prelude::*;

/// A square matrix from `;`-separated cells in the parser's syntax.
fn sq(ctx: &Context, n: usize, cells: &str) -> Matrix {
    let c: Vec<&str> = cells.split(';').collect();
    assert_eq!(c.len(), n * n);
    Matrix::new(
        (0..n)
            .map(|i| (0..n).map(|j| ctx.parse(c[i * n + j]).unwrap()).collect())
            .collect(),
    )
    .unwrap()
}

fn c64(e: &Ex) -> Complex64 {
    e.eval_complex64()
        .unwrap_or_else(|err| panic!("{e} does not evaluate: {err}"))
}

fn close(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() <= tol * (1.0 + b.norm())
}

/// `‖A·B − C‖` entrywise at the numeric values, after substituting `subs`.
fn assert_product(a: &Matrix, b: &Matrix, c: &Matrix, subs: &[(&Ex, &Ex)], what: &str) {
    let ab = a.matmul(b).unwrap().subs_map(subs);
    let c = c.subs_map(subs);
    for i in 0..ab.nrows() {
        for j in 0..ab.ncols() {
            let (u, v) = (c64(ab.get(i, j)), c64(c.get(i, j)));
            assert!(
                close(u, v, 1e-9),
                "{what}: entry ({i}, {j}) is {u}, want {v}"
            );
        }
    }
}

#[test]
fn ldl_and_cholesky_refuse_a_matrix_of_undecided_symmetry() {
    // Before: `[[x², x + y], [2x, 0]]` was taken as symmetric (`x + y − 2x`
    // is not provably 0) and factored from its lower triangle, so
    // L·D·Lᵀ = [[x², 2x], [2x, 0]] ≠ A.  SymPy:
    // Matrix([[x**2, x + y], [2*x, 0]]).LDLdecomposition() and .cholesky()
    // raise "Matrix must be Hermitian."; so does [[x, -y], [y, x]].
    let ctx = Context::new();
    for cells in ["x^2;x+y;2*x;0", "x;-y;y;x"] {
        let a = sq(&ctx, 2, cells);
        for r in [a.ldl().map(|_| ()), a.cholesky().map(|_| ())] {
            let err = r.unwrap_err();
            assert!(
                matches!(err, SymplexError::ComputationFailed { .. }),
                "{cells}: {err}"
            );
            assert!(err.to_string().contains("cannot decide"), "{cells}: {err}");
        }
    }
    // Provably symmetric symbolic matrices still decompose.
    let s = sq(&ctx, 2, "x;y;y;x");
    let Ldl { l, d } = s.ldl().unwrap();
    let back = (&(&l * &d) * &l.transpose()).simplify();
    assert_eq!(back, s);
}

#[test]
fn symbolic_eigenvalue_with_a_square_root_has_its_eigenvector() {
    // Before: `eigenvects` of [[x², x + y], [2x, 0]] returned no eigenvector
    // for either eigenvalue x²/2 ± √(x⁴ + 8x² + 8xy)/2 (the last pivot of
    // A − λI was not recognised as 0), and `diagonalize`/`jordan_form`/
    // `matrix_exp` were refused.  SymPy: eigenvects gives [λ/(2x), 1] for
    // each.  Oracle: A·v = λ·v at two points.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let a = sq(&ctx, 2, "x^2;x+y;2*x;0");
    let ev = a.eigenvects().unwrap();
    assert_eq!(ev.len(), 2);
    for (lam, mult, vecs) in &ev {
        assert_eq!(*mult, 1);
        assert_eq!(vecs.len(), 1, "no eigenvector for {lam}");
        let v = &vecs[0];
        for (xv, yv) in [
            (ctx.rational(2, 3), ctx.rational(5, 7)),
            (ctx.rational(-3, 2), ctx.rational(1, 3)),
        ] {
            let subs = [(&x, &xv), (&y, &yv)];
            assert_product(&a, v, &v.scale(lam), &subs, "A·v = λ·v");
            // SymPy's normalisation: v = [λ/(2x), 1]
            let want = (lam / &(&x * 2)).subs_map(&subs);
            assert!(close(c64(&v.get(0, 0).subs_map(&subs)), c64(&want), 1e-12));
        }
    }
    assert!(a.diagonalize().is_ok());
    assert!(a.matrix_exp().is_ok());
}

#[test]
fn symbolic_eigenvalue_multiplicities_come_from_the_factorisation() {
    // Before: the 4×4 P·diag(x, x, x, y)·P⁻¹ below had the eigenvalues
    // x (multiplicity 2) and (x + y ± √(x² − 2xy + y²))/2 — x and y region
    // by region — so x was counted twice, plus once in disguise, and
    // `jordan_form` failed ("expected 4 basis vectors, got 3").  SymPy:
    // eigenvals() == {x: 3, y: 1}.  Its 2×2 block [[2y − x, x − y],
    // [2y − 2x, 2x − y]] had the same disguised pair; SymPy: {x: 1, y: 1}.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let a = sq(
        &ctx,
        4,
        "x;0;0;0;x-y;2*y-x;0;x-y;0;0;x;0;x-y;2*y-2*x;0;2*x-y",
    );
    let mut ev = a.eigenvals_with_multiplicity().unwrap();
    ev.sort_by_key(|(_, m)| std::cmp::Reverse(*m));
    assert_eq!(ev, vec![(x.clone(), 3), (y.clone(), 1)]);
    let b = sq(&ctx, 2, "2*y-x;x-y;2*y-2*x;2*x-y");
    let mut ev = b.eigenvals().unwrap();
    ev.sort_by_key(|e| e.to_string());
    assert_eq!(ev, vec![x.clone(), y.clone()]);
    let JordanForm { p, j } = a.jordan_form().unwrap();
    let (xv, yv) = (ctx.rational(3, 7), ctx.rational(-5, 4));
    assert_product(&a, &p, &(&p * &j), &[(&x, &xv), (&y, &yv)], "A·P = P·J");
}

#[test]
fn is_diagonalizable_decides_symbolic_rational_matrices() {
    // Before: `None` for every symbolic defective matrix, although the zero
    // tests of A − λI are exact for rational functions of the symbols.
    // SymPy: Matrix([[x, 1], [0, x]]).is_diagonalizable() is False;
    // Matrix([[x, -1, x], [0, 2*x, x], [0, 0, x]]) is False (λ = x twice,
    // one eigenvector); Matrix([[x, y], [y, x]]) is True.
    let ctx = Context::new();
    assert_eq!(sq(&ctx, 2, "x;1;0;x").is_diagonalizable(), Some(false));
    assert_eq!(
        sq(&ctx, 3, "x;-1;x;0;2*x;x;0;0;x").is_diagonalizable(),
        Some(false)
    );
    assert_eq!(sq(&ctx, 2, "x;y;y;x").is_diagonalizable(), Some(true));
}

#[test]
fn pinv_of_a_complex_matrix_uses_the_conjugate_transpose() {
    // Before: the symbolic tier used Aᵀ, so [[1, i]] (A·Aᵀ = 0) was refused
    // and [[0, 1, √2·i], [0, −√2, √3], [0, 0, 2]] got a "pseudo-inverse"
    // with (A·A⁺)ᴴ ≠ A·A⁺.  SymPy: Matrix([[1, I]]).pinv() == [[1/2], [-I/2]];
    // the 3×3 one is [[0, 0, 0], [7/19 − 2√3i/19, −6√2/19 − √6i/19,
    // 2√6/19 − 2√2i/19], [√6/19 − 2√2i/19, √3/19 − 2i/19, 6/19]].
    let ctx = Context::new();
    let i = ctx.i_unit();
    let a = Matrix::new(vec![vec![ctx.one(), i.clone()]]).unwrap();
    let want =
        Matrix::col_vector(vec![ctx.rational(1, 2), (&i * &ctx.rational(-1, 2)).eval()]).unwrap();
    assert_eq!(a.pinv().unwrap(), want);
    let b = sq(&ctx, 3, "0;1;sqrt(2)*I;0;-sqrt(2);sqrt(3);0;0;2");
    let p = b.pinv().unwrap();
    let sympy = [
        ["0", "0", "0"],
        [
            "7/19 - 2*sqrt(3)*I/19",
            "-6*sqrt(2)/19 - sqrt(6)*I/19",
            "2*sqrt(6)/19 - 2*sqrt(2)*I/19",
        ],
        ["sqrt(6)/19 - 2*sqrt(2)*I/19", "sqrt(3)/19 - 2*I/19", "6/19"],
    ];
    for (r, row) in sympy.iter().enumerate() {
        for (c, s) in row.iter().enumerate() {
            let w = c64(&ctx.parse(s).unwrap());
            assert!(close(c64(p.get(r, c)), w, 1e-12), "pinv ({r}, {c})");
        }
    }
}

#[test]
fn least_squares_and_condition_number_refuse_symbolic_rank_deficiency() {
    // Before: [[x − 2, 2], [0, 0]] (rank 1) had a least-squares "solution"
    // dividing by an unrecognised zero (no value anywhere) and a condition
    // number Max(σ)/Min(σ) whose Min is identically 0 (SymPy:
    // condition_number() is zoo; solve_least_squares raises).
    let ctx = Context::new();
    let a = sq(&ctx, 2, "x-2;2;0;0");
    let b = Matrix::col_vector(vec![ctx.int(1), ctx.int(2)]).unwrap();
    assert!(matches!(
        a.solve_least_squares(&b),
        Err(SymplexError::ComputationFailed { .. })
    ));
    let err = a.condition_number().unwrap_err();
    assert!(err.to_string().contains("singular"), "{err}");
    // Full column rank keeps working.
    let c = sq(&ctx, 2, "x;1;0;1");
    assert!(c.solve_least_squares(&b).is_ok());
}

#[test]
fn qr_of_a_rank_deficient_matrix_is_the_reduced_factorisation() {
    // Before: refused ("columns are linearly dependent").  SymPy:
    // Matrix([[1, 2], [2, 4]]).QRdecomposition() ==
    //   ([[sqrt(5)/5], [2*sqrt(5)/5]], [[sqrt(5), 2*sqrt(5)]]);
    // Matrix([[1, 2, 0], [2, 4, 1], [0, 0, 1]]).QRdecomposition() has
    // R = [[sqrt(5), 2*sqrt(5), 2*sqrt(5)/5], [0, 0, sqrt(30)/5]].
    let ctx = Context::new();
    let s5 = ctx.int(5).sqrt();
    let Qr { q, r } = matrix![ctx, [1, 2], [2, 4]].qr().unwrap();
    assert_eq!(
        q,
        Matrix::col_vector(vec![(&s5 / 5).eval(), (&s5 * 2 / 5).eval()]).unwrap()
    );
    assert_eq!(
        r,
        Matrix::row_vector(vec![s5.clone(), (&s5 * 2).eval()]).unwrap()
    );
    let a = matrix![ctx, [1, 2, 0], [2, 4, 1], [0, 0, 1]];
    let Qr { q, r } = a.qr().unwrap();
    assert_eq!(q.shape(), (3, 2));
    assert_eq!(r[(0, 2)], (&s5 * 2 / 5).eval());
    assert!(r[(1, 0)].is_zero_structural() && r[(1, 1)].is_zero_structural());
    assert_eq!(r[(1, 2)], (ctx.int(30).sqrt() / 5).eval());
    assert_eq!((&q * &r).simplify(), a);
    assert_eq!(
        (&q.adjoint() * &q).simplify(),
        Matrix::identity(&ctx, 2).unwrap()
    );
    // The zero matrix has no Q.
    assert!(matrix![ctx, [0, 0], [0, 0]].qr().is_err());
}

#[test]
fn sqrt_and_symbolic_power_of_defective_matrices_use_jordan_blocks() {
    // Before: refused ("requires a diagonalizable matrix").  SymPy:
    // Matrix([[4, 1], [0, 4]])**(1/2) == [[2, 1/4], [0, 2]];
    // Matrix([[2, 1], [0, 2]])**n == [[2**n, 2**(n - 1)*n], [0, 2**n]];
    // Matrix([[0, 1], [0, 0]])**n stays unevaluated (refused here too).
    let ctx = Context::new();
    let n = ctx.symbol("n");
    assert_eq!(
        matrix![ctx, [4, 1], [0, 4]].matrix_sqrt().unwrap(),
        matrix![ctx, [2, 1 / 4], [0, 2]]
    );
    let p = matrix![ctx, [2, 1], [0, 2]]
        .matrix_pow_symbolic(&n)
        .unwrap();
    let two_n = ctx.int(2).pow(&n);
    assert_eq!(p[(0, 0)], two_n);
    let want = &(&n * &ctx.int(2).pow(&(&n - 1))) - &p[(0, 1)];
    assert!(want.subs(&n, &ctx.int(7)).eval().is_zero_structural());
    assert!(
        matrix![ctx, [0, 1], [0, 0]]
            .matrix_pow_symbolic(&n)
            .is_err()
    );
    assert!(matrix![ctx, [0, 1], [0, 0]].matrix_sqrt().is_err());
    // A 3×3 defective one: S² = A.
    let a = matrix![ctx, [5, 4, 2], [-1, 1, -1], [0, 0, 3]];
    let s = a.matrix_sqrt().unwrap();
    assert_product(&s, &s, &a, &[], "S·S = A");
}

#[test]
fn matrix_exp_over_q_sqrt2_sqrt3_i_is_not_expanded_into_megabytes() {
    // Before: `matrix_exp` of this Hermitian 3×3 matrix (Cardano
    // eigenvalues) expanded every entry after a trig rewrite that changed
    // nothing: 15.6 s (release) for a 179 MB result.  Now ~0.3 s, ~0.4 MB.
    // Oracle: mpmath expm at 30 digits, e^A[0,1] =
    // -2.06998249220637416118 + 7.95552583177113680053i,
    // e^A[2,2] = 18.1192883594330190970.
    let ctx = Context::new();
    let a = sq(
        &ctx,
        3,
        "0;-sqrt(3)+2*I;1/2+2*I;-sqrt(3)-2*I;-2;1+I;1/2-2*I;1-I;sqrt(2)",
    );
    let e = a.matrix_exp().unwrap();
    assert!(
        e.to_string().len() < 2_000_000,
        "{} chars",
        e.to_string().len()
    );
    assert!(close(
        c64(&e[(0, 1)]),
        Complex64::new(-2.069_982_492_206_374, 7.955_525_831_771_137),
        1e-10
    ));
    assert!(close(
        c64(&e[(2, 2)]),
        Complex64::new(18.119_288_359_433_02, 0.0),
        1e-10
    ));
}

#[test]
fn matrix_sqrt_with_cardano_eigenvalues_skips_a_futile_simplify() {
    // Before: 10 s (release; `simplify` spent ~1.1 s on each 69 KB entry
    // and returned it unchanged).  Oracle: scipy.linalg.sqrtm,
    // √A[0,0] = 0.9218077343167823 + 0.4028115112781792i,
    // √A[1,2] = −0.05748280163944108 − 0.2532714232002104i.
    let ctx = Context::new();
    let a = sq(
        &ctx,
        3,
        "1/2;-sqrt(2);sqrt(3)-I;-3;2*I;0;1-sqrt(3);1+sqrt(2);1+sqrt(2)",
    );
    let s = a.matrix_sqrt().unwrap();
    assert!(close(
        c64(&s[(0, 0)]),
        Complex64::new(0.921_807_734_316_782_3, 0.402_811_511_278_179_2),
        1e-10
    ));
    assert!(close(
        c64(&s[(1, 2)]),
        Complex64::new(-0.057_482_801_639_441_08, -0.253_271_423_200_210_4),
        1e-10
    ));
}

#[test]
fn matrix_exp_with_rootof_eigenvalues_falls_back_to_sylvester() {
    // Before: refused ("expression swell": P·e^J·P⁻¹ over four `RootOf`
    // eigenvalues of λ⁴ + 5λ³ − 59λ² − 283λ + 116 exceeded the budget;
    // SymPy's Matrix.exp() does not finish in 60 s).  Now Sylvester's
    // formula Σ e^λᵢ·adj(λᵢI − A)/p′(λᵢ).  Oracle: mpmath expm at 30
    // digits, e^A[0,0] = 224.22756080181951752, e^A[1,3] =
    // 439.94958491025360916, e^A[3,2] = 713.47808800603042749.
    let ctx = Context::new();
    let a = matrix![
        ctx,
        [-4, -5, -5, 2],
        [-1, -4, 3, 4],
        [-5, 5, -1, 2],
        [-2, 0, 3, 4]
    ];
    let e = a.matrix_exp().unwrap();
    for (r, c, want) in [
        (0, 0, 224.227_560_801_819_5),
        (1, 3, 439.949_584_910_253_6),
        (3, 2, 713.478_088_006_030_4),
    ] {
        assert!(
            close(c64(&e[(r, c)]), Complex64::new(want, 0.0), 1e-10),
            "e^A[{r},{c}]"
        );
    }
    // A simple eigenvalue structure throughout: sqrt and log work too.
    let s = a.matrix_sqrt().unwrap();
    assert_product(&s, &s, &a, &[], "S·S = A");
    assert!(a.matrix_log().is_ok());
}

#[test]
fn eigenvectors_of_a_symbolic_rank_one_matrix_are_in_normal_form() {
    // Before: eigenvals was refused (λ³·(λ − y² − 6x − 6) not factored);
    // with the factorisation alone the eigenvector of y² + 6x + 6 was a 2 KB
    // nest of quotients and inverting the eigenvector matrix for
    // `matrix_exp` took 14 s.  SymPy: eigenvects gives (x/2, y/2, x/2, 1);
    // exp(A) at x = 1/3, y = −1/2 has [0,0] = 464.83343290180681543 and
    // [3,1] = −463.83343290180681543.
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let a = sq(
        &ctx,
        4,
        "3*x;x*y;3*x;3*x;3*y;y*y;3*y;3*y;3*x;x*y;3*x;3*x;6;2*y;6;6",
    );
    let ev = a.eigenvects().unwrap();
    let half = ctx.rational(1, 2);
    let want = Matrix::col_vector(vec![
        (&x * &half).eval(),
        (&y * &half).eval(),
        (&x * &half).eval(),
        ctx.one(),
    ])
    .unwrap();
    let (_, mult, vecs) = ev.iter().find(|(_, m, _)| *m == 1).unwrap();
    assert_eq!((*mult, vecs.as_slice()), (1, std::slice::from_ref(&want)));
    let e = a.matrix_exp().unwrap();
    let (xv, yv) = (ctx.rational(1, 3), ctx.rational(-1, 2));
    let at = |r, c| c64(&e[(r, c)].subs_map(&[(&x, &xv), (&y, &yv)]));
    assert!(close(
        at(0, 0),
        Complex64::new(464.833_432_901_806_8, 0.0),
        1e-10
    ));
    assert!(close(
        at(3, 1),
        Complex64::new(-463.833_432_901_806_8, 0.0),
        1e-10
    ));
}

#[test]
fn sqrt_and_log_of_a_hermitian_matrix_with_negative_eigenvalues_evaluate() {
    // Before: every entry of √A and log A was unevaluable: √λ of a negative
    // eigenvalue in Cardano form sits on the branch cut with an imaginary
    // part that cancels only numerically.  The principal values are
    // i·√(−λ) and ln(−λ) + iπ.  Oracle: mpmath at 40 digits,
    // V·diag(f(λₖ))·V⁻¹ from mp.eig with those principal values (scipy's
    // sqrtm/logm put λ on the other side of the cut: rounding gives it an
    // imaginary part of −10⁻¹⁶).  First matrix (eigenvalues −4.91, −0.61,
    // 2.29): √A[0,0] = 0.46575172497053996 + 1.2123689985687554i,
    // √A[0,2] = 0.1837343399412136 + 1.0826046265140811i; second
    // (eigenvalues −4.43, 0.26, 3.58): log A[0,0] = 0.56358335181982777 +
    // 1.0192880951443806i, log A[1,2] = 1.3116740599708895 − 0.87103843346672452i.
    let ctx = Context::new();
    let a = sq(
        &ctx,
        3,
        "-sqrt(3);sqrt(2)+I;-sqrt(3)+2*I;sqrt(2)-I;-2;1/2-I;-sqrt(3)-2*I;1/2+I;1/2",
    );
    let s = a.matrix_sqrt().unwrap();
    assert!(close(
        c64(&s[(0, 0)]),
        Complex64::new(0.465_751_724_970_54, 1.212_368_998_568_755_4),
        1e-9
    ));
    assert!(close(
        c64(&s[(0, 2)]),
        Complex64::new(0.183_734_339_941_213_6, 1.082_604_626_514_081_1),
        1e-9
    ));
    let b = sq(
        &ctx,
        3,
        "0;-sqrt(3)+2*I;1/2+2*I;-sqrt(3)-2*I;-2;1+I;1/2-2*I;1-I;sqrt(2)",
    );
    let l = b.matrix_log().unwrap();
    assert!(close(
        c64(&l[(0, 0)]),
        Complex64::new(0.563_583_351_819_827_8, 1.019_288_095_144_380_6),
        1e-9
    ));
    assert!(close(
        c64(&l[(1, 2)]),
        Complex64::new(1.311_674_059_970_889_5, -0.871_038_433_466_724_5),
        1e-9
    ));
}
