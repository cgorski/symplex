//! symplex 0.9 — more matrix decompositions and utilities:
//! `singular_values` / `condition_number`, rank-deficient `pinv`,
//! `rank_decomposition`, `hessenberg`, `companion`, `jordan_block`,
//! `permanent`, `row_insert` / `col_insert` / `permute_rows` /
//! `permute_cols` / `row_del` / `col_del`, `casoratian`, `inv_mod`,
//! `ZMatrix::lll`, `matrix_log`.
//!
//! Reference values are from SymPy 1.14 (`symplex/.venv`), quoted in the
//! comments of each test.  Exact answers are asserted structurally (after
//! `simplify` / `eval` where the construction leaves unfolded constants);
//! radicals are additionally checked numerically.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Signed;
use symplex::matrix::{QMatrix, ZMatrix};
use symplex::prelude::*;

fn q(n: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

fn f64_of(e: &Ex) -> f64 {
    e.eval_f64()
        .unwrap_or_else(|err| panic!("{e} should evaluate numerically: {err}"))
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. singular_values / condition_number
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn singular_values_2x2_are_sqrt_of_15_plus_minus_sqrt_221() {
    // SymPy: Matrix([[1, 2], [3, 4]]).singular_values()
    //   == [sqrt(sqrt(221) + 15), sqrt(15 - sqrt(221))]
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let sv = m.singular_values().unwrap();
    assert_eq!(sv.len(), 2);
    let s221 = ctx.int(221).sqrt();
    assert_eq!(sv[0], (&s221 + 15).sqrt(), "got {}", sv[0]);
    assert_eq!(sv[1], (-&s221 + 15).sqrt(), "got {}", sv[1]);
    // Descending, and consistent with the Frobenius norm: σ₁² + σ₂² = 30.
    let (a, b) = (f64_of(&sv[0]), f64_of(&sv[1]));
    assert!(a > b);
    assert!((a * a + b * b - 30.0).abs() < 1e-12);
    assert!((a - 5.464985704219043).abs() < 1e-12);
    assert!((b - 0.3659661906262578).abs() < 1e-12);
}

#[test]
fn singular_values_diagonal_and_rectangular_match_sympy() {
    let ctx = Context::new();
    // SymPy: Matrix([[2, 0], [0, 3]]).singular_values() == [3, 2]
    assert_eq!(
        matrix![ctx, [2, 0], [0, 3]].singular_values().unwrap(),
        vec![ctx.int(3), ctx.int(2)]
    );
    // SymPy: Matrix([[1, 0], [0, 1], [1, 1]]).singular_values() == [sqrt(3), 1]
    let sv = matrix![ctx, [1, 0], [0, 1], [1, 1]]
        .singular_values()
        .unwrap();
    assert_eq!(sv, vec![ctx.int(3).sqrt(), ctx.int(1)], "got {sv:?}");
    // SymPy: Matrix([[3, 0, 0], [0, 4, 0]]).singular_values() == [4, 3, 0]
    assert_eq!(
        matrix![ctx, [3, 0, 0], [0, 4, 0]]
            .singular_values()
            .unwrap(),
        vec![ctx.int(4), ctx.int(3), ctx.int(0)]
    );
    // SymPy: Matrix([[3, 4]]).singular_values() == [5, 0]
    assert_eq!(
        matrix![ctx, [3, 4]].singular_values().unwrap(),
        vec![ctx.int(5), ctx.int(0)]
    );
}

#[test]
fn singular_values_symbolic_2x2_via_quadratic_formula() {
    let ctx = Context::new();
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let b = ctx.symbol_with("b", &[Assumption::Positive]).unwrap();
    let m = Matrix::diag(&[a.clone(), b.clone()]).unwrap();
    let sv = m.singular_values().unwrap();
    assert_eq!(sv.len(), 2);
    // {√(a²), √(b²)} = {a, b} as a set (order is not decidable symbolically).
    let vals: Vec<String> = sv.iter().map(|v| v.simplify().to_string()).collect();
    assert!(
        vals.contains(&"a".to_string()) && vals.contains(&"b".to_string()),
        "{vals:?}"
    );
}

#[test]
fn condition_number_matches_sympy_and_rejects_singular() {
    let ctx = Context::new();
    // SymPy: Matrix([[2, 0], [0, 3]]).condition_number() == 3/2
    assert_eq!(
        matrix![ctx, [2, 0], [0, 3]].condition_number().unwrap(),
        ctx.rational(3, 2)
    );
    // SymPy: Matrix([[1, 2], [3, 4]]).condition_number()
    //   == sqrt(sqrt(221) + 15)/sqrt(15 - sqrt(221)) ≈ 14.933034373659268
    let k = matrix![ctx, [1, 2], [3, 4]].condition_number().unwrap();
    assert!(
        (f64_of(&k) - 14.933_034_373_659_268).abs() < 1e-9,
        "got {k}"
    );
    // Singular: SymPy returns zoo; we return ComputationFailed "singular".
    let err = matrix![ctx, [1, 2], [2, 4]].condition_number().unwrap_err();
    assert!(
        matches!(&err, SymplexError::ComputationFailed { reason, .. } if reason.contains("singular")),
        "{err}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. pinv for any rank
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn pinv_rank_deficient_square_matches_sympy() {
    // SymPy: Matrix([[1, 2], [2, 4]]).pinv() == Matrix([[1/25, 2/25], [2/25, 4/25]])
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [2, 4]];
    let p = a.pinv().unwrap();
    let expected = Matrix::new(vec![
        vec![ctx.rational(1, 25), ctx.rational(2, 25)],
        vec![ctx.rational(2, 25), ctx.rational(4, 25)],
    ])
    .unwrap();
    assert_eq!(p, expected, "got {p}");
    // Moore–Penrose conditions.
    assert_eq!(&(&a * &p) * &a, a);
    assert_eq!(&(&p * &a) * &p, p);
    assert_eq!((&a * &p).transpose(), &a * &p);
    assert_eq!((&p * &a).transpose(), &p * &a);
}

#[test]
fn pinv_rank_deficient_rectangular_and_zero_matrix() {
    let ctx = Context::new();
    // SymPy: Matrix([[1, 2], [2, 4], [3, 6]]).pinv()
    //   == Matrix([[1/70, 1/35, 3/70], [1/35, 2/35, 3/35]])
    let a = matrix![ctx, [1, 2], [2, 4], [3, 6]];
    let p = a.pinv().unwrap();
    let expected = Matrix::new(vec![
        vec![
            ctx.rational(1, 70),
            ctx.rational(1, 35),
            ctx.rational(3, 70),
        ],
        vec![
            ctx.rational(1, 35),
            ctx.rational(2, 35),
            ctx.rational(3, 35),
        ],
    ])
    .unwrap();
    assert_eq!(p, expected, "got {p}");
    // SymPy: Matrix([[1, 2, 3], [4, 5, 6]]).pinv()  (full row rank, rank-deficient columns)
    //   == Matrix([[-17/18, 4/9], [-1/9, 1/9], [13/18, -2/9]])
    let b = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    let pb = b.pinv().unwrap();
    let expected_b = Matrix::new(vec![
        vec![ctx.rational(-17, 18), ctx.rational(4, 9)],
        vec![ctx.rational(-1, 9), ctx.rational(1, 9)],
        vec![ctx.rational(13, 18), ctx.rational(-2, 9)],
    ])
    .unwrap();
    assert_eq!(pb, expected_b, "got {pb}");
    // Zero matrix: A⁺ = 0 of the transposed shape.
    assert_eq!(
        Matrix::zeros(&ctx, 2, 3).unwrap().pinv().unwrap(),
        Matrix::zeros(&ctx, 3, 2).unwrap()
    );
    // Full column rank keeps the classical formula.
    let c = matrix![ctx, [1, 0], [0, 1], [1, 1]];
    assert_eq!(&c.pinv().unwrap() * &c, Matrix::identity(&ctx, 2).unwrap());
}

#[test]
fn pinv_symbolic_rank_one() {
    // A = x·[[1, 1], [1, 1]]  →  A⁺ = (1/(4x))·[[1, 1], [1, 1]]
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Positive]).unwrap();
    let a = Matrix::new(vec![vec![x.clone(), x.clone()], vec![x.clone(), x.clone()]]).unwrap();
    let p = a.pinv().unwrap().simplify();
    let quarter_over_x = &ctx.rational(1, 4) / &x;
    for e in p.iter() {
        assert_eq!(e, &quarter_over_x, "got {p}");
    }
    assert_eq!((&(&a * &p) * &a).simplify(), a);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. hessenberg
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hessenberg_is_a_similarity_transform_over_q() {
    let ctx = Context::new();
    let a = matrix![
        ctx,
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 12, 11],
        [13, 15, 14, 16]
    ];
    let Hessenberg { h, p } = a.hessenberg().unwrap();
    // Upper Hessenberg: zeros below the sub-diagonal.
    for i in 0..4 {
        for j in 0..4 {
            if i > j + 1 {
                assert!(
                    h[(i, j)].is_zero_structural(),
                    "h[{i}][{j}] = {}",
                    h[(i, j)]
                );
            }
        }
    }
    // A·P = P·H  and  H = P⁻¹·A·P  (exact rationals, structural equality).
    assert_eq!(&a * &p, &p * &h);
    assert_eq!(&(&p.inv().unwrap() * &a) * &p, h);
    // Similarity invariants.
    assert_eq!(h.trace().unwrap(), a.trace().unwrap());
    assert_eq!(h.det().unwrap(), a.det().unwrap());
    let lam = ctx.symbol("lambda");
    assert_eq!(h.char_poly(&lam).unwrap(), a.char_poly(&lam).unwrap());
}

#[test]
fn hessenberg_needs_a_row_swap_when_the_subdiagonal_entry_is_zero() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [0, 4, 5], [6, 7, 8]];
    let Hessenberg { h, p } = a.hessenberg().unwrap();
    assert!(h[(2, 0)].is_zero_structural());
    assert_eq!(&a * &p, &p * &h);
    // Already Hessenberg → unchanged with P = I.
    let t = matrix![ctx, [1, 2, 3], [4, 5, 6], [0, 7, 8]];
    let Hessenberg { h: h2, p: p2 } = t.hessenberg().unwrap();
    assert_eq!(h2, t);
    assert_eq!(p2, Matrix::identity(&ctx, 3).unwrap());
    assert!(matrix![ctx, [1, 2, 3]].hessenberg().is_err());
}

#[test]
fn hessenberg_symbolic_holds_generically() {
    let ctx = Context::new();
    let (a, b, c) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
    let m = Matrix::new(vec![
        vec![a.clone(), b.clone(), c.clone()],
        vec![ctx.int(1), a.clone(), ctx.int(0)],
        vec![b.clone(), ctx.int(1), c.clone()],
    ])
    .unwrap();
    let Hessenberg { h, p } = m.hessenberg().unwrap();
    assert!(h[(2, 0)].is_zero_structural());
    let lhs = (&m * &p).simplify();
    let rhs = (&p * &h).simplify();
    assert_eq!(lhs.equals(&rhs), Some(true), "{lhs} vs {rhs}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. rank_decomposition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rank_decomposition_matches_sympy() {
    // SymPy: Matrix([[1,2,3],[4,5,6],[7,8,9]]).rank_decomposition()
    //   == (Matrix([[1, 2], [4, 5], [7, 8]]), Matrix([[1, 0, -1], [0, 1, 2]]))
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let RankDecomposition { c, f } = a.rank_decomposition().unwrap();
    assert_eq!(c, matrix![ctx, [1, 2], [4, 5], [7, 8]]);
    assert_eq!(f, matrix![ctx, [1, 0, -1], [0, 1, 2]]);
    assert_eq!(&c * &f, a);
    // SymPy: Matrix([[1, 2], [2, 4]]).rank_decomposition() == (Matrix([[1], [2]]), Matrix([[1, 2]]))
    let RankDecomposition { c: c2, f: f2 } =
        matrix![ctx, [1, 2], [2, 4]].rank_decomposition().unwrap();
    assert_eq!(c2, matrix![ctx, [1], [2]]);
    assert_eq!(f2, matrix![ctx, [1, 2]]);
    // Rank 0 has no representable factors.
    assert!(matches!(
        Matrix::zeros(&ctx, 2, 2).unwrap().rank_decomposition(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // Symbolic path: rank 1 structurally.
    let x = ctx.symbol("x");
    let s = Matrix::new(vec![vec![x.clone(), &x * 2], vec![&x * 3, &x * 6]]).unwrap();
    let RankDecomposition { c: cs, f: fs } = s.rank_decomposition().unwrap();
    assert_eq!(cs.shape(), (2, 1));
    assert_eq!(fs, matrix![ctx, [1, 2]]);
    assert_eq!((&cs * &fs).simplify(), s);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Utilities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn companion_matrix_matches_sympy_and_has_the_right_char_poly() {
    // SymPy: Matrix.companion(Poly(x**3 + 2*x**2 + 3*x + 4, x))
    //   == Matrix([[0, 0, -4], [1, 0, -3], [0, 1, -2]])
    let ctx = Context::new();
    let c = Matrix::companion(&[ctx.int(4), ctx.int(3), ctx.int(2)]).unwrap();
    assert_eq!(c, matrix![ctx, [0, 0, -4], [1, 0, -3], [0, 1, -2]]);
    // char_poly is det(C − xI) = (−1)³·p(x); det(xI − C) = p(x).
    let x = ctx.symbol("x");
    let expected = &x.powi(3) + &x.powi(2) * 2 + &x * 3 + 4;
    assert_eq!(-c.char_poly(&x).unwrap(), expected);
    let xi_minus_c = Matrix::identity(&ctx, 3)
        .unwrap()
        .scale(&x)
        .sub(&c)
        .unwrap();
    assert_eq!(xi_minus_c.det().unwrap().expand(), expected);
    // Degree 1 and symbolic coefficients (even degree: char_poly is +p).
    assert_eq!(
        Matrix::companion(&[ctx.int(7)]).unwrap(),
        matrix![ctx, [-7]]
    );
    let (p, r) = (ctx.symbol("p"), ctx.symbol("r"));
    let cs = Matrix::companion(&[r.clone(), p.clone()]).unwrap();
    assert_eq!(
        cs.char_poly(&x).unwrap().expand(),
        (&x.powi(2) + &(&p * &x) + &r).expand()
    );
    assert!(Matrix::companion(&[]).is_err());
}

#[test]
fn jordan_block_matches_sympy() {
    // SymPy: Matrix.jordan_block(size=3, eigenvalue=2) == Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    let ctx = Context::new();
    let j = Matrix::jordan_block(&ctx.int(2), 3).unwrap();
    assert_eq!(j, matrix![ctx, [2, 1, 0], [0, 2, 1], [0, 0, 2]]);
    assert_eq!(
        Matrix::jordan_block(&ctx.int(5), 1).unwrap(),
        matrix![ctx, [5]]
    );
    // A Jordan block is its own Jordan form.
    let jf = j.jordan_form().unwrap().j;
    assert_eq!(jf, j);
    assert!(Matrix::jordan_block(&ctx.int(2), 0).is_err());
}

#[test]
fn permanent_matches_sympy_and_symbolic_is_expanded() {
    let ctx = Context::new();
    // SymPy: Matrix([[1, 2], [3, 4]]).per() == 10
    assert_eq!(
        matrix![ctx, [1, 2], [3, 4]].permanent().unwrap(),
        ctx.int(10)
    );
    // SymPy: Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).per() == 450
    assert_eq!(
        matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]]
            .permanent()
            .unwrap(),
        ctx.int(450)
    );
    // Permanent of the all-ones n×n matrix is n!.
    let ones = Matrix::from_fn(5, 5, |_, _| ctx.one()).unwrap();
    assert_eq!(ones.permanent().unwrap(), ctx.int(120));
    // Rational entries and the exact-matrix entry point agree.
    let half = Matrix::new(vec![
        vec![ctx.rational(1, 2), ctx.int(1)],
        vec![ctx.int(1), ctx.rational(1, 3)],
    ])
    .unwrap();
    assert_eq!(half.permanent().unwrap(), ctx.rational(7, 6));
    assert_eq!(
        QMatrix::new(vec![vec![q(1, 2), q(1, 1)], vec![q(1, 1), q(1, 3)]])
            .unwrap()
            .permanent()
            .unwrap(),
        q(7, 6)
    );
    // Symbolic 3×3: six monomials.
    let (a, b, c, d, e, f, g, h, i) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
        ctx.symbol("e"),
        ctx.symbol("f"),
        ctx.symbol("g"),
        ctx.symbol("h"),
        ctx.symbol("i"),
    );
    let m = Matrix::new(vec![
        vec![a.clone(), b.clone(), c.clone()],
        vec![d.clone(), e.clone(), f.clone()],
        vec![g.clone(), h.clone(), i.clone()],
    ])
    .unwrap();
    let per = m.permanent().unwrap();
    let expected = (&(&a * &e) * &i
        + &(&a * &f) * &h
        + &(&b * &d) * &i
        + &(&b * &f) * &g
        + &(&c * &d) * &h
        + &(&c * &e) * &g)
        .expand();
    assert_eq!(per, expected, "got {per}");
    // Shape and size guards.
    assert!(matches!(
        matrix![ctx, [1, 2, 3]].permanent(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        Matrix::identity(&ctx, 21).unwrap().permanent(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(ZMatrix::identity(21).unwrap().permanent().is_err());
}

#[test]
fn row_and_col_insert_match_sympy() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    // SymPy: Matrix([[1, 2], [3, 4]]).row_insert(1, Matrix([[5, 6]])) == [[1, 2], [5, 6], [3, 4]]
    assert_eq!(
        m.row_insert(1, &matrix![ctx, [5, 6]]).unwrap(),
        matrix![ctx, [1, 2], [5, 6], [3, 4]]
    );
    assert_eq!(
        m.row_insert(0, &matrix![ctx, [5, 6], [7, 8]]).unwrap(),
        matrix![ctx, [5, 6], [7, 8], [1, 2], [3, 4]]
    );
    assert_eq!(
        m.row_insert(2, &matrix![ctx, [5, 6]]).unwrap(),
        matrix![ctx, [1, 2], [3, 4], [5, 6]]
    );
    // SymPy: Matrix([[1, 2], [3, 4]]).col_insert(1, Matrix([[5], [6]])) == [[1, 5, 2], [3, 6, 4]]
    assert_eq!(
        m.col_insert(1, &matrix![ctx, [5], [6]]).unwrap(),
        matrix![ctx, [1, 5, 2], [3, 6, 4]]
    );
    assert_eq!(
        m.col_insert(2, &matrix![ctx, [5], [6]]).unwrap(),
        matrix![ctx, [1, 2, 5], [3, 4, 6]]
    );
    // Bad position / shape.
    assert!(matches!(
        m.row_insert(3, &matrix![ctx, [5, 6]]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        m.row_insert(0, &matrix![ctx, [5, 6, 7]]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        m.col_insert(3, &matrix![ctx, [5], [6]]),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        m.col_insert(0, &matrix![ctx, [5]]),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

#[test]
fn permute_rows_and_cols_match_sympy() {
    let ctx = Context::new();
    // SymPy: Matrix([[1], [2], [3]]).permute_rows([2, 0, 1]) == Matrix([[3], [1], [2]])
    let v = matrix![ctx, [1], [2], [3]];
    assert_eq!(
        v.permute_rows(&[2, 0, 1]).unwrap(),
        matrix![ctx, [3], [1], [2]]
    );
    // SymPy: Matrix([[1, 2, 3]]).permute_cols([2, 0, 1]) == Matrix([[3, 1, 2]])
    let r = matrix![ctx, [1, 2, 3]];
    assert_eq!(r.permute_cols(&[2, 0, 1]).unwrap(), matrix![ctx, [3, 1, 2]]);
    // Identity permutation, and inverse round trip.
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    assert_eq!(m.permute_rows(&[0, 1, 2]).unwrap(), m);
    let once = m.permute_rows(&[1, 2, 0]).unwrap();
    assert_eq!(once.permute_rows(&[2, 0, 1]).unwrap(), m);
    // Not a permutation: wrong length, repeated, out of range.
    for bad in [&[0usize, 1][..], &[0, 0, 1], &[0, 1, 3]] {
        assert!(
            matches!(
                m.permute_rows(bad),
                Err(SymplexError::InvalidArgument { .. })
            ),
            "{bad:?}"
        );
        assert!(
            matches!(
                m.permute_cols(bad),
                Err(SymplexError::InvalidArgument { .. })
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn row_del_and_col_del_are_the_sympy_names_for_delete_row_col() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    assert_eq!(m.row_del(0).unwrap(), matrix![ctx, [4, 5, 6]]);
    assert_eq!(m.col_del(1).unwrap(), matrix![ctx, [1, 3], [4, 6]]);
    assert_eq!(m.row_del(1).unwrap(), m.delete_row(1).unwrap());
    let err = m.row_del(2).unwrap_err();
    assert!(
        matches!(
            &err,
            SymplexError::InvalidArgument {
                operation: "row_del",
                ..
            }
        ),
        "{err}"
    );
    assert!(matrix![ctx, [1], [2]].col_del(0).is_err());
}

#[test]
fn casoratian_of_geometric_and_polynomial_sequences() {
    let ctx = Context::new();
    let n = ctx.symbol("n");
    // SymPy: casoratian([2**n, 3**n], n, zero=False) == 6**n; with zero=True (default) == 1
    let w = Matrix::casoratian(&[ctx.int(2).pow(&n), ctx.int(3).pow(&n)], &n).unwrap();
    assert_eq!(w.simplify(), ctx.int(6).pow(&n), "got {w}");
    assert_eq!(w.subs(&n, &ctx.int(0)).eval(), ctx.int(1));
    // SymPy: casoratian([1, n, n**2], n) == 2   (constant, so zero=True/False agree)
    let w2 = Matrix::casoratian(&[ctx.one(), n.clone(), n.powi(2)], &n).unwrap();
    assert_eq!(w2.expand(), ctx.int(2), "got {w2}");
    // Dependent sequences: zero.
    let w3 = Matrix::casoratian(&[n.clone(), &n * 3], &n).unwrap();
    assert!(w3.expand().is_zero_structural(), "got {w3}");
    assert!(Matrix::casoratian(&[], &n).is_err());
}

#[test]
fn inv_mod_matches_sympy() {
    let ctx = Context::new();
    // SymPy: Matrix([[1, 2], [3, 4]]).inv_mod(5) == Matrix([[3, 1], [4, 2]])
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = a.inv_mod(5).unwrap();
    assert_eq!(b, matrix![ctx, [3, 1], [4, 2]]);
    // A·B ≡ I (mod 5)
    let prod = &a * &b;
    for i in 0..2 {
        for j in 0..2 {
            let v = prod[(i, j)].as_i64().unwrap().rem_euclid(5);
            assert_eq!(v, if i == j { 1 } else { 0 });
        }
    }
    // SymPy: Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]]).inv_mod(7) == [[4, 1, 1], [4, 6, 5], [1, 5, 1]]
    assert_eq!(
        matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]]
            .inv_mod(7)
            .unwrap(),
        matrix![ctx, [4, 1, 1], [4, 6, 5], [1, 5, 1]]
    );
    // det = −2 is not coprime to 4; singular; non-integer entries; non-square.
    assert!(matches!(
        a.inv_mod(4),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        matrix![ctx, [1, 2], [2, 4]].inv_mod(5),
        Err(SymplexError::InvalidArgument { .. })
    ));
    let half = Matrix::new(vec![
        vec![ctx.rational(1, 2), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    assert!(matches!(
        half.inv_mod(5),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matrix![ctx, [1, 2, 3]].inv_mod(5).is_err());
    assert!(a.inv_mod(1).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. LLL
// ═══════════════════════════════════════════════════════════════════════════

/// Exact Gram–Schmidt of the rows: `(μ, ‖b*ᵢ‖²)`.
fn gram_schmidt_q(b: &ZMatrix) -> (Vec<Vec<Ratio<BigInt>>>, Vec<Ratio<BigInt>>) {
    let m = b.nrows();
    let rows: Vec<Vec<Ratio<BigInt>>> = b
        .rows()
        .map(|r| r.iter().map(|v| Ratio::from_integer(v.clone())).collect())
        .collect();
    let dot = |u: &[Ratio<BigInt>], v: &[Ratio<BigInt>]| -> Ratio<BigInt> {
        u.iter().zip(v).map(|(a, c)| a * c).sum()
    };
    let mut star: Vec<Vec<Ratio<BigInt>>> = Vec::new();
    let mut norms = Vec::new();
    let mut mu = vec![vec![Ratio::from_integer(BigInt::from(0)); m]; m];
    for i in 0..m {
        let mut v = rows[i].clone();
        for j in 0..i {
            mu[i][j] = dot(&rows[i], &star[j]) / &norms[j];
            for (vk, sk) in v.iter_mut().zip(&star[j]) {
                *vk -= &mu[i][j] * sk;
            }
        }
        norms.push(dot(&v, &v));
        star.push(v);
    }
    (mu, norms)
}

fn assert_lll_reduced(b: &ZMatrix, delta: Ratio<BigInt>) {
    let (mu, norms) = gram_schmidt_q(b);
    let half = q(1, 2);
    for (i, row) in mu.iter().enumerate() {
        for (j, mu_ij) in row.iter().enumerate().take(i) {
            assert!(
                mu_ij.abs() <= half,
                "size condition fails at ({i}, {j}): μ = {mu_ij}"
            );
        }
    }
    for k in 1..b.nrows() {
        let rhs = (&delta - &mu[k][k - 1] * &mu[k][k - 1]) * &norms[k - 1];
        assert!(
            norms[k] >= rhs,
            "Lovász condition fails at k = {k}: {} < {rhs}",
            norms[k]
        );
    }
}

#[test]
fn lll_matches_sympy_and_is_exactly_reduced() {
    // SymPy: Matrix([[1, 1, 1], [-1, 0, 2], [3, 5, 6]]).lll() == Matrix([[0, 1, 0], [1, 0, 1], [-1, 0, 2]])
    let b = ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap();
    let r = b.lll_default().unwrap();
    assert_eq!(
        r,
        ZMatrix::from_i64(&[&[0, 1, 0], &[1, 0, 1], &[-1, 0, 2]]).unwrap(),
        "got {r:?}"
    );
    assert_eq!(r, b.lll(Ratio::new(3, 4)).unwrap());
    // Same lattice (same row-HNF) and exactly LLL-reduced.
    assert_eq!(r.hermite_normal_form(), b.hermite_normal_form());
    assert_lll_reduced(&r, q(3, 4));
    // Transform: R = T·B with T unimodular.
    let LllReduction {
        reduced: r2,
        transform: t,
    } = b.lll_with_transform(Ratio::new(3, 4)).unwrap();
    assert_eq!(r2, r);
    assert_eq!(&t * &b, r);
    assert!(t.is_unimodular());
    // Idempotent.
    assert_eq!(r.lll_default().unwrap(), r);
}

#[test]
fn lll_larger_examples_match_sympy() {
    // SymPy: Matrix([[1, 0, 0, 1345], [0, 1, 0, 35], [0, 0, 1, 154]]).lll()
    //   == Matrix([[0, 9, -2, 7], [1, 1, -9, -6], [1, -3, -8, 8]])
    let b = ZMatrix::from_i64(&[&[1, 0, 0, 1345], &[0, 1, 0, 35], &[0, 0, 1, 154]]).unwrap();
    let r = b.lll_default().unwrap();
    assert_eq!(
        r,
        ZMatrix::from_i64(&[&[0, 9, -2, 7], &[1, 1, -9, -6], &[1, -3, -8, 8]]).unwrap(),
        "got {r:?}"
    );
    assert_eq!(r.hermite_normal_form(), b.hermite_normal_form());
    assert_lll_reduced(&r, q(3, 4));
    // SymPy: Matrix([[1,0,0,0,-20160],[0,1,0,0,33768],[0,0,1,0,-39578],[0,0,0,1,47757]]).lll()
    //   == Matrix([[10, -3, 2, 8, -4], [3, -9, -8, 1, -11], [3, -13, -9, 3, 9], [-12, -7, 11, 9, -1]])
    let b = ZMatrix::from_i64(&[
        &[1, 0, 0, 0, -20160],
        &[0, 1, 0, 0, 33768],
        &[0, 0, 1, 0, -39578],
        &[0, 0, 0, 1, 47757],
    ])
    .unwrap();
    let r = b.lll_default().unwrap();
    assert_eq!(
        r,
        ZMatrix::from_i64(&[
            &[10, -3, 2, 8, -4],
            &[3, -9, -8, 1, -11],
            &[3, -13, -9, 3, 9],
            &[-12, -7, 11, 9, -1],
        ])
        .unwrap(),
        "got {r:?}"
    );
    assert_eq!(r.hermite_normal_form(), b.hermite_normal_form());
    assert_lll_reduced(&r, q(3, 4));
    // A stricter δ is still exactly reduced for that δ.
    let r99 = b.lll(Ratio::new(99, 100)).unwrap();
    assert_lll_reduced(&r99, q(99, 100));
    assert_eq!(r99.hermite_normal_form(), b.hermite_normal_form());
}

#[test]
fn lll_rejects_bad_delta_and_dependent_rows() {
    let b = ZMatrix::from_i64(&[&[1, 1, 1], &[-1, 0, 2], &[3, 5, 6]]).unwrap();
    for bad in [
        Ratio::new(1, 4),
        Ratio::new(1, 1),
        Ratio::new(5, 4),
        Ratio::new(0, 1),
        Ratio::new_raw(1, 0),
    ] {
        assert!(
            matches!(b.lll(bad), Err(SymplexError::InvalidArgument { .. })),
            "delta {bad:?}"
        );
    }
    // Linearly dependent rows (a lattice basis is required).
    assert!(matches!(
        ZMatrix::from_i64(&[&[1, 2], &[2, 4]])
            .unwrap()
            .lll_default(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        ZMatrix::from_i64(&[&[1, 2], &[3, 4], &[5, 6]])
            .unwrap()
            .lll_default(),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Matrix entry point: integer literals only.
    let ctx = Context::new();
    let m = matrix![ctx, [1, 1, 1], [-1, 0, 2], [3, 5, 6]];
    assert_eq!(
        m.lll_default().unwrap(),
        matrix![ctx, [0, 1, 0], [1, 0, 1], [-1, 0, 2]]
    );
    let lll = symplex::normalforms::lll_with_transform(&m, Ratio::new(3, 4)).unwrap();
    assert_eq!((&lll.transform * &m).eval(), lll.reduced);
    let half = Matrix::new(vec![vec![ctx.rational(1, 2), ctx.int(1)]]).unwrap();
    assert!(matches!(
        half.lll_default(),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. matrix_log
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_log_diagonalizable_and_defective() {
    let ctx = Context::new();
    // SymPy: Matrix([[2, 0], [0, 3]]).log() == Matrix([[log(2), 0], [0, log(3)]])
    let l = matrix![ctx, [2, 0], [0, 3]].matrix_log().unwrap();
    assert_eq!(
        l,
        Matrix::diag(&[ctx.int(2).ln(), ctx.int(3).ln()]).unwrap()
    );
    // SymPy: simplify(Matrix([[4, 1], [0, 2]]).log()) == Matrix([[log(4), log(2)/2], [0, log(2)]])
    //   ≈ [[1.38629436111989, 0.346573590279973], [0, 0.693147180559945]]
    let la = matrix![ctx, [4, 1], [0, 2]].matrix_log().unwrap();
    let ln2 = std::f64::consts::LN_2;
    let expected = [[2.0 * ln2, ln2 / 2.0], [0.0, ln2]];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (f64_of(&la[(i, j)]) - expected[i][j]).abs() < 1e-12,
                "log A[{i}][{j}] = {} ≠ {}",
                la[(i, j)],
                expected[i][j]
            );
        }
    }
    // Defective Jordan block: log [[1, 1], [0, 1]] = [[0, 1], [0, 0]]
    let n = matrix![ctx, [1, 1], [0, 1]];
    assert_eq!(n.matrix_log().unwrap(), matrix![ctx, [0, 1], [0, 0]]);
    // J_3(2): ln 2 on the diagonal, (−1)^{d+1}/(d·2^d) d places above it.
    let j3 = Matrix::jordan_block(&ctx.int(2), 3).unwrap();
    let lj = j3.matrix_log().unwrap();
    let l2 = ctx.int(2).ln();
    let expected_lj = Matrix::new(vec![
        vec![l2.clone(), ctx.rational(1, 2), ctx.rational(-1, 8)],
        vec![ctx.int(0), l2.clone(), ctx.rational(1, 2)],
        vec![ctx.int(0), ctx.int(0), l2.clone()],
    ])
    .unwrap();
    assert_eq!(lj, expected_lj, "got {lj}");
    // exp(log J) = J numerically (the exact `matrix_exp` needs eigenvalues
    // of a matrix with `ln 2` entries, which is outside its solver).
    let back = lj.exp_series(40).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (f64_of(&back[(i, j)]) - f64_of(&j3[(i, j)])).abs() < 1e-10,
                "exp(log J)[{i}][{j}] = {}",
                back[(i, j)]
            );
        }
    }
    // Singular and non-square inputs are rejected.
    assert!(matches!(
        matrix![ctx, [1, 2], [2, 4]].matrix_log(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert!(matches!(
        matrix![ctx, [1, 2, 3]].matrix_log(),
        Err(SymplexError::InvalidArgument { .. })
    ));
}
