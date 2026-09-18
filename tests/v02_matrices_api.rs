//! symplex 0.2 — `Matrix` API consistency: dummy-free eigen family,
//! `Result` fallibility, std traits, accessors.

use symplex::matrix::Matrix;
use symplex::prelude::*;

fn mi(ctx: &Context, rows: &[&[i64]]) -> Matrix {
    Matrix::from_i64(ctx, rows).unwrap()
}

/// Every entry of `m` is zero: structurally after `expand().eval()`,
/// numerically (|·| < 1e-9) for constants the simplifier cannot collapse
/// (nested radicals in denominators), or structurally after `simplify()`.
fn assert_zero(m: &Matrix, label: &str) {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let d = m.get(i, j).expand().eval();
            if d.is_zero_structural() {
                continue;
            }
            if let Ok((re, im)) = d.eval_complex64() {
                assert!(
                    re.abs() < 1e-9 && im.abs() < 1e-9,
                    "{label}: ({i},{j}) ≈ {re}+{im}i"
                );
                continue;
            }
            let d = d.simplify();
            assert!(d.is_zero_structural(), "{label}: ({i},{j}) = {d}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Eigen family: no caller-supplied dummy, no leaked reserved names
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eigenvals_no_dummy_symbol_needed() {
    let ctx = Context::new();
    let m = matrix![ctx, [2, 1], [1, 2]];
    let mut ev: Vec<String> = m
        .eigenvals()
        .unwrap()
        .iter()
        .map(|e| e.to_string())
        .collect();
    ev.sort();
    assert_eq!(ev, ["1", "3"]);
    // Nothing named "__lambda" was created for the user to see.
    for e in m.eigenvals().unwrap() {
        assert!(
            e.free_symbols().is_empty(),
            "eigenvalue has free symbols: {e}"
        );
    }
}

#[test]
fn eigenvals_with_multiplicity_groups_repeats() {
    let ctx = Context::new();
    let m = matrix![ctx, [2, 1, 0], [0, 2, 0], [0, 0, 5]];
    let mut pairs = m.eigenvals_with_multiplicity().unwrap();
    pairs.sort_by_key(|(v, _)| v.to_string());
    assert_eq!(pairs, vec![(ctx.int(2), 2), (ctx.int(5), 1)]);
    assert_eq!(m.eigenvals().unwrap().len(), 3);
}

#[test]
fn eigenvects_satisfy_definition_3x3() {
    let ctx = Context::new();
    let m = matrix![ctx, [4, 1, 2], [1, 3, 1], [2, 1, 5]];
    let evs = m.eigenvects().unwrap();
    let total: usize = evs.iter().map(|(_, mult, _)| *mult).sum();
    assert_eq!(total, 3);
    for (val, _, vecs) in &evs {
        assert!(!vecs.is_empty());
        for v in vecs {
            let residual = &(&m * v) - &v.scale(val);
            for i in 0..3 {
                let r = residual.get(i, 0).expand().simplify();
                let r = r.eval_f64().unwrap_or(f64::NAN);
                assert!(r.abs() < 1e-9, "A v − λ v ≠ 0 for λ = {val}: {r}");
            }
        }
    }
}

#[test]
fn eigen_rootof_uses_display_lambda_not_reserved_name() {
    // Companion matrix of x^5 − x − 1 (unsolvable in radicals).
    let ctx = Context::new();
    let m = matrix![
        ctx,
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ];
    let evs = m.eigenvals().unwrap();
    assert_eq!(evs.len(), 5);
    for e in &evs {
        let s = e.to_string();
        assert!(s.contains("RootOf"), "{s}");
        assert!(!s.contains("__"), "reserved name leaked: {s}");
        assert!(s.contains('λ'), "bound variable should display as λ: {s}");
    }
    // Numeric evaluation of the real root ≈ 1.1673
    let real_root = evs
        .iter()
        .filter_map(|e| e.eval_f64().ok())
        .find(|v| (v - 1.1673).abs() < 1e-3);
    assert!(
        real_root.is_some(),
        "expected the real root ≈ 1.1673 among {evs:?}"
    );
}

#[test]
fn eigen_symbolic_2x2_closed_form() {
    let ctx = Context::new();
    let (a, b, c, d) = (
        ctx.symbol("a"),
        ctx.symbol("b"),
        ctx.symbol("c"),
        ctx.symbol("d"),
    );
    let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![c.clone(), d.clone()]]).unwrap();
    let evs = m.eigenvals().unwrap();
    assert_eq!(evs.len(), 2);
    // λ₁ + λ₂ = a + d and λ₁ λ₂ = ad − bc
    let sum = (&evs[0] + &evs[1]).expand().simplify();
    assert!(
        (&sum - &(&a + &d)).expand().is_zero_structural(),
        "sum = {sum}"
    );
    let prod = (&evs[0] * &evs[1]).expand().simplify();
    assert!(
        (&prod - &(&a * &d - &b * &c)).expand().is_zero_structural(),
        "prod = {prod}"
    );
}

#[test]
fn diagonalize_reconstructs_quadratic_radical_eigenvalues() {
    let ctx = Context::new();
    // det(A − λI) = (1 − λ)(λ² − 4λ + 2): eigenvalues 1 and 2 ± √2
    // (irrational but quadratic → closed-form radicals are kept).
    let m = matrix![ctx, [2, 2, 1], [1, 2, 1], [0, 0, 1]];
    assert_eq!(m.is_diagonalizable(), Some(true));
    let (p, d) = m.diagonalize().unwrap();
    assert_eq!(d.is_diagonal(), Some(true));
    assert!(d.iter().all(|e| !e.to_string().contains("RootOf")), "{d}");
    let two_root_two = &ctx.int(2) + &ctx.int(2).sqrt();
    assert!(
        d.diagonal()
            .iter()
            .any(|e| e.equals(&two_root_two) == Some(true)),
        "{d}"
    );
    let back = &(&p * &d) * &p.inv().unwrap();
    assert_zero(&(&back - &m), "P D P⁻¹ = A");
}

#[test]
fn diagonalize_reconstructs_rational_eigenvalues() {
    let ctx = Context::new();
    let m = matrix![ctx, [4, 1, 2], [0, 3, 1], [0, 0, 5]];
    let (p, d) = m.diagonalize().unwrap();
    let mut diag: Vec<String> = d.diagonal().iter().map(|e| e.to_string()).collect();
    diag.sort();
    assert_eq!(diag, ["3", "4", "5"]);
    let back = (&(&p * &d) * &p.inv().unwrap()).eval();
    assert_eq!(back, m);
}

#[test]
fn jordan_form_reconstructs_defective_3x3() {
    let ctx = Context::new();
    // Defective 3×3: one Jordan block of size 2 and one of size 1.
    let j3 = matrix![ctx, [5, 1, 0], [0, 5, 0], [0, 0, 7]];
    assert_eq!(j3.is_diagonalizable(), Some(false));
    assert!(j3.diagonalize().is_err());
    let (p, j) = j3.jordan_form().unwrap();
    assert_zero(&(&(&(&p * &j) * &p.inv().unwrap()) - &j3), "P J P⁻¹ = A");
    assert!(j.get(0, 1).is_one_structural() || j.get(1, 2).is_one_structural());
}

#[test]
fn jordan_form_two_nilpotent_blocks() {
    let ctx = Context::new();
    let n = matrix![ctx, [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]];
    let (p, j) = n.jordan_form().unwrap();
    assert_eq!((&(&p * &j) * &p.inv().unwrap()).eval(), n);
    let ones = j.iter().filter(|e| e.is_one_structural()).count();
    assert_eq!(ones, 2, "two J₂(0) blocks: {j}");
}

/// Irreducible cubic characteristic polynomial (casus irreducibilis):
/// the Cardano forms would make `P⁻¹` swell without bound, so the eigen
/// family must answer with `RootOf` values — quickly.
#[test]
fn cubic_eigenvalues_use_rootof_and_finish_fast() {
    let ctx = Context::new();
    let m = matrix![ctx, [4, 1, 2], [1, 3, 1], [2, 1, 5]];
    let start = std::time::Instant::now();
    let ev = m.eigenvals().unwrap();
    assert_eq!(ev.len(), 3);
    for e in &ev {
        let s = e.to_string();
        assert!(
            s.contains("RootOf") && s.contains('λ') && !s.contains("__"),
            "{s}"
        );
    }
    let (p, d) = m.diagonalize().unwrap();
    let back = &(&p * &d) * &p.inv().unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let v = (back.get(i, j) - m.get(i, j)).eval_f64().unwrap();
            assert!(v.abs() < 1e-9, "P D P⁻¹ ≠ A at ({i},{j}): {v}");
        }
    }
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "took {:?}",
        start.elapsed()
    );
}

#[test]
fn compact_cubic_and_biquadratic_keep_radicals() {
    let ctx = Context::new();
    // λ³ = 2 → cbrt(2)·ωᵏ (companion matrix)
    let c = matrix![ctx, [0, 0, 2], [1, 0, 0], [0, 1, 0]];
    let ev = c.eigenvals().unwrap();
    assert_eq!(ev.len(), 3);
    assert!(
        ev.iter().all(|e| !e.to_string().contains("RootOf")),
        "{ev:?}"
    );
    let real = ev
        .iter()
        .filter_map(|e| e.eval_f64().ok())
        .find(|v| (v - 2f64.cbrt()).abs() < 1e-9);
    assert!(real.is_some(), "{ev:?}");
    // λ⁴ − 10λ² + 1 → ±√2 ± √3
    let b = matrix![
        ctx,
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 10],
        [0, 0, 1, 0]
    ];
    let ev = b.eigenvals().unwrap();
    assert_eq!(ev.len(), 4);
    assert!(
        ev.iter().all(|e| !e.to_string().contains("RootOf")),
        "{ev:?}"
    );
    let mut vals: Vec<f64> = ev.iter().map(|e| e.eval_f64().unwrap()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (s2, s3) = (2f64.sqrt(), 3f64.sqrt());
    let expect = [-s2 - s3, -s3 + s2, s3 - s2, s2 + s3];
    for (g, e) in vals.iter().zip(expect) {
        assert!((g - e).abs() < 1e-9, "{vals:?}");
    }
}

#[test]
fn matrix_exp_t_rotation_generator_gives_trig() {
    let ctx = Context::new();
    let (w, t) = (ctx.symbol("omega"), ctx.symbol("t"));
    let g = Matrix::new(vec![vec![ctx.zero(), -&w], vec![w.clone(), ctx.zero()]]).unwrap();
    let e = g.matrix_exp_t(&t).unwrap();
    let wt = &w * &t;
    assert_eq!(e[(0, 0)], wt.cos());
    assert_eq!(e[(1, 1)], wt.cos());
    assert_eq!(e[(1, 0)], wt.sin());
    assert_eq!(e[(0, 1)], -&wt.sin());
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-swell budget
// ═══════════════════════════════════════════════════════════════════════════

/// A tiny DAG whose tree is enormous (≈ 2¹⁸ nodes): eₙ₊₁ = sin(eₙ) + cos(eₙ).
fn huge_tree(ctx: &Context) -> Ex {
    let mut e = ctx.symbol("x");
    for _ in 0..16 {
        e = &e.sin() + &e.cos();
    }
    e
}

fn is_swell<T>(r: Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { reason, .. }) if reason.starts_with("expression swell"))
}

#[test]
fn budget_rejects_huge_inputs_quickly() {
    let ctx = Context::new();
    let e = huge_tree(&ctx);
    let m = Matrix::new(vec![
        vec![e.clone(), ctx.int(1)],
        vec![ctx.int(1), e.clone()],
    ])
    .unwrap();
    let start = std::time::Instant::now();
    assert!(is_swell(m.det()));
    assert!(is_swell(m.inv()));
    assert!(is_swell(m.solve(&matrix![ctx, [1], [1]])));
    assert!(is_swell(m.char_poly_coeffs()));
    assert!(is_swell(m.diagonalize()));
    assert!(is_swell(m.jordan_form()));
    assert!(is_swell(m.matrix_exp()));
    assert!(is_swell(m.qr()));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn budget_passes_ordinary_symbolic_work() {
    let ctx = Context::new();
    let syms: Vec<Ex> = (0..25).map(|k| ctx.symbol(&format!("a{k}"))).collect();
    let m = Matrix::from_fn(5, 5, |i, j| syms[i * 5 + j].clone());
    let start = std::time::Instant::now();
    let d = m.det().unwrap();
    assert_eq!(d.term_count(), 120);
    let lam = ctx.symbol("lambda");
    let cp = m.char_poly(&lam).unwrap();
    // Leading term (−λ)⁵ present, constant term is det(A).
    assert!(cp.contains(&lam.powi(5)), "{cp}");
    assert_eq!(cp.subs(&lam, &ctx.zero()).expand(), d.expand());
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn matrix_exp_exact_and_exp_t() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let a = matrix![ctx, [1, 1], [0, 1]]; // Jordan block J₂(1)
    let e = a.matrix_exp().unwrap();
    // e^A = e · [[1, 1], [0, 1]]
    assert_eq!(e.get(0, 0), &ctx.e());
    assert_eq!(e.get(0, 1), &ctx.e());
    assert!(e.get(1, 0).is_zero_structural());
    let et = a.matrix_exp_t(&t).unwrap();
    // d/dt e^{At} = A e^{At}
    let lhs = et.diff(&t).simplify();
    let rhs = (&a * &et).simplify();
    assert_zero(&(&lhs - &rhs), "d/dt e^{At} = A e^{At}");
    // e^{A·0} = I
    assert_eq!(
        et.subs(&t, &ctx.int(0)).eval().simplify(),
        Matrix::identity(&ctx, 2)
    );
    // Non-square → InvalidArgument
    let err = matrix![ctx, [1, 2, 3]].matrix_exp().unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
}

#[test]
fn char_poly_coeffs_ascending_and_consistent() {
    let ctx = Context::new();
    let lam = ctx.symbol("lambda");
    let m = matrix![ctx, [1, 2, 3], [0, 4, 5], [1, 0, 6]];
    let coeffs = m.char_poly_coeffs().unwrap();
    assert_eq!(coeffs.len(), 4);
    assert_eq!(coeffs[0], m.det().unwrap());
    assert_eq!(coeffs[3], ctx.int(-1));
    let p = m.char_poly(&lam).unwrap();
    let rebuilt = coeffs
        .iter()
        .enumerate()
        .fold(ctx.zero(), |acc, (k, c)| acc + c * &lam.powi(k as i64));
    assert!((&p - &rebuilt).expand().is_zero_structural());
    // Cayley–Hamilton
    let mut ph = Matrix::zeros(&ctx, 3, 3);
    for (k, c) in coeffs.iter().enumerate() {
        ph = &ph + &(&m.powi(k as u32).unwrap() * c);
    }
    assert_zero(&ph, "p(A) = 0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Fallibility: every failing operation is an Err, never a panic / Option
// ═══════════════════════════════════════════════════════════════════════════

fn is_inv<T>(r: Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::InvalidArgument { .. }))
}

fn is_cf<T>(r: Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

#[test]
fn shape_errors_are_invalid_argument() {
    let ctx = Context::new();
    let rect = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    let sq = matrix![ctx, [1, 2], [3, 4]];
    assert!(is_inv(rect.det()));
    assert!(is_inv(rect.trace()));
    assert!(is_inv(rect.inv()));
    assert!(is_inv(rect.lu()));
    assert!(is_inv(rect.eigenvals()));
    assert!(is_inv(rect.eigenvects()));
    assert!(is_inv(rect.jordan_form()));
    assert!(is_inv(rect.diagonalize()));
    assert!(is_inv(rect.char_poly_coeffs()));
    assert!(is_inv(rect.cholesky()));
    assert!(is_inv(rect.ldl()));
    assert!(is_inv(rect.powi(2)));
    assert!(is_inv(rect.add(&sq)));
    assert!(is_inv(rect.sub(&sq)));
    assert!(is_inv(sq.matmul(&rect.transpose().transpose().transpose())));
    assert!(is_inv(Matrix::hstack(&[
        &rect,
        &sq.transpose().transpose(),
        &matrix![ctx, [1]]
    ])));
    assert!(is_inv(Matrix::vstack(&[&rect, &sq])));
    assert!(is_inv(Matrix::hstack(&[])));
    assert!(is_inv(Matrix::new(vec![])));
    assert!(is_inv(Matrix::from_i64(&ctx, &[&[1], &[2, 3]])));
}

#[test]
fn math_failures_are_computation_failed() {
    let ctx = Context::new();
    let singular = matrix![ctx, [1, 2], [2, 4]];
    assert!(is_cf(singular.inv()));
    assert!(is_cf(singular.lu()));
    assert!(is_cf(singular.solve(&matrix![ctx, [1], [1]])));
    assert!(is_cf(singular.pinv()));
    assert!(is_cf(matrix![ctx, [1, 1], [0, 1]].diagonalize()));
    assert!(is_cf(matrix![ctx, [1, 2], [2, 1]].cholesky()));
    assert!(is_cf(matrix![ctx, [1, 2], [2, 4]].qr()));
}

#[test]
fn try_get_and_index() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(m.try_get(1, 1), Some(&ctx.int(4)));
    assert_eq!(m.try_get(2, 0), None);
    assert_eq!(m[(0, 1)], ctx.int(2));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn index_out_of_bounds_panics_like_slices() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let _ = m[(5, 5)].clone();
}

// ═══════════════════════════════════════════════════════════════════════════
// std traits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn partial_eq_is_structural_same_context() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = mi(&ctx, &[&[1, 2], &[3, 4]]);
    assert_eq!(a, b);
    assert_ne!(a, a.transpose());
    let x = ctx.symbol("x");
    let s1 = Matrix::new(vec![vec![&x.sin().powi(2) + &x.cos().powi(2)]]).unwrap();
    let s2 = Matrix::new(vec![vec![ctx.int(1)]]).unwrap();
    assert_ne!(s1, s2, "structural equality does not simplify");
    assert_eq!(s1.equals(&s2), Some(true), "mathematical equality does");
}

#[test]
fn display_aligns_columns_and_debug_is_compact() {
    let ctx = Context::new();
    let m = mi(&ctx, &[&[1, 200], &[-30, 4]]);
    assert_eq!(format!("{m}"), "[\n  [  1, 200],\n  [-30,   4]\n]");
    assert_eq!(format!("{m:?}"), "Matrix(2×2, [[1, 200], [-30, 4]])");
    let row = matrix![ctx, [1, 2, 3]];
    assert_eq!(format!("{row}"), "[[1, 2, 3]]");
}

#[test]
fn operators_scalar_both_sides_and_div() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(&m * &x, &x * &m);
    assert_eq!(m.clone() * x.clone(), x.clone() * m.clone());
    assert_eq!(&m * 2, 2 * &m);
    assert_eq!(2 * m.clone(), m.clone() * 2);
    assert_eq!((&m / 2)[(1, 1)], ctx.int(2));
    assert_eq!((&m / &ctx.int(4))[(0, 1)], ctx.rational(1, 2));
    assert_eq!(-&m, &m * -1);
    assert_eq!(&(&m + &m) - &m, m);
    assert_eq!(&m * &Matrix::identity(&ctx, 2), m);
    let mut im = m.clone();
    im[(0, 0)] = ctx.int(9);
    assert_eq!(im[(0, 0)], ctx.int(9));
}

#[test]
fn try_from_nested_vec() {
    let ctx = Context::new();
    let ok: Result<Matrix, _> = vec![vec![ctx.int(1), ctx.int(2)]].try_into();
    assert_eq!(ok.unwrap().shape(), (1, 2));
    let bad: Result<Matrix, _> = vec![vec![ctx.int(1)], vec![]].try_into();
    assert!(bad.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessors and constructors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn accessors() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    assert_eq!(m.col(1), vec![ctx.int(2), ctx.int(5), ctx.int(8)]);
    assert_eq!(m.row(2), &[ctx.int(7), ctx.int(8), ctx.int(9)]);
    assert_eq!(m.diagonal(), vec![ctx.int(1), ctx.int(5), ctx.int(9)]);
    assert_eq!(m.submatrix(1..3, 1..3), matrix![ctx, [5, 6], [8, 9]]);
    assert_eq!(m.minor_matrix(0, 0).unwrap(), matrix![ctx, [5, 6], [8, 9]]);
    assert_eq!(m.minor(0, 0).unwrap(), ctx.int(-3));
    assert_eq!(m.cofactor(0, 1).unwrap(), ctx.int(6));
    assert_eq!(m.iter().count(), 9);
    assert_eq!(m.to_vec()[2][0], ctx.int(7));
    assert_eq!(m.eval_f64().unwrap()[1], vec![4.0, 5.0, 6.0]);
    let x = ctx.symbol("x");
    let sym = Matrix::new(vec![vec![x.clone()]]).unwrap();
    assert!(sym.eval_f64().is_err());
    let mut s = m.clone();
    s.set(1, 1, ctx.int(0));
    assert_eq!(s[(1, 1)], ctx.int(0));
    let indexed = m.map_indexed(|i, j, e| if i == j { e.clone() } else { ctx.zero() });
    assert_eq!(indexed, Matrix::diag(&[ctx.int(1), ctx.int(5), ctx.int(9)]));
    assert_eq!(m.vec().shape(), (9, 1));
    assert_eq!(m.vec()[(1, 0)], ctx.int(4));
    assert_eq!(m.context().int(3), ctx.int(3));
}

#[test]
fn block_diag_and_stacking() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = matrix![ctx, [5]];
    let bd = Matrix::block_diag(&[&a, &b]).unwrap();
    assert_eq!(bd, matrix![ctx, [1, 2, 0], [3, 4, 0], [0, 0, 5]]);
    assert_eq!(bd.det().unwrap(), ctx.int(-10));
    let h = Matrix::hstack(&[&a, &a]).unwrap();
    assert_eq!(h.shape(), (2, 4));
    let v = Matrix::vstack(&[&a, &a]).unwrap();
    assert_eq!(v.shape(), (4, 2));
    assert_eq!(a.kronecker(&Matrix::identity(&ctx, 2)).shape(), (4, 4));
}

#[test]
fn from_i64_takes_context() {
    let ctx = Context::new();
    let m = Matrix::from_i64(&ctx, &[&[1, 2], &[3, 4]]).unwrap();
    // Same context as ctx → can be combined with expressions from ctx.
    let x = ctx.symbol("x");
    let _ = &m * &x;
    assert_eq!(m, matrix![ctx, [1, 2], [3, 4]]);
}

#[test]
fn det_symbolic_4x4_is_expanded_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::from_fn(4, 4, |i, j| {
        if i == j {
            &x + ctx.int(i as i64)
        } else {
            ctx.int((i + j) as i64 % 3)
        }
    });
    let d = m.det().unwrap();
    assert!(d.is_polynomial(&x), "det must be a polynomial in x: {d}");
    assert_eq!(d.degree(&x), Some(4));
    // Agrees with numeric Bareiss after substitution
    for v in [0i64, 1, -2, 5] {
        let a = m.subs(&x, &ctx.int(v)).det().unwrap().eval();
        let b = d.subs(&x, &ctx.int(v)).eval();
        assert_eq!(a, b, "x = {v}");
    }
}

#[test]
fn integrate_and_simplify_elementwise() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x.powi(2), x.cos()]]).unwrap();
    let anti = m.integrate(&x);
    assert_eq!(anti.diff(&x).simplify(), m);
}
