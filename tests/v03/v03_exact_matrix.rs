//! symplex 0.3.5 — the exact matrix core: `QMatrix` / `ZMatrix`, the
//! fraction-free (Bareiss) elimination behind them, the `Matrix` fast paths
//! that route rational input through them, and the integer-pivoting simplex
//! tableau in `linprog`.
//!
//! Every fast path is cross-checked against an independent computation:
//! `QMatrix` against plain Gauss–Jordan over `Ratio<BigInt>`, the `Matrix`
//! methods against `QMatrix`, the symbolic `linsolve` route against the
//! numeric one, and every LP optimum against the exact KKT conditions.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use symplex::linprog::{
    Feasibility, LpProblem, LpStatus, Q, feasible_nonneg_certified, nonneg_combination, q, qi,
};
use symplex::matrix::{ExactMatrix, QMatrix, ZMatrix};
use symplex::normalforms;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }

    fn rational(&mut self, hi: i64, dmax: i64) -> Q {
        q(self.range(-hi, hi), self.range(1, dmax))
    }
}

fn random_q(n: usize, m: usize, seed: u64) -> QMatrix {
    let mut g = Lcg(seed);
    QMatrix::from_fn(n, m, |_, _| g.rational(9, 4))
}

fn random_z(n: usize, m: usize, seed: u64) -> ZMatrix {
    let mut g = Lcg(seed);
    ZMatrix::from_fn(n, m, |_, _| BigInt::from(g.range(-9, 9)))
}

/// Reference RREF: textbook Gauss–Jordan over `Ratio<BigInt>`.
fn reference_rref(a: &QMatrix) -> (QMatrix, Vec<usize>) {
    let mut rows = a.to_rows();
    let (n, m) = a.shape();
    let mut pivots = Vec::new();
    let mut pr = 0;
    for c in 0..m {
        if pr >= n {
            break;
        }
        let Some(p) = (pr..n).find(|&i| !rows[i][c].is_zero()) else {
            continue;
        };
        rows.swap(pr, p);
        let pv = rows[pr][c].clone();
        for v in rows[pr].iter_mut() {
            *v = &*v / &pv;
        }
        let prow = rows[pr].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i == pr || row[c].is_zero() {
                continue;
            }
            let f = row[c].clone();
            for (v, p) in row.iter_mut().zip(&prow) {
                *v = &*v - &f * p;
            }
        }
        pivots.push(c);
        pr += 1;
    }
    (QMatrix::new(rows).unwrap(), pivots)
}

fn dot(a: &[Q], b: &[Q]) -> Q {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// QMatrix / ZMatrix public surface
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn constructors_validate_shape() {
    assert!(QMatrix::new(vec![]).is_err());
    assert!(QMatrix::new(vec![vec![]]).is_err());
    assert!(QMatrix::new(vec![vec![qi(1)], vec![qi(1), qi(2)]]).is_err());
    assert!(ZMatrix::from_flat(2, 3, vec![BigInt::zero(); 5]).is_err());
    let m = ZMatrix::from_flat(2, 3, (1..=6).map(BigInt::from).collect()).unwrap();
    assert_eq!(
        m.row(1),
        &[BigInt::from(4), BigInt::from(5), BigInt::from(6)]
    );
    assert_eq!(m.into_flat().len(), 6);
    let e = ExactMatrix::<BigInt>::identity(2);
    assert!(e.is_identity());
    assert_eq!(
        ZMatrix::diag(&[BigInt::from(2), BigInt::from(3)]).diagonal(),
        vec![BigInt::from(2), BigInt::from(3)]
    );
}

#[test]
fn display_and_debug_match_matrix_layout() {
    let ctx = Context::new();
    let sym = matrix![ctx, [1, -20], [3, 4]];
    let z = ZMatrix::try_from(&sym).unwrap();
    assert_eq!(format!("{z}"), format!("{sym}"));
    assert_eq!(format!("{:?}", z), "ZMatrix(2×2, [[1, -20], [3, 4]])");
    let half = QMatrix::new(vec![vec![q(1, 2), qi(3)]]).unwrap();
    assert_eq!(format!("{half}"), "[[1/2, 3]]");
    assert_eq!(format!("{half:?}"), "QMatrix(1×2, [[1/2, 3]])");
}

#[test]
fn arithmetic_operators_and_shape_errors() {
    let a = QMatrix::from_i64(&[&[1, 2], &[3, 4]]).unwrap();
    let b = QMatrix::from_i64(&[&[0, 1], &[1, 0]]).unwrap();
    assert_eq!(&a * &b, QMatrix::from_i64(&[&[2, 1], &[4, 3]]).unwrap());
    assert_eq!(&a + &b - &b, a);
    assert_eq!(-&a + &a, QMatrix::zeros(2, 2));
    assert_eq!(
        &a * &q(1, 2),
        QMatrix::new(vec![vec![q(1, 2), qi(1)], vec![q(3, 2), qi(2)]]).unwrap()
    );
    assert_eq!(a.trace().unwrap(), qi(5));
    assert_eq!(a.transpose()[(0, 1)], qi(3));
    let wide = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
    assert_eq!(a.matmul(&wide).unwrap().shape(), (2, 3));
    assert!(wide.matmul(&a).is_err());
    assert!(a.add(&wide).is_err());
    assert!(wide.trace().is_err());
    assert!(wide.det().is_err());
    assert!(wide.inv().is_err());
    assert_eq!(QMatrix::hstack(&[&a, &wide]).unwrap().shape(), (2, 5));
    assert!(QMatrix::vstack(&[&a, &wide]).is_err());
    assert!(a.submatrix(0..2, 1..3).is_err());
}

#[test]
fn indexing_out_of_bounds_panics_but_try_get_does_not() {
    let a = ZMatrix::identity(2);
    assert!(a.try_get(2, 0).is_none());
    assert_eq!(a.try_get(1, 1), Some(&BigInt::one()));
    let r = std::panic::catch_unwind(|| a[(2, 0)].clone());
    assert!(r.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Fraction-free elimination is exact
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn qmatrix_rref_agrees_with_textbook_gauss_jordan() {
    for seed in 1..=20u64 {
        for &(n, m) in &[(3usize, 3usize), (4, 7), (7, 4), (6, 6), (5, 9)] {
            let a = random_q(n, m, seed * 97 + (n * 13 + m) as u64);
            let (r, p) = a.rref();
            let (r_ref, p_ref) = reference_rref(&a);
            assert_eq!(p, p_ref, "pivots, seed {seed}, {n}×{m}");
            assert_eq!(r, r_ref, "rref, seed {seed}, {n}×{m}");
            assert_eq!(a.rank(), p.len());
            for v in a.nullspace() {
                assert!((&a * &v).is_zero());
            }
            assert_eq!(a.nullspace().len(), m - p.len());
            assert_eq!(a.columnspace().len(), p.len());
            assert_eq!(a.rowspace().len(), p.len());
        }
    }
}

#[test]
fn qmatrix_rank_deficient_inputs() {
    // Rows 2 and 3 are multiples of row 1; a zero column in the middle.
    let a = QMatrix::new(vec![
        vec![q(1, 2), qi(0), q(1, 3)],
        vec![qi(1), qi(0), q(2, 3)],
        vec![q(-3, 2), qi(0), qi(-1)],
    ])
    .unwrap();
    let (r, p) = a.rref();
    assert_eq!(p, vec![0]);
    assert_eq!(r.row(0), &[qi(1), qi(0), q(2, 3)]);
    assert!(r.row(1).iter().all(Zero::is_zero));
    assert_eq!(a.nullspace().len(), 2);
    assert_eq!(a.det().unwrap(), Q::zero());
    assert!(a.inv().is_err());
    assert_eq!(QMatrix::zeros(3, 2).rank(), 0);
    assert_eq!(QMatrix::identity(4).rref().1, vec![0, 1, 2, 3]);
}

#[test]
fn qmatrix_det_inv_solve_identities() {
    for seed in 1..=15u64 {
        let a = random_q(5, 5, seed);
        let b = random_q(5, 2, seed + 500);
        let da = a.det().unwrap();
        // Multiplicativity against an independent matrix.
        let c = random_q(5, 5, seed + 1000);
        assert_eq!((&a * &c).det().unwrap(), &da * c.det().unwrap());
        // Transpose invariance.
        assert_eq!(a.transpose().det().unwrap(), da);
        if da.is_zero() {
            assert!(a.inv().is_err());
            assert!(a.solve(&b).is_err());
            continue;
        }
        let inv = a.inv().unwrap();
        assert!((&a * &inv).is_identity());
        assert!((&inv * &a).is_identity());
        assert_eq!(inv.det().unwrap(), Q::one() / &da);
        let x = a.solve(&b).unwrap();
        assert_eq!(&a * &x, b);
        assert_eq!(x, &inv * &b);
    }
}

#[test]
fn qmatrix_det_of_fractional_matrix() {
    // Hilbert-like 4×4: det(H₄) = 1/6048000.
    let h = QMatrix::from_fn(4, 4, |i, j| q(1, (i + j + 1) as i64));
    assert_eq!(h.det().unwrap(), q(1, 6_048_000));
    let inv = h.inv().unwrap();
    assert!(inv.is_integer(), "the inverse Hilbert matrix is integral");
    assert_eq!(inv[(3, 3)], qi(2800));
}

#[test]
fn zmatrix_det_rank_content() {
    let a = ZMatrix::from_i64(&[&[2, 0, 1], &[1, 3, 2], &[1, 1, 3]]).unwrap();
    // 2·(9 − 2) − 0 + 1·(1 − 3) = 12
    assert_eq!(a.det().unwrap(), BigInt::from(12));
    assert_eq!(a.rank(), 3);
    assert_eq!(a.content(), BigInt::one());
    let singular = ZMatrix::from_i64(&[&[2, 0, 1], &[1, 3, 2], &[1, 1, 1]]).unwrap();
    assert_eq!(singular.det().unwrap(), BigInt::zero());
    assert_eq!(singular.rank(), 2);
    assert_eq!(
        ZMatrix::from_i64(&[&[6, 9], &[3, 12]]).unwrap().content(),
        BigInt::from(3)
    );
    assert_eq!(ZMatrix::zeros(2, 2).content(), BigInt::zero());
    for seed in 1..=10u64 {
        let z = random_z(6, 6, seed);
        assert_eq!(
            z.det().unwrap(),
            *z.to_qmatrix().det().unwrap().numer(),
            "ℤ and ℚ determinants agree"
        );
        assert_eq!(z.rank(), z.to_qmatrix().rank());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix fast paths give the same answers as the exact core
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn matrix_rref_rank_nullspace_route_through_qmatrix() {
    let ctx = Context::new();
    for seed in 1..=8u64 {
        let qm = random_q(5, 7, seed + 40);
        let m = qm.to_matrix(&ctx);
        let (r, p) = m.rref();
        let (rq, pq) = qm.rref();
        assert_eq!(p, pq);
        assert_eq!(QMatrix::try_from(&r).unwrap(), rq);
        assert_eq!(m.rank(), qm.rank());
        let ns = m.nullspace();
        assert_eq!(ns.len(), qm.nullspace().len());
        for v in &ns {
            let prod = (&m * v).eval();
            assert!(prod.iter().all(|e| e.is_zero_structural()));
        }
        assert_eq!(m.columnspace().len(), p.len());
        assert_eq!(m.rowspace().len(), p.len());
        assert_eq!(m.left_nullspace().len(), 5 - p.len());
    }
}

#[test]
fn matrix_det_inv_solve_route_through_qmatrix() {
    let ctx = Context::new();
    for seed in 1..=8u64 {
        let qa = random_q(5, 5, seed + 70);
        let a = qa.to_matrix(&ctx);
        let d = a.det().unwrap();
        assert_eq!(d.as_rational().unwrap(), qa.det().unwrap());
        if d.is_zero_structural() {
            assert!(a.inv().is_err());
            assert!(a.solve(&Matrix::identity(&ctx, 5)).is_err());
            continue;
        }
        let inv = a.inv().unwrap();
        assert_eq!(QMatrix::try_from(&inv).unwrap(), qa.inv().unwrap());
        assert_eq!((&a * &inv).eval(), Matrix::identity(&ctx, 5));
        let b = random_q(5, 1, seed + 90).to_matrix(&ctx);
        let x = a.solve(&b).unwrap();
        assert_eq!((&a * &x).eval(), b);
    }
    // 4×4 is the first size that routes `det` through the exact core;
    // the 3×3 cofactor path and the exact path must agree.
    let z3 = random_z(3, 3, 3).to_matrix(&ctx);
    let z4 = random_z(4, 4, 4).to_matrix(&ctx);
    assert_eq!(
        z3.det().unwrap().as_bigint().unwrap(),
        ZMatrix::try_from(&z3).unwrap().det().unwrap()
    );
    assert_eq!(
        z4.det().unwrap().as_bigint().unwrap(),
        ZMatrix::try_from(&z4).unwrap().det().unwrap()
    );
}

#[test]
fn matrix_singular_errors_keep_their_messages() {
    let ctx = Context::new();
    let s = matrix![ctx, [1, 2], [2, 4]];
    let e = s.inv().unwrap_err().to_string();
    assert!(e.contains("inv") && e.contains("singular"), "{e}");
    let e = s.solve(&matrix![ctx, [1], [1]]).unwrap_err().to_string();
    assert!(e.contains("solve") && e.contains("singular"), "{e}");
    let e = matrix![ctx, [1, 2, 3]].det().unwrap_err().to_string();
    assert!(e.contains("square"), "{e}");
}

#[test]
fn symbolic_matrices_still_take_the_symbolic_path() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let m = Matrix::new(vec![vec![a.clone(), b.clone()], vec![b.clone(), a.clone()]]).unwrap();
    assert!(QMatrix::try_from(&m).is_err());
    let d = m.det().unwrap();
    assert_eq!(d, a.powi(2) - b.powi(2));
    let (r, p) = m.rref();
    assert_eq!(p, vec![0, 1]);
    assert_eq!(r, Matrix::identity(&ctx, 2));
    // Mixed: a single symbol disables the fast path but the answer matches
    // the numeric one after substitution.
    let mixed = Matrix::new(vec![
        vec![a.clone(), ctx.int(2)],
        vec![ctx.int(3), ctx.int(4)],
    ])
    .unwrap();
    let at5 = mixed.subs(&a, &ctx.int(5));
    assert_eq!(
        mixed.inv().unwrap().subs(&a, &ctx.int(5)).eval(),
        at5.inv().unwrap()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// linsolve / linsolve_matrix numeric fast path
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linsolve_numeric_fast_path_matches_symbolic_route() {
    let ctx = Context::new();
    let (x, y, z, a) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("a"),
    );
    // Symbolic coefficient `a` forces the expression route; substituting
    // `a = 2` afterwards must agree with solving the numeric system.
    let sym = linsolve(
        &[
            &a * &x + &y * 3 - &z - 7,
            &x * 2 - &y + &z * 4 - 1,
            &x + &y + &z - 6,
        ],
        &[x.clone(), y.clone(), z.clone()],
    )
    .unwrap();
    let num = linsolve(
        &[
            &x * 2 + &y * 3 - &z - 7,
            &x * 2 - &y + &z * 4 - 1,
            &x + &y + &z - 6,
        ],
        &[x.clone(), y.clone(), z.clone()],
    )
    .unwrap();
    let (LinearSolution::Unique(s), LinearSolution::Unique(n)) = (sym, num) else {
        panic!("expected unique solutions");
    };
    for ((vs, es), (vn, en)) in s.iter().zip(&n) {
        assert_eq!(vs, vn);
        assert_eq!(es.subs(&a, &ctx.int(2)).eval(), *en);
    }
}

#[test]
fn linsolve_numeric_parametric_and_inconsistent() {
    let ctx = Context::new();
    let (x, y, z) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("z"));
    // One equation, three unknowns with fractional coefficients.
    let sol = linsolve(
        &[&x * ctx.rational(1, 2) + &y * ctx.rational(1, 3) + &z - 1],
        &[x.clone(), y.clone(), z.clone()],
    )
    .unwrap();
    match sol {
        LinearSolution::Parametric { solution, free } => {
            assert_eq!(free, vec![y.clone(), z.clone()]);
            let (var, val) = &solution[0];
            assert_eq!(*var, x);
            // The residual of the equation vanishes identically.
            let residual = (&x * ctx.rational(1, 2) + &y * ctx.rational(1, 3) + &z - 1)
                .subs(&x, val)
                .expand()
                .eval();
            assert!(residual.is_zero_structural(), "residual {residual}");
            // Byte-identical to the symbolic route's output for this system.
            assert_eq!(val.to_string(), "-2/3*y - 2*z + 2");
        }
        other => panic!("expected parametric, got {other:?}"),
    }
    let bad = linsolve(&[&x + &y - 1, &x + &y - 2], &[x.clone(), y.clone()]).unwrap();
    assert!(matches!(bad, LinearSolution::Inconsistent));
    // Over-determined but consistent.
    let ok = linsolve(&[&x - 1, &y - 2, &x + &y - 3], &[x.clone(), y.clone()]).unwrap();
    assert!(matches!(ok, LinearSolution::Unique(_)));
}

#[test]
fn linsolve_matrix_numeric_rectangular() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    let b = Matrix::col_vector(vec![ctx.int(6), ctx.int(15)]);
    match linsolve_matrix(&a, &b).unwrap() {
        LinearSolution::Parametric { solution, free } => {
            assert_eq!(free.len(), 1);
            assert_eq!(free[0].to_string(), "x3");
            assert_eq!(solution[0].1.to_string(), "x3");
            assert_eq!(solution[1].1.to_string(), "-2*x3 + 3");
        }
        other => panic!("expected parametric, got {other:?}"),
    }
    let b_bad = Matrix::col_vector(vec![ctx.int(6), ctx.int(16)]);
    let a_bad = matrix![ctx, [1, 2, 3], [2, 4, 6]];
    assert!(matches!(
        linsolve_matrix(&a_bad, &b_bad).unwrap(),
        LinearSolution::Inconsistent
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Normal forms: ZMatrix methods and the Matrix wrappers agree
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn normal_forms_on_zmatrix_and_matrix_agree() {
    let ctx = Context::new();
    for seed in 1..=8u64 {
        let z = random_z(4, 5, seed + 200);
        let m = z.to_matrix(&ctx);
        assert_eq!(
            ZMatrix::try_from(&normalforms::hermite_normal_form(&m).unwrap()).unwrap(),
            z.hermite_normal_form()
        );
        assert_eq!(
            ZMatrix::try_from(&normalforms::column_hermite_normal_form(&m).unwrap()).unwrap(),
            z.column_hermite_normal_form()
        );
        assert_eq!(
            ZMatrix::try_from(&normalforms::smith_normal_form(&m).unwrap()).unwrap(),
            z.smith_normal_form()
        );
        let (h, u) = z.hermite_normal_form_with_transform();
        assert_eq!(&u * &z, h);
        assert!(u.is_unimodular());
        let (s, u, v) = z.smith_normal_form_with_transforms();
        assert_eq!(&(&u * &z) * &v, s);
        let diag = s.diagonal();
        for w in diag.windows(2) {
            if !w[1].is_zero() {
                assert!(
                    (&w[1] % &w[0]).is_zero(),
                    "invariant factors divide: {diag:?}"
                );
            }
        }
        let kernel = z.integer_nullspace();
        assert_eq!(kernel.len(), 5 - z.rank());
        for k in &kernel {
            assert!((&z * k).is_zero());
        }
        assert_eq!(
            normalforms::integer_nullspace(&m).unwrap().len(),
            kernel.len()
        );
        let square = random_z(4, 4, seed + 300);
        assert_eq!(
            normalforms::is_unimodular(&square.to_matrix(&ctx)).unwrap(),
            square.is_unimodular()
        );
        assert_eq!(square.is_unimodular(), square.det().unwrap().abs().is_one());
    }
    assert!(
        ZMatrix::from_i64(&[&[1, 2], &[2, 4]])
            .unwrap()
            .lattice_determinant()
            .is_err()
    );
    assert_eq!(
        ZMatrix::from_i64(&[&[2, 0, 1], &[0, 3, 1]])
            .unwrap()
            .lattice_determinant()
            .unwrap(),
        BigInt::one()
    );
    let e = normalforms::hermite_normal_form(&Matrix::new(vec![vec![ctx.rational(1, 2)]]).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("hermite_normal_form") && e.contains("integer literal"),
        "{e}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer-pivoting simplex
// ═══════════════════════════════════════════════════════════════════════════

/// Exact KKT check for `max c·x s.t. A x ≤ b, x ≥ 0`.
fn check_max_le_kkt(c: &[Q], a: &[Vec<Q>], b: &[Q], sol: &symplex::linprog::LpSolution) {
    assert_eq!(sol.status, LpStatus::Optimal);
    let x = &sol.x;
    let y = &sol.duals;
    assert!(x.iter().all(|v| !v.is_negative()), "x ≥ 0");
    for (row, rhs) in a.iter().zip(b) {
        assert!(dot(row, x) <= *rhs, "primal feasibility");
    }
    // Maximisation with ≤ rows: shadow prices are ≥ 0.
    assert!(y.iter().all(|v| !v.is_negative()), "duals ≥ 0: {y:?}");
    // Reduced costs c − Aᵀy ≤ 0, and = 0 where x_j > 0.
    for j in 0..c.len() {
        let aty: Q = a.iter().zip(y).map(|(row, yi)| &row[j] * yi).sum();
        let r = &c[j] - aty;
        assert!(!r.is_positive(), "reduced cost {r} at column {j}");
        if x[j].is_positive() {
            assert!(r.is_zero(), "complementary slackness (columns)");
        }
    }
    // y_i (a_i·x − b_i) = 0.
    for ((row, rhs), yi) in a.iter().zip(b).zip(y) {
        assert!(
            (&(dot(row, x) - rhs) * yi).is_zero(),
            "complementary slackness (rows)"
        );
    }
    // Strong duality.
    assert_eq!(dot(c, x), dot(y, b));
    assert_eq!(sol.objective.as_ref().unwrap(), &dot(c, x));
}

#[test]
fn random_fractional_lps_satisfy_kkt_exactly() {
    for seed in 1..=25u64 {
        let mut g = Lcg(seed * 7919);
        let (m, n) = (g.range(2, 7) as usize, g.range(2, 9) as usize);
        let c: Vec<Q> = (0..n).map(|_| g.rational(9, 5)).collect();
        // Non-negative rows and right-hand sides keep the problem feasible
        // (x = 0) and bounded (every column has a positive entry).
        let mut a: Vec<Vec<Q>> = (0..m)
            .map(|_| (0..n).map(|_| q(g.range(0, 9), g.range(1, 6))).collect())
            .collect();
        for j in 0..n {
            if a.iter().all(|row| row[j].is_zero()) {
                a[0][j] = qi(1);
            }
        }
        let b: Vec<Q> = (0..m).map(|_| q(g.range(1, 40), g.range(1, 4))).collect();
        let mut p = LpProblem::maximize(c.clone());
        for (row, rhs) in a.iter().zip(&b) {
            p = p.le(row.clone(), rhs.clone());
        }
        let sol = p.solve().unwrap();
        check_max_le_kkt(&c, &a, &b, &sol);
    }
}

#[test]
fn equality_rows_drive_out_artificials_on_any_sign() {
    // Rows with negative genuine entries make the artificial leave on a
    // negative pivot, flipping the sign of the tableau's common
    // denominator; results must be unaffected.
    let sol = LpProblem::minimize(vec![qi(1), qi(2), qi(3)])
        .eq(vec![qi(-1), qi(-1), qi(0)], qi(-2))
        .eq(vec![qi(0), qi(-2), qi(1)], qi(-1))
        .solve()
        .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    // x + y = 2, 2y − z = 1: minimise x + 2y + 3z → z = 0, y = 1/2, x = 3/2.
    assert_eq!(sol.x, vec![q(3, 2), q(1, 2), qi(0)]);
    assert_eq!(sol.objective, Some(q(5, 2)));
    // Duals reproduce the objective: y·b = c·x.
    assert_eq!(dot(&sol.duals, &[qi(-2), qi(-1)]), q(5, 2));
    // Redundant equality (a copy of the first, scaled by a fraction).
    let sol = LpProblem::minimize(vec![qi(1), qi(2), qi(3)])
        .eq(vec![qi(-1), qi(-1), qi(0)], qi(-2))
        .eq(vec![q(1, 3), q(1, 3), qi(0)], q(2, 3))
        .eq(vec![qi(0), qi(-2), qi(1)], qi(-1))
        .solve()
        .unwrap();
    assert_eq!(sol.x, vec![q(3, 2), q(1, 2), qi(0)]);
    assert_eq!(dot(&sol.duals, &[qi(-2), q(2, 3), qi(-1)]), q(5, 2));
}

#[test]
fn fractional_farkas_certificates_are_exact() {
    for seed in 1..=15u64 {
        let mut g = Lcg(seed * 104729);
        let (m, n) = (g.range(2, 5) as usize, g.range(2, 6) as usize);
        let a: Vec<Vec<Q>> = (0..m)
            .map(|_| (0..n).map(|_| g.rational(6, 4)).collect())
            .collect();
        let b: Vec<Q> = (0..m).map(|_| g.rational(9, 3)).collect();
        match feasible_nonneg_certified(&a, &b).unwrap() {
            Feasibility::Feasible(x) => {
                assert!(x.iter().all(|v| !v.is_negative()));
                for (row, rhs) in a.iter().zip(&b) {
                    assert_eq!(dot(row, &x), *rhs);
                }
            }
            Feasibility::Infeasible { farkas: Some(y) } => {
                // Aᵀy ≥ 0 and yᵀb < 0.
                for j in 0..n {
                    let g: Q = a.iter().zip(&y).map(|(row, yi)| &row[j] * yi).sum();
                    assert!(!g.is_negative(), "Aᵀy component {j} = {g}");
                }
                assert!(dot(&y, &b).is_negative(), "yᵀb = {}", dot(&y, &b));
            }
            Feasibility::Infeasible { farkas: None } => {
                panic!("equality system without certificate")
            }
        }
    }
}

#[test]
fn cone_membership_with_fractional_generators() {
    let cone = [
        vec![q(2, 3), qi(0), q(1, 2)],
        vec![qi(0), q(3, 5), q(1, 2)],
        vec![qi(1), qi(1), qi(0)],
    ];
    let target = [q(4, 3), q(3, 5), q(3, 2)];
    match nonneg_combination(&cone, &target).unwrap() {
        Feasibility::Feasible(lambda) => {
            assert!(lambda.iter().all(|v| !v.is_negative()));
            for i in 0..3 {
                let comb: Q = cone.iter().zip(&lambda).map(|(v, l)| &v[i] * l).sum();
                assert_eq!(comb, target[i]);
            }
        }
        other => panic!("expected a combination, got {other:?}"),
    }
    assert!(
        !nonneg_combination(&cone, &[qi(-1), qi(0), qi(0)])
            .unwrap()
            .is_feasible()
    );
}

#[test]
fn certificates_still_verify_after_tableau_rewrite() {
    use symplex::certificates::{BoxOutcome, prove_nonnegative_on_box};
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    // (1 − x)(1 − y) + xy + 1/2 ≥ 1/2 on [0, 1]²: a degree-2 Handelman
    // certificate with fractional weights.
    let goal = ((1 - &x) * (1 - &y) + &x * &y + ctx.rational(1, 2)).expand();
    let bounds = [
        (x.clone(), ctx.int(0), ctx.int(1)),
        (y.clone(), ctx.int(0), ctx.int(1)),
    ];
    match prove_nonnegative_on_box(&goal, &bounds, 2).unwrap() {
        BoxOutcome::Proved(c) => assert!(c.verify()),
        other => panic!("expected a certificate, got {other:?}"),
    }
    match prove_nonnegative_on_box(&(&goal - 3), &bounds, 2).unwrap() {
        BoxOutcome::Refuted { value, .. } => assert!(value.is_negative()),
        other => panic!("expected a refutation, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scale: sizes that were impractical over expressions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn forty_by_forty_rational_inverse_is_exact() {
    let ctx = Context::new();
    let z = random_z(40, 40, 4040);
    let qm = z.to_qmatrix();
    let d = qm.det().unwrap();
    assert!(
        !d.is_zero(),
        "random 40×40 integer matrix should be nonsingular"
    );
    let inv = qm.inv().unwrap();
    assert!((&qm * &inv).is_identity());
    // Through `Matrix` as well (fast path).
    let m = z.to_matrix(&ctx);
    let m_inv = m.inv().unwrap();
    assert_eq!(QMatrix::try_from(&m_inv).unwrap(), inv);
    let (_, pivots) = m.rref();
    assert_eq!(pivots.len(), 40);
}
