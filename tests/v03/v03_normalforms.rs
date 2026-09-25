//! symplex 0.3 — integer normal forms: Hermite (row and column style),
//! Smith, integer kernels, unimodularity, lattice determinants.
//!
//! Every test checks the defining *invariants* exactly (`H = U·A`,
//! `|det U| = 1`, echelon/positivity/reduction, `S = U·A·V`, divisibility
//! chain, `A·k = 0`) rather than trusting remembered answers; the few pinned
//! matrices were derived by hand or verified against SymPy 1.14.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;
use symplex::matrix::Matrix;
use symplex::normalforms::{
    column_hermite_normal_form, hermite_normal_form, hermite_normal_form_with_transform,
    integer_nullspace, is_unimodular, lattice_determinant, smith_normal_form,
    smith_normal_form_with_transforms,
};
use symplex::prelude::*;

fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

fn mi(ctx: &Context, rows: &[&[i64]]) -> Matrix {
    Matrix::from_i64(ctx, rows).unwrap()
}

fn ints(m: &Matrix) -> Vec<Vec<BigInt>> {
    m.to_bigint_rows().expect("integer matrix")
}

fn det_abs(m: &Matrix) -> BigInt {
    m.det().unwrap().as_bigint().expect("integer det").abs()
}

fn is_unimodular_det(m: &Matrix) -> bool {
    m.is_square() && det_abs(m).is_one()
}

/// Row-style HNF invariants: echelon, positive pivots, reduction above
/// pivots, zero rows last.  Returns the pivot columns.
fn assert_row_hnf_shape(h: &Matrix, label: &str) -> Vec<usize> {
    let rows = ints(h);
    let mut pivots: Vec<usize> = Vec::new();
    let mut seen_zero_row = false;
    for (i, row) in rows.iter().enumerate() {
        match row.iter().position(|v| !v.is_zero()) {
            None => seen_zero_row = true,
            Some(c) => {
                assert!(!seen_zero_row, "{label}: nonzero row {i} after a zero row");
                if let Some(&prev) = pivots.last() {
                    assert!(
                        c > prev,
                        "{label}: pivot column {c} of row {i} not right of {prev}"
                    );
                }
                assert!(
                    row[c].is_positive(),
                    "{label}: pivot ({i},{c}) = {} not positive",
                    row[c]
                );
                for (k, above) in rows.iter().enumerate().take(i) {
                    assert!(
                        !above[c].is_negative() && above[c] < row[c],
                        "{label}: entry ({k},{c}) = {} not in [0, {})",
                        above[c],
                        row[c]
                    );
                }
                pivots.push(c);
            }
        }
    }
    pivots
}

fn assert_hnf_invariants(a: &Matrix, label: &str) -> Matrix {
    let HermiteNormalForm { h, u } = hermite_normal_form_with_transform(a).unwrap();
    assert_eq!(h.shape(), a.shape(), "{label}: shape");
    assert_eq!(u.shape(), (a.nrows(), a.nrows()), "{label}: U shape");
    assert_eq!((&u * a).eval(), h, "{label}: H = U·A");
    assert!(is_unimodular_det(&u), "{label}: |det U| = {}", det_abs(&u));
    let pivots = assert_row_hnf_shape(&h, label);
    assert_eq!(pivots.len(), a.rank(), "{label}: pivot count = rank");
    assert_eq!(
        hermite_normal_form(a).unwrap(),
        h,
        "{label}: with/without transform agree"
    );
    assert_eq!(
        a.hermite_normal_form().unwrap(),
        h,
        "{label}: method delegates"
    );
    h
}

/// Column-style (Cohen/SymPy) HNF invariants.
fn assert_col_hnf_shape(h: &Matrix, label: &str) {
    let rows = ints(h);
    let (m, n) = h.shape();
    let mut seen_nonzero_col = false;
    let mut last_pivot_row: Option<usize> = None;
    for j in 0..n {
        let col: Vec<&BigInt> = (0..m).map(|i| &rows[i][j]).collect();
        match col.iter().rposition(|v| !v.is_zero()) {
            None => assert!(
                !seen_nonzero_col,
                "{label}: zero column {j} after a nonzero one"
            ),
            Some(r) => {
                seen_nonzero_col = true;
                if let Some(prev) = last_pivot_row {
                    assert!(
                        r > prev,
                        "{label}: pivot row {r} of column {j} not below {prev}"
                    );
                }
                assert!(
                    rows[r][j].is_positive(),
                    "{label}: pivot ({r},{j}) not positive"
                );
                for k in (j + 1)..n {
                    assert!(
                        !rows[r][k].is_negative() && rows[r][k] < rows[r][j],
                        "{label}: entry ({r},{k}) = {} not in [0, {})",
                        rows[r][k],
                        rows[r][j]
                    );
                }
                last_pivot_row = Some(r);
            }
        }
    }
}

fn assert_snf_invariants(a: &Matrix, label: &str) -> Matrix {
    let SmithNormalForm { s, u, v } = smith_normal_form_with_transforms(a).unwrap();
    assert_eq!(s.shape(), a.shape(), "{label}: shape");
    assert_eq!((&(&u * a) * &v).eval(), s, "{label}: S = U·A·V");
    assert!(is_unimodular_det(&u), "{label}: |det U| = {}", det_abs(&u));
    assert!(is_unimodular_det(&v), "{label}: |det V| = {}", det_abs(&v));
    let rows = ints(&s);
    let n = s.ncols();
    let mut diag: Vec<BigInt> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if i != j {
                assert!(v.is_zero(), "{label}: off-diagonal ({i},{j}) nonzero");
            }
        }
        if i < n {
            diag.push(row[i].clone());
        }
    }
    let r = diag.iter().filter(|d| !d.is_zero()).count();
    assert_eq!(r, a.rank(), "{label}: number of invariant factors = rank");
    for (k, d) in diag.iter().enumerate() {
        if k < r {
            assert!(d.is_positive(), "{label}: d{k} = {d} not positive");
            if k + 1 < r {
                assert!(
                    diag[k + 1].is_multiple_of(d),
                    "{label}: d{k} = {d} does not divide d{} = {}",
                    k + 1,
                    diag[k + 1]
                );
            }
        } else {
            assert!(d.is_zero(), "{label}: trailing d{k} = {d} not zero");
        }
    }
    assert_eq!(
        smith_normal_form(a).unwrap(),
        s,
        "{label}: with/without transforms agree"
    );
    assert_eq!(
        a.smith_normal_form().unwrap(),
        s,
        "{label}: method delegates"
    );
    s
}

fn assert_kernel_invariants(a: &Matrix, label: &str) -> Vec<Matrix> {
    let basis = integer_nullspace(a).unwrap();
    let n = a.ncols();
    assert_eq!(basis.len(), n - a.rank(), "{label}: kernel rank");
    for (k, v) in basis.iter().enumerate() {
        assert_eq!(v.shape(), (n, 1), "{label}: kernel vector {k} shape");
        assert!(
            v.to_bigint_rows().is_some(),
            "{label}: kernel vector {k} integer"
        );
        let prod = (a * v).eval();
        assert_eq!(prod.is_zero(), Some(true), "{label}: A·k{k} = {prod}");
    }
    if !basis.is_empty() {
        // The vectors are independent and span a *saturated* lattice
        // (the kernel is ℤⁿ ∩ subspace): the gcd of the maximal minors of
        // the basis matrix is 1.
        let k = Matrix::hstack(&basis.iter().collect::<Vec<_>>()).unwrap();
        assert_eq!(k.rank(), basis.len(), "{label}: kernel basis independent");
        assert_eq!(
            lattice_determinant(&k.transpose()).unwrap(),
            bi(1),
            "{label}: kernel basis saturated"
        );
    }
    assert_eq!(
        a.integer_nullspace().unwrap().len(),
        basis.len(),
        "{label}: method delegates"
    );
    basis
}

// ═══════════════════════════════════════════════════════════════════════════
// Hermite normal form — known answers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hnf_3x3_hand_derived() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let h = assert_hnf_invariants(&a, "3x3");
    assert_eq!(h, mi(&ctx, &[&[2, 4, 4], &[0, 6, 0], &[0, 0, 12]]));
    // Product of pivots = |det A| = 144.
    assert_eq!(det_abs(&a), bi(144));
}

#[test]
fn hnf_2x2_index_five() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[3, 1], &[1, 2]]);
    let h = assert_hnf_invariants(&a, "2x2");
    assert_eq!(h, mi(&ctx, &[&[1, 2], &[0, 5]]));
}

#[test]
fn hnf_identity_is_fixed() {
    let ctx = Context::new();
    let i = Matrix::identity(&ctx, 4).unwrap();
    assert_eq!(assert_hnf_invariants(&i, "identity"), i);
}

#[test]
fn hnf_unimodular_matrix_becomes_identity() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 1], &[1, 1]]);
    assert_eq!(
        assert_hnf_invariants(&a, "unimodular"),
        Matrix::identity(&ctx, 2).unwrap()
    );
}

#[test]
fn hnf_1x1_negative_becomes_positive() {
    let ctx = Context::new();
    assert_eq!(
        assert_hnf_invariants(&mi(&ctx, &[&[-5]]), "[-5]"),
        mi(&ctx, &[&[5]])
    );
    assert_eq!(
        assert_hnf_invariants(&mi(&ctx, &[&[0]]), "[0]"),
        mi(&ctx, &[&[0]])
    );
    assert_eq!(
        assert_hnf_invariants(&mi(&ctx, &[&[7]]), "[7]"),
        mi(&ctx, &[&[7]])
    );
}

#[test]
fn hnf_zero_matrix_is_zero() {
    let ctx = Context::new();
    let z = Matrix::zeros(&ctx, 2, 3).unwrap();
    assert_eq!(assert_hnf_invariants(&z, "zeros"), z);
}

#[test]
fn hnf_wide_single_row() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 6]]);
    assert_eq!(assert_hnf_invariants(&a, "wide"), a);
    let b = mi(&ctx, &[&[-2, 4, -6]]);
    assert_eq!(
        assert_hnf_invariants(&b, "wide neg"),
        mi(&ctx, &[&[2, -4, 6]])
    );
}

#[test]
fn hnf_tall_single_column() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[6], &[4], &[10]]);
    assert_eq!(
        assert_hnf_invariants(&a, "tall"),
        mi(&ctx, &[&[2], &[0], &[0]])
    );
}

#[test]
fn hnf_rank_deficient_square() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2, 3], &[2, 4, 6], &[1, 1, 1]]);
    let h = assert_hnf_invariants(&a, "rank 2");
    assert_eq!(h, mi(&ctx, &[&[1, 0, -1], &[0, 1, 2], &[0, 0, 0]]));
}

#[test]
fn hnf_rectangular_rank_deficient_wide() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 6, 8], &[1, 2, 3, 4], &[0, 0, 5, 5]]);
    let h = assert_hnf_invariants(&a, "3x4");
    assert_eq!(a.rank(), 2);
    assert_eq!(ints(&h)[2], vec![bi(0); 4]);
}

#[test]
fn hnf_rectangular_tall() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2], &[3, 4], &[5, 6], &[7, 8]]);
    let h = assert_hnf_invariants(&a, "4x2");
    // det of the 2×2 minors: gcd(-2, -4, -6, -2, -4, -2) = 2 → pivots 1, 2.
    assert_eq!(h, mi(&ctx, &[&[1, 0], &[0, 2], &[0, 0], &[0, 0]]));
}

#[test]
fn hnf_negative_entries() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[-3, -7], &[-2, -5]]);
    // det = 15 − 14 = 1 → unimodular → identity.
    assert_eq!(
        assert_hnf_invariants(&a, "negatives"),
        Matrix::identity(&ctx, 2).unwrap()
    );
    let b = mi(&ctx, &[&[-4, 2], &[6, -9]]);
    assert_hnf_invariants(&b, "negatives 2");
}

#[test]
fn hnf_large_entries_exact() {
    let ctx = Context::new();
    let big = BigInt::parse_bytes(b"12345678901234567890123", 10).unwrap();
    let a = Matrix::from_bigint(
        &ctx,
        &[vec![big.clone(), bi(2)], vec![bi(3), big.clone() + bi(1)]],
    )
    .unwrap();
    let h = assert_hnf_invariants(&a, "big");
    assert_eq!(det_abs(&h), det_abs(&a));
}

// ═══════════════════════════════════════════════════════════════════════════
// Hermite normal form — structural properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hnf_is_idempotent() {
    let ctx = Context::new();
    for a in [
        mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]),
        mi(&ctx, &[&[1, 2, 3], &[2, 4, 6], &[1, 1, 1]]),
        mi(&ctx, &[&[6, 4, 10]]),
    ] {
        let h = hermite_normal_form(&a).unwrap();
        assert_eq!(hermite_normal_form(&h).unwrap(), h);
    }
}

#[test]
fn hnf_invariant_under_unimodular_left_multiplication() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let h = hermite_normal_form(&a).unwrap();
    let vs = [
        mi(&ctx, &[&[1, 1, 0], &[0, 1, 0], &[0, 0, 1]]),
        mi(&ctx, &[&[0, 1, 0], &[1, 0, 0], &[0, 0, 1]]),
        mi(&ctx, &[&[2, 1, 3], &[1, 1, 2], &[0, 0, -1]]),
        mi(&ctx, &[&[1, -5, 2], &[0, 1, 7], &[0, 0, 1]]),
    ];
    for v in &vs {
        assert!(is_unimodular(v).unwrap());
        assert_eq!(hermite_normal_form(&(v * &a).eval()).unwrap(), h);
    }
}

#[test]
fn hnf_row_permutation_does_not_change_result() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 10]]);
    let p = a.select_rows(&[2, 0, 1]).unwrap();
    assert_eq!(
        hermite_normal_form(&a).unwrap(),
        hermite_normal_form(&p).unwrap()
    );
}

#[test]
fn hnf_pivot_product_is_abs_det_for_square_nonsingular() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[4, 7, 2], &[3, -1, 5], &[8, 0, 6]]);
    let h = hermite_normal_form(&a).unwrap();
    let prod: BigInt = ints(&h)
        .iter()
        .enumerate()
        .map(|(i, r)| r[i].clone())
        .product();
    assert_eq!(prod, det_abs(&a));
}

#[test]
fn hnf_rows_span_same_lattice_as_input() {
    // The HNF rows generate the row lattice of A: A = W·H for an integer
    // matrix W (here W = U⁻¹, also integer because U is unimodular).
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let HermiteNormalForm { h, u } = hermite_normal_form_with_transform(&a).unwrap();
    let u_inv = u.inv().unwrap().eval();
    assert!(u_inv.to_bigint_rows().is_some());
    assert_eq!((&u_inv * &h).eval(), a);
}

#[test]
fn hnf_transform_not_required_to_be_identity_when_rank_deficient() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 1], &[1, 1], &[1, 1]]);
    let HermiteNormalForm { h, u } = hermite_normal_form_with_transform(&a).unwrap();
    assert_eq!(h, mi(&ctx, &[&[1, 1], &[0, 0], &[0, 0]]));
    assert_eq!((&u * &a).eval(), h);
    assert!(is_unimodular_det(&u));
}

// ═══════════════════════════════════════════════════════════════════════════
// Column Hermite normal form (SymPy convention)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn column_hnf_matches_sympy_documented_example() {
    // SymPy 1.14: hermite_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]))
    //   == Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])
    let ctx = Context::new();
    let a = mi(&ctx, &[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
    let h = column_hermite_normal_form(&a).unwrap();
    assert_eq!(h, mi(&ctx, &[&[10, 0, 2], &[0, 15, 3], &[0, 0, 2]]));
    assert_col_hnf_shape(&h, "sympy");
    // Independently: H = A·V with V = [[1,-1,0],[-1,8,1],[1,-9,-1]], det V = 1.
    let v = mi(&ctx, &[&[1, -1, 0], &[-1, 8, 1], &[1, -9, -1]]);
    assert_eq!((&a * &v).eval(), h);
    assert!(is_unimodular(&v).unwrap());
}

#[test]
fn column_hnf_preserves_column_lattice() {
    // Same column lattice ⟺ same row-HNF of the transposes.
    let ctx = Context::new();
    for a in [
        mi(&ctx, &[&[2, 3, 6, 2], &[5, 6, 1, 6], &[8, 3, 1, 1]]),
        mi(&ctx, &[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]),
        mi(&ctx, &[&[2, 4], &[6, 8], &[1, 1]]),
    ] {
        let h = column_hermite_normal_form(&a).unwrap();
        assert_eq!(h.shape(), a.shape());
        assert_col_hnf_shape(&h, "col");
        assert_eq!(
            hermite_normal_form(&h.transpose()).unwrap(),
            hermite_normal_form(&a.transpose()).unwrap()
        );
    }
}

#[test]
fn column_hnf_of_wide_full_rank_matrix_has_leading_zero_columns() {
    // The 3×4 matrix has coprime maximal minors, so its column lattice is
    // all of ℤ³ and the nonzero block is the identity.
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 3, 6, 2], &[5, 6, 1, 6], &[8, 3, 1, 1]]);
    assert_eq!(lattice_determinant(&a).unwrap(), bi(1));
    let h = column_hermite_normal_form(&a).unwrap();
    assert_eq!(h, mi(&ctx, &[&[0, 1, 0, 0], &[0, 0, 1, 0], &[0, 0, 0, 1]]));
}

#[test]
fn column_hnf_square_nonsingular_is_upper_triangular() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let h = column_hermite_normal_form(&a).unwrap();
    let rows = ints(&h);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate().take(i) {
            assert!(v.is_zero(), "({i},{j}) below diagonal nonzero");
        }
        assert!(row[i].is_positive());
    }
    let prod: BigInt = rows.iter().enumerate().map(|(i, r)| r[i].clone()).product();
    assert_eq!(prod, bi(144));
}

#[test]
fn column_hnf_is_idempotent() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
    let h = column_hermite_normal_form(&a).unwrap();
    assert_eq!(column_hermite_normal_form(&h).unwrap(), h);
}

// ═══════════════════════════════════════════════════════════════════════════
// Smith normal form
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn snf_matches_sympy_documented_example() {
    // SymPy 1.14: smith_normal_form(Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]]), domain=ZZ)
    //   == Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])
    let ctx = Context::new();
    let a = mi(&ctx, &[&[12, 6, 4], &[3, 9, 6], &[2, 16, 14]]);
    let s = assert_snf_invariants(&a, "sympy");
    assert_eq!(s, mi(&ctx, &[&[1, 0, 0], &[0, 10, 0], &[0, 0, 30]]));
}

#[test]
fn snf_3x3_hand_derived() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let s = assert_snf_invariants(&a, "3x3");
    assert_eq!(s, mi(&ctx, &[&[2, 0, 0], &[0, 6, 0], &[0, 0, 12]]));
}

#[test]
fn snf_diagonal_needs_gcd_fix() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 0], &[0, 3]]);
    assert_eq!(
        assert_snf_invariants(&a, "diag(2,3)"),
        mi(&ctx, &[&[1, 0], &[0, 6]])
    );
    let b = mi(&ctx, &[&[4, 0, 0], &[0, 6, 0], &[0, 0, 10]]);
    // gcd(4,6,10) = 2; gcd of 2×2 minors = gcd(24, 40, 60) = 4 → d₂ = 2;
    // product = 240 → d₃ = 60.
    assert_eq!(
        assert_snf_invariants(&b, "diag(4,6,10)"),
        mi(&ctx, &[&[2, 0, 0], &[0, 2, 0], &[0, 0, 60]])
    );
}

#[test]
fn snf_rank_deficient() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2, 3], &[2, 4, 6], &[1, 1, 1]]);
    let s = assert_snf_invariants(&a, "rank 2");
    assert_eq!(s, mi(&ctx, &[&[1, 0, 0], &[0, 1, 0], &[0, 0, 0]]));
}

#[test]
fn snf_rectangular() {
    let ctx = Context::new();
    let wide = mi(&ctx, &[&[2, 4, 6, 8], &[3, 6, 9, 15]]);
    // gcd of entries = 1; 2×2 minors: 0,0,6,0,6,6 → gcd 6.
    assert_eq!(
        assert_snf_invariants(&wide, "wide"),
        mi(&ctx, &[&[1, 0, 0, 0], &[0, 6, 0, 0]])
    );
    let tall = wide.transpose();
    assert_eq!(
        assert_snf_invariants(&tall, "tall"),
        mi(&ctx, &[&[1, 0], &[0, 6], &[0, 0], &[0, 0]])
    );
}

#[test]
fn snf_1x1_and_zero() {
    let ctx = Context::new();
    assert_eq!(
        assert_snf_invariants(&mi(&ctx, &[&[-6]]), "[-6]"),
        mi(&ctx, &[&[6]])
    );
    assert_eq!(
        assert_snf_invariants(&mi(&ctx, &[&[0]]), "[0]"),
        mi(&ctx, &[&[0]])
    );
    let z = Matrix::zeros(&ctx, 3, 2).unwrap();
    assert_eq!(assert_snf_invariants(&z, "zeros"), z);
}

#[test]
fn snf_negative_entries() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[-4, 2, -6], &[3, -9, 6], &[-2, 4, -8]]);
    assert_snf_invariants(&a, "negatives");
}

#[test]
fn snf_product_of_invariant_factors_is_abs_det() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[4, 7, 2], &[3, -1, 5], &[8, 0, 6]]);
    let s = smith_normal_form(&a).unwrap();
    let prod: BigInt = ints(&s)
        .iter()
        .enumerate()
        .map(|(i, r)| r[i].clone())
        .product();
    assert_eq!(prod, det_abs(&a));
}

#[test]
fn snf_first_factor_is_gcd_of_entries() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[6, 10, 15], &[20, 30, 12], &[9, 21, 33]]);
    let s = smith_normal_form(&a).unwrap();
    let g = symplex::ntheory::gcd_many(&ints(&a).concat());
    assert_eq!(ints(&s)[0][0], g);
}

#[test]
fn snf_of_hnf_equals_snf_of_original() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    let h = hermite_normal_form(&a).unwrap();
    assert_eq!(
        smith_normal_form(&h).unwrap(),
        smith_normal_form(&a).unwrap()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Integer nullspace
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn integer_nullspace_single_row() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 6]]);
    let basis = assert_kernel_invariants(&a, "row");
    assert_eq!(basis.len(), 2);
}

#[test]
fn integer_nullspace_is_saturated_where_rational_basis_is_not() {
    // ker([2 1 1]) has ℤ-basis (1,−2,0), (0,1,−1).  Scaling the rational
    // RREF basis gives (−1,2,0), (−1,0,2), which spans only an index-2
    // sublattice.
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 1, 1]]);
    let basis = assert_kernel_invariants(&a, "saturation");
    assert_eq!(basis.len(), 2);
    let scaled = mi(&ctx, &[&[-1, -1], &[2, 0], &[0, 2]]);
    assert_eq!(lattice_determinant(&scaled.transpose()).unwrap(), bi(2));
}

#[test]
fn integer_nullspace_full_column_rank_is_empty() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 0], &[0, 1], &[1, 1]]);
    assert!(assert_kernel_invariants(&a, "full rank").is_empty());
    assert!(assert_kernel_invariants(&Matrix::identity(&ctx, 3).unwrap(), "identity").is_empty());
}

#[test]
fn integer_nullspace_zero_matrix_is_standard_basis() {
    let ctx = Context::new();
    let z = Matrix::zeros(&ctx, 2, 3).unwrap();
    let basis = assert_kernel_invariants(&z, "zeros");
    assert_eq!(basis.len(), 3);
    let k = Matrix::hstack(&basis.iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(
        hermite_normal_form(&k.transpose()).unwrap(),
        Matrix::identity(&ctx, 3).unwrap()
    );
}

#[test]
fn integer_nullspace_rank_deficient_square() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
    let basis = assert_kernel_invariants(&a, "rank 2");
    assert_eq!(basis.len(), 1);
    // ker = span{(1, −2, 1)} up to sign.
    let v = ints(&basis[0]).concat();
    assert!(
        v == vec![bi(1), bi(-2), bi(1)] || v == vec![bi(-1), bi(2), bi(-1)],
        "{v:?}"
    );
}

#[test]
fn integer_nullspace_wide_rank_deficient() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 6, 8], &[1, 2, 3, 4], &[0, 0, 5, 5]]);
    let basis = assert_kernel_invariants(&a, "3x4 rank 2");
    assert_eq!(basis.len(), 2);
}

#[test]
fn integer_nullspace_spans_rational_nullspace() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[3, 6, 9, 3], &[1, 2, 4, 2]]);
    let basis = assert_kernel_invariants(&a, "span");
    let rational = a.nullspace();
    assert_eq!(basis.len(), rational.len());
}

// ═══════════════════════════════════════════════════════════════════════════
// is_unimodular / lattice_determinant
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_unimodular_cases() {
    let ctx = Context::new();
    assert!(is_unimodular(&Matrix::identity(&ctx, 3).unwrap()).unwrap());
    assert!(is_unimodular(&mi(&ctx, &[&[2, 1], &[1, 1]]).transpose()).unwrap());
    assert!(is_unimodular(&mi(&ctx, &[&[-1]])).unwrap());
    assert!(is_unimodular(&mi(&ctx, &[&[1, 5, -3], &[0, 1, 4], &[0, 0, -1]])).unwrap());
    assert!(!is_unimodular(&mi(&ctx, &[&[2, 0], &[0, 1]])).unwrap());
    assert!(!is_unimodular(&mi(&ctx, &[&[1, 2], &[2, 4]])).unwrap());
    assert!(!is_unimodular(&mi(&ctx, &[&[1, 2, 3]])).unwrap());
    assert!(!is_unimodular(&mi(&ctx, &[&[0]])).unwrap());
}

#[test]
fn is_unimodular_agrees_with_det() {
    let ctx = Context::new();
    for a in [
        mi(&ctx, &[&[3, 2], &[4, 3]]),
        mi(&ctx, &[&[3, 2], &[4, 4]]),
        mi(&ctx, &[&[1, 2, 3], &[0, 1, 4], &[5, 6, 0]]),
        mi(&ctx, &[&[2, 4], &[1, 3]]),
    ] {
        assert_eq!(is_unimodular(&a).unwrap(), det_abs(&a).is_one(), "{a}");
    }
}

#[test]
fn lattice_determinant_square_is_abs_det() {
    let ctx = Context::new();
    let a = mi(&ctx, &[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]);
    assert_eq!(lattice_determinant(&a).unwrap(), bi(144));
    assert_eq!(lattice_determinant(&mi(&ctx, &[&[-7]])).unwrap(), bi(7));
    assert_eq!(
        lattice_determinant(&mi(&ctx, &[&[2, 0], &[0, 3]])).unwrap(),
        bi(6)
    );
}

#[test]
fn lattice_determinant_wide_is_gcd_of_maximal_minors() {
    let ctx = Context::new();
    // Columns (2,0), (0,3), (1,1): minors 6, 2, −3 → gcd 1.
    assert_eq!(
        lattice_determinant(&mi(&ctx, &[&[2, 0, 1], &[0, 3, 1]])).unwrap(),
        bi(1)
    );
    // Columns (2,0), (0,4), (2,2): minors 8, 4, −8 → gcd 4.
    assert_eq!(
        lattice_determinant(&mi(&ctx, &[&[2, 0, 2], &[0, 4, 2]])).unwrap(),
        bi(4)
    );
    // Single row: gcd of entries.
    assert_eq!(
        lattice_determinant(&mi(&ctx, &[&[6, 10, 15]])).unwrap(),
        bi(1)
    );
    assert_eq!(
        lattice_determinant(&mi(&ctx, &[&[6, 10, 14]])).unwrap(),
        bi(2)
    );
}

#[test]
fn lattice_determinant_rank_deficient_is_err() {
    let ctx = Context::new();
    let err = lattice_determinant(&mi(&ctx, &[&[1, 2], &[2, 4]])).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    assert!(lattice_determinant(&mi(&ctx, &[&[1, 2], &[3, 4], &[5, 6]])).is_err());
    assert!(lattice_determinant(&mi(&ctx, &[&[0]])).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Input validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn non_integer_literal_entries_are_rejected() {
    let ctx = Context::new();
    let half = Matrix::new(vec![vec![ctx.rational(1, 2), ctx.int(1)]]).unwrap();
    for r in [
        hermite_normal_form(&half).err(),
        column_hermite_normal_form(&half).err(),
        smith_normal_form(&half).err(),
        hermite_normal_form_with_transform(&half).err(),
        smith_normal_form_with_transforms(&half).err(),
        integer_nullspace(&half).err(),
        is_unimodular(&half).err(),
        lattice_determinant(&half).err(),
    ] {
        let err = r.expect("expected an error");
        assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    }
}

#[test]
fn symbolic_entries_are_rejected() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x, ctx.int(1)]]).unwrap();
    assert!(hermite_normal_form(&m).is_err());
    assert!(smith_normal_form(&m).is_err());
    assert!(integer_nullspace(&m).is_err());
    assert!(m.hermite_normal_form().is_err());
    assert!(m.smith_normal_form().is_err());
    assert!(m.integer_nullspace().is_err());
}

#[test]
fn constant_integer_expressions_are_folded() {
    let ctx = Context::new();
    // 2 + 3 and 6/2 are not literals until evaluated; the module folds them.
    let e1 = &ctx.int(2) + &ctx.int(3);
    let e2 = &ctx.int(6) / &ctx.int(2);
    let m = Matrix::new(vec![vec![e1, e2]]).unwrap();
    assert_eq!(
        hermite_normal_form(&m).unwrap(),
        mi(&ctx, &[&[5, 3]]).hermite_normal_form().unwrap()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Property tests
// ═══════════════════════════════════════════════════════════════════════════

fn matrix_from_flat(ctx: &Context, flat: &[i64], rows: usize, cols: usize) -> Matrix {
    let data: Vec<Vec<Ex>> = (0..rows)
        .map(|i| (0..cols).map(|j| ctx.int(flat[i * cols + j])).collect())
        .collect();
    Matrix::new(data).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// 6×6 random integer matrix: HNF invariants (H = U·A, U unimodular,
    /// echelon/positive/reduced, rank).
    #[test]
    fn prop_hnf_6x6(flat in prop::collection::vec(-9i64..=9, 36)) {
        let ctx = Context::new();
        let a = matrix_from_flat(&ctx, &flat, 6, 6);
        let HermiteNormalForm { h, u } = hermite_normal_form_with_transform(&a).unwrap();
        prop_assert_eq!((&u * &a).eval(), h.clone());
        prop_assert!(det_abs(&u).is_one());
        let pivots = assert_row_hnf_shape(&h, "prop 6x6");
        prop_assert_eq!(pivots.len(), a.rank());
        prop_assert_eq!(hermite_normal_form(&h).unwrap(), h);
    }

    /// 6×6 random integer matrix: SNF invariants (S = U·A·V, unimodular
    /// transforms, diagonal, divisibility chain, rank).
    #[test]
    fn prop_snf_6x6(flat in prop::collection::vec(-9i64..=9, 36)) {
        let ctx = Context::new();
        let a = matrix_from_flat(&ctx, &flat, 6, 6);
        let s = assert_snf_invariants(&a, "prop snf 6x6");
        let rows = ints(&s);
        // d₁ is the gcd of all entries.
        let g = symplex::ntheory::gcd_many(&ints(&a).concat());
        prop_assert_eq!(rows[0][0].clone(), g);
    }

    /// Rectangular random matrix: the integer kernel is a saturated
    /// ℤ-basis of the right size and every vector is annihilated.
    #[test]
    fn prop_integer_nullspace(flat in prop::collection::vec(-5i64..=5, 15)) {
        let ctx = Context::new();
        let a = matrix_from_flat(&ctx, &flat, 3, 5);
        let basis = assert_kernel_invariants(&a, "prop kernel");
        prop_assert_eq!(basis.len(), 5 - a.rank());
    }

    /// Row HNF is invariant under a random unimodular left factor.
    #[test]
    fn prop_hnf_unimodular_invariance(
        flat in prop::collection::vec(-6i64..=6, 9),
        p in -4i64..=4, q in -4i64..=4, r in -4i64..=4,
    ) {
        let ctx = Context::new();
        let a = matrix_from_flat(&ctx, &flat, 3, 3);
        // Unit upper triangular ⇒ det 1.
        let v = mi(&ctx, &[&[1, p, q], &[0, 1, r], &[0, 0, 1]]);
        prop_assert_eq!(
            hermite_normal_form(&(&v * &a).eval()).unwrap(),
            hermite_normal_form(&a).unwrap()
        );
    }
}
