//! symplex 0.3 — `Matrix` ergonomics: selection/deletion, three-valued
//! structure queries, exact numeric conversion, `subs_map`, `nnz`, and
//! the multi-argument gcd/lcm helpers in `ntheory`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use symplex::matrix::Matrix;
use symplex::ntheory::{gcd_many, igcd, ilcm, lcm_many, rational_lcm_of_denominators};
use symplex::prelude::*;

fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

fn q(n: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(bi(n), bi(d))
}

// ═══════════════════════════════════════════════════════════════════════════
// extract / select_rows / select_cols
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_reorders_and_repeats() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let e = m.extract(&[2, 0, 2], &[1, 1]).unwrap();
    assert_eq!(e, matrix![ctx, [8, 8], [2, 2], [8, 8]]);
}

#[test]
fn extract_identity_selection_is_clone() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    assert_eq!(m.extract(&[0, 1], &[0, 1]).unwrap(), m);
}

#[test]
fn extract_row_out_of_range_is_err() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let err = m.extract(&[2], &[0]).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }), "{err}");
    assert!(err.to_string().contains("row index 2"), "{err}");
}

#[test]
fn extract_col_out_of_range_is_err() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let err = m.extract(&[0], &[5]).unwrap_err();
    assert!(err.to_string().contains("column index 5"), "{err}");
}

#[test]
fn extract_empty_selection_is_err() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    assert!(m.extract(&[], &[0]).is_err());
    assert!(m.extract(&[0], &[]).is_err());
}

#[test]
fn select_rows_basic_and_errors() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4], [5, 6]];
    assert_eq!(m.select_rows(&[1]).unwrap(), matrix![ctx, [3, 4]]);
    assert_eq!(
        m.select_rows(&[2, 1, 0]).unwrap(),
        matrix![ctx, [5, 6], [3, 4], [1, 2]]
    );
    assert!(m.select_rows(&[]).is_err());
    assert!(m.select_rows(&[3]).is_err());
}

#[test]
fn select_cols_basic_and_errors() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    assert_eq!(
        m.select_cols(&[0, 2]).unwrap(),
        matrix![ctx, [1, 3], [4, 6]]
    );
    assert_eq!(
        m.select_cols(&[1, 1]).unwrap(),
        matrix![ctx, [2, 2], [5, 5]]
    );
    assert!(m.select_cols(&[]).is_err());
    assert!(m.select_cols(&[3]).is_err());
}

#[test]
fn select_matches_submatrix_for_ranges() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    assert_eq!(
        m.extract(&[1, 2], &[0, 1]).unwrap(),
        m.submatrix(1..3, 0..2)
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// delete_row / delete_col
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn delete_row_middle_first_last() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4], [5, 6]];
    assert_eq!(m.delete_row(1).unwrap(), matrix![ctx, [1, 2], [5, 6]]);
    assert_eq!(m.delete_row(0).unwrap(), matrix![ctx, [3, 4], [5, 6]]);
    assert_eq!(m.delete_row(2).unwrap(), matrix![ctx, [1, 2], [3, 4]]);
}

#[test]
fn delete_col_middle_first_last() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6]];
    assert_eq!(m.delete_col(1).unwrap(), matrix![ctx, [1, 3], [4, 6]]);
    assert_eq!(m.delete_col(0).unwrap(), matrix![ctx, [2, 3], [5, 6]]);
    assert_eq!(m.delete_col(2).unwrap(), matrix![ctx, [1, 2], [4, 5]]);
}

#[test]
fn delete_out_of_range_is_err() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    assert!(m.delete_row(2).is_err());
    assert!(m.delete_col(2).is_err());
}

#[test]
fn delete_last_remaining_row_or_col_is_err() {
    let ctx = Context::new();
    let row = matrix![ctx, [1, 2]];
    let col = matrix![ctx, [1], [2]];
    assert!(row.delete_row(0).is_err());
    assert!(col.delete_col(0).is_err());
    // …but the other axis still works.
    assert_eq!(row.delete_col(0).unwrap(), matrix![ctx, [2]]);
    assert_eq!(col.delete_row(0).unwrap(), matrix![ctx, [2]]);
}

#[test]
fn delete_row_then_col_equals_minor_matrix() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let d = m.delete_row(1).unwrap().delete_col(2).unwrap();
    assert_eq!(d, m.minor_matrix(1, 2).unwrap());
}

// ═══════════════════════════════════════════════════════════════════════════
// is_zero / is_integer_matrix — three-valued
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_zero_literal_zero() {
    let ctx = Context::new();
    assert_eq!(Matrix::zeros(&ctx, 2, 3).is_zero(), Some(true));
}

#[test]
fn is_zero_literal_nonzero() {
    let ctx = Context::new();
    assert_eq!(matrix![ctx, [0, 0], [0, 1]].is_zero(), Some(false));
}

#[test]
fn is_zero_symbolic_unknown() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![ctx.int(0), x]]).unwrap();
    assert_eq!(m.is_zero(), None);
}

#[test]
fn is_zero_symbolic_identity_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin²x + cos²x − 1 simplifies to zero.
    let e = &(&x.sin().powi(2) + &x.cos().powi(2)) - 1;
    let m = Matrix::new(vec![vec![e, ctx.int(0)]]).unwrap();
    assert_eq!(m.is_zero(), Some(true));
}

#[test]
fn is_zero_positive_symbol_is_false() {
    let ctx = Context::new();
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let m = Matrix::new(vec![vec![p]]).unwrap();
    assert_eq!(m.is_zero(), Some(false));
}

#[test]
fn is_zero_agrees_with_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for m in [
        matrix![ctx, [0, 0]],
        matrix![ctx, [0, 2]],
        Matrix::new(vec![vec![x]]).unwrap(),
    ] {
        assert_eq!(m.is_zero(), m.is_zero());
    }
}

#[test]
fn is_integer_matrix_three_valued() {
    let ctx = Context::new();
    assert_eq!(
        matrix![ctx, [1, -5], [0, 7]].is_integer_matrix(),
        Some(true)
    );
    let half = Matrix::new(vec![vec![ctx.int(1), ctx.rational(1, 2)]]).unwrap();
    assert_eq!(half.is_integer_matrix(), Some(false));
    let sym = Matrix::new(vec![vec![ctx.int(1), ctx.symbol("n")]]).unwrap();
    assert_eq!(sym.is_integer_matrix(), None);
}

#[test]
fn is_integer_matrix_fraction_beats_symbol() {
    // A definite "no" from a fraction is reported even if a symbol is present.
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.symbol("n"), ctx.rational(1, 3)]]).unwrap();
    assert_eq!(m.is_integer_matrix(), Some(false));
}

#[test]
fn is_integer_matrix_after_eval() {
    let ctx = Context::new();
    let e = &ctx.rational(1, 2) + &ctx.rational(1, 2);
    let m = Matrix::new(vec![vec![e]]).unwrap();
    assert_eq!(m.eval().is_integer_matrix(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// to_rational_rows / to_bigint_rows / from_ratio / from_bigint
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn to_rational_rows_round_trip() {
    let ctx = Context::new();
    let rows = vec![vec![q(1, 2), q(-3, 4)], vec![q(5, 1), q(0, 1)]];
    let m = Matrix::from_ratio(&ctx, &rows).unwrap();
    assert_eq!(m.shape(), (2, 2));
    assert_eq!(m.to_rational_rows().unwrap(), rows);
}

#[test]
fn to_rational_rows_none_for_symbolic() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(1), ctx.pi()]]).unwrap();
    assert!(m.to_rational_rows().is_none());
}

#[test]
fn to_rational_rows_after_eval() {
    let ctx = Context::new();
    let e = &ctx.rational(1, 2) + &ctx.rational(1, 3);
    let m = Matrix::new(vec![vec![e]]).unwrap();
    assert_eq!(m.eval().to_rational_rows().unwrap(), vec![vec![q(5, 6)]]);
}

#[test]
fn to_bigint_rows_round_trip() {
    let ctx = Context::new();
    let rows = vec![vec![bi(1), bi(-2)], vec![bi(3), bi(4)]];
    let m = Matrix::from_bigint(&ctx, &rows).unwrap();
    assert_eq!(m, matrix![ctx, [1, -2], [3, 4]]);
    assert_eq!(m.to_bigint_rows().unwrap(), rows);
}

#[test]
fn to_bigint_rows_none_for_fraction() {
    let ctx = Context::new();
    let m = Matrix::new(vec![vec![ctx.int(1), ctx.rational(1, 2)]]).unwrap();
    assert!(m.to_bigint_rows().is_none());
}

#[test]
fn to_bigint_rows_handles_huge_values() {
    let ctx = Context::new();
    let big = BigInt::parse_bytes(b"123456789012345678901234567890", 10).unwrap();
    let m = Matrix::from_bigint(&ctx, &[vec![big.clone()]]).unwrap();
    assert_eq!(m.to_bigint_rows().unwrap(), vec![vec![big]]);
}

#[test]
fn from_ratio_from_bigint_reject_empty_and_jagged() {
    let ctx = Context::new();
    assert!(Matrix::from_ratio(&ctx, &[]).is_err());
    assert!(Matrix::from_ratio(&ctx, &[vec![q(1, 1)], vec![]]).is_err());
    assert!(Matrix::from_bigint(&ctx, &[]).is_err());
    assert!(Matrix::from_bigint(&ctx, &[vec![bi(1), bi(2)], vec![bi(3)]]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// from_f64_rows — exact dyadic conversion
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn from_f64_rows_half_is_one_half() {
    let ctx = Context::new();
    let m = Matrix::from_f64_rows(&ctx, &[vec![0.5, 0.25, -1.5]]).unwrap();
    assert_eq!(m.get(0, 0), &ctx.rational(1, 2));
    assert_eq!(m.get(0, 1), &ctx.rational(1, 4));
    assert_eq!(m.get(0, 2), &ctx.rational(-3, 2));
}

#[test]
fn from_f64_rows_point_one_is_exact_dyadic_not_one_tenth() {
    let ctx = Context::new();
    let m = Matrix::from_f64_rows(&ctx, &[vec![0.1]]).unwrap();
    let r = m.get(0, 0).as_rational().unwrap();
    assert_ne!(r, q(1, 10));
    assert_eq!(*r.denom(), bi(1) << 55);
    assert_eq!(*r.numer(), bi(3602879701896397));
    // Bit-exact round trip.
    assert_eq!(m.get(0, 0).eval_f64().unwrap(), 0.1);
}

#[test]
fn from_f64_rows_integers_and_nan() {
    let ctx = Context::new();
    let m = Matrix::from_f64_rows(&ctx, &[vec![3.0, -0.0], vec![1e6, 2.0]]).unwrap();
    assert_eq!(m, matrix![ctx, [3, 0], [1000000, 2]]);
    let err = Matrix::from_f64_rows(&ctx, &[vec![1.0, f64::NAN]]).unwrap_err();
    assert!(matches!(err, SymplexError::InvalidArgument { .. }));
    assert!(Matrix::from_f64_rows(&ctx, &[]).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// nnz / subs_map
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nnz_counts_structural_nonzeros() {
    let ctx = Context::new();
    assert_eq!(matrix![ctx, [1, 0, 2], [0, 0, 3]].nnz(), 3);
    assert_eq!(Matrix::identity(&ctx, 5).nnz(), 5);
    assert_eq!(Matrix::zeros(&ctx, 3, 3).nnz(), 0);
}

#[test]
fn nnz_symbolic_counts_as_nonzero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![&x - &x, x.clone(), ctx.int(0)]]).unwrap();
    // x − x canonicalises to 0 at construction; x stays.
    assert_eq!(m.nnz(), 1);
}

#[test]
fn subs_map_simultaneous_swap() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let m = Matrix::new(vec![vec![x.clone(), y.clone()], vec![&x + &y, &x * &y]]).unwrap();
    let s = m.subs_map(&[(&x, &y), (&y, &x)]);
    let expected = Matrix::new(vec![vec![y.clone(), x.clone()], vec![&y + &x, &y * &x]]).unwrap();
    assert_eq!(s, expected);
}

#[test]
fn subs_map_matches_single_subs_when_independent() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let m = Matrix::new(vec![vec![&x.powi(2) + &y, &x * &y]]).unwrap();
    let a = m.subs_map(&[(&x, &ctx.int(2)), (&y, &ctx.int(3))]).eval();
    let b = m.subs(&x, &ctx.int(2)).subs(&y, &ctx.int(3)).eval();
    assert_eq!(a, b);
    assert_eq!(a, matrix![ctx, [7, 6]]);
}

#[test]
fn subs_map_empty_is_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let m = Matrix::new(vec![vec![x]]).unwrap();
    assert_eq!(m.subs_map(&[]), m);
}

// ═══════════════════════════════════════════════════════════════════════════
// ntheory: gcd_many / lcm_many / igcd / ilcm / rational_lcm_of_denominators
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_many_known() {
    assert_eq!(gcd_many(&[bi(12), bi(18), bi(30)]), bi(6));
    assert_eq!(igcd(&[12i64, 18, 30]), bi(6));
}

#[test]
fn gcd_many_negatives_and_zero() {
    assert_eq!(gcd_many(&[bi(-12), bi(-18)]), bi(6));
    assert_eq!(gcd_many(&[bi(0), bi(-9)]), bi(9));
    assert_eq!(gcd_many(&[bi(0)]), bi(0));
}

#[test]
fn gcd_many_empty_is_zero() {
    assert_eq!(gcd_many(&[]), bi(0));
    assert!(igcd::<i64>(&[]).is_zero());
}

#[test]
fn lcm_many_known() {
    assert_eq!(lcm_many(&[bi(4), bi(6), bi(10)]), bi(60));
    assert_eq!(ilcm(&[4i64, 6, 10]), bi(60));
    assert_eq!(ilcm(&[-4i64, 6]), bi(12));
}

#[test]
fn lcm_many_empty_is_one_and_zero_absorbs() {
    assert!(lcm_many(&[]).is_one());
    assert!(ilcm::<i64>(&[]).is_one());
    assert!(lcm_many(&[bi(5), bi(0), bi(7)]).is_zero());
}

#[test]
fn gcd_lcm_product_identity_for_pairs() {
    for (a, b) in [(12i64, 18), (7, 5), (-4, 6), (9, 9)] {
        let g = igcd(&[a, b]);
        let l = ilcm(&[a, b]);
        assert_eq!(g * l, bi((a * b).abs()), "{a}, {b}");
    }
}

#[test]
fn rational_lcm_of_denominators_clears_denominators() {
    let v = [q(1, 3), q(1, 7), q(2, 21), q(3, 1)];
    let l = rational_lcm_of_denominators(&v);
    assert_eq!(l, bi(21));
    for r in &v {
        assert!((r * Ratio::from_integer(l.clone())).is_integer());
    }
}

#[test]
fn rational_lcm_of_denominators_empty_and_integers() {
    assert!(rational_lcm_of_denominators(&[]).is_one());
    assert!(rational_lcm_of_denominators(&[q(2, 1), q(-5, 1)]).is_one());
}
