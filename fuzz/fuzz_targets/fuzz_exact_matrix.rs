//! Exact linear algebra: identities that hold for every rational matrix.
//!
//! `QMatrix` runs the fraction-free kernel (`exact_kernel`: Bareiss,
//! Berkowitz, resumable elimination over i64 → i128 → 256-bit → BigInt
//! cells), and `Matrix` lowers all-rational input to it.  Everything below
//! is exact, so any failure is a real bug (no tolerances):
//!
//! * `det(A·B) = det(A)·det(B)`;
//! * `A·A⁻¹ = I` whenever `det(A) ≠ 0`, and `inv` errors iff `det(A) = 0`;
//! * `rank(A) + dim null(A) = n`, every null vector satisfies `A·v = 0`;
//! * `rref` is idempotent and its pivots agree with `rank`;
//! * Cayley–Hamilton: `p(A) = 0` for `p = char_poly_coeffs(A)`, and
//!   `p(0) = det(A)·(−1)ⁿ`… checked through the constant term;
//! * `P·A = L·U` when `lu` succeeds;
//! * the symbolic `Matrix` tier agrees with `QMatrix` on `det`.
//!
//! Input bytes: `n` (1..=5), a scale selector, then `2·n²` entries.  Entries
//! mix small integers, fractions and a few huge values so the kernel's cell
//! widths are all exercised.
#![no_main]

use libfuzzer_sys::fuzz_target;
use symplex::matrix::QMatrix;
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::num_traits::{One, Zero};
use symplex::prelude::*;

type Q = Ratio<BigInt>;

fn entry(b: &[u8], i: usize, scale: u8) -> Q {
    let lo = *b.get(2 * i).unwrap_or(&0);
    let hi = *b.get(2 * i + 1).unwrap_or(&0);
    let num = (lo as i64 % 23) - 11;
    let den = (hi as i64 % 5) + 1;
    let base = Ratio::new(BigInt::from(num), BigInt::from(den));
    match scale % 4 {
        // mostly small, occasionally 10^25 or 10^-25 (forces the wide cells)
        3 if hi.is_multiple_of(7) => base * Ratio::from_integer(BigInt::from(10).pow(25)),
        2 if hi % 7 == 1 => base / Ratio::from_integer(BigInt::from(10).pow(25)),
        _ => base,
    }
}

fn matrix(b: &[u8], n: usize, offset: usize, scale: u8) -> QMatrix {
    let rows = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| entry(b, offset + i * n + j, scale))
                .collect()
        })
        .collect();
    QMatrix::new(rows).expect("square matrix from rows")
}

fn mul(a: &QMatrix, b: &QMatrix) -> QMatrix {
    a.matmul(b).expect("conformant")
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n = (data[0] as usize % 5) + 1;
    let scale = data[1];
    let body = &data[2..];
    let a = matrix(body, n, 0, scale);
    let b = matrix(body, n, n * n, scale.rotate_left(3));

    let det_a = a.det().expect("square");
    let det_b = b.det().expect("square");
    let ab = mul(&a, &b);
    assert_eq!(
        ab.det().expect("square"),
        &det_a * &det_b,
        "det(AB) = det(A)det(B)"
    );

    match a.inv() {
        Ok(ai) => {
            assert!(!det_a.is_zero(), "inverse of a singular matrix");
            assert!(mul(&a, &ai).is_identity(), "A·A⁻¹ = I");
            assert!(mul(&ai, &a).is_identity(), "A⁻¹·A = I");
        }
        Err(_) => assert!(det_a.is_zero(), "inv failed on det = {det_a}"),
    }

    let rank = a.rank();
    let null = a.nullspace();
    assert_eq!(rank + null.len(), n, "rank–nullity");
    for v in &null {
        let av = mul(&a, v);
        assert!(
            (0..n).all(|i| av[(i, 0)].is_zero()),
            "null vector is not in the kernel"
        );
    }
    let (r, pivots) = a.rref();
    assert_eq!(pivots.len(), rank, "rref pivots vs rank");
    let (rr, pivots2) = r.rref();
    assert_eq!(rr, r, "rref is idempotent");
    assert_eq!(pivots2, pivots);

    // Cayley–Hamilton with the Berkowitz coefficients (ascending, det(A − λI)).
    let p = a.char_poly_coeffs().expect("square");
    assert_eq!(p.len(), n + 1);
    let mut acc = QMatrix::zeros(n, n).unwrap();
    let mut power = QMatrix::identity(n).unwrap();
    for c in &p {
        let term = QMatrix::new(
            (0..n)
                .map(|i| (0..n).map(|j| c * &power[(i, j)]).collect())
                .collect(),
        )
        .expect("rows");
        acc = QMatrix::new(
            (0..n)
                .map(|i| (0..n).map(|j| &acc[(i, j)] + &term[(i, j)]).collect())
                .collect(),
        )
        .expect("rows");
        power = mul(&power, &a);
    }
    assert!(
        (0..n).all(|i| (0..n).all(|j| acc[(i, j)].is_zero())),
        "Cayley–Hamilton p(A) ≠ 0"
    );
    assert_eq!(p[0], det_a, "char poly constant term det(A − 0·I)");

    if let Ok(lu) = a.lu() {
        let pa = QMatrix::new(lu.perm.iter().map(|&k| a.row(k).to_vec()).collect()).expect("rows");
        assert_eq!(mul(&lu.l, &lu.u), pa, "P·A = L·U");
        assert!(
            (0..n).all(|i| lu.l[(i, i)].is_one()),
            "L is unit lower triangular"
        );
    }

    // The symbolic tier lowers to the same kernel.
    let ctx = Context::new();
    let rows: Vec<Vec<Q>> = (0..n).map(|i| a.row(i).to_vec()).collect();
    let m = Matrix::from_ratio(&ctx, &rows).expect("rational rows");
    let det_m = m.det().expect("square");
    assert_eq!(
        det_m.as_rational(),
        Some(det_a),
        "Matrix::det vs QMatrix::det"
    );
});
