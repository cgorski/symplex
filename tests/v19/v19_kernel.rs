//! 0.20 track: the shared fraction-free kernel (`domains::exact_kernel`).
//!
//! `QMatrix`/`ZMatrix` eliminations now run on fixed-width cells (`i64`,
//! `i128`, 256-bit) and widen to `BigInt` as the minors grow.  These tests
//! drive matrices whose intermediate minors outgrow each width through the
//! public API and check that the answers are the ones the `BigInt`-only
//! computation gives:
//!
//! * a row-scaled copy `D·A` whose first row is multiplied by `3·2^255`
//!   has an entry beyond 256 bits, so the kernel starts on `BigInt` cells
//!   from the first pivot — while `A` itself starts on `i64`.  RREF, rank
//!   and nullspace are invariant under row scaling, and `inv`, `solve`,
//!   `det` transform in the obvious way, so the two paths must agree
//!   exactly;
//! * an independent plain Gauss–Jordan over `Ratio<BigInt>` (no
//!   fraction-free invariant) as a second oracle for the RREF.
//!
//! The kernel's error path (a cell that misreports) cannot be reached from
//! outside the crate; it is exercised by the unit tests inside
//! `src/domains/exact_kernel.rs` (`kernels_report_a_failed_cell_operation`,
//! `i128_division_rejects_inexact_operands`).

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use symplex::linprog::Q;
use symplex::matrix::{QMatrix, ZMatrix};
use symplex::prelude::SymplexError;

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
}

/// Random `n × m` integer matrix with entries in `[-9, 9]` and a sprinkling
/// of zeros (so that zero pivots and row swaps occur).
fn small(n: usize, m: usize, seed: u64) -> QMatrix {
    let mut g = Lcg(seed);
    ZMatrix::from_fn(n, m, |i, j| {
        if (i * 7 + j * 3) % 11 == 0 {
            BigInt::zero()
        } else {
            BigInt::from(g.range(-9, 9))
        }
    })
    .unwrap()
    .to_qmatrix()
}

/// Random `n × m` matrix with entries of about `bits` bits.
fn wide(n: usize, m: usize, bits: u32, seed: u64) -> QMatrix {
    let mut g = Lcg(seed);
    ZMatrix::from_fn(n, m, |_, _| {
        let mut v = BigInt::zero();
        let mut left = bits;
        while left > 0 {
            let take = left.min(50);
            v = (v << take) + BigInt::from(g.next() & ((1u64 << take) - 1));
            left -= take;
        }
        if g.next() & 1 == 1 { -v } else { v }
    })
    .unwrap()
    .to_qmatrix()
}

/// `D · a` for `D = diag(3·2^255, s₁, s₂, …)` with small `sᵢ ∈ {1, 2, 3}`:
/// the same row space, one row beyond 256 bits (so the kernel runs on
/// `BigInt` cells from the start) and the rest merely rescaled.
fn row_scaled(a: &QMatrix) -> (QMatrix, Vec<Q>) {
    let big: BigInt = BigInt::from(3) << 255u32;
    let scales: Vec<Q> = (0..a.nrows())
        .map(|i| {
            Ratio::from_integer(if i == 0 {
                big.clone()
            } else {
                BigInt::from((i % 3) as i64 + 1)
            })
        })
        .collect();
    let scaled = QMatrix::from_fn(a.nrows(), a.ncols(), |i, j| &a[(i, j)] * &scales[i]).unwrap();
    (scaled, scales)
}

/// `f()`, with its wall-clock time added to `total`.
fn clocked<T>(total: &mut Duration, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let out = f();
    *total += t.elapsed();
    out
}

/// Reference RREF: plain Gauss–Jordan over `Ratio<BigInt>`.
fn naive_rref(a: &QMatrix) -> (QMatrix, Vec<usize>) {
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

/// Rectangular cases whose maximal minors (the last pivot) peak, in order,
/// on `i64`, in `i128` (a 26-row single-digit matrix reaches ~100 bits), in
/// the 256-bit cells, and in `BigInt` — the last two overflowing `i64` on
/// the very first pivot (products of two 40- or 100-bit entries).
fn rectangular_cases() -> Vec<(&'static str, QMatrix)> {
    vec![
        ("8x10 small (i64)", small(8, 10, 1)),
        ("16x19 small (into i128)", small(16, 19, 2)),
        ("26x31 small (well into i128)", small(26, 31, 3)),
        (
            "10x13 of 16-bit entries (into 256-bit)",
            wide(10, 13, 16, 6),
        ),
        ("10x12 of 40-bit entries (into BigInt)", wide(10, 12, 40, 4)),
        ("8x9 of 100-bit entries (into BigInt)", wide(8, 9, 100, 5)),
    ]
}

/// Square nonsingular cases for `inv`, `solve`, `det`, with the number of
/// bits the determinant must exceed.  The last pivot of the elimination is
/// `±det`, so a determinant beyond 63 / 127 / 255 bits proves that the
/// `i64` / `i128` / 256-bit stage overflowed and the next one took over.
fn square_cases() -> Vec<(&'static str, QMatrix, u64)> {
    let mut out = Vec::new();
    for (name, seed, n, bits, min_det_bits) in [
        ("9x9 small (stays on i64)", 11u64, 9usize, 0u32, 20u64),
        ("16x16 small (into i128)", 12, 16, 0, 63),
        ("10x10 of 16-bit entries (into 256-bit)", 13, 10, 16, 127),
        ("8x8 of 40-bit entries (into BigInt)", 14, 8, 40, 255),
        ("6x6 of 100-bit entries (into BigInt)", 15, 6, 100, 255),
    ] {
        // Retry the seed until the matrix is nonsingular with a determinant
        // of the required size (nearly always the first try).
        let mut s = seed;
        loop {
            let a = if bits == 0 {
                small(n, n, s)
            } else {
                wide(n, n, bits, s)
            };
            if a.rank() == n && a.det().unwrap().numer().bits() > min_det_bits {
                out.push((name, a, min_det_bits));
                break;
            }
            s += 1000;
        }
    }
    out
}

/// The timing guard compares the kernel with the naive oracle on the same
/// matrices, measured case by case side by side, instead of bounding the
/// wall clock: the oracle took ~1.1 s of the test's 1.3 s alone (debug
/// build) and the test failed at 5.03 s under a loaded `cargo test`, while
/// the kernel's share was ~0.18 s.  Seven kernel calls per matrix (RREF,
/// rank and nullspace of `A` and `D·A`, rank of `Aᵀ`) cost ~1/6 of one
/// plain Gauss–Jordan; a kernel that lost the fraction-free invariant or
/// its fixed-width cells would cost several, and the 0.14 regression
/// (hours) many thousands.
#[test]
fn rref_rank_nullspace_agree_with_bigint_only_and_naive_paths() {
    let mut kernel = Duration::ZERO;
    let mut oracle = Duration::ZERO;
    for (name, a) in rectangular_cases() {
        let (r, pivots) = clocked(&mut kernel, || a.rref());
        // Oracle 1: the invariant-free rational elimination.
        let (nr, npivots) = clocked(&mut oracle, || naive_rref(&a));
        assert_eq!(pivots, npivots, "{name}: pivots vs naive");
        assert_eq!(r, nr, "{name}: rref vs naive");
        // Oracle 2: the row-scaled copy, which the kernel reduces on BigInt
        // cells from the first pivot; RREF, rank and nullspace are invariant.
        let (b, _) = row_scaled(&a);
        let (rb, pb) = clocked(&mut kernel, || b.rref());
        assert_eq!(pb, pivots, "{name}: pivots vs BigInt-only");
        assert_eq!(rb, r, "{name}: rref vs BigInt-only");
        assert_eq!(
            clocked(&mut kernel, || a.rank()),
            pivots.len(),
            "{name}: rank"
        );
        assert_eq!(
            clocked(&mut kernel, || b.rank()),
            pivots.len(),
            "{name}: rank of scaled copy"
        );
        let ns = clocked(&mut kernel, || a.nullspace());
        let nsb = clocked(&mut kernel, || b.nullspace());
        assert_eq!(ns, nsb, "{name}: nullspace vs BigInt-only");
        assert_eq!(ns.len(), a.ncols() - pivots.len(), "{name}: nullity");
        for v in &ns {
            assert!((&a * v).is_zero(), "{name}: A·v ≠ 0");
        }
        // The RREF is the unique one: pivot columns are unit vectors.
        for (k, &c) in pivots.iter().enumerate() {
            for i in 0..a.nrows() {
                let want = if i == k { Q::one() } else { Q::zero() };
                assert_eq!(r[(i, c)], want, "{name}: pivot column {c}");
            }
        }
        // Transpose: rank is the same, and the row space of A^T is the column space of A.
        assert_eq!(
            clocked(&mut kernel, || a.transpose().rank()),
            pivots.len(),
            "{name}: rank of transpose"
        );
    }
    assert!(
        kernel < oracle,
        "kernel {kernel:?} (7 calls per matrix) vs naive Gauss–Jordan {oracle:?} (1 call)"
    );
}

#[test]
fn inv_solve_det_agree_with_bigint_only_path() {
    let started = Instant::now();
    let mut g = Lcg(99);
    for (name, a, min_det_bits) in square_cases() {
        let n = a.nrows();
        let inv = a.inv().unwrap_or_else(|e| panic!("{name}: inv: {e}"));
        assert!((&a * &inv).is_identity(), "{name}: A·A⁻¹ ≠ I");
        assert!((&inv * &a).is_identity(), "{name}: A⁻¹·A ≠ I");
        let det = a.det().unwrap();
        assert!(
            det.numer().bits() > min_det_bits,
            "{name}: det has {} bits",
            det.numer().bits()
        );
        // (D·A)⁻¹ = A⁻¹·D⁻¹  ⇒  (D·A)⁻¹·D = A⁻¹,  det(D·A) = det(D)·det(A).
        let (b, scales) = row_scaled(&a);
        let inv_b = b.inv().unwrap();
        let inv_b_d = QMatrix::from_fn(n, n, |i, j| &inv_b[(i, j)] * &scales[j]).unwrap();
        assert_eq!(inv_b_d, inv, "{name}: inv vs BigInt-only");
        let det_d: Q = scales.iter().product();
        assert_eq!(b.det().unwrap(), det_d * &det, "{name}: det vs BigInt-only");
        // A·X = B with a random 3-column right-hand side; (D·A)·X = D·B.
        let rhs = QMatrix::from_fn(n, 3, |_, _| {
            Ratio::from_integer(BigInt::from(g.range(-50, 50)))
        })
        .unwrap();
        let x = a.solve(&rhs).unwrap();
        assert_eq!(&a * &x, rhs, "{name}: A·X ≠ B");
        let rhs_d = QMatrix::from_fn(n, 3, |i, j| &rhs[(i, j)] * &scales[i]).unwrap();
        assert_eq!(b.solve(&rhs_d).unwrap(), x, "{name}: solve vs BigInt-only");
        assert_eq!(&inv * &rhs, x, "{name}: A⁻¹·B ≠ X");
        // det via the inverse: det(A⁻¹) = 1/det(A) (the inverse's entries
        // have ~n·log n-bit denominators, so keep this to the smaller cases).
        if n <= 16 {
            assert_eq!(
                inv.det().unwrap(),
                Q::one() / &det,
                "{name}: det of inverse"
            );
        }
        // ZMatrix::det takes the same kernel.
        if let Some(z) = a.to_zmatrix() {
            assert_eq!(
                Ratio::from_integer(z.det().unwrap()),
                det,
                "{name}: ZMatrix::det"
            );
            assert_eq!(z.rank(), n, "{name}: ZMatrix::rank");
        }
    }
    assert!(
        started.elapsed().as_secs() < 5,
        "kernel tests took {:?}",
        started.elapsed()
    );
}

/// Singular and degenerate shapes through the escalating kernel: the
/// fallible operations report singularity, not an internal error.
#[test]
fn singular_and_degenerate_inputs() {
    let sing = QMatrix::from_i64(&[&[1, 2, 3], &[2, 4, 6], &[1, 1, 1]]).unwrap();
    assert_eq!(sing.rank(), 2);
    assert_eq!(sing.det().unwrap(), Q::zero());
    assert!(matches!(
        sing.inv(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    assert_eq!(sing.nullspace().len(), 1);
    // The same, scaled beyond 256 bits (BigInt cells from the start).
    let (big, _) = row_scaled(&sing);
    assert_eq!(big.rank(), 2);
    assert_eq!(big.det().unwrap(), Q::zero());
    assert!(big.inv().is_err());
    // Zero pivots that force row swaps at every step, small and wide.
    let perm = QMatrix::from_i64(&[&[0, 0, 1], &[0, 1, 0], &[1, 0, 0]]).unwrap();
    assert_eq!(perm.det().unwrap(), Ratio::from_integer(BigInt::from(-1)));
    assert_eq!(perm.inv().unwrap(), perm);
    let (perm_big, scales) = row_scaled(&perm);
    let det_d: Q = scales.iter().product();
    assert_eq!(perm_big.det().unwrap(), -det_d);
    // The zero matrix and 1×1.
    assert_eq!(QMatrix::zeros(3, 3).unwrap().rank(), 0);
    assert_eq!(QMatrix::zeros(3, 3).unwrap().det().unwrap(), Q::zero());
    assert_eq!(QMatrix::zeros(3, 3).unwrap().nullspace().len(), 3);
    let one = QMatrix::from_i64(&[&[7]]).unwrap();
    assert_eq!(
        one.inv().unwrap()[(0, 0)],
        Ratio::new(BigInt::one(), BigInt::from(7))
    );
    // Rational entries: the row-wise integerisation feeds the kernel.
    let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
    let frac = QMatrix::new(vec![vec![q(1, 2), q(1, 3)], vec![q(1, 4), q(1, 5)]]).unwrap();
    assert_eq!(frac.det().unwrap(), q(1, 60));
    assert!((&frac * &frac.inv().unwrap()).is_identity());
    let (r, p) = frac.rref();
    assert!(r.is_identity());
    assert_eq!(p, vec![0, 1]);
}

/// A 34×41 single-digit matrix (the `qmatrix/rref/40` benchmark family,
/// whose minors reach ~190 bits and so end on the 256-bit cells) reduces
/// to the naive oracle's RREF, and much faster than the oracle.
#[test]
fn benchmark_shape_matches_naive_oracle() {
    let a = small(34, 41, 42);
    let (mut kernel, mut oracle) = (Duration::ZERO, Duration::ZERO);
    let (r, pivots) = clocked(&mut kernel, || a.rref());
    let (nr, npivots) = clocked(&mut oracle, || naive_rref(&a));
    assert_eq!(pivots, npivots);
    assert_eq!(r, nr);
    // Until 0.28 a 10 s wall-clock bound on both together (~2.5 s alone in
    // a debug build, 6.5 s observed under a fully parallel `cargo nextest`),
    // almost all of it the oracle's.  Measured side by side instead: the
    // kernel takes 1/130 of the oracle (15 ms against 2.0 s, debug build),
    // the 0.14 regression (two hours) thousands of times the oracle.
    assert!(
        kernel * 10 < oracle,
        "kernel rref {kernel:?} vs naive Gauss–Jordan {oracle:?}"
    );
}
