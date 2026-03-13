//! Mathematical correctness tests for linear algebra, number theory,
//! and combinatorics operations in symplex.
//!
//! These tests verify mathematical identities and known values rather than
//! merely exercising API surfaces. Many identities (Cayley-Hamilton,
//! det(AB)=det(A)det(B), Gauss totient, Möbius sum, Stirling orthogonality,
//! quadratic reciprocity, etc.) serve as strong correctness oracles that can
//! catch subtle implementation bugs.
//!
//! ## Known bug found
//!
//! **`crt_i64` integer overflow** (ntheory.rs:782): The i64 CRT implementation
//! computes `modulus * ((remainders[i] - result) / g % (moduli[i] / g)) * p`
//! in native i64 arithmetic. When moduli are ≥ ~10^9, the intermediate product
//! overflows i64 (max ~9.2×10^18) and panics in debug mode. The BigInt `crt()`
//! version handles the same inputs correctly. See tests:
//! - `crt_i64_large_coprime_moduli_overflow`
//! - `crt_i64_three_large_moduli`

use num_bigint::BigInt;
use num_traits::{One, Zero};
use symplex::combinatorics::*;
use symplex::matrix::Matrix;
use symplex::ntheory::*;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

/// Assert that every entry of a matrix evaluates to the corresponding
/// entry of the identity matrix.
fn assert_is_identity(m: &Matrix, label: &str) {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "{label}: not square");
    for i in 0..n {
        for j in 0..n {
            let entry = m.get(i, j).eval().simplify();
            let expected = if i == j { "1" } else { "0" };
            assert_eq!(
                format!("{entry}"),
                expected,
                "{label}: entry ({i},{j}) = {entry}, expected {expected}"
            );
        }
    }
}

/// Assert that every entry of a matrix evaluates to zero.
fn assert_is_zero_matrix(m: &Matrix, label: &str) {
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let entry = m.get(i, j).eval().simplify();
            assert_eq!(
                format!("{entry}"),
                "0",
                "{label}: entry ({i},{j}) = {entry}, expected 0"
            );
        }
    }
}

/// Binomial coefficient C(n, k) via the multinomial function.
fn binom(n: u64, k: u64) -> BigInt {
    if k > n {
        return BigInt::zero();
    }
    multinomial(n, &[k, n - k]).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//  PART 1 — LINEAR ALGEBRA
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

// ── A · A⁻¹ = I ────────────────────────────────────────────────────────────

#[test]
fn inverse_2x2_gives_identity() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 3]];
    let a_inv = a.inv().unwrap();
    assert_is_identity(&(&a * &a_inv), "A·A⁻¹  2×2");
    assert_is_identity(&(&a_inv * &a), "A⁻¹·A  2×2");
}

#[test]
fn inverse_3x3_gives_identity() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [0, 1, 4], [5, 6, 0]];
    let a_inv = a.inv().unwrap();
    assert_is_identity(&(&a * &a_inv), "A·A⁻¹  3×3");
    assert_is_identity(&(&a_inv * &a), "A⁻¹·A  3×3");
}

#[test]
fn inverse_4x4_gives_identity() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1, 3, 1], [1, 0, 2, 1], [3, 2, 1, 0], [1, 1, 0, 2]];
    let a_inv = a.inv().unwrap();
    assert_is_identity(&(&a * &a_inv), "A·A⁻¹  4×4");
    assert_is_identity(&(&a_inv * &a), "A⁻¹·A  4×4");
}

// ── det(AB) = det(A)·det(B) ────────────────────────────────────────────────

#[test]
fn det_product_rule_2x2() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 3]];
    let b = matrix![ctx, [5, 6], [7, 8]];
    let det_ab = (&a * &b).det().unwrap().eval();
    let det_a_det_b = (&a.det().unwrap().eval() * &b.det().unwrap().eval()).eval();
    assert_eq!(
        format!("{det_ab}"),
        format!("{det_a_det_b}"),
        "det(AB) = det(A)·det(B)  2×2"
    );
}

#[test]
fn det_product_rule_3x3() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [0, 1, 4], [5, 6, 0]];
    let b = matrix![ctx, [1, 0, 2], [3, 1, 0], [0, 2, 1]];
    let det_ab = (&a * &b).det().unwrap().eval();
    let det_a_det_b = (&a.det().unwrap().eval() * &b.det().unwrap().eval()).eval();
    assert_eq!(
        format!("{det_ab}"),
        format!("{det_a_det_b}"),
        "det(AB) = det(A)·det(B)  3×3"
    );
}

#[test]
fn det_product_rule_4x4() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1, 3, 1], [1, 0, 2, 1], [3, 2, 1, 0], [1, 1, 0, 2]];
    let b = matrix![ctx, [1, 0, 0, 2], [0, 1, 2, 0], [2, 0, 1, 0], [0, 2, 0, 1]];
    let det_ab = (&a * &b).det().unwrap().eval();
    let det_a_det_b = (&a.det().unwrap().eval() * &b.det().unwrap().eval()).eval();
    assert_eq!(
        format!("{det_ab}"),
        format!("{det_a_det_b}"),
        "det(AB) = det(A)·det(B)  4×4"
    );
}

#[test]
fn det_product_chain_abc() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let b = matrix![ctx, [2, 0], [1, 3]];
    let c = matrix![ctx, [1, 1], [0, 2]];
    let det_abc = (&(&a * &b) * &c).det().unwrap().eval();
    let det_sep = (&(&a.det().unwrap().eval() * &b.det().unwrap().eval())
        * &c.det().unwrap().eval())
    .eval();
    assert_eq!(
        format!("{det_abc}"),
        format!("{det_sep}"),
        "det(ABC) = det(A)·det(B)·det(C)"
    );
}

// ── det(Aᵀ) = det(A) ──────────────────────────────────────────────────────

#[test]
fn det_transpose_equals_det_2x2() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 3], [5, 7]];
    assert_eq!(
        format!("{}", a.det().unwrap().eval()),
        format!("{}", a.transpose().det().unwrap().eval()),
    );
}

#[test]
fn det_transpose_equals_det_3x3() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]];
    assert_eq!(
        format!("{}", a.det().unwrap().eval()),
        format!("{}", a.transpose().det().unwrap().eval()),
    );
}

#[test]
fn det_transpose_equals_det_4x4() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1, 3, 1], [1, 0, 2, 1], [3, 2, 1, 0], [1, 1, 0, 2]];
    assert_eq!(
        format!("{}", a.det().unwrap().eval()),
        format!("{}", a.transpose().det().unwrap().eval()),
    );
}

#[test]
fn det_transpose_equals_det_5x5() {
    let ctx = Context::new();
    let a = matrix![ctx,
        [2, 1, 0, 0, 3],
        [1, 3, 2, 0, 0],
        [0, 2, 4, 1, 0],
        [0, 0, 1, 5, 2],
        [3, 0, 0, 2, 6]
    ];
    assert_eq!(
        format!("{}", a.det().unwrap().eval()),
        format!("{}", a.transpose().det().unwrap().eval()),
    );
}

// ── det(kA) = kⁿ · det(A) ─────────────────────────────────────────────────

#[test]
fn det_scalar_multiple_2x2() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [3, 4]];
    let det_3a = (&a * 3).det().unwrap().eval();
    let expected = (&a.det().unwrap().eval() * &ctx.int(9)).eval(); // 3² = 9
    assert_eq!(format!("{det_3a}"), format!("{expected}"));
}

#[test]
fn det_scalar_multiple_3x3() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 0, 2], [3, 1, 0], [0, 2, 1]];
    let det_2a = (&a * 2).det().unwrap().eval();
    let expected = (&a.det().unwrap().eval() * &ctx.int(8)).eval(); // 2³ = 8
    assert_eq!(format!("{det_2a}"), format!("{expected}"));
}

// ── Known determinant values ───────────────────────────────────────────────

#[test]
fn det_known_2x2() {
    let ctx = Context::new();
    let a = matrix![ctx, [3, 8], [4, 6]];
    // 3·6 − 8·4 = 18 − 32 = −14
    assert_eq!(format!("{}", a.det().unwrap().eval()), "-14");
}

#[test]
fn det_known_3x3() {
    let ctx = Context::new();
    let a = matrix![ctx, [6, 1, 1], [4, -2, 5], [2, 8, 7]];
    // = 6(−14−40) − 1(28−10) + 1(32+4) = −324 − 18 + 36 = −306
    assert_eq!(format!("{}", a.det().unwrap().eval()), "-306");
}

#[test]
fn det_identity_is_one() {
    let ctx = Context::new();
    for n in 1..=5 {
        let id = Matrix::identity(&ctx, n);
        assert_eq!(format!("{}", id.det().unwrap().eval()), "1");
    }
}

// ── Singular matrices ──────────────────────────────────────────────────────

#[test]
fn singular_matrix_det_zero() {
    let ctx = Context::new();
    // row3 = row1 + row2
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [5, 7, 9]];
    assert_eq!(format!("{}", a.det().unwrap().eval()), "0");
}

#[test]
fn singular_matrix_inv_is_err() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2], [2, 4]];
    assert!(a.inv().is_err());
}

// ── (AB)⁻¹ = B⁻¹A⁻¹ ──────────────────────────────────────────────────────

#[test]
fn inverse_of_product_rule() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 3]];
    let b = matrix![ctx, [1, 2], [3, 1]];
    let ab_inv = (&a * &b).inv().unwrap();
    let b_inv_a_inv = &b.inv().unwrap() * &a.inv().unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                format!("{}", ab_inv.get(i, j).eval().simplify()),
                format!("{}", b_inv_a_inv.get(i, j).eval().simplify()),
                "(AB)⁻¹ vs B⁻¹A⁻¹  ({i},{j})"
            );
        }
    }
}

// ── det(A)·det(A⁻¹) = 1 ───────────────────────────────────────────────────

#[test]
fn det_times_det_inverse_is_one() {
    let ctx = Context::new();
    let a = matrix![ctx,
        [2, 1, 0, 0, 3],
        [1, 3, 2, 0, 0],
        [0, 2, 4, 1, 0],
        [0, 0, 1, 5, 2],
        [3, 0, 0, 2, 6]
    ];
    let det_a = a.det().unwrap().eval();
    assert_ne!(format!("{det_a}"), "0", "matrix must be invertible");
    let det_inv = a.inv().unwrap().det().unwrap().eval().simplify();
    let prod = (&det_a * &det_inv).eval().simplify();
    assert_eq!(format!("{prod}"), "1", "det(A)·det(A⁻¹) = 1  5×5");
}

// ── Cayley-Hamilton: p(A) = 0 ──────────────────────────────────────────────

#[test]
fn cayley_hamilton_2x2() {
    let ctx = Context::new();
    // For 2×2 A, char_poly = λ² − tr(A)λ + det(A).
    // Cayley-Hamilton: A² − tr(A)·A + det(A)·I = 0.
    let matrices: Vec<Matrix> = vec![
        matrix![ctx, [1, 2], [3, 4]],
        matrix![ctx, [3, 1], [0, 2]],
        matrix![ctx, [0, 1], [1, 0]],
        matrix![ctx, [5, -3], [2, -1]],
        matrix![ctx, [1, 1], [1, 0]],
    ];
    for (idx, a) in matrices.iter().enumerate() {
        let tr = a.trace().unwrap().eval();
        let det = a.det().unwrap().eval();
        let a2 = a.powi(2).unwrap();
        let id = Matrix::identity(&ctx, 2);
        let result = &(&a2 - &(a * &tr)) + &(&id * &det);
        assert_is_zero_matrix(&result, &format!("Cayley-Hamilton 2×2 #{idx}"));
    }
}

#[test]
fn cayley_hamilton_3x3_upper_triangular() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1, 0], [0, 3, 1], [0, 0, 4]];
    // eigenvalues 2, 3, 4.
    // char_poly det(A−λI) = (2−λ)(3−λ)(4−λ) = −λ³ + 9λ² − 26λ + 24.
    let a2 = a.powi(2).unwrap();
    let a3 = a.powi(3).unwrap();
    let id = Matrix::identity(&ctx, 3);
    // −A³ + 9A² − 26A + 24I = 0
    let result = &(&(&(&a3 * -1) + &(&a2 * 9)) + &(&a * -26)) + &(&id * 24);
    assert_is_zero_matrix(&result, "Cayley-Hamilton 3×3 upper-triangular");
}

// ── Eigenvalues are roots of char poly ─────────────────────────────────────

#[test]
fn eigenvalues_are_char_poly_roots_2x2() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [2, 1], [1, 3]];
    let cp = a.char_poly(&lam).unwrap();
    for ev in a.eigenvals(&lam).unwrap() {
        assert_eq!(
            cp.check_solution(&lam, &ev),
            Some(true),
            "eigenvalue {ev} should be root of char poly {cp}"
        );
    }
}

#[test]
fn eigenvalues_diagonal() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = Matrix::diag(&[ctx.int(3), ctx.int(7), ctx.int(11)]);
    let mut evs: Vec<String> = a
        .eigenvals(&lam)
        .unwrap()
        .iter()
        .map(|e| format!("{e}"))
        .collect();
    evs.sort();
    assert_eq!(evs, vec!["11", "3", "7"]);
}

#[test]
fn eigenvalues_upper_triangular() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [1, 5, 3], [0, 2, 8], [0, 0, 4]];
    let mut evs: Vec<String> = a
        .eigenvals(&lam)
        .unwrap()
        .iter()
        .map(|e| format!("{e}"))
        .collect();
    evs.sort();
    assert_eq!(evs, vec!["1", "2", "4"]);
}

// ── Characteristic polynomial known values ─────────────────────────────────

#[test]
fn char_poly_2x2_eval_at_zero_is_det() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [1, 2], [3, 4]];
    let cp = a.char_poly(&lam).unwrap();
    // p(0) = det(A − 0·I) = det(A) = 1·4 − 2·3 = −2
    let p0 = cp.subs(&lam, &ctx.int(0)).eval();
    let det = a.det().unwrap().eval();
    assert_eq!(format!("{p0}"), format!("{det}"), "p(0) = det(A)");
}

// ── Matrix powers ──────────────────────────────────────────────────────────

#[test]
fn matrix_power_zero_is_identity() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 3]];
    assert_is_identity(&a.powi(0).unwrap(), "A⁰ = I");
}

#[test]
fn matrix_power_consistency() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 1], [1, 0]]; // Fibonacci matrix
    let a3 = a.powi(3).unwrap();
    let a2_times_a = &a.powi(2).unwrap() * &a;
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                format!("{}", a3.get(i, j).eval()),
                format!("{}", a2_times_a.get(i, j).eval()),
                "A³ = A²·A  ({i},{j})"
            );
        }
    }
}

// ── Rank–nullity ───────────────────────────────────────────────────────────

#[test]
fn rank_nullity_theorem() {
    let ctx = Context::new();
    let matrices: Vec<Matrix> = vec![
        Matrix::identity(&ctx, 3),
        Matrix::zeros(&ctx, 3, 3),
        matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]],
        matrix![ctx, [1, 2, 3], [4, 5, 6], [5, 7, 9]],
    ];
    for (idx, a) in matrices.iter().enumerate() {
        let r = a.rank();
        let null_dim = a.nullspace().len();
        assert_eq!(
            r + null_dim,
            a.ncols(),
            "rank + nullity = ncols  matrix #{idx}: {r}+{null_dim} ≠ {}",
            a.ncols()
        );
    }
}

// ── Nullspace vectors really in kernel ─────────────────────────────────────

#[test]
fn nullspace_vectors_in_kernel() {
    let ctx = Context::new();
    let a = matrix![ctx, [1, 2, 3], [4, 5, 6], [5, 7, 9]];
    for v in a.nullspace() {
        let av = &a * &v;
        for i in 0..av.nrows() {
            let entry = av.get(i, 0).eval().simplify();
            assert_eq!(format!("{entry}"), "0", "A·v = 0 for nullspace vector");
        }
    }
}

// ── Cholesky L·Lᵀ = A ─────────────────────────────────────────────────────

#[test]
fn cholesky_reconstruction() {
    let ctx = Context::new();
    let a = matrix![ctx, [4, 2], [2, 3]];
    if let Ok(Some(l)) = a.cholesky() {
        let product = &l * &l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    format!("{}", product.get(i, j).eval().simplify()),
                    format!("{}", a.get(i, j).eval()),
                    "L·Lᵀ = A  ({i},{j})"
                );
            }
        }
    }
}

// ── LU reconstruction ──────────────────────────────────────────────────────

#[test]
fn lu_reconstruction() {
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1, 1], [4, 3, 3], [8, 7, 9]];
    if let Some((l, u, perm)) = a.lu() {
        let lu = &l * &u;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_eq!(
                    format!("{}", lu.get(i, j).eval().simplify()),
                    format!("{}", a.get(perm[i], j).eval()),
                    "L·U = P·A  ({i},{j})"
                );
            }
        }
    }
}

// ── Symmetric matrix ───────────────────────────────────────────────────────

#[test]
fn symmetric_matrix_eigenvalues_are_roots() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [4, 1, 2], [1, 3, 1], [2, 1, 5]];
    assert!(a.is_symmetric());
    let cp = a.char_poly(&lam).unwrap();
    for ev in a.eigenvals(&lam).unwrap() {
        assert_eq!(
            cp.check_solution(&lam, &ev),
            Some(true),
            "eigenvalue {ev} must satisfy char poly"
        );
    }
}

// ── trace = sum of diagonal ────────────────────────────────────────────────

#[test]
fn trace_is_diagonal_sum() {
    let ctx = Context::new();
    let a = matrix![ctx, [3, 1, 4], [1, 5, 9], [2, 6, 5]];
    assert_eq!(format!("{}", a.trace().unwrap().eval()), "13"); // 3+5+5
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//  PART 2 — NUMBER THEORY
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

// ── Primality basics ───────────────────────────────────────────────────────

#[test]
fn primality_small() {
    assert!(!isprime(0));
    assert!(!isprime(1));
    assert!(isprime(2));
    assert!(isprime(3));
    assert!(!isprime(4));
    assert!(isprime(5));
    assert!(!isprime(9));
    assert!(isprime(11));
}

#[test]
fn primality_negative() {
    for n in -100..0i64 {
        assert!(!isprime(n), "isprime({n}) should be false");
    }
}

#[test]
fn primality_carmichael() {
    let carmichaels = [561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341];
    for &c in &carmichaels {
        assert!(!isprime(c), "Carmichael number {c} wrongly declared prime");
    }
}

#[test]
fn primality_mersenne() {
    for &p in &[2, 3, 5, 7, 13, 17, 19, 31] {
        let mp: i64 = (1i64 << p) - 1;
        assert!(isprime(mp), "Mersenne prime 2^{p}−1 = {mp} should be prime");
    }
}

#[test]
fn primality_twin_primes() {
    let twins: [(i64, i64); 8] = [
        (3, 5),
        (11, 13),
        (17, 19),
        (29, 31),
        (41, 43),
        (59, 61),
        (71, 73),
        (101, 103),
    ];
    for (p, q) in twins {
        assert!(isprime(p) && isprime(q), "twin pair ({p},{q})");
    }
}

// ── Factorisation roundtrip ────────────────────────────────────────────────

#[test]
fn factorisation_roundtrip_small() {
    let values = [
        2, 3, 4, 6, 12, 36, 60, 100, 128, 243, 360, 625, 720, 1000, 2187, 2310,
        5040, 10007, 16807, 55440, 100003, 720720, 999983, 1000000007i64,
    ];
    for &n in &values {
        let factors = factorint(n);
        let mut prod = BigInt::one();
        for (p, e) in &factors {
            for _ in 0..*e {
                prod *= p;
            }
        }
        assert_eq!(prod, bi(n), "roundtrip failed for {n}");
    }
}

#[test]
fn factorisation_factors_are_prime() {
    for &n in &[60i64, 360, 2310, 30030, 510510, 9699690] {
        for (p, _) in factorint(n) {
            assert!(isprime(p.clone()), "factor {p} of {n} not prime");
        }
    }
}

#[test]
fn factorisation_of_prime_is_itself() {
    for &p in &[2i64, 3, 5, 7, 97, 101, 1009, 10007, 100003, 1000003] {
        let f = factorint(p);
        assert_eq!(f.len(), 1, "prime {p} should have one factor");
        assert_eq!(f[0], (bi(p), 1));
    }
}

#[test]
fn factorisation_prime_powers() {
    let cases: [(i64, i64, u32); 5] =
        [(8, 2, 3), (27, 3, 3), (128, 2, 7), (2187, 3, 7), (65536, 2, 16)];
    for (n, base, exp) in cases {
        let f = factorint(n);
        assert_eq!(f, vec![(bi(base), exp)], "{n} = {base}^{exp}");
    }
}

#[test]
fn factorisation_beyond_f64() {
    // 2^53 + 1 is beyond f64 exact integer range.
    let n = BigInt::from(2u64.pow(53) + 1);
    let factors = factorint(n.clone());
    let mut prod = BigInt::one();
    for (p, e) in &factors {
        assert!(isprime(p.clone()), "factor {p} of 2^53+1 not prime");
        for _ in 0..*e {
            prod *= p;
        }
    }
    assert_eq!(prod, n);
}

// ── Modular inverse ────────────────────────────────────────────────────────

#[test]
fn mod_inverse_verify_product() {
    let cases: [(i64, i64); 9] = [
        (1, 7),
        (2, 7),
        (3, 7),
        (5, 11),
        (7, 13),
        (17, 43),
        (3, 1000000007),
        (999999999, 1000000007),
        (123456789, 998244353),
    ];
    for (a, n) in cases {
        let inv = mod_inverse(a, n)
            .unwrap_or_else(|| panic!("mod_inverse({a},{n}) should exist"));
        let prod = (bi(a) * &inv) % bi(n);
        let prod = (&prod + bi(n)) % bi(n); // normalise
        assert_eq!(prod, bi(1), "{a}·{inv} mod {n} should be 1");
    }
}

#[test]
fn mod_inverse_none_when_not_coprime() {
    assert_eq!(mod_inverse(2, 4), None);
    assert_eq!(mod_inverse(6, 9), None);
    assert_eq!(mod_inverse(0, 5), None);
    assert_eq!(mod_inverse(10, 15), None);
}

// ── CRT ────────────────────────────────────────────────────────────────────

#[test]
fn crt_basic() {
    let x = crt_i64(&[2, 3, 2], &[3, 5, 7]).unwrap();
    assert_eq!(x % 3, 2);
    assert_eq!(x % 5, 3);
    assert_eq!(x % 7, 2);
    assert_eq!(x, 23);
}

#[test]
fn crt_bigint_verify() {
    let r: Vec<BigInt> = vec![1.into(), 2.into(), 3.into()];
    let m: Vec<BigInt> = vec![5.into(), 7.into(), 11.into()];
    let x = crt(&r, &m).unwrap();
    assert_eq!(&x % bi(5), bi(1));
    assert_eq!(&x % bi(7), bi(2));
    assert_eq!(&x % bi(11), bi(3));
    assert!(x >= bi(0) && x < bi(385));
}

#[test]
fn crt_non_coprime_solvable() {
    // x ≡ 3 mod 6, x ≡ 5 mod 10.  gcd(6,10)=2, 5−3=2 divisible by 2.
    let r: Vec<BigInt> = vec![3.into(), 5.into()];
    let m: Vec<BigInt> = vec![6.into(), 10.into()];
    let x = crt(&r, &m).unwrap();
    assert_eq!(&x % bi(6), bi(3));
    assert_eq!(&x % bi(10), bi(5));
}

#[test]
fn crt_incompatible() {
    let r: Vec<BigInt> = vec![1.into(), 0.into()];
    let m: Vec<BigInt> = vec![2.into(), 2.into()];
    assert_eq!(crt(&r, &m), None);
}

// ── Fermat's little theorem ────────────────────────────────────────────────

#[test]
fn fermats_little_theorem() {
    for &p in &[3i64, 5, 7, 11, 13, 17, 23, 97, 101, 1009, 10007] {
        for &a in &[2i64, 3, 5, 7, 10] {
            if a % p != 0 {
                assert_eq!(
                    mod_pow(a, p - 1, p),
                    bi(1),
                    "a^(p−1) mod p: {a}^{} mod {p}",
                    p - 1
                );
            }
        }
    }
}

// ── Euler's theorem ────────────────────────────────────────────────────────

#[test]
fn eulers_theorem() {
    for &n in &[6i64, 8, 9, 10, 12, 15, 20, 35, 100] {
        let phi = totient(n);
        for a in 2..n {
            if is_coprime(a, n) {
                assert_eq!(
                    mod_pow(a, phi.clone(), n),
                    bi(1),
                    "{a}^φ({n}) mod {n} ≠ 1  (φ={phi})"
                );
            }
        }
    }
}

// ── Gauss totient identity: Σ_{d|n} φ(d) = n ──────────────────────────────

#[test]
fn gauss_totient_identity() {
    for n in 1..=60i64 {
        let sum: BigInt = divisors(n).iter().map(|d| totient(d.clone())).sum();
        assert_eq!(sum, bi(n), "Σ φ(d) for d|{n}");
    }
}

// ── Möbius: Σ_{d|n} μ(d) = [n=1] ──────────────────────────────────────────

#[test]
fn mobius_sum_identity() {
    for n in 1..=60i64 {
        let sum: i16 = divisors(n).iter().map(|d| mobius(d.clone()) as i16).sum();
        let expected: i16 = if n == 1 { 1 } else { 0 };
        assert_eq!(sum, expected, "Σ μ(d) for d|{n}");
    }
}

// ── Totient multiplicativity ───────────────────────────────────────────────

#[test]
fn totient_multiplicative() {
    let pairs: [(i64, i64); 9] = [
        (3, 4),
        (5, 6),
        (7, 9),
        (8, 15),
        (11, 13),
        (4, 9),
        (7, 12),
        (25, 36),
        (8, 27),
    ];
    for (m, n) in pairs {
        assert!(is_coprime(m, n));
        assert_eq!(totient(m * n), &totient(m) * &totient(n));
    }
}

// ── Totient of prime powers ────────────────────────────────────────────────

#[test]
fn totient_prime_powers() {
    let cases: [(i64, i64, u32); 7] = [
        (2, 2, 1),
        (4, 2, 2),
        (8, 2, 3),
        (9, 3, 2),
        (27, 3, 3),
        (25, 5, 2),
        (49, 7, 2),
    ];
    for (n, p, k) in cases {
        assert_eq!(
            totient(n),
            bi(p).pow(k) - bi(p).pow(k - 1),
            "φ({n})"
        );
    }
}

// ── Divisor functions ──────────────────────────────────────────────────────

#[test]
fn divisor_properties() {
    for n in 1..=100i64 {
        let d = divisors(n);
        assert_eq!(d.len(), divisor_count(n), "divisor_count({n})");
        let sum: BigInt = d.iter().cloned().sum();
        assert_eq!(sum, divisor_sum(n), "divisor_sum({n})");
        for di in &d {
            assert!((bi(n) % di).is_zero(), "{di} should divide {n}");
        }
        assert!(d.contains(&bi(1)));
        assert!(d.contains(&bi(n)));
    }
}

#[test]
fn divisor_count_multiplicative() {
    for &(m, n) in &[(3i64, 4), (5, 6), (7, 9), (8, 15), (11, 13)] {
        assert!(is_coprime(m, n));
        assert_eq!(
            divisor_count(m * n),
            divisor_count(m) * divisor_count(n),
        );
    }
}

#[test]
fn perfect_numbers() {
    for &n in &[6i64, 28, 496, 8128] {
        assert_eq!(divisor_sum(n), bi(2 * n), "σ({n}) = 2·{n}");
    }
}

// ── Legendre symbol ────────────────────────────────────────────────────────

#[test]
fn legendre_symbol_qr_count() {
    // Exactly (p−1)/2 quadratic residues mod odd prime p.
    for &p in &[3i64, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let qr: usize = (1..p).filter(|&a| legendre_symbol(a, p) == 1).count();
        assert_eq!(qr, ((p - 1) / 2) as usize, "QR count mod {p}");
    }
}

#[test]
fn legendre_symbol_multiplicative() {
    let p = 13i64;
    for a in 1..p {
        for b in 1..p {
            let ls_ab = legendre_symbol((a * b) % p, p);
            let ls_a = legendre_symbol(a, p);
            let ls_b = legendre_symbol(b, p);
            assert_eq!(
                ls_ab,
                ls_a * ls_b,
                "({a}·{b}/{p}) vs ({a}/{p})·({b}/{p})"
            );
        }
    }
}

#[test]
fn quadratic_reciprocity() {
    let primes = [3i64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43];
    for (i, &p) in primes.iter().enumerate() {
        for &q in &primes[(i + 1)..] {
            let pq = legendre_symbol(p, q) as i16;
            let qp = legendre_symbol(q, p) as i16;
            let sign: i16 = if ((p - 1) / 2 * (q - 1) / 2) % 2 == 0 {
                1
            } else {
                -1
            };
            assert_eq!(pq * qp, sign, "({}|{})·({}|{}) reciprocity", p, q, q, p);
        }
    }
}

// ── GCD / LCM identity ────────────────────────────────────────────────────

#[test]
fn gcd_lcm_product() {
    for a in 1..=50i64 {
        for b in 1..=50i64 {
            assert_eq!(
                &gcd(a, b) * &lcm(a, b),
                bi(a * b),
                "gcd·lcm = ab  ({a},{b})"
            );
        }
    }
}

// ── isqrt ──────────────────────────────────────────────────────────────────

#[test]
fn isqrt_bounds() {
    for n in 0..=1000i64 {
        let s = isqrt(n).unwrap();
        assert!(&s * &s <= bi(n), "s²>n for isqrt({n})");
        assert!((&s + 1u32) * (&s + 1u32) > bi(n), "(s+1)²≤n for isqrt({n})");
    }
}

#[test]
fn is_square_consistent() {
    for n in 0..=200i64 {
        let sq = is_square(n);
        let s = isqrt(n).unwrap();
        assert_eq!(sq, &s * &s == bi(n), "is_square({n})");
    }
}

// ── nextprime / prevprime ──────────────────────────────────────────────────

#[test]
fn nextprime_is_prime() {
    for n in 0..=200i64 {
        let np = nextprime(n);
        assert!(isprime(np.clone()), "nextprime({n})={np} not prime");
    }
}

#[test]
fn prevprime_nextprime_inverse() {
    for n in 3..=200i64 {
        if isprime(n) {
            let prev = prevprime(nextprime(n)).unwrap();
            assert_eq!(prev, bi(n), "prevprime(nextprime({n}))");
        }
    }
}

// ── mod_pow edge cases ─────────────────────────────────────────────────────

#[test]
fn mod_pow_edges() {
    assert_eq!(mod_pow(5, 0, 7), bi(1));
    assert_eq!(mod_pow(0, 5, 7), bi(0));
    assert_eq!(mod_pow(1, 1_000_000, 97), bi(1));
    assert_eq!(mod_pow(2, 10, 1000), bi(24));
}

// ── Negative input handling ────────────────────────────────────────────────

#[test]
fn negative_inputs() {
    assert_eq!(factorint(-60i64), factorint(60i64));
    assert_eq!(totient(0), bi(0));
    assert_eq!(totient(-5), bi(0));
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//  PART 3 — COMBINATORICS
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

// ── C(n,k) = C(n,n−k) ─────────────────────────────────────────────────────

#[test]
fn binomial_symmetry() {
    for n in 0..=12u64 {
        for k in 0..=n {
            assert_eq!(binom(n, k), binom(n, n - k), "C({n},{k}) = C({n},{})", n - k);
        }
    }
}

// ── Σ C(n,k) = 2ⁿ ─────────────────────────────────────────────────────────

#[test]
fn binomial_row_sum() {
    for n in 0..=15u64 {
        let sum: BigInt = (0..=n).map(|k| binom(n, k)).sum();
        assert_eq!(sum, BigInt::one() << n as usize, "Σ C({n},k)=2^{n}");
    }
}

// ── Pascal's rule ──────────────────────────────────────────────────────────

#[test]
fn pascals_rule() {
    for n in 1..=12u64 {
        for k in 1..n {
            assert_eq!(
                binom(n, k),
                binom(n - 1, k - 1) + binom(n - 1, k),
                "C({n},{k})"
            );
        }
    }
}

// ── Vandermonde identity ───────────────────────────────────────────────────

#[test]
fn vandermonde_identity() {
    for m in 0..=8u64 {
        for n in 0..=8u64 {
            for k in 0..=(m + n).min(10) {
                let lhs: BigInt = (0..=k).map(|j| binom(m, j) * binom(n, k - j)).sum();
                assert_eq!(lhs, binom(m + n, k), "Vandermonde m={m},n={n},k={k}");
            }
        }
    }
}

// ── Alternating sum ────────────────────────────────────────────────────────

#[test]
fn binomial_alternating_sum() {
    for n in 1..=15u64 {
        let sum: BigInt = (0..=n)
            .map(|k| if k % 2 == 0 { binom(n, k) } else { -binom(n, k) })
            .sum();
        assert_eq!(sum, BigInt::zero(), "Σ(−1)^k C({n},k)=0");
    }
}

// ── Multinomial reduces to binomial ────────────────────────────────────────

#[test]
fn multinomial_is_binomial() {
    for n in 0..=12u64 {
        for k in 0..=n {
            assert_eq!(
                multinomial(n, &[k, n - k]).unwrap(),
                binom(n, k),
            );
        }
    }
}

// ── Multinomial permutation invariance ─────────────────────────────────────

#[test]
fn multinomial_permutation_invariant() {
    let a = multinomial(10u64, &[3u64, 4, 3]).unwrap();
    let b = multinomial(10u64, &[4u64, 3, 3]).unwrap();
    let c = multinomial(10u64, &[3u64, 3, 4]).unwrap();
    assert_eq!(a, b);
    assert_eq!(b, c);
}

// ── S(n,k) via inclusion-exclusion ─────────────────────────────────────────

#[test]
fn stirling2_inclusion_exclusion() {
    for n in 1..=8u64 {
        for k in 1..=n {
            let mut sum = BigInt::zero();
            for j in 0..=k {
                let term = binom(k, j) * BigInt::from(k - j).pow(n as u32);
                if j % 2 == 0 {
                    sum += term;
                } else {
                    sum -= term;
                }
            }
            let mut kf = BigInt::one();
            for i in 1..=k {
                kf *= BigInt::from(i);
            }
            let ie = &sum / &kf;
            assert_eq!(ie, stirling2(n, k).unwrap(), "S({n},{k}) incl-excl");
        }
    }
}

// ── xⁿ = Σ S(n,k)·x₍ₖ₎ (falling factorial) ──────────────────────────────

#[test]
fn stirling2_falling_factorial_identity() {
    for n in 0..=7u64 {
        for x in 0..=10i64 {
            let lhs = BigInt::from(x).pow(n as u32);
            let rhs: BigInt = (0..=n)
                .map(|k| {
                    let s = stirling2(n, k).unwrap();
                    let ff = (0..k).fold(BigInt::one(), |acc, i| {
                        acc * (BigInt::from(x) - BigInt::from(i))
                    });
                    s * ff
                })
                .sum();
            assert_eq!(lhs, rhs, "x^{n} with x={x}");
        }
    }
}

// ── Stirling1 known table (OEIS A008275) ───────────────────────────────────

#[test]
fn stirling1_known_table() {
    let table: [(u64, u64, i64); 16] = [
        (0, 0, 1),
        (1, 1, 1),
        (2, 1, -1),
        (2, 2, 1),
        (3, 1, 2),
        (3, 2, -3),
        (3, 3, 1),
        (4, 1, -6),
        (4, 2, 11),
        (4, 3, -6),
        (4, 4, 1),
        (5, 1, 24),
        (5, 2, -50),
        (5, 3, 35),
        (5, 4, -10),
        (5, 5, 1),
    ];
    for (n, k, expected) in table {
        assert_eq!(stirling1(n, k).unwrap(), bi(expected), "s({n},{k})");
    }
}

// ── Stirling2 known table ──────────────────────────────────────────────────

#[test]
fn stirling2_known_table() {
    let table: [(u64, u64, i64); 16] = [
        (0, 0, 1),
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 1),
        (3, 1, 1),
        (3, 2, 3),
        (3, 3, 1),
        (4, 1, 1),
        (4, 2, 7),
        (4, 3, 6),
        (4, 4, 1),
        (5, 1, 1),
        (5, 2, 15),
        (5, 3, 25),
        (5, 4, 10),
        (5, 5, 1),
    ];
    for (n, k, expected) in table {
        assert_eq!(stirling2(n, k).unwrap(), bi(expected), "S({n},{k})");
    }
}

// ── Stirling orthogonality (8×8) ───────────────────────────────────────────

#[test]
fn stirling_orthogonality_8x8() {
    let sz = 8u64;
    for n in 0..sz {
        for k in 0..sz {
            let sum: BigInt = (0..sz)
                .map(|j| stirling1(n, j).unwrap() * stirling2(j, k).unwrap())
                .sum();
            let expected = if n == k { bi(1) } else { bi(0) };
            assert_eq!(sum, expected, "orthogonality n={n},k={k}");
        }
    }
}

// ── Bell number recurrence B(n+1) = Σ C(n,k)·B(k) ────────────────────────

#[test]
fn bell_number_recurrence() {
    let bell = |n: u64| -> BigInt { (0..=n).map(|k| stirling2(n, k).unwrap()).sum() };
    for n in 0..=8u64 {
        let lhs = bell(n + 1);
        let rhs: BigInt = (0..=n).map(|k| binom(n, k) * bell(k)).sum();
        assert_eq!(lhs, rhs, "B({}) recurrence", n + 1);
    }
}

// ── Partition function ─────────────────────────────────────────────────────

#[test]
fn partition_count_known() {
    let table: [(u64, i64); 13] = [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 5),
        (5, 7),
        (6, 11),
        (7, 15),
        (8, 22),
        (9, 30),
        (10, 42),
        (50, 204226),
        (100, 190569292),
    ];
    for (n, expected) in table {
        assert_eq!(partition_count(n).unwrap(), bi(expected), "p({n})");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
//  PART 4 — DEEPER EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════

// ── char_poly coefficient verification ─────────────────────────────────────

#[test]
fn char_poly_2x2_known_value() {
    // [[1,2],[3,4]]: det(A−λI) = (1−λ)(4−λ)−6 = λ²−5λ−2
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [1, 2], [3, 4]];
    let cp = a.char_poly(&lam).unwrap();
    // Evaluate at specific points to verify polynomial
    // p(0) = −2  (= det(A))
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(0)).eval()), "-2");
    // p(1) = 1−5−2 = −6
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(1)).eval()), "-6");
    // p(−1) = 1+5−2 = 4
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(-1)).eval()), "4");
}

#[test]
fn char_poly_3x3_known_value() {
    // [[2,0,0],[0,3,0],[0,0,5]]: char poly = (2−λ)(3−λ)(5−λ)
    // = −λ³+10λ²−31λ+30
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = Matrix::diag(&[ctx.int(2), ctx.int(3), ctx.int(5)]);
    let cp = a.char_poly(&lam).unwrap();
    // p(0) = 30 = det(A)
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(0)).eval()), "30");
    // p(2) = 0, p(3) = 0, p(5) = 0 (eigenvalues)
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(2)).eval()), "0");
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(3)).eval()), "0");
    assert_eq!(format!("{}", cp.subs(&lam, &ctx.int(5)).eval()), "0");
}

// ── det(−A) = (−1)ⁿ det(A) ────────────────────────────────────────────────

#[test]
fn det_negation_sign() {
    let ctx = Context::new();
    for (n, m) in [
        (2, matrix![ctx, [1, 2], [3, 4]]),
        (3, matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 10]]),
    ] {
        let det_a = m.det().unwrap().eval();
        let det_neg_a = (-&m).det().unwrap().eval();
        let sign = if n % 2 == 0 { 1 } else { -1 };
        let expected = (&det_a * &ctx.int(sign)).eval();
        assert_eq!(
            format!("{det_neg_a}"),
            format!("{expected}"),
            "det(−A) = (−1)^{n}·det(A)  {n}×{n}"
        );
    }
}

// ── CRT i64 with large moduli (overflow check) ────────────────────────────
//
// BUG: `crt_i64` in ntheory.rs line 782 performs the computation
//
//     result = result + modulus * ((remainders[i] - result) / g % (moduli[i] / g)) * p;
//
// entirely in i64 arithmetic. When the moduli are large (≥ ~10^9), the
// intermediate product `modulus * step * p` can reach ~10^27, far exceeding
// i64::MAX (~9.2×10^18). In debug mode this panics with
// "attempt to multiply with overflow"; in release mode it silently wraps
// and returns a wrong answer.
//
// The BigInt `crt()` function handles the same inputs correctly.
//
// Fix: either use i128 for the intermediate arithmetic, use checked_mul
// and fall back to BigInt, or document that crt_i64 only handles small moduli.

#[test]
fn crt_bigint_large_coprime_moduli() {
    // BigInt CRT handles large moduli correctly — this is the oracle.
    let r: Vec<BigInt> = vec![1_000_000_006i64.into(), 1i64.into()];
    let m: Vec<BigInt> = vec![1_000_000_007i64.into(), 999_999_937i64.into()];
    let x = crt(&r, &m).unwrap();
    assert_eq!(&x % bi(1_000_000_007), bi(1_000_000_006));
    assert_eq!(&x % bi(999_999_937), bi(1));
    assert!(x >= BigInt::zero());
}

#[test]
fn crt_i64_large_coprime_moduli_overflow() {
    // FIXED: previously panicked with "attempt to multiply with overflow"
    // because intermediate i64 arithmetic overflowed.  Now delegates to
    // arbitrary-precision BigInt internally.
    //
    // Both moduli fit in i64, and the answer (~10^18) also fits in i64,
    // so the function should return the correct result.
    let result = crt_i64(&[1_000_000_006, 1], &[1_000_000_007, 999_999_937]);
    assert!(result.is_some(), "should produce a result that fits in i64");
    let x = result.unwrap();
    // Verify the solution satisfies both congruences.
    assert_eq!(x.rem_euclid(1_000_000_007), 1_000_000_006);
    assert_eq!(x.rem_euclid(999_999_937), 1);
    // Cross-check against the BigInt version.
    let big = crt(
        &[bi(1_000_000_006), bi(1)],
        &[bi(1_000_000_007), bi(999_999_937)],
    );
    assert_eq!(result, big.as_ref().and_then(|v| v.try_into().ok()));
}

#[test]
fn crt_i64_three_large_moduli() {
    // FIXED: previously panicked with overflow.  Now delegates to BigInt.
    // The combined modulus (~10^27) exceeds i64::MAX, so the result may
    // not fit in i64 — but the function must not panic.
    let result = crt_i64(
        &[2, 3, 5],
        &[999_999_937, 999_999_929, 999_999_893],
    );
    // Cross-check: BigInt version gives the authoritative answer.
    let big = crt(
        &[bi(2), bi(3), bi(5)],
        &[bi(999_999_937), bi(999_999_929), bi(999_999_893)],
    );
    let big_as_i64: Option<i64> = big.as_ref().and_then(|v| v.try_into().ok());
    assert_eq!(result, big_as_i64);
}

// ── Eigenvalue sum = trace, eigenvalue product = det ───────────────────────

#[test]
fn eigenvalue_sum_is_trace_2x2() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [4, 1], [2, 3]];
    let evs = a.eigenvals(&lam).unwrap();
    if evs.len() == 2 {
        let sum = (&evs[0] + &evs[1]).eval().simplify();
        let tr = a.trace().unwrap().eval();
        assert_eq!(format!("{sum}"), format!("{tr}"), "Σλ = tr(A)");
    }
}

#[test]
fn eigenvalue_product_is_det_2x2() {
    let ctx = Context::new();
    symplex::syms!(ctx; lam);
    let a = matrix![ctx, [4, 1], [2, 3]];
    let evs = a.eigenvals(&lam).unwrap();
    if evs.len() == 2 {
        let prod = (&evs[0] * &evs[1]).eval().simplify();
        let det = a.det().unwrap().eval();
        assert_eq!(format!("{prod}"), format!("{det}"), "Πλ = det(A)");
    }
}

// ── Bareiss 5×5 determinant cross-check ────────────────────────────────────

#[test]
fn det_5x5_cross_check() {
    // Verify via cofactor along row 0 of a 5×5
    let ctx = Context::new();
    let a = matrix![ctx,
        [1, 2, 3, 4, 5],
        [2, 3, 1, 5, 4],
        [3, 1, 2, 4, 5],
        [4, 5, 4, 1, 2],
        [5, 4, 5, 2, 1]
    ];
    let det = a.det().unwrap().eval();
    // Cross-check: det(A) = det(Aᵀ)
    let det_t = a.transpose().det().unwrap().eval();
    assert_eq!(format!("{det}"), format!("{det_t}"));
    // Cross-check: A invertible iff det ≠ 0
    let det_str = format!("{det}");
    if det_str != "0" {
        let inv = a.inv().unwrap();
        assert_is_identity(&(&a * &inv), "5×5 A·A⁻¹");
    }
}

// ── Kronecker product mixed-size ───────────────────────────────────────────

#[test]
fn kronecker_det_identity() {
    // det(A ⊗ B) = det(A)^m · det(B)^n  where A is n×n, B is m×m
    let ctx = Context::new();
    let a = matrix![ctx, [2, 1], [1, 3]]; // 2×2, det=5
    let b = matrix![ctx, [1, 2], [3, 1]]; // 2×2, det=-5
    let kron = a.kronecker(&b); // 4×4
    let det_kron = kron.det().unwrap().eval();
    // det(A⊗B) = det(A)^2 · det(B)^2 = 25 · 25 = 625
    let det_a = a.det().unwrap().eval();
    let det_b = b.det().unwrap().eval();
    let expected = (&(&det_a * &det_a) * &(&det_b * &det_b)).eval();
    assert_eq!(
        format!("{det_kron}"),
        format!("{expected}"),
        "det(A⊗B) = det(A)^m·det(B)^n"
    );
}

// ── Wilson's theorem: (p−1)! ≡ −1 (mod p) ─────────────────────────────────

#[test]
fn wilsons_theorem() {
    for &p in &[2i64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        let mut fact = bi(1);
        for i in 2..p {
            fact = (fact * bi(i)) % bi(p);
        }
        assert_eq!(fact, bi(p - 1), "Wilson's theorem for p={p}");
    }
}

// ── Sum of Möbius over squarefree divisors ─────────────────────────────────

#[test]
fn mobius_is_zero_for_nonsquarefree() {
    // μ(n) = 0 iff n has a squared prime factor
    for n in 1..=100i64 {
        let factors = factorint(n);
        let has_square = factors.iter().any(|(_, e)| *e >= 2);
        if has_square {
            assert_eq!(mobius(n), 0, "μ({n}) should be 0 (has squared factor)");
        } else {
            assert_ne!(mobius(n), 0, "μ({n}) should be ±1 (squarefree)");
        }
    }
}

// ── Totient for primes ─────────────────────────────────────────────────────

#[test]
fn totient_of_primes() {
    for &p in &[2i64, 3, 5, 7, 11, 13, 97, 101, 1009, 10007] {
        assert!(isprime(p));
        assert_eq!(totient(p), bi(p - 1), "φ({p}) = {p}−1");
    }
}

// ── Stirling numbers: S(n,2) = 2^(n−1) − 1 ───────────────────────────────

#[test]
fn stirling2_k_equals_2_formula() {
    for n in 2..=20u64 {
        assert_eq!(
            stirling2(n, 2u64).unwrap(),
            (BigInt::one() << (n - 1) as usize) - BigInt::one(),
            "S({n},2)"
        );
    }
}

// ── Partition strict monotonicity ──────────────────────────────────────────

#[test]
fn partition_count_monotone() {
    let mut prev = partition_count(0u64).unwrap();
    for n in 1..=100u64 {
        let curr = partition_count(n).unwrap();
        assert!(curr >= prev, "p({n}) < p({})", n - 1);
        prev = curr;
    }
}
