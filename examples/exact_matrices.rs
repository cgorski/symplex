//! Exact matrices over ℚ and ℤ: `QMatrix` and `ZMatrix` (symplex 0.3.5).
//!
//! Demonstrates:
//! - solving, inverting and reducing rational matrices without the
//!   expression arena, with fraction-free (Bareiss) elimination;
//! - the classic ill-conditioned case, the Hilbert matrix, handled exactly;
//! - rank, nullspace and the unique RREF of a singular matrix;
//! - ℤ-specific work: Bareiss determinant, Hermite and Smith normal forms,
//!   integer kernels, all verified by multiplying back;
//! - lossless conversion to and from `Matrix`, and the fact that `Matrix`
//!   routes rational input through the exact core on its own;
//! - a timing of `Matrix::inv` versus `QMatrix::inv` on the same data.
//!
//! Run with: `cargo run --release --example exact_matrices`

use std::time::Instant;

use symplex::linprog::q;
use symplex::prelude::*;

fn main() {
    println!("=== Exact matrices over ℚ and ℤ ===\n");

    // ── 1. Solve, invert, determinant ───────────────────────────────────
    let a = QMatrix::from_i64(&[&[2, 1, 0], &[1, 3, 1], &[0, 1, 4]]).unwrap();
    let b = QMatrix::new(vec![vec![q(1, 2)], vec![q(1, 3)], vec![q(1, 4)]]).unwrap();
    let x = a.solve(&b).unwrap();
    println!(
        "A =\n{a}\nb = {}\nx = A⁻¹b = {}",
        b.transpose(),
        x.transpose()
    );
    assert_eq!(&a * &x, b);
    println!("det A = {}", a.det().unwrap());
    let inv = a.inv().unwrap();
    println!("A⁻¹ =\n{inv}");
    assert!((&a * &inv).is_identity());

    // ── 2. The Hilbert matrix, exactly ──────────────────────────────────
    for n in [4usize, 8, 12] {
        let h = QMatrix::from_fn(n, n, |i, j| q(1, (i + j + 1) as i64));
        let det = h.det().unwrap();
        let hinv = h.inv().unwrap();
        assert!(hinv.is_integer(), "the inverse Hilbert matrix is integral");
        assert!((&h * &hinv).is_identity());
        println!(
            "H_{n}: det = 1/{}  ({} digits), H⁻¹ integral with largest entry {} digits",
            det.denom(),
            det.denom().to_string().len(),
            hinv.iter()
                .map(|v| v.numer().to_string().trim_start_matches('-').len())
                .max()
                .unwrap_or(0)
        );
    }

    // ── 3. RREF, rank, nullspace of a singular matrix ───────────────────
    let s = QMatrix::from_i64(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();
    let (r, pivots) = s.rref();
    println!(
        "\nS =\n{s}\nrref(S) =\n{r}\npivot columns {pivots:?}, rank {}",
        s.rank()
    );
    let ns = s.nullspace();
    println!("nullspace basis: {}", ns[0].transpose());
    for v in &ns {
        assert!((&s * v).is_zero());
    }
    assert_eq!(s.det().unwrap(), q(0, 1));
    println!("S⁻¹: {}", s.inv().unwrap_err());

    // ── 4. Integer matrices ─────────────────────────────────────────────
    let z = ZMatrix::from_i64(&[&[2, 4, 4], &[-6, 6, 12], &[10, -4, -16]]).unwrap();
    println!("\nZ =\n{z}\ndet Z = {}", z.det().unwrap());
    let HermiteNormalForm { h, u } = z.hermite_normal_form_with_transform();
    println!("row HNF H = U·Z:\nH =\n{h}\ndet U = {}", u.det().unwrap());
    assert_eq!(&u * &z, h);
    let SmithNormalForm {
        s: sn,
        u: us,
        v: vs,
    } = z.smith_normal_form_with_transforms();
    println!("Smith form diag {:?}", sn.diagonal());
    assert_eq!(&(&us * &z) * &vs, sn);
    let k = ZMatrix::from_i64(&[&[2, 1, 1]]).unwrap();
    let kernel = k.integer_nullspace();
    println!(
        "ℤ-kernel of {k}: {}",
        kernel
            .iter()
            .map(|v| v.transpose().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    for v in &kernel {
        assert!((&k * v).is_zero());
    }
    println!(
        "is_unimodular(U) = {}, lattice_determinant([[2,0,1],[0,3,1]]) = {}",
        u.is_unimodular(),
        ZMatrix::from_i64(&[&[2, 0, 1], &[0, 3, 1]])
            .unwrap()
            .lattice_determinant()
            .unwrap()
    );

    // ── 5. Conversions with Matrix ──────────────────────────────────────
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2], [3, 4]];
    let zm = ZMatrix::try_from(&m).unwrap();
    println!(
        "\nMatrix → {zm:?} → Matrix: {}",
        zm.to_qmatrix().inv().unwrap().to_matrix(&ctx)
    );
    let sym = Matrix::new(vec![vec![ctx.symbol("x"), ctx.int(1)]]).unwrap();
    println!(
        "symbolic entries are refused: {}",
        QMatrix::try_from(&sym).unwrap_err()
    );
    // `Matrix` methods use the exact core by themselves on rational input.
    assert_eq!(
        m.inv().unwrap(),
        zm.to_qmatrix().inv().unwrap().to_matrix(&ctx)
    );

    // ── 6. Timing: the same inverse through Matrix and through QMatrix ──
    let n = 30;
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) % 19) as i64 - 9
    };
    let qm = QMatrix::from_fn(n, n, |_, _| q(next(), 1));
    let mm = qm.to_matrix(&ctx);
    let t0 = Instant::now();
    let inv_q = qm.inv().unwrap();
    let dt_q = t0.elapsed();
    let t0 = Instant::now();
    let inv_m = mm.inv().unwrap();
    let dt_m = t0.elapsed();
    assert_eq!(QMatrix::try_from(&inv_m).unwrap(), inv_q);
    println!(
        "\n{n}×{n} random integer matrix: QMatrix::inv {:.2?}, Matrix::inv (routed through QMatrix, plus arena round trip) {:.2?}",
        dt_q, dt_m
    );
}
