//! Matrix decompositions and the 0.2 linear-algebra API.
//!
//! Demonstrates:
//! - `qr`, `cholesky`, `ldl`, `lu`, `gram_schmidt`;
//! - the eigen family **without a dummy variable**: `eigenvals`,
//!   `eigenvals_with_multiplicity`, `eigenvects`, `diagonalize`,
//!   `jordan_form`, `is_diagonalizable` (`Option<bool>`);
//! - `RootOf` eigenvalues for an irreducible cubic (exact, numerically
//!   evaluable, no Cardano swell);
//! - `matrix_exp` / `matrix_exp_t`, `matrix_pow_symbolic`, `matrix_sqrt`;
//! - structure tests (`is_symmetric`, `is_orthogonal`, `is_hermitian`,
//!   `is_positive_definite`, `is_nilpotent`, …) and norms;
//! - `hessian`, `wronskian`, `solve_least_squares`, `rowspace`,
//!   `left_nullspace`;
//! - ergonomics: `Index`/`IndexMut`, operators with scalars on both sides,
//!   `block_diag`, `hadamard`, `minor`/`minor_matrix`, `eval_f64`.
//!
//! Run with: `cargo run --example matrix_decompositions`

use symplex::matrix_decomp::{gram_schmidt, hessian, wronskian};
use symplex::prelude::*;

fn main() {
    println!("=== Matrix Decompositions ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y, t, n, theta);

    // ── 1. Cholesky / LDLᵀ / LU ─────────────────────────────────────────
    println!("--- Cholesky, LDLᵀ, LU ---");
    let spd = matrix![ctx, [4, 12, -16], [12, 37, -43], [-16, -43, 98]];
    println!("A = {spd}");
    println!(
        "symmetric: {:?}   positive definite: {:?}",
        spd.is_symmetric(),
        spd.is_positive_definite()
    );
    let l = spd.cholesky().unwrap();
    println!("Cholesky L = {l}");
    println!(
        "L·Lᵀ == A: {:?}",
        l.matmul(&l.transpose()).unwrap().equals(&spd)
    );
    let Ldl { l, d } = spd.ldl().unwrap();
    println!("LDLᵀ: L = {l}  D = {d}");
    // Preconditions are errors, not garbage.
    match matrix![ctx, [1, 2], [3, 4]].cholesky() {
        Err(e) => println!("cholesky of a non-symmetric matrix → Err: {e}"),
        Ok(m) => println!("unexpected {m}"),
    }
    let Lu { l, u, perm } = matrix![ctx, [2, 1], [4, 3]].lu().unwrap();
    println!("LU of [[2,1],[4,3]]: L = {l}  U = {u}  permutation = {perm:?}");

    // ── 2. QR and Gram–Schmidt ──────────────────────────────────────────
    println!("\n--- QR / Gram–Schmidt (exact radicals) ---");
    let m = matrix![ctx, [1, 1, 0], [1, 0, 1], [0, 1, 1]];
    let Qr { q, r } = m.qr().unwrap();
    println!("M = {m}");
    println!("Q = {q}");
    println!("R = {r}");
    println!(
        "Q orthogonal: {:?}    Q·R == M: {:?}",
        q.is_orthogonal(),
        q.matmul(&r).unwrap().simplify().equals(&m)
    );
    let v1 = Matrix::col_vector(vec![ctx.int(1), ctx.int(1), ctx.int(0)]);
    let v2 = Matrix::col_vector(vec![ctx.int(1), ctx.int(0), ctx.int(1)]);
    let basis = gram_schmidt(&[v1, v2], true).unwrap();
    println!("Gram–Schmidt of (1,1,0), (1,0,1):");
    for b in &basis {
        println!("  {}", b.transpose());
    }

    // ── 3. Eigen family (no dummy variable in 0.2) ──────────────────────
    println!("\n--- Eigenvalues / eigenvectors / diagonalization ---");
    let s = matrix![ctx, [2, 1], [1, 2]];
    println!("S = {s}");
    let ev: Vec<String> = s
        .eigenvals()
        .unwrap()
        .iter()
        .map(|e| e.to_string())
        .collect();
    println!("eigenvals(S) = [{}]", ev.join(", "));
    for (val, mult, vecs) in s.eigenvects().unwrap() {
        println!(
            "  λ = {val} (multiplicity {mult}): eigenvector {}",
            vecs[0].transpose()
        );
    }
    let Diagonalization { p, d } = s.diagonalize().unwrap();
    println!("S = P·D·P⁻¹ with D = {d}");
    println!(
        "check: {:?}",
        p.matmul(&d)
            .unwrap()
            .matmul(&p.inv().unwrap())
            .unwrap()
            .equals(&s)
    );

    let j = matrix![
        ctx,
        [5, 4, 2, 1],
        [0, 1, -1, -1],
        [-1, -1, 3, 0],
        [1, 1, -1, 2]
    ];
    let with_mult: Vec<String> = j
        .eigenvals_with_multiplicity()
        .unwrap()
        .iter()
        .map(|(v, m)| format!("{v} (×{m})"))
        .collect();
    println!("\nJ4 = {j}");
    println!("eigenvalues with multiplicity: {}", with_mult.join(", "));
    println!("diagonalizable: {:?}", j.is_diagonalizable());
    let jordan = j.jordan_form().unwrap().j;
    println!("Jordan normal form = {jordan}");

    // ── 4. RootOf eigenvalues ───────────────────────────────────────────
    println!("\n--- RootOf eigenvalues for an irreducible cubic ---");
    let c = matrix![ctx, [0, 1, 0], [0, 0, 1], [1, 1, 0]];
    println!("C = {c}    char poly: {}", c.char_poly(&x).unwrap());
    for ev in c.eigenvals().unwrap() {
        let Complex64 { re, im } = ev.eval_complex64().unwrap();
        println!(
            "  {ev}  ≈  {re:.6} {} {:.6}i",
            if im < 0.0 { "−" } else { "+" },
            im.abs()
        );
    }

    // ── 5. Matrix functions ─────────────────────────────────────────────
    println!("\n--- Matrix functions ---");
    let rot = matrix![ctx, [0, -1], [1, 0]];
    println!("exp([[0,−1],[1,0]])   = {}", rot.matrix_exp().unwrap());
    println!("exp(t·[[0,−1],[1,0]]) = {}", rot.matrix_exp_t(&t).unwrap());
    let defective = matrix![ctx, [2, 1], [0, 2]];
    println!(
        "exp(t·[[2,1],[0,2]])  = {}",
        defective.matrix_exp_t(&t).unwrap()
    );
    println!(
        "S^n                   = {}",
        s.matrix_pow_symbolic(&n).unwrap()
    );
    println!("√S                    = {}", s.matrix_sqrt().unwrap());
    println!(
        "√[[4,0],[0,9]]        = {}",
        matrix![ctx, [4, 0], [0, 9]].matrix_sqrt().unwrap()
    );

    // ── 6. Structure tests and norms ────────────────────────────────────
    println!("\n--- Structure tests (Option<bool>) and norms ---");
    let rs = Matrix::new(vec![
        vec![theta.cos(), -theta.sin()],
        vec![theta.sin(), theta.cos()],
    ])
    .unwrap();
    println!(
        "rotation R(θ): orthogonal {:?}, det = {}",
        rs.is_orthogonal(),
        rs.det().unwrap().simplify()
    );
    let ev: Vec<String> = rs
        .eigenvals()
        .unwrap()
        .iter()
        .map(|e| e.to_string())
        .collect();
    println!("eigenvals(R(θ)) = [{}]", ev.join(", "));
    let i = ctx.i_unit();
    let h = Matrix::new(vec![
        vec![ctx.int(2), &ctx.int(1) + &i],
        vec![&ctx.int(1) - &i, ctx.int(3)],
    ])
    .unwrap();
    println!(
        "H = {h}  hermitian: {:?},  adjoint = {}",
        h.is_hermitian(),
        h.adjoint()
    );
    println!(
        "[[0,1],[0,0]] nilpotent: {:?}    [[0,−1],[1,0]] unitary: {:?}",
        matrix![ctx, [0, 1], [0, 0]].is_nilpotent(),
        rot.is_unitary()
    );
    let m2 = matrix![ctx, [1, -2], [3, 4]];
    println!(
        "norms of {m2}: ‖·‖₁ = {}, ‖·‖∞ = {}, ‖·‖_F = {}",
        m2.norm_1(),
        m2.norm_inf(),
        m2.norm_frobenius()
    );
    println!(
        "‖(3, 4)‖₃ = {}",
        Matrix::col_vector(vec![ctx.int(3), ctx.int(4)])
            .norm_p(&ctx.int(3))
            .unwrap()
    );

    // ── 7. Calculus helpers and subspaces ───────────────────────────────
    println!("\n--- hessian / wronskian / least squares / subspaces ---");
    let f = &x.powi(3) * &y + &x * &y.powi(2);
    println!("hessian of {f}: {}", hessian(&f, &[&x, &y]));
    println!(
        "W(sin x, cos x) = {},   W(eˣ, e²ˣ) = {}",
        wronskian(&[&x.sin(), &x.cos()], &x).simplify(),
        wronskian(&[&x.exp(), &(&x * 2).exp()], &x).simplify()
    );
    let a = matrix![ctx, [1, 1], [1, 2], [1, 3]];
    let b = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(2)]);
    println!(
        "least-squares fit of (1,1),(2,2),(3,2): {}",
        a.solve_least_squares(&b).unwrap().transpose()
    );
    let rank1 = matrix![ctx, [1, 2], [2, 4]];
    println!(
        "rank-1 {rank1}: rank {}, rowspace {}, left nullspace {}",
        rank1.rank(),
        rank1.rowspace()[0],
        rank1.left_nullspace()[0].transpose()
    );

    // ── 8. Ergonomics ───────────────────────────────────────────────────
    println!("\n--- Ergonomics ---");
    let mut m = matrix![ctx, [1, 2], [3, 4]];
    m[(0, 1)] = ctx.int(7); // IndexMut
    println!("after m[(0,1)] = 7:  m = {m}   m[(1,0)] = {}", m[(1, 0)]);
    println!("2·m = {}   m/2 = {}", 2 * &m, m.clone() / 2);
    println!(
        "m + m = {}   m·m = {}   −m = {}",
        &m + &m,
        &m * &m,
        -m.clone()
    );
    println!(
        "block_diag(m, I₁) = {}",
        Matrix::block_diag(&[&m, &Matrix::identity(&ctx, 1)]).unwrap()
    );
    println!("m ∘ m (Hadamard) = {}", m.hadamard(&m).unwrap());
    println!(
        "minor(0,0) = {},  minor_matrix(0,0) = {}",
        m.minor(0, 0).unwrap(),
        m.minor_matrix(0, 0).unwrap()
    );
    println!("eval_f64 = {:?}", m.eval_f64().unwrap());
    println!(
        "ragged rows are rejected: {}",
        Matrix::try_from(vec![vec![ctx.int(1)], vec![ctx.int(2), ctx.int(3)]]).is_err()
    );

    println!("\n✓ Done!");
}
