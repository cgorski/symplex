//! Matrix Algebra — eigenvalues, decompositions, and code generation.
//!
//! Demonstrates:
//! - Construction (matrix!, zeros, identity, diag, from_fn)
//! - Determinant, trace, inverse
//! - Characteristic polynomial and eigenvalues
//! - Symbolic matrices and differentiation
//! - LU decomposition
//! - RREF, rank, and nullspace
//! - Jacobian computation
//! - Matrix code generation
//! - LaTeX output
//!
//! Run with: cargo run --example matrix_algebra

use symplex::matrix::jacobian;
use symplex::prelude::*;

fn main() {
    println!("=== Matrix Algebra ===\n");

    let ctx = Context::new();
    symplex::syms!(ctx; x, y);

    // ── 1. Construction ────────────────────────────────────────────
    println!("--- Construction ---");

    let a = matrix![ctx, [2, 1], [1, 3]];
    println!("A = {a}");

    let eye = Matrix::identity(&ctx, 3);
    println!("I₃ = {eye}");

    let z = Matrix::zeros(&ctx, 2, 3);
    println!("Zeros(2×3) = {z}");

    let d = Matrix::diag(&[ctx.int(1), ctx.int(2), ctx.int(3)]);
    println!("diag(1,2,3) = {d}");

    let built = Matrix::from_fn(3, 3, |i, j| ctx.int((i * 3 + j + 1) as i64));
    println!("from_fn(3×3) = {built}");

    let row = Matrix::row_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
    println!("Row vector = {row}");

    let col = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]);
    println!("Col vector = {col}");

    // ── 2. Basic operations ────────────────────────────────────────
    println!("\n--- Basic Operations ---");

    println!("det(A) = {}", a.det().unwrap());
    println!("trace(A) = {}", a.trace().unwrap());
    println!("Aᵀ = {}", a.transpose());

    let b = matrix![ctx, [5, 6], [7, 8]];
    println!("\nB = {b}");
    println!("A + B = {}", &a + &b);
    println!("A - B = {}", &a - &b);
    println!("A × B = {}", &a * &b);
    println!("3·A = {}", &a * 3);
    println!("-A = {}", -&a);

    // Matrix power
    let a_squared = a.powi(2).unwrap();
    println!("A² = {a_squared}");

    let a_cubed = a.powi(3).unwrap();
    println!("A³ = {a_cubed}");

    // ── 3. Inverse ─────────────────────────────────────────────────
    println!("\n--- Inverse ---");

    if let Ok(inv) = a.inv() {
        println!("A⁻¹ = {inv}");

        // Verify: A · A⁻¹ should be identity
        let product = &a * &inv;
        println!("A · A⁻¹ = {product}");
    }

    // Singular matrix — no inverse
    let singular = matrix![ctx, [1, 2], [2, 4]];
    println!("\nSingular matrix: {singular}");
    println!("det = {}", singular.det().unwrap());
    match singular.inv() {
        Ok(inv) => println!("Inverse: {inv}"),
        Err(_) => println!("No inverse (singular)"),
    }

    // ── 4. Eigenvalues ─────────────────────────────────────────────
    println!("\n--- Eigenvalues ---");

    let eigenvals = a.eigenvals(&x).unwrap();
    println!(
        "Eigenvalues of A: {:?}",
        eigenvals.iter().map(|e| format!("{e}")).collect::<Vec<_>>()
    );

    // Characteristic polynomial
    let char_p = a.char_poly(&x).unwrap();
    println!("Characteristic polynomial: {char_p}");

    // Verify: eigenvalues should be roots of the char poly
    for ev in &eigenvals {
        let verified = char_p.check_solution(&x, ev) == Some(true);
        println!("  λ = {ev}: root of char poly? {verified}");
    }

    // 3×3 eigenvalues
    let c = matrix![ctx, [1, 2, 0], [0, 3, 1], [0, 0, 2]];
    println!("\nC = {c}");
    let eigenvals_c = c.eigenvals(&x).unwrap();
    println!(
        "Eigenvalues of C: {:?}",
        eigenvals_c
            .iter()
            .map(|e| format!("{e}"))
            .collect::<Vec<_>>()
    );

    // ── 5. Symbolic matrices ───────────────────────────────────────
    println!("\n--- Symbolic Matrices ---");

    let sym_m = matrix![ctx, [x, 1], [0, x]];
    println!("B(x) = {sym_m}");
    println!("det(B) = {}", sym_m.det().unwrap());
    println!("B² = {}", &sym_m * &sym_m);

    // Differentiate a symbolic matrix
    let dm = sym_m.diff(&x);
    println!("dB/dx = {dm}");

    // Substitute a value
    let m_at_3 = sym_m.subs(&x, &ctx.int(3));
    println!("B(3) = {m_at_3}");

    // Simplification
    let trig_m = Matrix::new(vec![
        vec![&x.sin().powi(2) + &x.cos().powi(2), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    println!("\nTrig matrix: {trig_m}");
    println!("Simplified:  {}", trig_m.simplify());

    // Expansion
    let expand_m = Matrix::new(vec![
        vec![(&x + 1).powi(2), ctx.int(0)],
        vec![ctx.int(0), ctx.int(1)],
    ])
    .unwrap();
    println!("Before expand: {expand_m}");
    println!("After expand:  {}", expand_m.expand());

    // ── 6. LU Decomposition ────────────────────────────────────────
    println!("\n--- LU Decomposition ---");

    let lu_mat = matrix![ctx, [2, 1, 1], [4, 3, 3], [8, 7, 9]];
    println!("M = {lu_mat}");

    if let Some((l, u, perm)) = lu_mat.lu() {
        println!("L = {l}");
        println!("U = {u}");
        println!("Permutation: {perm:?}");

        // Verify: L × U should reconstruct M (up to permutation)
        let lu_product = &l * &u;
        println!("L × U = {lu_product}");
    } else {
        println!("LU decomposition failed (singular or non-square)");
    }

    // ── 7. Cholesky Decomposition ──────────────────────────────────
    println!("\n--- Cholesky Decomposition ---");

    let spd = matrix![ctx, [4, 2], [2, 3]]; // symmetric positive definite
    println!("SPD matrix: {spd}");
    if let Ok(Some(chol)) = spd.cholesky() {
        println!("L (Cholesky) = {chol}");
        let product = &chol * &chol.transpose();
        println!("L·Lᵀ = {product}");
    }

    // ── 8. RREF, Rank, Nullspace ───────────────────────────────────
    println!("\n--- RREF, Rank, Nullspace ---");

    let rank_mat = matrix![ctx, [1, 2, 3], [4, 5, 6], [7, 8, 9]];
    println!("M = {rank_mat}");

    let (rref, pivot_cols) = rank_mat.rref();
    println!("RREF = {rref}");
    println!("Pivot columns: {pivot_cols:?}");
    println!("Rank = {}", rank_mat.rank());

    let null = rank_mat.nullspace();
    println!("Nullspace basis vectors: {}", null.len());
    for (i, v) in null.iter().enumerate() {
        println!("  v{i} = {v}");
    }

    // Full rank example
    let full_rank = matrix![ctx, [1, 0, 0], [0, 1, 0], [0, 0, 1]];
    println!("\nIdentity rank = {}", full_rank.rank());
    println!(
        "Identity nullspace: {} vectors (trivial)",
        full_rank.nullspace().len()
    );

    // Column space
    let colspace = rank_mat.columnspace();
    println!("\nColumn space of M: {} basis vectors", colspace.len());

    // ── 9. Kronecker Product ───────────────────────────────────────
    println!("\n--- Kronecker Product ---");

    let k1 = matrix![ctx, [1, 0], [0, 1]];
    let k2 = matrix![ctx, [1, 2], [3, 4]];
    let kron = k1.kronecker(&k2);
    println!("I₂ ⊗ [[1,2],[3,4]] = {kron}");

    // ── 10. Stacking ───────────────────────────────────────────────
    println!("\n--- Matrix Stacking ---");

    let top = matrix![ctx, [1, 2, 3]];
    let bottom = matrix![ctx, [4, 5, 6], [7, 8, 9]];
    let vstacked = Matrix::vstack(&[&top, &bottom]).unwrap();
    println!("vstack = {vstacked}");

    let left = matrix![ctx, [1, 2], [3, 4]];
    let right = matrix![ctx, [5], [6]];
    let hstacked = Matrix::hstack(&[&left, &right]).unwrap();
    println!("hstack = {hstacked}");

    // ── 11. Properties ─────────────────────────────────────────────
    println!("\n--- Properties ---");

    println!("A is square: {}", a.is_square());
    println!("A is symmetric: {}", a.is_symmetric());

    let non_sym = matrix![ctx, [1, 2], [3, 4]];
    println!("[[1,2],[3,4]] is symmetric: {}", non_sym.is_symmetric());

    let sym = matrix![ctx, [1, 2], [2, 1]];
    println!("[[1,2],[2,1]] is symmetric: {}", sym.is_symmetric());

    // ── 12. Jacobian Computation ───────────────────────────────────
    println!("\n--- Jacobian ---");

    // f(x,y) = [x²+y, x·y²]
    let f1 = expr!(ctx, x ^ 2 + y);
    let f2 = expr!(ctx, x * y ^ 2);

    let jac = jacobian(&[&f1, &f2], &[&x, &y]);
    println!("f = [x²+y, x·y²]");
    println!("J = {jac}");
    println!("det(J) = {}", jac.det().unwrap());

    // Evaluate Jacobian at a point
    let jac_at_1_2 = jac.subs(&x, &ctx.int(1)).subs(&y, &ctx.int(2));
    println!("J(1,2) = {jac_at_1_2}");
    println!("det(J(1,2)) = {}", jac_at_1_2.det().unwrap());

    // ── 13. Dot and Cross Products ─────────────────────────────────
    println!("\n--- Dot & Cross Products ---");

    let v1 = Matrix::col_vector(vec![ctx.int(1), ctx.int(2), ctx.int(3)]);
    let v2 = Matrix::col_vector(vec![ctx.int(4), ctx.int(5), ctx.int(6)]);

    let dot_product = symplex::matrix::dot(&v1, &v2);
    println!("v1 · v2 = {dot_product}"); // 1·4 + 2·5 + 3·6 = 32

    let cross_product = symplex::matrix::cross(&v1, &v2);
    println!("v1 × v2 = {cross_product}");

    // ── 14. Matrix Exponential (Series) ────────────────────────────
    println!("\n--- Matrix Exponential ---");

    let rot = matrix![ctx, [0, 1], [-1, 0]]; // 90° rotation generator
    let exp_rot = rot.exp_series(6).unwrap();
    println!("exp([[0,1],[-1,0]]) ≈ {exp_rot}");

    // ── 15. Code Generation ────────────────────────────────────────
    println!("\n--- Matrix Code Generation ---");

    let neg_sin_x = -&x.sin();
    let rot_mat = Matrix::new(vec![vec![x.cos(), neg_sin_x], vec![x.sin(), x.cos()]]).unwrap();
    println!("R(x) = {rot_mat}");

    let code = rot_mat
        .to_rust_fn("rotation_2d", &["x"])
        .expect("codegen failed");
    println!("Generated Rust function:");
    println!("{code}");

    // Also generate with f32 options
    use symplex::matrix::CodegenOptions;
    let opts = CodegenOptions::embedded_f32();
    let code_f32 = rot_mat
        .to_rust_fn_with_options("rotation_2d_f32", &["x"], &opts)
        .expect("codegen failed");
    println!("f32 variant:");
    println!("{code_f32}");

    // Scalar code generation for a matrix entry
    let det_symbolic = rot_mat.det().unwrap();
    println!("det(R) = {det_symbolic}");
    println!("det(R) simplified = {}", det_symbolic.simplify_trig());

    // ── 16. LaTeX Output ───────────────────────────────────────────
    println!("\n--- LaTeX ---");

    println!("A:");
    println!("  {}", a.to_latex());

    println!("Rotation matrix:");
    println!("  {}", rot_mat.to_latex());

    println!("Symbolic matrix:");
    println!("  {}", sym_m.to_latex());

    // ── 17. Pseudoinverse ──────────────────────────────────────────
    println!("\n--- Pseudoinverse ---");

    let tall = matrix![ctx, [1, 0], [0, 1], [1, 1]]; // 3×2
    println!("Tall matrix (3×2): {tall}");
    if let Ok(pinv) = tall.pinv() {
        println!("Pseudoinverse: {pinv}");
    } else {
        println!("Pseudoinverse: not computable");
    }

    println!("\n✓ Done!");
}
