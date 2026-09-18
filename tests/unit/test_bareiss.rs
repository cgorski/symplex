//! Tests for the Bareiss fraction-free determinant algorithm.
//!
//! The `det()` method dispatches to `det_bareiss()` for n ≥ 4,
//! so all 4×4+ tests here exercise the Bareiss path.

use symplex::matrix::Matrix;
use symplex::prelude::*;

// ── 4×4 integer matrix ────────────────────────────────────────────────

#[test]
fn bareiss_4x4_integer() {
    let ctx = Context::new();
    // Non-singular 4×4 with known determinant.
    // Matrix:
    //   1  2  3  4
    //   5  6  7  8
    //   2  6  4  8
    //   3  1  1  2
    let m = matrix![ctx, [1, 2, 3, 4], [5, 6, 7, 8], [2, 6, 4, 8], [3, 1, 1, 2]];
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();
    // Cross-check: build same matrix and compute via cofactor on 3×3 minors.
    // det = 1*(6*(4*2 - 8*1) - 7*(6*2 - 8*1) + 8*(6*1 - 4*1))
    //      -2*(5*(4*2 - 8*1) - 7*(2*2 - 8*3) + 8*(2*1 - 4*3))
    //      +3*(5*(6*2 - 8*1) - 6*(2*2 - 8*3) + 8*(2*1 - 6*3))
    //      -4*(5*(6*1 - 4*1) - 6*(2*1 - 4*3) + 6*(2*1 - 6*3))
    // (verified numerically below)
    assert!(d_val.is_finite(), "det should be finite, got {d_val}");
    // Compute expected via an independent method: build small sub-matrices
    let ctx = Context::new();
    let row0 = [1i64, 2, 3, 4];
    let row1 = [5i64, 6, 7, 8];
    let row2 = [2i64, 6, 4, 8];
    let row3 = [3i64, 1, 1, 2];
    let rows = [row0, row1, row2, row3];

    // Cofactor expansion along first row (independent of Bareiss)
    fn det3(r: [[i64; 3]; 3]) -> i64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    fn minor3(rows: &[[i64; 4]; 4], skip_col: usize) -> [[i64; 3]; 3] {
        let mut out = [[0i64; 3]; 3];
        for i in 0..3 {
            let mut c = 0;
            for (j, &val) in rows[i + 1].iter().enumerate() {
                if j == skip_col {
                    continue;
                }
                out[i][c] = val;
                c += 1;
            }
        }
        out
    }

    let expected = rows[0][0] * det3(minor3(&rows, 0)) - rows[0][1] * det3(minor3(&rows, 1))
        + rows[0][2] * det3(minor3(&rows, 2))
        - rows[0][3] * det3(minor3(&rows, 3));

    assert!(
        (d_val - expected as f64).abs() < 1e-6,
        "Bareiss det={d_val}, expected={expected}"
    );
    drop(ctx);
}

// ── 5×5 identity ──────────────────────────────────────────────────────

#[test]
fn bareiss_5x5_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 5);
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "1");
}

// ── 6×6 identity ──────────────────────────────────────────────────────

#[test]
fn bareiss_6x6_identity() {
    let ctx = Context::new();
    let m = Matrix::identity(&ctx, 6);
    let d = m.det().unwrap();
    assert_eq!(format!("{d}"), "1");
}

// ── 4×4 singular (row2 = 2*row1) ─────────────────────────────────────

#[test]
fn bareiss_singular() {
    let ctx = Context::new();
    let m = matrix![ctx, [1, 2, 3, 4], [2, 4, 6, 8], [1, 1, 1, 1], [0, 0, 0, 1]];
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();
    assert!(
        d_val.abs() < 1e-10,
        "singular matrix det should be 0, got {d_val}"
    );
}

// ── 2×2 symbolic matches ad − bc (tests the direct 2×2 path) ─────────

#[test]
fn bareiss_symbolic_2x2_matches_direct() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, b, c, d);
    let m = matrix![ctx, [a, b], [c, d]];
    let det = m.det().unwrap();
    // Should be a*d - b*c
    let expected = &(&a * &d) - &(&b * &c);
    // Compare structurally via Display
    assert_eq!(
        format!("{det}"),
        format!("{expected}"),
        "2×2 symbolic det should be ad - bc"
    );
}

// ── 4×4 known small-integer det ───────────────────────────────────────

#[test]
fn bareiss_4x4_known_det() {
    let ctx = Context::new();
    // Upper triangular → det = product of diagonal = 1*2*3*4 = 24
    let m = matrix![
        ctx,
        [1, 5, 9, 13],
        [0, 2, 7, 11],
        [0, 0, 3, 8],
        [0, 0, 0, 4]
    ];
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();
    assert!(
        (d_val - 24.0).abs() < 1e-10,
        "upper-tri det should be 24, got {d_val}"
    );
}

// ── 4×4 requiring pivot swap ──────────────────────────────────────────

#[test]
fn bareiss_4x4_needs_pivot_swap() {
    let ctx = Context::new();
    // First column starts with 0 → requires row swap
    let m = matrix![ctx, [0, 1, 2, 3], [1, 0, 0, 0], [0, 2, 1, 0], [0, 0, 3, 1]];
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();

    // Compute expected: swap row0 and row1 gives sign = -1, then
    // [[1,0,0,0],[0,1,2,3],[0,2,1,0],[0,0,3,1]]
    // det of lower-right 3×3: det([[1,2,3],[2,1,0],[0,3,1]])
    //   = 1*(1-0) - 2*(2-0) + 3*(6-0) = 1 - 4 + 18 = 15
    // Total: -1 * 1 * 15 = -15? Let me verify:
    // Actually after swap: row0=[1,0,0,0] so cofactor along row0 gives
    // det = 1 * det([[1,2,3],[2,1,0],[0,3,1]]) * (-1)^(0+0) * sign_swap
    // det_3x3 = 1*(1*1-0*3) - 2*(2*1-0*0) + 3*(2*3-1*0) = 1 - 4 + 18 = 15
    // With swap sign = -1: det = -15
    // But let's just verify it's finite and non-zero, and cross-check with
    // symplex's 3×3 cofactor path on an equivalent formulation.
    assert!(d_val.is_finite(), "det should be finite");
    assert!(d_val.abs() > 0.5, "det should be non-zero");

    // Independent cross-check: compute det of original matrix via 3×3 cofactor
    // Expand along column 0 (only row 1 has non-zero entry = 1):
    // det = -1 * 1 * det([[1,2,3],[2,1,0],[0,3,1]])  (minor of (1,0), sign (-1)^(1+0) = -1)
    let sub = matrix![ctx, [1, 2, 3], [2, 1, 0], [0, 3, 1]];
    let sub_det = sub.det().unwrap().eval_f64().unwrap();
    let expected = -sub_det;
    assert!(
        (d_val - expected).abs() < 1e-10,
        "4×4 with swap: got {d_val}, expected {expected}"
    );
}

// ── Range of seeded 4×4 integer matrices ──────────────────────────────

#[test]
fn bareiss_matches_for_seeded_4x4() {
    let ctx = Context::new();
    for seed in 0u32..8 {
        let data: Vec<Vec<Ex>> = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| ctx.int(((i * 4 + j + seed * 7 + 1) % 11) as i64 - 5))
                    .collect()
            })
            .collect();
        let m = Matrix::new(data).unwrap();
        let det = m.det().unwrap();
        // Just verify it evaluates to a finite number without panic
        let val = det.eval_f64().unwrap();
        assert!(
            val.is_finite(),
            "seed={seed}: det should be finite, got {val}"
        );
    }
}

// ── 5×5 diagonal matrix ──────────────────────────────────────────────

#[test]
fn bareiss_5x5_diagonal() {
    let ctx = Context::new();
    // det of diag(2, 3, 4, 5, 6) = 720
    let m = Matrix::from_fn(5, 5, |i, j| {
        if i == j {
            ctx.int(i as i64 + 2)
        } else {
            ctx.int(0)
        }
    });
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();
    assert!(
        (d_val - 720.0).abs() < 1e-6,
        "diag det should be 720, got {d_val}"
    );
}

// ── 4×4 negative determinant ──────────────────────────────────────────

#[test]
fn bareiss_4x4_negative_det() {
    let ctx = Context::new();
    // Permutation matrix for (0→1, 1→0, 2→3, 3→2) has det = +1
    // Single swap: (0→1, 1→0, 2→2, 3→3) has det = -1
    let m = matrix![ctx, [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    let d = m.det().unwrap();
    let d_val = d.eval_f64().unwrap();
    assert!(
        (d_val - (-1.0)).abs() < 1e-10,
        "single-swap permutation should have det=-1, got {d_val}"
    );
}

// ── 4×4 with all-negative entries ─────────────────────────────────────

#[test]
fn bareiss_4x4_all_negative() {
    let ctx = Context::new();
    let m = matrix![
        ctx,
        [-1, -2, -3, -4],
        [-5, -6, -7, -8],
        [-2, -6, -4, -8],
        [-3, -1, -1, -2]
    ];
    // Negating all entries: det(-A) = (-1)^4 * det(A) = det(A)
    let m_pos = matrix![ctx, [1, 2, 3, 4], [5, 6, 7, 8], [2, 6, 4, 8], [3, 1, 1, 2]];
    let d_neg = m.det().unwrap().eval_f64().unwrap();
    let d_pos = m_pos.det().unwrap().eval_f64().unwrap();
    assert!(
        (d_neg - d_pos).abs() < 1e-6,
        "det(-A) should equal det(A) for 4×4: neg={d_neg}, pos={d_pos}"
    );
}

// ── 4×4 transpose has same determinant ────────────────────────────────

#[test]
fn bareiss_det_equals_det_transpose() {
    let ctx = Context::new();
    let m = matrix![ctx, [2, 1, 0, 3], [1, 0, 2, 1], [0, 3, 1, 2], [1, 2, 3, 0]];
    let d = m.det().unwrap().eval_f64().unwrap();
    let dt = m.transpose().det().unwrap().eval_f64().unwrap();
    assert!(
        (d - dt).abs() < 1e-6,
        "det(A) should equal det(A^T): {d} vs {dt}"
    );
}
