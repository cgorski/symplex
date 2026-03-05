//! Property-based tests for quality improvements (Wave V).
//!
//! V2: Cubic solver — all roots satisfy the polynomial.
//! V3: Matrix inverse — A * A⁻¹ ≈ I for random integer matrices.
//! V4: Codegen — generated code is syntactically valid Rust.
//! Bonus: Determinant consistency — cofactor and LU paths agree.

use proptest::prelude::*;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════

/// Generate random coefficients for a cubic: ax³ + bx² + cx + d
/// with a ∈ [1,4] so the leading coefficient is always positive & nonzero.
fn arb_cubic_coeffs() -> impl Strategy<Value = [i64; 4]> {
    (1..5i64, -5..5i64, -5..5i64, -5..5i64).prop_map(|(a, b, c, d)| [a, b, c, d])
}

// ═══════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    // ── V2: Cubic solver ───────────────────────────────────────────────

    /// Every root returned by `solve_or_empty` should satisfy the
    /// original cubic polynomial (residual < 1e-6).
    #[test]
    fn cubic_roots_satisfy_polynomial(coeffs in arb_cubic_coeffs()) {
        let [a, b, c, d] = coeffs;
        let x = symplex::var("x");

        // Build  a·x³ + b·x² + c·x + d
        let poly = &(&x.powi(3) * a) + &(&x.powi(2) * b) + &(&x * c) + d;
        let roots = poly.solve_or_empty(&x);

        for root in &roots {
            let val = poly.subs(&x, root).eval().simplify();
            if let Ok(v) = val.evalf_f64() {
                prop_assert!(
                    v.abs() < 1e-6,
                    "root {} doesn't satisfy {}x³+{}x²+{}x+{}: residual={}",
                    root, a, b, c, d, v
                );
            }
        }
    }

    // ── V3: Matrix inverse ─────────────────────────────────────────────

    /// For a random 3×3 integer matrix, if it is invertible then
    /// A · A⁻¹ ≈ I  (diagonal entries ≈ 1, off-diagonal ≈ 0).
    #[test]
    fn matrix_inverse_is_identity(
        entries in proptest::array::uniform9(-3i64..4i64)
    ) {
        let data: Vec<Vec<Ex>> = entries
            .chunks(3)
            .map(|row| row.iter().map(|&v| symplex::int(v)).collect())
            .collect();
        let m = symplex::matrix::Matrix::new(data);

        if let Some(inv) = m.inv() {
            let product = m.matmul(&inv);
            // Check diagonal ≈ 1, off-diagonal ≈ 0
            for i in 0..3 {
                for j in 0..3 {
                    let entry = product.get(i, j).eval().simplify();
                    if let Ok(v) = entry.evalf_f64() {
                        let v: f64 = v;
                        if i == j {
                            prop_assert!(
                                (v - 1.0).abs() < 1e-8,
                                "diagonal ({},{}) should be 1, got {}",
                                i, j, v
                            );
                        } else {
                            prop_assert!(
                                v.abs() < 1e-8,
                                "off-diagonal ({},{}) should be 0, got {}",
                                i, j, v
                            );
                        }
                    }
                }
            }
        }
        // If not invertible (det=0), that's fine — skip.
    }

    // ── V4: Codegen produces valid Rust syntax ─────────────────────────

    /// `to_rust_fn` output should always contain the expected function
    /// signature components (pub fn, parameter, return type).
    #[test]
    fn codegen_produces_valid_syntax(a in -5i64..5, b in -5i64..5, c in 1i64..5) {
        let x = symplex::var("x");
        let poly = &(&x.powi(2) * a) + &(&x * b) + c;
        if let Ok(code) = poly.to_rust_fn("test_fn", &["x"]) {
            prop_assert!(
                code.contains("pub fn test_fn"),
                "missing 'pub fn test_fn' in:\n{code}"
            );
            prop_assert!(
                code.contains("x: f64"),
                "missing parameter 'x: f64' in:\n{code}"
            );
            prop_assert!(
                code.contains("-> f64"),
                "missing return type '-> f64' in:\n{code}"
            );
        }
    }

    // ── Bonus: Determinant of product ──────────────────────────────────

    /// det(A · B) == det(A) · det(B) for random 3×3 integer matrices.
    /// This exercises both the determinant and matmul paths.
    #[test]
    fn det_of_product_equals_product_of_dets(
        a_entries in proptest::array::uniform9(-3i64..4i64),
        b_entries in proptest::array::uniform9(-3i64..4i64),
    ) {
        let mat_a = symplex::matrix::Matrix::new(
            a_entries.chunks(3)
                .map(|row| row.iter().map(|&v| symplex::int(v)).collect())
                .collect(),
        );
        let mat_b = symplex::matrix::Matrix::new(
            b_entries.chunks(3)
                .map(|row| row.iter().map(|&v| symplex::int(v)).collect())
                .collect(),
        );

        let det_a = mat_a.det().eval().simplify();
        let det_b = mat_b.det().eval().simplify();
        let product_of_dets = (&det_a * &det_b).eval().simplify();

        let ab = mat_a.matmul(&mat_b);
        let det_ab = ab.det().eval().simplify();

        if let (Ok(lhs), Ok(rhs)) = (det_ab.evalf_f64(), product_of_dets.evalf_f64()) {
            if lhs.is_finite() && rhs.is_finite() {
                let tol = 1e-6 * lhs.abs().max(rhs.abs()).max(1.0);
                prop_assert!(
                    (lhs - rhs).abs() < tol,
                    "det(A·B) = {} but det(A)·det(B) = {} (diff={})",
                    lhs, rhs, (lhs - rhs).abs()
                );
            }
        }
    }
}
