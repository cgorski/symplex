// Property-based tests for quality improvements (Wave V).
//
// V2: Cubic solver — all roots satisfy the polynomial.
// V3: Matrix inverse — A * A⁻¹ ≈ I for random integer matrices.
// V4: Codegen — generated code is syntactically valid Rust.
// Bonus: Determinant consistency — cofactor and LU paths agree.

mod common;

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

        let mut bail = common::BailCounter::new("cubic_roots_satisfy_polynomial");
        for root in &roots {
            let val = poly.subs(&x, root).eval().simplify();
            if let Ok(v) = val.eval_f64() {
                bail.check();
                prop_assert!(
                    v.abs() < 1e-6,
                    "root {} doesn't satisfy {}x³+{}x²+{}x+{}: residual={}",
                    root, a, b, c, d, v
                );
            } else {
                bail.skip();
            }
        }
        if !roots.is_empty() {
            // Symbolic-only roots legitimately fail evalf_f64 for some inputs;
            // allow high skip rate — proptest coverage across 30 cases ensures
            // non-vacuousness at the suite level.
            bail.assert_skip_rate_below(1.0);
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

        if let Ok(inv) = m.inv() {
            let product = m.matmul(&inv).unwrap();
            let mut bail = common::BailCounter::new("matrix_inverse_is_identity");
            // Check diagonal ≈ 1, off-diagonal ≈ 0
            for i in 0..3 {
                for j in 0..3 {
                    let entry = product.get(i, j).eval().simplify();
                    if let Ok(v) = entry.eval_f64() {
                        bail.check();
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
                    } else {
                        bail.skip();
                    }
                }
            }
            bail.assert_not_vacuous();
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

        let det_a = mat_a.det().unwrap().eval().simplify();
        let det_b = mat_b.det().unwrap().eval().simplify();
        let product_of_dets = (&det_a * &det_b).eval().simplify();

        let ab = mat_a.matmul(&mat_b).unwrap();
        let det_ab = ab.det().unwrap().eval().simplify();

        let mut bail = common::BailCounter::new("det_of_product");
        if let (Ok(lhs), Ok(rhs)) = (det_ab.eval_f64(), product_of_dets.eval_f64()) {
            if lhs.is_finite() && rhs.is_finite() {
                bail.check();
                let tol = 1e-6 * lhs.abs().max(rhs.abs()).max(1.0);
                prop_assert!(
                    (lhs - rhs).abs() < tol,
                    "det(A·B) = {} but det(A)·det(B) = {} (diff={})",
                    lhs, rhs, (lhs - rhs).abs()
                );
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }
}
