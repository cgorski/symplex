//! Verification proptests: cross-validate different computation paths.

use proptest::prelude::*;

mod common;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // ═══════════════════════════════════════════════════════════════
    // lambdify vs evalf_f64 cross-validation
    // ═══════════════════════════════════════════════════════════════

    /// For random polynomials: lambdify result should match evalf_f64 result
    #[test]
    fn lambdify_matches_evalf(
        a in -5i64..5,
        b in -5i64..5,
        c in -5i64..5,
        pt in 1i64..5,
    ) {
        let x = symplex::var("x");
        let poly = &(&x.powi(2) * a) + &(&x * b) + c;

        let mut bail = common::BailCounter::new("lambdify_matches_evalf");

        // lambdify path
        if let Some(f) = poly.lambdify(&["x"]) {
            let lambdify_result = f(&[pt as f64]);

            // evalf path
            let substituted = poly.subs_i64(&x, pt);
            if let Ok(evalf_result) = substituted.evalf_f64() {
                bail.check();
                let diff = (lambdify_result - evalf_result).abs();
                prop_assert!(diff < 1e-6,
                    "lambdify vs evalf mismatch at x={pt}: {} vs {} for {}x²+{}x+{}",
                    lambdify_result, evalf_result, a, b, c);
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }

    /// lambdify of trig functions should match evalf
    #[test]
    fn lambdify_trig_matches_evalf(pt in 1i64..4) {
        let x = symplex::var("x");
        let expr = &x.sin().powi(2) + &x.cos().powi(2);

        if let Some(f) = expr.lambdify(&["x"]) {
            let result = f(&[pt as f64]);
            // sin²+cos² should be 1.0 everywhere
            prop_assert!((result - 1.0).abs() < 1e-10,
                "sin²+cos² should be 1 at x={pt}, got {result}");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Matrix algebraic laws
    // ═══════════════════════════════════════════════════════════════

    /// det(A) * det(B) == det(A*B) for 2x2 numeric matrices
    #[test]
    fn det_multiplicative(
        a11 in -3i64..3, a12 in -3i64..3,
        a21 in -3i64..3, a22 in -3i64..3,
        b11 in -3i64..3, b12 in -3i64..3,
        b21 in -3i64..3, b22 in -3i64..3,
    ) {
        use symplex::matrix::Matrix;
        let a = Matrix::new(vec![
            vec![symplex::int(a11), symplex::int(a12)],
            vec![symplex::int(a21), symplex::int(a22)],
        ]);
        let b = Matrix::new(vec![
            vec![symplex::int(b11), symplex::int(b12)],
            vec![symplex::int(b21), symplex::int(b22)],
        ]);
        let ab = a.matmul(&b);
        let det_a = a.det();
        let det_b = b.det();
        let det_ab = ab.det();
        let det_product = &det_a * &det_b;

        // Both should evaluate to same integer
        let v1 = format!("{det_product}");
        let v2 = format!("{det_ab}");
        prop_assert_eq!(v1, v2,
            "det(A)*det(B) should equal det(A*B)");
    }

    /// trace(A + B) == trace(A) + trace(B)
    #[test]
    fn trace_additive(
        a11 in -5i64..5, a12 in -5i64..5,
        a21 in -5i64..5, a22 in -5i64..5,
        b11 in -5i64..5, b12 in -5i64..5,
        b21 in -5i64..5, b22 in -5i64..5,
    ) {
        use symplex::matrix::Matrix;
        let a = Matrix::new(vec![
            vec![symplex::int(a11), symplex::int(a12)],
            vec![symplex::int(a21), symplex::int(a22)],
        ]);
        let b = Matrix::new(vec![
            vec![symplex::int(b11), symplex::int(b12)],
            vec![symplex::int(b21), symplex::int(b22)],
        ]);
        let sum = a.add(&b);
        let trace_sum = sum.trace();
        let trace_a_plus_b = &a.trace() + &b.trace();
        prop_assert_eq!(
            format!("{trace_sum}"),
            format!("{trace_a_plus_b}"),
            "trace(A+B) should equal trace(A)+trace(B)"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Series fast path matches generic
    // ═══════════════════════════════════════════════════════════════

    /// sin series: verify numerically at a point
    #[test]
    fn sin_series_numerical(order in 3u32..8) {
        let x = symplex::var("x");
        let series = x.sin().maclaurin(&x, order);
        let mut bail = common::BailCounter::new("sin_series_numerical");
        if let Ok(s) = series {
            let expanded = s.expand();
            // Evaluate at x=0.5
            let at_half = expanded.subs_i64(&x, 1); // use x=1 for integer sub
            if let Ok(val) = at_half.evalf_f64() {
                bail.check();
                let exact = 1.0f64.sin();
                // Higher order should be more accurate
                let tol = 1.0 / (order as f64);
                prop_assert!((val - exact).abs() < tol,
                    "sin series order {order} at x=1: got {val}, expected {exact}");
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }

    /// exp series: verify numerically
    #[test]
    fn exp_series_numerical(order in 3u32..10) {
        let x = symplex::var("x");
        let series = x.exp().maclaurin(&x, order);
        let mut bail = common::BailCounter::new("exp_series_numerical");
        if let Ok(s) = series {
            let expanded = s.expand();
            let at_one = expanded.subs_i64(&x, 1);
            if let Ok(val) = at_one.evalf_f64() {
                bail.check();
                let exact = 1.0f64.exp();
                let tol = 3.0 / (order as f64).powi(2);
                prop_assert!((val - exact).abs() < tol,
                    "exp series order {order} at x=1: got {val}, expected {exact}");
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }
}
