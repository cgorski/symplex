//! Stress testing: run all operations on random expressions.
//!
//! Uses proptest to generate random expression trees, then applies
//! every public operation inside catch_unwind to detect panics.
//! Also verifies integration correctness via differentiation roundtrip.

use proptest::prelude::*;
use symplex::prelude::*;

/// Generate a random expression using the global context.
fn arb_expr(depth: u32) -> impl Strategy<Value = Ex> {
    let leaf = prop_oneof![
        (-10i64..10).prop_map(|n| symplex::int(n)),
        Just(symplex::var("x")),
        Just(symplex::var("y")),
    ];

    leaf.prop_recursive(depth, 64, 3, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| &a + &b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| &a * &b),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| &a - &b),
            inner.clone().prop_map(|a| -&a),
            inner.clone().prop_map(|a| a.sin()),
            inner.clone().prop_map(|a| a.cos()),
            inner.clone().prop_map(|a| a.powi(2)),
        ]
    })
}

/// Generate a polynomial expression in x with small coefficients.
#[allow(dead_code)]
fn arb_poly() -> impl Strategy<Value = Ex> {
    prop::collection::vec(-5i64..5, 1..5).prop_map(|coeffs| {
        let x = symplex::var("x");
        let mut result = symplex::int(0);
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                result = &result + &(&x.powi(i as i64) * c);
            }
        }
        result
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // ═══════════════════════════════════════════════════════════════
    // Operation stress tests: no panics
    // ═══════════════════════════════════════════════════════════════

    /// Display never panics
    #[test]
    fn stress_display(e in arb_expr(3)) {
        let _ = format!("{e}");
        let _ = format!("{e:?}");
    }

    /// Clone + equality never panics
    #[test]
    fn stress_clone_eq(e in arb_expr(3)) {
        let e2 = e.clone();
        let _ = e == e2;
    }

    /// eval never panics
    #[test]
    fn stress_eval(e in arb_expr(3)) {
        let _ = e.eval();
    }

    /// expand never panics
    #[test]
    fn stress_expand(e in arb_expr(2)) {
        let _ = e.expand();
    }

    /// simplify never panics
    #[test]
    fn stress_simplify(e in arb_expr(2)) {
        let _ = e.simplify();
    }

    /// full_simplify never panics
    #[test]
    fn stress_full_simplify(e in arb_expr(2)) {
        let _ = e.full_simplify();
    }

    /// smart_simplify never panics
    #[test]
    fn stress_smart_simplify(e in arb_expr(2)) {
        let _ = e.smart_simplify();
    }

    /// diff never panics
    #[test]
    fn stress_diff(e in arb_expr(3)) {
        let x = symplex::var("x");
        let _ = e.diff(&x);
    }

    /// integrate never panics (may return unevaluated Integral)
    #[test]
    fn stress_integrate(e in arb_expr(2)) {
        let x = symplex::var("x");
        let _ = e.integrate(&x);
    }

    /// solve never panics (may return empty)
    #[test]
    fn stress_solve(e in arb_expr(2)) {
        let x = symplex::var("x");
        let _ = e.solve_or_empty(&x);
    }

    /// subs never panics
    #[test]
    fn stress_subs(e in arb_expr(2)) {
        let x = symplex::var("x");
        let _ = e.subs_i64(&x, 3);
    }

    /// count_ops never panics
    #[test]
    fn stress_count_ops(e in arb_expr(3)) {
        let _ = e.count_ops();
    }

    /// args never panics
    #[test]
    fn stress_args(e in arb_expr(3)) {
        let _ = e.args();
    }

    /// expr_type never panics
    #[test]
    fn stress_expr_type(e in arb_expr(3)) {
        let _ = e.expr_type();
    }

    /// free_symbols never panics
    #[test]
    fn stress_free_symbols(e in arb_expr(3)) {
        let _ = e.free_symbols();
    }

    /// contains never panics
    #[test]
    fn stress_contains(e in arb_expr(3)) {
        let x = symplex::var("x");
        let _ = e.contains(&x);
    }

    /// term_count never panics
    #[test]
    fn stress_term_count(e in arb_expr(3)) {
        let _ = e.term_count();
    }

    /// to_tree / to_json never panics
    #[test]
    fn stress_serde(e in arb_expr(2)) {
        let tree = e.to_tree();
        let json = e.to_json();
        prop_assert!(!json.is_empty());
        // Round-trip
        let ctx = symplex::default_context();
        let back = ctx.from_tree(&tree);
        prop_assert_eq!(format!("{e}"), format!("{back}"));
    }

    /// expand_trig never panics
    #[test]
    fn stress_expand_trig(e in arb_expr(2)) {
        let _ = e.expand_trig();
    }

    /// expand_log never panics
    #[test]
    fn stress_expand_log(e in arb_expr(2)) {
        let _ = e.expand_log();
    }

    /// logcombine never panics
    #[test]
    fn stress_logcombine(e in arb_expr(2)) {
        let _ = e.logcombine();
    }

    /// trig_combine never panics
    #[test]
    fn stress_trig_combine(e in arb_expr(2)) {
        let _ = e.trig_combine();
    }

    /// factor_terms never panics
    #[test]
    fn stress_factor_terms(e in arb_expr(2)) {
        let _ = e.factor_terms();
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration verification
    // ═══════════════════════════════════════════════════════════════

    /// When integrate returns a non-Integral result for a polynomial,
    /// differentiating should give back the original.
    #[test]
    fn integration_roundtrip_poly(coeffs in prop::collection::vec(-5i64..5, 1..4)) {
        let x = symplex::var("x");
        let mut poly = symplex::int(0);
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                poly = &poly + &(&x.powi(i as i64) * c);
            }
        }
        let integral = poly.integrate(&x);
        let s = format!("{integral}");
        if !s.contains("Integral") {
            // Integration succeeded — verify roundtrip at x=3
            let roundtrip = integral.diff(&x);
            let orig_val = format!("{}", poly.subs_i64(&x, 3));
            let rt_val = format!("{}", roundtrip.subs_i64(&x, 3));
            prop_assert_eq!(orig_val, rt_val,
                "integrate-then-diff roundtrip at x=3 for poly with coeffs {:?}", coeffs);
        }
    }

    /// When simplify changes an expression, the result should be
    /// numerically equal at a test point.
    #[test]
    fn simplify_preserves_value_poly(
        a in -5i64..5,
        b in -5i64..5,
        pt in 1i64..5,
    ) {
        let x = symplex::var("x");
        let expr = &(&x.powi(2) * a) + &(&x * b);
        let simplified = expr.simplify();
        let v1 = format!("{}", expr.subs_i64(&x, pt));
        let v2 = format!("{}", simplified.subs_i64(&x, pt));
        prop_assert_eq!(v1, v2,
            "simplify should preserve value at x={}", pt);
    }

    /// When expand changes an expression, the result should be
    /// numerically equal at a test point.
    #[test]
    fn expand_preserves_value(
        a in -3i64..3,
        _b in -3i64..3,
        pt in 1i64..5,
    ) {
        let x = symplex::var("x");
        let expr = (&x + a).powi(2);
        let expanded = expr.expand();
        let v1 = format!("{}", expr.subs_i64(&x, pt));
        let v2 = format!("{}", expanded.subs_i64(&x, pt));
        prop_assert_eq!(v1, v2,
            "expand should preserve value at x={}", pt);
    }

    // ═══════════════════════════════════════════════════════════════
    // Factor/collect/together roundtrips
    // ═══════════════════════════════════════════════════════════════

    /// factor(p) * together should preserve numerical value
    #[test]
    fn factor_preserves_value(a in -3i64..3, b in -3i64..3, pt in 1i64..5) {
        let x = symplex::var("x");
        // Build (x-a)(x-b) expanded
        let p = (&x - a) * (&x - b);
        let expanded = p.expand();
        let factored = expanded.factor(&x);
        let v1 = format!("{}", expanded.subs_i64(&x, pt));
        let v2 = format!("{}", factored.subs_i64(&x, pt));
        prop_assert_eq!(v1, v2, "factor should preserve value at x={}", pt);
    }

    // ═══════════════════════════════════════════════════════════════
    // Solve verification
    // ═══════════════════════════════════════════════════════════════

    /// All returned roots should actually satisfy the equation.
    #[test]
    fn solve_roots_satisfy(
        a in 1i64..4,
        b in -5i64..5,
        c in -5i64..5,
    ) {
        let x = symplex::var("x");
        let eq = &(&x.powi(2) * a) + &(&x * b) + c;
        let roots = eq.solve_or_empty(&x);
        for root in &roots {
            let val = eq.subs(&x, root);
            let expanded = val.expand().eval();
            let s = format!("{expanded}");
            // Should be 0 or very close
            if s != "0" {
                if let Ok(f) = expanded.evalf_f64() {
                    prop_assert!(f.abs() < 1e-6,
                        "root {} of {}x²+{}x+{}=0 gives {f}", root, a, b, c);
                }
            }
        }
    }
}
