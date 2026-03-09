//! Stress testing: run all operations on random expressions.
//!
//! Uses proptest to generate random expression trees, then applies
//! every public operation inside catch_unwind to detect panics.
//! Also verifies integration correctness via differentiation roundtrip.

use proptest::prelude::*;
use symplex::prelude::*;

mod common;

/// Generate a random expression using the global context.
fn arb_expr(depth: u32) -> impl Strategy<Value = Ex> {
    let leaf = prop_oneof![
        (-10i64..10).prop_map(|n| symplex::default_context().int(n)),
        Just(symplex::default_context().symbol("x")),
        Just(symplex::default_context().symbol("y")),
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
        let x = symplex::default_context().symbol("x");
        let mut result = symplex::default_context().int(0);
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

    /// Display never panics and produces non-empty output
    #[test]
    fn stress_display(e in arb_expr(3)) {
        let s = format!("{e}");
        prop_assert!(!s.is_empty(), "display should produce non-empty string");
        let d = format!("{e:?}");
        prop_assert!(!d.is_empty(), "debug should produce non-empty string");
    }

    /// Clone + equality never panics; clone equals original
    #[test]
    fn stress_clone_eq(e in arb_expr(3)) {
        let e2 = e.clone();
        prop_assert!(e == e2, "clone should be equal to original: {} vs {}", e, e2);
    }

    /// eval never panics and produces non-empty display
    #[test]
    fn stress_eval(e in arb_expr(3)) {
        let result = e.eval();
        let s = format!("{result}");
        prop_assert!(!s.is_empty(), "eval result should display as non-empty");
    }

    /// expand never panics and produces non-empty display
    #[test]
    fn stress_expand(e in arb_expr(2)) {
        let result = e.expand();
        let s = format!("{result}");
        prop_assert!(!s.is_empty(), "expand result should display as non-empty");
    }

    /// simplify never panics and produces non-empty display
    #[test]
    fn stress_simplify(e in arb_expr(2)) {
        let result = e.simplify();
        let s = format!("{result}");
        prop_assert!(!s.is_empty(), "simplify result should display as non-empty");
    }

    /// full_simplify never panics and produces non-empty display
    #[test]
    fn stress_full_simplify(e in arb_expr(2)) {
        let result = e.full_simplify();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "full_simplify result should display as non-empty");
    }

    /// smart_simplify never panics and produces non-empty display
    #[test]
    fn stress_smart_simplify(e in arb_expr(2)) {
        let result = e.smart_simplify();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "smart_simplify result should display as non-empty");
    }

    /// diff never panics and produces non-empty display
    #[test]
    fn stress_diff(e in arb_expr(3)) {
        let x = symplex::default_context().symbol("x");
        let result = e.diff(&x);
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "diff result should display as non-empty");
    }

    /// integrate never panics (may return unevaluated Integral)
    #[test]
    fn stress_integrate(e in arb_expr(2)) {
        let x = symplex::default_context().symbol("x");
        let result = e.integrate(&x);
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "integrate result should display as non-empty");
    }

    /// solve never panics (may return empty)
    #[test]
    fn stress_solve(e in arb_expr(2)) {
        let x = symplex::default_context().symbol("x");
        let roots = e.solve_or_empty(&x);
        for r in &roots {
            let _s = format!("{r}");
            prop_assert!(!_s.is_empty(), "solve root should display as non-empty");
        }
    }

    /// subs never panics and produces non-empty display
    #[test]
    fn stress_subs(e in arb_expr(2)) {
        let x = symplex::default_context().symbol("x");
        let result = e.subs_i64(&x, 3);
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "subs result should display as non-empty");
    }

    /// count_ops never panics
    #[test]
    fn stress_count_ops(e in arb_expr(3)) {
        let _n = e.count_ops();
        let _s = format!("{_n}");
        prop_assert!(!_s.is_empty());
    }

    /// args never panics
    #[test]
    fn stress_args(e in arb_expr(3)) {
        let args = e.args();
        let _s = format!("{:?}", args);
        prop_assert!(!_s.is_empty());
    }

    /// expr_type never panics
    #[test]
    fn stress_expr_type(e in arb_expr(3)) {
        let ty = e.expr_type();
        let _s = format!("{:?}", ty);
        prop_assert!(!_s.is_empty());
    }

    /// free_symbols never panics
    #[test]
    fn stress_free_symbols(e in arb_expr(3)) {
        let syms = e.free_symbols();
        let _s = format!("{:?}", syms);
        prop_assert!(!_s.is_empty());
    }

    /// contains never panics
    #[test]
    fn stress_contains(e in arb_expr(3)) {
        let x = symplex::default_context().symbol("x");
        let _b = e.contains(&x);
        let _s = format!("{_b}");
        prop_assert!(!_s.is_empty());
    }

    /// term_count never panics
    #[test]
    fn stress_term_count(e in arb_expr(3)) {
        let _n = e.term_count();
        let _s = format!("{_n}");
        prop_assert!(!_s.is_empty());
    }

    /// to_tree / to_json never panics
    #[test]
    fn stress_serde(e in arb_expr(2)) {
        let tree = e.to_tree();
        let json = e.to_json().unwrap();
        prop_assert!(!json.is_empty());
        // Round-trip
        let ctx = symplex::default_context();
        let back = ctx.from_tree(&tree);
        prop_assert_eq!(format!("{e}"), format!("{back}"));
    }

    /// expand_trig never panics
    #[test]
    fn stress_expand_trig(e in arb_expr(2)) {
        let result = e.expand_trig();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "expand_trig result should display as non-empty");
    }

    /// expand_log never panics
    #[test]
    fn stress_expand_log(e in arb_expr(2)) {
        let result = e.expand_log();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "expand_log result should display as non-empty");
    }

    /// logcombine never panics
    #[test]
    fn stress_logcombine(e in arb_expr(2)) {
        let result = e.log_combine();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "logcombine result should display as non-empty");
    }

    /// trig_combine never panics
    #[test]
    fn stress_trig_combine(e in arb_expr(2)) {
        let result = e.trig_combine();
        let _s = format!("{result}");
        prop_assert!(!_s.is_empty(), "trig_combine result should display as non-empty");
    }

    /// factor_terms never panics
    #[test]
    fn stress_factor_terms(e in arb_expr(2)) {
        let (gcd, inner) = e.factor_terms();
        let _s1 = format!("{gcd}");
        let _s2 = format!("{inner}");
        prop_assert!(!_s1.is_empty(), "factor_terms gcd should display as non-empty");
        prop_assert!(!_s2.is_empty(), "factor_terms inner should display as non-empty");
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration verification
    // ═══════════════════════════════════════════════════════════════

    /// When integrate returns a non-Integral result for a polynomial,
    /// differentiating should give back the original.
    #[test]
    fn integration_roundtrip_poly(coeffs in prop::collection::vec(-5i64..5, 1..4)) {
        let x = symplex::default_context().symbol("x");
        let mut poly = symplex::default_context().int(0);
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
        let x = symplex::default_context().symbol("x");
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
        let x = symplex::default_context().symbol("x");
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
        let x = symplex::default_context().symbol("x");
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
        let x = symplex::default_context().symbol("x");
        let eq = &(&x.powi(2) * a) + &(&x * b) + c;
        let roots = eq.solve_or_empty(&x);
        let mut bail = common::BailCounter::new("solve_roots_satisfy");
        for root in &roots {
            let val = eq.subs(&x, root);
            let expanded = val.expand().eval();
            let s = format!("{expanded}");
            // Should be 0 or very close
            if s == "0" {
                bail.check();
            } else if let Ok(f) = expanded.eval_f64() {
                bail.check();
                prop_assert!(f.abs() < 1e-6,
                    "root {} of {}x²+{}x+{}=0 gives {f}", root, a, b, c);
            } else {
                bail.skip();
            }
        }
        if !roots.is_empty() {
            bail.assert_not_vacuous();
        }
    }
}
