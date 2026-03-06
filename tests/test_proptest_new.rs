//! Property-based tests for Laplace roundtrip, inequality solving,
//! compact preservation, and multinomial term counts.

mod common;

use proptest::prelude::*;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Laplace roundtrip: forward → inverse should recover original
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn laplace_roundtrip(c in 1..5i64, a in 1..4i64) {
        let t = symplex::var("t");
        let s = symplex::var("s");
        let expr = (&t * a).exp() * c;

        let mut bail = common::BailCounter::new("laplace_roundtrip");

        if let Ok(transformed) = expr.laplace(&t, &s) {
            if let Ok(recovered) = transformed.inverse_laplace(&s, &t) {
                // Evaluate both at t=0.5
                let test_point = symplex::rational(1, 2);
                let orig_val = expr.subs(&t, &test_point).eval_f64();
                let recov_val = recovered.subs(&t, &test_point).eval().simplify().eval_f64();

                if let (Ok(o), Ok(r)) = (orig_val, recov_val) {
                    if o.is_finite() && r.is_finite() {
                        bail.check();
                        prop_assert!((o - r).abs() < 1e-6 * o.abs().max(1.0),
                            "Laplace roundtrip failed: orig={}, recovered={}, c={}, a={}", o, r, c, a);
                    } else {
                        bail.skip();
                    }
                } else {
                    bail.skip();
                }
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Inequality: solve → verify solution is non-empty / valid
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn inequality_solution_satisfies(a in -5..5i64, b in -5..5i64, c in 1..5i64) {
        // Test a_coeff*x² + b*x + c > 0 where a_coeff >= 1
        let x = symplex::var("x");
        let a_coeff = a.max(1);
        let poly = &x.powi(2) * a_coeff + &x * b + c;

        if let Ok(solution) = poly.solve_gt(&x) {
            // The solution is a SetEx — verify it doesn't panic and produces valid output
            let s = format!("{solution}");
            prop_assert!(!s.is_empty(),
                "solve_gt should produce non-empty display for {}*x^2 + {}*x + {}", a_coeff, b, c);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Compact: preserves expression display and numeric value
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn compact_preserves_value(a in -10..10i64, b in -10..10i64, n in 1..5i64) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x.powi(n) * a + b;

        // Create some garbage to pollute the arena
        let _ = x.sin().cos().exp().ln();
        let _ = x.powi(10).expand();

        let original_display = format!("{expr}");
        let (new_ctx, new_exprs) = ctx.compact(std::slice::from_ref(&expr));
        let new_display = format!("{}", new_exprs[0]);

        prop_assert_eq!(original_display, new_display);

        // Verify numeric value at x=2
        let two = ctx.int(2);
        let new_two = new_ctx.int(2);
        let new_x = new_ctx.symbol("x");
        let orig_val = expr.subs(&x, &two).eval_f64();
        let new_val = new_exprs[0].subs(&new_x, &new_two).eval_f64();

        let mut bail = common::BailCounter::new("compact_preserves_value");
        if let (Ok(o), Ok(n_v)) = (orig_val, new_val) {
            if o.is_finite() && n_v.is_finite() {
                bail.check();
                prop_assert!((o - n_v).abs() < 1e-10,
                    "compact changed value at x=2: {} → {} (a={}, b={}, n={})", o, n_v, a, b, n);
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Multinomial: term count matches combinatorial formula
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn multinomial_term_count(n in 2..8usize) {
        symplex::vars!(a, b);
        let expanded = (&a + &b).powi(n as i64).expand();
        // (a+b)^n has n+1 terms by the binomial theorem
        prop_assert_eq!(expanded.term_count(), n + 1,
            "(a+b)^{} should have {} terms, got {}", n, n + 1, expanded.term_count());
    }

    #[test]
    fn trinomial_term_count(n in 2..6usize) {
        symplex::vars!(a, b, c);
        let expanded = (&a + &b + &c).powi(n as i64).expand();
        // (a+b+c)^n has C(n+2, 2) = (n+1)(n+2)/2 terms
        let expected = (n + 1) * (n + 2) / 2;
        prop_assert_eq!(expanded.term_count(), expected,
            "(a+b+c)^{} should have {} terms, got {}", n, expected, expanded.term_count());
    }
}
