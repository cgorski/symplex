//! Property-based tests for diff() and expand().

use proptest::prelude::*;
use symplex::prelude::*;

proptest! {
    #[test]
    fn diff_of_constant_is_zero(c in -100i64..100) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = ctx.int(c);
        let d = expr.diff(&x);
        prop_assert!(d.is_zero_structural(), "d/dx({c}) should be 0, got: {d}");
    }

    #[test]
    fn diff_is_linear_over_addition(a in 1i64..20, b in 1i64..20) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let f = &x * a;
        let g = &x.powi(2) * b;
        let sum = &f + &g;
        let d_sum = sum.diff(&x);
        let d_f = f.diff(&x);
        let d_g = g.diff(&x);
        let d_f_plus_d_g = &d_f + &d_g;
        // Both should have the same canonical form.
        prop_assert_eq!(
            format!("{d_sum}"), format!("{d_f_plus_d_g}"),
            "diff(f+g) should equal diff(f)+diff(g)"
        );
    }

    #[test]
    fn expand_is_idempotent(a in 1i64..10, b in 1i64..10) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = (&x * a + b).powi(2);
        let once = expr.expand();
        let twice = once.expand();
        prop_assert_eq!(
            format!("{once}"), format!("{twice}"),
            "expand should be idempotent"
        );
    }

    #[test]
    fn expand_preserves_value_at_point(a in 1i64..5, b in 1i64..5, n in 2i64..5) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let one = ctx.int(1);
        let expr = (&x * a + b).powi(n);
        let expanded = expr.expand();
        let orig_at_1 = expr.subs(&x, &one);
        let exp_at_1 = expanded.subs(&x, &one);
        prop_assert_eq!(
            format!("{orig_at_1}"), format!("{exp_at_1}"),
            "expand should preserve value: ({}*x+{})^{} at x=1", a, b, n
        );
    }

    #[test]
    fn diff_of_x_to_n(n in 2i64..10) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = x.powi(n);
        let d = expr.diff(&x);
        // Verify by substituting x=2: d/dx(x^n) at x=2 should be n*2^(n-1)
        let at_2 = d.subs(&x, &ctx.int(2));
        let expected = n * 2i64.pow((n - 1) as u32);
        prop_assert_eq!(
            format!("{at_2}"), format!("{expected}"),
            "d/dx(x^{}) at x=2 should be {}", n, expected
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration roundtrip for polynomials
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn integrate_diff_roundtrip_polynomial(a in -10i64..10, b in -10i64..10, c in -10i64..10, eval_pt in -5i64..5) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Build polynomial a*x^2 + b*x + c
        let poly = &(&x.powi(2) * a) + &(&x * b) + c;
        let integrated = poly.integrate(&x);
        let roundtrip = integrated.diff(&x);
        // Evaluate both at eval_pt and compare
        let original_val = poly.subs_i64(&x, eval_pt);
        let roundtrip_val = roundtrip.subs_i64(&x, eval_pt);
        let orig_s = format!("{original_val}");
        let rt_s = format!("{roundtrip_val}");
        prop_assert_eq!(orig_s, rt_s, "integrate-then-diff should roundtrip for {}x²+{}x+{} at x={}", a, b, c, eval_pt);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Eval idempotence
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn eval_is_idempotent(k in 0i64..12) {
        let ctx = Context::new();
        // Evaluate sin(k*π/6) twice
        let angle = &ctx.rational(k, 6) * &ctx.pi();
        let first = angle.sin().eval();
        let second = first.eval();
        let s1 = format!("{first}");
        let s2 = format!("{second}");
        prop_assert_eq!(s1, s2, "eval should be idempotent for sin({}π/6)", k);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplify idempotence
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn simplify_is_idempotent(a in -3i64..3, b in -3i64..3) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Build: a*sin(x)^2 + b*cos(x)^2
        let expr = &(&x.sin().powi(2) * a) + &(&x.cos().powi(2) * b);
        let first = expr.simplify();
        let second = first.simplify();
        let s1 = format!("{first}");
        let s2 = format!("{second}");
        prop_assert_eq!(s1, s2, "simplify should be idempotent");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expand/logcombine quasi-roundtrip
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #[test]
    fn expand_log_logcombine_roundtrip(a in 2i64..20, b in 2i64..20) {
        let ctx = Context::new();
        // ln(a*b) → expand → logcombine should give back ln(a*b)
        let product = &ctx.int(a) * &ctx.int(b);
        let ln_product = product.ln();
        let expanded = ln_product.expand_log();
        let recombined = expanded.logcombine();
        let orig_s = format!("{ln_product}");
        let recom_s = format!("{recombined}");
        prop_assert_eq!(orig_s, recom_s, "expand_log then logcombine should roundtrip for ln({}*{})", a, b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Value-preservation properties for simplification
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// simplify() must preserve mathematical value.
    /// For random polynomial expressions, simplify(e) evaluated at a point
    /// must equal e evaluated at the same point.
    #[test]
    fn simplify_preserves_value(a in -5i64..5, b in -5i64..5, c in -5i64..5) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Build a polynomial: a*x^2 + b*x + c + sin(x)^2 + cos(x)^2
        // The trig part should simplify to 1 via pythagorean rule
        let expr = &((&x * a).powi(2)) + &(&(&x * b) + c);
        let trig = &x.sin().powi(2) + &x.cos().powi(2);
        let full = &expr + &trig;

        let simplified = full.simplify();

        // Evaluate both at x = 7/10 (avoid integer points where division by zero is common)
        let point = ctx.rational(7, 10);
        let val_orig = full.subs(&x, &point).evalf_f64();
        let val_simp = simplified.subs(&x, &point).evalf_f64();

        if let (Ok(v1), Ok(v2)) = (val_orig, val_simp) {
            let diff = (v1 - v2).abs();
            prop_assert!(diff < 1e-8,
                "simplify changed value: {v1} vs {v2} for a={a}, b={b}, c={c}");
        }
    }

    /// smart_simplify() must preserve mathematical value.
    #[test]
    fn smart_simplify_preserves_value(a in -3i64..3, b in -3i64..3) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Build expression with numeric coefficients that trigger factor_terms
        let expr = &(&x * (a * 2)) + (b * 2);

        let simplified = expr.smart_simplify();

        let point = ctx.rational(3, 7);
        let val_orig = expr.subs(&x, &point).evalf_f64();
        let val_simp = simplified.subs(&x, &point).evalf_f64();

        if let (Ok(v1), Ok(v2)) = (val_orig, val_simp) {
            let diff = (v1 - v2).abs();
            prop_assert!(diff < 1e-8,
                "smart_simplify changed value: {v1} vs {v2}");
        }
    }

    /// full_simplify() must preserve mathematical value.
    #[test]
    fn full_simplify_preserves_value(a in 1i64..5, _b in 1i64..5) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = (&x + a).powi(2);

        let simplified = expr.full_simplify();

        let point = ctx.rational(1, 3);
        let val_orig = expr.subs(&x, &point).evalf_f64();
        let val_simp = simplified.subs(&x, &point).evalf_f64();

        if let (Ok(v1), Ok(v2)) = (val_orig, val_simp) {
            let diff = (v1 - v2).abs();
            prop_assert!(diff < 1e-6,
                "full_simplify changed value: {v1} vs {v2}");
        }
    }

    /// Integration FTC: d/dx(∫ p(x)·ln(x) dx) should equal p(x)·ln(x).
    #[test]
    fn integrate_poly_ln_ftc(a in 1i64..4, n in 1i64..3) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let poly_ln = &(&x.powi(n) * a) * &x.ln();

        let integral = poly_ln.integrate(&x);
        // Skip if integration returned unevaluated
        let integral_str = format!("{integral}");
        if integral_str.contains("Integral") {
            return Ok(());  // Can't verify unevaluated integrals
        }

        let derivative = integral.diff(&x);

        // Evaluate both at x = 2
        let point = ctx.int(2);
        let val_orig = poly_ln.subs(&x, &point).evalf_f64();
        let val_deriv = derivative.subs(&x, &point).evalf_f64();

        if let (Ok(v1), Ok(v2)) = (val_orig, val_deriv) {
            let diff = (v1 - v2).abs();
            prop_assert!(diff < 1e-6,
                "FTC failed: d/dx(∫ x^{n}·ln(x) dx) ≠ x^{n}·ln(x) at x=2: {v1} vs {v2}");
        }
    }
}
