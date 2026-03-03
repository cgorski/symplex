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
