//! Property-based round-trip tests for integrate and factor.
//!
//! These verify algebraic invariants:
//! - diff(integrate(f, x), x) == f (fundamental theorem of calculus)
//! - expand(factor(f, x)) == expand(f) (factoring preserves polynomial identity)

mod common;

use proptest::prelude::*;
use symplex::prelude::*;

proptest! {
    /// diff(∫ c*x^n dx, x) should equal c*x^n for small integer c and n.
    #[test]
    fn diff_of_integrate_power(c in 1i64..10, n in 0i64..6) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = &x.powi(n) * c;
        let anti = expr.integrate(&x);
        let back = anti.diff(&x);
        // The round-trip should recover the original expression.
        prop_assert_eq!(
            format!("{back}"), format!("{expr}"),
            "diff(integrate({}*x^{})) should recover {}*x^{}", c, n, c, n
        );
    }

    /// diff(∫ sin(x) dx, x) == sin(x) and diff(∫ cos(x) dx, x) == cos(x).
    #[test]
    fn diff_of_integrate_trig(choice in 0u8..2) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = if choice == 0 { x.sin() } else { x.cos() };
        let anti = expr.integrate(&x);
        let back = anti.diff(&x);
        prop_assert_eq!(
            format!("{back}"), format!("{expr}"),
            "diff(integrate(trig)) should round-trip"
        );
    }

    /// For a polynomial f with integer coefficients and known rational roots,
    /// expand(factor(f, x)) should equal expand(f).
    /// We construct f = (x - a)(x - b) for small integers a, b.
    #[test]
    fn factor_expand_roundtrip(a in -5i64..6, b in -5i64..6) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        // Build (x - a) * (x - b) then expand to get the polynomial
        let f1 = &x - a;
        let f2 = &x - b;
        let product = &f1 * &f2;
        let expanded_original = product.expand();

        // Factor the expanded polynomial
        let factored = expanded_original.factor(&x);

        // Re-expand the factored form
        let re_expanded = factored.expand();

        // Both should be the same polynomial
        prop_assert_eq!(
            format!("{re_expanded}"), format!("{expanded_original}"),
            "expand(factor((x-{})(x-{}))) should equal the original polynomial", a, b
        );
    }

    /// integrate(constant, x) == constant * x, and diff of that recovers the constant.
    #[test]
    fn integrate_constant_roundtrip(c in -20i64..21) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let constant = ctx.int(c);
        let anti = constant.integrate(&x);
        let back = anti.diff(&x);
        prop_assert_eq!(
            format!("{back}"), format!("{constant}"),
            "diff(integrate({})) should recover {}", c, c
        );
    }

    /// series(polynomial, x, 0, high_order) should equal the polynomial itself.
    /// For a polynomial of degree d, series to order d+1 is exact.
    #[test]
    fn series_of_polynomial_is_exact(a in -5i64..6, b in -5i64..6, c in -5i64..6) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let zero = ctx.int(0);
        // Build a*x^2 + b*x + c
        let poly = &x.powi(2) * a + &x * b + c;
        // Series to order 5 (well above degree 2) should be exact.
        let s = poly.series(&x, &zero, 5);
        prop_assert_eq!(
            format!("{s}"), format!("{poly}"),
            "series of {}*x^2+{}*x+{} should be exact", a, b, c
        );
    }
}
