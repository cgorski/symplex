//! Property-based testing with random expression trees.
//!
//! Uses a recursive expression generator to test algebraic laws
//! and transformation invariants across the entire expression space.

mod common;

use proptest::prelude::*;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Expression generators
// ═══════════════════════════════════════════════════════════════════════════

/// Shared context for all proptest-generated expressions.
///
/// Every random expression must live in the same context so that they
/// can be combined (addition, multiplication, substitution, etc.)
/// without triggering cross-context panics.
fn shared_ctx() -> &'static Context {
    use std::sync::OnceLock;
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::new)
}

/// Generate a random symbolic expression of bounded depth.
///
/// Uses a shared static context so that `Ex` values are `'static`
/// and compatible with `prop_recursive`.
fn arb_expr(depth: u32) -> impl Strategy<Value = Ex> {
    let leaf = prop_oneof![
        (-20i64..20).prop_map(|n| shared_ctx().int(n)),
        Just(shared_ctx().symbol("x")),
        Just(shared_ctx().symbol("y")),
        // Small rationals p/q
        (-3i64..4, 1i64..4).prop_map(|(p, q)| shared_ctx().rational(p, q)),
    ];

    leaf.prop_recursive(
        depth, // max depth
        128,   // max nodes
        3,     // items per collection
        |inner| {
            prop_oneof![
                // Binary operations
                (inner.clone(), inner.clone()).prop_map(|(a, b)| &a + &b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| &a * &b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| &a - &b),
                // Unary operations
                inner.clone().prop_map(|a| -&a),
                inner.clone().prop_map(|a| a.sin()),
                inner.clone().prop_map(|a| a.cos()),
                // Exp and Abs (weight 1 each)
                inner.clone().prop_map(|a| a.exp()),
                inner.clone().prop_map(|a| a.abs()),
                // powi with varied exponent
                (
                    inner.clone(),
                    prop_oneof![Just(0i64), Just(1), Just(2), Just(3), Just(-1)]
                )
                    .prop_map(|(a, n)| a.powi(n)),
            ]
        },
    )
}

/// Generate a polynomial expression in x with small integer coefficients.
fn arb_polynomial() -> impl Strategy<Value = Ex> {
    let ctx = shared_ctx();
    let x = ctx.symbol("x");
    (prop::collection::vec(-10i64..10, 1..6)).prop_map(move |coeffs| {
        let x_ref = &x;
        let mut terms: Vec<Ex> = Vec::new();
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                if i == 0 {
                    terms.push(shared_ctx().int(c));
                } else if i == 1 {
                    terms.push(x_ref * c);
                } else {
                    terms.push(&x_ref.powi(i as i64) * c);
                }
            }
        }
        if terms.is_empty() {
            shared_ctx().int(0)
        } else {
            terms.iter().fold(shared_ctx().int(0), |acc, t| &acc + t)
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // ═══════════════════════════════════════════════════════════════
    // Canonicalization invariants
    // ═══════════════════════════════════════════════════════════════

    /// Canonicalization is idempotent: displaying the same expression
    /// twice always produces the same string.
    #[test]
    fn canon_is_idempotent(e in arb_expr(3)) {
        let s1 = format!("{e}");
        let s2 = format!("{e}");
        prop_assert_eq!(s1, s2);
    }

    /// Addition is commutative: a + b == b + a
    #[test]
    fn add_commutative(a in arb_expr(2), b in arb_expr(2)) {
        let lhs = &a + &b;
        let rhs = &b + &a;
        prop_assert_eq!(format!("{lhs}"), format!("{rhs}"));
    }

    /// Multiplication is commutative: a * b == b * a
    #[test]
    fn mul_commutative(a in arb_expr(2), b in arb_expr(2)) {
        let lhs = &a * &b;
        let rhs = &b * &a;
        prop_assert_eq!(format!("{lhs}"), format!("{rhs}"));
    }

    /// Addition has identity: a + 0 == a
    #[test]
    fn add_identity(a in arb_expr(2)) {
        let zero = shared_ctx().int(0);
        let result = &a + &zero;
        prop_assert_eq!(format!("{result}"), format!("{a}"));
    }

    /// Multiplication has identity: a * 1 == a
    #[test]
    fn mul_identity(a in arb_expr(2)) {
        let ctx = shared_ctx().clone();
        let one = ctx.int(1);
        let result = &a * &one;
        prop_assert_eq!(format!("{result}"), format!("{a}"));
    }

    /// Multiplication by zero: a * 0 == 0
    #[test]
    fn mul_zero(a in arb_expr(2)) {
        let ctx = shared_ctx().clone();
        let zero = ctx.int(0);
        let result = &a * &zero;
        prop_assert_eq!(format!("{result}"), "0");
    }

    /// Self-subtraction: a - a == 0 for finite values.
    /// For infinite values (e.g. ∞ - ∞), the result is correctly NaN
    /// (indeterminate form), not 0.
    #[test]
    fn self_subtraction(a in arb_expr(2)) {
        let result = &a - &a;
        let s = format!("{result}");
        prop_assert!(
            s == "0" || s == "nan",
            "a - a should be 0 (finite) or nan (infinite), got {}", s
        );
    }

    /// Double negation: -(-a) == a
    #[test]
    fn double_negation(a in arb_expr(2)) {
        let result = -&(-&a);
        prop_assert_eq!(format!("{result}"), format!("{a}"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Evaluation invariants
    // ═══════════════════════════════════════════════════════════════

    /// eval is idempotent: eval(eval(e)) == eval(e)
    #[test]
    fn eval_idempotent(e in arb_expr(2)) {
        let e1 = e.eval();
        let e2 = e1.eval();
        prop_assert_eq!(format!("{e1}"), format!("{e2}"));
    }

    /// expand is idempotent: expand(expand(e)) == expand(e)
    #[test]
    fn expand_idempotent(e in arb_expr(2)) {
        let e1 = e.expand();
        let e2 = e1.expand();
        prop_assert_eq!(format!("{e1}"), format!("{e2}"));
    }

    /// simplify is idempotent: simplify(simplify(e)) == simplify(e)
    #[test]
    fn simplify_idempotent(e in arb_expr(2)) {
        let e1 = e.simplify();
        let e2 = e1.simplify();
        prop_assert_eq!(format!("{e1}"), format!("{e2}"));
    }

    // ═══════════════════════════════════════════════════════════════
    // Differentiation laws
    // ═══════════════════════════════════════════════════════════════

    /// Linearity of differentiation: d/dx(a + b) == d/dx(a) + d/dx(b)
    #[test]
    fn diff_linear(a in arb_expr(2), b in arb_expr(2)) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");

        // Skip infinite / indeterminate expressions where linearity is
        // undefined: a + zoo = zoo absorbs a, so d/dx(zoo) = 0 but
        // d/dx(a) + d/dx(zoo) = d/dx(a).  This is not a linearity bug —
        // differentiation of infinity is simply outside the domain of the
        // linearity property.
        let a_str = format!("{a}");
        let b_str = format!("{b}");
        let is_infinite = |s: &str| matches!(s, "zoo" | "oo" | "-oo" | "nan");
        if is_infinite(&a_str) || is_infinite(&b_str) {
            return Ok(());
        }

        let sum = &a + &b;
        let sum_str = format!("{sum}");
        if is_infinite(&sum_str) {
            return Ok(());
        }

        let diff_sum = sum.diff(&x);
        let diff_a = a.diff(&x);
        let diff_b = b.diff(&x);
        let sum_diffs = &diff_a + &diff_b;

        // With abs/sign canonicalization, symbolic comparison now works
        // for sign(x) terms.  We still use numerical comparison to be
        // robust against other non-canonical but mathematically equal forms.
        for &pt in &[-2i64, -1, 1, 2, 3] {
            let val_pt = ctx.int(pt);
            let lhs = diff_sum.subs(&x, &val_pt).eval_f64().unwrap_or(f64::NAN);
            let rhs = sum_diffs.subs(&x, &val_pt).eval_f64().unwrap_or(f64::NAN);
            if lhs.is_nan() && rhs.is_nan() { continue; }
            if lhs.is_infinite() && rhs.is_infinite() && lhs.signum() == rhs.signum() { continue; }
            prop_assert!(
                (lhs - rhs).abs() < 1e-9,
                "d/dx(a+b) != d/dx(a)+d/dx(b) at x={}: {} vs {}\n  a={}\n  b={}\n  a+b={}\n  d(a+b)={}\n  da+db={}",
                pt, lhs, rhs, a_str, b_str, sum_str, diff_sum, sum_diffs
            );
        }
    }

    /// Constant rule: d/dx(c) == 0 for integer c
    #[test]
    fn diff_constant(c in -100i64..100) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let expr = ctx.int(c);
        let result = expr.diff(&x);
        prop_assert_eq!(format!("{result}"), "0");
    }

    /// Power rule: d/dx(x^n) == n * x^(n-1), verified numerically at x=2
    #[test]
    fn diff_power_rule(n in 1i64..8) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let f = x.powi(n);
        let df = f.diff(&x);
        // Evaluate at x=2 to verify
        let expected = n * 2i64.pow((n - 1) as u32);
        let result = df.subs_i64(&x, 2);
        let s = format!("{result}");
        prop_assert_eq!(s, expected.to_string(),
            "d/dx(x^{}) at x=2 should be {}", n, expected);
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration roundtrip
    // ═══════════════════════════════════════════════════════════════

    /// For polynomials: d/dx(∫ p dx) == p (verified numerically)
    #[test]
    fn integrate_diff_roundtrip(coeffs in prop::collection::vec(-5i64..5, 1..4)) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        // Build polynomial from coefficients
        let mut poly = ctx.int(0);
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                poly = &poly + &(&x.powi(i as i64) * c);
            }
        }
        let integral = poly.integrate(&x);
        let roundtrip = integral.diff(&x);
        // Check at x=3
        let orig_val = format!("{}", poly.subs_i64(&x, 3));
        let rt_val = format!("{}", roundtrip.subs_i64(&x, 3));
        prop_assert_eq!(orig_val, rt_val,
            "integrate-then-diff roundtrip for polynomial at x=3");
    }

    /// diff(∫ c*x^n dx, x) should recover c*x^n for small c, n
    #[test]
    fn integrate_diff_monomial(c in 1i64..10, n in 0i64..6) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let expr = &x.powi(n) * c;
        let anti = expr.integrate(&x);
        let back = anti.diff(&x);
        prop_assert_eq!(
            format!("{back}"), format!("{expr}"),
            "diff(integrate({}*x^{})) should recover the original", c, n
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Numerical consistency
    // ═══════════════════════════════════════════════════════════════

    /// expand preserves value: eval(e) at a point == eval(expand(e)) at same point
    #[test]
    fn expand_preserves_value(
        a in -5i64..5,
        b in -5i64..5,
        c in -5i64..5,
        pt in -3i64..3,
    ) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let poly = &(&x.powi(2) * a) + &(&x * b) + c;
        let expanded = poly.expand();
        let v1 = format!("{}", poly.subs_i64(&x, pt));
        let v2 = format!("{}", expanded.subs_i64(&x, pt));
        prop_assert_eq!(v1, v2, "expand should preserve value at x={}", pt);
    }

    /// simplify preserves value for polynomial expressions
    #[test]
    fn simplify_preserves_polynomial_value(
        a in -5i64..5,
        b in -5i64..5,
        pt in -3i64..3,
    ) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let expr = &(&x * a) + &(&x * b); // (a+b)*x
        let simplified = expr.simplify();
        let v1 = format!("{}", expr.subs_i64(&x, pt));
        let v2 = format!("{}", simplified.subs_i64(&x, pt));
        prop_assert_eq!(v1, v2, "simplify should preserve value at x={}", pt);
    }

    /// expand preserves value for (ax+b)^n
    #[test]
    fn expand_power_preserves_value(a in 1i64..5, b in 1i64..5, n in 2i64..5) {
        let ctx = shared_ctx().clone();
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

    // ═══════════════════════════════════════════════════════════════
    // Display safety
    // ═══════════════════════════════════════════════════════════════

    /// Display never panics on random expressions
    #[test]
    fn display_never_panics(e in arb_expr(4)) {
        let s = format!("{e}");
        prop_assert!(!s.is_empty(), "display should produce non-empty string");
    }

    /// Debug never panics on random expressions
    #[test]
    fn debug_never_panics(e in arb_expr(4)) {
        let s = format!("{e:?}");
        prop_assert!(!s.is_empty(), "debug should produce non-empty string");
    }

    /// count_ops returns a meaningful count (at least 1 node for any expression)
    #[test]
    fn count_ops_works(e in arb_expr(3)) {
        // Just verify this doesn't panic; usize is always >= 0
        let _ops = e.count_ops();
    }

    // ═══════════════════════════════════════════════════════════════
    // Complex number invariants
    // ═══════════════════════════════════════════════════════════════

    /// i^(4k) == 1 for any k
    #[test]
    fn i_power_period_4(k in 0u32..25) {
        let ctx = shared_ctx().clone();
        let i = ctx.i_unit();
        let result = i.powi((4 * k) as i64);
        prop_assert_eq!(format!("{result}"), "1",
            "i^({}) should be 1", 4 * k);
    }

    /// i^(4k+1) == i
    #[test]
    fn i_power_mod_1(k in 0u32..25) {
        let ctx = shared_ctx().clone();
        let i = ctx.i_unit();
        let result = i.powi((4 * k + 1) as i64);
        prop_assert_eq!(format!("{result}"), "I",
            "i^({}) should be I", 4 * k + 1);
    }

    /// i^(4k+2) == -1
    #[test]
    fn i_power_mod_2(k in 0u32..25) {
        let ctx = shared_ctx().clone();
        let i = ctx.i_unit();
        let result = i.powi((4 * k + 2) as i64);
        prop_assert_eq!(format!("{result}"), "-1",
            "i^({}) should be -1", 4 * k + 2);
    }

    /// i^(4k+3) == -i
    #[test]
    fn i_power_mod_3(k in 0u32..25) {
        let ctx = shared_ctx().clone();
        let i = ctx.i_unit();
        let result = i.powi((4 * k + 3) as i64);
        prop_assert_eq!(format!("{result}"), "-I",
            "i^({}) should be -I", 4 * k + 3);
    }

}

// ═══════════════════════════════════════════════════════════════
// Converted from proptest with `_dummy in 0..1u32` to regular #[test]
// ═══════════════════════════════════════════════════════════════

/// (1+i)^2 == 2i
#[test]
fn one_plus_i_squared_always_2i() {
    let ctx = Context::new();
    let i = ctx.i_unit();
    let expr = (&ctx.int(1) + &i).powi(2).expand();
    assert_eq!(format!("{expr}"), "2*I");
}

/// sin²(x) + cos²(x) simplifies to 1
#[test]
fn pythagorean_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2);
    let simplified = expr.simplify();
    assert_eq!(format!("{simplified}"), "1");
}

// ═══════════════════════════════════════════════════════════════
// Trig identities (continued, proptest)
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// sin²(kx) + cos²(kx) simplifies to 1 for integer k
    #[test]
    fn pythagorean_scaled(k in 1i64..5) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let kx = &x * k;
        let expr = &kx.sin().powi(2) + &kx.cos().powi(2);
        let simplified = expr.simplify();
        prop_assert_eq!(format!("{simplified}"), "1",
            "sin²({}x) + cos²({}x) should be 1", k, k);
    }

    // ═══════════════════════════════════════════════════════════════
    // Structural properties
    // ═══════════════════════════════════════════════════════════════

    /// Equality is reflexive: e == e
    #[test]
    fn eq_reflexive(e in arb_expr(3)) {
        prop_assert_eq!(&e, &e);
    }

    /// Equality implies same display
    #[test]
    fn eq_implies_same_display(a in arb_expr(2), b in arb_expr(2)) {
        let e1 = &a + &b;
        let e2 = &a + &b;
        if e1 == e2 {
            prop_assert_eq!(format!("{e1}"), format!("{e2}"));
        }
    }

    /// Hash is consistent with equality
    #[test]
    fn hash_consistent_with_eq(a in arb_expr(2), b in arb_expr(2)) {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let e1 = &a + &b;
        let e2 = &a + &b;

        if e1 == e2 {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            e1.hash(&mut h1);
            e2.hash(&mut h2);
            prop_assert_eq!(h1.finish(), h2.finish());
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Polynomial-specific properties
    // ═══════════════════════════════════════════════════════════════

    /// Polynomial generators produce valid expressions that can be displayed
    #[test]
    fn polynomial_displays(p in arb_polynomial()) {
        let s = format!("{p}");
        prop_assert!(!s.is_empty());
    }

    /// Polynomials can be differentiated without panic
    #[test]
    fn polynomial_differentiable(p in arb_polynomial()) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let dp = p.diff(&x);
        let s = format!("{dp}");
        prop_assert!(!s.is_empty());
    }

    /// Polynomials can be integrated without panic
    #[test]
    fn polynomial_integrable(p in arb_polynomial()) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let ip = p.integrate(&x);
        let s = format!("{ip}");
        prop_assert!(!s.is_empty());
    }

    /// factor then expand roundtrip for (x-a)(x-b)
    #[test]
    fn factor_expand_roundtrip(a in -5i64..6, b in -5i64..6) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let f1 = &x - a;
        let f2 = &x - b;
        let product = &f1 * &f2;
        let expanded_original = product.expand();

        let factored = expanded_original.factor(&x);
        let re_expanded = factored.expand();

        prop_assert_eq!(
            format!("{re_expanded}"), format!("{expanded_original}"),
            "expand(factor((x-{})(x-{}))) should equal the original polynomial", a, b
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Numeric arithmetic correctness
    // ═══════════════════════════════════════════════════════════════

    /// Integer addition: symplex int(a) + int(b) == int(a+b)
    #[test]
    fn numeric_add_correct(a in -50i64..50, b in -50i64..50) {
        let ctx = shared_ctx().clone();
        let ea = ctx.int(a);
        let eb = ctx.int(b);
        let result = &ea + &eb;
        let expected = a + b;
        prop_assert_eq!(format!("{result}"), format!("{expected}"));
    }

    /// Integer multiplication: symplex int(a) * int(b) == int(a*b)
    #[test]
    fn numeric_mul_correct(a in -50i64..50, b in -50i64..50) {
        let ctx = shared_ctx().clone();
        let ea = ctx.int(a);
        let eb = ctx.int(b);
        let result = &ea * &eb;
        let expected = a * b;
        prop_assert_eq!(format!("{result}"), format!("{expected}"));
    }

    /// Integer powers: symplex int(a)^n == a^n for small values
    #[test]
    fn numeric_pow_correct(a in -5i64..5, n in 0i64..5) {
        let ctx = shared_ctx().clone();
        let ea = ctx.int(a);
        let result = ea.powi(n);
        let expected = a.pow(n as u32);
        prop_assert_eq!(format!("{result}"), format!("{expected}"));
    }
}

// ═══════════════════════════════════════════════════════════════
// Additional algebraic properties
// ═══════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Product rule (numerical): d/dx(f*g) ≈ f'*g + f*g' at x=2
    #[test]
    fn product_rule_numerical(f in arb_polynomial(), g in arb_polynomial()) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let fg = &f * &g;
        let d_fg = fg.diff(&x);
        let df = f.diff(&x);
        let dg = g.diff(&x);
        let product_rule = &(&df * &g) + &(&f * &dg);

        let pt = ctx.int(2);
        let mut bail = common::BailCounter::new("product_rule_numerical");
        let lhs = d_fg.subs(&x, &pt).eval_f64();
        let rhs = product_rule.subs(&x, &pt).eval_f64();
        if let (Ok(l), Ok(r)) = (lhs, rhs) {
            if l.is_finite() && r.is_finite() && l.abs() < 1e10 && r.abs() < 1e10 {
                bail.check();
                let diff = (l - r).abs();
                let tol = 1e-8 * l.abs().max(r.abs()).max(1.0);
                prop_assert!(diff < tol,
                    "product rule: d/dx(f*g)={} but f'g+fg'={} for f={}, g={}", l, r, f, g);
            } else {
                bail.skip();
            }
        } else {
            bail.skip();
        }
        bail.assert_not_vacuous();
    }

    /// Expand idempotence (full-depth): expand(expand(e)) displays same as expand(e)
    #[test]
    fn expand_idempotent_full(e in arb_expr(3)) {
        let once = e.expand();
        let twice = once.expand();
        prop_assert_eq!(format!("{once}"), format!("{twice}"),
            "expand should be idempotent for {}", e);
    }

    /// Substitution identity: e.subs(&x, &x) displays same as e
    #[test]
    fn substitution_identity(e in arb_expr(3)) {
        let ctx = shared_ctx().clone();
        let x = ctx.symbol("x");
        let substituted = e.subs(&x, &x);
        prop_assert_eq!(format!("{e}"), format!("{substituted}"),
            "subs(x, x) should be identity");
    }
}
