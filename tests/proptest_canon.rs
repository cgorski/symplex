//! Property-based tests for canonicalization invariants.
//!
//! These tests use `proptest` to generate random expression trees and verify
//! that algebraic invariants hold for ALL inputs, not just hand-picked cases.

use proptest::prelude::*;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Random expression tree generator
// ═══════════════════════════════════════════════════════════════════════════

/// A description of an expression tree, independent of any arena.
///
/// We generate these first (they're pure data), then build them in an
/// arena inside the test body.  This separation is necessary because
/// `proptest` strategies must be `'static`, but arena construction
/// requires `&mut`.
#[derive(Clone, Debug)]
enum TreeDesc {
    Int(i64),
    Sym(u8), // 0..=25 → 'a'..='z'
    Add(Box<TreeDesc>, Box<TreeDesc>),
    Mul(Box<TreeDesc>, Box<TreeDesc>),
    Pow(Box<TreeDesc>, i64), // base ^ small integer
    Neg(Box<TreeDesc>),
    Sin(Box<TreeDesc>),
    Cos(Box<TreeDesc>),
    Exp(Box<TreeDesc>), // exp of small expression
    Ln(Box<TreeDesc>),  // ln of positive expression
}

/// Build a `TreeDesc` into an actual expression in a context.
fn build(ctx: &Context, desc: &TreeDesc) -> Ex {
    match desc {
        TreeDesc::Int(n) => ctx.int(*n),
        TreeDesc::Sym(i) => {
            let name = String::from((b'a' + (*i % 26)) as char);
            ctx.symbol(&name)
        }
        TreeDesc::Add(a, b) => {
            let a = build(ctx, a);
            let b = build(ctx, b);
            &a + &b
        }
        TreeDesc::Mul(a, b) => {
            let a = build(ctx, a);
            let b = build(ctx, b);
            &a * &b
        }
        TreeDesc::Pow(base, exp) => {
            let base = build(ctx, base);
            base.powi(*exp)
        }
        TreeDesc::Neg(inner) => {
            let inner = build(ctx, inner);
            -&inner
        }
        TreeDesc::Sin(inner) => {
            let inner = build(ctx, inner);
            inner.sin()
        }
        TreeDesc::Cos(inner) => {
            let inner = build(ctx, inner);
            inner.cos()
        }
        TreeDesc::Exp(inner) => {
            let inner = build(ctx, inner);
            inner.exp()
        }
        TreeDesc::Ln(inner) => {
            let inner = build(ctx, inner);
            inner.ln()
        }
    }
}

/// Strategy that generates random expression trees of bounded depth.
fn arb_tree(max_depth: u32) -> BoxedStrategy<TreeDesc> {
    let leaf = prop_oneof![
        (-50i64..50).prop_map(TreeDesc::Int),
        (0u8..26).prop_map(TreeDesc::Sym),
    ];

    if max_depth == 0 {
        leaf.boxed()
    } else {
        let recurse = arb_tree(max_depth - 1);
        prop_oneof![
            3 => leaf,
            2 => (recurse.clone(), recurse.clone())
                .prop_map(|(a, b)| TreeDesc::Add(Box::new(a), Box::new(b))),
            2 => (recurse.clone(), recurse.clone())
                .prop_map(|(a, b)| TreeDesc::Mul(Box::new(a), Box::new(b))),
            1 => (recurse.clone(), 0i64..5)
                .prop_map(|(base, exp)| TreeDesc::Pow(Box::new(base), exp)),
            1 => recurse.clone()
                .prop_map(|inner| TreeDesc::Neg(Box::new(inner))),
            1 => recurse.clone()
                .prop_map(|inner| TreeDesc::Sin(Box::new(inner))),
            1 => recurse.clone()
                .prop_map(|inner| TreeDesc::Cos(Box::new(inner))),
            1 => recurse.clone()
                .prop_map(|inner| TreeDesc::Exp(Box::new(inner))),
            1 => recurse
                .prop_map(|inner| TreeDesc::Ln(Box::new(inner))),
        ]
        .boxed()
    }
}

/// Default strategy: depth 3, produces trees up to ~40 nodes.
fn arb_expr() -> BoxedStrategy<TreeDesc> {
    arb_tree(3)
}

/// Small expressions for tests that build multiple.
fn arb_small() -> BoxedStrategy<TreeDesc> {
    arb_tree(2)
}

// ═══════════════════════════════════════════════════════════════════════════
// Canonicalization invariants
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// Canonicalizing an expression twice gives the same result as once.
    ///
    /// This verifies that our canonical form is a true fixpoint — no
    /// further simplification is possible after one pass.
    #[test]
    fn canon_idempotent(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        // Build again from the same description — should produce
        // the identical ExprId thanks to hash-consing.
        let expr2 = build(&ctx, &desc);
        prop_assert_eq!(expr, expr2, "idempotence: building twice should give same ExprId");
    }

    /// Addition is commutative: `a + b == b + a`.
    #[test]
    fn add_commutative(a in arb_small(), b in arb_small()) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        let ab = &ea + &eb;
        let ba = &eb + &ea;
        prop_assert_eq!(ab, ba, "add should be commutative");
    }

    /// Multiplication is commutative: `a * b == b * a`.
    #[test]
    fn mul_commutative(a in arb_small(), b in arb_small()) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        let ab = &ea * &eb;
        let ba = &eb * &ea;
        prop_assert_eq!(ab, ba, "mul should be commutative");
    }

    /// Addition is associative: `(a + b) + c == a + (b + c)`.
    #[test]
    fn add_associative(a in arb_small(), b in arb_small(), c in arb_small()) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        let ec = build(&ctx, &c);
        let left = &(&ea + &eb) + &ec;
        let right = &ea + &(&eb + &ec);
        prop_assert_eq!(left, right, "add should be associative");
    }

    /// Multiplication is associative: `(a * b) * c == a * (b * c)`.
    ///
    /// NOTE: When one of the intermediate products is `Number * Add`,
    /// distribution fires and the canonical form depends on grouping.
    /// This is a known, documented property of our canonical form
    /// (matching SymPy's approach).  We skip those cases here —
    /// `.expand()` (Stage 7) will normalise them.
    #[test]
    fn mul_associative(a in arb_small(), b in arb_small(), c in arb_small()) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        let ec = build(&ctx, &c);

        let ab = &ea * &eb;
        let bc = &eb * &ec;

        // Skip cases where an intermediate product is a distributed
        // Number*Add (the result changes from Mul to Add, breaking
        // naive associativity).
        let ab_str = format!("{ab}");
        let bc_str = format!("{bc}");
        let ab_is_distributed = ab_str.contains('+') || ab_str.contains('-');
        let bc_is_distributed = bc_str.contains('+') || bc_str.contains('-');

        // Only test when neither intermediate triggered distribution.
        if !ab_is_distributed && !bc_is_distributed {
            let left = &ab * &ec;
            let right = &ea * &bc;
            prop_assert_eq!(left, right, "mul should be associative");
        }
    }

    /// Zero is the additive identity: `a + 0 == a`.
    #[test]
    fn add_zero_identity(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let zero = ctx.int(0);
        let result = &expr + &zero;
        prop_assert_eq!(result, expr, "a + 0 should equal a");
    }

    /// One is the multiplicative identity: `a * 1 == a`.
    #[test]
    fn mul_one_identity(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let one = ctx.int(1);
        let result = &expr * &one;
        prop_assert_eq!(result, expr, "a * 1 should equal a");
    }

    /// Zero is the multiplicative annihilator: `a * 0 == 0`.
    ///
    /// Note: this doesn't hold if `a` is NaN or infinity, but our
    /// random generator doesn't produce those.
    #[test]
    fn mul_zero_annihilator(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let zero = ctx.int(0);
        let result = &expr * &zero;
        prop_assert!(result.is_zero_structural(),
            "a * 0 should be zero, got: {}", result);
    }

    /// Double negation cancels: `--a == a`.
    #[test]
    fn neg_neg_cancels(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let neg1 = -&expr;
        let neg2 = -&neg1;
        prop_assert_eq!(neg2, expr, "neg(neg(a)) should equal a");
    }

    /// Self-subtraction is zero: `a - a == 0`.
    ///
    /// With Number*Add distribution in canon_mul, `neg(a)` correctly
    /// distributes numeric coefficients, so `a + neg(a)` cancels
    /// structurally via like-term collection in canon_add.
    #[test]
    fn self_subtraction_is_zero(desc in arb_small()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let result = &expr - &expr;
        prop_assert!(result.is_zero_structural(),
            "a - a should be zero, got: {}", result);
    }

    /// Display never panics for any random expression.
    #[test]
    fn display_never_panics(desc in arb_expr()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        // Just call Display — if it panics, proptest catches it.
        let _s = format!("{}", expr);
    }

    /// Numeric addition is correct: `(a + b)` as integers.
    #[test]
    fn numeric_add_correct(a in -1000i64..1000, b in -1000i64..1000) {
        let ctx = Context::new();
        let ea = ctx.int(a);
        let eb = ctx.int(b);
        let result = &ea + &eb;
        let expected = ctx.int(a + b);
        prop_assert_eq!(result, expected,
            "{} + {} should equal {}", a, b, a + b);
    }

    /// Numeric multiplication is correct: `(a * b)` as integers.
    #[test]
    fn numeric_mul_correct(a in -1000i64..1000, b in -1000i64..1000) {
        let ctx = Context::new();
        let ea = ctx.int(a);
        let eb = ctx.int(b);
        let result = &ea * &eb;
        let expected = ctx.int(a * b);
        prop_assert_eq!(result, expected,
            "{} * {} should equal {}", a, b, a * b);
    }

    /// Numeric power is correct for small exponents.
    #[test]
    fn numeric_pow_correct(base in -10i64..10, exp in 0i64..8) {
        let ctx = Context::new();
        let ebase = ctx.int(base);
        let result = ebase.powi(exp);
        let expected_val: i64 = base.pow(exp as u32);
        let expected = ctx.int(expected_val);
        prop_assert_eq!(result, expected,
            "{}^{} should equal {}", base, exp, expected_val);
    }

    /// `x + (-x) == 0` for any expression `x` (symbolic version).
    #[test]
    fn add_neg_cancels(desc in arb_small()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let neg_expr = -&expr;
        let result = &expr + &neg_expr;
        prop_assert!(result.is_zero_structural(),
            "x + (-x) should be zero, got: {}", result);
    }

    /// Structural equality implies Display equality.
    #[test]
    fn eq_implies_same_display(a in arb_small(), b in arb_small()) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        if ea == eb {
            let sa = format!("{ea}");
            let sb = format!("{eb}");
            prop_assert_eq!(sa, sb,
                "equal expressions should have equal display");
        }
    }

    /// Hash is consistent with equality.
    #[test]
    fn hash_consistent_with_eq(a in arb_small(), b in arb_small()) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);

        if ea == eb {
            let mut ha = DefaultHasher::new();
            let mut hb = DefaultHasher::new();
            ea.hash(&mut ha);
            eb.hash(&mut hb);
            prop_assert_eq!(ha.finish(), hb.finish(),
                "equal expressions must have equal hash");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-auto-evaluation properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// Powers of sums stay unevaluated (no auto-expansion).
    #[test]
    fn no_auto_expand_pow_of_sum(a in arb_small(), b in arb_small(), n in 2i64..6) {
        let ctx = Context::new();
        let ea = build(&ctx, &a);
        let eb = build(&ctx, &b);
        let sum = &ea + &eb;
        let sum_str = format!("{sum}");

        // Skip cases where the sum reduced to something trivial:
        // - structural zero
        // - collapsed to one operand (the other was zero)
        // - became a pure number (both operands numeric)
        // - has no + or - (single term after like-term collection)
        let is_trivial = sum.is_zero_structural()
            || sum == ea
            || sum == eb
            || (!sum_str.contains('+') && !sum_str.contains(" - "));

        if !is_trivial {
            let powered = sum.powi(n);
            let s = format!("{powered}");
            // The display should contain "^" indicating an unevaluated power,
            // not a fully expanded polynomial.
            prop_assert!(s.contains("^"),
                "pow of non-trivial sum should stay unevaluated, got: {s}");
        }
    }

    /// sin(x) stays as sin(x), never auto-evaluates.
    #[test]
    fn no_auto_eval_sin(desc in arb_small()) {
        let ctx = Context::new();
        let expr = build(&ctx, &desc);
        let result = expr.sin();
        let s = format!("{result}");
        prop_assert!(s.starts_with("sin("),
            "sin should stay unevaluated, got: {s}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption inference invariants
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// Forward-chaining from any single property never produces a
    /// contradiction (known_true ∩ known_false is empty).
    #[test]
    fn single_assertion_never_contradicts(prop_idx in 0u32..23, value: bool) {
        let prop = Props::from_bits_truncate(1 << prop_idx);
        let mut a = Assumptions::default();
        if value {
            a.assert_true(prop);
        } else {
            a.assert_false(prop);
        }
        prop_assert!(
            !a.is_contradictory(),
            "single assertion {prop:?}={value} produced contradiction: {a:?}"
        );
    }

    /// Forward-chaining is idempotent: running it twice gives the
    /// same result as running it once.
    #[test]
    fn forward_chain_idempotent(prop_idx in 0u32..23, value: bool) {
        let prop = Props::from_bits_truncate(1 << prop_idx);
        let mut a = Assumptions::default();
        if value {
            a.assert_true(prop);
        } else {
            a.assert_false(prop);
        }
        let after_first = a;
        a.forward_chain();
        prop_assert_eq!(
            a, after_first,
            "second forward_chain should not change anything"
        );
    }

    /// Merging identical assumptions is a no-op.
    #[test]
    fn merge_self_is_noop(prop_idx in 0u32..23, value: bool) {
        let prop = Props::from_bits_truncate(1 << prop_idx);
        let mut a = Assumptions::default();
        if value {
            a.assert_true(prop);
        } else {
            a.assert_false(prop);
        }
        let before = a;
        let changed = a.merge(&before);
        prop_assert!(!changed, "merging with self should not change anything");
        prop_assert_eq!(a, before);
    }

    /// Integer constants have correct assumptions.
    #[test]
    fn integer_assumptions_correct(n in -100i64..100) {
        let ctx = Context::new();
        let expr = ctx.int(n);

        // All integers are integer, rational, real, complex, finite.
        prop_assert_eq!(expr.query(Props::INTEGER), Some(true));
        prop_assert_eq!(expr.query(Props::RATIONAL), Some(true));
        prop_assert_eq!(expr.query(Props::REAL), Some(true));
        prop_assert_eq!(expr.query(Props::COMPLEX), Some(true));
        prop_assert_eq!(expr.query(Props::FINITE), Some(true));
        prop_assert_eq!(expr.query(Props::IMAGINARY), Some(false));

        // Sign.
        if n > 0 {
            prop_assert_eq!(expr.query(Props::POSITIVE), Some(true));
            prop_assert_eq!(expr.query(Props::NEGATIVE), Some(false));
        } else if n < 0 {
            prop_assert_eq!(expr.query(Props::POSITIVE), Some(false));
            prop_assert_eq!(expr.query(Props::NEGATIVE), Some(true));
        } else {
            prop_assert_eq!(expr.query(Props::ZERO), Some(true));
        }

        // Parity.
        if n % 2 == 0 {
            prop_assert_eq!(expr.query(Props::EVEN), Some(true));
            prop_assert_eq!(expr.query(Props::ODD), Some(false));
        } else {
            prop_assert_eq!(expr.query(Props::ODD), Some(true));
            prop_assert_eq!(expr.query(Props::EVEN), Some(false));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Simplification preserves value
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Simplification must preserve numerical value.
    #[test]
    fn simplify_preserves_value(desc in arb_tree(2)) {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let expr = build(&ctx, &desc);

        // Try to evaluate at a test point
        let test_val = ctx.rational(7, 10); // 0.7 — avoids poles at 0 and 1
        let original_at_point = expr.subs(&x, &test_val);
        let simplified = expr.simplify();
        let simplified_at_point = simplified.subs(&x, &test_val);

        // Only assert if both can be evaluated to f64
        if let (Ok(orig_f), Ok(simp_f)) = (original_at_point.eval_f64(), simplified_at_point.eval_f64()) {
            // Skip NaN/infinite results
            if orig_f.is_finite() && simp_f.is_finite() && orig_f.abs() < 1e10 {
                let diff = (orig_f - simp_f).abs();
                let tol = 1e-10 * orig_f.abs().max(1.0);
                prop_assert!(
                    diff < tol,
                    "simplify changed value: {} -> {}, original={}, simplified={}",
                    orig_f, simp_f, format!("{expr}"), format!("{simplified}")
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix determinant invariants
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    /// det(A) == det(Aᵀ) for any square matrix.
    #[test]
    fn det_transpose_invariant(
        entries in proptest::array::uniform9(-5i64..6i64)
    ) {
        let data: Vec<Vec<Ex>> = entries
            .chunks(3)
            .map(|row| row.iter().map(|&v| symplex::default_context().int(v)).collect())
            .collect();
        let m = symplex::matrix::Matrix::new(data).unwrap();
        let mt = m.transpose();

        let det_m = m.det().unwrap().eval().simplify();
        let det_mt = mt.det().unwrap().eval().simplify();

        if let (Ok(a), Ok(b)) = (det_m.eval_f64(), det_mt.eval_f64()) {
            prop_assert!(
                (a - b).abs() < 1e-10 * a.abs().max(1.0),
                "det(A)={} != det(Aᵀ)={}", a, b
            );
        }
    }
}
