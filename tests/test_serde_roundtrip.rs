//! Comprehensive serde round-trip tests for all ExprTree variants.

use symplex::prelude::*;
use symplex::tree::ExprTree;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Assert that serializing to tree and back preserves the display form.
fn assert_tree_roundtrip(expr: &Ex) {
    let tree = expr.to_tree();
    let ctx = symplex::default_context();
    let back = ctx.from_tree(&tree);
    assert_eq!(
        format!("{expr}"),
        format!("{back}"),
        "tree round-trip failed for expression"
    );
}

/// Assert that serializing to JSON and back preserves the display form.
fn assert_json_roundtrip(expr: &Ex) {
    let json = expr.to_json();
    let ctx = symplex::default_context();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(
        format!("{expr}"),
        format!("{back}"),
        "JSON round-trip failed for expression (json was: {json})"
    );
}

/// Combined: tree + JSON round-trip.
fn assert_full_roundtrip(expr: &Ex) {
    assert_tree_roundtrip(expr);
    assert_json_roundtrip(expr);
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Atoms: integer, rational, symbol, constants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_integer() {
    let expr = symplex::int(42);
    let tree = expr.to_tree();
    match &tree {
        ExprTree::Num { numer, denom } => {
            assert_eq!(numer, "42");
            assert_eq!(denom, "1");
        }
        other => panic!("expected Num, got {other:?}"),
    }
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_negative_integer() {
    let expr = symplex::int(-17);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_rational() {
    let expr = symplex::rational(3, 7);
    let tree = expr.to_tree();
    match &tree {
        ExprTree::Num { numer, denom } => {
            assert_eq!(numer, "3");
            assert_eq!(denom, "7");
        }
        other => panic!("expected Num, got {other:?}"),
    }
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_symbol() {
    let expr = symplex::var("alpha");
    let tree = expr.to_tree();
    match &tree {
        ExprTree::Symbol { name } => assert_eq!(name, "alpha"),
        other => panic!("expected Symbol, got {other:?}"),
    }
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_pi() {
    let expr = symplex::pi();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::Pi);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_e() {
    let expr = symplex::e();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::E);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_i_unit() {
    let expr = symplex::i_unit();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::ImaginaryUnit);
    assert_full_roundtrip(&expr);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Special values: infinity, neg-infinity, complex-infinity, NaN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_infinity() {
    let expr = symplex::infinity();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::Infinity);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_neg_infinity() {
    let expr = symplex::neg_infinity();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::NegInfinity);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_complex_infinity() {
    // ComplexInfinity has no top-level constructor; build via from_tree.
    let ctx = symplex::default_context();
    let expr = ctx.from_tree(&ExprTree::ComplexInfinity);
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::ComplexInfinity);
    // Also verify JSON path
    let json = expr.to_json();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn roundtrip_nan() {
    let ctx = Context::new();
    let expr = ctx.nan();
    let tree = expr.to_tree();
    assert_eq!(tree, ExprTree::NaN);
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{expr}"), format!("{back}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Unary functions — all trig / hyp / misc
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_all_trig_and_hyp() {
    let x = symplex::var("x");

    // Standard trig
    assert_full_roundtrip(&x.sin());
    assert_full_roundtrip(&x.cos());
    assert_full_roundtrip(&x.tan());

    // Inverse trig
    assert_full_roundtrip(&x.asin());
    assert_full_roundtrip(&x.acos());
    assert_full_roundtrip(&x.atan());

    // Hyperbolic
    assert_full_roundtrip(&x.sinh());
    assert_full_roundtrip(&x.cosh());
    assert_full_roundtrip(&x.tanh());

    // Inverse hyperbolic
    assert_full_roundtrip(&x.asinh());
    assert_full_roundtrip(&x.acosh());
    assert_full_roundtrip(&x.atanh());
}

#[test]
fn roundtrip_exp_ln_sqrt_abs_sign() {
    let x = symplex::var("x");

    assert_full_roundtrip(&x.exp());
    assert_full_roundtrip(&x.ln());
    assert_full_roundtrip(&x.sqrt());
    assert_full_roundtrip(&x.abs());
    assert_full_roundtrip(&x.sign());
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Compound expressions: Add, Mul, Pow, Neg
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_polynomial() {
    let x = symplex::var("x");
    // x^3 + 2*x^2 - 5*x + 7
    let expr = x.powi(3) + symplex::int(2) * x.powi(2) - symplex::int(5) * &x + symplex::int(7);
    assert_full_roundtrip(&expr);
}

#[test]
fn roundtrip_nested_functions() {
    let x = symplex::var("x");
    // sin(cos(exp(x)))
    let expr = x.exp().cos().sin();
    assert_full_roundtrip(&expr);

    // ln(x^2 + 1)
    let expr2 = (x.powi(2) + symplex::int(1)).ln();
    assert_full_roundtrip(&expr2);
}

#[test]
fn roundtrip_negation() {
    let x = symplex::var("x");
    let expr = -&x;
    assert_full_roundtrip(&expr);

    // nested negation in larger expression
    let expr2 = -&(x.sin());
    assert_full_roundtrip(&expr2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Factorial → Apply variant
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_factorial() {
    let n = symplex::int(5);
    let expr = n.factorial();
    let tree = expr.to_tree();
    // factorial is serialized as Apply { name: "factorial", args }
    match &tree {
        ExprTree::Apply { name, args } => {
            assert_eq!(name, "factorial");
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected Apply(factorial), got {other:?}"),
    }
    // Factorial → Apply is a known one-way mapping (ExprNode::Factorial
    // serializes to ExprTree::Apply, and deserializes back as a generic
    // ExprNode::Apply), so the display form changes ("5!" → "factorial(5)").
    // Verify instead that the *tree* itself survives a JSON round-trip.
    let json = serde_json::to_string(&tree).unwrap();
    let tree_back: ExprTree = serde_json::from_str(&json).unwrap();
    assert_eq!(tree, tree_back, "ExprTree should survive JSON round-trip");

    // Also verify the deserialized Ex is usable.
    let ctx = symplex::default_context();
    let back = ctx.from_tree(&tree);
    assert_eq!(format!("{back}"), "factorial(5)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Boolean / Logic / Piecewise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn roundtrip_bool_true_false() {
    let ctx = symplex::default_context();

    let t = ctx.from_tree(&ExprTree::BoolTrue);
    let tree_t = t.to_tree();
    assert_eq!(tree_t, ExprTree::BoolTrue);

    let f = ctx.from_tree(&ExprTree::BoolFalse);
    let tree_f = f.to_tree();
    assert_eq!(tree_f, ExprTree::BoolFalse);
}

#[test]
fn roundtrip_comparison_and_logic() {
    let x = symplex::var("x");
    let zero = symplex::int(0);

    // x > 0
    let gt: BoolEx = x.gt(&zero);
    let gt_ex = gt.as_ex();
    assert_full_roundtrip(&gt_ex);

    // x >= 0
    let ge: BoolEx = x.ge(&zero);
    let ge_ex = ge.as_ex();
    assert_full_roundtrip(&ge_ex);

    // !(x > 0)
    let not_gt = gt.not();
    let not_ex = not_gt.as_ex();
    assert_full_roundtrip(&not_ex);

    // (x > 0) & (x >= 0) — And variant
    let and_ex = gt.and(&ge).as_ex();
    assert_full_roundtrip(&and_ex);

    // (x > 0) | (x >= 0) — Or variant
    let or_ex = gt.or(&ge).as_ex();
    assert_full_roundtrip(&or_ex);
}

#[test]
fn roundtrip_piecewise() {
    let x = symplex::var("x");
    let zero = symplex::int(0);
    let cond = x.gt(&zero);
    let neg_cond = cond.not();

    let pw = Ex::piecewise(&[(&x, &cond), (&(-&x), &neg_cond)]);
    assert_full_roundtrip(&pw);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. JSON string round-trips (to_json / to_json_pretty / from_json)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn json_roundtrip_complex_expression() {
    let x = symplex::var("x");
    let expr = x.sin().powi(2) + x.cos();
    let json = expr.to_json();
    assert!(!json.is_empty());
    let ctx = symplex::default_context();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(format!("{expr}"), format!("{back}"));
}

#[test]
fn json_pretty_roundtrip() {
    let expr = symplex::var("x").exp();
    let json = expr.to_json_pretty();
    assert!(json.contains('\n'), "pretty JSON should contain newlines");
    let ctx = symplex::default_context();
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(format!("{expr}"), format!("{back}"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Error paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn json_deserialize_garbage_fails() {
    let ctx = Context::new();
    let result = ctx.from_json("not valid json at all");
    assert!(result.is_err());
}

#[test]
fn json_deserialize_wrong_structure() {
    let ctx = Context::new();
    let result = ctx.from_json(r#"{"type": "UnknownVariant"}"#);
    assert!(result.is_err());
}

#[test]
fn json_deserialize_empty_object_fails() {
    let ctx = Context::new();
    let result = ctx.from_json("{}");
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. ExprTree direct serde (bypassing the Ex convenience methods)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exprtree_serde_json_direct() {
    // Build an ExprTree by hand, serialize, deserialize, compare.
    let tree = ExprTree::Add {
        terms: vec![
            ExprTree::Num {
                numer: "1".into(),
                denom: "1".into(),
            },
            ExprTree::Pow {
                base: Box::new(ExprTree::Symbol { name: "x".into() }),
                exp: Box::new(ExprTree::Num {
                    numer: "2".into(),
                    denom: "1".into(),
                }),
            },
        ],
    };

    let json = serde_json::to_string(&tree).unwrap();
    let back: ExprTree = serde_json::from_str(&json).unwrap();
    assert_eq!(tree, back);
}
