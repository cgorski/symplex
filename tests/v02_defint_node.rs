//! v0.2 `DefiniteIntegral(body, var, lo, hi)` node: the bounded, unevaluated
//! form returned by definite integration when no closed form exists.
//!
//! Covers construction folds, every output format and its round trip,
//! scoping (`free_symbols`, `subs`), the Leibniz rule in `diff`, numeric
//! evaluation by quadrature, the definite-integration fallback path, nested
//! integrals, and rejection by the numeric back-ends.

use symplex::expr::ExprType;
use symplex::prelude::*;

fn ctx_x() -> (Context, Ex) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    (ctx, x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Construction: `definite_integral_node` and its cheap folds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn node_is_built_without_evaluation() {
    let (ctx, x) = ctx_x();
    // ∫₀¹ x² dx has a closed form, but the node constructor must not compute it.
    let node = x
        .powi(2)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert!(node.is_definite_integral());
    assert!(node.has_unevaluated());
    assert_eq!(node.expr_type(), ExprType::Unevaluated);
    assert_eq!(format!("{node}"), "Integral(x^2, x, 0, 1)");
}

#[test]
fn equal_bounds_fold_to_zero() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = x.sin().definite_integral_node(&x, &t, &t);
    assert_eq!(format!("{node}"), "0");
    assert!(!node.is_definite_integral());
}

#[test]
fn zero_integrand_folds_to_zero_even_over_infinite_interval() {
    let (ctx, x) = ctx_x();
    let node = ctx
        .int(0)
        .definite_integral_node(&x, &ctx.int(0), &ctx.infinity());
    assert_eq!(format!("{node}"), "0");
}

#[test]
fn constant_integrand_folds_to_width_times_constant() {
    let (ctx, x) = ctx_x();
    let (a, b, c) = (ctx.symbol("a"), ctx.symbol("b"), ctx.symbol("c"));
    let node = c.definite_integral_node(&x, &a, &b);
    assert!(!node.is_definite_integral());
    let expected = &c * &(&b - &a);
    assert_eq!(node, expected, "got {node}");
}

#[test]
fn constant_integrand_over_infinite_interval_is_left_as_node() {
    let (ctx, x) = ctx_x();
    let c = ctx.symbol("c");
    // Divergence depends on sign(c): the constructor must not decide.
    let node = c.definite_integral_node(&x, &ctx.int(0), &ctx.infinity());
    assert!(node.is_definite_integral(), "got {node}");
}

#[test]
fn reversed_numeric_bounds_are_flipped_with_a_sign() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(1), &ctx.int(0));
    assert_eq!(format!("{node}"), "-Integral(x^x, x, 0, 1)");
    let flipped = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(node, -&flipped);
}

#[test]
fn symbolic_bounds_are_not_reordered() {
    let (ctx, x) = ctx_x();
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let node = x.pow(&x).definite_integral_node(&x, &b, &a);
    assert_eq!(format!("{node}"), "Integral(x^x, x, b, a)");
}

#[test]
fn is_definite_integral_only_inspects_the_root() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let sum = &node + 1;
    assert!(!sum.is_definite_integral());
    assert!(sum.has_unevaluated());
    assert!(!x.sin().is_definite_integral());
}

#[test]
fn eval_rebuilds_children_but_does_not_integrate() {
    let (ctx, x) = ctx_x();
    // Upper bound 1 + 1 evaluates to 2; the body cos(0)·x^x evaluates to x^x.
    let body = &ctx.int(0).cos() * &x.pow(&x);
    let hi = &ctx.int(1) + &ctx.int(1);
    let node = body.definite_integral_node(&x, &ctx.int(0), &hi);
    let evaled = node.eval();
    assert_eq!(format!("{evaled}"), "Integral(x^x, x, 0, 2)");
    assert!(evaled.is_definite_integral());
}

#[test]
fn eval_applies_equal_bounds_fold_once_bounds_become_equal() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = x.pow(&x).definite_integral_node(&x, &ctx.int(1), &t);
    assert!(node.is_definite_integral());
    let at_one = node.subs_i64(&t, 1);
    assert_eq!(format!("{at_one}"), "0");
}

// ═══════════════════════════════════════════════════════════════════════════
// Output formats and round trips
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn display_is_the_four_argument_integral_form() {
    let (ctx, x) = ctx_x();
    let node = (&x.sin() / &x).definite_integral_node(&x, &ctx.int(0), &ctx.infinity());
    assert_eq!(format!("{node}"), "Integral(sin(x)/x, x, 0, oo)");
}

#[test]
fn display_round_trips_through_parse() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = (&x.exp() * &t).definite_integral_node(&x, &ctx.int(0), &t.powi(2));
    let back = ctx.parse(&format!("{node}")).unwrap();
    assert_eq!(back, node, "parsed {back} != {node}");
    assert!(back.is_definite_integral());
}

#[test]
fn parse_accepts_four_argument_integral_directly() {
    let ctx = Context::new();
    let e = ctx.parse("Integral(x^x, x, 0, 1)").unwrap();
    assert!(e.is_definite_integral());
    assert_eq!(format!("{e}"), "Integral(x^x, x, 0, 1)");
    // The constructor folds apply at parse time too.
    let z = ctx.parse("Integral(x^x, x, 2, 2)").unwrap();
    assert_eq!(format!("{z}"), "0");
}

#[test]
fn parse_accepts_two_argument_indefinite_integral() {
    let ctx = Context::new();
    let e = ctx.parse("Integral(sin(x), x)").unwrap();
    assert_eq!(e.expr_type(), ExprType::Integral);
    assert_eq!(format!("{e}"), "Integral(sin(x), x)");
}

#[test]
fn latex_uses_bounded_integral_sign() {
    let (ctx, x) = ctx_x();
    let node = (-x.powi(2))
        .exp()
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(
        node.to_latex(),
        r"\int_{0}^{1} \exp\left(-x^{2}\right)\, dx"
    );
    let t = ctx.symbol("t");
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t.powi(2));
    assert_eq!(node.to_latex(), r"\int_{0}^{t^{2}} \sin\left(x\right)\, dx");
    // Same `\, d` spacing as the indefinite form.
    let indef = x.sin().integrate(&x);
    if indef.expr_type() == ExprType::Integral {
        assert!(indef.to_latex().ends_with(r"\, dx"));
    }
}

#[test]
fn pretty_falls_back_to_display_form() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let unicode = node.pretty();
    let ascii = node.pretty_ascii();
    assert!(
        unicode.contains("Integral(x^x, x, 0, 1)"),
        "got {unicode:?}"
    );
    assert!(ascii.contains("Integral(x^x, x, 0, 1)"), "got {ascii:?}");
}

#[test]
fn json_round_trip_preserves_bounds() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = (&x * &t).sin().definite_integral_node(&x, &ctx.int(0), &t);
    let json = node.to_json().unwrap();
    assert!(json.contains("DefiniteIntegral"), "got {json}");
    let back = ctx.from_json(&json).unwrap();
    assert_eq!(back, node, "json {json} came back as {back}");
}

#[test]
fn tree_names_the_variant() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let tree = node.to_tree();
    let dbg = format!("{tree:?}");
    assert!(dbg.starts_with("DefiniteIntegral"), "got {dbg}");
    assert_eq!(ctx.from_tree(&tree), node);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scoping: `free_symbols`, `subs`, `has_unevaluated`
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn has_unevaluated_reports_the_node_anywhere_in_the_tree() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert!(node.has_unevaluated());
    assert!((&node * 2 + 1).sin().has_unevaluated());
    assert!(!x.sin().has_unevaluated());
}

#[test]
fn free_symbols_exclude_the_bound_variable_and_include_bound_symbols() {
    let (ctx, x) = ctx_x();
    let (a, t) = (ctx.symbol("a"), ctx.symbol("t"));
    let node = (&x * &t).sin().definite_integral_node(&x, &a, &t.powi(2));
    let mut names: Vec<String> = node.free_symbols().iter().map(|s| format!("{s}")).collect();
    names.sort();
    assert_eq!(names, ["a", "t"]);
}

#[test]
fn free_symbols_same_symbol_bound_inside_and_free_outside() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let e = &node + &x;
    let names: Vec<String> = e.free_symbols().iter().map(|s| format!("{s}")).collect();
    assert_eq!(names, ["x"]);
    // And the node alone has no free symbols at all.
    assert!(node.free_symbols().is_empty());
}

#[test]
fn free_symbols_bound_variable_appearing_in_a_bound_is_free() {
    let (ctx, x) = ctx_x();
    // ∫₀ˣ x² dx: the x in the upper bound is the outer x.
    let node = x.powi(2).definite_integral_node(&x, &ctx.int(0), &x);
    let names: Vec<String> = node.free_symbols().iter().map(|s| format!("{s}")).collect();
    assert_eq!(names, ["x"]);
}

#[test]
fn subs_does_not_touch_the_bound_variable_inside_the_body() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = x.pow(&x).definite_integral_node(&x, &ctx.int(0), &t);
    let after = node.subs_i64(&x, 3);
    assert_eq!(after, node, "got {after}");
}

#[test]
fn subs_rewrites_bounds_that_mention_the_bound_variable() {
    let (ctx, x) = ctx_x();
    // ∫₀ˣ x^x dx with x ↦ 2 is ∫₀² x^x dx.
    let node = x.pow(&x).definite_integral_node(&x, &ctx.int(0), &x);
    let after = node.subs_i64(&x, 2);
    assert_eq!(format!("{after}"), "Integral(x^x, x, 0, 2)");
}

#[test]
fn subs_of_a_free_parameter_reaches_into_the_body() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = (&x * &t)
        .sin()
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let after = node.subs_i64(&t, 2);
    assert_eq!(format!("{after}"), "Integral(sin(2*x), x, 0, 1)");
}

#[test]
fn subs_map_shadows_only_the_bound_key() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // {x ↦ 5, t ↦ 2}: x is bound (kept), t is free (replaced), the bound uses x.
    let node = (&x * &t).sin().definite_integral_node(&x, &ctx.int(0), &x);
    let after = node.subs_map(&[(&x, &ctx.int(5)), (&t, &ctx.int(2))]);
    assert_eq!(format!("{after}"), "Integral(sin(2*x), x, 0, 5)");
}

#[test]
fn subs_outside_the_binder_still_applies() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let e = &node + &x;
    let after = e.subs_i64(&x, 3);
    assert_eq!(format!("{after}"), "Integral(x^x, x, 0, 1) + 3");
}

// ═══════════════════════════════════════════════════════════════════════════
// Differentiation: Leibniz integral rule
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diff_wrt_bound_variable_is_zero() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(format!("{}", node.diff(&x)), "0");
}

#[test]
fn diff_wrt_unrelated_symbol_is_zero() {
    let (ctx, x) = ctx_x();
    let y = ctx.symbol("y");
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(format!("{}", node.diff(&y)), "0");
}

#[test]
fn diff_variable_upper_bound_gives_integrand_at_bound() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // d/dt ∫₀ᵗ sin(x) dx = sin(t)
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t);
    assert_eq!(format!("{}", node.diff(&t)), "sin(t)");
}

#[test]
fn diff_variable_lower_bound_gives_negated_integrand() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // d/dt ∫ₜ¹ x^x dx = −t^t
    let node = x.pow(&x).definite_integral_node(&x, &t, &ctx.int(1));
    assert_eq!(format!("{}", node.diff(&t)), "-t^t");
}

#[test]
fn diff_chain_rule_through_upper_bound() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // d/dt ∫₀^{t²} sin(x) dx = 2t·sin(t²)
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t.powi(2));
    let d = node.diff(&t);
    let expected = &(&ctx.int(2) * &t) * &t.powi(2).sin();
    assert_eq!(d, expected, "got {d}");
}

#[test]
fn diff_leibniz_matches_finite_differences() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // F(t) = ∫₀^{t²} sin(x) dx; F'(1) = 2·sin(1).
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t.powi(2));
    let d = node.diff(&t).subs_i64(&t, 1).eval_f64().unwrap();
    assert!((d - 2.0 * 1f64.sin()).abs() < 1e-9, "symbolic {d}");
    // Central difference on the node itself (evaluated by quadrature).
    let h = 1e-4;
    let f = |v: f64| {
        node.subs(&t, &ctx.from_f64(v).unwrap())
            .eval_f64()
            .unwrap_or_else(|e| panic!("F({v}) failed: {e}"))
    };
    let fd = (f(1.0 + h) - f(1.0 - h)) / (2.0 * h);
    assert!(
        (d - fd).abs() < 1e-6,
        "symbolic {d} vs finite difference {fd}"
    );
}

#[test]
fn diff_parameter_inside_integrand_moves_under_the_integral() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // d/dt ∫₀¹ sin(t·x) dx = ∫₀¹ x·cos(t·x) dx
    let node = (&t * &x)
        .sin()
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let d = node.diff(&t);
    assert!(d.is_definite_integral(), "got {d}");
    let expected = (&x * &(&t * &x).cos()).definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(d, expected, "got {d}");
}

#[test]
fn diff_full_leibniz_rule_all_three_terms() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // d/dt ∫_t^{2t} x·t dx = (2t·t)·2 − (t·t)·1 + ∫_t^{2t} x dx
    let node = (&x * &t).definite_integral_node(&x, &t, &(&ctx.int(2) * &t));
    let d = node.diff(&t);
    let boundary = &(&ctx.int(4) * &t.powi(2)) - &t.powi(2);
    let under = x.definite_integral_node(&x, &t, &(&ctx.int(2) * &t));
    let expected = &boundary + &under;
    assert_eq!(d, expected, "got {d}");
    // Evaluating the leftover integral: 3t² + (4t² − t²)/2 = 9t²/2.
    let total = d.eval_integrals();
    assert!(!total.has_unevaluated(), "got {total}");
    let expected_total = &ctx.rational(9, 2) * &t.powi(2);
    assert_eq!(total.expand(), expected_total, "got {total}");
    // Cross-check: F(t) = ∫_t^{2t} x·t dx = 3t³/2, so F'(t) = 9t²/2.
    let f = node.eval_integrals();
    assert_eq!(f.diff(&t).expand(), expected_total, "F = {f}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric evaluation by quadrature
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn eval_f64_gaussian_on_unit_interval() {
    let (ctx, x) = ctx_x();
    let node = (-x.powi(2))
        .exp()
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let v = node.eval_f64().unwrap();
    assert!((v - 0.746_824_132_812_427_3).abs() < 1e-10, "got {v}");
}

#[test]
fn eval_f64_of_sophomores_dream() {
    let (ctx, x) = ctx_x();
    // ∫₀¹ xˣ dx = Σ (−1)ⁿ⁺¹/nⁿ ≈ 0.7834305107
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let v = node.eval_f64().unwrap();
    assert!((v - 0.783_430_510_712_134_5).abs() < 1e-8, "got {v}");
}

#[test]
fn eval_f64_handles_infinite_bounds() {
    let (ctx, x) = ctx_x();
    let node = (-x.powi(2))
        .exp()
        .definite_integral_node(&x, &ctx.neg_infinity(), &ctx.infinity());
    let v = node.eval_f64().unwrap();
    assert!((v - std::f64::consts::PI.sqrt()).abs() < 1e-9, "got {v}");
}

#[test]
fn eval_f64_node_embedded_in_larger_expression() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let e = &(&node * 2) + 1;
    let v = e.eval_f64().unwrap();
    assert!(
        (v - (2.0 * 0.783_430_510_712_134_5 + 1.0)).abs() < 1e-8,
        "got {v}"
    );
}

#[test]
fn eval_f64_with_symbolic_bound_after_substitution() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = x.pow(&x).definite_integral_node(&x, &ctx.int(0), &t);
    assert!(matches!(
        node.eval_f64(),
        Err(SymplexError::FreeSymbol { .. })
    ));
    let v = node.subs_i64(&t, 1).eval_f64().unwrap();
    assert!((v - 0.783_430_510_712_134_5).abs() < 1e-8, "got {v}");
}

#[test]
fn eval_decimal_beyond_f64_is_not_implemented() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    match node.eval_decimal(30) {
        Err(SymplexError::NotImplemented(msg)) => {
            assert!(msg.contains("definite integral"), "got {msg}");
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
    // At f64 precision the quadrature result is served.
    let s = node.eval_decimal(10).unwrap();
    assert!(s.starts_with("0.78343051"), "got {s}");
}

#[test]
fn eval_f64_divergent_node_is_an_error_not_a_number() {
    let (ctx, x) = ctx_x();
    let node = x
        .powi(-2)
        .definite_integral_node(&x, &ctx.int(-1), &ctx.int(1));
    assert!(node.eval_f64().is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// The definite-integration fallback path
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn unevaluable_integrand_returns_the_node_with_bounds() {
    let (ctx, x) = ctx_x();
    let v = x.pow(&x).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(v.is_definite_integral(), "got {v}");
    assert!(v.has_unevaluated());
    assert_eq!(format!("{v}"), "Integral(x^x, x, 0, 1)");
}

#[test]
fn try_integrate_definite_errors_on_unevaluable_integrand() {
    let (ctx, x) = ctx_x();
    let r = x
        .pow(&x)
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(
        matches!(r, Err(SymplexError::ComputationFailed { .. })),
        "got {r:?}"
    );
}

#[test]
fn fallback_node_evaluates_numerically() {
    let (ctx, x) = ctx_x();
    let v = x.pow(&x).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    let f = v.eval_f64().unwrap();
    assert!((f - 0.783_430_510_712_134_5).abs() < 1e-8, "got {f}");
}

#[test]
fn divergent_integral_keeps_its_bounds() {
    let (ctx, x) = ctx_x();
    let v = x.powi(-2).integrate_definite(&x, &ctx.int(-1), &ctx.int(1));
    assert!(v.is_definite_integral(), "got {v}");
    assert_eq!(format!("{v}"), "Integral(x^(-2), x, -1, 1)");
}

#[test]
fn eval_integrals_resolves_a_node_with_a_closed_form() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t);
    let v = node.eval_integrals();
    assert!(!v.has_unevaluated(), "got {v}");
    assert_eq!(v, &ctx.int(1) - &t.cos(), "got {v}");
}

#[test]
fn eval_integrals_leaves_unevaluable_and_divergent_nodes_alone() {
    let (ctx, x) = ctx_x();
    let n = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(n.eval_integrals(), n);
    let d = x
        .powi(-2)
        .definite_integral_node(&x, &ctx.int(-1), &ctx.int(1));
    assert_eq!(d.eval_integrals(), d);
}

#[test]
fn eval_integrals_works_innermost_first_inside_a_larger_expression() {
    let (ctx, x) = ctx_x();
    let y = ctx.symbol("y");
    // sin(∫₀¹ [∫₀ʸ x dx] dy) = sin(1/6)
    let inner = x.definite_integral_node(&x, &ctx.int(0), &y);
    let outer = inner.definite_integral_node(&y, &ctx.int(0), &ctx.int(1));
    let e = outer.sin().eval_integrals();
    assert_eq!(format!("{e}"), "sin(1/6)");
}

#[test]
fn integrating_a_node_again_is_a_genuine_double_integral() {
    let (ctx, x) = ctx_x();
    let t = ctx.symbol("t");
    // ∫₀ᵗ [∫₀ᵗ sin(x) dx] dx = t·(1 − cos t): the inner is resolved first,
    // then integrated as a constant over the outer range.
    let node = x.sin().definite_integral_node(&x, &ctx.int(0), &t);
    let v = node.integrate_definite(&x, &ctx.int(0), &t);
    assert!(!v.has_unevaluated(), "got {v}");
    assert_eq!(
        v.expand(),
        (&t * &(&ctx.int(1) - &t.cos())).expand(),
        "got {v}"
    );
}

#[test]
fn integrate_definite_resolves_an_inner_node_inside_the_integrand() {
    let (ctx, x) = ctx_x();
    let y = ctx.symbol("y");
    // ∫₀¹ [∫₀ʸ x dx] dy, with the inner integral left formal on purpose.
    let inner = x.definite_integral_node(&x, &ctx.int(0), &y);
    assert!(inner.is_definite_integral());
    let v = inner
        .try_integrate_definite(&y, &ctx.int(0), &ctx.int(1))
        .unwrap();
    assert_eq!(format!("{v}"), "1/6");
}

#[test]
fn nested_integral_via_two_calls() {
    let (ctx, x) = ctx_x();
    let y = ctx.symbol("y");
    // ∫₀¹ ∫₀ʸ x dx dy = ∫₀¹ y²/2 dy = 1/6
    let inner = x.integrate_definite(&x, &ctx.int(0), &y);
    assert_eq!(format!("{inner}"), "1/2*y^2");
    let outer = inner.integrate_definite(&y, &ctx.int(0), &ctx.int(1));
    assert_eq!(format!("{outer}"), "1/6");
}

#[test]
fn nested_unevaluable_inner_stays_formal_and_is_numeric() {
    let (ctx, x) = ctx_x();
    let y = ctx.symbol("y");
    // ∫₀¹ [∫₀¹ x^x dx] dy = ∫₀¹ x^x dx (constant in y).
    let inner = x.pow(&x).integrate_definite(&x, &ctx.int(0), &ctx.int(1));
    assert!(inner.is_definite_integral());
    let outer = inner.integrate_definite(&y, &ctx.int(0), &ctx.int(1));
    assert_eq!(outer, inner, "got {outer}");
    let v = outer.eval_f64().unwrap();
    assert!((v - 0.783_430_510_712_134_5).abs() < 1e-8, "got {v}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric back-ends reject the node
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn compile_rejects_the_node() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let r = (&node + &x).compile(&["x"]);
    assert!(
        matches!(r, Err(SymplexError::NotImplemented(_))),
        "got {r:?}"
    );
}

#[test]
fn to_rust_fn_rejects_the_node() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let r = (&node + &x).to_rust_fn("f", &["x"]);
    assert!(
        matches!(r, Err(SymplexError::NotImplemented(_))),
        "got {r:?}"
    );
}

#[test]
fn to_c_fn_rejects_the_node() {
    let (ctx, x) = ctx_x();
    let node = x
        .pow(&x)
        .definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    let r = (&node + &x).to_c_fn("f", &["x"]);
    assert!(
        matches!(r, Err(SymplexError::NotImplemented(_))),
        "got {r:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumptions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn node_with_real_body_and_bounds_is_real() {
    let ctx = Context::new();
    let x = ctx.symbol_with("x", &[Assumption::Real]);
    let t = ctx.symbol_with("t", &[Assumption::Real]);
    let node = x.sin().exp().definite_integral_node(&x, &ctx.int(0), &t);
    assert_eq!(node.is_real(), Some(true));
    // Unknown symbol: undecided, never asserted.
    let z = ctx.symbol("z");
    let node = (&x * &z).definite_integral_node(&x, &ctx.int(0), &ctx.int(1));
    assert_eq!(node.is_real(), None);
}
