//! symplex 0.2 — public rewrite-rule engine (`Rule`, `RuleSet`, `Ex::rewrite*`,
//! `simplify_with_rules`, `simplify_traced`) and the `rule!` macro bridge.
//!
//! Every rewrite is checked for value preservation at several rational
//! points in addition to its structural assertion.

use symplex::macros::{RewriteOpts, RewriteStrategy, Rule, RuleSet};
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Sample points (p/q) used for numeric value-preservation checks.
const POINTS: &[(i64, i64)] = &[(3, 7), (-5, 3), (11, 4), (1, 9), (-13, 6), (7, 2)];

/// Assert `a` and `b` agree numerically at every sample point for every free symbol.
fn assert_same_value(a: &Ex, b: &Ex, label: &str) {
    assert_same_value_at(a, b, POINTS, label);
}

/// Like [`assert_same_value`] but only at positive sample points (for
/// identities such as `ln a + ln b = ln(ab)` that need positive arguments).
fn assert_same_value_positive(a: &Ex, b: &Ex, label: &str) {
    assert_same_value_at(
        a,
        b,
        &[(3, 7), (5, 3), (11, 4), (1, 9), (13, 6), (7, 2)],
        label,
    );
}

fn assert_same_value_at(a: &Ex, b: &Ex, points: &[(i64, i64)], label: &str) {
    let ctx = a.context();
    let mut syms = a.free_symbols();
    for s in b.free_symbols() {
        if !syms.contains(&s) {
            syms.push(s);
        }
    }
    let mut checked = 0;
    for (i, &(p, q)) in points.iter().enumerate() {
        let mut ea = a.clone();
        let mut eb = b.clone();
        for (j, s) in syms.iter().enumerate() {
            // Different value per symbol: rotate through the point list.
            let (pp, qq) = points[(i + j) % points.len()];
            let v = ctx.rational(pp + p, qq + q);
            ea = ea.subs(s, &v);
            eb = eb.subs(s, &v);
        }
        match (ea.eval_f64(), eb.eval_f64()) {
            (Ok(va), Ok(vb)) => {
                assert!(
                    (va - vb).abs() <= 1e-9 * (1.0 + va.abs().max(vb.abs())),
                    "{label}: value mismatch at point {i}: {va} vs {vb}\n  before: {a}\n  after:  {b}"
                );
                checked += 1;
            }
            (Err(_), Err(_)) => {} // both undefined at this point (e.g. domain)
            (ra, rb) => panic!("{label}: one side failed to evaluate: {ra:?} vs {rb:?}"),
        }
    }
    assert!(
        checked >= 2,
        "{label}: too few evaluable points ({checked})"
    );
}

fn s(e: &Ex) -> String {
    format!("{e}")
}

// ═══════════════════════════════════════════════════════════════════════════
// Rule construction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rule_new_reports_name_and_wildcards() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));
    let r = Rule::new("swap", &a.pow(&b), &b.pow(&a));
    assert_eq!(r.name(), "swap");
    assert_eq!(r.wildcards(), vec!["a_".to_string(), "b_".to_string()]);
}

#[test]
fn rule_try_new_rejects_unbound_rhs_wildcard() {
    let ctx = Context::new();
    let (a, b) = (ctx.symbol("a_"), ctx.symbol("b_"));
    let err = Rule::try_new("bad", &a.sin(), &b).expect_err("must fail");
    assert!(format!("{err}").contains("b_"), "{err}");
}

#[test]
fn rule_try_new_rejects_bare_wildcard_lhs() {
    let ctx = Context::new();
    let a = ctx.symbol("a_");
    assert!(Rule::try_new("bad", &a, &a.sin()).is_err());
}

#[test]
fn rule_matches_returns_bindings_by_name() {
    let ctx = Context::new();
    let (x, y, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("a_"),
        ctx.symbol("b_"),
    );
    let r = Rule::new("r", &(&a.sin() * &b.cos()), &a);
    let bnd = r.matches(&(&x.sin() * &y.cos())).expect("should match");
    assert_eq!(bnd.get("a_"), Some(&x));
    assert_eq!(bnd.get("b_"), Some(&y));
    assert!(bnd.get("c_").is_none());
    assert_eq!(bnd.iter().count(), 2);
    assert!(!bnd.is_empty());
}

#[test]
fn rule_matches_is_whole_node_only() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let r = Rule::new("r", &a.sin(), &a);
    assert!(r.matches(&x.sin()).is_some());
    assert!(
        r.matches(&(&x.sin() + 1)).is_none(),
        "matches() does not descend"
    );
}

#[test]
fn rule_apply_at_root_only() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let r = Rule::new("r", &a.ln().exp(), &a);
    assert_eq!(r.apply(&x.ln().exp()), Some(x.clone()));
    // Not at the root → None (apply does not traverse).
    assert_eq!(r.apply(&x.ln().exp().sin()), None);
}

#[test]
fn rule_is_send_sync_and_clone() {
    fn assert_send_sync<T: Send + Sync + Clone>(_: &T) {}
    let ctx = Context::new();
    let a = ctx.symbol("a_");
    let r = Rule::new_with_guard("g", &a.sin(), &a, |_| true);
    assert_send_sync(&r);
    let rs = RuleSet::from_rules(vec![r]);
    assert_send_sync(&rs);
}

// ═══════════════════════════════════════════════════════════════════════════
// Template rewriting
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rewrite_simple_unary_identity() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let rules = RuleSet::from_rules(vec![Rule::new("exp_ln", &a.ln().exp(), &a)]);
    let expr = &(&x + 1).ln().exp() * 3;
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "3*x + 3");
    assert_same_value(&expr, &out, "exp_ln");
}

#[test]
fn rewrite_pythagorean_inside_larger_sum() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let pyth = Rule::new("pyth", &(&a.sin().powi(2) + &a.cos().powi(2)), &ctx.int(1));
    let rules = RuleSet::from_rules(vec![pyth]);
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x + 3;
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "x + 4");
    assert_same_value(&expr, &out, "pyth partial");
}

#[test]
fn rewrite_two_wildcards_in_product_terms() {
    // sin(a_)cos(b_) + cos(a_)sin(b_) → sin(a_ + b_), with terms in any order.
    let ctx = Context::new();
    let (x, y, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("a_"),
        ctx.symbol("b_"),
    );
    let lhs = &a.sin() * &b.cos() + &a.cos() * &b.sin();
    let rules = RuleSet::from_rules(vec![Rule::new("sin_add", &lhs, &(&a + &b).sin())]);
    let expr = &y.cos() * &x.sin() + &x.cos() * &y.sin();
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "sin(x + y)");
    assert_same_value(&expr, &out, "sin_add");
    // Mismatched arguments must not fire.
    let z = ctx.symbol("z");
    let bad = &y.cos() * &x.sin() + &x.cos() * &z.sin();
    assert_eq!(bad.rewrite(&rules), bad);
}

#[test]
fn rewrite_structured_terms_match_a_subset_of_a_sum() {
    // ln(a_) + ln(b_) → ln(a_·b_): both pattern terms are structured (the
    // wildcards sit inside `ln`), so they pick two logs out of the sum and
    // the leftover `5` is re-attached.
    let ctx = Context::new();
    let (x, y, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("a_"),
        ctx.symbol("b_"),
    );
    let rules = RuleSet::from_rules(vec![Rule::new(
        "ln_add",
        &(&a.ln() + &b.ln()),
        &(&a * &b).ln(),
    )]);
    let expr = &x.ln() + &y.ln() + 5;
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "ln(x*y) + 5");
    // ln a + ln b = ln(ab) is an identity for positive a, b.
    assert_same_value_positive(&expr, &out, "ln_add");
}

#[test]
fn rewrite_last_plain_wildcard_takes_the_rest() {
    // a_ + b_ against x + y + z: one wildcard gets a term, the other the rest.
    let ctx = Context::new();
    let (x, y, z, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("a_"),
        ctx.symbol("b_"),
    );
    // Rewrite a_ + b_ → f-like marker: (a_)·(b_) so we can inspect the split.
    let r = Rule::new("split", &(&a + &b), &(&a * &b));
    let expr = &x + &y + &z;
    let out = r.apply(&expr).expect("must match");
    // The product of one term and the sum of the other two.
    let str_out = s(&out);
    assert!(
        str_out == "x*(y + z)" || str_out == "y*(x + z)" || str_out == "z*(x + y)",
        "got {str_out}"
    );
}

#[test]
fn rewrite_sequence_wildcard_binds_rest_possibly_empty() {
    let ctx = Context::new();
    let (x, y, a, rest) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("a_"),
        ctx.symbol("rest__"),
    );
    // sin(a_)^2 + rest__ → 1 - cos(a_)^2 + rest__
    let lhs = &a.sin().powi(2) + &rest;
    let rhs = &(1 - &a.cos().powi(2)) + &rest;
    let rules = RuleSet::from_rules(vec![Rule::new("sin_sq", &lhs, &rhs)]);

    let expr = &x.sin().powi(2) + &y + 2;
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "y - cos(x)^2 + 3");
    assert_same_value(&expr, &out, "seq rest");

    // Empty rest: a single term matches with rest__ = 0.
    let lone = x.sin().powi(2);
    let out2 = lone.rewrite(&rules);
    assert_eq!(s(&out2), "-cos(x)^2 + 1");
    assert_same_value(&lone, &out2, "seq rest empty");
}

#[test]
fn rewrite_mul_numeric_coefficient_splitting() {
    // 2·sin(a_)·cos(a_) → sin(2a_) also fires on 6·sin(x)·cos(x) → 3·sin(2x).
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let lhs = &(&a.sin() * &a.cos()) * 2;
    let rules = RuleSet::from_rules(vec![Rule::new("sin_double", &lhs, &(&a * 2).sin())]);
    let expr = &(&x.sin() * &x.cos()) * 6;
    let out = expr.rewrite(&rules);
    assert_eq!(s(&out), "3*sin(2*x)");
    assert_same_value(&expr, &out, "sin_double coeff");
    // Coefficient 1 is *not* split (conservative).
    let plain = &x.sin() * &x.cos();
    assert_eq!(plain.rewrite(&rules), plain);
}

#[test]
fn rewrite_nested_add_must_be_consumed_entirely() {
    // sin(a_ + b_) with two plain wildcards matches sin(x + y + z) (b_ = y + z),
    // but sin(a_ + 1) must not match sin(x + y + 2).
    let ctx = Context::new();
    let (x, y, z, a, b) = (
        ctx.symbol("x"),
        ctx.symbol("y"),
        ctx.symbol("z"),
        ctx.symbol("a_"),
        ctx.symbol("b_"),
    );
    let r1 = Rule::new("r1", &(&a + &b).sin(), &(&a + &b).cos());
    let e1 = (&x + &y + &z).sin();
    assert_eq!(s(&r1.apply(&e1).unwrap()), "cos(x + y + z)");

    let r2 = Rule::new("r2", &(&a + 1).sin(), &a.cos());
    let e2 = (&x + &y + 2).sin();
    assert_eq!(
        r2.apply(&e2),
        None,
        "nested Add without absorber must not match partially"
    );
    let e3 = (&x + &y + 1).sin();
    assert_eq!(s(&r2.apply(&e3).unwrap()), "cos(x + y)");
}

#[test]
fn rewrite_repeated_wildcard_requires_consistency() {
    let ctx = Context::new();
    let (x, y, a) = (ctx.symbol("x"), ctx.symbol("y"), ctx.symbol("a_"));
    let r = Rule::new("same", &(&a.sin() * &a.cos()), &((&a * 2).sin() / 2));
    assert!(r.apply(&(&x.sin() * &x.cos())).is_some());
    assert!(r.apply(&(&x.sin() * &y.cos())).is_none());
}

#[test]
fn rewrite_no_match_returns_identical_handle() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let rules = RuleSet::from_rules(vec![Rule::new("r", &a.sin(), &a.cos())]);
    let expr = &x.exp() + 1;
    assert_eq!(expr.rewrite(&rules), expr);
    assert_eq!(expr.rewrite_once(&rules), expr);
    assert!(expr.rewrite_traced(&rules).1.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Guards and closures
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn guard_can_inspect_bindings_with_ex_methods() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // abs(a_) → a_ only when a_ is known positive.
    let r = Rule::new_with_guard("abs_pos", &a.abs(), &a, |b| {
        b.get("a_").and_then(|v| v.is_positive()) == Some(true)
    });
    let rules = RuleSet::from_rules(vec![r]);
    assert_eq!(x.abs().rewrite(&rules), x.abs());
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    assert_eq!(p.abs().rewrite(&rules), p);
}

#[test]
fn guard_rejection_tries_other_ac_assignments() {
    // a_ + b_ → b_ with a guard requiring a_ to be a number: on x + 2 the
    // first AC assignment might bind a_ = x; the engine must try a_ = 2.
    let ctx = Context::new();
    let (x, a, b) = (ctx.symbol("x"), ctx.symbol("a_"), ctx.symbol("b_"));
    let r = Rule::new_with_guard("drop_num", &(&a + &b), &b, |bnd| {
        bnd.get("a_")
            .is_some_and(|v| v.expr_type() == ExprType::Number)
    });
    let out = r.apply(&(&x + 2)).expect("guard should accept a_ = 2");
    assert_eq!(out, x);
}

#[test]
fn new_fn_rule_computes_rhs() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // sin(a_) → a_ - a_^3/6 (Taylor) computed in a closure.
    let r = Rule::new_fn("taylor", &a.sin(), |b| {
        let v = b.get("a_")?;
        Some(v - &v.powi(3) / 6)
    });
    let rules = RuleSet::from_rules(vec![r]);
    let out = (&x * 2).sin().rewrite(&rules);
    assert_eq!(s(&out), "-4/3*x^3 + 2*x");
}

#[test]
fn new_fn_returning_none_means_no_match() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let r = Rule::new_fn("never", &a.sin(), |_| None);
    let rules = RuleSet::from_rules(vec![r]);
    assert_eq!(x.sin().rewrite(&rules), x.sin());
}

// ═══════════════════════════════════════════════════════════════════════════
// RuleSet
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ruleset_order_first_match_wins() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let r1 = Rule::new("to_cos", &a.sin(), &a.cos());
    let r2 = Rule::new("to_tan", &a.sin(), &a.tan());
    let rules = RuleSet::new().with(r1.clone()).with(r2.clone());
    assert_eq!(s(&x.sin().rewrite_once(&rules)), "cos(x)");
    let rules_rev = RuleSet::from_rules(vec![r2, r1]);
    assert_eq!(s(&x.sin().rewrite_once(&rules_rev)), "tan(x)");
}

#[test]
fn ruleset_collection_api() {
    let ctx = Context::new();
    let a = ctx.symbol("a_");
    let mut rules: RuleSet = vec![Rule::new("r1", &a.sin(), &a)].into_iter().collect();
    assert_eq!(rules.len(), 1);
    rules.push(Rule::new("r2", &a.cos(), &a));
    let more = RuleSet::from(vec![Rule::new("r3", &a.tan(), &a)]);
    rules.extend(&more);
    assert_eq!(rules.len(), 3);
    assert_eq!(rules[2].name(), "r3");
    assert_eq!(
        rules
            .iter()
            .map(|r| r.name().to_string())
            .collect::<Vec<_>>(),
        ["r1", "r2", "r3"]
    );
    assert!(RuleSet::new().is_empty());
}

#[test]
fn ruleset_standard_reproduces_builtin_rules() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let std = RuleSet::standard(&ctx);
    assert!(std.len() >= 20);
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x.ln().exp();
    let out = expr.rewrite(&std);
    assert_eq!(s(&out), "x + 1");
}

// ═══════════════════════════════════════════════════════════════════════════
// rule! macro bridge
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn macro_rules_end_to_end() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let raw = ctx.with_arena_mut(|arena| {
        vec![
            rule!(arena, "pythagorean", sin(w_)^2 + cos(w_)^2 => 1),
            rule!(arena, "exp_ln", exp(ln(w_)) => w_),
        ]
    });
    let rules = RuleSet::from_macro_rules(&ctx, raw);
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].name(), "pythagorean");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &(&x + 1).ln().exp();
    let (out, steps) = expr.rewrite_traced(&rules);
    assert_eq!(s(&out), "x + 2");
    let names: Vec<&str> = steps.iter().map(|st| st.rule_name.as_str()).collect();
    assert!(
        names.contains(&"pythagorean") && names.contains(&"exp_ln"),
        "{names:?}"
    );
    assert_same_value(&expr, &out, "macro rules");
}

#[test]
fn macro_rule_single_conversion() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let raw = ctx.with_arena_mut(|arena| rule!(arena, "sqrt_sq", sqrt(w_^2) => abs(w_)));
    let rule = Rule::from_macro_rule(&ctx, raw);
    assert_eq!(rule.name(), "sqrt_sq");
    assert_eq!(s(&rule.apply(&x.powi(2).sqrt()).unwrap()), "abs(x)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Traces and strategies
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn rewrite_traced_records_each_step_in_order() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let rules = RuleSet::from_rules(vec![
        Rule::new("exp_ln", &a.ln().exp(), &a),
        Rule::new("sin_asin", &a.asin().sin(), &a),
    ]);
    let expr = x.ln().exp().asin().sin();
    let (out, steps) = expr.rewrite_traced(&rules);
    assert_eq!(out, x);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].rule_name, "exp_ln");
    assert_eq!(s(&steps[0].before), "exp(ln(x))");
    assert_eq!(s(&steps[0].after), "x");
    assert_eq!(steps[1].rule_name, "sin_asin");
}

#[test]
fn rewrite_once_is_a_single_pass() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    let rules = RuleSet::from_rules(vec![Rule::new("dbl", &a.sin(), &(&a.sin() * 2))]);
    assert_eq!(s(&x.sin().rewrite_once(&rules)), "2*sin(x)");
    let opts = RewriteOpts::default().max_iterations(3);
    assert_eq!(s(&x.sin().rewrite_with(&rules, &opts)), "8*sin(x)");
}

#[test]
fn rewrite_iteration_cap_terminates_non_confluent_rules() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // sin → cos → sin … never reaches a fixpoint.
    let rules = RuleSet::from_rules(vec![
        Rule::new("s2c", &a.sin(), &a.cos()),
        Rule::new("c2s", &a.cos(), &a.sin()),
    ]);
    let out = x.sin().rewrite(&rules); // 50 passes → sin (even count)
    assert_eq!(s(&out), "sin(x)");
    let odd = x
        .sin()
        .rewrite_with(&rules, &RewriteOpts::default().max_iterations(7));
    assert_eq!(s(&odd), "cos(x)");
}

#[test]
fn rewrite_node_count_guard_stops_exploding_rules() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // sin(a_) → sin(a_)·sin(a_ + 1)·sin(a_ + 2): exponential growth.
    let rhs = &a.sin() * &(&a + 1).sin() * &(&a + 2).sin();
    let rules = RuleSet::from_rules(vec![Rule::new("boom", &a.sin(), &rhs)]);
    let start = std::time::Instant::now();
    let out = x.sin().rewrite(&rules);
    assert!(start.elapsed().as_secs() < 5, "took {:?}", start.elapsed());
    assert!(out.count_ops() <= 100_000);
}

#[test]
fn strategy_top_down_collapses_tower_in_one_pass() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // exp(exp(a_)) → exp(a_): the root is rewritten repeatedly.
    let rules = RuleSet::from_rules(vec![Rule::new("flat", &a.exp().exp(), &a.exp())]);
    let tower = x.exp().exp().exp().exp().exp();
    let opts = RewriteOpts::single_pass().strategy(RewriteStrategy::TopDown);
    let (out, steps) = tower.rewrite_with_traced(&rules, &opts);
    assert_eq!(s(&out), "exp(x)");
    assert_eq!(steps.len(), 4);
    assert_eq!(s(&steps[0].before), s(&tower));
}

#[test]
fn strategy_top_down_vs_bottom_up_differ_when_rules_overlap() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // r1 needs the *unrewritten* child: exp(sin(a_)) → cos(a_); r2: sin(a_) → a_.
    let rules = RuleSet::from_rules(vec![
        Rule::new("r1", &a.sin().exp(), &a.cos()),
        Rule::new("r2", &a.sin(), &a),
    ]);
    let expr = x.sin().exp();
    let td = expr.rewrite_with(
        &rules,
        &RewriteOpts::default().strategy(RewriteStrategy::TopDown),
    );
    assert_eq!(s(&td), "cos(x)");
    let bu = expr.rewrite(&rules);
    assert_eq!(s(&bu), "exp(x)");
}

#[test]
fn strategy_innermost_normalises_replacements() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // tan(a_) → sin(a_)/cos(a_); sin(a_) → 2·sin(a_/2)·cos(a_/2) would loop, so
    // use a terminating chain: cosh(a_) → exp(a_)... keep it simple:
    // r1: sinh(a_) → cosh(a_) - exp(-a_)   (introduces cosh)
    // r2: cosh(a_) → (exp(a_) + exp(-a_))/2
    let r1 = Rule::new("sinh", &a.sinh(), &(&a.cosh() - &(-&a).exp()));
    let r2 = Rule::new("cosh", &a.cosh(), &(&(&a.exp() + &(-&a).exp()) / 2));
    let rules = RuleSet::from_rules(vec![r1, r2]);
    let expr = x.sinh();
    let opts = RewriteOpts::single_pass().strategy(RewriteStrategy::Innermost);
    let out = expr.rewrite_with(&rules, &opts);
    assert!(
        !s(&out).contains("cosh"),
        "innermost should normalise the replacement: {out}"
    );
    assert_same_value(&expr, &out, "innermost");
    // All strategies agree at the fixpoint.
    let bu = expr.rewrite(&rules);
    let td = expr.rewrite_with(
        &rules,
        &RewriteOpts::default().strategy(RewriteStrategy::TopDown),
    );
    assert_eq!(bu, out);
    assert_eq!(td, out);
}

// ═══════════════════════════════════════════════════════════════════════════
// simplify_with_rules / simplify_traced
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn simplify_with_rules_interleaves_user_rules() {
    let ctx = Context::new();
    let (x, a) = (ctx.symbol("x"), ctx.symbol("a_"));
    // User identity: cosh(a_)^2 → 1 + sinh(a_)^2.
    let extra = RuleSet::from_rules(vec![Rule::new(
        "cosh_sq",
        &a.cosh().powi(2),
        &(1 + &a.sinh().powi(2)),
    )]);
    let expr = &x.cosh().powi(2) - &x.sinh().powi(2) + &x.sin().powi(2) + &x.cos().powi(2);
    let out = expr.simplify_with_rules(&extra);
    assert_eq!(s(&out), "2");
    assert_same_value(&expr, &out, "simplify_with_rules");
}

#[test]
fn simplify_with_rules_without_rules_equals_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = (&x + 1).powi(2) - &x.powi(2) - &x * 2;
    assert_eq!(expr.simplify_with_rules(&RuleSet::new()), expr.simplify());
}

#[test]
fn simplify_traced_reports_strategy_and_rules() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let expr = &x.sin().powi(2) + &x.cos().powi(2) + &x.ln().exp();
    let (out, steps) = expr.simplify_traced(&SimplifyOpts::default());
    assert_eq!(s(&out), "x + 1");
    assert!(!steps.is_empty());
    assert!(
        steps[0].rule_name.starts_with("strategy:"),
        "{}",
        steps[0].rule_name
    );
    assert_eq!(s(&steps[0].before), s(&expr));
    let names: Vec<&str> = steps.iter().map(|st| st.rule_name.as_str()).collect();
    assert!(names.contains(&"pythagorean"), "{names:?}");
    assert!(names.contains(&"exp_ln"), "{names:?}");
}

#[test]
fn simplify_traced_no_change_is_empty_trace() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (out, steps) = (&x + 1).simplify_traced(&SimplifyOpts::default());
    assert_eq!(s(&out), "x + 1");
    assert!(steps.is_empty());
}

#[test]
fn simplify_traced_result_matches_simplify() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let exprs = [
        (&x.powi(2) - 1) / (&x - 1),
        &x.cosh().powi(2) - &x.sinh().powi(2),
        (&x.sin() / &x.cos()) * 2,
        x.exp().ln() + &x.tanh().atanh(),
    ];
    for e in &exprs {
        let (traced, _) = e.simplify_traced(&SimplifyOpts::default());
        assert_eq!(traced, e.simplify(), "{e}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Value-preservation sweep over the built-in rules via the public engine
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn standard_rules_preserve_value_on_random_inputs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let std = RuleSet::standard(&ctx);
    let inputs = [
        &x.sin().powi(2) + &x.cos().powi(2) + &x,
        (&x + 2).ln().exp() * &x,
        &x.cosh().powi(2) - &x.sinh().powi(2) + &x.powi(3),
        (&x.sin() / &x.cos()) + (&x.sinh() / &x.cosh()),
        &x.exp() * &(&x + 1).exp(),
        x.powi(2).powi(3),
    ];
    for e in &inputs {
        let out = e.rewrite(&std);
        assert_same_value(e, &out, &format!("standard rules on {e}"));
    }
}
