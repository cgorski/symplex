//! v0.2 ergonomics — equality and ordering queries, evaluation shortcuts.
//!
//! `equals`, `compare_numeric`, `is_less_than` / `is_greater_than`,
//! `probably_equal`, `eval_at`, `eval_f64_with`, `subs_map_with`.

use std::cmp::Ordering;
use symplex::prelude::*;

// ── equals ─────────────────────────────────────────────────────────────

#[test]
fn equals_structural_and_expanded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.equals(&x), Some(true));
    assert_eq!(
        (&x + 1).powi(2).equals(&(&x.powi(2) + &x * 2 + 1)),
        Some(true)
    );
}

#[test]
fn equals_trig_identity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(
        (&x.sin().powi(2) + &x.cos().powi(2)).equals(&ctx.one()),
        Some(true)
    );
}

#[test]
fn equals_nonzero_constant_difference_is_false() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.equals(&(&x + 1)), Some(false));
    assert_eq!(
        (&x * 2).equals(&(&x * 2 - &ctx.rational(1, 3))),
        Some(false)
    );
    assert_eq!(ctx.int(1).equals(&ctx.int(2)), Some(false));
    assert_eq!(ctx.rational(1, 2).equals(&ctx.rational(1, 3)), Some(false));
}

#[test]
fn equals_numerically_distinct_constants_is_false() {
    let ctx = Context::new();
    assert_eq!(ctx.pi().equals(&ctx.rational(22, 7)), Some(false));
    assert_eq!(ctx.int(2).sqrt().equals(&ctx.rational(7, 5)), Some(false));
    assert_eq!(ctx.e().equals(&ctx.pi()), Some(false));
}

#[test]
fn equals_constants_that_are_equal() {
    let ctx = Context::new();
    assert_eq!(
        (&ctx.int(2).sqrt() * &ctx.int(2).sqrt()).equals(&ctx.int(2)),
        Some(true)
    );
    assert_eq!(
        (&ctx.i_unit() * &ctx.pi()).exp().equals(&ctx.int(-1)),
        Some(true)
    );
}

#[test]
fn equals_unknown_for_distinct_symbols() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(x.equals(&y), None);
    assert_eq!(x.equals(&ctx.int(3)), None);
    assert_eq!(x.sin().equals(&x.cos()), None);
}

#[test]
fn equals_uses_assumptions_for_nonzero() {
    let ctx = Context::new();
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    assert_eq!((&r.powi(2) + 1).equals(&ctx.zero()), Some(false));
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    assert_eq!(p.equals(&ctx.zero()), Some(false));
}

#[test]
fn equals_is_symmetric() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = (&x + 1).powi(2);
    let b = &x.powi(2) + &x * 2 + 1;
    assert_eq!(a.equals(&b), b.equals(&a));
    assert_eq!(x.equals(&(&x + 1)), (&x + 1).equals(&x));
}

// ── compare_numeric ────────────────────────────────────────────────────

#[test]
fn compare_numeric_literals_exact() {
    let ctx = Context::new();
    assert_eq!(
        ctx.int(1).compare_numeric(&ctx.int(2)),
        Some(Ordering::Less)
    );
    assert_eq!(
        ctx.int(2).compare_numeric(&ctx.int(1)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        ctx.rational(1, 3).compare_numeric(&ctx.rational(2, 6)),
        Some(Ordering::Equal)
    );
    // Values that are indistinguishable in f64 are still ordered exactly.
    let a = ctx
        .rational_str("1000000000000000000001/1000000000000000000000")
        .unwrap();
    let b = ctx.int(1);
    assert_eq!(a.compare_numeric(&b), Some(Ordering::Greater));
}

#[test]
fn compare_numeric_constants_numeric_fallback() {
    let ctx = Context::new();
    assert_eq!(
        ctx.pi().compare_numeric(&ctx.int(3)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        ctx.pi().compare_numeric(&ctx.rational(22, 7)),
        Some(Ordering::Less)
    );
    assert_eq!(ctx.e().compare_numeric(&ctx.pi()), Some(Ordering::Less));
    assert_eq!(
        ctx.int(2).sqrt().compare_numeric(&ctx.int(3).sqrt()),
        Some(Ordering::Less)
    );
}

#[test]
fn compare_numeric_equal_constants_via_eval() {
    let ctx = Context::new();
    assert_eq!(
        ctx.int(4).sqrt().compare_numeric(&ctx.int(2)),
        Some(Ordering::Equal)
    );
    assert_eq!(
        (&ctx.pi() - &ctx.pi()).compare_numeric(&ctx.zero()),
        Some(Ordering::Equal)
    );
}

#[test]
fn compare_numeric_infinities() {
    let ctx = Context::new();
    assert_eq!(
        ctx.infinity().compare_numeric(&ctx.int(1)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        ctx.int(1).compare_numeric(&ctx.infinity()),
        Some(Ordering::Less)
    );
    assert_eq!(
        ctx.neg_infinity().compare_numeric(&ctx.int(1)),
        Some(Ordering::Less)
    );
    assert_eq!(
        ctx.neg_infinity().compare_numeric(&ctx.infinity()),
        Some(Ordering::Less)
    );
}

#[test]
fn compare_numeric_complex_is_unordered() {
    let ctx = Context::new();
    assert_eq!(ctx.i_unit().compare_numeric(&ctx.zero()), None);
    assert_eq!(ctx.int(-1).sqrt().compare_numeric(&ctx.int(1)), None);
}

#[test]
fn compare_numeric_symbolic_via_difference() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x + 1).compare_numeric(&x), Some(Ordering::Greater));
    assert_eq!((&x - 1).compare_numeric(&x), Some(Ordering::Less));
    assert_eq!((&x * 2).compare_numeric(&(&x + &x)), Some(Ordering::Equal));
    assert_eq!(x.compare_numeric(&ctx.zero()), None);
    assert_eq!(x.compare_numeric(&ctx.symbol("y")), None);
}

#[test]
fn compare_numeric_with_assumptions() {
    let ctx = Context::new();
    let p = ctx.symbol_with("p", &[Assumption::Positive]);
    let n = ctx.symbol_with("n", &[Assumption::Negative]);
    assert_eq!(p.compare_numeric(&ctx.zero()), Some(Ordering::Greater));
    assert_eq!(n.compare_numeric(&ctx.zero()), Some(Ordering::Less));
    assert_eq!(p.compare_numeric(&n), Some(Ordering::Greater));
    assert_eq!((&p + &p).compare_numeric(&p), Some(Ordering::Greater));
}

#[test]
fn is_less_than_and_is_greater_than() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(ctx.int(1).is_less_than(&ctx.int(2)), Some(true));
    assert_eq!(ctx.int(2).is_less_than(&ctx.int(1)), Some(false));
    assert_eq!(ctx.int(2).is_less_than(&ctx.int(2)), Some(false));
    assert_eq!(ctx.int(2).is_greater_than(&ctx.int(1)), Some(true));
    assert_eq!(ctx.int(2).is_greater_than(&ctx.int(2)), Some(false));
    assert_eq!(x.is_less_than(&ctx.int(1)), None);
    assert_eq!(x.is_greater_than(&ctx.int(1)), None);
    assert_eq!((&x + 1).is_greater_than(&x), Some(true));
    assert_eq!((&x + 1).is_less_than(&x), Some(false));
}

#[test]
fn is_less_than_uses_nonnegativity() {
    let ctx = Context::new();
    let r = ctx.symbol_with("r", &[Assumption::Real]);
    // r² ≥ 0, so r² < 0 is false even though r² > 0 is unknown.
    assert_eq!(r.powi(2).is_less_than(&ctx.zero()), Some(false));
    assert_eq!(r.powi(2).is_greater_than(&ctx.zero()), None);
    assert_eq!(r.powi(2).is_less_than(&ctx.int(-1)), Some(false));
}

// ── probably_equal ─────────────────────────────────────────────────────

#[test]
fn probably_equal_true_for_polynomial_identity() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let lhs = (&x - &y) * (&x + &y);
    let rhs = &x.powi(2) - &y.powi(2);
    assert_eq!(lhs.probably_equal(&rhs, 3), Some(true));
}

#[test]
fn probably_equal_false_for_near_miss_polynomial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x + 1).powi(2);
    let rhs = &x.powi(2) + &x * 2 + 2;
    assert_eq!(lhs.probably_equal(&rhs, 3), Some(false));
}

#[test]
fn probably_equal_distinguishes_different_symbols() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert_eq!(x.probably_equal(&y, 3), Some(false));
    assert_eq!((&x * &y).probably_equal(&(&x + &y), 3), Some(false));
}

#[test]
fn probably_equal_transcendental() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let lhs = (&x * 2).sin();
    let rhs = &(&x.sin() * &x.cos()) * 2;
    assert_eq!(lhs.probably_equal(&rhs, 4), Some(true));
    assert_eq!(lhs.probably_equal(&x.sin(), 4), Some(false));
}

#[test]
fn probably_equal_exp_log() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // exp(ln(x)) = x fails for negative samples (complex), which are
    // skipped or agree; positive samples agree. Either way no false
    // negative.
    assert_eq!(x.ln().exp().probably_equal(&x, 6), Some(true));
}

#[test]
fn probably_equal_is_deterministic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = x.powi(3).probably_equal(&(&x * &x * &x + 1), 5);
    let b = x.powi(3).probably_equal(&(&x * &x * &x + 1), 5);
    assert_eq!(a, b);
    assert_eq!(a, Some(false));
}

#[test]
fn probably_equal_zero_samples_treated_as_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!(x.probably_equal(&(&x + 1), 0), Some(false));
    assert_eq!((&x * 1).probably_equal(&x, 0), Some(true));
}

#[test]
fn probably_equal_none_when_nothing_evaluable() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // An undefined function can never be evaluated at a sample point.
    let f = ctx.apply("f", &[&x]);
    let g = ctx.apply("g", &[&x]);
    assert_eq!(f.probably_equal(&g, 3), None);
}

// ── eval_at / eval_f64_with / subs_map_with ────────────────────────────

#[test]
fn eval_at_is_simultaneous_and_exact() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.powi(2) + &y;
    assert_eq!(
        f.eval_at(&[(&x, &ctx.rational(1, 2)), (&y, &ctx.int(1))]),
        ctx.rational(5, 4)
    );
    // Swap does not chain.
    assert_eq!(f.eval_at(&[(&x, &y), (&y, &x)]), &y.powi(2) + &x);
}

#[test]
fn eval_at_partial_binding_stays_symbolic() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x.sin() + &y;
    assert_eq!(f.eval_at(&[(&x, &ctx.zero())]), y);
}

#[test]
fn eval_f64_with_integers() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x * &y + 1;
    assert_eq!(f.eval_f64_with(&[(&x, 2), (&y, 3)]).unwrap(), 7.0);
    let typed: &[(&Ex, i64)] = &[(&x, 2), (&y, 3)];
    assert_eq!(f.eval_f64_with(typed).unwrap(), 7.0);
}

#[test]
fn eval_f64_with_floats_is_exact_substitution() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // (x * 10 - 1) at x = 0.1: exact rational arithmetic on the dyadic 0.1
    // gives a tiny positive number, not 0 — matching f64 semantics.
    let v = (&x * 10 - 1).eval_f64_with(&[(&x, 0.1)]).unwrap();
    assert!(v.abs() < 1e-15, "{v}");
    assert_eq!(x.powi(2).eval_f64_with(&[(&x, 1.5)]).unwrap(), 2.25);
}

#[test]
fn eval_f64_with_expression_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = x.sin().eval_f64_with(&[(&x, &(&ctx.pi() / 2))]).unwrap();
    assert!((v - 1.0).abs() < 1e-12);
}

#[test]
fn eval_f64_with_errors_on_unbound_symbol() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    assert!(matches!(
        (&x + &y).eval_f64_with(&[(&x, 1)]),
        Err(SymplexError::FreeSymbol { .. })
    ));
}

#[test]
fn eval_f64_with_rational_pairs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert_eq!((&x * 3).eval_f64_with_rational(&[(&x, 1, 3)]).unwrap(), 1.0);
}

#[test]
fn subs_map_with_and_subs_map_i64() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let f = &x + &y * 10;
    assert_eq!(f.subs_map_i64(&[(&x, 1), (&y, 2)]), ctx.int(21));
    assert_eq!(f.subs_map_with(&[(&x, 0.5), (&y, 0.25)]), ctx.int(3));
    assert_eq!(f.subs_map_with(&[(&x, &y), (&y, &x)]), &y + &x * 10);
}
