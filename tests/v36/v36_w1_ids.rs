//! Tests of the 0.49 refactor (w1): identifiers without an order, the checked context boundary, canonical ingestion of JSON trees.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

use std::panic::{AssertUnwindSafe, catch_unwind};

/// A context in which `pre` were created first: `ExprId`s (and so any
/// order by id) differ from a fresh context's.
fn context_with(pre: &[&str]) -> Context {
    let ctx = Context::new();
    for s in pre {
        let _ = ctx.symbol(s);
    }
    ctx
}

/// The panic message of `f`, or `None` if it returns.
fn panic_message<T>(f: impl FnOnce() -> T) -> Option<String> {
    let r = catch_unwind(AssertUnwindSafe(f));
    r.err().map(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(ToString::to_string))
            .unwrap_or_default()
    })
}

const CROSS_CONTEXT: &str = "cannot combine expressions from different contexts";

// ── Identifiers without an order (S5-B3) ────────────────────────────────

/// `factor_all(x³ − y³)` in a context where `y` was created before `x`
/// printed `-(-x + y)*(x^2 + x*y + y^2)`: `factor` sorted the variables by
/// `ExprId` (creation order).  Now they are in canonical order (by name),
/// and the answer is the fresh context's.  Oracle: the exact identity
/// `(x − y)(x² + xy + y²) = x³ − y³` (checked by `expand`), and SymPy's
/// `factor(x**3 - y**3)` = `(x - y)*(x**2 + x*y + y**2)`.
#[test]
fn factor_does_not_depend_on_symbol_creation_order() {
    let fresh = Context::new();
    let late = context_with(&["y", "b"]);
    for f in [
        "x^3 - y^3",
        "y^2 - x^2",
        "(y - x)^3",
        "a*b^2 - a^2*b",
        "x*y - x - y + 1",
    ] {
        let want = fresh.parse(f).unwrap().factor_all().to_string();
        let got = late.parse(f).unwrap().factor_all();
        assert_eq!(got.to_string(), want, "factor_all({f})");
        let e = late.parse(f).unwrap();
        assert_eq!((&got.expand() - &e.expand()).expand(), late.int(0), "{f}");
    }
    let ctx = context_with(&["y"]);
    let e = ctx.parse("x^3 - y^3").unwrap().factor_all();
    assert_eq!(e.to_string(), "(x - y)*(x^2 + x*y + y^2)");
}

/// `∫ 1/((x − a)(x − b)) dx` with `b` created before `a` swapped `a` and
/// `b` in the answer's arrangement (S5-B3, `zz_s5_order`: 18 of 96
/// task/order pairs differed).  Oracle: the fresh context's answer, whose
/// derivative is the integrand.
#[test]
fn integrate_does_not_depend_on_symbol_creation_order() {
    for f in ["1/((x - a)*(x - b))", "1/(x^2 - y^2)", "exp(a*x)*sin(b*x)"] {
        let fresh = Context::new();
        let x = fresh.symbol("x");
        let want = fresh.parse(f).unwrap().integrate(&x).to_string();
        for pre in [&["y", "b", "a"][..], &["b", "y"][..]] {
            let ctx = context_with(pre);
            let x = ctx.symbol("x");
            let got = ctx.parse(f).unwrap().integrate(&x);
            assert_eq!(got.to_string(), want, "∫{f} after {pre:?}");
        }
    }
}

// ── Canonical rebuilding (S5-B1) ────────────────────────────────────────

/// `{x, y}` with `x ↦ 1`, then `y ↦ 1` was `{1, 1}`, not `==` to `{1}`:
/// `rebuild_with` interned the set without its canonical constructor.
/// Oracle: the canonical constructor, `ctx.finite_set(&[1, 1])` = `{1}`.
#[test]
fn substitution_rebuilds_a_finite_set_canonically() {
    let ctx = Context::new();
    let (x, y, one) = (ctx.symbol("x"), ctx.symbol("y"), ctx.int(1));
    let s = ctx
        .finite_set(&[x.clone(), y.clone()])
        .subs(&x, &one)
        .subs(&y, &one);
    assert_eq!(s, ctx.finite_set(&[one.clone(), one.clone()]));
    assert_eq!(s.to_string(), "{1}");
}

/// `[x, y]` with `x ↦ 3`, `y ↦ 2` was `[3, 2]`, a non-canonical interval
/// (not `==` to `ctx.interval(3, 2)` = `EmptySet`).  Oracle: the canonical
/// constructor (an interval with `a > b` is empty, SymPy `Interval(3, 2)`
/// = `EmptySet`).
#[test]
fn substitution_rebuilds_an_interval_canonically() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let (two, three) = (ctx.int(2), ctx.int(3));
    let iv = ctx
        .interval(&x, &y, IntervalKind::Closed)
        .subs(&x, &three)
        .subs(&y, &two);
    assert_eq!(iv, ctx.interval(&three, &two, IntervalKind::Closed));
    assert_eq!(iv.to_string(), "EmptySet");
    // A union rebuilt from its parts: `[1, 2] ∪ {1}` is `[1, 2]`.
    let one = ctx.int(1);
    let u = ctx
        .interval(&x, &y, IntervalKind::Closed)
        .union(&ctx.finite_set(std::slice::from_ref(&x)))
        .subs(&x, &one)
        .subs(&y, &two);
    let direct = ctx
        .interval(&one, &two, IntervalKind::Closed)
        .union(&ctx.finite_set(std::slice::from_ref(&one)));
    assert_eq!(u, direct);
}

// ── Canonical ingestion (S5-B6) ─────────────────────────────────────────

const SET_ONE_ONE: &str = r#"{"type":"FiniteSet","elements":[{"type":"Num","numer":"1","denom":"1"},{"type":"Num","numer":"1","denom":"1"}]}"#;

/// A JSON tree `{1, 1}` read back as the non-canonical `{1, 1}`, typed as a
/// numeric `Ex`.  Now `from_json` builds `{1}` (still an `Ex`, for
/// compatibility) and `from_json_any` returns it as a set.  Oracle: the
/// canonical constructor `ctx.finite_set(&[1])`.
#[test]
fn json_sets_are_read_canonically_and_with_their_sort() {
    let ctx = Context::new();
    let one = ctx.int(1);
    let e = ctx.from_json(SET_ONE_ONE).unwrap();
    assert_eq!(e.to_string(), "{1}");
    assert_eq!(e, ctx.finite_set(std::slice::from_ref(&one)).as_ex());
    let AnyExpr::Set(s) = ctx.from_json_any(SET_ONE_ONE).unwrap() else {
        panic!("a set");
    };
    assert_eq!(s, ctx.finite_set(std::slice::from_ref(&one)));
    // An interval `[5, 3]` is empty.
    let iv = ctx.interval(&one, &ctx.int(3), IntervalKind::Closed);
    let json = iv.to_json().unwrap().replace("\"1\"", "\"5\"");
    let AnyExpr::Set(s) = ctx.from_json_any(&json).unwrap() else {
        panic!("a set");
    };
    assert_eq!(s.to_string(), "EmptySet");
}

/// `from_json_any` reads the sort of the root: relations and logic are
/// `Bool`, sets are `Set`, everything else is `Num`; each round-trips.
#[test]
fn from_json_any_reads_the_sort_of_the_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let rel = x.gt(&ctx.int(1));
    match ctx.from_json_any(&rel.to_json().unwrap()).unwrap() {
        AnyExpr::Bool(b) => assert_eq!(b, rel),
        other => panic!("{other:?}"),
    }
    let both = rel.and(&x.lt(&ctx.int(3)));
    assert!(
        matches!(ctx.from_json_any(&both.to_json().unwrap()).unwrap(), AnyExpr::Bool(b) if b == both)
    );
    let num = &x.sin() + 1;
    assert!(
        matches!(ctx.from_json_any(&num.to_json().unwrap()).unwrap(), AnyExpr::Num(e) if e == num)
    );
    let set = ctx.interval(&ctx.int(0), &x, IntervalKind::Open);
    assert!(
        matches!(ctx.from_json_any(&set.to_json().unwrap()).unwrap(), AnyExpr::Set(s) if s == set)
    );
    assert!(matches!(
        ctx.from_json_any("{not json"),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

/// A JSON number with denominator 0 panicked inside `from_json`
/// (`Ratio::new(1, 0)`: "denominator == 0"), and a malformed numerator was
/// silently read as 0.  Both are errors now.  Oracle: the documented
/// contract (`Err` for a JSON text that is not a valid tree).
#[test]
fn malformed_json_numbers_are_errors_not_panics() {
    let ctx = Context::new();
    for json in [
        r#"{"type":"Num","numer":"1","denom":"0"}"#,
        r#"{"type":"Num","numer":"abc","denom":"1"}"#,
        r#"{"type":"Add","terms":[{"type":"Symbol","name":"x"},{"type":"Num","numer":"2","denom":"0"}]}"#,
    ] {
        assert!(ctx.from_json(json).is_err(), "{json}");
        assert!(ctx.from_json_any(json).is_err(), "{json}");
    }
    let ok = r#"{"type":"Num","numer":"-3","denom":"6"}"#;
    assert_eq!(ctx.from_json(ok).unwrap(), ctx.rational(-1, 2));
}

// ── The checked context boundary (S5-B2) ────────────────────────────────

/// `a.query(&b_minus_7, POSITIVE)` answered `Some(true)`: the foreign id
/// indexed `a`'s arena and hit `a`'s node `7`.  Now it panics with the
/// cross-context message, the policy of every other operation
/// (`tests/unit/test_cross_context.rs`).
#[test]
fn query_rejects_a_foreign_expression() {
    let a = Context::new();
    let b = Context::new();
    let _ = a.int(7);
    let neg7 = b.int(-7);
    let msg = panic_message(|| a.query(&neg7, Props::POSITIVE)).expect("panics");
    assert!(msg.contains(CROSS_CONTEXT), "{msg}");
}

/// `a.finite_set(&[b_t])` and `a.interval(…)` with `b`'s expressions
/// panicked with "index out of bounds" (`arena.rs`), and the other
/// `Context` methods taking expressions used foreign ids unchecked.  Each
/// now gives the cross-context message.
#[test]
fn context_methods_reject_foreign_expressions() {
    let a = Context::new();
    let b = Context::new();
    let t = b.symbol("t");
    let neg7 = b.int(-7);
    let cases: Vec<(&str, Option<String>)> = vec![
        (
            "finite_set",
            panic_message(|| a.finite_set(std::slice::from_ref(&t))),
        ),
        (
            "interval",
            panic_message(|| a.interval(&neg7, &t, IntervalKind::Open)),
        ),
        (
            "physical_constant",
            panic_message(|| a.physical_constant("c", neg7.clone())),
        ),
        (
            "compact",
            panic_message(|| a.compact(std::slice::from_ref(&t))),
        ),
        (
            "liveness_ratio",
            panic_message(|| a.liveness_ratio(std::slice::from_ref(&t))),
        ),
        (
            "should_compact",
            panic_message(|| a.should_compact(std::slice::from_ref(&t))),
        ),
    ];
    for (name, msg) in cases {
        let msg = msg.unwrap_or_else(|| panic!("{name}: no panic"));
        assert!(msg.contains(CROSS_CONTEXT), "{name}: {msg}");
    }
    // The same calls with the context's own expressions work.
    let x = a.symbol("x");
    assert_eq!(a.query(&a.int(-7), Props::POSITIVE), Some(false));
    assert_eq!(a.finite_set(std::slice::from_ref(&x)).to_string(), "{x}");
    assert!(a.liveness_ratio(std::slice::from_ref(&x)) > 0.0);
}
