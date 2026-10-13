//! Tests of the 0.49 refactor (w7): the node-shape derive, IntervalKind and the canonical builder.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn iv(ctx: &Context, a: i64, b: i64, kind: IntervalKind) -> SetEx {
    ctx.interval(&ctx.int(a), &ctx.int(b), kind)
}

/// The set `s` read back from its own JSON.
fn json_round_trip(ctx: &Context, s: &SetEx) -> Ex {
    let json = s.to_json().expect("a set serialises");
    ctx.from_json(&json).expect("its own JSON reads back")
}

/// A JSON interval whose `flags` has bits beyond the two documented ones
/// (`{"flags": 4}`) read back as an interval node carrying the byte 4: it
/// printed `[0, 1]` but was not `==` to `ctx.interval(0, 1, Closed)`, and
/// was written back with `"flags":4`.  The node now holds an
/// `IntervalKind`, and the reader keeps the two documented bits.  Oracle:
/// the documented encoding of `ExprTree::Interval::flags` (bit 0 = lower
/// end open, bit 1 = upper end open).
#[test]
fn json_interval_flags_beyond_the_two_low_bits_are_ignored() {
    let ctx = Context::new();
    let closed = iv(&ctx, 0, 1, IntervalKind::Closed);
    let json = closed.to_json().expect("serialises");
    assert!(json.contains("\"flags\":0"), "{json}");
    for (flags, kind) in [
        (4u8, IntervalKind::Closed),
        (5, IntervalKind::LeftOpen),
        (6, IntervalKind::RightOpen),
        (7, IntervalKind::Open),
    ] {
        let edited = json.replace("\"flags\":0", &format!("\"flags\":{flags}"));
        let e = ctx.from_json(&edited).expect("reads");
        let want = iv(&ctx, 0, 1, kind);
        assert_eq!(e, want.as_ex(), "flags {flags}");
        let back = e.to_json().expect("serialises");
        assert!(back.contains(&format!("\"flags\":{}", flags & 3)), "{back}");
    }
}

/// Unchanged; pins the stable JSON encoding that the `u8` → `IntervalKind`
/// change of the node must keep: `[a, b]`, `(a, b]`, `[a, b)`, `(a, b)` are
/// `flags` 0, 1, 2, 3 and read back to the same set (and print with the
/// same brackets, in `Display`, LaTeX and MathML).  Oracle: the 0.48 JSON
/// (`ExprTree::Interval { flags }`, bit 0 = lower open, bit 1 = upper
/// open).
#[test]
fn interval_kinds_keep_their_json_flags_and_brackets() {
    let ctx = Context::new();
    for (kind, flags, text) in [
        (IntervalKind::Closed, 0, "[0, 1]"),
        (IntervalKind::LeftOpen, 1, "(0, 1]"),
        (IntervalKind::RightOpen, 2, "[0, 1)"),
        (IntervalKind::Open, 3, "(0, 1)"),
    ] {
        let s = iv(&ctx, 0, 1, kind);
        let json = s.to_json().expect("serialises");
        assert!(json.contains(&format!("\"flags\":{flags}")), "{json}");
        assert_eq!(json_round_trip(&ctx, &s), s.as_ex());
        assert_eq!(s.to_string(), text);
        let latex = s.as_ex().to_latex();
        assert!(
            latex.starts_with(&text[..1]) && latex.ends_with(&text[text.len() - 1..]),
            "{latex}"
        );
        let mathml = s.as_ex().to_mathml().expect("renders");
        assert!(
            mathml.contains(&format!("<mo>{}</mo>", &text[..1])),
            "{mathml}"
        );
    }
}

/// Unchanged; pins that substitution rebuilds every function through its
/// folding constructor now that `rebuild_with` is generated
/// (`map_children` + `Arena::make`): `|x|`, `sign(x)`, `re(x)`, `Si(x)`,
/// `ζ(x)` and `ψ⁽ⁿ⁾(x)` at exact points fold exactly as the constructors
/// do.  Oracle: the values `|−3| = 3`, `sign(−2) = −1`, `re(2 + 3i) = 2`,
/// `Si(0) = 0`, `ζ(2) = π²/6`, `ψ⁽¹⁾(1) = π²/6` (DLMF 25.6.1, 5.15.2).
#[test]
fn substitution_folds_through_the_constructors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        ("abs(x)", ctx.int(-3), "3"),
        ("sign(x)", ctx.int(-2), "-1"),
        ("re(x)", ctx.parse("2 + 3*I").expect("parses"), "2"),
        ("Si(x)", ctx.int(0), "0"),
        ("zeta(x)", ctx.int(2), "1/6*pi^2"),
        ("polygamma(1, x)", ctx.int(1), "1/6*pi^2"),
    ];
    for (f, at, want) in cases {
        let e = ctx.parse(f).expect("parses");
        let got = e.subs(&x, &at);
        let want_e = ctx.parse(want).expect("parses");
        assert_eq!(got, want_e, "{f} at {at}: {got}");
    }
}

/// Unchanged; pins the binder table that is now generated from the node
/// declaration (`#[binder(..)]`, `#[var_slot(..)]`): the variable of a
/// `Sum` is bound in the body and free in the limits, a `Derivative` is a
/// function of its variable.  Oracle: SymPy's `free_symbols`
/// (`Sum(f(k), (k, 1, n)).free_symbols == {n}`,
/// `Sum(f(k), (k, 1, k)).free_symbols == {k}`,
/// `Derivative(f(x), x).free_symbols == {x}`).
#[test]
fn generated_binder_table_keeps_free_symbols() {
    let ctx = Context::new();
    let names = |e: &Ex| -> Vec<String> {
        let mut v: Vec<String> = e.free_symbols().iter().map(|s| s.to_string()).collect();
        v.sort();
        v
    };
    let (k, n) = (ctx.symbol("k"), ctx.symbol("n"));
    let fk = ctx.apply("f", &[&k]).expect("applies");
    let s = fk.summation(&k, &ctx.int(1), &n);
    assert!(s.to_string().contains("Sum"), "{s}");
    assert_eq!(names(&s), ["n"]);
    let k_upper = fk.summation(&k, &ctx.int(1), &k);
    assert_eq!(names(&k_upper), ["k"]);
    let x = ctx.symbol("x");
    let d = ctx.apply("f", &[&x]).expect("applies").diff(&x);
    assert_eq!(names(&d), ["x"]);
}
