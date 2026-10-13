//! Tests of the 0.49 refactor (w2): three-valued decisions, typed memo keys and the strength of numeric evidence.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::polysys::{LinearSolution, linsolve};
use symplex::prelude::*;

/// The memo of constant facts was keyed by hand-picked `u8` codes:
/// `canon`'s "sign of a factor of a directed infinity" (codes 2 and 3) was
/// read by `eval` as the constant class "positive" (code 2), and its
/// sign of `re(g)` (codes 4 and 5) as the class "zero" (code 4).  After
/// `(ln 2 − 1)·∞` (which is `−∞`), `(ln 2 − 1).is_positive()` was
/// `Some(true)`; after `re(Γ(1 + i))·∞`, `Γ(1 + i).is_zero()` and
/// `.is_real()` were `Some(true)`.  Oracle (mpmath 1.3, 50 digits):
/// `log(2) - 1` = −0.306852819440054690…, `1 - sqrt(3)` = −0.732…,
/// `pi - 4` = −0.858…, `gamma(1+1j)` = 0.498015668118356042… −
/// 0.154949828301810685…i.  The answers must not depend on what was asked
/// before: fresh, after the directed infinity, and asked again.
#[test]
fn a_memoised_sign_of_a_directed_infinity_is_not_read_as_a_constant_class() {
    for src in ["log(2) - 1", "1 - sqrt(3)", "pi - 4"] {
        let ctx = Context::new();
        let c = ctx.parse(src).unwrap();
        let fresh = (c.is_positive(), c.is_negative(), c.is_zero(), c.is_real());
        assert_eq!(
            fresh,
            (Some(false), Some(true), Some(false), Some(true)),
            "{src}"
        );
        let ctx = Context::new();
        let inf = ctx.parse(&format!("({src})*oo")).unwrap();
        assert_eq!(inf, ctx.neg_infinity(), "({src})*oo");
        let c = ctx.parse(src).unwrap();
        for round in 0..2 {
            let after = (c.is_positive(), c.is_negative(), c.is_zero(), c.is_real());
            assert_eq!(after, fresh, "{src} after ({src})*oo, round {round}");
        }
    }
    let ctx = Context::new();
    let _ = ctx.parse("re(gamma(1 + I))*oo").unwrap();
    let g = ctx.parse("gamma(1 + I)").unwrap();
    assert_eq!(
        g.is_zero(),
        Some(false),
        "gamma(1 + I) after re(gamma(1 + I))*oo"
    );
    assert_eq!(
        g.is_real(),
        Some(false),
        "gamma(1 + I) after re(gamma(1 + I))*oo"
    );
}

/// The class memo itself answers as a cold proof does, for each class of
/// constant (the class moved from a private 8-variant enum to a
/// `SignSet`; this pins the re-encoding).  Oracle: `sqrt(2) - 1` =
/// 0.414…, `cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2` = 0 (the half-angle
/// formula, three times; mpmath: 0.0 at 50 digits), `gamma(1+1j)` as above.
#[test]
fn constant_class_queries_are_the_same_cold_and_warm() {
    type Facts = (Option<bool>, Option<bool>, Option<bool>, Option<bool>);
    let cases: [(&str, Facts); 3] = [
        (
            "sqrt(2) - 1",
            (Some(true), Some(false), Some(false), Some(true)),
        ),
        (
            "cos(pi/16) - sqrt(2 + sqrt(2 + sqrt(2)))/2",
            (Some(false), Some(false), Some(true), Some(true)),
        ),
        (
            "gamma(1 + I)",
            (Some(false), Some(false), Some(false), Some(false)),
        ),
    ];
    for (src, want) in cases {
        let ctx = Context::new();
        let c = ctx.parse(src).unwrap();
        for round in 0..2 {
            let got = (c.is_positive(), c.is_negative(), c.is_zero(), c.is_real());
            assert_eq!(got, want, "{src}, round {round}");
        }
    }
}

/// `c = e^{−10⁻¹⁵⁰} − 1` is `−10⁻¹⁵⁰` (mpmath at 400 digits:
/// `exp(-mpf(10)**-150) - 1` = −1.0e−150), but only zero to the precision
/// of an ordinary evaluation, and the limit engine took that zero for a
/// proof: `lim c·x` was `0`, `lim c·x² − 1` was `−1`, `lim e^{c·x}` was
/// `1`, `lim x²(c + 1/x)` and `lim c·x + ln x` were `∞`.  With `c < 0` they
/// are `−∞`, `−∞`, `0`, `−∞`, `−∞` (the deep zero search certifies `c`).
#[test]
fn a_constant_zero_only_to_the_precision_reached_is_not_zero_in_limits() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    let c = "(exp(-10^-150) - 1)";
    for (body, want) in [
        ("{c}*x", "-oo"),
        ("{c}*x^2 - 1", "-oo"),
        ("exp({c}*x)", "0"),
        ("x^2*({c} + 1/x)", "-oo"),
        ("{c}*x + log(x)", "-oo"),
    ] {
        let e = ctx.parse(&body.replace("{c}", c)).unwrap();
        let l = e
            .try_limit(&x, &oo)
            .unwrap_or_else(|err| panic!("{body}: {err}"));
        assert_eq!(l.to_string(), want, "lim_(x->oo) {body}");
    }
}

/// `c = Γ(2 + i·d)` with `d = e^{10⁻¹⁵⁰} − 1 − 10⁻¹⁵⁰` (mpmath at 800
/// digits: `d` = 5.0e−301, `im(gamma(2 + 1j*d))` = 2.1139216754923357e−301,
/// equal to `ψ(2)·d`) is not real, but the limit engine's realness test
/// accepted an imaginary part that was only zero to the precision reached:
/// `lim c·x` was `∞`, `lim (c − 1)·x` was `0`, `lim e^{(c−1)·x}` was `1`,
/// `lim (c − 1)·x² + x` was `∞`.  None of them is a real limit; the engine
/// must not give those answers (it now refuses).  The same for `atan(1 +
/// i·d)` (imaginary part `d/2`).
#[test]
fn an_imaginary_part_zero_only_to_the_precision_reached_is_not_realness() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    let d = "(exp(10^-150) - 1 - 10^-150)";
    for (c, wrong) in [
        ("gamma(2 + I*{d})", ["oo", "oo", "oo"]),
        ("(gamma(2 + I*{d}) - 1)", ["0", "1", "oo"]),
        ("atan(1 + I*{d})", ["oo", "oo", "oo"]),
        ("(atan(1 + I*{d}) - pi/4)", ["0", "1", "oo"]),
    ] {
        let c = c.replace("{d}", d);
        for (body, wrong) in ["{c}*x", "exp({c}*x)", "{c}*x^2 + x"].iter().zip(wrong) {
            let e = ctx.parse(&body.replace("{c}", &c)).unwrap();
            if let Ok(l) = e.try_limit(&x, &oo) {
                assert_ne!(l.to_string(), wrong, "lim_(x->oo) {body} with c = {c}");
            }
        }
    }
}

/// `linsolve` read "not proved zero" as "nonzero" at its consistency
/// check: for `x + y = 1, x + y = 1 + k` with `k = Γ(1/3)Γ(2/3) − 2π/√3`,
/// which is 0 by the reflection formula `Γ(z)Γ(1 − z) = π/sin(πz)` (mpmath:
/// `gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)` = 0.0 at 50 digits) but which no
/// test of the crate recognises, the system was `Inconsistent` (it is
/// `Parametric`, `x = 1 − y`).  The same with `Γ(1/4)Γ(3/4) − π√2`.  Now
/// the consistency is reported undecided; a right-hand side proved
/// nonzero still makes the system inconsistent (`√2 − 1`, and `e^{10⁻¹⁵⁰} −
/// 1 − 10⁻¹⁵⁰` = 5.0e−301 by mpmath at 800 digits, certified by the deep
/// zero search).
#[test]
fn an_undecided_right_hand_side_is_not_an_inconsistency() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let system = |k: &str| {
        let e1 = ctx.parse("x + y - 1").unwrap();
        let e2 = ctx.parse(&format!("x + y - 1 - ({k})")).unwrap();
        linsolve(&[e1, e2], &[x.clone(), y.clone()])
    };
    for k in [
        "gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)",
        "gamma(1/4)*gamma(3/4) - pi*sqrt(2)",
    ] {
        match system(k) {
            Err(SymplexError::ComputationFailed { operation, reason }) => {
                assert_eq!(operation, "linsolve");
                assert!(reason.contains("cannot decide"), "k = {k}: {reason}");
            }
            other => panic!("k = {k}: want an undecided consistency, got {other:?}"),
        }
    }
    for k in ["sqrt(2) - 1", "exp(10^-150) - 1 - 10^-150"] {
        assert!(
            matches!(system(k), Ok(LinearSolution::Inconsistent)),
            "k = {k}: {:?}",
            system(k)
        );
    }
}

/// A right-hand side with symbols that is not identically zero keeps the
/// generic-solution convention: `x + y = 1, x + y = 1 + sin(a)` is
/// `Inconsistent` (for every `a` but the zeros of `sin`), as `x + y = 1,
/// x + y = 1 + a` always was.  The first version of the evidence rule above
/// reported `sin(a)` undecided, because no exact test proves `sin(a) ≠ 0`;
/// a certified nonzero value at a sample point (`sin` of a Gaussian
/// rational) proves it is not identically zero.  A symbolic right-hand side
/// that *is* identically zero (`sin²a + cos²a − 1`) still leaves the system
/// consistent.
#[test]
fn a_symbolic_right_hand_side_not_identically_zero_is_generically_inconsistent() {
    let ctx = Context::new();
    let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
    let _a = ctx.symbol("a");
    let system = |k: &str| {
        let e1 = ctx.parse("x + y - 1").unwrap();
        let e2 = ctx.parse(&format!("x + y - 1 - ({k})")).unwrap();
        linsolve(&[e1, e2], &[x.clone(), y.clone()])
    };
    for k in ["a", "sin(a)", "exp(a) + 1", "gamma(a) - 2"] {
        assert!(
            matches!(system(k), Ok(LinearSolution::Inconsistent)),
            "k = {k}: {:?}",
            system(k)
        );
    }
    assert!(
        !matches!(
            system("sin(a)^2 + cos(a)^2 - 1"),
            Ok(LinearSolution::Inconsistent)
        ),
        "{:?}",
        system("sin(a)^2 + cos(a)^2 - 1")
    );
}

/// Gruntz decides the sign of constants such as `½·ln 4 − ln 2` (the
/// leading exponent of `2^{−x}·√(1 + 4^{1+x})` at `∞`), which are 0 but
/// which no evaluation proves 0.  Up to 0.48 the limit engine took any
/// value zero to the precision reached for 0; with the evidence rule it
/// takes the logarithm identities (`log_combine`, `expand_log`, where they
/// hold) as the proof, so these limits keep their values: `2` (exactly
/// `√(4·4ˣ + 1)/2ˣ → √4`) and `1/360` (Stirling's series `ln Γ(x + 1) =
/// (x + ½)·ln x − x + ½·ln 2π + 1/(12x) − 1/(360x³) + …`; the corpus case
/// is from SymPy's test suite, where it is `1/360`).  Without the
/// identities both were refused ("cannot determine the sign of −ln 2 +
/// ½·ln 4", of "−½·ln 2 − ½·ln π + ½·ln 2π").
#[test]
fn logarithm_identities_prove_the_zero_constants_of_gruntz() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let oo = ctx.infinity();
    for (body, want) in [
        ("2^(-x)*sqrt(1 + 4^(1 + x))", "2"),
        (
            "x^3*(log(2*pi)/2 - x - log(factorial(x)) + 1/(12*x) + (1/2 + x)*log(x))",
            "1/360",
        ),
    ] {
        let e = ctx.parse(body).unwrap();
        let l = e
            .try_limit(&x, &oo)
            .unwrap_or_else(|err| panic!("{body}: {err}"));
        assert_eq!(l.to_string(), want, "lim_(x->oo) {body}");
    }
}
