//! Tests of the 0.49 refactor (w3b): the Solutions outcome type and the complex branches of abs and Lambert.
//! Every test says what was wrong before and cites its oracle.
//!
//! Oracles: exact identities (a point substituted back and evaluated
//! exactly), and mpmath 1.3 (`mp.dps = 30`) where a value is numeric.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn real(ctx: &Context, name: &str) -> Ex {
    ctx.symbol_with(name, &[Assumption::Real]).unwrap()
}

fn is_refusal<T: std::fmt::Debug>(r: &Result<T, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::ComputationFailed { .. }))
}

/// `e` vanishes at `x = r` after exact evaluation.
fn assert_exact_root(e: &Ex, x: &Ex, r: &Ex) {
    let at = e.subs(x, r).eval();
    assert!(at.is_zero_structural(), "{e} at {x} = {r} is {at}");
}

/// The solutions of `e = 0` for `x`, as displayed, sorted.
fn shown(r: Result<Vec<Ex>, SymplexError>) -> Vec<String> {
    let mut v: Vec<String> = r.unwrap().iter().map(|s| format!("{s}")).collect();
    v.sort();
    v
}

/// `|x| = 1` for a plain (complex) `x`: up to 0.48 `Ok([1, -1])`, as if
/// those were all the roots; the solutions are the unit circle.  Oracle
/// (exact): `x = i` solves it, `|i| - 1 = 0`.  For a real `x` the answer
/// `[1, -1]` is unchanged.
#[test]
fn abs_of_a_complex_unknown_is_a_curve() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "Abs(x) - 1");
    assert_exact_root(&e, &x, &ctx.i_unit());
    assert!(is_refusal(&e.solve(&x)), "{:?}", e.solve(&x));
    let xr = real(&ctx, "x");
    let er = e.subs(&x, &xr);
    assert_eq!(shown(er.solve(&xr)), vec!["-1", "1"]);
}

/// `|x + 3| = 2|x - 3|` for a plain `x`: up to 0.48 `Ok([9, 1])`; the
/// solutions are the Apollonius circle `|x - 5| = 4`.  Oracle (exact):
/// `x = 5 + 4i` gives `|8 + 4i| = sqrt(80) = 2|2 + 4i|`.  Real `x`:
/// `[1, 9]` unchanged.
#[test]
fn apollonius_circle_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "Abs(x + 3) - 2*Abs(x - 3)");
    assert_exact_root(&e, &x, &parse(&ctx, "5 + 4*I"));
    assert!(is_refusal(&e.solve(&x)), "{:?}", e.solve(&x));
    let xr = real(&ctx, "x");
    assert_eq!(shown(e.subs(&x, &xr).solve(&xr)), vec!["1", "9"]);
}

/// `(x - 1)(|x| - 2)` for a plain `x`: up to 0.48 `Ok([1, 2, -2])`.  The
/// factor `|x| - 2` vanishes on the circle `|x| = 2`.  Oracle (exact):
/// `x = 2i` solves it.
#[test]
fn product_with_an_abs_factor_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "(x - 1)*(Abs(x) - 2)");
    assert_exact_root(&e, &x, &parse(&ctx, "2*I"));
    assert!(is_refusal(&e.solve(&x)), "{:?}", e.solve(&x));
    let xr = real(&ctx, "x");
    assert_eq!(shown(e.subs(&x, &xr).solve(&xr)), vec!["-2", "1", "2"]);
}

/// `|e^x| = 2` for a plain `x`: up to 0.48 `Ok([ln 2, i*pi + ln 2])` as all
/// the roots; the solutions are the line `Re x = ln 2`.  Oracle (exact):
/// `x = ln 2 + i` gives `|2*e^i| = 2`.  Real `x`: `[ln 2]` unchanged.
#[test]
fn abs_of_exp_is_a_line() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "Abs(exp(x)) - 2");
    assert_exact_root(&e, &x, &parse(&ctx, "log(2) + I"));
    assert!(is_refusal(&e.solve(&x)), "{:?}", e.solve(&x));
    let xr = real(&ctx, "x");
    assert_eq!(shown(e.subs(&x, &xr).solve(&xr)), vec!["ln(2)"]);
}

/// The sign cases of `|u|` must all be solved before "no solution": for a
/// real `x`, `L_3^(-1)(|x - EulerGamma|) = 0` (golden case 48064) was
/// `NoSolution("no choice of the signs of the absolute values has a real
/// root")` up to 0.48 — each sign choice is a cubic with `EulerGamma` in
/// its coefficients, which the solver does not solve, and the unsolved
/// cases were read as having no root.  Oracle (exact):
/// `L_3^(-1)(t) = -t + t^2 - t^3/6 = -t(1 - t + t^2/6)` (DLMF 18.5.12)
/// vanishes at `t = 0` and `t = 3 +- sqrt 3`, so `x = EulerGamma` and
/// `EulerGamma +- (3 +- sqrt 3)` are five real roots.
#[test]
fn unsolved_abs_case_is_not_a_proof() {
    let ctx = Context::new();
    let x = real(&ctx, "x");
    let e = parse(&ctx, "assoc_laguerre(3, -1, Abs(x - EulerGamma))").subs(&ctx.symbol("x"), &x);
    let at = e.subs(&x, &parse(&ctx, "EulerGamma")).eval();
    assert!(at.is_zero_structural(), "{at}");
    let r = e.solve(&x);
    assert!(!matches!(r, Err(SymplexError::NoSolution { .. })), "{r:?}");
}

/// Roots found next to an unsolved alternative are not all the roots: up
/// to 0.48 `sin(x - x^4) = 0` gave `[1, 0, w, w^2]` (the roots of `x - x^4
/// = 0`, the branch `sin u = 0` at `u = 0`) as if complete, while the
/// branch `u = pi` (`x - x^4 = pi`, a quartic whose roots were not found)
/// holds four more (golden case 35140).  Oracle (mpmath, dps 30):
/// `polyroots([1, 0, 0, -1, pi])` gives the roots of `x^4 - x + pi`,
/// `0.94657882451921557580 +- 0.79492292522369182548i` and
/// `-0.94657882451921557580 +- 1.07708889373633812600i`, where
/// `|sin(x - x^4)| < 2e-30`.  Now `Ex::solve` refuses, naming the roots
/// found.
#[test]
fn roots_next_to_an_unsolved_branch_are_partial() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sin(x - x**4)");
    match e.solve(&x) {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("not necessarily all"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

/// A general solve that nests periodic families refuses as a whole (up to
/// 0.48 by the thread-local flag `PERIOD_CONFLICT`, now the outcome
/// `Reason::NestedPeriods`, unchanged behaviour): `sin(tan x) = 1/2` peels
/// to `tan x = asin(1/2) + 2n*pi` (and `pi - asin(1/2) + 2n*pi`), whose
/// inversion `x = atan(pi/6 + 2n*pi) + m*pi` needs a second integer
/// parameter.  Oracle (exact): `x = atan(pi/6 + 2*pi) + pi` solves it
/// (`tan` has period `pi`, `sin` period `2*pi`), and is in no family of one
/// parameter `n` alone.
#[test]
fn nested_periodic_families_refuse() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let e = parse(&ctx, "sin(tan(x)) - 1/2");
    let r = e.solve_general(&x);
    match r {
        Err(SymplexError::ComputationFailed { reason, .. }) => {
            assert!(reason.contains("nests periodic families"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}
