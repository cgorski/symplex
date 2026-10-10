//! The speed of `integrate` (0.39).  After 0.38 the Rubi harness took 80 s
//! of wall time (930 s of CPU) with 12 jobs; the trigonometric chapters it
//! had just opened spent most of it.  Each test is a reproducer of one cost
//! found then, with its answer checked by differentiation (`F′ − f` at
//! sample points, the parameters at the Rubi harness's values substituted
//! after integrating), and a time limit generous for a debug build on a
//! loaded CI machine.  The times quoted are release builds (CPU seconds).

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// The Rubi harness's parameter values.
const PARAMS: [(&str, (i64, i64)); 7] = [
    ("a", (6, 5)),
    ("b", (3, 4)),
    ("c", (5, 3)),
    ("d", (2, 7)),
    ("A", (19, 10)),
    ("B", (23, 12)),
    ("C", (29, 14)),
];

/// `∫ src dx` within `limit`, its answer checked by differentiation (or
/// unevaluated, when `closed` is false).
fn integrate_within(src: &str, limit: Duration, closed: bool) {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse(src).unwrap();
    let t0 = Instant::now();
    let big_f = f.integrate(&x);
    let dt = t0.elapsed();
    assert!(dt < limit, "∫ {src} took {dt:?}");
    if !closed {
        assert!(big_f.has_unevaluated(), "∫ {src} = {big_f}");
        return;
    }
    assert!(!big_f.has_unevaluated(), "∫ {src} = {big_f}");
    let bind = |e: &Ex| {
        let mut e = e.clone();
        for (name, (p, q)) in PARAMS {
            e = e.subs(&ctx.symbol(name), &ctx.rational(p, q));
        }
        e
    };
    let residual = bind(&(&big_f.diff(&x) - &f));
    let scale = bind(&f);
    let mut checked = 0;
    for point in ["1/3", "7/5", "-5/7", "13/11"] {
        let point = ctx.parse(point).unwrap();
        let (Ok(r), Ok(s)) = (
            residual.subs(&x, &point).eval_complex64(),
            scale.subs(&x, &point).eval_complex64(),
        ) else {
            continue;
        };
        assert!(
            r.norm() < 1e-9 * s.norm().max(1.0),
            "∫ {src} = {big_f}: F′ − f = {r} at {point}"
        );
        checked += 1;
    }
    assert!(checked > 0, "∫ {src} = {big_f}: no sample point evaluates");
}

/// The self-check compares `F′` with `f` at sample points.  Up to 0.38 it
/// evaluated `F′ − f` as one expression, a true zero for a right answer,
/// which `evalf` settles only by re-evaluating the whole of `F′` at up to
/// two and a half times the working precision; `F′` and `f` on their own
/// settle in one pass, and their distance decides the point unless it lies
/// near the tolerance.  The check was half of the integration time of the
/// trigonometric chapters: here 0.039 s → 0.021 s.
#[test]
fn the_self_check_evaluates_the_derivative_on_its_own() {
    integrate_within("1/(a + b*sec(x))^3", Duration::from_secs(5), true);
}

/// Every degenerate case of the parameters re-runs the substitution
/// `t = tan(x/2)` on pieces met before; the jumps at the poles of `tan(x/2)`
/// (with the leading terms of their logarithms' arguments, multiplied out),
/// the normalised integrands in `t`, and the tests that a u-substitution's
/// quotient varies are now remembered for the whole call.  Rubi 4.5.4.2-
/// type: 0.074 s → 0.048 s; with `c + d·x` for `x`, 2.8 s → 1.1 s.
#[test]
fn degenerate_cases_reuse_the_jumps_through_the_half_angle() {
    integrate_within(
        "(A + B*sec(x) + C*sec(x)^2)/(a + b*sec(x))^2",
        Duration::from_secs(5),
        true,
    );
}

/// Rubi 4.1.3.1-type: 0.139 s → 0.099 s; `(A + B·sin(e + f·x))/((a +
/// a·sin(e + f·x))²·(c + d·sin(e + f·x))³)` (Rubi 4.1.3.1 #278) took 9.6 s
/// (a timeout under load), 8.8 s of it in the leading terms of the same ten
/// polynomials, recomputed 16 times each; now 1.4 s.
#[test]
fn the_jump_of_one_antiderivative_is_taken_once() {
    integrate_within(
        "(A + B*sin(x))/((1 + sin(x))^2*(c + d*sin(x))^2)",
        Duration::from_secs(10),
        true,
    );
}

/// The Euler substitution's answer for `(a + i·a·tan u)^(m/2)·(c −
/// i·c·tan u)^(n/2)·(A + B·tan u)` (Rubi 4.3.3.1) has a single product of
/// thousands of nodes; following it to `t = ±∞` for the jump at the poles
/// of `tan u` cost the limit engine 10 s before its work budget ran out, and
/// the integration timed out (#816; now 4.5 s, unevaluated as before).  A
/// term over 2,000 nodes now leaves the jump undecided at once (on the Rubi
/// suite every term followed had at most 243).  The members that reach
/// that bound take seconds even in a release build; this smaller one (0.13
/// s → 0.06 s) keeps the family's time and its unevaluated answer in view.
#[test]
fn a_huge_term_does_not_go_to_the_limit_engine() {
    integrate_within(
        "(a + I*a*tan(x))^(3/2)*(A + B*tan(x))*(c - I*c*tan(x))^(5/2)",
        Duration::from_secs(10),
        false,
    );
}
