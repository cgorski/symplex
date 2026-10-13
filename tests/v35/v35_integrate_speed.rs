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
/// s → 0.06 s) keeps the family's time in view.  Since 0.44 it integrates
/// (the limit engine decides the leading terms it refused before; the
/// integrand has a non-integrable singularity at every pole of `tan x`,
/// so there is no jump to correct), checked by differentiation.
#[test]
fn a_huge_term_does_not_go_to_the_limit_engine() {
    integrate_within(
        "(a + I*a*tan(x))^(3/2)*(A + B*tan(x))*(c - I*c*tan(x))^(5/2)",
        Duration::from_secs(10),
        true,
    );
}

/// The algebraic remainder of a rational function whose denominator has a
/// quartic factor goes to the heuristic integrator, which meets Ferrari's
/// radicals at every step.  In 0.44 before the release, `eval` proved the
/// sign of each radicand again at every pass (a deep zero search and a
/// minimal polynomial), and the CI Fuzz workflow timed out on
/// `∫ 1/(x⁵ + x + 2) dx` (`fuzz_roundtrip`, 0.54 s → 5.8 s) and
/// `∫ ln(x + x⁻³ − 1) dx` (`fuzz_integrate`, 0.45 s → 4.0 s); the proofs
/// are now made once per context (0.70 s, 0.53 s).
#[test]
fn ferrari_radicands_are_decided_once() {
    integrate_within("1/(x^5 + x + 2)", Duration::from_secs(20), true);
    integrate_within("log(x + x^(-3) - 1)", Duration::from_secs(20), true);
}

/// A rational function whose denominator is over ℚ and whose numerator
/// has constant coefficients that are not (`t⁸ + t⁷/e`) is integrated as
/// `Σ κ·∫ P/D` over its distinct constants `κ`, each piece by the rational
/// integrator.  Neither rational integrator took it (`1/e` is neither in ℚ
/// nor a parameter), and the search found the same answer through nested
/// by parts and sum distributions: 6.4 s for the first integrand, and 3.6 s
/// in 0.46 (5.2 s in the 0.47 development code) for the second, whose
/// `x = t²` substitution and real and imaginary parts lead to it — the
/// slow `fuzz_integrate` unit of the 0.47 campaign (10 s in the
/// instrumented build; CI's limit is 30 s).  Now 0.012 s and 0.041 s; the
/// third integrand's answer shrank from 234 characters to 94.
#[test]
fn constant_coefficients_over_a_rational_denominator_split_by_constant() {
    integrate_within(
        "(x^8 + x^7*exp(-1))/(x^12 + 1/2)",
        Duration::from_secs(10),
        true,
    );
    integrate_within(
        "(sqrt(x) + exp(-1))/(x^3 + sqrt(2)*I/2)",
        Duration::from_secs(10),
        true,
    );
    integrate_within("(E*x^3 + pi)/(x^2 + 1)^3", Duration::from_secs(10), true);
}

/// The real-variable convention accepts an `F` whose derivative differs
/// from `f` only where `f` is complex: `∫ √x·|√x| dx = x²·sign(√x)/2` (by
/// `u = √x`; at `x < 0` the integrand is imaginary), as `∫ asin(√x)/x² dx
/// = −asin(√x)/x − √(1/x − 1)·sign(√x)` (Rubi 5.1.5 #365).  Times a
/// constant that is not real that no longer holds: `∫ √3·i·√x·|√x| dx` was
/// `√3·i·x²·sign(√x)/2`, and at `x = −5/7` the integrand is the real
/// `−1.2372`, the answer's derivative `+1.2372` (up to 0.46; the CI Fuzz
/// workflow's `fuzz_integrate`).  A non-real constant split off an
/// integrand is now checked against it when the antiderivative is not
/// analytic; with `|u|` of a real `u` (`u = x² − 1`, `u = sin x`) and real
/// constants nothing changes.
#[test]
fn non_real_constant_factors_keep_the_real_variable_convention() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let src = "sqrt(3)*sqrt(x)*abs(sqrt(x))*I";
    let big_f = ctx.parse(src).unwrap().integrate(&x);
    assert!(big_f.has_unevaluated(), "∫ {src} = {big_f}");
    // Checked where the integrand is real (0 < x < 1).
    let f = ctx.parse("asin(sqrt(x))/x^2").unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "∫ {f} = {big_f}");
    let residual = &big_f.diff(&x) - &f;
    for point in ["1/3", "1/2", "5/7"] {
        let p = ctx.parse(point).unwrap();
        let r = residual.subs(&x, &p).eval_complex64().unwrap();
        assert!(r.norm() < 1e-9, "∫ {f} = {big_f}: F′ − f = {r} at {point}");
    }
    integrate_within("2*x*abs(x^2 - 1)", Duration::from_secs(20), true);
    integrate_within("cos(x)*sign(sin(x))", Duration::from_secs(20), true);
}
