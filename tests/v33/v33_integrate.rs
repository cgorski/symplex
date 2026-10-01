//! The integrator's search: a by-parts cycle, the order of the `|g|` rule,
//! and sums whose terms need different whole-integrand strategies.

use std::time::{Duration, Instant};
use symplex::prelude::*;

/// `F′ = f` at a few real points (30-digit evaluation).
fn assert_antiderivative(ctx: &Context, f: &Ex, big_f: &Ex, x: &Ex) {
    let df = big_f.diff(x);
    for p in ["3/10", "7/5", "13/4"] {
        let pt = ctx.parse(p).unwrap();
        let resid = (&df - f).subs(x, &pt).eval_complex64().unwrap();
        let scale = f.subs(x, &pt).eval_complex64().unwrap().norm().max(1.0);
        assert!(resid.norm() < 1e-9 * scale, "at {p}: F' - f = {resid}");
    }
}

/// The local `fuzz_integrate` timeout before the 0.33 release:
/// `∫ (√x + |x|)/(x + x⁻² + 1) dx` was refused after 74 s (51 s in 0.32).
/// By parts ran before the rule for `P·|g|`: on `x²·|x|/(x³ + x² + 1)` it
/// took `u = x²` and integrated `v·du` of the Cardano-root antiderivative
/// (17,000-character integrands).  The `|g|` rule now runs first, and the
/// terms of the distributed sum get every stage (`x = s²` for the `√x`
/// term), so the integral has a closed form.  (SymPy 1.14:
/// `integrate((sqrt(x) + Abs(x))/(x + x**-2 + 1), x)` did not finish in
/// 100 s.)
#[test]
fn abs_and_root_over_a_cubic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("(sqrt(x) + abs(x))/(x + x^(-2) + 1)").unwrap();
    let t = Instant::now();
    let big_f = f.integrate(&x);
    assert!(t.elapsed() < Duration::from_secs(20), "{:?}", t.elapsed());
    assert!(!big_f.has_unevaluated(), "{big_f}");
    assert_antiderivative(&ctx, &f, &big_f, &x);
}

/// By parts on `x²·sign(x)` leaves `∫ v·du = 2·∫ x²·sign(x)`, the integral
/// itself: `I = x³·sign(x) − 2I`.  The integrator recursed into the same
/// integral down to its depth limit (and, inside larger integrands, with
/// every other strategy at every level); the cycle is now solved, as
/// SymPy's `manualintegrate` does.  SymPy: `integrate(x**2*sign(x), x)` →
/// `x**3*sign(x)/3`.
#[test]
fn by_parts_cycle_is_solved() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (src, want) in [
        ("x^2*sign(x)", "1/3*x^3*sign(x)"),
        ("x*abs(x)", "1/3*x^3*sign(x)"),
    ] {
        let f = ctx.parse(src).unwrap();
        let big_f = f.integrate(&x);
        assert_eq!(big_f.to_string(), want, "{src}");
        assert_antiderivative(&ctx, &f, &big_f, &x);
    }
}

/// A sum whose terms each have a closed form by a different whole-integrand
/// strategy: `∫ x^(5/2)/(x³ + x² + 1)` needs `x = s²`, which refuses the
/// sum because of the `|x|` term.  The second term stayed unevaluated.
#[test]
fn terms_of_a_sum_through_every_stage() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx
        .parse("x^2*abs(x)/(x^3 + x^2 + 1) + x^(5/2)/(x^3 + x^2 + 1)")
        .unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    assert_antiderivative(&ctx, &f, &big_f, &x);
}
