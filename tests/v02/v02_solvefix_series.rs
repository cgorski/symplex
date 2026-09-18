//! Regression tests for the 0.2 "silently wrong solver results" campaign:
//! `series_at_infinity` for functions with a finite limit at `∞` that is
//! not a Taylor point in `t = 1/x`.
//!
//! Root cause fixed: the differentiation fallback substituted `t = 0`
//! directly, so `atan(1/t)` produced the constant term `atan(zoo)` — an
//! expression that was neither flagged as unevaluated nor rejected.  The
//! coefficients are now limits (`t → 0⁺`) when direct substitution is
//! singular, and any result still containing `∞`/`zoo`/`NaN` is an error.

use symplex::prelude::*;

/// Series truncated to exponents `< n` in `1/x`, checked against `f` at
/// a large point: the truncation error must be at most `tol`.
fn check_asymptotic(f: &Ex, x: &Ex, n: u32, at: i64, tol: f64) -> Ex {
    let s = f
        .try_series_at_infinity(x, n)
        .unwrap_or_else(|e| panic!("no expansion of {f}: {e}"));
    let ctx = f.context();
    assert!(!s.contains(&ctx.complex_infinity()), "{s}");
    assert!(!s.contains(&ctx.infinity()), "{s}");
    assert!(!s.has_unevaluated(), "{s}");
    let got = s.subs_i64(x, at).eval_f64().unwrap();
    let want = f.subs_i64(x, at).eval_f64().unwrap();
    assert!(
        (got - want).abs() < tol,
        "{f} ~ {s}: at x = {at} series gives {got}, function {want}"
    );
    s
}

#[test]
fn atan_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // pi/2 - 1/x + 1/(3x^3) - 1/(5x^5) + …
    let s = check_asymptotic(&x.atan(), &x, 6, 10, 1e-7);
    let text = format!("{s}");
    assert!(text.contains("pi"), "{text}");
    // Exact coefficients: the 1/x^3 coefficient is 1/3.
    let c3 = (&s - ctx.pi() / 2 + 1 / &x).eval() * x.powi(3);
    let c3 = c3.expand().eval().subs_i64(&x, 1000).eval_f64().unwrap();
    assert!((c3 - (1.0 / 3.0 - 1.0 / (5.0 * 1e6))).abs() < 1e-12, "{c3}");
}

#[test]
fn atan_at_negative_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = x.atan().series_at_neg_infinity(&x, 6);
    assert!(!s.has_unevaluated(), "{s}");
    let got = s.subs_i64(&x, -10).eval_f64().unwrap();
    assert!((got - (-10f64).atan()).abs() < 1e-7, "{s} at -10 = {got}");
}

#[test]
fn tanh_at_infinity_is_one_up_to_exponentially_small_terms() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = check_asymptotic(&x.tanh(), &x, 5, 20, 1e-12);
    assert_eq!(format!("{s}"), "1");
}

#[test]
fn erf_at_infinity_is_one() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = check_asymptotic(&x.erf(), &x, 5, 10, 1e-12);
    assert_eq!(format!("{s}"), "1");
}

#[test]
fn rational_function_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x/(x+1) = 1 - 1/x + 1/x^2 - 1/x^3 + 1/x^4 - …
    let s = check_asymptotic(&(&x / (&x + 1)), &x, 5, 10, 2e-5);
    let at_10 = s.subs_i64(&x, 10).eval_f64().unwrap();
    assert!((at_10 - 0.9091).abs() < 1e-12, "{s}");
}

#[test]
fn sqrt_difference_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sqrt(x^2 + 1) - x = 1/(2x) - 1/(8x^3) + 1/(16x^5) - …
    let s = check_asymptotic(&((x.powi(2) + 1).sqrt() - &x), &x, 5, 10, 1e-6);
    let at_2 = s.subs_i64(&x, 2).eval_f64().unwrap();
    assert!((at_2 - (0.25 - 1.0 / 64.0)).abs() < 1e-12, "{s}");
}

#[test]
fn log_ratio_at_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // ln((x+1)/x) = 1/x - 1/(2x^2) + 1/(3x^3) - 1/(4x^4) + …
    let s = check_asymptotic(&((&x + 1) / &x).ln(), &x, 5, 10, 3e-6);
    let at_10 = s.subs_i64(&x, 10).eval_f64().unwrap();
    let want = 0.1 - 0.005 + 1.0 / 3000.0 - 1.0 / 40000.0;
    assert!((at_10 - want).abs() < 1e-12, "{s}");
}

#[test]
fn x_times_atan_of_reciprocal() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // x·atan(1/x) = 1 - 1/(3x^2) + 1/(5x^4) - …
    check_asymptotic(&(&x * (1 / &x).atan()), &x, 5, 10, 2e-7);
}

#[test]
fn oscillating_function_has_no_expansion() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    assert!(x.sin().try_series_at_infinity(&x, 5).is_err());
    let s = x.cos().series_at_infinity(&x, 5);
    assert!(s.has_unevaluated(), "{s}");
}

#[test]
fn no_result_ever_contains_a_singular_atom() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fs = [
        x.atan(),
        x.powi(2).atan(),
        (&x + 1).atan(),
        ctx.pi() / 2 - x.atan(),
        (1 / &x).exp() * &x - &x,
        x.exp() / (x.exp() + 1),
    ];
    for f in &fs {
        let s = f.series_at_infinity(&x, 4);
        if s.has_unevaluated() {
            continue; // honest failure is acceptable
        }
        assert!(
            !s.contains(&ctx.complex_infinity())
                && !s.contains(&ctx.infinity())
                && !s.contains(&ctx.neg_infinity()),
            "{f} -> {s}"
        );
        let got = s.subs_i64(&x, 50).eval_f64().unwrap();
        let want = f.subs_i64(&x, 50).eval_f64().unwrap();
        assert!((got - want).abs() < 1e-4, "{f} -> {s}: {got} vs {want}");
    }
}
