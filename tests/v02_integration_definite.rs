//! v0.2 definite / improper integration: closed forms are cross-checked
//! against adaptive Gauss–Kronrod quadrature, and every "diverges" claim
//! is exercised.

use symplex::definite::QuadOpts;
use symplex::prelude::*;

const TOL: f64 = 1e-9;

/// Symbolic result must exist, equal `expected` numerically, and agree with
/// the numeric quadrature.
fn check(label: &str, f: &Ex, x: &Ex, lo: &Ex, hi: &Ex, expected: f64) {
    let sym = f
        .try_integrate_definite(x, lo, hi)
        .unwrap_or_else(|e| panic!("{label}: symbolic integration failed: {e}"));
    let v = sym
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: result {sym} not numeric: {e}"));
    assert!(
        (v - expected).abs() < TOL * expected.abs().max(1.0),
        "{label}: symbolic {sym} = {v}, expected {expected}"
    );
    let num = f
        .integrate_numeric(x, lo, hi)
        .unwrap_or_else(|e| panic!("{label}: numeric quadrature failed: {e}"));
    assert!(
        (num - expected).abs() < 1e-7 * expected.abs().max(1.0),
        "{label}: quadrature {num}, expected {expected}"
    );
}

/// Symbolic result must exist and equal `expected`; the quadrature
/// cross-check is skipped (conditionally convergent or oscillatory tails).
fn check_symbolic(label: &str, f: &Ex, x: &Ex, lo: &Ex, hi: &Ex, expected: f64) {
    let sym = f
        .try_integrate_definite(x, lo, hi)
        .unwrap_or_else(|e| panic!("{label}: symbolic integration failed: {e}"));
    let v = sym
        .eval_f64()
        .unwrap_or_else(|e| panic!("{label}: result {sym} not numeric: {e}"));
    assert!(
        (v - expected).abs() < TOL * expected.abs().max(1.0),
        "{label}: symbolic {sym} = {v}, expected {expected}"
    );
}

/// Symbolic result must be a proven divergence; numeric must not converge.
fn check_divergent(label: &str, f: &Ex, x: &Ex, lo: &Ex, hi: &Ex) {
    match f.try_integrate_definite(x, lo, hi) {
        Err(SymplexError::Divergent { .. }) => {}
        other => panic!("{label}: expected Divergent, got {other:?}"),
    }
    let cas = f.integrate_definite(x, lo, hi);
    assert!(
        cas.has_unevaluated(),
        "{label}: CAS-style result should be unevaluated, got {cas}"
    );
    assert!(
        f.integrate_numeric(x, lo, hi).is_err(),
        "{label}: numeric quadrature should not converge"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Basics: bounds, orientation, symmetry, symbolic bounds
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn equal_bounds_is_zero() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = x.sin().integrate_definite(&x, &ctx.int(5), &ctx.int(5));
    assert_eq!(format!("{v}"), "0");
    let t = ctx.symbol("t");
    let v = (&x.exp() / &x).integrate_definite(&x, &t, &t);
    assert_eq!(format!("{v}"), "0");
}

#[test]
fn reversed_bounds_negate() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let fwd = x
        .powi(2)
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(3))
        .unwrap();
    let rev = x
        .powi(2)
        .try_integrate_definite(&x, &ctx.int(3), &ctx.int(0))
        .unwrap();
    assert_eq!(format!("{fwd}"), "9");
    assert_eq!(format!("{rev}"), "-9");
}

#[test]
fn odd_integrand_symmetric_interval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // continuous odd function with no elementary antiderivative
    let f = &x * &x.powi(4).sin();
    let v = f
        .try_integrate_definite(&x, &ctx.int(-2), &ctx.int(2))
        .unwrap();
    assert_eq!(format!("{v}"), "0");
    let v = x
        .powi(3)
        .try_integrate_definite(&x, &ctx.int(-1), &ctx.int(1))
        .unwrap();
    assert_eq!(format!("{v}"), "0");
}

#[test]
fn even_integrand_symmetric_interval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(
        "x² on [-1,1]",
        &x.powi(2),
        &x,
        &ctx.int(-1),
        &ctx.int(1),
        2.0 / 3.0,
    );
    check("cos on [-π,π]", &x.cos(), &x, &(-ctx.pi()), &ctx.pi(), 0.0);
}

#[test]
fn symbolic_upper_bound_smooth_integrand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = ctx.symbol("t");
    let v = x
        .powi(2)
        .try_integrate_definite(&x, &ctx.int(0), &t)
        .unwrap();
    assert_eq!(format!("{v}"), "1/3*t^3");
    let v = (-&x)
        .exp()
        .try_integrate_definite(&x, &ctx.int(0), &t)
        .unwrap();
    assert_eq!(format!("{v}"), "-exp(-t) + 1");
    let v = (&ctx.int(1) / &(&x.powi(2) + 1))
        .try_integrate_definite(&x, &ctx.int(0), &t)
        .unwrap();
    assert_eq!(format!("{v}"), "atan(t)");
}

#[test]
fn symbolic_bounds_with_undecidable_pole_stay_unevaluated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let t = ctx.symbol("t");
    let f = &ctx.int(1) / &x;
    let r = f.try_integrate_definite(&x, &ctx.int(1), &t);
    assert!(
        matches!(r, Err(SymplexError::ComputationFailed { .. })),
        "{r:?}"
    );
    assert!(f.integrate_definite(&x, &ctx.int(1), &t).has_unevaluated());
    // With t > 0 the pole at 0 is provably outside [1, t] (or [t, 1]).
    let tp = ctx.symbol_with("tp", &[Assumption::Positive]);
    let v = f.try_integrate_definite(&x, &ctx.int(1), &tp).unwrap();
    let shown = format!("{v}");
    assert!(shown == "ln(tp)" || shown == "ln(abs(tp))", "{shown}");
}

#[test]
fn symbolic_parameter_in_integrand() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    // ∫₀^∞ e^{−a x} dx = 1/a
    let v = (-&(&a * &x))
        .exp()
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert_eq!(format!("{}", v.simplify()), "1/a");
    // ∫₋∞^∞ e^{−a x²} dx = √(π/a)
    let v = (-&(&a * &x.powi(2)))
        .exp()
        .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    let at2 = v.subs(&a, &ctx.int(2)).eval_f64().unwrap();
    assert!(
        (at2 - (std::f64::consts::PI / 2.0).sqrt()).abs() < 1e-12,
        "{v}"
    );
    // Without a sign assumption the Gaussian must not be guessed.
    let b = ctx.symbol("b");
    let r = (-&(&b * &x.powi(2))).exp().try_integrate_definite(
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
    );
    assert!(r.is_err(), "{r:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Divergence detection — the headline "never silently wrong" cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interior_double_pole_diverges() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_divergent("1/x² on [-1,1]", &x.powi(-2), &x, &ctx.int(-1), &ctx.int(1));
    check_divergent(
        "1/(x-1)² on [0,2]",
        &(&x - 1).powi(-2),
        &x,
        &ctx.int(0),
        &ctx.int(2),
    );
}

#[test]
fn interior_simple_pole_diverges_not_principal_value() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_divergent(
        "1/x on [-1,1]",
        &(&ctx.int(1) / &x),
        &x,
        &ctx.int(-1),
        &ctx.int(1),
    );
    check_divergent(
        "1/(x-1/2) on [0,1]",
        &(&ctx.int(1) / &(&x - &ctx.rational(1, 2))),
        &x,
        &ctx.int(0),
        &ctx.int(1),
    );
}

#[test]
fn endpoint_and_infinite_divergences() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check_divergent(
        "1/x on [0,1]",
        &(&ctx.int(1) / &x),
        &x,
        &ctx.int(0),
        &ctx.int(1),
    );
    check_divergent(
        "1/x on [1,∞)",
        &(&ctx.int(1) / &x),
        &x,
        &ctx.int(1),
        &ctx.infinity(),
    );
    check_divergent("x on (-∞,∞)", &x, &x, &ctx.neg_infinity(), &ctx.infinity());
    check_divergent(
        "x/(1+x²) on (-∞,∞) (odd but divergent)",
        &(&x / &(&x.powi(2) + 1)),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
    );
    check_divergent(
        "1/(x ln x) on [2,∞)",
        &(&ctx.int(1) / &(&x * &x.ln())),
        &x,
        &ctx.int(2),
        &ctx.infinity(),
    );
    check_divergent("1 on [0,∞)", &ctx.int(1), &x, &ctx.int(0), &ctx.infinity());
    check_divergent("tan on [0,π]", &x.tan(), &x, &ctx.int(0), &ctx.pi());
}

#[test]
fn divergence_without_antiderivative() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // sin(x)/x² ~ 1/x near 0 — no elementary antiderivative.
    check_divergent(
        "sin(x)/x² on [0,1]",
        &(&x.sin() / &x.powi(2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Proper integrals via FTC
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ftc_basics() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check("x²", &x.powi(2), &x, &ctx.int(0), &ctx.int(1), 1.0 / 3.0);
    check("sin", &x.sin(), &x, &ctx.int(0), &ctx.pi(), 2.0);
    check(
        "exp",
        &x.exp(),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        std::f64::consts::E - 1.0,
    );
    check(
        "1/x on [1,e]",
        &(&ctx.int(1) / &x),
        &x,
        &ctx.int(1),
        &ctx.e(),
        1.0,
    );
    check(
        "1/(1+x²) on [0,1]",
        &(&ctx.int(1) / &(&x.powi(2) + 1)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        std::f64::consts::FRAC_PI_4,
    );
    check("x e^x", &(&x * &x.exp()), &x, &ctx.int(0), &ctx.int(1), 1.0);
    check(
        "x sin x",
        &(&x * &x.sin()),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        std::f64::consts::PI,
    );
    check(
        "cos² on [0,π]",
        &x.cos().powi(2),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        std::f64::consts::PI / 2.0,
    );
}

#[test]
fn antiderivative_discontinuity_is_handled_or_refused() {
    // ∫₀^{2π} dx/(2 + cos x) = 2π/√3.  A naive FTC through the
    // Weierstrass antiderivative atan(tan(x/2)/√3) gives 0.
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = &ctx.int(1) / &(&x.cos() + 2);
    let two_pi = &ctx.int(2) * &ctx.pi();
    let expected = 2.0 * std::f64::consts::PI / 3f64.sqrt();
    match f.try_integrate_definite(&x, &ctx.int(0), &two_pi) {
        Ok(v) => {
            let fv = v.eval_f64().unwrap();
            assert!((fv - expected).abs() < 1e-9, "{v} = {fv}");
        }
        Err(SymplexError::ComputationFailed { .. }) => {}
        Err(e) => panic!("{e}"),
    }
    let num = f.integrate_numeric(&x, &ctx.int(0), &two_pi).unwrap();
    assert!((num - expected).abs() < 1e-8);
}

// ═══════════════════════════════════════════════════════════════════════════
// Improper integrals: endpoint singularities and infinite bounds via limits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn improper_endpoint_singularities() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check("ln x on [0,1]", &x.ln(), &x, &ctx.int(0), &ctx.int(1), -1.0);
    check(
        "ln² x on [0,1]",
        &x.ln().powi(2),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        2.0,
    );
    check(
        "1/√x on [0,1]",
        &x.pow(&ctx.rational(-1, 2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        2.0,
    );
    check(
        "1/√(1−x²) on [0,1]",
        &(&ctx.int(1) - &x.powi(2)).pow(&ctx.rational(-1, 2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        std::f64::consts::FRAC_PI_2,
    );
    check(
        "1/√(1−x²) on [-1,1]",
        &(&ctx.int(1) - &x.powi(2)).pow(&ctx.rational(-1, 2)),
        &x,
        &ctx.int(-1),
        &ctx.int(1),
        std::f64::consts::PI,
    );
    check(
        "√(1−x²) on [-1,1]",
        &(&ctx.int(1) - &x.powi(2)).sqrt(),
        &x,
        &ctx.int(-1),
        &ctx.int(1),
        std::f64::consts::FRAC_PI_2,
    );
    check(
        "x ln x on [0,1]",
        &(&x * &x.ln()),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        -0.25,
    );
}

#[test]
fn improper_infinite_bounds() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(
        "e^{-x} on [0,∞)",
        &(-&x).exp(),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        1.0,
    );
    check(
        "x e^{-x} on [0,∞)",
        &(&x * &(-&x).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        1.0,
    );
    check(
        "1/x² on [1,∞)",
        &x.powi(-2),
        &x,
        &ctx.int(1),
        &ctx.infinity(),
        1.0,
    );
    check(
        "1/(1+x²) on (-∞,∞)",
        &(&ctx.int(1) / &(&x.powi(2) + 1)),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        std::f64::consts::PI,
    );
    check(
        "1/(1+x²) on [0,∞)",
        &(&ctx.int(1) / &(&x.powi(2) + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        std::f64::consts::FRAC_PI_2,
    );
    check(
        "e^{x} on (-∞,0]",
        &x.exp(),
        &x,
        &ctx.neg_infinity(),
        &ctx.int(0),
        1.0,
    );
    check(
        "1/(1+e^x) on [0,∞)",
        &(&ctx.int(1) / &(&x.exp() + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        std::f64::consts::LN_2,
    );
    check(
        "x e^{-x²} on [0,∞)",
        &(&x * &(-x.powi(2)).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        0.5,
    );
    // 1/(x ln² x) decays too slowly for the quadrature to reach 1e-10: symbolic only.
    check_symbolic(
        "1/(x ln² x) on [2,∞)",
        &(&ctx.int(1) / &(&x * &x.ln().powi(2))),
        &x,
        &ctx.int(2),
        &ctx.infinity(),
        1.0 / std::f64::consts::LN_2,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Known-value table
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn table_gaussian_family() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt_pi = std::f64::consts::PI.sqrt();
    check(
        "e^{-x²} on (-∞,∞)",
        &(-x.powi(2)).exp(),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        sqrt_pi,
    );
    check(
        "e^{-3x²} on (-∞,∞)",
        &(-&(&x.powi(2) * 3)).exp(),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        (std::f64::consts::PI / 3.0).sqrt(),
    );
    check(
        "e^{-x²} on [0,∞)",
        &(-x.powi(2)).exp(),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        sqrt_pi / 2.0,
    );
    check(
        "x² e^{-x²} on [0,∞)",
        &(&x.powi(2) * &(-x.powi(2)).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        sqrt_pi / 4.0,
    );
    check(
        "x⁴ e^{-x²} on (-∞,∞)",
        &(&x.powi(4) * &(-x.powi(2)).exp()),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        3.0 * sqrt_pi / 4.0,
    );
    check(
        "x³ e^{-x²} on (-∞,∞) (odd)",
        &(&x.powi(3) * &(-x.powi(2)).exp()),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        0.0,
    );
    // shifted Gaussian e^{-x² + 2x}
    check(
        "e^{-x²+2x} on (-∞,∞)",
        &(&(-x.powi(2)) + &(&x * 2)).exp(),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        sqrt_pi * std::f64::consts::E,
    );
    check(
        "e^{-x²} cos(2x) on [0,∞)",
        &(&(-x.powi(2)).exp() * &(&x * 2).cos()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        sqrt_pi / 2.0 * (-1.0f64).exp(),
    );
    check(
        "e^{-x²} cos(x) on (-∞,∞)",
        &(&(-x.powi(2)).exp() * &x.cos()),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        sqrt_pi * (-0.25f64).exp(),
    );
}

#[test]
fn table_gamma_family() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let sqrt_pi = std::f64::consts::PI.sqrt();
    check(
        "x³ e^{-x}",
        &(&x.powi(3) * &(-&x).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        6.0,
    );
    check(
        "x⁵ e^{-2x}",
        &(&x.powi(5) * &(-&(&x * 2)).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        120.0 / 64.0,
    );
    check(
        "x^{-1/2} e^{-x} = Γ(1/2)",
        &(&x.pow(&ctx.rational(-1, 2)) * &(-&x).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        sqrt_pi,
    );
    check(
        "e^{-3x}/√x",
        &(&(-&(&x * 3)).exp() / &x.sqrt()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        (std::f64::consts::PI / 3.0).sqrt(),
    );
    check(
        "x^{1/2} e^{-x} = Γ(3/2)",
        &(&x.sqrt() * &(-&x).exp()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        sqrt_pi / 2.0,
    );
    // symbolic s: ∫₀^∞ x^{s−1} e^{−x} = Γ(s)
    let s = ctx.symbol_with("s", &[Assumption::Positive]);
    let f = &x.pow(&(&s - 1)) * &(-&x).exp();
    let v = f
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    let at = v.subs(&s, &ctx.int(5)).eval_f64().unwrap();
    assert!((at - 24.0).abs() < 1e-9, "{v}");
}

#[test]
fn table_dirichlet_and_sinc() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    // sin(x)/x is only conditionally convergent: symbolic checks only.
    check_symbolic(
        "sin x / x on [0,∞)",
        &(&x.sin() / &x),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 2.0,
    );
    check_symbolic(
        "sin 3x / x on [0,∞)",
        &(&(&x * 3).sin() / &x),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 2.0,
    );
    check_symbolic(
        "sin(-2x)/x on [0,∞)",
        &(&(&x * -2).sin() / &x),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        -pi / 2.0,
    );
    check_symbolic(
        "sin x / x on (-∞,∞)",
        &(&x.sin() / &x),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        pi,
    );
    check_symbolic(
        "sin² x / x² on [0,∞)",
        &(&x.sin().powi(2) / &x.powi(2)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 2.0,
    );
    // symbolic a: (π/2) sign(a)
    let a = ctx.symbol("a");
    let v = (&(&a * &x).sin() / &x)
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    let shown = format!("{v}");
    assert!(shown.contains("sign(a)") && shown.contains("pi"), "{shown}");
    assert!((v.subs(&a, &ctx.int(-3)).eval_f64().unwrap() + pi / 2.0).abs() < 1e-12);
}

#[test]
fn table_rational_and_algebraic_half_line() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    check(
        "1/(x²+4) on [0,∞)",
        &(&ctx.int(1) / &(&x.powi(2) + 4)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 4.0,
    );
    check(
        "1/(x²+4)² on (-∞,∞)",
        &(&x.powi(2) + 4).powi(-2),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        pi / 16.0,
    );
    check(
        "1/(1+x⁴) on [0,∞)",
        &(&ctx.int(1) / &(&x.powi(4) + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / (4.0 * (pi / 4.0).sin()),
    );
    check(
        "1/(1+x³) on [0,∞)",
        &(&ctx.int(1) / &(&x.powi(3) + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / (3.0 * (pi / 3.0).sin()),
    );
    check(
        "x^{-1/2}/(1+x) on [0,∞)",
        &(&x.pow(&ctx.rational(-1, 2)) / &(&x + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi,
    );
    check(
        "x^{1/3}/(1+x²) on [0,∞)",
        &(&x.pow(&ctx.rational(1, 3)) / &(&x.powi(2) + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        (pi / 2.0) / (pi * 2.0 / 3.0).sin(),
    );
    check(
        "x/(1+x)³ = B(2,1) on [0,∞)",
        &(&x / &(&x + 1).powi(3)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        0.5,
    );
    // symbolic a > 0: ∫₀^∞ 1/(x²+a²) = π/(2a)
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let v = (&ctx.int(1) / &(&x.powi(2) + &a.powi(2)))
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    let at3 = v.subs(&a, &ctx.int(3)).eval_f64().unwrap();
    assert!((at3 - pi / 6.0).abs() < 1e-12, "{v}");
    // ∫₋∞^∞ 1/(x²+a²)² = π/(2a³)
    let v = (&x.powi(2) + &a.powi(2))
        .powi(-2)
        .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    let at2 = v.subs(&a, &ctx.int(2)).eval_f64().unwrap();
    assert!((at2 - pi / 16.0).abs() < 1e-12, "{v}");
}

#[test]
fn table_exponential_trig_and_logs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    check(
        "e^{-2x} sin 3x",
        &(&(-&(&x * 2)).exp() * &(&x * 3).sin()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        3.0 / 13.0,
    );
    check(
        "e^{-2x} cos 3x",
        &(&(-&(&x * 2)).exp() * &(&x * 3).cos()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        2.0 / 13.0,
    );
    check_symbolic(
        "e^{-x} sin x / x",
        &(&(&(-&x).exp() * &x.sin()) / &x),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 4.0,
    );
    check(
        "ln x/(1+x²) on [0,∞)",
        &(&x.ln() / &(&x.powi(2) + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        0.0,
    );
    check(
        "ln x/(4+x²) on [0,∞)",
        &(&x.ln() / &(&x.powi(2) + 4)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi * 2f64.ln() / 4.0,
    );
    check(
        "ln x/(1−x) on [0,1]",
        &(&x.ln() / &(&ctx.int(1) - &x)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        -pi * pi / 6.0,
    );
    check(
        "ln x/(1+x) on [0,1]",
        &(&x.ln() / &(&x + 1)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        -pi * pi / 12.0,
    );
    check(
        "ln(1+x)/x on [0,1]",
        &(&(&x + 1).ln() / &x),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        pi * pi / 12.0,
    );
    check(
        "ln x/(1−x²) on [0,1]",
        &(&x.ln() / &(&ctx.int(1) - &x.powi(2))),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        -pi * pi / 8.0,
    );
    check(
        "x ln² x on [0,1]",
        &(&x * &x.ln().powi(2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        0.25,
    );
}

#[test]
fn table_bose_fermi_and_hyperbolic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    check(
        "x/(e^x−1)",
        &(&x / &(&x.exp() - 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi * pi / 6.0,
    );
    check(
        "x³/(e^x−1)",
        &(&x.powi(3) / &(&x.exp() - 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi.powi(4) / 15.0,
    );
    check(
        "x/(e^x+1)",
        &(&x / &(&x.exp() + 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi * pi / 12.0,
    );
    check(
        "x³/(e^{2x}−1)",
        &(&x.powi(3) / &(&(&x * 2).exp() - 1)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi.powi(4) / 15.0 / 16.0,
    );
    check(
        "1/cosh x on [0,∞)",
        &(&ctx.int(1) / &x.cosh()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi / 2.0,
    );
    check(
        "1/cosh x on (-∞,∞)",
        &(&ctx.int(1) / &x.cosh()),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        pi,
    );
    check(
        "x/sinh x on [0,∞)",
        &(&x / &x.sinh()),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi * pi / 4.0,
    );
}

#[test]
fn table_fresnel_and_fractional_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let fresnel = (pi / 8.0).sqrt();
    // Fresnel integrals converge only conditionally; skip the quadrature cross-check.
    let v = x
        .powi(2)
        .sin()
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!((v.eval_f64().unwrap() - fresnel).abs() < 1e-12, "{v}");
    let v = x
        .powi(2)
        .cos()
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!((v.eval_f64().unwrap() - fresnel).abs() < 1e-12, "{v}");
    let v = (&x.powi(2) * 2)
        .sin()
        .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    assert!(
        (v.eval_f64().unwrap() - 2.0 * (pi / 16.0).sqrt()).abs() < 1e-12,
        "{v}"
    );
    // x^{-1/2} sin x = √(π/2)
    let v = (&x.sin() / &x.sqrt())
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!(
        (v.eval_f64().unwrap() - (pi / 2.0).sqrt()).abs() < 1e-12,
        "{v}"
    );
    let v = (&x.cos() / &x.sqrt())
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!(
        (v.eval_f64().unwrap() - (pi / 2.0).sqrt()).abs() < 1e-12,
        "{v}"
    );
}

#[test]
fn table_laplace_type() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    // ∫₀^∞ cos(bx)/(x²+a²) = π e^{−ab}/(2a), a = 2, b = 3
    // Oscillatory tails converge too slowly for plain adaptive quadrature: symbolic only.
    check_symbolic(
        "cos 3x/(x²+4)",
        &(&(&x * 3).cos() / &(&x.powi(2) + 4)),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        pi * (-6.0f64).exp() / 4.0,
    );
    // ∫₀^∞ x sin(bx)/(x²+a²) = (π/2) e^{−ab}: conditionally convergent → symbolic only
    let v = (&(&x * &(&x * 3).sin()) / &(&x.powi(2) + 4))
        .try_integrate_definite(&x, &ctx.int(0), &ctx.infinity())
        .unwrap();
    assert!(
        (v.eval_f64().unwrap() - pi / 2.0 * (-6.0f64).exp()).abs() < 1e-12,
        "{v}"
    );
}

#[test]
fn table_beta_on_unit_interval() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let one_minus_x = &ctx.int(1) - &x;
    check(
        "x²(1−x)³ = B(3,4)",
        &(&x.powi(2) * &one_minus_x.powi(3)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        1.0 / 60.0,
    );
    check(
        "√x √(1−x) = B(3/2,3/2)",
        &(&x.sqrt() * &one_minus_x.sqrt()),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        pi / 8.0,
    );
    check(
        "1/√(x(1−x)) = B(1/2,1/2)",
        &(&x * &one_minus_x).pow(&ctx.rational(-1, 2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        pi,
    );
    check(
        "x^{-1/2}(1−x)^{-1/2}",
        &(&x.pow(&ctx.rational(-1, 2)) * &one_minus_x.pow(&ctx.rational(-1, 2))),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        pi,
    );
    check(
        "(1−x)^{-1/2}",
        &one_minus_x.pow(&ctx.rational(-1, 2)),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        2.0,
    );
    check(
        "x^{1/2}(1−x²)^{-1/2}",
        &(&x.sqrt() * &(&ctx.int(1) - &x.powi(2)).pow(&ctx.rational(-1, 2))),
        &x,
        &ctx.int(0),
        &ctx.int(1),
        0.5 * gamma(0.75) * gamma(0.5) / gamma(1.25),
    );
    // symbolic exponents: ∫₀¹ x^a (1−x)^b = B(a+1, b+1)
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let b = ctx.symbol_with("b", &[Assumption::Positive]);
    let f = &x.pow(&a) * &one_minus_x.pow(&b);
    let v = f
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(1))
        .unwrap();
    let at = v
        .subs(&a, &ctx.int(2))
        .subs(&b, &ctx.int(3))
        .eval_f64()
        .unwrap();
    assert!((at - 1.0 / 60.0).abs() < 1e-12, "{v}");
}

/// Lanczos approximation of Γ for test expectations.
fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * z).sin() * gamma(1.0 - z));
    }
    let g = 7.0;
    let c = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let mut a = c[0];
    let t = z + g + 0.5;
    for (i, &ci) in c.iter().enumerate().skip(1) {
        a += ci / (z + i as f64);
    }
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
}

#[test]
fn table_wallis_and_orthogonality() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let half_pi = &ctx.pi() / 2;
    let two_pi = &ctx.pi() * 2;
    check(
        "sin⁴ on [0,π/2]",
        &x.sin().powi(4),
        &x,
        &ctx.int(0),
        &half_pi,
        3.0 * pi / 16.0,
    );
    check(
        "cos⁵ on [0,π/2]",
        &x.cos().powi(5),
        &x,
        &ctx.int(0),
        &half_pi,
        8.0 / 15.0,
    );
    check(
        "sin² cos² on [0,π/2]",
        &(&x.sin().powi(2) * &x.cos().powi(2)),
        &x,
        &ctx.int(0),
        &half_pi,
        pi / 16.0,
    );
    check(
        "sin³ cos² on [0,π/2]",
        &(&x.sin().powi(3) * &x.cos().powi(2)),
        &x,
        &ctx.int(0),
        &half_pi,
        2.0 / 15.0,
    );
    check(
        "sin⁶ on [0,π]",
        &x.sin().powi(6),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        5.0 * pi / 16.0,
    );
    check(
        "cos³ on [0,π]",
        &x.cos().powi(3),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        0.0,
    );
    check(
        "sin⁴ on [0,2π]",
        &x.sin().powi(4),
        &x,
        &ctx.int(0),
        &two_pi,
        3.0 * pi / 4.0,
    );
    check(
        "√sin on [0,π/2]",
        &x.sin().sqrt(),
        &x,
        &ctx.int(0),
        &half_pi,
        0.5 * gamma(0.75) * gamma(0.5) / gamma(1.25),
    );
    check(
        "sin 2x sin 3x on [0,π]",
        &(&(&x * 2).sin() * &(&x * 3).sin()),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        0.0,
    );
    check(
        "sin 3x sin 3x on [0,π]",
        &(&(&x * 3).sin() * &(&x * 3).sin()),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        pi / 2.0,
    );
    check(
        "cos 2x cos 5x on [-π,π]",
        &(&(&x * 2).cos() * &(&x * 5).cos()),
        &x,
        &(-ctx.pi()),
        &ctx.pi(),
        0.0,
    );
    check(
        "cos 4x cos 4x on [0,2π]",
        &(&(&x * 4).cos() * &(&x * 4).cos()),
        &x,
        &ctx.int(0),
        &two_pi,
        pi,
    );
    check(
        "sin 2x cos 3x on [0,π]",
        &(&(&x * 2).sin() * &(&x * 3).cos()),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        4.0 / (4.0 - 9.0),
    );
}

#[test]
fn table_periodic_rational_trig() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pi = std::f64::consts::PI;
    let two_pi = &ctx.pi() * 2;
    check(
        "1/(3+cos x) on [0,2π]",
        &(&ctx.int(1) / &(&x.cos() + 3)),
        &x,
        &ctx.int(0),
        &two_pi,
        2.0 * pi / 8f64.sqrt(),
    );
    check(
        "1/(3+2 sin x) on [0,2π]",
        &(&ctx.int(1) / &(&(&x.sin() * 2) + 3)),
        &x,
        &ctx.int(0),
        &two_pi,
        2.0 * pi / 5f64.sqrt(),
    );
    check(
        "1/(2+cos x) on [0,π]",
        &(&ctx.int(1) / &(&x.cos() + 2)),
        &x,
        &ctx.int(0),
        &ctx.pi(),
        pi / 3f64.sqrt(),
    );
    check(
        "1/(2+cos x)² on [0,2π]",
        &(&x.cos() + 2).powi(-2),
        &x,
        &ctx.int(0),
        &two_pi,
        2.0 * pi * 2.0 / 3f64.powf(1.5),
    );
    // a ≤ |b| → not applicable (integrand has a pole); must not return a number.
    let f = &ctx.int(1) / &(&x.cos() + 1);
    let r = f.try_integrate_definite(&x, &ctx.int(0), &two_pi);
    assert!(r.is_err(), "{r:?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Special integrands
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dirac_delta() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let d = (&x - 2).dirac_delta();
    let f = &x.powi(3) * &d;
    assert_eq!(
        format!(
            "{}",
            f.try_integrate_definite(&x, &ctx.int(0), &ctx.int(5))
                .unwrap()
        ),
        "8"
    );
    assert_eq!(
        format!(
            "{}",
            f.try_integrate_definite(&x, &ctx.int(3), &ctx.int(5))
                .unwrap()
        ),
        "0"
    );
    assert_eq!(
        format!(
            "{}",
            f.try_integrate_definite(&x, &ctx.int(-5), &ctx.int(1))
                .unwrap()
        ),
        "0"
    );
    // scaled argument: δ(2x − 4) → 1/2 · g(2)
    let d2 = (&(&x * 2) - 4).dirac_delta();
    let v = (&x.exp() * &d2)
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(5))
        .unwrap();
    assert!(
        (v.eval_f64().unwrap() - 0.5 * 2f64.exp()).abs() < 1e-12,
        "{v}"
    );
    // delta at an endpoint is convention-dependent → not evaluated
    assert!(
        f.try_integrate_definite(&x, &ctx.int(2), &ctx.int(5))
            .is_err()
    );
    // bare δ(x) over the whole line
    assert_eq!(
        format!(
            "{}",
            x.dirac_delta()
                .try_integrate_definite(&x, &ctx.neg_infinity(), &ctx.infinity())
                .unwrap()
        ),
        "1"
    );
}

#[test]
fn heaviside() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let h = (&x - 1).heaviside();
    check(
        "x H(x−1) on [0,2]",
        &(&x * &h),
        &x,
        &ctx.int(0),
        &ctx.int(2),
        1.5,
    );
    check("H(x−1) on [3,4]", &h, &x, &ctx.int(3), &ctx.int(4), 1.0);
    check("H(x−1) on [-3,0]", &h, &x, &ctx.int(-3), &ctx.int(0), 0.0);
    let h2 = (&ctx.int(1) - &x).heaviside();
    check(
        "x² H(1−x) on [0,2]",
        &(&x.powi(2) * &h2),
        &x,
        &ctx.int(0),
        &ctx.int(2),
        1.0 / 3.0,
    );
    check(
        "e^{-x} H(x−1) on [0,∞)",
        &(&(-&x).exp() * &h),
        &x,
        &ctx.int(0),
        &ctx.infinity(),
        (-1.0f64).exp(),
    );
}

#[test]
fn abs_and_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    check(
        "|x| on [-1,2]",
        &x.abs(),
        &x,
        &ctx.int(-1),
        &ctx.int(2),
        2.5,
    );
    check(
        "|x²−1| on [0,2]",
        &(&x.powi(2) - 1).abs(),
        &x,
        &ctx.int(0),
        &ctx.int(2),
        2.0,
    );
    check(
        "|sin x| on [0,2π]",
        &x.sin().abs(),
        &x,
        &ctx.int(0),
        &(&ctx.pi() * 2),
        4.0,
    );
    check(
        "x sign(x) on [-2,3]",
        &(&x * &x.sign()),
        &x,
        &ctx.int(-2),
        &ctx.int(3),
        6.5,
    );
    check(
        "e^{-|x|} on (-∞,∞)",
        &(-x.abs()).exp(),
        &x,
        &ctx.neg_infinity(),
        &ctx.infinity(),
        2.0,
    );
    check(
        "|x−1|³ on [0,3]",
        &(&x - 1).abs().powi(3),
        &x,
        &ctx.int(0),
        &ctx.int(3),
        0.25 + 4.0,
    );
}

#[test]
fn piecewise_and_floor() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // f = x² for x < 1, 2 − x otherwise
    let cond = x.lt(&ctx.int(1));
    let f = Ex::piecewise(&[
        (&x.powi(2), &cond),
        (&(&ctx.int(2) - &x), &ctx.int(1).ge(&ctx.int(0))),
    ]);
    let v = f
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(2))
        .unwrap();
    assert_eq!(format!("{v}"), "5/6");
    // nested condition with And
    let c1 = x.ge(&ctx.int(0)).and(&x.lt(&ctx.int(1)));
    let g = Ex::piecewise(&[
        (&ctx.int(1), &c1),
        (&ctx.int(0), &ctx.int(1).ge(&ctx.int(0))),
    ]);
    let v = g
        .try_integrate_definite(&x, &ctx.int(-3), &ctx.int(3))
        .unwrap();
    assert_eq!(format!("{v}"), "1");
    // floor on integer-aligned bounds: Σ_{k=0}^{3} k = 6
    let v = x
        .floor()
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(4))
        .unwrap();
    assert_eq!(format!("{v}"), "6");
    // floor on non-aligned bounds: ∫_{0.5}^{2.5} ⌊x⌋ = 0·0.5 + 1·1 + 2·0.5 = 2
    let v = x
        .floor()
        .try_integrate_definite(&x, &ctx.rational(1, 2), &ctx.rational(5, 2))
        .unwrap();
    assert_eq!(format!("{v}"), "2");
}

// ═══════════════════════════════════════════════════════════════════════════
// Numeric quadrature
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadrature_basic_and_options() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = x
        .powi(2)
        .integrate_numeric(&x, &ctx.int(0), &ctx.int(1))
        .unwrap();
    assert!((v - 1.0 / 3.0).abs() < 1e-13);
    let (v, err) = x
        .exp()
        .integrate_numeric_with(&x, &ctx.int(0), &ctx.int(1), &QuadOpts::default())
        .unwrap();
    assert!((v - (std::f64::consts::E - 1.0)).abs() < 1e-12);
    assert!(err < 1e-10);
    // reversed bounds
    let v = x.integrate_numeric(&x, &ctx.int(2), &ctx.int(0)).unwrap();
    assert!((v + 2.0).abs() < 1e-13);
    // oscillatory
    let v = (&x * 50)
        .sin()
        .powi(2)
        .integrate_numeric(&x, &ctx.int(0), &ctx.pi())
        .unwrap();
    assert!((v - std::f64::consts::PI / 2.0).abs() < 1e-9, "{v}");
}

#[test]
fn quadrature_errors() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let a = ctx.symbol("a");
    assert!(matches!(
        (&a * &x).integrate_numeric(&x, &ctx.int(0), &ctx.int(1)),
        Err(SymplexError::FreeSymbol { .. })
    ));
    assert!(matches!(
        x.gamma().integrate_numeric(&x, &ctx.int(1), &ctx.int(2)),
        Err(SymplexError::NotImplemented(_))
    ));
    assert!(matches!(
        x.integrate_numeric(&x, &ctx.int(0), &a),
        Err(SymplexError::Unevaluable { .. })
    ));
    assert!(
        x.powi(-2)
            .integrate_numeric(&x, &ctx.int(-1), &ctx.int(1))
            .is_err()
    );
}

#[test]
fn quadrature_infinite_and_singular() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let v = (&ctx.int(1) / &(&x.powi(2) + 1))
        .integrate_numeric(&x, &ctx.neg_infinity(), &ctx.infinity())
        .unwrap();
    assert!((v - std::f64::consts::PI).abs() < 1e-10);
    let v = x
        .ln()
        .integrate_numeric(&x, &ctx.int(0), &ctx.int(1))
        .unwrap();
    assert!((v + 1.0).abs() < 1e-9, "{v}");
    let v = x
        .exp()
        .integrate_numeric(&x, &ctx.neg_infinity(), &ctx.int(0))
        .unwrap();
    assert!((v - 1.0).abs() < 1e-10);
}
