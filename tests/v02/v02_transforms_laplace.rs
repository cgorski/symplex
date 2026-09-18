//! symplex 0.2 — Laplace transform extensions: `t^ν`, `ln t`, `δ`/`H`
//! delays, Bessel and error functions, division by `t`, inverse of
//! arbitrary proper rational functions, delays, and the initial/final
//! value theorems.
//!
//! Forward entries are verified against `∫₀^∞ f(t) e^{−s₀t} dt` by
//! quadrature at a few real `s₀`.

use symplex::definite::QuadOpts;
use symplex::prelude::*;

fn lt(f: &Ex, t: &Ex, s: &Ex) -> Ex {
    f.try_laplace(t, s)
        .unwrap_or_else(|e| panic!("L{{{f}}} failed: {e}"))
}

fn ilt(big_f: &Ex, s: &Ex, t: &Ex) -> Ex {
    big_f
        .try_inverse_laplace(s, t)
        .unwrap_or_else(|e| panic!("L⁻¹{{{big_f}}} failed: {e}"))
}

/// Compare `F(s₀)` with the defining integral at real `s₀ > 0`.
fn verify(f: &Ex, t: &Ex, s: &Ex, big_f: &Ex, s_samples: &[f64]) {
    let ctx = f.context();
    let opts = QuadOpts {
        rel_tol: 1e-8,
        ..QuadOpts::default()
    };
    for &s0 in s_samples {
        let s0_ex = ctx.from_f64(s0).unwrap();
        let integrand = f * (-&s0_ex * t).exp();
        let (num, err) = integrand
            .integrate_numeric_with(t, &ctx.int(0), &ctx.infinity(), &opts)
            .unwrap_or_else(|e| panic!("quadrature of {integrand} failed: {e}"));
        let sym = big_f
            .subs(s, &s0_ex)
            .eval_f64()
            .unwrap_or_else(|e| panic!("{big_f} at s={s0}: {e}"));
        let scale = sym.abs().max(1.0);
        assert!(
            err < 1e-4 * scale,
            "L{{{f}}}: quadrature error {err:e} at s={s0}"
        );
        assert!(
            (num - sym).abs() < 1e-5 * scale,
            "L{{{f}}} = {big_f}: at s={s0} integral {num} vs {sym}"
        );
    }
}

fn assert_same(a: &Ex, b: &Ex, label: &str) {
    if a == b || (a - b).simplify().is_zero_structural() {
        return;
    }
    let ctx = a.context();
    let mut syms = a.free_symbols();
    for s in b.free_symbols() {
        if !syms.contains(&s) {
            syms.push(s);
        }
    }
    for base in [0.6f64, 1.3, 2.1] {
        let mut ea = a.clone();
        let mut eb = b.clone();
        for (j, s) in syms.iter().enumerate() {
            let v = ctx.from_f64(base + 0.29 * j as f64).unwrap();
            ea = ea.subs(s, &v);
            eb = eb.subs(s, &v);
        }
        let va = ea
            .eval_f64()
            .unwrap_or_else(|e| panic!("{label}: {a}: {e}"));
        let vb = eb
            .eval_f64()
            .unwrap_or_else(|e| panic!("{label}: {b}: {e}"));
        assert!(
            (va - vb).abs() < 1e-9 * vb.abs().max(1.0),
            "{label}: {a} ≠ {b} ({va} vs {vb})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward table extensions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fractional_and_symbolic_powers_via_gamma() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let pi = ctx.pi();

    let f = t.sqrt();
    let big_f = lt(&f, &t, &s);
    assert_same(
        &big_f,
        &(pi.sqrt() / (2 * s.pow(&ctx.rational(3, 2)))),
        "√t",
    );
    verify(&f, &t, &s, &big_f, &[1.0, 2.5]);

    let f = 1 / t.sqrt();
    let big_f = lt(&f, &t, &s);
    assert_same(&big_f, &(pi.sqrt() / s.sqrt()), "1/√t");
    verify(&f, &t, &s, &big_f, &[1.0, 3.0]);

    let f = t.pow(&ctx.rational(5, 2));
    let big_f = lt(&f, &t, &s);
    assert_same(
        &big_f,
        &(ctx.rational(15, 8) * pi.sqrt() / s.pow(&ctx.rational(7, 2))),
        "t^{5/2}",
    );

    let big_f = lt(&t.pow(&a), &t, &s);
    assert_eq!(big_f, (&a + 1).gamma() * s.pow(&(-&a - 1)));

    // ν ≤ −1 is not transformable; unknown ν is an error, not a guess.
    assert!(t.powi(-1).try_laplace(&t, &s).is_err());
    let b = ctx.symbol("b");
    assert!(t.pow(&b).try_laplace(&t, &s).is_err());
}

#[test]
fn logarithm_delta_and_step() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let gamma = ctx.euler_gamma();

    let f = t.ln();
    let big_f = lt(&f, &t, &s);
    assert_same(&big_f, &(-(&gamma + s.ln()) / &s), "ln t");
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);
    // ln(t) e^{−t} via the frequency shift
    assert_same(
        &lt(&(t.ln() * (-&t).exp()), &t, &s),
        &(-(&gamma + (&s + 1).ln()) / (&s + 1)),
        "ln(t) e^{-t}",
    );

    assert_eq!(lt(&t.dirac_delta(), &t, &s), ctx.int(1));
    assert_eq!(lt(&(&t - 2).dirac_delta(), &t, &s), (-2 * &s).exp());
    assert_eq!(lt(&(&t - &a).dirac_delta(), &t, &s), (-&a * &s).exp());
    assert_eq!(lt(&(&t - 3).heaviside(), &t, &s), (-3 * &s).exp() / &s);
    assert_eq!(lt(&(&t - &a).heaviside(), &t, &s), (-&a * &s).exp() / &s);
    // A delta at a negative time contributes nothing on [0, ∞).
    assert_eq!(lt(&(&t + 2).dirac_delta(), &t, &s), ctx.int(0));
    // Unknown shift sign → Err.
    let b = ctx.symbol("b");
    assert!((&t - &b).heaviside().try_laplace(&t, &s).is_err());
}

#[test]
fn bessel_and_error_functions() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);

    let f = t.bessel_j(&ctx.int(0));
    let big_f = lt(&f, &t, &s);
    assert_eq!(big_f, 1 / (s.powi(2) + 1).sqrt());
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);

    let big_f = lt(&(&a * &t).bessel_j(&ctx.int(0)), &t, &s);
    assert_eq!(big_f, 1 / (s.powi(2) + a.powi(2)).sqrt());

    let f = t.bessel_j(&ctx.int(1));
    let big_f = lt(&f, &t, &s);
    assert_same(
        &big_f,
        &(((s.powi(2) + 1).sqrt() - &s) / (s.powi(2) + 1).sqrt()),
        "J₁",
    );
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);

    let f = t.sqrt().erf();
    let big_f = lt(&f, &t, &s);
    assert_eq!(big_f, 1 / (&s * (&s + 1).sqrt()));
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);
    let big_f = lt(&(&a * t.sqrt()).erf(), &t, &s);
    assert_eq!(big_f, &a / (&s * (&s + a.powi(2)).sqrt()));
}

#[test]
fn division_by_t() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);

    let f = t.sin() / &t;
    let big_f = lt(&f, &t, &s);
    assert_eq!(big_f, (1 / &s).atan());
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);
    assert_eq!(lt(&((&a * &t).sin() / &t), &t, &s), (&a / &s).atan());

    let f = (1 - (2 * &t).cos()) / &t;
    let big_f = lt(&f, &t, &s);
    assert_same(&big_f, &((1 + 4 / s.powi(2)).ln() / 2), "(1 − cos 2t)/t");
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);

    // General rule via ∫_s^∞ F(u) du (antiderivative + limit): (e^{−t} − e^{−2t})/t → ln((s+2)/(s+1))
    let f = ((-&t).exp() - (-2 * &t).exp()) / &t;
    let big_f = lt(&f, &t, &s);
    assert_same(
        &big_f,
        &((&s + 2).ln() - (&s + 1).ln()),
        "(e^{-t} − e^{-2t})/t",
    );
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);

    // Not integrable at 0 → Err.
    assert!((1 / &t).try_laplace(&t, &s).is_err());
    assert!((t.cos() / &t).try_laplace(&t, &s).is_err());
}

#[test]
fn delays_and_frequency_differentiation() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);

    assert_eq!(
        lt(&((&t - 2).sin() * (&t - 2).heaviside()), &t, &s),
        (-2 * &s).exp() / (s.powi(2) + 1)
    );
    assert_same(
        &lt(&((&t - 2).powi(2) * (&t - 2).heaviside()), &t, &s),
        &(2 * (-2 * &s).exp() / s.powi(3)),
        "(t−2)²H(t−2)",
    );
    // Symbolic delay.
    assert_eq!(
        lt(&((&t - &a).exp() * (&t - &a).heaviside()), &t, &s),
        (-&a * &s).exp() / (&s - 1)
    );
    // tⁿ f(t) → (−1)ⁿ F⁽ⁿ⁾(s)
    let f = &t * t.sin();
    let big_f = lt(&f, &t, &s);
    assert_same(&big_f, &(2 * &s / (s.powi(2) + 1).powi(2)), "t sin t");
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);
    let f = t.powi(2) * (2 * &t).cos();
    let big_f = lt(&f, &t, &s);
    verify(&f, &t, &s, &big_f, &[1.0, 2.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_of_proper_rational_functions() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");

    // Repeated real roots.
    assert_eq!(
        ilt(&(1 / (&s + 1).powi(3)), &s, &t),
        t.powi(2) * (-&t).exp() / 2
    );
    assert_same(
        &ilt(&(&s / ((&s + 1).powi(2) * (&s + 2))), &s, &t),
        &(2 * (-&t).exp() - &t * (-&t).exp() - 2 * (-2 * &t).exp()),
        "s/((s+1)²(s+2))",
    );
    // Complex roots.
    assert_same(
        &ilt(&(1 / ((s.powi(2) + 1) * (s.powi(2) + 4))), &s, &t),
        &(t.sin() / 3 - (2 * &t).sin() / 6),
        "1/((s²+1)(s²+4))",
    );
    assert_eq!(
        ilt(&((&s + 1) / (s.powi(2) + 2 * &s + 5)), &s, &t),
        (2 * &t).cos() * (-&t).exp()
    );
    assert_same(
        &ilt(&(1 / (s.powi(4) - 1)), &s, &t),
        &((t.exp() - (-&t).exp()) / 4 - t.sin() / 2),
        "1/(s⁴−1)",
    );
    assert_same(
        &ilt(&(1 / (s.powi(2) * (&s + 1))), &s, &t),
        &(&t - 1 + (-&t).exp()),
        "1/(s²(s+1))",
    );
    // Improper: polynomial part becomes δ.
    assert_same(
        &ilt(&(s.powi(2) / (s.powi(2) + 1)), &s, &t),
        &(t.dirac_delta() - t.sin()),
        "s²/(s²+1)",
    );
    assert_eq!(ilt(&ctx.int(1), &s, &t), t.dirac_delta());

    // Round trips through partial fractions.
    for f in [
        t.powi(3) * (-2 * &t).exp(),
        t.sin() * (-&t).exp() + (3 * &t).cos(),
        &t * (2 * &t).cosh(),
    ] {
        let big_f = lt(&f, &t, &s).together();
        assert_same(&ilt(&big_f, &s, &t), &f, &format!("round trip {f}"));
    }
}

#[test]
fn inverse_special_entries_and_delays() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]);
    let pi = ctx.pi();

    assert_eq!(
        ilt(&((-2 * &s).exp() / (&s + 1)), &s, &t),
        (2 - &t).exp() * (&t - 2).heaviside()
    );
    assert_eq!(ilt(&((-2 * &s).exp() / &s), &s, &t), (&t - 2).heaviside());
    assert_eq!(ilt(&(-2 * &s).exp(), &s, &t), (&t - 2).dirac_delta());
    assert_same(
        &ilt(&(1 / s.sqrt()), &s, &t),
        &(1 / (t.sqrt() * pi.sqrt())),
        "1/√s",
    );
    assert_eq!(
        ilt(&s.pow(&ctx.rational(-3, 2)), &s, &t),
        2 * t.sqrt() / pi.sqrt()
    );
    assert_eq!(
        ilt(&(1 / (s.powi(2) + 1).sqrt()), &s, &t),
        t.bessel_j(&ctx.int(0))
    );
    assert_eq!(
        ilt(&(1 / (s.powi(2) + a.powi(2)).sqrt()), &s, &t),
        (&a * &t).bessel_j(&ctx.int(0))
    );
    assert_eq!(ilt(&(1 / &s).atan(), &s, &t), t.sin() / &t);
    assert_eq!(ilt(&(&a / &s).atan(), &s, &t), (&a * &t).sin() / &t);
    assert_eq!(ilt(&(1 / (&s * (&s + 1).sqrt())), &s, &t), t.sqrt().erf());
    assert_eq!(ilt(&(-(ctx.euler_gamma() + s.ln()) / &s), &s, &t), t.ln());
    // Symbolic parameters in rational forms.
    assert_eq!(ilt(&(1 / (&s - &a)), &s, &t), (&a * &t).exp());
    assert_eq!(
        ilt(&(1 / (s.powi(2) + a.powi(2))), &s, &t),
        (&a * &t).sin() / &a
    );
    assert_eq!(
        ilt(&(&s / (s.powi(2) + a.powi(2))), &s, &t),
        (&a * &t).cos()
    );
    assert_eq!(ilt(&(1 / (&s + &a).powi(2)), &s, &t), &t * (-&a * &t).exp());
    // Still out of reach → unevaluated node, never a wrong answer.
    assert!(
        (1 / (s.powi(2) + 2 * &s + 5).powi(2))
            .inverse_laplace(&s, &t)
            .has_unevaluated()
    );
    assert!(s.ln().inverse_laplace(&s, &t).has_unevaluated());
}

// ═══════════════════════════════════════════════════════════════════════════
// Initial / final value theorems
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn initial_and_final_value_theorems() {
    let ctx = Context::new();
    let s = ctx.symbol("s");
    let t = ctx.symbol("t");

    // f = e^{−t} cos 2t: f(0⁺) = 1
    let f = (&s + 1) / ((&s + 1).powi(2) + 4);
    assert_eq!(format!("{}", f.laplace_initial_value(&s).unwrap()), "1");
    // f = sin t: f(0⁺) = 0
    assert_eq!(
        format!(
            "{}",
            (1 / (s.powi(2) + 1)).laplace_initial_value(&s).unwrap()
        ),
        "0"
    );
    // f = 3 − 2e^{−t} (step response): f(0⁺) = 1, f(∞) = 3
    let big_f = 3 / &s - 2 / (&s + 1);
    assert_eq!(format!("{}", big_f.laplace_initial_value(&s).unwrap()), "1");
    assert_eq!(format!("{}", big_f.laplace_final_value(&s).unwrap()), "3");
    // Consistency with the time domain.
    let f_t = ilt(&big_f, &s, &t);
    assert_eq!(format!("{}", f_t.limit(&t, &ctx.infinity())), "3");
    // Stable second-order step response: 5/(s(s² + 2s + 5)) → 1
    let g = 5 / (&s * (s.powi(2) + 2 * &s + 5));
    assert_eq!(format!("{}", g.laplace_final_value(&s).unwrap()), "1");
    // Delay is fine: e^{−2s}/s is a delayed step → 1
    assert_eq!(
        format!(
            "{}",
            ((-2 * &s).exp() / &s).laplace_final_value(&s).unwrap()
        ),
        "1"
    );

    // Poles on/right of the imaginary axis: the theorem does not apply.
    assert!(matches!(
        (1 / (&s - 1)).laplace_final_value(&s),
        Err(SymplexError::Divergent { .. })
    ));
    assert!(matches!(
        (1 / (s.powi(2) + 1)).laplace_final_value(&s),
        Err(SymplexError::Divergent { .. })
    ));
    assert!(matches!(
        (1 / s.powi(2)).laplace_final_value(&s),
        Err(SymplexError::Divergent { .. })
    ));
    // f = 1/√t: F = √(π/s), f(0⁺) = ∞ is reported as the extended-real value `oo`.
    let h = (ctx.pi() / &s).sqrt();
    assert_eq!(format!("{}", h.laplace_initial_value(&s).unwrap()), "oo");
}
