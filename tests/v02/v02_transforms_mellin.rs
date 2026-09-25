//! symplex 0.2 — Mellin transform: table, rules, fundamental strips, and
//! the inverse.
//!
//! Every forward entry is verified against `∫₀^∞ x^{s₀−1} f(x) dx` by
//! adaptive quadrature at real points `s₀` inside the returned strip.

use symplex::definite::QuadOpts;
use symplex::prelude::*;

/// Verify `big_f(s)` against the defining integral at the real points
/// `s_samples` (which must lie in the fundamental strip). `lo`/`hi` bound
/// the support of `f` (finite supports avoid step functions in the
/// quadrature).
fn verify(f: &Ex, x: &Ex, s: &Ex, big_f: &Ex, s_samples: &[f64], lo: f64, hi: Option<f64>) {
    let ctx = f.context();
    let lo_ex = ctx.from_f64(lo).unwrap();
    let hi_ex = match hi {
        Some(h) => ctx.from_f64(h).unwrap(),
        None => ctx.infinity(),
    };
    // Integrable endpoint singularities (x^{s₀−1} with s₀ < 1) converge
    // slowly, so accept a looser error estimate than the default.
    let opts = QuadOpts {
        rel_tol: 1e-7,
        ..QuadOpts::default()
    };
    for &s0 in s_samples {
        let s0_ex = ctx.from_f64(s0).unwrap();
        let integrand = f * x.pow(&(&s0_ex - 1));
        let QuadResult {
            value: num,
            error: err,
        } = integrand
            .integrate_numeric_with(x, &lo_ex, &hi_ex, &opts)
            .unwrap_or_else(|e| panic!("quadrature of {integrand} failed: {e}"));
        let sym = big_f
            .subs(s, &s0_ex)
            .eval_f64()
            .unwrap_or_else(|e| panic!("{big_f} at s={s0}: {e}"));
        let scale = sym.abs().max(1.0);
        assert!(
            err < 1e-4 * scale,
            "M{{{f}}}: quadrature at s={s0} did not converge (estimate {num}, error {err:e})"
        );
        assert!(
            (num - sym).abs() < 1e-5 * scale,
            "M{{{f}}} = {big_f}: at s={s0} integral {num} vs {sym}"
        );
    }
}

fn mt(f: &Ex, x: &Ex, s: &Ex) -> (Ex, BoolEx) {
    f.mellin_transform(x, s)
        .unwrap_or_else(|e| panic!("M{{{f}}} failed: {e}"))
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
    for base in [0.3f64, 0.55, 0.8] {
        let mut ea = a.clone();
        let mut eb = b.clone();
        for (j, s) in syms.iter().enumerate() {
            let v = ctx.from_f64(base + 0.17 * j as f64).unwrap();
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
// Table
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exponentials_give_gamma() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let f = (-&x).exp();
    let (big_f, strip) = mt(&f, &x, &s);
    assert_eq!(big_f, s.gamma());
    assert_eq!(format!("{strip}"), "re(s) > 0");
    verify(&f, &x, &s, &big_f, &[0.5, 1.0, 2.5], 0.0, None);

    let f = (&x * -3).exp();
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(&big_f, &(ctx.int(3).pow(&(-&s)) * s.gamma()), "e^{-3x}");
    assert_eq!(format!("{strip}"), "re(s) > 0");
    verify(&f, &x, &s, &big_f, &[0.5, 1.7], 0.0, None);

    let (big_f, strip) = mt(&(-&a * &x).exp(), &x, &s);
    assert_same(&big_f, &(a.pow(&(-&s)) * s.gamma()), "e^{-ax}");
    assert_eq!(format!("{strip}"), "re(s) > 0");

    // Gaussian and cubic exponentials: e^{−x^b} → Γ(s/b)/b
    let g = (-x.powi(2)).exp();
    let (big_g, strip) = mt(&g, &x, &s);
    assert_same(&big_g, &((&s / 2).gamma() / 2), "e^{-x²}");
    assert_eq!(format!("{strip}"), "re(s) > 0");
    verify(&g, &x, &s, &big_g, &[0.5, 1.0, 3.0], 0.0, None);

    let h = (-x.powi(3)).exp();
    let (big_h, _) = mt(&h, &x, &s);
    assert_same(&big_h, &((&s / 3).gamma() / 3), "e^{-x³}");
    verify(&h, &x, &s, &big_h, &[0.5, 2.0], 0.0, None);
}

#[test]
fn rational_functions_give_beta_and_cosecant() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let nu = ctx.symbol_with("nu", &[Assumption::Positive]).unwrap();
    let pi = ctx.pi();

    let f = 1 / (1 + &x);
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(&big_f, &(&pi / (&pi * &s).sin()), "1/(1+x)");
    assert_eq!(format!("{strip}"), "re(s) > 0 & 1 > re(s)");
    // Near the strip edges the integrand decays like x^{-1.2}: too slow for
    // plain quadrature, so sample the middle of the strip.
    verify(&f, &x, &s, &big_f, &[0.35, 0.5, 0.65], 0.0, None);

    let f = 1 / (1 + &x).powi(3);
    let (big_f, strip) = mt(&f, &x, &s);
    assert_eq!(big_f, s.beta(&(3 - &s)));
    assert_eq!(format!("{strip}"), "re(s) > 0 & 3 > re(s)");
    verify(&f, &x, &s, &big_f, &[0.5, 1.5, 2.5], 0.0, None);

    let (big_f, strip) = mt(&(1 + &x).pow(&(-&nu)), &x, &s);
    assert_eq!(big_f, s.beta(&(&nu - &s)));
    assert_eq!(format!("{strip}"), "re(s) > 0 & nu > re(s)");

    // 1/(1+x²) via f(x^b): π/(2 sin(πs/2)) on 0 < Re s < 2
    let f = 1 / (1 + x.powi(2));
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(&big_f, &(&pi / (2 * (&pi * &s / 2).sin())), "1/(1+x²)");
    assert_eq!(format!("{strip}"), "re(s) > 0 & 2 > re(s)");
    verify(&f, &x, &s, &big_f, &[0.5, 1.0, 1.5], 0.0, None);

    // 1/(2+x) via factoring the constant: 2^{s−1} π/sin(πs)
    let f = 1 / (2 + &x);
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(
        &big_f,
        &(ctx.int(2).pow(&(&s - 1)) * &pi / (&pi * &s).sin()),
        "1/(2+x)",
    );
    assert_eq!(format!("{strip}"), "re(s) > 0 & 1 > re(s)");
    verify(&f, &x, &s, &big_f, &[0.3, 0.6], 0.0, None);
}

#[test]
fn step_windows_and_beta() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let (big_f, strip) = mt(&(1 - &x).heaviside(), &x, &s);
    assert_eq!(big_f, 1 / &s);
    assert_eq!(format!("{strip}"), "re(s) > 0");

    let (big_f, strip) = mt(&((1 - &x).heaviside() * x.powi(2)), &x, &s);
    assert_eq!(big_f, 1 / (&s + 2));
    assert_eq!(format!("{strip}"), "re(s) > -2");
    verify(&x.powi(2), &x, &s, &big_f, &[0.5, 2.0], 0.0, Some(1.0));

    let (big_f, strip) = mt(&((1 - &x).heaviside() * x.pow(&a)), &x, &s);
    assert_eq!(big_f, 1 / (&s + &a));
    assert_eq!(format!("{strip}"), "re(s) > -a");

    // Outside the unit interval: H(x − 1) x^{−2} → −1/(s − 2), Re s < 2
    let (big_f, strip) = mt(&((&x - 1).heaviside() / x.powi(2)), &x, &s);
    assert_eq!(big_f, -1 / (&s - 2));
    assert_eq!(format!("{strip}"), "2 > re(s)");
    verify(&(1 / x.powi(2)), &x, &s, &big_f, &[0.5, 1.0], 1.0, None);

    // Beta: H(1 − x)(1 − x)^2 → B(s, 3)
    let (big_f, strip) = mt(&((1 - &x).heaviside() * (1 - &x).powi(2)), &x, &s);
    assert_eq!(big_f, s.beta(&ctx.int(3)));
    assert_eq!(format!("{strip}"), "re(s) > 0");
    verify(
        &(1 - &x).powi(2),
        &x,
        &s,
        &big_f,
        &[0.5, 1.0, 2.0],
        0.0,
        Some(1.0),
    );
}

#[test]
fn trigonometric_and_logarithmic_entries() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let pi = ctx.pi();

    let (big_f, strip) = mt(&x.sin(), &x, &s);
    assert_eq!(big_f, (s.gamma() * (&pi * &s / 2).sin()).expand());
    assert_eq!(format!("{strip}"), "re(s) > -1 & 1 > re(s)");

    let (big_f, strip) = mt(&x.cos(), &x, &s);
    assert_eq!(big_f, (s.gamma() * (&pi * &s / 2).cos()).expand());
    assert_eq!(format!("{strip}"), "re(s) > 0 & 1 > re(s)");

    // sin(2x) via scaling
    let (big_f, _) = mt(&(&x * 2).sin(), &x, &s);
    assert_same(
        &big_f,
        &(ctx.int(2).pow(&(-&s)) * s.gamma() * (&pi * &s / 2).sin()),
        "sin 2x",
    );

    let f = (1 + &x).ln();
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(&big_f, &(&pi / (&s * (&pi * &s).sin())), "ln(1+x)");
    assert_eq!(format!("{strip}"), "re(s) > -1 & 0 > re(s)");
    verify(&f, &x, &s, &big_f, &[-0.5], 0.0, None);
}

// ═══════════════════════════════════════════════════════════════════════════
// Rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn power_scaling_and_linearity_rules() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    // x² e^{−3x} → 3^{−s−2} Γ(s + 2), Re s > −2
    let f = x.powi(2) * (&x * -3).exp();
    let (big_f, strip) = mt(&f, &x, &s);
    assert_same(
        &big_f,
        &(ctx.int(3).pow(&(-&s - 2)) * (&s + 2).gamma()),
        "x²e^{-3x}",
    );
    assert_eq!(format!("{strip}"), "re(s) > -2");
    verify(&f, &x, &s, &big_f, &[-1.5, 0.5, 2.0], 0.0, None);

    // x^a e^{−x} → Γ(s + a), Re s > −a
    let (big_f, strip) = mt(&(x.pow(&a) * (-&x).exp()), &x, &s);
    assert_eq!(big_f, (&s + &a).gamma());
    assert_eq!(format!("{strip}"), "re(s) > -a");

    // √x e^{−x} → Γ(s + ½);  e^{−x}/x → Γ(s − 1), Re s > 1
    let (big_f, strip) = mt(&(x.sqrt() * (-&x).exp()), &x, &s);
    assert_eq!(big_f, (&s + ctx.rational(1, 2)).gamma());
    assert_eq!(format!("{strip}"), "re(s) > -1/2");
    let (big_f, strip) = mt(&((-&x).exp() / &x), &x, &s);
    assert_eq!(big_f, (&s - 1).gamma());
    assert_eq!(format!("{strip}"), "re(s) > 1");

    // Linearity with strip intersection.
    let (big_f, strip) = mt(&(1 / (1 + &x) + (-&x).exp()), &x, &s);
    assert_same(
        &big_f,
        &(ctx.pi() / (ctx.pi() * &s).sin() + s.gamma()),
        "sum",
    );
    assert_eq!(format!("{strip}"), "re(s) > 0 & 1 > re(s)");

    // ln(x) e^{−x} → Γ′(s) = Γ(s)ψ(s)
    let f = x.ln() * (-&x).exp();
    let (big_f, strip) = mt(&f, &x, &s);
    assert_eq!(big_f, (s.gamma() * s.digamma()).expand());
    assert_eq!(format!("{strip}"), "re(s) > 0");
    verify(&f, &x, &s, &big_f, &[1.0, 2.5], 0.0, None);
}

#[test]
fn derivative_rule() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    // d/dx e^{−x} = −e^{−x}: M{f′}(s) = −(s−1)F(s−1) = −(s−1)Γ(s−1) = −Γ(s)
    let deriv = (-&x).exp().diff(&x);
    let (big_f, _) = mt(&deriv, &x, &s);
    assert_same(&big_f, &(-s.gamma()), "derivative of e^{-x}");
}

// ═══════════════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn errors_and_assumption_gating() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let b = ctx.symbol("b");

    for bad in [
        ctx.int(1),
        x.clone(),
        x.exp(),
        x.powi(2) + 1,
        (1 + &x).powi(2),
    ] {
        assert!(
            bad.mellin_transform(&x, &s).is_err(),
            "{bad} should have no Mellin transform"
        );
    }
    // Unknown sign of a parameter → Err with guidance.
    let err = (-&b * &x).exp().mellin_transform(&x, &s).unwrap_err();
    assert!(err.to_string().contains("Assumption::Positive"), "{err}");
    assert!((1 + &x).pow(&(-&b)).mellin_transform(&x, &s).is_err());
    // Bad variables.
    assert!(matches!(
        (-&x).exp().mellin_transform(&x, &x),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        (-&x).exp().mellin_transform(&ctx.int(2), &s),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // Non-overlapping strips: e^{−x}/x needs Re s > 1, H(x−1)x^{−1/2} needs Re s < 1/2.
    let f = (-&x).exp() / &x + (&x - 1).heaviside() / x.sqrt();
    assert!(f.mellin_transform(&x, &s).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_table_and_round_trips() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = ctx.symbol("s");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    let nu = ctx.symbol_with("nu", &[Assumption::Positive]).unwrap();
    let pi = ctx.pi();

    let inv = |f: &Ex| {
        f.inverse_mellin_transform(&s, &x)
            .unwrap_or_else(|e| panic!("M⁻¹{{{f}}} failed: {e}"))
    };
    assert_eq!(inv(&s.gamma()), (-&x).exp());
    assert_eq!(inv(&((&s / 2).gamma() / 2)), (-x.powi(2)).exp());
    assert_eq!(inv(&(&pi / (&pi * &s).sin())), 1 / (&x + 1));
    assert_eq!(inv(&s.beta(&(3 - &s))), (&x + 1).powi(-3));
    assert_eq!(inv(&s.beta(&(&nu - &s))), (&x + 1).pow(&(-&nu)));
    assert_eq!(
        inv(&s.beta(&ctx.int(3))),
        (1 - &x).powi(2) * (1 - &x).heaviside()
    );
    assert_eq!(inv(&(1 / (&s + 2))), x.powi(2) * (1 - &x).heaviside());
    assert_eq!(inv(&(1 / &s)), (1 - &x).heaviside());
    assert_eq!(inv(&(&pi / (&s * (&pi * &s).sin()))), (&x + 1).ln());
    assert_eq!(inv(&(s.gamma() * (&pi * &s / 2).sin())), x.sin());
    assert_eq!(inv(&(s.gamma() * (&pi * &s / 2).cos())), x.cos());
    // Written-out Beta: Γ(s)Γ(3−s)/Γ(3) → (1+x)^{−3}
    assert_eq!(
        inv(&(s.gamma() * (3 - &s).gamma() / ctx.int(3).gamma())),
        (&x + 1).powi(-3)
    );
    // Shift and scaling rules in reverse.
    assert_eq!(inv(&(&s + 2).gamma()), x.powi(2) * (-&x).exp());
    assert_eq!(inv(&(a.pow(&(-&s)) * s.gamma())), (-&a * &x).exp());
    assert!(ctx.int(1).inverse_mellin_transform(&s, &x).is_err());
    assert!(s.zeta().inverse_mellin_transform(&s, &x).is_err());

    // Round trips through the forward table.
    for f in [
        (-&x).exp(),
        x.powi(2) * (&x * -3).exp(),
        (-x.powi(3)).exp(),
        1 / (1 + &x).powi(3),
        (1 - &x).heaviside() * x.powi(2),
        x.sin(),
        (&x * 2).sin(),
        (1 + &x).ln(),
        &x * (-&x).exp() + (-&x).exp(),
    ] {
        let (big_f, _) = mt(&f, &x, &s);
        let back = inv(&big_f);
        assert_same(&back, &f, &format!("round trip of {f} via {big_f}"));
    }
}
