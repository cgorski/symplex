//! symplex 0.2 — public Fourier transform API: table entries, rules,
//! conventions, assumption gating, and inverse round trips.
//!
//! Absolutely integrable entries are verified against the defining integral
//! `F(ω₀) = ∫ f(t) cos(ω₀t) dt − i ∫ f(t) sin(ω₀t) dt` by adaptive
//! quadrature at a few sample frequencies.

use symplex::fourier_transform::FourierConvention;
use symplex::prelude::*;

/// Verify `big_f(ω)` against the defining integral of `f(t)` at a few
/// frequencies (non-unitary angular convention).
fn verify_forward(f: &Ex, t: &Ex, w: &Ex, big_f: &Ex) {
    let ctx = f.context();
    verify_forward_on(f, t, w, big_f, &ctx.neg_infinity(), &ctx.infinity());
}

/// Like [`verify_forward`] but integrating `f` over `[lo, hi]` only (for
/// one-sided functions whose step factor the quadrature cannot handle on
/// an infinite interval; `f` is given *without* the step).
fn verify_forward_on(f: &Ex, t: &Ex, w: &Ex, big_f: &Ex, lo: &Ex, hi: &Ex) {
    let ctx = f.context();
    for w0 in [0.0, 0.7, 1.9, -1.3] {
        let w0_ex = ctx.from_f64(w0).unwrap();
        let re = (f * (&w0_ex * t).cos())
            .integrate_numeric(t, lo, hi)
            .unwrap_or_else(|e| panic!("quadrature of {f} failed: {e}"));
        let im = -(f * (&w0_ex * t).sin())
            .integrate_numeric(t, lo, hi)
            .unwrap_or_else(|e| panic!("quadrature of {f} failed: {e}"));
        let Complex64 { re: gre, im: gim } = big_f
            .subs(w, &w0_ex)
            .eval_complex64()
            .unwrap_or_else(|e| panic!("{big_f} at {w0}: {e}"));
        let tol = 1e-6 * (re.abs() + im.abs()).max(1.0);
        assert!(
            (gre - re).abs() < tol && (gim - im).abs() < tol,
            "F{{{f}}} = {big_f}: at ω={w0} got ({gre}, {gim}), integral gives ({re}, {im})"
        );
    }
}

fn ft(f: &Ex, t: &Ex, w: &Ex) -> Ex {
    f.fourier_transform(t, w)
        .unwrap_or_else(|e| panic!("F{{{f}}} failed: {e}"))
}

fn ift(big_f: &Ex, w: &Ex, t: &Ex) -> Ex {
    big_f
        .inverse_fourier_transform(w, t)
        .unwrap_or_else(|e| panic!("F⁻¹{{{big_f}}} failed: {e}"))
}

/// Symbolic equality after simplification, with a numeric fallback: both
/// sides are evaluated (complex) at a few generic points of every free
/// symbol they contain. Distributions (`δ`) would vanish at generic points,
/// so callers compare those by display instead.
fn assert_same(a: &Ex, b: &Ex, label: &str) {
    if (a - b).simplify().is_zero_structural() {
        return;
    }
    let ctx = a.context();
    let mut syms = a.free_symbols();
    for s in b.free_symbols() {
        if !syms.contains(&s) {
            syms.push(s);
        }
    }
    for (k, base) in [0.7f64, 1.9, -1.3].into_iter().enumerate() {
        let mut ea = a.clone();
        let mut eb = b.clone();
        for (j, s) in syms.iter().enumerate() {
            let v = ctx
                .from_f64(base + 0.37 * j as f64 + 0.11 * k as f64)
                .unwrap();
            ea = ea.subs(s, &v);
            eb = eb.subs(s, &v);
        }
        let Complex64 { re: ar, im: ai } = ea
            .eval_complex64()
            .unwrap_or_else(|e| panic!("{label}: cannot evaluate {a}: {e}"));
        let Complex64 { re: br, im: bi } = eb
            .eval_complex64()
            .unwrap_or_else(|e| panic!("{label}: cannot evaluate {b}: {e}"));
        let tol = 1e-9 * (br.abs() + bi.abs()).max(1.0);
        assert!(
            (ar - br).abs() < tol && (ai - bi).abs() < tol,
            "{label}: {a} ≠ {b} (({ar}, {ai}) vs ({br}, {bi}))"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Table entries — verified against the integral
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn two_sided_exponential_and_gaussian() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let f = (-t.abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_eq!(format!("{big_f}"), "2/(w^2 + 1)");
    verify_forward(&f, &t, &w, &big_f);

    let f = (-&a * t.abs()).exp();
    assert_eq!(format!("{}", ft(&f, &t, &w)), "2*a/(a^2 + w^2)");

    let g = (-t.powi(2)).exp();
    let big_g = ft(&g, &t, &w);
    assert_eq!(format!("{big_g}"), "sqrt(pi)*exp(-1/4*w^2)");
    verify_forward(&g, &t, &w, &big_g);

    let g3 = (-3 * t.powi(2)).exp();
    verify_forward(&g3, &t, &w, &ft(&g3, &t, &w));
}

#[test]
fn causal_exponentials_with_powers_of_t() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let i = ctx.i_unit();
    let zero = ctx.int(0);
    let inf = ctx.infinity();
    let g = (&t * -2).exp();
    let f = &g * t.heaviside();
    let big_f = ft(&f, &t, &w);
    assert_same(&big_f, &(1 / (&i * &w + 2)), "e^{-2t}H(t)");
    verify_forward_on(&g, &t, &w, &big_f, &zero, &inf);

    let g = &t * (&t * -2).exp();
    let f = &g * t.heaviside();
    let big_f = ft(&f, &t, &w);
    assert_same(&big_f, &(1 / (&i * &w + 2).powi(2)), "t e^{-2t}H(t)");
    verify_forward_on(&g, &t, &w, &big_f, &zero, &inf);

    let g = t.powi(2) * (&t * -3).exp();
    let f = &g * t.heaviside();
    let big_f = ft(&f, &t, &w);
    assert_same(&big_f, &(2 / (&i * &w + 3).powi(3)), "t² e^{-3t}H(t)");
    verify_forward_on(&g, &t, &w, &big_f, &zero, &inf);

    let f = (-&a * &t).exp() * t.heaviside();
    assert_same(&ft(&f, &t, &w), &(1 / (&i * &w + &a)), "e^{-at}H(t)");

    // Anti-causal: e^{2t} H(−t) → 1/(2 − iω)
    let g = (&t * 2).exp();
    let f = &g * (-&t).heaviside();
    let big_f = ft(&f, &t, &w);
    assert_same(&big_f, &(1 / (2 - &i * &w)), "e^{2t}H(-t)");
    verify_forward_on(&g, &t, &w, &big_f, &ctx.neg_infinity(), &zero);
}

#[test]
fn rectangular_windows_and_sinc() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let rect = (&t + 1).heaviside() - (&t - 1).heaviside();
    let big_r = ft(&rect, &t, &w);
    assert_eq!(format!("{big_r}"), "2*sin(w)/w");
    // Numeric check at ω ≠ 0 (finite window: plain quadrature on [−1, 1]).
    for w0 in [0.5f64, 2.0] {
        let expected = 2.0 * w0.sin() / w0;
        let got = big_r
            .subs(&w, &ctx.from_f64(w0).unwrap())
            .eval_f64()
            .unwrap();
        assert!((got - expected).abs() < 1e-12);
    }

    // Window as a product of steps and with a symbolic half-width.
    let prod = (&t + 1).heaviside() * (1 - &t).heaviside();
    assert_eq!(format!("{}", ft(&prod, &t, &w)), "2*sin(w)/w");
    let recta = (&t + &a).heaviside() - (&t - &a).heaviside();
    assert_eq!(format!("{}", ft(&recta, &t, &w)), "2*sin(a*w)/w");

    // sinc ↔ rect
    let sinc = t.sin() / &t;
    let big_s = ft(&sinc, &t, &w);
    let rect_w = (ctx.pi() * ((&w + 1).heaviside() - (&w - 1).heaviside())).expand();
    assert_eq!(big_s, rect_w, "{big_s}");
    let sinc2 = (&t * 2).sin() / &t;
    let rect2 = (ctx.pi() * ((&w + 2).heaviside() - (&w - 2).heaviside())).expand();
    assert_eq!(ft(&sinc2, &t, &w), rect2);
}

#[test]
fn distributions_delta_constant_step_sign() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");

    let i = ctx.i_unit();
    let pi = ctx.pi();
    assert_eq!(ft(&t.dirac_delta(), &t, &w), ctx.int(1));
    assert_eq!(ft(&(&t - 2).dirac_delta(), &t, &w), (-2 * &i * &w).exp());
    assert_eq!(ft(&ctx.int(1), &t, &w), 2 * &pi * w.dirac_delta());
    assert_eq!(ft(&ctx.int(5), &t, &w), 10 * &pi * w.dirac_delta());
    assert_eq!(
        ft(&t.heaviside(), &t, &w),
        &pi * w.dirac_delta() + 1 / (&i * &w)
    );
    assert_eq!(
        ft(&(-&t).heaviside(), &t, &w),
        &pi * w.dirac_delta() - 1 / (&i * &w)
    );
    assert_eq!(ft(&t.sign(), &t, &w), 2 / (&i * &w));
    assert_eq!(ft(&(1 / &t), &t, &w), -&i * &pi * w.sign());
    assert_eq!(ft(&t.abs(), &t, &w), -2 / w.powi(2));
}

#[test]
fn sinusoids_become_delta_pairs() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let pi = ctx.pi();
    let i = ctx.i_unit();
    let c = ft(&(&t * 3).cos(), &t, &w);
    assert_eq!(
        c,
        &pi * (&w - 3).dirac_delta() + &pi * (&w + 3).dirac_delta()
    );
    let s = ft(&(&t * 3).sin(), &t, &w);
    assert_eq!(
        s,
        (&i * &pi * ((&w + 3).dirac_delta() - (&w - 3).dirac_delta())).expand()
    );
    let ca = ft(&(&a * &t).cos(), &t, &w);
    assert_eq!(
        ca,
        &pi * (&w - &a).dirac_delta() + &pi * (&w + &a).dirac_delta()
    );
    // e^{iω₀t} → 2π δ(ω − ω₀)
    let e = ft(&(&i * 5 * &t).exp(), &t, &w);
    assert_eq!(e, 2 * &pi * (&w - 5).dirac_delta());
}

// ═══════════════════════════════════════════════════════════════════════════
// Rules
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn linearity_shift_and_scaling() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");

    let i = ctx.i_unit();
    let lin = ft(&(3 * t.dirac_delta() + 2 * t.heaviside()), &t, &w);
    assert_eq!(lin, 3 + 2 * ctx.pi() * w.dirac_delta() + 2 / (&i * &w));

    // Shift: e^{−|t−1|} → e^{−iω}·2/(1+ω²)
    let f = (-(&t - 1).abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_same(&big_f, &(2 * (-&i * &w).exp() / (w.powi(2) + 1)), "shift");
    verify_forward(&f, &t, &w, &big_f);

    // Delayed causal exponential: H(t−2) e^{−t}
    let f = (&t - 2).heaviside() * (-&t).exp();
    let big_f = ft(&f, &t, &w);
    assert_same(
        &big_f,
        &((-2 * &i * &w - 2).exp() / (&i * &w + 1)),
        "delayed causal exponential",
    );
    verify_forward(&f, &t, &w, &big_f);

    // Scaling: e^{−|3t|} → 6/(9 + ω²)
    let f = (-(&t * 3).abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_eq!(format!("{big_f}"), "6/(w^2 + 9)");
    verify_forward(&f, &t, &w, &big_f);
}

#[test]
fn modulation_and_multiplication_by_t() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let i = ctx.i_unit();

    // e^{5it} e^{−|t|} → 2/(1 + (ω−5)²)
    let f = (&i * 5 * &t).exp() * (-t.abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_eq!(format!("{big_f}"), "2/(w^2 - 10*w + 26)");

    // cos(5t) e^{−|t|} → ½[G(ω−5) + G(ω+5)]
    let f = (&t * 5).cos() * (-t.abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_eq!(
        format!("{big_f}"),
        "1/(w^2 - 10*w + 26) + 1/(w^2 + 10*w + 26)"
    );
    verify_forward(&f, &t, &w, &big_f);

    // Damped oscillation e^{−t} sin(t) H(t)
    let f = (-&t).exp() * t.sin() * t.heaviside();
    let big_f = ft(&f, &t, &w);
    verify_forward(&f, &t, &w, &big_f);

    // t·f(t) → i F′(ω)
    let f = &t * (-t.abs()).exp();
    let big_f = ft(&f, &t, &w);
    assert_same(
        &big_f,
        &(-4 * &i * &w / (w.powi(2) + 1).powi(2)),
        "t e^{-|t|}",
    );
    verify_forward(&f, &t, &w, &big_f);

    let f = &t * (-t.powi(2)).exp();
    let big_f = ft(&f, &t, &w);
    assert_same(
        &big_f,
        &(-&i * ctx.pi().sqrt() * &w * (-w.powi(2) / 4).exp() / 2),
        "t e^{-t²}",
    );
    verify_forward(&f, &t, &w, &big_f);

    let f = t.powi(2) * (-t.powi(2)).exp();
    verify_forward(&f, &t, &w, &ft(&f, &t, &w));
}

#[test]
fn piecewise_inputs() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let otherwise = ctx.int(1).gt(&ctx.int(0));

    // Rectangular pulse written piecewise.
    let rect = Ex::piecewise(&[
        (&ctx.int(1), &t.abs().lt(&ctx.int(1))),
        (&ctx.int(0), &otherwise),
    ]);
    assert_eq!(format!("{}", ft(&rect, &t, &w)), "2*sin(w)/w");

    // Two-sided exponential written piecewise (sequential semantics).
    let f = Ex::piecewise(&[(&t.exp(), &t.lt(&ctx.int(0))), (&(-&t).exp(), &otherwise)]);
    let big_f = ft(&f, &t, &w);
    let expected = 2 / (w.powi(2) + 1);
    assert_same(&big_f, &expected, "piecewise e^{-|t|}");

    // Causal exponential written piecewise.
    let g = Ex::piecewise(&[
        (&(-&t).exp(), &t.gt(&ctx.int(0))),
        (&ctx.int(0), &otherwise),
    ]);
    assert_same(
        &ft(&g, &t, &w),
        &(1 / (ctx.i_unit() * &w + 1)),
        "piecewise causal",
    );

    // Window on [0, 2]: (1 − e^{−2iω})/(iω)
    let h = Ex::piecewise(&[
        (&ctx.int(1), &ctx.int(0).le(&t).and(&t.le(&ctx.int(2)))),
        (&ctx.int(0), &otherwise),
    ]);
    let big_h = ft(&h, &t, &w);
    for w0 in [0.5f64, 1.5] {
        let Complex64 { re, im } = big_h
            .subs(&w, &ctx.from_f64(w0).unwrap())
            .eval_complex64()
            .unwrap();
        // ∫₀² e^{−iω₀t} dt = sin(2ω₀)/ω₀ − i(1 − cos(2ω₀))/ω₀
        assert!((re - (2.0 * w0).sin() / w0).abs() < 1e-12);
        assert!((im + (1.0 - (2.0 * w0).cos()) / w0).abs() < 1e-12);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Errors and assumption gating
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn symbolic_parameters_are_gated_on_assumptions() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let b = ctx.symbol("b");
    let n = ctx.symbol_with("n", &[Assumption::Negative]).unwrap();

    // Unknown sign → Err with a helpful message, never a guess.
    let err = (-&b * t.abs()).exp().fourier_transform(&t, &w).unwrap_err();
    assert!(
        matches!(err, SymplexError::ComputationFailed { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("Assumption::Positive"), "{err}");
    assert!((-&b * t.powi(2)).exp().fourier_transform(&t, &w).is_err());
    assert!(
        ((&b * &t).exp() * t.heaviside())
            .fourier_transform(&t, &w)
            .is_err()
    );

    // Provably wrong sign → Err (not transformable), too.
    assert!((-&n * t.abs()).exp().fourier_transform(&t, &w).is_err());
    // Negative parameter in the causal exponential is fine: e^{nt} H(t), n < 0.
    let f = ((&n * &t).exp() * t.heaviside())
        .fourier_transform(&t, &w)
        .unwrap();
    assert_same(&f, &(1 / (ctx.i_unit() * &w - &n)), "e^{nt}H(t), n < 0");
}

#[test]
fn untransformable_inputs_are_errors_not_wrong_answers() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    // Growing exponentials.
    assert!((t.exp() * t.heaviside()).fourier_transform(&t, &w).is_err());
    assert!((-&t).exp().fourier_transform(&t, &w).is_err());
    // Needs δ′.
    assert!(t.fourier_transform(&t, &w).is_err());
    assert!((&t * t.heaviside()).fourier_transform(&t, &w).is_err());
    // Not in the table.
    assert!(t.powi(2).cos().fourier_transform(&t, &w).is_err());
    // Bad variables.
    assert!(matches!(
        t.exp().fourier_transform(&ctx.int(1), &w),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        t.exp().fourier_transform(&t, &t),
        Err(SymplexError::InvalidArgument { .. })
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse transform and round trips
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn inverse_table() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let i = ctx.i_unit();
    let a = ctx.symbol_with("a", &[Assumption::Positive]).unwrap();

    let pi = ctx.pi();
    assert_eq!(ift(&ctx.int(1), &w, &t), t.dirac_delta());
    assert_eq!(ift(&w.dirac_delta(), &w, &t), 1 / (2 * &pi));
    assert_eq!(
        ift(&(&w - 3).dirac_delta(), &w, &t),
        (3 * &i * &t).exp() / (2 * &pi)
    );
    assert_eq!(
        ift(&(1 / (&i * &w + 2)), &w, &t),
        (-2 * &t).exp() * t.heaviside()
    );
    assert_eq!(
        ift(&(1 / (&i * &w + &a)), &w, &t),
        (-&a * &t).exp() * t.heaviside()
    );
    assert_eq!(
        ift(&(1 / (&i * &w + 2).powi(2)), &w, &t),
        &t * (-2 * &t).exp() * t.heaviside()
    );
    assert_eq!(
        ift(&(1 / (&i * &w - 2)), &w, &t),
        -(2 * &t).exp() * (-&t).heaviside()
    );
    assert_eq!(ift(&(1 / (&i * &w)), &w, &t), t.sign() / 2);
    assert_eq!(ift(&(2 / (w.powi(2) + 1)), &w, &t), (-t.abs()).exp());
    assert_eq!(
        ift(&(2 * &a / (w.powi(2) + a.powi(2))), &w, &t),
        (-&a * t.abs()).exp()
    );
    assert_eq!(
        ift(&(&w / (w.powi(2) + 4)), &w, &t),
        &i * (-2 * t.abs()).exp() * t.sign() / 2
    );
    assert_same(
        &ift(&(-w.powi(2)).exp(), &w, &t),
        &((-t.powi(2) / 4).exp() / (2 * pi.sqrt())),
        "inverse Gaussian",
    );
    assert_eq!(
        ift(&(w.sin() / &w), &w, &t),
        ((&t + 1).heaviside() - (&t - 1).heaviside()) / 2
    );
    assert_eq!(
        ift(
            &(&pi * ((&w + 1).heaviside() - (&w - 1).heaviside())),
            &w,
            &t
        ),
        t.sin() / &t
    );
    assert_eq!(ift(&w.sign(), &w, &t), &i / (&pi * &t));
    assert_eq!(ift(&(-2 * &i * &w).exp(), &w, &t), (&t - 2).dirac_delta());
    assert_eq!(
        ift(&w.cos(), &w, &t),
        ((&t - 1).dirac_delta() + (&t + 1).dirac_delta()) / 2
    );
    let cos2 = ift(
        &(&pi * (&w - 2).dirac_delta() + &pi * (&w + 2).dirac_delta()),
        &w,
        &t,
    );
    assert_eq!(cos2, (2 * &t).cos());
}

#[test]
fn inverse_rules_shift_and_derivative() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let i = ctx.i_unit();
    // e^{−2iω}/(iω + 1) → e^{−(t−2)} H(t − 2)
    let f = ift(&((-2 * &i * &w).exp() / (&i * &w + 1)), &w, &t);
    assert_eq!(f, (2 - &t).exp() * (&t - 2).heaviside());
    // ω/(iω + 1) → −i d/dt[e^{−t}H(t)] = i e^{−t}H(t) − i δ(t)
    let g = ift(&(&w / (&i * &w + 1)), &w, &t);
    assert_eq!(g, &i * (-&t).exp() * t.heaviside() - &i * t.dirac_delta());
    // Not in the table → Err.
    assert!(
        (1 / (w.powi(2) + 1).powi(2))
            .inverse_fourier_transform(&w, &t)
            .is_err()
    );
    assert!(w.powi(2).cos().inverse_fourier_transform(&w, &t).is_err());
}

#[test]
fn round_trips() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let fs = [
        (-t.abs()).exp(),
        (&t * -2).exp() * t.heaviside(),
        &t * (&t * -2).exp() * t.heaviside(),
        (-t.powi(2)).exp(),
        (&t + 1).heaviside() - (&t - 1).heaviside(),
        t.sin() / &t,
        (&t - 2).dirac_delta(),
        (&t * 2).cos(),
    ];
    for f in &fs {
        let big_f = ft(f, &t, &w);
        let back = ift(&big_f, &w, &t);
        assert_same(&back, f, &format!("round trip of {f} via {big_f}"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Conventions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn conventions_are_consistent() {
    let ctx = Context::new();
    let t = ctx.symbol("t");
    let w = ctx.symbol("w");
    let nu = ctx.symbol("nu");
    let f = (-t.abs()).exp();

    let fa = f.fourier_transform(&t, &w).unwrap();
    let fu = f
        .fourier_transform_with(&t, &w, FourierConvention::UnitaryAngular)
        .unwrap();
    let fo = f
        .fourier_transform_with(&t, &nu, FourierConvention::Ordinary)
        .unwrap();

    // F_u = F_a/√(2π); F_o(ν) = F_a(2πν).
    let w0 = ctx.from_f64(0.9).unwrap();
    let fa0 = fa.subs(&w, &w0).eval_f64().unwrap();
    let fu0 = fu.subs(&w, &w0).eval_f64().unwrap();
    assert!((fu0 - fa0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
    let nu0 = ctx.from_f64(0.9 / (2.0 * std::f64::consts::PI)).unwrap();
    let fo0 = fo.subs(&nu, &nu0).eval_f64().unwrap();
    assert!((fo0 - fa0).abs() < 1e-9, "{fo0} vs {fa0}");

    // Each convention inverts itself.
    for conv in [
        FourierConvention::NonUnitaryAngular,
        FourierConvention::UnitaryAngular,
        FourierConvention::Ordinary,
    ] {
        let big_f = f.fourier_transform_with(&t, &w, conv).unwrap();
        let back = big_f.inverse_fourier_transform_with(&w, &t, conv).unwrap();
        assert_same(&back, &f, &format!("{conv:?} round trip"));
    }

    // e^{−πt²} is its own transform in the ordinary convention; δ ↔ 1 in all.
    let g = (-(ctx.pi() * t.powi(2))).exp();
    let go = g
        .fourier_transform_with(&t, &nu, FourierConvention::Ordinary)
        .unwrap();
    assert_eq!(go, (-(ctx.pi() * nu.powi(2))).exp());
    for conv in [
        FourierConvention::NonUnitaryAngular,
        FourierConvention::UnitaryAngular,
        FourierConvention::Ordinary,
    ] {
        let d = t
            .dirac_delta()
            .fourier_transform_with(&t, &w, conv)
            .unwrap();
        let expected = match conv {
            FourierConvention::UnitaryAngular => 1.0 / (2.0 * std::f64::consts::PI).sqrt(),
            _ => 1.0,
        };
        assert!(
            (d.eval_f64().unwrap() - expected).abs() < 1e-12,
            "{conv:?}: {d}"
        );
    }
    // Constant 1 in the ordinary convention is δ(ν) (no 2π).
    let one_o = ctx
        .int(1)
        .fourier_transform_with(&t, &nu, FourierConvention::Ordinary)
        .unwrap();
    assert_eq!(format!("{one_o}"), "DiracDelta(nu)");
    assert_eq!(
        FourierConvention::default(),
        FourierConvention::NonUnitaryAngular
    );
}
