//! Tests of session 17, wave 17 (s17a_transforms): the helpers shared by the Laplace, Z, Fourier and Mellin transforms.
//! Every test says what was wrong before and cites its oracle.
//!
//! The subject is one rule (`src/calculus/transform_util.rs`): an occurrence
//! of the transform variable that a binder inside the expression has bound
//! — the index of a `Sum`, the variable of a definite `Integral`, of a
//! `Limit`, of a `RootOf` — is not an occurrence of the transform variable.
//! Up to 0.59 the Fourier and Mellin transforms counted it as one, and the
//! passes of all the transforms that rewrite the function being transformed
//! (`Piecewise` into steps, steps read on the positive axis) walked into
//! such binders.
//!
//! The oracle of record is mpmath (`quad` of the defining integral at 20
//! and 40 digits, the constant computed from its definition with
//! `H(0) = 1/2`); the values quoted are the 40-digit ones.

use symplex::fourier_transform::FourierConvention;
use symplex::prelude::*;

/// `a` and `b` agree to `1e-12` relative.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

/// The value of `e` at `var = num/den` (`m = 2` where a case has that
/// parameter).
fn at(ctx: &Context, e: &Ex, var: &Ex, num: i64, den: i64) -> f64 {
    e.subs(var, &ctx.rational(num, den))
        .subs(&ctx.symbol("m"), &ctx.int(2))
        .eval_f64()
        .unwrap_or_else(|err| panic!("{e} at {var} = {num}/{den}: {err}"))
}

/// Structural equality, or a difference that simplifies to zero.
fn same(a: &Ex, b: &Ex) -> bool {
    a == b || (a - b).simplify().is_zero_structural()
}

// ── Laplace ─────────────────────────────────────────────────────────────────

/// `L{K·e^{−2t}}` with `K = Σ_{t=1}^{3} [t < 2] = 1` is `K/(s + 2)`.
///
/// Before: `3/(2(s + 2))` — the `Piecewise` under the sum was rewritten to
/// `H(2 − t)`, which is `1/2` at the breakpoint the bound `t` runs through,
/// so the constant became `1 + 1/2 + 0`.
///
/// Oracle: mpmath `quad(lambda t: 1*exp(-2*t)*exp(-s*t), [0, inf])` =
/// 0.4, 0.333333333333333333, 0.2 at `s = 1/2, 1, 3` (returned before:
/// 0.6, 0.5, 0.3).
#[test]
fn laplace_leaves_a_piecewise_under_a_sum_over_the_variable() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let k = ctx
        .parse("Sum(Piecewise(1 if t < 2, 0 if True), t, 1, 3)")
        .unwrap();
    assert_eq!(k.eval(), ctx.int(1));
    let f = (&k * (&t * -2).exp()).laplace(&t, &s);
    assert!(same(&f, &(&k / (&s + 2))), "{f}");
    for ((num, den), want) in [((1, 2), 0.4), ((1, 1), 1.0 / 3.0), ((3, 1), 0.2)] {
        let got = at(&ctx, &f, &s, num, den);
        assert!(close(got, want), "s = {num}/{den}: {got} vs {want}");
    }
}

/// A step under a binder of `t` is not read on `t > 0`: `K = Σ_{t=−3}^{3}
/// H(t + 1) = 9/2`, `Π_{t=−3}^{0} (H(t + 3) + 1) = 12`, `∫_{−3}^{3} H(t + 1)
/// dt = 4`, `Σ_{t=−3}^{3} H(−t − 1) = 5/2`.
///
/// Before: every bound step with a threshold `≤ 0` was replaced by `1`
/// (rising) or `0` (falling) — `21/(s² + 9)`, `16·e^{−s}/s`, `12/s³`, `1/s`.
///
/// Oracle: mpmath `quad` of `f(t)·e^{−st}` over `[0, ∞)` at `s = 1`:
/// 1.35, 4.41455329405730786, 8.0, 0.285714285714285714 (returned before:
/// 2.1, 5.88607105874, 12, 1).
#[test]
fn laplace_does_not_read_a_bound_step_on_the_positive_axis() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let one = |e: &Ex| at(&ctx, e, &s, 1, 1);

    let k = ctx.parse("Sum(Heaviside(t + 1), t, -3, 3)").unwrap();
    let f = (&k * (&t * 3).sin()).laplace(&t, &s);
    assert!(same(&f, &(&k * 3 / (s.powi(2) + 9))), "{f}");
    assert!(close(one(&f), 1.35), "{f}");

    let k = ctx
        .parse("Product(Heaviside(t + 3) + 1, t, -3, 0)")
        .unwrap();
    let f = (&k * (&t - 1).heaviside()).laplace(&t, &s);
    assert!(close(one(&f), 4.414_553_294_057_308), "{f}");

    let k = ctx.parse("Integral(Heaviside(t + 1), t, -3, 3)").unwrap();
    let f = (&k * t.powi(2)).laplace(&t, &s);
    assert!(close(one(&f), 8.0), "{f}");

    let k = ctx.parse("Sum(Heaviside(-t - 1), t, -3, 3)").unwrap();
    let f = (-(&k * &t)).exp().laplace(&t, &s);
    assert!(same(&f, &(1 / (&s + &k))), "{f}");
    assert!(close(one(&f), 2.0 / 7.0), "{f}");
}

/// The same with a symbolic upper limit and with a `Limit`: the constant
/// comes back as it was written.
///
/// Before: `Sum(Piecewise(sin(t²) if t < 2, 0 if True), t, 1, m)` became
/// `Sum(sin(t²) − sin(t²)·H(t − 2), t, 1, m)` (for `m = 2`: `sin 1 +
/// sin(4)/2` instead of `sin 1`), and `Limit(H(t + 1), t, −2)`, which is 0,
/// became `Limit(1, t, −2)`.
///
/// Oracle: mpmath `quad(lambda t: sin(1)*exp(-t)*exp(-s*t), [0, inf])` =
/// 0.420735492403948253 at `s = 1` (returned before: 0.231534868577); for
/// the limit, the definition (`H(t + 1) = 0` near `t = −2`).
#[test]
fn laplace_returns_constants_with_a_bound_variable_as_written() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let k = ctx
        .parse("Sum(Piecewise(sin(t^2) if t < 2, 0 if True), t, 1, m)")
        .unwrap();
    let f = (&k * (-&t).exp()).laplace(&t, &s);
    assert!(same(&f, &(&k / (&s + 1))), "{f}");
    assert!(
        close(at(&ctx, &f, &s, 1, 1), 0.420_735_492_403_948_25),
        "{f}"
    );

    let lim = ctx.parse("Limit(Heaviside(t + 1), t, -2)").unwrap();
    let f = (&lim * (-&t).exp() + &t).laplace(&t, &s);
    assert!(same(&f, &(&lim / (&s + 1) + s.powi(-2))), "{f}");
}

/// The arena shares nodes: in `H(t + 1)·e^{−t} + K·e^{−2t}` with `K =
/// Σ_{t=−3}^{3} H(t + 1)` the step outside the sum and the summand are one
/// node.  The free one is `1` on `t > 0`; the bound one is not touched.
///
/// Before: `1/(s + 1) + 7/(s + 2)`.
///
/// Oracle: `1/(s + 1) + (9/2)/(s + 2)`; at `s = 1`: mpmath
/// `quad(lambda t: (exp(-t) + 4.5*exp(-2*t))*exp(-t), [0, inf])` = 2.0
/// (returned before: 2.83333333333).
#[test]
fn laplace_rewrites_the_free_copy_of_a_shared_step_only() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let e = ctx
        .parse("Heaviside(t + 1)*exp(-t) + Sum(Heaviside(t + 1), t, -3, 3)*exp(-2*t)")
        .unwrap();
    let f = e.laplace(&t, &s);
    assert!(close(at(&ctx, &f, &s, 1, 1), 2.0), "{f}");
    assert!(close(at(&ctx, &f, &s, 3, 1), 0.25 + 0.9), "{f}");
}

/// What must not change: a binder of *another* variable is an ordinary
/// node (the transform variable is free in it), and the variable is free
/// in the limits of a sum over itself.
///
/// Oracle: linearity, `L{e^{−kt}} = 1/(s + k)`; `L{e^{−t}·H(t − k)} =
/// e^{−k(s+1)}/(s + 1)`; `Σ_{t=1}^{t} t² = t(t + 1)(2t + 1)/6`, whose
/// transform is `2/s⁴ + 1/s³ + 1/(6s²)` (19/6 at `s = 1`).
#[test]
fn laplace_still_enters_binders_of_other_variables() {
    let ctx = Context::new();
    let (t, s) = (ctx.symbol("t"), ctx.symbol("s"));
    let f = ctx
        .parse("Sum(exp(-k*t), k, 1, 3)")
        .unwrap()
        .laplace(&t, &s);
    assert!(
        same(&f, &(1 / (&s + 1) + 1 / (&s + 2) + 1 / (&s + 3))),
        "{f}"
    );
    let f = ctx
        .parse("Sum(Heaviside(t - k), k, 1, 2)*exp(-t)")
        .unwrap()
        .laplace(&t, &s);
    let want = (-2.0_f64).exp() / 2.0 + (-4.0_f64).exp() / 2.0;
    assert!(close(at(&ctx, &f, &s, 1, 1), want), "{f}");
    let f = ctx.parse("Sum(t^2, t, 1, t)").unwrap().laplace(&t, &s);
    assert!(close(at(&ctx, &f, &s, 1, 1), 19.0 / 6.0), "{f}");
    // A sum whose upper limit is `t` is not a constant factor.
    let g = ctx
        .parse("Sum(sin(k^2), k, 1, t)*exp(-t)")
        .unwrap()
        .laplace(&t, &s);
    assert!(g.has_unevaluated(), "{g}");
    // … while one over another range is.
    let k = ctx.parse("Sum(sin(k^2), k, 1, m)").unwrap();
    let g = (&k * (-&t).exp()).laplace(&t, &s);
    assert!(same(&g, &(&k / (&s + 1))), "{g}");
}

// ── Fourier ─────────────────────────────────────────────────────────────────

/// `F{K·e^{−x²}}` and `F{K/(x² + 1)}` with `K = Σ_{x=1}^{3} [x < 2] = 1` and
/// `K = Σ_{x=1}^{3} [x ≤ 2]·x = 3`.
///
/// Before: `(3/2)·√π·e^{−w²/4}` and `2π·e^{−|w|}` (the bound `Piecewise`
/// rewritten to steps: `1 + 1/2`, `1 + 2·(1/2)`).
///
/// Oracle: mpmath `quad(lambda x: K*exp(-x*x)*exp(-1j*w*x), [-inf, -2, 0, 2,
/// inf])` = 1.38038844704314297 at `w = 1`, and `2*quadosc(lambda x:
/// 3*cos(w*x)/(x*x + 1), [0, inf], omega=w)` = 3.46718204937276515
/// (returned before: 2.07058267056, 2.31145469958).
#[test]
fn fourier_leaves_a_piecewise_under_a_sum_over_the_variable() {
    let ctx = Context::new();
    let (x, w) = (ctx.symbol("x"), ctx.symbol("w"));
    let k = ctx
        .parse("Sum(Piecewise(1 if x < 2, 0 if True), x, 1, 3)")
        .unwrap();
    let f = (&k * (-x.powi(2)).exp()).fourier_transform(&x, &w).unwrap();
    let want = ctx.pi().sqrt() * (-w.powi(2) / 4).exp();
    assert!(same(&f, &want), "{f}");
    assert!(close(at(&ctx, &f, &w, 1, 1), 1.380_388_447_043_143), "{f}");

    let k = ctx
        .parse("Sum(Piecewise(x if x <= 2, 0 if True), x, 1, 3)")
        .unwrap();
    let f = (&k / (x.powi(2) + 1)).fourier_transform(&x, &w).unwrap();
    assert!(same(&f, &(ctx.pi() * 3 * (-w.abs()).exp())), "{f}");
    assert!(close(at(&ctx, &f, &w, 1, 1), 3.467_182_049_372_765), "{f}");

    // The inverse: (1/2π)∫ K e^{−w²} e^{iwx} dw = e^{−x²/4}/(2√π).
    // mpmath: 0.219695644733861199 at x = 1 (returned before: 0.329543467101).
    let k = ctx
        .parse("Sum(Piecewise(1 if w < 2, 0 if True), w, 1, 3)")
        .unwrap();
    let g = (&k * (-w.powi(2)).exp())
        .inverse_fourier_transform(&w, &x)
        .unwrap();
    assert!(
        close(at(&ctx, &g, &x, 1, 1), 0.219_695_644_733_861_2),
        "{g}"
    );
}

/// A constant is not a function with compact support.  The integral of
/// `H(x + 1)` over `[−3, 3]` is 4, whose transform is `8π·δ(w)`; times the
/// window `H(x + 1) − H(x − 1)` it is `8·sin(w)/w`.
///
/// Before: `−6i·e^{iw}/w` and `−6i·e^{iw}/w + 6i·e^{−iw}/w` (= `12·sin(w)/w`):
/// the constant "depended" on `x`, no rule applied, and the fallback for
/// compactly supported functions took the bound step for a step of the
/// function and read it on the positive axis.
///
/// The formal integral does not evaluate, and the Fourier transform refuses
/// a result that holds an unevaluated node, so both are refusals now.  With
/// the sum `Σ_{x=−3}^{3} H(x + 1) = 9/2`, which evaluates, the window gives
/// `9·sin(w)/w`.
///
/// Oracle: mpmath `quad(lambda x: 4*exp(-1j*w*x), [-1, 0, 1])` =
/// 6.73176787846317205 at `w = 1` (returned before: 10.0976518177); `9
/// sin(1)` = 7.57323886327106909.
#[test]
fn fourier_does_not_take_a_bound_step_for_compact_support() {
    let ctx = Context::new();
    let (x, w) = (ctx.symbol("x"), ctx.symbol("w"));
    let k = ctx.parse("Integral(Heaviside(x + 1), x, -3, 3)").unwrap();
    assert!(k.fourier_transform(&x, &w).is_err());
    let window = (&x + 1).heaviside() - (&x - 1).heaviside();
    assert!((&k * &window).fourier_transform(&x, &w).is_err());

    let k = ctx.parse("Sum(Heaviside(x + 1), x, -3, 3)").unwrap();
    let f = k.fourier_transform(&x, &w).unwrap();
    assert!(same(&f, &(ctx.pi() * 9 * w.dirac_delta())), "{f}");
    let f = (&k * &window).fourier_transform(&x, &w).unwrap();
    assert!(close(at(&ctx, &f, &w, 1, 1), 7.573_238_863_271_069), "{f}");
}

/// `RootOf(x⁵ − x + 1, 0)` is a number, whatever its polynomial's variable
/// is called.  The Fourier transform, its inverse and all three conventions
/// pull it out.
///
/// Before: "no transform rule applies to exp(-x^2)*RootOf(x^5 - x + 1, 0)"
/// (the Laplace transform answered).
///
/// Oracle: mpmath, the real root from `polyroots([1, 0, 0, 0, -1, 1])` =
/// −1.16730397826141868…: `quad(lambda x: K*exp(-x*x)*exp(-1j*w*x), …)` =
/// −1.61133292577956246 at `w = 1`; `K = polyroots([1, 0, 1, 1])` real,
/// `(1/π)·quadosc(lambda w: K*cos(w*x)/(w*w + 1), [0, inf], omega=x)` =
/// −0.125507185583994637 at `x = 1`.
#[test]
fn fourier_pulls_out_a_root_of_a_polynomial_in_the_variable() {
    let ctx = Context::new();
    let (x, w) = (ctx.symbol("x"), ctx.symbol("w"));
    let root = ctx.parse("RootOf(x^5 - x + 1, 0)").unwrap();
    let gauss = (-x.powi(2)).exp();
    let f = (&root * &gauss).fourier_transform(&x, &w).unwrap();
    let plain = gauss.fourier_transform(&x, &w).unwrap();
    assert!(same(&f, &(&root * &plain)), "{f}");
    assert!(
        close(at(&ctx, &f, &w, 1, 1), -1.611_332_925_779_562_5),
        "{f}"
    );
    for convention in [
        FourierConvention::UnitaryAngular,
        FourierConvention::Ordinary,
    ] {
        let f = (&root * &gauss)
            .fourier_transform_with(&x, &w, convention)
            .unwrap();
        let plain = gauss.fourier_transform_with(&x, &w, convention).unwrap();
        let (a, b) = (at(&ctx, &f, &w, 1, 2), at(&ctx, &plain, &w, 1, 2));
        assert!(close(a, -1.167_303_978_261_418_7 * b), "{f}");
    }
    // The constant alone: 2π·K·δ(w).
    let f = root.fourier_transform(&x, &w).unwrap();
    assert!(same(&f, &(ctx.pi() * 2 * &root * w.dirac_delta())), "{f}");

    let root = ctx.parse("RootOf(w^3 + w + 1, 0)").unwrap();
    let g = (&root / (w.powi(2) + 1))
        .inverse_fourier_transform(&w, &x)
        .unwrap();
    assert!(
        close(at(&ctx, &g, &x, 1, 1), -0.125_507_185_583_994_64),
        "{g}"
    );
}

// ── Mellin ──────────────────────────────────────────────────────────────────

/// `M{K·e^{−x}} = K·Γ(s)` with `K = Σ_{x=−3}^{3} H(x + 1) = 9/2`,
/// `M{K/(1 + x)} = K·π/sin(πs)` with `K = Σ_{x=1}^{3} [x < 2] = 1`, and
/// `M{K·x·H(1 − x)} = K/(s + 1)` with `K = Π_{x=−3}^{0} (H(x + 3) + 1) = 12`.
///
/// Before: `7·Γ(s)` (the bound step read on `x > 0`), `3π/(2 sin πs)` (the
/// bound `Piecewise` as steps), `16/(s + 1)`.
///
/// Oracle: mpmath `quad(lambda u: exp(s*u)*f(exp(u)), …)` (the Mellin
/// integral in `x = eᵘ`): 4.5 at `s = 1`, 3.14159265358979324 at `s =
/// 1/2`, 6.0 at `s = 1` (returned before: 7, 4.71238898038, 8).
#[test]
fn mellin_leaves_steps_and_piecewise_under_a_binder_of_the_variable() {
    let ctx = Context::new();
    let (x, s) = (ctx.symbol("x"), ctx.symbol("s"));

    let k = ctx.parse("Sum(Heaviside(x + 1), x, -3, 3)").unwrap();
    let (f, strip) = (&k * (-&x).exp()).mellin_transform(&x, &s).unwrap();
    assert!(same(&f, &(ctx.rational(9, 2) * s.gamma())), "{f}");
    assert_eq!(strip.to_string(), "re(s) > 0");

    let k = ctx
        .parse("Sum(Piecewise(1 if x < 2, 0 if True), x, 1, 3)")
        .unwrap();
    let (f, _) = (&k / (&x + 1)).mellin_transform(&x, &s).unwrap();
    assert!(close(at(&ctx, &f, &s, 1, 2), std::f64::consts::PI), "{f}");

    let k = ctx
        .parse("Product(Heaviside(x + 3) + 1, x, -3, 0)")
        .unwrap();
    let (f, _) = (&k * &x * (1 - &x).heaviside())
        .mellin_transform(&x, &s)
        .unwrap();
    assert!(same(&f, &(12 / (&s + 1))), "{f}");

    // The free step is read on x > 0, the bound one is not (shared node).
    // 9/2·2^{−s}·Γ(s) + Γ(s): 3.25 at s = 1 (returned before: 4.5).
    let e = ctx
        .parse("Heaviside(x + 1)*exp(-x) + Sum(Heaviside(x + 1), x, -3, 3)*exp(-2*x)")
        .unwrap();
    let (f, _) = e.mellin_transform(&x, &s).unwrap();
    assert!(close(at(&ctx, &f, &s, 1, 1), 3.25), "{f}");
}

/// `sin(x)/x + K` with `K = Σ_{x=−3}^{3} H(−x − 1) = 5/2` has no Mellin
/// transform (a non-zero constant: the integral diverges).
///
/// Before: `sin(π(s − 1)/2)·Γ(s − 1)` on `0 < Re s < 2` — the transform of
/// `sin(x)/x` alone; the bound falling steps were read as `0` on `x > 0`
/// and the constant vanished.
///
/// Oracle: `∫₀^∞ x^{s−1}·(5/2) dx` diverges for every `s`.
#[test]
fn mellin_does_not_drop_a_constant_made_of_bound_steps() {
    let ctx = Context::new();
    let (x, s) = (ctx.symbol("x"), ctx.symbol("s"));
    let k = ctx.parse("Sum(Heaviside(-x - 1), x, -3, 3)").unwrap();
    assert_eq!(k.eval(), ctx.rational(5, 2));
    let e = x.sin() / &x + &k;
    assert!(e.mellin_transform(&x, &s).is_err());
    // … and with the decaying factor it is 5/2·Γ(s).
    let (f, _) = (&k * (-&x).exp()).mellin_transform(&x, &s).unwrap();
    assert!(same(&f, &(ctx.rational(5, 2) * s.gamma())), "{f}");
}

/// A `RootOf` in the variable is a constant factor of the Mellin transform
/// and of its inverse.
///
/// Before: "no transform rule applies to exp(-x^2)*RootOf(x^3 + x + 1, 0)",
/// "no inverse rule applies to Gamma(s)*RootOf(s^5 - s + 1, 0)".
///
/// Oracle: mpmath, `K = polyroots([1, 0, 1, 1])` real = −0.682327803828019…,
/// `quad(lambda u: exp(s*u)*K*exp(-exp(2*u)), [-inf, 0, inf])` =
/// −0.604697271737438178 at `s = 1` (`K·Γ(1/2)/2`); the inverse of
/// `K₅·Γ(s)` is `K₅·e^{−x}` = −0.429427135200012… at `x = 1`.
#[test]
fn mellin_pulls_out_a_root_of_a_polynomial_in_the_variable() {
    let ctx = Context::new();
    let (x, s) = (ctx.symbol("x"), ctx.symbol("s"));
    let root = ctx.parse("RootOf(x^3 + x + 1, 0)").unwrap();
    let (f, strip) = (&root * (-x.powi(2)).exp())
        .mellin_transform(&x, &s)
        .unwrap();
    assert!(same(&f, &(&root * (&s / 2).gamma() / 2)), "{f}");
    assert_eq!(strip.to_string(), "re(s) > 0");
    assert!(
        close(at(&ctx, &f, &s, 1, 1), -0.604_697_271_737_438_2),
        "{f}"
    );

    let root = ctx.parse("RootOf(s^5 - s + 1, 0)").unwrap();
    let g = (&root * s.gamma())
        .inverse_mellin_transform(&s, &x)
        .unwrap();
    assert!(same(&g, &(&root * (-&x).exp())), "{g}");
    assert!(
        close(at(&ctx, &g, &x, 1, 1), -0.429_427_135_200_012_2),
        "{g}"
    );
}

// ── Z, and the inverses that already asked the right question ───────────────

/// The Z transform, the inverse Laplace and the inverse Z transforms took
/// constants with a bound variable out before as well (they asked for free
/// occurrences and have no pass that rewrites steps in depth).  Pinned so
/// that the shared helpers keep it so.
///
/// Oracle: mpmath, `K = Σ_{n=−3}^{3} H(n + 1) = 9/2`: `nsum(lambda n:
/// K*2**n*z**(-n), [0, inf])` = 13.5 at `z = 3`; `invertlaplace(lambda s:
/// K/(s + 1), 1)` = 1.65545748527149… (`9/(2e)`); the coefficient of
/// `z⁻⁴` of `K·z/(z − 2)` is `K·2⁴` = 72.
#[test]
fn z_and_the_inverse_transforms_keep_constants_with_a_bound_variable() {
    let ctx = Context::new();
    let (n, z) = (ctx.symbol("n"), ctx.symbol("z"));
    let k = ctx.parse("Sum(Heaviside(n + 1), n, -3, 3)").unwrap();
    let f = (&k * ctx.int(2).pow(&n)).z_transform(&n, &z).unwrap();
    assert!(same(&f, &(&k * &z / (&z - 2))), "{f}");
    assert!(close(at(&ctx, &f, &z, 3, 1), 13.5), "{f}");

    let k = ctx.parse("Sum(Heaviside(z + 1), z, -3, 3)").unwrap();
    let x = (&k * &z / (&z - 2)).inverse_z_transform(&z, &n).unwrap();
    assert!(same(&x, &(&k * ctx.int(2).pow(&n))), "{x}");
    assert!(close(at(&ctx, &x, &n, 4, 1), 72.0), "{x}");

    let (s, t) = (ctx.symbol("s"), ctx.symbol("t"));
    let k = ctx.parse("Sum(Heaviside(s + 1), s, -3, 3)").unwrap();
    let g = (&k / (&s + 1)).inverse_laplace(&s, &t);
    assert!(same(&g, &(&k * (-&t).exp())), "{g}");
    assert!(
        close(at(&ctx, &g, &t, 1, 1), 4.5 / std::f64::consts::E),
        "{g}"
    );
}

// ── One home ────────────────────────────────────────────────────────────────

/// The four transform modules keep no copy of the shared helpers and do not
/// ask the structural question (`walk::contains`) about their variable: the
/// only `walk::contains` calls left are the four searches for the imaginary
/// unit in `fourier_transform.rs`.  Before 0.60 each module had its own
/// `linear_in` (56 lines, four copies), `split_independent` /
/// `split_factors`, `extract_linear_coeff`, `require_symbol`, `param_sign`,
/// `need_sign` and `fail`, and the copies of the first two pairs had
/// drifted apart on whether a bound occurrence counts.
#[test]
fn the_transform_modules_share_their_helpers() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/calculus");
    let helpers = [
        "fn contains_var(",
        "fn depends_on(",
        "fn linear_in(",
        "fn extract_linear_coeff(",
        "fn split_independent(",
        "fn split_factors(",
        "fn require_symbol(",
        "fn param_sign(",
        "fn need_sign(",
        "fn fail(",
        "fn ifail(",
        "fn zfail(",
    ];
    for (file, structural) in [
        ("laplace.rs", 0),
        ("z_transform.rs", 0),
        ("fourier_transform.rs", 4),
        ("mellin.rs", 0),
    ] {
        let src = std::fs::read_to_string(dir.join(file)).unwrap();
        for h in helpers {
            assert!(!src.contains(h), "{file} defines its own `{h}…)`");
        }
        assert_eq!(
            src.matches("walk::contains(").count(),
            structural,
            "{file}: a dependence test must be transform_util::depends_on"
        );
        assert_eq!(src.matches("walk::has_free_var(").count(), 0, "{file}");
        assert!(src.contains("crate::calculus::transform_util::"), "{file}");
    }
    let shared = std::fs::read_to_string(dir.join("transform_util.rs")).unwrap();
    for h in &helpers[1..10] {
        assert_eq!(shared.matches(h).count(), 1, "transform_util.rs: `{h}…)`");
    }
}
