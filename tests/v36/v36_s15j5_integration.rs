//! Tests of session 15 (s15j5): integration leftovers and the series slowdown.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `∫₀¹ dx/(e^{(1+10⁻²⁰i)x} − 2) = −0.6654466341020272668 −
/// 1.5707963267948966192i` and `∫₀¹ dx/(e^{(1+i)x} − 2) =
/// −0.62222333466099226611 − 0.54905862282866483089i` (mpmath `quad` along a
/// path through the upper half plane, clear of the pole `7·10⁻²¹` below `ln 2`
/// of the first, 30 and 50 digits).  Up to 0.54 both were refused: the
/// exponential substitution `u = e^{λx}` gave logarithms of `u − 2`, which
/// jump once per turn of the spiral `u`, and no antiderivative continuous on
/// the real line was known (`eval_complex64` of the first one is
/// `PrecisionExhausted`, of the second the same value).
#[test]
fn spiral_exponential_over_the_interval_is_integrated() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let (zero, one) = (ctx.int(0), ctx.int(1));
    let cases = [
        (
            "1/(exp((1+10^-20*I)*x) - 2)",
            "-0.6654466341020272668 - 1.5707963267948966192*i",
        ),
        (
            "1/(exp((1+I)*x) - 2)",
            "-0.62222333466099226611 - 0.54905862282866483089*i",
        ),
    ];
    for (f, want) in cases {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &zero, &one)
            .unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(v.eval_decimal(20).unwrap(), want, "{f}");
    }
}

/// Over longer intervals the antiderivative changes form at `x = ln 2`
/// (`|e^{(1+i)x}| = 2`) and the logarithms cross no cut on either side
/// (mpmath `quad` split at 40 points, 30 and 50 digits): `[0, 4]`:
/// `−0.78615809999099231768 − 0.79817211001548271092i`; `[−3, 2]`:
/// `−2.3785502989197393122 − 0.58511108190886332252i`; `[0, 10]`:
/// `−0.78539146524308822598 − 0.7854295581904349377i`;
/// `∫₋₁² dx/(e^{(2−3i)x} − 1/2)` (a decaying spiral):
/// `−0.42775462031857613432 − 0.7505274117664090546i`.  All refused up to 0.54.
#[test]
fn spiral_exponential_across_the_turning_point() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (
            "1/(exp((1+I)*x) - 2)",
            0,
            4,
            "-0.78615809999099231768 - 0.79817211001548271092*i",
        ),
        (
            "1/(exp((1+I)*x) - 2)",
            -3,
            2,
            "-2.3785502989197393122 - 0.58511108190886332252*i",
        ),
        (
            "1/(exp((1+I)*x) - 2)",
            0,
            10,
            "-0.78539146524308822598 - 0.7854295581904349377*i",
        ),
        (
            "1/(exp((2-3*I)*x) - 1/2)",
            -1,
            2,
            "-0.42775462031857613432 - 0.7505274117664090546*i",
        ),
    ];
    for (f, a, b, want) in cases {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &ctx.int(a), &ctx.int(b))
            .unwrap_or_else(|e| panic!("{f} [{a}, {b}]: {e}"));
        assert_eq!(v.eval_decimal(20).unwrap(), want, "{f} [{a}, {b}]");
    }
}

/// `∫ e^{(1+i)x}/(e^{(1+i)x} − 3) dx` was `ln(e^{(1+i)x} − 3)/(1 + i)` up to
/// 0.54 (the u-substitution), which jumps by `2πi/(1 + i)` at `x = π`, where
/// `e^{(1+i)x} − 3 = −e^π − 3` crosses the cut of `ln`.  The antiderivative now
/// has no jump there and its derivative is the integrand; `∫₀³` is
/// `1.1608471406202166784 − 1.2840792214432286964i` (mpmath `quad`, 30 and 50
/// digits; refused up to 0.54).
#[test]
fn spiral_logarithm_of_the_u_substitution_is_continuous() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "exp((1+I)*x)/(exp((1+I)*x) - 3)");
    let big_f = f.integrate(&x);
    assert!(!big_f.to_string().contains("Integral"), "{big_f}");
    let at = |t: &str| {
        big_f
            .subs(&x, &parse(&ctx, t))
            .eval_complex64()
            .unwrap_or_else(|e| panic!("F({t}): {e}"))
    };
    let jump = at("pi + 10^-6") - at("pi - 10^-6");
    assert!(jump.norm() < 1e-4, "jump {jump:?}");
    // F′ = f at a few points on either side of ln 3 (where the form changes).
    for t in ["1/2", "2", "4"] {
        let d = big_f
            .diff(&x)
            .subs(&x, &parse(&ctx, t))
            .eval_complex64()
            .unwrap();
        let v = f.subs(&x, &parse(&ctx, t)).eval_complex64().unwrap();
        assert!(
            (d - v).norm() < 1e-12 * v.norm().max(1.0),
            "{t}: {d:?} vs {v:?}"
        );
    }
    let v = f
        .try_integrate_definite(&x, &ctx.int(0), &ctx.int(3))
        .unwrap();
    assert_eq!(
        v.eval_decimal(20).unwrap(),
        "1.1608471406202166784 - 1.2840792214432286964*i"
    );
}

/// `H(cos(x/3 + 1/2))` over `[10¹⁰⁰, 10¹⁰⁰ + 200]` is the measure of the
/// points where `cos > 0`: `100.22019327973250007` (exact: the zeros `3(π/2 +
/// kπ − 1/2)`, mpmath at 250 digits).  Up to 0.54 `try_integrate_definite`
/// refused it ("Heaviside with a non-linear argument") and `integrate_numeric`
/// and `eval_decimal` spent 18 s before `BudgetExhausted`; likewise
/// `H(cos(10x))` over `[10¹⁰⁰ + 1/3, 10¹⁰⁰ + 1/3 + 30]` = `14.920355262768992455`.
#[test]
fn heaviside_of_a_cosine_far_out_is_split_at_its_steps() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (
            "Heaviside(cos(x/3 + 1/2))",
            "10^100",
            "10^100 + 200",
            "100.22019327973250007",
        ),
        (
            "Heaviside(cos(10*x))",
            "10^100 + 1/3",
            "10^100 + 1/3 + 30",
            "14.920355262768992455",
        ),
    ];
    for (f, a, b, want) in cases {
        let (f, a, b) = (parse(&ctx, f), parse(&ctx, a), parse(&ctx, b));
        let v = f.try_integrate_definite(&x, &a, &b).unwrap();
        assert_eq!(v.eval_decimal(20).unwrap(), want);
        let n = f.integrate_numeric(&x, &a, &b).unwrap();
        let w: f64 = want.parse().unwrap();
        assert!((n / w - 1.0).abs() < 1e-14, "{n}");
    }
}

/// `Π(n | m) = Σⱼ nʲ·Jⱼ(m)` at `n → 0`: `J₀ = K(m)`, `J₁ = (K − E)/m`,
/// `J₂ = (2(1 + m)J₁ − J₀)/(3m)`.  `series(elliptic_pi(x, 1/2), x, 0, 3)` was
/// left unevaluated up to 0.54; its coefficients are `K(1/2) =
/// 1.8540746773013719184`, `J₁ = 2(K − E) = 1.0068615925073928318`, `J₂ =
/// 0.77767340014720451627` (mpmath `ellipk`, `ellipe`; central differences of
/// `ellippi(x, 1/2)` with step `10⁻⁸` at 50 digits agree to `10⁻¹⁶`).
#[test]
fn elliptic_pi_expands_at_a_vanishing_characteristic() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let s = parse(&ctx, "elliptic_pi(x, 1/2)").series(&x, &ctx.int(0), 3);
    assert!(!s.to_string().contains("Series"), "{s}");
    let want = [
        1.854_074_677_301_371_9,
        1.006_861_592_507_392_8,
        0.777_673_400_147_204_5,
    ];
    let mut c = s.clone();
    for (k, w) in want.iter().enumerate() {
        let c0 = c.subs(&x, &ctx.int(0)).eval_f64().unwrap();
        assert!((c0 - w).abs() < 1e-15, "x^{k}: {c0}");
        c = c.diff(&x) / ctx.int(k as i64 + 1);
    }
}

/// `series(elliptic_pi(x·y, x + y), x, 0, 4)` and
/// `elliptic_pi(2x, 1/cosh(x) + 1/2)` took 3.8 s and 4.1 s to stay unevaluated at 0.54 (0.3 s and 0.14 s
/// at 0.53): the differentiation fallback asked the limit engine for every
/// derivative, whose constants in `K` and `E` evaluate off the real range
/// since 0.54.  They are polynomials now; at `y = 3/10` the coefficient of
/// `x` of the first is `y·J₁(y) + K′(y)` = `0.85338459968877233551` (central
/// difference of mpmath's `ellippi(3x/10, x + 3/10)`, step `10⁻⁸`, 50
/// digits), and the second's coefficient of `x` is `2·J₁(3/2)` = `(4/3)(K(3/2)
/// − E(3/2))` = `1.2537683087153398690 − 2.3356950342597571494i` (on the cut:
/// the limit from below, as mpmath's `ellippi`, whose central difference
/// agrees to `3·10⁻¹⁶`).
#[test]
fn elliptic_pi_series_with_moving_parameter() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let y = ctx.symbol("y");
    let s = parse(&ctx, "elliptic_pi(x*y, x + y)").series(&x, &ctx.int(0), 4);
    assert!(!s.to_string().contains("Series"), "{s}");
    let c1 = s
        .diff(&x)
        .subs(&x, &ctx.int(0))
        .subs(&y, &ctx.rational(3, 10))
        .eval_f64()
        .unwrap();
    assert!((c1 - 0.853_384_599_688_772_3).abs() < 1e-14, "{c1}");
    let s2 = parse(&ctx, "elliptic_pi(2*x, 1/cosh(x) + 1/2)").series(&x, &ctx.int(0), 4);
    let c1 = s2.diff(&x).subs(&x, &ctx.int(0)).eval_complex64().unwrap();
    assert!((c1.re - 1.253_768_308_715_34).abs() < 1e-14, "{s2}");
    assert!((c1.im + 2.335_695_034_259_757).abs() < 1e-14, "{s2}");
}

/// `∫₋₂¹ ln(x² + (1 + i)x − 1) dx`: the argument meets the real axis at the
/// root `x = 0` of its imaginary part `x`, on the cut (`u(0) = −1`).  mpmath
/// `quad` split at 0 (30 and 50 digits): `0.48484569282837855449 −
/// 2.4980590722325776626i`; `∫₀¹`: `−0.24744489541501864804 +
/// 1.9914146263576491126i`; `∫₋₁¹ x·ln(…)`: `−0.2545050352350852167 +
/// 2.0805112725915349261i`; `∫₋₃³ (2x + i)/√(x² + ix − 2) dx`:
/// `−3.4373532139190091943i`.  Up to 0.54 all were refused ("the
/// antiderivative is discontinuous near x = 0"; the crossing was only
/// sampled); the crossings of a polynomial argument are located exactly.
#[test]
fn crossing_of_a_cut_by_a_polynomial_argument_is_continued() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let cases = [
        (
            "log(x^2 + (1+I)*x - 1)",
            -2,
            1,
            "0.48484569282837855449 - 2.4980590722325776626*i",
        ),
        (
            "log(x^2 + (1+I)*x - 1)",
            0,
            1,
            "-0.24744489541501864804 + 1.9914146263576491126*i",
        ),
        (
            "x*log(x^2 + (1+I)*x - 1)",
            -1,
            1,
            "-0.2545050352350852167 + 2.0805112725915349261*i",
        ),
        (
            "(2*x + I)/sqrt(x^2 + I*x - 2)",
            -3,
            3,
            "-3.4373532139190091943*i",
        ),
    ];
    for (f, a, b, want) in cases {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &ctx.int(a), &ctx.int(b))
            .unwrap_or_else(|e| panic!("{f} [{a}, {b}]: {e}"));
        assert_eq!(v.eval_decimal(20).unwrap(), want, "{f} [{a}, {b}]");
    }
}
