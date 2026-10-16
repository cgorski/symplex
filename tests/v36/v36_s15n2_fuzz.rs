//! Tests of session 15 (s15n2): the findings of the nightly Fuzz workflow
//! (runs 38856944938 and 38871441556) and of the campaigns that followed.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;
use symplex::stats::Distribution;
use symplex::stats::numdist;

use crate::perf_assert::{CpuTimer, assert_fast};

fn rel(a: f64, b: f64) -> f64 {
    ((a - b) / b).abs()
}

// ── numdist: the tails at the exact argument ───────────────────────────────

/// `gamma::cdf` rounded `x/θ` before taking the tail, and near the mean of a
/// large shape a tail moves by `κ ≈ √k·|z|` times a relative change of its
/// argument: fuzz_stats (run 38856944938, input `y8vLT9o=`) found
/// `1.5258789062474784e-5`, `1.6·10⁻¹²` relative off, at `k = 1.7·10⁷`, 4.2
/// standard deviations below the mean (scipy 1.18.1: `1.5258789062474779e-5`,
/// the same error).  Oracle: mpmath quadrature of the regularised integrand
/// in log space at the exact quotient, 40 and 80 digits:
/// `1.525878906245095412462416e-5`.
#[test]
fn gamma_cdf_is_taken_at_the_exact_quotient() {
    let k = 4594215465867089.0 / 268435456.0;
    let theta = 4611686018427389.0 / 2f64.powi(62);
    let x = 1.7097542248506146e4;
    let want = 1.5258789062450953e-5;
    let p = numdist::gamma::cdf(x, k, theta);
    assert!(rel(p, want) < 1e-14, "{p:e}");
    let q = numdist::gamma::sf(x, k, theta);
    assert!(((1.0 - q) - want).abs() < 1e-16, "{q:e}");
}

/// With `θ = 1` (and for χ², `θ = 2`) the quotient is exact and the tails
/// are those of the regularised incomplete gamma function, bit for bit.
/// Oracle: the identity `P(X ≤ x) = P(k, x/θ)` at an exact quotient.
#[test]
fn gamma_cdf_with_an_exact_quotient_is_the_incomplete_gamma_function() {
    for (k, x) in [
        (1.7e7, 1.70975e7),
        (3.5, 2.25),
        (0.25, 1e-3),
        (2e6, 2.0003e6),
    ] {
        assert_eq!(
            numdist::gamma::cdf(x, k, 1.0),
            numdist::gammainc_lower_regularized_f64(k, x)
        );
        assert_eq!(
            numdist::chi2::sf(2.0 * x, 2.0 * k),
            numdist::gammainc_upper_regularized_f64(k, x)
        );
    }
}

/// The F kernel rounded its beta argument `d₁x/(d₁x + d₂)` (a product, a
/// sum and a quotient) before taking the tail: at shapes near `10¹³` and
/// 22 standard deviations below the mean `f::cdf` was `3.3·10⁻⁹` relative
/// off (`7.359802160743224e-96`), at shapes near `5·10¹¹` `1.2·10⁻¹⁰`
/// (`4.924108302147043e-11`).  Oracle: mpmath, `I_z(d₁/2, d₂/2)` at the
/// exact `z` as `z^a y^(b−1)/B(a, b)·∫₀¹ (1 − v)^(a−1)(1 + (z/y)v)^(b−1) dv`
/// (tanh-sinh quadrature), 30 and 50 digits.
#[test]
fn f_cdf_is_taken_at_the_exact_beta_argument() {
    for (x, d1, d2, want) in [
        (
            0.9999882534670727,
            7418951616786.722,
            39091640841817.484,
            7.359802136194096e-96,
        ),
        (
            0.9999875672185353,
            4325790938773.364,
            98978933297627.03,
            6.135566681589205e-72,
        ),
        (
            0.9999778557448259,
            546410631043.79816,
            248233254780.30237,
            4.924108301547372e-11,
        ),
    ] {
        let p = numdist::f::cdf(x, d1, d2);
        assert!(rel(p, want) < 1e-12, "F({d1}, {d2}) at {x}: {p:e}");
    }
}

/// The incomplete beta kernel at a subnormal argument: `bgrat` formed
/// `−ν ln y` from `y = 1 − x` and the products of `x` with few bits left, and
/// `I_{5·10⁻³²⁴}(4.5·10⁻⁹, 10⁻³)` came out `0.9999921289941`, `1.1·10⁻¹⁰`
/// off (local fuzz_stats campaign, input `q7QABu0l`).  Oracle: mpmath,
/// `xᵃ/(a B(a, b))·₂F₁(a, 1 − b; a + 1; x)` and `betainc`, 40 and 80
/// digits: `0.99999212888009403149`, `1 − I = 7.8711199059685070483e-6`.
#[test]
fn beta_kernel_at_a_subnormal_argument() {
    let a = 5454849930934295.0 / 1208925819614629174706176.0;
    let b = 4611686018427389.0 / 2f64.powi(62);
    let x = f64::from_bits(1);
    assert!(rel(numdist::beta::cdf(x, a, b), 0.999992128880094) < 1e-15);
    assert!(rel(numdist::beta::sf(x, a, b), 7.871119905968507e-6) < 1e-14);
}

// ── fuzz_stats: the slow continued fraction ───────────────────────────────

/// fuzz_stats (run 38871441556, input `/gsLCwsL`: `F(3·10¹²⁸, 1)`
/// transformed by `x − 9`) timed out after 44 s: the symbolic tails at the
/// quantile of level `10⁻³⁰⁰`, `x = 7.28·10⁻⁴`, are `I_z(1.5·10¹²⁸, ½)` with
/// `1 − z = 4.6·10⁻¹²⁶`, and evalf's continued fraction rounded its argument
/// to the working precision, where it is 1: the fraction cannot converge
/// there and ran its two million steps (5.7 s in release) before refusing.
/// Oracle: the limit `d₁ → ∞` (the error is `O(1/d₁)`, `10⁻¹²⁸`):
/// `P(F ≤ x) = P(χ²₁ ≥ 1/x) = erfc(√(1/(2x)))` = `9.9999999999999056924e-301`
/// (mpmath, 40 and 80 digits), so `P(F > x)` is 1 to every digit.  The lower
/// tail may be refused (its bound is beyond the precision cap), but at once.
#[test]
fn f_with_a_huge_first_degree_of_freedom_answers_its_far_tail_at_once() {
    let ctx = Context::new();
    let d = Distribution::f_distribution(ctx.int(3) * ctx.int(10).powi(128), ctx.one());
    let x = ctx.from_f64(7.278695108077497e-4).unwrap();
    let sf = assert_fast!(Duration::from_secs(2), d.sf(&x).eval_f64());
    assert_eq!(sf.unwrap(), 1.0);
    if let Ok(p) = assert_fast!(Duration::from_secs(2), d.cdf(&x).eval_f64()) {
        assert!(rel(p, 9.999999999999906e-301) < 1e-12, "{p:e}");
    }
    // A central point and a nearer tail are computed.  Oracle: the same limit,
    // `erfc(√(1/(2x)))` at `x = 2.1981093383177646` and `0.0022029193507435035`.
    for (x, want) in [
        (2.1981093383177646, 0.5000000000000031),
        (0.0022029193507435035, 9.999999999999625e-101),
    ] {
        let x = ctx.from_f64(x).unwrap();
        let p = assert_fast!(Duration::from_secs(2), d.cdf(&x).eval_f64()).unwrap();
        assert!(rel(p, want) < 1e-14, "{p:e}");
    }
}

/// A continued fraction that cannot converge within its budget is refused
/// before its first step: `F(10²⁰, 10²⁰).cdf(1)` (the fraction at its switch
/// point needs about `0.4·(10²⁰·wp²)^{1/3} ≈ 3·10⁷` steps, against a budget
/// of `2·10⁶`) ran 18 s in release to refuse.  Oracle: the symmetry
/// `I_{1/2}(a, a) = 1/2`, which a value, if one is returned, must be.
#[test]
fn a_hopeless_beta_continued_fraction_is_refused_at_once() {
    let ctx = Context::new();
    let n = ctx.int(10).powi(20);
    let d = Distribution::f_distribution(n.clone(), n);
    if let Ok(p) = assert_fast!(Duration::from_secs(1), d.cdf(&ctx.one()).eval_f64()) {
        assert!((p - 0.5).abs() < 1e-15, "{p}");
    }
}

// ── The local fuzz_stats campaign ─────────────────────────────────────────────────

/// `Poisson(λ)` at a rate near `3·10¹⁰`: the symbolic tails are
/// `Γ(k + 1, λ)/k!` and `γ(k + 1, λ)/k!`, whose numerators (about
/// `2^(9.2·10¹¹)`) overflow the arbitrary-precision exponent range; the
/// upper one was refused at once, the lower one only after its series had
/// summed two million terms (1.1 s in release, a 10.6 s slow unit of
/// fuzz_stats, input `C////7YL`).  The `numdist` kernel answers.  Oracle:
/// mpmath, `Q(k + 1, λ)` as `λ^(k+1)e^(−λ)/Γ(k + 1)·∫₀^∞ e^(k ln(1 + v) − λv) dv`
/// (tanh-sinh), 30 and 50 digits: `0.50000231477623983`, and
/// `0.50000042619187921` at `λ = 1.9·10¹¹` (input `yyf7WsDL`).
#[test]
fn poisson_at_a_huge_rate_refuses_its_symbolic_tails_at_once() {
    let ctx = Context::new();
    for (num, den, k, want) in [
        (
            3620896395585425_i64,
            131072_i64,
            2.7625247159e10,
            0.5000023147762398,
        ),
        (1567614735189601, 8192, 1.91359220604e11, 0.5000004261918792),
    ] {
        let d = Distribution::poisson(ctx.rational(num, den));
        let x = ctx.from_f64(k).unwrap();
        for e in [d.cdf(&x), d.sf(&x)] {
            if let Ok(p) = assert_fast!(Duration::from_secs(1), e.eval_f64()) {
                assert!(rel(p, 0.5) < 1e-5, "{p}");
            }
        }
        let p = numdist::poisson::cdf(k, num as f64 / den as f64);
        assert!(rel(p, want) < 1e-15, "{p:e}");
    }
}

/// `Gamma(8·10²⁰⁴, 1)` at its quantile of level `10⁻³⁰⁰` (the double nearest
/// the shape, every quantile of it): the series of `γ(s, x)` was given a
/// budget check that is void far from the mean, where the sum may reach
/// `(s + 1)/(s − x + 1)` times its first term, `2¹⁰²²` here, and ran its ten
/// million terms (fuzz_stats timed out, input `TzPq8cs=`).  Oracle: `Γ(s)`
/// for `s = 8·10²⁰⁴` is about `2^(5.4·10²⁰⁷)`, beyond the exponent range: a
/// refusal, but at once.
#[test]
fn gamma_at_a_huge_shape_refuses_its_symbolic_tails_at_once() {
    let ctx = Context::new();
    let d = Distribution::gamma(ctx.int(8) * ctx.int(10).powi(204), ctx.one());
    let x = ctx.from_f64(8e204).unwrap();
    for e in [d.cdf(&x), d.sf(&x)] {
        if let Ok(p) = assert_fast!(Duration::from_secs(1), e.eval_f64()) {
            assert!((0.0..=1.0).contains(&p), "{p}");
        }
    }
}

/// `transformed` of a finite table merged the images by `equals` against
/// every earlier one that was not a rational: `√X` of a 2000-sided die made
/// about two million `equals` calls, each an evaluation (fuzz_stats timed
/// out after 30 s, input `/44AEwAg`; 1.9 s in release at 400 sides, now
/// 5 ms).  The certified values separate distinct images first.  The CDF of
/// the image asked the mass of each atom, and each mass was found by
/// `equals` against every atom before it (the same quadratic count; the
/// timeouts of `Die(1000)` under `atan` and `cbrt`).  Oracle: `k ↦ √k` is
/// injective, so the image of a 400-sided die has 400 atoms and
/// `P(√X ≤ √200) = 1/2`; `k ↦ √2·|k − 200|` merges `200 ± j`, so its image
/// has 201.
#[test]
fn transformed_table_merges_images_by_their_values_first() {
    let ctx = Context::new();
    let die = Distribution::die(ctx.int(400));
    let x = ctx.symbol("t");
    let atoms = |d: &Distribution| d.support().as_points().map(|p| p.len());
    let y = assert_fast!(Duration::from_secs(2), die.transformed(&x, &x.sqrt())).unwrap();
    assert_eq!(atoms(&y), Some(400));
    let half = assert_fast!(Duration::from_secs(2), y.cdf(&ctx.int(200).sqrt()));
    assert_eq!(half, ctx.rational(1, 2));
    let g = (&x - &ctx.int(200)).powi(2).sqrt() * ctx.int(2).sqrt();
    let y = assert_fast!(Duration::from_secs(2), die.transformed(&x, &g)).unwrap();
    assert_eq!(atoms(&y), Some(201));
    // `e^{−k}` underflows the doubles from `k = 745` on: those images are
    // separated by their logarithms (`Die(7000)` timed out, input
    // `Ty4ABgBA`).  Oracle: `k ↦ e^{−k}` is injective.
    let die = Distribution::die(ctx.int(900));
    let y = assert_fast!(Duration::from_secs(4), die.transformed(&x, &(-&x).exp())).unwrap();
    assert_eq!(atoms(&y), Some(900));
}

// ── fuzz_calculus: the slow refusal of a series ────────────────────────────

/// fuzz_calculus (run 38856944938, input
/// `ef//////////////eXl5eXl5ycnJyclfXw==`) timed out after 40 s on
/// `series(5/(3x·ln x·atanh(atanh x)), x, 1, 4)`: `atanh(atanh x)` tends to
/// `±iπ/2` with corrections in `1/ln(1 − x)`, so there is no power series
/// past the leading term and the expansion is refused, but Gruntz's limits
/// of the derivatives formatted every intermediate expression for disabled
/// debug lines (80 % of the time) and multiplied out products of `5·10⁵`
/// terms (`series(1/(ln x·atanh(atanh x)), x, 1, 4)`: 42 s in release; now
/// 0.3 s, and 0.4 s for the fuzz input, most of it the differentiation
/// fallback's limits of the retries `(x − 1)ᵏ·f`).  Oracle: the asymptotics
/// above (a refusal is the right answer).
#[test]
fn series_through_atanh_of_atanh_at_its_singular_point_is_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for f in [
        "((((5/3)/x)/ln(x))/atanh(atanh(x)))",
        "1/(ln(x)*atanh(atanh(x)))",
    ] {
        let e = ctx.parse(f).unwrap();
        let s = assert_fast!(Duration::from_secs(20), e.series(&x, &ctx.one(), 4));
        assert!(s.has_unevaluated(), "{f}: {s}");
    }
}
