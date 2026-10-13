//! The speed of `limit` and `series` (0.44), and a Gruntz leading term
//! that depended on the form of `exp(−ln …)`.  The nightly `fuzz_calculus`
//! run timed out (30 s) on a limit at `π` of inverse functions of `xˣ`, whose
//! arguments approach a point of their cuts (`π^π > 1`): every expansion of
//! such an argument was repeated at a growing precision until the last
//! round took it as real anyway — a round of 28 terms whose coefficients
//! are large constants in `π^π` took 50 s.  Each test is a reproducer of one
//! cost found then, with a time limit generous for a debug build on a
//! loaded machine (the times quoted are release builds, CPU seconds), and
//! its answer checked against the oracle cited: mpmath 1.3 at 40 digits
//! along `a ± 10⁻²⁰` or `x = 10ᵏ`, SymPy 1.14 as a second opinion.

use std::time::{Duration, Instant};

use symplex::prelude::*;

const B: Direction = Direction::Both;
const R: Direction = Direction::Right;

/// `lim_{x→point} src` (direction `dir`) within `bound`.
fn limit_within(ctx: &Context, src: &str, point: &str, dir: Direction, bound: Duration) -> Ex {
    let f = ctx
        .parse(src)
        .unwrap_or_else(|e| panic!("`{src}` does not parse: {e}"));
    let x = ctx.symbol("x");
    let p = ctx.parse(point).unwrap();
    let t0 = Instant::now();
    let r = f.try_limit_dir(&x, &p, dir);
    let dt = t0.elapsed();
    assert!(dt < bound, "limit of `{src}` at {point} took {dt:?}");
    r.unwrap_or_else(|e| panic!("limit of `{src}` at {point} {dir:?} failed: {e}"))
}

/// The limit is `want` (parsed and evaluated), and, when `value` is given,
/// its value is `value` to 12 digits.
fn same_within(
    ctx: &Context,
    src: &str,
    point: &str,
    dir: Direction,
    want: &str,
    value: Option<(f64, f64)>,
    bound: Duration,
) {
    let got = limit_within(ctx, src, point, dir, bound);
    let want_ex = ctx.parse(want).unwrap().eval();
    assert_eq!(
        got, want_ex,
        "`{src}` at {point} {dir:?}: {got} (want {want})"
    );
    if let Some((re, im)) = value {
        let z = got.eval_complex64().unwrap();
        let scale = re.hypot(im).max(1.0);
        assert!(
            (z.re - re).abs() < 1e-12 * scale && (z.im - im).abs() < 1e-12 * scale,
            "`{src}` at {point}: {got} = {z}, want {re} + {im}i"
        );
    }
}

/// Before: 111 s (the fuzz input `timeout-e669748`; 47.5 s already in
/// 0.44's parent commit), now 0.1 s.  `acos` of `(π + ω)^{π + ω}`, a real
/// argument on the cut `(1, ∞)` of `acos`: realness was asked of the whole
/// side (`ln(ω + π)` is not real for `ω < −π`), so its expansion was
/// undecided until the widest precision round; and an expansion that had
/// lost precision through the pole of `1/(sin u − sin π^π)` was repeated
/// although its leading term was in view.
///
/// Oracle: mpmath `(acos(xˣ) − acos(π^π))/(sin(xˣ) − sin(π^π))` at
/// `π ± 10⁻²⁰`: `0.0837287583960872074…·i` from both sides, and the value
/// of the closed form `−π^π/(π^π·√(1 − (π^π)²)·cos(π^π))`
/// `0.08372875839608720747607…·i`.  The variants: `±10⁻²⁰` give
/// `2.1455369404430523139…·i` (`(acos(xˣ) − acos(π^π))/(x − π)`),
/// `−0.0837287583960872074…·i` (`asin`), `−4.75469423106932176…·i`
/// (`acosh(xˣ/40)`, `π^π/40 < 1` on the cut `(−∞, 1)`).
#[test]
fn inverse_functions_of_a_power_tower_on_their_cuts() {
    let ctx = Context::new();
    let bound = Duration::from_secs(20);
    same_within(
        &ctx,
        "(acos(x^x) - acos(pi^pi))/(sin(x^x) - sin(pi^pi))",
        "pi",
        B,
        "-1/(sqrt(1 - (pi^pi)^2)*cos(pi^pi))",
        Some((0.0, 0.083_728_758_396_087_2)),
        bound,
    );
    same_within(
        &ctx,
        "(acos(x^x) - acos(pi^pi))/(x - pi)",
        "pi",
        B,
        "-(pi^pi + pi^pi*log(pi))/sqrt(1 - (pi^pi)^2)",
        Some((0.0, 2.145_536_940_443_052)),
        bound,
    );
    same_within(
        &ctx,
        "(asin(x^x) - asin(pi^pi))/(sin(x^x) - sin(pi^pi))",
        "pi",
        B,
        "1/(sqrt(1 - (pi^pi)^2)*cos(pi^pi))",
        Some((0.0, -0.083_728_758_396_087_2)),
        bound,
    );
    same_within(
        &ctx,
        "(acosh(x^x/40) - acosh(pi^pi/40))/(x - pi)",
        "pi",
        B,
        "-(pi^pi/40 + pi^pi*log(pi)/40)*I/sqrt(1 - (pi^pi)^2/1600)",
        Some((0.0, -4.754_694_231_069_322)),
        bound,
    );
}

/// Before: 2.0 s (the fuzz slow unit `slow-unit-bad4c2f`; 10 s and 17 s
/// with `acos⁴` in a debug build), now 0.03 s (0.1 s and 0.2 s debug).
/// `acos x → π/2` on the cut of the outer `acos`, and `acos x` is real only
/// near `x = 0`, not on the whole side: the same escalation as above, each
/// time `acos(acos ω)` was expanded.
///
/// Oracle: mpmath `acos(acos(acos(±10⁻¹⁰)))/10⁻²⁰` = `1.5708·10²⁰ −
/// 0.8977·10²⁰·i`, the direction of `acos(acos(π/2)) = 1.5707963 −
/// 0.8977029·i` (`acos(acos(acos(π/2)))` for `acos⁴`).
#[test]
fn nested_inverse_cosines_at_zero() {
    let ctx = Context::new();
    let t0 = Instant::now();
    let bound = Duration::from_secs(8);
    let got = limit_within(&ctx, "acos(acos(acos(x)))/x^2", "0", B, bound);
    let want = ctx.parse("acos(acos(pi/2))*oo").unwrap().eval();
    assert_eq!(got, want);
    let got = limit_within(&ctx, "acos(acos(acos(acos(x))))/x^2", "0", B, bound);
    let want = ctx.parse("acos(acos(acos(pi/2)))*oo").unwrap().eval();
    assert_eq!(got, want);
    let dt = t0.elapsed();
    assert!(dt < bound, "the two limits took {dt:?}");
}

/// Before: did not return in 20 s (a debug build), now 2 ms.  The argument
/// `∛(−x⁻³) = (−1)^{1/3}/x` of `atan` tends to infinity in the direction
/// `(−1)^{1/3} = e^{iπ/3}`, whose real and imaginary parts the exact
/// decomposition does not see; the expansion at infinity was refused and
/// left to the differentiation fallback, which took the limits of ever
/// larger derivatives (`(−1)^{−59/3}`-type coefficients).  The direction's
/// signs come from its certified value now.
///
/// Oracle: mpmath (principal `∛`) `f(x) − S(x)` = `7.14·10⁻¹⁰ −
/// 1.24·10⁻⁹·i` at `x = 10⁻²` and `7.14·10⁻¹⁴ − 1.24·10⁻¹³·i` at `10⁻³`:
/// `O(x⁴)`.  SymPy: `PoleError` (atan at `oo·sign((−1)^{1/3})`).
#[test]
fn atan_of_a_cube_root_tending_to_complex_infinity() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx
        .parse("-pi/2*x^(-3) + x^(-3)*atan(cbrt(-x^(-3)))")
        .unwrap();
    let t0 = Instant::now();
    let s = f.try_series_dir(&x, &ctx.int(0), 4, R).unwrap();
    let dt = t0.elapsed();
    assert!(dt < Duration::from_secs(10), "series took {dt:?}");
    let want = ctx
        .parse("1/5*(-1)^(-2/3)*x^2 - (-1)^(-1/3)*x^(-2) - 1/3")
        .unwrap()
        .eval();
    assert_eq!(s, want);
    let g = ctx.parse("atan(cbrt(-x^(-3)))").unwrap();
    let s = g.try_series_dir(&x, &ctx.int(0), 4, R).unwrap();
    let want = ctx.parse("pi/2 - (-1)^(-1/3)*x - x^3/3").unwrap().eval();
    assert_eq!(s, want);
}

/// Before: `∞` (wrong; refused for the last one).  `x·e^{ln Γ(x) − ln Γ(x +
/// 1)}` is rewritten as `ω⁻¹·e^{h}` with `h = ln Γ(1/ω) − ln Γ(1/ω + 1) →
/// −∞` like `ln ω`: the leading coefficient of `e^h` was taken to be
/// `e^{c}` for the leading coefficient `c = −x` of `h`, a power of `ω` in
/// disguise (`e^{ln ω} = ω`), and the leading exponent came out `−1`.  The
/// same exponential arises from `exp(−ln(1/ω) + …)` once `exp(k·ln z) =
/// z^k` is folded for a symbolic `z`: `Γ(x + 1)/(x·Γ(x))` at `∞` was `0`.
/// And the series engine, given the expression `ln ω = −x` for its `ln ω`,
/// found it only where it stood literally (`Γ(x + 1)/Γ(x) − x` came out
/// `−∞` with that fold).  The direction of `x^{1/3}` at `−∞` (`e^{iπ/3}`),
/// whose signs were not decided, made the last one a refusal.
///
/// Oracle: SymPy `limit(x*exp(loggamma(x) - loggamma(x + 1)), x, oo)` → `1`;
/// mpmath at `x = 10¹⁰, 10²⁰`: `1.0000000000…`, `2.4·10⁻³⁰` (the second),
/// `1.0000000001` (the third), and `x·(π/2 − atan ∛x)` at `−10³⁰`:
/// `−5.0·10¹⁹ + 8.66·10¹⁹·i` (`|x|^{2/3}·e^{2iπ/3}`).  The unchanged ones:
/// `Γ(x + 1)/(x·Γ(x)) = 1`, `Γ(x + 1)/Γ(x) − x = 0`, `x(Γ(x + ½)/(Γ(x)√x)
/// − 1) = −1/8` (SymPy), and `(π/2 − atan x²)/(e^{1/ln x} − 1) → 0`
/// (mpmath `4.2·10⁻³⁹` at `10²⁰`), whose `e^{1/ln x}` is a coefficient of
/// lower order (`e^{−1/ln ω}`).
#[test]
fn exponentials_of_logarithmic_growth_in_gruntz() {
    let ctx = Context::new();
    let bound = Duration::from_secs(20);
    same_within(
        &ctx,
        "x*exp(loggamma(x) - loggamma(x + 1))",
        "oo",
        B,
        "1",
        None,
        bound,
    );
    same_within(
        &ctx,
        "x*exp(loggamma(x) - loggamma(x + 1)) - 1",
        "oo",
        B,
        "0",
        None,
        bound,
    );
    same_within(
        &ctx,
        "x*exp(loggamma(x) - loggamma(x + 1) + 1/x)",
        "oo",
        B,
        "1",
        None,
        bound,
    );
    same_within(
        &ctx,
        "x*(pi/2 - atan(x^(1/3)))",
        "-oo",
        B,
        "-(-1)^(-1/3)*oo",
        None,
        bound,
    );
    same_within(&ctx, "gamma(x + 1)/(x*gamma(x))", "oo", B, "1", None, bound);
    same_within(&ctx, "gamma(x + 1)/gamma(x) - x", "oo", B, "0", None, bound);
    same_within(
        &ctx,
        "x*(gamma(x + 1/2)/(gamma(x)*sqrt(x)) - 1)",
        "oo",
        B,
        "-1/8",
        None,
        bound,
    );
    same_within(
        &ctx,
        "(pi/2 - atan(x^2))/(exp(1/log(x)) - 1)",
        "oo",
        B,
        "0",
        None,
        bound,
    );
}

/// Gruntz's last resort expands `sign(asin(asin(1 − u²)))` (the series
/// at `x = 1⁻` of the limit engine's two-sided test) to order 20 in `u`:
/// the compositions built coefficients such as `π⁸⁹·(1 − π²/4)^(−179/2)`,
/// 238,000 arena nodes, each coefficient tested for a hidden zero, for 21
/// s before the expansion failed; `series(sign(asin(asin(x))), x, 1, 3)`
/// took 22 s to stay unevaluated (`abs` the same), since 0.43 — the CI
/// Fuzz workflow's `fuzz_calculus` timed out on it (34 s against its 30 s
/// limit) during the 0.47 release.  A last-resort expansion now stops
/// adding terms after 20,000 nodes (the ones that find a leading term take
/// tens): 0.7 s and 1.8 s, the same answers.
///
/// There is no expansion in powers of `x − 1`: near 1, `asin x` is `π/2`
/// plus a real (left) or an imaginary (right) multiple of `√|x − 1|`, so the
/// value has a `√|x − 1|` term, a different one on each side (mpmath, 30
/// and 50 digits: `c = sign(asin(π/2)) = 0.837904562907634 −
/// 0.545816767293354i`, and `(sign(asin(asin(1 ∓ h))) − c)/√h` at `h =
/// 10⁻⁸` is `0.2848 + 0.4373i` on the left, `−0.1856 − 0.2849i` on the
/// right).
#[test]
fn last_resort_expansions_are_bounded() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let one = ctx.int(1);
    for src in ["sign(asin(asin(x)))", "abs(asin(asin(x)))"] {
        let f = ctx.parse(src).unwrap();
        let t0 = Instant::now();
        let s = f.try_series(&x, &one, 3);
        let dt = t0.elapsed();
        assert!(dt < Duration::from_secs(60), "series of {src} took {dt:?}");
        assert!(
            s.is_err(),
            "series of {src} at 1 = {:?}",
            s.map(|e| e.to_string())
        );
    }
}
