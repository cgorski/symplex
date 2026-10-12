//! Limits, second round (the hunt after the 0.44 series work): leading
//! terms that cancel in the Gruntz algorithm (fractional powers of the
//! comparison variable, inverse functions at infinity), complex leading
//! coefficients (directed infinities), and special functions at their
//! poles and branch points approached from one side.  Every reference
//! value cites its oracle: mpmath 1.3 at 60+ digits along the path
//! `x = ±10ᵏ` (`k = 10, 20, 40, 80`) or `a ± 10⁻ᵏ`, SymPy 1.14 as a second
//! opinion.  Infinite values are written as `subs` writes values at
//! infinities: `oo + I·b`, `I·oo + a`, and `c·oo` for any other direction
//! `c/|c|` (which claims nothing about a finite part).

use symplex::prelude::*;

fn lim(ctx: &Context, src: &str, point: &str, dir: Direction) -> Result<Ex, SymplexError> {
    let f = ctx
        .parse(src)
        .unwrap_or_else(|e| panic!("`{src}` does not parse: {e}"));
    let x = ctx.symbol("x");
    let p = ctx.parse(point).unwrap();
    f.try_limit_dir(&x, &p, dir)
}

fn same(ctx: &Context, src: &str, point: &str, dir: Direction, want: &str) {
    let got = lim(ctx, src, point, dir)
        .unwrap_or_else(|e| panic!("limit of `{src}` at {point} {dir:?} failed: {e}"));
    let want_ex = ctx.parse(want).unwrap().eval();
    assert_eq!(
        got, want_ex,
        "`{src}` at {point} {dir:?}: {got} (want {want})"
    );
}

const B: Direction = Direction::Both;
const R: Direction = Direction::Right;
const L: Direction = Direction::Left;

/// Before: refused with "leading coefficients cancel and series expansion
/// failed" — the leading term of a Puiseux series in Gruntz's `ω`
/// (`(e^{ω^{1/3}} − 1)/ω`) and of `atan(1/ω)`, `acot`, `asinh` at infinity
/// was out of reach of the truncated Laurent engine.  These blocked 155
/// convergence decisions of the limit-comparison test in the summation
/// hunt.  Now the leading term is found by the expansions of a series,
/// with `ω = u^q` for fractional powers.
///
/// Oracle: SymPy `limit(k*(exp(k**(-1/3)) - 1), k, oo)` → `oo` (mpmath
/// `2.15·10⁵³` at `10⁸⁰`, growing like `x^{2/3}`), `limit(k*(pi/2 -
/// atan(k)), k, oo)` → `1`, `x^{2/3}(e^{x^{−1/3}} − 1) − x^{1/3}` → `1/2`
/// (mpmath 0.5000000359 at `10²⁰`), `x²(asinh x − ln 2x)` → `1/4`,
/// `x·acot x` → `1` (mpmath 1.0 at `10¹⁰`…`10⁸⁰`).
#[test]
fn cancelling_leading_terms_of_puiseux_and_inverse_function_expansions() {
    let ctx = Context::new();
    same(&ctx, "x*(exp(x^(-1/3)) - 1)", "oo", B, "oo");
    same(&ctx, "x*(pi/2 - atan(x))", "oo", B, "1");
    same(
        &ctx,
        "x^(2/3)*(exp(x^(-1/3)) - 1) - x^(1/3)",
        "oo",
        B,
        "1/2",
    );
    same(&ctx, "x^2*(asinh(x) - log(2*x))", "oo", B, "1/4");
    same(&ctx, "x*acot(x)", "oo", B, "1");
}

/// Before: refused ("leading coefficients cancel"; "pole detected at
/// expansion point") — fractional powers with a common denominator above 6,
/// and the half-integer powers of `acos`, `asin`, `W` at their branch
/// points, whose expansions in `ω` are Puiseux series.
///
/// Oracle: mpmath `10²⁰^{1/3}(√(1 + 10^{−4}) − 1)` = 232.07 (growing),
/// `(W(−1/e + 10⁻⁴⁰) + 1)/10⁻²⁰` = 2.3316439816 = `√(2e)`,
/// `(acos(1 − ε) − √(2ε))/ε^{3/2}` = 0.117851130198 = `√2/12`.  SymPy:
/// `oo`, `oo` (wrong for `W`), `sqrt(2)/12`.
#[test]
fn ramified_leading_terms_at_infinity_and_branch_points() {
    let ctx = Context::new();
    same(&ctx, "x^(1/3)*(sqrt(1 + x^(-1/5)) - 1)", "oo", B, "oo");
    same(
        &ctx,
        "(lambertw(-exp(-1) + x) + 1)/sqrt(x)",
        "0",
        R,
        "sqrt(2*E)",
    );
    same(
        &ctx,
        "(acos(1 - x) - sqrt(2*x))/x^(3/2)",
        "0",
        R,
        "sqrt(2)/12",
    );
}

/// Before: refused with "cannot determine the sign of 1 + I": Gruntz had no
/// directed result for a complex leading coefficient.  Now `f ≈ c₀·ω^{e₀}`
/// with `e₀ < 0` tends to `c₀·oo` (SymPy's `limitinf`: `sign(c₀)·oo`), on
/// the imaginary axis `±I·oo` with its real part completed.  An argument
/// that turns (`e^{(1+i)x}`) still has no limit.
///
/// Oracle: mpmath `10⁴⁰(1 + i)`; `e^{x − i} − eˣ` direction
/// `−0.479 − 0.878i` = `(e^{−i} − 1)/|e^{−i} − 1|`; `e^{x+i}/x` direction
/// `0.540 + 0.841i` = `e^{i}`; `2ix + e^{−x}`: real part `→ 0`.  SymPy:
/// `oo*sign(1 + I)`, `oo*sign(1 - exp(I))`, `oo` (it drops the phase).
#[test]
fn complex_leading_coefficients_give_directed_infinities() {
    let ctx = Context::new();
    same(&ctx, "x + I*x", "oo", B, "(1 + I)*oo");
    same(&ctx, "x + I*x", "-oo", B, "(-1 - I)*oo");
    same(&ctx, "exp(x - I) - exp(x)", "oo", B, "(exp(-I) - 1)*oo");
    same(&ctx, "exp(x + I)/x", "oo", B, "exp(I)*oo");
    same(&ctx, "2*I*x + exp(-x)", "oo", B, "I*oo");
    same(&ctx, "(1 + I)/x", "0", R, "(1 + I)*oo");
    assert!(lim(&ctx, "exp((1 + I)*x)", "oo", B).is_err());
    // At `−∞` a fractional power turns the direction: mpmath direction
    // `0.5 + 0.866i` = `−(−1)^{4/3}` (SymPy: `oo*(-1)**(1/3)`).
    same(&ctx, "x^(4/3)*asin(1/x)", "-oo", B, "-(-1)^(4/3)*oo");
}

/// Before: `ln Γ(x)` at `0⁻` and `ln Γ(x − 1)` at `0±` were refused ("pole
/// at the limit of its argument"), `li(1 + x)` at `0⁻` too (its imaginary
/// part was not determined), and `E₁(x)` at `0⁻` gave the pole value
/// `expint(1, 0)`.  The one-sided expansions of the series engine are now
/// Gruntz's last resort, `ln Γ` and `E₁` have expansions at their poles,
/// and `li`, `ln Γ`, `Ei`, … of real arguments count as real where they are.
///
/// Oracle: mpmath `loggamma(−10⁻⁴⁰)` = `92.10 − 3.14159i`,
/// `loggamma(−1 + 10⁻⁴⁰)` = `92.10 − 3.14159i`, `loggamma(−1 − 10⁻⁴⁰)` =
/// `92.10 − 6.28319i`, `li(1 − 10⁻⁴⁰)` = `−91.53`, `e1(−10⁻⁴⁰)` = `91.53 −
/// 3.14159i`, `e1(10⁻⁴⁰) + ln 10⁻⁴⁰` = `−0.5772` (SymPy: `oo`, `oo`, `oo`,
/// `-oo`, `oo`, `-EulerGamma`; it drops the imaginary parts).
#[test]
fn special_functions_at_poles_from_one_side() {
    let ctx = Context::new();
    same(&ctx, "loggamma(x)", "0", L, "oo - I*pi");
    same(&ctx, "loggamma(x - 1)", "0", R, "oo - I*pi");
    same(&ctx, "loggamma(x - 1)", "0", L, "oo - 2*I*pi");
    same(&ctx, "li(1 + x)", "0", L, "-oo");
    same(&ctx, "expint(1, x)", "0", R, "oo");
    same(&ctx, "expint(1, x)", "0", L, "oo - I*pi");
    same(&ctx, "expint(1, x) + log(x)", "0", R, "-EulerGamma");
}

/// Before (wrong): `li(x)` at `0⁻` was `0` — `li 0 = 0` was taken for the
/// limit, but on its cut `(−∞, 0)` `li` takes the values from above and
/// `li(−ε) → iπ`; and `expint(1, x)` at `0⁺` was `expint(1, 0)`, a pole.
/// Now `li u = Ei(ln u)` at the branch point and `E₁(0)` is no limit value.
///
/// Oracle: mpmath `li(−10⁻⁴⁰)` = `1.1·10⁻⁴² + 3.14159i`, `li(10⁻⁴⁰)` → 0,
/// `e1(10⁻⁴⁰)` = `91.53` (SymPy: `0` for `li` from the left — it is real
/// only on the positive axis).
#[test]
fn branch_point_values_are_not_one_sided_limits() {
    let ctx = Context::new();
    same(&ctx, "li(x)", "0", L, "I*pi");
    same(&ctx, "li(-x^2)", "0", R, "I*pi");
    same(&ctx, "li(-1/x)", "oo", B, "I*pi");
    same(&ctx, "li(x)", "0", R, "0");
    same(&ctx, "expint(1, x)", "0", R, "oo");
}
