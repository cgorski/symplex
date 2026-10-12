//! Series of complex-valued functions: the side of a branch cut decided by
//! complex coefficients and exponents, imaginary parts beyond the working
//! precision, parameters that may lie on a cut, logarithmic constant terms
//! with an imaginary coefficient, and expansions that were refused (the
//! series hunt after 0.43).  Every reference was checked against mpmath
//! 1.3: the residual `f − S` at `x = a ± 10⁻ᵏ` (`x = ±10ᵏ` at infinity),
//! `k = 4, 6, 8, 10`, at `50 + 12n + 3k` digits must decay like the next
//! order (SymPy 1.14 `series` as a second opinion where it answers).

use symplex::prelude::*;

const B: Direction = Direction::Both;
const R: Direction = Direction::Right;
const L: Direction = Direction::Left;

fn ser(ctx: &Context, src: &str, point: &str, order: u32, dir: Direction) -> Ex {
    let f = ctx
        .parse(src)
        .unwrap_or_else(|e| panic!("`{src}` does not parse: {e}"));
    let x = ctx.symbol("x");
    let p = ctx.parse(point).unwrap();
    f.series_dir(&x, &p, order, dir)
}

fn text(ctx: &Context, src: &str, point: &str, order: u32, dir: Direction) -> String {
    let s = ser(ctx, src, point, order, dir);
    assert!(
        !s.has_unevaluated(),
        "series of `{src}` at {point} {dir:?} was refused"
    );
    s.to_string()
}

fn refused(ctx: &Context, src: &str, point: &str, order: u32, dir: Direction) -> bool {
    ser(ctx, src, point, order, dir).has_unevaluated()
}

/// `|f(x₁) − S(x₁)|` at the rational point `x₁ = num/den` (symplex's own
/// evaluation at a point off every cut).
fn residual(ctx: &Context, src: &str, s: &Ex, num: i64, den: i64) -> f64 {
    let x = ctx.symbol("x");
    let x1 = ctx.rational(num, den);
    let f = ctx.parse(src).unwrap().subs(&x, &x1).eval_complex64();
    let v = s.subs(&x, &x1).eval_complex64();
    let (f, v) = (f.unwrap(), v.unwrap());
    (f - v).norm()
}

/// Before: a power with a non-numeric exponent (`u^i`, `u^(1 + i)`, `u^π`)
/// of a base tending to the negative reals with a non-real correction had
/// the expansion of the principal value from above on both sides, and a
/// two-sided expansion where the sides differ: `(−2 − 3i·x)^i` was
/// `(−2)^i + 3x·(−2)^(−1+i) + …` for `x → 0⁺` too (it is `e^{2π}` times
/// that there: `−2 − 3i·x` is below the cut).
///
/// Oracle: mpmath `(−2 − 3i·10⁻⁸)^i = 23.14…·e^{i·ln 2}` (`e^{π}·…`), the
/// residual of the series from the right decays like `x³`; from the left
/// the principal value `(−2)^i = e^{−π}·e^{i ln 2}`.
#[test]
fn non_numeric_exponent_takes_the_side_of_the_cut() {
    let ctx = Context::new();
    let src = "(-2 + (-3*I)*x)^I";
    let right = ser(&ctx, src, "0", 3, R);
    assert!(right.to_string().contains("exp(2*pi)"), "{right}");
    assert!(residual(&ctx, src, &right, 1, 1000) < 1e-5);
    let left = ser(&ctx, src, "0", 3, L);
    assert!(!left.to_string().contains("exp(2*pi)"), "{left}");
    assert!(residual(&ctx, src, &left, -1, 1000) < 1e-9);
    assert!(refused(&ctx, src, "0", 3, B));
    // Even power of x: below the cut from both sides.
    assert_eq!(
        text(&ctx, "(-2 + (-3*I)*x^2)^(1 + I)", "0", 3, B),
        "-3*(-2)^I*x^2*(1 + I)*exp(2*pi)*I + (-2)^(1 + I)*exp(2*pi)"
    );
    // A real non-rational exponent: (−1 − i·0)^π = e^{−iπ²}.
    let s = ser(&ctx, "(-1 - I*x)^pi", "0", 2, R);
    assert!(residual(&ctx, "(-1 - I*x)^pi", &s, 1, 1000) < 1e-5, "{s}");
}

/// Before: an argument on a cut whose visible coefficients were all real
/// took the principal value there (the limit from above) even when a term
/// beyond the working precision moved it below: `ln(−1 − i·x⁶)` was `iπ`,
/// `√(−1 − i·x⁶)` was `i`, `ln(−x − i·x⁷)` for `x → 0⁺` was `iπ + ln x`.
/// The precision is now widened until the side shows (or the argument is
/// shown real).  The same defect gave wrong answers through the Puiseux
/// route (which expands at a lower precision): `Ei(x + 2i·x³)` for
/// `x → 0⁻` was `γ + ln(−x)`.
///
/// Oracle: mpmath `log(−1 − 10⁻⁴⁸i) = −πi`, `sqrt(−1 − 10⁻⁴⁸i) = −i`,
/// `ei(−10⁻⁸ − 10⁻⁵⁶i) = −17.84… − πi` (`= γ + ln 10⁻⁸ − πi + …`),
/// `ei(−10⁻⁸ − 2·10⁻²⁴i) − ei(−10⁻⁸) = −πi`, `acosh(−1 − 10⁻⁸ − 10⁻⁵⁶i) =
/// 1.414…·10⁻⁴ − πi`; the residuals decay like the next order.
#[test]
fn imaginary_part_beyond_the_precision_decides_the_side() {
    let ctx = Context::new();
    assert_eq!(text(&ctx, "log(-1 - I*x^6)", "0", 2, B), "-pi*I");
    assert_eq!(text(&ctx, "sqrt(-1 - I*x^6)", "0", 2, B), "-I");
    assert_eq!(text(&ctx, "log(-x - I*x^7)", "0", 2, R), "-pi*I + ln(x)");
    assert_eq!(text(&ctx, "(-x - I*x^7)^(1/2)", "0", 2, R), "-sqrt(x)*I");
    assert_eq!(
        text(&ctx, "Ei(-x - I*x^7)", "0", 2, R),
        "-x - pi*I + ln(x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "Ci(-x - I*x^7)", "0", 2, R),
        "-pi*I + ln(x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "Ei(x + 2*I*x^3)", "0", 1, L),
        "-pi*I + ln(-x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "acosh(-1 - x - I*x^7)", "0", 1, R),
        "sqrt(2)*sqrt(x) - pi*I"
    );
    assert_eq!(text(&ctx, "log(-1 - I/x^7)", "oo", 1, B), "-pi*I");
    // A visible imaginary part above the cut keeps the principal value.
    assert_eq!(text(&ctx, "log(-1 + I*x^6)", "0", 2, B), "pi*I");
}

/// Before: a constant term with a parameter declared real (or positive)
/// counted as off the cut even where it may lie on it, so the side of a
/// non-real correction was ignored: `asin(a − (x − 1) + i(x − 1)²)` (`a >
/// 0`) was `asin a + …`, wrong for `a > 1`, and `ln(a + i·x)` (`a` real)
/// was `ln a + i·x/a + …` from both sides, wrong for `a < 0`.  Undecided,
/// they are refused; a constant shown to be on the cut takes the side.
///
/// Oracle: mpmath `asin(3.4 + 10⁻¹⁶i) = π − asin(3.4)` (SymPy's value of
/// `asin` on the cut is the limit from below), `log(−1.3 ± 10⁻⁸i)` differ
/// by `2πi`.
#[test]
fn parameter_that_may_lie_on_a_cut() {
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Positive]).unwrap();
    assert!(refused(&ctx, "asin(a - (x - 1) + I*(x - 1)^2)", "1", 1, B));
    // Inside a product the refusal is definite too: the differentiation
    // fallback of the product took the derivatives along the cut
    // (`Ei(3)·asin(a)` before).
    assert!(refused(
        &ctx,
        "Ei(3 + (x - 1)^2/2)*asin(a - (x - 1) + I*(x - 1)^2)",
        "1",
        1,
        B
    ));
    assert_eq!(
        text(&ctx, "asin(2 + a - (x - 1) + I*(x - 1)^2)", "1", 1, B),
        "-asin(a + 2) + pi"
    );
    let ctx = Context::new();
    ctx.symbol_with("a", &[Assumption::Real]).unwrap();
    assert!(refused(&ctx, "log(a + I*x)", "0", 2, B));
    assert!(refused(&ctx, "sqrt(a + I*x^2)", "0", 2, B));
}

/// Before: `cos` and `sin` of an argument whose constant term has an
/// imaginary multiple of `ln x` kept the coefficients `cosh(ln x)`,
/// `sinh(ln x)`, which grow like `1/x`: `cos(i·ln x + x)` at order 3 was
/// `cosh(ln x) − i·x·sinh(ln x) − x²·cosh(ln x)/2`, whose omitted terms
/// `x³·sinh(ln x)/6 ≈ x²/12` are not `O(x³)`.  Now through `e^{±iu}`:
/// `cos(i·ln x + x) = (e^{ix}/x + x·e^{−ix})/2`.
///
/// Oracle: mpmath residual at `x = 10⁻ᵏ` decays like `x³`; exact identity.
#[test]
fn trigonometric_function_of_an_imaginary_logarithm() {
    let ctx = Context::new();
    assert_eq!(
        text(&ctx, "cos(I*log(x) + x)", "0", 3, R),
        "-7/12*x^2*I + 1/(2*x) + 1/4*x + 1/2*I"
    );
    let s = ser(&ctx, "sin(2*I*log(x) + x)", "0", 3, R);
    assert!(
        residual(&ctx, "sin(2*I*log(x) + x)", &s, 1, 100) < 1e-4,
        "{s}"
    );
}

/// Before: `ln(c·x^v·(1 + w))` was refused unless `c` was real of known
/// sign (and `w` real for `c < 0`): `ln(i·x)`, `ln(x + i)` at `−∞`,
/// `ln(−x + i·x²)` for `x → 0⁺`, and with it `Ei(i·x)`, `Ci(i·x)`,
/// `ln Γ(i·x)`, `ln Γ(−x)` for `x → 0⁺` (while `ln Γ(x)` for `x → 0⁻`
/// worked), `ψ(i·x)`, `ln Γ(i·x)` at `+∞`.
///
/// Oracle: mpmath, e.g. `log(10⁻⁸i) = −18.42 + πi/2`, `log(−10⁸ + i) =
/// 18.42 + πi − 10⁻⁸i`, `ei(10⁻⁸i) = γ + ln 10⁻⁸ + πi/2 + 10⁻⁸i`,
/// `loggamma(−10⁻⁸) = 18.42 − πi + …`, `digamma(10⁸i) = 18.42 + πi/2 +
/// 5·10⁻⁹i`; SymPy agrees where it answers (`series(log(I*x), x, 0)` =
/// `log(x) + I*pi/2`).
#[test]
fn logarithm_of_a_complex_leading_term() {
    let ctx = Context::new();
    assert_eq!(text(&ctx, "log(I*x)", "0", 3, R), "1/2*pi*I + ln(x)");
    assert_eq!(text(&ctx, "log(I*x)", "0", 3, L), "-1/2*pi*I + ln(-x)");
    assert_eq!(
        text(&ctx, "log(-x + I*x^2)", "0", 3, R),
        "1/2*x^2 - x*I + pi*I + ln(x)"
    );
    assert_eq!(
        text(&ctx, "log(x + I)", "-oo", 3, B),
        "1/2*x^(-2) + I/x + pi*I + ln(-x)"
    );
    assert_eq!(
        text(&ctx, "Ei(I*x)", "0", 3, R),
        "-1/4*x^2 + x*I + 1/2*pi*I + ln(x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "Ci(I*x)", "0", 3, R),
        "1/4*x^2 + 1/2*pi*I + ln(x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "loggamma(-x)", "0", 3, R),
        "1/12*x^2*pi^2 + x*EulerGamma - ln(x) - pi*I"
    );
    assert_eq!(
        text(&ctx, "digamma(I*x)", "oo", 3, B),
        "1/12*x^(-2) + I/(2*x) + 1/2*pi*I + ln(x)"
    );
    let s = ser(&ctx, "loggamma(I*x)", "oo", 3, B);
    assert!(residual(&ctx, "loggamma(I*x)", &s, 1000, 1) < 1e-8, "{s}");
}

/// Before: `asinh`, `acosh`, `asin`, `acos` of an argument tending to
/// infinity had no expansion (`asinh x` at `+∞` was refused); nor had a
/// power with a fractional exponent at infinity (`√(x + 1)`, `∛x`).
///
/// Oracle: SymPy `series(asinh(x), x, oo, 4)` = `log(2) + log(x) +
/// 1/(4x²) + O(x⁻⁴)`, `series(sqrt(x + 1), x, oo, 3)`; mpmath residuals at
/// `x = ±10ᵏ` (`asin(10⁸) = π/2 − 19.11…i`, `acosh(−10⁸) = 19.11… + πi`,
/// `sqrt(−10⁸ + 1) = 9999.99995i`).
#[test]
fn inverse_functions_and_fractional_powers_at_infinity() {
    let ctx = Context::new();
    assert_eq!(
        text(&ctx, "asinh(x)", "oo", 4, B),
        "1/4*x^(-2) + ln(2) + ln(x)"
    );
    assert_eq!(
        text(&ctx, "asinh(x)", "-oo", 4, B),
        "-1/4*x^(-2) - ln(2) - ln(-x)"
    );
    assert_eq!(
        text(&ctx, "acosh(x)", "-oo", 4, B),
        "-1/4*x^(-2) + pi*I + ln(2) + ln(-x)"
    );
    assert_eq!(
        text(&ctx, "asin(x)", "oo", 4, B),
        "1/4*x^(-2)*I - (ln(2) + ln(x))*I + 1/2*pi"
    );
    assert_eq!(
        text(&ctx, "acos(x + I)", "oo", 3, B),
        "1/x - 1/4*x^(-2)*I - (ln(2) + ln(x))*I"
    );
    assert_eq!(
        text(&ctx, "sqrt(x + 1)", "oo", 3, B),
        "sqrt(x) - 1/8*x^(-3/2) + 1/16*x^(-5/2) + 1/(2*sqrt(x))"
    );
    let s = ser(&ctx, "sqrt(x + 1)", "-oo", 3, B);
    assert!(residual(&ctx, "sqrt(x + 1)", &s, -1000, 1) < 1e-8, "{s}");
    assert_eq!(text(&ctx, "x^(1/3)", "oo", 3, B), "cbrt(x)");
    // Complex direction: atanh u = atanh(1/u) + iπ/2 above the real axis.
    assert_eq!(
        text(&ctx, "atanh((3 + I)*x^2 - 3)", "oo", 4, B),
        "x^(-2)/(3 + I) + 1/2*pi*I"
    );
    // acot u = atan(1/u) → −π/2 for 1/u → ∞ with Re < 0 (mpmath
    // `acot((−1 − i)·10⁻⁸) = −1.5707…`); a branch point reached at infinity.
    assert_eq!(
        text(&ctx, "acot((-1 - I)/x + 1/(2*x^2))", "oo", 1, B),
        "-1/2*pi"
    );
    let s = text(&ctx, "acosh(1 + 3/x)", "oo", 3, B);
    assert!(s.contains("sqrt(6)/sqrt(x)"), "{s}");
}

/// Before: a function of `ζ` at a regular point was refused at every order
/// (its derivatives have no closed form), even where only the value is
/// needed.  Now the expansion is exact below the first derivative.
///
/// Oracle: mpmath `zeta(−2 − i + 2·10⁻⁸)·atan(−1 − 10⁻⁸ + …)` →
/// `−π/4·zeta(−2 − i)`, residual `O(x)`.
#[test]
fn zeta_at_a_regular_point_to_low_order() {
    let ctx = Context::new();
    let src = "zeta((-2 - I) + 2*(x - 2))*atan(-1 - (x - 2) + 2*(x - 2)^2)";
    assert_eq!(text(&ctx, src, "2", 1, R), "-1/4*zeta(-I - 2)*pi");
    // Beyond the value it stays refused (no closed form for ζ′).
    assert!(refused(&ctx, src, "2", 2, R));
}

/// Before: `W` at its branch point `−1/e` was refused from both sides.
/// Now the Puiseux series `W₀ = −1 + p − p²/3 + 11p³/72 − …`, `p =
/// √(2(1 + e·x))` (Corless et al. 1996, eq. 4.22), with the principal root:
/// for `x < −1/e` the value from above (`Im W₀ > 0`).
///
/// Oracle: mpmath `lambertw(−1/e + 10⁻⁸) = −1 + 2.3316…·10⁻⁴ − …`,
/// `lambertw(−1/e − 10⁻⁸) = −1 + 2.3316…·10⁻⁴i + …`; residuals decay like
/// the next order.
#[test]
fn lambert_w_at_its_branch_point() {
    let ctx = Context::new();
    let right = ser(&ctx, "LambertW(x)", "-exp(-1)", 3, R);
    assert!(!right.has_unevaluated());
    assert!(
        right.to_string().contains("sqrt(2*E)*sqrt(x + exp(-1))"),
        "{right}"
    );
    let left = ser(&ctx, "LambertW(x)", "-exp(-1)", 2, L);
    assert!(
        left.to_string().contains("sqrt(-2*E)*sqrt(-x - exp(-1))"),
        "{left}"
    );
    assert!(refused(&ctx, "LambertW(x)", "-exp(-1)", 2, B));
    // On the cut (−∞, −1/e) `W₀` is the limit from above: an argument
    // leaving the cut upwards keeps it, downwards there is no form.
    assert!(!refused(&ctx, "LambertW(-3 + exp(I)*x^3 + x)", "0", 3, R));
    assert!(refused(&ctx, "LambertW(-3 + exp(I)*x^3 + x)", "0", 3, L));
}

/// Before: `acosh(−1 + s)` with a non-real `s` was refused (and one whose
/// imaginary part lay beyond the precision took the value from above).
/// Near `−1` the whole real line is the cut of `acosh`: above it the form
/// `iπ − i√(2s)·F(s/2)`, below its negative.
///
/// Oracle: mpmath `acosh(−1 − 10⁻⁸ ± 10⁻¹⁶i) = 1.414…·10⁻⁴ ± πi`.
#[test]
fn acosh_beside_minus_one() {
    let ctx = Context::new();
    assert_eq!(
        text(&ctx, "acosh(-1 - x + I*x^2)", "0", 1, R),
        "sqrt(2)*sqrt(x) + pi*I"
    );
    assert_eq!(
        text(&ctx, "acosh(-1 - x - I*x^2)", "0", 1, R),
        "sqrt(2)*sqrt(x) - pi*I"
    );
}

/// Before: `exp(ψ(−2 + √2·x²))` at `0` and `tanh(ln Γ(x))` for `x → 0⁻`
/// went to the differentiation fallback, whose limits of ever larger
/// derivatives did not return in 10 s, and `polylog(2, 2i·x + …)` likewise.
/// `e^a` with `Re a → −∞` is `0` to every order; `tanh u = 1 − 2/(e^{2u} +
/// 1)` for `u → +∞` logarithmically; `Li₂(u) = Σ uᵏ/k²`.
///
/// Oracle: `ψ(−2 + w) = −1/w + …`, so `exp(ψ) = O(e^{−1/(√2x²)})`; mpmath
/// `tanh(loggamma(−10⁻⁴)) = 1 − 2·10⁻⁸ + …`.
#[test]
fn slow_fallbacks_are_avoided() {
    let ctx = Context::new();
    assert_eq!(text(&ctx, "exp(digamma(-2 + sqrt(2)*x^2))", "0", 1, B), "0");
    assert_eq!(text(&ctx, "tanh(loggamma(x))", "0", 3, L), "-2*x^2 + 1");
    assert_eq!(
        text(&ctx, "polylog(2, x)", "0", 4, B),
        "1/9*x^3 + 1/4*x^2 + x"
    );
    assert_eq!(
        text(
            &ctx,
            "polylog(2, 2*I*(x - 1/2) + (1/2 - 1/3*I)*(x - 1/2)^2)",
            "1/2",
            2,
            L
        ),
        "2*(x - 1/2)*I"
    );
}

/// Before: special functions with a cut whose argument leaves it had no
/// expansion, or a wrong one through the Puiseux route (a lower precision,
/// see above): `expint(1, −1 − x − i·x²)` was `expint(1, −1)` for `x →
/// 0⁺`, `polylog(2, 2 + x/2 + (1 + i)x²)` was `polylog(2, 2)`,
/// `bessely(0, −1 + x + 2i·x³)` the expansion from above for `x → 0⁻`.
/// Now the jump across the cut where it is a constant form (`E₁` is `2πi`
/// more below the negative reals, `li` is `±iπ` off `(0, 1)`), the
/// expansion on the side the principal value is continuous with (`Yν`,
/// `Kν`, `Eₙ` above, `Liₛ`, `K(m)`, `E(m)` below), and a refusal on the
/// other side.
///
/// Oracle: mpmath `expint(1, −1 ∓ 10⁻¹⁶i) = −1.895… ± πi`,
/// `polylog(2, 3 − 10⁻¹⁵i) = polylog(2, 3)` (`2.32… − 3.45…i`), `+10⁻¹⁵i`
/// its conjugate; `bessely(0, −1 + 10⁻¹⁵i) = bessely(0, −1)`;
/// `li(1 − 10⁻⁴ − 3·10⁻⁸i)` against the series at `x = −10⁻⁴`.
#[test]
fn special_functions_beside_their_cuts() {
    let ctx = Context::new();
    assert_eq!(
        text(&ctx, "expint(1, -1 + I*x^2)", "0", 1, R),
        "expint(1, -1)"
    );
    assert_eq!(
        text(&ctx, "expint(1, -1 - I*x^2)", "0", 1, R),
        "2*pi*I + expint(1, -1)"
    );
    assert_eq!(
        text(&ctx, "li(1 - x - 3*I*x^2)", "0", 2, L),
        "x*(3*I - 1/2) + ln(-x) + EulerGamma"
    );
    assert_eq!(
        text(&ctx, "li(1/2 - x - I*x^2)", "0", 2, R),
        "-x/ln(1/2) - pi*I + li(1/2)"
    );
    assert_eq!(
        text(&ctx, "polylog(2, 2 + x/2 - (1 + I)*x^2)", "0", 1, R),
        "polylog(2, 2)"
    );
    assert!(refused(
        &ctx,
        "polylog(2, 2 + x/2 + (1 + I)*x^2)",
        "0",
        1,
        R
    ));
    assert!(!refused(&ctx, "bessely(0, -1 + x + 2*I*x^3)", "0", 3, R));
    assert!(refused(&ctx, "bessely(0, -1 + x + 2*I*x^3)", "0", 3, L));
}
