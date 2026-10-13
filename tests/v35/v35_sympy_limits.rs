//! Limits, series and sums from SymPy 1.14's own test suites
//! (`sympy/series/tests/test_limits.py`, `test_gruntz.py`, `test_series.py`,
//! `test_nseries.py`, `test_limitseq.py`, `concrete/tests/test_sums_products.py`)
//! that symplex answered wrongly or refused, plus the neighbours found
//! around them.  Every reference value cites its oracle: mpmath 1.3 along
//! `x = 10ᵏ` (`k = 5 … 40`, off-grid points `1.37·10ᵏ + 1/3`) or `a ± 10⁻ᵏ`
//! at `40 + 3k` digits, SymPy 1.14 as a second opinion.

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

/// The limit must be the finite number `want` (to 12 digits).
fn close(ctx: &Context, src: &str, point: &str, dir: Direction, want: f64) {
    let got = lim(ctx, src, point, dir)
        .unwrap_or_else(|e| panic!("limit of `{src}` at {point} {dir:?} failed: {e}"));
    let z = got
        .eval_complex64()
        .unwrap_or_else(|e| panic!("`{src}`: {got} does not evaluate: {e}"));
    assert!(
        (z.re - want).abs() <= 1e-12 * want.abs().max(1.0) && z.im.abs() <= 1e-12,
        "`{src}` at {point} {dir:?}: {got} = {z} (want {want})"
    );
}

const B: Direction = Direction::Both;
const R: Direction = Direction::Right;
const L: Direction = Direction::Left;

/// Before: `0` (then a `Piecewise` of special values of `a`): the constant
/// `1 − sin²a − cos²a` was taken for a non-zero denominator.  Now constants
/// that vanish identically in their parameters are recognised (two sample
/// points within the assumptions), and so is a `cos` of a hidden `−π/2`.
///
/// Oracle: SymPy test_issue_6364 `1/(cos(a)**2 - 1/2)` (`= 2/cos 2a`; mpmath
/// at `a = 1.3`: `−2.3340252653997290`, and `−2.33402777` for the
/// expression at `z = 10⁻²⁵`); test_issue_14411 `−oo` (mpmath
/// `−3.36·10³⁹` at `10⁻⁴⁰` right of the point).
#[test]
fn constants_that_vanish_identically() {
    let ctx = Context::new();
    let z = ctx.symbol("z");
    let f = ctx
        .parse("z/(1 - sqrt(1+z)*sin(a)^2 - sqrt(1-z)*cos(a)^2)")
        .unwrap();
    let got = f.try_limit_dir(&z, &ctx.int(0), R).unwrap();
    let a = ctx.symbol("a");
    let v = got
        .subs(&a, &ctx.parse("13/10").unwrap())
        .eval_f64()
        .unwrap();
    assert!((v - (-2.334_025_265_399_729)).abs() < 1e-9, "{got} = {v}");
    // `1 − sin²a − cos²a = 0`, so this is `x/x`: the limit is 1 (`same`
    // takes the limit in `x`).
    same(&ctx, "x/(1 - sin(a)^2 - cos(a)^2 + x)", "0", R, "1");
    same(&ctx, "3*sec(x/3 - 4*pi*x)", "3*pi/(24*pi - 2)", R, "-oo");
    same(&ctx, "3*sec(x/3 - 4*pi*x)", "3*pi/(24*pi - 2)", L, "oo");
}

/// Before: `∞` (test_issue_9471, the exponent ratio `ln 3/ln 27` taken for
/// irrational) and `0` (test_issue_22982, `e^{√e·x}` left in a leading
/// coefficient).  Oracle: SymPy `27`, `oo`; mpmath `27.000…` and
/// `2.09·10²⁶` at `10⁴⁰`.
#[test]
fn hidden_rationals_and_irrational_powers_of_omega() {
    let ctx = Context::new();
    close(&ctx, "27^(1 + log(x)/log(3))/x^3", "oo", B, 27.0);
    same(
        &ctx,
        "(log(E + 1/x) - 1)^(1 - sqrt(E + 1/x))",
        "oo",
        B,
        "oo",
    );
    same(&ctx, "(1/x)^(1 - sqrt(E + 1/x))", "oo", B, "oo");
}

/// Before: `exp(ψ(x)) − x` was `∞` at `∞` and the Gruntz-thesis limits
/// 8.11, 8.16, 8.17 were `∞` (exponentials of the class of ω split off
/// by the rewrite); 8.2–8.18 were refused (recursion depth 15).
/// Oracle: SymPy test_gruntz `-exp(2)`, `1`; `ψ(x) = ln x − 1/(2x) + …`
/// gives `−1/2` (mpmath `−0.49999…` at `10²⁰`).
#[test]
fn gruntz_rewrite_classes() {
    let ctx = Context::new();
    same(&ctx, "exp(digamma(x)) - x", "oo", B, "-1/2");
    same(&ctx, "exp(exp(digamma(log(x))))/x", "oo", B, "exp(-1/2)");
    same(
        &ctx,
        "(exp(x*exp(-x)/(exp(-x) + exp(-2*x^2/(x + 1)))) - exp(x))/x",
        "oo",
        B,
        "-exp(2)",
    );
    same(
        &ctx,
        "exp(x)*(exp(1/x + exp(-x) + exp(-x^2)) - exp(1/x - exp(-exp(x))))",
        "oo",
        B,
        "1",
    );
    same(
        &ctx,
        "(gamma(x + 1/gamma(x)) - gamma(x))/log(x)",
        "oo",
        B,
        "1",
    );
}

/// Before: `(−1/4)^x`, `(−1)^x/x²`, `(i/3)^x` refused ("cannot determine
/// the sign of πi + ln(1/4)"); `(−x)^x` and `(1 − x/2)^{3x}` were `∞` (the
/// phase turns: SymPy `zoo`, now refused).  Oracle: SymPy test_limitseq
/// `0`, test_issue_26513 `exp(-1)`; mpmath.
#[test]
fn turning_phases() {
    let ctx = Context::new();
    same(&ctx, "(-1/4)^x", "oo", B, "0");
    same(&ctx, "(-1)^x/x^2", "oo", B, "0");
    same(&ctx, "(I/3)^x*x^7", "oo", B, "0");
    same(&ctx, "((-1)^x + 2*x)/(x + 1)", "oo", B, "2");
    same(&ctx, "abs((-x/(1 + x))^x)", "oo", B, "exp(-1)");
    assert!(lim(&ctx, "(-x)^x", "oo", B).is_err());
    assert!(lim(&ctx, "(1 - x/2)^(3*x)", "oo", B).is_err());
}

/// Before: `x − ∞` at `∞` and `(1 + x)^∞` at `0±` were `∞` and `1`.  The
/// value is taken pointwise in the extended reals.  (SymPy leaves them
/// unevaluated.)
#[test]
fn infinite_constants_inside() {
    let ctx = Context::new();
    same(&ctx, "x - oo", "oo", B, "-oo");
    same(&ctx, "-oo + 1/(x + sin(x))", "0", R, "-oo");
    same(&ctx, "(1 + x)^oo", "0", R, "oo");
    same(&ctx, "(1 + x)^oo", "0", L, "0");
}

/// Before: refused.  `Ci(−x) = Ci(x) + iπ` on the cut (mpmath `ci(−10)` =
/// `−0.0455 + 3.14159i`), Fresnel and Bessel limits of SymPy's
/// test_issue_10102 and test_bessel_functions_at_infinity.
#[test]
fn special_functions_at_infinity() {
    let ctx = Context::new();
    same(&ctx, "Ci(x + I)", "oo", B, "0");
    same(&ctx, "Si(x + I)", "oo", B, "pi/2");
    same(&ctx, "Ci(-x)", "oo", B, "I*pi");
    same(&ctx, "fresnels(x)", "-oo", B, "-1/2");
    same(&ctx, "besselj(1, x)", "oo", B, "0");
    same(&ctx, "besselj(1, x)", "I*oo", B, "I*oo");
    same(
        &ctx,
        "fibonacci(x + 1)/fibonacci(x)",
        "oo",
        B,
        "GoldenRatio",
    );
    close(
        &ctx,
        "16^x/(x*binomial(2*x, x)^2)",
        "oo",
        B,
        std::f64::consts::PI,
    );
}

/// Before: 14 s (the imaginary part of `1/(acosh(2i + ix) − acosh(2i))`,
/// identically zero, searched by the Gruntz algorithm).
#[test]
fn acosh_on_the_imaginary_axis_is_fast() {
    let ctx = Context::new();
    let t = std::time::Instant::now();
    same(&ctx, "1/(acosh(2*I + I*x) - acosh(2*I))", "0", L, "-oo");
    assert!(t.elapsed().as_secs() < 5, "{:?}", t.elapsed());
}

/// Before: `0` for every series of order 0 (test_issue_14037; SymPy
/// `1/x + O(1)`), and `Σ_{x=0}^{b−1} (x − b)⁻²` was `zoo` for an integer
/// `b` (it is `H_b^{(2)} = π²/6 − ψ′(b + 1)`).
#[test]
fn series_principal_part_and_reflected_sum() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = ctx.parse("sin(x^50)/x^51").unwrap();
    let s = f.try_series_dir(&x, &ctx.int(0), 0, R).unwrap();
    assert_eq!(s, ctx.parse("1/x").unwrap().eval());
    let b = ctx
        .symbol_with("b", &[Assumption::Integer, Assumption::Positive])
        .unwrap();
    let term = ctx.parse("(x - b)^(-2)").unwrap();
    let upper = ctx.parse("b - 1").unwrap();
    let sum = term.try_summation(&x, &ctx.int(0), &upper).unwrap();
    let v = sum.subs(&b, &ctx.int(3)).eval_f64().unwrap();
    assert!((v - (1.0 + 0.25 + 1.0 / 9.0)).abs() < 1e-12, "{sum} = {v}");
}
