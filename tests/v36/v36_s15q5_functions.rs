//! Tests of session 15 (s15q5): special-function and integration leftovers.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn decimal(ctx: &Context, s: &str, digits: u32) -> String {
    parse(ctx, s)
        .eval_decimal(digits)
        .unwrap_or_else(|err| format!("{s}: {err:?}"))
}

// ── Item 1: elliptic integrals far off the axis ──

/// `E(m)` for a huge `m` next to the real axis: the parts lie 2,000 bits
/// apart, and the node's bound moved the rounded `10⁶⁰⁰` on both parts by
/// `|E′|·radius`, which swamped the small one; the route also stopped
/// `4·prec + 512` bits below the modulus.  Up to 0.58 all three were refused
/// ("elliptic_e: the value cancels beyond the working precision"), and with
/// only the evaluation fixed `E(10⁶⁰⁰ + i/10⁶⁰⁰)` printed `−1e300*i` (its
/// real part dropped: the radius of the imaginary part moved both parts).
/// mpmath 1.3 `ellipe(mpc(10**600, 1))` at 1,300 digits =
/// `1.28539816339744830961566084582e-300 - 1.0e+300j`, `ellipe(mpc(-10**600,
/// 1))` = `1.0e+300 - 5.0e-301j`, `ellipe(mpc(10**600, 10**-600))` (1,900
/// digits) = `7.8539816339744830961566084582e-301 - 1.0e+300j`.
#[test]
fn elliptic_e_far_off_the_axis_keeps_its_small_part() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "elliptic_e(10^600 + I)",
            "1.28539816339744830961566084582e-300 - 1e300*i",
        ),
        ("elliptic_e(-10^600 + I)", "1e300 - 5e-301*i"),
        (
            "elliptic_e(10^600 + I/10^600)",
            "7.8539816339744830961566084582e-301 - 1e300*i",
        ),
    ] {
        assert_eq!(decimal(&ctx, s, 30), want, "{s}");
    }
}

// ── Item 2: integrals of rational functions of complex powers `cˣ` ──

/// `(1+i)ˣ = 1 + i` at `x = 1` (`|(1+i)ˣ| = √2` there and the phase is
/// `π/4`): a simple pole, `(x − 1)·f → 1/ln(1 + i) ≠ 0`, so the integral
/// diverges.  Up to 0.58 the singular point came from the solver as
/// `(½ ln 2 + iπ/4)/(iπ/4 + ln √2)`, which the comparisons did not place:
/// refused ("the antiderivative is piecewise in the variable with conditions
/// it cannot be split at", "cannot decide whether the singular point … lies
/// inside").  Oracle: the exact value `(1+i)¹ − 1 − i = 0`.
#[test]
fn integral_across_a_pole_of_a_complex_power_is_divergent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "(1+I)^x/((1+I)^x - 1 - I)");
    for (lo, hi) in [(0, 2), (0, 1), (1, 2)] {
        let r = f.try_integrate_definite(&x, &ctx.int(lo), &ctx.int(hi));
        assert!(
            matches!(r, Err(SymplexError::Divergent { .. })),
            "[{lo}, {hi}]: {r:?}"
        );
    }
}

/// An imaginary rate: `u = e^{iωx}` meets a root `ρ` of modulus 1 on a whole
/// lattice `x = (arg ρ + 2πk)/ω`.  Up to 0.58 the solver's principal root
/// (`x = 1`) stood for the lattice and the poles inside were missed, a
/// silent wrong value: `∫₋₂⁰ dx/(e^{iπx} + 1)` was `2`, `∫₋₄⁰ dx/(iˣ + 1)` `4`,
/// `∫_{−3/2}^{−1/2} dx/(i^{2x} + 1)` `1.5` (poles at `x = −1, −2, −1`:
/// `e^{−iπ} = i^{−2} = −1` exactly; `(x + 1)/(e^{iπx} + 1) → i/π ≠ 0`,
/// not integrable).  Between the poles the value is unchanged:
/// `∫_{−½}^{½} dx/(e^{iπx} + 1) = 1/2` (`1/(e^{iθ} + 1) = ½ −
/// (i/2)·tan(θ/2)`, the odd part integrates to 0).
#[test]
fn integral_across_a_pole_of_an_imaginary_rate_is_divergent() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (f, lo, hi) in [
        ("1/(exp(I*pi*x) + 1)", "-2", "0"),
        ("1/(I^x + 1)", "-4", "0"),
        ("1/(I^(2*x) + 1)", "-3/2", "-1/2"),
    ] {
        let r = parse(&ctx, f).try_integrate_definite(&x, &parse(&ctx, lo), &parse(&ctx, hi));
        assert!(
            matches!(r, Err(SymplexError::Divergent { .. })),
            "{f} on [{lo}, {hi}]: {r:?}"
        );
    }
    let v = parse(&ctx, "1/(exp(I*pi*x) + 1)")
        .try_integrate_definite(&x, &ctx.rational(-1, 2), &ctx.rational(1, 2))
        .and_then(|v| v.eval_decimal(20))
        .unwrap_or_else(|e| format!("{e:?}"));
    assert_eq!(v, "0.5");
}

/// `R(cˣ)` is `R(e^{x·ln c})`: the exact scan of its singular points and the
/// exponential substitution apply (none of these has a pole on the path).
/// Up to 0.58 the first two were refused ("could not determine all
/// singularities of the integrand"; `eval` keeps the ratio `(−½ ln 2 −
/// iπ/4)/(½ ln 2 + iπ/4)` of the rates of `(1+i)^{−x}` and `(1+i)ˣ` a
/// quotient, so they were not seen as powers of one exponential), and so was
/// the third (rate `1 + iπ`).  mpmath 1.3 `quad` over 60 subintervals at 40
/// digits: `-1.460407280485147066949767 - 0.3704288315728084171209532j`,
/// `-1.124667002835546286997111 - 1.051925284980381755577417j`,
/// `-0.441401657542261545511711 - 0.01825874159101865718262945j`.
#[test]
fn integrals_of_rational_functions_of_complex_powers_have_values() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (f, lo, hi, want) in [
        (
            "1/((1+I)^x + (1+I)^(-x) - 3)",
            0,
            3,
            "-1.46040728048514706694976667 - 0.370428831572808417120953222*i",
        ),
        (
            "1/((2+I)^x + (2+I)^(-x) - 3)",
            0,
            4,
            "-1.12466700283554628699711129 - 1.05192528498038175557741679*i",
        ),
        (
            "1/(exp(x*(1 + I*pi)) + exp(-x*(1 + I*pi)) - 3)",
            0,
            3,
            "-0.441401657542261545511710988 - 0.0182587415910186571826294476*i",
        ),
    ] {
        let v = parse(&ctx, f)
            .try_integrate_definite(&x, &ctx.int(lo), &ctx.int(hi))
            .and_then(|v| v.eval_decimal(27))
            .unwrap_or_else(|e| format!("{e:?}"));
        assert_eq!(v, want, "{f}");
    }
}

/// The antiderivative of `1/((1+i)ˣ + (1+i)^{−x} − 3)` exists (up to 0.58
/// "integral contains unevaluated forms"); its derivative is the integrand
/// on each of its pieces (checked at `x = −4, 7/10, 4`, around the joins
/// `x = ∓ln((3 + √5)/2)/ln √2 ≈ ∓2.78`).
#[test]
fn antiderivative_of_complex_powers_in_both_directions() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let f = parse(&ctx, "1/((1+I)^x + (1+I)^(-x) - 3)");
    let big_f = f.try_integrate(&x).expect("an antiderivative");
    let d = big_f.diff(&x) - f.clone();
    for (p, q) in [(-4, 1), (7, 10), (4, 1)] {
        let at = d.subs(&x, &ctx.rational(p, q));
        let v = at.eval_decimal(20).expect("a value");
        assert_eq!(v, "0", "x = {p}/{q}");
    }
}

// ── Item 3: lengths from the data ──

/// A formal series computes its coefficient prefix only up to 2¹⁴ indices
/// (`cast::MAX_SERIES_TERMS`).  Up to 0.58 `coefficient(k)` built the whole
/// prefix `a_0 … a_k` for any `k` asked for: `sin(x).fps_maclaurin(x)
/// .coefficient(2⁴⁰)` asked for 8 TB and aborted.  Below the cap nothing
/// changes (`a_3 = −1/6`, the Maclaurin series of `sin`); a polynomial's
/// coefficient beyond it is its own (0).
#[test]
fn formal_series_coefficients_beyond_the_cap_are_refused() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let k = 1usize << 40;
    let s = x.sin().fps_maclaurin(&x);
    assert_eq!(s.coefficient(3).to_string(), "-1/6");
    assert!(s.try_coefficient(k).is_err());
    assert_eq!(s.coefficient_rational(k), None);
    assert_eq!(s.coefficient(k).to_string(), "nan");
    assert!(s.try_truncate(k).is_err());
    let p = (x.powi(2) + 1).fps_maclaurin(&x);
    let c = p.try_coefficient(k).map(|c| c.to_string());
    assert_eq!(c.unwrap_or_else(|e| format!("{e:?}")), "0");
}

/// The dense univariate view of a factor was sized by its `u32` degree:
/// `factor_multivariate(y·x^(2²⁵) + y)` (monomial content `y`, then
/// `x^(2²⁵) + 1` in one variable) laid out `2²⁵ + 1` rational coefficients
/// (2 GB) and went on to factor a univariate polynomial of degree `2²⁵`
/// (a degree near `2³¹` asked for over 100 GB).  Since 0.59 the view
/// refuses beyond `cast::MAX_DENSE_LIST` (`2²⁴`) entries and the
/// factorization gives up (`None`); a small degree is unchanged (`y·x³ + y
/// = y·(x + 1)·(x² − x + 1)`, three factors, checked by multiplying back).
#[test]
fn multivariate_factor_with_a_huge_univariate_part_gives_up() {
    use symplex::factor_zassenhaus::factor_multivariate;
    use symplex::multipoly::MultiPoly;
    let [x, y]: [MultiPoly; 2] = [MultiPoly::var(2, 0), MultiPoly::var(2, 1)];
    let mut big = x.clone();
    for _ in 0..25 {
        big = big.mul(&big);
    }
    assert!(factor_multivariate(&big.mul(&y).add(&y)).is_none());
    let small = x.mul(&x).mul(&x).mul(&y).add(&y);
    let (content, factors) = factor_multivariate(&small).expect("factors");
    assert_eq!(factors.len(), 3, "{factors:?}");
    let mut back = MultiPoly::constant(2, content);
    for (g, m) in &factors {
        for _ in 0..*m {
            back = back.mul(g);
        }
    }
    assert_eq!(back, small);
}
