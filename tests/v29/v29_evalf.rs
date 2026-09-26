//! The evalf self-consistency hunt (0.30): random constant expressions
//! evaluated at 16 and 30 digits against the same expression at 60 digits,
//! where every disagreement is a certified wrong digit.  Each test pins one
//! class the hunter found, says what was wrong before, and cites its oracle:
//! mpmath 1.3 with exact rational inputs (`mpf(11)/15`, never a decimal
//! literal), quoted at the digits it printed at two working precisions that
//! agree (the dps are given with each value).

use symplex::prelude::*;

/// `got` is the `f64` nearest to the oracle's decimal `want`, or next to it.
fn f64_near(got: f64, want: &str) {
    let w: f64 = want.parse().expect("oracle literal");
    assert!(
        (got - w).abs() <= w.abs() * f64::EPSILON,
        "got {got:e}, want {want}"
    );
}

fn dec(ctx: &Context, s: &str, digits: u32) -> Result<String, SymplexError> {
    ctx.parse(s)
        .unwrap_or_else(|e| panic!("parse {s}: {e:?}"))
        .eval_decimal(digits)
}

fn check(ctx: &Context, s: &str, digits: u32, want: &str) {
    match dec(ctx, s, digits) {
        Ok(got) => assert_eq!(got, want, "{s} at {digits} digits"),
        Err(e) => panic!("{s} at {digits} digits: {e:?}, want {want}"),
    }
}

fn refused(ctx: &Context, s: &str, digits: u32) {
    match dec(ctx, s, digits) {
        Err(
            SymplexError::PrecisionExhausted { .. }
            | SymplexError::Unevaluable { .. }
            | SymplexError::NotImplemented(_),
        ) => {}
        other => panic!("{s} at {digits} digits: want a refusal, got {other:?}"),
    }
}

// ── Agreement of two precisions is no certificate ──────────────────────────

/// Before: `1 + 10⁻¹⁵⁰` rounds to 1 at 128 and at 256 bits, so the power was
/// `1` at both, the two values agreed, and `evalf` accepted `1` at 16 and
/// 30 digits although the bound said `2^372`; `… − E` was `1 − e` there and
/// `0` at 60 digits.  Only the bound certifies digits now.
#[test]
fn agreement_of_two_precisions_is_not_accepted() {
    let ctx = Context::new();
    // mpmath (dps 400, 500): power(1 + mpf(10)**-150, mpf(10)**150) = 2.7182818284590452354
    check(&ctx, "(1 + 10^(-150))^(10^150)", 16, "2.718281828459045");
    // mpmath (dps 400, 500): power(1 + mpf(10)**-150, mpf(10)**150) - e = -1.3591409142295226177e-150
    check(
        &ctx,
        "(1 + 10^(-150))^(10^150) - E",
        16,
        "-1.359140914229523e-150",
    );
    // Before: 2.934635308511838e-80, I₅₀(1) alone — the first term (2.7·10⁻⁸¹)
    // rounded away identically at 128 and 256 bits.
    // mpmath (dps 200, 300): (exp(mpf(11)/15*mpf(10)**-40) - 1 - mpf(11)/15*mpf(10)**-40)
    //   + besseli(50, 1) = 3.2035241974007270309e-80
    check(
        &ctx,
        "exp(11/15*10^(-40)) - 1 - 11/15*10^(-40) + besseli(50, 1)",
        16,
        "3.203524197400727e-80",
    );
    let e = ctx
        .parse("exp(11/15*10^(-40)) - 1 - 11/15*10^(-40) + besseli(50, 1)")
        .unwrap();
    f64_near(e.eval_f64().unwrap(), "3.2035241974007270309e-80");
}

/// `sin` of a huge exact rational reduces the exact value (with `log₂|r|`
/// extra bits), and its bound now says so; before, only the agreement of
/// two precisions accepted it.
#[test]
fn sin_of_a_huge_rational_is_certified_by_its_bound() {
    let ctx = Context::new();
    // mpmath (dps 200, 300): sin(mpf(10)**100) = -0.3723761236612766882620866955531643
    check(&ctx, "sin(10^100)", 30, "-0.372376123661276688262086695553");
}

/// `sin(⋯ sin(0) + 1 ⋯) + 1` nested 2000 deep: a perfectly conditioned
/// computation (`|sin′| ≤ 1`) whose rounding errors add up linearly in the
/// depth.  Before the fractional bounds of 0.30 every node added about a bit
/// to an integer error exponent (`max(e, rounding) + 1`, `2·max` for a sum of
/// two, `2·max(1, |sin z|)` for `sin`), and once agreement of two precisions
/// no longer certified anything, depth 800 was `PrecisionExhausted` (a bound
/// of `2^1595` at 3,200 bits for depth 1200).
#[test]
fn a_deep_well_conditioned_chain_keeps_its_digits() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut e = x.clone();
    for _ in 0..2000 {
        e = e.sin() + ctx.int(1);
    }
    let at0 = e.subs(&x, &ctx.int(0));
    // mpmath (dps 40, 60): x = 0; 2000 times x = sin(x) + 1
    //   = 1.93456321075202426756326145376885
    f64_near(
        at0.eval_f64().unwrap(),
        "1.93456321075202426756326145376885",
    );
    assert_eq!(
        at0.eval_decimal(30).unwrap(),
        "1.93456321075202426756326145377"
    );
}

// ── Every printed part carries its digits ───────────────────────────────────

/// Before: only `|z|` was certified, and the smaller part printed
/// uncertified digits — the real part `−7.373338…·10⁻¹⁸` (truly
/// `−7.373306…`), the imaginary part `5.00000000000000016666666671757e-33`.
#[test]
fn each_printed_part_is_certified() {
    let ctx = Context::new();
    // mpmath (dps 100, 150): log(lambertw(-exp(-1) + mpf(10)**-35))
    //   = (-7.3733056744706376537e-18 + 3.1415926535897932385j)
    check(
        &ctx,
        "ln(lambertw(-exp(-1) + 10^(-35)))",
        16,
        "-7.373305674470638e-18 + 3.141592653589793*i",
    );
    // mpmath (dps 60, 90): ci(mpf(49436523)/22655660) = 0.37982755071555831293372009023200642
    // mpmath (dps 100, 150): exp(mpf(10)**-16) - 1 - mpf(10)**-16 = 5.0000000000000001666666666666666708e-33
    check(
        &ctx,
        "Ci(49436523/22655660) + I*(exp(10^(-16)) - 1 - 10^(-16))",
        30,
        "0.379827550715558312933720090232 + 5.00000000000000016666666666667e-33*i",
    );
}

// ── eval_f64 of a value that is not real ────────────────────────────────────

/// Before: an imaginary part up to `1e-15` in absolute value was dropped,
/// however large next to the real part: `sqrt(−10⁻⁴⁰)` (`10⁻²⁰·i`) was
/// `0.0`, and the imaginary part `2·10⁻¹²` of the value was lost below.
#[test]
fn eval_f64_refuses_a_value_that_is_not_real() {
    let ctx = Context::new();
    let s = ctx.parse("sqrt(-10^(-40))").unwrap();
    assert!(matches!(
        s.eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    let z = s.eval_complex64().unwrap();
    assert_eq!((z.re, z.im), (0.0, 1e-20));
    // mpmath (dps 60, 90): log(1 + 6*mpf(10)**-16)*mpc(1, 2*mpf(10)**-12)
    //   = (5.9999999999999982e-16 + 1.19999999999999964e-27j)
    let t = ctx.parse("ln(1 + 6*10^(-16))*(1 + 2*10^(-12)*I)").unwrap();
    assert!(matches!(
        t.eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    // A negligible imaginary part is still dropped.
    assert_eq!(
        ctx.parse("1 + I*10^(-30)").unwrap().eval_f64().unwrap(),
        1.0
    );
}

// ── Values hidden by cancellation beyond the old cap ────────────────────────

/// Before: `0.0` / `0` — zero to the precision cap (twice the working
/// precision plus 256 bits), which the cancellation of 493 bits
/// (`1/2 − erf(13√2)/2`), of about 1,400 bits (the closed form `eval` gives
/// `polygamma(497, 7)`) or 2,700 (`polygamma(1700, 3/2)`) exceeds.  The
/// zero search now goes 3,072 bits beyond the working precision.
#[test]
fn a_value_hidden_by_cancellation_is_resolved() {
    let ctx = Context::new();
    // mpmath (dps 60, 90): erfc(13*sqrt(2))/2 = 2.4760633155033892858e-149
    let e = ctx.parse("1/2 - erf(13*sqrt(2))/2").unwrap();
    f64_near(e.eval_f64().unwrap(), "2.4760633155033892858e-149");
    // mpmath (dps 60, 90): polygamma(497, 7) = 1.3592042233588819987e+705
    check(&ctx, "polygamma(497, 7)", 16, "1.359204223358882e705");
    // mpmath (dps 60, 90): polygamma(1700, mpf(3)/2) = -8.8237082017136432916e+4455
    check(&ctx, "polygamma(1700, 3/2)", 16, "-8.823708201713643e4455");
    // True zeros are still 0, and an underflow too.
    for s in [
        "sin(1)^2 + cos(1)^2 - 1",
        "gamma(1/3)*gamma(2/3) - 2*pi/sqrt(3)",
        "exp(-4*10^9)",
    ] {
        check(&ctx, s, 16, "0");
    }
}

/// Before: `fresnels(erfi(335))` hung (the phase `πx²/2` of an `x ≈
/// 2^161900` was reduced modulo 2π).  Far out the value is 1/2 to the
/// working precision, and `S(y) − 1/2 = O(1/y)` bounds its change over the
/// argument's ball, whose radius `2^(161900 − prec)` the derivative bound 1
/// turned into a ball containing 0.  A zero must also be noise itself (0,
/// or falling with the precision): a steady value in a wide ball is not
/// one.
#[test]
fn a_steady_value_in_a_wide_ball_is_not_a_zero() {
    let ctx = Context::new();
    check(&ctx, "fresnels(erfi(335))", 16, "0.5");
}

// ── Bessel functions ────────────────────────────────────────────────────────

/// Before: a half-integer order ends the Hankel expansion with exact zero
/// terms, which never met the convergence test; the power series took
/// over with `1.44·x` guard bits: `besselj(5/2, 10²⁵)` printed
/// `3.8·10¹⁰⁵²⁷`, `besselj(−3/2, 12!)` hung.
#[test]
fn a_terminating_hankel_expansion_is_complete() {
    let ctx = Context::new();
    // mpmath (dps 100, 150): besselj(mpf(5)/2, mpf(10)**25) = 1.8792034894385559467729213508085883e-13
    check(
        &ctx,
        "besselj(5/2, 10^25)",
        30,
        "1.87920348943855594677292135081e-13",
    );
    // mpmath (dps 60, 90): besselj(-mpf(3)/2, 479001600) = -0.00002342694815658865067
    check(
        &ctx,
        "besselj(-3/2, factorial(12))",
        16,
        "-2.342694815658865e-5",
    );
}

/// Before: `I_ν` had only its power series — `besseli(5/2, 5.2·10⁷⁰)` printed
/// `2.18·10⁴⁴⁸⁵⁷` (the value is beyond the exponent range), `besseli(2, 10¹⁶)`
/// hung.  The asymptotic expansion (DLMF 10.40.1) takes over for large `x`.
#[test]
fn bessel_i_of_a_large_argument() {
    let ctx = Context::new();
    // mpmath (dps 60, 90): besseli(mpf(3)/2, 1000) = 2.4828898739852145816654368784136858e+432
    check(
        &ctx,
        "besseli(3/2, 1000)",
        30,
        "2.48288987398521458166543687841e432",
    );
    refused(&ctx, "besseli(5/2, 26/5*10^70)", 16);
    refused(&ctx, "besseli(2, 10^16)", 16);
}

/// Before: the asymptotic expansion of `K_ν` stopped at the first growing
/// term, which for an order large next to `x` is the second, and returned
/// the partial sum as certified: `besselk(20, 60)` was `6.14·10⁻²⁷`,
/// `besselk(50, 100)` `6.29·10⁻⁴⁴`.
#[test]
fn bessel_k_of_a_large_order() {
    let ctx = Context::new();
    // mpmath (dps 60, 90): besselk(20, 60) = 3.7482954006874723836615623141961612e-26
    check(
        &ctx,
        "besselk(20, 60)",
        30,
        "3.7482954006874723836615623142e-26",
    );
    // mpmath (dps 60, 90): besselk(50, 100) = 9.2745226536133258846209386303361707e-40
    check(
        &ctx,
        "besselk(50, 100)",
        30,
        "9.27452265361332588462093863034e-40",
    );
}

/// Before: `(x/2)^ν` of a negative `x` was taken as `|x/2|^ν`, a real
/// number: `besselj(−1/2, −1)` was `0.431…` (mpmath: `besselj(-mpf(1)/2,
/// -1) = (0.0 - 0.4310988680183760795205209672985334000881j)`, dps 40).
/// A non-integer order at a negative argument is refused (complex).
#[test]
fn bessel_j_of_a_negative_argument_and_fractional_order_is_not_real() {
    let ctx = Context::new();
    refused(&ctx, "besselj(-1/2, -1)", 16);
    refused(&ctx, "besselj(1/3, -2)", 16);
}

// ── Phases of asymptotic expansions ─────────────────────────────────────────

/// Before: the phase `πx²/2` of the Fresnel integrals and `(2/3)|x|^{3/2}`
/// of the Airy functions were formed at the working precision, an absolute
/// error of `2^(bits before the point − prec)`: `fresnels(10⁴⁵)` ended
/// `…816212` at 60 digits, `airyai(−467·10⁴⁰)` was wrong from its 13th digit
/// at 30, `airyai(−29/16·10⁷⁰)` entirely.
#[test]
fn asymptotic_phases_carry_their_integer_bits() {
    let ctx = Context::new();
    // mpmath (dps 150, 200): fresnels(mpf(10)**45)
    //   = 0.49999999999999999999999999999999999999999999968169011381620932846
    check(
        &ctx,
        "fresnels(10^45)",
        60,
        "0.499999999999999999999999999999999999999999999681690113816209",
    );
    // mpmath (dps 100, 150): airyai(-467*mpf(10)**40) = 9.2781436045589000268004094281978775e-12
    check(
        &ctx,
        "airyai(-467*10^40)",
        30,
        "9.2781436045589000268004094282e-12",
    );
    // mpmath (dps 150, 200): coth(airyai(mpf(-29)/16*mpf(10)**70)) = -1853068792215882711.8142862580370607
    check(
        &ctx,
        "coth(airyai(-29/16*10^70))",
        30,
        "-1853068792215882711.81428625804",
    );
}

/// Before: the power series of `Ei(x)` for `x < 0` had `|x|·log₂e` guard
/// bits for a cancellation of about twice that: `Ei(−196)` at 60 digits
/// was wrong from its 24th digit.
#[test]
fn ei_of_a_negative_argument() {
    let ctx = Context::new();
    // mpmath (dps 80, 120): ei(-196)
    //   = -3.8355389744389197053028566698988141686823447008608951750990493013e-88
    check(
        &ctx,
        "Ei(-196)",
        60,
        "-3.83553897443891970530285666989881416868234470086089517509905e-88",
    );
}

// ── Decisions ───────────────────────────────────────────────────────────────

/// Before: the zero of `airyai`, `erfc`, `besselk` below the exponent range
/// was exact, and its sign `0`; the functions vanish at no rational point,
/// so the zero is an underflow and the sign undecidable.
#[test]
fn an_underflowing_special_function_is_not_an_exact_zero() {
    let ctx = Context::new();
    for s in [
        "sign(airyai(10^10))",
        "sign(erfc(10^5))",
        "sign(besselk(0, 10^10))",
    ] {
        refused(&ctx, s, 16);
    }
    check(&ctx, "erfc(10^5)", 16, "0");
}

/// Before: `tan` of an argument rounded onto its pole had the first-order
/// ball `±3·10³⁹` around `10³⁹` (the true value is `6.4·10⁷⁹`), and the
/// `Piecewise` below took its first branch at 16 digits.  A ball that
/// reaches an eighth of the distance to the pole has no bound.
#[test]
fn tan_next_to_its_pole_decides_nothing() {
    let ctx = Context::new();
    let big = ctx.parse("23/6*10^50").unwrap();
    let t = ctx.parse("tan(pi/2*(1 - 10^(-80)))").unwrap();
    let first = ctx.parse("uppergamma(1/3, 1091)").unwrap();
    let second = ctx.parse("expint(1, sqrt(17))").unwrap();
    let truth = ctx.bool_true();
    let pw = Ex::piecewise(&[(&first, &big.gt(&t)), (&second, &truth)]);
    // mpmath (dps 60, 90): expint(1, sqrt(17)) = 0.0032568139177394340857
    assert_eq!(pw.eval_decimal(16).unwrap(), "0.003256813917739434");
}

// ── Small arguments ─────────────────────────────────────────────────────────

/// Before: `ln(1 ± iz)` rounded `1 ± iz` to the working precision, losing
/// `log₂(1/|z|)` bits the bound did not know of: `atan(i·10⁻³⁰)` was
/// `9.99999998762973e-31*i` at 16 digits.
#[test]
fn inverse_functions_of_a_small_complex_argument() {
    let ctx = Context::new();
    // mpmath (dps 100, 150): atan(mpc(0, mpf(10)**-30)) = (0.0 + 1.0e-30j)
    check(&ctx, "atan(I*10^(-30))", 16, "1e-30*i");
    check(&ctx, "atan(I*10^(-30))", 30, "1e-30*i");
    // mpmath (dps 100, 150): asin(mpc(0, mpf(10)**-25)) = (0.0 + 1.0e-25j)
    check(&ctx, "asin(I*10^(-25))", 16, "1e-25*i");
}

/// Before: `x − k + 1` for a tiny `x` was formed with the mantissa's bits
/// only and lost `x`: `binomial(10⁻⁷⁰, 6)` — `Γ` next to its pole at `−5` —
/// was `−1.666666533708301e-71`.
#[test]
fn binomial_of_a_tiny_argument() {
    let ctx = Context::new();
    // mpmath (dps 150, 200): binomial(mpf(10)**-70, 6) = -1.6666666666666666667e-71
    check(&ctx, "binomial(10^(-70), 6)", 16, "-1.666666666666667e-71");
}

/// Before: `ComputationFailed` after a million steps of the argument shift;
/// it is a refusal, made up front.
#[test]
fn polygamma_far_left_of_zero_is_refused() {
    let ctx = Context::new();
    // erfi(−5) ≈ −8.3·10⁹
    refused(&ctx, "polygamma(100, erfi(-5))", 16);
}

/// Before: the pole of `ζ` was tested in `f64` (`1 + 10⁻⁴⁵` is 1 there)
/// and `1 − 2^{1−s}`, `1 − s` were rounded: `zeta(1 + 10⁻⁴⁵)` was
/// "a pole", `zeta(−10⁻⁶⁰)` printed `−0.499999999999999999141761477346`
/// at 30 digits, and `dirichlet_eta` of an argument that cancelled to `+0`
/// (the sign bit of a positive number) took Borwein's sum at `s = 0`:
/// `0.4913903202446295`.
#[test]
fn zeta_and_eta_next_to_the_pole_and_to_zero() {
    let ctx = Context::new();
    // mpmath (dps 100, 150): zeta(1 + mpf(10)**-45)
    //   = 1000000000000000000000000000000000000000000000.577215664901532861
    check(
        &ctx,
        "zeta(1 + 10^(-45))",
        60,
        "1000000000000000000000000000000000000000000000.57721566490153",
    );
    // mpmath (dps 100, 150): zeta(-mpf(10)**-60)
    //   = -0.49999999999999999999999999999999999999999999999999999999999908
    check(
        &ctx,
        "zeta(-10^(-60))",
        60,
        "-0.499999999999999999999999999999999999999999999999999999999999",
    );
    // mpmath (dps 100, 150): altzeta(-mpf(10)**-60) = 0.49999999999999999999999999999999999999999999999999999999999977
    check(&ctx, "dirichlet_eta(-10^(-60))", 30, "0.5");
    check(
        &ctx,
        "dirichlet_eta((sqrt(40) - sqrt(40)*(1 + 10^(-40)))/10^17)",
        16,
        "0.5",
    );
    // mpmath (dps 100, 150): gamma(-5 + mpf(10)**-70)
    //   = -8.3333333333333333333333333333333333333333333333333333333333333e+67
    check(&ctx, "gamma(-5 + 10^(-70))", 16, "-8.333333333333333e67");
    // The bound of ζ′ was at least 1: the argument `10²⁰π`, known to its
    // rounding, moved the value by `2^(66 − prec)`, and only the agreement
    // of two precisions accepted `1` (a larger argument,
    // `720·polygamma(1669, E) ≈ 10⁴⁰⁰⁰`, came out `0`).  `|ζ′(σ)| < 2^(1−σ)`
    // for `σ ≥ 3`.
    check(&ctx, "zeta(10^20*pi)", 30, "1");
}

/// Before: `Si` reduced its argument modulo 2π however large:
/// `Si(erfi(765))` (`≈ 2^844000`) hung.  Far out it is `±π/2` to the
/// working precision.
#[test]
fn si_of_a_huge_argument() {
    let ctx = Context::new();
    // mpmath (dps 100, 150): si(mpf(10)**60) = 1.5707963267948966192313216916397514420985846996875529104874729
    check(&ctx, "Si(10^60)", 30, "1.57079632679489661923132169164");
    check(&ctx, "Si(erfi(765))", 16, "1.570796326794897");
}

// ── The integrator's sign test (poly::algebraic::sign_checked) ──────────────

/// Before: the sign of the discriminant `exp(−40)` of `x² + 2x + 1 +
/// exp(−40)` (completed square `(x + 1)² + exp(−40)`) was decided from an
/// `f64` with a tolerance: `|v| ≤ 10⁻¹⁴` was sign 0, and the atan form was
/// not tried (the integral stayed unevaluated).  It is decided from
/// certified digits now; the antiderivative's derivative is the integrand
/// at 30 digits.
#[test]
fn a_tiny_discriminant_has_a_certified_sign() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for s in [
        "1/(x^2 + 2*x + 1 + exp(-40))",
        "1/(x^2 + 2*x + 1 - exp(-40))",
    ] {
        let f = ctx.parse(s).unwrap();
        let Ok(big_f) = f.try_integrate(&x) else {
            assert!(s.contains("- exp"), "{s} should integrate");
            continue;
        };
        let d = big_f.diff(&x);
        for (p, q) in [(1, 3), (7, 5), (-13, 4)] {
            let pt = ctx.rational(p, q);
            assert_eq!(
                d.subs(&x, &pt).eval_decimal(30).unwrap(),
                f.subs(&x, &pt).eval_decimal(30).unwrap(),
                "{s} at {p}/{q}: F = {big_f}"
            );
        }
    }
}
