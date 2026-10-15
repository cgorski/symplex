//! Tests of session 15 (s15h2): evalf: parts printed as 0, joint bounds, approximate realness.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

fn decimals(ctx: &Context, s: &str, digits: &[u32]) -> Vec<String> {
    let e = parse(ctx, s);
    digits
        .iter()
        .map(|&d| {
            e.eval_decimal(d)
                .unwrap_or_else(|err| format!("{s}: {err:?}"))
        })
        .collect()
}

// ── Item 1: a part O(1) next to a huge one, on the cut ──────────────────────

/// `E₁(−x ± i0) = −Ei(x) ∓ iπ` (DLMF 6.2.4) and `Γ(0, z) = E₁(z)`: the
/// imaginary part of the principal value (from above) is exactly `−π`.  Up
/// to 0.53 `im(expint(1, −160))` and `im(uppergamma(0, −160))` were `0` at
/// 10, 16 and 30 digits: the asymptotic expansion of `Γ(a, z)` on the
/// negative axis is a real sum (the `−iπ` lies inside its remainder bound),
/// and the exact-0 imaginary part of that sum was certified as the value's.
/// `eval_f64` returned the real part and `eval_complex64` had `im == 0.0`.
/// Oracle: mpmath `expint(1, -160)` = `gammainc(0, -160)` =
/// `-1.93080040363983273491483058378e67 - 3.14159265358979323846264338328j`.
#[test]
fn exponential_integral_on_its_cut_has_imaginary_part_minus_pi() {
    let ctx = Context::new();
    for s in ["im(expint(1, -160))", "im(uppergamma(0, -160))"] {
        assert_eq!(
            decimals(&ctx, s, &[10, 16, 30]),
            [
                "-3.141592654",
                "-3.141592653589793",
                "-3.14159265358979323846264338328"
            ],
            "{s}"
        );
    }
    let e = parse(&ctx, "expint(1, -160)");
    assert_eq!(
        e.eval_decimal(10).unwrap(),
        "-1.930800404e67 - 3.141592654*i"
    );
    assert!(matches!(
        e.eval_f64(),
        Err(SymplexError::ComputationFailed { .. })
    ));
    let z = e.eval_complex64().unwrap();
    assert_eq!(z.re, -1.9308004036398328e67);
    assert_eq!(z.im, -std::f64::consts::PI);
}

/// `E_n(−x) = x^(n−1)/(n−1)!·(−Ei(x) − iπ) + e^x·(polynomial)/(n−1)!`
/// (DLMF 8.19.7): `Im E₂(−530) = −530π`.  Up to 0.53 it was `0` at every
/// digit count (60 digits too).  Oracle: mpmath `expint(2, -530)` imaginary
/// part `-1665.04410640259041638520099314` (`= -530*pi`).  Same for
/// `Γ(−2, −300)`, `Im = −π/2` (DLMF 8.4.15): mpmath `gammainc(-2, -300)` =
/// `-7.26708908993561136967272375266e122 - 1.57079632679489661923132169164j`
/// (printed without its imaginary part up to 0.53).
#[test]
fn generalised_exponential_integral_on_its_cut_keeps_its_imaginary_part() {
    let ctx = Context::new();
    assert_eq!(
        decimals(&ctx, "im(expint(2, -530))", &[10, 30, 60]),
        [
            "-1665.044106",
            "-1665.04410640259041638520099314",
            "-1665.04410640259041638520099313813652862449978166880608511672"
        ]
    );
    assert_eq!(
        decimals(&ctx, "uppergamma(-2, -300)", &[10, 16]),
        [
            "-7.26708909e122 - 1.570796327*i",
            "-7.267089089935611e122 - 1.570796326794897*i"
        ]
    );
}

/// The argument's rounding moves `Im E_n(−x)` only by its derivative along
/// the cut (`0` for `E₁`), not by `abs(E_n′)·r` (`e^x`): up to 0.53
/// `expint(1, −1000π)` and `expint(3, −1000·ln 3)` printed no imaginary
/// part at 10 digits.  Oracle: mpmath `expint(1, -1000*pi)` =
/// `-7.57429604002791103874436650247e1360 - 3.14159265358979323846264338328j`,
/// `expint(3, -1000*log(3))` = `-1.20669888736792493334024115106e474 -
/// 1895870.99427332139390550003188j`.
#[test]
fn exponential_integral_at_an_inexact_point_of_the_cut() {
    let ctx = Context::new();
    assert_eq!(
        decimals(&ctx, "expint(1, -pi*1000)", &[10, 16]),
        [
            "-7.57429604e1360 - 3.141592654*i",
            "-7.574296040027911e1360 - 3.141592653589793*i"
        ]
    );
    assert_eq!(
        decimals(&ctx, "expint(3, -log(3)*1000)", &[10, 16]),
        [
            "-1.206698887e474 - 1895870.994*i",
            "-1.206698887367925e474 - 1895870.994273321*i"
        ]
    );
}

/// `Chi(−x) = Chi(x) + iπ`, `Ci(−x) = Ci(x) + iπ` (the `ln z` of DLMF 6.6.5,
/// 6.6.6, from above).  Up to 0.53 the real routine's value carried one
/// bound for both parts: `Chi(−200)` printed `1.815617617e84` at 10, 16, 30
/// and 60 digits, `Chi(−1000·√2)` `5.418870805e610`, and `Ci(−10⁴⁰)` lost
/// its real part (`3.141592653589793*i` at 16 digits).  Oracle: mpmath
/// `chi(-200)` = `1.81561761657967842619835502192e84 +
/// 3.14159265358979323846264338328j`, `chi(-1000*sqrt(2))` =
/// `5.41887080520404849436264382973e610 + 3.14159265358979323846264338328j`,
/// `ci(-10**40)` = `-5.69633400953636327308034181574e-41 +
/// 3.14159265358979323846264338328j`.
#[test]
fn hyperbolic_and_cosine_integrals_of_a_negative_argument() {
    let ctx = Context::new();
    assert_eq!(
        decimals(&ctx, "Chi(-200)", &[10, 30]),
        [
            "1.815617617e84 + 3.141592654*i",
            "1.81561761657967842619835502192e84 + 3.14159265358979323846264338328*i"
        ]
    );
    assert_eq!(
        parse(&ctx, "Chi(-sqrt(2)*1000)").eval_decimal(16).unwrap(),
        "5.418870805204048e610 + 3.141592653589793*i"
    );
    assert_eq!(
        parse(&ctx, "Ci(-10^40)").eval_decimal(16).unwrap(),
        "-5.696334009536363e-41 + 3.141592653589793*i"
    );
    assert!(parse(&ctx, "Chi(-200)").eval_f64().is_err());
}

/// `K_n(−x) = (−1)^n·K_n(x) − iπ·I_n(x)` (DLMF 10.34.2): the real part is
/// `e^(−x)`-small next to the imaginary one.  Up to 0.53 `besselk(0, −200)`
/// printed `-6.40786624e85*i` at every digit count and `besselk(2, −100e)`
/// `-8.538692334e116*i` (`re(besselk(1, −200))`, the part alone, was
/// already right).  Oracle: the identity with mpmath `besselk(0, 200)`
/// = `1.22568197977653345166005408751e-88`, `besseli(0, 200)·π` =
/// `6.4078662396053215269613772812e85`, `besselk(1, 200)` =
/// `1.22874237347298581204459101042e-88`, `besselk(2, 100e)` =
/// `6.76741811327961724249090359726e-120`, `π·besseli(2, 100e)` =
/// `8.5386923340823972155437973589e116` (mpmath's own `besselk(0, -200)`
/// has real part `0.0`).
#[test]
fn bessel_k_of_a_negative_argument_shows_its_small_real_part() {
    let ctx = Context::new();
    assert_eq!(
        decimals(&ctx, "besselk(0, -200)", &[10, 30]),
        [
            "1.22568198e-88 - 6.40786624e85*i",
            "1.22568197977653345166005408751e-88 - 6.4078662396053215269613772812e85*i"
        ]
    );
    assert_eq!(
        parse(&ctx, "re(besselk(1, -200))")
            .eval_decimal(16)
            .unwrap(),
        "-1.228742373472986e-88"
    );
    assert_eq!(
        parse(&ctx, "besselk(2, -E*100)").eval_decimal(16).unwrap(),
        "6.767418113279617e-120 - 8.538692334082397e116*i"
    );
}

/// For a half-integer `n`, `E_n(z) = z^(n−1)·Γ(1 − n, z)` with `Re Γ(1 − n,
/// −x) = Γ(1 − n)` and `z^(n−1) = ±i·x^(n−1)` on the cut: `Im E_{1/2}(−x) =
/// −√(π/x)`.  Up to 0.53 `expint(1/2, −130)` printed `-2.21832939e54`.
/// `uppergamma(1/2, −100) = √π·erfc(10i)` and `erfc(iy) = 1 − i·erfi(y)`:
/// the real parts `√π` and `1` were dropped (`-2.701764561e42*i`,
/// `-1.524307423e42*i` at every digit count), as was `Γ(−1/2) = −2√π` of
/// `uppergamma(−1/2, −150)`.  Oracle: mpmath `expint(0.5, -130)` =
/// `-2.21832939030748416166960001579e54 - 0.155454486378830835704590808478j`;
/// `erfc(10j)` = `1 - 1.52430742270866969936054661473e42j`; the identities
/// with mpmath `gamma(-0.5)` = `-3.54490770181103205459633496668`,
/// `sqrt(pi)` = `1.77245385090551602729816748334` (mpmath's
/// `gammainc(-0.5, -150)` loses the real part at 50 digits).
#[test]
fn half_integer_orders_on_the_cut_keep_their_exact_parts() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "expint(1/2, -130)").eval_decimal(10).unwrap(),
        "-2.21832939e54 - 0.1554544864*i"
    );
    assert_eq!(
        decimals(&ctx, "erfc(10*I)", &[10, 30]),
        [
            "1 - 1.524307423e42*i",
            "1 - 1.52430742270866969936054661473e42*i"
        ]
    );
    assert_eq!(
        parse(&ctx, "uppergamma(1/2, -100)")
            .eval_decimal(10)
            .unwrap(),
        "1.772453851 - 2.701764561e42*i"
    );
    assert_eq!(
        parse(&ctx, "uppergamma(-1/2, -150)")
            .eval_decimal(10)
            .unwrap(),
        "-3.544907702 + 7.663552811e61*i"
    );
    assert_eq!(
        parse(&ctx, "lowergamma(1/2, -100)")
            .eval_decimal(10)
            .unwrap(),
        "2.701764561e42*i"
    );
    let z = parse(&ctx, "erfc(I*sqrt(2)*10)").eval_complex64().unwrap();
    assert_eq!(z.re, 1.0);
}

// ── Item 3: float thresholds that decided a value ───────────────────────────────

/// `polygamma(n, x)` is evaluated for a non-negative integer order only
/// (`polygamma(1/2, 2)` is refused).  Up to 0.53 an order within `10⁻¹²`
/// of an integer was taken for it in `f64`, and an inexact order within
/// `2⁻⁴⁰` by the bound: `polygamma(1 + 10⁻²⁰, 2)` and `polygamma(1 +
/// 10⁻⁵⁰, 2)` were `ψ′(2) = π²/6 − 1 = 0.6449340668…` at 10 and 40
/// digits.  Exact integer orders are unchanged (`ψ‴(2) = π⁴/15 − 6`,
/// mpmath `psi(3, 2)` = `0.4939394022668291490960221792470074166485`).
#[test]
fn polygamma_of_a_non_integer_order_is_refused() {
    let ctx = Context::new();
    for s in ["polygamma(1 + 10^(-20), 2)", "polygamma(1 + 10^(-50), 2)"] {
        assert!(parse(&ctx, s).eval_decimal(40).is_err(), "{s}");
    }
    assert_eq!(
        parse(&ctx, "polygamma(3, 2)").eval_decimal(40).unwrap(),
        "0.4939394022668291490960221792470074166485"
    );
}

/// The range of a `Sum` or `Product` evaluated term by term must be
/// integers.  Up to 0.53 a bound within `10⁻⁹` of an integer in `f64` was
/// taken for it: `Σ_{k=1}^{3 − e⁻¹⁰⁰} k^k` was `32` and the product `108`
/// (under the floor convention it is `1 + 4 = 5`; SymPy leaves it a
/// `Sum`).  A bound proved an integer still evaluates (`floor(π) = 3`).
#[test]
fn sum_range_must_be_proved_integer() {
    let ctx = Context::new();
    for s in [
        "Sum(k^k, k, 1, 3 - exp(-100))",
        "Product(k^k, k, 1, 3 - exp(-100))",
    ] {
        assert!(parse(&ctx, s).eval_decimal(10).is_err(), "{s}");
    }
    assert_eq!(
        parse(&ctx, "Sum(k^k, k, 1, floor(pi))")
            .eval_decimal(10)
            .unwrap(),
        "32"
    );
}

/// `J_ν(0) = 0` for `ν > 0` and `J_0(0) = 1` (the series `(z/2)^ν·Σ …/Γ(ν
/// + k + 1)`, DLMF 10.2.2).  Up to 0.53 an order below `10⁻¹⁵` counted as 0:
/// `besselj(e⁻¹⁰⁰, 0)` was `1`.
#[test]
fn bessel_j_at_zero_of_a_tiny_positive_order() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "besselj(exp(-100), 0)")
            .eval_decimal(10)
            .unwrap(),
        "0"
    );
    assert_eq!(parse(&ctx, "besselj(0, 0)").eval_decimal(10).unwrap(), "1");
}

/// Hankel's expansion of `K_ν(w)` terminates for a half-integer `ν`; up to
/// 0.53 the `f64` simulation of its terms took `4ν² − 1` below `10⁻¹⁶` for 0
/// and stopped after one term, so `besselk(1/2 + 10⁻³⁰, 60 + 5i)` printed
/// `K_{1/2}(60 + 5i)` at 40 digits (`…6642292691474e-28 +
/// …7499637872119e-27*i`).  Oracle: mpmath `besselk(0.5 + 1e-30, 60 + 5j)`
/// at 60 digits = `4.572226065986346641618195406642339306011e-28 +
/// 1.338431568857739269514470777499648548981e-27j`.
#[test]
fn bessel_k_near_a_half_integer_order_is_not_truncated() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "besselk(1/2 + 10^(-30), 60 + 5*I)")
            .eval_decimal(40)
            .unwrap(),
        "4.572226065986346641618195406642339306011e-28 + \
         1.338431568857739269514470777499648548981e-27*i"
    );
}

// ── Item 2: a certified sign is the sign of a real value ─────────────────────────

/// The certified sign test (`poly::algebraic::sign_checked`) gave a value
/// whose imaginary part is certified nonzero the sign of its real part when
/// that part was below the 16 digits evaluated (`is_real_to_digits`), so
/// the definite integrator split `∫₀¹ abs(x² − 1/4 − i·e⁻¹⁰⁰) dx` at
/// `x = 1/2` and integrated `±(x² − 1/4 − i·e⁻¹⁰⁰)` on the pieces: up to
/// 0.53 a closed form whose value is `1/4 − 1.3839·10⁻⁸⁷ + 3.43·10⁻¹³¹i`
/// (mpmath at 400 digits), while the integral of a modulus is real, `1/4 +
/// 1.3928·10⁻⁸⁵` (mpmath `quad` of `sqrt((x² − 1/4)² + e⁻²⁰⁰)` at 120
/// digits).  With `−i/10` in place of `−i·e⁻¹⁰⁰` it was refused; now both
/// are (a correct value would do as well).
#[test]
fn modulus_of_a_nearly_real_quadratic_is_not_split_as_if_real() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for s in ["abs(x^2 - 1/4 - I*exp(-100))", "abs(x^2 - 1/4 - I/10)"] {
        let f = parse(&ctx, s);
        let r = f.try_integrate_definite(&x, &ctx.int(0), &ctx.int(1));
        assert!(r.is_err(), "{s}: {r:?}");
    }
}
