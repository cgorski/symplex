//! Tests of session 14 (s14f1): wrong decisions about constants.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use symplex::prelude::*;

use crate::perf_assert::{CpuTimer, cpu_time};

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// An integer power with an exponent beyond `u32` is not truncated.  Up to
/// 0.51 the minimal polynomial of `r^n` cast `n` with `as u32`:
/// `2^(2³² + 1)` became `2¹`, so `2^4294967297 − 2` had `is_zero() =
/// Some(true)`, `2^4294967297` equalled 2 and `2^4294967297 > 2` evaluated
/// to `False`; `3^(2³²)` became `3⁰ = 1` (`3^4294967296 − 1` zero) and
/// `3^(2⁴⁰) − 1` was not positive.  Such a power is now beyond the size
/// the exact test computes (the decision is left open).
///
/// Oracle: exact — `log₂ 2^4294967297 = 4294967297 > 1`, `log₂ 3^(2³²) =
/// 2³²·log₂ 3 > 0`; mpmath `2**mpf(4294967297) − 2` = 6.2066·10¹²⁹²⁹¹³⁹⁸⁶.
#[test]
fn integer_powers_beyond_u32_exponents_are_not_truncated() {
    let ctx = Context::new();
    for (src, positive) in [
        ("2^4294967297 - 2", true),
        ("3^4294967296 - 1", true),
        ("3^1099511627776 - 1", true),
        ("1 - 2^4294967297", false),
    ] {
        let e = parse(&ctx, src);
        assert_ne!(e.is_zero(), Some(true), "{src}");
        assert_ne!(e.is_positive(), Some(!positive), "{src}");
        assert_ne!(e.is_negative(), Some(positive), "{src}");
    }
    let big = parse(&ctx, "2^4294967297");
    let two = ctx.int(2);
    assert_ne!(big.equals(&two), Some(true));
    assert_ne!(big.gt(&two).eval().to_string(), "False");
    assert_ne!(big.le(&two).eval().to_string(), "True");
}

/// Powers of large rationals no longer make the exact zero test compute
/// them in full.  Up to 0.51 `1000000007^1000000007 − 2` (`is_positive`)
/// and `(10⁻⁶)^2000000016·(1 + i)` (`is_real`) multiplied out powers of
/// 3·10¹⁰ and 4·10¹⁰ bits and did not return within 15 s; the size bound
/// (2¹⁶ bits) leaves them to the other deciders at once.
///
/// Oracle: exact — `1000000007^1000000007 > 2`; `(10⁻⁶)^n·(1 + i)` has
/// argument π/4, so it is not real.
#[test]
fn powers_of_large_rationals_are_decided_or_refused_at_once() {
    let ctx = Context::new();
    let start = CpuTimer::start();
    let e = parse(&ctx, "1000000007^1000000007 - 2");
    assert_ne!(e.is_positive(), Some(false));
    assert_ne!(e.is_zero(), Some(true));
    let z = parse(&ctx, "(1/10^6)^2000000016*(1 + I)");
    assert_ne!(z.is_real(), Some(true));
    start.assert_within(Duration::from_secs(20), "both decisions");
}

/// The bound on exact powers keeps the exact zero test of algebraic
/// numbers with large coefficients: `(1 + √2)²⁰⁰` equals `a + b√2` with
/// 77-digit `a`, `b`, a difference that is 0 to every working precision
/// and is proved 0 by its minimal polynomial (as in 0.51).
///
/// Oracle: exact — `(1 + √2)ⁿ = aₙ + bₙ√2` with `aₙ₊₁ = aₙ + 2bₙ`,
/// `bₙ₊₁ = aₙ + bₙ` (Python integers).
#[test]
fn exact_zero_of_a_large_algebraic_power_is_still_proved() {
    let ctx = Context::new();
    let e = parse(
        &ctx,
        "(1+sqrt(2))^200 - (17951761589238335699019986481093877447231503480108456366071358164733843471937 + 12693812353994620481037986488739368440399451028645237163046012909971924256728*sqrt(2))",
    );
    assert_eq!(e.is_zero(), Some(true));
}

/// A complex power with a real value is not called non-real.  `evalf`
/// forms `b^x` as `exp(x·Log b)` and up to 0.51 counted that route as exact
/// for an exact base and exponent, so the rounding residue of `8π` in
/// `(−1)^(8 + 2i) = e^(iπ(8 + 2i))` became a certified imaginary part:
/// `is_real()` was `Some(false)` (also `(−1)^(6 + 2i)`, `(−1)^(100 + i)`,
/// `i^(16 + 4i)`, `i^(1000 + i)` and `ln((−1)^(100 + i)) = −π`).  Their
/// values are unchanged; genuinely non-real powers stay non-real.
///
/// Oracle: exact — `(−1)^(n + bi) = e^(−πb)` for an even integer `n`,
/// `i^(4k + 4bi) = e^(−2πb)`, `i^(1000 + i) = e^(−π/2)`; mpmath at 60
/// digits: `exp(-2*pi)` = 0.0018674427317079888144, `exp(-pi)` =
/// 0.0432139182637722497744, `exp(-pi/2)` = 0.2078795763507619085470,
/// `mpc(-1)**(mpf(1)/3 + 1j)` = 0.0216069591318861 + 0.0374243510134911i.
#[test]
fn complex_power_with_a_real_value_is_not_called_non_real() {
    let ctx = Context::new();
    for (src, value) in [
        ("(-1)^(8+2*I)", 0.001_867_442_731_707_988_8),
        ("(-1)^(6+2*I)", 0.001_867_442_731_707_988_8),
        ("(-1)^(100+I)", 0.043_213_918_263_772_25),
        ("I^(16+4*I)", 0.001_867_442_731_707_988_8),
        ("I^(1000+I)", 0.207_879_576_350_761_9),
        ("log((-1)^(100+I))", -std::f64::consts::PI),
    ] {
        let e = parse(&ctx, src);
        assert_ne!(e.is_real(), Some(false), "{src}");
        if value > 0.0 {
            assert_ne!(e.is_positive(), Some(false), "{src}");
        } else {
            assert_ne!(e.is_negative(), Some(false), "{src}");
        }
        let v = e.eval_f64().unwrap();
        assert!(
            (v - value).abs() <= 4.0 * f64::EPSILON * value.abs(),
            "{src}: {v}"
        );
    }
    for src in ["(-1)^(1/3+I)", "(2+I)^(3+4*I)", "(-8)^(1/3+I)"] {
        assert_eq!(parse(&ctx, src).is_real(), Some(false), "{src}");
    }
    // `|b| = 1` and `Re x = 0` exactly: the route makes no imaginary part,
    // and the real powers stay certified (`(−1)^i = e^(−π)`, `i^i = e^(−π/2)`).
    for src in ["(-1)^I", "I^I", "(-1)^(2*I)", "acos((-1)^I)"] {
        assert_eq!(parse(&ctx, src).is_real(), Some(true), "{src}");
        assert_eq!(parse(&ctx, src).is_positive(), Some(true), "{src}");
    }
}

/// The library functions that tend to 0 at infinity fold to 0 when they
/// are built, so `0/f(∞)` is seen as `0/0`.  Up to 0.51 only `eval` folded
/// them (`besselj`, `bessely`, `besselk` not even there), and the
/// canonical `0·x = 0` made `0/besselj(1, ∞)`, `0/airyai(∞)`,
/// `0/elliptic_k(∞)`, `0/expint(2, ∞)` and `0/uppergamma(2, ∞)` equal to
/// 0, and `eval` of `min(0, airyai(√2), 5)/besselj(1, ∞)` (golden case
/// 71887) gave 0.
///
/// Oracle: SymPy 1.14 `besselj(nu, oo)`, `bessely(nu, -oo)`,
/// `besselk(nu, oo)`, `airyai(-oo)`, `airyaiprime(oo)`, `airybi(-oo)`,
/// `elliptic_k(oo)`, `expint(2, oo)`, `uppergamma(2, oo)` → 0
/// (`J_ν(x) ~ √(2/(πx))·cos(x − νπ/2 − π/4)`, DLMF 10.7.8); `0/0` is nan.
#[test]
fn functions_vanishing_at_infinity_make_zero_over_them_nan() {
    let ctx = Context::new();
    for src in [
        "0/besselj(1, oo)",
        "0/besselj(1/2, -oo)",
        "0/bessely(0, -oo)",
        "0/besselk(2, oo)",
        "0/airyai(oo)",
        "0/airyai(-oo)",
        "0/airyaiprime(oo)",
        "0/airybi(-oo)",
        "0/elliptic_k(-oo)",
        "0/expint(2, oo)",
        "0/uppergamma(3, oo)",
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), "nan", "{src}");
    }
    let golden = parse(&ctx, "Min(0, airyai(sqrt(2)), 5)/besselj(1, oo)");
    assert_eq!(golden.eval().to_string(), "nan");
    for src in [
        "besselj(1, oo)",
        "bessely(0, oo)",
        "besselk(2, oo)",
        "besselj(x, -oo)",
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), "0", "{src}");
    }
    // `I_ν` grows; `K_ν(−∞)` is infinite.
    assert_eq!(parse(&ctx, "besseli(1, oo)").eval().to_string(), "oo");
    assert_ne!(parse(&ctx, "besselk(1, -oo)").eval().to_string(), "0");
}

/// The library functions that are infinite at infinity, where `eval`
/// knew it, are so when built, and `0·f(∞)` is `0·∞`; closed forms are
/// not evaluated at an infinite argument.  Up to 0.51 `0·airybi(∞)`,
/// `0·Chi(∞)`, `0·erfi(−∞)`, `0·Shi(∞)` and `0·besseli(1, ∞)` were 0, and
/// `expint(0, −∞)` (`e^{−x}/x` at `−∞`), `uppergamma(1, −∞)` and
/// `uppergamma(2, −∞)` (`(x + 1)·e^{−x}`) were `nan`; they stay
/// unevaluated now.
///
/// Oracle: `Bi(x) ~ e^{⅔x^{3/2}}/(√π·x^{1/4})`, `I_1(x) ~ e^x/√(2πx)`, `Shi`,
/// `Chi` `~ e^x/(2x)` (DLMF 9.7.7, 10.30.4, 6.12.4): `+∞`; `0·∞` is nan.
/// `E_0(x) = e^{−x}/x → −∞`, `Γ(1, x) = e^{−x} → ∞`, `Γ(2, x) = (x +
/// 1)·e^{−x} → −∞` as `x → −∞` (exact forms); SymPy 1.14 `airybi(oo)`,
/// `Chi(oo)`, `besseli(1, oo)` → oo.
#[test]
fn functions_infinite_at_infinity_make_zero_times_them_nan() {
    let ctx = Context::new();
    for src in [
        "0*airybi(oo)",
        "0*airybiprime(oo)",
        "0*Chi(oo)",
        "0*erfi(-oo)",
        "0*Shi(oo)",
        "0*besseli(1, oo)",
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), "nan", "{src}");
    }
    for src in ["expint(0, -oo)", "uppergamma(1, -oo)", "uppergamma(2, -oo)"] {
        assert_ne!(parse(&ctx, src).eval().to_string(), "nan", "{src}");
    }
}

/// `acos(−u) = π − acos(u)` when `acos(u)` folds.  Since 0.50 `eval`
/// reduces `cos(77π/80)` to `−cos(3π/80)` before `acos` sees it, and
/// `acos(cos(77π/80))`, `asec(sec(10π/18))`, `asec(csc(−11π/100))` and
/// `acos(sin(−18π/44))` stayed unevaluated as `acos(−cos(3π/80))`, … (0.49:
/// `77π/80`, `5π/9`, `61π/100`, `10π/11`).  A negated argument whose
/// `acos` does not fold keeps its form.
///
/// Oracle: the principal range `[0, π]` of `acos` (`77π/80 ∈ [0, π]`,
/// `acos(−sin(9π/22)) = π/2 + 9π/22`); SymPy 1.14 `acos(-cos(3*pi/80))` →
/// `77*pi/80`, `acos(-sin(9*pi/22))` → `10*pi/11`.
#[test]
fn acos_of_a_negated_trigonometric_value_folds() {
    let ctx = Context::new();
    for (src, value) in [
        ("acos(cos(77*pi/80))", "77/80*pi"),
        ("asec(sec(10*pi/18))", "5/9*pi"),
        ("asec(csc(-11*pi/100))", "61/100*pi"),
        ("acos(sin(-18*pi/44))", "10/11*pi"),
        ("acos(-cos(3*pi/80))", "77/80*pi"),
    ] {
        assert_eq!(parse(&ctx, src).eval().to_string(), value, "{src}");
    }
    assert_eq!(parse(&ctx, "acos(-2/3)").eval().to_string(), "acos(-2/3)");
}

/// The integer part of a constant is computed once per arena.  Every
/// `eval_f64`, `eval_decimal` and `eval_complex64` starts with `eval`, and
/// up to 0.51 each evaluated the constant again to as many digits as its
/// integer part has: `ceiling(Chi(1600))` (692 digits) took 1.15 s per
/// call in a release build, and the seven evaluations of `fuzz_evalf`
/// timed out.  The later calls now reuse the integer.
///
/// Oracle: mpmath at 60 and 120 digits: `ceil(chi(400))` =
/// 6.543236408537138671246974·10¹⁷⁰.
#[test]
fn integer_part_of_a_constant_is_computed_once() {
    let ctx = Context::new();
    let e = parse(&ctx, "ceiling(Chi(400))");
    let (exact, cold) = cpu_time(|| e.eval());
    assert_eq!(exact.to_string().len(), 171);
    assert!(exact.to_string().starts_with("6543236408537138671246974"));
    let start = CpuTimer::start();
    let d = e.eval_decimal(16).unwrap();
    let f = e.eval_f64().unwrap();
    let c = e.eval_complex64().unwrap();
    let d30 = e.eval_decimal(30).unwrap();
    let warm = start.cpu();
    assert_eq!(d, "6.543236408537139e170");
    assert!(d30.starts_with("6.54323640853713867124697"), "{d30}");
    assert!((f / 6.543_236_408_537_139e170 - 1.0).abs() < 1e-15);
    assert_eq!(c.im, 0.0);
    assert!(warm < cold, "warm {warm:?}, cold {cold:?}");
}
