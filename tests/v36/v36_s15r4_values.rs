//! Tests of session 17, wave 16 (s15r4): printing and evaluation of parts below the range; slow evaluations.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use std::time::Duration;

use crate::perf_assert::assert_fast;
use symplex::errors::SymplexError;
use symplex::prelude::*;

fn parse(ctx: &Context, src: &str) -> Ex {
    ctx.parse(src).unwrap()
}

/// Is `r` the refusal of a number one part of which is certainly not 0 but
/// lies below the exponent range (`imaginary`: which part)?
fn part_below_refused(r: &Result<String, SymplexError>, imaginary: bool) -> bool {
    let part = if imaginary {
        "the imaginary part is not 0 but underflows"
    } else {
        "the real part is not 0 but underflows"
    };
    matches!(r, Err(SymplexError::Unevaluable { reason }) if reason.contains(part))
}

/// Is `r` the refusal of a whole value below the exponent range?
fn below_refused(r: &Result<String, SymplexError>) -> bool {
    matches!(r, Err(SymplexError::Unevaluable { reason })
        if reason.contains("the value is not 0 but underflows"))
}

/// A part certified nonzero is never dropped (decision D3); one below the
/// exponent range cannot be printed, so the number is refused, as a whole
/// value below the range is (`SymplexError::Unevaluable`).  Before: the part
/// was dropped like a part whose ball contains 0 — `1 + I*exp(-10^10)`
/// printed `1`, `exp(-10^10) + I` printed `i`, `lowergamma(1, 10^10 + I)` and
/// `lowergamma(2, 10^10 + I)` printed `1`, `lowergamma(3, 10^10 + I)` printed
/// `2`, `I*besselk(0, 10^10) + 1` printed `1`, `2 + I*lowergamma(10^10, 1/2)`
/// printed `2`, `pi + I*exp(-10^10)` printed π, and `eval_f64` returned the
/// real part.
///
/// Oracle: mpmath 1.3 (unbounded exponents) at mp.dps 40 and 80:
/// `mpc(1, exp(-mpf(10)**10))` = `1 + 9.278584420324872578e-4342944820j`;
/// `1 - exp(-mpc(10**10, 1))` = `1 + 7.807659569793976068e-4342944820j`
/// (`lowergamma(1, z) = 1 − e^(−z)`); `2 - gammainc(3, mpc(10**10, 1))` = `2 +
/// 7.807659570352859870e-4342944800j`; `besselk(0, mpf(10)**10)` =
/// `1.162898102812314639e-4342944824`; `pi - gammainc(-2, mpc(10**10,
/// -mpf(10)**-5))` = `3.14159… − 9.278584420170229504e-4342944855j`; all
/// below `2^(−2³¹) ≈ 10^(−646456993)`.
#[test]
fn a_certified_nonzero_part_below_the_range_is_refused() {
    let ctx = Context::new();
    for (src, imaginary) in [
        ("1 + I*exp(-10^10)", true),
        ("exp(-10^10) + I", false),
        ("lowergamma(1, 10^10 + I)", true),
        ("lowergamma(2, 10^10 + I)", true),
        ("lowergamma(3, 10^10 + I)", true),
        ("I*besselk(0, 10^10) + 1", true),
        ("2 + I*lowergamma(10^10, 1/2)", true),
        ("pi + I*exp(-10^10)", true),
        ("1 + I*erfc(10^5)", true),
        ("gamma(1/3) + I*airyai(10^7)", true),
        // The scaled sum of the terms below the range decides their signs
        // next to a term in range that is not a rational.
        ("pi + I*exp(-10^10) - I*exp(-2*10^10)", true),
        ("pi - uppergamma(-2, 10^10 - I/10^5)", true),
        ("1 - E1(10^10) - I*besselj(10^9, 1)", true),
    ] {
        let e = parse(&ctx, src);
        for digits in [16, 30] {
            let d = e.eval_decimal(digits);
            assert!(part_below_refused(&d, imaginary), "{src}: {d:?}");
        }
        // A nonzero imaginary part, however small, is not an `f64`.
        assert!(e.eval_f64().is_err(), "{src}");
    }
    // `eval_complex64`: the part is 0.0, its correctly rounded `f64` (the
    // documented rule of the `f64` routes for a value below the range).
    let c = parse(&ctx, "1 + I*exp(-10^10)").eval_complex64().unwrap();
    assert_eq!((c.re, c.im), (1.0, 0.0));
}

/// The part is certified through arithmetic and the functions whose
/// derivative on the real axis has no zero: `Im f(a + iε) = f′(a)·ε·(1 +
/// O(ε²))`.  Before: every one of these printed its real part alone
/// (`2*(1 + I*exp(-10^10))` printed `2`, `sinh(2 + I*exp(-10^10))` printed
/// `3.62686040784702`).
///
/// Oracle: the derivative, and mpmath at mp.dps 40 and 80 for its factors:
/// `cosh(2)*exp(-mpf(10)**10)` = `3.490785032550194999e-4342944819` (the
/// imaginary part of `sinh(2 + iε)`); `2ε`, `−ε/π²·…` likewise nonzero.
#[test]
fn a_part_below_the_range_is_certified_through_arithmetic_and_monotone_functions() {
    let ctx = Context::new();
    for (src, imaginary) in [
        ("2*(1 + I*exp(-10^10))", true),
        ("I*(1 + I*exp(-10^10))", false),
        ("-I*pi*(pi + I*besselk(0, 10^10))", false),
        ("conjugate(pi + I*exp(-10^10))", true),
        ("(1 + I*exp(-10^10))^2", true),
        ("(pi + I*exp(-10^10))^2", true),
        ("(pi + I*exp(-10^10))^(-3)", true),
        ("1/(1 + I*exp(-10^10))", true),
        ("sqrt(2 + I*exp(-10^10))", true),
        ("exp(pi + I*exp(-10^10))", true),
        ("sinh(2 + I*exp(-10^10))", true),
        ("tanh(2 + I*exp(-10^10))", true),
        ("atan(2 + I*exp(-10^10))", true),
        ("log(2 + I*exp(-10^10))", true),
        ("erf(2 + I*exp(-10^10))", true),
        ("sin(1 + I*exp(-10^10))", true),
        ("cosh(2 + I*exp(-10^10))", true),
        // A product of two factors with such parts: `π·ε₂ − 3·ε₁`, of one sign.
        ("(pi + I*exp(-10^10))*(-3 - I*exp(-3*10^9))", true),
    ] {
        let d = parse(&ctx, src).eval_decimal(30);
        assert!(part_below_refused(&d, imaginary), "{src}: {d:?}");
    }
}

/// A part that is exactly 0, or 0 by proof, prints as before; so does a
/// real value next to a term below the range, and a part in range.
///
/// Oracle: exact identities (`ε − ε = 0`; a conjugate pair is real; `0·ε =
/// 0`), and mpmath at mp.dps 40 and 80: `mpc(1, 1)*… `: `(1 +
/// i·e^(−10¹⁰))/(2 + i)` = `0.4 − 0.2j` to 30 digits (both parts in range),
/// `exp(mpc(-1000, 1))*1j + 1` = `1 + 2.74255229683787186464236378906e-435j`.
#[test]
fn zero_parts_and_parts_in_range_print_as_before() {
    let ctx = Context::new();
    for (src, expected) in [
        ("1 + I*exp(-10^10) - I*exp(-10^10)", "1"),
        ("1 + I*exp(-10^10)*0", "1"),
        ("1 + exp(-10^10 + I) + exp(-10^10 - I)", "1"),
        ("exp(-10^10) + 1", "1"),
        ("1 + besselj(10^9, 1)", "1"),
        ("re(1 + I*exp(-10^10))", "1"),
        ("abs(1 + I*exp(-10^10))", "1"),
        ("1 + I + I*exp(-10^10)", "1 + i"),
        ("(1 + I*exp(-10^10))/(2 + I)", "0.4 - 0.2*i"),
        (
            "exp(-1000 + I)*I + 1",
            "1 + 2.74255229683787186464236378906e-435*i",
        ),
    ] {
        let d = parse(&ctx, src).eval_decimal(30).unwrap();
        assert_eq!(d, expected, "{src}");
    }
    // The part alone is a whole value below the range.
    assert!(below_refused(
        &parse(&ctx, "im(1 + I*exp(-10^10))").eval_decimal(30)
    ));
}

/// A part within the underflow bound that is *not certified nonzero* follows
/// the rule of every part whose ball contains 0: dropped (D3 drops a part
/// "consistent with 0").  This pins the limit of the certificate, not a
/// claim that the part is 0: two values below the range without a scaled
/// form may cancel (`K₀(10¹⁰) − Ai(10⁷)`; mpmath: `besselk(0, mpf(10)**10) -
/// airyai(mpf(10)**7)` = `1.1629e-4342944824`, the first dominates, which
/// the evaluation does not know), and `Γ′(π) ≠ 0` is not asked.
#[test]
fn a_part_below_the_range_that_is_not_certified_is_dropped_as_before() {
    let ctx = Context::new();
    for (src, expected) in [
        ("1 + I*(besselk(0, 10^10) - airyai(10^7))", "1"),
        ("gamma(pi + I*exp(-10^10))", "2.28803779534003"),
    ] {
        let d = parse(&ctx, src).eval_decimal(15).unwrap();
        assert_eq!(d, expected, "{src}");
    }
}

/// `E₁`, `E_n`, `Γ(a, z)` and `erfc` off the real axis below the exponent
/// range give the underflow outcome of a whole value below the range
/// (`SymplexError::Unevaluable`, "not 0 but underflows"), and `eval_f64` of
/// the modulus-free value its documented 0.  Before: `PrecisionExhausted`
/// ("requested 30 digits, achieved 0") — the prefactor's 0 was read as a
/// total cancellation.
///
/// Oracle: mpmath 1.3 at mp.dps 40 and 80, `mag(e1(mpc(10**10, 1)))` =
/// `−14426950441` (`2^(−1.44·10¹⁰)`, below `2^(−2³¹)`); likewise `expint(2,
/// mpc(10**10, -1))`, `gammainc(1/3, mpc(10**10, 1))`, `erfc(mpc(10**5, 1))`
/// (`abs ≈ e^(−10¹⁰)`).
#[test]
fn complex_exponential_integrals_below_the_range_are_an_underflow() {
    let ctx = Context::new();
    for src in [
        "E1(10^10 + I)",
        "E1(10^10 - I)",
        "expint(1, 10^10 + I)",
        "expint(2, 10^10 - I)",
        "expint(1/2, 10^10 + I)",
        "expint(2 + I, 10^10)",
        "uppergamma(1/3, 10^10 + I)",
        "uppergamma(1/2, 10^10 + I)",
        "uppergamma(-1/2, 10^10 - I)",
        "uppergamma(1/3 + I, 10^10)",
        "erfc(10^5 + I)",
        "erfc(10^5 - I/3)",
        // Just below the bottom of the range (`e^(−1488522236) < 2^(−2³¹)`).
        "E1(1488522236 + I)",
    ] {
        let e = parse(&ctx, src);
        for digits in [16, 30] {
            let d = e.eval_decimal(digits);
            assert!(below_refused(&d), "{src}: {d:?}");
        }
    }
    // A value within the range, 275 bits above its bottom, is printed (as
    // before).  mpmath: `e1(mpc(1488522000, 1))` =
    // `5.865288048029077766e-646456901 − 9.134644926824279880e-646456901j`.
    assert_eq!(
        parse(&ctx, "E1(1488522000 + I)").eval_decimal(30).unwrap(),
        "5.86528804802907776625799608656e-646456901 - 9.13464492682427987950564596176e-646456901*i"
    );
}

/// Such a value is held scaled (mantissa and binary scale), so what brings
/// it back into the range is evaluated, and a part of it next to a part in
/// range is certified.  Before: every one was `PrecisionExhausted`
/// (`uppergamma(1/3, 10^10)/exp(-10^10)`: "the value is not 0 but
/// underflows").
///
/// Oracle: mpmath 1.3 at mp.dps 40 and 80 — `e1(mpc(10**10, 1))*exp(mpf(10)**10)`
/// = `5.403023057299623883555654277866e-11 −
/// 8.414709847777796387561353064360e-11j`; `log(e1(mpc(10**10, 1)))` =
/// `−10000000023.02585093004045684017 − 1.00000000009999999999j`;
/// `gammainc(mpf(1)/3, mpf(10)**10)/exp(-mpf(10)**10)` =
/// `2.154434689888254742447772814952e-7`; `gammainc(mpf(1)/3, mpc(10**10,
/// 1))*exp(mpc(10**10, 1))` = `2.154434689888254742435803733343e-7 −
/// 1.436289793115207515666960104626e-17j`; `erfc(mpc(10**5,
/// 1))*exp(mpc(10**5, 1)**2)` = `5.641895834631278494427150137249e-6 −
/// 5.641895834067088911161488641910e-11j`.
#[test]
fn values_below_the_range_come_back_into_it() {
    let ctx = Context::new();
    for (src, expected) in [
        (
            "E1(10^10 + I)*exp(10^10)",
            "5.40302305729962388355565427787e-11 - 8.41470984777779638756135306436e-11*i",
        ),
        (
            "log(E1(10^10 + I))",
            "-10000000023.0258509300404568402 - 1.00000000009999999999*i",
        ),
        (
            "uppergamma(1/3, 10^10)/exp(-10^10)",
            "2.15443468988825474244777281495e-7",
        ),
        (
            "uppergamma(1/3, 10^10 + I)*exp(10^10 + I)",
            "2.15443468988825474243580373334e-7 - 1.43628979311520751566696010463e-17*i",
        ),
        (
            "erfc(10^5 + I)*exp((10^5 + I)^2)",
            "5.64189583463127849442715013725e-6 - 5.64189583406708891116148864191e-11*i",
        ),
    ] {
        let d = parse(&ctx, src).eval_decimal(30).unwrap();
        assert_eq!(d, expected, "{src}");
    }
    for (src, imaginary) in [("1 + I*E1(10^10 + I)", true), ("1 + E1(10^10 + I)", true)] {
        let d = parse(&ctx, src).eval_decimal(30);
        assert!(part_below_refused(&d, imaginary), "{src}: {d:?}");
    }
}

/// Where the certificate of a part does not reach — a product of two factors
/// with parts below the range of different signs, a function without a rule,
/// a sum of a scaled and an unscaled value — a number that depends on `E_n`,
/// `Γ(a, z)` or `erfc` below the range off the real axis is refused rather
/// than printed without the part.  Before (0.58): refused too, with
/// `PrecisionExhausted`, because the inner function failed; the first draft
/// of this wave printed `-9.42477796076938`, `-0.841470984807897*i`,
/// `0.666666666666667` and `-1`, hiding the part.  The values brought back
/// into the range (`values_below_the_range_come_back_into_it`) are not
/// affected: no part of them is dropped.
///
/// Oracle: mpmath 1.3 at mp.dps 70 and 110 (the hunt's `oracle.py`): `(pi +
/// pi*expint(2, mpc(10**10, -1)))*(-3 + (1 + 1j)*besselk(0, mpf(10)**10))` =
/// `−9.42477796077 + 3.65327855121e-4342944824j`; `sinh(-1j + 1j*expint(2,
/// mpc(10**10, -1)))` = `−4.21849646842e-4342944830 − 0.841470984808j`; `2/3 +
/// (1 + 1j)*airyai(mpf(10)**7) − 1j*pi*expint(2, mpc(10**10, -1))` has the
/// imaginary part `−1.3e-4342944829`; `(-1j − 1j*pi*erfc(mpc(10**5, 1)) −
/// airyai(mpf(10)**7))/1j` = `−1 − 3.19377461045e-4342944825j`.
#[test]
fn an_untracked_part_from_a_complex_route_below_the_range_is_refused() {
    let ctx = Context::new();
    for src in [
        "(pi + pi*expint(2, 10^10 - I))*(-3 + (1 + I)*besselk(0, 10^10))",
        "sinh(-I + I*expint(2, 10^10 - I))",
        "2/3 + (1 + I)*airyai(10^7) - I*pi*expint(2, 10^10 - I)",
        "(-I - I*pi*erfc(10^5 + I) - airyai(10^7))/I",
        "(5 + uppergamma(1/3, 10^10 + I))*(pi + I*erfc(10^5 + I))",
    ] {
        let e = parse(&ctx, src);
        let d = e.eval_decimal(15);
        assert!(
            matches!(&d, Err(SymplexError::Unevaluable { reason })
                if reason.contains("not dropped")),
            "{src}: {d:?}"
        );
        assert!(e.eval_f64().is_err(), "{src}");
    }
}

/// `E(m)` far off the axis: the gap between its parts (`abs(E) ≈ √m`, the
/// small part `≈ 1/√m`) is pursued while the exponent of `m` explains it and
/// the working precision stays within a work bound.  Before: `elliptic_e:
/// the value cancels beyond the working precision` (a gap of 4,319 bits
/// against a cap of 4,096).  18 s in a debug build: run with `--release
/// --run-ignored all` (0.8 s).
///
/// Oracle: mpmath 1.3 `ellipe(mpc(mpf(10)**1300, 1))` at mp.dps 3100 and
/// 3600 = `1.2853981633974483096156608458199e-650 − 1.0e+650j`;
/// `ellipe(mpc(-mpf(10)**1300, 1))` = `1.0e+650 − 5.0e-651j`.
#[test]
#[ignore = "about 20 s in a debug build (0.8 s in release)"]
fn elliptic_e_far_off_the_axis_beyond_the_old_gap_cap() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "elliptic_e(10^1300 + I)")
            .eval_decimal(16)
            .unwrap(),
        "1.285398163397448e-650 - 1e650*i"
    );
    assert_eq!(
        parse(&ctx, "elliptic_e(-10^1300 + I)")
            .eval_decimal(16)
            .unwrap(),
        "1e650 - 5e-651*i"
    );
}

/// The work bound: beyond it the gap is refused at once (3 ms), not pursued.
/// Oracle: none needed (a refusal); mpmath `ellipe(mpc(mpf(10)**3000, 1))` =
/// `1.2853981633974483e-1500 − 1.0e+1500j` exists.
#[test]
fn elliptic_e_gap_beyond_the_work_bound_is_refused_fast() {
    let ctx = Context::new();
    let e = parse(&ctx, "elliptic_e(10^3000 + I)");
    let r = assert_fast!(Duration::from_secs(1), e.eval_decimal(16));
    assert!(
        matches!(&r, Err(SymplexError::Unevaluable { reason })
            if reason.contains("cancels beyond the working precision")),
        "{r:?}"
    );
}

/// `Li_s(−x)` next to `x = 1` by the expansion at `z = −1`, `−Σ η(s −
/// k)·ln(x)^k/k!`.  Before: `2^(1−s)·Li_s(x²) − Li_s(x)` from the two
/// expansions at `z = 1`, whose terms `Γ(1 − s)·(−ln x)^(s−1)` cancel 580
/// bits for `polylog(-3/2, -1 + 10^(-70))`: 0.72 s at 60 digits in a release
/// build (0.91 s for the five evaluations of the fuzzer's check, a CI
/// timeout under AddressSanitizer), now 0.05 s.  The values are unchanged.
///
/// Oracle: mpmath 1.3 at mp.dps 40 and 80, `polylog(mpf(-3)/2, -1 +
/// mpf(10)**-70)` = `−0.118680870719840212043598557249193`;
/// `polylog(mpf(1)/3, -1 + mpf(10)**-40)` =
/// `−0.5717528338252776649364756811360327`; `polylog(3, -1 + mpf(10)**-30)`
/// = `−0.901542677369695714049803621132765`; `polylog(mpf(5)/2,
/// -mpf(24)/25)` = `−0.8364647778091577430942219528977354`.
#[test]
fn polylog_next_to_minus_one_by_its_expansion_there() {
    let ctx = Context::new();
    let e = parse(&ctx, "polylog(-3/2, -1 + 10^(-70))");
    // 0.5 s of CPU time in a debug build, ten times the 0.05 s of a release
    // build (in which it took 0.72 s before).
    let d = assert_fast!(Duration::from_secs(3), e.eval_decimal(60));
    assert_eq!(
        d.unwrap(),
        "-0.118680870719840212043598557249192987856012401631778263584591"
    );
    for (src, expected) in [
        (
            "polylog(1/3, -1 + 10^(-40))",
            "-0.571752833825277664936475681136",
        ),
        (
            "polylog(3, -1 + 10^(-30))",
            "-0.901542677369695714049803621133",
        ),
        ("polylog(5/2, -24/25)", "-0.836464777809157743094221952898"),
    ] {
        assert_eq!(
            parse(&ctx, src).eval_decimal(30).unwrap(),
            expected,
            "{src}"
        );
    }
}

/// `erf(u) + erfc(u)` is 1 exactly; the pair is folded in a sum.  Before:
/// two rounded values whose sum was 1 only within its error, so `erfc(√2) −
/// 1 + erf(√2)` was a zero ball at every precision and `floor` of it — a
/// step of `floor` at 0 — was refused (`PrecisionExhausted`) after a search
/// to the last precision: 0.2 s per call in a release build (1.0 s for the
/// fuzzer's five).
///
/// Oracle: the identity `erfc = 1 − erf` (DLMF 7.2.2); mpmath at mp.dps 40
/// and 80 gives `erfc(sqrt(2)) - 1 + erf(sqrt(2))` = `0.0`.
#[test]
fn erf_plus_erfc_is_one_exactly() {
    let ctx = Context::new();
    let e = parse(&ctx, "floor(erfc(sqrt(2)) - 1 + erf(sqrt(2)))");
    // 0.02 s of CPU time in a debug build (before: 0.2 s in a release build,
    // and a refusal).
    let d = assert_fast!(Duration::from_millis(500), e.eval_decimal(30));
    assert_eq!(d.unwrap(), "0");
    for (src, expected) in [
        ("erfc(sqrt(2)) - 1 + erf(sqrt(2))", "0"),
        ("floor(2*erf(3/7) + 2*erfc(3/7) - 2)", "0"),
        ("floor(erf(1 + I) + erfc(1 + I))", "1"),
        ("sign(erf(pi) + erfc(pi) - 1)", "0"),
        (
            "-3/7*erf(10) - 3/7*erfc(10)",
            "-0.428571428571428571428571428571",
        ),
        // Different arguments are not a pair.  mpmath: `erf(mpf(3)/7) +
        // erfc(pi)` = `0.45555489887166497999739245738581`.
        ("erf(3/7) + erfc(pi)", "0.455554898871664979997392457386"),
    ] {
        assert_eq!(
            parse(&ctx, src).eval_decimal(30).unwrap(),
            expected,
            "{src}"
        );
    }
}

/// The coefficients of the series of `W` at its branch point are kept, and
/// only as many are formed as the series takes.  Before: the first 64 exact
/// rationals were built for every evaluation next to `−1/e` (22 ms, whatever
/// the precision): `sinh(W(−1/e + 3·10⁻³⁰⁰))` took 0.09 s at 16 digits in a
/// release build (0.42 s for the fuzzer's five), now 2 ms.  The values are
/// unchanged.
///
/// Oracle: mpmath 1.3 at mp.dps 40 and 80, `sinh(lambertw(-1/e +
/// 3*mpf(10)**-300))` = `−1.175201193643801456882381850595601`;
/// `lambertw(-1/e + 3*mpf(10)**-30)` = `−0.9999999999999959614741587115939935`;
/// `lambertw(-1/e + mpf(10)**-20, -1)` = `−1.000000000233164398177834299194684`.
#[test]
fn lambert_w_next_to_its_branch_point_is_fast() {
    let ctx = Context::new();
    let e = parse(&ctx, "sinh(LambertW(-1/E + 3*10^(-300)))");
    // 0.02 s of CPU time in a debug build (before: 0.09 s in a release build).
    let d = assert_fast!(Duration::from_millis(300), e.eval_decimal(30));
    assert_eq!(d.unwrap(), "-1.1752011936438014568823818506");
    for (src, expected) in [
        (
            "LambertW(-1/E + 3*10^(-30))",
            "-0.999999999999995961474158711594",
        ),
        (
            "lambertw(-1/E + 10^(-20), -1)",
            "-1.00000000023316439817783429919",
        ),
    ] {
        assert_eq!(
            parse(&ctx, src).eval_decimal(30).unwrap(),
            expected,
            "{src}"
        );
    }
}

/// `ζ` of a rational next to a trivial zero from the exact rational:
/// `sin(πq/2) = (−1)ⁿ·sin(πd)` with `d = q/2 + n` exact.  Before: the float
/// argument carried its rounding as a radius far above the value, and
/// `zeta(-2 + 10^(-300))` took four evaluations up to 1,152 bits at 16
/// digits: 0.09 s in a release build (0.44 s for the fuzzer's five), now 2
/// ms.  The values are unchanged.
///
/// Oracle: mpmath 1.3 at mp.dps 340 and 400 (at 80 digits `-2 + 1e-300` is
/// `-2`): `zeta(-2 + mpf(10)**-300)` = `−3.04484570583932707802515304712e-302`,
/// and the identity `ζ′(−2) = −ζ(3)/(4π²)` = `−0.0304484570583932707802515…`;
/// at mp.dps 40 and 80, `zeta(-4 - mpf(10)**-40)` =
/// `−7.983811450268624280696670798789e-43`.
#[test]
fn zeta_next_to_a_trivial_zero_from_the_exact_argument() {
    let ctx = Context::new();
    let e = parse(&ctx, "zeta(-2 + 10^(-300))");
    // 0.03 s of CPU time in a debug build (before: 0.09 s in a release build).
    let d = assert_fast!(Duration::from_millis(300), e.eval_decimal(30));
    assert_eq!(d.unwrap(), "-3.04484570583932707802515304712e-302");
    assert_eq!(e.eval_f64().unwrap(), -3.044845705839327e-302);
    assert_eq!(
        parse(&ctx, "zeta(-4 - 10^(-40))").eval_decimal(30).unwrap(),
        "-7.98381145026862428069667079879e-43"
    );
    // On the zero, and away from it, as before (mpmath: `zeta(mpf(-5)/2)` =
    // `0.0085169287778503305423585670283445`).
    assert_eq!(parse(&ctx, "zeta(-2)").eval_decimal(30).unwrap(), "0");
    assert_eq!(
        parse(&ctx, "zeta(-5/2)").eval_decimal(30).unwrap(),
        "0.00851692877785033054235856702834"
    );
}
