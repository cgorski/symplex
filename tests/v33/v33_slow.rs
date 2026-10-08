//! Freezes and slowness at extreme parameters: Bessel functions of huge
//! order (the `fuzz_roundtrip` timeouts of the 0.34 release run, where
//! building an expression evaluated them in the canonical zero test), and
//! polygamma of high order (the nightly `fuzz_evalf` failure of 2026-10-05).
//! Every call must finish well within its time limit with the right value or
//! an honest refusal.

use std::time::{Duration, Instant};

use symplex::prelude::*;

/// Generous for a debug build on CI: the release times are milliseconds.
const LIMIT: Duration = Duration::from_secs(5);

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed();
    assert!(dt < LIMIT, "{what} took {dt:?}");
    r
}

fn dec(ctx: &Context, s: &str) -> Result<String, SymplexError> {
    timed(s, || ctx.parse(s).unwrap().eval_decimal(15))
}

fn nonzero_underflow(ctx: &Context, s: &str) {
    match dec(ctx, s) {
        Err(SymplexError::Unevaluable { reason }) if reason.contains("is not 0 but underflows") => {
        }
        other => panic!("{s}: want a nonzero underflow, got {other:?}"),
    }
}

/// The canonical zero test of a constant factor (`0·zoo = nan`) evaluates
/// it; a constant holding a special function at a parameter of `10⁵` or
/// more, or an orthogonal polynomial of degree 200 or more, is not tested
/// (as up to 0.33).  Before (0.34-dev, release): building the product with
/// `jacobi(999, 1/3, 1/5, 1/7)` took 2.8 s, `legendre(99999, 1/3)` 1.5 s,
/// `gegenbauer(999, 1/3, 1/3)` 0.75 s (the exact rational value in the
/// confirmation).  Below the bounds the test still decides `nan`.
#[test]
fn the_canonical_zero_test_has_a_bounded_cost() {
    let ctx = Context::new();
    let product = |f: &str| format!("(sin({f})^2 + cos({f})^2 - 1)*zoo");
    for f in [
        "jacobi(999, 1/3, 1/5, 1/7)",
        "legendre(99999, 1/3)",
        "gegenbauer(999, 1/3, 1/3)",
        "besselj(10^5, 1)",
        "besselj(8345185991999992, 61/10^27)",
    ] {
        let src = product(f);
        let _ = timed(&src, || ctx.parse(&src).unwrap());
    }
    for f in [
        "besselj(10^4, 1)",
        "legendre(150, 1/3)",
        "polygamma(3, 1/2)",
    ] {
        let src = product(f);
        let e = timed(&src, || ctx.parse(&src).unwrap());
        assert_eq!(e.to_string(), "nan", "{src}");
    }
}

/// Before: never finished (`fuzz_roundtrip`, more than 60 s each) — building
/// these through the API ran the canonical zero test of a constant factor,
/// which evaluated the Bessel function with a multiplication per unit of its
/// order (`8·10¹⁵` of them).  Construction alone, printed and re-parsed.
#[test]
fn building_huge_order_bessel_functions_is_quick() {
    let ctx = Context::new();
    let order171 = format!("39{}1", "0".repeat(168));
    for (src, printed) in [
        (
            "cos(besselj(8345185991999992, 61/10^27))/sin(besselj(8345185991999992, 61/10^27))"
                .to_string(),
            "cos(besselj(8345185991999992, 61/1000000000000000000000000000))/\
             sin(besselj(8345185991999992, 61/1000000000000000000000000000))"
                .to_string(),
        ),
        (
            "sin(besselj(655555555559150000000000, 29/5*10^49))^(-4)".to_string(),
            "sin(besselj(655555555559150000000000, \
             58000000000000000000000000000000000000000000000000))^(-4)"
                .to_string(),
        ),
        (
            format!("besselj({order171}, 1/10^78)/besselj(5, 1/10^78)"),
            String::new(),
        ),
    ] {
        let e = timed(&src, || ctx.parse(&src).unwrap());
        let s = e.to_string();
        if !printed.is_empty() {
            assert_eq!(s, printed);
        }
        let back = timed(&s, || ctx.parse(&s).unwrap());
        assert_eq!(back.to_string(), s);
    }
    // Through the API, as `fuzz_roundtrip` builds it.
    let b = ctx
        .parse("besselj(-10000000000, cosh(acos(-2/13)))")
        .unwrap();
    let z = ctx.parse("zoo").unwrap();
    let p = timed("besselj(-10^10, …)·zoo", || b.clone() * z);
    assert_eq!(p.to_string(), "zoo");
    let q = timed("sin(B)/cos(B)", || b.sin() / b.cos());
    assert_eq!(
        q.to_string(),
        "sin(besselj(-10000000000, cosh(acos(-2/13))))/cos(besselj(-10000000000, \
         cosh(acos(-2/13))))"
    );
}

/// Before: `besselj(8345185991999992, 61/10²⁷)` never finished (`eval_decimal`,
/// also at 0.32.0); `besselj(10⁸ + 1/2, 1)` printed `0` (a wrong value);
/// `besselj(10²⁰, −216)` never finished; a negative integer order beyond
/// `i64` (`besselj(−10²⁰, 1)`, `bessely(−10²⁴, 298)`) saturated an `i64`,
/// whose negation wrapped, and the reflection `J₋ₙ = (−1)ⁿ Jₙ` recursed until
/// the process aborted with a stack overflow (11 cases of the
/// extreme-parameter hunter).  All are nonzero below the exponent
/// range (`J_ν(x) ≈ (x/2)^ν/Γ(ν+1)`, no zero in `0 < x < ν`): mpmath
/// (`mp.dps=30`, the same at 45) `besselj(10**8 + mpf(1)/2, 1)` =
/// `1.186648592e-786673560`, beyond `2^(−2.1·10⁹) ≈ 10^(−6.5·10⁸)`; so are
/// `besseli(10¹², 10¹¹)` (its series would have taken `2.5·10⁹` terms) and
/// `besselj(−10¹⁰, cosh(acos(−2/13)))`.
#[test]
fn huge_order_bessel_values_below_the_range_are_refused_quickly() {
    let ctx = Context::new();
    for s in [
        "besselj(8345185991999992, 61/10^27)",
        "besselj(10^8 + 1/2, 1)",
        "besselj(-10^20, 1)",
        "besselj(10^20, -216)",
        "besselj(-10000000000, cosh(acos(-2/13)))",
        "besseli(10^12, 10^11)",
        "besseli(10^23, 199782498235300)",
    ] {
        nonzero_underflow(&ctx, s);
    }
    // Beyond the top of the range.
    for s in [
        "bessely(10^22, 95)",
        "bessely(-10^24, 298)",
        "besselk(-10^22, 1)",
    ] {
        match dec(&ctx, s) {
            Err(SymplexError::Unevaluable { reason }) if reason.contains("overflows") => {}
            other => panic!("{s}: want an overflow, got {other:?}"),
        }
    }
}

/// Before: `bessely(5000, 1)` 17 s, `besselk(5000, 1)`, `bessely(10⁵, 1)`,
/// `besselk(10⁵, 1)`, `bessely(10⁵, π)` never finished (the Neumann series
/// in exact rationals, `n` steps each reduced by a gcd); `besselj(10⁷, 1)`
/// 0.4 s.  mpmath (`mp.dps=30`, the same at 45):
/// `bessely(5000, 1)` = `-3.80254620909753046e+17826`,
/// `besselk(5000, 1)` = `5.97242822554816e+17826`,
/// `bessely(10**5, 1)` = `-8.980852880746014e+486670`,
/// `besselk(10**5, 1)` = `1.41070201805786e+486671`,
/// `bessely(10**5, pi)` = `-9.248212472049193e+436955`,
/// `besselk(10**5, mpf(9)/8)` = `7.892057869064266e+481555`,
/// `besselk(20000, mpf(1)/3)` = `4.817581101202995e+92895`,
/// `bessely(4698, mpf(1)/15392)` = `-3.539437993685697e+36294`,
/// `besselj(10**5, 1)` = `3.544316897586959e-486677`,
/// `besseli(10**5, 1)` = `3.544334619038536e-486677`,
/// `besselj(10**6, 3)` = `2.197191880888722e-5389618`,
/// `besselj(10**7, 1)` = `9.18973012565046024e-68667360`,
/// `besselj(-10**5 - mpf(1)/2, 1)` = `4.016354486958128e+486673`,
/// `besselj(10**7 + mpf(1)/2, 1000)` = `6.33768126694205882e-38667362`.
#[test]
fn large_order_bessel_values() {
    let ctx = Context::new();
    for (s, want) in [
        ("bessely(5000, 1)", "-3.80254620909753e17826"),
        ("besselk(5000, 1)", "5.97242822554816e17826"),
        ("bessely(10^5, 1)", "-8.98085288074601e486670"),
        ("besselk(10^5, 1)", "1.41070201805786e486671"),
        ("bessely(10^5, pi)", "-9.24821247204919e436955"),
        ("besselk(10^5, 9/8)", "7.89205786906427e481555"),
        ("besselk(20000, 1/3)", "4.817581101203e92895"),
        ("bessely(4698, 1/15392)", "-3.5394379936857e36294"),
        ("besselj(10^5, 1)", "3.54431689758696e-486677"),
        ("besseli(10^5, 1)", "3.54433461903854e-486677"),
        ("besselj(10^6, 3)", "2.19719188088872e-5389618"),
        ("besselj(10^7, 1)", "9.18973012565046e-68667360"),
        ("besselj(-10^5 - 1/2, 1)", "4.01635448695813e486673"),
        ("besselj(10^7 + 1/2, 1000)", "6.33768126694206e-38667362"),
    ] {
        assert_eq!(dec(&ctx, s).unwrap(), want, "{s}");
    }
}

/// Before (the nightly `fuzz_evalf` failure of 2026-10-05): `polygamma(1755,
/// 1/10)` 4.7 s in the fuzz build (32 s on CI), the `Piecewise` with
/// `polygamma(1779, 11)` timed out at 30 s; `polygamma(9000, 1/3)` never
/// finished, `polygamma(2758, 1000)` 0.9 s in release — `n!` and the
/// asymptotic series' coefficients in exact rationals, every product reduced
/// by a binary gcd with 1 (a step per bit).  mpmath (`mp.dps=30`, the same at
/// 50): `polygamma(1755, mpf(1)/10)` = `3.4736046171342926472e+6689`,
/// `polygamma(1779, 11)` = `6.2856038415847042373e+3157`,
/// `polygamma(9000, mpf(1)/3)` = `-2.998311610380396e+35976`,
/// `polygamma(9325, mpf(-27)/13)` = `1.581304781027893e+43358`,
/// `polygamma(2758, 1000)` = `-3.331687812232478e+16`.
#[test]
fn high_order_polygamma_is_quick() {
    let ctx = Context::new();
    for (s, want) in [
        ("polygamma(1755, 1/10)", "3.47360461713429e6689"),
        ("polygamma(1779, 11)", "6.2856038415847e3157"),
        ("polygamma(9000, 1/3)", "-2.9983116103804e35976"),
        ("polygamma(9325, -27/13)", "1.58130478102789e43358"),
        ("polygamma(2758, 1000)", "-33316878122324800"),
    ] {
        assert_eq!(dec(&ctx, s).unwrap(), want, "{s}");
    }
    // The nightly input: `polylog(2, 11)` is not real (mpmath
    // `0.3218540439999117 - 7.533210173101054j`), so the condition is not
    // decided; the evaluation must not take long either way.
    let a = ctx.parse("polygamma(1779, 11)").unwrap();
    let c = ctx.parse("polylog(2, 11)").unwrap();
    let d = ctx.parse("uppergamma(-3/2, 3)").unwrap();
    let b = ctx.int(-40);
    let t = ctx.bool_true();
    let pw = Ex::piecewise(&[(&a, &c.gt(&d)), (&b, &t)]);
    let _ = timed("the nightly Piecewise", || pw.eval_decimal(16));
}

/// Jacobi polynomials by their explicit sum.  Before: every term recomputed
/// its products and factorials, `O(n²)` operations at `2n` bits:
/// `jacobi(1100, 1/3, 1/5, sqrt(2)/5)` 6.1 s (release; 0.05 s now),
/// `jacobi(1700, …)` and up more than 20 s.  mpmath (`mp.dps=30`, the same
/// at 60): `jacobi(1100, mpf(1)/3, mpf(1)/5, sqrt(2)/5)` =
/// `-0.000813612976462495`, `jacobi(1100, mpf(1)/3, mpf(1)/5, -1)` =
/// `4.4198079703905`.  Beyond degree 2000 it is refused.
#[test]
fn jacobi_of_large_degree() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "jacobi(1100, 1/3, 1/5, sqrt(2)/5)").unwrap(),
        "-0.000813612976462495"
    );
    assert_eq!(
        dec(&ctx, "jacobi(1100, 1/3, 1/5, -1)").unwrap(),
        "4.4198079703905"
    );
    match dec(&ctx, "jacobi(9999, 1/3, 1/5, sqrt(2)/5)") {
        Err(SymplexError::NotImplemented(_)) => {}
        other => panic!("jacobi(9999, …): want a refusal, got {other:?}"),
    }
}

/// `Li₋ₙ(z) = Σ k!·S(n+1, k+1)·(z/(1−z))^{k+1}`.  Before: each Stirling
/// number computed from scratch (`O(n³)`): `polylog(−601, 4/13)` 18 s,
/// `polylog(−1401, 4/13)` never finished (the extreme-parameter hunter),
/// `polylog(−1000, −1/2)` 32 s — and wrong, `−1.84·10²³⁷⁶`: for `z < 0`
/// the terms alternate and cancel, and the loss went unmeasured
/// (`polylog(−200, −1/2)` was `2.03·10²⁹⁶`, `polylog(−100, −1/2)`
/// `6.05583168868966·10¹⁰⁵`).  mpmath (`mp.dps=1500`, the same at 2000):
/// `polylog(-1000, -mpf(1)/2)` = `-4.796943067035934e+2059`,
/// `polylog(-200, -mpf(1)/2)` = `-5.147757721068388e+272`,
/// `polylog(-100, -mpf(1)/2)` = `6.055831688762091e+105`,
/// `polylog(-301, -mpf(1)/3)` = `-1.763888548977255e+459`,
/// `polylog(-601, mpf(4)/13)` = `8.06059538480115e1367` (dps 600, 900),
/// `polylog(-1000, mpf(4)/13)` = `1.401380266870968e+2496`,
/// `polylog(-1000, 3+1j)` = `8.856958502112417e+2489 + 5.187343015233016e+2489j`.
/// Beyond an order of −1000 the finite sum is refused (since 0.35 the poles
/// serve there, to −10⁶; beyond, still refused).
#[test]
fn polylog_of_large_negative_order() {
    let ctx = Context::new();
    for (s, want) in [
        ("polylog(-1000, -1/2)", "-4.79694306703593e2059"),
        ("polylog(-200, -1/2)", "-5.14775772106839e272"),
        ("polylog(-100, -1/2)", "6.05583168876209e105"),
        ("polylog(-301, -1/3)", "-1.76388854897726e459"),
        ("polylog(-601, 4/13)", "8.06059538480115e1367"),
        ("polylog(-1000, 4/13)", "1.40138026687097e2496"),
        (
            "polylog(-1000, 3 + I)",
            "8.85695850211242e2489 + 5.18734301523302e2489*i",
        ),
    ] {
        assert_eq!(dec(&ctx, s).unwrap(), want, "{s}");
    }
    // Since 0.35 an order below −1000 (up to −10⁶) takes the poles of
    // `Li_{−n}` (Jonquière's formula) instead of the Stirling sum: a value,
    // not a refusal.  Reference: the exact rational `z·A_n(z)/(1 − z)^(n+1)`
    // (Eulerian numbers, Python fractions) = `3.9942724806637844986e+3701`.
    assert_eq!(
        dec(&ctx, "polylog(-1401, 4/13)").unwrap(),
        "3.99427248066378e3701"
    );
    let s = "polylog(-10^12, 1/3)";
    match dec(&ctx, s) {
        Err(SymplexError::NotImplemented(_)) => {}
        other => panic!("{s}: want a refusal, got {other:?}"),
    }
}

/// A local `fuzz_roundtrip` run before 0.34 (a 9.5 s "slow unit"):
/// building `−sin((−10)^10321809999995599999)·∞` took 0.8 s in a release
/// build.  Orienting `x·∞` asks whether the 20-digit exponent is prime, and
/// the assumption system tested it by trial division up to `√n` (half a
/// billion divisions); it now uses the deterministic Miller–Rabin test for
/// `u64`.  SymPy: `isprime(10321809999995599999)` → `False`.
#[test]
fn orienting_an_infinity_with_a_huge_exponent_is_quick() {
    let ctx = Context::new();
    for s in [
        "-sin((-10)^10321809999995599999)*oo",
        "sin(10^18446744073709551557)*oo",
    ] {
        timed(s, || ctx.parse(s).unwrap());
    }
}
