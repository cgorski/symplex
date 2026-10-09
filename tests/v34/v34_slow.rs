//! Slowness on the paths that *build* expressions from huge integers (the
//! construction-path hunt after 0.35: radicals, powers, products and
//! trigonometric special values of integers with thousands of digits).
//! Every call must finish well within its limit with the value it had
//! before (or SymPy's, where it changed).

use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::Pow;
use symplex::prelude::*;

/// Generous for a debug build on a loaded CI machine: the release times
/// are milliseconds.
const LIMIT: Duration = Duration::from_secs(5);

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed();
    assert!(dt < LIMIT, "{what} took {dt:?}");
    r
}

/// Before: 2.9 s of CPU in a release build (a 13,000-digit `v`: more than
/// a minute); now 20 ms.  Splitting `k`-th powers out of `v²` found the perfect square
/// and ran a BPSW primality test on its base, a modular exponentiation at
/// the full size of `v`, although for `k = 5` the exponent 2 extracts
/// nothing whatever the answer.  Bases over 2,048 bits are no longer
/// tested.  The radical keeps its form; its fifth power is `(v²)³`, the
/// power the digit guard leaves unevaluated (SymPy: `perfect_power(v**2)`
/// → `(v, 2)`, so `(v**2)**(3/5)` is `v**(6/5)`, the same number).
#[test]
fn radicals_of_perfect_powers_with_huge_bases_are_quick() {
    let ctx = Context::new();
    let v: BigInt = Pow::pow(BigInt::from(10), 5999u32) + 3;
    let v2 = &v * &v;
    let r = timed("(v^2)^(3/5)", || {
        ctx.from_bigint(v2.clone()).pow(&ctx.rational(3, 5))
    });
    assert_eq!(r.to_string(), format!("{v2}^(3/5)"));
    assert_eq!(r.powi(5), ctx.from_bigint(v2).powi(3));
}

/// A perfect power of a composite base over 2,048 bits now leaves the
/// radical, as SymPy writes it: `p = 2**1279 - 1` and `q = 2**2203 - 1`
/// are prime (`isprime` → `True`), and SymPy gives `sqrt(3*(p*q)**4)` →
/// `(p*q)**2*sqrt(3)` and `cbrt(5*(p*q)**3)` → `p*q*5**(1/3)` (its
/// `factorint` with a limit lists the base of a perfect-power remainder with
/// its exponent, prime or not).  Before, the base was tested for primality
/// (a BPSW test, see above) and, composite, stayed inside: `sqrt(3·(p·q)⁴)`
/// was its own canonical form.
#[test]
fn a_perfect_power_of_a_huge_composite_leaves_the_radical() {
    let ctx = Context::new();
    let p: BigInt = (BigInt::from(1) << 1279u32) - 1;
    let q: BigInt = (BigInt::from(1) << 2203u32) - 1;
    let b = &p * &q;
    let r = timed("sqrt(3*(p*q)^4)", || {
        ctx.from_bigint(Pow::pow(&b, 4u32) * 3).sqrt()
    });
    assert_eq!(r, &ctx.from_bigint(Pow::pow(&b, 2u32)) * &ctx.int(3).sqrt());
    let c = timed("cbrt(5*(p*q)^3)", || {
        ctx.from_bigint(Pow::pow(&b, 3u32) * 5).cbrt()
    });
    assert_eq!(c, &ctx.from_bigint(b.clone()) * &ctx.int(5).cbrt());
    // A prime base is extracted as before (`isprime(2**4423 - 1)` → `True`).
    let m: BigInt = (BigInt::from(1) << 4423u32) - 1;
    let r = timed("sqrt(7*m^4)", || {
        ctx.from_bigint(Pow::pow(&m, 4u32) * 7).sqrt()
    });
    assert_eq!(r, &ctx.from_bigint(Pow::pow(&m, 2u32)) * &ctx.int(7).sqrt());
}

/// Before: 2.1 s of CPU in a release build (`simplify` of the Pythagorean
/// identity at `q = 3⁸⁰⁰⁰/7⁴⁰⁰⁰`, numbers of 3,818 and 3,381 digits);
/// now 60 ms.
/// `eval` reduced `q` modulo 2 and through the quadrants with `Ratio`
/// arithmetic — gcds quadratic in the digits, an interned number per step
/// — at every one of the hundreds of `sin`/`cos` evaluations of the
/// simplifier, although no special value has a denominator outside the
/// divisors of 12.  The special values at huge numerators still fold:
/// `sin(π/6 + kπ) = (−1)ᵏ/2`, `cos(π/3 + kπ) = (−1)ᵏ/2`, `tan(3π/4 + kπ)
/// = −1`, `exp((k + 1/2)πi) = (−1)ᵏ·i` (`k = 10⁴⁰⁰⁰`, even; SymPy:
/// `sin((6*10**4000 + 1)*pi/6)` → `1/2`).
#[test]
fn special_values_at_huge_rational_multiples_of_pi_are_quick() {
    let ctx = Context::new();
    let a: BigInt = Pow::pow(BigInt::from(3), 8_000u32);
    let b: BigInt = Pow::pow(BigInt::from(7), 4_000u32);
    let src = format!("sin({a}*pi/{b})^2 + cos({a}*pi/{b})^2");
    let e = ctx.parse(&src).unwrap();
    let s = timed("sin(q pi)^2 + cos(q pi)^2", || e.simplify());
    assert_eq!(s.to_string(), "1");

    let k: BigInt = Pow::pow(BigInt::from(10), 4000u32);
    for (src, want) in [
        (format!("sin(({})/6*pi)", &k * 6 + 1), "1/2"),
        (format!("cos(({})/3*pi)", &k * 3 + 1), "1/2"),
        (format!("tan(({})/4*pi)", &k * 4 + 3), "-1"),
        (format!("exp(({})/2*pi*I)", &k * 2 + 1), "I"),
    ] {
        let v = timed(&src, || ctx.parse(&src).unwrap().eval());
        assert_eq!(v.to_string(), want, "{src}");
    }
}

/// Before: 22 s (release; a timeout of the local `fuzz_evalf` run before
/// 0.38).  `erfc` at an argument beyond the `f64` range (`Shi(5040) ≈
/// 6.9·10²¹⁸⁴`, known to its relative precision): the bound on `erfc′` over
/// the argument's ball was computed in `f64`, where `|x| − r` was `∞ − ∞`,
/// so the ball reached 0 and the bound was `2/√π` instead of
/// `e^(−10⁴³⁶⁹)`; the evaluation searched to the maximum precision.  The
/// value is below the exponent range (mpmath: `erfc(shi(5040))` underflows
/// to `0.0` at every precision, `log` of it ≈ `−4.8·10⁴³⁶⁹`): refused as such.
#[test]
fn erfc_of_an_argument_beyond_the_f64_range_is_quick() {
    let ctx = symplex::prelude::Context::new();
    let t = std::time::Instant::now();
    let r = ctx.parse("erfc(Shi(5040))").unwrap().eval_decimal(16);
    assert!(
        matches!(r, Err(symplex::prelude::SymplexError::Unevaluable { ref reason }) if reason.contains("underflows")),
        "{r:?}"
    );
    assert_eq!(
        ctx.parse("erf(Shi(5040))")
            .unwrap()
            .eval_decimal(16)
            .unwrap(),
        "1"
    );
    assert!(t.elapsed().as_secs_f64() < 5.0, "{:?}", t.elapsed());
}
