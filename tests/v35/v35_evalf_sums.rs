//! Numerical evaluation of sums, products and the zeta/Bernoulli family
//! (the evalf hunt after 0.43).  Every reference value cites its oracle:
//! mpmath 1.3 at 45–60 digits (`nsum`, `nprod`, `zeta`, `qp`, `bernfrac`,
//! `eulernum`), scipy 1.18 for the entropies, SymPy 1.14 for `euler(s)`,
//! or a closed form.

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use symplex::ntheory as nt;
use symplex::prelude::*;
use symplex::stats::Distribution;

fn dec(ctx: &Context, s: &str, digits: u32) -> String {
    ctx.parse(s)
        .unwrap()
        .eval_decimal(digits)
        .unwrap_or_else(|e| panic!("{s}: {e:?}"))
}

/// A positive decimal string as its significant digits and the decimal
/// exponent of the first one.
fn digits_of(s: &str) -> (String, i32) {
    let (mant, exp) = match s.find('e') {
        Some(i) => (&s[..i], s[i + 1..].parse::<i32>().unwrap()),
        None => (s, 0),
    };
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let all = format!("{int}{frac}");
    let lead = all.len() - all.trim_start_matches('0').len();
    (
        all.trim_start_matches('0').to_string(),
        int.len() as i32 - lead as i32 + exp,
    )
}

/// `got` (a positive decimal string of at most `digits` significant digits)
/// is the reference `want` (more digits) rounded to `digits`, give or take
/// one unit in the last digit.
fn assert_digits(got: &str, want: &str, digits: usize) {
    let (gd, ge) = digits_of(got);
    let (wd, we) = digits_of(want);
    assert_eq!(ge, we, "{got} vs {want}: exponent");
    let g: BigInt = format!("{gd:0<digits$}").parse().unwrap();
    let w: BigInt = wd[..digits + 1].parse().unwrap();
    let w: BigInt = (w + 5) / 10;
    let gap: BigInt = g - w;
    assert!(
        gap.abs() <= BigInt::from(1),
        "{got} vs {want} at {digits} digits"
    );
}

/// The entropy of a Poisson law is the infinite sum `−Σ p(k)·ln p(k)` of a
/// hypergeometric term times `ln(λᵏ/k!)`.  Before: refused ("the term is
/// not recognised as hypergeometric").  scipy 1.18
/// `stats.poisson(3.5).entropy()` = 2.015172522512972; mpmath `nsum` (45
/// digits) = 2.01517252251297228119170325756766848215.
#[test]
fn poisson_entropy_is_evaluated() {
    let ctx = Context::new();
    let h = Distribution::poisson(ctx.rational(7, 2)).entropy();
    let v = h.eval_f64().unwrap();
    assert!((v - 2.015172522512972).abs() < 1e-15, "{v}");
    assert_digits(
        &h.eval_decimal(30).unwrap(),
        "2.01517252251297228119170325756766848215",
        30,
    );
}

/// The entropy of a negative binomial law: `binomial(k + 2, k)·(2/3)ᵏ/27`
/// times its logarithm.  Before: refused as the Poisson one.  mpmath `nsum`
/// (45 digits) = 2.73009804463369354314606216805164281318 (scipy 1.18
/// `stats.nbinom(3, 1/3).entropy()` = 2.7300980446336953, two units off in
/// its last digit).
#[test]
fn negative_binomial_entropy_is_evaluated() {
    let ctx = Context::new();
    let h = Distribution::negative_binomial(ctx.int(3), ctx.rational(1, 3)).entropy();
    assert_digits(
        &h.eval_decimal(25).unwrap(),
        "2.73009804463369354314606216805164281318",
        25,
    );
}

/// Infinite sums of a hypergeometric term (or an exponential `e^(−k)`,
/// `√2^(−k)`) times factors of polynomial growth (`ln k`, `sin k`, `H_k`).
/// Before: each refused ("not recognised as hypergeometric").  mpmath: the
/// first 400 terms by `fsum`, `nsum` beyond, at 50 digits; `Σ H_k/2^k = 2
/// ln 2`.  A complex ratio: `Σ (i/2)^k·ln(k + 1)` = −0.197731604816482057194
/// + 0.216572592825321830816i.
#[test]
fn hypergeometric_terms_times_slow_factors() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "Sum(exp(-k)*log(k), k, 1, oo)",
            "0.192092805519242420451283200786137663872173",
        ),
        (
            "Sum(sqrt(2)^(-k)*log(k), k, 1, oo)",
            "2.25550946397907785589370098497093604606847",
        ),
        (
            "Product(1 + exp(-k), k, 0, oo)",
            "3.35585736997870839487814473967369369364167",
        ),
        (
            "Sum(log(k)/2^k, k, 1, oo)",
            "0.507833922868438392189041840722076374246218433",
        ),
        (
            "Sum(sin(k)/factorial(k), k, 0, oo)",
            "1.27988300137302249390846230120139188823563891",
        ),
        (
            "Sum(harmonic(k)/2^k, k, 1, oo)",
            "1.38629436111989061883446424291635313615100027",
        ),
    ] {
        assert_digits(&dec(&ctx, s, 20), want, 20);
        assert_digits(&dec(&ctx, s, 40), want, 40);
    }
    assert_eq!(
        dec(&ctx, "Sum((I/2)^k*log(k + 1), k, 0, oo)", 20),
        "-0.19773160481648205719 + 0.21657259282532183082*i"
    );
}

/// Sums of a non-integer power of a linear term, which `eval` leaves as
/// sums: Hurwitz zeta values.  Before: refused ("not recognised as
/// hypergeometric").  mpmath 1.3 (50 digits): `zeta(5/2)`, `(1 −
/// √2)·zeta(1/2)`, `3·(1 − 2^(−3/2))·zeta(3/2)`.
#[test]
fn powers_of_linear_terms_are_hurwitz_zeta_values() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "Sum(k^(-5/2), k, 1, oo)",
            "1.34148725725091717975676969334861213662304",
        ),
        (
            "Sum((-1)^k/sqrt(k + 1), k, 0, oo)",
            "0.604898643421630370247265914235955499759763",
        ),
        (
            "Sum(3/(2*k + 1)^(3/2), k, 0, oo)",
            "5.0662835599663444330737544897913247151249",
        ),
    ] {
        assert_digits(&dec(&ctx, s, 40), want, 40);
    }
}

/// Infinite products.  Before: every one refused ("cannot evaluate infinity
/// to finite precision").  References: `∏(1 + 1/k²) = sinh(π)/π`; mpmath
/// `qp(4/5)` = 0.00336800585242312126602925411217685656047; mpmath `nprod` at
/// 45 digits for `∏(1 + 1/k!)` and `∏_{k≥2}(1 − 1/k³)`; for `∏_{k≥2} (k² +
/// 1/2)/(k² + 3/2)` the Gamma formula `|Γ(2 + i√(3/2))|²/|Γ(2 + i/√2)|²`
/// (mpmath, 50 digits).
#[test]
fn infinite_products_are_evaluated() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "Product(1 + 1/k^2, k, 1, oo)",
            "3.67607791037497772069569749202826066650715635",
        ),
        (
            "Product(1 - (4/5)^k, k, 1, oo)",
            "0.00336800585242312126602925411217685656047389109",
        ),
        (
            "Product(1 + 1/factorial(k), k, 0, oo)",
            "7.36430827236725725637277250963105309565425684",
        ),
        (
            "Product(1 - 1/k^3, k, 2, oo)",
            "0.809396597366290109578680478726382119372787648",
        ),
        (
            "Product((k^2 + 1/2)/(k^2 + 3/2), k, 2, oo)",
            "0.56132630781694555193036643410895060246971162486313",
        ),
    ] {
        assert_digits(&dec(&ctx, s, 20), want, 20);
        assert_digits(&dec(&ctx, s, 40), want, 40);
    }
    // A factor that vanishes in the range: exactly 0.
    assert_eq!(dec(&ctx, "Product(1 - 4/k^2, k, 1, oo)", 20), "0");
    // Factors tending to 1 like 1 + c/k: no limit.
    let p = ctx
        .parse("Product(1 + 1/(k^2 + k)^(1/2), k, 1, oo)")
        .unwrap();
    assert!(p.eval_decimal(20).is_err());
}

/// The Euler function off the integers (SymPy's `euler(s)`, a combination
/// of Hurwitz zeta functions).  Before: refused ("evalf has no numerical
/// routine for 'euler_number'").  SymPy 1.14 `N(euler(s), 40)`:
/// `euler(1/2)` = 0.5503594824576405002333303353125779118239,
/// `euler(−1/2)` = 1.335382914379218353317381858600496964503,
/// `euler(301/4)` = 6.223115267055761662562108590336202573975e94,
/// `euler(−7/3)` = 1.878761399461173366968087824823992283897; mpmath (the same
/// formula, 60 digits) next to the zero at 3 and the cancelling poles at −1.
#[test]
fn euler_function_off_the_integers() {
    let ctx = Context::new();
    for (s, want) in [
        (
            "euler_number(1/2)",
            "0.5503594824576405002333303353125779118239",
        ),
        (
            "euler_number(-1/2)",
            "1.335382914379218353317381858600496964503",
        ),
        (
            "euler_number(301/4)",
            "6.223115267055761662562108590336202573975e94",
        ),
        (
            "euler_number(-7/3)",
            "1.878761399461173366968087824823992283897",
        ),
        (
            "euler_number(3 + 10^(-20))",
            "3.061918009248316088765436893323052506517e-20",
        ),
        (
            "euler_number(-1 + 10^(-10))",
            "1.5707963267563163558703977885227885028392",
        ),
    ] {
        assert_digits(&dec(&ctx, s, 39), want, 39);
    }
    assert_eq!(dec(&ctx, "euler_number(-1)", 20), "1.5707963267948966192");
}

/// `ζ(s)` left of 0 next to a trivial zero, at an exactly representable
/// argument: `sin(πs/2)` of a rounded `πs/2` lost `log₂(1/|s + 2|)` bits.
/// Before: `zeta(−2 − 2^(−150))` at 40 digits ended `…6427903386290 e−47`,
/// wrong from its 33rd digit.  mpmath (dps 60, exact input) =
/// 2.13336880584933136440784064279034111770157273e-47.
#[test]
fn zeta_next_to_a_trivial_zero() {
    let ctx = Context::new();
    assert_eq!(
        dec(&ctx, "zeta(-2 - 2^(-150))", 40),
        "2.133368805849331364407840642790341117702e-47"
    );
}

/// `B_n` and `E_n` beyond index 4096.  Before: `None` (the tangent and
/// secant numbers cost `O(n³)` bit operations; the table to `B₄₀₉₆` took
/// 44 s in a debug build).  Now one number by the zeta function: `B₄₀₉₈`
/// 0.14 s, `B₂₀₀₀₀` 6.2 s, `B₃₀₀₀₀` (the bound) 15 s, `E₂₀₀₀₀` 18 s, `E₃₀₀₀₀` 46
/// s (debug build; mpmath `bernfrac(20000)`: 1.4 s CPU).  mpmath 1.3
/// `bernfrac(4098)`: denominator 235339986, a positive numerator of 9765
/// digits starting 32967867009440868396 and ending 87626953057117446109;
/// `eulernum(4098, exact=True)`: negative, 12224 digits, starting
/// 15563945854957606391, ending 95790152543853424641.
#[test]
fn bernoulli_and_euler_numbers_beyond_4096() {
    let b = nt::bernoulli(4098).unwrap();
    assert_eq!(b.denom(), &BigInt::from(235_339_986u64));
    let s = b.numer().to_string();
    assert_eq!(s.len(), 9765);
    assert!(s.starts_with("32967867009440868396"));
    assert!(s.ends_with("87626953057117446109"));
    let e = nt::euler_number(4098).unwrap();
    assert!(e.is_negative());
    let s = e.abs().to_string();
    assert_eq!(s.len(), 12224);
    assert!(s.starts_with("15563945854957606391"));
    assert!(s.ends_with("95790152543853424641"));
    assert_eq!(nt::euler_number(4099), Some(BigInt::from(0)));
    // The work bound moved, it did not go away.
    assert_eq!(nt::bernoulli(30_002), None);
    assert_eq!(nt::euler_number(30_002), None);
    // B₁₀₀₀₀ = −N/2338224387510, N of 27691 digits (mpmath `bernfrac`).
    let b = nt::bernoulli(10_000).unwrap();
    assert_eq!(b.denom().to_u64(), Some(2_338_224_387_510));
    assert!(b.numer().is_negative());
    assert_eq!(b.numer().abs().to_string().len(), 27691);
}
