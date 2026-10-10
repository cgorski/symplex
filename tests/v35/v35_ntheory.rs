//! Number theory, Diophantine equations and combinatorial numbers after the
//! 0.40 hunt.  Every test names what was wrong before and the oracle of its
//! reference value: SymPy 1.14 (`sympy.ntheory`, `diop_DN`, the functions
//! of `sympy.functions.combinatorial.numbers`, `evalf(20)`), or — where
//! SymPy hangs or raises — a check by substitution or a classical identity.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use symplex::combinatorics as cb;
use symplex::diophantine as dp;
use symplex::ntheory as nt;
use symplex::prelude::*;

fn b(v: i64) -> BigInt {
    BigInt::from(v)
}

fn big(s: &str) -> BigInt {
    s.parse().unwrap()
}

fn pow2(k: usize) -> BigInt {
    BigInt::one() << k
}

/// `x² ≡ a` modulo a high prime power has its roots in residue classes,
/// which are no longer listed one by one.  Before: `sqrt_mod(0, 2¹⁰⁰)`
/// built the `2⁵⁰` roots (the process aborted on the allocation),
/// `is_quad_residue(0, p⁴⁰)` answered `false` once the root count passed
/// `u64`.  SymPy 1.14: `sqrt_mod(0, 2**100)` = 0, `sqrt_mod(4, 2**100)` =
/// 2, `is_quad_residue(0, 1000003**40)` = True, `is_quad_residue(4 *
/// 1000003**2, 1000003**40)` = True, `sqrt_mod(0, 2**127 - 1,
/// all_roots=True)` = [0].
#[test]
fn square_roots_modulo_high_prime_powers() {
    assert_eq!(nt::sqrt_mod(0, pow2(100)), Some(b(0)));
    assert_eq!(nt::sqrt_mod(4, pow2(100)), Some(b(2)));
    let p40 = b(1_000_003).pow(40u32);
    assert!(nt::is_quad_residue(0, p40.clone()));
    assert!(nt::is_quad_residue(
        b(4) * b(1_000_003).pow(2u32),
        p40.clone()
    ));
    assert!(!nt::is_quad_residue(b(1_000_003), p40));
    // 2⁵⁰ roots: refused by the `try_` form, an empty list from the old one.
    assert!(nt::try_sqrt_mod_all(0, pow2(100)).is_err());
    assert!(nt::sqrt_mod_all(0, pow2(100)).is_empty());
    assert_eq!(nt::sqrt_mod_all(0, pow2(127) - 1), vec![b(0)]);
    // x² ≡ 0 (mod 2²⁰) ⇔ x ≡ 0 (mod 2¹⁰): 1024 roots.
    let roots = nt::try_sqrt_mod_all(0, pow2(20)).unwrap();
    assert_eq!(roots.len(), 1024);
    assert!(roots.iter().all(|r| (r * r % pow2(20)).is_zero()));
    assert_eq!(roots[1], b(1024));
}

/// `sqrt_mod` returns the smallest root; beyond its combination search
/// (`2⁴⁴` candidates modulo the 45 odd primes below 200) it tries the
/// small `x` in turn.  Before: `None` there.  SymPy 1.14 returns another
/// root for both moduli (`sqrt_mod(4, 3·5·…·83)` =
/// 44510752614879308559270669665467); `2² = 4` and `7² = 49` are checked
/// by substitution, and nothing smaller squares to them.
#[test]
fn sqrt_mod_returns_the_smallest_root() {
    let odd_primes_below = |bound: u32| -> BigInt {
        (3..bound)
            .filter(|&p| (2..p).all(|d| p % d != 0))
            .map(BigInt::from)
            .product()
    };
    assert_eq!(nt::sqrt_mod(4, odd_primes_below(84)), Some(b(2)));
    assert_eq!(nt::sqrt_mod(49, odd_primes_below(200)), Some(b(7)));
    assert_eq!(nt::sqrt_mod(1, odd_primes_below(200)), Some(b(1)));
}

/// `xⁿ ≡ a` modulo prime powers, kept as classes.  Before:
/// `nthroot_mod(0, 5, 2³⁷, false)` enumerated the `2²⁹` roots (a hang; SymPy
/// 1.14 does not finish within 20 s either — the answer 0 is immediate).
/// SymPy 1.14: `nthroot_mod(2**10, 5, 2**37)` = 4, `nthroot_mod(8, 3,
/// 3**30, True)` = [2, 68630377364885, 137260754729768]; `nthroot_mod(0,
/// 3, 3**30, True)` (3²⁰ roots) does not finish, and is refused here.
#[test]
fn nth_roots_modulo_prime_powers() {
    assert_eq!(nt::nthroot_mod(0, 5, pow2(37), false), Some(vec![b(0)]));
    assert_eq!(nt::nthroot_mod(1024, 5, pow2(37), false), Some(vec![b(4)]));
    let m = b(3).pow(30u32);
    assert_eq!(
        nt::nthroot_mod(8, 3, m.clone(), true),
        Some(vec![b(2), b(68_630_377_364_885), b(137_260_754_729_768)])
    );
    assert_eq!(nt::nthroot_mod(0, 3, m.clone(), true), None);
    assert_eq!(nt::nthroot_mod(0, 3, m, false), Some(vec![b(0)]));
}

/// Polynomial congruences with root classes.  Before: `(x − 19)² ≡ 0 (mod
/// 2·10³²)` aborted the process on a 1.26·10¹⁷-byte allocation, and root
/// sets beyond `u64` came back empty.  SymPy 1.14:
/// `polynomial_congruence((x - 3)**2, 1024)` has 32 roots starting [3, 35,
/// 67]; `polynomial_congruence(x**3 - x, 2**10 * 3**5)` has 15 roots,
/// sorted starting [0, 1, 14336, 14337, 28673, 95743].
#[test]
fn polynomial_congruences_with_root_classes() {
    let c = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| b(t)).collect() };
    let roots = nt::try_polynomial_congruence(&c(&[1, -6, 9]), 1024).unwrap();
    assert_eq!(roots.len(), 32);
    assert_eq!(roots[..3], [b(3), b(35), b(67)]);
    let roots = nt::polynomial_congruence(&c(&[1, 0, -1, 0]), 1024 * 243);
    assert_eq!(roots.len(), 15);
    assert_eq!(
        roots[..6],
        [b(0), b(1), b(14336), b(14337), b(28673), b(95743)]
    );
    let m = b(2) * b(10).pow(32u32);
    assert!(nt::try_polynomial_congruence(&c(&[1, -38, 361]), m.clone()).is_err());
    assert!(nt::polynomial_congruence(&c(&[1, -38, 361]), m).is_empty());
}

/// `aˣ ≡ b (mod n)` with `gcd(a, n) > 1` and `n > 10⁶`.  Before: `None`
/// (only `n ≤ 10⁶` was searched).  SymPy 1.14 raises `ValueError` (base
/// not invertible); by substitution `4⁸⁶ ≡ 96896453306 (mod
/// 107546929490)`, no `x ≤ 37` works, and beyond `x = 37` the solutions are
/// `86 + 1260t` (SymPy on the odd part `n₂ = 53773464745`:
/// `discrete_log(n₂, b, 4)` = 86, `n_order(4, n₂)` = 1260).
#[test]
fn discrete_log_with_a_base_sharing_a_factor_with_the_modulus() {
    let n = b(107_546_929_490);
    let target = b(96_896_453_306);
    assert_eq!(b(4).modpow(&b(86), &n), target);
    assert_eq!(nt::discrete_log(4, target, n.clone()), Some(b(86)));
    // 3 is odd: 4ˣ is even for x ≥ 1 and 1 for x = 0.
    assert_eq!(nt::discrete_log(4, 3, n), None);
}

/// A discrete logarithm in a subgroup of prime order `q ≈ 10¹⁵`, beyond the
/// baby-step table bound (`2²¹` entries, `q` up to about `4·10¹²`).
/// Before: a table of `√q ≈ 3·10⁷` `BigInt` keys was built (gigabytes,
/// tens of seconds) for any answer.  Now a solution below `2²⁰` is found
/// by trial multiplication and a larger one is refused (`None`, documented
/// on `discrete_log`).  SymPy 1.14 (24 s each): `n_order(3, p)` =
/// 1000000000000223 for the safe prime `p = 2000000000000447`,
/// `discrete_log(p, 1477367204687310, 3)` = 1000,
/// `discrete_log(p, 1278647550151311, 3)` = 123456789012.
#[test]
fn discrete_log_beyond_the_baby_step_table() {
    let p = b(2_000_000_000_000_447);
    assert_eq!(nt::n_order(3, p.clone()), Some(b(1_000_000_000_000_223)));
    assert_eq!(
        nt::discrete_log(3, 1_477_367_204_687_310i64, p.clone()),
        Some(b(1000))
    );
    assert_eq!(nt::discrete_log(3, 1_278_647_550_151_311i64, p), None);
}

/// Orders modulo `p²` for a 28-digit prime `p`.  Before: `φ(p²) = p(p − 1)`
/// was factored as one number (a 54-digit composite with two 27-digit-plus
/// prime factors) and every one of these calls hung.  SymPy 1.14 with `p =
/// nextprime(10**27)`: `n_order(2, p**2)` =
/// 500000000000000000000000102500000000000000000000005253, `n_order(p**2 -
/// 1, p**2)` = 2, `primitive_root(p**2)` = 5, `is_primitive_root(2, p**2)`
/// = False.
#[test]
fn orders_modulo_the_square_of_a_large_prime() {
    let p = big("1000000000000000000000000103");
    let n = &p * &p;
    assert_eq!(
        nt::n_order(2, n.clone()),
        Some(big(
            "500000000000000000000000102500000000000000000000005253"
        ))
    );
    assert_eq!(nt::n_order(&n - 1, n.clone()), Some(b(2)));
    assert_eq!(nt::primitive_root(n.clone()), Some(b(5)));
    assert!(!nt::is_primitive_root(2, n));
}

/// `divisor_count` of a number with 70 distinct prime factors.  Before: the
/// `usize` product overflowed (a panic in a debug build, a wrapped value in
/// a release one).  SymPy 1.14: `divisor_count(primorial(70))` =
/// 1180591620717411303424 = 2⁷⁰.
#[test]
fn divisor_count_saturates() {
    assert_eq!(nt::divisor_count(nt::primorial(70)), usize::MAX);
    assert_eq!(nt::divisor_sigma(nt::primorial(70), 0), pow2(70));
    assert_eq!(nt::divisor_count(nt::primorial(10)), 1024);
}

/// Known Mersenne exponents below GIMPS's verified bound come from the
/// table.  Before: the Lucas–Lehmer test ran (18 s for `p = 44497`, no end
/// for `p = 1000003`).  SymPy 1.14: `is_mersenne_prime(2**44497 - 1)` =
/// True, `is_mersenne_prime(2**1000003 - 1)` = False.
#[test]
fn mersenne_primes_below_the_verified_bound() {
    assert!(nt::is_mersenne_prime(44497));
    assert!(nt::is_mersenne_prime(82_589_933));
    assert!(!nt::is_mersenne_prime(1_000_003));
    assert!(!nt::is_mersenne_prime(44501));
}

/// A remainder above 2,048 bits is not BPSW-tested by `factorint_bounded`:
/// `3·(2³²¹⁷ − 1)` keeps the Mersenne prime as its cofactor (the listed
/// factors stay prime).  Before: the test ran at any size (a minute at
/// 10,000 digits).
#[test]
fn factorint_bounded_leaves_a_huge_remainder_as_cofactor() {
    let m = pow2(3217) - 1;
    let (factors, cofactor) = nt::factorint_bounded(&(b(3) * &m), 64);
    assert_eq!(factors, vec![(b(3), 1)]);
    assert_eq!(cofactor, m);
}

/// Pell equations from one product tree of the period.  Before: every
/// convergent was tested against the equation (`pell(999999937)` took 14
/// s).  SymPy 1.14: `diop_DN(999999937, 1)` has an 88557-bit `x`;
/// `diop_DN(61, 1)` = [(1766319049, 226153980)], `diop_DN(61, -1)` =
/// [(29718, 3805)].
#[test]
fn pell_equations_with_long_periods() {
    let d = b(999_999_937);
    let (x, y) = dp::pell(d.clone()).unwrap();
    assert_eq!(x.bits(), 88557);
    assert_eq!(&x * &x - &d * &y * &y, b(1));
    assert_eq!(dp::pell(61), Some((b(1_766_319_049), b(226_153_980))));
    assert_eq!(dp::pell_negative(61), Some((b(29718), b(3805))));
    assert_eq!(dp::pell_negative(3), None);
    for (x, y) in dp::pell_solutions(13, 4) {
        assert_eq!(&x * &x - b(13) * &y * &y, b(1));
    }
}

/// `continued_fraction_reduce_periodic` strips common factors by gcds.
/// Before: it factored `gcd(α, γ)`, a number of the convergents' size (4 s
/// for the 116-term period of `√89134`).  SymPy 1.14:
/// `continued_fraction_periodic(0, 1, 89134)` has a 116-term period and
/// `continued_fraction_reduce` gives `sqrt(89134)`.
#[test]
fn continued_fraction_reduce_periodic_of_a_long_period() {
    let cf = nt::continued_fraction_periodic(89134).unwrap();
    assert_eq!(cf.period.len(), 116);
    let s = nt::continued_fraction_reduce_periodic(&cf.pre_period, &cf.period).unwrap();
    assert_eq!((s.p, s.q, s.d), (b(0), b(1), b(89134)));
}

/// `harmonic`, `bernoulli` and `euler_number` have work bounds (`n ≤ 2¹⁸`,
/// even `n ≤ 4096`) instead of running for minutes, and their values are
/// unchanged.  SymPy 1.14: `harmonic(30)` = 9304682830147/2329089562800;
/// `harmonic(700)` is checked against the term-by-term sum.
#[test]
fn harmonic_bernoulli_and_euler_numbers() {
    let q = |n: &str, d: &str| BigRational::new(big(n), big(d));
    assert_eq!(nt::harmonic(30), Some(q("9304682830147", "2329089562800")));
    let sum: BigRational = (1..=700)
        .map(|k| BigRational::new(b(1), b(k)))
        .fold(BigRational::zero(), |acc, t| acc + t);
    assert_eq!(nt::harmonic(700), Some(sum));
    assert_eq!(nt::harmonic((1i64 << 18) + 1), None);
    assert_eq!(nt::bernoulli(4098), None);
    assert_eq!(nt::bernoulli(4097), Some(BigRational::zero()));
    assert_eq!(nt::euler_number(4098), None);
    assert_eq!(nt::euler_number(4097), Some(b(0)));
    // SymPy 1.14: bernoulli(60), euler(20).
    assert_eq!(
        nt::bernoulli(60),
        Some(q(
            "-1215233140483755572040304994079820246041491",
            "56786730"
        ))
    );
    assert_eq!(nt::euler_number(20), Some(big("370371188237525")));
}

/// The Frobenius number of two coprime values near `2⁶⁴`.  Before: `ab −
/// a − b` wrapped in `i128` (`Some(−92233720368547758075)` in a release
/// build).  Oracle: Sylvester's `ab − a − b` (`(2⁶⁴ − 1)(2⁶⁴ − 2) − …` is
/// about `3.4·10³⁸ > i128::MAX`; `2³²·(2³² + 1) − 2³² − (2³² + 1)` =
/// 18446744069414584319).
#[test]
fn frobenius_number_of_two_huge_values() {
    assert_eq!(dp::frobenius_number(&[u64::MAX, u64::MAX - 1]), None);
    assert_eq!(
        dp::frobenius_number(&[1 << 32, (1 << 32) + 1]),
        Some(18_446_744_069_414_584_319)
    );
    assert_eq!(dp::frobenius_number(&[6, 9, 20]), Some(43));
}

/// `binomial(n, k)` with a negative `n` and `k ≥ 2⁶⁴` (before: 0), and a
/// large `min(k, n − k)` by the prime powers of the result (before:
/// quadratic, `catalan(300000)` took 16 s).  Oracles: `C(−1, k) = (−1)ᵏ`,
/// `C(−3, k) = (−1)ᵏ·(k + 1)(k + 2)/2` (SymPy 1.14 `binomial(-1, 2**64)`
/// does not finish within 8 s), and the term-by-term product for
/// `C(10000, 5000)`.
#[test]
fn binomials_with_negative_n_or_large_k() {
    let k = pow2(64);
    assert_eq!(cb::binomial(-1, k.clone()), b(1));
    assert_eq!(cb::binomial(-1, &k + 1u32), b(-1));
    let want: BigInt = -((&k + 2u32) * (&k + 3u32) / 2u32);
    assert_eq!(cb::binomial(-3, &k + 1u32), want);
    let mut running = BigInt::one();
    for i in 0..5000i64 {
        running = running * b(10000 - i) / b(i + 1);
    }
    assert_eq!(cb::binomial(10000, 5000), running);
    assert_eq!(cb::catalan(5000), Some(&running / b(5001)));
}

/// Combinatorial numbers whose recurrences were quadratic in the size of
/// the result, and multinomials beyond `u64` or the size bound.  Before:
/// `stirling2(100000, 50)`, `stirling1(100000, 3)`, `stirling1(10⁶, 1)`,
/// `derangements(200000)` and `multinomial(4·10⁶, &[2·10⁶, 2·10⁶])` ran
/// for minutes (row-by-row triangles, running products);
/// `multinomial(2⁶⁴ + 1, &[2⁶⁴ − 1, 2])` panicked in a debug build (the
/// `u64` sum of the parts overflowed) and `multinomial(2⁴⁰, &[2³⁹, 2³⁹])`, a
/// value of `2⁴⁰` bits, was attempted.  Oracles: SymPy 1.14 as (bit length,
/// value mod `2⁶¹ − 1`): `stirling(20000, 7)` = (56135,
/// 2181878950271146808), `subfactorial(20000)` = (256908,
/// 1040666495019072116), `stirling(20000, 3, kind=1, signed=True)` =
/// (256901, 2107693851366449484), negative; `stirling(1000, 1, kind=1,
/// signed=True)` = (8520, 1524544406649127209), negative; and
/// `binomial(4000, 2000)` for the multinomial `(4000; 2000, 2000)`.
#[test]
fn combinatorial_numbers_without_quadratic_recurrences() {
    let m61 = pow2(61) - 1;
    let print = |v: BigInt| (v.bits(), v.mod_floor(&m61), v.is_negative());
    let s2 = cb::stirling2(20000, 7).unwrap();
    assert_eq!(print(s2), (56135, b(2_181_878_950_271_146_808), false));
    let d = cb::derangements(20000).unwrap();
    assert_eq!(print(d), (256_908, b(1_040_666_495_019_072_116), false));
    let s1 = cb::stirling1(20000, 3).unwrap();
    assert_eq!(print(s1), (256_901, b(2_107_693_851_366_449_484), true));
    let s1 = cb::stirling1(1000, 1).unwrap();
    assert_eq!(print(s1), (8520, b(1_524_544_406_649_127_209), true));
    let big_n = BigInt::from(u64::MAX) + 2u32;
    assert_eq!(
        cb::multinomial(big_n, &[BigInt::from(u64::MAX), b(2)]),
        None
    );
    assert_eq!(cb::multinomial(1u64 << 40, &[1u64 << 39, 1u64 << 39]), None);
    assert_eq!(
        cb::multinomial(4000, &[2000, 2000]),
        Some(cb::binomial(4000, 2000))
    );
}

/// `prime(n)` beyond `n = 10⁷` by counting.  Before: `None`.  SymPy 1.14:
/// `prime(10**8)` = 2038074743, `prime(10**9)` = 22801763489.
#[test]
fn prime_beyond_ten_million() {
    assert_eq!(nt::prime(100_000_000), Some(b(2_038_074_743)));
    assert_eq!(nt::prime(1_000_000_000), Some(b(22_801_763_489)));
    assert_eq!(nt::prime(10_000_001), Some(b(179_424_691)));
    assert_eq!(nt::prime(37_607_912_019u64), None);
}

fn parse(ctx: &Context, s: &str) -> Ex {
    ctx.parse(s).unwrap()
}

/// `eval_decimal` of `s` against a SymPy 1.14 `evalf(20)` value, to 15
/// significant digits.
fn assert_decimal(ctx: &Context, s: &str, want: &str) {
    let got = parse(ctx, s)
        .eval()
        .eval_decimal(25)
        .unwrap_or_else(|e| panic!("{s}: {e}"));
    let (g, w): (f64, f64) = (got.parse().unwrap(), want.parse().unwrap());
    assert!((g - w).abs() <= 1e-15 * w.abs(), "{s}: {got} vs {want}");
}

/// Fibonacci and Lucas numbers at negative indices, and both functions off
/// the integers (`F(x) = (φˣ − cos(πx)·φ⁻ˣ)/√5`, `L(x) = φˣ +
/// cos(πx)·φ⁻ˣ`).  Before: unevaluated and not evaluable numerically.
/// SymPy 1.14: `fibonacci(-5)` = 5, `fibonacci(-8)` = -21, `lucas(-3)` =
/// -4, `lucas(-4)` = 7, `fibonacci(1/2).evalf(20)` =
/// 0.56886448100578310728, `fibonacci(-1/2).evalf(20)` =
/// 0.35157758425414292849 (SymPy has no numerical `lucas(1/2)`; `L(1/2) =
/// √φ` follows from the formula), `fibonacci(-20001/2).evalf(20)` =
/// 4.6732463766339095645e-2091 (a half-integer, where `cos(πx) = 0` leaves
/// the tiny `φˣ/√5`; the first version of the routine took the zero
/// cosine term for a cancellation and gave up with `PrecisionExhausted`).
#[test]
fn fibonacci_and_lucas_at_negative_and_fractional_indices() {
    let ctx = Context::new();
    for (s, want) in [
        ("fibonacci(-5)", "5"),
        ("fibonacci(-8)", "-21"),
        ("lucas(-3)", "-4"),
        ("lucas(-4)", "7"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want), "{s}");
    }
    assert_decimal(&ctx, "fibonacci(1/2)", "0.56886448100578310728");
    assert_decimal(&ctx, "fibonacci(-1/2)", "0.35157758425414292849");
    let sqrt_phi = ((1.0 + 5f64.sqrt()) / 2.0).sqrt();
    assert_decimal(&ctx, "lucas(1/2)", &sqrt_phi.to_string());
    let tiny = parse(&ctx, "fibonacci(-20001/2)")
        .eval()
        .eval_decimal(20)
        .unwrap();
    assert!(
        tiny.starts_with("4.67324637663390956") && tiny.ends_with("e-2091"),
        "{tiny}"
    );
}

/// The Catalan function off the non-negative integers: `C(x) = Γ(2x +
/// 1)/(Γ(x + 1)·Γ(x + 2))`.  Before: unevaluated, and `evalf` had no
/// routine.  SymPy 1.14: `catalan(-1)` = -1/2, `catalan(-2)` = 0,
/// `catalan(-1/2)` = `catalan(-3/2)` = zoo, `catalan(1/2).evalf(20)` =
/// 0.84882636315677512410, `catalan(-1/3).evalf(20)` =
/// 2.1914977293094775372.
#[test]
fn catalan_off_the_non_negative_integers() {
    let ctx = Context::new();
    for (s, want) in [
        ("catalan(-1)", "-1/2"),
        ("catalan(-2)", "0"),
        ("catalan(-7)", "0"),
    ] {
        assert_eq!(parse(&ctx, s).eval(), parse(&ctx, want), "{s}");
    }
    for s in ["catalan(-1/2)", "catalan(-3/2)"] {
        assert_eq!(parse(&ctx, s).eval().to_string(), "zoo", "{s}");
    }
    assert_decimal(&ctx, "catalan(1/2)", "0.84882636315677512410");
    assert_decimal(&ctx, "catalan(-1/3)", "2.1914977293094775372");
}

/// Gamma quotients with a factor `Γ(1)` or `Γ(2)`, whose logarithm is
/// exactly 0.  Before: `PrecisionExhausted`.  SymPy 1.14:
/// `rf(1/2, 1/2).evalf(20)` = 0.56418958354775628695 (`1/√π`),
/// `ff(3/2, 1/2).evalf(20)` = 1.3293403881791370205.
#[test]
fn gamma_quotients_with_unit_factors() {
    let ctx = Context::new();
    assert_decimal(&ctx, "rising_factorial(1/2, 1/2)", "0.56418958354775628695");
    assert_decimal(&ctx, "falling_factorial(3/2, 1/2)", "1.3293403881791370205");
}

/// `binomial(n, k)` with `n` beyond `u64` and a small `min(k, n − k)`.
/// Before: unevaluated.  SymPy 1.14: `binomial(10**20, 2)` =
/// `binomial(10**20, 10**20 - 2)` = 4999999999999999999950000000000000000000.
#[test]
fn binomial_with_a_huge_n() {
    let ctx = Context::new();
    let want = parse(&ctx, "4999999999999999999950000000000000000000");
    assert_eq!(parse(&ctx, "binomial(10^20, 2)").eval(), want);
    assert_eq!(parse(&ctx, "binomial(10^20, 10^20 - 2)").eval(), want);
}

/// `bernoulli(s) = −s·ζ(1 − s)` off the integers, and `euler_number(−1) =
/// π/2`.  Before: unevaluated and not evaluable numerically.  SymPy 1.14:
/// `euler(-1)` = pi/2; `bernoulli(s).evalf(20)` = 0.73017725440479340644
/// (`s = 1/2`), 1.6449340668482264365 (`-1`), 0.063713004724582589874
/// (`5/2`), 2.6771645527264000362 (`-7/3`), -0.00090664277014792935168
/// (`301/100`).  (`bernoulli(1)` stays `−1/2`, this crate's convention;
/// SymPy 1.14 has `+1/2`.)
#[test]
fn bernoulli_function_and_euler_number_at_minus_one() {
    let ctx = Context::new();
    assert_eq!(
        parse(&ctx, "euler_number(-1)").eval(),
        parse(&ctx, "pi/2").eval()
    );
    assert_eq!(parse(&ctx, "bernoulli(1)").eval(), parse(&ctx, "-1/2"));
    for (s, want) in [
        ("bernoulli(1/2)", "0.73017725440479340644"),
        ("bernoulli(-1)", "1.6449340668482264365"),
        ("bernoulli(5/2)", "0.063713004724582589874"),
        ("bernoulli(-7/3)", "2.6771645527264000362"),
        ("bernoulli(301/100)", "-0.00090664277014792935168"),
    ] {
        assert_decimal(&ctx, s, want);
    }
}
