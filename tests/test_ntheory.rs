//! Integration tests for the `ntheory` number-theory module.
//!
//! All tests use the **unified API** — one function name per operation,
//! accepting `impl Into<BigInt>` and returning `BigInt`-typed results.

use num_bigint::BigInt;
use num_traits::One;
use symplex::ntheory::*;

use symplex::prelude::*;
/// Shorthand for `BigInt::from(n)`.
fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

// ═══════════════════════════════════════════════════════════════════════════
// Primality — small values (dispatches to i64 fast path internally)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn primes_up_to_30() {
    let primes: Vec<i64> = (2..=30).filter(|&n| isprime(n)).collect();
    assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
}

#[test]
fn primes_up_to_30_sieve() {
    assert_eq!(primes_up_to(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
}

#[test]
fn isprime_negative_and_zero() {
    for n in -100..=1 {
        assert!(!isprime(n), "isprime({n}) should be false");
    }
}

#[test]
fn isprime_known_large_primes() {
    assert!(isprime(1_000_000_007));
    assert!(isprime(999_999_937));
    assert!(isprime(2_147_483_647)); // Mersenne prime 2^31 - 1
}

#[test]
fn isprime_carmichael_numbers() {
    let carmichaels = [561, 1105, 1729, 2465, 2821, 6601, 8911];
    for &c in &carmichaels {
        assert!(!isprime(c), "Carmichael number {c} should not be prime");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Primality — BigInt (same `isprime` function, larger values)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn isprime_bigint_small_primes() {
    let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        assert!(
            isprime(BigInt::from(p)),
            "{p} should be prime (BigInt)"
        );
    }
}

#[test]
fn isprime_bigint_small_composites() {
    let composites = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25];
    for &c in &composites {
        assert!(
            !isprime(BigInt::from(c)),
            "{c} should not be prime (BigInt)"
        );
    }
}

#[test]
fn isprime_bigint_negative() {
    for n in -100..=1 {
        assert!(
            !isprime(BigInt::from(n)),
            "isprime(BigInt::from({n})) should be false"
        );
    }
}

#[test]
fn isprime_bigint_mersenne_m31() {
    let m31 = BigInt::from(2_147_483_647i64);
    assert!(isprime(m31));
}

#[test]
fn isprime_bigint_mersenne_m61() {
    // 2^61 - 1 = 2305843009213693951 is a Mersenne prime (fits in i64)
    let m61 = BigInt::from(2u64.pow(61) - 1);
    assert!(isprime(m61));
}

#[test]
fn isprime_bigint_large_composite() {
    let m31 = BigInt::from(2_147_483_647i64);
    let other_prime = BigInt::from(1_000_000_007i64);
    let product = &m31 * &other_prime;
    assert!(
        !isprime(product),
        "product of two primes should be composite"
    );
}

#[test]
fn isprime_bigint_carmichael() {
    let carmichaels = [561i64, 1105, 1729, 2465, 2821, 6601, 8911];
    for &c in &carmichaels {
        assert!(
            !isprime(BigInt::from(c)),
            "Carmichael number {c} should not be prime (BigInt)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorization — returns Vec<(BigInt, u32)> for all inputs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorint_large_power_of_two() {
    assert_eq!(factorint(2_i64.pow(20)), vec![(bi(2), 20)]);
}

#[test]
fn factorint_product_of_primes() {
    // 2 * 3 * 5 * 7 * 11 * 13 = 30030
    assert_eq!(
        factorint(30030),
        vec![(bi(2), 1), (bi(3), 1), (bi(5), 1), (bi(7), 1), (bi(11), 1), (bi(13), 1)]
    );
}

#[test]
fn factorint_large_semiprime() {
    // 10007 and 10009 are both prime
    let n = 10007_i64 * 10009;
    let factors = factorint(n);
    assert_eq!(factors, vec![(bi(10007), 1), (bi(10009), 1)]);
}

#[test]
fn factorint_prime_cubed() {
    assert_eq!(factorint(27), vec![(bi(3), 3)]);
    assert_eq!(factorint(125), vec![(bi(5), 3)]);
}

#[test]
fn factorint_zero_and_one() {
    assert_eq!(factorint(0), vec![]);
    assert_eq!(factorint(1), vec![]);
}

#[test]
fn factorint_roundtrip() {
    for n in [60i64, 360, 2310, 100_000, 123456789, 999_999_937] {
        let factors = factorint(n);
        let mut product = BigInt::one();
        for (p, e) in &factors {
            for _ in 0..*e {
                product *= p;
            }
        }
        assert_eq!(product, bi(n), "factorization of {n} doesn't multiply back");
    }
}

#[test]
fn factorint_all_factors_are_prime() {
    for n in [60, 360, 2310, 100_000, 123456789] {
        let factors = factorint(n);
        for (p, _) in &factors {
            assert!(isprime(p.clone()), "factor {p} of {n} is not prime");
        }
    }
}

// ── factorint with BigInt input ──────────────────────────────────────────

#[test]
fn factorint_bigint_basic() {
    let factors = factorint(BigInt::from(60));
    assert_eq!(
        factors,
        vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]
    );
}

#[test]
fn factorint_bigint_zero_and_one() {
    assert_eq!(factorint(BigInt::from(0)), vec![]);
    assert_eq!(factorint(BigInt::from(1)), vec![]);
}

#[test]
fn factorint_bigint_negative() {
    let factors = factorint(BigInt::from(-60));
    assert_eq!(
        factors,
        vec![(bi(2), 2), (bi(3), 1), (bi(5), 1)]
    );
}

#[test]
fn factorint_bigint_prime_returns_itself() {
    let p = BigInt::from(104729i64);
    let factors = factorint(p.clone());
    assert_eq!(factors, vec![(p, 1)]);
}

#[test]
fn factorint_bigint_power_of_two_40() {
    let n = BigInt::from(1u64 << 40);
    let factors = factorint(n);
    assert_eq!(factors, vec![(bi(2), 40)]);
}

#[test]
fn factorint_bigint_mersenne_m31_is_prime() {
    let m31 = BigInt::from(2_147_483_647i64);
    let factors = factorint(m31.clone());
    assert_eq!(factors, vec![(m31, 1)]);
}

#[test]
fn factorint_bigint_mersenne_m61_is_prime() {
    let m61 = BigInt::from(2u64.pow(61) - 1);
    let factors = factorint(m61.clone());
    assert_eq!(factors, vec![(m61, 1)]);
}

#[test]
fn factorint_bigint_roundtrip() {
    for val in [60i64, 360, 2310, 100_000, 123456789, 999_999_937] {
        let n = BigInt::from(val);
        let factors = factorint(n.clone());
        let mut product = BigInt::one();
        for (p, e) in &factors {
            for _ in 0..*e {
                product *= p;
            }
        }
        assert_eq!(product, n, "factorization of {val} doesn't multiply back");
    }
}

#[test]
fn factorint_bigint_all_factors_prime() {
    for val in [60i64, 360, 2310, 100_000, 123456789] {
        let n = BigInt::from(val);
        let factors = factorint(n);
        for (p, _) in &factors {
            assert!(isprime(p.clone()), "factor {p} of {val} is not prime");
        }
    }
}

#[test]
fn factorint_bigint_beyond_f64_precision() {
    // 2^53 + 1 = 9007199254740993 is beyond exact f64 integer range.
    let n = BigInt::from(2u64.pow(53) + 1);
    let factors = factorint(n.clone());
    let mut product = BigInt::one();
    for (p, e) in &factors {
        for _ in 0..*e {
            product *= p;
        }
    }
    assert_eq!(
        product, n,
        "factorization beyond f64 precision must be exact"
    );
    for (p, _) in &factors {
        assert!(isprime(p.clone()), "factor {p} should be prime");
    }
}

#[test]
fn factorint_bigint_large_semiprime() {
    let p1 = BigInt::from(2_147_483_647i64); // 2^31 - 1, Mersenne prime
    let p2 = BigInt::from(1_000_000_007i64);
    let n = &p1 * &p2;
    let factors = factorint(n);
    assert_eq!(factors, vec![(p2, 1), (p1, 1)]); // sorted ascending
}

// ═══════════════════════════════════════════════════════════════════════════
// Next / previous prime — now returns BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nextprime_sequence() {
    let mut p = bi(2);
    let mut primes = vec![p.clone()];
    for _ in 0..9 {
        p = nextprime(p);
        primes.push(p.clone());
    }
    let expected: Vec<BigInt> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        .into_iter()
        .map(bi)
        .collect();
    assert_eq!(primes, expected);
}

#[test]
fn prevprime_sequence() {
    let mut p = bi(29);
    let mut primes = vec![p.clone()];
    while let Some(prev) = prevprime(p) {
        primes.push(prev.clone());
        p = prev;
    }
    primes.reverse();
    let expected: Vec<BigInt> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        .into_iter()
        .map(bi)
        .collect();
    assert_eq!(primes, expected);
}

#[test]
fn nextprime_prevprime_inverse() {
    for &p in &[2, 3, 5, 7, 11, 97, 1009, 10007] {
        let next = nextprime(p);
        assert_eq!(prevprime(next), Some(bi(p)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Divisors — now returns Vec<BigInt>
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn divisors_prime() {
    assert_eq!(divisors(13), vec![bi(1), bi(13)]);
}

#[test]
fn divisors_prime_power() {
    assert_eq!(divisors(8), vec![bi(1), bi(2), bi(4), bi(8)]);
    assert_eq!(divisors(27), vec![bi(1), bi(3), bi(9), bi(27)]);
}

#[test]
fn divisor_sum_formula() {
    // For n = p1^e1 * ... * pk^ek, the sum of divisors equals
    // product of (p^(e+1) - 1) / (p - 1)
    for &n in &[12i64, 60, 360, 2310, 100] {
        let factors = factorint(n);
        let mut expected = BigInt::one();
        for (p, e) in &factors {
            let num = p.pow(e + 1) - BigInt::one();
            let den = p - BigInt::one();
            expected *= num / den;
        }
        assert_eq!(
            divisor_sum(n),
            expected,
            "divisor sum formula failed for n={n}"
        );
    }
}

#[test]
fn divisor_count_formula() {
    // Number of divisors = product of (e_i + 1)
    for &n in &[12, 60, 360, 2310, 100, 1024] {
        let factors = factorint(n);
        let expected: usize = factors.iter().map(|(_, e)| (e + 1) as usize).product();
        assert_eq!(
            divisor_count(n),
            expected,
            "divisor count formula failed for n={n}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Totient — now returns BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn totient_multiplicative() {
    // φ(m·n) = φ(m)·φ(n) when gcd(m, n) = 1
    let pairs: &[(i64, i64)] = &[
        (3, 5),
        (4, 9),
        (7, 11),
        (8, 15),
        (13, 17),
        (100, 21),
    ];
    for &(m, n) in pairs {
        assert_eq!(gcd(m, n), bi(1), "precondition: gcd({m},{n})=1");
        let phi_mn = totient(m * n);
        let phi_m = totient(m);
        let phi_n = totient(n);
        assert_eq!(
            phi_mn,
            &phi_m * &phi_n,
            "φ({m}·{n}) ≠ φ({m})·φ({n})"
        );
    }
}

#[test]
fn totient_prime_power() {
    // φ(p^k) = p^k - p^(k-1) = p^(k-1) * (p-1)
    assert_eq!(totient(8), bi(4));   // 2^3: 2^2 * 1 = 4
    assert_eq!(totient(27), bi(18)); // 3^3: 3^2 * 2 = 18
    assert_eq!(totient(25), bi(20)); // 5^2: 5^1 * 4 = 20
}

#[test]
fn totient_divisor_sum_identity() {
    // sum_{d | n} φ(d) = n
    for &n in &[1i64, 6, 12, 30, 60, 100] {
        let divs = divisors(n);
        let sum: BigInt = divs.iter().map(|d| totient(d.clone())).sum();
        assert_eq!(sum, bi(n), "Σ φ(d) for d|{n} should equal {n}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Möbius — still returns i8
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mobius_squarefree() {
    assert_eq!(mobius(2), -1);     // 1 prime
    assert_eq!(mobius(3), -1);     // 1 prime
    assert_eq!(mobius(6), 1);      // 2 primes
    assert_eq!(mobius(2 * 3 * 5), -1); // 3 primes
    assert_eq!(mobius(2 * 3 * 5 * 7), 1); // 4 primes
}

#[test]
fn mobius_sum_identity() {
    // sum_{d | n} μ(d) = 1 if n=1, else 0
    for &n in &[1i64, 2, 6, 12, 30, 60] {
        let divs = divisors(n);
        let sum: i8 = divs.iter().map(|d| mobius(d.clone())).sum();
        if n == 1 {
            assert_eq!(sum, 1);
        } else {
            assert_eq!(sum, 0, "Σ μ(d) for d|{n} should be 0");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Modular arithmetic — now returns BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mod_inverse_all_coprime_mod_7() {
    for a in 1..7i64 {
        let inv = mod_inverse(a, 7).expect("should exist");
        assert_eq!(
            (bi(a) * &inv) % bi(7),
            bi(1),
            "inverse of {a} mod 7 is wrong"
        );
    }
}

#[test]
fn mod_pow_large() {
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
    let p = 1_000_000_007i64;
    assert_eq!(mod_pow(2, p - 1, p), bi(1));
    assert_eq!(mod_pow(123456, p - 1, p), bi(1));
}

// ═══════════════════════════════════════════════════════════════════════════
// CRT — BigInt version + i64 convenience
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crt_two_congruences() {
    let r: Vec<BigInt> = vec![bi(1), bi(2)];
    let m: Vec<BigInt> = vec![bi(3), bi(5)];
    assert_eq!(crt(&r, &m), Some(bi(7)));
}

#[test]
fn crt_verification() {
    let r: Vec<BigInt> = vec![bi(2), bi(3), bi(2)];
    let m: Vec<BigInt> = vec![bi(3), bi(5), bi(7)];
    let x = crt(&r, &m).unwrap();
    for i in 0..3 {
        assert_eq!(&x % &m[i], r[i], "CRT solution fails congruence {i}");
    }
}

#[test]
fn crt_i64_convenience() {
    assert_eq!(crt_i64(&[1, 2], &[3, 5]), Some(7));
    assert_eq!(crt_i64(&[2, 3, 2], &[3, 5, 7]), Some(23));
    // x ≡ 0 (mod 2) and x ≡ 1 (mod 4) has no solution
    assert_eq!(crt_i64(&[0, 1], &[2, 4]), None);
}

// ═══════════════════════════════════════════════════════════════════════════
// GCD / LCM — now `gcd` and `lcm`, returning BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_lcm_identity() {
    // gcd(a,b) * lcm(a,b) = |a*b|
    for &(a, b) in &[(12i64, 8), (15, 25), (7, 13), (100, 75)] {
        let product = &gcd(a, b) * &lcm(a, b);
        assert_eq!(
            product,
            bi((a * b).abs()),
            "gcd·lcm ≠ |a·b| for ({a},{b})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Misc helpers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_coprime_exhaustive_small() {
    for n in 1..=20i64 {
        assert!(is_coprime(1, n));
    }
}

#[test]
fn is_square_perfect_squares() {
    for i in 0..=50i64 {
        assert!(is_square(i * i), "{} should be a perfect square", i * i);
    }
}

#[test]
fn isqrt_consistency() {
    for n in 0..=1000i64 {
        let s = isqrt(n).unwrap();
        assert!(&s * &s <= bi(n));
        assert!((&s + bi(1)) * (&s + bi(1)) > bi(n));
    }
}

#[test]
fn legendre_symbol_quadratic_residues_mod_11() {
    let qr: Vec<i64> = (1..11).filter(|&a| legendre_symbol(a, 11) == 1).collect();
    assert_eq!(qr, vec![1, 3, 4, 5, 9]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-level integration — factorize (unified)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorize_via_expr() {
    let ctx = Context::new();
    let n = ctx.int(360);
    let factors = n.factorize().unwrap();
    assert_eq!(factors, vec![(bi(2), 3), (bi(3), 2), (bi(5), 1)]);
}

#[test]
fn factorize_non_integer_returns_none() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    assert!(half.factorize().is_none());
}

#[test]
fn factorize_zero_returns_none() {
    let ctx = Context::new();
    let zero = ctx.int(0);
    assert!(zero.factorize().is_none());
}

#[test]
fn factorize_prime_via_expr() {
    let ctx = Context::new();
    let n = ctx.int(104729);
    let factors = n.factorize().unwrap();
    assert_eq!(factors, vec![(bi(104729), 1)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-level integration — is_prime_value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_prime_value_true() {
    let ctx = Context::new();
    let n = ctx.int(104729);
    assert_eq!(n.is_prime_value(), Some(true));
}

#[test]
fn is_prime_value_false() {
    let ctx = Context::new();
    let n = ctx.int(60);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_non_integer_returns_none() {
    let ctx = Context::new();
    let half = ctx.rational(1, 2);
    assert_eq!(half.is_prime_value(), None);
}

#[test]
fn is_prime_value_zero() {
    let ctx = Context::new();
    let n = ctx.int(0);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_one() {
    let ctx = Context::new();
    let n = ctx.int(1);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_negative() {
    let ctx = Context::new();
    let n = ctx.int(-7);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_mersenne_m31() {
    let ctx = Context::new();
    let n = ctx.int(2_147_483_647);
    assert_eq!(n.is_prime_value(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// No f64 truncation: verify that the expr bridge is exact
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorize_does_not_go_through_f64() {
    let ctx = Context::new();
    // 2^31 - 1 = 2147483647 is a Mersenne prime
    let n = ctx.int(2_147_483_647);
    let factors = n.factorize().unwrap();
    assert_eq!(factors, vec![(bi(2_147_483_647), 1)]);
}

#[test]
fn is_prime_value_consistency_with_isprime() {
    let ctx = Context::new();
    for v in -5..=200 {
        let expr = ctx.int(v);
        let via_expr = expr.is_prime_value();
        let direct = isprime(v);
        match via_expr {
            Some(b) => assert_eq!(
                b, direct,
                "Mismatch at {v}: is_prime_value={b}, isprime={direct}"
            ),
            None => panic!("is_prime_value returned None for integer {v}"),
        }
    }
}
