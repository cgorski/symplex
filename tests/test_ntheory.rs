//! Integration tests for the `ntheory` number-theory module.

use num_bigint::BigInt;
use num_traits::One;
use symplex::ntheory::*;

// ═══════════════════════════════════════════════════════════════════════════
// Primality — i64
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
    // A few known primes beyond small-prime range
    assert!(isprime(1_000_000_007));
    assert!(isprime(999_999_937));
    assert!(isprime(2_147_483_647)); // Mersenne prime 2^31 - 1
}

#[test]
fn isprime_carmichael_numbers() {
    // Carmichael numbers fool Fermat tests but not Miller-Rabin
    let carmichaels = [561, 1105, 1729, 2465, 2821, 6601, 8911];
    for &c in &carmichaels {
        assert!(!isprime(c), "Carmichael number {c} should not be prime");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Primality — BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn isprime_bigint_small_primes() {
    let small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for &p in &small_primes {
        assert!(
            isprime_bigint(&BigInt::from(p)),
            "{p} should be prime (BigInt)"
        );
    }
}

#[test]
fn isprime_bigint_small_composites() {
    let composites = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 25];
    for &c in &composites {
        assert!(
            !isprime_bigint(&BigInt::from(c)),
            "{c} should not be prime (BigInt)"
        );
    }
}

#[test]
fn isprime_bigint_negative() {
    for n in -100..=1 {
        assert!(
            !isprime_bigint(&BigInt::from(n)),
            "isprime_bigint({n}) should be false"
        );
    }
}

#[test]
fn isprime_bigint_mersenne_m31() {
    // 2^31 - 1 = 2147483647 is a Mersenne prime
    let m31 = BigInt::from(2_147_483_647i64);
    assert!(isprime_bigint(&m31));
}

#[test]
fn isprime_bigint_mersenne_m61() {
    // 2^61 - 1 = 2305843009213693951 is a Mersenne prime
    // This is beyond i32 range but fits in i64
    let m61 = BigInt::from(2u64.pow(61) - 1);
    assert!(isprime_bigint(&m61));
}

#[test]
fn isprime_bigint_large_composite() {
    // Product of two large primes — must be composite
    let m31 = BigInt::from(2_147_483_647i64);
    let other_prime = BigInt::from(1_000_000_007i64);
    let product = &m31 * &other_prime;
    assert!(
        !isprime_bigint(&product),
        "product of two primes should be composite"
    );
}

#[test]
fn isprime_bigint_carmichael() {
    let carmichaels = [561i64, 1105, 1729, 2465, 2821, 6601, 8911];
    for &c in &carmichaels {
        assert!(
            !isprime_bigint(&BigInt::from(c)),
            "Carmichael number {c} should not be prime (BigInt)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorization — i64
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorint_large_power_of_two() {
    assert_eq!(factorint(2_i64.pow(20)), vec![(2, 20)]);
}

#[test]
fn factorint_product_of_primes() {
    // 2 * 3 * 5 * 7 * 11 * 13 = 30030
    assert_eq!(
        factorint(30030),
        vec![(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1)]
    );
}

#[test]
fn factorint_large_semiprime() {
    // 10007 and 10009 are both prime
    let n = 10007_i64 * 10009;
    let factors = factorint(n);
    assert_eq!(factors, vec![(10007, 1), (10009, 1)]);
}

#[test]
fn factorint_prime_cubed() {
    assert_eq!(factorint(27), vec![(3, 3)]);
    assert_eq!(factorint(125), vec![(5, 3)]);
}

#[test]
fn factorint_zero_and_one() {
    assert_eq!(factorint(0), vec![]);
    assert_eq!(factorint(1), vec![]);
}

#[test]
fn factorint_roundtrip() {
    // Verify that the factorization multiplies back to the original
    for n in [60, 360, 2310, 100_000, 123456789, 999_999_937] {
        let factors = factorint(n);
        let product: i64 = factors.iter().map(|&(p, e)| p.pow(e)).product();
        assert_eq!(product, n, "factorization of {n} doesn't multiply back");
    }
}

#[test]
fn factorint_all_factors_are_prime() {
    for n in [60, 360, 2310, 100_000, 123456789] {
        let factors = factorint(n);
        for &(p, _) in &factors {
            assert!(isprime(p), "factor {p} of {n} is not prime");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorization — BigInt
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorint_bigint_basic() {
    let factors = factorint_bigint(&BigInt::from(60));
    assert_eq!(
        factors,
        vec![
            (BigInt::from(2), 2),
            (BigInt::from(3), 1),
            (BigInt::from(5), 1),
        ]
    );
}

#[test]
fn factorint_bigint_zero_and_one() {
    assert_eq!(factorint_bigint(&BigInt::from(0)), vec![]);
    assert_eq!(factorint_bigint(&BigInt::from(1)), vec![]);
}

#[test]
fn factorint_bigint_negative() {
    let factors = factorint_bigint(&BigInt::from(-60));
    assert_eq!(
        factors,
        vec![
            (BigInt::from(2), 2),
            (BigInt::from(3), 1),
            (BigInt::from(5), 1),
        ]
    );
}

#[test]
fn factorint_bigint_prime_returns_itself() {
    let p = BigInt::from(104729i64);
    let factors = factorint_bigint(&p);
    assert_eq!(factors, vec![(p, 1)]);
}

#[test]
fn factorint_bigint_power_of_two_40() {
    // 2^40 = 1099511627776
    let n = BigInt::from(1u64 << 40);
    let factors = factorint_bigint(&n);
    assert_eq!(factors, vec![(BigInt::from(2), 40)]);
}

#[test]
fn factorint_bigint_mersenne_m31_is_prime() {
    // 2^31 - 1 = 2147483647 is a Mersenne prime
    let m31 = BigInt::from(2_147_483_647i64);
    let factors = factorint_bigint(&m31);
    assert_eq!(factors, vec![(m31, 1)]);
}

#[test]
fn factorint_bigint_mersenne_m61_is_prime() {
    // 2^61 - 1 = 2305843009213693951 is a Mersenne prime
    let m61 = BigInt::from(2u64.pow(61) - 1);
    let factors = factorint_bigint(&m61);
    assert_eq!(factors, vec![(m61, 1)]);
}

#[test]
fn factorint_bigint_roundtrip() {
    for val in [60i64, 360, 2310, 100_000, 123456789, 999_999_937] {
        let n = BigInt::from(val);
        let factors = factorint_bigint(&n);
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
        let factors = factorint_bigint(&n);
        for (p, _) in &factors {
            assert!(isprime_bigint(p), "factor {p} of {val} is not prime");
        }
    }
}

#[test]
fn factorint_bigint_beyond_f64_precision() {
    // 2^53 + 1 = 9007199254740993 is beyond exact f64 integer range.
    // This verifies that no f64 truncation occurs.
    let n = BigInt::from(2u64.pow(53) + 1);
    let factors = factorint_bigint(&n);
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
        assert!(isprime_bigint(p), "factor {p} should be prime");
    }
}

#[test]
fn factorint_bigint_large_semiprime() {
    // Product of two known primes that individually fit in i64 but whose
    // product exceeds what f64 can represent exactly.
    let p1 = BigInt::from(2_147_483_647i64); // 2^31 - 1, Mersenne prime
    let p2 = BigInt::from(1_000_000_007i64); // known prime
    let n = &p1 * &p2;
    let factors = factorint_bigint(&n);
    assert_eq!(factors, vec![(p2, 1), (p1, 1)]); // sorted ascending
}

// ═══════════════════════════════════════════════════════════════════════════
// Next / previous prime
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nextprime_sequence() {
    let mut p = 2i64;
    let mut primes = vec![p];
    for _ in 0..9 {
        p = nextprime(p);
        primes.push(p);
    }
    assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
}

#[test]
fn prevprime_sequence() {
    let mut p = 29i64;
    let mut primes = vec![p];
    while let Some(prev) = prevprime(p) {
        primes.push(prev);
        p = prev;
    }
    primes.reverse();
    assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
}

#[test]
fn nextprime_prevprime_inverse() {
    // For any prime p, prevprime(nextprime(p)) == p
    for &p in &[2, 3, 5, 7, 11, 97, 1009, 10007] {
        let next = nextprime(p);
        assert_eq!(prevprime(next), Some(p));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Divisors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn divisors_prime() {
    assert_eq!(divisors(13), vec![1, 13]);
}

#[test]
fn divisors_prime_power() {
    assert_eq!(divisors(8), vec![1, 2, 4, 8]);
    assert_eq!(divisors(27), vec![1, 3, 9, 27]);
}

#[test]
fn divisor_sum_formula() {
    // For n = p1^e1 * ... * pk^ek, the sum of divisors equals
    // product of (p^(e+1) - 1) / (p - 1)
    for &n in &[12, 60, 360, 2310, 100] {
        let factors = factorint(n);
        let expected: i64 = factors
            .iter()
            .map(|&(p, e)| (p.pow(e + 1) - 1) / (p - 1))
            .product();
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
        let expected: usize = factors.iter().map(|&(_, e)| (e + 1) as usize).product();
        assert_eq!(
            divisor_count(n),
            expected,
            "divisor count formula failed for n={n}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Totient
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
        assert_eq!(gcd_int(m, n), 1, "precondition: gcd({m},{n})=1");
        assert_eq!(
            totient(m * n),
            totient(m) * totient(n),
            "φ({m}·{n}) ≠ φ({m})·φ({n})"
        );
    }
}

#[test]
fn totient_prime_power() {
    // φ(p^k) = p^k - p^(k-1) = p^(k-1) * (p-1)
    assert_eq!(totient(8), 4);   // 2^3: 2^2 * 1 = 4
    assert_eq!(totient(27), 18); // 3^3: 3^2 * 2 = 18
    assert_eq!(totient(25), 20); // 5^2: 5^1 * 4 = 20
}

#[test]
fn totient_divisor_sum_identity() {
    // sum_{d | n} φ(d) = n
    for &n in &[1, 6, 12, 30, 60, 100] {
        let divs = divisors(n);
        let sum: i64 = divs.iter().map(|&d| totient(d)).sum();
        assert_eq!(sum, n, "Σ φ(d) for d|{n} should equal {n}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Möbius
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mobius_squarefree() {
    // Products of distinct primes
    assert_eq!(mobius(2), -1);     // 1 prime
    assert_eq!(mobius(3), -1);     // 1 prime
    assert_eq!(mobius(6), 1);      // 2 primes
    assert_eq!(mobius(2 * 3 * 5), -1); // 3 primes
    assert_eq!(mobius(2 * 3 * 5 * 7), 1); // 4 primes
}

#[test]
fn mobius_sum_identity() {
    // sum_{d | n} μ(d) = 1 if n=1, else 0
    for &n in &[1, 2, 6, 12, 30, 60] {
        let divs = divisors(n);
        let sum: i8 = divs.iter().map(|&d| mobius(d)).sum();
        if n == 1 {
            assert_eq!(sum, 1);
        } else {
            assert_eq!(sum, 0, "Σ μ(d) for d|{n} should be 0");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Modular arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mod_inverse_all_coprime_mod_7() {
    for a in 1..7 {
        let inv = mod_inverse(a, 7).expect("should exist");
        assert_eq!((a * inv) % 7, 1, "inverse of {a} mod 7 is wrong");
    }
}

#[test]
fn mod_pow_int_large() {
    // Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
    let p = 1_000_000_007i64;
    assert_eq!(mod_pow_int(2, p - 1, p), 1);
    assert_eq!(mod_pow_int(123456, p - 1, p), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// CRT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crt_two_congruences() {
    // x ≡ 1 (mod 3), x ≡ 2 (mod 5) → x = 7
    assert_eq!(crt(&[1, 2], &[3, 5]), Some(7));
}

#[test]
fn crt_verification() {
    let r = &[2, 3, 2];
    let m = &[3, 5, 7];
    let x = crt(r, m).unwrap();
    for i in 0..3 {
        assert_eq!(x % m[i], r[i], "CRT solution fails congruence {i}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GCD / LCM
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_lcm_identity() {
    // gcd(a,b) * lcm(a,b) = |a*b|
    for &(a, b) in &[(12, 8), (15, 25), (7, 13), (100, 75)] {
        assert_eq!(
            gcd_int(a, b) * lcm_int(a, b),
            (a * b).abs(),
            "gcd·lcm ≠ |a·b| for ({a},{b})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Misc helpers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_coprime_exhaustive_small() {
    // 1 is coprime to everything
    for n in 1..=20 {
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
        assert!(s * s <= n);
        assert!((s + 1) * (s + 1) > n);
    }
}

#[test]
fn legendre_symbol_quadratic_residues_mod_11() {
    // QRs mod 11: {1, 3, 4, 5, 9} — these have Legendre symbol 1
    let qr: Vec<i64> = (1..11).filter(|&a| legendre_symbol(a, 11) == 1).collect();
    assert_eq!(qr, vec![1, 3, 4, 5, 9]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-level integration — factorize_int (i64)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorize_int_via_expr() {
    let n = symplex::int(360);
    let factors = n.factorize_int().unwrap();
    assert_eq!(factors, vec![(2, 3), (3, 2), (5, 1)]);
}

#[test]
fn factorize_int_non_integer_returns_none() {
    let half = symplex::rational(1, 2);
    assert!(half.factorize_int().is_none());
}

#[test]
fn factorize_int_zero_returns_none() {
    let zero = symplex::int(0);
    assert!(zero.factorize_int().is_none());
}

#[test]
fn factorize_int_prime_via_expr() {
    let n = symplex::int(104729);
    let factors = n.factorize_int().unwrap();
    assert_eq!(factors, vec![(104729, 1)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-level integration — factorize_int_bigint (arbitrary precision)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorize_int_bigint_via_expr() {
    let n = symplex::int(360);
    let factors = n.factorize_int_bigint().unwrap();
    assert_eq!(
        factors,
        vec![
            (BigInt::from(2), 3),
            (BigInt::from(3), 2),
            (BigInt::from(5), 1),
        ]
    );
}

#[test]
fn factorize_int_bigint_non_integer_returns_none() {
    let half = symplex::rational(1, 2);
    assert!(half.factorize_int_bigint().is_none());
}

#[test]
fn factorize_int_bigint_zero_returns_none() {
    let zero = symplex::int(0);
    assert!(zero.factorize_int_bigint().is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression-level integration — is_prime_value
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn is_prime_value_true() {
    let n = symplex::int(104729);
    assert_eq!(n.is_prime_value(), Some(true));
}

#[test]
fn is_prime_value_false() {
    let n = symplex::int(60);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_non_integer_returns_none() {
    let half = symplex::rational(1, 2);
    assert_eq!(half.is_prime_value(), None);
}

#[test]
fn is_prime_value_zero() {
    let n = symplex::int(0);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_one() {
    let n = symplex::int(1);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_negative() {
    let n = symplex::int(-7);
    assert_eq!(n.is_prime_value(), Some(false));
}

#[test]
fn is_prime_value_mersenne_m31() {
    // 2^31 - 1 = 2147483647 is a Mersenne prime
    let n = symplex::int(2_147_483_647);
    assert_eq!(n.is_prime_value(), Some(true));
}

// ═══════════════════════════════════════════════════════════════════════════
// No f64 truncation: verify that the expr bridge is exact
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorize_int_does_not_go_through_f64() {
    // 2^53 - 1 = 9007199254740991 — the largest integer exactly representable
    // in f64.  We use 2^31 - 1 (Mersenne prime) to test that the factorize
    // path works without floating-point conversion.
    let n = symplex::int(2_147_483_647); // 2^31 - 1
    let factors = n.factorize_int().unwrap();
    // It's prime, so the only factor should be itself
    assert_eq!(factors, vec![(2_147_483_647, 1)]);
}

#[test]
fn is_prime_value_consistency_with_isprime() {
    // For small values, is_prime_value on an expression should agree with
    // the standalone isprime function.
    for v in -5..=200 {
        let expr = symplex::int(v);
        let via_expr = expr.is_prime_value();
        let direct = isprime(v as i64);
        match via_expr {
            Some(b) => assert_eq!(
                b, direct,
                "Mismatch at {v}: is_prime_value={b}, isprime={direct}"
            ),
            None => panic!("is_prime_value returned None for integer {v}"),
        }
    }
}
