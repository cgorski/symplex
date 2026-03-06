//! Number Theory — primality, factorization, and modular arithmetic.
//! Run with: cargo run --example number_theory
use num_bigint::BigInt;
use num_traits::One;
use symplex::ntheory::*;

fn main() {
    println!("=== Number Theory ===\n");

    // ── Primality testing (i64 fast path) ──────────────────────────────
    println!("Primality testing (i64):");
    println!("  Is 104729 prime? {}", isprime(104729));
    println!("  Is 104730 prime? {}", isprime(104730));
    println!(
        "  Is 2^31-1 (Mersenne M31) prime? {}",
        isprime(2_147_483_647)
    );
    println!("  Next prime after 100: {}", nextprime(100));
    println!(
        "  Previous prime before 100: {}",
        prevprime(100).unwrap()
    );

    // ── Primality testing (BigInt arbitrary precision) ──────────────────
    println!("\nPrimality testing (BigInt):");

    // Mersenne prime M31 = 2^31 - 1
    let m31 = BigInt::from(2_147_483_647i64);
    println!("  Is {} (M31) prime? {}", m31, isprime_bigint(&m31));

    // Mersenne prime M61 = 2^61 - 1 = 2305843009213693951
    let m61 = BigInt::from(2u64.pow(61) - 1);
    println!("  Is {} (M61) prime? {}", m61, isprime_bigint(&m61));

    // Large composite: product of two large primes
    let p1 = BigInt::from(1_000_000_007i64);
    let p2 = BigInt::from(1_000_000_009i64);
    let composite = &p1 * &p2;
    println!(
        "  Is {} × {} = {} prime? {}",
        p1,
        p2,
        composite,
        isprime_bigint(&composite)
    );

    // ── Sieve ──────────────────────────────────────────────────────────
    println!("\nPrimes up to 50: {:?}", primes_up_to(50));

    // ── Factorization (i64 fast path) ──────────────────────────────────
    println!("\nFactorization (i64):");
    for n in [60, 360, 2310, 100_000, 1_000_003] {
        let factors = factorint(n);
        let s: Vec<String> = factors
            .iter()
            .map(|(p, e)| {
                if *e == 1 {
                    format!("{p}")
                } else {
                    format!("{p}^{e}")
                }
            })
            .collect();
        println!("  {n} = {}", s.join(" × "));
    }

    // ── Factorization (BigInt) ─────────────────────────────────────────
    println!("\nFactorization (BigInt):");

    // Large number factorization — Mersenne prime M31
    let big = BigInt::from(2_147_483_647i64);
    let factors = factorint_bigint(&big);
    print_bigint_factors(&big, &factors);

    // Mersenne prime M61
    let factors = factorint_bigint(&m61);
    print_bigint_factors(&m61, &factors);

    // 2^40
    let pow2_40 = BigInt::from(1u64 << 40);
    let factors = factorint_bigint(&pow2_40);
    print_bigint_factors(&pow2_40, &factors);

    // Beyond f64 precision: 2^53 + 1 = 9007199254740993
    let beyond_f64 = BigInt::from(2u64.pow(53) + 1);
    let factors = factorint_bigint(&beyond_f64);
    print_bigint_factors(&beyond_f64, &factors);
    // Verify roundtrip
    let mut product = BigInt::one();
    for (p, e) in &factors {
        for _ in 0..*e {
            product *= p;
        }
    }
    assert_eq!(product, beyond_f64, "roundtrip must be exact beyond f64");
    println!("  ✓ Roundtrip verified (no f64 truncation)");

    // ── Divisors ───────────────────────────────────────────────────────
    println!("\nDivisors of 60: {:?}", divisors(60));
    println!("Divisor count of 60: {}", divisor_count(60));
    println!("Divisor sum of 60: {}", divisor_sum(60));

    // ── Euler's totient ────────────────────────────────────────────────
    println!("\nEuler's totient:");
    for n in [1, 6, 12, 36, 97] {
        println!("  φ({n}) = {}", totient(n));
    }

    // ── Möbius function ────────────────────────────────────────────────
    println!("\nMöbius function:");
    for n in [1, 2, 4, 6, 30] {
        println!("  μ({n}) = {}", mobius(n));
    }

    // ── GCD / LCM / coprimality ────────────────────────────────────────
    println!("\nGCD and LCM:");
    println!("  gcd(12, 8) = {}", gcd_int(12, 8));
    println!("  lcm(4, 6) = {}", lcm_int(4, 6));
    println!("  coprime(8, 15)? {}", is_coprime(8, 15));
    println!("  coprime(8, 12)? {}", is_coprime(8, 12));

    // ── Modular arithmetic ─────────────────────────────────────────────
    println!("\nModular arithmetic:");
    println!("  3⁻¹ mod 7 = {}", mod_inverse(3, 7).unwrap());
    println!("  17⁻¹ mod 43 = {}", mod_inverse(17, 43).unwrap());
    println!("  2^10 mod 1000 = {}", mod_pow_int(2, 10, 1000));
    println!(
        "  CRT: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = {}",
        crt(&[2, 3, 2], &[3, 5, 7]).unwrap()
    );

    // ── Perfect squares ────────────────────────────────────────────────
    println!("\nPerfect squares:");
    println!("  is_square(144) = {}", is_square(144));
    println!("  is_square(145) = {}", is_square(145));
    println!("  isqrt(50) = {}", isqrt(50).unwrap());

    // ── Legendre symbol ────────────────────────────────────────────────
    println!("\nLegendre symbol (a/p):");
    println!("  (2/7) = {}", legendre_symbol(2, 7));
    println!("  (3/7) = {}", legendre_symbol(3, 7));

    // ── Expression-level factorization ─────────────────────────────────
    println!("\nExpression-level factorization:");
    let n = symplex::int(360);
    if let Some(factors) = n.factorize_int() {
        let s: Vec<String> = factors
            .iter()
            .map(|(p, e)| {
                if *e == 1 {
                    format!("{p}")
                } else {
                    format!("{p}^{e}")
                }
            })
            .collect();
        println!("  360 = {}", s.join(" × "));
    }

    // Expression-level BigInt factorization
    let n = symplex::int(360);
    if let Some(factors) = n.factorize_int_bigint() {
        let s: Vec<String> = factors
            .iter()
            .map(|(p, e)| {
                if *e == 1 {
                    format!("{p}")
                } else {
                    format!("{p}^{e}")
                }
            })
            .collect();
        println!("  360 = {} (via BigInt)", s.join(" × "));
    }

    // Expression-level primality
    println!("\nExpression-level primality:");
    let n = symplex::int(104729);
    println!(
        "  Is 104729 prime? {:?}",
        n.is_prime_value()
    );
    let n = symplex::int(2_147_483_647);
    println!(
        "  Is 2147483647 (M31) prime? {:?}",
        n.is_prime_value()
    );
    let n = symplex::int(60);
    println!("  Is 60 prime? {:?}", n.is_prime_value());
    let half = symplex::rational(1, 2);
    println!("  Is 1/2 prime? {:?}", half.is_prime_value());

    println!("\n✓ Done!");
}

/// Helper to pretty-print BigInt factorization.
fn print_bigint_factors(n: &BigInt, factors: &[(BigInt, u32)]) {
    let s: Vec<String> = factors
        .iter()
        .map(|(p, e)| {
            if *e == 1 {
                format!("{p}")
            } else {
                format!("{p}^{e}")
            }
        })
        .collect();
    println!("  {n} = {}", s.join(" × "));
}
