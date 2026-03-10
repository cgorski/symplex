//! Number Theory — unified API demo.
//!
//! Every function accepts any integer type (`i64`, `u64`, `BigInt`, …) and
//! returns arbitrary-precision `BigInt` results.  Small values automatically
//! take an optimised i64 fast path under the hood.
//!
//! Run with: cargo run --example number_theory

use num_bigint::BigInt;
use num_traits::One;
use symplex::ntheory::*;
use symplex::prelude::*;

fn main() {
    let ctx = Context::new();
    println!("=== Number Theory — Unified API ===\n");

    // ── Primality testing ──────────────────────────────────────────────
    // The SAME `isprime` function works for i64, u64, BigInt, …
    println!("Primality testing:");
    println!("  Is 104729 prime?  {}", isprime(104729));
    println!("  Is 104730 prime?  {}", isprime(104730));
    println!("  Is 2^31-1 (M31) prime?  {}", isprime(2_147_483_647i64));

    // Pass a BigInt for truly huge numbers — same function name
    let m61 = BigInt::from(2u64.pow(61) - 1);
    println!("  Is M61 = {} prime?  {}", m61, isprime(m61.clone()));

    // Large composite: product of two large primes
    let p1 = BigInt::from(1_000_000_007i64);
    let p2 = BigInt::from(1_000_000_009i64);
    let composite = &p1 * &p2;
    let composite_is_prime = isprime(composite.clone());
    println!(
        "  Is {} × {} = {} prime?  {}",
        p1, p2, composite, composite_is_prime
    );

    // ── Next / previous prime ──────────────────────────────────────────
    println!("\nNext / previous prime:");
    println!("  nextprime(100) = {}", nextprime(100));
    println!("  prevprime(100) = {}", prevprime(100).unwrap());

    // ── Sieve ──────────────────────────────────────────────────────────
    println!("\nPrimes up to 50: {:?}", primes_up_to(50));

    // ── Factorization ──────────────────────────────────────────────────
    // One `factorint` for every size.  Returns Vec<(BigInt, u32)>.
    println!("\nFactorization:");
    for n in [60i64, 360, 2310, 100_000, 1_000_003] {
        print_factors(n, &factorint(n));
    }

    // BigInt input — same function
    println!("\n  BigInt inputs:");
    let m31_val = BigInt::from(2_147_483_647i64);
    print_factors_big(&m31_val, &factorint(m31_val.clone()));
    print_factors_big(&m61, &factorint(m61.clone()));

    let pow2_40 = BigInt::from(1u64 << 40);
    print_factors_big(&pow2_40, &factorint(pow2_40.clone()));

    // Beyond f64 precision: 2^53 + 1
    let beyond_f64 = BigInt::from(2u64.pow(53) + 1);
    let factors = factorint(beyond_f64.clone());
    print_factors_big(&beyond_f64, &factors);
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
    println!("Divisor sum of 60:   {}", divisor_sum(60));

    // ── Euler's totient ────────────────────────────────────────────────
    println!("\nEuler's totient:");
    for n in [1i64, 6, 12, 36, 97] {
        println!("  φ({n}) = {}", totient(n));
    }

    // ── Möbius function ────────────────────────────────────────────────
    println!("\nMöbius function:");
    for n in [1i64, 2, 4, 6, 30] {
        println!("  μ({n}) = {}", mobius(n));
    }

    // ── GCD / LCM / coprimality ────────────────────────────────────────
    println!("\nGCD and LCM:");
    println!("  gcd(12, 8)  = {}", gcd(12, 8));
    println!("  lcm(4, 6)   = {}", lcm(4, 6));
    println!("  coprime(8, 15)? {}", is_coprime(8, 15));
    println!("  coprime(8, 12)? {}", is_coprime(8, 12));

    // ── Modular arithmetic ─────────────────────────────────────────────
    println!("\nModular arithmetic:");
    println!("  3⁻¹ mod 7    = {}", mod_inverse(3, 7).unwrap());
    println!("  17⁻¹ mod 43  = {}", mod_inverse(17, 43).unwrap());
    println!("  2^10 mod 1000 = {}", mod_pow(2, 10, 1000));

    // CRT — BigInt version
    let r: Vec<BigInt> = vec![2.into(), 3.into(), 2.into()];
    let m: Vec<BigInt> = vec![3.into(), 5.into(), 7.into()];
    println!(
        "  CRT: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = {}",
        crt(&r, &m).unwrap()
    );
    // CRT — i64 convenience
    println!(
        "  CRT (i64): same → x = {}",
        crt_i64(&[2, 3, 2], &[3, 5, 7]).unwrap()
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
    let n = ctx.int(360);
    if let Some(factors) = n.factorize() {
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

    // Expression-level primality
    println!("\nExpression-level primality:");
    let n = ctx.int(104729);
    println!("  Is 104729 prime?          {:?}", n.is_prime_value());
    let n = ctx.int(2_147_483_647);
    println!("  Is 2147483647 (M31) prime? {:?}", n.is_prime_value());
    let n = ctx.int(60);
    println!("  Is 60 prime?              {:?}", n.is_prime_value());
    let half = ctx.rational(1, 2);
    println!("  Is 1/2 prime?             {:?}", half.is_prime_value());

    println!("\n✓ Done!");
}

// ── helpers ────────────────────────────────────────────────────────────────

fn print_factors(n: i64, factors: &[(BigInt, u32)]) {
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

fn print_factors_big(n: &BigInt, factors: &[(BigInt, u32)]) {
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
