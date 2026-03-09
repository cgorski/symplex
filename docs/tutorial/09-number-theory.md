# Chapter 9: Number Theory

Symplex includes a comprehensive number theory module with arbitrary-precision support. Every function in `symplex::ntheory` accepts `impl Into<BigInt>`, so you can pass `i64`, `u64`, `i32`, or `BigInt` values interchangeably — small values automatically take an optimized fast path.

## Primality Testing

The `isprime` function uses a deterministic Miller-Rabin test:

```rust
use symplex::ntheory::isprime;
use num_bigint::BigInt;

// Works with plain integers
assert!(isprime(104729));
assert!(!isprime(104730));

// Works with BigInt for truly huge numbers
let m61 = BigInt::from(2u64.pow(61) - 1);
assert!(isprime(m61));

// Large composites are detected correctly
let p1 = BigInt::from(1_000_000_007i64);
let p2 = BigInt::from(1_000_000_009i64);
assert!(!isprime(&p1 * &p2));
```

For small inputs (fits in `i64`), symplex uses a hardcoded witness set that makes Miller-Rabin deterministic up to 2⁶⁴. For `BigInt` inputs, it uses a probabilistic test with enough rounds to be reliable for cryptographic-size numbers.

## Prime Factorization

`factorint` returns `Vec<(BigInt, u32)>` — pairs of (prime, exponent) in ascending order:

```rust
use symplex::ntheory::factorint;
use num_bigint::BigInt;

let factors = factorint(60);
// [(2, 2), (3, 1), (5, 1)]  meaning 60 = 2² × 3 × 5
assert_eq!(factors, vec![
    (BigInt::from(2), 2),
    (BigInt::from(3), 1),
    (BigInt::from(5), 1),
]);

// Edge cases
assert_eq!(factorint(1), vec![]);     // 1 has no prime factors
assert_eq!(factorint(-12), vec![      // negative → factor |n|
    (BigInt::from(2), 2),
    (BigInt::from(3), 1),
]);

// Works beyond f64 precision: 2^53 + 1
let beyond_f64 = BigInt::from(2u64.pow(53) + 1);
let factors = factorint(beyond_f64.clone());
// Verify roundtrip — no floating-point truncation
let product: BigInt = factors.iter().fold(
    num_traits::One::one(),
    |acc: BigInt, (p, e)| acc * p.pow(*e),
);
assert_eq!(product, beyond_f64);
```

## Next and Previous Primes

Navigate the prime number line:

```rust
use symplex::ntheory::{nextprime, prevprime};
use num_bigint::BigInt;

assert_eq!(nextprime(100), BigInt::from(101));
assert_eq!(nextprime(11), BigInt::from(13)); // strictly greater

assert_eq!(prevprime(100), Some(BigInt::from(97)));
assert_eq!(prevprime(3), Some(BigInt::from(2)));
assert_eq!(prevprime(2), None); // no prime less than 2
```

## Sieve of Eratosthenes

`primes_up_to` generates all primes up to a limit. It uses `i64` because the sieve must fit in memory:

```rust
use symplex::ntheory::primes_up_to;

let primes = primes_up_to(50);
assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]);

// Count primes below 1000
let count = primes_up_to(1000).len();
println!("π(1000) = {count}"); // 168
```

## Divisors

Three functions for working with divisors:

```rust
use symplex::ntheory::{divisors, divisor_count, divisor_sum};
use num_bigint::BigInt;

// All positive divisors of |n|, sorted
let d = divisors(12);
assert_eq!(d, vec![1, 2, 3, 4, 6, 12]
    .into_iter().map(BigInt::from).collect::<Vec<_>>());

// Number of divisors: τ(n)
assert_eq!(divisor_count(12), 6);
assert_eq!(divisor_count(1), 1);

// Sum of divisors: σ(n)
assert_eq!(divisor_sum(12), BigInt::from(28));

// 12 is NOT a perfect number (σ(n) ≠ 2n),
// but 6 IS: σ(6) = 1+2+3+6 = 12 = 2×6
assert_eq!(divisor_sum(6), BigInt::from(12));
```

## Euler's Totient Function

`totient(n)` returns φ(n) — the count of integers in [1, n] coprime to n:

```rust
use symplex::ntheory::totient;
use num_bigint::BigInt;

assert_eq!(totient(12), BigInt::from(4));  // {1, 5, 7, 11}
assert_eq!(totient(13), BigInt::from(12)); // prime ⟹ φ(p) = p − 1
assert_eq!(totient(1), BigInt::from(1));

// Euler's product formula: φ(n) = n × ∏(1 - 1/p)
// For n=36 = 2²×3²: φ(36) = 36 × (1-1/2) × (1-1/3) = 12
assert_eq!(totient(36), BigInt::from(12));
```

## Möbius Function

`mobius(n)` returns μ(n), a key function in analytic number theory:

- μ(1) = 1
- μ(n) = 0 if n has a squared prime factor
- μ(n) = (−1)^k if n is a product of k distinct primes

```rust
use symplex::ntheory::mobius;

assert_eq!(mobius(1), 1);
assert_eq!(mobius(6), 1);    // 6 = 2·3 → (−1)² = 1
assert_eq!(mobius(30), -1);  // 30 = 2·3·5 → (−1)³ = −1
assert_eq!(mobius(4), 0);    // 4 = 2² → squared factor

// μ and φ are connected by the Möbius inversion formula:
// φ(n) = Σ_{d|n} μ(n/d) · d
```

## GCD, LCM, and Coprimality

The classic trio:

```rust
use symplex::ntheory::{gcd, lcm, is_coprime};
use num_bigint::BigInt;

assert_eq!(gcd(12, 8), BigInt::from(4));
assert_eq!(gcd(0, 5), BigInt::from(5));

assert_eq!(lcm(4, 6), BigInt::from(12));
assert_eq!(lcm(0, 5), BigInt::from(0));

assert!(is_coprime(8, 15));   // gcd = 1
assert!(!is_coprime(8, 12));  // gcd = 4
```

## Modular Arithmetic

### Modular Inverse

`mod_inverse(a, m)` returns `x` such that `a·x ≡ 1 (mod m)`, or `None` if gcd(a, m) ≠ 1:

```rust
use symplex::ntheory::mod_inverse;
use num_bigint::BigInt;

// 3·5 = 15 ≡ 1 (mod 7)
assert_eq!(mod_inverse(3, 7), Some(BigInt::from(5)));

// No inverse when gcd(a, m) ≠ 1
assert_eq!(mod_inverse(2, 4), None);

// Verify: a × a⁻¹ mod m = 1
let inv = mod_inverse(17, 43).unwrap();
let check = (BigInt::from(17) * &inv) % BigInt::from(43);
assert_eq!(check, BigInt::from(1));
```

### Modular Exponentiation

`mod_pow(base, exp, modulus)` computes base^exp mod modulus efficiently using repeated squaring:

```rust
use symplex::ntheory::mod_pow;
use num_bigint::BigInt;

assert_eq!(mod_pow(2, 10, 1000), BigInt::from(24));   // 1024 mod 1000
assert_eq!(mod_pow(3, 4, 17), BigInt::from(13));      // 81 mod 17

// Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
assert_eq!(mod_pow(7, 12, 13), BigInt::from(1));
```

### Chinese Remainder Theorem

`crt` solves systems of simultaneous congruences. There are two variants — a `BigInt` version and an `i64` convenience version:

```rust
use symplex::ntheory::{crt, crt_i64};
use num_bigint::BigInt;

// Solve: x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
// BigInt version takes separate slices
let r: Vec<BigInt> = vec![2.into(), 3.into(), 2.into()];
let m: Vec<BigInt> = vec![3.into(), 5.into(), 7.into()];
assert_eq!(crt(&r, &m), Some(BigInt::from(23)));

// i64 convenience version
assert_eq!(crt_i64(&[2, 3, 2], &[3, 5, 7]), Some(23));

// Returns None when no solution exists (non-coprime moduli with
// incompatible remainders)
assert_eq!(crt_i64(&[1, 2], &[4, 6]), None);
```

## Legendre Symbol

`legendre_symbol(a, p)` tests whether `a` is a quadratic residue modulo an odd prime `p`:

- Returns `1` if `a` is a quadratic residue (some x² ≡ a mod p)
- Returns `-1` if `a` is a non-residue
- Returns `0` if `a ≡ 0 (mod p)`

```rust
use symplex::ntheory::legendre_symbol;

// 2 is a quadratic residue mod 7 (3² = 9 ≡ 2 mod 7)
assert_eq!(legendre_symbol(2, 7), 1);

// 3 is NOT a quadratic residue mod 7
assert_eq!(legendre_symbol(3, 7), -1);

// a ≡ 0 mod p
assert_eq!(legendre_symbol(7, 7), 0);
```

The function panics if `p` is not an odd prime — this is a precondition, not a runtime error you need to handle.

## Perfect Squares and Integer Square Root

```rust
use symplex::ntheory::{is_square, isqrt};
use num_bigint::BigInt;

assert!(is_square(144));
assert!(!is_square(145));
assert!(is_square(0));

assert_eq!(isqrt(0), Some(BigInt::from(0)));
assert_eq!(isqrt(9), Some(BigInt::from(3)));
assert_eq!(isqrt(10), Some(BigInt::from(3)));  // floor(√10) = 3
assert_eq!(isqrt(-1i64), None);
```

## Expression-Level Number Theory

Symplex also exposes number theory operations on `Ex` expressions, bridging the symbolic and number-theoretic worlds.

### Expression-Level Factorization

`Ex::factorize()` evaluates an expression and, if it's an exact integer, returns its prime factorization:

```rust
use symplex::prelude::*;
use num_bigint::BigInt;

let ctx = Context::new();
let n = ctx.int(360);
let factors = n.factorize().unwrap();
assert_eq!(factors, vec![
    (BigInt::from(2), 3),
    (BigInt::from(3), 2),
    (BigInt::from(5), 1),
]);
// 360 = 2³ × 3² × 5

// Pretty-print the factorization
let s: Vec<String> = factors.iter().map(|(p, e)| {
    if *e == 1 { format!("{p}") } else { format!("{p}^{e}") }
}).collect();
println!("360 = {}", s.join(" × ")); // 360 = 2^3 × 3^2 × 5

// Works with computed expressions
let n = &ctx.int(12) * &ctx.int(5); // 60
let factors = n.factorize().unwrap();
assert_eq!(factors.len(), 3); // 2², 3, 5
```

### Expression-Level Primality

`Ex::is_prime_value()` returns `Some(true)`, `Some(false)`, or `None` (if the expression isn't an integer):

```rust
use symplex::prelude::*;

let ctx = Context::new();
let n = ctx.int(104729);
assert_eq!(n.is_prime_value(), Some(true));

let n = ctx.int(60);
assert_eq!(n.is_prime_value(), Some(false));

// Non-integer → None
let half = ctx.rational(1, 2);
assert_eq!(half.is_prime_value(), None);

// Mersenne prime M31
let n = ctx.int(2_147_483_647);
assert_eq!(n.is_prime_value(), Some(true));
```

## Practical Example: RSA Key Validation

Here's a real-world workflow combining several number theory functions:

```rust
use symplex::ntheory::*;
use num_bigint::BigInt;

fn main() {
    // Two "primes" for a toy RSA key
    let p: i64 = 61;
    let q: i64 = 53;

    assert!(isprime(p));
    assert!(isprime(q));

    let n = BigInt::from(p) * BigInt::from(q); // public modulus
    println!("n = {n}"); // 3233

    // Euler's totient: φ(n) = (p-1)(q-1)
    let phi = totient(p) * totient(q);
    println!("φ(n) = {phi}"); // 3120

    // Choose public exponent e, coprime to φ(n)
    let e: i64 = 17;
    assert!(is_coprime(e, 3120));

    // Compute private exponent d = e⁻¹ mod φ(n)
    let d = mod_inverse(e, 3120i64).unwrap();
    println!("d = {d}"); // 2753

    // Verify: e·d ≡ 1 (mod φ(n))
    let check = mod_pow(BigInt::from(e), BigInt::from(1), BigInt::from(3120));
    let ed_mod = (BigInt::from(e) * &d) % BigInt::from(3120);
    assert_eq!(ed_mod, BigInt::from(1));

    // Encrypt and decrypt a message
    let message = BigInt::from(65);
    let cipher = mod_pow(message.clone(), e, n.clone());
    println!("Encrypted: {cipher}");

    let decrypted = mod_pow(cipher, d, n);
    println!("Decrypted: {decrypted}");
    assert_eq!(decrypted, message);

    println!("✓ RSA roundtrip verified!");
}
```

## Practical Example: Exploring a Number

Combine the tools to fully characterize an integer:

```rust
use symplex::ntheory::*;

fn describe(n: i64) {
    println!("── n = {n} ──");

    // Factorization
    let factors = factorint(n);
    let f_str: Vec<String> = factors.iter().map(|(p, e)| {
        if *e == 1 { format!("{p}") } else { format!("{p}^{e}") }
    }).collect();
    println!("  Factorization: {}", f_str.join(" × "));

    // Divisors
    let divs = divisors(n);
    println!("  Divisors: {:?}", divs);
    println!("  τ({n}) = {}", divisor_count(n));
    println!("  σ({n}) = {}", divisor_sum(n));

    // Totient and Möbius
    println!("  φ({n}) = {}", totient(n));
    println!("  μ({n}) = {}", mobius(n));

    // Primality
    println!("  Prime? {}", isprime(n));

    // Perfect square?
    println!("  Perfect square? {}", is_square(n));

    println!();
}

fn main() {
    for n in [12, 28, 30, 97, 360] {
        describe(n);
    }
}
```

## What's Next

Several number theory features are planned for future releases:

```rust
// Primitive roots
// let g = symplex::ntheory::primitive_root(p);
// assert_eq!(mod_pow(g, totient(p), p), BigInt::from(1));

// Discrete logarithm: find x such that g^x ≡ h (mod p)
// let x = symplex::ntheory::discrete_log(g, h, p);

// Quadratic residues: list all residues mod p
// let residues = symplex::ntheory::quadratic_residues(p);

// Integer partitions: number of ways to write n as sum of positives
// let p_100 = symplex::ntheory::npartitions(100);

// Continued fraction expansion
// let cf = symplex::ntheory::continued_fraction(ctx.rational(355, 113));

// Jacobi symbol (generalization of Legendre to composite moduli)
// let j = symplex::ntheory::jacobi_symbol(a, n);

// Prime counting function
// let count = symplex::ntheory::primepi(1_000_000);
```

---

*[← Chapter 8: Code Generation](08-code-generation.md) | [Back to Table of Contents](index.md) | [Chapter 16: Complex Numbers →](16-complex-numbers.md)*