# Number Theory and Combinatorics

Integer functions live in `symplex::ntheory`, `symplex::diophantine` and `symplex::combinatorics`; they work on anything `Into<BigInt>` (`i64`, `u64`, `BigInt`, …) and return `BigInt`/`Ratio<BigInt>`/`Option`. Symbolic counterparts (`n.fibonacci()`, `n.factorial()`, `n.binomial(&k)`, `n.bell()`, …) live on `Ex` and evaluate when the argument is a concrete integer.

## Primality and factorization

0.2 replaces trial division with **Pollard–Brent rho** (Montgomery `u128` arithmetic) plus **ECM** for `BigInt`, and deterministic Miller–Rabin with the **BPSW** test (no Carmichael false positives).

```rust
use num_bigint::BigInt;
use symplex::ntheory::*;

fn main() {
    println!("{}", isprime(561));                                   // false (Carmichael number)
    let m127 = BigInt::parse_bytes(b"170141183460469231731687303715884105727", 10).unwrap();
    println!("{}", isprime(m127));                                  // true, well under a millisecond
    println!("{:?}", factorint(1_099_532_599_387u64));              // [(1048583, 1), (1048589, 1)]
    println!("{:?}", factorint(BigInt::from(2u128.pow(64) + 1)));   // [(274177, 1), (67280421310721, 1)]
    println!("{:?}", primepi(1_000_000));                           // Some(78498)
    println!("{} {:?}", nextprime(100), prevprime(100));            // 101 Some(97)
    println!("{:?}", divisors(28));                                 // [1, 2, 4, 7, 14, 28]
    println!("{} {} {}", totient(36), mobius(30), carmichael_lambda(8));   // 12 -1 2
    println!("{:?}", perfect_power(1024));                          // Some((2, 10))
}
```

## Modular arithmetic

```rust
use symplex::ntheory::*;

fn main() {
    println!("{:?}", mod_inverse(17, 43));               // Some(38)
    println!("{}", mod_pow(3, 200, 1_000_003));
    println!("{:?}", crt_i64(&[2, 3, 2], &[3, 5, 7]));   // Some(23)
    println!("{:?} {:?}", sqrt_mod(2, 7), sqrt_mod_all(2, 7));   // Some(3) [3, 4]
    println!("{:?}", sqrt_mod(3, 7));                    // None — not a quadratic residue
    println!("{:?}", sqrt_mod_all(1, 15));               // [1, 4, 11, 14]
    println!("{:?}", discrete_log(3, 13, 17));           // Some(4): 3⁴ ≡ 13 (mod 17)
    println!("{:?}", primitive_root(17));                // Some(3)
    println!("{:?}", multiplicative_order(2, 7));        // Some(3)
    println!("{} {:?} {}", legendre_symbol(2, 7), jacobi_symbol(1001, 9907), kronecker_symbol(3, 8));   // 1 Ok(-1) -1
}
```

## Continued fractions and Egyptian fractions

```rust
use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::ntheory::*;

fn main() {
    let r = Ratio::new(BigInt::from(415), BigInt::from(93));
    println!("{:?}", continued_fraction(&r));                  // [4, 2, 6, 7]
    println!("{:?}", continued_fraction_periodic(23));         // Some(([4], [1, 3, 1, 8]))  √23
    let terms: Vec<BigInt> = [3, 7, 15, 1].iter().map(|&k| BigInt::from(k)).collect();
    println!("{:?}", continued_fraction_convergents(&terms));  // 3, 22/7, 333/106, 355/113
    println!("{:?}", egyptian_fraction(&Ratio::new(BigInt::from(4), BigInt::from(13))));   // Some([4, 18, 468])
}
```

## Diophantine equations

```rust
use symplex::diophantine::*;

fn main() {
    println!("{:?}", linear_diophantine(3, 5, 1));       // Some((2, -1, 5, -3)): x = 2 + 5k, y = −1 − 3k
    println!("{:?}", pell(61));                          // Some((1766319049, 226153980))
    println!("{:?}", pell_solutions(2, 4));              // [(3, 2), (17, 12), (99, 70), (577, 408)]
    println!("{:?}", pell_negative(5));                  // x² − 5y² = −1
    println!("{:?}", sum_of_two_squares(65));            // Some((4, 7))
    println!("{:?}", sum_of_two_squares(2021));          // None (43·47, both ≡ 3 mod 4)
    println!("{:?}", sum_of_four_squares(7));
    println!("{:?}", pythagorean_triples(30));           // primitive triples with c ≤ 30
    println!("{:?}", frobenius_number(&[6, 9, 20]));     // Some(43)  (Chicken McNugget)
}
```

## Sequences and combinatorics

```rust
use symplex::combinatorics::*;
use symplex::ntheory;
use symplex::prelude::*;

fn main() {
    println!("{}", ntheory::fibonacci(100));                  // 354224848179261915075
    println!("{:?}", ntheory::bernoulli(12));                 // Some(-691/2730)
    println!("{:?}", ntheory::euler_number(10));              // Some(-50521)
    println!("{:?}", ntheory::harmonic(10));                  // Some(7381/2520)
    println!("{:?}", stirling2(10, 4));                       // Some(34105)
    println!("{:?}", stirling1(5, 2));                        // signed Stirling numbers of the first kind
    println!("{:?}", bell(10));                               // Some(115975)
    println!("{:?}", catalan(10));                            // Some(16796)
    println!("{:?}", derangements(10));                       // Some(1334961)
    println!("{:?}", partition_count(100));                   // Some(190569292)
    println!("{:?}", partitions(5).collect::<Vec<_>>());      // all partitions of 5
    println!("{:?}", multinomial(6, &[2, 2, 2]));             // Some(90)

    // Symbolic: stays a node until the argument is concrete
    let ctx = Context::new();
    symplex::syms!(ctx; n);
    println!("{} {}", n.fibonacci(), ctx.int(30).fibonacci().eval());     // fibonacci(n) 832040
    println!("{}", ctx.int(10).bell().eval());                             // 115975
}
```

## Polynomial factoring and algebra

Factoring over ℤ (Berlekamp–Zassenhaus), multivariate factoring, resultants, discriminants, square-free decomposition, root isolation and the rest of the polynomial toolbox are covered in [Algebra](./algebra.md#factoring).

See `cargo run --example factoring_and_ntheory`, `number_theory` and `crypto_rsa`.
