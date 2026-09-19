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

## More number theory and discrete transforms (0.9.1)

0.9.1 fills in the rest of SymPy's `ntheory` residue toolbox and adds an exact `symplex::discrete` module.

### Higher power residues and polynomial congruences

`nthroot_mod(a, n, m, all_roots)` solves `xⁿ ≡ a (mod m)` for **any** modulus: `m` is factored, each prime is handled with Johnston's generalised `q`-th root algorithm (a primitive root plus discrete logarithms *only inside the Sylow subgroups for the primes dividing `gcd(n, p−1)`*, so `p` may be huge as long as those primes are moderate), roots are Hensel-lifted to prime powers and combined by CRT. `n = 2` is `sqrt_mod_all`. The result is `None` when there is no root, otherwise the sorted roots (or just the smallest one).

```rust
use num_bigint::BigInt;
use symplex::ntheory::*;

fn main() {
    println!("{:?}", nthroot_mod(11, 4, 19, true));          // Some([8, 11])          x⁴ ≡ 11 (mod 19)
    println!("{:?}", nthroot_mod(68, 3, 109, false));        // Some([23])
    println!("{:?}", nthroot_mod(2, 3, 7, true));            // None — 2 is not a cube mod 7
    println!("{:?}", nthroot_mod(16, 4, 35, true));          // Some([2, 9, 12, 16, 19, 23, 26, 33])
    let m127 = (BigInt::from(1) << 127) - 1;
    println!("{:?}", nthroot_mod(8, 3, m127, false));        // Some([2])   cube roots modulo 2¹²⁷ − 1

    println!("{:?}", quadratic_residues(7));                 // [0, 1, 2, 4]
    println!("{} {}", is_nthpow_residue(2, 4, 7), is_nthpow_residue(2, 3, 7));   // true false

    // Roots of x⁶ − 2x⁵ − 35 modulo 6125 = 5³·7² (coefficients highest degree first)
    let f: Vec<BigInt> = [1, -2, 0, 0, 0, 0, -35].iter().map(|&c| BigInt::from(c)).collect();
    println!("{:?}", polynomial_congruence(&f, 6125));       // [3257]
    let g: Vec<BigInt> = [1, 0, 0, -3, 5].iter().map(|&c| BigInt::from(c)).collect();
    println!("{:?}", polynomial_congruence(&g, 1_000_003));  // [357940, 847957]  (Cantor–Zassenhaus mod a large prime)
}
```

`polynomial_congruence` solves linear and quadratic congruences and monic binomials `xⁿ − a` for any factorable modulus; for other polynomials it finds the roots modulo each prime `p | m` (brute force for `p ≤ 2¹⁶`, `gcd(f, xᵖ − x)` plus Cantor–Zassenhaus splitting for `2¹⁶ < p < 2⁶³`), Hensel-lifts them and combines them by CRT. A prime factor `p ≥ 2⁶³` in that general case is not supported and gives an empty result.

### Arithmetic functions

```rust
use symplex::ntheory::*;

fn main() {
    println!("{} {} {}", multiplicity(2, 40), primenu(72), primeomega(72));   // 3 2 5
    println!("{} {}", primorial(5), primorial_up_to(10));                     // 2310 210
    println!("{} {}", is_carmichael(561), is_carmichael(563));                // true false
    println!("{}", is_amicable(220, 284));                                    // true
    println!("{:?}", binomial_coefficients_list(4));                          // [1, 4, 6, 4, 1]
    println!("{:?}", binomial_coefficients(3));                               // [((0, 3), 1), ((1, 2), 3), ((2, 1), 3), ((3, 0), 1)]
}
```

### Continued fraction reduction

`continued_fraction_reduce` is the inverse of `continued_fraction`: a finite `[a₀; a₁, …]` back to a rational. The periodic form `(pre, period)` returned by `continued_fraction_periodic` is reduced by `continued_fraction_reduce_periodic` to the integer triple `(p, q, d)` meaning `(p + √d)/q` (`q` may be negative — that is how a negative radical coefficient is encoded); `continued_fraction_reduce_periodic_ex` builds the same value as an `Ex`, which canonicalises it.

```rust
use num_bigint::BigInt;
use symplex::ntheory::*;
use symplex::prelude::*;

fn main() {
    let cf: Vec<BigInt> = [4, 2, 6, 7].iter().map(|&t| BigInt::from(t)).collect();
    println!("{:?}", continued_fraction_reduce(&cf));                        // Some(415/93)

    let (pre, period) = continued_fraction_periodic(23).unwrap();            // ([4], [1, 3, 1, 8])
    println!("{:?}", continued_fraction_reduce_periodic(&pre, &period));     // Some((0, 1, 23))  = √23
    let one: Vec<BigInt> = vec![BigInt::from(1)];
    println!("{:?}", continued_fraction_reduce_periodic(&[], &one));         // Some((1, 2, 5))   = (1 + √5)/2

    let ctx = Context::new();
    let pre: Vec<BigInt> = [1, 2, 3].iter().map(|&t| BigInt::from(t)).collect();
    let per: Vec<BigInt> = [4, 5].iter().map(|&t| BigInt::from(t)).collect();
    println!("{}", continued_fraction_reduce_periodic_ex(&ctx, &pre, &per).unwrap());   // -1/52*sqrt(30) + 20/13  = (80 − √30)/52
}
```

### Discrete transforms (`symplex::discrete`)

Everything in `symplex::discrete` is exact: sequences are `Ratio<BigInt>` (or `BigInt` residues for the NTT). There is deliberately no floating-point FFT and no symbolic DFT over `Ex` roots of unity — `convolution` *is* exact polynomial multiplication, and `convolution_ex` does the same on symbolic `Ex` coefficients. Power-of-two transforms zero-pad their input like SymPy.

```rust
use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::discrete::*;

fn main() {
    let q = |v: &[i64]| -> Vec<Ratio<BigInt>> { v.iter().map(|&t| Ratio::from_integer(BigInt::from(t))).collect() };
    let b = |v: &[i64]| -> Vec<BigInt> { v.iter().map(|&t| BigInt::from(t)).collect() };

    println!("{:?}", convolution(&q(&[1, 2, 3]), &q(&[4, 5, 6])));            // [4, 13, 28, 27, 18]
    println!("{:?}", convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 3));  // [31, 31, 28]
    println!("{:?}", convolution_subset(&q(&[1, 2, 3, 4]), &q(&[5, 6, 7, 8])));   // [5, 16, 22, 60]

    // Number-theoretic transform modulo 998244353 = 119·2²³ + 1 (root 3, as in SymPy)
    let t = ntt(&b(&[1, 2, 3, 4]), 998_244_353).unwrap();
    println!("{:?}", t);                                                     // [10, 173167434, 998244351, 825076915]
    println!("{:?}", intt(&t, 998_244_353).unwrap());                        // [1, 2, 3, 4]
    println!("{:?}", convolution_ntt(&b(&[1, 2, 3]), &b(&[4, 5, 6]), 998_244_353).unwrap());   // [4, 13, 28, 27, 18]
    println!("{}", ntt(&b(&[1, 2, 3, 4]), 7).is_err());                     // true — 4 ∤ 7 − 1

    println!("{:?}", fwht(&q(&[1, 2, 3, 4])));                               // [10, -2, -4, 0]
    println!("{:?}", ifwht(&q(&[10, -2, -4, 0])));                           // [1, 2, 3, 4]
    println!("{:?}", mobius_transform(&q(&[1, 2, 3, 4])));                   // [1, 3, 4, 10]   subset sums
    println!("{:?}", inverse_mobius_transform(&q(&[1, 3, 4, 10])));          // [1, 2, 3, 4]
    println!("{:?}", mobius_transform_superset(&q(&[1, 2, 3, 4])));          // [10, 6, 7, 4]   superset sums
}
```
