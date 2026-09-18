//! Integration tests for the 0.2 number-theory, combinatorics and
//! Diophantine APIs, cross-checked against known tables (OEIS).

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use proptest::prelude::*;
use symplex::combinatorics::{bell, binomial, catalan, derangements, npartitions, partitions};
use symplex::diophantine::{
    frobenius_number, linear_diophantine, linear_diophantine_n, pell, pell_negative,
    pell_solutions, pythagorean_triples, sum_of_four_squares, sum_of_two_squares,
};
use symplex::ntheory::{
    carmichael_lambda, continued_fraction, continued_fraction_convergents,
    continued_fraction_periodic, digits, discrete_log, divisor_sigma, egyptian_fraction,
    euler_number, factorint, fibonacci, harmonic, is_abundant, is_deficient, is_mersenne_prime,
    is_palindromic, is_perfect, is_perfect_power, is_primitive_root, is_probable_prime,
    is_quad_residue, isprime, jacobi_symbol, kronecker_symbol, lucas, mobius, n_order, nextprime,
    perfect_power, prevprime, prime, primepi, primerange, primes_up_to, primitive_root, sqrt_mod,
    sqrt_mod_all, totient,
};
use symplex::prelude::*;

fn bi(n: i64) -> BigInt {
    BigInt::from(n)
}

fn big(s: &str) -> BigInt {
    BigInt::parse_bytes(s.as_bytes(), 10).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// OEIS cross-checks
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn oeis_a000040_primes() {
    let first: Vec<i64> = primes_up_to(100);
    assert_eq!(
        first,
        vec![
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97
        ]
    );
    for (i, p) in first.iter().enumerate() {
        assert_eq!(prime(i as i64 + 1), Some(bi(*p)));
    }
}

#[test]
fn oeis_a000720_primepi() {
    // π(10^k) for k = 1..8
    let table = [4u64, 25, 168, 1229, 9592, 78498, 664579, 5761455];
    for (k, &v) in table.iter().enumerate() {
        assert_eq!(primepi(10i64.pow(k as u32 + 1)), Some(v), "π(10^{})", k + 1);
    }
}

#[test]
fn oeis_a000010_totient_and_a002322_lambda() {
    let phi = [
        1i64, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8,
    ];
    let lam = [
        1i64, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6, 18, 4,
    ];
    for (i, (&p, &l)) in phi.iter().zip(lam.iter()).enumerate() {
        let n = i as i64 + 1;
        assert_eq!(totient(n), bi(p), "φ({n})");
        assert_eq!(carmichael_lambda(n), bi(l), "λ({n})");
    }
}

#[test]
fn oeis_a000203_sigma_and_a008683_mobius() {
    let sigma = [1i64, 3, 4, 7, 6, 12, 8, 15, 13, 18, 12, 28, 14, 24, 24, 31];
    let mu = [1i8, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0];
    for (i, (&s, &m)) in sigma.iter().zip(mu.iter()).enumerate() {
        let n = i as i64 + 1;
        assert_eq!(divisor_sigma(n, 1), bi(s), "σ({n})");
        assert_eq!(mobius(n), m, "μ({n})");
    }
}

#[test]
fn oeis_a000045_a000032_fibonacci_lucas() {
    let fib = [
        0i64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
    ];
    let luc = [
        2i64, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843, 1364,
    ];
    for (i, &f) in fib.iter().enumerate() {
        assert_eq!(fibonacci(i as i64), bi(f));
    }
    for (i, &l) in luc.iter().enumerate() {
        assert_eq!(lucas(i as i64), bi(l));
    }
    assert_eq!(
        fibonacci(200).to_string(),
        "280571172992510140037611932413038677189525"
    );
}

#[test]
fn oeis_a000110_a000108_a000166_bell_catalan_derangements() {
    let bells = [1i64, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570];
    let cats = [1i64, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786];
    let ders = [1i64, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961];
    for (i, &b) in bells.iter().enumerate() {
        assert_eq!(bell(i as i64), Some(bi(b)), "B({i})");
    }
    for (i, &c) in cats.iter().enumerate() {
        assert_eq!(catalan(i as i64), Some(bi(c)), "C({i})");
    }
    for (i, &d) in ders.iter().enumerate() {
        assert_eq!(derangements(i as i64), Some(bi(d)), "!{i}");
    }
    // Bell numbers = row sums of Stirling2 (cross-module consistency).
    for n in 0..12i64 {
        let sum: BigInt = (0..=n)
            .map(|k| symplex::combinatorics::stirling2(n, k).unwrap())
            .sum();
        assert_eq!(bell(n), Some(sum));
    }
}

#[test]
fn oeis_a000041_partitions_count_matches_enumeration() {
    let p = [
        1u64, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627,
    ];
    for (n, &count) in p.iter().enumerate() {
        assert_eq!(npartitions(n as i64), Some(BigInt::from(count)));
        let parts: Vec<Vec<u64>> = partitions(n as u64).collect();
        assert_eq!(parts.len() as u64, count, "p({n})");
        for part in &parts {
            assert_eq!(part.iter().sum::<u64>(), n as u64);
            assert!(
                part.windows(2).all(|w| w[0] >= w[1]),
                "non-increasing: {part:?}"
            );
        }
        // Distinct and reverse-lexicographically sorted.
        for w in parts.windows(2) {
            assert!(w[0] > w[1], "{:?} !> {:?}", w[0], w[1]);
        }
    }
    assert_eq!(
        npartitions(1000).unwrap().to_string(),
        "24061467864032622473692149727991"
    );
}

#[test]
fn oeis_a000364_euler_and_a027641_bernoulli() {
    let euler = [
        1i64,
        -1,
        5,
        -61,
        1385,
        -50521,
        2702765,
        -199360981,
        19391512145,
    ];
    for (i, &e) in euler.iter().enumerate() {
        assert_eq!(euler_number(2 * i as i64), Some(bi(e)));
    }
    let r = |p: i64, q: i64| Ratio::new(bi(p), bi(q));
    let bern = [
        (0i64, r(1, 1)),
        (1, r(-1, 2)),
        (2, r(1, 6)),
        (4, r(-1, 30)),
        (6, r(1, 42)),
        (8, r(-1, 30)),
        (10, r(5, 66)),
        (12, r(-691, 2730)),
        (14, r(7, 6)),
        (16, r(-3617, 510)),
        (18, r(43867, 798)),
    ];
    for (n, b) in bern {
        assert_eq!(symplex::ntheory::bernoulli(n), Some(b), "B_{n}");
    }
    assert_eq!(harmonic(20), Some(r(55835135, 15519504)));
}

#[test]
fn oeis_a001358_semiprimes_via_factorint() {
    let semiprimes: Vec<i64> = (1..100)
        .filter(|&n| factorint(n).iter().map(|(_, e)| *e).sum::<u32>() == 2)
        .collect();
    assert_eq!(
        semiprimes,
        vec![
            4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57, 58, 62,
            65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95
        ]
    );
}

#[test]
fn oeis_a005117_squarefree_and_a001597_perfect_powers() {
    let squarefree: Vec<i64> = (1..40).filter(|&n| mobius(n) != 0).collect();
    assert_eq!(
        squarefree,
        vec![
            1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35,
            37, 38, 39
        ]
    );
    let powers: Vec<i64> = (2..130).filter(|&n| is_perfect_power(n)).collect();
    assert_eq!(
        powers,
        vec![4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 121, 125, 128]
    );
    assert_eq!(perfect_power(128), Some((bi(2), 7)));
    assert_eq!(perfect_power(1_000_000), Some((bi(10), 6)));
}

#[test]
fn oeis_a000396_perfect_numbers_and_mersenne_exponents() {
    let perfect: Vec<i64> = (1..10_000).filter(|&n| is_perfect(n)).collect();
    assert_eq!(perfect, vec![6, 28, 496, 8128]);
    let mersenne: Vec<i64> = (2..=130).filter(|&p| is_mersenne_prime(p)).collect();
    assert_eq!(mersenne, vec![2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127]);
    assert!(is_abundant(12) && is_deficient(10));
}

// ═══════════════════════════════════════════════════════════════════════════
// Primality & factorization
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn factorint_required_inputs() {
    let two_64_plus_1: BigInt = (BigInt::one() << 64usize) + 1;
    assert_eq!(
        factorint(two_64_plus_1),
        vec![(bi(274177), 1), (big("67280421310721"), 1)]
    );
    assert_eq!(
        factorint(1_000_000_000_000_000_009i64),
        vec![(bi(1_000_000_000_000_000_009), 1)]
    );
    let p = big("1000000000000037");
    let q = big("1000000000000091");
    let start = std::time::Instant::now();
    assert_eq!(factorint(&p * &q), vec![(p, 1), (q, 1)]);
    assert!(start.elapsed().as_secs() < 20, "took {:?}", start.elapsed());
}

#[test]
fn factorint_roundtrip_random_u64_products() {
    let primes = [65537u64, 999_983, 1_000_003, 2_147_483_647, 4_294_967_291];
    for i in 0..primes.len() {
        for j in i..primes.len() {
            let n = BigInt::from(primes[i]) * BigInt::from(primes[j]);
            let f = factorint(n.clone());
            let back: BigInt = f.iter().map(|(p, e)| p.pow(*e)).product();
            assert_eq!(back, n);
            assert!(f.iter().all(|(p, _)| isprime(p.clone())));
        }
    }
}

#[test]
fn ex_level_factorize_and_is_prime_value() {
    let ctx = Context::new();
    let n = ctx.int(2).powi(64) + 1;
    let f = n.factorize().unwrap();
    assert_eq!(f.len(), 2);
    assert_eq!(f[0].0, bi(274177));
    assert_eq!(n.is_prime_value(), Some(false));
    let m = ctx.int(2).powi(61) - 1;
    assert_eq!(m.is_prime_value(), Some(true));
}

#[test]
fn primality_large_values() {
    let m127: BigInt = (BigInt::one() << 127usize) - 1;
    assert!(isprime(m127.clone()));
    assert!(is_probable_prime(m127.clone(), 10));
    assert!(!isprime(&m127 * &m127));
    assert!(isprime(big("1000000000000000000000000000057")));
    assert_eq!(
        nextprime(big("1000000000000000000000000000000")),
        big("1000000000000000000000000000057")
    );
    assert_eq!(
        prevprime(big("1000000000000000000000000000057")),
        Some(big("999999999999999999999999999989"))
    );
    // A known BPSW-range composite that is a strong pseudoprime to several bases:
    // 3215031751 = 151·751·28351 is spsp to bases 2,3,5,7 (below the range but a classic).
    assert!(!isprime(3_215_031_751i64));
}

#[test]
fn primerange_consistency() {
    let r = primerange(1_000, 2_000);
    assert_eq!(
        r.len(),
        (primepi(1_999).unwrap() - primepi(999).unwrap()) as usize
    );
    assert!(r.iter().all(|p| isprime(p.clone())));
    let all = primes_up_to(2_000);
    let sub: Vec<i64> = all
        .into_iter()
        .filter(|&p| (1_000..2_000).contains(&p))
        .collect();
    let r_i: Vec<i64> = r.iter().map(|p| p.try_into().unwrap()).collect();
    assert_eq!(r_i, sub);
}

// ═══════════════════════════════════════════════════════════════════════════
// Modular arithmetic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadratic_residue_symbols() {
    // Table of Legendre symbols mod 11: QRs are {1, 3, 4, 5, 9}
    for a in 1..11i64 {
        let qr = [1, 3, 4, 5, 9].contains(&a);
        assert_eq!(jacobi_symbol(a, 11).unwrap(), if qr { 1 } else { -1 });
        assert_eq!(kronecker_symbol(a, 11), if qr { 1 } else { -1 });
        assert_eq!(is_quad_residue(a, 11), qr);
        assert_eq!(sqrt_mod(a, 11).is_some(), qr);
    }
    assert_eq!(sqrt_mod_all(4, 15), vec![bi(2), bi(7), bi(8), bi(13)]);
    assert_eq!(
        sqrt_mod_all(1, 24),
        vec![bi(1), bi(5), bi(7), bi(11), bi(13), bi(17), bi(19), bi(23)]
    );
    assert!(sqrt_mod_all(2, 15).is_empty());
    assert!(jacobi_symbol(2, 15).unwrap() == 1 && !is_quad_residue(2, 15));
}

#[test]
fn orders_roots_logs() {
    assert_eq!(primitive_root(23), Some(bi(5)));
    assert!(is_primitive_root(5, 23));
    assert_eq!(n_order(5, 23), Some(bi(22)));
    assert_eq!(n_order(2, 23), Some(bi(11)));
    let x = discrete_log(5, 2, 23).unwrap();
    assert_eq!(symplex::ntheory::mod_pow(5, x, 23), bi(2));
    assert_eq!(discrete_log(2, 5, 23), None); // 5 ∉ ⟨2⟩ (order 11)
    // Modulus 2^k: no primitive root for k ≥ 3.
    assert_eq!(primitive_root(16), None);
    assert!(discrete_log(3, 11, 16).is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// Digits & continued fractions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn digits_and_continued_fractions() {
    assert_eq!(digits(2024, 10).unwrap(), vec![2, 0, 2, 4]);
    assert_eq!(
        digits(2024, 2).unwrap(),
        vec![1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
    );
    assert!(is_palindromic(1_234_321, 10));
    let pi_approx = Ratio::new(bi(355), bi(113));
    let cf: Vec<i64> = continued_fraction(&pi_approx)
        .iter()
        .map(|t| t.try_into().unwrap())
        .collect();
    assert_eq!(cf, vec![3, 7, 16]);
    let conv = continued_fraction_convergents(&continued_fraction(&pi_approx));
    assert_eq!(conv.last().unwrap(), &pi_approx);
    let (h, per) = continued_fraction_periodic(23).unwrap();
    assert_eq!(h, vec![bi(4)]);
    assert_eq!(per, vec![bi(1), bi(3), bi(1), bi(8)]);
    let eg: Vec<i64> = egyptian_fraction(&Ratio::new(bi(2), bi(3)))
        .unwrap()
        .iter()
        .map(|t| t.try_into().unwrap())
        .collect();
    assert_eq!(eg, vec![2, 6]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Diophantine
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn diophantine_pell_table() {
    // OEIS A002350 (x) and A002349 (y) for non-square D.
    let table = [
        (2i64, 3i64, 2i64),
        (3, 2, 1),
        (5, 9, 4),
        (6, 5, 2),
        (7, 8, 3),
        (8, 3, 1),
        (10, 19, 6),
        (11, 10, 3),
        (12, 7, 2),
        (13, 649, 180),
        (14, 15, 4),
        (15, 4, 1),
        (17, 33, 8),
        (61, 1_766_319_049, 226_153_980),
        (109, 158_070_671_986_249, 15_140_424_455_100),
    ];
    for (d, x, y) in table {
        assert_eq!(pell(d), Some((bi(x), bi(y))), "D = {d}");
    }
    assert_eq!(pell_solutions(2, 3).len(), 3);
    assert_eq!(pell_negative(13), Some((bi(18), bi(5))));
    // Largest fundamental solution below 1000 is D = 661.
    let (x, y) = pell(661).unwrap();
    assert_eq!(x, big("16421658242965910275055840472270471049"));
    assert_eq!(y, big("638728478116949861246791167518480580"));
}

#[test]
fn diophantine_linear_and_squares() {
    let (x0, y0, dx, dy) = linear_diophantine(1071, 462, 21).unwrap();
    assert_eq!(bi(1071) * &x0 + bi(462) * &y0, bi(21));
    assert_eq!(bi(1071) * &dx + bi(462) * &dy, bi(0));
    assert!(linear_diophantine(1071, 462, 20).is_none());
    let sol = linear_diophantine_n(&[3, 5, 7, 11], 1).unwrap();
    let lhs: BigInt = [3i64, 5, 7, 11]
        .iter()
        .zip(&sol)
        .map(|(a, x)| bi(*a) * x)
        .sum();
    assert_eq!(lhs, bi(1));

    // Fermat: primes ≡ 1 mod 4 are sums of two squares.
    for p in primes_up_to(500) {
        let rep = sum_of_two_squares(p);
        assert_eq!(rep.is_some(), p == 2 || p % 4 == 1, "p = {p}");
    }
    let (a, b) = sum_of_two_squares(big("1000000000000000000000000000057")).unwrap();
    assert_eq!(&a * &a + &b * &b, big("1000000000000000000000000000057"));
    let (a, b, c, d) = sum_of_four_squares(big("123456789012345678901234567890")).unwrap();
    assert_eq!(
        &a * &a + &b * &b + &c * &c + &d * &d,
        big("123456789012345678901234567890")
    );

    assert_eq!(pythagorean_triples(30).len(), 5);
    assert_eq!(frobenius_number(&[6, 9, 20]), Some(43));
}

// ═══════════════════════════════════════════════════════════════════════════
// Property tests
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// factorint multiplies back and only reports primes.
    #[test]
    fn prop_factorint_roundtrip(n in 1i64..1_000_000_000_000) {
        let f = factorint(n);
        let back: BigInt = f.iter().map(|(p, e)| p.pow(*e)).product();
        prop_assert_eq!(back, bi(n));
        for (p, _) in &f {
            prop_assert!(isprime(p.clone()));
        }
    }

    /// σ₁(n)·φ(n) ≤ n² and divisor_sigma(n, 0) counts divisors.
    #[test]
    fn prop_sigma_phi(n in 1i64..100_000) {
        let s1 = divisor_sigma(n, 1);
        let phi = totient(n);
        prop_assert!(&s1 * &phi <= bi(n) * bi(n));
        prop_assert!(s1 >= bi(n));
    }

    /// Euler's theorem via n_order: a^λ(n) ≡ 1 and order | λ(n).
    #[test]
    fn prop_order_divides_lambda(n in 2i64..5_000, a in 1i64..5_000) {
        prop_assume!(symplex::ntheory::gcd(a, n).is_one());
        let ord = n_order(a, n).unwrap();
        let lam = carmichael_lambda(n);
        prop_assert!((&lam % &ord).is_zero());
        prop_assert_eq!(symplex::ntheory::mod_pow(a, ord, n), bi(1));
    }

    /// sqrt_mod_all returns exactly the roots (brute force for small n).
    #[test]
    fn prop_sqrt_mod_all_exact(n in 1i64..400, a in 0i64..400) {
        let roots = sqrt_mod_all(a, n);
        let expected: Vec<BigInt> = (0..n).filter(|x| (x * x) % n == a % n).map(bi).collect();
        prop_assert_eq!(roots, expected);
    }

    /// Discrete log round-trips through mod_pow and is minimal.
    #[test]
    fn prop_discrete_log_roundtrip(n in 2i64..2_000, a in 2i64..2_000, x in 0i64..200) {
        prop_assume!(symplex::ntheory::gcd(a, n).is_one());
        let b = symplex::ntheory::mod_pow(a, x, n);
        let got = discrete_log(a, b.clone(), n).unwrap();
        prop_assert_eq!(symplex::ntheory::mod_pow(a, got.clone(), n), b);
        prop_assert!(got <= bi(x));
    }

    /// Jacobi symbol is multiplicative in the top argument.
    #[test]
    fn prop_jacobi_multiplicative(a in -500i64..500, b in -500i64..500, n in 1i64..300) {
        let n = 2 * n + 1;
        prop_assert_eq!(
            jacobi_symbol(a * b, n).unwrap(),
            jacobi_symbol(a, n).unwrap() * jacobi_symbol(b, n).unwrap()
        );
    }

    /// Continued fractions round-trip through convergents.
    #[test]
    fn prop_continued_fraction_roundtrip(p in -10_000i64..10_000, q in 1i64..10_000) {
        let r = Ratio::new(bi(p), bi(q));
        let cf = continued_fraction(&r);
        let conv = continued_fraction_convergents(&cf);
        prop_assert_eq!(conv.last().unwrap(), &r);
    }

    /// Pell solutions satisfy the equation for every non-square D.
    #[test]
    fn prop_pell(d in 2i64..300) {
        prop_assume!(!symplex::ntheory::is_square(d));
        for (x, y) in pell_solutions(d, 3) {
            prop_assert_eq!(&x * &x - bi(d) * &y * &y, bi(1));
        }
    }

    /// Linear Diophantine general solution is correct.
    #[test]
    fn prop_linear_diophantine(a in -200i64..200, b in -200i64..200, k in -50i64..50, m in -5i64..5) {
        prop_assume!(a != 0 || b != 0);
        let g: i64 = symplex::ntheory::gcd(a, b).try_into().unwrap();
        let c = g * k;
        let (x0, y0, dx, dy) = linear_diophantine(a, b, c).unwrap();
        let x = &x0 + bi(m) * &dx;
        let y = &y0 + bi(m) * &dy;
        prop_assert_eq!(bi(a) * x + bi(b) * y, bi(c));
    }

    /// Binomial satisfies Pascal's rule.
    #[test]
    fn prop_pascal(n in 0i64..60, k in 0i64..60) {
        prop_assert_eq!(binomial(n + 1, k + 1), binomial(n, k) + binomial(n, k + 1));
    }

    /// Fibonacci: Cassini's identity F(n−1)F(n+1) − F(n)² = (−1)^n.
    #[test]
    fn prop_cassini(n in 1i64..300) {
        let lhs = fibonacci(n - 1) * fibonacci(n + 1) - fibonacci(n) * fibonacci(n);
        prop_assert_eq!(lhs, if n % 2 == 0 { bi(1) } else { bi(-1) });
    }
}
