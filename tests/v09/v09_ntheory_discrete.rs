//! symplex 0.9.1 — more number theory (`nthroot_mod`, `quadratic_residues`,
//! `is_nthpow_residue`, `polynomial_congruence`, `multiplicity`, `primenu`,
//! `primeomega`, `primorial`, `continued_fraction_reduce`, `is_carmichael`,
//! `is_amicable`, `binomial_coefficients`) and the exact discrete
//! transforms in `symplex::discrete` (`convolution*`, `ntt`/`intt`,
//! `fwht`/`ifwht`, `mobius_transform`).
//!
//! Reference values are from SymPy 1.14 (`symplex/.venv`), cited inline.
//! Where SymPy is too slow or the answer is a mathematical identity
//! (round trips, brute-force enumeration) the test checks the identity.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Zero};
use symplex::discrete::*;
use symplex::ntheory::*;
use symplex::prelude::*;

fn b(v: i64) -> BigInt {
    BigInt::from(v)
}

fn bs(v: &[i64]) -> Vec<BigInt> {
    v.iter().map(|&t| BigInt::from(t)).collect()
}

fn q(v: &[i64]) -> Vec<Ratio<BigInt>> {
    v.iter()
        .map(|&t| Ratio::from_integer(BigInt::from(t)))
        .collect()
}

fn r(p: i64, d: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(p), BigInt::from(d))
}

/// `base^exp mod m` on machine words (`m < 2³²` in these tests).
fn pow_mod_u64(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base % m;
        }
        base = base * base % m;
        exp >>= 1;
    }
    acc
}

/// Every `x ∈ [0, m)` with `xⁿ ≡ a (mod m)`, by exhaustive search.
fn brute_nthroots(a: u64, n: u32, m: u64) -> Vec<BigInt> {
    (0..m)
        .filter(|&x| pow_mod_u64(x, u64::from(n), m) == a % m)
        .map(BigInt::from)
        .collect()
}

/// Every `x ∈ [0, m)` with `f(x) ≡ 0 (mod m)` (coefficients highest first).
fn brute_poly_roots(coeffs: &[i64], m: u64) -> Vec<BigInt> {
    let mb = BigInt::from(m);
    (0..m)
        .filter(|&x| {
            let xb = BigInt::from(x);
            let mut acc = BigInt::zero();
            for &c in coeffs {
                acc = (acc * &xb + BigInt::from(c)).mod_floor(&mb);
            }
            acc.is_zero()
        })
        .map(BigInt::from)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// nthroot_mod
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nthroot_mod_sympy_doc_values() {
    // sympy: nthroot_mod(11, 4, 19, True) == [8, 11]; nthroot_mod(11, 4, 19) == 8
    assert_eq!(nthroot_mod(11, 4, 19, true), Some(bs(&[8, 11])));
    assert_eq!(nthroot_mod(11, 4, 19, false), Some(bs(&[8])));
    // sympy: nthroot_mod(68, 3, 109, True) == [23, 32, 54]
    assert_eq!(nthroot_mod(68, 3, 109, true), Some(bs(&[23, 32, 54])));
    // sympy: nthroot_mod(1, 4, 17, True) == [1, 4, 13, 16]
    assert_eq!(nthroot_mod(1, 4, 17, true), Some(bs(&[1, 4, 13, 16])));
    // sympy: nthroot_mod(16, 4, 17, True) == [2, 8, 9, 15]
    assert_eq!(nthroot_mod(16, 4, 17, true), Some(bs(&[2, 8, 9, 15])));
    // sympy: nthroot_mod(16, 8, 17, True) == [3, 5, 6, 7, 10, 11, 12, 14]
    assert_eq!(
        nthroot_mod(16, 8, 17, true),
        Some(bs(&[3, 5, 6, 7, 10, 11, 12, 14]))
    );
    // sympy: nthroot_mod(1, 9, 37, True) == [1, 7, 9, 10, 12, 16, 26, 33, 34]
    assert_eq!(
        nthroot_mod(1, 9, 37, true),
        Some(bs(&[1, 7, 9, 10, 12, 16, 26, 33, 34]))
    );
}

#[test]
fn nthroot_mod_no_solution_and_edge_cases() {
    // sympy: nthroot_mod(2, 3, 7, True) == []  (and None without all_roots)
    assert_eq!(nthroot_mod(2, 3, 7, true), None);
    assert_eq!(nthroot_mod(2, 3, 7, false), None);
    // sympy: nthroot_mod(2, 8, 17, True) == []; nthroot_mod(10, 6, 19, True) == []
    assert_eq!(nthroot_mod(2, 8, 17, true), None);
    assert_eq!(nthroot_mod(10, 6, 19, true), None);
    // sympy: nthroot_mod(0, 3, 7, True) == [0]; nthroot_mod(5, 1, 7, True) == [5]
    assert_eq!(nthroot_mod(0, 3, 7, true), Some(bs(&[0])));
    assert_eq!(nthroot_mod(5, 1, 7, true), Some(bs(&[5])));
    // sympy: nthroot_mod(4, 2, 7, True) == [2, 5]   (n = 2 → sqrt_mod_all)
    assert_eq!(nthroot_mod(4, 2, 7, true), Some(bs(&[2, 5])));
    // sympy: nthroot_mod(5, 5, 1, True) == [0]; nthroot_mod(7, 5, 2, True) == [1]
    assert_eq!(nthroot_mod(5, 5, 1, true), Some(bs(&[0])));
    assert_eq!(nthroot_mod(7, 5, 2, true), Some(bs(&[1])));
    // sympy: nthroot_mod(1, 16, 17, True) == list(range(1, 17))
    assert_eq!(
        nthroot_mod(1, 16, 17, true),
        Some((1..17).map(BigInt::from).collect())
    );
    // invalid exponent / modulus
    assert_eq!(nthroot_mod(1, 0, 17, true), None);
    assert_eq!(nthroot_mod(1, 3, 0, true), None);
}

#[test]
fn nthroot_mod_composite_and_prime_power_moduli() {
    // sympy: nthroot_mod(16, 4, 35, True) == [2, 9, 12, 16, 19, 23, 26, 33]
    assert_eq!(
        nthroot_mod(16, 4, 35, true),
        Some(bs(&[2, 9, 12, 16, 19, 23, 26, 33]))
    );
    // sympy: nthroot_mod(1, 4, 15, True) == [1, 2, 4, 7, 8, 11, 13, 14]
    assert_eq!(
        nthroot_mod(1, 4, 15, true),
        Some(bs(&[1, 2, 4, 7, 8, 11, 13, 14]))
    );
    // sympy: nthroot_mod(1, 3, 63, True) == [1, 4, 16, 22, 25, 37, 43, 46, 58]
    assert_eq!(
        nthroot_mod(1, 3, 63, true),
        Some(bs(&[1, 4, 16, 22, 25, 37, 43, 46, 58]))
    );
    // Hensel with p | n (exhaustive lift): sympy nthroot_mod(8, 3, 27, True) == [2, 11, 20]
    assert_eq!(nthroot_mod(8, 3, 27, true), Some(bs(&[2, 11, 20])));
    // sympy: nthroot_mod(1, 3, 27, True) == [1, 10, 19]; nthroot_mod(1, 3, 9, True) == [1, 4, 7]
    assert_eq!(nthroot_mod(1, 3, 27, true), Some(bs(&[1, 10, 19])));
    assert_eq!(nthroot_mod(1, 3, 9, true), Some(bs(&[1, 4, 7])));
    // a divisible by p: sympy nthroot_mod(8, 3, 16, True) == [2, 6, 10, 14]
    assert_eq!(nthroot_mod(8, 3, 16, true), Some(bs(&[2, 6, 10, 14])));
    // sympy: nthroot_mod(0, 3, 27, True) == [0, 3, 6, ..., 24]; nthroot_mod(0, 3, 8, True) == [0, 2, 4, 6]
    assert_eq!(
        nthroot_mod(0, 3, 27, true),
        Some((0..9).map(|k| b(3 * k)).collect())
    );
    assert_eq!(nthroot_mod(0, 3, 8, true), Some(bs(&[0, 2, 4, 6])));
    // sympy: nthroot_mod(4, 4, 25, True) == []; nthroot_mod(2, 3, 9, True) == []
    assert_eq!(nthroot_mod(4, 4, 25, true), None);
    assert_eq!(nthroot_mod(2, 3, 9, true), None);
}

#[test]
fn nthroot_mod_matches_brute_force_small_moduli() {
    // Every modulus 1..=48 (primes, prime powers, composites), n = 1..=8, every a.
    for m in 1..=48u64 {
        for n in 1..=8u32 {
            for a in 0..m {
                let expected = brute_nthroots(a, n, m);
                let got = nthroot_mod(a, n, m, true).unwrap_or_default();
                assert_eq!(got, expected, "nthroot_mod({a}, {n}, {m})");
                let smallest = nthroot_mod(a, n, m, false).unwrap_or_default();
                assert_eq!(
                    smallest,
                    expected.iter().take(1).cloned().collect::<Vec<_>>(),
                    "smallest root of x^{n} = {a} mod {m}"
                );
            }
        }
    }
}

#[test]
fn nthroot_mod_matches_brute_force_above_the_brute_force_threshold() {
    // Primes just above 2¹⁰ (where the primitive-root / Sylow-subgroup
    // algorithm takes over from enumeration) with varied p − 1 factorisations:
    // 1030 = 2·5·103, 1032 = 2³·3·43, 1050 = 2·3·5²·7, 1092 = 2²·3·7·13, 1116 = 2²·3²·31.
    for p in [1031u64, 1033, 1051, 1093, 1117] {
        for n in [3u32, 4, 5, 6, 7, 8, 9, 12, 13] {
            let mut targets: Vec<u64> = (1..=5).map(|g| pow_mod_u64(g, u64::from(n), p)).collect();
            targets.extend([2, 3, 5]);
            for a in targets {
                let expected = brute_nthroots(a, n, p);
                assert_eq!(
                    nthroot_mod(a, n, p, true).unwrap_or_default(),
                    expected,
                    "nthroot_mod({a}, {n}, {p})"
                );
            }
        }
    }
    // Hensel lift above such a prime: p ∤ n, so the roots modulo p² are in
    // bijection with the roots modulo p; each one is verified directly.
    let p = 1051u64;
    let p2 = BigInt::from(p * p);
    for n in [3u32, 5, 6, 7] {
        let a = pow_mod_u64(7, u64::from(n), p * p);
        let mod_p = brute_nthroots(a, n, p);
        let mod_p2 = nthroot_mod(a, n, p * p, true).unwrap();
        assert_eq!(mod_p2.len(), mod_p.len(), "root count mod {p}² for n = {n}");
        for x in &mod_p2 {
            assert_eq!(x.modpow(&BigInt::from(n), &p2), BigInt::from(a));
        }
        assert!(mod_p2.contains(&b(7)));
    }
}

#[test]
fn nthroot_mod_large_prime_uses_sylow_subgroup_logs() {
    // p = 998244353 = 119·2²³ + 1; sympy values:
    let p = 998_244_353i64;
    // nthroot_mod(3**8 % p, 8, p, True) == [3, 119342119, 259751154, 467927632,
    //                                       530316721, 738493199, 878902234, 998244350]
    let a = mod_pow(3, 8, p);
    assert_eq!(
        nthroot_mod(a, 8, p, true),
        Some(bs(&[
            3,
            119_342_119,
            259_751_154,
            467_927_632,
            530_316_721,
            738_493_199,
            878_902_234,
            998_244_350
        ]))
    );
    // nthroot_mod(5**7 % p, 7, p, True) == [5, 72766955, 423388036, 457716001,
    //                                       483121572, 657185804, 900554686]
    let a = mod_pow(5, 7, p);
    assert_eq!(
        nthroot_mod(a, 7, p, true),
        Some(bs(&[
            5,
            72_766_955,
            423_388_036,
            457_716_001,
            483_121_572,
            657_185_804,
            900_554_686
        ]))
    );
    // gcd(12, p − 1) = 4 → four roots: nthroot_mod(2**12 % p, 12, p, True)
    //   == [2, 173167436, 825076917, 998244351]
    let a = mod_pow(2, 12, p);
    assert_eq!(
        nthroot_mod(a, 12, p, true),
        Some(bs(&[2, 173_167_436, 825_076_917, 998_244_351]))
    );
    // gcd(5, p − 1) = 1 → unique root: nthroot_mod(7**5 % p, 5, p, True) == [7]
    assert_eq!(nthroot_mod(mod_pow(7, 5, p), 5, p, true), Some(bs(&[7])));
    // nthroot_mod(11**6 % p, 6, p) == 11
    assert_eq!(nthroot_mod(mod_pow(11, 6, p), 6, p, false), Some(bs(&[11])));
}

#[test]
fn nthroot_mod_beyond_u64_prime() {
    // p = 2¹²⁷ − 1, p − 1 = 2·3³·7²·19·43·73·127·337·5419·92737·649657·77158673929
    let p: BigInt = (BigInt::one() << 127) - BigInt::one();
    // sympy: nthroot_mod(8, 3, p, True) == [2, 78676610129673952743199618487727214611,
    //                                        91464573330795278988487685228156891114]
    let roots = nthroot_mod(8, 3, p.clone(), true).unwrap();
    assert_eq!(
        roots,
        vec![
            b(2),
            BigInt::parse_bytes(b"78676610129673952743199618487727214611", 10).unwrap(),
            BigInt::parse_bytes(b"91464573330795278988487685228156891114", 10).unwrap(),
        ]
    );
    // sympy: nthroot_mod(5**4 % p, 4, p, True) == [5, p − 5]  (gcd(4, p−1) = 2)
    let a = mod_pow(5, 4, p.clone());
    assert_eq!(nthroot_mod(a, 4, p.clone(), true), Some(vec![b(5), &p - 5]));
    // sympy: nthroot_mod(5**7 % p, 7, p) == 5
    let a = mod_pow(5, 7, p.clone());
    assert_eq!(nthroot_mod(a, 7, p.clone(), false), Some(vec![b(5)]));
    // nine 9-th roots (9 | 27 | p − 1), each verified by exponentiation
    let a = mod_pow(5, 9, p.clone());
    let roots = nthroot_mod(a.clone(), 9, p.clone(), true).unwrap();
    assert_eq!(roots.len(), 9);
    assert!(roots.contains(&b(5)));
    for x in &roots {
        assert_eq!(x.modpow(&b(9), &p), a);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// quadratic_residues, is_nthpow_residue
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quadratic_residues_values() {
    // sympy: quadratic_residues(7) == [0, 1, 2, 4]
    assert_eq!(quadratic_residues(7), bs(&[0, 1, 2, 4]));
    // sympy: quadratic_residues(8) == [0, 1, 4]
    assert_eq!(quadratic_residues(8), bs(&[0, 1, 4]));
    // sympy: quadratic_residues(15) == [0, 1, 4, 6, 9, 10]
    assert_eq!(quadratic_residues(15), bs(&[0, 1, 4, 6, 9, 10]));
    // sympy: quadratic_residues(13) == [0, 1, 3, 4, 9, 10, 12]
    assert_eq!(quadratic_residues(13), bs(&[0, 1, 3, 4, 9, 10, 12]));
    // sympy: quadratic_residues(16) == [0, 1, 4, 9]
    assert_eq!(quadratic_residues(16), bs(&[0, 1, 4, 9]));
    // sympy: quadratic_residues(1) == [0]; quadratic_residues(2) == [0, 1]
    assert_eq!(quadratic_residues(1), bs(&[0]));
    assert_eq!(quadratic_residues(2), bs(&[0, 1]));
    assert!(quadratic_residues(0).is_empty());
    // For an odd prime there are (p + 1)/2 residues including 0, and
    // every one is recognised by is_quad_residue.
    let res = quadratic_residues(101);
    assert_eq!(res.len(), 51);
    assert!(res.iter().all(|a| is_quad_residue(a.clone(), 101)));
}

#[test]
fn is_nthpow_residue_values() {
    // sympy: is_nthpow_residue(2, 3, 7) == False; is_nthpow_residue(3, 3, 7) == False
    assert!(!is_nthpow_residue(2, 3, 7));
    assert!(!is_nthpow_residue(3, 3, 7));
    // sympy: is_nthpow_residue(2, 4, 7) == True; is_nthpow_residue(16, 4, 17) == True
    assert!(is_nthpow_residue(2, 4, 7));
    assert!(is_nthpow_residue(16, 4, 17));
    // sympy: is_nthpow_residue(2, 2, 15) == False; (4, 2, 15) == True; (0, 2, 15) == True
    assert!(!is_nthpow_residue(2, 2, 15));
    assert!(is_nthpow_residue(4, 2, 15));
    assert!(is_nthpow_residue(0, 2, 15));
    // sympy: is_nthpow_residue(3, 0, 7) == False; (1, 0, 7) == True; (3, 1, 7) == True
    assert!(!is_nthpow_residue(3, 0, 7));
    assert!(is_nthpow_residue(1, 0, 7));
    assert!(is_nthpow_residue(3, 1, 7));
    // prime powers — sympy: (2, 3, 9) False, (8, 3, 16) True, (9, 4, 16) False,
    //                       (17, 4, 32) True, (4, 4, 25) False, (10, 3, 25) False
    assert!(!is_nthpow_residue(2, 3, 9));
    assert!(is_nthpow_residue(8, 3, 16));
    assert!(!is_nthpow_residue(9, 4, 16));
    assert!(is_nthpow_residue(17, 4, 32));
    assert!(!is_nthpow_residue(4, 4, 25));
    assert!(!is_nthpow_residue(10, 3, 25));
    // sympy: is_nthpow_residue(5, 3, 1000003) == False
    assert!(!is_nthpow_residue(5, 3, 1_000_003));
    // large composite: 3³ is a cube modulo 998244353·469762049 (sympy: True)
    let m = BigInt::from(998_244_353u64) * BigInt::from(469_762_049u64);
    assert!(is_nthpow_residue(27, 3, m));
    // everything is a residue modulo 1; nothing for m ≤ 0 or n < 0
    assert!(is_nthpow_residue(7, 2, 1));
    assert!(!is_nthpow_residue(7, 2, 0));
    assert!(!is_nthpow_residue(7, -1, 5));
}

#[test]
fn is_nthpow_residue_agrees_with_nthroot_mod() {
    for m in 1..=40u64 {
        for n in 0..=7u32 {
            for a in 0..m {
                let has_root = if n == 0 {
                    // x⁰ = 1: solvable iff a ≡ 1 (mod m)
                    a % m == 1 % m
                } else {
                    !brute_nthroots(a, n, m).is_empty()
                };
                assert_eq!(
                    is_nthpow_residue(a, n, m),
                    has_root,
                    "is_nthpow_residue({a}, {n}, {m})"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// polynomial_congruence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polynomial_congruence_sympy_values() {
    // sympy: polynomial_congruence(x**2 - 1, 8) == [1, 3, 5, 7]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, -1]), 8),
        bs(&[1, 3, 5, 7])
    );
    // sympy: polynomial_congruence(x**3 + 3*x**2 + 2*x, 12) == [0, 2, 3, 4, 6, 7, 8, 10, 11]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 3, 2, 0]), 12),
        bs(&[0, 2, 3, 4, 6, 7, 8, 10, 11])
    );
    // sympy: polynomial_congruence(6*x**5 + 10*x**4 + 5*x**3 + x**2 + x + 1, 7) == [2, 6]
    assert_eq!(
        polynomial_congruence(&bs(&[6, 10, 5, 1, 1, 1]), 7),
        bs(&[2, 6])
    );
    // sympy: polynomial_congruence(x**6 - 2*x**5 - 35, 6125) == [3257]   (5³·7², Hensel)
    assert_eq!(
        polynomial_congruence(&bs(&[1, -2, 0, 0, 0, 0, -35]), 6125),
        bs(&[3257])
    );
    // sympy: polynomial_congruence(x**3 + x + 1, 27) == [7]
    assert_eq!(polynomial_congruence(&bs(&[1, 0, 1, 1]), 27), bs(&[7]));
    // sympy: polynomial_congruence(x**5 + 3*x**3 + 2*x + 1, 125) == [29]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 3, 0, 2, 1]), 125),
        bs(&[29])
    );
    // sympy: polynomial_congruence(x**3 + 2*x + 3, 45) == [7, 12, 17, 34, 39, 44]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 2, 3]), 45),
        bs(&[7, 12, 17, 34, 39, 44])
    );
    // sympy: polynomial_congruence(x**4 - 1, 65) has 16 roots
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, 0, -1]), 65),
        bs(&[1, 8, 12, 14, 18, 21, 27, 31, 34, 38, 44, 47, 51, 53, 57, 64])
    );
    // sympy: polynomial_congruence(7*x**3 + 7, 21) == [2, 5, 8, 11, 14, 17, 20]
    assert_eq!(
        polynomial_congruence(&bs(&[7, 0, 0, 7]), 21),
        bs(&[2, 5, 8, 11, 14, 17, 20])
    );
    // sympy: polynomial_congruence(x**3, 8) == [0, 2, 4, 6]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, 0]), 8),
        bs(&[0, 2, 4, 6])
    );
    // no roots — sympy: x**2 + 1 mod 7 == []; x**3 - 2 mod 7 == []; x**4 + x**2 + 1 mod 9 == []
    assert!(polynomial_congruence(&bs(&[1, 0, 1]), 7).is_empty());
    assert!(polynomial_congruence(&bs(&[1, 0, 0, -2]), 7).is_empty());
    assert!(polynomial_congruence(&bs(&[1, 0, 1, 0, 1]), 9).is_empty());
}

#[test]
fn polynomial_congruence_linear_and_quadratic() {
    // sympy: polynomial_congruence(2*x + 4, 6) == [4, 1]  (unsorted in SymPy)
    assert_eq!(polynomial_congruence(&bs(&[2, 4]), 6), bs(&[1, 4]));
    // sympy: polynomial_congruence(2*x + 3, 6) == []
    assert!(polynomial_congruence(&bs(&[2, 3]), 6).is_empty());
    // sympy: polynomial_congruence(2*x**2 + 5*x + 3, 7) == [2, 6]
    assert_eq!(polynomial_congruence(&bs(&[2, 5, 3]), 7), bs(&[2, 6]));
    // sympy: polynomial_congruence(8*x**2 + 6*x + 4, 15) == []
    assert!(polynomial_congruence(&bs(&[8, 6, 4]), 15).is_empty());
    // sympy: polynomial_congruence(x**2 - 2, 49) == [10, 39]; x**2 - 1 mod 1000003 == [1, 1000002]
    assert_eq!(polynomial_congruence(&bs(&[1, 0, -2]), 49), bs(&[10, 39]));
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, -1]), 1_000_003),
        bs(&[1, 1_000_002])
    );
    // sympy: polynomial_congruence(x**2 + x + 1, 9) == []
    assert!(polynomial_congruence(&bs(&[1, 1, 1]), 9).is_empty());
    // degenerate inputs: zero polynomial, non-zero constant, m ≤ 1
    assert_eq!(polynomial_congruence(&bs(&[0, 0]), 4), bs(&[0, 1, 2, 3]));
    assert!(polynomial_congruence(&bs(&[5]), 4).is_empty());
    assert_eq!(polynomial_congruence(&bs(&[1, 0, 1]), 1), bs(&[0]));
    assert!(polynomial_congruence(&bs(&[1, 0, 1]), 0).is_empty());
}

#[test]
fn polynomial_congruence_large_prime_cantor_zassenhaus() {
    // sympy: polynomial_congruence(x**4 - 3*x + 5, 1000003) == [357940, 847957]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, -3, 5]), 1_000_003),
        bs(&[357_940, 847_957])
    );
    // sympy: polynomial_congruence(x**3 - x, 1000003) == [0, 1, 1000002]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, -1, 0]), 1_000_003),
        bs(&[0, 1, 1_000_002])
    );
    // sympy: (x-3)(x-5)(x-7)(x-11)(x-13) mod 1000003 == [3, 5, 7, 11, 13]
    // expanded: x⁵ − 39x⁴ + 574x³ − 3954x² + 12673x − 15015
    assert_eq!(
        polynomial_congruence(&bs(&[1, -39, 574, -3954, 12673, -15015]), 1_000_003),
        bs(&[3, 5, 7, 11, 13])
    );
    // sympy: x**3 + x + 1 mod 1000003 == []; x**5 + 2*x**3 - x + 7 mod 1000003 == []
    assert!(polynomial_congruence(&bs(&[1, 0, 1, 1]), 1_000_003).is_empty());
    assert!(polynomial_congruence(&bs(&[1, 0, 2, 0, -1, 7]), 1_000_003).is_empty());
    // Hensel above a large prime: sympy x**4 - 3*x + 5 mod 1000003**2 == [223795519339, 410804590349]
    let m = BigInt::from(1_000_003i64) * BigInt::from(1_000_003i64);
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, -3, 5]), m),
        bs(&[223_795_519_339, 410_804_590_349])
    );
    // CRT with a small and a large prime: sympy x**3 + 5*x**2 + 7*x + 2 mod 7·1000003 == [7000019]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 5, 7, 2]), 7_000_021),
        bs(&[7_000_019])
    );
    // monic binomial goes through nthroot_mod: sympy x**3 - 3 mod 998244353 == [84611520]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, -3]), 998_244_353),
        bs(&[84_611_520])
    );
}

#[test]
fn polynomial_congruence_matches_brute_force() {
    let polys: [&[i64]; 7] = [
        &[1, 3, 2, 0],
        &[6, 10, 5, 1, 1, 1],
        &[1, 0, 0, -3, 5],
        &[3, 5, 0, 1],
        &[2, 0, 0, 0, 1, 1],
        &[1, 1, 1, 1],
        &[4, 0, -6, 2, 9],
    ];
    for m in 1..=72u64 {
        for coeffs in polys {
            let expected = brute_poly_roots(coeffs, m);
            let got = polynomial_congruence(&bs(coeffs), m);
            assert_eq!(got, expected, "roots of {coeffs:?} mod {m}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multiplicity_values() {
    // sympy: multiplicity(2, 40) == 3; (3, 81) == 4; (5, 7) == 0; (2, 1) == 0; (6, 72) == 2
    assert_eq!(multiplicity(2, 40), 3);
    assert_eq!(multiplicity(3, 81), 4);
    assert_eq!(multiplicity(5, 7), 0);
    assert_eq!(multiplicity(2, 1), 0);
    assert_eq!(multiplicity(6, 72), 2);
    // sympy: multiplicity(10, 1000000) == 6; multiplicity(2, -40) == 3
    assert_eq!(multiplicity(10, 1_000_000), 6);
    assert_eq!(multiplicity(2, -40), 3);
    // undefined in SymPy (raises): here 0
    assert_eq!(multiplicity(2, 0), 0);
    assert_eq!(multiplicity(1, 5), 0);
    assert_eq!(multiplicity(0, 5), 0);
    // BigInt: 3^100 · 7
    let n = BigInt::from(3).pow(100) * 7;
    assert_eq!(multiplicity(3, n), 100);
}

#[test]
fn primenu_values() {
    // sympy: primenu(1) == 0, primenu(30) == 3, primenu(72) == 2, primenu(1024) == 1
    assert_eq!(primenu(1), 0);
    assert_eq!(primenu(30), 3);
    assert_eq!(primenu(72), 2);
    assert_eq!(primenu(1024), 1);
    assert_eq!(primenu(0), 0);
    assert_eq!(primenu(-30), 3);
    // ν(n) = number of entries of factorint(n)
    for n in 2..200i64 {
        assert_eq!(primenu(n) as usize, factorint(n).len());
    }
}

#[test]
fn primeomega_values() {
    // sympy: primeomega(1) == 0, primeomega(30) == 3, primeomega(72) == 5, primeomega(1024) == 10
    assert_eq!(primeomega(1), 0);
    assert_eq!(primeomega(30), 3);
    assert_eq!(primeomega(72), 5);
    assert_eq!(primeomega(1024), 10);
    // Ω(n) ≥ ν(n), with equality iff n is square-free
    for n in 2..200i64 {
        assert!(primeomega(n) >= primenu(n));
        assert_eq!(primeomega(n) == primenu(n), mobius(n) != 0);
    }
}

#[test]
fn primorial_values() {
    // sympy: primorial(1) == 2, primorial(5) == 2310, primorial(10) == 6469693230
    assert_eq!(primorial(1), b(2));
    assert_eq!(primorial(5), b(2310));
    assert_eq!(primorial(10), b(6_469_693_230));
    // sympy: primorial(20) == 557940830126698960967415390
    assert_eq!(primorial(20).to_string(), "557940830126698960967415390");
    assert_eq!(primorial(0), BigInt::one());
    // p_n# = p_{n−1}# · p_n
    for n in 1..30u64 {
        assert_eq!(primorial(n), primorial(n - 1) * prime(n).unwrap());
    }
}

#[test]
fn primorial_up_to_values() {
    // sympy: primorial(1, nth=False) == 1, primorial(5, nth=False) == 30, primorial(10, nth=False) == 210
    assert_eq!(primorial_up_to(1), b(1));
    assert_eq!(primorial_up_to(5), b(30));
    assert_eq!(primorial_up_to(10), b(210));
    // sympy: primorial(100, nth=False) == 2305567963945518424753102147331756070
    assert_eq!(
        primorial_up_to(100).to_string(),
        "2305567963945518424753102147331756070"
    );
    assert_eq!(primorial_up_to(-3), b(1));
    // primorial_up_to(p_n) == primorial(n)
    assert_eq!(primorial_up_to(29), primorial(10));
    assert_eq!(primorial_up_to(30), primorial(10));
}

#[test]
fn is_carmichael_values() {
    // sympy: the first Carmichael numbers
    for n in [561, 1105, 1729, 2465, 2821, 6601, 8911, 41041, 62745] {
        assert!(is_carmichael(n), "{n}");
    }
    // sympy: False for 1, 2, 3, 4, 5, 7, 9, 15, 21, 45, 91, 563, 0, -561
    for n in [1, 2, 3, 4, 5, 7, 9, 15, 21, 45, 91, 563, 0, -561] {
        assert!(!is_carmichael(n), "{n}");
    }
    // Korselt ⇔ a^n ≡ a (mod n) for all a, checked directly for 561 and a non-example
    let n = 561i64;
    assert!((0..n).all(|a| mod_pow(a, n, n) == b(a)));
    assert!(!(0..15).all(|a| mod_pow(a, 15, 15) == b(a)));
    // exhaustive below 3000: exactly 561, 1105, 1729, 2465, 2821
    let found: Vec<i64> = (1..3000).filter(|&n| is_carmichael(n)).collect();
    assert_eq!(found, vec![561, 1105, 1729, 2465, 2821]);
}

#[test]
fn is_amicable_values() {
    // sympy: is_amicable(220, 284) == True, (1184, 1210) == True, (2620, 2924) == True
    assert!(is_amicable(220, 284));
    assert!(is_amicable(284, 220));
    assert!(is_amicable(1184, 1210));
    assert!(is_amicable(2620, 2924));
    // sympy: is_amicable(220, 220) == False, (6, 6) == False, (220, 285) == False,
    //        (0, 0) == False, (1, 1) == False, (-220, -284) == False
    assert!(!is_amicable(220, 220));
    assert!(!is_amicable(6, 6));
    assert!(!is_amicable(220, 285));
    assert!(!is_amicable(0, 0));
    assert!(!is_amicable(1, 1));
    assert!(!is_amicable(-220, -284));
    // definition: σ(a) − a = b and σ(b) − b = a
    assert_eq!(divisor_sum(220) - 220, b(284));
    assert_eq!(divisor_sum(284) - 284, b(220));
}

#[test]
fn binomial_coefficients_list_values() {
    // sympy: binomial_coefficients_list(0) == [1]; (4) == [1, 4, 6, 4, 1];
    //        (6) == [1, 6, 15, 20, 15, 6, 1];
    //        (10) == [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    assert_eq!(binomial_coefficients_list(0), bs(&[1]));
    assert_eq!(binomial_coefficients_list(4), bs(&[1, 4, 6, 4, 1]));
    assert_eq!(binomial_coefficients_list(6), bs(&[1, 6, 15, 20, 15, 6, 1]));
    assert_eq!(
        binomial_coefficients_list(10),
        bs(&[1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1])
    );
    // row sums to 2ⁿ and agrees with binomial(n, k)
    let row = binomial_coefficients_list(40);
    assert_eq!(row.iter().sum::<BigInt>(), BigInt::one() << 40);
    for (k, c) in row.iter().enumerate() {
        assert_eq!(*c, binomial(40, k as i64));
    }
}

#[test]
fn binomial_coefficients_pairs() {
    // sympy: binomial_coefficients(4) == {(0, 4): 1, (4, 0): 1, (1, 3): 4, (3, 1): 4, (2, 2): 6}
    assert_eq!(
        binomial_coefficients(4),
        vec![
            ((0, 4), b(1)),
            ((1, 3), b(4)),
            ((2, 2), b(6)),
            ((3, 1), b(4)),
            ((4, 0), b(1)),
        ]
    );
    assert_eq!(binomial_coefficients(0), vec![((0, 0), b(1))]);
    // consistent with the list form
    let n = 12;
    let pairs = binomial_coefficients(n);
    let list = binomial_coefficients_list(n);
    assert_eq!(pairs.len(), list.len());
    for (((k1, k2), c), l) in pairs.iter().zip(&list) {
        assert_eq!(k1 + k2, n);
        assert_eq!(c, l);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Continued fraction reduction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn continued_fraction_reduce_values() {
    // sympy: continued_fraction_reduce([4, 2, 6, 7]) == 415/93
    assert_eq!(
        continued_fraction_reduce(&bs(&[4, 2, 6, 7])),
        Some(r(415, 93))
    );
    // sympy: continued_fraction_reduce([3, 7, 15, 1]) == 355/113
    assert_eq!(
        continued_fraction_reduce(&bs(&[3, 7, 15, 1])),
        Some(r(355, 113))
    );
    // sympy: continued_fraction_reduce([-3, 1, 2]) == -7/3; ([5]) == 5; ([0]) == 0
    assert_eq!(continued_fraction_reduce(&bs(&[-3, 1, 2])), Some(r(-7, 3)));
    assert_eq!(continued_fraction_reduce(&bs(&[5])), Some(r(5, 1)));
    assert_eq!(continued_fraction_reduce(&bs(&[0])), Some(r(0, 1)));
    // sympy: continued_fraction_reduce([1, 0]) == zoo → None here; empty → None
    assert_eq!(continued_fraction_reduce(&bs(&[1, 0])), None);
    assert_eq!(continued_fraction_reduce(&[]), None);
    // inverse of continued_fraction
    for (p, d) in [
        (415, 93),
        (-7, 3),
        (1, 1),
        (0, 1),
        (100, 7),
        (-355, 113),
        (22, 7),
    ] {
        let x = r(p, d);
        assert_eq!(continued_fraction_reduce(&continued_fraction(&x)), Some(x));
    }
}

#[test]
fn continued_fraction_reduce_periodic_values() {
    let triple = |p: i64, q: i64, d: i64| {
        Some(QuadraticSurd {
            p: b(p),
            q: b(q),
            d: b(d),
        })
    };
    // sympy: continued_fraction_reduce([1, [2]]) == sqrt(2)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1]), &bs(&[2])),
        triple(0, 1, 2)
    );
    // sympy: continued_fraction_reduce([2, [1, 1, 1, 4]]) == sqrt(7)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[2]), &bs(&[1, 1, 1, 4])),
        triple(0, 1, 7)
    );
    // sympy: continued_fraction_reduce([4, [1, 3, 1, 8]]) == sqrt(23)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[4]), &bs(&[1, 3, 1, 8])),
        triple(0, 1, 23)
    );
    // sympy: continued_fraction_reduce([[1]]) == (1 + sqrt(5))/2 == continued_fraction_reduce([1, [1]])
    assert_eq!(
        continued_fraction_reduce_periodic(&[], &bs(&[1])),
        triple(1, 2, 5)
    );
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1]), &bs(&[1])),
        triple(1, 2, 5)
    );
    // sympy: continued_fraction_reduce([3, [2]]) == sqrt(2) + 2
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[3]), &bs(&[2])),
        triple(2, 1, 2)
    );
    // sympy: continued_fraction_reduce([0, 1, [2]]) == sqrt(2)/2  = (0 + √2)/2
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[0, 1]), &bs(&[2])),
        triple(0, 2, 2)
    );
    // sympy: continued_fraction_reduce([1, 2, [3, 4]]) == (sqrt(3) + 4)/4
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1, 2]), &bs(&[3, 4])),
        triple(4, 4, 3)
    );
    // sympy: continued_fraction_reduce([1, 2, 3, [4, 5]]) == (80 - sqrt(30))/52  = (−80 + √30)/(−52)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1, 2, 3]), &bs(&[4, 5])),
        triple(-80, -52, 30)
    );
    // sympy: continued_fraction_reduce([1, 1, 1, [2]]) == 3 - sqrt(2)  = (−3 + √2)/(−1)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1, 1, 1]), &bs(&[2])),
        triple(-3, -1, 2)
    );
    // sympy: continued_fraction_reduce([[2, 3]]) == 1 + sqrt(15)/3  = (3 + √15)/3
    assert_eq!(
        continued_fraction_reduce_periodic(&[], &bs(&[2, 3])),
        triple(3, 3, 15)
    );
    // sympy: continued_fraction_reduce([-1, [2]]) == -2 + sqrt(2)
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[-1]), &bs(&[2])),
        triple(-2, 1, 2)
    );
    // invalid: empty period, non-positive period term
    assert_eq!(continued_fraction_reduce_periodic(&bs(&[1]), &[]), None);
    assert_eq!(
        continued_fraction_reduce_periodic(&bs(&[1]), &bs(&[0, 2])),
        None
    );
}

#[test]
fn continued_fraction_reduce_periodic_inverts_continued_fraction_periodic() {
    // √d for every non-square d < 200 round-trips to (0, 1, d)
    for d in 2..200i64 {
        if is_square(d) {
            continue;
        }
        let cf = continued_fraction_periodic(d).unwrap();
        assert_eq!(
            continued_fraction_reduce_periodic(&cf.pre_period, &cf.period),
            Some(QuadraticSurd {
                p: b(0),
                q: b(1),
                d: b(d)
            }),
            "sqrt({d})"
        );
    }
}

#[test]
fn continued_fraction_reduce_periodic_ex_builds_the_surd() {
    let ctx = Context::new();
    // √2, √7, √23 canonicalise to the plain square roots
    assert_eq!(
        continued_fraction_reduce_periodic_ex(&ctx, &bs(&[1]), &bs(&[2])).unwrap(),
        ctx.int(2).sqrt()
    );
    assert_eq!(
        continued_fraction_reduce_periodic_ex(&ctx, &bs(&[2]), &bs(&[1, 1, 1, 4])).unwrap(),
        ctx.int(7).sqrt()
    );
    // golden ratio numerically
    let phi = continued_fraction_reduce_periodic_ex(&ctx, &[], &bs(&[1])).unwrap();
    assert!((phi.eval_f64().unwrap() - 1.618_033_988_749_895).abs() < 1e-12);
    // (80 − √30)/52 ≈ 1.4331 (sympy: N((80 - sqrt(30))/52) = 1.43310…)
    let x = continued_fraction_reduce_periodic_ex(&ctx, &bs(&[1, 2, 3]), &bs(&[4, 5])).unwrap();
    let expected = (80.0 - 30f64.sqrt()) / 52.0;
    assert!((x.eval_f64().unwrap() - expected).abs() < 1e-12);
    assert!(continued_fraction_reduce_periodic_ex(&ctx, &bs(&[1]), &[]).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// discrete: convolutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn convolution_values() {
    // sympy: convolution([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18]
    assert_eq!(
        convolution(&q(&[1, 2, 3]), &q(&[4, 5, 6])),
        q(&[4, 13, 28, 27, 18])
    );
    // sympy: convolution([1/2, 1/3], [3, 4]) == [3/2, 3, 4/3]
    assert_eq!(
        convolution(&[r(1, 2), r(1, 3)], &q(&[3, 4])),
        vec![r(3, 2), r(3, 1), r(4, 3)]
    );
    // identity and empty
    assert_eq!(convolution(&q(&[1]), &q(&[7, 8, 9])), q(&[7, 8, 9]));
    assert!(convolution(&[], &q(&[1, 2])).is_empty());
    assert!(convolution(&q(&[1, 2]), &[]).is_empty());
    // commutative, and (1 + x)^4 via repeated convolution is Pascal's row
    let a = q(&[3, -1, 4, 1, -5]);
    let bb = q(&[9, 2, -6]);
    assert_eq!(convolution(&a, &bb), convolution(&bb, &a));
    let one_plus_x = q(&[1, 1]);
    let mut pow = one_plus_x.clone();
    for _ in 0..3 {
        pow = convolution(&pow, &one_plus_x);
    }
    assert_eq!(pow, q(&[1, 4, 6, 4, 1]));
}

#[test]
fn convolution_cyclic_values() {
    // sympy: convolution([1, 2, 3], [4, 5, 6], cycle=3) == [31, 31, 28]
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 3),
        q(&[31, 31, 28])
    );
    // sympy: cycle=4 == [22, 13, 28, 27]; cycle=2 == [50, 40]; cycle=1 == [90]
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 4),
        q(&[22, 13, 28, 27])
    );
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 2),
        q(&[50, 40])
    );
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 1),
        q(&[90])
    );
    // sympy: cycle=5 == linear; cycle=6 == linear + [0]
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 5),
        q(&[4, 13, 28, 27, 18])
    );
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3]), &q(&[4, 5, 6]), 6),
        q(&[4, 13, 28, 27, 18, 0])
    );
    // sympy: convolution([1, 2, 3, 4, 5], [6, 7], cycle=3) == [51, 77, 67]
    assert_eq!(
        convolution_cyclic(&q(&[1, 2, 3, 4, 5]), &q(&[6, 7]), 3),
        q(&[51, 77, 67])
    );
    assert!(convolution_cyclic(&q(&[1, 2]), &q(&[3]), 0).is_empty());
    assert!(convolution_cyclic(&[], &q(&[3]), 4).is_empty());
}

#[test]
fn convolution_subset_values() {
    // sympy: convolution_subset([1, 2, 3, 4], [5, 6, 7, 8]) == [5, 16, 22, 60]
    assert_eq!(
        convolution_subset(&q(&[1, 2, 3, 4]), &q(&[5, 6, 7, 8])),
        q(&[5, 16, 22, 60])
    );
    // sympy: convolution_subset([1, 2], [3, 4, 5]) == [3, 10, 5, 10]
    assert_eq!(
        convolution_subset(&q(&[1, 2]), &q(&[3, 4, 5])),
        q(&[3, 10, 5, 10])
    );
    // sympy: convolution_subset([1, 2, 3], [4, 5]) == [4, 13, 12, 15]
    assert_eq!(
        convolution_subset(&q(&[1, 2, 3]), &q(&[4, 5])),
        q(&[4, 13, 12, 15])
    );
    // sympy: convolution_subset(range(1, 9), [1]*8) == [1, 3, 4, 10, 6, 14, 16, 36]
    //   (= subset sums of 1..8, i.e. the Möbius transform)
    let a = q(&[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        convolution_subset(&a, &q(&[1, 1, 1, 1, 1, 1, 1, 1])),
        q(&[1, 3, 4, 10, 6, 14, 16, 36])
    );
    assert_eq!(
        convolution_subset(&a, &q(&[1, 1, 1, 1, 1, 1, 1, 1])),
        mobius_transform(&a)
    );
    // sympy: convolution([1, 2, 3], [4, 5, 6], subset=True) == [4, 13, 18, 27]
    assert_eq!(
        convolution_subset(&q(&[1, 2, 3]), &q(&[4, 5, 6])),
        q(&[4, 13, 18, 27])
    );
    assert!(convolution_subset(&[], &q(&[1])).is_empty());
}

#[test]
fn convolution_subset_matches_definition() {
    // c[k] = Σ_{s ⊆ k} a[s]·b[k ∖ s], checked directly against the definition.
    let a = q(&[2, -1, 3, 5, 0, 7, 1, -4]);
    let bb = q(&[1, 4, -2, 6, 3, 0, -5, 2]);
    let c = convolution_subset(&a, &bb);
    for (k, ck) in c.iter().enumerate() {
        let expected: Ratio<BigInt> = (0..8usize)
            .filter(|&s| s & k == s)
            .map(|s| &a[s] * &bb[k ^ s])
            .sum();
        assert_eq!(*ck, expected, "k = {k}");
    }
}

#[test]
fn convolution_ex_symbolic_coefficients() {
    let ctx = Context::new();
    symplex::syms!(ctx; a, bb, c);
    // (a + b·x)(c + x) = ac + (a + bc)x + b x²
    let prod = convolution_ex(&[a.clone(), bb.clone()], &[c.clone(), ctx.one()]);
    assert_eq!(prod.len(), 3);
    assert_eq!(prod[0], &a * &c);
    assert_eq!(prod[1], &a + &bb * &c);
    assert_eq!(prod[2], bb.clone());
    // agrees with the exact rational convolution on numbers
    let x = q(&[1, 2, 3]);
    let y = q(&[4, 5, 6]);
    let xe: Vec<Ex> = x.iter().map(|v| ctx.from_ratio(v.clone())).collect();
    let ye: Vec<Ex> = y.iter().map(|v| ctx.from_ratio(v.clone())).collect();
    let ze: Vec<Ex> = convolution(&x, &y)
        .into_iter()
        .map(|v| ctx.from_ratio(v))
        .collect();
    assert_eq!(convolution_ex(&xe, &ye), ze);
    assert!(convolution_ex(&[], &xe).is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// discrete: number-theoretic transform
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ntt_values() {
    // sympy: ntt([1, 2, 3, 4], prime=3*2**8+1) == [10, 643, 767, 122]
    assert_eq!(
        ntt(&bs(&[1, 2, 3, 4]), 769).unwrap(),
        bs(&[10, 643, 767, 122])
    );
    // sympy: ntt([1, 2, 3, 4], prime=998244353) == [10, 173167434, 998244351, 825076915]
    assert_eq!(
        ntt(&bs(&[1, 2, 3, 4]), 998_244_353).unwrap(),
        bs(&[10, 173_167_434, 998_244_351, 825_076_915])
    );
    // sympy: ntt([1, 2, 3, 4], prime=7*2**26+1) == [10, 39220180, 469762047, 430541865]
    assert_eq!(
        ntt(&bs(&[1, 2, 3, 4]), 469_762_049).unwrap(),
        bs(&[10, 39_220_180, 469_762_047, 430_541_865])
    );
    // sympy: ntt([1..8], prime=998244353) ==
    //   [36, 894301004, 346334868, 201631260, 998244349, 796613085, 651909477, 103943341]
    assert_eq!(
        ntt(&bs(&[1, 2, 3, 4, 5, 6, 7, 8]), 998_244_353).unwrap(),
        bs(&[
            36,
            894_301_004,
            346_334_868,
            201_631_260,
            998_244_349,
            796_613_085,
            651_909_477,
            103_943_341
        ])
    );
    // sympy: ntt([1, 2, 3, 4], prime=13) == [10, 8, 11, 1]; prime=17 == [10, 6, 15, 7]
    assert_eq!(ntt(&bs(&[1, 2, 3, 4]), 13).unwrap(), bs(&[10, 8, 11, 1]));
    assert_eq!(ntt(&bs(&[1, 2, 3, 4]), 17).unwrap(), bs(&[10, 6, 15, 7]));
    // sympy: ntt([1, 2, 3], prime=998244353) == [6, 825076915, 2, 173167434]  (padded to 4)
    assert_eq!(
        ntt(&bs(&[1, 2, 3]), 998_244_353).unwrap(),
        bs(&[6, 825_076_915, 2, 173_167_434])
    );
    // sympy: ntt([5], prime=998244353) == [5]; ntt([], ...) == []
    assert_eq!(ntt(&bs(&[5]), 998_244_353).unwrap(), bs(&[5]));
    assert!(ntt(&[], 998_244_353).unwrap().is_empty());
    // inputs are reduced mod p; negative inputs allowed
    assert_eq!(
        ntt(&bs(&[1 + 769, 2 - 769, 3, 4]), 769).unwrap(),
        bs(&[10, 643, 767, 122])
    );
}

#[test]
fn ntt_rejects_bad_moduli() {
    // sympy raises ValueError("Expected prime modulus of the form (m*2**k + 1)") for these
    assert!(matches!(
        ntt(&bs(&[1, 2, 3, 4]), 7),
        Err(SymplexError::InvalidArgument {
            operation: "ntt",
            ..
        })
    ));
    assert!(matches!(
        ntt(&bs(&[1, 2, 3, 4, 5, 6, 7, 8]), 13),
        Err(SymplexError::InvalidArgument { .. })
    ));
    // composite modulus
    assert!(matches!(
        ntt(&bs(&[1, 2, 3, 4]), 15),
        Err(SymplexError::InvalidArgument { .. })
    ));
    assert!(matches!(
        intt(&bs(&[1, 2, 3, 4]), 7),
        Err(SymplexError::InvalidArgument {
            operation: "intt",
            ..
        })
    ));
}

#[test]
fn intt_values_and_round_trip() {
    // sympy: intt([1, 2, 3, 4], prime=998244353) == [499122179, 455830317, 499122176, 542414035]
    assert_eq!(
        intt(&bs(&[1, 2, 3, 4]), 998_244_353).unwrap(),
        bs(&[499_122_179, 455_830_317, 499_122_176, 542_414_035])
    );
    // sympy: intt(ntt([1, 2, 3, 4], 769), 769) == [1, 2, 3, 4]
    let a = bs(&[1, 2, 3, 4]);
    assert_eq!(intt(&ntt(&a, 769).unwrap(), 769).unwrap(), a);
    // round trips at both standard NTT primes and a length-16 input
    let a: Vec<BigInt> = (0..16).map(|i| b(i * i * 7 - 40)).collect();
    for p in [998_244_353i64, 469_762_049] {
        let reduced: Vec<BigInt> = a.iter().map(|x| x.mod_floor(&b(p))).collect();
        assert_eq!(intt(&ntt(&a, p).unwrap(), p).unwrap(), reduced);
        assert_eq!(ntt(&intt(&a, p).unwrap(), p).unwrap(), reduced);
    }
}

#[test]
fn ntt_of_convolution_is_pointwise_product() {
    let p = 998_244_353i64;
    let pb = b(p);
    let a = bs(&[1, 2, 3, 4, 5, 0, 0, 0]);
    let c = bs(&[6, 7, 8, 9, 0, 0, 0, 0]);
    // linear convolution has 5 + 4 − 1 = 8 entries: no wrap-around at length 8
    let lin: Vec<BigInt> = convolution(&q(&[1, 2, 3, 4, 5]), &q(&[6, 7, 8, 9]))
        .into_iter()
        .map(|v| v.to_integer().mod_floor(&pb))
        .collect();
    let ta = ntt(&a, p).unwrap();
    let tc = ntt(&c, p).unwrap();
    let pointwise: Vec<BigInt> = ta.iter().zip(&tc).map(|(x, y)| (x * y) % &pb).collect();
    assert_eq!(ntt(&lin, p).unwrap(), pointwise);
    assert_eq!(intt(&pointwise, p).unwrap(), lin);
    // the same at 7·2²⁶ + 1
    let p2 = 469_762_049i64;
    let ta = ntt(&a, p2).unwrap();
    let tc = ntt(&c, p2).unwrap();
    let pointwise: Vec<BigInt> = ta.iter().zip(&tc).map(|(x, y)| (x * y) % b(p2)).collect();
    assert_eq!(intt(&pointwise, p2).unwrap(), lin);
}

#[test]
fn convolution_ntt_values() {
    // sympy: convolution_ntt([1, 2, 3], [4, 5, 6], prime=998244353) == [4, 13, 28, 27, 18]
    assert_eq!(
        convolution_ntt(&bs(&[1, 2, 3]), &bs(&[4, 5, 6]), 998_244_353).unwrap(),
        bs(&[4, 13, 28, 27, 18])
    );
    // agrees with the exact convolution reduced mod p for a longer pair
    let p = 469_762_049i64;
    let x: Vec<i64> = (0..13).map(|i| 1000 * i - 6000).collect();
    let y: Vec<i64> = (0..9).map(|i| i * i + 1).collect();
    let exact: Vec<BigInt> = convolution(&q(&x), &q(&y))
        .into_iter()
        .map(|v| v.to_integer().mod_floor(&b(p)))
        .collect();
    assert_eq!(convolution_ntt(&bs(&x), &bs(&y), p).unwrap(), exact);
    assert!(convolution_ntt(&[], &bs(&[1]), p).unwrap().is_empty());
    // length 8 needs 8 | p − 1: 13 − 1 = 12 fails
    assert!(convolution_ntt(&bs(&[1, 2, 3, 4, 5]), &bs(&[1, 1]), 13).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// discrete: Walsh–Hadamard
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fwht_values() {
    // sympy: fwht([1, 2, 3, 4]) == [10, -2, -4, 0]
    assert_eq!(fwht(&q(&[1, 2, 3, 4])), q(&[10, -2, -4, 0]));
    // sympy: fwht([1, 2, 3]) == [6, 2, 0, -4]  (zero-padded)
    assert_eq!(fwht(&q(&[1, 2, 3])), q(&[6, 2, 0, -4]));
    // sympy: fwht([1..8]) == [36, -4, -8, 0, -16, 0, 0, 0]
    assert_eq!(
        fwht(&q(&[1, 2, 3, 4, 5, 6, 7, 8])),
        q(&[36, -4, -8, 0, -16, 0, 0, 0])
    );
    // sympy: fwht([]) == []; fwht([7]) == [7]
    assert!(fwht(&[]).is_empty());
    assert_eq!(fwht(&q(&[7])), q(&[7]));
    // sympy: fwht([1/2, 1/3, 1/4, 1/5]) == [77/60, 13/60, 23/60, 7/60]
    assert_eq!(
        fwht(&[r(1, 2), r(1, 3), r(1, 4), r(1, 5)]),
        vec![r(77, 60), r(13, 60), r(23, 60), r(7, 60)]
    );
    // definition: A[k] = Σ_j (−1)^{popcount(j & k)} a[j]
    let a = q(&[3, -1, 4, 1, -5, 9, 2, -6]);
    let t = fwht(&a);
    for (k, tk) in t.iter().enumerate() {
        let mut expected = Ratio::zero();
        for (j, aj) in a.iter().enumerate() {
            if (j & k).count_ones() % 2 == 0 {
                expected += aj;
            } else {
                expected -= aj;
            }
        }
        assert_eq!(*tk, expected);
    }
}

#[test]
fn ifwht_values_and_round_trip() {
    // sympy: ifwht([10, -2, -4, 0]) == [1, 2, 3, 4]
    assert_eq!(ifwht(&q(&[10, -2, -4, 0])), q(&[1, 2, 3, 4]));
    // sympy: ifwht([1, 2, 3, 4]) == [5/2, -1/2, -1, 0]
    assert_eq!(
        ifwht(&q(&[1, 2, 3, 4])),
        vec![r(5, 2), r(-1, 2), r(-1, 1), r(0, 1)]
    );
    // sympy: ifwht([1..8]) == [9/2, -1/2, -1, 0, -2, 0, 0, 0]
    assert_eq!(
        ifwht(&q(&[1, 2, 3, 4, 5, 6, 7, 8])),
        vec![
            r(9, 2),
            r(-1, 2),
            r(-1, 1),
            r(0, 1),
            r(-2, 1),
            r(0, 1),
            r(0, 1),
            r(0, 1)
        ]
    );
    // sympy: ifwht([1, 2, 3]) == [3/2, 1/2, 0, -1]
    assert_eq!(
        ifwht(&q(&[1, 2, 3])),
        vec![r(3, 2), r(1, 2), r(0, 1), r(-1, 1)]
    );
    assert!(ifwht(&[]).is_empty());
    // round trips
    let a: Vec<Ratio<BigInt>> = (0..16).map(|i| r(3 * i - 20, i + 1)).collect();
    assert_eq!(ifwht(&fwht(&a)), a);
    assert_eq!(fwht(&ifwht(&a)), a);
}

#[test]
fn fwht_diagonalises_dyadic_convolution() {
    // sympy: convolution([1, 2, 3], [4, 5, 6], dyadic=True) == [32, 13, 18, 27]
    // dyadic (XOR) convolution c[i ^ j] += a[i] b[j] equals ifwht(fwht(a) ⊙ fwht(b)).
    let a = q(&[1, 2, 3]);
    let bb = q(&[4, 5, 6]);
    let fa = fwht(&a);
    let fb = fwht(&bb);
    let prod: Vec<Ratio<BigInt>> = fa.iter().zip(&fb).map(|(x, y)| x * y).collect();
    assert_eq!(ifwht(&prod), q(&[32, 13, 18, 27]));
}

// ═══════════════════════════════════════════════════════════════════════════
// discrete: Möbius transform
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mobius_transform_values() {
    // sympy: mobius_transform([1, 2, 3, 4]) == [1, 3, 4, 10]
    assert_eq!(mobius_transform(&q(&[1, 2, 3, 4])), q(&[1, 3, 4, 10]));
    // sympy: mobius_transform([1, 2, 3]) == [1, 3, 4, 6]  (padded)
    assert_eq!(mobius_transform(&q(&[1, 2, 3])), q(&[1, 3, 4, 6]));
    // sympy: mobius_transform([1..8]) == [1, 3, 4, 10, 6, 14, 16, 36]
    assert_eq!(
        mobius_transform(&q(&[1, 2, 3, 4, 5, 6, 7, 8])),
        q(&[1, 3, 4, 10, 6, 14, 16, 36])
    );
    // sympy: mobius_transform([1, 2, 3, 4, 5]) == [1, 3, 4, 10, 6, 8, 9, 15]
    assert_eq!(
        mobius_transform(&q(&[1, 2, 3, 4, 5])),
        q(&[1, 3, 4, 10, 6, 8, 9, 15])
    );
    // sympy: mobius_transform([]) == []; mobius_transform([9]) == [9]
    assert!(mobius_transform(&[]).is_empty());
    assert_eq!(mobius_transform(&q(&[9])), q(&[9]));
    // definition: A[k] = Σ_{s ⊆ k} a[s]
    let a = q(&[3, -1, 4, 1, -5, 9, 2, -6]);
    let t = mobius_transform(&a);
    for (k, tk) in t.iter().enumerate() {
        let expected: Ratio<BigInt> = (0..8usize)
            .filter(|&s| s & k == s)
            .map(|s| a[s].clone())
            .sum();
        assert_eq!(*tk, expected);
    }
}

#[test]
fn inverse_mobius_transform_values_and_round_trip() {
    // sympy: inverse_mobius_transform([1, 3, 4, 10]) == [1, 2, 3, 4]
    assert_eq!(
        inverse_mobius_transform(&q(&[1, 3, 4, 10])),
        q(&[1, 2, 3, 4])
    );
    // sympy: inverse_mobius_transform([1, 2, 3, 4]) == [1, 1, 2, 0]
    assert_eq!(
        inverse_mobius_transform(&q(&[1, 2, 3, 4])),
        q(&[1, 1, 2, 0])
    );
    // sympy: inverse_mobius_transform([1..8]) == [1, 1, 2, 0, 4, 0, 0, 0]
    assert_eq!(
        inverse_mobius_transform(&q(&[1, 2, 3, 4, 5, 6, 7, 8])),
        q(&[1, 1, 2, 0, 4, 0, 0, 0])
    );
    let a: Vec<Ratio<BigInt>> = (0..16).map(|i| r(5 * i - 33, i + 2)).collect();
    assert_eq!(inverse_mobius_transform(&mobius_transform(&a)), a);
    assert_eq!(mobius_transform(&inverse_mobius_transform(&a)), a);
}

#[test]
fn mobius_transform_superset_values_and_round_trip() {
    // sympy: mobius_transform([1, 2, 3, 4], subset=False) == [10, 6, 7, 4]
    assert_eq!(
        mobius_transform_superset(&q(&[1, 2, 3, 4])),
        q(&[10, 6, 7, 4])
    );
    // definition: A[k] = Σ_{s ⊇ k} a[s]
    let a = q(&[3, -1, 4, 1, -5, 9, 2, -6]);
    let t = mobius_transform_superset(&a);
    for (k, tk) in t.iter().enumerate() {
        let expected: Ratio<BigInt> = (0..8usize)
            .filter(|&s| s & k == k)
            .map(|s| a[s].clone())
            .sum();
        assert_eq!(*tk, expected);
    }
    assert_eq!(inverse_mobius_transform_superset(&t), a);
    assert_eq!(
        mobius_transform_superset(&inverse_mobius_transform_superset(&a)),
        a
    );
}
