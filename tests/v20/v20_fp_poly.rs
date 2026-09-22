//! 0.21 track: the two hand-rolled polynomial rings replaced by the crate's
//! generic types.
//!
//! `ℤ[x]` inside Berlekamp–Zassenhaus is now `GenPoly<BigInt>` (`BigInt`
//! implements `Ring`/`EuclideanDomain`/`IntegralCoeff`/`CoeffDisplay`), and
//! `𝔽ₚ[x]` — hand-rolled once in `factor_zassenhaus` and once in `ntheory`
//! — is the single value-level ring `PolyIn<Fp64>` in `poly::modpoly`.
//! Both swaps are *output-preserving*: the factorisation strings below were
//! captured from the commit before the change and must not move; the
//! `PolyIn` primitives are checked against SymPy (`.venv/bin/python`,
//! SymPy 1.14.0 — the call is cited beside each value).

use std::time::Instant;

use symplex::factor_zassenhaus::modpoly::{DivRem, ExtendedGcd, Fp64, PolyIn, RingOps};
use symplex::factor_zassenhaus::traits::{EuclideanDomain, IntegralCoeff, Ring};
use symplex::factor_zassenhaus::{GenPoly, Poly, factor_mod_p, factor_zassenhaus_with_content};
use symplex::ntheory::polynomial_congruence;
use symplex::num_bigint::{BigInt, BigUint};
use symplex::num_rational::Ratio;
use symplex::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

type Z = GenPoly<BigInt>;

fn z(n: i64) -> BigInt {
    BigInt::from(n)
}

fn zp(c: &[i64]) -> Z {
    GenPoly::from_coeffs(c.iter().map(|&x| z(x)).collect())
}

fn poly(coeffs: &[i64]) -> Poly {
    Poly::from_coeffs(coeffs.iter().map(|&c| Ratio::from_integer(z(c))).collect())
}

fn fp(p: u64, c: &[u64]) -> PolyIn<Fp64> {
    PolyIn::over_prime(p, c.to_vec())
}

fn bs(v: &[i64]) -> Vec<BigInt> {
    v.iter().map(|&x| z(x)).collect()
}

fn xn_minus_1(n: usize) -> Vec<i64> {
    let mut c = vec![0i64; n + 1];
    c[0] = -1;
    c[n] = 1;
    c
}

fn mul(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn ex_from(ctx: &Context, coeffs: &[i64], x: &Ex) -> Ex {
    let mut e = ctx.int(0);
    for (i, &c) in coeffs.iter().enumerate() {
        if c != 0 {
            e += x.powi(i as i64) * c;
        }
    }
    e.expand()
}

/// `content; [f₁]^m₁ · [f₂]^m₂ · …` with the factors' own `Display`.
fn describe(f: &Poly) -> String {
    let (content, fs) = factor_zassenhaus_with_content(f);
    let parts: Vec<String> = fs.iter().map(|(g, m)| format!("[{g}]^{m}")).collect();
    format!("{content}; {}", parts.join(" · "))
}

// ═══════════════════════════════════════════════════════════════════════════
// BigInt as a coefficient ring; GenPoly<BigInt>
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn bigint_implements_the_coefficient_traits() {
    assert_eq!(<BigInt as Ring>::zero(), z(0));
    assert_eq!(<BigInt as Ring>::one(), z(1));
    assert_eq!(Ring::add(&z(7), &z(-9)), z(-2));
    assert_eq!(Ring::sub(&z(7), &z(-9)), z(16));
    assert_eq!(Ring::mul(&z(7), &z(-9)), z(-63));
    assert_eq!(Ring::neg(&z(7)), z(-7));
    // Truncated division: quotient toward zero, remainder with the sign of
    // the dividend — so an exact quotient is exact regardless of signs.
    assert_eq!(EuclideanDomain::div_rem(&z(-7), &z(2)), (z(-3), z(-1)));
    assert_eq!(EuclideanDomain::div_rem(&z(-6), &z(-3)), (z(2), z(0)));
    assert_eq!(<BigInt as EuclideanDomain>::gcd(&z(-12), &z(18)), z(6));
    assert_eq!(<BigInt as IntegralCoeff>::from_i64(-4), z(-4));
    assert_eq!(IntegralCoeff::to_integer(&z(5)), Some(z(5)));
}

#[test]
fn genpoly_bigint_arithmetic() {
    let a = zp(&[1, 2]); // 2θ + 1
    let b = zp(&[-3, 0, 1]); // θ² − 3
    assert_eq!(a.add(&b), zp(&[-2, 2, 1]));
    assert_eq!(a.sub(&b), zp(&[4, 2, -1]));
    assert_eq!(a.mul(&b), zp(&[-3, -6, 1, 2]));
    assert_eq!(a.neg(), zp(&[-1, -2]));
    assert_eq!(a.scale(&z(3)), zp(&[3, 6]));
    assert_eq!(b.derivative(), zp(&[0, 2]));
    assert_eq!(b.eval(&z(2)), z(1));
    assert_eq!(a.pow(2), zp(&[1, 4, 4]));
    assert_eq!(Z::from_int(5), zp(&[5]));
    assert_eq!(format!("{}", zp(&[-2, 0, 3])), "3*θ^2 - 2");
    assert_eq!(format!("{}", zp(&[1, -1])), "-θ + 1");
}

#[test]
fn genpoly_bigint_content_and_primitive_part() {
    assert_eq!(zp(&[6, -9, 12]).content(), z(3));
    assert_eq!(zp(&[6, -9, 12]).primitive_part(), zp(&[2, -3, 4]));
    // The content is non-negative; the sign pattern is kept.
    assert_eq!(zp(&[-4, -6]).content(), z(2));
    assert_eq!(zp(&[-4, -6]).primitive_part(), zp(&[-2, -3]));
    assert_eq!(zp(&[5, 7]).content(), z(1));
    assert_eq!(zp(&[5, 7]).primitive_part(), zp(&[5, 7]));
    assert_eq!(Z::zero().content(), z(0));
    assert_eq!(Z::zero().primitive_part(), Z::zero());
    assert_eq!(zp(&[0, 0, 10]).content(), z(10));
}

#[test]
fn genpoly_bigint_exact_division() {
    // (θ + 1)(3θ − 2) = 3θ² + θ − 2
    let f = zp(&[1, 1]).mul(&zp(&[-2, 3]));
    assert_eq!(f, zp(&[-2, 1, 3]));
    assert_eq!(f.div_exact(&zp(&[1, 1])), Some(zp(&[-2, 3])));
    assert_eq!(f.div_exact(&zp(&[-2, 3])), Some(zp(&[1, 1])));
    assert_eq!(f.div_exact(&zp(&[1, 2])), None);
    assert_eq!(f.div_exact(&zp(&[5, 1])), None);
    // Divisible over ℚ but not over ℤ.
    assert_eq!(zp(&[2, 2]).div_exact(&zp(&[4])), None);
    assert_eq!(zp(&[2, 2]).div_exact(&zp(&[2])), Some(zp(&[1, 1])));
    // Negative leading coefficients.
    let g = zp(&[1, -1]).mul(&zp(&[-3, 2, -1]));
    assert_eq!(g.div_exact(&zp(&[1, -1])), Some(zp(&[-3, 2, -1])));
    assert_eq!(g.div_exact(&zp(&[-3, 2, -1])), Some(zp(&[1, -1])));
    assert_eq!(Z::zero().div_exact(&zp(&[1, 1])), Some(Z::zero()));
    assert_eq!(zp(&[1, 1]).div_exact(&Z::zero()), None);
}

#[test]
fn genpoly_bigint_mul_mod() {
    let m = z(7);
    // (3θ + 5)(4θ + 6) = 12θ² + 38θ + 30 ≡ 5θ² + 3θ + 2 (mod 7)
    assert_eq!(zp(&[5, 3]).mul_mod(&zp(&[6, 4]), &m), zp(&[2, 3, 5]));
    // A leading coefficient that vanishes modulo m is stripped.
    assert_eq!(zp(&[1, 7]).mul_mod(&zp(&[1, 1]), &m), zp(&[1, 1]));
    // Coefficients land in [0, m).
    let big = z(1_000_003);
    let r = zp(&[-5, 999_999]).mul_mod(&zp(&[123_456, -7]), &big);
    assert!(r.coeffs().iter().all(|c| *c >= z(0) && *c < big));
}

// ═══════════════════════════════════════════════════════════════════════════
// PolyIn<Fp64> against SymPy
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fp64_ring_ops() {
    let f = Fp64::new(7);
    assert_eq!(f.modulus(), 7);
    assert_eq!(f.add(&5, &4), 2);
    assert_eq!(f.sub(&2, &5), 4);
    assert_eq!(f.mul(&5, &4), 6);
    assert_eq!(f.neg(&3), 4);
    assert_eq!(f.neg(&0), 0);
    assert_eq!(f.inv(&3), Some(5));
    assert_eq!(f.inv(&0), None);
    assert_eq!(f.embed_u64(100), 2);
    assert!(f.is_zero(&0) && f.is_one(&1));
    // p ≥ 2³²: the u128 product path.
    let big = Fp64::new(1_000_000_000_039);
    let a = 999_999_999_999u64;
    let inv = big.inv(&a).expect("unit");
    assert_eq!(big.mul(&inv, &a), 1);
}

#[test]
fn polyin_div_rem_against_sympy() {
    // sympy: gf_div([5,1,4,1,3], [1,0,2], 7, ZZ) → ([5,1,1], [6,1])
    let a = fp(7, &[3, 1, 4, 1, 5]);
    let b = fp(7, &[2, 0, 1]);
    let DivRem {
        quotient: q,
        remainder: r,
    } = a.div_rem(&b);
    assert_eq!(q.coeffs(), &[1, 1, 5]);
    assert_eq!(r.coeffs(), &[1, 6]);
    assert_eq!(q.mul(&b).add(&r), a);
    assert_eq!(a.rem(&b), r);
    assert_eq!(a.div(&b), q);
    // sympy: gf_gcd([5,1,4,1,3], [1,0,2], 7, ZZ) → [1]
    assert_eq!(a.gcd(&b).coeffs(), &[1]);
}

#[test]
fn polyin_gcd_and_extended_gcd_against_sympy() {
    // sympy: gf_gcdex([1,3,2,1], [1,1,5], 13, ZZ) → s = x + 7, t = 12x² + 4x + 4, h = 1
    let a = fp(13, &[1, 2, 3, 1]);
    let b = fp(13, &[5, 1, 1]);
    let ExtendedGcd { u, v, gcd: g } = a.extended_gcd(&b);
    assert_eq!(u.coeffs(), &[7, 1]);
    assert_eq!(v.coeffs(), &[4, 4, 12]);
    assert_eq!(g.coeffs(), &[1]);
    assert_eq!(u.mul(&a).add(&v.mul(&b)), g);

    // sympy: A = Poly((x+1)(x+2)(x²+1), x, modulus=101), B = Poly((x+1)(x+3)(x+50), x, modulus=101)
    //        A.gcd(B) → x + 1;  A.gcdex(B) → (89x + 75, 12x² + 20x + 32, x + 1)
    let a = fp(101, &[2, 3, 3, 3, 1]);
    let b = fp(101, &[49, 1, 54, 1]);
    assert_eq!(a.gcd(&b).coeffs(), &[1, 1]);
    assert_eq!(b.gcd(&a).coeffs(), &[1, 1]);
    let ExtendedGcd { u, v, gcd: g } = a.extended_gcd(&b);
    assert_eq!(u.coeffs(), &[75, 89]);
    assert_eq!(v.coeffs(), &[32, 20, 12]);
    assert_eq!(g.coeffs(), &[1, 1]);
    assert_eq!(u.mul(&a).add(&v.mul(&b)), g);

    // gcd with zero is the monic version of the other argument.
    assert_eq!(fp(7, &[2, 4]).gcd(&fp(7, &[])).coeffs(), &[4, 1]);
    assert!(fp(7, &[]).gcd(&fp(7, &[])).is_zero());
    assert!(fp(7, &[2, 4]).monic().is_monic());
}

#[test]
fn polyin_powmod_against_sympy() {
    // sympy: gf_pow_mod([1,0], 101, [1,0,0,0,3,1], 101, ZZ) → 93x³ + 62x² + 4x
    let f = fp(101, &[1, 3, 0, 0, 0, 1]);
    let x = PolyIn::x(Fp64::new(101));
    assert_eq!(x.powmod(101, &f).coeffs(), &[0, 4, 62, 93]);
    // sympy: gf_pow_mod([1,2], 50, [1,0,0,0,3,1], 101, ZZ) → 31x⁴ + 76x³ + 64x² + 51x + 25
    assert_eq!(
        fp(101, &[2, 1]).powmod(50, &f).coeffs(),
        &[25, 51, 64, 76, 31]
    );
    assert_eq!(
        fp(101, &[2, 1]).powmod_big(&BigUint::from(50u32), &f),
        fp(101, &[2, 1]).powmod(50, &f)
    );
    assert_eq!(x.powmod(0, &f).coeffs(), &[1]);
    // sympy: gf_pow_mod([1,0], 1000003, [1,0,0,-3,5], 1000003, ZZ)
    //        → 190351x³ + 352342x² + 144019x + 821714
    let p = 1_000_003u64;
    let f = fp(p, &[5, p - 3, 0, 0, 1]);
    assert_eq!(
        PolyIn::x(Fp64::new(p)).powmod(p, &f).coeffs(),
        &[821_714, 144_019, 352_342, 190_351]
    );
    // mul_mod is mul then rem.
    assert_eq!(
        fp(p, &[2, 1]).mul_mod(&fp(p, &[2, 1]), &f),
        fp(p, &[2, 1]).mul(&fp(p, &[2, 1])).rem(&f)
    );
}

#[test]
fn polyin_derivative_and_squarefree_against_sympy() {
    // sympy: gf_diff([3,2,0,1,0], 5, ZZ) → 2x³ + x² + 1
    assert_eq!(fp(5, &[0, 1, 0, 2, 3]).derivative().coeffs(), &[1, 0, 1, 2]);
    // sympy: gf_sqf_p([1,2,1], 5, ZZ) → False; gf_sqf_p([1,0,1], 5, ZZ) → True
    assert!(!fp(5, &[1, 2, 1]).is_squarefree());
    assert!(fp(5, &[1, 0, 1]).is_squarefree());
    // x⁵ + 1 = (x + 1)⁵ over GF(5): vanishing derivative, p-th root x + 1.
    assert!(!fp(5, &[1, 0, 0, 0, 0, 1]).is_squarefree());
    assert_eq!(fp(5, &[1, 0, 0, 0, 0, 1]).pth_root().coeffs(), &[1, 1]);
}

#[test]
fn polyin_roots_against_sympy() {
    let p = 1_000_003u64;
    // sympy: polynomial_congruence(x**4 - 3*x + 5, 1000003) == [357940, 847957]
    assert_eq!(fp(p, &[5, p - 3, 0, 0, 1]).roots(), vec![357_940, 847_957]);
    // sympy: polynomial_congruence(x**2 + 1, 1000003) == []
    assert!(fp(p, &[1, 0, 1]).roots().is_empty());
    // sympy: polynomial_congruence(x**6 - 1, 1000003)
    //        == [1, 499501, 499502, 500501, 500502, 1000002]
    assert_eq!(
        fp(p, &[p - 1, 0, 0, 0, 0, 0, 1]).roots(),
        vec![1, 499_501, 499_502, 500_501, 500_502, 1_000_002]
    );
    // sympy: polynomial_congruence((x-3)(x-5)(x-7)(x-11)(x-13), 1000003) == [3, 5, 7, 11, 13]
    let f = fp(p, &[p - 15_015, 12_673, p - 3_954, 574, p - 39, 1]);
    assert_eq!(f.roots(), vec![3, 5, 7, 11, 13]);
    // 7000021 = 7 · 1000003 is *not* prime (sympy's polynomial_congruence
    // works over composite moduli, which is how a wrong "oracle" slipped in);
    // 7000009 is: sympy.isprime(7000009) == True,
    // polynomial_congruence(2*x**3 + 5*x + 7, 7000009) == [3330300, 3669710, 7000008]
    let p = 7_000_009u64;
    assert_eq!(
        fp(p, &[7, 5, 0, 2]).roots(),
        vec![3_330_300, 3_669_710, 7_000_008]
    );
    // sympy: polynomial_congruence(x**3 - 3, 998244353) == [84611520]
    let p = 998_244_353u64;
    assert_eq!(fp(p, &[p - 3, 0, 0, 1]).roots(), vec![84_611_520]);
    assert!(fp(p, &[3]).roots().is_empty());
}

#[test]
fn factor_mod_p_against_sympy_gf_factor() {
    // sympy: gf_factor([1,0,0,0,1,0,0,0,1], 7, ZZ) → (1, [(x+2,1),(x+3,1),(x+4,1),(x+5,1),(x²+2,1),(x²+4,1)])
    let (lc, fs) = factor_mod_p(&poly(&[1, 0, 0, 0, 1, 0, 0, 0, 1]), 7).expect("prime");
    assert_eq!(lc, 1);
    assert_eq!(
        fs,
        vec![
            (vec![2, 1], 1),
            (vec![3, 1], 1),
            (vec![4, 1], 1),
            (vec![5, 1], 1),
            (vec![2, 0, 1], 1),
            (vec![4, 0, 1], 1),
        ]
    );
    // sympy: gf_factor([1,0,0,0,0,0,12], 13, ZZ) → six linear factors x+1, x+3, x+4, x+9, x+10, x+12
    let (_, fs) = factor_mod_p(&poly(&xn_minus_1(6)), 13).expect("prime");
    let got: Vec<Vec<u64>> = fs.iter().map(|(g, _)| g.clone()).collect();
    assert_eq!(
        got,
        vec![
            vec![1, 1],
            vec![3, 1],
            vec![4, 1],
            vec![9, 1],
            vec![10, 1],
            vec![12, 1]
        ]
    );
    // (x+1)² (x+2) = x³ + 4x² + 5x + 2 mod 7, with multiplicities.
    let (_, fs) = factor_mod_p(&poly(&[2, 5, 4, 1]), 7).expect("prime");
    assert_eq!(fs, vec![(vec![1, 1], 2), (vec![2, 1], 1)]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Factorisations: byte-identical to the pre-change output
// ═══════════════════════════════════════════════════════════════════════════

/// `(name, ascending coefficients, factor_zassenhaus_with_content Display,
/// Ex::factor Display)` — the two strings were captured at the commit
/// before `ℤ[x]`/`𝔽ₚ[x]` were swapped for `GenPoly<BigInt>`/`PolyIn<Fp64>`.
fn factoring_cases() -> Vec<(&'static str, Vec<i64>, &'static str, &'static str)> {
    let phi25 = vec![
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ];
    let phi33 = vec![
        1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1,
    ];
    let mut deg40a = vec![0i64; 21];
    deg40a[0] = 2;
    deg40a[1] = 2;
    deg40a[20] = 1;
    let mut deg40b = vec![0i64; 21];
    deg40b[0] = 3;
    deg40b[2] = 3;
    deg40b[20] = 1;
    let cube = |f: &[i64]| mul(&mul(f, f), f);
    vec![
        (
            "x^2-1",
            vec![-1, 0, 1],
            "1; [θ - 1]^1 · [θ + 1]^1",
            "(x - 1)*(x + 1)",
        ),
        (
            "x^4+4",
            vec![4, 0, 0, 0, 1],
            "1; [θ^2 - 2*θ + 2]^1 · [θ^2 + 2*θ + 2]^1",
            "(x^2 - 2*x + 2)*(x^2 + 2*x + 2)",
        ),
        (
            "6x^4-7x^3-8x^2+7x+2",
            vec![2, 7, -8, -7, 6],
            "1; [θ - 1]^1 · [θ + 1]^1 · [6*θ^2 - 7*θ - 2]^1",
            "(x - 1)*(6*x^2 - 7*x - 2)*(x + 1)",
        ),
        (
            "x^8+x^4+1",
            vec![1, 0, 0, 0, 1, 0, 0, 0, 1],
            "1; [θ^2 - θ + 1]^1 · [θ^2 + θ + 1]^1 · [θ^4 - θ^2 + 1]^1",
            "(x^2 + x + 1)*(x^2 - x + 1)*(x^4 - x^2 + 1)",
        ),
        (
            "x^12-1",
            xn_minus_1(12),
            "1; [θ - 1]^1 · [θ + 1]^1 · [θ^2 - θ + 1]^1 · [θ^2 + 1]^1 · [θ^2 + θ + 1]^1 · [θ^4 - θ^2 + 1]^1",
            "(x - 1)*(x + 1)*(x^2 + x + 1)*(x^2 + 1)*(x^2 - x + 1)*(x^4 - x^2 + 1)",
        ),
        (
            "swinnerton-dyer-8",
            vec![576, 0, -960, 0, 352, 0, -40, 0, 1],
            "1; [θ^8 - 40*θ^6 + 352*θ^4 - 960*θ^2 + 576]^1",
            "x^8 - 40*x^6 + 352*x^4 - 960*x^2 + 576",
        ),
        (
            "-3x^2+3",
            vec![3, 0, -3],
            "-3; [θ - 1]^1 · [θ + 1]^1",
            "-3*(x - 1)*(x + 1)",
        ),
        (
            "4x^3-4x",
            vec![0, -4, 0, 4],
            "4; [θ - 1]^1 · [θ]^1 · [θ + 1]^1",
            "4*x*(x - 1)*(x + 1)",
        ),
        (
            "12(x-1)^3(x^2+1)",
            mul(&mul(&[12], &cube(&[-1, 1])), &[1, 0, 1]),
            "12; [θ - 1]^3 · [θ^2 + 1]^1",
            "12*(x - 1)^3*(x^2 + 1)",
        ),
        (
            "(x^2+1)^2(x+1)^3(2x-3)",
            mul(&mul(&mul(&[1, 0, 1], &[1, 0, 1]), &cube(&[1, 1])), &[-3, 2]),
            "1; [2*θ - 3]^1 · [θ + 1]^3 · [θ^2 + 1]^2",
            "(x + 1)^3*(x^2 + 1)^2*(2*x - 3)",
        ),
        (
            "x^60-1",
            xn_minus_1(60),
            "1; [θ - 1]^1 · [θ + 1]^1 · [θ^2 - θ + 1]^1 · [θ^2 + 1]^1 · [θ^2 + θ + 1]^1 · [θ^4 - θ^3 + θ^2 - θ + 1]^1 · [θ^4 - θ^2 + 1]^1 · [θ^4 + θ^3 + θ^2 + θ + 1]^1 · [θ^8 - θ^7 + θ^5 - θ^4 + θ^3 - θ + 1]^1 · [θ^8 - θ^6 + θ^4 - θ^2 + 1]^1 · [θ^8 + θ^7 - θ^5 - θ^4 - θ^3 + θ + 1]^1 · [θ^16 + θ^14 - θ^10 - θ^8 - θ^6 + θ^2 + 1]^1",
            "(x - 1)*(x + 1)*(x^2 + x + 1)*(x^4 + x^3 + x^2 + x + 1)*(x^8 + x^7 - x^5 - x^4 - x^3 + x + 1)*(x^16 + x^14 - x^10 - x^8 - x^6 + x^2 + 1)*(x^2 + 1)*(x^4 - x^3 + x^2 - x + 1)*(x^2 - x + 1)*(x^8 - x^7 + x^5 - x^4 + x^3 - x + 1)*(x^8 - x^6 + x^4 - x^2 + 1)*(x^4 - x^2 + 1)",
        ),
        (
            "deg40: (x^20+2x+2)(x^20+3x^2+3)",
            mul(&deg40a, &deg40b),
            "1; [θ^20 + 2*θ + 2]^1 · [θ^20 + 3*θ^2 + 3]^1",
            "(x^20 + 2*x + 2)*(x^20 + 3*x^2 + 3)",
        ),
        (
            "deg40: Φ25·Φ33",
            mul(&phi25, &phi33),
            "1; [θ^20 - θ^19 + θ^17 - θ^16 + θ^14 - θ^13 + θ^11 - θ^10 + θ^9 - θ^7 + θ^6 - θ^4 + θ^3 - θ + 1]^1 · [θ^20 + θ^15 + θ^10 + θ^5 + 1]^1",
            "(x^20 + x^15 + x^10 + x^5 + 1)*(x^20 - x^19 + x^17 - x^16 + x^14 - x^13 + x^11 - x^10 + x^9 - x^7 + x^6 - x^4 + x^3 - x + 1)",
        ),
        (
            "(x^2+1)(2x^2+x+3)(x^3+x+1)",
            mul(&mul(&[1, 0, 1], &[3, 1, 2]), &[1, 1, 0, 1]),
            "1; [θ^2 + 1]^1 · [2*θ^2 + θ + 3]^1 · [θ^3 + θ + 1]^1",
            "(x^3 + x + 1)*(x^2 + 1)*(2*x^2 + x + 3)",
        ),
        (
            "(x^2+1)(x^2-2)(x^4+1)(5x^5+x^3+7)",
            mul(
                &mul(&mul(&[1, 0, 1], &[-2, 0, 1]), &[1, 0, 0, 0, 1]),
                &[7, 0, 0, 1, 0, 5],
            ),
            "1; [θ^2 - 2]^1 · [θ^2 + 1]^1 · [θ^4 + 1]^1 · [5*θ^5 + θ^3 + 7]^1",
            "(x^2 - 2)*(x^2 + 1)*(x^4 + 1)*(5*x^5 + x^3 + 7)",
        ),
    ]
}

#[test]
fn factorisations_are_byte_identical_to_pre_change_output() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    for (name, c, want_poly, want_ex) in factoring_cases() {
        assert_eq!(
            describe(&poly(&c)),
            want_poly,
            "Poly factorisation of {name}"
        );
        let e = ex_from(&ctx, &c, &x);
        assert_eq!(format!("{}", e.factor(&x)), want_ex, "Ex::factor of {name}");
    }
}

#[test]
fn factorisation_of_x105_minus_1_has_the_famous_minus_two() {
    let (content, fs) = factor_zassenhaus_with_content(&poly(&xn_minus_1(105)));
    assert_eq!(format!("{content}"), "1");
    assert_eq!(fs.len(), 8); // τ(105)
    let phi105 = fs
        .iter()
        .find(|(g, _)| g.degree() == Some(48))
        .expect("Φ₁₀₅");
    assert!(
        phi105
            .0
            .coeffs()
            .iter()
            .any(|c| *c == Ratio::from_integer(z(-2)))
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// ntheory: the one caller of the old `fpp_*` ring
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn polynomial_congruence_general_case_unchanged() {
    // Small moduli (brute force) — the documented values.
    // sympy: polynomial_congruence(x**2 - 1, 8) == [1, 3, 5, 7]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, -1]), 8),
        bs(&[1, 3, 5, 7])
    );
    // sympy: polynomial_congruence(x**6 - 2*x**5 - 35, 6125) == [3257]
    assert_eq!(
        polynomial_congruence(&bs(&[1, -2, 0, 0, 0, 0, -35]), 6125),
        bs(&[3257])
    );
    // sympy: polynomial_congruence(6*x**5 + 10*x**4 + 5*x**3 + x**2 + x + 1, 7) == [2, 6]
    assert_eq!(
        polynomial_congruence(&bs(&[6, 10, 5, 1, 1, 1]), 7),
        bs(&[2, 6])
    );
    // Large prime (> 2¹⁶): gcd(f, xᵖ − x) + Cantor–Zassenhaus through PolyIn::roots.
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
    // sympy: polynomial_congruence(x**3 + x + 1, 1000003) == []
    assert!(polynomial_congruence(&bs(&[1, 0, 1, 1]), 1_000_003).is_empty());
    // sympy: polynomial_congruence(x**3 - 3, 998244353) == [84611520]
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, -3]), 998_244_353),
        bs(&[84_611_520])
    );
    // Hensel above a large prime:
    // sympy: polynomial_congruence(x**4 - 3*x + 5, 1000003**2) == [223795519339, 410804590349]
    let m = BigInt::from(1_000_003i64) * BigInt::from(1_000_003i64);
    assert_eq!(
        polynomial_congruence(&bs(&[1, 0, 0, -3, 5]), m),
        bs(&[223_795_519_339, 410_804_590_349])
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Timing harness (run on demand: `cargo test --test v20 v20_fp_poly -- --ignored --nocapture`)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "timing report, not an assertion"]
fn factoring_timing_report() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let all = factoring_cases();
    for (_, c, _, _) in &all {
        let _ = factor_zassenhaus_with_content(&poly(c));
    }
    let reps = 10;
    let mut total_min = std::time::Duration::ZERO;
    for (name, c, _, _) in &all {
        let p = poly(c);
        let mut best = std::time::Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = factor_zassenhaus_with_content(&p);
            best = best.min(t.elapsed());
        }
        let e = ex_from(&ctx, c, &x);
        let mut best_ex = std::time::Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = e.factor(&x);
            best_ex = best_ex.min(t.elapsed());
        }
        total_min += best;
        println!("TIME {name:40} poly {best:?}  ex {best_ex:?}");
    }
    let x105 = poly(&xn_minus_1(105));
    let mut best = std::time::Duration::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = factor_zassenhaus_with_content(&x105);
        best = best.min(t.elapsed());
    }
    println!("TIME {:40} poly {best:?}", "x^105-1");
    println!("TIME ALL(poly, sum of mins) {total_min:?}");
}
