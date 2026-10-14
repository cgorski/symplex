//! Tests of the 0.50 refactor (w10b): one coefficient-domain algebra.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::factor_zassenhaus::GenPoly;
use symplex::factor_zassenhaus::domain::{CoeffDomain, GaussQ, GaussQQ, QQ, ZZ};
use symplex::factor_zassenhaus::modpoly::{DensePoly, Fp64, PolyIn, RingOps};
use symplex::prelude::*;

fn q(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

/// Before: `Fp64::inv` was Fermat's `a^(p−2)`, which for a modulus that
/// is not prime returns a non-inverse: `Fp64::new(15).inv(&3)` was
/// `Some(3)` (`3·3 = 9 ≢ 1 mod 15`).  Now the extended Euclidean
/// algorithm: `None` for a non-unit, the inverse otherwise (and 46–54% of
/// the time of Brown's modular gcd went into Fermat's 124 products).
/// Oracle: `a·a⁻¹ ≡ 1 (mod p)` checked by `u128` arithmetic.
#[test]
fn fp64_inverse_is_euclid_and_refuses_non_units() {
    let f = Fp64::new(15);
    assert_eq!(f.inv(&3), None);
    assert_eq!(f.inv(&2), Some(8));
    let p = (1u64 << 62) - 57;
    let f = Fp64::new(p);
    for a in [1u64, 2, 3, 12_345_678_901, p - 1, p / 3] {
        let i = f.inv(&a).unwrap();
        assert_eq!((u128::from(a) * u128::from(i)) % u128::from(p), 1, "{a}");
    }
}

/// New in 0.50: `DensePoly<D>` (the old `PolyIn`) has the resultant and
/// Yun's square-free decomposition for every domain, written once.
/// Oracles (exact identities): `res(f, f′) = −disc(f)` for a monic cubic
/// (`disc(x³ − 2x + 1) = −4·(−2)³ − 27 = 5`); `(x − 1)²(x + 2)³` by
/// construction; over `ℚ(i)`, `res((x − i)(x + 1), x + i) = (2i)(i − 1) =
/// −2 − 2i` (the product of `x + i` at the roots of the monic first
/// argument).  The same values as `GenPoly`.
#[test]
fn dense_poly_resultant_and_squarefree_over_q_and_q_i() {
    let f = DensePoly::from_coeffs(QQ, vec![q(1), q(-2), q(0), q(1)]);
    assert_eq!(f.resultant(&f.derivative()), Some(q(-5)));
    let g = GenPoly::from_coeffs(f.coeffs().to_vec());
    assert_eq!(GenPoly::resultant(&g, &g.derivative()), q(-5));

    let lin = |a: i64| DensePoly::from_coeffs(QQ, vec![q(a), q(1)]);
    let h = lin(-1).mul(&lin(-1)).mul(&lin(2)).mul(&lin(2)).mul(&lin(2));
    let sqf = h.squarefree_factors().unwrap();
    assert_eq!(sqf, vec![(lin(-1), 2), (lin(2), 3)]);

    let z = |re: i64, im: i64| GaussQ {
        re: q(re),
        im: q(im),
    };
    let a = DensePoly::from_coeffs(GaussQQ, vec![z(0, -1), z(1, 0)])
        .mul(&DensePoly::from_coeffs(GaussQQ, vec![z(1, 0), z(1, 0)]));
    let b = DensePoly::from_coeffs(GaussQQ, vec![z(0, 1), z(1, 0)]);
    assert_eq!(a.resultant(&b), Some(z(-2, -2)));
}

/// Before: `PolyIn::gcd` over a ring value whose leading coefficients are
/// not units looped forever (`div_rem` returned `(0, a)`, and Euclid
/// swapped the pair for ever); `ℤ` is such a domain now (`ZZ`).  Now the
/// last remainder is returned (a multiple of the gcd) and `try_gcd` says
/// `None`.  Oracle: `gcd(2x + 2, 3x + 1) = 1` over `ℚ`; `3x + 1` is a
/// multiple of it.
#[test]
fn dense_poly_gcd_over_a_ring_terminates() {
    let z = |c: &[i64]| DensePoly::from_coeffs(ZZ, c.iter().map(|&n| BigInt::from(n)).collect());
    let (a, b) = (z(&[2, 2]), z(&[1, 3]));
    assert_eq!(a.try_gcd(&b), None);
    assert_eq!(a.gcd(&b), b);
    // A monic divisor is fine over ℤ: (x + 1)(x + 2) and x + 1.
    let (c, d) = (z(&[2, 3, 1]), z(&[1, 1]));
    assert_eq!(c.try_gcd(&d), Some(z(&[1, 1])));
    let _: PolyIn<ZZ> = c;
}

/// Before: the common root index `L` of the radicals of a matrix
/// (`linalg::Tower::scan`) was an lcm computed in `u32` by a private gcd
/// copy: for `√3` met before `2^(1/2147483649)` it wrapped
/// (`2 · 2147483649 = 2³² + 2 → 2`): `eigenvects` of
/// `[[√3 + 2^(1/2147483649), 1], [0, 1]]` panicked in debug builds
/// ("attempt to multiply with overflow", linalg.rs:677) and built a tower
/// of index 2 for a radical of index 2³¹ + 1 in release.  Now the lcm is
/// taken in `u64` with `base::numeric::gcd_u64` and the tower is refused
/// (`L > 12`).  Oracle: the matrix is upper triangular, so its
/// eigenvalues are its diagonal entries, and `A·v = λ·v` (checked
/// numerically at 1e-12).
#[test]
fn huge_radical_index_does_not_wrap_the_tower_index() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let a = Matrix::new(vec![
        vec![p("sqrt(3) + 2^(1/2147483649)"), p("1")],
        vec![p("0"), p("1")],
    ])
    .unwrap();
    let ev = a.eigenvects().unwrap();
    let mut vals: Vec<String> = ev.iter().map(|(l, _, _)| l.to_string()).collect();
    vals.sort();
    assert_eq!(vals, ["1", "2^(1/2147483649) + sqrt(3)"]);
    for (lam, mult, vecs) in &ev {
        assert_eq!((*mult, vecs.len()), (1, 1), "{lam}");
        let r = &(&a * &vecs[0]) - &vecs[0].scale(lam);
        for i in 0..2 {
            let z = r.get(i, 0).eval_complex64().unwrap();
            assert!(z.norm() < 1e-12, "{lam}: {}", r.get(i, 0));
        }
    }
}
