//! Tests of the 0.50 refactor (w10): polynomial domains and the modular gcd.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use symplex::prelude::*;

use std::cmp::Ordering;
use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use symplex::multipoly::{GrLex, GrevLex, Lex, Monomial, MonomialOrd, MultiPoly};

/// All exponent vectors in `nv` variables with entries below `top`.
fn vectors(nv: usize, top: u32) -> Vec<Vec<u32>> {
    let mut out = vec![Vec::new()];
    for _ in 0..nv {
        out = out
            .into_iter()
            .flat_map(|v| {
                (0..top).map(move |e| {
                    let mut w = v.clone();
                    w.push(e);
                    w
                })
            })
            .collect();
    }
    out
}

fn assert_order_agrees<O: MonomialOrd>() {
    let vs = vectors(3, 4);
    for a in &vs {
        for b in &vs {
            assert_eq!(
                Monomial::<O>::new(a).cmp(&Monomial::new(b)),
                O::cmp_exponents(a, b),
                "{a:?} vs {b:?}"
            );
        }
    }
}

/// `Monomial`, the term key of `MultiPoly` since 0.50 (total degree cached
/// for the graded orders, exponents inline; before, a `MonoKey` summed both
/// `Vec<u32>`s on every comparison), orders exactly as the order's
/// `cmp_exponents`, which did not change: all 4,096 pairs of exponent
/// vectors in three variables with entries 0–3.  Everything printed from a
/// `MultiPoly` follows this order (byte identity).
#[test]
fn monomial_order_is_cmp_exponents_grevlex() {
    assert_order_agrees::<GrevLex>();
}

/// As [`monomial_order_is_cmp_exponents_grevlex`], graded lex.
#[test]
fn monomial_order_is_cmp_exponents_grlex() {
    assert_order_agrees::<GrLex>();
}

/// As [`monomial_order_is_cmp_exponents_grevlex`], pure lex.
#[test]
fn monomial_order_is_cmp_exponents_lex() {
    assert_order_agrees::<Lex>();
}

/// The total degree in the graded orders is a sum in `u64`.  Before 0.50
/// `GrevLex`/`GrLex` summed the exponents in `u32`: for `x^(2³²−1)·y`
/// against `x` the sum wrapped to 0 in release builds (the larger monomial
/// ranked *below* `x`) and panicked in debug builds.  Oracle: the definition
/// of the degree, `2³² > 1`.
#[test]
fn graded_degree_does_not_wrap() {
    let big = [u32::MAX, 1];
    assert_eq!(GrevLex::cmp_exponents(&big, &[1, 0]), Ordering::Greater);
    assert_eq!(GrLex::cmp_exponents(&big, &[1, 0]), Ordering::Greater);
    let m = Monomial::<GrevLex>::new(&big);
    assert_eq!(m.total_degree(), u64::from(u32::MAX) + 1);
    assert!(m > Monomial::new(&[1, 0]));
}

/// The schoolbook product over `BTreeMap<Vec<u32>, Ratio>`: the oracle.
fn schoolbook(a: &MultiPoly, b: &MultiPoly) -> MultiPoly {
    let mut acc: BTreeMap<Vec<u32>, Ratio<BigInt>> = BTreeMap::new();
    for (ea, ca) in a.terms() {
        for (eb, cb) in b.terms() {
            let e: Vec<u32> = ea.iter().zip(eb).map(|(x, y)| x + y).collect();
            *acc.entry(e).or_insert_with(Ratio::zero) += ca * cb;
        }
    }
    MultiPoly::from_terms(a.num_vars(), acc.into_iter().collect()).unwrap()
}

/// `(1 + x + y + z + w)⁴` (70 terms), so that `p·(p + 1)` has 4,900 term
/// pairs: above the size from which products accumulate in a hash map.
fn fateman_base() -> MultiPoly {
    let mut s: MultiPoly = MultiPoly::from_int(4, 1);
    for i in 0..4 {
        s = s.add(&MultiPoly::var(4, i));
    }
    s.pow(4)
}

/// Large products accumulate in a hash map, sorted once at the end, and in
/// `i128` when every coefficient sum provably fits (new in 0.50; Fateman's
/// `p·(p + 1)`, `p = (1 + x + y + z + w)¹⁰`, went from 210 ms to 44 ms).
/// Small coefficients (the `i128` route): equal to the schoolbook product.
#[test]
fn hashed_product_small_coefficients() {
    let p = fateman_base();
    let q = p.add(&MultiPoly::from_int(4, 1));
    assert_eq!(p.mul(&q), schoolbook(&p, &q));
}

/// As [`hashed_product_small_coefficients`], with coefficients near 2⁵⁰
/// and both signs (the `i128` route with sums near 2¹⁰⁷, cancellations)
/// and near 2⁶² (the bound fails: the `BigInt` route).
#[test]
fn hashed_product_large_coefficients() {
    let p = fateman_base();
    for c in [(1i64 << 50) + 3, (1i64 << 62) - 1] {
        let alt = MultiPoly::from_terms(
            4,
            p.terms()
                .enumerate()
                .map(|(k, (e, x))| {
                    let s = if k % 3 == 0 { -c } else { c - k as i64 };
                    (e.to_vec(), x * Ratio::from_integer(BigInt::from(s)))
                })
                .collect(),
        )
        .unwrap();
        let q = alt.add(&p);
        assert_eq!(alt.mul(&q), schoolbook(&alt, &q), "scale {c}");
    }
}

/// As [`hashed_product_small_coefficients`], with rational coefficients
/// (the product runs over a common denominator).
#[test]
fn hashed_product_rational_coefficients() {
    let p = fateman_base().scale(&Ratio::new(BigInt::from(2), BigInt::from(3)));
    let q = p.add(&MultiPoly::constant(
        4,
        Ratio::new(BigInt::from(1), BigInt::from(7)),
    ));
    assert_eq!(p.mul(&q), schoolbook(&p, &q));
}

/// `(A·x + y² + z³ + w⁴ + 1)²` with a 4,000-bit `A`, times `x − y` and times
/// `x + y + z + w`: the heuristic gcd would work on integers of about
/// fifteen million bits, so `MultiPoly::gcd` takes Brown's modular route
/// (new in 0.50; the survey's 21-digit version of this family, with
/// exponents 5 and 4, took 20.8 s and takes 0.65 s).  Oracle: the common
/// factor is known, the cofactors `x − y` and `x + y + z + w` are coprime
/// and the gcd is normalised to positive leading coefficient.
#[test]
fn dense_gcd_by_the_modular_route() {
    let [x, y, z, w]: [MultiPoly; 4] = std::array::from_fn(|i| MultiPoly::var(4, i));
    let a = Ratio::from_integer((BigInt::one() << 4000u32) + 12345);
    let base = x
        .scale(&a)
        .add(&y.pow(2))
        .add(&z.pow(3))
        .add(&w.pow(4))
        .add(&MultiPoly::from_int(4, 1));
    let h = base.pow(2);
    let f = h.mul(&x.sub(&y));
    let g = h.mul(&x.add(&y).add(&z).add(&w));
    assert_eq!(MultiPoly::gcd(&f, &g), h);
}
