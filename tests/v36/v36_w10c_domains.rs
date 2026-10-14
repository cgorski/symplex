//! Tests of the 0.50 refactor (w10c): the coefficient-domain algebra completed.
//! Every test says what was wrong before and cites its oracle.

#![allow(unused_imports)]
use num_bigint::BigInt;
use num_rational::Ratio;
use symplex::factor_zassenhaus::GenPoly;
use symplex::factor_zassenhaus::domain::{GaussQ, QQ, ZZ};
use symplex::factor_zassenhaus::modpoly::DensePoly;
use symplex::prelude::*;

fn q(n: i64) -> Ratio<BigInt> {
    Ratio::from_integer(BigInt::from(n))
}

fn zp(c: &[i64]) -> DensePoly<ZZ> {
    DensePoly::from_coeffs(ZZ, c.iter().map(|&k| BigInt::from(k)).collect())
}

fn qp(c: &[i64]) -> DensePoly<QQ> {
    DensePoly::from_coeffs(QQ, c.iter().map(|&k| q(k)).collect())
}

/// Before: `DensePoly::try_gcd`, documented as the monic gcd, returned the
/// last remainder when its leading coefficient was not a unit: over `ℤ`,
/// `try_gcd(2x + 2, 0)` was `Some(2x + 2)`.  Now `None` (the gcd over `ℤ`
/// is not monic; `gcd` still returns the remainder reached, as documented),
/// and `try_monic` is the fallible `monic` the algorithms over a tower of
/// radicals need.  Oracle: `2` is not a unit of `ℤ`; `−1` is.
#[test]
fn try_gcd_refuses_a_remainder_that_cannot_be_made_monic() {
    assert_eq!(zp(&[2, 2]).try_gcd(&zp(&[])), None);
    assert_eq!(zp(&[2, 2]).gcd(&zp(&[])), zp(&[2, 2]));
    assert_eq!(zp(&[1, -1]).try_gcd(&zp(&[])), Some(zp(&[-1, 1])));
    assert_eq!(zp(&[3, -1]).try_monic(), Some(zp(&[-3, 1])));
    assert_eq!(zp(&[1, 2]).try_monic(), None);
    assert_eq!(zp(&[]).try_monic(), None);
    // Yun over ℤ: a leading coefficient 2 cannot be made monic.
    assert_eq!(zp(&[1, 2, 1, 2]).squarefree_factors(), None);
}

/// New in 0.50: `DensePoly::try_extended_gcd`, `None` where `extended_gcd`
/// returns the row it reached (a non-unit leading coefficient).  Oracle:
/// the Bézout identity `u·a + v·b = gcd`, `gcd` monic, checked exactly;
/// over `ℤ`, `gcd(2x, 3)` meets the non-unit `2`.
#[test]
fn try_extended_gcd_satisfies_bezout_or_refuses() {
    // a = (x − 1)(x + 2), b = (x − 1)(x − 3): gcd x − 1.
    let (a, b) = (qp(&[-2, 1, 1]), qp(&[3, -4, 1]));
    let e = a.try_extended_gcd(&b).unwrap();
    assert_eq!(e.gcd, qp(&[-1, 1]));
    assert_eq!(e.u.mul(&a).add(&e.v.mul(&b)), e.gcd);
    assert!(zp(&[0, 2]).try_extended_gcd(&zp(&[3])).is_none());
    let row = zp(&[0, 2]).extended_gcd(&zp(&[3]));
    assert_eq!(row.u.mul(&zp(&[0, 2])).add(&row.v.mul(&zp(&[3]))), row.gcd);
}

/// New in 0.50: the polynomials over the constants `K = ℚ(√p…, i)` of
/// the matrix code (eigenvalues and eigenvectors over a square-root tower)
/// are `DensePoly` over the tower's field; up to 0.49 the division, monic
/// gcd, Yun's decomposition and the inverse modulo `g` were a second copy
/// on coefficient vectors (`kp_divrem`, `kp_gcd`, `kp_squarefree`,
/// `kp_inv_mod`).  The same answers.  Oracles: `λ² − √2·λ − 1` has the
/// roots `(√2 ± √6)/2` (quadratic formula), compared at 1e-12; the
/// Jordan block `[[√2, 1], [0, √2]]` has the double eigenvalue `√2` (Yun
/// over `K`: `(λ − √2)²`) with one eigenvector; `λ³ − √2` (companion
/// matrix) has three distinct roots `2^{1/6}·ωᵏ`; `A·v = λ·v` at 1e-12.
#[test]
fn polynomials_over_a_square_root_tower_are_dense_polys() {
    let ctx = Context::new();
    let p = |s: &str| ctx.parse(s).unwrap();
    let a = Matrix::new(vec![vec![p("sqrt(2)"), p("1")], vec![p("1"), p("0")]]).unwrap();
    let mut vals: Vec<f64> = a
        .eigenvals()
        .unwrap()
        .iter()
        .map(|l| l.eval_complex64().unwrap().re)
        .collect();
    vals.sort_by(f64::total_cmp);
    let (s2, s6) = (2f64.sqrt(), 6f64.sqrt());
    assert!((vals[0] - (s2 - s6) / 2.0).abs() < 1e-12, "{vals:?}");
    assert!((vals[1] - (s2 + s6) / 2.0).abs() < 1e-12, "{vals:?}");
    let j = Matrix::new(vec![vec![p("sqrt(2)"), p("1")], vec![p("0"), p("sqrt(2)")]]).unwrap();
    let ev = j.eigenvals_with_multiplicity().unwrap();
    assert_eq!(ev.len(), 1);
    assert_eq!((ev[0].0.to_string(), ev[0].1), ("sqrt(2)".to_string(), 2));
    let c = Matrix::new(vec![
        vec![p("0"), p("0"), p("sqrt(2)")],
        vec![p("1"), p("0"), p("0")],
        vec![p("0"), p("1"), p("0")],
    ])
    .unwrap();
    for m in [&a, &j, &c] {
        let vecs = m.eigenvects().unwrap();
        let n: usize = vecs.iter().map(|(_, k, _)| *k).sum();
        assert_eq!(n, m.nrows());
        for (lam, _, vs) in &vecs {
            for v in vs {
                let r = &(m * v) - &v.scale(lam);
                for i in 0..m.nrows() {
                    let z = r.get(i, 0).eval_complex64().unwrap();
                    assert!(z.norm() < 1e-12, "{lam}: {}", r.get(i, 0));
                }
            }
        }
    }
    assert_eq!(j.eigenvects().unwrap()[0].2.len(), 1);
    let lam = c.eigenvals().unwrap();
    assert_eq!(lam.len(), 3);
    for l in &lam {
        let z = l.eval_complex64().unwrap();
        assert!((z.norm() - 2f64.powf(1.0 / 6.0)).abs() < 1e-12, "{l}");
    }
}

fn g(re: i64, im: i64) -> GaussQ {
    GaussQ {
        re: q(re),
        im: q(im),
    }
}

fn gp(c: &[(i64, i64)]) -> GenPoly<GaussQ> {
    GenPoly::from_coeffs(c.iter().map(|&(a, b)| g(a, b)).collect())
}

/// New in 0.50: `GenPoly<C>`'s gcd, extended gcd, resultant and Yun's
/// decomposition are `DensePoly`'s over `FieldOf<C>` (one inversion per
/// division instead of a field division per quotient term), after the
/// field's fast paths (`ℚ`'s primitive PRS).  The same polynomials.
/// Oracles over `ℚ(i)`: `gcd((x − i)²(x + 1), (x − i)(x − 2)) = x − i`;
/// the Yun factors of `(x − i)²(x + 1)` by construction; `res(x − i, x +
/// i) = 2i` (the value of `x + i` at `i`) and `res(x² + 1, x² − 2x + 2) =
/// ∏ (α − β) = 5` over the roots `±i`, `1 ± i`; the Bezout identity.
#[test]
fn generic_polynomial_algorithms_run_on_dense_poly() {
    let x_i = gp(&[(0, -1), (1, 0)]); // x − i
    let f = x_i.mul(&x_i).mul(&gp(&[(1, 0), (1, 0)]));
    let h = x_i.mul(&gp(&[(-2, 0), (1, 0)]));
    assert_eq!(GenPoly::gcd(&f, &h), x_i);
    let e = GenPoly::extended_gcd(&f, &h);
    assert_eq!(e.gcd, x_i);
    assert_eq!(e.x.mul(&f).add(&e.y.mul(&h)), x_i);
    assert_eq!(
        f.squarefree_factors(),
        vec![(gp(&[(1, 0), (1, 0)]), 1), (x_i.clone(), 2)]
    );
    assert_eq!(GenPoly::resultant(&x_i, &gp(&[(0, 1), (1, 0)])), g(0, 2));
    let a = gp(&[(1, 0), (0, 0), (1, 0)]);
    let b = gp(&[(2, 0), (-2, 0), (1, 0)]);
    assert_eq!(GenPoly::resultant(&a, &b), g(5, 0));
}

/// Before: the parametric rational integrator proved the syntactic
/// factors of a high-degree denominator square-free and coprime by full
/// gcds in `ℤ[t, params]`; after the Euler substitution of
/// `∫ (A + B·x)(b·x + c·x²)^{3/2}/(d + e·x)⁸ dx` the factor is an eighth
/// power written out (degree 16 in `t`, four parameters), and its gcd with
/// its derivative (degree 14) took 0.19 s each of the 14 times the route
/// was tried: 3.3 s for the integral (Rubi 1.2.1.3 #1321 with the
/// harness's `e` → `e_`), now 0.57 s with modular images (the same answer:
/// unevaluated in both; release build, not asserted here).  Coprime
/// factors are still proved so and integrated by the factored route.
/// Oracle: the antiderivative is checked by differentiation
/// (`d/dx F − f` is 0 at `a = 2`, `b = 3`, the generic branch of its
/// `Piecewise` in `a ≠ 0`).
#[test]
fn factored_route_keeps_answers_with_modular_coprimality() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    // Denominator degree 19 > MAX_DENOM_DEGREE = 10, factors x² − a and x
    // as written: the factored route.
    let f = ctx.parse("(x^20 + b)/((x^2 - a)^8*x^3)").unwrap();
    let big_f = f.integrate(&x);
    assert!(!big_f.has_unevaluated(), "{big_f}");
    let (a, b) = (ctx.symbol("a"), ctx.symbol("b"));
    let check = (&big_f.diff(&x) - &f)
        .subs(&a, &ctx.int(2))
        .subs(&b, &ctx.int(3))
        .simplify();
    assert!(check.is_zero_structural(), "{check}");
}
