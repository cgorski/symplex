//! 0.21 track: the consolidated polynomial kernel.
//!
//! One gcd (the primitive PRS in `ℤ[x]` behind `Poly::gcd`), one sparse
//! integer-scaled multivariate kernel (`ZPoly` behind `MultiPoly::mul`,
//! `pow` and `eval`), one Newton interpolation and one Cauchy bound.  Each
//! consolidation is *output-preserving* — the monic gcd, the product, the
//! value and the interpolant are unique — so every test here compares the
//! crate against an independent reference computed in this file, through
//! the public API only.

use std::time::Instant;

use symplex::multipoly::{GrevLex, MultiPoly};
use symplex::num_bigint::BigInt;
use symplex::num_rational::Ratio;
use symplex::num_traits::{One, Signed, Zero};
use symplex::poly_ex::Poly;
use symplex::prelude::*;
use symplex::stats::Rng;

type Q = Ratio<BigInt>;

fn q(n: i64, d: i64) -> Q {
    Ratio::new(BigInt::from(n), BigInt::from(d))
}

fn qi(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference implementations (dense univariate over ℚ, ascending degree)
// ═══════════════════════════════════════════════════════════════════════════

fn normalize(mut c: Vec<Q>) -> Vec<Q> {
    while c.last().is_some_and(Zero::is_zero) {
        c.pop();
    }
    c
}

/// Schoolbook product.
fn ref_mul(a: &[Q], b: &[Q]) -> Vec<Q> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![qi(0); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    normalize(out)
}

/// Euclidean remainder `a mod b` (`b` non-zero).
fn ref_rem(a: &[Q], b: &[Q]) -> Vec<Q> {
    let mut r = a.to_vec();
    let lb = b.last().cloned().unwrap_or_else(|| qi(1));
    while r.len() >= b.len() && !r.is_empty() {
        let k = r.len() - b.len();
        let c = r.last().cloned().unwrap_or_else(|| qi(0)) / &lb;
        for (j, bj) in b.iter().enumerate() {
            r[k + j] -= &c * bj;
        }
        r = normalize(r);
        r.truncate(k + b.len() - 1);
        r = normalize(r);
    }
    r
}

/// Euclid's algorithm over ℚ, monic result (`gcd(0, 0) = 0`): the
/// reference `Poly::gcd` used to be, and must still equal.
fn ref_gcd_euclid(a: &[Q], b: &[Q]) -> Vec<Q> {
    let mut a = normalize(a.to_vec());
    let mut b = normalize(b.to_vec());
    while !b.is_empty() {
        let r = ref_rem(&a, &b);
        a = b;
        b = r;
    }
    let Some(lc) = a.last().cloned() else {
        return a;
    };
    a.iter().map(|c| c / &lc).collect()
}

/// Random polynomial of the given degree with coefficients `n/d`,
/// `n ∈ [-9, 9]`, `d ∈ [1, 5]` (integers only when `!frac`), non-zero
/// leading coefficient.
fn random_poly(rng: &mut Rng, deg: usize, frac: bool) -> Vec<Q> {
    let mut coeffs: Vec<Q> = (0..=deg)
        .map(|_| {
            let n = rng.below(19) as i64 - 9;
            let d = if frac { rng.below(5) as i64 + 1 } else { 1 };
            q(n, d)
        })
        .collect();
    if coeffs[deg].is_zero() {
        coeffs[deg] = q(rng.below(4) as i64 + 2, if frac { 3 } else { 1 });
    }
    coeffs
}

/// `Σ cₖ xᵏ` as a univariate `MultiPoly`.
fn mp_from_coeffs(coeffs: &[Q]) -> MultiPoly<GrevLex> {
    MultiPoly::from_terms(
        1,
        coeffs
            .iter()
            .enumerate()
            .map(|(k, c)| (vec![k as u32], c.clone()))
            .collect(),
    )
    .unwrap()
}

fn ex_from_coeffs(ctx: &Context, x: &Ex, coeffs: &[Q]) -> Ex {
    mp_from_coeffs(coeffs).to_ex(ctx, &[x]).unwrap()
}

/// Ascending dense coefficients of a univariate polynomial expression.
fn coeffs_of(e: &Ex, x: &Ex) -> Vec<Q> {
    let mp = Poly::new(e, &[x]).unwrap().to_multipoly().unwrap();
    let deg = mp.degree_in(0) as usize;
    let mut out = vec![qi(0); deg + 1];
    for (exp, c) in mp.terms() {
        out[exp[0] as usize] = c.clone();
    }
    normalize(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// gcd: the ℤ[x] fast path equals Euclid over ℚ
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn gcd_fast_path_equals_euclid_on_random_rational_polynomials() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0001);
    for case in 0..40 {
        let frac = case % 2 == 0;
        let (dg, dp, dr) = (1 + rng.below(4), rng.below(7), rng.below(7));
        let g = random_poly(&mut rng, dg, frac);
        let p = random_poly(&mut rng, dp, frac);
        let r = random_poly(&mut rng, dr, !frac);
        let a = ref_mul(&p, &g);
        let b = ref_mul(&r, &g);
        let expected = ref_gcd_euclid(&a, &b);
        let got = ex_from_coeffs(&ctx, &x, &a)
            .poly_gcd(&ex_from_coeffs(&ctx, &x, &b), &x)
            .unwrap();
        assert_eq!(
            coeffs_of(&got, &x),
            expected,
            "case {case}: a = {a:?}, b = {b:?}"
        );
        assert!(
            expected.len() >= g.len(),
            "case {case}: gcd lost the planted factor"
        );
        // Monic, whatever the leading coefficients of the inputs.
        assert!(expected.last().is_some_and(One::is_one));
    }
}

#[test]
fn gcd_fast_path_equals_euclid_on_coprime_nonmonic_and_constant_inputs() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0002);
    for case in 0..30 {
        let (da, db) = (rng.below(9), rng.below(9));
        let a = random_poly(&mut rng, da, true);
        let b = random_poly(&mut rng, db, case % 3 == 0);
        let expected = ref_gcd_euclid(&a, &b);
        let got = ex_from_coeffs(&ctx, &x, &a)
            .poly_gcd(&ex_from_coeffs(&ctx, &x, &b), &x)
            .unwrap();
        assert_eq!(coeffs_of(&got, &x), expected, "case {case}");
    }
    // Degree 0: gcd(c, p) = 1 for a non-zero constant c.
    let p = ex_from_coeffs(&ctx, &x, &[q(1, 2), q(-3, 4), qi(5)]);
    assert_eq!(ctx.rational(6, 7).poly_gcd(&p, &x).unwrap(), ctx.int(1));
    assert_eq!(p.poly_gcd(&ctx.int(-4), &x).unwrap(), ctx.int(1));
    assert_eq!(ctx.int(6).poly_gcd(&ctx.int(-4), &x).unwrap(), ctx.int(1));
    // gcd(p, 0) is the monic p.
    let monic = p.poly_gcd(&ctx.int(0), &x).unwrap();
    assert_eq!(coeffs_of(&monic, &x), vec![q(1, 10), q(-3, 20), qi(1)]);
    // Non-monic common factor comes out monic.
    let a = ex_from_coeffs(&ctx, &x, &ref_mul(&[qi(2), qi(4)], &[qi(-3), qi(0), qi(6)]));
    let b = ex_from_coeffs(&ctx, &x, &ref_mul(&[qi(2), qi(4)], &[qi(5), qi(7)]));
    assert_eq!(
        coeffs_of(&a.poly_gcd(&b, &x).unwrap(), &x),
        vec![q(1, 2), qi(1)]
    );
}

#[test]
fn square_free_part_uses_the_same_gcd() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0003);
    for case in 0..12 {
        let (df, dh) = (1 + rng.below(3), 1 + rng.below(3));
        let f = random_poly(&mut rng, df, case % 2 == 0);
        let h = random_poly(&mut rng, dh, false);
        let p = ref_mul(&ref_mul(&f, &f), &h);
        let e = ex_from_coeffs(&ctx, &x, &p);
        // p / gcd(p, p') with Euclid's gcd.
        let dp: Vec<Q> = p
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, c)| c * qi(k as i64))
            .collect();
        let g = ref_gcd_euclid(&p, &normalize(dp));
        let g_ex = ex_from_coeffs(&ctx, &x, &g);
        // `Ex::square_free_part` returns the primitive integer form; compare
        // the monic polynomials.
        let expected = e.poly_quo(&g_ex, &x).unwrap().monic(&x).unwrap();
        assert_eq!(
            e.square_free_part(&x).unwrap().monic(&x).unwrap(),
            expected,
            "case {case}"
        );
        assert_eq!(e.is_squarefree(&x), Some(g.len() == 1), "case {case}");
    }
}

#[test]
fn degree_30_gcd_with_degree_10_common_factor_is_fast_in_debug() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0030);
    let g = random_poly(&mut rng, 10, false);
    let a = ref_mul(&random_poly(&mut rng, 20, false), &g);
    let b = ref_mul(&random_poly(&mut rng, 20, false), &g);
    let ea = ex_from_coeffs(&ctx, &x, &a);
    let eb = ex_from_coeffs(&ctx, &x, &b);
    let start = Instant::now();
    let got = ea.poly_gcd(&eb, &x).unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "degree-30 gcd took {elapsed:?} (Euclid over ℚ took a third of a second and more)"
    );
    assert_eq!(got.degree(&x), Some(10));
    // Still the monic Euclid gcd.
    assert_eq!(coeffs_of(&got, &x), ref_gcd_euclid(&a, &b));
}

// ═══════════════════════════════════════════════════════════════════════════
// MultiPoly: mul, pow and eval through the common-denominator kernel
// ═══════════════════════════════════════════════════════════════════════════

/// Random sparse polynomial in `nv` variables with rational coefficients.
fn random_multipoly(rng: &mut Rng, nv: usize, terms: usize, max_exp: u32) -> MultiPoly<GrevLex> {
    let mut t = Vec::with_capacity(terms);
    for _ in 0..terms {
        let exps: Vec<u32> = (0..nv)
            .map(|_| rng.below(max_exp as usize + 1) as u32)
            .collect();
        let c = q(rng.below(41) as i64 - 20, rng.below(6) as i64 + 1);
        t.push((exps, c));
    }
    MultiPoly::from_terms(nv, t).unwrap()
}

/// The direct term-by-term product with `Ratio` arithmetic (the former
/// `MultiPoly::mul`).
fn ref_mp_mul(a: &MultiPoly<GrevLex>, b: &MultiPoly<GrevLex>) -> MultiPoly<GrevLex> {
    let mut terms = Vec::new();
    for (ea, ca) in a.terms() {
        for (eb, cb) in b.terms() {
            let e: Vec<u32> = ea.iter().zip(eb).map(|(p, r)| p + r).collect();
            terms.push((e, ca * cb));
        }
    }
    MultiPoly::from_terms(a.num_vars(), terms).unwrap()
}

/// The former linear-power evaluation: `Σ c · Π vᵢ^{eᵢ}` in `Ratio`s.
fn ref_mp_eval(p: &MultiPoly<GrevLex>, vals: &[Q]) -> Q {
    let mut acc = qi(0);
    for (e, c) in p.terms() {
        let mut t = c.clone();
        for (i, &ei) in e.iter().enumerate() {
            for _ in 0..ei {
                t *= &vals[i];
            }
        }
        acc += t;
    }
    acc
}

#[test]
fn multipoly_mul_equals_direct_rational_product() {
    let mut rng = Rng::new(0x5EED_0010);
    for case in 0..25 {
        let nv = 1 + rng.below(3);
        let (ta, tb) = (1 + rng.below(8), 1 + rng.below(8));
        let a = random_multipoly(&mut rng, nv, ta, 4);
        let b = random_multipoly(&mut rng, nv, tb, 4);
        let prod = a.mul(&b);
        assert_eq!(prod, ref_mp_mul(&a, &b), "case {case}");
        assert_eq!(
            prod.to_string(),
            ref_mp_mul(&a, &b).to_string(),
            "case {case}"
        );
        assert_eq!(a.try_mul(&b), Some(prod));
    }
    let zero: MultiPoly<GrevLex> = MultiPoly::zero(2);
    let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
    assert!(zero.mul(&x).is_zero());
    assert_eq!(x.mul(&MultiPoly::from_int(2, 1)), x);
}

#[test]
fn multipoly_pow_equals_repeated_mul() {
    let mut rng = Rng::new(0x5EED_0011);
    for case in 0..12 {
        let nv = 1 + rng.below(3);
        let tp = 1 + rng.below(4);
        let p = random_multipoly(&mut rng, nv, tp, 2);
        let mut acc = MultiPoly::from_int(nv, 1);
        for n in 0..=6u32 {
            assert_eq!(p.pow(n), acc, "case {case}, n = {n}");
            assert_eq!(p.try_pow(n), Some(acc.clone()));
            acc = acc.mul(&p);
        }
    }
    let zero: MultiPoly<GrevLex> = MultiPoly::zero(2);
    assert_eq!(zero.pow(0), MultiPoly::from_int(2, 1));
    assert!(zero.pow(3).is_zero());
}

#[test]
fn multipoly_mul_and_pow_report_exponent_overflow() {
    let huge: MultiPoly<GrevLex> = MultiPoly::monomial(qi(1), vec![u32::MAX, 0]);
    let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
    assert!(huge.try_mul(&x).is_none());
    assert!(huge.try_pow(2).is_none());
    // Different variable counts are reported, not asserted, by `try_mul`.
    let y1: MultiPoly<GrevLex> = MultiPoly::var(1, 0);
    assert!(x.try_mul(&y1).is_none());
}

#[test]
fn multipoly_eval_equals_linear_power_route() {
    let mut rng = Rng::new(0x5EED_0012);
    for case in 0..25 {
        let nv = 1 + rng.below(3);
        let tp = 1 + rng.below(8);
        let p = random_multipoly(&mut rng, nv, tp, 5);
        let vals: Vec<Q> = (0..nv)
            .map(|_| q(rng.below(21) as i64 - 10, rng.below(4) as i64 + 1))
            .collect();
        assert_eq!(
            p.eval(&vals).unwrap(),
            ref_mp_eval(&p, &vals),
            "case {case}"
        );
        // Integer points too (no denominator powers).
        let ints: Vec<Q> = (0..nv).map(|_| qi(rng.below(7) as i64 - 3)).collect();
        assert_eq!(
            p.eval(&ints).unwrap(),
            ref_mp_eval(&p, &ints),
            "case {case} (integers)"
        );
    }
    let zero: MultiPoly<GrevLex> = MultiPoly::zero(2);
    assert_eq!(zero.eval(&[q(1, 2), q(3, 4)]).unwrap(), qi(0));
    let c: MultiPoly<GrevLex> = MultiPoly::from_int(0, 7);
    assert_eq!(c.eval(&[]).unwrap(), qi(7));
}

// ═══════════════════════════════════════════════════════════════════════════
// Interpolation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn interpolation_reproduces_known_polynomials_through_n_plus_1_rational_points() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0020);
    for case in 0..15 {
        let n = rng.below(8);
        let p = random_poly(&mut rng, n, true);
        let mp = mp_from_coeffs(&p);
        // n + 1 distinct rational abscissae.
        let mut xs: Vec<Q> = Vec::new();
        while xs.len() < n + 1 {
            let xk = q(rng.below(31) as i64 - 15, rng.below(3) as i64 + 1);
            if !xs.contains(&xk) {
                xs.push(xk);
            }
        }
        let pts: Vec<(Ex, Ex)> = xs
            .iter()
            .map(|xk| {
                let y = mp.eval(std::slice::from_ref(xk)).unwrap();
                (ctx.from_ratio(xk.clone()), ctx.from_ratio(y))
            })
            .collect();
        let ip = Ex::poly_interpolate(&pts, &x).unwrap();
        assert_eq!(coeffs_of(&ip, &x), normalize(p.clone()), "case {case}");
    }
    // The doc example, and a zero polynomial.
    let pts = [
        (ctx.int(0), ctx.int(1)),
        (ctx.int(1), ctx.int(2)),
        (ctx.int(2), ctx.int(5)),
    ];
    assert_eq!(
        Ex::poly_interpolate(&pts, &x).unwrap().to_string(),
        "x^2 + 1"
    );
    let pts = [(ctx.int(0), ctx.int(0)), (ctx.int(3), ctx.int(0))];
    assert_eq!(Ex::poly_interpolate(&pts, &x).unwrap(), ctx.int(0));
}

#[test]
fn interpolation_rejects_repeated_abscissa() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts = [
        (ctx.int(1), ctx.int(1)),
        (ctx.rational(1, 2), ctx.int(3)),
        (ctx.int(1), ctx.int(2)),
    ];
    assert!(Ex::poly_interpolate(&pts, &x).is_none());
    // …even when the ordinates agree.
    let pts = [(ctx.int(1), ctx.int(1)), (ctx.int(1), ctx.int(1))];
    assert!(Ex::poly_interpolate(&pts, &x).is_none());
}

#[test]
fn interpolation_at_31_points_is_fast_in_debug() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let pts: Vec<(Ex, Ex)> = (0..=30i64)
        .map(|k| (ctx.int(k), ctx.rational(k * k * k - 7 * k + 3, k + 1)))
        .collect();
    let start = Instant::now();
    let ip = Ex::poly_interpolate(&pts, &x).unwrap();
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs_f64() < 2.0, "took {elapsed:?}");
    assert_eq!(ip.degree(&x), Some(30));
    // It really interpolates.
    for k in [0i64, 7, 30] {
        let v = ip.subs(&x, &ctx.int(k));
        assert_eq!(v, ctx.rational(k * k * k - 7 * k + 3, k + 1), "at {k}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cauchy bound
// ═══════════════════════════════════════════════════════════════════════════

/// `1 + maxᵢ |aᵢ / aₙ|`, the bound the crate's Sturm and Aberth code use.
fn cauchy_bound(coeffs: &[Q]) -> Q {
    let lc = coeffs.last().cloned().unwrap_or_else(|| qi(1));
    let mut m = qi(0);
    for c in &coeffs[..coeffs.len() - 1] {
        let r = (c / &lc).abs();
        if r > m {
            m = r;
        }
    }
    m + qi(1)
}

#[test]
fn cauchy_bound_bounds_every_real_root() {
    let ctx = Context::new();
    let x = ctx.symbol("x");
    let mut rng = Rng::new(0x5EED_0040);
    for case in 0..12 {
        // A polynomial with known rational roots (some large), times a
        // root-free quadratic, with a non-monic leading coefficient.
        let mut roots: Vec<Q> = Vec::new();
        for _ in 0..(1 + rng.below(4)) {
            let r = q(rng.below(2001) as i64 - 1000, rng.below(7) as i64 + 1);
            if !roots.contains(&r) {
                roots.push(r);
            }
        }
        let mut p = vec![q(rng.below(5) as i64 + 1, rng.below(3) as i64 + 1)];
        for r in &roots {
            p = ref_mul(&p, &[-r.clone(), qi(1)]);
        }
        p = ref_mul(
            &p,
            &[
                qi(1 + rng.below(9) as i64),
                qi(rng.below(5) as i64 - 2),
                qi(1),
            ],
        );
        let bound = cauchy_bound(&p);
        for r in &roots {
            assert!(
                r.abs() <= bound,
                "case {case}: root {r} beyond bound {bound}"
            );
        }
        // Exact Sturm counts agree: every real root lies in (−B, B].
        let e = ex_from_coeffs(&ctx, &x, &p);
        let b = ctx.from_ratio(bound.clone());
        let nb = ctx.from_ratio(-bound);
        assert_eq!(e.count_real_roots(&x), Some(roots.len()), "case {case}");
        assert_eq!(
            e.count_real_roots_in(&x, &nb, &b),
            Some(roots.len()),
            "case {case}"
        );
        // …and the isolating intervals (found inside the crate's bound)
        // capture each planted root.
        let ivs = e.real_roots_isolate(&x);
        assert_eq!(ivs.len(), roots.len(), "case {case}");
        for r in &roots {
            assert!(
                ivs.iter().any(|iv| {
                    let lo = iv.lower.as_rational().unwrap();
                    let hi = iv.upper.as_rational().unwrap();
                    lo <= *r && *r <= hi
                }),
                "case {case}: root {r} not in any interval {ivs:?}"
            );
        }
    }
}
