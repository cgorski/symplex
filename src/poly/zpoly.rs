//! Integer-scaled working forms for polynomials over ℚ.
//!
//! Every `Ratio<BigInt>` operation reduces its fraction with an integer
//! gcd.  Algorithms that perform many operations per output coefficient
//! (polynomial products, powers, evaluation, the Euclidean remainder
//! sequence) are much faster when they first multiply through by the
//! least common multiple of the denominators, work in `ℤ` and reduce once
//! at the end.  This module collects those forms:
//!
//! - the common-denominator helpers [`denominator_lcm`],
//!   [`integer_scaled`], [`integer_content`], [`z_primitive`],
//!   [`z_normalize`] and [`powers`];
//! - the primitive polynomial remainder sequence in `ℤ\[x\]`
//!   ([`pseudo_rem_pos`], [`z_gcd`]) and the monic gcd over ℚ computed
//!   through it ([`gcd_via_z`], the fast path behind
//!   [`GenPoly::gcd`](super::generic::GenPoly::gcd) for rational
//!   coefficients);
//! - [`ZPoly`], a sparse multivariate polynomial over a common denominator
//!   — the working form behind [`MultiPoly::mul`](super::multipoly::MultiPoly::mul),
//!   [`MultiPoly::pow`](super::multipoly::MultiPoly::pow) and
//!   [`MultiPoly::eval`](super::multipoly::MultiPoly::eval).
//!
//! # Why the gcd goes through ℤ\[x\]
//!
//! Euclid's algorithm over ℚ on a degree-30 polynomial produces a
//! remainder sequence whose coefficients have hundreds of bits, and every
//! rational operation on them costs an integer gcd — over a second in a
//! debug build.  The *primitive PRS* in `ℤ\[x\]` scales the inputs to
//! integers, takes pseudo-remainders (no division at all) and strips the
//! integer content after each step.  The monic gcd over a field is unique,
//! so the result is exactly what Euclid returns.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::multipoly::{MonoKey, MonomialOrd, MultiPoly};

// ═══════════════════════════════════════════════════════════════════════════
// Common-denominator helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Positive least common multiple of the denominators of `coeffs` (`1`
/// when there are none).
pub(crate) fn denominator_lcm<'a>(coeffs: impl IntoIterator<Item = &'a Ratio<BigInt>>) -> BigInt {
    coeffs.into_iter().fold(BigInt::one(), |acc, c| {
        if c.denom().is_one() {
            acc
        } else {
            acc.lcm(c.denom())
        }
    })
}

/// Non-negative gcd of `xs` (`0` when there are none or all are zero).
/// Stops early once the gcd reaches `1`.
pub(crate) fn integer_content<'a>(xs: impl IntoIterator<Item = &'a BigInt>) -> BigInt {
    let mut g = BigInt::zero();
    for x in xs {
        g = g.gcd(x);
        if g.is_one() {
            break;
        }
    }
    g
}

/// The numerator of `c` over the common denominator `den` (a multiple of
/// `denom(c)`): `c · den`.
#[inline]
fn scaled_numer(c: &Ratio<BigInt>, den: &BigInt) -> BigInt {
    if c.denom().is_one() {
        c.numer() * den
    } else {
        c.numer() * (den / c.denom())
    }
}

/// `coeffs` scaled by the (positive) least common multiple of their
/// denominators: integer coefficients in the same order, the same sign as
/// the input at every point.
pub(crate) fn integer_scaled(coeffs: &[Ratio<BigInt>]) -> Vec<BigInt> {
    let lcm = denominator_lcm(coeffs);
    coeffs.iter().map(|c| scaled_numer(c, &lcm)).collect()
}

/// Drop trailing zero coefficients.
pub(crate) fn z_normalize(c: &mut Vec<BigInt>) {
    while c.last().is_some_and(Zero::is_zero) {
        c.pop();
    }
}

/// `c / gcd(coefficients)` with the gcd taken positive: the unique
/// primitive integer polynomial that is a positive multiple of `c`.
pub(crate) fn z_primitive(c: &[BigInt]) -> Vec<BigInt> {
    let g = integer_content(c);
    if g.is_zero() || g.is_one() {
        return c.to_vec();
    }
    c.iter().map(|x| x / &g).collect()
}

/// `base^exp` of a reduced rational, reduced (no gcd is needed: powers of
/// coprime integers stay coprime).
pub(crate) fn pow_ratio(base: &Ratio<BigInt>, exp: u32) -> Ratio<BigInt> {
    if base.is_zero() {
        return if exp == 0 {
            Ratio::one()
        } else {
            Ratio::zero()
        };
    }
    Ratio::new_raw(base.numer().pow(exp), base.denom().pow(exp))
}

/// `[1, b, b², …, b^n]`.
pub(crate) fn powers(b: &BigInt, n: usize) -> Vec<BigInt> {
    let mut out = Vec::with_capacity(n + 1);
    let mut acc = BigInt::one();
    for _ in 0..=n {
        out.push(acc.clone());
        acc *= b;
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Primitive PRS in ℤ[x]
// ═══════════════════════════════════════════════════════════════════════════

/// `|lc(b)|^{deg a − deg b + 1} · rem(a, b)` in `ℤ\[x\]` (ascending, normalised;
/// empty for zero) — the pseudo-remainder with a *positive* scale factor,
/// so its sign agrees with `rem(a, b)` everywhere (the Sturm chain relies
/// on that).  `b` must be non-zero.  Returns `a` itself when
/// `deg a < deg b`.
pub(crate) fn pseudo_rem_pos(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let mut r = a.to_vec();
    z_normalize(&mut r);
    let Some(lc_b) = b.last() else {
        return r;
    };
    let m = b.len() - 1;
    let abs_lc = lc_b.abs();
    let neg_lc = lc_b.is_negative();
    while r.len() > m {
        let k = r.len() - 1;
        // r ← |lc_b| · r − sign(lc_b) · r_k · x^{k−m} · b   (kills the x^k term)
        let rk = r[k].clone();
        for c in &mut r {
            *c *= &abs_lc;
        }
        for (j, bj) in b.iter().enumerate() {
            if neg_lc {
                r[k - m + j] += &rk * bj;
            } else {
                r[k - m + j] -= &rk * bj;
            }
        }
        r.truncate(k);
        z_normalize(&mut r);
    }
    r
}

/// `gcd(a, b)` in `ℤ\[x\]` by the primitive PRS, primitive with positive
/// leading coefficient (`[1]` when coprime; empty when both are zero).
pub(crate) fn z_gcd(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    z_normalize(&mut a);
    z_normalize(&mut b);
    if a.len() < b.len() {
        std::mem::swap(&mut a, &mut b);
    }
    while !b.is_empty() {
        let r = z_primitive(&pseudo_rem_pos(&a, &b));
        a = b;
        b = r;
    }
    let mut g = z_primitive(&a);
    if g.last().is_some_and(Signed::is_negative) {
        for c in &mut g {
            *c = -std::mem::take(c);
        }
    }
    g
}

/// The monic `gcd(a, b)` over ℚ of two coefficient vectors (ascending
/// degree) — exactly what Euclid's algorithm over ℚ returns, including the
/// conventions `gcd(a, 0) = monic(a)` and `gcd(0, 0) = 0` (the empty
/// vector) — computed through the primitive PRS in `ℤ\[x\]` (see the module
/// note).
pub(crate) fn gcd_via_z(a: &[Ratio<BigInt>], b: &[Ratio<BigInt>]) -> Vec<Ratio<BigInt>> {
    let g = z_gcd(&integer_scaled(a), &integer_scaled(b));
    let Some(lc) = g.last() else {
        return Vec::new();
    };
    if lc.is_one() {
        g.into_iter().map(Ratio::from_integer).collect()
    } else {
        g.iter()
            .map(|c| Ratio::new(c.clone(), lc.clone()))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Extended gcd through the primitive PRS
// ═══════════════════════════════════════════════════════════════════════════

/// Pseudo-division in `ℤ[x]`: `(q, r, l)` with `l·a = q·b + r`,
/// `deg r < deg b`, and `l = lc(b)^k` for the `k` reduction steps taken
/// (at most `deg a − deg b + 1`).  `b` must be non-zero and normalised.
fn pseudo_divmod(a: &[BigInt], b: &[BigInt]) -> (Vec<BigInt>, Vec<BigInt>, BigInt) {
    let mut r = a.to_vec();
    z_normalize(&mut r);
    let mut l = BigInt::one();
    let Some(lc_b) = b.last() else {
        return (Vec::new(), r, l);
    };
    let m = b.len() - 1;
    let mut q = vec![BigInt::zero(); r.len().saturating_sub(m)];
    while r.len() > m {
        // q ← lc_b·q + r_k·x^s,  r ← lc_b·r − r_k·x^s·b   (kills the x^k term)
        let k = r.len() - 1;
        let shift = k - m;
        let rk = r[k].clone();
        for c in &mut q {
            *c *= lc_b;
        }
        q[shift] += &rk;
        for c in &mut r {
            *c *= lc_b;
        }
        for (j, bj) in b.iter().enumerate() {
            r[shift + j] -= &rk * bj;
        }
        r.truncate(k);
        z_normalize(&mut r);
        l *= lc_b;
    }
    z_normalize(&mut q);
    (q, r, l)
}

/// `a·b` in `ℤ[x]` (ascending, normalised).
fn z_mul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    z_normalize(&mut out);
    out
}

/// `u·a − v·b` in `ℤ[x]` for integer scalars `u`, `v` (normalised).
fn z_lin(u: &BigInt, a: &[BigInt], v: &BigInt, b: &[BigInt]) -> Vec<BigInt> {
    let mut out = vec![BigInt::zero(); a.len().max(b.len())];
    for (o, ai) in out.iter_mut().zip(a) {
        *o += u * ai;
    }
    for (o, bi) in out.iter_mut().zip(b) {
        *o -= v * bi;
    }
    z_normalize(&mut out);
    out
}

/// One row of the extended remainder sequence: `s·A + t·B = d·r`, with
/// `r` the (primitive) remainder and the cofactors `s/d`, `t/d` held over
/// one common denominator in lowest terms.
struct ExtRow {
    r: Vec<BigInt>,
    s: Vec<BigInt>,
    t: Vec<BigInt>,
    d: BigInt,
}

/// The monic `gcd(a, b)` over ℚ with its Bézout cofactors `x·a + y·b = gcd`
/// (ascending coefficient vectors) — exactly what the extended Euclidean
/// algorithm over ℚ returns, computed through the primitive PRS in `ℤ[x]`.
/// `None` when `b` is zero (the caller's convention for that case).
///
/// Every member of the primitive PRS is a non-zero scalar multiple of the
/// corresponding Euclidean remainder, and its cofactors are the same
/// multiples of Euclid's; normalising the last member to be monic divides
/// the multiple out, so `gcd`, `x` and `y` agree with Euclid exactly (they
/// are also the unique cofactors with `deg x < deg b − deg gcd`).  The
/// cofactors are kept as integer vectors over one denominator per row, so a
/// step costs integer arithmetic and one content gcd instead of a rational
/// reduction per coefficient operation — Euclid over ℚ on degree-12 inputs
/// with 40-digit coefficients took 4 s.
pub(crate) fn extended_gcd_via_z(
    a: &[Ratio<BigInt>],
    b: &[Ratio<BigInt>],
) -> Option<num_integer::ExtendedGcd<Vec<Ratio<BigInt>>>> {
    let mut side_a = (integer_scaled(a), denominator_lcm(a));
    let mut side_b = (integer_scaled(b), denominator_lcm(b));
    z_normalize(&mut side_a.0);
    z_normalize(&mut side_b.0);
    if side_b.0.is_empty() {
        return None;
    }
    if side_a.0.is_empty() {
        // gcd(0, b) = monic(b) = (1/lc(b))·b: x = 0, y = 1/lc(b).
        let lc = b.iter().rev().find(|c| !c.is_zero())?;
        let gcd = b.iter().take(side_b.0.len()).map(|c| c / lc).collect();
        return Some(num_integer::ExtendedGcd {
            gcd,
            x: Vec::new(),
            y: vec![lc.recip()],
        });
    }
    // Euclid's first step on deg a < deg b is the swap itself.
    let swapped = side_a.0.len() < side_b.0.len();
    if swapped {
        std::mem::swap(&mut side_a, &mut side_b);
    }
    let (za, scale_a) = side_a;
    let (zb, scale_b) = side_b;

    let mut prev = ExtRow {
        r: za,
        s: vec![BigInt::one()],
        t: Vec::new(),
        d: BigInt::one(),
    };
    let mut curr = ExtRow {
        r: zb,
        s: Vec::new(),
        t: vec![BigInt::one()],
        d: BigInt::one(),
    };
    loop {
        let (q, rem, l) = pseudo_divmod(&prev.r, &curr.r);
        if rem.is_empty() {
            break;
        }
        // l·r_prev − q·r_curr = rem, with r = (s·A + t·B)/d on each row:
        // (l·d_c·s_p − q·d_p·s_c)·A + (…)·B = d_p·d_c·rem.
        let ld = &l * &curr.d;
        let qs = z_mul(&q, &curr.s);
        let qt = z_mul(&q, &curr.t);
        let mut s = z_lin(&ld, &prev.s, &prev.d, &qs);
        let mut t = z_lin(&ld, &prev.t, &prev.d, &qt);
        let mut d = &prev.d * &curr.d;
        let c = integer_content(&rem);
        let r: Vec<BigInt> = rem.iter().map(|x| x / &c).collect();
        d *= &c;
        let g = integer_content(s.iter().chain(t.iter()).chain(std::iter::once(&d)));
        if !g.is_one() && !g.is_zero() {
            for x in s.iter_mut().chain(t.iter_mut()) {
                *x /= &g;
            }
            d /= &g;
        }
        prev = curr;
        curr = ExtRow { r, s, t, d };
    }

    // s·A + t·B = d·r with A = scale_a·a', B = scale_b·b' (a', b' the
    // inputs in swapped order), so the monic gcd is r/lc and the cofactors
    // are s·scale_a/(d·lc), t·scale_b/(d·lc).
    let lc = curr.r.last()?.clone();
    let den = &curr.d * &lc;
    let over = |v: &[BigInt], scale: &BigInt| -> Vec<Ratio<BigInt>> {
        v.iter()
            .map(|c| Ratio::new(c * scale, den.clone()))
            .collect()
    };
    let gcd: Vec<Ratio<BigInt>> = curr
        .r
        .iter()
        .map(|c| Ratio::new(c.clone(), lc.clone()))
        .collect();
    let cof_a = over(&curr.s, &scale_a);
    let cof_b = over(&curr.t, &scale_b);
    let (x, y) = if swapped {
        (cof_b, cof_a)
    } else {
        (cof_a, cof_b)
    };
    Some(num_integer::ExtendedGcd { gcd, x, y })
}

// ═══════════════════════════════════════════════════════════════════════════
// ZPoly — sparse multivariate polynomial over a common denominator
// ═══════════════════════════════════════════════════════════════════════════

/// An exact polynomial over a common denominator, `Σ nᵢ · xᵉⁱ / den`: the
/// working form for products, powers and evaluation, where accumulating
/// integer numerators avoids a gcd reduction per term product.  The
/// result is reduced once when converted back to a
/// [`MultiPoly`].  Terms are keyed by the same monomial order `O`.
pub(crate) struct ZPoly<O: MonomialOrd> {
    den: BigInt,
    terms: BTreeMap<MonoKey<O>, BigInt>,
}

impl<O: MonomialOrd> ZPoly<O> {
    /// The constant `1` in `nv` variables.
    pub(crate) fn one(nv: usize) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(MonoKey::new(vec![0u32; nv]), BigInt::one());
        ZPoly {
            den: BigInt::one(),
            terms,
        }
    }

    /// `mp` over the least common multiple of its coefficient denominators.
    pub(crate) fn from_multipoly(mp: &MultiPoly<O>) -> Self {
        let den = denominator_lcm(mp.terms().map(|(_, c)| c));
        let terms = mp
            .terms()
            .map(|(e, c)| {
                let n = if den.is_one() {
                    c.numer().clone()
                } else {
                    c.numer() * (&den / c.denom())
                };
                (MonoKey::new(e.to_vec()), n)
            })
            .collect();
        ZPoly { den, terms }
    }

    /// Back to reduced rational coefficients (one reduction per term).
    pub(crate) fn into_multipoly(self, nv: usize) -> MultiPoly<O> {
        let den = self.den;
        let terms: BTreeMap<MonoKey<O>, Ratio<BigInt>> = if den.is_one() {
            self.terms
                .into_iter()
                .map(|(e, n)| (e, Ratio::from_integer(n)))
                .collect()
        } else {
            self.terms
                .into_iter()
                .map(|(e, n)| (e, Ratio::new(n, den.clone())))
                .collect()
        };
        MultiPoly::from_term_map(nv, terms)
    }

    /// Product; `None` if some exponent would overflow `u32`.
    pub(crate) fn mul(&self, other: &Self) -> Option<Self> {
        let mut terms: BTreeMap<MonoKey<O>, BigInt> = BTreeMap::new();
        for (ea, na) in &self.terms {
            for (eb, nb) in &other.terms {
                let e: Option<Vec<u32>> = ea
                    .exponents
                    .iter()
                    .zip(&eb.exponents)
                    .map(|(a, b)| a.checked_add(*b))
                    .collect();
                let prod = na * nb;
                match terms.entry(MonoKey::new(e?)) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(prod);
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        *slot.get_mut() += prod;
                    }
                }
            }
        }
        terms.retain(|_, n| !n.is_zero());
        Some(ZPoly {
            den: &self.den * &other.den,
            terms,
        })
    }

    /// `self^n` by repeated squaring; `None` on exponent overflow.
    pub(crate) fn pow(self, nv: usize, n: u32) -> Option<Self> {
        let mut result = ZPoly::one(nv);
        let mut base = self;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                result = result.mul(&base)?;
            }
            k >>= 1;
            if k > 0 {
                base = base.mul(&base)?;
            }
        }
        Some(result)
    }

    /// Maximum exponent of each variable (all zeros for the zero polynomial).
    fn degree_list(&self, nv: usize) -> Vec<u32> {
        let mut out = vec![0u32; nv];
        for e in self.terms.keys() {
            for (o, &x) in out.iter_mut().zip(&e.exponents) {
                *o = (*o).max(x);
            }
        }
        out
    }

    /// Exact value at the rational point `vals` (`vals.len()` variables),
    /// with one reduction at the end: every term is brought over the
    /// common denominator `den · Π qᵢ^{dᵢ}` with `dᵢ` the degree in
    /// variable `i`.
    pub(crate) fn eval(&self, vals: &[Ratio<BigInt>]) -> Ratio<BigInt> {
        let degs = self.degree_list(vals.len());
        let num_pows: Vec<Vec<BigInt>> = vals
            .iter()
            .zip(&degs)
            .map(|(v, &d)| powers(v.numer(), d as usize))
            .collect();
        let den_pows: Vec<Vec<BigInt>> = vals
            .iter()
            .zip(&degs)
            .map(|(v, &d)| {
                if v.denom().is_one() {
                    Vec::new()
                } else {
                    powers(v.denom(), d as usize)
                }
            })
            .collect();
        let mut sum = BigInt::zero();
        for (e, n) in &self.terms {
            let mut t = n.clone();
            for (i, &ei) in e.exponents.iter().enumerate() {
                if ei > 0 {
                    t *= &num_pows[i][ei as usize];
                }
                let rest = degs[i] - ei;
                if rest > 0 && !den_pows[i].is_empty() {
                    t *= &den_pows[i][rest as usize];
                }
            }
            sum += t;
        }
        let mut den = self.den.clone();
        for (i, &d) in degs.iter().enumerate() {
            if d > 0 && !den_pows[i].is_empty() {
                den *= &den_pows[i][d as usize];
            }
        }
        Ratio::new(sum, den)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::multipoly::GrevLex;

    fn r(n: i64, d: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(n), BigInt::from(d))
    }

    /// The fast extended gcd must return exactly Euclid's `(gcd, x, y)` —
    /// the same three polynomials, not merely a valid Bézout identity —
    /// on edge cases and on seeded random inputs with repeated factors,
    /// rational coefficients and every degree ordering.
    #[test]
    fn extended_gcd_via_z_matches_euclid() {
        use crate::base::rng::SplitMix64;
        use crate::poly::dense::Poly;
        let p = |c: &[Ratio<BigInt>]| Poly::from_coeffs(c.to_vec());
        let check = |a: &Poly, b: &Poly| {
            let fast = Poly::extended_gcd(a, b);
            let slow = Poly::extended_gcd_euclid(a, b);
            assert_eq!(fast.gcd, slow.gcd, "gcd of {a:?}, {b:?}");
            assert_eq!(fast.x, slow.x, "x of {a:?}, {b:?}");
            assert_eq!(fast.y, slow.y, "y of {a:?}, {b:?}");
            assert_eq!(&(&fast.x * a) + &(&fast.y * b), fast.gcd);
        };
        let zero = Poly::zero();
        let one = p(&[r(1, 1)]);
        let c = p(&[r(-7, 3)]);
        let x2m1 = p(&[r(-1, 1), r(0, 1), r(1, 1)]);
        let xm1 = p(&[r(-1, 1), r(1, 1)]);
        let half = p(&[r(1, 2), r(1, 2)]);
        for (a, b) in [
            (&zero, &x2m1),
            (&x2m1, &zero),
            (&zero, &zero),
            (&c, &x2m1),
            (&x2m1, &c),
            (&one, &c),
            (&x2m1, &xm1),
            (&xm1, &x2m1),
            (&x2m1, &x2m1),
            (&half, &x2m1),
        ] {
            check(a, b);
        }
        let mut rng = SplitMix64::new(20260922);
        let mut rand_poly = |deg: usize, big: bool| -> Poly {
            let coeffs: Vec<Ratio<BigInt>> = (0..=deg)
                .map(|_| {
                    let n = (rng.next_u64() % 41) as i64 - 20;
                    let d = (rng.next_u64() % 5) as i64 + 1;
                    let mut v = r(n, d);
                    if big {
                        v *= Ratio::from_integer(BigInt::from(10u32).pow(30));
                    }
                    v
                })
                .collect();
            p(&coeffs)
        };
        for i in 0..120 {
            let common = rand_poly(i % 4, false);
            let a = &rand_poly(i % 7, i % 5 == 0) * &common;
            let b = &rand_poly((i * 3) % 6, i % 3 == 0) * &common;
            let b = if i % 11 == 0 { &b * &common } else { b };
            check(&a, &b);
            check(&b, &a);
        }
    }

    #[test]
    fn denominator_lcm_and_integer_scaled() {
        let c = [r(1, 2), r(3, 4), r(5, 1), r(-1, 6)];
        assert_eq!(denominator_lcm(&c), BigInt::from(12));
        assert_eq!(
            integer_scaled(&c),
            vec![
                BigInt::from(6),
                BigInt::from(9),
                BigInt::from(60),
                BigInt::from(-2)
            ]
        );
        assert_eq!(denominator_lcm(&[]), BigInt::one());
        assert!(integer_scaled(&[]).is_empty());
    }

    #[test]
    fn content_and_primitive() {
        let c = [BigInt::from(6), BigInt::from(-9), BigInt::from(15)];
        assert_eq!(integer_content(&c), BigInt::from(3));
        assert_eq!(
            z_primitive(&c),
            vec![BigInt::from(2), BigInt::from(-3), BigInt::from(5)]
        );
        assert_eq!(integer_content(&[]), BigInt::zero());
        assert_eq!(integer_content(&[BigInt::zero()]), BigInt::zero());
    }

    #[test]
    fn pow_ratio_matches_repeated_product() {
        for (n, d) in [(2i64, 3i64), (-5, 7), (0, 1), (4, 1), (-1, 1)] {
            let b = r(n, d);
            let mut acc = Ratio::one();
            for e in 0..6u32 {
                assert_eq!(pow_ratio(&b, e), acc, "({n}/{d})^{e}");
                acc *= &b;
            }
        }
    }

    #[test]
    fn powers_table() {
        let p = powers(&BigInt::from(3), 4);
        assert_eq!(
            p,
            vec![
                BigInt::from(1),
                BigInt::from(3),
                BigInt::from(9),
                BigInt::from(27),
                BigInt::from(81)
            ]
        );
        assert_eq!(powers(&BigInt::from(7), 0), vec![BigInt::one()]);
    }

    #[test]
    fn gcd_via_z_conventions() {
        let a = [r(-2, 1), r(0, 1), r(2, 1)]; // 2x² − 2 = 2(x−1)(x+1)
        let b = [r(-3, 1), r(3, 1)]; // 3x − 3
        assert_eq!(gcd_via_z(&a, &b), vec![r(-1, 1), r(1, 1)]);
        assert_eq!(gcd_via_z(&a, &[]), vec![r(-1, 1), r(0, 1), r(1, 1)]);
        assert_eq!(gcd_via_z(&[], &b), vec![r(-1, 1), r(1, 1)]);
        assert!(gcd_via_z(&[], &[]).is_empty());
        assert_eq!(gcd_via_z(&[r(6, 1)], &[r(-4, 1)]), vec![r(1, 1)]);
        // Rational coefficients: gcd(x/2 + 1/2, x² − 1) = x + 1.
        assert_eq!(
            gcd_via_z(&[r(1, 2), r(1, 2)], &[r(-1, 1), r(0, 1), r(1, 1)]),
            vec![r(1, 1), r(1, 1)]
        );
    }

    #[test]
    fn zpoly_round_trip_and_mul() {
        let x: MultiPoly<GrevLex> = MultiPoly::var(2, 0);
        let y: MultiPoly<GrevLex> = MultiPoly::var(2, 1);
        let f = x.scale(&r(1, 2)).add(&y.scale(&r(2, 3)));
        let z = ZPoly::from_multipoly(&f);
        assert_eq!(z.den, BigInt::from(6));
        assert_eq!(z.clone_into_multipoly(2), f);
        let sq = ZPoly::from_multipoly(&f)
            .mul(&ZPoly::from_multipoly(&f))
            .map(|z| z.into_multipoly(2));
        assert_eq!(sq, Some(f.mul(&f)));
    }

    #[test]
    fn zpoly_mul_overflow_is_none() {
        let big: MultiPoly<GrevLex> = MultiPoly::monomial(r(1, 1), vec![u32::MAX]);
        let x: MultiPoly<GrevLex> = MultiPoly::var(1, 0);
        assert!(
            ZPoly::from_multipoly(&big)
                .mul(&ZPoly::from_multipoly(&x))
                .is_none()
        );
    }

    impl<O: MonomialOrd> ZPoly<O> {
        fn clone_into_multipoly(&self, nv: usize) -> MultiPoly<O> {
            ZPoly {
                den: self.den.clone(),
                terms: self.terms.clone(),
            }
            .into_multipoly(nv)
        }
    }
}
