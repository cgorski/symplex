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
//! - the subresultant PRS in `ℤ[t][x]` ([`ztx_subresultant_prs`]), behind
//!   the Lazard–Rioboo–Trager logarithmic part and the resultant over ℚ
//!   ([`resultant_via_z`], the fast path behind
//!   [`GenPoly::resultant`](super::generic::GenPoly::resultant));
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
// Subresultant PRS in ℤ[t][x]
// ═══════════════════════════════════════════════════════════════════════════

/// A polynomial in `x` over `ℤ[t]`: entry `k` is the coefficient of `x^k`,
/// itself a dense polynomial in `t` (ascending, normalised, empty for 0).
/// The outer vector is normalised too (empty for the zero polynomial).
pub(crate) type ZtxPoly = Vec<Vec<BigInt>>;

fn zt_mul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![BigInt::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (o, bj) in out[i..].iter_mut().zip(b) {
            *o += ai * bj;
        }
    }
    z_normalize(&mut out);
    out
}

fn zt_sub(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let mut out: Vec<BigInt> = (0..a.len().max(b.len()))
        .map(|i| match (a.get(i), b.get(i)) {
            (Some(x), Some(y)) => x - y,
            (Some(x), None) => x.clone(),
            (None, Some(y)) => -y,
            (None, None) => BigInt::zero(),
        })
        .collect();
    z_normalize(&mut out);
    out
}

fn zt_neg(a: &[BigInt]) -> Vec<BigInt> {
    a.iter().map(|c| -c).collect()
}

fn zt_pow(a: &[BigInt], mut n: usize) -> Vec<BigInt> {
    let mut acc = vec![BigInt::one()];
    let mut base = a.to_vec();
    while n > 0 {
        if n & 1 == 1 {
            acc = zt_mul(&acc, &base);
        }
        n >>= 1;
        if n > 0 {
            base = zt_mul(&base, &base);
        }
    }
    acc
}

/// `a / b` in `ℤ[t]` when `b` divides `a` exactly; `None` when it does not
/// or `b` is zero.
fn zt_div_exact(a: &[BigInt], b: &[BigInt]) -> Option<Vec<BigInt>> {
    let lc_b = b.last()?;
    if a.is_empty() {
        return Some(Vec::new());
    }
    let m = b.len() - 1;
    if a.len() <= m {
        return None;
    }
    let mut r = a.to_vec();
    let mut q = vec![BigInt::zero(); a.len() - m];
    for k in (m..a.len()).rev() {
        let rk = std::mem::take(&mut r[k]);
        if rk.is_zero() {
            continue;
        }
        let (qk, rem) = rk.div_rem(lc_b);
        if !rem.is_zero() {
            return None;
        }
        for (rj, bj) in r[k - m..k].iter_mut().zip(b) {
            *rj -= &qk * bj;
        }
        q[k - m] = qk;
    }
    if r.iter().any(|c| !c.is_zero()) {
        return None;
    }
    z_normalize(&mut q);
    Some(q)
}

fn ztx_normalize(p: &mut ZtxPoly) {
    while p.last().is_some_and(Vec::is_empty) {
        p.pop();
    }
}

/// `c · p` for `c ∈ ℤ[t]`.
fn ztx_scale(p: &ZtxPoly, c: &[BigInt]) -> ZtxPoly {
    let mut out: ZtxPoly = p.iter().map(|pk| zt_mul(pk, c)).collect();
    ztx_normalize(&mut out);
    out
}

/// `p / c` coefficient by coefficient, when `c ∈ ℤ[t]` divides every
/// coefficient exactly.
fn ztx_div_exact(p: &ZtxPoly, c: &[BigInt]) -> Option<ZtxPoly> {
    p.iter().map(|pk| zt_div_exact(pk, c)).collect()
}

/// The pseudo-remainder `lc(g)^{deg f − deg g + 1} · f mod g` in `ℤ[t][x]`
/// (`f` itself when `deg f < deg g`), computed without division (Knuth,
/// *TAOCP* vol. 2, §4.6.1, Algorithm R); `None` for `g = 0`.
fn ztx_prem(f: &ZtxPoly, g: &ZtxPoly) -> Option<ZtxPoly> {
    let lc_g = g.last()?;
    let dg = g.len() - 1;
    let mut r = f.clone();
    ztx_normalize(&mut r);
    if r.len() <= dg {
        return Some(r);
    }
    // Each pass lowers the degree of `r` by at least one.
    let mut passes_left = r.len() - dg;
    while r.len() > dg {
        let top = r.len() - 1;
        let lc_r = std::mem::take(&mut r[top]);
        let shift = top - dg;
        // r ← lc(g)·r − lc(r)·x^shift·g; the x^top terms cancel exactly.
        r.pop();
        for c in r.iter_mut() {
            *c = zt_mul(c, lc_g);
        }
        for (rk, gk) in r[shift..].iter_mut().zip(&g[..dg]) {
            *rk = zt_sub(rk, &zt_mul(&lc_r, gk));
        }
        ztx_normalize(&mut r);
        passes_left = passes_left.saturating_sub(1);
    }
    if passes_left > 0 {
        r = ztx_scale(&r, &zt_pow(lc_g, passes_left));
    }
    Some(r)
}

/// The subresultant chain of two polynomials in `ℤ[t][x]`: see
/// [`ztx_subresultant_prs`].
pub(crate) struct Subresultants {
    /// The members of the subresultant PRS, keyed by degree in `x`.
    pub(crate) members: BTreeMap<usize, ZtxPoly>,
    /// `res_x(f, g) ∈ ℤ[t]` (empty when it is zero: `f` and `g` have a
    /// common factor of positive degree in `x`).
    pub(crate) resultant: Vec<BigInt>,
}

/// The subresultant polynomial remainder sequence of `f` and `g` in
/// `ℤ[t][x]`, keyed by degree in `x`, and their resultant.  `f` and `g`
/// are the first two members (swapped if `deg f < deg g`); every later
/// member is `±` the subresultant `S_{d−1}(f, g)` whose index is one below
/// the degree `d` of the member before it (Brown's fundamental theorem),
/// so a member of degree `k` is a multiple of `S_k` by a factor that
/// vanishes only where `S_k` itself degenerates.  The resultant is the
/// last scalar subresultant when the sequence ends in a constant.
///
/// Unlike Euclid's algorithm over `ℚ(t)`, whose coefficients grow in
/// degree with every step, the coefficients here are polynomials in `t`
/// of bounded degree (the subresultant is a determinant in the inputs'
/// coefficients), and each step costs one pseudo-remainder and one exact
/// division — no gcd at all.
///
/// The structure follows SymPy's `dup_inner_subresultants` and
/// `dup_prs_resultant` (`sympy/polys/euclidtools.py`, BSD-3), which
/// implement W. S. Brown, "The subresultant PRS algorithm", *ACM TOMS* 4
/// (1978) 237–249 (after G. E. Collins, "Subresultants and reduced
/// polynomial remainder sequences", *J. ACM* 14 (1967) 128–142, and
/// W. S. Brown & J. F. Traub, *J. ACM* 18 (1971) 505–514).  `None` if an
/// exact division is not exact, which the theory rules out, or if `f` or
/// `g` is zero.
pub(crate) fn ztx_subresultant_prs(f: &ZtxPoly, g: &ZtxPoly) -> Option<Subresultants> {
    let (mut f, mut g) = (f.clone(), g.clone());
    ztx_normalize(&mut f);
    ztx_normalize(&mut g);
    if f.len() < g.len() {
        std::mem::swap(&mut f, &mut g);
    }
    if g.is_empty() {
        return None;
    }
    let mut members = BTreeMap::new();
    members.insert(f.len() - 1, f.clone());
    members.insert(g.len() - 1, g.clone());

    let mut m = g.len() - 1;
    let d = f.len() - g.len();
    // h = (−1)^{d+1} prem(f, g)
    let mut h = ztx_prem(&f, &g)?;
    if d % 2 == 0 {
        h = h.iter().map(|c| zt_neg(c)).collect();
    }
    let mut lc = g.last()?.clone();
    // c is the negated scalar subresultant of the newest member `g`
    // (SymPy's convention): −lc(g)^d to begin with.
    let mut c = zt_neg(&zt_pow(&lc, d));
    while !h.is_empty() {
        let k = h.len() - 1;
        members.insert(k, h.clone());
        let d = m - k;
        m = k;
        f = std::mem::replace(&mut g, h);
        let b = zt_neg(&zt_mul(&lc, &zt_pow(&c, d)));
        h = ztx_div_exact(&ztx_prem(&f, &g)?, &b)?;
        lc = g.last()?.clone();
        c = if d > 1 {
            zt_div_exact(&zt_pow(&zt_neg(&lc), d), &zt_pow(&c, d - 1))?
        } else {
            zt_neg(&lc)
        };
    }
    let resultant = if g.len() == 1 { zt_neg(&c) } else { Vec::new() };
    Some(Subresultants { members, resultant })
}

/// `res(a, b)` of two polynomials over ℚ with `deg a ≥ deg b ≥ 1`
/// (ascending, no trailing zeros), through the subresultant PRS of their
/// integer-scaled forms in `ℤ[x]` ([`ztx_subresultant_prs`] with constant
/// coefficients): `res(λ·a, μ·b) = λ^{deg b}·μ^{deg a}·res(a, b)`.  Euclid
/// over ℚ reduces a fraction by an integer gcd at every operation, 0.3 s
/// per resultant of two degree-36 polynomials in a debug build.  `None`
/// outside the degree precondition.
pub(crate) fn resultant_via_z(a: &[Ratio<BigInt>], b: &[Ratio<BigInt>]) -> Option<Ratio<BigInt>> {
    if b.len() < 2 || a.len() < b.len() || a.last()?.is_zero() || b.last()?.is_zero() {
        return None;
    }
    let scale = |p: &[Ratio<BigInt>]| -> (BigInt, ZtxPoly) {
        let den = denominator_lcm(p);
        let z = p
            .iter()
            .map(|c| {
                let n = scaled_numer(c, &den);
                if n.is_zero() { Vec::new() } else { vec![n] }
            })
            .collect();
        (den, z)
    };
    let (la, az) = scale(a);
    let (lb, bz) = scale(b);
    let res = ztx_subresultant_prs(&az, &bz)?
        .resultant
        .first()
        .cloned()
        .unwrap_or_default();
    let (m, n) = (a.len() - 1, b.len() - 1);
    let unscale = la.pow(u32::try_from(n).ok()?) * lb.pow(u32::try_from(m).ok()?);
    Some(Ratio::new(res, unscale))
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

    /// The subresultant PRS in `ℤ[t][x]` against SymPy 1.14:
    /// `subresultants(x**6 + t*x + 1, x**4 + 1, x)` gives
    /// `[x**6 + t*x + 1, x**4 + 1, -t*x + x**2 - 1,`
    /// `-t**3*x - t**2 - 2*t*x - 2, t**4 + 4*t**2 + 4]` and
    /// `resultant(x**6 + t*x + 1, x**4 + 1, x)` gives `t**4 + 4*t**2 + 4`.
    /// The degree drop 4 → 2 takes the abnormal branch.
    #[test]
    fn ztx_subresultant_prs_matches_sympy() {
        let z = |v: &[i64]| -> Vec<BigInt> {
            let mut c: Vec<BigInt> = v.iter().map(|&k| BigInt::from(k)).collect();
            z_normalize(&mut c);
            c
        };
        let f: ZtxPoly = vec![z(&[1]), z(&[0, 1]), z(&[]), z(&[]), z(&[]), z(&[]), z(&[1])];
        let g: ZtxPoly = vec![z(&[1]), z(&[]), z(&[]), z(&[]), z(&[1])];
        let chain = ztx_subresultant_prs(&f, &g).unwrap();
        assert_eq!(
            chain.members.keys().copied().collect::<Vec<_>>(),
            [0, 1, 2, 4, 6]
        );
        assert_eq!(chain.members[&2], vec![z(&[-1]), z(&[0, -1]), z(&[1])]);
        assert_eq!(chain.members[&1], vec![z(&[-2, 0, -1]), z(&[0, -2, 0, -1])]);
        assert_eq!(chain.members[&0], vec![z(&[4, 0, 4, 0, 1])]);
        assert_eq!(chain.resultant, z(&[4, 0, 4, 0, 1]));
        // A common factor: the resultant is zero.
        let h: ZtxPoly = vec![z(&[0, 1]), z(&[1])]; // x + t
        let fh: ZtxPoly = vec![z(&[0, 1]), z(&[1]), z(&[0, 1]), z(&[1])]; // (x + t)(x² + 1)
        assert!(ztx_subresultant_prs(&fh, &h).unwrap().resultant.is_empty());
    }

    /// `resultant_via_z` returns exactly Euclid's resultant over ℚ.  The
    /// fixed pair is SymPy 1.14's `resultant(x**5 - 3*x**2 + Rational(7, 2)*x
    /// - Rational(1, 3), Rational(2, 5)*x**3 + x - 4, x)` = `-47418733/84375`.
    #[test]
    fn resultant_via_z_matches_euclid() {
        use crate::base::rng::SplitMix64;
        use crate::poly::dense::Poly;
        fn euclid(a: &Poly, b: &Poly) -> Ratio<BigInt> {
            let (m, n) = (a.degree().unwrap(), b.degree().unwrap());
            if n == 0 {
                return pow_ratio(&b.coeff(0), m as u32);
            }
            let rem = a.rem(b);
            if rem.is_zero() {
                return Ratio::zero();
            }
            let s = rem.degree().unwrap();
            let sign = if (m * n) % 2 == 0 { r(1, 1) } else { r(-1, 1) };
            sign * pow_ratio(&b.coeff(n), (m - s) as u32) * euclid(b, &rem)
        }
        let a = [r(-1, 3), r(7, 2), r(-3, 1), r(0, 1), r(0, 1), r(1, 1)];
        let b = [r(-4, 1), r(1, 1), r(0, 1), r(2, 5)];
        assert_eq!(resultant_via_z(&a, &b), Some(r(-47418733, 84375)));
        let mut rng = SplitMix64::new(20260924);
        let mut rand_poly = |deg: usize| -> Poly {
            let mut coeffs: Vec<Ratio<BigInt>> = (0..deg)
                .map(|_| {
                    r(
                        (rng.next_u64() % 41) as i64 - 20,
                        (rng.next_u64() % 5) as i64 + 1,
                    )
                })
                .collect();
            coeffs.push(r(
                (rng.next_u64() % 9) as i64 + 1,
                (rng.next_u64() % 3) as i64 + 1,
            ));
            Poly::from_coeffs(coeffs)
        };
        for i in 0..80 {
            let n = 1 + i % 5;
            let m = n + (i * 7) % 6;
            let (a, b) = (rand_poly(m), rand_poly(n));
            let (a, b) = if i % 9 == 0 { (&a * &b, b) } else { (a, b) };
            assert_eq!(
                resultant_via_z(a.coeffs(), b.coeffs()),
                Some(euclid(&a, &b)),
                "res({a:?}, {b:?})"
            );
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
