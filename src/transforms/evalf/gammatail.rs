//! The series sum of `γ(s, x)` far below the mean of a huge shape.
//!
//! `γ(s, x) = x^s e^{−x}·I(s, x)` with
//!
//! ```text
//! I(s, x) = Σ_{k ≥ 0} x^k / (s(s + 1)⋯(s + k)) = ∫₀^∞ e^{−ψ(u)} du,   ψ(u) = s·u − x(1 − e^{−u})
//! ```
//!
//! (`t = x e^{−u}` in `γ(s, x) = ∫₀^x t^{s−1} e^{−t} dt`).  The series
//! needs about `wp·ln 2·s/d` terms for `d = s − x ≫ √s`: 170,000 at `s =
//! 1.7·10¹⁰` and the quantile of `10⁻³⁰⁰` (0.3 s per value; the lower tail
//! of `ChiSquared(3.4·10¹⁰)`, a `fuzz_stats` slow unit of 14–28 s).
//!
//! For `0 < x < s`, `ψ` is increasing with `ψ′(u) = s − x e^{−u} ≥ d > 0`,
//! so with `v = ψ(u)`, `I = ∫₀^∞ e^{−v} g(v) dv`, `g = 1/ψ′`.  Writing
//! `w = x e^{−u} ∈ (0, x]` (so `dw/dv = −w/(s − w)`), induction gives
//!
//! ```text
//! g⁽ᵏ⁾(v) = (−1)ᵏ · sᵏ b_k(w/s) / (s − w)^(2k+1),
//! b₀ = 1,   b_k(λ) = λ(1 − λ) b′_{k−1}(λ) + (2k − 1) λ b_{k−1}(λ)
//! ```
//!
//! (the polynomials of DLMF 8.11.6).  For `k ≥ 1`, `b_k(λ) = Σ_{j<k}
//! e_{k,j} λ^{j+1}` with `e_{k,j} = (j + 1) e_{k−1,j} + (2k − 1 − j)
//! e_{k−1,j−1}`: the coefficients are sums of products of non-negative
//! integers (Eulerian numbers of the second kind), so `b_k` increases on
//! `λ > 0`, and `|g⁽ᵏ⁾|` — an increasing function of `w` — is largest at
//! `w = x`, that is at `v = 0`.  Taylor's formula with Lagrange's remainder
//! and `∫₀^∞ e^{−v} vᵏ/k! dv = 1` then give, for every `K ≥ 1`,
//!
//! ```text
//! I = Σ_{k<K} (−1)ᵏ T_k + θ·T_K,   |θ| ≤ 1,   T_k = sᵏ b_k(x/s) / d^(2k+1):
//! ```
//!
//! **the error is at most the first omitted term**.  `T_k` falls like
//! `(2k − 1)!!·(s/d²)ᵏ` until `k ≈ d²/2s`, so the expansion reaches `wp`
//! bits when `d²/s` exceeds about `1.4·wp`; where it does not, the route
//! declines and the series runs.  The terms are real balls ([`RBall`]): the
//! roundings are part of the bound that is checked against the precision.

use astro_float::BigFloat;
use num_bigint::BigInt;
use num_traits::One;

use super::accuracy;
use super::ball::RBall;
use crate::base::numeric::Q;

/// Most terms of the expansion.
const MAX_TERMS: usize = 300;

/// The series needs fewer terms than this: it is as fast.
const MIN_SERIES_TERMS: f64 = 256.0;

/// `I(s, x)` (see the module documentation) for exact `0 < x < s` to a
/// relative error below `2^(−wp)`, and the precision it was formed at;
/// `None` when the expansion does not reach that precision or the series
/// is short (`series_terms`: the caller's estimate of its length).
pub(super) fn far_below_sum(
    s: &BigFloat,
    x: &BigFloat,
    series_terms: f64,
    wp: usize,
) -> Option<(BigFloat, usize)> {
    let (s_f, x_f) = (super::f64_of(s).ok()?, super::f64_of(x).ok()?);
    let d_f = s_f - x_f;
    // A guide: the least term is about `e^{−d²/2s}`.  The bound below is
    // checked on the terms themselves.
    let reach = d_f * d_f / s_f;
    let wanted = 1.4 * (wp as f64 + 24.0);
    if x_f.is_nan() || x_f <= 0.0 || d_f.is_nan() || d_f <= 0.0 {
        return None;
    }
    if reach.is_nan() || reach < wanted || series_terms.is_nan() || series_terms < MIN_SERIES_TERMS
    {
        return None;
    }
    // `d = s − x` loses `log₂(s/d)` bits, `(s/d²)ᵏ` a few more.
    let iwp = wp + 64 + 2 * super::magnitude_bits(s_f / d_f);
    let exact = |v: &BigFloat| RBall {
        m: v.clone(),
        r: f64::NEG_INFINITY,
    };
    let (s, x) = (exact(s), exact(x));
    let d = s.add(&x.neg(), iwp);
    if !d.certainly_positive() || !x.certainly_positive() {
        return None;
    }
    let inv_d = d.recip(iwp)?;
    let lambda = x.mul(&s.recip(iwp)?, iwp);
    let rho = s.mul(&inv_d, iwp).mul(&inv_d, iwp);

    // `e_{k,·}` of the current `k` (empty for `k = 0`).
    let mut e: Vec<BigInt> = Vec::new();
    let mut scale = inv_d;
    let mut sum = RBall::zero(iwp);
    let mut previous = f64::INFINITY;
    for k in 0..=MAX_TERMS {
        let b = if k == 0 {
            RBall::from_int(1, iwp)
        } else {
            let mut acc = RBall::zero(iwp);
            for c in e.iter().rev() {
                acc = acc
                    .mul(&lambda, iwp)
                    .add(&RBall::from_q(&Q::from_integer(c.clone()), iwp), iwp);
            }
            acc.mul(&lambda, iwp)
        };
        let term = b.mul(&scale, iwp);
        let size = term.hi();
        if k > 0 {
            // `T_k` bounds what the `k` terms summed so far leave out.
            if accuracy::lsum(sum.r, size) <= sum.lo() - wp as f64 - 2.0 {
                return Some((sum.m, iwp));
            }
            if size.is_nan() || size >= previous {
                return None;
            }
        }
        previous = size;
        sum = if k % 2 == 0 {
            sum.add(&term, iwp)
        } else {
            sum.add(&term.neg(), iwp)
        };
        // `e_{k+1,j} = (j + 1) e_{k,j} + (2k + 1 − j) e_{k,j−1}`, `j ≤ k`.
        let mut next: Vec<BigInt> = Vec::new();
        for j in 0..=k {
            let mut c = BigInt::from(0);
            if let Some(same) = e.get(j) {
                c += same * BigInt::from(j + 1);
            }
            if j > 0
                && let Some(before) = e.get(j - 1)
            {
                c += before * BigInt::from(2 * k + 1 - j);
            }
            next.push(c);
        }
        if k == 0 {
            next = vec![BigInt::one()];
        }
        e = next;
        scale = scale.mul(&rho, iwp);
    }
    None
}
