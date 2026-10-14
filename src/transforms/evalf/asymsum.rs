//! Infinite sums of smooth terms by their expansion at infinity, with a
//! rigorous remainder.
//!
//! `Σ_{k ≥ a} f(k)` for a summand built from the index `k`, real constants
//! (`√2`, `π`, any evaluable subexpression free of `k`), `+`, `·`, rational
//! powers, `ln`, `exp`, `sin`, `cos`, `atan` and `harmonic`, times `(−1)^k`
//! at most — `ln k/k²`, `1/(k² + √2)`, `(−1)^k ln(k + 1)/(k + 1)`, `H_k/k³`,
//! `(−1)^k/ln(k + 1)` — when its terms are neither hypergeometric nor
//! dominated by a geometric series (`hypsum.rs`), nor rational (`emsum.rs`).  Before 0.44 such sums
//! were refused ("neither hypergeometric … nor a hypergeometric term times
//! factors of polynomial growth").
//!
//! # Method
//!
//! The terms below `N` are summed one by one.  For `x ≥ N`, with `u = 1/x`
//! and `ℓ = ln x`, the summand is written as a *model*
//!
//! ```text
//! f(x) = (−1)^(alt·x) · x^(−β) · ( Σ_{j ≤ J} Σ_m c_{j,m} u^j ℓ^m  +  R(x) ),
//! |R(x)| ≤ E · u^(J+1) · (1 + ℓ)^L   for every real x ≥ N,
//! ```
//!
//! (`β` rational, the coefficients balls, `m` an integer) built bottom-up over
//! the expression — a Taylor model in `u` with Laurent polynomials in `ℓ` as
//! coefficients (a negative power from `1/g` with `g = c₀·ℓ^p·(1 + v)`, the
//! only term of order `u⁰`; `ℓ^m ≤ (ln N)^m` for `m < 0` in every bound).  Sums and
//! products of models are exact up to the terms beyond `u^J`, which go into
//! the remainder bounded by their size at `u = 1/N` (`u^j ≤ u^(J+1)·N^(J+1−j)`
//! for `j > J`, `ℓ^m ≤ (1 + ℓ)^L`); the polynomial part of a factor is bounded
//! by `Σ abs(c_{j,m})·N^(−j)·(1 + ℓ)^L`.  A function of a model `g = x^(−β)·c₀·(1 +
//! v)`, `v → 0`, is its Taylor polynomial in `v` up to degree `J` (Horner's
//! scheme in model arithmetic) plus the Taylor remainder for real `abs(v) ≤ V <
//! 1`, `V` the supremum of `abs(v)` over `x ≥ N` (from `sup_{x≥N} u^j ℓ^m`):
//!
//! | function | expansion | remainder after degree `J` |
//! |---|---|---|
//! | `1/g` | `x^β c₀⁻¹ Σ (−v)^i` | `abs(v)^(J+1)/(1 − V)` |
//! | `g^α` (`c₀ > 0`) | `x^(−αβ) c₀^α Σ C(α, i) v^i` | `abs(C(α, J+1))·abs(v)^(J+1)/(1 − ρV)`, `ρ = max(1, (J + 1 + abs(α))/(J + 2))` (`J + 1 > abs(α)`) |
//! | `ln g` (`c₀ > 0`) | `−βℓ + ln c₀ + Σ (−1)^(i+1) v^i/i` | `abs(v)^(J+1)/((J+1)(1 − V))` |
//! | `exp(c₀ + v)` | `e^(c₀) Σ v^i/i!` | `e^V abs(v)^(J+1)/(J+1)!` |
//! | `sin`, `cos` of `c₀ + v` | `sin c₀ cos v + cos c₀ sin v`, … | `abs(v)^(J+1)/(J+1)!` |
//! | `atan` | of `v → 0` its series; `atan(c₀ + v) = atan c₀ + atan(v/(1 + c₀² + c₀v))`; `π/2 − atan(1/g)` for `g → +∞` | `abs(v)^(J+1)/(J+1)` (`V ≤ 1`) |
//! | `harmonic(g)` (`g → +∞`, real) | `ln g + γ + 1/(2g) − Σ_{i≤I} B₂ᵢ/(2i)·g^(−2i)` | `abs(B₂ᵢ)/(2I)·g^(−2I)` (Euler–Maclaurin for `ψ`, whose derivatives have constant sign) |
//!
//! with `abs(v)^(J+1) ≤ (V₁ u (1 + ℓ)^L₁)^(J+1)`, `abs(v) ≤ V₁ u (1 + ℓ)^L₁` on `x ≥ N`.
//!
//! The tail is then `Σ_{j,m} c_{j,m}·Σ_{k≥N} k^(−β−j)(ln k)^m` plus the sum of
//! the remainder, at most `E·Σ_{k≥N} k^(−β−J−1)(1 + ln k)^L ≤ E·(g(N) + ∫_N^∞
//! g)` for the unimodal `g(x) = x^(−q)(1 + ln x)^L` (`∫_N^∞ g = N^(1−q)·Σ_{r≤L}
//! L!/r!·(1 + ln N)^r/(q − 1)^(L+1−r)`, the incomplete gamma function of an
//! integer order).  The power sums are by Euler–Maclaurin for `f(t) =
//! t^(−s)(ln t)^m`, whose derivatives are `t^(−s−n)·Σ_i p_{n,i}(ln t)^i` with
//! `p_{n+1,i} = −(s + n)p_{n,i} + (i + 1)p_{n,i+1}`:
//!
//! ```text
//! Σ_{k≥N} f(k) = ∫_N^∞ f + f(N)/2 − Σ_{j=1}^{M} B₂ⱼ/(2j)!·f^(2j−1)(N) + R,
//! |R| ≤ abs(B₂ₘ)/(2M)!·∫_N^∞ abs(f^(2M)),
//! ```
//!
//! and an alternating one by the Euler–Boole formula, which has no integral
//! (`Σ_{k≥0} (−1)^k g(k) = 2Σ g(2k) − Σ g(k)`, each by Euler–Maclaurin):
//!
//! ```text
//! Σ_{i≥0} (−1)^i f(N + i) = f(N)/2 − Σ_{j=1}^{M} (2^(2j) − 1)B₂ⱼ/(2j)!·f^(2j−1)(N) + R,
//! |R| ≤ (2^(2M) + 1)·abs(B₂ₘ)/(2M)!·∫_N^∞ abs(f^(2M)),
//! ```
//!
//! (DLMF 2.10.1, 24.17; `∫_N^∞ t^(−q)(ln t)^i dt = N^(1−q)·Σ_{r≤i} i!/(i−r)!·(ln
//! N)^(i−r)/(q − 1)^(r+1)`, and for a negative power an incomplete gamma
//! function from `E₁`).  A coefficient of `k^(−s)(ln k)^m` certainly not 0
//! with `s < 1`, or `s = 1` and `m ≥ −1` (`s < 0`, or `s = 0` and `m ≥ 0`,
//! alternating) makes the sum divergent.
//!
//! SymPy's `evalf_sum` (`sympy/core/evalf.py`, BSD-3) applies the
//! Euler–Maclaurin formula to the summand itself with an error estimate from
//! the last correction, and mpmath's `nsum` (`calculus/extrapolation.py`,
//! BSD) by numerical differentiation and quadrature; here the formula is
//! applied to the powers `k^(−s)(ln k)^m` only, where its remainder has a
//! proven bound, and the expansion of the summand has one too.

use super::ball::{CBall, RBall, two_ulps, ulp};
use super::wp::{RM, Wp};
use astro_float::{BigFloat, Consts};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;

use super::accuracy::{self, Bound};
use super::hypsum::PartFn;
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::{Q, ratio_to_bigfloat};
use crate::base::walk;

const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `log₂ 2π`.
const LOG2_TWO_PI: f64 = 2.651_496_129_472_319;
/// Highest power of `ℓ` in a model.
const MAX_LOG_POWER: usize = 12;
/// Most negative power of `ℓ` in a model.
const MAX_NEG_LOG_POWER: usize = 400;
/// Highest power of `ℓ` in a remainder.
const MAX_REM_LOG_POWER: usize = 400;
/// Most terms summed directly before the tail.
const MAX_N: u64 = 16_384;
/// Highest truncation order `J` of a model (its arithmetic costs `J³`).
const MAX_ORDER: usize = 160;

// ── Balls ──────────────────────────────────────────────────────────────────

// The real balls are `ball::RBall`.

/// `log₂(2^a + 2^b)`, rounded up.
fn lsum(a: f64, b: f64) -> f64 {
    accuracy::lsum(a, b)
}

// ── Models ─────────────────────────────────────────────────────────────────

/// The shared parameters: the truncation order `J`, `N`, precision.
struct Ctx {
    /// Set when a function of a model failed for `abs(v)` not below 1 on `x ≥
    /// N` (a larger `N` may do).
    small_n: std::cell::Cell<bool>,
    j: usize,
    /// `log₂ N` (`log₂ u_max = −ln`).
    lg_n: f64,
    /// `ln N`.
    ln_n: f64,
    wp: usize,
}

/// `x^(−β)·(Σ c[j][i] u^j ℓ^(i−off) + R)`, `|R| ≤ 2^le·u^(J+1)·(1 + ℓ)^l`,
/// times `(−1)^x` when `alt` (see the module documentation).  The offset
/// `off` holds negative powers of `ℓ ≥ ln N > 1` (`1/ln(k + c)`).
#[derive(Clone, Debug)]
struct Model {
    beta: Q,
    alt: bool,
    c: Vec<Vec<RBall>>,
    off: usize,
    le: f64,
    l: usize,
}

impl Model {
    /// The power of `ℓ` at index `i` of a row.
    fn pow_at(&self, i: usize) -> i64 {
        i as i64 - self.off as i64
    }

    /// Its non-negative part (what `(1 + ℓ)^L` must cover: `ℓ^e ≤ 1` for
    /// `e ≤ 0`).
    fn pos_pow_at(&self, i: usize) -> usize {
        i.saturating_sub(self.off)
    }
}

impl Ctx {
    /// `sup_{x ≥ N} x^(−j)·(a + ln x)^m`, `a` 0 or 1, as a `log₂` (`+∞` for `j =
    /// 0 < m`; for `m < 0` both factors decrease, `ln N > 1`).
    fn sup_pow_log(&self, j: f64, m: i64, a: f64) -> f64 {
        if m == 0 {
            return -j * self.lg_n;
        }
        if m < 0 {
            return -j * self.lg_n + m as f64 * (a + self.ln_n).log2();
        }
        if j <= 0.0 {
            return f64::INFINITY;
        }
        let mf = m as f64;
        // The maximum of x^(−j)(a + ln x)^m is at a + ln x = m/j.
        if a + self.ln_n >= mf / j {
            -j * self.lg_n + mf * (a + self.ln_n).log2()
        } else {
            (-mf + j * a) * LOG2E + mf * (mf / j).log2()
        }
    }

    fn zero(&self) -> Model {
        Model {
            beta: Q::zero(),
            alt: false,
            c: vec![Vec::new(); self.j + 1],
            off: 0,
            le: f64::NEG_INFINITY,
            l: 0,
        }
    }

    fn constant(&self, b: RBall) -> Model {
        let mut m = self.zero();
        m.c[0] = vec![b];
        m
    }

    /// `a·ℓ^(−d)` (`ℓ^(−d) ≤ (ln N)^(−d)` in the remainder).
    fn ell_shift_down(&self, a: &Model, d: usize) -> Model {
        let mut out = a.clone();
        out.off += d;
        out.le = a.le - d as f64 * self.ln_n.log2();
        out
    }

    fn var(&self) -> Model {
        let mut m = self.constant(RBall::from_int(1, self.wp));
        m.beta = -Q::one();
        m
    }

    /// The largest (non-negative) power of `ℓ` with a coefficient that is not
    /// an exact 0.
    fn log_degree(&self, a: &Model) -> usize {
        a.c.iter()
            .flat_map(|row| row.iter().enumerate())
            .filter(|(_, b)| !b.exact_zero())
            .map(|(m, _)| a.pos_pow_at(m))
            .max()
            .unwrap_or(0)
    }

    /// `log₂ Σ abs(c_{j,m})·N^(−j)`, the bound of the polynomial part over
    /// `(1 + ℓ)^(log_degree)`; and of the part with `j ≥ 1` over `u` (`Σ
    /// abs(c)·N^(1−j)`).
    fn poly_bounds(&self, a: &Model) -> (f64, f64) {
        let mut all = f64::NEG_INFINITY;
        let mut over_u = f64::NEG_INFINITY;
        for (j, row) in a.c.iter().enumerate() {
            for b in row {
                if b.exact_zero() {
                    continue;
                }
                all = lsum(all, b.hi() - j as f64 * self.lg_n);
                if j >= 1 {
                    over_u = lsum(over_u, b.hi() - (j as f64 - 1.0) * self.lg_n);
                }
            }
        }
        (all, over_u)
    }

    /// `a·u^d`: the coefficients beyond `u^J` into the remainder.
    fn shift(&self, a: &Model, d: usize) -> Option<Model> {
        if d == 0 {
            return Some(a.clone());
        }
        let jj = self.j;
        let mut out = self.zero();
        out.alt = a.alt;
        out.beta = a.beta.clone();
        out.off = a.off;
        out.le = a.le - d as f64 * self.lg_n;
        out.l = a.l;
        for (j, row) in a.c.iter().enumerate() {
            let t = j + d;
            if t <= jj {
                out.c[t] = row.clone();
            } else {
                for (m, b) in row.iter().enumerate() {
                    if b.exact_zero() {
                        continue;
                    }
                    // u^t ℓ^m ≤ N^(J+1−t)·u^(J+1)·(1 + ℓ)^max(m, 0).
                    out.le = lsum(out.le, b.hi() - (t - jj - 1) as f64 * self.lg_n);
                    out.l = out.l.max(a.pos_pow_at(m));
                }
            }
        }
        Some(out)
    }

    /// `a` with `β` lowered to `beta` (an integer step `β_a − beta ≥ 0`).
    fn lower_beta(&self, a: &Model, beta: &Q) -> Option<Model> {
        let d = &a.beta - beta;
        if !d.is_integer() || d.is_negative() {
            return None;
        }
        let mut out = self.shift(a, d.to_integer().to_usize()?)?;
        out.beta = beta.clone();
        Some(out)
    }

    fn add(&self, a: &Model, b: &Model) -> Option<Model> {
        if a.alt != b.alt {
            if self.is_zero(a) {
                return Some(b.clone());
            }
            if self.is_zero(b) {
                return Some(a.clone());
            }
            return None;
        }
        let beta = if a.beta <= b.beta {
            a.beta.clone()
        } else {
            b.beta.clone()
        };
        let off = a.off.max(b.off);
        let a = self.with_off(&self.lower_beta(a, &beta)?, off);
        let b = self.with_off(&self.lower_beta(b, &beta)?, off);
        let mut out = self.zero();
        out.alt = a.alt;
        out.beta = beta;
        out.off = off;
        for j in 0..=self.j {
            let n = a.c[j].len().max(b.c[j].len());
            let mut row = Vec::with_capacity(n);
            for m in 0..n {
                let x = a.c[j]
                    .get(m)
                    .cloned()
                    .unwrap_or_else(|| RBall::zero(self.wp));
                let y = b.c[j]
                    .get(m)
                    .cloned()
                    .unwrap_or_else(|| RBall::zero(self.wp));
                row.push(x.add(&y, self.wp));
            }
            out.c[j] = row;
        }
        out.le = lsum(a.le, b.le);
        out.l = a.l.max(b.l);
        Some(out)
    }

    /// `a` re-indexed to the offset `off ≥ a.off` (zeros for the lower
    /// powers).
    fn with_off(&self, a: &Model, off: usize) -> Model {
        let d = off - a.off.min(off);
        if d == 0 {
            return a.clone();
        }
        let mut out = a.clone();
        out.off = off;
        for row in out.c.iter_mut() {
            if row.is_empty() {
                continue;
            }
            let mut r = vec![RBall::zero(self.wp); d];
            r.append(row);
            *row = r;
        }
        out
    }

    fn is_zero(&self, a: &Model) -> bool {
        a.le == f64::NEG_INFINITY && a.c.iter().all(|row| row.iter().all(RBall::exact_zero))
    }

    fn scale(&self, a: &Model, s: &RBall) -> Model {
        let mut out = a.clone();
        for row in out.c.iter_mut() {
            for b in row.iter_mut() {
                *b = b.mul(s, self.wp);
            }
        }
        out.le = a.le + s.hi();
        out
    }

    fn mul(&self, a: &Model, b: &Model) -> Option<Model> {
        let jj = self.j;
        let mut out = self.zero();
        out.alt = a.alt ^ b.alt;
        out.beta = &a.beta + &b.beta;
        out.off = a.off + b.off;
        let (la, lb) = (self.log_degree(a), self.log_degree(b));
        if la + lb > MAX_LOG_POWER || out.off > MAX_NEG_LOG_POWER {
            return None;
        }
        let mut dropped = f64::NEG_INFINITY;
        let mut dropped_l = 0usize;
        for (j1, r1) in a.c.iter().enumerate() {
            for (m1, x) in r1.iter().enumerate() {
                if x.exact_zero() {
                    continue;
                }
                for (j2, r2) in b.c.iter().enumerate() {
                    for (m2, y) in r2.iter().enumerate() {
                        if y.exact_zero() {
                            continue;
                        }
                        let t = j1 + j2;
                        // The index of ℓ^(e₁ + e₂).
                        let m = m1 + m2;
                        if t <= jj {
                            let row = &mut out.c[t];
                            if row.len() <= m {
                                row.resize(m + 1, RBall::zero(self.wp));
                            }
                            row[m] = row[m].add(&x.mul(y, self.wp), self.wp);
                        } else {
                            dropped =
                                lsum(dropped, x.hi() + y.hi() - (t - jj - 1) as f64 * self.lg_n);
                            dropped_l = dropped_l.max(m.saturating_sub(a.off + b.off));
                        }
                    }
                }
            }
        }
        // P_a·R_b + P_b·R_a + R_a·R_b.
        let (pa, _) = self.poly_bounds(a);
        let (pb, _) = self.poly_bounds(b);
        let mut le = dropped;
        let mut l = dropped_l;
        if a.le > f64::NEG_INFINITY {
            le = lsum(le, pb + a.le);
            l = l.max(lb + a.l);
        }
        if b.le > f64::NEG_INFINITY {
            le = lsum(le, pa + b.le);
            l = l.max(la + b.l);
        }
        if a.le > f64::NEG_INFINITY && b.le > f64::NEG_INFINITY {
            le = lsum(le, a.le + b.le - (jj as f64 + 1.0) * self.lg_n);
            l = l.max(a.l + b.l);
        }
        if l > MAX_REM_LOG_POWER || (!le.is_finite() && le != f64::NEG_INFINITY) {
            return None;
        }
        out.le = le;
        out.l = l;
        Some(self.trim(out))
    }

    /// `a` with the leading exact zeros of its rows dropped, down to the
    /// lowest power held (`off` lowered accordingly, not below 0).
    fn trim(&self, mut a: Model) -> Model {
        let lowest =
            a.c.iter()
                .filter_map(|row| row.iter().position(|b| !b.exact_zero()))
                .min()
                .unwrap_or(0)
                .min(a.off);
        if lowest > 0 {
            for row in a.c.iter_mut() {
                let n = lowest.min(row.len());
                row.drain(..n);
            }
            a.off -= lowest;
        }
        a
    }

    /// `a = x^(−β)·c₀·ℓ^p·(1 + v)`: `c₀` (certainly not 0, the only term of
    /// order `u⁰`, with a power `p ≥ 0` of `ℓ`), `p`, and `v` (`β = 0`, no term
    /// of order `u⁰`).
    fn split_lead(&self, a: &Model) -> Option<(RBall, usize, Model)> {
        let row0 = &a.c[0];
        let mut lead: Option<(usize, RBall)> = None;
        for (i, b) in row0.iter().enumerate() {
            if b.exact_zero() {
                continue;
            }
            if lead.is_some() {
                return None;
            }
            lead = Some((i, b.clone()));
        }
        let (i, c0) = lead?;
        if i < a.off || !c0.certainly_nonzero() {
            return None;
        }
        let p = i - a.off;
        let inv = c0.recip(self.wp)?;
        let mut v = self.ell_shift_down(&self.scale(a, &inv), p);
        v.beta = Q::zero();
        v.alt = false;
        v.c[0] = Vec::new();
        // The remainder over |c₀|·ℓ^p (scale used the upper bound of 1/|c₀|).
        v.le = a.le - c0.lo() - p as f64 * self.ln_n.log2();
        Some((c0, p, v))
    }

    /// `V = sup_{x≥N} abs(v)` and `(V₁, L₁)` with `abs(v) ≤ V₁·u·(1 + ℓ)^L₁` (as `log₂`).
    fn v_bounds(&self, v: &Model) -> (f64, f64, usize) {
        let mut sup = f64::NEG_INFINITY;
        for (j, row) in v.c.iter().enumerate() {
            for (m, b) in row.iter().enumerate() {
                if b.exact_zero() {
                    continue;
                }
                sup = lsum(sup, b.hi() + self.sup_pow_log(j as f64, v.pow_at(m), 0.0));
            }
        }
        if v.le > f64::NEG_INFINITY {
            sup = lsum(
                sup,
                v.le + self.sup_pow_log(self.j as f64 + 1.0, v.l as i64, 1.0),
            );
        }
        let (_, over_u) = self.poly_bounds(v);
        let v1 = lsum(over_u, v.le - self.j as f64 * self.lg_n);
        let l1 = self
            .log_degree(v)
            .max(if v.le > f64::NEG_INFINITY { v.l } else { 0 });
        (sup, v1, l1)
    }

    /// `Σ_{i≤J} coeffs[i]·v^i` (Horner) plus the remainder `2^tail·abs(v)^(J+1)`.
    fn compose(&self, v: &Model, coeffs: &[RBall], tail: f64) -> Option<Model> {
        let jj = self.j;
        let mut acc = self.constant(
            coeffs
                .get(jj)
                .cloned()
                .unwrap_or_else(|| RBall::zero(self.wp)),
        );
        for i in (0..jj).rev() {
            acc = self.mul(&acc, v)?;
            let c = coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(|| RBall::zero(self.wp));
            acc = self.add(&acc, &self.constant(c))?;
        }
        let (_, v1, l1) = self.v_bounds(v);
        if v1 > f64::NEG_INFINITY {
            let n = jj as f64 + 1.0;
            acc.le = lsum(acc.le, tail + n * v1);
            acc.l = acc.l.max(l1 * (jj + 1));
            if acc.l > MAX_REM_LOG_POWER {
                return None;
            }
        }
        Some(acc)
    }

    /// `V < 1` (`sup` its `log₂`); records a failure in `small_n`.
    fn below_one(&self, sup: f64, limit: f64) -> bool {
        let ok = sup < limit;
        if !ok {
            self.small_n.set(true);
        }
        ok
    }

    /// `1/a`.
    fn recip(&self, a: &Model) -> Option<Model> {
        let (c0, p, v) = self.split_lead(a)?;
        let (sup, _, _) = self.v_bounds(&v);
        if !self.below_one(sup, -1e-3) {
            return None;
        }
        let coeffs: Vec<RBall> = (0..=self.j)
            .map(|i| RBall::from_int(if i.is_multiple_of(2) { 1 } else { -1 }, self.wp))
            .collect();
        let tail = -(1.0 - sup.exp2()).log2();
        let s = self.compose(&v, &coeffs, tail)?;
        // ·c₀⁻¹·ℓ^(−p).
        let mut out = self.ell_shift_down(&self.scale(&s, &c0.recip(self.wp)?), p);
        out.beta = -a.beta.clone();
        out.alt = a.alt;
        Some(out)
    }

    /// `a^n` for an integer `n`.
    fn powi(&self, a: &Model, n: i64) -> Option<Model> {
        if n == 0 {
            return Some(self.constant(RBall::from_int(1, self.wp)));
        }
        let base = if n < 0 { self.recip(a)? } else { a.clone() };
        let mut e = n.unsigned_abs();
        let mut result: Option<Model> = None;
        let mut sq = base;
        while e > 0 {
            if e & 1 == 1 {
                result = Some(match result {
                    None => sq.clone(),
                    Some(r) => self.mul(&r, &sq)?,
                });
            }
            e >>= 1;
            if e > 0 {
                sq = self.mul(&sq, &sq)?;
            }
        }
        result
    }

    /// `a^α` for a non-integer rational `α` (`c₀ > 0`).
    fn pow_q(&self, a: &Model, alpha: &Q, cc: &mut Consts) -> Option<Model> {
        if a.alt {
            return None;
        }
        let (c0, p, v) = self.split_lead(a)?;
        if p != 0 || !c0.certainly_positive() {
            return None;
        }
        let (sup, _, _) = self.v_bounds(&v);
        let af = alpha.to_f64()?;
        if (self.j as f64 + 1.0) <= af.abs() || !self.below_one(sup, -1e-3) {
            return None;
        }
        let mut coeffs = Vec::with_capacity(self.j + 1);
        let mut binom = Q::one();
        for i in 0..=self.j {
            coeffs.push(RBall::from_q(&binom, self.wp));
            binom = binom * (alpha - Q::from_integer(BigInt::from(i)))
                / Q::from_integer(BigInt::from(i + 1));
        }
        // Σ_{i>J} abs(C(α, i))·V^(i−J−1) ≤ abs(C(α, J+1))/(1 − ρV): the ratio
        // abs(C(α, i+1)/C(α, i)) = abs(α − i)/(i + 1) is at most ρ = max(1, (J + 1 +
        // abs(α))/(J + 2)) for i > J (above 1 for α < −1: the coefficients grow).
        let rho =
            ((self.j as f64 + 1.0 + af.abs()) / (self.j as f64 + 2.0)).max(1.0) * (1.0 + 1e-12);
        let vr = sup.exp2() * rho;
        if !self.below_one(vr.log2(), -1e-3) {
            return None;
        }
        let lb = binom.abs().to_f64().map(f64::log2)?;
        let tail = lb - (1.0 - vr).log2();
        let s = self.compose(&v, &coeffs, tail)?;
        // c₀^α = exp(α ln c₀).
        let wp = self.wp;
        let ln_c0 = c0.m.ln(wp + 16, RM, cc);
        let alpha_b = ratio_to_bigfloat(alpha, wp + 16, RM);
        let p = ln_c0.mul(&alpha_b, wp + 16, RM).exp(wp, RM, cc);
        // |Δ ln c₀| ≤ r/(c₀ − r); the result moves by about |α|·that.
        let dl = if c0.r == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            c0.r - c0.lo()
        };
        let r = lsum(
            accuracy::part_lg(&p) + af.abs().max(1e-300).log2() + dl + 1.0,
            two_ulps(&p, wp) + 1.0,
        );
        let mut out = self.scale(&s, &RBall { m: p, r });
        out.beta = &a.beta * alpha;
        Some(out)
    }

    /// `ln a` (`c₀ > 0`).
    fn ln(&self, a: &Model, cc: &mut Consts) -> Option<Model> {
        if a.alt {
            return None;
        }
        let (c0, p, v) = self.split_lead(a)?;
        if p != 0 || !c0.certainly_positive() {
            return None;
        }
        let (sup, _, _) = self.v_bounds(&v);
        if !self.below_one(sup, -1e-3) {
            return None;
        }
        let mut coeffs = vec![RBall::zero(self.wp)];
        for i in 1..=self.j {
            let q = Q::new(
                BigInt::from(if !i.is_multiple_of(2) { 1 } else { -1 }),
                BigInt::from(i),
            );
            coeffs.push(RBall::from_q(&q, self.wp));
        }
        let tail = -((self.j as f64 + 1.0).log2()) - (1.0 - sup.exp2()).log2();
        let s = self.compose(&v, &coeffs, tail)?;
        let wp = self.wp;
        let lc = c0.m.ln(wp, RM, cc);
        let dl = if c0.r == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            c0.r - c0.lo()
        };
        let r = lsum(dl, two_ulps(&lc, wp));
        let mut out = self.add(&s, &self.constant(RBall { m: lc, r }))?;
        if !a.beta.is_zero() {
            let mut lterm = self.zero();
            lterm.c[0] = vec![RBall::zero(wp), RBall::from_q(&-a.beta.clone(), wp)];
            out = self.add(&out, &lterm)?;
        }
        Some(out)
    }

    /// `a` as `c₀ + v` with `β = 0` (`a` bounded: an integer `β ≥ 0`, no `ℓ`
    /// beside the constant): `(c₀, v)`.
    fn split_const(&self, a: &Model) -> Option<(RBall, Model)> {
        if a.alt {
            return None;
        }
        let a0 = self.lower_beta(a, &Q::zero())?;
        if a0.c[0]
            .iter()
            .enumerate()
            .any(|(i, b)| i != a0.off && !b.exact_zero())
        {
            return None;
        }
        let c0 = a0.c[0]
            .get(a0.off)
            .cloned()
            .unwrap_or_else(|| RBall::zero(self.wp));
        let mut v = a0;
        v.c[0] = Vec::new();
        Some((c0, v))
    }

    fn exp(&self, a: &Model, cc: &mut Consts) -> Option<Model> {
        let (c0, v) = self.split_const(a)?;
        let (sup, _, _) = self.v_bounds(&v);
        if !self.below_one(sup, 4.0) {
            return None;
        }
        let mut coeffs = Vec::with_capacity(self.j + 1);
        let mut f = Q::one();
        for i in 0..=self.j {
            coeffs.push(RBall::from_q(&f, self.wp));
            f /= Q::from_integer(BigInt::from(i + 1));
        }
        let tail = sup.exp2() * LOG2E - log2_factorial(self.j + 1);
        let s = self.compose(&v, &coeffs, tail)?;
        let wp = self.wp;
        let e = c0.m.exp(wp, RM, cc);
        let r = lsum(
            accuracy::part_lg(&e)
                + if c0.r == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    c0.r + 1.0
                },
            two_ulps(&e, wp),
        );
        Some(self.scale(&s, &RBall { m: e, r }))
    }

    /// `sin a` (`cos = false`) or `cos a`.
    fn sin_cos(&self, a: &Model, cos: bool, cc: &mut Consts) -> Option<Model> {
        let (c0, v) = self.split_const(a)?;
        let (sup, _, _) = self.v_bounds(&v);
        if !self.below_one(sup, 4.0) {
            return None;
        }
        let wp = self.wp;
        let mut sin_c = Vec::with_capacity(self.j + 1);
        let mut cos_c = Vec::with_capacity(self.j + 1);
        let mut f = Q::one();
        for i in 0..=self.j {
            let sign = if (i / 2).is_multiple_of(2) {
                f.clone()
            } else {
                -f.clone()
            };
            let (s, c) = if !i.is_multiple_of(2) {
                (RBall::from_q(&sign, wp), RBall::zero(wp))
            } else {
                (RBall::zero(wp), RBall::from_q(&sign, wp))
            };
            sin_c.push(s);
            cos_c.push(c);
            f /= Q::from_integer(BigInt::from(i + 1));
        }
        let tail = -log2_factorial(self.j + 1);
        let sv = self.compose(&v, &sin_c, tail)?;
        let cv = self.compose(&v, &cos_c, tail)?;
        let sc = c0.m.sin(wp, RM, cc);
        let ccv = c0.m.cos(wp, RM, cc);
        let ball = |x: BigFloat| {
            let r = lsum(c0.r, two_ulps(&x, wp));
            RBall { m: x, r }
        };
        let (sb, cb) = (ball(sc), ball(ccv));
        if cos {
            // cos(c₀ + v) = cos c₀ cos v − sin c₀ sin v.
            self.add(&self.scale(&cv, &cb), &self.scale(&sv, &sb.neg()))
        } else {
            self.add(&self.scale(&cv, &sb), &self.scale(&sv, &cb))
        }
    }

    /// `atan` of a model `v → 0` (`β = 0`, no constant term), `V < 1`.
    fn atan_small(&self, v: &Model) -> Option<Model> {
        let (sup, _, _) = self.v_bounds(v);
        if !self.below_one(sup, -1e-3) {
            return None;
        }
        let wp = self.wp;
        let coeffs: Vec<RBall> = (0..=self.j)
            .map(|i| {
                if i.is_multiple_of(2) {
                    RBall::zero(wp)
                } else {
                    let sign = if (i / 2).is_multiple_of(2) { 1 } else { -1 };
                    RBall::from_q(&Q::new(BigInt::from(sign), BigInt::from(i)), wp)
                }
            })
            .collect();
        let tail = -((self.j as f64 + 1.0).log2());
        self.compose(v, &coeffs, tail)
    }

    fn atan(&self, a: &Model, cc: &mut Consts) -> Option<Model> {
        if a.alt {
            return None;
        }
        let wp = self.wp;
        if a.beta.is_negative() {
            // g → +∞ (c₀ > 0): π/2 − atan(1/g).
            let (c0, p, _) = self.split_lead(a)?;
            if p != 0 || !c0.certainly_positive() {
                return None;
            }
            let y = self.recip(a)?;
            let (z, v) = self.split_const(&y)?;
            if !z.exact_zero() {
                return None;
            }
            let t = self.atan_small(&v)?;
            let pi = cc.pi(wp, RM).clone();
            let half = pi.div(&BigFloat::from_i32(2, 64), wp, RM);
            let r = two_ulps(&half, wp);
            return self.add(
                &self.constant(RBall { m: half, r }),
                &self.scale(&t, &RBall::from_int(-1, wp)),
            );
        }
        let (c0, v) = self.split_const(a)?;
        if c0.exact_zero() {
            return self.atan_small(&v);
        }
        // atan(c₀ + v) = atan c₀ + atan(v/(1 + c₀² + c₀v)).
        let one = RBall::from_int(1, wp);
        let d0 = one.add(&c0.mul(&c0, wp), wp);
        let den = self.add(&self.constant(d0), &self.scale(&v, &c0))?;
        let y = self.mul(&v, &self.recip(&den)?)?;
        let (z, w) = self.split_const(&y)?;
        if !z.exact_zero() {
            return None;
        }
        let t = self.atan_small(&w)?;
        let at = c0.m.atan(wp, RM, cc);
        let r = lsum(c0.r, two_ulps(&at, wp));
        self.add(&self.constant(RBall { m: at, r }), &t)
    }

    /// `harmonic(a)` for `a → +∞` (real, `c₀ > 0`, `β < 0` an integer).
    fn harmonic(&self, a: &Model, gamma: &RBall, cc: &mut Consts) -> Option<Model> {
        if a.alt || !a.beta.is_negative() || !a.beta.is_integer() {
            return None;
        }
        let (c0, p, _) = self.split_lead(a)?;
        if p != 0 || !c0.certainly_positive() {
            return None;
        }
        let wp = self.wp;
        let y = self.recip(a)?; // β > 0
        let mut out = self.ln(a, cc)?;
        out = self.add(&out, &self.constant(gamma.clone()))?;
        out = self.add(
            &out,
            &self.scale(&y, &RBall::from_q(&Q::new(1.into(), 2.into()), wp)),
        )?;
        // −Σ_{i≤I} B₂ᵢ/(2i)·y^(2i), with 2I·β_y ≥ J + 1.
        let by = y.beta.to_integer().to_usize()?.max(1);
        let big_i = (self.j + 1).div_ceil(2 * by).max(1);
        let y2 = self.mul(&y, &y)?;
        let mut pw = y2.clone();
        for i in 1..=big_i {
            if i > 1 {
                pw = self.mul(&pw, &y2)?;
            }
            let b = super::bernoulli::even(i) / Q::from_integer(BigInt::from(2 * i));
            out = self.add(&out, &self.scale(&pw, &RBall::from_q(&-b, wp)))?;
        }
        // |R| ≤ |B₂ᵢ|/(2I)·y^(2I) with y ≤ Y·u^β_y, Y = sup over x ≥ N.
        let b = super::bernoulli::even(big_i) / Q::from_integer(BigInt::from(2 * big_i));
        let lb = b.abs().to_f64().map(f64::log2)?;
        let (ysup, _, _) = self.v_bounds(&{
            let mut t = y.clone();
            t.beta = Q::zero();
            t
        });
        let pow = 2 * big_i * by; // u-order of the remainder
        if pow < self.j + 1 {
            return None;
        }
        out.le = lsum(
            out.le,
            lb + 2.0 * big_i as f64 * ysup - (pow - self.j - 1) as f64 * self.lg_n,
        );
        Some(out)
    }
}

/// `log₂ n!`.
fn log2_factorial(n: usize) -> f64 {
    (2..=n).map(|k| (k as f64).log2()).sum()
}

// ── The summand as a model ────────────────────────────────────────────────

/// Why a model could not be built.
enum Fail {
    /// The summand is not of the class (or not real).
    Unsupported,
    /// A composition needs a larger `N` (`V ≥ 1`).
    SmallN,
}

/// The summand `body` as a model at `N` (`ctx`).
#[allow(clippy::too_many_arguments)]
fn build(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    ctx: &Ctx,
    gamma: &RBall,
    part: &mut PartFn<'_>,
    cc: &mut Consts,
) -> Result<Model, Fail> {
    let order = walk::post_order_ids(arena, body);
    let mut dep: FxHashMap<ExprId, bool> = FxHashMap::default();
    for &id in &order {
        let d = id == var
            || arena
                .node(id)
                .children()
                .iter()
                .any(|c| dep.get(c) == Some(&true));
        dep.insert(id, d);
    }
    let mut models: FxHashMap<ExprId, Model> = FxHashMap::default();
    let small = |o: Option<Model>| {
        o.ok_or(if ctx.small_n.get() {
            Fail::SmallN
        } else {
            Fail::Unsupported
        })
    };
    for &id in &order {
        let m = if id == var {
            ctx.var()
        } else if dep.get(&id) != Some(&true) {
            if let Some(q) = arena.as_num(id) {
                ctx.constant(RBall::from_q(q, ctx.wp))
            } else {
                let CBall { mid: v, err: b } =
                    part(id, lo, ctx.wp).map_err(|_| Fail::Unsupported)?;
                if !accuracy::exactly_real(&v, b) || b.is_unknown() {
                    return Err(Fail::Unsupported);
                }
                ctx.constant(RBall { m: v.0, r: b.re })
            }
        } else {
            let get = |c: &ExprId| models.get(c).ok_or(Fail::Unsupported);
            match arena.node(id) {
                ExprNode::Add(children) => {
                    let mut acc = ctx.zero();
                    for c in children.iter() {
                        acc = ctx.add(&acc, get(c)?).ok_or(Fail::Unsupported)?;
                    }
                    acc
                }
                ExprNode::Mul(children) => {
                    let mut acc = ctx.constant(RBall::from_int(1, ctx.wp));
                    for c in children.iter() {
                        acc = small(ctx.mul(&acc, get(c)?))?;
                    }
                    acc
                }
                ExprNode::Neg(c) => ctx.scale(get(c)?, &RBall::from_int(-1, ctx.wp)),
                ExprNode::Pow(b, e) => {
                    if let Some(q) = arena.as_num(*e) {
                        let base = get(b)?;
                        if q.is_integer() {
                            let n = q
                                .to_integer()
                                .to_i64()
                                .filter(|n| n.abs() <= 64)
                                .ok_or(Fail::Unsupported)?;
                            small(ctx.powi(base, n))?
                        } else {
                            small(ctx.pow_q(base, q, cc))?
                        }
                    } else if arena.as_num(*b).is_some_and(|q| *q == -Q::one()) {
                        // (−1)^(s·k + c) for integers s, c.
                        alternation(arena, *e, var, ctx).ok_or(Fail::Unsupported)?
                    } else {
                        return Err(Fail::Unsupported);
                    }
                }
                ExprNode::Ln(c) => small(ctx.ln(get(c)?, cc))?,
                ExprNode::Exp(c) => small(ctx.exp(get(c)?, cc))?,
                ExprNode::Sin(c) => small(ctx.sin_cos(get(c)?, false, cc))?,
                ExprNode::Cos(c) => small(ctx.sin_cos(get(c)?, true, cc))?,
                ExprNode::Atan(c) => small(ctx.atan(get(c)?, cc))?,
                ExprNode::Lib(LibFn::Harmonic, args) if args.len() == 1 => {
                    small(ctx.harmonic(get(&args[0])?, gamma, cc))?
                }
                _ => return Err(Fail::Unsupported),
            }
        };
        models.insert(id, m);
    }
    models.remove(&body).ok_or(Fail::Unsupported)
}

/// `(−1)^e` for `e = s·k + c` with integers `s`, `c`: the alternating unit
/// (for an odd `s`) times `(−1)^c`.
fn alternation(arena: &Arena, e: ExprId, var: ExprId, ctx: &Ctx) -> Option<Model> {
    let (s, c) = linear_integer(arena, e, var)?;
    let mut m = ctx.constant(RBall::from_int(
        if c.rem_euclid(2) == 0 { 1 } else { -1 },
        ctx.wp,
    ));
    m.alt = s.rem_euclid(2) == 1;
    Some(m)
}

/// `e` as `s·k + c` with integers `s`, `c` (`k` the index `var`).
fn linear_integer(arena: &Arena, e: ExprId, var: ExprId) -> Option<(i64, i64)> {
    if e == var {
        return Some((1, 0));
    }
    let int = |id: ExprId| -> Option<i64> {
        let q = arena.as_num(id)?;
        if q.is_integer() {
            q.to_integer().to_i64()
        } else {
            None
        }
    };
    match arena.node(e) {
        ExprNode::Add(children) => {
            let (mut s, mut c) = (0i64, 0i64);
            for &t in children.iter() {
                if let Some(n) = int(t) {
                    c = c.checked_add(n)?;
                } else {
                    let (s1, c1) = linear_term(arena, t, var, int)?;
                    s = s.checked_add(s1)?;
                    c = c.checked_add(c1)?;
                }
            }
            Some((s, c))
        }
        _ => linear_term(arena, e, var, int),
    }
}

fn linear_term(
    arena: &Arena,
    t: ExprId,
    var: ExprId,
    int: impl Fn(ExprId) -> Option<i64>,
) -> Option<(i64, i64)> {
    if t == var {
        return Some((1, 0));
    }
    match arena.node(t) {
        ExprNode::Mul(children) if children.len() == 2 => {
            let (a, b) = (children[0], children[1]);
            if b == var {
                Some((int(a)?, 0))
            } else if a == var {
                Some((int(b)?, 0))
            } else {
                None
            }
        }
        ExprNode::Neg(c) if *c == var => Some((-1, 0)),
        _ => None,
    }
}

// ── Power sums ─────────────────────────────────────────────────────────────

/// `log₂ ∫_N^∞ t^(−q)(ln t)^i dt = N^(1−q)·Σ_{r≤i} i!/(i−r)!·(ln N)^(i−r)/(q − 1)^(r+1)`, `q > 1`.
fn log2_power_log_integral(q: f64, i: usize, lg_n: f64, ln_n: f64) -> f64 {
    let mut acc = f64::NEG_INFINITY;
    let mut fall = 0.0f64; // log₂ i!/(i−r)!
    for r in 0..=i {
        if r > 0 {
            fall += ((i - r + 1) as f64).log2();
        }
        let t = fall + (i - r) as f64 * ln_n.log2() - (r as f64 + 1.0) * (q - 1.0).log2();
        acc = lsum(acc, t);
    }
    (1.0 - q) * lg_n + acc
}

/// `Σ_{k≥N} k^(−s)(ln k)^e` (or with `(−1)^(k−N)` when `alt`) for a real `s`
/// and an integer `e` (`s > 1` or `s = 1 > e + 1`; alternating `s > 0` or
/// `s = 0 > e`; `sf` the `f64` value of `s`): the value and `log₂` of its
/// error bound (see the module documentation).  The corrections stop once
/// the remainder is below `2^−wp` of the largest term or below `2^floor`;
/// when the corrections start to grow first, the smallest remainder reached
/// is the bound.  For `e < 0` the integral is `(s − 1)^(−e−1)·Γ(e + 1, x)`
/// with `x = (s − 1) ln N` (`Γ(0, x) = E₁(x)`, `Γ(a, x) = (Γ(a + 1, x) − x^a
/// e^(−x))/a`), or `(ln N)^(e+1)/(−e − 1)` for `s = 1`, and the remainder
/// takes `∫_N^∞ t^(−q)(ln t)^(−i) dt ≤ (ln N)^(−i)·N^(1−q)/(q − 1)`.
#[allow(clippy::too_many_arguments)]
fn power_log_tail(
    s: &BigFloat,
    sf: f64,
    e: i64,
    n: u64,
    alt: bool,
    floor: f64,
    wpc: &mut Wp<'_>,
) -> Option<RBall> {
    let wp = wpc.bits();
    let p = wp + 32;
    let nb = BigFloat::from_u64(n, 64);
    let ln_n = nb.ln(p, RM, wpc.cc);
    let inv_ln = BigFloat::from_i32(1, 64).div(&ln_n, p, RM);
    let lnf = (n as f64).ln();
    let lg_n = (n as f64).log2();
    // (ln N)^k for an integer k.
    let lpow = |k: i64| -> BigFloat {
        let b = if k >= 0 { &ln_n } else { &inv_ln };
        b.powi(k.unsigned_abs() as usize, p, RM)
    };
    // N^(−s).
    let nps = ln_n.mul(s, p, RM).neg().exp(p, RM, wpc.cc);
    let one = BigFloat::from_i32(1, 64);
    let mut value = BigFloat::new(p);
    if !alt {
        let integral = if e >= 0 {
            // N^(1−s)·Σ_r e!/(e−r)!·(ln N)^(e−r)/(s−1)^(r+1).
            let m = e as usize;
            let s1 = s.sub(&one, p, RM);
            let mut acc = BigFloat::new(p);
            let mut fall = BigFloat::from_i32(1, p);
            let mut den = s1.clone();
            for r in 0..=m {
                if r > 0 {
                    fall = fall.mul(&BigFloat::from_u64((m - r + 1) as u64, 64), p, RM);
                    den = den.mul(&s1, p, RM);
                }
                acc = acc.add(
                    &fall.mul(&lpow((m - r) as i64), p, RM).div(&den, p, RM),
                    p,
                    RM,
                );
            }
            acc.mul(&nps, p, RM).mul(&nb, p, RM)
        } else if *s == BigFloat::from_i32(1, 64) {
            if e >= -1 {
                return None;
            }
            lpow(e + 1).div(&BigFloat::from_i64(-e - 1, 64), p, RM)
        } else {
            // The recurrence cancels about log₂(x/k) bits at step k: as many
            // more.
            let steps = (-e - 1) as usize;
            let xf = (sf - 1.0) * lnf;
            let pg = p + steps * (super::magnitude_bits(xf.max(1.0)) + 2);
            let s1 = s.sub(&one, pg, RM);
            let x = s1.mul(&ln_n, pg, RM);
            let mut g = super::arb_expint(&BigFloat::from_i32(1, 64), &x, &mut wpc.at(pg)).ok()?;
            let ex = x.neg().exp(pg, RM, wpc.cc);
            let inv_x = BigFloat::from_i32(1, 64).div(&x, pg, RM);
            let mut xk = BigFloat::from_i32(1, pg); // x^(−k)
            for k in 1..=steps {
                xk = xk.mul(&inv_x, pg, RM);
                g = xk.mul(&ex, pg, RM).sub(&g, pg, RM).div(
                    &BigFloat::from_u64(k as u64, 64),
                    pg,
                    RM,
                );
            }
            super::round_to(s1.powi(steps, pg, RM).mul(&g, pg, RM), p)
        };
        value = value.add(&integral, p, RM);
    }
    // f(N)/2.
    let f_n = nps.mul(&lpow(e), p, RM);
    value = value.add(&f_n.div(&BigFloat::from_i32(2, 64), p, RM), p, RM);
    let mut largest = accuracy::part_lg(&value).max(accuracy::part_lg(&f_n));
    // The derivatives f^(k)(N) = N^(−s−k)·Σ_d coef[d]·(ln N)^(e−d):
    // d/dt t^(−a)(ln t)^q = t^(−a−1)·(−a(ln t)^q + q(ln t)^(q−1)).
    let mut coef: Vec<BigFloat> = vec![BigFloat::from_i32(1, p)];
    let inv_n = BigFloat::from_i32(1, 64).div(&nb, p, RM);
    let mut npow = nps.clone(); // N^(−s−order)
    let mut fact = BigFloat::from_i32(1, p); // (order)!
    let max_m = 4 * wp + 64;
    let mut best: Option<RBall> = None;
    for order in 0..(2 * max_m) {
        let so = s.add(&BigFloat::from_u64(order as u64, 64), p, RM);
        let mut next = vec![BigFloat::new(p); coef.len() + 1];
        for (d, c) in coef.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            next[d] = next[d].sub(&c.mul(&so, p, RM), p, RM);
            let q = e - d as i64;
            if q != 0 {
                next[d + 1] = next[d + 1].add(&c.mul(&BigFloat::from_i64(q, 64), p, RM), p, RM);
            }
        }
        while next.len() > 1 && next.last().is_some_and(BigFloat::is_zero) {
            next.pop();
        }
        coef = next;
        npow = npow.mul(&inv_n, p, RM);
        let k = order + 1; // the order of `coef`
        fact = fact.mul(&BigFloat::from_u64(k as u64, 64), p, RM);
        if !k.is_multiple_of(2) {
            let j = k.div_ceil(2);
            // −B₂ⱼ/(2j)!·f^(2j−1)(N) (times 2^(2j) − 1 alternating).
            let mut d = BigFloat::new(p);
            for (i, c) in coef.iter().enumerate() {
                d = d.add(&c.mul(&lpow(e - i as i64), p, RM), p, RM);
            }
            let deriv = d.mul(&npow, p, RM);
            let b = ratio_to_bigfloat(&super::bernoulli::even(j), p, RM);
            let fact2j = fact.mul(&BigFloat::from_u64((k + 1) as u64, 64), p, RM);
            let mut t = b.mul(&deriv, p, RM).div(&fact2j, p, RM).neg();
            if alt {
                let w = BigFloat::from_i32(2, 64)
                    .powi(2 * j, p, RM)
                    .sub(&one, p, RM);
                t = t.mul(&w, p, RM);
            }
            largest = largest.max(accuracy::part_lg(&t));
            value = value.add(&t, p, RM);
        } else {
            // The remainder after j = k/2 corrections: f^(2j).
            let j = k / 2;
            let q = sf + k as f64;
            if q <= 1.0 {
                continue;
            }
            let mut ib = f64::NEG_INFINITY;
            for (i, c) in coef.iter().enumerate() {
                if c.is_zero() {
                    continue;
                }
                let pw = e - i as i64;
                let int = if pw >= 0 {
                    log2_power_log_integral(q, pw as usize, lg_n, lnf)
                } else {
                    pw as f64 * lnf.log2() + (1.0 - q) * lg_n - (q - 1.0).log2()
                };
                ib = lsum(ib, accuracy::part_lg(c) + int);
            }
            // |B₂ⱼ|/(2j)! ≤ 2ζ(2)/(2π)^(2j) < 3.3·(2π)^(−2j).
            let lb = 3.3f64.log2() - 2.0 * j as f64 * LOG2_TWO_PI;
            let mult = if alt { lsum(2.0 * j as f64, 0.0) } else { 0.0 };
            let rem = lb + mult + ib;
            let err = lsum(rem, largest - p as f64 + 8.0);
            if rem < largest - wp as f64 - 8.0 || rem < floor {
                return Some(RBall {
                    m: super::round_to(value, wp),
                    r: err,
                });
            }
            match &best {
                Some(b) if b.r + 24.0 < err => break,
                Some(b) if b.r <= err => {}
                _ => {
                    best = Some(RBall {
                        m: value.clone(),
                        r: err,
                    })
                }
            }
            if k > 2 * max_m - 2 {
                break;
            }
        }
    }
    best.map(|b| RBall {
        m: super::round_to(b.m, wp),
        r: b.r,
    })
}

// ── The sum ────────────────────────────────────────────────────────────────

/// `log₂` of a bound on `Σ_{k≥N} k^(−q)(1 + ln k)^L`, `q > 1`.
fn log2_remainder_sum(q: f64, l: usize, n: f64) -> f64 {
    let ln_n = n.ln();
    let lg_n = n.log2();
    let t = 1.0 + ln_n;
    let lf = l as f64;
    // g at N or at its maximum (1 + ln x = L/q).
    let gmax = if q * t >= lf {
        -q * lg_n + lf * t.log2()
    } else {
        // x* = e^(L/q − 1): x*^(−q) = e^(q − L).
        (q - lf) * LOG2E + lf * (lf / q).log2()
    };
    // ∫_N^∞ g = N^(1−q)·Σ_{r≤L} L!/r!·T^r/(q − 1)^(L+1−r).
    let mut acc = f64::NEG_INFINITY;
    let mut lfr = 0.0f64; // log₂ L!/r!
    for r in (0..=l).rev() {
        if r < l {
            lfr += ((r + 1) as f64).log2();
        }
        acc = lsum(
            acc,
            lfr + r as f64 * t.log2() - (lf + 1.0 - r as f64) * (q - 1.0).log2(),
        );
    }
    lsum(gmax, (1.0 - q) * lg_n + acc) + 1e-9
}

/// `Σ_{k ≥ lo} body` by the expansion of the summand at infinity (see the
/// module documentation): `None` when the summand is not of the class.
#[allow(clippy::too_many_arguments)]
pub(super) fn asymptotic_sum(
    arena: &Arena,
    body: ExprId,
    var: ExprId,
    lo: &BigInt,
    prec: usize,
    part: &mut PartFn<'_>,
) -> Option<Result<CBall, SymplexError>> {
    if !supported_shape(arena, body, var) {
        return None;
    }
    // (`part` holds the evaluator's constants.)
    let mut consts = Consts::new().ok()?;
    let cc = &mut consts;
    let wp = prec + 32;
    let g = super::arb_euler_gamma(&mut Wp::new(wp, cc)).ok()?;
    let gr = two_ulps(&g, wp);
    let gamma = RBall { m: g, r: gr };
    let start = lo.to_i64()?;
    // N = 64 first (the terms below it are evaluated one by one, at the cost
    // of a full evaluation each: `harmonic(k)` takes milliseconds), larger
    // when a function of the model needs it or the remainder is too large.
    let mut n: u64 = (start.max(1) as u64).next_power_of_two().max(64);
    if log_in_denominator(arena, body, var) {
        // Negative powers of ℓ make the model's rows long (model arithmetic
        // costs about J⁴ then): fewer orders, more terms summed directly.
        n = n.max(1024);
    }
    let mut direct = super::SumAcc::new(prec);
    let mut k = lo.clone();
    let mut attempts = 0;
    loop {
        attempts += 1;
        if n > MAX_N || attempts > 6 {
            return Some(Err(super::special_exhausted(prec)));
        }
        let mut lg_n = (n as f64).log2();
        let mut j = ((wp as f64 + 24.0) / lg_n).ceil() as usize + 4;
        if j > MAX_ORDER {
            // Model arithmetic costs J³: a larger N instead.
            let bits = ((wp as f64 + 24.0) / (MAX_ORDER - 4) as f64).ceil();
            if bits > (MAX_N as f64).log2() {
                return Some(Err(super::special_exhausted(prec)));
            }
            n = n.max(bits.exp2() as u64);
            lg_n = (n as f64).log2();
            j = ((wp as f64 + 24.0) / lg_n).ceil() as usize + 4;
        }
        let ctx = Ctx {
            small_n: std::cell::Cell::new(false),
            j,
            lg_n,
            ln_n: (n as f64).ln(),
            wp,
        };
        let model = match build(arena, body, var, lo, &ctx, &gamma, part, cc) {
            Ok(m) => m,
            Err(Fail::Unsupported) => return None,
            Err(Fail::SmallN) => {
                n *= 4;
                continue;
            }
        };
        // The terms below N.
        let n_big = BigInt::from(n);
        while k < n_big {
            match part(body, &k, prec) {
                Ok(CBall { mid: t, err: e }) => direct.add(&t, e),
                Err(e) => return Some(Err(e)),
            }
            k += 1;
        }
        match tail(&model, &ctx, n, cc) {
            TailResult::Done(RBall { m: value, r: err }) => {
                let target = accuracy::part_lg(&value)
                    .max(accuracy::lg_abs(&direct.value))
                    .max(-1e6)
                    - prec as f64
                    - 4.0;
                if err <= target || (attempts >= 4 && err.is_finite()) {
                    let mut out = std::mem::replace(&mut direct, super::SumAcc::new(prec));
                    let v = (super::round_to(value, prec), BigFloat::new(prec));
                    let e = lsum(err, accuracy::rounding(&v, prec));
                    out.add(&v, Bound::real(e));
                    tracing::debug!(n, j, err, "evalf: sum by its expansion at infinity");
                    return Some(Ok(out.finish()));
                }
                n *= 4;
            }
            TailResult::Divergent => {
                return Some(Err(SymplexError::Divergent {
                    operation: "evalf",
                    reason: format!(
                        "the infinite sum of `{}` diverges (its terms decay like k^(-s), s <= 1)",
                        arena.display(body)
                    ),
                }));
            }
            TailResult::Undecided => {
                return Some(Err(super::unevaluable(format!(
                    "infinite sum of `{}`: the leading coefficient of its expansion is only \
                     known to be within its error of 0, so convergence is undecided",
                    arena.display(body)
                ))));
            }
            TailResult::Failed => n *= 4,
        }
    }
}

enum TailResult {
    Done(RBall),
    Divergent,
    Undecided,
    Failed,
}

/// `Σ_{k≥N} model(k)`: the power sums times the coefficients, plus the sum
/// of the remainder.
fn tail(model: &Model, ctx: &Ctx, n: u64, cc: &mut Consts) -> TailResult {
    let wp = ctx.wp;
    let p = wp + 16;
    let min_s = if model.alt { 0.0 } else { 1.0 };
    // Σ k^(−s)(ln k)^e converges for s above `min_s`, and at `min_s` when
    // `e < −1` (`e < 0` alternating).
    let max_e_at_min = if model.alt { -1 } else { -2 };
    let mut value = BigFloat::new(p);
    let mut err = f64::NEG_INFINITY;
    // An estimate of the largest contribution, `abs(c)·N^(1−s)(ln N)^m` (the
    // power sums need no more than `2^−wp` of it in absolute terms).
    let lln = ctx.ln_n.log2();
    let mut lead = f64::NEG_INFINITY;
    for (j, row) in model.c.iter().enumerate() {
        let sf = (&model.beta + Q::from_integer(BigInt::from(j)))
            .to_f64()
            .unwrap_or(0.0);
        for (m, c) in row.iter().enumerate() {
            if !c.exact_zero() {
                let e = model.pow_at(m) as f64;
                let (size, e) = match (model.alt, sf <= min_s + 1e-12) {
                    (true, _) => (-sf, e),
                    (false, false) => (1.0 - sf, e),
                    // Σ_{k≥N} k^(−1)(ln k)^e ≈ (ln N)^(e+1)/(−e − 1).
                    (false, true) => (0.0, e + 1.0),
                };
                lead = lead.max(c.hi() + size * ctx.lg_n + e * lln);
            }
        }
    }
    let floor = lead - wp as f64 - 12.0;
    for (j, row) in model.c.iter().enumerate() {
        let s_q = &model.beta + Q::from_integer(BigInt::from(j));
        let sf = match s_q.to_f64() {
            Some(v) => v,
            None => return TailResult::Failed,
        };
        for (m, c) in row.iter().enumerate() {
            if c.exact_zero() {
                continue;
            }
            let e = model.pow_at(m);
            let min_q = Q::from_integer(BigInt::from(if model.alt { 0 } else { 1 }));
            if s_q < min_q || (s_q == min_q && e > max_e_at_min) {
                return if c.certainly_nonzero() {
                    TailResult::Divergent
                } else {
                    TailResult::Undecided
                };
            }
            let sb = ratio_to_bigfloat(&s_q, p, RM);
            let Some(RBall { m: z, r: ez }) = power_log_tail(
                &sb,
                sf,
                e,
                n,
                model.alt,
                floor - c.hi(),
                &mut Wp::new(wp, cc),
            ) else {
                return TailResult::Failed;
            };
            // The rounding of s moves the sum by |Δs|·Σ k^(−s)(ln k)^(m+1).
            let es = if s_q.denom().is_one() {
                f64::NEG_INFINITY
            } else {
                two_ulps(&sb, p) + accuracy::part_lg(&z) + (ctx.ln_n + 1.0).log2() + 1.0
            };
            let t = c.m.mul(&z, p, RM);
            value = value.add(&t, p, RM);
            err = lsum(
                err,
                lsum(c.hi() + lsum(ez, es), c.r + accuracy::part_lg(&z)),
            );
            err = lsum(err, ulp(&t, p) + 1.0);
        }
    }
    if model.le > f64::NEG_INFINITY {
        let q: f64 = match (&model.beta + Q::from_integer(BigInt::from(ctx.j + 1))).to_f64() {
            Some(v) => v,
            None => return TailResult::Failed,
        };
        if q <= 1.0 {
            return TailResult::Failed;
        }
        err = lsum(err, model.le + log2_remainder_sum(q, model.l, n as f64));
    }
    if model.alt && !n.is_multiple_of(2) {
        value = value.neg();
    }
    if !err.is_finite() && err != f64::NEG_INFINITY {
        return TailResult::Failed;
    }
    TailResult::Done(RBall { m: value, r: err })
}

/// Does a logarithm of the index appear under a negative power (`1/ln(k +
/// c)`, which brings negative powers of `ℓ` into the model)?
fn log_in_denominator(arena: &Arena, body: ExprId, var: ExprId) -> bool {
    walk::post_order_ids(arena, body).into_iter().any(|id| {
        matches!(arena.node(id), ExprNode::Pow(b, e)
        if arena.as_num(*e).is_some_and(Signed::is_negative)
            && walk::post_order_ids(arena, *b).into_iter().any(|c| {
                matches!(arena.node(c), ExprNode::Ln(a) if walk::contains(arena, *a, var))
            }))
    })
}

/// A cheap structural test: every node that depends on the index is one the
/// models handle (before constants are evaluated).
fn supported_shape(arena: &Arena, body: ExprId, var: ExprId) -> bool {
    let order = walk::post_order_ids(arena, body);
    let mut dep: FxHashMap<ExprId, bool> = FxHashMap::default();
    for &id in &order {
        let d = id == var
            || arena
                .node(id)
                .children()
                .iter()
                .any(|c| dep.get(c) == Some(&true));
        dep.insert(id, d);
        if !d || id == var {
            continue;
        }
        let ok = match arena.node(id) {
            ExprNode::Add(_)
            | ExprNode::Mul(_)
            | ExprNode::Neg(_)
            | ExprNode::Pow(..)
            | ExprNode::Ln(_)
            | ExprNode::Exp(_)
            | ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Atan(_) => true,
            ExprNode::Lib(LibFn::Harmonic, args) => args.len() == 1,
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(s: i32, m: i64, alt: bool, want: &str) {
        let mut cc = Consts::new().unwrap();
        let sb = BigFloat::from_i32(s, 64);
        let RBall { m: v, r: e } = power_log_tail(
            &sb,
            f64::from(s),
            m,
            64,
            alt,
            -1e9,
            &mut Wp::new(160, &mut cc),
        )
        .unwrap();
        let w = BigFloat::parse(want, astro_float::Radix::Dec, 200, RM, &mut cc);
        let d = v.sub(&w, 200, RM);
        // The bound holds, and is small.
        assert!(e < -120.0, "bound 2^{e}");
        // (The rounding to 160 bits, and the reference's 60 digits.)
        let tol = lsum(e, accuracy::part_lg(&w) - 158.0);
        assert!(
            d.is_zero() || accuracy::part_lg(&d) <= tol,
            "{v} vs {want}, bound 2^{e}"
        );
    }

    /// mpmath at 60 digits: `zeta(2, 64)`, `-zeta(2, 64, 1)`, `(zeta(2, 32) -
    /// zeta(2, 32.5))/4` (alternating) and `(-zeta(3,32,1)+zeta(3,32.5,1) +
    /// log(2)*(zeta(3,32)-zeta(3,32.5)))/8` (alternating, with a logarithm).
    #[test]
    fn power_log_tails_match_hurwitz_zeta() {
        check(
            2,
            0,
            false,
            "0.0157477060643389301557440030713504645073005070344386137372834",
        );
        check(
            2,
            1,
            false,
            "0.0811175504948841371389988877705417436920369778981509917402354",
        );
        check(
            2,
            0,
            true,
            "0.000123977195812114907546837148020249063363925032646318980115058",
        );
        check(
            3,
            1,
            true,
            "0.00000810339401971460841448553577145658790973720132177634585547489",
        );
    }

    /// `Σ_{k≥N} k^(−q)(1 + ln k)^L` against a direct sum to 10⁶ plus its tail.
    #[test]
    fn remainder_sum_bounds_the_sum() {
        for &(q, l) in &[(2.0f64, 0usize), (2.5, 3), (1.5, 2), (3.0, 10)] {
            let n = 64.0f64;
            let mut s = 0.0f64;
            let mut k = n;
            while k < 1e6 {
                s += k.powf(-q) * (1.0 + k.ln()).powi(l as i32);
                k += 1.0;
            }
            assert!(s.log2() < log2_remainder_sum(q, l, n), "q {q} L {l}");
        }
    }
}
