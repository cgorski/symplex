//! Bessel functions of large order by the uniform asymptotic (Debye)
//! expansions, with Olver's error bounds.
//!
//! The series of `evalf.rs` and `bessel_order.rs` take a term per unit of
//! `x²/(4ν)` (the ascending series of `J_ν`, `I_ν`) or of the order (the
//! Neumann sums of `Y_n`, `K_n`), and cancel as many bits as their terms
//! grow: before 0.35 `bessely(20000, 300)`, `besselk(20000, 300)`,
//! `bessely(20000, 30000)` and `besseli(1000, 120000)` were refused.  For a
//! large order the expansions of DLMF 10.41 (`ν → ∞` uniformly in `z = x/ν`)
//! take a handful of terms at any `x`:
//!
//! ```text
//! K_ν(νz) = (π/(2ν))^½ e^(−νη) (1 + z²)^(−¼) (Σ_{k<n} (−1)^k U_k(p)/ν^k + ε),
//! I_ν(νz) ∝ e^(νη) (1 + z²)^(−¼) (Σ_{k<n} U_k(p)/ν^k + ε),
//! η = (1 + z²)^½ + ln(z/(1 + (1 + z²)^½)),  p = (1 + z²)^(−½),
//! U_{k+1}(p) = ½p²(1 − p²)U_k′(p) + ⅛∫₀ᵖ (1 − 5t²)U_k(t) dt   (10.41.9)
//! ```
//!
//! with the coefficients of `U_k` exact rationals.  They are the
//! Liouville–Green solutions of the Bessel equation in the variable `νη`
//! (DLMF 2.8(ii), Olver 1997b Ch. 10), whose remainder after `n` terms is
//! at most `2·exp(2𝒱(U₁)/ν)·𝒱(U_n)/νⁿ`, the variations `𝒱` taken along a
//! progressive path from the end where the solution is recessive.  Here:
//!
//! * `K_ν(x)`, `x > 0` (DLMF 10.41.4): recessive at `z = ∞` (`p = 0`, where
//!   `U_k(0) = 0`, so the normalisation is exact); `𝒱_{0,p}(U) ≤ Σ abs(c_j)·p^j`.
//! * `I_ν(x)` (10.41.3): recessive at `z = 0` (`p = 1`), where `U_k(1) ≠ 0`
//!   (the Stirling coefficients): the solution with its error vanishing
//!   there is `ν^ν e^(−ν)/Γ(ν + 1)·e^(νη)(1 + z²)^(−¼)(S_n(p) + ε)/S_n(1)`
//!   exactly, `S_n(p) = Σ_{k<n} U_k(p)/ν^k` (compare `z → 0` with
//!   `(νz/2)^ν/Γ(ν + 1)`), `abs(ε) ≤ 2·exp(2𝒱/ν)·𝒱_{p,1}(U_n)/νⁿ`.
//! * `J_ν(x)`, `0 < x < ν`: `J_ν(x) = e^(−iνπ/2) I_ν(ix)` (10.27.6) along
//!   the imaginary axis, where `p = (1 − z²)^(−½) ∈ [1, ∞)` is real and
//!   `Re η` increases (the path is progressive): the same formula with
//!   `η = (1 − z²)^½ + ln(z/(1 + (1 − z²)^½))` (10.19.3), `𝒱_{1,p}`.
//! * `J_ν(x)`, `Y_ν(x)`, `x > ν`: `H⁽¹⁾_ν(x) = (2/(πi)) e^(−iνπ/2) K_ν(−ix)`
//!   (10.27.8) along the imaginary axis from `−i∞`, where `Re η = 0` and
//!   `p = iν/w` (`w = (x² − ν²)^½`): `H⁽¹⁾_ν(x) = (2/(πw))^½ e^(iξ)
//!   (Σ_k (−i)^k V_k(ν/w)/ν^k + ε)`, `ξ = w − ν·atan(w/ν) − π/4`,
//!   `U_k(iq) = i^k V_k(q)` (10.19.6); `J`, `Y` its real and imaginary parts.
//!
//! The turning point `x = ν` (`p → ∞`) is not covered (DLMF 10.20, Airy
//! functions): the variations grow like `p^(3n)` and the terms stop
//! falling; nor is `Y_ν(x)` for `x < ν`, the solution dominant at `z = 0`,
//! whose error is controlled from the turning point.  `None` there, and the
//! callers refuse as before.
//!
//! The bound is applied to the sum at `n` terms chosen so that it is below
//! `2^(−wp−6)`; the roundings of the sum (Horner's rule on `U_k` at an
//! argument rounded at `wp` bits, running powers of `1/ν`, the additions)
//! are at most `(10k + 10)·2^(−wp)·Σ abs(c_j)·t^j/ν^k` per term.  An
//! oscillating `J` or `Y` near a zero cancels: the bits lost are measured
//! and the evaluation repeated with as many more.

use std::cell::RefCell;

use astro_float::{BigFloat, Consts, RoundingMode};
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::base::errors::SymplexError;

/// The function of [`debye`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    J,
    Y,
    I,
    K,
}

/// The most terms of an expansion (`U_k` of degree up to `3k`).
const MAX_TERMS: usize = 60;

/// The smallest order: below, the expansions reach few bits.
const MIN_ORDER: f64 = 16.0;

/// `U_k`: its nonzero coefficients `(j, c_j)` (`j ≡ k mod 2`, `k ≤ j ≤ 3k`)
/// and upper bounds of `log₂ abs(c_j)`.
struct UPoly {
    coeffs: Vec<(usize, Ratio<BigInt>)>,
    lg: Vec<f64>,
}

thread_local! {
    /// `U_0, U_1, …` as far as asked for.
    static U: RefCell<Vec<UPoly>> = const { RefCell::new(Vec::new()) };
}

/// An upper bound of `log₂ abs(c)` for a nonzero rational.
fn lg_upper(c: &Ratio<BigInt>) -> f64 {
    let lg_int = |n: &BigInt| -> (f64, f64) {
        // (lower, upper) bounds of log₂ abs(n).
        let bits = n.bits() as f64;
        match n.abs().to_f64() {
            Some(f) if f.is_finite() && f > 0.0 => {
                let l = f.log2();
                (l - 1e-9, l + 1e-9)
            }
            _ => (bits - 1.0, bits),
        }
    };
    let (_, num_hi) = lg_int(c.numer());
    let (den_lo, _) = lg_int(c.denom());
    num_hi - den_lo
}

/// `U_{k+1}` from `U_k` by 10.41.9: `c_j p^j` gives
/// `c_j·(j/2 + 1/(8(j + 1)))·p^(j+1)` and `−c_j·(j/2 + 5/(8(j + 3)))·p^(j+3)`.
fn next_poly(u: &UPoly) -> UPoly {
    let top = u.coeffs.iter().map(|(j, _)| *j).max().unwrap_or(0) + 3;
    let mut dense: Vec<Ratio<BigInt>> = vec![Ratio::zero(); top + 1];
    for (j, c) in &u.coeffs {
        let j = *j;
        let jb = BigInt::from(j as u64);
        let a = Ratio::new(
            BigInt::from(4u64) * &jb * BigInt::from(j as u64 + 1) + BigInt::from(1u64),
            BigInt::from(8u64) * BigInt::from(j as u64 + 1),
        );
        let b = Ratio::new(
            BigInt::from(4u64) * &jb * BigInt::from(j as u64 + 3) + BigInt::from(5u64),
            BigInt::from(8u64) * BigInt::from(j as u64 + 3),
        );
        dense[j + 1] += c * a;
        dense[j + 3] -= c * b;
    }
    let coeffs: Vec<(usize, Ratio<BigInt>)> = dense
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .collect();
    let lg = coeffs.iter().map(|(_, c)| lg_upper(c)).collect();
    UPoly { coeffs, lg }
}

/// `f(U_0, …, U_n)`, the polynomials made as needed.
fn with_u<R>(n: usize, f: impl FnOnce(&[UPoly]) -> R) -> R {
    U.with(|cell| {
        let mut us = cell.borrow_mut();
        if us.is_empty() {
            us.push(UPoly {
                coeffs: vec![(0, Ratio::from_integer(BigInt::from(1u64)))],
                lg: vec![0.0],
            });
        }
        while us.len() <= n {
            let next = match us.last() {
                Some(u) => next_poly(u),
                None => break,
            };
            us.push(next);
        }
        f(&us[..=n.min(us.len() - 1)])
    })
}

/// `log₂(2^a + 2^b)` (`−∞` for an empty sum).
fn lg_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp2().ln_1p() / std::f64::consts::LN_2 + 1e-12
}

/// An upper bound of `log₂ Σ abs(c_j)·t^j` for `log₂ t = lg_t`: the
/// variation of `U` along a segment from 0 to `t` (or from 1 to `t ≥ 1`),
/// and the size of its value there.
fn lg_abs_sum(u: &UPoly, lg_t: f64) -> f64 {
    u.coeffs
        .iter()
        .zip(&u.lg)
        .fold(f64::NEG_INFINITY, |acc, ((j, _), lc)| {
            lg_add(acc, lc + (*j as f64) * lg_t)
        })
}

/// The number of terms and what is known of the sum `S = Σ_{k<n} ±U_k(t)/ν^k`
/// for `log₂ t = lg_t`, `log₂ ν = lg_nu`.
struct Plan {
    n: usize,
    /// `log₂` of the bound of the remainder (absolute, on `S`).
    lg_trunc: f64,
    /// A lower bound of `abs(S)`.
    s_low: f64,
    /// An upper bound of `Σ_{k<n} abs(c)·t^j/ν^k`.
    abs_sum: f64,
}

/// The fewest terms whose remainder bound `2·exp(2𝒱(U₁)/ν)·𝒱(U_n)/νⁿ` is
/// below `2^(−wp−6)`, the variations bounded by `Σ abs(c_j)·t^j`; `None` when
/// none up to [`MAX_TERMS`] is, or when the terms after the first add up to
/// more than ½ (the sum is not dominated by 1).
fn plan(lg_t: f64, nu_f: f64, wp: usize) -> Option<Plan> {
    let lg_nu = nu_f.log2();
    // The polynomials are made as far as asked for (the first call of a
    // thread makes `U_1, U_2, …` up to the `k` it needs).
    let lg_abs = |k: usize| with_u(k, |us| us.get(k).map(|u| lg_abs_sum(u, lg_t)));
    let v1 = lg_abs(1)?.exp2();
    if !v1.is_finite() {
        return None;
    }
    let factor = 1.0 + 2.0 * v1 / nu_f * std::f64::consts::LOG2_E;
    let mut tail = 0.0f64;
    for k in 1..=MAX_TERMS {
        let lg_term = lg_abs(k)? - (k as f64) * lg_nu;
        let lg_trunc = factor + lg_term;
        if lg_trunc < -((wp + 6) as f64) {
            let s_low = 1.0 - tail;
            return (s_low >= 0.5).then_some(Plan {
                n: k,
                lg_trunc,
                s_low,
                abs_sum: 1.0 + tail,
            });
        }
        tail += lg_term.exp2();
        if !tail.is_finite() || tail > 0.5 {
            return None;
        }
    }
    None
}

/// `Σ_{k<n} s_k·U_k(t)/ν^k` at `wp` bits, with `s_k = (−1)^k` when
/// `alternating`, else 1.
fn sum_terms(
    n: usize,
    t: &BigFloat,
    nu: &BigFloat,
    alternating: bool,
    wp: usize,
    rm: RoundingMode,
) -> BigFloat {
    with_u(n, |us| {
        let inv_nu = BigFloat::from_i32(1, wp).div(nu, wp, rm);
        let mut scale = BigFloat::from_i32(1, wp);
        let mut sum = BigFloat::new(wp);
        for (k, u) in us.iter().enumerate().take(n) {
            if k > 0 {
                scale = scale.mul(&inv_nu, wp, rm);
            }
            let term = eval_dense(u, k, t, false, wp, rm).mul(&scale, wp, rm);
            sum = if alternating && k % 2 == 1 {
                sum.sub(&term, wp, rm)
            } else {
                sum.add(&term, wp, rm)
            };
        }
        sum
    })
}

/// `U_k(t)` at `wp` bits by Horner's rule in `t²` over the powers `k, k +
/// 2, …, 3k` (a vanishing coefficient is a 0 step); `alt`: `V_k(t) =
/// i^(−k)·U_k(it)`, the coefficient of `t^(k+2m)` with the sign `(−1)^m`.
fn eval_dense(
    u: &UPoly,
    k: usize,
    t: &BigFloat,
    alt: bool,
    wp: usize,
    rm: RoundingMode,
) -> BigFloat {
    let t2 = t.mul(t, wp, rm);
    let mut acc = BigFloat::new(wp);
    let mut idx = u.coeffs.len();
    let top = u.coeffs.last().map_or(k, |(j, _)| *j);
    let mut j = top;
    loop {
        let c = if idx > 0 && u.coeffs[idx - 1].0 == j {
            idx -= 1;
            let m = (j - k) / 2;
            let cf = super::ratio_to_bigfloat(&u.coeffs[idx].1, wp, rm);
            if alt && m % 2 == 1 { cf.neg() } else { cf }
        } else {
            BigFloat::new(wp)
        };
        acc = acc.mul(&t2, wp, rm).add(&c, wp, rm);
        if j < k + 2 {
            break;
        }
        j -= 2;
    }
    acc.mul(&t.powi(k, wp, rm), wp, rm)
}

/// `log₂ abs(x)` of a nonzero finite `x` (`−∞` for 0).
fn lg(x: &BigFloat) -> f64 {
    match x.exponent() {
        Some(e) if !x.is_zero() => {
            let mut m = x.abs();
            m.set_exponent(0);
            let mf = super::bigfloat_to_f64_rounded(&m, RoundingMode::ToEven).unwrap_or(0.5);
            f64::from(e) + mf.log2()
        }
        _ => f64::NEG_INFINITY,
    }
}

/// Bits of `abs(v)` before the binary point (at least 0).
fn int_bits(v: f64) -> usize {
    if v.is_finite() && v > 1.0 {
        v.log2().ceil() as usize + 1
    } else {
        1
    }
}

fn overflow(kind: Kind, lg2: f64) -> SymplexError {
    let what = match kind {
        Kind::J => "besselj",
        Kind::Y => "bessely",
        Kind::I => "besseli",
        Kind::K => "besselk",
    };
    SymplexError::Unevaluable {
        reason: format!(
            "{what}: the value (about 2^{lg2:.3e}) overflows the arbitrary-precision exponent range"
        ),
    }
}

/// `ν^ν e^(−ν)/Γ(ν + 1)` as its logarithm `ν ln ν − ν − ln Γ(ν + 1)` at `lp` bits.
fn ln_stirling_ratio(
    nu: &BigFloat,
    lp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let one = BigFloat::from_i32(1, 64);
    let nu1 = nu.add(&one, super::exact_bits(nu, lp), rm);
    let lg = super::arb_log_gamma(&nu1, lp, rm, cc)?;
    Ok(nu
        .mul(&nu.ln(lp, rm, cc), lp, rm)
        .sub(nu, lp, rm)
        .sub(&lg.0, lp, rm))
}

/// `e^E·f` at `prec` bits for `E` known to an absolute `2^(−wp−8)` and a
/// factor `f` in range: 0 below the exponent range (`I`, `J`, `K`, whose
/// underflow the accuracy layer holds as a nonzero number), refused above.
fn scale_exp(
    kind: Kind,
    e: &BigFloat,
    f: &BigFloat,
    prec: usize,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let l2 = super::bigfloat_to_f64_rounded(e, rm)? * std::f64::consts::LOG2_E + lg(f);
    let min = f64::from(astro_float::EXPONENT_MIN) + 64.0;
    let max = f64::from(astro_float::EXPONENT_MAX) - 64.0;
    if l2 < min {
        return Ok(BigFloat::new(prec));
    }
    if l2 > max {
        return Err(overflow(kind, l2));
    }
    let v = e.exp(wp, rm, cc).mul(f, wp, rm);
    if v.is_zero() || v.is_inf() || v.is_nan() {
        return Err(overflow(kind, l2));
    }
    Ok(super::round_to(v, prec, rm))
}

/// `J_ν(x)`, `Y_ν(x)`, `I_ν(x)` or `K_ν(x)` for `ν ≥ 16`, `x > 0` by the
/// expansions of the module documentation, to `prec` bits; `None` where
/// they do not apply or do not reach the precision (the turning point, a
/// `Y_ν(x)` with `x < ν`, too small an order), `Some(Err)` for a value
/// beyond the exponent range.
pub(super) fn debye(
    kind: Kind,
    nu: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    if !super::bf_strictly_positive(nu)
        || !super::bf_strictly_positive(x)
        || nu.is_inf()
        || x.is_inf()
        || nu.is_nan()
        || x.is_nan()
    {
        return None;
    }
    let nu_f = super::bigfloat_to_f64_rounded(nu, rm).ok()?;
    let x_f = super::bigfloat_to_f64_rounded(x, rm).ok()?;
    if nu_f.is_nan() || nu_f < MIN_ORDER || !nu_f.is_finite() || !x_f.is_finite() || x_f <= 0.0 {
        return None;
    }
    match kind {
        Kind::K | Kind::I => modified(kind, nu, x, nu_f, x_f, prec, rm, cc),
        Kind::J | Kind::Y => match x.cmp(nu) {
            Some(c) if c < 0 => {
                if kind == Kind::Y {
                    return None;
                }
                j_below(nu, x, nu_f, x_f, prec, rm, cc)
            }
            Some(c) if c > 0 => hankel(kind, nu, x, nu_f, x_f, prec, rm, cc),
            _ => None,
        },
    }
}

/// `K_ν(x)` or `I_ν(x)` (real `p = ν/r`, `r = (ν² + x²)^½`).
#[allow(clippy::too_many_arguments)]
fn modified(
    kind: Kind,
    nu: &BigFloat,
    x: &BigFloat,
    nu_f: f64,
    x_f: f64,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    let wp = prec + 40;
    // Sizes of the parts of the exponent, for the bits it needs.
    let r_f = nu_f.hypot(x_f);
    let ln_part = nu_f * ((nu_f + r_f) / x_f).ln().abs();
    let stirling = if kind == Kind::I {
        nu_f * nu_f.ln().abs() + nu_f
    } else {
        0.0
    };
    let lp = wp + 16 + int_bits(r_f + ln_part + stirling + 1.0);
    let r = nu
        .mul(nu, lp, rm)
        .add(&x.mul(x, lp, rm), lp, rm)
        .sqrt(lp, rm);
    let p = nu.div(&r, wp, rm);
    let lg_t = if kind == Kind::I { 0.0 } else { lg(&p) + 1e-9 };
    let pl = plan(lg_t, nu_f, wp)?;
    let s = sum_terms(pl.n, &p, nu, kind == Kind::K, wp, rm);
    // E and the factor in front.
    let nu_r = nu.add(&r, lp, rm);
    let (e, f) = if kind == Kind::K {
        // −νη = −r + ν·ln((ν + r)/x); (π/(2ν))^½(1 + z²)^(−¼) = (π/(2r))^½.
        let e = nu
            .mul(&nu_r.div(x, lp, rm).ln(lp, rm, cc), lp, rm)
            .sub(&r, lp, rm);
        let pi = cc.pi(wp, rm).clone();
        let f = pi
            .div(&r.mul(&BigFloat::from_i32(2, 64), wp, rm), wp, rm)
            .sqrt(wp, rm);
        (e, f)
    } else {
        // νη = r + ν·ln(x/(ν + r)); (1 + z²)^(−¼) = (ν/r)^½; over S_n(1).
        let e = match ln_stirling_ratio(nu, lp, rm, cc) {
            Ok(v) => v,
            Err(err) => return Some(Err(err)),
        };
        let e = e.add(&r, lp, rm).add(
            &nu.mul(&x.div(&nu_r, lp, rm).ln(lp, rm, cc), lp, rm),
            lp,
            rm,
        );
        let s1 = sum_terms(pl.n, &BigFloat::from_i32(1, wp), nu, false, wp, rm);
        let f = p.sqrt(wp, rm).div(&s1, wp, rm);
        (e, f)
    };
    // Relative error: the remainder over `abs(S)`, the roundings of `S` (and
    // of `S_n(1)`, `I`), `E` to `2^(−wp−8)`, the factor and the products.
    let n = pl.n as f64;
    let rounding = ((10.0 * n + 10.0) * pl.abs_sum).log2() - wp as f64;
    let rel = lg_add(
        lg_add(pl.lg_trunc, rounding) - pl.s_low.log2() + 1.0,
        -(wp as f64) + 4.0,
    );
    if rel > -((prec + 4) as f64) {
        return None;
    }
    let factor = f.mul(&s, wp, rm);
    Some(scale_exp(kind, &e, &factor, prec, wp, rm, cc))
}

/// `J_ν(x)` for `0 < x < ν` (real `p = ν/w > 1`, `w = (ν² − x²)^½`).
#[allow(clippy::too_many_arguments)]
fn j_below(
    nu: &BigFloat,
    x: &BigFloat,
    nu_f: f64,
    x_f: f64,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    let wp = prec + 40;
    let stirling = nu_f * nu_f.ln().abs() + nu_f;
    let ln_part = nu_f * (x_f / (2.0 * nu_f)).ln().abs();
    let lp = wp + 16 + int_bits(stirling + ln_part + nu_f + 1.0);
    // w = ((ν − x)(ν + x))^½, the difference exact enough at `lp` bits.
    let d = nu.sub(x, lp, rm);
    let w = d.mul(&nu.add(x, lp, rm), lp, rm).sqrt(lp, rm);
    let p = nu.div(&w, wp, rm);
    let lg_t = lg(&p) + 1e-9;
    let pl = plan(lg_t, nu_f, wp)?;
    let s = sum_terms(pl.n, &p, nu, false, wp, rm);
    let s1 = sum_terms(pl.n, &BigFloat::from_i32(1, wp), nu, false, wp, rm);
    let e = match ln_stirling_ratio(nu, lp, rm, cc) {
        Ok(v) => v,
        Err(err) => return Some(Err(err)),
    };
    let nu_w = nu.add(&w, lp, rm);
    let e = e.add(&w, lp, rm).add(
        &nu.mul(&x.div(&nu_w, lp, rm).ln(lp, rm, cc), lp, rm),
        lp,
        rm,
    );
    let f = p.sqrt(wp, rm).div(&s1, wp, rm);
    let n = pl.n as f64;
    let rounding = ((10.0 * n + 10.0) * pl.abs_sum).log2() - wp as f64;
    let rel = lg_add(
        lg_add(pl.lg_trunc, rounding) - pl.s_low.log2() + 1.0,
        -(wp as f64) + 4.0,
    );
    if rel > -((prec + 4) as f64) {
        return None;
    }
    let factor = f.mul(&s, wp, rm);
    Some(scale_exp(Kind::J, &e, &factor, prec, wp, rm, cc))
}

/// `J_ν(x)` or `Y_ν(x)` for `x > ν` from `H⁽¹⁾_ν(x)`; the cancellation of a
/// value near a zero measured and made up with more bits (up to `4·prec +
/// 1024`, then `None`).
#[allow(clippy::too_many_arguments)]
fn hankel(
    kind: Kind,
    nu: &BigFloat,
    x: &BigFloat,
    nu_f: f64,
    x_f: f64,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    let cap = prec * 4 + 1024;
    let mut wp = prec + 40;
    loop {
        let lp = wp + 16 + int_bits(x_f + nu_f + 1.0);
        let d = x.sub(nu, lp, rm);
        let w = d.mul(&x.add(nu, lp, rm), lp, rm).sqrt(lp, rm);
        let q = nu.div(&w, wp, rm);
        let lg_t = lg(&q) + 1e-9;
        let pl = plan(lg_t, nu_f, wp)?;
        let (re_s, im_s) = with_signs(pl.n, &q, nu, wp, rm);
        // ξ = w − ν·atan(w/ν) − π/4, to an absolute 2^(−wp−8).
        let pi = cc.pi(lp, rm).clone();
        let xi = w
            .sub(&nu.mul(&w.div(nu, lp, rm).atan(lp, rm, cc), lp, rm), lp, rm)
            .sub(&pi.div(&BigFloat::from_i32(4, 64), lp, rm), lp, rm);
        let (c, s) = (xi.cos(lp, rm, cc), xi.sin(lp, rm, cc));
        let amp = BigFloat::from_i32(2, wp)
            .div(&pi.mul(&w, wp, rm), wp, rm)
            .sqrt(wp, rm);
        let combo = if kind == Kind::J {
            c.mul(&re_s, wp, rm).sub(&s.mul(&im_s, wp, rm), wp, rm)
        } else {
            s.mul(&re_s, wp, rm).add(&c.mul(&im_s, wp, rm), wp, rm)
        };
        // Absolute error of the combination (relative to the amplitude):
        // the remainder, the roundings of S, of ξ (2^(−wp−8)) and of the
        // products.
        let n = pl.n as f64;
        let rounding = ((10.0 * n + 10.0) * pl.abs_sum).log2() - wp as f64;
        let abs_err = lg_add(lg_add(pl.lg_trunc, rounding) + 1.0, -(wp as f64) + 4.0);
        let lost = -lg(&combo);
        if combo.is_zero() || abs_err + lost > -((prec + 4) as f64) {
            // Near a zero: as many more bits as the value lost (the plan
            // takes more terms, while the expansion allows).
            if wp >= cap {
                return None;
            }
            let more = if combo.is_zero() {
                wp
            } else {
                lost.ceil() as usize + 16
            };
            wp = (wp + more).min(cap);
            continue;
        }
        return Some(Ok(super::round_to(amp.mul(&combo, wp, rm), prec, rm)));
    }
}

/// `(Re S, Im S)` for `S = Σ_{k<n} (−i)^k V_k(q)/ν^k`.
fn with_signs(
    n: usize,
    q: &BigFloat,
    nu: &BigFloat,
    wp: usize,
    rm: RoundingMode,
) -> (BigFloat, BigFloat) {
    with_u(n, |us| {
        let inv_nu = BigFloat::from_i32(1, wp).div(nu, wp, rm);
        let mut scale = BigFloat::from_i32(1, wp);
        let mut re = BigFloat::new(wp);
        let mut im = BigFloat::new(wp);
        for (k, u) in us.iter().enumerate().take(n) {
            if k > 0 {
                scale = scale.mul(&inv_nu, wp, rm);
            }
            let term = eval_dense(u, k, q, true, wp, rm).mul(&scale, wp, rm);
            // (−i)^k: 1, −i, −1, i.
            match k % 4 {
                0 => re = re.add(&term, wp, rm),
                1 => im = im.sub(&term, wp, rm),
                2 => re = re.sub(&term, wp, rm),
                _ => im = im.add(&term, wp, rm),
            }
        }
        (re, im)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_polynomials_match_dlmf() {
        with_u(3, |us| {
            let r = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
            assert_eq!(us[1].coeffs, vec![(1, r(1, 8)), (3, r(-5, 24))]);
            assert_eq!(
                us[2].coeffs,
                vec![(2, r(81, 1152)), (4, r(-462, 1152)), (6, r(385, 1152))]
            );
            assert_eq!(
                us[3].coeffs,
                vec![
                    (3, r(30375, 414720)),
                    (5, r(-369603, 414720)),
                    (7, r(765765, 414720)),
                    (9, r(-425425, 414720))
                ]
            );
        });
    }
}
