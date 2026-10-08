//! Bessel functions of large order, `|ν| ≥ 10⁴` ([`LARGE_ORDER`]).
//!
//! The series of `evalf.rs` took a step per unit of the order — `(x/2)ⁿ` by
//! repeated multiplication for `J_n`, the finite sums of `Y_n` and `K_n`
//! (A&S 9.1.11, 9.6.11) over all `n` terms — so `besselj(8345185991999992,
//! 61/10²⁷)` and `bessely(10⁵, 1)` never finished (and, through the zero
//! test of the canonical constructors, neither did building
//! `cos(besselj(…))/sin(besselj(…))`), and an order beyond `2⁶³` went through
//! a saturating `i64`.  Their factor `(x/2)^ν/Γ(ν + 1)` was also rounded at
//! the working precision in two parts of magnitude `|ν·ln(x/2)|` and
//! `ln Γ(ν + 1)`, which lose that many bits.
//!
//! mpmath (`functions/bessel.py`, BSD; see THIRD-PARTY-NOTICES.md) writes
//! `J_ν(z) = (z/2)^ν/Γ(ν + 1)·₀F₁(; ν + 1; −z²/4)` (`hypercomb`) with
//! unbounded exponents, and the series is short when `ν` is large next to
//! `z²`.  Here:
//!
//! * the factor in front is `±exp(ν·ln(x/2) − ln|Γ(ν + 1)|)`
//!   ([`power_over_gamma`]), its exponent to an absolute `2^(−wp−16)`; a
//!   value the series bounds below the exponent range of `BigFloat` (about
//!   `2^(−2.1·10⁹)`) is 0, the underflow of a nonzero number (`J_ν(x)` has
//!   no zero in `0 < x < ν`, `I_ν(x) > 0`), which `extended` scales by
//!   [`log_power_over_gamma`]; one above it is refused;
//! * `Y_n(x)` and `K_n(x)` for `x² ≤ 2(n − 1)` ([`neumann_leading`]) are the
//!   finite sum of the Neumann series alone: its terms fall by at least ½,
//!   and the logarithmic part, below `(x/2)^{2n}/(n!·(n − 1)!)` times it
//!   (DLMF 10.14.4 bounds `|J_n|`, `I_n`), is checked negligible;
//! * the ascending series of `I_ν` is refused when it would take more than
//!   [`MAX_TERMS`] terms (its terms grow until `k(ν + k) = x²/4`).

use astro_float::{BigFloat, Consts, RoundingMode};

use crate::base::errors::SymplexError;

/// The order from which the series use the factor of [`power_over_gamma`]
/// and `Y_n`, `K_n` the leading part of [`neumann_leading`].  Below it the
/// series are unchanged (the same digits as before).
pub(super) const LARGE_ORDER: f64 = 1.0e4;

/// The most terms the ascending series of `I_ν` may take.
const MAX_TERMS: f64 = 50_000.0;

/// Bits kept clear of the ends of the exponent range.
const MARGIN: f64 = 64.0;

/// Is the integer `n` odd?  Exactly, at any magnitude (`n/2` is an integer
/// for an even `n`; beyond the precision of the mantissa `n` is even).
pub(super) fn is_odd(n: &BigFloat) -> bool {
    if n.is_zero() {
        return false;
    }
    let mut half = n.clone();
    match n.exponent() {
        Some(e) => {
            half.set_exponent(e - 1);
            !half.is_int()
        }
        None => false,
    }
}

/// `ln|x|` in `f64` for a nonzero finite `x` at any exponent (`±∞` beyond
/// the `f64` range of the logarithm, which no `BigFloat` reaches).
fn ln_abs_f64(x: &BigFloat, rm: RoundingMode) -> f64 {
    let Some(e) = x.exponent() else {
        return f64::NEG_INFINITY;
    };
    let mut m = x.abs();
    m.set_exponent(0);
    let mf = super::bigfloat_to_f64_rounded(&m, rm).unwrap_or(0.5);
    f64::from(e) * std::f64::consts::LN_2 + mf.ln()
}

/// `ln Γ(y)` in `f64` for `y ≥ 10` (Stirling's series to `1/(12y)`).
fn ln_gamma_f64(y: f64) -> f64 {
    (y - 0.5) * y.ln() - y + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * y)
}

fn overflow(what: &str, lg: f64) -> SymplexError {
    SymplexError::Unevaluable {
        reason: format!(
            "{what}: the value (about 2^{lg:.3e}) overflows the arbitrary-precision exponent range"
        ),
    }
}

fn out_of_range(what: &str, lg: f64) -> SymplexError {
    SymplexError::Unevaluable {
        reason: format!(
            "{what} of large order: the factor (x/2)^nu/Gamma(nu + 1) (about 2^{lg:.3e}) is \
             beyond the arbitrary-precision exponent range"
        ),
    }
}

/// `log₂` bounds of the magnitude of a series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Lg2Bounds {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

/// `log₂` bounds of the series `S = Σ_k (±x²/4)^k/(k!·(ν + 1)_k)` that
/// multiplies [`power_over_gamma`] (`alternating` for `J`, not for `I`), for
/// `ν > 0`: `|S| ≤ e^{x²/(4(ν+1))}`, and `S ≥ 1` (`I`) or `S ≥ 1 −
/// x²/(4(ν + 1))` (`J`, whose terms then fall).  Unbounded for a negative
/// order.
pub(super) fn series_bounds(nu: f64, x: f64, alternating: bool) -> Lg2Bounds {
    if nu.is_nan() || nu <= 0.0 || !x.is_finite() {
        return Lg2Bounds {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        };
    }
    let y = x * x / (4.0 * (nu + 1.0));
    let hi = y * std::f64::consts::LOG2_E;
    let lo = if !alternating {
        0.0
    } else if y < 1.0 {
        (1.0 - y).log2()
    } else {
        f64::NEG_INFINITY
    };
    Lg2Bounds { lo, hi }
}

/// `(x/2)^ν/Γ(ν + 1)` at `wp` bits, for `x/2 = x_half > 0` and a real `ν`
/// off the negative integers: `±exp(ν·ln(x/2) − ln|Γ(ν + 1)|)` with its
/// exponent `L` to an absolute `2^(−wp−16)`, the sign of `Γ(ν + 1)`.
///
/// `bounds` are `log₂` bounds of the series the factor multiplies
/// ([`series_bounds`]).  `Ok(None)` when even the largest series leaves the
/// value below the exponent range (a nonzero number, for the caller to
/// report); refused when `L` leaves the range otherwise: as an overflow when
/// the smallest series leaves the value above it, else as out of range.
pub(super) fn power_over_gamma(
    nu: &BigFloat,
    x_half: &BigFloat,
    bounds: Lg2Bounds,
    what: &str,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Option<BigFloat>, SymplexError> {
    let min = f64::from(astro_float::EXPONENT_MIN) + MARGIN;
    let max = f64::from(astro_float::EXPONENT_MAX) - MARGIN;
    let decide = |l2: f64| -> Option<Result<Option<BigFloat>, SymplexError>> {
        if l2 + bounds.hi < min {
            Some(Ok(None))
        } else if l2 + bounds.lo > max {
            Some(Err(overflow(what, l2 + bounds.lo)))
        } else if !(min..=max).contains(&l2) {
            Some(Err(out_of_range(what, l2)))
        } else {
            None
        }
    };
    // An estimate first: the exponent and `ln Γ` of a huge order decide
    // without any work at the precision `L` would need.
    let nu_f = super::bigfloat_to_f64_rounded(nu, rm)?;
    let ln_xh = ln_abs_f64(x_half, rm);
    let t = nu_f * ln_xh;
    let (lg_est, lg_size) = if nu_f > 0.0 {
        let g = ln_gamma_f64(nu_f + 1.0);
        (g, g.abs())
    } else {
        // |Γ(ν + 1)| = π/(|sin π(ν + 1)|·Γ(−ν)) ≥ π/Γ(−ν): an upper bound of `L`.
        let g = (std::f64::consts::PI).ln() - ln_gamma_f64(-nu_f);
        (g, g.abs())
    };
    let l_est = t - lg_est;
    if !l_est.is_finite() || !lg_size.is_finite() {
        return if nu_f > 0.0 && l_est < 0.0 && bounds.hi.is_finite() {
            Ok(None)
        } else {
            Err(out_of_range(what, l_est * std::f64::consts::LOG2_E))
        };
    }
    // The estimate is within `slack` bits of `log₂` of the factor (for a
    // negative order it is an upper bound): the clear cases.
    let slack = 1e-9 * (t.abs() + lg_size) + 64.0;
    let l2_est = l_est * std::f64::consts::LOG2_E;
    if l2_est + slack + bounds.hi < min {
        return Ok(None);
    }
    if l2_est + slack < min {
        return Err(out_of_range(what, l2_est));
    }
    if nu_f > 0.0 && l2_est - slack + bounds.lo > max {
        return Err(overflow(what, l2_est + bounds.lo));
    }
    if nu_f > 0.0 && l2_est - slack > max {
        return Err(out_of_range(what, l2_est));
    }
    let l = log_power_over_gamma(nu, x_half, wp, rm, cc)?;
    let l2 = super::bigfloat_to_f64_rounded(&l, rm)? * std::f64::consts::LOG2_E;
    if let Some(r) = decide(l2) {
        return r;
    }
    let v = l.exp(wp, rm, cc);
    if v.is_zero() || v.is_inf() || v.is_nan() {
        return Err(out_of_range(what, l2));
    }
    // Γ(ν + 1) < 0 between the poles −2m − 1 and −2m: ⌊ν + 1⌋ odd.
    let nu1 = nu.add(&BigFloat::from_i32(1, 64), super::exact_bits(nu, wp), rm);
    let negative = nu1.is_negative() && is_odd(&nu1.floor());
    Ok(Some(if negative { v.neg() } else { v }))
}

/// `L = ν·ln(x/2) − ln|Γ(ν + 1)|` (the logarithm of the magnitude of the
/// factor of [`power_over_gamma`]) for `x/2 = x_half > 0` and a real `ν` off
/// the negative integers, to an absolute `2^(−wp−16)`: at as many more bits
/// as `abs(ν·ln(x/2))` and `abs(ln Γ(ν + 1))` have before the point (and,
/// left of 0, the bits `ln abs(sin π(ν + 1))` may add: those of the
/// mantissa of ν).  At any magnitude: `extended` scales a factor below the
/// exponent range by it.
pub(super) fn log_power_over_gamma(
    nu: &BigFloat,
    x_half: &BigFloat,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<BigFloat, SymplexError> {
    let nu_f = super::bigfloat_to_f64_rounded(nu, rm)?;
    let t = nu_f * ln_abs_f64(x_half, rm);
    let lg_size = if nu_f > 0.0 {
        ln_gamma_f64(nu_f + 1.0).abs()
    } else {
        ((std::f64::consts::PI).ln() - ln_gamma_f64(-nu_f)).abs()
    };
    let sin_bits = if nu_f > 0.0 {
        0.0
    } else {
        nu.mantissa_max_bit_len().unwrap_or(wp) as f64
    };
    let size = t.abs() + lg_size + sin_bits + 2.0;
    if !size.is_finite() {
        return Err(out_of_range("a Bessel function", f64::INFINITY));
    }
    let lp = wp + 16 + size.log2().ceil().max(0.0) as usize;
    let one = BigFloat::from_i32(1, 64);
    let nu1 = nu.add(&one, super::exact_bits(nu, lp), rm);
    let lg = super::arb_log_gamma(&nu1, lp, rm, cc)?;
    Ok(nu.mul(&x_half.ln(lp, rm, cc), lp, rm).sub(&lg.0, lp, rm))
}

/// The ascending series of `I_ν(x)` would take more than [`MAX_TERMS`]
/// terms: they grow until `k(|ν| + k) = x²/4`.
pub(super) fn refuse_long_i_series(nu: f64, x: f64) -> Result<(), SymplexError> {
    let nu = nu.abs();
    let peak = ((nu * nu + x * x).sqrt() - nu) / 2.0;
    if !peak.is_finite() || peak > MAX_TERMS {
        return Err(SymplexError::NotImplemented(format!(
            "besseli of order {nu:e} at {x:e}: the ascending series would take more than \
             {MAX_TERMS:e} terms (no uniform asymptotic expansion)"
        )));
    }
    Ok(())
}

/// The most terms of the sum of [`neumann_leading`] when they rise first.
const MAX_RISING_STEPS: u64 = 400_000;

/// The neglected part of `Y_n(x)`'s series relative to the finite sum, as
/// [`neumann_leading`] bounds it, when `y = x²/4 > (n − 1)/2`: the terms
/// `c_k = (n−k−1)!/k!·y^k` of the finite sum (all positive) then rise
/// while `k(n − k) < y` and fall after.  Their ratios `y/(k(n − k))`
/// decrease up to `k = n/2` and increase after, so the sum stops at a term
/// `c_K` falling by a ratio `r < 1` (`K < n/2`) and below
/// `2^(−wp−9)·(1 − r)/n` of the sum: the terms up to `n/2` are below
/// `c_K·r/(1 − r)` in total, and the ones after (log-convex) below
/// `n/2·max(c_{n/2}, c_{n−1})`,
/// with `c_{n−1}/c_0 = y^(n−1)/((n − 1)!)²`.  The log and ψ parts of A&S
/// 9.1.11 over `c_0`: `|J_n| ≤ (x/2)ⁿ/n!` and `Σ_k |ψ(k+1) +
/// ψ(n+k+1)|·y^k/(k!(n+k)!) ≤ (2 ln(n + 1) + 2 + 2y/(n + 1)²)·e^(y/(n+1))/n!`,
/// so the part is below
/// `(x/2)^(2n)/(n!(n−1)!)·(2·abs(ln(x/2)) + (2 ln(n+1) + 2 + 2y/(n+1)²)·e^(y/(n+1)))`.
/// `log₂` of that part and of
/// `n/2·c_{n−1}`, both over `c_0`, which the caller checks below
/// `2^(−wp−16)` of the sum.  Before 0.35 `x² > 2(n − 1)` was refused:
/// `bessely(20000, 300)`.
fn neumann_rising(nf: f64, xf: f64, ln_x2: f64) -> Option<(f64, f64)> {
    let y = xf * xf / 4.0;
    let last = (nf / 2.0).ln() + (nf - 1.0) * y.ln() - 2.0 * ln_gamma_f64(nf);
    if !last.is_finite() {
        return None;
    }
    let psi = 2.0 * (nf + 1.0).ln() + 2.0 + 2.0 * y / ((nf + 1.0) * (nf + 1.0));
    let ln_psi = psi.ln() + y / (nf + 1.0);
    let ln_log = (2.0 * ln_x2.abs()).max(f64::MIN_POSITIVE).ln();
    let hi = ln_psi.max(ln_log);
    let both = hi + ((ln_psi - hi).exp() + (ln_log - hi).exp()).ln();
    let rest = 2.0 * nf * ln_x2 - ln_gamma_f64(nf + 1.0) - ln_gamma_f64(nf) + both;
    let l2 = std::f64::consts::LOG2_E;
    (rest.is_finite()).then_some((rest * l2, last * l2))
}

/// The function of [`neumann_leading`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Second {
    /// `Y_n`.
    Y,
    /// `K_n`.
    K,
}

/// `Y_n(x)` or `K_n(x)` for an integer `n ≥ LARGE_ORDER` and `0 < x`,
/// `x² ≤ 2(n − 1)`: the finite part of the Neumann series,
///
/// ```text
/// Y_n(x) ≈ −(1/π)·Γ(n)·(x/2)^(−n)·Σ_{k<n} c_k,   K_n(x) ≈ ½·Γ(n)·(x/2)^(−n)·Σ_{k<n} (−1)^k c_k,
/// c_0 = 1,  c_k = c_{k−1}·(x²/4)/(k(n − k)),
/// ```
///
/// whose terms fall by at least ½ (`k(n − k) ≥ n − 1`), summed until one is
/// below `2^(−wp−8)` of the sum.  The rest of A&S 9.1.11 / 9.6.11 is at most
/// `7·(x/2)^{2n}/(n!·(n − 1)!)·(|ln(x/2)| + 2·ln(n + 1) + 2)` times the
/// part summed (`|J_n(x)| ≤ (x/2)ⁿ/n!`, `I_n(x) ≤ e^{x²/(4(n+1))}·(x/2)ⁿ/n!`,
/// `|ψ(m)| ≤ ln m + 1`); `None` unless that is below `2^(−wp−16)` (or
/// unless `x² ≤ 2(n − 1)`): the caller's series serves.  `Y_n` with
/// `x² > 2(n − 1)`, whose terms rise first, as [`neumann_rising`] bounds
/// it (to about `x = 0.73·n`).
pub(super) fn neumann_leading(
    kind: Second,
    n: &BigFloat,
    x: &BigFloat,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Option<Result<BigFloat, SymplexError>> {
    let what = match kind {
        Second::Y => "bessely",
        Second::K => "besselk",
    };
    let nf = super::bigfloat_to_f64_rounded(n, rm).ok()?;
    let xf = super::bigfloat_to_f64_rounded(x, rm).ok()?;
    // `x² > 2(n − 1)`: the terms of `Y_n`'s sum (all positive) rise
    // first, see [`neumann_rising`]; `K_n`'s alternate and would cancel.
    let rising = xf * xf > 2.0 * (nf - 1.0);
    // (Callers ask from `LARGE_ORDER` on, and for `Y_n` from 16 where its
    // full series would cancel beyond the cap; `ln_gamma_f64` needs 10.)
    let applies = nf >= 16.0 && (!rising || kind == Second::Y);
    if !applies {
        return None;
    }
    let wp = prec + 40;
    let ln_x2 = ln_abs_f64(x, rm) - std::f64::consts::LN_2;
    // The neglected parts over the first term `c_0 = 1`, in `log₂`: checked
    // here against it, or (rising terms) against the sum below.
    let (rest, last) = if rising {
        neumann_rising(nf, xf, ln_x2)?
    } else {
        let rest = 7f64.ln() + 2.0 * nf * ln_x2 - ln_gamma_f64(nf + 1.0) - ln_gamma_f64(nf)
            + (ln_x2.abs() + 2.0 * (nf + 1.0).ln() + 2.0).ln();
        (rest * std::f64::consts::LOG2_E, f64::NEG_INFINITY)
    };
    let negligible = |lg_sum: f64| rest.max(last) - lg_sum < -((wp + 16) as f64);
    if !rising && !negligible(0.0) {
        return None;
    }
    // The relative size below which the sum stops (see `neumann_rising`).
    let stop_bits = if rising {
        (wp + 9) as i64 + nf.log2().ceil() as i64
    } else {
        (wp + 8) as i64
    };
    let max_steps: u64 = if rising {
        MAX_RISING_STEPS
    } else {
        (4 * wp) as u64
    };
    let two = BigFloat::from_i32(2, 64);
    let x_half = x.div(&two, wp, rm);
    // Γ(n)·(x/2)^(−n) = 1/((x/2)^(n−1)/Γ(n)) / (x/2): through
    // `power_over_gamma` of the order n − 1 (the series is at least ½, `K`,
    // and at most 2).
    let nm1 = n.sub(&BigFloat::from_i32(1, 64), super::exact_bits(n, wp), rm);
    let bounds = Lg2Bounds { lo: -1.0, hi: 1.0 };
    let p = match power_over_gamma(&nm1, &x_half, bounds, what, wp, rm, cc) {
        Ok(Some(p)) => p,
        // Its reciprocal: an underflow of the factor is an overflow here.
        Ok(None) => {
            let l2 = (ln_gamma_f64(nf) - nf * ln_x2) * std::f64::consts::LOG2_E;
            return Some(Err(overflow(what, l2)));
        }
        Err(e) => return Some(Err(e)),
    };
    let q = p.mul(&x_half, wp, rm).reciprocal(wp, rm);
    let xh2 = x_half.mul(&x_half, wp, rm);
    let mut c = BigFloat::from_i32(1, wp);
    let mut sum = c.clone();
    let mut k: u64 = 1;
    loop {
        let kb = BigFloat::from_u64(k, 64);
        let nk = n.sub(&kb, super::exact_bits(n, wp), rm);
        c = c.mul(&xh2, wp, rm).div(&kb.mul(&nk, wp, rm), wp, rm);
        // Rising terms: past their peak, falling by the ratio `r < 1` (and
        // by less after, up to the middle of the sum), the rest of them
        // below `c·r/(1 − r)` (`neumann_rising`).
        let kf = k as f64;
        let r = xf * xf / 4.0 / (kf * (nf - kf));
        let geometric = if !rising {
            0
        } else if r < 0.99 && kf < nf / 2.0 {
            (1.0 / (1.0 - r)).log2().ceil() as i64
        } else {
            i64::MAX / 4
        };
        let small = match (c.exponent(), sum.exponent()) {
            (Some(ce), Some(se)) => i64::from(se) - i64::from(ce) > stop_bits + geometric,
            _ => c.is_zero(),
        };
        if small {
            break;
        }
        if rising && kf >= nf / 2.0 {
            // Not reached: the terms fall well before the middle.
            return None;
        }
        sum = if kind == Second::K && k % 2 == 1 {
            sum.sub(&c, wp, rm)
        } else {
            sum.add(&c, wp, rm)
        };
        if k as f64 >= nf - 1.0 {
            break;
        }
        k += 1;
        if k > max_steps {
            // Not reached without rising terms: they fall by ½ or more.
            return None;
        }
    }
    // The neglected parts against the sum (`neumann_rising`).
    if rising {
        let lg_sum = match sum.exponent() {
            Some(e) if !sum.is_zero() => f64::from(e) - 1.0,
            _ => return None,
        };
        if !negligible(lg_sum) {
            return None;
        }
    }
    let v = match kind {
        Second::Y => q
            .mul(&sum, wp, rm)
            .div(&cc.pi(wp, rm).clone(), wp, rm)
            .neg(),
        Second::K => q.mul(&sum, wp, rm).div(&two, wp, rm),
    };
    if v.is_zero() || v.is_inf() || v.is_nan() {
        let l2 = (ln_gamma_f64(nf) - nf * ln_x2) * std::f64::consts::LOG2_E;
        return Some(Err(overflow(what, l2)));
    }
    Some(Ok(super::round_to(v, prec, rm)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_of_huge_integers() {
        let p = 256;
        assert!(is_odd(&BigFloat::from_i32(-7, p)));
        assert!(!is_odd(&BigFloat::from_i32(10, p)));
        assert!(!is_odd(&BigFloat::from_i32(0, p)));
        // 10²⁰ is even, 10²⁰ + 1 odd (beyond an i64).
        let e20 = BigFloat::from_u64(10_000_000_000, p).mul(
            &BigFloat::from_u64(10_000_000_000, p),
            p,
            RoundingMode::ToEven,
        );
        assert!(!is_odd(&e20));
        let e20p1 = e20.add(&BigFloat::from_i32(1, p), p, RoundingMode::ToEven);
        assert!(is_odd(&e20p1));
    }

    #[test]
    fn series_bounds_of_j_and_i() {
        let b = series_bounds(1.0e4, 1.0, true);
        assert!(b.lo < 0.0 && b.lo > -1e-4 && b.hi > 0.0 && b.hi < 1e-4);
        assert_eq!(series_bounds(1.0e4, 1.0, false).lo, 0.0);
        assert_eq!(series_bounds(-1.0e4, 1.0, true).hi, f64::INFINITY);
    }
}
