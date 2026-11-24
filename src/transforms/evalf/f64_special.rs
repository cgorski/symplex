//! Special function helpers (f64-based) (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Special function helpers (f64-based)
// ═══════════════════════════════════════════════════════════════════════════

/// `bf` as an `f64` ([`crate::base::bigfloat::to_f64_or_inf`]: the nearest
/// `f64`, ties to even, `±∞` beyond the range; zero is `+0.0`).
///
/// # Errors
///
/// [`SymplexError::Unevaluable`] for NaN.
pub(super) fn f64_of(bf: &BigFloat) -> Result<f64, SymplexError> {
    crate::base::bigfloat::to_f64_or_inf(bf).ok_or_else(|| SymplexError::Unevaluable {
        reason: "NaN in BigFloat to f64 conversion".into(),
    })
}

/// `erf x` by its Taylor series in `f64`, for the Newton start of
/// [`arb_erfinv`] (`abs(x) < 1/2`; the start only guides the certified
/// Halley iteration).  (An unused Lanczos `Γ` in `f64` and the asymptotic
/// branch of this function, which no caller reached, were removed at 0.54.)
pub(super) fn erf_f64(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_{n=0}^{inf} (-1)^n * x^(2n+1) / (n! * (2n+1))
    let mut sum = 0.0;
    let mut term = x; // first term: x
    sum += term;
    for n in 1..50 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// `loggamma(x)` for real `x` (not a pole): the analytic continuation of
/// `ln Γ` from `x > 0` (SymPy's and mpmath's `loggamma`), real for `x > 0`
/// and on the cut `x < 0` its limit from above, `ln|Γ(x)| − iπ⌈−x⌉`
/// (mpmath: `loggamma(-2.5) = −0.0562 − 3πi`).  Before 0.30 the value left
/// of 0 was the real `ln|Γ(x)|` (the compiled back-ends' `lgamma`, which is
/// now `ln(abs(gamma(x)))`).
///
/// * `x > 0`: Stirling's series ([`stirling_log_gamma`]).
/// * `x < 0`: real part `ln π − ln|sin πd| − ln Γ(1 − x)`, `d = x − round(x)`
///   exact; imaginary part `−π·⌈−x⌉ = π·⌊x⌋`.
///
/// Near a zero of `ln|Γ|` (`x = 1`, `2`, and two between consecutive poles
/// left of 0) the terms summed are much larger than the value: the loss is
/// measured against a bound on them and the evaluation repeated with that
/// many more bits, up to [`cancellation_cap`].  Before 0.29 neither:
/// `loggamma(1 + 2⁻¹⁰⁰)` was wrong from its 13th digit, `πx` was rounded
/// before `sin`, and arguments within `10⁻¹²` of a pole (in `f64`) were
/// refused as poles.
pub(super) fn arb_log_gamma(x: &BigFloat, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("LogGamma of special float value"));
    }
    if (x.is_int() || x.is_zero()) && !bf_strictly_positive(x) {
        return Err(infinite("LogGamma at non-positive integer pole"));
    }
    // The zeros at 1 and 2 are exact: a value 0 is no cancellation to make
    // up, and the argument's error ball decides the precision.  Up to 0.57
    // the loss was measured against a value of 0 and the loop ran to the cap:
    // `loggamma(1 + 10⁻⁷⁰)`, whose argument rounds to 1 at the first
    // precision, was `PrecisionExhausted` (mpmath `−5.7722…·10⁻⁷¹`).
    if x.is_int() && matches!(f64_of(x), Ok(v) if v == 1.0 || v == 2.0) {
        return Ok((BigFloat::new(prec), BigFloat::new(prec)));
    }
    let exact = x.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
    let cap = cancellation_cap(prec);
    // On the cut: −π·⌈−x⌉ = π·⌊x⌋, the limit from the upper half-plane.
    let imaginary = if bf_strictly_positive(x) {
        BigFloat::new(prec)
    } else {
        let pi = w.cc.pi(prec + 16, RM).clone();
        round_to(pi.mul(&x.floor(), prec + 16, RM), prec)
    };
    // The first attempt covers the terms' bits (a value of magnitude 2⁻⁸
    // and up needs no second one).
    let zs0 = f64_of(&x.abs())?.min(1e300) + 0.5 * (prec + 48) as f64 + 3.0;
    let mut extra = (24 + terms_bits(2.0 * zs0 * (zs0.ln() + 1.0)).max(0) as usize).min(cap);
    loop {
        let wp = prec + extra;
        // The value and a bound on the terms summed.
        let (value, largest) = if bf_strictly_positive(x) {
            // Stirling at z + s, s ‹ 0.5·wp: terms up to (z+s)·(|ln(z+s)| + 1),
            // and the shift Σ ln(z + i) below that.
            let v = stirling_log_gamma(x, &mut w.at(wp))?;
            let zs = f64_of(x)?.max(0.0) + 0.5 * wp as f64 + 2.0;
            (v, 2.0 * zs * (zs.ln() + 1.0))
        } else {
            let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
            let d = x.sub(&n, exact, RM);
            let pi = w.cc.pi(wp, RM).clone();
            let ln_sin = pi.mul(&d, wp, RM).sin(wp, RM, w.cc).abs().ln(wp, RM, w.cc);
            let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
            let lg1 = stirling_log_gamma(&one_minus_x, &mut w.at(wp))?;
            let ln_pi = pi.ln(wp, RM, w.cc);
            let v = ln_pi.sub(&ln_sin, wp, RM).sub(&lg1, wp, RM);
            let zs = f64_of(&one_minus_x)? + 0.5 * wp as f64 + 2.0;
            let terms = 2.0 * zs * (zs.ln() + 1.0) + f64_of(&ln_sin)?.abs() + 2.0;
            (v, terms)
        };
        // The loss relative to the whole value: on the cut its imaginary
        // part, at least π, bounds it below (a zero of ln|Γ| there costs no
        // extra bits).
        let magnitude = match (value.exponent(), imaginary.exponent()) {
            _ if value.is_zero() && imaginary.is_zero() => None,
            (Some(e), Some(i)) if !value.is_zero() && !imaginary.is_zero() => Some(e.max(i)),
            (Some(e), _) if !value.is_zero() => Some(e),
            (_, i) => i,
        };
        let lost = match magnitude {
            Some(e) => (terms_bits(largest) - i64::from(e)).max(0) as usize,
            None => wp,
        };
        if lost + 16 <= extra {
            return Ok((round_to(value, prec), imaginary));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// A precision at which a sum or difference of `x` and a small integer is
/// exact (or within `2^−64` of it relative to its larger operand's bits).
pub(super) fn exact_bits(x: &BigFloat, prec: usize) -> usize {
    // The lowest bit of `x` is at `e − m`; `x + c` for `|c| < 2^62` reaches
    // up to `max(e, 62)`.  Before 0.30 only `m + 64` bits: a tiny `x` lost
    // its bits in `x − k + 1`, and `binomial(10⁻⁷⁰, 6)` — `Γ` next to its
    // pole at `−5` — was wrong from its 8th digit.
    // (At most 2¹⁶ bits for the offset: beyond, `x + c` rounds to `c`, and
    // a pole of `Γ` there is refused rather than met with a million-bit
    // number.)
    let m = x.mantissa_max_bit_len().unwrap_or(prec).max(prec);
    let below = x.exponent().map_or(0, |e| {
        usize::try_from(64 - i64::from(e)).unwrap_or(0).min(1 << 16)
    });
    m + below + 64
}

/// `⌈log₂ v⌉ + 1` for a positive `f64` magnitude bound (`i64::MAX/4` beyond
/// range).
fn terms_bits(v: f64) -> i64 {
    if v.is_finite() && v > 0.0 {
        v.log2().ceil() as i64 + 1
    } else {
        i64::MAX / 4
    }
}

/// `ψ(x)` for real `x` (not a pole) to a relative error of about `2^−prec`.
///
/// 1. `x < 0`: the reflection `ψ(x) = ψ(1 − x) − π cot(πd)`, `d = x − round(x)`
///    exact (`cot` has period 1), so a pole is only an integer `x` and `πd`
///    is rounded relative to `d`, not to `x`.
/// 2. `x > 0`: `ψ(x) = ψ(x + N) − Σ_{k<N} 1/(x + k)` with `x + N ≥ wp/3`.
/// 3. `ψ(y) ~ ln y − 1/(2y) − Σ_{k≥1} B_{2k}/(2k·y^{2k})` (DLMF 5.11.2), with
///    exact Bernoulli numbers until a term is below the working precision
///    (the smallest term, near `k ≈ πy`, is `~e^{−2πy}`).
///
/// Near a zero of `ψ` (`x₀ = 1.4616…`, and one between consecutive poles
/// left of 0) the terms are much larger than the value: the loss, `log₂` of
/// the largest term over the result, is measured and the evaluation
/// repeated with that many more bits, up to [`cancellation_cap`].  Before
/// 0.29 the series stopped after 12 terms and the loss went unmeasured:
/// `digamma(1/3)` was wrong from its 46th digit, and `digamma` at a
/// 33-digit rational next to `x₀` from its 5th; `πx` was rounded before
/// `cot`, and poles were taken within `2^{−wp/2}`.
pub(super) fn arb_digamma(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Digamma of special float value".into(),
        });
    }
    let cap = cancellation_cap(prec);
    let mut extra = 20;
    loop {
        let wp = prec + extra;
        let (value, largest) = digamma_wp(x, &mut w.at(wp))?;
        let lost = match (largest.exponent(), value.exponent()) {
            _ if value.is_zero() => wp,
            (Some(l), Some(v)) => usize::try_from(i64::from(l) - i64::from(v)).unwrap_or(0),
            _ => 0,
        };
        if lost + 16 <= extra {
            return Ok(round_to(value, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `ψ(x)` at working precision `wp` (see [`arb_digamma`]), with the
/// magnitude of the largest term summed: the absolute error is about
/// `2^−wp` of it.
fn digamma_wp(x: &BigFloat, w: &mut Wp<'_>) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    let pole = || infinite("Digamma at non-positive integer pole");
    let exact = x.mantissa_max_bit_len().unwrap_or(wp).max(wp) + 64;
    let one = BigFloat::from_i32(1, wp);
    if x.is_negative() {
        // x = n + d, n = round(x), |d| ≤ ½, both exact.
        let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = x.sub(&n, exact, RM);
        if d.is_zero() {
            return Err(pole());
        }
        let pi = w.cc.pi(wp, RM).clone();
        let pi_d = pi.mul(&d, wp, RM);
        let pi_cot = pi.mul(
            &pi_d.cos(wp, RM, w.cc).div(&pi_d.sin(wp, RM, w.cc), wp, RM),
            wp,
            RM,
        );
        let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
        let (psi, largest) = digamma_wp(&one_minus_x, w)?;
        let value = psi.sub(&pi_cot, wp, RM);
        let largest = if pi_cot.abs() > largest {
            pi_cot.abs()
        } else {
            largest
        };
        return Ok((value, largest));
    }
    if x.is_zero() {
        return Err(pole());
    }
    // ψ(x) = ψ(y) − Σ_{k<N} 1/(x + k), y = x + N ≥ wp/3.
    let threshold = BigFloat::from_i32(i32::try_from(wp / 3).unwrap_or(i32::MAX).max(10), wp);
    let mut shift = BigFloat::new(wp);
    let mut y = x.clone();
    while bf_lt(&y, &threshold) {
        shift = shift.add(&one.div(&y, wp, RM), wp, RM);
        y = y.add(&one, exact, RM);
    }
    let ln_y = y.ln(wp, RM, w.cc);
    let largest = if shift > ln_y.abs() {
        shift.clone()
    } else {
        ln_y.abs()
    };
    let two = BigFloat::from_i32(2, wp);
    let mut result = ln_y
        .sub(&one.div(&y.mul(&two, wp, RM), wp, RM), wp, RM)
        .sub(&shift, wp, RM);
    let y2 = y.mul(&y, wp, RM);
    let mut y_pow = y2.clone();
    let floor = largest.exponent().map_or(i64::MIN / 2, i64::from) - wp as i64 - 2;
    let mut prev: Option<i32> = None;
    for k in 1..=wp {
        let b2k = bernoulli::even(k);
        let denom = y_pow.mul(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
        let term = ratio_to_bigfloat(&b2k, wp).div(&denom, wp, RM);
        match term.exponent() {
            Some(te) if i64::from(te) < floor => break,
            Some(te) if prev.is_some_and(|p| te > p + 2) => break, // diverging
            Some(te) => prev = Some(te),
            None => {}
        }
        result = result.sub(&term, wp, RM);
        y_pow = y_pow.mul(&y2, wp, RM);
    }
    Ok((result, largest))
}
