//! Arbitrary-precision erf via Taylor series / asymptotic expansion (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision erf via Taylor series / asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// `erfc(x)` for `x > 0` from the asymptotic expansion
///
/// ```text
/// erfc(x) = exp(−x²)/(x√π) · Σ_{n≥0} (−1)^n (2n−1)!! / (2x²)^n
/// ```
///
/// truncated at its smallest term (near `n ≈ x²`), so the *relative* error
/// is `≈ e^{−x²}`: callers must ensure `x² ≳ wp·ln 2` when they need
/// `erfc` itself to `wp` bits, or `2x² ≳ wp·ln 2` when they only need
/// `erf = 1 − erfc` to `wp` bits (the absolute error is then `≈ e^{−2x²}`).
pub(super) fn erfc_asymptotic(ax: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let x_sq = ax.mul(ax, wp, RM);
    let (sum, x_sqrt_pi) = erfc_asymptotic_series(ax, w);
    let exp_neg_x_sq = x_sq.neg().exp(wp, RM, w.cc);
    exp_neg_x_sq.mul(&sum, wp, RM).div(&x_sqrt_pi, wp, RM)
}

/// The sum `Σ_{n≥0} (−1)^n (2n−1)!!/(2x²)^n` of [`erfc_asymptotic`] at `wp`
/// bits, truncated at its smallest term, and `x·√π`: `erfc(x)` is
/// `e^(−x²)·sum/(x√π)` (`extended` takes `e^(−x²)` at its own scale, below
/// the exponent range).
pub(super) fn erfc_asymptotic_series(ax: &BigFloat, w: &mut Wp<'_>) -> (BigFloat, BigFloat) {
    let wp = w.bits();
    let x_sq = ax.mul(ax, wp, RM);
    let two_x_sq = x_sq.mul(&BigFloat::from_i32(2, wp), wp, RM);

    let mut sum = BigFloat::from_i32(1, wp);
    let mut term = BigFloat::from_i32(1, wp);
    let mut prev_abs: Option<BigFloat> = None;
    // The terms shrink until `n ≈ x²`; past that they grow and the loop
    // stops, so `wp + 40` is only reached when the terms are already tiny.
    let max_terms = wp + 40;
    for n in 1..=max_terms {
        // term *= −(2n−1) / (2x²)
        let factor = BigFloat::from_i128(2 * n as i128 - 1, wp);
        term = term.mul(&factor, wp, RM).div(&two_x_sq, wp, RM).neg();
        let t_abs = term.abs();
        if let Some(ref p) = prev_abs
            && bf_gt(&t_abs, p)
        {
            break; // past the smallest term: optimal truncation
        }
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            break;
        }
        prev_abs = Some(t_abs);
    }

    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let x_sqrt_pi = ax.mul(&sqrt_pi, wp, RM);
    (sum, x_sqrt_pi)
}

/// Compute erf(x) at arbitrary precision for real x.  The result carries
/// `prec + 32` bits.
///
/// For small-to-moderate |x|, uses the Taylor series:
///   erf(x) = (2/√π) Σ_{n=0}^{N} (-1)^n x^{2n+1} / (n! (2n+1))
/// whose terms peak near `n ≈ x²` at `≈ e^{x²}` before the alternating sum
/// cancels down to `|erf| < 1`, so the working precision carries
/// `x² log₂ e` extra bits.
///
/// For large |x| (`2x² ≳ wp·ln 2`), uses [`erfc_asymptotic`]:
///   erf(x) = sign(x) · (1 − erfc(|x|))
pub(super) fn arb_erf(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }

    let guard = 32;
    let wp = prec + guard;

    let x_approx = f64_of(x)?;
    // Threshold: for |x| beyond this, the asymptotic expansion (absolute
    // error ≈ e^{−2x²} for erf) is accurate to wp bits. Roughly √(wp·ln 2/2).
    let threshold = ((wp as f64) * 0.35).sqrt() + 2.0;

    if x_approx.abs() < threshold {
        // ── Taylor series ──────────────────────────────────────────────────────────
        // erf(x) = (2/√π) · Σ_{n=0}^{N} prod_n / (2n+1)
        // where prod_0 = x, prod_{n+1} = prod_n · (-x²) / (n+1)
        let x_sq_f = x_approx * x_approx;
        let wpt = Prec::new(wp)
            .guarded(
                Guard::for_log2(x_sq_f * std::f64::consts::LOG2_E),
                Prec::MAX,
            )
            .ok_or_else(|| special_exhausted(prec))?
            .plus(8)
            .bits();
        let mut xw = x.clone();
        let _ = xw.set_precision(wpt, RM);
        let neg_x_sq = xw.mul(&xw, wpt, RM).neg();
        let mut prod = xw.clone(); // (-x²)^n · x / n!
        let mut sum = xw.clone(); // accumulator (first term = x)

        // |term_n| < 2^{−wpt} needs n ≈ 5x² at the threshold, fewer below it.
        let max_terms = (6.0 * x_sq_f) as usize + wpt + 60;
        let mut converged = false;
        for n in 1..=max_terms {
            // prod *= -x² / n
            prod = prod.mul(&neg_x_sq, wpt, RM);
            prod = prod.div(&BigFloat::from_i128(n as i128, wpt), wpt, RM);

            // term = prod / (2n+1)
            let divisor = BigFloat::from_i128(2 * n as i128 + 1, wpt);
            let term = prod.div(&divisor, wpt, RM);

            if negligible(&term, &sum, wpt) {
                converged = true;
                break;
            }
            sum = sum.add(&term, wpt, RM);
        }
        if !converged {
            return Err(series_did_not_converge("erf", max_terms));
        }

        // Multiply by 2/√π
        let two = BigFloat::from_i32(2, wpt);
        let pi_val = w.cc.pi(wpt, RM).clone();
        let sqrt_pi = pi_val.sqrt(wpt, RM);
        let two_over_sqrt_pi = two.div(&sqrt_pi, wpt, RM);

        Ok(round_to(sum.mul(&two_over_sqrt_pi, wpt, RM), wp))
    } else {
        // ── Asymptotic expansion for large |x| ────────────────────────────────────
        let mut ax = x.abs();
        let _ = ax.set_precision(wp, RM);
        let erfc_val = erfc_asymptotic(&ax, &mut w.at(wp));
        let one = BigFloat::from_i32(1, wp);
        let erf_val = one.sub(&erfc_val, wp, RM);

        if x.is_negative() {
            Ok(erf_val.neg())
        } else {
            Ok(erf_val)
        }
    }
}

// ── Error-function family ──────────────────────────────────────────────────────

/// `erfi(x) = (2/√π) Σ x^{2n+1}/(n!(2n+1))` (all terms of one sign, so no
/// cancellation) for `x² ≤ wp·ln 2`, otherwise the asymptotic expansion
/// `erfi(x) ~ e^{x²}/(x√π) Σ (2k−1)!!/(2x²)^k` (optimal truncation error
/// `~e^{−x²}` relative to the leading term).
pub(super) fn arb_erfi(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("erfi of special float value"));
    }
    let wp = prec + 32;
    let x_f = f64_of(x)?;
    let ax = x_f.abs();
    let two = BigFloat::from_i32(2, wp);
    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let x_sq = x.mul(x, wp, RM);

    if ax * ax > (wp as f64) * std::f64::consts::LN_2 + 4.0 {
        let two_x_sq = x_sq.mul(&two, wp, RM);
        let mut sum = BigFloat::from_i32(1, wp);
        let mut term = BigFloat::from_i32(1, wp);
        let mut prev: Option<BigFloat> = None;
        for k in 1..(wp * 2 + 100) {
            let f = BigFloat::from_i128(2 * k as i128 - 1, wp);
            term = term.mul(&f, wp, RM).div(&two_x_sq, wp, RM);
            if let Some(ref p) = prev
                && bf_gt(&term, p)
            {
                break;
            }
            sum = sum.add(&term, wp, RM);
            if negligible(&term, &sum, wp) {
                break;
            }
            prev = Some(term.clone());
        }
        let e = x_sq.exp(wp, RM, w.cc);
        let denom = x.abs().mul(&sqrt_pi, wp, RM);
        let r = e.mul(&sum, wp, RM).div(&denom, wp, RM);
        let r = if x.is_negative() { r.neg() } else { r };
        return Ok(round_to(r, prec));
    }

    let mut prod = x.clone();
    let _ = prod.set_precision(wp, RM);
    let mut sum = prod.clone();
    let max_terms = (3.0 * ax * ax) as usize + wp + 60;
    let mut converged = false;
    for n in 1..=max_terms {
        let n_bf = BigFloat::from_i128(n as i128, wp);
        prod = prod.mul(&x_sq, wp, RM).div(&n_bf, wp, RM);
        let d = BigFloat::from_i128(2 * n as i128 + 1, wp);
        let term = prod.div(&d, wp, RM);
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("erfi", max_terms));
    }
    let r = sum.mul(&two, wp, RM).div(&sqrt_pi, wp, RM);
    Ok(round_to(r, prec))
}

/// `erfc(x)` with *relative* precision `prec` for `x > 0` (absolute
/// precision, via `1 − erf`, for `x ≤ 1/2`, where nothing cancels).
///
/// * `x² log₂ e ≥ wp + 8`: the asymptotic expansion alone (relative error
///   `≈ e^{−x²}`) already delivers `wp` bits.
/// * `x ≥ 5/4`: `erfc(x) = Γ(½, x²)/√π` by Legendre's continued fraction
///   ([`uppergamma_cf`], in its regime `x² ≥ max(1, ½ + 1)`), which cancels
///   nothing.  Before 0.30 this range took `1 − erf(x)` with `x² log₂ e`
///   extra bits, a Taylor series of some `e·x²` terms at that precision
///   (`erfc(18.5)` at 300 digits: 11 ms, now 2 ms).
/// * otherwise `1 − erf(x)` with `x² log₂ e + 8` extra bits to survive the
///   cancellation (`erfc(x) ≈ e^{−x²}/(x√π)`).
pub(super) fn arb_erfc(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let x_f = f64_of(x)?;
    if x_f <= 0.5 {
        let erf = arb_erf(x, w)?;
        let one = BigFloat::from_i32(1, wp);
        return Ok(round_to(one.sub(&erf, wp, RM), prec));
    }
    let x_sq_bits = (x_f * x_f * std::f64::consts::LOG2_E).ceil() as usize;
    let mut xw = x.clone();
    if x_sq_bits >= wp + 8 {
        let _ = xw.set_precision(wp, RM);
        return Ok(round_to(erfc_asymptotic(&xw, &mut w.at(wp)), prec));
    }
    if x_f >= 1.25 {
        // x² to a relative 2^−(wp + log₂ x² + 8): e^{−x²} to 2^−(wp + 8).
        let p = wp + magnitude_bits(x_f * x_f) + 8;
        let _ = xw.set_precision(p, RM);
        let x_sq = xw.mul(&xw, p, RM);
        let half = BigFloat::from_f64(0.5, 64);
        let g = uppergamma_cf(&half, &x_sq, &mut w.at(p))?;
        let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
        return Ok(round_to(g.div(&sqrt_pi, wp, RM), prec));
    }
    let wp2 = wp + x_sq_bits + 8;
    let _ = xw.set_precision(wp2, RM);
    let erf = arb_erf(&xw, &mut w.at(wp2))?;
    let one = BigFloat::from_i32(1, wp2);
    Ok(round_to(one.sub(&erf, wp2, RM), prec))
}

/// Halley iteration for `erf(x) = y` (`solve_erfc == false`) or
/// `erfc(x) = y` (`solve_erfc == true`) from the starting point `x0`,
/// safeguarded by the bracket `[lo, hi]` that contains the root (both
/// functions are monotone): a step that leaves the bracket, or does not at
/// least halve it, is replaced by bisection, and the bracket shrinks with
/// every evaluation.  An iteration that does not converge is refused.
///
/// With `g = f/f'`, the Halley step for both equations is
/// `x ← x − g / (1 + x g)` (since `f'' = −2x f'`).
///
/// Before 0.30 there was no bracket and no convergence check: from a poor
/// start the iteration drifted to where `erfc` saturates at 2 and returned
/// its 200th iterate, `erfcinv(941/2³⁴) = −10573.05…` (truly `3.84317…`),
/// certified.
#[allow(clippy::too_many_arguments)]
fn erf_inverse_halley(
    y: &BigFloat,
    x0: f64,
    lo: f64,
    hi: f64,
    solve_erfc: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let mut yw = y.clone();
    let _ = yw.set_precision(wp, RM);
    let half_sqrt_pi =
        w.cc.pi(wp, RM)
            .clone()
            .sqrt(wp, RM)
            .div(&BigFloat::from_i32(2, wp), wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let (mut lo, mut hi) = (bigfloat::from_f64(lo, wp), bigfloat::from_f64(hi, wp));
    let mut x = bigfloat::from_f64(x0.clamp(lo_f(&lo), hi_f(&hi)), wp);
    fn lo_f(v: &BigFloat) -> f64 {
        f64_of(v).unwrap_or(f64::NEG_INFINITY)
    }
    fn hi_f(v: &BigFloat) -> f64 {
        f64_of(v).unwrap_or(f64::INFINITY)
    }
    for _ in 0..(4 * wp + 200) {
        let f = if solve_erfc {
            arb_erfc(&x, &mut w.at(wp))?.sub(&yw, wp, RM)
        } else {
            arb_erf(&x, &mut w.at(wp))?.sub(&yw, wp, RM)
        };
        if f.is_zero() {
            return Ok(round_to(x, prec));
        }
        // erf increases, erfc decreases: the root is above x when
        // erf(x) < y or erfc(x) > y.
        let root_above = f.is_negative() != solve_erfc;
        if root_above {
            lo = x.clone();
        } else {
            hi = x.clone();
        }
        // f' = ±(2/√π) e^{−x²}  ⇒  g = f/f' = ±f (√π/2) e^{x²}
        let x_sq = x.mul(&x, wp, RM);
        let g = f
            .mul(&half_sqrt_pi, wp, RM)
            .mul(&x_sq.exp(wp, RM, w.cc), wp, RM);
        let g = if solve_erfc { g.neg() } else { g };
        let denom = one.add(&x.mul(&g, wp, RM), wp, RM);
        let delta = if denom.is_zero() {
            g.clone()
        } else {
            g.div(&denom, wp, RM)
        };
        // Converged to `prec + 8` bits (the working precision carries 32
        // guard bits, of which the last few are rounding noise in `f`, so
        // demanding all of them never ends: `erfinv(erf(−9))` at 60
        // digits stalled on a one-ulp bracket).
        let tol = prec + 8;
        let width = hi.sub(&lo, wp, RM);
        let candidate = x.sub(&delta, wp, RM);
        // A step below the tolerance is convergence, before the bracket is
        // consulted: below an ulp of x the candidate rounds onto x, which is
        // now an end of the bracket, and "leaving the bracket" sent a
        // converged iteration into bisection from its last bracket down to
        // the tolerance (0.29: some 260 halvings, `erfcinv(sin(1 + 10⁻¹⁵⁰)
        // − sin(1))` at 60 digits evaluated `erfc` 1,104 times).
        if negligible(&delta, &x, tol) {
            return Ok(round_to(candidate, prec));
        }
        let inside = bf_gt(&candidate, &lo) && bf_lt(&candidate, &hi);
        let halves = bf_lt(&delta.abs().mul(&two, wp, RM), &width);
        if inside && halves {
            x = candidate;
            if negligible(&delta, &x, tol) {
                return Ok(round_to(x, prec));
            }
        } else {
            let mid = lo.add(&hi, wp, RM).div(&two, wp, RM);
            let stuck = mid == lo || mid == hi;
            x = mid;
            if stuck || negligible(&width, &x, tol) {
                return Ok(round_to(x, prec));
            }
        }
    }
    Err(SymplexError::computation_failed(
        "evalf",
        "the inverse error function iteration did not converge",
    ))
}

/// `erfinv(y)` for `|y| < 1`: Winitzki's closed-form approximation refined
/// by `f64` Newton steps, then Halley iteration on `erf` (`|y| ≤ 1/2`) or
/// on `erfc(x) = 1 − |y|` (`|y| > 1/2`, where `erf` alone would lose the
/// digits of `1 − y`).
pub(super) fn arb_erfinv(y: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if y.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let ay = y.abs();
    let one_minus_ay = one.sub(&ay, wp, RM);
    if one_minus_ay.is_zero() || one_minus_ay.is_negative() {
        return Err(unevaluable("erfinv: argument must satisfy |y| < 1"));
    }
    // A tiny argument: erfinv(y) = (√π/2)·y·(1 + πy²/12 + …), whose
    // correction is below the working precision once |y| < 2^(−wp/2).  (The
    // bracketed iteration starts from (−½, ½) and cannot bisect down to
    // `erfinv(sech(8!)) ≈ 3·10⁻¹⁷⁵¹¹`.)
    if let Some(e) = y.exponent()
        && i64::from(e) < -(wp as i64) / 2 - 4
    {
        let half_sqrt_pi =
            w.cc.pi(wp, RM)
                .clone()
                .sqrt(wp, RM)
                .div(&BigFloat::from_i32(2, wp), wp, RM);
        return Ok(round_to(y.mul(&half_sqrt_pi, wp, RM), prec));
    }
    let y_f = f64_of(y)?;
    let r = if y_f.abs() > 0.5 {
        arb_erfcinv_small(&one_minus_ay, w)?
    } else {
        // Winitzki (a = 0.147), |relative error| < 2·10⁻³.
        let a = 0.147_f64;
        let l = (1.0 - y_f * y_f).ln();
        let t = 2.0 / (std::f64::consts::PI * a) + l / 2.0;
        let mut x0 = ((t * t - l / a).sqrt() - t).sqrt().copysign(y_f);
        for _ in 0..3 {
            let f = erf_f64(x0) - y_f;
            x0 -= f * (std::f64::consts::PI.sqrt() / 2.0) * (x0 * x0).exp();
        }
        // |y| ≤ 1/2 < erf(1/2): the root lies in (−1/2, 1/2).
        erf_inverse_halley(y, x0, -0.5, 0.5, false, w)?
    };
    Ok(if y.is_negative() && !r.is_negative() && y_f.abs() > 0.5 {
        r.neg()
    } else {
        r
    })
}

/// `erfcinv(u)` for `0 < u ≤ 1/2`, Halley iteration on `erfc`.  The initial
/// guess inverts the leading asymptotic `erfc(x) ≈ e^{−x²}/(x√π)`, which
/// also covers `u` far below the `f64` range; the root lies in
/// `(0, max(√L, 1)]` with `L = −ln u` (`erfc(0) = 1 > u`, and
/// `erfc(x) < e^{−x²}/(x√π) ≤ u` there).  (0.29 refined the guess by `f64`
/// Newton steps on `1 − erf_f64(x)`, a 50-term Taylor sum whose truncation
/// near `x = 4` is `10⁻⁶`: the steps diverged.)
pub(super) fn arb_erfcinv_small(u: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    // L = −ln u  (computed in BigFloat so tiny u is fine)
    let l = u.ln(wp, RM, w.cc).neg();
    let l_f = f64_of(&l)?;
    let mut x0 = l_f.max(0.5).sqrt();
    for _ in 0..4 {
        // x² = L − ln(x√π)
        let v = l_f - (x0 * std::f64::consts::PI.sqrt()).ln();
        x0 = v.max(0.25).sqrt();
    }
    let hi = l_f.max(0.0).sqrt().max(1.0) * (1.0 + f64::EPSILON * 16.0);
    erf_inverse_halley(u, x0, 0.0, hi, true, w)
}

/// `erfcinv(u)` for `0 < u < 2`: `erfcinv(u) = −erfcinv(2 − u)` maps to
/// `u ≤ 1`, then `erfinv(1 − u)` for `u ≥ 1/2` and the `erfc` Halley
/// iteration for smaller `u`.
pub(super) fn arb_erfcinv(u: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    if u.is_zero() || u.is_negative() || !bf_lt(u, &two) {
        return Err(unevaluable("erfcinv: argument must satisfy 0 < u < 2"));
    }
    if bf_gt(u, &one) {
        let v = two.sub(u, wp, RM);
        return Ok(arb_erfcinv(&v, w)?.neg());
    }
    let u_f = f64_of(u)?;
    if u_f >= 0.5 {
        let y = one.sub(u, wp, RM);
        return arb_erfinv(&y, w);
    }
    arb_erfcinv_small(u, w)
}
