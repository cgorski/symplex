//! Arbitrary-precision Gamma via Stirling series (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Arbitrary-precision Gamma via Stirling series
// ═══════════════════════════════════════════════════════════════════════════

/// Compute log Γ(z) for real z > 0 using the Stirling asymptotic series.
///
/// Algorithm:
/// 1. Argument reduction — shift z by integer r until z + r ≥ 0.5 · p.
/// 2. Stirling series:
///    log Γ(z) = (z − ½) ln(z) − z + ½ ln(2π) + Σ B₂ₖ / [2k(2k−1) z^{2k−1}].
/// 3. Undo shift:
///    log Γ(z) = log Γ(z + r) − ln Π_{i=0}^{r−1} (z + i).
///
/// The series' smallest term is about `e^{−2π(z+r)}`; at `z + r ≥ 0.5·p`
/// it is far below `2^{−p}` after about `0.2·p` terms (the Bernoulli
/// numbers `B₂ₖ`, exact rationals, dominate the cost: at `0.2·p` they took
/// `0.35·p` terms, and `Γ(1/3)` at 1,200 digits 20 s before 0.30).
pub(super) fn stirling_log_gamma(z: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    tracing::debug!(prec, "stirling_log_gamma: entry");

    let guard = 20; // extra guard bits
    let wp = prec + guard;

    // Determine how far to shift z for convergence.
    let z_approx = f64_of(z)?;
    let threshold = ((wp as f64) * 0.5).ceil() as i64;
    let shift = if z_approx < threshold as f64 {
        (threshold - z_approx.floor() as i64).max(0) as usize
    } else {
        0
    };

    tracing::trace!(
        z_approx,
        threshold,
        shift,
        "stirling_log_gamma: argument reduction"
    );

    // z_shifted = z + shift
    let shift_bf = BigFloat::from_i128(shift as i128, wp);
    let z_shifted = z.add(&shift_bf, wp, RM);

    // Main Stirling formula:
    //   log Γ(z) = (z − 1/2)·ln(z) − z + (1/2)·ln(2π) + Σ ...
    let half = BigFloat::from_f64(0.5, wp);
    let z_minus_half = z_shifted.sub(&half, wp, RM);
    let log_z = z_shifted.ln(wp, RM, w.cc);

    let mut result = z_minus_half.mul(&log_z, wp, RM);
    result = result.sub(&z_shifted, wp, RM);

    // (1/2) · ln(2π) at working precision.
    let two = BigFloat::from_i32(2, wp);
    let pi_val = w.cc.pi(wp, RM).clone();
    let two_pi = two.mul(&pi_val, wp, RM);
    let half_log_2pi = two_pi.ln(wp, RM, w.cc).mul(&half, wp, RM);
    result = result.add(&half_log_2pi, wp, RM);

    // Series: Σ_{k=1}^{N} B_{2k} / [2k(2k−1) · z^{2k−1}]
    let z_sq = z_shifted.mul(&z_shifted, wp, RM);
    let mut z_pow = z_shifted.clone(); // z^1 → z^3 → z^5 → …

    // Upper bound on useful terms: safely below the divergence point πz.
    let max_terms = ((wp as f64) * 0.35) as usize + 10;
    let mut prev_term_exp: Option<i32> = None;

    for k in 1..=max_terms {
        let b2k = bernoulli::even(k);
        if b2k.is_zero() {
            continue;
        }

        // Advance z_pow: z^1, z^3, z^5, …
        if k > 1 {
            z_pow = z_pow.mul(&z_sq, wp, RM);
        }

        // term = B_{2k} / (2k · (2k−1) · z^{2k−1})
        let b2k_bf = ratio_to_bigfloat(&b2k, wp);
        let denom_int = (2 * k * (2 * k - 1)) as i128;
        let denom = BigFloat::from_i128(denom_int, wp);
        let denom_full = denom.mul(&z_pow, wp, RM);
        let term = b2k_bf.div(&denom_full, wp, RM);

        // Divergence / convergence detection via binary exponents.
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                tracing::trace!(k, "stirling series: diverging, stopping");
                break;
            }
            prev_term_exp = Some(t_exp);

            // Term is negligible relative to accumulated result.
            if let Some(r_exp) = result.exponent()
                && (r_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "stirling series: converged");
                break;
            }
        }

        result = result.add(&term, wp, RM);
    }

    // Undo argument reduction:
    //   log Γ(z) = log Γ(z+shift) − ln Π_{i=0}^{shift−1} (z + i),
    // one logarithm of the product instead of one per factor (850
    // logarithms at 4,000 bits took a second).  The product carries the
    // roundings of its factors, `shift·2^−wp` relative, as the sum of
    // logarithms did.
    if shift > 0 {
        let mut product = BigFloat::from_i32(1, wp);
        for i in 0..shift {
            let z_plus_i = z.add(&BigFloat::from_i128(i as i128, wp), wp, RM);
            product = product.mul(&z_plus_i, wp, RM);
        }
        result = result.sub(&product.ln(wp, RM, w.cc), wp, RM);
    }

    Ok(result)
}

/// Compute Γ(x) at arbitrary precision for real x.
///
/// For positive x, evaluates exp(stirling_log_gamma(x)).
/// For negative non-integer x, applies the reflection formula
/// Γ(z) = π / (sin(πd) · (−1)ⁿ · Γ(1 − z)) with `z = n + d`, `n = round(z)`
/// and `d` exact, so that `sin` is accurate next to a pole (before 0.29 `πz`
/// was rounded first, losing `log₂(1/d)` bits, and arguments within
/// `10⁻¹²` of a pole — in `f64` — or `2^{−prec/2}` were refused as poles).
pub(super) fn arb_gamma_real(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    // The poles: the non-positive integers, exactly.
    if x.is_zero() || (x.is_negative() && x.is_int()) {
        return Err(infinite("Gamma at non-positive integer pole"));
    }

    if !x.is_negative() {
        // Direct Stirling path.
        tracing::debug!(prec, "arb_gamma_real: positive argument");
        let log_gamma = stirling_log_gamma(x, w)?;
        Ok(log_gamma.exp(prec, RM, w.cc))
    } else {
        tracing::debug!(prec, "arb_gamma_real: reflection formula");
        let exact = x.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
        let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
        let log_gamma_1mx = stirling_log_gamma(&one_minus_x, w)?;
        let gamma_1mx = log_gamma_1mx.exp(prec, RM, w.cc);

        // sin(πz) = (−1)ⁿ sin(πd).
        let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = x.sub(&n, exact, RM);
        let pi_val = w.cc.pi(prec, RM).clone();
        let mut sin_pi_x = pi_val.mul(&d, prec, RM).sin(prec, RM, w.cc);
        let odd = !n
            .div(&BigFloat::from_i32(2, 64), exact, RM)
            .fract()
            .is_zero();
        if odd {
            sin_pi_x = sin_pi_x.neg();
        }
        if sin_pi_x.is_zero() {
            return Err(infinite("Gamma at non-positive integer pole"));
        }

        let denom = sin_pi_x.mul(&gamma_1mx, prec, RM);
        Ok(pi_val.div(&denom, prec, RM))
    }
}
