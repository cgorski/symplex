//! Euler-Mascheroni constant at arbitrary precision (Brent-McMillan B1) (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Mascheroni constant at arbitrary precision (Brent-McMillan B1)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Euler-Mascheroni constant γ at `prec` bits of precision
/// using the Brent-McMillan B1 algorithm.
///
/// This is the fastest known algorithm for computing γ.  It uses the identity
/// involving modified Bessel functions, simplified to a fixed-point summation
/// with convergence rate O(e^{-4n}) where n = 2^p.
///
/// Reference: Brent & McMillan (1980), mpmath `euler_fixed`.
pub(super) fn arb_euler_gamma(w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    // We need ln(2) at working precision.
    let extra = 30;
    let wp = prec + extra;

    // Choose p such that e^{-4·2^p} < 2^{-wp}, i.e. 4·2^p > wp·ln(2),
    // i.e. p > log2(wp·ln(2)/4).
    let p = ((wp as f64 / 4.0) * std::f64::consts::LN_2).log2().ceil() as u64 + 1;
    let n: i128 = 1i128 << p;
    let n_sq = n * n;

    // ln(2) at working precision.
    let two_bf = BigFloat::from_i32(2, wp);
    let ln2 = two_bf.ln(wp, RM, w.cc);

    // A = U = -p · ln(2)   (= -ln(n) since n = 2^p)
    let p_bf = BigFloat::from_i128(p as i128, wp);
    let neg_p_ln2 = p_bf.mul(&ln2, wp, RM).neg();

    let mut a = neg_p_ln2.clone();
    let mut u = neg_p_ln2;
    // B = V = 1
    let one = BigFloat::from_i32(1, wp);
    let mut b = one.clone();
    let mut v = one.clone();

    let n_sq_bf = BigFloat::from_i128(n_sq, wp);

    let mut k: i128 = 1;
    loop {
        let k_bf = BigFloat::from_i128(k, wp);
        let k_sq_bf = k_bf.mul(&k_bf, wp, RM);

        // B = B · n² / k²
        b = b.mul(&n_sq_bf, wp, RM).div(&k_sq_bf, wp, RM);

        // A = (A · n² / k + B) / k
        a = a.mul(&n_sq_bf, wp, RM).div(&k_bf, wp, RM);
        a = a.add(&b, wp, RM).div(&k_bf, wp, RM);

        u = u.add(&a, wp, RM);
        v = v.add(&b, wp, RM);

        // Convergence: both A and B are negligibly small.
        let a_small = a.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        let b_small = b.exponent().is_none_or(|e| e < -(wp as i32) + 10);
        if a_small && b_small {
            tracing::trace!(k, "arb_euler_gamma: converged");
            break;
        }

        k += 1;
        if k > 10 * (wp as i128) {
            return Err(SymplexError::ComputationFailed {
                operation: "euler_gamma",
                reason: "Brent-McMillan did not converge".into(),
            });
        }
    }

    // γ = U / V
    Ok(u.div(&v, prec, RM))
}

// ═══════════════════════════════════════════════════════════════════════════
// Catalan's constant
// ═══════════════════════════════════════════════════════════════════════════

/// Catalan's constant `G` at `prec` bits via Ramanujan's formula
///
/// ```text
/// G = (π/8)·ln(2 + √3) + (3/8)·Σ_{n≥0} 1 / ((2n+1)² · C(2n, n))
/// ```
///
/// The series gains ~2 bits per term (`1/C(2n,n) ~ 4^{-n}`), so about
/// `prec/2` terms are needed; every term is positive (no cancellation).
pub(super) fn arb_catalan(w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let three = BigFloat::from_i32(3, wp);
    let eight = BigFloat::from_i32(8, wp);

    // (π/8)·ln(2 + √3)
    let sqrt3 = three.sqrt(wp, RM);
    let ln_term = two.add(&sqrt3, wp, RM).ln(wp, RM, w.cc);
    let pi = w.cc.pi(wp, RM).clone();
    let first = pi.div(&eight, wp, RM).mul(&ln_term, wp, RM);

    // Σ b_n / (2n+1)² with b_n = 1/C(2n,n), b_{n+1} = b_n·(n+1)/(2(2n+1)).
    let mut b = one.clone();
    let mut sum = BigFloat::new(wp);
    let max_terms = wp / 2 + 64;
    for n in 0..max_terms {
        let two_n_plus_1 = BigFloat::from_i128(2 * n as i128 + 1, wp);
        let denom = two_n_plus_1.mul(&two_n_plus_1, wp, RM);
        let term = b.div(&denom, wp, RM);
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        let n_plus_1 = BigFloat::from_i128(n as i128 + 1, wp);
        let two_2n_plus_1 = two.mul(&two_n_plus_1, wp, RM);
        b = b.mul(&n_plus_1, wp, RM).div(&two_2n_plus_1, wp, RM);
    }
    let second = three.div(&eight, wp, RM).mul(&sum, wp, RM);
    let mut g = first.add(&second, wp, RM);
    g.set_precision(prec, RM)
        .map_err(|e| SymplexError::ComputationFailed {
            operation: "catalan",
            reason: format!("precision adjustment failed: {e:?}"),
        })?;
    Ok(g)
}
