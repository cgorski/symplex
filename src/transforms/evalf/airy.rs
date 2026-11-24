//! Airy functions (part of [`evalf`](super)).

use super::*;

// ── Airy functions ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum AiryKind {
    Ai,
    Bi,
    AiPrime,
    BiPrime,
}

/// Airy functions for real `x`.
///
/// * `|x|^{3/2} ≤ (3/4)·wp·ln 2`: Maclaurin series (DLMF 9.4.1–9.4.4)
///   `Ai = c₁ f − c₂ g`, `Bi = √3 (c₁ f + c₂ g)` with
///   `f = Σ x^{3k}/∏_{j≤k}(3j−1)(3j)`, `g = Σ x^{3k+1}/∏_{j≤k}(3j)(3j+1)`,
///   `c₁ = 3^{−2/3}/Γ(2/3)`, `c₂ = 3^{−1/3}/Γ(1/3)`; the working precision
///   carries `(4/3)|x|^{3/2} log₂ e` guard bits for the cancellation.
/// * beyond, with `ζ = (2/3)|x|^{3/2}`: the asymptotic expansions
///   DLMF 9.7.5–9.7.12 (`u_k`, `v_k = −(6k+1)/(6k−1) u_k`), whose optimal
///   truncation error is `~e^{−2ζ}`.
pub(super) fn arb_airy(
    x: &BigFloat,
    kind: AiryKind,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Airy function of special float value"));
    }
    let base_wp = prec + 32;
    let x_f = f64_of(x)?;
    let ax_f = x_f.abs();
    let zeta_f = 2.0 / 3.0 * ax_f.powf(1.5);

    if 2.0 * zeta_f > (base_wp as f64) * std::f64::consts::LN_2 {
        // The phase `(2/3)|x|^{3/2}` needs its bits before the point.
        let z_bits = 3 * x.exponent().map_or(0, i64::from).max(0) / 2;
        if !zeta_f.is_finite() || z_bits > i64::try_from(cancellation_cap(prec)).unwrap_or(i64::MAX)
        {
            return Err(special_exhausted(prec));
        }
        return Ok(round_to(airy_asymptotic(x, kind, &mut w.at(base_wp)), prec));
    }

    let wp = Prec::new(base_wp)
        .guarded(
            Guard::for_log2(2.0 * zeta_f * std::f64::consts::LOG2_E),
            Prec::MAX,
        )
        .ok_or_else(|| special_exhausted(prec))?
        .plus(16)
        .bits();
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let three = BigFloat::from_i32(3, wp);
    let third = one.div(&three, wp, RM);
    let two_thirds = third.mul(&BigFloat::from_i32(2, wp), wp, RM);
    let g13 = arb_gamma_real(&third, &mut w.at(wp))?;
    let g23 = arb_gamma_real(&two_thirds, &mut w.at(wp))?;
    let c1 = one.div(
        &bf_pow(&three, &two_thirds, &mut w.at(wp)).mul(&g23, wp, RM),
        wp,
        RM,
    );
    let c2 = one.div(
        &bf_pow(&three, &third, &mut w.at(wp)).mul(&g13, wp, RM),
        wp,
        RM,
    );

    // Series with term recurrences; the derivatives use
    // f'_k = 3k f_k / x, g'_k = (3k+1) g_k / x (k = 0 handled directly).
    let x3 = xw.mul(&xw, wp, RM).mul(&xw, wp, RM);
    let derivative = matches!(kind, AiryKind::AiPrime | AiryKind::BiPrime);
    let (mut f_sum, mut g_sum) = if derivative {
        (BigFloat::new(wp), one.clone()) // f'(0) = 0, g'(0) = 1
    } else {
        (one.clone(), xw.clone()) // f(0) = 1, g_0 = x
    };
    let mut fk = one.clone(); // x^{3k}/∏(3j−1)(3j)
    let mut gk = xw.clone(); // x^{3k+1}/∏(3j)(3j+1)
    // At x = 0 the k = 0 terms above are the exact values (every later
    // term vanishes, and the derivative recurrences would divide by x).
    let max_terms = if xw.is_zero() {
        0
    } else {
        (ax_f * 2.0) as usize + wp + 40
    };
    for k in 1..=max_terms {
        let k3 = 3 * k as i128;
        fk = fk
            .mul(&x3, wp, RM)
            .div(&BigFloat::from_i128((k3 - 1) * k3, wp), wp, RM);
        gk = gk
            .mul(&x3, wp, RM)
            .div(&BigFloat::from_i128(k3 * (k3 + 1), wp), wp, RM);
        let (tf, tg) = if derivative {
            // 3k x^{3k−1}/∏ = (3k/x) f_k ;  (3k+1) x^{3k}/∏ = ((3k+1)/x) g_k
            (
                fk.mul(&BigFloat::from_i128(k3, wp), wp, RM)
                    .div(&xw, wp, RM),
                gk.mul(&BigFloat::from_i128(k3 + 1, wp), wp, RM)
                    .div(&xw, wp, RM),
            )
        } else {
            (fk.clone(), gk.clone())
        };
        f_sum = f_sum.add(&tf, wp, RM);
        g_sum = g_sum.add(&tg, wp, RM);
        if negligible(&tf, &one, wp + 4) && negligible(&tg, &one, wp + 4) {
            break;
        }
    }
    let a = c1.mul(&f_sum, wp, RM);
    let b = c2.mul(&g_sum, wp, RM);
    let r = match kind {
        AiryKind::Ai | AiryKind::AiPrime => a.sub(&b, wp, RM),
        AiryKind::Bi | AiryKind::BiPrime => three.sqrt(wp, RM).mul(&a.add(&b, wp, RM), wp, RM),
    };
    Ok(round_to(r, prec))
}

/// Asymptotic expansions of the Airy functions for large `|x|`
/// (see [`arb_airy`]).  Returns the value at working precision `wp`.
pub(super) fn airy_asymptotic(x: &BigFloat, kind: AiryKind, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let ax = x.abs();
    let sqrt_ax = ax.sqrt(wp, RM);
    let x_quarter = sqrt_ax.sqrt(wp, RM); // |x|^{1/4}
    // `ζ = (2/3)|x|^{3/2}` enters `e^{±ζ}` and the phase `ζ + π/4` through
    // its absolute error: it is formed with the `Z` bits it has before the
    // point as guard (the caller refuses beyond `cancellation_cap`).
    // Before 0.30 it was formed at `wp` bits: `airyai(−467·10⁴⁰)` was wrong
    // from its 13th digit at 30, `airyai(−29/16·10⁷⁰)` entirely.
    let wz =
        wp + usize::try_from(3 * ax.exponent().map_or(0, i64::from).max(0) / 2 + 8).unwrap_or(0);
    let ax_z = at_precision(&ax, wz);
    let zeta = two.mul(&ax_z, wz, RM).mul(&ax_z.sqrt(wz, RM), wz, RM).div(
        &BigFloat::from_i32(3, wz),
        wz,
        RM,
    );
    let inv_zeta = one.div(&zeta, wp, RM);
    let sqrt_pi = w.cc.pi(wp, RM).clone().sqrt(wp, RM);
    let derivative = matches!(kind, AiryKind::AiPrime | AiryKind::BiPrime);

    // Coefficients c_k = u_k (or v_k) / ζ^k, u_k = u_{k−1}(6k−5)(6k−3)(6k−1)/(216 k (2k−1)),
    // v_k = −(6k+1)/(6k−1) u_k.  Optimal truncation: stop when |c_k| grows.
    let mut coeffs: Vec<BigFloat> = Vec::new();
    let mut u = one.clone();
    let mut zeta_pow = one.clone();
    let mut prev: Option<BigFloat> = None;
    for k in 0..(wp * 2 + 100) {
        if k > 0 {
            let k_i = k as i128;
            let num = BigFloat::from_i128((6 * k_i - 5) * (6 * k_i - 3) * (6 * k_i - 1), wp);
            let den = BigFloat::from_i128(216 * k_i * (2 * k_i - 1), wp);
            u = u.mul(&num, wp, RM).div(&den, wp, RM);
            zeta_pow = zeta_pow.mul(&inv_zeta, wp, RM);
        }
        let mut c = u.mul(&zeta_pow, wp, RM);
        if derivative {
            let k_i = k as i128;
            let ratio = BigFloat::from_i128(6 * k_i + 1, wp).div(
                &BigFloat::from_i128(6 * k_i - 1, wp),
                wp,
                RM,
            );
            c = c.mul(&ratio, wp, RM).neg();
        }
        if let Some(ref p) = prev
            && bf_gt(&c.abs(), p)
        {
            break;
        }
        let done = negligible(&c, &one, wp);
        prev = Some(c.abs());
        coeffs.push(c);
        if done {
            break;
        }
    }

    if !x.is_negative() {
        // Ai ~ e^{−ζ}/(2√π x^{1/4}) Σ (−1)^k u_k/ζ^k,   Bi ~ e^{ζ}/(√π x^{1/4}) Σ u_k/ζ^k
        // Ai' ~ −x^{1/4} e^{−ζ}/(2√π) Σ (−1)^k v_k/ζ^k, Bi' ~ x^{1/4} e^{ζ}/√π Σ v_k/ζ^k
        let mut sum = BigFloat::new(wp);
        let alternate = matches!(kind, AiryKind::Ai | AiryKind::AiPrime);
        for (k, c) in coeffs.iter().enumerate() {
            if alternate && k % 2 == 1 {
                sum = sum.sub(c, wp, RM);
            } else {
                sum = sum.add(c, wp, RM);
            }
        }
        let amp = match kind {
            AiryKind::Ai => zeta.neg().exp(wp, RM, w.cc).div(
                &two.mul(&sqrt_pi, wp, RM).mul(&x_quarter, wp, RM),
                wp,
                RM,
            ),
            AiryKind::Bi => zeta
                .exp(wp, RM, w.cc)
                .div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
            AiryKind::AiPrime => x_quarter
                .mul(&zeta.neg().exp(wp, RM, w.cc), wp, RM)
                .div(&two.mul(&sqrt_pi, wp, RM), wp, RM)
                .neg(),
            AiryKind::BiPrime => x_quarter
                .mul(&zeta.exp(wp, RM, w.cc), wp, RM)
                .div(&sqrt_pi, wp, RM),
        };
        return amp.mul(&sum, wp, RM);
    }

    // x < 0: with θ = ζ + π/4, P = Σ (−1)^k c_{2k}, Q = Σ (−1)^k c_{2k+1}:
    // Ai  ~ (sinθ P − cosθ Q)/(√π x^{1/4}),  Bi  ~ (cosθ P + sinθ Q)/(√π x^{1/4})
    // Ai' ~ −x^{1/4}(cosθ P + sinθ Q)/√π,  Bi' ~  x^{1/4}(sinθ P − cosθ Q)/√π
    let mut p = BigFloat::new(wp);
    let mut q = BigFloat::new(wp);
    for (k, c) in coeffs.iter().enumerate() {
        let sign_neg = (k / 2) % 2 == 1;
        let target = if k % 2 == 0 { &mut p } else { &mut q };
        *target = if sign_neg {
            target.sub(c, wp, RM)
        } else {
            target.add(c, wp, RM)
        };
    }
    let pi = w.cc.pi(wz, RM).clone();
    let theta = zeta.add(&pi.div(&BigFloat::from_i32(4, wz), wz, RM), wz, RM);
    let s = theta.sin(wp, RM, w.cc);
    let c = theta.cos(wp, RM, w.cc);
    let sp_minus_cq = s.mul(&p, wp, RM).sub(&c.mul(&q, wp, RM), wp, RM);
    let cp_plus_sq = c.mul(&p, wp, RM).add(&s.mul(&q, wp, RM), wp, RM);
    match kind {
        AiryKind::Ai => sp_minus_cq.div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
        AiryKind::Bi => cp_plus_sq.div(&sqrt_pi.mul(&x_quarter, wp, RM), wp, RM),
        AiryKind::AiPrime => x_quarter
            .mul(&cp_plus_sq, wp, RM)
            .div(&sqrt_pi, wp, RM)
            .neg(),
        AiryKind::BiPrime => x_quarter.mul(&sp_minus_cq, wp, RM).div(&sqrt_pi, wp, RM),
    }
}
