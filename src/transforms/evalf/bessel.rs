//! Hankel P/Q series for Bessel asymptotic expansion (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Hankel P/Q series for Bessel asymptotic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Hankel auxiliary functions P_ν(z) and Q_ν(z) for the
/// Bessel asymptotic expansion (DLMF §10.17).
///
/// P_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k}(ν) / z^{2k}
/// Q_ν(z) = Σ_{k=0}^{N} (-1)^k · a_{2k+1}(ν) / z^{2k+1}
///
/// where a_k(ν) = (4ν²−1²)(4ν²−3²)⋯(4ν²−(2k−1)²) / (k! · 8^k)
/// (DLMF 10.17.1), so a_k = a_{k−1} · (4ν² − (2k−1)²) / (8k).
///
/// Uses optimal truncation: stops when terms start increasing (divergent
/// series).  The error is bounded by the first omitted term for real
/// positive z (DLMF §10.17(iii)).  The returned flag is `true` when the
/// truncation happened because the terms fell below `2^{-prec}` relative
/// to the sums — i.e. the expansion actually delivered `prec` bits.  When
/// it is `false` the series started diverging before reaching the target
/// and the caller must fall back to the power series.
fn hankel_pq(
    order: &BigFloat,
    z: &BigFloat,
    prec: usize,
    _cc: &mut Consts,
) -> (BigFloat, BigFloat, bool) {
    let one = BigFloat::from_i32(1, prec);
    let four = BigFloat::from_i32(4, prec);
    let eight = BigFloat::from_i32(8, prec);

    let four_nu_sq = four.mul(&order.mul(order, prec, RM), prec, RM);

    let z_inv = one.div(z, prec, RM);

    // P and Q accumulators; the k = 0 term (a_0 = 1) is added by the loop.
    let mut p_sum = BigFloat::new(prec);
    let mut q_sum = BigFloat::new(prec);

    let mut a_k = one.clone(); // a_0 = 1
    let mut z_power = one.clone(); // z^0 = 1

    // The terms reach their minimum near k ≈ 2|z|; allow a little slack.
    let max_terms = (3 * prec + 100).min(100_000);

    let mut prev_a_abs: Option<BigFloat> = None;
    let mut converged = false;

    for k in 0..max_terms {
        if k > 0 {
            // a_k = a_{k-1} · (4ν² - (2k-1)²) / (8k)
            let two_km1 = BigFloat::from_i32((2 * k as i32) - 1, prec);
            let two_km1_sq = two_km1.mul(&two_km1, prec, RM);
            let numer = four_nu_sq.sub(&two_km1_sq, prec, RM);
            let k_bf = BigFloat::from_i32(k as i32, prec);
            let denom = eight.mul(&k_bf, prec, RM);
            a_k = a_k.mul(&numer, prec, RM).div(&denom, prec, RM);
            // A half-integer order ends the expansion: `4ν² = (2k−1)²`
            // exactly, and every later coefficient has the factor 0 — the
            // sums are complete (spherical Bessel functions).  Before 0.30
            // the zero terms never met the convergence test, the expansion
            // was reported as failing, and the power series took over
            // with `1.44·x` guard bits: `besselj(5/2, 10²⁵)` printed
            // `3.8·10¹⁰⁵²⁷`, `besselj(−3/2, 12!)` hung.
            if a_k.is_zero() {
                converged = true;
                break;
            }

            // Update z_power: multiply by 1/z each step.
            z_power = z_power.mul(&z_inv, prec, RM);
        }

        // term = a_k / z^k
        let term = a_k.mul(&z_power, prec, RM);

        // Check for divergence: if |term| > |prev_term|, stop.
        let term_abs = term.abs();
        if let Some(ref prev) = prev_a_abs
            && term_abs.partial_cmp(prev) == Some(std::cmp::Ordering::Greater)
            && k > 2
        {
            tracing::trace!(k, "hankel_pq: optimal truncation (terms diverging)");
            break;
        }

        // Convergence: term negligible relative to accumulated sums.
        let p_exp = p_sum.exponent().unwrap_or(0);
        let q_exp = q_sum.exponent().unwrap_or(0);
        let ref_exp = p_exp.max(q_exp);
        let t_exp = term.exponent().unwrap_or(i32::MIN / 2);
        if k > 0 && (ref_exp as i64 - t_exp as i64) > prec as i64 {
            tracing::trace!(k, "hankel_pq: converged (term negligible)");
            converged = true;
            break;
        }

        prev_a_abs = Some(term_abs);

        // Even k → contributes to P with sign (-1)^(k/2)
        // Odd k → contributes to Q with sign (-1)^((k-1)/2)
        if k % 2 == 0 {
            // P term: (-1)^(k/2) · a_k / z^k
            if (k / 2) % 2 == 0 {
                p_sum = p_sum.add(&term, prec, RM);
            } else {
                p_sum = p_sum.sub(&term, prec, RM);
            }
        } else {
            // Q term: (-1)^((k-1)/2) · a_k / z^k
            if ((k - 1) / 2) % 2 == 0 {
                q_sum = q_sum.add(&term, prec, RM);
            } else {
                q_sum = q_sum.sub(&term, prec, RM);
            }
        }
    }

    (p_sum, q_sum, converged)
}

/// Smallest `|x|` at which the Hankel expansion is expected to reach `wp`
/// bits for order `ν`.
///
/// The smallest term of the (divergent) expansion is `~e^{−2|x|}` times a
/// factor that grows like `e^{2|ν|}` (the first `|ν|` coefficients are
/// inflated by `4ν²/(2j−1)²`), so `2|x| ≳ (wp + 3|ν|)·ln 2`.  This is
/// only a heuristic for choosing the algorithm; [`hankel_pq`] reports
/// whether it actually converged.
fn hankel_threshold(wp: usize, order_abs: f64) -> f64 {
    (wp as f64 + 3.0 * order_abs) * std::f64::consts::LN_2 / 2.0 + 4.0
}

/// Evaluate `J_ν(x)` (`want_j == true`) or `Y_ν(x)` via the Hankel
/// asymptotic expansion at `x > 0`,
///
/// ```text
/// J_ν(x) = √(2/(πx)) · [cos(ω)·P_ν(x) − sin(ω)·Q_ν(x)]
/// Y_ν(x) = √(2/(πx)) · [sin(ω)·P_ν(x) + cos(ω)·Q_ν(x)],   ω = x − νπ/2 − π/4
/// ```
///
/// Returns `None` when the expansion does not reach `wp` bits at this
/// `x` (the caller then uses the power series).
pub(super) fn bessel_jy_hankel(
    order: &BigFloat,
    x: &BigFloat,
    want_j: bool,
    w: &mut Wp<'_>,
) -> Option<BigFloat> {
    let wp = w.bits();
    // The phase ω = x − νπ/2 − π/4 loses the leading `log2 x` bits of x to
    // cancellation, so the trigonometric part needs that many extra bits
    // (the caller refuses beyond `cancellation_cap`).
    let wp = wp + x.exponent().unwrap_or(0).max(0) as usize;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let mut nu = order.clone();
    let _ = nu.set_precision(wp, RM);

    let (p_val, q_val, converged) = hankel_pq(&nu, &xw, wp, w.cc);
    if !converged {
        tracing::trace!("bessel_jy_hankel: expansion did not reach target precision");
        return None;
    }
    let pi = w.cc.pi(wp, RM).clone();
    let two = BigFloat::from_i32(2, wp);
    let four = BigFloat::from_i32(4, wp);

    let two_over_pi_x = two.div(&pi.mul(&xw, wp, RM), wp, RM);
    let amplitude = two_over_pi_x.sqrt(wp, RM);

    let nu_pi_half = nu.mul(&pi, wp, RM).div(&two, wp, RM);
    let pi_quarter = pi.div(&four, wp, RM);
    let phase = xw.sub(&nu_pi_half, wp, RM).sub(&pi_quarter, wp, RM);

    let cos_phase = phase.cos(wp, RM, w.cc);
    let sin_phase = phase.sin(wp, RM, w.cc);

    let combo = if want_j {
        cos_phase
            .mul(&p_val, wp, RM)
            .sub(&sin_phase.mul(&q_val, wp, RM), wp, RM)
    } else {
        sin_phase
            .mul(&p_val, wp, RM)
            .add(&cos_phase.mul(&q_val, wp, RM), wp, RM)
    };
    Some(amplitude.mul(&combo, wp, RM))
}

// ═══════════════════════════════════════════════════════════════════════════
// Bessel function evaluation (arbitrary precision)
// ═══════════════════════════════════════════════════════════════════════════

/// Arbitrary-precision Bessel function of the first kind J_ν(x).
///
/// * `|x|` below [`hankel_threshold`] (or whenever the asymptotic expansion
///   cannot reach the target precision): ascending power series
///   `J_ν(x) = Σ_k (-1)^k (x/2)^(ν+2k) / (k! Γ(ν+k+1))`.  Its terms grow
///   to `~e^{|x|}` before the alternating sum cancels down to `O(1)`, so
///   the working precision is raised by [`cancellation_guard_bits`] — this
///   makes the series correct at *any* `x`, merely slower for huge `x`.
/// * Otherwise the Hankel expansion with the full `P/Q` series
///   ([`bessel_jy_hankel`]), whose optimal-truncation error is `~e^{−2|x|}`.
pub(super) fn arb_bessel_j(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_j_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::J, order, x, w, classic)
}

/// Where the series and the Hankel expansion of the Bessel functions give
/// up (`classic` is an error), the uniform expansions of large order
/// (`bessel_debye`), at `abs(ν)` for `K_ν` and for the integer orders of
/// `I_ν` (`I_{−n} = I_n`) and at `abs(x)` for an integer order of `J_ν` and
/// `I_ν` (`f_n(−x) = (−1)ⁿ f_n(x)`); the error stands where they do not
/// apply.  Before 0.35 `bessely(20000, 300)`, `besselk(20000, 300)`,
/// `bessely(20000, 30000)`, `besseli(1000, 120000)` and `besselj(20000,
/// 19000)` were refused.
fn debye_fallback(
    kind: bessel_debye::Kind,
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
    classic: Result<BigFloat, SymplexError>,
) -> Result<BigFloat, SymplexError> {
    use bessel_debye::Kind;
    let Err(err) = classic else {
        return classic;
    };
    let integer = order.is_int();
    let negative_x = x.is_negative();
    // `J_{−ν}`, `Y_{−ν}` of a non-integer order from `J_ν` and `Y_ν`.
    if order.is_negative()
        && !integer
        && matches!(kind, Kind::J | Kind::Y)
        && !negative_x
        && let Some(v) = bessel_reflected(kind, &order.abs(), x, w)
    {
        return v;
    }
    let nu = match kind {
        Kind::K => order.abs(),
        Kind::I if integer => order.abs(),
        _ if order.is_negative() => return Err(err),
        _ => order.clone(),
    };
    if negative_x && !(integer && matches!(kind, Kind::J | Kind::I)) {
        return Err(err);
    }
    match bessel_jy_large(kind, &nu, &x.abs(), w) {
        Some(Ok(v)) if negative_x && bessel_order::is_odd(order) => Ok(v.neg()),
        Some(r) => r,
        None => Err(err),
    }
}

/// A Bessel function of large positive order `nu` at `x > 0` by the uniform
/// expansions (`bessel_debye`), and for `J`, `Y` where they stop short — the
/// turning point `x ≈ ν`, `Y_ν(x)` for `x < ν` — by the recurrence in the
/// order from where they do not (`bessel_recur`).  Before 0.37
/// `besselj(20000, 20000)`, `bessely(20000, 20100)` and `bessely(20000,
/// 14800)` were refused.
fn bessel_jy_large(
    kind: bessel_debye::Kind,
    nu: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Option<Result<BigFloat, SymplexError>> {
    use bessel_debye::Kind;
    match bessel_debye::debye(kind, nu, x, w) {
        Some(r) => Some(r),
        None if matches!(kind, Kind::J | Kind::Y) => bessel_recur::recur(kind, nu, x, w),
        None => None,
    }
}

/// `J_{−ν}(x) = cos(νπ)·J_ν(x) − sin(νπ)·Y_ν(x)` and
/// `Y_{−ν}(x) = sin(νπ)·J_ν(x) + cos(νπ)·Y_ν(x)` (DLMF 10.4.7, 10.4.8)
/// for a non-integer `ν > 0` and `x > 0`, `J_ν` and `Y_ν` by
/// [`bessel_jy_large`] (`None` where they are not available).  `νπ` is reduced exactly: `ν = n + f` with an integer `n`, the
/// signs `(−1)ⁿ`.  The cancellation of the two terms is measured and made up
/// with more bits.  Before 0.37 `besselj(−(1000 + 1/3), 2000)` was refused.
fn bessel_reflected(
    kind: bessel_debye::Kind,
    nu: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Option<Result<BigFloat, SymplexError>> {
    let prec = w.bits();
    use bessel_debye::Kind;
    let mut wp = prec + 32;
    let cap = 4 * prec + 1024;
    loop {
        let j = match bessel_jy_large(Kind::J, nu, x, &mut w.at(wp))? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let y = match bessel_jy_large(Kind::Y, nu, x, &mut w.at(wp))? {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        // ν = n + f, 0 < f < 1: cos(νπ) = (−1)ⁿ cos(fπ), sin(νπ) = (−1)ⁿ sin(fπ).
        let n = nu.floor();
        let f = nu.sub(&n, exact_bits(nu, wp), RM);
        let pi = w.cc.pi(wp + 16, RM).clone();
        let angle = f.mul(&pi, wp + 16, RM);
        let odd = bessel_order::is_odd(&n);
        let sign = |v: BigFloat| if odd { v.neg() } else { v };
        let (c, s) = (sign(angle.cos(wp, RM, w.cc)), sign(angle.sin(wp, RM, w.cc)));
        let (a, b) = match kind {
            Kind::J => (c.mul(&j, wp, RM), s.mul(&y, wp, RM).neg()),
            _ => (s.mul(&j, wp, RM), c.mul(&y, wp, RM)),
        };
        let v = a.add(&b, wp, RM);
        // The error is at most `2^(2 − wp)·(abs(J) + abs(Y))` (`J`, `Y` to
        // `2^(−wp)` relative, the sine and cosine to `2^(1 − wp)` absolute, the
        // products) and `2^(−wp)·abs(v)`: the bits lost against the larger
        // of `J`, `Y`.
        let big = j
            .exponent()
            .unwrap_or(i32::MIN / 2)
            .max(y.exponent().unwrap_or(i32::MIN / 2));
        let lost = if v.is_zero() {
            wp as i64
        } else {
            i64::from(big) - i64::from(v.exponent().unwrap_or(i32::MIN / 2)) + 1
        };
        if lost + 6 + (prec as i64) <= wp as i64 {
            return Some(Ok(round_to(v, prec)));
        }
        if wp >= cap {
            return None;
        }
        wp = (wp + lost.max(32) as usize).min(cap);
    }
}

/// [`arb_bessel_j`] by the ascending series and the Hankel expansion.
fn arb_bessel_j_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let guard = 32;
    let wp = prec + guard;

    // Special case: x = 0 (`J_0(0) = 1` for the order exactly 0: up to 0.53
    // an order below `10⁻¹⁵`, `besselj(e⁻¹⁰⁰, 0)`, gave 1, truly 0).
    if x.is_zero() {
        let order_f64 = f64_of(order)?;
        return if order.is_zero() {
            // J_0(0) = 1
            Ok(BigFloat::from_i32(1, prec))
        } else if order_f64 > 0.0 {
            // J_n(0) = 0 for n > 0
            Ok(BigFloat::new(prec))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselJ at x=0 with negative order".into(),
            })
        };
    }

    refuse_bessel_argument(x, prec)?;
    // `(x/2)^ν` of a negative `x` is complex for a non-integer order.
    // Before 0.30 the power series took `|x/2|^ν` and returned a real
    // number: `besselj(−1/2, −1)` was `0.431…`, truly `−0.431…·i`.
    if x.is_negative() && !order.is_int() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselJ of negative argument with non-integer order is complex".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;

    // Check for integer order (most common case) — exactly: before 0.29 an
    // order within 10⁻¹² of an integer was taken for it, and
    // `besselj(2 + 10⁻¹⁴, 1)` was wrong from its 15th digit.
    let order_int = order_f64.round() as i64;
    let is_int_order = order.is_int();
    let large = order_f64.abs() >= bessel_order::LARGE_ORDER;

    // Negative integer order: J_{-n}(x) = (-1)^n J_n(x) — the order negated
    // and its parity taken exactly (before 0.34 through an `i64`, which
    // saturates beyond 9.2·10¹⁸: `besselj(−10²⁰, x)` recursed without end).
    if is_int_order && order.is_negative() {
        let j = arb_bessel_j(&order.neg(), x, w)?;
        return Ok(if bessel_order::is_odd(order) {
            j.neg()
        } else {
            j
        });
    }

    // Large |x|: Hankel asymptotic expansion (J is even/odd in x for
    // integer order; non-integer order with x < 0 is complex and rejected).
    if x_f64.abs() >= hankel_threshold(wp, order_f64.abs()) {
        if x.is_negative() {
            if !is_int_order {
                return Err(SymplexError::Unevaluable {
                    reason: "BesselJ of negative argument with non-integer order is complex".into(),
                });
            }
            let j = arb_bessel_j(order, &x.abs(), w)?;
            return Ok(if bessel_order::is_odd(order) {
                j.neg()
            } else {
                j
            });
        }
        if let Some(j) = bessel_jy_hankel(order, x, true, &mut w.at(wp)) {
            return Ok(round_to(j, prec));
        }
    }

    // ── Ascending series with cancellation guard bits ──────────────────
    // J_ν(x) = (x/2)^ν · Σ_{k=0}^N (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
    {
        let wp = wp + series_guard_bits(x_f64.abs(), prec)?;
        let mut xw = x.clone();
        let _ = xw.set_precision(wp, RM);
        let mut order_w = order.clone();
        let _ = order_w.set_precision(wp, RM);
        let order = &order_w;
        let two = BigFloat::from_i32(2, wp);
        let x_half = xw.div(&two, wp, RM);
        let x_half_sq = x_half.mul(&x_half, wp, RM);
        let neg_x_half_sq = x_half_sq.neg();

        // Series: sum = Σ (-1)^k · (x/2)^{2k} / (k! · Γ(ν+k+1))
        // We track term = (-x²/4)^k / (k! · Γ(ν+k+1)) incrementally.
        // term_{k+1} = term_k · (-x²/4) / ((k+1) · (ν+k+1))
        let mut sum = BigFloat::new(wp); // will add 1/Γ(ν+1) as first term

        // A large order: the factor (x/2)^ν/Γ(ν+1) at once, its exponent
        // to an absolute 2^(−wp) (`bessel_order`), and the series from 1.
        // Before 0.34 an integer order took a multiplication per unit:
        // `besselj(8345185991999992, 61/10²⁷)` never finished.
        let (prefix, mut term) = if large {
            let bounds = bessel_order::series_bounds(order_f64, x_f64.abs(), true);
            // Below the range (only for `0 < abs(x) < ν`): 0, which the
            // accuracy layer holds as the underflow of a nonzero `J_ν`
            // (`J_ν` has no zero there, `accuracy::underflowed`) and
            // `extended` scales.  Before 0.34 `besselj(10⁸ + 1/2, 1)`
            // printed `0`; in 0.34 it was refused where it arose, and so was
            // `1 + besselj(10⁹, 1)`.
            let Some(p) = bessel_order::power_over_gamma(
                order,
                &x_half.abs(),
                bounds,
                "besselj",
                &mut w.at(wp),
            )?
            else {
                return Ok(BigFloat::new(prec));
            };
            // J_n(−x) = (−1)ⁿ J_n(x) (a negative x has an integer order here).
            let p = if x.is_negative() && bessel_order::is_odd(order) {
                p.neg()
            } else {
                p
            };
            (p, BigFloat::from_i32(1, wp))
        } else {
            // Compute (x/2)^ν.  For integer order, use repeated multiplication.
            let prefix = if is_int_order && order_int >= 0 {
                let mut p = BigFloat::from_i32(1, wp);
                for _ in 0..order_int {
                    p = p.mul(&x_half, wp, RM);
                }
                p
            } else {
                // General: (x/2)^ν = exp(ν · ln(x/2))
                let ln_xh = x_half.abs().ln(wp, RM, w.cc);
                let nu_ln = order.mul(&ln_xh, wp, RM);
                nu_ln.exp(wp, RM, w.cc)
            };
            // First term (k=0): 1 / Γ(ν+1)
            let gamma_nu1 = arb_gamma_real(
                &order.add(&BigFloat::from_i32(1, wp), wp, RM),
                &mut w.at(wp),
            )?;
            (prefix, BigFloat::from_i32(1, wp).div(&gamma_nu1, wp, RM))
        };
        sum = sum.add(&term, wp, RM);

        // The terms peak near k ≈ |x|/2 and then decay super-exponentially;
        // `|x| + wp` comfortably covers the tail down to 2^{-wp}.
        let max_terms = (x_f64.abs() * 1.5) as usize + wp + 40;
        for k in 1..=max_terms {
            // term *= (-x²/4) / (k · (ν + k))
            let k_bf = BigFloat::from_i32(k as i32, wp);
            let nu_plus_k = order.add(&k_bf, wp, RM);
            let denom = k_bf.mul(&nu_plus_k, wp, RM);
            term = term.mul(&neg_x_half_sq, wp, RM);
            term = term.div(&denom, wp, RM);

            // Convergence check (only once past the peak of the terms).
            if (k as f64) > x_f64.abs() / 2.0
                && let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "arb_bessel_j: series converged");
                break;
            }

            sum = sum.add(&term, wp, RM);
        }

        let r = prefix.mul(&sum, wp, RM);
        if large && (r.is_zero() || r.is_inf()) {
            return Err(SymplexError::Unevaluable {
                reason: "besselj of large order: the value leaves the arbitrary-precision \
                         exponent range"
                    .into(),
            });
        }
        Ok(round_to(r, prec))
    }
}

/// Arbitrary-precision Bessel function of the second kind Y_ν(x), `x > 0`.
///
/// * Large `x` (see [`hankel_threshold`]): Hankel expansion
///   ([`bessel_jy_hankel`]), falling back to the series if it cannot reach
///   the target precision.
/// * Otherwise, integer order `n` only: the Neumann series (A&S 9.1.11)
///   with guard bits for the `e^{x}` cancellation.
pub(super) fn arb_bessel_y(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_y_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::Y, order, x, w, classic)
}

/// `Y_ν(x)`, `x > 0`, for a non-integer order below the Hankel range, from
/// the Bessel functions of the first kind (DLMF 10.2.3):
/// `Y_ν = (J_ν cos νπ − J_{−ν})/sin νπ`, and for a half-integer order
/// `ν = n + ½` exactly `Y_ν = (−1)^{n+1} J_{−ν}` (`cos νπ = 0`; a finite
/// trigonometric form, DLMF 10.16.1: `Y_{1/2}(x) = −√(2/(πx)) cos x`).
/// The difference cancels where `sin νπ` is small or `J_ν cos νπ ≈
/// J_{−ν}`: it is evaluated again with the bits lost.  Before 0.54 refused
/// ("not yet implemented in small-|x| regime": `bessely(1/2, 2)`).
fn bessel_y_from_j(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let exact = order.mantissa_max_bit_len().unwrap_or(64) + 8;
    let twice = order.mul(&BigFloat::from_i32(2, 8), exact, RM);
    if twice.is_int() {
        // ν = n + ½ with 2ν = 2n + 1: (−1)^{n+1} is −1 for 2ν ≡ 1 (mod 4).
        let Some(t) = bf_as_int(&twice)? else {
            return Err(unevaluable("bessely: half-integer order out of range"));
        };
        let j = arb_bessel_j(&order.neg(), x, w)?;
        return Ok(if t.rem_euclid(4) == 1 { j.neg() } else { j });
    }
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        let mut ww = w.at(wp);
        let jp = arb_bessel_j(order, x, &mut ww)?;
        let jm = arb_bessel_j(&order.neg(), x, &mut ww)?;
        let pi = ww.cc.pi(wp, RM).clone();
        let t = order.mul(&pi, wp, RM);
        let (s, c) = (t.sin(wp, RM, ww.cc), t.cos(wp, RM, ww.cc));
        let jc = jp.mul(&c, wp, RM);
        let num = jc.sub(&jm, wp, RM);
        let e = |v: &BigFloat| v.exponent().map_or(i64::MIN / 4, i64::from);
        // Bits lost in the difference, and in `sin νπ` (its argument's
        // rounding is relative to `νπ`, not to the sine).
        let lost = (e(&jc).max(e(&jm)) - e(&num)).max(0) + (e(&t) - e(&s)).max(0) + 4;
        let lost = usize::try_from(lost).unwrap_or(usize::MAX / 8);
        if lost + 16 <= extra {
            return Ok(round_to(num.div(&s, wp, RM), prec));
        }
        if lost > 4 * prec + 512 {
            break;
        }
        extra = lost + 32;
    }
    Err(unevaluable(
        "bessely: J_ν cos νπ − J_{−ν} cancels beyond the working precision",
    ))
}

/// [`arb_bessel_y`] by the Hankel expansion and the Neumann series.
fn arb_bessel_y_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let guard = 32;
    let wp = prec + guard;

    if x.is_zero() || x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselY undefined at x ≤ 0".into(),
        });
    }

    refuse_bessel_argument(x, prec)?;
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`): a non-integer order near an integer is
    // refused below the Hankel range rather than taken for the integer.
    let is_int_order = order.is_int();

    // Negative integer order: Y_{-n}(x) = (-1)^n Y_n(x) (exactly, see
    // `arb_bessel_j`).
    if is_int_order && order.is_negative() {
        let y = arb_bessel_y(&order.neg(), x, w)?;
        return Ok(if bessel_order::is_odd(order) {
            y.neg()
        } else {
            y
        });
    }

    if x_f64 >= hankel_threshold(wp, order_f64.abs())
        && let Some(y) = bessel_jy_hankel(order, x, false, &mut w.at(wp))
    {
        tracing::trace!("arb_bessel_y: used Hankel P/Q expansion");
        return Ok(round_to(y, prec));
    }

    // ── Small |x|, integer order n ≥ 0: Neumann series (A&S 9.1.11) ─────
    //
    //   Y_n(x) = −(1/π)(x/2)^{−n} Σ_{k=0}^{n−1} (n−k−1)!/k! (x²/4)^k
    //            + (2/π) ln(x/2) J_n(x)
    //            − (1/π)(x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    //
    // with ψ(m+1) = −γ + H_m.  Computed directly for every n (no forward
    // recurrence or Wronskian division, which is unstable near zeros of J).
    // The log term and the series cancel to O(e^{x}) relative size, so a
    // few extra guard bits are added.

    if !is_int_order {
        return bessel_y_from_j(order, x, w);
    }
    // A large order: the finite part of the series alone where the rest is
    // negligible (`bessel_order::neumann_leading`).  Before 0.34 the series
    // took `n` steps in exact rationals: `bessely(5000, 1)` 17 s,
    // `bessely(10⁵, 1)` never finished.
    if order_f64 >= bessel_order::LARGE_ORDER {
        return bessel_order::neumann_leading(bessel_order::Second::Y, order, x, w)
            .unwrap_or_else(|| Err(large_order_refused("bessely", order_f64, x_f64)));
    }
    let n = order_int as usize;
    // Beyond the cap of the cancellation, the finite part of the series alone
    // where the rest is negligible (`bessel_order::neumann_leading`): before
    // 0.35 `bessely(3909, 3909/2)` was refused.
    let guard = match series_guard_bits(x_f64, prec) {
        Ok(g) => g,
        Err(e) => {
            return bessel_order::neumann_leading(bessel_order::Second::Y, order, x, w)
                .unwrap_or(Err(e));
        }
    };
    let wp = wp + guard;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);

    let pi = w.cc.pi(wp, RM).clone();
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let inv_pi = one.div(&pi, wp, RM);
    let x_half = xw.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let ln_x_half = x_half.ln(wp, RM, w.cc);
    let euler_gamma = arb_euler_gamma(&mut w.at(wp))?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let j_n = arb_bessel_j(&n_bf, &xw, &mut w.at(wp))?;

    // Finite sum: −(1/π)(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        let mut f = NeumannCoefficients::new(n);
        let mut pow = one.clone();
        for k in 0..n {
            if k > 0 {
                pow = pow.mul(&x_half_sq, wp, RM);
            }
            let coeff = f.next(wp);
            finite = finite.add(&coeff.mul(&pow, wp, RM), wp, RM);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, RM), wp, RM);
        finite = inv_pi
            .mul(&x_half_pow_neg_n, wp, RM)
            .mul(&finite, wp, RM)
            .neg();
    }

    // Log term: (2/π) ln(x/2) J_n(x)
    let log_term = two
        .mul(&inv_pi, wp, RM)
        .mul(&ln_x_half, wp, RM)
        .mul(&j_n, wp, RM);

    // Series: −(1/π)(x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (−x²/4)^k / (k!(n+k)!)
    let mut h_k = BigFloat::new(wp);
    let mut h_nk = BigFloat::new(wp);
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, RM),
            wp,
            RM,
        );
    }
    let n_fact = factorial_exact(n);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp), wp, RM);
    let neg_x_half_sq = x_half_sq.neg();
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = euler_gamma.mul(&two, wp, RM);
    let max_terms = (x_f64.abs() * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, RM), wp, RM);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, RM), wp, RM);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, RM), wp, RM);
            pow = pow.mul(&neg_x_half_sq, wp, RM);
        }
        let psi_sum = h_k.add(&h_nk, wp, RM).sub(&two_gamma, wp, RM);
        let term = psi_sum.mul(&pow, wp, RM).mul(&inv_fact, wp, RM);
        series = series.add(&term, wp, RM);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            tracing::trace!(k, "arb_bessel_y: Neumann series converged");
            break;
        }
    }
    let series_term = inv_pi
        .mul(&x_half.powi(n, wp, RM), wp, RM)
        .mul(&series, wp, RM)
        .neg();

    let y = finite.add(&log_term, wp, RM).add(&series_term, wp, RM);
    Ok(round_to(y, prec))
}

/// `n!` as an exact rational (an integer): a product of integers.  Not
/// `Ratio` arithmetic, whose every product is reduced by a gcd with the
/// denominator 1 — a binary gcd that takes a step per bit of the numerator,
/// `O(n²)` words for `n!`: before 0.34 `polygamma(1755, 1/10)` spent 5 s
/// in it in the fuzz build, `bessely(5000, 1)` 17 s.  The same value.
pub(super) fn factorial_exact(n: usize) -> Ratio<BigInt> {
    Ratio::from_integer(crate::base::combinatorics::factorial(n as u64))
}

/// The coefficients `f_k = (n−k−1)!/k!`, `k = 0 … n−1`, of the finite sums of
/// `Y_n` and `K_n` (A&S 9.1.11, 9.6.11), as BigFloats: exactly the rounding of
/// the reduced fraction `ratio_to_bigfloat` made of the `Ratio` recurrence
/// `f_k = f_{k−1}/((n−k)·k)` before 0.34, without its gcds.  The reduced
/// fraction is an integer `(k+1)⋯(n−k−1)` while `k ≤ n−k−1` (`k!` divides
/// `(n−k−1)!`) and the unit fraction `1/((n−k)⋯k)` after: integer
/// recurrences with exact divisions and products.
pub(super) struct NeumannCoefficients {
    n: usize,
    k: usize,
    /// `(n−k−1)!/k!` in the integer phase, the denominator in the other.
    value: BigInt,
}

impl NeumannCoefficients {
    pub(super) fn new(n: usize) -> Self {
        let value = crate::base::combinatorics::factorial(n.saturating_sub(1) as u64);
        NeumannCoefficients { n, k: 0, value }
    }

    /// `f_k` at `wp` bits, then `k + 1`.
    pub(super) fn next(&mut self, wp: usize) -> BigFloat {
        let (n, k) = (self.n, self.k);
        self.k += 1;
        if k > 0 {
            let (a, b) = (BigInt::from((n - k) as u64), BigInt::from(k as u64));
            if 2 * k < n {
                // (n−k)!/(k−1)! ÷ ((n−k)·k): exact.
                self.value = &self.value / (a * b);
            } else if 2 * k < n + 2 {
                // The first unit fraction: 1/k (n = 2k), 1/((k−1)·k) (n = 2k−1).
                self.value = (n - k..=k).map(|j| BigInt::from(j as u64)).product();
            } else {
                self.value = &self.value * a * b;
            }
        }
        if 2 * k < n {
            ratio_to_bigfloat(&Ratio::from_integer(self.value.clone()), wp)
        } else {
            ratio_to_bigfloat(&Ratio::new_raw(BigInt::from(1), self.value.clone()), wp)
        }
    }
}

/// `Y_n` or `K_n` of a large order where [`bessel_order::neumann_leading`]
/// does not apply (`x² > 2(n − 1)`): the full series would take `n` steps.
fn large_order_refused(what: &str, order: f64, x: f64) -> SymplexError {
    SymplexError::NotImplemented(format!(
        "{what} of integer order {order:e} at {x:e} (above sqrt(2n)): the Neumann series would \
         take n terms (no uniform asymptotic expansion)"
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Modified Bessel functions
// ═══════════════════════════════════════════════════════════════════════════

/// Ascending series for `I_ν(x)` at working precision `wp`:
/// `I_ν(x) = (x/2)^ν Σ_{k≥0} (x²/4)^k / (k! Γ(ν+k+1))`.
///
/// All terms are positive for `x > 0`, so no cancellation occurs; the
/// number of terms is `O(|x| + wp)`.
fn bessel_i_series(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let two = BigFloat::from_i32(2, wp);
    let x_half = x.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`).
    let is_int_order = order.is_int();

    if x.is_zero() {
        return if is_int_order && order_int == 0 {
            Ok(BigFloat::from_i32(1, wp))
        } else if order_f64 > 0.0 {
            Ok(BigFloat::new(wp))
        } else {
            Err(SymplexError::Unevaluable {
                reason: "BesselI at x=0 with negative order".into(),
            })
        };
    }

    if !is_int_order && x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselI of negative argument with non-integer order is complex".into(),
        });
    }
    let one = BigFloat::from_i32(1, wp);
    let x_f64 = f64_of(x)?.abs();
    // A large order (I₋ₙ = Iₙ): the factor (x/2)^ν/Γ(ν+1) at once, its
    // exponent to an absolute 2^(−wp) (`bessel_order`), the series from 1 —
    // refused when it would be long.  Before 0.34 an order beyond 2⁶³
    // saturated an `i64`, and `besseli(10¹², 10¹¹)` summed for ever.
    let large = order_f64.abs() >= bessel_order::LARGE_ORDER;
    let (prefix, eff_order, mut term) = if large {
        let eff_order = if is_int_order {
            order.abs()
        } else {
            order.clone()
        };
        let eff_f64 = f64_of(&eff_order)?;
        let bounds = bessel_order::series_bounds(eff_f64, x_f64, false);
        // Below the range: 0, which the accuracy layer holds as the
        // underflow of a nonzero `I_ν` (`underflowed`), as before 0.34.
        let Some(p) =
            bessel_order::power_over_gamma(&eff_order, &x_half.abs(), bounds, "besseli", w)?
        else {
            return Ok(BigFloat::new(wp));
        };
        bessel_order::refuse_long_i_series(eff_f64, x_f64)?;
        // I_n(−x) = (−1)ⁿ I_n(x).
        let p = if x.is_negative() && bessel_order::is_odd(order) {
            p.neg()
        } else {
            p
        };
        (p, eff_order, one.clone())
    } else {
        // (x/2)^ν
        let prefix = if is_int_order && order_int >= 0 {
            x_half.powi(order_int as usize, wp, RM)
        } else if is_int_order {
            // Negative integer order: I_{-n} = I_n.
            x_half.powi((-order_int) as usize, wp, RM)
        } else {
            let ln_xh = x_half.ln(wp, RM, w.cc);
            order.mul(&ln_xh, wp, RM).exp(wp, RM, w.cc)
        };
        let eff_order = if is_int_order && order_int < 0 {
            BigFloat::from_i128((-order_int) as i128, wp)
        } else {
            order.clone()
        };
        // term_0 = 1/Γ(ν+1)
        let gamma_nu1 = arb_gamma_real(&eff_order.add(&one, wp, RM), w)?;
        let term = one.div(&gamma_nu1, wp, RM);
        (prefix, eff_order, term)
    };

    // term_{k+1} = term_k · (x²/4) / ((k+1)(ν+k+1))
    let mut sum = term.clone();
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 1..=max_terms {
        let k_bf = BigFloat::from_i128(k as i128, wp);
        let nu_plus_k = eff_order.add(&k_bf, wp, RM);
        let denom = k_bf.mul(&nu_plus_k, wp, RM);
        term = term.mul(&x_half_sq, wp, RM).div(&denom, wp, RM);
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    let r = prefix.mul(&sum, wp, RM);
    if large && (r.is_zero() || r.is_inf()) {
        return Err(SymplexError::Unevaluable {
            reason: "besseli of large order: the value leaves the arbitrary-precision exponent \
                     range"
                .into(),
        });
    }
    Ok(r)
}

/// Asymptotic expansion of `I_ν(x)` for large `x > 0` (DLMF 10.40.1):
///
/// ```text
/// I_ν(x) ~ e^x/√(2πx) · Σ_k (−1)^k a_k(ν)/x^k,
/// a_k(ν) = (4ν² − 1²)(4ν² − 3²)⋯(4ν² − (2k−1)²) / (k!·8^k),
/// ```
///
/// the coefficients of [`hankel_pq`].  The part it leaves out is `~e^{−2x}`
/// relative to the value, below `2^{−wp}` from [`hankel_threshold`] on;
/// the sum stops at a term below `2^{−wp}` relative (or ends, for a
/// half-integer order).  `None` when the terms start growing first (an
/// order large next to `x`).
fn bessel_i_asymptotic(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Option<BigFloat> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let eight = BigFloat::from_i32(8, wp);
    let four_nu_sq = BigFloat::from_i32(4, wp).mul(&order.mul(order, wp, RM), wp, RM);
    let inv_x = one.div(x, wp, RM);
    let mut a = one.clone(); // a_k/x^k
    let mut sum = one.clone();
    let mut prev = one.clone();
    let mut converged = false;
    for k in 1..(4 * wp + 100) {
        let two_km1 = BigFloat::from_i128(2 * k as i128 - 1, wp);
        let numer = four_nu_sq.sub(&two_km1.mul(&two_km1, wp, RM), wp, RM);
        let denom = eight.mul(&BigFloat::from_i128(k as i128, wp), wp, RM);
        a = a
            .mul(&numer, wp, RM)
            .div(&denom, wp, RM)
            .mul(&inv_x, wp, RM);
        if a.is_zero() {
            converged = true;
            break;
        }
        // A growing term, or one beyond the leading 1: an alternating sum
        // that would cancel bits the working precision does not carry.
        let size = a.abs();
        if size.cmp(&prev).is_some_and(|c| c > 0) || size.exponent().is_some_and(|e| e > 0) {
            break;
        }
        sum = if k % 2 == 1 {
            sum.sub(&a, wp, RM)
        } else {
            sum.add(&a, wp, RM)
        };
        if let (Some(t), Some(s)) = (a.exponent(), sum.exponent())
            && i64::from(s) - i64::from(t) > wp as i64
        {
            converged = true;
            break;
        }
        prev = size;
    }
    if !converged {
        return None;
    }
    let two_pi_x =
        w.cc.pi(wp, RM)
            .mul(&BigFloat::from_i32(2, wp), wp, RM)
            .mul(x, wp, RM);
    let amplitude = two_pi_x.sqrt(wp, RM).reciprocal(wp, RM);
    Some(
        x.exp(wp, RM, w.cc)
            .mul(&amplitude, wp, RM)
            .mul(&sum, wp, RM),
    )
}

/// Modified Bessel function of the first kind `I_ν(x)` (real `ν`, `x`).
///
/// * `|x|` from [`hankel_threshold`] on: the asymptotic expansion
///   ([`bessel_i_asymptotic`]), with `I_n(−x) = (−1)ⁿ I_n(x)`;
/// * otherwise, or where the expansion does not converge, the ascending
///   series ([`bessel_i_series`], about `|x| + wp` terms), refused beyond
///   [`cancellation_cap`] terms.
///
/// Before 0.30 the series served every `x`: `besseli(5/2, 5.2·10⁷⁰)`
/// printed `2.18·10⁴⁴⁸⁵⁷` (truly beyond the exponent range),
/// `besseli(2, 10¹⁶)` and `besseli(0, 32/15·10⁸)` hung.
pub(super) fn arb_bessel_i(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_i_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::I, order, x, w, classic)
}

/// [`arb_bessel_i`] by the asymptotic expansion and the ascending series.
fn arb_bessel_i_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let wp = prec + 32;
    if x.is_inf() {
        return Err(special_exhausted(prec));
    }
    let x_abs = f64_of(x)?.abs();
    let order_f64 = f64_of(order)?;
    if x_abs >= hankel_threshold(wp, order_f64.abs()) {
        if x.is_negative() {
            if !order.is_int() {
                return Err(SymplexError::Unevaluable {
                    reason: "BesselI of negative argument with non-integer order is complex".into(),
                });
            }
            let v = arb_bessel_i(order, &x.abs(), w)?;
            return Ok(if bessel_order::is_odd(order) {
                v.neg()
            } else {
                v
            });
        }
        if let Some(v) = bessel_i_asymptotic(order, x, &mut w.at(wp)) {
            return Ok(round_to(v, prec));
        }
        if !x_abs.is_finite() || x_abs > cancellation_cap(prec) as f64 {
            return Err(special_exhausted(prec));
        }
    }
    let r = bessel_i_series(order, x, &mut w.at(wp))?;
    Ok(round_to(r, prec))
}

/// Asymptotic expansion for `K_ν(x)`, large `x > 0` (DLMF 10.40.2):
/// `K_ν(x) ~ √(π/(2x)) e^{−x} Σ_k a_k(ν)/x^k`,
/// `a_k(ν) = ∏_{j=1}^{k} (4ν² − (2j−1)²) / (k! 8^k)`.
///
/// While `(2k−1)² < 4ν²` the terms are positive and may grow (an order
/// large next to `x`); past that they alternate, and the first one that
/// grows marks the divergence of the expansion.  `None` when the terms do
/// not fall below `2^{−wp}` relative before that (the caller uses the
/// series); a half-integer order ends the expansion exactly.  Before 0.30
/// the first growing term stopped the sum, whatever the order, and the
/// partial sum was returned: `besselk(20, 60)` was `6.14·10⁻²⁷` (truly
/// `3.75·10⁻²⁶`), `besselk(50, 100)` `6.29·10⁻⁴⁴` (truly
/// `9.27·10⁻⁴⁰`), both certified.
fn bessel_k_asymptotic(order: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Option<BigFloat> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    let four = BigFloat::from_i32(4, wp);
    let eight = BigFloat::from_i32(8, wp);
    let four_nu_sq = four.mul(&order.mul(order, wp, RM), wp, RM);
    let inv_x = one.div(x, wp, RM);
    let mut a_k = one.clone();
    let mut sum = one.clone();
    let mut prev: Option<BigFloat> = None;
    let mut converged = false;
    let max_terms = wp * 2 + 100;
    for k in 1..max_terms {
        let two_km1 = BigFloat::from_i128(2 * k as i128 - 1, wp);
        let two_km1_sq = two_km1.mul(&two_km1, wp, RM);
        let numer = four_nu_sq.sub(&two_km1_sq, wp, RM);
        let denom = eight.mul(&BigFloat::from_i128(k as i128, wp), wp, RM);
        a_k = a_k
            .mul(&numer, wp, RM)
            .div(&denom, wp, RM)
            .mul(&inv_x, wp, RM);
        if a_k.is_zero() {
            converged = true;
            break;
        }
        let alternating = two_km1_sq.cmp(&four_nu_sq).is_some_and(|c| c > 0);
        if alternating
            && let Some(ref p) = prev
            && a_k.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        sum = sum.add(&a_k, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (a_k.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            converged = true;
            break;
        }
        prev = Some(a_k.abs());
    }
    if !converged {
        return None;
    }
    let pi = w.cc.pi(wp, RM).clone();
    let two = BigFloat::from_i32(2, wp);
    let amp = pi.div(&two.mul(x, wp, RM), wp, RM).sqrt(wp, RM);
    let e_neg_x = x.neg().exp(wp, RM, w.cc);
    Some(amp.mul(&e_neg_x, wp, RM).mul(&sum, wp, RM))
}

/// Modified Bessel function of the second kind `K_ν(x)`, `x > 0`.
///
/// * Large `x` (`2x > wp·ln 2`): asymptotic expansion.
/// * Integer order `n` (A&S 9.6.11):
///   `K_n(x) = ½(x/2)^{−n} Σ_{k<n} (n−k−1)!/k! (−x²/4)^k + (−1)^{n+1} ln(x/2) I_n(x)
///           + (−1)^n ½ (x/2)^n Σ_{k≥0} [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)`
///   with `ψ(m+1) = −γ + H_m`.
/// * Non-integer order: `K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ)`.
///
/// Both series formulas suffer `e^{x}`-scale cancellation (the result is
/// `~e^{−x}`), which is absorbed by extra guard bits.
pub(super) fn arb_bessel_k(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let classic = arb_bessel_k_classic(order, x, w);
    debye_fallback(bessel_debye::Kind::K, order, x, w, classic)
}

/// [`arb_bessel_k`] by the asymptotic expansion and the series.
fn arb_bessel_k_classic(
    order: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_zero() || x.is_negative() || x.is_nan() {
        return Err(SymplexError::Unevaluable {
            reason: "BesselK undefined at x ≤ 0".into(),
        });
    }
    let base_wp = prec + 32;
    let x_f64 = f64_of(x)?;
    let order_f64 = f64_of(order)?;
    let order_int = order_f64.round() as i64;
    // Exactly (see `arb_bessel_j`); a non-integer order near an integer is
    // paid for with precision below.
    let is_int_order = order.is_int();
    // |n| exactly (before 0.34 through a saturating `i64`).
    let order_abs = if is_int_order && order_f64.abs() < bessel_order::LARGE_ORDER {
        BigFloat::from_i128(order_int.unsigned_abs() as i128, base_wp)
    } else {
        order.abs()
    };

    // Large x: asymptotic (relative truncation error ~ e^{-2x}).
    if 2.0 * x_f64 > (base_wp as f64) * std::f64::consts::LN_2
        && let Some(r) = bessel_k_asymptotic(&order_abs, x, &mut w.at(base_wp))
    {
        return Ok(round_to(r, prec));
    }

    // Series: guard against e^{x} cancellation (two exponentially large
    // terms of opposite sign nearly cancel).
    let wp = base_wp + 2 * series_guard_bits(x_f64, prec)?;
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);

    if !is_int_order {
        // K_ν = (π/2)(I_{−ν} − I_ν)/sin(νπ), with sin(νπ) = (−1)ⁿ sin(dπ),
        // ν = n + d exactly.  Near an integer order the difference cancels
        // log₂(1/|d|) bits (and more against e^{x}): the loss is measured
        // and the series repeated with that many more bits, up to
        // `cancellation_cap`.  (Before 0.29 an order within 10⁻¹² of an
        // integer was taken for it — `besselk(1 + 10⁻¹⁴, 1)` was wrong from
        // its 15th digit — and one a little further lost its bits unseen.)
        let exact = exact_bits(order, wp);
        let n = order.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = order.sub(&n, exact, RM);
        let odd = !n
            .div(&BigFloat::from_i32(2, 64), exact, RM)
            .fract()
            .is_zero();
        let neg_order = order.neg();
        let cap = cancellation_cap(prec);
        let mut extra = wp - prec;
        loop {
            let wq = prec + extra;
            let mut xq = x.clone();
            let _ = xq.set_precision(wq, RM);
            let i_neg = bessel_i_series(&neg_order, &xq, &mut w.at(wq))?;
            let i_pos = bessel_i_series(order, &xq, &mut w.at(wq))?;
            let diff = i_neg.sub(&i_pos, wq, RM);
            let big = if i_neg.abs() > i_pos.abs() {
                i_neg.abs()
            } else {
                i_pos.abs()
            };
            let lost = match (big.exponent(), diff.exponent()) {
                _ if diff.is_zero() => wq,
                (Some(b), Some(e)) => usize::try_from(i64::from(b) - i64::from(e)).unwrap_or(0),
                _ => 0,
            };
            if lost + 16 <= extra {
                let pi = w.cc.pi(wq, RM).clone();
                let two = BigFloat::from_i32(2, wq);
                let mut sin_nu_pi = d.mul(&pi, wq, RM).sin(wq, RM, w.cc);
                if odd {
                    sin_nu_pi = sin_nu_pi.neg();
                }
                let r = pi
                    .div(&two, wq, RM)
                    .mul(&diff, wq, RM)
                    .div(&sin_nu_pi, wq, RM);
                return Ok(round_to(r, prec));
            }
            if extra >= cap {
                return Err(special_exhausted(prec));
            }
            extra = (lost + 48).max(2 * extra).min(cap);
        }
    }

    // A large order (K₋ₙ = Kₙ): the finite part of the series alone where
    // the rest is negligible (`bessel_order::neumann_leading`).  Before 0.34
    // the series took `n` steps in exact rationals (`besselk(5000, 1)` never
    // finished) and an order beyond 2⁶³ saturated.
    if order_f64.abs() >= bessel_order::LARGE_ORDER {
        return bessel_order::neumann_leading(bessel_order::Second::K, &order_abs, x, w)
            .unwrap_or_else(|| Err(large_order_refused("besselk", order_f64, x_f64)));
    }
    let n = order_int.unsigned_abs() as usize;
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let half = one.div(&two, wp, RM);
    let x_half = xw.div(&two, wp, RM);
    let x_half_sq = x_half.mul(&x_half, wp, RM);
    let ln_x_half = x_half.ln(wp, RM, w.cc);
    let gamma = arb_euler_gamma(&mut w.at(wp))?;
    let n_bf = BigFloat::from_i128(n as i128, wp);
    let i_n = bessel_i_series(&n_bf, &xw, &mut w.at(wp))?;

    // Finite sum: ½ (x/2)^{-n} Σ_{k=0}^{n-1} (n−k−1)!/k! (−x²/4)^k
    let mut finite = BigFloat::new(wp);
    if n > 0 {
        // f_k = (n−k−1)!/k! ; f_0 = (n−1)!
        let mut f = NeumannCoefficients::new(n);
        let mut pow = one.clone(); // (−x²/4)^k
        let neg_x_half_sq = x_half_sq.neg();
        for k in 0..n {
            if k > 0 {
                pow = pow.mul(&neg_x_half_sq, wp, RM);
            }
            let coeff = f.next(wp);
            finite = finite.add(&coeff.mul(&pow, wp, RM), wp, RM);
        }
        let x_half_pow_neg_n = one.div(&x_half.powi(n, wp, RM), wp, RM);
        finite = half.mul(&x_half_pow_neg_n, wp, RM).mul(&finite, wp, RM);
    }

    // Log term: (−1)^{n+1} ln(x/2) I_n(x)
    let log_term = ln_x_half.mul(&i_n, wp, RM);
    let log_term = if n.is_multiple_of(2) {
        log_term.neg()
    } else {
        log_term
    };

    // Infinite sum: (−1)^n ½ (x/2)^n Σ_k [ψ(k+1) + ψ(n+k+1)] (x²/4)^k/(k!(n+k)!)
    // ψ(m+1) = −γ + H_m.
    let mut h_k = BigFloat::new(wp); // H_0 = 0
    let mut h_nk = BigFloat::new(wp); // H_n
    for m in 1..=n {
        h_nk = h_nk.add(
            &one.div(&BigFloat::from_i128(m as i128, wp), wp, RM),
            wp,
            RM,
        );
    }
    // 1/(k!(n+k)!) start: 1/n!
    let n_fact = factorial_exact(n);
    let mut inv_fact = one.div(&ratio_to_bigfloat(&n_fact, wp), wp, RM);
    let mut pow = one.clone();
    let mut series = BigFloat::new(wp);
    let two_gamma = gamma.mul(&two, wp, RM);
    let max_terms = (x_f64 * 2.0) as usize + wp + 40;
    for k in 0..=max_terms {
        if k > 0 {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            let nk_bf = BigFloat::from_i128((n + k) as i128, wp);
            h_k = h_k.add(&one.div(&k_bf, wp, RM), wp, RM);
            h_nk = h_nk.add(&one.div(&nk_bf, wp, RM), wp, RM);
            inv_fact = inv_fact.div(&k_bf.mul(&nk_bf, wp, RM), wp, RM);
            pow = pow.mul(&x_half_sq, wp, RM);
        }
        let psi_sum = h_k.add(&h_nk, wp, RM).sub(&two_gamma, wp, RM);
        let term = psi_sum.mul(&pow, wp, RM).mul(&inv_fact, wp, RM);
        series = series.add(&term, wp, RM);
        if k > 2
            && let (Some(t_exp), Some(s_exp)) = (term.exponent(), series.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
    }
    let series_term = half
        .mul(&x_half.powi(n, wp, RM), wp, RM)
        .mul(&series, wp, RM);
    let series_term = if n.is_multiple_of(2) {
        series_term
    } else {
        series_term.neg()
    };

    let r = finite.add(&log_term, wp, RM).add(&series_term, wp, RM);
    Ok(round_to(r, prec))
}
