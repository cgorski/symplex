//! Trigonometric / exponential / logarithmic integrals (part of [`evalf`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Trigonometric / exponential / logarithmic integrals
// ═══════════════════════════════════════════════════════════════════════════

/// Number of guard bits needed to absorb the cancellation in an
/// alternating power series whose largest term is `~e^{|x|}`.
///
/// Saturating: `as usize` takes a huge or infinite `|x|·log₂e` to
/// `usize::MAX`, and the sum must stay there for the callers' cap to refuse
/// it.  Up to 0.46 the `+ 32` overflowed: a panic with overflow checks (the
/// CI fuzz builds: `besselj` of an argument near `2¹⁰⁰⁰` that the Hankel
/// expansion declined, met by `fuzz_roundtrip`), and in release builds a
/// wrap to 31 guard bits for a series that needed `10³⁰⁰`.
pub(super) fn cancellation_guard_bits(x_abs: f64) -> usize {
    ((x_abs * std::f64::consts::LOG2_E).ceil() as usize).saturating_add(32)
}

/// [`cancellation_guard_bits`] for the power series of `J_ν` and `Y_n` at
/// `|x|`, refused beyond [`cancellation_cap`]: the series also takes about
/// `|x|` terms.  Before 0.30 nothing bounded them — `besselj(5/2, 10²⁵)`
/// asked for `1.4·10²⁵` bits and printed garbage (`3.8·10¹⁰⁵²⁷`).
pub(super) fn series_guard_bits(x_abs: f64, prec: usize) -> Result<usize, SymplexError> {
    let guard = cancellation_guard_bits(x_abs);
    if !x_abs.is_finite() || guard > cancellation_cap(prec) {
        return Err(special_exhausted(prec));
    }
    Ok(guard)
}

/// Refuse a Bessel argument whose Hankel phase `x − νπ/2 − π/4` would
/// need more than [`cancellation_cap`] extra bits (`log₂|x|` of them, see
/// [`bessel_jy_hankel`]): `besselj(−5, erfi(1210))`, an argument of
/// `2^(2·10⁶)`, ran for minutes.
pub(super) fn refuse_bessel_argument(x: &BigFloat, prec: usize) -> Result<(), SymplexError> {
    let cap = i64::try_from(cancellation_cap(prec)).unwrap_or(i64::MAX);
    if x.is_inf() || x.exponent().is_some_and(|e| i64::from(e) > cap) {
        return Err(special_exhausted(prec));
    }
    Ok(())
}

/// Auxiliary functions `f(x)`, `g(x)` of the asymptotic expansions
///
/// ```text
/// Si(x) = π/2 − f(x) cos x − g(x) sin x,   Ci(x) = f(x) sin x − g(x) cos x
/// f(x) ~ (1/x) Σ (−1)^k (2k)!/x^{2k},   g(x) ~ (1/x²) Σ (−1)^k (2k+1)!/x^{2k}
/// ```
///
/// for `x > 0`.  Uses optimal truncation (stop when terms grow); the
/// caller guarantees `x` is large enough that the smallest term is below
/// the working precision.
fn si_ci_asymptotic_fg(x: &BigFloat, wp: usize) -> (BigFloat, BigFloat) {
    let one = BigFloat::from_i32(1, wp);
    let inv_x = one.div(x, wp, RM);
    let inv_x2 = inv_x.mul(&inv_x, wp, RM);

    let mut f_sum = BigFloat::new(wp);
    let mut g_sum = BigFloat::new(wp);
    // t_f = (2k)!/x^{2k}, t_g = (2k+1)!/x^{2k}
    let mut t_f = one.clone();
    let mut t_g = one.clone();
    let mut prev_f: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 0..max_terms {
        if k > 0 {
            // (2k)! / (2k-2)! = (2k-1)(2k);  (2k+1)!/(2k-1)! = (2k)(2k+1)
            let a = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            let b = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
            t_f = t_f.mul(&a, wp, RM).mul(&inv_x2, wp, RM);
            t_g = t_g.mul(&b, wp, RM).mul(&inv_x2, wp, RM);
        }
        // Optimal truncation: stop once terms start growing.
        if let Some(ref p) = prev_f
            && t_f.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        if k % 2 == 0 {
            f_sum = f_sum.add(&t_f, wp, RM);
            g_sum = g_sum.add(&t_g, wp, RM);
        } else {
            f_sum = f_sum.sub(&t_f, wp, RM);
            g_sum = g_sum.sub(&t_g, wp, RM);
        }
        if let (Some(t_exp), Some(s_exp)) = (t_f.exponent(), f_sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev_f = Some(t_f.abs());
    }
    (f_sum.mul(&inv_x, wp, RM), g_sum.mul(&inv_x2, wp, RM))
}

/// Sine and cosine integrals for real `x`.
///
/// * `want_si == true`: returns `(Si(x), 0)` — valid for all real `x`
///   (`Si` is odd).
/// * `want_si == false`: returns `(0, Ci(x))` — requires `x > 0`.
///
/// Power series for moderate `|x|` (with guard bits for the `e^{|x|}`
/// cancellation) and the Hankel-type asymptotic expansion when
/// `|x| > wp·ln 2` (where its optimal-truncation error `~e^{−x}` is below
/// the working precision).
pub(super) fn arb_si_ci(
    x: &BigFloat,
    want_si: bool,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() {
        return Err(SymplexError::Unevaluable {
            reason: "Si/Ci of special float value".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;

    if x.is_zero() {
        return if want_si {
            Ok((BigFloat::new(prec), BigFloat::new(prec)))
        } else {
            Err(infinite("Ci(0) is zoo"))
        };
    }
    if !want_si && x.is_negative() {
        return Err(SymplexError::Unevaluable {
            reason: "Ci requires a positive argument here".into(),
        });
    }

    // Far out `f`, `g` are below `2^−wp` next to π/2: `Si(x) = ±π/2`; `Ci(x)
    // ≈ sin(x)/x` needs `x` reduced modulo 2π, refused beyond
    // `cancellation_cap` bits (before 0.30 both reduced any `x`, and
    // `Si(erfi(765))`, `x ≈ 2^844000`, hung).
    let e = usize::try_from(x.exponent().map_or(0, i64::from).max(0)).unwrap_or(0);
    if e > base_wp + 2 {
        if !want_si {
            if e > cancellation_cap(prec) {
                return Err(special_exhausted(prec));
            }
        } else {
            let half_pi =
                w.cc.pi(base_wp, RM)
                    .div(&BigFloat::from_i32(2, base_wp), base_wp, RM);
            let si = if x.is_negative() {
                half_pi.neg()
            } else {
                half_pi
            };
            return Ok((round_to(si, prec), BigFloat::new(prec)));
        }
    }
    let switch = (base_wp as f64) * std::f64::consts::LN_2;
    if x_abs > switch {
        // ── Asymptotic expansion ──
        let wp = base_wp;
        let ax = x.abs();
        let (f, g) = si_ci_asymptotic_fg(&ax, wp);
        let sin_x = ax.sin(wp, RM, w.cc);
        let cos_x = ax.cos(wp, RM, w.cc);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, wp);
        let half_pi = pi.div(&two, wp, RM);
        let mut si =
            half_pi
                .sub(&f.mul(&cos_x, wp, RM), wp, RM)
                .sub(&g.mul(&sin_x, wp, RM), wp, RM);
        if x.is_negative() {
            si = si.neg();
        }
        let ci = f.mul(&sin_x, wp, RM).sub(&g.mul(&cos_x, wp, RM), wp, RM);
        if !want_si {
            // `f sin x − g cos x` cancels next to a zero of `Ci`: its error is
            // a part of `|f| + |g|`, not of the value (`wp.rs`, "A route's own
            // error"; up to 0.59 `Ci(x)` for the 128-bit dyadic `x` nearest
            // the zero at 113.9 printed 11 correct digits of 16).  `f` and
            // `g` are sums of at most `2·wp + 100` terms, cut at their
            // smallest: the remainder is at most the first term left out
            // (DLMF 6.12(ii)), `√(2πx)·e^(−x)/x`, below `√(4.4·wp)·2^(−wp)` of
            // `f` from the switch on.
            w.charge(Summed::of(&[&f, &g]).steps(2 * wp + 100).error(wp, 0.0));
        }
        return Ok((round_to(si, prec), round_to(ci, prec)));
    }

    // ── Power series ──
    let wp = base_wp + cancellation_guard_bits(x_abs);
    let mut xw = x.clone();
    let _ = xw.set_precision(wp, RM);
    let x2 = xw.mul(&xw, wp, RM);
    let max_terms = (x_abs * 1.5) as usize + wp + 40;

    let si = if want_si {
        // Si(x) = Σ (−1)^k t_k/(2k+1),  t_k = x^{2k+1}/(2k+1)!
        let mut t = xw.clone();
        let mut sum = BigFloat::new(wp);
        for k in 0..max_terms {
            if k > 0 {
                let d = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
                t = t.mul(&x2, wp, RM).div(&d, wp, RM);
            }
            let term = t.div(&BigFloat::from_i128(2 * k as i128 + 1, wp), wp, RM);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        round_to(sum, prec)
    } else {
        BigFloat::new(prec)
    };

    let ci = if want_si {
        BigFloat::new(prec)
    } else {
        // Ci(x) = γ + ln x + Σ_{k≥1} (−1)^k u_k/(2k),  u_k = x^{2k}/(2k)!
        let gamma = arb_euler_gamma(&mut w.at(wp))?;
        let ln_x = xw.ln(wp, RM, w.cc);
        let mut sum = BigFloat::new(wp);
        let mut u = BigFloat::from_i32(1, wp);
        let mut seen = Summed::new();
        for k in 1..max_terms {
            let d = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
            u = u.mul(&x2, wp, RM).div(&d, wp, RM);
            let term = u.div(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
            seen.term(&term);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
            {
                break;
            }
        }
        // The series cancels within itself, and `γ + ln x` against it
        // (wholly, next to a zero of `Ci`): the error is a part of the
        // terms, not of the value (`wp.rs`, "A route's own error"; up to
        // 0.59 `Ci(x)` for the 166-bit dyadic `x` nearest the first zero was
        // wrong from the 25th digit of 30).
        w.charge(seen.error(wp, 0.0));
        w.charge(Summed::of(&[&gamma, &ln_x, &sum]).error(wp, 0.0));
        let ci = gamma.add(&ln_x, wp, RM).add(&sum, wp, RM);
        round_to(ci, prec)
    };

    Ok((si, ci))
}

/// Exponential integral `Ei(x)` for real `x ≠ 0`.
///
/// * `abs(x) > wp·ln 2`: the asymptotic expansion `Ei(x) ~ (eˣ/x) Σ k!/xᵏ`
///   with optimal truncation (relative error `~e^{−|x|}`);
/// * `x ≤ −max(1, wp/64)`: `Ei(x) = −E₁(−x)` by Legendre's continued
///   fraction ([`uppergamma_cf`] at `s = 0`), which does not cancel;
/// * otherwise `Ei(x) = γ + ln|x| + Σ_{k≥1} xᵏ/(k·k!)`, with guard bits for
///   the alternating cancellation when `x < 0`.
///
/// For `x < 0` the series cancels `2·abs(x)·log₂e` bits and takes about
/// `e·abs(x)` terms at that many more bits (and Euler's constant at them),
/// the fraction `(wp·ln 2)²/(16·abs(x))` steps at `wp` bits: measured, they
/// cost the same at `abs(x) ≈ wp/60` (3 at 16 digits, 8 at 100, 20 at 300, 45
/// at 1,000).  Up to 0.60 the series ran up to the asymptotic range:
/// `Ei(−691)` at 280 digits took 128 ms (the fraction: 0.5 ms), and
/// `li(sin(1 + 10⁻³⁰⁰) − sin 1)`, whose argument needs 1,152 bits, 145 ms at
/// 16 digits (found by `fuzz_evalf` as a slow unit).
pub(super) fn arb_ei(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf_pos() || x.is_inf_neg() || x.is_zero() {
        return Err(SymplexError::Unevaluable {
            reason: "Ei of zero or special float value".into(),
        });
    }
    let x_f64 = f64_of(x)?;
    let x_abs = x_f64.abs();
    let base_wp = prec + 32;
    let switch = (base_wp as f64) * std::f64::consts::LN_2;

    if x_abs > switch {
        // ── Asymptotic: Ei(x) ~ (e^x / x) Σ_{k≥0} k! / x^k ──
        let wp = base_wp;
        let (inv_x, sum) = ei_asymptotic_series(x, wp);
        let ex = x.exp(wp, RM, w.cc);
        let r = ex.mul(&inv_x, wp, RM).mul(&sum, wp, RM);
        return Ok(round_to(r, prec));
    }

    // ── Continued fraction: Ei(x) = −E₁(−x) = −Γ(0, −x) for x ≤ −1 ──
    if x.is_negative() && e1_by_fraction(x_abs, base_wp) {
        let e1 = uppergamma_cf(&BigFloat::new(base_wp), &x.neg(), &mut w.at(base_wp))?;
        return Ok(round_to(e1.neg(), prec));
    }

    // ── Power series ──
    // For x < 0 the terms reach `~e^{|x|}` and `Ei(x) ~ e^{−|x|}/|x|`: the
    // sum cancels about `2|x|·log₂e` bits, and the value near the zero
    // `x ≈ 0.3725` more.  The loss is measured (the largest term against
    // the value) and the series repeated with that many more bits, up to
    // `cancellation_cap`.  Before 0.30 the guard was `|x|·log₂e`, half
    // of it: `Ei(−196)` at 60 digits was wrong from its 24th digit.
    let cap = cancellation_cap(prec);
    let mut extra = if x.is_negative() {
        cancellation_guard_bits(2.0 * x_abs)
    } else {
        32
    };
    loop {
        let wp = prec + extra;
        let mut xw = x.clone();
        let _ = xw.set_precision(wp, RM);
        let gamma = arb_euler_gamma(&mut w.at(wp))?;
        let ln_abs_x = xw.abs().ln(wp, RM, w.cc);
        let mut largest = gamma.abs().max(&ln_abs_x.abs());
        let mut v = BigFloat::from_i32(1, wp); // x^k/k!
        let mut sum = BigFloat::new(wp);
        let max_terms = (x_abs * 3.0) as usize + wp + 40;
        for k in 1..max_terms {
            let k_bf = BigFloat::from_i128(k as i128, wp);
            v = v.mul(&xw, wp, RM).div(&k_bf, wp, RM);
            let term = v.div(&k_bf, wp, RM);
            sum = sum.add(&term, wp, RM);
            largest = largest.max(&term.abs()).max(&sum.abs());
            if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
                && (s_exp as i64 - t_exp as i64) > wp as i64
                && (k as f64) > x_abs
            {
                break;
            }
        }
        let r = gamma.add(&ln_abs_x, wp, RM).add(&sum, wp, RM);
        let lost = match (largest.exponent(), r.exponent()) {
            (Some(l), Some(v)) if !r.is_zero() => (i64::from(l) - i64::from(v)).max(0) as usize,
            _ => wp,
        };
        if lost + 32 <= extra {
            return Ok(round_to(r, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 64).min(cap).max(extra + 32);
    }
}

/// Is `E₁(x) = −Ei(−x)` for `x > 0` at `wp` bits cheaper by Legendre's
/// continued fraction than by the series of `Ei` (see [`arb_ei`])?  From
/// `x = max(1, wp/64)` on.
pub(super) fn e1_by_fraction(x: f64, wp: usize) -> bool {
    x >= 1.0 && x >= wp as f64 / 64.0
}

/// The asymptotic series of `Ei(x) ~ (e^x/x)·Σ_{k≥0} k!/x^k` at `wp` bits,
/// truncated at its smallest term: `(1/x, Σ)`.  [`arb_ei`] multiplies them
/// by `e^x`; `extended` by `e^x` at its own scale, below the exponent range.
pub(super) fn ei_asymptotic_series(x: &BigFloat, wp: usize) -> (BigFloat, BigFloat) {
    let one = BigFloat::from_i32(1, wp);
    let inv_x = one.div(x, wp, RM);
    let mut term = one.clone();
    let mut sum = BigFloat::new(wp);
    let mut prev: Option<BigFloat> = None;
    let max_terms = wp * 2 + 100;
    for k in 0..max_terms {
        if k > 0 {
            term = term
                .mul(&BigFloat::from_i128(k as i128, wp), wp, RM)
                .mul(&inv_x, wp, RM);
        }
        if let Some(ref p) = prev
            && term.abs().cmp(p).is_some_and(|c| c > 0)
        {
            break;
        }
        sum = sum.add(&term, wp, RM);
        if let (Some(t_exp), Some(s_exp)) = (term.exponent(), sum.exponent())
            && (s_exp as i64 - t_exp as i64) > wp as i64
        {
            break;
        }
        prev = Some(term.abs());
    }
    (inv_x, sum)
}

/// Round a `BigFloat` down to `prec` bits (no-op on failure).
pub(super) fn round_to(mut v: BigFloat, prec: usize) -> BigFloat {
    let _ = v.set_precision(prec, RM);
    v
}

/// `b^e` for real `b > 0` (or integer `e`): [`crate::base::bigfloat::pow`],
/// which never calls the non-terminating `BigFloat::pow`.
pub(super) fn bf_pow(b: &BigFloat, e: &BigFloat, w: &mut Wp<'_>) -> BigFloat {
    let wp = w.bits();
    crate::base::bigfloat::pow(b, e, wp, RM, w.cc)
}
