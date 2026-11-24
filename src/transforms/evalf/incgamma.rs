//! Incomplete gamma / exponential integrals (part of [`evalf`](super)).

use super::*;

// ── Incomplete gamma / exponential integrals ───────────────────────────────────

/// `v > 0`: astro-float's `is_positive` is the sign bit alone, true for
/// `+0` (`expint(1 + 10⁻²⁰⁰, ⅓)` rounds its order to 1 and `s = 1 − n` to
/// `+0`, which must take the integer route, not `Γ(0)`).
pub(super) fn bf_strictly_positive(v: &BigFloat) -> bool {
    is_strictly_positive(v)
}

/// `⌈log₂ v⌉` for `v > 1`, `0` for `v ≤ 1`, 1100 for a non-finite `v`: the
/// guard bits a quantity of magnitude `v` needs for its *absolute* error to
/// stay at the unit of the working precision (an exponent `s ln x ≈ 10⁹`
/// needs 30 bits more than the relative precision of its exponential).
pub(super) fn magnitude_bits(v: f64) -> usize {
    if !v.is_finite() {
        1100
    } else if v > 1.0 {
        v.log2().ceil() as usize
    } else {
        0
    }
}

/// `⌈log₂(1/s)⌉` for `0 < s < 1`, from the binary exponent (so also for an
/// `s` below the `f64` range); `0` for `s ≥ 1`.
pub(super) fn inverse_bits(s: &BigFloat) -> usize {
    match s.exponent() {
        Some(e) if e < 1 => usize::try_from(1 - i64::from(e)).unwrap_or(0),
        _ => 0,
    }
}

/// Guard bits for `ln Γ(s)`, `s > 0`, whose magnitude is below
/// `s·|ln s| + s + 2` for `s ≥ 1` and `|ln s| + 2` below.
pub(super) fn lngamma_bits(s: &BigFloat, s_f: f64) -> usize {
    if s_f >= 1.0 {
        magnitude_bits(s_f * s_f.ln() + s_f + 2.0)
    } else {
        magnitude_bits(inverse_bits(s) as f64 * std::f64::consts::LN_2 + 2.0)
    }
}

/// The most bits of cancellation a special function pays for with working
/// precision before it refuses: `4·wp + 1024` (a shape of `10⁻³⁰⁰` costs
/// about 1000, and is evaluated; one of `10⁻¹⁰⁰⁰` is refused at 16 and 50
/// digits — `ln Γ` by Stirling's series at 3500 bits takes seconds).
pub(super) fn cancellation_cap(wp: usize) -> usize {
    wp.saturating_mul(4).saturating_add(1024)
}

/// The refusal of a special function whose cancellation exceeds
/// [`cancellation_cap`]: [`SymplexError::PrecisionExhausted`] at the digits
/// the working precision `prec` stands for (the inverse of
/// `working_precision`).  The routines below `arb_*` raise it at their own
/// working precision; [`requested_at`] restates it at the caller's.
pub(super) fn special_exhausted(prec: usize) -> SymplexError {
    SymplexError::PrecisionExhausted {
        requested: u32::try_from(prec.saturating_sub(64) * 10 / 34).unwrap_or(u32::MAX),
        achieved: 0,
    }
}

/// A [`SymplexError::PrecisionExhausted`] from inside a special function,
/// restated at the precision `prec` its caller asked for (other errors
/// pass through).
pub(super) fn requested_at(prec: usize) -> impl Fn(SymplexError) -> SymplexError {
    move |e| match e {
        SymplexError::PrecisionExhausted { .. } => special_exhausted(prec),
        other => other,
    }
}

/// `v` at precision `p` (a copy; the value is unchanged when `p` is larger).
pub(super) fn at_precision(v: &BigFloat, p: usize) -> BigFloat {
    let mut w = v.clone();
    let _ = w.set_precision(p, RM);
    w
}

/// `x^s e^{−x}` for `x > 0`, as `exp(s ln x − x)` with the exponent formed
/// to an absolute error of `2^−wp` (the relative error of the result): it
/// carries the bits of its magnitude as guard.  0.28 multiplied `e^{−x}` by
/// `x^s`, two factors near the ends of the exponent range for `s ≈ x ≈ 10⁸`.
fn gamma_prefactor(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?.abs();
    let x_f = f64_of(x)?;
    // |ln x| from the binary exponent when x is outside the f64 range.
    let ln_x = x
        .exponent()
        .map_or(0.0, |e| (f64::from(e).abs() + 1.0) * std::f64::consts::LN_2);
    let wpe = wp + magnitude_bits(s_f * ln_x + x_f) + 8;
    let e = s.mul(&x.ln(wpe, RM, w.cc), wpe, RM).sub(x, wpe, RM);
    Ok(e.exp(wp, RM, w.cc))
}

/// Legendre's continued fraction for `Γ(s, x)`, `x > 0` (modified Lentz):
///
/// ```text
/// Γ(s, x) = e^{−x} x^s / (x + 1 − s − 1·(1−s)/(x + 3 − s − 2·(2−s)/(x + 5 − s − …)))
/// ```
///
/// Converges for every `x > 0` and every real `s`; it is the method of
/// choice for `x ≥ max(1, s + 1)`.  At `x = s + 1` it needs about
/// `27·s^{1/3}` steps to 266 bits (573 at `s = 10⁴`, 5779 at `s = 10⁷`) and a
/// few hundred from `x = s + 10√s` on, so the budget is
/// `40·wp + 2000 + 64·s^{1/3}` (0.28's `40·wp + 2000` fell short near
/// `s = 10⁸`), the last term at most `2·10⁶` (seconds; beyond, from
/// `s ≈ 3·10¹⁷`, the fraction is refused rather than run for minutes); the
/// fraction is evaluated with `log₂(budget) + 8` guard bits for the
/// roundings of its steps.
pub(super) fn uppergamma_cf(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let (h, wp) = uppergamma_cf_fraction(s, x, w)?;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let pref = gamma_prefactor(s, x, &mut w.at(wp))?;
    Ok(pref.mul(&h, wp, RM))
}

/// The fraction `1/(x + 1 − s − …)` of [`uppergamma_cf`] (without the
/// prefactor `x^s e^{−x}`), and the precision it was evaluated at.
fn uppergamma_cf_fraction(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    let max_iter = (40 * wp + 2000).saturating_add((64.0 * s_f.abs().cbrt()).min(2e6) as usize);
    let wp = wp + magnitude_bits(max_iter as f64) + 8;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    // "tiny" replaces exact zeros in Lentz's algorithm.
    let mut tiny = BigFloat::from_i32(1, wp);
    tiny.set_exponent(-(4 * wp as i32));
    let mut b = x.add(&one, wp, RM).sub(s, wp, RM);
    let mut c = one.div(&tiny, wp, RM);
    let mut d = if b.is_zero() {
        one.div(&tiny, wp, RM)
    } else {
        one.div(&b, wp, RM)
    };
    let mut h = d.clone();
    let mut converged = false;
    for i in 1..=max_iter {
        let i_bf = BigFloat::from_i128(i as i128, wp);
        // a_i = −i (i − s)
        let an = i_bf.mul(&i_bf.sub(s, wp, RM), wp, RM).neg();
        b = b.add(&two, wp, RM);
        d = an.mul(&d, wp, RM).add(&b, wp, RM);
        if d.is_zero() {
            d = tiny.clone();
        }
        c = b.add(&an.div(&c, wp, RM), wp, RM);
        if c.is_zero() {
            c = tiny.clone();
        }
        d = one.div(&d, wp, RM);
        let del = d.mul(&c, wp, RM);
        h = h.mul(&del, wp, RM);
        let dev = del.sub(&one, wp, RM);
        if negligible(&dev, &one, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: format!(
                "continued fraction for the incomplete gamma function did not converge in \
                 {max_iter} steps"
            ),
        });
    }
    Ok((h, wp))
}

/// Series for `γ(s, x) = x^s e^{−x} Σ_{k≥0} x^k / (s (s+1) … (s+k))`,
/// `s > 0`, `x ≥ 0`: positive terms, decreasing from `k ≈ x − s` on.  Near
/// `x ≈ s` about `√(2s·wp·ln 2)` of them are needed (60 000 at `s = 10⁷`),
/// each term rounding its predecessor's error once more, so the sum carries
/// `log₂(budget) + 8` guard bits.  The budget `2x + wp + 50` is capped at
/// `10⁷` terms (a refusal beyond, for shapes near `10¹³` and above).
fn lowergamma_series(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if x.is_zero() {
        return Ok(BigFloat::new(wp));
    }
    let (sum, wp) = lowergamma_series_sum(s, x, w)?;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let pref = gamma_prefactor(s, x, &mut w.at(wp))?;
    Ok(pref.mul(&sum, wp, RM))
}

/// The sum `Σ x^k / (s (s+1) … (s+k))` of [`lowergamma_series`] (without
/// the prefactor `x^s e^{−x}`), for `x > 0`, and the precision it was
/// evaluated at.  Where the series would take more than a few hundred
/// terms and `(s − x)²/s` exceeds about `1.4·wp`, the sum comes from
/// [`gammatail::far_below_sum`] instead (up to 0.58 the 170,000 terms of
/// the lower tail of `ChiSquared(3.4·10¹⁰)` at its quantile of `10⁻³⁰⁰`).
fn lowergamma_series_sum(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let x_f = f64_of(x)?;
    let max_terms = ((2.0 * x_f).min(1e7) as usize).saturating_add(wp + 50);
    // The terms fall by `Π_{j ≤ k} x/(s + j) ≥ e^{−(d·k + k(k + 1)/2)/x}` (`d =
    // s − x > −1`, `ln(1 + u) ≤ u`), and the sum stops once a term is below
    // `2^{−wp}` of the sum, at most `(s + 1)/(d + 1)` times the first term
    // (`x < s + 1`): so it needs at least the `k` where `(d·k + k(k + 1)/2)/x`
    // reaches `t = wp·ln 2 − ln((s + 1)/(d + 1))`.  Beyond the budget,
    // refuse at once (up to 0.56 a shape near `10¹⁴` at its mean ran the ten
    // million terms, 10–20 s, at every precision tried).  The partial sum
    // after `k` terms is also at most `k + 1` times the first, so `t` is at
    // least `wp·ln 2 − ln(budget + 1)`: far below the mean of a huge shape
    // `(s + 1)/(d + 1)` exceeds `2^wp`, and up to 0.57 the first bound alone
    // was void there (`Gamma(8·10¹⁹⁰, …)` at its quantile of `10⁻³⁰⁰` ran the
    // ten million terms).
    let s_f = f64_of(s)?;
    let d = s_f - x_f;
    let ratio = ((s_f + 1.0) / (d + 1.0)).min(max_terms as f64 + 1.0);
    let t = wp as f64 * std::f64::consts::LN_2 - ratio.ln();
    if d > -1.0 && t > 0.0 {
        let b = d + 0.5;
        let needed = (b * b + 2.0 * x_f * t).sqrt() - b;
        if let Some(found) = gammatail::far_below_sum(s, x, needed, wp) {
            return Ok(found);
        }
        if needed > max_terms as f64 {
            return Err(series_did_not_converge("lowergamma", max_terms));
        }
    }
    let wp = wp + magnitude_bits(max_terms as f64) + 8;
    let (s, x) = (&at_precision(s, wp), &at_precision(x, wp));
    let one = BigFloat::from_i32(1, wp);
    let mut term = one.div(s, wp, RM);
    let mut sum = term.clone();
    let mut converged = false;
    for k in 1..=max_terms {
        let s_plus_k = s.add(&BigFloat::from_i128(k as i128, wp), wp, RM);
        term = term.mul(x, wp, RM).div(&s_plus_k, wp, RM);
        sum = sum.add(&term, wp, RM);
        if negligible(&term, &sum, wp) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(series_did_not_converge("lowergamma", max_terms));
    }
    Ok((sum, wp))
}

/// `ln Γ(s, x)` (`upper`) or `ln γ(s, x)` for `s ≥ 2`, `x > 0`, to an
/// absolute error below `2^−(prec + 8)` (the relative error of the value):
/// the value itself may lie far beyond the exponent range — `Γ(s)`
/// overflows from `s ≈ 7.8·10⁷` on — so it is assembled from logarithms.
/// `L = s ln x − x` (the prefactor), plus `ln` of the series sum of
/// [`lowergamma_series`] (`x < s + 1`) or of the fraction of
/// [`uppergamma_cf`] (`x ≥ s + 1`); the other function is `ln Γ(s) +
/// ln(1 − R)`, with `R = e^{L + ln(·) − ln Γ(s)}` the regularised
/// function computed directly: at most `P(s, s + 1) ≤ P(2, 3) = 1 − 4/e³ <
/// 0.81` or `Q(s, s + 1) < 0.49` (see [`arb_lowergamma`]; both near ½ for
/// the huge shapes this is for), so `1 − R` cancels at most three bits.  Every term is
/// formed with the bits of the largest magnitude, `s·|ln x| + x + ln Γ(s)`,
/// as guard.  For [`extended`]'s scaled values: `uppergamma(s, x)/Γ(s)`, the
/// survival function of a gamma, χ² or Poisson family, at shapes beyond
/// `10⁸`.
pub(super) fn ln_incomplete_gamma(
    s: &BigFloat,
    x: &BigFloat,
    upper: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let s_f = f64_of(s)?;
    let x_f = f64_of(x)?;
    if !(s_f >= 2.0 && bf_strictly_positive(x)) || !x_f.is_finite() {
        return Err(unevaluable(
            "ln_incomplete_gamma: needs s ≥ 2 and a finite x > 0",
        ));
    }
    let mag = s_f * ln_magnitude(x) + x_f + lngamma_magnitude(s, s_f) + 2.0;
    let wp = prec + magnitude_bits(mag) + 24;
    let series = bf_lt(x, &s.add(&BigFloat::from_i32(1, 64), exact_bits(s, wp), RM));
    // The steps the series or the fraction will take, from the decay of
    // the terms (`Σ_{j ≤ k} ln((s + j)/x) ≈ (d·k + k²/2)/x` reaching `wp·ln 2`,
    // `d = s − x`) or the fraction's `27·s^{1/3}` near `x = s + 1`: beyond
    // `LN_INCOMPLETE_GAMMA_MAX_STEPS` the value is refused at once rather
    // than after seconds at every precision tried (an estimate that only
    // decides an early refusal).
    let steps = if series {
        let d = s_f - x_f;
        (d * d + 2.0 * x_f * wp as f64 * std::f64::consts::LN_2).sqrt() - d
    } else if x_f - s_f < 10.0 * s_f.sqrt() {
        27.0 * s_f.cbrt()
    } else {
        0.0
    };
    let (sw, xw) = (at_precision(s, wp), at_precision(x, wp));
    // (Far below the mean the sum has an expansion however long its series:
    // up to 0.58 the lower tail of `Gamma(2·10¹⁰, 1)` 25 standard
    // deviations out was refused here.)
    let far = if series && steps > LN_INCOMPLETE_GAMMA_MAX_STEPS {
        gammatail::far_below_sum(&sw, &xw, steps, wp)
    } else {
        None
    };
    if steps > LN_INCOMPLETE_GAMMA_MAX_STEPS && far.is_none() {
        return Err(special_exhausted(prec));
    }
    let l_pref = sw.mul(&xw.ln(wp, RM, w.cc), wp, RM).sub(&xw, wp, RM);
    let (part, pwp) = match far {
        Some(found) => found,
        None if series => lowergamma_series_sum(&sw, &xw, &mut w.at(wp))?,
        None => uppergamma_cf_fraction(&sw, &xw, &mut w.at(wp))?,
    };
    if !bf_strictly_positive(&part) {
        return Err(special_exhausted(prec));
    }
    let direct = l_pref.add(&part.ln(pwp.max(wp), RM, w.cc), wp, RM);
    if series != upper {
        return Ok(direct);
    }
    let lg = stirling_log_gamma(&sw, &mut w.at(wp))?;
    let r = direct.sub(&lg, wp, RM).exp(wp, RM, w.cc);
    let rest = BigFloat::from_i32(1, wp).sub(&r, wp, RM);
    if !bf_strictly_positive(&rest) {
        return Err(special_exhausted(prec));
    }
    Ok(lg.add(&rest.ln(wp, RM, w.cc), wp, RM))
}

/// The most series terms or fraction steps [`ln_incomplete_gamma`] takes
/// (about a second in a release build); near the centre of a shape beyond
/// `≈ 10¹⁰` the series needs more.
const LN_INCOMPLETE_GAMMA_MAX_STEPS: f64 = 1e6;

/// `Γ(s, x)` at working precision `wp` for real `s`, `x > 0`, to a relative
/// error of about `2^−wp`:
///
/// * `x ≥ max(1, s + 1)`: the continued fraction [`uppergamma_cf`];
/// * `s > 0`, smaller `x`: `Γ(s) − γ(s, x)` ([`uppergamma_by_difference`]);
/// * `s ≤ 0`, `x < 1`: the downward recurrence ([`uppergamma_downward`]).
fn uppergamma_wp(s: &BigFloat, x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    // |s| below the working precision, 0 < x < 2: Γ(s, x) = E₁(x) + δ with
    // |δ| = |∫_x^∞ (t^s − 1) e^{−t}/t dt| ≤ 2|s|((ln x)²/2 + 1) and E₁(x) ≥
    // E₁(2) = 0.0489, so δ is below 2^{−wp} of the value once
    // log₂(1/|s|) ≥ wp + 8 + 2·log₂(|ln x| + 1).  (0.28 reached this limit
    // only when s underflowed f64, by taking it for the integer 0.)
    if !s.is_zero()
        && f64_of(x)? < 2.0
        && inverse_bits(&s.abs()) >= wp + 8 + 2 * magnitude_bits(ln_magnitude(x) + 1.0)
    {
        return Ok(arb_ei(&x.neg(), w)?.neg());
    }
    if bf_strictly_positive(s) {
        return uppergamma_positive(s, x, w);
    }
    if f64_of(x)? >= 1.0 {
        return uppergamma_cf(s, x, w);
    }
    // Far left of 0 the downward recurrence takes |s| steps, while the
    // continued fraction's partial quotients are about −i/|s| for i ≪ |s|
    // and it converges in a few steps.  (0.28 ran the recurrence for any
    // s: `expint(10²⁰/3, 1/7)` overflowed `2·steps` — a panic in debug
    // builds, an endless loop in release ones.)
    if f64_of(s)? < -(DOWNWARD_MAX_STEPS as f64) {
        return uppergamma_cf(s, x, w);
    }
    uppergamma_downward(s, x, w)
}

/// The most steps of the downward recurrence for `Γ(s, x)`, `s ≤ 0`,
/// `x < 1`; further left the continued fraction is used.
const DOWNWARD_MAX_STEPS: usize = 256;

/// `Γ(s, x)` for `s > 0`, `x > 0`: the continued fraction for
/// `x ≥ s + 1`, [`uppergamma_by_difference`] below.
fn uppergamma_positive(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    // x ≥ s + 1 exactly (in f64, s + 1 is s from 2⁵³ on, and a huge s = x
    // went to the fraction, which needs ~27·s^{1/3} steps there).
    if !bf_lt(x, &s.add(&BigFloat::from_i32(1, 64), exact_bits(s, wp), RM)) {
        return uppergamma_cf(s, x, w);
    }
    uppergamma_by_difference(s, s_f, x, w)
}

/// A refusal when `Γ(s) > 2^{EXPONENT_MAX + 3}` (`s ≥ 1`): the incomplete
/// gamma functions `Γ(s, x)` for `x ≤ s + 1` (at least `e^{−2}·Γ(s)`,
/// see [`uppergamma_by_difference`]) and `γ(s, x)` for `x ≥ s` (at least
/// `Γ(s)/2`, `P(s, s) > ½`) overflow the arbitrary-precision exponent
/// range.  Before 0.30 `uppergamma(10²⁰/3, 10²⁰/3)` (about `10^(6.3·10²⁰)`)
/// ran the continued fraction for two million steps (20 s in a debug
/// build) and failed to converge.
fn incomplete_gamma_overflows(s: &BigFloat, what: &str) -> Result<(), SymplexError> {
    let Ok(s_f) = f64_of(s) else {
        return Ok(());
    };
    if s_f.is_nan() || s_f < 1.0 {
        return Ok(());
    }
    // log₂ Γ(s) ≥ ((s − ½) ln s − s)/ln 2 (Stirling, the rest positive).
    let lg = ((s_f - 0.5) * s_f.ln() - s_f) * std::f64::consts::LOG2_E;
    if lg - 3.0 > f64::from(astro_float::EXPONENT_MAX) {
        return Err(unevaluable(format!(
            "{what}: the value (about 2^{lg:.3e}) overflows the arbitrary-precision exponent range"
        )));
    }
    Ok(())
}

/// `Γ(s, x) = Γ(s) − γ(s, x)` for `s > 0`, `x < s + 1` (the region of the
/// series).  The difference cancels `log₂(1/Q(s, x))` bits, and there
/// `Q(s, x) ≥ Q(s, s + 1)`, which is at least `e^{−2}` for `s ≥ 1` (it rises
/// from `Q(1, 2) = e^{−2}` towards ½) and at least
/// `Γ(s, 2)/Γ(s) ≥ E₁(2)·s > 0.048·s` for `s < 1` (`t^{s−1} ≥ 1/t` on
/// `[2, ∞)`, and `Γ(s) ≤ 1/s`; mpmath finds `min Q(s, s + 1)/s = e^{−2}`
/// over `s ≤ 1`): at most `log₂(1/min(1, s)) + 5` bits, added to the working
/// precision up front, with the bits of `ln Γ(s)` for the relative accuracy
/// of `Γ(s)` itself.  0.28 assumed `Q` was never small here and added
/// guard bits for `x` instead — `1.44·x` of them (10⁷ bits, a hang, at
/// `x ≈ 7·10⁶`) and none for a small `s`: `uppergamma(10⁻¹⁰⁰, ½)` came out
/// `0.5649…`, truly `0.5597…`.
fn uppergamma_by_difference(
    s: &BigFloat,
    s_f: f64,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    if inverse_bits(s) > cancellation_cap(wp) {
        return Err(special_exhausted(wp));
    }
    let wp2 = wp + 8 + inverse_bits(s) + lngamma_bits(s, s_f);
    let sw = at_precision(s, wp2);
    let xw = at_precision(x, wp2);
    let g = arb_gamma_real(&sw, &mut w.at(wp2))?;
    let lower = lowergamma_series(&sw, &xw, &mut w.at(wp2))?;
    Ok(round_to(g.sub(&lower, wp2, RM), wp))
}

/// `Γ(s, x)` for `s ≤ 0`, `0 < x < 1`: from `Γ(s₀, x)` with
/// `s₀ = s + n ∈ (0, 1)` (or from `Γ(0, x) = E₁(x)` for an integer
/// `s = −n`) by `n` downward steps `Γ(σ − 1, x) = (t − Γ(σ, x))/(1 − σ)`,
/// `t = x^{σ−1}e^{−x}`.  Each step subtracts two positive numbers, and for
/// `x ≤ 1`
///
/// ```text
/// t − Γ(σ, x) = t·∫₀^∞ (1 − (1 + u/x)^{σ−1}) e^{−u} du ≥ 0.40·min(1, 1 − σ)·t
/// ```
///
/// (convexity in the exponent, and `∫₀^∞ u/(1 + u) e^{−u} du = 1 − e·E₁(1)
/// = 0.4037`), so a step loses at most `log₂(1/min(1, 1 − σ)) + 1.4` bits —
/// many in the first step when `s` sits just below an integer.  The losses
/// are measured as they occur and the evaluation is repeated with a larger
/// reserve when they exceed it.  0.28 reserved 16 bits per step and took an
/// `s` within `10⁻¹²` of an integer for that integer:
/// `expint(3 + 10⁻¹⁴, ⅓)` was wrong from its 14th digit,
/// `uppergamma(−2 + 10⁻¹⁵, ½)` from its 16th.
fn uppergamma_downward(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let s_f = f64_of(s)?;
    let integer = s.is_int();
    let steps = if integer {
        (-s_f).round() as usize
    } else {
        (-s_f).ceil() as usize
    };
    let cap = cancellation_cap(wp);
    let mut reserve = steps.saturating_mul(2).saturating_add(32);
    loop {
        let wp2 = wp + reserve;
        let sw = at_precision(s, wp2);
        let xw = at_precision(x, wp2);
        // Base point: Γ(0, x) = E₁(x) = −Ei(−x) for an integer s; otherwise
        // Γ(s₀, x) with s₀ = s + steps ∈ (0, 1).
        let (mut cur_s, mut val) = if integer {
            (BigFloat::new(wp2), arb_ei(&xw.neg(), &mut w.at(wp2))?.neg())
        } else {
            let s0 = sw.add(&BigFloat::from_i128(steps as i128, wp2), wp2, RM);
            let v = uppergamma_positive(&s0, &xw, &mut w.at(wp2))?;
            (s0, v)
        };
        let one = BigFloat::from_i32(1, wp2);
        let mut lost = 0usize;
        for _ in 0..steps {
            let next_s = cur_s.sub(&one, wp2, RM);
            let t = gamma_prefactor(&next_s, &xw, &mut w.at(wp2))?;
            let diff = t.sub(&val, wp2, RM);
            match (t.exponent(), diff.exponent()) {
                (Some(et), Some(ed)) if bf_strictly_positive(&diff) => {
                    lost += usize::try_from(i64::from(et) - i64::from(ed)).unwrap_or(0);
                }
                _ => {
                    lost = wp2;
                    break;
                }
            }
            val = diff.div(&next_s.neg(), wp2, RM);
            cur_s = next_s;
        }
        if lost + 16 <= reserve {
            return Ok(round_to(val, wp));
        }
        if reserve >= cap {
            return Err(special_exhausted(wp));
        }
        reserve = (lost + 32).max(2 * reserve).min(cap);
    }
}

/// Upper incomplete gamma `Γ(s, x)` for real `s`, `x ≥ 0`.
pub(super) fn arb_uppergamma(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || s.is_nan() || x.is_inf_neg() || s.is_inf() {
        return Err(unevaluable("uppergamma of special float value"));
    }
    if x.is_inf_pos() {
        return Ok(BigFloat::new(prec));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "uppergamma of negative argument not yet supported in evalf",
        ));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, 64);
    if bf_strictly_positive(s) && !bf_gt(x, &s.add(&one, exact_bits(s, wp), RM)) {
        incomplete_gamma_overflows(s, "uppergamma")?;
    }
    if x.is_zero() {
        if !bf_strictly_positive(s) {
            return Err(infinite("uppergamma(s, 0) diverges for s ≤ 0"));
        }
        let s_f = f64_of(s)?;
        let wp = wp + lngamma_bits(s, s_f);
        return Ok(round_to(arb_gamma_real(s, &mut w.at(wp))?, prec));
    }
    let r = uppergamma_wp(s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
    Ok(round_to(r, prec))
}

/// Lower incomplete gamma `γ(s, x)` for real `s > 0`, `x ≥ 0`: the series
/// for `x < s + 1`, `Γ(s) − Γ(s, x)` beyond, where `P(s, x) ≥ P(s, s + 1)
/// > 0.51` (mpmath, over `s` from `10⁻²⁷` to `10³`, rising towards ½ + … for
/// larger `s`) and at most a bit cancels.  For `x > 0` the result is 0 only
/// when the series' prefactor `xˢe⁻ˣ` underflowed: an underflow, not an
/// exact 0 (`accuracy::underflowed`; up to 0.57 `lowergamma(10¹⁰, ½)`
/// printed `0`).
pub(super) fn arb_lowergamma(
    s: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || s.is_nan() || s.is_inf() || x.is_inf_neg() {
        return Err(unevaluable("lowergamma of special float value"));
    }
    if !bf_strictly_positive(s) {
        return Err(unevaluable(
            "lowergamma requires s > 0 for numerical evaluation",
        ));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "lowergamma of negative argument not yet supported in evalf",
        ));
    }
    let s_f = f64_of(s)?;
    let wp = prec + 32;
    if x.is_inf_pos() {
        let wp = wp + lngamma_bits(s, s_f);
        return Ok(round_to(arb_gamma_real(s, &mut w.at(wp))?, prec));
    }
    if !bf_lt(x, s) {
        incomplete_gamma_overflows(s, "lowergamma")?;
    }
    if bf_lt(x, &s.add(&BigFloat::from_i32(1, 64), exact_bits(s, wp), RM)) {
        // γ(s, x) ≥ xˢe⁻ˣ/s, the first term of the series: beyond the
        // exponent range, refuse before summing (up to 0.57 the Poisson
        // survival function `P(27625247160, 2.7625247…10¹⁰)`, about
        // `2^(9.2·10¹¹)·½/k!`, summed two million terms first).
        let x_f = f64_of(x)?;
        let lg = (s_f * x_f.ln() - x_f - s_f.ln()) * std::f64::consts::LOG2_E;
        if lg - 3.0 > f64::from(astro_float::EXPONENT_MAX) {
            return Err(unevaluable(format!(
                "lowergamma: the value (at least 2^{lg:.3e}) overflows the arbitrary-precision \
                 exponent range"
            )));
        }
        let r = lowergamma_series(s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
        return Ok(round_to(r, prec));
    }
    let wp = wp + 4 + lngamma_bits(s, s_f);
    let g = arb_gamma_real(s, &mut w.at(wp))?;
    let upper = uppergamma_cf(s, x, &mut w.at(wp))?;
    Ok(round_to(g.sub(&upper, wp, RM), prec))
}

/// Generalised exponential integral `E_n(x) = x^{n−1} Γ(1−n, x)` for real
/// `n` and `x > 0` (`E_n(0) = 1/(n−1)` for `n > 1`).
pub(super) fn arb_expint(
    n: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || n.is_nan() || n.is_inf() {
        return Err(unevaluable("expint of special float value"));
    }
    if x.is_inf_pos() {
        return Ok(BigFloat::new(prec));
    }
    let wp = prec + 32;
    let one = BigFloat::from_i32(1, wp);
    if x.is_zero() {
        // `E_n(0) = +∞` for `n ≤ 1`: `E₁` has a logarithmic pole at 0.  (Up
        // to 0.49 `n − 1 = +0` counted as positive and `expint(1, 0)` came
        // out `1/0`, refused as "precision exhausted".)
        let n_minus_1 = n.sub(&one, wp, RM);
        if !is_strictly_positive(&n_minus_1) {
            return Err(infinite("expint(n, 0) diverges for n ≤ 1"));
        }
        return Ok(round_to(one.div(&n_minus_1, wp, RM), prec));
    }
    if x.is_negative() {
        return Err(unevaluable(
            "expint of negative argument is complex; not yet supported in evalf",
        ));
    }
    let s = one.sub(n, wp, RM);
    let g = uppergamma_wp(&s, x, &mut w.at(wp)).map_err(requested_at(prec))?;
    let x_pow = bf_pow(x, &n.sub(&one, wp, RM), &mut w.at(wp));
    Ok(round_to(x_pow.mul(&g, wp, RM), prec))
}

/// `Shi(x)` (`want_shi`) or `Chi(x)` for `x > 0` (`Shi` is odd and is
/// returned for any real `x`).
///
/// * `|x| ≤ 40`: `Shi(x) = Σ x^{2k+1}/((2k+1)(2k+1)!)`,
///   `Chi(x) = γ + ln x + Σ_{k≥1} x^{2k}/((2k)(2k)!)` — all terms of the
///   series positive; `γ + ln x` cancels against it next to the zero
///   `0.5238…` of `Chi`, so the sum's error is charged to `w` as a part of
///   its three summands (`wp.rs`, "A route's own error"; up to 0.59
///   `Chi(x)` for the 128-bit dyadic `x` nearest the zero printed 14
///   correct digits of 16).
/// * larger `|x|`: `Shi = (Ei(x) − Ei(−x))/2`, `Chi = (Ei(x) + Ei(−x))/2`.
pub(super) fn arb_shi_chi(
    x: &BigFloat,
    want_shi: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Shi/Chi of special float value"));
    }
    if x.is_zero() {
        return if want_shi {
            Ok(BigFloat::new(prec))
        } else {
            Err(infinite("Chi(0) is zoo"))
        };
    }
    let wp = prec + 32;
    let ax = x.abs();
    let x_f = f64_of(&ax)?;
    let r = if x_f > 40.0 {
        let ei_pos = arb_ei(&ax, &mut w.at(wp))?;
        let ei_neg = arb_ei(&ax.neg(), &mut w.at(wp))?;
        let two = BigFloat::from_i32(2, wp);
        if want_shi {
            ei_pos.sub(&ei_neg, wp, RM).div(&two, wp, RM)
        } else {
            ei_pos.add(&ei_neg, wp, RM).div(&two, wp, RM)
        }
    } else {
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let x2 = xw.mul(&xw, wp, RM);
        let max_terms = (x_f * 1.5) as usize + wp + 40;
        let name = if want_shi { "Shi" } else { "Chi" };
        let mut converged = false;
        let series = if want_shi {
            // t_k = x^{2k+1}/(2k+1)!
            let mut t = xw.clone();
            let mut sum = BigFloat::new(wp);
            for k in 0..max_terms {
                if k > 0 {
                    let d = BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp);
                    t = t.mul(&x2, wp, RM).div(&d, wp, RM);
                }
                let term = t.div(&BigFloat::from_i128(2 * k as i128 + 1, wp), wp, RM);
                sum = sum.add(&term, wp, RM);
                if negligible(&term, &sum, wp) {
                    converged = true;
                    break;
                }
            }
            sum
        } else {
            let gamma = arb_euler_gamma(&mut w.at(wp))?;
            let ln_x = xw.ln(wp, RM, w.cc);
            let mut u = BigFloat::from_i32(1, wp); // x^{2k}/(2k)!
            let mut sum = BigFloat::new(wp);
            let mut terms = 0usize;
            for k in 1..max_terms {
                let d = BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp);
                u = u.mul(&x2, wp, RM).div(&d, wp, RM);
                let term = u.div(&BigFloat::from_i128(2 * k as i128, wp), wp, RM);
                sum = sum.add(&term, wp, RM);
                terms = k;
                if negligible(&term, &sum, wp) {
                    converged = true;
                    break;
                }
            }
            // The series, `terms` terms of one sign, is one of three
            // summands that cancel next to the zero of `Chi`.
            w.charge(
                Summed::of(&[&gamma, &ln_x, &sum])
                    .steps(terms)
                    .error(wp, 0.0),
            );
            gamma.add(&ln_x, wp, RM).add(&sum, wp, RM)
        };
        if !converged {
            return Err(series_did_not_converge(name, max_terms));
        }
        series
    };
    let r = if want_shi && x.is_negative() {
        r.neg()
    } else {
        r
    };
    Ok(round_to(r, prec))
}

/// Fresnel integrals `S(x)` (`want_s`) and `C(x)` for real `x` (both odd).
///
/// * `πx²/2 ≤ wp·ln 2`: Maclaurin series
///   `S = Σ (−1)^k (π/2)^{2k+1} x^{4k+3}/((2k+1)!(4k+3))`,
///   `C = Σ (−1)^k (π/2)^{2k} x^{4k+1}/((2k)!(4k+1))`, with guard bits for
///   the `e^{πx²/2}` cancellation.
/// * beyond: DLMF 7.12 auxiliary functions
///   `S = 1/2 − f cos(πx²/2) − g sin(πx²/2)`, `C = 1/2 + f sin(πx²/2) − g cos(πx²/2)`,
///   `f ~ (1/πx) Σ (−1)^m (4m−1)!!/(πx²)^{2m}`, `g ~ (1/πx) Σ (−1)^m (4m+1)!!/(πx²)^{2m+1}`
///   (optimal truncation error `~e^{−πx²/2}`).
pub(super) fn arb_fresnel(
    x: &BigFloat,
    want_s: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    if x.is_nan() || x.is_inf() {
        return Err(unevaluable("Fresnel integral of special float value"));
    }
    if x.is_zero() {
        return Ok(BigFloat::new(prec));
    }
    let base_wp = prec + 32;
    let ax = x.abs();
    let x_f = f64_of(&ax)?;
    let arg_f = std::f64::consts::FRAC_PI_2 * x_f * x_f;

    // `x = 2^(e−1)… 2^e`: the auxiliary functions are `~1/(πx)`, and the
    // phase `πx²/2` has `2e` bits before the point.
    let e = usize::try_from(ax.exponent().map_or(0, i64::from).max(0)).unwrap_or(0);
    let r = if e > base_wp + 2 {
        // `f cos + g sin` is below `2^−wp` next to 1/2: before 0.30 the
        // phase was reduced anyway (`fresnels(erfi(335))`, `x ≈ 2^161900`,
        // asked for π to 323,800 bits and hung).
        BigFloat::from_f64(0.5, base_wp)
    } else if arg_f > (base_wp as f64) * std::f64::consts::LN_2 {
        let wp = base_wp;
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let one = BigFloat::from_i32(1, wp);
        let pi = w.cc.pi(wp, RM).clone();
        let pi_x2 = pi.mul(&xw, wp, RM).mul(&xw, wp, RM);
        let inv = one.div(&pi_x2, wp, RM); // 1/(πx²)
        let inv2 = inv.mul(&inv, wp, RM);
        // f-series: (4m−1)!!/(πx²)^{2m}; g-series: (4m+1)!!/(πx²)^{2m+1}
        let mut tf = one.clone();
        let mut tg = inv.clone();
        let mut f_sum = BigFloat::new(wp);
        let mut g_sum = BigFloat::new(wp);
        let mut prev: Option<BigFloat> = None;
        for m in 0..(wp * 2 + 100) {
            if m > 0 {
                // (4m−1)!!/(4m−5)!! = (4m−3)(4m−1); (4m+1)!!/(4m−3)!! = (4m−1)(4m+1)
                let a = BigFloat::from_i128((4 * m as i128 - 3) * (4 * m as i128 - 1), wp);
                let b = BigFloat::from_i128((4 * m as i128 - 1) * (4 * m as i128 + 1), wp);
                tf = tf.mul(&a, wp, RM).mul(&inv2, wp, RM);
                tg = tg.mul(&b, wp, RM).mul(&inv2, wp, RM);
            }
            if let Some(ref p) = prev
                && bf_gt(&tf, p)
            {
                break;
            }
            if m % 2 == 0 {
                f_sum = f_sum.add(&tf, wp, RM);
                g_sum = g_sum.add(&tg, wp, RM);
            } else {
                f_sum = f_sum.sub(&tf, wp, RM);
                g_sum = g_sum.sub(&tg, wp, RM);
            }
            if negligible(&tf, &one, wp) {
                break;
            }
            prev = Some(tf.clone());
        }
        let pi_x = pi.mul(&xw, wp, RM);
        let f = f_sum.div(&pi_x, wp, RM);
        let g = g_sum.div(&pi_x, wp, RM);
        let two = BigFloat::from_i32(2, wp);
        // The phase to an absolute `2^−wp`: `2e` more bits.  Before 0.30 it
        // was formed at `wp` bits, an absolute error of `2^(2e−wp)` times
        // the amplitude `2^−e` — wrong digits once `e` passed the guard
        // bits: `fresnels(10⁴⁵)` at 60 digits ended `…816212`, truly
        // `…816209`.
        let wt = wp + 2 * e;
        let mut xt = ax.clone();
        let _ = xt.set_precision(wt, RM);
        let theta = w.cc.pi(wt, RM).mul(&xt, wt, RM).mul(&xt, wt, RM).div(
            &BigFloat::from_i32(2, wt),
            wt,
            RM,
        );
        let s = theta.sin(wp, RM, w.cc);
        let c = theta.cos(wp, RM, w.cc);
        let half = one.div(&two, wp, RM);
        if want_s {
            half.sub(&f.mul(&c, wp, RM), wp, RM)
                .sub(&g.mul(&s, wp, RM), wp, RM)
        } else {
            half.add(&f.mul(&s, wp, RM), wp, RM)
                .sub(&g.mul(&c, wp, RM), wp, RM)
        }
    } else {
        let wp = Prec::new(base_wp)
            .guarded(Guard::for_log2(arg_f * std::f64::consts::LOG2_E), Prec::MAX)
            .ok_or_else(|| special_exhausted(prec))?
            .plus(16)
            .bits();
        let mut xw = ax.clone();
        let _ = xw.set_precision(wp, RM);
        let pi = w.cc.pi(wp, RM).clone();
        let two = BigFloat::from_i32(2, wp);
        let half_pi = pi.div(&two, wp, RM);
        let x2 = xw.mul(&xw, wp, RM);
        // u = (π/2) x²;  terms: S_k = (−1)^k u^{2k+1} x /((2k+1)! (4k+3)),
        //                          C_k = (−1)^k u^{2k} x /((2k)! (4k+1))
        let u = half_pi.mul(&x2, wp, RM);
        let u2 = u.mul(&u, wp, RM);
        let mut t = if want_s {
            u.mul(&xw, wp, RM)
        } else {
            xw.clone()
        };
        let mut sum = BigFloat::new(wp);
        let max_terms = (arg_f * 2.0) as usize + wp + 40;
        let mut converged = false;
        for k in 0..max_terms {
            if k > 0 {
                let d = if want_s {
                    BigFloat::from_i128((2 * k as i128) * (2 * k as i128 + 1), wp)
                } else {
                    BigFloat::from_i128((2 * k as i128 - 1) * (2 * k as i128), wp)
                };
                t = t.mul(&u2, wp, RM).div(&d, wp, RM);
            }
            let div = BigFloat::from_i128(4 * k as i128 + if want_s { 3 } else { 1 }, wp);
            let term = t.div(&div, wp, RM);
            if k % 2 == 0 {
                sum = sum.add(&term, wp, RM);
            } else {
                sum = sum.sub(&term, wp, RM);
            }
            if negligible(&term, &sum, wp) {
                converged = true;
                break;
            }
        }
        if !converged {
            let name = if want_s { "fresnels" } else { "fresnelc" };
            return Err(series_did_not_converge(name, max_terms));
        }
        sum
    };
    let r = if x.is_negative() { r.neg() } else { r };
    Ok(round_to(r, prec))
}
