//! Incomplete beta (part of [`evalf`](super)).

use super::*;

// ── Incomplete beta ───────────────────────────────────────────────────────────────

/// `|ln v|` bounded from the binary exponent of `v > 0` (so for any
/// magnitude, and never below `ln 2`).
pub(super) fn ln_magnitude(v: &BigFloat) -> f64 {
    v.exponent()
        .map_or(0.0, |e| (f64::from(e).abs() + 1.0) * std::f64::consts::LN_2)
}

/// A bound on `|ln Γ(s)|` for `s > 0` (see [`lngamma_bits`]).
pub(super) fn lngamma_magnitude(s: &BigFloat, s_f: f64) -> f64 {
    if s_f >= 1.0 {
        s_f * s_f.ln() + s_f + 2.0
    } else {
        ln_magnitude(s) + 2.0
    }
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)` for `a, b > 0` to an
/// absolute error below `2^−wp`: each `ln Γ` to `2^(−wp−8)`, a bound derived
/// from its terms (`log_gamma_abs`; up to 0.60 an estimate of their
/// magnitude as guard bits), and the two sums at the bits of the largest
/// magnitude beyond that (they are `≈ 1.7·10⁹` for shapes near `10⁸` and
/// cancel to the size of `ln B`), which also covers the rounding of
/// `a + b`.
pub(super) fn ln_beta_wp(
    a: &BigFloat,
    b: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let a_f = f64_of(a)?;
    let b_f = f64_of(b)?;
    let ab_f = a_f + b_f;
    let mag = lngamma_magnitude(a, a_f)
        .max(lngamma_magnitude(b, b_f))
        .max(ab_f * ab_f.ln().abs() + ab_f + 2.0);
    let p = wp + magnitude_bits(mag) + 8;
    let a_plus_b = a.add(b, p, RM);
    let la = log_gamma_abs(a, wp + 8, w)?;
    let lb = log_gamma_abs(b, wp + 8, w)?;
    let lab = log_gamma_abs(&a_plus_b, wp + 8, w)?;
    Ok(la.add(&lb, p, RM).sub(&lab, p, RM))
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a+b)` at working precision `wp` for `a, b > 0`, as
/// `exp(ln B(a, b))`: the three gamma values need not be representable
/// (`Γ(10⁸)` is beyond the exponent range, `B(10⁸, 10⁸) = 2^{−2·10⁸}` is
/// not).  0 when `ln B` lies below the exponent range: the accuracy layer
/// holds it as the underflow of a positive number (`accuracy::underflowed`;
/// up to 0.57 refused as "beyond the exponent range" in `eval_f64` too).
pub(super) fn beta_wp(
    a: &BigFloat,
    b: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let v = ln_beta_wp(a, b, w)?.exp(wp, RM, w.cc);
    // `exp` of an exponent above the range is ∞, not the value.
    if v.is_inf() || v.is_nan() {
        return Err(unevaluable(
            "Beta: the value lies beyond the arbitrary-precision exponent range",
        ));
    }
    Ok(v)
}

/// From this `a + b` on, the `Beta` node of positive shapes is `exp(ln B)`
/// ([`beta_wp`]) rather than `Γ(a)Γ(b)/Γ(a + b)`: `2²⁶`, below
/// `x ≈ 7.8·10⁷` where `Γ(x)` leaves the exponent range.
pub(super) const BETA_VIA_LN_MIN: f64 = 67_108_864.0;

/// The continued fraction of the regularised incomplete beta function
/// (DLMF 8.17.22; evaluated by the modified Lentz algorithm of Thompson &
/// Barnett, J. Comput. Phys. 64 (1986) 490):
///
/// ```text
/// I_x(a, b) = x^a (1−x)^b / (a B(a, b)) · 1/(1 + d₁/(1 + d₂/(1 + …)))
/// d_{2m}   =  m (b−m) x / ((a+2m−1)(a+2m))
/// d_{2m+1} = −(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
/// ```
///
/// Returns the fraction `1/(1 + d₁/…)` only (the prefactor is applied by
/// the caller).  Converges quickly for `x < (a+1)/(a+b+2)`.  Evaluated with
/// `log₂(2·budget) + 8` guard bits for the roundings of its steps and the
/// bits of `a + b`: near the switch point `1 − (a + b)x/(a + 1)` cancels
/// them, and where the mass lies within `1/a` of 1 they are the bits that
/// keep `x` away from 1.  Up to 0.57 the fraction rounded `x` to the working
/// precision alone: the survival function of `F(3·10¹²⁸, 1)` at `7.3·10⁻⁴`,
/// `I_y(1.5·10¹²⁸, ½)` with `1 − y = 4.6·10⁻¹²⁶`, ran its fraction at `y = 1`,
/// which does not converge, through its 2·10⁶ steps (5.7 s) before the
/// evaluation was refused; it converges in about ten steps at the bits of
/// `a + b`.
///
/// The shape-dependent part of the step budget, `10·√(a + b)`, is capped
/// at `2·10⁶` steps (seconds), and a fraction that needs more by
/// [`betainc_cf_steps`] is refused before its first step (up to 0.57
/// `F(10²⁰, 10²⁰).cdf(1)` ran 18 s to refuse).  For `a + b` beyond
/// `2^`[`BETAINC_CF_ENVELOPE_BITS`] the budget is
/// [`BETAINC_CF_WORK_BEYOND`] bit-steps instead, and the same refusal holds
/// against it.
fn betainc_cf(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let ab_f = f64_of(&a.add(b, 64, RM))?;
    let max_iter = (40 * wp + 2000).saturating_add((10.0 * ab_f.abs().sqrt()).min(2e6) as usize);
    // The convergence test as before; the steps also carry the bits of a + b.
    let tol = wp + magnitude_bits(2.0 * max_iter as f64) + 8;
    let ab_bits = magnitude_bits(ab_f);
    let wp = tol + ab_bits;
    // Beyond the envelope the budget is work (steps × working bits), fixed
    // before the first step: see `BETAINC_CF_WORK_BEYOND`.
    let max_iter = if ab_bits > BETAINC_CF_ENVELOPE_BITS {
        max_iter.min(BETAINC_CF_WORK_BEYOND / wp)
    } else {
        max_iter
    };
    let below = betainc_switch_point(a, b, wp).sub(x, wp, RM);
    // Float evidence: a work estimate that can only refuse (an `f64` guide).
    let z = betainc_sigmas(a, b, &below)?;
    let need = betainc_cf_steps(f64_of(a)?.min(f64_of(b)?), z, tol);
    if need > max_iter as f64 {
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: format!(
                "continued fraction for the incomplete beta function: about {need:.1e} steps \
                 needed, {max_iter} allowed"
            ),
        });
    }
    let (a, b, x) = (
        &at_precision(a, wp),
        &at_precision(b, wp),
        &at_precision(x, wp),
    );
    let one = BigFloat::from_i32(1, wp);
    // "tiny" replaces exact zeros in Lentz's algorithm.
    let mut tiny = BigFloat::from_i32(1, wp);
    tiny.set_exponent(-(4 * wp as i32));
    let qab = a.add(b, wp, RM);
    let qap = a.add(&one, wp, RM);
    let qam = a.sub(&one, wp, RM);
    let mut c = one.clone();
    let mut d = one.sub(&qab.mul(x, wp, RM).div(&qap, wp, RM), wp, RM);
    if d.is_zero() {
        d = tiny.clone();
    }
    d = one.div(&d, wp, RM);
    let mut h = d.clone();
    let mut converged = false;
    // One Lentz step with partial numerator `aa`; returns `del = d·c`.
    let step = |aa: &BigFloat, c: &mut BigFloat, d: &mut BigFloat| -> BigFloat {
        *d = one.add(&aa.mul(d, wp, RM), wp, RM);
        if d.is_zero() {
            *d = tiny.clone();
        }
        *c = one.add(&aa.div(c, wp, RM), wp, RM);
        if c.is_zero() {
            *c = tiny.clone();
        }
        *d = one.div(d, wp, RM);
        d.mul(c, wp, RM)
    };
    for m in 1..=max_iter {
        let m_bf = BigFloat::from_i128(m as i128, wp);
        let m2_bf = BigFloat::from_i128(2 * m as i128, wp);
        // Even step: d_{2m} = m (b−m) x / ((a+2m−1)(a+2m)).
        let num = m_bf.mul(&b.sub(&m_bf, wp, RM), wp, RM).mul(x, wp, RM);
        let den = qam.add(&m2_bf, wp, RM).mul(&a.add(&m2_bf, wp, RM), wp, RM);
        let aa = num.div(&den, wp, RM);
        let del = step(&aa, &mut c, &mut d);
        h = h.mul(&del, wp, RM);
        // Odd step: d_{2m+1} = −(a+m)(a+b+m) x / ((a+2m)(a+2m+1)).
        let num = a
            .add(&m_bf, wp, RM)
            .mul(&qab.add(&m_bf, wp, RM), wp, RM)
            .mul(x, wp, RM)
            .neg();
        let den = a.add(&m2_bf, wp, RM).mul(&qap.add(&m2_bf, wp, RM), wp, RM);
        let aa = num.div(&den, wp, RM);
        let del = step(&aa, &mut c, &mut d);
        h = h.mul(&del, wp, RM);
        let dev = del.sub(&one, wp, RM);
        if negligible(&dev, &one, tol) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: format!(
                "continued fraction for the incomplete beta function did not converge in \
                 {max_iter} steps"
            ),
        });
    }
    Ok(h)
}

/// `a + b ≤ 2⁵⁴` holds for all shapes inside the envelope in which the
/// statistics module must answer (parameters up to `2⁵³`): there the step
/// budget of [`betainc_cf`] stands as it is.
const BETAINC_CF_ENVELOPE_BITS: usize = 54;

/// The budget of [`betainc_cf`] beyond the envelope, in bit-steps (steps ×
/// working bits): there a step carries up to 1024 guard bits for `a + b`
/// and costs up to twenty times a step inside it, so a budget in steps
/// bounds nothing (up to 0.60 `betainc_regularized(10²⁵⁰, 10¹⁴, 0, x)` at
/// the mean ran 2 050 000 steps at 2060 bits, more than 40 s, to be refused).
/// `2²⁵` bit-steps are 28 000 steps at 1186 bits and 16 000 at 2060 bits:
/// 0.26 s and 0.34 s at the 9.2 µs and 20.8 µs per step measured there in
/// a release build.  The count is deterministic: no clock is read.
const BETAINC_CF_WORK_BEYOND: usize = 1 << 25;

/// `delta/σ` as an `f64` (`±∞` beyond its range), `σ² = ab/(a + b)³` the
/// variance of the beta law: the distance `delta` in standard deviations.
/// `σ` is formed in arbitrary precision, whose exponent range holds it for
/// all shapes.  Up to 0.60 it was formed in `f64`, where `σ²` is 0 from
/// `ab/(a + b)³ < 5·10⁻³²⁴` on (`a = 2·10¹³⁶`, `b = 2·10²⁵⁶`: `5·10⁻³⁷⁷`):
/// `delta/0 = ∞` standard deviations read as "far below the switch point",
/// the estimate of [`betainc_cf_steps`] was 0 steps where it is `10⁴⁷`, and
/// `FDistribution(4·10¹³⁶, 4·10²⁵⁶).cdf(1)` ran the fraction through its
/// 2 014 160 steps (18.5 s) before it was refused.
fn betainc_sigmas(a: &BigFloat, b: &BigFloat, delta: &BigFloat) -> Result<f64, SymplexError> {
    let p = 64;
    let n = a.add(b, p, RM);
    let variance = a
        .div(&n, p, RM)
        .mul(&b.div(&n, p, RM), p, RM)
        .div(&n, p, RM);
    f64_of(&delta.div(&variance.sqrt(p, RM), p, RM))
}

/// An estimate from below of the steps [`betainc_cf`] needs to converge to
/// `tol` bits, `z` standard deviations ([`betainc_sigmas`]) below its switch
/// point `(a + 1)/(a + b + 2)`, for the smaller shape `m`: an `f64` guide
/// that can only refuse.  Measured with mpmath's Lentz steps
/// (shapes `10⁴`–`10¹⁶`, `b/a` from `10⁻⁶` to `10²`, 74 to 235 bits): at the
/// switch point `0.39–0.41·(m·tol²)^{1/3}` steps, `m = min(a, b)` (up to
/// `0.65·(…)` when `b` is the smaller); `z = delta/σ` standard deviations
/// below it (`σ² = ab/(a + b)³`) `12.7–15.7·tol/z²` once that is the
/// smaller.  The estimate, `0.8` of the harmonic combination of
/// `0.39·(m·tol²)^{1/3}` and `12·tol/z²`, lies below every measured count
/// (221 counts of at least 20 steps: between 0.14 and 0.92 of it).
fn betainc_cf_steps(m: f64, z: f64, tol: usize) -> f64 {
    let tol = tol as f64;
    let at_switch = 0.39 * (m * tol * tol).cbrt();
    0.8 / (1.0 / at_switch + z * z / (12.0 * tol))
}

/// `x^a y^b / (a B(a, b))` for `0 < x < 1`, `y = 1 − x`, as one
/// exponential of `a ln x + b ln y − ln a − ln B(a, b)`: the exponent is
/// formed with the bits of its magnitude as guard, so that its absolute
/// error — the relative error of the result — is `2^−wp`, and `ln B` comes
/// from [`ln_beta_wp`].  0 when the exponent lies below the exponent range:
/// a tail that is 0 for no other reason, which the accuracy layer holds as
/// a nonzero number (`accuracy::underflowed`).  0.28 formed `x^a`, `y^b` and
/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` separately: `Γ(10⁸)` is beyond the exponent
/// range, and `betainc_regularized(10⁸ + ⅓, 10⁸ + 1/7, 0, ½)` was refused.
fn betainc_prefactor(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    y: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let a_f = f64_of(a)?;
    let b_f = f64_of(b)?;
    let ab_f = a_f + b_f;
    let mag = a_f * ln_magnitude(x)
        + b_f * ln_magnitude(y)
        + ln_magnitude(a)
        + lngamma_magnitude(a, a_f)
        + lngamma_magnitude(b, b_f)
        + ab_f * ab_f.ln().abs()
        + ab_f
        + 2.0;
    let p = wp + magnitude_bits(mag) + 8;
    let e = a
        .mul(&x.ln(p, RM, w.cc), p, RM)
        .add(&b.mul(&y.ln(p, RM, w.cc), p, RM), p, RM)
        .sub(&a.ln(p, RM, w.cc), p, RM)
        .sub(&ln_beta_wp(a, b, &mut w.at(p))?, p, RM);
    Ok(e.exp(wp, RM, w.cc))
}

/// `I_x(a, b)` for `x ≤ (a+1)/(a+b+2)` (`one_minus_x = 1 − x` is passed in
/// so that it is exact): continued fraction times `x^a (1−x)^b / (a B(a, b))`.
fn betainc_reg_direct(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    one_minus_x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let wp = w.bits();
    let h = betainc_cf(a, b, x, w)?;
    let pref = betainc_prefactor(a, b, x, one_minus_x, w)?;
    Ok(pref.mul(&h, wp, RM))
}

/// The switch point `(a+1)/(a+b+2)` of the continued fraction: below it
/// `I_x(a, b)` is computed directly, above it `I_{1−x}(b, a)`.
fn betainc_switch_point(a: &BigFloat, b: &BigFloat, wp: usize) -> BigFloat {
    let one = BigFloat::from_i32(1, wp);
    let two = BigFloat::from_i32(2, wp);
    let a_plus_1 = a.add(&one, wp, RM);
    let a_plus_b_plus_2 = a.add(b, wp, RM).add(&two, wp, RM);
    a_plus_1.div(&a_plus_b_plus_2, wp, RM)
}

/// From this many continued-fraction steps by the estimate
/// ([`betainc_cf_steps`], a lower bound: the fraction takes up to 7 times
/// as many) on, [`betainc_tails_central`] is tried first.  Below it the
/// fraction takes milliseconds.  Not applied for `a + b` beyond
/// `2^`[`BETAINC_CF_ENVELOPE_BITS`], where the route is always tried.
const BETAINC_CENTRAL_MIN_STEPS: f64 = 2_000.0;

/// The central route applies within this many standard deviations of the
/// mode; beyond, the continued fraction needs a few hundred steps at most.
const BETAINC_CENTRAL_MAX_Z: f64 = 6.0;

/// The most Taylor terms the central route takes (beyond, it declines and
/// the continued fraction is used).
const BETAINC_CENTRAL_MAX_TERMS: usize = 160;

/// Both tails `(I_x(a, b), I_{1−x}(b, a))` for large shapes (`a, b > 2`)
/// and `x` within [`BETAINC_CENTRAL_MAX_Z`] standard deviations of the mode,
/// where the continued fraction needs about `(min(a, b)·wp²)^{1/3}` steps
/// (`2·10⁶` at shapes `10¹⁶`: up to 0.57 `F(10¹⁶, 10¹⁶).cdf(1)` ran
/// 20 s to refuse).  `None` where the route does not apply or cannot
/// certify its truncation, and the caller uses the fraction.
///
/// The integrand `t^{a−1}(1 − t)^{b−1}` is taken relative to its value at
/// the mode `t₀ = (a − 1)/N`, `N = a + b − 2`, in the variable
/// `u = (t − t₀)/σ`, `σ² = t₀(1 − t₀)/N`: it is `e^{−u²/2}·E(u)`, `E =
/// exp ψ`, `ψ(u) = Σ_{k≥3} d_k uᵏ`, `d_k = ((−1)^{k−1}(1 − t₀)s^{k−2} −
/// t₀ r^{k−2})/k`, `s = σ/t₀`, `r = σ/(1 − t₀)` (Laplace's method with
/// the convergent Taylor series of `E` instead of an asymptotic one).  `E`
/// is analytic for `|u| < min(t₀, 1 − t₀)/σ`, so its coefficients `c_k`
/// (`k c_k = Σ j d_j c_{k−j}`) obey Cauchy's bound `|c_k| ≤ M(ρ)/ρᵏ` with
/// `ln M(ρ) ≤ Σ |d_k| ρᵏ`.  `L = Σ_{k<K} c_k ∫_{−∞}^{v} uᵏ e^{−u²/2}`
/// and `R = Σ_{k<K} c_k ∫_{v}^{∞} …` (`v = (x − t₀)/σ`, the moments by
/// their recurrence from `erfc`) give `I_x = L/(L + R)` and `I_{1−x}(b, a) =
/// R/(L + R)` (no Beta function is formed).  Their error, each bounded and
/// required below `2^−(wp+8)` of the value: the truncation on `|u| ≤ U`
/// (`√(2π)·M(ρ)(U/ρ)^K/(1 − U/ρ)`), the integrand beyond `±U` (concave
/// in the exponent: `e^{−U²/2 + B(U)}/(U − B′(U))`, `B(U) ≥ |ψ|`, `B′(U) ≥
/// |ψ′|` on `|u| ≤ U`), and the polynomial's Gaussian moments beyond `±U`.
/// The roundings of the recurrences are covered by guard bits from their
/// absolute-value twins.  The bounds are `f64` guides with 1 % slack.
fn betainc_tails_central(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<Option<(BigFloat, BigFloat)>, SymplexError> {
    use std::f64::consts::{LN_2, PI};
    let wp = w.bits();
    let (a_f, b_f) = (f64_of(a)?, f64_of(b)?);
    let n_f = a_f + b_f - 2.0;
    if !(a_f > 2.0 && b_f > 2.0 && n_f.is_finite()) {
        return Ok(None);
    }
    let (t0_f, s1_f) = ((a_f - 1.0) / n_f, (b_f - 1.0) / n_f);
    // σ and the radii as products of square roots: `t₀(1 − t₀)/N` itself is
    // below the range of an `f64` from `N` near `10¹⁰⁸` on (0, and the route
    // declined: `a = 2·10¹³⁶`, `b = 2·10²⁵⁶` up to 0.60), its factors'
    // roots are not (`t₀, 1 − t₀ > 1/N`, `N < 1.8·10³⁰⁸`).
    let (t0_rt, s1_rt, n_rt) = (t0_f.sqrt(), s1_f.sqrt(), n_f.sqrt());
    let sigma_f = t0_rt * s1_rt / n_rt;
    // The radii of convergence of E on the two sides (the ends 0 and 1):
    // t₀/σ and (1 − t₀)/σ (`∞` beyond the range: the far end).
    let (r0, r1) = (t0_rt / s1_rt * n_rt, s1_rt / t0_rt * n_rt);
    if !(sigma_f > 0.0 && r0.min(r1) > 64.0) {
        return Ok(None);
    }
    // v = (x − t₀)/σ at a relative 2^−p: t₀ to the bits that resolve x − t₀,
    // and N, a − 1, b − 1 exact for integer shapes.
    let mode = |p: usize| {
        let pn = p + magnitude_bits(n_f) + 8;
        let pt = pn + magnitude_bits(1.0 / sigma_f);
        let one = BigFloat::from_i32(1, pn);
        let am1 = a.sub(&one, pn, RM);
        let bm1 = b.sub(&one, pn, RM);
        let n = am1.add(&bm1, pn, RM);
        let t0 = am1.div(&n, pt, RM);
        let s1 = bm1.div(&n, pt, RM);
        let sigma = t0.mul(&s1, p, RM).div(&n, p, RM).sqrt(p, RM);
        let v = x.sub(&t0, pt, RM).div(&sigma, p, RM);
        (t0, s1, sigma, v)
    };
    let z = f64_of(&mode(64).3)?;
    if z.is_nan() || z.abs() > BETAINC_CENTRAL_MAX_Z {
        return Ok(None);
    }
    // Beyond the envelope the estimate does not gate the route: it is
    // calibrated for `b/a` from `10⁻⁶` to `10²` and lies far below the count
    // outside (about 1 200 steps estimated, more than 18 088 taken at
    // shapes near `10¹⁸⁷` and `10¹³`, 3.1 standard deviations above the mode,
    // at 1200 bits), and a step of the fraction costs up to twenty times a
    // step inside.
    let beyond = magnitude_bits(a_f + b_f) > BETAINC_CF_ENVELOPE_BITS;
    let steps = betainc_cf_steps(a_f.min(b_f), z, wp + 30);
    if !beyond && steps < BETAINC_CENTRAL_MIN_STEPS {
        return Ok(None);
    }
    // ln of the target absolute error: 2^−(wp+10) of a lower bound of the
    // smaller side (Mills' ratio; ∫_1^∞ e^{−u²/2} for |v| < 1).
    let ln_side = -z * z / 2.0 + (z.abs().max(1.0) / (z * z + 1.0)).ln() - LN_2;
    let ln_eps = ln_side - (wp as f64 + 10.0) * LN_2;
    // Σ_{j≥1} q₀ʲ(1 − t₀) + q₁ʲ t₀ at q = (ρ/r₀, ρ/r₁): |ψ| ≤ ρ²/3·S,
    // |ψ′| ≤ ρ·S on |u| ≤ ρ.
    let geo = |rho: f64| {
        let (q0, q1) = (rho / r0, rho / r1);
        1.01 * (s1_f * q0 / (1.0 - q0) + t0_f * q1 / (1.0 - q1))
    };
    // Terms for the truncation at radius ρ of Cauchy's bound.
    let terms_at = |u: f64, rho: f64| -> f64 {
        let ln_m = rho * rho / 3.0 * geo(rho);
        let ln_tail = ln_m + (2.0 * PI).sqrt().ln() - (1.0 - u / rho).ln() - ln_eps;
        (ln_tail / (rho / u).ln()).ceil().max(1.0)
    };
    let best_terms = |u: f64| -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        let mut rho = 1.5 * u;
        while rho < 0.5 * r0.min(r1) {
            let k = terms_at(u, rho);
            if best.is_none_or(|(bk, _)| k < bk) {
                best = Some((k, rho));
            }
            rho *= 1.25;
        }
        best
    };
    let mut u_cut = (z * z + 2.0 * LN_2 * (wp as f64 + 24.0)).sqrt() + 1.0;
    let Some((mut k_f, mut rho)) = best_terms(u_cut) else {
        return Ok(None);
    };
    if u_cut * u_cut < 2.0 * k_f {
        u_cut = (2.0 * k_f).sqrt();
        let Some(again) = best_terms(u_cut) else {
            return Ok(None);
        };
        (k_f, rho) = again;
    }
    if !(k_f <= BETAINC_CENTRAL_MAX_TERMS as f64 && u_cut * u_cut >= 2.0 * k_f) {
        return Ok(None);
    }
    let Some(terms) = crate::base::cast::f64_to_i64(k_f).and_then(|k| usize::try_from(k).ok())
    else {
        return Ok(None);
    };
    // The integrand beyond ±U, both sides alike.
    let s_u = geo(u_cut);
    let (b_u, bp_u) = (u_cut * u_cut / 3.0 * s_u, u_cut * s_u);
    if !u_cut.is_finite() || !bp_u.is_finite() || u_cut <= 1.01 * bp_u {
        return Ok(None);
    }
    // The three bounds as logarithms (at thousands of bits they are below
    // the range of an `f64`).
    let ln_beyond = -u_cut * u_cut / 2.0 + b_u - (u_cut - bp_u).ln();
    let ln_m = rho * rho / 3.0 * geo(rho);
    let ln_truncation = (2.0 * PI).sqrt().ln() + ln_m + (terms as f64) * (u_cut / rho).ln()
        - (1.0 - u_cut / rho).ln();
    // Absolute-value twins in f64: D_k ≥ |d_k|, C_k ≥ |c_k|, A_k ≥ both
    // moments of uᵏ; they bound the roundings and the moments beyond ±U.
    let (sf, rf) = (1.0 / r0, 1.0 / r1);
    let mut d_abs = vec![0.0f64; terms];
    for (k, d) in d_abs.iter_mut().enumerate().skip(3) {
        let j = (k - 2) as i32;
        *d = 1.01 * (s1_f * sf.powi(j) + t0_f * rf.powi(j)) / k as f64;
    }
    let mut c_abs = vec![0.0f64; terms];
    c_abs[0] = 1.0;
    for k in 1..terms {
        let s: f64 = (3..=k).map(|j| j as f64 * d_abs[j] * c_abs[k - j]).sum();
        c_abs[k] = s / k as f64;
    }
    let e_f = (-z * z / 2.0).exp();
    let mut a_abs = vec![0.0f64; terms];
    a_abs[0] = (2.0 * PI).sqrt();
    if terms > 1 {
        a_abs[1] = 1.0;
    }
    for k in 2..terms {
        a_abs[k] = (k - 1) as f64 * a_abs[k - 2] + z.abs().powi((k - 1) as i32) * e_f;
    }
    let outside: f64 = (0..terms)
        .map(|k| {
            c_abs[k] * u_cut.powi(k as i32 - 1)
                / (1.0 - (k as f64 - 1.0).max(0.0) / (u_cut * u_cut))
        })
        .sum();
    let ln_outside = outside.ln() - u_cut * u_cut / 2.0;
    let rounding_mass: f64 = (0..terms)
        .map(|k| 8.0 * ((k + 2) * (k + 2)) as f64 * c_abs[k] * a_abs[k])
        .sum();
    let ln_top = ln_truncation.max(ln_beyond).max(ln_outside);
    let ln_err = ln_top
        + ((ln_truncation - ln_top).exp()
            + (ln_beyond - ln_top).exp()
            + (ln_outside - ln_top).exp())
        .ln()
        + 0.01;
    if !(ln_err.is_finite() && rounding_mass.is_finite()) {
        return Ok(None);
    }
    // Guard bits for the roundings: 2^−q·rounding_mass below the target.
    let q = wp + 48 + magnitude_bits(rounding_mass / ln_side.exp());
    let (t0, s1, sigma, v) = mode(q);
    let s = sigma.div(&t0, q, RM);
    let r = sigma.div(&s1, q, RM);
    let one = BigFloat::from_i32(1, q);
    // d_k (k ≥ 3), then c_k by k c_k = Σ j d_j c_{k−j}.
    let mut d = vec![BigFloat::new(q); terms];
    let (mut sp, mut rp) = (s.clone(), r.clone());
    for (k, dk) in d.iter_mut().enumerate().skip(3) {
        let left = s1.mul(&sp, q, RM);
        let right = t0.mul(&rp, q, RM);
        let num = if k % 2 == 1 {
            left.sub(&right, q, RM)
        } else {
            left.neg().sub(&right, q, RM)
        };
        *dk = num.div(&BigFloat::from_i128(k as i128, q), q, RM);
        sp = sp.mul(&s, q, RM);
        rp = rp.mul(&r, q, RM);
    }
    let mut c = vec![BigFloat::new(q); terms];
    c[0] = one.clone();
    for k in 1..terms {
        let mut acc = BigFloat::new(q);
        for j in 3..=k {
            let t = d[j]
                .mul(&c[k - j], q, RM)
                .mul(&BigFloat::from_i128(j as i128, q), q, RM);
            acc = acc.add(&t, q, RM);
        }
        c[k] = acc.div(&BigFloat::from_i128(k as i128, q), q, RM);
    }
    // The moments over (−∞, v] and [v, ∞).
    let half = one.div(&BigFloat::from_i32(2, q), q, RM);
    let e = v.mul(&v, q, RM).mul(&half, q, RM).neg().exp(q, RM, w.cc);
    let sqrt_half_pi = w.cc.pi(q, RM).clone().mul(&half, q, RM).sqrt(q, RM);
    let arg = v.mul(&half.sqrt(q, RM), q, RM);
    let (Ok(erfc_hi), Ok(erfc_lo)) = (
        arb_erfc(&arg, &mut w.at(q)),
        arb_erfc(&arg.neg(), &mut w.at(q)),
    ) else {
        return Ok(None);
    };
    let mut lower_m = vec![BigFloat::new(q); terms];
    let mut upper_m = vec![BigFloat::new(q); terms];
    lower_m[0] = sqrt_half_pi.mul(&erfc_lo, q, RM);
    upper_m[0] = sqrt_half_pi.mul(&erfc_hi, q, RM);
    if terms > 1 {
        lower_m[1] = e.neg();
        upper_m[1] = e.clone();
    }
    let mut vpow = v.mul(&e, q, RM);
    for k in 2..terms {
        let km1 = BigFloat::from_i128((k - 1) as i128, q);
        lower_m[k] = km1.mul(&lower_m[k - 2], q, RM).sub(&vpow, q, RM);
        upper_m[k] = km1.mul(&upper_m[k - 2], q, RM).add(&vpow, q, RM);
        vpow = vpow.mul(&v, q, RM);
    }
    let mut lo_sum = BigFloat::new(q);
    let mut hi_sum = BigFloat::new(q);
    for ((ck, lm), um) in c.iter().zip(&lower_m).zip(&upper_m) {
        lo_sum = lo_sum.add(&ck.mul(lm, q, RM), q, RM);
        hi_sum = hi_sum.add(&ck.mul(um, q, RM), q, RM);
    }
    let (lo_f, hi_f) = (f64_of(&lo_sum)?, f64_of(&hi_sum)?);
    let enough = |s: f64| s > 0.0 && ln_err <= s.ln() - (wp as f64 + 8.0) * LN_2;
    if !(enough(lo_f) && enough(hi_f)) {
        return Ok(None);
    }
    let total = lo_sum.add(&hi_sum, q, RM);
    let lower = round_to(lo_sum.div(&total, q, RM), wp);
    let upper = round_to(hi_sum.div(&total, q, RM), wp);
    Ok(Some((lower, upper)))
}

/// `(I_x(a, b), I_{1−x}(b, a))` for `a, b > 0`, `0 < x < 1`: both tails of
/// the regularised incomplete beta function, each to a relative error of
/// about `2^−wp`.
///
/// The continued fraction converges fast on the side of the smaller
/// argument: it gives `I_x(a, b)` directly for `x ≤ (a+1)/(a+b+2)` and
/// `I_{1−x}(b, a)` above, and the other tail is `1 −` it.  That difference
/// cancels when the directly computed tail is close to 1: a small first
/// shape piles the mass against 0 (`I_{1/10}(10⁻⁶⁰, ⅓) = 1 − 4.8·10⁻⁶⁰`).
/// 0.28 assumed at most a bit was lost there and printed garbage as
/// certified digits: `betainc_regularized(⅓, 10⁻¹⁰⁰, 0, 9/10)` was
/// `1.28·10⁻⁸²`, truly `4.79·10⁻¹⁰⁰`.  The loss is measured instead: the
/// direct tail `d`, computed to a relative `2^−p`, leaves `1 − d` with an
/// absolute error of `2^−p`, i.e. `p − log₂(1/(1 − d))` correct bits, and
/// `d` is recomputed at a precision covering the loss (a first shape
/// `s < 1` costs about `log₂(1/s)` bits, which are added up front), up to
/// [`cancellation_cap`].
fn betainc_tails_wp(
    a: &BigFloat,
    b: &BigFloat,
    x: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    if let Some(tails) = betainc_tails_central(a, b, x, w)? {
        return Ok(tails);
    }
    let direct_lower = !bf_gt(x, &betainc_switch_point(a, b, wp));
    let first = if direct_lower { a } else { b };
    let ab_bits = magnitude_bits(f64_of(&a.add(b, 64, RM))?);
    let cap = cancellation_cap(wp);
    // The loss is about log₂(1/s) for a first shape s < 1 (less by at most
    // log₂|ln x|): beyond the cap, refuse before computing at that precision.
    let mut extra = 16 + inverse_bits(first);
    if extra > cap {
        return Err(special_exhausted(wp));
    }
    loop {
        let p = wp + extra;
        // 1 − x: exact for x ≥ ½ (the argument's own bits); rounded far
        // below the unit of `b ln y` otherwise.
        let py = p + 64 + ab_bits;
        let y = BigFloat::from_i32(1, py).sub(x, py, RM);
        let d = if direct_lower {
            betainc_reg_direct(a, b, x, &y, &mut w.at(p))?
        } else {
            betainc_reg_direct(b, a, &y, x, &mut w.at(p))?
        };
        let c = BigFloat::from_i32(1, p).sub(&d, p, RM);
        let lost = match c.exponent() {
            Some(e) if bf_strictly_positive(&c) => usize::try_from(1 - i64::from(e)).unwrap_or(0),
            _ => p,
        };
        if lost + 8 <= extra {
            let (d, c) = (round_to(d, wp), round_to(c, wp));
            return Ok(if direct_lower { (d, c) } else { (c, d) });
        }
        if extra >= cap {
            return Err(special_exhausted(wp));
        }
        extra = (lost + 16).max(2 * extra).min(cap);
    }
}

/// Both tails at a limit `t ∈ [0, 1]` (exact at the ends).
fn betainc_tails_at(
    a: &BigFloat,
    b: &BigFloat,
    t: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, BigFloat), SymplexError> {
    let wp = w.bits();
    let one = BigFloat::from_i32(1, wp);
    if t.is_zero() {
        return Ok((BigFloat::new(wp), one));
    }
    if t.cmp(&one) == Some(0) {
        return Ok((one, BigFloat::new(wp)));
    }
    betainc_tails_wp(a, b, t, w)
}

/// `I_{hi}(a, b) − I_{lo}(a, b)` for `0 ≤ lo < hi ≤ 1` at working precision
/// `wp`, together with the number of bits lost to cancellation in the
/// subtraction (`wp` when the difference vanished entirely).
///
/// Both tails are known at each limit ([`betainc_tails_at`]), so the
/// difference is `L(hi) − L(lo) = U(lo) − U(hi)` with `L(t) = I_t(a, b)`
/// and `U(t) = I_{1−t}(b, a)`, and the form whose larger term is the smaller
/// is taken: no `1 − …` is formed here, so a lower tail (`lo = 0`) or an
/// upper one (`hi = 1`) is a single directly computed tail, and limits near
/// 1 keep their relative accuracy.  What remains is the cancellation of
/// close limits, which the caller covers with precision.
fn betainc_reg_diff_wp(
    a: &BigFloat,
    b: &BigFloat,
    lo: &BigFloat,
    hi: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let (l_lo, u_lo) = betainc_tails_at(a, b, lo, w)?;
    let (l_hi, u_hi) = betainc_tails_at(a, b, hi, w)?;
    let (p, q) = if bf_gt(&l_hi, &u_lo) {
        (u_lo, u_hi)
    } else {
        (l_hi, l_lo)
    };
    let diff = p.sub(&q, wp, RM);
    let lost = match (p.exponent(), q.exponent(), diff.exponent()) {
        _ if q.is_zero() => 0,
        _ if !bf_strictly_positive(&diff) => wp,
        (Some(ep), Some(eq), Some(ed)) => {
            usize::try_from(i64::from(ep.max(eq)) - i64::from(ed)).unwrap_or(0)
        }
        _ => 0,
    };
    Ok((diff, lost))
}

/// `B_{(x1, x2)}(a, b) = ∫_{x1}^{x2} t^{a−1}(1 − t)^{b−1} dt` (`regularized`:
/// over `B(a, b) = Γ(a)Γ(b)/Γ(a + b)`) for real shapes of which one is `≤ 0`,
/// where the integral converges: no limit at 0 for `a ≤ 0`, none at 1 for
/// `b ≤ 0` (refused before 0.56, "requires a, b > 0"; `betainc(−½, 2, ¼, ½)
/// = 0.7573593129`, mpmath and quadrature).  With a limit at a singular end
/// the integral diverges and stays refused (mpmath's `betainc` gives a
/// continuation there, `betainc(−½, 2, 0, ½) = −4.24`, not the integral);
/// the regularized function is refused where `B(a, b)` is infinite or 0
/// (a pole of `Γ(a)`, `Γ(b)` or `Γ(a + b)`: no standard value).
///
/// The integral is split at ½, and each piece is a power series in the
/// variable nearer its end: `∫_u^v t^{p−1}(1 − t)^{q−1} dt = Σ_j c_j·I_j` with
/// `c_j = (1 − q)_j/j!` and `I_j = ∫_u^v t^{p+j−1} dt = (v^{p+j} −
/// u^{p+j})/(p + j)` (`ln(v/u)` at `p + j = 0`), `0 ≤ u < v ≤ ½`, and the
/// piece on `[½, 1]` the same in `1 − t` with the shapes exchanged.  Since `I_{j+1} ≤
/// v·I_j` and `|c_{j+1}/c_j| ≤ 1 + |q|/(j + 1)`, every term after the `j`-th
/// is at most `ρ = v(1 + |q|/(j + 1))` times the one before, and the sum stops
/// once `|T_j|·ρ/(1 − ρ)` is below `2^−wp` of the largest term.  The bits lost
/// to the differences `v^s − u^s`, to a small `p + j`, and to terms of both
/// signs are measured and made up.
fn betainc_shape_le_zero(
    a: &BigFloat,
    b: &BigFloat,
    x1: &BigFloat,
    x2: &BigFloat,
    regularized: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let (lo, hi, negate) = match x1.cmp(x2) {
        Some(c) if c < 0 => (x1, x2, false),
        Some(c) if c > 0 => (x2, x1, true),
        _ => return Ok(BigFloat::new(prec)),
    };
    let one = BigFloat::from_i32(1, prec);
    // B(a, b), when regularized: finite and nonzero.
    let gamma_pole = |x: &BigFloat| x.is_zero() || (x.is_negative() && x.is_int());
    let ab = a.add(b, prec + 64, RM);
    let b_pole = regularized && (gamma_pole(a) || gamma_pole(b) || gamma_pole(&ab));
    // The integral of a positive function diverges: infinite, unless it is
    // divided by a `B(a, b)` that is infinite or 0 as well.
    let diverges = |reason: &str| {
        if b_pole {
            unevaluable(reason)
        } else {
            infinite(reason)
        }
    };
    if !bf_strictly_positive(a) && lo.is_zero() {
        return Err(diverges("the integral diverges at 0 for a shape a ≤ 0"));
    }
    if !bf_strictly_positive(b) && hi.cmp(&one) == Some(0) {
        return Err(diverges("the integral diverges at 1 for a shape b ≤ 0"));
    }
    if b_pole {
        return Err(unevaluable(
            "B(a, b) is infinite or 0 (a pole of Gamma): no regularized value",
        ));
    }
    let cap = cancellation_cap(prec);
    let mut extra = 32usize;
    loop {
        let wp = prec + extra;
        let half = BigFloat::from_f64(0.5, wp);
        let onew = BigFloat::from_i32(1, wp);
        let mut parts: Vec<BigFloat> = Vec::new();
        let mut lost = 0usize;
        if bf_gt(&half, lo) {
            let v = if bf_gt(hi, &half) { &half } else { hi };
            let (s, l) = betainc_piece(a, b, lo, v, &mut w.at(wp))?;
            parts.push(s);
            lost = lost.max(l);
        }
        if bf_gt(hi, &half) {
            let u = if bf_gt(lo, &half) { lo } else { &half };
            // 1 − t at the bits of t (exact for t ≥ ½).
            let p1 = wp + 64 + hi.mantissa_max_bit_len().unwrap_or(wp);
            let (s, l) = betainc_piece(
                b,
                a,
                &onew.sub(hi, p1, RM),
                &onew.sub(u, p1, RM),
                &mut w.at(wp),
            )?;
            parts.push(s);
            lost = lost.max(l);
        }
        let mut total = BigFloat::new(wp);
        for s in &parts {
            total = total.add(s, wp, RM);
        }
        let top = parts.iter().filter_map(|s| s.exponent()).max();
        if let (Some(t), Some(e)) = (top, total.exponent()) {
            lost = lost.max(usize::try_from(i64::from(t) - i64::from(e)).unwrap_or(0));
        } else if !parts.iter().all(BigFloat::is_zero) {
            lost = lost.max(wp);
        }
        if lost + 16 <= extra {
            let mut v = total;
            if regularized {
                let p = wp + 32;
                let ga = arb_gamma_real(a, &mut w.at(p))?;
                let gb = arb_gamma_real(b, &mut w.at(p))?;
                let gab = arb_gamma_real(&a.add(b, p, RM), &mut w.at(p))?;
                v = v.mul(&gab, wp, RM).div(&ga.mul(&gb, p, RM), wp, RM);
            }
            if negate {
                v = v.neg();
            }
            return Ok(round_to(v, prec));
        }
        if extra >= cap {
            return Err(special_exhausted(prec));
        }
        extra = (lost + 32).max(2 * extra).min(cap);
    }
}

/// `∫_u^v t^{p−1}(1 − t)^{q−1} dt` for `0 ≤ u < v ≤ ½` (`u > 0` when `p ≤
/// 0`) by the series of [`betainc_shape_le_zero`], and the bits it lost.
fn betainc_piece(
    p: &BigFloat,
    q: &BigFloat,
    u: &BigFloat,
    v: &BigFloat,
    w: &mut Wp<'_>,
) -> Result<(BigFloat, usize), SymplexError> {
    let wp = w.bits();
    let qf = f64_of(q)?.abs();
    let pf = f64_of(p)?.abs();
    if !(qf.is_finite() && pf.is_finite()) || qf > 1e5 || pf > 1e15 {
        return Err(unevaluable("shapes out of range for the series"));
    }
    let vf = f64_of(v)?;
    let one = BigFloat::from_i32(1, wp);
    // v^p, u^p and then v^{p+j}, u^{p+j} by products.
    let mut pw = |x: &BigFloat| -> BigFloat {
        if x.is_zero() {
            BigFloat::new(wp)
        } else {
            p.mul(&x.ln(wp, RM, w.cc), wp, RM).exp(wp, RM, w.cc)
        }
    };
    let (mut vs, mut us) = (pw(v), pw(u));
    let log_ratio = if u.is_zero() {
        BigFloat::new(wp)
    } else {
        v.div(u, wp, RM).ln(wp, RM, w.cc)
    };
    let mut c = one.clone();
    let mut sum = BigFloat::new(wp);
    let mut top: Option<i32> = None;
    let mut lost = 0usize;
    let limit = 4 * wp + 4 * (qf as usize) + 64;
    for j in 0..limit {
        let jb = BigFloat::from_u64(j as u64, wp);
        let s = p.add(&jb, wp + 64, RM);
        let ij = if s.is_zero() {
            log_ratio.clone()
        } else {
            let d = vs.sub(&us, wp, RM);
            // The difference and a small p + j lose bits.
            if let (Some(ev), Some(ed)) = (vs.exponent(), d.exponent()) {
                lost = lost.max(usize::try_from(i64::from(ev) - i64::from(ed)).unwrap_or(0));
            }
            if let (Some(ep), Some(es)) = (p.exponent(), s.exponent()) {
                lost = lost.max(usize::try_from(i64::from(ep) - i64::from(es)).unwrap_or(0));
            }
            d.div(&s, wp, RM)
        };
        let term = c.mul(&ij, wp, RM);
        sum = sum.add(&term, wp, RM);
        if let Some(e) = term.exponent().filter(|_| !term.is_zero()) {
            top = Some(top.map_or(e, |t| t.max(e)));
        }
        // c_{j+1} = c_j·(1 − q + j)/(j + 1); a terminating series ends here.
        let next = one.sub(q, wp, RM).add(&jb, wp, RM);
        c = c
            .mul(&next, wp, RM)
            .div(&BigFloat::from_u64(j as u64 + 1, wp), wp, RM);
        if c.is_zero() {
            return Ok((sum, lost));
        }
        vs = vs.mul(v, wp, RM);
        us = us.mul(u, wp, RM);
        let rho = vf * (1.0 + qf / (j as f64 + 1.0));
        if rho < 1.0 && !term.is_zero() {
            let tail = f64::from(term.exponent().unwrap_or(0)) + (rho / (1.0 - rho)).log2();
            let scale = f64::from(top.unwrap_or(0));
            if tail < scale - wp as f64 - 4.0 {
                let total_lost = match (top, sum.exponent()) {
                    (Some(t), Some(e)) => usize::try_from(i64::from(t) - i64::from(e)).unwrap_or(0),
                    _ => wp,
                };
                // The roundings of c_j accumulate over the terms.
                let terms = usize::try_from(accuracy::ceil_log2(j + 2)).unwrap_or(0);
                return Ok((sum, lost.max(total_lost) + terms));
            }
        }
    }
    Err(unevaluable(
        "the series did not converge within its term budget",
    ))
}

/// Generalised incomplete beta `B_{(x1, x2)}(a, b) = ∫_{x1}^{x2} t^{a−1}(1−t)^{b−1} dt`
/// (or `I_{(x1, x2)}(a, b) = B_{(x1, x2)}(a, b)/B(a, b)` when `regularized`)
/// for real `0 ≤ x1, x2 ≤ 1`: shapes `a, b > 0` here, one `≤ 0` by
/// [`betainc_shape_le_zero`].
///
/// Computed as `I_{x2}(a, b) − I_{x1}(a, b)` ([`betainc_reg_diff_wp`]); when
/// the two values nearly cancel (`x1 ≈ x2`) the working precision is raised
/// by the number of bits lost and the difference recomputed, until the
/// loss is covered — or refused beyond [`cancellation_cap`].  0.28 returned
/// its third attempt whatever it had lost.  For `x1 ≠ x2` the result is 0
/// only when the tails underflowed ([`betainc_prefactor`]): an underflow,
/// not an exact 0 (`accuracy::underflowed`; up to 0.57
/// `betainc_regularized(1.4·10³⁰, 18, 0, ½)` printed `0`).
#[allow(clippy::too_many_arguments)]
pub(super) fn arb_betainc(
    a: &BigFloat,
    b: &BigFloat,
    x1: &BigFloat,
    x2: &BigFloat,
    regularized: bool,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    let name = if regularized {
        "betainc_regularized"
    } else {
        "betainc"
    };
    if [a, b, x1, x2].iter().any(|v| v.is_nan() || v.is_inf()) {
        return Err(unevaluable(format!("{name} of special float value")));
    }
    let one = BigFloat::from_i32(1, prec);
    if [x1, x2].iter().any(|x| x.is_negative() || bf_gt(x, &one)) {
        return Err(unevaluable(format!(
            "{name}: the limits x1, x2 must lie in [0, 1] for numerical evaluation"
        )));
    }
    if !bf_strictly_positive(a) || !bf_strictly_positive(b) {
        return betainc_shape_le_zero(a, b, x1, x2, regularized, w).map_err(|e| match e {
            SymplexError::Unevaluable { reason } => unevaluable(format!("{name}: {reason}")),
            SymplexError::Infinite { reason } => infinite(format!("{name}: {reason}")),
            other => other,
        });
    }
    // Orient the limits so that lo < hi; the sign is restored at the end.
    let (lo, hi, negate) = match x1.cmp(x2) {
        Some(c) if c < 0 => (x1, x2, false),
        Some(c) if c > 0 => (x2, x1, true),
        _ => return Ok(BigFloat::new(prec)),
    };
    let cap = prec + cancellation_cap(prec);
    let mut wp = prec + 32;
    let mut diff = loop {
        let (d, lost) =
            betainc_reg_diff_wp(a, b, lo, hi, &mut w.at(wp)).map_err(requested_at(prec))?;
        // Done once the guard bits cover the cancellation (16 to spare).
        if lost + prec + 16 <= wp {
            break d;
        }
        if wp >= cap {
            return Err(special_exhausted(prec));
        }
        wp = (prec + 32 + lost).min(cap);
    };
    if !regularized {
        let beta = beta_wp(a, b, &mut w.at(wp))?;
        diff = diff.mul(&beta, wp, RM);
    }
    if negate {
        diff = diff.neg();
    }
    Ok(round_to(diff, prec))
}
