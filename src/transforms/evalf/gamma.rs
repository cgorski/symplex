//! Arbitrary-precision Gamma via Stirling series (part of [`evalf`](super)).
//!
//! # The error of `ln Γ`, and of `Γ`
//!
//! `Γ(x) = exp(ln Γ(x))`: the *absolute* error of the logarithm is the
//! *relative* error of the value.  [`stirling_log_gamma`] sums at `wp =
//! prec + 20` bits, and its sum is known relative to its *terms*, which are
//! as large as `ln Γ` itself (`x·ln x`): `2²⁰` at `x = 10⁵`, `2³⁰` at `x =
//! 7·10⁷`.  Up to 0.60 `Γ(x)` was `exp` of that sum at `prec` bits and the
//! node's bound gave it two units in the last place (`accuracy::finish`);
//! it had `log₂ ln Γ(x) − 20` bits fewer, from `x ≈ 3·10⁵` on (10 at
//! `7·10⁷`).  The digits of `Γ(x)` itself survived on the slack of the
//! working precision; a cancellation of 50 to 62 bits showed the rest:
//! `gamma(280000001/4) − 17854627671268857·2^1723270899` at 32 digits ended
//! in `…4717` (truly `…4705`, 12.3 units of the last digit), and the same
//! through `x!`, `binomial` and `beta`, which are quotients of such values.
//!
//! What the sum is known to is derived from what was summed, not assumed:
//!
//! * **The terms.**  With `s = z + shift ≥ wp/2` the summands are `(s −
//!   ½)·ln s`, `−s`, `½·ln 2π`, the terms `B₂ₖ/(2k(2k−1)s^(2k−1))` and, for a
//!   shifted argument, `−ln Π_{i<shift}(z + i)`.  Each is within
//!   `100·(k + 1)·2^(−wp)` of itself: the first three are a handful of
//!   correctly rounded operations on `s` (rounded once) and on `π`; the
//!   `k`-th term of the series is `k − 1` products by `s²`, a product and
//!   a quotient, `(5k + 3)·2^(−wp)` at most; the product of the `shift`
//!   factors is within `shift·2^(−wp)` of itself, so its logarithm is
//!   within `(shift + 1)·2^(−wp)` *absolutely* — charged as `shift` more
//!   steps on every summand ([`Summed::steps`]), which covers it because
//!   the summands include `s ≥ 10`.
//! * **The sum.**  `n` summands, each partial sum rounded once: with the
//!   above, `2^(−wp)·2⁷·(n + shift + 1)·Σ 2^(e_k)` ([`Summed::error`]).
//! * **The tail.**  For real `z > 0` the remainder of Stirling's series is
//!   smaller than the first term left out and has its sign (DLMF 5.11(ii)):
//!   the term at which the loop stops, doubled for its own rounding.  A
//!   loop that ends without having stopped at a term has no bound.
//!
//! [`stirling_log_gamma`] charges the sum of the three to the [`Wp`] it was
//! called with ([`Wp::charge`]): an absolute error of `ln Γ(z)`.  A caller
//! that takes the logarithm as an ingredient at `w.at(p)` drops the charge
//! and says itself what `p` gives it (`arb_log_gamma` and `euler_positive`
//! carry the bits of the terms in `p` and measure what the value lost).  A
//! caller that takes `exp` of the logarithm needs its absolute error
//! ([`arb_gamma_real`], `ln_beta_wp`, `ln_incomplete_gamma`) and asks
//! [`log_gamma_abs`], which reads the charge, compares it with what was
//! asked and repeats with the missing bits — so the bound is the one
//! derived from the terms, whatever the estimate that chose the first
//! precision.
//!
//! [`arb_gamma_real`] asks for `ln Γ` to `2^(−prec−16)` and forms `exp`, and
//! on the reflection side `π/(sin(πd)·Γ(1 − x))`, at `prec + 16` bits, then
//! rounds once: its value is within `(½ + 2⁻¹²)` units in the last place of
//! `Γ(x)`, at every argument.  That is a value the two units of
//! `accuracy::finish` are true of, and one a quotient of several such values
//! (`binomial`, `beta`, the prefactors of the Bessel and hypergeometric
//! routes) can take as correctly rounded — which is why [`arb_gamma_real`]
//! itself passes no charge on: an absolute error of `Γ(n + 1)` is not an
//! error of `C(n, k)`, and a route that measures its error and makes up the
//! bits needs none.

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
///
/// The sum is formed at `prec + 20` bits and is known relative to its
/// terms, not to `2^−prec`: its absolute error, derived from the terms
/// (see the module documentation), is charged to `w` ([`Wp::charge`]) and
/// is about `2^(−prec − 20)·100·n·(z + r)·ln(z + r)`.  A caller that needs
/// an absolute error uses [`log_gamma_abs`].
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
    // (astro-float builds an `i128` only at 128 bits or more: NaN below,
    // which up to 0.60 was the value of the whole sum under 108 bits.)
    let int_bits = wp.max(128);
    let shift_bf = BigFloat::from_i128(shift as i128, int_bits);
    let z_shifted = z.add(&shift_bf, wp, RM);

    // Main Stirling formula:
    //   log Γ(z) = (z − 1/2)·ln(z) − z + (1/2)·ln(2π) + Σ ...
    let half = BigFloat::from_f64(0.5, wp);
    let z_minus_half = z_shifted.sub(&half, wp, RM);
    let log_z = z_shifted.ln(wp, RM, w.cc);

    let mut result = z_minus_half.mul(&log_z, wp, RM);
    // What is summed, for the error of the sum (the module documentation).
    let mut seen = Summed::of(&[&result, &z_shifted]);
    result = result.sub(&z_shifted, wp, RM);

    // (1/2) · ln(2π) at working precision.
    let two = BigFloat::from_i32(2, wp);
    let pi_val = w.cc.pi(wp, RM).clone();
    let two_pi = two.mul(&pi_val, wp, RM);
    let half_log_2pi = two_pi.ln(wp, RM, w.cc).mul(&half, wp, RM);
    seen.term(&half_log_2pi);
    result = result.add(&half_log_2pi, wp, RM);

    // Series: Σ_{k=1}^{N} B_{2k} / [2k(2k−1) · z^{2k−1}]
    let z_sq = z_shifted.mul(&z_shifted, wp, RM);
    let mut z_pow = z_shifted.clone(); // z^1 → z^3 → z^5 → …

    // Upper bound on useful terms: safely below the divergence point πz.
    let max_terms = ((wp as f64) * 0.35) as usize + 10;
    let mut prev_term_exp: Option<i32> = None;
    // The remainder: below the first term left out (twice it, for its own
    // rounding); unknown when the loop ends without one.
    let mut tail = accuracy::UNKNOWN;

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
        let denom = BigFloat::from_i128(denom_int, int_bits);
        let denom_full = denom.mul(&z_pow, wp, RM);
        let term = b2k_bf.div(&denom_full, wp, RM);

        // Divergence / convergence detection via binary exponents.
        if let Some(t_exp) = term.exponent() {
            if let Some(prev) = prev_term_exp
                && t_exp > prev + 10
            {
                tracing::trace!(k, "stirling series: diverging, stopping");
                tail = accuracy::part_lg(&term) + 1.0;
                break;
            }
            prev_term_exp = Some(t_exp);

            // Term is negligible relative to accumulated result.
            if let Some(r_exp) = result.exponent()
                && (r_exp as i64 - t_exp as i64) > wp as i64
            {
                tracing::trace!(k, "stirling series: converged");
                tail = accuracy::part_lg(&term) + 1.0;
                break;
            }
        }

        seen.term(&term);
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
            let z_plus_i = z.add(&BigFloat::from_i128(i as i128, int_bits), wp, RM);
            product = product.mul(&z_plus_i, wp, RM);
        }
        let ln_product = product.ln(wp, RM, w.cc);
        seen.term(&ln_product);
        seen = seen.steps(shift);
        result = result.sub(&ln_product, wp, RM);
    }

    w.charge(accuracy::lsum(seen.error(wp, 0.0), tail));
    Ok(result)
}

/// The bits [`arb_gamma_real`] carries beyond its precision: `ln Γ` to an
/// absolute error of `2^(−prec − GAMMA_GUARD)`, `exp` and the reflection at
/// `prec + GAMMA_GUARD` bits.
const GAMMA_GUARD: usize = 16;

/// How often [`log_gamma_abs`] makes up missing bits before it refuses.
/// (The first precision covers the bound it will read back; a second pass
/// is for an estimate that fell short.)
const LOG_GAMMA_PASSES: usize = 4;

/// An upper estimate of `log₂(2^wp·err)`, `err` the absolute error
/// [`stirling_log_gamma`] charges for `z > 0` at `wp` working bits: `Σ
/// 2^(e_k) ≤ 4·s·ln s + 2s + 2·abs(ln z) + 2` over the summands (`s ≤
/// max(z, wp/2 + 2)` the shifted argument, `2^(e_k) ≤ 2·abs(t_k)`), at most
/// `0.85·wp + 16` summands and steps, and the `2⁷` of [`Summed::error`].  It
/// only chooses the first precision: the bound is read from the charge.
fn stirling_error_lg(z: &BigFloat, wp: usize) -> Result<f64, SymplexError> {
    let z_f = f64_of(z)?;
    let s = z_f.max(0.5 * wp as f64 + 2.0);
    // `abs(ln z)` of a tiny `z` from its exponent (`z` may lie below `f64`).
    let ln_z = if z_f < 1.0 { ln_magnitude(z) } else { 0.0 };
    let terms = 4.0 * s * s.ln() + 2.0 * s + 2.0 * ln_z + 2.0;
    Ok(terms.log2() + (0.85 * wp as f64 + 16.0).log2() + 7.0 + 0.01)
}

/// `ln Γ(x)` for real `x > 0` to an absolute error below `2^(−bits)` — the
/// relative error of `exp` of it.  [`stirling_log_gamma`] is run with the
/// bits of its terms on top (an estimate, [`stirling_error_lg`]); the
/// bound it derived from the terms is read back, and the sum is repeated
/// with the missing bits if that bound is larger than what was asked.
///
/// A logarithm beyond `2³²` whose bound is a part of it is returned as it
/// is: `Γ(x)` lies beyond the exponent range (`ln Γ > 2³¹·ln 2`) whatever
/// the error, and `exp` of it is the `∞` the callers handle.
///
/// # Errors
///
/// [`SymplexError::PrecisionExhausted`] when the bits needed exceed
/// [`cancellation_cap`] or the sum has no bound.
pub(super) fn log_gamma_abs(
    x: &BigFloat,
    bits: usize,
    w: &mut Wp<'_>,
) -> Result<BigFloat, SymplexError> {
    let asked = Prec::new(bits);
    let cap = asked.plus(cancellation_cap(bits));
    // `stirling_log_gamma` adds 20 bits to the precision it is given; the
    // estimate is taken at the precision an ordinary argument ends at.
    let first = stirling_error_lg(x, bits.saturating_add(52))? - 20.0;
    // (An argument beyond `f64`, or beyond the cap: one pass as it is,
    // which finds the value beyond the exponent range.)
    let mut p = asked.guarded(Guard::for_log2(first), cap).unwrap_or(asked);
    for _ in 0..LOG_GAMMA_PASSES {
        let mut sub = w.at(p.bits());
        let l = stirling_log_gamma(x, &mut sub)?;
        let err = sub.take_own();
        // (`missing` is NaN or `+∞` for a sum without a bound.)
        let missing = err + bits as f64;
        if missing <= 0.0 {
            return Ok(l);
        }
        let lg = accuracy::part_lg_low(&l);
        if !l.is_negative() && lg >= 32.0 && err <= lg - 2.0 {
            return Ok(l);
        }
        match p.guarded(Guard::for_log2(missing + 2.0), cap) {
            Some(more) => p = more,
            None => break,
        }
    }
    Err(special_exhausted(bits))
}

/// Compute Γ(x) at arbitrary precision for real x, to within `½ + 2⁻¹²`
/// units in the last place of `prec` bits (see the module documentation;
/// up to 0.60 a large argument had `log₂ ln Γ(x) − 20` bits fewer).
///
/// For positive x, evaluates exp([`log_gamma_abs`]).
/// For negative non-integer x, applies the reflection formula
/// Γ(z) = π / (sin(πd) · (−1)ⁿ · Γ(1 − z)) with `z = n + d`, `n = round(z)`
/// and `d` exact, so that `sin` is accurate next to a pole (before 0.29 `πz`
/// was rounded first, losing `log₂(1/d)` bits, and arguments within
/// `10⁻¹²` of a pole — in `f64` — or `2^{−prec/2}` were refused as poles).
/// Both at [`GAMMA_GUARD`] bits beyond `prec` and rounded once: the six
/// rounded operations of the reflection (`π`, `πd`, `sin` — whose condition
/// `πd·cot πd` is at most 1 for `abs(d) ≤ ½` — `exp`, the product and the
/// quotient) and the error of the logarithm stay below `2⁻¹²` of a unit.
///
/// A value beyond the exponent range is `∞` (0 on the reflection side), as
/// `exp` returns it.
pub(super) fn arb_gamma_real(x: &BigFloat, w: &mut Wp<'_>) -> Result<BigFloat, SymplexError> {
    let prec = w.bits();
    // The poles: the non-positive integers, exactly.
    if x.is_zero() || (x.is_negative() && x.is_int()) {
        return Err(infinite("Gamma at non-positive integer pole"));
    }
    let gp = Prec::new(prec).plus(GAMMA_GUARD).bits();

    if !x.is_negative() {
        // Direct Stirling path.
        tracing::debug!(prec, "arb_gamma_real: positive argument");
        let log_gamma = log_gamma_abs(x, gp, w)?;
        Ok(round_to(log_gamma.exp(gp, RM, w.cc), prec))
    } else {
        tracing::debug!(prec, "arb_gamma_real: reflection formula");
        let exact = x.mantissa_max_bit_len().unwrap_or(prec).max(prec) + 64;
        let one_minus_x = BigFloat::from_i32(1, 64).sub(x, exact, RM);
        let log_gamma_1mx = log_gamma_abs(&one_minus_x, gp, w)?;
        let gamma_1mx = log_gamma_1mx.exp(gp, RM, w.cc);

        // sin(πz) = (−1)ⁿ sin(πd).
        let n = x.add(&BigFloat::from_f64(0.5, 64), exact, RM).floor();
        let d = x.sub(&n, exact, RM);
        let pi_val = w.cc.pi(gp, RM).clone();
        let mut sin_pi_x = pi_val.mul(&d, gp, RM).sin(gp, RM, w.cc);
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

        let denom = sin_pi_x.mul(&gamma_1mx, gp, RM);
        Ok(round_to(pi_val.div(&denom, gp, RM), prec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::numeric::bigint_to_bigfloat;

    /// `n/2^k`, exactly.
    fn dyadic(n: i64, k: i32) -> BigFloat {
        let mut x = BigFloat::from_i64(n, 128);
        if !x.is_zero() {
            x.set_exponent(x.exponent().unwrap_or(0) - k);
        }
        x
    }

    /// `n·2^(−k)` for a big integer `n` (an oracle value scaled by `2^k`).
    fn scaled(n: &str, k: i32) -> BigFloat {
        let mut x = bigint_to_bigfloat(&n.parse::<BigInt>().unwrap(), 512);
        x.set_exponent(x.exponent().unwrap_or(0) - k);
        x
    }

    /// The arguments of the tests below: small, next to the zeros of `ln Γ`,
    /// tiny, and large (`ln Γ` up to `2³⁰`).
    fn arguments() -> Vec<BigFloat> {
        vec![
            dyadic(1, 2),
            dyadic(3, 1),
            dyadic(1025, 10),
            dyadic(2001, 1),
            dyadic(100_001, 2),
            dyadic(21_196_207, 2),
            dyadic(280_000_001, 2),
            dyadic(1, 200),
        ]
    }

    /// What [`stirling_log_gamma`] charges to its `Wp` is a bound: the
    /// difference to the same sum formed with 400 more bits lies within it,
    /// at every argument and precision tried, and it is the size the
    /// module documentation derives (the terms' bits, `7 + log₂ n` more).
    /// Up to 0.60 nothing was charged, and `Γ(x)` was given two units in the
    /// last place of `exp` of a sum that had `log₂ ln Γ(x) − 20` bits fewer.
    ///
    /// Oracle: the sum itself at `prec + 400` bits (its error is `2⁴⁰⁰`
    /// times smaller); the anchors against mpmath are in the next test.
    #[test]
    fn the_stirling_sum_charges_a_bound_of_its_error() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        for x in arguments() {
            for prec in [64usize, 128, 300, 1000] {
                let mut w = Wp::new(prec, &mut cc);
                let l = stirling_log_gamma(&x, &mut w).unwrap();
                let err = w.take_own();
                let h = stirling_log_gamma(&x, &mut Wp::new(prec + 400, &mut cc)).unwrap();
                let d = l.sub(&h, prec + 440, RM);
                assert!(err.is_finite(), "x = {x}, prec = {prec}: no bound ({err})");
                assert!(
                    d.is_zero() || accuracy::part_lg(&d) <= err,
                    "x = {x}, prec = {prec}: error 2^{} above the charge 2^{err}",
                    accuracy::part_lg(&d)
                );
                // Not a vacuous bound: `abs(ln Γ) < 2³¹`, at most `2¹⁰` steps.
                assert!(err <= 30.0 - prec as f64, "x = {x}, prec = {prec}: {err}");
            }
        }
    }

    /// [`log_gamma_abs`] returns `ln Γ(x)` to the absolute error asked,
    /// `2⁻¹⁶⁰`, at a large argument as at a small one.  Up to 0.60 the
    /// logarithm behind `gamma(280000001/4)` at 128 bits was wrong from
    /// `2⁻¹¹⁸` on (`ln Γ ≈ 2³⁰` summed at 148 bits).
    ///
    /// Oracle: mpmath 1.3.0, `floor(loggamma(mpf(p)/q) * mpf(2)**160)` with
    /// `mp.prec = 600` and `900` (equal): `p/q = 280000001/4` →
    /// `1745735063872963066546970851977884184866920967383041842737`, `1/4` →
    /// `1882447028765323546834616426182295490785238579562`, `2001/2` →
    /// `8635536982457665541499611717864631364955790251019616`.
    #[test]
    fn the_logarithm_has_the_absolute_error_asked_for() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        let one = BigFloat::from_i32(1, 64);
        for (x, floor) in [
            (
                dyadic(280_000_001, 2),
                "1745735063872963066546970851977884184866920967383041842737",
            ),
            (
                dyadic(1, 2),
                "1882447028765323546834616426182295490785238579562",
            ),
            (
                dyadic(2001, 1),
                "8635536982457665541499611717864631364955790251019616",
            ),
        ] {
            for bits in [160usize, 400] {
                let l = log_gamma_abs(&x, bits, &mut Wp::new(64, &mut cc)).unwrap();
                // `0 ≤ ln Γ·2¹⁶⁰ − floor < 1`, and the value is within `2⁻¹⁶⁰`.
                let mut d = l.sub(&scaled(floor, 160), 700, RM);
                if !d.is_zero() {
                    d.set_exponent(d.exponent().unwrap_or(0) + 160);
                }
                assert!(
                    d > one.neg() && d < BigFloat::from_i32(2, 64),
                    "x = {x}, bits = {bits}: off by {d} units of 2^-160"
                );
            }
        }
    }

    /// [`arb_gamma_real`] is within `0.51` units in the last place of
    /// `Γ(x)`, on both sides of 0, at 64, 128 and 300 bits.  Up to 0.60
    /// `Γ(280000001/4)` at 128 bits was 300 units off (`2^(30.2 − 21)`
    /// times half a unit), and the reflection side formed its six rounded
    /// operations at the precision of the result.
    ///
    /// Oracle: mpmath 1.3.0 with `mp.prec = 600` and `900` (equal):
    /// `floor(gamma(mpf(1)/3) * mpf(2)**200)` =
    /// `4304888249653313123336247521829122275727275893629585420726299`;
    /// `g = gamma(mpf(-2001)/2)` is negative, `floor(log(abs(g), 2))` =
    /// `-8533`, `floor(abs(g) * mpf(2)**(199 + 8533))` =
    /// `968868825697366274427799283572260689675039840463669674949957`.
    /// The other arguments: the routine itself with 300 more bits.
    #[test]
    fn gamma_is_within_half_a_unit_and_a_hair() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        // `v` within 0.51 units in the last place (of `prec` bits) of `r`.
        let close = |v: &BigFloat, r: &BigFloat, prec: usize| {
            let d = v.sub(r, 900, RM);
            let ulp = f64::from(v.exponent().unwrap_or(0)) - prec as f64;
            d.is_zero() || accuracy::part_lg(&d) <= ulp + 0.51f64.log2()
        };
        let third = BigFloat::from_i32(1, 64).div(&BigFloat::from_i32(3, 64), 1200, RM);
        let g13 = scaled(
            "4304888249653313123336247521829122275727275893629585420726299",
            200,
        );
        let g_neg = scaled(
            "968868825697366274427799283572260689675039840463669674949957",
            199 + 8533,
        )
        .neg();
        for prec in [64usize, 128] {
            let v = arb_gamma_real(&third, &mut Wp::new(prec, &mut cc)).unwrap();
            assert!(close(&v, &g13, prec), "gamma(1/3) at {prec} bits: {v}");
            let v = arb_gamma_real(&dyadic(-2001, 1), &mut Wp::new(prec, &mut cc)).unwrap();
            assert!(v.is_negative());
            assert!(
                close(&v, &g_neg, prec),
                "gamma(-2001/2) at {prec} bits: {v}"
            );
        }
        let mut xs = arguments();
        xs.extend([
            dyadic(-1, 2),
            dyadic(-7, 1),
            dyadic(-100_001, 2),
            dyadic(-1, 200),
        ]);
        for x in xs {
            for prec in [64usize, 128, 300] {
                let v = arb_gamma_real(&x, &mut Wp::new(prec, &mut cc)).unwrap();
                let r = arb_gamma_real(&x, &mut Wp::new(prec + 300, &mut cc)).unwrap();
                assert!(
                    close(&v, &r, prec),
                    "gamma({x}) at {prec} bits: {v} for {r}"
                );
            }
        }
    }

    /// A value beyond the exponent range is `∞` (0 on the reflection side),
    /// as before: the callers fall back to their logarithmic forms on it.
    ///
    /// Oracle: `ln Γ(10⁹) ≈ 1.97·10¹⁰ > 2³¹·ln 2` (mpmath
    /// `loggamma(mpf(10)**9)` = `19723265837.7…`).
    #[test]
    fn gamma_beyond_the_exponent_range_is_infinite() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        let big = BigFloat::from_i64(1_000_000_000, 128);
        let v = arb_gamma_real(&big, &mut Wp::new(128, &mut cc)).unwrap();
        assert!(v.is_inf() && !v.is_negative());
        let mut huge = BigFloat::from_i32(1, 128);
        huge.set_exponent(2000);
        let v = arb_gamma_real(&huge, &mut Wp::new(128, &mut cc)).unwrap();
        assert!(v.is_inf() && !v.is_negative());
        let v = arb_gamma_real(&dyadic(-2_000_000_001, 1), &mut Wp::new(128, &mut cc)).unwrap();
        assert!(v.is_zero());
    }
}
