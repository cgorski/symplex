//! `erf`, `erfc` and `erfi` at non-real arguments, and the error bounds of
//! those nodes.  (The real routines are `arb_erf`, `arb_erfc` and `arb_erfi`
//! in `evalf.rs`.)  Before 0.44 a non-real argument was refused: `erf(1 +
//! i)`, "erf of complex argument not yet supported in evalf".
//!
//! `erf` is entire, and summed from one of its two power series (DLMF 7.6.1,
//! 7.6.2):
//!
//! ```text
//! erf z = (2/√π)·Σ_{n≥0} (−1)^n z^(2n+1)/(n!(2n+1))
//!       = (2/√π)·e^(−z²)·Σ_{n≥0} 2^n z^(2n+1)/(1·3⋯(2n+1)),
//! ```
//!
//! the second for `abs(Re z) ≥ abs(Im z)` (its terms cancel by about `2y²` bits
//! against the value, the first's by `abs(z)²`).  Both are stopped by a
//! geometric bound on the rest: the ratio of consecutive terms is at most
//! `abs(z)²/(n + 2)`, resp. `2abs(z)²/(2n + 5)`, below 1 from some `n` on, and
//! the rest is at most the next term over `1 − q`.  `erfc = 1 − erf` and `erfi
//! z = −i·erf(iz)`.  The loss to cancellation (the largest term over the
//! value; for `erfc` with `Re z` large, the bits of `erf` against 1) is
//! measured and made up as in `polylog.rs`, up to `cancellation_cap`;
//! arguments with `abs(z)² > 10⁵` are refused by the series.
//!
//! Where it converges to the working precision, the asymptotic expansion
//! `erfc z = Γ(1/2, z²)/√π ~ e^(−z²)/(z√π)·Σ (−1)^m (1/2)_m z^(−2m)` (DLMF
//! 7.11.2, 7.12.1) for `Re z > 0` takes over (`erf` is odd, `erfc(−z) = 2 −
//! erfc z`), stopped by the remainder bound of `cexpint.rs` (the first
//! neglected term for `abs(ph z) ≤ π/4`, as DLMF 7.12(i), and a proven
//! factor beyond): before 0.45 `erf(30 + 31i)` exhausted the precision.
//! The Fresnel integrals of a non-real argument come from `erf` (DLMF
//! 7.5.9).
//!
//! The node's bound is its own `2^−(prec+12)` relative plus twice the radius
//! times the derivative over the ball: `abs(erf′(u)) = (2/√π)·e^(−Re u²) ≤
//! (2/√π)·e^((abs(y) + r)² − max(abs(x) − r, 0)²)`, and `e^(Re u²)` for `erfi`.
//!
//! On the imaginary axis (a real part exactly 0) the parts are real
//! functions: `erf(iy) = i·erfi(y)`, `erfc(iy) = 1 − i·erfi(y)`, `erfi(iy) =
//! i·erf(y)` (from `erfi z = −i·erf(iz)`), the imaginary part from the real routine
//! with its own relative bound, the real part exact.  Before 0.54 the value
//! carried one bound, `abs(v)·2^−(prec+12)`, for both parts, and the `1` of
//! `erfc(10i) = 1 − 1.52·10⁴²i` lay inside it: it printed
//! `-1.524307423e42*i` at every digit count, as did `uppergamma(1/2, −100)`
//! (`√π·erfc(10i)`, real part `√π`) and every half-integer `Γ(a, −x)`.
//! An argument whose real part is exactly 0 moves only the imaginary part.

use super::wp::{RM, Wp};
use crate::base::bigfloat::{self, is_strictly_positive, to_f64};
use astro_float::BigFloat;

use super::accuracy::{self, Bound};
use super::zeta::Sum;
use crate::base::bigcomplex::{Complex, c_add, c_mul, c_neg, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;

const LOG2E: f64 = std::f64::consts::LOG2_E;
/// `log₂(2/√π)`.
const LOG2_TWO_OVER_SQRT_PI: f64 = 0.176_656_383_560_998_7;
/// Largest `abs(z)²` evaluated.
const MAX_ABS2: f64 = 1.0e5;

/// Which function of `erf`'s family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Erf,
    Erfc,
    Erfi,
}

fn round_c(v: Complex, prec: usize) -> Complex {
    (super::round_to(v.0, prec), super::round_to(v.1, prec))
}

fn add_term(sum: &mut (Complex, f64), t: &Complex, wp: usize) {
    sum.1 = sum.1.max(accuracy::lg_abs(t));
    sum.0 = c_add(&sum.0, t, wp, RM);
}

/// `erf z` at working precision `wp` (see the module documentation), with
/// `log₂` of the largest term (times the factor in front).
pub(super) fn erf_at(z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let (Some(x), Some(y)) = (to_f64(&z.0), to_f64(&z.1)) else {
        return Err(super::unevaluable("erf argument beyond f64"));
    };
    let a2 = x * x + y * y;
    if a2.is_nan() || a2 > MAX_ABS2 {
        return Err(super::unevaluable(
            "erf of a complex argument this large not supported in evalf",
        ));
    }
    let kummer = x.abs() >= y.abs();
    let max_terms = (8.0 * a2) as usize + 4 * wp + 64;
    // Guard bits for the roundings of the terms and of z^(2n+1).
    let p = wp + accuracy::ceil_log2(max_terms) as usize + 8;
    let z2 = c_mul(z, z, p, RM);
    let mut sum: (Complex, f64) = (c_zero(p), f64::NEG_INFINITY);
    let mut t = z.clone(); // z^(2n+1)·coefficient
    for n in 0..max_terms {
        let nf = n as f64;
        let term = if kummer {
            t.clone()
        } else {
            let d = BigFloat::from_u64(2 * n as u64 + 1, 64);
            (t.0.div(&d, p, RM), t.1.div(&d, p, RM))
        };
        add_term(&mut sum, &term, p);
        // The next term and the ratio bound beyond it.
        let (num, den) = if kummer {
            (c_mul(&t, &z2, p, RM), 2.0 * nf + 3.0)
        } else {
            (c_neg(&c_mul(&t, &z2, p, RM)), nf + 1.0)
        };
        let scale = if kummer { 2.0 } else { 1.0 };
        let d = bigfloat::from_f64(den, 64);
        let s = bigfloat::from_f64(scale, 64);
        t = (
            num.0.mul(&s, p, RM).div(&d, p, RM),
            num.1.mul(&s, p, RM).div(&d, p, RM),
        );
        let q = if kummer {
            2.0 * a2 / (2.0 * nf + 5.0)
        } else {
            a2 / (nf + 2.0)
        };
        if q < 0.5 {
            // |next term| ≤ |t| (Taylor: over 2n + 3), the rest ≤ twice that.
            let lt = accuracy::lg_abs(&t) + 1.0 + 1e-9;
            let lt = if kummer {
                lt
            } else {
                lt - (2.0 * nf + 3.0).log2()
            };
            // (`lt` bounds the rest: the stop is certified.)
            if lt < sum.1 - wp as f64 - 8.0 || (t.0.is_zero() && t.1.is_zero()) {
                break;
            }
        }
        if n + 1 == max_terms {
            return Err(super::series_did_not_converge("erf", max_terms));
        }
    }
    let two_over_sqrt_pi =
        BigFloat::from_i32(2, 64).div(&w.cc.pi(p, RM).clone().sqrt(p, RM), p, RM);
    let mut value = (
        sum.0.0.mul(&two_over_sqrt_pi, p, RM),
        sum.0.1.mul(&two_over_sqrt_pi, p, RM),
    );
    let mut largest = sum.1 + LOG2_TWO_OVER_SQRT_PI;
    if kummer {
        // e^(−z²): its exponent's absolute error is its relative one.
        let pe = p + super::magnitude_bits(a2) + 8;
        let e = super::c_exp(&c_neg(&c_mul(z, z, pe, RM)), &mut w.at(pe));
        largest += accuracy::lg_abs(&e);
        value = c_mul(&value, &e, p, RM);
    }
    Ok(Sum {
        value: round_c(value, wp),
        largest,
    })
}

/// `erfc s` for `Re s > 0` (or `s` on the positive imaginary axis) by the
/// asymptotic expansion of `Γ(1/2, s²)/√π` (DLMF 7.11.2, 7.12.1) with the
/// remainder bound of `cexpint.rs`, at working precision `wp`, or `None`
/// where it does not reach `2^−wp`.
fn erfc_asymptotic(s: &Complex, wpc: &mut Wp<'_>) -> Option<Sum> {
    let wp = wpc.bits();
    let (x, y) = (to_f64(&s.0)?, to_f64(&s.1)?);
    let a2 = x * x + y * y;
    // Γ(1/2, w) moves by about abs(Δw) relative: abs(s)² more bits for s².
    let p = wp + super::magnitude_bits(a2) + 8;
    let w = c_mul(s, s, p, RM);
    let half = (BigFloat::from_f64(0.5, 64), BigFloat::new(64));
    let g = super::cexpint::upper_asymptotic(&half, &w, &mut wpc.at(wp + 2))?;
    let sqrt_pi = wpc.cc.pi(p, RM).clone().sqrt(p, RM);
    Some(Sum {
        value: (
            g.value.0.div(&sqrt_pi, p, RM),
            g.value.1.div(&sqrt_pi, p, RM),
        ),
        largest: g.largest - 0.82,
    })
}

/// `erf z` by [`erfc_asymptotic`] at `±z` (`erf` is odd), or `None`.
pub(super) fn erf_asymptotic(z: &Complex, w: &mut Wp<'_>) -> Option<Sum> {
    let wp = w.bits();
    let right = is_strictly_positive(&z.0) || (z.0.is_zero() && !z.1.is_negative());
    let s = if right { z.clone() } else { c_neg(z) };
    let e = erfc_asymptotic(&s, w)?;
    let v = c_sub(&c_one(wp + 8), &e.value, wp + 8, RM);
    Some(Sum {
        value: if right { v } else { c_neg(&v) },
        largest: e.largest.max(0.0),
    })
}

/// `erf`, `erfc` or `erfi` of `z` at working precision `wp`: the asymptotic
/// expansion where it converges (refused or slow before 0.45: `erf(30 +
/// 31i)` exhausted the precision), the power series otherwise.
fn family_at(kind: Kind, arg: &Complex, w: &mut Wp<'_>) -> Result<(Complex, f64), SymplexError> {
    let wp = w.bits();
    if kind == Kind::Erfc {
        // erfc z directly for Re z > 0 and on the positive imaginary axis,
        // 2 − erfc(−z) otherwise, as `erf_asymptotic` splits it: the
        // expansion is of `Γ(1/2, s²)` with `s²` on the principal branch,
        // which on the imaginary axis (`s² < 0`) is `2·arg s` only for `Im s
        // > 0`.  Up to 0.46 `erfc(14i)` came out `+5.35·10⁸³i` (mpmath: `1 −
        // 5.347…·10⁸³i`), and so did `uppergamma(−7/2, −401/2)`, which `eval`
        // writes with `erfc(i·√802/2)`: `2.57·10⁸⁵i` (mpmath: `0.27 −
        // 5.33·10⁷⁶i`).
        let right = is_strictly_positive(&arg.0) || (arg.0.is_zero() && !arg.1.is_negative());
        let s = if right { arg.clone() } else { c_neg(arg) };
        if let Some(e) = erfc_asymptotic(&s, w) {
            if right {
                return Ok((e.value, e.largest));
            }
            let two = (BigFloat::from_i32(2, 64), BigFloat::new(64));
            return Ok((c_sub(&two, &e.value, wp + 8, RM), e.largest.max(1.0)));
        }
    } else if let Some(e) = erf_asymptotic(arg, w) {
        return Ok(match kind {
            // −i·erf(iz) = (Im, −Re).
            Kind::Erfi => ((e.value.1.clone(), e.value.0.neg()), e.largest),
            _ => (e.value, e.largest),
        });
    }
    let s = erf_at(arg, w)?;
    Ok(match kind {
        Kind::Erf => (s.value, s.largest),
        Kind::Erfc => (c_sub(&c_one(wp), &s.value, wp, RM), s.largest.max(0.0)),
        Kind::Erfi => ((s.value.1.clone(), s.value.0.neg()), s.largest),
    })
}

/// `erf`, `erfc` or `erfi` of a non-real `z`, rounded to `prec + 16` bits,
/// its error below `2^−(prec+12)` of its modulus.
pub(super) fn erf_family(kind: Kind, z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("erf of special float value"));
    }
    tracing::debug!(prec, ?kind, "evalf: erf family at a complex argument");
    if z.0.is_zero() {
        return on_axis(kind, &z.1, w);
    }
    let arg = match kind {
        Kind::Erfi => (z.1.neg(), z.0.clone()), // iz
        _ => z.clone(),
    };
    let cap = super::cancellation_cap(prec);
    let mut extra = 32 + accuracy::ceil_log2(prec) as usize;
    loop {
        let wp = prec + extra;
        let (value, largest) =
            family_at(kind, &arg, &mut w.at(wp)).map_err(super::requested_at(prec))?;
        let lv = accuracy::lg_abs(&value);
        let lost = if lv.is_finite() && largest.is_finite() {
            ((largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else {
            wp
        };
        if lost + 24 <= extra {
            return Ok(round_c(value, prec + 16));
        }
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// `erf`, `erfc` or `erfi` of `iy` (module documentation): the real part
/// exact, the imaginary part rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of itself.
fn on_axis(kind: Kind, y: &BigFloat, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let p = w.bits() + 16;
    let (re, im) = match kind {
        Kind::Erf => (BigFloat::new(p), super::arb_erfi(y, &mut w.at(p))?),
        Kind::Erfc => (
            BigFloat::from_i32(1, p),
            super::arb_erfi(y, &mut w.at(p))?.neg(),
        ),
        Kind::Erfi => (BigFloat::new(p), super::arb_erf(y, &mut w.at(p))?),
    };
    if im.is_zero() || im.is_nan() || im.is_inf() {
        return Err(super::unevaluable(
            "erf of an imaginary argument: the value overflows the arbitrary-precision exponent range",
        ));
    }
    Ok((re, super::round_to(im, p)))
}

/// The Fresnel integral `S(z)` (`want_s`) or `C(z)` of a non-real `z`, from
/// `C(z) ± iS(z) = ((1 ± i)/2)·erf(ζ±)`, `ζ± = (√π/2)(1 ∓ i)z` (DLMF 7.5.9),
/// the loss to cancellation made up as for `erf`; rounded to `prec + 16`
/// bits, its error below `2^−(prec+12)` of its modulus.  (Refused before
/// 0.45: "fresnels of complex argument not yet supported in evalf".)
pub(super) fn fresnel(want_s: bool, z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable(
            "Fresnel integral of special float value",
        ));
    }
    tracing::debug!(
        prec,
        want_s,
        "evalf: Fresnel integral at a complex argument"
    );
    super::zeta::with_loss(w, &mut |wp, cc| {
        let (Some(x), Some(y)) = (to_f64(&z.0), to_f64(&z.1)) else {
            return Err(super::unevaluable("Fresnel argument beyond f64"));
        };
        // erf′(ζ)·Δζ is up to abs(ζ)² times erf's own scale: as many guard
        // bits for the rounding of ζ.
        let a2 = (x * x + y * y) * std::f64::consts::PI / 2.0;
        let p = wp + super::magnitude_bits(a2) + 8;
        let c = cc
            .pi(p, RM)
            .clone()
            .sqrt(p, RM)
            .div(&BigFloat::from_i32(2, 64), p, RM);
        // (1 − i)z = (x + y) + i(y − x), (1 + i)z = (x − y) + i(x + y).
        let s = z.0.add(&z.1, p, RM);
        let d = z.1.sub(&z.0, p, RM);
        let zeta1 = (s.mul(&c, p, RM), d.mul(&c, p, RM));
        let zeta2 = (d.neg().mul(&c, p, RM), s.mul(&c, p, RM));
        let e1 = match erf_asymptotic(&zeta1, &mut Wp::new(wp, cc)) {
            Some(e) => e,
            None => erf_at(&zeta1, &mut Wp::new(wp, cc))?,
        };
        let e2 = match erf_asymptotic(&zeta2, &mut Wp::new(wp, cc)) {
            Some(e) => e,
            None => erf_at(&zeta2, &mut Wp::new(wp, cc))?,
        };
        // A = C + iS = ((1 + i)/2)·erf(ζ₁), B = C − iS = ((1 − i)/2)·erf(ζ₂).
        let half = BigFloat::from_f64(0.5, 64);
        let one_i = (half.clone(), half.clone());
        let one_mi = (half.clone(), half.neg());
        let a = c_mul(&one_i, &e1.value, wp, RM);
        let b = c_mul(&one_mi, &e2.value, wp, RM);
        let v = if want_s {
            // S = (A − B)/(2i) = −i(A − B)/2.
            let d = c_sub(&a, &b, wp, RM);
            (d.1.mul(&half, wp, RM), d.0.neg().mul(&half, wp, RM))
        } else {
            let s = c_add(&a, &b, wp, RM);
            (s.0.mul(&half, wp, RM), s.1.mul(&half, wp, RM))
        };
        Ok(Sum {
            value: v,
            largest: e1.largest.max(e2.largest),
        })
    })
}

/// The error bound of `erf`, `erfc` or `erfi` (value `v`) at a non-real `z ±
/// bz` (see the module documentation).
pub(super) fn erf_bound(kind: Kind, z: &Complex, bz: Bound, v: &Complex, prec: usize) -> Bound {
    if bz.is_unknown() {
        return Bound::UNKNOWN;
    }
    let lv = accuracy::lg_abs(v);
    if !lv.is_finite() {
        return Bound::UNKNOWN;
    }
    // On the imaginary axis ([`on_axis`]) each part has its own bound.
    let axis = z.0.is_zero();
    let own = if axis {
        Bound {
            re: accuracy::EXACT,
            im: accuracy::part_lg(&v.1) - prec as f64 - 12.0,
        }
    } else {
        Bound::both(lv - prec as f64 - 12.0)
    };
    if bz.is_exact() {
        return own;
    }
    let (Some(x), Some(y)) = (to_f64(&z.0), to_f64(&z.1)) else {
        return Bound::UNKNOWN;
    };
    let r = bz.joint().exp2();
    if !r.is_finite() || r > 0.0625 {
        return Bound::UNKNOWN;
    }
    // erfi(u) = −i·erf(iu): swap the parts.
    let (a, b) = match kind {
        Kind::Erfi => (y.abs(), x.abs()),
        _ => (x.abs(), y.abs()),
    };
    let lo = (a - r).max(0.0);
    let hi = b + r;
    let ld = LOG2_TWO_OVER_SQRT_PI + (hi * hi - lo * lo) * LOG2E;
    if !ld.is_finite() {
        return Bound::UNKNOWN;
    }
    // The derivative over the ball moves the value by at most its bound
    // times the radius (twice, for the first-order estimate); along the
    // imaginary axis only the imaginary part.
    let moved = ld + bz.joint() + 1.0;
    Bound {
        re: if axis && accuracy::exact_zero(&z.0, bz.re) {
            own.re
        } else {
            accuracy::lsum(own.re, moved)
        },
        im: accuracy::lsum(own.im, moved),
    }
}
