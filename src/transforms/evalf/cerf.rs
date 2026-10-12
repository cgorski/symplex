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
//! arguments with `abs(z)² > 10⁵` are refused.
//!
//! The node's bound is its own `2^−(prec+12)` relative plus twice the radius
//! times the derivative over the ball: `abs(erf′(u)) = (2/√π)·e^(−Re u²) ≤
//! (2/√π)·e^((abs(y) + r)² − max(abs(x) − r, 0)²)`, and `e^(Re u²)` for `erfi`.

use astro_float::{BigFloat, Consts, RoundingMode};

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

fn to_f64(x: &BigFloat) -> Option<f64> {
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven)
        .ok()
        .filter(|v| v.is_finite())
}

fn round_c(v: Complex, prec: usize, rm: RoundingMode) -> Complex {
    (
        super::round_to(v.0, prec, rm),
        super::round_to(v.1, prec, rm),
    )
}

fn add_term(sum: &mut (Complex, f64), t: &Complex, wp: usize, rm: RoundingMode) {
    sum.1 = sum.1.max(accuracy::lg_abs(t));
    sum.0 = c_add(&sum.0, t, wp, rm);
}

/// `erf z` at working precision `wp` (see the module documentation), with
/// `log₂` of the largest term (times the factor in front).
fn erf_at(z: &Complex, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Result<Sum, SymplexError> {
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
    let z2 = c_mul(z, z, p, rm);
    let mut sum: (Complex, f64) = (c_zero(p), f64::NEG_INFINITY);
    let mut t = z.clone(); // z^(2n+1)·coefficient
    for n in 0..max_terms {
        let nf = n as f64;
        let term = if kummer {
            t.clone()
        } else {
            let d = BigFloat::from_u64(2 * n as u64 + 1, 64);
            (t.0.div(&d, p, rm), t.1.div(&d, p, rm))
        };
        add_term(&mut sum, &term, p, rm);
        // The next term and the ratio bound beyond it.
        let (num, den) = if kummer {
            (c_mul(&t, &z2, p, rm), 2.0 * nf + 3.0)
        } else {
            (c_neg(&c_mul(&t, &z2, p, rm)), nf + 1.0)
        };
        let scale = if kummer { 2.0 } else { 1.0 };
        let d = BigFloat::from_f64(den, 64);
        let s = BigFloat::from_f64(scale, 64);
        t = (
            num.0.mul(&s, p, rm).div(&d, p, rm),
            num.1.mul(&s, p, rm).div(&d, p, rm),
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
            if lt < sum.1 - wp as f64 - 8.0 || (t.0.is_zero() && t.1.is_zero()) {
                break;
            }
        }
        if n + 1 == max_terms {
            return Err(super::series_did_not_converge("erf", max_terms));
        }
    }
    let two_over_sqrt_pi = BigFloat::from_i32(2, 64).div(&cc.pi(p, rm).clone().sqrt(p, rm), p, rm);
    let mut value = (
        sum.0.0.mul(&two_over_sqrt_pi, p, rm),
        sum.0.1.mul(&two_over_sqrt_pi, p, rm),
    );
    let mut largest = sum.1 + LOG2_TWO_OVER_SQRT_PI;
    if kummer {
        // e^(−z²): its exponent's absolute error is its relative one.
        let pe = p + super::magnitude_bits(a2) + 8;
        let e = super::c_exp(&c_neg(&c_mul(z, z, pe, rm)), pe, rm, cc);
        largest += accuracy::lg_abs(&e);
        value = c_mul(&value, &e, p, rm);
    }
    Ok(Sum {
        value: round_c(value, wp, rm),
        largest,
    })
}

/// `erf`, `erfc` or `erfi` of a non-real `z`, rounded to `prec + 16` bits,
/// its error below `2^−(prec+12)` of its modulus.
pub(super) fn erf_family(
    kind: Kind,
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("erf of special float value"));
    }
    tracing::debug!(prec, ?kind, "evalf: erf family at a complex argument");
    let arg = match kind {
        Kind::Erfi => (z.1.neg(), z.0.clone()), // iz
        _ => z.clone(),
    };
    let cap = super::cancellation_cap(prec);
    let mut extra = 32 + accuracy::ceil_log2(prec) as usize;
    loop {
        let wp = prec + extra;
        let s = erf_at(&arg, wp, rm, cc).map_err(super::requested_at(prec))?;
        let (value, largest) = match kind {
            Kind::Erf => (s.value, s.largest),
            Kind::Erfc => (c_sub(&c_one(wp), &s.value, wp, rm), s.largest.max(0.0)),
            // −i·erf(iz) = (Im, −Re).
            Kind::Erfi => ((s.value.1.clone(), s.value.0.neg()), s.largest),
        };
        let lv = accuracy::lg_abs(&value);
        let lost = if lv.is_finite() && largest.is_finite() {
            ((largest - lv).max(0.0).ceil() as usize).min(cap + 1)
        } else {
            wp
        };
        if lost + 24 <= extra {
            return Ok(round_c(value, prec + 16, rm));
        }
        if extra >= cap || lost + 24 > cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
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
    let own = lv - prec as f64 - 12.0;
    if bz.is_exact() {
        return Bound::both(own);
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
    // times the radius (twice, for the first-order estimate).
    Bound::both(accuracy::lsum(own, ld + bz.joint() + 1.0))
}
