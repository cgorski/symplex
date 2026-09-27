//! `loggamma` at a non-real argument: the analytic continuation of `ln Γ`
//! from the positive reals to `ℂ ∖ (−∞, 0]` (SymPy's and mpmath's
//! `loggamma`, Mathematica's `LogGamma`), and the error bound of the node.
//!
//! `loggamma(z̄) = conj(loggamma(z))`, so an argument below the real axis is
//! reflected to one above it.  For `Im z > 0`:
//!
//! * `Re z ≥ ½`: Stirling's series at `w = z + n`, `|w| ≥ wp/2`, stopped by
//!   the remainder bound of DLMF 5.11(ii) — the first neglected term times
//!   `sec^{2K}(½ ph w)`, at most `2^K` for `Re w > 0` — and
//!   `loggamma(z) = loggamma(w) − Σ_{k<n} ln(z + k)`: `loggamma(u + 1) =
//!   loggamma(u) + ln u` holds off the cut, so the sum of *principal*
//!   logarithms is the right branch.  It is taken as one logarithm of the
//!   product, moved by the multiple of `2πi` that brings its imaginary part
//!   to `Σ arg(z + k)` summed in `f64` (off by far less than the `π` that
//!   decides the multiple).
//! * `Re z < ½`: the reflection `loggamma(z) = ln π − S(z) − loggamma(1 − z)`,
//!   where `S(z) = −iπz + iπ/2 − ln 2 + ln(1 − e^{2πiz})` is a logarithm of
//!   `sin πz` holomorphic in the upper half-plane (`|e^{2πiz}| < 1` there, so
//!   the principal logarithm of `1 − e^{2πiz}` is continuous).  On
//!   `(0, 1) + i0` it is the real `ln sin πx`, which fixes the constant; on
//!   the negative axis its limit gives `ln|Γ(x)| − iπ⌈−x⌉`, the value on the
//!   cut (`arb_log_gamma`).  `1 − e^{2πiz}` is formed as
//!   `−2i·sin(πz′)·e^{iπz′}` with `z′ = z − round(Re z)` exact: its real part
//!   `2e^{−πy}(sin²(πx′)cosh(πy) + cos²(πx′)sinh(πy))` is a sum of positive
//!   terms, accurate next to a pole of `Γ`.
//!
//! The terms can be much larger than the value (next to the zeros at 1 and
//! 2): the loss is measured and the evaluation repeated with that many more
//! bits, up to `cancellation_cap`, as for a real argument.  The overall
//! structure — the reflection left of `Re z = ½`, Stirling's series after a
//! shift, the branch of the shifted product fixed by a low-precision
//! estimate of its imaginary part — follows mpmath's `mpc_loggamma` /
//! `mpc_gamma` (`mpmath/libmp/gammazeta.py`, BSD); the reflection is written
//! with `S(z)` instead of mpmath's floor terms, and the stopping rule is the
//! DLMF bound.

use astro_float::{BigFloat, Consts, RoundingMode};

use super::accuracy::{self, Bound};
use crate::base::bigcomplex::{Complex, c_abs, c_add, c_div, c_mul, c_sub};
use crate::base::errors::SymplexError;

/// `loggamma(z)` for `Im z ≠ 0`, rounded to `prec` bits: its absolute error
/// on each part is below `2^−(prec+16)·|loggamma(z)|`.
pub(super) fn loggamma_complex(
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if z.0.is_nan() || z.1.is_nan() || z.0.is_inf() || z.1.is_inf() {
        return Err(super::unevaluable("LogGamma of special float value"));
    }
    let below = z.1.is_negative();
    let upper = if below {
        (z.0.clone(), z.1.neg())
    } else {
        z.clone()
    };
    let cap = super::cancellation_cap(prec);
    let mut extra = 40 + accuracy::ceil_log2(prec) as usize;
    loop {
        let wp = prec + extra;
        let (value, largest) = upper_half(&upper, wp, rm, cc)?;
        let lv = accuracy::lg_abs(&value);
        let lost = if lv.is_finite() {
            (largest - lv).max(0.0).ceil() as usize
        } else {
            wp
        };
        if lost + 24 <= extra {
            let v = (
                super::round_to(value.0, prec + 16, rm),
                super::round_to(value.1, prec + 16, rm),
            );
            return Ok(if below { (v.0, v.1.neg()) } else { v });
        }
        if extra >= cap {
            return Err(super::special_exhausted(prec));
        }
        extra = (lost + 40).max(2 * extra).min(cap);
    }
}

/// `loggamma(z)` for `Im z > 0` at working precision `wp`, and `log₂` of the
/// largest term summed (the absolute error is a few units of `2^−wp` of it).
fn upper_half(
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(Complex, f64), SymplexError> {
    let half = BigFloat::from_f64(0.5, 64);
    if !z.0.sub(&half, wp, rm).is_negative() {
        return shifted_stirling(z, wp, rm, cc);
    }
    // Reflection: ln π − S(z) − conj(loggamma(1 − z̄)).
    let one = BigFloat::from_i32(1, 64);
    let mirror = (one.sub(&z.0, super::exact_bits(&z.0, wp), rm), z.1.clone());
    let (g, lg) = shifted_stirling(&mirror, wp, rm, cc)?;
    let s = log_sin_pi(z, wp, rm, cc);
    let ln_pi = cc.pi(wp, rm).clone().ln(wp, rm, cc);
    let re = ln_pi.sub(&s.0, wp, rm).sub(&g.0, wp, rm);
    let im = s.1.neg().add(&g.1, wp, rm);
    let largest = lg.max(accuracy::lg_abs(&s)).max(1.0);
    Ok(((re, im), largest))
}

/// `S(z)` for `Im z > 0`: the logarithm of `sin πz` continuous in the upper
/// half-plane that is real on `(0, 1)` (see the module documentation).
fn log_sin_pi(z: &Complex, wp: usize, rm: RoundingMode, cc: &mut Consts) -> Complex {
    let pi = cc.pi(wp, rm).clone();
    let (x, y) = (&z.0, &z.1);
    let n = x
        .add(&BigFloat::from_f64(0.5, 64), super::exact_bits(x, wp), rm)
        .floor();
    let d = x.sub(&n, super::exact_bits(x, wp), rm);
    let quarter = BigFloat::from_f64(0.25, 64);
    // q = 1 − e^{2πiz′}, z′ = d + iy.
    let q: Complex = if !y.sub(&quarter, wp, rm).is_negative() {
        // |e^{2πiz′}| ≤ e^{−π/2}: no cancellation in 1 − e^{2πiz′}.
        let two_pi = pi.mul(&BigFloat::from_i32(2, 64), wp, rm);
        let w = super::c_exp(
            &(two_pi.mul(y, wp, rm).neg(), two_pi.mul(&d, wp, rm)),
            wp,
            rm,
            cc,
        );
        (BigFloat::from_i32(1, 64).sub(&w.0, wp, rm), w.1.neg())
    } else {
        // −2i·sin(πz′)·e^{iπz′}.
        let theta = (pi.mul(&d, wp, rm), pi.mul(y, wp, rm));
        let s = super::c_sin(&theta, wp, rm, cc);
        let e = super::c_exp(&(theta.1.neg(), theta.0.clone()), wp, rm, cc);
        let p = c_mul(&s, &e, wp, rm);
        let two = BigFloat::from_i32(2, 64);
        (p.1.mul(&two, wp, rm), p.0.mul(&two, wp, rm).neg())
    };
    let lq = super::c_ln(&q, wp, rm, cc);
    let ln2 = BigFloat::from_i32(2, 64).ln(wp, rm, cc);
    let half_pi = pi.div(&BigFloat::from_i32(2, 64), wp, rm);
    let re = pi.mul(y, wp, rm).sub(&ln2, wp, rm).add(&lq.0, wp, rm);
    let im = pi
        .mul(x, wp, rm)
        .neg()
        .add(&half_pi, wp, rm)
        .add(&lq.1, wp, rm);
    (re, im)
}

/// `loggamma(z)` for `Im z > 0`, `Re z > 0` at working precision `wp` by
/// Stirling's series after a shift (see the module documentation), and
/// `log₂` of the largest term.
fn shifted_stirling(
    z: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(Complex, f64), SymplexError> {
    let x = super::bigfloat_to_f64(&z.0, rm, cc)?;
    let y = super::bigfloat_to_f64(&z.1, rm, cc)?;
    let target = (0.5 * wp as f64).max(12.0);
    // The shift n: |z + n| ≥ target.
    let n: u64 = if x.hypot(y) >= target || y >= target {
        0
    } else {
        ((target * target - y * y).sqrt() - x).ceil().max(0.0) as u64
    };
    let exact = super::exact_bits(&z.0, wp);
    let w = (z.0.add(&BigFloat::from_u64(n, 64), exact, rm), z.1.clone());
    let (g, lg) = stirling_series(&w, wp, rm, cc)?;
    if n == 0 {
        return Ok((g, lg));
    }
    // Σ_{k<n} ln(z + k) = ln Π(z + k) + 2πim: the product with log₂ n guard
    // bits for the roundings of its n factors.
    let wp2 = wp + accuracy::ceil_log2(n as usize) as usize + 8;
    let mut p: Complex = (z.0.clone(), z.1.clone());
    for k in 1..n {
        let f = (z.0.add(&BigFloat::from_u64(k, 64), exact, rm), z.1.clone());
        p = c_mul(&p, &f, wp2, rm);
    }
    let mut l = super::c_ln(&p, wp2, rm, cc);
    let args: f64 = (0..n).map(|k| y.atan2(x + k as f64)).sum();
    let im0 = super::bigfloat_to_f64(&l.1, rm, cc)?;
    let two_pi_f = 2.0 * std::f64::consts::PI;
    let m = ((args - im0) / two_pi_f).round();
    if m != 0.0 {
        let two_pi = cc
            .pi(wp2, rm)
            .clone()
            .mul(&BigFloat::from_i32(2, 64), wp2, rm);
        let shift = two_pi.mul(&BigFloat::from_f64(m, 64), wp2, rm);
        l.1 = l.1.add(&shift, wp2, rm);
    }
    let v = c_sub(&g, &l, wp, rm);
    Ok((v, lg.max(accuracy::lg_abs(&l))))
}

/// Stirling's series `(w − ½)ln w − w + ½ln 2π + Σ_{k≥1} B₂ₖ/(2k(2k−1)w^{2k−1})`
/// for `Re w > 0`, `|w| ≥ 12`, to an absolute error below `2^−wp` of the
/// leading part; `log₂` of the largest term.
fn stirling_series(
    w: &Complex,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<(Complex, f64), SymplexError> {
    let half = BigFloat::from_f64(0.5, 64);
    let ln_w = super::c_ln(w, wp, rm, cc);
    let w_half = (w.0.sub(&half, wp, rm), w.1.clone());
    let lead = c_sub(&c_mul(&w_half, &ln_w, wp, rm), w, wp, rm);
    let two_pi = cc
        .pi(wp, rm)
        .clone()
        .mul(&BigFloat::from_i32(2, 64), wp, rm);
    let half_ln_2pi = two_pi.ln(wp, rm, cc).mul(&half, wp, rm);
    let mut v = (lead.0.add(&half_ln_2pi, wp, rm), lead.1);
    let largest = accuracy::lg_abs(&v).max(accuracy::lg_abs(w));
    // |w| and sec²(½ ph w) = 2|w|/(|w| + Re w) ≤ 2, in f64 (logarithms).
    let lw = accuracy::lg_abs_low(w);
    let abs_w = super::bigfloat_to_f64(&c_abs(w, wp, rm), rm, cc)?;
    let re_w = super::bigfloat_to_f64(&w.0, rm, cc)?;
    let lsec2 = if abs_w.is_finite() && abs_w > 0.0 {
        (2.0 * abs_w / (abs_w + re_w.max(0.0)))
            .log2()
            .clamp(0.0, 1.0)
    } else {
        1.0
    };
    let goal = accuracy::lg_abs_low(&v) - wp as f64 - 4.0;
    let one = (BigFloat::from_i32(1, 64), BigFloat::new(64));
    let u = c_div(&one, w, wp, rm);
    let u2 = c_mul(&u, &u, wp, rm);
    let mut pw = u; // w^{−(2k−1)}
    let max_terms = (wp as f64 * 0.4) as usize + 16;
    for k in 1..=max_terms {
        // Bound on the remainder after the terms below k: the first
        // neglected term (k) times sec^{2k}(½ ph w).
        let lb =
            log2_abs_bernoulli(k) - ((2 * k * (2 * k - 1)) as f64).log2() - (2 * k - 1) as f64 * lw
                + k as f64 * lsec2;
        if lb < goal {
            return Ok((v, largest));
        }
        let b = super::ratio_to_bigfloat(&super::bernoulli::even(k), wp, rm);
        let c = b.div(
            &BigFloat::from_u64((2 * k * (2 * k - 1)) as u64, 64),
            wp,
            rm,
        );
        let term = (pw.0.mul(&c, wp, rm), pw.1.mul(&c, wp, rm));
        v = c_add(&v, &term, wp, rm);
        pw = c_mul(&pw, &u2, wp, rm);
    }
    Err(super::series_did_not_converge(
        "loggamma Stirling series",
        max_terms,
    ))
}

/// An upper bound on `log₂|B₂ₖ|` from the bit lengths of its numerator
/// and denominator.
fn log2_abs_bernoulli(k: usize) -> f64 {
    let b = super::bernoulli::even(k);
    b.numer().bits() as f64 - b.denom().bits() as f64 + 1.0
}

/// The error bound of `loggamma(z)` (value `v`) for a non-real `z ± bz`:
/// the routine's own error (see [`loggamma_complex`]) plus twice a bound on
/// `|ψ|` over the argument's ball times its radius.  A ball whose real part
/// reaches the cut `(−∞, 0]` while its imaginary part may be 0 has no bound
/// (the side of the cut is undecidable), nor one that is not well inside
/// the domain of analyticity (radius above an eighth of the distance to a
/// pole of `Γ`).
pub(super) fn error_bound(z: &Complex, bz: Bound, v: &Complex, prec: usize) -> Bound {
    if bz.is_unknown() {
        return Bound::UNKNOWN;
    }
    let own = accuracy::lg_abs(v) - prec as f64 - 16.0;
    if bz.is_exact() {
        return Bound::both(own);
    }
    let r = bz.joint();
    let (Some(x), Some(y)) = (to_f64(&z.0), to_f64(&z.1)) else {
        return Bound::UNKNOWN;
    };
    let rad = r.exp2();
    if accuracy::part_contains_zero(&z.1, bz.im) && x - rad <= 0.0 {
        return Bound::UNKNOWN;
    }
    let Some(lpsi) = log2_psi_bound(x, y, rad) else {
        return Bound::UNKNOWN;
    };
    Bound::both(accuracy::lsum(own, lpsi + r + 1.0))
}

/// `log₂` of an upper bound on `|ψ(u)|` for `|u − (x + iy)| ≤ r`, or `None`
/// when the disk comes within eight radii of a pole (0, −1, −2, …) or its
/// radius is not small.
///
/// For `Re u ≥ ¼`, `ψ(u) = ln u − 1/(2u) − 2∫₀^∞ t dt/((t² + u²)(e^{2πt} − 1))`
/// (Binet), `|t² + u²| ≥ (Re u)²` and `∫₀^∞ t/(e^{2πt} − 1) dt = 1/24`:
/// `|ψ(u)| ≤ |ln|u|| + π/2 + 2 + 4/3`.  Left of that, `ψ(u) = ψ(1 − u) −
/// π cot πu` with `|cot πu| ≤ cosh(πb)/max(2d, sinh(π|b|))`, `b = Im u` and
/// `d` the distance to the nearest integer (`|sin πt| ≥ 2|t|` for `|t| ≤ ½`,
/// `sinh(π|b|) ≥ π|b|`).
fn log2_psi_bound(x: f64, y: f64, r: f64) -> Option<f64> {
    if !(r.is_finite() && r <= 0.125) {
        return None;
    }
    let right = |re: f64, im: f64| -> f64 {
        let m_hi = re.hypot(im) + r;
        let m_lo = (re - r).max(0.25);
        m_hi.ln().max(-m_lo.ln()) + std::f64::consts::FRAC_PI_2 + 2.0 + 4.0 / 3.0
    };
    if x - r >= 0.25 {
        return Some(right(x, y).log2());
    }
    // The distance from the disk to the nearest pole, and to the nearest
    // integer (for the cotangent).
    let dx = (x - x.round()).abs();
    let d = dx.hypot(y) - r;
    if d <= 8.0 * r {
        return None;
    }
    let b = y.abs() + r;
    let b_lo = (y.abs() - r).max(0.0);
    let pi = std::f64::consts::PI;
    // Far from the real axis `cosh(πb)/sinh(πb_lo)` is taken as
    // `e^{π(b − b_lo)}·(1 + e^{−2πb})/(1 − e^{−2πb_lo})`: both overflow `f64`
    // from `b ≈ 226`, and before 0.31 their quotient, NaN, left
    // `loggamma(−6/17 + 10⁸i)` without a bound at every precision.
    let cot = if pi * b_lo > 20.0 {
        (pi * (b - b_lo)).exp() * (1.0 + (-2.0 * pi * b).exp()) / (1.0 - (-2.0 * pi * b_lo).exp())
    } else {
        (pi * b).cosh() / (2.0 * d.min(0.5)).max((pi * b_lo).sinh())
    };
    let psi_mirror = right(1.0 - x, y);
    let total = psi_mirror + pi * cot;
    total.is_finite().then(|| total.log2())
}

/// A finite `f64` value of `x` (`None` beyond the range or for NaN).
fn to_f64(x: &BigFloat) -> Option<f64> {
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven)
        .ok()
        .filter(|v| v.is_finite())
}
