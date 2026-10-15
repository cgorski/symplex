//! The exponential integrals and the incomplete gamma functions at non-real
//! arguments (and at real ones where the value is not real): `Γ(a, z)`,
//! `γ(a, z)`, `E_n(z)` (`expint`, `E1`), `Ei`, `li`, `Si`, `Ci`, `Shi` and
//! `Chi`.  (The real routines are `arb_uppergamma`, `arb_expint`, `arb_ei`,
//! `arb_si_ci`, `arb_shi_chi` in `evalf.rs`.)  Before 0.45 every one of them
//! refused a non-real argument: `Ei(1 + i)`, "Ei of complex argument not yet
//! supported in evalf", and `expint(1, −1)`, `li(−2)` as well.
//!
//! # Branches
//!
//! Principal branches, as mpmath and SymPy: `Γ(a, z)`, `E_n(z)` and `γ(a,
//! z)` (through `z^a`) are cut along `(−∞, 0]`, `Ei`, `Ci`, `Chi` through
//! `ln z`, and `li(z) = Ei(ln z)`.  An argument exactly on a cut takes the
//! value from above (`E1(−1) = −Ei(1) − iπ`, `Ci(−2) = Ci(2) + iπ`), whatever
//! the sign of its zero imaginary part.  (`Ei` of a real argument is real
//! on both sides of 0 and stays with the real routine.)
//!
//! # `Γ(a, z)` for large `abs(z)`
//!
//! `Γ(a, z) = z^(a−1)e^(−z)(Σ_{k<n} u_k z^(−k) + R_n)`, `u_k = (a − 1)(a −
//! 2)⋯(a − k)`, since integrating by parts `n` times leaves `R_n =
//! u_n·z^(1−a)e^z·Γ(a − n, z)` (DLMF 8.11.2).  For `Re b ≤ 1`, `Γ(b, z) =
//! ∫ t^(b−1)e^(−t) dt` along the ray `t = z + s·e^(iφ)`, `s ≥ 0`, with
//! `abs(φ) < π/2` and `φ` on the side of `z` (the ray does not cross the
//! cut): on it `abs(t) ≥ abs(z)·σ` (`σ = 1` if the angle `ψ` between `z` and
//! the ray is at most `π/2`, `sin ψ` beyond), `arg t` lies between `arg z`
//! and `φ`, and `abs(e^(−t)) = e^(−Re z − s·cos φ)`, so
//!
//! ```text
//! abs(R_n) ≤ abs(u_n z^(−n))·σ^(Re a − n − 1)·e^(abs(Im a)·ψ)/cos φ
//! ```
//!
//! for `n ≥ Re a − 1`, minimised over a grid of `φ` (any `φ` gives a bound).
//! On the negative axis (`ψ > π/2` for every `φ`) the factor is about
//! `√(e·n)` at its best `φ`.  The expansion is used when that bound reaches
//! `2^−wp` of the leading term; it terminates (exactly) for a positive
//! integer `a`.  `E_n(z) = z^(n−1)Γ(1 − n, z)` has the same expansion with
//! the prefactor `e^(−z)/z` (`z^(n−1)·z^(−n) = 1/z` on the principal branch).
//!
//! # Small and moderate `abs(z)`
//!
//! * `γ(a, z) = z^a Σ_{k≥0} (−z)^k/(k!(a + k))` for `Re z < 0`, and `z^a
//!   e^(−z) Σ_{k≥0} z^k/((a)_{k+1})` for `Re z ≥ 0` (DLMF 8.7.1, 8.5.1, the
//!   form whose terms cancel least); `Γ(a, z) = Γ(a) − γ(a, z)`.  The rest
//!   after a term is at most the next term over `1 − q`, `q` the bound on
//!   the ratio of the terms beyond (`abs(z)/(n + 2)`, resp. `abs(z)/(Re a + n
//!   + 2)`), once `q < 1`.
//! * A positive integer `a = n`: `Γ(n, z) = (n − 1)!·e^(−z)·Σ_{k<n} z^k/k!`
//!   (DLMF 8.4.8).
//! * `a = −m`, a non-positive integer: `Γ(−m, z) = ((−1)^m/m!)·(E1(z) −
//!   e^(−z)·Σ_{k<m} (−1)^k k!/z^(k+1))` (DLMF 8.4.15) with `E1(z) = −γ − ln
//!   z − Σ_{k≥1} (−z)^k/(k·k!)` (DLMF 6.6.2).
//! * `Ei(z) = γ + ln z − Ein(−z)` continued off the real axis (DLMF 6.2.7;
//!   mpmath's and SymPy's `Ei`), that is `−E1(−z) ± iπ` for `±Im z > 0` by
//!   DLMF 6.2.4; `li(z) = Ei(ln z)` (DLMF 6.2.8).
//! * `Si`, `Shi`: the series `Σ (∓1)^k z^(2k+1)/((2k+1)(2k+1)!)` and its
//!   hyperbolic twin; `Ci`, `Chi`: `γ + ln z` plus `Σ_{k≥1} (∓1)^k
//!   z^(2k)/(2k(2k)!)` and its twin (DLMF 6.6.5, 6.6.6).  For large `abs(z)`
//!   from `E1(±z)`, `E1(±iz)`: `Shi(z)` is `(E1(z) − E1(−z))/2` plus `(ln
//!   z − ln(−z))/2`, `Chi(z)` is `−(E1(z) + E1(−z))/2` plus the same, `Si(z)`
//!   is `−i·Shi(iz)`, and `Ci(z)` is `−(E1(iz) + E1(−iz))/2` plus `c`, with
//!   `c = 0` for `arg z ∈ (−π/2, π/2]`, `iπ` above and `−iπ` below the left
//!   half-plane.
//!
//! Every truncation is below `2^−wp` of the largest term summed, the terms
//! are formed with `2·log₂ N + 8` guard bits for the roundings of `N` terms
//! and their partial sums, and the loss to cancellation (largest term over
//! the value) is measured and made up as in `zeta.rs` (`with_loss`).  The
//! structure — the asymptotic series where it converges, the `γ(a, z)`
//! series otherwise, the reductions of `Ei`, `Si`, `Ci`, `Shi`, `Chi` to
//! `E1` with their branch corrections — follows mpmath
//! (`mpmath/functions/expintegrals.py`, BSD); the remainder bound replaces
//! its size test.

use super::wp::{RM, Wp};
use crate::base::bigfloat::{is_strictly_positive, to_f64};
use astro_float::BigFloat;

use super::accuracy;
use super::zeta::{Sum, plus_int};
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_neg, c_one, c_sub, c_zero};
use crate::base::errors::SymplexError;

const LOG2E: f64 = std::f64::consts::LOG2_E;
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

/// Most terms a series or an expansion may take.
const MAX_TERMS: usize = 400_000;

/// Largest non-positive integer order `−m` evaluated by DLMF 8.4.15, and
/// positive integer order by the finite sum (beyond: the general series).
const MAX_FINITE: u64 = 5_000;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// The parts of `z` in `f64`, a zero imaginary part as `+0`.
fn parts(z: &Complex) -> Option<(f64, f64)> {
    let x = to_f64(&z.0)?;
    let y = to_f64(&z.1)?;
    Some((x, if y == 0.0 { 0.0 } else { y }))
}

fn round_c(v: Complex, prec: usize) -> Complex {
    (super::round_to(v.0, prec), super::round_to(v.1, prec))
}

fn is_zero(z: &Complex) -> bool {
    z.0.is_zero() && z.1.is_zero()
}

/// A complex number's parts in `f64` (`cbessel.rs`, `cspecial.rs`).
#[derive(Clone, Copy, Debug)]
pub(super) struct Parts {
    pub(super) x: f64,
    pub(super) y: f64,
}

/// [`parts`] as a [`Parts`].
pub(super) fn f64_parts(z: &Complex) -> Option<Parts> {
    parts(z).map(|(x, y)| Parts { x, y })
}

/// `z` with a zero imaginary part made `+0` (the side of the cut is "from
/// above" for every routine here).
fn above(z: &Complex) -> Complex {
    if z.1.is_zero() {
        (z.0.clone(), BigFloat::new(64))
    } else {
        z.clone()
    }
}

/// Is `z` in the closed upper half-plane (a zero imaginary part counts as
/// `+0`)?
fn upper(z: &Complex) -> bool {
    z.1.is_zero() || is_strictly_positive(&z.1)
}

/// `i·z`.
fn times_i(z: &Complex) -> Complex {
    (z.1.neg(), z.0.clone())
}

/// `−i·z`.
fn times_minus_i(z: &Complex) -> Complex {
    (z.1.clone(), z.0.neg())
}

/// `z/n` for a positive integer `n`.
fn div_u(z: &Complex, n: u64, p: usize) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.div(&d, p, RM), z.1.div(&d, p, RM))
}

/// `z·n` for a positive integer `n`.
fn mul_u(z: &Complex, n: u64, p: usize) -> Complex {
    let d = BigFloat::from_u64(n, 64);
    (z.0.mul(&d, p, RM), z.1.mul(&d, p, RM))
}

/// `z/2` exactly.
fn half(z: &Complex) -> Complex {
    let mut v = z.clone();
    for x in [&mut v.0, &mut v.1] {
        if let Some(e) = x.exponent() {
            x.set_exponent(e - 1);
        }
    }
    v
}

/// `z·x` for a real `x`.
fn scale(z: &Complex, x: &BigFloat, p: usize) -> Complex {
    (z.0.mul(x, p, RM), z.1.mul(x, p, RM))
}

/// `a + c` exactly for an integer `c` (the real part; `a` complex).
fn shift(a: &Complex, c: i64, p: usize) -> Complex {
    (plus_int(&a.0, c, p), a.1.clone())
}

/// Guard bits for the roundings of `n` terms (see the module documentation).
pub(super) fn guard(n: usize) -> usize {
    2 * accuracy::ceil_log2(n.max(2)) as usize + 8
}

fn add_term(sum: &mut Sum, t: &Complex, p: usize) {
    sum.largest = sum.largest.max(accuracy::lg_abs(t));
    sum.value = c_add(&sum.value, t, p, RM);
}

fn new_sum(p: usize) -> Sum {
    Sum {
        value: c_zero(p),
        largest: f64::NEG_INFINITY,
    }
}

/// `Ln z` (principal; a negative real `z` from above).
pub(super) fn ln_c(z: &Complex, w: &mut Wp<'_>) -> Complex {
    super::c_ln(&above(z), w)
}

/// `exp(e)` with the bits of `abs(e)` as guard (its absolute error is the
/// result's relative one), rounded to `p` bits.
fn exp_c(e: &Complex, w: &mut Wp<'_>) -> Complex {
    let p = w.bits();
    round_c(super::c_exp(e, w), p)
}

/// `exp(c·Ln z − z)` (`minus_z`) or `exp(c·Ln z)` for a complex `c`: `z^c
/// e^(−z)` or `z^c`, to `p` bits relative (the exponent formed to an
/// absolute `2^−p`).
pub(super) fn power_exp(z: &Complex, c: &Complex, minus_z: bool, w: &mut Wp<'_>) -> Complex {
    let p = w.bits();
    let lz = accuracy::lg_abs(z);
    let lnz = (lz.abs() * std::f64::consts::LN_2) + 4.0;
    let size = accuracy::lg_abs(c).exp2() * lnz + if minus_z { lz.exp2() } else { 0.0 };
    let pe = p + super::magnitude_bits(size) + 8;
    let l = ln_c(z, &mut w.at(pe));
    let mut e = c_mul(c, &l, pe, RM);
    if minus_z {
        e = c_sub(&e, z, pe, RM);
    }
    exp_c(&e, &mut w.at(pe.max(p)))
}

/// `e^(−z)/z` to `p` bits relative.
fn exp_over(z: &Complex, w: &mut Wp<'_>) -> Complex {
    let p = w.bits();
    let pe = p + super::magnitude_bits(accuracy::lg_abs(z).exp2()) + 8;
    let e = super::c_exp(&c_neg(z), &mut w.at(pe));
    round_c(c_div(&e, z, pe, RM), p)
}

/// `Γ(a)` for `a` off the poles: `arb_gamma_real` for a real `a`,
/// `gamma_complex` otherwise, to `p` bits relative (also `cspecial.rs`).
pub(super) fn gamma_of(a: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let p = w.bits();
    if a.1.is_zero() {
        let af = to_f64(&a.0).unwrap_or(f64::INFINITY).abs();
        let g = super::magnitude_bits((af + 2.0) * ((af + 2.0).ln() + 1.0));
        let v = super::arb_gamma_real(&a.0, &mut w.at(p + g + 16))?;
        Ok((super::round_to(v, p), BigFloat::new(p)))
    } else {
        super::zeta::gamma_complex(a, w)
    }
}

/// `a` as a non-positive integer `−m` (exactly), `m ≤ MAX_FINITE`.
fn nonpositive_int(a: &Complex) -> Option<u64> {
    if !a.1.is_zero() || !(a.0.is_zero() || (a.0.is_int() && a.0.is_negative())) {
        return None;
    }
    let m = -to_f64(&a.0)?;
    (m <= MAX_FINITE as f64).then_some(m as u64)
}

/// `a` as a positive integer `n` (exactly), `n ≤ MAX_FINITE`.
fn positive_int(a: &Complex) -> Option<u64> {
    if !a.1.is_zero() || a.0.is_zero() || !a.0.is_int() || a.0.is_negative() {
        return None;
    }
    let n = to_f64(&a.0)?;
    (n <= MAX_FINITE as f64).then_some(n as u64)
}

// ── The asymptotic expansion ────────────────────────────────────────────────

/// `log₂` of the factor `σ^(Re a − n − 1)·e^(abs(Im a)·ψ)/cos φ` of the
/// remainder bound after `n` terms (module documentation), minimised over a
/// grid of rays; `θ = arg z`.  `+∞` when no ray gives a bound.
fn remainder_factor(re_a: f64, im_a: f64, theta: f64, n: usize) -> f64 {
    let m = n as f64 + 1.0 - re_a;
    let tau = im_a.abs();
    // (The side only restricts the rays tried: every ray gives a bound, and
    // the bound is symmetric in the sign of `θ`.)
    let side = if theta > 0.0 {
        1.0
    } else if theta < 0.0 {
        -1.0
    } else {
        0.0
    };
    let mut best = f64::INFINITY;
    let mut try_phi = |phi: f64| {
        if phi.abs() >= HALF_PI || phi * side < 0.0 {
            return;
        }
        let psi = (theta - phi).abs();
        let s = if psi <= HALF_PI { 1.0 } else { psi.sin() };
        let c = phi.cos();
        if !(s > 0.0 && c > 0.0) {
            return;
        }
        let l = -m * s.log2() + tau * psi * LOG2E - c.log2();
        if l < best {
            best = l;
        }
    };
    try_phi(0.0);
    try_phi(theta / 2.0);
    let dir = if side == 0.0 { 1.0 } else { side };
    for j in 1..64 {
        try_phi(dir * f64::from(j) / 64.0 * HALF_PI);
        if side == 0.0 {
            try_phi(-f64::from(j) / 64.0 * HALF_PI);
        }
    }
    for k in 2..80 {
        try_phi(dir * (HALF_PI - (-f64::from(k) / 4.0).exp2()));
    }
    // A margin for the `f64` arithmetic.
    best + 1e-6 * best.abs() + 1e-6
}

/// The number of terms `n` of the expansion of `Γ(a, z)` after which the
/// remainder bound is below `2^−(wp+2)` of the leading term, or `None` when
/// the expansion does not get there (simulated in `f64`).
fn asymptotic_terms(re_a: f64, im_a: f64, abs_z: f64, theta: f64, wp: usize) -> Option<usize> {
    if !(abs_z.is_finite() && abs_z > 0.0 && re_a.is_finite() && im_a.is_finite()) {
        return None;
    }
    // The terms decrease while `abs(a − k − 1) < abs(z)`: a rough gate.
    let lz = abs_z.log2();
    if abs_z < 2.0 {
        return None;
    }
    let start = (re_a - 1.0).ceil().max(0.0) as usize;
    let past = abs_z + re_a.abs() + im_a.abs() + 8.0;
    let target = -(wp as f64) - 2.0;
    let mut lt = 0.0f64;
    for k in 0..MAX_TERMS {
        if k >= start && lt + remainder_factor(re_a, im_a, theta, k) < target {
            return Some(k);
        }
        // (A terminating expansion in `f64`; `asymptotic_bracket` bounds the
        // actual next term.)
        let f = (re_a - k as f64 - 1.0).hypot(im_a);
        if f == 0.0 {
            return Some(k + 1);
        }
        let next = lt + f.log2() - lz;
        if k as f64 > past && next > lt {
            return None;
        }
        lt = next;
    }
    None
}

/// `Σ_{k<n} u_k z^(−k)` of the expansion of `Γ(a, z)` (module
/// documentation), with its remainder below `2^−wp` (relative to the
/// leading term 1), or `None` where the expansion does not converge so far.
fn asymptotic_bracket(a: &Complex, z: &Complex, wp: usize) -> Option<Sum> {
    let (ar, ai) = parts(a)?;
    let (x, y) = parts(z)?;
    let theta = y.atan2(x);
    let n = asymptotic_terms(ar, ai, x.hypot(y), theta, wp)?;
    let p = wp + guard(n + 2);
    let mut sum = new_sum(p);
    let mut t = c_one(p);
    let mut exact = false;
    for k in 0..n {
        add_term(&mut sum, &t, p);
        let f = shift(a, -(k as i64) - 1, p);
        t = c_div(&c_mul(&t, &f, p, RM), z, p, RM);
        if is_zero(&t) {
            exact = true;
            break;
        }
    }
    if !exact {
        // The bound on the actual first neglected term.
        let b = accuracy::lg_abs(&t) + remainder_factor(ar, ai, theta, n);
        if b.is_nan() || b >= -(wp as f64) - 1.0 {
            return None;
        }
    }
    Some(sum)
}

// ── Series ──────────────────────────────────────────────────────────────────

/// A term budget for a series in `abs(z)` with parameter `a`.
pub(super) fn budget(abs_z: f64, abs_a: f64, wp: usize) -> usize {
    let n = 4.0 * abs_z + 2.0 * abs_a + 2.0 * wp as f64 + 64.0;
    if n.is_finite() {
        (n as usize).min(MAX_TERMS)
    } else {
        MAX_TERMS
    }
}

/// `γ(a, z)` by its series (module documentation) for `a` not a non-positive
/// integer and `z ≠ 0`, at working precision `wp`.
fn lower_series(a: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let ((ar, ai), (x, y)) = match (parts(a), parts(z)) {
        (Some(pa), Some(pz)) => (pa, pz),
        _ => return Err(super::unevaluable("incomplete gamma argument beyond f64")),
    };
    let abs_z = x.hypot(y);
    let max = budget(abs_z, ar.hypot(ai), wp);
    let p = wp + guard(max);
    let kummer = x < 0.0;
    let mut sum = new_sum(p);
    let mut converged = false;
    if kummer {
        // Σ t_k/(a + k), t_k = (−z)^k/k!.
        let mz = c_neg(z);
        let mut t = c_one(p);
        for k in 0..max {
            let d = shift(a, k as i64, p);
            add_term(&mut sum, &c_div(&t, &d, p, RM), p);
            t = div_u(&c_mul(&t, &mz, p, RM), k as u64 + 1, p);
            let q = abs_z / (k as f64 + 2.0);
            let dmin = ar + k as f64 + 1.0;
            if q < 1.0 && dmin > 0.0 {
                let rest = accuracy::lg_abs(&t) - dmin.log2() - (1.0 - q).log2();
                if rest < sum.largest - wp as f64 || is_zero(&t) {
                    converged = true;
                    break;
                }
            }
        }
    } else {
        // Σ s_k, s_0 = 1/a, s_{k+1} = s_k·z/(a + k + 1).
        let mut s = c_div(&c_one(p), &shift(a, 0, p), p, RM);
        for k in 0..max {
            add_term(&mut sum, &s, p);
            s = c_div(&c_mul(&s, z, p, RM), &shift(a, k as i64 + 1, p), p, RM);
            let dmin = ar + k as f64 + 2.0;
            let q = abs_z / dmin;
            if dmin > 0.0 && q < 1.0 {
                let rest = accuracy::lg_abs(&s) - (1.0 - q).log2();
                if rest < sum.largest - wp as f64 || is_zero(&s) {
                    converged = true;
                    break;
                }
            }
        }
    }
    if !converged {
        return Err(super::series_did_not_converge("incomplete gamma", max));
    }
    let pref = power_exp(z, a, !kummer, &mut w.at(p));
    Ok(Sum {
        largest: accuracy::lg_abs(&pref) + sum.largest,
        value: round_c(c_mul(&pref, &sum.value, p, RM), wp),
    })
}

/// `E1(z) = −γ − ln z − Σ_{k≥1} (−z)^k/(k·k!)` for `z ≠ 0`.
fn e1_series(z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let Some((x, y)) = parts(z) else {
        return Err(super::unevaluable("E1 argument beyond f64"));
    };
    let abs_z = x.hypot(y);
    let max = budget(abs_z, 0.0, wp);
    let p = wp + guard(max);
    let mz = c_neg(z);
    let mut sum = new_sum(p);
    let mut t = mz.clone(); // (−z)^k/k!
    let mut converged = false;
    for k in 1..max {
        add_term(&mut sum, &div_u(&t, k as u64, p), p);
        t = div_u(&c_mul(&t, &mz, p, RM), k as u64 + 1, p);
        let q = abs_z / (k as f64 + 2.0);
        if q < 1.0 {
            let rest = accuracy::lg_abs(&t) - ((k + 1) as f64).log2() - (1.0 - q).log2();
            if rest < sum.largest.max(0.0) - wp as f64 || is_zero(&t) {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return Err(super::series_did_not_converge("E1", max));
    }
    let g = super::arb_euler_gamma(&mut w.at(p))?;
    let l = ln_c(z, &mut w.at(p));
    let v = c_neg(&c_add(
        &c_add(&sum.value, &l, p, RM),
        &(g, BigFloat::new(p)),
        p,
        RM,
    ));
    Ok(Sum {
        largest: sum.largest.max(accuracy::lg_abs(&l)).max(0.0),
        value: v,
    })
}

/// `Γ(−m, z)` by DLMF 8.4.15 (module documentation), `z ≠ 0`.
fn upper_negint(m: u64, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let p = wp + guard(m as usize + 2);
    let e1 = e1_series(z, &mut w.at(p))?;
    if m == 0 {
        return Ok(e1);
    }
    // Σ_{k<m} (−1)^k k!/z^(k+1), the terms by `t_{k+1} = −t_k·(k+1)/z`.
    let mut s = new_sum(p);
    let mut t = c_div(&c_one(p), z, p, RM);
    for k in 0..m {
        add_term(&mut s, &t, p);
        t = c_neg(&c_div(&mul_u(&t, k + 1, p), z, p, RM));
    }
    let e = exp_c(
        &c_neg(z),
        &mut w.at(p + super::magnitude_bits(accuracy::lg_abs(z).exp2()) + 8),
    );
    let es = c_mul(&e, &s.value, p, RM);
    let mut v = c_sub(&e1.value, &es, p, RM);
    let largest = e1.largest.max(accuracy::lg_abs(&e) + s.largest);
    // (−1)^m/m!.
    let mut f = BigFloat::from_i32(1, p);
    for k in 2..=m {
        f = f.mul(&BigFloat::from_u64(k, 64), p, RM);
    }
    let lf = super::zeta::log2_factorial(m);
    v = (v.0.div(&f, p, RM), v.1.div(&f, p, RM));
    if m % 2 == 1 {
        v = c_neg(&v);
    }
    Ok(Sum {
        value: v,
        largest: largest - lf + 1e-9,
    })
}

/// `Γ(n, z) = (n − 1)!·e^(−z)·Σ_{k<n} z^k/k!` for a positive integer `n`.
fn upper_posint(n: u64, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let p = wp + guard(n as usize + 2);
    let mut s = new_sum(p);
    let mut t = c_one(p);
    for k in 0..n {
        add_term(&mut s, &t, p);
        t = div_u(&c_mul(&t, z, p, RM), k + 1, p);
    }
    // (n − 1)!·e^(−z) = exp(ln((n − 1)!) − z).
    let mut f = BigFloat::from_i32(1, p);
    for k in 2..n {
        f = f.mul(&BigFloat::from_u64(k, 64), p, RM);
    }
    let e = exp_c(
        &c_neg(z),
        &mut w.at(p + super::magnitude_bits(accuracy::lg_abs(z).exp2()) + 8),
    );
    let e = scale(&e, &f, p);
    Ok(Sum {
        largest: accuracy::lg_abs(&e) + s.largest,
        value: c_mul(&e, &s.value, p, RM),
    })
}

// ── Γ(a, z), γ(a, z), E_n(z) at a working precision ────────────────────────

/// `Γ(a, z)` at working precision `wp` by the asymptotic expansion, or `None`
/// where it does not converge to `2^−wp` (`erfc z = Γ(1/2, z²)/√π`,
/// `cerf.rs`).
pub(super) fn upper_asymptotic(a: &Complex, z: &Complex, w: &mut Wp<'_>) -> Option<Sum> {
    let wp = w.bits();
    let b = asymptotic_bracket(a, z, wp)?;
    let p = wp + 8;
    let pref = power_exp(z, &shift(a, -1, p), true, &mut w.at(p));
    Some(Sum {
        largest: accuracy::lg_abs(&pref) + b.largest,
        value: c_mul(&pref, &b.value, p, RM),
    })
}

/// `Γ(a, z)` at working precision `wp`.
fn upper_at(a: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    if is_zero(z) {
        if is_strictly_positive(&a.0) {
            let g = gamma_of(a, w)?;
            return Ok(Sum {
                largest: accuracy::lg_abs(&g),
                value: g,
            });
        }
        return Err(super::unevaluable(
            "uppergamma(a, 0) with Re a ≤ 0 is infinite",
        ));
    }
    if let Some(s) = upper_asymptotic(a, z, w) {
        return Ok(s);
    }
    if let Some(m) = nonpositive_int(a) {
        return upper_negint(m, z, w);
    }
    if a.1.is_zero() && (a.0.is_zero() || (a.0.is_int() && a.0.is_negative())) {
        return Err(super::unevaluable(
            "uppergamma at a non-positive integer order this large not supported in evalf",
        ));
    }
    if let Some(n) = positive_int(a) {
        return upper_posint(n, z, w);
    }
    let low = lower_series(a, z, w)?;
    let g = gamma_of(a, &mut w.at(wp + 8))?;
    Ok(Sum {
        largest: low.largest.max(accuracy::lg_abs(&g)),
        value: c_sub(&g, &low.value, wp + 8, RM),
    })
}

/// `γ(a, z)` at working precision `wp` (`a` not a non-positive integer).
fn lower_at(a: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    if nonpositive_int(a).is_some() || (a.1.is_zero() && a.0.is_int() && a.0.is_negative()) {
        return Err(super::unevaluable(
            "lowergamma at a non-positive integer order is infinite",
        ));
    }
    if is_zero(z) {
        if is_strictly_positive(&a.0) {
            return Ok(Sum {
                value: c_zero(wp),
                largest: f64::NEG_INFINITY,
            });
        }
        return Err(super::unevaluable(
            "lowergamma(a, 0) with Re a ≤ 0 is infinite",
        ));
    }
    if let Some(b) = asymptotic_bracket(a, z, wp) {
        let p = wp + 8;
        let pref = power_exp(z, &shift(a, -1, p), true, &mut w.at(p));
        let up = c_mul(&pref, &b.value, p, RM);
        let g = gamma_of(a, &mut w.at(p))?;
        return Ok(Sum {
            largest: (accuracy::lg_abs(&pref) + b.largest).max(accuracy::lg_abs(&g)),
            value: c_sub(&g, &up, p, RM),
        });
    }
    lower_series(a, z, w)
}

/// `E_n(z) = z^(n−1)Γ(1 − n, z)` at working precision `wp`, `z ≠ 0`.
fn expint_at(n: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let p = wp + 8;
    let a = (plus_int(&n.0.neg(), 1, p), n.1.neg());
    if let Some(b) = asymptotic_bracket(&a, z, wp) {
        let pref = exp_over(z, &mut w.at(p));
        return Ok(Sum {
            largest: accuracy::lg_abs(&pref) + b.largest,
            value: c_mul(&pref, &b.value, p, RM),
        });
    }
    let g = upper_at(&a, z, w)?;
    // z^(n−1): an integer power exactly by repeated products, else `exp`.
    let nm1 = shift(n, -1, p);
    let pw = if nm1.1.is_zero() && (nm1.0.is_zero() || nm1.0.is_int()) {
        let k = to_f64(&nm1.0).unwrap_or(f64::INFINITY);
        if k.abs() > MAX_FINITE as f64 {
            power_exp(z, &nm1, false, &mut w.at(p))
        } else {
            let q = p + guard(k.abs() as usize + 2);
            let pos = crate::base::bigcomplex::c_powi(z, k.abs() as usize, q, RM);
            if k < 0.0 {
                c_div(&c_one(q), &pos, q, RM)
            } else {
                pos
            }
        }
    } else {
        power_exp(z, &nm1, false, &mut w.at(p))
    };
    let lp = accuracy::lg_abs(&pw);
    Ok(Sum {
        largest: g.largest + lp,
        value: c_mul(&pw, &g.value, p, RM),
    })
}

/// `E1(z)` at working precision `wp`, `z ≠ 0`.
fn e1_at(z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let one = (BigFloat::from_i32(1, 64), BigFloat::new(64));
    expint_at(&one, z, w)
}

/// `iπ` or `−iπ` (`negative`), halved when `halve`.
fn i_pi(negative: bool, halve: bool, w: &mut Wp<'_>) -> Complex {
    let p = w.bits();
    let mut pi = w.cc.pi(p, RM).clone();
    if halve && let Some(e) = pi.exponent() {
        pi.set_exponent(e - 1);
    }
    (BigFloat::new(p), if negative { pi.neg() } else { pi })
}

/// `Ei(z)` for a non-real `z` at working precision `wp`: `−E1(−z) ± iπ`.
fn ei_at(z: &Complex, w: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = w.bits();
    let e = e1_at(&c_neg(z), w)?;
    let p = wp + 8;
    let ip = i_pi(z.1.is_negative() && !z.1.is_zero(), false, &mut w.at(p));
    Ok(Sum {
        largest: e.largest.max(2.0),
        value: c_sub(&ip, &e.value, p, RM),
    })
}

/// Which trigonometric or hyperbolic integral.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TrigInt {
    Si,
    Ci,
    Shi,
    Chi,
}

/// `Si`, `Ci`, `Shi` or `Chi` of `z ≠ 0` at working precision `wp`.
fn trig_at(kind: TrigInt, z: &Complex, wpc: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    let z = &above(z);
    let w = match kind {
        TrigInt::Si | TrigInt::Ci => times_i(z),
        _ => z.clone(),
    };
    let zero = (BigFloat::new(64), BigFloat::new(64));
    let mw = c_neg(&w);
    let p = wp + 8;
    // Large `abs(z)`: from `E1(±w)` when both expansions converge.
    if let (Some(bp), Some(bm)) = (
        asymptotic_bracket(&zero, &w, wp + 2),
        asymptotic_bracket(&zero, &mw, wp + 2),
    ) {
        let ep = exp_over(&w, &mut wpc.at(p));
        let em = exp_over(&mw, &mut wpc.at(p));
        let e_plus = c_mul(&ep, &bp.value, p, RM);
        let e_minus = c_mul(&em, &bm.value, p, RM);
        let largest = (accuracy::lg_abs(&ep) + bp.largest)
            .max(accuracy::lg_abs(&em) + bm.largest)
            .max(2.0);
        let value = match kind {
            TrigInt::Shi | TrigInt::Si => {
                // (E1(w) − E1(−w))/2 + (ln w − ln(−w))/2.
                let c = i_pi(
                    !is_strictly_positive(&w.1) && !w_neg_real(&w),
                    true,
                    &mut wpc.at(p),
                );
                let s = c_add(&half(&c_sub(&e_plus, &e_minus, p, RM)), &c, p, RM);
                if kind == TrigInt::Si {
                    times_minus_i(&s)
                } else {
                    s
                }
            }
            TrigInt::Chi => {
                let c = i_pi(
                    !is_strictly_positive(&w.1) && !w_neg_real(&w),
                    true,
                    &mut wpc.at(p),
                );
                c_sub(&c, &half(&c_add(&e_plus, &e_minus, p, RM)), p, RM)
            }
            TrigInt::Ci => {
                let s = c_neg(&half(&c_add(&e_plus, &e_minus, p, RM)));
                let (x_neg, x_zero) = (z.0.is_negative() && !z.0.is_zero(), z.0.is_zero());
                if x_neg {
                    c_add(&s, &i_pi(!upper(z), false, &mut wpc.at(p)), p, RM)
                } else if x_zero && !upper(z) {
                    c_add(&s, &i_pi(true, false, &mut wpc.at(p)), p, RM)
                } else {
                    s
                }
            }
        };
        return Ok(Sum { value, largest });
    }
    trig_series(kind, z, wpc)
}

/// Is `w` a negative real number (on the cut, from above)?
fn w_neg_real(w: &Complex) -> bool {
    w.1.is_zero() && w.0.is_negative() && !w.0.is_zero()
}

/// The power series of `Si`, `Ci`, `Shi`, `Chi` (module documentation).
fn trig_series(kind: TrigInt, z: &Complex, wpc: &mut Wp<'_>) -> Result<Sum, SymplexError> {
    let wp = wpc.bits();
    let Some((x, y)) = parts(z) else {
        return Err(super::unevaluable(
            "sine/cosine integral argument beyond f64",
        ));
    };
    let abs_z = x.hypot(y);
    let max = budget(abs_z, 0.0, wp);
    let p = wp + guard(max);
    let z2 = c_mul(z, z, p, RM);
    let w = match kind {
        TrigInt::Si | TrigInt::Ci => c_neg(&z2),
        _ => z2,
    };
    let aw = abs_z * abs_z;
    let odd = matches!(kind, TrigInt::Si | TrigInt::Shi);
    let mut sum = new_sum(p);
    let mut converged = false;
    if odd {
        // c_k = z^(2k+1)/(2k+1)!, term c_k/(2k+1).
        let mut c = z.clone();
        for k in 0..max as u64 {
            add_term(&mut sum, &div_u(&c, 2 * k + 1, p), p);
            c = div_u(&c_mul(&c, &w, p, RM), (2 * k + 2) * (2 * k + 3), p);
            let q = aw / ((2 * k + 4) as f64 * (2 * k + 5) as f64);
            if q < 1.0 {
                let rest = accuracy::lg_abs(&c) - ((2 * k + 3) as f64).log2() - (1.0 - q).log2();
                if rest < sum.largest - wp as f64 || is_zero(&c) {
                    converged = true;
                    break;
                }
            }
        }
    } else {
        // d_k = z^(2k)/(2k)!, term d_k/(2k), k ≥ 1.
        let mut d = div_u(&w, 2, p);
        for k in 1..max as u64 {
            add_term(&mut sum, &div_u(&d, 2 * k, p), p);
            d = div_u(&c_mul(&d, &w, p, RM), (2 * k + 1) * (2 * k + 2), p);
            let q = aw / ((2 * k + 3) as f64 * (2 * k + 4) as f64);
            if q < 1.0 {
                let rest = accuracy::lg_abs(&d) - ((2 * k + 2) as f64).log2() - (1.0 - q).log2();
                if rest < sum.largest.max(0.0) - wp as f64 || is_zero(&d) {
                    converged = true;
                    break;
                }
            }
        }
    }
    if !converged {
        return Err(super::series_did_not_converge("sine/cosine integral", max));
    }
    if odd {
        return Ok(sum);
    }
    let g = super::arb_euler_gamma(&mut wpc.at(p))?;
    let l = ln_c(z, &mut wpc.at(p));
    Ok(Sum {
        largest: sum.largest.max(accuracy::lg_abs(&l)).max(0.0),
        value: c_add(&c_add(&sum.value, &l, p, RM), &(g, BigFloat::new(p)), p, RM),
    })
}

// ── Entry points ───────────────────────────────────────────────────────────

fn special_input(z: &Complex) -> bool {
    [&z.0, &z.1].iter().any(|x| x.is_nan() || x.is_inf())
}

/// `Γ(a, z)` (complex), rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus.
pub(super) fn upper_gamma(
    a: &Complex,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if special_input(a) || special_input(z) {
        return Err(super::unevaluable("uppergamma of special float value"));
    }
    tracing::debug!(prec, "evalf: uppergamma at a complex argument");
    super::zeta::with_loss(w, &mut |wp, cc| upper_at(a, z, &mut Wp::new(wp, cc)))
}

/// `γ(a, z)` (complex), as [`upper_gamma`].
pub(super) fn lower_gamma(
    a: &Complex,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if special_input(a) || special_input(z) {
        return Err(super::unevaluable("lowergamma of special float value"));
    }
    if is_zero(z) && is_strictly_positive(&a.0) {
        return Ok(c_zero(prec + 16));
    }
    tracing::debug!(prec, "evalf: lowergamma at a complex argument");
    super::zeta::with_loss(w, &mut |wp, cc| lower_at(a, z, &mut Wp::new(wp, cc)))
}

/// `E_n(z)` (complex), as [`upper_gamma`].
pub(super) fn expint(n: &Complex, z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if special_input(n) || special_input(z) {
        return Err(super::unevaluable("expint of special float value"));
    }
    if is_zero(z) {
        // E_n(0) = 1/(n − 1) for Re n > 1.
        let nm1 = shift(n, -1, prec + 64);
        if is_strictly_positive(&nm1.0) {
            return Ok(round_c(
                c_div(&c_one(prec + 32), &nm1, prec + 32, RM),
                prec + 16,
            ));
        }
        return Err(super::unevaluable("expint(n, 0) with Re n ≤ 1 is infinite"));
    }
    tracing::debug!(prec, "evalf: expint at a complex argument");
    super::zeta::with_loss(w, &mut |wp, cc| expint_at(n, z, &mut Wp::new(wp, cc)))
}

/// `Ei(z)` for a non-real `z`, as [`upper_gamma`].
pub(super) fn ei(z: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if special_input(z) {
        return Err(super::unevaluable("Ei of special float value"));
    }
    tracing::debug!(prec, "evalf: Ei at a complex argument");
    super::zeta::with_loss(w, &mut |wp, cc| ei_at(z, &mut Wp::new(wp, cc)))
}

/// `li(z) = Ei(ln z)` for `z` off `[0, ∞)`, as [`upper_gamma`].
pub(super) fn li(z: &Complex, wpc: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = wpc.bits();
    if special_input(z) {
        return Err(super::unevaluable("li of special float value"));
    }
    if is_zero(z) {
        return Ok(c_zero(prec + 16));
    }
    tracing::debug!(prec, "evalf: li at a complex argument");
    super::zeta::with_loss(wpc, &mut |wp, cc| {
        // The rounding of ln z moves Ei by abs(e^w/w)·abs(Δw) ≤ abs(z)·2^−wp.
        let w = ln_c(z, &mut Wp::new(wp + 16, cc));
        if is_zero(&w) {
            return Err(super::unevaluable("li(1) is -∞"));
        }
        let e = ei_at(&w, &mut Wp::new(wp, cc))?;
        Ok(Sum {
            largest: e.largest.max(accuracy::lg_abs(z)),
            value: e.value,
        })
    })
}

/// `Si`, `Ci`, `Shi` or `Chi` (complex), as [`upper_gamma`].
pub(super) fn trig_integral(
    kind: TrigInt,
    z: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    if special_input(z) {
        return Err(super::unevaluable(
            "sine/cosine integral of special float value",
        ));
    }
    if is_zero(z) {
        return match kind {
            TrigInt::Si | TrigInt::Shi => Ok(c_zero(prec + 16)),
            _ => Err(super::unevaluable("Ci(0) and Chi(0) are -∞")),
        };
    }
    tracing::debug!(
        prec,
        ?kind,
        "evalf: sine/cosine integral at a complex argument"
    );
    super::zeta::with_loss(w, &mut |wp, cc| trig_at(kind, z, &mut Wp::new(wp, cc)))
}
