//! Complete and incomplete elliptic integrals `K(m)`, `E(m)`, `F(φ | m)` and
//! `Π(n | m)` where the value or a parameter is not real, by Carlson's
//! symmetric integrals over ℂ.
//!
//! The real routines of `evalf.rs` (`arb_elliptic_*`) cover a real
//! parameter with a real value: `m ≤ 1`, and `1 − m sin²φ > 0` for `F`.
//! Beyond that the value is complex, and so it is for a non-real `m`.
//! These routines take every such case with
//!
//! ```text
//! K(m) = R_F(0, 1 − m, 1)
//! E(m) = R_F(0, 1 − m, 1) − (m/3)·R_D(0, 1 − m, 1)
//! F(φ | m) = sin φ₀ · R_F(cos²φ₀, 1 − m sin²φ₀, 1) + 2k·K(m),   φ = φ₀ + kπ, |Re φ₀| ≤ π/2
//! ```
//!
//! (DLMF 19.25.1, 19.25.5; the reduction is DLMF 19.2.10).  For a real `m >
//! 1`, `K` and `E` are taken by the reciprocal-modulus transformation (DLMF
//! 19.7.3, `K(m) = (K(1/m) − i·K(1 − 1/m))/√m` here) with each part a real
//! integral computed to its own precision ([`beyond_one`]; up to 0.55
//! `E(10⁶⁰⁰)` was refused, its real part `7.85·10⁻³⁰¹` 2,000 bits below the
//! imaginary one).  A non-real `φ` takes the same formula with the complex
//! `sin φ₀`, `cos φ₀` ([`incomplete_f_complex`], mpmath's `ellipf` and so
//! SymPy's; refused before 0.56), whose cuts are the lines `Re φ = (k + ½)π`
//! and the curves where `1 − m sin²φ₀ < 0`.
//!
//! # The complete integral of the third kind
//!
//! ```text
//! Π(n | m) = R_F(0, 1 − m, 1) + (n/3)·R_J(0, 1 − m, 1, 1 − n)       (DLMF 19.25.2)
//! ```
//!
//! with `R_J` by Carlson's duplication (Carlson 1995, the form with `R_C(1,
//! 1 + eₘ)` that mpmath's `RJ_calc` uses), which is valid when the
//! arguments have positive real parts.  Elsewhere mpmath first integrates
//! numerically along a path into the right half-plane; here exact
//! transformations do it:
//!
//! * arguments `1 − m`, 1, `1 − n` that lie in an open half-plane are
//!   rotated into the right one, `R_J(x, y, z, p) = q^{3/2}·R_J(qx, qy, qz,
//!   qp)` (`R_F` with `q^{1/2}`) for `|arg q| < π/2` with `Re(q·w) > 0` for
//!   each argument `w`: the integral along `[0, ∞)` turned to the ray of
//!   `1/q` sweeps no singularity, and the principal roots split;
//! * a real `n > 1` with a real `m < 1`: `Π(n | m) = K(m) − Π(m/n | m) ∓
//!   i·(π/2)·√(n/((n − 1)(n − m)))` (DLMF 19.6.5 for the real part, the
//!   principal value; the imaginary part is half the residue at the pole
//!   `sin²θ = 1/n`);
//! * a real `m > 1`: the integral over `θ ∈ [0, π/2]` split where `1 − m
//!   sin²θ` changes sign, `Π(n | m) = Π(n/m | 1/m)/√m − i·Π(n₁ | 1 −
//!   1/m)/((1 − n)√m)` with `n₁ = n(m − 1)/(m(n − 1))` (the first part by
//!   the reciprocal-modulus substitution `sin ψ = √m·sin θ`, DLMF 19.7.4;
//!   the second by `m·sin²θ = 1 + (m − 1)cos²ψ`), for a real or a non-real
//!   `n`.
//!
//! * a real `n > 1` with a non-real `m`, and non-real arguments that lie in
//!   no half-plane together with 1 (since 0.56): `Π(n | m) = K(m) − Π(m/n |
//!   m) + (π/2)/(√(1 − n)·√(1 − m/n))` with the two principal roots taken
//!   apart ([`pi_involution`], DLMF 19.7.8 continued); the arguments of
//!   `Π(m/n | m)` then lie in a half-plane.
//! * a non-real `m` with `Re m > 1` that neither of these reaches (`m` next
//!   to the cut `(1, ∞)`, `Π(½ − i/3 | 4 + i/10)`; since 0.59): the split of
//!   the third bullet continued off the axis, the path of `t = sin²θ`
//!   moved through the branch point `1/m` ([`pi_split_off_axis`]), with the
//!   residue at `t = 1/n` when the move sweeps the pole.
//!
//! What none of them reaches is refused.  On the cuts the values are
//! mpmath's (and so SymPy's): the limits from below, `m − i0` and `n − i0`
//! (`ellippi(1/2, 2) = 1.5327 − 2.1041i`, `ellippi(2, 1/2) = −0.3135 −
//! 1.8138i`, `ellippi(e, i) = 0.3346 − 1.3484i`); `n₁` then lies on its cut
//! from above when `n > 1`, since `dn₁/dn < 0`.  Up to 0.54 every such `Π`
//! was refused.
//!
//! # Branch convention
//!
//! On the cut `m ∈ (1, ∞)` the value is the limit from below, `m − i0`, as
//! mpmath and SymPy take it: `K(2) = 1.3110287771 − 1.3110287771i`,
//! `E(2) = 0.5990701174 + 0.5990701174i`, `F(1 | 2) = 1.3110287771 −
//! 0.6571634186i` (`mpmath.ellipk`, `ellipe`, `ellipf`; the values at
//! `m = 2 − 10⁻²⁵i` agree to 25 digits).  In Carlson's form that is
//! `y + i0` for the argument `y = 1 − m sin²φ < 0`: the principal square
//! root of a negative real is `+i√|y|`, which is the first duplication
//! step's root of `y + iε`, and every later iterate lies in the upper half
//! plane, where the principal roots are continuous.
//!
//! # Accuracy
//!
//! The duplication is iterated until the arguments are within `2^−(wp/6 +
//! 2)` of their weighted mean `A`, where Carlson's series truncated after
//! degree 5 is within `2^−wp` of the value ([`series_ready`]: the degree-`N`
//! part is at most `(a)_N/N!·rᴺ` relative), at `wp = prec + 32` bits (the
//! real routines of `evalf.rs` likewise).  Up to 0.55 the duplication ran
//! until the arguments agreed to `2^−(wp/2 + 8)` and took `R(μ, μ, μ)`,
//! three times as many steps.  The error is relative to the modulus of the
//! value, so a part much smaller than the other (`E(m)` for `m` just above
//! 1 has an imaginary part of order `m − 1`), and the difference `R_F −
//! (m/3)R_D` of `E` near `m = 1`, are evaluated again with the bits they
//! lose.  Each part is then rounded to `prec` bits.
//!
//! The node's bound ([`bound`]), for every case these routines take, is
//! their own `2^−(prec+12)` relative to each part plus, for each inexact
//! argument, the change of the value with that argument moved by its radius
//! (`sensitivity::complex_route_bound`: the radius of the argument's real
//! part moves each part by its own share, since 0.59 also off the axis,
//! where `E(10⁶⁰⁰ + i) = 1.29·10⁻³⁰⁰ − 10³⁰⁰i` was refused, and a part far
//! below the other is computed up to [`MAX_PART_GAP`] bits below the
//! modulus); none when an argument's ball
//! reaches a cut with an undecided side: the parameter's `[1, ∞)`, `Π`'s
//! characteristic's `[1, ∞)`, the cut of `√(1 − m/n)` in [`pi_involution`],
//! and those of a non-real `φ`.  (Up to 0.55 a complex value at real
//! arguments was bounded by the rules of the real routines and the rounding
//! at the value's magnitude, which hid a part far below the other.)

use super::accuracy::{self, Bound};
use super::ball::Cache;
use super::wp::{RM, Wp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_abs, c_add, c_div, c_mul, c_sub};
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::truth::Truth;
use astro_float::BigFloat;

/// Does `f(args)` go to these routines?  `K`, `E` of a non-real parameter
/// or one above 1; `F` of a non-real `φ` or a non-real parameter, or of real
/// ones for which the real routine refuses (`1 − m sin²φ₀ ≤ 0`, or `m > 1`
/// with `|φ| > π/2`, which needs `K(m)`).
pub(super) fn takes(f: LibFn, args: &[ExprId], cache: Cache<'_>) -> bool {
    let value = |i: usize| args.get(i).and_then(|a| cache.get(a));
    match f {
        LibFn::EllipticK | LibFn::EllipticE => {
            value(0).is_some_and(|m| !m.1.is_zero() || above_one(&m.0))
        }
        LibFn::EllipticF => {
            let (Some(phi), Some(m)) = (value(0), value(1)) else {
                return false;
            };
            if !phi.1.is_zero() {
                return true;
            }
            !m.1.is_zero() || (above_one(&m.0) && !real_f_applies(&phi.0, &m.0))
        }
        // `n > 1` or `m > 1` (`n = 1`, `m = 1`: the real routine's infinity),
        // or a non-real argument.
        LibFn::EllipticPi => {
            let (Some(n), Some(m)) = (value(0), value(1)) else {
                return false;
            };
            !n.1.is_zero() || !m.1.is_zero() || above_one(&n.0) || above_one(&m.0)
        }
        _ => false,
    }
}

/// `x > 1`.
fn above_one(x: &BigFloat) -> bool {
    let one = BigFloat::from_i32(1, 64);
    x.cmp(&one).is_some_and(|c| c > 0)
}

/// Is `F(φ | m)` (real `φ`, real `m > 1`) left to the real routine: `|φ| ≤
/// π/2` after no reduction, with `1 − m sin²φ > 0`?
fn real_f_applies(phi: &BigFloat, m: &BigFloat) -> bool {
    let p = 128;
    let mut cc = match astro_float::Consts::new() {
        Ok(cc) => cc,
        Err(_) => return false,
    };
    let half_pi = cc.pi(p, RM).div(&BigFloat::from_i32(2, p), p, RM);
    if phi.abs().cmp(&half_pi).is_none_or(|c| c > 0) {
        return false;
    }
    let s = phi.sin(p, RM, &mut cc);
    let y = BigFloat::from_i32(1, p).sub(&m.mul(&s.mul(&s, p, RM), p, RM), p, RM);
    // Well inside: a `y` near 0 goes to the complex routine, which gives
    // the real value there too.
    crate::base::bigfloat::is_strictly_positive(&y) && y.exponent().is_some_and(|e| e > -100)
}

/// The value of `f(args)` (see [`takes`]), each part rounded to `prec` bits.
pub(super) fn eval(
    f: LibFn,
    args: &[ExprId],
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let arg = |i: usize| -> Result<&Complex, SymplexError> {
        args.get(i)
            .and_then(|a| cache.get(a))
            .ok_or_else(|| super::unevaluable("elliptic integral lost its argument"))
    };
    // An argument that is infinite or undefined (often only to the working
    // precision: `1/(sin(1 + 10⁻⁶⁰) − sin 1)` is `+∞` at 16 digits) is a
    // domain error the precision search can retry, as in the real routines;
    // the duplication would run on `nan` and report non-convergence.
    for i in 0..args.len() {
        let a = arg(i)?;
        if [&a.0, &a.1].iter().any(|x| x.is_nan() || x.is_inf()) {
            return Err(super::unevaluable(format!(
                "{} of special float value",
                f.name()
            )));
        }
    }
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        // `K`, `E` of a real `m > 1`: each part by real integrals, to its
        // own precision.
        let real_beyond = matches!(f, LibFn::EllipticK | LibFn::EllipticE)
            && arg(0).is_ok_and(|m| m.1.is_zero() && above_one(&m.0));
        let (v, lost, own_parts) = match f {
            LibFn::EllipticK | LibFn::EllipticE if real_beyond => {
                let (v, lost) = beyond_one(&arg(0)?.0, f == LibFn::EllipticE, wp)?;
                (v, lost, true)
            }
            LibFn::EllipticK => (complete_k(arg(0)?, wp)?, 0, false),
            LibFn::EllipticE => {
                let (v, lost) = complete_e(arg(0)?, wp)?;
                (v, lost, false)
            }
            LibFn::EllipticF => {
                let (v, lost) = incomplete_f(arg(0)?, arg(1)?, &mut w.at(wp))?;
                (v, lost, false)
            }
            LibFn::EllipticPi => {
                let (v, lost) = complete_pi(arg(0)?, arg(1)?, &mut w.at(wp))?;
                (v, lost, false)
            }
            _ => {
                return Err(super::unevaluable(
                    "not a complete or incomplete elliptic integral",
                ));
            }
        };
        // Bits a part loses next to the modulus (the error is relative to
        // it, except for a route that computes each part on its own), and
        // those of a cancelling difference.
        let gap = if own_parts { 0 } else { part_gap(&v) };
        let need = lost + gap + 16;
        if need <= extra {
            return Ok((super::round_to(v.0, prec), super::round_to(v.1, prec)));
        }
        if need > (4 * prec + 512).max(MAX_PART_GAP) {
            break;
        }
        extra = need + 16;
    }
    Err(super::unevaluable(format!(
        "{}: the value cancels beyond the working precision",
        f.name()
    )))
}

/// The most bits [`eval`] adds to the precision for a part far below the
/// other, whatever the precision asked for: off the axis the routes compute
/// both parts relative to the modulus, and a part `2^g` below it needs `g`
/// bits more (`E(10⁶⁰⁰ + i) = 1.29·10⁻³⁰⁰ − 10³⁰⁰i`, 2,000 bits; up to 0.58
/// it was refused beyond `4·prec + 512`).  The imaginary part of `(R_F −
/// R_D/3)(0, 1/m, 1)` in the split of `E(m)` at `t = 1/m` cancels the same
/// bits, so no route avoids them.
const MAX_PART_GAP: usize = 4096;

/// How many bits below the modulus of `v` its smaller nonzero part lies.
fn part_gap(v: &Complex) -> usize {
    let (Some(a), Some(b)) = (accuracy::part_mag(&v.0), accuracy::part_mag(&v.1)) else {
        return 0;
    };
    usize::try_from((a - b).abs()).unwrap_or(usize::MAX / 8)
}

/// The principal square root (`+i√|y|` for a negative real `y`, the
/// convention of the module documentation).
fn c_sqrt(z: &Complex, wp: usize) -> Complex {
    let (a, b) = (&z.0, &z.1);
    if b.is_zero() {
        if a.is_zero() {
            return (BigFloat::new(wp), BigFloat::new(wp));
        }
        if a.is_negative() {
            return (BigFloat::new(wp), a.neg().sqrt(wp, RM));
        }
        return (a.sqrt(wp, RM), BigFloat::new(wp));
    }
    let r = c_abs(z, wp, RM);
    let two = BigFloat::from_i32(2, wp);
    if !a.is_negative() {
        let s = r.add(a, wp, RM).div(&two, wp, RM).sqrt(wp, RM);
        let t = b.div(&s.mul(&two, wp, RM), wp, RM);
        (s, t)
    } else {
        let t = r.sub(a, wp, RM).div(&two, wp, RM).sqrt(wp, RM);
        let s = b.abs().div(&t.mul(&two, wp, RM), wp, RM);
        (s, if b.is_negative() { t.neg() } else { t })
    }
}

/// `c` as a complex number.
fn real(c: i32, wp: usize) -> Complex {
    (BigFloat::from_i32(c, wp), BigFloat::new(wp))
}

/// Are the deviations `Wᵢ = 1 − vᵢ/A` of the arguments from their weighted
/// mean `A` small enough for Carlson's series truncated after degree 5 to
/// be within `2^−wp` of the value (relative)?  The degree-`N` part of
/// `R_{−a}(b; A(1 − W)) = A^{−a}·Σ (a)_N/(c)_N·T_N(b, W)` (DLMF 19.19.7,
/// `c = Σ bᵢ`) is at most `A^{−a}·(a)_N/N!·rᴺ` with `r = max |Wᵢ|`, since
/// `|T_N| ≤ (c)_N·rᴺ/N!`; for `a ≤ 3/2` the parts of degree 6 and more sum
/// to less than `3r⁶/(1 − 2r)`, below `2^−wp` once `r ≤ 2^−(wp/6 + 2)`.
/// Up to 0.55 the duplication ran until the spread was `2^−(wp/2 + 8)`,
/// three times as many steps.
pub(super) fn series_ready(devs: &[&Complex], wp: usize) -> bool {
    let k = i64::try_from(wp / 6 + 2).unwrap_or(i64::MAX / 4);
    // `|W| < 2^(mag + 1)` (a part below `2^mag`): `mag + 1 ≤ −k`.
    devs.iter().all(|d| accuracy::mag(d).is_none_or(|e| e < -k))
}

/// The deviations `1 − v/A` (`None` for `A = 0`, where the arguments are
/// still far apart).
fn deviations(vals: &[&Complex], a: &Complex, wp: usize) -> Option<Vec<Complex>> {
    if a.0.is_zero() && a.1.is_zero() {
        return None;
    }
    let one = real(1, wp);
    Some(
        vals.iter()
            .map(|v| c_sub(&one, &c_div(v, a, wp, RM), wp, RM))
            .collect(),
    )
}

/// The elementary symmetric functions `E₂ … E₅` of `ws` (`E₁ = 0` here).
fn elementary(ws: &[&Complex], wp: usize) -> [Complex; 6] {
    let mut e: [Complex; 6] = std::array::from_fn(|_| real(0, wp));
    e[0] = real(1, wp);
    for w in ws {
        for k in (1..6).rev() {
            let t = c_mul(w, &e[k - 1], wp, RM);
            e[k] = c_add(&e[k], &t, wp, RM);
        }
    }
    e
}

/// `Σ (num/den)·t` over the terms.
fn rational_sum(terms: &[(i32, i32, &Complex)], wp: usize) -> Complex {
    let mut acc = real(0, wp);
    for (num, den, t) in terms {
        let q = BigFloat::from_i32(*num, wp).div(&BigFloat::from_i32(*den, wp), wp, RM);
        acc = c_add(&acc, &(t.0.mul(&q, wp, RM), t.1.mul(&q, wp, RM)), wp, RM);
    }
    acc
}

/// Carlson's series of `R_F` truncated after degree 5 (DLMF 19.36.1): `1 −
/// E₂/10 + E₃/14 + E₂²/24 − 3E₂E₃/44` for the deviations `X`, `Y`, `Z` of
/// `x`, `y`, `z` from `A = (x + y + z)/3`; `R_F ≈ A^{−1/2}` times it.
pub(super) fn rf_series(devs: &[&Complex], wp: usize) -> Complex {
    let e = elementary(devs, wp);
    let e22 = c_mul(&e[2], &e[2], wp, RM);
    let e23 = c_mul(&e[2], &e[3], wp, RM);
    let one = real(1, wp);
    rational_sum(
        &[
            (1, 1, &one),
            (-1, 10, &e[2]),
            (1, 14, &e[3]),
            (1, 24, &e22),
            (-3, 44, &e23),
        ],
        wp,
    )
}

/// Carlson's series of `R_D`, `R_J` truncated after degree 5 (DLMF 19.36.2):
/// `1 − 3E₂/14 + E₃/6 + 9E₂²/88 − 3E₄/22 − 9E₂E₃/52 + 3E₅/26` for the
/// deviations of `(x, y, z, p, p)` from `A = (x + y + z + 2p)/5` (`p = z`
/// for `R_D`); the remainder times `A^{−3/2}`.
pub(super) fn rj_series(devs: &[&Complex], wp: usize) -> Complex {
    let e = elementary(devs, wp);
    let e22 = c_mul(&e[2], &e[2], wp, RM);
    let e23 = c_mul(&e[2], &e[3], wp, RM);
    let one = real(1, wp);
    rational_sum(
        &[
            (1, 1, &one),
            (-3, 14, &e[2]),
            (1, 6, &e[3]),
            (9, 88, &e22),
            (-3, 22, &e[4]),
            (-9, 52, &e23),
            (3, 26, &e[5]),
        ],
        wp,
    )
}

/// `λ = √x√y + √y√z + √z√x` and the roots.
fn lambda(x: &Complex, y: &Complex, z: &Complex, wp: usize) -> (Complex, Complex) {
    let (sx, sy, sz) = (c_sqrt(x, wp), c_sqrt(y, wp), c_sqrt(z, wp));
    let l = c_add(
        &c_add(&c_mul(&sx, &sy, wp, RM), &c_mul(&sy, &sz, wp, RM), wp, RM),
        &c_mul(&sz, &sx, wp, RM),
        wp,
        RM,
    );
    (l, sz)
}

/// `(v + λ)/4`.
fn step(v: &Complex, l: &Complex, wp: usize) -> Complex {
    let four = real(4, wp);
    c_div(&c_add(v, l, wp, RM), &four, wp, RM)
}

/// Carlson's `R_F(x, y, z)` over ℂ (at most one argument 0).
fn rf(x: &Complex, y: &Complex, z: &Complex, wp: usize) -> Result<Complex, SymplexError> {
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    for _ in 0..(wp * 2 + 200) {
        let mu = c_div(
            &c_add(&c_add(&x, &y, wp, RM), &z, wp, RM),
            &real(3, wp),
            wp,
            RM,
        );
        if let Some(d) = deviations(&[&x, &y], &mu, wp) {
            let dz = (
                d[0].0.add(&d[1].0, wp, RM).neg(),
                d[0].1.add(&d[1].1, wp, RM).neg(),
            );
            let devs = [&d[0], &d[1], &dz];
            if series_ready(&devs, wp) {
                let s = rf_series(&devs, wp);
                return Ok(c_div(&s, &c_sqrt(&mu, wp), wp, RM));
            }
        }
        let (l, _) = lambda(&x, &y, &z, wp);
        x = step(&x, &l, wp);
        y = step(&y, &l, wp);
        z = step(&z, &l, wp);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_F did not converge".into(),
    })
}

/// Carlson's `R_D(x, y, z)` over ℂ: `R_D = R_D(x′, y′, z′)/4 + 3/(√z (z + λ))`.
fn rd(x: &Complex, y: &Complex, z: &Complex, wp: usize) -> Result<Complex, SymplexError> {
    let (mut x, mut y, mut z) = (x.clone(), y.clone(), z.clone());
    let mut acc = real(0, wp);
    let mut scale = real(1, wp);
    let quarter = c_div(&real(1, wp), &real(4, wp), wp, RM);
    for _ in 0..(wp * 2 + 200) {
        // A = (x + y + 3z)/5 and the series of the remainder.
        let three_z = c_mul(&real(3, wp), &z, wp, RM);
        let mu = c_div(
            &c_add(&c_add(&x, &y, wp, RM), &three_z, wp, RM),
            &real(5, wp),
            wp,
            RM,
        );
        if let Some(d) = deviations(&[&x, &y, &z], &mu, wp) {
            let devs = [&d[0], &d[1], &d[2], &d[2], &d[2]];
            if series_ready(&devs, wp) {
                let s = rj_series(&devs, wp);
                let tail = c_div(&s, &c_mul(&mu, &c_sqrt(&mu, wp), wp, RM), wp, RM);
                return Ok(c_add(&acc, &c_mul(&scale, &tail, wp, RM), wp, RM));
            }
        }
        let (l, sz) = lambda(&x, &y, &z, wp);
        let term = c_div(
            &real(3, wp),
            &c_mul(&sz, &c_add(&z, &l, wp, RM), wp, RM),
            wp,
            RM,
        );
        acc = c_add(&acc, &c_mul(&scale, &term, wp, RM), wp, RM);
        scale = c_mul(&scale, &quarter, wp, RM);
        x = step(&x, &l, wp);
        y = step(&y, &l, wp);
        z = step(&z, &l, wp);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_D did not converge".into(),
    })
}

/// `1 − m`, refusing `m = 1` (where `K` is infinite).
fn one_minus(m: &Complex, wp: usize, what: &str) -> Result<Complex, SymplexError> {
    let y = c_sub(&real(1, wp), m, wp, RM);
    if y.0.is_zero() && y.1.is_zero() {
        return Err(super::infinite(format!("{what}(1) is infinite")));
    }
    Ok(y)
}

/// `K(m)` (`second`: `E(m)`) for a real `m > 1`, the limit from below, by
/// the reciprocal-modulus transformation (DLMF 19.7.3) with `μ = 1/m`:
///
/// ```text
/// K(m) = √μ·R_F(0, 1 − μ, 1) − i·√μ·R_F(0, μ, 1)
/// E(m) = √μ·(R_F − R_D/3)(0, 1 − μ, 1) + i·((1 − μ)/√μ)·(R_F − R_D/3)(0, μ, 1)
/// ```
///
/// (`K(m) = √μ(K(μ) − iK(1 − μ))`, `E(m) = (E(μ) + iE(1 − μ) − (1 − μ)K(μ)
/// − iμK(1 − μ))/√μ`, the combinations written without their cancelling
/// terms), each part by the real Carlson routines to its own precision;
/// with the bits the differences `R_F − R_D/3` cancel (logarithmically, near
/// `μ = 0` and `μ = 1`).  Up to 0.55 these values came from the duplication
/// over ℂ from `1 − m < 0`: `E(10⁶⁰⁰) = 7.85·10⁻³⁰¹ + 10³⁰⁰i` was refused, its
/// real part 2,000 bits below the modulus.
fn beyond_one(m: &BigFloat, second: bool, wp: usize) -> Result<(Complex, usize), SymplexError> {
    let one = BigFloat::from_i32(1, wp);
    let mu = one.div(m, wp, RM);
    let mu1 = one.sub(&mu, wp, RM);
    let root = mu.sqrt(wp, RM);
    let zero = BigFloat::new(wp);
    let rf_at = |y: &BigFloat| super::carlson_rf(&zero, y, &one, wp);
    if !second {
        let re = root.mul(&rf_at(&mu1)?, wp, RM);
        let im = root.mul(&rf_at(&mu)?, wp, RM).neg();
        return Ok(((re, im), 0));
    }
    // (R_F − R_D/3)(0, y, 1) and the bits it cancels.
    let difference = |y: &BigFloat| -> Result<(BigFloat, usize), SymplexError> {
        let f = rf_at(y)?;
        let d = super::carlson_rd(&zero, y, &one, wp)?.div(&BigFloat::from_i32(3, wp), wp, RM);
        let v = f.sub(&d, wp, RM);
        let lost = lost_bits(
            &[&(f, zero.clone()), &(d, zero.clone())],
            &(v.clone(), zero.clone()),
        );
        Ok((v, lost))
    };
    let (a, la) = difference(&mu1)?;
    let (b, lb) = difference(&mu)?;
    let re = root.mul(&a, wp, RM);
    let im = mu1.mul(&b, wp, RM).div(&root, wp, RM);
    Ok(((re, im), la.max(lb)))
}

/// `K(m) = R_F(0, 1 − m, 1)`.
fn complete_k(m: &Complex, wp: usize) -> Result<Complex, SymplexError> {
    let y = one_minus(m, wp, "elliptic_k")?;
    rf(&real(0, wp), &y, &real(1, wp), wp)
}

/// `E(m) = R_F(0, 1 − m, 1) − (m/3)·R_D(0, 1 − m, 1)`, and the bits the
/// difference cancels.
fn complete_e(m: &Complex, wp: usize) -> Result<(Complex, usize), SymplexError> {
    let y = c_sub(&real(1, wp), m, wp, RM);
    if y.0.is_zero() && y.1.is_zero() {
        return Ok((real(1, wp), 0));
    }
    let (zero, one) = (real(0, wp), real(1, wp));
    let f = rf(&zero, &y, &one, wp)?;
    let d = rd(&zero, &y, &one, wp)?;
    let t = c_div(&c_mul(m, &d, wp, RM), &real(3, wp), wp, RM);
    let v = c_sub(&f, &t, wp, RM);
    Ok((v.clone(), lost_bits(&[&f, &t], &v)))
}

/// Bits lost forming `v` from the terms `parts`.
fn lost_bits(parts: &[&Complex], v: &Complex) -> usize {
    let top = parts.iter().filter_map(|p| accuracy::mag(p)).max();
    match (top, accuracy::mag(v)) {
        (Some(t), Some(m)) => usize::try_from(t - m).unwrap_or(0),
        (Some(_), None) => usize::MAX / 8,
        _ => 0,
    }
}

/// `F(φ | m)`: `sin φ₀ · R_F(cos²φ₀, 1 − m sin²φ₀, 1) + 2k·K(m)` with `φ =
/// φ₀ + kπ`, `|Re φ₀| ≤ π/2` (mpmath's reduction), and the bits the sum
/// cancels.
fn incomplete_f(
    phi: &Complex,
    m: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    if !phi.1.is_zero() {
        return incomplete_f_complex(phi, m, w);
    }
    let phi = &phi.0;
    let wp = w.bits();
    if phi.is_zero() {
        return Ok((real(0, wp), 0));
    }
    if phi.is_nan() || phi.is_inf() {
        return Err(super::unevaluable("elliptic_f of special float value"));
    }
    // Reduce with the bits of φ's integer multiple of π.
    let k_bits = phi
        .exponent()
        .map_or(0, |e| usize::try_from(e.max(0)).unwrap_or(0));
    let wr = wp + k_bits;
    let pi = w.cc.pi(wr, RM).clone();
    let k = phi
        .div(&pi, wr, RM)
        .add(&BigFloat::from_f64(0.5, wr), wr, RM)
        .floor();
    let phi0 = phi.sub(&k.mul(&pi, wr, RM), wr, RM);
    let s = phi0.sin(wp, RM, w.cc);
    let c = phi0.cos(wp, RM, w.cc);
    let s2 = s.mul(&s, wp, RM);
    let c2 = (c.mul(&c, wp, RM), BigFloat::new(wp));
    let ms2 = c_mul(m, &(s2, BigFloat::new(wp)), wp, RM);
    let y = c_sub(&real(1, wp), &ms2, wp, RM);
    let r = rf(&c2, &y, &real(1, wp), wp)?;
    let head = c_mul(&(s, BigFloat::new(wp)), &r, wp, RM);
    if k.is_zero() {
        return Ok((head, 0));
    }
    let kk = complete_k(m, wp)?;
    let two_k = (k.mul(&BigFloat::from_i32(2, wp), wp, RM), BigFloat::new(wp));
    let tail = c_mul(&two_k, &kk, wp, RM);
    let v = c_add(&head, &tail, wp, RM);
    Ok((v.clone(), lost_bits(&[&head, &tail], &v)))
}

/// The integer `k` nearest to `Re φ/π` and `φ₀ = φ − kπ`, with the bits of
/// `kπ` as guard.
fn reduce_phi(phi: &Complex, w: &mut Wp<'_>) -> (BigFloat, Complex) {
    let wp = w.bits();
    let k_bits = phi
        .0
        .exponent()
        .map_or(0, |e| usize::try_from(e.max(0)).unwrap_or(0));
    let wr = wp + k_bits;
    let pi = w.cc.pi(wr, RM).clone();
    let k = phi
        .0
        .div(&pi, wr, RM)
        .add(&BigFloat::from_f64(0.5, wr), wr, RM)
        .floor();
    let re0 = phi.0.sub(&k.mul(&pi, wr, RM), wr, RM);
    (k, (super::round_to(re0, wp), phi.1.clone()))
}

/// `F(φ | m)` for a non-real `φ`, by the formula of [`incomplete_f`] with the
/// complex `sin φ₀`, `cos φ₀` (mpmath's `ellipf`, and so SymPy's): `R_F` of
/// arguments off `(−∞, 0]` by the duplication over ℂ.  `cos²φ₀` lies off
/// `(−∞, 0]` for `|Re φ₀| < π/2` (`cos φ₀` has a positive real part); the
/// lines `Re φ = (k + ½)π` and the curves where `1 − m sin²φ₀ < 0` are this
/// convention's cuts, and a ball that reaches one has no bound ([`bound`]).
/// Refused before 0.56.
fn incomplete_f_complex(
    phi: &Complex,
    m: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let (k, phi0) = reduce_phi(phi, w);
    let s = super::c_sin(&phi0, w);
    let c = super::c_cos(&phi0, w);
    let c2 = c_mul(&c, &c, wp, RM);
    let y = c_sub(
        &real(1, wp),
        &c_mul(m, &c_mul(&s, &s, wp, RM), wp, RM),
        wp,
        RM,
    );
    if [&s.0, &s.1, &c2.0, &c2.1, &y.0, &y.1]
        .iter()
        .any(|x| x.is_nan() || x.is_inf())
    {
        return Err(super::unevaluable(
            "elliptic_f: the amplitude's imaginary part is beyond the exponent range",
        ));
    }
    if (c2.0.is_zero() && c2.1.is_zero()) || (y.0.is_zero() && y.1.is_zero()) {
        return Err(super::unevaluable(
            "elliptic_f: a singular point of the integrand",
        ));
    }
    let r = rf(&c2, &y, &real(1, wp), wp)?;
    let head = c_mul(&s, &r, wp, RM);
    if k.is_zero() {
        return Ok((head, 0));
    }
    let kk = complete_k(m, wp)?;
    let two_k = (k.mul(&BigFloat::from_i32(2, wp), wp, RM), BigFloat::new(wp));
    let tail = c_mul(&two_k, &kk, wp, RM);
    let v = c_add(&head, &tail, wp, RM);
    Ok((v.clone(), lost_bits(&[&head, &tail], &v)))
}

/// `x = 1`.
fn is_one(x: &BigFloat) -> bool {
    x.cmp(&BigFloat::from_i32(1, 64)) == Some(0)
}

/// `Π(n | m)` (see the module documentation), and the bits its sums cancel.
fn complete_pi(n: &Complex, m: &Complex, w: &mut Wp<'_>) -> Result<(Complex, usize), SymplexError> {
    if [&n.0, &n.1, &m.0, &m.1]
        .iter()
        .any(|x| x.is_nan() || x.is_inf())
    {
        return Err(super::unevaluable("elliptic_pi of special float value"));
    }
    pi_value(n, m, false, w)
}

/// `Π(n | m)` for the arguments [`takes`] sends here and those of the real
/// routine (`n < 1`, `m < 1`, real); `above`: a real `n > 1` lies on its cut
/// from above (`n + i0`; `n − i0` otherwise).
fn pi_value(
    n: &Complex,
    m: &Complex,
    above: bool,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let (n_real, m_real) = (n.1.is_zero(), m.1.is_zero());
    if m_real && is_one(&m.0) {
        return Err(super::infinite("elliptic_pi(n, 1) is infinite"));
    }
    if n_real && is_one(&n.0) {
        return Err(super::infinite("elliptic_pi(1, m) is infinite"));
    }
    if m_real && above_one(&m.0) {
        return pi_split(n, &m.0, w);
    }
    if n_real && m_real {
        if above_one(&n.0) {
            return pi_beyond_pole(&n.0, &m.0, above, w);
        }
        let v = super::arb_elliptic_pi(&n.0, &m.0, w)?;
        return Ok(((v, BigFloat::new(wp)), 0));
    }
    if uses_off_axis_split(n, m, w.bits()) {
        return pi_split_off_axis(n, m, above, w);
    }
    if uses_involution(n, m, w.bits()) {
        return pi_involution(n, m, above, w);
    }
    pi_principal(n, m, w)
}

/// Does [`pi_value`] take `Π(n | m)` through [`pi_split_off_axis`]: a
/// non-real `m` with `Re m > 1`, where neither `1 − m`, 1, `1 − n` nor the
/// arguments `1 − m`, 1, `1 − m/n` of [`pi_involution`]'s `Π(m/n | m)` lie in
/// a half-plane (so that no other route reaches it)?  (`Re m > 1` keeps the
/// parameters `1/m`, `1 − 1/m` of the split's parts below 1 in real part,
/// where this route is not taken again.)
fn uses_off_axis_split(n: &Complex, m: &Complex, wp: usize) -> bool {
    if m.1.is_zero() || !above_one(&m.0) || !needs_involution(n, m, wp) {
        return false;
    }
    let y = c_sub(&real(1, wp), m, wp, RM);
    let t = c_sub(&real(1, wp), &c_div(m, n, wp, RM), wp, RM);
    rotation(&[&y, &t], wp).is_none()
}

/// `Π(n | m)` for a non-real `m` ([`uses_off_axis_split`]): the split of
/// [`pi_split`] continued off the real axis,
///
/// ```text
/// Π(n | m) = Π(n/m | 1/m)/√m + √(1 − 1/m)/((1 − n)·√(1 − m))·Π(n₁ | 1 − 1/m) + σ·2πi·Res,
/// n₁ = n(m − 1)/(m(n − 1)),
/// ```
///
/// every root principal.  With `t = sin²θ`, `Π(n | m) = ½∫₀¹ dt/((1 − nt)√t
/// √(1 − t)√(1 − mt))` along `[0, 1]`; the path is moved to `0 → t₀ → 1`
/// through the branch point `t₀ = 1/m` (off the real axis): the cut `t₀·[1,
/// ∞)` of `√(1 − mt)` leaves the triangle `0, 1, t₀` at its vertex and
/// those of `√t`, `√(1 − t)` meet it only at 0 and 1, so the only
/// singularity the move may sweep is the pole `t = 1/n`.  On `[0, t₀]`, `t =
/// t₀s` gives the first part (`1 − mt = 1 − s`); on `[t₀, 1]`, `t = 1 − (1
/// − 1/m)r` gives `1 − nt = (1 − n)(1 − n₁r)`, `1 − mt = (1 − m)(1 − r)`
/// and the second.  A pole `1/n` inside the triangle adds its residue,
/// `Res = −1/(2n·√P·√(1 − P)·√(1 − m/n))` at `P = 1/n` (the principal
/// branches are analytic inside), with `σ = sign(Im t₀)` (the boundary `0 →
/// 1 → t₀` runs counterclockwise for `Im t₀ > 0`); a pole on or near its
/// boundary (`n/m` or `n₁` on the cut `[1, ∞)` of its part) is refused.  A
/// real `n > 1` puts the pole on `[0, 1]` itself: the value is the limit
/// `n ∓ i0` (`above`: `n + i0`), whose pole lies just above (below) the
/// segment, so inside the triangle exactly when `t₀` is on that side.  For
/// a real `m > 1` approached from below it is [`pi_split`]'s identity.
/// mpmath (and a quadrature along `[0, π/2]`): `Π(½ − i/3 | 4 + i/10) =
/// 1.33515537350355 + 1.30525487404705i`, `Π(2 + i | 2 − i/1000) =
/// 2.78305593932435 + 2.02073751897057i`.  Refused up to 0.58 ("the
/// arguments 1 - m, 1 and 1 - n lie in no half-plane").
fn pi_split_off_axis(
    n: &Complex,
    m: &Complex,
    above: bool,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let one = real(1, wp);
    let t0 = c_div(&one, m, wp, RM);
    let on_path = n.1.is_zero() && above_one(&n.0);
    let inside = if on_path {
        t0.1.is_negative() == above
    } else {
        match pole_in_triangle(n, &t0, wp) {
            Truth::True => true,
            Truth::False => false,
            Truth::Unknown => {
                return Err(super::unevaluable(
                    "elliptic_pi: the pole 1/n is on the boundary of the moved path",
                ));
            }
        }
    };
    let m1 = c_sub(&one, &t0, wp, RM);
    let n_over_m = c_mul(n, &t0, wp, RM);
    let n1 = c_div(&c_mul(n, &m1, wp, RM), &c_sub(n, &one, wp, RM), wp, RM);
    let (p1, lost1) = pi_value(&n_over_m, &t0, false, w)?;
    let (p2, lost2) = pi_value(&n1, &m1, false, w)?;
    let a = c_mul(&c_sqrt(&t0, wp), &p1, wp, RM);
    let den = c_mul(
        &c_sub(&one, n, wp, RM),
        &c_sqrt(&c_sub(&one, m, wp, RM), wp),
        wp,
        RM,
    );
    let b = c_mul(&c_div(&c_sqrt(&m1, wp), &den, wp, RM), &p2, wp, RM);
    let mut parts = vec![a, b];
    if inside {
        // σ·2πi·Res = ∓πi/(n·√P·√(1 − P)·√(1 − m/n)), P = 1/n.
        let p = c_div(&one, n, wp, RM);
        let roots = c_mul(
            &c_mul(
                &c_sqrt(&p, wp),
                &c_sqrt(&c_sub(&one, &p, wp, RM), wp),
                wp,
                RM,
            ),
            &c_sqrt(&c_sub(&one, &c_mul(m, &p, wp, RM), wp, RM), wp),
            wp,
            RM,
        );
        let pi = w.cc.pi(wp, RM).clone();
        let pi_i = if t0.1.is_negative() {
            (BigFloat::new(wp), pi)
        } else {
            (BigFloat::new(wp), pi.neg())
        };
        parts.push(c_div(&pi_i, &c_mul(n, &roots, wp, RM), wp, RM));
    }
    let v = parts.iter().fold(real(0, wp), |s, t| c_add(&s, t, wp, RM));
    let refs: Vec<&Complex> = parts.iter().collect();
    let lost = lost1.max(lost2).max(lost_bits(&refs, &v));
    Ok((v, lost))
}

/// The barycentric coordinates `(α, β, γ)` of `P = 1/n` in the triangle
/// `1, t₀, 0` (`P = α + βt₀`, `γ = 1 − α − β`) for a non-real `t₀`.
fn barycentric(n: &Complex, t0: &Complex, wp: usize) -> [BigFloat; 3] {
    let p = c_div(&real(1, wp), n, wp, RM);
    let beta = p.1.div(&t0.1, wp, RM);
    let alpha = p.0.sub(&beta.mul(&t0.0, wp, RM), wp, RM);
    let gamma = BigFloat::from_i32(1, wp)
        .sub(&alpha, wp, RM)
        .sub(&beta, wp, RM);
    [alpha, beta, gamma]
}

/// Is the pole `1/n` inside the triangle `0, 1, t₀` (`True`), outside it
/// (`False`), or within `2^−(wp/2)` of its boundary in a barycentric
/// coordinate (`Unknown`)?
fn pole_in_triangle(n: &Complex, t0: &Complex, wp: usize) -> Truth {
    let coords = barycentric(n, t0, wp);
    let scale = coords
        .iter()
        .filter_map(accuracy::part_mag)
        .max()
        .unwrap_or(0)
        .max(0);
    let limit = scale - i64::try_from(wp / 2).unwrap_or(i64::MAX / 4);
    let clear = |c: &BigFloat| accuracy::part_mag(c).is_some_and(|e| e > limit);
    if coords.iter().any(|c| c.is_negative() && clear(c)) {
        return Truth::False;
    }
    if coords.iter().all(|c| !c.is_negative() && clear(c)) {
        Truth::True
    } else {
        Truth::Unknown
    }
}

/// Does [`pi_value`] take `Π(n | m)` through [`pi_involution`]: a non-real
/// `m` with a real `n > 1`, or arguments in no half-plane, but neither a
/// real `m ≥ 1` ([`pi_split`]) nor real `n` and `m`?
fn uses_involution(n: &Complex, m: &Complex, wp: usize) -> bool {
    let (n_real, m_real) = (n.1.is_zero(), m.1.is_zero());
    if m_real && (is_one(&m.0) || above_one(&m.0)) {
        return false;
    }
    if n_real && (m_real || is_one(&n.0)) {
        return false;
    }
    (n_real && above_one(&n.0)) || needs_involution(n, m, wp)
}

/// Do `1 − m`, 1 and `1 − n` lie in no common half-plane ([`rotation`]), so
/// that `Π(n | m)` is taken through [`pi_involution`]?
fn needs_involution(n: &Complex, m: &Complex, wp: usize) -> bool {
    let y = c_sub(&real(1, wp), m, wp, RM);
    let p = c_sub(&real(1, wp), n, wp, RM);
    let zero = |z: &Complex| z.0.is_zero() && z.1.is_zero();
    !zero(&y) && !zero(&p) && !zero(n) && rotation(&[&y, &p], wp).is_none()
}

/// `Π(n | m) = K(m) − Π(m/n | m) + (π/2)/(√(1 − n)·√(1 − m/n))` (DLMF 19.7.8
/// with the two principal roots taken apart), for a non-real `m` with a
/// real `n > 1` (`√(1 − n) = ±i√(n − 1)` for `n ∓ i0`; `above`: `n + i0`),
/// and for arguments that lie in no half-plane with 1 (`Π(m/n | m)` then
/// does, by the rotation).  The sum `h(n) = Π(n | m) + Π(m/n | m) − K(m)` is
/// analytic in `n` off the cuts `[1, ∞)` of `Π(n | ·)` and `(0, m]` of
/// `Π(m/n | ·)`, with `h² = (π/2)²n/((1 − n)(n − m))` (DLMF 19.7.8 for real
/// `0 < m < 1`, `n < 0`, continued); `1/(√(1 − n)√(1 − m/n))` is a branch
/// of that root analytic on the same domain and positive there, so it is
/// `h/(π/2)` for every complex `m ∉ [1, ∞)`.  (With the single root
/// `√((1 − n)(1 − m/n))` the identity fails where the product crosses the
/// negative axis.)  Values: `Π(e | i) = 0.3345818233 − 1.348376093i` (n − i0;
/// mpmath, and a quadrature below the pole, at 30 and 50 digits), `Π(½ −
/// i/3 | 2 + i) = 1.995339437 + 0.9025251621i`.  Refused before 0.56.
fn pi_involution(
    n: &Complex,
    m: &Complex,
    above: bool,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    if n.0.is_zero() && n.1.is_zero() {
        return Err(super::unevaluable(
            "elliptic_pi: no transformation at n = 0",
        ));
    }
    let one = real(1, wp);
    let q = c_div(m, n, wp, RM);
    let s1 = if n.1.is_zero() && above_one(&n.0) {
        let r = n.0.sub(&one.0, wp, RM).sqrt(wp, RM);
        let zero = BigFloat::new(wp);
        if above { (zero, r.neg()) } else { (zero, r) }
    } else {
        c_sqrt(&c_sub(&one, n, wp, RM), wp)
    };
    let s2 = c_sqrt(&c_sub(&one, &q, wp, RM), wp);
    let half_pi = (
        w.cc.pi(wp, RM).div(&BigFloat::from_i32(2, wp), wp, RM),
        BigFloat::new(wp),
    );
    let term = c_div(&half_pi, &c_mul(&s1, &s2, wp, RM), wp, RM);
    let k = complete_k(m, wp)?;
    let (p, lost_p) = pi_principal(&q, m, w)?;
    let v = c_add(&c_sub(&k, &p, wp, RM), &term, wp, RM);
    let lost = lost_p.max(lost_bits(&[&k, &p, &term], &v));
    Ok((v, lost))
}

/// `Π(n | m)` for a real `n > 1` and a real `m < 1`: `K(m) − Π(m/n | m) ∓
/// i·(π/2)·√(n/((n − 1)(n − m)))`, `−` for `n − i0`, `+` for `n + i0`
/// (`above`).  The real part is the Cauchy principal value (DLMF 19.6.5),
/// the imaginary part half the residue at the pole `sin²θ = 1/n`.
fn pi_beyond_pole(
    n: &BigFloat,
    m: &BigFloat,
    above: bool,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let k = super::arb_elliptic_k(m, w)?;
    let p = super::arb_elliptic_pi(&m.div(n, wp, RM), m, w)?;
    let re = k.sub(&p, wp, RM);
    let one = BigFloat::from_i32(1, wp);
    let q = n.div(&n.sub(&one, wp, RM).mul(&n.sub(m, wp, RM), wp, RM), wp, RM);
    let half_pi = w.cc.pi(wp, RM).div(&BigFloat::from_i32(2, wp), wp, RM);
    let im = half_pi.mul(&q.sqrt(wp, RM), wp, RM);
    let im = if above { im } else { im.neg() };
    let zero = BigFloat::new(wp);
    let lost = lost_bits(
        &[&(k, zero.clone()), &(p, zero.clone())],
        &(re.clone(), zero),
    );
    Ok(((re, im), lost))
}

/// `Π(n | m)` for a real `m > 1` (`m − i0`) and a real or non-real `n`:
/// `Π(n/m | 1/m)/√m − i·Π(n₁ | 1 − 1/m)/((1 − n)√m)`, `n₁ = n(m − 1)/(m(n −
/// 1))` (see the module documentation).  At `n = m` both parts are
/// infinite; the value there is `E(m)/(1 − m)` (DLMF 19.6.1 continued, as
/// mpmath and `eval`'s rewrite of `elliptic_pi(m, m)` take it).
fn pi_split(n: &Complex, m: &BigFloat, w: &mut Wp<'_>) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let zero = BigFloat::new(wp);
    if n.1.is_zero() && n.0.cmp(m) == Some(0) {
        let (e, lost) = complete_e(n, wp)?;
        let one_minus_m = c_sub(&real(1, wp), n, wp, RM);
        return Ok((c_div(&e, &one_minus_m, wp, RM), lost));
    }
    let one = BigFloat::from_i32(1, wp);
    let inv_m = one.div(m, wp, RM);
    let n_over_m = (n.0.mul(&inv_m, wp, RM), n.1.mul(&inv_m, wp, RM));
    let (p1, lost1) = pi_value(&n_over_m, &(inv_m.clone(), zero.clone()), false, w)?;
    let m_minus_one = (m.sub(&one, wp, RM), zero.clone());
    let n_minus_one = c_sub(n, &real(1, wp), wp, RM);
    let n1 = c_div(
        &c_mul(n, &m_minus_one, wp, RM),
        &c_mul(&(m.clone(), zero.clone()), &n_minus_one, wp, RM),
        wp,
        RM,
    );
    let m1 = (one.sub(&inv_m, wp, RM), zero.clone());
    let n_on_cut_above = n.1.is_zero() && above_one(&n.0);
    let (p2, lost2) = pi_value(&n1, &m1, n_on_cut_above, w)?;
    let root_m = m.sqrt(wp, RM);
    let a = (p1.0.div(&root_m, wp, RM), p1.1.div(&root_m, wp, RM));
    let one_minus_n = c_sub(&real(1, wp), n, wp, RM);
    let c = (
        one_minus_n.0.mul(&root_m, wp, RM),
        one_minus_n.1.mul(&root_m, wp, RM),
    );
    // −i·(u + iv) = v − iu
    let q = c_div(&p2, &c, wp, RM);
    let b = (q.1, q.0.neg());
    let v = c_add(&a, &b, wp, RM);
    let lost = lost1.max(lost2).max(lost_bits(&[&a, &b], &v));
    Ok((v, lost))
}

/// `Π(n | m) = R_F(0, y, 1) + (n/3)·R_J(0, y, 1, p)`, `y = 1 − m`, `p = 1 −
/// n`, for arguments `y`, 1, `p` in an open half-plane ([`rotation`]),
/// brought into the right one.
fn pi_principal(
    n: &Complex,
    m: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
    let wp = w.bits();
    let y = c_sub(&real(1, wp), m, wp, RM);
    let p = c_sub(&real(1, wp), n, wp, RM);
    if y.0.is_zero() && y.1.is_zero() {
        return Err(super::infinite("elliptic_pi(n, 1) is infinite"));
    }
    if p.0.is_zero() && p.1.is_zero() {
        return Err(super::infinite("elliptic_pi(1, m) is infinite"));
    }
    let Some(q) = rotation(&[&y, &p], wp) else {
        return Err(super::unevaluable(
            "elliptic_pi: the arguments 1 - m, 1 and 1 - n lie in no half-plane: not yet supported",
        ));
    };
    let (y, z, p) = (c_mul(&y, &q, wp, RM), q.clone(), c_mul(&p, &q, wp, RM));
    let x = real(0, wp);
    let sq = c_sqrt(&q, wp);
    let sq3 = c_mul(&c_mul(&sq, &sq, wp, RM), &sq, wp, RM);
    let f = c_mul(&sq, &rf(&x, &y, &z, wp)?, wp, RM);
    let j = c_mul(&sq3, &rj(&x, &y, &z, &p, w)?, wp, RM);
    let t = c_div(&c_mul(n, &j, wp, RM), &real(3, wp), wp, RM);
    let v = c_add(&f, &t, wp, RM);
    let lost = lost_bits(&[&f, &t], &v);
    Ok((v, lost))
}

/// A factor `q = e^{−iθ}` with `Re(q·w) > 0` for `w = 1` and every
/// argument of `ws` (none 0 or a negative real), when their arguments lie
/// within an arc shorter than `π − 0.2` (`θ` its middle, a guide: the margin
/// covers its `f64` rounding); `None` otherwise.
fn rotation(ws: &[&Complex], wp: usize) -> Option<Complex> {
    // Already in the right half-plane: no rotation.
    if ws
        .iter()
        .all(|z| crate::base::bigfloat::is_strictly_positive(&z.0))
    {
        return Some(real(1, wp));
    }
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for z in ws {
        let a = angle(z)?;
        lo = lo.min(a);
        hi = hi.max(a);
    }
    if hi - lo > std::f64::consts::PI - 0.2 {
        return None;
    }
    let theta = (lo + hi) / 2.0;
    let (s, c) = theta.sin_cos();
    Some((
        crate::base::bigfloat::from_f64(c, wp),
        crate::base::bigfloat::from_f64(-s, wp),
    ))
}

/// The argument of `z` in `(−π, π]` as an `f64` (a guide), from its parts
/// scaled by a common power of two; `None` for 0.
fn angle(z: &Complex) -> Option<f64> {
    let e = match (accuracy::part_mag(&z.0), accuracy::part_mag(&z.1)) {
        (Some(a), Some(b)) => a.max(b),
        (a, b) => a.or(b)?,
    };
    let scaled = |x: &BigFloat| -> f64 {
        if x.is_zero() {
            return 0.0;
        }
        let mut y = x.clone();
        let k = x.exponent().map_or(0, i64::from) - e;
        y.set_exponent(i32::try_from(k).unwrap_or(i32::MIN / 2));
        crate::base::bigfloat::to_f64(&y).unwrap_or(0.0)
    };
    Some(scaled(&z.1).atan2(scaled(&z.0)))
}

/// Carlson's `R_J(x, y, z, p)` over ℂ by duplication (Carlson 1995, eq.
/// 2.36–2.39; mpmath's `RJ_calc` without its initial integration):
///
/// `R_J = 4⁻ᴹ·R_J(xᴹ, yᴹ, zᴹ, pᴹ) + 6·Σ_{k<M} 4⁻ᵏ·R_C(1, 1 + eₖ)/dₖ`,
/// `dₖ = (√pₖ + √xₖ)(√pₖ + √yₖ)(√pₖ + √zₖ)`, `eₖ = δ·4⁻³ᵏ/dₖ²`, `δ = (p −
/// x)(p − y)(p − z)`, until the arguments agree to `2^−(wp/2 + 8)`, where
/// `R_J(μ, μ, μ, μ) = μ^{−3/2}` with `μ = (x + y + z + 2p)/5` is within
/// `O(spread²)` (as for [`rd`]).  Valid for arguments with positive real
/// parts (`x` may be 0); then `|eₖ| < 1` (each factor of `e₀`, `(√p −
/// √w)/(√p + √w)`, has modulus below 1).
fn rj(
    x: &Complex,
    y: &Complex,
    z: &Complex,
    p: &Complex,
    w: &mut Wp<'_>,
) -> Result<Complex, SymplexError> {
    let wp = w.bits();
    let delta = c_mul(
        &c_mul(&c_sub(p, x, wp, RM), &c_sub(p, y, wp, RM), wp, RM),
        &c_sub(p, z, wp, RM),
        wp,
        RM,
    );
    let (mut x, mut y, mut z, mut p) = (x.clone(), y.clone(), z.clone(), p.clone());
    let mut sum = real(0, wp);
    // `4⁻ᵏ`, exactly.
    let mut scale = BigFloat::from_i32(1, wp);
    for _ in 0..(wp * 2 + 200) {
        let two_p = c_add(&p, &p, wp, RM);
        let mu = c_div(
            &c_add(&c_add(&c_add(&x, &y, wp, RM), &z, wp, RM), &two_p, wp, RM),
            &real(5, wp),
            wp,
            RM,
        );
        if let Some(d) = deviations(&[&x, &y, &z, &p], &mu, wp) {
            let devs = [&d[0], &d[1], &d[2], &d[3], &d[3]];
            if series_ready(&devs, wp) {
                let s = rj_series(&devs, wp);
                let tail = c_div(&s, &c_mul(&mu, &c_sqrt(&mu, wp), wp, RM), wp, RM);
                let tail = (tail.0.mul(&scale, wp, RM), tail.1.mul(&scale, wp, RM));
                let six = real(6, wp);
                return Ok(c_add(&c_mul(&six, &sum, wp, RM), &tail, wp, RM));
            }
        }
        let (l, sz) = lambda(&x, &y, &z, wp);
        let (sx, sy, sp) = (c_sqrt(&x, wp), c_sqrt(&y, wp), c_sqrt(&p, wp));
        let d = c_mul(
            &c_mul(&c_add(&sp, &sx, wp, RM), &c_add(&sp, &sy, wp, RM), wp, RM),
            &c_add(&sp, &sz, wp, RM),
            wp,
            RM,
        );
        let s3 = scale.mul(&scale, wp, RM).mul(&scale, wp, RM);
        let e = c_div(
            &(delta.0.mul(&s3, wp, RM), delta.1.mul(&s3, wp, RM)),
            &c_mul(&d, &d, wp, RM),
            wp,
            RM,
        );
        let term = c_div(&rc_one(&e, w)?, &d, wp, RM);
        sum = c_add(
            &sum,
            &(term.0.mul(&scale, wp, RM), term.1.mul(&scale, wp, RM)),
            wp,
            RM,
        );
        x = step(&x, &l, wp);
        y = step(&y, &l, wp);
        z = step(&z, &l, wp);
        p = step(&p, &l, wp);
        scale = scale.div(&BigFloat::from_i32(4, wp), wp, RM);
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "Carlson R_J did not converge".into(),
    })
}

/// `R_C(1, 1 + e) = atan(√e)/√e` for `|e| < 1` (an even function of `√e`,
/// analytic there): `Σ (−e)ᵏ/(2k + 1)` below `|e| < 2⁻¹`, with a tail
/// below twice the first term left out, else by the principal `atan`.
fn rc_one(e: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let wp = w.bits();
    let Some(k) = accuracy::mag(e) else {
        return Ok(real(1, wp));
    };
    if k <= -2 {
        let neg_e = (e.0.neg(), e.1.neg());
        let mut power = real(1, wp);
        let mut acc = real(1, wp);
        let limit = -i64::try_from(wp).unwrap_or(i64::MAX / 4) - 8;
        for j in 1..(4 * wp + 64) {
            power = c_mul(&power, &neg_e, wp, RM);
            let odd = BigFloat::from_i64(2 * j as i64 + 1, wp);
            acc = c_add(
                &acc,
                &(power.0.div(&odd, wp, RM), power.1.div(&odd, wp, RM)),
                wp,
                RM,
            );
            if accuracy::mag(&power).is_none_or(|m| m < limit) {
                return Ok(acc);
            }
        }
        return Err(SymplexError::ComputationFailed {
            operation: "evalf",
            reason: "Carlson R_C series did not converge".into(),
        });
    }
    if k >= 1 {
        return Err(super::unevaluable(
            "Carlson R_J: an argument off its domain",
        ));
    }
    let s = c_sqrt(e, wp);
    let a = super::c_atan(&s, w);
    Ok(c_div(&a, &s, wp, RM))
}

/// The bound of an elliptic-integral node with a non-real argument (see
/// the module documentation): none when an argument's ball reaches a cut
/// with an undecided side, else `sensitivity::complex_route_bound`.
pub(super) fn bound(
    arena: &Arena,
    id: ExprId,
    args: &[ExprId],
    v: &Complex,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Bound {
    let ExprNode::Lib(f, _) = arena.node(id) else {
        return Bound::UNKNOWN;
    };
    let mut balls = Vec::with_capacity(args.len());
    for a in args {
        let (Some(val), Some(b)) = (cache.get(a), cache.bound(a)) else {
            return Bound::UNKNOWN;
        };
        if b.is_unknown() {
            return Bound::UNKNOWN;
        }
        balls.push((val, b));
    }
    // The parameter is the last argument: its cut `[1, ∞)` and the branch
    // point 1 (singular for `K`, `Π`, and for `E`'s derivative).
    if balls
        .last()
        .is_some_and(|(m, b)| on_cut_undecided(m, *b) || near_one(m, *b))
    {
        return Bound::UNKNOWN;
    }
    let undecided = match (f, balls.as_slice()) {
        (LibFn::EllipticPi, [(n, bn), (m, bm)]) => {
            on_cut_undecided(n, *bn)
                || near_one(n, *bn)
                || involution_cut_undecided(n, *bn, m, *bm, w.bits())
                || split_pole_undecided(n, *bn, m, *bm, w.bits())
        }
        (LibFn::EllipticF, [(phi, bp), (m, bm)]) => f_cut_undecided(phi, *bp, m, *bm, w),
        _ => false,
    };
    if undecided {
        return Bound::UNKNOWN;
    }
    super::sensitivity::complex_route_bound(arena, id, args, v, cache, None, w)
}

/// Does the ball of `m` reach the cut `[1, ∞)` with its imaginary part
/// within its error of 0 but not exactly 0 (an undecided side)?  (For `Π`
/// also the characteristic's cut, since 0.56.)
fn on_cut_undecided(m: &Complex, b: Bound) -> bool {
    if accuracy::exact_zero(&m.1, b.im) || !accuracy::part_contains_zero(&m.1, b.im) {
        return false;
    }
    let p = m.0.mantissa_max_bit_len().unwrap_or(64).max(64) + 64;
    let d = m.0.sub(&BigFloat::from_i32(1, 64), p, RM);
    !d.is_negative() || accuracy::part_contains_zero(&d, b.re)
}

/// Does the ball of `z` reach within 8 radii of 1 (an inexact argument)?
fn near_one(z: &Complex, b: Bound) -> bool {
    if b.is_exact() {
        return false;
    }
    let p = z.0.mantissa_max_bit_len().unwrap_or(64).max(64) + 64;
    let d = (z.0.sub(&BigFloat::from_i32(1, 64), p, RM), z.1.clone());
    accuracy::lg_abs(&d) <= b.joint() + 3.0
}

/// Is `Π(n | m)` taken through [`pi_involution`] with `1 − m/n` within its
/// error of the cut `(−∞, 0]` of the principal root (`n` next to the segment
/// `(0, m]`), where the side is undecided?
fn involution_cut_undecided(n: &Complex, bn: Bound, m: &Complex, bm: Bound, prec: usize) -> bool {
    let wp = prec + 32;
    if !uses_involution(n, m, wp) || (bn.is_exact() && bm.is_exact()) {
        return false;
    }
    // |Δ(m/n)| ≤ (|Δm| + |m/n|·|Δn|)/|n|, doubled.
    let q = c_div(m, n, wp, RM);
    let r =
        accuracy::lsum(bm.joint(), accuracy::lg_abs(&q) + bn.joint()) - accuracy::lg_abs(n) + 1.0;
    let t = c_sub(&real(1, wp), &q, wp, RM);
    accuracy::part_contains_zero(&t.1, r)
        && (t.0.is_negative() || accuracy::part_contains_zero(&t.0, r))
}

/// Is `Π(n | m)` taken through [`pi_split_off_axis`] with `m`'s ball reaching
/// the real axis, or with the pole `P = 1/n` within its error of the
/// boundary of the triangle `0, 1, t₀ = 1/m` (a barycentric coordinate that
/// decides inside or outside within the radius it may move by)?
fn split_pole_undecided(n: &Complex, bn: Bound, m: &Complex, bm: Bound, prec: usize) -> bool {
    let wp = prec + 32;
    if !uses_off_axis_split(n, m, wp) || (bn.is_exact() && bm.is_exact()) {
        return false;
    }
    if accuracy::part_contains_zero(&m.1, bm.im) {
        return true;
    }
    // An exactly real `n > 1` (the pole on the path): the convention `n − i0`
    // decides, with the side of `t₀` decided above.
    if n.1.is_zero() && accuracy::is_exact(bn.im) && above_one(&n.0) {
        return false;
    }
    let t0 = c_div(&real(1, wp), m, wp, RM);
    let coords = barycentric(n, &t0, wp);
    // |ΔP| ≤ 2|Δn|/|n|², |Δt₀| ≤ 2|Δm|/|m|²; with β = Im P/Im t₀ and α = Re P −
    // β·Re t₀ a coordinate moves by at most (|ΔP| + (1 + |β|)|Δt₀|)·(1 +
    // |t₀|)/|Im t₀|, doubled.
    let beta = accuracy::lsum(0.0, accuracy::part_lg(&coords[1]));
    let r = accuracy::lsum(
        bn.joint() - 2.0 * accuracy::lg_abs_low(n) + 1.0,
        bm.joint() - 2.0 * accuracy::lg_abs_low(m) + 1.0 + beta,
    ) + accuracy::lsum(0.0, accuracy::lg_abs(&t0))
        - accuracy::part_lg_low(&t0.1)
        + 1.0;
    let negative = coords
        .iter()
        .filter(|c| c.is_negative())
        .map(accuracy::part_lg_low)
        .fold(f64::NEG_INFINITY, f64::max);
    if negative > f64::NEG_INFINITY {
        return negative <= r;
    }
    coords
        .iter()
        .map(accuracy::part_lg_low)
        .fold(f64::INFINITY, f64::min)
        <= r
}

/// For `F(φ | m)`: does the ball of `1 − m sin²φ₀` reach within 8 radii of
/// 0 (a branch point of the integrand) — and, for a non-real `φ`, does the
/// ball of `Re φ` reach a line `(k + ½)π`, or that of `1 − m sin²φ₀` the
/// negative axis, where the convention of [`incomplete_f_complex`] has its
/// cuts?
fn f_cut_undecided(phi: &Complex, bp: Bound, m: &Complex, bm: Bound, w: &mut Wp<'_>) -> bool {
    let wp = w.bits() + 32;
    let mut ww = w.at(wp);
    let (_, phi0) = reduce_phi(phi, &mut ww);
    if phi.1.is_zero() {
        if bp.is_exact() && bm.is_exact() {
            return false;
        }
        let s = phi0.0.sin(wp, RM, ww.cc);
        let s2 = s.mul(&s, wp, RM);
        let sin2 = phi0.0.add(&phi0.0, wp, RM).sin(wp, RM, ww.cc);
        let y = c_sub(
            &real(1, wp),
            &c_mul(m, &(s2.clone(), BigFloat::new(wp)), wp, RM),
            wp,
            RM,
        );
        let r = accuracy::lsum(
            accuracy::lg_abs(m) + accuracy::part_lg(&sin2) + bp.joint(),
            accuracy::part_lg(&s2) + bm.joint(),
        ) + 1.0;
        return accuracy::lg_abs(&y) <= r + 3.0;
    }
    // Distance of Re φ₀ to ±π/2 against the radius of Re φ (and its rounding).
    let half_pi = ww.cc.pi(wp, RM).div(&BigFloat::from_i32(2, wp), wp, RM);
    let gap = half_pi.sub(&phi0.0.abs(), wp, RM);
    let floor = f64::from(phi0.0.exponent().unwrap_or(0)) - wp as f64 + 4.0;
    if !crate::base::bigfloat::is_strictly_positive(&gap)
        || accuracy::part_contains_zero(&gap, accuracy::lsum(bp.re, floor) + 1.0)
    {
        return true;
    }
    if bp.is_exact() && bm.is_exact() {
        return false;
    }
    // y = 1 − m sin²φ₀: |Δy| ≤ |m|·|sin 2φ₀|·|Δφ| + |sin²φ₀|·|Δm|, doubled.
    let s = super::c_sin(&phi0, &mut ww);
    let s2 = c_mul(&s, &s, wp, RM);
    let sin2 = super::c_sin(&c_add(&phi0, &phi0, wp, RM), &mut ww);
    let y = c_sub(&real(1, wp), &c_mul(m, &s2, wp, RM), wp, RM);
    let r = accuracy::lsum(
        accuracy::lg_abs(m) + accuracy::lg_abs(&sin2) + bp.joint(),
        accuracy::lg_abs(&s2) + bm.joint(),
    ) + 1.0;
    // An amplitude on the imaginary axis exactly (`Re φ₀ = 0` with no error)
    // and an exactly real parameter keep `y` exactly real: on `y < 0` the
    // side is the convention's (`y + i0`, mpmath's principal root of a
    // negative real), not undecided.
    let y_exactly_real =
        phi0.0.is_zero() && accuracy::is_exact(bp.re) && m.1.is_zero() && accuracy::is_exact(bm.im);
    accuracy::lg_abs(&y) <= r + 3.0
        || (!y_exactly_real
            && accuracy::part_contains_zero(&y.1, r)
            && (y.0.is_negative() || accuracy::part_contains_zero(&y.0, r)))
}
