//! Complete and incomplete elliptic integrals `K(m)`, `E(m)`, `F(φ | m)` and
//! `Π(n | m)` where the value or a parameter is not real, by Carlson's
//! symmetric integrals over ℂ.
//!
//! The real routines of `evalf.rs` (`arb_elliptic_*`) cover a real
//! parameter with a real value: `m ≤ 1`, and `1 − m sin²φ > 0` for `F`.
//! Beyond that the value is complex (DLMF 19.7.3: `K(m)` for `m > 1` is
//! `(K(1/m) − i·K(1 − 1/m))/√m` here), and so it is for a non-real `m`.
//! These routines take every such case with
//!
//! ```text
//! K(m) = R_F(0, 1 − m, 1)
//! E(m) = R_F(0, 1 − m, 1) − (m/3)·R_D(0, 1 − m, 1)
//! F(φ | m) = sin φ · R_F(cos²φ, 1 − m sin²φ, 1) + 2k·K(m),   φ = φ₀ + kπ, |φ₀| ≤ π/2
//! ```
//!
//! (DLMF 19.25.1, 19.25.5; the reduction is DLMF 19.2.10), for a real `φ`.
//! A non-real `φ` stays refused.
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
//! A real `n > 1` with a non-real `m`, and non-real arguments that lie in
//! no half-plane together with 1, stay refused.  On the cuts the values are
//! mpmath's (and so SymPy's): the limits from below, `m − i0` and `n − i0`
//! (`ellippi(1/2, 2) = 1.5327 − 2.1041i`, `ellippi(2, 1/2) = −0.3135 −
//! 1.8138i`); `n₁` then lies on its cut from above when `n > 1`, since
//! `dn₁/dn < 0`.  Up to 0.54 every such `Π` was refused.
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
//! The duplication is iterated until the arguments agree to `2^−(wp/2 + 8)`
//! relative, where the remainder of `R(μ, μ, μ)` is `O(spread²)` (as in the
//! real routines), at `wp = prec + 32` bits.  The error is relative to the
//! modulus of the value, so a part much smaller than the other (`E(m)` for
//! `m` just above 1 has an imaginary part of order `m − 1`), and the
//! difference `R_F − (m/3)R_D` of `E` near `m = 1`, are evaluated again with
//! the bits they lose.  Each part is then rounded to `prec` bits.
//!
//! The node's bound ([`bound`]) for a non-real parameter is its own
//! `2^−(prec+12)` relative to each part plus, for each inexact argument, the change of
//! the value with that argument moved by its radius (evaluated again:
//! `sensitivity::numeric_change`); a parameter whose ball reaches the cut
//! `[1, ∞)` with an undecided side has no bound.  An exactly real parameter
//! is bounded by the sensitivity rules of the real routines
//! (`sensitivity::lib_sens`, numerically beyond `m = 1`).

use super::accuracy::{self, Bound};
use super::ball::Cache;
use super::wp::{RM, Wp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_abs, c_add, c_div, c_mul, c_sub};
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::ExprId;
use astro_float::BigFloat;

/// Does `f(args)` go to these routines?  `K`, `E` of a non-real parameter
/// or one above 1; `F` of a real `φ` with a non-real parameter, or a real
/// one for which the real routine refuses (`1 − m sin²φ₀ ≤ 0`, or `m > 1`
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
                return false;
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

/// Does `f(args)` have a non-real argument (the node bound of [`bound`])?
pub(super) fn nonreal_args(f: LibFn, args: &[ExprId], cache: Cache<'_>) -> bool {
    takes(f, args, cache)
        && args
            .iter()
            .any(|a| cache.get(a).is_some_and(|v| !v.1.is_zero()))
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
        let (v, lost) = match f {
            LibFn::EllipticK => (complete_k(arg(0)?, wp)?, 0),
            LibFn::EllipticE => complete_e(arg(0)?, wp)?,
            LibFn::EllipticF => incomplete_f(&arg(0)?.0, arg(1)?, &mut w.at(wp))?,
            LibFn::EllipticPi => complete_pi(arg(0)?, arg(1)?, &mut w.at(wp))?,
            _ => {
                return Err(super::unevaluable(
                    "not a complete or incomplete elliptic integral",
                ));
            }
        };
        // Bits a part loses next to the modulus (the error is relative to
        // it), and those of a cancelling difference.
        let need = lost + part_gap(&v) + 16;
        if need <= extra {
            return Ok((super::round_to(v.0, prec), super::round_to(v.1, prec)));
        }
        if need > 4 * prec + 512 {
            break;
        }
        extra = need + 16;
    }
    Err(super::unevaluable(format!(
        "{}: the value cancels beyond the working precision",
        f.name()
    )))
}

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

/// Do the values agree to `2^−bits` relative (to the first one)?
fn close(vals: &[&Complex], bits: usize, wp: usize) -> bool {
    let first = vals[0];
    let Some(scale) = accuracy::mag(first) else {
        return vals.iter().all(|v| v.0.is_zero() && v.1.is_zero());
    };
    let limit = scale - i64::try_from(bits).unwrap_or(i64::MAX / 4);
    vals[1..].iter().all(|v| {
        let d = c_sub(v, first, wp, RM);
        accuracy::mag(&d).is_none_or(|m| m < limit)
    })
}

/// The stopping spread of the duplication (see the module documentation).
fn spread_bits(wp: usize) -> usize {
    wp / 2 + 8
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
        if close(&[&x, &y, &z], spread_bits(wp), wp) {
            // R_F(μ, μ, μ) = μ^{−1/2}
            let mu = c_div(
                &c_add(&c_add(&x, &y, wp, RM), &z, wp, RM),
                &real(3, wp),
                wp,
                RM,
            );
            return Ok(c_div(&real(1, wp), &c_sqrt(&mu, wp), wp, RM));
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
        if close(&[&x, &y, &z], spread_bits(wp), wp) {
            // R_D(μ, μ, μ) = μ^{−3/2}, μ = (x + y + 3z)/5
            let three_z = c_mul(&real(3, wp), &z, wp, RM);
            let mu = c_div(
                &c_add(&c_add(&x, &y, wp, RM), &three_z, wp, RM),
                &real(5, wp),
                wp,
                RM,
            );
            let tail = c_div(&real(1, wp), &c_mul(&mu, &c_sqrt(&mu, wp), wp, RM), wp, RM);
            return Ok(c_add(&acc, &c_mul(&scale, &tail, wp, RM), wp, RM));
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
        return Err(super::unevaluable(format!("{what}(1) is infinite")));
    }
    Ok(y)
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

/// `F(φ | m)` for a real `φ`: `sin φ₀ · R_F(cos²φ₀, 1 − m sin²φ₀, 1) +
/// 2k·K(m)` with `φ = φ₀ + kπ`, `|φ₀| ≤ π/2` (mpmath's reduction), and the
/// bits the sum cancels.
fn incomplete_f(
    phi: &BigFloat,
    m: &Complex,
    w: &mut Wp<'_>,
) -> Result<(Complex, usize), SymplexError> {
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
        return Err(super::unevaluable("elliptic_pi(n, 1) is infinite"));
    }
    if n_real && is_one(&n.0) {
        return Err(super::unevaluable("elliptic_pi(1, m) is infinite"));
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
    if n_real && above_one(&n.0) {
        return Err(super::unevaluable(
            "elliptic_pi: a characteristic n > 1 with a non-real parameter m is not yet supported",
        ));
    }
    pi_principal(n, m, w)
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
        return Err(super::unevaluable("elliptic_pi(n, 1) is infinite"));
    }
    if p.0.is_zero() && p.1.is_zero() {
        return Err(super::unevaluable("elliptic_pi(1, m) is infinite"));
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
        if close(&[&x, &y, &z, &p], spread_bits(wp), wp) {
            let two_p = c_add(&p, &p, wp, RM);
            let mu = c_div(
                &c_add(&c_add(&c_add(&x, &y, wp, RM), &z, wp, RM), &two_p, wp, RM),
                &real(5, wp),
                wp,
                RM,
            );
            let tail = c_div(&real(1, wp), &c_mul(&mu, &c_sqrt(&mu, wp), wp, RM), wp, RM);
            let tail = (tail.0.mul(&scale, wp, RM), tail.1.mul(&scale, wp, RM));
            let six = real(6, wp);
            return Ok(c_add(&c_mul(&six, &sum, wp, RM), &tail, wp, RM));
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
/// the module documentation).
pub(super) fn bound(
    arena: &Arena,
    id: ExprId,
    args: &[ExprId],
    v: &Complex,
    cache: Cache<'_>,
    w: &mut Wp<'_>,
) -> Bound {
    let prec = w.bits();
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Bound::UNKNOWN;
    }
    // The routine's own error, each part within `2^−(prec+12)` of itself
    // (`eval` adds the bits a smaller part lies below the modulus); a part
    // that is 0, within that of the modulus.
    let joint = accuracy::lg_abs(v) - prec as f64 - 12.0;
    let own = |x: &BigFloat| {
        if x.is_zero() {
            joint
        } else {
            accuracy::part_lg(x) - prec as f64 - 12.0
        }
    };
    let mut total = accuracy::EXACT;
    // The parameter is the last argument.
    let last = args.len().saturating_sub(1);
    for (i, a) in args.iter().enumerate() {
        let (Some(val), Some(b)) = (cache.get(a), cache.bound(a)) else {
            return Bound::UNKNOWN;
        };
        if b.is_unknown() {
            return Bound::UNKNOWN;
        }
        if i == last && on_cut_undecided(val, b) {
            return Bound::UNKNOWN;
        }
        if b.is_exact() || args[..i].contains(a) {
            continue;
        }
        match super::sensitivity::numeric_change(arena, id, *a, b.joint(), v, cache, w) {
            Some(c) => total = accuracy::lsum(total, c),
            None => return Bound::UNKNOWN,
        }
    }
    if accuracy::is_unknown(total) {
        return Bound::UNKNOWN;
    }
    Bound {
        re: accuracy::lsum(own(&v.0), total),
        im: accuracy::lsum(own(&v.1), total),
    }
}

/// Does the parameter's ball reach the cut `[1, ∞)` with its imaginary part
/// within its error of 0 but not exactly 0 (an undecided side)?
fn on_cut_undecided(m: &Complex, b: Bound) -> bool {
    if accuracy::exact_zero(&m.1, b.im) || !accuracy::part_contains_zero(&m.1, b.im) {
        return false;
    }
    let p = m.0.mantissa_max_bit_len().unwrap_or(64).max(64) + 64;
    let d = m.0.sub(&BigFloat::from_i32(1, 64), p, RM);
    !d.is_negative() || accuracy::part_contains_zero(&d, b.re)
}
