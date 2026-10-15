//! Complete and incomplete elliptic integrals `K(m)`, `E(m)` and `F(φ | m)`
//! where the value or the parameter is not real, by Carlson's symmetric
//! integrals over ℂ.
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
//! A non-real `φ` and `Π(n | m)` off the real range stay refused.
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
//! `2^−(prec+12)` relative plus, for each inexact argument, the change of
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
    let mut extra = 32usize;
    for _ in 0..4 {
        let wp = prec + extra;
        let (v, lost) = match f {
            LibFn::EllipticK => (complete_k(arg(0)?, wp)?, 0),
            LibFn::EllipticE => complete_e(arg(0)?, wp)?,
            LibFn::EllipticF => incomplete_f(&arg(0)?.0, arg(1)?, &mut w.at(wp))?,
            _ => {
                return Err(super::unevaluable(
                    "not an elliptic integral of the first two kinds",
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
    let own = accuracy::lg_abs(v) - prec as f64 - 12.0;
    let mut total = own;
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
    Bound::both(total)
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
