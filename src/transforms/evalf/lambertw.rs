//! Lambert W on every branch over ℂ: `W_k(z)`, the solution `w` of
//! `w·eʷ = z` on branch `k` (`ExprNode::LambertW(z)` is `k = 0`, the library
//! call `lambertw(z, k)` any other integer `k`).
//!
//! The algorithm is mpmath's `lambertw` (`mpmath/functions/functions.py`,
//! BSD licence; see THIRD-PARTY-NOTICES.md): a starting value — the series
//! in `p = ±√(2(e·z + 1))` at the branch point `−1/e` (Corless et al.,
//! coefficients by mpmath's recurrence), mpmath's piecewise `f64`
//! approximations of `W₀` and `W₋₁` for moderate `|z|`, or the asymptotic
//! series `L₁ − L₂ + L₂/L₁ + L₂(L₂ − 2)/(2L₁²)`, `L₁ = ln z + 2πik`,
//! `L₂ = ln L₁` — refined by Halley's iteration on `w·eʷ − z`.  The branch
//! cuts and the values on them follow mpmath and SymPy (counter-clockwise
//! continuity): the cut of `W₀` is `(−∞, −1/e]`, that of every other branch
//! `(−∞, 0]`, and an exactly real argument on a cut takes the limit from
//! above, so `W₀(x)` for `x < −1/e` is the value with positive imaginary
//! part and `W₋₁(x) = conj W₀(x)` there, while `W₋₁` is real on
//! `[−1/e, 0)`.
//!
//! Near the branch point the problem is ill-conditioned: with
//! `δ = z + 1/e`, `W′ ≈ e/√(2eδ)` and a residual computed to `2^−wp` moves
//! `w` by about `2^−wp/√δ`.  The offset `δ` is computed with as many more
//! bits as it cancels, and the iteration runs with half of them more
//! (before 0.30 the principal branch had only a real Halley iteration at
//! `prec + 32` bits and refused every `x < −1/e`, "outside principal
//! branch domain", although `W₀(−1) = −0.3181 + 1.3372i`).
//!
//! The propagated error ([`error_bound`]) is `|W′(z)|·r` with
//! `W′ = e^{−W}/(1 + W)`, bounded over the argument's error ball once the
//! first-order change is below a sixteenth of `min(1, |1 + W|)` (which keeps
//! the ball away from both singular points, `−1/e` and, for `k ≠ 0`, `0`);
//! an argument whose imaginary part is only within its error of 0 while its
//! real part may lie on the branch's cut has an undecidable side and no
//! bound.

use astro_float::{BigFloat, Consts, RoundingMode};
use num_bigint::BigInt;
use num_complex::Complex64;
use num_traits::{ToPrimitive, Zero};

use super::accuracy::{self, Bound, ErrExp};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_add, c_div, c_mul, c_sub, c_zero};
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// `−1/e` rounded to `f64` (mpmath's `r` of the `f64` approximations).
const NEG_INV_E: f64 = -0.367_879_441_171_442_3;

/// The argument and branch of a Lambert W node: `(x, 0)` for
/// `ExprNode::LambertW(x)` and `lambertw(x)`, `(x, k)` for `lambertw(x, k)`
/// with an integer literal `k` (an error for any other branch index);
/// `None` for every other node.
pub(super) fn lambert_parts(
    arena: &Arena,
    node: &ExprNode,
) -> Option<Result<(ExprId, i64), SymplexError>> {
    match node {
        ExprNode::LambertW(x) => Some(Ok((*x, 0))),
        ExprNode::Apply(sid, args) if arena.lib_fn(*sid) == Some(LibFn::LambertW) => {
            Some(match args[..] {
                [x] => Ok((x, 0)),
                [x, k] => match arena.as_num(k) {
                    Some(q) if q.is_integer() => {
                        q.to_integer().to_i64().map(|k| (x, k)).ok_or_else(|| {
                            SymplexError::NotImplemented(format!(
                                "evaluation of the Lambert W branch {q} (beyond 64 bits)"
                            ))
                        })
                    }
                    _ => Err(SymplexError::Unevaluable {
                        reason: format!(
                            "LambertW(x, k) needs an integer branch index k, got {}",
                            arena.display(k)
                        ),
                    }),
                },
                _ => Err(SymplexError::Unevaluable {
                    reason: format!("lambertw called with {} arguments", args.len()),
                }),
            })
        }
        _ => None,
    }
}

fn unevaluable(reason: impl Into<String>) -> SymplexError {
    SymplexError::Unevaluable {
        reason: reason.into(),
    }
}

fn is_zero(z: &Complex) -> bool {
    z.0.is_zero() && z.1.is_zero()
}

/// `x < 0` strictly (astro-float's sign bit is set for `−0` too).
fn neg(x: &BigFloat) -> bool {
    x.is_negative() && !x.is_zero()
}

fn to_f64(x: &BigFloat) -> f64 {
    super::bigfloat_to_f64_rounded(x, RoundingMode::ToEven).unwrap_or(f64::NAN)
}

fn from_c64(z: Complex64, prec: usize) -> Complex {
    let im = if z.im == 0.0 {
        BigFloat::new(prec)
    } else {
        BigFloat::from_f64(z.im, prec)
    };
    (BigFloat::from_f64(z.re, prec), im)
}

fn real(x: BigFloat, prec: usize) -> Complex {
    (x, BigFloat::new(prec))
}

fn int(n: i64, prec: usize) -> Complex {
    real(BigFloat::from_i64(n, prec), prec)
}

/// `W_k(z)` to `prec` bits (see the module documentation).
///
/// # Errors
///
/// [`SymplexError::Unevaluable`] at the logarithmic singularity `z = 0` of
/// the branches `k ≠ 0` (`W_k(0) = −∞`) and for a non-finite `z`;
/// [`SymplexError::ComputationFailed`] if Halley's iteration does not
/// converge.
pub(super) fn lambert_w(
    z: &Complex,
    k: i64,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    if [&z.0, &z.1].iter().any(|p| p.is_nan() || p.is_inf()) {
        return Err(unevaluable("LambertW of a non-finite argument"));
    }
    if is_zero(z) {
        return if k == 0 {
            Ok(c_zero(prec))
        } else {
            Err(unevaluable(format!(
                "LambertW(0, {k}) is -oo (logarithmic singularity of the branch k = {k})"
            )))
        };
    }
    let real_z = z.1.is_zero();
    let (delta, cancel) = branch_point_offset(z, prec, rm, cc);
    // The two real branches on their real domains: `W₀` on `[−1/e, ∞)`,
    // `W₋₁` on `[−1/e, 0)` (`δ = z + 1/e` is never 0, `1/e` irrational).
    let real_value = real_z && (k == 0 || (k == -1 && neg(&z.0))) && !neg(&delta.0);
    let magz = accuracy::mag(z).unwrap_or(0);
    let near_bp = magz < 1 && accuracy::lg_abs(&delta) < (0.05f64).log2();
    let k_bits = 64 - k.unsigned_abs().leading_zeros() as usize;
    let wp = prec + 24 + k_bits + if near_bp { cancel / 2 + 16 } else { 0 };
    let tol = (wp - 5) as i64;

    // The branch point is on the boundary of `W₀` (both sides), of `W₋₁`
    // from above and of `W₁` from below.
    let touches = k == 0 || (k == -1 && !neg(&z.1)) || (k == 1 && neg(&z.1));
    let start = if near_bp && touches {
        match branch_point_series(&delta, k != 0, cancel, tol, wp, rm, cc) {
            (w, true) => return Ok(finish(w, real_value, prec, rm)),
            (w, false) => w,
        }
    } else if (k == 0 || k == -1) && (-9..900).contains(&magz) {
        let zf = Complex64::new(to_f64(&z.0), if real_z { 0.0 } else { to_f64(&z.1) });
        from_c64(hybrid_start(zf, real_z, k), wp)
    } else {
        asymptotic_start(z, k, magz, wp, rm, cc)
    };
    let w = halley(z, start, tol, wp, rm, cc)?;
    Ok(finish(w, real_value, prec, rm))
}

/// Rounds `w` to `prec` bits, with an exactly zero imaginary part when the
/// value is real (`real`).
fn finish(w: Complex, real: bool, prec: usize, rm: RoundingMode) -> Complex {
    let re = super::round_to(w.0, prec, rm);
    let im = if real {
        BigFloat::new(prec)
    } else {
        super::round_to(w.1, prec, rm)
    };
    (re, im)
}

/// `δ = z + 1/e` and the bits it cancels (`log₂(|z|/|δ|)`, 0 when `|δ|`
/// is not small), computed with those bits more so that `δ` is accurate
/// relative to itself.
fn branch_point_offset(
    z: &Complex,
    prec: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> (Complex, usize) {
    let cap = 3 * prec + 256;
    let mut p = prec + 64;
    loop {
        let inv_e = BigFloat::from_i32(1, p).div(&cc.e(p, rm), p, rm);
        let re = z.0.add(&inv_e, p, rm);
        let delta = (re, z.1.clone());
        let lost = match (accuracy::mag(z), accuracy::mag(&delta)) {
            (Some(a), Some(b)) => usize::try_from(a - b).unwrap_or(0),
            (Some(_), None) => p,
            _ => 0,
        };
        if p >= lost + prec + 32 || p >= cap {
            return (delta, lost);
        }
        p = (lost + prec + 64).min(cap);
    }
}

/// The coefficients `u_l` of `W = Σ u_l p^l` at the branch point
/// (Corless et al.; mpmath's recurrence): `−1, 1, −1/3, 11/72, −43/540, …`.
fn branch_point_coefficients(n: usize) -> Vec<Q> {
    let q = |a: i64, b: i64| Q::new(BigInt::from(a), BigInt::from(b));
    let mut u = vec![q(-1, 1), q(1, 1)];
    let mut a = vec![q(2, 1), q(-1, 1)];
    for l in 2..n {
        let mut al = Q::zero();
        for j in 2..l {
            al += &u[j] * &u[l + 1 - j];
        }
        let li = i64::try_from(l).unwrap_or(i64::MAX);
        let ul = q(li - 1, li + 1) * (&u[l - 2] / q(2, 1) + &a[l - 2] / q(4, 1))
            - &al / q(2, 1)
            - &u[l - 1] / q(li + 1, 1);
        a.push(al);
        u.push(ul);
    }
    u.truncate(n);
    u
}

/// The series `Σ u_l p^l` with `p = ±√(2e·δ)` (`−` on the branches other
/// than `W₀`): the value and whether its terms fell below `2^−tol` (then it
/// is the result; otherwise a starting value).
fn branch_point_series(
    delta: &Complex,
    minus: bool,
    cancel: usize,
    tol: i64,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> (Complex, bool) {
    let two_e = real(cc.e(wp, rm).mul(&BigFloat::from_i32(2, wp), wp, rm), wp);
    let mut p = super::c_sqrt(&c_mul(&two_e, delta, wp, rm), wp, rm, cc);
    if minus {
        p = (p.0.neg(), p.1.neg());
    }
    let n = cancel.clamp(2, 64);
    let coeffs = branch_point_coefficients(n);
    let mut s = c_zero(wp);
    let mut pl = int(1, wp);
    for u in &coeffs {
        let term = c_mul(&real(super::ratio_to_bigfloat(u, wp, rm), wp), &pl, wp, rm);
        s = c_add(&s, &term, wp, rm);
        if accuracy::mag(&term).is_none_or(|m| m < -tol) {
            return (s, true);
        }
        pl = c_mul(&pl, &p, wp, rm);
    }
    (s, false)
}

/// mpmath's `_lambertw_approx_hybrid`: a starting value for `W₀` and `W₋₁`
/// from `f64` pieces — Taylor polynomials in the half-planes and near `−1`,
/// the branch-point expansion near `−1/e`, and the asymptotic series.
/// `real` marks an exactly real argument (mpmath's real type), whose value
/// stays real where the branch is.  (The literals are mpmath's; `−0.318`
/// is `Re W₀(−1)` rounded, not `−1/π`.)
#[allow(clippy::approx_constant)]
fn hybrid_start(z: Complex64, real: bool, k: i64) -> Complex64 {
    let c = Complex64::new;
    let (x, y) = (z.re, if real { 0.0 } else { z.im });
    let imag_sign = if real || y == 0.0 {
        0
    } else if y > 0.0 {
        1
    } else {
        -1
    };
    let r = NEG_INV_E;
    let sqrt_part = |s: f64| -> Complex64 {
        // `s·√(z − r)` with `√` real when `z` is real and above `r`.
        let d = c(x - r, y);
        let root = if imag_sign == 0 && x > r {
            c((x - r).sqrt(), 0.0)
        } else {
            d.sqrt()
        };
        c(-1.0, 0.0) + root * (s * 2.331_643_981_597_12) - d * 1.812_187_885_639_36
    };
    let (l1, l2) = if k == 0 {
        if -4.0 < y && y < 4.0 && -1.0 < x && x < 2.5 {
            if imag_sign != 0 {
                if y > 1.0 {
                    return c(0.876, 0.645) + c(0.118, -0.174) * (z - c(0.75, 2.5));
                }
                if y > 0.25 {
                    return c(0.505, 0.204) + c(0.375, -0.132) * (z - c(0.75, 0.5));
                }
                if y < -1.0 {
                    return c(0.876, -0.645) + c(0.118, 0.174) * (z - c(0.75, -2.5));
                }
                if y < -0.25 {
                    return c(0.505, -0.204) + c(0.375, 0.132) * (z - c(0.75, -0.5));
                }
            }
            if x < -0.5 {
                return if imag_sign >= 0 {
                    c(-0.318, 1.34) + c(-0.697, -0.593) * (c(x, y) + 1.0)
                } else {
                    c(-0.318, -1.34) + c(-0.697, 0.593) * (c(x, y) + 1.0)
                };
            }
            if x < -0.2 {
                return sqrt_part(1.0);
            }
            if x < 0.5 {
                return c(x, y);
            }
            return c(0.2, 0.0) + c(x, y) * 0.3;
        }
        if imag_sign == 0 && x > 0.0 {
            let l1 = x.ln();
            (c(l1, 0.0), c(l1.ln(), 0.0))
        } else {
            let l1 = c(x, y).ln();
            (l1, l1.ln())
        }
    } else {
        if imag_sign >= 0 && y < 0.1 && -0.6 < x && x < -0.2 {
            return sqrt_part(-1.0);
        }
        if imag_sign == 0 && (-0.2..0.0).contains(&x) {
            let l1 = (-x).ln();
            return c(l1 - (-l1).ln(), 0.0);
        }
        let l1 = c(x, y).ln() - c(0.0, 2.0 * std::f64::consts::PI);
        (l1, l1.ln())
    };
    l1 - l2 + l2 / l1 + l2 * (l2 - 2.0) / (l1 * l1 * 2.0)
}

/// mpmath's asymptotic starting values (`_lambertw_series`) at `wp` bits:
/// `z(1 − z)` for a small `z` on `W₀`, `L₁ − ln(−L₁)` with `L₁ = ln(−z)` for
/// `W₋₁` on `(−1/e, 0)`, otherwise `L₁ − L₂ + L₂/L₁ + L₂(L₂ − 2)/(2L₁²)`
/// with `L₁ = ln z + 2πik`, `L₂ = ln L₁` (relative error `O(1/ln z)` both as
/// `z → 0` and as `z → ∞`).
fn asymptotic_start(
    z: &Complex,
    k: i64,
    magz: i64,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Complex {
    if k == 0 && magz < -1 {
        return c_mul(z, &c_sub(&int(1, wp), z, wp, rm), wp, rm);
    }
    if k == -1 && z.1.is_zero() && neg(&z.0) && to_f64(&z.0) > NEG_INV_E {
        let l1 = z.0.neg().ln(wp, rm, cc);
        let l2 = l1.neg().ln(wp, rm, cc);
        return real(l1.sub(&l2, wp, rm), wp);
    }
    let two_pi_k = cc
        .pi(wp, rm)
        .mul(&BigFloat::from_i128(2 * i128::from(k), wp.max(128)), wp, rm);
    let ln_z = super::c_ln(z, wp, rm, cc);
    let l1 = (ln_z.0, ln_z.1.add(&two_pi_k, wp, rm));
    let l2 = super::c_ln(&l1, wp, rm, cc);
    let two = int(2, wp);
    let a = c_sub(&l1, &l2, wp, rm);
    let b = c_div(&l2, &l1, wp, rm);
    let l1sq2 = c_mul(&c_mul(&l1, &l1, wp, rm), &two, wp, rm);
    let c = c_div(
        &c_mul(&l2, &c_sub(&l2, &two, wp, rm), wp, rm),
        &l1sq2,
        wp,
        rm,
    );
    c_add(&c_add(&a, &b, wp, rm), &c, wp, rm)
}

/// Halley's iteration on `w·eʷ − z` (mpmath's form) until a step is below
/// `2^−tol` relative to the iterate.
fn halley(
    z: &Complex,
    mut w: Complex,
    tol: i64,
    wp: usize,
    rm: RoundingMode,
    cc: &mut Consts,
) -> Result<Complex, SymplexError> {
    let one = int(1, wp);
    let two = int(2, wp);
    for _ in 0..100 {
        let ew = super::c_exp(&w, wp, rm, cc);
        let wew = c_mul(&w, &ew, wp, rm);
        let wewz = c_sub(&wew, z, wp, rm);
        if is_zero(&wewz) {
            return Ok(w);
        }
        let w2 = c_mul(&c_add(&w, &one, wp, rm), &two, wp, rm);
        if is_zero(&w2) {
            break;
        }
        let corr = c_div(&c_mul(&c_add(&w, &two, wp, rm), &wewz, wp, rm), &w2, wp, rm);
        let denom = c_sub(&c_add(&wew, &ew, wp, rm), &corr, wp, rm);
        if is_zero(&denom) {
            break;
        }
        let step = c_div(&wewz, &denom, wp, rm);
        let wn = c_sub(&w, &step, wp, rm);
        let done = match (accuracy::mag(&step), accuracy::mag(&wn)) {
            (None, _) => true,
            (Some(s), Some(m)) => s <= m - tol,
            (Some(_), None) => false,
        };
        w = wn;
        if done {
            return Ok(w);
        }
    }
    Err(SymplexError::ComputationFailed {
        operation: "evalf",
        reason: "LambertW: Halley iteration did not converge".into(),
    })
}

/// `log₂` of the factor by which `|W′|` can grow over a ball on which `W`
/// moves by at most an eighth of `min(1, |1 + W|)`: `e^{1/8}·8/7`.
const GROWTH: f64 = 0.372_988_913_606_792_2;

/// The error bound of the value `w = W_k(z)` for an argument `z ± bz`,
/// without the rounding of `w` (the caller's `accuracy::reported` adds
/// it): the kernel's own error (it works with at least 24 bits more than
/// `prec`), plus `2·max|W′|·r` over the ball of radius `r = bz.joint()`
/// (see the module documentation).  Exactly real (an exact imaginary part)
/// where the branch is real on an exactly real argument.
pub(super) fn error_bound(z: &Complex, bz: Bound, w: &Complex, k: i64, prec: usize) -> Bound {
    if bz.is_unknown() {
        return Bound::UNKNOWN;
    }
    let real_arg = accuracy::exactly_real(z, bz);
    let real_value = real_arg && w.1.is_zero();
    let own = accuracy::lg_abs(w) - prec as f64 - 16.0;
    let bound = |e: ErrExp| {
        if real_value {
            Bound::real(e)
        } else {
            Bound::both(e)
        }
    };
    if bz.is_exact() {
        return bound(own);
    }
    // An imaginary part within its error of 0: the side of the cut is
    // undecidable where the real part may lie on it.
    if !real_arg && accuracy::part_contains_zero(&z.1, bz.im) {
        let lo = to_f64(&z.0) - bz.re.exp2();
        let cut_end = if k == 0 { NEG_INV_E + 1e-9 } else { 1e-300 };
        if lo.is_nan() || lo <= cut_end {
            return Bound::UNKNOWN;
        }
    }
    let r = bz.joint();
    let wp1 = w.0.add(
        &BigFloat::from_i32(1, 64),
        w.0.mantissa_max_bit_len().unwrap_or(64) + 64,
        RoundingMode::ToEven,
    );
    let l1w = accuracy::lg_abs_low(&(wp1, w.1.clone()));
    // log₂|W′(z)| = −Re W·log₂ e − log₂|1 + W|.
    let lder = -to_f64(&w.0) * std::f64::consts::LOG2_E - l1w;
    if !lder.is_finite() || lder + r > l1w.min(0.0) - 4.0 {
        return Bound::UNKNOWN;
    }
    bound(accuracy::lsum(lder + GROWTH + r + 1.0, own))
}
