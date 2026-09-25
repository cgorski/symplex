//! Error bounds for numerical evaluation.
//!
//! `evalf` evaluates an expression bottom-up at a fixed binary working
//! precision.  Alongside every value it keeps an estimate `2^e` of the
//! value's absolute error ([`node_error`]), propagated from the children's
//! bounds to first order:
//!
//! | node | error of the result |
//! |---|---|
//! | exact rational (dyadic, fits the precision), `i` | 0 |
//! | other rational, `π`, `e`, constants | rounding: `2^(mag − prec)` |
//! | `a₁ + … + aₙ` | `Σ err(aᵢ)` — absolute errors add, so cancellation shows as a result much smaller than its error |
//! | `a₁ · … · aₙ` | `Σ err(aᵢ)·Π_{j≠i} abs(aⱼ)` |
//! | `b^n`, `b^p`, `b^e` | relative errors scale by the exponent, plus `abs(ln b)·err(e)`; a negative power of a value indistinguishable from 0 has no bound |
//! | `exp z` | `abs(exp z)·err(z)` |
//! | `ln z` | `err(z)/abs(z)` |
//! | `sin`, `cos`, `sinh`, `cosh` | `max(1, abs(f(z)))·err(z)` |
//! | `tan`, `tanh` | `(1 + abs(f(z))²)·err(z)` |
//! | `atan`, `asin`, `acos`, `asinh`, `acosh`, `atanh` | `err(z)·abs(f′(z))`, from `1 ± z²` |
//! | `sign`, `floor`, `ceiling`, `heaviside`, `KroneckerDelta` | exact, or unknown when the argument (difference) is within its error of the threshold |
//! | special functions | the arguments' relative error carries over, plus 4 bits |
//! | definite integrals (`f64` quadrature, see `evalf.rs`) | their rounding, plus 16 bits |
//!
//! Nodes whose evaluator sees more than its children's values report their
//! own bound ([`reported`] adds the rounding, as [`node_error`] does):
//!
//! | node | error of the result |
//! |---|---|
//! | finite `Sum` / `Product` | the terms' bounds (evaluated with the index bound exactly), as for `+` / `·`, plus the rounding of every partial result |
//! | infinite `Sum` of a hypergeometric term | the first term's relative error carried by the recurrence, the recurrence's roundings, and a rigorous geometric bound on the tail (`hypsum.rs`) |
//! | `RootOf` | the radius of a certified inclusion disk of the root (`poly::roots::root_balls`); unknown when the disks do not isolate the roots |
//! | `RootSum` | the body's bound at each root, the root bounded by its inclusion disk, as for `+` |
//! | `Piecewise` | the chosen branch's bound when every condition up to it is decided with certainty (a comparison whose difference is outside its error ball, or of exact values); unknown otherwise, like `sign` at its threshold |
//! | physical constants | their value's bound |
//!
//! Every result also carries its own rounding.  These are estimates, not
//! proofs (interval arithmetic would need a rigorous bound for every special
//! function); they catch the loss that matters in practice — catastrophic
//! cancellation, and the amplification by `exp`, `tan` or a division — and
//! the evaluator re-evaluates at a higher precision until the requested
//! digits are covered (`evaluate_adaptive` in `evalf.rs`).

use astro_float::{BigFloat, RoundingMode};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_mul, c_one, c_sub};
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Base-2 exponent `e` of an absolute error bound: `|error| ≤ 2^e`.
pub(super) type ErrExp = i64;

/// The value is exact.
pub(super) const EXACT: ErrExp = i64::MIN / 4;

/// No bound is known: a division by a value indistinguishable from zero,
/// or a decision (`sign`, `floor`) at its threshold.
pub(super) const UNKNOWN: ErrExp = i64::MAX / 4;

/// The bound of a value that underflowed to 0: below the smallest positive
/// float.
const UNDERFLOW: ErrExp = astro_float::EXPONENT_MIN as ErrExp;

/// Is `e` the bound of an exact value (possibly after harmless arithmetic
/// on [`EXACT`])?
pub(super) fn is_exact(e: ErrExp) -> bool {
    e < EXACT / 2
}

/// Is `e` [`UNKNOWN`] (possibly after arithmetic)?
pub(super) fn is_unknown(e: ErrExp) -> bool {
    e > UNKNOWN / 2
}

fn clamp(e: ErrExp) -> ErrExp {
    e.clamp(EXACT, UNKNOWN)
}

/// `m` with `2^(m−1) ≤ |x| < 2^m`; `None` for zero.
fn part_mag(x: &BigFloat) -> Option<i64> {
    if x.is_zero() {
        None
    } else {
        x.exponent().map(i64::from)
    }
}

fn is_finite(z: &Complex) -> bool {
    !(z.0.is_inf() || z.0.is_nan() || z.1.is_inf() || z.1.is_nan())
}

/// The magnitude exponent of a complex value (the larger of its parts'),
/// `None` for zero.
pub(super) fn mag(z: &Complex) -> Option<i64> {
    match (part_mag(&z.0), part_mag(&z.1)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

/// An exponent bounding the *true* value: its magnitude or its error.
fn upper(z: &Complex, err: ErrExp) -> ErrExp {
    mag(z).map_or(err, |m| m.max(err))
}

/// Does the ball `z ± 2^err` contain 0?
pub(super) fn contains_zero(z: &Complex, err: ErrExp) -> bool {
    match mag(z) {
        None => true,
        Some(m) => m <= err.saturating_add(1),
    }
}

/// The accurate bits of `z ± 2^err` relative to `|z|` (`None` for a value
/// that is zero or indistinguishable from zero).
pub(super) fn accurate_bits(z: &Complex, err: ErrExp) -> Option<i64> {
    let m = mag(z)?;
    if is_exact(err) {
        return Some(i64::MAX / 8);
    }
    (m > err).then_some(m - err)
}

pub(super) fn rounding(z: &Complex, prec: usize) -> ErrExp {
    mag(z).map_or(EXACT, |m| m - prec as i64)
}

/// `⌈log₂ n⌉` (0 for `n ≤ 1`).
pub(super) fn ceil_log2(n: usize) -> i64 {
    i64::from(usize::BITS - n.saturating_sub(1).leading_zeros())
}

/// `⌈log₂|q|⌉`, for a non-zero rational (approximately).
fn log2_abs(q: &Q) -> i64 {
    i64::try_from(q.numer().bits()).unwrap_or(i64::MAX / 8)
        - i64::try_from(q.denom().bits()).unwrap_or(0)
        + 1
}

/// Is the rational exactly representable as a `prec`-bit binary float?
fn exactly_representable(q: &Q, prec: usize) -> bool {
    let d = q.denom();
    let power_of_two = (d & (d - BigInt::one())).is_zero();
    power_of_two && q.numer().bits() <= prec as u64
}

/// The magnitude exponent of `w(z)` at low precision (for condition
/// estimates), `None` if it is zero.
fn mag_of(z: &Complex, f: impl Fn(&Complex, &Complex) -> Complex) -> Option<i64> {
    const P: usize = 64;
    let z2 = c_mul(z, z, P, RoundingMode::ToEven);
    mag(&f(&c_one(P), &z2))
}

/// The error bound of `value`, computed at working precision `prec` by a
/// routine that bounds its own error by `2^err`: that bound plus the
/// rounding of the result, finished as [`node_error`] finishes a propagated
/// bound.
pub(super) fn reported(value: &Complex, err: ErrExp, prec: usize) -> ErrExp {
    if !is_finite(value) || is_unknown(err) {
        return UNKNOWN;
    }
    if is_exact(err) && mag(value).is_none() {
        return EXACT;
    }
    clamp(err.max(rounding(value, prec)).saturating_add(1))
}

/// The propagated bound of a product of factors `v ± 2^e`:
/// `Σ err(aᵢ)·Π_{j≠i} abs(aⱼ)`, times the number of factors (the roundings of
/// the partial products are the caller's).  An exact zero factor makes the
/// product exact.
pub(super) fn product_error<'a>(parts: impl IntoIterator<Item = (&'a Complex, ErrExp)>) -> ErrExp {
    let mut bounds: Vec<(ErrExp, ErrExp)> = Vec::new();
    for (v, e) in parts {
        if is_unknown(e) {
            return UNKNOWN;
        }
        if mag(v).is_none() && is_exact(e) {
            return EXACT; // an exact zero factor
        }
        bounds.push((e, upper(v, e)));
    }
    let total: ErrExp = bounds.iter().map(|&(_, u)| u).sum();
    let worst = bounds
        .iter()
        .map(|&(e, u)| if is_exact(e) { EXACT } else { e + total - u })
        .max()
        .unwrap_or(EXACT);
    worst.saturating_add(ceil_log2(bounds.len()))
}

/// The error bound of the value `value` of node `id`, from the values
/// (`cache`) and error bounds (`errs`) of its children, at working
/// precision `prec`.  See the module documentation.
pub(super) fn node_error(
    arena: &Arena,
    id: ExprId,
    value: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, ErrExp>,
    prec: usize,
) -> ErrExp {
    if !is_finite(value) {
        return UNKNOWN;
    }
    let child = |c: ExprId| -> Option<(&Complex, ErrExp)> {
        Some((cache.get(&c)?, errs.get(&c).copied().unwrap_or(UNKNOWN)))
    };
    let ub_out = mag(value).unwrap_or(EXACT);
    let propagated: ErrExp = match arena.node(id) {
        ExprNode::Num(nid) => {
            if exactly_representable(arena.num(*nid), prec) {
                return EXACT;
            }
            rounding(value, prec)
        }
        ExprNode::ImaginaryUnit | ExprNode::BoolTrue | ExprNode::BoolFalse => return EXACT,
        ExprNode::Pi | ExprNode::E | ExprNode::EulerGamma | ExprNode::Catalan => {
            rounding(value, prec)
        }
        ExprNode::GoldenRatio | ExprNode::PhysicalConstant(..) => rounding(value, prec),

        ExprNode::Add(children) => {
            let mut worst = EXACT;
            for &c in children.iter() {
                let Some((_, e)) = child(c) else {
                    return UNKNOWN;
                };
                worst = worst.max(e);
            }
            if is_unknown(worst) {
                return UNKNOWN;
            }
            worst.saturating_add(ceil_log2(children.len()))
        }
        ExprNode::Neg(c) | ExprNode::Conjugate(c) | ExprNode::Re(c) | ExprNode::Im(c) => {
            match child(*c) {
                Some((_, e)) => e,
                None => return UNKNOWN,
            }
        }
        ExprNode::Mul(children) => {
            let mut parts: Vec<(&Complex, ErrExp)> = Vec::with_capacity(children.len());
            for &c in children.iter() {
                let Some(part) = child(c) else {
                    return UNKNOWN;
                };
                parts.push(part);
            }
            product_error(parts)
        }
        ExprNode::Pow(base, exp) => {
            let (Some((b, eb)), Some((ev, ee))) = (child(*base), child(*exp)) else {
                return UNKNOWN;
            };
            if is_unknown(eb) || is_unknown(ee) {
                return UNKNOWN;
            }
            pow_error(arena.as_num(*exp), b, eb, ev, ee, ub_out)
        }

        // `d exp z = exp z · dz`: the argument's absolute error is the
        // result's relative error.  Before 0.29 the magnitude was clamped
        // at 1 (`ub_out.max(0)`), so `exp(−260.1) ≈ 2⁻³⁷⁵` carried the
        // absolute error of its argument, `2^(9 − prec)`: at the 384-bit cap
        // that ball contained 0, `ln(exp(−260.1))` had no bound and was
        // refused.
        ExprNode::Exp(c) => match child(*c) {
            Some((_, e)) if !is_unknown(e) => e.saturating_add(ub_out),
            _ => return UNKNOWN,
        },
        ExprNode::Ln(c) => match child(*c) {
            Some((z, e)) if !is_unknown(e) => {
                if is_exact(e) {
                    EXACT
                } else if contains_zero(z, e) {
                    return UNKNOWN;
                } else {
                    e - mag(z).unwrap_or(0) + 1
                }
            }
            _ => return UNKNOWN,
        },
        ExprNode::Sin(c) | ExprNode::Cos(c) | ExprNode::Sinh(c) | ExprNode::Cosh(c) => {
            match child(*c) {
                Some((_, e)) if !is_unknown(e) => e.saturating_add(ub_out.max(0) + 1),
                _ => return UNKNOWN,
            }
        }
        ExprNode::Tan(c) | ExprNode::Tanh(c) => match child(*c) {
            Some((_, e)) if !is_unknown(e) => e.saturating_add((2 * ub_out).max(0) + 1),
            _ => return UNKNOWN,
        },
        ExprNode::Atan(c)
        | ExprNode::Asin(c)
        | ExprNode::Acos(c)
        | ExprNode::Asinh(c)
        | ExprNode::Acosh(c)
        | ExprNode::Atanh(c) => {
            let Some((z, e)) = child(*c) else {
                return UNKNOWN;
            };
            if is_unknown(e) {
                return UNKNOWN;
            }
            if is_exact(e) {
                EXACT
            } else {
                // |f′(z)| = |w|^(−k) with w = 1 + z² (atan, asinh), 1 − z²
                // (asin, acos, atanh) or z² − 1 (acosh), k = 1 or ½.
                let (w, halve) = match arena.node(id) {
                    ExprNode::Atan(_) => (
                        mag_of(z, |one, z2| {
                            crate::base::bigcomplex::c_add(one, z2, 64, RoundingMode::ToEven)
                        }),
                        false,
                    ),
                    ExprNode::Asinh(_) => (
                        mag_of(z, |one, z2| {
                            crate::base::bigcomplex::c_add(one, z2, 64, RoundingMode::ToEven)
                        }),
                        true,
                    ),
                    ExprNode::Acosh(_) => (
                        mag_of(z, |one, z2| c_sub(z2, one, 64, RoundingMode::ToEven)),
                        true,
                    ),
                    ExprNode::Atanh(_) => (
                        mag_of(z, |one, z2| c_sub(one, z2, 64, RoundingMode::ToEven)),
                        false,
                    ),
                    _ => (
                        mag_of(z, |one, z2| c_sub(one, z2, 64, RoundingMode::ToEven)),
                        true,
                    ),
                };
                let Some(w) = w else {
                    return UNKNOWN; // at the singular point
                };
                let amplification = if halve { -w.div_euclid(2) } else { -w };
                e.saturating_add(amplification.max(0) + 1)
            }
        }
        ExprNode::Abs(c) => match child(*c) {
            Some((_, e)) => e.saturating_add(1),
            None => return UNKNOWN,
        },
        ExprNode::Arg(c) => match child(*c) {
            Some((z, e)) if !is_unknown(e) => {
                if is_exact(e) {
                    EXACT
                } else if contains_zero(z, e) {
                    return UNKNOWN;
                } else {
                    e - mag(z).unwrap_or(0) + 1
                }
            }
            _ => return UNKNOWN,
        },
        ExprNode::Atan2(y, x) => {
            let (Some((yv, ey)), Some((xv, ex))) = (child(*y), child(*x)) else {
                return UNKNOWN;
            };
            let worst = ey.max(ex);
            if is_unknown(worst) {
                return UNKNOWN;
            }
            if is_exact(worst) {
                EXACT
            } else {
                let r = mag(yv).max(mag(xv));
                match r {
                    Some(r) if r > worst + 1 => worst - r + 1,
                    _ => return UNKNOWN,
                }
            }
        }

        // Decisions: exact, unless the argument is within its error of the
        // threshold.
        ExprNode::Sign(c) | ExprNode::Heaviside(c) => match child(*c) {
            Some((z, e)) if !is_unknown(e) => {
                if z.1.is_zero() {
                    if is_exact(e) || !contains_zero(z, e) {
                        return EXACT; // −1, 0 or 1
                    }
                    return UNKNOWN;
                } else if contains_zero(z, e) {
                    return UNKNOWN;
                } else {
                    e - mag(z).unwrap_or(0) + 1
                }
            }
            _ => return UNKNOWN,
        },
        ExprNode::Floor(c) | ExprNode::Ceiling(c) => match child(*c) {
            Some((z, e)) if !is_unknown(e) => {
                if is_exact(e) {
                    return EXACT;
                } else {
                    // Distance from the argument to the integer it rounded
                    // to, and to the next one.
                    let p = 64 + prec;
                    let d1 = c_sub(z, value, p, RoundingMode::ToEven);
                    let d1 = mag(&d1);
                    let one = c_one(p);
                    let next = match arena.node(id) {
                        ExprNode::Floor(_) => {
                            crate::base::bigcomplex::c_add(value, &one, p, RoundingMode::ToEven)
                        }
                        _ => c_sub(value, &one, p, RoundingMode::ToEven),
                    };
                    let d2 = mag(&c_sub(&next, z, p, RoundingMode::ToEven));
                    let margin = d1.unwrap_or(i64::MIN).min(d2.unwrap_or(i64::MIN));
                    if d1.is_none() || d2.is_none() || margin <= e + 1 {
                        return UNKNOWN;
                    }
                    return EXACT; // an integer
                }
            }
            _ => return UNKNOWN,
        },
        ExprNode::Min(children) | ExprNode::Max(children) => {
            let mut worst = EXACT;
            for &c in children.iter() {
                match child(c) {
                    Some((_, e)) => worst = worst.max(e),
                    None => return UNKNOWN,
                }
            }
            worst
        }
        // A decision on `i − j = 0`: exact when both are exact or their
        // difference is outside its error ball.
        ExprNode::KroneckerDelta(i, j) => {
            let (Some((iv, ei)), Some((jv, ej))) = (child(*i), child(*j)) else {
                return UNKNOWN;
            };
            if is_exact(ei) && is_exact(ej) {
                return EXACT;
            }
            let d = c_sub(iv, jv, prec + 64, RoundingMode::ToEven);
            if contains_zero(&d, ei.max(ej).saturating_add(1)) {
                return UNKNOWN;
            }
            return EXACT;
        }

        // Special functions: the arguments' relative error carries over.
        ExprNode::Gamma(_)
        | ExprNode::LogGamma(_)
        | ExprNode::Digamma(_)
        | ExprNode::Erf(_)
        | ExprNode::Erfc(_)
        | ExprNode::LambertW(_)
        | ExprNode::Beta(_, _)
        | ExprNode::Si(_)
        | ExprNode::Ci(_)
        | ExprNode::Ei(_)
        | ExprNode::Li(_)
        | ExprNode::Zeta(_)
        | ExprNode::Polygamma(_, _)
        | ExprNode::Factorial(_)
        | ExprNode::Binomial(_, _)
        | ExprNode::Apply(_, _)
        | ExprNode::DiracDelta(_) => {
            let mut worst = EXACT;
            for c in arena.node(id).children() {
                let Some((v, e)) = child(c) else {
                    return UNKNOWN;
                };
                if is_unknown(e) {
                    return UNKNOWN;
                }
                if is_exact(e) {
                    continue;
                }
                worst = worst.max(match mag(v) {
                    // Relative error in, relative error out.
                    Some(m) if m > e => ub_out + (e - m) + 4,
                    // An argument indistinguishable from 0: absolute.
                    _ => e + ub_out.max(0) + 4,
                });
            }
            worst
        }

        // Evaluated by their own routines (series, quadrature, root
        // finding, branch selection): trust them to their precision.
        _ => rounding(value, prec).saturating_add(16),
    };
    if is_unknown(propagated) {
        return UNKNOWN;
    }
    if is_exact(propagated) && mag(value).is_none() {
        return if underflowed(arena, id, cache) {
            UNDERFLOW
        } else {
            EXACT
        };
    }
    clamp(propagated.max(rounding(value, prec)).saturating_add(1))
}

/// Is the zero value of node `id`, computed from exact children, an
/// underflow?  `exp` is never 0, nor a product or a power of non-zero
/// values.  (astro-float's `exp` returns 0 below its exponent range without
/// flagging it inexact, and before 0.29 that zero counted as exact:
/// `Piecewise((1, exp(−4·10⁹) > 0), (0, True))` came out `0`.)
fn underflowed(arena: &Arena, id: ExprId, cache: &FxHashMap<ExprId, Complex>) -> bool {
    let nonzero = |c: &ExprId| cache.get(c).is_some_and(|v| mag(v).is_some());
    match arena.node(id) {
        ExprNode::Exp(_) => true,
        ExprNode::Mul(children) => children.iter().all(nonzero),
        ExprNode::Pow(base, _) => nonzero(base),
        _ => false,
    }
}

/// The error bound of `b^x`, with `b ± 2^eb`, `x ± 2^ex`, the result's
/// magnitude exponent `ub_out` and `x`'s exact value when it is a rational
/// literal.
fn pow_error(
    x_literal: Option<&Q>,
    b: &Complex,
    eb: ErrExp,
    x: &Complex,
    ex: ErrExp,
    ub_out: ErrExp,
) -> ErrExp {
    if let Some(q) = x_literal {
        if q.is_zero() {
            return EXACT;
        }
        if is_exact(eb) {
            return EXACT; // only the rounding of the result
        }
        // Relative error scales by |q|: rel_out = |q|·rel_b.
        let log_q = log2_abs(q);
        if contains_zero(b, eb) {
            // A positive power of a value near 0 stays near 0; a negative
            // one is unbounded.
            return if q.is_positive() && q.is_integer() {
                let n = q.to_integer();
                let n = i64::try_from(n).unwrap_or(i64::MAX / 8);
                eb.saturating_mul(n).saturating_add(log_q)
            } else if q.is_positive() {
                // (ε)^p ≤ 2^(p·eb): p ≥ 1/2 for the powers met in practice.
                eb / 2 + 1
            } else {
                UNKNOWN
            };
        }
        let rel_b = eb - mag(b).unwrap_or(0);
        return ub_out.saturating_add(rel_b + log_q + 1);
    }
    // General exponent: b^x = exp(x·ln b), so
    // rel_out ≈ |x|·rel_b + |ln b|·err(x).
    if contains_zero(b, eb) && !is_exact(eb) {
        return UNKNOWN;
    }
    let rel_b = if is_exact(eb) {
        EXACT
    } else {
        eb - mag(b).unwrap_or(0)
    };
    let term_b = rel_b.saturating_add(upper(x, ex).max(0));
    // |ln b| ≤ |ln|b|| + π; ln|b| ≈ mag(b)·ln 2.
    let log_ln_b = {
        let m = mag(b).unwrap_or(0).unsigned_abs();
        let bound = (m as f64 * std::f64::consts::LN_2).max(std::f64::consts::PI) + 1.0;
        bound.log2().ceil() as i64
    };
    let term_x = if is_exact(ex) {
        EXACT
    } else {
        ex.saturating_add(log_ln_b)
    };
    ub_out.saturating_add(term_b.max(term_x) + 1)
}
