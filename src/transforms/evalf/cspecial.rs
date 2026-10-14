//! Which special-function nodes are evaluated by the complex routines
//! (`cexpint.rs`, `cbessel.rs`, `binomial` and the Fresnel integrals here),
//! and the error bounds of those nodes.
//!
//! A node goes to them when an argument is not real, or when its real
//! arguments lie where the value is not real and the real routine refuses:
//! `expint(n, x)`, `uppergamma(a, x)`, `lowergamma(a, x)` for `x < 0`,
//! `lowergamma(a, x)` for `a < 0`, `li(x)` for `x < 0`, `bessely`,
//! `besselk` of a negative argument and `besselj`, `besseli` of one with a
//! non-integer order (refused before 0.45: "expint of negative argument is
//! complex").  Every other real case stays with the real routine.
//!
//! # Error bounds
//!
//! The node's bound is its own `2^−(prec+12)` relative plus, for each
//! inexact argument of radius `r`, twice `r` times a bound on the derivative
//! over the argument's ball:
//!
//! | function | derivative in `z`, bounded over `abs(u − z) ≤ r` |
//! |---|---|
//! | `Ei`, `E1`-like | `abs(e^u/u) ≤ e^(x+r)/(abs(z) − r)` |
//! | `li` | `1/abs(ln u) ≤ 1/(abs(ln z) − r/(abs(z) − r))` |
//! | `Si`, `Shi` | `abs(sin u/u), abs(sinh u/u) ≤ min(sinh(abs(z)+r)/(abs(z)+r), cosh(abs(y or x)+r)/(abs(z)−r))` |
//! | `Ci`, `Chi` | `cosh(abs(y), resp. abs(x), + r)/(abs(z) − r)` |
//! | `E_n` | `(abs(n − 1)·abs(E_n) + e^(−x+r))/(abs(z) − r)` (from `E_n′ = ((n−1)E_n − e^(−z))/z`) |
//! | `Γ(a, z)`, `γ(a, z)` | `abs(z^(a−1)e^(−z))·e^(abs(a−1)·r/(abs(z)−r) + r)` |
//! | Fresnel | `abs(sin, cos(πu²/2)) ≤ cosh(π(abs(x)+r)(abs(y)+r))` |
//!
//! and numerically (`sensitivity::numeric_change`: the node evaluated again
//! with the argument moved) for the orders and parameters (`E_n` in `n`,
//! `Γ(a, z)` in `a`, the Bessel orders), the Bessel and Airy functions and
//! `binomial`.  A function with a cut on `(−∞, 0]` (`(−∞, 1]` for `li`) has
//! no bound when the argument's ball reaches the cut with an undecided side,
//! and every function singular at 0 none when the ball reaches within eight
//! radii of 0.  A value whose arguments are all exactly real and whose
//! imaginary part came out exactly 0 is real.
//!
//! # `erfinv`, `erfcinv` next to the ends of their domains
//!
//! `erfinv(1 − u) = erfcinv(u)`: an argument `1 − 10⁻¹⁰⁰⁰` rounds onto 1 below
//! 3,322 bits, and before 0.45 it was refused at every precision the
//! configuration allows.  When the argument is an exact rational within
//! `2^−10` of `±1` (of 2 for `erfcinv`), or a sum `±1 + t` (`2 + t`) whose
//! term `t` is that small, the complement `u` is formed exactly from the
//! rational, or is `∓t` with `t`'s own bound, and `erfcinv(u)` evaluated:
//! `erfinv(±(1 − u)) = ±erfcinv(u)`, `erfcinv(2 − u) = −erfcinv(u)`.  The
//! derivative in `u` is `(√π/2)·e^(x²) < 1/(2ux)` for `x = erfcinv(u)` (from
//! `erfc x < e^(−x²)/(x√π)`), over the ball `1/(2(u − r)x)`, with no bound
//! when the ball reaches within eight radii of 0.

use super::ball::CBall;
use super::wp::{RM, Wp};
use crate::base::bigfloat::{is_strictly_positive, to_f64};
use astro_float::{BigFloat, Consts};
use rustc_hash::FxHashMap;

use super::accuracy::{self, Bound, Cut};
use super::cexpint::{self, TrigInt};
use crate::base::arena::Arena;
use crate::base::bigcomplex::{Complex, c_div, c_mul, c_one, c_zero};
use crate::base::errors::SymplexError;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};

const LOG2E: f64 = std::f64::consts::LOG2_E;
const PI: f64 = std::f64::consts::PI;

/// A special function evaluated by the complex routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fun {
    Ei,
    Li,
    Trig(TrigInt),
    ExpInt,
    UpperGamma,
    LowerGamma,
    Bessel(super::cbessel::Kind),
    Airy(super::AiryKind),
    Binomial,
    FresnelS,
    FresnelC,
}

fn nonreal(cache: &FxHashMap<ExprId, Complex>, c: ExprId) -> bool {
    cache.get(&c).is_some_and(|v| !v.1.is_zero())
}

fn negative_real(cache: &FxHashMap<ExprId, Complex>, c: ExprId) -> bool {
    cache
        .get(&c)
        .is_some_and(|v| v.1.is_zero() && v.0.is_negative() && !v.0.is_zero())
}

fn integer_real(cache: &FxHashMap<ExprId, Complex>, c: ExprId) -> bool {
    cache
        .get(&c)
        .is_some_and(|v| v.1.is_zero() && (v.0.is_zero() || v.0.is_int()))
}

/// The function of `node` and its arguments, when the complex routines
/// evaluate it (module documentation).
pub(super) fn classify(
    arena: &Arena,
    node: &ExprNode,
    cache: &FxHashMap<ExprId, Complex>,
) -> Option<(Fun, Vec<ExprId>)> {
    use super::cbessel::Kind as B;
    let (f, args): (Fun, Vec<ExprId>) = match node {
        ExprNode::Ei(c) if nonreal(cache, *c) => (Fun::Ei, vec![*c]),
        ExprNode::Li(c) if nonreal(cache, *c) || negative_real(cache, *c) => (Fun::Li, vec![*c]),
        ExprNode::Si(c) if nonreal(cache, *c) => (Fun::Trig(TrigInt::Si), vec![*c]),
        ExprNode::Ci(c) if nonreal(cache, *c) => (Fun::Trig(TrigInt::Ci), vec![*c]),
        ExprNode::Binomial(n, k) if nonreal(cache, *n) || nonreal(cache, *k) => {
            (Fun::Binomial, vec![*n, *k])
        }
        ExprNode::Apply(sid, a) => {
            let f = arena.lib_fn(*sid)?;
            let any_nonreal = a.iter().any(|&c| nonreal(cache, c));
            let f = match (f, a.len()) {
                (LibFn::Shi, 1) if any_nonreal => Fun::Trig(TrigInt::Shi),
                (LibFn::Chi, 1) if any_nonreal => Fun::Trig(TrigInt::Chi),
                (LibFn::ExpInt, 2) if any_nonreal || negative_real(cache, a[1]) => Fun::ExpInt,
                (LibFn::UpperGamma, 2) if any_nonreal || negative_real(cache, a[1]) => {
                    Fun::UpperGamma
                }
                (LibFn::LowerGamma, 2)
                    if any_nonreal
                        || negative_real(cache, a[1])
                        || (negative_real(cache, a[0]) && !integer_real(cache, a[0])) =>
                {
                    Fun::LowerGamma
                }
                (LibFn::BesselJ | LibFn::BesselY | LibFn::BesselI | LibFn::BesselK, 2) => {
                    let kind = match f {
                        LibFn::BesselJ => B::J,
                        LibFn::BesselY => B::Y,
                        LibFn::BesselI => B::I,
                        _ => B::K,
                    };
                    let negative_cut = negative_real(cache, a[1])
                        && (matches!(kind, B::Y | B::K) || !integer_real(cache, a[0]));
                    if !(any_nonreal || negative_cut) {
                        return None;
                    }
                    Fun::Bessel(kind)
                }
                (LibFn::AiryAi, 1) if any_nonreal => Fun::Airy(super::AiryKind::Ai),
                (LibFn::AiryBi, 1) if any_nonreal => Fun::Airy(super::AiryKind::Bi),
                (LibFn::AiryAiPrime, 1) if any_nonreal => Fun::Airy(super::AiryKind::AiPrime),
                (LibFn::AiryBiPrime, 1) if any_nonreal => Fun::Airy(super::AiryKind::BiPrime),
                (LibFn::FresnelS, 1) if any_nonreal => Fun::FresnelS,
                (LibFn::FresnelC, 1) if any_nonreal => Fun::FresnelC,
                _ => return None,
            };
            (f, a.to_vec())
        }
        _ => return None,
    };
    args.iter()
        .all(|a| cache.contains_key(a))
        .then_some((f, args))
}

/// The value of `f` at `args` (rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus).
pub(super) fn eval(f: Fun, args: &[&Complex], w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    match f {
        Fun::Ei => cexpint::ei(args[0], w),
        Fun::Li => cexpint::li(args[0], w),
        Fun::Trig(kind) => cexpint::trig_integral(kind, args[0], w),
        Fun::ExpInt => cexpint::expint(args[0], args[1], w),
        Fun::UpperGamma => cexpint::upper_gamma(args[0], args[1], w),
        Fun::LowerGamma => cexpint::lower_gamma(args[0], args[1], w),
        Fun::Bessel(kind) => super::cbessel::bessel(kind, args[0], args[1], w),
        Fun::Airy(kind) => super::cbessel::airy(kind, args[0], w),
        Fun::Binomial => binomial(args[0], args[1], w),
        Fun::FresnelS => super::cerf::fresnel(true, args[0], w),
        Fun::FresnelC => super::cerf::fresnel(false, args[0], w),
    }
}

// ── binomial ────────────────────────────────────────────────────────────────

/// The parts of `z` in `f64` (a zero imaginary part as `+0`).
fn parts(z: &Complex) -> Option<(f64, f64)> {
    cexpint::f64_parts(z).map(|p| (p.x, p.y))
}

/// `x` as an integer exactly (`None` beyond `2^53` or not an integer).
fn exact_int(z: &Complex) -> Option<i64> {
    if !z.1.is_zero() || !(z.0.is_zero() || z.0.is_int()) {
        return None;
    }
    let v = to_f64(&z.0)?;
    (v.abs() < 9.0e15).then_some(v as i64)
}

/// Most factors of the falling factorial `a(a − 1)⋯(a − m + 1)/m!`.
const MAX_FALLING: i64 = 100_000;

/// `a(a − 1)⋯(a − m + 1)/m!`, `0 ≤ m ≤ MAX_FALLING`, to `p` bits relative.
fn falling_over_factorial(a: &Complex, m: i64, p: usize) -> Complex {
    let q = p + cexpint::guard(m as usize + 2);
    let mut v = c_one(q);
    for j in 0..m {
        let f = (super::zeta::plus_int(&a.0, -j, q), a.1.clone());
        v = c_mul(&v, &f, q, RM);
        let d = BigFloat::from_u64(j as u64 + 1, 64);
        v = (v.0.div(&d, q, RM), v.1.div(&d, q, RM));
    }
    v
}

/// `binomial(a, b)` with an argument not real: `Γ(a + 1)/(Γ(b + 1)Γ(a − b +
/// 1))`, or for an integer `b` (or `a − b`) `m ≥ 0` the falling factorial
/// `a(a−1)⋯(a−m+1)/m!`, `0` for a negative one (`1/Γ` vanishes at its poles:
/// mpmath's `binomial`).  Rounded to `prec + 16` bits, its error below
/// `2^−(prec+12)` of its modulus.
fn binomial(a: &Complex, b: &Complex, w: &mut Wp<'_>) -> Result<Complex, SymplexError> {
    let prec = w.bits();
    let p = prec + 24;
    let exact = super::exact_bits(&a.0, p).max(super::exact_bits(&b.0, p));
    let amb = (a.0.sub(&b.0, exact, RM), a.1.sub(&b.1, p + 64, RM));
    for m in [exact_int(b), exact_int(&amb)].into_iter().flatten() {
        if m < 0 {
            return Ok(c_zero(prec + 16));
        }
        if m <= MAX_FALLING {
            let v = falling_over_factorial(a, m, p);
            return Ok((
                super::round_to(v.0, prec + 16),
                super::round_to(v.1, prec + 16),
            ));
        }
    }
    let gamma = |z: &Complex, cc: &mut Consts| -> Result<Complex, SymplexError> {
        if z.1.is_zero() {
            let zf = to_f64(&z.0).unwrap_or(f64::INFINITY).abs();
            let g = super::magnitude_bits((zf + 2.0) * ((zf + 2.0).ln() + 1.0));
            let v = super::arb_gamma_real(&z.0, &mut Wp::new(p + g + 16, cc))?;
            Ok((v, BigFloat::new(p)))
        } else {
            super::zeta::gamma_complex(z, &mut Wp::new(p, cc))
        }
    };
    let one = BigFloat::from_i32(1, 64);
    let a1 = (a.0.add(&one, super::exact_bits(&a.0, p), RM), a.1.clone());
    let b1 = (b.0.add(&one, super::exact_bits(&b.0, p), RM), b.1.clone());
    let amb1 = (
        amb.0.add(&one, super::exact_bits(&amb.0, p), RM),
        amb.1.clone(),
    );
    let ga = gamma(&a1, w.cc)?;
    let gb = gamma(&b1, w.cc)?;
    let gab = gamma(&amb1, w.cc)?;
    let den = c_mul(&gb, &gab, p, RM);
    if den.0.is_zero() && den.1.is_zero() {
        return Err(super::unevaluable(
            "binomial: Gamma(b + 1)·Gamma(a − b + 1) is zero",
        ));
    }
    let v = c_div(&ga, &den, p, RM);
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Err(super::unevaluable("binomial beyond the exponent range"));
    }
    Ok((
        super::round_to(v.0, prec + 16),
        super::round_to(v.1, prec + 16),
    ))
}

// ── Error bounds ────────────────────────────────────────────────────────────

/// `log₂ cosh t`, also for large `t`.
fn log2_cosh(t: f64) -> f64 {
    let t = t.abs();
    if t > 30.0 {
        t * LOG2E - 1.0
    } else {
        t.cosh().log2()
    }
}

/// `log₂(sinh t/t)` for `t ≥ 0`.
fn log2_sinhc(t: f64) -> f64 {
    if t < 1e-8 {
        0.0
    } else if t > 30.0 {
        t * LOG2E - 1.0 - t.log2()
    } else {
        (t.sinh() / t).log2()
    }
}

/// The bound on `log₂ abs(f′)` over the ball of radius `r` around `z = x + iy`
/// (the closed forms of the module documentation), `None` without one
/// (then the derivative is measured numerically), `Some(+∞)` where the ball
/// reaches a singular point.
fn derivative(f: Fun, vals: &[&Complex], v: &Complex, r: f64) -> Option<f64> {
    let z = vals.last()?;
    let (x, y) = parts(z)?;
    let az = x.hypot(y);
    // Within eight radii of 0 (or a NaN): no bound.
    let near0 = (r * 8.0).partial_cmp(&az) != Some(std::cmp::Ordering::Less);
    let up = |l: f64| l + 1e-9 * l.abs() + 1e-9;
    let d = match f {
        Fun::Ei => {
            if near0 {
                return Some(f64::INFINITY);
            }
            (x + r) * LOG2E - (az - r).log2()
        }
        Fun::Li => {
            if near0 {
                return Some(f64::INFINITY);
            }
            let lnz = (az.ln()).hypot(y.atan2(x));
            let move_ln = r / (az - r) * 1.0001;
            let low = lnz - move_ln;
            let well_inside = low > 0.0 && 8.0 * move_ln <= lnz;
            if !well_inside {
                return Some(f64::INFINITY);
            }
            -low.log2()
        }
        Fun::Trig(kind) => {
            let across = match kind {
                TrigInt::Si | TrigInt::Ci => y.abs(),
                _ => x.abs(),
            };
            let far = if near0 {
                f64::INFINITY
            } else {
                log2_cosh(across + r) - (az - r).log2()
            };
            match kind {
                TrigInt::Si | TrigInt::Shi => far.min(log2_sinhc(az + r)),
                _ => far,
            }
        }
        Fun::ExpInt => {
            if near0 {
                return Some(f64::INFINITY);
            }
            let n = vals[0];
            let (nx, ny) = parts(n)?;
            let nm1 = (nx - 1.0).hypot(ny);
            let lv = accuracy::lg_abs(v);
            let a = if nm1 == 0.0 || lv == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                nm1.log2() + lv + (r * (1.0 + nm1 / (az - r)) * 2.0) * LOG2E
            };
            let b = (r - x) * LOG2E;
            lsum(a, b) - (az - r).log2()
        }
        Fun::UpperGamma | Fun::LowerGamma => {
            if near0 {
                return Some(f64::INFINITY);
            }
            let (ax, ay) = parts(vals[0])?;
            let am1 = (ax - 1.0).hypot(ay);
            let theta = y.atan2(x);
            // log₂ abs(z^(a−1) e^(−z)).
            let base = ((ax - 1.0) * az.ln() - ay * theta - x) * LOG2E;
            base + (am1 * r / (az - r) * 1.0001 + r) * LOG2E
        }
        Fun::FresnelS | Fun::FresnelC => log2_cosh(PI * (x.abs() + r) * (y.abs() + r)),
        Fun::Bessel(_) | Fun::Airy(_) | Fun::Binomial => return None,
    };
    Some(up(d))
}

// ── erfinv, erfcinv next to the ends ─────────────────────────────────────────────────

/// The complement of the argument of an `erfinv`/`erfcinv` node next to an
/// end of its domain (module documentation): `u`, its error exponent, and
/// whether the value is `−erfcinv(u)`.
struct EndComplement {
    u: BigFloat,
    err: f64,
    negate: bool,
}

/// Largest complement taken this way (`2^−10`).
const END_COMPLEMENT: f64 = 1.0 / 1024.0;

fn end_complement(
    arena: &Arena,
    node: &ExprNode,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    prec: usize,
) -> Option<EndComplement> {
    use crate::base::numeric::Q;
    let ExprNode::Apply(sid, args) = node else {
        return None;
    };
    let inverse_erfc = match arena.lib_fn(*sid)? {
        LibFn::ErfInv => false,
        LibFn::ErfcInv => true,
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }
    let small =
        |u: &BigFloat| is_strictly_positive(u) && to_f64(u).is_some_and(|f| f < END_COMPLEMENT);
    match arena.node(args[0]) {
        ExprNode::Num(n) => {
            let q = arena.num(*n);
            let one = Q::from_integer(1.into());
            let (c, negate) = if inverse_erfc {
                (Q::from_integer(2.into()) - q, true)
            } else if *q.numer() < 0.into() {
                (one + q, true)
            } else {
                (one - q, false)
            };
            let u = super::ratio_to_bigfloat(&c, prec + 64);
            if !small(&u) {
                return None;
            }
            let err = accuracy::part_lg(&u) - (prec + 64) as f64 + 1.0;
            Some(EndComplement { u, err, negate })
        }
        ExprNode::Add(children) if children.len() == 2 => {
            // ±1 + t (erfinv) or 2 + t (erfcinv), with the constant exact.
            let (k, t) = match (arena.as_num(children[0]), arena.as_num(children[1])) {
                (Some(k), None) => (k.clone(), children[1]),
                (None, Some(k)) => (k.clone(), children[0]),
                _ => return None,
            };
            let tv = cache.get(&t)?;
            let tb = errs.get(&t).copied()?;
            if tb.is_unknown() || !accuracy::exactly_real(tv, tb) {
                return None;
            }
            let (u, negate) = if inverse_erfc && k == Q::from_integer(2.into()) {
                (tv.0.neg(), true)
            } else if !inverse_erfc && k == Q::from_integer(1.into()) {
                (tv.0.neg(), false)
            } else if !inverse_erfc && k == Q::from_integer((-1).into()) {
                (tv.0.clone(), true)
            } else {
                return None;
            };
            small(&u).then_some(EndComplement {
                u,
                err: tb.re,
                negate,
            })
        }
        _ => None,
    }
}

/// `erfinv`/`erfcinv` of node `node` next to an end of its domain, with its
/// bound (module documentation), or `None` when the argument is not of that
/// form.  (`erfcinv(u) > 2.3` for `u < 2^−10`.)
pub(super) fn inverse_erf_end(
    arena: &Arena,
    node: &ExprNode,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    w: &mut Wp<'_>,
) -> Option<Result<CBall, SymplexError>> {
    let prec = w.bits();
    let end = end_complement(arena, node, cache, errs, prec)?;
    let lu = accuracy::part_lg(&end.u);
    let well_inside = end.err + 3.0 <= lu;
    if !accuracy::is_exact(end.err) && !well_inside {
        // The ball reaches within eight radii of 0: no bound yet.
        return Some(Ok(CBall::new(super::c_zero(prec), Bound::UNKNOWN)));
    }
    let x = match super::arb_erfcinv_small(&end.u, &mut w.at(prec + 8)) {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let lx = accuracy::part_lg(&x);
    let own = lx - prec as f64 + 1.0;
    let total = if accuracy::is_exact(end.err) {
        own
    } else {
        // 1/(2(u − r)x), with a quarter more for the change of x over the
        // ball.
        // In log₂ (u may lie far below the `f64` range).
        let xf = to_f64(&x).unwrap_or(0.0);
        let shrink = (1.0 - (end.err - lu).exp2()).log2();
        if !(xf > 1.0 && shrink.is_finite()) {
            return Some(Ok(CBall::new(super::c_zero(prec), Bound::UNKNOWN)));
        }
        let d = -(1.0 + lu + shrink + xf.log2()) + 0.33;
        accuracy::lsum(own, d + end.err + 1.0)
    };
    let v = super::round_to(x, prec);
    let v = if end.negate { v.neg() } else { v };
    Some(Ok(CBall::new((v, BigFloat::new(prec)), Bound::real(total))))
}

/// `log₂(2^a + 2^b)`.
fn lsum(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == f64::NEG_INFINITY || !hi.is_finite() {
        return hi;
    }
    hi + (lo - hi).exp2().ln_1p() * LOG2E
}

/// The cut of `f` in its last argument, given the other arguments (none for
/// an entire function or an integer order of `J`, `I`).
fn cut_of(f: Fun, vals: &[&Complex]) -> Option<Cut> {
    use super::cbessel::Kind as B;
    let int = |z: &Complex| exact_int(z).is_some();
    match f {
        Fun::Trig(TrigInt::Si | TrigInt::Shi)
        | Fun::Airy(_)
        | Fun::Binomial
        | Fun::FresnelS
        | Fun::FresnelC => None,
        Fun::Li => Some(Cut::RealBelowOne),
        Fun::Bessel(B::J | B::I) if int(vals[0]) => None,
        Fun::UpperGamma | Fun::LowerGamma if exact_int(vals[0]).is_some_and(|n| n > 0) => None,
        _ => Some(Cut::NegativeReal),
    }
}

/// The error bound of node `id` (function `f` of `args`, value `v`).
#[allow(clippy::too_many_arguments)]
pub(super) fn bound(
    arena: &Arena,
    id: ExprId,
    f: Fun,
    args: &[ExprId],
    v: &Complex,
    cache: &FxHashMap<ExprId, Complex>,
    errs: &FxHashMap<ExprId, Bound>,
    w: &mut Wp<'_>,
) -> Bound {
    let prec = w.bits();
    if [&v.0, &v.1].iter().any(|x| x.is_nan() || x.is_inf()) {
        return Bound::UNKNOWN;
    }
    let mut vals: Vec<&Complex> = Vec::with_capacity(args.len());
    let mut bs: Vec<Bound> = Vec::with_capacity(args.len());
    for a in args {
        let (Some(val), Some(b)) = (cache.get(a), errs.get(a)) else {
            return Bound::UNKNOWN;
        };
        if b.is_unknown() {
            return Bound::UNKNOWN;
        }
        vals.push(val);
        bs.push(*b);
    }
    let lv = accuracy::lg_abs(v);
    let own = lv - prec as f64 - 12.0;
    let all_real = vals
        .iter()
        .zip(&bs)
        .all(|(z, b)| accuracy::exactly_real(z, *b));
    let finish = |e: f64| {
        if all_real && v.1.is_zero() {
            Bound::real(e)
        } else {
            Bound::both(e)
        }
    };
    if bs.iter().all(|b| b.is_exact()) {
        return finish(own);
    }
    // The side of the cut must be decided.
    let last = vals.len() - 1;
    if let Some(cut) = cut_of(f, &vals)
        && accuracy::cut_undecided(vals[last], bs[last], cut)
    {
        return Bound::UNKNOWN;
    }
    let mut total = own;
    for (i, a) in args.iter().enumerate() {
        if bs[i].is_exact() || args[..i].contains(a) {
            continue;
        }
        let e = bs[i].joint();
        let closed = if i == last {
            derivative(f, &vals, v, e.exp2())
        } else {
            None
        };
        let c = match closed {
            Some(d) if d.is_finite() => d + e + 1.0,
            Some(_) => return Bound::UNKNOWN,
            None => match super::sensitivity::numeric_change(arena, id, *a, e, v, cache, w) {
                Some(k) => k,
                None => return Bound::UNKNOWN,
            },
        };
        total = accuracy::lsum(total, c);
    }

    if accuracy::is_unknown(total) {
        return Bound::UNKNOWN;
    }
    finish(total)
}
