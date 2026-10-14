//! A value together with its error bound: [`CBall`], the complex number
//! `mid.0 ± 2^err.re` + i·(`mid.1 ± 2^err.im`) that `evalf` computes for
//! every node (see `accuracy` for what the bound means and how each node
//! propagates it).
//!
//! It replaces the `(Complex, Bound)` tuples that carried a value and its
//! bound side by side through the evaluator, its special-function routines
//! and its sums.  A tuple let the two be separated, swapped or rebuilt from
//! different sources; a `CBall` names them, and the questions that need
//! both — the sign of a part, the side of a branch cut — are its methods,
//! answered from the ball rather than from the midpoint.  See
//! CONTRIBUTING.md, "Tuples versus structs".
//!
//! A [`Ball`] owns its value ([`CBall`]) or borrows it (`Ball<&Complex>`:
//! a node's cached value with its bound, a view that copies no number).
//! The bound rules of the nodes — the elementary functions, powers, the
//! inverse trigonometric and hyperbolic functions, `atan2`, the decisions
//! (`sign`, `floor`, `min`, `KroneckerDelta`, …) and the special functions —
//! are `Ball` methods next to `accuracy::node_error`, which hands them the
//! borrowed balls of the children.
//!
//! The evaluated nodes are one map of balls, [`Balls`], read through the
//! view [`Cache`]: a node's value, its bound, or both.  Up to 0.49 the
//! values and the bounds were two maps passed side by side through every
//! rule (`cache` and `errs`), which could disagree on which nodes they
//! held.
//!
//! [`RBall`] is the real ball `m ± 2^r` with its own arithmetic (the
//! asymptotic expansions of `asymsum.rs` compute with it).
//!
//! # Which function
//!
//! | Need | Function |
//! |---|---|
//! | a value with a bound | [`CBall::new`]; [`CBall::exact`], [`CBall::unknown`] |
//! | the value, the bound, or both, of an evaluated node | [`Cache::get`], [`Cache::err`] ([`Cache::bound`]), [`Cache::ball`] |
//! | the same values with every bound hidden | [`Cache::values_only`] |
//! | a real ball and its arithmetic | [`RBall`]: [`RBall::add`], [`RBall::mul`], [`RBall::recip`], … |
//! | `a₁ + … + aₙ`, `a₁ · … · aₙ` at `prec` bits, with the bound | [`CBall::sum`], [`CBall::product`] (`accuracy::add_with_bound`, `accuracy::mul_with_bound`) |
//! | the sign of the real or imaginary part | [`Ball::sign_re`], [`Ball::sign_im`] |
//! | which side of a branch cut the ball lies on | [`Ball::cut_side`] |
//! | an exactly real value (imaginary part exactly 0) | [`Ball::is_exactly_real`] |
//!
//! # Corner cases
//!
//! - **Signed zeros.** astro-float's `+0` answers `is_positive()`; a sign
//!   read from a midpoint therefore called an exact 0 positive (`evalf`
//!   once decided `a > a` that way).  [`Ball::sign_re`] of an
//!   exact 0 is [`SignSet::ZERO`], of a part that is 0 or within its error
//!   of 0 the whole [`SignSet::REAL`], and a definite sign only for a part
//!   whose ball excludes 0.
//! - **No bound.** A part with an unknown bound has every real sign, and
//!   its side of a cut is undecided whenever its line meets the cut.
//! - **Not finite.** A NaN or infinite midpoint has no sign
//!   ([`SignSet::REAL`]).
//! - **A value without a bound.** A node can have a value and no bound:
//!   in a view with the bounds hidden ([`Cache::values_only`]), or in the
//!   two maps of `quad.rs`.  [`Cache::bound`] is then `None` and
//!   [`Cache::err`] [`Bound::UNKNOWN`]; a rule that needs a bound gives none.

use std::borrow::Borrow;

use super::accuracy::{self, Bound, Cut, CutSide};
use super::wp::RM;
use crate::base::bigcomplex::Complex;
use crate::base::node::ExprId;
use crate::base::numeric::{Q, ratio_to_bigfloat};
use crate::base::truth::SignSet;
use astro_float::BigFloat;
use num_traits::Zero;
use rustc_hash::FxHashMap;

/// A complex value and the error bound of each of its parts (see the
/// module documentation), the value owned ([`CBall`]) or borrowed
/// (`Ball<&Complex>`, a node's cached value with its bound: a view that
/// costs no copy of the number).
///
/// ```ignore
/// let x = CBall::exact((BigFloat::from_i32(-2, 64), BigFloat::new(64)));
/// assert_eq!(x.sign_re(), SignSet::NEGATIVE);
/// assert_eq!(x.sign_im(), SignSet::ZERO);
/// assert_eq!(x.cut_side(Cut::NegativeReal), CutSide::On);
/// let view = Ball { mid: &x.mid, err: x.err };
/// assert!(view.is_exactly_real());
/// ```
#[derive(Clone, Copy, Debug)]
pub(super) struct Ball<Z = Complex> {
    /// The computed value `re + i·im`.
    pub(super) mid: Z,
    /// The error bound of each part.
    pub(super) err: Bound,
}

/// A ball that owns its value.
pub(super) type CBall = Ball<Complex>;

impl CBall {
    /// `mid` with the bound `err`.
    pub(super) fn new(mid: Complex, err: Bound) -> CBall {
        CBall { mid, err }
    }

    /// `mid`, exactly.
    pub(super) fn exact(mid: Complex) -> CBall {
        CBall {
            mid,
            err: Bound::EXACT,
        }
    }

    /// `mid` with no bound.
    pub(super) fn unknown(mid: Complex) -> CBall {
        CBall {
            mid,
            err: Bound::UNKNOWN,
        }
    }

    /// `a₁ + … + aₙ` summed left to right from 0 at `prec` bits, with its
    /// bound (`accuracy::add_with_bound`).
    pub(super) fn sum(terms: &[&CBall], prec: usize) -> CBall {
        let parts: Vec<(&Complex, Bound)> = terms.iter().map(|b| (&b.mid, b.err)).collect();
        accuracy::add_with_bound(&parts, prec)
    }

    /// `a₁ · … · aₙ` multiplied left to right from 1 at `prec` bits, with
    /// its bound (`accuracy::mul_with_bound`).
    pub(super) fn product(factors: &[&CBall], prec: usize) -> CBall {
        let parts: Vec<(&Complex, Bound)> = factors.iter().map(|b| (&b.mid, b.err)).collect();
        accuracy::mul_with_bound(&parts, prec)
    }
}

impl<Z: Borrow<Complex>> Ball<Z> {
    /// The value.
    pub(super) fn z(&self) -> &Complex {
        self.mid.borrow()
    }

    /// The signs the real part may have.
    pub(super) fn sign_re(&self) -> SignSet {
        part_sign(&self.z().0, self.err.re)
    }

    /// The signs the imaginary part may have ([`SignSet::ZERO`] exactly
    /// for an exactly real value).
    pub(super) fn sign_im(&self) -> SignSet {
        part_sign(&self.z().1, self.err.im)
    }

    /// Where the ball lies relative to the branch cut `cut`
    /// (`accuracy::cut_side`).
    pub(super) fn cut_side(&self, cut: Cut) -> CutSide {
        accuracy::cut_side(self.z(), self.err, cut)
    }

    /// Is the imaginary part exactly 0?
    pub(super) fn is_exactly_real(&self) -> bool {
        self.sign_im() == SignSet::ZERO
    }
}

/// The value and bound of every evaluated node.
pub(super) type Balls = FxHashMap<ExprId, CBall>;

/// The evaluated nodes as a rule reads them: a node's value
/// ([`Cache::get`]), its bound ([`Cache::err`]) or both ([`Cache::ball`]).
/// A view, copied freely, of a [`Balls`] map — with or without its bounds
/// — or of values and bounds held in two maps.
///
/// ```ignore
/// let mut balls = Balls::default();
/// balls.insert(x, CBall::new(two, Bound::real(-100.0)));
/// let cache = Cache::Balls(&balls);
/// assert_eq!(cache.err(&x).re, -100.0);
/// assert!(cache.values_only().bound(&x).is_none());
/// assert!(cache.values_only().err(&x).is_unknown());
/// ```
#[derive(Clone, Copy, Debug)]
pub(super) enum Cache<'a> {
    /// The values with their bounds.
    Balls(&'a Balls),
    /// The values only: every bound hidden.
    Values(&'a Balls),
    /// Values and bounds (`None`: hidden) in two maps — quadrature's
    /// integrand (`quad.rs`) keeps them so.
    Split(
        &'a FxHashMap<ExprId, Complex>,
        Option<&'a FxHashMap<ExprId, Bound>>,
    ),
}

impl<'a> Cache<'a> {
    /// The value of `id`, if it was evaluated.
    pub(super) fn get(self, id: &ExprId) -> Option<&'a Complex> {
        match self {
            Cache::Balls(m) | Cache::Values(m) => m.get(id).map(|b| &b.mid),
            Cache::Split(v, _) => v.get(id),
        }
    }

    /// Was `id` evaluated?
    pub(super) fn contains_key(self, id: &ExprId) -> bool {
        self.get(id).is_some()
    }

    /// The bound of `id`: `None` when it was not evaluated or its bound is
    /// hidden.
    pub(super) fn bound(self, id: &ExprId) -> Option<Bound> {
        match self {
            Cache::Balls(m) => m.get(id).map(|b| b.err),
            Cache::Values(_) | Cache::Split(_, None) => None,
            Cache::Split(_, Some(e)) => e.get(id).copied(),
        }
    }

    /// The bound of `id`, [`Bound::UNKNOWN`] when none is known.
    pub(super) fn err(self, id: &ExprId) -> Bound {
        self.bound(id).unwrap_or(Bound::UNKNOWN)
    }

    /// The value of `id` with its bound ([`Cache::err`]), if it was
    /// evaluated: a view.
    pub(super) fn ball(self, id: &ExprId) -> Option<Ball<&'a Complex>> {
        match self {
            Cache::Balls(m) => m.get(id).map(|b| Ball {
                mid: &b.mid,
                err: b.err,
            }),
            _ => Some(Ball {
                mid: self.get(id)?,
                err: self.err(id),
            }),
        }
    }

    /// The same values with every bound hidden.
    pub(super) fn values_only(self) -> Cache<'a> {
        match self {
            Cache::Balls(m) | Cache::Values(m) => Cache::Values(m),
            Cache::Split(v, _) => Cache::Split(v, None),
        }
    }
}

/// A real number `m ± 2^r` (`r = −∞`: exact) with its arithmetic at a
/// working precision `wp`: each operation adds the rounding of its result,
/// an ulp at its magnitude.  The asymptotic expansions (`asymsum.rs`) and
/// the hypergeometric series (`hypsum.rs`) compute with it.  (Up to 0.49
/// it was private to `asymsum.rs`.)
///
/// ```ignore
/// let third = RBall::from_q(&Q::new(1.into(), 3.into()), 64);
/// let one = third.mul(&RBall::from_int(3, 64), 64);
/// assert!(one.certainly_positive());
/// assert!(one.r > -66.0 && one.r < -60.0);
/// assert!(RBall::zero(64).exact_zero());
/// ```
#[derive(Clone, Debug)]
pub(super) struct RBall {
    /// The midpoint.
    pub(super) m: BigFloat,
    /// `log₂` of the radius.
    pub(super) r: accuracy::ErrExp,
}

impl RBall {
    /// An exact 0 at `wp` bits.
    pub(super) fn zero(wp: usize) -> RBall {
        RBall {
            m: BigFloat::new(wp),
            r: f64::NEG_INFINITY,
        }
    }

    /// Is it exactly 0?
    pub(super) fn exact_zero(&self) -> bool {
        self.m.is_zero() && self.r == f64::NEG_INFINITY
    }

    /// The rational `q` at `wp` bits: exact when it is representable,
    /// otherwise with two ulps of rounding.
    pub(super) fn from_q(q: &Q, wp: usize) -> RBall {
        let m = ratio_to_bigfloat(q, wp, RM);
        let r = if q.is_zero() || accuracy::exactly_representable(q, wp) {
            f64::NEG_INFINITY
        } else {
            two_ulps(&m, wp)
        };
        RBall { m, r }
    }

    /// The integer `n`, exactly.
    pub(super) fn from_int(n: i64, wp: usize) -> RBall {
        RBall {
            m: BigFloat::from_i64(n, wp),
            r: f64::NEG_INFINITY,
        }
    }

    /// `log₂` of an upper bound on the absolute value.
    pub(super) fn hi(&self) -> f64 {
        accuracy::lsum(accuracy::part_lg(&self.m), self.r)
    }

    /// `log₂` of a lower bound on the absolute value (`−∞` when the ball
    /// holds 0).
    pub(super) fn lo(&self) -> f64 {
        let l = accuracy::part_lg_low(&self.m);
        if self.r == f64::NEG_INFINITY {
            return l;
        }
        if l.is_nan() || l <= self.r + 1e-9 {
            return f64::NEG_INFINITY;
        }
        l + (1.0 - (self.r - l).exp2()).log2() - 1e-12
    }

    /// Does the ball exclude 0?
    pub(super) fn certainly_nonzero(&self) -> bool {
        self.lo() > f64::NEG_INFINITY
    }

    /// Does the ball lie right of 0?
    pub(super) fn certainly_positive(&self) -> bool {
        self.certainly_nonzero() && super::bf_strictly_positive(&self.m)
    }

    /// `self + o` at `wp` bits.
    pub(super) fn add(&self, o: &RBall, wp: usize) -> RBall {
        if self.exact_zero() {
            return o.clone();
        }
        if o.exact_zero() {
            return self.clone();
        }
        let m = self.m.add(&o.m, wp, RM);
        let r = accuracy::lsum(accuracy::lsum(self.r, o.r), ulp(&m, wp));
        RBall { m, r }
    }

    /// `−self`, exactly.
    pub(super) fn neg(&self) -> RBall {
        RBall {
            m: self.m.neg(),
            r: self.r,
        }
    }

    /// `self · o` at `wp` bits ([`accuracy::product_rule`]).
    pub(super) fn mul(&self, o: &RBall, wp: usize) -> RBall {
        if self.exact_zero() || o.exact_zero() {
            return RBall::zero(wp);
        }
        let m = self.m.mul(&o.m, wp, RM);
        let (la, lb) = (accuracy::part_lg(&self.m), accuracy::part_lg(&o.m));
        let r = accuracy::lsum(accuracy::product_rule(la, self.r, lb, o.r), ulp(&m, wp));
        RBall { m, r }
    }

    /// `1/self` at `wp` bits; `None` when the ball holds 0.
    pub(super) fn recip(&self, wp: usize) -> Option<RBall> {
        let lo = self.lo();
        if lo == f64::NEG_INFINITY {
            return None;
        }
        let m = BigFloat::from_i32(1, 64).div(&self.m, wp, RM);
        // |1/(m+δ) − 1/m| ≤ |δ|/(|m|·(|m| − |δ|)).
        let r = if self.r == f64::NEG_INFINITY {
            ulp(&m, wp)
        } else {
            accuracy::lsum(self.r - lo - accuracy::part_lg_low(&self.m), ulp(&m, wp))
        };
        Some(RBall { m, r })
    }
}

/// The rounding of a `wp`-bit result `m`: an ulp at its magnitude.
pub(super) fn ulp(m: &BigFloat, wp: usize) -> accuracy::ErrExp {
    accuracy::part_mag(m).map_or(f64::NEG_INFINITY, |e| (e - wp as i64) as f64)
}

/// Two ulps of a `wp`-bit result `m`: the rounding of a conversion or of a
/// transcendental function.
pub(super) fn two_ulps(m: &BigFloat, wp: usize) -> accuracy::ErrExp {
    accuracy::part_mag(m).map_or(f64::NEG_INFINITY, |e| (e - wp as i64 + 1) as f64)
}

/// The signs of the part `x ± 2^e`: [`SignSet::ZERO`] for an exact 0, one
/// sign for a ball that excludes 0, otherwise [`SignSet::REAL`].
fn part_sign(x: &BigFloat, e: accuracy::ErrExp) -> SignSet {
    if accuracy::exact_zero(x, e) {
        return SignSet::ZERO;
    }
    if x.is_nan() || x.is_inf() || accuracy::is_unknown(e) || accuracy::part_contains_zero(x, e) {
        return SignSet::REAL;
    }
    if x.is_negative() {
        SignSet::NEGATIVE
    } else {
        SignSet::POSITIVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::evalf::accuracy::{EXACT, UNKNOWN};

    fn c(re: i32, im: i32) -> Complex {
        (BigFloat::from_i32(re, 64), BigFloat::from_i32(im, 64))
    }

    #[test]
    fn exact_zero_is_zero_not_positive() {
        let z = CBall::exact(c(0, 0));
        assert_eq!(z.sign_re(), SignSet::ZERO);
        assert_eq!(z.sign_im(), SignSet::ZERO);
        // A zero midpoint with an error: either sign.
        let r = CBall::new(c(0, 0), Bound::both(-100.0));
        assert_eq!(r.sign_re(), SignSet::REAL);
        // A nonzero midpoint inside its error ball: either sign.
        let s = CBall::new(c(1, 0), Bound::real(2.0));
        assert_eq!(s.sign_re(), SignSet::REAL);
        assert_eq!(s.sign_im(), SignSet::ZERO);
        let t = CBall::new(c(-3, 5), Bound::both(-10.0));
        assert_eq!(t.sign_re(), SignSet::NEGATIVE);
        assert_eq!(t.sign_im(), SignSet::POSITIVE);
        let u = CBall::unknown(c(-3, 5));
        assert_eq!(u.sign_re(), SignSet::REAL);
        assert_eq!(u.sign_im(), SignSet::REAL);
    }

    #[test]
    fn sides_of_the_negative_real_axis() {
        let on = CBall::exact(c(-2, 0));
        assert_eq!(on.cut_side(Cut::NegativeReal), CutSide::On);
        assert!(on.is_exactly_real());
        let undecided = CBall::new(
            c(-2, 0),
            Bound {
                re: EXACT,
                im: -50.0,
            },
        );
        assert_eq!(undecided.cut_side(Cut::NegativeReal), CutSide::Undecided);
        assert!(!undecided.is_exactly_real());
        // Within its error of 0 but on the positive axis: away from the cut.
        let off = CBall::new(
            c(2, 0),
            Bound {
                re: EXACT,
                im: -50.0,
            },
        );
        assert_eq!(off.cut_side(Cut::NegativeReal), CutSide::Off);
        // A certain sign of the imaginary part: off the cut.
        let above = CBall::new(c(-2, 1), Bound::both(-50.0));
        assert_eq!(above.cut_side(Cut::NegativeReal), CutSide::Off);
        let unbounded = CBall::new(
            c(-2, 0),
            Bound {
                re: EXACT,
                im: UNKNOWN,
            },
        );
        assert_eq!(unbounded.cut_side(Cut::NegativeReal), CutSide::Undecided);
    }

    #[test]
    fn sum_and_product_carry_the_bound() {
        let a = CBall::exact(c(1, 0));
        let b = CBall::new(c(2, 0), Bound::real(-60.0));
        let s = CBall::sum(&[&a, &b], 128);
        assert_eq!(s.mid.0, BigFloat::from_i32(3, 64));
        assert!(s.is_exactly_real());
        assert!(s.err.re > -61.0 && s.err.re < -59.0);
        let p = CBall::product(&[&b, &b], 128);
        assert_eq!(p.mid.0, BigFloat::from_i32(4, 64));
        assert!(p.err.re > -60.0 && p.err.re < -56.0);
        let zero = CBall::exact(c(0, 0));
        let z = CBall::product(&[&zero, &CBall::unknown(c(7, 7))], 128);
        assert!(z.err.is_unknown());
    }

    #[test]
    fn cache_views_agree() {
        let mut arena = crate::base::arena::Arena::new();
        let (x, y, z) = (arena.int(2), arena.int(3), arena.int(4));
        let mut balls = Balls::default();
        balls.insert(x, CBall::new(c(2, 0), Bound::real(-100.0)));
        balls.insert(y, CBall::unknown(c(3, 0)));
        let mut values: FxHashMap<ExprId, Complex> = FxHashMap::default();
        let mut errs: FxHashMap<ExprId, Bound> = FxHashMap::default();
        for (id, b) in &balls {
            values.insert(*id, b.mid.clone());
            errs.insert(*id, b.err);
        }
        for cache in [Cache::Balls(&balls), Cache::Split(&values, Some(&errs))] {
            assert_eq!(cache.get(&x), Some(&c(2, 0)));
            assert_eq!(cache.bound(&x), Some(Bound::real(-100.0)));
            assert!(cache.err(&y).is_unknown());
            assert!(cache.get(&z).is_none() && cache.bound(&z).is_none());
            assert!(cache.err(&z).is_unknown() && cache.ball(&z).is_none());
            let b = cache.ball(&x).map(|b| b.err);
            assert_eq!(b, Some(Bound::real(-100.0)));
            // Values only: every bound hidden, as an empty bound map.
            let blind = cache.values_only();
            assert_eq!(blind.get(&x), Some(&c(2, 0)));
            assert!(blind.bound(&x).is_none() && blind.err(&x).is_unknown());
            assert!(blind.ball(&x).is_some_and(|b| b.err.is_unknown()));
        }
    }

    #[test]
    fn real_ball_arithmetic() {
        let wp = 64;
        let third = RBall::from_q(&Q::new(1.into(), 3.into()), wp);
        assert!((-66.0..=-62.0).contains(&third.r));
        let half = RBall::from_q(&Q::new(1.into(), 2.into()), wp);
        assert_eq!(half.r, f64::NEG_INFINITY);
        let one = third.mul(&RBall::from_int(3, wp), wp);
        assert!(one.certainly_positive());
        assert!(one.hi() > -1e-6 && one.lo() < 1e-6);
        let zero = third.add(&third.neg(), wp);
        assert!(!zero.certainly_nonzero() && !zero.exact_zero());
        assert!(zero.recip(wp).is_none());
        assert!(RBall::zero(wp).exact_zero());
        let three = third.recip(wp).map(|t| t.m);
        assert_eq!(
            three.map(|m| m.round(0, RM)),
            Some(BigFloat::from_i32(3, 64))
        );
        // The product rule is the one of `accuracy::mul_with_bound`.
        let p = half.mul(&third, wp);
        let lt = accuracy::part_lg(&third.m);
        let rule = accuracy::product_rule(accuracy::part_lg(&half.m), half.r, lt, third.r);
        assert_eq!(p.r, accuracy::lsum(rule, ulp(&p.m, wp)));
    }
}
