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
//! The bound rules of the elementary nodes — `exp`, `ln` and `arg`, `sin`,
//! `cos`, `sinh`, `cosh`, `tan`, `tanh`, `abs`, `sign`, `heaviside`,
//! `DiracDelta` — are `Ball` methods next to `accuracy::node_error`, which
//! hands them the borrowed ball of the child.
//!
//! # Which function
//!
//! | Need | Function |
//! |---|---|
//! | a value with a bound | [`CBall::new`]; [`CBall::exact`], [`CBall::unknown`] |
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

use std::borrow::Borrow;

use super::accuracy::{self, Bound, Cut, CutSide};
use crate::base::bigcomplex::Complex;
use crate::base::truth::SignSet;
use astro_float::BigFloat;

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
}
