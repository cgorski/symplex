//! The working precision of `evalf`: [`Prec`] (a count of bits), [`Wp`]
//! (the precision together with astro-float's constant cache) and [`RM`],
//! the one rounding mode every evaluation uses.
//!
//! These replace the `prec: usize, rm: RoundingMode, cc: &mut Consts`
//! triple that every evaluation routine used to take.  The rounding mode
//! was a parameter of about 390 functions, yet every evaluation started
//! from `RoundingMode::ToEven` and passed it on unchanged; a parameter
//! suggested that a caller could pick another mode, which no bound in
//! `accuracy` allows (they assume round-to-nearest).  Directed roundings
//! are spelled out where they are meant: the upper bounds of
//! `emsum`/`hypsum` add and multiply with `RoundingMode::Up` at 64 bits.
//!
//! # Which type
//!
//! | Need | Type |
//! |---|---|
//! | a routine that draws on the constant cache (`π`, `ln 2`, `exp`, `ln`, the trigonometric family) | `w: &mut Wp<'_>` |
//! | the same routine at another precision | `&mut w.at(bits)` (same cache) |
//! | a routine that only rounds (`+`, `·`, `/`, `√`, `powi`) | `prec: usize`, the bits of a [`Prec`] |
//! | the rounding mode of an astro-float primitive | [`RM`]; a directed mode literally (`RoundingMode::Up`) |
//! | a precision raised by a guard computed in `f64` (`log₂` of a magnitude) | [`Prec::guarded`] with a [`Guard`] |
//!
//! # Corner cases
//!
//! - **Saturation.** A [`Prec`] holds at most `u32::MAX` bits; [`Prec::new`]
//!   and [`Prec::plus`] saturate there instead of wrapping (a `usize`
//!   precision plus a guard computed in `f64` wrapped silently in release
//!   builds, and astro-float cannot allocate four billion bits anyway).
//! - **Unbounded guards.** [`Guard::for_log2`] of NaN or `+∞` is
//!   [`Guard::Unbounded`], and [`Prec::guarded`] refuses it like a guard
//!   beyond the cap: `None`, for the caller to report as a lack of
//!   precision rather than to evaluate at a wrapped or saturated count.
//! - **Rounding.** astro-float 0.9.6 rounds `ToEven` and `ToOdd` to
//!   nearest; `Up`, `Down`, `ToZero` and `FromZero` are *directed* (toward
//!   `+∞`, `−∞`, zero, away from zero), although its documentation calls
//!   them "round half …" (`base::bigfloat`, measured).

use astro_float::{Consts, RoundingMode};

/// The rounding mode of every `evalf` operation: to nearest, ties to even.
/// The error bounds of `accuracy` (half an ulp per operation, two ulps per
/// transcendental result) assume it.
pub(super) const RM: RoundingMode = RoundingMode::ToEven;

/// A binary working precision, in bits.
///
/// ```ignore
/// let p = Prec::new(128).plus(32);
/// assert_eq!(p.bits(), 160);
/// assert_eq!(Prec::new(usize::MAX), Prec::MAX);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Prec(u32);

impl Prec {
    /// The largest precision: `u32::MAX` bits.
    pub(super) const MAX: Prec = Prec(u32::MAX);

    /// `bits` bits, saturating at [`Prec::MAX`].
    pub(super) fn new(bits: usize) -> Prec {
        Prec(u32::try_from(bits).unwrap_or(u32::MAX))
    }

    /// The number of bits, as astro-float takes it.
    pub(super) fn bits(self) -> usize {
        self.0 as usize
    }

    /// `self + extra` bits, saturating at [`Prec::MAX`].
    pub(super) fn plus(self, extra: usize) -> Prec {
        Prec::new(self.bits().saturating_add(extra))
    }

    /// `self` raised by the guard `g`, or `None` when the guard is
    /// unbounded or the result exceeds `cap`.
    pub(super) fn guarded(self, g: Guard, cap: Prec) -> Option<Prec> {
        match g {
            Guard::Bits(k) => {
                let p = self.0.checked_add(k)?;
                (p <= cap.0).then_some(Prec(p))
            }
            Guard::Unbounded => None,
        }
    }
}

/// Extra bits a computation needs, from an estimate in `f64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Guard {
    /// This many bits.
    Bits(u32),
    /// No finite number of bits: the estimate was NaN or `+∞`.
    Unbounded,
}

impl Guard {
    /// `⌈x⌉` bits for the `log₂` estimate `x`: 0 for `x ≤ 0` (and `−∞`),
    /// [`Guard::Unbounded`] for NaN, `+∞` or beyond `u32::MAX`.
    ///
    /// ```ignore
    /// assert_eq!(Guard::for_log2(10.2), Guard::Bits(11));
    /// assert_eq!(Guard::for_log2(-3.0), Guard::Bits(0));
    /// assert_eq!(Guard::for_log2(f64::NAN), Guard::Unbounded);
    /// ```
    pub(super) fn for_log2(x: f64) -> Guard {
        if x <= 0.0 {
            return Guard::Bits(0);
        }
        crate::base::cast::f64_to_i64(x.ceil())
            .and_then(|k| u32::try_from(k).ok())
            .map_or(Guard::Unbounded, Guard::Bits)
    }
}

/// The working precision and the constant cache of one evaluation.
///
/// ```ignore
/// let mut cc = Consts::new()?;
/// let mut w = Wp::new(128, &mut cc);
/// let pi = w.cc.pi(w.bits(), RM);
/// let finer = c_exp(&z, &mut w.at(w.bits() + 32));
/// ```
pub(super) struct Wp<'a> {
    prec: Prec,
    /// astro-float's cache of `π`, `ln 2`, … at the precisions reached.
    pub(super) cc: &'a mut Consts,
}

impl<'a> Wp<'a> {
    /// `bits` bits (saturating, [`Prec::new`]) with the cache `cc`.
    pub(super) fn new(bits: usize, cc: &'a mut Consts) -> Wp<'a> {
        Wp {
            prec: Prec::new(bits),
            cc,
        }
    }

    /// The precision in bits.
    pub(super) fn bits(&self) -> usize {
        self.prec.bits()
    }

    /// The same cache at `bits` bits.
    pub(super) fn at(&mut self, bits: usize) -> Wp<'_> {
        Wp {
            prec: Prec::new(bits),
            cc: self.cc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prec_saturates_instead_of_wrapping() {
        assert_eq!(Prec::new(128).plus(32).bits(), 160);
        assert_eq!(Prec::new(usize::MAX), Prec::MAX);
        assert_eq!(Prec::MAX.plus(1), Prec::MAX);
    }

    #[test]
    fn guards_from_f64_estimates() {
        assert_eq!(Guard::for_log2(10.2), Guard::Bits(11));
        assert_eq!(Guard::for_log2(-3.0), Guard::Bits(0));
        assert_eq!(Guard::for_log2(f64::NEG_INFINITY), Guard::Bits(0));
        assert_eq!(Guard::for_log2(f64::NAN), Guard::Unbounded);
        assert_eq!(Guard::for_log2(f64::INFINITY), Guard::Unbounded);
        assert_eq!(Guard::for_log2(1e300), Guard::Unbounded);
        let cap = Prec::new(1000);
        assert_eq!(
            Prec::new(900).guarded(Guard::Bits(100), cap),
            Some(Prec::new(1000))
        );
        assert_eq!(Prec::new(900).guarded(Guard::Bits(101), cap), None);
        assert_eq!(Prec::new(900).guarded(Guard::Unbounded, cap), None);
        assert_eq!(Prec::MAX.guarded(Guard::Bits(1), Prec::MAX), None);
    }

    #[test]
    fn wp_reborrows_the_cache_at_another_precision() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        let mut w = Wp::new(128, &mut cc);
        let finer = w.at(256);
        assert_eq!(finer.bits(), 256);
        assert_eq!(w.bits(), 128);
    }
}
