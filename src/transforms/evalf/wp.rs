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
//! | the same, its value returned as it is: what the routine charged to it is kept | [`Wp::carry`] |
//! | a route whose value is a sum that cancels: the absolute error of that sum | [`Summed`] (the terms), then [`Wp::charge`] |
//! | a route whose value is the end of a three-term recurrence that oscillates | [`Recurred`] (the values), then [`Wp::charge`] |
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
//!
//! # A route's own error
//!
//! A real route returns a bare `BigFloat`, and the node's bound gives it
//! two ulps *of that value* (`accuracy::finish`).  That is the truth of a
//! value formed by correctly rounded operations from its arguments, and of
//! a sum whose terms have one sign.  It is not the truth of a sum that
//! cancels: formed at `wp` bits from terms as large as `2^t`, the sum is
//! known to about `2^(t − wp)` whatever its own size, and next to a zero
//! of the function its own size is anything.  Up to 0.59 `J_ν`, `Y_ν`,
//! `Ci`, `Chi`, the Airy functions, `li` and `I_ν` of a negative order
//! printed uncertified digits there, at an argument that is exactly
//! representable (so that nothing but the route's own error is in the
//! bound): `besselj(0, x)` for the 166-bit dyadic `x` nearest the first
//! zero, `6.5·10⁻⁵¹`, was wrong from its 27th digit at 30 digits; `Chi`
//! and `Ci` next to a zero from the 15th and 12th digit at 16.
//!
//! Such a route says what its sum was made of — every term to a
//! [`Summed`] — and charges the sum's error to the [`Wp`] it was called
//! with ([`Wp::charge`] of [`Summed::error`]):
//!
//! ```text
//! 2^(−wp) · 2^OPS_BITS · (n + 1) · Σ 2^(e_k),      2^(e_k − 1) ≤ |t_k| < 2^(e_k)
//! ```
//!
//! for `n` terms summed at `wp` bits: a term `t_k` that a recurrence of
//! rounded operations brought within `100·(k + 1)·2^(−wp)` of itself, and
//! `n` partial sums each rounded once and each at most `Σ |t_k|`, are
//! together within `101·n·2^(−wp)·Σ |t_k|` of the exact sum (`OPS_BITS =
//! 7`).  The charge is an absolute error, not a part of the value's size;
//! `sensitivity::special_error` adds it to the node's propagated error
//! (and `cspecial::bound` to the real part of `Ci(−x)`, `Chi(−x)`), so a
//! value far below its terms has a bound that says so, and
//! `evaluate_adaptive` raises the precision until the digits are there —
//! or refuses within its budget.  A value of ordinary size is not
//! affected: the routes carry at least 32 bits beyond the precision, and
//! the charge stays more than 2¹⁵ below the value's own rounding.
//!
//! - **One node.** What a route charged belongs to the node being
//!   evaluated: `eval_ball` clears it first ([`Wp::clear_own`]) and the
//!   bound takes it ([`Wp::take_own`]).  A [`Wp::at`] starts with nothing
//!   charged, and drops what was charged to it — right for a value that is
//!   an ingredient (scaled, divided, compared), whose user says what it
//!   does to the error; [`Wp::carry`] keeps it, for a value returned as it
//!   is.
//! - **Charge last.** A route charges immediately before it returns its
//!   value: a route that fails after charging would leave its charge to
//!   the route tried next (`debye_fallback`).
//! - **Not finite.** A NaN or infinite term makes the charge unknown, and
//!   with it the node's bound.

use astro_float::{BigFloat, Consts, RoundingMode};

use super::accuracy::{self, ErrExp};

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

/// `log₂` of the factor by which [`Summed::error`] multiplies `2^(−wp)·(n +
/// 1)·Σ 2^(e_k)` (see the module documentation): `101 < 2⁷`.
const OPS_BITS: f64 = 7.0;

/// `2^d` for `d ≤ 0` (0 below the range of `f64`).
fn pow2(d: i64) -> f64 {
    match u64::try_from(d.saturating_add(1023)) {
        Ok(biased @ 1..=1023) => f64::from_bits(biased << 52),
        _ => 0.0,
    }
}

/// What a route summed at its working precision: the sizes of the terms
/// of a sum that may cancel, from which the absolute error of the sum
/// follows ([`Summed::error`]; see the module documentation, "A route's
/// own error").
///
/// ```ignore
/// let mut seen = Summed::new();
/// seen.term(&BigFloat::from_i32(1024, 64));
/// seen.term(&BigFloat::from_i32(-1023, 64));
/// // 2¹¹ + 2¹⁰ for the two terms, three for `n + 1`, `OPS_BITS`.
/// assert!((seen.error(160, 0.0) - (11.585 + 1.585 + 7.0 - 160.0)).abs() < 1e-3);
/// w.charge(seen.error(160, 0.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub(super) struct Summed {
    /// The largest binary exponent `e_k` among the terms (`None`: no term
    /// but exact zeros yet).
    top: Option<i64>,
    /// `Σ 2^(e_k − top)` over the terms: their absolute values sum to less
    /// than `mass·2^top`.
    mass: f64,
    /// The number of terms, and of rounded steps behind them
    /// ([`Summed::steps`]).
    count: u32,
}

impl Summed {
    /// Nothing summed yet.
    pub(super) const fn new() -> Summed {
        Summed {
            top: None,
            mass: 0.0,
            count: 0,
        }
    }

    /// The sum of a few `summands`.
    pub(super) fn of(summands: &[&BigFloat]) -> Summed {
        let mut seen = Summed::new();
        for s in summands {
            seen.term(s);
        }
        seen
    }

    /// A term of the sum (an exact zero counts as a step only).
    pub(super) fn term(&mut self, t: &BigFloat) {
        self.count = self.count.saturating_add(1);
        let Some(e) = accuracy::part_mag(t) else {
            if !t.is_zero() {
                // NaN or infinite: no bound.
                self.mass = f64::INFINITY;
                self.top.get_or_insert(0);
            }
            return;
        };
        match self.top {
            None => {
                self.top = Some(e);
                self.mass = 1.0;
            }
            Some(top) if e > top => {
                self.mass = self.mass * pow2(top.saturating_sub(e)) + 1.0;
                self.top = Some(e);
            }
            Some(top) => self.mass += pow2(e.saturating_sub(top)),
        }
    }

    /// `n` more rounded steps lie behind the summands (the terms of a
    /// series of one sign that is one of them, the coefficients of an
    /// expansion): each summand is within `100·(n + 1)·2^(−wp)` of itself.
    pub(super) fn steps(mut self, n: usize) -> Summed {
        self.count = self
            .count
            .saturating_add(u32::try_from(n).unwrap_or(u32::MAX));
        self
    }

    /// `log₂` of the absolute error of the sum, formed at `wp` bits and
    /// multiplied by a factor of at most `2^scale` on its way into the
    /// value (see the module documentation): `accuracy::EXACT` when
    /// nothing but exact zeros was summed, `accuracy::UNKNOWN` after a
    /// term that is not finite.
    pub(super) fn error(&self, wp: usize, scale: f64) -> ErrExp {
        let Some(top) = self.top else {
            return accuracy::EXACT;
        };
        let sum = top as f64 + self.mass.log2() + (f64::from(self.count) + 1.0).log2() + OPS_BITS
            - wp as f64;
        // (A `mass` or `scale` that is not finite is `UNKNOWN`, a factor 0
        // `EXACT`.)
        accuracy::shift(sum, scale)
    }
}

/// What a three-term recurrence `p_(k+1) = (α_k p_k − γ_k p_(k−1))/d_k` ran
/// through at its working precision: the envelope of its values at the
/// last index, from which the absolute error of the last value follows
/// ([`Recurred::error`]).
///
/// The value of an orthogonal polynomial next to one of its zeros is a
/// difference of two terms of the size of its neighbours, and the rounding
/// of every earlier step has reached it: a perturbation `δ` of `p_k` is a
/// solution of the same recurrence, so it arrives as `δ·(h_n/h_k)/sin θ` at
/// most, `h_k` the norm of `p_k` (the growth of the family's values with
/// the index) and `θ` the phase the oscillation advances per step — at
/// least `π/(2n)` at a zero (`T_n`: exactly `δ·U_(n−k−1)(x)`).  With the
/// envelope `E = max_k |p_k|·h_n/h_k` the last value is within
///
/// ```text
/// 2^(−wp) · 2^OPS_BITS · (n + 1)² · E
/// ```
///
/// of the exact one, whatever its own size.  Where the values grow faster
/// than their norm (outside the interval of the zeros) `E` is the last
/// value itself, and the bound is a part of it.  Up to 0.59
/// `legendre(40, x)` for the 234-bit dyadic `x` nearest a zero was wrong
/// from the 26th digit of 50.
///
/// ```ignore
/// let mut seen = Recurred::new(&[&p0, &p1]);
/// for k in 1..n {
///     // … p_next from p_curr, p_prev …
///     seen.step(0.0, &p_next);       // `log₂(h_(k+1)/h_k)`, the new value
/// }
/// w.charge(seen.error(wp, 0.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub(super) struct Recurred {
    /// `log₂` of the envelope at the current index (`−∞`: zeros only).
    lg: f64,
    /// The number of steps taken.
    steps: u32,
}

impl Recurred {
    /// The starting values.
    pub(super) fn new(first: &[&BigFloat]) -> Recurred {
        let mut lg = f64::NEG_INFINITY;
        for v in first {
            lg = lg.max(Recurred::lg_up(v));
        }
        Recurred { lg, steps: 0 }
    }

    /// An upper bound of `log₂|v|` from the binary exponent alone (this
    /// runs once per step of a recurrence): `−∞` for 0, `+∞` for a value
    /// that is not finite.
    fn lg_up(v: &BigFloat) -> f64 {
        match accuracy::part_mag(v) {
            Some(e) => e as f64,
            None if v.is_zero() => f64::NEG_INFINITY,
            None => f64::INFINITY,
        }
    }

    /// One step: the norm grows by `2^growth` (at least 1: a negative or
    /// undefined `growth` counts as 0), the new value is `value`.
    pub(super) fn step(&mut self, growth: f64, value: &BigFloat) {
        let growth = if growth.is_finite() {
            growth.max(0.0)
        } else {
            0.0
        };
        self.lg = (self.lg + growth).max(Recurred::lg_up(value));
        self.steps = self.steps.saturating_add(1);
    }

    /// `log₂` of the absolute error of the last value, the recurrence run
    /// at `wp` bits and the value multiplied by a factor of at most
    /// `2^scale` afterwards: `accuracy::EXACT` before the first step (the
    /// starting values have their rounding only) and for zeros only,
    /// `accuracy::UNKNOWN` after a value that is not finite.
    pub(super) fn error(&self, wp: usize, scale: f64) -> ErrExp {
        if self.steps == 0 || self.lg == f64::NEG_INFINITY {
            return accuracy::EXACT;
        }
        let n = f64::from(self.steps) + 2.0;
        accuracy::shift(self.lg + 2.0 * n.log2() + OPS_BITS - wp as f64, scale)
    }
}

/// The working precision and the constant cache of one evaluation, and
/// what the route running at it charged to its value (see the module
/// documentation, "A route's own error").
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
    /// `log₂` of the absolute error charged to the value being computed
    /// (`accuracy::EXACT`: none).
    own: ErrExp,
}

impl<'a> Wp<'a> {
    /// `bits` bits (saturating, [`Prec::new`]) with the cache `cc`.
    pub(super) fn new(bits: usize, cc: &'a mut Consts) -> Wp<'a> {
        Wp {
            prec: Prec::new(bits),
            cc,
            own: accuracy::EXACT,
        }
    }

    /// The precision in bits.
    pub(super) fn bits(&self) -> usize {
        self.prec.bits()
    }

    /// The same cache at `bits` bits (nothing charged; what a routine
    /// charges to it is dropped with it: [`Wp::carry`] keeps it).
    pub(super) fn at(&mut self, bits: usize) -> Wp<'_> {
        Wp {
            prec: Prec::new(bits),
            cc: self.cc,
            own: accuracy::EXACT,
        }
    }

    /// `f` at `bits` bits, as with [`Wp::at`], and what it charged to its
    /// value charged to this one's: for a value that is returned as it is
    /// (rounded, or with its sign changed).
    pub(super) fn carry<T>(&mut self, bits: usize, f: impl FnOnce(&mut Wp<'_>) -> T) -> T {
        let mut sub = self.at(bits);
        let out = f(&mut sub);
        let charged = sub.own;
        self.charge(charged);
        out
    }

    /// The value the route returns has an absolute error of at most `2^lg`
    /// besides its rounding (`accuracy::EXACT`: none; NaN or `+∞`: no
    /// bound).  Charges add.
    pub(super) fn charge(&mut self, lg: ErrExp) {
        self.own = accuracy::lsum(self.own, lg);
    }

    /// `log₂` of the absolute error charged to the value
    /// (`accuracy::EXACT`: none; `accuracy::UNKNOWN`: no bound), which is
    /// forgotten.
    pub(super) fn take_own(&mut self) -> ErrExp {
        std::mem::replace(&mut self.own, accuracy::EXACT)
    }

    /// Forget what was charged: a new node (`eval_ball`).
    pub(super) fn clear_own(&mut self) {
        self.own = accuracy::EXACT;
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

    /// The error of a sum is `2^(OPS_BITS − wp)·(n + 1)·Σ 2^(e_k)` times the
    /// scale; `at` drops a charge, `carry` keeps it, `take_own` forgets it.
    #[test]
    fn a_sum_charges_its_terms() {
        let Ok(mut cc) = Consts::new() else {
            return;
        };
        let mut w = Wp::new(128, &mut cc);
        assert_eq!(w.take_own(), f64::NEG_INFINITY);
        let mut seen = Summed::new();
        seen.term(&BigFloat::from_i32(1024, 64));
        seen.term(&BigFloat::from_i32(-1023, 64));
        seen.term(&BigFloat::new(64));
        // 2¹¹ + 2¹⁰ for the two nonzero terms, four for `n + 1`.
        let expected = 3072f64.log2() + 4f64.log2() + OPS_BITS + 3.0 - 160.0;
        let got = seen.error(160, 3.0);
        assert!((got - expected).abs() < 1e-9, "{got} {expected}");
        // The order of the terms does not matter.
        let mut other = Summed::new();
        other.term(&BigFloat::new(64));
        other.term(&BigFloat::from_i32(-1023, 64));
        other.term(&BigFloat::from_i32(1024, 64));
        assert!((other.error(160, 3.0) - expected).abs() < 1e-9);
        // A factor of 0 on the way into the value: no error.
        assert_eq!(seen.error(160, f64::NEG_INFINITY), f64::NEG_INFINITY);

        // Exact zeros charge nothing.
        let mut zeros = Summed::new();
        zeros.term(&BigFloat::new(64));
        assert_eq!(zeros.error(160, 0.0), f64::NEG_INFINITY);
        w.charge(zeros.error(160, 0.0));
        assert_eq!(w.take_own(), f64::NEG_INFINITY);

        // `at` drops, `carry` keeps, charges add, `take_own` forgets.
        w.at(64).charge(-10.0);
        assert_eq!(w.take_own(), f64::NEG_INFINITY);
        let seven = w.carry(64, |ww| {
            ww.charge(-10.0);
            7
        });
        assert_eq!(seven, 7);
        w.charge(-10.0);
        assert!((w.take_own() + 9.0).abs() < 1e-9);
        assert_eq!(w.take_own(), f64::NEG_INFINITY);

        // `of` and `steps`: the summands and `n` more steps.
        let (a, b) = (BigFloat::from_i32(3, 64), BigFloat::from_i32(-3, 64));
        let expected = 8f64.log2() + 8f64.log2() + OPS_BITS - 100.0;
        let got = Summed::of(&[&a, &b]).steps(5).error(100, 0.0);
        assert!((got - expected).abs() < 1e-9, "{got} {expected}");

        // A term that is not finite: no bound; `clear_own` forgets it.
        let mut bad = Summed::new();
        bad.term(&astro_float::INF_POS);
        bad.term(&BigFloat::from_i32(1, 64));
        assert!(accuracy::is_unknown(bad.error(100, 0.0)));
        w.charge(bad.error(100, 0.0));
        w.clear_own();
        assert_eq!(w.take_own(), f64::NEG_INFINITY);
    }

    /// The envelope of a recurrence: older values scaled by the growth of
    /// the norm, the error `(n + 1)²·2^(OPS_BITS − wp)` of it.
    #[test]
    fn a_recurrence_charges_its_envelope() {
        let (one, seven) = (BigFloat::from_i32(1, 64), BigFloat::from_i32(7, 64));
        let tiny = BigFloat::from_i32(1, 64).div(&BigFloat::from_i32(1024, 64), 64, RM);
        let mut seen = Recurred::new(&[&one, &seven]);
        // 7 < 2³ grown by 2² is above the new value; a negative growth is none.
        seen.step(2.0, &tiny);
        seen.step(-1.0, &tiny);
        seen.step(f64::NAN, &tiny);
        let expected = 5.0 + 2.0 * 5f64.log2() + OPS_BITS - 100.0 + 1.0;
        let got = seen.error(100, 1.0);
        assert!((got - expected).abs() < 1e-6, "{got} {expected}");
        // A value above the envelope is the envelope (`2²⁰ < 2²¹`).
        seen.step(0.0, &BigFloat::from_i32(1 << 20, 64));
        let expected = 21.0 + 2.0 * 6f64.log2() + OPS_BITS - 100.0;
        assert!((seen.error(100, 0.0) - expected).abs() < 1e-6);
        // No step, or zeros only: exact; a value that is not finite: no bound.
        let zero = BigFloat::new(64);
        assert_eq!(Recurred::new(&[&one]).error(100, 0.0), f64::NEG_INFINITY);
        let mut zeros = Recurred::new(&[&zero]);
        zeros.step(1.0, &zero);
        assert_eq!(zeros.error(100, 0.0), f64::NEG_INFINITY);
        let mut bad = Recurred::new(&[&one]);
        bad.step(0.0, &astro_float::INF_POS);
        assert!(accuracy::is_unknown(bad.error(100, 0.0)));
    }

    #[test]
    fn powers_of_two_down_to_the_bottom_of_f64() {
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(-1), 0.5);
        assert_eq!(pow2(-1022), f64::MIN_POSITIVE);
        assert_eq!(pow2(-1023), 0.0);
        assert_eq!(pow2(i64::MIN), 0.0);
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
