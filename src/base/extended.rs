//! The extended line: a value of `T` or one of the two infinities.
//!
//! [`Extended<T>`] is the one spelling of "an endpoint that may be `±∞`,
//! as a value" — the exact rational ends of a polynomial sign query, the
//! limits of a `Σ`, the value an antiderivative tends to at an endpoint.
//! Unlike `Option<T>` it orders the way the real line does
//! (`-∞ < a < b < +∞`), so `Interval<Extended<T>>` is a legitimate
//! unbounded interval whose [`contains`](Interval::contains) is right; see
//! the "Which type" table in [`interval`](crate::base::interval).
//!
//! It is a *value*, not a set: `Extended::PosInf` is the point `+∞`, and
//! whether it belongs to an interval is the interval's `kind`.  For a
//! closed box side that is merely absent use [`Bounds<T>`]
//! (`crate::base::interval::Bounds`); for symbolic infinities inside an
//! expression use the arena's `Infinity` / `NegInfinity` nodes.
//!
//! [`Bounds<T>`]: crate::base::interval::Bounds

use std::cmp::Ordering;
use std::fmt;
use std::ops::Neg;

use crate::base::interval::Interval;

/// A value of `T` or one of the two infinities: the extended line.
///
/// Ordered `NegInf < Finite(a) < Finite(b) < PosInf` (with `a < b`); the
/// two infinities compare equal to themselves.
///
/// ```
/// use symplex::{Extended, Interval};
///
/// let ray = Interval::right_open(Extended::Finite(0), Extended::PosInf); // [0, ∞)
/// assert!(ray.contains(&Extended::Finite(7)));
/// assert!(!ray.contains(&Extended::Finite(-1)) && !ray.contains(&Extended::PosInf));
/// assert!(!ray.is_bounded());
/// assert_eq!(ray.to_string(), "[0, ∞)");
///
/// assert!(Extended::NegInf < Extended::Finite(i32::MIN));
/// assert_eq!(-Extended::<i32>::PosInf, Extended::NegInf);
/// assert_eq!(Extended::from_f64(f64::INFINITY), Extended::PosInf);
/// assert_eq!(Extended::from(2.5).finite(), Some(&2.5));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extended<T> {
    /// `-∞`.
    NegInf,
    /// A finite value.
    Finite(T),
    /// `+∞`.
    PosInf,
}

impl<T> Extended<T> {
    /// Is this a finite value (not `±∞`)?
    pub const fn is_finite(&self) -> bool {
        matches!(self, Extended::Finite(_))
    }

    /// The finite value, if any.
    pub const fn finite(&self) -> Option<&T> {
        match self {
            Extended::Finite(v) => Some(v),
            Extended::NegInf | Extended::PosInf => None,
        }
    }

    /// The finite value, if any, by value.
    pub fn into_finite(self) -> Option<T> {
        match self {
            Extended::Finite(v) => Some(v),
            Extended::NegInf | Extended::PosInf => None,
        }
    }

    /// Apply `f` to a finite value; the infinities pass through.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Extended<U> {
        match self {
            Extended::Finite(v) => Extended::Finite(f(v)),
            Extended::NegInf => Extended::NegInf,
            Extended::PosInf => Extended::PosInf,
        }
    }

    /// Borrow the finite value.
    pub const fn as_ref(&self) -> Extended<&T> {
        match self {
            Extended::Finite(v) => Extended::Finite(v),
            Extended::NegInf => Extended::NegInf,
            Extended::PosInf => Extended::PosInf,
        }
    }
}

impl<T: Clone + Neg<Output = T>> Extended<T> {
    /// `-self`: negates a finite value and swaps the infinities.  The same
    /// as the `-` operator, for call chains and non-`Copy` `T`.
    #[must_use]
    pub fn neg(&self) -> Self {
        -self.clone()
    }
}

/// `-(+∞) = -∞`, `-(-∞) = +∞`, and `T`'s own negation in between.
impl<T: Neg> Neg for Extended<T> {
    type Output = Extended<T::Output>;

    fn neg(self) -> Self::Output {
        match self {
            Extended::Finite(v) => Extended::Finite(-v),
            Extended::NegInf => Extended::PosInf,
            Extended::PosInf => Extended::NegInf,
        }
    }
}

impl Extended<f64> {
    /// `f64::INFINITY` and `f64::NEG_INFINITY` become the infinities;
    /// every other value — **including NaN** — is `Finite`.  NaN is not a
    /// point of the extended line, but it is not an infinity either, so it
    /// stays where `f64`'s own comparisons treat it: unordered against
    /// every `Finite` value, and `Finite(NaN) != Finite(NaN)`.  The
    /// infinities still bound it (`NegInf < Finite(NaN) < PosInf`), since
    /// they compare before the finite payload is looked at.
    pub fn from_f64(x: f64) -> Self {
        if x == f64::INFINITY {
            Extended::PosInf
        } else if x == f64::NEG_INFINITY {
            Extended::NegInf
        } else {
            Extended::Finite(x)
        }
    }
}

/// A `T` is a finite point of the extended line.
impl<T> From<T> for Extended<T> {
    fn from(value: T) -> Self {
        Extended::Finite(value)
    }
}

impl<T: PartialOrd> PartialOrd for Extended<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Extended::*;
        match (self, other) {
            (NegInf, NegInf) | (PosInf, PosInf) => Some(Ordering::Equal),
            (NegInf, _) | (_, PosInf) => Some(Ordering::Less),
            (PosInf, _) | (_, NegInf) => Some(Ordering::Greater),
            (Finite(a), Finite(b)) => a.partial_cmp(b),
        }
    }
}

impl<T: Ord> Ord for Extended<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        use Extended::*;
        match (self, other) {
            (NegInf, NegInf) | (PosInf, PosInf) => Ordering::Equal,
            (NegInf, _) | (_, PosInf) => Ordering::Less,
            (PosInf, _) | (_, NegInf) => Ordering::Greater,
            (Finite(a), Finite(b)) => a.cmp(b),
        }
    }
}

/// `-∞`, `∞`, or the finite value's own `Display` (the spelling
/// [`Bounds`](crate::base::interval::Bounds) uses for its absent sides).
impl<T: fmt::Display> fmt::Display for Extended<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Extended::NegInf => f.write_str("-∞"),
            Extended::Finite(v) => v.fmt(f),
            Extended::PosInf => f.write_str("∞"),
        }
    }
}

impl<T> Interval<Extended<T>> {
    /// Are both endpoints finite?  `[0, ∞)` is not; `[0, 1]` is.  (The
    /// kind is irrelevant: an infinite end is unbounded whether the
    /// interval is written closed or open there.)
    pub const fn is_bounded(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Extended::*;

    #[test]
    fn order_is_the_real_line_with_infinities_at_the_ends() {
        // Pinned by hand so that reordering the variants cannot silently
        // change it.
        assert!(NegInf < Finite(i64::MIN));
        assert!(Finite(i64::MAX) < PosInf);
        assert!(Finite(-5) < Finite(0) && Finite(0) < Finite(7));
        assert!(Extended::<i32>::NegInf < PosInf);
        assert_eq!(
            NegInf.partial_cmp(&Extended::<i32>::NegInf),
            Some(Ordering::Equal)
        );
        assert_eq!(
            PosInf.partial_cmp(&Extended::<i32>::PosInf),
            Some(Ordering::Equal)
        );
        assert_eq!(Finite(3).cmp(&Finite(3)), Ordering::Equal);

        let mut v = vec![Finite(2), PosInf, Finite(-1), NegInf, Finite(0)];
        v.sort();
        assert_eq!(v, [NegInf, Finite(-1), Finite(0), Finite(2), PosInf]);

        // `Ord` agrees with `PartialOrd`.
        for a in &v {
            for b in &v {
                assert_eq!(a.partial_cmp(b), Some(a.cmp(b)));
            }
        }
    }

    #[test]
    fn nan_is_unordered_and_finite() {
        let nan = Extended::from_f64(f64::NAN);
        assert!(nan.is_finite());
        assert!(nan.partial_cmp(&Finite(0.0)).is_none());
        assert!(
            nan.partial_cmp(&PosInf).is_some(),
            "an infinity beats even NaN"
        );
        assert_ne!(nan, nan);
    }

    #[test]
    fn from_f64_maps_the_infinities() {
        assert_eq!(Extended::from_f64(f64::INFINITY), PosInf);
        assert_eq!(Extended::from_f64(f64::NEG_INFINITY), NegInf);
        assert_eq!(Extended::from_f64(1.5), Finite(1.5));
        assert_eq!(Extended::from_f64(f64::MAX), Finite(f64::MAX));
    }

    #[test]
    fn helpers() {
        let x = Finite(String::from("a"));
        assert!(x.is_finite() && !Extended::<String>::PosInf.is_finite());
        assert_eq!(x.finite().map(String::as_str), Some("a"));
        assert_eq!(x.as_ref(), Finite(&String::from("a")));
        assert_eq!(x.clone().into_finite(), Some(String::from("a")));
        assert_eq!(Extended::<String>::NegInf.into_finite(), None);
        assert_eq!(x.map(|s| s.len()), Finite(1));
        assert_eq!(Extended::<String>::PosInf.map(|s| s.len()), PosInf);
        assert_eq!(Extended::from(4), Finite(4));
    }

    #[test]
    fn negation_swaps_the_infinities() {
        assert_eq!(-Extended::<i32>::PosInf, NegInf);
        assert_eq!(-Extended::<i32>::NegInf, PosInf);
        assert_eq!(-Finite(3), Finite(-3));
        assert_eq!(Finite(3).neg(), Finite(-3));
        assert_eq!(Extended::<i32>::PosInf.neg(), NegInf);
    }

    #[test]
    fn display() {
        assert_eq!(Extended::<i32>::NegInf.to_string(), "-∞");
        assert_eq!(Extended::<i32>::PosInf.to_string(), "∞");
        assert_eq!(Finite(2.5).to_string(), "2.5");
        assert_eq!(format!("{:>4}", Finite(7)), "   7");
    }

    #[test]
    fn interval_of_extended_is_an_unbounded_interval() {
        let ray = Interval::right_open(Finite(0), PosInf); // [0, ∞)
        assert!(ray.contains(&Finite(7)));
        assert!(ray.contains(&Finite(0)));
        assert!(!ray.contains(&Finite(-1)));
        assert!(!ray.contains(&PosInf));
        assert!(!ray.contains(&NegInf));
        assert!(!ray.is_bounded() && ray.is_ordered() && !ray.is_empty());
        assert_eq!(ray.to_string(), "[0, ∞)");

        let line = Interval::open(NegInf, PosInf);
        assert!(line.contains(&Finite(i32::MIN)) && line.contains(&Finite(i32::MAX)));
        assert!(!line.is_bounded());

        let seg = Interval::closed(Finite(1), Finite(2));
        assert!(seg.is_bounded());
        assert!(Interval::closed(Finite(3), Finite(3)).is_point());
        assert!(Interval::closed(PosInf, Extended::<i32>::PosInf).is_point());
        assert!(Interval::open(Finite(1), NegInf).is_empty());
    }
}
