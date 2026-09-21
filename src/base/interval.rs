//! A closed interval `[lower, upper]` with *named* endpoints.
//!
//! This is the type behind every "pair of bounds" in the public API —
//! confidence and credible intervals, root-isolating intervals, search
//! ranges, per-variable bounds for optimisers and linear programs.  It
//! replaces the `(lower, upper)` tuples those used to be: a struct whose two
//! fields have the same type costs nothing at runtime and makes the
//! endpoints impossible to transpose silently (`ci.lower` reads as what it
//! is; `ci.0` does not).  See CONTRIBUTING.md, "Tuples versus structs".
//!
//! The struct is deliberately small: two public fields, a constructor, and
//! a handful of shape-preserving helpers.  It carries no ordering
//! invariant — `lower <= upper` is the caller's business — because the
//! element type is often `Ex` or `Option<Q>`, where "ordered" is not even
//! decidable.

use std::fmt;
use std::ops::Sub;

/// The closed interval `[lower, upper]`.
///
/// ```
/// use symplex::Interval;
///
/// let ci: Interval<f64> = Interval { lower: 0.108, upper: 0.603 };
/// assert!(ci.contains(&0.3));
/// assert!((ci.width() - 0.495).abs() < 1e-12);
/// assert_eq!(ci.to_string(), "[0.108, 0.603]");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Interval<T> {
    /// The lower endpoint.
    pub lower: T,
    /// The upper endpoint.
    pub upper: T,
}

impl<T> Interval<T> {
    /// `[lower, upper]`.  Prefer the struct literal at call sites where the
    /// two arguments are not visibly ordered (`Interval { lower: a, upper: b }`
    /// cannot be transposed by accident; `Interval::new(a, b)` can).
    pub const fn new(lower: T, upper: T) -> Self {
        Interval { lower, upper }
    }

    /// Apply `f` to both endpoints.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Interval<U> {
        Interval {
            lower: f(self.lower),
            upper: f(self.upper),
        }
    }

    /// Borrow both endpoints.
    pub const fn as_ref(&self) -> Interval<&T> {
        Interval {
            lower: &self.lower,
            upper: &self.upper,
        }
    }

    /// The endpoints as a `(lower, upper)` tuple, for destructuring:
    /// `let (lo, hi) = ci.into_pair();`.
    pub fn into_pair(self) -> (T, T) {
        (self.lower, self.upper)
    }
}

impl<T: PartialOrd> Interval<T> {
    /// `lower <= x <= upper`.
    pub fn contains(&self, x: &T) -> bool {
        self.lower <= *x && *x <= self.upper
    }

    /// `lower <= upper` (false for NaN endpoints).
    pub fn is_ordered(&self) -> bool {
        self.lower <= self.upper
    }
}

impl<T: Clone + Sub<Output = T>> Interval<T> {
    /// `upper − lower`.
    pub fn width(&self) -> T {
        self.upper.clone() - self.lower.clone()
    }
}

impl<T: fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.lower, self.upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        let iv = Interval::new(1, 4);
        assert_eq!(iv.width(), 3);
        assert!(iv.contains(&1) && iv.contains(&4) && !iv.contains(&5));
        assert!(iv.is_ordered());
        assert!(!Interval::new(2, 1).is_ordered());
        assert_eq!(iv.map(|x| x * 2), Interval::new(2, 8));
        assert_eq!(iv.as_ref(), Interval::new(&1, &4));
        assert_eq!(iv.into_pair(), (1, 4));
        assert_eq!(iv.to_string(), "[1, 4]");
        assert_eq!(Interval::<f64>::default(), Interval::new(0.0, 0.0));
    }

    #[test]
    fn option_endpoints_have_no_order_requirement() {
        let b: Interval<Option<i32>> = Interval {
            lower: None,
            upper: Some(3),
        };
        assert_eq!(b.map(|e| e.unwrap_or(0)), Interval::new(0, 3));
    }
}
