//! Intervals and bounds with *named* endpoints: [`Interval<T>`] for a set
//! `[a, b]`, `(a, b)`, `(a, b]` or `[a, b)` between two finite endpoints,
//! [`Bounds<T>`] for a closed constraint box side whose ends may be absent.
//!
//! These replace the `(lower, upper)` tuples — and the unstated assumptions
//! that travelled with them — behind confidence and credible intervals,
//! root-isolating intervals, search ranges, optimiser boxes and linear
//! programming variable bounds.  A tuple could not say whether its endpoints
//! belonged to the set, so `(lo, hi]` isolating intervals and `[lo, hi]`
//! confidence intervals looked identical; here the kind is a field, so a
//! `Vec<Interval<Q>>` can mix kinds (a bisection that hits a root exactly
//! returns `[r, r]` next to `(lo, hi]` neighbours) and [`contains`],
//! [`is_empty`] and `Display` are right for each value.  See
//! CONTRIBUTING.md, "Tuples versus structs".
//!
//! # Which type
//!
//! | Need | Type |
//! |---|---|
//! | two finite endpoints, any of the four kinds | [`Interval<T>`] |
//! | closed box side, either end possibly unbounded (LP, polytope) | [`Bounds<T>`] |
//! | symbolic endpoints, `±∞`, unions, set algebra | `SetEx` (`Context::interval`) |
//! | the support of a distribution (intervals + points, lattice or continuous) | `stats::Support` |
//! | oriented limits `∫ₐᵇ`, `Σₐᵇ` (reversal flips the sign) | separate `lower`, `upper` arguments — not a set |
//!
//! `Interval<Option<T>>` is *not* the way to spell an unbounded end:
//! `Option`'s ordering puts `None` first, so `contains` would be wrong for
//! an upper-unbounded interval.  Use [`Bounds<T>`].
//!
//! Neither type carries an ordering invariant — `lower <= upper` is the
//! caller's business, because the endpoint type is often `Ex`, where
//! "ordered" is not even decidable.  [`Interval::is_ordered`] and
//! [`Interval::is_empty`] answer the question when `T: PartialOrd`.
//!
//! [`contains`]: Interval::contains
//! [`is_empty`]: Interval::is_empty

use std::fmt;
use std::ops::{Bound, Range, RangeBounds, RangeInclusive, Sub};

/// Which endpoints of an [`Interval`] belong to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IntervalKind {
    /// `[a, b]`: both endpoints included.
    #[default]
    Closed,
    /// `(a, b)`: neither endpoint included.
    Open,
    /// `(a, b]`: open at the lower end, closed at the upper.
    LeftOpen,
    /// `[a, b)`: closed at the lower end, open at the upper.
    RightOpen,
}

impl IntervalKind {
    /// The kind with the given openness at each end.
    pub const fn from_open_ends(lower_open: bool, upper_open: bool) -> Self {
        match (lower_open, upper_open) {
            (false, false) => IntervalKind::Closed,
            (true, true) => IntervalKind::Open,
            (true, false) => IntervalKind::LeftOpen,
            (false, true) => IntervalKind::RightOpen,
        }
    }

    /// Is the lower endpoint excluded?
    pub const fn lower_open(self) -> bool {
        matches!(self, IntervalKind::Open | IntervalKind::LeftOpen)
    }

    /// Is the upper endpoint excluded?
    pub const fn upper_open(self) -> bool {
        matches!(self, IntervalKind::Open | IntervalKind::RightOpen)
    }

    /// The kind with the lower end's openness replaced.
    pub const fn with_lower_open(self, open: bool) -> Self {
        Self::from_open_ends(open, self.upper_open())
    }

    /// The kind with the upper end's openness replaced.
    pub const fn with_upper_open(self, open: bool) -> Self {
        Self::from_open_ends(self.lower_open(), open)
    }

    /// The kind with the ends swapped: `(a, b]` ↔ `[a, b)`.  What a
    /// decreasing map does to an interval's kind.
    pub const fn reversed(self) -> Self {
        Self::from_open_ends(self.upper_open(), self.lower_open())
    }
}

/// The interval between `lower` and `upper`, with the endpoints included or
/// excluded according to `kind`.
///
/// ```
/// use symplex::{Interval, IntervalKind};
///
/// let ci = Interval::closed(0.108_f64, 0.603);
/// assert!(ci.contains(&0.3) && ci.contains(&0.603));
/// assert!((ci.width() - 0.495).abs() < 1e-12);
/// assert_eq!(ci.to_string(), "[0.108, 0.603]");
///
/// let iso = Interval::left_open(1, 2); // (1, 2]
/// assert!(!iso.contains(&1) && iso.contains(&2));
/// assert_eq!(iso.kind, IntervalKind::LeftOpen);
/// assert_eq!(iso.to_string(), "(1, 2]");
///
/// // Range syntax: `a..=b` is `[a, b]`, `a..b` is `[a, b)`.
/// assert_eq!(Interval::from(0.0..=1.0), Interval::closed(0.0, 1.0));
/// assert_eq!(Interval::from(0..5), Interval::right_open(0, 5));
///
/// assert!(Interval::open(3, 3).is_empty() && !Interval::closed(3, 3).is_empty());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Interval<T> {
    /// The lower endpoint.
    pub lower: T,
    /// The upper endpoint.
    pub upper: T,
    /// Which endpoints belong to the interval.
    pub kind: IntervalKind,
}

impl<T> Interval<T> {
    /// `[lower, upper]`.
    pub const fn closed(lower: T, upper: T) -> Self {
        Interval {
            lower,
            upper,
            kind: IntervalKind::Closed,
        }
    }

    /// `(lower, upper)`.
    pub const fn open(lower: T, upper: T) -> Self {
        Interval {
            lower,
            upper,
            kind: IntervalKind::Open,
        }
    }

    /// `(lower, upper]`.
    pub const fn left_open(lower: T, upper: T) -> Self {
        Interval {
            lower,
            upper,
            kind: IntervalKind::LeftOpen,
        }
    }

    /// `[lower, upper)`.
    pub const fn right_open(lower: T, upper: T) -> Self {
        Interval {
            lower,
            upper,
            kind: IntervalKind::RightOpen,
        }
    }

    /// The singleton `[value, value]`.
    pub fn point(value: T) -> Self
    where
        T: Clone,
    {
        Interval::closed(value.clone(), value)
    }

    /// Apply `f` to both endpoints, keeping the kind.  (For a *decreasing*
    /// `f` the caller must also swap the endpoints and
    /// [`reverse`](Self::reversed) the kind.)
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Interval<U> {
        Interval {
            lower: f(self.lower),
            upper: f(self.upper),
            kind: self.kind,
        }
    }

    /// Borrow both endpoints.
    pub const fn as_ref(&self) -> Interval<&T> {
        Interval {
            lower: &self.lower,
            upper: &self.upper,
            kind: self.kind,
        }
    }

    /// The same endpoints with a different kind.
    pub fn with_kind(self, kind: IntervalKind) -> Self {
        Interval { kind, ..self }
    }

    /// The image under a decreasing map that has already been applied to
    /// the endpoints: swap them and swap the openness with them, so that
    /// `[a, b)` becomes `(f(b), f(a)]`.
    pub fn reversed(self) -> Self {
        Interval {
            lower: self.upper,
            upper: self.lower,
            kind: self.kind.reversed(),
        }
    }

    /// The endpoints as a `(lower, upper)` tuple, for destructuring:
    /// `let (lo, hi) = ci.into_pair();`.  The kind is dropped.
    pub fn into_pair(self) -> (T, T) {
        (self.lower, self.upper)
    }
}

impl<T: Clone> Interval<&T> {
    /// Clone both endpoints.
    pub fn cloned(self) -> Interval<T> {
        Interval {
            lower: self.lower.clone(),
            upper: self.upper.clone(),
            kind: self.kind,
        }
    }
}

impl<T: PartialOrd> Interval<T> {
    /// Is `x` in the interval (endpoints according to `kind`)?
    pub fn contains(&self, x: &T) -> bool {
        let above_lower = if self.kind.lower_open() {
            self.lower < *x
        } else {
            self.lower <= *x
        };
        let below_upper = if self.kind.upper_open() {
            *x < self.upper
        } else {
            *x <= self.upper
        };
        above_lower && below_upper
    }

    /// `lower <= upper` (false for NaN endpoints).
    pub fn is_ordered(&self) -> bool {
        self.lower <= self.upper
    }

    /// Does the interval contain no point?  `[a, a]` is the singleton
    /// `{a}`; `(a, a)`, `(a, a]`, `[a, a)`, anything with `lower > upper`
    /// and anything with an unordered (NaN) endpoint are empty.  Between
    /// distinct endpoints an open interval of a *discrete* type may still be
    /// empty — `(1, 2)` over the integers — which this does not detect.
    pub fn is_empty(&self) -> bool {
        match self.lower.partial_cmp(&self.upper) {
            Some(std::cmp::Ordering::Less) => false,
            Some(std::cmp::Ordering::Equal) => self.kind != IntervalKind::Closed,
            _ => true,
        }
    }

    /// Is this the singleton `[a, a]`?
    pub fn is_point(&self) -> bool {
        self.kind == IntervalKind::Closed && self.lower == self.upper
    }
}

impl<T: Clone + Sub<Output = T>> Interval<T> {
    /// `upper − lower`.
    pub fn width(&self) -> T {
        self.upper.clone() - self.lower.clone()
    }
}

impl<T> RangeBounds<T> for Interval<T> {
    fn start_bound(&self) -> Bound<&T> {
        if self.kind.lower_open() {
            Bound::Excluded(&self.lower)
        } else {
            Bound::Included(&self.lower)
        }
    }

    fn end_bound(&self) -> Bound<&T> {
        if self.kind.upper_open() {
            Bound::Excluded(&self.upper)
        } else {
            Bound::Included(&self.upper)
        }
    }
}

/// `a..=b` is `[a, b]`.
impl<T> From<RangeInclusive<T>> for Interval<T> {
    fn from(r: RangeInclusive<T>) -> Self {
        let (lower, upper) = r.into_inner();
        Interval::closed(lower, upper)
    }
}

/// `a..b` is `[a, b)`, as in Rust.
impl<T> From<Range<T>> for Interval<T> {
    fn from(r: Range<T>) -> Self {
        Interval::right_open(r.start, r.end)
    }
}

impl<T: fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let open = if self.kind.lower_open() { '(' } else { '[' };
        let close = if self.kind.upper_open() { ')' } else { ']' };
        write!(f, "{open}{}, {}{close}", self.lower, self.upper)
    }
}

/// A closed constraint `lower ≤ x ≤ upper` in which either side may be
/// absent: the per-variable bounds of a linear program, the sides of a
/// bounding box.  `None` means unbounded on that side; both `None` is a free
/// variable.  (scipy calls the same thing `Bounds(lb, ub)` with `±inf`.)
///
/// ```
/// use symplex::Bounds;
///
/// let nonneg: Bounds<i32> = Bounds::at_least(0);
/// assert!(nonneg.contains(&7) && !nonneg.contains(&-1));
/// assert!(!nonneg.is_bounded() && Bounds::closed(0, 1).is_bounded());
/// assert_eq!(nonneg.to_string(), "[0, ∞)");
/// assert_eq!(Bounds::<i32>::free().to_string(), "(-∞, ∞)");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bounds<T> {
    /// The lower bound, if any (included).
    pub lower: Option<T>,
    /// The upper bound, if any (included).
    pub upper: Option<T>,
}

impl<T> Bounds<T> {
    /// `lower ≤ x ≤ upper`.
    pub const fn closed(lower: T, upper: T) -> Self {
        Bounds {
            lower: Some(lower),
            upper: Some(upper),
        }
    }

    /// `x ≥ lower`.
    pub const fn at_least(lower: T) -> Self {
        Bounds {
            lower: Some(lower),
            upper: None,
        }
    }

    /// `x ≤ upper`.
    pub const fn at_most(upper: T) -> Self {
        Bounds {
            lower: None,
            upper: Some(upper),
        }
    }

    /// No constraint on either side.
    pub const fn free() -> Self {
        Bounds {
            lower: None,
            upper: None,
        }
    }

    /// Are both sides bounded?
    pub const fn is_bounded(&self) -> bool {
        self.lower.is_some() && self.upper.is_some()
    }

    /// Borrow both bounds.
    pub const fn as_ref(&self) -> Bounds<&T> {
        Bounds {
            lower: self.lower.as_ref(),
            upper: self.upper.as_ref(),
        }
    }

    /// Apply `f` to whichever bounds are present.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Bounds<U> {
        Bounds {
            lower: self.lower.map(&mut f),
            upper: self.upper.map(&mut f),
        }
    }

    /// The closed interval `[lower, upper]` when both sides are bounded.
    pub fn into_interval(self) -> Option<Interval<T>> {
        match (self.lower, self.upper) {
            (Some(lower), Some(upper)) => Some(Interval::closed(lower, upper)),
            _ => None,
        }
    }
}

impl<T: PartialOrd> Bounds<T> {
    /// `lower ≤ x ≤ upper`, each side only if present.
    pub fn contains(&self, x: &T) -> bool {
        self.lower.as_ref().is_none_or(|lo| lo <= x) && self.upper.as_ref().is_none_or(|hi| x <= hi)
    }

    /// Is there no `x` satisfying the constraint?  Only `lower > upper`
    /// (or a NaN side) is; an unbounded side never is.
    pub fn is_empty(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(lo), Some(hi)) => !(lo <= hi),
            _ => false,
        }
    }
}

impl<T> From<Interval<T>> for Bounds<T> {
    /// The closure `[lower, upper]` of the interval (the kind is dropped —
    /// constraint boxes are closed).
    fn from(iv: Interval<T>) -> Self {
        Bounds::closed(iv.lower, iv.upper)
    }
}

impl<T: fmt::Display> fmt::Display for Bounds<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.lower {
            Some(lo) => write!(f, "[{lo}, ")?,
            None => write!(f, "(-∞, ")?,
        }
        match &self.upper {
            Some(hi) => write!(f, "{hi}]"),
            None => write!(f, "∞)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_membership() {
        let closed = Interval::closed(1, 4);
        let open = Interval::open(1, 4);
        let left = Interval::left_open(1, 4);
        let right = Interval::right_open(1, 4);
        for iv in [closed, open, left, right] {
            assert!(iv.contains(&2) && !iv.contains(&0) && !iv.contains(&5));
            assert!(iv.is_ordered() && !iv.is_empty() && !iv.is_point());
            assert_eq!(iv.width(), 3);
        }
        assert_eq!(
            [closed, open, left, right].map(|iv| iv.contains(&1)),
            [true, false, false, true]
        );
        assert_eq!(
            [closed, open, left, right].map(|iv| iv.contains(&4)),
            [true, false, true, false]
        );
        assert_eq!(
            [closed, open, left, right].map(|iv| iv.to_string()),
            ["[1, 4]", "(1, 4)", "(1, 4]", "[1, 4)"]
        );
        assert_ne!(closed, open);
        assert_eq!(open.with_kind(IntervalKind::Closed), closed);
    }

    #[test]
    fn kind_helpers() {
        use IntervalKind::*;
        assert_eq!(from_ends(true, false), LeftOpen);
        assert_eq!(from_ends(false, true), RightOpen);
        assert!(Open.lower_open() && Open.upper_open());
        assert!(!Closed.lower_open() && !Closed.upper_open());
        assert_eq!(LeftOpen.reversed(), RightOpen);
        assert_eq!(Open.reversed(), Open);
        assert_eq!(Closed.with_lower_open(true), LeftOpen);
        assert_eq!(LeftOpen.with_upper_open(true), Open);
        fn from_ends(a: bool, b: bool) -> IntervalKind {
            IntervalKind::from_open_ends(a, b)
        }
    }

    #[test]
    fn degenerate_and_reversed() {
        assert!(!Interval::closed(3, 3).is_empty());
        assert!(Interval::closed(3, 3).is_point());
        assert_eq!(Interval::point(3), Interval::closed(3, 3));
        assert!(Interval::open(3, 3).is_empty());
        assert!(Interval::left_open(3, 3).is_empty());
        assert!(Interval::right_open(3, 3).is_empty());
        assert!(Interval::closed(4, 3).is_empty());
        assert!(!Interval::closed(4, 3).is_ordered());
        assert!(Interval::closed(f64::NAN, 1.0).is_empty());
        // A decreasing map: x ↦ −x sends [1, 4) to (−4, −1].
        assert_eq!(
            Interval::right_open(1, 4).map(|x| -x).reversed(),
            Interval::left_open(-4, -1)
        );
    }

    #[test]
    fn helpers_and_conversions() {
        let iv = Interval::left_open(1, 4);
        assert_eq!(iv.map(|x| x * 2), Interval::left_open(2, 8));
        assert_eq!(iv.as_ref(), Interval::left_open(&1, &4));
        assert_eq!(iv.as_ref().cloned(), iv);
        assert_eq!(iv.into_pair(), (1, 4));
        assert_eq!(Interval::<f64>::default(), Interval::closed(0.0, 0.0));
        assert_eq!(Interval::from(2..=5), Interval::closed(2, 5));
        assert_eq!(Interval::from(2..5), Interval::right_open(2, 5));
        // RangeBounds: usable wherever std takes a range.
        assert_eq!(iv.start_bound(), Bound::Excluded(&1));
        assert_eq!(iv.end_bound(), Bound::Included(&4));
        let map: std::collections::BTreeMap<i32, &str> =
            [(1, "a"), (2, "b"), (4, "d"), (5, "e")].into();
        let keys: Vec<i32> = map.range(iv).map(|(k, _)| *k).collect();
        assert_eq!(keys, [2, 4]);
    }

    #[test]
    fn bounds() {
        let b = Bounds::at_least(0);
        assert!(b.contains(&0) && b.contains(&1_000_000) && !b.contains(&-1));
        assert!(!b.is_bounded() && !b.is_empty());
        assert_eq!(b.into_interval(), None);
        let c = Bounds::closed(2, 5);
        assert!(c.is_bounded() && c.contains(&5) && !c.contains(&6));
        assert_eq!(c.into_interval(), Some(Interval::closed(2, 5)));
        assert!(Bounds::closed(5, 2).is_empty());
        assert!(!Bounds::closed(5, 5).is_empty());
        let free = Bounds::<i32>::free();
        assert!(free.contains(&i32::MIN) && free.contains(&i32::MAX));
        assert_eq!(Bounds::at_most(3).map(|x| x * 2), Bounds::at_most(6));
        assert_eq!(Bounds::from(Interval::open(1, 2)), Bounds::closed(1, 2));
        assert_eq!(Bounds::closed(1, 2).as_ref(), Bounds::closed(&1, &2));
        assert_eq!(
            [
                b.to_string(),
                c.to_string(),
                free.to_string(),
                Bounds::at_most(3).to_string()
            ],
            ["[0, ∞)", "[2, 5]", "(-∞, ∞)", "(-∞, 3]"]
        );
        assert_eq!(Bounds::<i32>::default(), free);
    }
}
