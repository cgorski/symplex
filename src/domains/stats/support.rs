//! Where a distribution lives, and the regions events cut out of the line:
//! a finite union of intervals and points with expression endpoints, on
//! the continuum or on the integer lattice.
//!
//! `SetEx` is the crate's general set type, but its normal form leaves
//! intersections with *symbolic* endpoints unevaluated
//! (`(a, ∞) ∩ [0, ∞)` stays a `SetIntersection`), and `Uniform(a, b)` or
//! `P(X > a)` need exactly those.  [`Support`] is the small typed region
//! the statistics module computes with — every clipping of an event to a
//! support happens in [`Support::intersect`], once — and it converts to
//! and from `SetEx` at the boundary ([`to_set`](Support::to_set),
//! [`from_set`](Support::from_set)).

use std::fmt;

use crate::api::context::Context;
use crate::api::expr::{Ex, SetEx};
use crate::base::interval::{Interval, IntervalKind};

/// Whether a distribution puts its mass on intervals of ℝ (a density) or
/// on isolated points (a probability mass function).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    /// A density on intervals of the real line.
    Continuous,
    /// Point masses: on the integer lattice within intervals, or on an
    /// explicit list of values.
    Discrete,
}

/// One connected part of a [`Support`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Piece {
    /// An interval; either end may be `±∞` (then it is open there —
    /// [`Support::from_pieces`] and the other constructors enforce this).
    /// For a [`Kind::Discrete`] support the interval stands for the
    /// *integers* in it.
    Interval(Interval<Ex>),
    /// A single value (not necessarily an integer, even on a discrete
    /// support: a `Finite` table lists arbitrary values).
    Point(Ex),
}

/// A finite union of intervals and points on the line, with the
/// [`Kind`] that says how a density over it is read.
///
/// Pieces are kept as given (in particular they are not merged or sorted
/// when endpoints are symbolic); the constructors produce disjoint pieces,
/// [`intersect`](Self::intersect) preserves disjointness, and
/// [`from_set`](Self::from_set) takes the disjoint normal form the set
/// module computes.
#[derive(Clone, Debug, PartialEq)]
pub struct Support {
    kind: Kind,
    pieces: Vec<Piece>,
}

/// Is `e` the node `+∞`?
pub(crate) fn is_pos_inf(e: &Ex) -> bool {
    *e == e.context().infinity()
}

/// Is `e` the node `−∞`?
pub(crate) fn is_neg_inf(e: &Ex) -> bool {
    *e == e.context().neg_infinity()
}

/// The interval with a `−∞` lower end and a `+∞` upper end forced open,
/// the invariant every [`Support`] constructor maintains.
fn with_infinite_ends_open(iv: Interval<Ex>) -> Interval<Ex> {
    let kind = IntervalKind::from_open_ends(
        iv.kind.lower_open() || is_neg_inf(&iv.lower),
        iv.kind.upper_open() || is_pos_inf(&iv.upper),
    );
    iv.with_kind(kind)
}

/// The larger of the two intervals' lower ends (`−∞` loses; numeric ends
/// compare; symbolic ends become `max(a, b)`), and whether it is open:
/// when one end is strictly larger its own openness is kept, when they
/// coincide either being open makes the result open, and when the order
/// is unknown the result is open if either is (irrelevant for a
/// continuous variable; a discrete one has been normalised to closed
/// integer ends before any intersection).
fn max_lo(a: &Interval<Ex>, b: &Interval<Ex>) -> (Ex, bool) {
    let (a_open, b_open) = (a.kind.lower_open(), b.kind.lower_open());
    let (a, b) = (&a.lower, &b.lower);
    if is_neg_inf(a) {
        return (b.clone(), b_open);
    }
    if is_neg_inf(b) {
        return (a.clone(), a_open);
    }
    match (a - b).is_positive() {
        Some(true) => (a.clone(), a_open),
        Some(false) => match (a - b).is_zero() {
            Some(true) => (a.clone(), a_open || b_open),
            _ => (b.clone(), b_open),
        },
        None => (a.max_with(b), a_open || b_open),
    }
}

/// The smaller of the two intervals' upper ends; see [`max_lo`].
fn min_hi(a: &Interval<Ex>, b: &Interval<Ex>) -> (Ex, bool) {
    let (a_open, b_open) = (a.kind.upper_open(), b.kind.upper_open());
    let (a, b) = (&a.upper, &b.upper);
    if is_pos_inf(a) {
        return (b.clone(), b_open);
    }
    if is_pos_inf(b) {
        return (a.clone(), a_open);
    }
    match (b - a).is_positive() {
        Some(true) => (a.clone(), a_open),
        Some(false) => match (a - b).is_zero() {
            Some(true) => (a.clone(), a_open || b_open),
            _ => (b.clone(), b_open),
        },
        None => (a.min_with(b), a_open || b_open),
    }
}

/// Is the interval known to be empty?  `Some(true)` when `hi < lo`, or
/// `hi = lo` with an open end; `Some(false)` when `hi > lo`; `None` when
/// the order is unknown.
fn interval_empty(iv: &Interval<Ex>) -> Option<bool> {
    let (lo, hi) = (&iv.lower, &iv.upper);
    if is_neg_inf(lo) || is_pos_inf(hi) {
        return Some(false);
    }
    if is_pos_inf(lo) || is_neg_inf(hi) {
        return Some(true);
    }
    let d = hi - lo;
    match d.is_positive() {
        Some(true) => Some(false),
        Some(false) => match d.is_zero() {
            Some(true) => Some(iv.kind != IntervalKind::Closed),
            Some(false) => Some(true),
            None => None,
        },
        None => None,
    }
}

/// Does the interval contain `v`?  `None` when undecidable.
fn interval_contains(iv: &Interval<Ex>, v: &Ex) -> Option<bool> {
    let above = if is_neg_inf(&iv.lower) {
        Some(true)
    } else {
        let d = v - &iv.lower;
        if iv.kind.lower_open() {
            d.is_positive()
        } else {
            d.is_nonnegative()
        }
    };
    let below = if is_pos_inf(&iv.upper) {
        Some(true)
    } else {
        let d = &iv.upper - v;
        if iv.kind.upper_open() {
            d.is_positive()
        } else {
            d.is_nonnegative()
        }
    };
    match (above, below) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

impl Support {
    /// The whole real line (continuous).
    pub fn reals(ctx: &Context) -> Self {
        Support::interval(ctx.neg_infinity(), ctx.infinity())
    }

    /// The closed interval `[lo, hi]` (continuous; `±∞` ends are open).
    pub fn interval(lo: Ex, hi: Ex) -> Self {
        Support {
            kind: Kind::Continuous,
            pieces: vec![Piece::Interval(with_infinite_ends_open(Interval::closed(
                lo, hi,
            )))],
        }
    }

    /// The half-line `[lo, ∞)` (continuous).
    pub fn half_line(lo: Ex) -> Self {
        let inf = lo.context().infinity();
        Support::interval(lo, inf)
    }

    /// The integers `lo ..= hi` (discrete); `None` means unbounded on that
    /// side.
    pub fn integers(ctx: &Context, lo: Option<Ex>, hi: Option<Ex>) -> Self {
        let lo = lo.unwrap_or_else(|| ctx.neg_infinity());
        let hi = hi.unwrap_or_else(|| ctx.infinity());
        Support {
            kind: Kind::Discrete,
            pieces: vec![Piece::Interval(with_infinite_ends_open(Interval::closed(
                lo, hi,
            )))],
        }
    }

    /// An explicit list of values (discrete; they need not be integers).
    pub fn points(values: Vec<Ex>) -> Self {
        Support {
            kind: Kind::Discrete,
            pieces: values.into_iter().map(Piece::Point).collect(),
        }
    }

    /// A support from explicit pieces.  An interval end at `−∞` / `+∞` is
    /// made open, as the other constructors do; the pieces are otherwise
    /// kept as given.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::{Kind, Piece, Support};
    ///
    /// let ctx = Context::new();
    /// let tail = Support::from_pieces(
    ///     Kind::Continuous,
    ///     vec![Piece::Interval(Interval::closed(ctx.int(2), ctx.infinity()))],
    /// );
    /// assert_eq!(tail.to_string(), "[2, oo)");
    /// assert_eq!(tail.as_interval().map(|iv| iv.kind), Some(IntervalKind::RightOpen));
    /// ```
    pub fn from_pieces(kind: Kind, pieces: Vec<Piece>) -> Self {
        let pieces = pieces
            .into_iter()
            .map(|p| match p {
                Piece::Interval(iv) => Piece::Interval(with_infinite_ends_open(iv)),
                Piece::Point(v) => Piece::Point(v),
            })
            .collect();
        Support { kind, pieces }
    }

    /// Continuous or discrete.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// The pieces, as given.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// The same pieces read with another kind.
    pub fn with_kind(mut self, kind: Kind) -> Self {
        self.kind = kind;
        self
    }

    /// `true` when there are no pieces.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The mass of `f` in `var` between `lo` and `hi`, read by this
    /// support's kind: `∫_lo^hi f dvar` for a density, `Σ_{var=lo}^{hi} f`
    /// for a mass function on the integer lattice.  The one place the kind
    /// decides between integration and summation.
    pub(crate) fn accumulate(&self, f: &Ex, var: &Ex, lo: &Ex, hi: &Ex) -> Ex {
        match self.kind {
            Kind::Continuous => f.integrate_definite(var, lo, hi),
            Kind::Discrete => f.summation(var, lo, hi),
        }
    }

    /// `true` when the support is a single interval (no points).
    pub fn is_interval(&self) -> bool {
        matches!(self.pieces.as_slice(), [Piece::Interval(_)])
    }

    /// The single interval, when the support is one interval.
    ///
    /// ```
    /// use symplex::prelude::*;
    /// use symplex::stats::Support;
    ///
    /// let ctx = Context::new();
    /// let half = Support::half_line(ctx.int(0));
    /// let iv = half.as_interval().unwrap();
    /// assert_eq!(iv.lower, ctx.int(0));
    /// assert_eq!(iv.kind, IntervalKind::RightOpen); // [0, oo)
    /// assert!(Support::points(vec![ctx.int(1)]).as_interval().is_none());
    /// ```
    pub fn as_interval(&self) -> Option<&Interval<Ex>> {
        match self.pieces.as_slice() {
            [Piece::Interval(iv)] => Some(iv),
            _ => None,
        }
    }

    /// The values, when the support is a list of points.
    pub fn as_points(&self) -> Option<Vec<Ex>> {
        self.pieces
            .iter()
            .map(|p| match p {
                Piece::Point(v) => Some(v.clone()),
                Piece::Interval(_) => None,
            })
            .collect()
    }

    /// Is `v` in the support?  `None` when undecidable (symbolic ends or
    /// values).  On a discrete lattice support a non-integer `v` is out.
    pub fn contains(&self, v: &Ex) -> Option<bool> {
        let mut undecided = false;
        for p in &self.pieces {
            let inside = match p {
                Piece::Point(w) => w.equals(v),
                Piece::Interval(iv) => {
                    let in_interval = interval_contains(iv, v);
                    if self.kind == Kind::Discrete {
                        match (in_interval, v.is_integer()) {
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            (Some(true), Some(true)) => Some(true),
                            _ => None,
                        }
                    } else {
                        in_interval
                    }
                }
            };
            match inside {
                Some(true) => return Some(true),
                Some(false) => {}
                None => undecided = true,
            }
        }
        if undecided { None } else { Some(false) }
    }

    /// For a discrete support: replace open ends and non-integer ends by
    /// the tightest closed integer ends (`(a, …` → `⌊a⌋ + 1`, `[a, …` →
    /// `⌈a⌉`, symmetrically above), so that later intersections compare
    /// integers with integers.  A continuous support is returned unchanged.
    pub fn normalize_lattice(&self) -> Self {
        if self.kind != Kind::Discrete {
            return self.clone();
        }
        let pieces = self
            .pieces
            .iter()
            .map(|p| match p {
                Piece::Interval(iv) => {
                    let ctx = iv.lower.context();
                    let lo = if is_neg_inf(&iv.lower) {
                        iv.lower.clone()
                    } else if iv.kind.lower_open() {
                        (iv.lower.floor() + ctx.one()).simplify()
                    } else {
                        iv.lower.ceiling().simplify()
                    };
                    let hi = if is_pos_inf(&iv.upper) {
                        iv.upper.clone()
                    } else if iv.kind.upper_open() {
                        (iv.upper.ceiling() - ctx.one()).simplify()
                    } else {
                        iv.upper.floor().simplify()
                    };
                    // Closed integer ends; only an infinite end stays open.
                    Piece::Interval(with_infinite_ends_open(Interval::closed(lo, hi)))
                }
                Piece::Point(v) => Piece::Point(v.clone()),
            })
            .collect();
        Support {
            kind: self.kind,
            pieces,
        }
    }

    /// The intersection with a region of the line, keeping this support's
    /// kind (the region's kind is ignored: an event is a subset of ℝ).
    /// Pieces known to be empty are dropped.  `None` when a point's
    /// membership cannot be decided (symbolic table values against
    /// symbolic bounds); the caller reports that as unsupported.
    pub fn intersect(&self, region: &Support) -> Option<Self> {
        let me = self.normalize_lattice();
        let other = if self.kind == Kind::Discrete {
            region.clone().with_kind(Kind::Discrete).normalize_lattice()
        } else {
            region.clone()
        };
        let mut out = Vec::new();
        for a in &me.pieces {
            // `normalize_lattice` maps pieces one to one, so `raw` is the
            // same piece of `region` before any lattice rounding.
            for (b, raw) in other.pieces.iter().zip(&region.pieces) {
                match (a, b) {
                    (Piece::Interval(ia), Piece::Interval(ib)) => {
                        let (lo, lo_open) = max_lo(ia, ib);
                        let (hi, hi_open) = min_hi(ia, ib);
                        let iv = Interval {
                            lower: lo,
                            upper: hi,
                            kind: IntervalKind::from_open_ends(lo_open, hi_open),
                        };
                        if interval_empty(&iv) != Some(true) {
                            out.push(Piece::Interval(iv));
                        }
                    }
                    (Piece::Point(v), Piece::Interval(_)) => {
                        // A table value is a member by listing, so only the
                        // interval test applies, against the region's ends
                        // as given: the value need not be an integer, and
                        // rounding `(1, 5/2]` to `[2, 2]` would drop `5/2`.
                        let single = Support::from_pieces(Kind::Continuous, vec![raw.clone()]);
                        if single.contains(v)? {
                            out.push(Piece::Point(v.clone()));
                        }
                    }
                    (Piece::Interval(_), Piece::Point(v)) => {
                        let single = Support::from_pieces(self.kind, vec![a.clone()]);
                        match single.contains(v) {
                            Some(true) => out.push(Piece::Point(v.clone())),
                            Some(false) => {}
                            // A symbolic point against the support: keep it —
                            // the caller's promise that `X = a` names a
                            // value of the variable (the mass function
                            // decides).
                            None => out.push(Piece::Point(v.clone())),
                        }
                    }
                    (Piece::Point(v), Piece::Point(w)) => match v.equals(w) {
                        Some(true) => out.push(Piece::Point(v.clone())),
                        Some(false) => {}
                        None => return None,
                    },
                }
            }
        }
        Some(Support {
            kind: self.kind,
            pieces: out,
        })
    }

    /// The region as a `SetEx` (intervals and finite sets, unioned).
    pub fn to_set(&self, ctx: &Context) -> SetEx {
        let mut acc: Option<SetEx> = None;
        let mut points = Vec::new();
        for p in &self.pieces {
            match p {
                Piece::Interval(iv) => {
                    let s = ctx.interval(&iv.lower, &iv.upper, iv.kind);
                    acc = Some(match acc {
                        Some(a) => a.union(&s),
                        None => s,
                    });
                }
                Piece::Point(v) => points.push(v.clone()),
            }
        }
        if !points.is_empty() {
            let s = ctx.finite_set(&points);
            acc = Some(match acc {
                Some(a) => a.union(&s),
                None => s,
            });
        }
        acc.unwrap_or_else(|| ctx.empty_set())
    }

    /// A region from a `SetEx` in the set module's normal form (a finite
    /// union of intervals and points with numeric ends); `None` when the
    /// set is not of that shape (a `ConditionSet`, symbolic ends, …).
    pub fn from_set(kind: Kind, set: &SetEx) -> Option<Self> {
        if let Some(values) = set.as_finite_set() {
            return Some(Support {
                kind,
                pieces: values.into_iter().map(Piece::Point).collect(),
            });
        }
        let parts = set.as_intervals()?;
        let pieces = parts
            .into_iter()
            .map(|iv| {
                if iv.kind == IntervalKind::Closed && iv.lower == iv.upper {
                    Piece::Point(iv.lower)
                } else {
                    Piece::Interval(iv)
                }
            })
            .collect();
        Some(Support { kind, pieces })
    }
}

impl fmt::Display for Support {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pieces.is_empty() {
            return f.write_str("∅");
        }
        let mut points = Vec::new();
        let mut first = true;
        for p in &self.pieces {
            match p {
                Piece::Interval(iv) => {
                    if !first {
                        f.write_str(" ∪ ")?;
                    }
                    first = false;
                    // `Interval`'s own `Display` is `[lo, hi]` / `(lo, hi)` / …
                    write!(f, "{iv}")?;
                }
                Piece::Point(v) => points.push(v.to_string()),
            }
        }
        if !points.is_empty() {
            if !first {
                f.write_str(" ∪ ")?;
            }
            write!(f, "{{{}}}", points.join(", "))?;
        }
        if self.kind == Kind::Discrete
            && self.pieces.iter().any(|p| matches!(p, Piece::Interval(_)))
        {
            f.write_str(" ∩ ℤ")?;
        }
        Ok(())
    }
}
