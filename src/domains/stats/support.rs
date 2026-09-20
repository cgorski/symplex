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
    /// An interval; either end may be `±∞` (then it is open there).  For
    /// a [`Kind::Discrete`] support the interval stands for the *integers*
    /// in it.
    Interval {
        /// Lower end.
        lo: Ex,
        /// Upper end.
        hi: Ex,
        /// `true` for `(lo, …`.
        lo_open: bool,
        /// `true` for `…, hi)`.
        hi_open: bool,
    },
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

/// The larger of two lower ends (`−∞` loses; numeric ends compare; symbolic
/// ends become `max(a, b)`).  The second value says whether the result is
/// open: when one end is strictly larger its own openness is kept, when
/// they coincide either being open makes the result open, and when the
/// order is unknown the result is open if either is (irrelevant for a
/// continuous variable; a discrete one has been normalised to closed
/// integer ends before any intersection).
fn max_lo(a: &Ex, a_open: bool, b: &Ex, b_open: bool) -> (Ex, bool) {
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

/// The smaller of two upper ends; see [`max_lo`].
fn min_hi(a: &Ex, a_open: bool, b: &Ex, b_open: bool) -> (Ex, bool) {
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

/// Is the interval `lo..hi` (with the given openness) known to be empty?
/// `Some(true)` when `hi < lo`, or `hi = lo` with an open end;
/// `Some(false)` when `hi > lo`; `None` when the order is unknown.
fn interval_empty(lo: &Ex, hi: &Ex, lo_open: bool, hi_open: bool) -> Option<bool> {
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
            Some(true) => Some(lo_open || hi_open),
            Some(false) => Some(true),
            None => None,
        },
        None => None,
    }
}

/// Does the interval contain `v`?  `None` when undecidable.
fn interval_contains(lo: &Ex, hi: &Ex, lo_open: bool, hi_open: bool, v: &Ex) -> Option<bool> {
    let above = if is_neg_inf(lo) {
        Some(true)
    } else {
        let d = v - lo;
        if lo_open {
            d.is_positive()
        } else {
            d.is_nonnegative()
        }
    };
    let below = if is_pos_inf(hi) {
        Some(true)
    } else {
        let d = hi - v;
        if hi_open {
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
        let lo_open = is_neg_inf(&lo);
        let hi_open = is_pos_inf(&hi);
        Support {
            kind: Kind::Continuous,
            pieces: vec![Piece::Interval {
                lo,
                hi,
                lo_open,
                hi_open,
            }],
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
        let lo_open = is_neg_inf(&lo);
        let hi_open = is_pos_inf(&hi);
        Support {
            kind: Kind::Discrete,
            pieces: vec![Piece::Interval {
                lo,
                hi,
                lo_open,
                hi_open,
            }],
        }
    }

    /// An explicit list of values (discrete; they need not be integers).
    pub fn points(values: Vec<Ex>) -> Self {
        Support {
            kind: Kind::Discrete,
            pieces: values.into_iter().map(Piece::Point).collect(),
        }
    }

    /// A support from explicit pieces.
    pub fn from_pieces(kind: Kind, pieces: Vec<Piece>) -> Self {
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

    /// `true` when the support is a single interval (no points).
    pub fn is_interval(&self) -> bool {
        matches!(self.pieces.as_slice(), [Piece::Interval { .. }])
    }

    /// The single interval's ends, when the support is one interval.
    pub fn as_interval(&self) -> Option<(&Ex, &Ex, bool, bool)> {
        match self.pieces.as_slice() {
            [
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                },
            ] => Some((lo, hi, *lo_open, *hi_open)),
            _ => None,
        }
    }

    /// The values, when the support is a list of points.
    pub fn as_points(&self) -> Option<Vec<Ex>> {
        self.pieces
            .iter()
            .map(|p| match p {
                Piece::Point(v) => Some(v.clone()),
                Piece::Interval { .. } => None,
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
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                } => {
                    let in_interval = interval_contains(lo, hi, *lo_open, *hi_open, v);
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
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                } => {
                    let ctx = lo.context();
                    let lo = if is_neg_inf(lo) {
                        lo.clone()
                    } else if *lo_open {
                        (lo.floor() + ctx.one()).simplify()
                    } else {
                        lo.ceiling().simplify()
                    };
                    let hi = if is_pos_inf(hi) {
                        hi.clone()
                    } else if *hi_open {
                        (hi.ceiling() - ctx.one()).simplify()
                    } else {
                        hi.floor().simplify()
                    };
                    Piece::Interval {
                        lo_open: is_neg_inf(&lo),
                        hi_open: is_pos_inf(&hi),
                        lo,
                        hi,
                    }
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
            for b in &other.pieces {
                match (a, b) {
                    (
                        Piece::Interval {
                            lo: alo,
                            hi: ahi,
                            lo_open: alo_o,
                            hi_open: ahi_o,
                        },
                        Piece::Interval {
                            lo: blo,
                            hi: bhi,
                            lo_open: blo_o,
                            hi_open: bhi_o,
                        },
                    ) => {
                        let (lo, lo_open) = max_lo(alo, *alo_o, blo, *blo_o);
                        let (hi, hi_open) = min_hi(ahi, *ahi_o, bhi, *bhi_o);
                        if interval_empty(&lo, &hi, lo_open, hi_open) != Some(true) {
                            out.push(Piece::Interval {
                                lo,
                                hi,
                                lo_open,
                                hi_open,
                            });
                        }
                    }
                    (Piece::Point(v), Piece::Interval { .. }) => {
                        let single = Support::from_pieces(self.kind, vec![b.clone()]);
                        // A table value is a member by listing, so only the
                        // interval test applies (no lattice test).
                        let single = single.with_kind(Kind::Continuous);
                        if single.contains(v)? {
                            out.push(Piece::Point(v.clone()));
                        }
                    }
                    (Piece::Interval { .. }, Piece::Point(v)) => {
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
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                } => {
                    let s = ctx.interval(lo, hi, *lo_open, *hi_open);
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
            .map(|(lo, hi, lo_open, hi_open)| {
                if lo == hi && !lo_open && !hi_open {
                    Piece::Point(lo)
                } else {
                    Piece::Interval {
                        lo,
                        hi,
                        lo_open,
                        hi_open,
                    }
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
                Piece::Interval {
                    lo,
                    hi,
                    lo_open,
                    hi_open,
                } => {
                    if !first {
                        f.write_str(" ∪ ")?;
                    }
                    first = false;
                    let l = if *lo_open { "(" } else { "[" };
                    let r = if *hi_open { ")" } else { "]" };
                    write!(f, "{l}{lo}, {hi}{r}")?;
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
            && self
                .pieces
                .iter()
                .any(|p| matches!(p, Piece::Interval { .. }))
        {
            f.write_str(" ∩ ℤ")?;
        }
        Ok(())
    }
}
