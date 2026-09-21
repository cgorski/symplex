//! Events on one variable as regions of the line.
//!
//! Two routes, tried in order:
//!
//! 1. **Linear fast path** — a relation or a conjunction of relations with
//!    the variable alone on one side and a variable-free bound on the
//!    other (`X > a`, `a ≤ X ∧ X < b`, `X = a`).  The bounds may be
//!    symbolic; the result is one interval (or one point inside it).
//! 2. **The set machinery** — when every bound is numeric, the event is
//!    solved with [`reduce_inequalities`](crate::api::expr::SetEx::reduce_inequalities),
//!    which handles disjunctions, negations and non-linear atoms
//!    (`X² < 1`, `|X| > 2`, `X < −1 ∨ X > 1`), and the resulting set in
//!    normal form (disjoint intervals and points) becomes the region.
//!
//! Anything else — a non-linear or disjunctive event with symbolic
//! bounds — is `NotImplemented`.

use crate::api::expr::{BoolEx, Ex, SetEx};
use crate::base::errors::SymplexError;
use crate::base::interval::{Interval, IntervalKind};
use crate::base::node::ExprNode;

use super::support::{Kind, Piece, Support};

/// The kind of a relation node: `l > r`, `l ≥ r`, `l = r`.  Relations are
/// stored canonically as `Gt`/`Ge` (`X < a` is `a > X`) and `Eq_`.
#[derive(Clone, Copy)]
pub(crate) enum Rel {
    Gt,
    Ge,
    Eq,
}

/// One relation of an event, with the relation node itself as a `BoolEx`
/// so that per-variable conjunctions can be rebuilt without re-encoding.
pub(crate) struct Relation {
    pub(crate) kind: Rel,
    pub(crate) lhs: Ex,
    pub(crate) rhs: Ex,
    pub(crate) node: BoolEx,
}

/// Flatten an event — a relation or a (nested) conjunction of relations —
/// into its relations; `None` for any other shape (`Or`, `Not`, …).
pub(crate) fn flatten_relations(event: &BoolEx) -> Option<Vec<Relation>> {
    let inner = event.inner.read();
    let arena = &inner.arena;
    let mut rels = Vec::new();
    let mut stack = vec![event.raw_id()];
    while let Some(id) = stack.pop() {
        let (kind, l, r) = match arena.node(id) {
            ExprNode::And(parts) => {
                stack.extend(parts.iter().copied());
                continue;
            }
            ExprNode::Gt(l, r) => (Rel::Gt, *l, *r),
            ExprNode::Ge(l, r) => (Rel::Ge, *l, *r),
            ExprNode::Eq_(l, r) => (Rel::Eq, *l, *r),
            _ => return None,
        };
        rels.push(Relation {
            kind,
            lhs: event.wrap_as(l),
            rhs: event.wrap_as(r),
            node: event.wrap(id),
        });
    }
    Some(rels)
}

/// Is the event the constant `True` / `False`?
fn constant(event: &BoolEx) -> Option<bool> {
    let inner = event.inner.read();
    match inner.arena.node(event.raw_id()) {
        ExprNode::BoolTrue => Some(true),
        ExprNode::BoolFalse => Some(false),
        _ => None,
    }
}

fn unsupported(event: &BoolEx) -> SymplexError {
    SymplexError::NotImplemented(format!(
        "the event `{event}`: supported are relations `X < a`, `X ≤ a`, `X > a`, `X ≥ a`, \
         `X = a` in the random variable and their conjunctions (any bounds), and any \
         boolean combination of relations in `X` with numeric bounds (`X² < 1`, \
         `X < -1 ∨ X > 1`)"
    ))
}

/// The region of the line an event in the single variable `x` describes.
pub(crate) fn event_region(x: &Ex, event: &BoolEx) -> Result<Support, SymplexError> {
    let ctx = x.context();
    match constant(event) {
        Some(true) => return Ok(Support::reals(&ctx)),
        Some(false) => return Ok(Support::from_pieces(Kind::Continuous, vec![])),
        None => {}
    }
    if let Some(rels) = flatten_relations(event)
        && let Some(region) = linear_region(x, &rels)
    {
        return Ok(region);
    }
    // Every symbol other than `x` would be a symbolic bound the set
    // machinery cannot place on the line.
    if event.free_symbols().iter().any(|s| s != x) {
        return Err(unsupported(event));
    }
    let set = SetEx::reduce_inequalities(std::slice::from_ref(event), x).map_err(|e| {
        SymplexError::NotImplemented(format!(
            "the event `{event}` could not be reduced to a set: {e}"
        ))
    })?;
    Support::from_set(Kind::Continuous, &set).ok_or_else(|| unsupported(event))
}

/// One interval (with an optional point inside it) from a conjunction of
/// linear relations in `x`; `None` if any relation has another shape.
fn linear_region(x: &Ex, rels: &[Relation]) -> Option<Support> {
    let ctx = x.context();
    let mut lo = ctx.neg_infinity();
    let mut hi = ctx.infinity();
    let mut lo_open = true;
    let mut hi_open = true;
    let mut point: Option<Ex> = None;
    for rel in rels {
        let (bound, x_on_left) = if rel.lhs == *x && !rel.rhs.contains(x) {
            (rel.rhs.clone(), true)
        } else if rel.rhs == *x && !rel.lhs.contains(x) {
            (rel.lhs.clone(), false)
        } else {
            return None;
        };
        // `[lo, ∞)` or `(lo, ∞)` as a one-interval support.
        let above = |lo: Ex, open: bool| {
            Support::from_pieces(
                Kind::Continuous,
                vec![Piece::Interval(Interval {
                    lower: lo,
                    upper: ctx.infinity(),
                    kind: IntervalKind::from_open_ends(open, true),
                })],
            )
        };
        // `(−∞, hi]` or `(−∞, hi)`.
        let below = |hi: Ex, open: bool| {
            Support::from_pieces(
                Kind::Continuous,
                vec![Piece::Interval(Interval {
                    lower: ctx.neg_infinity(),
                    upper: hi,
                    kind: IntervalKind::from_open_ends(true, open),
                })],
            )
        };
        let raise = |lo: &mut Ex, lo_open: &mut bool, a: Ex, strict: bool| {
            let one = above(lo.clone(), *lo_open);
            let other = above(a, strict);
            // `intersect` never fails on two intervals.
            if let Some(r) = one.intersect(&other)
                && let Some(iv) = r.as_interval()
            {
                *lo = iv.lower.clone();
                *lo_open = iv.kind.lower_open();
            } else {
                // Provably empty: a lower end above +∞ cannot occur, so
                // this is `lo > hi` with numeric values — represent as an
                // empty interval.
                *lo = ctx.infinity();
                *lo_open = true;
            }
        };
        let lower = |hi: &mut Ex, hi_open: &mut bool, b: Ex, strict: bool| {
            let one = below(hi.clone(), *hi_open);
            let other = below(b, strict);
            if let Some(r) = one.intersect(&other)
                && let Some(iv) = r.as_interval()
            {
                *hi = iv.upper.clone();
                *hi_open = iv.kind.upper_open();
            } else {
                *hi = ctx.neg_infinity();
                *hi_open = true;
            }
        };
        match (rel.kind, x_on_left) {
            // X > a / X ≥ a
            (Rel::Gt, true) => raise(&mut lo, &mut lo_open, bound, true),
            (Rel::Ge, true) => raise(&mut lo, &mut lo_open, bound, false),
            // a > X / a ≥ X
            (Rel::Gt, false) => lower(&mut hi, &mut hi_open, bound, true),
            (Rel::Ge, false) => lower(&mut hi, &mut hi_open, bound, false),
            (Rel::Eq, _) => {
                match &point {
                    // Two different points: empty unless they agree.
                    Some(p) => match p.equals(&bound) {
                        Some(true) => {}
                        Some(false) => {
                            return Some(Support::from_pieces(Kind::Continuous, vec![]));
                        }
                        None => return None,
                    },
                    None => point = Some(bound),
                }
            }
        }
    }
    // The interval, possibly empty.
    let interval = Support::from_pieces(
        Kind::Continuous,
        vec![Piece::Interval(Interval {
            lower: lo,
            upper: hi,
            kind: IntervalKind::from_open_ends(lo_open, hi_open),
        })],
    );
    let interval = interval.intersect(&Support::reals(&ctx))?; // drops a provably empty piece
    match point {
        None => Some(interval),
        Some(p) => {
            // The point clipped to the interval: in, out, or undecided
            // (kept — the mass function decides).
            let region = Support::from_pieces(Kind::Continuous, vec![Piece::Point(p.clone())]);
            if interval.is_empty() {
                return Some(interval);
            }
            match interval.contains(&p) {
                Some(false) => Some(Support::from_pieces(Kind::Continuous, vec![])),
                _ => Some(region),
            }
        }
    }
}
