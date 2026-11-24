//! Set constructors (canonical) (part of [`canon`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Set constructors (canonical)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a canonical `Interval` node.
///
/// Handles degenerate cases:
/// - If both endpoints are numeric and start > end → EmptySet.
/// - If both endpoints are numeric and start == end:
///   - both closed → FiniteSet({start})
///   - any open → EmptySet
pub(crate) fn canon_interval(
    arena: &mut Arena,
    start: ExprId,
    end: ExprId,
    kind: crate::base::interval::IntervalKind,
) -> ExprId {
    tracing::debug!(
        "canon_interval: start={:?}, end={:?}, flags={}",
        start,
        end,
        kind.flag_bits()
    );

    // Try to compare endpoints numerically.
    if let (Some(s), Some(e)) = (arena.as_num(start), arena.as_num(end)) {
        let s = s.clone();
        let e = e.clone();
        use std::cmp::Ordering;
        match s.cmp(&e) {
            Ordering::Greater => {
                // start > end → empty set
                return arena.intern(ExprNode::EmptySet);
            }
            Ordering::Equal => {
                // start == end
                if kind.lower_open() || kind.upper_open() {
                    // Any open endpoint on a point interval → empty
                    return arena.intern(ExprNode::EmptySet);
                } else {
                    // Both closed, single point → FiniteSet({start})
                    return arena.intern(ExprNode::FiniteSet(smallvec![start]));
                }
            }
            Ordering::Less => {
                // start < end: valid interval, fall through
            }
        }
    }

    arena.intern(ExprNode::Interval(start, end, kind))
}

/// Build a canonical `FiniteSet` node.
///
/// Sorts elements by sort key and deduplicates.
/// An empty set of elements returns EmptySet.
pub(crate) fn canon_finite_set(arena: &mut Arena, elements: &[ExprId]) -> ExprId {
    tracing::debug!("canon_finite_set: {} elements", elements.len());

    if elements.is_empty() {
        return arena.intern(ExprNode::EmptySet);
    }

    // Deduplicate (ExprIds are hash-consed, so equality is ExprId equality).
    let mut deduped: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut seen = FxHashSet::default();
    for &elem in elements {
        if seen.insert(elem) {
            deduped.push(elem);
        }
    }

    // Sort by sort key for canonical ordering.
    deduped.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if deduped.len() == 1 {
        // A single-element set is still a FiniteSet — don't collapse.
    }

    arena.intern(ExprNode::FiniteSet(deduped))
}

/// Build a canonical `SetUnion` node.
///
/// Follows the LatticeOp pattern:
/// 1. Flatten nested SetUnion.
/// 2. Remove EmptySet children (identity for union).
/// 3. Short-circuit on UniversalSet (absorbing element for union).
/// 4. Deduplicate and sort children by sort key, then put the real
///    intervals in ascending order of their lower ends
///    ([`order_real_intervals`]): `(−∞, −2) ∪ (2, ∞)`, `(0, 1) ∪ [2, 3]`.
/// 5. Single child → return that child directly.
///
/// Up to 0.51 the order was the sort key's alone, which prints `(2, ∞) ∪
/// (−∞, −2)` (the key of an interval starts with its kind, and `−∞` sorts
/// after the numbers), so the set evaluator kept its ascending unions as
/// raw nodes, not `==` to the canonical union of the same parts: `(0, 1) ∪
/// [2, 3]` and its JSON round trip `[2, 3] ∪ (0, 1)` compared unequal.
pub(crate) fn canon_set_union(arena: &mut Arena, sets: &[ExprId]) -> ExprId {
    tracing::debug!("canon_set_union: {} children", sets.len());

    let mut flat: SmallVec<[ExprId; 4]> = SmallVec::new();

    // Flatten and filter.
    let mut stack: SmallVec<[ExprId; 8]> = sets.iter().copied().collect();
    while let Some(s) = stack.pop() {
        match arena.node(s).clone() {
            ExprNode::SetUnion(children) => {
                // Flatten nested union.
                for &child in &children {
                    stack.push(child);
                }
            }
            ExprNode::EmptySet => {
                // Identity element: skip.
            }
            ExprNode::UniversalSet => {
                // Absorbing element: union with UniversalSet = UniversalSet.
                return arena.intern(ExprNode::UniversalSet);
            }
            _ => {
                flat.push(s);
            }
        }
    }

    if flat.is_empty() {
        return arena.intern(ExprNode::EmptySet);
    }

    // Deduplicate.
    let mut seen = FxHashSet::default();
    flat.retain(|id| seen.insert(*id));

    // Sort by sort key.
    flat.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if flat.len() == 1 {
        return flat[0];
    }

    order_real_intervals(arena, &mut flat);
    arena.intern(ExprNode::SetUnion(flat))
}

/// The lower end of an interval in a union, for [`order_real_intervals`]:
/// `−∞`, a finite real number (its `f64` value, and the exact value of a
/// rational end), or `+∞`.
enum LowerEnd {
    NegInf,
    Finite(f64, Option<Q>),
    PosInf,
}

impl LowerEnd {
    /// The end `start` when it is `±∞`, a number, or a constant with a
    /// finite real value (`√2`, `π/2`, `ln 3`, from 17 digits of `evalf`);
    /// `None` for one with symbols or without a real value.
    fn of(arena: &Arena, start: ExprId) -> Option<LowerEnd> {
        match arena.node(start) {
            ExprNode::NegInfinity => Some(LowerEnd::NegInf),
            ExprNode::Infinity => Some(LowerEnd::PosInf),
            ExprNode::Num(_) => {
                let q = arena.as_num(start)?.clone();
                let huge = if q.is_negative() { f64::MIN } else { f64::MAX };
                let f = crate::base::numeric::ratio_to_f64(&q).unwrap_or(huge);
                Some(LowerEnd::Finite(f, Some(q)))
            }
            _ if arena.is_symbol_free(start) => {
                let f = arena.const_f64(start, 17)?;
                f.is_finite().then_some(LowerEnd::Finite(f, None))
            }
            _ => None,
        }
    }

    /// Ascending order: by value, and of two ends with one `f64` value the
    /// rational one first (exactly between two rationals).
    fn cmp_ascending(&self, other: &LowerEnd) -> Ordering {
        match (self, other) {
            (LowerEnd::NegInf, LowerEnd::NegInf) | (LowerEnd::PosInf, LowerEnd::PosInf) => {
                Ordering::Equal
            }
            (LowerEnd::NegInf, _) | (_, LowerEnd::PosInf) => Ordering::Less,
            (_, LowerEnd::NegInf) | (LowerEnd::PosInf, _) => Ordering::Greater,
            (LowerEnd::Finite(a, qa), LowerEnd::Finite(b, qb)) => {
                // `+ 0.0` makes `−0.0` equal to `0.0` under `total_cmp`
                (a + 0.0).total_cmp(&(b + 0.0)).then(match (qa, qb) {
                    (Some(x), Some(y)) => x.cmp(y),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
                })
            }
        }
    }
}

/// Put the real intervals among the sort-key-ordered `children` of a union
/// in ascending order of their lower ends ([`LowerEnd`]), in the places the
/// sort key gave them (all intervals are adjacent in that order: their keys
/// share the set rank and the interval sub-rank); the other children
/// (finite sets, image sets, intervals with symbolic ends) keep their
/// places.  Ties (equal lower ends, or two constants with one `f64` value)
/// go by sort key, so the order is a total order that depends on the parts
/// only.  For disjoint intervals the order of the lower ends is the order
/// on the line, which the set evaluator's normal form uses (intervals
/// ascending, then the finite set of isolated points).  The `f64` values
/// only order the parts of a set, which is the same set in any order.
fn order_real_intervals(arena: &Arena, children: &mut [ExprId]) {
    let intervals = children
        .iter()
        .filter(|&&c| matches!(arena.node(c), ExprNode::Interval(..)))
        .count();
    if intervals < 2 {
        return;
    }
    let mut slots: SmallVec<[usize; 4]> = SmallVec::new();
    let mut keyed: Vec<(LowerEnd, ExprId)> = Vec::with_capacity(intervals);
    for (i, &c) in children.iter().enumerate() {
        if let ExprNode::Interval(start, _, _) = *arena.node(c)
            && let Some(end) = LowerEnd::of(arena, start)
        {
            slots.push(i);
            keyed.push((end, c));
        }
    }
    if keyed.len() < 2 {
        return;
    }
    keyed.sort_by(|(ea, a), (eb, b)| {
        ea.cmp_ascending(eb)
            .then_with(|| arena.sort_key(*a).cmp(arena.sort_key(*b)))
    });
    for (slot, (_, c)) in slots.into_iter().zip(keyed) {
        children[slot] = c;
    }
}

/// Build a canonical `SetIntersection` node.
///
/// Follows the LatticeOp pattern:
/// 1. Flatten nested SetIntersection.
/// 2. Remove UniversalSet children (identity for intersection).
/// 3. Short-circuit on EmptySet (absorbing element for intersection).
/// 4. Deduplicate and sort children by sort key.
/// 5. Single child → return that child directly.
pub(crate) fn canon_set_intersection(arena: &mut Arena, sets: &[ExprId]) -> ExprId {
    tracing::debug!("canon_set_intersection: {} children", sets.len());

    let mut flat: SmallVec<[ExprId; 4]> = SmallVec::new();

    // Flatten and filter.
    let mut stack: SmallVec<[ExprId; 8]> = sets.iter().copied().collect();
    while let Some(s) = stack.pop() {
        match arena.node(s).clone() {
            ExprNode::SetIntersection(children) => {
                // Flatten nested intersection.
                for &child in &children {
                    stack.push(child);
                }
            }
            ExprNode::UniversalSet => {
                // Identity element: skip.
            }
            ExprNode::EmptySet => {
                // Absorbing element: intersection with EmptySet = EmptySet.
                return arena.intern(ExprNode::EmptySet);
            }
            _ => {
                flat.push(s);
            }
        }
    }

    if flat.is_empty() {
        // Intersection of no sets: by convention, return UniversalSet
        // (the identity element).
        return arena.intern(ExprNode::UniversalSet);
    }

    // Deduplicate.
    let mut seen = FxHashSet::default();
    flat.retain(|id| seen.insert(*id));

    // Sort by sort key.
    flat.sort_by(|&a, &b| arena.sort_key(a).cmp(arena.sort_key(b)));

    if flat.len() == 1 {
        return flat[0];
    }

    arena.intern(ExprNode::SetIntersection(flat))
}
