//! Set algebra: interval arithmetic, membership, subset tests, inf/sup, measure.
//!
//! Sets in symplex are subsets of the real line built from the set nodes in
//! `ExprNode`: `Interval`, `FiniteSet`, `SetUnion`, `SetIntersection`,
//! `SetComplement` (relative complement `A \ B`), `EmptySet`,
//! `UniversalSet` (identified with ℝ for the purposes of this module) and
//! `ConditionSet`.
//!
//! Construction is structural — `[0, 2] ∩ [1, 3]` is stored as a
//! `SetIntersection` node.  This module provides the *evaluator*:
//! [`SetEx::simplify`](crate::expr::SetEx::simplify) rewrites a set
//! expression into a **normal form**
//!
//! > a union of pairwise-disjoint, ascending, non-adjacent intervals,
//! > followed by one finite set of isolated points,
//!
//! whenever every endpoint is a comparable real number (`Num`, `±∞`, or a
//! constant such as `π`, `e`, `√2` that can be ordered numerically).
//! Endpoints that cannot be ordered safely (symbolic endpoints, or two
//! constants that agree to within `1e-9` relative and cannot be separated
//! exactly) are left structural, and only *safe* identities are applied
//! (`A ∪ ∅ = A`, `A ∩ U = A`, `A ∪ A = A`, `A ∩ ∅ = ∅`, `A \ A = ∅`,
//! `(Aᶜ)ᶜ = A`, flattening, deduplication, deterministic ordering).
//!
//! All queries are three-valued (`Option<bool>`): `None` means "cannot be
//! decided", never "false".
//!
//! The public entry points are the methods on
//! [`SetEx`](crate::expr::SetEx); everything in this module takes
//! `&mut Arena` + `ExprId`.

use std::cmp::Ordering;

use num_traits::{Signed, ToPrimitive, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::extended::Extended;
use crate::base::interval::{Interval, IntervalKind};
use crate::base::node::{ExprId, ExprNode, INTERVAL_LEFT_OPEN, INTERVAL_RIGHT_OPEN};
use crate::base::numeric::Q;

/// Relative tolerance below which two floating-point endpoint
/// approximations are considered "too close to order safely".
const CLOSE_REL_TOL: f64 = 1e-9;

// ═══════════════════════════════════════════════════════════════════════════
// Numeric endpoint values and the rank table
// ═══════════════════════════════════════════════════════════════════════════

/// Numeric value of an endpoint: exact rational or floating approximation.
#[derive(Clone, Debug)]
enum Val {
    Exact(Q),
    Approx(f64),
}

impl Val {
    fn to_f64(&self) -> Option<f64> {
        match self {
            Val::Exact(r) => r.to_f64().filter(|f| f.is_finite()),
            Val::Approx(f) => Some(*f),
        }
    }
}

/// Position of an endpoint on the extended real line after ranking: a
/// finite endpoint's rank in the table, or `±∞`.
///
/// `Extended`'s order is `NegInf < Finite(_) < PosInf`, and ranks are
/// assigned in ascending numeric order, so `Pos` comparisons are
/// numeric comparisons.
type Pos = Extended<usize>;

/// Evaluate an expression to a finite real number, if possible.
///
/// Returns the evaluated id (e.g. `sqrt(4)` → `2`) together with its
/// value.  Returns `None` for symbolic expressions, infinities, `NaN`,
/// complex results, or anything that cannot be evaluated numerically.
fn numeric_value(arena: &mut Arena, id: ExprId) -> Option<(ExprId, Val)> {
    let ev = crate::transforms::eval::eval(arena, id);
    match arena.node(ev) {
        ExprNode::Num(n) => {
            let r = arena.num(*n).clone();
            Some((ev, Val::Exact(r)))
        }
        ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse => None,
        _ => {
            if !crate::base::walk::free_symbols(arena, ev).is_empty() {
                return None;
            }
            let f = crate::transforms::evalf::eval_const_f64(arena, ev)?;
            if !f.is_finite() {
                return None;
            }
            Some((ev, Val::Approx(f)))
        }
    }
}

/// Exact sign of `a - b`, when it can be established by exact evaluation.
fn exact_cmp(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<Ordering> {
    if a == b {
        return Some(Ordering::Equal);
    }
    let d = arena.sub(a, b);
    let d = crate::transforms::eval::eval(arena, d);
    let r = arena.as_num(d)?;
    Some(if r.is_zero() {
        Ordering::Equal
    } else if r.is_positive() {
        Ordering::Greater
    } else {
        Ordering::Less
    })
}

/// Compare two numeric values: exact when both are rational, otherwise
/// by `f64` approximation.
fn cmp_vals(a: &Val, b: &Val) -> Ordering {
    match (a, b) {
        (Val::Exact(x), Val::Exact(y)) => x.cmp(y),
        _ => {
            let fa = a.to_f64().unwrap_or(f64::NAN);
            let fb = b.to_f64().unwrap_or(f64::NAN);
            fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
        }
    }
}

/// Are two approximations too close to be ordered by floating point?
fn too_close(a: &Val, b: &Val) -> bool {
    if let (Val::Exact(_), Val::Exact(_)) = (a, b) {
        return false;
    }
    match (a.to_f64(), b.to_f64()) {
        (Some(fa), Some(fb)) => {
            let scale = fa.abs().max(fb.abs()).max(1.0);
            (fa - fb).abs() <= CLOSE_REL_TOL * scale
        }
        _ => true,
    }
}

/// The rank table: a consistent total order over every numeric endpoint
/// appearing in a family of set expressions.
struct Table {
    /// raw endpoint id → evaluated id
    evaluated: FxHashMap<ExprId, ExprId>,
    /// evaluated id → position (finite reals get a `Rank`)
    pos: FxHashMap<ExprId, Pos>,
    /// rank → representative evaluated id
    reps: Vec<ExprId>,
}

impl Table {
    /// Build the table from a list of candidate endpoint ids.
    fn build(arena: &mut Arena, ids: &[ExprId]) -> Table {
        let mut evaluated: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        let mut pos: FxHashMap<ExprId, Pos> = FxHashMap::default();
        // evaluated id → value, for finite candidates
        let mut cand: Vec<(ExprId, Val)> = Vec::new();
        let mut seen_ev: FxHashSet<ExprId> = FxHashSet::default();

        for &raw in ids {
            if evaluated.contains_key(&raw) {
                continue;
            }
            let ev = crate::transforms::eval::eval(arena, raw);
            evaluated.insert(raw, ev);
            match arena.node(ev) {
                ExprNode::Infinity => {
                    pos.insert(ev, Pos::PosInf);
                }
                ExprNode::NegInfinity => {
                    pos.insert(ev, Pos::NegInf);
                }
                _ => {
                    if seen_ev.insert(ev)
                        && let Some((ev2, val)) = numeric_value(arena, ev)
                    {
                        // `eval` is idempotent, so ev2 == ev; keep ev2 anyway.
                        if ev2 != ev {
                            evaluated.insert(raw, ev2);
                        }
                        cand.push((ev2, val));
                    }
                }
            }
        }

        cand.sort_by(|x, y| cmp_vals(&x.1, &y.1));

        // Resolve pairs that are too close for floating-point ordering:
        // either separate/merge them exactly, or drop both from the
        // numeric world (their sets stay structural).
        let mut alias: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        loop {
            let mut bad: FxHashSet<ExprId> = FxHashSet::default();
            let mut i = 0;
            while i + 1 < cand.len() {
                let (a, va) = (cand[i].0, &cand[i].1);
                let (b, vb) = (cand[i + 1].0, &cand[i + 1].1);
                if too_close(va, vb) {
                    match exact_cmp(arena, a, b) {
                        Some(Ordering::Equal) => {
                            alias.insert(b, a);
                        }
                        Some(Ordering::Less) => {}
                        _ => {
                            bad.insert(a);
                            bad.insert(b);
                        }
                    }
                }
                i += 1;
            }
            // Remove aliases (they take the rank of their representative).
            cand.retain(|(id, _)| !alias.contains_key(id));
            if bad.is_empty() {
                break;
            }
            cand.retain(|(id, _)| !bad.contains(id));
        }

        let mut reps = Vec::with_capacity(cand.len());
        for (rank, (id, _)) in cand.iter().enumerate() {
            pos.insert(*id, Pos::Finite(rank));
            reps.push(*id);
        }
        // Aliases may chain (c → b → a); resolve transitively.
        for (&from, &to) in &alias {
            let mut root = to;
            let mut guard = 0;
            while let Some(&next) = alias.get(&root) {
                root = next;
                guard += 1;
                if guard > alias.len() {
                    break;
                }
            }
            if let Some(&p) = pos.get(&root) {
                pos.insert(from, p);
            }
        }

        Table {
            evaluated,
            pos,
            reps,
        }
    }

    /// Position of a raw endpoint id, if it is a comparable extended real.
    fn pos_of(&self, raw: ExprId) -> Option<Pos> {
        let ev = self.evaluated.get(&raw).copied().unwrap_or(raw);
        self.pos.get(&ev).copied()
    }

    /// Evaluated form of a raw endpoint id.
    fn ev(&self, raw: ExprId) -> ExprId {
        self.evaluated.get(&raw).copied().unwrap_or(raw)
    }

    /// Representative expression for a position.
    fn rep(&self, arena: &Arena, p: Pos) -> ExprId {
        match p {
            Pos::NegInf => arena.neg_infinity,
            Pos::PosInf => arena.infinity,
            Pos::Finite(r) => self.reps[r],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RankSet — the normal form over ranks
// ═══════════════════════════════════════════════════════════════════════════

/// One connected piece: an interval between two positions, or an isolated
/// point (`lower == upper`, closed).  In a normalised [`RankSet`] a piece
/// with `lower == upper` is always closed at both ends.
type Piece = Interval<Pos>;

/// A piece with the given ends and openness.
fn piece(lower: Pos, upper: Pos, lower_open: bool, upper_open: bool) -> Piece {
    Interval {
        lower,
        upper,
        kind: IntervalKind::from_open_ends(lower_open, upper_open),
    }
}

/// The isolated point `{p}`.
fn point(p: Pos) -> Piece {
    Interval::point(p)
}

/// Force infinite endpoints open and report whether the piece is
/// non-empty.
fn fix(p: &mut Piece) -> bool {
    let lower_open = p.kind.lower_open() || p.lower == Pos::NegInf;
    let upper_open = p.kind.upper_open() || p.upper == Pos::PosInf;
    p.kind = IntervalKind::from_open_ends(lower_open, upper_open);
    match p.lower.cmp(&p.upper) {
        Ordering::Less => true,
        Ordering::Equal => p.lower.is_finite() && p.kind == IntervalKind::Closed,
        Ordering::Greater => false,
    }
}

/// A subset of ℝ in normal form: sorted, pairwise-disjoint, non-mergeable
/// pieces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RankSet {
    pieces: Vec<Piece>,
}

impl RankSet {
    fn empty() -> RankSet {
        RankSet { pieces: Vec::new() }
    }

    fn full() -> RankSet {
        RankSet {
            pieces: vec![Interval::open(Pos::NegInf, Pos::PosInf)],
        }
    }

    /// Build the normal form from arbitrary (possibly overlapping,
    /// unsorted, invalid) pieces.
    fn normalize(mut pieces: Vec<Piece>) -> RankSet {
        pieces.retain_mut(fix);
        pieces.sort_by(|x, y| {
            x.lower
                .cmp(&y.lower)
                .then(x.kind.lower_open().cmp(&y.kind.lower_open()))
        });
        let mut out: Vec<Piece> = Vec::with_capacity(pieces.len());
        for p in pieces {
            if let Some(cur) = out.last_mut() {
                let merge = p.lower < cur.upper
                    || (p.lower == cur.upper && !(cur.kind.upper_open() && p.kind.lower_open()));
                if merge {
                    match p.upper.cmp(&cur.upper) {
                        Ordering::Greater => {
                            cur.upper = p.upper;
                            cur.kind = cur.kind.with_upper_open(p.kind.upper_open());
                        }
                        Ordering::Equal => {
                            cur.kind = cur
                                .kind
                                .with_upper_open(cur.kind.upper_open() && p.kind.upper_open());
                        }
                        Ordering::Less => {}
                    }
                    continue;
                }
            }
            out.push(p);
        }
        RankSet { pieces: out }
    }

    fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    fn is_full(&self) -> bool {
        self.pieces.len() == 1
            && self.pieces[0].lower == Pos::NegInf
            && self.pieces[0].upper == Pos::PosInf
    }

    fn union(&self, other: &RankSet) -> RankSet {
        let mut all = self.pieces.clone();
        all.extend_from_slice(&other.pieces);
        RankSet::normalize(all)
    }

    fn complement(&self) -> RankSet {
        let mut out = Vec::with_capacity(self.pieces.len() + 1);
        let mut prev_hi = Pos::NegInf;
        let mut prev_hi_open = true;
        for p in &self.pieces {
            out.push(piece(prev_hi, p.lower, !prev_hi_open, !p.kind.lower_open()));
            prev_hi = p.upper;
            prev_hi_open = p.kind.upper_open();
        }
        out.push(piece(prev_hi, Pos::PosInf, !prev_hi_open, true));
        RankSet::normalize(out)
    }

    fn intersection(&self, other: &RankSet) -> RankSet {
        self.complement().union(&other.complement()).complement()
    }

    fn difference(&self, other: &RankSet) -> RankSet {
        self.intersection(&other.complement())
    }

    fn contains(&self, p: Pos) -> bool {
        if !p.is_finite() {
            return false;
        }
        self.pieces.iter().any(|pc| pc.contains(&p))
    }

    fn is_subset(&self, other: &RankSet) -> bool {
        self.difference(other).is_empty()
    }

    fn closure(&self) -> RankSet {
        let pieces = self
            .pieces
            .iter()
            .map(|p| p.with_kind(IntervalKind::Closed))
            .collect();
        RankSet::normalize(pieces)
    }

    fn interior(&self) -> RankSet {
        let pieces = self
            .pieces
            .iter()
            .filter(|p| !p.is_point())
            .map(|p| p.with_kind(IntervalKind::Open))
            .collect();
        RankSet::normalize(pieces)
    }

    fn boundary(&self) -> RankSet {
        let mut pts: Vec<Piece> = Vec::new();
        for p in &self.pieces {
            if p.lower.is_finite() {
                pts.push(point(p.lower));
            }
            if p.upper.is_finite() {
                pts.push(point(p.upper));
            }
        }
        RankSet::normalize(pts)
    }

    fn is_open(&self) -> bool {
        self.pieces
            .iter()
            .all(|p| !p.is_point() && p.kind == IntervalKind::Open)
    }

    fn is_closed(&self) -> bool {
        self.pieces.iter().all(|p| {
            (p.kind.lower_open() == (p.lower == Pos::NegInf))
                && (p.kind.upper_open() == (p.upper == Pos::PosInf))
        })
    }

    fn inf(&self) -> Option<Pos> {
        self.pieces.first().map(|p| p.lower)
    }

    fn sup(&self) -> Option<Pos> {
        self.pieces.last().map(|p| p.upper)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SetVal — partially evaluated sets
// ═══════════════════════════════════════════════════════════════════════════

/// The value of a set node after evaluation.
#[derive(Clone, Debug)]
enum SetVal {
    /// Fully evaluated normal form.
    Exact(RankSet),
    /// `exact ∪ syms` where every element of `syms` is an opaque,
    /// already-simplified set expression (non-empty list).
    Mixed(RankSet, Vec<ExprId>),
    /// Opaque, already-simplified set expression.
    Sym(ExprId),
}

impl SetVal {
    fn exact(&self) -> Option<&RankSet> {
        match self {
            SetVal::Exact(rs) => Some(rs),
            _ => None,
        }
    }

    fn is_known_empty(&self) -> bool {
        matches!(self, SetVal::Exact(rs) if rs.is_empty())
    }

    fn is_known_full(&self) -> bool {
        matches!(self, SetVal::Exact(rs) if rs.is_full())
    }
}

/// Convert a normal form back into a set expression.
///
/// Intervals are emitted in ascending order followed by one `FiniteSet`
/// of the isolated points.  The whole line is emitted as `(-∞, ∞)`.
fn rankset_to_expr(arena: &mut Arena, table: &Table, rs: &RankSet) -> ExprId {
    if rs.is_empty() {
        return arena.empty_set;
    }
    let mut parts: SmallVec<[ExprId; 4]> = SmallVec::new();
    let mut points: Vec<ExprId> = Vec::new();
    for p in &rs.pieces {
        if p.is_point() {
            points.push(table.rep(arena, p.lower));
        } else {
            let lo = table.rep(arena, p.lower);
            let hi = table.rep(arena, p.upper);
            let mut flags = 0u8;
            if p.kind.lower_open() {
                flags |= INTERVAL_LEFT_OPEN;
            }
            if p.kind.upper_open() {
                flags |= INTERVAL_RIGHT_OPEN;
            }
            parts.push(arena.intern(ExprNode::Interval(lo, hi, flags)));
        }
    }
    if !points.is_empty() {
        parts.push(arena.finite_set(&points));
    }
    if parts.len() == 1 {
        parts[0]
    } else {
        arena.intern(ExprNode::SetUnion(parts))
    }
}

fn setval_to_expr(arena: &mut Arena, table: &Table, v: &SetVal) -> ExprId {
    match v {
        SetVal::Exact(rs) => rankset_to_expr(arena, table, rs),
        SetVal::Mixed(rs, syms) => {
            let mut parts: Vec<ExprId> = Vec::with_capacity(syms.len() + 1);
            if !rs.is_empty() {
                parts.push(rankset_to_expr(arena, table, rs));
            }
            parts.extend_from_slice(syms);
            arena.set_union(&parts)
        }
        SetVal::Sym(id) => *id,
    }
}

/// Collect the set-structured nodes reachable from `roots` in post-order
/// (children before parents), together with every numeric endpoint /
/// element id encountered.
fn collect(arena: &Arena, roots: &[ExprId]) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut order = Vec::new();
    let mut endpoints = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<(ExprId, bool)> = roots.iter().rev().map(|&r| (r, false)).collect();

    while let Some((id, expanded)) = stack.pop() {
        if visited.contains(&id) {
            continue;
        }
        if expanded {
            visited.insert(id);
            order.push(id);
            continue;
        }
        stack.push((id, true));
        match arena.node(id) {
            ExprNode::SetUnion(children) | ExprNode::SetIntersection(children) => {
                for &c in children.iter().rev() {
                    if !visited.contains(&c) {
                        stack.push((c, false));
                    }
                }
            }
            ExprNode::SetComplement(a, b) => {
                for &c in [*b, *a].iter() {
                    if !visited.contains(&c) {
                        stack.push((c, false));
                    }
                }
            }
            ExprNode::Interval(a, b, _) => {
                endpoints.push(*a);
                endpoints.push(*b);
            }
            ExprNode::FiniteSet(elems) => {
                endpoints.extend(elems.iter().copied());
            }
            _ => {}
        }
    }
    (order, endpoints)
}

/// Evaluate a family of set expressions against one shared rank table.
///
/// `extra_points` are additional numeric expressions (e.g. a membership
/// candidate) that must be ranked together with the endpoints.
struct Evaluated {
    table: Table,
    vals: FxHashMap<ExprId, SetVal>,
}

fn evaluate(arena: &mut Arena, roots: &[ExprId], extra_points: &[ExprId]) -> Evaluated {
    let (order, mut endpoints) = collect(arena, roots);
    endpoints.extend_from_slice(extra_points);
    let table = Table::build(arena, &endpoints);
    let mut vals: FxHashMap<ExprId, SetVal> = FxHashMap::default();

    for &id in &order {
        let v = eval_node(arena, &table, &vals, id);
        vals.insert(id, v);
    }
    Evaluated { table, vals }
}

fn eval_node(
    arena: &mut Arena,
    table: &Table,
    vals: &FxHashMap<ExprId, SetVal>,
    id: ExprId,
) -> SetVal {
    match arena.node(id).clone() {
        ExprNode::EmptySet => SetVal::Exact(RankSet::empty()),
        ExprNode::UniversalSet => SetVal::Exact(RankSet::full()),

        ExprNode::Interval(a, b, flags) => match (table.pos_of(a), table.pos_of(b)) {
            (Some(lo), Some(hi)) => {
                let piece = piece(
                    lo,
                    hi,
                    flags & INTERVAL_LEFT_OPEN != 0,
                    flags & INTERVAL_RIGHT_OPEN != 0,
                );
                SetVal::Exact(RankSet::normalize(vec![piece]))
            }
            _ => {
                // Symbolic endpoints: keep structural, but use evaluated
                // endpoint forms for determinism.
                let ea = table.ev(a);
                let eb = table.ev(b);
                if ea == a && eb == b {
                    SetVal::Sym(id)
                } else {
                    SetVal::Sym(arena.interval(ea, eb, flags))
                }
            }
        },

        ExprNode::FiniteSet(elems) => {
            let mut pieces = Vec::new();
            let mut syms = Vec::new();
            for &e in &elems {
                match table.pos_of(e) {
                    Some(p @ Pos::Finite(_)) => pieces.push(point(p)),
                    _ => syms.push(table.ev(e)),
                }
            }
            let rs = RankSet::normalize(pieces);
            if syms.is_empty() {
                SetVal::Exact(rs)
            } else {
                let sym_set = arena.finite_set(&syms);
                if rs.is_empty() {
                    SetVal::Sym(sym_set)
                } else {
                    SetVal::Mixed(rs, vec![sym_set])
                }
            }
        }

        ExprNode::SetUnion(children) => {
            let mut exact = RankSet::empty();
            let mut syms: Vec<ExprId> = Vec::new();
            for c in children {
                match vals.get(&c) {
                    Some(SetVal::Exact(rs)) => exact = exact.union(rs),
                    Some(SetVal::Mixed(rs, s)) => {
                        exact = exact.union(rs);
                        syms.extend_from_slice(s);
                    }
                    Some(SetVal::Sym(s)) => syms.push(*s),
                    None => syms.push(c),
                }
            }
            finish_union(exact, syms)
        }

        ExprNode::SetIntersection(children) => {
            let mut exact: Option<RankSet> = None;
            let mut opaque: Vec<ExprId> = Vec::new();
            for c in children {
                match vals.get(&c) {
                    Some(SetVal::Exact(rs)) => {
                        exact = Some(match exact {
                            Some(e) => e.intersection(rs),
                            None => rs.clone(),
                        });
                    }
                    Some(v) => opaque.push(setval_to_expr(arena, table, v)),
                    None => opaque.push(c),
                }
            }
            match exact {
                Some(e) if e.is_empty() => SetVal::Exact(RankSet::empty()),
                Some(e) if opaque.is_empty() => SetVal::Exact(e),
                Some(e) => {
                    let mut parts = Vec::with_capacity(opaque.len() + 1);
                    if !e.is_full() {
                        parts.push(rankset_to_expr(arena, table, &e));
                    }
                    parts.extend(opaque);
                    SetVal::Sym(arena.set_intersection(&parts))
                }
                None => SetVal::Sym(arena.set_intersection(&opaque)),
            }
        }

        ExprNode::SetComplement(a, b) => {
            let va = vals.get(&a).cloned().unwrap_or(SetVal::Sym(a));
            let vb = vals.get(&b).cloned().unwrap_or(SetVal::Sym(b));
            eval_difference(arena, table, vals, a, b, &va, &vb)
        }

        ExprNode::ConditionSet(var, cond) => {
            let c = crate::transforms::logic::simplify_bool(arena, cond);
            if c == arena.bool_true {
                SetVal::Exact(RankSet::full())
            } else if c == arena.bool_false {
                SetVal::Exact(RankSet::empty())
            } else if c == cond {
                SetVal::Sym(id)
            } else {
                SetVal::Sym(arena.intern(ExprNode::ConditionSet(var, c)))
            }
        }

        _ => SetVal::Sym(id),
    }
}

fn finish_union(exact: RankSet, mut syms: Vec<ExprId>) -> SetVal {
    if exact.is_full() {
        return SetVal::Exact(exact);
    }
    let mut seen = FxHashSet::default();
    syms.retain(|s| seen.insert(*s));
    if syms.is_empty() {
        SetVal::Exact(exact)
    } else {
        SetVal::Mixed(exact, syms)
    }
}

/// Evaluate `A \ B`.
#[allow(clippy::too_many_arguments)]
fn eval_difference(
    arena: &mut Arena,
    table: &Table,
    vals: &FxHashMap<ExprId, SetVal>,
    a: ExprId,
    b: ExprId,
    va: &SetVal,
    vb: &SetVal,
) -> SetVal {
    // A \ A = ∅
    if a == b {
        return SetVal::Exact(RankSet::empty());
    }
    // ∅ \ B = ∅ ;  A \ ∅ = A ;  A \ U = ∅
    if va.is_known_empty() || vb.is_known_full() {
        return SetVal::Exact(RankSet::empty());
    }
    if vb.is_known_empty() {
        return va.clone();
    }
    // (Aᶜ)ᶜ = A :  U \ (U \ X) = X
    if va.is_known_full()
        && let ExprNode::SetComplement(u, x) = arena.node(b).clone()
        && vals.get(&u).is_some_and(|v| v.is_known_full())
    {
        return vals.get(&x).cloned().unwrap_or(SetVal::Sym(x));
    }

    match (va, vb) {
        (SetVal::Exact(ea), SetVal::Exact(eb)) => SetVal::Exact(ea.difference(eb)),
        (SetVal::Exact(ea), SetVal::Mixed(eb, sb)) => {
            // A \ (E ∪ S) = (A \ E) \ S
            let d = ea.difference(eb);
            if d.is_empty() {
                return SetVal::Exact(d);
            }
            let d_expr = rankset_to_expr(arena, table, &d);
            let s_expr = arena.set_union(sb);
            SetVal::Sym(arena.set_complement(d_expr, s_expr))
        }
        (SetVal::Mixed(ea, sa), SetVal::Exact(eb)) => {
            // (E ∪ S) \ B = (E \ B) ∪ (S \ B)
            let d = ea.difference(eb);
            let s_expr = arena.set_union(sa);
            let b_expr = rankset_to_expr(arena, table, eb);
            let rest = arena.set_complement(s_expr, b_expr);
            finish_union(d, vec![rest])
        }
        _ => {
            let a_expr = setval_to_expr(arena, table, va);
            let b_expr = setval_to_expr(arena, table, vb);
            if a_expr == b_expr {
                SetVal::Exact(RankSet::empty())
            } else {
                SetVal::Sym(arena.set_complement(a_expr, b_expr))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public backend API
// ═══════════════════════════════════════════════════════════════════════════

/// Rewrite a set expression into normal form (see the module docs).
///
/// Returns the input unchanged when it is already in normal form or when
/// nothing can be simplified.
pub(crate) fn simplify_set(arena: &mut Arena, set: ExprId) -> ExprId {
    if set == arena.universal_set || set == arena.empty_set {
        return set;
    }
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set) {
        Some(v) => setval_to_expr(arena, &ev.table, v),
        None => set,
    }
}

/// `a \ b` in normal form (structural when undecidable).
pub(crate) fn set_difference(arena: &mut Arena, a: ExprId, b: ExprId) -> ExprId {
    let d = arena.set_complement(a, b);
    simplify_set(arena, d)
}

/// `(a \ b) ∪ (b \ a)` in normal form.
pub(crate) fn symmetric_difference(arena: &mut Arena, a: ExprId, b: ExprId) -> ExprId {
    let ab = arena.set_complement(a, b);
    let ba = arena.set_complement(b, a);
    let u = arena.set_union(&[ab, ba]);
    simplify_set(arena, u)
}

/// Absolute complement `ℝ \ a` in normal form.
pub(crate) fn absolute_complement(arena: &mut Arena, a: ExprId) -> ExprId {
    let u = arena.universal_set;
    let c = arena.set_complement(u, a);
    simplify_set(arena, c)
}

/// Compare two real-valued expressions, if the order can be established.
///
/// Handles `±∞`, exact rationals, exact differences (`x + 1` vs `x`),
/// and well-separated numeric constants.  Returns `None` when the order
/// cannot be decided safely.
pub(crate) fn cmp_exprs(arena: &mut Arena, p: ExprId, q: ExprId) -> Option<Ordering> {
    if p == q {
        return Some(Ordering::Equal);
    }
    let inf = arena.infinity;
    let ninf = arena.neg_infinity;
    match (p == ninf, p == inf, q == ninf, q == inf) {
        (true, _, true, _) | (_, true, _, true) => return Some(Ordering::Equal),
        (true, _, _, _) | (_, _, _, true) => return Some(Ordering::Less),
        (_, true, _, _) | (_, _, true, _) => return Some(Ordering::Greater),
        _ => {}
    }
    if let Some(o) = exact_cmp(arena, p, q) {
        return Some(o);
    }
    let (_, vp) = numeric_value(arena, p)?;
    let (_, vq) = numeric_value(arena, q)?;
    if too_close(&vp, &vq) {
        return None;
    }
    Some(cmp_vals(&vp, &vq))
}

/// Is `elem` an element of the set `set`?
///
/// The literal `UniversalSet` contains everything; the real line
/// `(-∞, ∞)` contains an element only if it is known to be real.
pub(crate) fn set_contains(arena: &mut Arena, set: ExprId, elem: ExprId) -> Option<bool> {
    if set == arena.universal_set {
        return Some(true);
    }
    let ev = evaluate(arena, &[set], &[elem]);
    let elem_pos = ev.table.pos_of(elem);
    let elem_ev = ev.table.ev(elem);
    let (order, _) = collect(arena, &[set]);
    let mut memo: FxHashMap<ExprId, Option<bool>> = FxHashMap::default();

    for &id in &order {
        // Exact fast path.
        if let Some(rs) = ev.vals.get(&id).and_then(SetVal::exact) {
            let r = match elem_pos {
                Some(p @ Pos::Finite(_)) => Some(rs.contains(p)),
                Some(_) => Some(false), // ±∞ is never a member of a real set
                None => {
                    if rs.is_empty() {
                        Some(false)
                    } else if rs.is_full() && is_known_real(arena, elem_ev) {
                        Some(true)
                    } else {
                        None
                    }
                }
            };
            memo.insert(id, r);
            continue;
        }
        let r = match arena.node(id).clone() {
            ExprNode::EmptySet => Some(false),
            ExprNode::UniversalSet => Some(true),
            ExprNode::Interval(a, b, flags) => {
                let lo = cmp_exprs(arena, a, elem_ev);
                let hi = cmp_exprs(arena, elem_ev, b);
                let l_ok = lo.map(|o| match o {
                    Ordering::Less => true,
                    Ordering::Equal => flags & INTERVAL_LEFT_OPEN == 0,
                    Ordering::Greater => false,
                });
                let h_ok = hi.map(|o| match o {
                    Ordering::Less => true,
                    Ordering::Equal => flags & INTERVAL_RIGHT_OPEN == 0,
                    Ordering::Greater => false,
                });
                and3(l_ok, h_ok)
            }
            ExprNode::FiniteSet(elems) => {
                let mut any_unknown = false;
                let mut found = false;
                for &e in &elems {
                    match cmp_exprs(arena, e, elem_ev) {
                        Some(Ordering::Equal) => {
                            found = true;
                            break;
                        }
                        Some(_) => {}
                        None => any_unknown = true,
                    }
                }
                if found {
                    Some(true)
                } else if any_unknown {
                    None
                } else {
                    Some(false)
                }
            }
            ExprNode::SetUnion(children) => {
                let mut acc = Some(false);
                for c in children {
                    acc = or3(acc, memo.get(&c).copied().flatten());
                }
                acc
            }
            ExprNode::SetIntersection(children) => {
                let mut acc = Some(true);
                for c in children {
                    acc = and3(acc, memo.get(&c).copied().flatten());
                }
                acc
            }
            ExprNode::SetComplement(a, b) => {
                let ia = memo.get(&a).copied().flatten();
                let ib = memo.get(&b).copied().flatten();
                and3(ia, ib.map(|x| !x))
            }
            ExprNode::ConditionSet(var, cond) => {
                let c = crate::transforms::subs::subs(arena, cond, var, elem_ev);
                let c = crate::transforms::logic::simplify_bool(arena, c);
                if c == arena.bool_true {
                    Some(true)
                } else if c == arena.bool_false {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        };
        memo.insert(id, r);
    }
    memo.get(&set).copied().flatten()
}

fn is_known_real(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => true,
        ExprNode::Symbol(sid) => {
            arena
                .symbol_assumptions(*sid)
                .query(crate::base::assumptions::Props::REAL)
                == Some(true)
        }
        _ => false,
    }
}

fn and3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn or3(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// Structural emptiness for sets that could not be evaluated exactly.
///
/// Bottom-up over the set structure (explicit post-order, no recursion).
fn structural_is_empty(
    arena: &Arena,
    vals: &FxHashMap<ExprId, SetVal>,
    root: ExprId,
) -> Option<bool> {
    let (order, _) = collect(arena, &[root]);
    let mut memo: FxHashMap<ExprId, Option<bool>> = FxHashMap::default();
    for &id in &order {
        let known = match vals.get(&id) {
            Some(SetVal::Exact(rs)) => Some(Some(rs.is_empty())),
            Some(SetVal::Mixed(rs, _)) if !rs.is_empty() => Some(Some(false)),
            _ => None,
        };
        let r = if let Some(k) = known {
            k
        } else {
            match arena.node(id) {
                ExprNode::EmptySet => Some(true),
                ExprNode::UniversalSet => Some(false),
                ExprNode::FiniteSet(elems) => Some(elems.is_empty()),
                ExprNode::SetUnion(children) => {
                    let mut acc = Some(true);
                    for c in children {
                        let e = memo.get(c).copied().flatten();
                        acc = and3(acc, e);
                    }
                    acc
                }
                ExprNode::SetIntersection(children) => {
                    if children
                        .iter()
                        .any(|c| memo.get(c).copied().flatten() == Some(true))
                    {
                        Some(true)
                    } else {
                        None
                    }
                }
                ExprNode::SetComplement(a, b) => {
                    let ea = memo.get(a).copied().flatten();
                    let eb = memo.get(b).copied().flatten();
                    match (ea, eb) {
                        (Some(true), _) => Some(true),
                        (Some(false), Some(true)) => Some(false),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        memo.insert(id, r);
    }
    memo.get(&root).copied().flatten()
}

/// Is the set empty?
pub(crate) fn is_empty(arena: &mut Arena, set: ExprId) -> Option<bool> {
    let ev = evaluate(arena, &[set], &[]);
    structural_is_empty(arena, &ev.vals, set)
}

/// Is the set the whole real line?
pub(crate) fn is_full(arena: &mut Arena, set: ExprId) -> Option<bool> {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set)? {
        SetVal::Exact(rs) => Some(rs.is_full()),
        SetVal::Mixed(rs, _) if rs.is_full() => Some(true),
        _ => None,
    }
}

/// Is `a ⊆ b`?
pub(crate) fn is_subset(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<bool> {
    if a == b {
        return Some(true);
    }
    let ev = evaluate(arena, &[a, b], &[]);
    let va = ev.vals.get(&a)?;
    let vb = ev.vals.get(&b)?;
    if va.is_known_empty() || vb.is_known_full() {
        return Some(true);
    }
    if let (Some(ea), Some(eb)) = (va.exact(), vb.exact()) {
        return Some(ea.is_subset(eb));
    }
    // {…} ⊆ {…} structurally (element identity).
    if let (ExprNode::FiniteSet(xa), ExprNode::FiniteSet(xb)) = (arena.node(a), arena.node(b))
        && xa.iter().all(|e| xb.contains(e))
    {
        return Some(true);
    }
    // Exact non-empty A ⊆ structural B: decide only if B is the empty set.
    if let Some(ea) = va.exact()
        && !ea.is_empty()
        && vb.is_known_empty()
    {
        return Some(false);
    }
    None
}

/// Is `a ∩ b = ∅`?
pub(crate) fn is_disjoint(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<bool> {
    let ev = evaluate(arena, &[a, b], &[]);
    let va = ev.vals.get(&a)?;
    let vb = ev.vals.get(&b)?;
    if va.is_known_empty() || vb.is_known_empty() {
        return Some(true);
    }
    if let (Some(ea), Some(eb)) = (va.exact(), vb.exact()) {
        return Some(ea.intersection(eb).is_empty());
    }
    if a == b {
        // A ∩ A = A: disjoint from itself only if empty.
        return structural_is_empty(arena, &ev.vals, a);
    }
    None
}

/// Greatest lower bound (`-∞` for sets unbounded below); `None` if the
/// set is empty or cannot be evaluated.
pub(crate) fn inf(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    let p = rs.inf()?;
    Some(ev.table.rep(arena, p))
}

/// Least upper bound (`∞` for sets unbounded above); `None` if the set
/// is empty or cannot be evaluated.
pub(crate) fn sup(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    let p = rs.sup()?;
    Some(ev.table.rep(arena, p))
}

/// Lebesgue measure (total length) of the set; `∞` when unbounded.
pub(crate) fn measure(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?.clone();
    let mut terms: Vec<ExprId> = Vec::new();
    for p in &rs.pieces {
        if p.is_point() {
            continue;
        }
        if !p.is_bounded() {
            return Some(arena.infinity);
        }
        let lo = ev.table.rep(arena, p.lower);
        let hi = ev.table.rep(arena, p.upper);
        terms.push(arena.sub(hi, lo));
    }
    let total = if terms.is_empty() {
        arena.zero
    } else {
        arena.add(&terms)
    };
    Some(crate::transforms::eval::eval(arena, total))
}

/// Apply a `RankSet → RankSet` topological operation and re-emit.
fn topo_op(arena: &mut Arena, set: ExprId, f: impl Fn(&RankSet) -> RankSet) -> Option<ExprId> {
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    let out = f(rs);
    Some(rankset_to_expr(arena, &ev.table, &out))
}

/// Topological closure.
pub(crate) fn closure(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    topo_op(arena, set, RankSet::closure)
}

/// Topological interior.
pub(crate) fn interior(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    topo_op(arena, set, RankSet::interior)
}

/// Topological boundary.
pub(crate) fn boundary(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    topo_op(arena, set, RankSet::boundary)
}

/// Is the set open (in ℝ)?
pub(crate) fn is_open(arena: &mut Arena, set: ExprId) -> Option<bool> {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set)? {
        SetVal::Exact(rs) => Some(rs.is_open()),
        _ => match arena.node(set) {
            // A non-empty finite set is never open.
            ExprNode::FiniteSet(e) if !e.is_empty() => Some(false),
            _ => None,
        },
    }
}

/// Is the set closed (in ℝ)?
pub(crate) fn is_closed(arena: &mut Arena, set: ExprId) -> Option<bool> {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set)? {
        SetVal::Exact(rs) => Some(rs.is_closed()),
        _ => match arena.node(set) {
            ExprNode::FiniteSet(_) => Some(true),
            _ => None,
        },
    }
}

/// The normal form as intervals (isolated points appear as `[p, p]`), or
/// `None` if the set cannot be fully evaluated.
pub(crate) fn as_intervals(arena: &mut Arena, set: ExprId) -> Option<Vec<Interval<ExprId>>> {
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    Some(
        rs.pieces
            .iter()
            .map(|p| p.as_ref().map(|pos| ev.table.rep(arena, *pos)))
            .collect(),
    )
}

/// The elements of a finite set (possibly symbolic), or `None` if the set
/// is not known to be finite.
pub(crate) fn as_finite_set(arena: &mut Arena, set: ExprId) -> Option<Vec<ExprId>> {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set)? {
        SetVal::Exact(rs) => {
            if rs.pieces.iter().all(Piece::is_point) {
                Some(
                    rs.pieces
                        .iter()
                        .map(|p| ev.table.rep(arena, p.lower))
                        .collect(),
                )
            } else {
                None
            }
        }
        SetVal::Mixed(rs, syms) => {
            if !rs.pieces.iter().all(Piece::is_point) {
                return None;
            }
            let mut out: Vec<ExprId> = rs
                .pieces
                .iter()
                .map(|p| ev.table.rep(arena, p.lower))
                .collect();
            for &s in syms {
                match arena.node(s) {
                    ExprNode::FiniteSet(elems) => out.extend(elems.iter().copied()),
                    _ => return None,
                }
            }
            Some(out)
        }
        SetVal::Sym(s) => match arena.node(*s) {
            ExprNode::FiniteSet(elems) => Some(elems.iter().copied().collect()),
            _ => None,
        },
    }
}

/// `And` with constant folding (`true` dropped, `false` absorbing).
fn mk_and(arena: &mut Arena, parts: &[ExprId]) -> ExprId {
    let t = arena.bool_true;
    let f = arena.bool_false;
    if parts.contains(&f) {
        return f;
    }
    let kept: Vec<ExprId> = parts.iter().copied().filter(|&p| p != t).collect();
    arena.and(&kept)
}

/// `Or` with constant folding (`false` dropped, `true` absorbing).
fn mk_or(arena: &mut Arena, parts: &[ExprId]) -> ExprId {
    let t = arena.bool_true;
    let f = arena.bool_false;
    if parts.contains(&t) {
        return t;
    }
    let kept: Vec<ExprId> = parts.iter().copied().filter(|&p| p != f).collect();
    arena.or(&kept)
}

/// Membership of `var` in `set` as a boolean expression.
///
/// Returns `Err(InvalidArgument)` if `set` contains a node that is not a
/// set constructor.
pub(crate) fn to_condition(
    arena: &mut Arena,
    set: ExprId,
    var: ExprId,
) -> Result<ExprId, SymplexError> {
    let (order, _) = collect(arena, &[set]);
    let mut memo: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &order {
        let c = match arena.node(id).clone() {
            ExprNode::EmptySet => arena.bool_false,
            ExprNode::UniversalSet => arena.bool_true,
            ExprNode::Interval(a, b, flags) => {
                let mut conj: SmallVec<[ExprId; 2]> = SmallVec::new();
                if a != arena.neg_infinity {
                    conj.push(if flags & INTERVAL_LEFT_OPEN != 0 {
                        arena.gt(var, a)
                    } else {
                        arena.ge(var, a)
                    });
                }
                if b != arena.infinity {
                    conj.push(if flags & INTERVAL_RIGHT_OPEN != 0 {
                        arena.gt(b, var)
                    } else {
                        arena.ge(b, var)
                    });
                }
                arena.and(&conj)
            }
            ExprNode::FiniteSet(elems) => {
                let eqs: Vec<ExprId> = elems.iter().map(|&e| arena.eq_(var, e)).collect();
                mk_or(arena, &eqs)
            }
            ExprNode::SetUnion(children) => {
                let parts: Vec<ExprId> = children
                    .iter()
                    .map(|c| memo.get(c).copied().unwrap_or(arena.bool_false))
                    .collect();
                mk_or(arena, &parts)
            }
            ExprNode::SetIntersection(children) => {
                let parts: Vec<ExprId> = children
                    .iter()
                    .map(|c| memo.get(c).copied().unwrap_or(arena.bool_true))
                    .collect();
                mk_and(arena, &parts)
            }
            ExprNode::SetComplement(a, b) => {
                let ca = memo.get(&a).copied().unwrap_or(arena.bool_true);
                let cb = memo.get(&b).copied().unwrap_or(arena.bool_false);
                let ncb = arena.not(cb);
                mk_and(arena, &[ca, ncb])
            }
            ExprNode::ConditionSet(v, cond) => crate::transforms::subs::subs(arena, cond, v, var),
            other => {
                return Err(SymplexError::InvalidArgument {
                    operation: "to_condition",
                    reason: format!("not a set expression: {other:?}"),
                });
            }
        };
        memo.insert(id, c);
    }
    memo.get(&set)
        .copied()
        .ok_or(SymplexError::InvalidArgument {
            operation: "to_condition",
            reason: "empty set expression".into(),
        })
}

// ═══════════════════════════════════════════════════════════════════════════
// reduce_inequalities
// ═══════════════════════════════════════════════════════════════════════════

/// Solution set of a real root list: numeric roots become points,
/// symbolic roots stay in a structural finite set, complex roots are
/// dropped.
fn real_roots_set(arena: &mut Arena, roots: &[ExprId]) -> ExprId {
    let mut keep: Vec<ExprId> = Vec::new();
    for &r in roots {
        let ev = crate::transforms::eval::eval(arena, r);
        if numeric_value(arena, ev).is_some() {
            keep.push(ev);
            continue;
        }
        if !crate::base::walk::free_symbols(arena, ev).is_empty() {
            keep.push(ev);
            continue;
        }
        // Constant that is not a finite real: complex or unevaluable.
        match crate::transforms::evalf::evalf(arena, ev, 16) {
            Ok(s) if s.contains('I') || s.contains('i') => {}
            Ok(_) => keep.push(ev),
            Err(_) => keep.push(ev),
        }
    }
    arena.finite_set(&keep)
}

/// Convert one relational atom in `var` to its solution set.
fn atom_to_set(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    rel: crate::transforms::inequalities::Relation,
    is_eq: bool,
    var: ExprId,
) -> Result<ExprId, SymplexError> {
    use crate::transforms::inequalities::Relation;

    let d = arena.sub(lhs, rhs);
    let d = crate::transforms::eval::eval(arena, d);

    if !crate::base::walk::contains(arena, d, var) {
        // Constant condition (with respect to `var`).
        let holds = arena.as_num(d).cloned().map(|r| {
            if is_eq {
                r.is_zero()
            } else {
                match rel {
                    Relation::Gt => r.is_positive(),
                    Relation::Ge => !r.is_negative(),
                    Relation::Lt => r.is_negative(),
                    Relation::Le => !r.is_positive(),
                }
            }
        });
        return match holds {
            Some(true) => Ok(arena.universal_set),
            Some(false) => Ok(arena.empty_set),
            None => Err(SymplexError::InvalidArgument {
                operation: "reduce_inequalities",
                reason: "condition does not involve the variable and cannot be decided".into(),
            }),
        };
    }

    if is_eq {
        let sols = crate::transforms::solve::solve(arena, d, var);
        let roots: Vec<ExprId> = sols.into_iter().map(|s| s.value).collect();
        if roots.is_empty() && crate::poly::polybridge::expr_to_poly(arena, d, var).is_none() {
            return Err(SymplexError::ComputationFailed {
                operation: "reduce_inequalities",
                reason: "could not solve the equation".into(),
            });
        }
        return Ok(real_roots_set(arena, &roots));
    }

    crate::transforms::inequalities::solve_inequality(arena, d, var, rel).map_err(|e| {
        SymplexError::ComputationFailed {
            operation: "reduce_inequalities",
            reason: format!("could not solve inequality: {e}"),
        }
    })
}

/// Reduce a conjunction of univariate boolean conditions in `var` to the
/// solution set in normal form.
///
/// Each condition may be any combination of `And`/`Or`/`Not` over
/// relational atoms (`>`, `>=`, `<`, `<=`, `=`, `!=`) involving `var`.
pub(crate) fn reduce_inequalities(
    arena: &mut Arena,
    conds: &[ExprId],
    var: ExprId,
) -> Result<ExprId, SymplexError> {
    use crate::transforms::inequalities::Relation;

    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::InvalidArgument {
            operation: "reduce_inequalities",
            reason: "variable must be a symbol".into(),
        });
    }

    let mut sets: Vec<ExprId> = Vec::with_capacity(conds.len());
    for &cond in conds {
        // Post-order over the boolean structure.
        let mut order: Vec<ExprId> = Vec::new();
        let mut visited: FxHashSet<ExprId> = FxHashSet::default();
        let mut stack: Vec<(ExprId, bool)> = vec![(cond, false)];
        while let Some((id, expanded)) = stack.pop() {
            if visited.contains(&id) {
                continue;
            }
            if expanded {
                visited.insert(id);
                order.push(id);
                continue;
            }
            stack.push((id, true));
            match arena.node(id) {
                ExprNode::And(ch) | ExprNode::Or(ch) => {
                    for &c in ch.iter().rev() {
                        stack.push((c, false));
                    }
                }
                ExprNode::Not(x) => stack.push((*x, false)),
                _ => {}
            }
        }

        let mut memo: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for &id in &order {
            let s = match arena.node(id).clone() {
                ExprNode::BoolTrue => arena.universal_set,
                ExprNode::BoolFalse => arena.empty_set,
                ExprNode::Gt(a, b) => atom_to_set(arena, a, b, Relation::Gt, false, var)?,
                ExprNode::Ge(a, b) => atom_to_set(arena, a, b, Relation::Ge, false, var)?,
                ExprNode::Eq_(a, b) => atom_to_set(arena, a, b, Relation::Ge, true, var)?,
                ExprNode::Ne(a, b) => {
                    let eq = atom_to_set(arena, a, b, Relation::Ge, true, var)?;
                    let u = arena.universal_set;
                    arena.set_complement(u, eq)
                }
                ExprNode::And(ch) => {
                    let parts: Vec<ExprId> = ch
                        .iter()
                        .map(|c| memo.get(c).copied().unwrap_or(arena.universal_set))
                        .collect();
                    arena.set_intersection(&parts)
                }
                ExprNode::Or(ch) => {
                    let parts: Vec<ExprId> = ch
                        .iter()
                        .map(|c| memo.get(c).copied().unwrap_or(arena.empty_set))
                        .collect();
                    arena.set_union(&parts)
                }
                ExprNode::Not(x) => {
                    let inner = memo.get(&x).copied().unwrap_or(arena.empty_set);
                    let u = arena.universal_set;
                    arena.set_complement(u, inner)
                }
                other => {
                    return Err(SymplexError::InvalidArgument {
                        operation: "reduce_inequalities",
                        reason: format!("not a relational condition: {other:?}"),
                    });
                }
            };
            memo.insert(id, s);
        }
        if let Some(&s) = memo.get(&cond) {
            sets.push(s);
        }
    }

    let combined = arena.set_intersection(&sets);
    Ok(simplify_set(arena, combined))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::node::{INTERVAL_BOTH_CLOSED, INTERVAL_BOTH_OPEN};

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    fn iv(arena: &mut Arena, a: i64, b: i64, flags: u8) -> ExprId {
        let a = arena.int(a);
        let b = arena.int(b);
        arena.interval(a, b, flags)
    }

    // ── RankSet unit tests ────────────────────────────────────────────

    fn r(lo: usize, hi: usize, lo_open: bool, hi_open: bool) -> Piece {
        piece(Pos::Finite(lo), Pos::Finite(hi), lo_open, hi_open)
    }

    #[test]
    fn normalize_merges_overlapping() {
        let s = RankSet::normalize(vec![r(0, 2, false, false), r(1, 3, false, false)]);
        assert_eq!(s.pieces, vec![r(0, 3, false, false)]);
    }

    #[test]
    fn normalize_merges_touching_when_one_side_closed() {
        let s = RankSet::normalize(vec![r(0, 1, false, true), r(1, 2, false, false)]);
        assert_eq!(s.pieces, vec![r(0, 2, false, false)]);
        let s = RankSet::normalize(vec![r(0, 1, false, true), r(1, 2, true, false)]);
        assert_eq!(s.pieces.len(), 2, "both open at 1: gap stays");
    }

    #[test]
    fn normalize_absorbs_points_and_bridges() {
        let s = RankSet::normalize(vec![
            r(0, 1, true, true),
            point(Pos::Finite(1)),
            r(1, 2, true, true),
        ]);
        assert_eq!(s.pieces, vec![r(0, 2, true, true)]);
    }

    #[test]
    fn normalize_drops_invalid() {
        let s = RankSet::normalize(vec![r(2, 1, false, false), r(1, 1, true, false)]);
        assert!(s.is_empty());
        let s = RankSet::normalize(vec![piece(Pos::NegInf, Pos::NegInf, false, false)]);
        assert!(s.is_empty());
    }

    #[test]
    fn complement_of_closed_interval() {
        let s = RankSet::normalize(vec![r(0, 1, false, false)]);
        let c = s.complement();
        assert_eq!(
            c.pieces,
            vec![
                Interval::open(Pos::NegInf, Pos::Finite(0)),
                Interval::open(Pos::Finite(1), Pos::PosInf),
            ]
        );
        assert_eq!(c.complement(), s, "double complement");
    }

    #[test]
    fn complement_of_punctured_line_is_point() {
        let s = RankSet::normalize(vec![
            Interval::open(Pos::NegInf, Pos::Finite(0)),
            Interval::open(Pos::Finite(0), Pos::PosInf),
        ]);
        assert_eq!(s.complement().pieces, vec![point(Pos::Finite(0))]);
        assert!(RankSet::full().complement().is_empty());
        assert!(RankSet::empty().complement().is_full());
    }

    #[test]
    fn intersection_and_difference() {
        let a = RankSet::normalize(vec![r(0, 2, false, false)]);
        let b = RankSet::normalize(vec![r(1, 3, true, true)]);
        assert_eq!(a.intersection(&b).pieces, vec![r(1, 2, true, false)]);
        assert_eq!(a.difference(&b).pieces, vec![r(0, 1, false, false)]);
        let sym = a.difference(&b).union(&b.difference(&a));
        assert_eq!(sym.pieces, vec![r(0, 1, false, false), r(2, 3, true, true)]);
        assert!(a.intersection(&b).is_subset(&a));
        assert!(!a.is_subset(&b));
    }

    #[test]
    fn topology() {
        let a = RankSet::normalize(vec![r(0, 1, true, false), point(Pos::Finite(2))]);
        assert_eq!(
            a.closure().pieces,
            vec![r(0, 1, false, false), point(Pos::Finite(2))]
        );
        assert_eq!(a.interior().pieces, vec![r(0, 1, true, true)]);
        assert_eq!(
            a.boundary().pieces,
            vec![
                point(Pos::Finite(0)),
                point(Pos::Finite(1)),
                point(Pos::Finite(2))
            ]
        );
        assert!(!a.is_open());
        assert!(!a.is_closed());
        assert!(a.closure().is_closed());
        assert!(a.interior().is_open());
        assert!(RankSet::full().is_open() && RankSet::full().is_closed());
        assert!(RankSet::empty().is_open() && RankSet::empty().is_closed());
        assert!(RankSet::full().boundary().is_empty());
    }

    #[test]
    fn contains_respects_openness() {
        let a = RankSet::normalize(vec![r(0, 1, true, false)]);
        assert!(!a.contains(Pos::Finite(0)));
        assert!(a.contains(Pos::Finite(1)));
        assert!(!a.contains(Pos::PosInf));
    }

    // ── Arena-level tests ────────────────────────────────────────────

    #[test]
    fn simplify_intersection_of_intervals() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 2, INTERVAL_BOTH_CLOSED);
        let b = iv(&mut arena, 1, 3, INTERVAL_BOTH_CLOSED);
        let i = arena.set_intersection(&[a, b]);
        let s = simplify_set(&mut arena, i);
        assert_eq!(display(&arena, s), "[1, 2]");
    }

    #[test]
    fn simplify_union_merges_adjacent() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_CLOSED);
        let b = iv(&mut arena, 1, 2, INTERVAL_BOTH_OPEN);
        let u = arena.set_union(&[a, b]);
        let s = simplify_set(&mut arena, u);
        assert_eq!(display(&arena, s), "[0, 2)");
    }

    #[test]
    fn simplify_solve_gt_style_output() {
        // (−∞,−2) ∪ (2,∞) ∩ [0, 5] → (2, 5]
        let mut arena = Arena::new();
        let m2 = arena.int(-2);
        let p2 = arena.int(2);
        let left = arena.interval(arena.neg_infinity, m2, INTERVAL_BOTH_OPEN);
        let right = arena.interval(p2, arena.infinity, INTERVAL_BOTH_OPEN);
        let u = arena.set_union(&[left, right]);
        let box_ = iv(&mut arena, 0, 5, INTERVAL_BOTH_CLOSED);
        let i = arena.set_intersection(&[u, box_]);
        let s = simplify_set(&mut arena, i);
        assert_eq!(display(&arena, s), "(2, 5]");
    }

    #[test]
    fn simplify_points_absorbed_into_intervals() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 2, INTERVAL_BOTH_OPEN);
        let one = arena.int(1);
        let five = arena.int(5);
        let fs = arena.finite_set(&[one, five]);
        let u = arena.set_union(&[a, fs]);
        let s = simplify_set(&mut arena, u);
        assert_eq!(display(&arena, s), "(0, 2) ∪ {5}");
    }

    #[test]
    fn simplify_symbolic_keeps_safe_identities() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.int(1);
        let symbolic = arena.interval(x, one, INTERVAL_BOTH_CLOSED);
        let empty = arena.empty_set;
        let u = arena.set_union(&[symbolic, empty]);
        assert_eq!(simplify_set(&mut arena, u), symbolic);
        let univ = arena.universal_set;
        let comp = arena.set_complement(univ, symbolic);
        let comp2 = arena.set_complement(univ, comp);
        assert_eq!(simplify_set(&mut arena, comp2), symbolic, "(A^c)^c = A");
        let diff = arena.set_complement(symbolic, symbolic);
        assert_eq!(
            simplify_set(&mut arena, diff),
            arena.empty_set,
            "A \\ A = ∅"
        );
    }

    #[test]
    fn simplify_complement_relative_to_reals() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_CLOSED);
        let c = absolute_complement(&mut arena, a);
        assert_eq!(display(&arena, c), "(-oo, 0) ∪ (1, oo)");
    }

    #[test]
    fn simplify_mixed_finite_set() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.int(1);
        let fs = arena.finite_set(&[x, one]);
        let a = iv(&mut arena, 0, 2, INTERVAL_BOTH_CLOSED);
        let u = arena.set_union(&[fs, a]);
        let s = simplify_set(&mut arena, u);
        let d = display(&arena, s);
        assert!(d.contains("[0, 2]") && d.contains("{x}"), "{d}");
    }

    #[test]
    fn simplify_pi_endpoints() {
        let mut arena = Arena::new();
        let three = arena.int(3);
        let four = arena.int(4);
        let pi = arena.pi;
        let a = arena.interval(three, pi, INTERVAL_BOTH_CLOSED);
        let b = arena.interval(pi, four, INTERVAL_BOTH_OPEN);
        let u = arena.set_union(&[a, b]);
        let s = simplify_set(&mut arena, u);
        assert_eq!(display(&arena, s), "[3, 4)");
    }

    #[test]
    fn membership_queries() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_LEFT_OPEN);
        let zero = arena.zero;
        let one = arena.one;
        let half = arena.rational(1, 2);
        let x = arena.symbol("x");
        assert_eq!(set_contains(&mut arena, a, zero), Some(false));
        assert_eq!(set_contains(&mut arena, a, one), Some(true));
        assert_eq!(set_contains(&mut arena, a, half), Some(true));
        assert_eq!(set_contains(&mut arena, a, x), None);
        let e = arena.empty_set;
        assert_eq!(set_contains(&mut arena, e, x), Some(false));
        // symbolic interval, exact difference
        let xp1 = arena.add(&[x, arena.one]);
        let sym = arena.interval(x, xp1, INTERVAL_BOTH_CLOSED);
        assert_eq!(set_contains(&mut arena, sym, x), Some(true));
        assert_eq!(set_contains(&mut arena, sym, xp1), Some(true));
        let two = arena.int(2);
        let xp2 = arena.add(&[x, two]);
        assert_eq!(set_contains(&mut arena, sym, xp2), Some(false));
    }

    #[test]
    fn condition_set_membership_by_substitution() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let zero = arena.zero;
        let cond = arena.gt(x, zero);
        let cs = arena.intern(ExprNode::ConditionSet(x, cond));
        let five = arena.int(5);
        let m5 = arena.int(-5);
        assert_eq!(set_contains(&mut arena, cs, five), Some(true));
        assert_eq!(set_contains(&mut arena, cs, m5), Some(false));
    }

    #[test]
    fn subset_disjoint_empty() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_CLOSED);
        let b = iv(&mut arena, 0, 2, INTERVAL_BOTH_CLOSED);
        let c = iv(&mut arena, 3, 4, INTERVAL_BOTH_CLOSED);
        assert_eq!(is_subset(&mut arena, a, b), Some(true));
        assert_eq!(is_subset(&mut arena, b, a), Some(false));
        assert_eq!(is_disjoint(&mut arena, a, c), Some(true));
        assert_eq!(is_disjoint(&mut arena, a, b), Some(false));
        assert_eq!(is_empty(&mut arena, a), Some(false));
        let e = arena.set_intersection(&[a, c]);
        assert_eq!(is_empty(&mut arena, e), Some(true));
        let x = arena.symbol("x");
        let one = arena.one;
        let sym = arena.interval(x, one, INTERVAL_BOTH_CLOSED);
        assert_eq!(is_empty(&mut arena, sym), None);
        assert_eq!(is_subset(&mut arena, sym, sym), Some(true));
    }

    #[test]
    fn inf_sup_measure() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_OPEN);
        let b = iv(&mut arena, 2, 5, INTERVAL_BOTH_CLOSED);
        let u = arena.set_union(&[a, b]);
        assert_eq!(inf(&mut arena, u), Some(arena.zero));
        assert_eq!(inf(&mut arena, u), Some(arena.zero));
        let five = arena.int(5);
        assert_eq!(sup(&mut arena, u), Some(five));
        let four = arena.int(4);
        assert_eq!(measure(&mut arena, u), Some(four));
        let ray = arena.interval(arena.zero, arena.infinity, INTERVAL_BOTH_OPEN);
        assert_eq!(measure(&mut arena, ray), Some(arena.infinity));
        assert_eq!(sup(&mut arena, ray), Some(arena.infinity));
        let e = arena.empty_set;
        assert_eq!(inf(&mut arena, e), None);
        assert_eq!(measure(&mut arena, e), Some(arena.zero));
    }

    #[test]
    fn topology_via_arena() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_OPEN);
        let c = closure(&mut arena, a).unwrap();
        assert_eq!(display(&arena, c), "[0, 1]");
        let i = interior(&mut arena, c).unwrap();
        assert_eq!(i, a);
        let b = boundary(&mut arena, a).unwrap();
        assert_eq!(display(&arena, b), "{0, 1}");
        assert_eq!(is_open(&mut arena, a), Some(true));
        assert_eq!(is_closed(&mut arena, a), Some(false));
        assert_eq!(is_closed(&mut arena, c), Some(true));
    }

    #[test]
    fn accessors() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, INTERVAL_BOTH_OPEN);
        let three = arena.int(3);
        let fs = arena.finite_set(&[three]);
        let u = arena.set_union(&[a, fs]);
        let ivs = as_intervals(&mut arena, u).unwrap();
        assert_eq!(ivs.len(), 2);
        assert_eq!(ivs[0], Interval::open(arena.zero, arena.one));
        assert_eq!(ivs[1], Interval::point(three));
        assert!(as_finite_set(&mut arena, u).is_none());
        assert_eq!(as_finite_set(&mut arena, fs), Some(vec![three]));
    }

    #[test]
    fn condition_roundtrip() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let a = iv(&mut arena, 0, 1, INTERVAL_LEFT_OPEN);
        let two = arena.int(2);
        let fs = arena.finite_set(&[two]);
        let u = arena.set_union(&[a, fs]);
        let c = to_condition(&mut arena, u, x).unwrap();
        let d = display(&arena, c);
        assert!(
            d.contains("x > 0") && d.contains("1 >= x") && d.contains("x == 2"),
            "{d}"
        );
        let back = reduce_inequalities(&mut arena, &[c], x).unwrap();
        assert_eq!(display(&arena, back), "(0, 1] ∪ {2}");
    }

    #[test]
    fn reduce_inequalities_examples() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let four = arena.int(4);
        let five = arena.int(5);
        let x2 = arena.pow(x, two);
        let c1 = arena.gt(x2, four);
        let c2 = arena.gt(five, x);
        let s = reduce_inequalities(&mut arena, &[c1, c2], x).unwrap();
        assert_eq!(display(&arena, s), "(-oo, -2) ∪ (2, 5)");

        let zero = arena.zero;
        let m3 = arena.int(-3);
        let ge0 = arena.ge(x, zero);
        let ltm3 = arena.gt(m3, x);
        let or = arena.or(&[ge0, ltm3]);
        let s = reduce_inequalities(&mut arena, &[or], x).unwrap();
        assert_eq!(display(&arena, s), "(-oo, -3) ∪ [0, oo)");

        let one = arena.one;
        let le1 = arena.ge(one, x2);
        let not = arena.not(le1);
        let s = reduce_inequalities(&mut arena, &[not], x).unwrap();
        assert_eq!(display(&arena, s), "(-oo, -1) ∪ (1, oo)");

        let eq = arena.eq_(x2, four);
        let s = reduce_inequalities(&mut arena, &[eq], x).unwrap();
        assert_eq!(display(&arena, s), "{-2, 2}");

        let ne = arena.ne_(x, two);
        let s = reduce_inequalities(&mut arena, &[ne], x).unwrap();
        assert_eq!(display(&arena, s), "(-oo, 2) ∪ (2, oo)");
    }

    #[test]
    fn reduce_inequalities_rejects_non_relational() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let y = arena.symbol("y");
        let zero = arena.zero;
        let cond = arena.gt(y, zero);
        assert!(reduce_inequalities(&mut arena, &[cond], x).is_err());
        assert!(reduce_inequalities(&mut arena, &[cond], zero).is_err());
    }
}
