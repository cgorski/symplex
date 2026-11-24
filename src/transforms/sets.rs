//! Set algebra: interval arithmetic, membership, subset tests, inf/sup, measure.
//!
//! Sets in symplex are built from the set nodes in `ExprNode`: `Interval`,
//! `FiniteSet`, `SetUnion`, `SetIntersection`, `SetComplement` (relative
//! complement `A \ B`), `EmptySet`, `UniversalSet` (every value, real or
//! not), `ConditionSet` and `ImageSet` (the image `{f(n) : n ∈ ℤ}` of the
//! integers: periodic solution families, see [Image sets](#image-sets)).
//! Intervals are sets of real numbers, and so is the normal form; finite
//! sets, image sets and condition sets may hold members that are not real
//! (the roots `±i` of `x² + 1`, the family `2nπi`), and the universal set
//! holds them all.
//!
//! # Real members
//!
//! An interval, the real line `(−∞, ∞)` and every set in normal form hold
//! real numbers only, so whatever they do to an element — contain it,
//! absorb it in a union, remove it in a difference — needs the element
//! **proved real** (`member_realness`, three-valued; `members_real` for a
//! whole operand): proved real, it is compared with the ends; certainly
//! not real (`i`, `±∞`, `zoo`), it is not a member; undecided (a symbol
//! without assumptions), the membership is undecided and the expression
//! stays as it is — `{i} ∪ ℝ` and `{x} ∪ ℝ` keep their points, `{i} \ ℝ` is
//! `{i}`, `{x} \ ℝ` stays, `ℝ ∩ {x}` keeps its `ℝ`.  Up to 0.59 the evaluator
//! gave the universal set and the real line one value, and its shortcuts
//! on "the whole line" took every operand for a set of real numbers: `{i} ∪
//! ℝ` was `(−∞, ∞)`, `{i} \ ℝ` was `∅`, and `U \ {0}` was `ℝ \ {0}`.
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
//! All queries are three-valued ([`Truth`]; `Option<bool>` at the public
//! API, converted with [`Truth::to_option`]): `Unknown` means "cannot be
//! decided", never "false".
//!
//! # Image sets
//!
//! `ImageSet(n, f)` is `{f(n) : n ∈ ℤ}`, SymPy's `ImageSet(Lambda(n, f),
//! Integers)` (`sympy/sets/fancysets.py`).  `image_set` builds the
//! canonical form: the bound variable is `_n` (declared integer; `_n_1`, …
//! when `_n` is free in `f` or declared otherwise), and a family linear in
//! `n`, `a·n + b`, has `a` with a positive coefficient (`n ↦ −n`) and the
//! part of `b` that is a rational multiple `k·a` reduced to `0 ≤ k < 1`, as
//! SymPy's `_set_function` for the integers reduces `b mod a`
//! (`sets/handlers/functions.py`).  Simplification
//! (`resolve_image_sets`, before the normal form):
//!
//! * a union merges families of commensurable steps into the coarsest
//!   lattices (`2nπ ∪ 2nπ + π = nπ`; SymPy keeps both) and drops the
//!   elements of finite sets that are members (`{0} ∪ nπ = nπ`);
//! * an intersection with a bounded set lists the members of each family
//!   in its hull (`nπ ∩ [−4, 4] = {−π, 0, π}`, SymPy's `ImageSet ∩
//!   Interval` handler), one with a finite set keeps its members, one with
//!   the real line keeps the real members of a family with a non-real step
//!   (`2nπi ∩ ℝ = {0}`);
//! * `A \ {p, …}` forgets the points that are not members of `A`.
//!
//! Membership (`image_set_contains`) solves `f(n) = x` for `n` and asks
//! whether the solution is an integer: exactly for a rational, by the
//! assumptions for a symbolic one, and for a constant by a certified
//! evaluation (not within `10⁻⁹` of an integer: not a member; within it:
//! the member at the nearest integer is compared with `x` exactly or to a
//! certified digit), as SymPy's `ImageSet._contains` solves over the base
//! set.
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
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::truth::Truth;

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
///
/// The value places the expression on the real line (the rank table, the
/// order of two constants), so a constant that is not a rational number
/// must be **proved** real ([`member_realness`]: by the assumptions or a
/// certified imaginary part).  Its 16-digit value alone is no proof: an
/// imaginary part that is zero only to the precision reached reads as 0
/// there, and up to 0.59 `1 + i·(sin²1 + cos²1 − 1 + e⁻⁵⁰⁰⁰)` was the
/// point `1` of `[0, 2]`.
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
            // not proved real: no place on the line
            if !member_realness(arena, ev).is_true() {
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
/// (Certified 16-digit values: farther apart than the tolerance their order
/// is certain; closer pairs are ordered exactly or left out of the table.)
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
    /// Fully evaluated normal form: a set of real numbers.
    Exact(RankSet),
    /// `exact ∪ syms` where every element of `syms` is an opaque,
    /// already-simplified set expression (non-empty list).  The opaque
    /// parts may hold members that are not real.
    Mixed(RankSet, Vec<ExprId>),
    /// Opaque, already-simplified set expression.
    Sym(ExprId),
    /// The universal set: every value, real or not.  It is not
    /// `Exact(RankSet::full())`, which is the real line: up to 0.59 the two
    /// had one value, and what holds for one was concluded for the other
    /// (`{i} \ ℝ = ∅`, `{i} ∪ ℝ = ℝ`, `U \ {0} = ℝ \ {0}`).
    Universe,
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

    /// Is the value exactly the real line `(−∞, ∞)`?
    fn is_real_line(&self) -> bool {
        matches!(self, SetVal::Exact(rs) if rs.is_full())
    }

    fn is_universe(&self) -> bool {
        matches!(self, SetVal::Universe)
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
            parts.push(arena.make(ExprNode::Interval(lo, hi, p.kind)));
        }
    }
    if !points.is_empty() {
        parts.push(arena.finite_set(&points));
    }
    // The canonical union keeps this order (real intervals ascending, then
    // the finite set).  Up to 0.51 a raw node: `(0, 1) ∪ [2, 3]` was not
    // `==` to the canonical union of its parts (its JSON round trip).
    arena.set_union(&parts)
}

fn setval_to_expr(arena: &mut Arena, table: &Table, v: &SetVal) -> ExprId {
    match v {
        SetVal::Universe => arena.universal_set,
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
        ExprNode::UniversalSet => SetVal::Universe,

        ExprNode::Interval(a, b, kind) => match (table.pos_of(a), table.pos_of(b)) {
            (Some(lo), Some(hi)) => {
                let piece = piece(lo, hi, kind.lower_open(), kind.upper_open());
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
                    SetVal::Sym(arena.interval(ea, eb, kind))
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
                    // the universal set absorbs every set
                    Some(SetVal::Universe) => return SetVal::Universe,
                    Some(SetVal::Exact(rs)) => exact = exact.union(rs),
                    Some(SetVal::Mixed(rs, s)) => {
                        exact = exact.union(rs);
                        syms.extend_from_slice(s);
                    }
                    Some(SetVal::Sym(s)) => syms.push(*s),
                    None => syms.push(c),
                }
            }
            finish_union(arena, exact, syms)
        }

        ExprNode::SetIntersection(children) => {
            // An operand `U \ X` of the universal set `U` takes `X` out of
            // the meet of the others: `A ∩ (U \ X) = A \ X` (the domain of
            // a nonzero symbol not declared real is `U \ {0}`).
            let mut operands: Vec<SetVal> = Vec::with_capacity(children.len());
            let mut removed: Vec<SetVal> = Vec::new();
            for c in children {
                if let ExprNode::SetComplement(u, x) = *arena.node(c)
                    && vals.get(&u).is_some_and(SetVal::is_universe)
                {
                    removed.push(vals.get(&x).cloned().unwrap_or(SetVal::Sym(x)));
                } else {
                    operands.push(vals.get(&c).cloned().unwrap_or(SetVal::Sym(c)));
                }
            }
            let mut meet = meet_vals(arena, table, operands);
            for vx in &removed {
                meet = difference_vals(arena, table, &meet, vx);
            }
            meet
        }

        ExprNode::SetComplement(a, b) => {
            let va = vals.get(&a).cloned().unwrap_or(SetVal::Sym(a));
            let vb = vals.get(&b).cloned().unwrap_or(SetVal::Sym(b));
            eval_difference(arena, table, vals, a, b, &va, &vb)
        }

        ExprNode::ConditionSet(var, cond) => {
            let c = crate::transforms::logic::simplify_bool(arena, cond);
            if c == arena.bool_true {
                // Every value of `var` in its declared domain: for a plain
                // symbol the universal set, not the real line `RankSet::full`
                // stands for (up to 0.52 `ConditionSet(x, y ≥ 0)` at `y = 1`
                // simplified to `(−∞, ∞)` for a complex `x`).
                match declared_domain_set(arena, var, false) {
                    DeclaredDomain::Unrestricted => SetVal::Universe,
                    DeclaredDomain::Set(d) => {
                        let (ninf, inf) = (arena.neg_infinity, arena.infinity);
                        if d == arena.interval(ninf, inf, IntervalKind::Open) {
                            SetVal::Exact(RankSet::full())
                        } else {
                            SetVal::Sym(d)
                        }
                    }
                    DeclaredDomain::Unsupported => {
                        SetVal::Sym(arena.intern(ExprNode::ConditionSet(var, c)))
                    }
                }
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

/// `⋂ operands` (the universal set for no operand).
fn meet_vals(arena: &mut Arena, table: &Table, operands: Vec<SetVal>) -> SetVal {
    let mut exact: Option<RankSet> = None;
    let mut others: Vec<SetVal> = Vec::new();
    for v in operands {
        match v {
            // the universal set cuts nothing
            SetVal::Universe => {}
            SetVal::Exact(rs) => {
                exact = Some(match exact {
                    Some(e) => e.intersection(&rs),
                    None => rs,
                });
            }
            SetVal::Mixed(..) | SetVal::Sym(_) => others.push(v),
        }
    }
    let Some(e) = exact else {
        return match others.len() {
            // every operand is the universal set
            0 => SetVal::Universe,
            1 => others.swap_remove(0),
            _ => {
                let mut opaque: Vec<ExprId> = Vec::with_capacity(others.len());
                for v in &others {
                    opaque.push(setval_to_expr(arena, table, v));
                }
                SetVal::Sym(arena.set_intersection(&opaque))
            }
        };
    };
    if e.is_empty() {
        return SetVal::Exact(e);
    }
    let mut opaque: Vec<ExprId> = Vec::with_capacity(others.len());
    for v in &others {
        // An operand that holds every number of `e` cuts nothing: `e ∩ (R ∪
        // S) = e` for `e ⊆ R` (`[0, 1] ∩ (ℝ ∪ {i})`).
        if let SetVal::Mixed(rs, _) = v
            && e.is_subset(rs)
        {
            continue;
        }
        opaque.push(setval_to_expr(arena, table, v));
    }
    if opaque.is_empty() {
        return SetVal::Exact(e);
    }
    if e.is_full() {
        return match meet_with_line(arena, table, opaque) {
            Some(meet) => SetVal::Sym(meet),
            None => SetVal::Exact(RankSet::empty()),
        };
    }
    // A set of real numbers meets no member that is certainly not real
    // (`[0, 1] ∩ {i} = ∅`).
    if matches!(cut_parts_to_line(arena, &mut opaque), LineCut::Empty) {
        return SetVal::Exact(RankSet::empty());
    }
    let mut parts = Vec::with_capacity(opaque.len() + 1);
    parts.push(rankset_to_expr(arena, table, &e));
    parts.extend(opaque);
    SetVal::Sym(arena.set_intersection(&parts))
}

/// What [`cut_parts_to_line`] left of the finite sets among the parts of an
/// intersection with a set of real numbers.
enum LineCut {
    /// A part lost every member: the meet is empty.
    Empty,
    /// Every member left is proved real.
    Real,
    /// A member not known to be real is left.
    Undecided,
}

/// The opaque `parts` of an intersection with a set of real numbers, each
/// with the members of its finite sets that are certainly not real dropped
/// ([`cut_finite_to_line`]).
fn cut_parts_to_line(arena: &mut Arena, parts: &mut [ExprId]) -> LineCut {
    let mut undecided_finite = false;
    for p in parts.iter_mut() {
        let (cut, undecided) = cut_finite_to_line(arena, *p);
        if cut == arena.empty_set {
            return LineCut::Empty;
        }
        *p = cut;
        undecided_finite |= undecided;
    }
    if undecided_finite {
        LineCut::Undecided
    } else {
        LineCut::Real
    }
}

/// `exact ∪ ⋃ syms`.  When the exact part is the whole real line it absorbs
/// what is proved to lie on it and nothing else
/// ([`parts_beyond_the_line`]): up to 0.59 it absorbed every part, and
/// `{i} ∪ ℝ`, `{x} ∪ ℝ` for a plain `x` and `{2nπi} ∪ ℝ` were `(−∞, ∞)`.
fn finish_union(arena: &mut Arena, exact: RankSet, syms: Vec<ExprId>) -> SetVal {
    let mut syms = if exact.is_full() {
        parts_beyond_the_line(arena, syms)
    } else {
        syms
    };
    let mut seen = FxHashSet::default();
    syms.retain(|s| seen.insert(*s));
    if syms.is_empty() {
        SetVal::Exact(exact)
    } else {
        SetVal::Mixed(exact, syms)
    }
}

/// What the opaque `parts` of a union add to the real line: a part whose
/// members are all proved real adds nothing, a finite set adds its members
/// that are not proved real, and any other part stays whole (a member not
/// proved real may lie off the line, so the line does not absorb it).
fn parts_beyond_the_line(arena: &mut Arena, parts: Vec<ExprId>) -> Vec<ExprId> {
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        if let ExprNode::FiniteSet(elems) = arena.node(p).clone() {
            let mut kept: Vec<ExprId> = Vec::with_capacity(elems.len());
            for &e in &elems {
                // not proved real: kept beside the line
                if !element_realness(arena, e).is_true() {
                    kept.push(e);
                }
            }
            if kept.len() == elems.len() {
                out.push(p);
            } else if !kept.is_empty() {
                out.push(arena.finite_set(&kept));
            }
        } else if !members_real(arena, p).is_true() {
            // not proved to be a set of real numbers
            out.push(p);
        }
    }
    out
}

/// `ℝ ∩ ⋂ opaque` for opaque (already simplified) parts; `None` when the
/// meet is empty.
///
/// The real line adds nothing to a set of real numbers, but it cuts every
/// other set.  The members of finite sets that are certainly not real are
/// dropped, and the line is kept unless one part is proved to hold real
/// numbers only ([`members_real`]): a finite set with a member not known
/// to be real (`±√(−a)` for a plain `a`: up to 0.46 the real solutions of
/// `x² + a = 0` were `{√(−a), −√(−a)}`, imaginary for `a > 0`; SymPy: `{−√(−a),
/// √(−a)} ∩ ℝ`), an image set whose members are not all real (`√(2nπ + π/6)`
/// is imaginary for `n < 0`: before 0.45 `ℝ ∩ {√(2nπ + π/6)}` lost its `ℝ`),
/// and, since 0.60, every other part not proved real (the condition set of
/// a variable not declared real: up to 0.59 `ℝ ∩ {x | x = cos x}` lost its
/// `ℝ` for a plain `x`).  Up to 0.59 the cut was made only next to an
/// operand written as an interval: `((−∞, 0] ∪ (0, ∞)) ∩ {i}` was `{i}`.
fn meet_with_line(arena: &mut Arena, table: &Table, mut opaque: Vec<ExprId>) -> Option<ExprId> {
    let undecided_finite = match cut_parts_to_line(arena, &mut opaque) {
        LineCut::Empty => return None,
        LineCut::Real => false,
        LineCut::Undecided => true,
    };
    let mut keep_line = undecided_finite
        || opaque
            .iter()
            .any(|&p| contains_image_set(arena, p) && !all_members_real(arena, p));
    if !keep_line {
        // (each part is asked until one is proved real)
        let mut proved = false;
        for &p in &opaque {
            if members_real(arena, p).is_true() {
                proved = true;
                break;
            }
        }
        keep_line = !proved;
    }
    let mut parts = Vec::with_capacity(opaque.len() + 1);
    if keep_line {
        parts.push(rankset_to_expr(arena, table, &RankSet::full()));
    }
    parts.extend(opaque);
    Some(arena.set_intersection(&parts))
}

/// `d \ ⋃ parts` for a set `d` of real numbers and opaque `parts`.  Only a
/// real number can be taken out of `d`: the members of finite sets that are
/// certainly not real are forgotten (`[0, 1] \ {i} = [0, 1]`).
fn real_minus_parts(arena: &mut Arena, table: &Table, d: RankSet, parts: &[ExprId]) -> SetVal {
    if d.is_empty() {
        return SetVal::Exact(d);
    }
    let mut removed: Vec<ExprId> = Vec::with_capacity(parts.len());
    for &s in parts {
        if let ExprNode::FiniteSet(elems) = arena.node(s).clone() {
            let mut kept: Vec<ExprId> = Vec::with_capacity(elems.len());
            for &e in &elems {
                // not refuted: it may be a real number of `d`
                if !element_realness(arena, e).is_false() {
                    kept.push(e);
                }
            }
            if kept.len() == elems.len() {
                removed.push(s);
            } else if !kept.is_empty() {
                removed.push(arena.finite_set(&kept));
            }
        } else {
            removed.push(s);
        }
    }
    if removed.is_empty() {
        return SetVal::Exact(d);
    }
    let d_expr = rankset_to_expr(arena, table, &d);
    let s_expr = arena.set_union(&removed);
    SetVal::Sym(arena.set_complement(d_expr, s_expr))
}

/// `(⋃ parts) \ B` for opaque `parts` and a non-empty set `B` of real
/// numbers (`eb`), as the parts of a union.  A member is removed only when
/// it is proved to be in `B`, which needs it proved real: a member of a
/// finite set that is certainly not real is not in `B` and stays, one
/// proved real is removed by the whole line (and stays under the complement
/// of a smaller `B`, like an undecided one), and a part other than a
/// finite set is removed by the line when all its members are proved real.
/// Up to 0.59 the line removed everything: `{i} \ ℝ`, `{x} \ ℝ` for a plain
/// `x` and `{2nπi} \ ℝ` were `∅`.
fn parts_minus_real(
    arena: &mut Arena,
    table: &Table,
    parts: &[ExprId],
    eb: &RankSet,
) -> Vec<ExprId> {
    let line = eb.is_full();
    let mut free: Vec<ExprId> = Vec::new();
    let mut under: Vec<ExprId> = Vec::new();
    for &s in parts {
        if let ExprNode::FiniteSet(elems) = arena.node(s).clone() {
            let mut kept: Vec<ExprId> = Vec::with_capacity(elems.len());
            for &e in &elems {
                match element_realness(arena, e) {
                    Truth::False => free.push(e),
                    Truth::True if line => {}
                    // a real number not located in a part of the line, or
                    // a member not known to be real
                    Truth::True | Truth::Unknown => kept.push(e),
                }
            }
            if kept.len() == elems.len() {
                under.push(s);
            } else if !kept.is_empty() {
                under.push(arena.finite_set(&kept));
            }
        } else if !(line && members_real(arena, s).is_true()) {
            under.push(s);
        }
    }
    let mut out: Vec<ExprId> = Vec::with_capacity(2);
    if !free.is_empty() {
        out.push(arena.finite_set(&free));
    }
    if !under.is_empty() {
        let s_expr = arena.set_union(&under);
        let b_expr = rankset_to_expr(arena, table, eb);
        out.push(arena.set_complement(s_expr, b_expr));
    }
    out
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
    // ∅ \ B = ∅ ;  A \ U = ∅ for the universal set `U` (the real line removes
    // the real numbers only: `{i} \ ℝ = {i}`)
    if va.is_known_empty() || vb.is_universe() {
        return SetVal::Exact(RankSet::empty());
    }
    // A \ ∅ = A
    if vb.is_known_empty() {
        return va.clone();
    }
    if let ExprNode::SetComplement(u, x) = *arena.node(b) {
        let vu = vals.get(&u);
        // A \ (U \ X) = A ∩ X for the universal set `U`; with `A = U`,
        // (Xᶜ)ᶜ = X.
        if vu.is_some_and(SetVal::is_universe) {
            let vx = vals.get(&x).cloned().unwrap_or(SetVal::Sym(x));
            return meet_vals(arena, table, vec![va.clone(), vx]);
        }
        // ℝ \ (ℝ \ X) = ℝ ∩ X, which is `X` for a set of real numbers
        if va.is_real_line() && vu.is_some_and(SetVal::is_real_line) {
            let vx = vals.get(&x).cloned().unwrap_or(SetVal::Sym(x));
            if let SetVal::Mixed(..) | SetVal::Sym(_) = vx {
                let x_expr = setval_to_expr(arena, table, &vx);
                if members_real(arena, x_expr).is_true() {
                    return vx;
                }
            }
            return match vx {
                SetVal::Universe => va.clone(),
                SetVal::Exact(rs) => SetVal::Exact(rs),
                SetVal::Mixed(rs, syms) => {
                    let s = arena.set_union(&syms);
                    match meet_with_line(arena, table, vec![s]) {
                        Some(m) => finish_union(arena, rs, vec![m]),
                        None => SetVal::Exact(rs),
                    }
                }
                SetVal::Sym(s) => match meet_with_line(arena, table, vec![s]) {
                    Some(m) => SetVal::Sym(m),
                    None => SetVal::Exact(RankSet::empty()),
                },
            };
        }
    }
    difference_vals(arena, table, va, vb)
}

/// `A \ B` from the values of `A` and `B`.
fn difference_vals(arena: &mut Arena, table: &Table, va: &SetVal, vb: &SetVal) -> SetVal {
    if va.is_known_empty() {
        return SetVal::Exact(RankSet::empty());
    }
    if vb.is_known_empty() {
        return va.clone();
    }
    match (va, vb) {
        (_, SetVal::Universe) => SetVal::Exact(RankSet::empty()),
        // U \ B: the normal form holds real numbers only, and says nothing
        // of the other values of `U`.  Up to 0.59 `U \ {0}` was `ℝ \ {0}`.
        (SetVal::Universe, _) => {
            let b_expr = setval_to_expr(arena, table, vb);
            let u = arena.universal_set;
            SetVal::Sym(arena.set_complement(u, b_expr))
        }
        (SetVal::Exact(ea), SetVal::Exact(eb)) => SetVal::Exact(ea.difference(eb)),
        (SetVal::Exact(ea), SetVal::Mixed(eb, sb)) => {
            // A \ (E ∪ S) = (A \ E) \ S
            let d = ea.difference(eb);
            real_minus_parts(arena, table, d, sb)
        }
        (SetVal::Exact(ea), SetVal::Sym(sb)) => real_minus_parts(arena, table, ea.clone(), &[*sb]),
        (SetVal::Mixed(ea, sa), SetVal::Exact(eb)) => {
            // (E ∪ S) \ B = (E \ B) ∪ (S \ B)
            let d = ea.difference(eb);
            let rest = parts_minus_real(arena, table, sa, eb);
            finish_union(arena, d, rest)
        }
        (SetVal::Sym(sa), SetVal::Exact(eb)) => {
            let rest = parts_minus_real(arena, table, &[*sa], eb);
            match rest.len() {
                0 => SetVal::Exact(RankSet::empty()),
                1 => SetVal::Sym(rest[0]),
                _ => SetVal::Mixed(RankSet::empty(), rest),
            }
        }
        (SetVal::Mixed(..) | SetVal::Sym(_), SetVal::Mixed(..) | SetVal::Sym(_)) => {
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
    let set = resolve_image_sets(arena, set);
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

/// Absolute complement `ℝ \ a` in normal form.  (Built on the real line:
/// up to 0.59 on `UniversalSet`, which the evaluator took for the line.)
pub(crate) fn absolute_complement(arena: &mut Arena, a: ExprId) -> ExprId {
    let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
    let line = arena.interval(ninf, pinf, IntervalKind::Open);
    let c = arena.set_complement(line, a);
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
/// The literal `UniversalSet` contains everything.  An interval, the real
/// line `(-∞, ∞)` and every set in normal form hold real numbers only: an
/// element is a member only when it is proved real ([`member_realness`]),
/// one that is certainly not real (`i`, `±∞`, `zoo`) is not a member, and
/// for one not known to be real the question is undecided.
pub(crate) fn set_contains(arena: &mut Arena, set: ExprId, elem: ExprId) -> Truth {
    if set == arena.universal_set {
        return Truth::True;
    }
    let ev = evaluate(arena, &[set], &[elem]);
    let elem_pos = ev.table.pos_of(elem);
    let elem_ev = ev.table.ev(elem);
    let (order, _) = collect(arena, &[set]);
    let mut memo: FxHashMap<ExprId, Truth> = FxHashMap::default();
    // Is the element a real number?  Asked once, and only of an element
    // without a place on the line.
    let mut elem_real: Option<Truth> = None;
    let mut realness = |arena: &mut Arena| -> Truth {
        *elem_real.get_or_insert_with(|| member_realness(arena, elem_ev))
    };

    for &id in &order {
        // Exact fast path.
        let known = match ev.vals.get(&id) {
            Some(SetVal::Universe) => Some(Truth::True),
            Some(SetVal::Exact(rs)) => Some(match elem_pos {
                Some(p @ Pos::Finite(_)) => Truth::from_bool(rs.contains(p)),
                Some(_) => Truth::False, // ±∞ is never a member of a real set
                None if rs.is_empty() => Truth::False,
                // a set of real numbers: all of them, when it is the line
                None => match realness(arena) {
                    Truth::False => Truth::False,
                    Truth::True if rs.is_full() => Truth::True,
                    Truth::True | Truth::Unknown => Truth::Unknown,
                },
            }),
            Some(SetVal::Mixed(..) | SetVal::Sym(_)) | None => None,
        };
        if let Some(r) = known {
            memo.insert(id, r);
            continue;
        }
        let r = match arena.node(id).clone() {
            ExprNode::EmptySet => Truth::False,
            ExprNode::UniversalSet => Truth::True,
            // An interval holds real numbers only, whatever its ends: an
            // element that is certainly not real is not compared with them
            // (up to 0.59 `i + 1/2 ∈ [i, i + 1]` held by the order of the
            // differences).  The order of an element not known to be real
            // is taken where the interval is defined (`x ∈ [x, x + 1]`).
            ExprNode::Interval(a, b, kind) => {
                let real = match elem_pos {
                    Some(Pos::Finite(_)) => Truth::True,
                    Some(Pos::NegInf | Pos::PosInf) => Truth::False,
                    None => realness(arena),
                };
                if real.is_false() {
                    Truth::False
                } else {
                    let lo = cmp_exprs(arena, a, elem_ev);
                    let hi = cmp_exprs(arena, elem_ev, b);
                    inside_end(lo, kind.lower_open()).and(inside_end(hi, kind.upper_open()))
                }
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
                    Truth::True
                } else if any_unknown {
                    Truth::Unknown
                } else {
                    Truth::False
                }
            }
            ExprNode::SetUnion(children) => {
                Truth::any(children.iter().map(|&c| verdict_of(&memo, c)))
            }
            ExprNode::SetIntersection(children) => {
                Truth::all(children.iter().map(|&c| verdict_of(&memo, c)))
            }
            ExprNode::SetComplement(a, b) => verdict_of(&memo, a).and(verdict_of(&memo, b).not()),
            // `{x | cond}` for `x` in its declared domain: up to 0.42 a
            // value outside it was a member when it met the condition
            // (`−7 ∈ {x | x³ − 9x < 0}` for a prime `x`).
            ExprNode::ConditionSet(var, cond) => {
                let c = crate::transforms::subs::subs(arena, cond, var, elem_ev);
                let c = crate::transforms::logic::simplify_bool(arena, c);
                if c == arena.bool_false {
                    Truth::False
                } else {
                    let in_domain = Truth::from(crate::transforms::solve::assumptions_hold(
                        arena, var, elem_ev,
                    ));
                    match in_domain {
                        Truth::False => Truth::False,
                        Truth::True if c == arena.bool_true => Truth::True,
                        // in the domain with an undecided condition, or an
                        // undecided domain: no claim
                        Truth::True | Truth::Unknown => Truth::Unknown,
                    }
                }
            }
            ExprNode::ImageSet(var, body) => image_set_contains(arena, var, body, elem_ev),
            _ => Truth::Unknown,
        };
        memo.insert(id, r);
    }
    verdict_of(&memo, set)
}

/// The verdict a bottom-up pass recorded for the operand `id`; `Unknown`
/// for one it did not reach.
fn verdict_of(memo: &FxHashMap<ExprId, Truth>, id: ExprId) -> Truth {
    memo.get(&id).copied().unwrap_or(Truth::Unknown)
}

/// Is a point on the inner side of an interval end?  `order` compares the
/// lower end with the point, or the point with the upper end: `Less` is
/// inside, `Equal` is inside a closed end, and an undecided order is
/// `Unknown`.
fn inside_end(order: Option<Ordering>, open: bool) -> Truth {
    match order {
        Some(Ordering::Less) => Truth::True,
        Some(Ordering::Equal) => Truth::from_bool(!open),
        Some(Ordering::Greater) => Truth::False,
        None => Truth::Unknown,
    }
}

/// Is `e` a real number: a certified evaluation for a constant, the
/// assumptions of its symbols otherwise (a symbol without assumptions is
/// not known to be real).  `±∞` and `zoo` are not real numbers; `nan` is no
/// value, and nothing is decided for it.
///
/// This is the test every membership in a set of real numbers goes
/// through — an interval, the real line, a set in normal form: proved real,
/// the element is compared with the ends; certainly not real, it is not a
/// member; undecided, the membership is, and the set expression that asks
/// stays as it is.
fn member_realness(arena: &mut Arena, e: ExprId) -> Truth {
    use crate::base::value_class::ValueClass;
    match ValueClass::special_atom(arena, e) {
        Some(ValueClass::NaN) => return Truth::Unknown,
        // an infinity is no real number
        Some(ValueClass::PosInf | ValueClass::NegInf | ValueClass::ComplexInf) => {
            return Truth::False;
        }
        // (the other classes are no atoms)
        Some(
            ValueClass::Finite
            | ValueClass::DirectedInf
            | ValueClass::Infinite
            | ValueClass::Unknown,
        )
        | None => {}
    }
    if is_known_real(arena, e) {
        return Truth::True;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    if crate::base::walk::free_symbols(arena, e).is_empty() {
        return Truth::from(crate::transforms::realness::constant_realness_as_declared(
            arena, e, 20, &mut cache,
        ));
    }
    Truth::from(cache.query(arena, e, crate::base::assumptions::Props::REAL))
}

/// Is the member `e` of a finite set a real number?  A member with a real
/// numeric value is; any other is asked of [`member_realness`].
fn element_realness(arena: &mut Arena, e: ExprId) -> Truth {
    if numeric_value(arena, e).is_some() {
        Truth::True
    } else {
        member_realness(arena, e)
    }
}

/// A part of an intersection with the real line, with the members of its
/// finite sets (the part itself or the children of a union) that are
/// certainly not real dropped (`EmptySet` when nothing is left), and
/// whether a member remains that is not known to be real.
fn cut_finite_to_line(arena: &mut Arena, p: ExprId) -> (ExprId, bool) {
    let cut = |arena: &mut Arena, elems: &[ExprId]| -> (ExprId, bool) {
        let mut kept = Vec::with_capacity(elems.len());
        let mut undecided = false;
        for &m in elems {
            match member_realness(arena, m) {
                Truth::True => kept.push(m),
                Truth::False => {}
                Truth::Unknown => {
                    undecided = true;
                    kept.push(m);
                }
            }
        }
        if kept.is_empty() {
            (arena.empty_set, false)
        } else if kept.len() == elems.len() {
            (arena.finite_set(elems), undecided)
        } else {
            (arena.finite_set(&kept), undecided)
        }
    };
    match arena.node(p).clone() {
        ExprNode::FiniteSet(elems) => cut(arena, &elems),
        ExprNode::SetUnion(children) => {
            let mut parts = Vec::with_capacity(children.len());
            let mut undecided = false;
            for c in children {
                if let ExprNode::FiniteSet(elems) = arena.node(c).clone() {
                    let (s, u) = cut(arena, &elems);
                    undecided |= u;
                    if s != arena.empty_set {
                        parts.push(s);
                    }
                } else {
                    parts.push(c);
                }
            }
            match parts.len() {
                0 => (arena.empty_set, false),
                1 => (parts[0], undecided),
                _ => (arena.set_union(&parts), undecided),
            }
        }
        _ => (p, false),
    }
}

/// Is the constant `id` certainly not real (a certified evaluation)?
fn certainly_nonreal(arena: &mut Arena, id: ExprId) -> bool {
    if !crate::base::walk::free_symbols(arena, id).is_empty() {
        return false;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    Truth::from(crate::transforms::realness::constant_realness_as_declared(
        arena, id, 20, &mut cache,
    ))
    .is_false()
}

fn is_known_real(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => true,
        ExprNode::Symbol(sid) => Truth::from(
            arena
                .symbol_assumptions(*sid)
                .query(crate::base::assumptions::Props::REAL),
        )
        .is_true(),
        _ => false,
    }
}

/// Structural emptiness for sets that could not be evaluated exactly.
///
/// Bottom-up over the set structure (explicit post-order, no recursion).
fn structural_is_empty(arena: &Arena, vals: &FxHashMap<ExprId, SetVal>, root: ExprId) -> Truth {
    let (order, _) = collect(arena, &[root]);
    let mut memo: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for &id in &order {
        // (`None`: no exact value, the structure decides)
        let known = match vals.get(&id) {
            Some(SetVal::Universe) => Some(Truth::False),
            Some(SetVal::Exact(rs)) => Some(Truth::from_bool(rs.is_empty())),
            Some(SetVal::Mixed(rs, _)) if !rs.is_empty() => Some(Truth::False),
            _ => None,
        };
        let r = if let Some(k) = known {
            k
        } else {
            match arena.node(id) {
                ExprNode::EmptySet => Truth::True,
                ExprNode::UniversalSet => Truth::False,
                // The image of the (non-empty) integers.
                ExprNode::ImageSet(..) => Truth::False,
                ExprNode::FiniteSet(elems) => Truth::from_bool(elems.is_empty()),
                ExprNode::SetUnion(children) => {
                    Truth::all(children.iter().map(|&c| verdict_of(&memo, c)))
                }
                ExprNode::SetIntersection(children) => {
                    // an empty operand proves it; operands that are not
                    // empty prove nothing about their meet
                    if children.iter().any(|&c| verdict_of(&memo, c).is_true()) {
                        Truth::True
                    } else {
                        Truth::Unknown
                    }
                }
                ExprNode::SetComplement(a, b) => {
                    match (verdict_of(&memo, *a), verdict_of(&memo, *b)) {
                        (Truth::True, _) => Truth::True,
                        (Truth::False, Truth::True) => Truth::False,
                        (Truth::False, Truth::False | Truth::Unknown) | (Truth::Unknown, _) => {
                            Truth::Unknown
                        }
                    }
                }
                _ => Truth::Unknown,
            }
        };
        memo.insert(id, r);
    }
    verdict_of(&memo, root)
}

/// Is the set empty?
pub(crate) fn is_empty(arena: &mut Arena, set: ExprId) -> Truth {
    let set = resolve_image_sets(arena, set);
    let ev = evaluate(arena, &[set], &[]);
    structural_is_empty(arena, &ev.vals, set)
}

/// Does the set hold the whole real line (the line itself, the line with
/// other members, or the universal set)?
pub(crate) fn is_full(arena: &mut Arena, set: ExprId) -> Truth {
    let set = resolve_image_sets(arena, set);
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set) {
        Some(SetVal::Universe) => Truth::True,
        Some(SetVal::Exact(rs)) => Truth::from_bool(rs.is_full()),
        Some(SetVal::Mixed(rs, _)) if rs.is_full() => Truth::True,
        _ => Truth::Unknown,
    }
}

/// Is the set known exactly: a set of real numbers in normal form (what
/// [`as_intervals`] lists) or the universal set?
pub(crate) fn is_exact(arena: &mut Arena, set: ExprId) -> bool {
    let set = resolve_image_sets(arena, set);
    let ev = evaluate(arena, &[set], &[]);
    matches!(ev.vals.get(&set), Some(SetVal::Exact(_) | SetVal::Universe))
}

/// Is `a ⊆ b`?
pub(crate) fn is_subset(arena: &mut Arena, a: ExprId, b: ExprId) -> Truth {
    if a == b {
        return Truth::True;
    }
    let ev = evaluate(arena, &[a, b], &[]);
    let (Some(va), Some(vb)) = (ev.vals.get(&a), ev.vals.get(&b)) else {
        return Truth::Unknown;
    };
    // ∅ ⊆ B ;  A ⊆ U
    if va.is_known_empty() || vb.is_universe() {
        return Truth::True;
    }
    if let (Some(ea), Some(eb)) = (va.exact(), vb.exact()) {
        return Truth::from_bool(ea.is_subset(eb));
    }
    // The real line holds every member of `a` that is proved real, and a
    // set of real numbers holds no member that is certainly not real.  Up
    // to 0.59 everything was a subset of a `b` that evaluated to the whole
    // line: `{i} ⊆ ℝ`, `{2nπi} ⊆ ℝ` and `{x} ⊆ ℝ` for a plain `x` were
    // `Some(true)` (fixed at 0.60 with the set verdicts in `Truth`).
    let holds_the_line = match vb {
        SetVal::Exact(rs) | SetVal::Mixed(rs, _) => rs.is_full(),
        SetVal::Sym(_) | SetVal::Universe => false,
    };
    let reals_only = vb.exact().is_some();
    if holds_the_line || reals_only {
        match members_real(arena, a) {
            Truth::True if holds_the_line => return Truth::True,
            Truth::False if reals_only => return Truth::False,
            Truth::True | Truth::False | Truth::Unknown => {}
        }
    }
    // The real numbers of `a` among those of `b`.
    if let (Some(ea), SetVal::Mixed(eb, _)) = (va.exact(), vb)
        && ea.is_subset(eb)
    {
        return Truth::True;
    }
    // {…} ⊆ {…} structurally (element identity).
    if let (ExprNode::FiniteSet(xa), ExprNode::FiniteSet(xb)) = (arena.node(a), arena.node(b))
        && xa.iter().all(|e| xb.contains(e))
    {
        return Truth::True;
    }
    // {…} ⊆ B with image sets in B: element by element.
    if let ExprNode::FiniteSet(xa) = arena.node(a).clone()
        && contains_image_set(arena, b)
    {
        // (every element is asked, as before: no short circuit)
        let mut acc = Truth::True;
        for e in xa {
            acc = acc.and(set_contains(arena, b, e));
        }
        return acc;
    }
    // Exact non-empty A ⊆ structural B: decide only if B is the empty set.
    if let Some(ea) = va.exact()
        && !ea.is_empty()
        && vb.is_known_empty()
    {
        return Truth::False;
    }
    Truth::Unknown
}

/// Is every member of the set expression `set` real?  `True`: every member
/// is proved real (intervals, real numbers and symbols, real families);
/// `False`: some member is certainly not real (`i` in a finite set, a
/// family with a non-real step or offset, the universal set); `Unknown`
/// otherwise.  Bottom-up over the set structure (no recursion).
///
/// The set-level form of [`member_realness`]: the real line absorbs,
/// removes or contains a whole part only when this is `True`.
/// [`all_members_real`] is the older two-valued test of the normal form (a
/// symbol declared real is not a real member there); it is left as it is.
fn members_real(arena: &mut Arena, set: ExprId) -> Truth {
    let (order, _) = collect(arena, &[set]);
    let mut memo: FxHashMap<ExprId, Truth> = FxHashMap::default();
    for &id in &order {
        let r = match arena.node(id).clone() {
            // an interval holds real numbers only
            ExprNode::EmptySet | ExprNode::Interval(..) => Truth::True,
            // it holds `i`
            ExprNode::UniversalSet => Truth::False,
            ExprNode::FiniteSet(elems) => {
                let mut acc = Truth::True;
                for &e in &elems {
                    acc = acc.and(element_realness(arena, e));
                }
                acc
            }
            // a step or an offset that is certainly not real gives a member
            // that is not (at index 0 or 1); otherwise a body that is real
            // for every index (`n²`, `eⁿ` for the integer `n`)
            ExprNode::ImageSet(v, b) => image_set_is_real(arena, v, b).or_else(|| {
                let mut cache = crate::base::assumptions::AssumptionCache::new();
                let real = cache.query(arena, b, crate::base::assumptions::Props::REAL);
                if Truth::from(real).is_true() {
                    Truth::True
                } else {
                    Truth::Unknown
                }
            }),
            ExprNode::SetUnion(ch) => Truth::all(ch.iter().map(|&c| verdict_of(&memo, c))),
            // a part of a real set is real; operands with members that are
            // not real may still meet in real ones only
            ExprNode::SetIntersection(ch) => {
                if ch.iter().any(|&c| verdict_of(&memo, c).is_true()) {
                    Truth::True
                } else {
                    Truth::Unknown
                }
            }
            ExprNode::SetComplement(a, _) => {
                if verdict_of(&memo, a).is_true() {
                    Truth::True
                } else {
                    Truth::Unknown
                }
            }
            // the values of the variable: real when it is declared real,
            // or when the condition puts it in an order relation
            ExprNode::ConditionSet(var, cond) => {
                if is_known_real(arena, var) || condition_orders_the_variable(arena, var, cond) {
                    Truth::True
                } else {
                    Truth::Unknown
                }
            }
            _ => Truth::Unknown,
        };
        memo.insert(id, r);
    }
    verdict_of(&memo, set)
}

/// Does the condition of `{var | cond}` hold for a real `var` only: an order
/// relation with `var` itself as a side, alone or in a conjunction?  (An
/// order relation holds only between real numbers: `{x | x > 1}` is a set
/// of real numbers whatever the assumptions of `x`.)
fn condition_orders_the_variable(arena: &Arena, var: ExprId, cond: ExprId) -> bool {
    let orders = |c: ExprId| match *arena.node(c) {
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) => a == var || b == var,
        _ => false,
    };
    match arena.node(cond) {
        ExprNode::And(ch) => ch.iter().any(|&c| orders(c)),
        _ => orders(cond),
    }
}

/// Is `a ∩ b = ∅`?
pub(crate) fn is_disjoint(arena: &mut Arena, a: ExprId, b: ExprId) -> Truth {
    if contains_image_set(arena, a) || contains_image_set(arena, b) {
        let both = arena.set_intersection(&[a, b]);
        return is_empty(arena, both);
    }
    let ev = evaluate(arena, &[a, b], &[]);
    let (Some(va), Some(vb)) = (ev.vals.get(&a), ev.vals.get(&b)) else {
        return Truth::Unknown;
    };
    if va.is_known_empty() || vb.is_known_empty() {
        return Truth::True;
    }
    // U ∩ B = B
    if va.is_universe() {
        return structural_is_empty(arena, &ev.vals, b);
    }
    if vb.is_universe() {
        return structural_is_empty(arena, &ev.vals, a);
    }
    if let (Some(ea), Some(eb)) = (va.exact(), vb.exact()) {
        return Truth::from_bool(ea.intersection(eb).is_empty());
    }
    // A real number of both (the real parts of sets with other members).
    if let (SetVal::Exact(ea) | SetVal::Mixed(ea, _), SetVal::Exact(eb) | SetVal::Mixed(eb, _)) =
        (va, vb)
        && !ea.intersection(eb).is_empty()
    {
        return Truth::False;
    }
    if a == b {
        // A ∩ A = A: disjoint from itself only if empty.
        return structural_is_empty(arena, &ev.vals, a);
    }
    Truth::Unknown
}

/// Greatest lower bound (`-∞` for sets unbounded below); `None` if the
/// set is empty or cannot be evaluated.
pub(crate) fn inf(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let set = resolve_image_sets(arena, set);
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    let p = rs.inf()?;
    Some(ev.table.rep(arena, p))
}

/// Least upper bound (`∞` for sets unbounded above); `None` if the set
/// is empty or cannot be evaluated.
pub(crate) fn sup(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let set = resolve_image_sets(arena, set);
    let ev = evaluate(arena, &[set], &[]);
    let rs = ev.vals.get(&set)?.exact()?;
    let p = rs.sup()?;
    Some(ev.table.rep(arena, p))
}

/// Lebesgue measure (total length) of the set; `∞` when unbounded.
pub(crate) fn measure(arena: &mut Arena, set: ExprId) -> Option<ExprId> {
    let set = resolve_image_sets(arena, set);
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
    let set = resolve_image_sets(arena, set);
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
pub(crate) fn is_open(arena: &mut Arena, set: ExprId) -> Truth {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set) {
        None => Truth::Unknown,
        // (open and closed in every topology)
        Some(SetVal::Universe) => Truth::True,
        Some(SetVal::Exact(rs)) => Truth::from_bool(rs.is_open()),
        Some(SetVal::Mixed(..) | SetVal::Sym(_)) => match arena.node(set) {
            // A non-empty finite set is never open.
            ExprNode::FiniteSet(e) if !e.is_empty() => Truth::False,
            _ => Truth::Unknown,
        },
    }
}

/// Is the set closed (in ℝ)?
pub(crate) fn is_closed(arena: &mut Arena, set: ExprId) -> Truth {
    let ev = evaluate(arena, &[set], &[]);
    match ev.vals.get(&set) {
        None => Truth::Unknown,
        Some(SetVal::Universe) => Truth::True,
        Some(SetVal::Exact(rs)) => Truth::from_bool(rs.is_closed()),
        Some(SetVal::Mixed(..) | SetVal::Sym(_)) => match arena.node(set) {
            ExprNode::FiniteSet(_) => Truth::True,
            _ => Truth::Unknown,
        },
    }
}

/// The normal form as intervals (isolated points appear as `[p, p]`), or
/// `None` if the set cannot be fully evaluated.
pub(crate) fn as_intervals(arena: &mut Arena, set: ExprId) -> Option<Vec<Interval<ExprId>>> {
    let set = resolve_image_sets(arena, set);
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
    let set = resolve_image_sets(arena, set);
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
        SetVal::Universe => None,
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
            ExprNode::Interval(a, b, kind) => {
                let mut conj: SmallVec<[ExprId; 2]> = SmallVec::new();
                if a != arena.neg_infinity {
                    conj.push(if kind.lower_open() {
                        arena.gt(var, a)
                    } else {
                        arena.ge(var, a)
                    });
                }
                if b != arena.infinity {
                    conj.push(if kind.upper_open() {
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
            // var ∈ ⋃ₙ (lo + s·n, hi + s·n)  ⇔  t = var − lo − s·⌊(var − lo)/s⌋,
            // the offset of var in its period (`0 ≤ t < s`), is below
            // hi − lo (and above 0 for an open lower end).
            ExprNode::ImageSet(n, body) if interval_family(arena, n, body).is_some() => {
                let Some(fam) = interval_family(arena, n, body) else {
                    continue;
                };
                let d = arena.sub(var, fam.lo);
                let u = arena.div(d, fam.step);
                let fl = arena.floor(u);
                let s_fl = arena.mul(&[fam.step, fl]);
                let t = arena.sub(d, s_fl);
                let width = arena.sub(fam.hi, fam.lo);
                let width = crate::transforms::eval::eval(arena, width);
                let upper = if fam.kind.upper_open() {
                    arena.gt(width, t)
                } else {
                    arena.ge(width, t)
                };
                if fam.kind.lower_open() {
                    let zero = arena.zero;
                    let lower = arena.gt(t, zero);
                    mk_and(arena, &[lower, upper])
                } else {
                    upper
                }
            }
            // var = a·n + b for an integer n  ⇔  sin(π·(var − b)/a) = 0 (the zeros of
            // sin(πt) are the integers t, in ℂ too).
            ExprNode::ImageSet(n, body) => match image_linear_parts(arena, n, body) {
                Some(Lattice { step: a, offset: b }) => {
                    let d = arena.sub(var, b);
                    let t = arena.div(d, a);
                    let pi = arena.pi;
                    let arg = arena.mul(&[pi, t]);
                    let s = arena.sin(arg);
                    let zero = arena.zero;
                    arena.eq_(s, zero)
                }
                None => {
                    return Err(SymplexError::InvalidArgument {
                        operation: "to_condition",
                        reason: format!(
                            "membership in the image set {} has no closed condition (the family is not linear in {})",
                            arena.display(id),
                            arena.display(n)
                        ),
                    });
                }
            },
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
// Image sets over the integers
// ═══════════════════════════════════════════════════════════════════════════

/// Most members of one family listed in a bounded window (also for the
/// periodic equations of `reduce_inequalities`).
const MAX_IMAGE_MEMBERS: usize = 10_000;

/// Largest common period, in units of the finest step, over which the
/// families of one class of a union are merged ([`merge_union_parts`]).
const MAX_MERGE_RESIDUES: i64 = 4096;

/// Distance from the nearest integer below which a parameter value is
/// checked exactly rather than declared a non-integer.
const NEAR_INTEGER: f64 = 1e-9;

/// The assumptions of the canonical bound variable of an image set.
fn integer_declared() -> crate::base::assumptions::Assumptions {
    use crate::base::assumptions::{Assumption, Assumptions};
    let mut declared = Assumptions::default().with(Assumption::Integer);
    declared.normalize_declared();
    declared
}

/// The canonical bound variable of `{body : var ∈ ℤ}`: the first of `_n`,
/// `_n_1`, … that is declared integer (created so when new) and is not free
/// in `body` but as `var` itself.  A user's `_n` with other assumptions is
/// skipped.
pub(crate) fn image_param(arena: &mut Arena, var: ExprId, body: ExprId) -> ExprId {
    let declared = integer_declared();
    let mut k = 0usize;
    loop {
        let name = if k == 0 {
            "_n".to_owned()
        } else {
            format!("_n_{k}")
        };
        k += 1;
        match arena.symbols.get(&name) {
            None => {
                let s = arena.symbol(&name);
                if let ExprNode::Symbol(sid) = *arena.node(s) {
                    arena.set_symbol_assumptions(sid, declared);
                }
                return s;
            }
            Some(sid) if arena.symbol_assumptions(sid) == declared => {
                let s = arena.intern(ExprNode::Symbol(sid));
                if s == var || !crate::base::walk::has_free_var(arena, body, s) {
                    return s;
                }
            }
            Some(_) => {}
        }
    }
}

/// Declare `name` integer when it is a canonical bound-variable name (`_n`,
/// `_n_1`, …) not yet in use, so that the printed form of an image set
/// parses back to itself in a new context.
pub(crate) fn predeclare_image_param(arena: &mut Arena, name: &str) {
    let canonical = name == "_n"
        || name
            .strip_prefix("_n_")
            .is_some_and(|k| !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()));
    if canonical && arena.symbols.get(name).is_none() {
        let s = arena.symbol(name);
        if let ExprNode::Symbol(sid) = *arena.node(s) {
            arena.set_symbol_assumptions(sid, integer_declared());
        }
    }
}

/// A family `{step·n + offset : n ∈ ℤ}` linear in its parameter.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lattice {
    /// The step `a` (nonzero).
    pub(crate) step: ExprId,
    /// The offset `b`.
    pub(crate) offset: ExprId,
}

/// `body = a·var + b` with `a ≠ 0` and `a`, `b` free of `var`: the step
/// and offset of a family linear in its parameter.
pub(crate) fn image_linear_parts(arena: &mut Arena, var: ExprId, body: ExprId) -> Option<Lattice> {
    use crate::base::walk;
    use crate::transforms::eval::eval;
    if !walk::has_free_var(arena, body, var) {
        return None;
    }
    let a = crate::transforms::diff::diff(arena, body, var);
    let a = eval(arena, a);
    if a == arena.zero || walk::has_free_var(arena, a, var) || walk::has_unevaluated(arena, a) {
        return None;
    }
    let zero = arena.zero;
    let b = crate::transforms::subs::subs(arena, body, var, zero);
    let b = eval(arena, b);
    if walk::has_free_var(arena, b, var) || walk::has_unevaluated(arena, b) {
        return None;
    }
    let av = arena.mul(&[a, var]);
    let lin = arena.add(&[av, b]);
    let d = arena.sub(body, lin);
    let d = crate::transforms::expand::expand(arena, d);
    let d = eval(arena, d);
    (d == arena.zero).then_some(Lattice { step: a, offset: b })
}

/// The canonical step and offset of `{a·n + b : n ∈ ℤ}`: `a` with a
/// positive numeric coefficient (`n ↦ −n`), and the terms of `b` that are
/// rational multiples of `a`, summed to `k·a`, reduced to `0 ≤ k < 1`.
fn reduce_lattice(arena: &mut Arena, a: ExprId, b: ExprId) -> Lattice {
    use crate::transforms::eval::eval;
    let (c, t) = arena.as_coeff_term(a);
    let a = if c.is_negative() {
        arena.make_coeff_term(-c, t)
    } else {
        a
    };
    let terms: Vec<ExprId> = match arena.node(b) {
        ExprNode::Add(ch) => ch.to_vec(),
        _ => vec![b],
    };
    let mut k = Q::zero();
    let mut rest: Vec<ExprId> = Vec::with_capacity(terms.len() + 1);
    for term in terms {
        let r = arena.div(term, a);
        let r = eval(arena, r);
        match arena.as_num(r) {
            Some(q) => k += q,
            None => rest.push(term),
        }
    }
    let k = &k - k.floor();
    if !k.is_zero() {
        rest.push(arena.make_coeff_term(k, a));
    }
    let b = if rest.is_empty() {
        arena.zero
    } else {
        arena.add(&rest)
    };
    Lattice {
        step: a,
        offset: eval(arena, b),
    }
}

/// A periodic family of intervals `⋃ₙ I(n)`, the image set whose body is
/// the interval `I(n) = ⟨lo + step·n, hi + step·n⟩` (both ends with the
/// same step): the solution set of a periodic inequality (`sin x > 1/2`:
/// `⋃ₙ (π/6 + 2nπ, 5π/6 + 2nπ)`, printed `ImageSet(Lambda(_n, (2*_n*pi +
/// 1/6*pi, 2*_n*pi + 5/6*pi)), Integers)`).  The members are the points of
/// the intervals; the constructions keep `lo < hi < lo + |step|`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntervalFamily {
    /// The lower end at `n = 0`.
    pub(crate) lo: ExprId,
    /// The upper end at `n = 0`.
    pub(crate) hi: ExprId,
    /// Which ends of the intervals belong to them.
    pub(crate) kind: IntervalKind,
    /// The period (positive).
    pub(crate) step: ExprId,
}

/// The [`IntervalFamily`] of the image set `{body : var ∈ ℤ}`, `None` when
/// `body` is not an interval with both ends linear in `var` with one step.
pub(crate) fn interval_family(
    arena: &mut Arena,
    var: ExprId,
    body: ExprId,
) -> Option<IntervalFamily> {
    let ExprNode::Interval(a, b, kind) = *arena.node(body) else {
        return None;
    };
    let la = image_linear_parts(arena, var, a)?;
    let lb = image_linear_parts(arena, var, b)?;
    let d = arena.sub(la.step, lb.step);
    let d = crate::transforms::eval::eval(arena, d);
    if d != arena.zero {
        return None;
    }
    Some(IntervalFamily {
        lo: la.offset,
        hi: lb.offset,
        kind,
        step: la.step,
    })
}

/// The periodic family `⋃ₙ ⟨lo + period·n, hi + period·n⟩` (an image set
/// over an interval, [`IntervalFamily`]); `lo < hi < lo + period` and a
/// positive `period` are the caller's.
pub(crate) fn periodic_interval_set(
    arena: &mut Arena,
    lo: ExprId,
    hi: ExprId,
    kind: IntervalKind,
    period: ExprId,
) -> ExprId {
    let scope = arena.finite_set(&[lo, hi, period]);
    let zero = arena.zero;
    let n = image_param(arena, zero, scope);
    let shift = arena.mul(&[period, n]);
    let a = arena.add(&[lo, shift]);
    let a = crate::transforms::eval::eval(arena, a);
    let b = arena.add(&[hi, shift]);
    let b = crate::transforms::eval::eval(arena, b);
    let body = arena.interval(a, b, kind);
    arena.intern(ExprNode::ImageSet(n, body))
}

/// Is `elem` in the [`IntervalFamily`] `fam`: in `I(k)` for the integer `k`
/// that its value places it in (or a neighbour), decided exactly by
/// [`cmp_exprs`]; `false` for an element that is certainly not real.
fn interval_family_contains(arena: &mut Arena, fam: &IntervalFamily, elem: ExprId) -> Truth {
    if certainly_nonreal(arena, elem) {
        return Truth::False;
    }
    let Some(x) = approx_real_f64(arena, elem) else {
        return Truth::Unknown;
    };
    let Some(lo) = approx_real_f64(arena, fam.lo) else {
        return Truth::Unknown;
    };
    let Some(step) = approx_real_f64(arena, fam.step) else {
        return Truth::Unknown;
    };
    if !x.is_finite() || !lo.is_finite() || !step.is_finite() || step <= 0.0 {
        return Truth::Unknown;
    }
    let k0 = ((x - lo) / step).floor();
    if !k0.is_finite() || k0.abs() > 1e12 {
        return Truth::Unknown;
    }
    // (all three candidates are asked, as before: no short circuit)
    let mut acc = Truth::False;
    for k in [k0 as i64 - 1, k0 as i64, k0 as i64 + 1] {
        let kk = arena.int(k);
        let shift = arena.mul(&[fam.step, kk]);
        let a = arena.add(&[fam.lo, shift]);
        let a = crate::transforms::eval::eval(arena, a);
        let b = arena.add(&[fam.hi, shift]);
        let b = crate::transforms::eval::eval(arena, b);
        let l_ok = inside_end(cmp_exprs(arena, a, elem), fam.kind.lower_open());
        let h_ok = inside_end(cmp_exprs(arena, elem, b), fam.kind.upper_open());
        acc = acc.or(l_ok.and(h_ok));
    }
    acc
}

/// The intervals `I(k) ∩ [lo, hi]` of the [`IntervalFamily`] `fam` that may
/// meet the window (with a margin of one interval at either end: the exact
/// ends are decided by the intersection they are cut to).  `None` beyond
/// [`MAX_IMAGE_MEMBERS`] or when the ends cannot be located.  The indices
/// are exact ([`integers_around`](crate::transforms::solve::integers_around));
/// up to 0.53 they were `f64` values cast with `as i64`, wrong far from
/// the family's origin.
fn interval_family_in(
    arena: &mut Arena,
    fam: &IntervalFamily,
    window: Window,
) -> Option<Vec<ExprId>> {
    let a0 = approx_real_f64(arena, fam.lo)?;
    let b0 = approx_real_f64(arena, fam.hi)?;
    let step = approx_real_f64(arena, fam.step)?;
    if !a0.is_finite() || !b0.is_finite() || !step.is_finite() || step <= 0.0 {
        return None;
    }
    if !window.is_bounded() {
        return None;
    }
    // I(k) meets the window for (lo − b₀)/step ≤ k ≤ (hi − a₀)/step
    let below = arena.sub(window.lo, fam.hi);
    let below = arena.div(below, fam.step);
    let above = arena.sub(window.hi, fam.lo);
    let above = arena.div(above, fam.step);
    let indices =
        crate::transforms::solve::integers_around(arena, [below, above], MAX_IMAGE_MEMBERS)?;
    let mut out = Vec::new();
    for k in indices {
        let kk = arena.big_int(k);
        let shift = arena.mul(&[fam.step, kk]);
        let a = arena.add(&[fam.lo, shift]);
        let a = crate::transforms::eval::eval(arena, a);
        let b = arena.add(&[fam.hi, shift]);
        let b = crate::transforms::eval::eval(arena, b);
        out.push(arena.interval(a, b, fam.kind));
    }
    Some(out)
}

/// The canonical image set `{body : var ∈ ℤ}` (see [Image sets](#image-sets)):
/// `{body}` when `body` does not depend on `var`.  A `var` that is not a
/// symbol is malformed and interned as it is.
pub(crate) fn image_set(arena: &mut Arena, var: ExprId, body: ExprId) -> ExprId {
    use crate::transforms::eval::eval;
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return arena.intern(ExprNode::ImageSet(var, body));
    }
    // A family of intervals: its ends in the canonical parameter.
    if let ExprNode::Interval(a, b, kind) = *arena.node(body) {
        if !crate::base::walk::has_free_var(arena, body, var) {
            return body;
        }
        let n = image_param(arena, var, body);
        let (a, b) = if n == var {
            (a, b)
        } else {
            (
                crate::transforms::subs::subs(arena, a, var, n),
                crate::transforms::subs::subs(arena, b, var, n),
            )
        };
        let (a, b) = (eval(arena, a), eval(arena, b));
        let body = arena.interval(a, b, kind);
        return arena.intern(ExprNode::ImageSet(n, body));
    }
    let body = eval(arena, body);
    if !crate::base::walk::has_free_var(arena, body, var) {
        return arena.finite_set(&[body]);
    }
    let n = image_param(arena, var, body);
    let body = if n == var {
        body
    } else {
        let b = crate::transforms::subs::subs(arena, body, var, n);
        eval(arena, b)
    };
    let body = match image_linear_parts(arena, n, body) {
        Some(Lattice { step, offset }) => {
            let Lattice { step: a, offset: b } = reduce_lattice(arena, step, offset);
            let an = arena.mul(&[a, n]);
            let s = arena.add(&[an, b]);
            eval(arena, s)
        }
        None => body,
    };
    arena.intern(ExprNode::ImageSet(n, body))
}

/// Does the set expression `set` contain an `ImageSet` node?
pub(crate) fn contains_image_set(arena: &Arena, set: ExprId) -> bool {
    let mut stack = vec![set];
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::ImageSet(..) => return true,
            ExprNode::SetUnion(ch) | ExprNode::SetIntersection(ch) => stack.extend(ch.iter()),
            ExprNode::SetComplement(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            _ => {}
        }
    }
    false
}

/// Is `elem` = `body` at `var = k` (an integer node)?  Exactly, or by a
/// certified digit of the difference; `Unknown` when that cannot be
/// decided.
fn member_at(arena: &mut Arena, var: ExprId, body: ExprId, elem: ExprId, k: ExprId) -> Truth {
    let m = crate::transforms::subs::subs(arena, body, var, k);
    let m = crate::transforms::eval::eval(arena, m);
    same_value(arena, elem, m)
}

/// Are the expressions `p` and `q` equal?  Exactly (structurally after
/// evaluation), by a certified digit of `p − q`, or by an identity when
/// the difference is zero to the digits reached (`√(π²) = π`); `Unknown`
/// when undecided.
fn same_value(arena: &mut Arena, p: ExprId, q: ExprId) -> Truth {
    use crate::transforms::eval::eval;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    if p == q {
        return Truth::True;
    }
    let d = arena.sub(p, q);
    let d = eval(arena, d);
    if d == arena.zero {
        return Truth::True;
    }
    if !crate::base::walk::free_symbols(arena, d).is_empty()
        || crate::base::walk::has_unevaluated(arena, d)
    {
        return Truth::Unknown;
    }
    match crate::transforms::evalf::evalf_settled(arena, d, 20, ZeroSearch::Deep) {
        Ok((z, Settled::Certified)) if !(z.0.is_zero() && z.1.is_zero()) => Truth::False,
        // Zero to the digits reached: exactly 0 by an identity (`√(π²) − π`)?
        // (An identity proves equality; none found proves nothing.)
        Ok(_) => {
            if arena.vanishes_by_identity(d) {
                Truth::True
            } else {
                Truth::Unknown
            }
        }
        Err(_) => Truth::Unknown,
    }
}

/// Is the parameter value `t` an integer?  Exactly for a rational, by the
/// assumptions for an expression with free symbols, and for a constant by
/// a certified approximation: farther than [`NEAR_INTEGER`] from every
/// integer (or off the real line) it is not one; otherwise `at(k)` decides
/// at the nearest integer `k`.
fn integer_test(
    arena: &mut Arena,
    t: ExprId,
    at: impl FnOnce(&mut Arena, ExprId) -> Truth,
) -> Truth {
    use crate::base::walk;
    if let Some(q) = arena.as_num(t) {
        return Truth::from_bool(q.is_integer());
    }
    if matches!(
        arena.node(t),
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
    ) || walk::has_unevaluated(arena, t)
    {
        return Truth::Unknown;
    }
    if !walk::free_symbols(arena, t).is_empty() {
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        return Truth::from(cache.query(arena, t, crate::base::assumptions::Props::INTEGER));
    }
    let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, t) else {
        return Truth::Unknown;
    };
    if !z.re.is_finite() || !z.im.is_finite() {
        return Truth::Unknown;
    }
    let scale = 1.0 + z.re.abs();
    if z.im.abs() > NEAR_INTEGER * scale {
        return Truth::False;
    }
    if z.re.abs() > 1e12 {
        return Truth::Unknown;
    }
    let k = z.re.round();
    if (z.re - k).abs() > NEAR_INTEGER * scale {
        return Truth::False;
    }
    let k = arena.int(k as i64);
    at(arena, k)
}

/// Is `elem` in `{body : var ∈ ℤ}`?  For a family linear in `var`, `elem =
/// a·var + b` at `var = (elem − b)/a`, which must be an integer
/// ([`integer_test`]).  Otherwise `body = elem` is solved for `var`: a member
/// when a solution is an integer; not one when the equation is polynomial
/// in `var` (so the solutions are all of them) and none is.  `Unknown` when
/// undecided.
pub(crate) fn image_set_contains(
    arena: &mut Arena,
    var: ExprId,
    body: ExprId,
    elem: ExprId,
) -> Truth {
    use crate::transforms::eval::eval;
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Truth::Unknown;
    }
    if matches!(
        arena.node(elem),
        ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
    ) {
        return Truth::False;
    }
    if crate::base::walk::has_unevaluated(arena, elem) || arena.node(elem).is_set_node() {
        return Truth::Unknown;
    }
    if arena.node(body).is_set_node() {
        return match interval_family(arena, var, body) {
            Some(fam) => interval_family_contains(arena, &fam, elem),
            None => Truth::Unknown,
        };
    }
    if !crate::base::walk::has_free_var(arena, body, var) {
        let zero = arena.zero;
        return member_at(arena, var, body, elem, zero);
    }
    if let Some(Lattice { step: a, offset: b }) = image_linear_parts(arena, var, body) {
        let d = arena.sub(elem, b);
        let t = arena.div(d, a);
        let t = eval(arena, t);
        return integer_test(arena, t, |arena, k| member_at(arena, var, body, elem, k));
    }
    use crate::transforms::solve::{RootSet, Solutions};
    let eq = arena.sub(body, elem);
    let eq = eval(arena, eq);
    let outcome = crate::transforms::solve::solve_classified(arena, eq, var);
    match outcome {
        Solutions::Complete(RootSet::All) => Truth::True,
        Solutions::Complete(RootSet::Empty(_)) => Truth::False,
        Solutions::Refused(_) => Truth::Unknown,
        found => {
            // (Principal roots: every root only for a polynomial equation.)
            let complete = matches!(found, Solutions::Complete(_))
                && crate::poly::polybridge::expr_to_poly(arena, eq, var).is_some();
            let mut undecided = !complete;
            for s in found.claimed_points() {
                match integer_test(arena, s, |arena, k| member_at(arena, var, body, elem, k)) {
                    Truth::True => return Truth::True,
                    Truth::False => {}
                    Truth::Unknown => undecided = true,
                }
            }
            if undecided {
                Truth::Unknown
            } else {
                Truth::False
            }
        }
    }
}

/// The member of `set` (a union of finite sets and image sets) that equals
/// `p`, written as `set` writes it: `π` for `√(π²)` in `{n·π}`.  `None`
/// when `p` is not found to be a member.
pub(crate) fn member_form(arena: &mut Arena, set: ExprId, p: ExprId) -> Option<ExprId> {
    use crate::transforms::eval::eval;
    let parts: Vec<ExprId> = match arena.node(set) {
        ExprNode::SetUnion(ch) => ch.to_vec(),
        _ => vec![set],
    };
    for part in parts {
        match arena.node(part).clone() {
            ExprNode::FiniteSet(elems) => {
                for e in elems {
                    if same_value(arena, e, p).is_true() {
                        return Some(e);
                    }
                }
            }
            ExprNode::ImageSet(var, body) => {
                let Some(Lattice { step: a, offset: b }) = image_linear_parts(arena, var, body)
                else {
                    continue;
                };
                let d = arena.sub(p, b);
                let t = arena.div(d, a);
                let t = eval(arena, t);
                let found = std::cell::Cell::new(None);
                let member = integer_test(arena, t, |arena, k| {
                    let r = member_at(arena, var, body, p, k);
                    found.set(Some(k));
                    r
                });
                if member.is_true() {
                    let k = found.get().unwrap_or(t);
                    let m = crate::transforms::subs::subs(arena, body, var, k);
                    return Some(eval(arena, m));
                }
            }
            _ => {}
        }
    }
    None
}

/// A window `[lo, hi]` of the real line (`lo`, `hi` real constants or
/// `±∞`): the exact ends, which give the index range of a family
/// ([`lattice_indices`](crate::transforms::solve::lattice_indices)), and
/// their 16-digit values, which only drop the members far outside it.
#[derive(Clone, Copy)]
pub(crate) struct Window {
    lo: ExprId,
    hi: ExprId,
    lo_f: f64,
    hi_f: f64,
}

impl Window {
    /// The window between the exact ends `lo` and `hi`; `None` when one
    /// has no 16-digit value.
    pub(crate) fn new(arena: &mut Arena, lo: ExprId, hi: ExprId) -> Option<Window> {
        Some(Window {
            lo,
            hi,
            lo_f: approx_real_f64(arena, lo)?,
            hi_f: approx_real_f64(arena, hi)?,
        })
    }

    /// The whole real line.
    fn line(arena: &Arena) -> Window {
        Window {
            lo: arena.neg_infinity,
            hi: arena.infinity,
            lo_f: f64::NEG_INFINITY,
            hi_f: f64::INFINITY,
        }
    }

    fn is_bounded(&self) -> bool {
        self.lo_f.is_finite() && self.hi_f.is_finite()
    }

    /// Is the value `x` (16 digits) possibly in the window?  A guide with a
    /// margin far above the error of `x`: the exact ends are decided by the
    /// intersection the members are cut to.
    fn may_contain(&self, x: f64) -> bool {
        let margin = 1e-9 * (1.0 + self.lo_f.abs().max(self.hi_f.abs()));
        x >= self.lo_f - margin && x <= self.hi_f + margin
    }
}

/// The real members of the family `{body : var ∈ ℤ}` in the window (with
/// a margin: the exact ends are decided by the intersection they are cut
/// to).  `None` when the family is not linear, cannot be located, or has
/// more than [`MAX_IMAGE_MEMBERS`] members there.  A family with a
/// non-real step has at most one real member; one with a real step and a
/// non-real offset none.  The index range of a real family is exact
/// ([`lattice_indices`](crate::transforms::solve::lattice_indices)): up to
/// 0.53 it was computed in `f64` and cast with `as i64`, and
/// `solve_as_set(sin(x − 10²⁰)) ∩ [0, 10]` (three members, `n ≈
/// −3.18·10¹⁹`) was empty.
pub(crate) fn image_set_points_in(
    arena: &mut Arena,
    var: ExprId,
    body: ExprId,
    window: Window,
) -> Option<Vec<ExprId>> {
    use crate::transforms::eval::eval;
    // a family with an infinite term (`atan(i) + nπ = i∞ + nπ`) has no
    // finite member
    if crate::base::walk::post_order_ids(arena, body)
        .into_iter()
        .any(|id| {
            matches!(
                arena.node(id),
                ExprNode::Infinity
                    | ExprNode::NegInfinity
                    | ExprNode::ComplexInfinity
                    | ExprNode::NaN
            )
        })
    {
        return Some(Vec::new());
    }
    let Some(Lattice { step: a, offset: b }) = image_linear_parts(arena, var, body) else {
        return monotone_members_in(arena, var, body, window);
    };
    let member = |arena: &mut Arena, k: i64| -> ExprId {
        let kk = arena.int(k);
        let m = crate::transforms::subs::subs(arena, body, var, kk);
        eval(arena, m)
    };
    let real_family = image_set_is_real(arena, var, body).is_true();
    let (a_re, a_im) = crate::base::complex::as_real_imag(arena, a);
    let a_im = eval(arena, a_im);
    let (b_re, b_im) = crate::base::complex::as_real_imag(arena, b);
    let b_im = eval(arena, b_im);
    let (a_re, b_re, a_im, b_im) = if real_family {
        (a, b, arena.zero, arena.zero)
    } else {
        (a_re, b_re, a_im, b_im)
    };
    if a_im != arena.zero {
        // Im(a·k + b) = 0 at k = −Im b / Im a.
        let (Some(ai), Some(bi)) = (approx_real_f64(arena, a_im), approx_real_f64(arena, b_im))
        else {
            return None;
        };
        // (a part 0 to its digits but not structurally: refused)
        if ai == 0.0 || !ai.is_finite() || !bi.is_finite() {
            return None;
        }
        let kf = -bi / ai;
        if kf.abs() > 1e12 {
            return None;
        }
        let k = kf.round();
        // certified 17-digit parts: farther than the margin from an integer,
        // the index is not one (no real member); nearer, the member decides
        if (kf - k).abs() > NEAR_INTEGER * (1.0 + kf.abs()) {
            return Some(Vec::new());
        }
        let m = member(arena, k as i64);
        let (m_re, m_im) = crate::base::complex::as_real_imag(arena, m);
        let m_im = eval(arena, m_im);
        if m_im != arena.zero {
            return match is_nonzero_constant(arena, m_im) {
                Truth::True => Some(Vec::new()),
                // an imaginary part that is 0 without being written 0, or
                // an undecided one: refused
                Truth::False | Truth::Unknown => None,
            };
        }
        let m_re = eval(arena, m_re);
        let x = approx_real_f64(arena, m_re)?;
        return Some(if window.may_contain(x) {
            vec![m_re]
        } else {
            Vec::new()
        });
    }
    if b_im != arena.zero {
        return match is_nonzero_constant(arena, b_im) {
            Truth::True => Some(Vec::new()),
            // as above: refused
            Truth::False | Truth::Unknown => None,
        };
    }
    if !window.is_bounded() {
        return None;
    }
    let a_re = eval(arena, a_re);
    let b_re = eval(arena, b_re);
    let indices = crate::transforms::solve::lattice_indices(
        arena,
        b_re,
        a_re,
        window.lo,
        window.hi,
        MAX_IMAGE_MEMBERS,
    )?;
    let mut points = Vec::new();
    for k in indices {
        let kk = arena.big_int(k);
        let m = crate::transforms::subs::subs(arena, body, var, kk);
        let m = eval(arena, m);
        let x = approx_real_f64(arena, m)?;
        if window.may_contain(x) {
            points.push(m);
        }
    }
    Some(points)
}

/// One step `t ↦ s(t)` of a parametrisation `body = sₘ(…s₁(u)…)` of a
/// family over a linear `u = a·n + b` ([`monotone_members_in`]).
#[derive(Clone, Debug)]
enum Step {
    /// `c·t + d` for real constants `c ≠ 0`, `d`.
    Affine { c: ExprId, d: ExprId },
    /// `tᵏ` for a nonzero integer `k`.
    IntPow(i64),
    /// The principal `t^q` for a rational `q` that is not an integer: real
    /// only for `t ≥ 0` (`arg(t^q) = q·π` for `t < 0`).
    RatPow(Q),
    /// `eᵗ`.
    Exp,
    /// The principal `ln t`: real only for `t > 0`.
    Ln,
}

impl Step {
    /// Is `s(t)` real for every real `t` where it is defined?
    fn real_on_the_line(&self) -> bool {
        !matches!(self, Step::RatPow(_) | Step::Ln)
    }

    /// Is `s(z)` non-real for every non-real `z` (where defined)?  So that a
    /// value left non-real by an inner step never becomes real again:
    /// `ln`, `1/z`, `z^q` for `|q| < 1` and real affine maps; not `z²`
    /// (`(√−1)² = −1`), not `eᶻ` (`e^{iπ} = −1`).
    fn keeps_non_real(&self) -> bool {
        match self {
            Step::Affine { .. } | Step::Ln => true,
            Step::IntPow(k) => *k == -1,
            Step::RatPow(q) => q.abs() < Q::from_integer(1.into()),
            Step::Exp => false,
        }
    }
}

/// Most steps of a parametrisation [`monotone_members_in`] follows.
const MAX_PARAMETRISATION_STEPS: usize = 8;

/// The sign of the real constant `e`, one of `NEGATIVE`, `ZERO`, `POSITIVE`
/// (the end of an interval `±∞` counts as the side of 0 it lies on): exact
/// for a number, else from a certified value whose imaginary part is
/// exactly 0
/// ([`imaginary_part_sign`](crate::transforms::realness::imaginary_part_sign));
/// `None` when it is not decided.
fn decided_sign(arena: &mut Arena, e: ExprId) -> Option<crate::base::truth::SignSet> {
    use crate::base::truth::SignSet;
    use crate::transforms::evalf::{Settled, ZeroSearch};
    if e == arena.infinity {
        return Some(SignSet::POSITIVE);
    }
    if e == arena.neg_infinity {
        return Some(SignSet::NEGATIVE);
    }
    if let Some(r) = arena.as_num(e) {
        return Some(SignSet::from_ordering(r.cmp(&Q::zero())));
    }
    if !crate::base::walk::free_symbols(arena, e).is_empty()
        || crate::transforms::realness::imaginary_part_sign(arena, e) != SignSet::ZERO
    {
        return None;
    }
    match crate::transforms::evalf::evalf_settled(arena, e, 20, ZeroSearch::Cap) {
        Ok((z, Settled::Certified)) => Some(if z.0.is_zero() {
            SignSet::ZERO
        } else if crate::base::bigfloat::is_strictly_positive(&z.0) {
            SignSet::POSITIVE
        } else {
            SignSet::NEGATIVE
        }),
        _ => None,
    }
}

/// `body = sₘ(…s₁(u)…)` over a linear `u = a·var + b` (the steps
/// outermost first, see [`Step`]), each step's constants real.  `None`
/// when `body` is not such a chain.
fn parametrisation(arena: &mut Arena, var: ExprId, body: ExprId) -> Option<(Vec<Step>, Lattice)> {
    use crate::base::walk::has_free_var;
    use crate::transforms::eval::eval;
    let real = |arena: &mut Arena, c: ExprId| {
        crate::transforms::realness::imaginary_part_sign(arena, c)
            == crate::base::truth::SignSet::ZERO
    };
    let mut steps = Vec::new();
    let mut cur = body;
    loop {
        if let Some(lattice) = image_linear_parts(arena, var, cur) {
            return Some((steps, lattice));
        }
        if steps.len() >= MAX_PARAMETRISATION_STEPS {
            return None;
        }
        let step = match arena.node(cur).clone() {
            ExprNode::Add(ch) | ExprNode::Mul(ch) => {
                let is_add = matches!(arena.node(cur), ExprNode::Add(_));
                let (inner, outer): (Vec<ExprId>, Vec<ExprId>) =
                    ch.iter().partition(|&&t| has_free_var(arena, t, var));
                let [inner] = inner[..] else {
                    return None;
                };
                cur = inner;
                let rest = if is_add {
                    arena.add(&outer)
                } else {
                    arena.mul(&outer)
                };
                let rest = eval(arena, rest);
                if !real(arena, rest) {
                    return None;
                }
                if is_add {
                    Step::Affine {
                        c: arena.one,
                        d: rest,
                    }
                } else {
                    match decided_sign(arena, rest) {
                        Some(s) if s != crate::base::truth::SignSet::ZERO => Step::Affine {
                            c: rest,
                            d: arena.zero,
                        },
                        _ => return None,
                    }
                }
            }
            ExprNode::Pow(base, e) => {
                let q = arena.as_num(e)?.clone();
                cur = base;
                if q.is_integer() {
                    Step::IntPow(q.to_integer().to_i64()?)
                } else {
                    Step::RatPow(q)
                }
            }
            ExprNode::Exp(arg) => {
                cur = arg;
                Step::Exp
            }
            ExprNode::Ln(arg) => {
                cur = arg;
                Step::Ln
            }
            _ => return None,
        };
        steps.push(step);
    }
}

/// The preimage of the closed interval `[lo, hi]` (ends real constants or
/// `±∞`, `lo ≤ hi`) under `step` within the real `t` where `step(t)` is
/// real, as closed intervals (a superset at the ends is harmless: the
/// members found are cut to the window exactly).  `None` when an end's sign
/// is not decided.
fn step_preimage(
    arena: &mut Arena,
    step: &Step,
    lo: ExprId,
    hi: ExprId,
) -> Option<Vec<(ExprId, ExprId)>> {
    use crate::transforms::eval::eval;
    const NEG: crate::base::truth::SignSet = crate::base::truth::SignSet::NEGATIVE;
    const POS: crate::base::truth::SignSet = crate::base::truth::SignSet::POSITIVE;
    let (zero, inf, ninf) = (arena.zero, arena.infinity, arena.neg_infinity);
    // `e^(1/k)` of an end `e ≥ 0` (`∞` stays)
    let root = |arena: &mut Arena, e: ExprId, k: &Q| -> ExprId {
        if e == inf {
            return inf;
        }
        let r = arena.num_ratio(k.recip());
        let p = arena.pow(e, r);
        eval(arena, p)
    };
    let positive_part = |arena: &mut Arena| -> Option<Option<(ExprId, ExprId)>> {
        if decided_sign(arena, hi)? == NEG {
            return Some(None);
        }
        let lo = if decided_sign(arena, lo)? == NEG {
            zero
        } else {
            lo
        };
        Some(Some((lo, hi)))
    };
    Some(match step.clone() {
        Step::Affine { c, d } => {
            let map = |arena: &mut Arena, e: ExprId| -> ExprId {
                if e == inf || e == ninf {
                    return e;
                }
                let s = arena.sub(e, d);
                let q = arena.div(s, c);
                eval(arena, q)
            };
            let (a, b) = (map(arena, lo), map(arena, hi));
            match decided_sign(arena, c)? {
                s if s == POS => vec![(a, b)],
                _ => {
                    let flip = |e: ExprId| match e {
                        e if e == inf => ninf,
                        e if e == ninf => inf,
                        e => e,
                    };
                    vec![(flip(b), flip(a))]
                }
            }
        }
        Step::IntPow(k) if k > 0 && k % 2 == 0 => {
            let Some((a, b)) = positive_part(arena)? else {
                return Some(Vec::new());
            };
            let k = Q::from_integer(k.into());
            let (ra, rb) = (root(arena, a, &k), root(arena, b, &k));
            let neg = |arena: &mut Arena, e: ExprId| -> ExprId {
                if e == inf {
                    ninf
                } else {
                    let n = arena.neg(e);
                    eval(arena, n)
                }
            };
            let (na, nb) = (neg(arena, ra), neg(arena, rb));
            vec![(nb, na), (ra, rb)]
        }
        Step::IntPow(k) if k > 0 => {
            // odd: the real root, increasing on the line
            let k = Q::from_integer(k.into());
            let real_root = |arena: &mut Arena, e: ExprId| -> Option<ExprId> {
                if e == ninf {
                    return Some(ninf);
                }
                if decided_sign(arena, e)? == NEG {
                    let m = arena.neg(e);
                    let m = eval(arena, m);
                    let r = root(arena, m, &k);
                    let r = arena.neg(r);
                    Some(eval(arena, r))
                } else {
                    Some(root(arena, e, &k))
                }
            };
            vec![(real_root(arena, lo)?, real_root(arena, hi)?)]
        }
        Step::IntPow(k) => {
            // `tᵏ = 1/t^|k|`: the reciprocal, then the positive power
            let recip = |arena: &mut Arena, e: ExprId| -> ExprId {
                if e == inf || e == ninf {
                    return zero;
                }
                let one = arena.one;
                let r = arena.div(one, e);
                eval(arena, r)
            };
            let (slo, shi) = (decided_sign(arena, lo)?, decided_sign(arena, hi)?);
            let mut pieces = Vec::new();
            if shi == POS {
                let upper = if slo == POS { recip(arena, lo) } else { inf };
                pieces.push((recip(arena, hi), upper));
            }
            if slo == NEG {
                let lower = if shi == NEG { recip(arena, hi) } else { ninf };
                pieces.push((lower, recip(arena, lo)));
            }
            let mut out = Vec::new();
            for (a, b) in pieces {
                out.extend(step_preimage(arena, &Step::IntPow(-k), a, b)?);
            }
            out
        }
        Step::RatPow(q) => {
            let Some((a, b)) = positive_part(arena)? else {
                return Some(Vec::new());
            };
            if q.is_positive() {
                vec![(root(arena, a, &q), root(arena, b, &q))]
            } else {
                // decreasing on (0, ∞); `0^(1/q) = ∞`, `∞^(1/q) = 0`
                let upper = if decided_sign(arena, a)? == crate::base::truth::SignSet::ZERO {
                    inf
                } else {
                    root(arena, a, &q)
                };
                let lower = if b == inf { zero } else { root(arena, b, &q) };
                vec![(lower, upper)]
            }
        }
        Step::Exp => {
            if decided_sign(arena, hi)? != POS {
                return Some(Vec::new());
            }
            let ln = |arena: &mut Arena, e: ExprId| -> ExprId {
                if e == inf {
                    return inf;
                }
                let l = arena.ln(e);
                eval(arena, l)
            };
            let a = if decided_sign(arena, lo)? == POS {
                ln(arena, lo)
            } else {
                ninf
            };
            vec![(a, ln(arena, hi))]
        }
        Step::Ln => {
            let exp = |arena: &mut Arena, e: ExprId| -> ExprId {
                if e == inf {
                    return inf;
                }
                if e == ninf {
                    return zero;
                }
                let x = arena.exp(e);
                eval(arena, x)
            };
            vec![(exp(arena, lo), exp(arena, hi))]
        }
    })
}

/// The real members in the window of a family `{body : var ∈ ℤ}` that is
/// not linear in `var` but a chain of monotone pieces over a linear
/// `u = a·n + b` ([`parametrisation`]: real affine maps, integer and
/// rational powers, `exp`, `ln`): the window is pulled back step by step
/// (each step's inverse on the part of the line where it is real and
/// monotone, `t²` in two pieces), and the members are those of `u` in the
/// pulled-back intervals, from the exact index range of each
/// ([`lattice_indices`](crate::transforms::solve::lattice_indices)).  `u` is
/// real, or `u = i·v` with `v` real under an even power (`(2nπi + πi)² =
/// −(2nπ + π)²`).  A step that is real only on part of the line (`ln`, a
/// non-integer power) may be followed only by steps that keep non-real
/// values non-real, so that no real member comes from outside the
/// pulled-back intervals.  Each member's realness is decided
/// ([`imaginary_part_sign`](crate::transforms::realness::imaginary_part_sign)).
/// `None` when the family is not such a chain, a sign is not decided, or
/// the index range is unbounded or longer than [`MAX_IMAGE_MEMBERS`].
///
/// Up to 0.53 such families were located by a linear extrapolation from two
/// members in `f64` (`reduce_inequalities([sin(x²) = 1/2, −7 < x < 7])`
/// listed 24 of its 32 roots); 0.54 left them as the `ConditionSet` (`e^{√x}
/// = −1`, `cosh √x = 1/2`, `sin(x²) = 1/2` in `(−7, 7)`).
fn monotone_members_in(
    arena: &mut Arena,
    var: ExprId,
    body: ExprId,
    window: Window,
) -> Option<Vec<ExprId>> {
    use crate::base::truth::SignSet;
    use crate::transforms::eval::eval;
    if !window.is_bounded() {
        return None;
    }
    let (mut steps, Lattice { step: a, offset: b }) = parametrisation(arena, var, body)?;
    let real = |arena: &mut Arena, c: ExprId| {
        crate::transforms::realness::imaginary_part_sign(arena, c) == SignSet::ZERO
    };
    let (a, b) = if real(arena, a) && real(arena, b) {
        (a, b)
    } else {
        // `u = i·v`, `v = −i·u` real, under an even power `uᵏ = (−1)^(k/2)·vᵏ`
        let i = arena.i_unit;
        let mi = arena.neg(i);
        let va = arena.mul(&[mi, a]);
        let va = eval(arena, va);
        let vb = arena.mul(&[mi, b]);
        let vb = eval(arena, vb);
        let Some(&Step::IntPow(k)) = steps.last() else {
            return None;
        };
        if k % 2 != 0 || !real(arena, va) || !real(arena, vb) {
            return None;
        }
        let sign = if k % 4 == 0 { arena.one } else { arena.neg_one };
        let last = steps.len() - 1;
        steps.insert(
            last,
            Step::Affine {
                c: sign,
                d: arena.zero,
            },
        );
        (va, vb)
    };
    // innermost first: once a step is real only on part of the line, every
    // later one must keep non-real values non-real
    let mut partial = false;
    for s in steps.iter().rev() {
        if partial && !s.keeps_non_real() {
            return None;
        }
        partial |= !s.real_on_the_line();
    }
    let mut pieces = vec![(window.lo, window.hi)];
    for s in &steps {
        let mut next = Vec::new();
        for (lo, hi) in pieces {
            next.extend(step_preimage(arena, s, lo, hi)?);
        }
        pieces = next;
    }
    let mut indices: Vec<num_bigint::BigInt> = Vec::new();
    for (lo, hi) in pieces {
        for k in crate::transforms::solve::lattice_indices(arena, b, a, lo, hi, MAX_IMAGE_MEMBERS)?
        {
            if !indices.contains(&k) {
                indices.push(k);
            }
        }
        if indices.len() > MAX_IMAGE_MEMBERS {
            return None;
        }
    }
    indices.sort();
    let mut points = Vec::new();
    for k in indices {
        let kk = arena.big_int(k);
        let m = crate::transforms::subs::subs(arena, body, var, kk);
        let m = eval(arena, m);
        let m = even_powers_normalised(arena, m);
        // (a pole of the family, `1/(2nπ)` at `n = 0`, is no member)
        if matches!(
            arena.node(m),
            ExprNode::Infinity | ExprNode::NegInfinity | ExprNode::ComplexInfinity | ExprNode::NaN
        ) {
            continue;
        }
        match crate::transforms::realness::imaginary_part_sign(arena, m) {
            SignSet::ZERO => {}
            s if s.is_nonzero().is_true() => continue,
            _ => return None,
        }
        let x = approx_real_f64(arena, m)?;
        if window.may_contain(x) && !points.contains(&m) {
            points.push(m);
        }
    }
    Some(points)
}

/// `m` with each even power `B²ʲ` written over whichever of `B`, `−B` comes
/// first in the canonical order, so that one value has one form: the
/// members `(2π − acos(1/3))²` and `(acos(1/3) − 2π)²` of the two families
/// of `cos √x = 1/3` are one point, which the sets of members must not hold
/// twice (two forms of one value cannot be ordered, and the window's set
/// stayed an unevaluated intersection).
fn even_powers_normalised(arena: &mut Arena, m: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, m);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut changed = false;
    for &id in &post {
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match *arena.node(rebuilt) {
            ExprNode::Pow(b, e)
                if arena.as_num(e).is_some_and(|q| {
                    q.is_integer() && num_integer::Integer::is_even(&q.to_integer())
                }) =>
            {
                let nb = arena.neg(b);
                let nb = crate::transforms::eval::eval(arena, nb);
                if arena.canonical_cmp(nb, b) == Ordering::Less {
                    changed = true;
                    arena.pow(nb, e)
                } else {
                    rebuilt
                }
            }
            _ => rebuilt,
        };
        if new != id {
            cache.insert(id, new);
        }
    }
    if !changed {
        return m;
    }
    let out = cache.get(&m).copied().unwrap_or(m);
    crate::transforms::eval::eval(arena, out)
}

/// Is the constant `e` certainly nonzero (`True`), certainly zero
/// (`False`), or undecided?
fn is_nonzero_constant(arena: &mut Arena, e: ExprId) -> Truth {
    use crate::transforms::evalf::{Settled, ZeroSearch};
    if e == arena.zero {
        return Truth::False;
    }
    if !crate::base::walk::free_symbols(arena, e).is_empty()
        || crate::base::walk::has_unevaluated(arena, e)
    {
        return Truth::Unknown;
    }
    match crate::transforms::evalf::evalf_settled(arena, e, 20, ZeroSearch::Deep) {
        Ok((z, Settled::Certified)) => Truth::from_bool(!(z.0.is_zero() && z.1.is_zero())),
        Ok((_, Settled::ZeroToPrecision)) | Err(_) => Truth::Unknown,
    }
}

/// A linear family of a union, relative to the first family of its class:
/// step `q·a₀` and offset `b₀ + o·a₀` with rationals `q > 0`, `o`.
struct ClassMember {
    q: Q,
    o: Q,
}

/// A class of commensurable linear families of a union: base step `a₀`
/// and offset `b₀`, bound variable `var`.
struct LatticeClass {
    a0: ExprId,
    b0: ExprId,
    var: ExprId,
    members: Vec<ClassMember>,
    /// The original image-set nodes (returned unchanged when not merged).
    nodes: Vec<ExprId>,
}

/// The divisors of `p > 0`, ascending.
fn divisors(p: i64) -> Vec<i64> {
    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut d = 1i64;
    while d * d <= p {
        if p % d == 0 {
            small.push(d);
            if d * d != p {
                large.push(p / d);
            }
        }
        d += 1;
    }
    small.extend(large.into_iter().rev());
    small
}

/// The union of the families of `class` in the coarsest lattices: with
/// every step and offset over a common denominator `M` (steps `Qᵢ/M·a₀`,
/// offsets `b₀ + Oᵢ/M·a₀`), the union is `b₀ + a₀/M·R` for the set `R` of
/// residues modulo `P = lcm(Qᵢ)` covered by the classes `Oᵢ mod Qᵢ`; each
/// residue class `c mod d` (`d | P` ascending) inside `R` becomes one family
/// `a₀·d/M·n + b₀ + a₀·c/M`.  `None` when `P` exceeds [`MAX_MERGE_RESIDUES`].
fn merge_class(arena: &mut Arena, class: &LatticeClass) -> Option<Vec<ExprId>> {
    use num_integer::Integer;
    let mut m = num_bigint::BigInt::from(1);
    for cm in &class.members {
        m = m.lcm(cm.q.denom()).lcm(cm.o.denom());
    }
    let scale = Q::from_integer(m.clone());
    let mut steps: Vec<i64> = Vec::with_capacity(class.members.len());
    let mut offsets: Vec<i64> = Vec::with_capacity(class.members.len());
    for cm in &class.members {
        let qi = (&cm.q * &scale).to_integer().to_i64()?;
        let oi = (&cm.o * &scale).to_integer().to_i64()?;
        if qi <= 0 {
            return None;
        }
        steps.push(qi);
        offsets.push(oi);
    }
    let mut p: i64 = 1;
    for &qi in &steps {
        p = p.lcm(&qi);
        if p > MAX_MERGE_RESIDUES {
            return None;
        }
    }
    let pu = crate::base::cast::bounded_len(p, MAX_MERGE_RESIDUES)?;
    let mut covered_by_family = vec![false; pu];
    for (&qi, &oi) in steps.iter().zip(&offsets) {
        let mut r = oi.rem_euclid(qi);
        while r < p {
            covered_by_family[r as usize] = true;
            r += qi;
        }
    }
    let mut emitted = vec![false; pu];
    let mut out = Vec::new();
    let m_id = arena.big_int(m);
    for d in divisors(p) {
        for c in 0..d {
            let class_residues = (c..p).step_by(d as usize);
            let all_in = class_residues
                .clone()
                .all(|r| covered_by_family[r as usize]);
            let any_new = class_residues.clone().any(|r| !emitted[r as usize]);
            if !all_in || !any_new {
                continue;
            }
            for r in class_residues {
                emitted[r as usize] = true;
            }
            // a₀·d/M·n + b₀ + a₀·c/M
            let d_id = arena.int(d);
            let c_id = arena.int(c);
            let one = arena.one;
            let inv_m = arena.div(one, m_id);
            let step = arena.mul(&[class.a0, d_id, inv_m]);
            let shift = arena.mul(&[class.a0, c_id, inv_m]);
            let sn = arena.mul(&[step, class.var]);
            let body = arena.add(&[sn, class.b0, shift]);
            out.push(image_set(arena, class.var, body));
        }
    }
    Some(out)
}

/// The parts of a union with its image sets merged: commensurable linear
/// families combine into the coarsest lattices ([`merge_class`]), and the
/// elements of finite sets that are members of an image set of the union
/// are dropped.
fn merge_union_parts(arena: &mut Arena, parts: &[ExprId]) -> Vec<ExprId> {
    use crate::transforms::eval::eval;
    let mut classes: Vec<LatticeClass> = Vec::new();
    let mut others: Vec<ExprId> = Vec::new();
    for &p in parts {
        let ExprNode::ImageSet(var, body) = *arena.node(p) else {
            others.push(p);
            continue;
        };
        let Some(Lattice { step: a, offset: b }) = image_linear_parts(arena, var, body) else {
            classes.push(LatticeClass {
                a0: arena.zero,
                b0: arena.zero,
                var,
                members: Vec::new(),
                nodes: vec![p],
            });
            continue;
        };
        let mut placed = false;
        for class in classes.iter_mut() {
            if class.members.is_empty() {
                continue;
            }
            let q = arena.div(a, class.a0);
            let q = eval(arena, q);
            let Some(q) = arena.as_num(q).cloned() else {
                continue;
            };
            let db = arena.sub(b, class.b0);
            let o = arena.div(db, class.a0);
            let o = eval(arena, o);
            let Some(o) = arena.as_num(o).cloned() else {
                continue;
            };
            class.members.push(ClassMember { q: q.abs(), o });
            class.nodes.push(p);
            placed = true;
            break;
        }
        if !placed {
            classes.push(LatticeClass {
                a0: a,
                b0: b,
                var,
                members: vec![ClassMember {
                    q: Q::from_integer(1.into()),
                    o: Q::zero(),
                }],
                nodes: vec![p],
            });
        }
    }
    let mut merged: Vec<ExprId> = Vec::new();
    for class in &classes {
        if class.nodes.len() < 2 {
            merged.extend(class.nodes.iter().copied());
            continue;
        }
        match merge_class(arena, class) {
            Some(fams) => merged.extend(fams),
            None => merged.extend(class.nodes.iter().copied()),
        }
    }
    if merged.is_empty() {
        return parts.to_vec();
    }
    // Drop the elements of finite sets that are members of a family.
    let mut out = merged.clone();
    for o in others {
        let ExprNode::FiniteSet(elems) = arena.node(o).clone() else {
            out.push(o);
            continue;
        };
        let mut kept: Vec<ExprId> = Vec::with_capacity(elems.len());
        for e in elems {
            let mut member = false;
            for &f in &merged {
                if let ExprNode::ImageSet(v, body) = *arena.node(f)
                    && image_set_contains(arena, v, body, e).is_true()
                {
                    member = true;
                    break;
                }
            }
            if !member {
                kept.push(e);
            }
        }
        if !kept.is_empty() {
            out.push(arena.finite_set(&kept));
        }
    }
    out
}

/// Replace every `ImageSet` node of `set` (a set expression intersected
/// with a set whose hull is `[lo, hi]`) by the finite set of its members in
/// that window; one whose members cannot be listed stays.  Valid because
/// membership of a point of the window in `set` depends only on its
/// membership in the image sets.
fn cut_image_sets(arena: &mut Arena, set: ExprId, window: Window) -> ExprId {
    let mut out = set;
    for id in crate::base::walk::post_order_ids(arena, set) {
        let ExprNode::ImageSet(var, body) = *arena.node(id) else {
            continue;
        };
        if arena.node(body).is_set_node() {
            // a family of intervals: its intervals in the window
            if let Some(fam) = interval_family(arena, var, body)
                && let Some(pieces) = interval_family_in(arena, &fam, window)
            {
                let union = arena.set_union(&pieces);
                out = arena.subs_structural(out, id, union);
            }
            continue;
        }
        if let Some(points) = image_set_points_in(arena, var, body, window) {
            let finite = arena.finite_set(&points);
            out = arena.subs_structural(out, id, finite);
        }
    }
    out
}

/// Replace the image sets of `set` (intersected with the real line) whose
/// families have a non-real step or offset by their real members.
fn real_image_sets(arena: &mut Arena, set: ExprId) -> ExprId {
    let mut out = set;
    for id in crate::base::walk::post_order_ids(arena, set) {
        let ExprNode::ImageSet(var, body) = *arena.node(id) else {
            continue;
        };
        if image_set_is_real(arena, var, body).is_true() {
            continue;
        }
        let line = Window::line(arena);
        if let Some(points) = image_set_points_in(arena, var, body, line) {
            let finite = arena.finite_set(&points);
            out = arena.subs_structural(out, id, finite);
        }
    }
    out
}

/// Is every member of the linear family `{body : var ∈ ℤ}` real (real step
/// and offset)?
fn image_set_is_real(arena: &mut Arena, var: ExprId, body: ExprId) -> Truth {
    use crate::transforms::realness::constant_realness_as_declared;
    if matches!(arena.node(body), ExprNode::Interval(..)) {
        // the points of real intervals
        return Truth::True;
    }
    let Some(Lattice { step: a, offset: b }) = image_linear_parts(arena, var, body) else {
        return Truth::Unknown;
    };
    // exactly by the assumptions, else by a certified evaluation (an
    // unassumed parameter is complex)
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    let ra = Truth::from(constant_realness_as_declared(arena, a, 20, &mut cache));
    let rb = Truth::from(constant_realness_as_declared(arena, b, 20, &mut cache));
    ra.and(rb)
}

/// `⋂ parts` where some parts contain image sets (see [Image sets](#image-sets)).
fn intersect_image_parts(arena: &mut Arena, parts: &[ExprId]) -> ExprId {
    let (img, plain): (Vec<ExprId>, Vec<ExprId>) =
        parts.iter().partition(|&&p| contains_image_set(arena, p));
    if img.is_empty() || plain.is_empty() {
        return arena.set_intersection(parts);
    }
    let hull = arena.set_intersection(&plain);
    let hull = simplify_set(arena, hull);
    if hull == arena.empty_set {
        return hull;
    }
    let img_set = arena.set_intersection(&img);
    // A finite set: its members.
    if let ExprNode::FiniteSet(elems) = arena.node(hull).clone() {
        let mut kept = Vec::with_capacity(elems.len());
        let mut decided = true;
        for e in elems {
            match set_contains(arena, img_set, e) {
                Truth::True => kept.push(e),
                Truth::False => {}
                Truth::Unknown => {
                    decided = false;
                    break;
                }
            }
        }
        if decided {
            return arena.finite_set(&kept);
        }
    }
    let window = match (inf(arena, hull), sup(arena, hull)) {
        (Some(lo), Some(hi)) => Window::new(arena, lo, hi),
        _ => None,
    };
    let Some(window) = window else {
        return arena.set_intersection(&[hull, img_set]);
    };
    let cut = if window.is_bounded() {
        cut_image_sets(arena, img_set, window)
    } else {
        real_image_sets(arena, img_set)
    };
    // The real line adds nothing to sets of real families.
    let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
    let full = arena.interval(ninf, pinf, IntervalKind::Open);
    if hull == full && all_real_image_parts(arena, cut) {
        return cut;
    }
    arena.set_intersection(&[hull, cut])
}

/// Is `set` an image set of a real family, or a union of such and of
/// finite sets of real numbers?
fn all_real_image_parts(arena: &mut Arena, set: ExprId) -> bool {
    let parts: Vec<ExprId> = match arena.node(set) {
        ExprNode::SetUnion(ch) => ch.to_vec(),
        _ => vec![set],
    };
    parts.into_iter().all(|p| match arena.node(p).clone() {
        ExprNode::ImageSet(v, b) => image_set_is_real(arena, v, b).is_true(),
        ExprNode::FiniteSet(elems) => elems.iter().all(|&e| numeric_value(arena, e).is_some()),
        ExprNode::EmptySet => true,
        _ => false,
    })
}

/// Does `set` hold an intersection of the whole real line with image sets
/// (`ℝ ∩ {2nπ + asin(a)}`, families whose members are not all known to
/// be real, which the normal form keeps cut by the line)?
fn line_cuts_a_family(arena: &Arena, set: ExprId) -> bool {
    let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
    let is_line = |c: ExprId| match *arena.node(c) {
        ExprNode::Interval(a, b, _) => a == ninf && b == pinf,
        _ => false,
    };
    crate::base::walk::post_order_ids(arena, set)
        .into_iter()
        .any(|id| match arena.node(id) {
            ExprNode::SetIntersection(ch) => {
                ch.iter().any(|&c| is_line(c)) && ch.iter().any(|&c| contains_image_set(arena, c))
            }
            _ => false,
        })
}

/// Is every member of the set expression `set` known to be real (intervals,
/// finite sets of real numbers, image sets of real families)?
fn all_members_real(arena: &mut Arena, set: ExprId) -> bool {
    // bottom-up over the set structure (post-order from `collect`)
    let (order, _) = collect(arena, &[set]);
    let mut real: FxHashMap<ExprId, bool> = FxHashMap::default();
    for &id in &order {
        let get =
            |real: &FxHashMap<ExprId, bool>, c: ExprId| real.get(&c).copied().unwrap_or(false);
        let r = match arena.node(id).clone() {
            ExprNode::EmptySet | ExprNode::Interval(..) => true,
            ExprNode::SetUnion(ch) => ch.iter().all(|&c| get(&real, c)),
            // an intersection with a real set is real
            ExprNode::SetIntersection(ch) => ch.iter().any(|&c| get(&real, c)),
            // `A \ B` is real when `A` is
            ExprNode::SetComplement(a, _) => get(&real, a),
            ExprNode::FiniteSet(elems) => elems.iter().all(|&e| numeric_value(arena, e).is_some()),
            ExprNode::ImageSet(v, b) => image_set_is_real(arena, v, b).is_true(),
            _ => false,
        };
        real.insert(id, r);
    }
    real.get(&set).copied().unwrap_or(false)
}

/// `a \ b` where `a` contains image sets: the points of a finite `b` that
/// are not members of `a` are dropped.
fn complement_image_parts(arena: &mut Arena, a: ExprId, b: ExprId) -> ExprId {
    if let ExprNode::FiniteSet(elems) = arena.node(b).clone()
        && contains_image_set(arena, a)
    {
        let mut kept = Vec::with_capacity(elems.len());
        for e in elems {
            // not refuted: a point is forgotten only when it is proved
            // not to be a member of `a`
            if !set_contains(arena, a, e).is_false() {
                kept.push(e);
            }
        }
        if kept.is_empty() {
            return a;
        }
        let b = arena.finite_set(&kept);
        return arena.set_complement(a, b);
    }
    arena.set_complement(a, b)
}

/// Simplify the image sets of a set expression before its normal form is
/// taken (see [Image sets](#image-sets)): bottom-up over the set structure,
/// each image set is made canonical, each union merges its families, each
/// intersection cuts its image sets to the hull of its other operands, and
/// each complement drops the excluded points that are not members.
/// Returns `set` when it has no image set.
pub(crate) fn resolve_image_sets(arena: &mut Arena, set: ExprId) -> ExprId {
    if !contains_image_set(arena, set) {
        return set;
    }
    let (order, _) = collect(arena, &[set]);
    let mut memo: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &order {
        let get = |memo: &FxHashMap<ExprId, ExprId>, c: ExprId| memo.get(&c).copied().unwrap_or(c);
        let new = match arena.node(id).clone() {
            ExprNode::ImageSet(v, b) => image_set(arena, v, b),
            ExprNode::SetUnion(ch) => {
                let parts: Vec<ExprId> = ch.iter().map(|&c| get(&memo, c)).collect();
                let flat = arena.set_union(&parts);
                let flat: Vec<ExprId> = match arena.node(flat) {
                    ExprNode::SetUnion(ch) => ch.to_vec(),
                    _ => vec![flat],
                };
                let merged = merge_union_parts(arena, &flat);
                arena.set_union(&merged)
            }
            ExprNode::SetIntersection(ch) => {
                let parts: Vec<ExprId> = ch.iter().map(|&c| get(&memo, c)).collect();
                intersect_image_parts(arena, &parts)
            }
            ExprNode::SetComplement(a, b) => {
                let (a, b) = (get(&memo, a), get(&memo, b));
                complement_image_parts(arena, a, b)
            }
            _ => id,
        };
        memo.insert(id, new);
    }
    memo.get(&set).copied().unwrap_or(set)
}

// ═══════════════════════════════════════════════════════════════════════════
// Declared domains
// ═══════════════════════════════════════════════════════════════════════════

/// The declared domain of a variable as a set ([`declared_domain_set`]).
pub(crate) enum DeclaredDomain {
    /// No restriction (within the line, for `real_line`).
    Unrestricted,
    /// The domain.
    Set(ExprId),
    /// A restriction no set expression writes (rational, prime, imaginary,
    /// …).
    Unsupported,
}

/// The declared domain of the symbol `var` as a set: unions of intervals
/// for the signs (`(0, ∞)` for positive, `ℝ \ {0}` for a real nonzero, …),
/// the integers `{n : n ∈ ℤ}` (even `{2n}`, odd `{2n + 1}`) intersected with
/// them, and `ℂ \ {0}` for a nonzero symbol not declared real.  With
/// `real_line` the domain within ℝ (for inequalities: `ℝ` itself is no
/// restriction there, and a nonzero symbol ranges over `ℝ \ {0}`).
pub(crate) fn declared_domain_set(
    arena: &mut Arena,
    var: ExprId,
    real_line: bool,
) -> DeclaredDomain {
    declared_domain_set_with(arena, var, real_line, false)
}

/// [`declared_domain_set`]; with `primes_as_integers` a prime or composite
/// symbol ranges over the integers of its signs (the primality is then
/// left to the caller).
fn declared_domain_set_with(
    arena: &mut Arena,
    var: ExprId,
    real_line: bool,
    primes_as_integers: bool,
) -> DeclaredDomain {
    use crate::base::assumptions::Props;
    let Some(a) = crate::transforms::solve::declared_domain(arena, var) else {
        return DeclaredDomain::Unrestricted;
    };
    if Truth::from(a.query(Props::REAL)).is_false() || a.known_true.intersects(Props::IMAGINARY) {
        return DeclaredDomain::Unsupported;
    }
    let integer = Truth::from(a.query(Props::INTEGER)).is_true();
    let kind = Props::RATIONAL | Props::IRRATIONAL | Props::ALGEBRAIC | Props::TRANSCENDENTAL;
    let unsupported = if integer {
        a.known_true
            .intersects(Props::IRRATIONAL | Props::TRANSCENDENTAL)
            || a.known_false.intersects(Props::RATIONAL | Props::ALGEBRAIC)
            || (!primes_as_integers
                && (a.query(Props::PRIME).is_some() || a.query(Props::COMPOSITE).is_some()))
    } else {
        let parity = Props::EVEN | Props::ODD | Props::PRIME | Props::COMPOSITE;
        a.known_true.intersects(kind | parity)
            || a.known_false.intersects(kind | parity | Props::INTEGER)
    };
    if unsupported {
        return DeclaredDomain::Unsupported;
    }
    let real = Truth::from(a.query(Props::REAL)).is_true() || real_line;
    let (zero, inf, ninf) = (arena.zero, arena.infinity, arena.neg_infinity);
    let open = IntervalKind::Open;
    let holds = |p: Props| Truth::from(a.query(p)).is_true();
    let fails = |p: Props| Truth::from(a.query(p)).is_false();
    let mut parts: Vec<ExprId> = Vec::new();
    if holds(Props::POSITIVE) || (real && fails(Props::NONPOSITIVE)) {
        parts.push(arena.interval(zero, inf, open));
    }
    if holds(Props::NEGATIVE) || (real && fails(Props::NONNEGATIVE)) {
        parts.push(arena.interval(ninf, zero, open));
    }
    if holds(Props::NONNEGATIVE) || (real && fails(Props::NEGATIVE)) {
        parts.push(arena.interval(zero, inf, IntervalKind::RightOpen));
    }
    if holds(Props::NONPOSITIVE) || (real && fails(Props::POSITIVE)) {
        parts.push(arena.interval(ninf, zero, IntervalKind::LeftOpen));
    }
    if holds(Props::ZERO) || fails(Props::NONZERO) {
        parts.push(arena.finite_set(&[zero]));
    }
    // `≠ 0` beyond the signs (and odd integers): the set without `0`, as
    // a complement (an intersection with `(−∞, 0) ∪ (0, ∞)` would print
    // as `A ∩ B ∪ C`).
    let strict = holds(Props::POSITIVE) || holds(Props::NEGATIVE) || holds(Props::ODD);
    let nonzero = (holds(Props::NONZERO) || fails(Props::ZERO)) && !strict;
    let origin = arena.finite_set(&[zero]);
    if integer {
        let scope = arena.finite_set(&[var]);
        let n = image_param(arena, zero, scope);
        let body = if holds(Props::EVEN) {
            let two = arena.int(2);
            arena.mul(&[two, n])
        } else if holds(Props::ODD) {
            let two = arena.int(2);
            let two_n = arena.mul(&[two, n]);
            let one = arena.one;
            arena.add(&[two_n, one])
        } else {
            n
        };
        let lattice = image_set(arena, n, body);
        parts.push(if nonzero {
            arena.set_complement(lattice, origin)
        } else {
            lattice
        });
    } else if nonzero {
        let all = if real {
            arena.interval(ninf, inf, open)
        } else {
            arena.universal_set
        };
        parts.push(arena.set_complement(all, origin));
    } else if Truth::from(a.query(Props::REAL)).is_true() && parts.is_empty() && !real_line {
        parts.push(arena.interval(ninf, inf, open));
    }
    if parts.is_empty() {
        return DeclaredDomain::Unrestricted;
    }
    let meet = arena.set_intersection(&parts);
    DeclaredDomain::Set(simplify_set(arena, meet))
}

/// The solution set `set` of an equation in `var` restricted to the
/// declared domain of `var`: the elements of its finite parts that
/// certainly violate an assumption of `var` are dropped, those that
/// satisfy them all are kept, and the rest (undecided elements, image
/// sets) is intersected with [`declared_domain_set`] — `{−√a, √a} ∩ (0, ∞)`
/// for a positive `x` and a plain `a`, `{nπ : n ∈ ℤ} ∩ (0, ∞)`, as SymPy's
/// `solveset` intersects with its domain.  For a domain no set writes
/// (rational, prime, …) the undecided elements are kept and a set with
/// image sets is not known (`None`).  `UniversalSet` (an identity) becomes
/// the domain.
pub(crate) fn restrict_set_to_domain(
    arena: &mut Arena,
    var: ExprId,
    set: ExprId,
) -> Option<ExprId> {
    use crate::transforms::solve::assumptions_hold;
    let domain = declared_domain_set(arena, var, false);
    if matches!(domain, DeclaredDomain::Unrestricted) {
        return Some(set);
    }
    if set == arena.universal_set {
        return Some(match domain {
            DeclaredDomain::Set(d) => d,
            _ => set,
        });
    }
    let parts: Vec<ExprId> = match arena.node(set) {
        ExprNode::SetUnion(ch) => ch.to_vec(),
        _ => vec![set],
    };
    let mut decided: Vec<ExprId> = Vec::new();
    let mut decided_sets: Vec<ExprId> = Vec::new();
    let mut undecided: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for p in parts {
        match arena.node(p).clone() {
            ExprNode::FiniteSet(elems) => {
                for e in elems {
                    match Truth::from(assumptions_hold(arena, var, e)) {
                        Truth::True => decided.push(e),
                        Truth::False => {}
                        Truth::Unknown => {
                            // A member that is exactly 0 without being
                            // written 0 is 0 (up to 0.46 `x − atan(1/2) −
                            // atan(1/3) + π/4` for a positive `x` gave
                            // `(0, ∞) ∩ {atan(1/2) + atan(1/3) − π/4}`).
                            let z = crate::transforms::solve::exact_zero_form(arena, e);
                            let exact = if z != e {
                                Truth::from(assumptions_hold(arena, var, z))
                            } else {
                                Truth::Unknown
                            };
                            match exact {
                                Truth::True => decided.push(z),
                                Truth::False => {}
                                Truth::Unknown => undecided.push(e),
                            }
                        }
                    }
                }
            }
            ExprNode::EmptySet => {}
            // A family of intervals: cut by the domain below.
            ExprNode::ImageSet(_, body) if arena.node(body).is_set_node() => rest.push(p),
            // A family: all, none or one of its members, or without `0`
            // (`{nπ} \ {0}`); otherwise cut by the domain below.
            ExprNode::ImageSet(n, body) => {
                use crate::transforms::solve::{FamilyInDomain, family_in_domain};
                match family_in_domain(arena, var, body, n, n) {
                    FamilyInDomain::All => decided_sets.push(p),
                    FamilyInDomain::Empty => {}
                    FamilyInDomain::Point(m) => decided.push(m),
                    FamilyInDomain::Rays(rays) if rays.len() == 2 => {
                        let origin = arena.finite_set(&[arena.zero]);
                        decided_sets.push(arena.set_complement(p, origin));
                    }
                    FamilyInDomain::Rays(_) | FamilyInDomain::Undecided => rest.push(p),
                }
            }
            _ => rest.push(p),
        }
    }
    let mut out = vec![arena.finite_set(&decided)];
    out.extend(decided_sets);
    match domain {
        DeclaredDomain::Set(d) => {
            // For a real `var`, a member real exactly under a condition on
            // the parameters (`±√y` for a real `y`: `y ≥ 0`) is written
            // with it: `{√y, −√y} ∩ ConditionSet(x, y ≥ 0)`, the members if
            // `y ≥ 0` and `∅` otherwise (up to 0.52 `ℝ ∩ {√y, −√y}`, the
            // same set less legibly).
            let (ninf, inf) = (arena.neg_infinity, arena.infinity);
            let line = arena.interval(ninf, inf, IntervalKind::Open);
            if d == line {
                let mut groups: Vec<(ExprId, Vec<ExprId>)> = Vec::new();
                let mut plain: Vec<ExprId> = Vec::with_capacity(undecided.len());
                for e in std::mem::take(&mut undecided) {
                    match crate::transforms::solve::realness_condition(arena, e) {
                        Some(cond) => match groups.iter_mut().find(|g| g.0 == cond) {
                            Some(g) => g.1.push(e),
                            None => groups.push((cond, vec![e])),
                        },
                        None => plain.push(e),
                    }
                }
                undecided = plain;
                for (cond, members) in groups {
                    let members = arena.finite_set(&members);
                    let when = arena.intern(ExprNode::ConditionSet(var, cond));
                    out.push(arena.set_intersection(&[members, when]));
                }
            }
            if !undecided.is_empty() {
                rest.push(arena.finite_set(&undecided));
            }
            // part by part, as SymPy writes `⋃ (Aᵢ ∩ D)`
            for p in rest {
                let meet = arena.set_intersection(&[p, d]);
                out.push(simplify_set(arena, meet));
            }
        }
        _ => {
            if !rest.is_empty() {
                return None;
            }
            out.push(arena.finite_set(&undecided));
        }
    }
    let union = arena.set_union(&out);
    Some(simplify_set(arena, union))
}

/// `set` with each family (`ImageSet` of points) that
/// [`family_in_domain`](crate::transforms::solve::family_in_domain) decides
/// for the declared domain of `var` replaced by its members there: none
/// (dropped) or one point; the others and every other part unchanged (the
/// same `ExprId` when nothing is decided).  As [`restrict_set_to_domain`]
/// does for `solve_as_set`: before 0.50 `reduce_inequalities` of
/// `e^(ix) + 1 = 0` for an integer `x` was `ℤ ∩ {2nπ + π : n ∈ ℤ}` where
/// `solve_as_set` gave `∅` (`π` is irrational).
fn families_in_domain(arena: &mut Arena, var: ExprId, set: ExprId) -> ExprId {
    use crate::transforms::solve::{FamilyInDomain, family_in_domain};
    let parts: Vec<ExprId> = match arena.node(set) {
        ExprNode::SetUnion(ch) => ch.to_vec(),
        _ => vec![set],
    };
    let mut changed = false;
    let mut out: Vec<ExprId> = Vec::with_capacity(parts.len());
    for p in parts {
        if let ExprNode::ImageSet(n, body) = *arena.node(p)
            && !arena.node(body).is_set_node()
        {
            match family_in_domain(arena, var, body, n, n) {
                FamilyInDomain::Empty => {
                    changed = true;
                    continue;
                }
                FamilyInDomain::Point(m) => {
                    changed = true;
                    out.push(arena.finite_set(&[m]));
                    continue;
                }
                FamilyInDomain::All | FamilyInDomain::Rays(_) | FamilyInDomain::Undecided => {}
            }
        }
        out.push(p);
    }
    if changed { arena.set_union(&out) } else { set }
}

/// A set of real numbers (an inequality's solution in `var`) intersected
/// with the declared domain of `var` within ℝ ([`declared_domain_set`]
/// with `real_line`); `None` for a domain no set writes.
pub(crate) fn restrict_real_set_to_domain(
    arena: &mut Arena,
    var: ExprId,
    set: ExprId,
) -> Option<ExprId> {
    match declared_domain_set(arena, var, true) {
        DeclaredDomain::Unrestricted => Some(set),
        DeclaredDomain::Set(d) => {
            let set = families_in_domain(arena, var, set);
            let meet = arena.set_intersection(&[set, d]);
            Some(simplify_set(arena, meet))
        }
        DeclaredDomain::Unsupported => {
            // A prime (composite) symbol: the integers of a bounded set,
            // each tested.
            let DeclaredDomain::Set(d) = declared_domain_set_with(arena, var, true, true) else {
                return None;
            };
            let meet = arena.set_intersection(&[set, d]);
            let meet = simplify_set(arena, meet);
            let elems = match arena.node(meet).clone() {
                ExprNode::FiniteSet(elems) => elems.to_vec(),
                ExprNode::EmptySet => return Some(meet),
                _ => return None,
            };
            let mut kept = Vec::with_capacity(elems.len());
            for e in elems {
                match Truth::from(crate::transforms::solve::assumptions_hold(arena, var, e)) {
                    Truth::True => kept.push(e),
                    Truth::False => {}
                    Truth::Unknown => return None,
                }
            }
            Some(arena.finite_set(&kept))
        }
    }
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
        // Constant that is not a finite real: complex (an imaginary part its
        // value keeps is certified nonzero; up to 0.57 the printed decimal
        // was searched for `i`) or unevaluable.
        match crate::transforms::evalf::const_value(arena, ev, 16) {
            Ok(v) if v.im_sign().is_nonzero().is_true() => {}
            Ok(_) | Err(_) => keep.push(ev),
        }
    }
    arena.finite_set(&keep)
}

/// An equation atom with periodic solution families: its set stands as the
/// exact `{var | expr = 0}` until bounds from the other conditions allow
/// listing the members ([`resolve_periodic`]).
struct PeriodicAtom {
    /// The `ConditionSet` node standing for the solution set.
    set: ExprId,
    /// The equation `expr = 0`.
    expr: ExprId,
    /// Its general-solve candidates.
    cands: crate::transforms::solve::PeriodicCandidates,
}

/// Convert one relational atom in `var` to its solution set.
///
/// An equation with periodic solutions (`sin x = 0`) becomes
/// `{var | sin x = 0}`, recorded in `periodic`: its principal solutions
/// `{0, π}` are not its solution set.  Before 0.39 they were, and
/// `sin(x) = 0 ∧ −4 < x < 4` gave `{0, π}` without `−π` (SymPy's
/// `{0, π}` without `−π` (SymPy's
/// `reduce_inequalities` gives the same wrong set).
///
/// The set is cut to the points where both sides are defined, which
/// `lhs − rhs` can forget: up to 0.46 `x + 1/x > −2 + 1/x` was `(−2, ∞)`,
/// with the pole `0` (SymPy: `(−2, 0) ∪ (0, ∞)`).
fn atom_to_set(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    rel: crate::transforms::inequalities::Relation,
    is_eq: bool,
    var: ExprId,
    periodic: &mut Vec<PeriodicAtom>,
) -> Result<ExprId, SymplexError> {
    // An order relation holds only between real numbers.  A finite side
    // free of `var` that is certainly not real makes it hold nowhere,
    // whatever the difference of the sides is: up to 0.59 `x + i > i` was
    // `(0, ∞)` and `|x| > i` was `(i, ∞) ∪ (−∞, −i)`.  (An equation is
    // about the difference alone.)
    if !is_eq {
        use crate::base::value_class::ValueClass;
        let mut cache = crate::base::assumptions::AssumptionCache::new();
        for (side, other) in [(lhs, rhs), (rhs, lhs)] {
            if !crate::base::walk::contains(arena, side, var)
                && crate::base::walk::contains(arena, other, var)
                && ValueClass::of(arena, side).is_finite_proved()
                && crate::transforms::logic::known_nonreal(arena, &mut cache, side)
            {
                return Ok(arena.empty_set);
            }
        }
    }
    // A side ±∞ against a finite side in `var`: decided by the sign of
    // the infinity, wherever the finite side is defined (`−∞ < x` is ℝ;
    // up to 0.46 "does not involve the variable and cannot be decided").
    let infinite = |arena: &Arena, e: ExprId| -> Option<std::cmp::Ordering> {
        if e == arena.infinity {
            Some(std::cmp::Ordering::Greater)
        } else if e == arena.neg_infinity {
            Some(std::cmp::Ordering::Less)
        } else {
            None
        }
    };
    let (li, ri) = (infinite(arena, lhs), infinite(arena, rhs));
    if li.is_some() != ri.is_some() {
        let finite = if li.is_some() { rhs } else { lhs };
        if crate::base::walk::contains(arena, finite, var) {
            // the sign of lhs − rhs
            let sign = match (li, ri) {
                (Some(s), None) => s,
                (None, Some(s)) => s.reverse(),
                _ => std::cmp::Ordering::Equal,
            };
            let holds = !is_eq && rel.accepts(sign);
            if !holds {
                return Ok(arena.empty_set);
            }
            let zero = arena.zero;
            return match crate::transforms::inequalities::lost_domain(arena, &[finite], zero, var) {
                Ok(Some(domain)) => Ok(domain),
                Ok(None) => {
                    let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
                    Ok(arena.interval(ninf, pinf, IntervalKind::Open))
                }
                Err(e) => Err(SymplexError::ComputationFailed {
                    operation: "reduce_inequalities",
                    reason: format!("could not find where the side is defined: {e}"),
                }),
            };
        }
    }
    let set = atom_to_set_of_difference(arena, lhs, rhs, rel, is_eq, var, periodic)?;
    let d = arena.sub(lhs, rhs);
    let d = crate::transforms::eval::eval(arena, d);
    let mut set = match crate::transforms::inequalities::lost_domain(arena, &[lhs, rhs], d, var) {
        Ok(Some(domain)) => {
            let meet = arena.set_intersection(&[set, domain]);
            simplify_set(arena, meet)
        }
        Ok(None) => set,
        Err(e) => {
            return Err(SymplexError::ComputationFailed {
                operation: "reduce_inequalities",
                reason: format!("could not find where both sides are defined: {e}"),
            });
        }
    };
    // The order holds only where both sides are real, which `lhs − rhs` can
    // forget as well: a side with non-real coefficients is real at the
    // zeros of its imaginary part only (up to 0.59 `t + i > 2t + i` was
    // `(−∞, 0)`, and `i·t > i·t − 1`, which holds at `t = 0` alone, every
    // value).
    if !is_eq {
        for side in [lhs, rhs] {
            if let Some(real_at) = crate::transforms::inequalities::real_points(arena, side, var) {
                let real_at = real_at.map_err(|e| SymplexError::ComputationFailed {
                    operation: "reduce_inequalities",
                    reason: format!("could not find where a side is real: {e}"),
                })?;
                let meet = arena.set_intersection(&[set, real_at]);
                set = simplify_set(arena, meet);
            }
        }
    }
    Ok(set)
}

/// What `d = 0` is as an equation between rational functions of the
/// variable (not a truth value: "not rational" is no undecided case).
enum RationalForm {
    /// `d` is not a quotient of polynomials in the variable.
    NotRational,
    /// A rational function whose numerator is the zero polynomial.
    ZeroNumerator,
    /// A rational function with a nonzero numerator.
    NonzeroNumerator,
}

/// [`atom_to_set`] for `lhs − rhs` alone.
fn atom_to_set_of_difference(
    arena: &mut Arena,
    lhs: ExprId,
    rhs: ExprId,
    rel: crate::transforms::inequalities::Relation,
    is_eq: bool,
    var: ExprId,
    periodic: &mut Vec<PeriodicAtom>,
) -> Result<ExprId, SymplexError> {
    use crate::transforms::inequalities::Relation;

    let d = arena.sub(lhs, rhs);
    let d = crate::transforms::eval::eval(arena, d);

    if !crate::base::walk::contains(arena, d, var) {
        // Constant condition (with respect to `var`): decided for a
        // number only.
        let holds = match arena.as_num(d).cloned() {
            Some(r) => Truth::from_bool(if is_eq {
                r.is_zero()
            } else {
                match rel {
                    Relation::Gt => r.is_positive(),
                    Relation::Ge => !r.is_negative(),
                    Relation::Lt => r.is_negative(),
                    Relation::Le => !r.is_positive(),
                }
            }),
            None => Truth::Unknown,
        };
        return match holds {
            Truth::True => Ok(arena.universal_set),
            Truth::False => Ok(arena.empty_set),
            Truth::Unknown => Err(SymplexError::InvalidArgument {
                operation: "reduce_inequalities",
                reason: "condition does not involve the variable and cannot be decided".into(),
            }),
        };
    }

    if is_eq {
        // The exact solution set as image sets, cut to the real line, when
        // the general solution has periodic families.
        if crate::poly::polybridge::expr_to_poly(arena, d, var).is_none()
            && let Some(set) =
                crate::transforms::inequalities::general_solution_set(arena, d, var, false)
            && !crate::base::walk::has_unevaluated(arena, set)
            && contains_image_set(arena, set)
        {
            let (ninf, pinf) = (arena.neg_infinity, arena.infinity);
            let reals = arena.interval(ninf, pinf, IntervalKind::Open);
            let real = arena.set_intersection(&[set, reals]);
            let real = simplify_set(arena, real);
            // Families not known to be real (`2nπ + asin(a)` for a parameter
            // `a`) take the condition-set path: the normal form keeps them
            // as `ℝ ∩ {…}`.
            if all_members_real(arena, real) && !line_cuts_a_family(arena, real) {
                return Ok(real);
            }
        }
        if crate::poly::polybridge::expr_to_poly(arena, d, var).is_none()
            && let Some(cands) = crate::transforms::solve::periodic_candidates(arena, d, var)
        {
            let zero = arena.zero;
            let cond = arena.eq_(d, zero);
            let set = arena.intern(ExprNode::ConditionSet(var, cond));
            periodic.push(PeriodicAtom {
                set,
                expr: d,
                cands,
            });
            return Ok(set);
        }
        // Every value where `d` is defined solves `d = 0` when the solver
        // finds an identity or `d` is a rational function with the zero
        // numerator: the set is that of `d ≥ 0` (`d` is 0 wherever it is
        // defined), which leaves out the poles (`(x² − 1)/(x − 1) = x + 1`
        // holds on ℝ \ {1}).  Up to 0.41 both were read as an empty list of
        // roots and the set was `EmptySet` (SymPy 1.14: `Reals`, and
        // `Union(Interval.open(-oo, 1), Interval.open(1, oo))` for the
        // quotient).
        let identity = |arena: &mut Arena| {
            crate::transforms::inequalities::solve_inequality_over_reals(
                arena,
                d,
                var,
                Relation::Ge,
            )
            .map_err(|e| SymplexError::ComputationFailed {
                operation: "reduce_inequalities",
                reason: format!("could not find the domain of an identity: {e}"),
            })
        };
        // A provable contradiction (`cos x = −2`, `x/sin x = 0`: every root a
        // pole) has no solution; before 0.40 it was "could not solve".  The
        // variable is real here: the range restrictions take it so.
        // (Roots found, or none: the completeness test is below; a set
        // needs every root, so a partial outcome is refused.)
        use crate::transforms::solve::{RootSet, Solutions};
        let roots: Vec<ExprId> =
            match crate::transforms::solve::solve_classified_real(arena, d, var) {
                Solutions::Complete(RootSet::Empty(_)) => return Ok(arena.empty_set),
                Solutions::Complete(RootSet::All) => return identity(arena),
                Solutions::Partial(..) => {
                    return Err(SymplexError::ComputationFailed {
                        operation: "reduce_inequalities",
                        reason: "could not solve the equation (some of its roots were not found)"
                            .into(),
                    });
                }
                outcome => outcome.claimed_points(),
            };
        // No root is an answer for a rational function (the solver finds
        // every root of a polynomial numerator); before 0.30 only for a
        // polynomial, so `−1/(x² + 7x + 12) = 0` was an error.
        let rational = {
            let combined = crate::poly::polybridge::together(arena, d);
            let (n, den) = crate::poly::polybridge::as_numer_denom(arena, combined);
            match (
                crate::poly::polybridge::expr_to_poly(arena, n, var),
                crate::poly::polybridge::expr_to_poly(arena, den, var),
            ) {
                (Some(pn), Some(_)) if pn.is_zero() => RationalForm::ZeroNumerator,
                (Some(_), Some(_)) => RationalForm::NonzeroNumerator,
                _ => RationalForm::NotRational,
            }
        };
        if matches!(rational, RationalForm::ZeroNumerator) {
            return identity(arena);
        }
        if roots.is_empty() && matches!(rational, RationalForm::NotRational) {
            return Err(SymplexError::ComputationFailed {
                operation: "reduce_inequalities",
                reason: "could not solve the equation".into(),
            });
        }
        return Ok(real_roots_set(arena, &roots));
    }

    crate::transforms::inequalities::solve_inequality_over_reals(arena, d, var, rel).map_err(|e| {
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
///
/// A relation holds only where both sides are defined, and so does its
/// negation: `a ≠ b` is `a > b ∨ a < b`, and `Not` is pushed down to the
/// atoms (`¬(a > b)` is `a ≤ b`, De Morgan through `And`/`Or`), as SymPy
/// negates relationals.  Before 0.30 both were set complements, so
/// `(x + 2)/(x − 4) ≠ 0` and `¬(1/x > 0)` contained the pole.
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

    // Top-level conjunctions are split into their conjuncts (the same
    // conjunction), so that the bounds of a window `a < x ∧ x < b` given in
    // one condition reach the periodic equations ([`resolve_periodic`]).
    let mut flat: Vec<ExprId> = Vec::with_capacity(conds.len());
    let mut pending: Vec<ExprId> = conds.iter().rev().copied().collect();
    while let Some(c) = pending.pop() {
        match arena.node(c) {
            ExprNode::And(ch) => pending.extend(ch.iter().rev().copied()),
            _ => flat.push(c),
        }
    }
    let mut sets: Vec<ExprId> = Vec::with_capacity(flat.len());
    let mut periodic: Vec<PeriodicAtom> = Vec::new();
    for &cond in &flat {
        // Post-order over the boolean structure; a node is visited with
        // the parity of the `Not`s above it (`negated`).
        let mut order: Vec<(ExprId, bool)> = Vec::new();
        let mut visited: FxHashSet<(ExprId, bool)> = FxHashSet::default();
        let mut stack: Vec<((ExprId, bool), bool)> = vec![((cond, false), false)];
        while let Some((key, expanded)) = stack.pop() {
            if visited.contains(&key) {
                continue;
            }
            if expanded {
                visited.insert(key);
                order.push(key);
                continue;
            }
            stack.push((key, true));
            let (id, negated) = key;
            match arena.node(id) {
                ExprNode::And(ch) | ExprNode::Or(ch) => {
                    for &c in ch.iter().rev() {
                        stack.push(((c, negated), false));
                    }
                }
                ExprNode::Not(x) => stack.push(((*x, !negated), false)),
                _ => {}
            }
        }

        let mut memo: FxHashMap<(ExprId, bool), ExprId> = FxHashMap::default();
        for &(id, negated) in &order {
            let s = match (arena.node(id).clone(), negated) {
                (ExprNode::BoolTrue, false) | (ExprNode::BoolFalse, true) => arena.universal_set,
                (ExprNode::BoolFalse, false) | (ExprNode::BoolTrue, true) => arena.empty_set,
                (ExprNode::Gt(a, b), false) => {
                    atom_to_set(arena, a, b, Relation::Gt, false, var, &mut periodic)?
                }
                (ExprNode::Gt(a, b), true) => {
                    atom_to_set(arena, b, a, Relation::Ge, false, var, &mut periodic)?
                }
                (ExprNode::Ge(a, b), false) => {
                    atom_to_set(arena, a, b, Relation::Ge, false, var, &mut periodic)?
                }
                (ExprNode::Ge(a, b), true) => {
                    atom_to_set(arena, b, a, Relation::Gt, false, var, &mut periodic)?
                }
                (ExprNode::Eq_(a, b), false) | (ExprNode::Ne(a, b), true) => {
                    atom_to_set(arena, a, b, Relation::Ge, true, var, &mut periodic)?
                }
                (ExprNode::Ne(a, b), false) | (ExprNode::Eq_(a, b), true) => {
                    let above = atom_to_set(arena, a, b, Relation::Gt, false, var, &mut periodic)?;
                    let below = atom_to_set(arena, b, a, Relation::Gt, false, var, &mut periodic)?;
                    arena.set_union(&[above, below])
                }
                (ExprNode::And(ch), _) | (ExprNode::Or(ch), _) => {
                    // `Not` over `And` is `Or` of the negations and vice versa.
                    let is_and = matches!(arena.node(id), ExprNode::And(_)) != negated;
                    let default = if is_and {
                        arena.universal_set
                    } else {
                        arena.empty_set
                    };
                    let parts: Vec<ExprId> = ch
                        .iter()
                        .map(|c| memo.get(&(*c, negated)).copied().unwrap_or(default))
                        .collect();
                    if is_and {
                        arena.set_intersection(&parts)
                    } else {
                        arena.set_union(&parts)
                    }
                }
                (ExprNode::Not(x), _) => {
                    memo.get(&(x, !negated)).copied().unwrap_or(arena.empty_set)
                }
                (other, _) => {
                    return Err(SymplexError::InvalidArgument {
                        operation: "reduce_inequalities",
                        reason: format!("not a relational condition: {other:?}"),
                    });
                }
            };
            memo.insert((id, negated), s);
        }
        if let Some(&s) = memo.get(&(cond, false)) {
            sets.push(s);
        }
    }

    if !periodic.is_empty() {
        resolve_periodic(arena, &mut sets, &periodic, var);
    }
    let combined = arena.set_intersection(&sets);
    Ok(simplify_set(arena, combined))
}

/// Replace the `{var | expr = 0}` sets of periodic equations by the finite
/// set of their solutions in the bounded hull of the conditions that do not
/// involve them.  The conditions only combine sets by union and
/// intersection (negations are pushed to the atoms), so cutting each
/// periodic set to that hull leaves the conjunction unchanged.  Every
/// member listed is checked against the equation, so a pole on a family
/// (`0` in `2nπ` for `sin(x)/x = 0`) is left out.  Without such a hull the
/// sets stay as they are, exact.
fn resolve_periodic(
    arena: &mut Arena,
    sets: &mut [ExprId],
    periodic: &[PeriodicAtom],
    var: ExprId,
) {
    let bounding: Vec<ExprId> = sets
        .iter()
        .copied()
        .filter(|&s| {
            !periodic
                .iter()
                .any(|p| crate::base::walk::contains(arena, s, p.set))
        })
        .collect();
    if bounding.is_empty() {
        return;
    }
    let hull = arena.set_intersection(&bounding);
    let hull = simplify_set(arena, hull);
    let (Some(lo), Some(hi)) = (inf(arena, hull), sup(arena, hull)) else {
        return;
    };
    let Some(window) = Window::new(arena, lo, hi) else {
        return;
    };
    if !window.is_bounded() {
        return;
    }
    for atom in periodic {
        let Some(points) = periodic_points_in(arena, atom, var, window) else {
            continue;
        };
        let finite = real_roots_set(arena, &points);
        for s in sets.iter_mut() {
            *s = arena.subs_structural(*s, atom.set, finite);
        }
    }
}

/// The members of the periodic atom's families (and its other candidates)
/// within the window (with a margin; exact bounds come from the
/// intersection) that are not certainly non-solutions.  `None` when a
/// family cannot be located or has too many members there.  The families
/// are located as image sets ([`image_set_points_in`]: exact index ranges,
/// exact real and imaginary parts); up to 0.53 by `f64` values, so that a
/// step or offset with an imaginary part below `10⁻¹²` of it was taken as
/// real or as non-real at random, and the indices far from the origin were
/// lost (`sin(x − 10²⁰) = 0` had no zero in `[0, 10]`).
fn periodic_points_in(
    arena: &mut Arena,
    atom: &PeriodicAtom,
    var: ExprId,
    window: Window,
) -> Option<Vec<ExprId>> {
    let n = atom.cands.param;
    let mut points: Vec<ExprId> = Vec::new();
    let push = |arena: &mut Arena, p: ExprId, points: &mut Vec<ExprId>| {
        if !points.contains(&p)
            && !crate::transforms::solve::certainly_not_a_root(arena, atom.expr, var, p)
        {
            points.push(p);
        }
    };
    for &p in &atom.cands.plain {
        push(arena, p, &mut points);
    }
    for &fam in &atom.cands.families {
        if !crate::base::walk::has_free_var(arena, fam, n) {
            push(arena, fam, &mut points);
            continue;
        }
        for m in image_set_points_in(arena, n, fam, window)? {
            push(arena, m, &mut points);
        }
    }
    Some(points)
}

/// A 16-digit approximation of a real constant (`±∞` for the infinities):
/// the `f64` correctly rounded from its 17-digit value (up to 0.57 the
/// printed decimal was parsed).
fn approx_real_f64(arena: &mut Arena, e: ExprId) -> Option<f64> {
    if e == arena.infinity {
        return Some(f64::INFINITY);
    }
    if e == arena.neg_infinity {
        return Some(f64::NEG_INFINITY);
    }
    if let Some(r) = arena.as_num(e) {
        return num_traits::ToPrimitive::to_f64(r);
    }
    crate::transforms::evalf::const_value(arena, e, 17)
        .ok()?
        .to_f64()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::interval::IntervalKind::{Closed, LeftOpen, Open};

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    fn iv(arena: &mut Arena, a: i64, b: i64, kind: IntervalKind) -> ExprId {
        let a = arena.int(a);
        let b = arena.int(b);
        arena.interval(a, b, kind)
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
        let a = iv(&mut arena, 0, 2, Closed);
        let b = iv(&mut arena, 1, 3, Closed);
        let i = arena.set_intersection(&[a, b]);
        let s = simplify_set(&mut arena, i);
        assert_eq!(display(&arena, s), "[1, 2]");
    }

    #[test]
    fn simplify_union_merges_adjacent() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, Closed);
        let b = iv(&mut arena, 1, 2, Open);
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
        let left = arena.interval(arena.neg_infinity, m2, Open);
        let right = arena.interval(p2, arena.infinity, Open);
        let u = arena.set_union(&[left, right]);
        let box_ = iv(&mut arena, 0, 5, Closed);
        let i = arena.set_intersection(&[u, box_]);
        let s = simplify_set(&mut arena, i);
        assert_eq!(display(&arena, s), "(2, 5]");
    }

    #[test]
    fn simplify_points_absorbed_into_intervals() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 2, Open);
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
        let symbolic = arena.interval(x, one, Closed);
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
        let a = iv(&mut arena, 0, 1, Closed);
        let c = absolute_complement(&mut arena, a);
        assert_eq!(display(&arena, c), "(-oo, 0) ∪ (1, oo)");
    }

    #[test]
    fn simplify_mixed_finite_set() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let one = arena.int(1);
        let fs = arena.finite_set(&[x, one]);
        let a = iv(&mut arena, 0, 2, Closed);
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
        let a = arena.interval(three, pi, Closed);
        let b = arena.interval(pi, four, Open);
        let u = arena.set_union(&[a, b]);
        let s = simplify_set(&mut arena, u);
        assert_eq!(display(&arena, s), "[3, 4)");
    }

    #[test]
    fn membership_queries() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, LeftOpen);
        let zero = arena.zero;
        let one = arena.one;
        let half = arena.rational(1, 2);
        let x = arena.symbol("x");
        assert_eq!(set_contains(&mut arena, a, zero), Truth::False);
        assert_eq!(set_contains(&mut arena, a, one), Truth::True);
        assert_eq!(set_contains(&mut arena, a, half), Truth::True);
        assert_eq!(set_contains(&mut arena, a, x), Truth::Unknown);
        let e = arena.empty_set;
        assert_eq!(set_contains(&mut arena, e, x), Truth::False);
        // symbolic interval, exact difference
        let xp1 = arena.add(&[x, arena.one]);
        let sym = arena.interval(x, xp1, Closed);
        assert_eq!(set_contains(&mut arena, sym, x), Truth::True);
        assert_eq!(set_contains(&mut arena, sym, xp1), Truth::True);
        let two = arena.int(2);
        let xp2 = arena.add(&[x, two]);
        assert_eq!(set_contains(&mut arena, sym, xp2), Truth::False);
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
        assert_eq!(set_contains(&mut arena, cs, five), Truth::True);
        assert_eq!(set_contains(&mut arena, cs, m5), Truth::False);
    }

    #[test]
    fn subset_disjoint_empty() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, Closed);
        let b = iv(&mut arena, 0, 2, Closed);
        let c = iv(&mut arena, 3, 4, Closed);
        assert_eq!(is_subset(&mut arena, a, b), Truth::True);
        assert_eq!(is_subset(&mut arena, b, a), Truth::False);
        assert_eq!(is_disjoint(&mut arena, a, c), Truth::True);
        assert_eq!(is_disjoint(&mut arena, a, b), Truth::False);
        assert_eq!(is_empty(&mut arena, a), Truth::False);
        let e = arena.set_intersection(&[a, c]);
        assert_eq!(is_empty(&mut arena, e), Truth::True);
        let x = arena.symbol("x");
        let one = arena.one;
        let sym = arena.interval(x, one, Closed);
        assert_eq!(is_empty(&mut arena, sym), Truth::Unknown);
        assert_eq!(is_subset(&mut arena, sym, sym), Truth::True);
    }

    #[test]
    fn inf_sup_measure() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, Open);
        let b = iv(&mut arena, 2, 5, Closed);
        let u = arena.set_union(&[a, b]);
        assert_eq!(inf(&mut arena, u), Some(arena.zero));
        assert_eq!(inf(&mut arena, u), Some(arena.zero));
        let five = arena.int(5);
        assert_eq!(sup(&mut arena, u), Some(five));
        let four = arena.int(4);
        assert_eq!(measure(&mut arena, u), Some(four));
        let ray = arena.interval(arena.zero, arena.infinity, Open);
        assert_eq!(measure(&mut arena, ray), Some(arena.infinity));
        assert_eq!(sup(&mut arena, ray), Some(arena.infinity));
        let e = arena.empty_set;
        assert_eq!(inf(&mut arena, e), None);
        assert_eq!(measure(&mut arena, e), Some(arena.zero));
    }

    #[test]
    fn topology_via_arena() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, Open);
        let c = closure(&mut arena, a).unwrap();
        assert_eq!(display(&arena, c), "[0, 1]");
        let i = interior(&mut arena, c).unwrap();
        assert_eq!(i, a);
        let b = boundary(&mut arena, a).unwrap();
        assert_eq!(display(&arena, b), "{0, 1}");
        assert_eq!(is_open(&mut arena, a), Truth::True);
        assert_eq!(is_closed(&mut arena, a), Truth::False);
        assert_eq!(is_closed(&mut arena, c), Truth::True);
    }

    #[test]
    fn accessors() {
        let mut arena = Arena::new();
        let a = iv(&mut arena, 0, 1, Open);
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
        let a = iv(&mut arena, 0, 1, LeftOpen);
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

    /// `{a·n + b}` over ℤ: the canonical form is a fixed point of
    /// `image_set`, with a positive step and the offset in `[0, step)`.
    #[test]
    fn image_set_canonical_form_is_a_fixed_point() {
        let mut arena = Arena::new();
        let k = arena.symbol("k");
        let pi = arena.pi;
        // −2kπ + 5π/2  →  2·_n·π + π/2
        let m2 = arena.int(-2);
        let step = arena.mul(&[m2, k, pi]);
        let five_halves = arena.rational(5, 2);
        let off = arena.mul(&[five_halves, pi]);
        let body = arena.add(&[step, off]);
        let s = image_set(&mut arena, k, body);
        assert_eq!(
            display(&arena, s),
            "ImageSet(Lambda(_n, 2*_n*pi + 1/2*pi), Integers)"
        );
        let ExprNode::ImageSet(n, b) = *arena.node(s) else {
            panic!("{}", display(&arena, s));
        };
        assert_eq!(image_set(&mut arena, n, b), s);
        // the integers themselves: k + 7 → _n
        let seven = arena.int(7);
        let shifted = arena.add(&[k, seven]);
        let z = image_set(&mut arena, k, shifted);
        assert_eq!(display(&arena, z), "ImageSet(Lambda(_n, _n), Integers)");
    }

    /// A union of families merges by residues: `2nπ`, `2nπ + π/2`, `2nπ +
    /// π`, `2nπ + 3π/2` is `nπ/2`; `2nπ`, `2nπ + π`, `2nπ + π/2` is `nπ ∪
    /// 2nπ + π/2`.
    #[test]
    fn unions_of_families_merge_into_coarser_lattices() {
        let mut arena = Arena::new();
        let n = arena.symbol("n");
        let pi = arena.pi;
        let two = arena.int(2);
        let two_pi_n = arena.mul(&[two, pi, n]);
        let fam = |arena: &mut Arena, p: i64, q: i64| {
            let r = arena.rational(p, q);
            let off = arena.mul(&[r, pi]);
            let body = arena.add(&[two_pi_n, off]);
            image_set(arena, n, body)
        };
        let all: Vec<ExprId> = [(0, 1), (1, 2), (1, 1), (3, 2)]
            .iter()
            .map(|&(p, q)| fam(&mut arena, p, q))
            .collect();
        let u = arena.set_union(&all);
        let u = simplify_set(&mut arena, u);
        assert_eq!(
            display(&arena, u),
            "ImageSet(Lambda(_n, 1/2*_n*pi), Integers)"
        );
        let three = [all[0], all[2], all[1]];
        let u = arena.set_union(&three);
        let u = simplify_set(&mut arena, u);
        assert_eq!(
            display(&arena, u),
            "ImageSet(Lambda(_n, _n*pi), Integers) ∪ ImageSet(Lambda(_n, 2*_n*pi + 1/2*pi), Integers)"
        );
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
