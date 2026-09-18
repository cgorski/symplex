//! Pattern matching and rewrite-rule engine.
//!
//! This module provides structural pattern matching against canonical
//! expression trees, plus the rule-based rewriting system that powers
//! `.simplify()`, `.simplify_traced()` and the public
//! [`Ex::rewrite`](crate::api::expr::Ex::rewrite) family.
//!
//! # Design
//!
//! Patterns are descriptions of expression shapes with **wild** (pattern
//! variable) slots that can bind to any sub-expression.  Matching is
//! structural for every node type except `Add` and `Mul`, which are
//! matched **associatively and commutatively**:
//!
//! * a structured pattern term (`sin(a_)^2`) may match *any* subject term;
//! * a bare wild (`a_`) matches a single subject term, except that the
//!   last unassigned bare wild absorbs *all* remaining terms (so `a_ + b_`
//!   matches `x + y + z` with `b_ = y + z`);
//! * a **sequence wild** (name ending in `__`, e.g. `rest__`) absorbs the
//!   remaining terms and may be empty (binding to `0` / `1`);
//! * when a pattern has no wild that can absorb leftovers, a match against
//!   a *subset* of the subject's terms still succeeds at the root of the
//!   rule application; the unmatched terms are re-attached to the
//!   replacement (`3 + sin²x + cos²x → 3 + 1`);
//! * inside a `Mul`, a numeric pattern coefficient `c` matches a numeric
//!   subject coefficient `d` by leaving the factor `d/c` as a leftover
//!   (so `2·sin(a_)·cos(a_)` matches `6·sin x·cos x` with leftover `3`).
//!
//! The associative-commutative search is a backtracking search with a
//! fixed **step budget** ([`MATCH_BUDGET`]) per rule application, so a
//! pathological pattern can never make matching super-linear in practice:
//! when the budget is exhausted the match simply fails.
//!
//! A [`Rule`] pairs a pattern (the LHS) with a template (the RHS) and an
//! optional condition.  [`apply_rules`] walks an expression bottom-up,
//! trying each rule at every node, and returns the rewritten expression
//! plus a trace of which rules fired.

use std::cell::{Cell, RefCell};

use num_traits::{Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::assumptions::Props;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;

// ═══════════════════════════════════════════════════════════════════════════
// WildId — pattern variable identifier
// ═══════════════════════════════════════════════════════════════════════════

/// Identifies a pattern variable ("wild") inside a [`Pattern`].
///
/// Wilds are created via `Arena::wild` (or, for user rules, by naming a
/// symbol with a trailing underscore) and are represented in the
/// expression tree as ordinary `Symbol` nodes.  During matching, a wild
/// binds to whatever sub-expression it is matched against.  If the same
/// `WildId` appears multiple times in a pattern, all occurrences must
/// bind to the **same** `ExprId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WildId(pub(crate) u32);

// ═══════════════════════════════════════════════════════════════════════════
// Substitution — wild bindings
// ═══════════════════════════════════════════════════════════════════════════

/// A mapping from [`WildId`]s to the [`ExprId`]s they matched.
pub type Substitution = FxHashMap<WildId, ExprId>;

// ═══════════════════════════════════════════════════════════════════════════
// Pattern — what to match
// ═══════════════════════════════════════════════════════════════════════════

/// A pattern describes the shape of an expression to match.
///
/// Patterns are built from the same `ExprId` system as regular
/// expressions, but certain "wild" symbols act as pattern variables
/// that can bind to any sub-expression during matching.
///
/// `Add` and `Mul` nodes are matched associatively and commutatively;
/// every other node is matched structurally.  See the module docs for
/// the full matching semantics.
#[derive(Clone, Debug)]
pub struct Pattern {
    /// The root of the pattern expression (built in the arena).
    pub root: ExprId,
    /// Which symbols in the pattern are wilds.
    pub wilds: FxHashMap<ExprId, WildId>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Arena extensions for pattern construction
// ═══════════════════════════════════════════════════════════════════════════

/// Counter for generating unique wild names / ids.
static NEXT_WILD_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Allocate a fresh, process-unique [`WildId`].
pub(crate) fn fresh_wild_id() -> WildId {
    WildId(NEXT_WILD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

impl Arena {
    /// Create a new wild (pattern variable) symbol.
    ///
    /// Returns the `ExprId` of the wild symbol and its `WildId`.
    /// The wild is represented as a regular symbol with a unique
    /// internal name (e.g., `_w0`, `_w1`, ...).
    pub fn wild(&mut self) -> (ExprId, WildId) {
        let wid = fresh_wild_id();
        let name = format!("_w{}", wid.0);
        let expr_id = self.symbol(&name);
        (expr_id, wid)
    }
}

/// Returns `true` if `id` is a symbol whose name ends in a double
/// underscore — the convention for **sequence wilds** that absorb the
/// remaining terms of an `Add`/`Mul` (possibly none).
pub(crate) fn is_sequence_wild_symbol(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Symbol(sid) => arena.symbol_name(*sid).ends_with("__"),
        _ => false,
    }
}

/// Returns `true` if `id` is a symbol whose name ends in an underscore —
/// the naming convention for wilds in user-constructed rules.
pub(crate) fn is_wild_symbol(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Symbol(sid) => {
            let name = arena.symbol_name(*sid);
            name.len() > 1 && name.ends_with('_')
        }
        _ => false,
    }
}

/// Build a [`Pattern`] from an expression by treating every symbol whose
/// name ends in `_` as a wild.
///
/// Returns the pattern and the map `WildId → symbol name`.
pub(crate) fn pattern_from_expr(
    arena: &Arena,
    root: ExprId,
) -> (Pattern, FxHashMap<WildId, String>) {
    let mut wilds = FxHashMap::default();
    let mut names = FxHashMap::default();
    for id in walk::post_order_ids(arena, root) {
        if let ExprNode::Symbol(sid) = arena.node(id)
            && is_wild_symbol(arena, id)
        {
            let wid = fresh_wild_id();
            wilds.insert(id, wid);
            names.insert(wid, arena.symbol_name(*sid).to_string());
        }
    }
    (Pattern { root, wilds }, names)
}

// ═══════════════════════════════════════════════════════════════════════════
// Matching
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum number of elementary matching steps spent on a single rule
/// application (one rule at one node).  The associative-commutative
/// search over `Add`/`Mul` children is a backtracking search; this cap
/// bounds its cost.  When the budget is exhausted the match fails.
pub const MATCH_BUDGET: usize = 4_000;

/// Maximum pattern nesting depth followed by the matcher.
const MAX_PATTERN_DEPTH: usize = 128;

/// The outcome of a successful match.
#[derive(Clone, Debug, Default)]
pub(crate) struct MatchResult {
    /// Wild bindings.
    pub(crate) bindings: Substitution,
    /// Terms of the top-level `Add`/`Mul` subject that were **not**
    /// consumed by the pattern (empty unless the pattern matched a
    /// proper subset of the subject's terms).
    pub(crate) leftover: SmallVec<[ExprId; 6]>,
}

/// Which associative-commutative operator an n-ary match is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AcKind {
    Add,
    Mul,
}

/// Backtracking matcher state shared across the recursive descent.
///
/// The recursion follows the *pattern* (not the subject), so its depth is
/// bounded by the pattern's nesting depth (capped at
/// [`MAX_PATTERN_DEPTH`]); subject expressions of arbitrary size are only
/// ever iterated over.
struct Matcher<'p> {
    pattern: &'p Pattern,
    /// Wild symbols that are sequence wilds.
    seq_wilds: FxHashSet<ExprId>,
    /// Pattern sub-terms that contain at least one wild.
    wild_bearing: FxHashSet<ExprId>,
    /// Remaining step budget.
    budget: Cell<usize>,
    /// Current pattern depth.
    depth: Cell<usize>,
    /// Leftover subject terms of the root `Add`/`Mul` (set just before
    /// the final continuation is invoked).
    leftover: RefCell<SmallVec<[ExprId; 6]>>,
}

/// Continuation type threaded through the backtracking search.
type Cont<'k> = dyn FnMut(&mut Arena, &mut Substitution) -> bool + 'k;

impl<'p> Matcher<'p> {
    fn new(arena: &Arena, pattern: &'p Pattern, budget: usize) -> Self {
        let mut seq_wilds = FxHashSet::default();
        for &w in pattern.wilds.keys() {
            if is_sequence_wild_symbol(arena, w) {
                seq_wilds.insert(w);
            }
        }
        let mut wild_bearing = FxHashSet::default();
        for id in walk::post_order_ids(arena, pattern.root) {
            if pattern.wilds.contains_key(&id) {
                wild_bearing.insert(id);
                continue;
            }
            let mut any = false;
            arena.node(id).for_each_child(|c| {
                if wild_bearing.contains(&c) {
                    any = true;
                }
            });
            if any {
                wild_bearing.insert(id);
            }
        }
        Matcher {
            pattern,
            seq_wilds,
            wild_bearing,
            budget: Cell::new(budget),
            depth: Cell::new(0),
            leftover: RefCell::new(SmallVec::new()),
        }
    }

    /// Consume one unit of budget; returns `false` when exhausted.
    #[inline]
    fn tick(&self) -> bool {
        let b = self.budget.get();
        if b == 0 {
            return false;
        }
        self.budget.set(b - 1);
        true
    }

    /// Match pattern node `pat` against subject `subj`, then invoke `k`.
    fn m(
        &self,
        arena: &mut Arena,
        pat: ExprId,
        subj: ExprId,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        if !self.tick() {
            return false;
        }

        // ── Wild ────────────────────────────────────────────────────
        if let Some(&wid) = self.pattern.wilds.get(&pat) {
            if let Some(&existing) = b.get(&wid) {
                return existing == subj && k(arena, b);
            }
            b.insert(wid, subj);
            let ok = k(arena, b);
            if !ok {
                b.remove(&wid);
            }
            return ok;
        }

        // ── Wild-free pattern sub-term: must be identical (hash-consing). ──
        if !self.wild_bearing.contains(&pat) {
            return pat == subj && k(arena, b);
        }

        let depth = self.depth.get();
        if depth >= MAX_PATTERN_DEPTH {
            return false;
        }
        self.depth.set(depth + 1);
        let result = self.m_structural(arena, pat, subj, b, k);
        self.depth.set(depth);
        result
    }

    /// Structural / AC dispatch for a non-wild pattern node.
    fn m_structural(
        &self,
        arena: &mut Arena,
        pat: ExprId,
        subj: ExprId,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        let pat_node = arena.node(pat).clone();
        let subj_node = arena.node(subj).clone();

        match (&pat_node, &subj_node) {
            (ExprNode::Add(pc), ExprNode::Add(sc)) => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = sc.iter().copied().collect();
                self.m_ac(arena, &pc, &sc, AcKind::Add, false, b, k)
            }
            (ExprNode::Mul(pc), ExprNode::Mul(sc)) => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = sc.iter().copied().collect();
                self.m_ac(arena, &pc, &sc, AcKind::Mul, false, b, k)
            }
            // An Add/Mul pattern that contains a sequence wild may match a
            // single-term subject (the sequence wild binds to 0 / 1).
            (ExprNode::Add(pc), _) if self.has_seq_wild(pc) => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = smallvec::smallvec![subj];
                self.m_ac(arena, &pc, &sc, AcKind::Add, false, b, k)
            }
            (ExprNode::Mul(pc), _) if self.has_seq_wild(pc) => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = smallvec::smallvec![subj];
                self.m_ac(arena, &pc, &sc, AcKind::Mul, false, b, k)
            }
            _ => {
                if !same_head(&pat_node, &subj_node, pat, subj) {
                    return false;
                }
                let pcs = pat_node.children();
                let scs = subj_node.children();
                if pcs.len() != scs.len() {
                    return false;
                }
                self.m_seq(arena, &pcs, &scs, 0, b, k)
            }
        }
    }

    fn has_seq_wild(&self, children: &[ExprId]) -> bool {
        children.iter().any(|c| self.seq_wilds.contains(c))
    }

    /// Match children positionally, threading the continuation.
    fn m_seq(
        &self,
        arena: &mut Arena,
        pcs: &[ExprId],
        scs: &[ExprId],
        i: usize,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        if i == pcs.len() {
            return k(arena, b);
        }
        let mut next =
            |arena: &mut Arena, b: &mut Substitution| self.m_seq(arena, pcs, scs, i + 1, b, k);
        self.m(arena, pcs[i], scs[i], b, &mut next)
    }

    /// Associative-commutative match of pattern children `pc` against
    /// subject children `sc`.
    #[allow(clippy::too_many_arguments)]
    fn m_ac(
        &self,
        arena: &mut Arena,
        pc: &[ExprId],
        sc: &[ExprId],
        kind: AcKind,
        root: bool,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        // ── Classify pattern children ───────────────────────────────
        let mut seq: Option<ExprId> = None;
        let mut plains: SmallVec<[ExprId; 4]> = SmallVec::new();
        let mut structured: SmallVec<[ExprId; 6]> = SmallVec::new();
        for &p in pc {
            if let Some(wid) = self.pattern.wilds.get(&p) {
                if b.contains_key(wid) {
                    structured.push(p);
                } else if seq.is_none() && self.seq_wilds.contains(&p) {
                    seq = Some(p);
                } else {
                    plains.push(p);
                }
            } else {
                structured.push(p);
            }
        }
        // Wild-free structured terms first: they are matched by identity
        // and prune the search fastest.
        structured.sort_by_key(|p| self.wild_bearing.contains(p));

        // ── Virtual subject (Mul coefficient splitting) ─────────────
        let mut subject: SmallVec<[ExprId; 8]> = sc.iter().copied().collect();
        if kind == AcKind::Mul {
            let pat_num = structured
                .iter()
                .copied()
                .find(|&p| matches!(arena.node(p), ExprNode::Num(_)));
            let subj_num_idx = subject
                .iter()
                .position(|&s| matches!(arena.node(s), ExprNode::Num(_)));
            if let (Some(c_id), Some(j)) = (pat_num, subj_num_idx)
                && subject[j] != c_id
            {
                let c = arena.as_num(c_id).cloned();
                let d = arena.as_num(subject[j]).cloned();
                if let (Some(c), Some(d)) = (c, d)
                    && !c.is_zero()
                {
                    let q = d / c;
                    let nid = arena.intern_num(q);
                    let q_id = arena.intern(ExprNode::Num(nid));
                    subject[j] = c_id;
                    subject.push(q_id);
                }
            }
        }

        let n = subject.len();
        if structured.len() + plains.len() > n {
            return false;
        }
        if seq.is_none() && plains.is_empty() && !root && structured.len() != n {
            // Nested AC node without an absorber: must consume everything.
            return false;
        }

        let used = RefCell::new(vec![false; n]);
        let ctx = AcCtx {
            structured: &structured,
            plains: &plains,
            seq,
            subject: &subject,
            used: &used,
            kind,
            root,
        };
        self.assign_structured(arena, &ctx, 0, b, k)
    }

    /// Assign structured pattern term `i` to some unused subject term.
    fn assign_structured(
        &self,
        arena: &mut Arena,
        ctx: &AcCtx<'_>,
        i: usize,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        if i == ctx.structured.len() {
            return self.assign_plains(arena, ctx, 0, b, k);
        }
        if !self.tick() {
            return false;
        }
        let p = ctx.structured[i];
        let n = ctx.subject.len();

        // Wild-free term (or an already-bound bare wild): identity match.
        let fixed_value = if let Some(wid) = self.pattern.wilds.get(&p) {
            b.get(wid).copied()
        } else if !self.wild_bearing.contains(&p) {
            Some(p)
        } else {
            None
        };
        if let Some(value) = fixed_value {
            let slot = (0..n).find(|&j| !ctx.used.borrow()[j] && ctx.subject[j] == value);
            let Some(j) = slot else {
                return false;
            };
            ctx.used.borrow_mut()[j] = true;
            let ok = self.assign_structured(arena, ctx, i + 1, b, k);
            if !ok {
                ctx.used.borrow_mut()[j] = false;
            }
            return ok;
        }

        let p_node = arena.node(p).clone();
        for j in 0..n {
            if ctx.used.borrow()[j] {
                continue;
            }
            let s = ctx.subject[j];
            if !head_compatible(arena, &p_node, p, s) {
                continue;
            }
            ctx.used.borrow_mut()[j] = true;
            let mut next = |arena: &mut Arena, b: &mut Substitution| {
                self.assign_structured(arena, ctx, i + 1, b, k)
            };
            let ok = self.m(arena, p, s, b, &mut next);
            if ok {
                return true;
            }
            ctx.used.borrow_mut()[j] = false;
            if self.budget.get() == 0 {
                return false;
            }
        }
        false
    }

    /// Assign bare (plain) wild `i`.
    fn assign_plains(
        &self,
        arena: &mut Arena,
        ctx: &AcCtx<'_>,
        i: usize,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        if i == ctx.plains.len() {
            return self.finish(arena, ctx, b, k);
        }
        if !self.tick() {
            return false;
        }
        let p = ctx.plains[i];
        let wid = self.pattern.wilds[&p];
        let n = ctx.subject.len();

        // Bound in the meantime (by a structured sibling)?  Identity match.
        if let Some(value) = b.get(&wid).copied() {
            let slot = (0..n).find(|&j| !ctx.used.borrow()[j] && ctx.subject[j] == value);
            let Some(j) = slot else {
                return false;
            };
            ctx.used.borrow_mut()[j] = true;
            let ok = self.assign_plains(arena, ctx, i + 1, b, k);
            if !ok {
                ctx.used.borrow_mut()[j] = false;
            }
            return ok;
        }

        let is_last = i + 1 == ctx.plains.len();
        if is_last && ctx.seq.is_none() {
            // The last plain wild absorbs every remaining term.
            let rest: SmallVec<[ExprId; 6]> = (0..n)
                .filter(|&j| !ctx.used.borrow()[j])
                .map(|j| ctx.subject[j])
                .collect();
            if rest.is_empty() {
                return false;
            }
            let value = combine(arena, ctx.kind, &rest);
            let snapshot = ctx.used.borrow().clone();
            for j in 0..n {
                ctx.used.borrow_mut()[j] = true;
            }
            b.insert(wid, value);
            let ok = self.assign_plains(arena, ctx, i + 1, b, k);
            if !ok {
                b.remove(&wid);
                *ctx.used.borrow_mut() = snapshot;
            }
            return ok;
        }

        // Otherwise bind to a single unused term (try each).
        for j in 0..n {
            if ctx.used.borrow()[j] {
                continue;
            }
            ctx.used.borrow_mut()[j] = true;
            b.insert(wid, ctx.subject[j]);
            let ok = self.assign_plains(arena, ctx, i + 1, b, k);
            if ok {
                return true;
            }
            b.remove(&wid);
            ctx.used.borrow_mut()[j] = false;
            if self.budget.get() == 0 {
                return false;
            }
        }
        false
    }

    /// All pattern children are assigned: handle the sequence wild /
    /// leftovers and invoke the continuation.
    fn finish(
        &self,
        arena: &mut Arena,
        ctx: &AcCtx<'_>,
        b: &mut Substitution,
        k: &mut Cont<'_>,
    ) -> bool {
        let n = ctx.subject.len();
        let rest: SmallVec<[ExprId; 6]> = (0..n)
            .filter(|&j| !ctx.used.borrow()[j])
            .map(|j| ctx.subject[j])
            .collect();

        if let Some(seq) = ctx.seq {
            let wid = self.pattern.wilds[&seq];
            let value = combine(arena, ctx.kind, &rest);
            if let Some(&existing) = b.get(&wid) {
                return existing == value && k(arena, b);
            }
            b.insert(wid, value);
            let ok = k(arena, b);
            if !ok {
                b.remove(&wid);
            }
            return ok;
        }

        if rest.is_empty() {
            return k(arena, b);
        }
        if !ctx.root {
            return false;
        }
        // Partial (sub-expression) match at the root.
        let saved = std::mem::replace(&mut *self.leftover.borrow_mut(), rest);
        let ok = k(arena, b);
        if !ok {
            *self.leftover.borrow_mut() = saved;
        }
        ok
    }
}

/// Shared read-only state of one AC-matching level.
struct AcCtx<'a> {
    structured: &'a [ExprId],
    plains: &'a [ExprId],
    seq: Option<ExprId>,
    subject: &'a [ExprId],
    used: &'a RefCell<Vec<bool>>,
    kind: AcKind,
    root: bool,
}

/// Combine subject terms with the AC operator (identity for none).
fn combine(arena: &mut Arena, kind: AcKind, terms: &[ExprId]) -> ExprId {
    match (kind, terms.len()) {
        (AcKind::Add, 0) => arena.zero,
        (AcKind::Mul, 0) => arena.one,
        (_, 1) => terms[0],
        (AcKind::Add, _) => arena.add(terms),
        (AcKind::Mul, _) => arena.mul(terms),
    }
}

/// Cheap pre-filter: can pattern node `p` possibly match subject `s`?
fn head_compatible(arena: &Arena, p_node: &ExprNode, p: ExprId, s: ExprId) -> bool {
    let s_node = arena.node(s);
    match p_node {
        // Add/Mul patterns may also match single terms when they carry a
        // sequence wild; be permissive and let `m` decide.
        ExprNode::Add(_) | ExprNode::Mul(_) => true,
        _ => same_head(p_node, s_node, p, s),
    }
}

/// Structural head equality: same variant, and identical payload for
/// atoms / nodes with non-child data.
fn same_head(p_node: &ExprNode, s_node: &ExprNode, p: ExprId, s: ExprId) -> bool {
    if std::mem::discriminant(p_node) != std::mem::discriminant(s_node) {
        return false;
    }
    if p_node.is_atom() {
        return p == s;
    }
    match (p_node, s_node) {
        (ExprNode::Apply(pf, _), ExprNode::Apply(sf, _)) => pf == sf,
        (ExprNode::Interval(_, _, pf), ExprNode::Interval(_, _, sf)) => pf == sf,
        _ => true,
    }
}

/// Try to match `pattern` against `expr`, returning bindings for wilds
/// on success.
///
/// This is a *whole-node* match: `Add`/`Mul` subjects must be consumed
/// entirely (no leftover terms).  Returns `None` if the pattern does not
/// match.
#[cfg(test)]
pub(crate) fn match_pattern(
    arena: &mut Arena,
    pattern: &Pattern,
    expr: ExprId,
) -> Option<Substitution> {
    let results = match_all(arena, pattern, expr, false, 1, MATCH_BUDGET);
    results.into_iter().next().map(|r| r.bindings)
}

/// Enumerate up to `max_results` matches of `pattern` against `expr`.
///
/// When `allow_partial` is `true` and the pattern root is an `Add`/`Mul`,
/// the pattern may match a subset of the subject's terms; the unmatched
/// terms are reported in [`MatchResult::leftover`].
pub(crate) fn match_all(
    arena: &mut Arena,
    pattern: &Pattern,
    expr: ExprId,
    allow_partial: bool,
    max_results: usize,
    budget: usize,
) -> Vec<MatchResult> {
    let matcher = Matcher::new(arena, pattern, budget);
    let mut results: Vec<MatchResult> = Vec::new();
    let mut bindings = Substitution::default();

    let pat = pattern.root;
    let root_is_wild = pattern.wilds.contains_key(&pat);
    let pat_node = arena.node(pat).clone();
    let subj_node = arena.node(expr).clone();

    {
        let matcher_ref = &matcher;
        let results_ref = &mut results;
        let mut accept = |_arena: &mut Arena, b: &mut Substitution| -> bool {
            results_ref.push(MatchResult {
                bindings: b.clone(),
                leftover: matcher_ref.leftover.borrow().clone(),
            });
            results_ref.len() >= max_results
        };

        match (&pat_node, &subj_node) {
            (ExprNode::Add(pc), ExprNode::Add(sc)) if !root_is_wild => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = sc.iter().copied().collect();
                matcher.m_ac(
                    arena,
                    &pc,
                    &sc,
                    AcKind::Add,
                    allow_partial,
                    &mut bindings,
                    &mut accept,
                );
            }
            (ExprNode::Mul(pc), ExprNode::Mul(sc)) if !root_is_wild => {
                let pc: SmallVec<[ExprId; 6]> = pc.iter().copied().collect();
                let sc: SmallVec<[ExprId; 8]> = sc.iter().copied().collect();
                matcher.m_ac(
                    arena,
                    &pc,
                    &sc,
                    AcKind::Mul,
                    allow_partial,
                    &mut bindings,
                    &mut accept,
                );
            }
            _ => {
                matcher.m(arena, pat, expr, &mut bindings, &mut accept);
            }
        }
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Template instantiation
// ═══════════════════════════════════════════════════════════════════════════

/// Instantiate a template expression by replacing wilds with their
/// bound values from a [`Substitution`].
///
/// This is used after a successful match to build the replacement
/// expression from the RHS template of a rule.
pub(crate) fn instantiate(
    arena: &mut Arena,
    template: ExprId,
    wilds: &FxHashMap<ExprId, WildId>,
    bindings: &Substitution,
) -> ExprId {
    // Build a replacement map: wild_expr_id → bound_value.
    let replacements: FxHashMap<ExprId, ExprId> = wilds
        .iter()
        .filter_map(|(&expr_id, &wid)| bindings.get(&wid).map(|&val| (expr_id, val)))
        .collect();

    if replacements.is_empty() {
        return template;
    }

    walk::walk_and_rebuild(arena, template, &|_arena, id| {
        replacements.get(&id).copied()
    })
}

/// Size of the expression *tree* rooted at `root` (nodes counted with
/// multiplicity, unlike the de-duplicated DAG walk used by `count_ops`),
/// saturating at `cap`.
///
/// Hash-consing keeps the DAG small even when the unfolded tree grows
/// exponentially, so rewrite-growth guards must look at the tree size.
/// Computed by a single bottom-up pass over the DAG with memoisation.
pub(crate) fn tree_size_capped(arena: &Arena, root: ExprId, cap: usize) -> usize {
    let post_order = walk::post_order_ids(arena, root);
    let mut size: FxHashMap<ExprId, usize> = FxHashMap::default();
    for &id in &post_order {
        let mut total = 1usize;
        arena.node(id).for_each_child(|c| {
            total = total.saturating_add(size.get(&c).copied().unwrap_or(1));
        });
        size.insert(id, total.min(cap));
    }
    size.get(&root).copied().unwrap_or(1)
}

/// Re-attach leftover terms of a partial `Add`/`Mul` match to the
/// replacement produced by a rule.
pub(crate) fn reattach_leftover(
    arena: &mut Arena,
    subject: ExprId,
    replacement: ExprId,
    leftover: &[ExprId],
) -> ExprId {
    if leftover.is_empty() {
        return replacement;
    }
    let mut all: SmallVec<[ExprId; 8]> = leftover.iter().copied().collect();
    all.push(replacement);
    match arena.node(subject) {
        ExprNode::Mul(_) => arena.mul(&all),
        _ => arena.add(&all),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rule — named rewrite rule
// ═══════════════════════════════════════════════════════════════════════════

/// A named rewrite rule: pattern (LHS) → template (RHS).
///
/// Optionally carries a condition function that is checked after the
/// pattern matches but before the replacement is applied.
///
/// This is the arena-level rule produced by the [`rule!`](crate::rule)
/// macro.  The user-facing, context-aware rule type is
/// [`Rule`](crate::api::expr_rules_ext::Rule) in the public API; convert
/// with [`RuleSet::from_macro_rules`](crate::api::expr_rules_ext::RuleSet::from_macro_rules).
#[derive(Clone)]
pub struct Rule {
    /// Human-readable name for tracing.
    pub name: &'static str,
    /// The pattern to match (LHS).
    pub pattern: Pattern,
    /// The template to instantiate (RHS).
    pub template: ExprId,
    /// Optional condition.  Called with the arena and the bindings
    /// after a successful structural match.  If it returns `false`,
    /// the match is rejected.
    pub condition: Option<fn(&Arena, &Substitution) -> bool>,
}

impl Rule {
    /// Create a rule without a condition.
    pub fn new(name: &'static str, pattern: Pattern, template: ExprId) -> Self {
        Rule {
            name,
            pattern,
            template,
            condition: None,
        }
    }

    /// Try to apply this rule at `expr`, optionally allowing a partial
    /// match of an `Add`/`Mul` subject (leftover terms are re-attached).
    ///
    /// Returns `Some(replacement)` if the rule matched and the condition
    /// (if any) was satisfied; `None` otherwise.
    pub(crate) fn try_apply_at(
        &self,
        arena: &mut Arena,
        expr: ExprId,
        allow_partial: bool,
    ) -> Option<ExprId> {
        let max_results = if self.condition.is_some() { 8 } else { 1 };
        let matches = match_all(
            arena,
            &self.pattern,
            expr,
            allow_partial,
            max_results,
            MATCH_BUDGET,
        );
        for m in matches {
            if let Some(cond) = self.condition
                && !cond(arena, &m.bindings)
            {
                continue;
            }
            let replacement = instantiate(arena, self.template, &self.pattern.wilds, &m.bindings);
            return Some(reattach_leftover(arena, expr, replacement, &m.leftover));
        }
        None
    }
}

impl std::fmt::Debug for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rule").field("name", &self.name).finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RawStep — arena-level trace entry
// ═══════════════════════════════════════════════════════════════════════════

/// A record of one rewrite step at the arena level.
///
/// Produced by [`apply_rules`] and the simplification engine; converted
/// to the user-facing [`Step`] by the `Ex` layer.
#[derive(Clone, Debug)]
pub struct RawStep {
    /// The name of the rule (or simplification strategy) that fired.
    pub rule_name: String,
    /// The sub-expression before the rule was applied.
    pub before: ExprId,
    /// The sub-expression after the rule was applied.
    pub after: ExprId,
}

// ═══════════════════════════════════════════════════════════════════════════
// apply_rules — bottom-up rewriting with tracing
// ═══════════════════════════════════════════════════════════════════════════

/// Apply a list of rules to an expression in a single bottom-up pass.
///
/// At each node (processed leaves-first), the rules are tried in order.
/// The first rule that matches is applied, and the rewritten node
/// replaces the original.  `Add`/`Mul` nodes may be matched partially
/// (the unmatched terms are re-attached).  The pass is NOT iterated —
/// for fixpoint rewriting, call this in a loop until no more steps are
/// produced.
///
/// Returns the rewritten expression and a trace of all steps.
pub(crate) fn apply_rules(
    arena: &mut Arena,
    expr: ExprId,
    rules: &[Rule],
) -> (ExprId, Vec<RawStep>) {
    let mut steps: Vec<RawStep> = Vec::new();
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    let post_order = walk::post_order_ids(arena, expr);

    for &id in &post_order {
        // First, rebuild with any already-rewritten children.
        let rebuilt = if arena.node(id).is_atom() {
            id
        } else {
            walk::rebuild_with_cache(arena, id, &cache)
        };

        // Try each rule; first match wins.
        let mut rewritten = rebuilt;
        for rule in rules {
            if let Some(replacement) = rule.try_apply_at(arena, rewritten, true)
                && replacement != rewritten
            {
                tracing::debug!(rule = rule.name, "rule fired");
                steps.push(RawStep {
                    rule_name: rule.name.to_string(),
                    before: rebuilt,
                    after: replacement,
                });
                rewritten = replacement;
                break;
            }
        }

        cache.insert(id, rewritten);
    }

    let result = cache.get(&expr).copied().unwrap_or(expr);
    (result, steps)
}

// ═══════════════════════════════════════════════════════════════════════════
// Algebraic substitution
// ═══════════════════════════════════════════════════════════════════════════

use num_bigint::BigInt;
use num_rational::Ratio;

/// How the `old` expression of an algebraic substitution is decomposed.
enum OldShape {
    /// `base^exp` (also `exp(u)` as `E^u`, and any other expression as
    /// `old^1`).
    Power { base: ExprId, exp: ExprId },
    /// `coeff · Π base_i^exp_i` with at least two symbolic factors.
    Product {
        coeff: Ratio<BigInt>,
        factors: Vec<(ExprId, ExprId)>,
    },
    /// A sum with at least one symbolic term.
    Sum,
}

/// Decompose `id` as `(base, exponent)`, treating `exp(u)` as `E^u`.
fn base_exp_ext(arena: &Arena, id: ExprId) -> (ExprId, ExprId) {
    match arena.node(id) {
        ExprNode::Pow(b, e) => (*b, *e),
        ExprNode::Exp(u) => (arena.e_const(), *u),
        _ => (id, arena.one),
    }
}

/// `base^exp`, producing `exp(…)` for base `e` (and `1` for a zero exponent).
fn make_pow_ext(arena: &mut Arena, base: ExprId, exp: ExprId) -> ExprId {
    if exp == arena.zero {
        arena.one
    } else if base == arena.e_const() {
        arena.exp(exp)
    } else {
        arena.pow(base, exp)
    }
}

/// Integer `k` (truncated toward zero, never `0`) such that `f` "contains"
/// `k·e`: for numeric `f`, `e` this is `trunc(f/e)`; otherwise a term of
/// `f` with the same symbolic part as `e` supplies the ratio.
fn integer_multiple(arena: &mut Arena, f: ExprId, e: ExprId) -> Option<BigInt> {
    let ratio = |cf: &Ratio<BigInt>, ce: &Ratio<BigInt>| -> Option<BigInt> {
        if ce.is_zero() {
            return None;
        }
        let q = cf / ce;
        let k = q.trunc().to_integer();
        if k.is_zero() { None } else { Some(k) }
    };
    if let (Some(fv), Some(ev)) = (arena.as_num(f).cloned(), arena.as_num(e).cloned()) {
        return ratio(&fv, &ev);
    }
    let (ce, te) = arena.as_coeff_term(e);
    let terms: SmallVec<[ExprId; 6]> = match arena.node(f) {
        ExprNode::Add(c) => c.iter().copied().collect(),
        _ => smallvec::smallvec![f],
    };
    for t in terms {
        let (cf, tf) = arena.as_coeff_term(t);
        if tf == te {
            return ratio(&cf, &ce);
        }
    }
    None
}

/// Algebraic substitution `old → new` in `expr`.
///
/// Unlike structural substitution, this recognises `old` inside powers,
/// products and sums:
///
/// * **powers** — `old = b^e`: every `b^f` with `f = k·e + r` (integer
///   `k ≠ 0`) becomes `new^k · b^r`; `exp(u)` is treated as `e^u`
///   (`x^4 → y^2`, `x^3 → x·y`, `1/x^2 → 1/y`, `exp(2x) → t^2` for
///   `old = exp(x)`);
/// * **products** — `old = c·Π b_i^e_i`: a product containing every
///   `b_i^(k·e_i + r_i)` with a common integer `k ≠ 0` (same sign for all
///   factors) becomes `new^k · (rest)` with the numeric coefficient
///   divided by `c^k` (`2xyz → 2wz` for `old = xy`);
/// * **sums** — `old = Σ c_i t_i`: a sum containing every symbolic term
///   `t_i` with coefficients `k·c_i` for a common rational `k ≠ 0`
///   becomes `k·new + (sum − k·old)` (`a + b + c → c + d` for
///   `old = a + b`);
/// * exact occurrences of `old` are replaced as in structural substitution.
///
/// Every rewrite is an identity in `old`: substituting `old` back for
/// `new` recovers a value-equal expression.  The walk is bottom-up.
pub(crate) fn subs_algebraic(arena: &mut Arena, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
    if old == new {
        return expr;
    }
    let shape = classify_old(arena, old);
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        // An exact occurrence is replaced before its children are rebuilt
        // (otherwise `√x → y` would first turn the inner `x` into `y²`).
        if id == old {
            cache.insert(id, new);
            continue;
        }
        let rebuilt = if arena.node(id).is_atom() {
            id
        } else {
            walk::rebuild_with_cache(arena, id, &cache)
        };
        let result = if rebuilt == old {
            new
        } else {
            match &shape {
                OldShape::Power { base, exp } => subs_power(arena, rebuilt, *base, *exp, new),
                OldShape::Product { coeff, factors } => {
                    subs_product(arena, rebuilt, coeff, factors, new)
                }
                OldShape::Sum => subs_sum(arena, rebuilt, old, new),
            }
            .unwrap_or(rebuilt)
        };
        cache.insert(id, result);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

fn classify_old(arena: &mut Arena, old: ExprId) -> OldShape {
    match arena.node(old).clone() {
        ExprNode::Add(_) => OldShape::Sum,
        ExprNode::Mul(_) => {
            let (coeff, term) = arena.as_coeff_term(old);
            let symbolic: SmallVec<[ExprId; 6]> = match arena.node(term) {
                ExprNode::Mul(c) => c.iter().copied().collect(),
                _ => smallvec::smallvec![term],
            };
            if symbolic.len() >= 2 {
                let factors = symbolic.iter().map(|&c| base_exp_ext(arena, c)).collect();
                OldShape::Product { coeff, factors }
            } else {
                // `2·x` style (numeric coefficient times one factor): only
                // exact occurrences are replaced.
                let (base, exp) = base_exp_ext(arena, old);
                OldShape::Power { base, exp }
            }
        }
        _ => {
            let (base, exp) = base_exp_ext(arena, old);
            OldShape::Power { base, exp }
        }
    }
}

/// Power case: `node = base^f` with `f = k·e + r`.
fn subs_power(
    arena: &mut Arena,
    node: ExprId,
    base: ExprId,
    e: ExprId,
    new: ExprId,
) -> Option<ExprId> {
    let (b, f) = base_exp_ext(arena, node);
    if b != base {
        return None;
    }
    let k = integer_multiple(arena, f, e)?;
    let k_id = arena.big_int(k);
    let ke = arena.mul(&[k_id, e]);
    let r = arena.sub(f, ke);
    let new_k = arena.pow(new, k_id);
    let rest = make_pow_ext(arena, base, r);
    Some(arena.mul(&[new_k, rest]))
}

/// Product case.
fn subs_product(
    arena: &mut Arena,
    node: ExprId,
    old_coeff: &Ratio<BigInt>,
    old_factors: &[(ExprId, ExprId)],
    new: ExprId,
) -> Option<ExprId> {
    if !matches!(arena.node(node), ExprNode::Mul(_)) {
        return None;
    }
    let (coeff, term) = arena.as_coeff_term(node);
    let children: SmallVec<[ExprId; 6]> = match arena.node(term) {
        ExprNode::Mul(c) => c.iter().copied().collect(),
        _ => smallvec::smallvec![term],
    };
    // base → (index, exponent) of the target's factors
    let mut target: FxHashMap<ExprId, (usize, ExprId)> = FxHashMap::default();
    for (i, &c) in children.iter().enumerate() {
        let (b, f) = base_exp_ext(arena, c);
        target.insert(b, (i, f));
    }

    // Common integer multiple k across all old factors.
    let mut k: Option<BigInt> = None;
    for &(b, e) in old_factors {
        let &(_, f) = target.get(&b)?;
        let kb = if f == e {
            BigInt::from(1)
        } else {
            integer_multiple(arena, f, e)?
        };
        k = Some(match k {
            None => kb,
            Some(prev) => {
                if prev.is_positive() != kb.is_positive() {
                    return None;
                }
                if kb.abs() < prev.abs() { kb } else { prev }
            }
        });
    }
    let k = k?;
    let k_id = arena.big_int(k.clone());

    // Rebuild: reduced old factors + untouched factors + coefficient.
    let mut used: FxHashSet<usize> = FxHashSet::default();
    let mut parts: SmallVec<[ExprId; 8]> = SmallVec::new();
    for &(b, e) in old_factors {
        let &(i, f) = target.get(&b)?;
        used.insert(i);
        let ke = arena.mul(&[k_id, e]);
        let r = arena.sub(f, ke);
        parts.push(make_pow_ext(arena, b, r));
    }
    for (i, &c) in children.iter().enumerate() {
        if !used.contains(&i) {
            parts.push(c);
        }
    }
    // coefficient / old_coeff^k
    let k_i64: i64 = k.clone().try_into().ok()?;
    let old_pow = pow_ratio(old_coeff, k_i64)?;
    let new_coeff = coeff / old_pow;
    parts.push(arena.make_coeff_term(new_coeff, arena.one));
    parts.push(arena.pow(new, k_id));
    Some(arena.mul(&parts))
}

/// `r^n` for integer `n` (returns `None` for `0^negative`).
fn pow_ratio(r: &Ratio<BigInt>, n: i64) -> Option<Ratio<BigInt>> {
    if n == 0 {
        return Some(Ratio::from_integer(BigInt::from(1)));
    }
    if r.is_zero() {
        return if n > 0 { Some(r.clone()) } else { None };
    }
    let base = if n > 0 {
        r.clone()
    } else {
        Ratio::new(r.denom().clone(), r.numer().clone())
    };
    let mut acc = Ratio::from_integer(BigInt::from(1));
    for _ in 0..n.unsigned_abs() {
        acc *= &base;
    }
    Some(acc)
}

/// Sum case: `node = k·old + rest`.
fn subs_sum(arena: &mut Arena, node: ExprId, old: ExprId, new: ExprId) -> Option<ExprId> {
    let node_terms: SmallVec<[ExprId; 6]> = match arena.node(node) {
        ExprNode::Add(c) => c.iter().copied().collect(),
        _ => return None,
    };
    let old_terms: SmallVec<[ExprId; 6]> = match arena.node(old) {
        ExprNode::Add(c) => c.iter().copied().collect(),
        _ => return None,
    };
    // symbolic term → coefficient in the target
    let mut target: FxHashMap<ExprId, Ratio<BigInt>> = FxHashMap::default();
    for &t in &node_terms {
        let (c, term) = arena.as_coeff_term(t);
        target.insert(term, c);
    }
    let mut k: Option<Ratio<BigInt>> = None;
    let mut any_symbolic = false;
    for &t in &old_terms {
        let (c, term) = arena.as_coeff_term(t);
        if term == arena.one {
            continue; // constant term handled by the remainder
        }
        any_symbolic = true;
        let d = target.get(&term)?;
        let ratio = d / &c;
        match &k {
            None => k = Some(ratio),
            Some(prev) if *prev == ratio => {}
            Some(_) => return None,
        }
    }
    if !any_symbolic {
        return None;
    }
    let k = k?;
    if k.is_zero() {
        return None;
    }
    let k_old = arena.make_coeff_term(k.clone(), old);
    let remainder = arena.sub(node, k_old);
    let k_new = arena.make_coeff_term(k, new);
    Some(arena.add(&[k_new, remainder]))
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in rules
// ═══════════════════════════════════════════════════════════════════════════

/// Build the Pythagorean identity rule: `sin²(w) + cos²(w) → 1`.
///
/// Returns the rule.  The pattern and template are interned in the
/// given arena.
pub(crate) fn rule_pythagorean(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);

    // Pattern: sin(w)^2 + cos(w)^2
    // Build: Pow(Sin(w), 2) and Pow(Cos(w), 2)
    let sin_w = arena.sin(w_expr);
    let cos_w = arena.cos(w_expr);
    let sin_sq = arena.pow(sin_w, two);
    let cos_sq = arena.pow(cos_w, two);
    let pattern_expr = arena.add(&[sin_sq, cos_sq]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);

    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };

    Rule::new("pythagorean", pattern, arena.one)
}

/// Helper: build a rule matching `outer(inner(w)) → template(w)`.
fn unary_compose_rule(
    arena: &mut Arena,
    name: &'static str,
    outer_fn: fn(&mut Arena, ExprId) -> ExprId,
    inner_fn: fn(&mut Arena, ExprId) -> ExprId,
    template_fn: fn(&mut Arena, ExprId) -> ExprId,
) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let inner_w = inner_fn(arena, w_expr);
    let pattern_expr = outer_fn(arena, inner_w);
    let template = template_fn(arena, w_expr);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    Rule::new(
        name,
        Pattern {
            root: pattern_expr,
            wilds,
        },
        template,
    )
}

/// Identity template: just returns the wild itself.
fn identity(_arena: &mut Arena, w: ExprId) -> ExprId {
    w
}

/// Build the rule: `sqrt(w^2) → abs(w)`.
///
/// Valid for real `w` (`√(w²) = |w|`).  For non-real `w` the principal
/// square root gives `±w`, not `|w|`, so the rule is gated on `w` not
/// being known non-real (see [`condition_wild_not_known_nonreal`]).
fn rule_sqrt_sq(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);
    let half = arena.rational(1, 2);
    let w_sq = arena.pow(w_expr, two);
    let sqrt_w_sq = arena.pow(w_sq, half); // Pow(Pow(w, 2), 1/2)
    let abs_w = arena.abs(w_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: sqrt_w_sq,
        wilds,
    };
    let mut r = Rule::new("sqrt_sq", pattern, abs_w);
    r.condition = Some(condition_wild_not_known_nonreal);
    r
}

/// Build the hyperbolic Pythagorean identity: `cosh(w)^2 - sinh(w)^2 → 1`.
fn rule_cosh_sinh_identity(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let two = arena.int(2);
    let sinh_w = arena.sinh(w_expr);
    let cosh_w = arena.cosh(w_expr);
    let sinh_sq = arena.pow(sinh_w, two);
    let cosh_sq = arena.pow(cosh_w, two);
    let neg_sinh_sq = arena.neg(sinh_sq);
    let pattern_expr = arena.add(&[cosh_sq, neg_sinh_sq]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("cosh_sinh_identity", pattern, arena.one)
}

/// Power of power: `(w1^w2)^w3 → w1^(w2*w3)`.
///
/// Note: strictly valid for positive bases or integer exponents.
/// May produce incorrect results for negative bases with fractional exponents.
fn rule_pow_pow(arena: &mut Arena) -> Rule {
    let (w1_expr, w1_id) = arena.wild();
    let (w2_expr, w2_id) = arena.wild();
    let (w3_expr, w3_id) = arena.wild();

    let inner_pow = arena.pow(w1_expr, w2_expr);
    let outer_pow = arena.pow(inner_pow, w3_expr);

    let product = arena.mul(&[w2_expr, w3_expr]);
    let template = arena.pow(w1_expr, product);

    let mut wilds = FxHashMap::default();
    wilds.insert(w1_expr, w1_id);
    wilds.insert(w2_expr, w2_id);
    wilds.insert(w3_expr, w3_id);
    let pattern = Pattern {
        root: outer_pow,
        wilds,
    };
    let mut r = Rule::new("pow_pow", pattern, template);
    r.condition = Some(|arena, bindings| {
        // Only fire when at least one bound value is a known integer
        for &val in bindings.values() {
            if let crate::base::node::ExprNode::Num(nid) = arena.node(val) {
                let r = arena.num(*nid);
                if r.is_integer() {
                    return true;
                }
            }
        }
        false
    });
    r
}

/// Build a basic set of simplification rules.
/// sin(w) / cos(w) → tan(w)
/// Canonical form of sin(x)/cos(x) is Mul([Sin(x), Pow(Cos(x), -1)])
fn rule_sin_div_cos(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let sin_w = arena.sin(w_expr);
    let cos_w = arena.cos(w_expr);
    let neg_one = arena.int(-1);
    let cos_w_inv = arena.pow(cos_w, neg_one);
    let pattern_expr = arena.mul(&[sin_w, cos_w_inv]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("sin_div_cos", pattern, arena.tan(w_expr))
}

/// cos(w) / sin(w) → 1/tan(w)  (i.e., Pow(tan(w), -1))
fn rule_cos_div_sin(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let cos_w = arena.cos(w_expr);
    let sin_w = arena.sin(w_expr);
    let neg_one = arena.int(-1);
    let sin_w_inv = arena.pow(sin_w, neg_one);
    let pattern_expr = arena.mul(&[cos_w, sin_w_inv]);
    let tan_w = arena.tan(w_expr);
    let template = arena.pow(tan_w, neg_one);
    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("cos_div_sin", pattern, template)
}

/// sinh(w) / cosh(w) → tanh(w)
fn rule_sinh_div_cosh(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let sinh_w = arena.sinh(w_expr);
    let cosh_w = arena.cosh(w_expr);
    let neg_one = arena.int(-1);
    let cosh_w_inv = arena.pow(cosh_w, neg_one);
    let pattern_expr = arena.mul(&[sinh_w, cosh_w_inv]);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("sinh_div_cosh", pattern, arena.tanh(w_expr))
}

/// exp(a) * exp(b) → exp(a + b)
fn rule_exp_mul(arena: &mut Arena) -> Rule {
    let (a_expr, a_id) = arena.wild();
    let (b_expr, b_id) = arena.wild();
    let exp_a = arena.exp(a_expr);
    let exp_b = arena.exp(b_expr);
    let pattern_expr = arena.mul(&[exp_a, exp_b]);

    let sum = arena.add(&[a_expr, b_expr]);
    let template = arena.exp(sum);

    let mut wilds = FxHashMap::default();
    wilds.insert(a_expr, a_id);
    wilds.insert(b_expr, b_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("exp_mul", pattern, template)
}

/// exp(a * ln(b)) → b^a  (exp-log denesting)
///
/// Only matches when the argument to exp is a 2-child Mul where one
/// child is ln(something). For 3+ child Mul (like exp(2*x*ln(y))),
/// the pattern system's strict positional matching won't fire.
fn rule_exp_log_denest(arena: &mut Arena) -> Rule {
    let (a_expr, a_id) = arena.wild();
    let (b_expr, b_id) = arena.wild();
    let ln_b = arena.ln(b_expr);
    let product = arena.mul(&[a_expr, ln_b]);
    let pattern_expr = arena.exp(product);
    let template = arena.pow(b_expr, a_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(a_expr, a_id);
    wilds.insert(b_expr, b_id);
    let pattern = Pattern {
        root: pattern_expr,
        wilds,
    };
    Rule::new("exp_log_denest", pattern, template)
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditional rule helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Condition: every wild-bound value in the substitution is a numeric
/// literal whose value is strictly positive.  Non-numeric bindings
/// (symbols, compound expressions) cause the condition to return `false`
/// because we cannot determine their sign structurally.
fn condition_wild_positive(arena: &Arena, subs: &Substitution) -> bool {
    for &id in subs.values() {
        if let ExprNode::Num(nid) = arena.node(id) {
            if !arena.num(*nid).is_positive() {
                return false;
            }
        } else {
            // Non-numeric: sign unknown → reject.
            return false;
        }
    }
    true
}

/// Condition: every wild-bound value in the substitution is known to be real.
///
/// Numeric literals are always real.  Symbols are checked via their
/// stored assumptions.  The constants π and *e* are real.  Anything
/// else (compound expressions, imaginary unit, etc.) is conservatively
/// rejected.
#[allow(dead_code)]
fn condition_wild_real(arena: &Arena, subs: &Substitution) -> bool {
    for &id in subs.values() {
        match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => {
                // These are unconditionally real.
            }
            ExprNode::Symbol(sid) => {
                let assumptions = arena.symbol_assumptions(*sid);
                if assumptions.query(Props::REAL) != Some(true) {
                    return false;
                }
            }
            _ => {
                // Compound or unknown expression: cannot confirm real → reject.
                return false;
            }
        }
    }
    true
}

/// Condition: every wild-bound value is *not known* to be non-real.
///
/// This is a weaker gate than [`condition_wild_real`]: it fires unless
/// we have positive evidence that the value is **not** real (e.g. the
/// symbol has an `Imaginary` assumption, or the expression is `i·x` for
/// real `x`).  Unknown assumptions are treated as "probably real", which
/// matches the common case where users don't annotate variables.
///
/// # Branch reasoning
///
/// `ln(exp(x)) = x` holds exactly when `Im(x) ∈ (−π, π]`, which is
/// guaranteed for real `x`.  For `x` known to be non-real the principal
/// branch of `ln` may differ from `x` by a multiple of `2πi`, so the rule
/// must not fire.  Compound expressions are checked through the
/// assumption system (`Props::REAL == Some(false)` blocks the rewrite).
pub(crate) fn condition_wild_not_known_nonreal(arena: &Arena, subs: &Substitution) -> bool {
    for &id in subs.values() {
        match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Pi | ExprNode::E => {
                // These are unconditionally real.
            }
            ExprNode::ImaginaryUnit => return false,
            ExprNode::Symbol(sid) => {
                let assumptions = arena.symbol_assumptions(*sid);
                if assumptions.query(Props::REAL) == Some(false) {
                    return false;
                }
            }
            _ => {
                // Compound expression: ask the assumption system.  Only a
                // definite "not real" blocks the rewrite.
                let mut cache = crate::base::assumptions::AssumptionCache::new();
                if cache.query(arena, id, Props::REAL) == Some(false) {
                    return false;
                }
            }
        }
    }
    true
}

/// `abs(w) → w` when `w` is a positive numeric literal.
fn rule_abs_positive(arena: &mut Arena) -> Rule {
    let (w_expr, w_id) = arena.wild();
    let abs_w = arena.abs(w_expr);

    let mut wilds = FxHashMap::default();
    wilds.insert(w_expr, w_id);
    let pattern = Pattern { root: abs_w, wilds };

    Rule {
        name: "abs_positive",
        pattern,
        template: w_expr,
        condition: Some(condition_wild_positive),
    }
}

pub(crate) fn basic_rules(arena: &mut Arena) -> Vec<Rule> {
    vec![
        rule_pythagorean(arena),
        unary_compose_rule(arena, "exp_ln", Arena::exp, Arena::ln, identity),
        {
            let mut r = unary_compose_rule(arena, "ln_exp", Arena::ln, Arena::exp, identity);
            r.condition = Some(condition_wild_not_known_nonreal);
            r
        },
        unary_compose_rule(arena, "abs_abs", Arena::abs, Arena::abs, Arena::abs),
        rule_sqrt_sq(arena),
        rule_cosh_sinh_identity(arena),
        rule_pow_pow(arena),
        unary_compose_rule(arena, "asinh_sinh", Arena::asinh, Arena::sinh, identity),
        unary_compose_rule(arena, "acosh_cosh", Arena::acosh, Arena::cosh, Arena::abs),
        unary_compose_rule(arena, "atanh_tanh", Arena::atanh, Arena::tanh, identity),
        rule_sin_div_cos(arena),
        rule_cos_div_sin(arena),
        rule_sinh_div_cosh(arena),
        rule_exp_mul(arena),
        rule_exp_log_denest(arena),
        rule_abs_positive(arena),
        unary_compose_rule(arena, "sin_asin", Arena::sin, Arena::asin, identity),
        unary_compose_rule(arena, "cos_acos", Arena::cos, Arena::acos, identity),
        unary_compose_rule(arena, "tan_atan", Arena::tan, Arena::atan, identity),
        unary_compose_rule(arena, "sinh_asinh", Arena::sinh, Arena::asinh, identity),
        unary_compose_rule(arena, "cosh_acosh", Arena::cosh, Arena::acosh, identity),
        unary_compose_rule(arena, "tanh_atanh", Arena::tanh, Arena::atanh, identity),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }

    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Wild matching ───────────────────────────────────────────────

    #[test]
    fn wild_matches_anything() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        let pattern = Pattern {
            root: w,
            wilds: {
                let mut m = FxHashMap::default();
                m.insert(w, wid);
                m
            },
        };

        let result = match_pattern(&mut a, &pattern, x);
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert_eq!(bindings[&wid], x);
    }

    #[test]
    fn wild_consistency_check() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: w + w (same wild twice)
        let pat_expr = a.add(&[w, w]); // canonicalizes to 2*w
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern {
            root: pat_expr,
            wilds,
        };

        // Expression: x + x = 2*x
        let expr = a.add(&[x, x]); // canonicalizes to 2*x

        let result = match_pattern(&mut a, &pattern, expr);
        assert!(result.is_some(), "2*w should match 2*x");
        assert_eq!(result.unwrap()[&wid], x);
    }

    #[test]
    fn wild_consistency_fails_on_mismatch() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Pattern: w + w = 2*w
        let pat_expr = a.add(&[w, w]);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern {
            root: pat_expr,
            wilds,
        };

        // Expression: x + y (different symbols — 2*w can't match x + y)
        let expr = a.add(&[x, y]);

        let result = match_pattern(&mut a, &pattern, expr);
        assert!(result.is_none(), "2*w should not match x + y");
    }

    // ── Exact matching ──────────────────────────────────────────────

    #[test]
    fn exact_match_symbol() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        let pattern = Pattern {
            root: x,
            wilds: FxHashMap::default(),
        };

        assert!(match_pattern(&mut a, &pattern, x).is_some());
        let y = sym(&mut a, "y");
        assert!(match_pattern(&mut a, &pattern, y).is_none());
    }

    #[test]
    fn exact_match_integer() {
        let mut a = Arena::new();
        let two = a.int(2);
        let three = a.int(3);

        let pattern = Pattern {
            root: two,
            wilds: FxHashMap::default(),
        };

        assert!(match_pattern(&mut a, &pattern, two).is_some());
        assert!(match_pattern(&mut a, &pattern, three).is_none());
    }

    // ── Structural matching ─────────────────────────────────────────

    #[test]
    fn match_sin_of_wild() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: sin(w)
        let pat = a.sin(w);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: sin(x)
        let expr = a.sin(x);
        let result = match_pattern(&mut a, &pattern, expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[&wid], x);
    }

    #[test]
    fn match_pow_of_wild() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Pattern: w^2
        let pat = a.pow(w, two);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: x^2
        let expr = a.pow(x, two);
        let result = match_pattern(&mut a, &pattern, expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[&wid], x);

        // Non-match: x^3
        let three = a.int(3);
        let expr3 = a.pow(x, three);
        assert!(match_pattern(&mut a, &pattern, expr3).is_none());
    }

    #[test]
    fn match_fails_different_structure() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Pattern: sin(w)
        let pat = a.sin(w);
        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);
        let pattern = Pattern { root: pat, wilds };

        // Expression: cos(x) — wrong function
        let expr = a.cos(x);
        assert!(match_pattern(&mut a, &pattern, expr).is_none());
    }

    // ── Instantiation ───────────────────────────────────────────────

    #[test]
    fn instantiate_replaces_wilds() {
        let mut a = Arena::new();
        let (w, wid) = a.wild();
        let x = sym(&mut a, "x");

        // Template: w^2
        let two = a.int(2);
        let template = a.pow(w, two);

        let mut wilds = FxHashMap::default();
        wilds.insert(w, wid);

        let mut bindings = Substitution::default();
        bindings.insert(wid, x);

        let result = instantiate(&mut a, template, &wilds, &bindings);
        assert_eq!(display(&a, result), "x^2");
    }

    // ── Rule application ────────────────────────────────────────────

    #[test]
    fn rule_pythagorean_identity() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Build sin^2(x) + cos^2(x)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        // Apply rules.
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "1",
            "sin²(x) + cos²(x) should simplify to 1"
        );
        assert_eq!(steps.len(), 1, "should have one step");
        assert_eq!(steps[0].rule_name, "pythagorean");
    }

    #[test]
    fn rule_pythagorean_with_y() {
        let mut a = Arena::new();
        let y = sym(&mut a, "y");
        let two = a.int(2);

        // Build sin^2(y) + cos^2(y) — different symbol
        let sin_y = a.sin(y);
        let cos_y = a.cos(y);
        let sin_sq = a.pow(sin_y, two);
        let cos_sq = a.pow(cos_y, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (result, _steps) = apply_rules(&mut a, expr, &rules);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn rule_no_match_returns_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(result, expr, "no rule should fire on sin(x) alone");
        assert!(steps.is_empty());
    }

    #[test]
    fn rule_pythagorean_in_larger_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // Build 3 + sin^2(x) + cos^2(x)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[three, sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        // Sub-expression matching finds the Pythagorean pair inside
        // the 3-term Add and reduces sin²(x) + cos²(x) → 1, then
        // canonicalization combines 3 + 1 → 4.
        assert_eq!(
            display(&a, result),
            "4",
            "sub-expression matching should find sin²+cos² inside larger Add"
        );
        assert_eq!(steps.len(), 1, "one rule should fire (pythagorean)");
        assert_eq!(steps[0].rule_name, "pythagorean");
    }

    // ── apply_rules tracing ─────────────────────────────────────────

    #[test]
    fn trace_records_all_steps() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let sin_sq = a.pow(sin_x, two);
        let cos_sq = a.pow(cos_x, two);
        let expr = a.add(&[sin_sq, cos_sq]);

        let rules = basic_rules(&mut a);
        let (_result, steps) = apply_rules(&mut a, expr, &rules);

        assert!(!steps.is_empty(), "trace should have at least one entry");
        for step in &steps {
            assert!(!step.rule_name.is_empty());
            // before and after should be different
            assert_ne!(step.before, step.after);
        }
    }

    #[test]
    fn mul_sub_match_exp_combine() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");

        // Build exp(x) * exp(y) * z
        let exp_x = a.exp(x);
        let exp_y = a.exp(y);
        let expr = a.mul(&[exp_x, exp_y, z]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        let result_str = display(&a, result);
        // Should contain exp(x + y) and z multiplied together
        assert!(
            result_str.contains("exp("),
            "result should contain exp: got {result_str}"
        );
        assert!(
            result_str.contains("z"),
            "result should contain z: got {result_str}"
        );
        assert!(
            steps.iter().any(|s| s.rule_name == "exp_mul"),
            "exp_mul rule should have fired"
        );
    }

    #[test]
    fn simplify_sin_over_cos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Build sin(x) * cos(x)^(-1)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let neg_one = a.int(-1);
        let cos_x_inv = a.pow(cos_x, neg_one);
        let expr = a.mul(&[sin_x, cos_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "tan(x)",
            "sin(x)/cos(x) should simplify to tan(x)"
        );
        assert!(steps.iter().any(|s| s.rule_name == "sin_div_cos"));
    }

    #[test]
    fn simplify_sinh_over_cosh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");

        // Build sinh(x) * cosh(x)^(-1)
        let sinh_x = a.sinh(x);
        let cosh_x = a.cosh(x);
        let neg_one = a.int(-1);
        let cosh_x_inv = a.pow(cosh_x, neg_one);
        let expr = a.mul(&[sinh_x, cosh_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        assert_eq!(
            display(&a, result),
            "tanh(x)",
            "sinh(x)/cosh(x) should simplify to tanh(x)"
        );
        assert!(steps.iter().any(|s| s.rule_name == "sinh_div_cosh"));
    }

    #[test]
    fn simplify_sin_over_cos_in_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);

        // Build 2 * sin(x) / cos(x) = 2 * sin(x) * cos(x)^(-1)
        let sin_x = a.sin(x);
        let cos_x = a.cos(x);
        let neg_one = a.int(-1);
        let cos_x_inv = a.pow(cos_x, neg_one);
        let expr = a.mul(&[two, sin_x, cos_x_inv]);

        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, expr, &rules);

        let result_str = display(&a, result);
        assert!(
            result_str.contains("tan(x)"),
            "2*sin(x)/cos(x) should simplify to 2*tan(x): got {result_str}"
        );
        assert!(
            result_str.contains("2"),
            "result should still contain factor 2: got {result_str}"
        );
        assert!(
            steps.iter().any(|s| s.rule_name == "sin_div_cos"),
            "sin_div_cos rule should have fired"
        );
    }

    // ── Two-wild pattern ────────────────────────────────────────────

    #[test]
    fn two_wild_pattern() {
        let mut a = Arena::new();
        let (w1, wid1) = a.wild();
        let (w2, wid2) = a.wild();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Pattern: w1 + w2
        let pat = a.add(&[w1, w2]);
        let mut wilds = FxHashMap::default();
        wilds.insert(w1, wid1);
        wilds.insert(w2, wid2);
        let pattern = Pattern { root: pat, wilds };

        // Expression: x + y
        let expr = a.add(&[x, y]);
        let result = match_pattern(&mut a, &pattern, expr);
        assert!(result.is_some());
        let bindings = result.unwrap();
        // The exact binding depends on canonical order — just check both
        // wilds are bound.
        assert!(bindings.contains_key(&wid1));
        assert!(bindings.contains_key(&wid2));
    }

    // ── Conditional rules ───────────────────────────────────────────

    #[test]
    fn conditional_abs_positive() {
        let mut a = Arena::new();
        let five = a.int(5);
        let abs_five = a.abs(five);
        // abs(5) is now canonicalized eagerly to 5 at arena level,
        // so the pattern rule doesn't need to fire.
        assert_eq!(display(&a, abs_five), "5");
        // Applying rules to the already-simplified value is a no-op.
        let rules = basic_rules(&mut a);
        let (result, _steps) = apply_rules(&mut a, abs_five, &rules);
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn conditional_abs_negative_unchanged() {
        let mut a = Arena::new();
        let neg_five = a.int(-5);
        let abs_neg = a.abs(neg_five);
        // abs(-5) is now canonicalized eagerly to 5 at arena level.
        assert_eq!(display(&a, abs_neg), "5");
        let rules = basic_rules(&mut a);
        let (result, _) = apply_rules(&mut a, abs_neg, &rules);
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn conditional_abs_symbol_unchanged() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let abs_x = a.abs(x);
        let rules = basic_rules(&mut a);
        let (result, _) = apply_rules(&mut a, abs_x, &rules);
        // Symbol has unknown sign — should not fire
        assert_eq!(display(&a, result), "abs(x)");
    }

    #[test]
    fn simplify_sin_asin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asin_x = a.asin(x);
        let sin_asin_x = a.sin(asin_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, sin_asin_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "sin_asin"));
    }

    #[test]
    fn simplify_cos_acos() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let acos_x = a.acos(x);
        let cos_acos_x = a.cos(acos_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, cos_acos_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "cos_acos"));
    }

    #[test]
    fn simplify_tan_atan() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let atan_x = a.atan(x);
        let tan_atan_x = a.tan(atan_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, tan_atan_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "tan_atan"));
    }

    #[test]
    fn simplify_sinh_asinh() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let asinh_x = a.asinh(x);
        let sinh_asinh_x = a.sinh(asinh_x);
        let rules = basic_rules(&mut a);
        let (result, steps) = apply_rules(&mut a, sinh_asinh_x, &rules);
        assert_eq!(display(&a, result), "x");
        assert!(steps.iter().any(|s| s.rule_name == "sinh_asinh"));
    }

    #[test]
    fn pow_pow_blocked_for_fractional_exponents() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let half = arena.rational(1, 2);
        let third = arena.rational(1, 3);
        let inner = arena.pow(x, half);
        let expr = arena.pow(inner, third); // (x^(1/2))^(1/3)
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        // Should NOT simplify to x^(1/6) because no exponent is integer
        assert_eq!(
            result, expr,
            "pow_pow should not fire for fractional exponents"
        );
    }

    #[test]
    fn pow_pow_fires_for_integer_exponent() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let three = arena.int(3);
        let inner = arena.pow(x, two);
        let expr = arena.pow(inner, three); // (x^2)^3
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let six = arena.int(6);
        let expected = arena.pow(x, six); // x^6
        assert_eq!(
            result, expected,
            "pow_pow should fire for integer exponents"
        );
    }

    #[test]
    fn acosh_cosh_gives_abs() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let cosh_x = arena.cosh(x);
        let expr = arena.acosh(cosh_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let expected = arena.abs(x);
        assert_eq!(result, expected, "acosh(cosh(x)) should give |x|");
    }

    #[test]
    fn asin_sin_no_longer_simplifies() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let sin_x = arena.sin(x);
        let expr = arena.asin(sin_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        assert_eq!(result, expr, "asin(sin(x)) should stay (rule removed)");
    }

    #[test]
    fn exp_log_denest_simplifies() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let three = arena.int(3);
        let ln_x = arena.ln(x);
        let product = arena.mul(&[three, ln_x]);
        let expr = arena.exp(product); // exp(3*ln(x))
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        let expected = arena.pow(x, three); // x^3
        assert_eq!(result, expected, "exp(3*ln(x)) should simplify to x^3");
    }

    #[test]
    fn ln_exp_fires_without_assumption() {
        let mut arena = Arena::new();
        let x = arena.symbol("x"); // no assumptions at all
        let exp_x = arena.exp(x);
        let expr = arena.ln(exp_x); // ln(exp(x))
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        assert_eq!(
            result, x,
            "ln(exp(x)) should simplify to x without assumptions"
        );
    }

    #[test]
    fn ln_exp_fires_with_real_assumption() {
        use crate::base::assumptions::{Assumptions, Props};
        use crate::base::node::ExprNode;
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        if let ExprNode::Symbol(sid) = *arena.node(x) {
            let mut a = Assumptions::default();
            a.assert_true(Props::REAL);
            arena.set_symbol_assumptions(sid, a);
        }
        let exp_x = arena.exp(x);
        let expr = arena.ln(exp_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        assert_eq!(
            result, x,
            "ln(exp(x)) should simplify to x with Real assumption"
        );
    }

    // ── AC matching (0.2 engine) ──────────────────────────────────────────

    /// Build a pattern from an expression whose `_`-suffixed symbols are wilds.
    fn pat(a: &Arena, root: ExprId) -> Pattern {
        pattern_from_expr(a, root).0
    }

    fn binding(a: &Arena, p: &Pattern, b: &Substitution, name: &str) -> Option<ExprId> {
        let wild_expr = p.wilds.keys().copied().find(|&id| match a.node(id) {
            ExprNode::Symbol(sid) => a.symbol_name(*sid) == name,
            _ => false,
        })?;
        b.get(&p.wilds[&wild_expr]).copied()
    }

    #[test]
    fn ac_two_wild_structured_terms_any_order() {
        let mut a = Arena::new();
        let (x, y) = (a.symbol("x"), a.symbol("y"));
        let (aw, bw) = (a.symbol("a_"), a.symbol("b_"));
        // sin(a_)cos(b_) + cos(a_)sin(b_)
        let (sa, cb, ca, sb) = (a.sin(aw), a.cos(bw), a.cos(aw), a.sin(bw));
        let t1 = a.mul(&[sa, cb]);
        let t2 = a.mul(&[ca, sb]);
        let root = a.add(&[t1, t2]);
        let p = pat(&a, root);
        // subject: cos(y)sin(x) + sin(y)cos(x)
        let (sx, cy, cx, sy) = (a.sin(x), a.cos(y), a.cos(x), a.sin(y));
        let u1 = a.mul(&[cy, sx]);
        let u2 = a.mul(&[cx, sy]);
        let subj = a.add(&[u1, u2]);
        let m = match_pattern(&mut a, &p, subj).expect("should match");
        assert_eq!(binding(&a, &p, &m, "a_"), Some(x));
        assert_eq!(binding(&a, &p, &m, "b_"), Some(y));
    }

    #[test]
    fn ac_last_plain_wild_absorbs_rest() {
        let mut a = Arena::new();
        let (x, y, z) = (a.symbol("x"), a.symbol("y"), a.symbol("z"));
        let aw = a.symbol("a_");
        let sx = a.sin(x);
        let root = a.add(&[sx, aw]); // sin(x) + a_
        let p = pat(&a, root);
        let subj = a.add(&[sx, y, z]);
        let m = match_pattern(&mut a, &p, subj).expect("should match");
        let yz = a.add(&[y, z]);
        assert_eq!(binding(&a, &p, &m, "a_"), Some(yz));
    }

    #[test]
    fn ac_sequence_wild_may_be_empty() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let rest = a.symbol("rest__");
        let sx = a.sin(x);
        let root = a.add(&[sx, rest]);
        let p = pat(&a, root);
        // Single-term subject: rest__ → 0
        let m = match_pattern(&mut a, &p, sx).expect("should match");
        assert_eq!(binding(&a, &p, &m, "rest__"), Some(a.zero));
        // Multi-term subject: rest__ → the leftovers
        let (y, z) = (a.symbol("y"), a.symbol("z"));
        let subj = a.add(&[sx, y, z]);
        let m = match_pattern(&mut a, &p, subj).expect("should match");
        let yz = a.add(&[y, z]);
        assert_eq!(binding(&a, &p, &m, "rest__"), Some(yz));
    }

    #[test]
    fn ac_partial_match_reports_leftover_only_at_root() {
        let mut a = Arena::new();
        let (x, y) = (a.symbol("x"), a.symbol("y"));
        let aw = a.symbol("a_");
        let two = a.int(2);
        let (s, c) = (a.sin(aw), a.cos(aw));
        let s2 = a.pow(s, two);
        let c2 = a.pow(c, two);
        let root = a.add(&[s2, c2]);
        let p = pat(&a, root);
        let (sx, cx) = (a.sin(x), a.cos(x));
        let sx2 = a.pow(sx, two);
        let cx2 = a.pow(cx, two);
        let subj = a.add(&[sx2, cx2, y]);
        assert!(
            match_pattern(&mut a, &p, subj).is_none(),
            "whole-node match must fail"
        );
        let partial = match_all(&mut a, &p, subj, true, 1, MATCH_BUDGET);
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0].leftover.as_slice(), &[y]);
    }

    #[test]
    fn ac_mul_coefficient_splitting() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let aw = a.symbol("a_");
        let two = a.int(2);
        let (s, c) = (a.sin(aw), a.cos(aw));
        let root = a.mul(&[two, s, c]);
        let p = pat(&a, root);
        let six = a.int(6);
        let (sx, cx) = (a.sin(x), a.cos(x));
        let subj = a.mul(&[six, sx, cx]);
        let m = match_all(&mut a, &p, subj, true, 1, MATCH_BUDGET);
        assert_eq!(m.len(), 1);
        let three = a.int(3);
        assert_eq!(m[0].leftover.as_slice(), &[three]);
        // Coefficient 1 in the subject is not split.
        let plain = a.mul(&[sx, cx]);
        assert!(match_all(&mut a, &p, plain, true, 1, MATCH_BUDGET).is_empty());
    }

    #[test]
    fn ac_nested_add_without_absorber_must_consume_all() {
        let mut a = Arena::new();
        let (x, y) = (a.symbol("x"), a.symbol("y"));
        let aw = a.symbol("a_");
        let one = a.one;
        let inner = a.add(&[aw, one]);
        let root = a.sin(inner); // sin(a_ + 1)
        let p = pat(&a, root);
        let two = a.int(2);
        let bad = a.add(&[x, y, two]);
        let bad_s = a.sin(bad);
        assert!(match_pattern(&mut a, &p, bad_s).is_none());
        let good = a.add(&[x, y, one]);
        let good_s = a.sin(good);
        let m = match_pattern(&mut a, &p, good_s).expect("should match");
        let xy = a.add(&[x, y]);
        assert_eq!(binding(&a, &p, &m, "a_"), Some(xy));
    }

    #[test]
    fn ac_budget_exhaustion_terminates() {
        let mut a = Arena::new();
        // Pattern a_ + b_ + c_ + d_ vs a 40-term sum, asking for many results
        // so the search would enumerate ~40·39·38 assignments without a cap.
        let wilds: Vec<ExprId> = ["a_", "b_", "c_", "d_"]
            .iter()
            .map(|n| a.symbol(n))
            .collect();
        let root = a.add(&wilds);
        let p = pat(&a, root);
        let terms: Vec<ExprId> = (0..40).map(|i| a.symbol(&format!("t{i}"))).collect();
        let subj = a.add(&terms);
        let start = std::time::Instant::now();
        let results = match_all(&mut a, &p, subj, false, usize::MAX, 500);
        assert!(start.elapsed().as_millis() < 500);
        assert!(
            results.len() < 500,
            "budget must bound the enumeration: {}",
            results.len()
        );
    }

    #[test]
    fn pattern_from_expr_detects_wild_naming() {
        let mut a = Arena::new();
        let (x, aw, rest) = (a.symbol("x"), a.symbol("a_"), a.symbol("rest__"));
        let under = a.symbol("_"); // a lone underscore is not a wild
        let root = a.add(&[x, aw, rest, under]);
        let (p, names) = pattern_from_expr(&a, root);
        assert_eq!(p.wilds.len(), 2);
        let mut ns: Vec<&String> = names.values().collect();
        ns.sort();
        assert_eq!(ns, ["a_", "rest__"]);
        assert!(is_sequence_wild_symbol(&a, rest));
        assert!(!is_sequence_wild_symbol(&a, aw));
    }

    #[test]
    fn tree_size_counts_with_multiplicity() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let sx = a.sin(x);
        let two = a.int(2);
        // sin(x)^2 * sin(x): DAG has sin(x) once, tree has it twice.
        let sq = a.pow(sx, two);
        let prod = a.mul(&[sq, sx]);
        let dag = crate::simplify::simplify_engine::count_ops(&a, prod);
        let tree = tree_size_capped(&a, prod, 1000);
        assert!(tree > dag);
        assert_eq!(tree_size_capped(&a, prod, 3), 3, "saturates at the cap");
    }

    // ── subs_algebraic ─────────────────────────────────────────────────────────

    #[test]
    fn subs_algebraic_power_cases() {
        let mut a = Arena::new();
        let (x, y) = (a.symbol("x"), a.symbol("y"));
        let two = a.int(2);
        let old = a.pow(x, two);
        let four = a.int(4);
        let x4 = a.pow(x, four);
        let r = subs_algebraic(&mut a, x4, old, y);
        assert_eq!(display(&a, r), "y^2");
        let three = a.int(3);
        let x3 = a.pow(x, three);
        let r = subs_algebraic(&mut a, x3, old, y);
        assert_eq!(display(&a, r), "x*y");
        let m2 = a.int(-2);
        let xm2 = a.pow(x, m2);
        let r = subs_algebraic(&mut a, xm2, old, y);
        assert_eq!(display(&a, r), "1/y");
        assert_eq!(
            subs_algebraic(&mut a, x, old, y),
            x,
            "x is not a multiple of x^2"
        );
    }

    #[test]
    fn subs_algebraic_product_and_sum_cases() {
        let mut a = Arena::new();
        let (x, y, z, w) = (a.symbol("x"), a.symbol("y"), a.symbol("z"), a.symbol("w"));
        let two = a.int(2);
        let xy = a.mul(&[x, y]);
        let e = a.mul(&[two, x, y, z]);
        let r = subs_algebraic(&mut a, e, xy, w);
        assert_eq!(display(&a, r), "2*w*z");
        let (p, q, r, d) = (a.symbol("p"), a.symbol("q"), a.symbol("r"), a.symbol("d"));
        let pq = a.add(&[p, q]);
        let pqr = a.add(&[p, q, r]);
        let r = subs_algebraic(&mut a, pqr, pq, d);
        assert_eq!(display(&a, r), "d + r");
    }

    #[test]
    fn subs_algebraic_exp_as_power() {
        let mut a = Arena::new();
        let (x, t) = (a.symbol("x"), a.symbol("t"));
        let ex = a.exp(x);
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let e2x = a.exp(two_x);
        let r = subs_algebraic(&mut a, e2x, ex, t);
        assert_eq!(display(&a, r), "t^2");
    }

    #[test]
    fn ln_exp_blocked_for_imaginary() {
        use crate::base::assumptions::{Assumptions, Props};
        use crate::base::node::ExprNode;
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        if let ExprNode::Symbol(sid) = *arena.node(x) {
            let mut a = Assumptions::default();
            a.assert_true(Props::IMAGINARY);
            arena.set_symbol_assumptions(sid, a);
        }
        let exp_x = arena.exp(x);
        let expr = arena.ln(exp_x);
        let rules = basic_rules(&mut arena);
        let (result, _) = apply_rules(&mut arena, expr, &rules);
        assert_eq!(
            result, expr,
            "ln(exp(x)) should NOT simplify for Imaginary x"
        );
    }
}
