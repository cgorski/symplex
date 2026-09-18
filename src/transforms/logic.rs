//! Boolean-logic simplification: flatten/absorb, CNF/DNF, satisfiability.
//!
//! Boolean expressions are built from `BoolTrue`/`BoolFalse`, the
//! relational nodes `Gt`/`Ge`/`Eq_`/`Ne` (there are no `Lt`/`Le` nodes —
//! `a < b` is stored as `Gt(b, a)`), the connectives `And`/`Or`/`Not`,
//! and opaque atoms (anything else, e.g. a symbol used as a proposition).
//!
//! # Relational atoms
//!
//! Every relational atom is normalised to a *pair* `(a, b)` (numeric
//! literals on the right, otherwise sort-key order) plus a 3-bit *mask*
//! over the trichotomy
//! `{a < b, a = b, a > b}`.  Negation is mask complement, conjunction of
//! two atoms on the same pair is mask intersection, disjunction is mask
//! union.  This gives `¬(a < b) = a ≥ b`, `(x > 0) ∧ (x ≥ 0) = x > 0`,
//! `(x > 0) ∨ (x = 0) = x ≥ 0`, `(x > 0) ∧ (x < 0) = false` and
//! `(x > 0) ∨ (x ≤ 0) = true` for free.  The trichotomy assumes the
//! operands are real-valued, which is the domain of symplex's relational
//! nodes and inequality solver.
//!
//! Atoms whose two sides can be ordered (both numeric, exact difference,
//! or well-separated constants) are folded to `true`/`false`.
//!
//! # Satisfiability
//!
//! [`BoolEx::is_tautology`](crate::expr::BoolEx::is_tautology),
//! [`BoolEx::is_contradiction`](crate::expr::BoolEx::is_contradiction) and
//! [`BoolEx::satisfiable`](crate::expr::BoolEx::satisfiable) first try a
//! propositional search (each relational pair is a 3-valued variable, each
//! opaque atom a 2-valued one; ≤ 24 variables, bounded budget).  A
//! propositional proof is always sound.  When the propositional answer is
//! inconclusive and every atom is a relational in one and the same free
//! symbol, the question is decided *exactly* through the inequality solver
//! and set algebra (`x > 1 → x > 0` is recognised as a tautology).
//! Otherwise the answer is `None` — never a guess.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::assumptions::{AssumptionCache, Props};
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};

/// Trichotomy bit: `a < b`.
const LT: u8 = 0b001;
/// Trichotomy bit: `a = b`.
const EQ: u8 = 0b010;
/// Trichotomy bit: `a > b`.
const GT: u8 = 0b100;
/// All three outcomes.
const ALL: u8 = 0b111;

/// Maximum number of propositional variables for the SAT search.
const MAX_SAT_VARS: usize = 24;
/// Maximum number of partial evaluations in one SAT search.
const SAT_BUDGET: usize = 200_000;
/// Maximum number of clauses produced by CNF/DNF distribution.
const MAX_CLAUSES: usize = 4096;
/// Maximum total number of literals produced by CNF/DNF distribution.
const MAX_LITERALS: usize = 50_000;
/// Maximum number of variables for a truth table.
const MAX_TRUTH_TABLE_VARS: usize = 8;
/// Fixpoint iterations for `simplify_bool`.
const MAX_SIMPLIFY_PASSES: usize = 6;

// ═══════════════════════════════════════════════════════════════════════════
// Relational atoms
// ═══════════════════════════════════════════════════════════════════════════

/// A relational atom in canonical orientation: a numeric literal goes on
/// the right; otherwise operands are ordered by sort key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rel {
    a: ExprId,
    b: ExprId,
    mask: u8,
}

fn mirror(mask: u8) -> u8 {
    let mut m = mask & EQ;
    if mask & LT != 0 {
        m |= GT;
    }
    if mask & GT != 0 {
        m |= LT;
    }
    m
}

/// Should the operands `(p, q)` be swapped into canonical orientation?
fn should_swap(arena: &Arena, p: ExprId, q: ExprId) -> bool {
    if p == q {
        return false;
    }
    let pn = matches!(arena.node(p), ExprNode::Num(_));
    let qn = matches!(arena.node(q), ExprNode::Num(_));
    match (pn, qn) {
        (true, false) => true,
        (false, true) => false,
        _ => arena.sort_key(p) > arena.sort_key(q),
    }
}

/// Decode a relational node into canonical pair + mask.
fn rel_of(arena: &Arena, id: ExprId) -> Option<Rel> {
    let (p, q, mask) = match arena.node(id) {
        ExprNode::Gt(p, q) => (*p, *q, GT),
        ExprNode::Ge(p, q) => (*p, *q, GT | EQ),
        ExprNode::Eq_(p, q) => (*p, *q, EQ),
        ExprNode::Ne(p, q) => (*p, *q, LT | GT),
        _ => return None,
    };
    Some(if should_swap(arena, p, q) {
        Rel {
            a: q,
            b: p,
            mask: mirror(mask),
        }
    } else {
        Rel { a: p, b: q, mask }
    })
}

/// Emit a relational atom, folding it to a constant when the order of
/// the operands is known.
fn fold_rel(arena: &mut Arena, rel: Rel) -> ExprId {
    let Rel { a, b, mask } = rel;
    if mask == 0 {
        return arena.bool_false;
    }
    if mask == ALL {
        return arena.bool_true;
    }
    if let Some(o) = crate::transforms::sets::cmp_exprs(arena, a, b) {
        let bit = match o {
            std::cmp::Ordering::Less => LT,
            std::cmp::Ordering::Equal => EQ,
            std::cmp::Ordering::Greater => GT,
        };
        return if mask & bit != 0 {
            arena.bool_true
        } else {
            arena.bool_false
        };
    }
    match mask {
        GT => arena.gt(a, b),
        LT => arena.gt(b, a),
        EQ => arena.eq_(a, b),
        m if m == GT | EQ => arena.ge(a, b),
        m if m == LT | EQ => arena.ge(b, a),
        _ => arena.ne_(a, b),
    }
}

/// Negate a literal (an atom or the negation of an atom).
fn neg_lit(arena: &mut Arena, lit: ExprId) -> ExprId {
    if lit == arena.bool_true {
        return arena.bool_false;
    }
    if lit == arena.bool_false {
        return arena.bool_true;
    }
    if let Some(r) = rel_of(arena, lit) {
        return fold_rel(
            arena,
            Rel {
                mask: ALL & !r.mask,
                ..r
            },
        );
    }
    arena.not(lit)
}

fn is_connective(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::And(_) | ExprNode::Or(_) | ExprNode::Not(_)
    )
}

/// Post-order over the boolean structure only (`And`/`Or`/`Not`).
fn bool_post_order(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut order = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
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
                    if !visited.contains(&c) {
                        stack.push((c, false));
                    }
                }
            }
            ExprNode::Not(x) if !visited.contains(x) => stack.push((*x, false)),
            _ => {}
        }
    }
    order
}

// ═══════════════════════════════════════════════════════════════════════════
// Negation normal form
// ═══════════════════════════════════════════════════════════════════════════

/// Push negations inward (De Morgan), negate relationals by mask
/// complement, and fold decidable relationals.
pub(crate) fn to_nnf(arena: &mut Arena, root: ExprId) -> ExprId {
    let mut memo: FxHashMap<(ExprId, bool), ExprId> = FxHashMap::default();
    let mut stack: Vec<(ExprId, bool, bool)> = vec![(root, true, false)];
    while let Some((id, pol, expanded)) = stack.pop() {
        if memo.contains_key(&(id, pol)) {
            continue;
        }
        if !expanded {
            stack.push((id, pol, true));
            match arena.node(id) {
                ExprNode::And(ch) | ExprNode::Or(ch) => {
                    for &c in ch.iter().rev() {
                        stack.push((c, pol, false));
                    }
                }
                ExprNode::Not(x) => stack.push((*x, !pol, false)),
                _ => {}
            }
            continue;
        }
        let result = match arena.node(id).clone() {
            ExprNode::BoolTrue => {
                if pol {
                    arena.bool_true
                } else {
                    arena.bool_false
                }
            }
            ExprNode::BoolFalse => {
                if pol {
                    arena.bool_false
                } else {
                    arena.bool_true
                }
            }
            ExprNode::Gt(..) | ExprNode::Ge(..) | ExprNode::Eq_(..) | ExprNode::Ne(..) => {
                let r = rel_of(arena, id).unwrap_or(Rel {
                    a: id,
                    b: id,
                    mask: ALL,
                });
                let mask = if pol { r.mask } else { ALL & !r.mask };
                fold_rel(arena, Rel { mask, ..r })
            }
            ExprNode::And(ch) => {
                let kids: SmallVec<[ExprId; 6]> = ch
                    .iter()
                    .map(|c| memo.get(&(*c, pol)).copied().unwrap_or(*c))
                    .collect();
                if pol {
                    arena.and(&kids)
                } else {
                    arena.or(&kids)
                }
            }
            ExprNode::Or(ch) => {
                let kids: SmallVec<[ExprId; 6]> = ch
                    .iter()
                    .map(|c| memo.get(&(*c, pol)).copied().unwrap_or(*c))
                    .collect();
                if pol {
                    arena.or(&kids)
                } else {
                    arena.and(&kids)
                }
            }
            ExprNode::Not(x) => memo.get(&(x, !pol)).copied().unwrap_or(id),
            _ => {
                if pol {
                    id
                } else {
                    arena.not(id)
                }
            }
        };
        memo.insert((id, pol), result);
    }
    memo.get(&(root, true)).copied().unwrap_or(root)
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten / fold / absorb
// ═══════════════════════════════════════════════════════════════════════════

/// What a connective "knows" about its literals, for reducing dual
/// children in context.
#[derive(Default)]
struct Context {
    /// pair → mask of states known possible in this context
    pairs: FxHashMap<(ExprId, ExprId), u8>,
    /// opaque atom → known truth value
    atoms: FxHashMap<ExprId, bool>,
}

/// Simplify one flattened connective whose children are already
/// simplified.  `is_and` selects `And` (else `Or`).
fn simplify_connective(arena: &mut Arena, is_and: bool, children: &[ExprId]) -> ExprId {
    let identity = if is_and {
        arena.bool_true
    } else {
        arena.bool_false
    };
    let absorbing = if is_and {
        arena.bool_false
    } else {
        arena.bool_true
    };

    // 1. Flatten + constants + dedupe.
    let mut lits: Vec<ExprId> = Vec::new();
    let mut seen: FxHashSet<ExprId> = FxHashSet::default();
    let mut work: Vec<ExprId> = children.iter().rev().copied().collect();
    while let Some(c) = work.pop() {
        let same = match arena.node(c) {
            ExprNode::And(ch) if is_and => Some(ch.clone()),
            ExprNode::Or(ch) if !is_and => Some(ch.clone()),
            _ => None,
        };
        if let Some(ch) = same {
            for &k in ch.iter().rev() {
                work.push(k);
            }
            continue;
        }
        if c == identity {
            continue;
        }
        if c == absorbing {
            return absorbing;
        }
        if seen.insert(c) {
            lits.push(c);
        }
    }

    // 2. Merge relationals on the same pair; build the context.
    let mut ctx = Context::default();
    let mut pair_order: Vec<(ExprId, ExprId)> = Vec::new();
    let mut others: Vec<ExprId> = Vec::new();
    for &l in &lits {
        if let Some(r) = rel_of(arena, l) {
            let key = (r.a, r.b);
            let entry = ctx.pairs.entry(key).or_insert_with(|| {
                pair_order.push(key);
                if is_and { ALL } else { 0 }
            });
            if is_and {
                *entry &= r.mask;
            } else {
                *entry |= r.mask;
            }
        } else {
            others.push(l);
        }
    }
    let mut merged: Vec<ExprId> = Vec::new();
    for key in &pair_order {
        let mask = ctx.pairs[key];
        let e = fold_rel(
            arena,
            Rel {
                a: key.0,
                b: key.1,
                mask,
            },
        );
        if e == absorbing {
            return absorbing;
        }
        if e != identity {
            merged.push(e);
        }
    }
    // In an `Or`, a pair with mask m means "the state is NOT in ¬m" for
    // the purpose of reducing sibling `And`s; store the complement.
    if !is_and {
        for m in ctx.pairs.values_mut() {
            *m = ALL & !*m;
        }
    }

    // 3. Opaque literals: complement detection + context.
    let mut opaque: Vec<ExprId> = Vec::new();
    let mut duals: Vec<ExprId> = Vec::new();
    for &l in &others {
        let is_dual = match arena.node(l) {
            ExprNode::Or(_) => is_and,
            ExprNode::And(_) => !is_and,
            _ => false,
        };
        if is_dual {
            duals.push(l);
            continue;
        }
        let (atom, val) = match arena.node(l) {
            ExprNode::Not(x) => (*x, false),
            _ => (l, true),
        };
        // In an `And` the literal is true; in an `Or` it is false.
        let known = if is_and { val } else { !val };
        if let Some(&prev) = ctx.atoms.get(&atom)
            && prev != known
        {
            return absorbing;
        }
        ctx.atoms.insert(atom, known);
        opaque.push(l);
    }

    // 4. Reduce dual children in context (absorption and its relatives).
    let mut reduced_duals: Vec<ExprId> = Vec::new();
    for &d in &duals {
        let kids: SmallVec<[ExprId; 6]> = match arena.node(d) {
            ExprNode::Or(ch) | ExprNode::And(ch) => ch.clone(),
            _ => SmallVec::new(),
        };
        // Inside the dual, a literal known TRUE (from context) makes the
        // whole dual equal to the outer identity (drop it); a literal
        // known FALSE is removed from the dual.
        let mut kept: SmallVec<[ExprId; 6]> = SmallVec::new();
        let mut drop_dual = false;
        for &k in &kids {
            let status: Option<bool> = if let Some(r) = rel_of(arena, k) {
                match ctx.pairs.get(&(r.a, r.b)) {
                    Some(&known) => {
                        if known & !r.mask == 0 {
                            Some(true) // known ⊆ mask
                        } else if known & r.mask == 0 {
                            Some(false)
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            } else {
                let (atom, val) = match arena.node(k) {
                    ExprNode::Not(x) => (*x, false),
                    _ => (k, true),
                };
                ctx.atoms.get(&atom).map(|&kn| kn == val)
            };
            // A literal inside the dual is "true for the dual" when the
            // dual is an Or and the literal holds, or when the dual is an
            // And and the literal fails (then the And collapses to false,
            // i.e. the outer Or's identity).
            match status {
                Some(true) => {
                    if is_and {
                        drop_dual = true; // Or child is true → identity for And
                        break;
                    } else {
                        // And child: literal true → remove from And
                        continue;
                    }
                }
                Some(false) => {
                    if is_and {
                        continue; // Or child: literal false → remove from Or
                    } else {
                        drop_dual = true; // And child false → identity for Or
                        break;
                    }
                }
                None => kept.push(k),
            }
        }
        if drop_dual {
            continue;
        }
        let rebuilt = if kept.len() == kids.len() {
            d
        } else if is_and {
            arena.or(&kept) // empty → false → absorbing for And
        } else {
            arena.and(&kept) // empty → true → absorbing for Or
        };
        if rebuilt == absorbing {
            return absorbing;
        }
        if rebuilt != identity {
            reduced_duals.push(rebuilt);
        }
    }

    // 5. Assemble deterministically.
    let mut out: Vec<ExprId> =
        Vec::with_capacity(merged.len() + opaque.len() + reduced_duals.len());
    out.extend(merged);
    out.extend(opaque);
    out.extend(reduced_duals);
    let mut seen2: FxHashSet<ExprId> = FxHashSet::default();
    out.retain(|c| seen2.insert(*c));
    out.sort_by(|&x, &y| arena.sort_key(x).cmp(arena.sort_key(y)));
    if out.is_empty() {
        identity
    } else if out.len() == 1 {
        out[0]
    } else if is_and {
        arena.and(&out)
    } else {
        arena.or(&out)
    }
}

/// One bottom-up pass over an NNF expression.
fn simplify_pass(arena: &mut Arena, root: ExprId) -> ExprId {
    let order = bool_post_order(arena, root);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &order {
        let new = match arena.node(id).clone() {
            ExprNode::And(ch) => {
                let kids: SmallVec<[ExprId; 6]> = ch
                    .iter()
                    .map(|c| cache.get(c).copied().unwrap_or(*c))
                    .collect();
                simplify_connective(arena, true, &kids)
            }
            ExprNode::Or(ch) => {
                let kids: SmallVec<[ExprId; 6]> = ch
                    .iter()
                    .map(|c| cache.get(c).copied().unwrap_or(*c))
                    .collect();
                simplify_connective(arena, false, &kids)
            }
            ExprNode::Not(x) => {
                let nx = cache.get(&x).copied().unwrap_or(x);
                neg_lit(arena, nx)
            }
            ExprNode::Gt(..) | ExprNode::Ge(..) | ExprNode::Eq_(..) | ExprNode::Ne(..) => {
                match rel_of(arena, id) {
                    Some(r) => fold_rel(arena, r),
                    None => id,
                }
            }
            _ => id,
        };
        cache.insert(id, new);
    }
    cache.get(&root).copied().unwrap_or(root)
}

/// Simplify a boolean expression (see module docs).
pub(crate) fn simplify_bool(arena: &mut Arena, root: ExprId) -> ExprId {
    let mut cur = root;
    for _ in 0..MAX_SIMPLIFY_PASSES {
        let n = to_nnf(arena, cur);
        let s = simplify_pass(arena, n);
        if s == cur {
            break;
        }
        cur = s;
    }
    cur
}

/// Simplify the numeric operands of every relational atom with the
/// numeric simplification engine, then apply [`simplify_bool`].
///
/// Operands are simplified individually (memoised), which keeps the cost
/// linear in the number of atoms instead of running the whole engine over
/// a large boolean tree.
pub(crate) fn simplify_bool_full(arena: &mut Arena, root: ExprId) -> ExprId {
    use crate::simplify::simplify_engine::{SimplifyOpts, unified_simplify};

    let atoms_list = atoms(arena, root);
    let opts = SimplifyOpts::default();
    let mut operand_memo: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let mut pairs: Vec<(ExprId, ExprId)> = Vec::new();
    for atom in atoms_list {
        let (a, b) = match arena.node(atom) {
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                (*a, *b)
            }
            _ => continue,
        };
        let mut simp = |arena: &mut Arena, e: ExprId| -> ExprId {
            if let Some(&s) = operand_memo.get(&e) {
                return s;
            }
            let s = unified_simplify(arena, e, &opts).expr;
            operand_memo.insert(e, s);
            s
        };
        let na = simp(arena, a);
        let nb = simp(arena, b);
        if na == a && nb == b {
            continue;
        }
        let new_atom = match arena.node(atom).clone() {
            ExprNode::Gt(..) => arena.gt(na, nb),
            ExprNode::Ge(..) => arena.ge(na, nb),
            ExprNode::Eq_(..) => arena.eq_(na, nb),
            _ => arena.ne_(na, nb),
        };
        pairs.push((atom, new_atom));
    }
    let rebuilt = if pairs.is_empty() {
        root
    } else {
        arena.subs_map_structural(root, &pairs)
    };
    simplify_bool(arena, rebuilt)
}

// ═══════════════════════════════════════════════════════════════════════════
// CNF / DNF
// ═══════════════════════════════════════════════════════════════════════════

/// A clause is a list of literals.  In CNF mode a clause is a disjunction
/// (and the list of clauses a conjunction); in DNF mode the roles swap.
type Clauses = Vec<Vec<ExprId>>;

/// Combine two clause lists by cross product (the "distributing" side).
fn cross(a: &Clauses, b: &Clauses) -> Option<Clauses> {
    if a.len().saturating_mul(b.len()) > MAX_CLAUSES {
        return None;
    }
    let la: usize = a.iter().map(Vec::len).sum();
    let lb: usize = b.iter().map(Vec::len).sum();
    if la
        .saturating_mul(b.len())
        .saturating_add(lb.saturating_mul(a.len()))
        > MAX_LITERALS
    {
        return None;
    }
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            let mut c = x.clone();
            for &l in y {
                if !c.contains(&l) {
                    c.push(l);
                }
            }
            out.push(c);
        }
    }
    Some(out)
}

/// Simplify a clause: merge relationals on the same pair, detect
/// complementary literals.  Returns `None` if the clause is trivially
/// the absorbing element (a tautological disjunction / contradictory
/// conjunction).
fn clean_clause(arena: &mut Arena, clause: &[ExprId], cnf: bool) -> Option<Vec<ExprId>> {
    // In CNF a clause is an Or (merge masks by union); in DNF an And.
    let e = simplify_connective(arena, !cnf, clause);
    let absorbing = if cnf {
        arena.bool_true
    } else {
        arena.bool_false
    };
    let identity = if cnf {
        arena.bool_false
    } else {
        arena.bool_true
    };
    if e == absorbing {
        return None;
    }
    if e == identity {
        return Some(Vec::new());
    }
    match arena.node(e) {
        ExprNode::Or(ch) if cnf => Some(ch.iter().copied().collect()),
        ExprNode::And(ch) if !cnf => Some(ch.iter().copied().collect()),
        _ => Some(vec![e]),
    }
}

fn distribute(arena: &mut Arena, root: ExprId, cnf: bool) -> ExprId {
    let nnf = simplify_bool(arena, root);
    if nnf == arena.bool_true || nnf == arena.bool_false {
        return nnf;
    }
    let order = bool_post_order(arena, nnf);
    let mut memo: FxHashMap<ExprId, Clauses> = FxHashMap::default();
    for &id in &order {
        let clauses: Clauses = match arena.node(id).clone() {
            ExprNode::And(ch) | ExprNode::Or(ch) => {
                let is_and = matches!(arena.node(id), ExprNode::And(_));
                // "concat" side: And in CNF, Or in DNF.
                let concat = is_and == cnf;
                let mut acc: Clauses = if concat { Vec::new() } else { vec![Vec::new()] };
                for c in &ch {
                    let sub = memo.get(c).cloned().unwrap_or_else(|| vec![vec![*c]]);
                    if concat {
                        acc.extend(sub);
                    } else {
                        match cross(&acc, &sub) {
                            Some(x) => acc = x,
                            None => return nnf, // blow-up guard
                        }
                    }
                    if acc.len() > MAX_CLAUSES
                        || acc.iter().map(Vec::len).sum::<usize>() > MAX_LITERALS
                    {
                        return nnf;
                    }
                }
                acc
            }
            ExprNode::BoolTrue => {
                if cnf {
                    Vec::new()
                } else {
                    vec![Vec::new()]
                }
            }
            ExprNode::BoolFalse => {
                if cnf {
                    vec![Vec::new()]
                } else {
                    Vec::new()
                }
            }
            _ => vec![vec![id]],
        };
        memo.insert(id, clauses);
    }
    let raw = memo.remove(&nnf).unwrap_or_default();

    // Clean clauses and drop subsumed / duplicate ones.
    let mut cleaned: Vec<Vec<ExprId>> = Vec::new();
    for c in &raw {
        if let Some(cc) = clean_clause(arena, c, cnf) {
            if cc.is_empty() {
                // Empty clause: identity of the inner connective, i.e. the
                // absorbing element of the outer one.
                return if cnf {
                    arena.bool_false
                } else {
                    arena.bool_true
                };
            }
            cleaned.push(cc);
        }
    }
    cleaned.sort_by_key(Vec::len);
    // Subsumption is quadratic; only worth it for modest clause counts.
    let check_subsumption = cleaned.len() <= 512;
    let mut kept: Vec<Vec<ExprId>> = Vec::new();
    let mut seen: FxHashSet<Vec<ExprId>> = FxHashSet::default();
    for c in cleaned {
        if !seen.insert(c.clone()) {
            continue;
        }
        let subsumed = check_subsumption && kept.iter().any(|k| k.iter().all(|l| c.contains(l)));
        if !subsumed {
            kept.push(c);
        }
    }
    let inner: Vec<ExprId> = kept
        .iter()
        .map(|c| if cnf { arena.or(c) } else { arena.and(c) })
        .collect();
    let mut inner = inner;
    inner.sort_by(|&x, &y| arena.sort_key(x).cmp(arena.sort_key(y)));
    if cnf {
        arena.and(&inner)
    } else {
        arena.or(&inner)
    }
}

/// Conjunctive normal form.
pub(crate) fn to_cnf(arena: &mut Arena, root: ExprId) -> ExprId {
    distribute(arena, root, true)
}

/// Disjunctive normal form.
pub(crate) fn to_dnf(arena: &mut Arena, root: ExprId) -> ExprId {
    distribute(arena, root, false)
}

// ═══════════════════════════════════════════════════════════════════════════
// Atoms
// ═══════════════════════════════════════════════════════════════════════════

/// Distinct atomic sub-formulas (relationals and opaque propositions), in
/// post-order.  Constants are excluded.
pub(crate) fn atoms(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    bool_post_order(arena, root)
        .into_iter()
        .filter(|&id| {
            !is_connective(arena, id)
                && !matches!(arena.node(id), ExprNode::BoolTrue | ExprNode::BoolFalse)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Propositional search
// ═══════════════════════════════════════════════════════════════════════════

enum FNode {
    Const(bool),
    Rel(usize, u8),
    Atom(usize, bool),
    And(Vec<usize>),
    Or(Vec<usize>),
}

struct Formula {
    nodes: Vec<FNode>,
    root: usize,
    pair_keys: Vec<(ExprId, ExprId)>,
    atom_keys: Vec<ExprId>,
}

impl Formula {
    fn build(arena: &Arena, root: ExprId) -> Formula {
        let order = bool_post_order(arena, root);
        let mut index: FxHashMap<ExprId, usize> = FxHashMap::default();
        let mut pair_index: FxHashMap<(ExprId, ExprId), usize> = FxHashMap::default();
        let mut atom_index: FxHashMap<ExprId, usize> = FxHashMap::default();
        let mut nodes: Vec<FNode> = Vec::with_capacity(order.len());
        let mut pair_keys = Vec::new();
        let mut atom_keys = Vec::new();

        for &id in &order {
            let node = match arena.node(id) {
                ExprNode::BoolTrue => FNode::Const(true),
                ExprNode::BoolFalse => FNode::Const(false),
                ExprNode::And(ch) => {
                    FNode::And(ch.iter().filter_map(|c| index.get(c).copied()).collect())
                }
                ExprNode::Or(ch) => {
                    FNode::Or(ch.iter().filter_map(|c| index.get(c).copied()).collect())
                }
                ExprNode::Not(x) => {
                    // NNF guarantees `x` is an atom.
                    match rel_of(arena, *x) {
                        Some(r) => {
                            let p = *pair_index.entry((r.a, r.b)).or_insert_with(|| {
                                pair_keys.push((r.a, r.b));
                                pair_keys.len() - 1
                            });
                            FNode::Rel(p, ALL & !r.mask)
                        }
                        None => {
                            let a = *atom_index.entry(*x).or_insert_with(|| {
                                atom_keys.push(*x);
                                atom_keys.len() - 1
                            });
                            FNode::Atom(a, false)
                        }
                    }
                }
                _ => match rel_of(arena, id) {
                    Some(r) => {
                        let p = *pair_index.entry((r.a, r.b)).or_insert_with(|| {
                            pair_keys.push((r.a, r.b));
                            pair_keys.len() - 1
                        });
                        FNode::Rel(p, r.mask)
                    }
                    None => {
                        let a = *atom_index.entry(id).or_insert_with(|| {
                            atom_keys.push(id);
                            atom_keys.len() - 1
                        });
                        FNode::Atom(a, true)
                    }
                },
            };
            index.insert(id, nodes.len());
            nodes.push(node);
        }
        let root = index.get(&root).copied().unwrap_or(0);
        Formula {
            nodes,
            root,
            pair_keys,
            atom_keys,
        }
    }

    fn var_count(&self) -> usize {
        self.pair_keys.len() + self.atom_keys.len()
    }
}

/// Outcome of one unit-propagation round.
enum Prop {
    /// The root is determined under the current assignment.
    Determined(bool),
    /// A forced literal contradicts the assignment.
    Conflict,
    /// Some variable was narrowed; propagate again.
    Changed,
    /// Nothing more can be inferred; branch.
    Stuck,
}

impl Formula {
    /// Bottom-up values of every node under a partial assignment.
    fn values(&self, pairs: &[u8], atoms: &[Option<bool>]) -> Vec<Option<bool>> {
        let mut vals: Vec<Option<bool>> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let v = match n {
                FNode::Const(b) => Some(*b),
                FNode::Rel(p, mask) => {
                    let s = pairs[*p];
                    if s & !mask == 0 {
                        Some(true)
                    } else if s & mask == 0 {
                        Some(false)
                    } else {
                        None
                    }
                }
                FNode::Atom(a, pos) => atoms[*a].map(|v| v == *pos),
                FNode::And(ch) => {
                    let mut acc = Some(true);
                    for &c in ch {
                        match vals[c] {
                            Some(false) => {
                                acc = Some(false);
                                break;
                            }
                            None => acc = None,
                            Some(true) => {}
                        }
                    }
                    acc
                }
                FNode::Or(ch) => {
                    let mut acc = Some(false);
                    for &c in ch {
                        match vals[c] {
                            Some(true) => {
                                acc = Some(true);
                                break;
                            }
                            None => acc = None,
                            Some(false) => {}
                        }
                    }
                    acc
                }
            };
            vals.push(v);
        }
        vals
    }

    /// Three-valued evaluation under a partial assignment.
    /// `pairs[i]` is the set of states still possible for pair `i`.
    fn eval(&self, pairs: &[u8], atoms: &[Option<bool>]) -> Option<bool> {
        self.values(pairs, atoms)[self.root]
    }

    /// One round of unit propagation towards making the root `target`.
    fn propagate(&self, target: bool, pairs: &mut [u8], atoms: &mut [Option<bool>]) -> Prop {
        let vals = self.values(pairs, atoms);
        if let Some(v) = vals[self.root] {
            return Prop::Determined(v);
        }
        // Top-down wants (children have smaller indices than parents).
        let mut want: Vec<Option<bool>> = vec![None; self.nodes.len()];
        want[self.root] = Some(target);
        let mut changed = false;
        for i in (0..self.nodes.len()).rev() {
            let Some(w) = want[i] else { continue };
            if vals[i].is_some() {
                continue;
            }
            match &self.nodes[i] {
                FNode::Const(_) => {}
                FNode::Rel(p, mask) => {
                    let m = if w { *mask } else { ALL & !mask };
                    let new = pairs[*p] & m;
                    if new == 0 {
                        return Prop::Conflict;
                    }
                    if new != pairs[*p] {
                        pairs[*p] = new;
                        changed = true;
                    }
                }
                FNode::Atom(a, pos) => {
                    let v = w == *pos;
                    match atoms[*a] {
                        Some(cur) if cur != v => return Prop::Conflict,
                        Some(_) => {}
                        None => {
                            atoms[*a] = Some(v);
                            changed = true;
                        }
                    }
                }
                FNode::And(ch) | FNode::Or(ch) => {
                    let is_and = matches!(self.nodes[i], FNode::And(_));
                    // "All children forced" when (And wants true) or (Or wants false);
                    // "single undetermined child forced" otherwise.
                    let all = is_and == w;
                    let undetermined: Vec<usize> =
                        ch.iter().copied().filter(|&c| vals[c].is_none()).collect();
                    let forced: &[usize] = if all || undetermined.len() == 1 {
                        &undetermined
                    } else {
                        &[]
                    };
                    for &c in forced {
                        match want[c] {
                            Some(prev) if prev != w => return Prop::Conflict,
                            _ => want[c] = Some(w),
                        }
                    }
                }
            }
        }
        if changed { Prop::Changed } else { Prop::Stuck }
    }
}

/// Search for an assignment under which the formula evaluates to
/// `target`.  Returns `Some(true)` if one exists, `Some(false)` if none
/// exists, `None` if the budget was exhausted.
fn search_for(f: &Formula, target: bool) -> Option<bool> {
    if f.var_count() > MAX_SAT_VARS {
        return None;
    }
    let pairs = vec![ALL; f.pair_keys.len()];
    let atoms = vec![None; f.atom_keys.len()];
    let mut budget = SAT_BUDGET;
    search_rec(f, target, pairs, atoms, &mut budget)
}

/// DPLL with unit propagation.  Recursion depth is bounded by the number
/// of variables (≤ 24), not by the size of the expression tree.
fn search_rec(
    f: &Formula,
    target: bool,
    mut pairs: Vec<u8>,
    mut atoms: Vec<Option<bool>>,
    budget: &mut usize,
) -> Option<bool> {
    loop {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        match f.propagate(target, &mut pairs, &mut atoms) {
            Prop::Determined(v) => return Some(v == target),
            Prop::Conflict => return Some(false),
            Prop::Changed => continue,
            Prop::Stuck => break,
        }
    }
    // Branch on the most constrained undecided pair first.
    let pick = pairs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.count_ones() > 1)
        .min_by_key(|(_, s)| s.count_ones())
        .map(|(i, _)| i);
    if let Some(i) = pick {
        let allowed = pairs[i];
        for state in [LT, EQ, GT] {
            if allowed & state == 0 {
                continue;
            }
            let mut p2 = pairs.clone();
            p2[i] = state;
            match search_rec(f, target, p2, atoms.clone(), budget) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => return None,
            }
        }
        return Some(false);
    }
    if let Some(i) = atoms.iter().position(Option::is_none) {
        for v in [true, false] {
            let mut a2 = atoms.clone();
            a2[i] = Some(v);
            match search_rec(f, target, pairs.clone(), a2, budget) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => return None,
            }
        }
        return Some(false);
    }
    // Fully assigned but undetermined cannot happen.
    None
}

/// Are the relational pairs mutually independent and unconstrained?
///
/// True when every pair `(a, b)` has `a - b` linear in a single free
/// symbol that occurs in no other pair.  Then every pair can realise each
/// of `<`, `=`, `>` independently of the others, and propositional
/// answers are exact.
fn pairs_independent(arena: &mut Arena, f: &Formula) -> bool {
    let mut used: FxHashSet<ExprId> = FxHashSet::default();
    for &(a, b) in &f.pair_keys {
        let d = arena.sub(a, b);
        let d = crate::transforms::eval::eval(arena, d);
        let syms = crate::base::walk::free_symbols(arena, d);
        if syms.len() != 1 || !used.insert(syms[0]) {
            return false;
        }
        match crate::poly::polybridge::expr_to_poly(arena, d, syms[0]) {
            Some(p) if p.degree() == Some(1) => {}
            _ => return false,
        }
    }
    true
}

/// If every relational atom of `root` is univariate in one common free
/// symbol, return that symbol.
fn single_variable(arena: &Arena, root: ExprId) -> Option<ExprId> {
    let syms = crate::base::walk::free_symbols(arena, root);
    if syms.len() != 1 {
        return None;
    }
    Some(syms[0])
}

/// Exact decision via the inequality solver for univariate formulas.
/// Returns `(is_empty, is_full)` of the solution set when it is exact.
fn univariate_status(arena: &mut Arena, root: ExprId) -> Option<(bool, bool)> {
    let var = single_variable(arena, root)?;
    let set = crate::transforms::sets::reduce_inequalities(arena, &[root], var).ok()?;
    let empty = crate::transforms::sets::is_empty(arena, set)?;
    let full = crate::transforms::sets::is_full(arena, set)?;
    // Only trust the answer when the set is exactly known.
    crate::transforms::sets::as_intervals(arena, set)?;
    Some((empty, full))
}

/// Is the formula true under every assignment?
pub(crate) fn is_tautology(arena: &mut Arena, root: ExprId) -> Option<bool> {
    let s = simplify_bool(arena, root);
    if s == arena.bool_true {
        return Some(true);
    }
    if s == arena.bool_false {
        return Some(false);
    }
    let f = Formula::build(arena, s);
    match search_for(&f, false) {
        Some(false) => return Some(true), // no falsifying assignment
        Some(true) if pairs_independent(arena, &f) => return Some(false),
        _ => {}
    }
    univariate_status(arena, s).map(|(_, full)| full)
}

/// Is the formula false under every assignment?
pub(crate) fn is_contradiction(arena: &mut Arena, root: ExprId) -> Option<bool> {
    let s = simplify_bool(arena, root);
    if s == arena.bool_false {
        return Some(true);
    }
    if s == arena.bool_true {
        return Some(false);
    }
    let f = Formula::build(arena, s);
    match search_for(&f, true) {
        Some(false) => return Some(true), // no satisfying assignment
        Some(true) if pairs_independent(arena, &f) => return Some(false),
        _ => {}
    }
    univariate_status(arena, s).map(|(empty, _)| empty)
}

/// Does some assignment make the formula true?
pub(crate) fn satisfiable(arena: &mut Arena, root: ExprId) -> Option<bool> {
    is_contradiction(arena, root).map(|c| !c)
}

/// Truth table over the given variables (≤ 8).
///
/// Rows whose variable values are mutually inconsistent (e.g. `x > 0`
/// and `x < 0` both `true`) are omitted.
pub(crate) fn truth_table(
    arena: &mut Arena,
    root: ExprId,
    vars: &[ExprId],
) -> Result<Vec<(Vec<bool>, bool)>, SymplexError> {
    if vars.len() > MAX_TRUTH_TABLE_VARS {
        return Err(SymplexError::InvalidArgument {
            operation: "truth_table",
            reason: format!(
                "at most {MAX_TRUTH_TABLE_VARS} variables supported, got {}",
                vars.len()
            ),
        });
    }
    let s = simplify_bool(arena, root);
    let f = Formula::build(arena, s);

    enum VarRef {
        Pair(usize, u8),
        Atom(usize),
        Absent,
    }
    let mut refs: Vec<VarRef> = Vec::with_capacity(vars.len());
    for &v in vars {
        if v == arena.bool_true || v == arena.bool_false {
            return Err(SymplexError::InvalidArgument {
                operation: "truth_table",
                reason: "constants are not variables".into(),
            });
        }
        let r = if let Some(rel) = rel_of(arena, v) {
            match f.pair_keys.iter().position(|k| *k == (rel.a, rel.b)) {
                Some(p) => VarRef::Pair(p, rel.mask),
                None => VarRef::Absent,
            }
        } else {
            let (atom, pos) = match arena.node(v) {
                ExprNode::Not(x) => (*x, false),
                _ => (v, true),
            };
            match f.atom_keys.iter().position(|k| *k == atom) {
                Some(a) if pos => VarRef::Atom(a),
                Some(_) => {
                    return Err(SymplexError::InvalidArgument {
                        operation: "truth_table",
                        reason: "negated atoms are not variables".into(),
                    });
                }
                None => VarRef::Absent,
            }
        };
        refs.push(r);
    }

    let n = vars.len();
    let mut rows = Vec::with_capacity(1 << n);
    for bits in 0..(1usize << n) {
        let mut pairs = vec![ALL; f.pair_keys.len()];
        let mut atoms = vec![None; f.atom_keys.len()];
        let mut values = Vec::with_capacity(n);
        let mut consistent = true;
        for (i, r) in refs.iter().enumerate() {
            // Most significant variable first, like a textbook table.
            let val = (bits >> (n - 1 - i)) & 1 == 1;
            values.push(val);
            match r {
                VarRef::Pair(p, mask) => {
                    let m = if val { *mask } else { ALL & !mask };
                    pairs[*p] &= m;
                    if pairs[*p] == 0 {
                        consistent = false;
                    }
                }
                VarRef::Atom(a) => atoms[*a] = Some(val),
                VarRef::Absent => {}
            }
        }
        if !consistent {
            continue;
        }
        match f.eval(&pairs, &atoms) {
            Some(v) => rows.push((values, v)),
            None => {
                return Err(SymplexError::ComputationFailed {
                    operation: "truth_table",
                    reason: "the formula is not determined by the given variables".into(),
                });
            }
        }
    }
    Ok(rows)
}

// ═══════════════════════════════════════════════════════════════════════════
// Assumption-aware evaluation
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate a boolean expression: exact numeric folding (via
/// [`eval`](crate::transforms::eval::eval)) followed by folding of
/// relationals whose truth follows from the assumption system
/// (`x > 0` is `true` when `x` is a positive symbol).
pub(crate) fn eval_bool(
    arena: &mut Arena,
    assumptions: &mut AssumptionCache,
    root: ExprId,
) -> ExprId {
    let e = crate::transforms::eval::eval(arena, root);
    let order = bool_post_order(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    let t = arena.bool_true;
    let f = arena.bool_false;

    for &id in &order {
        let new = match arena.node(id).clone() {
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                let d = arena.sub(a, b);
                let d = crate::transforms::eval::eval(arena, d);
                let (yes, no) = match arena.node(id) {
                    ExprNode::Gt(..) => (Props::POSITIVE, Props::NONPOSITIVE),
                    ExprNode::Ge(..) => (Props::NONNEGATIVE, Props::NEGATIVE),
                    ExprNode::Eq_(..) => (Props::ZERO, Props::NONZERO),
                    _ => (Props::NONZERO, Props::ZERO),
                };
                if assumptions.query(&*arena, d, yes) == Some(true) {
                    t
                } else if assumptions.query(&*arena, d, no) == Some(true) {
                    f
                } else {
                    id
                }
            }
            ExprNode::And(ch) => {
                let mut kids: SmallVec<[ExprId; 6]> = SmallVec::new();
                let mut any_false = false;
                for c in &ch {
                    let k = cache.get(c).copied().unwrap_or(*c);
                    if k == f {
                        any_false = true;
                        break;
                    }
                    if k != t {
                        kids.push(k);
                    }
                }
                if any_false {
                    f
                } else if kids.len() == ch.len() && kids.iter().zip(ch.iter()).all(|(a, b)| a == b)
                {
                    id
                } else {
                    arena.and(&kids)
                }
            }
            ExprNode::Or(ch) => {
                let mut kids: SmallVec<[ExprId; 6]> = SmallVec::new();
                let mut any_true = false;
                for c in &ch {
                    let k = cache.get(c).copied().unwrap_or(*c);
                    if k == t {
                        any_true = true;
                        break;
                    }
                    if k != f {
                        kids.push(k);
                    }
                }
                if any_true {
                    t
                } else if kids.len() == ch.len() && kids.iter().zip(ch.iter()).all(|(a, b)| a == b)
                {
                    id
                } else {
                    arena.or(&kids)
                }
            }
            ExprNode::Not(x) => {
                let nx = cache.get(&x).copied().unwrap_or(x);
                if nx == x { id } else { arena.not(nx) }
            }
            _ => id,
        };
        cache.insert(id, new);
    }
    cache.get(&e).copied().unwrap_or(e)
}

// ═══════════════════════════════════════════════════════════════════════════
// Piecewise
// ═══════════════════════════════════════════════════════════════════════════

/// Simplify every `Piecewise` node in an expression:
///
/// * conditions are simplified with [`simplify_bool`];
/// * branches with a `false` condition are dropped;
/// * evaluation stops at the first `true` condition;
/// * a branch whose condition repeats an earlier one is unreachable and
///   dropped;
/// * adjacent branches with identical values are merged (`c₁ ∨ c₂`);
/// * a single remaining branch with condition `true` collapses to its
///   value; no remaining branch collapses to `NaN`.
pub(crate) fn piecewise_simplify(arena: &mut Arena, root: ExprId) -> ExprId {
    let order = crate::base::walk::post_order_ids(arena, root);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
    for &id in &order {
        let new = match arena.node(id).clone() {
            ExprNode::Piecewise(pairs) => {
                let t = arena.bool_true;
                let f = arena.bool_false;
                let mut out: Vec<(ExprId, ExprId)> = Vec::new();
                let mut seen_conds: FxHashSet<ExprId> = FxHashSet::default();
                for &(v, c) in &pairs {
                    let nv = cache.get(&v).copied().unwrap_or(v);
                    let nc = cache.get(&c).copied().unwrap_or(c);
                    let nc = simplify_bool(arena, nc);
                    if nc == f || !seen_conds.insert(nc) {
                        continue;
                    }
                    if let Some(last) = out.last_mut()
                        && last.0 == nv
                    {
                        let merged = arena.or(&[last.1, nc]);
                        last.1 = simplify_bool(arena, merged);
                        if last.1 == t {
                            break;
                        }
                        continue;
                    }
                    out.push((nv, nc));
                    if nc == t {
                        break;
                    }
                }
                if out.is_empty() {
                    arena.nan
                } else if out.len() == 1 && out[0].1 == t {
                    out[0].0
                } else if out.len() == pairs.len()
                    && out.iter().zip(pairs.iter()).all(|(a, b)| a == b)
                {
                    id
                } else {
                    arena.piecewise(&out)
                }
            }
            _ => {
                if arena.node(id).is_atom() {
                    id
                } else {
                    crate::base::walk::rebuild_with_cache(arena, id, &cache)
                }
            }
        };
        cache.insert(id, new);
    }
    cache.get(&root).copied().unwrap_or(root)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    struct Fx {
        arena: Arena,
        x: ExprId,
        y: ExprId,
        zero: ExprId,
        one: ExprId,
    }

    fn fx() -> Fx {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let y = arena.symbol("y");
        let zero = arena.zero;
        let one = arena.one;
        Fx {
            arena,
            x,
            y,
            zero,
            one,
        }
    }

    #[test]
    fn relational_negation() {
        let mut f = fx();
        let gt = f.arena.gt(f.x, f.zero);
        let n = f.arena.not(gt);
        let s = simplify_bool(&mut f.arena, n);
        assert_eq!(display(&f.arena, s), "0 >= x");
        let ge = f.arena.ge(f.x, f.zero);
        let n = f.arena.not(ge);
        let s = simplify_bool(&mut f.arena, n);
        assert_eq!(display(&f.arena, s), "0 > x");
        let eq = f.arena.eq_(f.x, f.zero);
        let n = f.arena.not(eq);
        let s = simplify_bool(&mut f.arena, n);
        assert_eq!(display(&f.arena, s), "x != 0");
    }

    #[test]
    fn numeric_folding() {
        let mut f = fx();
        let two = f.arena.int(2);
        let g = f.arena.gt(two, f.one);
        assert_eq!(simplify_bool(&mut f.arena, g), f.arena.bool_true);
        let pi = f.arena.pi;
        let three = f.arena.int(3);
        let g = f.arena.gt(pi, three);
        assert_eq!(simplify_bool(&mut f.arena, g), f.arena.bool_true);
        // x + 1 > x  (exact difference)
        let xp1 = f.arena.add(&[f.x, f.one]);
        let g = f.arena.gt(xp1, f.x);
        assert_eq!(simplify_bool(&mut f.arena, g), f.arena.bool_true);
        // x > x is false, x >= x true
        let g = f.arena.gt(f.x, f.x);
        assert_eq!(simplify_bool(&mut f.arena, g), f.arena.bool_false);
        let g = f.arena.ge(f.x, f.x);
        assert_eq!(simplify_bool(&mut f.arena, g), f.arena.bool_true);
    }

    #[test]
    fn same_pair_merging() {
        let mut f = fx();
        let gt = f.arena.gt(f.x, f.zero);
        let ge = f.arena.ge(f.x, f.zero);
        let eq = f.arena.eq_(f.x, f.zero);
        let lt = f.arena.gt(f.zero, f.x);
        let le = f.arena.ge(f.zero, f.x);

        let a = f.arena.and(&[gt, ge]);
        assert_eq!(simplify_bool(&mut f.arena, a), gt);
        let o = f.arena.or(&[gt, eq]);
        assert_eq!(simplify_bool(&mut f.arena, o), ge);
        let a = f.arena.and(&[gt, lt]);
        assert_eq!(simplify_bool(&mut f.arena, a), f.arena.bool_false);
        let o = f.arena.or(&[gt, le]);
        assert_eq!(simplify_bool(&mut f.arena, o), f.arena.bool_true);
        let a = f.arena.and(&[ge, le]);
        assert_eq!(simplify_bool(&mut f.arena, a), eq);
    }

    #[test]
    fn flatten_dedupe_constants() {
        let mut f = fx();
        let p = f.arena.gt(f.x, f.zero);
        let q = f.arena.gt(f.y, f.zero);
        let inner = f.arena.and(&[p, q]);
        let t = f.arena.bool_true;
        let outer = f
            .arena
            .intern(ExprNode::And(smallvec::smallvec![inner, p, t]));
        let s = simplify_bool(&mut f.arena, outer);
        match f.arena.node(s) {
            ExprNode::And(ch) => assert_eq!(ch.len(), 2, "{}", display(&f.arena, s)),
            _ => panic!("expected And: {}", display(&f.arena, s)),
        }
        let fl = f.arena.bool_false;
        let dead = f.arena.intern(ExprNode::And(smallvec::smallvec![p, fl]));
        assert_eq!(simplify_bool(&mut f.arena, dead), fl);
        let o = f.arena.intern(ExprNode::Or(smallvec::smallvec![p, t]));
        assert_eq!(simplify_bool(&mut f.arena, o), t);
    }

    #[test]
    fn absorption_and_complement() {
        let mut f = fx();
        let p = f.arena.gt(f.x, f.zero);
        let q = f.arena.gt(f.y, f.zero);
        let pq = f.arena.or(&[p, q]);
        let a = f.arena.and(&[p, pq]);
        assert_eq!(simplify_bool(&mut f.arena, a), p, "A ∧ (A ∨ B) = A");
        let pq = f.arena.and(&[p, q]);
        let o = f.arena.or(&[p, pq]);
        assert_eq!(simplify_bool(&mut f.arena, o), p, "A ∨ (A ∧ B) = A");
        let np = f.arena.not(p);
        let a = f.arena.and(&[p, np]);
        assert_eq!(simplify_bool(&mut f.arena, a), f.arena.bool_false);
        let o = f.arena.or(&[p, np]);
        assert_eq!(simplify_bool(&mut f.arena, o), f.arena.bool_true);
        // A ∧ (¬A ∨ B) = A ∧ B
        let npq = f.arena.or(&[np, q]);
        let a = f.arena.and(&[p, npq]);
        let s = simplify_bool(&mut f.arena, a);
        let expected = simplify_bool(&mut f.arena, pq);
        assert_eq!(s, expected, "{}", display(&f.arena, s));
        // opaque atoms
        let s1 = f.arena.symbol("p");
        let ns1 = f.arena.not(s1);
        let a = f.arena.and(&[s1, ns1]);
        assert_eq!(simplify_bool(&mut f.arena, a), f.arena.bool_false);
    }

    #[test]
    fn de_morgan_and_double_negation() {
        let mut f = fx();
        let p = f.arena.gt(f.x, f.zero);
        let q = f.arena.gt(f.y, f.zero);
        let a = f.arena.and(&[p, q]);
        let n = f.arena.not(a);
        let s = simplify_bool(&mut f.arena, n);
        assert!(
            matches!(f.arena.node(s), ExprNode::Or(_)),
            "{}",
            display(&f.arena, s)
        );
        let d = display(&f.arena, s);
        assert!(d.contains("0 >= x") && d.contains("0 >= y"), "{d}");
        let n1 = f.arena.not(s);
        let nn = f.arena.not(n1);
        assert_eq!(simplify_bool(&mut f.arena, nn), s);
    }

    #[test]
    fn nnf_cnf_dnf() {
        let mut f = fx();
        let p = f.arena.symbol("p");
        let q = f.arena.symbol("q");
        let r = f.arena.symbol("r");
        // p ∨ (q ∧ r)  → CNF (p ∨ q) ∧ (p ∨ r)
        let qr = f.arena.and(&[q, r]);
        let e = f.arena.or(&[p, qr]);
        let c = to_cnf(&mut f.arena, e);
        match f.arena.node(c).clone() {
            ExprNode::And(ch) => {
                assert_eq!(ch.len(), 2);
                for k in ch {
                    assert!(matches!(f.arena.node(k), ExprNode::Or(_)));
                }
            }
            _ => panic!("expected CNF And: {}", display(&f.arena, c)),
        }
        assert_eq!(to_dnf(&mut f.arena, e), simplify_bool(&mut f.arena, e));
        // (p ∧ q) ∨ (p ∧ r) → DNF unchanged, CNF p ∧ (q ∨ r)
        let pq = f.arena.and(&[p, q]);
        let pr = f.arena.and(&[p, r]);
        let e2 = f.arena.or(&[pq, pr]);
        let c2 = to_cnf(&mut f.arena, e2);
        let qr_or = f.arena.or(&[q, r]);
        let expected = f.arena.and(&[p, qr_or]);
        let expected = simplify_bool(&mut f.arena, expected);
        assert_eq!(c2, expected, "{}", display(&f.arena, c2));
        // ¬(p ∧ q) NNF
        let n = f.arena.not(pq);
        let nnf = to_nnf(&mut f.arena, n);
        assert!(matches!(f.arena.node(nnf), ExprNode::Or(_)));
    }

    #[test]
    fn tautology_and_satisfiability() {
        let mut f = fx();
        let p = f.arena.symbol("p");
        let q = f.arena.symbol("q");
        let np = f.arena.not(p);
        let nq = f.arena.not(q);
        // (p → q) ∧ p → q
        let pq = f.arena.or(&[np, q]);
        let ante = f.arena.and(&[pq, p]);
        let nante = f.arena.not(ante);
        let mp = f.arena.or(&[nante, q]);
        assert_eq!(is_tautology(&mut f.arena, mp), Some(true));
        assert_eq!(is_contradiction(&mut f.arena, mp), Some(false));
        // p ∧ ¬p ∨ (q ∧ ¬q)
        let a = f.arena.and(&[p, np]);
        let b = f.arena.and(&[q, nq]);
        let c = f.arena.or(&[a, b]);
        assert_eq!(is_contradiction(&mut f.arena, c), Some(true));
        assert_eq!(satisfiable(&mut f.arena, c), Some(false));
        // p ∨ q : satisfiable, not a tautology
        let o = f.arena.or(&[p, q]);
        assert_eq!(satisfiable(&mut f.arena, o), Some(true));
        assert_eq!(is_tautology(&mut f.arena, o), Some(false));
        // relational: x > 1 → x > 0  (exact univariate route)
        let gt1 = f.arena.gt(f.x, f.one);
        let gt0 = f.arena.gt(f.x, f.zero);
        let ngt1 = f.arena.not(gt1);
        let imp = f.arena.or(&[ngt1, gt0]);
        assert_eq!(is_tautology(&mut f.arena, imp), Some(true));
        // x > 1 ∧ x < 0 : contradiction (not propositionally!)
        let lt0 = f.arena.gt(f.zero, f.x);
        let both = f.arena.and(&[gt1, lt0]);
        assert_eq!(is_contradiction(&mut f.arena, both), Some(true));
        assert_eq!(satisfiable(&mut f.arena, both), Some(false));
        // x > 0 alone: satisfiable, not tautology
        assert_eq!(satisfiable(&mut f.arena, gt0), Some(true));
        assert_eq!(is_tautology(&mut f.arena, gt0), Some(false));
        // independent linear pairs in distinct symbols: propositional answer is exact
        let gy = f.arena.gt(f.y, f.zero);
        let both = f.arena.and(&[gt0, gy]);
        assert_eq!(satisfiable(&mut f.arena, both), Some(true));
        assert_eq!(is_tautology(&mut f.arena, both), Some(false));
        // dependent multivariate relationals: honest None
        let gxy = f.arena.gt(f.x, f.y);
        let dep = f.arena.and(&[gxy, gt0]);
        assert_eq!(satisfiable(&mut f.arena, dep), None);
        assert_eq!(is_tautology(&mut f.arena, gxy), None);
        // non-linear pair is not "free": x² > 0 is not a tautology propositionally
        // but x² ≥ 0 is decided exactly through the solver
        let two = f.arena.int(2);
        let x2 = f.arena.pow(f.x, two);
        let ge0 = f.arena.ge(x2, f.zero);
        assert_eq!(is_tautology(&mut f.arena, ge0), Some(true));
        let lt0 = f.arena.gt(f.zero, x2);
        assert_eq!(satisfiable(&mut f.arena, lt0), Some(false));
    }

    #[test]
    fn unit_propagation_solves_pigeonhole() {
        let mut f = fx();
        let p = |arena: &mut Arena, i: usize, j: usize| {
            let s = arena.symbol(&format!("p{i}{j}"));
            let zero = arena.zero;
            arena.gt(s, zero)
        };
        let mut clauses: Vec<ExprId> = Vec::new();
        for i in 0..4 {
            let a = p(&mut f.arena, i, 0);
            let b = p(&mut f.arena, i, 1);
            let c = p(&mut f.arena, i, 2);
            clauses.push(f.arena.or(&[a, b, c]));
        }
        for j in 0..3 {
            for i in 0..4 {
                for k in (i + 1)..4 {
                    let a = p(&mut f.arena, i, j);
                    let b = p(&mut f.arena, k, j);
                    let na = f.arena.not(a);
                    let nb = f.arena.not(b);
                    clauses.push(f.arena.or(&[na, nb]));
                }
            }
        }
        let php = f.arena.and(&clauses);
        assert_eq!(satisfiable(&mut f.arena, php), Some(false));
        assert_eq!(is_contradiction(&mut f.arena, php), Some(true));
    }

    #[test]
    fn truth_table_basic() {
        let mut f = fx();
        let p = f.arena.symbol("p");
        let q = f.arena.symbol("q");
        let e = f.arena.and(&[p, q]);
        let rows = truth_table(&mut f.arena, e, &[p, q]).unwrap();
        assert_eq!(
            rows,
            vec![
                (vec![false, false], false),
                (vec![false, true], false),
                (vec![true, false], false),
                (vec![true, true], true),
            ]
        );
        // relational var and its negation are the same variable
        let gt = f.arena.gt(f.x, f.zero);
        let le = f.arena.ge(f.zero, f.x);
        let e = f.arena.or(&[gt, le]);
        let rows = truth_table(&mut f.arena, e, &[gt]).unwrap();
        assert!(rows.iter().all(|(_, v)| *v));
        // undetermined
        let e = f.arena.and(&[p, q]);
        assert!(truth_table(&mut f.arena, e, &[p]).is_err());
        let many: Vec<ExprId> = (0..9).map(|i| f.arena.symbol(&format!("v{i}"))).collect();
        assert!(truth_table(&mut f.arena, e, &many).is_err());
    }

    #[test]
    fn atoms_list() {
        let mut f = fx();
        let p = f.arena.gt(f.x, f.zero);
        let q = f.arena.symbol("q");
        let nq = f.arena.not(q);
        let t = f.arena.bool_true;
        let e = f.arena.and(&[p, nq, t]);
        let a = atoms(&f.arena, e);
        assert_eq!(a.len(), 2);
        assert!(a.contains(&p) && a.contains(&q));
    }

    #[test]
    fn eval_with_assumptions() {
        let mut f = fx();
        let mut cache = AssumptionCache::new();
        let sid = match f.arena.node(f.x) {
            ExprNode::Symbol(s) => *s,
            _ => unreachable!(),
        };
        let mut a = f.arena.symbol_assumptions(sid);
        a.assert_true(Props::POSITIVE);
        f.arena.set_symbol_assumptions(sid, a);
        let gt = f.arena.gt(f.x, f.zero);
        assert_eq!(eval_bool(&mut f.arena, &mut cache, gt), f.arena.bool_true);
        let le = f.arena.ge(f.zero, f.x);
        assert_eq!(eval_bool(&mut f.arena, &mut cache, le), f.arena.bool_false);
        let ne = f.arena.ne_(f.x, f.zero);
        assert_eq!(eval_bool(&mut f.arena, &mut cache, ne), f.arena.bool_true);
        // y is unconstrained
        let gy = f.arena.gt(f.y, f.zero);
        assert_eq!(eval_bool(&mut f.arena, &mut cache, gy), gy);
        let both = f.arena.and(&[gt, gy]);
        assert_eq!(eval_bool(&mut f.arena, &mut cache, both), gy);
    }

    #[test]
    fn piecewise_simplification() {
        let mut f = fx();
        let t = f.arena.bool_true;
        let fl = f.arena.bool_false;
        let gt = f.arena.gt(f.x, f.zero);
        let le = f.arena.ge(f.zero, f.x);
        let negx = f.arena.neg(f.x);
        // (x, false), (x, x>0), (x, x<=0) → x   (adjacent merge: x>0 ∨ x<=0 = true)
        let pw = f.arena.piecewise(&[(f.x, fl), (f.x, gt), (f.x, le)]);
        assert_eq!(piecewise_simplify(&mut f.arena, pw), f.x);
        // (x, x>0), (-x, true), (1, x>0) → stops at true, third dropped
        let pw = f.arena.piecewise(&[(f.x, gt), (negx, t), (f.one, gt)]);
        let s = piecewise_simplify(&mut f.arena, pw);
        match f.arena.node(s) {
            ExprNode::Piecewise(p) => assert_eq!(p.len(), 2),
            _ => panic!("expected Piecewise: {}", display(&f.arena, s)),
        }
        // nested inside an Add
        let inner = f.arena.piecewise(&[(f.one, t)]);
        let sum = f.arena.add(&[inner, f.x]);
        let s = piecewise_simplify(&mut f.arena, sum);
        let expected = f.arena.add(&[f.one, f.x]);
        assert_eq!(s, expected);
        // all branches false → NaN
        let pw = f.arena.piecewise(&[(f.x, fl)]);
        assert_eq!(piecewise_simplify(&mut f.arena, pw), f.arena.nan);
    }

    #[test]
    fn simplify_is_idempotent_and_sorted() {
        let mut f = fx();
        let p = f.arena.gt(f.x, f.zero);
        let q = f.arena.gt(f.y, f.zero);
        let a = f.arena.and(&[q, p]);
        let b = f.arena.and(&[p, q]);
        let sa = simplify_bool(&mut f.arena, a);
        let sb = simplify_bool(&mut f.arena, b);
        assert_eq!(sa, sb, "order-independent normal form");
        assert_eq!(simplify_bool(&mut f.arena, sa), sa);
    }
}
