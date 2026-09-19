//! Iterative tree traversal infrastructure.
//!
//! This module provides the shared building blocks for all tree-walking
//! operations in symplex: substitution, differentiation, pattern
//! matching, simplification, and any future tree transformations.
//!
//! **No recursion.**  All traversals use explicit stacks, guaranteeing
//! stack safety for arbitrarily deep expression trees (Principle 5).
//!
//! # Core functions
//!
//! - [`post_order_ids`] — compute a de-duplicated post-order traversal
//!   of the expression DAG.
//! - [`walk_and_rebuild`] — walk bottom-up, applying a transformation at
//!   each node, and rebuild the tree with canonical constructors.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::SymbolId;
use crate::base::node::{ExprId, ExprNode};

// ═══════════════════════════════════════════════════════════════════════════
// Post-order traversal
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a post-order traversal of the expression DAG rooted at `root`.
///
/// Each [`ExprId`] appears **at most once** in the output (de-duplicated
/// via a visited set).  Leaves appear before their parents, so any
/// bottom-up computation can process nodes in the returned order and
/// be guaranteed that all children are already processed.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn post_order_ids(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    // Stack entries: (id, children_pushed).
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];

    while let Some((id, children_pushed)) = stack.last_mut() {
        let id = *id;
        if visited.contains(&id) {
            stack.pop();
            continue;
        }

        if !*children_pushed {
            *children_pushed = true;
            let node = arena.node(id);
            let mut child_buf: SmallVec<[ExprId; 6]> = SmallVec::new();
            node.for_each_child(|c| child_buf.push(c));
            // Push children in reverse so they're processed left-to-right.
            for &child in child_buf.iter().rev() {
                if !visited.contains(&child) {
                    stack.push((child, false));
                }
            }
        } else {
            stack.pop();
            visited.insert(id);
            result.push(id);
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Walk-and-rebuild
// ═══════════════════════════════════════════════════════════════════════════

/// Walk an expression tree bottom-up and rebuild it with a
/// transformation function applied at each node.
///
/// `transform` is called on every node **after** its children have
/// been processed.  It returns `Some(new_id)` to replace the node, or
/// `None` to keep it (possibly with rebuilt children).
///
/// **This function never recurses.**  It uses [`post_order_ids`] for
/// traversal and an explicit cache for rebuilt results.
///
/// The rebuilt expression is re-canonicalized through the standard
/// `Arena` constructors (`add`, `mul`, `pow`, etc.), preserving all
/// canonical-form invariants.
///
/// If no transformation fires and no children change, the original
/// `root` [`ExprId`] is returned unchanged — no unnecessary allocation.
pub(crate) fn walk_and_rebuild(
    arena: &mut Arena,
    root: ExprId,
    transform: &dyn Fn(&Arena, ExprId) -> Option<ExprId>,
) -> ExprId {
    // Cache: original ExprId → rebuilt ExprId.
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    // Phase 1: Collect post-order traversal.
    let post_order = post_order_ids(arena, root);

    // Phase 2: Process each node in post-order (leaves first).
    for &id in &post_order {
        // Check the transform first.
        if let Some(replacement) = transform(arena, id) {
            cache.insert(id, replacement);
            continue;
        }

        // Atoms: no children to rebuild.
        if arena.node(id).is_atom() {
            cache.insert(id, id);
            continue;
        }

        // Rebuild with substituted children.
        let new_id = rebuild_with_cache(arena, id, &cache);
        cache.insert(id, new_id);
    }

    cache.get(&root).copied().unwrap_or(root)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rebuild helper
// ═══════════════════════════════════════════════════════════════════════════

/// Rebuild a single node with children looked up from the cache.
///
/// If all children map to themselves (no changes), returns the
/// original `id` unchanged — avoiding unnecessary allocation.
///
/// When children have changed, uses the canonical constructors
/// (`arena.add`, `arena.mul`, etc.) to maintain canonical form.
pub(crate) fn rebuild_with_cache(
    arena: &mut Arena,
    id: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
) -> ExprId {
    let node = arena.node(id).clone();

    macro_rules! rebuild_intern_unary {
        ($arena:expr, $id:expr, $inner:expr, $cache:expr, $Variant:ident) => {{
            let new_inner = $cache.get(&$inner).copied().unwrap_or($inner);
            if new_inner == $inner {
                $id
            } else {
                $arena.intern(ExprNode::$Variant(new_inner))
            }
        }};
    }

    match node {
        // Atoms: no children.
        ExprNode::Num(_)
        | ExprNode::Symbol(_)
        | ExprNode::Pi
        | ExprNode::E
        | ExprNode::ImaginaryUnit
        | ExprNode::EulerGamma
        | ExprNode::Catalan
        | ExprNode::GoldenRatio
        | ExprNode::PhysicalConstant(_, _)
        | ExprNode::Infinity
        | ExprNode::NegInfinity
        | ExprNode::ComplexInfinity
        | ExprNode::NaN
        | ExprNode::BoolTrue
        | ExprNode::BoolFalse
        | ExprNode::EmptySet
        | ExprNode::UniversalSet => id,

        // N-ary: Add, Mul
        ExprNode::Add(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.add(&new_children)
            }
        }

        ExprNode::Mul(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.mul(&new_children)
            }
        }

        // Binary: Pow, Derivative, Integral
        ExprNode::Pow(base, exp) => {
            let new_base = cache.get(&base).copied().unwrap_or(base);
            let new_exp = cache.get(&exp).copied().unwrap_or(exp);
            if new_base == base && new_exp == exp {
                id
            } else {
                arena.pow(new_base, new_exp)
            }
        }

        ExprNode::Derivative(body, var) => {
            let new_body = cache.get(&body).copied().unwrap_or(body);
            let new_var = cache.get(&var).copied().unwrap_or(var);
            if new_body == body && new_var == var {
                id
            } else {
                arena.intern(ExprNode::Derivative(new_body, new_var))
            }
        }

        ExprNode::Integral(body, var) => {
            let new_body = cache.get(&body).copied().unwrap_or(body);
            let new_var = cache.get(&var).copied().unwrap_or(var);
            if new_body == body && new_var == var {
                id
            } else {
                arena.intern(ExprNode::Integral(new_body, new_var))
            }
        }

        ExprNode::DefiniteIntegral(body, var, lo, hi) => {
            let nb = cache.get(&body).copied().unwrap_or(body);
            let nv = cache.get(&var).copied().unwrap_or(var);
            let nl = cache.get(&lo).copied().unwrap_or(lo);
            let nh = cache.get(&hi).copied().unwrap_or(hi);
            if nb == body && nv == var && nl == lo && nh == hi {
                id
            } else {
                arena.definite_integral(nb, nv, nl, nh)
            }
        }

        // Unary: Neg, Sin, Cos, Tan, Exp, Ln, Sqrt, Abs
        ExprNode::Neg(inner) => rebuild_unary(arena, id, inner, cache, Arena::neg),
        ExprNode::Sin(inner) => rebuild_unary(arena, id, inner, cache, Arena::sin),
        ExprNode::Cos(inner) => rebuild_unary(arena, id, inner, cache, Arena::cos),
        ExprNode::Tan(inner) => rebuild_unary(arena, id, inner, cache, Arena::tan),
        ExprNode::Exp(inner) => rebuild_unary(arena, id, inner, cache, Arena::exp),
        ExprNode::Ln(inner) => rebuild_unary(arena, id, inner, cache, Arena::ln),
        ExprNode::Abs(inner) => rebuild_unary(arena, id, inner, cache, Arena::abs),

        // Unary: Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
        ExprNode::Asin(inner) => rebuild_intern_unary!(arena, id, inner, cache, Asin),
        ExprNode::Acos(inner) => rebuild_intern_unary!(arena, id, inner, cache, Acos),
        ExprNode::Atan(inner) => rebuild_intern_unary!(arena, id, inner, cache, Atan),
        ExprNode::Atan2(y, x) => {
            let ny = cache.get(&y).copied().unwrap_or(y);
            let nx = cache.get(&x).copied().unwrap_or(x);
            if ny == y && nx == x {
                id
            } else {
                arena.atan2(ny, nx)
            }
        }
        ExprNode::Sinh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Sinh),
        ExprNode::Cosh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Cosh),
        ExprNode::Tanh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Tanh),
        ExprNode::Asinh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Asinh),
        ExprNode::Acosh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Acosh),
        ExprNode::Atanh(inner) => rebuild_intern_unary!(arena, id, inner, cache, Atanh),

        ExprNode::Sign(inner) => {
            let ni = cache.get(&inner).copied().unwrap_or(inner);
            if ni == inner { id } else { arena.sign(ni) }
        }

        ExprNode::Heaviside(inner) => rebuild_intern_unary!(arena, id, inner, cache, Heaviside),
        ExprNode::DiracDelta(inner) => rebuild_intern_unary!(arena, id, inner, cache, DiracDelta),

        // Complex analysis — go through the canonical constructors so that
        // e.g. `re(x)` with `x := 3 + 4i` collapses to `3`.
        ExprNode::Re(inner) => rebuild_unary(arena, id, inner, cache, Arena::re),
        ExprNode::Im(inner) => rebuild_unary(arena, id, inner, cache, Arena::im),
        ExprNode::Conjugate(inner) => rebuild_unary(arena, id, inner, cache, Arena::conjugate),
        ExprNode::Arg(inner) => rebuild_unary(arena, id, inner, cache, Arena::arg),

        // Special functions (0.2) — canonical constructors fold exact values.
        ExprNode::Si(inner) => rebuild_unary(arena, id, inner, cache, Arena::si),
        ExprNode::Ci(inner) => rebuild_unary(arena, id, inner, cache, Arena::ci),
        ExprNode::Ei(inner) => rebuild_unary(arena, id, inner, cache, Arena::ei),
        ExprNode::Li(inner) => rebuild_unary(arena, id, inner, cache, Arena::li),
        ExprNode::Zeta(inner) => rebuild_unary(arena, id, inner, cache, Arena::zeta),
        ExprNode::Polygamma(n, x) => {
            let nn = cache.get(&n).copied().unwrap_or(n);
            let nx = cache.get(&x).copied().unwrap_or(x);
            if nn == n && nx == x {
                id
            } else {
                arena.polygamma(nn, nx)
            }
        }
        ExprNode::KroneckerDelta(i, j) => {
            let ni = cache.get(&i).copied().unwrap_or(i);
            let nj = cache.get(&j).copied().unwrap_or(j);
            if ni == i && nj == j {
                id
            } else {
                arena.kronecker_delta(ni, nj)
            }
        }

        ExprNode::Gamma(inner) => rebuild_intern_unary!(arena, id, inner, cache, Gamma),
        ExprNode::LogGamma(inner) => rebuild_intern_unary!(arena, id, inner, cache, LogGamma),
        ExprNode::Digamma(inner) => rebuild_intern_unary!(arena, id, inner, cache, Digamma),
        ExprNode::Erf(inner) => rebuild_intern_unary!(arena, id, inner, cache, Erf),
        ExprNode::Erfc(inner) => rebuild_intern_unary!(arena, id, inner, cache, Erfc),
        ExprNode::LambertW(inner) => rebuild_intern_unary!(arena, id, inner, cache, LambertW),
        ExprNode::Beta(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.beta(na, nb)
            }
        }

        ExprNode::Floor(inner) => rebuild_intern_unary!(arena, id, inner, cache, Floor),
        ExprNode::Ceiling(inner) => rebuild_intern_unary!(arena, id, inner, cache, Ceiling),

        ExprNode::Min(ref children) => {
            let new_children: SmallVec<[ExprId; 4]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.intern(ExprNode::Min(new_children))
            }
        }

        ExprNode::Max(ref children) => {
            let new_children: SmallVec<[ExprId; 4]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.intern(ExprNode::Max(new_children))
            }
        }

        ExprNode::Sum(body, var, lo, hi) => {
            let nb = cache.get(&body).copied().unwrap_or(body);
            let nv = cache.get(&var).copied().unwrap_or(var);
            let nl = cache.get(&lo).copied().unwrap_or(lo);
            let nh = cache.get(&hi).copied().unwrap_or(hi);
            if nb == body && nv == var && nl == lo && nh == hi {
                id
            } else {
                arena.intern(ExprNode::Sum(nb, nv, nl, nh))
            }
        }

        ExprNode::Product_(body, var, lo, hi) => {
            let nb = cache.get(&body).copied().unwrap_or(body);
            let nv = cache.get(&var).copied().unwrap_or(var);
            let nl = cache.get(&lo).copied().unwrap_or(lo);
            let nh = cache.get(&hi).copied().unwrap_or(hi);
            if nb == body && nv == var && nl == lo && nh == hi {
                id
            } else {
                arena.intern(ExprNode::Product_(nb, nv, nl, nh))
            }
        }

        // Apply: user-defined function
        ExprNode::Apply(func_id, ref args) => {
            let new_args: SmallVec<[ExprId; 2]> = args
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_args == *args {
                id
            } else {
                arena.intern(ExprNode::Apply(func_id, new_args))
            }
        }

        // Combinatorial: Factorial, Binomial
        ExprNode::Factorial(inner) => rebuild_unary(arena, id, inner, cache, Arena::factorial),
        ExprNode::Binomial(n, k) => {
            let new_n = cache.get(&n).copied().unwrap_or(n);
            let new_k = cache.get(&k).copied().unwrap_or(k);
            if new_n == n && new_k == k {
                id
            } else {
                arena.binomial(new_n, new_k)
            }
        }

        // Binary relational: Gt, Ge, Eq_, Ne
        ExprNode::Gt(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.gt(na, nb)
            }
        }
        ExprNode::Ge(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.ge(na, nb)
            }
        }
        ExprNode::Eq_(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.eq_(na, nb)
            }
        }
        ExprNode::Ne(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.ne_(na, nb)
            }
        }

        // Unary logical: Not
        ExprNode::Not(inner) => rebuild_unary(arena, id, inner, cache, Arena::not),

        // N-ary logical: And, Or
        ExprNode::And(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.and(&new_children)
            }
        }
        ExprNode::Or(ref children) => {
            let new_children: SmallVec<[ExprId; 6]> = children
                .iter()
                .map(|&c| cache.get(&c).copied().unwrap_or(c))
                .collect();
            if new_children == *children {
                id
            } else {
                arena.or(&new_children)
            }
        }

        // Piecewise
        ExprNode::Piecewise(ref pairs) => {
            let new_pairs: SmallVec<[(ExprId, ExprId); 3]> = pairs
                .iter()
                .map(|&(val, cond)| {
                    let nv = cache.get(&val).copied().unwrap_or(val);
                    let nc = cache.get(&cond).copied().unwrap_or(cond);
                    (nv, nc)
                })
                .collect();
            if new_pairs == *pairs {
                id
            } else {
                arena.intern(ExprNode::Piecewise(new_pairs))
            }
        }

        // Set constructors
        ExprNode::Interval(a, b, flags) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.intern(ExprNode::Interval(na, nb, flags))
            }
        }

        ExprNode::FiniteSet(ref elems) => {
            let new_elems: SmallVec<[ExprId; 4]> = elems
                .iter()
                .map(|&e| cache.get(&e).copied().unwrap_or(e))
                .collect();
            if new_elems == *elems {
                id
            } else {
                arena.intern(ExprNode::FiniteSet(new_elems))
            }
        }

        ExprNode::SetUnion(ref sets) => {
            let new_sets: SmallVec<[ExprId; 4]> = sets
                .iter()
                .map(|&s| cache.get(&s).copied().unwrap_or(s))
                .collect();
            if new_sets == *sets {
                id
            } else {
                arena.intern(ExprNode::SetUnion(new_sets))
            }
        }

        ExprNode::SetIntersection(ref sets) => {
            let new_sets: SmallVec<[ExprId; 4]> = sets
                .iter()
                .map(|&s| cache.get(&s).copied().unwrap_or(s))
                .collect();
            if new_sets == *sets {
                id
            } else {
                arena.intern(ExprNode::SetIntersection(new_sets))
            }
        }

        ExprNode::SetComplement(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.intern(ExprNode::SetComplement(na, nb))
            }
        }

        // 3-field formal nodes
        ExprNode::Limit(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::Limit(na, nb, nc))
            }
        }

        ExprNode::LaplaceTransform(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::LaplaceTransform(na, nb, nc))
            }
        }

        ExprNode::InverseLaplaceTransform(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::InverseLaplaceTransform(na, nb, nc))
            }
        }

        ExprNode::Residue(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::Residue(na, nb, nc))
            }
        }

        ExprNode::DSolve(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::DSolve(na, nb, nc))
            }
        }

        ExprNode::RootSum(a, b, c) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            if na == a && nb == b && nc == c {
                id
            } else {
                arena.intern(ExprNode::RootSum(na, nb, nc))
            }
        }

        // 4-field formal node
        ExprNode::Series(a, b, c, d) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            let nc = cache.get(&c).copied().unwrap_or(c);
            let nd = cache.get(&d).copied().unwrap_or(d);
            if na == a && nb == b && nc == c && nd == d {
                id
            } else {
                arena.intern(ExprNode::Series(na, nb, nc, nd))
            }
        }

        // 2-field formal nodes
        ExprNode::RootOf(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.intern(ExprNode::RootOf(na, nb))
            }
        }

        ExprNode::ConditionSet(a, b) => {
            let na = cache.get(&a).copied().unwrap_or(a);
            let nb = cache.get(&b).copied().unwrap_or(b);
            if na == a && nb == b {
                id
            } else {
                arena.intern(ExprNode::ConditionSet(na, nb))
            }
        }
    }
}

/// Helper for rebuilding unary nodes — avoids repeating the same
/// pattern 8 times.
#[inline]
fn rebuild_unary(
    arena: &mut Arena,
    id: ExprId,
    inner: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
    constructor: fn(&mut Arena, ExprId) -> ExprId,
) -> ExprId {
    let new_inner = cache.get(&inner).copied().unwrap_or(inner);
    if new_inner == inner {
        id
    } else {
        constructor(arena, new_inner)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: check if an expression contains a given sub-expression
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `true` if `needle` appears anywhere in the expression tree
/// rooted at `haystack`.
///
/// Uses an explicit stack — never recurses.
// Used for structural queries.
pub(crate) fn contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }

    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![haystack];

    while let Some(id) = stack.pop() {
        if id == needle {
            return true;
        }
        if visited.contains(&id) {
            continue;
        }
        visited.insert(id);

        arena.node(id).for_each_child(|c| stack.push(c));
    }

    false
}

/// Collect the [`ExprId`] of every symbol that appears anywhere in the
/// expression tree, ignoring binders.  Each symbol appears at most once.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn all_symbols(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let mut result = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack: Vec<ExprId> = vec![root];

    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if matches!(arena.node(id), ExprNode::Symbol(_)) {
            result.push(id);
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }

    result
}

/// Collect the [`ExprId`] of every *free* symbol that appears in the
/// expression tree.  Each symbol appears at most once.
///
/// Binders hide their variable inside the sub-expression they scope over
/// (the variable remains free in the other operands):
///
/// | Node                          | Bound in                                 |
/// |-------------------------------|------------------------------------------|
/// | `Sum(body, var, lo, hi)`      | `body` (not `lo`, `hi`)                  |
/// | `Product_(body, var, lo, hi)` | `body` (not `lo`, `hi`)                  |
/// | `DefiniteIntegral(body, var, lo, hi)` | `body` (not `lo`, `hi`)          |
/// | `RootSum(poly, body, var)`    | `body` (`poly` is a polynomial in `var`) |
/// | `ConditionSet(var, cond)`     | `cond`                                   |
/// | `RootOf(poly, idx)`           | `poly`, when it has exactly one symbol   |
///
/// `Integral`, `Derivative`, `Limit`, `Series`, … do **not** bind: their
/// variable is a genuine free symbol of the (anti)derivative / limit form.
///
/// Because the arena is a hash-consed DAG, the same node can occur both
/// under a binder and outside it (`x + Σ_{x=0}^{3} x`), so visited-ness is
/// tracked per *(node, scope)* pair rather than per node.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn free_symbols(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    // Scope 0 is the empty scope; every other scope is a sorted list of
    // bound symbol ids.  Scopes are interned so that `(node, scope)` is a
    // cheap hashable key.
    let mut scopes: Vec<SmallVec<[ExprId; 4]>> = vec![SmallVec::new()];
    let mut result = Vec::new();
    let mut visited: FxHashSet<(ExprId, u32)> = FxHashSet::default();
    let mut seen_syms: FxHashSet<SymbolId> = FxHashSet::default();
    let mut stack: Vec<(ExprId, u32)> = vec![(root, 0)];

    // Return the scope `scopes[sc] ∪ {var}` (interned).
    fn extend_scope(scopes: &mut Vec<SmallVec<[ExprId; 4]>>, sc: u32, var: ExprId) -> u32 {
        if scopes[sc as usize].contains(&var) {
            return sc;
        }
        let mut new_scope = scopes[sc as usize].clone();
        new_scope.push(var);
        new_scope.sort_unstable();
        if let Some(pos) = scopes.iter().position(|s| *s == new_scope) {
            return pos as u32;
        }
        scopes.push(new_scope);
        (scopes.len() - 1) as u32
    }

    while let Some((id, sc)) = stack.pop() {
        if !visited.insert((id, sc)) {
            continue;
        }

        match arena.node(id) {
            ExprNode::Symbol(sid) => {
                if !scopes[sc as usize].contains(&id) && seen_syms.insert(*sid) {
                    result.push(id);
                }
            }
            ExprNode::Sum(body, var, lo, hi)
            | ExprNode::Product_(body, var, lo, hi)
            | ExprNode::DefiniteIntegral(body, var, lo, hi) => {
                stack.push((*lo, sc));
                stack.push((*hi, sc));
                let inner = extend_scope(&mut scopes, sc, *var);
                stack.push((*body, inner));
            }
            ExprNode::RootSum(poly, body, var) => {
                let inner = extend_scope(&mut scopes, sc, *var);
                stack.push((*poly, inner));
                stack.push((*body, inner));
            }
            ExprNode::ConditionSet(var, cond) => {
                let inner = extend_scope(&mut scopes, sc, *var);
                stack.push((*cond, inner));
            }
            ExprNode::RootOf(poly, idx) => {
                stack.push((*idx, sc));
                // The polynomial's variable is bound.  It is only
                // well-defined when the polynomial is univariate; anything
                // else is reported conservatively (all symbols free).
                let poly_syms = all_symbols(arena, *poly);
                let inner = if let [var] = poly_syms[..] {
                    extend_scope(&mut scopes, sc, var)
                } else {
                    sc
                };
                stack.push((*poly, inner));
            }
            node => node.for_each_child(|c| stack.push((c, sc))),
        }
    }

    result
}

/// Returns `true` if the expression tree rooted at `root` contains any
/// unevaluated formal node: `Integral`, `DefiniteIntegral`, `Derivative`,
/// `Limit`, `Series`, `LaplaceTransform`, `InverseLaplaceTransform`,
/// `Residue`, `DSolve`, `ConditionSet`, formal `Sum`, or formal `Product_`.
///
/// `RootOf` and `RootSum` are **not** counted: they are complete algebraic
/// answers (an exact description of a polynomial root / a sum over all
/// roots), not pending computations.  Likewise, function nodes such as
/// `Re`, `Im`, `Si`, `Zeta`, … are ordinary functions and never count as
/// unevaluated.
pub(crate) fn has_unevaluated(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut visited = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Integral(..)
            | ExprNode::DefiniteIntegral(..)
            | ExprNode::Derivative(..)
            | ExprNode::Limit(..)
            | ExprNode::Series(..)
            | ExprNode::LaplaceTransform(..)
            | ExprNode::InverseLaplaceTransform(..)
            | ExprNode::Residue(..)
            | ExprNode::DSolve(..)
            | ExprNode::ConditionSet(..)
            | ExprNode::Sum(..)
            | ExprNode::Product_(..) => return true,
            node => {
                for child in node.children() {
                    stack.push(child);
                }
            }
        }
    }
    false
}

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

    // ── post_order_ids ──────────────────────────────────────────────

    #[test]
    fn post_order_atom() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let order = post_order_ids(&a, x);
        assert_eq!(order, vec![x]);
    }

    #[test]
    fn post_order_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        let order = post_order_ids(&a, sum);
        // Children before parent. x and y come before sum.
        assert!(
            order.iter().position(|&id| id == x).unwrap()
                < order.iter().position(|&id| id == sum).unwrap()
        );
        assert!(
            order.iter().position(|&id| id == y).unwrap()
                < order.iter().position(|&id| id == sum).unwrap()
        );
    }

    #[test]
    fn post_order_deduplicates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + x canonicalizes to 2*x which has x as a child.
        // x should appear only once in the traversal.
        let expr = a.add(&[x, x]);
        let order = post_order_ids(&a, expr);
        let x_count = order.iter().filter(|&&id| id == x).count();
        assert_eq!(x_count, 1, "x should appear exactly once");
    }

    // ── walk_and_rebuild ────────────────────────────────────────────

    #[test]
    fn walk_identity_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let xp = a.pow(x, two);
        let one = a.one;
        let expr = a.add(&[xp, x, one]);
        let result = walk_and_rebuild(&mut a, expr, &|_, _| None);
        assert_eq!(result, expr, "identity walk should return same ExprId");
    }

    #[test]
    fn walk_replace_leaf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let result = walk_and_rebuild(&mut a, expr, &|_, id| {
            if id == x { Some(y) } else { None }
        });
        assert_eq!(display(&a, result), "y^2");
    }

    #[test]
    fn walk_deep_no_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // 1000-deep sin chain.
        let mut expr = x;
        for _ in 0..1000 {
            expr = a.sin(expr);
        }
        let result = walk_and_rebuild(&mut a, expr, &|_, id| {
            if id == x { Some(y) } else { None }
        });
        assert_ne!(result, expr);
    }

    // ── contains ────────────────────────────────────────────────────

    #[test]
    fn contains_self() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        assert!(contains(&a, x, x));
    }

    #[test]
    fn contains_child() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sum = a.add(&[x, y]);
        assert!(contains(&a, sum, x));
        assert!(contains(&a, sum, y));
    }

    #[test]
    fn does_not_contain() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let z = sym(&mut a, "z");
        let sum = a.add(&[x, y]);
        assert!(!contains(&a, sum, z));
    }

    #[test]
    fn contains_deep() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let xp = a.pow(x, two);
        let expr = a.sin(xp);
        assert!(contains(&a, expr, x));
        assert!(contains(&a, expr, xp));
    }

    // ── free_symbols ────────────────────────────────────────────────

    #[test]
    fn free_symbols_atom() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let syms = free_symbols(&a, x);
        assert_eq!(syms.len(), 1);
    }

    #[test]
    fn free_symbols_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, y]);
        let syms = free_symbols(&a, expr);
        assert_eq!(syms.len(), 2);
    }

    #[test]
    fn free_symbols_no_duplicates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + x = 2*x, which contains x once.
        let expr = a.add(&[x, x]);
        let syms = free_symbols(&a, expr);
        assert_eq!(syms.len(), 1, "x should appear once even in 2*x");
    }

    #[test]
    fn free_symbols_constant_has_none() {
        let a = Arena::new();
        let syms = free_symbols(&a, a.pi);
        assert!(syms.is_empty(), "pi has no free symbols");
    }

    #[test]
    fn free_symbols_sum_binds_index_but_not_bounds() {
        let mut a = Arena::new();
        let k = sym(&mut a, "k");
        let n = sym(&mut a, "n");
        let body = a.sin(k);
        let s = a.intern(ExprNode::Sum(body, k, a.one, n));
        assert_eq!(free_symbols(&a, s), vec![n]);
        let p = a.intern(ExprNode::Product_(body, k, a.one, n));
        assert_eq!(free_symbols(&a, p), vec![n]);
        // The index variable in a bound is the *outer* k.
        let s2 = a.intern(ExprNode::Sum(body, k, a.one, k));
        assert_eq!(free_symbols(&a, s2), vec![k]);
    }

    #[test]
    fn free_symbols_same_node_bound_and_free() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let s = a.intern(ExprNode::Sum(x, x, a.zero, three));
        assert!(free_symbols(&a, s).is_empty());
        // x + Σ_{x=0}^{3} x : the first x is free even though the same
        // arena node is bound inside the sum.
        let e = a.add(&[x, s]);
        assert_eq!(free_symbols(&a, e), vec![x]);
        let e2 = a.add(&[s, x]);
        assert_eq!(free_symbols(&a, e2), vec![x]);
    }

    #[test]
    fn free_symbols_rootof_and_rootsum_bind_polynomial_variable() {
        let mut a = Arena::new();
        let lam = sym(&mut a, "lambda");
        let three = a.int(3);
        let l3 = a.pow(lam, three);
        let neg_lam = a.neg(lam);
        let poly = a.add(&[l3, neg_lam, a.neg_one]);
        let root = a.intern(ExprNode::RootOf(poly, a.zero));
        assert!(free_symbols(&a, root).is_empty(), "RootOf is a constant");
        // Root index is not scoped.
        let n = sym(&mut a, "n");
        let root_n = a.intern(ExprNode::RootOf(poly, n));
        assert_eq!(free_symbols(&a, root_n), vec![n]);
        // RootSum(poly(t), body(t, x), t): only x is free.
        let t = sym(&mut a, "t");
        let x = sym(&mut a, "x");
        let t3 = a.pow(t, three);
        let poly_t = a.add(&[t3, t, a.one]);
        let tx = a.mul(&[t, x]);
        let body = a.ln(tx);
        let rs = a.intern(ExprNode::RootSum(poly_t, body, t));
        assert_eq!(free_symbols(&a, rs), vec![x]);
    }

    #[test]
    fn free_symbols_rootof_multivariate_polynomial_is_conservative() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = sym(&mut a, "p");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let px = a.mul(&[p, x]);
        let poly = a.add(&[x5, px, a.one]);
        let root = a.intern(ExprNode::RootOf(poly, a.zero));
        let mut syms = free_symbols(&a, root);
        syms.sort_unstable();
        let mut expected = vec![x, p];
        expected.sort_unstable();
        assert_eq!(syms, expected);
    }

    #[test]
    fn free_symbols_condition_set_binds_variable() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let sx = a.sin(x);
        let cond = a.gt(sx, y);
        let cs = a.intern(ExprNode::ConditionSet(x, cond));
        assert_eq!(free_symbols(&a, cs), vec![y]);
    }

    #[test]
    fn free_symbols_integral_and_derivative_do_not_bind() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sx = a.sin(x);
        let integral = a.intern(ExprNode::Integral(sx, x));
        assert_eq!(free_symbols(&a, integral), vec![x]);
        let deriv = a.intern(ExprNode::Derivative(sx, x));
        assert_eq!(free_symbols(&a, deriv), vec![x]);
    }

    #[test]
    fn free_symbols_nested_binders() {
        let mut a = Arena::new();
        let i = sym(&mut a, "i");
        let j = sym(&mut a, "j");
        let n = sym(&mut a, "n");
        let ij = a.mul(&[i, j]);
        let inner = a.intern(ExprNode::Sum(ij, j, a.one, i));
        assert_eq!(free_symbols(&a, inner), vec![i]);
        let outer = a.intern(ExprNode::Sum(inner, i, a.one, n));
        assert_eq!(free_symbols(&a, outer), vec![n]);
    }

    // ── has_unevaluated ────────────────────────────────────────────────

    #[test]
    fn has_unevaluated_formal_nodes() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let integral = a.intern(ExprNode::Integral(x, x));
        assert!(has_unevaluated(&a, integral));
        let deriv = a.intern(ExprNode::Derivative(x, x));
        assert!(has_unevaluated(&a, deriv));
        let lim = a.intern(ExprNode::Limit(x, x, a.zero));
        assert!(has_unevaluated(&a, lim));
        // Nested inside an Add.
        let sum = a.add(&[integral, a.one]);
        assert!(has_unevaluated(&a, sum));
    }

    #[test]
    fn root_of_and_root_sum_are_not_unevaluated() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let poly = a.add(&[x5, x, a.one]);
        let root = a.intern(ExprNode::RootOf(poly, a.zero));
        assert!(
            !has_unevaluated(&a, root),
            "RootOf is a complete algebraic answer"
        );
        let t = sym(&mut a, "t");
        let body = a.ln(t);
        let rs = a.intern(ExprNode::RootSum(poly, body, t));
        assert!(!has_unevaluated(&a, rs), "RootSum is not unevaluated");
    }

    #[test]
    fn function_nodes_are_not_unevaluated() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        for id in [
            a.intern(ExprNode::Re(x)),
            a.intern(ExprNode::Im(x)),
            a.intern(ExprNode::Conjugate(x)),
            a.intern(ExprNode::Arg(x)),
            a.intern(ExprNode::Si(x)),
            a.intern(ExprNode::Ci(x)),
            a.intern(ExprNode::Ei(x)),
            a.intern(ExprNode::Li(x)),
            a.intern(ExprNode::Zeta(x)),
            a.intern(ExprNode::Polygamma(a.one, x)),
            a.intern(ExprNode::KroneckerDelta(x, a.one)),
        ] {
            assert!(!has_unevaluated(&a, id), "{:?} is a function", a.node(id));
        }
    }
}
