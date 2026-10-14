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

use crate::base::arena::{Arena, SHARED_SYMBOL_BIT};
use crate::base::node::SymbolId;
use crate::base::node::{ExprId, ExprNode};
use crate::base::truth::Truth;

// ═══════════════════════════════════════════════════════════════════════════
// Post-order traversal
// ═══════════════════════════════════════════════════════════════════════════

/// The largest list and visited set [`post_order_ids`] allocates up front
/// (a larger DAG grows them as before).
const POST_ORDER_PRESIZE: usize = 256;

/// Compute a post-order traversal of the expression DAG rooted at `root`.
///
/// Each [`ExprId`] appears **at most once** in the output (de-duplicated
/// via a visited set).  Leaves appear before their parents, so any
/// bottom-up computation can process nodes in the returned order and
/// be guaranteed that all children are already processed.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn post_order_ids(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    // The DAG has at most as many nodes as the unfolded tree, whose size
    // the node facts know: a small tree (most are) gets its list and
    // visited set allocated once, at their final size.
    let tree_size = arena.facts(root).tree_size as usize;
    if tree_size == 1 {
        return vec![root];
    }
    let hint = tree_size.min(POST_ORDER_PRESIZE);
    let mut result = Vec::with_capacity(hint);
    let mut visited: FxHashSet<ExprId> =
        FxHashSet::with_capacity_and_hasher(hint, Default::default());
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
    rebuild_with(arena, id, &|c| cache.get(&c).copied().unwrap_or(c))
}

/// [`rebuild_with_cache`] with the rebuilt form of each child given by
/// `get` (a child that `get` maps to itself is unchanged).  Every child
/// operand is looked up, including the variable operand of a binder; a
/// pass that treats binders specially (substitution) rebuilds them with
/// [`rebuild_binder`] instead.  A node whose children changed is built by
/// [`Arena::make`], i.e. by its variant's canonical constructor: `{x, y}`
/// with `x, y := 1` is `{1}` and `[x, y]` with `x := 3, y := 2` is `∅`,
/// `==` to the sets built directly.
pub(crate) fn rebuild_with<F: Fn(ExprId) -> ExprId>(
    arena: &mut Arena,
    id: ExprId,
    get: &F,
) -> ExprId {
    match arena.node(id).map_children(get) {
        None => id,
        Some(node) => arena.make(node),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: check if an expression contains a given sub-expression
// ═══════════════════════════════════════════════════════════════════════════

/// Returns `true` if `needle` appears anywhere in the expression tree
/// rooted at `haystack`.
///
/// A symbol in a tree without binders is found by its free-symbol bit
/// ([`NodeFacts`](crate::base::arena::NodeFacts)); otherwise subtrees that
/// are too small or too low to hold `needle`, or lack a structure it has,
/// are not entered.
///
/// Uses an explicit stack — never recurses.
// Used for structural queries.
pub(crate) fn contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }
    let inner = arena.facts(needle);
    if !arena.facts(haystack).may_contain(inner) {
        return false;
    }
    if let ExprNode::Symbol(sym) = *arena.node(needle)
        && !arena.facts(haystack).has_binder()
    {
        // Without binders every occurrence is free.
        return has_free_symbol(arena, haystack, sym);
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

        arena.node(id).for_each_child(|c| {
            if arena.facts(c).may_contain(inner) {
                stack.push(c);
            }
        });
    }

    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Binders
// ═══════════════════════════════════════════════════════════════════════════

/// A node that binds a variable: `var` is bound in the `scoped` operands
/// and the `outer` operands belong to the enclosing scope.  See [`binder`].
#[derive(Clone, Debug)]
pub(crate) struct Binder {
    /// The bound variable (a `Symbol` node).
    pub(crate) var: ExprId,
    /// The operands in which `var` is bound, in node order.
    pub(crate) scoped: SmallVec<[ExprId; 2]>,
    /// The operands in the enclosing scope, in node order.
    pub(crate) outer: SmallVec<[ExprId; 2]>,
}

/// The binder table: which variable `id` binds, and where.  `None` for a
/// node that binds nothing.
///
/// | Node                                   | Binds  | in            | outer      |
/// |----------------------------------------|--------|---------------|------------|
/// | `Sum(body, var, lo, hi)`               | `var`  | `body`        | `lo`, `hi` |
/// | `Product_(body, var, lo, hi)`          | `var`  | `body`        | `lo`, `hi` |
/// | `DefiniteIntegral(body, var, lo, hi)`  | `var`  | `body`        | `lo`, `hi` |
/// | `RootSum(poly, body, var)`             | `var`  | `poly`, `body`| —          |
/// | `ConditionSet(var, cond)`              | `var`  | `cond`        | —          |
/// | `ImageSet(var, body)`                  | `var`  | `body`        | —          |
/// | `RootOf(poly, var, idx)`               | `var`  | `poly`        | `idx`      |
/// | `Limit(body, var, point)`              | `var`  | `body`        | `point`    |
/// | `Residue(body, var, point)`            | `var`  | `body`        | `point`    |
/// | `LaplaceTransform(body, t, s)`         | `t`    | `body`        | `s`        |
/// | `InverseLaplaceTransform(body, s, t)`  | `s`    | `body`        | `t`        |
/// | `Subs(body, var, point)`               | `var`  | `body`        | `point`    |
///
/// `lim_{x→a} f(x)`, `Res_{z=a} f`, `ℒ{f(t)}(s)`, `ℒ⁻¹{F(s)}(t)` and
/// `f′(x)|_{x=a}` are functions of `a`, `s`, `t` — not of the dummy
/// (SymPy's `free_symbols` agrees for `Limit`, `Subs` and the integral
/// transforms).  An indefinite `Integral(body, var)`, a `Derivative(body,
/// var)`, a `Series(body, var, point, order)` and a `DSolve` do **not**
/// bind: `∫ f dx`, `f′(x)`, a series in `x` and the solution `y(x)` are
/// functions of `x` (see [`var_slot`] for how substitution treats them).
/// A binder whose variable operand is not a symbol is malformed and binds
/// nothing.
///
/// Every binder-aware pass — [`free_symbols`], [`has_free_symbol`],
/// substitution (`transforms::subs`), the integrator's dependence test —
/// reads this one table, which is the `#[binder(..)]` attributes of the
/// `ExprNode` declaration (`ExprNode::binder_operands`).
pub(crate) fn binder(arena: &Arena, id: ExprId) -> Option<Binder> {
    arena
        .node(id)
        .binder_operands()
        .filter(|b| matches!(arena.node(b.var), ExprNode::Symbol(_)))
}

/// The binder `id` (for which [`binder`] is `Some`) rebuilt with bound
/// variable `var` and the given `scoped` / `outer` operands, in the order
/// [`binder`] lists them, through [`Arena::make`] (a `DefiniteIntegral`
/// applies its cheap folds, a `Subs` its normal form; the other binders
/// have no canonical constructor and are interned as they are).  Returns
/// `id` for a node that is not a binder.
pub(crate) fn rebuild_binder(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    scoped: &[ExprId],
    outer: &[ExprId],
) -> ExprId {
    match arena.node(id).with_binder_operands(var, scoped, outer) {
        Some(node) => arena.make(node),
        None => id,
    }
}

/// The variable slot of a node that is a function *of* its variable:
/// `var` is bound in the `scoped` operands as for a binder, and free in
/// the node itself.  `None` for any other node, or for a malformed one
/// whose variable operand is not a symbol.
///
/// | Node                               | `var`  | in              | outer            |
/// |------------------------------------|--------|-----------------|------------------|
/// | `Derivative(body, var)`            | `var`  | `body`          | —                |
/// | `Integral(body, var)`              | `var`  | `body`          | —                |
/// | `Series(body, var, point, order)`  | `var`  | `body`          | `point`, `order` |
/// | `DSolve(eq, func, var)`            | `var`  | `eq`, `func`    | —                |
///
/// `f′(x)` is `(λx. f′(x))` applied to `x`, so replacing `x` by a value `a`
/// is not a rewrite of both operands (`Derivative(f(0), 0)` is
/// meaningless) but the evaluation of that function at `a`:
/// `Subs(Derivative(f(x), x), x, a)`.  The substitution pass
/// (`transforms::subs`) reads this table to rename the variable when `a`
/// is a fresh symbol, and to build the [`subs_node`] otherwise.  For every
/// other pass the node binds nothing: [`free_symbols`] of `f′(x)` is `{x}`.
/// The table is the `#[var_slot(..)]` attributes of the `ExprNode`
/// declaration (`ExprNode::var_slot_operands`).
pub(crate) fn var_slot(arena: &Arena, id: ExprId) -> Option<Binder> {
    arena
        .node(id)
        .var_slot_operands()
        .filter(|b| matches!(arena.node(b.var), ExprNode::Symbol(_)))
}

/// The node `id` (for which [`var_slot`] is `Some`) rebuilt with variable
/// `var` and the given `scoped` / `outer` operands, in the order
/// [`var_slot`] lists them, through [`Arena::make`].  Returns `id` for any
/// other node.
pub(crate) fn rebuild_var_slot(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    scoped: &[ExprId],
    outer: &[ExprId],
) -> ExprId {
    match arena.node(id).with_var_slot_operands(var, scoped, outer) {
        Some(node) => arena.make(node),
        None => id,
    }
}

/// `Subs(body, var, point)`, with the folds that make it a normal form:
/// `body` itself when `point` is `var` or when `var` is not free in
/// `body`.  A `var` that is not a symbol is malformed and interned as it
/// is.  (This is the node constructor; substituting `point` into `body`
/// where that is possible is the substitution pass's business.)
pub(crate) fn subs_node(arena: &mut Arena, body: ExprId, var: ExprId, point: ExprId) -> ExprId {
    if let ExprNode::Symbol(sym) = *arena.node(var)
        && (point == var || !has_free_symbol(arena, body, sym))
    {
        return body;
    }
    arena.intern(ExprNode::Subs(body, var, point))
}

/// A node of the tree rooted at `root` whose variable operand is not a
/// symbol (`Sum(1/9*k, 1/3=0..3)`: a malformed binder), skipping the
/// nodes of the tree rooted at `except`.  `None` if there is none.
pub(crate) fn malformed_binder(arena: &Arena, root: ExprId, except: ExprId) -> Option<ExprId> {
    let old: FxHashSet<ExprId> = post_order_ids(arena, except).into_iter().collect();
    post_order_ids(arena, root).into_iter().find(|id| {
        !old.contains(id)
            && arena
                .node(*id)
                .var_operand()
                .is_some_and(|v| !matches!(arena.node(v), ExprNode::Symbol(_)))
    })
}

/// The variable a `RootOf` of `poly` takes when none is written: the one
/// free symbol of `poly`, or `None` when it has none or several (the
/// variable is then ambiguous and must be given, as in `RootOf(x⁵ − a·x +
/// 1, x, 0)`).  The printers write the variable exactly when this does
/// not name it, and the parser reads the two-argument form through it.
pub(crate) fn root_of_implied_var(arena: &Arena, poly: ExprId) -> Option<ExprId> {
    match free_symbols(arena, poly)[..] {
        [var] => Some(var),
        _ => None,
    }
}

/// Does the symbol `sym` occur *free* in the expression rooted at `root`?
///
/// Binders are those of [`binder`]: the variable of a `Sum` is bound in
/// the body and free in the limits, a `RootOf(x⁵ − x + 1, 0)` is a
/// constant, and so on.  Only one symbol is looked for, so a node is
/// either entered with `sym` free or not entered at all, and per-node
/// visited-ness suffices.
///
/// O(1) from the free-symbol mask of the
/// [`NodeFacts`](crate::base::arena::NodeFacts) unless `sym` shares bit 63;
/// then the walk skips the subtrees whose mask lacks that bit.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn has_free_symbol(arena: &Arena, root: ExprId, sym: SymbolId) -> bool {
    match arena.free_symbol_fact(root, sym) {
        Truth::True => return true,
        Truth::False => return false,
        Truth::Unknown => {}
    }
    let bit = SHARED_SYMBOL_BIT;
    let mut stack: Vec<ExprId> = vec![root];
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    while let Some(id) = stack.pop() {
        if arena.free_symbol_mask(id) & bit == 0 || !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Symbol(s) => {
                if *s == sym {
                    return true;
                }
            }
            node if node.is_atom() => {}
            node => match binder(arena, id) {
                Some(b) => {
                    stack.extend_from_slice(&b.outer);
                    if !matches!(arena.node(b.var), ExprNode::Symbol(s) if *s == sym) {
                        stack.extend_from_slice(&b.scoped);
                    }
                }
                None => node.for_each_child(|c| stack.push(c)),
            },
        }
    }
    false
}

/// [`has_free_symbol`] for a variable given as its `Symbol` node: does
/// `var` occur free in the tree rooted at `root`?  A `var` that is not a
/// symbol is looked for as a sub-expression.
pub(crate) fn has_free_var(arena: &Arena, root: ExprId, var: ExprId) -> bool {
    match *arena.node(var) {
        ExprNode::Symbol(sym) => has_free_symbol(arena, root, sym),
        _ => contains(arena, root, var),
    }
}

/// Collect the [`ExprId`] of every *free* symbol that appears in the
/// expression tree.  Each symbol appears at most once.
///
/// Binders (the table of [`binder`]) hide their variable inside the
/// operands they scope over; the variable remains free in the other
/// operands (`Σ_{k=1}^{k} f(k)` has the free symbol `k`, from its upper
/// limit).
///
/// Because the arena is a hash-consed DAG, the same node can occur both
/// under a binder and outside it (`x + Σ_{x=0}^{3} x`), so visited-ness is
/// tracked per *(node, scope)* pair rather than per node.
///
/// The symbols are listed in the order the depth-first walk first meets
/// them.  The free-symbol masks of the
/// [`NodeFacts`](crate::base::arena::NodeFacts) answer a constant and a tree
/// with one free symbol in O(1); otherwise the walk does not enter a subtree
/// whose symbols have all been found, and stops when the mask of `root`
/// says all have (which changes neither the list nor its order).
///
/// Uses an explicit stack — never recurses.
pub(crate) fn free_symbols(arena: &Arena, root: ExprId) -> Vec<ExprId> {
    let all = arena.free_symbol_mask(root);
    if all == 0 {
        return Vec::new();
    }
    if all & SHARED_SYMBOL_BIT == 0
        && all.is_power_of_two()
        && let Some(sym) = arena.bit_symbol(all.trailing_zeros())
    {
        return vec![sym];
    }
    // Scope 0 is the empty scope; every other scope is a sorted list of
    // bound symbol ids.  Scopes are interned so that `(node, scope)` is a
    // cheap hashable key.
    let mut scopes: Vec<SmallVec<[ExprId; 4]>> = vec![SmallVec::new()];
    let mut result = Vec::new();
    let mut visited: FxHashSet<(ExprId, u32)> = FxHashSet::default();
    let mut seen_syms: FxHashSet<SymbolId> = FxHashSet::default();
    // The bits of the symbols found (bit 63 never: a subtree that may hold
    // a symbol sharing it is always entered).
    let mut found: u64 = 0;
    let mut stack: Vec<(ExprId, u32)> = vec![(root, 0)];

    // Return the scope `scopes[sc] ∪ {var}` (interned).
    fn extend_scope(scopes: &mut Vec<SmallVec<[ExprId; 4]>>, sc: u32, var: ExprId) -> u32 {
        if scopes[sc as usize].contains(&var) {
            return sc;
        }
        let mut new_scope = scopes[sc as usize].clone();
        new_scope.push(var);
        new_scope.sort_unstable_by_key(|&v| crate::base::arena::ArenaOrder(v));
        if let Some(pos) = scopes.iter().position(|s| *s == new_scope) {
            return pos as u32;
        }
        scopes.push(new_scope);
        (scopes.len() - 1) as u32
    }

    while let Some((id, sc)) = stack.pop() {
        if all & !found == 0 {
            break;
        }
        if arena.free_symbol_mask(id) & !found == 0 || !visited.insert((id, sc)) {
            continue;
        }

        match arena.node(id) {
            ExprNode::Symbol(sid) => {
                if !scopes[sc as usize].contains(&id) && seen_syms.insert(*sid) {
                    result.push(id);
                    found |= arena.free_symbol_mask(id) & !SHARED_SYMBOL_BIT;
                }
            }
            node if node.is_atom() => {}
            node => match binder(arena, id) {
                Some(b) => {
                    for &c in &b.outer {
                        stack.push((c, sc));
                    }
                    let inner = extend_scope(&mut scopes, sc, b.var);
                    for &c in &b.scoped {
                        stack.push((c, inner));
                    }
                }
                None => node.for_each_child(|c| stack.push((c, sc))),
            },
        }
    }

    result
}

/// Returns `true` if the expression tree rooted at `root` contains any
/// unevaluated formal node: `Integral`, `DefiniteIntegral`, `Derivative`,
/// `Limit`, `Series`, `LaplaceTransform`, `InverseLaplaceTransform`,
/// `Residue`, `DSolve`, `ConditionSet`, `Subs`, formal `Sum`, or formal
/// `Product_`.
///
/// `RootOf` and `RootSum` are **not** counted: they are complete algebraic
/// answers (an exact description of a polynomial root / a sum over all
/// roots), not pending computations.  Likewise, function nodes such as
/// `Re`, `Im`, `Si`, `Zeta`, … are ordinary functions and never count as
/// unevaluated.
///
/// O(1): a flag of the node's [`NodeFacts`](crate::base::arena::NodeFacts),
/// set when the node is interned.
#[inline]
pub(crate) fn has_unevaluated(arena: &Arena, root: ExprId) -> bool {
    arena.facts(root).has_unevaluated()
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
        let root = a.intern(ExprNode::RootOf(poly, lam, a.zero));
        assert!(free_symbols(&a, root).is_empty(), "RootOf is a constant");
        // Root index is not scoped.
        let n = sym(&mut a, "n");
        let root_n = a.intern(ExprNode::RootOf(poly, lam, n));
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
    fn free_symbols_rootof_multivariate_polynomial_binds_its_variable_only() {
        // Up to 0.28 a `RootOf` did not record its variable, so one whose
        // polynomial had a parameter bound nothing: `x` counted as free.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let p = sym(&mut a, "p");
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let px = a.mul(&[p, x]);
        let poly = a.add(&[x5, px, a.one]);
        let root = a.intern(ExprNode::RootOf(poly, x, a.zero));
        assert_eq!(free_symbols(&a, root), vec![p]);
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

    #[test]
    fn free_symbols_limit_residue_and_transforms_bind_their_dummy() {
        // SymPy 1.14: `Limit(sin(x*y)/x, x, 0).free_symbols` is `{y}`,
        // `LaplaceTransform(exp(a*t), t, s).free_symbols` is `{a, s}`.
        let mut a = Arena::new();
        let (x, y, t, s) = (
            sym(&mut a, "x"),
            sym(&mut a, "y"),
            sym(&mut a, "t"),
            sym(&mut a, "s"),
        );
        let xy = a.mul(&[x, y]);
        let lim = a.intern(ExprNode::Limit(xy, x, a.zero));
        assert_eq!(free_symbols(&a, lim), vec![y]);
        let res = a.intern(ExprNode::Residue(xy, x, y));
        assert_eq!(free_symbols(&a, res), vec![y]);
        let ty = a.mul(&[t, y]);
        let lt = a.intern(ExprNode::LaplaceTransform(ty, t, s));
        let by_id = |v: &ExprId| crate::base::arena::ArenaOrder(*v);
        let mut got = free_symbols(&a, lt);
        got.sort_unstable_by_key(by_id);
        let mut want = vec![y, s];
        want.sort_unstable_by_key(by_id);
        assert_eq!(got, want);
        let ilt = a.intern(ExprNode::InverseLaplaceTransform(ty, t, s));
        let mut got = free_symbols(&a, ilt);
        got.sort_unstable_by_key(by_id);
        assert_eq!(got, want);
        // A limit point in the dummy is the outer variable.
        let lim_x = a.intern(ExprNode::Limit(xy, x, x));
        let mut got = free_symbols(&a, lim_x);
        got.sort_unstable_by_key(by_id);
        let mut want = vec![x, y];
        want.sort_unstable_by_key(by_id);
        assert_eq!(got, want);
    }

    #[test]
    fn has_free_symbol_agrees_with_free_symbols() {
        let mut a = Arena::new();
        let (x, k, n) = (sym(&mut a, "x"), sym(&mut a, "k"), sym(&mut a, "n"));
        let ExprNode::Symbol(x_sym) = *a.node(x) else {
            unreachable!()
        };
        let three = a.int(3);
        let xk = a.mul(&[x, k]);
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let neg_x = a.neg(x);
        let poly = a.add(&[x5, neg_x, a.one]);
        let root = a.intern(ExprNode::RootOf(poly, x, a.zero));
        let sum_x = a.intern(ExprNode::Sum(xk, x, a.zero, three));
        let sum_k = a.intern(ExprNode::Sum(xk, k, a.zero, n));
        let sum_x_to_x = a.intern(ExprNode::Sum(xk, x, a.zero, x));
        let integral = a.intern(ExprNode::Integral(xk, x));
        let shared = a.add(&[x, sum_x]);
        for (e, free) in [
            (root, false),
            (sum_x, false),
            (sum_k, true),
            (sum_x_to_x, true),
            (integral, true),
            (shared, true),
        ] {
            assert_eq!(has_free_symbol(&a, e, x_sym), free, "{}", display(&a, e));
            assert_eq!(free_symbols(&a, e).contains(&x), free, "{}", display(&a, e));
        }
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
        let root = a.intern(ExprNode::RootOf(poly, x, a.zero));
        assert!(
            !has_unevaluated(&a, root),
            "RootOf is a complete algebraic answer"
        );
        let t = sym(&mut a, "t");
        let body = a.ln(t);
        let rs = a.intern(ExprNode::RootSum(poly, body, t));
        assert!(!has_unevaluated(&a, rs), "RootSum is not unevaluated");
    }

    // ── the node facts agree with the walks they replaced ──────────────

    /// `has_unevaluated` as the walk of 0.49.
    fn walk_has_unevaluated(a: &Arena, root: ExprId) -> bool {
        let mut stack = vec![root];
        let mut visited = FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if matches!(
                a.node(id),
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
                    | ExprNode::Subs(..)
                    | ExprNode::Sum(..)
                    | ExprNode::Product_(..)
            ) {
                return true;
            }
            stack.extend(a.node(id).children());
        }
        false
    }

    /// `has_free_symbol` as the walk of 0.49.
    fn walk_has_free_symbol(a: &Arena, root: ExprId, sym: SymbolId) -> bool {
        let mut stack = vec![root];
        let mut visited = FxHashSet::default();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            match a.node(id) {
                ExprNode::Symbol(s) if *s == sym => return true,
                node if node.is_atom() => {}
                node => match binder(a, id) {
                    Some(b) => {
                        stack.extend_from_slice(&b.outer);
                        if !matches!(a.node(b.var), ExprNode::Symbol(s) if *s == sym) {
                            stack.extend_from_slice(&b.scoped);
                        }
                    }
                    None => node.for_each_child(|c| stack.push(c)),
                },
            }
        }
        false
    }

    /// `free_symbols` as the walk of 0.49, in its order.
    fn walk_free_symbols(a: &Arena, root: ExprId) -> Vec<ExprId> {
        let mut scopes: Vec<SmallVec<[ExprId; 4]>> = vec![SmallVec::new()];
        let mut result = Vec::new();
        let mut visited: FxHashSet<(ExprId, u32)> = FxHashSet::default();
        let mut seen: FxHashSet<SymbolId> = FxHashSet::default();
        let mut stack: Vec<(ExprId, u32)> = vec![(root, 0)];
        while let Some((id, sc)) = stack.pop() {
            if !visited.insert((id, sc)) {
                continue;
            }
            match a.node(id) {
                ExprNode::Symbol(s) => {
                    if !scopes[sc as usize].contains(&id) && seen.insert(*s) {
                        result.push(id);
                    }
                }
                node if node.is_atom() => {}
                node => match binder(a, id) {
                    Some(b) => {
                        for &c in &b.outer {
                            stack.push((c, sc));
                        }
                        let mut scope = scopes[sc as usize].clone();
                        if !scope.contains(&b.var) {
                            scope.push(b.var);
                            scope.sort_unstable_by_key(|&v| crate::base::arena::ArenaOrder(v));
                        }
                        let inner = match scopes.iter().position(|s| *s == scope) {
                            Some(p) => p as u32,
                            None => {
                                scopes.push(scope);
                                (scopes.len() - 1) as u32
                            }
                        };
                        for &c in &b.scoped {
                            stack.push((c, inner));
                        }
                    }
                    None => node.for_each_child(|c| stack.push((c, sc))),
                },
            }
        }
        result
    }

    /// `contains` as the walk of 0.49.
    fn walk_contains(a: &Arena, haystack: ExprId, needle: ExprId) -> bool {
        let mut stack = vec![haystack];
        let mut visited = FxHashSet::default();
        while let Some(id) = stack.pop() {
            if id == needle {
                return true;
            }
            if visited.insert(id) {
                a.node(id).for_each_child(|c| stack.push(c));
            }
        }
        false
    }

    /// The unfolded tree size and the height of `root`, by a memoised walk.
    fn walk_size_height(a: &Arena, root: ExprId) -> (u128, u32) {
        let mut memo: FxHashMap<ExprId, (u128, u32)> = FxHashMap::default();
        for id in post_order_ids(a, root) {
            let (mut size, mut height) = (1u128, 0u32);
            a.node(id).for_each_child(|c| {
                let (s, h) = memo[&c];
                size += s;
                height = height.max(h);
            });
            memo.insert(id, (size, height + 1));
        }
        memo[&root]
    }

    /// Every query the facts answer, on every node of `a`, against the
    /// walks of 0.49.
    fn assert_facts_agree(a: &Arena) {
        let ids: Vec<ExprId> = (0..a.node_count() as u32).map(ExprId).collect();
        let symbols: Vec<(ExprId, SymbolId)> = ids
            .iter()
            .filter_map(|&id| match *a.node(id) {
                ExprNode::Symbol(s) => Some((id, s)),
                _ => None,
            })
            .collect();
        for &id in &ids {
            // Built only on failure: an exponential tree prints exponentially.
            let what = || format!("{id:?} = {:?}", a.node(id));
            assert_eq!(
                has_unevaluated(a, id),
                walk_has_unevaluated(a, id),
                "{}",
                what()
            );
            let free = walk_free_symbols(a, id);
            assert_eq!(free_symbols(a, id), free, "{}", what());
            assert_eq!(a.is_symbol_free(id), free.is_empty(), "{}", what());
            let any_symbol = symbols.iter().any(|&(s, _)| walk_contains(a, id, s));
            assert_eq!(a.facts(id).has_symbol(), any_symbol, "{}", what());
            for &(s_id, s) in &symbols {
                let free_s = walk_has_free_symbol(a, id, s);
                assert_eq!(has_free_symbol(a, id, s), free_s, "{s_id:?} in {}", what());
                assert_eq!(free.contains(&s_id), free_s, "{s_id:?} in {}", what());
                let has_s = walk_contains(a, id, s_id);
                assert_eq!(contains(a, id, s_id), has_s, "{s_id:?} in {}", what());
            }
            let (size, height) = walk_size_height(a, id);
            let f = a.facts(id);
            assert_eq!(
                u128::from(f.tree_size),
                size.min(u32::MAX.into()),
                "{}",
                what()
            );
            assert_eq!(
                u32::from(f.height),
                height.min(u16::MAX.into()),
                "{}",
                what()
            );
            for cap in [0, 1, 7, 1000, usize::MAX] {
                let want = usize::try_from(size).unwrap_or(usize::MAX).min(cap);
                assert_eq!(a.tree_size_capped(id, cap), want, "{}", what());
            }
        }
        if ids.len() <= 600 {
            for &h in &ids {
                for &n in &ids {
                    assert_eq!(contains(a, h, n), walk_contains(a, h, n));
                }
            }
        }
    }

    /// An arena with every binder of the table, shadowing, a malformed
    /// binder, a physical constant (its value is not a child) and more
    /// than 63 symbols, so binders of symbols sharing the free-symbol bit
    /// 63: the facts must answer exactly what the walks answer.
    #[test]
    fn node_facts_agree_with_the_walks_on_every_node() {
        let mut a = Arena::new();
        let s: Vec<ExprId> = (0..70).map(|i| sym(&mut a, &format!("s{i}"))).collect();
        let (x, y, t) = (s[0], s[1], s[2]);
        let (k, j) = (s[69], s[68]);
        let f = a.symbols.intern("f");
        let app =
            |a: &mut Arena, arg: ExprId| a.intern(ExprNode::Apply(f, smallvec::smallvec![arg]));
        let three = a.int(3);
        let five = a.int(5);
        let fk = app(&mut a, k);
        let fk_j = a.mul(&[fk, j]);
        // Shared-bit binders: k bound, j free; then j bound too (bit 63
        // stays set: the facts must walk to see that nothing is free).
        let sum1 = a.intern(ExprNode::Sum(fk_j, k, a.one, j));
        let sum2 = a.intern(ExprNode::Sum(sum1, j, a.one, three));
        let kx = a.add(&[k, x]);
        let sum3 = a.intern(ExprNode::Sum(kx, k, a.zero, y));
        // Shadowing with an own-bit symbol.
        let inner = a.intern(ExprNode::Sum(x, x, a.one, x));
        let outer = a.intern(ExprNode::Sum(inner, x, a.one, three));
        // The other binders.
        let x5 = a.pow(x, five);
        let poly = a.add(&[x5, x, a.one]);
        let root = a.intern(ExprNode::RootOf(poly, x, a.zero));
        let t5 = a.pow(t, five);
        let poly_t = a.add(&[t5, y]);
        let ty = a.mul(&[t, y]);
        let body = a.ln(ty);
        let root_sum = a.intern(ExprNode::RootSum(poly_t, body, t));
        let cond = a.gt(x, y);
        let cset = a.intern(ExprNode::ConditionSet(x, cond));
        let image_body = a.mul(&[k, a.pi]);
        let image = a.intern(ExprNode::ImageSet(k, image_body));
        let xy = a.mul(&[x, y]);
        let lim = a.intern(ExprNode::Limit(xy, x, x));
        let fx = app(&mut a, x);
        let deriv = a.intern(ExprNode::Derivative(fx, x));
        let subs = a.intern(ExprNode::Subs(deriv, x, a.zero));
        let integral = a.intern(ExprNode::Integral(ty, t));
        let lt = a.intern(ExprNode::LaplaceTransform(ty, t, j));
        // A malformed binder (its variable is a number) binds nothing.
        let bad = a.intern(ExprNode::Sum(kx, three, a.zero, k));
        let c = a.symbols.intern("c");
        let phys = a.intern(ExprNode::PhysicalConstant(c, x));
        let all = a.add(&[
            sum2, sum3, outer, root, root_sum, lim, subs, integral, lt, bad, phys, s[66],
        ]);
        let _ = a.mul(&[all, cset, image]);
        assert_facts_agree(&a);
    }

    /// Unfolded trees beyond `u32::MAX` nodes: the size saturates and
    /// [`Arena::tree_size_capped`] recounts for a larger cap.
    #[test]
    fn node_facts_saturate_on_exponential_trees() {
        let mut a = Arena::new();
        let mut e = sym(&mut a, "x");
        for _ in 0..40 {
            let (s, c) = (a.sin(e), a.cos(e));
            e = a.add(&[s, c]);
        }
        assert_eq!(a.facts(e).tree_size, u32::MAX);
        assert_facts_agree(&a);
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
