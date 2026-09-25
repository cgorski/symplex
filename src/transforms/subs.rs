//! Structural substitution.
//!
//! This module provides [`subs`] (structural replacement) and
//! [`subs_map`] (simultaneous multi-replacement).
//!
//! # Design
//!
//! **Structural substitution** replaces exact node matches only — it
//! never performs algebraic reasoning.  `(1/x).subs(x², 1)` returns
//! `1/x` unchanged because `x²` does not appear as a node in `1/x`.
//! This is safe by construction: it can never produce mathematically
//! wrong results.  Use the future `.alg_subs()` for algebraic
//! substitution with documented caveats.
//!
//! **No recursive tree walks.**  All traversals are bottom-up over an
//! explicit post-order ([`crate::base::walk::post_order_ids`]) with a
//! rebuilt-children cache.  Stack overflow is impossible regardless of
//! expression depth.
//!
//! **Binders.**  Only *free* occurrences are replaced.  A node of the
//! binder table ([`crate::base::walk::binder`]: `Sum`, `Product_`,
//! `DefiniteIntegral`, `RootSum`, `ConditionSet`, a univariate `RootOf`,
//! `Limit`, `Residue`, the Laplace transforms) binds its variable in some
//! operands, so substitution is the capture-avoiding one of the λ-calculus:
//!
//! - *Shadowing.*  Under a binder of `v`, a replacement whose `old` has `v`
//!   free no longer applies: its `v` is the outer one.  The operands in
//!   the enclosing scope (the limits of a `Sum`) are substituted as usual.
//!   `∫₀ˣ x² dx` with `x ↦ 3` is `∫₀³ x² dx`; `RootOf(x⁵ − x + 1, 0)` with
//!   `x ↦ 1/3` is itself.
//! - *Capture.*  A replacement that brings a free `v` into the scope of a
//!   binder of `v` renames the binder first, to `v_1` (or the first `v_n`
//!   that is not in use): `Σ_{k=0}^{n} x·k` with `x ↦ k` is
//!   `Σ_{k_1=0}^{n} k·k_1`, not `Σ k²`.
//!
//! The first rule is SymPy's behaviour (`ExprWithLimits._eval_subs` in
//! `sympy/concrete/expr_with_limits.py`: no substitution into the function
//! when `old` has a limit variable free); the implementation is
//! independent.  SymPy 1.14 does not rename (`Sum(x*k, (k, 0, n)).subs(x,
//! k)` is `Sum(k**2, (k, 0, n))`, and likewise for `Lambda` and
//! `ConditionSet`); the renaming is the standard capture-avoiding
//! substitution (H. P. Barendregt, *The Lambda Calculus*, 1984, §2.1).
//!
//! **Variable slots.**  A `Derivative`, an indefinite `Integral`, a
//! `Series` and a `DSolve` do not bind their variable `x` — `f′(x)` is a
//! function of `x` — but they are not plain operators on their operands
//! either ([`crate::base::walk::var_slot`]).  Replacements other than `x`
//! itself go into the operands, as in SymPy (`Derivative(f(x), x)` with
//! `f(x) ↦ sin(x)` is `Derivative(sin(x), x)`).  Replacing `x`:
//!
//! - by a symbol `t` that the operands do not mention renames the slot:
//!   `Derivative(f(t), t)`, `Integral(t²y, t)`;
//! - by anything else is evaluation at a point.  A `Derivative` whose
//!   operand depends on `x` alone and that `diff` can now take is
//!   differentiated and the point substituted into the result (a formal
//!   `Derivative(y, x)` built for an ODE is kept: `y` stands for `y(x)`);
//!   otherwise the node is wrapped:
//!   `f′(x)` at `x = 0` is `Subs(Derivative(f(x), x), x, 0)`, where up to
//!   0.28 it was the meaningless `Derivative(f(0), 0)`.  The same wrapping
//!   applies when another replacement would bring a free `x` into the
//!   operands (`Derivative(f(x, y), x)` with `y ↦ x` is `∂₁f(x, x)`,
//!   `Subs(Derivative(f(x_1, x), x_1), x_1, x)`, not `Derivative(f(x, x),
//!   x)`).
//!
//! This follows SymPy's `Derivative._eval_subs` (`sympy/core/function.py`,
//! BSD-3) in outcome; SymPy keeps an indefinite integral at a point as
//! `Integral(f, (x, a))` where symplex uses the same `Subs` node, and
//! SymPy renames an integral's variable even to a symbol its integrand
//! already contains (`Integral(y*f(x), x).subs(x, y)` is `Integral(y*f(y),
//! y)`), which symplex does not.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::walk::Binder;

// ═══════════════════════════════════════════════════════════════════════════
// Structural substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Replace every free occurrence of `old` with `new` in the expression
/// rooted at `expr` (bound occurrences are left alone and binders are
/// renamed rather than capture — see the module documentation).
///
/// This is **structural** substitution: only exact `ExprId` matches are
/// replaced.  The result is re-canonicalized through the normal
/// `Arena::add` / `Arena::mul` / etc. constructors, so like-term
/// collection and other canonical-form invariants are maintained.
///
/// Returns `expr` unchanged (same `ExprId`) if `old` does not appear
/// anywhere in the tree — no unnecessary allocation.
pub(crate) fn subs(arena: &mut Arena, expr: ExprId, old: ExprId, new: ExprId) -> ExprId {
    // Fast path: if old == new, nothing to do.
    if old == new {
        return expr;
    }
    // Fast path: if expr IS the thing we're replacing, return new.
    if expr == old {
        return new;
    }
    // Fast path: atoms that aren't the target can't contain it.
    if arena.node(expr).is_atom() {
        return expr;
    }

    let map: FxHashMap<ExprId, ExprId> = std::iter::once((old, new)).collect();
    subs_scoped(arena, expr, &map, true, None)
}

/// An algebraic rewrite of `old` into `new`: tried on every node, after
/// its operands are rebuilt, in a scope where the replacement is in force.
type Rewrite<'a> = (
    ExprId,
    ExprId,
    &'a dyn Fn(&mut Arena, ExprId) -> Option<ExprId>,
);

/// Substitution of `old` by `new` that, besides exact occurrences, lets
/// `rewrite` recognise `old` inside a node (`x²` inside `x⁴`; see
/// `transforms::pattern::subs_algebraic`).  Binders and variable slots are
/// handled exactly as by [`subs`]: under a binder of `v`, an `old` with
/// `v` free is not rewritten.
pub(crate) fn subs_with_rewrite(
    arena: &mut Arena,
    expr: ExprId,
    old: ExprId,
    new: ExprId,
    rewrite: &dyn Fn(&mut Arena, ExprId) -> Option<ExprId>,
) -> ExprId {
    if old == new {
        return expr;
    }
    if expr == old {
        return new;
    }
    let map: FxHashMap<ExprId, ExprId> = std::iter::once((old, new)).collect();
    subs_scoped(arena, expr, &map, true, Some((old, new, rewrite)))
}

/// Simultaneous substitution of multiple `(old, new)` pairs.
///
/// All replacements happen "at once" — earlier substitutions do not
/// affect later ones.  This avoids the order-dependence issues that
/// sequential substitution can cause.
pub(crate) fn subs_map(
    arena: &mut Arena,
    expr: ExprId,
    replacements: &[(ExprId, ExprId)],
) -> ExprId {
    if replacements.is_empty() {
        return expr;
    }

    let map: FxHashMap<ExprId, ExprId> = replacements.iter().copied().collect();

    // Fast path: expr itself is in the map.
    if let Some(&new) = map.get(&expr) {
        return new;
    }

    if arena.node(expr).is_atom() {
        return expr;
    }

    subs_scoped(arena, expr, &map, true, None)
}

/// The substitution maps in force in the scopes met during one
/// [`subs_scoped`] walk, interned so that `(node, scope)` is a cheap key.
/// Scope 0 is the identity (nothing left to substitute).
struct Scopes {
    maps: Vec<FxHashMap<ExprId, ExprId>>,
    index: FxHashMap<Vec<(ExprId, ExprId)>, u32>,
    /// Keys are matched algebraically (see [`subs_with_rewrite`]), so a
    /// key may act on an operand that does not contain it as a node.
    algebraic: bool,
}

impl Scopes {
    /// The scopes, and the scope of the caller's `map`.
    fn new(map: &FxHashMap<ExprId, ExprId>, algebraic: bool) -> (Self, u32) {
        let mut scopes = Scopes {
            maps: vec![FxHashMap::default()],
            index: FxHashMap::default(),
            algebraic,
        };
        scopes.index.insert(Vec::new(), 0);
        let mut pairs: Vec<(ExprId, ExprId)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_unstable();
        let top = scopes.intern(pairs);
        (scopes, top)
    }

    /// Can the key `k` act on the operand `c`?  A structural key only if
    /// `c` contains it; an algebraic one (`x²` acts on `x⁴`) if `c`
    /// contains every free symbol of `k`.
    fn may_act(&self, arena: &Arena, c: ExprId, k: ExprId) -> bool {
        crate::base::walk::contains(arena, c, k)
            || (self.algebraic
                && crate::base::walk::free_symbols(arena, k)
                    .into_iter()
                    .all(|s| crate::base::walk::contains(arena, c, s)))
    }

    fn intern(&mut self, pairs: Vec<(ExprId, ExprId)>) -> u32 {
        if let Some(&sc) = self.index.get(&pairs) {
            return sc;
        }
        let sc = self.maps.len() as u32;
        self.maps.push(pairs.iter().copied().collect());
        self.index.insert(pairs, sc);
        sc
    }

    /// The scope inside binder `b` entered from scope `sc`, and the bound
    /// variable the rebuilt binder uses.
    ///
    /// A replacement survives if its `old` does not have the bound
    /// variable free (shadowing) and occurs in a scoped operand at all.
    /// If a surviving `new` has the bound variable free, the binder is
    /// renamed to a fresh variable, by the same simultaneous substitution.
    fn enter(&mut self, arena: &mut Arena, sc: u32, b: &Binder) -> (u32, ExprId) {
        let ExprNode::Symbol(var_sym) = *arena.node(b.var) else {
            return (0, b.var);
        };
        let mut inner: Vec<(ExprId, ExprId)> = self.maps[sc as usize]
            .iter()
            .map(|(&k, &v)| (k, v))
            .filter(|&(k, _)| {
                !crate::base::walk::has_free_symbol(arena, k, var_sym)
                    && b.scoped.iter().any(|&c| self.may_act(arena, c, k))
            })
            .collect();
        if inner.is_empty() {
            return (0, b.var);
        }
        let captured = inner
            .iter()
            .any(|&(_, v)| crate::base::walk::has_free_symbol(arena, v, var_sym));
        let var = if captured {
            let fresh = fresh_bound_variable(arena, var_sym, &b.scoped, &inner);
            inner.push((b.var, fresh));
            fresh
        } else {
            b.var
        };
        inner.sort_unstable();
        (self.intern(inner), var)
    }

    /// The plan for the variable-slot node `b` (see
    /// [`crate::base::walk::var_slot`]) entered from scope `sc`: the
    /// scope of its scoped operands, the variable it is rebuilt with, and
    /// the point to evaluate it at (`None`: rebuild only).  See the module
    /// documentation on variable slots.
    fn enter_slot(&mut self, arena: &mut Arena, sc: u32, b: &Binder) -> SlotPlan {
        let ExprNode::Symbol(var_sym) = *arena.node(b.var) else {
            return SlotPlan::rebuild(0, b.var);
        };
        let map = &self.maps[sc as usize];
        let point = map.get(&b.var).copied().unwrap_or(b.var);
        let mut inner: Vec<(ExprId, ExprId)> = map
            .iter()
            .map(|(&k, &v)| (k, v))
            .filter(|&(k, _)| k != b.var && b.scoped.iter().any(|&c| self.may_act(arena, c, k)))
            .collect();
        let brings = |arena: &Arena, inner: &[(ExprId, ExprId)], sym: SymbolId| {
            inner.iter().any(|&(k, v)| {
                crate::base::walk::has_free_symbol(arena, v, sym)
                    && !crate::base::walk::has_free_symbol(arena, k, sym)
            })
        };
        let captured = brings(arena, &inner, var_sym);
        if point == b.var && !captured {
            inner.sort_unstable();
            return SlotPlan::rebuild(self.intern(inner), b.var);
        }
        // A clean rename to a symbol the operands and the replacements do
        // not mention.
        if let ExprNode::Symbol(to_sym) = *arena.node(point)
            && !b
                .scoped
                .iter()
                .any(|&c| crate::base::walk::has_free_symbol(arena, c, to_sym))
            && !inner.iter().any(|&(_, v)| {
                crate::base::walk::has_free_symbol(arena, v, to_sym)
                    || crate::base::walk::has_free_symbol(arena, v, var_sym)
            })
        {
            inner.push((b.var, point));
            inner.sort_unstable();
            return SlotPlan::rebuild(self.intern(inner), point);
        }
        // Evaluation at `point`: `Subs(node, var, point)` binds `var` in the
        // whole node, so the variable is renamed if a replacement or an
        // outer operand (a series' expansion point) would be captured.
        let outer_mentions_var = b.outer.iter().any(|&o| {
            crate::base::walk::has_free_symbol(arena, o, var_sym)
                || map.iter().any(|(&k, &v)| {
                    crate::base::walk::has_free_symbol(arena, v, var_sym)
                        && crate::base::walk::contains(arena, o, k)
                })
        });
        let var = if captured || outer_mentions_var {
            let fresh = fresh_bound_variable(arena, var_sym, &b.scoped, &inner);
            inner.push((b.var, fresh));
            fresh
        } else {
            b.var
        };
        inner.sort_unstable();
        SlotPlan {
            inner: self.intern(inner),
            var,
            point: Some(point),
        }
    }
}

/// How a variable-slot node is rebuilt by [`subs_scoped`].
#[derive(Clone, Copy)]
struct SlotPlan {
    /// The scope of the scoped operands.
    inner: u32,
    /// The variable the node is rebuilt with.
    var: ExprId,
    /// `Some(p)`: the rebuilt node is evaluated at `var = p`.
    point: Option<ExprId>,
}

impl SlotPlan {
    fn rebuild(inner: u32, var: ExprId) -> Self {
        SlotPlan {
            inner,
            var,
            point: None,
        }
    }
}

/// A node met by [`subs_scoped`] whose operands are not all in the same
/// scope.
enum Entered {
    /// A binder: its scoped operands' scope and its (possibly renamed)
    /// variable.
    Binder(Binder, u32, ExprId),
    /// A variable-slot node and its plan.
    Slot(Binder, SlotPlan),
}

/// A symbol named `base`, `base_1`, `base_2`, … — the first that carries
/// no assumptions and occurs in none of `avoid` (a name that has never been
/// interned is created).  Used for the dummy variable of a `Subs` built by
/// `diff`: `∂f/∂u` at `u = x²` is `Subs(Derivative(f(_xi), _xi), _xi, x²)`.
/// The name is a function of the expression only, so equal inputs give
/// equal nodes.
pub(crate) fn fresh_symbol(arena: &mut Arena, base: &str, avoid: &[ExprId]) -> ExprId {
    let mut n = 0usize;
    loop {
        let name = if n == 0 {
            base.to_owned()
        } else {
            format!("{base}_{n}")
        };
        match arena.symbols.get(&name) {
            None => return arena.symbol(&name),
            Some(sid)
                if arena.symbol_assumptions(sid)
                    == crate::base::assumptions::Assumptions::default() =>
            {
                let candidate = arena.intern(ExprNode::Symbol(sid));
                if !avoid
                    .iter()
                    .any(|&e| crate::base::walk::contains(arena, e, candidate))
                {
                    return candidate;
                }
            }
            Some(_) => {}
        }
        n += 1;
    }
}

/// A variable to rename a binder of `var` to: `var_1`, `var_2`, …, the
/// first that occurs neither in the binder's scoped operands nor in the
/// replacements (so it captures nothing) and carries the same assumptions
/// as `var` (a summation index declared `integer` stays one).  A name that
/// has never been interned is created with `var`'s assumptions.
fn fresh_bound_variable(
    arena: &mut Arena,
    var: SymbolId,
    scoped: &[ExprId],
    replacements: &[(ExprId, ExprId)],
) -> ExprId {
    let base = arena.symbol_name(var).to_owned();
    let assumptions = arena.symbol_assumptions(var);
    let mut n = 1usize;
    loop {
        let name = format!("{base}_{n}");
        match arena.symbols.get(&name) {
            None => {
                let fresh = arena.symbol(&name);
                if let ExprNode::Symbol(sid) = *arena.node(fresh)
                    && assumptions != crate::base::assumptions::Assumptions::default()
                {
                    arena.set_symbol_assumptions(sid, assumptions);
                }
                return fresh;
            }
            Some(sid) if arena.symbol_assumptions(sid) == assumptions => {
                let candidate = arena.intern(ExprNode::Symbol(sid));
                let in_use = scoped
                    .iter()
                    .chain(replacements.iter().flat_map(|(k, v)| [k, v]))
                    .any(|&e| crate::base::walk::contains(arena, e, candidate));
                if !in_use {
                    return candidate;
                }
            }
            Some(_) => {}
        }
        n += 1;
    }
}

/// Bottom-up simultaneous substitution of `map` that replaces only free
/// occurrences (see the module documentation on binders and variable
/// slots).
///
/// The walk is over *(node, scope)* pairs, with an explicit stack: the
/// arena is a hash-consed DAG, and the same node can occur both inside a
/// binder and outside it (`x + Σ_{x=0}^{3} x`), where different
/// replacements apply.  A node reached in the identity scope is its own
/// result and is not visited.  Never recurses; `differentiate` lets a
/// `Derivative` evaluated at a point be differentiated first, and is off
/// in the one nested call that makes (so nesting is one level deep).
/// `rewrite`, if given, is applied to every node (atoms included) in each
/// scope where its replacement is in force.
fn subs_scoped(
    arena: &mut Arena,
    expr: ExprId,
    map: &FxHashMap<ExprId, ExprId>,
    differentiate: bool,
    rewrite: Option<Rewrite<'_>>,
) -> ExprId {
    let (mut scopes, top) = Scopes::new(map, rewrite.is_some());
    // `r` after the algebraic rewrite, when one is in force in scope `sc`.
    let rewritten = |arena: &mut Arena, scopes: &Scopes, sc: u32, r: ExprId| -> ExprId {
        match rewrite {
            Some((old, new, f)) if scopes.maps[sc as usize].get(&old) == Some(&new) => {
                if r == old {
                    new
                } else {
                    f(arena, r).unwrap_or(r)
                }
            }
            _ => r,
        }
    };
    if top == 0 {
        return expr;
    }
    let mut cache: FxHashMap<(ExprId, u32), ExprId> = FxHashMap::default();
    // For each binder or variable slot visited: the scope of its scoped
    // operands, the variable it is rebuilt with, and (a slot) the point.
    let mut entered: FxHashMap<(ExprId, u32), Entered> = FxHashMap::default();
    let mut stack: Vec<(ExprId, u32, bool)> = vec![(expr, top, false)];
    let mut children: SmallVec<[(ExprId, u32); 6]> = SmallVec::new();

    while let Some(&(id, sc, expanded)) = stack.last() {
        if cache.contains_key(&(id, sc)) {
            stack.pop();
            continue;
        }
        if !expanded {
            let leaf = if let Some(&new) = scopes.maps[sc as usize].get(&id) {
                Some(new)
            } else if arena.node(id).is_atom() {
                Some(rewritten(arena, &scopes, sc, id))
            } else {
                None
            };
            if let Some(result) = leaf {
                cache.insert((id, sc), result);
                stack.pop();
                continue;
            }
            if let Some(top) = stack.last_mut() {
                top.2 = true;
            }
            children.clear();
            if let Some(b) = crate::base::walk::binder(arena, id) {
                let (inner, var) = scopes.enter(arena, sc, &b);
                children.extend(b.outer.iter().map(|&c| (c, sc)));
                if inner != 0 {
                    children.extend(b.scoped.iter().map(|&c| (c, inner)));
                }
                entered.insert((id, sc), Entered::Binder(b, inner, var));
            } else if let Some(b) = crate::base::walk::var_slot(arena, id) {
                let plan = scopes.enter_slot(arena, sc, &b);
                children.extend(b.outer.iter().map(|&c| (c, sc)));
                if plan.inner != 0 {
                    children.extend(b.scoped.iter().map(|&c| (c, plan.inner)));
                }
                entered.insert((id, sc), Entered::Slot(b, plan));
            } else {
                arena.node(id).for_each_child(|c| children.push((c, sc)));
            }
            for &(c, csc) in children.iter().rev() {
                if !cache.contains_key(&(c, csc)) {
                    stack.push((c, csc, false));
                }
            }
        } else {
            stack.pop();
            let get = |c: ExprId, s: u32| cache.get(&(c, s)).copied().unwrap_or(c);
            let operands = |b: &Binder, inner: u32| {
                let scoped: SmallVec<[ExprId; 2]> =
                    b.scoped.iter().map(|&c| get(c, inner)).collect();
                let outer: SmallVec<[ExprId; 2]> = b.outer.iter().map(|&c| get(c, sc)).collect();
                (scoped, outer)
            };
            let rebuilt = match entered.get(&(id, sc)) {
                Some(Entered::Binder(b, inner, var)) => {
                    let (scoped, outer) = operands(b, *inner);
                    if *var == b.var && scoped == b.scoped && outer == b.outer {
                        id
                    } else {
                        crate::base::walk::rebuild_binder(arena, id, *var, &scoped, &outer)
                    }
                }
                Some(Entered::Slot(b, plan)) => {
                    let (scoped, outer) = operands(b, plan.inner);
                    let node = if plan.var == b.var && scoped == b.scoped && outer == b.outer {
                        id
                    } else {
                        crate::base::walk::rebuild_var_slot(arena, id, plan.var, &scoped, &outer)
                    };
                    match plan.point {
                        None => node,
                        Some(point) => at_point(arena, node, plan.var, point, differentiate),
                    }
                }
                None => crate::base::walk::rebuild_with(arena, id, &|c| get(c, sc)),
            };
            let rebuilt = rewritten(arena, &scopes, sc, rebuilt);
            cache.insert((id, sc), rebuilt);
        }
    }

    cache.get(&(expr, top)).copied().unwrap_or(expr)
}

/// The variable-slot node `node` (a function of the symbol `var`)
/// evaluated at `var = point`.  A `Derivative` that `diff` makes progress
/// on (with `differentiate` on) is differentiated and `point` substituted
/// into the result; anything else is `Subs(node, var, point)`.
///
/// Only an operand whose one free symbol is `var` is differentiated: a
/// formal `Derivative(y, x)` built for an ODE declares that `y` depends on
/// `x`, which `diff` (for which `y` is a constant) would turn into `0`.
fn at_point(
    arena: &mut Arena,
    node: ExprId,
    var: ExprId,
    point: ExprId,
    differentiate: bool,
) -> ExprId {
    if differentiate
        && let ExprNode::Derivative(body, dvar) = *arena.node(node)
        && crate::base::walk::free_symbols(arena, body)
            .iter()
            .all(|&s| s == dvar)
    {
        let d = crate::transforms::diff::diff(arena, body, dvar);
        // Progress (the product rule on `x²·f(x)`, say) is enough: what
        // stays formal in `d` is wrapped by the nested call, which does not
        // differentiate again.
        if d != node {
            let map: FxHashMap<ExprId, ExprId> = std::iter::once((var, point)).collect();
            return subs_scoped(arena, d, &map, false, None);
        }
    }
    crate::base::walk::subs_node(arena, node, var, point)
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluate formal derivatives
// ═══════════════════════════════════════════════════════════════════════════

/// Walk the expression tree bottom-up and concretely evaluate any
/// `Derivative(inner, var)` nodes by calling [`crate::transforms::diff::diff`].
///
/// This is the *doit* pattern: formal derivative placeholders become
/// concrete differentiation results.  Because the traversal is
/// bottom-up, nested derivatives (e.g. `d/dx(d/dx(x²))`) are
/// resolved from the inside out.
///
/// Nodes that are not `Derivative` are rebuilt with their (possibly
/// updated) children, preserving canonical form.
pub(crate) fn eval_derivatives(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        match node {
            ExprNode::Derivative(inner, var) => {
                // Look up the rebuilt inner / var from the cache so
                // that nested Derivatives are already resolved.
                let new_inner = cache.get(&inner).copied().unwrap_or(inner);
                let new_var = cache.get(&var).copied().unwrap_or(var);
                // Concretely differentiate.
                let result = crate::transforms::diff::diff(arena, new_inner, new_var);
                cache.insert(id, result);
            }
            // A derivative at a point: substitute the point into the
            // evaluated derivative (it re-wraps what is still formal).
            ExprNode::Subs(body, var, point) => {
                let new_body = cache.get(&body).copied().unwrap_or(body);
                let new_point = cache.get(&point).copied().unwrap_or(point);
                let result = subs(arena, new_body, var, new_point);
                cache.insert(id, result);
            }
            _ => {
                // Rebuild non-Derivative nodes with substituted children.
                let new_id = crate::base::walk::rebuild_with_cache(arena, id, &cache);
                cache.insert(id, new_id);
            }
        }
    }

    cache.get(&expr).copied().unwrap_or(expr)
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

    // ── Basic subs ──────────────────────────────────────────────────

    #[test]
    fn subs_symbol_for_number() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn subs_symbol_in_product() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let expr = a.mul(&[two, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    #[test]
    fn subs_in_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let expr = a.pow(x, two);
        let three = a.int(3);
        let result = subs(&mut a, expr, x, three);
        assert_eq!(display(&a, result), "9");
    }

    #[test]
    fn subs_in_sin() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.sin(x);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y)");
    }

    #[test]
    fn subs_no_match_returns_same_id() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let expr = a.add(&[x, a.one]);
        let n99 = a.int(99);
        let result = subs(&mut a, expr, y, n99);
        assert_eq!(result, expr, "no match should return same ExprId");
    }

    #[test]
    fn subs_entire_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let result = subs(&mut a, x, x, y);
        assert_eq!(result, y);
    }

    #[test]
    fn subs_old_equals_new_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs(&mut a, expr, x, x);
        assert_eq!(result, expr);
    }

    // ── Nested substitution ─────────────────────────────────────────

    #[test]
    fn subs_in_nested_add_mul() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let three = a.int(3);

        // x^2 + 2*x + 1, substitute x → 3
        let x_sq = a.pow(x, two);
        let two_x = a.mul(&[two, x]);
        let expr = a.add(&[x_sq, two_x, a.one]);
        let result = subs(&mut a, expr, x, three);
        // 9 + 6 + 1 = 16
        assert_eq!(display(&a, result), "16");
    }

    #[test]
    fn subs_in_function_of_pow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);

        // sin(x^2), substitute x → y
        let xp = a.pow(x, two);
        let expr = a.sin(xp);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "sin(y^2)");
    }

    #[test]
    fn subs_replaces_all_occurrences() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // x + x → y + y = 2*y
        let expr = a.add(&[x, x]);
        let result = subs(&mut a, expr, x, y);
        assert_eq!(display(&a, result), "2*y");
    }

    // ── Structural correctness ──────────────────────────────────────

    #[test]
    fn subs_does_not_match_algebraic_subexpressions() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let one = a.one;

        // (1/x).subs(x^2, 1) should return 1/x unchanged.
        // x^(-1) does not structurally contain x^2.
        let x_inv = a.pow(x, a.neg_one);
        let x_sq = a.pow(x, two);
        let result = subs(&mut a, x_inv, x_sq, one);
        assert_eq!(
            result, x_inv,
            "structural subs should not match x^2 in x^(-1)"
        );
    }

    #[test]
    fn subs_with_zero_evaluates() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let zero = a.zero;

        // (x*y).subs(y, 0) → x*0 → 0
        let expr = a.mul(&[x, y]);
        let result = subs(&mut a, expr, y, zero);
        assert_eq!(result, a.zero);
    }

    // ── Simultaneous substitution ───────────────────────────────────

    #[test]
    fn subs_map_simultaneous() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // (x + y).subs({x→y, y→x}) should give y + x = x + y (commutative)
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, y), (y, x)]);
        // Should be the same canonical form since x+y and y+x are equal.
        assert_eq!(result, expr, "swapping x↔y in x+y should give x+y");
    }

    #[test]
    fn subs_map_empty_is_noop() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.add(&[x, a.one]);
        let result = subs_map(&mut a, expr, &[]);
        assert_eq!(result, expr);
    }

    #[test]
    fn subs_map_multiple() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let two = a.int(2);
        let three = a.int(3);

        // (x + y).subs({x→2, y→3}) = 5
        let expr = a.add(&[x, y]);
        let result = subs_map(&mut a, expr, &[(x, two), (y, three)]);
        assert_eq!(display(&a, result), "5");
    }

    // ── Deep expressions (stack safety) ─────────────────────────────

    #[test]
    fn subs_deep_expression_no_stack_overflow() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Build a deeply nested expression: sin(sin(sin(...sin(x)...)))
        // 10,000 levels deep — tests that our iterative walker doesn't
        // stack-overflow.
        let mut expr = x;
        for _ in 0..10_000 {
            expr = a.sin(expr);
        }

        // Substitute x → y deep inside.
        let result = subs(&mut a, expr, x, y);

        // Verify it changed (not the same ExprId).
        assert_ne!(
            result, expr,
            "substitution should have changed the deep expression"
        );
        // NOTE: We don't test Display here because fmt_expr in
        // display.rs is still recursive and would overflow at this
        // depth.  That will be fixed when display is converted to an
        // iterative walker (tracked as a known issue).
    }

    #[test]
    fn subs_moderate_depth_display_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");

        // Moderate depth that the recursive display can still handle.
        let mut expr = x;
        for _ in 0..50 {
            expr = a.sin(expr);
        }

        let result = subs(&mut a, expr, x, y);
        let s = format!("{}", a.display(result));
        assert!(s.starts_with("sin("), "should still start with sin(");
        assert!(s.contains('y'), "should contain y after substitution");
        assert!(!s.contains('x'), "should not contain x after substitution");
    }

    // ── Binders ───────────────────────────────────────────────────────────

    #[test]
    fn subs_leaves_every_binder_kind_alone() {
        // For each binder of `x`: substituting for `x` rewrites only the
        // operands in the enclosing scope.
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let three = a.int(3);
        let body = a.mul(&[x, y]);
        let five = a.int(5);
        let x5 = a.pow(x, five);
        let neg_x = a.neg(x);
        let poly = a.add(&[x5, neg_x, a.one]);
        let binders = [
            a.intern(ExprNode::Sum(body, x, a.zero, three)),
            a.intern(ExprNode::Product_(body, x, a.one, three)),
            a.definite_integral(body, x, a.zero, three),
            a.intern(ExprNode::RootSum(poly, body, x)),
            a.intern(ExprNode::ConditionSet(x, body)),
            a.intern(ExprNode::RootOf(poly, x, a.zero)),
            a.intern(ExprNode::Limit(body, x, a.zero)),
            a.intern(ExprNode::Residue(body, x, a.zero)),
            a.intern(ExprNode::LaplaceTransform(body, x, y)),
            a.intern(ExprNode::InverseLaplaceTransform(body, x, y)),
        ];
        let half = a.rational(1, 2);
        for e in binders {
            assert_eq!(subs(&mut a, e, x, half), e, "{}", display(&a, e));
        }
        // The free `y` is substituted, the binder kept.
        let s = a.intern(ExprNode::Sum(body, x, a.zero, three));
        let got = subs(&mut a, s, y, half);
        assert_eq!(display(&a, got), "Sum(1/2*x, x=0..3)");
    }

    #[test]
    fn subs_renames_a_binder_that_would_capture() {
        // Σ_{k=0}^{n} x·k with x ↦ k + 1: Σ_{k_1=0}^{n} (k + 1)·k_1.
        let mut a = Arena::new();
        let (x, k, n) = (sym(&mut a, "x"), sym(&mut a, "k"), sym(&mut a, "n"));
        let body = a.mul(&[x, k]);
        let s = a.intern(ExprNode::Sum(body, k, a.zero, n));
        let k1 = a.add(&[k, a.one]);
        let got = subs(&mut a, s, x, k1);
        assert_eq!(display(&a, got), "Sum(k_1*(k + 1), k_1=0..n)");
        // `k_1` taken by the expression: the next free name.
        let k_1 = sym(&mut a, "k_1");
        let body = a.mul(&[x, k, k_1]);
        let s = a.intern(ExprNode::Sum(body, k, a.zero, n));
        let got = subs(&mut a, s, x, k);
        assert_eq!(display(&a, got), "Sum(k*k_1*k_2, k_2=0..n)");
    }

    #[test]
    fn subs_nested_binders_of_the_same_variable() {
        // Σ_{x=0}^{y} (x + Σ_{x=0}^{x} x·y): y ↦ x renames the outer
        // binder, and the inner binder shadows it.
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, "x"), sym(&mut a, "y"));
        let xy = a.mul(&[x, y]);
        let inner = a.intern(ExprNode::Sum(xy, x, a.zero, x));
        let body = a.add(&[x, inner]);
        let outer = a.intern(ExprNode::Sum(body, x, a.zero, y));
        let got = subs(&mut a, outer, y, x);
        let free: Vec<String> = crate::base::walk::free_symbols(&a, got)
            .into_iter()
            .map(|s| display(&a, s))
            .collect();
        assert_eq!(free, vec!["x".to_string()], "{}", display(&a, got));
        assert_eq!(
            display(&a, got),
            "Sum(x_1 + Sum(x*x_1, x_1=0..x_1), x_1=0..x)"
        );
    }

    // ── eval_derivatives ────────────────────────────────────────────

    #[test]
    fn eval_derivatives_sin_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // Derivative(sin(x), x)  →  cos(x)
        let sin_x = a.sin(x);
        let formal = a.intern(crate::base::node::ExprNode::Derivative(sin_x, x));
        let result = super::eval_derivatives(&mut a, formal);
        assert_eq!(display(&a, result), "cos(x)");
    }

    #[test]
    fn eval_derivatives_nested() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // Derivative(Derivative(x^2, x), x)  →  d/dx(2x) = 2
        let x2 = a.pow(x, two);
        let d1 = a.intern(crate::base::node::ExprNode::Derivative(x2, x));
        let d2 = a.intern(crate::base::node::ExprNode::Derivative(d1, x));
        let result = super::eval_derivatives(&mut a, d2);
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn eval_derivatives_no_derivative_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // x^2 + 1 — contains no Derivative nodes, should be unchanged
        let x2 = a.pow(x, two);
        let expr = a.add(&[x2, a.one]);
        let result = super::eval_derivatives(&mut a, expr);
        assert_eq!(result, expr);
    }

    #[test]
    fn eval_derivatives_inside_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // x + Derivative(x^2, x)  →  x + 2*x = 3*x
        let x2 = a.pow(x, two);
        let d = a.intern(crate::base::node::ExprNode::Derivative(x2, x));
        let expr = a.add(&[x, d]);
        let result = super::eval_derivatives(&mut a, expr);
        assert_eq!(display(&a, result), "3*x");
    }
}
