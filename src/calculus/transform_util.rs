//! What the integral transforms share: the test "does this expression
//! depend on the transform variable?", the decompositions built on it
//! (constant factors, `a·v + b`), the walk over the part of an expression
//! in which the variable *is* the transform variable, and the error and
//! sign helpers.  The Laplace, Z, Fourier and Mellin transforms
//! ([`laplace`](super::laplace), [`z_transform`](super::z_transform),
//! [`fourier_transform`](super::fourier_transform),
//! [`mellin`](super::mellin)) and their inverses call these and keep no
//! copies.
//!
//! # The bound-variable rule
//!
//! A transform in `t` integrates (or sums) over `t`; an occurrence of `t`
//! that a binder inside the expression has bound is a different variable
//! that happens to share the name.  `Σ_{t=1}^{3} tᵏ`, `∫₀¹ f(t) dt`,
//! `lim_{t→0} g(t)` and `RootOf(t⁵ − t + 1, 0)` are constants of the
//! transform.  The binders are those of
//! [`walk::binder`], which also says in which
//! operands the variable stays free (the limits of a sum, the point of a
//! limit).  Two things follow, and both are decided here and nowhere else:
//!
//! 1. **Dependence is free occurrence.**  [`depends_on`] is the one test;
//!    [`split_factors`], [`split_independent`], [`linear_in`] and
//!    [`extract_linear_coeff`] are built on it.  A transform must not ask
//!    [`walk::contains`] about its variable:
//!    that is the structural question (is this node somewhere in the
//!    tree?), which also finds bound occurrences.
//! 2. **A rewrite of the function being transformed stops at a binder of
//!    the variable.**  A pass that replaces sub-expressions by forms that
//!    are equal only *as functions of the transform variable* — `Piecewise`
//!    by Heaviside steps (equal almost everywhere, not at the breakpoints),
//!    `H(t + 1)` by `1` because the Laplace integral sees `t > 0` only —
//!    walks [`scope_post_order`] and rebuilds with [`rebuild_in_scope`],
//!    which do not enter the operands a binder of the variable scopes over;
//!    a search through the sub-expressions pushes
//!    [`for_each_child_in_scope`].  Inside `Σ_{t=−3}^{3} H(t + 1)` the
//!    bound `t` runs over negative integers and through the breakpoint.
//!
//! Up to 0.59 the Laplace and Z transforms asked for free occurrences and
//! the Fourier and Mellin transforms for structural ones, and every pass
//! walked into binders: `fourier_transform(Integral(H(x + 1), x, -3, 3), x,
//! w)` was `−6i·e^{iw}/w` (the constant 4 has the transform `8π·δ(w)`), and
//! the Laplace transform of `Sum(Piecewise(1 if t < 2, 0 if True), t, 1,
//! 3)·e^{−2t}` was `3/(2(s + 2))` (the sum is 1: `1/(s + 2)`).
//!
//! A tree without binders has no bound occurrences, so there both readings
//! and both walks agree; the node facts answer "is there a binder?" in
//! O(1) ([`NodeFacts`](crate::base::arena::NodeFacts)) and the plain
//! [`walk::post_order_ids`] is used.
//!
//! # Corner cases
//!
//! - **The variable is not a symbol.**  The entry points refuse it
//!   ([`require_symbol`], [`symbol_id`]); [`depends_on`] then looks for the
//!   expression structurally, as [`walk::has_free_var`] does.
//! - **A binder of another variable** (`Σ_{k=1}^{n} e^{−kt}` in a
//!   transform over `t`) is an ordinary node: `t` is free in its summand,
//!   and the passes enter it.
//! - **`Derivative(f, t)`, the indefinite `Integral(f, t)`, `Series`** are
//!   functions of `t`, not binders
//!   ([`walk::var_slot`]): they depend on `t`.
//! - **Shared nodes.**  The arena is a DAG, so the same node can lie under
//!   a binder of the variable and outside it (`H(t + 1) + Σ_{t} H(t + 1)`).
//!   The walk lists it (it is reachable outside), a rewrite maps it, and
//!   [`rebuild_in_scope`] keeps the binder's own copy.
//! - **The destination variable** is treated like any parameter.  A result
//!   such as `Σ_{s=1}^{m} f(s) / (s + 1)` is correct: the sum's `s` is
//!   bound.
//! - **Unevaluated constants.**  The Fourier and Mellin transforms refuse a
//!   result that holds a formal `Sum`/`Integral`/`Limit` node wherever it
//!   came from, so of the constants with a bound variable they answer the
//!   ones that evaluate and `RootOf`/`RootSum`; Laplace and Z keep formal
//!   constants.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::walk;

// ═══════════════════════════════════════════════════════════════════════════
// Dependence
// ═══════════════════════════════════════════════════════════════════════════

/// Does `expr` depend on the transform variable `var`?  Only *free*
/// occurrences count: a `Sum`, `Product`, definite `Integral`, `Limit`,
/// `Subs`, `Residue`, `RootOf`, `RootSum` or formal Laplace transform over
/// `var` is a constant unless `var` is also free in it (in a limit of
/// summation, say).  This is the only dependence test of the integral
/// transforms; see the module documentation.
pub(crate) fn depends_on(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    walk::has_free_var(arena, expr, var)
}

// ═══════════════════════════════════════════════════════════════════════════
// The scope of the transform variable
// ═══════════════════════════════════════════════════════════════════════════

/// Calls `f` on the operands of `id` in which `var` is still the transform
/// variable, in node order: every operand, except that a binder of `var`
/// itself gives only its outer operands (`lo`, `hi` of `Σ_{var=lo}^{hi}
/// body` — neither `body` nor the variable operand).
pub(crate) fn for_each_child_in_scope(
    arena: &Arena,
    id: ExprId,
    var: ExprId,
    mut f: impl FnMut(ExprId),
) {
    if arena.facts(id).has_binder()
        && let Some(b) = walk::binder(arena, id)
        && b.var == var
    {
        for &c in &b.outer {
            f(c);
        }
        return;
    }
    arena.node(id).for_each_child(f);
}

/// The nodes of the tree rooted at `root` that lie in the scope of the
/// transform variable `var`, children before parents, each once:
/// [`walk::post_order_ids`] without the operands that a binder of `var`
/// scopes over.  A node that is reachable both under such a binder and
/// outside it is listed.  For a tree without binders (and for one whose
/// binders bind other variables) this is `post_order_ids` itself, in the
/// same order.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn scope_post_order(arena: &Arena, root: ExprId, var: ExprId) -> Vec<ExprId> {
    if !arena.facts(root).has_binder() {
        return walk::post_order_ids(arena, root);
    }
    let mut result = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    // Stack entries: (id, children pushed).
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
    while let Some((id, children_pushed)) = stack.last_mut() {
        let id = *id;
        if visited.contains(&id) {
            stack.pop();
            continue;
        }
        if *children_pushed {
            stack.pop();
            visited.insert(id);
            result.push(id);
        } else {
            *children_pushed = true;
            let mut children: SmallVec<[ExprId; 6]> = SmallVec::new();
            for_each_child_in_scope(arena, id, var, |c| children.push(c));
            // Reversed, so that they are processed left to right.
            for &child in children.iter().rev() {
                if !visited.contains(&child) {
                    stack.push((child, false));
                }
            }
        }
    }
    result
}

/// The node `id` rebuilt from the rewritten forms in `cache` of its
/// operands in the scope of `var` — the step of a rewriting pass over
/// [`scope_post_order`].  [`walk::rebuild_with_cache`], except that a
/// binder of `var` keeps the operands it scopes over (and its variable) as
/// they are, even when `cache` holds a rewritten form of one of them from
/// another place in the DAG.
pub(crate) fn rebuild_in_scope(
    arena: &mut Arena,
    id: ExprId,
    var: ExprId,
    cache: &FxHashMap<ExprId, ExprId>,
) -> ExprId {
    if arena.facts(id).has_binder()
        && let Some(b) = walk::binder(arena, id)
        && b.var == var
    {
        let outer: SmallVec<[ExprId; 2]> = b
            .outer
            .iter()
            .map(|c| cache.get(c).copied().unwrap_or(*c))
            .collect();
        if outer == b.outer {
            return id;
        }
        return walk::rebuild_binder(arena, id, b.var, &b.scoped, &outer);
    }
    walk::rebuild_with_cache(arena, id, cache)
}

/// Does a node satisfying `pred` lie in the scope of `var` in the tree
/// rooted at `root` ([`scope_post_order`])?
pub(crate) fn any_in_scope(
    arena: &Arena,
    root: ExprId,
    var: ExprId,
    pred: impl Fn(&ExprNode) -> bool,
) -> bool {
    scope_post_order(arena, root, var)
        .into_iter()
        .any(|id| pred(arena.node(id)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant factors
// ═══════════════════════════════════════════════════════════════════════════

/// The factors of a product sorted by dependence on the variable, each
/// list in the order of the product.  See [`split_factors`].
#[derive(Clone, Debug)]
pub(crate) struct Factors {
    /// The constants: the factors free of the variable.
    pub(crate) consts: Vec<ExprId>,
    /// The factors that depend on the variable.
    pub(crate) deps: Vec<ExprId>,
}

/// The factors of `expr` (a `Mul`, or a single factor) split into those
/// free of `var` and those that depend on it ([`depends_on`]).
pub(crate) fn split_factors(arena: &Arena, expr: ExprId, var: ExprId) -> Factors {
    let children: Vec<ExprId> = match arena.node(expr) {
        ExprNode::Mul(ch) => ch.iter().copied().collect(),
        _ => vec![expr],
    };
    let mut consts = Vec::new();
    let mut deps = Vec::new();
    for c in children {
        if depends_on(arena, c, var) {
            deps.push(c);
        } else {
            consts.push(c);
        }
    }
    Factors { consts, deps }
}

/// `expr = coeff · body`.  See [`split_independent`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Split {
    /// The constant coefficient: the product of the factors free of the
    /// variable (`1` if none was split off).
    pub(crate) coeff: ExprId,
    /// The rest, which depends on the variable.
    pub(crate) body: ExprId,
}

/// `expr` as `coeff · body`, the coefficient being the product of the
/// factors of a `Mul` that are free of `var` ([`depends_on`]).  When `expr`
/// is not a product, or all of its factors are on one side, the coefficient
/// is `1` and the body is `expr` itself (also for a constant `expr`).
pub(crate) fn split_independent(arena: &mut Arena, expr: ExprId, var: ExprId) -> Split {
    let factors = split_factors(arena, expr, var);
    if factors.consts.is_empty() || factors.deps.is_empty() {
        return Split {
            coeff: arena.one(),
            body: expr,
        };
    }
    let product = |arena: &mut Arena, fs: &[ExprId]| match fs {
        [single] => *single,
        _ => arena.mul(fs),
    };
    Split {
        coeff: product(arena, &factors.consts),
        body: product(arena, &factors.deps),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear arguments
// ═══════════════════════════════════════════════════════════════════════════

/// `a·var + b` with `a`, `b` free of the variable.  See [`linear_in`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Linear {
    /// The coefficient `a` of the variable in `a·var + b`.
    pub(crate) a: ExprId,
    /// The term `b` free of the variable.
    pub(crate) b: ExprId,
}

/// `expr = a·var + b` with `a`, `b` free of `var` ([`depends_on`]), read
/// structurally: `var` itself, a product with `var` as one factor and
/// constant cofactors, a constant, and sums and negations of these.  `None`
/// when `expr` is not written that way (`var²`, `var·var`, `(var + 1)·c`:
/// nothing is expanded).  A constant gives `a = 0`, also one in which `var`
/// occurs bound.
///
/// Uses an explicit stack — never recurses.
pub(crate) fn linear_in(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<Linear> {
    enum Step {
        Enter(ExprId),
        Negate,
        Sum(usize),
    }
    let mut steps = vec![Step::Enter(expr)];
    let mut done: Vec<Linear> = Vec::new();
    while let Some(step) = steps.pop() {
        match step {
            Step::Enter(e) if e == var => done.push(Linear {
                a: arena.one(),
                b: arena.zero(),
            }),
            Step::Enter(e) if !depends_on(arena, e, var) => done.push(Linear {
                a: arena.zero(),
                b: e,
            }),
            Step::Enter(e) => match arena.node(e).clone() {
                ExprNode::Neg(inner) => {
                    steps.push(Step::Negate);
                    steps.push(Step::Enter(inner));
                }
                ExprNode::Mul(children) => {
                    let mut coeff: SmallVec<[ExprId; 4]> = SmallVec::new();
                    let mut seen = false;
                    for &c in &children {
                        if c == var {
                            if seen {
                                return None;
                            }
                            seen = true;
                        } else if depends_on(arena, c, var) {
                            return None;
                        } else {
                            coeff.push(c);
                        }
                    }
                    if !seen {
                        return None;
                    }
                    let a = if coeff.is_empty() {
                        arena.one()
                    } else {
                        arena.mul(&coeff)
                    };
                    done.push(Linear { a, b: arena.zero() });
                }
                ExprNode::Add(children) => {
                    steps.push(Step::Sum(children.len()));
                    // Reversed, so that the terms are read left to right.
                    for &c in children.iter().rev() {
                        steps.push(Step::Enter(c));
                    }
                }
                _ => return None,
            },
            Step::Negate => {
                let inner = done.pop()?;
                done.push(Linear {
                    a: arena.neg(inner.a),
                    b: arena.neg(inner.b),
                });
            }
            Step::Sum(n) => {
                let terms = done.split_off(done.len().checked_sub(n)?);
                let mut a_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
                let mut b_terms: SmallVec<[ExprId; 4]> = SmallVec::new();
                for term in terms {
                    if !arena.is_zero_structural(term.a) {
                        a_terms.push(term.a);
                    }
                    if !arena.is_zero_structural(term.b) {
                        b_terms.push(term.b);
                    }
                }
                let a = arena.add(&a_terms);
                let b = arena.add(&b_terms);
                done.push(Linear { a, b });
            }
        }
    }
    done.pop()
}

/// The coefficient `a` of an expression of the form `a·var` (or `var`
/// itself, `a = 1`), `a` free of `var` ([`depends_on`]); `None` when the
/// expression is not proportional to `var`.  A product with `var` as
/// exactly one factor and constant cofactors is read structurally;
/// otherwise the expression must be a polynomial of degree one in `var`
/// with rational coefficients and no constant term.
pub(crate) fn extract_linear_coeff(arena: &mut Arena, expr: ExprId, var: ExprId) -> Option<ExprId> {
    if expr == var {
        return Some(arena.one());
    }
    if let ExprNode::Mul(children) = arena.node(expr).clone() {
        let mut var_count = 0usize;
        let mut others = Vec::new();
        for &child in &children {
            if child == var {
                var_count += 1;
            } else if depends_on(arena, child, var) {
                // A factor that contains the variable but is not it (`var²`).
                return None;
            } else {
                others.push(child);
            }
        }
        if var_count == 1 {
            return Some(match others[..] {
                [] => arena.one(),
                [single] => single,
                _ => arena.mul(&others),
            });
        }
    }
    // Numeric coefficients: the polynomial reading.
    let poly = crate::poly::polybridge::expr_to_poly(arena, expr, var)?;
    if poly.degree()? != 1 || !num_traits::Zero::is_zero(&poly.coeff(0)) {
        return None;
    }
    let a = poly.coeff(1);
    if num_traits::Zero::is_zero(&a) {
        return None;
    }
    let nid = arena.intern_num(a);
    Some(arena.intern(ExprNode::Num(nid)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Arguments, signs and errors
// ═══════════════════════════════════════════════════════════════════════════

/// [`ComputationFailed`](SymplexError::ComputationFailed) for `operation`:
/// the local name the transforms give
/// [`SymplexError::computation_failed`].
pub(crate) fn fail(operation: &'static str, reason: impl Into<String>) -> SymplexError {
    SymplexError::computation_failed(operation, reason)
}

/// `Ok` when `id` is a symbol; otherwise
/// [`InvalidArgument`](SymplexError::InvalidArgument) naming `what` ("the
/// time variable") and the expression given.  The Fourier and Mellin entry
/// points.
pub(crate) fn require_symbol(
    arena: &Arena,
    id: ExprId,
    operation: &'static str,
    what: &str,
) -> Result<(), SymplexError> {
    if matches!(arena.node(id), ExprNode::Symbol(_)) {
        Ok(())
    } else {
        Err(SymplexError::InvalidArgument {
            operation,
            reason: format!("{what} must be a symbol, got {}", arena.display(id)),
        })
    }
}

/// The symbol of the variable `id`; otherwise
/// [`ComputationFailed`](SymplexError::ComputationFailed) "`name` must be a
/// symbol".  The Laplace and Z entry points, whose error kind and text
/// predate [`require_symbol`].
pub(crate) fn symbol_id(
    arena: &Arena,
    id: ExprId,
    operation: &'static str,
    name: &str,
) -> Result<SymbolId, SymplexError> {
    match arena.node(id) {
        ExprNode::Symbol(sym) => Ok(*sym),
        _ => Err(fail(operation, format!("{name} must be a symbol"))),
    }
}

/// The sign (`-1`, `0`, `1`) of a real parameter, from numbers and
/// assumptions ([`limit::const_sign`](crate::calculus::limit::const_sign));
/// `None` when it is not known.
pub(crate) fn param_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    crate::calculus::limit::const_sign(arena, e)
}

/// The refusal of `operation` for a parameter `e` whose sign is needed and
/// unknown: "requires `cond` (declare the sign of `e` with …)".
pub(crate) fn need_sign(
    operation: &'static str,
    arena: &Arena,
    e: ExprId,
    cond: &str,
) -> SymplexError {
    fail(
        operation,
        format!(
            "requires {cond} (declare the sign of {} with Assumption::Positive / Assumption::Negative)",
            arena.display(e)
        ),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// `Sum(body, var, 1, hi)`.
    fn sum(a: &mut Arena, body: ExprId, var: ExprId, hi: ExprId) -> ExprId {
        let one = a.one();
        a.intern(ExprNode::Sum(body, var, one, hi))
    }

    #[test]
    fn dependence_is_free_occurrence() {
        let mut a = Arena::new();
        let t = a.symbol("t");
        let m = a.symbol("m");
        let two = a.int(2);
        let t2 = a.pow(t, two);
        // Σ_{t=1}^{m} t²: `t` occurs, but only bound.
        let bound = sum(&mut a, t2, t, m);
        assert!(walk::contains(&a, bound, t));
        assert!(!depends_on(&a, bound, t));
        assert!(depends_on(&a, bound, m));
        // Σ_{t=1}^{t} t²: free in the upper limit.
        let upper = sum(&mut a, t2, t, t);
        assert!(depends_on(&a, upper, t));
        // t·Σ: the factor `t` is free.
        let prod = a.mul(&[t, bound]);
        assert!(depends_on(&a, prod, t));
        let f = split_factors(&a, prod, t);
        assert_eq!((f.consts, f.deps), (vec![bound], vec![t]));
        let s = split_independent(&mut a, prod, t);
        assert_eq!((s.coeff, s.body), (bound, t));
        // A constant is not split.
        let s = split_independent(&mut a, bound, t);
        assert_eq!((s.coeff, s.body), (a.one(), bound));
    }

    #[test]
    fn linear_in_reads_sums_products_and_bound_constants() {
        let mut a = Arena::new();
        let t = a.symbol("t");
        let c = a.symbol("c");
        let m = a.symbol("m");
        let three = a.int(3);
        // 3·t + c
        let three_t = a.mul(&[three, t]);
        let e = a.add(&[three_t, c]);
        let l = linear_in(&mut a, e, t).unwrap();
        assert_eq!((l.a, l.b), (three, c));
        // −(c·t) − 3
        let ct = a.mul(&[c, t]);
        let neg_ct = a.neg(ct);
        let neg_three = a.int(-3);
        let e = a.add(&[neg_ct, neg_three]);
        let l = linear_in(&mut a, e, t).unwrap();
        let neg_c = a.neg(c);
        assert_eq!((l.a, l.b), (neg_c, neg_three));
        // t², t·t and sin(t) are not linear.
        let two = a.int(2);
        let t2 = a.pow(t, two);
        assert!(linear_in(&mut a, t2, t).is_none());
        let sin_t = a.sin(t);
        assert!(linear_in(&mut a, sin_t, t).is_none());
        // K·t + K with K = Σ_{t=1}^{m} t²: a = b = K.
        let k = sum(&mut a, t2, t, m);
        let kt = a.mul(&[k, t]);
        let e = a.add(&[kt, k]);
        let l = linear_in(&mut a, e, t).unwrap();
        assert_eq!((l.a, l.b), (k, k));
        assert_eq!(extract_linear_coeff(&mut a, kt, t), Some(k));
        assert_eq!(extract_linear_coeff(&mut a, e, t), None);
        // A constant: a = 0.
        let l = linear_in(&mut a, k, t).unwrap();
        assert_eq!((l.a, l.b), (a.zero(), k));
    }

    #[test]
    fn the_scope_walk_stops_at_a_binder_of_the_variable() {
        let mut a = Arena::new();
        let t = a.symbol("t");
        let k = a.symbol("k");
        let m = a.symbol("m");
        let one = a.one();
        let tp1 = a.add(&[t, one]);
        let step = a.heaviside(tp1);
        // H(t + 1) + Σ_{t=1}^{m} H(t + 1): the step is shared.
        let bound = sum(&mut a, step, t, m);
        let e = a.add(&[step, bound]);
        let order = scope_post_order(&a, e, t);
        assert!(order.contains(&step) && order.contains(&bound) && order.contains(&m));
        assert_eq!(order.last(), Some(&e));
        // Alone, the sum's body is out of scope; the limit is in.
        let order = scope_post_order(&a, bound, t);
        assert!(!order.contains(&step) && !order.contains(&t));
        assert!(order.contains(&m));
        assert!(!any_in_scope(&a, bound, t, |n| matches!(
            n,
            ExprNode::Heaviside(_)
        )));
        // A sum over another variable is entered.
        let other = sum(&mut a, step, k, m);
        assert!(any_in_scope(&a, other, t, |n| matches!(
            n,
            ExprNode::Heaviside(_)
        )));
        assert_eq!(
            scope_post_order(&a, other, t),
            walk::post_order_ids(&a, other)
        );
        // Rewriting H(t + 1) → 1 reaches the free copy only.
        let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for id in scope_post_order(&a, e, t) {
            let rebuilt = rebuild_in_scope(&mut a, id, t, &cache);
            let new = if rebuilt == step { one } else { rebuilt };
            cache.insert(id, new);
        }
        let want = a.add(&[one, bound]);
        assert_eq!(cache[&e], want);
        // … and the upper limit of a sum over `t` is rewritten.
        let upper = sum(&mut a, step, t, step);
        let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();
        for id in scope_post_order(&a, upper, t) {
            let rebuilt = rebuild_in_scope(&mut a, id, t, &cache);
            let new = if rebuilt == step { one } else { rebuilt };
            cache.insert(id, new);
        }
        let want = sum(&mut a, step, t, one);
        assert_eq!(cache[&upper], want);
    }

    #[test]
    fn entry_point_errors() {
        let mut a = Arena::new();
        let t = a.symbol("t");
        let two = a.int(2);
        assert!(require_symbol(&a, t, "op", "the variable").is_ok());
        let err = require_symbol(&a, two, "op", "the variable").unwrap_err();
        assert_eq!(
            err.to_string(),
            SymplexError::InvalidArgument {
                operation: "op",
                reason: "the variable must be a symbol, got 2".into()
            }
            .to_string()
        );
        assert!(symbol_id(&a, t, "op", "t").is_ok());
        assert!(matches!(
            symbol_id(&a, two, "op", "t"),
            Err(SymplexError::ComputationFailed { operation: "op", reason }) if reason == "t must be a symbol"
        ));
        assert_eq!(param_sign(&mut a, two), Some(1));
    }
}
