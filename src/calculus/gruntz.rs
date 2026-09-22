//! Gruntz algorithm for computing symbolic limits.
//!
//! Implements the algorithm from Dominik Gruntz's PhD thesis (ETH Zürich, 1996)
//! for computing limits of expressions as a variable approaches infinity.
//! Any limit `lim(x→a)` is first converted to `lim(x→∞)` via substitution.
//!
//! # Algorithm Overview
//!
//! 1. Find the **MRV (Most Rapidly Varying)** set — subexpressions growing fastest
//! 2. If `x` is in the MRV set, "move up" by replacing `x → exp(x)`
//! 3. Pick `ω` from the MRV set such that `ω → 0` as `x → ∞`
//! 4. Rewrite the expression in terms of `ω`
//! 5. Extract the leading term `c₀ · ω^e₀`
//! 6. If `e₀ > 0`: limit is 0. If `e₀ < 0`: limit is ±∞. If `e₀ = 0`: recurse on `c₀`.
//!
//! # References
//!
//! - Gruntz, D. "On Computing Limits in a Symbolic Manipulation System"
//!   PhD Thesis, ETH Zürich, 1996.
//! - SymPy implementation: `sympy/series/gruntz.py`

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, ToPrimitive, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;

/// Maximum recursion depth for the Gruntz algorithm.
const MAX_DEPTH: usize = 15;

/// Total work units one top-level Gruntz invocation may spend before it
/// gives up. Every recursive step of the algorithm (`limitinf`, `mrv`,
/// `leadterm`, `compare`, `sign_at_inf`, `rewrite`) costs one unit and every
/// series expansion costs [`SERIES_COST`]. The battery of ~100 standard
/// limits in `tests/v02_transforms_limits.rs` uses well under 2 000 units
/// per invocation; the cap is set an order of magnitude higher so that
/// legitimate limits never hit it while pathological inputs are cut off
/// within a fraction of a second.
const MAX_WORK: u32 = 20_000;

/// Work charged for one series expansion (see [`MAX_WORK`]), on top of a
/// size-proportional charge.
const SERIES_COST: u32 = 25;

/// Largest expression *tree* (nodes with multiplicity) the series-based
/// fallbacks will expand. Nested function series substituted into each
/// other grow exponentially; beyond this size a single `expand` can take
/// seconds, so the attempt is abandoned instead.
const MAX_TREE_SIZE: usize = 4_000;

/// Work budget shared by one top-level Gruntz invocation.
///
/// The Gruntz algorithm is recursive in several mutually dependent ways
/// (`mrv` calls `limitinf` on exponents, `compare` calls `limitinf` on log
/// ratios, `leadterm` falls back to series expansion, …). Each recursion is
/// depth-limited, but the *breadth* of the search is not, so a pathological
/// input can still take a very long time without any single recursion
/// exceeding its depth limit. The budget bounds the total amount of work:
/// once it is exhausted every step fails with `ComputationFailed`, the
/// error propagates to the top, and the caller returns an unevaluated
/// `Limit` node.
///
/// The budget also hands out the fresh dummy symbols (`__gw0`, `__gw1`, …)
/// used for MRV substitutions.
#[derive(Debug)]
pub(crate) struct Budget {
    dummies: u32,
    work: u32,
}

impl Budget {
    pub(crate) fn new() -> Self {
        Self {
            dummies: 0,
            work: 0,
        }
    }

    /// Charge `weight` units of work; `Err` once the budget is exhausted.
    fn tick(&mut self, weight: u32) -> Result<(), crate::base::errors::SymplexError> {
        self.work = self.work.saturating_add(weight);
        if self.work > MAX_WORK {
            tracing::warn!(work = self.work, "gruntz: work budget exhausted");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "work budget exhausted (expression too complex for the limit engine)"
                    .into(),
            });
        }
        Ok(())
    }

    /// Charge for an expensive operation on `e` proportionally to its tree
    /// size; `Err` if `e` is larger than [`MAX_TREE_SIZE`] or the budget is
    /// exhausted.
    fn charge_size(
        &mut self,
        arena: &Arena,
        e: ExprId,
    ) -> Result<(), crate::base::errors::SymplexError> {
        let size = crate::transforms::pattern::tree_size_capped(arena, e, MAX_TREE_SIZE + 1);
        if size > MAX_TREE_SIZE {
            tracing::warn!(size, "gruntz: expression too large for series fallback");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "intermediate expression too large for the limit engine".into(),
            });
        }
        self.tick((size / 8) as u32)
    }

    /// Work units spent so far (for diagnostics and tests).
    #[cfg(test)]
    fn work(&self) -> u32 {
        self.work
    }
}

fn fresh_dummy(arena: &mut Arena, budget: &mut Budget) -> ExprId {
    let n = budget.dummies;
    budget.dummies += 1;
    arena.symbol(&format!("__gw{n}"))
}

/// Maclaurin series of `expr` in `var` to `order` terms, charged against
/// the work budget.
///
/// This is a guarded re-implementation of the Taylor loop in
/// [`series`](crate::calculus::series::series):
///
/// * every derivative is size-checked before it is differentiated again
///   (derivatives of nested functions grow exponentially, and one
///   unguarded `series` of `sin(sin x) − tan(tan x)` to order 10 takes
///   seconds);
/// * the value at `var = 0` is obtained with
///   [`safe_substitute`](crate::calculus::limit::safe_substitute), so a
///   pole hidden by canonicalization (`0·zoo → 0`) is reported as an
///   error instead of silently producing a wrong coefficient.
fn budgeted_series(
    arena: &mut Arena,
    budget: &mut Budget,
    expr: ExprId,
    var: ExprId,
    order: u32,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    budget.tick(SERIES_COST)?;
    let zero = arena.zero();
    let pole = || crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::series",
        reason: "pole detected at expansion point".into(),
    };

    let mut terms: Vec<ExprId> = Vec::with_capacity(order as usize);
    let mut deriv = expr;
    let mut factorial = Ratio::from_integer(BigInt::from(1));
    for k in 0..order {
        budget.charge_size(arena, deriv)?;
        let value = if crate::base::walk::contains(arena, deriv, var) {
            crate::calculus::limit::safe_substitute(arena, deriv, var, zero).ok_or_else(pole)?
        } else {
            crate::transforms::eval::eval(arena, deriv)
        };
        if is_infinite(arena, value) || value == arena.nan() || contains_singular_atom(arena, value)
        {
            return Err(pole());
        }
        if !arena.is_zero_structural(value) {
            let term = if k == 0 {
                value
            } else {
                let k_id = arena.int(i64::from(k));
                let power = arena.pow(var, k_id);
                let coeff_nid =
                    arena.intern_num(Ratio::from_integer(BigInt::from(1)) / factorial.clone());
                let coeff = arena.intern(ExprNode::Num(coeff_nid));
                arena.mul(&[value, coeff, power])
            };
            terms.push(term);
        }
        if k + 1 < order {
            if !crate::base::walk::contains(arena, deriv, var) {
                break;
            }
            deriv = crate::transforms::diff::diff(arena, deriv, var);
            deriv = crate::transforms::eval::eval(arena, deriv);
            factorial *= Ratio::from_integer(BigInt::from(i64::from(k) + 1));
        }
    }

    Ok(match terms.len() {
        0 => zero,
        1 => terms[0],
        _ => arena.add(&terms),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// SubsSet — the core data structure
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks MRV expressions, their dummy substitutions, and inter-MRV rewrites.
///
/// Mirrors SymPy's `SubsSet`. Maps `original_expr → dummy_variable`.
/// The `rewrites` dict maps `dummy → rewritten_form` for MRV elements
/// that contain other MRV elements as subexpressions.
///
/// Example: for MRV = {exp(x), exp(-x), exp(x - exp(-x))}
///   exprs = {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}
///   rewrites = {d3: exp(x - d2)}  ← d3 contains d2 as subexpression
#[derive(Clone, Debug)]
struct SubsSet {
    /// Maps: original MRV expression → dummy variable
    exprs: FxHashMap<ExprId, ExprId>,
    /// Maps: dummy variable → rewritten form (in terms of other dummies)
    rewrites: FxHashMap<ExprId, ExprId>,
}

impl SubsSet {
    fn new() -> Self {
        Self {
            exprs: FxHashMap::default(),
            rewrites: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    /// One member of the set, standing for its growth class (the callers
    /// have already returned on an empty set; the `Err` names the
    /// invariant instead of unwrapping).
    fn representative(&self) -> Result<ExprId, crate::base::errors::SymplexError> {
        self.exprs.keys().next().copied().ok_or_else(|| {
            crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz",
                reason: "internal: MRV set unexpectedly empty".into(),
            }
        })
    }

    fn len(&self) -> usize {
        self.exprs.len()
    }

    fn contains_key(&self, e: &ExprId) -> bool {
        self.exprs.contains_key(e)
    }

    /// Get or create a dummy for the given expression.
    /// Like SymPy's `SubsSet.__getitem__`: auto-creates on first access.
    fn get_or_create_dummy(
        &mut self,
        expr: ExprId,
        arena: &mut Arena,
        budget: &mut Budget,
    ) -> ExprId {
        if let Some(&d) = self.exprs.get(&expr) {
            return d;
        }
        let d = fresh_dummy(arena, budget);
        self.exprs.insert(expr, d);
        d
    }

    /// Apply all substitutions (expr → dummy) to an expression.
    fn do_subs(&self, arena: &mut Arena, e: ExprId) -> ExprId {
        let mut result = e;
        for (&orig, &dummy) in &self.exprs {
            result = crate::transforms::subs::subs(arena, result, orig, dummy);
        }
        result
    }

    /// Undo all substitutions (dummy → expr) in an expression.
    ///
    /// Used when this set loses an MRV comparison: its elements are no
    /// longer "most rapidly varying" and must reappear as ordinary
    /// sub-expressions of `x` rather than as opaque dummies.
    fn undo_subs(&self, arena: &mut Arena, e: ExprId) -> ExprId {
        let mut result = e;
        for (&orig, &dummy) in &self.exprs {
            result = crate::transforms::subs::subs(arena, result, dummy, orig);
        }
        result
    }

    /// Union two SubsSets (merge b into self).
    fn union_with(&mut self, other: &SubsSet) {
        for (&expr, &dummy) in &other.exprs {
            self.exprs.entry(expr).or_insert(dummy);
        }
        for (&dummy, &rewrite) in &other.rewrites {
            self.rewrites.entry(dummy).or_insert(rewrite);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Growth comparison
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrowthOrder {
    Less,
    Equal,
    Greater,
}

/// Compare the growth rates of `a` and `b` as `x → ∞`.
fn compare(
    arena: &mut Arena,
    a: ExprId,
    b: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<GrowthOrder, crate::base::errors::SymplexError> {
    tracing::debug!(depth, "gruntz::compare: comparing growth rates");
    budget.tick(1)?;

    let la = log_of(arena, a);
    let lb = log_of(arena, b);
    let ratio = arena.div(la, lb);

    tracing::trace!("gruntz::compare: computing limitinf of log ratio");
    let c = limitinf(arena, ratio, x, depth + 1, budget)?;

    if arena.is_zero_structural(c) {
        tracing::debug!("gruntz::compare → Less (a grows slower)");
        Ok(GrowthOrder::Less)
    } else if is_infinite(arena, c) {
        tracing::debug!("gruntz::compare → Greater (a grows faster)");
        Ok(GrowthOrder::Greater)
    } else {
        tracing::debug!("gruntz::compare → Equal (same comparability class)");
        Ok(GrowthOrder::Equal)
    }
}

/// Compute `log(|e|)` — if `e` is `exp(f)`, return `f` directly.
fn log_of(arena: &mut Arena, e: ExprId) -> ExprId {
    match arena.node(e).clone() {
        ExprNode::Exp(inner) => {
            tracing::trace!("gruntz::log_of: exp(f) → f");
            inner
        }
        _ => {
            tracing::trace!("gruntz::log_of: general → ln(|e|)");
            let abs_e = arena.abs(e);
            arena.ln(abs_e)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign determination
// ═══════════════════════════════════════════════════════════════════════════

/// Determine the *eventual* sign of `e` for large `x`: +1, -1, or 0.
///
/// Mirrors SymPy's `gruntz.sign`: `e ≈ c₀·ω^{e₀}` with `ω → 0⁺`, so the
/// sign of `e` for large `x` is the sign of `c₀` (recursively). Unlike a
/// limit-based test this is correct for expressions that tend to zero
/// (`1/x` has eventual sign `+1`, `−1/x` has `−1`).
fn sign_at_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<i32, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::sign_at_inf: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    let e = crate::transforms::eval::eval(arena, e);

    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::sign_at_inf: constant expression");
        return sign_of_constant(arena, e);
    }

    // x itself → +∞ as x → ∞, so sign is +1.
    // This is the critical base case that prevents infinite recursion
    // in the limitinf → mrv_leadterm → rewrite → sign_at_inf chain.
    if e == x {
        tracing::trace!("gruntz::sign_at_inf: e == x → +1 (x → +∞)");
        return Ok(1);
    }

    match arena.node(e).clone() {
        // Pow(x, c) with a real constant exponent: positive for large x > 0
        ExprNode::Pow(base, exp) if base == x && arena.as_num(exp).is_some() => {
            tracing::trace!("gruntz::sign_at_inf: x^c → +1");
            return Ok(1);
        }
        ExprNode::Exp(_) => {
            tracing::trace!("gruntz::sign_at_inf: exp() is always positive → +1");
            return Ok(1);
        }
        ExprNode::Neg(inner) => {
            return sign_at_inf(arena, inner, x, depth + 1, budget).map(|s| -s);
        }
        ExprNode::Mul(children) => {
            let mut result = 1i32;
            for c in children {
                let s = sign_at_inf(arena, c, x, depth + 1, budget)?;
                if s == 0 {
                    return Ok(0);
                }
                result *= s;
            }
            return Ok(result);
        }
        _ => {}
    }

    tracing::trace!("gruntz::sign_at_inf: computing leading term to determine sign");
    let (c0, _e0) = mrv_leadterm(arena, e, x, depth + 1, budget)?;
    if contains_foreign_dummy(arena, c0, x) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "leading coefficient is not a genuine constant".into(),
        });
    }
    if arena.is_zero_structural(c0) {
        return Ok(0);
    }
    let result = sign_at_inf(arena, c0, x, depth + 1, budget);
    tracing::debug!(sign = ?result, "gruntz::sign_at_inf result");
    result
}

/// Determine the sign of a constant expression (no free variable x).
///
/// Consults symbol assumptions (`a > 0`) through
/// [`limit::const_sign`](crate::calculus::limit::const_sign).
fn sign_of_constant(
    arena: &mut Arena,
    e: ExprId,
) -> Result<i32, crate::base::errors::SymplexError> {
    match crate::calculus::limit::const_sign(arena, e) {
        Some(s) => Ok(s),
        None => Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_of_constant",
            reason: format!("cannot determine the sign of {}", arena.display(e)),
        }),
    }
}

/// `true` if `e` contains an internal dummy symbol other than `x` itself.
///
/// Used to detect leading coefficients that still depend on the Gruntz
/// `ω` variable — a sign that the leading-term extraction gave up.
fn contains_foreign_dummy(arena: &Arena, e: ExprId, x: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if id != x
            && let ExprNode::Symbol(sid) = arena.node(id)
        {
            let name = arena.symbol_name(*sid);
            if name.starts_with("__gw") || name.starts_with("__lim") {
                return true;
            }
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// MRV (Most Rapidly Varying) set computation
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the MRV set of `e` with respect to `x`.
///
/// Returns `(SubsSet, rewritten_expr)` where the rewritten expression has
/// all MRV elements replaced by their dummy variables.
fn mrv(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // Simplify first (SymPy does powsimp here)
    let e = crate::transforms::eval::eval(arena, e);

    // Base case: e doesn't depend on x
    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::mrv: constant → empty MRV set");
        return Ok((SubsSet::new(), e));
    }

    // Base case: e IS x
    if e == x {
        tracing::trace!("gruntz::mrv: e == x → MRV = {{x}}");
        let mut s = SubsSet::new();
        let d = s.get_or_create_dummy(x, arena, budget);
        return Ok((s, d));
    }

    let node = arena.node(e).clone();

    match node {
        // ── Add or Mul: merge children's MRV sets ──
        ExprNode::Add(ref children) | ExprNode::Mul(ref children) => {
            let is_add = matches!(node, ExprNode::Add(_));
            let children_vec: SmallVec<[ExprId; 6]> = children.clone();
            tracing::trace!(
                n = children_vec.len(),
                kind = if is_add { "Add" } else { "Mul" },
                "gruntz::mrv: processing n-ary node"
            );
            mrv_nary(arena, e, &children_vec, x, depth, is_add, budget)
        }

        // ── Pow: handle x^n vs x^f(x) ──
        ExprNode::Pow(base, exp) => {
            tracing::trace!("gruntz::mrv: Pow node");
            if crate::base::walk::contains(arena, exp, x) {
                // x^f(x) → rewrite as exp(f(x)*ln(x))
                tracing::debug!("gruntz::mrv: Pow with x-dependent exponent → exp(exp*ln(base))");
                let ln_base = arena.ln(base);
                let product = arena.mul(&[exp, ln_base]);
                let as_exp = arena.exp(product);
                mrv(arena, as_exp, x, depth + 1, budget)
            } else {
                // x^const: MRV is just MRV of base
                let (s, rw_base) = mrv(arena, base, x, depth + 1, budget)?;
                let rebuilt = arena.pow(rw_base, exp);
                Ok((s, rebuilt))
            }
        }

        // ── Exp: the critical case ──
        ExprNode::Exp(arg) => {
            tracing::debug!("gruntz::mrv: Exp node — checking if exponent → ±∞");

            // SymPy: if exp(log(...)), simplify to avoid non-termination
            if let ExprNode::Ln(inner) = arena.node(arg).clone() {
                tracing::trace!("gruntz::mrv: exp(ln(f)) → mrv(f)");
                return mrv(arena, inner, x, depth + 1, budget);
            }

            // Check if the exponent goes to ±∞
            let li = limitinf(arena, arg, x, depth + 1, budget)?;
            let li_is_inf = is_infinite(arena, li);

            if li_is_inf {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → ∞ — new comparability class");
                // exp(arg) creates a new comparability class
                let mut s1 = SubsSet::new();
                let e1 = s1.get_or_create_dummy(e, arena, budget);

                // Also compute MRV of the exponent
                let (s2, e2) = mrv(arena, arg, x, depth + 1, budget)?;

                // Record the rewrite: dummy_for_exp(arg) = exp(rewritten_arg)
                let exp_e2 = arena.exp(e2);

                // Merge using mrv_max3 logic
                mrv_max3(arena, s1, e1, s2, exp_e2, x, depth, budget)
            } else {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → finite — same class as arg");
                let (s, rw_arg) = mrv(arena, arg, x, depth + 1, budget)?;
                let rebuilt = arena.exp(rw_arg);
                Ok((s, rebuilt))
            }
        }

        // ── Ln: always in a lower comparability class ──
        ExprNode::Ln(inner) => {
            tracing::trace!("gruntz::mrv: Ln node — recurse into argument");
            let (s, rw) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.ln(rw)))
        }

        // ── Neg ──
        ExprNode::Neg(inner) => {
            let (s, rw) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.neg(rw)))
        }

        // ── Unary functions: recurse into argument ──
        ExprNode::Sin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sin(r)))
        }
        ExprNode::Cos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.cos(r)))
        }
        ExprNode::Tan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.tan(r)))
        }
        ExprNode::Asin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.asin(r)))
        }
        ExprNode::Acos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.acos(r)))
        }
        ExprNode::Atan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.atan(r)))
        }
        ExprNode::Sinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sinh(r)))
        }
        ExprNode::Cosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.cosh(r)))
        }
        ExprNode::Tanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.tanh(r)))
        }
        ExprNode::Abs(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.abs(r)))
        }
        ExprNode::Sign(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.sign(r)))
        }
        ExprNode::Asinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.asinh(r)))
        }
        ExprNode::Acosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.acosh(r)))
        }
        ExprNode::Atanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, arena.atanh(r)))
        }
        // Special functions: the MRV set is that of the argument (the
        // function itself is handled by `unary_leadterm`, which knows the
        // asymptotic behaviour or rejects it). Treating `W(exp(x))` as an
        // opaque function of `x` would lose the inner `exp(x)` class.
        ExprNode::Erf(inner)
        | ExprNode::Erfc(inner)
        | ExprNode::Gamma(inner)
        | ExprNode::LogGamma(inner)
        | ExprNode::Digamma(inner)
        | ExprNode::LambertW(inner)
        | ExprNode::Factorial(inner)
        | ExprNode::Heaviside(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, budget)?;
            Ok((s, apply_unary(arena, &node, r)))
        }

        // ── Fallback: treat as containing x somewhere ──
        _ => {
            tracing::trace!("gruntz::mrv: fallback — treating expression as atomic with x");
            let mut s = SubsSet::new();
            let d = s.get_or_create_dummy(x, arena, budget);
            // Substitute x → d in the whole expression
            let rw = crate::transforms::subs::subs(arena, e, x, d);
            Ok((s, rw))
        }
    }
}

/// MRV for n-ary (Add/Mul) nodes.
///
/// Processes children pairwise: `mrv_max1(mrv(a), mrv(b), ...)`.
fn mrv_nary(
    arena: &mut Arena,
    _original: ExprId,
    children: &[ExprId],
    x: ExprId,
    depth: usize,
    is_add: bool,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if children.is_empty() {
        return Ok((
            SubsSet::new(),
            if is_add { arena.zero() } else { arena.one() },
        ));
    }

    // Split into x-independent and x-dependent parts
    let mut indep: SmallVec<[ExprId; 6]> = SmallVec::new();
    let mut dep: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &child in children {
        if crate::base::walk::contains(arena, child, x) {
            dep.push(child);
        } else {
            indep.push(child);
        }
    }

    if dep.is_empty() {
        return Ok((
            SubsSet::new(),
            if is_add {
                arena.add(children)
            } else {
                arena.mul(children)
            },
        ));
    }

    // Process dependent children pairwise
    let (mut combined_set, mut rw_first) = mrv(arena, dep[0], x, depth + 1, budget)?;

    for &child in &dep[1..] {
        let (child_set, rw_child) = mrv(arena, child, x, depth + 1, budget)?;

        // Merge the two MRV sets
        let (merged, rw_a, rw_b) = mrv_max1(
            arena,
            &combined_set,
            rw_first,
            &child_set,
            rw_child,
            x,
            depth,
            budget,
        )?;
        combined_set = merged;

        // Rebuild the pair using the new rewritten forms
        rw_first = if is_add {
            arena.add(&[rw_a, rw_b])
        } else {
            arena.mul(&[rw_a, rw_b])
        };
    }

    // Re-attach independent parts
    if !indep.is_empty() {
        if is_add {
            indep.push(rw_first);
            rw_first = arena.add(&indep);
        } else {
            indep.push(rw_first);
            rw_first = arena.mul(&indep);
        }
    }

    Ok((combined_set, rw_first))
}

/// Merge two MRV sets, keeping the faster-growing one.
///
/// Returns `(merged_set, rewritten_e1, rewritten_e2)`.
/// When one set dominates, the other's expressions are rewritten using
/// the dominating set's dummies.
#[allow(clippy::too_many_arguments)]
fn mrv_max1(
    arena: &mut Arena,
    s1: &SubsSet,
    e1: ExprId,
    s2: &SubsSet,
    e2: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId, ExprId), crate::base::errors::SymplexError> {
    if s1.is_empty() {
        return Ok((s2.clone(), e1, e2));
    }
    if s2.is_empty() {
        return Ok((s1.clone(), e1, e2));
    }

    let a_rep = s1.representative()?;
    let b_rep = s2.representative()?;

    // Same representative — union and unify dummies so e2 uses s1's dummies
    if a_rep == b_rep {
        let mut rw_e2 = e2;
        for (&expr, &d1_dummy) in &s1.exprs {
            if let Some(&d2_dummy) = s2.exprs.get(&expr)
                && d1_dummy != d2_dummy
            {
                tracing::trace!("gruntz::mrv_max1: unifying dummy for shared key");
                rw_e2 = crate::transforms::subs::subs(arena, rw_e2, d2_dummy, d1_dummy);
            }
        }
        let mut merged = s1.clone();
        merged.union_with(s2);
        return Ok((merged, e1, rw_e2));
    }

    tracing::debug!("gruntz::mrv_max1: comparing MRV representatives");
    match compare(arena, a_rep, b_rep, x, depth + 1, budget)? {
        GrowthOrder::Greater => {
            tracing::debug!(
                "gruntz::mrv_max1: s1 grows faster — keeping s1, rewriting e2 with s1 dummies"
            );
            // s2's elements are no longer MRV: restore them, then apply s1.
            let restored = s2.undo_subs(arena, e2);
            let rw_e2 = s1.do_subs(arena, restored);
            Ok((s1.clone(), e1, rw_e2))
        }
        GrowthOrder::Less => {
            tracing::debug!(
                "gruntz::mrv_max1: s2 grows faster — keeping s2, rewriting e1 with s2 dummies"
            );
            let restored = s1.undo_subs(arena, e1);
            let rw_e1 = s2.do_subs(arena, restored);
            Ok((s2.clone(), rw_e1, e2))
        }
        GrowthOrder::Equal => {
            tracing::debug!("gruntz::mrv_max1: same class — merging both sets");
            let mut rw_e2 = e2;
            for (&expr, &d1_dummy) in &s1.exprs {
                if let Some(&d2_dummy) = s2.exprs.get(&expr)
                    && d1_dummy != d2_dummy
                {
                    tracing::trace!("gruntz::mrv_max1: unifying dummy in Equal merge");
                    rw_e2 = crate::transforms::subs::subs(arena, rw_e2, d2_dummy, d1_dummy);
                }
            }
            let mut merged = s1.clone();
            merged.union_with(s2);
            Ok((merged, e1, rw_e2))
        }
    }
}

/// Three-way MRV merge for the Exp case.
///
/// s1 contains exp(arg) itself, s2 contains MRV of arg.
/// Returns the merged set with the correct rewrite tracked.
#[allow(clippy::too_many_arguments)]
fn mrv_max3(
    arena: &mut Arena,
    mut s1: SubsSet, // {exp(arg): d1}
    e1: ExprId,      // d1
    s2: SubsSet,     // MRV of arg
    exp_e2: ExprId,  // exp(rewritten_arg)
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if s2.is_empty() {
        return Ok((s1, e1));
    }

    let a_rep = s1.representative()?;
    let b_rep = s2.representative()?;

    tracing::debug!("gruntz::mrv_max3: comparing exp vs arg MRV");
    match compare(arena, a_rep, b_rep, x, depth + 1, budget)? {
        GrowthOrder::Greater => {
            tracing::debug!("gruntz::mrv_max3: exp dominates — keeping exp as MRV");
            Ok((s1, e1))
        }
        GrowthOrder::Less => {
            tracing::debug!("gruntz::mrv_max3: arg's MRV dominates — using that");
            Ok((s2, exp_e2))
        }
        GrowthOrder::Equal => {
            tracing::debug!("gruntz::mrv_max3: same class — merging, recording rewrite");
            // Record that d1 (the dummy for the exp) rewrites to exp(rewritten_arg)
            s1.rewrites.insert(e1, exp_e2);
            s1.union_with(&s2);
            Ok((s1, e1))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// moveup: replace x → exp(x) in a SubsSet
// ═══════════════════════════════════════════════════════════════════════════

/// Replace `x` with `exp(x)` in a single expression.
fn moveup_expr(arena: &mut Arena, e: ExprId, x: ExprId) -> ExprId {
    let exp_x = arena.exp(x);
    crate::transforms::subs::subs(arena, e, x, exp_x)
}

/// Move up a SubsSet: replace x with exp(x) in all keys and rewrite values.
fn moveup_subsset(arena: &mut Arena, s: &SubsSet, x: ExprId) -> SubsSet {
    let mut result = SubsSet::new();
    for (&expr, &dummy) in &s.exprs {
        let moved = moveup_expr(arena, expr, x);
        result.exprs.insert(moved, dummy);
    }
    for (&dummy, &rw) in &s.rewrites {
        result.rewrites.insert(dummy, moveup_expr(arena, rw, x));
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Rewrite in terms of ω
// ═══════════════════════════════════════════════════════════════════════════

/// Rewrite `exps` (the dummy-substituted expression) in terms of `w → 0`.
///
/// All elements of `omega` must be `Exp(something)` at this point
/// (guaranteed by moveup). Returns `(rewritten_f, log_w)`.
fn rewrite(
    arena: &mut Arena,
    exps: ExprId,
    omega: &SubsSet,
    x: ExprId,
    wsym: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }
    budget.tick(1)?;

    let exps_display = arena.display(exps).to_string();
    tracing::debug!(mrv_size = omega.len(), expr = %exps_display, "gruntz::rewrite: starting rewrite");

    // Collect all MRV entries: (original_expr, dummy)
    let entries: Vec<(ExprId, ExprId)> = omega.exprs.iter().map(|(&e, &d)| (e, d)).collect();

    // All entries must be Exp nodes. Pick the last one as g (representative).
    // (SymPy sorts by expression tree height; we just pick one.)
    let (g_expr, _g_dummy) = entries[entries.len() - 1];

    // Get g's exponent
    let g_exp = match arena.node(g_expr).clone() {
        ExprNode::Exp(inner) => inner,
        _ => {
            tracing::warn!("gruntz::rewrite: MRV element is not Exp after moveup!");
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::rewrite",
                reason: "MRV element is not Exp (moveup may have failed)".into(),
            });
        }
    };

    // Determine sign of g's exponent (g_exp → ±∞ by construction of the
    // MRV set, so this decides whether ω = g or ω = 1/g).
    let sig = sign_at_inf(arena, g_exp, x, depth + 1, budget)?;
    tracing::debug!(
        sign = sig,
        "gruntz::rewrite: sign of representative's exponent"
    );

    // If sig > 0 (g → ∞), use w = 1/g, so log(w) = -g_exp
    // If sig < 0 (g → 0), use w = g, so log(w) = g_exp
    let w_actual = if sig >= 0 {
        tracing::debug!("gruntz::rewrite: g → ∞, using ω = 1/g (so ω → 0)");
        arena.div(arena.one(), wsym) // w → 1/wsym effectively makes wsym = 1/w
    } else {
        tracing::debug!("gruntz::rewrite: g → 0, using ω = g (already → 0)");
        wsym
    };
    let _ = w_actual; // used conceptually

    // Build the rewrite table: for each (f = exp(f_exp), dummy_f) in omega:
    //   c = lim(f_exp / g_exp)
    //   rewrite dummy_f → exp(f_exp - c*g_exp) * wsym^(±c)
    let mut subs_table: Vec<(ExprId, ExprId)> = Vec::new();

    for &(f_expr, f_dummy) in &entries {
        // Get f's exponent (either from the Exp directly, or from rewrites)
        let f_exp = if let Some(&rewrite_form) = omega.rewrites.get(&f_dummy) {
            // f_dummy rewrites to exp(rewritten_arg), get the arg
            match arena.node(rewrite_form).clone() {
                ExprNode::Exp(inner) => inner,
                _ => rewrite_form, // shouldn't happen
            }
        } else {
            match arena.node(f_expr).clone() {
                ExprNode::Exp(inner) => inner,
                _ => {
                    tracing::warn!("gruntz::rewrite: non-Exp MRV element without rewrite");
                    continue;
                }
            }
        };

        // Compute c = lim(f_exp / g_exp, x → ∞)
        let ratio = arena.div(f_exp, g_exp);
        let c = limitinf(arena, ratio, x, depth + 1, budget)?;
        let c_display = arena.display(c).to_string();
        let f_exp_display = arena.display(f_exp).to_string();
        let g_exp_display = arena.display(g_exp).to_string();
        tracing::debug!(f_exp = %f_exp_display, g_exp = %g_exp_display, c = %c_display, "gruntz::rewrite: c = limitinf(f_exp/g_exp)");

        // Compute remainder = f_exp - c * g_exp
        let c_times_gexp = arena.mul(&[c, g_exp]);
        let remainder = arena.sub(f_exp, c_times_gexp);
        let remainder = crate::transforms::eval::eval(arena, remainder);

        // f = exp(remainder) * w^(±c)
        let exp_remainder = if arena.is_zero_structural(remainder) {
            arena.one()
        } else {
            arena.exp(remainder)
        };

        let w_power = if sig >= 0 { arena.neg(c) } else { c };
        let w_to_c = arena.pow(wsym, w_power);
        let replacement = arena.mul(&[exp_remainder, w_to_c]);
        let replacement = crate::transforms::eval::eval(arena, replacement);

        subs_table.push((f_dummy, replacement));
    }

    // Apply all substitutions to the expression
    let mut f = exps;
    for &(dummy, replacement) in &subs_table {
        f = crate::transforms::subs::subs(arena, f, dummy, replacement);
    }

    // Simplify
    let pre_simplify = arena.display(f).to_string();
    f = simplify_positive_powers(arena, f, wsym);
    f = crate::transforms::eval::eval(arena, f);
    f = crate::transforms::expand::expand(arena, f);
    f = crate::transforms::eval::eval(arena, f);
    let post_simplify = arena.display(f).to_string();
    tracing::debug!(before = %pre_simplify, after = %post_simplify, "gruntz::rewrite: simplification of rewritten expression");

    // Compute log(w)
    let logw = if sig >= 0 {
        arena.neg(g_exp) // w = exp(-g_exp), so log(w) = -g_exp
    } else {
        g_exp // w = exp(g_exp), so log(w) = g_exp
    };

    tracing::debug!("gruntz::rewrite: rewrite complete");
    Ok((f, logw))
}

// ═══════════════════════════════════════════════════════════════════════════
// Leading term extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Extract the leading term of `f` as a power of `w`.
///
/// Returns `(c0, e0)` where `f ≈ c0 * w^e0` as `w → 0`.
/// `c0` may depend on other variables (including `x`) but NOT on `w`.
///
/// `x`, `depth` and `budget` are threaded through so that sign decisions
/// for divergent arguments (`atan(g)` with `g → ±∞`) can use
/// [`sign_at_inf`].
#[allow(clippy::too_many_arguments)]
fn leadterm(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::leadterm: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // If f doesn't depend on w: (f, 0)
    if !crate::base::walk::contains(arena, f, w) {
        tracing::trace!("gruntz::leadterm: constant wrt ω → (f, 0)");
        return Ok((f, arena.zero()));
    }

    // If f IS w: (1, 1)
    if f == w {
        tracing::trace!("gruntz::leadterm: f == ω → (1, 1)");
        return Ok((arena.one(), arena.one()));
    }

    let node = arena.node(f).clone();

    match node {
        ExprNode::Mul(ref children) => {
            tracing::trace!(n = children.len(), "gruntz::leadterm: Mul");
            let mut total_coeff = arena.one();
            let mut total_exp = arena.zero();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw, x, depth + 1, budget)?;
                total_coeff = arena.mul(&[total_coeff, c]);
                total_exp = arena.add(&[total_exp, e]);
            }
            total_coeff = crate::transforms::eval::eval(arena, total_coeff);
            total_exp = crate::transforms::eval::eval(arena, total_exp);
            Ok((total_coeff, total_exp))
        }

        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, exp, w) => {
            tracing::trace!("gruntz::leadterm: Pow with constant exponent");
            let (c_b, e_b) = leadterm(arena, base, w, logw, x, depth + 1, budget)?;
            if crate::base::walk::contains(arena, c_b, w) {
                // A bounded-oscillation marker stays bounded only under
                // positive integer powers (`sin(1/ω)⁻¹` is unbounded).
                match arena.as_num(exp) {
                    Some(r) if r.is_integer() && r.is_positive() => {}
                    _ => return Err(oscillation_err()),
                }
            }
            let new_coeff = arena.pow(c_b, exp);
            let new_exp = arena.mul(&[e_b, exp]);
            Ok((
                crate::transforms::eval::eval(arena, new_coeff),
                crate::transforms::eval::eval(arena, new_exp),
            ))
        }

        // b^g with ω-dependent exponent: b^g = exp(g·ln b).
        ExprNode::Pow(base, exp) => {
            tracing::trace!("gruntz::leadterm: Pow with ω-dependent exponent → exp(g·ln b)");
            let ln_b = arena.ln(base);
            let prod = arena.mul(&[exp, ln_b]);
            let as_exp = arena.exp(prod);
            leadterm(arena, as_exp, w, logw, x, depth + 1, budget)
        }

        ExprNode::Add(ref children) => {
            tracing::trace!(
                n = children.len(),
                "gruntz::leadterm: Add — finding dominant term"
            );
            if children.is_empty() {
                return Ok((arena.zero(), arena.zero()));
            }

            // Find the term with the SMALLEST exponent (it dominates as w→0)
            let mut terms: Vec<(ExprId, ExprId, Q)> = Vec::new();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw, x, depth + 1, budget)?;
                let e_eval = crate::transforms::eval::eval(arena, e);
                let e_num = arena.as_num(e_eval).cloned();
                if let Some(r) = e_num {
                    terms.push((c, e_eval, r));
                } else {
                    // Can't determine exponent numerically; treat as O(1)
                    terms.push((c, e_eval, Ratio::from_integer(BigInt::from(0))));
                }
            }

            // Sort by exponent (ascending) — smallest exponent dominates
            terms.sort_by(|a, b| a.2.cmp(&b.2));

            // Group terms with the same leading exponent
            let min_exp = terms[0].2.clone();
            let mut coeff_sum = arena.zero();
            let leading_exp_id = terms[0].1;

            for (c, _, e_val) in &terms {
                if *e_val == min_exp {
                    coeff_sum = arena.add(&[coeff_sum, *c]);
                }
            }
            coeff_sum = crate::transforms::eval::eval(arena, coeff_sum);

            // If the leading coefficients cancel to 0 OR the coefficient
            // still depends on w (meaning further decomposition is needed),
            // fall back to series expansion.
            //
            // Example: (exp(w) - 1)/w after expansion is exp(w)/w - 1/w.
            // Both terms have exponent -1. Coefficient sum is exp(w) - 1,
            // which still contains w. The TRUE leading term requires knowing
            // that exp(w) - 1 ≈ w, so the product is w * w^(-1) = w^0.
            // Only series expansion can resolve this.
            let needs_series = arena.is_zero_structural(coeff_sum)
                || crate::base::walk::contains(arena, coeff_sum, w);

            if needs_series {
                let coeff_display = arena.display(coeff_sum).to_string();
                let f_display = arena.display(f).to_string();
                tracing::debug!(
                    coeff = %coeff_display,
                    full_expr = %f_display,
                    "gruntz::leadterm: Add coefficients cancel or depend on ω, using series expansion"
                );
                return leadterm_add_by_series(arena, f, w, logw, &min_exp, x, depth, budget);
            }

            Ok((coeff_sum, leading_exp_id))
        }

        ExprNode::Neg(inner) => {
            tracing::trace!("gruntz::leadterm: Neg");
            let (c, e) = leadterm(arena, inner, w, logw, x, depth + 1, budget)?;
            Ok((arena.neg(c), e))
        }

        ExprNode::Exp(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Exp(arg) where arg depends on ω");
            let (c_arg, e_arg) = leadterm(arena, arg, w, logw, x, depth + 1, budget)?;
            let e_eval = crate::transforms::eval::eval(arena, e_arg);
            if let Some(r) = arena.as_num(e_eval) {
                let r = r.clone();
                if r.is_positive() {
                    // arg → 0 as w → 0 → exp(0) = 1
                    return Ok((arena.one(), arena.zero()));
                }
                if r.is_zero() {
                    // arg → c_arg → exp(c_arg) as coefficient
                    let coeff = arena.exp(c_arg);
                    return Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()));
                }
            }
            // Negative exponent: exp(g) with g → ±∞ is its own comparability
            // class and should have been rewritten in terms of ω by the MRV
            // machinery.  Reaching here means the analysis is inconsistent.
            Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "exp of a divergent argument was not captured by the MRV set".into(),
            })
        }

        ExprNode::Ln(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Ln(arg) where arg depends on ω");
            // ln(c * w^e) = ln(c) + e*ln(w) = ln(c) + e*logw
            let (c_arg, e_arg) = leadterm(arena, arg, w, logw, x, depth + 1, budget)?;
            if crate::base::walk::contains(arena, c_arg, w) {
                // ln of a bounded oscillation is unbounded (and complex).
                return Err(oscillation_err());
            }
            let e_eval = crate::transforms::eval::eval(arena, e_arg);
            if arena.is_zero_structural(e_eval) {
                // arg → c_arg as w → 0, so ln(arg) → ln(c_arg)
                let coeff = arena.ln(c_arg);
                let coeff = crate::transforms::eval::eval(arena, coeff);
                if !arena.is_zero_structural(coeff) {
                    return Ok((coeff, arena.zero()));
                }
                // ln(c_arg) = 0, meaning c_arg = 1 (or equivalent).
                // We have ln(1 + δ) where δ = arg − c_arg → 0 as w → 0.
                // Taylor: ln(1 + δ) = δ − δ²/2 + δ³/3 − …
                // The leading term of ln(1 + δ) equals the leading term of δ.
                let delta = arena.sub(arg, c_arg);
                let delta = crate::transforms::eval::eval(arena, delta);
                if !arena.is_zero_structural(delta) && crate::base::walk::contains(arena, delta, w)
                {
                    tracing::debug!("gruntz::leadterm: ln(arg) with arg→1, using ln(1+δ) ≈ δ");
                    return leadterm(arena, delta, w, logw, x, depth + 1, budget);
                }
                return Ok((arena.zero(), arena.zero()));
            }
            // ln(c * w^e) = ln(c) + e*logw — treat as coefficient with power 0
            // since logw doesn't involve w (it involves x)
            let ln_c = arena.ln(c_arg);
            let e_logw = arena.mul(&[e_arg, logw]);
            let coeff = arena.add(&[ln_c, e_logw]);
            Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()))
        }

        // ── Unary functions of an ω-dependent argument ──
        ExprNode::Sin(inner)
        | ExprNode::Cos(inner)
        | ExprNode::Tan(inner)
        | ExprNode::Asin(inner)
        | ExprNode::Acos(inner)
        | ExprNode::Atan(inner)
        | ExprNode::Sinh(inner)
        | ExprNode::Cosh(inner)
        | ExprNode::Tanh(inner)
        | ExprNode::Asinh(inner)
        | ExprNode::Acosh(inner)
        | ExprNode::Atanh(inner)
        | ExprNode::Erf(inner)
        | ExprNode::Erfc(inner)
        | ExprNode::Gamma(inner)
        | ExprNode::LogGamma(inner)
        | ExprNode::Digamma(inner)
        | ExprNode::LambertW(inner)
        | ExprNode::Abs(inner)
        | ExprNode::Sign(inner)
        | ExprNode::Heaviside(inner)
            if crate::base::walk::contains(arena, inner, w) =>
        {
            unary_leadterm(arena, f, &node, inner, w, logw, x, depth, budget)
        }

        // x! = Γ(x + 1): reuse the Gamma pole and asymptotic rules (the
        // poles of x! at the negative integers are Γ's at 0, −1, …).
        ExprNode::Factorial(inner) if crate::base::walk::contains(arena, inner, w) => {
            let one = arena.one();
            let shifted = arena.add(&[inner, one]);
            let g = arena.gamma(shifted);
            leadterm(arena, g, w, logw, x, depth + 1, budget)
        }

        // Fallback: try series expansion
        _ => {
            tracing::debug!("gruntz::leadterm: fallback — trying series expansion");
            if let Ok(series) = budgeted_series(arena, budget, f, w, 4) {
                let evaled = crate::transforms::eval::eval(arena, series);
                if evaled != f && !crate::base::walk::contains(arena, evaled, w) {
                    return Ok((evaled, arena.zero()));
                }
                if evaled != f {
                    return leadterm(arena, evaled, w, logw, x, depth + 1, budget);
                }
            }
            Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: format!(
                    "cannot determine the leading behaviour of {}",
                    arena.display(f)
                ),
            })
        }
    }
}

/// Error for a bounded-oscillation marker (`sin(1/ω)`, …) reaching an
/// operation under which it is no longer bounded.
fn oscillation_err() -> crate::base::errors::SymplexError {
    crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "unbounded function of a bounded oscillation has no limit".into(),
    }
}

/// `true` if `F` is bounded and continuous on the whole real line, so that
/// `F(bounded oscillation)` is again a bounded oscillation.
fn unary_is_bounded(template: &ExprNode) -> bool {
    matches!(
        template,
        ExprNode::Sin(_)
            | ExprNode::Cos(_)
            | ExprNode::Atan(_)
            | ExprNode::Tanh(_)
            | ExprNode::Erf(_)
            | ExprNode::Erfc(_)
            | ExprNode::Abs(_)
            | ExprNode::Sign(_)
            | ExprNode::Heaviside(_)
    )
}

/// Rebuild a unary function node of the same kind as `template` applied to
/// `arg`.
fn apply_unary(arena: &mut Arena, template: &ExprNode, arg: ExprId) -> ExprId {
    match template {
        ExprNode::Sin(_) => arena.sin(arg),
        ExprNode::Cos(_) => arena.cos(arg),
        ExprNode::Tan(_) => arena.tan(arg),
        ExprNode::Asin(_) => arena.asin(arg),
        ExprNode::Acos(_) => arena.acos(arg),
        ExprNode::Atan(_) => arena.atan(arg),
        ExprNode::Sinh(_) => arena.sinh(arg),
        ExprNode::Cosh(_) => arena.cosh(arg),
        ExprNode::Tanh(_) => arena.tanh(arg),
        ExprNode::Asinh(_) => arena.asinh(arg),
        ExprNode::Acosh(_) => arena.acosh(arg),
        ExprNode::Atanh(_) => arena.atanh(arg),
        ExprNode::Erf(_) => arena.erf(arg),
        ExprNode::Erfc(_) => arena.erfc(arg),
        ExprNode::Gamma(_) => arena.gamma(arg),
        ExprNode::LogGamma(_) => arena.log_gamma(arg),
        ExprNode::Digamma(_) => arena.digamma(arg),
        ExprNode::LambertW(_) => arena.lambertw(arg),
        ExprNode::Factorial(_) => arena.factorial(arg),
        ExprNode::Abs(_) => arena.abs(arg),
        ExprNode::Sign(_) => arena.sign(arg),
        ExprNode::Heaviside(_) => arena.heaviside(arg),
        ExprNode::Exp(_) => arena.exp(arg),
        ExprNode::Ln(_) => arena.ln(arg),
        _ => arg,
    }
}

/// `true` if `F(c)` is a pole/undefined point for the function kind.
fn unary_is_singular_at(arena: &Arena, template: &ExprNode, c: ExprId) -> bool {
    let r = arena.as_num(c);
    match template {
        ExprNode::Gamma(_) | ExprNode::LogGamma(_) | ExprNode::Digamma(_) => match r {
            Some(r) => r.is_integer() && !r.is_positive(),
            None => false,
        },
        ExprNode::Factorial(_) => match r {
            Some(r) => r.is_integer() && r.is_negative(),
            None => false,
        },
        ExprNode::Atanh(_) => match r {
            Some(r) => r.abs() == Ratio::from_integer(BigInt::from(1)),
            None => false,
        },
        ExprNode::Sign(_) | ExprNode::Heaviside(_) => r.is_some_and(|r| r.is_zero()),
        _ => false,
    }
}

/// Leading term of `F(inner)` for a unary function `F`.
///
/// * `inner → 0`: expand `F` at 0 and take the first non-vanishing term.
/// * `inner → c` (finite): `F(c)` if nonzero, otherwise expand `F(c + δ)`.
/// * `inner → ±∞`: use the known asymptotic value of `F` (via the eventual
///   sign of the argument); bounded oscillating functions are treated as
///   `O(1)` and unbounded ones are rejected.
#[allow(clippy::too_many_arguments)]
fn unary_leadterm(
    arena: &mut Arena,
    f: ExprId,
    template: &ExprNode,
    inner: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let (c_in, e_in) = leadterm(arena, inner, w, logw, x, depth + 1, budget)?;
    let e_eval = crate::transforms::eval::eval(arena, e_in);
    let Some(r) = arena.as_num(e_eval).cloned() else {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "non-numeric exponent in function argument".into(),
        });
    };
    let zero = arena.zero();

    if r.is_positive() {
        // inner → 0.
        if unary_is_singular_at(arena, template, zero) {
            if crate::base::walk::contains(arena, c_in, w) {
                // 1/(bounded oscillation) is unbounded.
                return Err(oscillation_err());
            }
            // Γ(t) ≈ 1/t and ψ(t) ≈ −1/t near the origin.
            if matches!(template, ExprNode::Gamma(_) | ExprNode::Digamma(_)) {
                let neg_e = arena.neg(e_in);
                let inv_c = arena.pow(c_in, arena.neg_one());
                let inv_c = crate::transforms::eval::eval(arena, inv_c);
                let coeff = if matches!(template, ExprNode::Gamma(_)) {
                    inv_c
                } else {
                    arena.neg(inv_c)
                };
                return Ok((coeff, crate::transforms::eval::eval(arena, neg_e)));
            }
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function has a pole at the limit of its argument".into(),
            });
        }
        tracing::trace!("gruntz::leadterm: F(inner) with inner → 0, expanding F at 0");
        let t = fresh_dummy(arena, budget);
        let ft = apply_unary(arena, template, t);
        let p = budgeted_series(arena, budget, ft, t, 6)?;
        let p0 = crate::transforms::subs::subs(arena, p, t, zero);
        let p0 = crate::transforms::eval::eval(arena, p0);
        if !arena.is_zero_structural(p0) {
            return Ok((p0, zero));
        }
        let p_inner = crate::transforms::subs::subs(arena, p, t, inner);
        let p_inner = crate::transforms::eval::eval(arena, p_inner);
        if arena.is_zero_structural(p_inner) || p_inner == f {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "series expansion of function did not resolve the leading term".into(),
            });
        }
        return leadterm(arena, p_inner, w, logw, x, depth + 1, budget);
    }

    if r.is_zero() {
        // inner → c_in (a constant, possibly depending on x).
        if crate::base::walk::contains(arena, c_in, w) && !unary_is_bounded(template) {
            // `F(bounded oscillation)` is only a bounded oscillation when
            // `F` itself is bounded (ln, Γ, tan, … are not).
            return Err(oscillation_err());
        }
        if unary_is_singular_at(arena, template, c_in) {
            // Γ(−n + δ) ≈ (−1)ⁿ / (n!·δ) at its poles.
            if let ExprNode::Gamma(_) = template
                && let Some(cn) = arena.as_num(c_in).cloned()
                && cn.is_integer()
                && !cn.is_positive()
            {
                let n = (-cn.to_integer()).to_u64().unwrap_or(0);
                let mut fact = BigInt::from(1u64);
                for i in 2..=n {
                    fact *= BigInt::from(i);
                }
                let sign = if n % 2 == 0 { 1 } else { -1 };
                let coeff_rat = Ratio::from_integer(BigInt::from(sign)) / Ratio::from_integer(fact);
                let coeff = {
                    let nid = arena.intern_num(coeff_rat);
                    arena.intern(ExprNode::Num(nid))
                };
                let delta = arena.sub(inner, c_in);
                let delta = crate::transforms::eval::eval(arena, delta);
                if arena.is_zero_structural(delta) {
                    return Err(crate::base::errors::SymplexError::ComputationFailed {
                        operation: "gruntz::leadterm",
                        reason: "Γ evaluated exactly at a pole".into(),
                    });
                }
                let inv_delta = arena.pow(delta, arena.neg_one());
                let approx = arena.mul(&[coeff, inv_delta]);
                return leadterm(arena, approx, w, logw, x, depth + 1, budget);
            }
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function has a pole at the limit of its argument".into(),
            });
        }
        let v = apply_unary(arena, template, c_in);
        let v = crate::transforms::eval::eval(arena, v);
        if is_infinite(arena, v) || v == arena.nan() {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "function value is infinite at the limit of its argument".into(),
            });
        }
        if !arena.is_zero_structural(v) {
            return Ok((v, zero));
        }
        // F(c) = 0: expand F(c + δ) with δ = inner − c → 0.
        let delta = arena.sub(inner, c_in);
        let delta = crate::transforms::eval::eval(arena, delta);
        if arena.is_zero_structural(delta) {
            return Ok((zero, zero));
        }
        tracing::trace!("gruntz::leadterm: F(c) = 0, expanding F(c + δ)");
        let t = fresh_dummy(arena, budget);
        let arg = arena.add(&[c_in, t]);
        let ft = apply_unary(arena, template, arg);
        let p = budgeted_series(arena, budget, ft, t, 6)?;
        let p_delta = crate::transforms::subs::subs(arena, p, t, delta);
        let p_delta = crate::transforms::eval::eval(arena, p_delta);
        if arena.is_zero_structural(p_delta) || p_delta == f {
            return Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::leadterm",
                reason: "series expansion of function did not resolve the leading term".into(),
            });
        }
        return leadterm(arena, p_delta, w, logw, x, depth + 1, budget);
    }

    // inner → ±∞: sign of the leading coefficient decides the side.
    let s = if crate::base::walk::contains(arena, c_in, x) {
        sign_at_inf(arena, c_in, x, depth + 1, budget)?
    } else {
        sign_of_constant(arena, c_in)?
    };
    let pos = s > 0;
    let unbounded = || {
        Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::leadterm",
            reason: "unbounded function of a divergent argument".into(),
        })
    };
    match template {
        ExprNode::Atan(_) => {
            let two = arena.int(2);
            let pi = arena.pi();
            let half_pi = arena.div(pi, two);
            let v = if pos { half_pi } else { arena.neg(half_pi) };
            Ok((v, zero))
        }
        ExprNode::Erf(_) | ExprNode::Tanh(_) | ExprNode::Sign(_) => {
            let v = if pos { arena.one() } else { arena.neg_one() };
            Ok((v, zero))
        }
        ExprNode::Erfc(_) => {
            if pos {
                // erfc(+∞) decays like exp(−t²): not a power of ω.
                unbounded()
            } else {
                Ok((arena.int(2), zero))
            }
        }
        ExprNode::Heaviside(_) => Ok((if pos { arena.one() } else { zero }, zero)),
        ExprNode::Abs(_) => {
            if s == 0 {
                return unbounded();
            }
            let c = if pos { c_in } else { arena.neg(c_in) };
            Ok((crate::transforms::eval::eval(arena, c), e_in))
        }
        // Bounded oscillation: O(1), keep the function as the coefficient.
        ExprNode::Sin(_) | ExprNode::Cos(_) => Ok((f, zero)),
        // W(u) ~ ln u as u → +∞ (leading order only).
        ExprNode::LambertW(_) if pos => {
            let ln_inner = arena.ln(inner);
            leadterm(arena, ln_inner, w, logw, x, depth + 1, budget)
        }
        _ => unbounded(),
    }
}

/// Resolve an `Add` whose leading coefficients cancel (or still depend on
/// `w`) by series expansion.
///
/// The sum is first *normalised* so that every pole in `w` is explicit
/// (`(A)^k → w^{ek}·(A·w^{−e})^k`, `ln A → e·logw + ln(A·w^{−e})`, `b^g →
/// exp(g·ln b)`), then multiplied by `w^{−e_min}` to make it regular at
/// `w = 0`, and finally expanded as a Taylor series in `w`.
#[allow(clippy::too_many_arguments)]
fn leadterm_add_by_series(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    min_exp: &Q,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    let min_exp_id = {
        let nid = arena.intern_num(min_exp.clone());
        arena.intern(ExprNode::Num(nid))
    };
    budget.charge_size(arena, f)?;

    // Strategy 1: normalise poles and expand as a regular Taylor series.
    if let Ok(normalized) = normalize_poles(arena, f, w, logw, x, depth, budget) {
        let neg_min = {
            let nid = arena.intern_num(-min_exp.clone());
            arena.intern(ExprNode::Num(nid))
        };
        let w_shift = arena.pow(w, neg_min);
        let g = arena.mul(&[normalized, w_shift]);
        let g = simplify_positive_powers(arena, g, w);
        let g = crate::transforms::eval::eval(arena, g);
        budget.charge_size(arena, g)?;
        let g = crate::transforms::expand::expand(arena, g);
        let g = crate::transforms::eval::eval(arena, g);
        let g_display = arena.display(g).to_string();
        tracing::debug!(regular = %g_display, "gruntz::leadterm_add_by_series: normalised expression");

        for order in [3u32, 5, 8, 12] {
            match budgeted_series(arena, budget, g, w, order) {
                Ok(s) => {
                    let h = crate::transforms::expand::expand(arena, s);
                    let h = crate::transforms::eval::eval(arena, h);
                    if arena.is_zero_structural(h) {
                        continue;
                    }
                    let h_display = arena.display(h).to_string();
                    tracing::debug!(order, series = %h_display, "gruntz::leadterm_add_by_series: series");
                    let (c, e) = leadterm(arena, h, w, logw, x, depth + 1, budget)?;
                    if arena.is_zero_structural(c) {
                        continue;
                    }
                    let e_total = arena.add(&[e, min_exp_id]);
                    let e_total = crate::transforms::eval::eval(arena, e_total);
                    return Ok((c, e_total));
                }
                Err(_) => break,
            }
        }
    }

    // Strategy 2: expand individual functions as Taylor series, substitute
    // back, simplify algebraically, and retry.
    //
    // Substituting a *truncated* series into an expression with poles can
    // manufacture a spurious leading term when the true cancellation runs
    // deeper than the truncation order, so the result is only trusted when
    // two different orders agree.
    let mut agreed: Option<(ExprId, ExprId)> = None;
    for order in [6u32, 10, 14] {
        let func_expanded = expand_functions_as_series(arena, budget, f, w, order)?;
        if func_expanded == f {
            break;
        }
        let simplified = crate::transforms::eval::eval(arena, func_expanded);
        budget.charge_size(arena, simplified)?;
        let simplified = crate::transforms::expand::expand(arena, simplified);
        let simplified = crate::transforms::eval::eval(arena, simplified);
        budget.charge_size(arena, simplified)?;
        if simplified == f {
            break;
        }
        if arena.is_zero_structural(simplified) {
            // Everything cancelled to this order: the true leading term is
            // of higher order, try again with more terms.
            agreed = None;
            continue;
        }
        let Ok((c, e)) = leadterm(arena, simplified, w, logw, x, depth + 1, budget) else {
            break;
        };
        let c = crate::transforms::eval::eval(arena, c);
        let e = crate::transforms::eval::eval(arena, e);
        match agreed {
            Some((c0, e0)) if c0 == c && e0 == e => {
                tracing::debug!(
                    "gruntz::leadterm_add_by_series: function-series strategy agreed at two consecutive orders"
                );
                return Ok((c, e));
            }
            _ => agreed = Some((c, e)),
        }
    }

    // Strategy 3: full series expansion of f itself (works when f is
    // regular at w = 0).
    for order in 2..10 {
        if let Ok(series) = budgeted_series(arena, budget, f, w, order) {
            let expanded = crate::transforms::expand::expand(arena, series);
            let evaled = crate::transforms::eval::eval(arena, expanded);
            if evaled != f && !arena.is_zero_structural(evaled) {
                return leadterm(arena, evaled, w, logw, x, depth + 1, budget);
            }
        } else {
            break;
        }
    }

    Err(crate::base::errors::SymplexError::ComputationFailed {
        operation: "gruntz::leadterm",
        reason: "leading coefficients cancel and series expansion failed".into(),
    })
}

/// `(ω^a)^b → ω^{ab}` and `(c·ω^a)^b → c^b·ω^{ab}` (for `c` a product of
/// positive numbers and exponentials, or integer `b`).
///
/// These identities hold because `ω → 0⁺` is positive; canonicalization
/// cannot apply them to a general symbol (`√(1/ω) ≠ 1/√ω` for `ω < 0`).
/// Without them expressions such as `√(1/ω)·√(1/ω + 1) − 1/ω` never become
/// regular enough for series expansion.
fn simplify_positive_powers(arena: &mut Arena, f: ExprId, w: ExprId) -> ExprId {
    let post = crate::base::walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, w) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let new = match arena.node(rebuilt).clone() {
            ExprNode::Pow(base, b) if arena.as_num(b).is_some() => match arena.node(base).clone() {
                ExprNode::Pow(inner, a) if inner == w && arena.as_num(a).is_some() => {
                    let ab = arena.mul(&[a, b]);
                    let ab = crate::transforms::eval::eval(arena, ab);
                    arena.pow(w, ab)
                }
                ExprNode::Mul(children) => {
                    let mut w_exp = Ratio::from_integer(BigInt::from(0));
                    let mut rest: SmallVec<[ExprId; 4]> = SmallVec::new();
                    for &c in &children {
                        if c == w {
                            w_exp += Ratio::from_integer(BigInt::from(1));
                        } else if let ExprNode::Pow(inner, a) = arena.node(c).clone()
                            && inner == w
                            && let Some(r) = arena.as_num(a)
                        {
                            w_exp += r.clone();
                        } else {
                            rest.push(c);
                        }
                    }
                    let b_int = arena.as_num(b).is_some_and(|r| r.is_integer());
                    let rest_positive = rest.iter().all(|&c| {
                        arena.as_num(c).is_some_and(|r| r.is_positive())
                            || matches!(arena.node(c), ExprNode::Exp(_))
                    });
                    if w_exp.is_zero() || !(b_int || rest_positive) {
                        rebuilt
                    } else {
                        let w_exp_id = {
                            let nid = arena.intern_num(w_exp);
                            arena.intern(ExprNode::Num(nid))
                        };
                        let ab = arena.mul(&[w_exp_id, b]);
                        let ab = crate::transforms::eval::eval(arena, ab);
                        let w_part = arena.pow(w, ab);
                        let rest_mul = arena.mul(&rest);
                        let rest_part = arena.pow(rest_mul, b);
                        arena.mul(&[w_part, rest_part])
                    }
                }
                _ => rebuilt,
            },
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    cache.get(&f).copied().unwrap_or(f)
}

/// Make every pole in `w` explicit so that `f · w^{−e_min}` is regular at
/// `w = 0` and amenable to Taylor expansion. See [`leadterm_add_by_series`].
#[allow(clippy::too_many_arguments)]
fn normalize_poles(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let post = crate::base::walk::post_order_ids(arena, f);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, w) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(rebuilt).clone();
        let new = match node {
            ExprNode::Pow(base, k)
                if base != w
                    && !crate::base::walk::contains(arena, k, w)
                    && crate::base::walk::contains(arena, base, w)
                    && matches!(arena.node(base), ExprNode::Add(_)) =>
            {
                let (_, e) = leadterm(arena, base, w, logw, x, depth + 1, budget)?;
                let e = crate::transforms::eval::eval(arena, e);
                if arena.as_num(e).is_some() && !arena.is_zero_structural(e) {
                    let neg_e = arena.neg(e);
                    let w_neg_e = arena.pow(w, neg_e);
                    let scaled = arena.mul(&[base, w_neg_e]);
                    let scaled = crate::transforms::expand::expand(arena, scaled);
                    let scaled = crate::transforms::eval::eval(arena, scaled);
                    let ek = arena.mul(&[e, k]);
                    let w_ek = arena.pow(w, ek);
                    let regular_pow = arena.pow(scaled, k);
                    arena.mul(&[w_ek, regular_pow])
                } else {
                    rebuilt
                }
            }
            ExprNode::Pow(base, k) if crate::base::walk::contains(arena, k, w) => {
                let ln_b = arena.ln(base);
                let prod = arena.mul(&[k, ln_b]);
                arena.exp(prod)
            }
            ExprNode::Ln(arg) => {
                let (_, e) = leadterm(arena, arg, w, logw, x, depth + 1, budget)?;
                let e = crate::transforms::eval::eval(arena, e);
                if arena.as_num(e).is_some() && !arena.is_zero_structural(e) {
                    let neg_e = arena.neg(e);
                    let w_neg_e = arena.pow(w, neg_e);
                    let scaled = arena.mul(&[arg, w_neg_e]);
                    let scaled = crate::transforms::expand::expand(arena, scaled);
                    let scaled = crate::transforms::eval::eval(arena, scaled);
                    let e_logw = arena.mul(&[e, logw]);
                    let ln_scaled = arena.ln(scaled);
                    arena.add(&[e_logw, ln_scaled])
                } else {
                    rebuilt
                }
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    Ok(cache.get(&f).copied().unwrap_or(f))
}

// ═══════════════════════════════════════════════════════════════════════════
// Core: mrv_leadterm and limitinf
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the leading term of `e` as `x → ∞`.
///
/// Returns `(c0, e0)` where `e ≈ c0 · ω^e0` and `ω → 0`.
fn mrv_leadterm(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv_leadterm: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv_leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    if !crate::base::walk::contains(arena, e, x) {
        return Ok((e, arena.zero()));
    }

    let e_display = arena.display(e).to_string();
    tracing::debug!(depth, expr = %e_display, "gruntz::mrv_leadterm: Step 1 — computing MRV set");

    // Step 1: Compute MRV set
    let (omega, exps) = mrv(arena, e, x, depth + 1, budget)?;

    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }

    let exps_display = arena.display(exps).to_string();
    let mrv_keys: Vec<String> = omega
        .exprs
        .keys()
        .map(|k| arena.display(*k).to_string())
        .collect();
    tracing::debug!(
        mrv_size = omega.len(),
        mrv_keys = ?mrv_keys,
        rewritten = %exps_display,
        "gruntz::mrv_leadterm: MRV set computed"
    );

    // Step 2: If x is in the MRV set, move up (x → exp(x))
    // CRITICAL: do NOT recompute MRV! Just transform the existing set.
    let (omega, exps) = if omega.contains_key(&x) {
        tracing::debug!("gruntz::mrv_leadterm: Step 2 — x in MRV, moving up (x → exp(x))");
        let omega_up = moveup_subsset(arena, &omega, x);
        let exps_up = moveup_expr(arena, exps, x);
        let up_display = arena.display(exps_up).to_string();
        let up_keys: Vec<String> = omega_up
            .exprs
            .keys()
            .map(|k| arena.display(*k).to_string())
            .collect();
        tracing::debug!(moved_expr = %up_display, moved_mrv = ?up_keys, "gruntz::mrv_leadterm: after moveup");
        (omega_up, exps_up)
    } else {
        tracing::trace!("gruntz::mrv_leadterm: x not in MRV, no moveup needed");
        (omega, exps)
    };

    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }

    // Step 3: Rewrite in terms of w
    tracing::debug!("gruntz::mrv_leadterm: Step 3 — rewriting in terms of ω");
    let w = fresh_dummy(arena, budget);
    let (f, logw) = rewrite(arena, exps, &omega, x, w, depth, budget)?;

    let f_display = arena.display(f).to_string();
    let w_display = arena.display(w).to_string();
    let logw_display = arena.display(logw).to_string();
    tracing::debug!(rewritten_f = %f_display, w = %w_display, logw = %logw_display, "gruntz::mrv_leadterm: Step 4 — extracting leading term from rewritten expression");

    // Step 4: Extract leading term
    let (c0, e0) = leadterm(arena, f, w, logw, x, depth + 1, budget)?;

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::mrv_leadterm: leading term extracted");

    // Replace log(w) → logw in c0 (SymPy does this as the last step)
    let ln_w = arena.ln(w);
    let c0 = crate::transforms::subs::subs(arena, c0, ln_w, logw);

    tracing::debug!("gruntz::mrv_leadterm: done");
    Ok((c0, e0))
}

/// Compute `lim(x→∞) e` using the Gruntz algorithm.
pub(crate) fn limitinf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!(depth, "gruntz::limitinf: max recursion depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }
    budget.tick(1)?;

    // Simplify
    let e = crate::transforms::eval::eval(arena, e);

    // Base case: doesn't depend on x
    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::limitinf: constant → returning as-is");
        return Ok(e);
    }

    // Rewrite into a "tractable" form: tan → sin/cos, hyperbolic → exp,
    // inverse hyperbolic → ln, and resolve sign-dependent functions
    // (abs, sign, H, floor, piecewise, min, max) by their eventual sign.
    let e = rewrite_tractable(arena, e, x, depth, budget)?;
    if !crate::base::walk::contains(arena, e, x) {
        let v = crate::transforms::eval::eval(arena, e);
        return Ok(v);
    }

    let e_display = arena.display(e).to_string();
    let x_display = arena.display(x).to_string();
    tracing::debug!(depth, expr = %e_display, var = %x_display, "gruntz::limitinf: computing limit at infinity");

    // Compute leading term
    let (c0, e0) = mrv_leadterm(arena, e, x, depth, budget)?;
    let e0_eval = crate::transforms::eval::eval(arena, e0);

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0_eval).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::limitinf: leading term extracted, checking exponent sign");

    // f ≈ 0·ω^e0 → the expression vanishes identically.
    if arena.is_zero_structural(c0) {
        return Ok(arena.zero());
    }

    // Determine sign of e0
    let e0_sign = if let Some(r) = arena.as_num(e0_eval) {
        let r = r.clone();
        if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        }
    } else {
        sign_at_inf(arena, e0_eval, x, depth + 1, budget)?
    };

    // A coefficient that still contains ω (or another dummy) is a bounded
    // oscillation marker (`sin`/`cos` of a divergent argument).  Multiplied
    // by something that vanishes it still gives 0; otherwise there is no
    // limit — refuse rather than leak internal symbols into the answer.
    if contains_foreign_dummy(arena, c0, x) {
        if e0_sign > 0 {
            tracing::info!("gruntz::limitinf: bounded × vanishing → limit is 0");
            return Ok(arena.zero());
        }
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "expression has no limit (bounded oscillation)".into(),
        });
    }

    match e0_sign {
        s if s > 0 => {
            tracing::info!("gruntz::limitinf: e0 > 0 → limit is 0");
            Ok(arena.zero())
        }
        s if s < 0 => {
            let c0_sign = sign_at_inf(arena, c0, x, depth + 1, budget)?;
            tracing::info!(c0_sign, "gruntz::limitinf: e0 < 0 → limit is ±∞");
            if c0_sign > 0 {
                Ok(arena.infinity())
            } else if c0_sign < 0 {
                Ok(arena.neg_infinity())
            } else {
                Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "gruntz::limitinf",
                    reason: "indeterminate 0·∞ form in leading term".into(),
                })
            }
        }
        _ => {
            tracing::debug!("gruntz::limitinf: e0 = 0 → recursing on coefficient c0");
            limitinf(arena, c0, x, depth + 1, budget)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tractable rewriting
// ═══════════════════════════════════════════════════════════════════════════

/// `true` if `e` contains a node that [`rewrite_tractable`] would touch.
fn needs_tractable_rewrite(arena: &Arena, e: ExprId) -> bool {
    let mut stack = vec![e];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if matches!(
            node,
            ExprNode::Tan(_)
                | ExprNode::Sinh(_)
                | ExprNode::Cosh(_)
                | ExprNode::Tanh(_)
                | ExprNode::Asinh(_)
                | ExprNode::Acosh(_)
                | ExprNode::Atanh(_)
                | ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::DiracDelta(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Piecewise(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
        ) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// Rewrite `e` into a form the Gruntz machinery handles well, using the
/// fact that `x → +∞`:
///
/// * `tan u → sin u / cos u`
/// * `sinh u, cosh u, tanh u → ` exponentials
/// * `asinh u, acosh u, atanh u → ` logarithms
/// * `|u| → ±u`, `sign u → ±1`, `H(u) → 0/1`, `δ(u) → 0` by the eventual
///   sign of `u`
/// * `⌊u⌋, ⌈u⌉ → n` when `u` converges to a finite value
/// * `min/max → ` the eventually smallest/largest argument
/// * `Piecewise → ` the branch whose condition eventually holds
///
/// Nodes whose eventual sign cannot be determined are left untouched.
fn rewrite_tractable(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if !needs_tractable_rewrite(arena, e) {
        return Ok(e);
    }

    let post = crate::base::walk::post_order_ids(arena, e);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, x) {
            cache.insert(id, id);
            continue;
        }
        let rebuilt = crate::base::walk::rebuild_with_cache(arena, id, &cache);
        let node = arena.node(rebuilt).clone();
        let new = match node {
            ExprNode::Tan(u) => {
                let s = arena.sin(u);
                let c = arena.cos(u);
                arena.div(s, c)
            }
            ExprNode::Sinh(u) => {
                let neg_u = arena.neg(u);
                let ep = arena.exp(u);
                let em = arena.exp(neg_u);
                let diff = arena.sub(ep, em);
                let two = arena.int(2);
                arena.div(diff, two)
            }
            ExprNode::Cosh(u) => {
                let neg_u = arena.neg(u);
                let ep = arena.exp(u);
                let em = arena.exp(neg_u);
                let sum = arena.add(&[ep, em]);
                let two = arena.int(2);
                arena.div(sum, two)
            }
            ExprNode::Tanh(u) => {
                let two = arena.int(2);
                let two_u = arena.mul(&[two, u]);
                let e2u = arena.exp(two_u);
                let one = arena.one();
                let num = arena.sub(e2u, one);
                let den = arena.add(&[e2u, one]);
                arena.div(num, den)
            }
            ExprNode::Asinh(u) => {
                let two = arena.int(2);
                let u2 = arena.pow(u, two);
                let one = arena.one();
                let inner = arena.add(&[u2, one]);
                let root = arena.sqrt(inner);
                let sum = arena.add(&[u, root]);
                arena.ln(sum)
            }
            ExprNode::Acosh(u) => {
                let two = arena.int(2);
                let u2 = arena.pow(u, two);
                let one = arena.one();
                let inner = arena.sub(u2, one);
                let root = arena.sqrt(inner);
                let sum = arena.add(&[u, root]);
                arena.ln(sum)
            }
            ExprNode::Atanh(u) => {
                let one = arena.one();
                let p = arena.add(&[one, u]);
                let m = arena.sub(one, u);
                let lp = arena.ln(p);
                let lm = arena.ln(m);
                let diff = arena.sub(lp, lm);
                let two = arena.int(2);
                arena.div(diff, two)
            }
            ExprNode::Abs(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s > 0 => u,
                Ok(s) if s < 0 => arena.neg(u),
                _ => rebuilt,
            },
            ExprNode::Sign(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) => arena.int(s as i64),
                Err(_) => rebuilt,
            },
            ExprNode::Heaviside(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s > 0 => arena.one(),
                Ok(s) if s < 0 => arena.zero(),
                _ => rebuilt,
            },
            ExprNode::DiracDelta(u) => match sign_at_inf(arena, u, x, depth + 1, budget) {
                Ok(s) if s != 0 => arena.zero(),
                _ => rebuilt,
            },
            ExprNode::Floor(u) | ExprNode::Ceiling(u) => {
                let is_floor = matches!(node, ExprNode::Floor(_));
                match limitinf(arena, u, x, depth + 1, budget) {
                    Ok(l) if !is_infinite(arena, l) => match arena.as_num(l).cloned() {
                        Some(r) if r.is_integer() => {
                            let n_id = l;
                            let diff = arena.sub(u, n_id);
                            match sign_at_inf(arena, diff, x, depth + 1, budget) {
                                Ok(s) => {
                                    let one = arena.one();
                                    if s > 0 {
                                        if is_floor {
                                            n_id
                                        } else {
                                            arena.add(&[n_id, one])
                                        }
                                    } else if s < 0 {
                                        if is_floor { arena.sub(n_id, one) } else { n_id }
                                    } else {
                                        n_id
                                    }
                                }
                                Err(_) => rebuilt,
                            }
                        }
                        Some(r) => {
                            let v = if is_floor { r.floor() } else { r.ceil() };
                            let nid = arena.intern_num(v);
                            arena.intern(ExprNode::Num(nid))
                        }
                        None => rebuilt,
                    },
                    _ => rebuilt,
                }
            }
            ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
                let is_min = matches!(node, ExprNode::Min(_));
                let args = args.clone();
                let mut best = args[0];
                let mut ok = true;
                for &a in &args[1..] {
                    let diff = arena.sub(a, best);
                    match sign_at_inf(arena, diff, x, depth + 1, budget) {
                        Ok(s) => {
                            if (is_min && s < 0) || (!is_min && s > 0) {
                                best = a;
                            }
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok { best } else { rebuilt }
            }
            ExprNode::Piecewise(ref pairs) => {
                let pairs = pairs.clone();
                let mut chosen = None;
                for &(val, cond) in &pairs {
                    match eventually_true(arena, cond, x, depth, budget) {
                        Some(true) => {
                            chosen = Some(val);
                            break;
                        }
                        Some(false) => continue,
                        None => break,
                    }
                }
                chosen.unwrap_or(rebuilt)
            }
            _ => rebuilt,
        };
        cache.insert(id, new);
    }

    let result = cache.get(&e).copied().unwrap_or(e);
    if result != e {
        let r_display = arena.display(result).to_string();
        tracing::debug!(rewritten = %r_display, "gruntz::rewrite_tractable");
    }
    Ok(crate::transforms::eval::eval(arena, result))
}

/// Decide whether a boolean condition holds for all sufficiently large `x`.
fn eventually_true(
    arena: &mut Arena,
    cond: ExprId,
    x: ExprId,
    depth: usize,
    budget: &mut Budget,
) -> Option<bool> {
    if depth > MAX_DEPTH {
        return None;
    }
    let node = arena.node(cond).clone();
    match node {
        ExprNode::BoolTrue => Some(true),
        ExprNode::BoolFalse => Some(false),
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            let diff = arena.sub(a, b);
            let s = sign_at_inf(arena, diff, x, depth + 1, budget).ok()?;
            Some(match node {
                ExprNode::Gt(..) => s > 0,
                ExprNode::Ge(..) => s >= 0,
                ExprNode::Eq_(..) => s == 0,
                _ => s != 0,
            })
        }
        ExprNode::And(children) => {
            let mut all_true = true;
            for c in children {
                match eventually_true(arena, c, x, depth + 1, budget) {
                    Some(true) => {}
                    Some(false) => return Some(false),
                    None => all_true = false,
                }
            }
            if all_true { Some(true) } else { None }
        }
        ExprNode::Or(children) => {
            let mut all_false = true;
            for c in children {
                match eventually_true(arena, c, x, depth + 1, budget) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => all_false = false,
                }
            }
            if all_false { Some(false) } else { None }
        }
        ExprNode::Not(inner) => eventually_true(arena, inner, x, depth + 1, budget).map(|b| !b),
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Validate a Gruntz result: it must not contain the variable or any
/// internal dummy symbol.
fn validate_result(
    arena: &Arena,
    r: ExprId,
    x: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if crate::base::walk::contains(arena, r, x) || contains_foreign_dummy(arena, r, x) {
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz",
            reason: "expression has no limit or the limit could not be determined".into(),
        });
    }
    Ok(r)
}

/// Compute `lim(x→+∞) e` using the Gruntz algorithm, with result validation.
///
/// This is the entry point used by the one-sided limit machinery in
/// [`limit`](crate::calculus::limit), after the substitution `x = a ± 1/w`.
pub(crate) fn limit_pos_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let mut budget = Budget::new();
    let r = limitinf(arena, e, x, 0, &mut budget)?;
    validate_result(arena, r, x)
}

/// Compute `lim(z → z0) e` using the Gruntz algorithm.
///
/// For a finite `z0` this computes the *right-hand* limit `z → z0⁺` via the
/// substitution `z = z0 + 1/x`, `x → +∞`.
pub(crate) fn gruntz(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    let mut budget = Budget::new();
    gruntz_with_budget(arena, e, z, z0, &mut budget)
}

/// [`gruntz`] with an explicit work [`Budget`].
fn gruntz_with_budget(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
    budget: &mut Budget,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    tracing::info!("gruntz: entry point");

    if z0 == arena.infinity() {
        tracing::debug!("gruntz: limit at +∞");
        let r = limitinf(arena, e, z, 0, budget)?;
        return validate_result(arena, r, z);
    }

    if z0 == arena.neg_infinity() {
        tracing::debug!("gruntz: limit at -∞, substituting z = -x");
        let x = fresh_dummy(arena, budget);
        let neg_x = arena.neg(x);
        let e_sub = crate::transforms::subs::subs(arena, e, z, neg_x);
        let r = limitinf(arena, e_sub, x, 0, budget)?;
        let r = validate_result(arena, r, x)?;
        return validate_result(arena, r, z);
    }

    let e_display = arena.display(e).to_string();
    let z0_display = arena.display(z0).to_string();
    tracing::debug!(expr = %e_display, z0 = %z0_display, "gruntz: finite-point limit, substituting z = z0 + 1/x");
    let x = fresh_dummy(arena, budget);
    let one = arena.one();
    let inv_x = arena.div(one, x);
    let z0_plus_inv_x = arena.add(&[z0, inv_x]);
    let e_sub = crate::transforms::subs::subs(arena, e, z, z0_plus_inv_x);
    let sub_display = arena.display(e_sub).to_string();
    tracing::debug!(after_sub = %sub_display, "gruntz: after z = z0 + 1/x substitution");

    // Clear nested fractions, simplify.
    let e_together = crate::poly::polybridge::together(arena, e_sub);
    let e_cancelled = crate::poly::polybridge::cancel(arena, e_together, x);
    let e_simplified = crate::transforms::eval::eval(arena, e_cancelled);
    let e_simplified = crate::transforms::expand::expand(arena, e_simplified);
    let e_simplified = crate::transforms::eval::eval(arena, e_simplified);

    let simplified_display = arena.display(e_simplified).to_string();
    tracing::debug!(simplified = %simplified_display, "gruntz: finite-point expression simplified, calling limitinf");
    let r = limitinf(arena, e_simplified, x, 0, budget)?;
    let r = validate_result(arena, r, x)?;
    validate_result(arena, r, z)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Replace exp(arg), sin(arg), cos(arg), … nodes (where arg depends on w)
/// with their Taylor series in w around w=0.
///
/// This avoids pole issues with full series expansion on expressions like
/// `(exp(w)-1)/w`. By expanding `exp(w) → 1 + w + w²/2 + ...` and
/// substituting back, the algebraic simplification handles the rest:
/// `(1 + w + w²/2 - 1)/w = 1 + w/2 + ...`
///
/// Functions whose series cannot be formed (poles at `w = 0`) are left in
/// place; the only error is an exhausted work budget.
fn expand_functions_as_series(
    arena: &mut Arena,
    budget: &mut Budget,
    expr: ExprId,
    w: ExprId,
    order: u32,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    // Collect all function nodes that depend on w
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let mut result = expr;

    for &id in &post_order {
        budget.tick(1)?;
        let should_expand = match arena.node(id).clone() {
            ExprNode::Exp(arg)
            | ExprNode::Ln(arg)
            | ExprNode::Sin(arg)
            | ExprNode::Cos(arg)
            | ExprNode::Tan(arg)
            | ExprNode::Asin(arg)
            | ExprNode::Atan(arg)
            | ExprNode::Sinh(arg)
            | ExprNode::Cosh(arg)
            | ExprNode::Tanh(arg)
            | ExprNode::Asinh(arg)
            | ExprNode::Atanh(arg)
            | ExprNode::Erf(arg) => crate::base::walk::contains(arena, arg, w),
            // Non-integer powers of a regular base, e.g. `(1 + ω)^{1/2}`.
            ExprNode::Pow(base, e) => {
                crate::base::walk::contains(arena, base, w)
                    && arena.as_num(e).is_some_and(|r| !r.is_integer())
            }
            _ => false,
        };

        if should_expand && let Ok(series) = budgeted_series(arena, budget, id, w, order) {
            let expanded = crate::transforms::expand::expand(arena, series);
            let evaled = crate::transforms::eval::eval(arena, expanded);
            // `series` only rejects a value that *is* ±∞/zoo/NaN; a function
            // with a pole at ω = 0 (`ln(1/ω + 1)`) can still yield a
            // "series" containing one. Skip such nodes.
            if contains_singular_atom(arena, evaled) {
                continue;
            }
            let old_display = arena.display(id).to_string();
            let new_display = arena.display(evaled).to_string();
            tracing::trace!(
                original = %old_display,
                series = %new_display,
                "gruntz::expand_functions_as_series: expanded function"
            );
            result = crate::transforms::subs::subs(arena, result, id, evaled);
            budget.charge_size(arena, result)?;
        }
    }

    Ok(result)
}

fn is_infinite(arena: &Arena, e: ExprId) -> bool {
    e == arena.infinity() || e == arena.neg_infinity() || e == arena.complex_infinity()
}

/// `true` if `e` contains `∞`, `−∞`, `zoo`, or `NaN` anywhere.
fn contains_singular_atom(arena: &Arena, e: ExprId) -> bool {
    crate::base::walk::contains(arena, e, arena.infinity())
        || crate::base::walk::contains(arena, e, arena.neg_infinity())
        || crate::base::walk::contains(arena, e, arena.complex_infinity())
        || crate::base::walk::contains(arena, e, arena.nan())
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Constant ──

    #[test]
    fn gruntz_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let inf = a.infinity();
        let result = gruntz(&mut a, five, x, inf).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    // ── Rational functions at infinity ──

    #[test]
    fn gruntz_1_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let expr = a.div(one, x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(1/x, x→∞) = 0");
    }

    #[test]
    fn gruntz_x_over_x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let denom = a.add(&[x, one]);
        let expr = a.div(x, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "1", "lim(x/(x+1), x→∞) = 1");
    }

    #[test]
    fn gruntz_3x2_plus_1_over_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let three = a.int(3);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let three_x_sq = a.mul(&[three, x_sq]);
        let numer = a.add(&[three_x_sq, one]);
        let denom = a.add(&[x_sq, one]);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "3", "lim((3x²+1)/(x²+1), x→∞) = 3");
    }

    #[test]
    fn gruntz_x_over_x2_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let denom = a.add(&[x_sq, one]);
        let expr = a.div(x, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(x/(x²+1), x→∞) = 0");
    }

    #[test]
    fn gruntz_2x_plus_1_over_x_plus_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let two_x = a.mul(&[two, x]);
        let numer = a.add(&[two_x, one]);
        let denom = a.add(&[x, one]);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "2", "lim((2x+1)/(x+1), x→∞) = 2");
    }

    // ── Exponential limits ──

    #[test]
    fn gruntz_exp_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let expr = a.exp(neg_x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(exp(-x), x→∞) = 0");
    }

    #[test]
    fn gruntz_x_times_exp_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let expr = a.mul(&[x, exp_neg_x]);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(x·exp(-x), x→∞) = 0");
    }

    #[test]
    fn gruntz_exp_x_over_x2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        let numer = a.exp(x);
        let denom = a.pow(x, two);
        let expr = a.div(numer, denom);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert!(
            is_infinite(&a, result),
            "lim(exp(x)/x², x→∞) = ∞, got: {}",
            display(&a, result)
        );
    }

    // ── Logarithmic limits ──

    #[test]
    fn gruntz_ln_x_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ln_x = a.ln(x);
        let expr = a.div(ln_x, x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(ln(x)/x, x→∞) = 0");
    }

    // ── Finite-point limits via z0 + 1/x substitution ──

    #[test]
    fn gruntz_sin_x_over_x_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1", "lim(sin(x)/x, x→0) = 1");
    }

    #[test]
    fn gruntz_1_minus_cos_over_x2_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let cos_x = a.cos(x);
        let numer = a.sub(one, cos_x);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.div(numer, x_sq);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1/2", "lim((1-cos(x))/x², x→0) = 1/2");
    }

    #[test]
    fn gruntz_exp_minus_1_over_x_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let exp_x = a.exp(x);
        let numer = a.sub(exp_x, one);
        let expr = a.div(numer, x);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1", "lim((exp(x)-1)/x, x→0) = 1");
    }

    #[test]
    fn gruntz_exp_minus_1_minus_x_over_x2_at_0() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let exp_x = a.exp(x);
        let exp_m1 = a.sub(exp_x, one);
        let numer = a.sub(exp_m1, x);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let expr = a.div(numer, x_sq);
        let zero = a.zero();
        let result = gruntz(&mut a, expr, x, zero).unwrap();
        assert_eq!(
            display(&a, result),
            "1/2",
            "lim((exp(x)-1-x)/x², x→0) = 1/2"
        );
    }

    #[test]
    fn gruntz_x2_minus_1_over_x_minus_1_at_1() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two = a.int(2);
        let x_sq = a.pow(x, two);
        let numer = a.sub(x_sq, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = gruntz(&mut a, expr, x, one).unwrap();
        assert_eq!(display(&a, result), "2", "lim((x²-1)/(x-1), x→1) = 2");
    }

    // ── Negative infinity ──

    #[test]
    fn gruntz_exp_x_at_neg_inf() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let neg_inf = a.neg_infinity();
        let result = gruntz(&mut a, expr, x, neg_inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(exp(x), x→-∞) = 0");
    }

    // ── Direct substitution ──

    #[test]
    fn gruntz_polynomial_at_2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let two_c = a.int(2);
        let x_sq = a.pow(x, two_c);
        let expr = a.add(&[x_sq, one]);
        let two = a.int(2);
        let result = gruntz(&mut a, expr, x, two).unwrap();
        assert_eq!(display(&a, result), "5", "lim(x²+1, x→2) = 5");
    }

    // ── Work budget ──

    /// Representative hard limits must stay far below the work cap, so that
    /// the cap only ever triggers on pathological inputs.
    #[test]
    fn budget_headroom_on_hard_limits() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let inf = a.infinity();
        let zero = a.zero();
        let one = a.one();
        let two = a.int(2);
        let three = a.int(3);

        // (tan x − sin x)/x³ at 0 → 1/2 (series-heavy)
        let tan_x = a.tan(x);
        let sin_x = a.sin(x);
        let num = a.sub(tan_x, sin_x);
        let x3 = a.pow(x, three);
        let e1 = a.div(num, x3);
        // exp(x − exp(−x)) − exp(x) at ∞ → −1 (nested MRV classes)
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let arg = a.sub(x, exp_neg_x);
        let e_arg = a.exp(arg);
        let exp_x = a.exp(x);
        let e2 = a.sub(e_arg, exp_x);
        // (1/x² − 1/sin²x) at 0 → −1/3 (cancellation of poles)
        let x2 = a.pow(x, two);
        let inv_x2 = a.div(one, x2);
        let sin2 = a.pow(sin_x, two);
        let inv_sin2 = a.div(one, sin2);
        let e3 = a.sub(inv_x2, inv_sin2);

        for (e, p, want) in [(e1, zero, "1/2"), (e2, inf, "-1"), (e3, zero, "-1/3")] {
            let mut budget = Budget::new();
            let r = gruntz_with_budget(&mut a, e, x, p, &mut budget).unwrap();
            assert_eq!(display(&a, r), want);
            assert!(
                budget.work() < MAX_WORK / 4,
                "work {} too close to the cap {MAX_WORK} for {}",
                budget.work(),
                display(&a, e)
            );
        }
    }

    /// An exhausted budget aborts with `ComputationFailed` instead of spinning.
    #[test]
    fn exhausted_budget_fails_cleanly() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let inf = a.infinity();
        let neg_x = a.neg(x);
        let exp_neg_x = a.exp(neg_x);
        let e = a.mul(&[x, exp_neg_x]);
        let mut budget = Budget {
            dummies: 0,
            work: MAX_WORK,
        };
        let err = gruntz_with_budget(&mut a, e, x, inf, &mut budget).unwrap_err();
        assert!(err.to_string().contains("budget"), "{err}");
    }
}
