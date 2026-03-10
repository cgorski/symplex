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
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};

/// Maximum recursion depth for the Gruntz algorithm.
const MAX_DEPTH: usize = 15;

fn fresh_dummy(arena: &mut Arena, counter: &mut u32) -> ExprId {
    let n = *counter;
    *counter += 1;
    arena.symbol(&format!("__gw{n}"))
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
        counter: &mut u32,
    ) -> ExprId {
        if let Some(&d) = self.exprs.get(&expr) {
            return d;
        }
        let d = fresh_dummy(arena, counter);
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
    counter: &mut u32,
) -> Result<GrowthOrder, crate::base::errors::SymplexError> {
    tracing::debug!(depth, "gruntz::compare: comparing growth rates");

    let la = log_of(arena, a);
    let lb = log_of(arena, b);
    let ratio = arena.div(la, lb);

    tracing::trace!("gruntz::compare: computing limitinf of log ratio");
    let c = limitinf(arena, ratio, x, depth + 1, counter)?;

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

/// Determine the sign of `e` for large `x`: +1, -1, or 0.
fn sign_at_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
    counter: &mut u32,
) -> Result<i32, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::sign_at_inf: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

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

    // Pow(x, n) where n is a positive integer: positive for large x
    if let ExprNode::Pow(base, exp) = arena.node(e).clone()
        && base == x
        && let Some(r) = arena.as_num(exp)
        && r.is_positive()
    {
        tracing::trace!("gruntz::sign_at_inf: x^(positive) → +1");
        return Ok(1);
    }

    if let ExprNode::Exp(_) = arena.node(e) {
        tracing::trace!("gruntz::sign_at_inf: exp() is always positive → +1");
        return Ok(1);
    }

    tracing::trace!("gruntz::sign_at_inf: computing limit to determine sign");
    let lim = limitinf(arena, e, x, depth + 1, counter)?;
    let result = sign_of_constant(arena, lim);
    tracing::debug!(sign = ?result, "gruntz::sign_at_inf result");
    result
}

/// Determine the sign of a constant expression (no free variable x).
fn sign_of_constant(
    arena: &mut Arena,
    e: ExprId,
) -> Result<i32, crate::base::errors::SymplexError> {
    if arena.is_zero_structural(e) {
        return Ok(0);
    }
    if e == arena.pi() || e == arena.e_const() || e == arena.infinity() {
        return Ok(1);
    }
    if e == arena.neg_infinity() {
        return Ok(-1);
    }

    // Try numeric check
    if let Some(r) = arena.as_num(e) {
        let r = r.clone();
        return Ok(if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        });
    }

    // Evaluate and retry
    let evaled = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(evaled) {
        let r = r.clone();
        return Ok(if r.is_positive() {
            1
        } else if r.is_negative() {
            -1
        } else {
            0
        });
    }

    // Structural cases
    match arena.node(e).clone() {
        ExprNode::Neg(inner) => {
            let s = sign_of_constant(arena, inner)?;
            Ok(-s)
        }
        ExprNode::Mul(ref children) => {
            let mut result = 1i32;
            for &child in children {
                let s = sign_of_constant(arena, child)?;
                if s == 0 {
                    return Ok(0);
                }
                result *= s;
            }
            Ok(result)
        }
        ExprNode::Exp(_) => Ok(1),
        _ => {
            // Last resort: try evalf
            Err(crate::base::errors::SymplexError::ComputationFailed {
                operation: "gruntz::sign_of_constant",
                reason: "cannot determine sign".into(),
            })
        }
    }
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
    counter: &mut u32,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

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
        let d = s.get_or_create_dummy(x, arena, counter);
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
            mrv_nary(arena, e, &children_vec, x, depth, is_add, counter)
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
                mrv(arena, as_exp, x, depth + 1, counter)
            } else {
                // x^const: MRV is just MRV of base
                let (s, rw_base) = mrv(arena, base, x, depth + 1, counter)?;
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
                return mrv(arena, inner, x, depth + 1, counter);
            }

            // Check if the exponent goes to ±∞
            let li = limitinf(arena, arg, x, depth + 1, counter)?;
            let li_is_inf = is_infinite(arena, li);

            if li_is_inf {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → ∞ — new comparability class");
                // exp(arg) creates a new comparability class
                let mut s1 = SubsSet::new();
                let e1 = s1.get_or_create_dummy(e, arena, counter);

                // Also compute MRV of the exponent
                let (s2, e2) = mrv(arena, arg, x, depth + 1, counter)?;

                // Record the rewrite: dummy_for_exp(arg) = exp(rewritten_arg)
                let exp_e2 = arena.exp(e2);

                // Merge using mrv_max3 logic
                mrv_max3(arena, s1, e1, s2, exp_e2, x, depth, counter)
            } else {
                tracing::debug!("gruntz::mrv: exp(arg) with arg → finite — same class as arg");
                let (s, rw_arg) = mrv(arena, arg, x, depth + 1, counter)?;
                let rebuilt = arena.exp(rw_arg);
                Ok((s, rebuilt))
            }
        }

        // ── Ln: always in a lower comparability class ──
        ExprNode::Ln(inner) => {
            tracing::trace!("gruntz::mrv: Ln node — recurse into argument");
            let (s, rw) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.ln(rw)))
        }

        // ── Neg ──
        ExprNode::Neg(inner) => {
            let (s, rw) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.neg(rw)))
        }

        // ── Unary functions: recurse into argument ──
        ExprNode::Sin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.sin(r)))
        }
        ExprNode::Cos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.cos(r)))
        }
        ExprNode::Tan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.tan(r)))
        }
        ExprNode::Asin(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.asin(r)))
        }
        ExprNode::Acos(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.acos(r)))
        }
        ExprNode::Atan(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.atan(r)))
        }
        ExprNode::Sinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.sinh(r)))
        }
        ExprNode::Cosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.cosh(r)))
        }
        ExprNode::Tanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.tanh(r)))
        }
        ExprNode::Abs(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.abs(r)))
        }
        ExprNode::Sign(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.sign(r)))
        }
        ExprNode::Asinh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.asinh(r)))
        }
        ExprNode::Acosh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.acosh(r)))
        }
        ExprNode::Atanh(inner) => {
            let (s, r) = mrv(arena, inner, x, depth + 1, counter)?;
            Ok((s, arena.atanh(r)))
        }

        // ── Fallback: treat as containing x somewhere ──
        _ => {
            tracing::trace!("gruntz::mrv: fallback — treating expression as atomic with x");
            let mut s = SubsSet::new();
            let d = s.get_or_create_dummy(x, arena, counter);
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
    counter: &mut u32,
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
    let (mut combined_set, mut rw_first) = mrv(arena, dep[0], x, depth + 1, counter)?;

    for &child in &dep[1..] {
        let (child_set, rw_child) = mrv(arena, child, x, depth + 1, counter)?;

        // Merge the two MRV sets
        let (merged, rw_a, rw_b) = mrv_max1(
            arena,
            &combined_set,
            rw_first,
            &child_set,
            rw_child,
            x,
            depth,
            counter,
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
    counter: &mut u32,
) -> Result<(SubsSet, ExprId, ExprId), crate::base::errors::SymplexError> {
    if s1.is_empty() {
        return Ok((s2.clone(), e1, e2));
    }
    if s2.is_empty() {
        return Ok((s1.clone(), e1, e2));
    }

    let a_rep = *s1.exprs.keys().next().unwrap();
    let b_rep = *s2.exprs.keys().next().unwrap();

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
    match compare(arena, a_rep, b_rep, x, depth + 1, counter)? {
        GrowthOrder::Greater => {
            tracing::debug!(
                "gruntz::mrv_max1: s1 grows faster — keeping s1, rewriting e2 with s1 dummies"
            );
            let rw_e2 = s1.do_subs(arena, e2);
            Ok((s1.clone(), e1, rw_e2))
        }
        GrowthOrder::Less => {
            tracing::debug!(
                "gruntz::mrv_max1: s2 grows faster — keeping s2, rewriting e1 with s2 dummies"
            );
            let rw_e1 = s2.do_subs(arena, e1);
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
    counter: &mut u32,
) -> Result<(SubsSet, ExprId), crate::base::errors::SymplexError> {
    if s2.is_empty() {
        return Ok((s1, e1));
    }

    let a_rep = *s1.exprs.keys().next().unwrap();
    let b_rep = *s2.exprs.keys().next().unwrap();

    tracing::debug!("gruntz::mrv_max3: comparing exp vs arg MRV");
    match compare(arena, a_rep, b_rep, x, depth + 1, counter)? {
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
    counter: &mut u32,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if omega.is_empty() {
        return Ok((exps, arena.zero()));
    }

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

    // Determine sign of g's exponent
    let sig = sign_at_inf(arena, g_exp, x, depth + 1, counter).unwrap_or(1);
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
        let c = limitinf(arena, ratio, x, depth + 1, counter)?;
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
/// `c0` may depend on other variables but NOT on `w`.
fn leadterm(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
    logw: ExprId,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
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
                let (c, e) = leadterm(arena, child, w, logw)?;
                total_coeff = arena.mul(&[total_coeff, c]);
                total_exp = arena.add(&[total_exp, e]);
            }
            total_coeff = crate::transforms::eval::eval(arena, total_coeff);
            total_exp = crate::transforms::eval::eval(arena, total_exp);
            Ok((total_coeff, total_exp))
        }

        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, exp, w) => {
            tracing::trace!("gruntz::leadterm: Pow with constant exponent");
            let (c_b, e_b) = leadterm(arena, base, w, logw)?;
            let new_coeff = arena.pow(c_b, exp);
            let new_exp = arena.mul(&[e_b, exp]);
            Ok((
                crate::transforms::eval::eval(arena, new_coeff),
                crate::transforms::eval::eval(arena, new_exp),
            ))
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
            let mut terms: Vec<(ExprId, ExprId, Ratio<BigInt>)> = Vec::new();
            for &child in children {
                let (c, e) = leadterm(arena, child, w, logw)?;
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
            // fall back to series expansion on the full expression.
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
                    "gruntz::leadterm: Add coefficients cancel or depend on ω, using function expansion"
                );

                // Strategy 1: Expand individual exp/sin/cos as Taylor series,
                // substitute back, simplify algebraically. This avoids pole issues
                // with full series expansion (e.g., (exp(w)-1)/w has a pole at w=0
                // but exp(w) = 1 + w + w²/2 + ... is well-behaved).
                let func_expanded = expand_functions_as_series(arena, f, w, 6);
                if func_expanded != f {
                    let simplified = crate::transforms::eval::eval(arena, func_expanded);
                    let simplified = crate::transforms::expand::expand(arena, simplified);
                    let simplified = crate::transforms::eval::eval(arena, simplified);
                    let simp_display = arena.display(simplified).to_string();
                    tracing::debug!(result = %simp_display, "gruntz::leadterm: after function expansion");
                    if !arena.is_zero_structural(simplified) && simplified != f {
                        return leadterm(arena, simplified, w, logw);
                    }
                }

                // Strategy 2: Full series expansion (works for non-pole cases)
                let zero = arena.zero();
                for order in 2..10 {
                    if let Ok(series) = crate::calculus::series::series(arena, f, w, zero, order) {
                        let expanded = crate::transforms::expand::expand(arena, series);
                        let evaled = crate::transforms::eval::eval(arena, expanded);
                        let series_display = arena.display(evaled).to_string();
                        tracing::debug!(order, series = %series_display, "gruntz::leadterm: full series expansion result");
                        if evaled != f
                            && !arena.is_zero_structural(evaled)
                            && evaled != arena.zero()
                        {
                            return leadterm(arena, evaled, w, logw);
                        }
                    } else {
                        tracing::trace!(
                            order,
                            "gruntz::leadterm: full series expansion failed at this order"
                        );
                    }
                }
                // If nothing works, the limit is likely 0
                return Ok((arena.zero(), arena.zero()));
            }

            Ok((coeff_sum, leading_exp_id))
        }

        ExprNode::Neg(inner) => {
            tracing::trace!("gruntz::leadterm: Neg");
            let (c, e) = leadterm(arena, inner, w, logw)?;
            Ok((arena.neg(c), e))
        }

        ExprNode::Exp(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Exp(arg) where arg depends on ω");
            let (c_arg, e_arg) = leadterm(arena, arg, w, logw)?;
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
            // Negative exponent: exp(stuff) with stuff → ∞. Treat as O(1).
            Ok((f, arena.zero()))
        }

        ExprNode::Ln(arg) if crate::base::walk::contains(arena, arg, w) => {
            tracing::trace!("gruntz::leadterm: Ln(arg) where arg depends on ω");
            // ln(c * w^e) = ln(c) + e*ln(w) = ln(c) + e*logw
            let (c_arg, e_arg) = leadterm(arena, arg, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_arg);
            if arena.is_zero_structural(e_eval) {
                // arg → c_arg, so ln(arg) → ln(c_arg)
                let coeff = arena.ln(c_arg);
                return Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()));
            }
            // ln(c * w^e) = ln(c) + e*logw — treat as coefficient with power 0
            // since logw doesn't involve w (it involves x)
            let ln_c = arena.ln(c_arg);
            let e_logw = arena.mul(&[e_arg, logw]);
            let coeff = arena.add(&[ln_c, e_logw]);
            Ok((crate::transforms::eval::eval(arena, coeff), arena.zero()))
        }

        // ── Trig/hyperbolic functions: use Taylor approximation ──
        // sin(t) ≈ t for small t, cos(t) ≈ 1, tan(t) ≈ t
        // sinh(t) ≈ t, cosh(t) ≈ 1, tanh(t) ≈ t
        ExprNode::Sin(inner) if crate::base::walk::contains(arena, inner, w) => {
            tracing::trace!("gruntz::leadterm: Sin(inner) where inner depends on ω");
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    // inner → 0: sin(inner) ≈ inner → (c_inner, e_inner)
                    tracing::trace!("gruntz::leadterm: sin(→0) ≈ inner");
                    return Ok((c_inner, e_inner));
                }
                if r.is_zero() {
                    // inner → c_inner: sin(inner) → sin(c_inner)
                    let val = arena.sin(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            // sin(→∞) oscillates — treat as O(1) bounded
            Ok((f, arena.zero()))
        }
        ExprNode::Cos(inner) if crate::base::walk::contains(arena, inner, w) => {
            tracing::trace!("gruntz::leadterm: Cos(inner) where inner depends on ω");
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    // inner → 0: cos(inner) ≈ 1 → (1, 0)
                    tracing::trace!("gruntz::leadterm: cos(→0) ≈ 1");
                    return Ok((arena.one(), arena.zero()));
                }
                if r.is_zero() {
                    let val = arena.cos(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            Ok((f, arena.zero()))
        }
        ExprNode::Tan(inner) if crate::base::walk::contains(arena, inner, w) => {
            tracing::trace!("gruntz::leadterm: Tan(inner) where inner depends on ω");
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    // inner → 0: tan(inner) ≈ inner
                    return Ok((c_inner, e_inner));
                }
                if r.is_zero() {
                    let val = arena.tan(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            Ok((f, arena.zero()))
        }
        ExprNode::Sinh(inner) if crate::base::walk::contains(arena, inner, w) => {
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    return Ok((c_inner, e_inner)); // sinh(t) ≈ t
                }
                if r.is_zero() {
                    let val = arena.sinh(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            Ok((f, arena.zero()))
        }
        ExprNode::Cosh(inner) if crate::base::walk::contains(arena, inner, w) => {
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    return Ok((arena.one(), arena.zero())); // cosh(t) ≈ 1
                }
                if r.is_zero() {
                    let val = arena.cosh(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            Ok((f, arena.zero()))
        }
        ExprNode::Tanh(inner) if crate::base::walk::contains(arena, inner, w) => {
            let (c_inner, e_inner) = leadterm(arena, inner, w, logw)?;
            let e_eval = crate::transforms::eval::eval(arena, e_inner);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_positive() {
                    return Ok((c_inner, e_inner)); // tanh(t) ≈ t
                }
                if r.is_zero() {
                    let val = arena.tanh(c_inner);
                    return Ok((crate::transforms::eval::eval(arena, val), arena.zero()));
                }
            }
            Ok((f, arena.zero()))
        }

        // Fallback: try series expansion
        _ => {
            tracing::debug!("gruntz::leadterm: fallback — trying series expansion");
            let zero = arena.zero();
            if let Ok(series) = crate::calculus::series::series(arena, f, w, zero, 4) {
                let evaled = crate::transforms::eval::eval(arena, series);
                if evaled != f && !crate::base::walk::contains(arena, evaled, w) {
                    return Ok((evaled, arena.zero()));
                }
                if evaled != f {
                    return leadterm(arena, evaled, w, logw);
                }
            }
            // Ultimate fallback
            tracing::debug!("gruntz::leadterm: treating as O(1) coefficient");
            Ok((f, arena.zero()))
        }
    }
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
    counter: &mut u32,
) -> Result<(ExprId, ExprId), crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("gruntz::mrv_leadterm: max depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv_leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    if !crate::base::walk::contains(arena, e, x) {
        return Ok((e, arena.zero()));
    }

    let e_display = arena.display(e).to_string();
    tracing::debug!(depth, expr = %e_display, "gruntz::mrv_leadterm: Step 1 — computing MRV set");

    // Step 1: Compute MRV set
    let (omega, exps) = mrv(arena, e, x, depth + 1, counter)?;

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
    let w = fresh_dummy(arena, counter);
    let (f, logw) = rewrite(arena, exps, &omega, x, w, depth, counter)?;

    let f_display = arena.display(f).to_string();
    let w_display = arena.display(w).to_string();
    let logw_display = arena.display(logw).to_string();
    tracing::debug!(rewritten_f = %f_display, w = %w_display, logw = %logw_display, "gruntz::mrv_leadterm: Step 4 — extracting leading term from rewritten expression");

    // Step 4: Extract leading term
    let (c0, e0) = leadterm(arena, f, w, logw)?;

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
    counter: &mut u32,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!(depth, "gruntz::limitinf: max recursion depth exceeded");
        return Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    // Simplify
    let e = crate::transforms::eval::eval(arena, e);

    // Base case: doesn't depend on x
    if !crate::base::walk::contains(arena, e, x) {
        tracing::trace!("gruntz::limitinf: constant → returning as-is");
        return Ok(e);
    }

    let e_display = arena.display(e).to_string();
    let x_display = arena.display(x).to_string();
    tracing::debug!(depth, expr = %e_display, var = %x_display, "gruntz::limitinf: computing limit at infinity");

    // Compute leading term
    let (c0, e0) = mrv_leadterm(arena, e, x, depth, counter)?;
    let e0_eval = crate::transforms::eval::eval(arena, e0);

    let c0_display = arena.display(c0).to_string();
    let e0_display = arena.display(e0_eval).to_string();
    tracing::debug!(c0 = %c0_display, e0 = %e0_display, "gruntz::limitinf: leading term extracted, checking exponent sign");

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
        sign_at_inf(arena, e0_eval, x, depth + 1, counter).unwrap_or(0)
    };

    match e0_sign {
        s if s > 0 => {
            tracing::info!("gruntz::limitinf: e0 > 0 → limit is 0");
            Ok(arena.zero())
        }
        s if s < 0 => {
            let c0_sign = sign_at_inf(arena, c0, x, depth + 1, counter).unwrap_or(1);
            tracing::info!(c0_sign, "gruntz::limitinf: e0 < 0 → limit is ±∞");
            if c0_sign >= 0 {
                Ok(arena.infinity())
            } else {
                Ok(arena.neg_infinity())
            }
        }
        _ => {
            tracing::debug!("gruntz::limitinf: e0 = 0 → recursing on coefficient c0");
            limitinf(arena, c0, x, depth + 1, counter)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `lim(z → z0) e` using the Gruntz algorithm.
pub(crate) fn gruntz(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
) -> Result<ExprId, crate::base::errors::SymplexError> {
    tracing::info!("gruntz: entry point");
    let mut gruntz_counter: u32 = 0;

    if z0 == arena.infinity() {
        tracing::debug!("gruntz: limit at +∞");
        return limitinf(arena, e, z, 0, &mut gruntz_counter);
    }

    if z0 == arena.neg_infinity() {
        tracing::debug!("gruntz: limit at -∞, substituting z = -x");
        let x = fresh_dummy(arena, &mut gruntz_counter);
        let neg_x = arena.neg(x);
        let e_sub = crate::transforms::subs::subs(arena, e, z, neg_x);
        return limitinf(arena, e_sub, x, 0, &mut gruntz_counter);
    }

    let e_display = arena.display(e).to_string();
    let z0_display = arena.display(z0).to_string();
    tracing::debug!(expr = %e_display, z0 = %z0_display, "gruntz: finite-point limit, substituting z = z0 + 1/x");
    let x = fresh_dummy(arena, &mut gruntz_counter);
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
    limitinf(arena, e_simplified, x, 0, &mut gruntz_counter)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Replace exp(arg), sin(arg), cos(arg) nodes (where arg depends on w)
/// with their Taylor series in w around w=0.
///
/// This avoids pole issues with full series expansion on expressions like
/// `(exp(w)-1)/w`. By expanding `exp(w) → 1 + w + w²/2 + ...` and
/// substituting back, the algebraic simplification handles the rest:
/// `(1 + w + w²/2 - 1)/w = 1 + w/2 + ...`
fn expand_functions_as_series(arena: &mut Arena, expr: ExprId, w: ExprId, order: u32) -> ExprId {
    // Collect all function nodes that depend on w
    let post_order = crate::base::walk::post_order_ids(arena, expr);
    let zero = arena.zero();
    let mut result = expr;

    for &id in &post_order {
        let should_expand = match arena.node(id).clone() {
            ExprNode::Exp(arg) if crate::base::walk::contains(arena, arg, w) => true,
            ExprNode::Sin(arg) if crate::base::walk::contains(arena, arg, w) => true,
            ExprNode::Cos(arg) if crate::base::walk::contains(arena, arg, w) => true,
            ExprNode::Sinh(arg) if crate::base::walk::contains(arena, arg, w) => true,
            ExprNode::Cosh(arg) if crate::base::walk::contains(arena, arg, w) => true,
            _ => false,
        };

        if should_expand
            && let Ok(series) = crate::calculus::series::series(arena, id, w, zero, order)
        {
            let expanded = crate::transforms::expand::expand(arena, series);
            let evaled = crate::transforms::eval::eval(arena, expanded);
            let old_display = arena.display(id).to_string();
            let new_display = arena.display(evaled).to_string();
            tracing::trace!(
                original = %old_display,
                series = %new_display,
                "gruntz::expand_functions_as_series: expanded function"
            );
            result = crate::transforms::subs::subs(arena, result, id, evaled);
        }
    }

    result
}

fn is_infinite(arena: &Arena, e: ExprId) -> bool {
    e == arena.infinity() || e == arena.neg_infinity() || e == arena.complex_infinity()
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

    #[test]
    fn debug_exp_minus_1_over_x_at_0() {
        // Enable tracing for this test
        let _ = tracing_subscriber::fmt()
            .with_env_filter("symplex::gruntz=trace")
            .with_test_writer()
            .try_init();

        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let exp_x = a.exp(x);
        let numer = a.sub(exp_x, one);
        let expr = a.div(numer, x);
        let zero = a.zero();

        // Show what the expression looks like after substitution
        let t = a.symbol("__test_t");
        let one_over_t = a.div(one, t);
        let z0_plus = a.add(&[zero, one_over_t]);
        let subbed = crate::transforms::subs::subs(&mut a, expr, x, z0_plus);
        let together = crate::poly::polybridge::together(&mut a, subbed);
        let cancelled = crate::poly::polybridge::cancel(&mut a, together, t);
        let simplified = crate::transforms::eval::eval(&mut a, cancelled);
        let expanded = crate::transforms::expand::expand(&mut a, simplified);
        let evaled = crate::transforms::eval::eval(&mut a, expanded);

        println!("Original: {}", a.display(expr));
        println!("After z=0+1/t: {}", a.display(subbed));
        println!("After together: {}", a.display(together));
        println!("After cancel: {}", a.display(cancelled));
        println!("After eval+expand: {}", a.display(evaled));

        // Now try gruntz
        let result = gruntz(&mut a, expr, x, zero);
        match result {
            Ok(r) => println!("Gruntz result: {}", display(&a, r)),
            Err(e) => println!("Gruntz error: {e}"),
        }
    }
}
