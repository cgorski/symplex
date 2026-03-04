//! Gruntz algorithm for computing symbolic limits.
//!
//! Implements the algorithm from Dominik Gruntz's PhD thesis (ETH Zürich, 1996)
//! for computing limits of expressions as a variable approaches infinity.
//! Any limit `lim(x→a)` is first converted to `lim(x→∞)` via substitution.
//!
//! # Algorithm Overview
//!
//! The core idea: all elementary functions can be ordered by their "growth rate"
//! at infinity. The algorithm:
//!
//! 1. Find the **MRV (Most Rapidly Varying)** set — subexpressions that grow fastest
//! 2. Pick `ω` from the MRV set such that `ω → 0` as `x → ∞`
//! 3. Rewrite the expression in terms of `ω`
//! 4. Extract the leading term `c₀ · ω^e₀`
//! 5. If `e₀ > 0`: limit is 0 (leading term vanishes)
//! 6. If `e₀ < 0`: limit is ±∞ (leading term blows up)
//! 7. If `e₀ = 0`: limit = `lim(c₀)` (recurse on the coefficient)
//!
//! # Growth Rate Hierarchy
//!
//! ```text
//! constants < log(x) < x^n < exp(x) < exp(x^2) < exp(exp(x))
//! ```
//!
//! Only `exp` and `log` create new comparability classes. All algebraic
//! operations stay within the same class. This guarantees termination.
//!
//! # References
//!
//! - Gruntz, D. "On Computing Limits in a Symbolic Manipulation System"
//!   PhD Thesis, ETH Zürich, 1996.
//! - SymPy implementation: `sympy/series/gruntz.py`

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

/// Maximum recursion depth for the Gruntz algorithm.
/// Bounded by exp-nesting depth of the input (typically ≤ 5).
const MAX_DEPTH: usize = 15;

/// Counter for generating unique dummy variable names.
static GRUNTZ_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn fresh_dummy(arena: &mut Arena) -> ExprId {
    let n = GRUNTZ_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    arena.symbol(&format!("__gw{}", n))
}

// ═══════════════════════════════════════════════════════════════════════════
// Growth comparison
// ═══════════════════════════════════════════════════════════════════════════

/// Result of comparing growth rates of two expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ordering {
    /// `a` grows slower than `b`
    Less,
    /// `a` and `b` grow at the same rate
    Equal,
    /// `a` grows faster than `b`
    Greater,
}

/// Compare the growth rates of `a` and `b` as `x → ∞`.
///
/// Uses the relation: `compare(a, b) = sign(lim log|a| / log|b|)`
fn compare(
    arena: &mut Arena,
    a: ExprId,
    b: ExprId,
    x: ExprId,
    depth: usize,
) -> Result<Ordering, crate::errors::SymplexError> {
    tracing::debug!(depth, "compare: comparing growth rates of two expressions");

    // log of exp(f) is just f; log of anything else is ln(|anything|)
    let la = log_of(arena, a);
    let lb = log_of(arena, b);

    let ratio = arena.div(la, lb);
    tracing::trace!("compare: computing limitinf of log ratio");
    let c = limitinf(arena, ratio, x, depth + 1)?;

    let result = if arena.is_zero_structural(c) {
        tracing::debug!("compare: result = Less (a grows slower)");
        Ordering::Less
    } else if is_infinite(arena, c) {
        tracing::debug!("compare: result = Greater (a grows faster)");
        Ordering::Greater
    } else {
        tracing::debug!("compare: result = Equal (same growth class)");
        Ordering::Equal
    };
    Ok(result)
}

/// Compute `log(|e|)` — but if `e` is `exp(f)`, return `f` directly
/// to avoid `log(exp(...))` chains.
fn log_of(arena: &mut Arena, e: ExprId) -> ExprId {
    match arena.node(e).clone() {
        ExprNode::Exp(inner) => {
            tracing::trace!("log_of: exp(f) → f (avoiding log(exp) chain)");
            inner
        }
        _ => {
            tracing::trace!("log_of: general case → ln(|e|)");
            let abs_e = arena.abs(e);
            arena.ln(abs_e)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign determination
// ═══════════════════════════════════════════════════════════════════════════

/// Determine the sign of `e` for large `x` (as `x → ∞`).
///
/// Returns `1` for positive, `-1` for negative, `0` for zero.
fn sign_at_inf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
) -> Result<i32, crate::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("sign_at_inf: max depth exceeded");
        return Err(crate::errors::SymplexError::ComputationFailed {
            operation: "gruntz::sign_at_inf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    // If e doesn't depend on x, determine sign from the expression itself
    if !crate::walk::contains(arena, e, x) {
        tracing::trace!("sign_at_inf: expression is constant, checking sign directly");
        return sign_of_constant(arena, e);
    }

    // For exp(f): always positive
    if let ExprNode::Exp(_) = arena.node(e) {
        tracing::trace!("sign_at_inf: exp(...) is always positive");
        return Ok(1);
    }

    // Try evaluating the limit and checking the sign of the result
    tracing::trace!("sign_at_inf: computing limit to determine sign");
    let lim = limitinf(arena, e, x, depth + 1)?;
    let result = sign_of_constant(arena, lim);
    tracing::debug!(sign = ?result, "sign_at_inf: determined sign");
    result
}

/// Determine the sign of a constant expression (no free variable).
fn sign_of_constant(arena: &mut Arena, e: ExprId) -> Result<i32, crate::errors::SymplexError> {
    if arena.is_zero_structural(e) {
        return Ok(0);
    }

    // Check for known positive constants
    if e == arena.pi() || e == arena.e_const() || e == arena.infinity() {
        return Ok(1);
    }
    if e == arena.neg_infinity() {
        return Ok(-1);
    }

    // Try to evaluate numerically
    if let Some(r) = arena.as_num(e) {
        let r = r.clone();
        if r.is_positive() {
            return Ok(1);
        }
        if r.is_negative() {
            return Ok(-1);
        }
        if r.is_zero() {
            return Ok(0);
        }
    }

    // Try evaluating to float
    let evaled = crate::eval::eval(arena, e);
    if let Some(r) = arena.as_num(evaled) {
        let r = r.clone();
        if r.is_positive() {
            return Ok(1);
        }
        if r.is_negative() {
            return Ok(-1);
        }
        if r.is_zero() {
            return Ok(0);
        }
    }

    // Check for Neg
    if let ExprNode::Neg(inner) = arena.node(e).clone() {
        let inner_sign = sign_of_constant(arena, inner)?;
        return Ok(-inner_sign);
    }

    // Check for Mul: sign is product of signs
    if let ExprNode::Mul(ref children) = arena.node(e).clone() {
        let mut result = 1i32;
        for &child in children {
            let s = sign_of_constant(arena, child)?;
            if s == 0 {
                return Ok(0);
            }
            result *= s;
        }
        return Ok(result);
    }

    // Check for Exp: always positive
    if let ExprNode::Exp(_) = arena.node(e) {
        return Ok(1);
    }

    // Can't determine sign
    Err(crate::errors::SymplexError::ComputationFailed {
        operation: "gruntz::sign_of_constant",
        reason: format!("cannot determine sign of expression"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// MRV (Most Rapidly Varying) set
// ═══════════════════════════════════════════════════════════════════════════

/// The MRV set: maps each MRV expression to a fresh dummy variable.
/// Also tracks how dummies should be rewritten in terms of each other.
#[derive(Clone, Debug)]
struct MrvSet {
    /// Maps: original MRV expression → dummy variable
    exprs: FxHashMap<ExprId, ExprId>,
}

impl MrvSet {
    fn new() -> Self {
        Self {
            exprs: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.exprs.is_empty()
    }

    fn contains(&self, e: &ExprId) -> bool {
        self.exprs.contains_key(e)
    }

    fn insert(&mut self, expr: ExprId, dummy: ExprId) {
        self.exprs.insert(expr, dummy);
    }
}

/// Compute the MRV set of `e` with respect to `x`.
///
/// Returns the MRV set and the expression rewritten with dummies
/// substituted for MRV elements.
fn mrv(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
) -> Result<(MrvSet, ExprId), crate::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("mrv: max depth exceeded");
        return Err(crate::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    // Base case: e doesn't depend on x
    if !crate::walk::contains(arena, e, x) {
        tracing::trace!("mrv: expression is constant (no x), empty MRV set");
        return Ok((MrvSet::new(), e));
    }

    // Base case: e is x itself
    if e == x {
        tracing::trace!("mrv: expression is x itself, MRV = {{x}}");
        let d = fresh_dummy(arena);
        let mut set = MrvSet::new();
        set.insert(x, d);
        return Ok((set, d));
    }

    let node = arena.node(e).clone();

    match node {
        // Exp node: the most important case
        ExprNode::Exp(arg) => {
            tracing::debug!("mrv: processing Exp node, checking if exponent → ±∞");
            // Check if the exponent goes to ±∞
            match limitinf(arena, arg, x, depth + 1) {
                Ok(lim) if is_infinite(arena, lim) => {
                    tracing::debug!("mrv: exp(arg) where arg → ∞, creates new comparability class");
                    // exp(arg) creates a new comparability class
                    // It's in the MRV set itself
                    let d = fresh_dummy(arena);
                    let mut set = MrvSet::new();
                    set.insert(e, d);

                    // Also need to compute MRV of the argument
                    // and merge (the argument might have its own fast-growing parts)
                    let (arg_set, _arg_rewritten) = mrv(arena, arg, x, depth + 1)?;

                    // Merge: keep the faster-growing set
                    if arg_set.is_empty() {
                        tracing::trace!("mrv: arg has empty MRV, using exp as sole MRV element");
                        Ok((set, d))
                    } else {
                        // Compare growth rates
                        let arg_rep = *arg_set.exprs.keys().next().unwrap();
                        tracing::debug!("mrv: comparing exp(arg) growth vs arg's MRV");
                        match compare(arena, e, arg_rep, x, depth + 1)? {
                            Ordering::Greater => {
                                tracing::debug!("mrv: exp(arg) grows faster, keeping it");
                                Ok((set, d))
                            }
                            Ordering::Less => {
                                tracing::debug!("mrv: arg's MRV grows faster, using that");
                                let rewritten = rewrite_with_mrv(arena, e, &arg_set)?;
                                Ok((arg_set, rewritten))
                            }
                            Ordering::Equal => {
                                tracing::debug!("mrv: same growth class, merging MRV sets");
                                let mut merged = set;
                                for (&expr, &dummy) in &arg_set.exprs {
                                    merged.insert(expr, dummy);
                                }
                                Ok((merged, d))
                            }
                        }
                    }
                }
                _ => {
                    tracing::debug!("mrv: exp(arg) where arg → finite, recursing into arg");
                    // exp(arg) where arg → finite: stay in same class as arg
                    // Recurse into arg
                    mrv(arena, arg, x, depth + 1).map(|(set, rewritten_arg)| {
                        let new_exp = arena.exp(rewritten_arg);
                        (set, new_exp)
                    })
                }
            }
        }

        // Ln node: recurse into argument
        ExprNode::Ln(inner) => {
            tracing::trace!("mrv: processing Ln node, recursing into argument");
            mrv(arena, inner, x, depth + 1).map(|(set, rewritten)| {
                let new_ln = arena.ln(rewritten);
                (set, new_ln)
            })
        }

        // Add: merge MRV sets of all children
        ExprNode::Add(ref children) => {
            tracing::trace!(n_children = children.len(), "mrv: processing Add node");
            let children_vec: SmallVec<[ExprId; 6]> = children.clone();
            mrv_add_or_mul(arena, &children_vec, x, depth, true)
        }

        // Mul: merge MRV sets of all children
        ExprNode::Mul(ref children) => {
            tracing::trace!(n_children = children.len(), "mrv: processing Mul node");
            let children_vec: SmallVec<[ExprId; 6]> = children.clone();
            mrv_add_or_mul(arena, &children_vec, x, depth, false)
        }

        // Pow: recurse into base and exponent
        ExprNode::Pow(base, exp) => {
            tracing::trace!("mrv: processing Pow node");
            let (set_b, rw_b) = mrv(arena, base, x, depth + 1)?;
            let (set_e, rw_e) = mrv(arena, exp, x, depth + 1)?;
            let merged = merge_mrv_sets(arena, set_b, set_e, x, depth)?;
            let new_pow = arena.pow(rw_b, rw_e);
            Ok((merged, new_pow))
        }

        // Neg: recurse
        ExprNode::Neg(inner) => {
            mrv(arena, inner, x, depth + 1).map(|(set, rw)| (set, arena.neg(rw)))
        }

        // Unary functions: recurse into argument
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
        | ExprNode::Abs(inner)
        | ExprNode::Sign(inner)
        | ExprNode::Factorial(inner) => {
            let (set, rw) = mrv(arena, inner, x, depth + 1)?;
            let rebuilt = rebuild_unary(arena, &node, rw);
            Ok((set, rebuilt))
        }

        // Anything else: treat as atomic if it contains x
        _ => {
            // Fallback: x is somewhere in this expression
            let d = fresh_dummy(arena);
            let mut set = MrvSet::new();
            set.insert(x, d);
            Ok((set, d))
        }
    }
}

/// Rebuild a unary node with a new inner expression.
fn rebuild_unary(arena: &mut Arena, original: &ExprNode, new_inner: ExprId) -> ExprId {
    match original {
        ExprNode::Sin(_) => arena.sin(new_inner),
        ExprNode::Cos(_) => arena.cos(new_inner),
        ExprNode::Tan(_) => arena.tan(new_inner),
        ExprNode::Asin(_) => arena.asin(new_inner),
        ExprNode::Acos(_) => arena.acos(new_inner),
        ExprNode::Atan(_) => arena.atan(new_inner),
        ExprNode::Sinh(_) => arena.sinh(new_inner),
        ExprNode::Cosh(_) => arena.cosh(new_inner),
        ExprNode::Tanh(_) => arena.tanh(new_inner),
        ExprNode::Asinh(_) => arena.asinh(new_inner),
        ExprNode::Acosh(_) => arena.acosh(new_inner),
        ExprNode::Atanh(_) => arena.atanh(new_inner),
        ExprNode::Abs(_) => arena.abs(new_inner),
        ExprNode::Sign(_) => arena.sign(new_inner),
        ExprNode::Factorial(_) => arena.factorial(new_inner),
        ExprNode::Ln(_) => arena.ln(new_inner),
        ExprNode::Exp(_) => arena.exp(new_inner),
        ExprNode::Neg(_) => arena.neg(new_inner),
        _ => new_inner, // fallback
    }
}

/// MRV for Add or Mul nodes: merge children's MRV sets.
fn mrv_add_or_mul(
    arena: &mut Arena,
    children: &[ExprId],
    x: ExprId,
    depth: usize,
    is_add: bool,
) -> Result<(MrvSet, ExprId), crate::errors::SymplexError> {
    let mut combined_set = MrvSet::new();
    let mut rewritten_children: SmallVec<[ExprId; 6]> = SmallVec::new();

    for &child in children {
        let (child_set, rw_child) = mrv(arena, child, x, depth + 1)?;
        combined_set = merge_mrv_sets(arena, combined_set, child_set, x, depth)?;
        rewritten_children.push(rw_child);
    }

    let rebuilt = if is_add {
        arena.add(&rewritten_children)
    } else {
        arena.mul(&rewritten_children)
    };

    Ok((combined_set, rebuilt))
}

/// Merge two MRV sets, keeping the faster-growing one (or merging if equal).
fn merge_mrv_sets(
    arena: &mut Arena,
    a: MrvSet,
    b: MrvSet,
    x: ExprId,
    depth: usize,
) -> Result<MrvSet, crate::errors::SymplexError> {
    if a.is_empty() {
        tracing::trace!("merge_mrv_sets: a is empty, returning b");
        return Ok(b);
    }
    if b.is_empty() {
        tracing::trace!("merge_mrv_sets: b is empty, returning a");
        return Ok(a);
    }

    let a_rep = *a.exprs.keys().next().unwrap();
    let b_rep = *b.exprs.keys().next().unwrap();

    if a_rep == b_rep {
        // Same representative — merge
        let mut merged = a;
        for (&expr, &dummy) in &b.exprs {
            if !merged.contains(&expr) {
                merged.insert(expr, dummy);
            }
        }
        return Ok(merged);
    }

    tracing::debug!("merge_mrv_sets: comparing representatives");
    match compare(arena, a_rep, b_rep, x, depth + 1)? {
        Ordering::Greater => {
            tracing::debug!("merge_mrv_sets: a grows faster, keeping a");
            Ok(a)
        }
        Ordering::Less => {
            tracing::debug!("merge_mrv_sets: b grows faster, keeping b");
            Ok(b)
        }
        Ordering::Equal => {
            tracing::debug!("merge_mrv_sets: equal growth, merging both");
            let mut merged = a;
            for (&expr, &dummy) in &b.exprs {
                if !merged.contains(&expr) {
                    merged.insert(expr, dummy);
                }
            }
            Ok(merged)
        }
    }
}

/// Rewrite an expression by substituting MRV elements with their dummies.
fn rewrite_with_mrv(
    arena: &mut Arena,
    e: ExprId,
    mrv_set: &MrvSet,
) -> Result<ExprId, crate::errors::SymplexError> {
    let mut result = e;
    for (&orig, &dummy) in &mrv_set.exprs {
        result = crate::subs::subs(arena, result, orig, dummy);
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rewrite & Leading Term
// ═══════════════════════════════════════════════════════════════════════════

/// The "moveup" operation: replace `x` with `exp(x)` everywhere.
/// This ensures all MRV elements are exponentials.
fn moveup(arena: &mut Arena, e: ExprId, x: ExprId) -> ExprId {
    tracing::debug!("moveup: replacing x with exp(x)");
    let exp_x = arena.exp(x);
    crate::subs::subs(arena, e, x, exp_x)
}

/// Rewrite `e` in terms of `w` (where `w → 0`), given the MRV set.
///
/// Returns `(rewritten_expr, log_w)` where `log_w` is the exponent
/// such that `w = exp(log_w)`.
fn rewrite_in_w(
    arena: &mut Arena,
    e: ExprId,
    mrv_set: &MrvSet,
    x: ExprId,
    w: ExprId,
    depth: usize,
) -> Result<(ExprId, ExprId), crate::errors::SymplexError> {
    if mrv_set.is_empty() {
        tracing::trace!("rewrite_in_w: empty MRV set, returning expression unchanged");
        return Ok((e, arena.zero()));
    }

    tracing::debug!(
        mrv_size = mrv_set.exprs.len(),
        "rewrite_in_w: starting rewrite"
    );

    // Pick a representative g from the MRV set.
    // All elements are comparable. Pick the simplest (first).
    let (&g_expr, _) = mrv_set.exprs.iter().next().unwrap();

    // g must be exp(something) after moveup. Get the exponent.
    let g_exp = match arena.node(g_expr).clone() {
        ExprNode::Exp(inner) => {
            tracing::trace!("rewrite_in_w: representative g is Exp, using inner as g_exp");
            inner
        }
        _ => {
            tracing::trace!("rewrite_in_w: representative g is not Exp, using ln(g)");
            arena.ln(g_expr)
        }
    };

    // Determine sign of g's exponent to decide if w = g or w = 1/g
    let sig = sign_at_inf(arena, g_exp, x, depth + 1).unwrap_or(1);
    tracing::debug!(sign = sig, "rewrite_in_w: sign of g's exponent");

    // If g's exponent is positive (g → ∞), use w = 1/g so w → 0
    // If g's exponent is negative (g → 0), use w = g (already → 0)
    let (log_w, w_is_reciprocal) = if sig >= 0 {
        tracing::debug!("rewrite_in_w: g → ∞, using w = 1/g (w → 0)");
        // w = exp(-g_exp), so log(w) = -g_exp
        (arena.neg(g_exp), true)
    } else {
        tracing::debug!("rewrite_in_w: g → 0, using w = g (already → 0)");
        // w = exp(g_exp), so log(w) = g_exp
        (g_exp, false)
    };

    // Now rewrite each MRV element in terms of w.
    // For each f = exp(f_exp) in the MRV set:
    //   f = exp(f_exp) = exp((f_exp/g_exp) * g_exp) = w^(-f_exp/g_exp)  if w_is_reciprocal
    //   f = exp(f_exp) = exp((f_exp/g_exp) * g_exp) = w^(f_exp/g_exp)   otherwise
    let mut result = e;

    tracing::debug!("rewrite_in_w: substituting MRV elements with w-expressions");
    for (&f_expr, _) in &mrv_set.exprs {
        let f_exp = match arena.node(f_expr).clone() {
            ExprNode::Exp(inner) => inner,
            _ => arena.ln(f_expr),
        };

        // Compute the ratio c = lim(f_exp / g_exp)
        let ratio = arena.div(f_exp, g_exp);
        let c = limitinf(arena, ratio, x, depth + 1)?;

        // f_exp - c * g_exp should be in a lower comparability class
        let c_times_gexp = arena.mul(&[c, g_exp]);
        let remainder = arena.sub(f_exp, c_times_gexp);
        let remainder = crate::eval::eval(arena, remainder);

        // f = exp(remainder) * w^(±c)
        let exp_remainder = if arena.is_zero_structural(remainder) {
            arena.one()
        } else {
            arena.exp(remainder)
        };

        let w_power = if w_is_reciprocal { arena.neg(c) } else { c };

        let w_to_c = arena.pow(w, w_power);
        let replacement = arena.mul(&[exp_remainder, w_to_c]);
        let replacement = crate::eval::eval(arena, replacement);

        result = crate::subs::subs(arena, result, f_expr, replacement);
    }

    // Simplify the rewritten expression
    tracing::trace!("rewrite_in_w: simplifying rewritten expression (eval → expand → eval)");
    result = crate::eval::eval(arena, result);
    result = crate::expand::expand(arena, result);
    result = crate::eval::eval(arena, result);

    tracing::debug!("rewrite_in_w: rewrite complete");
    Ok((result, log_w))
}

/// Extract the leading term of `f` as a power series in `w`.
///
/// Returns `(c0, e0)` where `f ≈ c0 * w^e0` as `w → 0`.
/// `c0` may still depend on other variables but NOT on `w`.
fn leadterm(
    arena: &mut Arena,
    f: ExprId,
    w: ExprId,
) -> Result<(ExprId, ExprId), crate::errors::SymplexError> {
    // If f doesn't depend on w, it's a constant: (f, 0)
    if !crate::walk::contains(arena, f, w) {
        tracing::trace!("leadterm: expression is constant wrt w → (f, 0)");
        return Ok((f, arena.zero()));
    }

    // If f is exactly w: (1, 1)
    if f == w {
        tracing::trace!("leadterm: expression is w itself → (1, 1)");
        return Ok((arena.one(), arena.one()));
    }

    let node = arena.node(f).clone();

    match node {
        // Mul: leadterm is product of leadterms
        ExprNode::Mul(ref children) => {
            tracing::trace!(
                n = children.len(),
                "leadterm: Mul — multiplying child leadterms"
            );
            let mut total_coeff = arena.one();
            let mut total_exp = arena.zero();

            for &child in children {
                let (c, e) = leadterm(arena, child, w)?;
                total_coeff = arena.mul(&[total_coeff, c]);
                total_exp = arena.add(&[total_exp, e]);
            }

            total_coeff = crate::eval::eval(arena, total_coeff);
            total_exp = crate::eval::eval(arena, total_exp);

            Ok((total_coeff, total_exp))
        }

        // Pow(base, exp) where exp is constant wrt w
        ExprNode::Pow(base, exp) if !crate::walk::contains(arena, exp, w) => {
            tracing::trace!("leadterm: Pow with constant exponent");
            let (c_b, e_b) = leadterm(arena, base, w)?;
            // (c_b * w^e_b)^exp = c_b^exp * w^(e_b*exp)
            let new_coeff = arena.pow(c_b, exp);
            let new_exp = arena.mul(&[e_b, exp]);
            let new_coeff = crate::eval::eval(arena, new_coeff);
            let new_exp = crate::eval::eval(arena, new_exp);
            Ok((new_coeff, new_exp))
        }

        // Add: take the term with the smallest exponent (dominant term)
        ExprNode::Add(ref children) => {
            tracing::trace!(
                n = children.len(),
                "leadterm: Add — finding dominant term (smallest exponent)"
            );
            if children.is_empty() {
                return Ok((arena.zero(), arena.zero()));
            }

            let mut best_coeff = arena.zero();
            let mut best_exp: Option<ExprId> = None;
            let mut best_exp_val: Option<Ratio<BigInt>> = None;

            for &child in children {
                let (c, e) = leadterm(arena, child, w)?;
                let e_eval = crate::eval::eval(arena, e);

                let e_num = arena.as_num(e_eval).cloned();

                match (&best_exp_val, &e_num) {
                    (None, _) => {
                        best_coeff = c;
                        best_exp = Some(e_eval);
                        best_exp_val = e_num;
                    }
                    (Some(current_best), Some(this_e)) => {
                        if this_e < current_best {
                            // This term has a smaller exponent (dominates)
                            best_coeff = c;
                            best_exp = Some(e_eval);
                            best_exp_val = Some(this_e.clone());
                        } else if this_e == current_best {
                            // Same exponent — add coefficients
                            best_coeff = arena.add(&[best_coeff, c]);
                            best_coeff = crate::eval::eval(arena, best_coeff);
                        }
                        // If this_e > current_best, this term is subdominant; skip
                    }
                    (Some(_), None) => {
                        // Can't compare symbolically — keep what we have
                    }
                }
            }

            Ok((best_coeff, best_exp.unwrap_or_else(|| arena.zero())))
        }

        // Neg: leadterm of -f is (-c, e) from leadterm of f
        ExprNode::Neg(inner) => {
            tracing::trace!("leadterm: Neg — negating coefficient");
            let (c, e) = leadterm(arena, inner, w)?;
            Ok((arena.neg(c), e))
        }

        // Exp(arg) where arg depends on w
        ExprNode::Exp(arg) => {
            tracing::trace!("leadterm: Exp(arg) where arg depends on w");
            // exp(arg) where arg involves w
            // Try to extract the leading power of w from arg
            let (c_arg, e_arg) = leadterm(arena, arg, w)?;

            // If e_arg < 0: arg blows up → exp blows up → leadterm is (exp(c_arg * w^e_arg), 0)
            // But this means the expression wasn't properly rewritten. For now treat as coeff.
            // If e_arg = 0: arg → c_arg → exp(c_arg) is the coefficient
            // If e_arg > 0: arg → 0 → exp(0) = 1

            let e_eval = crate::eval::eval(arena, e_arg);
            if let Some(r) = arena.as_num(e_eval) {
                let r = r.clone();
                if r.is_positive() {
                    // arg → 0 → exp(0) = 1
                    return Ok((arena.one(), arena.zero()));
                }
                if r.is_zero() {
                    // arg → c_arg → exp(c_arg)
                    let coeff = arena.exp(c_arg);
                    let coeff = crate::eval::eval(arena, coeff);
                    return Ok((coeff, arena.zero()));
                }
            }

            // arg has negative power of w → exp grows. Treat as coefficient.
            Ok((f, arena.zero()))
        }

        // Ln(arg) where arg depends on w
        ExprNode::Ln(arg) => {
            tracing::trace!("leadterm: Ln(arg) where arg depends on w");
            // ln(c * w^e) = ln(c) + e*ln(w)
            let (c_arg, e_arg) = leadterm(arena, arg, w)?;
            let e_eval = crate::eval::eval(arena, e_arg);
            if let Some(r) = arena.as_num(e_eval) {
                if r.is_zero() {
                    // arg → c_arg → ln(c_arg)
                    let coeff = arena.ln(c_arg);
                    let coeff = crate::eval::eval(arena, coeff);
                    return Ok((coeff, arena.zero()));
                }
            }
            // General case: treat as coefficient with power 0
            Ok((f, arena.zero()))
        }

        // Any other node that contains w: try series expansion as fallback
        _ => {
            tracing::debug!("leadterm: unknown node type, trying series fallback");
            // Try using the existing series infrastructure
            let zero = arena.zero();
            if let Ok(series) = crate::series::series(arena, f, w, zero, 2) {
                let evaled = crate::eval::eval(arena, series);
                if evaled != f {
                    return leadterm(arena, evaled, w);
                }
            }

            // Ultimate fallback: treat as O(1) coefficient
            tracing::debug!("leadterm: treating unknown expression as O(1) coefficient");
            Ok((f, arena.zero()))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Core algorithm: mrv_leadterm and limitinf
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the leading term of `e` as `x → ∞`.
///
/// Returns `(c0, e0)` where `e ≈ c0 · ω^e0` and `ω → 0`.
fn mrv_leadterm(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
) -> Result<(ExprId, ExprId), crate::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!("mrv_leadterm: max depth exceeded");
        return Err(crate::errors::SymplexError::ComputationFailed {
            operation: "gruntz::mrv_leadterm",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    // If e doesn't depend on x, leading term is (e, 0)
    if !crate::walk::contains(arena, e, x) {
        tracing::trace!("mrv_leadterm: expression is constant, returning (e, 0)");
        return Ok((e, arena.zero()));
    }

    tracing::debug!(depth, "mrv_leadterm: Step 1 — computing MRV set");

    // Step 1: Compute MRV set
    let (mrv_set, _rewritten) = mrv(arena, e, x, depth + 1)?;

    if mrv_set.is_empty() {
        tracing::debug!("mrv_leadterm: empty MRV set, expression is effectively constant");
        return Ok((e, arena.zero()));
    }

    tracing::debug!(
        mrv_size = mrv_set.exprs.len(),
        "mrv_leadterm: MRV set computed"
    );

    // Step 2: If x is in the MRV set, "move up": replace x → exp(x)
    let (mrv_set, e_moved) = if mrv_set.contains(&x) {
        tracing::debug!("mrv_leadterm: Step 2 — x is in MRV set, moving up (x → exp(x))");
        let e_up = moveup(arena, e, x);
        // Recompute MRV after moveup
        let (new_set, _new_rw) = mrv(arena, e_up, x, depth + 1)?;
        tracing::debug!(
            new_mrv_size = new_set.exprs.len(),
            "mrv_leadterm: MRV recomputed after moveup"
        );
        (new_set, e_up)
    } else {
        tracing::trace!("mrv_leadterm: x is NOT in MRV set, no moveup needed");
        (mrv_set, e)
    };

    if mrv_set.is_empty() {
        tracing::debug!("mrv_leadterm: empty MRV after moveup, returning as constant");
        return Ok((e_moved, arena.zero()));
    }

    // Step 3: Introduce fresh w and rewrite e in terms of w
    tracing::debug!("mrv_leadterm: Step 3 — introducing fresh ω and rewriting");
    let w = fresh_dummy(arena);
    let (f, _log_w) = rewrite_in_w(arena, e_moved, &mrv_set, x, w, depth)?;

    tracing::debug!("mrv_leadterm: Step 4 — extracting leading term from rewritten expression");

    // Step 4: Extract leading term of f in w
    let (c0, e0) = leadterm(arena, f, w)?;
    tracing::debug!("mrv_leadterm: leading term extracted successfully");

    // c0 and e0 should not depend on w
    // Substitute back: replace w-dummies with their actual values
    // Actually, c0 and e0 should be free of w by construction.
    // But they may contain x, which is what we want (limitinf will recurse on c0).

    Ok((c0, e0))
}

/// Compute `lim(x→∞) e` using the Gruntz algorithm.
///
/// This is the core recursive function. It:
/// 1. Computes the leading term `c0 · ω^e0`
/// 2. Checks the sign of `e0`
/// 3. If `e0 > 0`: returns 0
/// 4. If `e0 < 0`: returns ±∞
/// 5. If `e0 = 0`: recurses on `c0`
pub(crate) fn limitinf(
    arena: &mut Arena,
    e: ExprId,
    x: ExprId,
    depth: usize,
) -> Result<ExprId, crate::errors::SymplexError> {
    if depth > MAX_DEPTH {
        tracing::warn!(depth, "limitinf: max recursion depth exceeded");
        return Err(crate::errors::SymplexError::ComputationFailed {
            operation: "gruntz::limitinf",
            reason: "maximum recursion depth exceeded".into(),
        });
    }

    // Simplify first
    let e = crate::eval::eval(arena, e);

    // Base case: e doesn't depend on x
    if !crate::walk::contains(arena, e, x) {
        tracing::trace!("limitinf: expression is constant (no x), returning as-is");
        return Ok(e);
    }

    tracing::debug!(
        depth,
        "limitinf: computing limit at infinity — extracting leading term"
    );

    // Compute leading term
    let (c0, e0) = mrv_leadterm(arena, e, x, depth)?;

    let e0_eval = crate::eval::eval(arena, e0);

    tracing::debug!("limitinf: leading term (c0, e0) obtained, determining sign of e0");

    // Check the sign of the exponent e0
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
        // Try sign_at_inf for symbolic e0
        sign_at_inf(arena, e0_eval, x, depth + 1).unwrap_or(0)
    };

    match e0_sign {
        s if s > 0 => {
            // e0 > 0: leading term vanishes → limit is 0
            tracing::info!("limitinf: e0 > 0 → ω^e0 vanishes → limit is 0");
            Ok(arena.zero())
        }
        s if s < 0 => {
            // e0 < 0: leading term blows up → limit is ±∞
            tracing::debug!("limitinf: e0 < 0, determining sign of coefficient c0");
            let c0_sign = sign_at_inf(arena, c0, x, depth + 1).unwrap_or(1);
            tracing::info!(c0_sign, "limitinf: e0 < 0 → ω^e0 blows up → limit is ±∞");
            if c0_sign >= 0 {
                Ok(arena.infinity())
            } else {
                Ok(arena.neg_infinity())
            }
        }
        _ => {
            // e0 = 0: limit = lim(c0)
            tracing::debug!("limitinf: e0 = 0 → limit depends on coefficient, recursing on c0");
            limitinf(arena, c0, x, depth + 1)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `lim(z → z0) e` using the Gruntz algorithm.
///
/// Converts any limit to `lim(x → ∞)` via substitution, then applies
/// the core algorithm.
pub(crate) fn gruntz(
    arena: &mut Arena,
    e: ExprId,
    z: ExprId,
    z0: ExprId,
) -> Result<ExprId, crate::errors::SymplexError> {
    tracing::info!("gruntz: entry point — computing limit");

    if z0 == arena.infinity() {
        tracing::debug!("gruntz: limit at +∞, calling limitinf directly");
        return limitinf(arena, e, z, 0);
    }

    if z0 == arena.neg_infinity() {
        tracing::debug!("gruntz: limit at -∞, substituting z = -x and calling limitinf");
        let x = fresh_dummy(arena);
        let neg_x = arena.neg(x);
        let e_sub = crate::subs::subs(arena, e, z, neg_x);
        return limitinf(arena, e_sub, x, 0);
    }

    tracing::debug!("gruntz: finite-point limit, substituting z = z0 + 1/x and calling limitinf");
    // lim(z → z0) f(z) = lim(x → ∞) f(z0 + 1/x)
    let x = fresh_dummy(arena);
    let one = arena.one();
    let inv_x = arena.div(one, x); // 1/x
    let z0_plus_inv_x = arena.add(&[z0, inv_x]); // z0 + 1/x
    let e_sub = crate::subs::subs(arena, e, z, z0_plus_inv_x);
    let e_simplified = crate::eval::eval(arena, e_sub);

    limitinf(arena, e_simplified, x, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if an expression is ±∞.
fn is_infinite(arena: &Arena, e: ExprId) -> bool {
    e == arena.infinity() || e == arena.neg_infinity() || e == arena.complex_infinity()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    // ── Basic limits at infinity ──

    #[test]
    fn gruntz_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let inf = a.infinity();
        let result = gruntz(&mut a, five, x, inf).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn gruntz_x_to_inf_of_1_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.int(1);
        let expr = a.div(one, x);
        let inf = a.infinity();
        let result = gruntz(&mut a, expr, x, inf).unwrap();
        assert_eq!(display(&a, result), "0", "lim(1/x, x→∞) = 0");
    }

    #[test]
    fn gruntz_polynomial_ratio_same_degree() {
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
    fn gruntz_polynomial_ratio_numer_smaller() {
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
    fn gruntz_exp_x_dominates_polynomial() {
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

    // ── Finite-point limits via Gruntz ──

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
    fn gruntz_1_minus_cos_over_x_sq_at_0() {
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
    fn gruntz_exp_minus_1_minus_x_over_x_sq_at_0() {
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
    fn gruntz_x_sq_minus_1_over_x_minus_1_at_1() {
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

    // ── Direct substitution (trivial case) ──

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
    fn gruntz_2x_plus_1_over_x_plus_1_at_inf() {
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
}
