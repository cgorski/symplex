//! Symbolic limit computation — two-sided and one-sided.
//!
//! [`limit`] computes the two-sided limit `lim_{var → point} expr`;
//! [`limit_dir`] additionally takes a [`Direction`] for one-sided limits
//! `x → a⁺` / `x → a⁻`.
//!
//! # Algorithm
//!
//! 1. **Constant:** if `expr` does not depend on `var`, return `eval(expr)`.
//! 2. **Point at ±∞:** the `1^∞` heuristic for `Pow` nodes, then a
//!    compositional rule for an outer unary function (`atan(∞) = π/2`,
//!    `Γ(∞) = ∞`, …), then the Gruntz algorithm, then polynomial-degree
//!    analysis as a last resort.
//! 3. **Finite point:**
//!    a. **Safe direct substitution.** Every `var`-dependent sub-expression is
//!       evaluated at the point bottom-up; if none is singular (`∞`, `zoo`,
//!       `NaN`, `ln 0`, `0^0`, a pole of `Γ`, …) and no discontinuous node
//!       (`sign`, `H`, `⌊·⌋`, `⌈·⌉`, `Piecewise`) sits exactly at its
//!       discontinuity, the substituted value *is* the limit.
//!    b. **Compositional rule** for an outer continuous unary function.
//!    c. **One-sided Gruntz.** `x = a ± 1/w` reduces `x → a^±` to
//!       `w → +∞`, which the Gruntz algorithm handles (with `Abs`, `Sign`,
//!       `Heaviside`, `Floor`, `Ceiling`, `Piecewise`, `Min`, `Max` resolved
//!       by their eventual sign as `w → ∞`). For [`Direction::Both`] both
//!       one-sided limits are computed and must agree.
//!    d. **L'Hôpital + series fallback** for expressions without
//!       discontinuous nodes.
//!
//! Every candidate result is validated: it must be free of `var`, free of
//! internal dummy symbols, free of unevaluated nodes, and must not be `NaN` or
//! complex infinity. `+∞` and `−∞` are legitimate limit values.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};

/// Maximum L'Hôpital iterations to prevent infinite loops.
const MAX_LHOPITAL: usize = 5;

/// Maximum nesting depth for the compositional rule.
const MAX_COMPOSE_DEPTH: usize = 48;

/// Name of the dummy used for the `x = a ± 1/w` substitution.
const LIM_DUMMY: &str = "__lim_w";

/// The direction from which a limit point is approached.
///
/// ```
/// use symplex::prelude::*;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let f = 1 / &x;
/// assert_eq!(format!("{}", f.limit_right(&x, &ctx.int(0))), "oo");
/// assert_eq!(format!("{}", f.limit_left(&x, &ctx.int(0))), "-oo");
/// // The two-sided limit does not exist:
/// assert!(f.limit(&x, &ctx.int(0)).has_unevaluated());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Direction {
    /// Approach from below: `x → a⁻` (`x < a`).
    Left,
    /// Approach from above: `x → a⁺` (`x > a`).
    Right,
    /// Two-sided limit (the default). Both one-sided limits must exist and
    /// agree; otherwise the limit is reported as non-existent.
    #[default]
    Both,
}

fn fail(reason: impl Into<String>) -> SymplexError {
    SymplexError::ComputationFailed {
        operation: "limit",
        reason: reason.into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the two-sided limit of `expr` as `var` approaches `point`.
pub(crate) fn limit(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    limit_dir(arena, expr, var, point, Direction::Both)
}

/// Compute the limit of `expr` as `var` approaches `point` from `dir`.
///
/// Returns `Err` when the limit does not exist (e.g. the one-sided limits
/// differ, or the expression oscillates) or cannot be determined.
pub(crate) fn limit_dir(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
) -> Result<ExprId, SymplexError> {
    if !matches!(arena.node(var), ExprNode::Symbol(_)) {
        return Err(SymplexError::InvalidArgument {
            operation: "limit",
            reason: "the limit variable must be a symbol".into(),
        });
    }
    limit_impl(arena, expr, var, point, dir, 0)
}

fn limit_impl(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    // ── Constant expression ─────────────────────────────────────────
    if !crate::base::walk::contains(arena, expr, var) {
        let v = crate::transforms::eval::eval(arena, expr);
        return if is_valid_limit_value(arena, v, var) {
            Ok(v)
        } else {
            Err(fail(
                "expression is a constant that is not a valid limit value",
            ))
        };
    }

    if point == arena.infinity() || point == arena.neg_infinity() {
        let positive = point == arena.infinity();
        return limit_at_infinity_dir(arena, expr, var, positive, depth);
    }

    if point == arena.nan() || point == arena.complex_infinity() {
        return Err(fail("limit point must be finite or ±∞"));
    }

    // ── Finite point ────────────────────────────────────────────────
    tracing::debug!(dir = ?dir, "limit: finite point");

    // (a) Safe direct substitution.
    if let Some(v) = try_direct_substitution(arena, expr, var, point) {
        tracing::debug!("limit: safe direct substitution succeeded");
        return Ok(v);
    }

    // (b) Compositional rule.
    if depth < MAX_COMPOSE_DEPTH
        && let Some(r) = try_compose(arena, expr, var, point, dir, depth)
    {
        return r;
    }

    // (c) One-sided Gruntz.
    let directional = has_directional_nodes(arena, expr, var);
    match dir {
        Direction::Right | Direction::Left => {
            let right = dir == Direction::Right;
            match one_sided_gruntz(arena, expr, var, point, right) {
                Ok(v) => Ok(v),
                Err(e) => {
                    if directional {
                        return Err(e);
                    }
                    // One-sided fallback: L'Hôpital / series are two-sided
                    // tools, but for expressions built only from analytic
                    // functions a two-sided limit (when it exists) equals
                    // each one-sided limit.
                    analytic_fallback(arena, expr, var, point).map_err(|_| e)
                }
            }
        }
        Direction::Both => {
            let r = one_sided_gruntz(arena, expr, var, point, true);
            let l = one_sided_gruntz(arena, expr, var, point, false);
            match (r, l) {
                (Ok(r), Ok(l)) => {
                    if same_value(arena, r, l) {
                        Ok(r)
                    } else {
                        let rs = arena.display(r).to_string();
                        let ls = arena.display(l).to_string();
                        Err(fail(format!(
                            "left and right limits differ: left = {ls}, right = {rs}"
                        )))
                    }
                }
                (Ok(_), Err(e)) | (Err(e), Ok(_)) | (Err(e), Err(_)) => {
                    if directional {
                        return Err(e);
                    }
                    analytic_fallback(arena, expr, var, point).map_err(|_| e)
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Result validation
// ═══════════════════════════════════════════════════════════════════════════

/// A limit value is acceptable when it is a genuine constant: free of the
/// limit variable and of internal dummies, contains no unevaluated nodes,
/// and is neither `NaN` nor complex infinity. `±∞` on their own are fine,
/// but `∞` buried inside another expression (`exp(oo)`, `atan(oo)`) is not.
pub(crate) fn is_valid_limit_value(arena: &Arena, id: ExprId, var: ExprId) -> bool {
    if id == arena.nan() || id == arena.complex_infinity() {
        return false;
    }
    if id == arena.infinity() || id == arena.neg_infinity() {
        return true;
    }
    if crate::base::walk::contains(arena, id, var) || contains_dummy(arena, id) {
        return false;
    }
    if crate::base::walk::has_unevaluated(arena, id) {
        return false;
    }
    !contains_singular_atom(arena, id) && !contains_hidden_singularity(arena, id)
}

/// Detect symbolic forms that `eval` leaves untouched but that denote
/// poles or undefined values: `ln 0`, `Γ(−n)`, `ψ(−n)`, `(−n)!`, `0^{−k}`,
/// `tan(π/2 + kπ)` is caught separately as `zoo`.
fn contains_hidden_singularity(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        match node {
            ExprNode::Ln(a) | ExprNode::LogGamma(a) if arena.is_zero_structural(*a) => {
                return true;
            }
            ExprNode::Gamma(a) | ExprNode::Digamma(a) if is_nonpositive_integer(arena, *a) => {
                return true;
            }
            ExprNode::Factorial(a) => {
                if let Some(r) = arena.as_num(*a)
                    && r.is_integer()
                    && r.is_negative()
                {
                    return true;
                }
            }
            ExprNode::Pow(b, e) if arena.is_zero_structural(*b) => match arena.as_num(*e) {
                Some(r) if r.is_positive() => {}
                _ => return true,
            },
            _ => {}
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

/// `true` if the expression contains `∞`, `−∞`, `zoo`, or `NaN` anywhere.
fn contains_singular_atom(arena: &Arena, id: ExprId) -> bool {
    crate::base::walk::contains(arena, id, arena.infinity())
        || crate::base::walk::contains(arena, id, arena.neg_infinity())
        || crate::base::walk::contains(arena, id, arena.complex_infinity())
        || crate::base::walk::contains(arena, id, arena.nan())
}

/// `true` if `id` is `±∞`, `zoo`, or `NaN`, or contains one of them.
fn is_singular(arena: &Arena, id: ExprId) -> bool {
    id == arena.infinity()
        || id == arena.neg_infinity()
        || id == arena.complex_infinity()
        || id == arena.nan()
        || contains_singular_atom(arena, id)
}

/// `true` if the expression contains an internal dummy symbol (`__gw…`,
/// `__lim…`, `__limit_t`).
pub(crate) fn contains_dummy(arena: &Arena, root: ExprId) -> bool {
    let mut stack = vec![root];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if let ExprNode::Symbol(sid) = arena.node(id) {
            let name = arena.symbol_name(*sid);
            if name.starts_with("__gw") || name.starts_with("__lim") {
                return true;
            }
        }
        arena.node(id).for_each_child(|c| stack.push(c));
    }
    false
}

/// Structural check whether two computed limit values agree.
fn same_value(arena: &mut Arena, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    let a_inf = a == arena.infinity() || a == arena.neg_infinity();
    let b_inf = b == arena.infinity() || b == arena.neg_infinity();
    if a_inf || b_inf {
        return false;
    }
    if let (Some(ra), Some(rb)) = (arena.as_num(a), arena.as_num(b)) {
        return ra == rb;
    }
    let diff = arena.sub(a, b);
    let diff = crate::transforms::eval::eval(arena, diff);
    if arena.is_zero_structural(diff) {
        return true;
    }
    let simplified = crate::transforms::expand::expand(arena, diff);
    let simplified = crate::transforms::eval::eval(arena, simplified);
    arena.is_zero_structural(simplified)
}

// ═══════════════════════════════════════════════════════════════════════════
// Directional (discontinuous) node detection
// ═══════════════════════════════════════════════════════════════════════════

/// `true` if `expr` contains a node whose value or derivative is
/// discontinuous and whose argument depends on `var` (`Abs`, `Sign`,
/// `Heaviside`, `DiracDelta`, `Floor`, `Ceiling`, `Piecewise`, `Min`, `Max`).
///
/// For such expressions the two-sided L'Hôpital / series fallbacks are not
/// trustworthy, so only the one-sided Gruntz path is used.
pub(crate) fn has_directional_nodes(arena: &Arena, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];
    let mut visited = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let node = arena.node(id);
        let directional = matches!(
            node,
            ExprNode::Abs(_)
                | ExprNode::Sign(_)
                | ExprNode::Heaviside(_)
                | ExprNode::DiracDelta(_)
                | ExprNode::Floor(_)
                | ExprNode::Ceiling(_)
                | ExprNode::Piecewise(_)
                | ExprNode::Min(_)
                | ExprNode::Max(_)
        );
        if directional && crate::base::walk::contains(arena, id, var) {
            return true;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// (a) Safe direct substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Substitute `var = point` and evaluate, but only accept the result when the
/// substitution is provably continuous: every `var`-dependent sub-expression
/// is evaluated at the point and must be finite, and discontinuous functions
/// must not be evaluated exactly at their discontinuity.
///
/// This guards against canonicalization folding indeterminate forms
/// (e.g. `0 · 0⁻¹ → 0`) into plausible-looking finite numbers.
fn try_direct_substitution(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let post = crate::base::walk::post_order_ids(arena, expr);
    let mut values: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post {
        if !crate::base::walk::contains(arena, id, var) {
            continue;
        }
        let node = arena.node(id).clone();

        // Value of a child at the point (constants are their own value).
        let val_of = |arena: &mut Arena, values: &FxHashMap<ExprId, ExprId>, c: ExprId| {
            if let Some(&v) = values.get(&c) {
                v
            } else {
                crate::transforms::eval::eval(arena, c)
            }
        };

        // Node-specific continuity checks on the *argument* values.
        match node {
            ExprNode::Piecewise(_) => return None,
            ExprNode::Ln(a) | ExprNode::LogGamma(a) => {
                let av = val_of(arena, &values, a);
                if arena.is_zero_structural(av) || !is_definite_constant(arena, av) {
                    return None;
                }
            }
            ExprNode::Gamma(a) | ExprNode::Digamma(a) => {
                let av = val_of(arena, &values, a);
                if is_nonpositive_integer(arena, av) {
                    return None;
                }
            }
            ExprNode::Factorial(a) => {
                let av = val_of(arena, &values, a);
                if let Some(r) = arena.as_num(av)
                    && r.is_integer()
                    && r.is_negative()
                {
                    return None;
                }
            }
            ExprNode::Tan(a) => {
                let av = val_of(arena, &values, a);
                let c = arena.cos(av);
                let c = crate::transforms::eval::eval(arena, c);
                if arena.is_zero_structural(c) {
                    return None;
                }
            }
            ExprNode::Pow(b, e) => {
                let bv = val_of(arena, &values, b);
                let ev = val_of(arena, &values, e);
                if arena.is_zero_structural(bv) {
                    // 0^e is continuous only for a definite positive exponent.
                    match arena.as_num(ev) {
                        Some(r) if r.is_positive() => {}
                        _ => return None,
                    }
                }
                // 1^∞ / ∞^0 forms are caught by the singular check below
                // because the offending sub-expression evaluates to ∞.
            }
            ExprNode::Sign(a) | ExprNode::Heaviside(a) | ExprNode::DiracDelta(a) => {
                let av = val_of(arena, &values, a);
                match arena.as_num(av) {
                    Some(r) if !r.is_zero() => {}
                    _ => return None,
                }
            }
            ExprNode::Floor(a) | ExprNode::Ceiling(a) => {
                let av = val_of(arena, &values, a);
                match arena.as_num(av) {
                    Some(r) if !r.is_integer() => {}
                    _ => return None,
                }
            }
            ExprNode::Atanh(a) => {
                let av = val_of(arena, &values, a);
                if let Some(r) = arena.as_num(av)
                    && r.abs() == Ratio::from_integer(BigInt::from(1))
                {
                    return None;
                }
            }
            _ => {}
        }

        let subst = crate::transforms::subs::subs(arena, id, var, point);
        let v = crate::transforms::eval::eval(arena, subst);
        if is_singular(arena, v) {
            return None;
        }
        values.insert(id, v);
    }

    let v = *values.get(&expr)?;
    if is_valid_limit_value(arena, v, var) && v != arena.infinity() && v != arena.neg_infinity() {
        Some(v)
    } else {
        None
    }
}

/// A "definite" constant: numeric, or a symbolic constant with a known sign.
fn is_definite_constant(arena: &mut Arena, id: ExprId) -> bool {
    arena.as_num(id).is_some() || const_sign(arena, id).is_some()
}

fn is_nonpositive_integer(arena: &Arena, id: ExprId) -> bool {
    match arena.as_num(id) {
        Some(r) => r.is_integer() && !r.is_positive(),
        None => false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sign of constants (with assumptions)
// ═══════════════════════════════════════════════════════════════════════════

/// Sign of a constant expression: `Some(1)`, `Some(-1)`, `Some(0)`, or
/// `None` if unknown. Consults symbol assumptions (`a > 0`).
pub(crate) fn const_sign(arena: &mut Arena, e: ExprId) -> Option<i32> {
    const_sign_depth(arena, e, 0)
}

fn const_sign_depth(arena: &mut Arena, e: ExprId, depth: usize) -> Option<i32> {
    if depth > 16 {
        return None;
    }
    if let Some(r) = arena.as_num(e) {
        return Some(sign_of_ratio(r));
    }
    if e == arena.pi() || e == arena.e_const() || e == arena.infinity() {
        return Some(1);
    }
    if e == arena.neg_infinity() {
        return Some(-1);
    }
    if e == arena.nan() || e == arena.complex_infinity() || e == arena.i_unit() {
        return None;
    }
    let evaled = crate::transforms::eval::eval(arena, e);
    if evaled != e
        && let Some(r) = arena.as_num(evaled)
    {
        return Some(sign_of_ratio(r));
    }
    match arena.node(e).clone() {
        ExprNode::Neg(inner) => const_sign_depth(arena, inner, depth + 1).map(|s| -s),
        ExprNode::Mul(children) => {
            let mut result = 1;
            for c in children {
                let s = const_sign_depth(arena, c, depth + 1)?;
                if s == 0 {
                    return Some(0);
                }
                result *= s;
            }
            Some(result)
        }
        ExprNode::Add(children) => {
            // All terms share a sign → that sign.
            let mut sign = 0;
            for c in children {
                let s = const_sign_depth(arena, c, depth + 1)?;
                if s == 0 {
                    continue;
                }
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return assumption_sign(arena, e);
                }
            }
            Some(sign)
        }
        ExprNode::Exp(_) | ExprNode::Cosh(_) => Some(1),
        ExprNode::Abs(inner) => {
            let s = const_sign_depth(arena, inner, depth + 1)?;
            Some(if s == 0 { 0 } else { 1 })
        }
        ExprNode::Pow(base, exp) => {
            let bs = const_sign_depth(arena, base, depth + 1)?;
            if let Some(r) = arena.as_num(exp).cloned() {
                if r.is_integer() {
                    let odd = (r.to_integer() % BigInt::from(2)) != BigInt::zero();
                    if bs == 0 {
                        return if r.is_positive() { Some(0) } else { None };
                    }
                    return Some(if odd { bs } else { 1 });
                }
                if bs > 0 {
                    return Some(1);
                }
                if bs == 0 && r.is_positive() {
                    return Some(0);
                }
                return None;
            }
            if bs > 0 { Some(1) } else { None }
        }
        ExprNode::Ln(inner) => {
            let r = arena.as_num(inner).cloned()?;
            if !r.is_positive() {
                return None;
            }
            let one = Ratio::from_integer(BigInt::from(1));
            Some(if r > one {
                1
            } else if r == one {
                0
            } else {
                -1
            })
        }
        _ => assumption_sign(arena, e),
    }
}

fn sign_of_ratio(r: &Ratio<BigInt>) -> i32 {
    if r.is_positive() {
        1
    } else if r.is_negative() {
        -1
    } else {
        0
    }
}

/// Sign of a symbolic constant via the assumption system.
fn assumption_sign(arena: &Arena, e: ExprId) -> Option<i32> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let mut cache = AssumptionCache::new();
    if cache.query(arena, e, Props::POSITIVE) == Some(true) {
        return Some(1);
    }
    if cache.query(arena, e, Props::NEGATIVE) == Some(true) {
        return Some(-1);
    }
    if cache.query(arena, e, Props::ZERO) == Some(true) {
        return Some(0);
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// (b) Compositional rule
// ═══════════════════════════════════════════════════════════════════════════

/// Extended-real classification of a computed limit value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ext {
    Finite,
    PosInf,
    NegInf,
}

fn classify(arena: &Arena, v: ExprId) -> Ext {
    if v == arena.infinity() {
        Ext::PosInf
    } else if v == arena.neg_infinity() {
        Ext::NegInf
    } else {
        Ext::Finite
    }
}

/// Wrap a finite candidate: evaluate and validate.
fn finite_candidate(arena: &mut Arena, v: ExprId, var: ExprId) -> Option<ExprId> {
    let v = crate::transforms::eval::eval(arena, v);
    if is_valid_limit_value(arena, v, var) && classify(arena, v) == Ext::Finite {
        Some(v)
    } else {
        None
    }
}

/// `lim F(g(x)) = F(lim g(x))` for an outer function `F` that is continuous
/// at the inner limit, extended to `±∞` via a table of known asymptotic
/// values. Returns `None` when the rule does not apply (the caller then
/// falls through to the Gruntz algorithm); returns `Some(Err(_))` when the
/// rule proves the limit does not exist.
fn try_compose(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    dir: Direction,
    depth: usize,
) -> Option<Result<ExprId, SymplexError>> {
    let node = arena.node(expr).clone();

    // Helper: limit of a sub-expression with the same direction.
    let inner_limit = |arena: &mut Arena, inner: ExprId| -> Option<ExprId> {
        limit_impl(arena, inner, var, point, dir, depth + 1).ok()
    };

    match node {
        // ── Linear combinations with a single var-dependent part ──
        ExprNode::Add(ref children) => {
            let (consts, deps) = split_by_var(arena, children, var);
            if deps.len() != 1 {
                return None;
            }
            let c = if consts.is_empty() {
                return None;
            } else {
                arena.add(&consts)
            };
            let l = inner_limit(arena, deps[0])?;
            match classify(arena, l) {
                Ext::Finite => {
                    let s = arena.add(&[c, l]);
                    finite_candidate(arena, s, var).map(Ok)
                }
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => Some(Ok(arena.neg_infinity())),
            }
        }
        ExprNode::Mul(ref children) => {
            let (consts, deps) = split_by_var(arena, children, var);
            if deps.len() != 1 || consts.is_empty() {
                return None;
            }
            let c = arena.mul(&consts);
            let l = inner_limit(arena, deps[0])?;
            match classify(arena, l) {
                Ext::Finite => {
                    let p = arena.mul(&[c, l]);
                    finite_candidate(arena, p, var).map(Ok)
                }
                inf => {
                    let cs = const_sign(arena, c)?;
                    if cs == 0 {
                        return None;
                    }
                    let positive = (inf == Ext::PosInf) == (cs > 0);
                    Some(Ok(if positive {
                        arena.infinity()
                    } else {
                        arena.neg_infinity()
                    }))
                }
            }
        }
        ExprNode::Neg(inner) => {
            let l = inner_limit(arena, inner)?;
            match classify(arena, l) {
                Ext::Finite => {
                    let n = arena.neg(l);
                    finite_candidate(arena, n, var).map(Ok)
                }
                Ext::PosInf => Some(Ok(arena.neg_infinity())),
                Ext::NegInf => Some(Ok(arena.infinity())),
            }
        }

        // ── Powers with a constant exponent ──
        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, exp, var) => {
            let k = arena.as_num(exp).cloned();
            let l = inner_limit(arena, base)?;
            match classify(arena, l) {
                Ext::Finite => {
                    if arena.is_zero_structural(l) {
                        // 0^k: continuous only for definite positive k.
                        return match &k {
                            Some(r) if r.is_positive() => Some(Ok(arena.zero())),
                            _ => None,
                        };
                    }
                    if const_sign(arena, l).is_none() && k.as_ref().is_none_or(|r| !r.is_integer())
                    {
                        // Symbolic base of unknown sign with a non-integer
                        // exponent — could be complex; leave to Gruntz.
                        return None;
                    }
                    let p = arena.pow(l, exp);
                    finite_candidate(arena, p, var).map(Ok)
                }
                Ext::PosInf => {
                    let r = k?;
                    Some(Ok(if r.is_positive() {
                        arena.infinity()
                    } else if r.is_zero() {
                        arena.one()
                    } else {
                        arena.zero()
                    }))
                }
                Ext::NegInf => {
                    let r = k?;
                    if r.is_zero() {
                        return Some(Ok(arena.one()));
                    }
                    if r.is_negative() {
                        return Some(Ok(arena.zero()));
                    }
                    if !r.is_integer() {
                        return None;
                    }
                    let odd = (r.to_integer() % BigInt::from(2)) != BigInt::zero();
                    Some(Ok(if odd {
                        arena.neg_infinity()
                    } else {
                        arena.infinity()
                    }))
                }
            }
        }

        // ── Constant base, variable exponent: c^g ──
        ExprNode::Pow(base, exp) if !crate::base::walk::contains(arena, base, var) => {
            let c = arena.as_num(base).cloned()?;
            if !c.is_positive() {
                return None;
            }
            let l = inner_limit(arena, exp)?;
            let one = Ratio::from_integer(BigInt::from(1));
            match classify(arena, l) {
                Ext::Finite => {
                    let p = arena.pow(base, l);
                    finite_candidate(arena, p, var).map(Ok)
                }
                inf => {
                    if c == one {
                        return Some(Ok(arena.one()));
                    }
                    let grows = (c > one) == (inf == Ext::PosInf);
                    Some(Ok(if grows {
                        arena.infinity()
                    } else {
                        arena.zero()
                    }))
                }
            }
        }

        // ── Min / Max: continuous, extended to ±∞ ──
        ExprNode::Min(ref args) | ExprNode::Max(ref args) => {
            let is_min = matches!(node, ExprNode::Min(_));
            let args = args.clone();
            let mut limits = Vec::with_capacity(args.len());
            for &a in &args {
                limits.push(inner_limit(arena, a)?);
            }
            // Exact if all are numbers or infinities.
            let mut best: Option<(ExprId, Option<Ratio<BigInt>>, Ext)> = None;
            let mut all_ordered = true;
            for &l in &limits {
                let cls = classify(arena, l);
                let num = arena.as_num(l).cloned();
                if cls == Ext::Finite && num.is_none() {
                    all_ordered = false;
                    break;
                }
                let better = match &best {
                    None => true,
                    Some((_, bnum, bcls)) => {
                        let ord = ext_cmp(&num, cls, bnum, *bcls);
                        if is_min {
                            ord == std::cmp::Ordering::Less
                        } else {
                            ord == std::cmp::Ordering::Greater
                        }
                    }
                };
                if better {
                    best = Some((l, num, cls));
                }
            }
            if all_ordered {
                return best.map(|(l, _, _)| Ok(l));
            }
            // Symbolic finite limits: rebuild the node and evaluate.
            if limits.iter().any(|&l| classify(arena, l) != Ext::Finite) {
                return None;
            }
            let sv: smallvec::SmallVec<[ExprId; 4]> = limits.iter().copied().collect();
            let rebuilt = if is_min {
                arena.intern(ExprNode::Min(sv))
            } else {
                arena.intern(ExprNode::Max(sv))
            };
            finite_candidate(arena, rebuilt, var).map(Ok)
        }

        // ── Unary functions ──
        ExprNode::Exp(a) => {
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::Finite => {
                    let v = arena.exp(l);
                    finite_candidate(arena, v, var).map(Ok)
                }
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => Some(Ok(arena.zero())),
            }
        }
        ExprNode::Ln(a) => {
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::Finite => {
                    if arena.is_zero_structural(l) {
                        return None; // ln 0: one-sided behaviour, leave to Gruntz
                    }
                    match const_sign(arena, l) {
                        Some(s) if s > 0 => {
                            let v = arena.ln(l);
                            finite_candidate(arena, v, var).map(Ok)
                        }
                        _ => None,
                    }
                }
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => None,
            }
        }
        ExprNode::Atan(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| {
                let two = arena.int(2);
                let pi = arena.pi();
                let half_pi = arena.div(pi, two);
                Some(if side { half_pi } else { arena.neg(half_pi) })
            },
            |arena, l| arena.atan(l),
        ),
        ExprNode::Erf(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| Some(if side { arena.one() } else { arena.neg_one() }),
            |arena, l| arena.erf(l),
        ),
        ExprNode::Erfc(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| Some(if side { arena.zero() } else { arena.int(2) }),
            |arena, l| arena.erfc(l),
        ),
        ExprNode::Tanh(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| Some(if side { arena.one() } else { arena.neg_one() }),
            |arena, l| arena.tanh(l),
        ),
        ExprNode::Sinh(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| {
                Some(if side {
                    arena.infinity()
                } else {
                    arena.neg_infinity()
                })
            },
            |arena, l| arena.sinh(l),
        ),
        ExprNode::Cosh(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, _| Some(arena.infinity()),
            |arena, l| arena.cosh(l),
        ),
        ExprNode::Asinh(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| {
                Some(if side {
                    arena.infinity()
                } else {
                    arena.neg_infinity()
                })
            },
            |arena, l| arena.asinh(l),
        ),
        ExprNode::Acosh(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, side| if side { Some(arena.infinity()) } else { None },
            |arena, l| arena.acosh(l),
        ),
        ExprNode::Abs(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |arena, _| Some(arena.infinity()),
            |arena, l| arena.abs(l),
        ),
        ExprNode::Sin(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |_, _| None,
            |arena, l| arena.sin(l),
        ),
        ExprNode::Cos(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |_, _| None,
            |arena, l| arena.cos(l),
        ),
        ExprNode::Asin(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |_, _| None,
            |arena, l| arena.asin(l),
        ),
        ExprNode::Acos(a) => unary_with_asymptotes(
            arena,
            a,
            var,
            &inner_limit,
            |_, _| None,
            |arena, l| arena.acos(l),
        ),
        ExprNode::Gamma(a) => {
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => Some(Err(fail("Γ(x) has no limit as x → −∞"))),
                Ext::Finite => {
                    if is_nonpositive_integer(arena, l) {
                        return None;
                    }
                    let v = arena.gamma(l);
                    finite_candidate(arena, v, var).map(Ok)
                }
            }
        }
        ExprNode::LogGamma(a) | ExprNode::Digamma(a) | ExprNode::LambertW(a) => {
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => None,
                Ext::Finite => {
                    if matches!(node, ExprNode::LogGamma(_) | ExprNode::Digamma(_))
                        && is_nonpositive_integer(arena, l)
                    {
                        return None;
                    }
                    let v = match node {
                        ExprNode::LogGamma(_) => arena.log_gamma(l),
                        ExprNode::Digamma(_) => arena.digamma(l),
                        _ => arena.lambertw(l),
                    };
                    finite_candidate(arena, v, var).map(Ok)
                }
            }
        }
        ExprNode::Factorial(a) => {
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => None,
                Ext::Finite => {
                    let v = arena.factorial(l);
                    finite_candidate(arena, v, var).map(Ok)
                }
            }
        }
        ExprNode::Sign(a) | ExprNode::Heaviside(a) => {
            let is_sign = matches!(node, ExprNode::Sign(_));
            let l = inner_limit(arena, a)?;
            let s = match classify(arena, l) {
                Ext::PosInf => 1,
                Ext::NegInf => -1,
                Ext::Finite => match arena.as_num(l) {
                    Some(r) if !r.is_zero() => sign_of_ratio(r),
                    _ => return None,
                },
            };
            Some(Ok(if is_sign {
                arena.int(s as i64)
            } else if s > 0 {
                arena.one()
            } else {
                arena.zero()
            }))
        }
        ExprNode::Floor(a) | ExprNode::Ceiling(a) => {
            let is_floor = matches!(node, ExprNode::Floor(_));
            let l = inner_limit(arena, a)?;
            match classify(arena, l) {
                Ext::PosInf => Some(Ok(arena.infinity())),
                Ext::NegInf => Some(Ok(arena.neg_infinity())),
                Ext::Finite => {
                    let r = arena.as_num(l).cloned()?;
                    if r.is_integer() {
                        return None;
                    }
                    let v = if is_floor { r.floor() } else { r.ceil() };
                    let nid = arena.intern_num(v);
                    Some(Ok(arena.intern(ExprNode::Num(nid))))
                }
            }
        }
        _ => None,
    }
}

/// Compare two extended-real values (numbers or `±∞`).
fn ext_cmp(
    a: &Option<Ratio<BigInt>>,
    a_cls: Ext,
    b: &Option<Ratio<BigInt>>,
    b_cls: Ext,
) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let rank = |c: Ext| match c {
        Ext::NegInf => 0,
        Ext::Finite => 1,
        Ext::PosInf => 2,
    };
    match rank(a_cls).cmp(&rank(b_cls)) {
        Equal if a_cls == Ext::Finite => match (a, b) {
            (Some(x), Some(y)) => x.cmp(y),
            _ => Equal,
        },
        other => other,
    }
}

/// Split n-ary children into (`var`-free, `var`-dependent).
fn split_by_var(arena: &Arena, children: &[ExprId], var: ExprId) -> (Vec<ExprId>, Vec<ExprId>) {
    let mut consts = Vec::new();
    let mut deps = Vec::new();
    for &c in children {
        if crate::base::walk::contains(arena, c, var) {
            deps.push(c);
        } else {
            consts.push(c);
        }
    }
    (consts, deps)
}

/// Compositional rule for a unary function `F` that is continuous on its
/// domain, with `at_inf(arena, positive)` giving `lim_{t→±∞} F(t)` (or
/// `None` when no such limit exists) and `build` constructing `F(l)`.
fn unary_with_asymptotes(
    arena: &mut Arena,
    inner: ExprId,
    var: ExprId,
    inner_limit: &dyn Fn(&mut Arena, ExprId) -> Option<ExprId>,
    at_inf: impl Fn(&mut Arena, bool) -> Option<ExprId>,
    build: impl Fn(&mut Arena, ExprId) -> ExprId,
) -> Option<Result<ExprId, SymplexError>> {
    let l = inner_limit(arena, inner)?;
    match classify(arena, l) {
        Ext::Finite => {
            let v = build(arena, l);
            finite_candidate(arena, v, var).map(Ok)
        }
        Ext::PosInf => match at_inf(arena, true) {
            Some(v) => Some(Ok(v)),
            None => Some(Err(fail(
                "the function oscillates or is undefined as its argument diverges",
            ))),
        },
        Ext::NegInf => match at_inf(arena, false) {
            Some(v) => Some(Ok(v)),
            None => Some(Err(fail(
                "the function oscillates or is undefined as its argument diverges",
            ))),
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// (c) One-sided limits via Gruntz
// ═══════════════════════════════════════════════════════════════════════════

/// `lim_{x → a^±} f(x)` via the substitution `x = a ± 1/w`, `w → +∞`.
fn one_sided_gruntz(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    right: bool,
) -> Result<ExprId, SymplexError> {
    let w = arena.symbol(LIM_DUMMY);
    let one = arena.one();
    let inv_w = arena.div(one, w);
    let shift = if right { inv_w } else { arena.neg(inv_w) };
    let sub_point = arena.add(&[point, shift]);
    let e = crate::transforms::subs::subs(arena, expr, var, sub_point);

    // Clear nested fractions and simplify (same preprocessing as `gruntz`).
    let e = crate::poly::polybridge::together(arena, e);
    let e = crate::poly::polybridge::cancel(arena, e, w);
    let e = crate::transforms::eval::eval(arena, e);
    let e = crate::transforms::expand::expand(arena, e);
    let e = crate::transforms::eval::eval(arena, e);

    let side = if right { "+" } else { "-" };
    tracing::debug!(side, expr = %arena.display(e).to_string(), "limit: one-sided Gruntz");

    let r = crate::calculus::gruntz::limit_pos_inf(arena, e, w)?;
    if is_valid_limit_value(arena, r, var) && !crate::base::walk::contains(arena, r, w) {
        Ok(r)
    } else {
        Err(fail(format!(
            "could not determine the limit from the {side} side"
        )))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// (d) L'Hôpital + series fallback (analytic expressions only)
// ═══════════════════════════════════════════════════════════════════════════

fn analytic_fallback(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Result<ExprId, SymplexError> {
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if denom != arena.one()
        && let Some(r) = try_lhopital(arena, numer, denom, var, point, 0)
        && is_valid_limit_value(arena, r, var)
    {
        return Ok(r);
    }

    Err(fail(
        "could not determine the limit via substitution, the Gruntz algorithm, or L'Hôpital's rule",
    ))
}

/// Try L'Hôpital's rule: lim f/g = lim f'/g' when f(a)=g(a)=0 or both →∞.
fn try_lhopital(
    arena: &mut Arena,
    numer: ExprId,
    denom: ExprId,
    var: ExprId,
    point: ExprId,
    depth: usize,
) -> Option<ExprId> {
    if depth >= MAX_LHOPITAL {
        return None;
    }

    // Evaluate numerator and denominator at the point *separately*
    // to avoid premature cancellation in Mul canonicalization.
    let n_subst = crate::transforms::subs::subs(arena, numer, var, point);
    let n_at_point = crate::transforms::eval::eval(arena, n_subst);
    let d_subst = crate::transforms::subs::subs(arena, denom, var, point);
    let d_at_point = crate::transforms::eval::eval(arena, d_subst);

    let n_is_zero = arena.is_zero_structural(n_at_point);
    let d_is_zero = arena.is_zero_structural(d_at_point);

    let n_is_inf = n_at_point == arena.infinity() || n_at_point == arena.neg_infinity();
    let d_is_inf = d_at_point == arena.infinity() || d_at_point == arena.neg_infinity();

    // Only apply L'Hôpital to genuine indeterminate forms: 0/0 or ∞/∞.
    if !((n_is_zero && d_is_zero) || (n_is_inf && d_is_inf)) {
        return None;
    }

    tracing::debug!(depth, "L'Hôpital: indeterminate form detected");

    let n_prime = crate::transforms::diff::diff(arena, numer, var);
    let d_prime = crate::transforms::diff::diff(arena, denom, var);

    // Evaluate the differentiated num/denom SEPARATELY first so we can
    // detect another 0/0 indeterminate form.
    let np_subst = crate::transforms::subs::subs(arena, n_prime, var, point);
    let np_val = crate::transforms::eval::eval(arena, np_subst);
    let dp_subst = crate::transforms::subs::subs(arena, d_prime, var, point);
    let dp_val = crate::transforms::eval::eval(arena, dp_subst);

    let np_zero = arena.is_zero_structural(np_val);
    let dp_zero = arena.is_zero_structural(dp_val);

    if np_zero && dp_zero {
        return try_lhopital(arena, n_prime, d_prime, var, point, depth + 1);
    }

    if is_finite_constant(arena, np_val, var) && is_finite_constant(arena, dp_val, var) && !dp_zero
    {
        let result_ratio = arena.div(np_val, dp_val);
        let result = crate::transforms::eval::eval(arena, result_ratio);
        if is_finite_constant(arena, result, var) {
            return Some(result);
        }
    }

    // Combined ratio as a fallback.
    let ratio = arena.div(n_prime, d_prime);
    let subst = crate::transforms::subs::subs(arena, ratio, var, point);
    let evaled = crate::transforms::eval::eval(arena, subst);
    if is_finite_constant(arena, evaled, var) {
        return Some(evaled);
    }

    try_lhopital(arena, n_prime, d_prime, var, point, depth + 1)
}

/// A valid, finite limit value (excludes `±∞`).
fn is_finite_constant(arena: &Arena, id: ExprId, var: ExprId) -> bool {
    is_valid_limit_value(arena, id, var) && id != arena.infinity() && id != arena.neg_infinity()
}

// ═══════════════════════════════════════════════════════════════════════════
// Limits at ±∞
// ═══════════════════════════════════════════════════════════════════════════

fn limit_at_infinity_dir(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    positive: bool,
    depth: usize,
) -> Result<ExprId, SymplexError> {
    let point = if positive {
        arena.infinity()
    } else {
        arena.neg_infinity()
    };

    // ── 1^∞ heuristic for Pow expressions ──────────────────────
    //
    // When expr = base^exp and the exponent depends on var, try the
    // rewrite  b^e → exp(e · (b − 1)).  This is mathematically exact
    // in the limit when b → 1:
    //
    //   b^e = exp(e·ln(b)) = exp(e·ln(1 + (b−1))) ≈ exp(e·(b−1))
    //
    // If the inner product e·(b−1) converges to a finite value L,
    // the answer is exp(L).  If it diverges or fails, we fall through
    // to the normal Gruntz path.
    if let ExprNode::Pow(base, exponent) = arena.node(expr).clone()
        && crate::base::walk::contains(arena, exponent, var)
    {
        let base_tends_to_one = if !crate::base::walk::contains(arena, base, var) {
            base == arena.one()
        } else {
            match crate::calculus::gruntz::gruntz(arena, base, var, point) {
                Ok(lim_base) => lim_base == arena.one(),
                Err(_) => false,
            }
        };

        if base_tends_to_one {
            tracing::debug!("limit: base → 1 confirmed, applying 1^∞ heuristic");
            let one = arena.one();
            let base_minus_1 = arena.sub(base, one);
            let product = arena.mul(&[exponent, base_minus_1]);
            if let Ok(inner_lim) = crate::calculus::gruntz::gruntz(arena, product, var, point)
                && is_finite_constant(arena, inner_lim, var)
            {
                let result = arena.exp(inner_lim);
                let result = crate::transforms::eval::eval(arena, result);
                if is_valid_limit_value(arena, result, var) {
                    return Ok(result);
                }
            }
        }
    }

    // ── Compositional rule ──
    if depth < MAX_COMPOSE_DEPTH
        && let Some(r) = try_compose(arena, expr, var, point, Direction::Both, depth)
    {
        return r;
    }

    // ── Gruntz ──
    tracing::debug!("limit: at infinity, trying Gruntz algorithm");
    let gruntz_err = match crate::calculus::gruntz::gruntz(arena, expr, var, point) {
        Ok(result) if is_valid_limit_value(arena, result, var) => return Ok(result),
        Ok(_) => fail("Gruntz algorithm produced an indeterminate result"),
        Err(e) => e,
    };

    // ── Polynomial-degree fallback ──
    tracing::debug!("limit: Gruntz failed, falling back to polynomial degree analysis");
    match limit_at_infinity(arena, expr, var, positive) {
        Ok(r) if is_valid_limit_value(arena, r, var) => Ok(r),
        _ => Err(gruntz_err),
    }
}

/// Compute lim(x→∞) expr or lim(x→-∞) expr by polynomial-degree analysis.
///
/// Strategies:
/// 0. Direct polynomial degree comparison on the original expression
/// 1. Substitution x = 1/t, together+cancel to clear nested fractions, lim(t→0)
/// 2. Polynomial degree analysis on the substituted form
/// 3. L'Hôpital on the substituted form
pub(crate) fn limit_at_infinity(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    positive: bool, // true for +∞, false for -∞
) -> Result<ExprId, SymplexError> {
    // ── Strategy 0: Direct polynomial degree comparison on original expr ──
    let (orig_numer, orig_denom) = crate::poly::polybridge::as_numer_denom(arena, expr);
    if orig_denom != arena.one() {
        let n_deg = crate::poly::polybridge::poly_degree(arena, orig_numer, var);
        let d_deg = crate::poly::polybridge::poly_degree(arena, orig_denom, var);

        if let (Some(nd), Some(dd)) = (n_deg, d_deg) {
            if nd < dd {
                return Ok(arena.zero());
            }
            if nd == dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_numer, var);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_denom, var);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                {
                    let ratio = arena.div(*n_lead, *d_lead);
                    let result = crate::transforms::eval::eval(arena, ratio);
                    if is_finite_result(arena, result) {
                        return Ok(result);
                    }
                }
            }
            if nd > dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_numer, var);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, orig_denom, var);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                    && let (Some(nr), Some(dr)) = (arena.as_num(*n_lead), arena.as_num(*d_lead))
                {
                    let ratio_positive = nr.is_positive() == dr.is_positive();
                    // For x → -∞, an odd degree difference flips the sign
                    let flip = !positive && ((nd - dd) % 2 == 1);
                    let result_positive = ratio_positive ^ flip;
                    if result_positive {
                        return Ok(arena.infinity());
                    } else {
                        return Ok(arena.neg_infinity());
                    }
                }
            }
        }
    } else {
        let deg = crate::poly::polybridge::poly_degree(arena, expr, var);
        if let Some(d) = deg
            && d == 0
        {
            let result = crate::transforms::eval::eval(arena, expr);
            if is_finite_result(arena, result) {
                return Ok(result);
            }
        }
    }

    // ── Strategy 1: Substitution x = 1/t, then together+cancel ──
    let t = arena.symbol("__limit_t");
    let one = arena.one();
    let t_inv = arena.div(one, t);

    let sub_expr = if positive {
        arena.subs_structural(expr, var, t_inv)
    } else {
        let neg_t_inv = arena.neg(t_inv);
        arena.subs_structural(expr, var, neg_t_inv)
    };

    let together = crate::poly::polybridge::together(arena, sub_expr);
    let cancelled = crate::poly::polybridge::cancel(arena, together, t);
    let simplified = crate::transforms::eval::eval(arena, cancelled);
    let expanded = crate::transforms::expand::expand(arena, simplified);
    let evaled = crate::transforms::eval::eval(arena, expanded);

    let zero = arena.zero();
    let at_zero = arena.subs_structural(evaled, t, zero);
    let at_zero_eval = crate::transforms::eval::eval(arena, at_zero);

    if is_finite_result(arena, at_zero_eval) {
        return Ok(at_zero_eval);
    }

    // ── Strategy 2: Polynomial degree analysis on the substituted form ──
    let (numer, denom) = crate::poly::polybridge::as_numer_denom(arena, evaled);
    if denom != arena.one() {
        let n_deg = crate::poly::polybridge::poly_degree(arena, numer, t);
        let d_deg = crate::poly::polybridge::poly_degree(arena, denom, t);

        if let (Some(nd), Some(dd)) = (n_deg, d_deg) {
            if nd < dd {
                return Ok(arena.zero());
            }
            if nd == dd {
                let cancelled = crate::poly::polybridge::cancel(arena, evaled, t);
                let result = arena.subs_structural(cancelled, t, zero);
                let result = crate::transforms::eval::eval(arena, result);
                if is_finite_result(arena, result) {
                    return Ok(result);
                }
            }
            if nd > dd {
                let n_coeffs = crate::poly::polybridge::poly_coefficients(arena, numer, t);
                let d_coeffs = crate::poly::polybridge::poly_coefficients(arena, denom, t);
                if let (Some(nc), Some(dc)) = (n_coeffs, d_coeffs)
                    && let (Some(n_lead), Some(d_lead)) = (nc.last(), dc.last())
                    && let (Some(nr), Some(dr)) = (arena.as_num(*n_lead), arena.as_num(*d_lead))
                {
                    let ratio_positive = nr.is_positive() == dr.is_positive();
                    if ratio_positive {
                        return Ok(arena.infinity());
                    } else {
                        return Ok(arena.neg_infinity());
                    }
                }
            }
        }
    }

    // Strategy 3: L'Hôpital on the substituted form (t → 0⁺ ≡ two-sided
    // for the analytic expressions this fallback is meant for).
    if has_directional_nodes(arena, evaled, t) {
        return Err(fail("could not compute limit at infinity"));
    }
    match analytic_fallback(arena, evaled, t, zero) {
        Ok(result) if is_valid_limit_value(arena, result, var) => Ok(result),
        _ => Err(fail("could not compute limit at infinity")),
    }
}

/// Check if a result is a finite number (not infinity, NaN, etc.)
fn is_finite_result(arena: &Arena, id: ExprId) -> bool {
    matches!(
        arena.node(id),
        ExprNode::Num(_) | ExprNode::Pi | ExprNode::E
    )
}

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

    #[test]
    fn limit_polynomial_direct_sub() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let two = a.int(2);
        // lim_{x→2} x^2 = 4
        let x2 = a.pow(x, two);
        let result = limit(&mut a, x2, x, two).unwrap();
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn limit_sin_x_over_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let sin_x = a.sin(x);
        let expr = a.div(sin_x, x);
        let result = limit(&mut a, expr, x, zero).unwrap();
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn limit_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let five = a.int(5);
        let zero = a.zero;
        let result = limit(&mut a, five, x, zero).unwrap();
        assert_eq!(display(&a, result), "5");
    }

    #[test]
    fn limit_symbolic_constant() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let c = sym(&mut a, "c");
        let zero = a.zero;
        let result = limit(&mut a, c, x, zero).unwrap();
        assert_eq!(result, c);
    }

    #[test]
    fn limit_x_squared_minus_one_over_x_minus_one() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        let result = limit(&mut a, expr, x, one).unwrap();
        assert_eq!(display(&a, result), "2");
    }

    #[test]
    fn limit_direct_sub_works() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let three = a.int(3);
        let expr = a.add(&[x, a.one]);
        let result = limit(&mut a, expr, x, three).unwrap();
        assert_eq!(display(&a, result), "4");
    }

    #[test]
    fn one_sided_reciprocal() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let one = a.one;
        let expr = a.div(one, x);
        let r = limit_dir(&mut a, expr, x, zero, Direction::Right).unwrap();
        assert_eq!(r, a.infinity());
        let l = limit_dir(&mut a, expr, x, zero, Direction::Left).unwrap();
        assert_eq!(l, a.neg_infinity());
        assert!(limit(&mut a, expr, x, zero).is_err());
    }

    #[test]
    fn two_sided_differ_error_message() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let zero = a.zero;
        let abs_x = a.abs(x);
        let expr = a.div(abs_x, x);
        let err = limit(&mut a, expr, x, zero).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("differ"), "unexpected message: {msg}");
    }

    #[test]
    fn direct_substitution_rejects_folded_indeterminate() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let one = a.one;
        let two = a.int(2);
        let x2 = a.pow(x, two);
        let numer = a.sub(x2, one);
        let denom = a.sub(x, one);
        let expr = a.div(numer, denom);
        // subs folds (0)·(0)⁻¹ → 0; direct substitution must refuse it.
        assert!(try_direct_substitution(&mut a, expr, x, one).is_none());
    }

    #[test]
    fn var_must_be_symbol() {
        let mut a = Arena::new();
        let two = a.int(2);
        let zero = a.zero;
        assert!(limit(&mut a, two, two, zero).is_err());
    }
}
