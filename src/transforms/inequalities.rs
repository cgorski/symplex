//! Inequality solving via the sign-chart method.
//!
//! # Algorithm
//!
//! Given `f(x) > 0` (or `>=`, `<`, `<=`):
//! 1. Find all real roots of `f(x) = 0`
//! 2. Sort roots numerically
//! 3. Test the sign of `f` in each region between roots
//! 4. Collect intervals where the sign matches the relation
//! 5. Return Union of Intervals (a `SetEx`)

use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{
    ExprId, INTERVAL_BOTH_CLOSED, INTERVAL_BOTH_OPEN, INTERVAL_LEFT_OPEN, INTERVAL_RIGHT_OPEN,
};

/// Relation type for inequalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// Strictly greater than (`> 0`).
    Gt,
    /// Greater than or equal (`>= 0`).
    Ge,
    /// Strictly less than (`< 0`).
    Lt,
    /// Less than or equal (`<= 0`).
    Le,
}

/// Solve `expr rel 0` for `var`, returning the solution as a set `ExprId`.
///
/// The result is a union of intervals (and possibly isolated points for
/// non-strict inequalities) that encodes the solution set on the real line.
pub(crate) fn solve_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Result<ExprId, SymplexError> {
    tracing::debug!("solve_inequality: rel={:?}", rel);

    // ── Absolute-value fast path: c·|a·x + b| + d rel 0 → interval / union
    //    (also handles symbolic endpoints). ──
    if let Some(set) = try_solve_abs_inequality(arena, expr, var, rel) {
        return Ok(set);
    }

    // ── Sturm fast path: if the expression is polynomial, use Sturm
    //    chains to detect the no-real-roots case without solving. ──
    if let Some(poly) = crate::poly::polybridge::expr_to_poly(arena, expr, var) {
        let chain = crate::poly::sturm::SturmChain::new(&poly);
        if chain.has_no_real_roots() {
            // The polynomial has no real roots ⇒ constant sign on ℝ.
            let ls = chain.leading_sign_of_original();
            let is_positive = ls > 0;
            let is_negative = ls < 0;
            let is_zero = ls == 0; // identically zero polynomial
            let matches = match rel {
                Relation::Gt => is_positive,
                Relation::Ge => is_positive || is_zero, // 0 >= 0 is true
                Relation::Lt => is_negative,
                Relation::Le => is_negative || is_zero, // 0 <= 0 is true
            };
            return if matches {
                Ok(arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN))
            } else {
                Ok(arena.empty_set)
            };
        }
    }

    // Step 1: Find roots of expr = 0
    let solutions = crate::transforms::solve::solve(arena, expr, var);
    let roots: Vec<ExprId> = solutions.into_iter().map(|s| s.value).collect();

    tracing::debug!("solve_inequality: found {} roots", roots.len());

    if roots.is_empty() {
        // No roots — the expression doesn't cross zero.
        // Test sign at a point (e.g., x = 0) to determine which way.
        return solve_no_roots(arena, expr, var, rel);
    }

    // Step 2: Evaluate roots to f64 so we can sort them.
    let mut root_vals: Vec<(ExprId, f64)> = Vec::new();
    for &root in &roots {
        let evaled = crate::transforms::eval::eval(arena, root);
        if let Some(val) = try_evalf_f64(arena, evaled)
            && val.is_finite()
        {
            root_vals.push((root, val));
        }
    }
    root_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Deduplicate roots that are numerically very close.
    root_vals.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-12);

    tracing::debug!(
        "solve_inequality: {} numeric roots after sort/dedup",
        root_vals.len()
    );

    if root_vals.is_empty() {
        // All roots were complex (no real roots) — the polynomial has
        // constant sign on ℝ. Fall through to sign-probe logic.
        return solve_no_roots(arena, expr, var, rel);
    }

    // Step 3: Build intervals by testing sign in each region.
    let include_endpoint = matches!(rel, Relation::Ge | Relation::Le);
    let want_positive = matches!(rel, Relation::Gt | Relation::Ge);

    let mut intervals: Vec<ExprId> = Vec::new();

    // Region before first root: (-∞, root[0])
    {
        let test_x = root_vals[0].1 - 1.0;
        if sign_matches(arena, expr, var, test_x, want_positive) {
            let flags = if include_endpoint {
                INTERVAL_LEFT_OPEN | INTERVAL_RIGHT_OPEN
            } else {
                INTERVAL_BOTH_OPEN
            };
            intervals.push(arena.interval(arena.neg_infinity, root_vals[0].0, flags));
        }
    }

    // Regions between consecutive roots.
    for i in 0..root_vals.len().saturating_sub(1) {
        let mid = (root_vals[i].1 + root_vals[i + 1].1) / 2.0;
        if sign_matches(arena, expr, var, mid, want_positive) {
            let flags = if include_endpoint {
                INTERVAL_BOTH_CLOSED
            } else {
                INTERVAL_BOTH_OPEN
            };
            intervals.push(arena.interval(root_vals[i].0, root_vals[i + 1].0, flags));
        } else if include_endpoint {
            // Even if the interior doesn't match, endpoints themselves
            // satisfy non-strict inequalities — they are added below.
        }
    }

    // Region after last root: (root[n-1], ∞)
    {
        let last = root_vals.last().unwrap();
        let test_x = last.1 + 1.0;
        if sign_matches(arena, expr, var, test_x, want_positive) {
            let flags = if include_endpoint {
                INTERVAL_LEFT_OPEN | INTERVAL_RIGHT_OPEN
            } else {
                INTERVAL_BOTH_OPEN
            };
            intervals.push(arena.interval(last.0, arena.infinity, flags));
        }
    }

    // For non-strict inequalities, roots themselves satisfy the relation
    // (since f(root) = 0 and 0 >= 0 / 0 <= 0 are both true). Include
    // them as isolated points so the solution set is correct even when
    // the surrounding open intervals were not collected.
    if include_endpoint {
        let root_ids: Vec<ExprId> = root_vals.iter().map(|(id, _)| *id).collect();
        if !root_ids.is_empty() {
            intervals.push(arena.finite_set(&root_ids));
        }
    }

    // Step 4: Union all collected pieces.
    if intervals.is_empty() {
        Ok(arena.empty_set)
    } else if intervals.len() == 1 {
        Ok(intervals[0])
    } else {
        Ok(arena.set_union(&intervals))
    }
}

/// Solve `expr = 0`, returning the solution set as an `ExprId`.
///
/// - Identity `0 = 0` → `UniversalSet`
/// - Contradiction / no roots found → `EmptySet`
/// - Otherwise a `FiniteSet` of the roots
pub(crate) fn solveset(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    use crate::transforms::solve::SolveOutcome;
    match crate::transforms::solve::solve_classified(arena, expr, var) {
        SolveOutcome::Identity => arena.universal_set,
        SolveOutcome::NoSolution(_) => arena.empty_set,
        SolveOutcome::Solutions(solutions) => {
            let root_ids: Vec<ExprId> = solutions.into_iter().map(|s| s.value).collect();
            if root_ids.is_empty() {
                arena.empty_set
            } else {
                arena.finite_set(&root_ids)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Absolute-value inequalities: c·|f(x)| + d  rel  0  with f linear in x
// ═══════════════════════════════════════════════════════════════════════════

/// Try to solve `expr rel 0` when `expr` has the shape `c·|a·x + b| + d`
/// (a single absolute-value term plus var-free terms).
///
/// Rewrites to `|f| rel' k` and returns the interval / union directly,
/// which also works for **symbolic** `a`, `b`, `k` where numeric root
/// sorting would fail:
///
/// - `|f| < k`  → `-k < f < k`  → one interval in `x`
/// - `|f| > k`  → `f < -k ∪ f > k` → union of two rays
///
/// Returns `None` if the expression does not have this shape or the sign
/// of the leading coefficient `a` cannot be determined.
fn try_solve_abs_inequality(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Option<ExprId> {
    use crate::base::node::ExprNode;

    let terms: Vec<ExprId> = match arena.node(expr).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expr],
    };

    // Locate the single |f(x)| term and its (var-free) coefficient.
    let mut abs_inner: Option<ExprId> = None;
    let mut abs_coeff: Option<ExprId> = None;
    let mut rest: Vec<ExprId> = Vec::new();
    for &t in &terms {
        if !crate::base::walk::contains(arena, t, var) {
            rest.push(t);
            continue;
        }
        let (inner, coeff) = match arena.node(t).clone() {
            ExprNode::Abs(inner) => (inner, arena.one),
            ExprNode::Neg(n) => match arena.node(n).clone() {
                ExprNode::Abs(inner) => (inner, arena.neg_one),
                _ => return None,
            },
            ExprNode::Mul(children) => {
                let mut inner = None;
                let mut consts = Vec::new();
                for &c in &children {
                    if !crate::base::walk::contains(arena, c, var) {
                        consts.push(c);
                    } else if let ExprNode::Abs(i) = arena.node(c).clone()
                        && inner.is_none()
                    {
                        inner = Some(i);
                    } else {
                        return None;
                    }
                }
                let coeff = match consts.len() {
                    0 => arena.one,
                    1 => consts[0],
                    _ => arena.mul(&consts),
                };
                (inner?, coeff)
            }
            _ => return None,
        };
        if abs_inner.is_some() {
            return None; // two abs terms
        }
        abs_inner = Some(inner);
        abs_coeff = Some(coeff);
    }
    let inner = abs_inner?;
    let coeff = abs_coeff?;

    // f = a·x + b must be linear in x.
    let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, inner, var)?;
    if coeffs.len() != 2 {
        return None;
    }
    let a = coeffs[1];
    let b = coeffs[0];

    // c·|f| + d rel 0  ⇔  |f| rel'  (-d/c)   (flip if c < 0)
    let d = match rest.len() {
        0 => arena.zero,
        1 => rest[0],
        _ => arena.add(&rest),
    };
    let c_sign = sign_of(arena, coeff)?;
    let neg_d = arena.neg(d);
    let k = arena.div(neg_d, coeff);
    let k = crate::transforms::eval::eval(arena, k);
    let rel = if c_sign < 0 { flip(rel) } else { rel };

    // If k is a known negative number, |f| < k is empty and |f| > k is ℝ.
    if let Some(kv) = arena.as_num(k).cloned() {
        use num_traits::Signed;
        if kv.is_negative() {
            return Some(match rel {
                Relation::Lt | Relation::Le => arena.empty_set,
                Relation::Gt | Relation::Ge => {
                    arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN)
                }
            });
        }
    }

    // Endpoints in x: f = ±k  ⇒  x = (±k - b)/a
    let a_sign = sign_of(arena, a)?;
    let k_minus_b = arena.sub(k, b);
    let neg_k = arena.neg(k);
    let neg_k_minus_b = arena.sub(neg_k, b);
    let x_hi = arena.div(k_minus_b, a);
    let x_lo = arena.div(neg_k_minus_b, a);
    let x_hi = crate::transforms::eval::eval(arena, x_hi);
    let x_lo = crate::transforms::eval::eval(arena, x_lo);
    let (lo, hi) = if a_sign > 0 {
        (x_lo, x_hi)
    } else {
        (x_hi, x_lo)
    };

    Some(match rel {
        Relation::Lt => arena.interval(lo, hi, INTERVAL_BOTH_OPEN),
        Relation::Le => arena.interval(lo, hi, INTERVAL_BOTH_CLOSED),
        Relation::Gt => {
            let left = arena.interval(arena.neg_infinity, lo, INTERVAL_BOTH_OPEN);
            let right = arena.interval(hi, arena.infinity, INTERVAL_BOTH_OPEN);
            arena.set_union(&[left, right])
        }
        Relation::Ge => {
            let left = arena.interval(arena.neg_infinity, lo, INTERVAL_LEFT_OPEN);
            let right = arena.interval(hi, arena.infinity, INTERVAL_RIGHT_OPEN);
            arena.set_union(&[left, right])
        }
    })
}

/// Reverse a relation (used when dividing by a negative coefficient).
fn flip(rel: Relation) -> Relation {
    match rel {
        Relation::Gt => Relation::Lt,
        Relation::Ge => Relation::Le,
        Relation::Lt => Relation::Gt,
        Relation::Le => Relation::Ge,
    }
}

/// Sign of a var-free expression: `Some(1)`, `Some(-1)`, or `None` if
/// unknown (symbolic without a determinable sign, or zero).
fn sign_of(arena: &mut Arena, e: ExprId) -> Option<i8> {
    use num_traits::Signed;
    let ev = crate::transforms::eval::eval(arena, e);
    if let Some(r) = arena.as_num(ev) {
        return if r.is_positive() {
            Some(1)
        } else if r.is_negative() {
            Some(-1)
        } else {
            None
        };
    }
    if let Some(v) = try_evalf_f64(arena, ev) {
        if v > 0.0 {
            return Some(1);
        }
        if v < 0.0 {
            return Some(-1);
        }
        return None;
    }
    // Symbolic: consult stored symbol assumptions for a bare symbol.
    if let crate::base::node::ExprNode::Symbol(sid) = arena.node(ev) {
        let a = arena.symbol_assumptions(*sid);
        if a.known_true
            .contains(crate::base::assumptions::Props::POSITIVE)
        {
            return Some(1);
        }
        if a.known_true
            .contains(crate::base::assumptions::Props::NEGATIVE)
        {
            return Some(-1);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Handle the case where the expression has no roots.
///
/// If `f(x)` has no real roots then it either has constant sign everywhere
/// or we cannot determine it. We probe a few sample points.
fn solve_no_roots(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    rel: Relation,
) -> Result<ExprId, SymplexError> {
    // Try exact evaluation at x = 0 first.
    let zero = arena.zero;
    let val = crate::transforms::subs::subs(arena, expr, var, zero);
    let val_eval = crate::transforms::eval::eval(arena, val);

    if let Some(r) = arena.as_num(val_eval).cloned() {
        use num_traits::{Signed, Zero};
        let sign_positive = r.is_positive();
        let sign_negative = r.is_negative();
        let sign_zero = r.is_zero();

        let matches = match rel {
            Relation::Gt => sign_positive,
            Relation::Ge => sign_positive || sign_zero,
            Relation::Lt => sign_negative,
            Relation::Le => sign_negative || sign_zero,
        };

        return if matches {
            // Whole real line satisfies the inequality.
            Ok(arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN))
        } else {
            Ok(arena.empty_set)
        };
    }

    // Fall back to numerical sign probing at several points.
    let want_positive = matches!(rel, Relation::Gt | Relation::Ge);
    for &test_val in &[0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0] {
        let positive_here = sign_matches(arena, expr, var, test_val, true);
        let negative_here = sign_matches(arena, expr, var, test_val, false);

        // If we got a definitive reading, use it.
        if positive_here || negative_here {
            let matches = if want_positive {
                positive_here
            } else {
                negative_here
            };
            return if matches {
                Ok(arena.interval(arena.neg_infinity, arena.infinity, INTERVAL_BOTH_OPEN))
            } else {
                Ok(arena.empty_set)
            };
        }
    }

    // Truly could not determine the sign — report failure.
    Err(SymplexError::ComputationFailed {
        operation: "solve_inequality",
        reason: "could not determine sign of expression (no roots found and evaluation failed)"
            .to_string(),
    })
}

/// Test whether the sign of `expr` at `var = test_val` is positive
/// (if `want_positive`) or negative (otherwise).
fn sign_matches(
    arena: &mut Arena,
    expr: ExprId,
    var: ExprId,
    test_val: f64,
    want_positive: bool,
) -> bool {
    let test_expr = rational_approx(arena, test_val);
    let substituted = crate::transforms::subs::subs(arena, expr, var, test_expr);
    let evaluated = crate::transforms::eval::eval(arena, substituted);

    // Try exact rational check first.
    if let Some(r) = arena.as_num(evaluated).cloned() {
        use num_traits::{Signed, Zero};
        if r.is_zero() {
            return false; // zero is neither positive nor negative
        }
        return if want_positive {
            r.is_positive()
        } else {
            r.is_negative()
        };
    }

    // Fall back to numerical evaluation.
    if let Some(val) = try_evalf_f64(arena, evaluated) {
        if val.abs() < 1e-15 {
            return false; // treat as zero
        }
        return if want_positive { val > 0.0 } else { val < 0.0 };
    }

    false
}

/// Convert an `f64` to a rational `ExprId` via fixed-point approximation.
///
/// Multiplies by 10^9, rounds, and creates the ratio `round(val * 10^9) / 10^9`.
fn rational_approx(arena: &mut Arena, val: f64) -> ExprId {
    const SCALE: i64 = 1_000_000_000;
    let numer = (val * SCALE as f64).round() as i64;
    arena.rational(numer, SCALE)
}

/// Try to evaluate an `ExprId` to `f64` via the `evalf` module.
///
/// Returns `None` if the expression contains free symbols or otherwise
/// cannot be numerically evaluated.
fn try_evalf_f64(arena: &Arena, expr: ExprId) -> Option<f64> {
    // Use 16 digits of precision — more than enough for sign testing.
    let s = crate::transforms::evalf::evalf(arena, expr, 16).ok()?;
    // The string might be something like "3.14159265358979" or "-2.0".
    // It could also be complex like "1.0 + 2.0*I" — skip those.
    if s.contains('I') || s.contains('i') {
        return None;
    }
    s.trim().parse::<f64>().ok()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::arena::Arena;

    /// Helper: build `x^2 - 4` in the arena and solve `> 0`.
    #[test]
    fn sign_chart_x2_minus_4_gt() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let two = arena.int(2);
        let x2 = arena.pow(x, two);
        let four = arena.int(4);
        let expr = arena.sub(x2, four); // x² - 4

        let result = solve_inequality(&mut arena, expr, x, Relation::Gt);
        assert!(result.is_ok(), "should succeed: {:?}", result);
    }

    #[test]
    fn positive_constant_gt_zero() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let five = arena.int(5);

        let result = solve_inequality(&mut arena, five, x, Relation::Gt).unwrap();
        // 5 > 0 always true → should be (-∞, ∞), not EmptySet
        assert_ne!(result, arena.empty_set, "5 > 0 should not be EmptySet");
    }

    #[test]
    fn negative_constant_gt_zero() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        let neg3 = arena.int(-3);

        let result = solve_inequality(&mut arena, neg3, x, Relation::Gt).unwrap();
        assert_eq!(result, arena.empty_set, "-3 > 0 should be EmptySet");
    }

    #[test]
    fn solveset_quadratic_arena() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        // x² - 5x + 6 = (x-2)(x-3)
        let two = arena.int(2);
        let x2 = arena.pow(x, two);
        let five = arena.int(5);
        let five_x = arena.mul(&[five, x]);
        let neg_five_x = arena.neg(five_x);
        let six = arena.int(6);
        let expr = arena.add(&[x2, neg_five_x, six]);

        let result = solveset(&mut arena, expr, x);
        assert_ne!(result, arena.empty_set, "should find roots of x²-5x+6");
    }
}
