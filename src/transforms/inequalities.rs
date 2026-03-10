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

/// Solve `expr = 0`, returning solutions as a `FiniteSet` `ExprId`.
pub(crate) fn solveset(arena: &mut Arena, expr: ExprId, var: ExprId) -> ExprId {
    let solutions = crate::transforms::solve::solve(arena, expr, var);
    let root_ids: Vec<ExprId> = solutions.into_iter().map(|s| s.value).collect();
    if root_ids.is_empty() {
        arena.empty_set
    } else {
        arena.finite_set(&root_ids)
    }
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
