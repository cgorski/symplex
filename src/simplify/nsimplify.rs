//! Find a simple closed-form expression for a numerical value.
//!
//! Given a floating-point approximation, search for:
//! 1. Simple rational p/q with small p, q (continued fraction algorithm)
//! 2. Multiples of π (k·π/n for small k, n)
//! 3. Square roots of small integers
//! 4. Combinations of the above
//!
//! The function tries each strategy in order and returns the first match
//! within the given tolerance. If nothing is found, the original
//! expression is returned unchanged.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::f64_to_ratio_approx;

/// Largest denominator accepted for the rational factor / offset in
/// [`nsimplify_with_constants`].
const CONST_MAX_DENOM: u64 = 1_000;

/// Largest numerator / denominator accepted for the rational exponent in
/// the `c^(p/q)` form of [`nsimplify_with_constants`].
const CONST_MAX_EXP: i64 = 12;

/// The default constant table used by
/// [`Ex::nsimplify`](crate::api::expr::Ex::nsimplify):
/// `π, e, √2, √3, √5, ln 2, φ, γ`.
pub(crate) fn default_constants(arena: &mut Arena) -> Vec<ExprId> {
    let half = arena.rational(1, 2);
    let two = arena.int(2);
    let three = arena.int(3);
    let five = arena.int(5);
    let sqrt2 = arena.pow(two, half);
    let sqrt3 = arena.pow(three, half);
    let sqrt5 = arena.pow(five, half);
    let ln2 = arena.ln(two);
    vec![
        arena.pi,
        arena.e_const(),
        sqrt2,
        sqrt3,
        sqrt5,
        ln2,
        arena.golden_ratio(),
        arena.euler_gamma(),
    ]
}

/// Recognise a numerical expression as a simple combination of the given
/// constants.
///
/// After trying a plain rational, each constant `c` (evaluated to `f64`)
/// is tested in turn for the forms
///
/// 1. `q·c`  — `q` rational with denominator ≤ 1000,
/// 2. `q + c` — likewise,
/// 3. `c^(p/q)` — `|p|, q ≤ 12` (only for `c > 0`, `c ≠ 1`),
///
/// accepting the first candidate whose value is within `tolerance` of the
/// input.  Falls back to the π-multiple / square-root heuristics of
/// [`nsimplify`].  Returns the input unchanged if nothing matches or it
/// cannot be evaluated.
pub(crate) fn nsimplify_with_constants(
    arena: &mut Arena,
    expr: ExprId,
    constants: &[ExprId],
    tolerance: f64,
) -> ExprId {
    let val = match eval_to_f64(arena, expr) {
        Some(v) if v.is_finite() => v,
        _ => return expr,
    };

    // Small-denominator rationals win over constant combinations (so that
    // `3` is not reported as `q·π` under a loose tolerance); larger
    // denominators are only tried after the constant forms.
    if let Some(q) = f64_to_ratio_approx(val, 100)
        && let Some(qf) = q.to_f64()
        && (qf - val).abs() < tolerance
    {
        return num_expr(arena, q);
    }

    let const_vals: Vec<(ExprId, f64)> = constants
        .iter()
        .filter_map(|&c| eval_to_f64(arena, c).map(|v| (c, v)))
        .filter(|(_, v)| v.is_finite() && v.abs() > 1e-12)
        .collect();

    // 1. q·c
    for &(c, cv) in &const_vals {
        if let Some(q) = f64_to_ratio_approx(val / cv, CONST_MAX_DENOM)
            && !q.is_zero()
            && let Some(qf) = q.to_f64()
            && (qf * cv - val).abs() < tolerance
        {
            return arena.make_coeff_term(q, c);
        }
    }

    // 2. q + c
    for &(c, cv) in &const_vals {
        if let Some(q) = f64_to_ratio_approx(val - cv, CONST_MAX_DENOM)
            && let Some(qf) = q.to_f64()
            && (qf + cv - val).abs() < tolerance
        {
            if q.is_zero() {
                return c;
            }
            let q_id = num_expr(arena, q);
            return arena.add(&[q_id, c]);
        }
    }

    // 3. c^(p/q)
    if val > 0.0 {
        for &(c, cv) in &const_vals {
            if cv <= 0.0 || (cv - 1.0).abs() < 1e-12 {
                continue;
            }
            let t = val.ln() / cv.ln();
            for q in 1..=CONST_MAX_EXP {
                let p = (t * q as f64).round() as i64;
                if p == 0 || p.abs() > CONST_MAX_EXP {
                    continue;
                }
                let approx = cv.powf(p as f64 / q as f64);
                if (approx - val).abs() < tolerance {
                    let e = Ratio::new(BigInt::from(p), BigInt::from(q));
                    if e.is_one() {
                        return c;
                    }
                    let e_id = num_expr(arena, e);
                    return arena.pow(c, e_id);
                }
            }
        }
    }

    if let Some(r) = find_rational(arena, val, tolerance) {
        return r;
    }

    nsimplify(arena, expr, tolerance)
}

/// Intern a rational number as an expression node.
fn num_expr(arena: &mut Arena, r: Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r);
    arena.intern(ExprNode::Num(nid))
}

/// Try to find a simple closed-form for a numerical expression.
///
/// Evaluates `expr` to `f64`, then searches for rational approximations,
/// π-multiples, and square roots within `tolerance`.
pub(crate) fn nsimplify(arena: &mut Arena, expr: ExprId, tolerance: f64) -> ExprId {
    // First, try to evaluate to f64 via the evalf machinery.
    let val = match eval_to_f64(arena, expr) {
        Some(v) if v.is_finite() => v,
        _ => return expr, // Can't evaluate — return unchanged
    };

    // Strategy 1: Try multiples of π: k*π/n for small k, n
    // (checked first so that π/2 isn't falsely matched as 355/226)
    if let Some(pi_mult) = find_pi_multiple(arena, val, tolerance) {
        return pi_mult;
    }

    // Strategy 2: Try simple rationals p/q via continued fractions
    if let Some(rational) = find_rational(arena, val, tolerance) {
        return rational;
    }

    // Strategy 3: Try √n for small n (positive values only)
    if let Some(sqrt_n) = find_sqrt(arena, val, tolerance) {
        return sqrt_n;
    }

    // Strategy 4: Try rational multiples of √n: (p/q)*√n
    if let Some(rsqrt) = find_rational_sqrt(arena, val, tolerance) {
        return rsqrt;
    }

    expr // Nothing found — return original
}

/// Evaluate an expression to f64 using the evalf machinery.
///
/// Returns `None` if the expression cannot be numerically evaluated
/// (e.g. it contains free symbols).
fn eval_to_f64(arena: &Arena, expr: ExprId) -> Option<f64> {
    // Use 16-digit precision, parse the string result.
    let s = crate::transforms::evalf::evalf(arena, expr, 16).ok()?;
    s.parse::<f64>().ok()
}

/// Find a rational approximation p/q using continued fraction convergents.
///
/// Uses the standard continued fraction algorithm to compute convergents
/// of `val`. Returns the first convergent within `tol` whose denominator
/// does not exceed 10 000.
fn find_rational(arena: &mut Arena, val: f64, tol: f64) -> Option<ExprId> {
    if !val.is_finite() {
        return None;
    }

    // Handle zero specially.
    if val.abs() < tol {
        return Some(arena.zero);
    }

    // Handle negative values: nsimplify(-val) then negate.
    let (sign, abs_val) = if val < 0.0 { (-1i64, -val) } else { (1, val) };

    // Continued-fraction convergents.
    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut x = abs_val;

    for _ in 0..50 {
        let a = x.floor() as i64;

        let p2 = a.checked_mul(p1)?.checked_add(p0)?;
        let q2 = a.checked_mul(q1)?.checked_add(q0)?;

        if !(0..=10_000).contains(&q2) {
            break;
        }
        if q2 != 0 {
            let approx = p2 as f64 / q2 as f64;
            if (approx - abs_val).abs() < tol {
                let p_final = sign * p2;
                // Build the expression.
                if q2 == 1 {
                    return Some(arena.int(p_final));
                } else {
                    return Some(arena.rational(p_final, q2));
                }
            }
        }

        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;

        let frac = x - a as f64;
        if frac.abs() < 1e-15 {
            break;
        }
        x = 1.0 / frac;

        // Guard against overflow in the next iteration.
        if x > 1e15 {
            break;
        }
    }

    None
}

/// Try to express `val` as k*π/n for small integer k and n (1..=12).
fn find_pi_multiple(arena: &mut Arena, val: f64, tol: f64) -> Option<ExprId> {
    let pi = std::f64::consts::PI;

    // Try val = k*pi/n for n in 1..=12 and |k| <= 12*n (reasonable range).
    for n in 1..=12i64 {
        let candidate = val * n as f64 / pi;
        let k = candidate.round() as i64;
        if k == 0 {
            continue;
        }
        let approx = k as f64 * pi / n as f64;
        if (approx - val).abs() < tol {
            // val ≈ k*π/n — build the expression.
            return Some(build_pi_fraction(arena, k, n));
        }
    }
    None
}

/// Build the expression k*π/n, simplifying trivial cases.
fn build_pi_fraction(arena: &mut Arena, k: i64, n: i64) -> ExprId {
    let pi_expr = arena.pi;

    // Reduce k/n to lowest terms.
    let g = gcd_i64(k.unsigned_abs(), n.unsigned_abs()) as i64;
    let k_red = k / g;
    let n_red = n / g;

    // k_red * π / n_red
    let numerator = if k_red == 1 {
        pi_expr
    } else if k_red == -1 {
        arena.neg(pi_expr)
    } else {
        let k_expr = arena.int(k_red);
        arena.mul(&[k_expr, pi_expr])
    };

    if n_red == 1 {
        numerator
    } else {
        let n_expr = arena.int(n_red);
        arena.div(numerator, n_expr)
    }
}

/// Try to express `val` as ±√n for small integer n (2..=100).
fn find_sqrt(arena: &mut Arena, val: f64, tol: f64) -> Option<ExprId> {
    let (sign, abs_val) = if val < 0.0 { (-1, -val) } else { (1, val) };

    // Don't match values very close to an integer (those should be
    // caught by find_rational already).
    if (abs_val - abs_val.round()).abs() < tol {
        return None;
    }

    let val_sq = abs_val * abs_val;
    for n in 2..=100i64 {
        if (val_sq - n as f64).abs() < tol * val_sq.max(1.0) {
            // val ≈ √n — check more precisely.
            let sqrt_n = (n as f64).sqrt();
            if (sqrt_n - abs_val).abs() < tol {
                // Build √n = n^(1/2).
                let half = arena.rational(1, 2);
                let n_expr = arena.int(n);
                let result = arena.pow(n_expr, half);
                if sign < 0 {
                    return Some(arena.neg(result));
                } else {
                    return Some(result);
                }
            }
        }
    }
    None
}

/// Try to express `val` as (p/q)*√n for small p, q, n.
fn find_rational_sqrt(arena: &mut Arena, val: f64, tol: f64) -> Option<ExprId> {
    if val.abs() < tol {
        return None;
    }

    let (sign, abs_val) = if val < 0.0 { (-1i64, -val) } else { (1, val) };

    // Try √n for n = 2..=20, then see if val/√n is a simple rational.
    for n in 2..=20i64 {
        let sqrt_n = (n as f64).sqrt();
        let ratio = abs_val / sqrt_n;

        // Use continued fractions to find a simple rational for ratio.
        if let Some((p, q)) = find_rational_pair(ratio, tol / sqrt_n)
            && q <= 100
            && p.unsigned_abs() <= 100
        {
            let p_signed = sign * p;
            // Build (p/q) * √n.
            let half = arena.rational(1, 2);
            let n_expr = arena.int(n);
            let sqrt_expr = arena.pow(n_expr, half);

            if p_signed == 1 && q == 1 {
                return Some(sqrt_expr);
            } else if q == 1 {
                let p_expr = arena.int(p_signed);
                return Some(arena.mul(&[p_expr, sqrt_expr]));
            } else {
                let coeff = arena.rational(p_signed, q);
                return Some(arena.mul(&[coeff, sqrt_expr]));
            }
        }
    }
    None
}

/// Find a rational approximation p/q of `val` within `tol`, returning (p, q).
fn find_rational_pair(val: f64, tol: f64) -> Option<(i64, i64)> {
    if !val.is_finite() {
        return None;
    }

    let abs_val = val.abs();
    let sign: i64 = if val < 0.0 { -1 } else { 1 };

    let mut p0: i64 = 0;
    let mut q0: i64 = 1;
    let mut p1: i64 = 1;
    let mut q1: i64 = 0;
    let mut x = abs_val;

    for _ in 0..50 {
        let a = x.floor() as i64;
        let p2 = a.checked_mul(p1)?.checked_add(p0)?;
        let q2 = a.checked_mul(q1)?.checked_add(q0)?;

        if !(0..=1000).contains(&q2) {
            break;
        }
        if q2 != 0 {
            let approx = p2 as f64 / q2 as f64;
            if (approx - abs_val).abs() < tol {
                return Some((sign * p2, q2));
            }
        }

        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;

        let frac = x - a as f64;
        if frac.abs() < 1e-15 {
            break;
        }
        x = 1.0 / frac;
        if x > 1e15 {
            break;
        }
    }
    None
}

/// Simple GCD for unsigned 64-bit integers.
fn gcd_i64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn nsimplify_integer() {
        let mut arena = Arena::new();
        let expr = arena.rational(500001, 100000); // ≈ 5.00001
        let result = nsimplify(&mut arena, expr, 1e-3);
        assert_eq!(display(&arena, result), "5");
    }

    #[test]
    fn nsimplify_one_third() {
        let mut arena = Arena::new();
        let expr = arena.rational(333333, 1000000); // ≈ 0.333333
        let result = nsimplify(&mut arena, expr, 1e-5);
        assert_eq!(display(&arena, result), "1/3");
    }

    #[test]
    fn nsimplify_pi() {
        let mut arena = Arena::new();
        // Build a rational approximation of π: 314159265/100000000
        let expr = arena.rational(314159265, 100000000);
        let result = nsimplify(&mut arena, expr, 1e-7);
        let s = display(&arena, result);
        assert!(s.contains("pi"), "should find π, got: {s}");
    }

    #[test]
    fn nsimplify_half_pi() {
        let mut arena = Arena::new();
        // π/2 ≈ 1.5707963
        let expr = arena.rational(15707963, 10000000);
        let result = nsimplify(&mut arena, expr, 1e-6);
        let s = display(&arena, result);
        assert!(s.contains("pi"), "should find π/2, got: {s}");
    }

    #[test]
    fn nsimplify_sqrt2() {
        let mut arena = Arena::new();
        // √2 ≈ 1.41421356
        let expr = arena.rational(14142, 10000);
        let result = nsimplify(&mut arena, expr, 1e-3);
        let s = display(&arena, result);
        // Should produce 2^(1/2) or equivalent
        assert!(
            s.contains("1/2") || s.contains("^"),
            "should find √2, got: {s}"
        );
    }

    #[test]
    fn nsimplify_exact_integer_stays_integer() {
        let mut arena = Arena::new();
        let expr = arena.int(42);
        let result = nsimplify(&mut arena, expr, 1e-10);
        assert_eq!(display(&arena, result), "42");
    }

    #[test]
    fn nsimplify_zero() {
        let mut arena = Arena::new();
        let expr = arena.rational(0, 1);
        let result = nsimplify(&mut arena, expr, 1e-10);
        assert_eq!(display(&arena, result), "0");
    }

    #[test]
    fn nsimplify_negative_rational() {
        let mut arena = Arena::new();
        // -1/7 ≈ -0.142857
        let expr = arena.rational(-142857, 1000000);
        let result = nsimplify(&mut arena, expr, 1e-5);
        let s = display(&arena, result);
        assert!(
            s.contains("1/7") || s.contains("-1/7"),
            "should find -1/7, got: {s}"
        );
    }

    #[test]
    fn nsimplify_no_match_returns_original() {
        let mut arena = Arena::new();
        let x = arena.symbol("x");
        // Free symbol — can't evaluate
        let result = nsimplify(&mut arena, x, 1e-10);
        assert_eq!(result, x);
    }
}
