//! Exact evaluation of known special values.
//!
//! This module implements [`eval`], which replaces function applications
//! with their exact values when the arguments are known constants.
//!
//! # What `eval` does
//!
//! - `sin(0)` → `0`
//! - `sin(π/6)` → `1/2`
//! - `sin(π/2)` → `1`
//! - `sin(5π/6)` → `1/2`
//! - `sin(π)` → `0`
//! - `sin(7π/6)` → `-1/2`
//! - `sin(3π/2)` → `-1`
//! - `sin(11π/6)` → `-1/2`
//! - `cos(0)` → `1`
//! - `cos(π/3)` → `1/2`
//! - `cos(π/2)` → `0`
//! - `cos(2π/3)` → `-1/2`
//! - `cos(π)` → `-1`
//! - `cos(4π/3)` → `-1/2`
//! - `cos(3π/2)` → `0`
//! - `cos(5π/3)` → `1/2`
//! - `tan(0)` → `0`
//! - `tan(π/4)` → `1`
//! - `tan(3π/4)` → `-1`
//! - `exp(0)` → `1`
//! - `exp(1)` → `E`
//! - `ln(1)` → `0`
//! - `ln(E)` → `1`
//! - `sqrt(0)` → `0`
//! - `sqrt(1)` → `1`
//! - `sqrt(p/q)` → `√p/√q` when both `p` and `q` are perfect squares
//! - `abs(x)` → `x` when `x` is a non-negative number
//! - `abs(x)` → `-x` when `x` is a negative number
//!
//! Only evaluates when the result is an exact atom (number or constant).
//! Does **not** evaluate `cos(π/4)` → `√2/2` because that would create
//! a more complex expression, violating our "eval returns simpler" rule.
//!
//! # Design
//!
//! Uses the iterative bottom-up walker — no recursion.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

/// Evaluate known special values in an expression.
///
/// Walks the expression bottom-up and replaces function applications
/// with their exact values when the arguments are known constants.
///
/// The result is fully canonicalized.
pub(crate) fn eval(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let node = arena.node(id).clone();
        let evaluated = match node {
            ExprNode::Sin(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_sin(arena, inner).unwrap_or_else(|| arena.sin(inner))
            }
            ExprNode::Cos(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_cos(arena, inner).unwrap_or_else(|| arena.cos(inner))
            }
            ExprNode::Tan(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_tan(arena, inner).unwrap_or_else(|| arena.tan(inner))
            }
            ExprNode::Exp(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_exp(arena, inner).unwrap_or_else(|| arena.exp(inner))
            }
            ExprNode::Ln(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_ln(arena, inner).unwrap_or_else(|| arena.ln(inner))
            }
            ExprNode::Sqrt(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_sqrt(arena, inner).unwrap_or_else(|| arena.sqrt(inner))
            }
            ExprNode::Abs(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_abs(arena, inner).unwrap_or_else(|| arena.abs(inner))
            }
            ExprNode::Asin(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_asin(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Asin(inner)))
            }
            ExprNode::Acos(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_acos(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Acos(inner)))
            }
            ExprNode::Atan(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_atan(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Atan(inner)))
            }
            ExprNode::Sinh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_sinh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Sinh(inner)))
            }
            ExprNode::Cosh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_cosh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Cosh(inner)))
            }
            ExprNode::Tanh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_tanh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Tanh(inner)))
            }
            // Rebuild Add/Mul/Pow/Neg with evaluated children.
            ExprNode::Add(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.add(&new)
                }
            }
            ExprNode::Mul(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new == *children {
                    id
                } else {
                    arena.mul(&new)
                }
            }
            ExprNode::Pow(base, exp) => {
                let nb = cache.get(&base).copied().unwrap_or(base);
                let ne = cache.get(&exp).copied().unwrap_or(exp);
                if nb == base && ne == exp {
                    id
                } else {
                    arena.pow(nb, ne)
                }
            }
            ExprNode::Neg(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if ni == inner { id } else { arena.neg(ni) }
            }
            // Everything else: unchanged.
            _ => id,
        };

        cache.insert(id, evaluated);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Special value tables
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is the constant `π`.
fn is_pi(arena: &Arena, id: ExprId) -> bool {
    id == arena.pi
}

/// Check if `id` is the constant `E`.
fn is_e(arena: &Arena, id: ExprId) -> bool {
    id == arena.e_const
}

/// Check if `id` is a rational multiple of π: returns the multiplier
/// as `Some(Ratio)`, or `None` if not a multiple of π.
fn as_pi_multiple(arena: &Arena, id: ExprId) -> Option<Ratio<BigInt>> {
    // Exact π.
    if is_pi(arena, id) {
        return Some(Ratio::one());
    }

    // c * π where c is numeric.
    if let ExprNode::Mul(ref children) = arena.node(id).clone()
        && children.len() == 2
        && let ExprNode::Num(nid) = arena.node(children[0])
    {
        let nid = *nid;
        if children[1] == arena.pi {
            return Some(arena.num(nid).clone());
        }
    }

    // Numeric 0 (= 0 * π).
    if let Some(r) = arena.as_num(id)
        && r.is_zero()
    {
        return Some(Ratio::zero());
    }

    None
}

/// Evaluate `sin(inner)` for known special values.
fn eval_sin(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let coeff = as_pi_multiple(arena, inner)?;

    // Reduce modulo 2 (sin has period 2π).
    let two: Ratio<BigInt> = Ratio::from_integer(2.into());
    let coeff = mod_positive(&coeff, &two);

    // Known values of sin(k*π).
    // sin(0) = 0
    if coeff.is_zero() {
        return Some(arena.zero);
    }
    // sin(π/2) = 1
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.one);
    }
    // sin(π) = 0
    if coeff == Ratio::one() {
        return Some(arena.zero);
    }
    // sin(3π/2) = -1
    if coeff == Ratio::new(3.into(), 2.into()) {
        return Some(arena.neg_one);
    }
    // sin(π/6) = 1/2
    if coeff == Ratio::new(1.into(), 6.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // sin(5π/6) = 1/2
    if coeff == Ratio::new(5.into(), 6.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // sin(7π/6) = -1/2
    if coeff == Ratio::new(7.into(), 6.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // sin(11π/6) = -1/2
    if coeff == Ratio::new(11.into(), 6.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }

    None
}

/// Evaluate `cos(inner)` for known special values.
fn eval_cos(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let coeff = as_pi_multiple(arena, inner)?;

    let two: Ratio<BigInt> = Ratio::from_integer(2.into());
    let coeff = mod_positive(&coeff, &two);

    // cos(0) = 1
    if coeff.is_zero() {
        return Some(arena.one);
    }
    // cos(π/2) = 0
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.zero);
    }
    // cos(π) = -1
    if coeff == Ratio::one() {
        return Some(arena.neg_one);
    }
    // cos(3π/2) = 0
    if coeff == Ratio::new(3.into(), 2.into()) {
        return Some(arena.zero);
    }
    // cos(π/3) = 1/2
    if coeff == Ratio::new(1.into(), 3.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }
    // cos(2π/3) = -1/2
    if coeff == Ratio::new(2.into(), 3.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // cos(4π/3) = -1/2
    if coeff == Ratio::new(4.into(), 3.into()) {
        let neg_half = arena.rational(-1, 2);
        return Some(neg_half);
    }
    // cos(5π/3) = 1/2
    if coeff == Ratio::new(5.into(), 3.into()) {
        let half = arena.rational(1, 2);
        return Some(half);
    }

    None
}

/// Evaluate `tan(inner)` for known special values.
fn eval_tan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let coeff = as_pi_multiple(arena, inner)?;

    let one_ratio: Ratio<BigInt> = Ratio::one();
    let coeff = mod_positive(&coeff, &one_ratio);

    // tan(0) = 0
    if coeff.is_zero() {
        return Some(arena.zero);
    }
    // tan(π/2) is undefined — leave unevaluated.
    // tan(π/4) = 1
    if coeff == Ratio::new(1.into(), 4.into()) {
        return Some(arena.one);
    }
    // tan(3π/4) = -1
    if coeff == Ratio::new(3.into(), 4.into()) {
        return Some(arena.neg_one);
    }

    None
}

/// Evaluate `exp(inner)` for known special values.
fn eval_exp(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // exp(0) = 1
    if inner == arena.zero {
        return Some(arena.one);
    }
    // exp(1) = E
    if inner == arena.one {
        return Some(arena.e_const);
    }

    None
}

/// Evaluate `ln(inner)` for known special values.
fn eval_ln(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // ln(1) = 0
    if inner == arena.one {
        return Some(arena.zero);
    }
    // ln(E) = 1
    if is_e(arena, inner) {
        return Some(arena.one);
    }

    None
}

/// Evaluate `sqrt(inner)` for known special values.
fn eval_sqrt(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // sqrt(0) = 0
    if inner == arena.zero {
        return Some(arena.zero);
    }
    // sqrt(1) = 1
    if inner == arena.one {
        return Some(arena.one);
    }

    // sqrt(n^2) for positive integer n.
    if let Some(r) = arena.as_num(inner)
        && r.is_integer()
        && r.is_positive()
    {
        let n = r.to_integer();
        let sqrt_n = n.sqrt();
        if &sqrt_n * &sqrt_n == n {
            let nid = arena.intern_num(Ratio::from_integer(sqrt_n));
            return Some(arena.intern(ExprNode::Num(nid)));
        }
    }

    // sqrt(p/q) for perfect square p and q.
    if let Some(r) = arena.as_num(inner)
        && !r.is_integer()
        && r.is_positive()
    {
        let r = r.clone();
        let n = r.numer().abs();
        let d = r.denom().abs();
        let sqrt_n = n.sqrt();
        let sqrt_d = d.sqrt();
        if &sqrt_n * &sqrt_n == n && &sqrt_d * &sqrt_d == d {
            let result = Ratio::new(sqrt_n, sqrt_d);
            let nid = arena.intern_num(result);
            return Some(arena.intern(ExprNode::Num(nid)));
        }
    }

    None
}

/// Evaluate `abs(inner)` for known numeric values.
fn eval_abs(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(inner) {
        let r = r.clone();
        if r.is_negative() {
            let pos = -r;
            let nid = arena.intern_num(pos);
            return Some(arena.intern(ExprNode::Num(nid)));
        } else {
            return Some(inner);
        }
    }

    None
}

fn eval_asin(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // asin(0) = 0
    if inner == arena.one {
        // asin(1) = π/2
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if inner == arena.neg_one {
        // asin(-1) = -π/2
        let neg_half = arena.rational(-1, 2);
        return Some(arena.mul(&[neg_half, arena.pi]));
    }
    None
}

fn eval_acos(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.one {
        return Some(arena.zero);
    } // acos(1) = 0
    if inner == arena.zero {
        // acos(0) = π/2
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if inner == arena.neg_one {
        return Some(arena.pi);
    } // acos(-1) = π
    None
}

fn eval_atan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atan(0) = 0
    if inner == arena.one {
        // atan(1) = π/4
        let quarter = arena.rational(1, 4);
        return Some(arena.mul(&[quarter, arena.pi]));
    }
    if inner == arena.neg_one {
        // atan(-1) = -π/4
        let neg_quarter = arena.rational(-1, 4);
        return Some(arena.mul(&[neg_quarter, arena.pi]));
    }
    None
}

fn eval_sinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // sinh(0) = 0
    None
}

fn eval_cosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.one);
    } // cosh(0) = 1
    None
}

fn eval_tanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // tanh(0) = 0
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute `a mod m` in the range `[0, m)` for positive `m`.
fn mod_positive(a: &Ratio<BigInt>, m: &Ratio<BigInt>) -> Ratio<BigInt> {
    let mut result = a % m;
    if result.is_negative() {
        result += m;
    }
    result
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

    // ── sin ──────────────────────────────────────────────────────────

    #[test]
    fn eval_sin_zero() {
        let mut a = Arena::new();
        let expr = a.sin(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "sin(0) should be 0");
    }

    #[test]
    fn eval_sin_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.sin(pi);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "sin(π) should be 0");
    }

    #[test]
    fn eval_sin_pi_over_2() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let pi = a.pi;
        let pi_over_2 = a.mul(&[half, pi]);
        let expr = a.sin(pi_over_2);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "sin(π/2) should be 1");
    }

    #[test]
    fn eval_sin_3pi_over_2() {
        let mut a = Arena::new();
        let three_halves = a.rational(3, 2);
        let pi = a.pi;
        let arg = a.mul(&[three_halves, pi]);
        let expr = a.sin(arg);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.neg_one, "sin(3π/2) should be -1");
    }

    #[test]
    fn eval_sin_symbolic_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.sin(x);
        let result = eval(&mut a, expr);
        assert_eq!(result, expr, "sin(x) should stay unevaluated");
    }

    // ── cos ──────────────────────────────────────────────────────────

    #[test]
    fn eval_cos_zero() {
        let mut a = Arena::new();
        let expr = a.cos(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "cos(0) should be 1");
    }

    #[test]
    fn eval_cos_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.cos(pi);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.neg_one, "cos(π) should be -1");
    }

    #[test]
    fn eval_cos_pi_over_2() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let pi = a.pi;
        let pi_over_2 = a.mul(&[half, pi]);
        let expr = a.cos(pi_over_2);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "cos(π/2) should be 0");
    }

    #[test]
    fn eval_cos_pi_over_4_unchanged() {
        let mut a = Arena::new();
        let quarter = a.rational(1, 4);
        let pi = a.pi;
        let arg = a.mul(&[quarter, pi]);
        let expr = a.cos(arg);
        let result = eval(&mut a, expr);
        // cos(π/4) = √2/2, but we don't evaluate that (would be more complex).
        assert_eq!(
            display(&a, result),
            "cos(1/4*pi)",
            "cos(π/4) should stay unevaluated"
        );
    }

    // ── tan ──────────────────────────────────────────────────────────

    #[test]
    fn eval_tan_zero() {
        let mut a = Arena::new();
        let expr = a.tan(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "tan(0) should be 0");
    }

    #[test]
    fn eval_tan_pi() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.tan(pi);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "tan(π) should be 0");
    }

    // ── exp ──────────────────────────────────────────────────────────

    #[test]
    fn eval_exp_zero() {
        let mut a = Arena::new();
        let expr = a.exp(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "exp(0) should be 1");
    }

    #[test]
    fn eval_exp_one() {
        let mut a = Arena::new();
        let expr = a.exp(a.one);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.e_const, "exp(1) should be E");
    }

    #[test]
    fn eval_exp_symbolic_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.exp(x);
        let result = eval(&mut a, expr);
        assert_eq!(result, expr, "exp(x) should stay unevaluated");
    }

    // ── ln ──────────────────────────────────────────────────────────

    #[test]
    fn eval_ln_one() {
        let mut a = Arena::new();
        let expr = a.ln(a.one);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "ln(1) should be 0");
    }

    #[test]
    fn eval_ln_e() {
        let mut a = Arena::new();
        let e = a.e_const;
        let expr = a.ln(e);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "ln(E) should be 1");
    }

    // ── sqrt ────────────────────────────────────────────────────────

    #[test]
    fn eval_sqrt_zero() {
        let mut a = Arena::new();
        let expr = a.sqrt(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "sqrt(0) should be 0");
    }

    #[test]
    fn eval_sqrt_one() {
        let mut a = Arena::new();
        let expr = a.sqrt(a.one);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "sqrt(1) should be 1");
    }

    #[test]
    fn eval_sqrt_four() {
        let mut a = Arena::new();
        let four = a.int(4);
        let expr = a.sqrt(four);
        let result = eval(&mut a, expr);
        let two = a.int(2);
        assert_eq!(result, two, "sqrt(4) should be 2");
    }

    #[test]
    fn eval_sqrt_nine() {
        let mut a = Arena::new();
        let nine = a.int(9);
        let expr = a.sqrt(nine);
        let result = eval(&mut a, expr);
        let three = a.int(3);
        assert_eq!(result, three, "sqrt(9) should be 3");
    }

    #[test]
    fn eval_sqrt_two_unchanged() {
        let mut a = Arena::new();
        let two = a.int(2);
        let expr = a.sqrt(two);
        let result = eval(&mut a, expr);
        assert_eq!(
            display(&a, result),
            "sqrt(2)",
            "sqrt(2) should stay unevaluated (irrational)"
        );
    }

    // ── abs ──────────────────────────────────────────────────────────

    #[test]
    fn eval_abs_positive() {
        let mut a = Arena::new();
        let five = a.int(5);
        let expr = a.abs(five);
        let result = eval(&mut a, expr);
        assert_eq!(result, five, "abs(5) should be 5");
    }

    #[test]
    fn eval_abs_negative() {
        let mut a = Arena::new();
        let neg_three = a.int(-3);
        let expr = a.abs(neg_three);
        let result = eval(&mut a, expr);
        let three = a.int(3);
        assert_eq!(result, three, "abs(-3) should be 3");
    }

    #[test]
    fn eval_abs_zero() {
        let mut a = Arena::new();
        let expr = a.abs(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.zero, "abs(0) should be 0");
    }

    #[test]
    fn eval_abs_symbolic_unchanged() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let expr = a.abs(x);
        let result = eval(&mut a, expr);
        assert_eq!(result, expr, "abs(x) should stay unevaluated");
    }

    // ── Composite ───────────────────────────────────────────────────

    #[test]
    fn eval_nested_sin_of_zero_in_add() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x + sin(0) → x + 0 → x
        let sin_zero = a.sin(a.zero);
        let expr = a.add(&[x, sin_zero]);
        let result = eval(&mut a, expr);
        assert_eq!(result, x, "x + sin(0) should simplify to x");
    }

    #[test]
    fn eval_exp_zero_in_mul() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x * exp(0) → x * 1 → x
        let exp_zero = a.exp(a.zero);
        let expr = a.mul(&[x, exp_zero]);
        let result = eval(&mut a, expr);
        assert_eq!(result, x, "x * exp(0) should simplify to x");
    }

    #[test]
    fn eval_cos_pi_in_expression() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let pi = a.pi;
        // x + cos(π) → x + (-1) → -1 + x
        let cos_pi = a.cos(pi);
        let expr = a.add(&[x, cos_pi]);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "-1 + x");
    }

    #[test]
    fn eval_multiple_functions() {
        let mut a = Arena::new();
        // sin(0) + cos(0) + ln(1) = 0 + 1 + 0 = 1
        let sin_0 = a.sin(a.zero);
        let cos_0 = a.cos(a.zero);
        let ln_1 = a.ln(a.one);
        let expr = a.add(&[sin_0, cos_0, ln_1]);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "sin(0) + cos(0) + ln(1) should be 1");
    }

    #[test]
    fn eval_is_idempotent() {
        let mut a = Arena::new();
        let pi = a.pi;
        let expr = a.sin(pi);
        let first = eval(&mut a, expr);
        let second = eval(&mut a, first);
        assert_eq!(first, second, "eval should be idempotent");
    }

    #[test]
    fn eval_sin_pi_over_6() {
        let mut a = Arena::new();
        let sixth = a.rational(1, 6);
        let pi = a.pi;
        let arg = a.mul(&[sixth, pi]);
        let expr = a.sin(arg);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1/2", "sin(π/6) should be 1/2");
    }

    #[test]
    fn eval_cos_pi_over_3() {
        let mut a = Arena::new();
        let third = a.rational(1, 3);
        let pi = a.pi;
        let arg = a.mul(&[third, pi]);
        let expr = a.cos(arg);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1/2", "cos(π/3) should be 1/2");
    }

    #[test]
    fn eval_tan_pi_over_4() {
        let mut a = Arena::new();
        let quarter = a.rational(1, 4);
        let pi = a.pi;
        let arg = a.mul(&[quarter, pi]);
        let expr = a.tan(arg);
        let result = eval(&mut a, expr);
        assert_eq!(result, a.one, "tan(π/4) should be 1");
    }

    #[test]
    fn eval_sqrt_rational_perfect_square() {
        let mut a = Arena::new();
        let nine_fourths = a.rational(9, 4);
        let expr = a.sqrt(nine_fourths);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "3/2", "sqrt(9/4) should be 3/2");
    }
}
