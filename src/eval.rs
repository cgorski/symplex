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
//! - `sin(π/4)` → `√2/2`
//! - `sin(π/3)` → `√3/2`
//! - `cos(π/4)` → `√2/2`
//! - `cos(π/6)` → `√3/2`
//! - `Pow(n, 1/k)` → exact integer when `n` is a perfect `k`th power
//! - `sinh(-x)` → `-sinh(x)` (odd function)
//! - `cosh(-x)` → `cosh(x)` (even function)
//! - `tanh(-x)` → `-tanh(x)` (odd function)
//!
//! # Design
//!
//! Uses the iterative bottom-up walker — no recursion.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};
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
            ExprNode::Asinh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_asinh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Asinh(inner)))
            }
            ExprNode::Acosh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_acosh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Acosh(inner)))
            }
            ExprNode::Atanh(inner) => {
                let inner = cache.get(&inner).copied().unwrap_or(inner);
                eval_atanh(arena, inner).unwrap_or_else(|| arena.intern(ExprNode::Atanh(inner)))
            }
            ExprNode::Sign(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_sign(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.sign(ni)
                }
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
                // Detect Pow(base, 1/n) and try perfect nth root
                if let Some(result) = eval_pow_root(arena, nb, ne) {
                    result
                } else if nb == base && ne == exp {
                    id
                } else {
                    arena.pow(nb, ne)
                }
            }
            ExprNode::Neg(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if ni == inner { id } else { arena.neg(ni) }
            }
            ExprNode::Factorial(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_factorial(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.factorial(ni)
                }
            }
            ExprNode::Binomial(n, k) => {
                let nn = cache.get(&n).copied().unwrap_or(n);
                let nk = cache.get(&k).copied().unwrap_or(k);
                if let Some(result) = eval_binomial(arena, nn, nk) {
                    result
                } else if nn == n && nk == k {
                    id
                } else {
                    arena.binomial(nn, nk)
                }
            }
            // ── Boolean atoms ──────────────────────────────────────
            ExprNode::BoolTrue | ExprNode::BoolFalse => id,

            // ── Relational operators ───────────────────────────────
            ExprNode::Gt(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if let (Some(ra), Some(rb)) = (arena.as_num(na), arena.as_num(nb)) {
                    if ra > rb {
                        arena.bool_true
                    } else {
                        arena.bool_false
                    }
                } else if na == a && nb == b {
                    id
                } else {
                    arena.gt(na, nb)
                }
            }
            ExprNode::Ge(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if let (Some(ra), Some(rb)) = (arena.as_num(na), arena.as_num(nb)) {
                    if ra >= rb {
                        arena.bool_true
                    } else {
                        arena.bool_false
                    }
                } else if na == a && nb == b {
                    id
                } else {
                    arena.ge(na, nb)
                }
            }
            ExprNode::Eq_(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if let (Some(ra), Some(rb)) = (arena.as_num(na), arena.as_num(nb)) {
                    if ra == rb {
                        arena.bool_true
                    } else {
                        arena.bool_false
                    }
                } else if na == a && nb == b {
                    id
                } else {
                    arena.eq_(na, nb)
                }
            }
            ExprNode::Ne(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if let (Some(ra), Some(rb)) = (arena.as_num(na), arena.as_num(nb)) {
                    if ra != rb {
                        arena.bool_true
                    } else {
                        arena.bool_false
                    }
                } else if na == a && nb == b {
                    id
                } else {
                    arena.ne_(na, nb)
                }
            }

            // ── Logical connectives ────────────────────────────────
            ExprNode::And(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // Short-circuit: if any child is BoolFalse → false;
                // filter out BoolTrue; if all removed → true.
                let mut filtered: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
                for &c in &new {
                    if c == arena.bool_false {
                        // entire And is false
                        cache.insert(id, arena.bool_false);
                        continue;
                    }
                    if c == arena.bool_true {
                        continue; // skip trivially true
                    }
                    filtered.push(c);
                }
                // Check if we short-circuited to false
                if new.iter().any(|&c| c == arena.bool_false) {
                    arena.bool_false
                } else if filtered.is_empty() {
                    arena.bool_true
                } else if filtered.len() == 1 {
                    filtered[0]
                } else if filtered == *children {
                    id
                } else {
                    arena.and(&filtered)
                }
            }
            ExprNode::Or(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // Short-circuit: if any child is BoolTrue → true;
                // filter out BoolFalse; if all removed → false.
                if new.iter().any(|&c| c == arena.bool_true) {
                    arena.bool_true
                } else {
                    let filtered: smallvec::SmallVec<[ExprId; 6]> = new
                        .iter()
                        .copied()
                        .filter(|&c| c != arena.bool_false)
                        .collect();
                    if filtered.is_empty() {
                        arena.bool_false
                    } else if filtered.len() == 1 {
                        filtered[0]
                    } else if filtered == *children {
                        id
                    } else {
                        arena.or(&filtered)
                    }
                }
            }
            ExprNode::Not(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if ni == arena.bool_true {
                    arena.bool_false
                } else if ni == arena.bool_false {
                    arena.bool_true
                } else if ni == inner {
                    id
                } else {
                    arena.not(ni)
                }
            }

            // ── Piecewise ──────────────────────────────────────────
            ExprNode::Piecewise(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 6]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // Try to find the first piece whose condition is BoolTrue.
                let mut result: Option<ExprId> = None;
                for i in (0..new.len()).step_by(2) {
                    let cond = new[i + 1];
                    if cond == arena.bool_true {
                        result = Some(new[i]);
                        break;
                    }
                }
                if let Some(val) = result {
                    val
                } else if new == *children {
                    id
                } else {
                    arena.intern(ExprNode::Piecewise(new))
                }
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
fn eval_factorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // No artificial limit — BigInt handles arbitrary precision.
    // EvalConfig guards against runaway computation at a higher level.
    let mut result = num_rational::Ratio::<num_bigint::BigInt>::one();
    for i in 2..=n {
        result *= num_rational::Ratio::from_integer(num_bigint::BigInt::from(i));
    }
    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

fn eval_binomial(arena: &mut Arena, n: ExprId, k: ExprId) -> Option<ExprId> {
    let nr = arena.as_num(n)?;
    let kr = arena.as_num(k)?;
    if !nr.is_integer() || !kr.is_integer() || nr.is_negative() || kr.is_negative() {
        return None;
    }
    let n_u64: u64 = nr.to_integer().try_into().ok()?;
    let k_u64: u64 = kr.to_integer().try_into().ok()?;
    if k_u64 > n_u64 {
        return None;
    }
    // C(n,k) = n! / (k! * (n-k)!)
    let mut result = num_bigint::BigInt::from(1);
    for i in 0..k_u64 {
        result *= num_bigint::BigInt::from(n_u64 - i);
        result /= num_bigint::BigInt::from(i + 1);
    }
    let ratio = num_rational::Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

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
    // Odd function: sin(-x) = -sin(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sin_pos = arena.sin(pos_inner);
        return Some(arena.neg(sin_pos));
    }

    // sin(i*x) = i*sinh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        let sinh_val = arena.sinh(real_part);
        return Some(arena.mul(&[arena.i_unit, sinh_val]));
    }

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
    // sin(π/4) = √2/2
    if coeff == Ratio::new(1.into(), 4.into()) {
        let two = arena.int(2);
        let half_exp = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt2]));
    }
    // sin(π/3) = √3/2
    if coeff == Ratio::new(1.into(), 3.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt3]));
    }

    // ── Quadrant reductions (covers all remaining standard angles) ──
    // Q2: sin(π - x) = sin(x), for coeff in (1/2, 1)
    let half = Ratio::new(1.into(), 2.into());
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_id = arena.mul(&[coeff_id, arena.pi]);
        return eval_sin(arena, reflected_id);
    }
    // Q3: sin(π + x) = -sin(x), for coeff in (1, 3/2)
    let three_half = Ratio::new(3.into(), 2.into());
    if coeff > Ratio::one() && coeff < three_half {
        let reduced = &coeff - Ratio::one();
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_sin(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }
    // Q4: sin(2π - x) = -sin(x), for coeff in (3/2, 2)
    let two: Ratio<BigInt> = Ratio::from_integer(2.into());
    if coeff > three_half && coeff < two {
        let reduced = two - &coeff;
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_sin(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }

    None
}

/// Evaluate `cos(inner)` for known special values.
fn eval_cos(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Even function: cos(-x) = cos(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        return Some(arena.cos(pos_inner));
    }

    // cos(i*x) = cosh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        return Some(arena.cosh(real_part));
    }

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
    // cos(π/4) = √2/2
    if coeff == Ratio::new(1.into(), 4.into()) {
        let two = arena.int(2);
        let half_exp = arena.rational(1, 2);
        let sqrt2 = arena.pow(two, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt2]));
    }
    // cos(π/6) = √3/2
    if coeff == Ratio::new(1.into(), 6.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, sqrt3]));
    }

    // ── Quadrant reductions (covers all remaining standard angles) ──
    // Q2: cos(π - x) = -cos(x), for coeff in (1/2, 1)
    let half = Ratio::new(1.into(), 2.into());
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_cos(arena, reflected_id) {
            return Some(arena.neg(val));
        }
    }
    // Q3: cos(π + x) = -cos(x), for coeff in (1, 3/2)
    let three_half = Ratio::new(3.into(), 2.into());
    if coeff > Ratio::one() && coeff < three_half {
        let reduced = &coeff - Ratio::one();
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        if let Some(val) = eval_cos(arena, reduced_id) {
            return Some(arena.neg(val));
        }
    }
    // Q4: cos(2π - x) = cos(x), for coeff in (3/2, 2)
    let two: Ratio<BigInt> = Ratio::from_integer(2.into());
    if coeff > three_half && coeff < two {
        let reduced = two - &coeff;
        let nid = arena.intern_num(reduced);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reduced_id = arena.mul(&[coeff_id, arena.pi]);
        return eval_cos(arena, reduced_id);
    }

    None
}

/// Evaluate `tan(inner)` for known special values.
fn eval_tan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: tan(-x) = -tan(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let tan_pos = arena.tan(pos_inner);
        return Some(arena.neg(tan_pos));
    }

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
    // tan(π/6) = 1/√3 = √3/3
    if coeff == Ratio::new(1.into(), 6.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        let sqrt3 = arena.pow(three, half_exp);
        let third = arena.rational(1, 3);
        return Some(arena.mul(&[third, sqrt3]));
    }
    // tan(π/3) = √3
    if coeff == Ratio::new(1.into(), 3.into()) {
        let three = arena.int(3);
        let half_exp = arena.rational(1, 2);
        return Some(arena.pow(three, half_exp));
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

    // Euler's formula: exp(i*k*π) = cos(kπ) + i*sin(kπ)
    // Detect if inner is i * (something that's a π-multiple)
    if let Some(pi_coeff) = as_imaginary_pi_multiple(arena, inner) {
        // We have exp(i * coeff * π)
        // = cos(coeff*π) + i*sin(coeff*π)
        // Build the angle as coeff*π and evaluate sin/cos
        let coeff_id = {
            let nid = arena.intern_num(pi_coeff.clone());
            arena.intern(ExprNode::Num(nid))
        };
        let angle = arena.mul(&[coeff_id, arena.pi]);

        let cos_val = eval_cos(arena, angle);
        let sin_val = eval_sin(arena, angle);

        if let (Some(c), Some(s)) = (cos_val, sin_val) {
            // cos(kπ) + i*sin(kπ)
            if s == arena.zero {
                return Some(c); // Pure real
            }
            if c == arena.zero {
                // Pure imaginary: i*sin(kπ)
                return Some(arena.mul(&[arena.i_unit, s]));
            }
            // General: cos + i*sin
            let i_sin = arena.mul(&[arena.i_unit, s]);
            return Some(arena.add(&[c, i_sin]));
        }
    }

    None
}

/// Check if `id` is of the form `i * k * π` for some rational `k`.
/// Returns `Some(k)` if so, `None` otherwise.
fn as_imaginary_pi_multiple(arena: &mut Arena, id: ExprId) -> Option<Ratio<BigInt>> {
    // The canonical form of i * k * π is Mul([k, π, I]) or variations
    // (sorted by sort key: Num < Pi < ImaginaryUnit).
    // We need to check if the expression is a product containing exactly
    // one ImaginaryUnit factor and the rest forming a π-multiple.
    let children = match arena.node(id).clone() {
        ExprNode::Mul(children) => children,
        _ => return None,
    };

    let mut has_i = false;
    let mut remaining: Vec<ExprId> = Vec::new();
    for &child in &children {
        if child == arena.i_unit {
            if has_i {
                return None;
            } // multiple i's
            has_i = true;
        } else {
            remaining.push(child);
        }
    }
    if !has_i {
        return None;
    }

    // Build the product of the remaining factors and check if it's a π-multiple
    let product = if remaining.is_empty() {
        // Just i alone — not a π-multiple
        return None;
    } else if remaining.len() == 1 {
        remaining[0]
    } else {
        arena.mul(&remaining)
    };

    as_pi_multiple(arena, product)
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

    // ln(-1) = i*π
    if inner == arena.neg_one {
        let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
        return Some(i_pi);
    }

    // ln(negative rational) = ln(|r|) + i*π  (principal branch)
    if let Some(r) = arena.as_num(inner)
        && r.is_negative()
    {
        let abs_r = -r.clone();
        if abs_r.is_one() {
            // Already handled above: ln(-1)
        } else {
            let abs_id = {
                let nid = arena.intern_num(abs_r);
                arena.intern(ExprNode::Num(nid))
            };
            let ln_abs = arena.ln(abs_id);
            let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
            return Some(arena.add(&[ln_abs, i_pi]));
        }
    }

    None
}

/// Factor out the largest perfect q-th power from n.
/// Returns (k, m) such that n = k^q * m and m has no q-th power factors.
fn extract_perfect_power(n: &BigInt, q: usize) -> Option<(BigInt, BigInt)> {
    let mut k = BigInt::from(1);
    let mut m = n.clone();
    let mut d = BigInt::from(2);
    while &d * &d <= m {
        let mut power = BigInt::from(1);
        for _ in 0..q {
            power *= &d;
        }
        while (&m % &power).is_zero() {
            m /= &power;
            k *= &d;
        }
        d += 1;
    }
    Some((k, m))
}

/// Evaluate `Pow(base, exp)` when the exponent is a fractional 1/2 (square root)
/// and the base is a numeric value with a perfect square root.
fn eval_pow_root(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    let base_r = arena.as_num(base)?.clone();
    let exp_r = arena.as_num(exp)?.clone();

    // exp must be 1/n for positive integer n ≥ 2
    if *exp_r.numer() != BigInt::from(1) || exp_r.is_negative() {
        return None;
    }
    let n: u32 = exp_r.denom().to_u32()?;
    if n < 2 {
        return None;
    }

    // For integer base
    if base_r.is_integer() && base_r.is_positive() {
        let base_int = base_r.to_integer();
        if let Some(Some(r)) = integer_nth_root(&base_int, n) {
            let nid = arena.intern_num(Ratio::from_integer(r));
            return Some(arena.intern(ExprNode::Num(nid)));
        }
    }

    // For rational base p/q, try root(p)/root(q)
    if base_r.is_positive() && !base_r.is_integer() {
        let numer = base_r.numer().clone();
        let denom = base_r.denom().clone();
        if let (Some(Some(rn)), Some(Some(rd))) =
            (integer_nth_root(&numer, n), integer_nth_root(&denom, n))
        {
            let result = Ratio::new(rn, rd);
            let nid = arena.intern_num(result);
            return Some(arena.intern(ExprNode::Num(nid)));
        }
    }

    // Partial extraction: n^(1/q) → k * m^(1/q) where n = k^q * m
    if exp_r.numer().is_one() && !exp_r.denom().is_one() {
        let q = exp_r.denom().clone();
        let q_usize: usize = (&q).try_into().unwrap_or(0);
        if q_usize >= 2 && base_r.is_integer() && base_r.is_positive() {
            let n = base_r.to_integer();
            if let Some((k, m)) = extract_perfect_power(&n, q_usize)
                && !k.is_one()
            {
                // n^(1/q) = k * m^(1/q)
                let k_id = arena.big_int(k);
                if m.is_one() {
                    return Some(k_id); // perfect power
                }
                let m_id = arena.big_int(m);
                let root = arena.pow(m_id, exp);
                return Some(arena.mul(&[k_id, root]));
            }
        }
    }

    // Handle base == 0 or base == 1 (which as_num covers)
    if base_r.is_zero() {
        return Some(arena.zero);
    }
    if base_r.is_one() {
        return Some(arena.one);
    }

    None
}

/// Try to find the exact integer nth root of `val`.
///
/// Returns `Some(Some(root))` if `val` is a perfect `n`th power,
/// `Some(None)` if it is not a perfect power, and `None` if the input
/// is invalid (e.g. negative).
fn integer_nth_root(val: &BigInt, n: u32) -> Option<Option<BigInt>> {
    if val.is_negative() {
        return None;
    }
    if val.is_zero() {
        return Some(Some(BigInt::from(0)));
    }
    if *val == BigInt::from(1) {
        return Some(Some(BigInt::from(1)));
    }

    // Use Newton's method to find the integer nth root.
    let mut x = val.clone();
    let n_big = BigInt::from(n);
    let n_minus_1 = BigInt::from(n - 1);

    loop {
        // x_new = ((n-1)*x + val / x^(n-1)) / n
        let mut x_pow = BigInt::from(1);
        for _ in 0..(n - 1) {
            x_pow *= &x;
        }
        if x_pow.is_zero() {
            return Some(None);
        }
        let x_new = (&n_minus_1 * &x + val / &x_pow) / &n_big;
        if x_new >= x {
            break;
        }
        x = x_new;
    }

    // Verify: x^n == val?
    let mut check = BigInt::from(1);
    for _ in 0..n {
        check *= &x;
    }
    if check == *val {
        Some(Some(x))
    } else {
        Some(None)
    }
}

/// Evaluate `abs(inner)` for known numeric values.
fn eval_abs(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // abs(i) = 1
    if inner == arena.i_unit {
        return Some(arena.one);
    }
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

fn eval_sign(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(inner) {
        use num_traits::{Signed, Zero};
        if r.is_positive() {
            return Some(arena.one);
        }
        if r.is_negative() {
            return Some(arena.neg_one);
        }
        if r.is_zero() {
            return Some(arena.zero);
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
    // asin(1/2) = π/6
    if let Some(r) = arena.as_num(inner) {
        let half = Ratio::new(1.into(), 2.into());
        if *r == half {
            let sixth = arena.rational(1, 6);
            return Some(arena.mul(&[sixth, arena.pi]));
        }
        let neg_half = Ratio::new((-1).into(), 2.into());
        if *r == neg_half {
            let neg_sixth = arena.rational(-1, 6);
            return Some(arena.mul(&[neg_sixth, arena.pi]));
        }
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
    // acos(1/2) = π/3
    if let Some(r) = arena.as_num(inner) {
        let half = Ratio::new(1.into(), 2.into());
        if *r == half {
            let third = arena.rational(1, 3);
            return Some(arena.mul(&[third, arena.pi]));
        }
        let neg_half = Ratio::new((-1).into(), 2.into());
        if *r == neg_half {
            // acos(-1/2) = 2π/3
            let two_thirds = arena.rational(2, 3);
            return Some(arena.mul(&[two_thirds, arena.pi]));
        }
    }
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
    // atan(1) is already handled (= π/4)
    // atan(-1) is already handled (= -π/4) via odd function
    None
}

fn eval_sinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: sinh(-x) = -sinh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sinh_pos = arena.sinh(pos_inner);
        return Some(arena.neg(sinh_pos));
    }

    if inner == arena.zero {
        return Some(arena.zero);
    } // sinh(0) = 0
    None
}

fn eval_cosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Even function: cosh(-x) = cosh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        return Some(arena.cosh(pos_inner));
    }

    if inner == arena.zero {
        return Some(arena.one);
    } // cosh(0) = 1
    None
}

fn eval_tanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: tanh(-x) = -tanh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let tanh_pos = arena.tanh(pos_inner);
        return Some(arena.neg(tanh_pos));
    }

    if inner == arena.zero {
        return Some(arena.zero);
    } // tanh(0) = 0
    None
}

fn eval_asinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // asinh(0) = 0
    None
}

fn eval_acosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.one {
        return Some(arena.zero);
    } // acosh(1) = 0
    None
}

fn eval_atanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atanh(0) = 0
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if `id` is `i * something` (pure imaginary multiple).
/// Returns the "something" if so.
fn as_pure_imaginary(arena: &Arena, id: ExprId) -> Option<ExprId> {
    if let ExprNode::Mul(ref children) = arena.node(id).clone() {
        let mut has_i = false;
        let mut others: Vec<ExprId> = Vec::new();
        for &child in children {
            if child == arena.i_unit {
                if has_i {
                    return None;
                }
                has_i = true;
            } else {
                others.push(child);
            }
        }
        if has_i {
            return Some(if others.len() == 1 {
                others[0]
            } else if others.is_empty() {
                arena.one // just i alone → real part is 1
            } else {
                // Can't rebuild Mul without &mut Arena, skip multi-factor for now.
                return None;
            });
        }
    }
    None
}

/// Detect if `id` represents a negated expression: `-x` or `Mul(-1, x)`.
/// Returns `Some(positive_inner)` if negated, `None` otherwise.
fn as_negated(arena: &Arena, id: ExprId) -> Option<ExprId> {
    match arena.node(id).clone() {
        ExprNode::Neg(inner) => Some(inner),
        ExprNode::Mul(ref children) if children.len() >= 2 => {
            if let ExprNode::Num(nid) = arena.node(children[0]) {
                let r = arena.num(*nid);
                if r.is_negative() {
                    // Leading negative coefficient: negate it and rebuild.
                    let pos_coeff = -r.clone();
                    if pos_coeff.is_one() {
                        // Mul(-1, rest...) → rest (or Mul(rest...) if multiple)
                        if children.len() == 2 {
                            return Some(children[1]);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

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
    fn eval_cos_pi_over_4() {
        let mut a = Arena::new();
        let quarter = a.rational(1, 4);
        let pi = a.pi;
        let arg = a.mul(&[quarter, pi]);
        let expr = a.cos(arg);
        let result = eval(&mut a, expr);
        // cos(π/4) = √2/2 = 1/2 * sqrt(2)
        let two = a.int(2);
        let half_exp = a.rational(1, 2);
        let sqrt2 = a.pow(two, half_exp);
        let half = a.rational(1, 2);
        let expected = a.mul(&[half, sqrt2]);
        assert_eq!(result, expected, "cos(π/4) should be √2/2");
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
        assert_eq!(display(&a, result), "x + -1");
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

    // ── nth root ────────────────────────────────────────────────────

    #[test]
    fn eval_cube_root_8() {
        let mut a = Arena::new();
        let eight = a.int(8);
        let third = a.rational(1, 3);
        let expr = a.pow(eight, third);
        let result = eval(&mut a, expr);
        let two = a.int(2);
        assert_eq!(result, two, "8^(1/3) should be 2");
    }

    #[test]
    fn eval_cube_root_27() {
        let mut a = Arena::new();
        let twenty_seven = a.int(27);
        let third = a.rational(1, 3);
        let expr = a.pow(twenty_seven, third);
        let result = eval(&mut a, expr);
        let three = a.int(3);
        assert_eq!(result, three, "27^(1/3) should be 3");
    }

    #[test]
    fn eval_fourth_root_16() {
        let mut a = Arena::new();
        let sixteen = a.int(16);
        let quarter = a.rational(1, 4);
        let expr = a.pow(sixteen, quarter);
        let result = eval(&mut a, expr);
        let two = a.int(2);
        assert_eq!(result, two, "16^(1/4) should be 2");
    }

    #[test]
    fn eval_cube_root_7_unchanged() {
        let mut a = Arena::new();
        let seven = a.int(7);
        let third = a.rational(1, 3);
        let expr = a.pow(seven, third);
        let result = eval(&mut a, expr);
        // 7 is not a perfect cube, should stay unevaluated
        assert_eq!(result, expr, "7^(1/3) should stay unevaluated");
    }

    #[test]
    fn eval_cube_root_rational_perfect() {
        let mut a = Arena::new();
        // (27/8)^(1/3) = 3/2
        let base = a.rational(27, 8);
        let third = a.rational(1, 3);
        let expr = a.pow(base, third);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "3/2", "(27/8)^(1/3) should be 3/2");
    }

    // ── irrational trig special values ──────────────────────────────

    #[test]
    fn eval_sin_pi_over_4() {
        let mut a = Arena::new();
        let quarter = a.rational(1, 4);
        let pi = a.pi;
        let arg = a.mul(&[quarter, pi]);
        let expr = a.sin(arg);
        let result = eval(&mut a, expr);
        // sin(π/4) = √2/2 = 1/2 * sqrt(2)
        let two = a.int(2);
        let half_exp = a.rational(1, 2);
        let sqrt2 = a.pow(two, half_exp);
        let half = a.rational(1, 2);
        let expected = a.mul(&[half, sqrt2]);
        assert_eq!(result, expected, "sin(π/4) should be √2/2");
    }

    #[test]
    fn eval_sin_pi_over_3() {
        let mut a = Arena::new();
        let third = a.rational(1, 3);
        let pi = a.pi;
        let arg = a.mul(&[third, pi]);
        let expr = a.sin(arg);
        let result = eval(&mut a, expr);
        // sin(π/3) = √3/2 = 1/2 * sqrt(3)
        let three = a.int(3);
        let half_exp = a.rational(1, 2);
        let sqrt3 = a.pow(three, half_exp);
        let half = a.rational(1, 2);
        let expected = a.mul(&[half, sqrt3]);
        assert_eq!(result, expected, "sin(π/3) should be √3/2");
    }

    #[test]
    fn eval_cos_pi_over_6() {
        let mut a = Arena::new();
        let sixth = a.rational(1, 6);
        let pi = a.pi;
        let arg = a.mul(&[sixth, pi]);
        let expr = a.cos(arg);
        let result = eval(&mut a, expr);
        // cos(π/6) = √3/2 = 1/2 * sqrt(3)
        let three = a.int(3);
        let half_exp = a.rational(1, 2);
        let sqrt3 = a.pow(three, half_exp);
        let half = a.rational(1, 2);
        let expected = a.mul(&[half, sqrt3]);
        assert_eq!(result, expected, "cos(π/6) should be √3/2");
    }

    // ── hyperbolic odd/even ─────────────────────────────────────────

    #[test]
    fn eval_sinh_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let expr = a.sinh(neg_x);
        let result = eval(&mut a, expr);
        // sinh(-x) = -sinh(x)
        let sinh_x = a.sinh(x);
        let expected = a.neg(sinh_x);
        assert_eq!(result, expected, "sinh(-x) should be -sinh(x)");
    }

    #[test]
    fn eval_cosh_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let expr = a.cosh(neg_x);
        let result = eval(&mut a, expr);
        // cosh(-x) = cosh(x)
        let expected = a.cosh(x);
        assert_eq!(result, expected, "cosh(-x) should be cosh(x)");
    }

    #[test]
    fn eval_tanh_neg_x() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let neg_x = a.neg(x);
        let expr = a.tanh(neg_x);
        let result = eval(&mut a, expr);
        // tanh(-x) = -tanh(x)
        let tanh_x = a.tanh(x);
        let expected = a.neg(tanh_x);
        assert_eq!(result, expected, "tanh(-x) should be -tanh(x)");
    }

    // ── Sprint C: quadrant reductions & tan special values ──────────

    #[test]
    fn eval_sin_2pi_over_3() {
        // sin(2π/3) = √3/2
        let mut a = Arena::new();
        let coeff = a.rational(2, 3);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.sin(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.contains("3") && s.contains("1/2"),
            "sin(2π/3) should be √3/2, got: {s}"
        );
    }

    #[test]
    fn eval_sin_3pi_over_4() {
        // sin(3π/4) = √2/2
        let mut a = Arena::new();
        let coeff = a.rational(3, 4);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.sin(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.contains("2") && s.contains("1/2"),
            "sin(3π/4) should be √2/2, got: {s}"
        );
    }

    #[test]
    fn eval_cos_3pi_over_4() {
        // cos(3π/4) = -√2/2
        let mut a = Arena::new();
        let coeff = a.rational(3, 4);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.cos(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.contains("2") && s.contains("1/2"),
            "cos(3π/4) should be -√2/2, got: {s}"
        );
    }

    #[test]
    fn eval_cos_5pi_over_6() {
        // cos(5π/6) = -√3/2
        let mut a = Arena::new();
        let coeff = a.rational(5, 6);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.cos(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(
            s.contains("3") && s.contains("1/2"),
            "cos(5π/6) should be -√3/2, got: {s}"
        );
    }

    #[test]
    fn eval_tan_pi_over_6() {
        // tan(π/6) = √3/3
        let mut a = Arena::new();
        let coeff = a.rational(1, 6);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.tan(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("3"), "tan(π/6) should involve √3, got: {s}");
    }

    #[test]
    fn eval_tan_pi_over_3() {
        // tan(π/3) = √3
        let mut a = Arena::new();
        let coeff = a.rational(1, 3);
        let angle = a.mul(&[coeff, a.pi]);
        let expr = a.tan(angle);
        let result = eval(&mut a, expr);
        let s = display(&a, result);
        assert!(s.contains("3"), "tan(π/3) should be √3, got: {s}");
    }

    // ── Euler's formula / complex eval ──────────────────────────────

    #[test]
    fn eval_exp_i_pi() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let pi = a.pi;
        let i_pi = a.mul(&[i, pi]);
        let expr = a.exp(i_pi);
        let result = eval(&mut a, expr);
        // exp(i*π) = -1
        assert_eq!(display(&a, result), "-1");
    }

    #[test]
    fn eval_exp_i_pi_over_2() {
        let mut a = Arena::new();
        let i = a.i_unit;
        let half = a.rational(1, 2);
        let pi = a.pi;
        let half_pi = a.mul(&[half, pi]);
        let i_half_pi = a.mul(&[i, half_pi]);
        let expr = a.exp(i_half_pi);
        let result = eval(&mut a, expr);
        // exp(i*π/2) = i
        assert_eq!(display(&a, result), "I");
    }

    #[test]
    fn eval_abs_i() {
        let mut a = Arena::new();
        let expr = a.abs(a.i_unit);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    // ── ln of negatives ─────────────────────────────────────────────

    #[test]
    fn eval_ln_neg_one() {
        let mut a = Arena::new();
        let expr = a.ln(a.neg_one);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "pi*I");
    }

    #[test]
    fn eval_ln_neg_two() {
        let mut a = Arena::new();
        let neg_two = a.rational(-2, 1);
        let expr = a.ln(neg_two);
        let result = eval(&mut a, expr);
        let two = a.rational(2, 1);
        let ln2 = a.ln(two);
        let i_pi = a.mul(&[a.i_unit, a.pi]);
        let expected = a.add(&[ln2, i_pi]);
        assert_eq!(display(&a, result), display(&a, expected));
    }

    // ── inverse trig ────────────────────────────────────────────────

    #[test]
    fn eval_asin_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let expr = a.asin(half);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1/6*pi");
    }

    #[test]
    fn eval_asin_neg_half() {
        let mut a = Arena::new();
        let neg_half = a.rational(-1, 2);
        let expr = a.asin(neg_half);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "-1/6*pi");
    }

    #[test]
    fn eval_acos_half() {
        let mut a = Arena::new();
        let half = a.rational(1, 2);
        let expr = a.acos(half);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1/3*pi");
    }

    #[test]
    fn eval_acos_neg_half() {
        let mut a = Arena::new();
        let neg_half = a.rational(-1, 2);
        let expr = a.acos(neg_half);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "2/3*pi");
    }

    // ── sqrt partial extraction ─────────────────────────────────────

    #[test]
    fn eval_sqrt_8() {
        let mut a = Arena::new();
        let eight = a.int(8);
        let half = a.rational(1, 2);
        let expr = a.pow(eight, half);
        let result = eval(&mut a, expr);
        let d = display(&a, result);
        assert!(
            d.contains("2") && d != "8^(1/2)",
            "sqrt(8) should simplify to 2*sqrt(2), got: {d}"
        );
    }

    #[test]
    fn eval_sqrt_12() {
        let mut a = Arena::new();
        let twelve = a.int(12);
        let half = a.rational(1, 2);
        let expr = a.pow(twelve, half);
        let result = eval(&mut a, expr);
        let d = display(&a, result);
        assert!(
            d.contains("2") && d.contains("3"),
            "sqrt(12) should simplify to 2*sqrt(3), got: {d}"
        );
    }

    #[test]
    fn eval_sqrt_50() {
        let mut a = Arena::new();
        let fifty = a.int(50);
        let half = a.rational(1, 2);
        let expr = a.pow(fifty, half);
        let result = eval(&mut a, expr);
        let d = display(&a, result);
        assert!(
            d.contains("5") && d.contains("2"),
            "sqrt(50) should simplify to 5*sqrt(2), got: {d}"
        );
    }

    // ── trig-hyperbolic bridge ──────────────────────────────────────

    #[test]
    fn eval_sin_ix() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ix = a.mul(&[a.i_unit, x]);
        let expr = a.sin(ix);
        let result = eval(&mut a, expr);
        let d = display(&a, result);
        assert!(
            d.contains("sinh") && d.contains("I"),
            "sin(i*x) should be i*sinh(x), got: {d}"
        );
    }

    #[test]
    fn eval_cos_ix() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let ix = a.mul(&[a.i_unit, x]);
        let expr = a.cos(ix);
        let result = eval(&mut a, expr);
        let d = display(&a, result);
        assert!(d.contains("cosh"), "cos(i*x) should be cosh(x), got: {d}");
    }

    #[test]
    fn eval_factorial_5() {
        let mut a = Arena::new();
        let five = a.int(5);
        let expr = a.factorial(five);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "120");
    }

    #[test]
    fn eval_factorial_0() {
        let mut a = Arena::new();
        let zero = a.zero;
        let expr = a.factorial(zero);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn eval_binomial_5_2() {
        let mut a = Arena::new();
        let five = a.int(5);
        let two = a.int(2);
        let expr = a.binomial(five, two);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "10");
    }

    #[test]
    fn eval_sign_positive() {
        let mut a = Arena::new();
        let five = a.int(5);
        let expr = a.sign(five);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "1");
    }

    #[test]
    fn eval_sign_negative() {
        let mut a = Arena::new();
        let neg = a.int(-3);
        let expr = a.sign(neg);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "-1");
    }

    #[test]
    fn eval_sign_zero() {
        let mut a = Arena::new();
        let expr = a.sign(a.zero);
        let result = eval(&mut a, expr);
        assert_eq!(display(&a, result), "0");
    }
}
