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

use crate::base::arena::Arena;
use crate::base::libfn::LibFn;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;

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
            ExprNode::Atan2(y, x) => {
                let ny = cache.get(&y).copied().unwrap_or(y);
                let nx = cache.get(&x).copied().unwrap_or(x);
                if let Some(result) = eval_atan2(arena, ny, nx) {
                    result
                } else if ny == y && nx == x {
                    id
                } else {
                    arena.atan2(ny, nx)
                }
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
            ExprNode::Heaviside(inner) => {
                let evaled = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(r) = arena.as_num(evaled).cloned() {
                    if r.is_positive() {
                        arena.one
                    } else if r.is_negative() {
                        arena.zero
                    } else {
                        // r.is_zero() — convention: Heaviside(0) = 1/2
                        arena.rational(1, 2)
                    }
                } else if evaled == inner {
                    id
                } else {
                    arena.intern(ExprNode::Heaviside(evaled))
                }
            }
            ExprNode::DiracDelta(inner) => {
                let evaled = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(r) = arena.as_num(evaled).cloned() {
                    if !r.is_zero() {
                        arena.zero
                    } else {
                        // At zero, leave unevaluated
                        if evaled == inner {
                            id
                        } else {
                            arena.intern(ExprNode::DiracDelta(evaled))
                        }
                    }
                } else if evaled == inner {
                    id
                } else {
                    arena.intern(ExprNode::DiracDelta(evaled))
                }
            }
            ExprNode::Gamma(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_gamma(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.gamma(ni)
                }
            }
            ExprNode::LogGamma(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_log_gamma(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.log_gamma(ni)
                }
            }
            ExprNode::Digamma(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_digamma(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.digamma(ni)
                }
            }

            // ── Complex analysis: always re-run the canonical constructor so
            //    that newly evaluated children (or assumptions) fold. ─────────
            ExprNode::Re(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.re(ni)
            }
            ExprNode::Im(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.im(ni)
            }
            ExprNode::Conjugate(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.conjugate(ni)
            }
            ExprNode::Arg(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.arg(ni)
            }

            // ── Special functions (0.2): constructors fold exact values ───────
            ExprNode::Si(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.si(ni)
            }
            ExprNode::Ci(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.ci(ni)
            }
            ExprNode::Ei(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.ei(ni)
            }
            ExprNode::Li(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.li(ni)
            }
            ExprNode::Zeta(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                arena.zeta(ni)
            }
            ExprNode::Polygamma(n, x) => {
                let nn = cache.get(&n).copied().unwrap_or(n);
                let nx = cache.get(&x).copied().unwrap_or(x);
                arena.polygamma(nn, nx)
            }
            ExprNode::KroneckerDelta(i, j) => {
                let ni = cache.get(&i).copied().unwrap_or(i);
                let nj = cache.get(&j).copied().unwrap_or(j);
                arena.kronecker_delta(ni, nj)
            }
            ExprNode::Erf(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_erf(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.erf(ni)
                }
            }
            ExprNode::Erfc(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_erfc(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.erfc(ni)
                }
            }
            ExprNode::LambertW(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_lambertw(arena, ni) {
                    result
                } else if ni == inner {
                    id
                } else {
                    arena.lambertw(ni)
                }
            }
            ExprNode::Beta(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                if let Some(result) = eval_beta(arena, na, nb) {
                    result
                } else if na == a && nb == b {
                    id
                } else {
                    arena.beta(na, nb)
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
                }
                // Pow(Exp(f), g) → Exp(f·g) when that is the principal power
                // (see `exp_pow_merges`: integer g, or real f).
                //
                // Critical for the Gruntz algorithm: without this, exp(x)^(1/x)
                // stays as Pow(Exp(x), 1/x) and mrv creates dangling dummy
                // variables when trying to rewrite via exp((1/x)·ln(exp(x))).
                // Gruntz's variable is a positive dummy, so the rule applies
                // there: exp(x)^(1/x) = Exp(x·(1/x)) = Exp(1).
                //
                // Placed in eval (not canon_pow) so the solver's intermediate
                // Pow(Exp(x), k) nodes survive until substitution completes.
                else if let ExprNode::Exp(inner) = arena.node(nb).clone()
                    && exp_pow_merges(arena, inner, ne)
                {
                    // Re-evaluate the new argument so `exp(-1)^(-1)` becomes
                    // `E`, not an `exp(1)` that a second `eval` would fold.
                    let product = arena.mul(&[inner, ne]);
                    let product = eval(arena, product);
                    eval_exp(arena, product).unwrap_or_else(|| arena.exp(product))
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
                } else if let Some(holds) = order_against_zero(arena, na, nb, true) {
                    if holds {
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
                } else if let Some(holds) = order_against_zero(arena, na, nb, false) {
                    if holds {
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
                if new.contains(&arena.bool_false) {
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
                if new.contains(&arena.bool_true) {
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
            //
            // Branches are examined *in order*.  A `False` condition is
            // dropped, a `True` condition selects its branch and makes
            // every later branch unreachable, and an undecided condition
            // stops the search — a later `True` must never shadow an
            // earlier branch whose condition is still open.
            ExprNode::Piecewise(ref pairs) => {
                let t = arena.bool_true;
                let f = arena.bool_false;
                let mut kept: smallvec::SmallVec<[(ExprId, ExprId); 3]> = smallvec::SmallVec::new();
                for &(val, cond) in pairs.iter() {
                    let nv = cache.get(&val).copied().unwrap_or(val);
                    let nc = cache.get(&cond).copied().unwrap_or(cond);
                    if nc == f {
                        continue;
                    }
                    kept.push((nv, nc));
                    if nc == t {
                        break;
                    }
                }
                if kept.is_empty() {
                    // Every branch was ruled out: the function is undefined.
                    arena.nan
                } else if kept.len() == 1 && kept[0].1 == t {
                    kept[0].0
                } else if kept == *pairs {
                    id
                } else {
                    arena.intern(ExprNode::Piecewise(kept))
                }
            }

            // ── Apply (library special functions) ──────────────────
            ExprNode::Apply(name_sid, ref args) => {
                let new_args: smallvec::SmallVec<[ExprId; 2]> = args
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // A library function with the right number of arguments may
                // fold to an exact value; anything else — a user function or
                // a wrong arity — is rebuilt on its evaluated arguments.
                let folded = match arena.lib_fn(name_sid) {
                    Some(f) if f.arity().accepts(new_args.len()) => {
                        eval_lib_fn(arena, f, &new_args)
                    }
                    _ => None,
                };
                match folded {
                    Some(result) => result,
                    None if new_args[..] == args[..] => id,
                    None => arena.intern(ExprNode::Apply(name_sid, new_args)),
                }
            }

            // ── Floor ──────────────────────────────────────────────
            ExprNode::Floor(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(r) = arena.as_num(ni).cloned() {
                    if r.is_integer() {
                        ni
                    } else {
                        let p = r.numer().clone();
                        let q = r.denom().clone();
                        // Rust BigInt `/` truncates toward zero; adjust for negative
                        let floor_val = if p.is_negative() && !(&p % &q).is_zero() {
                            &p / &q - BigInt::from(1)
                        } else {
                            &p / &q
                        };
                        arena.big_int(floor_val)
                    }
                } else if ni == inner {
                    id
                } else {
                    arena.floor(ni)
                }
            }

            // ── Ceiling ────────────────────────────────────────────
            ExprNode::Ceiling(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(r) = arena.as_num(ni).cloned() {
                    if r.is_integer() {
                        ni
                    } else {
                        let p = r.numer().clone();
                        let q = r.denom().clone();
                        // Rust BigInt `/` truncates toward zero; adjust for positive
                        let ceil_val = if p.is_positive() && !(&p % &q).is_zero() {
                            &p / &q + BigInt::from(1)
                        } else {
                            &p / &q
                        };
                        arena.big_int(ceil_val)
                    }
                } else if ni == inner {
                    id
                } else {
                    arena.ceiling(ni)
                }
            }

            // ── Min ────────────────────────────────────────────────
            ExprNode::Min(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 4]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // If all args are numeric, return the smallest.
                let all_numeric: Option<Vec<(Q, ExprId)>> = new
                    .iter()
                    .map(|&c| arena.as_num(c).cloned().map(|r| (r, c)))
                    .collect();
                if let Some(nums) = all_numeric {
                    if let Some((_, min_id)) = nums.iter().min_by_key(|(r, _)| r.clone()) {
                        *min_id
                    } else {
                        id
                    }
                } else if new[..] == children[..] {
                    id
                } else {
                    arena.intern(ExprNode::Min(new))
                }
            }

            // ── Max ────────────────────────────────────────────────
            ExprNode::Max(ref children) => {
                let new: smallvec::SmallVec<[ExprId; 4]> = children
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // If all args are numeric, return the largest.
                let all_numeric: Option<Vec<(Q, ExprId)>> = new
                    .iter()
                    .map(|&c| arena.as_num(c).cloned().map(|r| (r, c)))
                    .collect();
                if let Some(nums) = all_numeric {
                    if let Some((_, max_id)) = nums.iter().max_by_key(|(r, _)| r.clone()) {
                        *max_id
                    } else {
                        id
                    }
                } else if new[..] == children[..] {
                    id
                } else {
                    arena.intern(ExprNode::Max(new))
                }
            }

            // ── Sum: try closed-form first, then finite substitution ───
            ExprNode::Sum(body, sum_var, lo, hi) => {
                let nbody = cache.get(&body).copied().unwrap_or(body);
                let nvar = cache.get(&sum_var).copied().unwrap_or(sum_var);
                let nlo = cache.get(&lo).copied().unwrap_or(lo);
                let nhi = cache.get(&hi).copied().unwrap_or(hi);

                let lo_int = arena.as_num(nlo).and_then(|r| {
                    if r.is_integer() {
                        r.numer().to_i64()
                    } else {
                        None
                    }
                });
                let hi_int = arena.as_num(nhi).and_then(|r| {
                    if r.is_integer() {
                        r.numer().to_i64()
                    } else {
                        None
                    }
                });

                // Check for empty range before anything else
                if let (Some(lo_val), Some(hi_val)) = (lo_int, hi_int)
                    && hi_val < lo_val
                {
                    // Empty range: sum is 0
                    cache.insert(id, arena.zero);
                    continue;
                }

                // Try closed-form symbolic evaluation (Faulhaber, geometric, etc.)
                if let Some(closed) =
                    crate::calculus::sum_eval::eval_sum_symbolic(arena, nbody, nvar, nlo, nhi)
                {
                    let result = eval(arena, closed);
                    cache.insert(id, result);
                    continue;
                }

                if let (Some(lo_val), Some(hi_val)) = (lo_int, hi_int) {
                    if (hi_val - lo_val) <= 1000 {
                        let mut terms = smallvec::SmallVec::<[ExprId; 8]>::new();
                        for k in lo_val..=hi_val {
                            let k_expr = arena.int(k);
                            let substituted = arena.subs_structural(nbody, nvar, k_expr);
                            let evaluated_term = eval(arena, substituted);
                            terms.push(evaluated_term);
                        }
                        if terms.is_empty() {
                            arena.zero
                        } else {
                            arena.add(&terms)
                        }
                    } else if nbody == body && nvar == sum_var && nlo == lo && nhi == hi {
                        id
                    } else {
                        arena.intern(ExprNode::Sum(nbody, nvar, nlo, nhi))
                    }
                } else if nbody == body && nvar == sum_var && nlo == lo && nhi == hi {
                    id
                } else {
                    arena.intern(ExprNode::Sum(nbody, nvar, nlo, nhi))
                }
            }

            // ── Product_: evaluate by substitution for finite integer bounds
            ExprNode::Product_(body, prod_var, lo, hi) => {
                let nbody = cache.get(&body).copied().unwrap_or(body);
                let nvar = cache.get(&prod_var).copied().unwrap_or(prod_var);
                let nlo = cache.get(&lo).copied().unwrap_or(lo);
                let nhi = cache.get(&hi).copied().unwrap_or(hi);

                let lo_int = arena.as_num(nlo).and_then(|r| {
                    if r.is_integer() {
                        r.numer().to_i64()
                    } else {
                        None
                    }
                });
                let hi_int = arena.as_num(nhi).and_then(|r| {
                    if r.is_integer() {
                        r.numer().to_i64()
                    } else {
                        None
                    }
                });

                // Try closed-form symbolic evaluation first (Π k = n!, telescoping,
                // Π a^f(k) = a^Σf, …), mirroring the `Sum` arm above.
                if let Some(closed) =
                    crate::calculus::sum_eval::eval_product_symbolic(arena, nbody, nvar, nlo, nhi)
                {
                    let result = eval(arena, closed);
                    cache.insert(id, result);
                    continue;
                }

                if let (Some(lo_val), Some(hi_val)) = (lo_int, hi_int) {
                    if hi_val < lo_val {
                        // Empty range: product is 1
                        arena.one
                    } else if (hi_val - lo_val) <= 1000 {
                        let mut factors = smallvec::SmallVec::<[ExprId; 8]>::new();
                        for k in lo_val..=hi_val {
                            let k_expr = arena.int(k);
                            let substituted = arena.subs_structural(nbody, nvar, k_expr);
                            let evaluated_term = eval(arena, substituted);
                            factors.push(evaluated_term);
                        }
                        if factors.is_empty() {
                            arena.one
                        } else {
                            arena.mul(&factors)
                        }
                    } else if nbody == body && nvar == prod_var && nlo == lo && nhi == hi {
                        id
                    } else {
                        arena.intern(ExprNode::Product_(nbody, nvar, nlo, nhi))
                    }
                } else if nbody == body && nvar == prod_var && nlo == lo && nhi == hi {
                    id
                } else {
                    arena.intern(ExprNode::Product_(nbody, nvar, nlo, nhi))
                }
            }

            // ── DefiniteIntegral: rebuild with evaluated children ────────
            // `eval` stays cheap: only the constructor's structural folds
            // (`lo == hi`, constant integrand, reversed numeric bounds) are
            // applied.  Actual integration is `definite::integrate_definite`.
            ExprNode::DefiniteIntegral(body, int_var, lo, hi) => {
                let nbody = cache.get(&body).copied().unwrap_or(body);
                let nvar = cache.get(&int_var).copied().unwrap_or(int_var);
                let nlo = cache.get(&lo).copied().unwrap_or(lo);
                let nhi = cache.get(&hi).copied().unwrap_or(hi);
                // Always go through the constructor so that directly
                // interned nodes pick up the structural folds too.
                arena.definite_integral(nbody, nvar, nlo, nhi)
            }

            // Everything else: unchanged.
            // ── RootSum: try to expand by solving the polynomial ───
            ExprNode::RootSum(poly, body, sumvar) => {
                let poly = cache.get(&poly).copied().unwrap_or(poly);
                let body = cache.get(&body).copied().unwrap_or(body);
                let sumvar = cache.get(&sumvar).copied().unwrap_or(sumvar);
                // Try to expand: solve poly, substitute each root into body, sum.
                // This succeeds for degree ≤ 4 (exact radicals via Cardano/Ferrari).
                if let Some(expanded) =
                    crate::calculus::risch::log_to_real::rootsum_doit(arena, poly, body, sumvar)
                {
                    tracing::debug!("eval: RootSum expanded via rootsum_doit");
                    expanded
                } else if let Some(rational_val) =
                    crate::calculus::risch::log_to_real::vieta_rootsum_poly_body(
                        arena, poly, body, sumvar,
                    )
                {
                    // Body is a polynomial in sumvar — Vieta's formulas give
                    // the exact rational sum without finding roots.
                    tracing::debug!("eval: RootSum evaluated via Vieta's formulas");
                    let num_id = arena.intern_num(rational_val);
                    arena.intern(ExprNode::Num(num_id))
                } else {
                    // Can't expand — rebuild with evaluated children.
                    arena.intern(ExprNode::RootSum(poly, body, sumvar))
                }
            }

            _ => id,
        };

        cache.insert(id, evaluated);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Special value tables
// ═══════════════════════════════════════════════════════════════════════════

// ── The digit guard of exact products ────────────────────────────────────
//
// `n!`, `Γ(p/q)` (a rising factorial times `Γ` of the fractional part),
// `C(n, k)`, `(x)ₙ` and `x^(n)` are exact rationals of about `n·log₁₀ n`
// digits.  They follow canon's policy for integer powers: a result beyond
// `max_result_digits` digits is not computed and the node stays symbolic,
// for `evalf` to evaluate numerically.  A lower bound on the digits refuses
// before any multiplication; the exact count of the product decides at the
// limit.  Before 0.29 nothing was bounded and every factor went through a
// `Ratio` multiplication (a gcd per step): `Γ(1000 + 1/3)` took 2.9 s in
// eval (debug), `Γ(3000 + 1/3)`, `3000!` and `ln Γ(3000)` more than a minute,
// so `uppergamma(s, x)/gamma(s)` hung before `evalf` ran.

/// A lower bound on `log₁₀ n!` (`n! ≥ (n/e)ⁿ`), for `n ≥ 0`.
fn log10_factorial_lower(n: f64) -> f64 {
    if n < 1.0 {
        0.0
    } else {
        n * (n / std::f64::consts::E).log10().max(0.0)
    }
}

/// Is an exact result of at least `digits` decimal digits (a lower bound)
/// beyond the digit guard?  `NaN` and `∞` are.
fn beyond_digit_guard(arena: &Arena, digits: f64) -> bool {
    digits.is_nan() || digits > arena.config.max_result_digits as f64
}

/// `r` as a node when its numerator and denominator together have at most
/// `max_result_digits` digits (canon's count), otherwise `None`.
fn guarded_num(arena: &mut Arena, r: Q) -> Option<ExprId> {
    // A b-bit integer has ⌊b·log₁₀2⌋ or one more digits: the bit counts
    // decide unless the total is within two digits of the limit, where the
    // decimal strings are counted (they cost more than the arithmetic for
    // the thousand coefficients of a degree-1000 polynomial).
    let limit = arena.config.max_result_digits as f64;
    let est = (r.numer().bits() + r.denom().bits()) as f64 * std::f64::consts::LOG10_2;
    if est > limit + 2.0 {
        return None;
    }
    if est >= limit - 2.0 {
        let digits = r.numer().to_string().len() + r.denom().to_string().len();
        if digits > arena.config.max_result_digits {
            return None;
        }
    }
    let nid = arena.intern_num(r);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// `∏_{i<n} (a + i·d)` for integers `a`, `d`.
fn progression_product(a: &BigInt, d: &BigInt, n: u64) -> BigInt {
    let mut acc = BigInt::one();
    let mut term = a.clone();
    for _ in 0..n {
        acc *= &term;
        term += d;
    }
    acc
}

/// A lower bound on the decimal digits of `∏_{i<n} |a + i·d|` (`d ≠ 0`, no
/// factor zero): the factors are distinct non-zero integers `|d|` apart, so
/// the `j`-th smallest of them is at least `|d|·⌊j/2⌋`, and when they all
/// have the sign of `a` (`a·d > 0`) the `j`-th is at least `|d|·j` (`j ≥ 1`).
fn progression_digits_lower(a: &BigInt, d: &BigInt, n: u64) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let n = n as f64;
    let log_d = d.abs().to_f64().unwrap_or(f64::INFINITY).log10();
    if a.sign() == d.sign() {
        log10_factorial_lower(n - 1.0) + (n - 1.0) * log_d
    } else {
        2.0 * log10_factorial_lower(((n - 1.0) / 2.0).floor()) + (n - 2.0).max(0.0) * log_d
    }
}

/// `(p/q)ₙ = ∏_{i<n} (p/q + i) = ∏(p + i·q)/qⁿ` (`step = 1`) or the falling
/// `∏_{i<n} (p/q − i)` (`step = −1`) exactly, `None` beyond the digit guard.
/// Numerator and denominator are coprime (`gcd(p + i·q, q) = gcd(p, q) = 1`),
/// so no gcd is taken.
fn exact_factorial_power(arena: &mut Arena, x: &Q, n: u64, step: i32) -> Option<ExprId> {
    let (p, q) = (x.numer(), x.denom());
    let d = q * BigInt::from(step);
    // A zero factor: x a non-positive (rising) or non-negative (falling)
    // integer within n steps.
    if x.is_integer() {
        let reach = BigInt::from(n) - BigInt::one();
        let hits_zero = if step > 0 {
            !p.is_positive() && -p <= reach
        } else {
            !p.is_negative() && *p <= reach
        };
        if hits_zero {
            return Some(arena.zero);
        }
    }
    let q_digits = (n as f64) * q.to_f64().unwrap_or(f64::INFINITY).log10();
    if beyond_digit_guard(arena, progression_digits_lower(p, &d, n) + q_digits) {
        return None;
    }
    let numer = progression_product(p, &d, n);
    let denom = num_traits::Pow::pow(q.clone(), n);
    guarded_num(arena, Ratio::new_raw(numer, denom))
}

/// `n!` for a non-negative integer literal `n`, within the digit guard.
fn eval_factorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    if beyond_digit_guard(arena, log10_factorial_lower(n as f64)) {
        return None;
    }
    let result = Ratio::from_integer(crate::base::combinatorics::factorial(n));
    guarded_num(arena, result)
}

/// Gamma(n) for positive integer n → (n-1)!
/// Gamma(1/2) → √π
///
/// Every expansion is bounded by the digit guard (see
/// [`beyond_digit_guard`]); beyond it the node stays symbolic.
fn eval_gamma(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?.clone();

    // Half-integer: Gamma(p/2) for positive odd p.
    // Uses: Gamma((2k+1)/2) = (2k-1)!! / 2^k · √π
    // where (2k-1)!! = 1·3·5·…·(2k-1) (empty product = 1 when k=0).
    if *r.denom() == BigInt::from(2) && r.is_positive() {
        // p is guaranteed odd because the fraction is in lowest terms with denom 2.
        let p = r.numer().clone();
        // k = (p - 1) / 2
        let k_big = (&p - BigInt::from(1)) / BigInt::from(2);
        let k: u64 = k_big.try_into().ok()?;
        // (2k−1)!! ≥ k!, and the denominator 2^k.
        let kf = k as f64;
        if beyond_digit_guard(
            arena,
            log10_factorial_lower(kf) + kf * std::f64::consts::LOG10_2,
        ) {
            return None;
        }

        // Compute (2k-1)!! = product of odd numbers 1, 3, 5, …, 2k-1.
        let mut double_fact = BigInt::from(1);
        for i in 0..k {
            double_fact *= BigInt::from(2 * i + 1);
        }

        // coefficient = (2k-1)!! / 2^k
        let two_pow_k = BigInt::from(1) << (k as usize);
        let coeff = Ratio::new(double_fact, two_pow_k);

        if coeff.is_one() {
            let pi = arena.pi;
            return Some(arena.sqrt(pi));
        }

        let coeff_node = guarded_num(arena, coeff)?;
        let pi = arena.pi;
        let sqrt_pi = arena.sqrt(pi);
        let result = arena.mul(&[coeff_node, sqrt_pi]);
        return Some(result);
    }

    // Positive integer: Gamma(n) = (n-1)!
    if r.is_integer() && r.is_positive() {
        let n: u64 = r.to_integer().try_into().ok()?;
        if beyond_digit_guard(arena, log10_factorial_lower((n - 1) as f64)) {
            return None;
        }
        let result = Ratio::from_integer(crate::base::combinatorics::factorial(n - 1));
        return guarded_num(arena, result);
    }

    // Gamma recurrence for positive rationals > 1 with denom ∉ {1, 2}:
    // Gamma(frac + n) = rising_factorial(frac, n) * Gamma(frac)
    // where frac ∈ (0,1) and n ≥ 1.
    if r.is_positive() && !r.is_integer() && *r.denom() != BigInt::from(2) {
        let n_big = r.to_integer(); // floor for positive values
        if n_big >= BigInt::from(1) {
            let n: u64 = n_big.clone().try_into().ok()?;
            let frac = &r - Ratio::from_integer(n_big);
            // rising_factorial(frac, n) = frac * (frac+1) * … * (frac+n-1)
            let prod_node = exact_factorial_power(arena, &frac, n, 1)?;
            let frac_id = {
                let nid = arena.intern_num(frac);
                arena.intern(ExprNode::Num(nid))
            };
            // Try to evaluate the inner Gamma (e.g. Gamma(1/2) → √π)
            let gamma_frac = eval_gamma(arena, frac_id).unwrap_or_else(|| arena.gamma(frac_id));
            if prod_node == arena.one {
                return Some(gamma_frac);
            }
            return Some(arena.mul(&[prod_node, gamma_frac]));
        }
    }

    // Gamma recurrence for negative non-integer rationals:
    // Gamma(x) = Gamma(x+m) / (x·(x+1)·…·(x+m-1))
    // where m = ⌈-x⌉ shifts x into (0, 1).
    if r.is_negative() && !r.is_integer() {
        let neg_r = -&r; // positive
        // m = ceil(neg_r); since neg_r is positive and not an integer,
        // ceil = floor + 1 = to_integer + 1.
        let m_big = if neg_r.is_integer() {
            neg_r.to_integer()
        } else {
            neg_r.to_integer() + BigInt::from(1)
        };
        let m: u64 = m_big.clone().try_into().ok()?;
        let frac = &r + Ratio::from_integer(m_big); // frac ∈ (0, 1)
        // Denominator product: x·(x+1)·…·(x+m-1), non-zero as r is not an
        // integer.
        let denom_node = exact_factorial_power(arena, &r, m, 1)?;
        let denom_product = arena.as_num(denom_node)?.clone();
        if denom_product.is_zero() {
            return None;
        }
        let frac_id = {
            let nid = arena.intern_num(frac);
            arena.intern(ExprNode::Num(nid))
        };
        // Try to evaluate the inner Gamma (handles half-integer base, etc.)
        let gamma_frac = eval_gamma(arena, frac_id).unwrap_or_else(|| arena.gamma(frac_id));
        // Gamma(r) = Gamma(frac) / denom_product = (1/denom_product) * Gamma(frac)
        let inv_denom = Ratio::one() / denom_product;
        let coeff_nid = arena.intern_num(inv_denom);
        let coeff_node = arena.intern(ExprNode::Num(coeff_nid));
        return Some(arena.mul(&[coeff_node, gamma_frac]));
    }

    None
}

/// `a > 0`, `0 > b` (`strict`) or `a ≥ 0`, `0 ≥ b` for a constant (free of
/// symbols) `a` or `b` whose sign the assumption engine knows from its
/// structure — `exp` of a real is positive, a square of a real is
/// non-negative, … — decided before any numerical evaluation.  `None` when
/// neither side is 0 or the sign is not known.
///
/// Before 0.29 only two rationals were compared: `Piecewise((1, exp(−4·10⁹)
/// > 0), (0, True))` was left to `evalf`, whose `exp(−4·10⁹)` underflows, and
/// was refused (`PrecisionExhausted`).
fn order_against_zero(arena: &Arena, a: ExprId, b: ExprId, strict: bool) -> Option<bool> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let is_zero = |id: ExprId| arena.as_num(id).is_some_and(Zero::is_zero);
    // `x > 0` / `x ≥ 0` with the sign of `x`; `0 > x` is `−x > 0`.
    let (x, flipped) = if is_zero(b) {
        (a, false)
    } else if is_zero(a) {
        (b, true)
    } else {
        return None;
    };
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let mut signs = AssumptionCache::new();
    let mut known = |p: Props| signs.query(arena, x, p) == Some(true);
    let (holds, fails) = match (strict, flipped) {
        (true, false) => (Props::POSITIVE, Props::NONPOSITIVE),
        (false, false) => (Props::NONNEGATIVE, Props::NEGATIVE),
        (true, true) => (Props::NEGATIVE, Props::NONNEGATIVE),
        (false, true) => (Props::NONPOSITIVE, Props::POSITIVE),
    };
    if known(holds) {
        Some(true)
    } else if known(fails) {
        Some(false)
    } else {
        None
    }
}

/// LogGamma(n) for positive integer n → ln((n-1)!)
fn eval_log_gamma(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?.clone();
    if !r.is_integer() || !r.is_positive() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // (n-1)!, within the digit guard (beyond it `ln Γ(n)` stays symbolic).
    if beyond_digit_guard(arena, log10_factorial_lower((n - 1) as f64)) {
        return None;
    }
    let fact = Ratio::from_integer(crate::base::combinatorics::factorial(n - 1));
    // ln(1) = 0 — handle n=1 and n=2 where (n-1)! = 1
    if fact == Ratio::<BigInt>::one() {
        return Some(arena.zero);
    }
    let fact_id = guarded_num(arena, fact)?;
    Some(arena.ln(fact_id))
}

/// erf(0) → 0
fn eval_erf(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // erf(0) = 0
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        return Some(arena.zero);
    }

    // erf(∞) = 1
    if matches!(arena.node(inner), ExprNode::Infinity) {
        return Some(arena.one);
    }

    // erf(-∞) = -1
    if matches!(arena.node(inner), ExprNode::NegInfinity) {
        return Some(arena.neg_one);
    }

    // Odd function: erf(-x) = -erf(x)
    if let ExprNode::Neg(pos_inner) = arena.node(inner).clone() {
        let erf_pos = arena.erf(pos_inner);
        return Some(arena.neg(erf_pos));
    }

    None
}

/// erfc(0) → 1
fn eval_lambertw(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // W(0) = 0
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        tracing::debug!("eval: LambertW(0) = 0");
        return Some(arena.zero);
    }

    // W(e) = 1
    if inner == arena.e_const {
        tracing::debug!("eval: LambertW(e) = 1");
        return Some(arena.one);
    }

    // W(∞) = ∞
    if inner == arena.infinity {
        tracing::debug!("eval: LambertW(∞) = ∞");
        return Some(arena.infinity);
    }

    // W(-1/e) = -1
    {
        let e = arena.e_const;
        let neg1 = arena.neg_one;
        let neg_inv_e = arena.div(neg1, e);
        if inner == neg_inv_e {
            tracing::debug!("eval: LambertW(-1/e) = -1");
            return Some(arena.neg_one);
        }
    }

    // W(-ln(2)/2) = -ln(2)
    {
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let neg_ln2 = arena.neg(ln2);
        let target = arena.div(neg_ln2, two);
        if inner == target {
            tracing::debug!("eval: LambertW(-ln(2)/2) = -ln(2)");
            return Some(neg_ln2);
        }
    }

    // W(2*ln(2)) = ln(2)
    {
        let two = arena.int(2);
        let ln2 = arena.ln(two);
        let target = arena.mul(&[two, ln2]);
        if inner == target {
            tracing::debug!("eval: LambertW(2*ln(2)) = ln(2)");
            return Some(ln2);
        }
    }

    // W(-π/2) = iπ/2
    {
        let pi = arena.pi;
        let two = arena.int(2);
        let neg_pi = arena.neg(pi);
        let target = arena.div(neg_pi, two);
        if inner == target {
            tracing::debug!("eval: LambertW(-π/2) = iπ/2");
            let i = arena.i_unit;
            let pi_over_2 = arena.div(pi, two);
            return Some(arena.mul(&[i, pi_over_2]));
        }
    }

    // W(e^(1+e)) = e
    {
        let e = arena.e_const;
        let one = arena.one;
        let one_plus_e = arena.add(&[one, e]);
        let target_exp = arena.exp(one_plus_e);
        if inner == target_exp {
            tracing::debug!("eval: LambertW(e^(1+e)) = e");
            return Some(e);
        }
        let target_pow = arena.pow(e, one_plus_e);
        if inner == target_pow {
            tracing::debug!("eval: LambertW(e^(1+e)) = e");
            return Some(e);
        }
    }

    None
}

fn eval_erfc(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // erfc(0) = 1
    if let Some(r) = arena.as_num(inner)
        && r.is_zero()
    {
        return Some(arena.one);
    }

    // erfc(∞) = 0
    if matches!(arena.node(inner), ExprNode::Infinity) {
        return Some(arena.zero);
    }

    // erfc(-∞) = 2
    if matches!(arena.node(inner), ExprNode::NegInfinity) {
        return Some(arena.int(2));
    }

    None
}

/// `B(a, b)` for positive integers: `(a−1)!(b−1)!/(a+b−1)! =
/// 1/(b·C(a+b−1, b))`, the binomial taken with `j = min(a−1, b)` factors
/// and within the digit guard (`C(n, j) ≥ (n/j)^j`).
///
/// Before 0.30 the three factorials were multiplied out (a `Ratio` product
/// of `a + b` factors): `beta(6, 10^7)`, `beta(10^12, 3)`, `beta(10^8, 17)`
/// and `beta(8, 14!)` never returned, though each is a small rational.
fn eval_beta(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<ExprId> {
    let ra = arena.as_num(a)?.clone();
    let rb = arena.as_num(b)?.clone();
    if !ra.is_integer() || !ra.is_positive() || !rb.is_integer() || !rb.is_positive() {
        return None;
    }
    let a_int = ra.to_integer();
    let b_int = rb.to_integer();
    let n = &a_int + &b_int - BigInt::one();
    let j = (&a_int - BigInt::one()).min(b_int.clone());
    let (nf, jf) = (n.to_f64()?, j.to_f64()?);
    if jf >= 1.0 && beyond_digit_guard(arena, jf * (nf / jf).log10()) {
        return None;
    }
    let c = crate::base::combinatorics::binomial(n, j);
    guarded_num(arena, Ratio::new(BigInt::one(), c * b_int))
}

/// `C(n, k)` for non-negative integer literals.  A negative `n` is left
/// unevaluated on purpose: the generalised `C(−3, 2) = 6` of
/// `combinatorics::binomial` is not what the symbolic node promises.
fn eval_binomial(arena: &mut Arena, n: ExprId, k: ExprId) -> Option<ExprId> {
    let nr = arena.as_num(n)?;
    let kr = arena.as_num(k)?;
    if !nr.is_integer() || !kr.is_integer() || nr.is_negative() || kr.is_negative() {
        return None;
    }
    let n_u64: u64 = nr.to_integer().try_into().ok()?;
    let k_u64: u64 = kr.to_integer().try_into().ok()?;
    // C(n, j) ≥ (n/j)^j for j = min(k, n − k): within the digit guard.
    if k_u64 <= n_u64 {
        let j = k_u64.min(n_u64 - k_u64) as f64;
        if j >= 1.0 && beyond_digit_guard(arena, j * (n_u64 as f64 / j).log10()) {
            return None;
        }
    }
    // `binomial` gives 0 for 0 ≤ n < k (SymPy: `binomial(1, 2) == 0`).
    let result = crate::base::combinatorics::binomial(n_u64, k_u64);
    guarded_num(arena, Ratio::from_integer(result))
}

// ═══════════════════════════════════════════════════════════════════════════
// Special functions (0.2): Si, Ci, Ei, li, ζ, ψ, ψ⁽ⁿ⁾, δᵢⱼ
// ═══════════════════════════════════════════════════════════════════════════

/// Split `x` into `(-1, y)` when `x = -y` in canonical form (negative
/// numeric coefficient or `Neg`), returning `y`.  Unlike [`as_negated`]
/// this also handles bare negative numbers and `-3*x`.
fn as_negated_general(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if let ExprNode::Neg(inner) = arena.node(x) {
        return Some(*inner);
    }
    let (coeff, term) = arena.as_coeff_term(x);
    if coeff.is_negative() {
        Some(arena.make_coeff_term(-coeff, term))
    } else {
        None
    }
}

/// Exact values of the sine integral.
///
/// * `Si(0) = 0`, `Si(∞) = π/2`, `Si(−∞) = −π/2`
/// * odd: `Si(−x) = −Si(x)` (the argument is normalised to a positive
///   leading coefficient)
pub(crate) fn eval_si(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.infinity {
        let half = arena.rational(1, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
    if x == arena.neg_infinity {
        let neg_half = arena.rational(-1, 2);
        return Some(arena.mul(&[neg_half, arena.pi]));
    }
    if x == arena.nan {
        return Some(arena.nan);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let si_y = arena.si(y);
        return Some(arena.neg(si_y));
    }
    None
}

/// Exact values of the cosine integral: `Ci(∞) = 0`, `Ci(0) = −∞`.
///
/// `Ci(−x) = Ci(x) + iπ` (for `x > 0`) is *not* applied automatically
/// because it changes the real/complex character of the expression.
pub(crate) fn eval_ci(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.infinity {
        return Some(arena.zero);
    }
    if x == arena.zero {
        return Some(arena.neg_infinity);
    }
    if x == arena.nan {
        return Some(arena.nan);
    }
    None
}

/// Exact values of the exponential integral:
/// `Ei(−∞) = 0`, `Ei(∞) = ∞`, `Ei(0) = −∞`.
pub(crate) fn eval_ei(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.neg_infinity {
        return Some(arena.zero);
    }
    if x == arena.infinity {
        return Some(arena.infinity);
    }
    if x == arena.zero {
        return Some(arena.neg_infinity);
    }
    if x == arena.nan {
        return Some(arena.nan);
    }
    None
}

/// Exact values of the logarithmic integral:
/// `li(0) = 0`, `li(1) = −∞`, `li(∞) = ∞`, `li(e) = Ei(1)`, `li(eʸ) = Ei(y)`.
pub(crate) fn eval_li(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.one {
        return Some(arena.neg_infinity);
    }
    if x == arena.infinity {
        return Some(arena.infinity);
    }
    if x == arena.nan {
        return Some(arena.nan);
    }
    if x == arena.e_const {
        return Some(arena.ei(arena.one));
    }
    if let ExprNode::Exp(y) = arena.node(x).clone() {
        return Some(arena.ei(y));
    }
    None
}

/// Largest even argument for which `ζ(2k)` is expanded into an exact
/// rational multiple of `π^{2k}`.
const MAX_EXACT_ZETA_EVEN: i64 = 200;

/// Exact values of the Riemann zeta function.
///
/// * `ζ(1) = z∞` (pole), `ζ(∞) = 1`, `ζ(0) = −1/2`
/// * `ζ(−n) = −Bₙ₊₁/(n+1)` for positive integers `n` (so `ζ(−2k) = 0`)
/// * `ζ(2k) = (−1)^{k+1} B₂ₖ (2π)^{2k} / (2·(2k)!)` = `π²/6, π⁴/90, …`
/// * `ζ(2k+1)` for `k ≥ 1` stays symbolic.
pub(crate) fn eval_zeta(arena: &mut Arena, s: ExprId) -> Option<ExprId> {
    if s == arena.infinity {
        return Some(arena.one);
    }
    if s == arena.nan {
        return Some(arena.nan);
    }
    let r = arena.as_num(s)?.clone();
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n == 1 {
        return Some(arena.complex_infinity);
    }
    if n == 0 {
        return Some(arena.rational(-1, 2));
    }
    if n < 0 {
        // ζ(−n) = −B_{n+1}/(n+1)
        let m = (-n) as usize;
        if m.is_multiple_of(2) {
            return Some(arena.zero);
        }
        // Within the digit guard of `B_{m+1}` (`zeta(−10^5)` ran the
        // quadratic rational recurrence for minutes).
        let b = exact_bernoulli(arena, m as u64 + 1)?;
        let val = -b / Ratio::from_integer(BigInt::from(m as i64 + 1));
        return guarded_num(arena, val);
    }
    if n % 2 == 0 && n <= MAX_EXACT_ZETA_EVEN {
        // ζ(2k) = |B_{2k}| · 2^{2k−1} · π^{2k} / (2k)!
        let two_k = n as usize;
        let b = crate::base::bernoulli::bernoulli(two_k).abs();
        let mut fact = BigInt::from(1);
        for i in 2..=two_k {
            fact *= BigInt::from(i as u64);
        }
        let pow2 = BigInt::from(1) << (two_k - 1);
        let coeff = b * Ratio::from_integer(pow2) / Ratio::from_integer(fact);
        let nid = arena.intern_num(coeff);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let exp_id = arena.int(n);
        let pi_pow = arena.pow(arena.pi, exp_id);
        return Some(arena.mul(&[coeff_id, pi_pow]));
    }
    None
}

/// Largest integer / half-integer shift handled exactly by the polygamma
/// and digamma recurrences.
const MAX_POLYGAMMA_SHIFT: i64 = 64;

/// Most bits the exact shifted form of `ψ⁽ⁿ⁾` may cancel (see
/// [`eval_polygamma`]); `evalf` recovers a few hundred bits of cancellation
/// within its precision search.
const MAX_POLYGAMMA_CANCEL_BITS: f64 = 200.0;

/// `n!` as a rational.
fn factorial_ratio(n: usize) -> Q {
    Ratio::from_integer(crate::base::combinatorics::factorial(n as u64))
}

/// Exact values of the digamma function:
/// `ψ(1) = −γ`, `ψ(m) = −γ + H_{m−1}`, `ψ(1/2) = −γ − 2 ln 2`,
/// `ψ(m + 1/2) = ψ(1/2) + Σ_{k<m} 1/(k + 1/2)`, poles at non-positive integers.
pub(crate) fn eval_digamma(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    let r = arena.as_num(x)?.clone();
    let two = BigInt::from(2);
    if r.is_integer() {
        let m: i64 = r.to_integer().try_into().ok()?;
        if m <= 0 {
            return Some(arena.complex_infinity);
        }
        if m > MAX_POLYGAMMA_SHIFT {
            return None;
        }
        // ψ(m) = −γ + H_{m−1}
        let mut h = Ratio::<BigInt>::zero();
        for k in 1..m {
            h += Ratio::new(BigInt::from(1), BigInt::from(k));
        }
        let neg_gamma = arena.neg(arena.euler_gamma);
        let nid = arena.intern_num(h);
        let h_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[neg_gamma, h_id]));
    }
    if *r.denom() == two {
        // x = m + 1/2 with m = floor(x)
        let m: i64 = r.floor().to_integer().try_into().ok()?;
        if !(0..=MAX_POLYGAMMA_SHIFT).contains(&m) {
            return None;
        }
        // ψ(1/2) = −γ − 2 ln 2
        let neg_gamma = arena.neg(arena.euler_gamma);
        let two_id = arena.int(2);
        let ln2 = arena.ln(two_id);
        let neg_two = arena.int(-2);
        let neg_two_ln2 = arena.mul(&[neg_two, ln2]);
        // Σ_{k=0}^{m−1} 1/(k + 1/2) = Σ 2/(2k+1)
        let mut h = Ratio::<BigInt>::zero();
        for k in 0..m {
            h += Ratio::new(BigInt::from(2), BigInt::from(2 * k + 1));
        }
        let nid = arena.intern_num(h);
        let h_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[neg_gamma, neg_two_ln2, h_id]));
    }
    None
}

/// Exact values of the polygamma function `ψ⁽ⁿ⁾(x)`.
///
/// * `n = 0` → [`Digamma`](ExprNode::Digamma)
/// * `ψ⁽ⁿ⁾(1) = (−1)^{n+1} n! ζ(n+1)`
/// * `ψ⁽ⁿ⁾(1/2) = (−1)^{n+1} n! (2^{n+1} − 1) ζ(n+1)`
/// * integer / half-integer arguments are shifted to those base points via
///   `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/x^{n+1}`
/// * poles at non-positive integers → `z∞`
pub(crate) fn eval_polygamma(arena: &mut Arena, n: ExprId, x: ExprId) -> Option<ExprId> {
    let nr = arena.as_num(n)?.clone();
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    if nr.is_zero() {
        return Some(arena.digamma(x));
    }
    let n_usize: usize = nr.to_integer().try_into().ok()?;
    let xr = arena.as_num(x)?.clone();
    if xr.is_integer() && !xr.is_positive() {
        return Some(arena.complex_infinity);
    }
    // The shift to the base point: m − 1 terms 1/k^{n+1} (integer x = m) or
    // m terms 2^{n+1}/(2k+1)^{n+1} (x = m + 1/2); anything else stays
    // symbolic (and n! is not computed for it).
    let half = *xr.denom() == BigInt::from(2);
    let m: i64 = if xr.is_integer() || half {
        xr.floor().to_integer().try_into().ok()?
    } else {
        return None;
    };
    if !(0..=MAX_POLYGAMMA_SHIFT).contains(&m) {
        return None;
    }
    // n! is within the digit guard, and the shifted form does not cancel:
    // ψ⁽ⁿ⁾(m) = (−1)^{n+1} n! (ζ(n+1) − Σ_{k<m} k^{−n−1}) is ≈ n!·m^{−n−1}
    // while its terms are ≈ n!, and ψ⁽ⁿ⁾(m + 1/2) ≈ n!·(m + 1/2)^{−n−1}
    // while its terms are ≈ n!·2^{n+1}, so (n+1)·log₂ m (resp.
    // (n+1)·log₂(2m+1)) bits cancel.  `polygamma(3000, 3/2)` cancelled
    // 4757 bits, beyond what `evalf` searches, and printed 0
    // (mpmath: −1.47269444412519507002495959e8602); beyond the bound the
    // node stays symbolic for `evalf`.
    if beyond_digit_guard(arena, log10_factorial_lower(n_usize as f64)) {
        return None;
    }
    let shifted_from = if half { 2 * m + 1 } else { m };
    if shifted_from >= 2
        && (n_usize as f64 + 1.0) * (shifted_from as f64).log2() > MAX_POLYGAMMA_CANCEL_BITS
    {
        return None;
    }
    let sign = if n_usize.is_multiple_of(2) { -1 } else { 1 }; // (−1)^{n+1}
    let n_fact = factorial_ratio(n_usize);
    let np1 = arena.int(n_usize as i64 + 1);
    let zeta_np1 = arena.zeta(np1);

    if xr.is_integer() {
        // ψ⁽ⁿ⁾(m) = ψ⁽ⁿ⁾(1) + (−1)ⁿ n! Σ_{k=1}^{m−1} 1/k^{n+1}
        let base_coeff = &n_fact * Ratio::from_integer(BigInt::from(sign));
        let nid = arena.intern_num(base_coeff);
        let base_coeff_id = arena.intern(ExprNode::Num(nid));
        let base = arena.mul(&[base_coeff_id, zeta_np1]);
        let mut shift = Ratio::<BigInt>::zero();
        for k in 1..m {
            let kp = BigInt::from(k).pow((n_usize + 1) as u32);
            shift += Ratio::new(BigInt::from(1), kp);
        }
        let shift = shift * &n_fact * Ratio::from_integer(BigInt::from(-sign));
        let nid = arena.intern_num(shift);
        let shift_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[base, shift_id]));
    }
    if half {
        // ψ⁽ⁿ⁾(1/2) = (−1)^{n+1} n! (2^{n+1} − 1) ζ(n+1)
        let two_pow = (BigInt::from(1) << (n_usize + 1)) - BigInt::from(1);
        let base_coeff =
            &n_fact * Ratio::from_integer(two_pow) * Ratio::from_integer(BigInt::from(sign));
        let nid = arena.intern_num(base_coeff);
        let base_coeff_id = arena.intern(ExprNode::Num(nid));
        let base = arena.mul(&[base_coeff_id, zeta_np1]);
        // shift: (−1)ⁿ n! Σ_{k=0}^{m−1} 1/(k+1/2)^{n+1} = (−1)ⁿ n! Σ 2^{n+1}/(2k+1)^{n+1}
        let mut shift = Ratio::<BigInt>::zero();
        for k in 0..m {
            let denom = BigInt::from(2 * k + 1).pow((n_usize + 1) as u32);
            let numer = BigInt::from(1) << (n_usize + 1);
            shift += Ratio::new(numer, denom);
        }
        let shift = shift * &n_fact * Ratio::from_integer(BigInt::from(-sign));
        let nid = arena.intern_num(shift);
        let shift_id = arena.intern(ExprNode::Num(nid));
        return Some(arena.add(&[base, shift_id]));
    }
    None
}

/// Exact values of the Kronecker delta: `δᵢᵢ = 1`; `δᵢⱼ = 0` whenever
/// `i − j` canonicalises to a non-zero number.
pub(crate) fn eval_kronecker_delta(arena: &mut Arena, i: ExprId, j: ExprId) -> Option<ExprId> {
    if i == j {
        return Some(arena.one);
    }
    let d = arena.sub(i, j);
    let r = arena.as_num(d)?;
    if r.is_zero() {
        Some(arena.one)
    } else {
        Some(arena.zero)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Combinatorial helper functions
// ═══════════════════════════════════════════════════════════════════════════

// ── The digit guard of the integer sequences ─────────────────────────────
//
// `n!!`, `!n`, `Fₙ`, `Lₙ`, `Bₙ`, `Hₙ`, `Cₙ`, Bell and Euler numbers,
// Stirling numbers and `p(n)` follow the policy of `n!` above: a lower
// bound on the digits of the result refuses before any arithmetic, the
// exact count decides at the limit ([`guarded_num`]), and beyond it the
// node stays symbolic for `evalf`.  Up to 0.29 none was bounded:
// `fibonacci(10^7)`, `bernoulli(10^5)`, `harmonic(10^6)`, `catalan(10^6)`,
// `bell(10^4)`, `euler_number(10^4)`, `stirling2(10^4, 5000)` and
// `partition_count(10^7)` each ran for more than 20 s (release), and
// `subfactorial(10^5)` built a 456 574-digit integer in 4 s.  The Stirling
// triangle and the pentagonal recurrence of `p(n)` are also bounded in
// work, since their cost outgrows the digits of the result.

/// `log₁₀ φ` of the golden ratio.
const LOG10_PHI: f64 = 0.208_987_640_249_978_73;

/// A non-negative integer argument that fits in a `u64`.
fn nonneg_int_arg(arena: &Arena, id: ExprId) -> Option<u64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    r.to_integer().try_into().ok()
}

/// An integer as a guarded node (see [`guarded_num`]).
fn guarded_int(arena: &mut Arena, n: BigInt) -> Option<ExprId> {
    guarded_num(arena, Ratio::from_integer(n))
}

/// Double factorial: n!! = n * (n-2) * (n-4) * ... * 1 (or 2).
/// 0!! = 1, 1!! = 1, (-1)!! = 1.  Within the digit guard: `n!! ≥ (2m)!! =
/// 2^m·m!` for `m = ⌊n/2⌋`.
fn eval_factorial2(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n < -1 {
        return None;
    }
    let m = (n.max(0) / 2) as f64;
    if beyond_digit_guard(
        arena,
        m * std::f64::consts::LOG10_2 + log10_factorial_lower(m),
    ) {
        return None;
    }
    let mut result = BigInt::from(1);
    let mut k = n;
    while k > 1 {
        result *= BigInt::from(k);
        k -= 2;
    }
    guarded_int(arena, result)
}

/// Subfactorial (derangement count): !n.
/// Uses recurrence: !0 = 1, !1 = 0, !n = (n-1)(!(n-1) + !(n-2)).
/// Within the digit guard: `!n = round(n!/e) ≥ n!/3` for `n ≥ 4`.
fn eval_subfactorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if n >= 4 && beyond_digit_guard(arena, log10_factorial_lower(n as f64) - 3f64.log10()) {
        return None;
    }
    let result = if n == 0 {
        BigInt::from(1)
    } else {
        let mut prev = BigInt::from(1); // !0
        let mut curr = BigInt::from(0); // !1
        for i in 2..=n {
            let next = BigInt::from(i - 1) * (&curr + &prev);
            prev = curr;
            curr = next;
        }
        curr
    };
    guarded_int(arena, result)
}

/// Rising factorial (Pochhammer): (x)_n = x * (x+1) * ... * (x+n-1).
/// Works for rational x; n must be a non-negative integer.  Within the
/// digit guard (see [`beyond_digit_guard`]).
fn eval_rising_factorial(arena: &mut Arena, x_id: ExprId, n_id: ExprId) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, 1)
}

/// Falling factorial: x^(n) = x * (x-1) * ... * (x-n+1).
/// Works for rational x; n must be a non-negative integer.  Within the
/// digit guard (see [`beyond_digit_guard`]).
fn eval_falling_factorial(arena: &mut Arena, x_id: ExprId, n_id: ExprId) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, -1)
}

/// Fibonacci number F(n) using iterative computation.
/// F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2).  Within the digit guard:
/// `Fₙ ≥ φ^(n−2)`.
fn eval_fibonacci(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n as f64 - 2.0) * LOG10_PHI) {
        return None;
    }
    let result = if n == 0 {
        BigInt::from(0)
    } else {
        let mut a = BigInt::from(0);
        let mut b = BigInt::from(1);
        for _ in 1..n {
            let tmp = &a + &b;
            a = b;
            b = tmp;
        }
        b
    };
    guarded_int(arena, result)
}

/// Lucas number L(n) using iterative computation.
/// L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2).  Within the digit guard:
/// `Lₙ ≥ φ^(n−1)`.
fn eval_lucas(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n as f64 - 1.0) * LOG10_PHI) {
        return None;
    }
    let result = if n == 0 {
        BigInt::from(2)
    } else {
        let mut a = BigInt::from(2);
        let mut b = BigInt::from(1);
        for _ in 1..n {
            let tmp = &a + &b;
            a = b;
            b = tmp;
        }
        b
    };
    guarded_int(arena, result)
}

/// The Bernoulli number `Bₙ` exactly, or `None` beyond the digit guard.
///
/// `|Bₙ| = 2·n!·ζ(n)/(2π)ⁿ ≥ 2·n!/(2π)ⁿ` for even `n ≥ 2` bounds the
/// digits first.  Even indices come from the tangent numbers,
/// `B₂ₘ = (−1)^(m−1)·2m·Tₘ/(2^(2m)·(2^(2m) − 1))`, computed by Brent and
/// Harvey's integer recurrence ("Fast computation of Bernoulli, Tangent
/// and Secant numbers", 2011, Algorithm TangentNumbers): `O(m²)`
/// small-integer multiplications and additions and one final gcd, where
/// the rational recurrence `Bₘ = −Σ C(m+1, k)·Bₖ/(m+1)` took a gcd per
/// term (`bernoulli(2000)` inside the guard would have taken minutes).
pub(crate) fn exact_bernoulli(arena: &Arena, n: u64) -> Option<Q> {
    match n {
        0 => return Some(Ratio::one()),
        1 => return Some(Ratio::new(BigInt::from(-1), BigInt::from(2))),
        _ if n % 2 == 1 => return Some(Ratio::zero()),
        _ => {}
    }
    let nf = n as f64;
    let lower = std::f64::consts::LOG10_2 + log10_factorial_lower(nf)
        - nf * (2.0 * std::f64::consts::PI).log10();
    if beyond_digit_guard(arena, lower) {
        return None;
    }
    let m = (n / 2) as usize;
    let t = tangent_numbers(m);
    let two_2m = BigInt::one() << (2 * m);
    let den = &two_2m * (&two_2m - BigInt::one());
    let num = BigInt::from(2 * m as u64) * &t[m];
    let b = Ratio::new(num, den);
    let r = if m % 2 == 1 { b } else { -b };
    let digits = r.numer().to_string().len() + r.denom().to_string().len();
    (digits <= arena.config.max_result_digits).then_some(r)
}

/// The tangent numbers `T₁, …, Tₘ` (index 0 unused), by Brent and
/// Harvey's in-place recurrence (Algorithm TangentNumbers of the paper
/// cited at [`exact_bernoulli`]).
fn tangent_numbers(m: usize) -> Vec<BigInt> {
    let mut t = vec![BigInt::zero(); m + 1];
    if m == 0 {
        return t;
    }
    t[1] = BigInt::one();
    for k in 2..=m {
        t[k] = &t[k - 1] * BigInt::from(k as u64 - 1);
    }
    for k in 2..=m {
        for j in k..=m {
            let a = &t[j - 1] * BigInt::from((j - k) as u64);
            let b = &t[j] * BigInt::from((j - k + 2) as u64);
            t[j] = a + b;
        }
    }
    t
}

/// Bernoulli number B(n), within the digit guard ([`exact_bernoulli`]).
fn eval_bernoulli(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    let b = exact_bernoulli(arena, n)?;
    guarded_num(arena, b)
}

/// A lower bound on the decimal digits of the reduced denominator of
/// `Hₙ`: every prime `p ∈ (n/2, n]` divides it (`1/p` is the only term
/// with a factor `p`), so it is at least `e^(θ(n) − θ(n/2))`, and Rosser
/// and Schoenfeld's bounds (1962, Theorem 4 and its corollary:
/// `θ(x) > x(1 − 1/ln x)` for `x ≥ 41`, `θ(x) < x(1 + 1/(2 ln x))` for
/// `x > 1`) bound the difference.
fn harmonic_digits_lower(n: u64) -> f64 {
    if n < 82 {
        return 0.0;
    }
    let x = n as f64;
    let h = x / 2.0;
    let theta_diff = x * (1.0 - 1.0 / x.ln()) - h * (1.0 + 1.0 / (2.0 * h.ln()));
    (theta_diff / std::f64::consts::LN_10).max(0.0)
}

/// Harmonic number H(n) = 1 + 1/2 + 1/3 + ... + 1/n.
/// H(0) = 0.  Within the digit guard ([`harmonic_digits_lower`]); the sum
/// is taken over the common denominator `lcm(1, …, n)` and reduced once,
/// where a gcd per term made `Hₙ` quadratic in its digits.
fn eval_harmonic(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if beyond_digit_guard(arena, harmonic_digits_lower(n)) {
        return None;
    }
    let mut lcm = BigInt::one();
    for k in 1..=n {
        let kb = BigInt::from(k);
        let g = num_integer::Integer::gcd(&(&lcm % &kb), &kb);
        lcm *= kb / g;
    }
    let mut numer = BigInt::zero();
    for k in 1..=n {
        numer += &lcm / BigInt::from(k);
    }
    guarded_num(arena, Ratio::new(numer, lcm))
}

/// Catalan number C(n) = (2n)! / ((n+1)! * n!).  Within the digit guard:
/// `Cₙ ≥ 4ⁿ/((2n + 1)(n + 1))`.
fn eval_catalan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    let nf = n as f64;
    if beyond_digit_guard(
        arena,
        nf * 4f64.log10() - ((2.0 * nf + 1.0) * (nf + 1.0)).log10(),
    ) {
        return None;
    }
    // C(n) = C(2n, n) / (n+1) — compute using incremental binomial
    let mut binom = BigInt::from(1); // C(2n, n) built incrementally
    for i in 0..n {
        binom *= BigInt::from(2 * n - i);
        binom /= BigInt::from(i + 1);
    }
    guarded_num(arena, Ratio::new(binom, BigInt::from(n + 1)))
}

/// Largest `n·k·limbs` of the Stirling triangle (`O(n·k)` big-integer
/// operations on numbers of the result's size) computed exactly.
const MAX_STIRLING_WORK: f64 = 2.0e8;

/// `(n, k)` of a Stirling number within the digit guard and the work
/// bound: `S(n, k) ≥ k^(n−k)` (the first `k` elements in distinct blocks,
/// the others anywhere), and `|s(n, k)| ≥ S(n, k)` (every partition into
/// `k` blocks gives a permutation with `k` cycles), `|s(n, 1)| = (n − 1)!`.
/// The closed forms of `combinatorics` for `k ≤ 2` (second kind),
/// `k = 1` (first kind) and `k ≥ n − 1` cost no triangle.
fn stirling_args(
    arena: &Arena,
    n_id: ExprId,
    k_id: ExprId,
    first_kind: bool,
) -> Option<(u64, u64)> {
    let n = nonneg_int_arg(arena, n_id)?;
    let k = nonneg_int_arg(arena, k_id)?;
    if k == 0 || k > n {
        return Some((n, k));
    }
    let (nf, kf) = (n as f64, k as f64);
    let mut lower = (nf - kf) * kf.log10();
    if first_kind && k == 1 {
        lower = log10_factorial_lower(nf - 1.0);
    }
    if beyond_digit_guard(arena, lower) {
        return None;
    }
    let closed_form = k + 1 >= n || k == 1 || (!first_kind && k == 2);
    if !closed_form && nf * kf * (1.0 + lower / 19.0) > MAX_STIRLING_WORK {
        return None;
    }
    Some((n, k))
}

/// Stirling number of the second kind S(n, k), within the digit guard and
/// the work bound ([`stirling_args`]).
fn eval_stirling2(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    let (n, k) = stirling_args(arena, n_id, k_id, false)?;
    let result = crate::domains::combinatorics::stirling2(n, k)?;
    guarded_int(arena, result)
}

/// Signed Stirling number of the first kind s(n, k), within the digit guard
/// and the work bound ([`stirling_args`]).
fn eval_stirling1(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    let (n, k) = stirling_args(arena, n_id, k_id, true)?;
    let result = crate::domains::combinatorics::stirling1(n, k)?;
    guarded_int(arena, result)
}

/// Largest `n` whose partition count is computed exactly: the pentagonal
/// recurrence costs `O(n^{3/2})` additions and keeps all `p(k)`, `k ≤ n`
/// (`p(40000)` has 218 digits, far inside the digit guard, and takes
/// about half a second in release).
const MAX_PARTITION_N: u64 = 40_000;

/// Number of integer partitions p(n), for `n ≤` [`MAX_PARTITION_N`].
fn eval_partition_count(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if n > MAX_PARTITION_N {
        return None;
    }
    let result = crate::domains::combinatorics::partition_count(n)?;
    guarded_int(arena, result)
}

/// Bell number B(n) using the Bell triangle.
/// B(0) = 1, B(1) = 1, B(2) = 2, B(3) = 5, B(4) = 15, B(5) = 52.
/// Within the digit guard: `Bₙ ≥ S(n, k) ≥ k^(n−k)` for any `k`, taken at
/// `k ≈ n/ln n`.
fn eval_bell(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    if n >= 3 {
        let nf = n as f64;
        let k = (nf / nf.ln()).round().clamp(1.0, nf);
        if beyond_digit_guard(arena, (nf - k) * k.log10()) {
            return None;
        }
    }
    if n == 0 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(1)));
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    // Bell triangle: row[0] = B(n-1), then row[j] = row[j-1] + prev_row[j-1]
    let mut row = vec![BigInt::from(1)]; // B(0) = 1, start of row 1
    for _ in 1..n {
        let mut new_row = Vec::with_capacity(row.len() + 1);
        new_row.push(row.last()?.clone()); // first element = last of prev row
        for j in 1..=row.len() {
            let val = &new_row[j - 1] + &row[j - 1];
            new_row.push(val);
        }
        row = new_row;
    }
    // B(n) = last element of the nth row
    let result = row.pop()?;
    guarded_int(arena, result)
}

/// Euler number E(n). Odd indices give 0.
/// E(0) = 1, E(2) = -1, E(4) = 5, E(6) = -61, ...
/// Recurrence for even n >= 2: E(n) = -sum_{k=0,2,4,...,n-2} C(n, k) * E(k).
///
/// Within the digit guard: `|Eₙ| = 2^(n+2)·n!·β(n+1)/π^(n+1)` with the
/// Dirichlet beta `β(s) ≥ 1 − 3^(−s) ≥ 2/3`.
fn eval_euler_number(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = nonneg_int_arg(arena, inner)?;
    // Odd Euler numbers are 0
    if n % 2 == 1 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(0)));
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    let nf = n as f64;
    let lower = log10_factorial_lower(nf) + (nf + 2.0) * std::f64::consts::LOG10_2
        - (nf + 1.0) * std::f64::consts::PI.log10()
        + (2.0f64 / 3.0).log10();
    if n >= 2 && beyond_digit_guard(arena, lower) {
        return None;
    }
    // Build table of E(0), E(2), E(4), ..., E(n)
    let half = (n / 2) as usize;
    let mut e_vals: Vec<BigInt> = Vec::with_capacity(half + 1);
    e_vals.push(BigInt::from(1)); // E(0) = 1
    for m_half in 1..=half {
        let m = (m_half * 2) as u64; // the actual index
        // E(m) = -sum_{k=0,2,...,m-2} C(m, k) * E(k)
        let mut sum = BigInt::from(0);
        let mut binom = BigInt::from(1); // C(m, 0)
        for (k_half, e_val) in e_vals[..m_half].iter().enumerate() {
            let k = (k_half * 2) as u64;
            sum += &binom * e_val;
            // Advance binom from C(m, k) to C(m, k+2)
            // C(m, k+1) = C(m, k) * (m-k) / (k+1)
            // C(m, k+2) = C(m, k+1) * (m-k-1) / (k+2)
            binom *= BigInt::from(m - k);
            binom /= BigInt::from(k + 1);
            binom *= BigInt::from(m - k - 1);
            binom /= BigInt::from(k + 2);
        }
        e_vals.push(-sum);
    }
    let result = e_vals.pop()?;
    guarded_int(arena, result)
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
fn as_pi_multiple(arena: &Arena, id: ExprId) -> Option<Q> {
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
    // The special value of the positive argument is taken too: returning
    // `−sin(π)` left `eval(sin(−π))` non-zero to structural tests (Gruntz
    // took it for a leading coefficient, and `lim_{x→0⁺} x/sin(x − π)` was 0).
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sin_pos = eval_sin(arena, pos_inner).unwrap_or_else(|| arena.sin(pos_inner));
        return Some(arena.neg(sin_pos));
    }

    // sin(i*x) = i*sinh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        let sinh_val = arena.sinh(real_part);
        return Some(arena.mul(&[arena.i_unit, sinh_val]));
    }

    let coeff = as_pi_multiple(arena, inner)?;

    // Reduce modulo 2 (sin has period 2π).
    let two: Q = Ratio::from_integer(2.into());
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
    let two: Q = Ratio::from_integer(2.into());
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
        return Some(eval_cos(arena, pos_inner).unwrap_or_else(|| arena.cos(pos_inner)));
    }

    // cos(i*x) = cosh(x) — trig-hyperbolic bridge
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        return Some(arena.cosh(real_part));
    }

    let coeff = as_pi_multiple(arena, inner)?;

    let two: Q = Ratio::from_integer(2.into());
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
    let two: Q = Ratio::from_integer(2.into());
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
        let tan_pos = eval_tan(arena, pos_inner).unwrap_or_else(|| arena.tan(pos_inner));
        return Some(arena.neg(tan_pos));
    }

    let coeff = as_pi_multiple(arena, inner)?;

    let one_ratio: Q = Ratio::one();
    let coeff = mod_positive(&coeff, &one_ratio);

    // tan(0) = 0
    if coeff.is_zero() {
        return Some(arena.zero);
    }
    // tan(π/2) is a pole → ComplexInfinity.
    // The mod_positive reduction ensures 3π/2, 5π/2, etc. all map to coeff = 1/2.
    if coeff == Ratio::new(1.into(), 2.into()) {
        return Some(arena.complex_infinity);
    }
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

    // Q2 reduction: tan(π - x) = -tan(x) for coeff in (1/2, 1)
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    if coeff > half && coeff < Ratio::one() {
        let reflected = Ratio::one() - &coeff;
        let nid = arena.intern_num(reflected);
        let coeff_id = arena.intern(ExprNode::Num(nid));
        let reflected_angle = arena.mul(&[coeff_id, arena.pi()]);
        if let Some(val) = eval_tan(arena, reflected_angle) {
            return Some(arena.neg(val));
        }
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
    // exp(ln w) = w for every w ≠ 0 on the principal branch (the logarithm
    // is defined exactly there), as SymPy folds it.
    if let ExprNode::Ln(w) = *arena.node(inner) {
        return Some(w);
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

    // exp(w + i·k·π) = exp(w)·exp(i·k·π) for a sum with an imaginary
    // π-multiple term whose exponential folds (k integer or half-integer):
    // exp(ln 2 − 3πi) = −2 exactly.  Otherwise the numeric evaluator meets
    // e^{ln 2}·(cos 3π − i sin 3π) with a rounding residue in the imaginary
    // part whose sign picks the side of the next branch cut (0.24,
    // fuzz_simplify: √((e^{ln(−2)})^{−3}) came out −(√2/4)i at 30 digits).
    if let ExprNode::Add(terms) = arena.node(inner).clone() {
        for (k, &t) in terms.iter().enumerate() {
            if let Some(pi_coeff) = as_imaginary_pi_multiple(arena, t)
                && (&pi_coeff * Q::from_integer(BigInt::from(2))).is_integer()
                && let Some(unit) = eval_exp(arena, t)
            {
                let rest: smallvec::SmallVec<[ExprId; 6]> = terms
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != k)
                    .map(|(_, &c)| c)
                    .collect();
                let rest = arena.add(&rest);
                let rest_exp = eval_exp(arena, rest).unwrap_or_else(|| arena.exp(rest));
                return Some(arena.mul(&[unit, rest_exp]));
            }
        }
    }

    None
}

/// Check if `id` is of the form `i * k * π` for some rational `k`.
/// Returns `Some(k)` if so, `None` otherwise.
fn as_imaginary_pi_multiple(arena: &mut Arena, id: ExprId) -> Option<Q> {
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

/// `(e^f)^g = e^(f·g)` on the principal branch?  Always for an integer
/// `g`; for any `g` when `f` is real (then `e^f > 0`, so `ln(e^f) = f`).
/// In general it needs `Im f ∈ (−π, π]`: `√(e^{4i}) = −e^{2i}`.  Before
/// 0.23 the merge was unconditional, which also corrupted `eval_decimal`
/// of such an expression (it calls `eval` first).
fn exp_pow_merges(arena: &Arena, f: ExprId, g: ExprId) -> bool {
    if arena.as_num(g).is_some_and(|r| r.is_integer()) {
        return true;
    }
    let mut cache = crate::base::assumptions::AssumptionCache::new();
    cache.query(arena, g, crate::base::assumptions::Props::INTEGER) == Some(true)
        || cache.query(arena, f, crate::base::assumptions::Props::REAL) == Some(true)
}

/// Evaluate `Pow(base, 1/n)` for a rational `base` and an integer `n ≥ 2`:
/// the perfect `n`-th power factor of a positive base comes out
/// ([`positive_root`]), and a negative base takes the principal branch.
fn eval_pow_root(arena: &mut Arena, base: ExprId, exp: ExprId) -> Option<ExprId> {
    let base_r = arena.as_num(base)?.clone();
    let exp_r = arena.as_num(exp)?;

    // exp must be 1/n for positive integer n ≥ 2
    if !exp_r.numer().is_one() {
        return None;
    }
    let n: u32 = exp_r.denom().to_u32()?;
    if n < 2 {
        return None;
    }

    // Roots of negative rationals take the principal branch, like every
    // other evaluator: (−r)^(1/k) = r^(1/k)·(−1)^(1/k), exact because
    // arg(−r) = π (so ∛(−8) = 2·(−1)^(1/3) = 1 + √3·i, SymPy's
    // `2*(-1)**(1/3)`).  Before 0.23 odd roots of negative integers folded to
    // the real root (∛(−8) → −2) while `evalf` took the principal one; the
    // real root is `Ex::real_root`.
    if base_r.is_negative() && base_r != -Ratio::from_integer(BigInt::from(1)) {
        let abs_r = -base_r;
        // Before 0.29 `r^(1/k)` was built and handed to a fresh `eval`
        // pass, which extracted its radical again (through this function).
        let abs_root = positive_root(arena, &abs_r, exp, n).unwrap_or_else(|| {
            let abs_id = arena.num_ratio(abs_r);
            arena.pow(abs_id, exp)
        });
        let neg_one_root = arena.pow(arena.neg_one, exp);
        return Some(arena.mul(&[abs_root, neg_one_root]));
    }
    positive_root(arena, &base_r, exp, n)
}

/// `r^(1/n)` (`exp` is the node `1/n`) for a non-negative rational `r`, when
/// it simplifies: `0`, `1`, a perfect power (an exact rational), or an
/// integer `k^n·m` with `k > 1` (`k·m^(1/n)`), each factor split once by
/// [`crate::base::canon::split_perfect_power`] (which recognises exact
/// powers by an integer root first).  `None` otherwise.
fn positive_root(arena: &mut Arena, r: &Q, exp: ExprId, n: u32) -> Option<ExprId> {
    use crate::base::canon::split_perfect_power;
    if r.is_zero() {
        return Some(arena.zero);
    }
    if r.is_one() {
        return Some(arena.one);
    }
    if r.is_negative() {
        return None;
    }
    let (kn, mn) = split_perfect_power(r.numer(), n);
    if r.is_integer() {
        if mn.is_one() {
            return Some(arena.big_int(kn)); // perfect power
        }
        if kn.is_one() {
            return None;
        }
        // n^(1/q) = k · m^(1/q)
        let k_id = arena.big_int(kn);
        let m_id = arena.big_int(mn);
        let root = arena.pow(m_id, exp);
        return Some(arena.mul(&[k_id, root]));
    }
    // p/q: only a perfect power of both folds.
    if !mn.is_one() {
        return None;
    }
    let (kd, md) = split_perfect_power(r.denom(), n);
    if !md.is_one() {
        return None;
    }
    Some(arena.num_ratio(Ratio::new(kn, kd)))
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

    // Complex modulus of a numeric constant: |a + b·i| = √(a² + b²) whenever
    // the real/imaginary decomposition is exact, both parts are arithmetic
    // constants (numbers, π, e, … combined with + − × ^, so radicals are
    // included) and the imaginary part is structurally non-zero:
    // |3 + 4i| = 5, |1 + i| = √2, |1 + √3·i| = 2, |e + πi| = √(e² + π²).
    // Real constants are left to the sign-aware simplifier (`√(π²)` would
    // be a step backwards), as are transcendental parts (|e^{i}| stays put
    // rather than becoming √(sin²1 + cos²1)).
    if !crate::base::complex::is_real_node(arena, inner)
        && crate::base::walk::free_symbols(arena, inner).is_empty()
    {
        let parts = crate::base::complex::decompose(arena, inner);
        if parts.exact
            && parts.im != arena.zero
            && is_arithmetic_constant(arena, parts.re)
            && is_arithmetic_constant(arena, parts.im)
        {
            tracing::trace!("eval_abs: folding modulus of a numeric complex constant");
            let two = arena.int(2);
            let re2 = arena.pow(parts.re, two);
            let im2 = arena.pow(parts.im, two);
            let sum = arena.add(&[re2, im2]);
            let sum = eval(arena, sum);
            let half = arena.rational(1, 2);
            return Some(arena.pow(sum, half));
        }
    }

    None
}

/// True when `id` is built only from numbers, named real constants and
/// `Add`/`Mul`/`Neg`/`Pow` — i.e. an exact constant whose square is again
/// such a constant (`3`, `√3`, `π/2`, `1 + √2`), with no symbols and no
/// transcendental function applications.
fn is_arithmetic_constant(arena: &Arena, id: ExprId) -> bool {
    crate::base::walk::post_order_ids(arena, id)
        .iter()
        .all(|&n| {
            matches!(
                arena.node(n),
                ExprNode::Num(_)
                    | ExprNode::Pi
                    | ExprNode::E
                    | ExprNode::EulerGamma
                    | ExprNode::Catalan
                    | ExprNode::GoldenRatio
                    | ExprNode::Add(_)
                    | ExprNode::Mul(_)
                    | ExprNode::Neg(_)
                    | ExprNode::Pow(_, _)
            )
        })
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

/// Evaluate `atan2(y, x)` with quadrant-aware logic.
///
/// When both arguments are numeric rationals, returns an exact symbolic
/// result.  Returns `None` when no simplification is possible.
pub(crate) fn eval_atan2(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    // Clone numeric values up-front so we can use arena mutably below.
    let y_val = arena.as_num(y).cloned();
    let x_val = arena.as_num(x).cloned();

    match (y_val, x_val) {
        (Some(yv), Some(xv)) => {
            let y_zero = yv.is_zero();
            let x_zero = xv.is_zero();
            let y_pos = !yv.is_negative() && !y_zero;
            let x_pos = !xv.is_negative() && !x_zero;
            let y_neg = yv.is_negative();
            let x_neg = xv.is_negative();

            if y_zero && x_zero {
                // atan2(0, 0) = 0 (convention)
                Some(arena.zero)
            } else if y_zero && x_pos {
                // atan2(0, x>0) = 0
                Some(arena.zero)
            } else if y_zero && x_neg {
                // atan2(0, x<0) = π
                Some(arena.pi)
            } else if y_pos && x_zero {
                // atan2(y>0, 0) = π/2
                let half = arena.rational(1, 2);
                Some(arena.mul(&[half, arena.pi]))
            } else if y_neg && x_zero {
                // atan2(y<0, 0) = -π/2
                let neg_half = arena.rational(-1, 2);
                Some(arena.mul(&[neg_half, arena.pi]))
            } else if x_pos {
                // Quadrant I or IV: atan2(y, x) = atan(y/x)
                let ratio = arena.div(y, x);
                let atan_node = arena.atan(ratio);
                Some(eval_atan(arena, ratio).unwrap_or(atan_node))
            } else if x_neg && !y_neg {
                // Quadrant II (y >= 0, x < 0): atan2(y, x) = atan(y/x) + π
                let ratio = arena.div(y, x);
                let atan_val = eval_atan(arena, ratio).unwrap_or_else(|| arena.atan(ratio));
                Some(arena.add(&[atan_val, arena.pi]))
            } else {
                // Quadrant III (y < 0, x < 0): atan2(y, x) = atan(y/x) - π
                let ratio = arena.div(y, x);
                let atan_val = eval_atan(arena, ratio).unwrap_or_else(|| arena.atan(ratio));
                let neg_pi = arena.neg(arena.pi);
                Some(arena.add(&[atan_val, neg_pi]))
            }
        }
        _ => None,
    }
}

fn eval_sinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: sinh(-x) = -sinh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let sinh_pos = eval_sinh(arena, pos_inner).unwrap_or_else(|| arena.sinh(pos_inner));
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
        return Some(eval_cosh(arena, pos_inner).unwrap_or_else(|| arena.cosh(pos_inner)));
    }

    if inner == arena.zero {
        return Some(arena.one);
    } // cosh(0) = 1
    None
}

fn eval_tanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Odd function: tanh(-x) = -tanh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        let tanh_pos = eval_tanh(arena, pos_inner).unwrap_or_else(|| arena.tanh(pos_inner));
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
fn as_pure_imaginary(arena: &mut Arena, id: ExprId) -> Option<ExprId> {
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
            return Some(match others.len() {
                0 => arena.one,
                1 => others[0],
                _ => arena.mul(&others),
            });
        }
    }
    None
}

/// Detect if `id` represents a negated expression: `-x` or `Mul(-1, x)`.
/// Returns `Some(positive_inner)` if negated, `None` otherwise.
fn as_negated(arena: &mut Arena, id: ExprId) -> Option<ExprId> {
    match arena.node(id).clone() {
        ExprNode::Neg(inner) => Some(inner),
        ExprNode::Mul(ref children) if children.len() >= 2 => {
            if let ExprNode::Num(nid) = arena.node(children[0]) {
                let r = arena.num(*nid);
                if r.is_negative() {
                    let pos_coeff = -r.clone();
                    if pos_coeff.is_one() {
                        if children.len() == 2 {
                            return Some(children[1]);
                        } else {
                            // 3+ children: rebuild Mul without the -1
                            let rest: smallvec::SmallVec<[crate::base::node::ExprId; 6]> =
                                children[1..].iter().copied().collect();
                            return Some(arena.mul(&rest));
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
fn mod_positive(a: &Q, m: &Q) -> Q {
    let mut result = a % m;
    if result.is_negative() {
        result += m;
    }
    result
}

// ── Explicit coefficients of the classical orthogonal polynomials ─────────
//
// Up to 0.29 `p_n(x)` was built by its three-term recurrence, each step
// expanded and re-canonicalised: `O(n)` canonicalisations of polynomials
// with `O(n)` big coefficients.  `legendre(300, x)` took 6 s in `eval`, and
// `legendre(1000, x)`, `hermite(1000, x)`, `gegenbauer(1000, 1/3, x)`,
// `assoc_legendre(1000, 3, x)`, `jacobi(1000, 1/3, 1/5, x)` did not finish
// within a minute.  The coefficients have closed forms (hypergeometric sums,
// DLMF §18.5): each follows from the previous one by a rational factor, so
// the whole polynomial costs `O(n)` rational operations (Jacobi, written in
// powers of `(x − 1)/2`, needs an `O(n²)` change of basis for a symbolic
// `x`).  Every coefficient, and a numeric value, is within the digit guard.

/// A family of [`ortho_coeffs`], with its numeric parameters.
#[derive(Clone, Debug)]
enum OrthoFamily {
    Legendre,
    ChebyshevT,
    ChebyshevU,
    Hermite,
    Laguerre,
    /// `C_n^{(α)}`
    Gegenbauer(Q),
    /// `L_n^{(α)}`
    AssocLaguerre(Q),
}

fn q_int(n: i64) -> Q {
    Ratio::from_integer(BigInt::from(n))
}

/// Is the rational `r` within the digit guard (numerator and denominator
/// bits, a cheap count)?
fn within_guard_bits(arena: &Arena, r: &Q) -> bool {
    let bits = r.numer().bits() + r.denom().bits();
    (bits as f64) * std::f64::consts::LOG10_2 <= arena.config.max_result_digits as f64 + 2.0
}

/// The coefficients `c[i]` of `xⁱ` (`i = 0..=n`) of the family's `p_n`,
/// from the explicit sums (DLMF 18.5.10–18.5.13, 18.5.12 for Laguerre):
///
/// * `P_n = 2⁻ⁿ Σ_k (−1)^k C(n, k) C(2n − 2k, n) x^{n−2k}`
/// * `T_n = (n/2) Σ_k (−1)^k (n−k−1)!/(k!(n−2k)!) (2x)^{n−2k}` (`n ≥ 1`)
/// * `U_n = Σ_k (−1)^k C(n−k, k) (2x)^{n−2k}`
/// * `H_n = n! Σ_k (−1)^k (2x)^{n−2k}/(k!(n−2k)!)`
/// * `C_n^{(α)} = Σ_k (−1)^k (α)_{n−k}/(k!(n−2k)!) (2x)^{n−2k}`
/// * `L_n^{(α)} = Σ_k (−1)^k C(n+α, n−k) x^k/k!`
///
/// `None` beyond the digit guard, or when a ratio would divide by zero
/// (`α` a non-positive integer, where the recurrences degenerate).
fn ortho_coeffs(arena: &Arena, family: &OrthoFamily, n: usize) -> Option<Vec<Q>> {
    let mut c = vec![Q::zero(); n + 1];
    let ni = n as i64;
    let even_family = !matches!(
        family,
        OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_)
    );
    if even_family {
        // Leading coefficient, then c[n−2k−2] = c[n−2k] · ratio(k).
        let mut lead = match family {
            OrthoFamily::Legendre => {
                // C(2n, n)/2ⁿ
                Ratio::new(
                    crate::base::combinatorics::binomial(2 * n as u64, n as u64),
                    BigInt::one() << n,
                )
            }
            OrthoFamily::ChebyshevT => {
                if n == 0 {
                    Q::one()
                } else {
                    Ratio::from_integer(BigInt::one() << (n - 1))
                }
            }
            OrthoFamily::ChebyshevU | OrthoFamily::Hermite => {
                Ratio::from_integer(BigInt::one() << n)
            }
            OrthoFamily::Gegenbauer(a) => {
                // (α)_n 2ⁿ/n!
                let mut acc = Q::one();
                for j in 0..ni {
                    acc *= a + q_int(j);
                    acc /= q_int(j + 1);
                    acc *= q_int(2);
                }
                acc
            }
            OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_) => return None,
        };
        if matches!(family, OrthoFamily::ChebyshevT) && n == 0 {
            c[0] = Q::one();
            return Some(c);
        }
        for k in 0..=(ni / 2) {
            if !within_guard_bits(arena, &lead) {
                return None;
            }
            c[(ni - 2 * k) as usize] = lead.clone();
            if 2 * k + 2 > ni {
                break;
            }
            let (m, m1) = (q_int(ni - 2 * k), q_int(ni - 2 * k - 1));
            let kp1 = q_int(k + 1);
            let den = match family {
                OrthoFamily::Legendre => q_int(2) * &kp1 * q_int(2 * ni - 2 * k - 1),
                OrthoFamily::ChebyshevT => q_int(4) * &kp1 * q_int(ni - k - 1),
                OrthoFamily::ChebyshevU => q_int(4) * &kp1 * q_int(ni - k),
                OrthoFamily::Hermite => q_int(4) * &kp1,
                OrthoFamily::Gegenbauer(a) => q_int(4) * &kp1 * (q_int(ni - k - 1) + a),
                OrthoFamily::Laguerre | OrthoFamily::AssocLaguerre(_) => return None,
            };
            if den.is_zero() {
                return None;
            }
            lead = -(lead * m * m1) / den;
        }
        return Some(c);
    }
    // Laguerre families: c[0] = C(n+α, n), c[k+1] = −c[k]·(n−k)/((k+1)(α+k+1)).
    let alpha = match family {
        OrthoFamily::AssocLaguerre(a) => a.clone(),
        _ => Q::zero(),
    };
    let mut term = Q::one();
    for j in 0..ni {
        term *= &alpha + q_int(j + 1);
        term /= q_int(j + 1);
    }
    for k in 0..=ni {
        if !within_guard_bits(arena, &term) {
            return None;
        }
        c[k as usize] = term.clone();
        if k == ni {
            break;
        }
        let den = q_int(k + 1) * (&alpha + q_int(k + 1));
        if den.is_zero() {
            return None;
        }
        term = -(term * q_int(ni - k)) / den;
    }
    Some(c)
}

/// The Jacobi coefficients `d[ℓ]` of `((x − 1)/2)^ℓ` in `P_n^{(α,β)}`
/// (DLMF 18.5.8): `d_ℓ = (n+α+β+1)_ℓ (α+ℓ+1)_{n−ℓ}/(ℓ!(n−ℓ)!)`,
/// `d_{ℓ+1} = d_ℓ (n+α+β+1+ℓ)(n−ℓ)/((ℓ+1)(α+ℓ+1))`.
fn jacobi_shifted_coeffs(arena: &Arena, n: usize, a: &Q, b: &Q) -> Option<Vec<Q>> {
    let ni = n as i64;
    let mut d = Vec::with_capacity(n + 1);
    // d_0 = (α+1)_n/n!
    let mut term = Q::one();
    for j in 0..ni {
        term *= a + q_int(j + 1);
        term /= q_int(j + 1);
    }
    for l in 0..=ni {
        if !within_guard_bits(arena, &term) {
            return None;
        }
        d.push(term.clone());
        if l == ni {
            break;
        }
        let den = q_int(l + 1) * (a + q_int(l + 1));
        if den.is_zero() {
            return None;
        }
        term = term * (q_int(ni + 1 + l) + a + b) * q_int(ni - l) / den;
    }
    Some(d)
}

/// `Σ c[i] tⁱ` by Horner's rule, refused (`None`) as soon as the partial
/// value outgrows twice the digit guard (`chebyshev_t(1000, 1/10^20)` built
/// 20 000-digit intermediates for 11 s before the final guard refused).
fn horner(arena: &Arena, c: &[Q], t: &Q) -> Option<Q> {
    let limit_bits =
        2.0 * arena.config.max_result_digits as f64 / std::f64::consts::LOG10_2 + 256.0;
    let mut acc = Q::zero();
    for ci in c.iter().rev() {
        acc = acc * t + ci;
        if (acc.numer().bits() + acc.denom().bits()) as f64 > limit_bits {
            return None;
        }
    }
    Some(acc)
}

/// `Σ c[i] xⁱ` as an expression; expanded when `x` is a compound (the
/// recurrences expanded at every step, so `legendre(2, y + 1)` is
/// `3/2·y² + 3y + 1`).  `None` if a coefficient is beyond the digit guard.
fn poly_from_coeffs(arena: &mut Arena, c: &[Q], x: ExprId) -> Option<ExprId> {
    let mut terms = Vec::with_capacity(c.len());
    for (i, ci) in c.iter().enumerate() {
        if ci.is_zero() {
            continue;
        }
        let coeff = guarded_num(arena, ci.clone())?;
        let term = match i {
            0 => coeff,
            1 => arena.mul(&[coeff, x]),
            _ => {
                let e = arena.int(i as i64);
                let p = arena.pow(x, e);
                arena.mul(&[coeff, p])
            }
        };
        terms.push(term);
    }
    let sum = match terms.len() {
        0 => arena.zero,
        1 => terms[0],
        _ => arena.add(&terms),
    };
    if matches!(arena.node(x), ExprNode::Symbol(_)) {
        Some(sum)
    } else {
        Some(expand_eval(arena, sum))
    }
}

/// `p_n(x)` of `family` from its coefficients: the exact value for a
/// rational `x`, else the explicit polynomial.
fn ortho_value(arena: &mut Arena, family: &OrthoFamily, n: usize, x: ExprId) -> Option<ExprId> {
    let c = ortho_coeffs(arena, family, n)?;
    if let Some(xr) = arena.as_num(x).cloned() {
        let v = horner(arena, &c, &xr)?;
        return guarded_num(arena, v);
    }
    poly_from_coeffs(arena, &c, x)
}

/// `P_n^{(α,β)}(x)` for numeric `α`, `β`: Horner in `(x − 1)/2` for a
/// rational `x`, else the monomial coefficients
/// `Σ_{ℓ≥j} d_ℓ 2^{−ℓ} C(ℓ, j) (−1)^{ℓ−j}` of `xʲ`.
fn jacobi_value(arena: &mut Arena, n: usize, a: &Q, b: &Q, x: ExprId) -> Option<ExprId> {
    let d = jacobi_shifted_coeffs(arena, n, a, b)?;
    if let Some(xr) = arena.as_num(x).cloned() {
        let y = (xr - Q::one()) / q_int(2);
        let v = horner(arena, &d, &y)?;
        return guarded_num(arena, v);
    }
    // With u = x − 1, Σ d_ℓ (u/2)^ℓ = Σ e_ℓ u^ℓ (e_ℓ = d_ℓ/2^ℓ); over a
    // common denominator the Taylor shift u = x − 1 is `O(n²)` integer
    // subtractions (Knuth's scheme: a_j −= a_{j+1} for j = n−1 … i).
    let e: Vec<Q> = d
        .iter()
        .enumerate()
        .map(|(l, dl)| dl / Ratio::from_integer(BigInt::one() << l))
        .collect();
    let mut den = BigInt::one();
    for el in &e {
        den = num_integer::Integer::lcm(&den, el.denom());
    }
    let mut a: Vec<BigInt> = e
        .iter()
        .map(|el| el.numer() * (&den / el.denom()))
        .collect();
    for i in 0..n {
        for j in (i..n).rev() {
            let next = a[j + 1].clone();
            a[j] -= next;
        }
    }
    let c: Vec<Q> = a
        .into_iter()
        .map(|aj| Ratio::new(aj, den.clone()))
        .collect();
    poly_from_coeffs(arena, &c, x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomial evaluation via recurrence relations
// ═══════════════════════════════════════════════════════════════════════════

/// Legendre polynomial P_n(x) via Bonnet's recurrence.
/// P_0(x) = 1, P_1(x) = x, (k+1)·P_{k+1} = (2k+1)·x·P_k − k·P_{k-1}
#[cfg(test)]
fn eval_legendre(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return x;
    }

    let mut p_prev = arena.one;
    let mut p_curr = x;

    for k in 1..n {
        let two_k_plus_1 = arena.int(2 * k as i64 + 1);
        let k_val = arena.int(k as i64);
        let k_plus_1 = arena.int(k as i64 + 1);

        let term1 = arena.mul(&[two_k_plus_1, x, p_curr]);
        let term2 = arena.mul(&[k_val, p_prev]);
        let numer = arena.sub(term1, term2);
        let p_next = arena.div(numer, k_plus_1);

        // Expand and eval at each step to keep expressions manageable
        let p_next = crate::transforms::expand::expand(arena, p_next);
        let p_next = eval(arena, p_next);

        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Chebyshev polynomial of the first kind T_n(x) via recurrence.
/// T_0(x) = 1, T_1(x) = x, T_{k+1} = 2·x·T_k − T_{k-1}
#[cfg(test)]
fn eval_chebyshev_t(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return x;
    }

    let two = arena.int(2);
    let mut t_prev = arena.one;
    let mut t_curr = x;

    for _ in 1..n {
        let term1 = arena.mul(&[two, x, t_curr]);
        let t_next = arena.sub(term1, t_prev);

        let t_next = crate::transforms::expand::expand(arena, t_next);
        let t_next = eval(arena, t_next);

        t_prev = t_curr;
        t_curr = t_next;
    }

    t_curr
}

/// Chebyshev polynomial of the second kind U_n(x) via recurrence.
/// U_0(x) = 1, U_1(x) = 2x, U_{k+1} = 2·x·U_k − U_{k-1}
#[cfg(test)]
fn eval_chebyshev_u(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    let two = arena.int(2);
    if n == 1 {
        return arena.mul(&[two, x]);
    }

    let mut u_prev = arena.one;
    let mut u_curr = arena.mul(&[two, x]);

    for _ in 1..n {
        let term1 = arena.mul(&[two, x, u_curr]);
        let u_next = arena.sub(term1, u_prev);

        let u_next = crate::transforms::expand::expand(arena, u_next);
        let u_next = eval(arena, u_next);

        u_prev = u_curr;
        u_curr = u_next;
    }

    u_curr
}

/// Physicist's Hermite polynomial H_n(x) via recurrence.
/// H_0(x) = 1, H_1(x) = 2x, H_{k+1} = 2·x·H_k − 2k·H_{k-1}
#[cfg(test)]
fn eval_hermite(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    let two = arena.int(2);
    if n == 1 {
        return arena.mul(&[two, x]);
    }

    let mut h_prev = arena.one;
    let mut h_curr = arena.mul(&[two, x]);

    for k in 1..n {
        let two_k = arena.int(2 * k as i64);
        let term1 = arena.mul(&[two, x, h_curr]);
        let term2 = arena.mul(&[two_k, h_prev]);
        let h_next = arena.sub(term1, term2);

        let h_next = crate::transforms::expand::expand(arena, h_next);
        let h_next = eval(arena, h_next);

        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

/// Laguerre polynomial L_n(x) via recurrence.
/// L_0(x) = 1, L_1(x) = 1−x, (k+1)·L_{k+1} = (2k+1−x)·L_k − k·L_{k-1}
#[cfg(test)]
fn eval_laguerre(arena: &mut Arena, n: usize, x: ExprId) -> ExprId {
    if n == 0 {
        return arena.one;
    }
    if n == 1 {
        return arena.sub(arena.one, x);
    }

    let mut l_prev = arena.one;
    let mut l_curr = arena.sub(arena.one, x);

    for k in 1..n {
        let two_k_plus_1 = arena.int(2 * k as i64 + 1);
        let k_val = arena.int(k as i64);
        let k_plus_1 = arena.int(k as i64 + 1);

        let coeff = arena.sub(two_k_plus_1, x); // (2k+1 − x)
        let term1 = arena.mul(&[coeff, l_curr]);
        let term2 = arena.mul(&[k_val, l_prev]);
        let numer = arena.sub(term1, term2);
        let l_next = arena.div(numer, k_plus_1);

        let l_next = crate::transforms::expand::expand(arena, l_next);
        let l_next = eval(arena, l_next);

        l_prev = l_curr;
        l_curr = l_next;
    }

    l_curr
}

// ═══════════════════════════════════════════════════════════════════════════
// Library functions: exact values
// ═══════════════════════════════════════════════════════════════════════════

/// Exact value of the library function `f` on `args` (already evaluated and
/// of the arity `f` declares); `None` leaves the node as is.
///
/// Exhaustive over [`LibFn`]: a function with no exact rules (`bessely`,
/// `besselk`, `lambertw`) says so here rather than in a wildcard arm.
fn eval_lib_fn(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> Option<ExprId> {
    match f {
        // ── Integer sequences and combinatorial counts ──
        LibFn::Factorial2 => eval_factorial2(arena, args[0]),
        LibFn::Subfactorial => eval_subfactorial(arena, args[0]),
        LibFn::RisingFactorial => eval_rising_factorial(arena, args[0], args[1]),
        LibFn::FallingFactorial => eval_falling_factorial(arena, args[0], args[1]),
        LibFn::Fibonacci => eval_fibonacci(arena, args[0]),
        LibFn::Lucas => eval_lucas(arena, args[0]),
        LibFn::Bernoulli => eval_bernoulli(arena, args[0]),
        LibFn::Harmonic => eval_harmonic(arena, args[0]),
        LibFn::Catalan => eval_catalan(arena, args[0]),
        LibFn::Bell => eval_bell(arena, args[0]),
        LibFn::EulerNumber => eval_euler_number(arena, args[0]),
        LibFn::Stirling1 => eval_stirling1(arena, args[0], args[1]),
        LibFn::Stirling2 => eval_stirling2(arena, args[0], args[1]),
        LibFn::PartitionCount => eval_partition_count(arena, args[0]),
        // The arena builds `ExprNode::LambertW`, which has its own rules.
        LibFn::LambertW => None,

        // ── Bessel functions: J and I at the origin; Y and K diverge there ──
        LibFn::BesselJ | LibFn::BesselI => eval_bessel_regular_at_zero(arena, args[0], args[1]),
        LibFn::BesselY | LibFn::BesselK => None,

        // ── Classical orthogonal polynomials of small explicit degree ──
        LibFn::Legendre => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Legendre),
        LibFn::ChebyshevT => eval_orthopoly(arena, args[0], args[1], OrthoFamily::ChebyshevT),
        LibFn::ChebyshevU => eval_orthopoly(arena, args[0], args[1], OrthoFamily::ChebyshevU),
        LibFn::Hermite => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Hermite),
        LibFn::Laguerre => eval_orthopoly(arena, args[0], args[1], OrthoFamily::Laguerre),

        // ── More special functions (0.9) ──
        LibFn::Erfi => eval_erfi(arena, args[0]),
        LibFn::ErfInv => eval_erfinv(arena, args[0]),
        LibFn::ErfcInv => eval_erfcinv(arena, args[0]),
        LibFn::ExpInt => eval_expint(arena, args[0], args[1]),
        LibFn::Shi => eval_shi(arena, args[0]),
        LibFn::Chi => eval_chi(arena, args[0]),
        LibFn::FresnelS | LibFn::FresnelC => eval_fresnel(arena, f, args[0]),
        LibFn::LowerGamma => eval_lowergamma(arena, args[0], args[1]),
        LibFn::UpperGamma => eval_uppergamma(arena, args[0], args[1]),
        LibFn::PolyLog => eval_polylog(arena, args[0], args[1]),
        LibFn::DirichletEta => eval_dirichlet_eta(arena, args[0]),
        LibFn::AiryAi | LibFn::AiryBi | LibFn::AiryAiPrime | LibFn::AiryBiPrime => {
            eval_airy(arena, f, args[0])
        }
        LibFn::EllipticK => eval_elliptic_k(arena, args[0]),
        LibFn::EllipticE => eval_elliptic_e(arena, args[0]),
        LibFn::EllipticF => eval_elliptic_f(arena, args[0], args[1]),
        LibFn::EllipticPi => eval_elliptic_pi(arena, args[0], args[1]),
        LibFn::Gegenbauer => eval_gegenbauer(arena, args[0], args[1], args[2]),
        LibFn::Jacobi => eval_jacobi(arena, args[0], args[1], args[2], args[3]),
        LibFn::AssocLegendre => eval_assoc_legendre(arena, args[0], args[1], args[2]),
        LibFn::AssocLaguerre => eval_assoc_laguerre(arena, args[0], args[1], args[2]),
        LibFn::BetaInc | LibFn::BetaIncRegularized => eval_betainc(
            arena,
            f == LibFn::BetaIncRegularized,
            args[0],
            args[1],
            args[2],
            args[3],
        ),
    }
}

/// `J_0(0) = I_0(0) = 1` and `J_n(0) = I_n(0) = 0` for integer `n > 0`, when
/// both order and argument are numbers.
fn eval_bessel_regular_at_zero(arena: &mut Arena, order: ExprId, x: ExprId) -> Option<ExprId> {
    let order_num = arena.as_num(order)?;
    let arg_num = arena.as_num(x)?;
    if !arg_num.is_zero() {
        return None;
    }
    if order_num.is_zero() {
        Some(arena.one)
    } else if order_num.is_positive() && order_num.is_integer() {
        Some(arena.zero)
    } else {
        None
    }
}

/// A classical orthogonal polynomial `p_n(x)` expanded by `expand` when `n`
/// is a non-negative integer no larger than the configured power limit.
fn eval_orthopoly(arena: &mut Arena, n: ExprId, x: ExprId, family: OrthoFamily) -> Option<ExprId> {
    let n_num = arena.as_num(n)?;
    if !n_num.is_integer() || n_num.is_negative() {
        return None;
    }
    let n_int = n_num.to_integer().to_u64()?;
    if n_int > arena.config.max_pow_exponent as u64 {
        return None;
    }
    ortho_value(arena, &family, n_int as usize, x)
}

/// Intern `f(args)` as a library `Apply` node (no folding).
pub(crate) fn apply_named(arena: &mut Arena, f: LibFn, args: &[ExprId]) -> ExprId {
    arena.lib_apply(f, args)
}

/// Largest integer / half-integer parameter for which the incomplete gamma
/// functions, `polylog(−n, z)` and the parametrised orthogonal polynomials
/// are expanded into closed forms.
const MAX_SPECIAL_EXPANSION: i64 = 64;

/// `id` as a small integer in `[lo, hi]`.
fn as_int_in(arena: &Arena, id: ExprId, lo: i64, hi: i64) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    (lo..=hi).contains(&n).then_some(n)
}

/// `id` as a rational number.
fn as_ratio(arena: &Arena, id: ExprId) -> Option<Q> {
    arena.as_num(id).cloned()
}

/// `π/2`.
fn half_pi(arena: &mut Arena) -> ExprId {
    let half = arena.rational(1, 2);
    arena.mul(&[half, arena.pi])
}

/// `erfi(0) = 0`, `erfi(±∞) = ±∞`, odd.
fn eval_erfi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.infinity {
        return Some(arena.infinity);
    }
    if x == arena.neg_infinity {
        return Some(arena.neg_infinity);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::Erfi, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `erfinv(0) = 0`, `erfinv(±1) = ±∞`, odd.
fn eval_erfinv(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.one {
        return Some(arena.infinity);
    }
    if x == arena.neg_one {
        return Some(arena.neg_infinity);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::ErfInv, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `erfcinv(1) = 0`, `erfcinv(0) = ∞`, `erfcinv(2) = −∞`.
fn eval_erfcinv(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.one {
        return Some(arena.zero);
    }
    if x == arena.zero {
        return Some(arena.infinity);
    }
    if as_int_in(arena, x, 2, 2).is_some() {
        return Some(arena.neg_infinity);
    }
    None
}

/// `E_n(∞) = 0`, `E_n(0) = 1/(n−1)` for `n > 1`, `E_0(x) = e^{−x}/x`.
fn eval_expint(arena: &mut Arena, n: ExprId, x: ExprId) -> Option<ExprId> {
    if x == arena.infinity {
        return Some(arena.zero);
    }
    if n == arena.zero {
        let neg_x = arena.neg(x);
        let e = arena.exp(neg_x);
        return Some(arena.div(e, x));
    }
    if x == arena.zero
        && let Some(r) = as_ratio(arena, n)
        && r > Ratio::one()
    {
        let v = (r - Ratio::one()).recip();
        let nid = arena.intern_num(v);
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    None
}

/// `Shi(0) = 0`, `Shi(±∞) = ±∞`, odd.
fn eval_shi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.infinity {
        return Some(arena.infinity);
    }
    if x == arena.neg_infinity {
        return Some(arena.neg_infinity);
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, LibFn::Shi, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `Chi(0) = −∞`, `Chi(∞) = ∞`.
fn eval_chi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.neg_infinity);
    }
    if x == arena.infinity {
        return Some(arena.infinity);
    }
    None
}

/// Fresnel `S`/`C`: `0 ↦ 0`, `±∞ ↦ ±1/2`, odd.
fn eval_fresnel(arena: &mut Arena, f: LibFn, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.zero);
    }
    if x == arena.infinity {
        return Some(arena.rational(1, 2));
    }
    if x == arena.neg_infinity {
        return Some(arena.rational(-1, 2));
    }
    if let Some(y) = as_negated_general(arena, x) {
        let e = apply_named(arena, f, &[y]);
        return Some(arena.neg(e));
    }
    None
}

/// `Γ(s, x)` for `s = base + k` (`k ∈ ℤ`) from the base value `Γ(base, x)`
/// via `Γ(s+1, x) = s Γ(s, x) + x^s e^{−x}`, upward or downward.
fn shift_uppergamma(arena: &mut Arena, base: Q, base_val: ExprId, k: i64, x: ExprId) -> ExprId {
    let neg_x = arena.neg(x);
    let e_neg_x = arena.exp(neg_x);
    let mut s = base;
    let mut val = base_val;
    if k >= 0 {
        for _ in 0..k {
            // Γ(s+1) = s Γ(s) + x^s e^{-x}
            let s_id = arena.intern_num(s.clone());
            let s_id = arena.intern(ExprNode::Num(s_id));
            let x_pow = arena.pow(x, s_id);
            let a = arena.mul(&[s_id, val]);
            let b = arena.mul(&[x_pow, e_neg_x]);
            val = arena.add(&[a, b]);
            s += Ratio::one();
        }
    } else {
        for _ in 0..(-k) {
            // Γ(s−1) = (Γ(s) − x^{s−1} e^{-x}) / (s−1)
            s -= Ratio::one();
            let s_id = arena.intern_num(s.clone());
            let s_id = arena.intern(ExprNode::Num(s_id));
            let x_pow = arena.pow(x, s_id);
            let b = arena.mul(&[x_pow, e_neg_x]);
            let numer = arena.sub(val, b);
            val = arena.div(numer, s_id);
        }
    }
    let expanded = crate::transforms::expand::expand(arena, val);
    eval(arena, expanded)
}

/// Closed form of `Γ(s, x)` for integer or half-integer `s` with
/// `|s| ≤ MAX_SPECIAL_EXPANSION`: bases `Γ(1, x) = e^{−x}`,
/// `Γ(0, x) = E₁(x)`, `Γ(1/2, x) = √π erfc(√x)`.
fn uppergamma_closed(arena: &mut Arena, s: &Q, x: ExprId) -> Option<ExprId> {
    let two = BigInt::from(2);
    let bound = Ratio::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION));
    if s.abs() > bound {
        return None;
    }
    let neg_x = arena.neg(x);
    let e_neg_x = arena.exp(neg_x);
    if s.is_integer() {
        let n: i64 = s.to_integer().try_into().ok()?;
        if n >= 1 {
            // Γ(n, x) = (n−1)! e^{−x} Σ_{k<n} x^k/k!
            let mut terms: Vec<ExprId> = Vec::with_capacity(n as usize);
            let mut inv_fact = Ratio::<BigInt>::one();
            for k in 0..n {
                if k > 0 {
                    inv_fact /= Ratio::from_integer(BigInt::from(k));
                }
                let c = arena.intern_num(inv_fact.clone());
                let c = arena.intern(ExprNode::Num(c));
                let k_id = arena.int(k);
                let xk = arena.pow(x, k_id);
                terms.push(arena.mul(&[c, xk]));
            }
            let sum = arena.add(&terms);
            let fact = factorial_ratio((n - 1) as usize);
            let f = arena.intern_num(fact);
            let f = arena.intern(ExprNode::Num(f));
            let v = arena.mul(&[f, e_neg_x, sum]);
            let v = crate::transforms::expand::expand(arena, v);
            return Some(eval(arena, v));
        }
        // n ≤ 0: shift down from Γ(0, x) = E₁(x).
        let e1 = apply_named(arena, LibFn::ExpInt, &[arena.one, x]);
        return Some(shift_uppergamma(arena, Ratio::zero(), e1, n, x));
    }
    if s.denom() == &two {
        // s = 1/2 + k
        let half = Ratio::new(BigInt::one(), two);
        let k: i64 = (s - &half).to_integer().try_into().ok()?;
        let sqrt_pi = arena.sqrt(arena.pi);
        let sqrt_x = arena.sqrt(x);
        let erfc = arena.erfc(sqrt_x);
        let base = arena.mul(&[sqrt_pi, erfc]);
        return Some(shift_uppergamma(arena, half, base, k, x));
    }
    None
}

/// `s` is *known* to be non-positive at the arena level: a numeric `s ≤ 0`
/// or `−∞`.  (Symbolic parameters stay `false` — assumptions are not
/// visible here — so folds that hold for `s > 0` are kept for them, which
/// the Gamma-distribution CDF `γ(k, 0)/Γ(k) = 0` relies on.)
fn known_nonpositive(arena: &Arena, s: ExprId) -> bool {
    if s == arena.neg_infinity {
        return true;
    }
    arena.as_num(s).is_some_and(|r| !r.is_positive())
}

/// Bits the closed form `γ(s, x) = Γ(s) − Γ(s, x)` may lose to cancellation
/// at a rational `x` before the fold keeps `γ(s, x)` instead.
const LOWERGAMMA_FOLD_MAX_LOSS_BITS: f64 = 64.0;

/// `ln q` for a positive rational, without overflowing on huge numerators
/// or denominators.
fn ln_positive_ratio(q: &Q) -> f64 {
    fn ln_big(n: &BigInt) -> f64 {
        let bits = n.bits();
        if bits <= 1000 {
            return n.to_f64().unwrap_or(f64::INFINITY).ln();
        }
        let shift = bits - 64;
        let top: BigInt = n >> shift;
        top.to_f64().unwrap_or(f64::INFINITY).ln() + shift as f64 * std::f64::consts::LN_2
    }
    ln_big(q.numer()) - ln_big(q.denom())
}

/// Does the closed form `Γ(s) − Γ(s, x)` of `γ(s, x)`, for an integer or
/// half-integer `s > 0` and a rational `x > 0`, subtract two numbers that
/// agree to more than [`LOWERGAMMA_FOLD_MAX_LOSS_BITS`]?  It loses about
/// `−log₂ P(s, x)` bits (`P = γ/Γ`, regularised), which is large only for
/// `x` well below `s`, where `P(s, x) ≤ xˢ e⁻ˣ/Γ(s + 1) · (s + 1)/(s + 1 − x)`
/// (the power series).  There the closed form is an exact expression that
/// evaluates to `0`: `γ(61, 1)/60!` (a Poisson(1) tail, `7.4·10⁻⁸⁵`) is
/// `1 − e⁻¹ Σ_{k ≤ 60} 1/k!`, `γ(5, 10⁻³⁰)` loses 500 bits.
fn lowergamma_closure_cancels(s: &Q, x: &Q) -> bool {
    if !x.is_positive() || x >= s {
        return false;
    }
    let Some(x_f) = x.to_f64() else {
        return false;
    };
    lowergamma_closure_cancels_at(s, x_f, ln_positive_ratio(x))
}

/// [`lowergamma_closure_cancels`] for a real `0 < x < s` known as `x ≈ x_f`,
/// `ln x ≈ ln_x` (`x_f` may underflow to 0).  Before 0.29 only a rational
/// `x` was guarded: `γ(5, √2/10³⁰)` folded into a closed form that
/// evaluated to `0` (true value `1.13·10⁻¹⁵⁰`).
fn lowergamma_closure_cancels_at(s: &Q, x_f: f64, ln_x: f64) -> bool {
    let half_integer = s.denom() == &BigInt::from(2);
    if !(s.is_integer() || half_integer)
        || *s > Q::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION))
    {
        return false;
    }
    let Some(s_f) = s.to_f64() else {
        return false;
    };
    // ln Γ(s + 1) for an integer or half-integer s ≤ MAX_SPECIAL_EXPANSION.
    let mut ln_gamma = if s.is_integer() {
        0.0
    } else {
        0.5 * std::f64::consts::PI.ln()
    };
    let mut k = if s.is_integer() { 2.0 } else { 0.5 };
    while k <= s_f + 0.25 {
        ln_gamma += f64::ln(k);
        k += 1.0;
    }
    let ln_p = s_f * ln_x - x_f - ln_gamma + ((s_f + 1.0) / (s_f + 1.0 - x_f)).ln();
    -ln_p / std::f64::consts::LN_2 > LOWERGAMMA_FOLD_MAX_LOSS_BITS
}

/// `γ(s, 0) = 0`, `γ(s, ∞) = Γ(s)`, closed forms for integer / half-integer `s > 0`.
///
/// `γ(s, 0) = 0` holds only for `s > 0` (`∫₀ˣ t^{s−1} e^{−t} dt` diverges at
/// the lower end otherwise), so that fold is refused for `s` known to be `≤ 0`.
/// The closed form is also refused at a real constant `x` where it would
/// cancel catastrophically ([`lowergamma_closure_cancels`]): `γ(s, x)`
/// stays, and `evalf` computes it by its power series.  An irrational `x`
/// (`√2/10³⁰`, `π/10¹⁰`) is sized by a 16-digit evaluation.
fn eval_lowergamma(arena: &mut Arena, s: ExprId, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        if known_nonpositive(arena, s) {
            return None;
        }
        return Some(arena.zero);
    }
    if x == arena.infinity {
        return Some(arena.gamma(s));
    }
    let r = as_ratio(arena, s)?;
    if !r.is_positive() {
        return None;
    }
    if let Some(xr) = as_ratio(arena, x) {
        if lowergamma_closure_cancels(&r, &xr) {
            return None;
        }
    } else if let Some(ln_x) = crate::transforms::evalf::ln_of_positive_constant(arena, x)
        && r.to_f64().is_some_and(|s_f| ln_x < s_f.ln())
        && lowergamma_closure_cancels_at(&r, ln_x.exp(), ln_x)
    {
        return None;
    }
    let upper = uppergamma_closed(arena, &r, x)?;
    let gamma_s = eval_gamma(arena, s).unwrap_or_else(|| arena.gamma(s));
    let v = arena.sub(gamma_s, upper);
    let v = crate::transforms::expand::expand(arena, v);
    Some(eval(arena, v))
}

/// `Γ(s, 0) = Γ(s)`, `Γ(s, ∞) = 0`, closed forms for integer / half-integer `s`.
///
/// `Γ(s, 0) = ∫₀^∞ t^{s−1} e^{−t} dt` diverges for `s ≤ 0` (`Γ(s)` there is
/// the analytic continuation, not this integral), so the fold at `x = 0` is
/// refused for `s` known to be non-positive.
fn eval_uppergamma(arena: &mut Arena, s: ExprId, x: ExprId) -> Option<ExprId> {
    if x == arena.infinity {
        return Some(arena.zero);
    }
    if x == arena.zero {
        if known_nonpositive(arena, s) {
            return None;
        }
        return Some(arena.gamma(s));
    }
    let r = as_ratio(arena, s)?;
    uppergamma_closed(arena, &r, x)
}

/// Exact polylogarithm values (see [`Ex::polylog`](crate::expr::Ex::polylog)).
fn eval_polylog(arena: &mut Arena, s: ExprId, z: ExprId) -> Option<ExprId> {
    if z == arena.zero {
        return Some(arena.zero);
    }
    if z == arena.one {
        return Some(arena.zeta(s));
    }
    if z == arena.neg_one {
        let eta = eval_dirichlet_eta(arena, s)
            .unwrap_or_else(|| apply_named(arena, LibFn::DirichletEta, &[s]));
        return Some(arena.neg(eta));
    }
    if s == arena.one {
        let one_minus_z = arena.sub(arena.one, z);
        let l = arena.ln(one_minus_z);
        return Some(arena.neg(l));
    }
    if s == arena.zero {
        let one_minus_z = arena.sub(arena.one, z);
        return Some(arena.div(z, one_minus_z));
    }
    // Li₂(1/2) = π²/12 − ln²2/2
    if as_int_in(arena, s, 2, 2).is_some()
        && let Some(r) = as_ratio(arena, z)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        let two = arena.int(2);
        let pi2 = arena.pow(arena.pi, two);
        let c12 = arena.rational(1, 12);
        let a = arena.mul(&[c12, pi2]);
        let ln2 = arena.ln(two);
        let ln2_sq = arena.pow(ln2, two);
        let half = arena.rational(1, 2);
        let b = arena.mul(&[half, ln2_sq]);
        return Some(arena.sub(a, b));
    }
    // Li_{−n}(z) = [Σ_{k=0}^{n} k! S(n+1, k+1) z^{k+1} (1−z)^{n−k}] / (1−z)^{n+1}
    if let Some(neg_n) = as_int_in(arena, s, -MAX_SPECIAL_EXPANSION, -1) {
        let n = (-neg_n) as usize;
        let one_minus_z = arena.sub(arena.one, z);
        let mut terms: Vec<ExprId> = Vec::with_capacity(n + 1);
        let mut k_fact = BigInt::one();
        for k in 0..=n {
            if k > 0 {
                k_fact *= BigInt::from(k as u64);
            }
            let s2 = crate::domains::combinatorics::stirling2(n as u64 + 1, k as u64 + 1)?;
            let coeff = Ratio::from_integer(&k_fact * s2);
            let c = arena.intern_num(coeff);
            let c = arena.intern(ExprNode::Num(c));
            let e1 = arena.int(k as i64 + 1);
            let zk = arena.pow(z, e1);
            let e2 = arena.int((n - k) as i64);
            let wk = arena.pow(one_minus_z, e2);
            terms.push(arena.mul(&[c, zk, wk]));
        }
        let numer = arena.add(&terms);
        let numer = crate::transforms::expand::expand(arena, numer);
        let numer = eval(arena, numer);
        let e = arena.int(n as i64 + 1);
        let denom = arena.pow(one_minus_z, e);
        return Some(arena.div(numer, denom));
    }
    None
}

/// `η(1) = ln 2`, `η(∞) = 1`, `η(−2k) = 0`, and `η(s) = (1 − 2^{1−s}) ζ(s)`
/// whenever `ζ(s)` itself folds (integer `s`); negative integers below
/// `−MAX_SPECIAL_EXPANSION` are left alone (the factor `2^{1−s}` and the
/// Bernoulli number behind `ζ(s)` both grow with `|s|`).
fn eval_dirichlet_eta(arena: &mut Arena, s: ExprId) -> Option<ExprId> {
    if s == arena.one {
        let two = arena.int(2);
        return Some(arena.ln(two));
    }
    if s == arena.infinity {
        return Some(arena.one);
    }
    let r = as_ratio(arena, s)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n < -MAX_SPECIAL_EXPANSION {
        // Trivial zeros of ζ are the only cheap exact values out here.
        return (n % 2 == 0).then_some(arena.zero);
    }
    let zeta = eval_zeta(arena, s)?;
    if matches!(arena.node(zeta), ExprNode::Zeta(_)) {
        return None;
    }
    // 1 − 2^{1−s} as an exact rational.
    let e = 1 - n;
    let two_pow = if e >= 0 {
        Ratio::from_integer(BigInt::one() << (e as usize))
    } else {
        Ratio::new(BigInt::one(), BigInt::one() << ((-e) as usize))
    };
    let coeff = Ratio::one() - two_pow;
    let c = arena.intern_num(coeff);
    let c = arena.intern(ExprNode::Num(c));
    Some(arena.mul(&[c, zeta]))
}

/// Airy functions at `0` and `±∞`.
fn eval_airy(arena: &mut Arena, f: LibFn, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        let three = arena.int(3);
        let third = arena.rational(1, 3);
        let two_thirds = arena.rational(2, 3);
        let g13 = arena.gamma(third);
        let g23 = arena.gamma(two_thirds);
        return Some(match f {
            LibFn::AiryAi => {
                // 3^{-2/3} / Γ(2/3)
                let e = arena.rational(-2, 3);
                let p = arena.pow(three, e);
                arena.div(p, g23)
            }
            LibFn::AiryBi => {
                // 3^{-1/6} / Γ(2/3)
                let e = arena.rational(-1, 6);
                let p = arena.pow(three, e);
                arena.div(p, g23)
            }
            LibFn::AiryAiPrime => {
                // -3^{-1/3} / Γ(1/3)
                let e = arena.rational(-1, 3);
                let p = arena.pow(three, e);
                let q = arena.div(p, g13);
                arena.neg(q)
            }
            _ => {
                // 3^{1/6} / Γ(1/3)
                let e = arena.rational(1, 6);
                let p = arena.pow(three, e);
                arena.div(p, g13)
            }
        });
    }
    if x == arena.infinity {
        return Some(match f {
            LibFn::AiryAi | LibFn::AiryAiPrime => arena.zero,
            _ => arena.infinity,
        });
    }
    if x == arena.neg_infinity {
        return Some(arena.zero);
    }
    None
}

/// `K(0) = π/2`, `K(1) = z∞`, `K(1/2) = Γ(1/4)²/(4√π)`, `K(±∞) = 0`.
fn eval_elliptic_k(arena: &mut Arena, m: ExprId) -> Option<ExprId> {
    if m == arena.zero {
        return Some(half_pi(arena));
    }
    if m == arena.one {
        return Some(arena.complex_infinity);
    }
    if m == arena.infinity || m == arena.neg_infinity {
        return Some(arena.zero);
    }
    if let Some(r) = as_ratio(arena, m)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        let quarter = arena.rational(1, 4);
        let g = arena.gamma(quarter);
        let two = arena.int(2);
        let g2 = arena.pow(g, two);
        let sqrt_pi = arena.sqrt(arena.pi);
        let four = arena.int(4);
        let denom = arena.mul(&[four, sqrt_pi]);
        return Some(arena.div(g2, denom));
    }
    None
}

/// `E(0) = π/2`, `E(1) = 1`.
fn eval_elliptic_e(arena: &mut Arena, m: ExprId) -> Option<ExprId> {
    if m == arena.zero {
        return Some(half_pi(arena));
    }
    if m == arena.one {
        return Some(arena.one);
    }
    None
}

/// `F(0 | m) = 0`, `F(φ | 0) = φ`, `F(π/2 | m) = K(m)`, odd in `φ`.
fn eval_elliptic_f(arena: &mut Arena, phi: ExprId, m: ExprId) -> Option<ExprId> {
    if phi == arena.zero {
        return Some(arena.zero);
    }
    if m == arena.zero {
        return Some(phi);
    }
    if let Some(r) = as_pi_multiple(arena, phi)
        && r == Ratio::new(BigInt::one(), BigInt::from(2))
    {
        return Some(
            eval_elliptic_k(arena, m).unwrap_or_else(|| apply_named(arena, LibFn::EllipticK, &[m])),
        );
    }
    if let Some(y) = as_negated_general(arena, phi) {
        let e = apply_named(arena, LibFn::EllipticF, &[y, m]);
        return Some(arena.neg(e));
    }
    None
}

/// `Π(0 | m) = K(m)`, `Π(n | 0) = π/(2√(1−n))`, `Π(n | n) = E(n)/(1−n)`, `Π(1 | m) = z∞`.
fn eval_elliptic_pi(arena: &mut Arena, n: ExprId, m: ExprId) -> Option<ExprId> {
    if n == arena.zero {
        return Some(
            eval_elliptic_k(arena, m).unwrap_or_else(|| apply_named(arena, LibFn::EllipticK, &[m])),
        );
    }
    if n == arena.one {
        return Some(arena.complex_infinity);
    }
    if m == arena.zero {
        let one_minus_n = arena.sub(arena.one, n);
        let root = arena.sqrt(one_minus_n);
        let two = arena.int(2);
        let denom = arena.mul(&[two, root]);
        return Some(arena.div(arena.pi, denom));
    }
    if n == m {
        let e =
            eval_elliptic_e(arena, n).unwrap_or_else(|| apply_named(arena, LibFn::EllipticE, &[n]));
        let one_minus_n = arena.sub(arena.one, n);
        return Some(arena.div(e, one_minus_n));
    }
    None
}

/// Degree parameter of an orthogonal polynomial: a non-negative integer no
/// larger than the configured power-expansion limit.
fn as_poly_degree(arena: &Arena, n: ExprId) -> Option<usize> {
    let r = arena.as_num(n)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let d = r.to_integer().to_u64()?;
    (d <= arena.config.max_pow_exponent as u64).then_some(d as usize)
}

/// Largest degree for which the parametrised orthogonal polynomials are
/// expanded when a parameter (`a`, `b`) is not a plain number: every step
/// then expands a multivariate polynomial in the parameters and `x`, whose
/// size (and the cost of expanding it) grows quickly with the degree.
/// Numeric parameters keep the `max_pow_exponent` bound of
/// [`as_poly_degree`].
const MAX_SYMBOLIC_PARAM_DEGREE: usize = 16;

/// [`as_poly_degree`], further capped at [`MAX_SYMBOLIC_PARAM_DEGREE`]
/// unless every parameter in `params` is numeric.
fn as_param_poly_degree(arena: &Arena, n: ExprId, params: &[ExprId]) -> Option<usize> {
    let deg = as_poly_degree(arena, n)?;
    let numeric = params.iter().all(|&p| arena.as_num(p).is_some());
    (numeric || deg <= MAX_SYMBOLIC_PARAM_DEGREE).then_some(deg)
}

/// The degree for the recurrence path of the parametrised polynomials:
/// symbolic parameters (see [`as_param_poly_degree`]) or numeric ones
/// where the explicit coefficients degenerate (a non-positive integer
/// Gegenbauer `α`, a negative integer Laguerre / Jacobi `α`), capped at
/// [`MAX_DEGENERATE_DEGREE`] for the latter since the recurrence
/// re-expands the polynomial at every step.
fn degenerate_param_degree(arena: &Arena, n: ExprId, params: &[ExprId]) -> Option<usize> {
    let deg = as_param_poly_degree(arena, n, params)?;
    let numeric = params.iter().all(|&p| arena.as_num(p).is_some());
    (!numeric || deg <= MAX_DEGENERATE_DEGREE).then_some(deg)
}

/// Largest degree expanded by the recurrences for degenerate numeric
/// parameters (see [`degenerate_param_degree`]).
const MAX_DEGENERATE_DEGREE: usize = 64;

/// Expand and fold one recurrence step.
fn expand_eval(arena: &mut Arena, e: ExprId) -> ExprId {
    let e = crate::transforms::expand::expand(arena, e);
    eval(arena, e)
}

/// `p · q` distributed (both already expanded, so like terms are merged
/// at every step instead of once at the end); not folded.
fn expand_mul2(arena: &mut Arena, p: ExprId, q: ExprId) -> ExprId {
    let m = arena.mul(&[p, q]);
    crate::transforms::expand::expand(arena, m)
}

/// Gegenbauer `C_n^{(a)}(x)`: explicit polynomial for integer `n ≥ 0` via
/// `(k+1) C_{k+1} = 2(k+a) x C_k − (k+2a−1) C_{k−1}`; `C_n^{(1/2)} = P_n`,
/// `C_n^{(1)} = U_n`.
fn eval_gegenbauer(arena: &mut Arena, n: ExprId, a: ExprId, x: ExprId) -> Option<ExprId> {
    if let (Some(deg), Some(alpha)) = (as_poly_degree(arena, n), as_ratio(arena, a))
        && !(alpha.is_integer() && !alpha.is_positive())
    {
        return ortho_value(arena, &OrthoFamily::Gegenbauer(alpha), deg, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let two = arena.int(2);
        let c1 = arena.mul(&[two, a, x]);
        if deg == 1 {
            return Some(expand_eval(arena, c1));
        }
        let mut prev = arena.one;
        let mut curr = expand_eval(arena, c1);
        for k in 1..deg {
            let k_id = arena.int(k as i64);
            let k_plus_a = arena.add(&[k_id, a]);
            let t1 = arena.mul(&[two, k_plus_a, x, curr]);
            let two_a = arena.mul(&[two, a]);
            let km1 = arena.int(k as i64 - 1);
            let coeff = arena.add(&[km1, two_a]);
            let t2 = arena.mul(&[coeff, prev]);
            let numer = arena.sub(t1, t2);
            let kp1 = arena.int(k as i64 + 1);
            let next = arena.div(numer, kp1);
            let next = expand_eval(arena, next);
            prev = curr;
            curr = next;
        }
        return Some(curr);
    }
    if let Some(r) = as_ratio(arena, a) {
        if r == Ratio::new(BigInt::one(), BigInt::from(2)) {
            return Some(arena.legendre(n, x));
        }
        if r.is_one() {
            return Some(arena.chebyshev_u(n, x));
        }
    }
    None
}

/// `[1, p, p², …, p^n]` with every power expanded (each from the previous
/// one, so intermediate like terms are merged).
fn expanded_powers(arena: &mut Arena, p: ExprId, n: usize) -> Vec<ExprId> {
    let mut pows: Vec<ExprId> = Vec::with_capacity(n + 1);
    pows.push(arena.one);
    for k in 1..=n {
        let next = expand_mul2(arena, pows[k - 1], p);
        pows.push(next);
    }
    pows
}

/// Jacobi `P_n^{(a,b)}(x)` for integer `n ≥ 0`:
/// `Σ_{s=0}^{n} C(n+a, n−s) C(n+b, s) ((x−1)/2)^s ((x+1)/2)^{n−s}` with the
/// binomials written as polynomials in `a`, `b`; `P_n^{(0,0)} = P_n`.
///
/// Every factor (`∏ (a+j)`, `∏ (b+j)`, the two powers) is expanded
/// incrementally and the summands are multiplied out pairwise, so like
/// terms are merged at each step; distributing the whole `2n`-factor
/// product at once is exponential in `n`.
fn eval_jacobi(arena: &mut Arena, n: ExprId, a: ExprId, b: ExprId, x: ExprId) -> Option<ExprId> {
    if let (Some(deg), Some(ar), Some(br)) = (
        as_poly_degree(arena, n),
        as_ratio(arena, a),
        as_ratio(arena, b),
    ) && !(ar.is_integer() && ar.is_negative())
    {
        return jacobi_value(arena, deg, &ar, &br, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a, b]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let half = arena.rational(1, 2);
        let x_minus_1 = arena.sub(x, arena.one);
        let x_plus_1 = arena.add(&[x, arena.one]);
        let lo = arena.mul(&[half, x_minus_1]);
        let lo = expand_eval(arena, lo);
        let hi = arena.mul(&[half, x_plus_1]);
        let hi = expand_eval(arena, hi);
        let lo_pows = expanded_powers(arena, lo, deg);
        let hi_pows = expanded_powers(arena, hi, deg);
        // pa[s] = ∏_{j=s+1}^{n} (a+j)  (pa[n] = 1, pa[s] = pa[s+1]·(a+s+1))
        let mut pa: Vec<ExprId> = vec![arena.one; deg + 1];
        for s in (0..deg).rev() {
            let j = arena.int(s as i64 + 1);
            let factor = arena.add(&[a, j]);
            pa[s] = expand_mul2(arena, pa[s + 1], factor);
        }
        // pb[s] = ∏_{j=n−s+1}^{n} (b+j)  (pb[0] = 1, pb[s] = pb[s−1]·(b+n−s+1))
        let mut pb: Vec<ExprId> = vec![arena.one; deg + 1];
        for s in 1..=deg {
            let j = arena.int((deg - s) as i64 + 1);
            let factor = arena.add(&[b, j]);
            pb[s] = expand_mul2(arena, pb[s - 1], factor);
        }
        let mut terms: Vec<ExprId> = Vec::with_capacity(deg + 1);
        for s in 0..=deg {
            // C(n+a, n−s) C(n+b, s) = pa[s] pb[s] / ((n−s)! s!)
            let denom = factorial_ratio(deg - s) * factorial_ratio(s);
            let c = arena.intern_num(denom.recip());
            let c = arena.intern(ExprNode::Num(c));
            let ab = arena.mul(&[c, pa[s]]);
            let ab = expand_mul2(arena, ab, pb[s]);
            // ((x−1)/2)^s ((x+1)/2)^{n−s}: `n+1` terms once merged.
            let xp = expand_mul2(arena, lo_pows[s], hi_pows[deg - s]);
            terms.push(expand_mul2(arena, ab, xp));
        }
        // Every summand is already distributed; `add` merges like terms.
        let sum = arena.add(&terms);
        return Some(eval(arena, sum));
    }
    if a == arena.zero && b == arena.zero {
        return Some(arena.legendre(n, x));
    }
    None
}

/// Associated Legendre `P_n^m(x)` (Condon–Shortley phase) for integer
/// `n ≥ 0`, integer `m`: zero for `|m| > n`, `P_n^0 = P_n`,
/// `P_n^{−m} = (−1)^m (n−m)!/(n+m)! P_n^m`, and for `m > 0`
/// `P_m^m = (−1)^m (2m−1)!! (1−x²)^{m/2}`, `P_{m+1}^m = (2m+1) x P_m^m`,
/// `(k−m+1) P_{k+1}^m = (2k+1) x P_k^m − (k+m) P_{k−1}^m`.
fn eval_assoc_legendre(arena: &mut Arena, n: ExprId, m: ExprId, x: ExprId) -> Option<ExprId> {
    let deg = match as_poly_degree(arena, n) {
        Some(d) => d,
        None => {
            return (m == arena.zero).then(|| arena.legendre(n, x));
        }
    };
    let order = as_int_in(arena, m, -(deg as i64), deg as i64);
    let order = match order {
        Some(o) => o,
        None => {
            // Integer |m| > n ↦ 0; non-integer m stays symbolic.
            let r = as_ratio(arena, m)?;
            return r.is_integer().then_some(arena.zero);
        }
    };
    if order == 0 {
        return ortho_value(arena, &OrthoFamily::Legendre, deg, x);
    }
    let mu = order.unsigned_abs() as usize;
    // (1 − x²)^{μ/2}
    let two = arena.int(2);
    let x2 = arena.pow(x, two);
    let one_minus_x2 = arena.sub(arena.one, x2);
    let half_mu = arena.rational(mu as i64, 2);
    let root_pow = arena.pow(one_minus_x2, half_mu);
    // P_n^μ = (−1)^μ (1 − x²)^{μ/2} d^μP_n/dx^μ (Condon–Shortley, DLMF
    // 14.6.1 with Ferrers' sign): the polynomial part from the explicit
    // Legendre coefficients, c[i]·i!/(i−μ)! at x^{i−μ}.
    let c = ortho_coeffs(arena, &OrthoFamily::Legendre, deg)?;
    let mut r = vec![Q::zero(); deg - mu + 1];
    let mut falling = factorial_ratio(mu); // i!/(i−μ)! at i = μ
    for i in mu..=deg {
        if i > mu {
            falling = falling * q_int(i as i64) / q_int((i - mu) as i64);
        }
        let mut v = &c[i] * &falling;
        if mu % 2 == 1 {
            v = -v;
        }
        if !within_guard_bits(arena, &v) {
            return None;
        }
        r[i - mu] = v;
    }
    let poly = match arena.as_num(x).cloned() {
        Some(xr) => {
            let v = horner(arena, &r, &xr)?;
            guarded_num(arena, v)?
        }
        None => poly_from_coeffs(arena, &r, x)?,
    };
    let prod = arena.mul(&[poly, root_pow]);
    let value = expand_eval(arena, prod);
    if order < 0 {
        // P_n^{−μ} = (−1)^μ (n−μ)!/(n+μ)! P_n^μ
        let mut ratio = factorial_ratio(deg - mu) / factorial_ratio(deg + mu);
        if mu % 2 == 1 {
            ratio = -ratio;
        }
        let c = arena.intern_num(ratio);
        let c = arena.intern(ExprNode::Num(c));
        let v = arena.mul(&[c, value]);
        return Some(expand_eval(arena, v));
    }
    Some(value)
}

/// Generalised Laguerre `L_n^{(a)}(x)` for integer `n ≥ 0` via
/// `(k+1) L_{k+1} = (2k+1+a−x) L_k − (k+a) L_{k−1}`; `L_n^{(0)} = L_n`.
fn eval_assoc_laguerre(arena: &mut Arena, n: ExprId, a: ExprId, x: ExprId) -> Option<ExprId> {
    if let (Some(deg), Some(alpha)) = (as_poly_degree(arena, n), as_ratio(arena, a))
        && !(alpha.is_integer() && alpha.is_negative())
    {
        return ortho_value(arena, &OrthoFamily::AssocLaguerre(alpha), deg, x);
    }
    if let Some(deg) = degenerate_param_degree(arena, n, &[a]) {
        if deg == 0 {
            return Some(arena.one);
        }
        let one_plus_a = arena.add(&[arena.one, a]);
        let l1 = arena.sub(one_plus_a, x);
        let l1 = expand_eval(arena, l1);
        if deg == 1 {
            return Some(l1);
        }
        let mut prev = arena.one;
        let mut curr = l1;
        for k in 1..deg {
            let two_k_p1 = arena.int(2 * k as i64 + 1);
            let coeff = arena.add(&[two_k_p1, a]);
            let coeff = arena.sub(coeff, x);
            let t1 = arena.mul(&[coeff, curr]);
            let k_id = arena.int(k as i64);
            let k_plus_a = arena.add(&[k_id, a]);
            let t2 = arena.mul(&[k_plus_a, prev]);
            let numer = arena.sub(t1, t2);
            let kp1 = arena.int(k as i64 + 1);
            let next = arena.div(numer, kp1);
            let next = expand_eval(arena, next);
            prev = curr;
            curr = next;
        }
        return Some(curr);
    }
    if a == arena.zero {
        return Some(arena.laguerre(n, x));
    }
    None
}

/// Largest positive integer parameter `a`, `b` for which
/// `betainc(a, b, x1, x2)` is expanded into an explicit polynomial
/// (degree `a + b − 1`, `b` monomials in each limit).
const MAX_BETAINC_POLY_PARAM: i64 = 20;

/// Exact values of the generalised incomplete beta functions
/// `B_{(x1, x2)}(a, b) = ∫_{x1}^{x2} t^{a−1}(1−t)^{b−1} dt` and
/// `I_{(x1, x2)}(a, b) = B_{(x1, x2)}(a, b) / B(a, b)` (`regularized`):
///
/// * `x1 = x2` → `0`;
/// * `(x1, x2) = (0, 1)` → `B(a, b)` resp. `1` (the integral converges only
///   for `a, b > 0`, so this is refused for parameters known to be `≤ 0`);
/// * positive integers `a, b ≤ MAX_BETAINC_POLY_PARAM` → the integrand is
///   the polynomial `Σ_k C(b−1, k) (−1)^k t^{a−1+k}`, so the value is
///   `F(x2) − F(x1)` with `F(t) = Σ_k C(b−1, k) (−1)^k t^{a+k}/(a+k)`
///   (divided by the exact rational `B(a, b)` when regularised), expanded
///   and folded.  This is what makes the Beta-distribution CDF close for
///   integer shape parameters.
fn eval_betainc(
    arena: &mut Arena,
    regularized: bool,
    a: ExprId,
    b: ExprId,
    x1: ExprId,
    x2: ExprId,
) -> Option<ExprId> {
    if x1 == x2 {
        return Some(arena.zero);
    }
    if x1 == arena.zero && x2 == arena.one {
        if known_nonpositive(arena, a) || known_nonpositive(arena, b) {
            return None;
        }
        return Some(if regularized {
            arena.one
        } else {
            let beta = arena.beta(a, b);
            eval_beta(arena, a, b).unwrap_or(beta)
        });
    }
    let a_int = as_int_in(arena, a, 1, MAX_BETAINC_POLY_PARAM)?;
    let b_int = as_int_in(arena, b, 1, MAX_BETAINC_POLY_PARAM)?;
    // 1/B(a, b) = (a+b−1)! / ((a−1)! (b−1)!) — exact.
    let scale = if regularized {
        factorial_ratio((a_int + b_int - 1) as usize)
            / (factorial_ratio((a_int - 1) as usize) * factorial_ratio((b_int - 1) as usize))
    } else {
        Ratio::one()
    };
    // Antiderivative coefficients c_k of t^{a+k}, k = 0..b−1.
    let mut coeffs: Vec<Q> = Vec::with_capacity(b_int as usize);
    let mut binom = BigInt::one();
    for k in 0..b_int {
        if k > 0 {
            // C(b−1, k) = C(b−1, k−1) · (b−k) / k
            binom = binom * BigInt::from(b_int - k) / BigInt::from(k);
        }
        let mut c = Ratio::new(binom.clone(), BigInt::from(a_int + k)) * &scale;
        if k % 2 == 1 {
            c = -c;
        }
        coeffs.push(c);
    }
    let antiderivative = |arena: &mut Arena, t: ExprId| -> ExprId {
        if t == arena.zero {
            return arena.zero;
        }
        let mut terms: Vec<ExprId> = Vec::with_capacity(coeffs.len());
        for (k, c) in coeffs.iter().enumerate() {
            let cid = arena.intern_num(c.clone());
            let cid = arena.intern(ExprNode::Num(cid));
            let e = arena.int(a_int + k as i64);
            let p = arena.pow(t, e);
            terms.push(arena.mul(&[cid, p]));
        }
        arena.add(&terms)
    };
    let upper = antiderivative(arena, x2);
    let lower = antiderivative(arena, x1);
    let v = arena.sub(upper, lower);
    Some(expand_eval(arena, v))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

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
        assert_eq!(display(&a, result), "x - 1");
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

    #[test]
    fn eval_tan_two_thirds_pi() {
        let mut arena = Arena::new();
        let two_thirds = arena.rational(2, 3);
        let angle = arena.mul(&[two_thirds, arena.pi()]);
        let tan_expr = arena.tan(angle);
        let result = eval(&mut arena, tan_expr);
        // tan(2π/3) = -√3
        let d = display(&arena, result);
        assert!(
            d.contains("3") || d.contains("sqrt"),
            "tan(2π/3) should evaluate, got: {d}"
        );
    }

    #[test]
    fn eval_negative_cube_root_is_principal() {
        // SymPy: Integer(-8)**Rational(1, 3) = 2*(-1)**(1/3) (= 1 + sqrt(3)*I),
        // the principal root, like evalf; the real root is Ex::real_root.
        let mut arena = Arena::new();
        let neg8 = arena.int(-8);
        let third = arena.rational(1, 3);
        let expr = arena.pow(neg8, third);
        let result = eval(&mut arena, expr);
        let two = arena.int(2);
        let neg_one_root = arena.pow(arena.neg_one, third);
        let expected = arena.mul(&[two, neg_one_root]);
        assert_eq!(result, expected, "(-8)^(1/3) should be 2*(-1)^(1/3)");
    }

    // ── 0.2 special functions ─────────────────────────────────────────────

    #[test]
    fn eval_zeta_table() {
        let mut a = Arena::new();
        let cases: [(i64, &str); 8] = [
            (0, "-1/2"),
            (-1, "-1/12"),
            (-2, "0"),
            (-3, "1/120"),
            (-9, "-1/132"),
            (2, "1/6*pi^2"),
            (4, "1/90*pi^4"),
            (14, "2/18243225*pi^14"),
        ];
        for (s, expected) in cases {
            let sid = a.int(s);
            let z = a.zeta(sid);
            assert_eq!(display(&a, z), expected, "zeta({s})");
        }
        let one = a.one;
        assert_eq!(a.zeta(one), a.complex_infinity);
        let three = a.int(3);
        let z3 = a.zeta(three);
        assert!(matches!(a.node(z3), ExprNode::Zeta(_)));
        // Huge even arguments are left symbolic (no gigantic rationals).
        let big = a.int(400);
        let zb = a.zeta(big);
        assert!(matches!(a.node(zb), ExprNode::Zeta(_)));
    }

    #[test]
    fn eval_si_ci_ei_li_special_points() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        assert_eq!(a.si(a.zero), a.zero);
        let si_inf = a.si(a.infinity);
        assert_eq!(display(&a, si_inf), "1/2*pi");
        let neg_x = a.neg(x);
        let si_neg = a.si(neg_x);
        let si_x = a.si(x);
        assert_eq!(si_neg, a.neg(si_x));
        assert_eq!(a.ci(a.infinity), a.zero);
        assert_eq!(a.ei(a.neg_infinity), a.zero);
        assert_eq!(a.li(a.zero), a.zero);
        assert_eq!(a.li(a.one), a.neg_infinity);
        let ex = a.exp(x);
        let li_ex = a.li(ex);
        let ei_x = a.ei(x);
        assert_eq!(li_ex, ei_x);
        // eval() refolds after substitution
        let si_x = a.si(x);
        let sub = a.subs_structural(si_x, x, a.zero);
        assert_eq!(sub, a.zero);
    }

    #[test]
    fn eval_polygamma_and_digamma_values() {
        let mut a = Arena::new();
        let one = a.one;
        let two = a.int(2);
        let half = a.rational(1, 2);
        // ψ'(1) = ζ(2)
        let p11 = a.polygamma(one, one);
        let z2 = a.zeta(two);
        assert_eq!(p11, z2);
        // ψ''(1/2) = −14 ζ(3)
        let p2h = a.polygamma(two, half);
        assert_eq!(display(&a, p2h), "-14*zeta(3)");
        // ψ'(4) = π²/6 − 1 − 1/4 − 1/9 = π²/6 − 49/36
        let four = a.int(4);
        let p14 = a.polygamma(one, four);
        assert_eq!(display(&a, p14), "1/6*pi^2 - 49/36");
        // ψ(4) = −γ + 11/6
        let d4 = a.digamma(four);
        let d4e = eval(&mut a, d4);
        assert_eq!(display(&a, d4e), "-EulerGamma + 11/6");
        // order 0 → digamma node
        let x = sym(&mut a, "x");
        let p0 = a.polygamma(a.zero, x);
        assert!(matches!(a.node(p0), ExprNode::Digamma(_)));
        // non-integer order stays symbolic
        let ph = a.polygamma(half, x);
        assert!(matches!(a.node(ph), ExprNode::Polygamma(_, _)));
    }

    #[test]
    fn eval_kronecker_delta_values() {
        let mut a = Arena::new();
        let i = sym(&mut a, "i");
        let j = sym(&mut a, "j");
        assert_eq!(a.kronecker_delta(i, i), a.one);
        let two = a.int(2);
        let three = a.int(3);
        assert_eq!(a.kronecker_delta(two, three), a.zero);
        assert_eq!(a.kronecker_delta(two, two), a.one);
        let ip1 = a.add(&[i, a.one]);
        assert_eq!(a.kronecker_delta(ip1, i), a.zero);
        let d1 = a.kronecker_delta(i, j);
        let d2 = a.kronecker_delta(j, i);
        assert_eq!(d1, d2, "canonical argument order");
        assert!(matches!(a.node(d1), ExprNode::KroneckerDelta(_, _)));
    }

    #[test]
    fn eval_refolds_complex_nodes() {
        let mut a = Arena::new();
        let z = sym(&mut a, "z");
        let re_z = a.re(z);
        assert!(matches!(a.node(re_z), ExprNode::Re(_)));
        let three = a.int(3);
        let four = a.int(4);
        let four_i = a.mul(&[a.i_unit, four]);
        let w = a.add(&[three, four_i]);
        let sub = a.subs_structural(re_z, z, w);
        let folded = eval(&mut a, sub);
        assert_eq!(folded, three);
    }

    #[test]
    fn explicit_orthogonal_coefficients_match_the_recurrences() {
        // The coefficient formulas (DLMF §18.5) against the three-term
        // recurrences they replace, symbolically and at rational points.
        type Old = fn(&mut Arena, usize, ExprId) -> ExprId;
        let families: [(OrthoFamily, Old); 5] = [
            (OrthoFamily::Legendre, eval_legendre),
            (OrthoFamily::ChebyshevT, eval_chebyshev_t),
            (OrthoFamily::ChebyshevU, eval_chebyshev_u),
            (OrthoFamily::Hermite, eval_hermite),
            (OrthoFamily::Laguerre, eval_laguerre),
        ];
        let mut a = Arena::new();
        let x = a.symbol("x");
        let pts = [a.rational(1, 3), a.rational(-7, 5), a.int(2)];
        for (family, old) in families {
            for n in 0..=14 {
                let new = ortho_value(&mut a, &family, n, x).unwrap();
                let reference = old(&mut a, n, x);
                assert_eq!(new, reference, "{family:?} n = {n}");
                for &p in &pts {
                    let new = ortho_value(&mut a, &family, n, p).unwrap();
                    let reference = old(&mut a, n, p);
                    let reference = eval(&mut a, reference);
                    assert_eq!(new, reference, "{family:?} n = {n} at {}", display(&a, p));
                }
            }
        }
    }

    #[test]
    fn tangent_numbers_give_the_bernoulli_numbers() {
        // Brent–Harvey tangent numbers against the cached rational
        // recurrence of `base::bernoulli`.
        let a = Arena::new();
        for n in 0..=120u64 {
            let fast = exact_bernoulli(&a, n).unwrap();
            let slow = crate::base::bernoulli::bernoulli(n as usize);
            assert_eq!(fast, slow, "B_{n}");
        }
    }
}
