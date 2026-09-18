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

use crate::base::arena::{
    Arena, FN_BELL, FN_BERNOULLI, FN_BESSELI, FN_BESSELJ, FN_BESSELK, FN_BESSELY, FN_CATALAN,
    FN_CHEBYSHEV_T, FN_CHEBYSHEV_U, FN_EULER_NUMBER, FN_FACTORIAL2, FN_FALLING_FACTORIAL,
    FN_FIBONACCI, FN_HARMONIC, FN_HERMITE, FN_LAGUERRE, FN_LEGENDRE, FN_LUCAS, FN_PARTITION_COUNT,
    FN_RISING_FACTORIAL, FN_STIRLING1, FN_STIRLING2, FN_SUBFACTORIAL,
};
use crate::base::node::{ExprId, ExprNode};
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
                // Pow(Exp(f), g) → Exp(f·g): valid because exp(f) > 0 for all
                // real f, so (exp(f))^g = exp(f·g) without branch-cut issues.
                //
                // Critical for the Gruntz algorithm: without this, exp(x)^(1/x)
                // stays as Pow(Exp(x), 1/x) and mrv creates dangling dummy
                // variables when trying to rewrite via exp((1/x)·ln(exp(x))).
                // With this rule, eval simplifies it to Exp(x·(1/x)) = Exp(1).
                //
                // Placed in eval (not canon_pow) so the solver's intermediate
                // Pow(Exp(x), k) nodes survive until substitution completes.
                else if let ExprNode::Exp(inner) = arena.node(nb).clone() {
                    let product = arena.mul(&[inner, ne]);
                    arena.exp(product)
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

            // ── Apply (named combinatorial functions) ──────────────
            ExprNode::Apply(name_sid, ref args) => {
                let new_args: smallvec::SmallVec<[ExprId; 2]> = args
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                let name = arena.symbols.name(name_sid).to_owned();
                match name.as_str() {
                    FN_FACTORIAL2 if new_args.len() == 1 => {
                        if let Some(result) = eval_factorial2(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.factorial2(new_args[0])
                        }
                    }
                    FN_SUBFACTORIAL if new_args.len() == 1 => {
                        if let Some(result) = eval_subfactorial(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.subfactorial(new_args[0])
                        }
                    }
                    FN_RISING_FACTORIAL if new_args.len() == 2 => {
                        if let Some(result) = eval_rising_factorial(arena, new_args[0], new_args[1])
                        {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.rising_factorial(new_args[0], new_args[1])
                        }
                    }
                    FN_FALLING_FACTORIAL if new_args.len() == 2 => {
                        if let Some(result) =
                            eval_falling_factorial(arena, new_args[0], new_args[1])
                        {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.falling_factorial(new_args[0], new_args[1])
                        }
                    }
                    FN_FIBONACCI if new_args.len() == 1 => {
                        if let Some(result) = eval_fibonacci(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.fibonacci(new_args[0])
                        }
                    }
                    FN_LUCAS if new_args.len() == 1 => {
                        if let Some(result) = eval_lucas(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.lucas(new_args[0])
                        }
                    }
                    FN_BERNOULLI if new_args.len() == 1 => {
                        if let Some(result) = eval_bernoulli(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.bernoulli_number(new_args[0])
                        }
                    }
                    FN_HARMONIC if new_args.len() == 1 => {
                        if let Some(result) = eval_harmonic(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.harmonic(new_args[0])
                        }
                    }
                    FN_CATALAN if new_args.len() == 1 => {
                        if let Some(result) = eval_catalan(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.catalan_number(new_args[0])
                        }
                    }
                    FN_BELL if new_args.len() == 1 => {
                        if let Some(result) = eval_bell(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.bell(new_args[0])
                        }
                    }
                    FN_EULER_NUMBER if new_args.len() == 1 => {
                        if let Some(result) = eval_euler_number(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.euler_number(new_args[0])
                        }
                    }

                    // ── Combinatorial (Phase 1) ────────────────────
                    FN_STIRLING2 if new_args.len() == 2 => {
                        if let Some(result) = eval_stirling2(arena, new_args[0], new_args[1]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.stirling2(new_args[0], new_args[1])
                        }
                    }
                    FN_STIRLING1 if new_args.len() == 2 => {
                        if let Some(result) = eval_stirling1(arena, new_args[0], new_args[1]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.stirling1(new_args[0], new_args[1])
                        }
                    }
                    FN_PARTITION_COUNT if new_args.len() == 1 => {
                        if let Some(result) = eval_partition_count(arena, new_args[0]) {
                            result
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.partition_count(new_args[0])
                        }
                    }

                    // ── Bessel functions ────────────────────────────
                    FN_BESSELJ if new_args.len() == 2 => {
                        if let (Some(order_num), Some(arg_num)) = (
                            arena.as_num(new_args[0]).cloned(),
                            arena.as_num(new_args[1]).cloned(),
                        ) {
                            if arg_num.is_zero() {
                                if order_num.is_zero() {
                                    arena.one // J_0(0) = 1
                                } else if order_num.is_positive() && order_num.is_integer() {
                                    arena.zero // J_n(0) = 0 for integer n > 0
                                } else {
                                    arena.besselj(new_args[0], new_args[1])
                                }
                            } else {
                                arena.besselj(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.besselj(new_args[0], new_args[1])
                        }
                    }
                    FN_BESSELY if new_args.len() == 2 => {
                        // Y_n(0) diverges, leave symbolic
                        if new_args[..] == args[..] {
                            id
                        } else {
                            arena.bessely(new_args[0], new_args[1])
                        }
                    }
                    FN_BESSELI if new_args.len() == 2 => {
                        if let (Some(order_num), Some(arg_num)) = (
                            arena.as_num(new_args[0]).cloned(),
                            arena.as_num(new_args[1]).cloned(),
                        ) {
                            if arg_num.is_zero() {
                                if order_num.is_zero() {
                                    arena.one // I_0(0) = 1
                                } else if order_num.is_positive() && order_num.is_integer() {
                                    arena.zero // I_n(0) = 0 for integer n > 0
                                } else {
                                    arena.besseli(new_args[0], new_args[1])
                                }
                            } else {
                                arena.besseli(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.besseli(new_args[0], new_args[1])
                        }
                    }
                    FN_BESSELK if new_args.len() == 2 => {
                        // K_n(0) diverges, leave symbolic
                        if new_args[..] == args[..] {
                            id
                        } else {
                            arena.besselk(new_args[0], new_args[1])
                        }
                    }

                    // ── Orthogonal polynomials ─────────────────────
                    FN_LEGENDRE if new_args.len() == 2 => {
                        if let Some(n_num) = arena.as_num(new_args[0]).cloned() {
                            if n_num.is_integer() && !n_num.is_negative() {
                                if let Some(n_int) = n_num.to_integer().to_u64() {
                                    if n_int <= arena.config.max_pow_exponent as u64 {
                                        let x = new_args[1];
                                        eval_legendre(arena, n_int as usize, x)
                                    } else {
                                        arena.legendre(new_args[0], new_args[1])
                                    }
                                } else {
                                    arena.legendre(new_args[0], new_args[1])
                                }
                            } else {
                                arena.legendre(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.legendre(new_args[0], new_args[1])
                        }
                    }
                    FN_CHEBYSHEV_T if new_args.len() == 2 => {
                        if let Some(n_num) = arena.as_num(new_args[0]).cloned() {
                            if n_num.is_integer() && !n_num.is_negative() {
                                if let Some(n_int) = n_num.to_integer().to_u64() {
                                    if n_int <= arena.config.max_pow_exponent as u64 {
                                        let x = new_args[1];
                                        eval_chebyshev_t(arena, n_int as usize, x)
                                    } else {
                                        arena.chebyshev_t(new_args[0], new_args[1])
                                    }
                                } else {
                                    arena.chebyshev_t(new_args[0], new_args[1])
                                }
                            } else {
                                arena.chebyshev_t(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.chebyshev_t(new_args[0], new_args[1])
                        }
                    }
                    FN_CHEBYSHEV_U if new_args.len() == 2 => {
                        if let Some(n_num) = arena.as_num(new_args[0]).cloned() {
                            if n_num.is_integer() && !n_num.is_negative() {
                                if let Some(n_int) = n_num.to_integer().to_u64() {
                                    if n_int <= arena.config.max_pow_exponent as u64 {
                                        let x = new_args[1];
                                        eval_chebyshev_u(arena, n_int as usize, x)
                                    } else {
                                        arena.chebyshev_u(new_args[0], new_args[1])
                                    }
                                } else {
                                    arena.chebyshev_u(new_args[0], new_args[1])
                                }
                            } else {
                                arena.chebyshev_u(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.chebyshev_u(new_args[0], new_args[1])
                        }
                    }
                    FN_HERMITE if new_args.len() == 2 => {
                        if let Some(n_num) = arena.as_num(new_args[0]).cloned() {
                            if n_num.is_integer() && !n_num.is_negative() {
                                if let Some(n_int) = n_num.to_integer().to_u64() {
                                    if n_int <= arena.config.max_pow_exponent as u64 {
                                        let x = new_args[1];
                                        eval_hermite(arena, n_int as usize, x)
                                    } else {
                                        arena.hermite(new_args[0], new_args[1])
                                    }
                                } else {
                                    arena.hermite(new_args[0], new_args[1])
                                }
                            } else {
                                arena.hermite(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.hermite(new_args[0], new_args[1])
                        }
                    }
                    FN_LAGUERRE if new_args.len() == 2 => {
                        if let Some(n_num) = arena.as_num(new_args[0]).cloned() {
                            if n_num.is_integer() && !n_num.is_negative() {
                                if let Some(n_int) = n_num.to_integer().to_u64() {
                                    if n_int <= arena.config.max_pow_exponent as u64 {
                                        let x = new_args[1];
                                        eval_laguerre(arena, n_int as usize, x)
                                    } else {
                                        arena.laguerre(new_args[0], new_args[1])
                                    }
                                } else {
                                    arena.laguerre(new_args[0], new_args[1])
                                }
                            } else {
                                arena.laguerre(new_args[0], new_args[1])
                            }
                        } else if new_args[..] == args[..] {
                            id
                        } else {
                            arena.laguerre(new_args[0], new_args[1])
                        }
                    }

                    _ => {
                        if new_args[..] == args[..] {
                            id
                        } else {
                            let sv: smallvec::SmallVec<[ExprId; 2]> = new_args;
                            arena.intern(ExprNode::Apply(name_sid, sv))
                        }
                    }
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
                let all_numeric: Option<Vec<(Ratio<BigInt>, ExprId)>> = new
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
                let all_numeric: Option<Vec<(Ratio<BigInt>, ExprId)>> = new
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
                    crate::transforms::sum_eval::eval_sum_symbolic(arena, nbody, nvar, nlo, nhi)
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
                    crate::transforms::sum_eval::eval_product_symbolic(arena, nbody, nvar, nlo, nhi)
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

/// Gamma(n) for positive integer n → (n-1)!
/// Gamma(1/2) → √π
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

        // Compute (2k-1)!! = product of odd numbers 1, 3, 5, …, 2k-1.
        let mut double_fact = BigInt::from(1);
        for i in 0..k {
            double_fact *= BigInt::from(2 * i + 1);
        }

        // coefficient = (2k-1)!! / 2^k
        let two_pow_k = BigInt::from(1) << (k as usize);
        let coeff = Ratio::new(double_fact, two_pow_k);

        let pi = arena.pi;
        let sqrt_pi = arena.sqrt(pi);

        if coeff.is_one() {
            return Some(sqrt_pi);
        }

        let coeff_id = arena.intern_num(coeff);
        let coeff_node = arena.intern(ExprNode::Num(coeff_id));
        let result = arena.mul(&[coeff_node, sqrt_pi]);
        return Some(result);
    }

    // Positive integer: Gamma(n) = (n-1)!
    if r.is_integer() && r.is_positive() {
        let n: u64 = r.to_integer().try_into().ok()?;
        let mut result = Ratio::<BigInt>::one();
        for i in 2..n {
            result *= Ratio::from_integer(BigInt::from(i));
        }
        let nid = arena.intern_num(result);
        return Some(arena.intern(ExprNode::Num(nid)));
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
            let mut product = Ratio::<BigInt>::one();
            for i in 0..n {
                product *= &frac + Ratio::from_integer(BigInt::from(i));
            }
            let frac_id = {
                let nid = arena.intern_num(frac);
                arena.intern(ExprNode::Num(nid))
            };
            // Try to evaluate the inner Gamma (e.g. Gamma(1/2) → √π)
            let gamma_frac = eval_gamma(arena, frac_id).unwrap_or_else(|| arena.gamma(frac_id));
            if product.is_one() {
                return Some(gamma_frac);
            }
            let prod_nid = arena.intern_num(product);
            let prod_node = arena.intern(ExprNode::Num(prod_nid));
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
        // Denominator product: x·(x+1)·…·(x+m-1)
        let mut denom_product = Ratio::<BigInt>::one();
        for i in 0..m {
            denom_product *= &r + Ratio::from_integer(BigInt::from(i));
        }
        if denom_product.is_zero() {
            return None; // pole — should not happen since r is not an integer
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

/// LogGamma(n) for positive integer n → ln((n-1)!)
fn eval_log_gamma(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?.clone();
    if !r.is_integer() || !r.is_positive() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // (n-1)!
    let mut fact = Ratio::<BigInt>::one();
    for i in 2..n {
        fact *= Ratio::from_integer(BigInt::from(i));
    }
    // ln(1) = 0 — handle n=1 and n=2 where (n-1)! = 1
    if fact == Ratio::<BigInt>::one() {
        return Some(arena.zero);
    }
    let nid = arena.intern_num(fact);
    let fact_id = arena.intern(ExprNode::Num(nid));
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

/// Beta(a,b) for positive integers → (a-1)!(b-1)!/(a+b-1)!
fn eval_beta(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<ExprId> {
    let ra = arena.as_num(a)?.clone();
    let rb = arena.as_num(b)?.clone();
    if !ra.is_integer() || !ra.is_positive() || !rb.is_integer() || !rb.is_positive() {
        return None;
    }
    let a_u64: u64 = ra.to_integer().try_into().ok()?;
    let b_u64: u64 = rb.to_integer().try_into().ok()?;

    // B(a,b) = (a-1)!(b-1)! / (a+b-1)!
    let mut numer = Ratio::<BigInt>::one();
    for i in 2..a_u64 {
        numer *= Ratio::from_integer(BigInt::from(i));
    }
    let mut b_fact = Ratio::<BigInt>::one();
    for i in 2..b_u64 {
        b_fact *= Ratio::from_integer(BigInt::from(i));
    }
    numer *= b_fact;

    let ab = a_u64.checked_add(b_u64)?;
    let mut denom = Ratio::<BigInt>::one();
    for i in 2..ab {
        denom *= Ratio::from_integer(BigInt::from(i));
    }

    let result = numer / denom;
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
        let b = crate::base::bernoulli::bernoulli(m + 1);
        let val = -b / Ratio::from_integer(BigInt::from(m as i64 + 1));
        let nid = arena.intern_num(val);
        return Some(arena.intern(ExprNode::Num(nid)));
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

/// `n!` as a rational.
fn factorial_ratio(n: usize) -> Ratio<BigInt> {
    let mut f = BigInt::from(1);
    for i in 2..=n {
        f *= BigInt::from(i as u64);
    }
    Ratio::from_integer(f)
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
    let sign = if n_usize.is_multiple_of(2) { -1 } else { 1 }; // (−1)^{n+1}
    let n_fact = factorial_ratio(n_usize);
    let np1 = arena.int(n_usize as i64 + 1);
    let zeta_np1 = arena.zeta(np1);

    if xr.is_integer() {
        let m: i64 = xr.to_integer().try_into().ok()?;
        if m <= 0 {
            return Some(arena.complex_infinity);
        }
        if m > MAX_POLYGAMMA_SHIFT {
            return None;
        }
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
    if *xr.denom() == BigInt::from(2) {
        let m: i64 = xr.floor().to_integer().try_into().ok()?;
        if !(0..=MAX_POLYGAMMA_SHIFT).contains(&m) {
            return None;
        }
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

/// Double factorial: n!! = n * (n-2) * (n-4) * ... * 1 (or 2).
/// 0!! = 1, 1!! = 1, (-1)!! = 1.
fn eval_factorial2(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n < -1 {
        return None;
    }
    let mut result = BigInt::from(1);
    let mut k = n;
    while k > 1 {
        result *= BigInt::from(k);
        k -= 2;
    }
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Subfactorial (derangement count): !n.
/// Uses recurrence: !0 = 1, !1 = 0, !n = (n-1)(!(n-1) + !(n-2)).
fn eval_subfactorial(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
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
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Rising factorial (Pochhammer): (x)_n = x * (x+1) * ... * (x+n-1).
/// Works for rational x; n must be a non-negative integer.
fn eval_rising_factorial(arena: &mut Arena, x_id: ExprId, n_id: ExprId) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    let mut result = Ratio::<BigInt>::one();
    for i in 0..n {
        result *= &xr + Ratio::from_integer(BigInt::from(i));
    }
    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Falling factorial: x^(n) = x * (x-1) * ... * (x-n+1).
/// Works for rational x; n must be a non-negative integer.
fn eval_falling_factorial(arena: &mut Arena, x_id: ExprId, n_id: ExprId) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() || nr.is_negative() {
        return None;
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    let mut result = Ratio::<BigInt>::one();
    for i in 0..n {
        result *= &xr - Ratio::from_integer(BigInt::from(i));
    }
    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Fibonacci number F(n) using iterative computation.
/// F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2).
fn eval_fibonacci(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
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
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Lucas number L(n) using iterative computation.
/// L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2).
fn eval_lucas(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
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
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Bernoulli number B(n).
/// B(0) = 1, and for n >= 1:
///   B(n) = -1/(n+1) * sum_{k=0}^{n-1} C(n+1, k) * B(k)
fn eval_bernoulli(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // Build table of B(0)..B(n)
    let mut b_vals: Vec<Ratio<BigInt>> = Vec::with_capacity((n + 1) as usize);
    b_vals.push(Ratio::one()); // B(0) = 1
    for m in 1..=n {
        // B(m) = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B(k)
        let mut sum = Ratio::<BigInt>::zero();
        let mut binom = BigInt::from(1); // C(m+1, 0) = 1
        for k in 0..m {
            sum += Ratio::from_integer(binom.clone()) * &b_vals[k as usize];
            // C(m+1, k+1) = C(m+1, k) * (m+1-k) / (k+1)
            binom *= BigInt::from(m + 1 - k);
            binom /= BigInt::from(k + 1);
        }
        let result = -sum / Ratio::from_integer(BigInt::from(m + 1));
        b_vals.push(result);
    }
    let ratio = b_vals.into_iter().last().unwrap();
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Harmonic number H(n) = 1 + 1/2 + 1/3 + ... + 1/n.
/// H(0) = 0.
fn eval_harmonic(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    let mut result = Ratio::<BigInt>::zero();
    for k in 1..=n {
        result += Ratio::new(BigInt::from(1), BigInt::from(k));
    }
    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Catalan number C(n) = (2n)! / ((n+1)! * n!).
fn eval_catalan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // C(n) = C(2n, n) / (n+1) — compute using incremental binomial
    let mut binom = BigInt::from(1); // C(2n, n) built incrementally
    for i in 0..n {
        binom *= BigInt::from(2 * n - i);
        binom /= BigInt::from(i + 1);
    }
    let result = Ratio::new(binom, BigInt::from(n + 1));
    let nid = arena.intern_num(result);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Stirling number of the second kind S(n, k).
/// Delegates to `combinatorics::stirling2` which returns `Option<BigInt>`.
/// `None` propagates → the expression node stays unevaluated.
fn eval_stirling2(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    let n_r = arena.as_num(n_id)?;
    let k_r = arena.as_num(k_id)?;
    if !n_r.is_integer() || !k_r.is_integer() || n_r.is_negative() || k_r.is_negative() {
        return None;
    }
    let result = crate::domains::combinatorics::stirling2(n_r.to_integer(), k_r.to_integer())?;
    let nid = arena.intern_num(Ratio::from_integer(result));
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Signed Stirling number of the first kind s(n, k).
/// Delegates to `combinatorics::stirling1` which returns `Option<BigInt>`.
fn eval_stirling1(arena: &mut Arena, n_id: ExprId, k_id: ExprId) -> Option<ExprId> {
    let n_r = arena.as_num(n_id)?;
    let k_r = arena.as_num(k_id)?;
    if !n_r.is_integer() || !k_r.is_integer() || n_r.is_negative() || k_r.is_negative() {
        return None;
    }
    let result = crate::domains::combinatorics::stirling1(n_r.to_integer(), k_r.to_integer())?;
    let nid = arena.intern_num(Ratio::from_integer(result));
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Number of integer partitions p(n).
/// Delegates to `combinatorics::partition_count` which returns `Option<BigInt>`.
fn eval_partition_count(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let result = crate::domains::combinatorics::partition_count(r.to_integer())?;
    let nid = arena.intern_num(Ratio::from_integer(result));
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Bell number B(n) using the Bell triangle.
/// B(0) = 1, B(1) = 1, B(2) = 2, B(3) = 5, B(4) = 15, B(5) = 52.
fn eval_bell(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    if n == 0 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(1)));
        return Some(arena.intern(ExprNode::Num(nid)));
    }
    // Bell triangle: row[0] = B(n-1), then row[j] = row[j-1] + prev_row[j-1]
    let mut row = vec![BigInt::from(1)]; // B(0) = 1, start of row 1
    for _ in 1..n {
        let mut new_row = Vec::with_capacity(row.len() + 1);
        new_row.push(row.last().unwrap().clone()); // first element = last of prev row
        for j in 1..=row.len() {
            let val = &new_row[j - 1] + &row[j - 1];
            new_row.push(val);
        }
        row = new_row;
    }
    // B(n) = last element of the nth row = row.last()
    let result = row.last().unwrap().clone();
    let ratio = Ratio::from_integer(result);
    let nid = arena.intern_num(ratio);
    Some(arena.intern(ExprNode::Num(nid)))
}

/// Euler number E(n). Odd indices give 0.
/// E(0) = 1, E(2) = -1, E(4) = 5, E(6) = -61, ...
/// Recurrence for even n >= 2: E(n) = -sum_{k=0,2,4,...,n-2} C(n, k) * E(k).
fn eval_euler_number(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let r = arena.as_num(inner)?;
    if !r.is_integer() || r.is_negative() {
        return None;
    }
    let n: u64 = r.to_integer().try_into().ok()?;
    // Odd Euler numbers are 0
    if n % 2 == 1 {
        let nid = arena.intern_num(Ratio::from_integer(BigInt::from(0)));
        return Some(arena.intern(ExprNode::Num(nid)));
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
    let result = e_vals.last().unwrap().clone();
    let ratio = Ratio::from_integer(result);
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
/// Returns (k, m) such that n = k^q * m and m has no q-th power factors
/// (bounded factorisation — see [`crate::base::canon::split_perfect_power`]).
fn extract_perfect_power(n: &BigInt, q: usize) -> Option<(BigInt, BigInt)> {
    let q = u32::try_from(q).ok()?;
    Some(crate::base::canon::split_perfect_power(n, q))
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

    // Odd roots of negative integers: (-n)^(1/k) = -(n^(1/k)) when k is odd
    if base_r.is_negative()
        && base_r.is_integer()
        && let Some(exp_r) = arena.as_num(exp)
    {
        let exp_r = exp_r.clone();
        if *exp_r.numer() == BigInt::from(1) {
            let k = exp_r.denom().clone();
            // Check k is odd
            if &k % BigInt::from(2) != BigInt::from(0) {
                let abs_base = -base_r.clone();
                let abs_base_id = {
                    let nid = arena.intern_num(Ratio::from_integer(abs_base.to_integer()));
                    arena.intern(ExprNode::Num(nid))
                };
                let root = arena.pow(abs_base_id, exp);
                let root_eval = eval(arena, root);
                return Some(arena.neg(root_eval));
            }
        }
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
fn mod_positive(a: &Ratio<BigInt>, m: &Ratio<BigInt>) -> Ratio<BigInt> {
    let mut result = a % m;
    if result.is_negative() {
        result += m;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Orthogonal polynomial evaluation via recurrence relations
// ═══════════════════════════════════════════════════════════════════════════

/// Legendre polynomial P_n(x) via Bonnet's recurrence.
/// P_0(x) = 1, P_1(x) = x, (k+1)·P_{k+1} = (2k+1)·x·P_k − k·P_{k-1}
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
    fn eval_negative_cube_root() {
        let mut arena = Arena::new();
        let neg8 = arena.int(-8);
        let third = arena.rational(1, 3);
        let expr = arena.pow(neg8, third);
        let result = eval(&mut arena, expr);
        let expected = arena.int(-2);
        assert_eq!(result, expected, "(-8)^(1/3) should be -2");
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
}
