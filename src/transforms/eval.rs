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
//! - `cos(π/5)` → `1/4 + √5/4`, `sin(π/12)` → `−√2/4 + √6/4`, `tan(π/8)` →
//!   `−1 + √2`, `cot(π/8)` → `1 + √2`: `sin`, `cos` (so `sec`, `csc`) at
//!   `kπ/n` for `n ∈ {5, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}`, `tan`
//!   and `cot` also at `n ∈ {16, 48, 80, 240}`, in square roots — the
//!   denominators and normal forms of SymPy's `cos.eval`/`tan.eval`/
//!   `cot.eval` (see `cos_pi_radical`); `cos(π/7)` stays.  The hyperbolic
//!   functions at `i·kπ/n` likewise (`cosh(iπ/5) = cos(π/5)`).
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
//! - `acos(√2/2)` → `π/4`, `asin(√(2+√3)/2)` → `5π/12`, `atan(2 − √3)` →
//!   `π/12`, `atan2(√3/2, 1/2)` → `π/3`: the inverse functions of
//!   `±cos(kπ/n)` and `±tan(kπ/n)`, `n ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12}`, in
//!   any radical form (see `special_angle_of`); `asin(sin(7π/102))` →
//!   `7π/102`, `acos(cos(14π/9))` → `4π/9` at the angles that stay
//!   unevaluated (see `inverse_of_trig_at_pi_multiple`)
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
use crate::base::value_class::{self, ValueClass};
use crate::base::walk;

mod atan_sum;
mod cyclotomic;

mod digit_guard;
mod integer_sequences;
mod lib_values;
mod orthopoly;
mod quadratic_tower;
mod special02;
mod trig_radicals;
pub(crate) use digit_guard::*;
pub(crate) use integer_sequences::*;
pub(crate) use lib_values::*;
use orthopoly::*;
use quadratic_tower::*;
pub(crate) use special02::*;
pub(crate) use trig_radicals::*;

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
            ExprNode::Atan(original) => {
                let inner = cache.get(&original).copied().unwrap_or(original);
                reciprocal_at_zero(arena, "acot", original, inner, &cache)
                    .or_else(|| eval_atan(arena, inner))
                    .unwrap_or_else(|| arena.intern(ExprNode::Atan(inner)))
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
            ExprNode::Acosh(original) => {
                let inner = cache.get(&original).copied().unwrap_or(original);
                reciprocal_at_zero(arena, "asech", original, inner, &cache)
                    .or_else(|| eval_acosh(arena, inner))
                    .unwrap_or_else(|| arena.intern(ExprNode::Acosh(inner)))
            }
            ExprNode::Atanh(original) => {
                let inner = cache.get(&original).copied().unwrap_or(original);
                reciprocal_at_zero(arena, "acoth", original, inner, &cache)
                    .or_else(|| eval_atanh(arena, inner))
                    .unwrap_or_else(|| arena.intern(ExprNode::Atanh(inner)))
            }
            ExprNode::Sign(inner) => {
                let ni = cache.get(&inner).copied().unwrap_or(inner);
                if let Some(result) = eval_sign(arena, ni, inner) {
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
                } else if let Some(s) = real_constant_sign(arena, evaled) {
                    // SymPy: `Heaviside(1 - sqrt(2))` → 0.
                    if s == std::cmp::Ordering::Greater {
                        arena.one
                    } else {
                        arena.zero
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
                let v = arena.im(ni);
                zero_real_lambert_parts(arena, v)
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
                if let Some(folded) = eval_cot_quotients(arena, children, &cache) {
                    folded
                } else {
                    let new: smallvec::SmallVec<[ExprId; 6]> = children
                        .iter()
                        .map(|&c| cache.get(&c).copied().unwrap_or(c))
                        .collect();
                    if let Some(folded) = eval_log_quotient(arena, &new) {
                        folded
                    } else if new == *children {
                        id
                    } else {
                        arena.mul(&new)
                    }
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
            ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
                let na = cache.get(&a).copied().unwrap_or(a);
                let nb = cache.get(&b).copied().unwrap_or(b);
                let kind = RelationKind::of(arena.node(id));
                if let Some(holds) = relation_truth(arena, kind, na, nb, (a, b)) {
                    if holds {
                        arena.bool_true
                    } else {
                        arena.bool_false
                    }
                } else if na == a && nb == b {
                    id
                } else {
                    match kind {
                        RelationKind::Gt => arena.gt(na, nb),
                        RelationKind::Ge => arena.ge(na, nb),
                        RelationKind::Eq => arena.eq_(na, nb),
                        RelationKind::Ne => arena.ne_(na, nb),
                    }
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
            // earlier branch whose condition is still open.  A condition
            // without a truth value at this point (an order relation with
            // a `nan` operand: [`condition_is_undefined`]) makes the value
            // `nan` from there on.
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
                    if condition_is_undefined(arena, nc) {
                        kept.push((arena.nan, t));
                        break;
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
                    arena.make(ExprNode::Piecewise(kept))
                }
            }

            // ── Library special functions ───────────────────────────
            ExprNode::Lib(f, ref args) => {
                let new_args: smallvec::SmallVec<[ExprId; 2]> = args
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                // It may fold to an exact value; otherwise it is rebuilt on
                // its evaluated arguments.
                let folded = if f.arity().accepts(new_args.len()) {
                    eval_lib_fn(arena, f, &new_args)
                } else {
                    None
                };
                match folded {
                    Some(result) => result,
                    None if new_args[..] == args[..] => id,
                    None => arena.lib_apply(f, &new_args),
                }
            }

            // ── Apply (user functions): rebuilt on the evaluated arguments ──
            ExprNode::Apply(name_sid, ref args) => {
                let new_args: smallvec::SmallVec<[ExprId; 2]> = args
                    .iter()
                    .map(|&c| cache.get(&c).copied().unwrap_or(c))
                    .collect();
                if new_args[..] == args[..] {
                    id
                } else {
                    arena.intern(ExprNode::Apply(name_sid, new_args))
                }
            }
            // ── Internal functions of the limit engine: likewise ──
            ExprNode::Internal(..) => walk::rebuild_with_cache(arena, id, &cache),

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
                } else if let Some(n) = integer_part_of_constant(arena, ni, false) {
                    n
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
                } else if let Some(n) = integer_part_of_constant(arena, ni, true) {
                    n
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
                } else if let Some(m) = extremum_of_real_constants(arena, &new, true) {
                    m
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
                } else if let Some(m) = extremum_of_real_constants(arena, &new, false) {
                    m
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

                // The empty range `hi = lo − 1` before anything else (a
                // range with `hi < lo − 1` is minus the sum over
                // `[hi + 1, lo − 1]`, Karr's convention: the engine's).
                if let (Some(lo_val), Some(hi_val)) = (lo_int, hi_int)
                    && hi_val.checked_add(1) == Some(lo_val)
                {
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
                    if hi_val
                        .checked_sub(lo_val)
                        .is_some_and(|d| (0..=1000).contains(&d))
                    {
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

                // The empty range `hi = lo − 1` before anything else (a
                // range with `hi < lo − 1` is the reciprocal of the product
                // over `[hi + 1, lo − 1]`, Karr's convention: the engine's).
                if let (Some(lo_val), Some(hi_val)) = (lo_int, hi_int)
                    && hi_val.checked_add(1) == Some(lo_val)
                {
                    cache.insert(id, arena.one);
                    continue;
                }

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
                    if hi_val
                        .checked_sub(lo_val)
                        .is_some_and(|d| (0..=1000).contains(&d))
                    {
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
///
/// A constant with any negative rational coefficient is negated too (a
/// negative rational, `−√2/2`, `−2π/7`), as SymPy extracts the sign
/// (`cos(-sqrt(2)/2)` → `cos(sqrt(2)/2)`, `sinh(-1)` → `-sinh(1)`): up to
/// 0.46 `cos(−√2)` was `cos(√2)` but `cos(−√2/2)` and `cos(−2)` stayed.  A
/// symbolic product keeps its coefficient (`sin(−2x)`), as before.
fn as_negated(arena: &mut Arena, id: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(id) {
        return r
            .is_negative()
            .then(|| -r.clone())
            .map(|q| arena.num_ratio(q));
    }
    if let ExprNode::Mul(children) = arena.node(id)
        && children.len() >= 2
        && arena
            .as_num(children[0])
            .is_some_and(|c| c.is_negative() && !(-c.clone()).is_one())
        && walk::free_symbols(arena, id).is_empty()
    {
        return as_negated_general(arena, id);
    }
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
///
/// For an integer `m` (every caller): `(p mod q·m)/q` for `a = p/q`, in
/// lowest terms as `a` is, by integer arithmetic linear in the size of
/// `p`; `Ratio`'s `%` reduces with a gcd quadratic in it even against a
/// one-word operand.
fn mod_positive(a: &Q, m: &Q) -> Q {
    use num_integer::Integer;
    if m.is_integer() && m.is_positive() {
        let r = a.numer().mod_floor(&(a.denom() * m.numer()));
        return if r.is_zero() {
            Q::zero()
        } else {
            Ratio::new_raw(r, a.denom().clone())
        };
    }
    let mut result = a % m;
    if result.is_negative() {
        result += m;
    }
    result
}

/// Does the denominator of `q` (in lowest terms) divide `n`?  The explicit
/// special values of `sin`, `cos` and `tan` at `q·π` are all at
/// denominators dividing 4 or 6 (the others are at the few denominators of
/// [`COS_PI_DENOMINATORS`]), and the period and quadrant reductions (`q mod
/// 2`, `1 − q`, `q − 1`, `2 − q`) keep the denominator, so any other `q` has
/// none: returning early spares the reductions, each a few gcds and an
/// interned number at the full size of `q` (`simplify(sin(qπ)² + cos(qπ)²)`
/// for a 5,000-digit `q` took 4 s, through hundreds of such evaluations).
fn denominator_divides(q: &Q, n: u32) -> bool {
    q.denom()
        .to_u32()
        .is_some_and(|d| d != 0 && n.is_multiple_of(d))
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

    /// Before (0.44): `sign_of_real_constant(−1 − i·(e^{10⁻³⁰} − 1 −
    /// 10⁻³⁰))` was `Less`: the certified 16-digit value shows no
    /// imaginary part (it is `−5·10⁻⁶¹`; mpmath `exp(mpf(10)**-30) - 1 -
    /// mpf(10)**-30` = `5.0000000000000000000000000000016…e-61`).  Now the
    /// constant must be real by the assumptions.
    #[test]
    fn sign_of_real_constant_needs_realness() {
        let mut a = Arena::new();
        let one = a.one;
        let ten = a.int(10);
        let minus_thirty = a.int(-30);
        let h = a.pow(ten, minus_thirty);
        let e = a.exp(h);
        let neg_h = a.neg(h);
        let neg_one = a.neg_one;
        let hidden = a.add(&[e, neg_one, neg_h]);
        let im = a.mul(&[a.i_unit, hidden]);
        let neg_im = a.neg(im);
        let z = a.add(&[a.neg_one, neg_im]);
        assert_eq!(a.sign_of_real_constant(z), None, "{}", display(&a, z));
        // Real constants keep their certified sign.
        let ln2 = {
            let two = a.int(2);
            a.ln(two)
        };
        assert_eq!(
            a.sign_of_real_constant(ln2),
            Some(std::cmp::Ordering::Greater)
        );
        let sqrt2 = {
            let two = a.int(2);
            a.sqrt(two)
        };
        let d = a.sub(sqrt2, one);
        assert_eq!(
            a.sign_of_real_constant(d),
            Some(std::cmp::Ordering::Greater)
        );
        let d = a.sub(one, sqrt2);
        assert_eq!(a.sign_of_real_constant(d), Some(std::cmp::Ordering::Less));
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

    /// Every table value is `cos(kπ/n)` / `tan(kπ/n)` (to 1e-14), and the
    /// inverse function of it and of its negative reduces to the angle.
    #[test]
    fn special_angle_tables_are_right_and_invert() {
        use std::f64::consts::PI;
        for (table, entries) in [
            (SpecialTable::Cos, SPECIAL_COS),
            (SpecialTable::Tan, SPECIAL_TAN),
        ] {
            for e in entries {
                let mut a = Arena::new();
                let v = special_value_node(&mut a, e);
                let angle = PI * e.k as f64 / e.n as f64;
                let exact = match table {
                    SpecialTable::Cos => angle.cos(),
                    SpecialTable::Tan => angle.tan(),
                };
                let got = crate::transforms::evalf::evalf_f64(&a, v).unwrap();
                assert!(
                    (got - exact).abs() < 1e-14,
                    "{}/{}: {got} vs {exact}",
                    e.k,
                    e.n
                );
                let q = Ratio::new(BigInt::from(e.k), BigInt::from(e.n));
                let neg = a.neg(v);
                let neg = eval(&mut a, neg);
                let (f_pos, f_neg, want_pos, want_neg) = match table {
                    SpecialTable::Cos => {
                        (a.acos(v), a.acos(neg), q.clone(), Ratio::one() - q.clone())
                    }
                    SpecialTable::Tan => (a.atan(v), a.atan(neg), q.clone(), -q.clone()),
                };
                let got_pos = eval(&mut a, f_pos);
                let got_neg = eval(&mut a, f_neg);
                assert_eq!(
                    as_pi_multiple(&a, got_pos),
                    Some(want_pos),
                    "{}/{}",
                    e.k,
                    e.n
                );
                assert_eq!(
                    as_pi_multiple(&a, got_neg),
                    Some(want_neg),
                    "-{}/{}",
                    e.k,
                    e.n
                );
            }
        }
    }

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
        // li(e^t) = Ei(t) needs Im t ∈ (−π, π]: a real t, not a complex x
        // (mpmath li(exp(4j)) ≠ ei(4j)).
        let t = sym(&mut a, "t");
        let ExprNode::Symbol(sid) = *a.node(t) else {
            panic!("symbol expected")
        };
        let mut real = a.symbol_assumptions(sid);
        real.assert_true(crate::base::assumptions::Props::REAL);
        a.set_symbol_assumptions(sid, real);
        let et = a.exp(t);
        let li_et = a.li(et);
        let ei_t = a.ei(t);
        assert_eq!(li_et, ei_t);
        let ex = a.exp(x);
        let li_ex = a.li(ex);
        assert!(matches!(a.node(li_ex), ExprNode::Li(_)));
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
