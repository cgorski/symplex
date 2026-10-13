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
use crate::base::walk;

mod atan_sum;
mod cyclotomic;

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

/// The four relational node kinds (`a < b` is `Gt(b, a)`, `a ≤ b` is
/// `Ge(b, a)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationKind {
    Gt,
    Ge,
    Eq,
    Ne,
}

impl RelationKind {
    /// The kind of a relational node; `Ne` for anything else (callers match
    /// the four relational nodes first).
    fn of(node: &ExprNode) -> Self {
        match node {
            ExprNode::Gt(..) => Self::Gt,
            ExprNode::Ge(..) => Self::Ge,
            ExprNode::Eq_(..) => Self::Eq,
            _ => Self::Ne,
        }
    }
}

/// The truth of the relation `kind` between the evaluated operands `a`,
/// `b`, as `eval` decides it: two rationals compare; an order against 0 by
/// the assumptions ([`order_against_zero`]); an order or an equality by the
/// difference `a − b` (a rational, a certified sign, the square-root tower
/// zero test, the minimal polynomial: [`order_by_difference`],
/// [`equal_by_difference`]).  `None` when undecided.  `orig` are the
/// operands before evaluation, whose difference the exact zero tests also
/// try ([`Difference`]).
fn relation_truth(
    arena: &mut Arena,
    kind: RelationKind,
    a: ExprId,
    b: ExprId,
    orig: (ExprId, ExprId),
) -> Option<bool> {
    // `nan` equals nothing, itself included (SymPy 1.14: `Eq(nan, -1)` and
    // `Eq(nan, nan)` are `False`), as Boolean `simplify` decides it
    // (`logic::fold_rel`); it has no order, which stays.  Up to 0.46 `eval`
    // left `nan == -1` while `simplify` gave `False`.
    let nan = arena.nan;
    if (a == nan || b == nan) && matches!(kind, RelationKind::Eq | RelationKind::Ne) {
        return Some(kind == RelationKind::Ne);
    }
    let d = Difference { a, b, orig };
    match kind {
        RelationKind::Gt | RelationKind::Ge => {
            let strict = kind == RelationKind::Gt;
            if let (Some(ra), Some(rb)) = (arena.as_num(a), arena.as_num(b)) {
                return Some(if strict { ra > rb } else { ra >= rb });
            }
            order_against_zero(arena, a, b, strict)
                .or_else(|| order_by_difference(arena, d, strict))
        }
        RelationKind::Eq | RelationKind::Ne => {
            let equal = gaussian_equal(arena, a, b).or_else(|| equal_by_difference(arena, d))?;
            Some(equal == (kind == RelationKind::Eq))
        }
    }
}

/// The operands `a`, `b` of a relation after evaluation, and `orig` before:
/// the exact zero tests try `a − b`, then the difference of the original
/// operands, where the trigonometric values are not yet radicals — the
/// cyclotomic test decides `cos(13π/16)·tan(23π/12)·cos(23π/12) =
/// −√(…)·sin(23π/12)` on the original form, not once `tan(23π/12)` is `√3 −
/// 2` and the cosine `(√6 + √2)/4` (radicals it takes as independent
/// generators).
#[derive(Clone, Copy)]
struct Difference {
    a: ExprId,
    b: ExprId,
    orig: (ExprId, ExprId),
}

impl Difference {
    /// The difference of the original operands, when they differ from the
    /// evaluated ones.
    fn original(self, arena: &mut Arena) -> Option<ExprId> {
        (self.orig != (self.a, self.b)).then(|| arena.sub(self.orig.0, self.orig.1))
    }
}

/// The truth of the relational node `rel` (`Gt`, `Ge`, `Eq_`, `Ne`) as
/// [`eval`] decides it, after evaluating its two operands; `None` when
/// `eval` leaves it a relation (or `rel` is not one).  For Boolean
/// `simplify`, which must decide every relation `eval` decides
/// (`Eq(cos(π/8), √(2 + √2)/2)`: `eval` gave `True` by the square-root
/// tower, `simplify` kept the relation up to 0.44).
pub(crate) fn decide_relation(arena: &mut Arena, rel: ExprId) -> Option<bool> {
    let (a, b) = match *arena.node(rel) {
        ExprNode::Gt(a, b) | ExprNode::Ge(a, b) | ExprNode::Eq_(a, b) | ExprNode::Ne(a, b) => {
            (a, b)
        }
        _ => return None,
    };
    let kind = RelationKind::of(arena.node(rel));
    let ea = eval(arena, a);
    let eb = eval(arena, b);
    relation_truth(arena, kind, ea, eb, (a, b))
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

/// `a > b` (`strict`) or `a ≥ b` for real `a`, `b` (by their assumptions)
/// whose canonical difference `a − b` is a rational number: `tanh 1 > tanh 1`
/// is false, `tanh 1 ≥ tanh 1` and `π + 1 > π` true.  `None` otherwise
/// (`∞ − ∞` is `nan`, not a number).
///
/// Before 0.41 such a relation stayed for `evalf`, which decided `a > a`
/// true: `Piecewise((π, tanh 1 < tanh 1), (0, True))` evaluated to `π`
/// (SymPy 1.14: `0`).
///
/// A constant difference that is an algebraic number is decided by its
/// minimal polynomial ([`algebraic_rational_value`]) and, when irrational,
/// by its certified sign: `√(5 − 2√6) ≥ √3 − √2` is true (the two are
/// equal; `evalf` refused it before 0.42).
fn order_by_difference(arena: &mut Arena, diff: Difference, strict: bool) -> Option<bool> {
    let (a, b) = (diff.a, diff.b);
    let d = arena.sub(a, b);
    let rational = arena.as_num(d).cloned();
    if rational.is_none() && !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    if !known_real(arena, a) || !known_real(arena, b) {
        return None;
    }
    let q = match rational {
        Some(q) => q,
        None => {
            // A certified nonzero value decides at once (the minimal
            // polynomial of a long radical can take seconds).  `d` is real:
            // `a` and `b` are.
            if let Some(s) = certified_real_sign(arena, d) {
                return Some(s == std::cmp::Ordering::Greater);
            }
            // Arctangents of rationals: `atan(1/2) + atan(1/3) ≥ π/4`.
            let d = atan_sum::without_zero_atan_part(arena, d);
            if let Some(zero) = atan_sum::is_zero(arena, d) {
                return zero.then_some(!strict);
            }
            if sqrt_tower_is_zero(arena, d) {
                return Some(!strict);
            }
            match algebraic_rational_value(arena, d) {
                Some(Some(q)) => q,
                // Irrational, so nonzero: the certified sign decides.
                Some(None) => {
                    return Some(certified_real_sign(arena, d)? == std::cmp::Ordering::Greater);
                }
                // Trigonometric values at rational multiples of π (and
                // radicals over them): an exact zero is `a = b`; a nonzero
                // difference whose sign no evaluation certified stays.
                None => {
                    let zero = cyclotomic::is_zero(arena, d).or_else(|| {
                        let original = diff.original(arena)?;
                        exact_zero_test(arena, original)
                    })?;
                    return zero.then_some(!strict);
                }
            }
        }
    };
    Some(if strict {
        q.is_positive()
    } else {
        !q.is_negative()
    })
}

/// Is `x` real: by its assumptions, or, for a constant they leave open
/// (`√(5 − 2√6)`), by a certified evaluation without imaginary part?
///
/// The numerical half evaluates `im(x)` on its own
/// ([`constant_realness_as_declared`](crate::transforms::realness::constant_realness_as_declared)):
/// a certified value of `x` whose imaginary part shows as 0 proves
/// nothing, since an imaginary part below the relative precision of the
/// whole value is not shown — `−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)` (imaginary
/// part `−5·10⁻⁶¹`) passed for a negative real, and `ln` of it folded to
/// `ln(…) + iπ` on the wrong side of the cut (it is `−iπ + …`).
///
/// `im(x)` must be a certified exact 0 (`Settled::Certified`): one that
/// is only 0 to the precision of the zero search may hide a smaller part
/// (`sin²1 + cos²1 − 1` is exactly 0, but no evaluation proves it, and the
/// side of the cut stays undecided, as `evalf` leaves it).
///
/// An algebraic `x` whose imaginary part is only 0 to the precision
/// reached is decided by its minimal polynomial ([`algebraic_is_real`]):
/// the real root of a cubic written by Cardano's formula, whose complex
/// cube roots cancel only exactly, is real (up to 0.44 it was undecided,
/// and `√λ`, `ln λ` of such a negative eigenvalue were refused by `evalf`
/// as `PrecisionExhausted` on the cut).  Beyond the reach of the minimal
/// polynomial (a root by Ferrari's formula, a radical over trigonometric
/// values) `x − x̄` is tested for zero exactly in `ℚ(ζ)[radicals]`
/// ([`cyclotomic::is_real`]): up to 0.44 the eigenvalues of a 4×4 matrix
/// with a parameter did not evaluate where the quartic has complex roots.
///
/// The zero search stops at the cap of the evaluation (twice the digits
/// plus 256 bits): searching deeper only certifies a smaller nonzero part,
/// and `x` is then not real either — an exact zero shows at any precision,
/// and one that cancels only exactly (the radicands of Ferrari's formula)
/// shows at none.  Up to 0.44 a deep search (1024 bits more) ran in vain on
/// every `eval` of a quartic's roots, and `solve` was ten times slower.
fn known_real(arena: &mut Arena, x: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    match AssumptionCache::new().query(arena, x, Props::REAL) {
        Some(real) => real,
        None => {
            if !walk::free_symbols(arena, x).is_empty() {
                return false;
            }
            constant_fact(arena, x, FACT_REAL, |arena| {
                // A certified nonzero imaginary part decides at once (the
                // exact tests below are for parts that cancel only exactly;
                // a certified 16-digit value shows a part of any size, here
                // 10⁻¹³ of the modulus).  Without it the cyclotomic walk did
                // exact radical arithmetic on `(((10⁻¹²i +
                // 13/4)^(−3/4))^(−3/2))^(−3/2)` for a second per `eval`
                // (`fuzz_simplify` timed out, 0.47 development).
                if let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 16, ZeroSearch::Cap)
                    && !z.1.is_zero()
                {
                    return false;
                }
                // The exact conjugation test next: a walk over the
                // radicals, which settles a root by Cardano's or Ferrari's
                // formula at once, where the zero search of its imaginary
                // part runs to the cap and the minimal polynomial takes a
                // Zassenhaus factorisation (`solve` of a septic with a
                // quartic factor took 32 ms in 0.46, 1 ms in 0.43).  Any
                // other constant leaves the walk at its first node of
                // another kind.
                if let Some(real) = cyclotomic::is_real(arena, x) {
                    return real;
                }
                let im = arena.im(x);
                let im = eval(arena, im);
                if arena.is_zero_structural(im) {
                    return true;
                }
                if !walk::has_unevaluated(arena, im)
                    && let Ok((z, Settled::Certified)) =
                        evalf_settled(arena, im, 16, ZeroSearch::Cap)
                {
                    return z.0.is_zero() && z.1.is_zero();
                }
                algebraic_is_real(arena, x)
            })
        }
    }
}

/// [`known_real`] of a symbol-free constant (a key of
/// [`Arena::constant_facts`]).
const FACT_REAL: u8 = 0;
/// [`constant_is_negative`] (a key of [`Arena::constant_facts`]).
const FACT_NEGATIVE: u8 = 1;

/// The answer `prove` gives for the symbol-free constant `x`, memoised in
/// [`Arena::constant_facts`] under `fact`: the proofs evaluate, search for
/// zeros and take minimal polynomials, and `eval` asks again on every pass
/// over the same radicals.
fn constant_fact(
    arena: &mut Arena,
    x: ExprId,
    fact: u8,
    prove: impl FnOnce(&mut Arena) -> bool,
) -> bool {
    if let Some(&known) = arena.constant_facts.get(&(x, fact)) {
        return known;
    }
    let answer = prove(arena);
    arena.constant_facts.insert((x, fact), answer);
    answer
}

/// Is the constant algebraic number `x` real, by its minimal polynomial
/// `m` (small expressions only, see below)?  `m` is irreducible, so `x`
/// is one of its roots; the roots get certified disjoint inclusion disks
/// ([`crate::poly::roots::root_balls`], which also certifies a root real
/// when the mirror image of its disk meets no other disk), and `x` its
/// certified 30-digit value.  When the value's error disk meets exactly one
/// root disk, `x` is that root, and real when the root is.  `false` when
/// anything is not certified.
///
/// `x` is [`small_algebraic`], or a sum of such terms whose minimal
/// polynomials have degrees of product at most [`MAX_RESULTANT_DEGREE`]
/// ([`small_algebraic_sum`]).  The static bound of
/// [`largest_resultant_degree`] counts the radicals of `ω·∛u` (`ω` a cube
/// root of unity, `u` a quadratic complex number) as independent, and a
/// root by Cardano's formula exceeds it, though its two terms have minimal
/// polynomials of degree 6 and the resultant of the sum has degree 36 (70
/// ms in a debug build).  The test runs only for a constant whose
/// imaginary part the zero search to the cap left at 0 to its precision.
fn algebraic_is_real(arena: &mut Arena, x: ExprId) -> bool {
    use crate::base::bigcomplex::{c_abs, c_sub};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    use astro_float::{BigFloat, RoundingMode};
    if !small_algebraic(arena, x) && !small_algebraic_sum(arena, x) {
        return false;
    }
    let Some(mp) = crate::poly::algebraic::minimal_polynomial(arena, x) else {
        return false;
    };
    match mp.degree() {
        Some(1) => return true,
        Some(d) if (2..=64).contains(&d) => {}
        _ => return false,
    }
    let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 30, ZeroSearch::Cap) else {
        return false;
    };
    const PREC: usize = 256;
    let rm = RoundingMode::Up;
    let roots = crate::poly::roots::rootof_roots(&mp, PREC);
    let balls = crate::poly::roots::root_balls(&mp, &roots, PREC);
    // The value carries 30 digits relative to its modulus: its error is
    // below |z|·10⁻²⁸ (two digits of margin).
    let margin = BigFloat::from_f64(1e-28, 64);
    let eps = c_abs(&z, 64, rm).mul(&margin, 64, rm);
    let mut meets = None;
    for (k, ball) in balls.iter().enumerate() {
        let Some(r) = &ball.radius else {
            return false;
        };
        let d = c_abs(
            &c_sub(&z, &ball.value, PREC, RoundingMode::ToEven),
            64,
            RoundingMode::Down,
        );
        // Reach of the two disks, enlarged by 2⁻³⁰ for the roundings.
        let reach = r.add(&eps, 64, rm);
        let reach = reach.add(&reach.mul(&BigFloat::from_f64(1e-9, 64), 64, rm), 64, rm);
        if d.cmp(&reach).is_none_or(|o| o <= 0) {
            if meets.is_some() {
                return false;
            }
            meets = Some(k);
        }
    }
    meets.is_some_and(|k| balls[k].real)
}

/// The sign of the symbol-free constant `x`, which the caller knows to be
/// real, from a certified 16-digit value (its error ball excludes 0);
/// `None` when it does not settle or is not real to the digits.  The
/// caller's realness proof matters: an imaginary part below the relative
/// precision of the value is not shown (`−1 − i·(e^{10⁻³⁰} − 1 − 10⁻³⁰)`
/// evaluates to `−1`).  [`Arena::sign_of_real_constant`] proves realness
/// by the assumptions.
pub(crate) fn certified_real_sign(arena: &Arena, x: ExprId) -> Option<std::cmp::Ordering> {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    match evalf_settled(arena, x, 16, ZeroSearch::Cap) {
        Ok((z, Settled::Certified)) if z.1.is_zero() && !z.0.is_zero() => {
            Some(if z.0.is_negative() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            })
        }
        _ => None,
    }
}

/// The signs of the real and imaginary parts of the symbol-free constant
/// `x` from one certified 16-digit value: a part is settled when it is
/// nonzero and within `2⁴⁰` of the larger part, far inside the digits the
/// value carries (a smaller part may be rounding noise next to the larger
/// one, and is left to an evaluation of its own).  `None` when the value
/// does not settle.
pub(crate) fn certified_part_signs(
    arena: &Arena,
    x: ExprId,
) -> Option<crate::base::arena::PartSigns> {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    use std::cmp::Ordering;
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let Ok((z, Settled::Certified)) = evalf_settled(arena, x, 16, ZeroSearch::Cap) else {
        return None;
    };
    let exponent = |part: &astro_float::BigFloat| {
        if part.is_zero() || part.is_nan() || part.is_inf() {
            None
        } else {
            part.exponent()
        }
    };
    let top = exponent(&z.0).max(exponent(&z.1));
    let sign = |part: &astro_float::BigFloat| -> Option<Ordering> {
        let e = exponent(part)?;
        if top.is_some_and(|t| i64::from(e) < i64::from(t) - 40) {
            return None;
        }
        Some(if part.is_negative() {
            Ordering::Less
        } else {
            Ordering::Greater
        })
    };
    Some(crate::base::arena::PartSigns {
        re: sign(&z.0),
        im: sign(&z.1),
    })
}

/// `a = b` decided by the difference `a − b`: 0 or a nonzero Gaussian
/// rational in canonical form (`airyai(18/7) + 3 − 10⁻¹⁵⁰ ≠ 3 +
/// airyai(18/7)`, `x + 1 ≠ x`), or a constant algebraic number that its
/// minimal polynomial decides ([`algebraic_rational_value`]: `√(3 + 2√2) =
/// 1 + √2`).  `None` otherwise.
///
/// Up to 0.41 only two Gaussian rationals were compared, and both relations
/// stayed for `evalf`, which decides an inequality numerically but cannot
/// certify an equality (it refused `√(3 + 2√2) = 1 + √2`; SymPy 1.14:
/// `True`, by `minimal_polynomial` in `Expr.equals`).
fn equal_by_difference(arena: &mut Arena, diff: Difference) -> Option<bool> {
    let (a, b) = (diff.a, diff.b);
    // Numbers only: no difference of sets, truth values or relations.
    let non_scalar = |id: ExprId| {
        let node = arena.node(id);
        node.is_set_node()
            || matches!(
                node,
                ExprNode::BoolTrue
                    | ExprNode::BoolFalse
                    | ExprNode::Gt(..)
                    | ExprNode::Ge(..)
                    | ExprNode::Eq_(..)
                    | ExprNode::Ne(..)
                    | ExprNode::And(_)
                    | ExprNode::Or(_)
                    | ExprNode::Not(_)
            )
    };
    if non_scalar(a) || non_scalar(b) {
        return None;
    }
    let d = arena.sub(a, b);
    if let Some((re, im)) = gaussian_rational(arena, d) {
        return Some(re.is_zero() && im.is_zero());
    }
    if !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    // A certified nonzero value decides at once, before the minimal
    // polynomial: `tan(79π/40)·cos(79π/40) = sin(79π/40) + 1/448` took 25 s
    // there once the values were radicals.
    if constant_is_nonzero(arena, d) {
        return Some(false);
    }
    exact_zero_test(arena, d).or_else(|| {
        let original = diff.original(arena)?;
        exact_zero_test(arena, original)
    })
}

/// Is the constant `d` zero, as `eval` decides `Eq(d, 0)`?  `Some(false)`
/// when a certified 16-digit value excludes 0, else the exact tests of
/// [`exact_zero_test`] on `eval(d)`, then on `d`; `None` when nothing
/// decides (or `d` has symbols).  The zero test of `eval` of `Eq`/`Ne`
/// ([`equal_by_difference`]), of [`eval_sign`] and of the assumption
/// queries of constants ([`constant_facts`]).
pub(crate) fn constant_zero_decision(arena: &mut Arena, d: ExprId) -> Option<bool> {
    if !walk::free_symbols(arena, d).is_empty() {
        return None;
    }
    if constant_is_nonzero(arena, d) {
        return Some(false);
    }
    let v = eval(arena, d);
    if arena.is_zero_structural(v) {
        return Some(true);
    }
    exact_zero_test(arena, v).or_else(|| (v != d).then(|| exact_zero_test(arena, d)).flatten())
}

/// The exact zero tests of a symbol-free constant `d` whose certified value
/// did not exclude 0: sums of arctangents of rationals and multiples of π
/// ([`atan_sum::is_zero`]), the square-root tower, the minimal polynomial,
/// and trigonometric values at rational multiples of π, roots of unity and
/// radicals over them, exactly in `ℚ(ζ)[radicals]` ([`cyclotomic::is_zero`]).
fn exact_zero_test(arena: &mut Arena, d: ExprId) -> Option<bool> {
    // An arctangent part that is 0 by itself leaves the rest to decide.
    let d = atan_sum::without_zero_atan_part(arena, d);
    if let Some(zero) = atan_sum::is_zero(arena, d) {
        return Some(zero);
    }
    if sqrt_tower_is_zero(arena, d) {
        return Some(true);
    }
    match algebraic_rational_value(arena, d) {
        Some(Some(q)) => Some(q.is_zero()),
        Some(None) => Some(false),
        None => cyclotomic::is_zero(arena, d),
    }
}

/// The constant algebraic number `d` as a rational, by its minimal
/// polynomial over ℚ: `Some(Some(q))` when the polynomial is linear,
/// `Some(None)` when `d` is irrational (so not 0).  The polynomial is
/// certified (`minimal_polynomial` returns a factor only when it is the one
/// irreducible factor of the resultant that vanishes at the 320-bit value
/// of `d`; two candidates, a tiny nonzero `d` among them, give `None`).
/// `None` as well when `d` is not small ([`small_algebraic`]).
fn algebraic_rational_value(arena: &mut Arena, d: ExprId) -> Option<Option<Q>> {
    if !small_algebraic(arena, d) {
        return None;
    }
    let mp = crate::poly::algebraic::minimal_polynomial(arena, d)?;
    match mp.degree()? {
        1 => Some(Some(-mp.coeff(0) / mp.coeff(1))),
        _ => Some(None),
    }
}

/// Is `d` built from rationals, `i`, `φ` and rational powers by sums and
/// products, in at most 160 nodes, with the product of the orders of its
/// roots (a bound on the degree of its minimal polynomial) at most 64, and
/// no intermediate resultant above [`MAX_RESULTANT_DEGREE`]
/// ([`largest_resultant_degree`])?  The bounds keep the resultants of
/// [`algebraic_rational_value`] small.
fn small_algebraic(arena: &Arena, d: ExprId) -> bool {
    let order = walk::post_order_ids(arena, d);
    if order.len() > 160 {
        return false;
    }
    let mut degree: u64 = 1;
    for &id in &order {
        let factor = match arena.node(id) {
            ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => 1,
            ExprNode::ImaginaryUnit | ExprNode::GoldenRatio => 2,
            ExprNode::Pow(_, e) => match arena.as_num(*e) {
                Some(q) => q.denom().to_u64().unwrap_or(u64::MAX),
                None => return false,
            },
            _ => return false,
        };
        degree = degree.saturating_mul(factor);
        if degree > 64 {
            return false;
        }
    }
    largest_resultant_degree(arena, &order) <= MAX_RESULTANT_DEGREE
}

/// The largest resultant [`algebraic_rational_value`] lets
/// `minimal_polynomial` compute (see [`largest_resultant_degree`]).
const MAX_RESULTANT_DEGREE: u64 = 64;

/// Is `x` a sum of [`small_algebraic`] terms, in at most 160 nodes, whose
/// minimal polynomials have degrees of product at most
/// [`MAX_RESULTANT_DEGREE`]?  `minimal_polynomial` composes a sum term by
/// term, by resultants of degree `deg(partial sum) · deg(term)`, so the
/// product bounds every resultant it forms on `x` (the terms' own work is
/// bounded by [`small_algebraic`]).
fn small_algebraic_sum(arena: &mut Arena, x: ExprId) -> bool {
    let ExprNode::Add(terms) = arena.node(x).clone() else {
        return false;
    };
    if walk::post_order_ids(arena, x).len() > 160 {
        return false;
    }
    let mut degree: usize = 1;
    for t in terms {
        if arena.as_num(t).is_some() {
            continue;
        }
        if !small_algebraic(arena, t) {
            return false;
        }
        let Some(d) = crate::poly::algebraic::minimal_polynomial(arena, t).and_then(|m| m.degree())
        else {
            return false;
        };
        degree = degree.saturating_mul(d);
        if degree as u64 > MAX_RESULTANT_DEGREE {
            return false;
        }
    }
    true
}

/// An upper bound on the degree of the largest resultant
/// `minimal_polynomial` forms on the nodes `order` (a post-order of a
/// [`small_algebraic`] expression): it composes sums and products one
/// operand at a time, by a resultant of degree `deg a · deg b` that is only
/// then factored, and `a^(p/q)` by one of degree `q · deg a`; a minimal
/// polynomial's degree is at most the product of the orders of the distinct
/// radicals in its expression.  The final degree alone does not bound the
/// work: `sin(7π/60)² + cos(7π/60)² − 1` has degree at most 16 but
/// resultants of degree 256 (23 s in a debug build once the values were
/// radicals).
fn largest_resultant_degree(arena: &Arena, order: &[ExprId]) -> u64 {
    // The distinct radicals `(base, order)` of each node.
    type Radicals = smallvec::SmallVec<[(ExprId, u64); 8]>;
    fn field_degree(r: &Radicals) -> u64 {
        r.iter().fold(1u64, |d, &(_, k)| d.saturating_mul(k))
    }
    fn union(a: &mut Radicals, b: &Radicals) {
        for &x in b {
            if !a.contains(&x) {
                a.push(x);
            }
        }
    }
    let mut radicals: FxHashMap<ExprId, Radicals> = FxHashMap::default();
    let mut worst = 1u64;
    for &id in order {
        let mut own = Radicals::new();
        match arena.node(id) {
            ExprNode::ImaginaryUnit | ExprNode::GoldenRatio => own.push((id, 2)),
            ExprNode::Neg(a) => {
                if let Some(r) = radicals.get(a) {
                    own = r.clone();
                }
            }
            ExprNode::Pow(b, e) => {
                let inner = radicals.get(b).cloned().unwrap_or_default();
                let q = arena
                    .as_num(*e)
                    .and_then(|q| q.denom().to_u64())
                    .unwrap_or(u64::MAX);
                own = inner;
                if q > 1 {
                    worst = worst.max(q.saturating_mul(field_degree(&own)));
                    if !own.contains(&(*b, q)) {
                        own.push((*b, q));
                    }
                }
            }
            ExprNode::Add(cs) | ExprNode::Mul(cs) => {
                let mut first = true;
                for c in cs.iter() {
                    let Some(r) = radicals.get(c).filter(|r| !r.is_empty()) else {
                        continue;
                    };
                    if !first {
                        worst = worst.max(field_degree(&own).saturating_mul(field_degree(r)));
                    }
                    first = false;
                    union(&mut own, r);
                }
            }
            _ => {}
        }
        radicals.insert(id, own);
    }
    worst
}

// ── Exact zero test in a tower of real quadratic extensions ─────────────────
//
// The values of `sin`, `cos`, `tan` at the denominators of
// [`COS_PI_DENOMINATORS`] are square-root towers (`√(√5/8 + 5/8)`, `1/(√6/4
// − √2/4)`), and an identity between them (`sin²x + cos²x = 1` at `x =
// 7π/60`) is a zero that `minimal_polynomial` finds only through
// resultants of degree up to 256 (23 s in a debug build).  Here the
// constant is computed exactly in `ℚ(t₁, …, tₖ)`, `tᵢ² = rᵢ` with `rᵢ` in
// the tower below: an element has `2ᵏ` rational coordinates (the
// monomials in the `tᵢ`), a product costs about `4ᵏ` rational products.
// Sending each `tᵢ` to the positive root `√rᵢ` is a ring homomorphism to
// ℝ, so an element computed as 0 has the value 0, whatever relations the
// generators satisfy.  Relations are what make the zero visible
// (`√(1/2 − √2/4) = (√2 − 1)·√(1/2 + √2/4)`), so a new radicand is first
// tried as a square in the tower ([`SqrtTower::sqrt`], every root verified
// by squaring it) and its root taken with the sign of its certified value;
// only otherwise is it a new generator.

/// The most generators a [`SqrtTower`] takes.
const SQRT_TOWER_MAX_GENERATORS: usize = 7;

/// A tower `ℚ(t₁, …, tₖ)` of quadratic extensions (see the section
/// comment).  An element at level `k` is a vector of `2ᵏ` coordinates: the
/// first half the part without `tₖ`, the second half the coefficient of
/// `tₖ`, recursively.
struct SqrtTower {
    /// `rᵢ = tᵢ²`, an element of level `i`.
    radicands: Vec<Vec<Q>>,
    /// The real radical `√b` each generator stands for.
    roots: Vec<ExprId>,
}

fn tower_add(x: &[Q], y: &[Q]) -> Vec<Q> {
    x.iter().zip(y).map(|(a, b)| a + b).collect()
}

fn tower_sub(x: &[Q], y: &[Q]) -> Vec<Q> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}

fn tower_is_zero(x: &[Q]) -> bool {
    x.iter().all(Zero::is_zero)
}

/// `x` as an element of level `k ≥` its own (zero coordinates for the new
/// monomials).
fn tower_pad(x: &[Q], k: usize) -> Vec<Q> {
    let mut v = x.to_vec();
    v.resize(1 << k, Q::zero());
    v
}

/// The square root of a non-negative rational, when rational.
fn rational_sqrt(q: &Q) -> Option<Q> {
    if q.is_negative() {
        return None;
    }
    let (n, d) = (q.numer().sqrt(), q.denom().sqrt());
    (&n * &n == *q.numer() && &d * &d == *q.denom()).then(|| Ratio::new(n, d))
}

impl SqrtTower {
    fn level(&self) -> usize {
        self.radicands.len()
    }

    /// `x·y` at level `k`: `(x₀ + x₁t)(y₀ + y₁t) = x₀y₀ + x₁y₁·r +
    /// ((x₀ + x₁)(y₀ + y₁) − x₀y₀ − x₁y₁)·t`.  The recursion is on the
    /// level (at most [`SQRT_TOWER_MAX_GENERATORS`] deep).
    fn mul(&self, x: &[Q], y: &[Q], k: usize) -> Vec<Q> {
        if k == 0 {
            return vec![&x[0] * &y[0]];
        }
        if tower_is_zero(x) || tower_is_zero(y) {
            return vec![Q::zero(); 1 << k];
        }
        let h = 1 << (k - 1);
        let (x0, x1) = x.split_at(h);
        let (y0, y1) = y.split_at(h);
        let p = self.mul(x0, y0, k - 1);
        let q = self.mul(x1, y1, k - 1);
        let s = self.mul(&tower_add(x0, x1), &tower_add(y0, y1), k - 1);
        let qr = self.mul(&q, &self.radicands[k - 1], k - 1);
        let mut out = tower_add(&p, &qr);
        out.extend(tower_sub(&tower_sub(&s, &p), &q));
        out
    }

    /// `1/x` at level `k`, by `1/(x₀ + x₁t) = (x₀ − x₁t)/(x₀² − x₁²r)`;
    /// `None` when a norm on the way is 0.
    fn inverse(&self, x: &[Q], k: usize) -> Option<Vec<Q>> {
        if k == 0 {
            return (!x[0].is_zero()).then(|| vec![x[0].recip()]);
        }
        let h = 1 << (k - 1);
        let (x0, x1) = x.split_at(h);
        let x1_sq = self.mul(x1, x1, k - 1);
        let norm = tower_sub(
            &self.mul(x0, x0, k - 1),
            &self.mul(&x1_sq, &self.radicands[k - 1], k - 1),
        );
        let inv = self.inverse(&norm, k - 1)?;
        let mut out = self.mul(x0, &inv, k - 1);
        out.extend(self.mul(x1, &inv, k - 1).into_iter().map(|c| -c));
        Some(out)
    }

    /// `xⁿ` at level `k` (`|n| ≤ 64`).
    fn pow(&self, x: &[Q], n: i64, k: usize) -> Option<Vec<Q>> {
        if n.unsigned_abs() > 64 {
            return None;
        }
        let mut base = if n < 0 {
            self.inverse(x, k)?
        } else {
            x.to_vec()
        };
        let mut out = tower_pad(&[Q::one()], k);
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 {
                out = self.mul(&out, &base, k);
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(&base, &base, k);
            }
        }
        Some(out)
    }

    /// A square root of `x` in the tower of level `k`, when it has one
    /// there; every candidate is verified by squaring.  For `x = a + b·t`:
    /// `√a` or `√(a/r)·t` when `b = 0`, else `c + d·t` with `c² = (a ±
    /// √(a² − b²r))/2`, `d = b/(2c)`.
    fn sqrt(&self, x: &[Q], k: usize) -> Option<Vec<Q>> {
        if tower_is_zero(x) {
            return Some(vec![Q::zero(); 1 << k]);
        }
        if k == 0 {
            return rational_sqrt(&x[0]).map(|r| vec![r]);
        }
        let h = 1 << (k - 1);
        let (a, b) = x.split_at(h);
        let zero = vec![Q::zero(); h];
        let mut candidates: Vec<Vec<Q>> = Vec::new();
        if tower_is_zero(b) {
            if let Some(c) = self.sqrt(a, k - 1) {
                candidates.push([c, zero.clone()].concat());
            } else if let Some(r_inv) = self.inverse(&self.radicands[k - 1], k - 1)
                && let Some(d) = self.sqrt(&self.mul(a, &r_inv, k - 1), k - 1)
            {
                candidates.push([zero.clone(), d].concat());
            }
        } else {
            let b_sq = self.mul(b, b, k - 1);
            let norm = tower_sub(
                &self.mul(a, a, k - 1),
                &self.mul(&b_sq, &self.radicands[k - 1], k - 1),
            );
            let s = self.sqrt(&norm, k - 1)?;
            let half = tower_pad(&[Ratio::new(BigInt::from(1), BigInt::from(2))], k - 1);
            for c_sq in [tower_add(a, &s), tower_sub(a, &s)] {
                let c_sq = self.mul(&c_sq, &half, k - 1);
                let Some(c) = self.sqrt(&c_sq, k - 1) else {
                    continue;
                };
                let two_c = tower_add(&c, &c);
                let Some(inv) = self.inverse(&two_c, k - 1) else {
                    continue;
                };
                let d = self.mul(b, &inv, k - 1);
                candidates.push([c, d].concat());
            }
        }
        candidates.into_iter().find(|c| self.mul(c, c, k) == x)
    }

    /// The arena constant of the element `x` of level `k`.
    fn to_expr(&self, arena: &mut Arena, x: &[Q], k: usize) -> ExprId {
        let mut terms: smallvec::SmallVec<[ExprId; 8]> = smallvec::SmallVec::new();
        for (m, c) in x.iter().enumerate().take(1 << k) {
            if c.is_zero() {
                continue;
            }
            let mut factors: smallvec::SmallVec<[ExprId; 8]> =
                smallvec::smallvec![arena.num_ratio(c.clone())];
            for (i, &root) in self.roots.iter().enumerate() {
                if m & (1 << i) != 0 {
                    factors.push(root);
                }
            }
            terms.push(arena.mul(&factors));
        }
        arena.add(&terms)
    }
}

/// Is the constant `d` (rationals, `φ`, `+ − ×`, integer powers and square
/// roots of positive values) exactly 0 by [`SqrtTower`] arithmetic?  `false`
/// means not shown (anything else in `d`, a negative or undecided radicand,
/// more than [`SQRT_TOWER_MAX_GENERATORS`] independent radicals, more than
/// 400 nodes, a zero divisor).
fn sqrt_tower_is_zero(arena: &mut Arena, d: ExprId) -> bool {
    let order = walk::post_order_ids(arena, d);
    if order.len() > 400
        || !order.iter().any(|&id| {
            matches!(arena.node(id), ExprNode::Pow(_, e)
                if arena.as_num(*e).is_some_and(|q| *q.denom() == BigInt::from(2)))
        })
    {
        return false;
    }
    let mut tower = SqrtTower {
        radicands: Vec::new(),
        roots: Vec::new(),
    };
    let mut values: FxHashMap<ExprId, Vec<Q>> = FxHashMap::default();
    // `√b` as an element, by the arena node `b`.
    let mut sqrt_of: FxHashMap<ExprId, Vec<Q>> = FxHashMap::default();
    let half = arena.rational(1, 2);
    for &id in &order {
        let node = arena.node(id).clone();
        let value = match node {
            ExprNode::Num(n) => tower_pad(&[arena.num(n).clone()], tower.level()),
            ExprNode::Neg(a) => {
                let Some(v) = values.get(&a) else {
                    return false;
                };
                tower_pad(v, tower.level())
                    .into_iter()
                    .map(|c| -c)
                    .collect()
            }
            ExprNode::Add(cs) => {
                let k = tower.level();
                let mut acc = tower_pad(&[], k);
                for c in cs.iter() {
                    let Some(v) = values.get(c) else {
                        return false;
                    };
                    acc = tower_add(&acc, &tower_pad(v, k));
                }
                acc
            }
            ExprNode::Mul(cs) => {
                let k = tower.level();
                let mut acc = tower_pad(&[Q::one()], k);
                for c in cs.iter() {
                    let Some(v) = values.get(c) else {
                        return false;
                    };
                    acc = tower.mul(&acc, &tower_pad(v, k), k);
                }
                acc
            }
            ExprNode::GoldenRatio | ExprNode::Pow(..) => {
                // φ = 1/2 + √5/2; b^(p/q) for q ∈ {1, 2}.
                let (b, p, root) = match node {
                    ExprNode::Pow(b, e) => {
                        let Some(q) = arena.as_num(e) else {
                            return false;
                        };
                        let Some(p) = q.numer().to_i64() else {
                            return false;
                        };
                        match q.denom().to_i64() {
                            Some(1) => (b, p, false),
                            Some(2) => (b, p, true),
                            _ => return false,
                        }
                    }
                    _ => (arena.int(5), 1, true),
                };
                let base = if root {
                    if let Some(r) = sqrt_of.get(&b) {
                        r.clone()
                    } else {
                        let k = tower.level();
                        let radicand = match values.get(&b) {
                            Some(v) => tower_pad(v, k),
                            None => match arena.as_num(b) {
                                Some(q) => tower_pad(std::slice::from_ref(q), k),
                                None => return false,
                            },
                        };
                        let r = if tower_is_zero(&radicand) {
                            radicand
                        } else {
                            // Tower elements are real (every radicand so far
                            // was certified positive).
                            if certified_real_sign(arena, b) != Some(std::cmp::Ordering::Greater) {
                                return false;
                            }
                            match tower.sqrt(&radicand, k) {
                                Some(s) => {
                                    let e = tower.to_expr(arena, &s, k);
                                    match certified_real_sign(arena, e) {
                                        Some(std::cmp::Ordering::Greater) => s,
                                        Some(std::cmp::Ordering::Less) => {
                                            s.into_iter().map(|c| -c).collect()
                                        }
                                        _ => return false,
                                    }
                                }
                                None if k < SQRT_TOWER_MAX_GENERATORS => {
                                    tower.radicands.push(radicand);
                                    let root_node = arena.pow(b, half);
                                    tower.roots.push(root_node);
                                    let mut t = tower_pad(&[], k + 1);
                                    t[1 << k] = Q::one();
                                    t
                                }
                                None => return false,
                            }
                        };
                        sqrt_of.insert(b, r.clone());
                        r
                    }
                } else {
                    match values.get(&b) {
                        Some(v) => v.clone(),
                        None => return false,
                    }
                };
                let k = tower.level();
                let base = tower_pad(&base, k);
                let powered = match tower.pow(&base, p, k) {
                    Some(v) => v,
                    None => return false,
                };
                if matches!(node, ExprNode::GoldenRatio) {
                    let h = tower_pad(&[Ratio::new(BigInt::from(1), BigInt::from(2))], k);
                    tower.mul(&tower_add(&powered, &tower_pad(&[Q::one()], k)), &h, k)
                } else {
                    powered
                }
            }
            _ => return false,
        };
        values.insert(id, value);
    }
    values.get(&d).is_some_and(|v| tower_is_zero(v))
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
    // A constant's sign is extracted (SymPy: `erf(-2)` → `-erf(2)`).
    if let Some(pos) = negated_constant(arena, inner) {
        let erf_pos = eval_erf(arena, pos).unwrap_or_else(|| arena.erf(pos));
        return Some(arena.neg(erf_pos));
    }

    None
}

/// `c` for a symbol-free argument `−c` with a negative rational coefficient
/// ([`as_negated`]): the odd functions without a sign rule for symbols
/// (`erf`, `asin`, `atan`, `asinh`, `atanh`) extract it from a constant, as
/// SymPy does (`atanh(-1/2)` → `-atanh(1/2)`).  The identities hold on the
/// principal branches, cuts included.
fn negated_constant(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if !walk::free_symbols(arena, inner).is_empty() {
        return None;
    }
    as_negated(arena, inner)
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

    // W(w·e^w) = w for the real w ≥ −1 the principal branch reaches.
    if let Some(w) = lambert_fixed_point(arena, inner, LambertRealBranch::Principal) {
        tracing::debug!("eval: LambertW(w*exp(w)) = w");
        return Some(w);
    }

    None
}

/// `v` (an imaginary part) with every `im(W_k(a))` that is 0 replaced by 0:
/// `W₀(a)` is real for a real constant `a ≥ −1/e`, `W₋₁(a)` for `−1/e ≤ a <
/// 0` (certified signs of `a + 1/e` and `a`).  Up to 0.45 these stayed, and
/// the realness of `W₋₁(−ln(2)/(5·⁵√2)) + ln(2)/5` was not known, so its
/// sign was not decided though its difference from 0 was.
fn zero_real_lambert_parts(arena: &mut Arena, v: ExprId) -> ExprId {
    let targets: Vec<ExprId> = walk::post_order_ids(arena, v)
        .into_iter()
        .filter(|&id| matches!(*arena.node(id), ExprNode::Im(w) if lambert_real_argument(arena, w).is_some()))
        .collect();
    let mut out = v;
    for t in targets {
        let ExprNode::Im(w) = *arena.node(t) else {
            continue;
        };
        if lambert_value_is_real(arena, w) {
            let zero = arena.zero;
            out = crate::transforms::subs::subs(arena, out, t, zero);
        }
    }
    out
}

/// `(a, branch)` for `w = W₀(a)` or `W₋₁(a)` with a symbol-free `a`.
fn lambert_real_argument(arena: &Arena, w: ExprId) -> Option<(ExprId, LambertRealBranch)> {
    match arena.node(w) {
        ExprNode::LambertW(a) => Some((*a, LambertRealBranch::Principal)),
        ExprNode::Apply(f, args) if arena.lib_fn(*f) == Some(LibFn::LambertW) => match args[..] {
            [a, k] if arena.as_num(k).is_some_and(|q| *q == -Q::one()) => {
                Some((a, LambertRealBranch::MinusOne))
            }
            _ => None,
        },
        _ => None,
    }
    .filter(|&(a, _)| walk::free_symbols(arena, a).is_empty())
}

/// Is `w = W_k(a)` real: `a` real with `a + 1/e ≥ 0` (`W₀`), and `a < 0`
/// as well for `W₋₁`, by certified signs?  (`a = −1/e` exactly is `W =
/// −1`, which `eval` writes as such.)
fn lambert_value_is_real(arena: &mut Arena, w: ExprId) -> bool {
    let Some((a, branch)) = lambert_real_argument(arena, w) else {
        return false;
    };
    if !known_real(arena, a) {
        return false;
    }
    let neg_one = arena.neg_one;
    let inv_e = arena.exp(neg_one);
    let shifted = arena.add(&[a, inv_e]);
    if certified_real_sign(arena, shifted) != Some(std::cmp::Ordering::Greater) {
        return false;
    }
    branch == LambertRealBranch::Principal
        || certified_real_sign(arena, a) == Some(std::cmp::Ordering::Less)
}

/// The two branches of Lambert's W that are real on part of the real axis:
/// `W₀` (values `≥ −1`) and `W₋₁` (values `≤ −1`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum LambertRealBranch {
    Principal,
    MinusOne,
}

/// The exact real `y` with `y·e^y = x` on `branch` (`y ≥ −1` for `W₀`, `y ≤
/// −1` for `W₋₁`, the ranges on which `y ↦ y·e^y` is injective, so that
/// `W_k(x) = y`): the visible forms of [`lambert_visible_fixed_point`], then
/// for a constant `x` a factor `exp(c)` whose cofactor is the real `c`
/// (`W(√2·e^{√2}) = √2`), or a factor `ln m` (`m` a positive rational) with
/// `y = r·ln m`, `e^y = m^r` (`r` rational, read off the value of `W_k(x)`,
/// then verified exactly): `W(√2·ln(2)/2) = ln(2)/2`, `W(−2·ln(5/2)/5) =
/// −ln(5/2)`, `W₋₁(−3√2·ln(2)/8) = −3·ln(2)/2`.  The branch is the
/// certified sign of `y + 1`: `W₋₁(−2·ln(5/2)/5)` is not `−ln(5/2) > −1`
/// and stays.
///
/// Up to 0.45 only rational `c` and integer `t`, `p` in `t·p^t·ln p` were
/// recognised: `W(√2·ln(2)/2)` stayed, a value no evaluation tells from
/// `ln(2)/2`, and the solver recognised `W_k(y·e^y) = y` for its own
/// candidates only.
fn lambert_fixed_point(arena: &mut Arena, x: ExprId, branch: LambertRealBranch) -> Option<ExprId> {
    if let Some(y) = lambert_visible_fixed_point(arena, x, branch) {
        return Some(y);
    }
    if !walk::free_symbols(arena, x).is_empty() {
        return None;
    }
    let ExprNode::Mul(factors) = arena.node(x).clone() else {
        return None;
    };
    // y·e^y with e^y written `exp(y)`.
    for &f in &factors {
        if let ExprNode::Exp(c) = *arena.node(f) {
            let rest = arena.div(x, f);
            if rest == c && known_real(arena, c) {
                return lambert_on_branch(arena, c, branch);
            }
        }
    }
    // y = r·ln m, e^y = m^r.
    for &f in &factors {
        let ExprNode::Ln(m) = *arena.node(f) else {
            continue;
        };
        let Some(ln_m) = arena
            .as_num(m)
            .filter(|q| q.is_positive() && !q.is_one())
            .and_then(|q| Some(q.to_f64()?.ln()))
            .filter(|l| l.is_finite() && *l != 0.0)
        else {
            continue;
        };
        let w = match branch {
            LambertRealBranch::Principal => arena.lambertw(x),
            LambertRealBranch::MinusOne => {
                let k = arena.neg_one;
                arena.lambertw_branch(x, k)
            }
        };
        let Ok(value) = crate::transforms::evalf::evalf_complex64(arena, w) else {
            continue;
        };
        if !value.re.is_finite() || value.im.abs() > 1e-12 * value.re.abs().max(1.0) {
            continue;
        }
        let Some(r) = small_rational_near(value.re / ln_m, LAMBERT_MAX_DENOMINATOR) else {
            continue;
        };
        let r = arena.num_ratio(r);
        let y = arena.mul(&[r, f]);
        let e_y = arena.pow(m, r);
        let product = arena.mul(&[y, e_y]);
        let exact = product == x || {
            let d = arena.sub(product, x);
            constant_zero_decision(arena, d) == Some(true)
        };
        if exact {
            return lambert_on_branch(arena, y, branch);
        }
    }
    None
}

/// Largest denominator of the rational `r` in `W_k(r·ln m·m^r) = r·ln m`
/// that [`lambert_fixed_point`] reads off a value.
const LAMBERT_MAX_DENOMINATOR: i64 = 24;

/// The rational with denominator at most `max_den` within `10⁻¹⁰`
/// (relative) of `x`, the one with the least denominator.
fn small_rational_near(x: f64, max_den: i64) -> Option<Q> {
    if !x.is_finite() || x.abs() > 1e9 {
        return None;
    }
    (1..=max_den).find_map(|q| {
        let p = (x * q as f64).round();
        ((p / q as f64 - x).abs() <= 1e-10 * x.abs().max(1.0))
            .then(|| Q::new(BigInt::from(p as i64), BigInt::from(q)))
    })
}

/// `y` when the real constant `y` lies on `branch` of `W` (`y + 1 ≥ 0` for
/// `W₀`, `≤ 0` for `W₋₁`, by its certified sign); `None` otherwise.
fn lambert_on_branch(arena: &mut Arena, y: ExprId, branch: LambertRealBranch) -> Option<ExprId> {
    let one = arena.one;
    let y_plus_one = arena.add(&[y, one]);
    let sign = if arena.is_zero_structural(y_plus_one) {
        std::cmp::Ordering::Equal
    } else {
        certified_real_sign(arena, y_plus_one)?
    };
    let on_branch = match branch {
        LambertRealBranch::Principal => sign.is_ge(),
        LambertRealBranch::MinusOne => sign.is_le(),
    };
    on_branch.then_some(y)
}

/// The exact `w` with `w·e^w = x` on `branch` (`w ≥ −1` for `W₀`, `w ≤ −1`
/// for `W₋₁`), when `x` is visibly of that form: `c·exp(c)` for a rational
/// `c` (`W₋₁(−2e⁻²) = −2`, `W₀(3e³) = 3`), or `t·p^t·ln p` for an integer
/// `p ≥ 2` and a small integer `t` (`w = t·ln p`: `W₀(−ln(2)/2) = −ln 2`,
/// `W₋₁(−ln(2)/2) = −2·ln 2`, `W₀(3·ln 3) = ln 3`).
fn lambert_visible_fixed_point(
    arena: &mut Arena,
    x: ExprId,
    branch: LambertRealBranch,
) -> Option<ExprId> {
    let on_branch = |w_plus_one: std::cmp::Ordering| match branch {
        LambertRealBranch::Principal => w_plus_one.is_ge(),
        LambertRealBranch::MinusOne => w_plus_one.is_le(),
    };
    let ExprNode::Mul(factors) = arena.node(x).clone() else {
        return None;
    };
    let [a, b] = factors[..] else {
        return None;
    };
    let q = arena.as_num(a)?.clone();
    match arena.node(b).clone() {
        // c·exp(c)
        ExprNode::Exp(c) => {
            let c = arena.as_num(c)?.clone();
            if c != q {
                return None;
            }
            let w_plus_one = (&c + Q::from_integer(BigInt::from(1))).cmp(&Q::zero());
            on_branch(w_plus_one).then(|| arena.num_ratio(c))
        }
        // t·p^t·ln p, w = t·ln p
        ExprNode::Ln(p_id) => {
            let p = arena.as_num(p_id)?.clone();
            if !p.is_integer() || p <= Q::from_integer(BigInt::from(1)) {
                return None;
            }
            let base = p.to_integer();
            let ln_p = base.to_f64()?.ln();
            // `q = t·p^t`: for `t > 0` the numerator of `q` has the bits of
            // `p^t`, for `t < 0` its denominator does; the powers stop once
            // they outgrow them.
            for negative in [false, true] {
                let limit = if negative { q.denom() } else { q.numer() }.bits() + 8;
                let mut power = BigInt::from(1);
                for t_abs in 1i32..=64 {
                    power *= &base;
                    if power.bits() > limit {
                        break;
                    }
                    let t = if negative { -t_abs } else { t_abs };
                    let candidate = if negative {
                        Q::new(BigInt::from(t), power.clone())
                    } else {
                        Q::from_integer(BigInt::from(t) * &power)
                    };
                    if candidate != q {
                        continue;
                    }
                    // `t·ln p + 1` is never 0 (`e^{−1/t}` is not an integer).
                    let w_plus_one = (f64::from(t) * ln_p + 1.0)
                        .partial_cmp(&0.0)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if on_branch(w_plus_one) {
                        let t_id = arena.int(i64::from(t));
                        let ln_id = arena.ln(p_id);
                        return Some(arena.mul(&[t_id, ln_id]));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Exact values of `lambertw(x)` and `lambertw(x, k)` (SymPy's
/// `LambertW.eval`): the principal branch, `k = 0` or no `k`, is the
/// `ExprNode::LambertW` node with its rules ([`eval_lambertw`]); every other
/// branch is `−∞` at 0 (logarithmic singularity), and `W₋₁` has
/// `W₋₁(−1/e) = −1`, `W₋₁(−π/2) = −iπ/2` and `W₋₁(w·e^w) = w` for the
/// real `w ≤ −1` of [`lambert_fixed_point`].  A branch index that is not an
/// integer is left alone.
fn eval_lambertw_call(arena: &mut Arena, args: &[ExprId]) -> Option<ExprId> {
    let x = args[0];
    let principal =
        |arena: &mut Arena| eval_lambertw(arena, x).unwrap_or_else(|| arena.lambertw(x));
    let Some(&k) = args.get(1) else {
        return Some(principal(arena));
    };
    let kq = arena.as_num(k)?.clone();
    if !kq.is_integer() {
        return None;
    }
    if kq.is_zero() {
        return Some(principal(arena));
    }
    if arena.as_num(x).is_some_and(|q| q.is_zero()) {
        tracing::debug!("eval: LambertW(0, k) = -oo for k != 0");
        return Some(arena.neg_infinity);
    }
    if kq != Q::from_integer(BigInt::from(-1)) {
        return None;
    }
    // W₋₁(−1/e) = −1.
    let e = arena.e_const;
    let neg1 = arena.neg_one;
    let neg_inv_e = arena.div(neg1, e);
    let exp_neg1 = arena.exp(neg1);
    let neg_exp_neg1 = arena.neg(exp_neg1);
    if x == neg_inv_e || x == neg_exp_neg1 {
        tracing::debug!("eval: LambertW(-1/e, -1) = -1");
        return Some(neg1);
    }
    // W₋₁(−π/2) = −iπ/2.
    let pi = arena.pi;
    let neg_half = arena.rational(-1, 2);
    let neg_pi_2 = arena.mul(&[neg_half, pi]);
    if x == neg_pi_2 {
        tracing::debug!("eval: LambertW(-pi/2, -1) = -i*pi/2");
        let i = arena.i_unit;
        return Some(arena.mul(&[neg_half, i, pi]));
    }
    lambert_fixed_point(arena, x, LambertRealBranch::MinusOne)
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
///
/// Other rationals off the poles: `Γ(a)Γ(b)/Γ(a + b)` when the three `Γ`
/// fold (half-integers, a positive integer beside any rational): `B(1,
/// 5/2) = 2/5`, `B(1/2, 1/2) = π` (SymPy folds them; before 0.47 they
/// stayed).
fn eval_beta(arena: &mut Arena, a: ExprId, b: ExprId) -> Option<ExprId> {
    let ra = arena.as_num(a)?.clone();
    let rb = arena.as_num(b)?.clone();
    if !ra.is_integer() || !ra.is_positive() || !rb.is_integer() || !rb.is_positive() {
        return beta_by_gamma(arena, &ra, &rb);
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

/// `ln(a)/ln(b)` in a product, `a` and `b` positive rationals other than 1
/// with `a = b^k` for a rational `k` (`a^q = b^p`, `q ≤ 64`): `k` — the
/// `log(a, b)` the parser writes as that quotient.  SymPy folds `log(8, 2)`
/// to 3 and `log(27, 3)` to 3; before 0.47 they stayed `ln(8)/ln(2)`.
/// `None` unless the product holds exactly one such pair.
fn eval_log_quotient(arena: &mut Arena, factors: &[ExprId]) -> Option<ExprId> {
    let positive_rational = |arena: &Arena, id: ExprId| -> Option<Q> {
        let ExprNode::Ln(a) = *arena.node(id) else {
            return None;
        };
        let r = arena.as_num(a)?;
        (r.is_positive() && !r.is_one() && r.numer().bits() <= 256 && r.denom().bits() <= 256)
            .then(|| r.clone())
    };
    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    for (i, &f) in factors.iter().enumerate() {
        if let Some(a) = positive_rational(arena, f) {
            numerators.push((i, a));
        } else if let ExprNode::Pow(base, e) = *arena.node(f)
            && arena.as_num(e).is_some_and(|q| *q == -Q::one())
            && let Some(b) = positive_rational(arena, base)
        {
            denominators.push((i, b));
        }
    }
    let ([(i, a)], [(j, b)]) = (numerators.as_slice(), denominators.as_slice()) else {
        return None;
    };
    let k = rational_log(a, b)?;
    let k = arena.num_ratio(k);
    let mut rest: smallvec::SmallVec<[ExprId; 6]> = factors
        .iter()
        .enumerate()
        .filter(|(n, _)| n != i && n != j)
        .map(|(_, &f)| f)
        .collect();
    rest.push(k);
    Some(arena.mul(&rest))
}

/// The rational `k = p/q` (`q ≤ 64`) with `a = b^k` for positive rationals
/// `a`, `b ≠ 1`, checked exactly (`a^q = b^p`).
fn rational_log(a: &Q, b: &Q) -> Option<Q> {
    use num_traits::Pow;
    let lg = |q: &Q| -> Option<f64> {
        let n = q.numer().to_f64()?.ln();
        let d = q.denom().to_f64()?.ln();
        Some(n - d)
    };
    let estimate = lg(a)? / lg(b)?;
    if !estimate.is_finite() || estimate.abs() > 4096.0 {
        return None;
    }
    for q in 1..=64u32 {
        let p = (estimate * f64::from(q)).round();
        if (estimate * f64::from(q) - p).abs() > 1e-6 || p == 0.0 {
            continue;
        }
        let p = p as i64;
        let e = u32::try_from(p.unsigned_abs()).ok()?;
        // a^q = b^p, as `an^q·bd^p = bn^p·ad^q` (`b^−|p|` swapped).
        let (bn, bd) = if p > 0 {
            (b.numer(), b.denom())
        } else {
            (b.denom(), b.numer())
        };
        let lhs: BigInt = Pow::pow(a.numer(), q) * Pow::pow(bd, e);
        let rhs: BigInt = Pow::pow(bn, e) * Pow::pow(a.denom(), q);
        if lhs == rhs {
            return Some(Q::new(BigInt::from(p), BigInt::from(q)));
        }
    }
    None
}

/// Most nodes of a constant whose floor or ceiling is evaluated.
const MAX_INTEGER_PART_NODES: usize = 60;

/// `⌊x⌋` (`ceiling`: `⌈x⌉`) of a real constant `x` that is not a rational:
/// the integer its certified decimal value lies strictly inside, or, when
/// that value is within its last digit of an integer `n`, `n` if `x − n` is
/// exactly 0 ([`exact_zero_test`]).  SymPy's `floor.eval` (`get_integer_part`)
/// folds `floor(E)` to 2 and `ceiling(50!/e)` to an integer; before 0.47
/// they stayed.  `None` when `x` is not known real, its value does not
/// settle, or the integer is beyond 1,000 digits.
///
/// The size is read off the printed exponent before any exact value is
/// built: the 0.47 development code built `digits·10^shift` first, and
/// `floor(Shi(12!))` (`Shi(12!) ≈ 5.3·10²⁰⁸⁰²⁷⁷⁴²`) raised 10 to the
/// 208,027,742nd power (the `fuzz_evalf` timeout of the release campaign).
/// A value below `10⁻⁵` in modulus (a certified nonzero value was printed)
/// is settled by its sign alone, without its `10^−k` either.
fn integer_part_of_constant(arena: &mut Arena, x: ExprId, ceiling: bool) -> Option<ExprId> {
    if arena.as_num(x).is_some()
        || !walk::free_symbols(arena, x).is_empty()
        || walk::has_unevaluated(arena, x)
        || walk::post_order_ids(arena, x).len() > MAX_INTEGER_PART_NODES
        || !known_real(arena, x)
    {
        return None;
    }
    let mut digits = 25u32;
    let (value, ulp) = loop {
        let printed = crate::transforms::evalf::evalf(arena, x, digits).ok()?;
        let parts = DecimalParts::of(&printed)?;
        let exponent = parts.leading_exponent()?;
        if exponent < -5 && parts.is_nonzero() {
            // 0 < |x| < 10⁻⁵.
            let n = match (parts.negative, ceiling) {
                (false, false) | (true, true) => 0,
                (false, true) => 1,
                (true, false) => -1,
            };
            return Some(arena.int(n));
        }
        if exponent + 20 < i64::from(digits) {
            break parts.value_and_ulp()?;
        }
        if digits > 25 {
            return None;
        }
        digits = u32::try_from(exponent + 25).ok().filter(|&d| d <= 1000)?;
    };
    let round = |q: Q| if ceiling { q.ceil() } else { q.floor() };
    let low = round(&value - &ulp);
    let high = round(&value + &ulp);
    if low == high {
        return Some(arena.num_ratio(low));
    }
    // Within a digit of an integer: that integer if the difference is 0.
    let n = value.round();
    let n_id = arena.num_ratio(n.clone());
    let diff = arena.sub(x, n_id);
    (exact_zero_test(arena, diff) == Some(true)).then(|| arena.num_ratio(n))
}

/// A decimal `evalf` printed (`-2.5e+3`, `3.14`), split into its sign,
/// digits and exponent; `None` for anything else (a complex value).
struct DecimalParts<'a> {
    negative: bool,
    int_part: &'a str,
    frac_part: &'a str,
    exp: i64,
}

impl<'a> DecimalParts<'a> {
    fn of(s: &'a str) -> Option<Self> {
        let s = s.trim();
        let (mantissa, exp) = match s.split_once('e') {
            Some((m, e)) => (m, e.trim_start_matches('+').parse::<i64>().ok()?),
            None => (s, 0),
        };
        let negative = mantissa.starts_with('-');
        let mantissa = mantissa.trim_start_matches(['-', '+']);
        let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if int_part.is_empty()
            || !int_part
                .bytes()
                .chain(frac_part.bytes())
                .all(|b| b.is_ascii_digit())
        {
            return None;
        }
        Some(DecimalParts {
            negative,
            int_part,
            frac_part,
            exp,
        })
    }

    /// Is a digit other than 0 printed?
    fn is_nonzero(&self) -> bool {
        self.int_part
            .bytes()
            .chain(self.frac_part.bytes())
            .any(|b| b != b'0')
    }

    /// The decimal exponent of the leading digit (an upper bound when the
    /// integer part is 0).
    fn leading_exponent(&self) -> Option<i64> {
        self.exp
            .checked_add(i64::try_from(self.int_part.trim_start_matches('0').len()).ok()?)?
            .checked_sub(1)
    }

    /// The exact value and a unit in the last digit.  It costs a power of
    /// ten of the size of the exponent: callers bound that first.
    fn value_and_ulp(&self) -> Option<(Q, Q)> {
        let digits: BigInt = format!("{}{}", self.int_part, self.frac_part)
            .parse()
            .ok()?;
        let shift = self
            .exp
            .checked_sub(i64::try_from(self.frac_part.len()).ok()?)?;
        let p = num_traits::pow(
            BigInt::from(10),
            usize::try_from(shift.unsigned_abs()).ok()?,
        );
        let ulp = if shift >= 0 {
            Q::from_integer(p)
        } else {
            Q::new(BigInt::one(), p)
        };
        let mut value = Q::from_integer(digits) * &ulp;
        if self.negative {
            value = -value;
        }
        Some((value, ulp))
    }
}

/// `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` for rationals `a`, `b` none of which nor
/// `a + b` is a pole of `Γ`, when each `Γ` folds to an expression free of
/// `Γ` (a positive integer one of them, or half-integers: `Γ(n + ½)` is
/// a rational times `√π`); `None` otherwise.
fn beta_by_gamma(arena: &mut Arena, ra: &Q, rb: &Q) -> Option<ExprId> {
    let pole = |q: &Q| q.is_integer() && !q.is_positive();
    let sum = ra + rb;
    if pole(ra) || pole(rb) || pole(&sum) {
        return None;
    }
    // Γ of a rational of a large numerator is a large exact number.
    let small = |q: &Q| q.numer().bits() <= 16 && q.denom().bits() <= 16;
    if !small(ra) || !small(rb) || !small(&sum) {
        return None;
    }
    let mut folded = |q: &Q| -> Option<ExprId> {
        let n = arena.num_ratio(q.clone());
        let g = arena.gamma(n);
        let g = eval(arena, g);
        let free = !walk::post_order_ids(arena, g)
            .iter()
            .any(|&id| matches!(arena.node(id), ExprNode::Gamma(_)));
        free.then_some(g)
    };
    let ga = folded(ra)?;
    let gb = folded(rb)?;
    let gs = folded(&sum)?;
    let num = arena.mul(&[ga, gb]);
    let quotient = arena.div(num, gs);
    Some(eval(arena, quotient))
}

/// `C(n, k)` for non-negative integer literals.  A negative `n` is left
/// unevaluated on purpose: the generalised `C(−3, 2) = 6` of
/// `combinatorics::binomial` is not what the symbolic node promises.
/// The value `p + q·i` of `id` when it is a rational, `i`, or a rational
/// multiple of `i` plus a rational (the canonical forms of an exact
/// Gaussian rational: `1/2`, `I`, `3*I`, `1/2 + 2*I`).
fn gaussian_rational(arena: &Arena, id: ExprId) -> Option<(Q, Q)> {
    let imag = |t: ExprId| -> Option<Q> {
        if t == arena.i_unit {
            return Some(Q::one());
        }
        match arena.node(t) {
            ExprNode::Mul(cs) if cs.len() == 2 => {
                let (c, u) = (cs[0], cs[1]);
                match (arena.as_num(c), arena.as_num(u)) {
                    (Some(q), None) if u == arena.i_unit => Some(q.clone()),
                    (None, Some(q)) if c == arena.i_unit => Some(q.clone()),
                    _ => None,
                }
            }
            _ => None,
        }
    };
    if let Some(q) = arena.as_num(id) {
        return Some((q.clone(), Q::zero()));
    }
    if let Some(q) = imag(id) {
        return Some((Q::zero(), q));
    }
    if let ExprNode::Add(cs) = arena.node(id)
        && cs.len() == 2
    {
        for (r, i) in [(cs[0], cs[1]), (cs[1], cs[0])] {
            if let (Some(re), Some(im)) = (arena.as_num(r), imag(i)) {
                return Some((re.clone(), im));
            }
        }
    }
    None
}

/// Are `a` and `b` equal, when both are exact Gaussian rationals
/// ([`gaussian_rational`])?  `Eq`/`Ne` of two numbers fold; before 0.37 only
/// two real rationals did, so `Ne(1/2, I)` — the condition the integrator
/// attaches to `∫ e^(a·x)·cos x` for `a ≠ ±i` — stayed undecided after
/// `a = 1/2` was substituted, and a `Piecewise` on it with it.
fn gaussian_equal(arena: &Arena, a: ExprId, b: ExprId) -> Option<bool> {
    let x = gaussian_rational(arena, a)?;
    let y = gaussian_rational(arena, b)?;
    Some(x == y)
}

fn eval_binomial(arena: &mut Arena, n: ExprId, k: ExprId) -> Option<ExprId> {
    let nr = arena.as_num(n)?.clone();
    let kr = arena.as_num(k)?.clone();
    if !kr.is_integer() {
        return None;
    }
    // `n(n − 1)⋯(n − k + 1)/k!` for a rational `n` and an integer `k ≥ 0`, and
    // `0` for a negative integer `k` unless `n` is a negative integer too
    // (SymPy and mpmath agree: `binomial(−1, 3) = −1`, `binomial(7/2, 2) =
    // 35/8`, `binomial(5, −1) = 0`).  Both negative integers is where the
    // conventions part — SymPy `binomial(−7, −9) = 0`, mpmath's limit (and
    // Kronenburg's extension) `28`, which `evalf` follows — so that is left
    // as it is.  Before 0.37 only `0 ≤ k`, `0 ≤ n` integers folded, and
    // `binomial(−1, 3)` was not even evaluable numerically.
    if kr.is_negative() {
        if nr.is_integer() && nr.is_negative() {
            return None;
        }
        return Some(arena.zero);
    }
    if !nr.is_integer() || nr.is_negative() {
        let k_u64: u64 = kr.to_integer().try_into().ok()?;
        let falling = exact_factorial_power(arena, &nr, k_u64, -1)?;
        let falling = arena.as_num(falling)?.clone();
        let k_digits = log10_factorial_lower(k_u64 as f64);
        if beyond_digit_guard(arena, k_digits) {
            return None;
        }
        let mut fact = BigInt::one();
        for i in 2..=k_u64 {
            fact *= BigInt::from(i);
        }
        return guarded_num(arena, falling / Ratio::from_integer(fact));
    }
    let (n_big, k_big) = (nr.to_integer(), kr.to_integer());
    if k_big > n_big {
        // 0 ≤ n < k (SymPy: `binomial(1, 2) == 0`).
        return Some(arena.zero);
    }
    // C(n, j) ≥ (n/j)^j for j = min(k, n − k): within the digit guard.
    // (An `n` beyond `u64` with a small `j` folds too: before 0.40
    // `binomial(10²⁰, 2)` stayed unevaluated.)
    let j_big = (&n_big - &k_big).min(k_big.clone());
    let j: u64 = j_big.try_into().ok()?;
    let n_f = n_big.to_f64().unwrap_or(f64::INFINITY);
    if j >= 1 && beyond_digit_guard(arena, j as f64 * (n_f / j as f64).log10()) {
        return None;
    }
    let result = match (u64::try_from(&n_big), u64::try_from(&k_big)) {
        (Ok(n_u64), Ok(k_u64)) => crate::base::combinatorics::binomial(n_u64, k_u64),
        _ => {
            // n(n − 1)⋯(n − j + 1)/j!
            let mut numer = BigInt::one();
            for i in 0..j {
                numer *= &n_big - BigInt::from(i);
            }
            numer / crate::base::combinatorics::factorial(j)
        }
    };
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

/// Exact values of the cosine integral: `Ci(∞) = 0`, `Ci(0) = zoo`.
///
/// `Ci(z) = γ + ln z + ∫₀ᶻ (cos t − 1)/t dt` has the singularity of `ln z`
/// at 0 and takes its value, `ln 0 = zoo` (SymPy: `Ci(0)` → zoo; along the
/// negative reals the value is `Ci(|x|) + iπ`).  Before 0.30 it was `−∞`,
/// the limit along the positive reals only.  `Ei(0) = li(1) = −∞` keep
/// theirs: those functions are real on both sides.
///
/// `Ci(−x) = Ci(x) + iπ` (for `x > 0`) is *not* applied automatically
/// because it changes the real/complex character of the expression.
pub(crate) fn eval_ci(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.infinity {
        return Some(arena.zero);
    }
    if x == arena.zero {
        return Some(arena.complex_infinity);
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
///
/// `li z = Ei(Log z)` on all of ℂ (mpmath's and SymPy's definition), so
/// `li(eʸ) = Ei(y)` exactly when `Log eʸ = y`, i.e. `Im y ∈ (−π, π]`
/// ([`log_exp_is_identity`]): a real `y`, or a constant inside the strip.
/// Before 0.41 it was applied to every `y`: `li(e^{4i})` became `Ei(4i)`
/// (mpmath `li(exp(4j))` = `0.352018… − 3.287491…j`, `ei(4j)` =
/// `−0.140981… + 3.328999…j`), and `li(eˣ)` became `Ei(x)` for a complex
/// symbol `x`.
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
    if let ExprNode::Exp(y) = arena.node(x).clone()
        && log_exp_is_identity(arena, y)
    {
        return Some(arena.ei(y));
    }
    None
}

/// Is `Log(eʸ) = y`, i.e. `Im y ∈ (−π, π]`?  `true` for a `y` real by its
/// assumptions, and for a constant whose exact imaginary part is `0`, `π`,
/// or certified (16 digits) strictly inside `(−π, π)` with a relative
/// margin of `10⁻⁹`; `false` when not known (a complex symbol, an
/// imaginary part too close to `±π` to decide cheaply).
fn log_exp_is_identity(arena: &mut Arena, y: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if AssumptionCache::new().query(arena, y, Props::REAL) == Some(true) {
        return true;
    }
    if !walk::free_symbols(arena, y).is_empty() {
        return false;
    }
    let parts = crate::base::complex::decompose(arena, y);
    if !parts.exact {
        return false;
    }
    let im = eval(arena, parts.im);
    if im == arena.pi || arena.is_zero_structural(im) {
        return true;
    }
    crate::transforms::evalf::evalf_f64(arena, im)
        .is_ok_and(|v| v.is_finite() && v.abs() < std::f64::consts::PI * (1.0 - 1e-9))
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
    if n == arena.nan || x == arena.nan {
        return Some(arena.nan);
    }
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
/// `i − j` canonicalises to a non-zero number; `nan` with a `nan` index (as
/// every function of `nan`; before 0.41 `δ(nan, nan)` was `1`, a value for
/// an undefined comparison — SymPy 1.14 leaves `KroneckerDelta(nan, nan)`).
pub(crate) fn eval_kronecker_delta(arena: &mut Arena, i: ExprId, j: ExprId) -> Option<ExprId> {
    if i == arena.nan || j == arena.nan {
        return Some(arena.nan);
    }
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
    // A negative odd `n = −(2m + 1)`: `n!! = (n + 2)!!/(n + 2)`, so
    // `(−1)^m/(2m − 1)!!` (SymPy: `factorial2(−3) = −1`, `factorial2(−17) =
    // 1/2027025`).  A negative even `n` is a pole of the extension: left
    // alone.  Before 0.37 every `n < −1` stayed unevaluated.
    if n < -1 && n % 2 != 0 {
        let m = (-n - 1) / 2;
        if beyond_digit_guard(
            arena,
            (m as f64) * std::f64::consts::LOG10_2 + log10_factorial_lower(m as f64),
        ) {
            return None;
        }
        let mut odd = BigInt::one();
        let mut k = 2 * m - 1;
        while k > 1 {
            odd *= BigInt::from(k);
            k -= 2;
        }
        let sign = if m % 2 == 0 {
            BigInt::one()
        } else {
            -BigInt::one()
        };
        return guarded_num(arena, Ratio::new(sign, odd));
    }
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
    if !nr.is_integer() {
        return None;
    }
    // A negative order: `(x)₋ₘ = 1/((x − 1)(x − 2)⋯(x − m))`, `zoo` at a zero
    // factor (SymPy's `rf(3, −1) = 1/2`, `rf(3, −3) = zoo`).
    if nr.is_negative() {
        let m: u64 = (-nr.to_integer()).try_into().ok()?;
        let below = xr - Ratio::one();
        return reciprocal_factorial_power(arena, &below, m, -1);
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, 1)
}

/// `1/((x)(x + s)⋯(x + (m − 1)s))` for `s = step = ±1`: `zoo` when a factor
/// is 0 (a negative-order rising or falling factorial).
fn reciprocal_factorial_power(arena: &mut Arena, x: &Q, m: u64, step: i32) -> Option<ExprId> {
    let p = exact_factorial_power(arena, x, m, step)?;
    let v = arena.as_num(p)?.clone();
    if v.is_zero() {
        return Some(arena.complex_infinity);
    }
    guarded_num(arena, v.recip())
}

/// Falling factorial: x^(n) = x * (x-1) * ... * (x-n+1).
/// Works for rational x; n must be a non-negative integer.  Within the
/// digit guard (see [`beyond_digit_guard`]).
fn eval_falling_factorial(arena: &mut Arena, x_id: ExprId, n_id: ExprId) -> Option<ExprId> {
    let xr = arena.as_num(x_id)?.clone();
    let nr = arena.as_num(n_id)?;
    if !nr.is_integer() {
        return None;
    }
    // A negative order: `x^(−m) = 1/((x + 1)(x + 2)⋯(x + m))`, `zoo` at a zero
    // factor (SymPy's `ff(3, −1) = 1/4`, `ff(−1, −1) = zoo`).
    if nr.is_negative() {
        let m: u64 = (-nr.to_integer()).try_into().ok()?;
        let above = xr + Ratio::one();
        return reciprocal_factorial_power(arena, &above, m, 1);
    }
    let n: u64 = nr.to_integer().try_into().ok()?;
    exact_factorial_power(arena, &xr, n, -1)
}

/// An integer argument that fits in an `i64`.
fn int_arg(arena: &Arena, id: ExprId) -> Option<i64> {
    let r = arena.as_num(id)?;
    if !r.is_integer() {
        return None;
    }
    r.to_integer().try_into().ok()
}

/// Fibonacci number `F(n)` for any integer `n` (`F(−n) = (−1)^{n+1}·F(n)`,
/// SymPy: `fibonacci(−5) = 5`; before 0.40 a negative index stayed
/// unevaluated), by fast doubling ([`crate::base::combinatorics::fibonacci`]).
/// Within the digit guard: `|Fₙ| ≥ φ^(|n|−2)`.
fn eval_fibonacci(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n.unsigned_abs() as f64 - 2.0) * LOG10_PHI) {
        return None;
    }
    guarded_int(arena, crate::base::combinatorics::fibonacci(i128::from(n)))
}

/// Lucas number `L(n)` for any integer `n` (`L(−n) = (−1)ⁿ·L(n)`, SymPy:
/// `lucas(−3) = −4`; before 0.40 a negative index stayed unevaluated).
/// Within the digit guard: `|Lₙ| ≥ φ^(|n|−1)`.
fn eval_lucas(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let n = int_arg(arena, inner)?;
    if beyond_digit_guard(arena, (n.unsigned_abs() as f64 - 1.0) * LOG10_PHI) {
        return None;
    }
    guarded_int(arena, crate::base::combinatorics::lucas(i128::from(n)))
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
/// The numbers are those of the asymptotic series of `evalf`, one cached
/// implementation (`evalf::bernoulli`; before 0.30 this file had its own
/// copy of the recurrence).
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
    let m = usize::try_from(n / 2).ok()?;
    let r = crate::transforms::evalf::bernoulli::even(m);
    let digits = r.numer().to_string().len() + r.denom().to_string().len();
    (digits <= arena.config.max_result_digits).then_some(r)
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
/// where a gcd per term made `Hₙ` quadratic in its digits.  A negative
/// integer is a pole of `H(z) = ψ(z + 1) + γ`: `zoo` (SymPy:
/// `harmonic(-1)` → zoo; before 0.31 it stayed unevaluated).
fn eval_harmonic(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if arena
        .as_num(inner)
        .is_some_and(|r| r.is_integer() && r.is_negative())
    {
        return Some(arena.complex_infinity);
    }
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
///
/// Off the non-negative integers `C(x) = Γ(2x + 1)/(Γ(x + 1)·Γ(x + 2))`,
/// whose limits along the integers SymPy takes: `C(−1) = −1/2`, `C(n) = 0`
/// for `n ≤ −2`, and a pole (`zoo`) at the negative half-integers, where
/// only the numerator meets one.  (Before 0.40 all of these stayed
/// unevaluated.)
fn eval_catalan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if let Some(r) = arena.as_num(inner)
        && r.is_negative()
    {
        if r.is_integer() {
            return Some(if r.to_integer() == BigInt::from(-1) {
                arena.rational(-1, 2)
            } else {
                arena.zero
            });
        }
        if *r.denom() == BigInt::from(2) {
            return Some(arena.complex_infinity);
        }
        return None;
    }
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
///
/// `E₋₁ = π/2`: SymPy extends `Eₛ = 2^(s+1)·(ζ(−s, 1/2) − 2^(s+1)·ζ(−s, 3/4))`
/// off the non-negative integers, and at `s = −1` the poles of the two
/// Hurwitz zetas cancel to `ψ(3/4) − ψ(1/2) + ln 2 = π/2` (SymPy:
/// `euler(−1) = pi/2`; before 0.40 it stayed unevaluated).
fn eval_euler_number(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if arena
        .as_num(inner)
        .is_some_and(|r| r.is_integer() && r.to_integer() == BigInt::from(-1))
    {
        let half = arena.rational(1, 2);
        let pi = arena.pi;
        return Some(arena.mul(&[half, pi]));
    }
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
    // The secant numbers (Brent–Harvey) behind `ntheory::euler_number`,
    // bounded here by the digit guard alone; before 0.40 a recurrence over
    // binomials, products of big numbers: 0.47 s for `E₁₈₀₀`.
    let result = crate::base::combinatorics::euler_even(usize::try_from(n / 2).ok()?);
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

/// Which function [`first_quadrant`] reduces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigKind {
    Sin,
    Cos,
    Tan,
}

/// `sin(qπ)`, `cos(qπ)`, `tan(qπ)` at a rational `q ≥ 0` without a value in
/// radicals, reduced to an angle in `[0, π/2]` by the periods and
/// reflections, as SymPy's `sin.eval` and `cos.eval` reduce them
/// (`sin(104π/105)` → `sin(π/105)`, `cos(106π/105)` → `−cos(π/105)`,
/// `tan(10π/7)` → `tan(3π/7)`); `None` when the angle is already there.
/// Up to 0.46 they stayed, and `sin(104π/105) − sin(π/105)` was not 0.
fn first_quadrant(arena: &mut Arena, q: &Q, kind: TrigKind) -> Option<ExprId> {
    let one = Q::one();
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let period = if kind == TrigKind::Tan {
        one.clone()
    } else {
        Q::from_integer(BigInt::from(2))
    };
    let mut x = mod_positive(q, &period);
    let mut negative = false;
    match kind {
        TrigKind::Sin => {
            if x > one {
                x -= &one;
                negative = true;
            }
            if x > half {
                x = &one - &x;
            }
        }
        TrigKind::Cos => {
            if x > one {
                x -= &one;
                negative = true;
            }
            if x > half {
                x = &one - &x;
                negative = !negative;
            }
        }
        TrigKind::Tan => {
            if x > half {
                x = &one - &x;
                negative = true;
            }
        }
    }
    if x == *q {
        return None;
    }
    let coeff = arena.num_ratio(x);
    let angle = arena.mul(&[coeff, arena.pi]);
    let value = match kind {
        TrigKind::Sin => eval_sin(arena, angle).unwrap_or_else(|| arena.sin(angle)),
        TrigKind::Cos => eval_cos(arena, angle).unwrap_or_else(|| arena.cos(angle)),
        TrigKind::Tan => eval_tan(arena, angle).unwrap_or_else(|| arena.tan(angle)),
    };
    Some(if negative { arena.neg(value) } else { value })
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
    // Every value below is at a denominator dividing 4 or 6, which the
    // reductions keep (see `denominator_divides`); the others SymPy
    // evaluates are in radicals ([`sin_pi_radical`]).
    if !is_classical_denominator(&coeff) {
        return sin_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Sin));
    }

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
    if !is_classical_denominator(&coeff) {
        return cos_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Cos));
    }

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
    // tan(i·x) = i·tanh(x), the bridge `sin` and `cos` take (SymPy:
    // `tan(I*pi)` → `I*tanh(pi)`); up to 0.46 `tan(i·x)` stayed.
    if let Some(real_part) = as_pure_imaginary(arena, inner) {
        let t = eval_tanh(arena, real_part).unwrap_or_else(|| arena.tanh(real_part));
        return Some(arena.mul(&[arena.i_unit, t]));
    }

    let coeff = as_pi_multiple(arena, inner)?;
    if !is_classical_denominator(&coeff) {
        return tan_pi_radical(arena, &coeff)
            .or_else(|| first_quadrant(arena, &coeff, TrigKind::Tan));
    }

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

// ── sin, cos, tan and cot at rational multiples of π, in radicals ───────────
//
// SymPy 1.14's `cos.eval` (and `sin.eval`, which reduces to it, and so
// `sec`/`csc`) gives `cos(pπ/n)` in square roots for exactly the
// denominators `n` of [`COS_PI_DENOMINATORS`]: the classical ones (`n | 4`,
// `n | 6`), `n = 5` (Chebyshev polynomials of `cos(π/5) = (1 + √5)/4`),
// `n = 8, 10` (half angles), and `n = 12, 15, 20, 24, 30, 40, 60, 120`
// (`cos(x − y)` for the pairs of its `_table2`, `pπ/12 = pπ/3 − pπ/4`, …).
// `tan.eval` and `cot.eval` add twice those through the half-angle formula
// (`n = 16, 48, 80, 240`).  Other denominators (`cos(π/7)`, `cos(π/16)`)
// stay, as there; `.rewrite(sqrt)` (Fermat primes, `n = 17`, `257`, …) is
// not automatic in SymPy and not done here.  The values are built by
// SymPy's formulas through the canonical arithmetic, so they come out in
// its normal forms (`cos(π/15)` = `−1/8 + √5/8 + √3·√(√5/8 + 5/8)/2`),
// except for the square roots of differences ([`sqrt_of_surd`]: `sin(π/5)`
// is `√5/(4·√(√5/8 + 5/8))`, SymPy's `sqrt(5/8 - sqrt(5)/8)`).
// Before 0.43 only the classical denominators were evaluated, and an
// equality like `sin(π/12) = (√6 − √2)/4` was left undecided.

/// The denominators `n` at which SymPy's `cos.eval`/`sin.eval` evaluate
/// `cos(pπ/n)`, `sin(pπ/n)` (`gcd(p, n) = 1`).
const COS_PI_DENOMINATORS: [i64; 16] = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120];

/// The denominators at which SymPy's `tan.eval`/`cot.eval` evaluate: those
/// of [`COS_PI_DENOMINATORS`] and, by `tan(x/2) = (1 − cos x)/sin x`, twice
/// them.
const TAN_PI_DENOMINATORS: [i64; 20] = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240,
];

/// Is the denominator of `q` one of the classical `1, 2, 3, 4, 6`, whose
/// values the explicit tables of [`eval_sin`], [`eval_cos`] and [`eval_tan`]
/// give?
fn is_classical_denominator(q: &Q) -> bool {
    denominator_divides(q, 4) || denominator_divides(q, 6)
}

/// The denominator of `q` when it is one of `denominators`.
fn listed_denominator(q: &Q, denominators: &[i64]) -> Option<i64> {
    let n = q.denom().to_i64()?;
    denominators.contains(&n).then_some(n)
}

/// `q mod m` (`m > 0`) as a multiple of `1/n`, `n` the denominator of `q`.
fn numerator_mod(q: &Q, m: i64) -> Option<i64> {
    use num_integer::Integer;
    q.numer().mod_floor(&(q.denom() * BigInt::from(m))).to_i64()
}

/// `cos(q·π)` in radicals at the denominators of [`COS_PI_DENOMINATORS`]
/// (SymPy's `cos.eval`); `None` at any other.
fn cos_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &COS_PI_DENOMINATORS)?;
    let p = numerator_mod(q, 2)?;
    Some(cos_pi_reduced(arena, p, n))
}

/// `sin(q·π) = cos((q − 1/2)·π)`: the denominator of `q − 1/2` is in
/// [`COS_PI_DENOMINATORS`] exactly when that of `q` is (`n` and `2n` for an
/// odd `n` of the list, `n` and `n/2` for an even one).
fn sin_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    listed_denominator(q, &COS_PI_DENOMINATORS)?;
    let shifted = q - Ratio::new(BigInt::from(1), BigInt::from(2));
    cos_pi_radical(arena, &shifted)
}

/// `cos(pπ/n)` for `0 ≤ p < 2n`, `p/n` in lowest terms and `n` in
/// [`COS_PI_DENOMINATORS`].  The recursion is on the denominator (at most
/// two reflections, then `n` falls to a proper divisor of the table: depth
/// ≤ 12 for `n = 120`).
fn cos_pi_reduced(arena: &mut Arena, p: i64, n: i64) -> ExprId {
    // cos x = −cos(x − π) on (π, 2π), cos x = −cos(π − x) on (π/2, π].
    if p > n {
        let v = cos_pi_reduced(arena, p - n, n);
        return arena.neg(v);
    }
    if 2 * p > n {
        let v = cos_pi_reduced(arena, n - p, n);
        return arena.neg(v);
    }
    let sqrt_of = |arena: &mut Arena, k: i64| {
        let k = arena.int(k);
        let half = arena.rational(1, 2);
        arena.pow(k, half)
    };
    // 0 ≤ p/n ≤ 1/2.
    match n {
        1 => arena.one,
        2 => arena.zero,
        3 => arena.rational(1, 2),
        4 | 6 => {
            let r = sqrt_of(arena, if n == 4 { 2 } else { 3 });
            let half = arena.rational(1, 2);
            arena.mul(&[half, r])
        }
        // T_p((1 + √5)/4) expanded: (1 + √5)/4, (√5 − 1)/4.
        5 => {
            let r5 = sqrt_of(arena, 5);
            let quarter = arena.rational(1, 4);
            let a = arena.rational(if p == 1 { 1 } else { -1 }, 4);
            let b = arena.mul(&[quarter, r5]);
            arena.add(&[a, b])
        }
        // √((1 + cos 2x)/2), positive on [0, π/2]: √(1/2 ± √2/4) at π/8,
        // 3π/8 and √(5/8 ± √5/8) at π/10, 3π/10 (see `sqrt_of_surd`).
        8 | 10 => {
            let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
            let (a, b, c) = if n == 8 {
                (r(1, 2), r(1, 4), 2)
            } else {
                (r(5, 8), r(1, 8), 5)
            };
            // cos 2x = cos(pπ/(n/2)) is positive exactly when 4p < n.
            let b = if 4 * p < n { b } else { -b };
            sqrt_of_surd(arena, a, b, c)
        }
        // cos(x − y) = cos x·cos y + cos(π/2 − x)·cos(π/2 − y) for
        // x = pπ/a, y = pπ/b, 1/n = 1/a − 1/b.
        _ => {
            let (a, b) = match n {
                12 => (3, 4),
                15 => (6, 10),
                20 => (4, 5),
                24 => (6, 8),
                30 => (5, 6),
                40 => (8, 10),
                60 => (20, 30),
                _ => (40, 60),
            };
            let half = Ratio::new(BigInt::from(1), BigInt::from(2));
            let x = Ratio::new(BigInt::from(p), BigInt::from(a));
            let y = Ratio::new(BigInt::from(p), BigInt::from(b));
            let cos_of = |arena: &mut Arena, r: Q| {
                // Every argument here has a listed denominator (a divisor of
                // `a` or `b`, or twice an odd one).
                cos_pi_radical(arena, &r).unwrap_or_else(|| {
                    let angle = pi_times(arena, r);
                    arena.cos(angle)
                })
            };
            let (cx, cy) = (cos_of(arena, x.clone()), cos_of(arena, y.clone()));
            let (sx, sy) = (cos_of(arena, &half - x), cos_of(arena, &half - y));
            let first = arena.mul(&[cx, cy]);
            let second = arena.mul(&[sx, sy]);
            arena.add(&[first, second])
        }
    }
}

/// `√(a + b·√c)` for rationals `a > |b|·√c`, an integer `c > 1`: as it
/// reads when `b > 0`, and as `√(a² − b²c)/√(a − b·√c)` when `b < 0`
/// (SymPy prints `sqrt(1/2 - sqrt(2)/4)`; here `√2/4/√(√2/4 + 1/2)`).  The
/// radicand of a square root then is a sum of positive terms, which the
/// assumptions see is positive — so the value is known real, `re`/`im`/
/// `|·|` of it decompose (the integrals over the roots of `x⁸ + 1` stay
/// compact), and the root is the same as its partner's (`sin(π/8)` and
/// `cos(π/8)` share `√(√2/4 + 1/2)`), which keeps the zero tests of
/// identities between them small.
fn sqrt_of_surd(arena: &mut Arena, a: Q, b: Q, c: i64) -> ExprId {
    let half = arena.rational(1, 2);
    let c_id = arena.int(c);
    let root_c = arena.pow(c_id, half);
    let positive = |arena: &mut Arena, b: Q| {
        let (a_id, b_id) = (arena.num_ratio(a.clone()), arena.num_ratio(b));
        let b_root = arena.mul(&[b_id, root_c]);
        arena.add(&[a_id, b_root])
    };
    if !b.is_negative() {
        let radicand = positive(arena, b);
        return arena.pow(radicand, half);
    }
    let norm = &a * &a - &b * &b * Ratio::from_integer(BigInt::from(c));
    let norm_id = arena.num_ratio(norm);
    let numerator = arena.pow(norm_id, half);
    let conjugate = positive(arena, -b);
    let minus_half = arena.rational(-1, 2);
    let denominator = arena.pow(conjugate, minus_half);
    arena.mul(&[numerator, denominator])
}

/// `tan(q·π)` in radicals at the denominators of [`TAN_PI_DENOMINATORS`]
/// (SymPy's `tan.eval`); `None` at any other.  The classical denominators
/// are [`eval_tan`]'s.
fn tan_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &TAN_PI_DENOMINATORS)?;
    let p = numerator_mod(q, 1)?;
    if p == 0 {
        return Some(arena.zero);
    }
    if is_classical_denominator(q) {
        let angle = pi_times(arena, q.clone());
        return eval_tan(arena, angle);
    }
    let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
    match n {
        // SymPy's `table10`: √(1 ∓ 2√5/5), √(5 ∓ 2√5) at π/10, π/5, 3π/10,
        // 2π/5 (the differences as in [`sqrt_of_surd`]).
        5 | 10 => {
            let k = 10 * p / n;
            let (k, sign) = if k > 5 { (10 - k, -1) } else { (k, 1) };
            let (a, b) = match k {
                1 => (r(1, 1), r(-2, 5)),
                2 => (r(5, 1), r(-2, 1)),
                3 => (r(1, 1), r(2, 5)),
                _ => (r(5, 1), r(2, 1)),
            };
            let v = sqrt_of_surd(arena, a, b, 5);
            Some(if sign < 0 { arena.neg(v) } else { v })
        }
        // A rational tangent of a non-classical angle does not exist
        // (Niven), so 15 is the only odd denominator left: tan(x − y) for
        // x = pπ/6, y = pπ/10.
        15 => {
            let ta = tan_pi_radical(arena, &r(p, 6))?;
            let tb = tan_pi_radical(arena, &r(p, 10))?;
            let num = arena.sub(ta, tb);
            let prod = arena.mul(&[ta, tb]);
            let den = arena.add(&[arena.one, prod]);
            Some(arena.div(num, den))
        }
        // tan(x/2) = 1/sin x − cos x/sin x, x = 2pπ/n.
        _ => {
            let x = r(2 * p, n);
            let c = cos_pi_radical(arena, &x)?;
            let s = sin_pi_radical(arena, &x)?;
            if arena.is_zero_structural(s) {
                return Some(arena.complex_infinity);
            }
            let inv_s = arena.pow(s, arena.neg_one);
            let c_over_s = arena.mul(&[c, inv_s]);
            Some(arena.sub(inv_s, c_over_s))
        }
    }
}

/// `cot(q·π)` in radicals at the non-classical denominators of
/// [`TAN_PI_DENOMINATORS`] (SymPy's `cot.eval`): `cot` is the quotient
/// `cos/sin` here, which [`eval_cot_quotients`] folds whole, so that
/// `cot(π/8)` is SymPy's `1 + √2` (not `√(√2/4 + 1/2)/√(1/2 − √2/4)`) and
/// `cot(π/16)` is evaluated as `tan(π/16)` is.
fn cot_pi_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    let n = listed_denominator(q, &TAN_PI_DENOMINATORS)?;
    if is_classical_denominator(q) {
        return None;
    }
    let p = numerator_mod(q, 1)?;
    let r = |p: i64, n: i64| Ratio::new(BigInt::from(p), BigInt::from(n));
    match n {
        // cot x = tan(π/2 − x).
        5 | 10 => tan_pi_radical(arena, &(r(1, 2) - r(p, n))),
        // cot(x − y) = (1 + cot x·cot y)/(cot y − cot x), x = pπ/6, y = pπ/10.
        15 => {
            let ca = cot_pi_classical_or_radical(arena, &r(p, 6))?;
            let cb = cot_pi_classical_or_radical(arena, &r(p, 10))?;
            let prod = arena.mul(&[ca, cb]);
            let num = arena.add(&[arena.one, prod]);
            let den = arena.sub(cb, ca);
            Some(arena.div(num, den))
        }
        // cot(x/2) = 1/sin x + cos x/sin x, x = 2pπ/n.
        _ => {
            let x = r(2 * p, n);
            let c = cos_pi_radical(arena, &x)?;
            let s = sin_pi_radical(arena, &x)?;
            if arena.is_zero_structural(s) {
                return Some(arena.complex_infinity);
            }
            let inv_s = arena.pow(s, arena.neg_one);
            let c_over_s = arena.mul(&[c, inv_s]);
            Some(arena.add(&[inv_s, c_over_s]))
        }
    }
}

/// `cot(q·π)` at any denominator of [`TAN_PI_DENOMINATORS`]: `1/tan` at the
/// classical ones (`cot(π/6) = √3`, `cot(π/3) = √3/3`).
fn cot_pi_classical_or_radical(arena: &mut Arena, q: &Q) -> Option<ExprId> {
    if is_classical_denominator(q) {
        let angle = pi_times(arena, q.clone());
        let t = eval_tan(arena, angle)?;
        if arena.is_zero_structural(t) {
            return Some(arena.complex_infinity);
        }
        return Some(arena.pow(t, arena.neg_one));
    }
    cot_pi_radical(arena, q)
}

/// `cos(u)^k·sin(u)^(−k)` → `cot(u)^k` and `cosh(u)^k·sinh(u)^(−k)` →
/// `coth(u)^k = (−i·cot(u/i))^k` (`k ≥ 1`) among the factors `children` of a
/// product, for `u = qπ` (`u = i·qπ`) with `q` at a non-classical
/// denominator of [`TAN_PI_DENOMINATORS`]: `cot` and `coth` are these
/// quotients, and the quotient of the separately evaluated `cos` and `sin`
/// is not SymPy's `cot.eval` form (or not evaluated at all: `cos(π/16)` and
/// `sin(π/16)` are not, `cot(π/16)` is).  `None` when no pair folds.  The
/// arguments are compared after evaluation (`cache`).
fn eval_cot_quotients(
    arena: &mut Arena,
    children: &[ExprId],
    cache: &FxHashMap<ExprId, ExprId>,
) -> Option<ExprId> {
    // (argument, hyperbolic, integer exponent) of a factor `f(u)^k`, `f`
    // one of cos, sin, cosh, sinh.
    let factor = |arena: &Arena, c: ExprId| -> Option<(ExprId, bool, bool, i64)> {
        let (b, k) = match *arena.node(c) {
            ExprNode::Pow(b, e) => {
                let e = arena.as_num(e)?;
                if !e.is_integer() {
                    return None;
                }
                (b, e.to_integer().to_i64()?)
            }
            _ => (c, 1),
        };
        let (u, hyperbolic, cosine) = match *arena.node(b) {
            ExprNode::Cos(u) => (u, false, true),
            ExprNode::Sin(u) => (u, false, false),
            ExprNode::Cosh(u) => (u, true, true),
            ExprNode::Sinh(u) => (u, true, false),
            _ => return None,
        };
        Some((cache.get(&u).copied().unwrap_or(u), hyperbolic, cosine, k))
    };
    if children.len() < 2 || !children.iter().any(|&c| factor(arena, c).is_some()) {
        return None;
    }
    let mut used = vec![false; children.len()];
    let mut values: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
    for i in 0..children.len() {
        let Some((u, hyperbolic, true, k)) = factor(arena, children[i]) else {
            continue;
        };
        if k < 1 || used[i] {
            continue;
        }
        let partner = (0..children.len()).find(|&j| {
            !used[j] && j != i && factor(arena, children[j]) == Some((u, hyperbolic, false, -k))
        });
        let Some(j) = partner else { continue };
        let q = if hyperbolic {
            as_imaginary_pi_multiple(arena, u)
        } else {
            as_pi_multiple(arena, u)
        };
        let Some(cot) = q.and_then(|q| cot_pi_radical(arena, &q)) else {
            continue;
        };
        let value = if hyperbolic {
            let minus_i = arena.neg(arena.i_unit);
            arena.mul(&[minus_i, cot])
        } else {
            cot
        };
        let kk = arena.int(k);
        values.push(arena.pow(value, kk));
        used[i] = true;
        used[j] = true;
    }
    if values.is_empty() {
        return None;
    }
    for (c, u) in children.iter().zip(&used) {
        if !u {
            values.push(cache.get(c).copied().unwrap_or(*c));
        }
    }
    Some(arena.mul(&values))
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
    // is defined exactly there), as SymPy folds it.  A symbolic `w` is
    // generic (`x/x = 1` likewise); a constant one must be known non-zero:
    // `ln 0 = zoo` and `exp(zoo)` has no value, and before 0.30
    // `exp(ln((1 + √2)² − 3 − 2√2))` (a zero the canonical form does not
    // see) evaluated to 0.
    if let ExprNode::Ln(w) = *arena.node(inner)
        && (!walk::free_symbols(arena, w).is_empty() || constant_is_nonzero(arena, w))
    {
        return Some(w);
    }
    // exp(k·ln z) = z^k, the principal power by definition.
    if let Some(power) = log_multiple_power(arena, inner) {
        return Some(power);
    }

    // Euler's formula: exp(i*k*π) = cos(kπ) + i*sin(kπ)
    // Detect if inner is i * (something that's a π-multiple)
    // Only at the classical denominators (`e^{iπ/4} = (1 + i)·√2/2`):
    // SymPy keeps `exp(I*pi/5)`, and the radical values of `cos`/`sin` at
    // the other denominators would only make it longer.
    if let Some(pi_coeff) = as_imaginary_pi_multiple(arena, inner)
        && is_classical_denominator(&pi_coeff)
    {
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

    // exp(i·q·π) with `q` reduced to (−1, 1] (period 2πi), alone or beside
    // other terms, as SymPy's `exp.eval` does (`exp(23πi/9)` →
    // `exp(5πi/9)`, `exp(2 − 17πi/5)` → `exp(2 + 3πi/5)`); up to 0.46 they
    // stayed, and `exp(37πi/8) − exp(−11πi/8)` was not 0.
    if let Some(reduced) = principal_imaginary_pi(arena, inner) {
        return Some(eval_exp(arena, reduced).unwrap_or_else(|| arena.exp(reduced)));
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
                && denominator_divides(&pi_coeff, 2)
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
        // exp(Σ kⱼ·ln zⱼ + w) = ∏ zⱼ^kⱼ · exp(w) for real constants `kⱼ`, as SymPy's
        // `exp.eval` splits a sum (`exp(x + 2*log(3))` → `9*exp(x)`,
        // `exp(log(-8)/3)` → `2*exp(I*pi/3)`): `exp(a + b) = exp(a)·exp(b)` on
        // all of ℂ.  A symbolic `zⱼ` is generic ([`log_multiple_power`]:
        // `exp(x + 2·ln y) = y²·eˣ`, as in SymPy).
        let mut powers: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
        let mut rest: smallvec::SmallVec<[ExprId; 6]> = smallvec::SmallVec::new();
        for &t in &terms {
            match log_multiple_power(arena, t) {
                Some(p) => powers.push(p),
                None => rest.push(t),
            }
        }
        if !powers.is_empty() {
            let rest = arena.add(&rest);
            let rest_exp = eval_exp(arena, rest).unwrap_or_else(|| arena.exp(rest));
            powers.push(rest_exp);
            return Some(arena.mul(&powers));
        }
    }

    None
}

/// `inner` with its term `i·q·π` (`q` rational) moved into `(−1, 1]` by a
/// multiple of 2, when it is outside; `None` otherwise.
fn principal_imaginary_pi(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    let terms: smallvec::SmallVec<[ExprId; 6]> = match arena.node(inner) {
        ExprNode::Add(terms) => terms.clone(),
        _ => smallvec::smallvec![inner],
    };
    let two = Q::from_integer(BigInt::from(2));
    for (k, &t) in terms.iter().enumerate() {
        let Some(q) = as_imaginary_pi_multiple(arena, t) else {
            continue;
        };
        // (−1, 1]: q − 2·⌈(q − 1)/2⌉.
        let shift = ((&q - Q::one()) / &two).ceil() * &two;
        if shift.is_zero() {
            return None;
        }
        let r = &q - &shift;
        let r = arena.num_ratio(r);
        let reduced = arena.mul(&[r, arena.i_unit, arena.pi]);
        let rest: smallvec::SmallVec<[ExprId; 6]> = terms
            .iter()
            .enumerate()
            .map(|(j, &c)| if j == k { reduced } else { c })
            .collect();
        return Some(if rest.len() == 1 {
            rest[0]
        } else {
            arena.add(&rest)
        });
    }
    None
}

/// `z^k` for a term `t = k·ln z` of an exponent, `z ≠ 0` (a constant, or
/// nonzero by the assumptions) and `k` a finite real constant (a product of
/// real constant factors, as SymPy's `exp.eval` asks of the cofactor of its
/// one `log`): `exp(k·Log z) = z^k` is the definition of the principal
/// power.  So `exp(2·ln 10) = 100`, `exp(−ln(2)·3/2) = √2/4`, `exp(2·ln(√2))
/// = 2`, `exp(π·ln 2) = 2^π`, `exp(ln(2 + i)/2) = √(2 + i)`, `exp(2·ln x) =
/// x²`; a symbolic or non-real `k` stays (`exp(x·ln 2)`, `exp(i·ln 2)`, as
/// in SymPy).  Up to 0.43 all stayed, and in 0.44 a symbolic `z` stayed
/// (the Gruntz rewrite then mishandled `exp(−ln(1/ω))` folding to `ω`,
/// fixed in 0.44).  A symbolic `z` is generic, as for `exp(ln z) = z` in
/// [`eval_exp`] (and `x/x = 1`): up to 0.46 `exp(2·ln x)` folded to `x²`
/// only for an `x` known nonzero while `exp(ln x)` folded to `x` for every
/// `x` (SymPy folds both); a constant `z` must still be known nonzero (a
/// zero the canonical form does not see).
fn log_multiple_power(arena: &mut Arena, t: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let children = match arena.node(t).clone() {
        ExprNode::Mul(children) => children,
        // A bare `ln z` in a sum (`exp(ln 2 + ln 3) = 6`).
        ExprNode::Ln(z) => return constant_is_nonzero(arena, z).then_some(z),
        _ => return None,
    };
    let mut log_arg = None;
    let mut coeffs: smallvec::SmallVec<[ExprId; 4]> = smallvec::SmallVec::new();
    for &c in &children {
        match *arena.node(c) {
            ExprNode::Ln(z) if log_arg.is_none() => log_arg = Some(z),
            ExprNode::Ln(_) => return None,
            _ => coeffs.push(c),
        }
    }
    let z = log_arg?;
    if walk::free_symbols(arena, z).is_empty() && !constant_is_nonzero(arena, z) {
        return None;
    }
    let k = arena.mul(&coeffs);
    if !walk::free_symbols(arena, k).is_empty()
        || AssumptionCache::new().query(arena, k, Props::FINITE) != Some(true)
        || !known_real(arena, k)
    {
        return None;
    }
    Some(arena.pow(z, k))
}

/// Is `w` provably non-zero: by the assumption engine, or, for a constant
/// (symbol-free) `w`, by a certified 16-digit value whose error ball
/// excludes 0?
fn constant_is_nonzero(arena: &Arena, w: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    if AssumptionCache::new().query(arena, w, Props::NONZERO) == Some(true) {
        return true;
    }
    walk::free_symbols(arena, w).is_empty()
        && matches!(
        crate::transforms::evalf::evalf_settled(
            arena,
            w,
            16,
            crate::transforms::evalf::ZeroSearch::Cap
        ),
            Ok((ref z, crate::transforms::evalf::Settled::Certified))
                if !(z.0.is_zero() && z.1.is_zero())
        )
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

    if let ExprNode::Exp(y) = *arena.node(inner)
        && let Some(v) = ln_of_exp(arena, y)
    {
        return Some(v);
    }

    // ln w = ln(−w) + iπ for any negative real constant `w`, as for the
    // rationals above (SymPy's `log.eval`: `log(-E)` → 1 + I*pi, `log(1 -
    // sqrt(2))` → `log(-1 + sqrt(2)) + I*pi`).  Up to 0.43 they stayed.
    if arena.as_num(inner).is_none() && constant_is_negative(arena, inner) {
        let w = arena.neg(inner);
        let ln_w = eval_ln(arena, w).unwrap_or_else(|| arena.ln(w));
        let i_pi = arena.mul(&[arena.i_unit, arena.pi]);
        return Some(arena.add(&[ln_w, i_pi]));
    }

    ln_of_special_complex(arena, inner)
}

/// Is the finite constant `w` negative: by the assumptions, or by a
/// certified 16-digit value that is real with a negative real part?
fn constant_is_negative(arena: &mut Arena, w: ExprId) -> bool {
    use crate::base::assumptions::{AssumptionCache, Props};
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if !walk::free_symbols(arena, w).is_empty() {
        return false;
    }
    constant_fact(arena, w, FACT_NEGATIVE, |arena| {
        let mut cache = AssumptionCache::new();
        if cache.query(arena, w, Props::FINITE) == Some(false) {
            return false;
        }
        match cache.query(arena, w, Props::NEGATIVE) {
            Some(negative) => negative && cache.query(arena, w, Props::FINITE) == Some(true),
            // The value first: realness may take a minimal polynomial.
            None => {
                matches!(
                    evalf_settled(arena, w, 16, ZeroSearch::Cap),
                    Ok((z, Settled::Certified)) if z.1.is_zero() && z.0.is_negative()
                ) && known_real(arena, w)
            }
        }
    })
}

/// `Log(e^y) = y − 2πik` for a constant `y`, with the integer `k` that puts
/// the imaginary part in `(−π, π]` (SymPy's `log.eval`: `log(exp(5*I))` →
/// `-2*I*pi + 5*I`, `log(exp(2 + 3*I))` → `2 + 3*I`): exactly for an
/// imaginary part `q·π`, else by a 16-digit value away from the odd
/// multiples of `π`.  A real `y` folds when built.  Up to 0.43 they stayed.
fn ln_of_exp(arena: &mut Arena, y: ExprId) -> Option<ExprId> {
    if !walk::free_symbols(arena, y).is_empty() {
        return None;
    }
    if log_exp_is_identity(arena, y) {
        return Some(y);
    }
    let parts = crate::base::complex::decompose(arena, y);
    if !parts.exact {
        return None;
    }
    let im = eval(arena, parts.im);
    let turns = if let Some(q) = as_pi_multiple(arena, im) {
        // q − r for the r ∈ (−1, 1] with q ≡ r (mod 2), in units of 2π.
        let two = Ratio::from_integer(BigInt::from(2));
        let shifted = mod_positive(&(q.clone() + Ratio::one()), &two);
        let r = if shifted.is_zero() {
            Ratio::one()
        } else {
            shifted - Ratio::one()
        };
        (q - r) / two
    } else {
        let v = crate::transforms::evalf::evalf_f64(arena, im).ok()?;
        let tau = 2.0 * std::f64::consts::PI;
        if !v.is_finite() || v.abs() > 1e6 {
            return None;
        }
        let k = (v / tau).round();
        if (v - k * tau).abs() >= std::f64::consts::PI * (1.0 - 1e-9) {
            return None;
        }
        Ratio::from_integer(BigInt::from(k as i64))
    };
    if turns.is_zero() {
        return Some(y);
    }
    let shift = pi_times(arena, turns * Ratio::from_integer(BigInt::from(-2)));
    let i_shift = arena.mul(&[arena.i_unit, shift]);
    Some(arena.add(&[y, i_shift]))
}

/// `Log z = ln|z| + i·Arg z` for a constant `z = a + b·i` (`b ≠ 0`, exact
/// parts built from rationals and square roots) whose argument is a
/// rational multiple of π the special-angle table knows (`tan(Arg z)` = 1,
/// `√3`, `2 − √3`, …; for `a = 0` it is `±π/2`).  SymPy folds these:
/// `log(I)` → `I*pi/2`, `log(1 + I)` → `log(sqrt(2)) + I*pi/4`, `log(1/2 +
/// sqrt(3)*I/2)` → `I*pi/3`.  Before 0.39 they stayed `ln(…)`, and `solve`
/// returned `exp(ix) = i` as `−ln(I)·I`.
fn ln_of_special_complex(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    if crate::base::complex::is_real_node(arena, z)
        || !crate::base::walk::free_symbols(arena, z).is_empty()
    {
        return None;
    }
    let parts = crate::base::complex::decompose(arena, z);
    if !parts.exact || arena.is_zero_structural(parts.im) {
        return None;
    }
    let (re, im) = (eval(arena, parts.re), eval(arena, parts.im));
    let tower =
        |arena: &Arena, v: ExprId| arena.as_num(v).is_some() || is_square_root_tower(arena, v);
    if !tower(arena, re) || !tower(arena, im) {
        return None;
    }
    let im_sign = crate::poly::algebraic::sign_checked(arena, im)?;
    // `q·π`, the angle of `|b|/|a|` in [0, π/2].
    let (re_sign, q) = if arena.is_zero_structural(re) {
        (0, Ratio::new(BigInt::from(1), BigInt::from(2)))
    } else {
        let re_sign = crate::poly::algebraic::sign_checked(arena, re)?;
        let ratio = arena.div(im, re);
        let ratio = eval(arena, ratio);
        let q = match arena.as_num(ratio) {
            // Niven: a rational tangent of a rational multiple of π is 0 or ±1.
            Some(r) if r.abs().is_one() => Ratio::new(BigInt::from(1), BigInt::from(4)),
            Some(_) => return None,
            None => special_angle_of(arena, ratio, SpecialTable::Tan)?.1,
        };
        (re_sign, q)
    };
    if im_sign == 0 || (re_sign != 0 && q.is_zero()) {
        return None;
    }
    let pi_q = if re_sign >= 0 { q } else { Ratio::one() - q };
    let arg = if im_sign > 0 { pi_q } else { -pi_q };
    let i_arg = {
        let t = pi_times(arena, arg);
        arena.mul(&[arena.i_unit, t])
    };
    // `ln|z| = ln √(a² + b²)`, the squares expanded: on the unit circle
    // `a² + b²` is 1 and the logarithm 0 (`log(cos(7π/12) + i·sin(7π/12))`
    // → `7πi/12`, as SymPy gives it; up to 0.46 `ln √((√2/4 − √6/4)² +
    // …)` stayed beside it).
    let two = arena.int(2);
    let (re2, im2) = (arena.pow(re, two), arena.pow(im, two));
    let m2 = arena.add(&[re2, im2]);
    let m2 = crate::transforms::expand::expand(arena, m2);
    let m2 = eval(arena, m2);
    let half = arena.rational(1, 2);
    let modulus = arena.pow(m2, half);
    let modulus = eval(arena, modulus);
    let ln_abs = arena.ln(modulus);
    let out = arena.add(&[ln_abs, i_arg]);
    Some(eval(arena, out))
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
    if arena.as_num(base).is_none() {
        return negative_constant_power(arena, base, exp);
    }
    let base_r = arena.as_num(base)?.clone();
    let exp_r = arena.as_num(exp)?;

    // (−r)^q = r^q·(−1)^q for a negative rational base and any non-integer
    // rational q on the principal branch (arg(−r) = π), each factor in its
    // normal form (SymPy: `(-8)**(7/3)` → `128*(-1)**(1/3)`, `(-8)**(-1/3)`
    // → `-(-1)**(2/3)/2`).  Up to 0.44 only `(−r)^(1/n)` was split, and
    // `(−8)^(7/3)` stayed.
    if !exp_r.numer().is_one() {
        if exp_r.is_integer() || !base_r.is_negative() || base_r == -Q::one() {
            return None;
        }
        let abs_id = arena.num_ratio(-base_r);
        let magnitude = arena.pow(abs_id, exp);
        let unit = arena.pow(arena.neg_one, exp);
        return Some(arena.mul(&[magnitude, unit]));
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

/// `w^q = (−w)^q·(−1)^q` for a negative real constant `w` that is not a
/// rational and a non-integer rational `q`, on the principal branch
/// (`arg w = π`):
///
/// * a product with a negative coefficient that the assumptions know
///   negative, as SymPy's `Mul._eval_power` (`sqrt(-pi)` → `I*sqrt(pi)`,
///   `(-pi)**(1/3)` → `(-1)**(1/3)*pi**(1/3)`);
/// * a constant whose realness the assumptions leave open (it has non-real
///   parts, as the real root of a cubic by Cardano's formula) and which is
///   real ([`known_real`]: its minimal polynomial) and certified negative.
///   `evalf` cannot take `√w` or `ln w` on the cut when the imaginary part
///   of `w` cancels only exactly; `(−w)^q` it can.  Up to 0.44 `√λ` of
///   such an eigenvalue was refused (`PrecisionExhausted`).
///
/// A real sum such as `1 − √2` keeps its power, as in SymPy.
///
/// A product with a rational coefficient `c`, `|c| ≠ 1`, whose rest
/// `sgn(c)·z` is not known positive gives up its magnitude first, `(c·z)^q
/// = |c|^q·(sgn(c)·z)^q` (scaling by `|c| > 0` keeps the argument):
/// `(2i)^(7/3) = −4·∛2·∛i`, as SymPy writes it (`√(2π)` keeps its
/// form).
fn negative_constant_power(arena: &mut Arena, w: ExprId, exp: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    if arena.as_num(exp)?.is_integer() || !walk::free_symbols(arena, w).is_empty() {
        return None;
    }
    let mut cache = AssumptionCache::new();
    if let ExprNode::Mul(cs) = arena.node(w)
        && let Some(c) = cs.first().and_then(|&c| arena.as_num(c)).cloned()
        && !c.abs().is_one()
    {
        let magnitude = arena.num_ratio(c.abs());
        let rest = arena.div(w, magnitude);
        if cache.query(arena, rest, Props::POSITIVE) == Some(true) {
            return None;
        }
        let outer = arena.pow(magnitude, exp);
        let inner =
            negative_constant_power(arena, rest, exp).unwrap_or_else(|| arena.pow(rest, exp));
        return Some(arena.mul(&[outer, inner]));
    }

    let negative = match arena.node(w) {
        ExprNode::Mul(cs) => {
            cs.first()
                .is_some_and(|&c| arena.as_num(c).is_some_and(Signed::is_negative))
                && cache.query(arena, w, Props::NEGATIVE) == Some(true)
                && cache.query(arena, w, Props::FINITE) == Some(true)
        }
        ExprNode::Add(_) => {
            cache.query(arena, w, Props::REAL).is_none() && constant_is_negative(arena, w)
        }
        _ => false,
    };
    if !negative {
        return None;
    }
    let minus_w = arena.neg(w);
    let magnitude = arena.pow(minus_w, exp);
    let unit = arena.pow(arena.neg_one, exp);
    Some(arena.mul(&[magnitude, unit]))
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

    // `|e^z| = e^(Re z)` on all of ℂ (SymPy's `Abs.eval` takes
    // `exp(re(arg))`): `|exp(i)| = 1`.  Before 0.35 it stayed `abs(exp(i))`,
    // and the norm of the matrix `[[exp(i)]]` printed `√(sin²1 + cos²1)`.
    // For a constant exponent whose real part is exact.
    if let ExprNode::Exp(z) = *arena.node(inner)
        && crate::base::walk::free_symbols(arena, z).is_empty()
    {
        let parts = crate::base::complex::decompose(arena, z);
        if parts.exact {
            let re = eval(arena, parts.re);
            let e = arena.exp(re);
            return Some(eval(arena, e));
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

    // `|c·t| = |c|·|t|` for a rational `c`: the canonical `Abs` takes the
    // coefficient out, and the rules above apply to `t` (before 0.41
    // `eval(|−e^π|)` stopped at `|e^π|`, which a second `eval` folded).
    let (coeff, term) = arena.as_coeff_term(inner);
    if term != inner && !coeff.is_zero() && !coeff.is_one() {
        let t = eval_abs(arena, term)?;
        return Some(arena.make_coeff_term(coeff.abs(), t));
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

/// `sign(inner)` for the evaluated argument `inner` (`original` before
/// evaluation, which the exact zero tests also try, see [`Difference`]).
fn eval_sign(arena: &mut Arena, inner: ExprId, original: ExprId) -> Option<ExprId> {
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
    // A real constant of certified sign (SymPy: `sign(1 - sqrt(2))` → -1);
    // up to 0.44 only rationals folded, and `sign(λ)` of a negative root
    // by Cardano's formula stayed though `λ < 0` was decided.
    match real_constant_sign(arena, inner) {
        Some(std::cmp::Ordering::Greater) => Some(arena.one),
        Some(_) => Some(arena.neg_one),
        // A constant that is exactly 0 without being written 0: `sign(cos(π/16)
        // − √(2 + √(2 + √2))/2)` stayed up to 0.44.  The exact tests run
        // only for a value that is 0 to the precision of the evaluation; they
        // are those of `Eq(inner, 0)` ([`constant_zero_decision`]): up to
        // 0.45 the minimal polynomial and the arctangent sums were missing
        // here, and `sign(atan(1/2) + atan(1/3) − π/4)` stayed.
        None => {
            let zero = match constant_zero_decision(arena, inner) {
                None if original != inner && walk::free_symbols(arena, original).is_empty() => {
                    exact_zero_test(arena, original)
                }
                decided => decided,
            };
            (zero == Some(true)).then_some(arena.zero)
        }
    }
}

/// The sign of the symbol-free constant `x` when it is real by
/// [`known_real`] (the assumptions, a certified exact zero imaginary part,
/// or its minimal polynomial) and its certified value settles nonzero
/// ([`certified_real_sign`]); `None` otherwise.
fn real_constant_sign(arena: &mut Arena, x: ExprId) -> Option<std::cmp::Ordering> {
    use crate::base::assumptions::{AssumptionCache, Props};
    // A certified value is finite: finiteness the assumptions do not know
    // (`W₋₁(−ln(2)/(5·⁵√2))`) does not keep the sign open (up to 0.45
    // `sign` stayed there while `eval` decided `x < 0`).
    if !walk::free_symbols(arena, x).is_empty()
        || AssumptionCache::new().query(arena, x, Props::FINITE) == Some(false)
    {
        return None;
    }
    // The value first: realness may take a minimal polynomial.
    let sign = certified_real_sign(arena, x)?;
    known_real(arena, x).then_some(sign)
}

/// What [`constant_facts`] found about a symbol-free constant, memoised in
/// [`Arena::constant_facts`] under one key per class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstantClass {
    /// Real (by [`known_real`]) with a certified positive value.
    Positive,
    /// Real with a certified negative value.
    Negative,
    /// Exactly 0 ([`exact_zero_test`]).
    Zero,
    /// A certified nonzero imaginary part.
    NonReal,
    /// Real and not 0 (an exact test), sign not decided.
    NonzeroReal,
    /// Not 0 (a certified nonzero value, or an exact test), realness not
    /// decided.
    Nonzero,
    /// Real, nothing else decided.
    Real,
    /// Nothing decided.
    Unknown,
}

impl ConstantClass {
    const ALL: [Self; 8] = [
        Self::Positive,
        Self::Negative,
        Self::Zero,
        Self::NonReal,
        Self::NonzeroReal,
        Self::Nonzero,
        Self::Real,
        Self::Unknown,
    ];

    /// The key of the class in [`Arena::constant_facts`] (after
    /// [`FACT_REAL`] and [`FACT_NEGATIVE`]).
    fn key(self) -> u8 {
        2 + self as u8
    }
}

/// The class of the symbol-free constant `x`, decided on its evaluated form
/// `v = eval(x)` as `eval` decides relations (`W(√3·e^{√3}) − √3` is `0`):
/// the certified 16-digit value of `v` first (a nonzero imaginary part, or
/// a nonzero real part and realness by [`known_real`]), the exact zero
/// tests of `Eq(v, 0)` when the value is 0 to its precision.  Memoised per
/// arena.
fn constant_class(arena: &mut Arena, x: ExprId) -> ConstantClass {
    use crate::transforms::evalf::{Settled, ZeroSearch, evalf_settled};
    if let Some(&c) = ConstantClass::ALL
        .iter()
        .find(|c| arena.constant_facts.contains_key(&(x, c.key())))
    {
        return c;
    }
    let v = eval(arena, x);
    let class = if arena.is_zero_structural(v) {
        ConstantClass::Zero
    } else {
        let value = evalf_settled(arena, v, 16, ZeroSearch::Cap);
        let (nonzero, nonreal) = match &value {
            Ok((z, Settled::Certified)) => (!(z.0.is_zero() && z.1.is_zero()), !z.1.is_zero()),
            _ => (false, false),
        };
        if nonreal {
            ConstantClass::NonReal
        } else if nonzero {
            match (known_real(arena, v), value) {
                (true, Ok((z, _))) if z.0.is_negative() => ConstantClass::Negative,
                (true, _) => ConstantClass::Positive,
                (false, _) => ConstantClass::Nonzero,
            }
        } else {
            let real = known_real(arena, v);
            let zero = exact_zero_test(arena, v)
                .or_else(|| (v != x).then(|| exact_zero_test(arena, x)).flatten());
            match (zero, real) {
                (Some(true), _) => ConstantClass::Zero,
                (Some(false), true) => ConstantClass::NonzeroReal,
                (Some(false), false) => ConstantClass::Nonzero,
                (None, true) => ConstantClass::Real,
                (None, false) => ConstantClass::Unknown,
            }
        }
    };
    arena.constant_facts.insert((x, class.key()), true);
    class
}

/// The sign, zero and realness facts `eval` decides about the symbol-free
/// constant `x` (empty for an expression with symbols or one known not to
/// be finite): what the assumption queries of the API add to the
/// structural rules of [`AssumptionCache`](crate::base::assumptions::AssumptionCache)
/// for constants, so that `cos(π/16).is_positive()` agrees with `eval` of
/// `cos(π/16) > 0` (both `True`; the query was `None` up to 0.45), and
/// `(cos(π/16) − √(2 + √(2 + √2))/2).is_zero()` with `Eq` and `sign`.
///
/// A certified 16-digit value decides the sign of a constant real by
/// [`known_real`] and shows a nonzero imaginary part; a value 0 to its
/// precision goes to the exact zero tests of `Eq(x, 0)`
/// ([`exact_zero_test`]).  The work is memoised per arena
/// ([`constant_class`], [`Arena::constant_facts`]) and nothing recurses
/// over `x`; the structural cache is left alone, so a query answers the
/// same whatever was asked before.  The facts are closed under the
/// implication rules.
pub(crate) fn constant_facts(
    arena: &mut Arena,
    x: ExprId,
) -> crate::base::assumptions::Assumptions {
    use crate::base::assumptions::{AssumptionCache, Assumptions, Props};
    let mut facts = Assumptions::default();
    if !walk::free_symbols(arena, x).is_empty()
        || AssumptionCache::new().query(arena, x, Props::FINITE) == Some(false)
    {
        return facts;
    }
    let class = constant_class(arena, x);
    match class {
        ConstantClass::Positive => facts.known_true |= Props::REAL | Props::POSITIVE,
        ConstantClass::Negative => facts.known_true |= Props::REAL | Props::NEGATIVE,
        ConstantClass::Zero => facts.known_true |= Props::ZERO,
        ConstantClass::NonReal => {
            facts.known_true |= Props::NONZERO | Props::COMPLEX;
            facts.known_false |= Props::REAL | Props::EXTENDED_REAL;
        }
        ConstantClass::NonzeroReal => facts.known_true |= Props::NONZERO | Props::REAL,
        ConstantClass::Nonzero => facts.known_true |= Props::NONZERO,
        ConstantClass::Real => facts.known_true |= Props::REAL,
        ConstantClass::Unknown => {}
    }
    facts.forward_chain();
    facts
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
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Asin) {
        return Some(v);
    }
    // asin(i·c) = i·asinh(c) on all of ℂ (SymPy's `asin.eval`): asin(±i) =
    // ±i·ln(1 + √2), and `acsc(±i) = asin(∓i)`.  Up to 0.43 they stayed.
    if let Some(c) = imaginary_real_cofactor(arena, inner) {
        let v = eval_asinh(arena, c)?;
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    // asin(v) = π/2 − acos(v) = s·(1/2 − q)·π for |v| = cos(qπ).
    if let Some((s, q)) = special_angle_of(arena, inner, SpecialTable::Cos) {
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        return Some(pi_times(arena, (half - q) * s));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_asin(arena, pos).unwrap_or_else(|| arena.asin(pos));
    Some(arena.neg(v))
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
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Acos) {
        return Some(v);
    }
    // acos(i·c) = π/2 − i·asinh(c) (`acos = π/2 − asin`; SymPy: `acos(I)` →
    // pi/2 - I*log(1 + sqrt(2))), and `asec(±i) = acos(∓i)`.
    if let Some(c) = imaginary_real_cofactor(arena, inner) {
        let v = eval_asinh(arena, c)?;
        let i_v = arena.mul(&[arena.i_unit, v]);
        let minus_i_v = arena.neg(i_v);
        let quarter_turn = half_pi(arena);
        return Some(arena.add(&[quarter_turn, minus_i_v]));
    }
    // acos(±cos(qπ)) = qπ or (1 − q)π (principal range [0, π]).
    let (s, q) = special_angle_of(arena, inner, SpecialTable::Cos)?;
    let angle = if s.is_negative() { Ratio::one() - q } else { q };
    Some(pi_times(arena, angle))
}

fn eval_atan(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atan(0) = 0
    // atan(±∞) = ±π/2 (SymPy: `atan(oo)` → pi/2).  Before 0.30 it stayed,
    // and `evalf` refused it (an infinite argument).
    // At `±∞ + c` too (any finite `c`), and at `c ± i·∞` it is ±π/2 by the
    // half plane, `sign(Re c)·π/2`, SymPy's `atan(±oo*I) = ±pi/2` on the
    // cut (mpmath `atan(mpc(-1, 1e20))` = −1.5707…).
    if let Some(inf) = crate::base::canon::infinite_argument(arena, inner) {
        let positive = if inf.imaginary {
            match inf.side? {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => !inf.negative,
            }
        } else {
            !inf.negative
        };
        let half = arena.rational(if positive { 1 } else { -1 }, 2);
        return Some(arena.mul(&[half, arena.pi]));
    }
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
    if let Some(v) = inverse_of_trig_at_pi_multiple(arena, inner, InverseTrig::Atan) {
        return Some(v);
    }
    // atan(±tan(qπ)) = ±qπ (principal range (−π/2, π/2)).
    if let Some((s, q)) = special_angle_of(arena, inner, SpecialTable::Tan) {
        return Some(pi_times(arena, q * s));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_atan(arena, pos).unwrap_or_else(|| arena.atan(pos));
    Some(arena.neg(v))
}

/// Which inverse function [`inverse_of_trig_at_pi_multiple`] applies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InverseTrig {
    Asin,
    Acos,
    Atan,
}

/// The principal value of `asin`, `acos` or `atan` at `sin(qπ)`, `cos(qπ)`
/// or (for `atan`) `tan(qπ)` with `q` rational, an application `eval` kept
/// (`q` off SymPy's denominators): `asin(sin(7π/102)) = 7π/102`,
/// `acos(cos(14π/9)) = 4π/9`, `asin(cos(49π/85)) = −13π/170`,
/// `atan(tan(−34π/43)) = 9π/43` — SymPy's `asin.eval`, `acos.eval`,
/// `atan.eval` rules for a trigonometric argument (there for any comparable
/// angle, here for rational multiples of `π`, exactly).  Before 0.43 they
/// stayed.
fn inverse_of_trig_at_pi_multiple(
    arena: &mut Arena,
    inner: ExprId,
    which: InverseTrig,
) -> Option<ExprId> {
    // `cot` is `cos/sin` or `1/tan` here (`acot(tan x) = atan(1/tan x)`):
    // atan(cot(qπ)) = atan(tan((1/2 − q)π)).
    let cot_argument = |arena: &Arena| -> Option<ExprId> {
        match arena.node(inner) {
            ExprNode::Pow(b, e) if *e == arena.neg_one => match *arena.node(*b) {
                ExprNode::Tan(u) => Some(u),
                _ => None,
            },
            ExprNode::Mul(cs) if cs.len() == 2 => {
                let (c, s) = (arena.node(cs[0]), arena.node(cs[1]));
                match (c, s) {
                    (ExprNode::Cos(u), ExprNode::Pow(b, e))
                    | (ExprNode::Pow(b, e), ExprNode::Cos(u))
                        if *e == arena.neg_one
                            && matches!(*arena.node(*b), ExprNode::Sin(v) if v == *u) =>
                    {
                        Some(*u)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let (u, f, cot) = match *arena.node(inner) {
        ExprNode::Sin(u) => (u, InverseTrig::Asin, false),
        ExprNode::Cos(u) => (u, InverseTrig::Acos, false),
        ExprNode::Tan(u) if which == InverseTrig::Atan => (u, InverseTrig::Atan, false),
        _ if which == InverseTrig::Atan => (cot_argument(arena)?, InverseTrig::Atan, true),
        _ => return None,
    };
    let q = as_pi_multiple(arena, u)?;
    let q = if cot {
        Ratio::new(BigInt::from(1), BigInt::from(2)) - q
    } else {
        q
    };
    let one = Q::one();
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    // asin(sin(qπ)) in [−π/2, π/2] and acos(cos(qπ)) in [0, π], in units of π.
    let asin_of_sin = |q: &Q| {
        let mut r = mod_positive(q, &Ratio::from_integer(BigInt::from(2)));
        if r > one {
            r = &one - r;
        }
        if r > half {
            r = &one - r;
        } else if r < -half.clone() {
            r = -&one - r;
        }
        r
    };
    let acos_of_cos = |q: &Q| {
        let two = Ratio::from_integer(BigInt::from(2));
        let r = mod_positive(q, &two);
        if r > one { two - r } else { r }
    };
    let r = match (which, f) {
        (InverseTrig::Atan, InverseTrig::Atan) => {
            let r = mod_positive(&q, &one);
            // A pole of `tan` (only reached through `cot`, at a zero of `sin`).
            if r == half {
                return None;
            }
            if r > half { r - &one } else { r }
        }
        (InverseTrig::Asin, InverseTrig::Asin) => asin_of_sin(&q),
        (InverseTrig::Acos, InverseTrig::Acos) => acos_of_cos(&q),
        // asin(x) + acos(x) = π/2.
        (InverseTrig::Asin, InverseTrig::Acos) => &half - acos_of_cos(&q),
        (InverseTrig::Acos, InverseTrig::Asin) => &half - asin_of_sin(&q),
        _ => return None,
    };
    Some(pi_times(arena, r))
}

/// `r·π` as a canonical node.
fn pi_times(arena: &mut Arena, r: Q) -> ExprId {
    if r.is_zero() {
        return arena.zero;
    }
    let c = arena.num_ratio(r);
    arena.mul(&[c, arena.pi])
}

/// Which trigonometric function a [`SPECIAL_COS`] / [`SPECIAL_TAN`] lookup
/// inverts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpecialTable {
    Cos,
    Tan,
}

/// A value `(a + b·√m)/d`, or `√(a + b·√m)/d` when `root`, of a
/// trigonometric function at the angle `kπ/n`.
struct SpecialValue {
    k: i64,
    n: i64,
    a: i64,
    b: i64,
    m: i64,
    d: i64,
    root: bool,
}

const fn sv(k: i64, n: i64, a: i64, b: i64, m: i64, d: i64, root: bool) -> SpecialValue {
    SpecialValue {
        k,
        n,
        a,
        b,
        m,
        d,
        root,
    }
}

/// `cos(kπ/n)` for `0 ≤ k/n ≤ 1/2`, `n ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12}`
/// (the angles whose cosines are square-root towers of degree ≤ 4).
const SPECIAL_COS: &[SpecialValue] = &[
    sv(0, 1, 1, 0, 0, 1, false),   // 1
    sv(1, 12, 2, 1, 3, 2, true),   // √(2 + √3)/2 = (√6 + √2)/4
    sv(1, 10, 10, 2, 5, 4, true),  // √(10 + 2√5)/4
    sv(1, 8, 2, 1, 2, 2, true),    // √(2 + √2)/2
    sv(1, 6, 0, 1, 3, 2, false),   // √3/2
    sv(1, 5, 1, 1, 5, 4, false),   // (1 + √5)/4
    sv(1, 4, 0, 1, 2, 2, false),   // √2/2
    sv(3, 10, 10, -2, 5, 4, true), // √(10 − 2√5)/4
    sv(1, 3, 1, 0, 0, 2, false),   // 1/2
    sv(3, 8, 2, -1, 2, 2, true),   // √(2 − √2)/2
    sv(2, 5, -1, 1, 5, 4, false),  // (√5 − 1)/4
    sv(5, 12, 2, -1, 3, 2, true),  // √(2 − √3)/2 = (√6 − √2)/4
    sv(1, 2, 0, 0, 0, 1, false),   // 0
];

/// `tan(kπ/n)` for `0 ≤ k/n < 1/2`, the same denominators.
const SPECIAL_TAN: &[SpecialValue] = &[
    sv(0, 1, 0, 0, 0, 1, false),    // 0
    sv(1, 12, 2, -1, 3, 1, false),  // 2 − √3
    sv(1, 10, 25, -10, 5, 5, true), // √(25 − 10√5)/5
    sv(1, 8, -1, 1, 2, 1, false),   // √2 − 1
    sv(1, 6, 0, 1, 3, 3, false),    // √3/3
    sv(1, 5, 5, -2, 5, 1, true),    // √(5 − 2√5)
    sv(1, 4, 1, 0, 0, 1, false),    // 1
    sv(3, 10, 25, 10, 5, 5, true),  // √(25 + 10√5)/5
    sv(1, 3, 0, 1, 3, 1, false),    // √3
    sv(3, 8, 1, 1, 2, 1, false),    // 1 + √2
    sv(2, 5, 5, 2, 5, 1, true),     // √(5 + 2√5)
    sv(5, 12, 2, 1, 3, 1, false),   // 2 + √3
];

/// Least common multiple of the table denominators.
const SPECIAL_LCM: i64 = 120;

/// Build the arena node of a table value.
fn special_value_node(arena: &mut Arena, e: &SpecialValue) -> ExprId {
    let a = arena.int(e.a);
    let mut inner = if e.b == 0 {
        a
    } else {
        let m = arena.int(e.m);
        let half = arena.rational(1, 2);
        let sqrt_m = arena.pow(m, half);
        let b = arena.int(e.b);
        let b_sqrt_m = arena.mul(&[b, sqrt_m]);
        arena.add(&[a, b_sqrt_m])
    };
    if e.root {
        let half = arena.rational(1, 2);
        inner = arena.pow(inner, half);
    }
    let inv_d = arena.rational(1, e.d);
    let v = arena.mul(&[inv_d, inner]);
    eval(arena, v)
}

/// Is `id` a real constant built from rationals, `φ` and square roots
/// (`+ − ×`, integer and half-integer powers), with at least one square
/// root and at most 64 nodes?  The table values are of this form; anything
/// else (symbols, `π`, `i`, cube roots, functions) cannot be one.
fn is_square_root_tower(arena: &Arena, id: ExprId) -> bool {
    // Explicit stack, stopping at the first node that disqualifies (the
    // common case: an argument with a symbol) or past 64 nodes.
    let mut stack = vec![id];
    let mut visited = 0usize;
    let mut irrational = false;
    while let Some(n) = stack.pop() {
        visited += 1;
        if visited > 64 {
            return false;
        }
        match arena.node(n) {
            ExprNode::Num(_) => {}
            ExprNode::Add(cs) | ExprNode::Mul(cs) => stack.extend(cs.iter().copied()),
            ExprNode::Neg(a) => stack.push(*a),
            ExprNode::GoldenRatio => irrational = true,
            ExprNode::Pow(b, e) => {
                match arena.as_num(*e) {
                    Some(q) if q.is_integer() => {}
                    Some(q) if *q.denom() == BigInt::from(2) => irrational = true,
                    _ => return false,
                }
                stack.push(*b);
            }
            _ => return false,
        }
    }
    irrational
}

/// Monic minimal polynomial over ℚ of an algebraic constant.
fn monic_minimal_polynomial(arena: &mut Arena, id: ExprId) -> Option<crate::poly::Poly> {
    let p = crate::poly::algebraic::minimal_polynomial(arena, id)?;
    (!p.is_zero()).then(|| p.make_monic())
}

/// Recognise `v = s·f(qπ)` for a table angle `q ∈ [0, 1/2]` (`f = cos`) or
/// `q ∈ [0, 1/2)` (`f = tan`) and a sign `s = ±1`; returns `(s, q)`.
///
/// The candidate angle comes from the certified `f64` value of `v` (the
/// nearest multiple of `π/120`, within `10⁻⁶` of a step); it is accepted
/// only when `|v|` is exactly the table value: structurally equal after
/// [`eval`], or with the same minimal polynomial over ℚ.  Equal minimal
/// polynomials and close values identify the number, because the
/// the roots of every table polynomial are more than `0.3` apart (they are
/// the conjugates `±f(jqπ)`, `gcd(j, n) = 1`) while the tolerance allows at
/// most `4·10⁻⁷` (`tan` at `5π/12`).  So the reduction does not depend
/// on the radical form of the input: `(√6 + √2)/4`, `√(2 + √3)/2` and
/// `1/(√6 − √2)` all give `acos = π/12`.
fn special_angle_of(arena: &mut Arena, v: ExprId, table: SpecialTable) -> Option<(Q, Q)> {
    if !is_square_root_tower(arena, v) {
        return None;
    }
    let x = crate::transforms::evalf::evalf_f64(arena, v).ok()?;
    if !x.is_finite() || x.abs() < 1e-12 {
        return None;
    }
    let theta = match table {
        SpecialTable::Cos if x.abs() <= 1.0 => x.abs().acos(),
        SpecialTable::Cos => return None,
        SpecialTable::Tan => x.abs().atan(),
    } / std::f64::consts::PI;
    let steps = theta * SPECIAL_LCM as f64;
    let r = steps.round();
    if (steps - r).abs() > 1e-6 {
        return None;
    }
    let q = Ratio::new(BigInt::from(r as i64), BigInt::from(SPECIAL_LCM));
    let entries = match table {
        SpecialTable::Cos => SPECIAL_COS,
        SpecialTable::Tan => SPECIAL_TAN,
    };
    let entry = entries
        .iter()
        .find(|e| Ratio::new(BigInt::from(e.k), BigInt::from(e.n)) == q)?;
    let abs_v = if x < 0.0 {
        let neg = arena.neg(v);
        eval(arena, neg)
    } else {
        v
    };
    let target = special_value_node(arena, entry);
    if abs_v != target {
        let p = monic_minimal_polynomial(arena, abs_v)?;
        if monic_minimal_polynomial(arena, target)? != p {
            return None;
        }
    }
    let s: Q = if x < 0.0 { -Ratio::one() } else { Ratio::one() };
    Some((s, q))
}

/// Evaluate `atan2(y, x)` with quadrant-aware logic.
///
/// When both arguments are numeric rationals, returns an exact symbolic
/// result.  Returns `None` when no simplification is possible.
pub(crate) fn eval_atan2(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    if let Some(angle) = eval_atan2_infinite(arena, y, x) {
        return Some(angle);
    }
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
                // atan2(0, 0) is the argument of 0: undefined (SymPy: nan),
                // as `arg(0)` is.  Before 0.30 it was 0 "by convention", and
                // `atan2(sin(π/4) − cos(π/4), 0)` evaluated to 0.
                Some(arena.nan)
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
        _ => eval_atan2_special(arena, y, x),
    }
}

/// The least (`min`) or greatest of the symbol-free real constants `args`
/// (`Min(π, 3) = 3`, `Max(√2, 3/2) = 3/2`; SymPy compares them the same
/// way), each comparison decided by the certified sign of a difference
/// ([`Arena::sign_of_real_constant`]).  `None` when an argument is not such
/// a constant or a comparison is not decided (equal values included).
fn extremum_of_real_constants(arena: &mut Arena, args: &[ExprId], min: bool) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    if args.len() < 2 || args.len() > 8 {
        return None;
    }
    let mut cache = AssumptionCache::new();
    for &a in args {
        if !crate::base::walk::free_symbols(arena, a).is_empty()
            || cache.query(arena, a, Props::FINITE) != Some(true)
        {
            return None;
        }
    }
    // Real by the assumptions, else by `known_real` (a root by Cardano's
    // formula is, by its minimal polynomial; up to 0.44 `Max(λ, −4)` stayed).
    for &a in args {
        if cache.query(arena, a, Props::REAL) != Some(true) && !known_real(arena, a) {
            return None;
        }
    }
    let mut best = args[0];
    for &a in &args[1..] {
        let d = arena.sub(a, best);
        let s = certified_real_sign(arena, d)?;
        if (min && s == std::cmp::Ordering::Less) || (!min && s == std::cmp::Ordering::Greater) {
            best = a;
        }
    }
    Some(best)
}

/// `atan2(y, x)` with one argument `±∞` and the other real and finite (as
/// SymPy: `atan2(1, oo)` → 0, `atan2(-1, -oo)` → -pi, `atan2(0, -oo)` →
/// pi, `atan2(oo, 1)` → pi/2): the limit of the angle.  `arg(∞ + i)` is
/// `atan2(1, ∞) = 0`.  Both infinite, or the other's sign unknown: `None`.
fn eval_atan2_infinite(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    use std::cmp::Ordering;
    let is_inf = |arena: &Arena, id: ExprId| id == arena.infinity || id == arena.neg_infinity;
    if is_inf(arena, y) == is_inf(arena, x) {
        return None;
    }
    let finite = if is_inf(arena, y) { x } else { y };
    let mut cache = AssumptionCache::new();
    if cache.query(arena, finite, Props::REAL) != Some(true) {
        return None;
    }
    let sign = if let Some(q) = arena.as_num(finite) {
        q.cmp(&Q::zero())
    } else if cache.query(arena, finite, Props::POSITIVE) == Some(true) {
        Ordering::Greater
    } else if cache.query(arena, finite, Props::NEGATIVE) == Some(true) {
        Ordering::Less
    } else if cache.query(arena, finite, Props::ZERO) == Some(true) {
        Ordering::Equal
    } else {
        arena.sign_of_real_constant(finite)?
    };
    let angle: Q = if is_inf(arena, x) {
        if x == arena.infinity {
            Q::zero()
        } else if sign == Ordering::Less {
            -Q::one()
        } else {
            Q::one()
        }
    } else {
        let half = Ratio::new(BigInt::from(1), BigInt::from(2));
        if y == arena.infinity { half } else { -half }
    };
    Some(pi_times(arena, angle))
}

/// `atan2(y, x)` for real square-root towers (at least one irrational) whose
/// quotient is a special tangent: the quadrant from the certified signs,
/// the angle from [`eval_atan`] of `y/x` (`atan2(√3/2, 1/2) = π/3`,
/// `atan2(−√2, −√6) = −5π/6`).  `None` unless the angle is a rational
/// multiple of `π`.
fn eval_atan2_special(arena: &mut Arena, y: ExprId, x: ExprId) -> Option<ExprId> {
    let tower_or_rational =
        |arena: &Arena, id: ExprId| arena.as_num(id).is_some() || is_square_root_tower(arena, id);
    if !tower_or_rational(arena, y) || !tower_or_rational(arena, x) {
        return None;
    }
    let sign_of = |arena: &Arena, id: ExprId| -> Option<i8> {
        if let Some(r) = arena.as_num(id) {
            return Some(if r.is_zero() {
                0
            } else if r.is_negative() {
                -1
            } else {
                1
            });
        }
        // An irrational tower is not zero; its sign is certified at 1e-12.
        let v = crate::transforms::evalf::evalf_f64(arena, id).ok()?;
        (v.is_finite() && v.abs() >= 1e-12).then_some(if v < 0.0 { -1 } else { 1 })
    };
    let sy = sign_of(arena, y)?;
    let sx = sign_of(arena, x)?;
    let half = Ratio::new(BigInt::from(1), BigInt::from(2));
    let angle: Q = match (sy, sx) {
        (0, 0) => return None,
        (0, 1) => Ratio::zero(),
        (0, _) => Ratio::one(),
        (1, 0) => half,
        (_, 0) => -half,
        _ => {
            let ratio = arena.div(y, x);
            let ratio = eval(arena, ratio);
            let a = eval_atan(arena, ratio)?;
            let q = as_pi_multiple(arena, a)?;
            match (sy, sx) {
                (_, 1) => q,
                (1, _) => q + Ratio::one(),
                _ => q - Ratio::one(),
            }
        }
    };
    Some(pi_times(arena, angle))
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
    // sinh(i·qπ) = i·sin(qπ) where that evaluates (SymPy's `sinh.eval`:
    // `sinh(I*pi/8)` → `I*sqrt(1/2 - sqrt(2)/4)`); before 0.43 only the
    // rational values folded (in the canonical constructor).
    // At any other rational `q` the sine stays (`sinh(iπ/105)` →
    // `i·sin(π/105)`, as SymPy writes it; up to 0.46 the `sinh` stayed).
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    let s = eval_sin(arena, angle).unwrap_or_else(|| arena.sin(angle));
    Some(arena.mul(&[arena.i_unit, s]))
}

fn eval_cosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    // Even function: cosh(-x) = cosh(x)
    if let Some(pos_inner) = as_negated(arena, inner) {
        return Some(eval_cosh(arena, pos_inner).unwrap_or_else(|| arena.cosh(pos_inner)));
    }

    if inner == arena.zero {
        return Some(arena.one);
    } // cosh(0) = 1
    // cosh(i·qπ) = cos(qπ) (SymPy: `cosh(I*pi/5)` → `1/4 + sqrt(5)/4`).
    // (`cosh(iπ/105)` → `cos(π/105)` at any other rational `q`.)
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    Some(eval_cos(arena, angle).unwrap_or_else(|| arena.cos(angle)))
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
    // tanh(i·qπ) = i·tan(qπ) (SymPy: `tanh(I*pi/12)` → `I*(2 - sqrt(3))`).
    // (`tanh(iπ/105)` → `i·tan(π/105)` at any other rational `q`.)
    let q = as_imaginary_pi_multiple(arena, inner)?;
    let angle = pi_times(arena, q);
    let t = eval_tan(arena, angle).unwrap_or_else(|| arena.tan(angle));
    Some(arena.mul(&[arena.i_unit, t]))
}

/// `c` for an argument `i·c` with `c` a real constant (`c = 1` for `i`
/// itself, `−1` for `−i`): the argument of the identities between the
/// inverse trigonometric and hyperbolic functions on the imaginary axis.
fn imaginary_real_cofactor(arena: &mut Arena, z: ExprId) -> Option<ExprId> {
    use crate::base::assumptions::{AssumptionCache, Props};
    let c = if z == arena.i_unit {
        arena.one
    } else {
        as_pure_imaginary(arena, z)?
    };
    (walk::free_symbols(arena, c).is_empty()
        && AssumptionCache::new().query(arena, c, Props::REAL) == Some(true))
    .then_some(c)
}

/// `ln(1 + √2) = asinh(1)`.
fn ln_one_plus_sqrt2(arena: &mut Arena) -> ExprId {
    let two = arena.int(2);
    let sqrt2 = arena.sqrt(two);
    let sum = arena.add(&[arena.one, sqrt2]);
    arena.ln(sum)
}

/// The special values of `asinh`, SymPy's `asinh.eval`: `asinh(±1) =
/// ±ln(1 + √2)` and `asinh(i·c) = i·asin(c)` for a real constant `c` whose
/// `asin` folds (`asinh(±i) = ±iπ/2`, `asinh(i√3/2) = iπ/3`); the identity
/// holds on all of ℂ (`asinh z = ln(z + √(1 + z²))`, `asin z = −i·ln(iz +
/// √(1 − z²))`).  Through `acsch(z) = asinh(1/z)` they give `acsch(±1)` and
/// `acsch(±i) = ∓iπ/2`.  Up to 0.43 only `asinh(0)` folded.
fn eval_asinh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // asinh(0) = 0
    if inner == arena.one || inner == arena.neg_one {
        let v = ln_one_plus_sqrt2(arena);
        return Some(if inner == arena.one { v } else { arena.neg(v) });
    }
    // asinh(sinh z) = z for a real constant z (SymPy: `asinh(sinh(3))` → 3).
    if let ExprNode::Sinh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        return Some(z);
    }
    if let Some(c) = imaginary_real_cofactor(arena, inner)
        && let Some(v) = eval_asin(arena, c)
    {
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_asinh(arena, pos).unwrap_or_else(|| arena.asinh(pos));
    Some(arena.neg(v))
}

/// The special values of `acosh` (SymPy's `acosh.eval` and its table):
/// `acosh(x) = i·acos(x)` for a real `x ∈ [−1, 1]` whose `acos` folds
/// (`acosh(0) = iπ/2`, `acosh(−1) = iπ`, `acosh(−1/2) = 2iπ/3`), and
/// `acosh(i·c) = asinh(|c|) + sign(c)·iπ/2` for a real `c` whose `asinh`
/// folds (`acosh(±i) = ln(1 + √2) ± iπ/2`), both on the principal branch
/// `acosh z = ln(z + √(z + 1)·√(z − 1))` (mpmath: `acosh(0.5)` = 1.0471…j,
/// `acosh(-2j)` = 1.4436… − 1.5707…j).  Through `asech(z) = acosh(1/z)`
/// they give `asech(−1) = iπ`, `asech(2) = iπ/3`, `asech(±∞) = iπ/2`.  Up
/// to 0.43 only `acosh(1)` folded, and `series(tanh(acosh(x)), x, 0, 3)`
/// kept `tanh(acosh(0))`, a pole, in its coefficients.
fn eval_acosh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.one {
        return Some(arena.zero);
    } // acosh(1) = 0
    // acosh(cosh z) = |z| for a real constant z (SymPy: `acosh(cosh(-2))` → 2).
    if let ExprNode::Cosh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        let a = arena.abs(z);
        return Some(eval_abs(arena, z).unwrap_or(a));
    }
    if known_real(arena, inner) {
        // `acos` folds only to `qπ` with `q ∈ [0, 1]`, so `x = cos(qπ)` is
        // in [−1, 1]; anything else is left alone.
        let v = eval_acos(arena, inner)?;
        let q = as_pi_multiple(arena, v)?;
        if q.is_negative() || q > Ratio::one() {
            return None;
        }
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let c = imaginary_real_cofactor(arena, inner)?;
    let r = arena.as_num(c)?.clone();
    if r.is_zero() {
        return None;
    }
    let abs_c = arena.num_ratio(r.abs());
    let s = eval_asinh(arena, abs_c)?;
    let half = arena.rational(if r.is_positive() { 1 } else { -1 }, 2);
    let i_half_pi = arena.mul(&[half, arena.i_unit, arena.pi]);
    Some(arena.add(&[s, i_half_pi]))
}

fn eval_atanh(arena: &mut Arena, inner: ExprId) -> Option<ExprId> {
    if inner == arena.zero {
        return Some(arena.zero);
    } // atanh(0) = 0
    // atanh(±∞) = ∓iπ/2, the limit along the real axis, where the
    // principal value is `atanh(x) = acoth(x) ∓ iπ/2` for `±x > 1` (mpmath:
    // `atanh(2)` = 0.549… − 1.5708…j; SymPy: `atanh(oo)` → -I*pi/2).
    // Before 0.30 it stayed, and `evalf` refused it.
    //
    // Off the real axis the side decides: atanh(z) → sign(Im z)·iπ/2 as |z|
    // grows, so atanh(±∞ + c) = sign(Im c)·iπ/2 and atanh(c ± i·∞) = ±iπ/2
    // (mpmath `atanh(mpc(1e20, 1))` = 1.0e-20 + 1.5707…j).  Up to 0.40 `∞ +
    // i` was `∞` and `atanh(∞ + i)` was −iπ/2.
    if let Some(inf) = crate::base::canon::infinite_argument(arena, inner) {
        let positive = if inf.imaginary {
            !inf.negative
        } else {
            match inf.side? {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => inf.negative,
            }
        };
        let half = arena.rational(if positive { 1 } else { -1 }, 2);
        return Some(arena.mul(&[half, arena.i_unit, arena.pi]));
    }
    // atanh(tanh z) = z for a real constant z (SymPy: `atanh(tanh(1))` → 1).
    if let ExprNode::Tanh(z) = *arena.node(inner)
        && walk::free_symbols(arena, z).is_empty()
        && known_real(arena, z)
    {
        return Some(z);
    }
    // atanh(i·c) = i·atan(c) on all of ℂ (SymPy's `atanh.eval`): `atanh(±i)
    // = ±iπ/4`, `atanh(i√3) = iπ/3`, and through `acoth(z) = atanh(1/z)`,
    // `acoth(±i) = ∓iπ/4`.  Up to 0.43 they stayed.
    if let Some(c) = imaginary_real_cofactor(arena, inner)
        && let Some(v) = eval_atan(arena, c)
    {
        return Some(arena.mul(&[arena.i_unit, v]));
    }
    let pos = negated_constant(arena, inner)?;
    let v = eval_atanh(arena, pos).unwrap_or_else(|| arena.atanh(pos));
    Some(arena.neg(v))
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
        // `lambertw(x)` and `lambertw(x, 0)` become `ExprNode::LambertW`.
        LibFn::LambertW => eval_lambertw_call(arena, args),

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

/// `J_0(0) = I_0(0) = 1`; `J_ν(0) = I_ν(0) = 0` for `ν > 0` and for every
/// integer `ν ≠ 0` (`J_{−n} = (−1)ⁿ J_n`, `I_{−n} = I_n`); `zoo` for a
/// negative non-integer `ν` (SymPy: `besselj(-3, 0)` → 0,
/// `besselj(-1/2, 0)` → zoo), when both order and argument are numbers.
/// Before 0.30 a negative order stayed.  (The zeros and poles fold when the
/// application is built; see `canon_function`.)
fn eval_bessel_regular_at_zero(arena: &mut Arena, order: ExprId, x: ExprId) -> Option<ExprId> {
    let order_num = arena.as_num(order)?;
    let arg_num = arena.as_num(x)?;
    if !arg_num.is_zero() {
        return None;
    }
    if order_num.is_zero() {
        Some(arena.one)
    } else if order_num.is_positive() || order_num.is_integer() {
        Some(arena.zero)
    } else {
        Some(arena.complex_infinity)
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
        let v = arena.div(e, x);
        return Some(eval(arena, v));
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

/// `Chi(0) = zoo` (the singularity of `ln z`, as `Ci(0)`; before 0.30
/// `−∞`), `Chi(∞) = ∞`.
fn eval_chi(arena: &mut Arena, x: ExprId) -> Option<ExprId> {
    if x == arena.zero {
        return Some(arena.complex_infinity);
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

/// [`lowergamma_closure_cancels`] for an `x = re + i·im` off the positive
/// axis (a complex or negative constant): the closed form's terms are
/// at most `Γ(s)·max(1, e^{|x| − re})`, and for `|x| < (s + 1)/2` the power
/// series `γ(s, x) = xˢe⁻ˣ/s · Σ xᵏ/((s + 1)⋯(s + k))` bounds `|γ(s, x)|`
/// below by `|x|ˢe^{−re}/s · (s + 1 − 2|x|)/(s + 1 − |x|)`; the ratio is the
/// loss.  Up to 0.45 only a real positive `x` was guarded: `γ(30, (−13/100 +
/// 99i/100)/10¹¹)` folded into `29!·(1 − e^{−x}·Σ…)`, which lost 1,200 bits
/// and was refused at 43 digits (`PrecisionExhausted`).
fn lowergamma_closure_cancels_complex(s: &Q, re: f64, im: f64) -> bool {
    let half_integer = s.denom() == &BigInt::from(2);
    if !(s.is_integer() || half_integer)
        || *s > Q::from_integer(BigInt::from(MAX_SPECIAL_EXPANSION))
    {
        return false;
    }
    let Some(s_f) = s.to_f64() else {
        return false;
    };
    let rho = re.hypot(im);
    if !rho.is_finite() || rho >= (s_f + 1.0) / 2.0 {
        return false;
    }
    if rho == 0.0 {
        // Below the range of `f64`: the series' first term is all there is.
        return true;
    }
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
    let loss = ln_gamma - s_f * rho.ln() + re + (rho - re).max(0.0)
        - ((s_f + 1.0 - 2.0 * rho) / (s_f + 1.0 - rho)).ln();
    loss / std::f64::consts::LN_2 > LOWERGAMMA_FOLD_MAX_LOSS_BITS
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
        if !xr.is_positive()
            && let Some(x_f) = xr.to_f64()
            && lowergamma_closure_cancels_complex(&r, x_f, 0.0)
        {
            return None;
        }
    } else if let Some(ln_x) = crate::transforms::evalf::ln_of_positive_constant(arena, x) {
        if r.to_f64().is_some_and(|s_f| ln_x < s_f.ln())
            && lowergamma_closure_cancels_at(&r, ln_x.exp(), ln_x)
        {
            return None;
        }
    } else if walk::free_symbols(arena, x).is_empty()
        && let Ok(z) = crate::transforms::evalf::evalf_complex64(arena, x)
        && lowergamma_closure_cancels_complex(&r, z.re, z.im)
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
        // `Γ(s)` evaluated (`uppergamma(4, 0)` is 6, as SymPy gives it;
        // up to 0.46 it was `Gamma(4)` until a second `eval`).
        return Some(eval_gamma(arena, s).unwrap_or_else(|| arena.gamma(s)));
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
        let l = eval_ln(arena, one_minus_z).unwrap_or_else(|| arena.ln(one_minus_z));
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
///
/// At `−∞` only `Ai` and `Bi` tend to `0` (`Ai(−x) ~ x^(−1/4)·sin(⅔x^(3/2) +
/// π/4)/√π`); their derivatives oscillate with the growing amplitude
/// `x^(1/4)/√π` (DLMF 9.7.10, 9.7.12) and have no limit, so `Ai′(−∞)` and
/// `Bi′(−∞)` stay (SymPy 1.14 leaves `airyaiprime(-oo)`; its
/// `airybiprime(-oo)` → 0 is wrong).  Before 0.41 both folded to `0`.
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
    if x == arena.neg_infinity && matches!(f, LibFn::AiryAi | LibFn::AiryBi) {
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
        let e = eval_elliptic_f(arena, y, m)
            .unwrap_or_else(|| apply_named(arena, LibFn::EllipticF, &[y, m]));
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
