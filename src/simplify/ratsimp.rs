//! Rational-function normal form (`ratsimp`).
//!
//! An expression built from numbers, `Add`, `Mul`, `Neg` and integer
//! powers over a set of *generators* — the free symbols together with every
//! maximal non-rational subexpression (function applications, constants
//! such as π or e, powers with non-integer exponents, …) — is a rational
//! function `P/Q` with `P, Q ∈ ℚ[generators]`.  [`ratsimp`] computes that
//! single fraction, cancels `gcd(P, Q)` with the heuristic multivariate GCD,
//! clears denominators so that both parts have integer coefficients with no
//! common integer factor, makes the leading coefficient of `Q` positive, and
//! rebuilds `P / Q`.
//!
//! This mirrors what SymPy's `cancel` does: opaque subexpressions are
//! treated as independent indeterminates, so `sin(x)² / sin(x)` becomes
//! `sin(x)` but no trigonometric identity is applied.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode};
use crate::base::numeric::Q;
use crate::base::walk;
use crate::poly::multipoly::{GrevLex, Lex, MonomialOrd, MultiPoly};
use crate::poly::polybridge::multipoly_to_expr;

type RatPoly = MultiPoly<GrevLex>;

/// Largest number of terms any intermediate polynomial may reach before the
/// conversion gives up and the input is returned unchanged.
const MAX_TERMS: usize = 20_000;

/// Largest total degree any intermediate polynomial may reach.  Keeps
/// exponent arithmetic far away from `u32` overflow.
const MAX_DEGREE: u32 = 1 << 20;

/// Rational-function normal form of `expr` (see the module docs).
///
/// Returns `expr` unchanged when it contains `±∞`, `zoo`, `NaN` or an
/// unevaluated node, when a division by a polynomial that is identically
/// zero occurs, or when an intermediate polynomial exceeds the size budget.
pub(crate) fn ratsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    if !is_admissible(arena, expr) {
        return expr;
    }
    let gens = collect_generators(arena, expr);
    let Some((p, q)) = to_rational_function(arena, expr, &gens) else {
        return expr;
    };
    rebuild(arena, &p, &q, &gens).unwrap_or(expr)
}

/// No infinities, NaN or unevaluated nodes anywhere in the tree.
fn is_admissible(arena: &Arena, expr: ExprId) -> bool {
    if walk::has_unevaluated(arena, expr) {
        return false;
    }
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Infinity
            | ExprNode::NegInfinity
            | ExprNode::ComplexInfinity
            | ExprNode::NaN => return false,
            node => stack.extend(node.children()),
        }
    }
    true
}

/// Integer exponent of a `Pow` node, if it is one within the degree budget.
fn integer_exponent(arena: &Arena, exp: ExprId) -> Option<i64> {
    let r = arena.as_num(exp)?;
    if !r.is_integer() {
        return None;
    }
    let n: i64 = r.to_integer().try_into().ok()?;
    if n.unsigned_abs() > u64::from(MAX_DEGREE) {
        return None;
    }
    Some(n)
}

/// Is this node handled structurally (as opposed to being a generator)?
fn is_structural(arena: &Arena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Num(_) | ExprNode::Add(_) | ExprNode::Mul(_) | ExprNode::Neg(_) => true,
        ExprNode::Pow(_, exp) => integer_exponent(arena, *exp).is_some(),
        _ => false,
    }
}

/// The maximal non-structural subexpressions of `expr`, in canonical order.
fn collect_generators(arena: &Arena, expr: ExprId) -> Vec<ExprId> {
    let mut gens: Vec<ExprId> = Vec::new();
    let mut visited: FxHashSet<ExprId> = FxHashSet::default();
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        if is_structural(arena, id) {
            match arena.node(id) {
                ExprNode::Pow(base, _) => stack.push(*base),
                node => stack.extend(node.children()),
            }
        } else {
            gens.push(id);
        }
    }
    gens.sort_by(|a, b| {
        arena
            .sort_key(*a)
            .cmp(arena.sort_key(*b))
            .then_with(|| a.0.cmp(&b.0))
    });
    gens
}

/// Is `p` the constant polynomial 1?
fn is_one(p: &RatPoly) -> bool {
    p.num_terms() == 1 && p.total_degree() == Some(0) && p.leading_coeff().is_some_and(One::is_one)
}

/// Enforce the size budget.
fn within_budget(p: &RatPoly) -> bool {
    p.num_terms() <= MAX_TERMS && p.total_degree().unwrap_or(0) <= MAX_DEGREE
}

/// `base^n` for `n ≥ 0` by repeated squaring, `None` if the budget is hit.
fn pow(base: &RatPoly, mut n: u64) -> Option<RatPoly> {
    let nv = base.num_vars();
    let mut result = RatPoly::from_int(nv, 1);
    let mut sq = base.clone();
    while n > 0 {
        if n & 1 == 1 {
            result = result.mul(&sq);
            if !within_budget(&result) {
                return None;
            }
        }
        n >>= 1;
        if n > 0 {
            sq = sq.mul(&sq);
            if !within_budget(&sq) {
                return None;
            }
        }
    }
    Some(result)
}

/// `p1/q1 + p2/q2` over the least common multiple of the denominators.
fn add_fractions(p1: &RatPoly, q1: &RatPoly, p2: &RatPoly, q2: &RatPoly) -> (RatPoly, RatPoly) {
    if q1 == q2 {
        return (p1.add(p2), q1.clone());
    }
    if is_one(q1) {
        return (p1.mul(q2).add(p2), q2.clone());
    }
    if is_one(q2) {
        return (p1.add(&p2.mul(q1)), q1.clone());
    }
    let l = RatPoly::lcm(q1, q2);
    match (l.div_exact(q1), l.div_exact(q2)) {
        (Some(m1), Some(m2)) => (p1.mul(&m1).add(&p2.mul(&m2)), l),
        _ => (p1.mul(q2).add(&p2.mul(q1)), q1.mul(q2)),
    }
}

/// Convert `expr` to `(P, Q)` over the generators.  Iterative post-order
/// over the structural nodes; generators are leaves.
fn to_rational_function(
    arena: &Arena,
    expr: ExprId,
    gens: &[ExprId],
) -> Option<(RatPoly, RatPoly)> {
    let nv = gens.len();
    let gen_index: FxHashMap<ExprId, usize> =
        gens.iter().enumerate().map(|(i, &g)| (g, i)).collect();
    let mut cache: FxHashMap<ExprId, (RatPoly, RatPoly)> = FxHashMap::default();
    let one = RatPoly::from_int(nv, 1);
    // (node, children already pushed?)
    let mut stack: Vec<(ExprId, bool)> = vec![(expr, false)];

    while let Some(&(id, expanded)) = stack.last() {
        if cache.contains_key(&id) {
            stack.pop();
            continue;
        }
        if let Some(&i) = gen_index.get(&id) {
            cache.insert(id, (RatPoly::var(nv, i), one.clone()));
            stack.pop();
            continue;
        }
        let node = arena.node(id).clone();
        if !expanded {
            if let Some(top) = stack.last_mut() {
                top.1 = true;
            }
            match &node {
                ExprNode::Pow(base, _) => stack.push((*base, false)),
                n => {
                    for child in n.children() {
                        stack.push((child, false));
                    }
                }
            }
            continue;
        }
        let value: (RatPoly, RatPoly) = match &node {
            ExprNode::Num(nid) => (RatPoly::constant(nv, arena.num(*nid).clone()), one.clone()),
            ExprNode::Add(children) => {
                let mut acc_p = RatPoly::zero(nv);
                let mut acc_q = one.clone();
                for c in children.iter() {
                    let (p, q) = cache.get(c)?;
                    let (np, nq) = add_fractions(&acc_p, &acc_q, p, q);
                    if !within_budget(&np) || !within_budget(&nq) {
                        return None;
                    }
                    acc_p = np;
                    acc_q = nq;
                }
                (acc_p, acc_q)
            }
            ExprNode::Mul(children) => {
                let mut acc_p = one.clone();
                let mut acc_q = one.clone();
                for c in children.iter() {
                    let (p, q) = cache.get(c)?;
                    acc_p = acc_p.mul(p);
                    acc_q = acc_q.mul(q);
                    if !within_budget(&acc_p) || !within_budget(&acc_q) {
                        return None;
                    }
                }
                (acc_p, acc_q)
            }
            ExprNode::Neg(inner) => {
                let (p, q) = cache.get(inner)?;
                (p.neg(), q.clone())
            }
            ExprNode::Pow(base, exp) => {
                let n = integer_exponent(arena, *exp)?;
                let (p, q) = cache.get(base)?;
                if n >= 0 {
                    (pow(p, n.unsigned_abs())?, pow(q, n.unsigned_abs())?)
                } else {
                    if p.is_zero() {
                        return None;
                    }
                    (pow(q, n.unsigned_abs())?, pow(p, n.unsigned_abs())?)
                }
            }
            // Unreachable: non-structural nodes are generators.
            _ => return None,
        };
        cache.insert(id, value);
        stack.pop();
    }
    cache.remove(&expr)
}

/// Leading coefficient under the lexicographic order.
fn lex_leading_coeff(p: &RatPoly) -> Option<Q> {
    let mut best: Option<(&[u32], &Q)> = None;
    for (exp, c) in p.terms() {
        match best {
            Some((be, _)) if Lex::cmp_exponents(exp, be) != std::cmp::Ordering::Greater => {}
            _ => best = Some((exp, c)),
        }
    }
    best.map(|(_, c)| c.clone())
}

/// Cancel, normalise to integer-primitive parts and rebuild `P / Q`.
fn rebuild(arena: &mut Arena, p: &RatPoly, q: &RatPoly, gens: &[ExprId]) -> Option<ExprId> {
    if q.is_zero() {
        return None;
    }
    if p.is_zero() {
        return Some(arena.zero);
    }
    let mut p = p.clone();
    let mut q = q.clone();

    // Cancel the polynomial GCD.
    let g = RatPoly::gcd(&p, &q);
    if g.total_degree().unwrap_or(0) > 0
        && let (Some(pq), Some(qq)) = (p.div_exact(&g), q.div_exact(&g))
    {
        p = pq;
        q = qq;
    }

    // Integer coefficients: P/Q = (Pz/dP) / (Qz/dQ) = (Pz·dQ) / (Qz·dP).
    let (dp, pz) = p.clear_denominators();
    let (dq, qz) = q.clear_denominators();
    p = pz.scale(&Ratio::from_integer(dq));
    q = qz.scale(&Ratio::from_integer(dp));

    // Remove the common integer content.
    let c = p.integer_content().gcd(&q.integer_content());
    if !c.is_zero() && !c.is_one() {
        let inv = Ratio::new(BigInt::one(), c);
        p = p.scale(&inv);
        q = q.scale(&inv);
    }

    // Positive leading coefficient in the denominator.
    if lex_leading_coeff(&q).is_some_and(|lc| lc.is_negative()) {
        p = p.neg();
        q = q.neg();
    }

    let p_expr = multipoly_to_expr(arena, &p, gens);
    if is_one(&q) {
        return Some(p_expr);
    }
    let q_expr = multipoly_to_expr(arena, &q, gens);
    Some(arena.div(p_expr, q_expr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn linear_over_linear_cancels_and_normalises() {
        let mut a = Arena::new();
        let j = a.symbol("j");
        let one = a.one;
        let two = a.int(2);
        // (j² − 1)/(2j) − (j − 1)/2  →  (j − 1)/(2j)
        let j2 = a.pow(j, two);
        let num1 = a.sub(j2, one);
        let den1 = a.mul(&[two, j]);
        let t1 = a.div(num1, den1);
        let num2 = a.sub(j, one);
        let t2 = a.div(num2, two);
        let e = a.sub(t1, t2);
        let r = ratsimp(&mut a, e);
        let expected = a.div(num2, den1);
        assert_eq!(r, expected, "got {}", show(&a, r));
    }

    #[test]
    fn opaque_subexpressions_are_generators() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let s = a.sin(x);
        let two = a.int(2);
        let s2 = a.pow(s, two);
        let e = a.div(s2, s);
        assert_eq!(ratsimp(&mut a, e), s);
    }

    #[test]
    fn infinity_is_left_alone() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let inf = a.infinity;
        let e = a.add(&[x, inf]);
        assert_eq!(ratsimp(&mut a, e), e);
    }

    #[test]
    fn already_normal_input_is_unchanged() {
        let mut a = Arena::new();
        let x = a.symbol("x");
        let one = a.one;
        let e = a.add(&[x, one]);
        assert_eq!(ratsimp(&mut a, e), e);
        let inv = a.div(one, x);
        assert_eq!(ratsimp(&mut a, inv), inv);
    }
}
