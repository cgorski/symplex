//! Closed-form evaluation of symbolic sums.
//!
//! Strategies tried in order:
//! 1. Linearity — factor out constants, distribute over addition
//! 2. Polynomial sums (Faulhaber) — Σk^n for n=0..4
//! 3. Geometric series — Σr^k = (r^a - r^(b+1))/(1-r)
//! 4. Constant body — body independent of var: body * (upper - lower + 1)

use num_traits::{Signed, ToPrimitive};
use tracing::trace;

use crate::base::arena::Arena;
use crate::base::node::{ExprId, ExprNode, SymbolId};
use crate::base::walk;

/// Attempt closed-form evaluation of `Σ_{var=lower}^{upper} body`.
///
/// Returns `Some(result)` if a closed form was found, `None` otherwise.
pub(crate) fn eval_sum_symbolic(
    arena: &mut Arena,
    body: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
) -> Option<ExprId> {
    let _var_sym: SymbolId = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    trace!("eval_sum_symbolic: attempting closed-form sum");

    // Strategy 1: Linearity — if body is Add, sum each term separately
    if let ExprNode::Add(ref children) = arena.node(body).clone() {
        trace!("eval_sum_symbolic: trying linearity over Add");
        let child_vec: Vec<ExprId> = children.to_vec();
        let terms: Vec<Option<ExprId>> = child_vec
            .iter()
            .map(|&c| eval_sum_symbolic(arena, c, var, lower, upper))
            .collect();
        if terms.iter().all(|t| t.is_some()) {
            let evaluated: Vec<ExprId> = terms.into_iter().map(|t| t.unwrap()).collect();
            trace!(
                "eval_sum_symbolic: linearity succeeded for all {} terms",
                evaluated.len()
            );
            return Some(arena.add(&evaluated));
        }
    }

    // Strategy 1b: Constant factor — if body is c*f(k), extract c
    if let ExprNode::Mul(ref children) = arena.node(body).clone() {
        let child_vec: Vec<ExprId> = children.to_vec();
        // Partition into constant and variable-dependent factors
        let mut const_factors: Vec<ExprId> = Vec::new();
        let mut var_factors: Vec<ExprId> = Vec::new();

        for &c in &child_vec {
            let syms = walk::free_symbols(arena, c);
            if syms.contains(&var) {
                var_factors.push(c);
            } else {
                const_factors.push(c);
            }
        }

        if !const_factors.is_empty() && !var_factors.is_empty() {
            trace!(
                "eval_sum_symbolic: extracting {} constant factor(s)",
                const_factors.len()
            );
            let var_body = if var_factors.len() == 1 {
                var_factors[0]
            } else {
                arena.mul(&var_factors)
            };
            if let Some(inner_sum) = eval_sum_symbolic(arena, var_body, var, lower, upper) {
                let const_part = if const_factors.len() == 1 {
                    const_factors[0]
                } else {
                    arena.mul(&const_factors)
                };
                let all = vec![const_part, inner_sum];
                return Some(arena.mul(&all));
            }
        }
    }

    // Strategy 2: Polynomial — check if body is var or var^n for small n
    if body == var {
        trace!("eval_sum_symbolic: matched k^1 (Faulhaber)");
        return Some(faulhaber(arena, 1, lower, upper));
    }

    if let ExprNode::Pow(base, exp) = arena.node(body).clone()
        && base == var
            && let Some(n) = arena.as_num(exp).and_then(|r| {
                if r.is_integer() && r.is_positive() {
                    r.to_integer().to_usize()
                } else {
                    None
                }
            })
                && n <= 4 {
                    trace!("eval_sum_symbolic: matched k^{} (Faulhaber)", n);
                    return Some(faulhaber(arena, n, lower, upper));
                }
            // Also check if exp == var (geometric case handled below)

    // Strategy 3: Constant body (independent of var): body * (upper - lower + 1)
    {
        let syms = walk::free_symbols(arena, body);
        if !syms.contains(&var) {
            trace!("eval_sum_symbolic: constant body detected");
            let one = arena.one;
            let diff = arena.sub(upper, lower);
            let count = arena.add(&[diff, one]);
            return Some(arena.mul(&[body, count]));
        }
    }

    // Strategy 4: Geometric — check if body is r^k where r doesn't depend on var
    if let ExprNode::Pow(base, exp) = arena.node(body).clone()
        && exp == var {
            let base_syms = walk::free_symbols(arena, base);
            if !base_syms.contains(&var) {
                trace!("eval_sum_symbolic: matched geometric series r^k");
                // Σ r^k from a to b = (r^a - r^(b+1)) / (1 - r)
                let r = base;
                let r_a = arena.pow(r, lower);
                let one = arena.one;
                let b_plus_1 = arena.add(&[upper, one]);
                let r_b1 = arena.pow(r, b_plus_1);
                let numer = arena.sub(r_a, r_b1);
                let denom = arena.sub(one, r);
                return Some(arena.div(numer, denom));
            }
        }

    trace!("eval_sum_symbolic: no closed form found");
    None
}

/// Compute `Σ_{k=lower}^{upper} k^power` using Faulhaber formulas.
///
/// Uses the identity: `Σ_{k=a}^{b} k^p = F(p, b) - F(p, a-1)`
/// where `F(p, n) = Σ_{k=1}^{n} k^p`.
fn faulhaber(arena: &mut Arena, power: usize, lower: ExprId, upper: ExprId) -> ExprId {
    let f_upper = faulhaber_from_1(arena, power, upper);
    let one = arena.one;
    let lower_minus_1 = arena.sub(lower, one);
    let f_lower = faulhaber_from_1(arena, power, lower_minus_1);
    arena.sub(f_upper, f_lower)
}

/// Compute `Σ_{k=1}^{n} k^power` as a closed-form expression in `n`.
fn faulhaber_from_1(arena: &mut Arena, power: usize, n: ExprId) -> ExprId {
    let one = arena.one;
    match power {
        0 => {
            // Σ_{k=1}^{n} 1 = n
            n
        }
        1 => {
            // Σ_{k=1}^{n} k = n(n+1)/2
            let n_plus_1 = arena.add(&[n, one]);
            let prod = arena.mul(&[n, n_plus_1]);
            let two = arena.int(2);
            arena.div(prod, two)
        }
        2 => {
            // Σ_{k=1}^{n} k² = n(n+1)(2n+1)/6
            let n_plus_1 = arena.add(&[n, one]);
            let two = arena.int(2);
            let two_n = arena.mul(&[two, n]);
            let two_n_plus_1 = arena.add(&[two_n, one]);
            let prod = arena.mul(&[n, n_plus_1, two_n_plus_1]);
            let six = arena.int(6);
            arena.div(prod, six)
        }
        3 => {
            // Σ_{k=1}^{n} k³ = [n(n+1)/2]²
            let n_plus_1 = arena.add(&[n, one]);
            let two = arena.int(2);
            let nn1 = arena.mul(&[n, n_plus_1]);
            let half_prod = arena.div(nn1, two);
            let two_exp = arena.int(2);
            arena.pow(half_prod, two_exp)
        }
        4 => {
            // Σ_{k=1}^{n} k⁴ = n(n+1)(2n+1)(3n²+3n-1)/30
            let n_plus_1 = arena.add(&[n, one]);
            let two = arena.int(2);
            let three = arena.int(3);
            let two_n = arena.mul(&[two, n]);
            let two_n_plus_1 = arena.add(&[two_n, one]);
            let n_sq = arena.pow(n, two);
            let three_n_sq = arena.mul(&[three, n_sq]);
            let three_n = arena.mul(&[three, n]);
            let neg_one = arena.neg_one;
            let bracket = arena.add(&[three_n_sq, three_n, neg_one]);
            let prod = arena.mul(&[n, n_plus_1, two_n_plus_1, bracket]);
            let thirty = arena.int(30);
            arena.div(prod, thirty)
        }
        _ => {
            // For powers > 4, we can't produce a closed form — return a Sum node
            let var_placeholder = n; // not ideal but safe fallback
            let pow_exp = arena.int(power as i64);
            let body = arena.pow(var_placeholder, pow_exp);
            arena.intern(ExprNode::Sum(body, var_placeholder, one, n))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build arena, create sum, try closed form, then evaluate numerically.
    fn eval_sum(power: usize, lo: i64, hi: i64) -> String {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(lo);
        let upper = arena.int(hi);

        let body = if power == 1 {
            var
        } else {
            let exp = arena.int(power as i64);
            arena.pow(var, exp)
        };

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(
            result.is_some(),
            "Expected closed form for k^{power} from {lo} to {hi}"
        );
        let expr = result.unwrap();
        // Evaluate the closed form
        let evaled = crate::transforms::eval::eval(&mut arena, expr);
        arena.display(evaled).to_string()
    }

    #[test]
    fn faulhaber_k_1_to_10() {
        // Σk from 1 to 10 = 55
        assert_eq!(eval_sum(1, 1, 10), "55");
    }

    #[test]
    fn faulhaber_k_sq_1_to_10() {
        // Σk² from 1 to 10 = 385
        assert_eq!(eval_sum(2, 1, 10), "385");
    }

    #[test]
    fn faulhaber_k_cube_1_to_10() {
        // Σk³ from 1 to 10 = 3025
        assert_eq!(eval_sum(3, 1, 10), "3025");
    }

    #[test]
    fn faulhaber_k_4_1_to_10() {
        // Σk⁴ from 1 to 10 = 25333
        assert_eq!(eval_sum(4, 1, 10), "25333");
    }

    #[test]
    fn constant_body() {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(1);
        let upper = arena.int(10);
        let body = arena.int(5);

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(result.is_some());
        let evaled = crate::transforms::eval::eval(&mut arena, result.unwrap());
        assert_eq!(arena.display(evaled).to_string(), "50");
    }

    #[test]
    fn geometric_series() {
        let mut arena = Arena::new();
        let var = arena.symbol("k");
        let lower = arena.int(0);
        let upper = arena.int(9);
        let two = arena.int(2);
        let body = arena.pow(two, var); // 2^k

        let result = eval_sum_symbolic(&mut arena, body, var, lower, upper);
        assert!(result.is_some());
        let evaled = crate::transforms::eval::eval(&mut arena, result.unwrap());
        assert_eq!(arena.display(evaled).to_string(), "1023");
    }
}
