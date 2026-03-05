#![allow(clippy::needless_range_loop)]
//! Combinatorial simplification.
//!
//! Simplifies expressions involving factorial and binomial coefficients.
//! Strategy (inspired by SymPy):
//! 1. Identify factorial/binomial subexpressions
//! 2. Cancel common factorials in numerator/denominator
//! 3. Apply: n!/((n-1)!) → n, C(n,k)*k! → n!/(n-k)!, etc.

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};
use crate::walk;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::ToPrimitive;
use rustc_hash::FxHashMap;

/// Apply combinatorial simplification rules bottom-up.
///
/// Walks the expression and at each `Mul` node looks for factorial
/// cancellations such as `n! / (n-1)!` → `n`.
pub(crate) fn combsimp(arena: &mut Arena, expr: ExprId) -> ExprId {
    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, ExprId> = FxHashMap::default();

    for &id in &post_order {
        let rebuilt = crate::walk::rebuild_with_cache(arena, id, &cache);
        let simplified = combsimp_node(arena, rebuilt);
        cache.insert(id, simplified);
    }

    cache.get(&expr).copied().unwrap_or(expr)
}

/// Try to simplify a single node. Only `Mul` nodes are candidates
/// for factorial cancellation; everything else passes through.
fn combsimp_node(arena: &mut Arena, id: ExprId) -> ExprId {
    let node = arena.node(id).clone();
    match node {
        ExprNode::Mul(ref children) => {
            let children_vec: Vec<ExprId> = children.iter().copied().collect();
            simplify_factorial_mul(arena, id, &children_vec)
        }
        _ => id,
    }
}

/// Information about a factorial term found inside a `Mul` node.
struct FactorialTerm {
    /// Index into the `Mul` children slice.
    child_idx: usize,
    /// The argument of the factorial (e.g. `n` in `n!`).
    arg: ExprId,
}

/// Scan a `Mul` node for factorial numerator/denominator pairs and
/// simplify them.
///
/// Recognises these patterns:
///
/// - `Factorial(n) * Pow(Factorial(m), -1)` where `n − m = k` (small
///   positive integer) → replace with `(m+1)*(m+2)*…*n`
/// - `Factorial(n) * Pow(Factorial(m), -1)` where `n = m` → cancel to 1
/// - `Pow(Factorial(n), -1) * Pow(Factorial(k), -1) * Factorial(m)`
///   where `n + k = m` → replace with `Binomial(m, n)`
fn simplify_factorial_mul(arena: &mut Arena, original: ExprId, children: &[ExprId]) -> ExprId {
    let neg_one = Ratio::from_integer(BigInt::from(-1));

    // Collect factorial terms that appear in the numerator and denominator.
    let mut numer_facts: Vec<FactorialTerm> = Vec::new();
    let mut denom_facts: Vec<FactorialTerm> = Vec::new();

    for (idx, &child) in children.iter().enumerate() {
        match arena.node(child).clone() {
            ExprNode::Factorial(arg) => {
                numer_facts.push(FactorialTerm {
                    child_idx: idx,
                    arg,
                });
            }
            ExprNode::Pow(base, exp) => {
                // Check for Pow(Factorial(arg), -1)
                if let Some(val) = arena.as_num(exp)
                    && *val == neg_one
                {
                    match arena.node(base).clone() {
                        ExprNode::Factorial(arg) => {
                            denom_facts.push(FactorialTerm {
                                child_idx: idx,
                                arg,
                            });
                        }
                        ExprNode::Mul(ref mul_children) => {
                            // Pow(Mul([Factorial(a), Factorial(b), ...]), -1)
                            // Extract each Factorial child as a denom term.
                            let mut all_factorial = true;
                            let mut fact_args = Vec::new();
                            for &mc in mul_children.iter() {
                                if let ExprNode::Factorial(arg) = arena.node(mc).clone() {
                                    fact_args.push(arg);
                                } else {
                                    all_factorial = false;
                                    break;
                                }
                            }
                            if all_factorial {
                                for arg in fact_args {
                                    denom_facts.push(FactorialTerm {
                                        child_idx: idx,
                                        arg,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // ── Pattern 1: n! / m! where n − m = k (small positive int) ───────
    // Also handles n! / n! = 1 (k = 0).
    for ni in 0..numer_facts.len() {
        for di in 0..denom_facts.len() {
            let n_arg = numer_facts[ni].arg;
            let d_arg = denom_facts[di].arg;

            // Quick structural equality check first (handles n! / n! = 1).
            if n_arg == d_arg {
                return remove_and_replace(
                    arena,
                    children,
                    original,
                    &[numer_facts[ni].child_idx, denom_facts[di].child_idx],
                    None,
                );
            }

            // Compute difference n_arg − d_arg and try to evaluate it.
            let diff_expr = arena.sub(n_arg, d_arg);
            let diff_eval = crate::eval::eval(arena, diff_expr);

            if let Some(val) = arena.as_num(diff_eval).cloned() {
                if !val.is_integer() {
                    continue;
                }
                let k: Option<i64> = val.to_integer().to_i64();
                let k = match k {
                    Some(k) => k,
                    None => continue,
                };

                if k > 0 && k <= 20 {
                    // n! / m! = (m+1)*(m+2)*…*(m+k)  (= (m+1)*…*n)
                    let replacement = if k == 1 {
                        n_arg
                    } else {
                        let mut factors = Vec::with_capacity(k as usize);
                        for i in 1..=k {
                            let offset = arena.int(i);
                            let factor = arena.add(&[d_arg, offset]);
                            let factor_eval = crate::eval::eval(arena, factor);
                            factors.push(factor_eval);
                        }
                        arena.mul(&factors)
                    };

                    return remove_and_replace(
                        arena,
                        children,
                        original,
                        &[numer_facts[ni].child_idx, denom_facts[di].child_idx],
                        Some(replacement),
                    );
                } else if k == 0 {
                    // n! / n! = 1 — remove both, no replacement
                    return remove_and_replace(
                        arena,
                        children,
                        original,
                        &[numer_facts[ni].child_idx, denom_facts[di].child_idx],
                        None,
                    );
                }
                // k < 0 means denom > numer, i.e. 1 / (m!/n!) — skip for now
            }
        }
    }

    // ── Pattern 2: n! / (k! * (n−k)!) → C(n, k) ──────────────────────
    if numer_facts.len() == 1 && denom_facts.len() == 2 {
        let n_arg = numer_facts[0].arg;
        let k1_arg = denom_facts[0].arg;
        let k2_arg = denom_facts[1].arg;

        // Check if k1 + k2 == n
        let sum_expr = arena.add(&[k1_arg, k2_arg]);
        let sum_eval = crate::eval::eval(arena, sum_expr);
        let diff_expr = arena.sub(n_arg, sum_eval);
        let diff_eval = crate::eval::eval(arena, diff_expr);

        if arena.is_zero_structural(diff_eval) {
            // n! / (k1! * k2!) = C(n, k1)
            let binom = arena.binomial(n_arg, k1_arg);
            return remove_and_replace(
                arena,
                children,
                original,
                &[
                    numer_facts[0].child_idx,
                    denom_facts[0].child_idx,
                    denom_facts[1].child_idx,
                ],
                Some(binom),
            );
        }
    }

    original
}

/// Remove the children at `used_indices` from the `Mul`, optionally
/// insert `replacement`, and return the rebuilt product.
///
/// If no children remain and no replacement is given, returns `arena.one`.
fn remove_and_replace(
    arena: &mut Arena,
    children: &[ExprId],
    _original: ExprId,
    used_indices: &[usize],
    replacement: Option<ExprId>,
) -> ExprId {
    // Deduplicate indices (multiple denom facts may share the same
    // Pow(Mul([...]), -1) child).
    let mut new_children: Vec<ExprId> = children
        .iter()
        .enumerate()
        .filter(|&(idx, _)| !used_indices.contains(&idx))
        .map(|(_, &c)| c)
        .collect();

    if let Some(r) = replacement {
        new_children.push(r);
    }

    match new_children.len() {
        0 => arena.one,
        1 => new_children[0],
        _ => arena.mul(&new_children),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(arena: &mut Arena, name: &str) -> ExprId {
        arena.symbol(name)
    }

    fn display(arena: &Arena, id: ExprId) -> String {
        arena.display(id).to_string()
    }

    #[test]
    fn factorial_ratio_concrete_5_over_4() {
        let mut arena = Arena::new();
        let five = arena.int(5);
        let four = arena.int(4);
        let five_fact = arena.factorial(five);
        let four_fact = arena.factorial(four);
        let ratio = arena.div(five_fact, four_fact);
        let evaled = crate::eval::eval(&mut arena, ratio);
        let result = combsimp(&mut arena, evaled);
        assert_eq!(display(&arena, result), "5");
    }

    #[test]
    fn factorial_ratio_symbolic_n_over_n_minus_1() {
        let mut arena = Arena::new();
        let n = sym(&mut arena, "n");
        let one = arena.int(1);
        let n_minus_1 = arena.sub(n, one);
        let n_fact = arena.factorial(n);
        let nm1_fact = arena.factorial(n_minus_1);
        let ratio = arena.div(n_fact, nm1_fact);
        let result = combsimp(&mut arena, ratio);
        assert_eq!(display(&arena, result), "n");
    }

    #[test]
    fn factorial_ratio_same_cancels() {
        let mut arena = Arena::new();
        let n = sym(&mut arena, "n");
        let n_fact = arena.factorial(n);
        let ratio = arena.div(n_fact, n_fact);
        let result = combsimp(&mut arena, ratio);
        assert_eq!(display(&arena, result), "1");
    }

    #[test]
    fn factorial_ratio_difference_2() {
        let mut arena = Arena::new();
        let n = sym(&mut arena, "n");
        let two = arena.int(2);
        let n_minus_2 = arena.sub(n, two);
        let n_fact = arena.factorial(n);
        let nm2_fact = arena.factorial(n_minus_2);
        let ratio = arena.div(n_fact, nm2_fact);
        let result = combsimp(&mut arena, ratio);
        let s = display(&arena, result);
        // n! / (n-2)! = (n-1)*n
        assert!(
            !s.contains('!') && !s.contains("factorial"),
            "should not contain factorials: {s}"
        );
    }

    #[test]
    fn combsimp_no_factorial_unchanged() {
        let mut arena = Arena::new();
        let x = sym(&mut arena, "x");
        let y = sym(&mut arena, "y");
        let expr = arena.add(&[x, y]);
        let result = combsimp(&mut arena, expr);
        assert_eq!(result, expr);
    }

    #[test]
    fn binomial_detection() {
        let mut arena = Arena::new();
        let n = sym(&mut arena, "n");
        let k = sym(&mut arena, "k");
        let n_minus_k = arena.sub(n, k);
        let n_fact = arena.factorial(n);
        let k_fact = arena.factorial(k);
        let nmk_fact = arena.factorial(n_minus_k);
        // n! / (k! * (n-k)!)
        let denom = arena.mul(&[k_fact, nmk_fact]);
        let ratio = arena.div(n_fact, denom);
        let result = combsimp(&mut arena, ratio);
        let s = display(&arena, result);
        assert!(
            s.contains("C(") || s.contains("binomial"),
            "should produce binomial coefficient, got: {s}"
        );
    }
}
