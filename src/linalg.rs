//! Linear system solving via Gaussian elimination.
//!
//! Implements [`solve_linear_system`], which solves a system of linear
//! equations over exact rationals (`Ratio<BigInt>`).
//!
//! # Algorithm
//!
//! 1. Extract the coefficient matrix from the linear expressions.
//! 2. Augment with the constant terms (negated).
//! 3. Perform Gaussian elimination with partial pivoting.
//! 4. Back-substitute to find the solution.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode};

/// Result of solving a linear system.
#[derive(Debug)]
pub(crate) struct LinearSolution {
    /// Pairs of (variable, value) for each solved variable.
    pub pairs: Vec<(ExprId, ExprId)>,
}

/// Solve a system of linear equations.
///
/// `equations` are expressions that equal zero.
/// `variables` are the symbols to solve for.
///
/// Returns `None` if the system has no unique solution (underdetermined,
/// overdetermined, or inconsistent).
pub(crate) fn solve_linear_system(
    arena: &mut Arena,
    equations: &[ExprId],
    variables: &[ExprId],
) -> Option<LinearSolution> {
    let n_eqs = equations.len();
    let n_vars = variables.len();

    if n_eqs == 0 || n_vars == 0 {
        return None;
    }

    // Build the augmented matrix [A | b] where A·x = -constant.
    // Each equation is a linear combination of variables + constant = 0.
    // So: coeff_1·var_1 + coeff_2·var_2 + ... + constant = 0
    // → A·x = -constant
    let mut matrix: Vec<Vec<Ratio<BigInt>>> = Vec::with_capacity(n_eqs);

    for &eq in equations {
        let mut row = Vec::with_capacity(n_vars + 1);

        for &var in variables {
            let coeff = extract_linear_coeff(arena, eq, var);
            row.push(coeff);
        }

        // The constant term (terms not involving any variable).
        let constant = extract_constant(arena, eq, variables);
        // b = -constant (since equation = 0)
        row.push(-constant);

        matrix.push(row);
    }

    // Gaussian elimination with partial pivoting.
    // Indexed access is intentional — we need simultaneous access to the
    // pivot row (matrix[col]) and the elimination row (matrix[row]).
    let cols = n_vars + 1;
    #[allow(clippy::needless_range_loop)]
    for col in 0..n_vars.min(n_eqs) {
        // Find pivot.
        let mut pivot_row = None;
        for row in col..n_eqs {
            if !matrix[row][col].is_zero() {
                pivot_row = Some(row);
                break;
            }
        }

        let pivot_row = match pivot_row {
            Some(r) => r,
            None => continue, // No pivot in this column — skip.
        };

        // Swap pivot row into position.
        if pivot_row != col {
            matrix.swap(col, pivot_row);
        }

        // Eliminate below.
        let pivot_val = matrix[col][col].clone();
        for row in (col + 1)..n_eqs {
            if matrix[row][col].is_zero() {
                continue;
            }
            let factor = matrix[row][col].clone() / pivot_val.clone();
            for j in col..cols {
                let sub = factor.clone() * matrix[col][j].clone();
                matrix[row][j] -= sub;
            }
        }
    }

    // Check for inconsistency: a row [0 0 ... 0 | nonzero].
    for row in &matrix {
        let all_zero = row[..n_vars].iter().all(|x| x.is_zero());
        if all_zero && !row[n_vars].is_zero() {
            return None; // Inconsistent.
        }
    }

    // Back-substitution.
    let mut solution: Vec<Ratio<BigInt>> = vec![Ratio::zero(); n_vars];
    for col in (0..n_vars).rev() {
        if col >= n_eqs || matrix[col][col].is_zero() {
            return None; // Underdetermined (free variable).
        }

        let mut val = matrix[col][n_vars].clone();
        for j in (col + 1)..n_vars {
            val -= matrix[col][j].clone() * solution[j].clone();
        }
        val /= matrix[col][col].clone();
        solution[col] = val;
    }

    // Convert rational solutions to ExprIds.
    let mut pairs = Vec::with_capacity(n_vars);
    for (i, &var) in variables.iter().enumerate() {
        let val = &solution[i];
        let nid = arena.intern_num(val.clone());
        let expr_id = arena.intern(ExprNode::Num(nid));
        pairs.push((var, expr_id));
    }

    Some(LinearSolution { pairs })
}

/// Extract the coefficient of `var` in a linear expression.
///
/// Walks the Add terms, looking for terms that contain `var` as a factor.
fn extract_linear_coeff(arena: &Arena, expr: ExprId, var: ExprId) -> Ratio<BigInt> {
    match arena.node(expr).clone() {
        ExprNode::Symbol(_) => {
            if expr == var {
                Ratio::one()
            } else {
                Ratio::zero()
            }
        }
        ExprNode::Mul(children) => {
            // Look for var in the factors.
            let mut has_var = false;
            let mut coeff = Ratio::one();
            for &child in &children {
                if child == var {
                    has_var = true;
                } else if let Some(r) = arena.as_num(child) {
                    coeff *= r.clone();
                } else {
                    // Non-linear term or other variable.
                    return Ratio::zero();
                }
            }
            if has_var { coeff } else { Ratio::zero() }
        }
        ExprNode::Add(children) => {
            let mut total = Ratio::zero();
            for &child in &children {
                total += extract_linear_coeff(arena, child, var);
            }
            total
        }
        ExprNode::Neg(inner) => -extract_linear_coeff(arena, inner, var),
        ExprNode::Num(_) => Ratio::zero(), // Constant — no var.
        _ => Ratio::zero(),                // Non-linear.
    }
}

/// Extract the constant part of an expression (terms not involving any variable).
fn extract_constant(arena: &Arena, expr: ExprId, variables: &[ExprId]) -> Ratio<BigInt> {
    match arena.node(expr).clone() {
        ExprNode::Num(nid) => arena.num(nid).clone(),
        ExprNode::Symbol(_) => {
            if variables.contains(&expr) {
                Ratio::zero()
            } else {
                // Unknown symbol — can't extract constant.
                Ratio::zero()
            }
        }
        ExprNode::Add(children) => {
            let mut total = Ratio::zero();
            for &child in &children {
                total += extract_constant(arena, child, variables);
            }
            total
        }
        ExprNode::Neg(inner) => -extract_constant(arena, inner, variables),
        ExprNode::Mul(children) => {
            // Only a constant if no children are variables.
            let involves_var = children
                .iter()
                .any(|&c| contains_any_var(arena, c, variables));
            if involves_var {
                Ratio::zero()
            } else {
                // All constant — evaluate.
                let mut prod = Ratio::one();
                for &c in &children {
                    if let Some(r) = arena.as_num(c) {
                        prod *= r.clone();
                    } else {
                        return Ratio::zero();
                    }
                }
                prod
            }
        }
        _ => Ratio::zero(),
    }
}

fn contains_any_var(arena: &Arena, expr: ExprId, variables: &[ExprId]) -> bool {
    if variables.contains(&expr) {
        return true;
    }
    let children = arena.node(expr).children();
    children
        .iter()
        .any(|&c| contains_any_var(arena, c, variables))
}

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

    #[test]
    fn solve_simple_2x2() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // x + y - 3 = 0  →  x + y = 3
        // x - y - 1 = 0  →  x - y = 1
        // Solution: x = 2, y = 1
        let three = a.int(3);
        let neg_three = a.neg(three);
        let one = a.one;
        let neg_y = a.neg(y);
        let neg_one = a.neg(one);
        let eq1 = a.add(&[x, y, neg_three]);
        let eq2 = a.add(&[x, neg_y, neg_one]);

        let result = solve_linear_system(&mut a, &[eq1, eq2], &[x, y]).unwrap();
        assert_eq!(result.pairs.len(), 2);
        assert_eq!(display(&a, result.pairs[0].1), "2");
        assert_eq!(display(&a, result.pairs[1].1), "1");
    }

    #[test]
    fn solve_with_coefficients() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        // 2x + 3y - 7 = 0
        // x - y - 1 = 0
        // Solution: x = 2, y = 1
        let two = a.int(2);
        let three = a.int(3);
        let seven = a.int(7);
        let one = a.one;
        let two_x = a.mul(&[two, x]);
        let three_y = a.mul(&[three, y]);
        let neg_seven = a.neg(seven);
        let neg_y = a.neg(y);
        let neg_one = a.neg(one);
        let eq1 = a.add(&[two_x, three_y, neg_seven]);
        let eq2 = a.add(&[x, neg_y, neg_one]);

        let result = solve_linear_system(&mut a, &[eq1, eq2], &[x, y]).unwrap();
        assert_eq!(display(&a, result.pairs[0].1), "2");
        assert_eq!(display(&a, result.pairs[1].1), "1");
    }

    #[test]
    fn solve_single_equation() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 2x - 6 = 0 → x = 3
        let two = a.int(2);
        let six = a.int(6);
        let two_x = a.mul(&[two, x]);
        let neg_six = a.neg(six);
        let eq = a.add(&[two_x, neg_six]);
        let result = solve_linear_system(&mut a, &[eq], &[x]).unwrap();
        assert_eq!(display(&a, result.pairs[0].1), "3");
    }

    #[test]
    fn solve_inconsistent_returns_none() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // x - 1 = 0 AND x - 2 = 0 → inconsistent
        let one = a.one;
        let eq1 = a.sub(x, one);
        let two = a.int(2);
        let eq2 = a.sub(x, two);
        assert!(solve_linear_system(&mut a, &[eq1, eq2], &[x]).is_none());
    }

    #[test]
    fn solve_rational_solution() {
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        // 3x - 1 = 0 → x = 1/3
        let three = a.int(3);
        let three_x = a.mul(&[three, x]);
        let one = a.one;
        let eq = a.sub(three_x, one);
        let result = solve_linear_system(&mut a, &[eq], &[x]).unwrap();
        assert_eq!(display(&a, result.pairs[0].1), "1/3");
    }
}
