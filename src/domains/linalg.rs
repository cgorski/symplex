//! Linear system solving via Gaussian elimination.
//!
//! Two solvers live here:
//!
//! - [`solve_linear_system`] — fast path over exact rationals
//!   (`Ratio<BigInt>`), unique solutions only.
//! - [`linsolve_symbolic`] — symbolic reduced row-echelon form over
//!   expressions.  Handles symbolic coefficients, under- and
//!   over-determined systems, reports free variables and inconsistency.
//!   This is the backend of the public `linsolve` API.
//!
//! # Algorithm (rational fast path)
//!
//! 1. Extract the coefficient matrix from the linear expressions.
//! 2. Augment with the constant terms (negated).
//! 3. Perform Gaussian elimination with partial pivoting.
//! 4. Back-substitute to find the solution.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::api::expr::{Ex, SimplifyOpts};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};

/// Result of solving a linear system.
#[derive(Debug)]
pub(crate) struct LinearSolution {
    /// Pairs of (variable, value) for each solved variable.
    pub pairs: Vec<(ExprId, ExprId)>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic RREF solver
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the symbolic RREF solver.
#[derive(Debug, Clone)]
pub(crate) struct SymbolicLinearResult {
    /// One value per unknown, in input order.  Free unknowns map to
    /// themselves; pivot unknowns are expressed in terms of the free ones.
    pub values: Vec<Ex>,
    /// Indices (into the unknowns) of the free variables.
    pub free: Vec<usize>,
    /// `true` if a row `0 = c` with `c ≠ 0` was found.
    pub inconsistent: bool,
}

/// Decompose `eq` (an expression equal to zero) into a linear row over
/// `vars`: returns `(coefficients, rhs)` with `Σ coeffᵢ·varᵢ = rhs`.
///
/// Returns `None` if `eq` is not linear in the unknowns (a term contains
/// two unknowns, a power of an unknown, or an unknown inside a function).
pub(crate) fn extract_linear_row(
    arena: &mut Arena,
    eq: ExprId,
    vars: &[ExprId],
) -> Option<(Vec<ExprId>, ExprId)> {
    let expanded = crate::transforms::expand::expand(arena, eq);
    let expanded = crate::transforms::eval::eval(arena, expanded);
    let terms: Vec<ExprId> = match arena.node(expanded).clone() {
        ExprNode::Add(children) => children.to_vec(),
        _ => vec![expanded],
    };

    let mut coeff_parts: Vec<Vec<ExprId>> = vec![Vec::new(); vars.len()];
    let mut const_parts: Vec<ExprId> = Vec::new();

    for term in terms {
        let mut which: Option<usize> = None;
        for (i, &v) in vars.iter().enumerate() {
            if crate::base::walk::contains(arena, term, v) {
                if which.is_some() {
                    return None; // two unknowns in one term → nonlinear
                }
                which = Some(i);
            }
        }
        match which {
            None => const_parts.push(term),
            Some(i) => {
                let coeffs = crate::transforms::solve::symbolic_poly_coeffs(arena, term, vars[i])?;
                // Must be exactly c·var (degree 1, zero constant part).
                if coeffs.len() != 2 || !arena.is_zero_structural(coeffs[0]) {
                    return None;
                }
                coeff_parts[i].push(coeffs[1]);
            }
        }
    }

    let mut coeffs = Vec::with_capacity(vars.len());
    for parts in coeff_parts {
        let c = match parts.len() {
            0 => arena.zero,
            1 => parts[0],
            _ => arena.add(&parts),
        };
        coeffs.push(crate::transforms::eval::eval(arena, c));
    }
    let constant = match const_parts.len() {
        0 => arena.zero,
        1 => const_parts[0],
        _ => arena.add(&const_parts),
    };
    let rhs = arena.neg(constant);
    let rhs = crate::transforms::eval::eval(arena, rhs);
    Some((coeffs, rhs))
}

/// Final clean-up of a solved value: simplify, and combine over a common
/// denominator when that is shorter.
fn tidy(e: &Ex) -> Ex {
    let s = e.eval().simplify();
    let t = s.together().simplify();
    if t.count_ops() < s.count_ops() { t } else { s }
}

/// Light-weight normalisation used between elimination steps.
fn normalize(e: &Ex) -> Ex {
    let e1 = e.eval();
    if e1.is_zero_structural() || is_number(&e1) {
        return e1;
    }
    let e2 = e1.simplify_with(&SimplifyOpts::single_pass());
    if e2.count_ops() <= e1.count_ops() { e2 } else { e1 }
}

/// `true` if `e` is a numeric literal.
fn is_number(e: &Ex) -> bool {
    e.inner.read().arena.as_num(e.raw_id()).is_some()
}

/// `true` if `e` is a **nonzero** numeric literal.
fn is_nonzero_number(e: &Ex) -> bool {
    e.inner
        .read()
        .arena
        .as_num(e.raw_id())
        .is_some_and(|r| !r.is_zero())
}

/// Decide whether an (already normalised) entry is zero.
///
/// Structural zero → `true`; numeric literal → by value; symbolic → a
/// full `simplify` pass is tried, then the assumption system.
fn entry_is_zero(e: &Ex) -> bool {
    if e.is_zero_structural() {
        return true;
    }
    if is_number(e) {
        return false;
    }
    let s = e.simplify();
    if s.is_zero_structural() {
        return true;
    }
    // Purely numeric (no free symbols) but not folded: decide numerically.
    if s.free_symbols().is_empty()
        && let Ok((re, im)) = s.eval_complex64()
    {
        return re.hypot(im) < 1e-12;
    }
    false
}

/// Symbolic reduced-row-echelon solve of `rows · x = rhs`.
///
/// `rows[i]` holds the coefficients of equation `i`; `rhs[i]` its
/// right-hand side.  `n_vars` is the number of unknowns; `unknowns` are the
/// symbols the free variables should be expressed with.
///
/// Pivot choice prefers nonzero numeric literals; a symbolic pivot is used
/// only when no numeric one is available and is then **assumed nonzero**
/// (generic solution).
pub(crate) fn rref_solve(
    rows: Vec<Vec<Ex>>,
    rhs: Vec<Ex>,
    unknowns: &[Ex],
) -> Result<SymbolicLinearResult, SymplexError> {
    let n_vars = unknowns.len();
    let m = rows.len();
    if m == 0 || n_vars == 0 {
        return Err(SymplexError::InvalidArgument {
            operation: "linsolve",
            reason: "need at least one equation and one unknown".into(),
        });
    }
    // Augmented matrix.
    let mut mat: Vec<Vec<Ex>> = rows
        .into_iter()
        .zip(rhs)
        .map(|(mut r, b)| {
            r.push(b);
            r
        })
        .collect();
    let ncols = n_vars + 1;
    for r in &mat {
        if r.len() != ncols {
            return Err(SymplexError::InvalidArgument {
                operation: "linsolve",
                reason: "row length does not match number of unknowns".into(),
            });
        }
    }

    let mut pivots: Vec<usize> = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..n_vars {
        if pivot_row >= m {
            break;
        }
        // Candidate pivots: prefer numeric nonzero, then simplest symbolic.
        let mut numeric: Option<usize> = None;
        let mut symbolic: Option<(usize, usize)> = None; // (row, ops)
        for r in pivot_row..m {
            let e = normalize(&mat[r][col]);
            mat[r][col] = e.clone();
            if is_nonzero_number(&e) {
                numeric = Some(r);
                break;
            }
            if !entry_is_zero(&e) {
                let ops = e.count_ops();
                if symbolic.is_none_or(|(_, o)| ops < o) {
                    symbolic = Some((r, ops));
                }
            } else {
                mat[r][col] = e.context().zero();
            }
        }
        let found = match (numeric, symbolic) {
            (Some(r), _) => r,
            (None, Some((r, _))) => r,
            (None, None) => continue,
        };
        if found != pivot_row {
            mat.swap(pivot_row, found);
        }
        // Scale pivot row.
        let pv = mat[pivot_row][col].clone();
        for j in 0..ncols {
            if j == col {
                mat[pivot_row][j] = pv.context().one();
            } else {
                let v = &mat[pivot_row][j] / &pv;
                mat[pivot_row][j] = normalize(&v);
            }
        }
        // Eliminate in all other rows.
        for i in 0..m {
            if i == pivot_row {
                continue;
            }
            let factor = normalize(&mat[i][col]);
            if entry_is_zero(&factor) {
                mat[i][col] = factor.context().zero();
                continue;
            }
            for j in 0..ncols {
                if j == col {
                    mat[i][j] = factor.context().zero();
                } else {
                    let t = &factor * &mat[pivot_row][j];
                    let v = &mat[i][j] - &t;
                    mat[i][j] = normalize(&v);
                }
            }
        }
        pivots.push(col);
        pivot_row += 1;
    }

    // Inconsistency check: rows with zero coefficients but nonzero rhs.
    for r in &mat {
        let all_zero = r[..n_vars].iter().all(entry_is_zero);
        if all_zero && !entry_is_zero(&r[n_vars]) {
            return Ok(SymbolicLinearResult {
                values: Vec::new(),
                free: Vec::new(),
                inconsistent: true,
            });
        }
    }

    let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
    let free: Vec<usize> = (0..n_vars).filter(|c| !pivot_set.contains(c)).collect();

    let mut values: Vec<Ex> = unknowns.to_vec();
    for (r, &pc) in pivots.iter().enumerate() {
        let mut v = mat[r][n_vars].clone();
        for &f in &free {
            if !entry_is_zero(&mat[r][f]) {
                let t = &mat[r][f] * &unknowns[f];
                v = &v - &t;
            }
        }
        values[pc] = tidy(&v);
    }

    Ok(SymbolicLinearResult {
        values,
        free,
        inconsistent: false,
    })
}

/// Solve the linear system `eqs = 0` for `vars` symbolically.
///
/// See [`rref_solve`] for the pivoting policy.  Returns
/// [`SymplexError::InvalidArgument`] if an equation is not linear in the
/// unknowns or the input is empty.
pub(crate) fn linsolve_symbolic(
    eqs: &[Ex],
    vars: &[Ex],
) -> Result<SymbolicLinearResult, SymplexError> {
    if eqs.is_empty() || vars.is_empty() {
        return Err(SymplexError::InvalidArgument {
            operation: "linsolve",
            reason: "need at least one equation and one unknown".into(),
        });
    }
    let first = &eqs[0];
    let var_ids: Vec<ExprId> = vars.iter().map(|v| first.checked_id(v)).collect();
    let eq_ids: Vec<ExprId> = eqs.iter().map(|e| first.checked_id(e)).collect();

    let mut rows: Vec<Vec<ExprId>> = Vec::with_capacity(eqs.len());
    let mut rhs: Vec<ExprId> = Vec::with_capacity(eqs.len());
    {
        let mut guard = first.inner.write();
        let arena = &mut guard.arena;
        for (k, &eq) in eq_ids.iter().enumerate() {
            match extract_linear_row(arena, eq, &var_ids) {
                Some((coeffs, b)) => {
                    rows.push(coeffs);
                    rhs.push(b);
                }
                None => {
                    let shown = arena.display(eq).to_string();
                    drop(guard);
                    return Err(SymplexError::InvalidArgument {
                        operation: "linsolve",
                        reason: format!("equation {k} is not linear in the unknowns: {shown}"),
                    });
                }
            }
        }
    }
    let rows: Vec<Vec<Ex>> = rows
        .into_iter()
        .map(|r| r.into_iter().map(|id| first.wrap(id)).collect())
        .collect();
    let rhs: Vec<Ex> = rhs.into_iter().map(|id| first.wrap(id)).collect();
    rref_solve(rows, rhs, vars)
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
    use crate::base::arena::Arena;

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
