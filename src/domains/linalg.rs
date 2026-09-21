//! Linear system solving via symbolic Gaussian elimination.
//!
//! [`linsolve_symbolic`] computes a reduced row-echelon form over
//! expressions.  It handles symbolic coefficients, under- and
//! over-determined systems, and reports free variables and inconsistency.
//! It is the backend of the public `linsolve` API (and of
//! `Context::solve_system`).

use num_traits::Zero;

use crate::api::expr::{Ex, SimplifyOpts};
use crate::base::arena::Arena;
use crate::base::errors::SymplexError;
use crate::base::node::{ExprId, ExprNode};
use crate::domains::exact_matrix::QMatrix;
use crate::domains::linprog::Q;

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
    // Combine over a common denominator and expand the numerator so that
    // cancellations like -a*b + b*(a+1) → b are found.
    let (num, den) = s.together().as_numer_denom();
    let num = num.expand().eval();
    let t = if den.is_one_structural() {
        num
    } else {
        (&num / &den).eval()
    };
    // Prefer the single-fraction form unless it is clearly larger
    // (`count_ops` is DAG-based, so a shared denominator makes the split
    // form look deceptively small; allow one extra node for the fraction).
    if t.count_ops() <= s.count_ops() + 1 {
        t
    } else {
        s
    }
}

/// Light-weight normalisation used between elimination steps.
fn normalize(e: &Ex) -> Ex {
    let e1 = e.eval();
    if e1.is_zero_structural() || is_number(&e1) {
        return e1;
    }
    let e2 = e1.simplify_with(&SimplifyOpts::single_pass());
    if e2.count_ops() <= e1.count_ops() {
        e2
    } else {
        e1
    }
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
        && let Ok(z) = s.eval_complex64()
    {
        return z.norm() < 1e-12;
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

    if let Some(result) = rref_solve_rational(&mat, n_vars, unknowns) {
        return Ok(result);
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
        for (r, row) in mat.iter_mut().enumerate().skip(pivot_row) {
            let e = normalize(&row[col]);
            row[col] = e.clone();
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
                row[col] = e.context().zero();
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
        for (j, entry) in mat[pivot_row].iter_mut().enumerate() {
            if j == col {
                *entry = pv.context().one();
            } else {
                let v = &*entry / &pv;
                *entry = normalize(&v);
            }
        }
        // Eliminate in all other rows.
        let pivot_vals = mat[pivot_row].clone();
        for (i, row) in mat.iter_mut().enumerate() {
            if i == pivot_row {
                continue;
            }
            let factor = normalize(&row[col]);
            if entry_is_zero(&factor) {
                row[col] = factor.context().zero();
                continue;
            }
            for (j, entry) in row.iter_mut().enumerate() {
                if j == col {
                    *entry = factor.context().zero();
                } else {
                    let t = &factor * &pivot_vals[j];
                    let v = &*entry - &t;
                    *entry = normalize(&v);
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

/// Exact fast path of [`rref_solve`] for an augmented matrix whose entries
/// are all rational literals: fraction-free elimination on a [`QMatrix`]
/// instead of expression arithmetic.  Returns `None` if any entry is
/// symbolic.
fn rref_solve_rational(
    mat: &[Vec<Ex>],
    n_vars: usize,
    unknowns: &[Ex],
) -> Option<SymbolicLinearResult> {
    let rows: Vec<Vec<Q>> = mat
        .iter()
        .map(|r| r.iter().map(Ex::as_rational).collect())
        .collect::<Option<_>>()?;
    let aug = QMatrix::new(rows).ok()?;
    let (r, pivots) = aug.rref_limited(n_vars);
    let rank = pivots.len();
    // Rows below the rank are zero in the coefficient part; a nonzero
    // right-hand side there means `0 = c`.
    if (rank..aug.nrows()).any(|i| !r[(i, n_vars)].is_zero()) {
        return Some(SymbolicLinearResult {
            values: Vec::new(),
            free: Vec::new(),
            inconsistent: true,
        });
    }
    let mut is_pivot = vec![false; n_vars];
    for &c in &pivots {
        is_pivot[c] = true;
    }
    let free: Vec<usize> = (0..n_vars).filter(|&c| !is_pivot[c]).collect();
    let ctx = unknowns.first()?.context();
    let mut values: Vec<Ex> = unknowns.to_vec();
    for (i, &pc) in pivots.iter().enumerate() {
        let mut v = ctx.from_ratio(r[(i, n_vars)].clone());
        for &f in &free {
            let c = &r[(i, f)];
            if !c.is_zero() {
                v -= ctx.from_ratio(c.clone()) * &unknowns[f];
            }
        }
        // Same clean-up as the symbolic path, so parametric solutions print
        // identically whichever route produced them.
        values[pc] = if free.is_empty() { v } else { tidy(&v) };
    }
    Some(SymbolicLinearResult {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::context::Context;

    fn show(e: &Ex) -> String {
        format!("{e}")
    }

    #[test]
    fn extract_linear_row_reads_coefficients_and_constant() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        // 2x - 3y + 5 = 0   →   coeffs [2, -3], rhs -5
        let eq = &x * 2 - &y * 3 + 5;
        let (coeffs, rhs) = ctx.with_arena_mut(|a| {
            extract_linear_row(a, eq.raw_id(), &[x.raw_id(), y.raw_id()])
                .map(|(c, r)| {
                    (
                        c.iter()
                            .map(|&i| a.display(i).to_string())
                            .collect::<Vec<_>>(),
                        a.display(r).to_string(),
                    )
                })
                .expect("linear")
        });
        assert_eq!(coeffs, ["2", "-3"]);
        assert_eq!(rhs, "-5");
    }

    #[test]
    fn extract_linear_row_rejects_products_powers_and_functions() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        for bad in [&x * &y, x.powi(2), x.sin(), &x / &y] {
            let none = ctx.with_arena_mut(|a| {
                extract_linear_row(a, bad.raw_id(), &[x.raw_id(), y.raw_id()]).is_none()
            });
            assert!(none, "{bad} should not be linear");
        }
    }

    #[test]
    fn unique_solution() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let r = linsolve_symbolic(&[&x + &y - 3, &x - &y - 1], &[x, y]).unwrap();
        assert!(!r.inconsistent);
        assert!(r.free.is_empty());
        assert_eq!(r.values.iter().map(show).collect::<Vec<_>>(), ["2", "1"]);
    }

    #[test]
    fn inconsistent_system_is_flagged() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        let r = linsolve_symbolic(&[&x - 1, &x - 2], &[x]).unwrap();
        assert!(r.inconsistent);
    }

    #[test]
    fn underdetermined_system_reports_free_variable() {
        let ctx = Context::new();
        let (x, y) = (ctx.symbol("x"), ctx.symbol("y"));
        let r = linsolve_symbolic(&[&x + &y - 5], &[x.clone(), y.clone()]).unwrap();
        assert!(!r.inconsistent);
        assert_eq!(r.free.len(), 1);
        let f = r.free[0];
        // The free unknown maps to itself; the other is expressed through it.
        assert_eq!(show(&r.values[f]), show([&x, &y][f]));
        let pinned = 1 - f;
        let residual = (&x + &y - 5)
            .subs([&x, &y][pinned], &r.values[pinned])
            .simplify();
        assert_eq!(show(&residual), "0");
    }

    #[test]
    fn symbolic_coefficients_and_rational_answer() {
        let ctx = Context::new();
        let (x, a) = (ctx.symbol("x"), ctx.symbol("a"));
        // 3x = 1  →  1/3 ;  a·x = 1 → 1/a (a assumed nonzero as pivot)
        let r = linsolve_symbolic(&[&x * 3 - 1], std::slice::from_ref(&x)).unwrap();
        assert_eq!(show(&r.values[0]), "1/3");
        let r = linsolve_symbolic(&[&a * &x - 1], std::slice::from_ref(&x)).unwrap();
        let at2 = r.values[0].subs(&a, &ctx.int(2)).eval();
        assert_eq!(show(&at2), "1/2");
    }

    #[test]
    fn empty_input_is_invalid_argument() {
        let ctx = Context::new();
        let x = ctx.symbol("x");
        assert!(matches!(
            linsolve_symbolic(&[], std::slice::from_ref(&x)),
            Err(SymplexError::InvalidArgument { .. })
        ));
        assert!(matches!(
            linsolve_symbolic(&[&x - 1], &[]),
            Err(SymplexError::InvalidArgument { .. })
        ));
    }
}
