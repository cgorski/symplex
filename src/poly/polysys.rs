//! Polynomial system solving via Gröbner bases.
//!
//! Pipeline: grevlex Buchberger → FGLM to lex → back-substitution.

use crate::base::arena::Arena;
use crate::poly::groebner;
use crate::poly::multipoly::*;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};

use crate::api::expr::Ex;
use crate::base::node::{ExprId, ExprNode};
use crate::base::walk;
use rustc_hash::FxHashMap;

/// Solve a system of multivariate polynomial equations over ℚ.
///
/// Given polynomials p₁, …, pₖ in ℚ[x₀, …, x_{n-1}], finds all common
/// rational roots: points (a₀, …, a_{n-1}) ∈ ℚⁿ with pᵢ(a) = 0 for all i.
///
/// # Algorithm
///
/// 1. Compute a grevlex Gröbner basis (Buchberger).
/// 2. Convert to lex ordering via FGLM (requires zero-dimensional ideal).
/// 3. Back-substitute using the triangular structure of the lex basis.
///
/// # Returns
///
/// - `Ok(solutions)` where each solution is a vector `[x₀, x₁, …, x_{n-1}]`.
///   An empty vec means the system is inconsistent (no rational solutions).
/// - `Err(msg)` if the ideal is not zero-dimensional (infinitely many solutions).
pub fn solve_polynomial_system(
    polys: &[MultiPoly<GrevLex>],
) -> Result<Vec<Vec<Ratio<BigInt>>>, String> {
    if polys.is_empty() {
        return Ok(vec![vec![]]);
    }

    let nonzero: Vec<_> = polys.iter().filter(|p| !p.is_zero()).cloned().collect();
    if nonzero.is_empty() {
        return Ok(vec![vec![]]);
    }

    let num_vars = nonzero[0].num_vars();
    if num_vars == 0 {
        // Constant polynomials — check if any is nonzero
        for p in &nonzero {
            if !p.is_zero() {
                return Ok(vec![]); // inconsistent
            }
        }
        return Ok(vec![vec![]]);
    }

    // Step 1: Compute grevlex Gröbner basis
    let grevlex_gb = groebner::groebner_basis(&nonzero);

    if grevlex_gb.is_empty() {
        // Ideal is {0}, every point is a solution — not zero-dimensional
        return Err("trivial ideal: system is underdetermined".into());
    }

    // Step 2: Check for inconsistency (basis contains a nonzero constant)
    for p in &grevlex_gb {
        if let Some(0) = p.total_degree() {
            // Nonzero constant in basis → ideal = whole ring → no solutions
            return Ok(vec![]);
        }
    }

    // Step 3: Check zero-dimensionality
    if !groebner::is_zero_dimensional(&grevlex_gb) {
        return Err("ideal is not zero-dimensional: infinitely many solutions".into());
    }

    // Step 4: FGLM to lex ordering
    let lex_gb = groebner::groebner_basis_lex(&nonzero.to_vec());

    if lex_gb.is_empty() {
        return Ok(vec![]);
    }

    // Step 5: Back-substitute
    solve_triangular(&lex_gb, num_vars)
}

/// Back-substitute through a lex Gröbner basis to find all rational solutions.
///
/// In lex order with x₀ > x₁ > … > x_{n-1}, the basis has a triangular
/// structure. The last element is univariate in x_{n-1}. We find its rational
/// roots, substitute each back, and recurse.
fn solve_triangular(
    basis: &[MultiPoly<Lex>],
    num_vars: usize,
) -> Result<Vec<Vec<Ratio<BigInt>>>, String> {
    if num_vars == 0 {
        return Ok(vec![vec![]]);
    }
    if basis.is_empty() {
        // No constraints — underdetermined for this sub-problem.
        // Return empty; a zero-dimensional system shouldn't reach here.
        return Ok(vec![vec![]]);
    }

    let last_var = num_vars - 1;

    // Find a polynomial that is univariate in the last variable.
    // In lex order, such a polynomial involves only x_{last_var}.
    let univariate = basis.iter().find(|p| {
        !p.is_zero()
            && p.terms().all(|(exp, _)| {
                exp.iter()
                    .enumerate()
                    .all(|(i, &e)| i == last_var || e == 0)
            })
    });

    let univariate = match univariate {
        Some(u) => u,
        None => {
            return Err(format!(
                "no univariate polynomial found in variable {last_var} for back-substitution"
            ));
        }
    };

    // Find rational roots of the univariate polynomial
    let roots = rational_roots_of_univariate(univariate, last_var);

    let mut all_solutions = Vec::new();

    for root in &roots {
        if num_vars == 1 {
            // Base case: single variable
            all_solutions.push(vec![root.clone()]);
        } else {
            // Substitute this root into all basis elements.
            // MultiPoly::substitute removes the variable dimension and shifts
            // higher indices down. Since last_var is the last index, no
            // shifting occurs for variables 0..last_var-1.
            let reduced: Vec<MultiPoly<Lex>> = basis
                .iter()
                .map(|p| p.substitute(last_var, root))
                .filter(|p| !p.is_zero())
                .collect();

            match solve_triangular(&reduced, num_vars - 1) {
                Ok(sub_sols) => {
                    for mut sol in sub_sols {
                        // sol has values for variables 0..last_var-1
                        // Append the root for variable last_var
                        sol.push(root.clone());
                        all_solutions.push(sol);
                    }
                }
                Err(_) => {
                    // This root doesn't extend to a full solution; skip.
                    continue;
                }
            }
        }
    }

    Ok(all_solutions)
}

/// Extract rational roots from a multivariate polynomial that is known to be
/// univariate in `var_idx` (all other variables have exponent 0).
fn rational_roots_of_univariate(poly: &MultiPoly<Lex>, var_idx: usize) -> Vec<Ratio<BigInt>> {
    // Determine the maximum degree in var_idx
    let mut max_deg: u32 = 0;
    for (exp, _) in poly.terms() {
        max_deg = max_deg.max(exp[var_idx]);
    }

    // Build a coefficient vector indexed by degree (ascending)
    let mut coeffs = vec![Ratio::<BigInt>::zero(); max_deg as usize + 1];
    for (exp, coeff) in poly.terms() {
        let deg = exp[var_idx] as usize;
        coeffs[deg] = coeffs[deg].clone() + coeff.clone();
    }

    let uni = crate::poly::Poly::from_coeffs(coeffs);
    rational_roots_of_poly(&uni)
}

/// Find all rational roots of a univariate polynomial using the rational root
/// theorem.
///
/// If p/q is a rational root of a polynomial with integer coefficients, then
/// p divides the constant term and q divides the leading coefficient.
fn rational_roots_of_poly(poly: &crate::poly::Poly) -> Vec<Ratio<BigInt>> {
    if poly.is_zero() {
        return vec![];
    }

    // Degree 0 → constant, no roots (unless zero, handled above)
    if poly.is_constant() {
        return vec![];
    }

    let mut roots = Vec::new();

    // Check if 0 is a root
    if poly.eval(&Ratio::zero()).is_zero() {
        roots.push(Ratio::zero());
    }

    // Work with the primitive part to get integer coefficients
    let prim = poly.primitive_part();

    let const_term = prim.eval(&Ratio::zero());
    let lc = match prim.leading_coeff() {
        Some(c) => c.clone(),
        None => return roots,
    };

    // Extract integer values (they should be integers after primitive_part)
    let const_abs = const_term.numer().to_i64().map(|n| n.abs()).unwrap_or(0);
    let lc_abs = lc.numer().to_i64().map(|n| n.abs()).unwrap_or(1);

    if const_abs == 0 {
        // 0 is already handled above; p | 0 is anything, so we'd test all q-divisors.
        // Just return what we have—0 is the root from the constant term being 0.
        // But there could be other roots too, so fall through with const_abs = 0 handled.
        // We need to factor out x and check the quotient.
        let x_poly = crate::poly::Poly::from_coeffs(vec![Ratio::zero(), Ratio::one()]);
        let (quotient, _) = poly.div_rem(&x_poly);
        let mut other_roots = rational_roots_of_poly(&quotient);
        // Merge, avoiding duplicates
        for r in other_roots.drain(..) {
            if !roots.contains(&r) {
                roots.push(r);
            }
        }
        return roots;
    }

    // Cap divisor enumeration to avoid combinatorial explosion
    let const_for_divs = if const_abs > super::MAX_DIVISOR_COEFFICIENT as i64 {
        tracing::debug!(
            "polysys: coefficient {} exceeds divisor cap {}; truncating for rational root search",
            const_abs,
            super::MAX_DIVISOR_COEFFICIENT
        );
        super::MAX_DIVISOR_COEFFICIENT as i64
    } else {
        const_abs
    };
    let lc_for_divs = if lc_abs > super::MAX_DIVISOR_COEFFICIENT as i64 {
        tracing::debug!(
            "polysys: leading coefficient {} exceeds divisor cap {}; truncating for rational root search",
            lc_abs,
            super::MAX_DIVISOR_COEFFICIENT
        );
        super::MAX_DIVISOR_COEFFICIENT as i64
    } else {
        lc_abs
    };

    let p_divs = divisors(const_for_divs);
    let q_divs = divisors(lc_for_divs);

    for &p in &p_divs {
        for &q in &q_divs {
            for &sign in &[1i64, -1] {
                let candidate = Ratio::new(BigInt::from(sign * p), BigInt::from(q));
                if !roots.contains(&candidate) && poly.eval(&candidate).is_zero() {
                    roots.push(candidate);
                }
            }
        }
    }

    roots
}

/// Return all positive divisors of `n` (including 1 and n itself).
///
/// Returns `[1]` for n = 0.
fn divisors(n: i64) -> Vec<i64> {
    if n == 0 {
        return vec![1];
    }
    let n = n.abs();
    let mut result = Vec::new();
    let mut d = 1i64;
    while d * d <= n {
        if n % d == 0 {
            result.push(d);
            if d != n / d {
                result.push(n / d);
            }
        }
        d += 1;
    }
    result.sort();
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic expression → MultiPoly bridge
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a symbolic expression to a multivariate polynomial.
///
/// `var_map` maps each variable's [`ExprId`] to its column index in the
/// exponent vector.  Returns `None` if the expression is not polynomial
/// in the listed variables (e.g. contains `sin`, fractional powers, or
/// symbols not in `var_map`).
fn expr_to_multipoly(
    arena: &crate::base::arena::Arena,
    expr: ExprId,
    num_vars: usize,
    var_map: &FxHashMap<ExprId, usize>,
) -> Option<MultiPoly<GrevLex>> {
    // Trivial case: the expression is one of the variables.
    if let Some(&idx) = var_map.get(&expr) {
        return Some(MultiPoly::var(num_vars, idx));
    }

    let post_order = walk::post_order_ids(arena, expr);
    let mut cache: FxHashMap<ExprId, MultiPoly<GrevLex>> = FxHashMap::default();

    for &id in &post_order {
        let poly = convert_node_multi(arena, id, num_vars, var_map, &cache)?;
        cache.insert(id, poly);
    }

    cache.remove(&expr)
}

/// Convert a single expression node to a [`MultiPoly`], using previously
/// converted children from `cache`.
fn convert_node_multi(
    arena: &crate::base::arena::Arena,
    id: ExprId,
    num_vars: usize,
    var_map: &FxHashMap<ExprId, usize>,
    cache: &FxHashMap<ExprId, MultiPoly<GrevLex>>,
) -> Option<MultiPoly<GrevLex>> {
    // The variable itself.
    if let Some(&idx) = var_map.get(&id) {
        return Some(MultiPoly::var(num_vars, idx));
    }

    let node = arena.node(id);

    match node {
        // Numeric literal → constant polynomial.
        ExprNode::Num(nid) => {
            let r = arena.num(*nid).clone();
            Some(MultiPoly::constant(num_vars, r))
        }

        // Symbol not in var_map — cannot represent as polynomial.
        ExprNode::Symbol(_) => None,

        // Add: sum of child polynomials.
        ExprNode::Add(children) => {
            let mut result = MultiPoly::zero(num_vars);
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = result.add(child_poly);
            }
            Some(result)
        }

        // Mul: product of child polynomials.
        ExprNode::Mul(children) => {
            let mut result = MultiPoly::from_int(num_vars, 1);
            for &child in children.iter() {
                let child_poly = cache.get(&child)?;
                result = result.mul(child_poly);
            }
            Some(result)
        }

        // Pow: base^exp where exp must be a non-negative integer constant.
        ExprNode::Pow(base, exp) => {
            let base_poly = cache.get(base)?;

            // Exponent must not be one of the solve-variables.
            if var_map.contains_key(exp) {
                return None;
            }

            let exp_val = match arena.node(*exp) {
                ExprNode::Num(nid) => arena.num(*nid).clone(),
                _ => return None,
            };
            if !exp_val.is_integer() {
                return None;
            }
            let n: i64 = exp_val.to_integer().try_into().ok()?;
            if n < 0 {
                return None;
            }

            let mut result = MultiPoly::from_int(num_vars, 1);
            for _ in 0..n {
                result = result.mul(base_poly);
            }
            Some(result)
        }

        // Neg: negate the child polynomial.
        ExprNode::Neg(inner_id) => {
            let inner_poly = cache.get(inner_id)?;
            let neg_one = MultiPoly::from_int(num_vars, -1);
            Some(neg_one.mul(inner_poly))
        }

        // Everything else is not polynomial.
        _ => None,
    }
}

/// Solve a system of polynomial equations given as symbolic expressions.
///
/// Converts each expression to a multivariate polynomial, computes a
/// Gröbner basis, and finds all rational solutions via back-substitution.
///
/// # Arguments
///
/// * `eqs` — slice of symbolic expressions, each treated as `expr = 0`.
/// * `vars` — slice of symbolic variables to solve for.
///
/// # Returns
///
/// A vector of solution vectors.  Each inner vector has one entry per
/// variable in `vars`, in the same order.  An empty outer vector means
/// the system is inconsistent (no rational solutions).
///
/// # Errors
///
/// Returns `Err` if an expression is not polynomial in the given
/// variables, or if the ideal is not zero-dimensional.
///
/// # Example
///
/// ```
/// use symplex::prelude::*;
/// use symplex::polysys::solve_system_ex;
///
/// let ctx = Context::new();
/// let x = ctx.symbol("x");
/// let y = ctx.symbol("y");
/// // Solve: x + y - 1 = 0  and  x - y = 0
/// let solutions = solve_system_ex(
///     &[&x + &y - 1, &x - &y],
///     &[x, y],
/// ).unwrap();
/// ```
pub fn solve_system_ex(
    eqs: &[Ex],
    vars: &[Ex],
) -> Result<Vec<Vec<Ex>>, crate::base::errors::SymplexError> {
    if eqs.is_empty() || vars.is_empty() {
        return Ok(vec![vec![]]);
    }

    let first = &eqs[0];
    let num_vars = vars.len();

    // Build variable ExprId → column-index map.
    let var_ids: Vec<ExprId> = vars.iter().map(|v| v.raw_id()).collect();
    let var_map: FxHashMap<ExprId, usize> =
        var_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Convert each equation to MultiPoly<GrevLex>.
    let inner = first.inner.read();
    let arena = &inner.arena;

    let mut polys = Vec::with_capacity(eqs.len());
    for eq_expr in eqs {
        match expr_to_multipoly(arena, eq_expr.raw_id(), num_vars, &var_map) {
            Some(p) => polys.push(p),
            None => {
                return Err(crate::base::errors::SymplexError::ComputationFailed {
                    operation: "solve_system_ex",
                    reason: "expression is not polynomial in the given variables".into(),
                });
            }
        }
    }
    drop(inner);

    // Solve via Gröbner basis + back-substitution.
    match solve_polynomial_system(&polys) {
        Ok(solutions) if !solutions.is_empty() => {
            // Convert rational solutions back to Ex.
            let result: Vec<Vec<Ex>> = solutions
                .into_iter()
                .map(|sol| {
                    sol.into_iter()
                        .map(|r| {
                            let mut guard = first.inner.write();
                            let nid = guard.arena.intern_num(r);
                            let id = guard.arena.intern(ExprNode::Num(nid));
                            drop(guard);
                            first.wrap(id)
                        })
                        .collect()
                })
                .collect();
            Ok(result)
        }
        Ok(_empty) => {
            // No rational roots found.  Fall back to symbolic solving
            // which can discover irrational roots (e.g. √(3/2)).
            solve_system_symbolic_fallback(first, &polys, &var_ids, num_vars)
        }
        Err(msg) => Err(crate::base::errors::SymplexError::ComputationFailed {
            operation: "solve_system_ex",
            reason: msg,
        }),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic fallback for irrational roots
// ═══════════════════════════════════════════════════════════════════════════

/// Fallback solver that uses the general symbolic equation solver to find
/// irrational roots when the rational root theorem returns empty.
///
/// Computes a lex Gröbner basis, converts the univariate polynomials to
/// symbolic expressions, solves them with [`crate::transforms::solve::solve`], then
/// back-substitutes symbolically through the remaining basis elements.
fn solve_system_symbolic_fallback(
    first: &Ex,
    polys: &[MultiPoly<GrevLex>],
    var_ids: &[ExprId],
    num_vars: usize,
) -> Result<Vec<Vec<Ex>>, crate::base::errors::SymplexError> {
    // Compute lex Gröbner basis.
    let lex_gb = groebner::groebner_basis_lex(polys);

    if lex_gb.is_empty() {
        return Ok(vec![]);
    }

    // Perform symbolic triangular back-substitution under a write lock.
    let mut guard = first.inner.write();
    let arena = &mut guard.arena;

    let solutions = solve_triangular_symbolic(arena, &lex_gb, var_ids, num_vars);
    drop(guard);

    // Wrap ExprId solutions as Ex.
    let result: Vec<Vec<Ex>> = solutions
        .into_iter()
        .map(|sol| sol.into_iter().map(|id| first.wrap(id)).collect())
        .collect();
    Ok(result)
}

/// Symbolic triangular back-substitution through a lex Gröbner basis.
///
/// Like [`solve_triangular`] but uses the general symbolic solver instead
/// of the rational root theorem, enabling discovery of irrational roots.
fn solve_triangular_symbolic(
    arena: &mut Arena,
    basis: &[MultiPoly<Lex>],
    var_ids: &[ExprId],
    num_vars: usize,
) -> Vec<Vec<ExprId>> {
    if num_vars == 0 || basis.is_empty() {
        return vec![vec![]];
    }

    let last_var_idx = num_vars - 1;
    let last_var_id = var_ids[last_var_idx];

    // Find a polynomial that is univariate in the last variable.
    let univariate = basis.iter().find(|p| {
        !p.is_zero()
            && p.terms().all(|(exp, _)| {
                exp.iter()
                    .enumerate()
                    .all(|(i, &e)| i == last_var_idx || e == 0)
            })
    });

    let univariate = match univariate {
        Some(u) => u,
        None => return vec![],
    };

    // Convert univariate MultiPoly to a symbolic expression and solve.
    let expr_id = univariate_multipoly_to_expr(arena, univariate, last_var_id, last_var_idx);
    let roots = crate::transforms::solve::solve(arena, expr_id, last_var_id);

    if roots.is_empty() {
        return vec![];
    }

    let mut all_solutions = Vec::new();

    for root in &roots {
        if num_vars == 1 {
            all_solutions.push(vec![root.value]);
        } else {
            // Convert remaining basis elements to symbolic expressions,
            // substitute the root for the last variable, and recurse.
            let mut reduced_exprs = Vec::new();
            for p in basis {
                let poly_expr = multipoly_to_expr(arena, p, var_ids);
                let subst =
                    crate::transforms::subs::subs(arena, poly_expr, last_var_id, root.value);
                if !arena.is_zero_structural(subst) {
                    reduced_exprs.push(subst);
                }
            }

            let sub_solutions =
                solve_remaining_symbolic(arena, &reduced_exprs, &var_ids[..last_var_idx]);

            for mut sub_sol in sub_solutions {
                sub_sol.push(root.value);
                all_solutions.push(sub_sol);
            }
        }
    }

    all_solutions
}

/// Recursively solve a set of symbolic expressions for the given variables.
///
/// Solves each variable from last to first.  For each solvable expression,
/// substitutes the found root into the remaining expressions and recurses.
fn solve_remaining_symbolic(
    arena: &mut Arena,
    exprs: &[ExprId],
    var_ids: &[ExprId],
) -> Vec<Vec<ExprId>> {
    if var_ids.is_empty() {
        return vec![vec![]];
    }
    if exprs.is_empty() {
        return vec![vec![]];
    }

    let last_idx = var_ids.len() - 1;
    let last_var = var_ids[last_idx];

    // Try each expression to find one solvable for the last variable.
    for (i, &expr) in exprs.iter().enumerate() {
        // Check that the expression actually involves this variable.
        if !crate::transforms::solve::expr_contains_var_pub(arena, expr, last_var) {
            continue;
        }

        let roots = crate::transforms::solve::solve(arena, expr, last_var);
        if roots.is_empty() {
            continue;
        }

        let mut all_solutions = Vec::new();

        for root in &roots {
            if var_ids.len() == 1 {
                all_solutions.push(vec![root.value]);
            } else {
                // Substitute this root into remaining expressions.
                let mut reduced = Vec::new();
                for (j, &e) in exprs.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let subst = crate::transforms::subs::subs(arena, e, last_var, root.value);
                    if !arena.is_zero_structural(subst) {
                        reduced.push(subst);
                    }
                }

                let sub_sols = solve_remaining_symbolic(arena, &reduced, &var_ids[..last_idx]);

                for mut sub_sol in sub_sols {
                    sub_sol.push(root.value);
                    all_solutions.push(sub_sol);
                }
            }
        }

        return all_solutions;
    }

    vec![]
}

/// Convert a univariate [`MultiPoly<Lex>`] (univariate in `var_idx`) to a
/// symbolic expression in the arena.
fn univariate_multipoly_to_expr(
    arena: &mut Arena,
    poly: &MultiPoly<Lex>,
    var_id: ExprId,
    var_idx: usize,
) -> ExprId {
    let mut terms = Vec::new();
    for (exp, coeff) in poly.terms() {
        let deg = exp[var_idx];
        let coeff_id = ratio_to_expr(arena, coeff);
        if deg == 0 {
            terms.push(coeff_id);
        } else if deg == 1 {
            terms.push(arena.mul(&[coeff_id, var_id]));
        } else {
            let exp_id = arena.int(i64::from(deg));
            let var_pow = arena.pow(var_id, exp_id);
            terms.push(arena.mul(&[coeff_id, var_pow]));
        }
    }
    if terms.is_empty() {
        arena.zero
    } else {
        arena.add(&terms)
    }
}

/// Convert a general [`MultiPoly<Lex>`] to a symbolic expression in the
/// arena, using the given variable ExprIds.
fn multipoly_to_expr(arena: &mut Arena, poly: &MultiPoly<Lex>, var_ids: &[ExprId]) -> ExprId {
    let mut terms = Vec::new();
    for (exp, coeff) in poly.terms() {
        let coeff_id = ratio_to_expr(arena, coeff);
        let mut factors = vec![coeff_id];
        for (i, &e) in exp.iter().enumerate() {
            if e > 0 && i < var_ids.len() {
                if e == 1 {
                    factors.push(var_ids[i]);
                } else {
                    let exp_id = arena.int(i64::from(e));
                    factors.push(arena.pow(var_ids[i], exp_id));
                }
            }
        }
        terms.push(arena.mul(&factors));
    }
    if terms.is_empty() {
        arena.zero
    } else {
        arena.add(&terms)
    }
}

/// Convert a `Ratio<BigInt>` to an expression in the arena.
fn ratio_to_expr(arena: &mut Arena, r: &Ratio<BigInt>) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Ratio<BigInt> {
        Ratio::from_integer(BigInt::from(n))
    }

    fn ratio(p: i64, q: i64) -> Ratio<BigInt> {
        Ratio::new(BigInt::from(p), BigInt::from(q))
    }

    #[test]
    fn test_divisors_basic() {
        assert_eq!(divisors(1), vec![1]);
        assert_eq!(divisors(6), vec![1, 2, 3, 6]);
        assert_eq!(divisors(0), vec![1]);
    }

    #[test]
    fn test_rational_roots_simple() {
        // x^2 - 4 = 0 → roots ±2
        let poly = crate::poly::Poly::from_coeffs(vec![rat(-4), rat(0), rat(1)]);
        let mut roots = rational_roots_of_poly(&poly);
        roots.sort();
        assert_eq!(roots, vec![rat(-2), rat(2)]);
    }

    #[test]
    fn test_rational_roots_with_rational_root() {
        // 2x - 1 = 0 → root 1/2
        let poly = crate::poly::Poly::from_coeffs(vec![rat(-1), rat(2)]);
        let roots = rational_roots_of_poly(&poly);
        assert_eq!(roots, vec![ratio(1, 2)]);
    }

    #[test]
    fn test_rational_roots_no_rational() {
        // x^2 - 2 = 0 → no rational roots
        let poly = crate::poly::Poly::from_coeffs(vec![rat(-2), rat(0), rat(1)]);
        let roots = rational_roots_of_poly(&poly);
        assert!(roots.is_empty());
    }

    #[test]
    fn test_rational_roots_zero_root() {
        // x^2 - x = x(x-1) = 0 → roots 0, 1
        let poly = crate::poly::Poly::from_coeffs(vec![rat(0), rat(-1), rat(1)]);
        let mut roots = rational_roots_of_poly(&poly);
        roots.sort();
        assert_eq!(roots, vec![rat(0), rat(1)]);
    }

    #[test]
    fn test_solve_linear_single_var() {
        // x - 3 = 0 → x = 3
        let p = MultiPoly::<GrevLex>::var(1, 0).add(&MultiPoly::from_int(1, -3));
        let sols = solve_polynomial_system(&[p]).unwrap();
        assert_eq!(sols.len(), 1);
        assert_eq!(sols[0], vec![rat(3)]);
    }
}
