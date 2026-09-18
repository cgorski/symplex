//! Polynomial system solving via Gröbner bases.
//!
//! Pipeline: grevlex Buchberger → FGLM to lex → back-substitution.
//!
//! This module also re-exports the other system solvers so that they are
//! reachable from one place: [`linsolve`] / [`linsolve_matrix`] for linear
//! systems and [`solve_numeric_system`] for Newton iteration.

pub use crate::api::expr_solve_ext::{
    GeneralSolution, LinearSolution, NewtonOpts, ZeroForm, linsolve, linsolve_matrix,
    solve_numeric_system, solve_numeric_system_with,
};

use crate::base::arena::Arena;
use crate::poly::groebner;
use crate::poly::multipoly::*;
use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};

use crate::api::expr::Ex;
use crate::base::errors::SymplexError;
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
/// Converts each expression to a multivariate polynomial over ℚ, computes
/// a lex Gröbner basis (grevlex Buchberger + FGLM), and back-substitutes
/// through its triangular structure.  The univariate polynomial in the
/// last variable is solved with the full univariate solver (exact
/// radicals through degree 4, all roots of `xⁿ = c`, `RootOf` beyond),
/// each root is substituted into the remaining basis elements, and the
/// process repeats for the next variable — so **algebraic** (not just
/// rational) solutions are found.  Every value is simplified.
///
/// Linear systems are delegated to [`linsolve`].
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
/// the system is inconsistent (no solutions).
///
/// # Errors
///
/// - [`SymplexError::ComputationFailed`] if an expression is not
///   polynomial in the given variables.
/// - [`SymplexError::InfiniteSolutions`] if the ideal is
///   positive-dimensional (infinitely many solutions), e.g. an
///   under-determined system.
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
/// // Circle ∩ line: x² + y² = 1, y = x  →  (±1/√2, ±1/√2)
/// let solutions = solve_system_ex(
///     &[&x.powi(2) + &y.powi(2) - 1, &y - &x],
///     &[x.clone(), y.clone()],
/// ).unwrap();
/// assert_eq!(solutions.len(), 2);
/// for sol in &solutions {
///     let v = sol[0].eval_f64().unwrap();
///     assert!((v.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
///     assert_eq!(sol[0], sol[1]);
/// }
/// ```
pub fn solve_system_ex(eqs: &[Ex], vars: &[Ex]) -> Result<Vec<Vec<Ex>>, SymplexError> {
    if eqs.is_empty() || vars.is_empty() {
        return Ok(vec![vec![]]);
    }

    let first = &eqs[0];
    let num_vars = vars.len();

    // Build variable ExprId → column-index map (validating contexts).
    let var_ids: Vec<ExprId> = vars.iter().map(|v| first.checked_id(v)).collect();
    let eq_ids: Vec<ExprId> = eqs.iter().map(|e| first.checked_id(e)).collect();
    let var_map: FxHashMap<ExprId, usize> =
        var_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Convert each equation to MultiPoly<GrevLex>.
    let mut polys = Vec::with_capacity(eqs.len());
    {
        let inner = first.inner.read();
        let arena = &inner.arena;
        for &eq_id in &eq_ids {
            match expr_to_multipoly(arena, eq_id, num_vars, &var_map) {
                Some(p) => polys.push(p),
                None => {
                    let shown = arena.display(eq_id).to_string();
                    return Err(SymplexError::ComputationFailed {
                        operation: "solve_system_ex",
                        reason: format!(
                            "expression is not polynomial in the given variables: {shown}"
                        ),
                    });
                }
            }
        }
    }

    let nonzero: Vec<MultiPoly<GrevLex>> = polys.into_iter().filter(|p| !p.is_zero()).collect();
    if nonzero.is_empty() {
        return Err(SymplexError::InfiniteSolutions {
            operation: "solve_system_ex",
            reason: "every equation is identically zero".into(),
        });
    }

    // Linear systems: delegate to the symbolic linear solver.
    if nonzero.iter().all(|p| p.total_degree().unwrap_or(0) <= 1) {
        return match linsolve(eqs, vars)? {
            LinearSolution::Unique(pairs) => Ok(vec![pairs.into_iter().map(|(_, v)| v).collect()]),
            LinearSolution::Inconsistent => Ok(vec![]),
            LinearSolution::Parametric { free, .. } => {
                let names: Vec<String> = free.iter().map(|f| format!("{f}")).collect();
                Err(SymplexError::InfiniteSolutions {
                    operation: "solve_system_ex",
                    reason: format!(
                        "linear system is under-determined (free variables: {})",
                        names.join(", ")
                    ),
                })
            }
        };
    }

    // Gröbner basis (grevlex) for structural checks.
    let grevlex_gb = groebner::groebner_basis(&nonzero);
    if grevlex_gb.is_empty() {
        return Err(SymplexError::InfiniteSolutions {
            operation: "solve_system_ex",
            reason: "trivial ideal: system is under-determined".into(),
        });
    }
    if grevlex_gb.iter().any(|p| p.total_degree() == Some(0)) {
        return Ok(vec![]); // 1 ∈ ideal ⇒ inconsistent
    }
    if !groebner::is_zero_dimensional(&grevlex_gb) {
        return Err(SymplexError::InfiniteSolutions {
            operation: "solve_system_ex",
            reason: "ideal is positive-dimensional: infinitely many solutions".into(),
        });
    }

    // Lex basis for triangular back-substitution.
    let lex_gb = groebner::groebner_basis_lex(&nonzero);
    if lex_gb.is_empty() {
        return Ok(vec![]);
    }

    let solutions = {
        let mut guard = first.inner.write();
        let arena = &mut guard.arena;
        let basis_exprs: Vec<ExprId> = lex_gb
            .iter()
            .map(|p| multipoly_to_expr(arena, p, &var_ids))
            .collect();
        solve_triangular_symbolic(arena, &basis_exprs, &var_ids)
    };

    // Wrap and simplify.
    let result: Vec<Vec<Ex>> = solutions
        .into_iter()
        .map(|sol| {
            sol.into_iter()
                .map(|id| first.wrap(id).eval().simplify())
                .collect()
        })
        .collect();
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════
// Symbolic triangular back-substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Decide whether an expression (free of the solve variables) is zero.
///
/// `Some(true)` — provably zero; `Some(false)` — provably nonzero;
/// `None` — undecidable (symbolic parameters remain).
fn constant_is_zero(arena: &mut Arena, e: ExprId) -> Option<bool> {
    let e1 = crate::transforms::eval::eval(arena, e);
    if arena.is_zero_structural(e1) {
        return Some(true);
    }
    if let Some(r) = arena.as_num(e1) {
        return Some(r.is_zero());
    }
    let e2 = crate::transforms::expand::expand(arena, e1);
    let e2 = crate::transforms::eval::eval(arena, e2);
    if arena.is_zero_structural(e2) {
        return Some(true);
    }
    if let Some(r) = arena.as_num(e2) {
        return Some(r.is_zero());
    }
    if crate::base::walk::free_symbols(arena, e2).is_empty()
        && let Ok(s) = crate::transforms::evalf::evalf(arena, e2, 20)
        && let Some(mag) = crate::transforms::solve::parse_evalf_magnitude(&s)
    {
        return Some(mag < 1e-10);
    }
    None
}

/// Substitute `var := value` into `exprs`, evaluate, and drop the ones
/// that become zero.  Returns `None` if some expression becomes a
/// provably **nonzero** constant (the branch is inconsistent).
fn substitute_and_filter(
    arena: &mut Arena,
    exprs: &[ExprId],
    var: ExprId,
    value: ExprId,
    remaining_vars: &[ExprId],
) -> Option<Vec<ExprId>> {
    let mut out = Vec::with_capacity(exprs.len());
    for &e in exprs {
        let s = crate::transforms::subs::subs(arena, e, var, value);
        let s = crate::transforms::eval::eval(arena, s);
        let s = crate::transforms::expand::expand(arena, s);
        let s = crate::transforms::eval::eval(arena, s);
        let involves_var = remaining_vars
            .iter()
            .any(|&v| crate::base::walk::contains(arena, s, v));
        if involves_var {
            out.push(s);
        } else {
            match constant_is_zero(arena, s) {
                Some(true) => {}
                Some(false) => return None,
                None => {} // undecidable constant: assume consistent
            }
        }
    }
    Some(out)
}

/// Symbolic triangular back-substitution through (the expression form of)
/// a lex Gröbner basis.
///
/// Solves for the **last** variable first: picks the lowest-degree
/// polynomial that involves only that variable, solves it with the
/// univariate solver, checks each root against the other univariate
/// constraints, substitutes it everywhere, and recurses on the remaining
/// variables.  If no purely univariate polynomial is available (after
/// earlier substitutions made one vanish) the lowest-degree polynomial in
/// that variable is solved with symbolic coefficients and the result is
/// back-substituted once the earlier variables are known.
fn solve_triangular_symbolic(
    arena: &mut Arena,
    polys: &[ExprId],
    var_ids: &[ExprId],
) -> Vec<Vec<ExprId>> {
    if var_ids.is_empty() {
        // Consistent iff every remaining constraint is zero.
        for &p in polys {
            if constant_is_zero(arena, p) == Some(false) {
                return vec![];
            }
        }
        return vec![vec![]];
    }
    let last_idx = var_ids.len() - 1;
    let last = var_ids[last_idx];
    let earlier = &var_ids[..last_idx];

    // Polynomials that involve `last` and none of the earlier variables.
    let mut univariate: Vec<(usize, ExprId)> = Vec::new();
    let mut with_last: Vec<(usize, ExprId)> = Vec::new();
    for &p in polys {
        if !crate::base::walk::contains(arena, p, last) {
            continue;
        }
        let deg = arena.degree_of(p, last).unwrap_or(usize::MAX);
        if earlier
            .iter()
            .any(|&v| crate::base::walk::contains(arena, p, v))
        {
            with_last.push((deg, p));
        } else {
            univariate.push((deg, p));
        }
    }

    if let Some(&(_, pivot)) = univariate.iter().min_by_key(|(d, _)| *d) {
        let roots = match crate::transforms::solve::solve_classified(arena, pivot, last) {
            crate::transforms::solve::SolveOutcome::Solutions(s) => s,
            _ => return vec![],
        };
        let mut all = Vec::new();
        for root in roots {
            // The root must satisfy every other univariate constraint.
            let others: Vec<ExprId> = univariate
                .iter()
                .filter(|(_, p)| *p != pivot)
                .map(|(_, p)| *p)
                .collect();
            if substitute_and_filter(arena, &others, last, root.value, &[]).is_none() {
                continue;
            }
            let rest: Vec<ExprId> = polys
                .iter()
                .copied()
                .filter(|p| !univariate.iter().any(|(_, u)| u == p))
                .collect();
            let Some(reduced) = substitute_and_filter(arena, &rest, last, root.value, earlier)
            else {
                continue;
            };
            for mut sub in solve_triangular_symbolic(arena, &reduced, earlier) {
                sub.push(root.value);
                all.push(sub);
            }
        }
        return all;
    }

    if let Some(&(_, pivot)) = with_last.iter().min_by_key(|(d, _)| *d) {
        // Solve for `last` with coefficients depending on earlier variables.
        let roots = match crate::transforms::solve::solve_classified(arena, pivot, last) {
            crate::transforms::solve::SolveOutcome::Solutions(s) if !s.is_empty() => s,
            _ => return vec![],
        };
        let mut all = Vec::new();
        for root in roots {
            let rest: Vec<ExprId> = polys.iter().copied().filter(|&p| p != pivot).collect();
            let Some(reduced) = substitute_and_filter(arena, &rest, last, root.value, earlier)
            else {
                continue;
            };
            for sub in solve_triangular_symbolic(arena, &reduced, earlier) {
                // Back-substitute the earlier variables into the root.
                let mut val = root.value;
                for (i, &v) in earlier.iter().enumerate() {
                    val = crate::transforms::subs::subs(arena, val, v, sub[i]);
                }
                let val = crate::transforms::eval::eval(arena, val);
                let mut full = sub;
                full.push(val);
                all.push(full);
            }
        }
        return all;
    }

    // `last` is unconstrained here — the zero-dimensional check should
    // prevent this; treat as no solution rather than inventing one.
    vec![]
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
