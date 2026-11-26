//! Variable-coefficient first-order linear: y' + P(x)*y = Q(x) (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Variable-coefficient first-order linear: y' + P(x)*y = Q(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y' + P(x)*y = Q(x) via integrating factor.
///
/// Algorithm:
/// 1. Extract dy/dx from the ODE expression
/// 2. Among remaining terms, find those containing func (y) → these give P(x)*y
/// 3. Terms not containing func give -Q(x)
/// 4. Extract P(x) by dividing the y-containing terms by func
/// 5. Compute integrating factor: μ = exp(∫P(x)dx)
/// 6. Solution: y = (1/μ)·[∫Q(x)·μ dx + C1]
pub(super) fn try_first_order_linear_general(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    first_order_linear(arena, expr, func, var, func_sym, var_sym, false)
}

/// [`try_first_order_linear_general`] when `∫P dx` has no closed form:
/// the integrating factor `e^{∫P dx}` with a formal `Integral`, SymPy's
/// `exp(−Integral(P, x))·(C1 + Integral(Q·exp(Integral(P, x)), x))`
/// (`(1 + x²)·y′ + cos(x)·y = eˣ` was refused).  Tried last, after every
/// method that could find a closed form.
pub(super) fn try_first_order_linear_formal(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    first_order_linear(arena, expr, func, var, func_sym, var_sym, true)
}

/// See [`try_first_order_linear_general`]; `formal`: a non-elementary
/// `∫P dx` stays a formal `Integral` instead of refusing.
fn first_order_linear(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
    formal: bool,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying variable-coefficient first-order linear");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff_rational = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut y_terms: Vec<ExprId> = Vec::new(); // terms that contain func (y)
    let mut free_terms: Vec<ExprId> = Vec::new(); // terms free of func
    // Coefficients a(x) of `y′` that are not plain numbers (`x·y′`).
    let mut dy_coeff_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff_rational += coeff;
        } else if expr_contains(arena, child, dy_dx) {
            // a(x)·y′ with a(x) free of y.
            let ExprNode::Mul(factors) = arena.node(term).clone() else {
                return None;
            };
            let (with_dy, rest): (Vec<ExprId>, Vec<ExprId>) =
                factors.iter().copied().partition(|&f| f == dy_dx);
            if with_dy.len() != 1 || rest.iter().any(|&f| contains_sym(arena, f, func_sym)) {
                return None;
            }
            has_dy = true;
            let c = ode_ratio_to_expr(arena, &coeff);
            let mut parts = vec![c];
            parts.extend(rest);
            dy_coeff_terms.push(arena.mul(&parts));
        } else if contains_sym(arena, child, func_sym) {
            y_terms.push(child);
        } else {
            free_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy {
        return None;
    }
    // A coefficient a(x) of y′ that is not a number: divide through by it.
    let dy_coeff_expr: Option<ExprId> = if dy_coeff_terms.is_empty() {
        None
    } else {
        let mut parts = dy_coeff_terms.clone();
        if !dy_coeff_rational.is_zero() {
            parts.push(ode_ratio_to_expr(arena, &dy_coeff_rational));
        }
        let a = arena.add(&parts);
        let a = crate::transforms::eval::eval(arena, a);
        match arena.as_num(a) {
            Some(r) => {
                dy_coeff_rational = r.clone();
                None
            }
            None => Some(a),
        }
    };
    if dy_coeff_expr.is_none() && dy_coeff_rational.is_zero() {
        return None;
    }

    // We need at least one y-term for this to be a linear ODE with y dependence.
    // If there are no y-terms, it's simple separable.
    if y_terms.is_empty() {
        return None;
    }

    // Check that the y-terms are LINEAR in y: each y-term = something * y
    // For each y-term, try to extract the coefficient of y.
    let mut p_x_terms: Vec<ExprId> = Vec::new();

    for &yt in &y_terms {
        {
            let px = extract_coeff_of_func(arena, yt, func, func_sym, var_sym)?;
            p_x_terms.push(px);
        }
    }

    // Normalize by dy_coeff: divide P(x) and Q(x) by the coefficient of y'
    if let Some(a) = dy_coeff_expr {
        let neg_one = arena.int(-1);
        let inv_a = arena.pow(a, neg_one);
        p_x_terms = p_x_terms
            .iter()
            .map(|&p| {
                let t = arena.mul(&[inv_a, p]);
                crate::transforms::eval::eval(arena, t)
            })
            .collect();
        free_terms = free_terms
            .iter()
            .map(|&f| {
                let t = arena.mul(&[inv_a, f]);
                crate::transforms::eval::eval(arena, t)
            })
            .collect();
    } else if !dy_coeff_rational.is_one() {
        let inv_coeff = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff_rational;
        let inv_id = {
            let nid = arena.intern_num(inv_coeff);
            arena.intern(ExprNode::Num(nid))
        };
        let mut scaled_p = Vec::new();
        for &p in &p_x_terms {
            scaled_p.push(arena.mul(&[inv_id, p]));
        }
        p_x_terms = scaled_p;

        let mut scaled_f = Vec::new();
        for &f in &free_terms {
            scaled_f.push(arena.mul(&[inv_id, f]));
        }
        free_terms = scaled_f;
    }

    // P(x) = sum of p_x_terms
    let p_x = if p_x_terms.is_empty() {
        arena.zero
    } else if p_x_terms.len() == 1 {
        p_x_terms[0]
    } else {
        arena.add(&p_x_terms)
    };

    // Check that P(x) doesn't contain y (it shouldn't at this point, but verify)
    if contains_sym(arena, p_x, func_sym) {
        return None;
    }

    // Try the constant-coefficient path first for efficiency — if P(x) is a
    // pure rational number, delegate to the existing constant-coefficient solver
    // which handles the nonhomogeneous case (y' + a*y = Q(x)).
    // The constant-coefficient solver wants `y′ + a·y + f(x)` with a unit
    // coefficient on `y′`: the normalised equation is built for it (before,
    // `2y′ + y = cos x` was handed over as is, refused there, and not
    // solved at all).
    let p_x = crate::transforms::eval::eval(arena, p_x);
    if arena.as_num(p_x).is_some() {
        let normalised = if dy_coeff_expr.is_none() && dy_coeff_rational.is_one() {
            expr
        } else {
            let py = arena.mul(&[p_x, func]);
            let mut parts = vec![dy_dx, py];
            parts.extend(free_terms.iter().copied());
            let e = arena.add(&parts);
            crate::transforms::eval::eval(arena, e)
        };
        if let Some(r) = try_first_order_linear(arena, normalised, func, var, func_sym, var_sym) {
            return Some(r);
        }
    }

    // Check that P(x) actually depends on x — if it's constant, it would have
    // been caught above (as a numeric). But it might be a symbolic constant.
    // We proceed regardless.

    // Q(x) = -(free_terms)  since expr = y' + P(x)*y + free_terms = 0
    //                        means y' + P(x)*y = -free_terms = Q(x)
    let q_x = if free_terms.is_empty() {
        arena.zero
    } else {
        let sum = arena.add(&free_terms);
        arena.neg(sum)
    };

    // Integrating factor: μ = exp(∫P(x)dx)
    let int_px = crate::transforms::integrate::integrate(arena, p_x, var);

    // Check if integration failed (returned an unevaluated Integral node)
    let mu = if integral_failed(arena, int_px) {
        // Integration of P(x) failed: no closed integrating factor.
        if !formal {
            return None;
        }
        let p_x = crate::transforms::eval::eval(arena, p_x);
        let formal_int = arena.intern(ExprNode::Integral(p_x, var));
        arena.exp(formal_int)
    } else if formal {
        return None; // the closed form is the other method's
    } else {
        exp_of_log_sum(arena, int_px)
    };

    // Solution: y = (1/μ) · [∫ Q(x)·μ dx + C1]
    let neg_one = arena.int(-1);
    let inv_mu = arena.pow(mu, neg_one);

    let c1 = arena.symbol("C1");

    if arena.is_zero_structural(q_x) {
        // Homogeneous: y' + P(x)*y = 0 → y = C1 * exp(-∫P(x)dx)
        let solution = arena.mul(&[c1, inv_mu]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // Nonhomogeneous: y = (1/μ) * [∫ Q(x)*μ dx + C1]
    let integrand = arena.mul(&[q_x, mu]);
    // Simplify products of exponentials before integrating
    let integrand = crate::transforms::eval::eval(arena, integrand);
    let integral = integrate_forms(arena, integrand, var);

    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[inv_mu, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// Try to extract the coefficient of `func` (y) from an expression that should
/// be of the form `P(x) * y`. Returns `Some(P(x))` if the expression is linear
/// in y, `None` otherwise.
pub(super) fn extract_coeff_of_func(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<ExprId> {
    // Case 1: expr is exactly y
    if expr == func {
        return Some(arena.one);
    }

    // Case 2: expr is Neg(y) → coefficient is -1
    if let ExprNode::Neg(inner) = arena.node(expr).clone() {
        if inner == func {
            let neg_one = arena.int(-1);
            return Some(neg_one);
        }
        // Neg(something) — recurse
        if let Some(inner_coeff) = extract_coeff_of_func(arena, inner, func, func_sym, _var_sym) {
            let result = arena.neg(inner_coeff);
            return Some(result);
        }
        return None;
    }

    // Case 3: expr is Mul([...]) containing func exactly once
    if let ExprNode::Mul(ref children) = arena.node(expr).clone() {
        let mut found_y = false;
        let mut other_factors: Vec<ExprId> = Vec::new();
        let mut y_count = 0;

        for &child in children {
            if child == func {
                y_count += 1;
                if y_count > 1 {
                    return None; // y^2 or higher — nonlinear
                }
                found_y = true;
            } else if contains_sym(arena, child, func_sym) {
                // A factor that contains y but isn't y itself — nonlinear
                return None;
            } else {
                other_factors.push(child);
            }
        }

        if found_y {
            let coeff = if other_factors.is_empty() {
                arena.one
            } else if other_factors.len() == 1 {
                other_factors[0]
            } else {
                arena.mul(&other_factors)
            };
            return Some(coeff);
        }
    }

    // Case 4: expr = coeff_num * something_with_y
    // Use as_coeff_term to peel off a numeric coefficient, then check the term
    {
        let (coeff, term) = arena.as_coeff_term(expr);
        if !coeff.is_one()
            && term != expr
            && let Some(inner_coeff) = extract_coeff_of_func(arena, term, func, func_sym, _var_sym)
        {
            let coeff_id = {
                let nid = arena.intern_num(coeff);
                arena.intern(ExprNode::Num(nid))
            };
            let result = arena.mul(&[coeff_id, inner_coeff]);
            return Some(result);
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Exact ODE solver: M(x,y)dx + N(x,y)dy = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Extract M(x,y) and N(x,y) from an ODE expression of the form `M + N·y' = 0`.
///
/// Returns `(M, N)` where M is the sum of terms not containing `dy/dx`
/// and N is the total coefficient of `dy/dx`.
///
/// Returns `None` when the expression cannot be decomposed (e.g. `(y')²`).
pub(super) fn extract_m_n(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
) -> Option<(ExprId, ExprId)> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Handle single-term expressions.
    if expr == dy_dx {
        return Some((arena.zero, arena.one));
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(c) => c,
        _ => return None,
    };

    let mut m_terms: Vec<ExprId> = Vec::new();
    let mut n_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            // Simple numeric coefficient of dy/dx.
            let coeff_id = {
                let nid = arena.intern_num(coeff);
                arena.intern(ExprNode::Num(nid))
            };
            n_terms.push(coeff_id);
        } else if expr_contains(arena, child, dy_dx) {
            // child contains dy/dx inside a product — try to peel it off.
            if let ExprNode::Mul(ref mul_children) = arena.node(child).clone() {
                let mut found_dy = false;
                let mut other_factors: Vec<ExprId> = Vec::new();
                for &mc in mul_children.iter() {
                    if mc == dy_dx && !found_dy {
                        found_dy = true;
                    } else {
                        other_factors.push(mc);
                    }
                }
                if found_dy {
                    let n_factor = match other_factors.len() {
                        0 => arena.one,
                        1 => other_factors[0],
                        _ => arena.mul(&other_factors),
                    };
                    n_terms.push(n_factor);
                } else {
                    return None; // dy/dx in non-simple position
                }
            } else {
                return None;
            }
        } else {
            m_terms.push(child);
        }
    }

    if n_terms.is_empty() {
        return None; // No dy/dx term
    }

    let m_expr = match m_terms.len() {
        0 => arena.zero,
        1 => m_terms[0],
        _ => arena.add(&m_terms),
    };

    let n_expr = match n_terms.len() {
        1 => n_terms[0],
        _ => arena.add(&n_terms),
    };

    Some((m_expr, n_expr))
}

/// Solve an exact first-order ODE: `M(x,y) + N(x,y)·y' = 0`
/// where `∂M/∂y = ∂N/∂x`.
///
/// The potential function F(x,y) satisfying `∂F/∂x = M` and `∂F/∂y = N`
/// is computed as:
///   1. `F = ∫M dx + g(y)`
///   2. `g'(y) = N − ∂(∫M dx)/∂y`
///   3. `g(y) = ∫ g'(y) dy`
///
/// The implicit solution is `F(x,y) = C1`.  When possible the solver
/// also tries to solve explicitly for `y`.
pub(super) fn try_exact_ode(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying exact ODE");

    let (m_expr, n_expr) = extract_m_n(arena, expr, func, var)?;

    // At least one of M, N must depend on the dependent variable for this
    // to be a genuinely exact ODE (otherwise the linear / separable solvers
    // are better suited).
    if !contains_sym(arena, m_expr, func_sym) && !contains_sym(arena, n_expr, func_sym) {
        return None;
    }

    // ── Exactness check: ∂M/∂y = ∂N/∂x ───────────────────────────
    let dm_dy = crate::transforms::diff::diff(arena, m_expr, func);
    let dn_dx = crate::transforms::diff::diff(arena, n_expr, var);

    let diff_check = arena.sub(dm_dy, dn_dx);
    let diff_eval = crate::transforms::eval::eval(arena, diff_check);
    let diff_expanded = crate::transforms::expand::expand(arena, diff_eval);
    let diff_simplified = crate::transforms::eval::eval(arena, diff_expanded);

    if diff_simplified != arena.zero && !rational_zero(arena, diff_simplified) {
        return None; // Not exact
    }

    // ── Build potential function F(x,y) ───────────────────────────
    // Step 1: F_partial = ∫ M dx  (treating y as constant).  A case split
    // on `y` (`∫ y·e^{xy} dx = y·Piecewise(e^{xy}/y if y ≠ 0, x)`) is taken
    // on its generic branch: a potential is needed only off a line (`y·e^{xy}
    // + (x·e^{xy} + 1)·y′ = 0` was refused).
    let integral_m = crate::transforms::integrate::integrate(arena, m_expr, var);
    if integral_failed(arena, integral_m) {
        return None; // Integration of M w.r.t. x failed
    }
    let integral_m = generic_branches(arena, integral_m, var).unwrap_or(integral_m);

    // Step 2: g'(y) = N − ∂(∫M dx)/∂y, free of x.  When `∫M dx` has a
    // `ln|u|` whose partner in `N` is `ln u` (`(y/x + 3x²) + (ln x + 1)·y′`:
    // `g′ = ln x + 1 − ln|x|`), the potential is taken with `ln u`, as SymPy
    // does (it differs by a constant on each interval where `u` keeps its
    // sign).
    let g_prime_of = |arena: &mut Arena, im: ExprId| -> ExprId {
        let d_intm_dy = crate::transforms::diff::diff(arena, im, func);
        let g_prime = arena.sub(n_expr, d_intm_dy);
        let g_prime = crate::transforms::eval::eval(arena, g_prime);
        let g_prime = crate::transforms::expand::expand(arena, g_prime);
        crate::transforms::eval::eval(arena, g_prime)
    };
    let mut integral_m = integral_m;
    let mut g_prime = g_prime_of(arena, integral_m);
    if contains_sym(arena, g_prime, var_sym)
        && let Some(stripped) = ln_abs_to_ln(arena, integral_m)
    {
        let g2 = g_prime_of(arena, stripped);
        if !contains_sym(arena, g2, var_sym) {
            integral_m = stripped;
            g_prime = g2;
        }
    }

    // g'(y) must be free of x.
    if contains_sym(arena, g_prime, var_sym) {
        return None;
    }

    // Step 3: g(y) = ∫ g'(y) dy
    let g_y = crate::transforms::integrate::integrate(arena, g_prime, func);
    if integral_failed(arena, g_y) {
        return None;
    }

    // F(x,y) = ∫M dx + g(y)
    let potential = arena.add(&[integral_m, g_y]);
    let potential = crate::transforms::eval::eval(arena, potential);

    let c1 = arena.symbol("C1");

    // Try to solve F(x,y) = C1 for y explicitly.  Several roots stay
    // implicit here: a later method may find nicer families, and
    // [`dsolve`] takes them all in the end (`explicit_families`).
    let f_minus_c1 = arena.sub(potential, c1);
    let solutions = known_roots(arena, f_minus_c1, func);

    if solutions.len() == 1 {
        return Some(OdeResult {
            solution: solutions[0].value,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // The implicit solution `F(x, y) − C1` (= 0), in the form of the other
    // implicit solutions.  Before, `F` alone was returned: the constant was
    // missing from the answer (`2xy + (x² + 3y²)y′ = 0` gave `y³ + x²y`).
    let implicit = crate::transforms::eval::eval(arena, f_minus_c1);
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}

/// Attempt to find an integrating factor for a non-exact first-order ODE.
///
/// Given `M + N·y' = 0` with `∂M/∂y ≠ ∂N/∂x`:
///
/// 1. **μ = μ(x):**  if `(∂M/∂y − ∂N/∂x) / N` depends only on `x`,
///    then `μ = exp(∫ that dx)`.
/// 2. **μ = μ(y):**  if `(∂N/∂x − ∂M/∂y) / M` depends only on `y`,
///    then `μ = exp(∫ that dy)`.
///
/// After multiplying through by μ the ODE becomes exact and is solved
/// via [`try_exact_ode`].
pub(super) fn try_integrating_factor_ode(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying integrating factor for non-exact ODE");

    let (m_expr, n_expr) = extract_m_n(arena, expr, func, var)?;

    // Need y-dependence for this to be meaningful.
    if !contains_sym(arena, m_expr, func_sym) && !contains_sym(arena, n_expr, func_sym) {
        return None;
    }

    let dm_dy = crate::transforms::diff::diff(arena, m_expr, func);
    let dn_dx = crate::transforms::diff::diff(arena, n_expr, var);

    let diff_mn = arena.sub(dm_dy, dn_dx); // ∂M/∂y − ∂N/∂x
    let diff_eval = crate::transforms::eval::eval(arena, diff_mn);
    let diff_expanded = crate::transforms::expand::expand(arena, diff_eval);
    let diff_simplified = crate::transforms::eval::eval(arena, diff_expanded);

    if diff_simplified == arena.zero || rational_zero(arena, diff_simplified) {
        // Already exact — delegate.
        return try_exact_ode(arena, expr, func, var, func_sym, var_sym);
    }

    // ── Try μ(x): (∂M/∂y − ∂N/∂x) / N free of y ─────────────────
    {
        let ratio = arena.div(diff_simplified, n_expr);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = crate::transforms::expand::expand(arena, ratio);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        // Multivariate normal form: the univariate `cancel` in `x` could
        // not cancel `x + 2y` from `(x + 2y)/x² · x/(x + 2y)` (coefficients
        // in `y`), so μ = x of `(2x + y)/x + ((2y + x)/x)·y′` was missed.
        let ratio_cancelled = crate::simplify::ratsimp::ratsimp(arena, ratio);
        let ratio_cancelled = crate::transforms::eval::eval(arena, ratio_cancelled);

        if !contains_sym(arena, ratio_cancelled, func_sym) {
            let int_ratio = crate::transforms::integrate::integrate(arena, ratio_cancelled, var);
            if !integral_failed(arena, int_ratio) {
                let mu = exp_of_log_sum(arena, int_ratio);

                // New M' = μ·M,  N' = μ·N
                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::transforms::eval::eval(arena, new_expr);

                if let Some(result) = try_exact_ode(arena, new_expr, func, var, func_sym, var_sym) {
                    return Some(result);
                }
            }
        }
    }

    // ── Try μ(y): (∂N/∂x − ∂M/∂y) / M free of x ─────────────────
    {
        let neg_diff = arena.neg(diff_simplified); // ∂N/∂x − ∂M/∂y
        let ratio = arena.div(neg_diff, m_expr);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio = crate::transforms::expand::expand(arena, ratio);
        let ratio = crate::transforms::eval::eval(arena, ratio);
        let ratio_cancelled = crate::simplify::ratsimp::ratsimp(arena, ratio);
        let ratio_cancelled = crate::transforms::eval::eval(arena, ratio_cancelled);

        if !contains_sym(arena, ratio_cancelled, var_sym) {
            let int_ratio = crate::transforms::integrate::integrate(arena, ratio_cancelled, func);
            if !integral_failed(arena, int_ratio) {
                let mu = exp_of_log_sum(arena, int_ratio);

                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::transforms::eval::eval(arena, new_expr);

                if let Some(result) = try_exact_ode(arena, new_expr, func, var, func_sym, var_sym) {
                    return Some(result);
                }
            }
        }
    }

    None
}

// Check if an expression contains a specific symbol.
// ═══════════════════════════════════════════════════════════════════════════
// Homogeneous coefficient ODE: y' = f(y/x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a first-order ODE of the form y' = f(y/x).
///
/// Detection: substitute y = v*x in the RHS.  If the result simplifies to a
/// function of v alone (no x), the equation is homogeneous of degree 0.
///
/// Solution via v = y/x:
///   y = v*x  →  y' = v + x*v'
///   v + x*v' = f(v)  →  dv/(f(v) - v) = dx/x
///   ∫ dv/(f(v) - v) = ln|x| + C1
///   Back-substitute v = y/x.
pub(super) fn try_homogeneous_coefficient(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying homogeneous coefficient");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must be first-order only
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    // Extract the RHS: expr = dy/dx + ... = 0  →  dy/dx = -...
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut other_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff += coeff;
        } else {
            other_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy || dy_coeff.is_zero() {
        return None;
    }

    // RHS of y' = RHS (negate the other terms, normalize by dy_coeff)
    let rhs = if other_terms.is_empty() {
        return None;
    } else {
        let sum = arena.add(&other_terms);
        arena.neg(sum)
    };

    let rhs = if !dy_coeff.is_one() {
        let inv_id = ode_ratio_to_expr(
            arena,
            &(num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff),
        );
        let s = arena.mul(&[inv_id, rhs]);
        crate::transforms::eval::eval(arena, s)
    } else {
        rhs
    };

    // The RHS must depend on both x and y for this to be interesting.
    if !contains_sym(arena, rhs, func_sym) || !contains_sym(arena, rhs, var_sym) {
        return None;
    }

    // For a degree-0 homogeneous function f(x,y), f(tx, ty) = f(x,y).
    // In particular f(x, y) = f(1, y/x).  So f(v) = RHS|_{y→v, x→1}.
    //
    // Detection: substitute y→v, x→1.  If the result is free of x, the
    // equation is homogeneous of degree 0.
    //
    // This avoids the need to simplify (v*x)^n / x^n etc.
    let v = arena.internal_symbol("__v");

    // Compute f(v) = RHS(x=1, y=v)
    let rhs_sub = crate::transforms::subs::subs(arena, rhs, func, v);
    let rhs_sub = crate::transforms::subs::subs(arena, rhs_sub, var, arena.one);
    let rhs_sub = crate::transforms::eval::eval(arena, rhs_sub);
    let rhs_sub = crate::transforms::expand::expand(arena, rhs_sub);
    let rhs_sub = crate::transforms::eval::eval(arena, rhs_sub);

    // f(v) must be free of x (it should be, since we set x=1).
    if contains_sym(arena, rhs_sub, var_sym) {
        return None;
    }

    // Verify homogeneity of degree 0: RHS(x, v·x) must reduce to f(v).
    // Without this check any polynomial RHS (e.g. y² + x², degree 2)
    // would be misclassified.  The difference is brought to the
    // multivariate rational normal form (`ratsimp`): the univariate
    // `cancel` in `x` it used could not cancel `x` from `(3vx − x)/(vx + x)`
    // (coefficients in `v`), so every homogeneous RHS that is a quotient of
    // sums (`(3y − x)/(x + y)`, `2xy/(x² − y²)`) was refused.
    {
        let vx = arena.mul(&[v, var]);
        let probe = crate::transforms::subs::subs(arena, rhs, func, vx);
        let probe = crate::transforms::eval::eval(arena, probe);
        let diff = arena.sub(probe, rhs_sub);
        let diff = crate::transforms::eval::eval(arena, diff);
        let diff = crate::simplify::ratsimp::ratsimp(arena, diff);
        let diff = crate::transforms::eval::eval(arena, diff);
        if !arena.is_zero_structural(diff) {
            return None;
        }
    }

    // Now we have:  v + x*v' = f(v)  →  dv/(f(v) - v) = dx/x
    // Integrate:  ∫ dv/(f(v) - v) = ln|x| + C1
    let f_v_minus_v = arena.sub(rhs_sub, v);
    let f_v_minus_v = crate::transforms::eval::eval(arena, f_v_minus_v);
    let f_v_minus_v = crate::simplify::ratsimp::ratsimp(arena, f_v_minus_v);
    let f_v_minus_v = crate::transforms::eval::eval(arena, f_v_minus_v);

    if f_v_minus_v == arena.zero {
        // f(v) = v means y' = y/x → y = C1*x (linear through origin)
        let c1 = arena.symbol("C1");
        let solution = arena.mul(&[c1, var]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    let neg_one = arena.int(-1);
    let inv_fv = arena.pow(f_v_minus_v, neg_one);
    let lhs_integral = integrate_forms(arena, inv_fv, v);

    // If integration of 1/(f(v)-v) failed, bail out.
    if integral_failed(arena, lhs_integral) {
        return None;
    }

    // lhs_integral = ln|x| + C1
    let abs_x = arena.abs(var);
    let ln_abs_x = arena.ln(abs_x);
    let c1 = arena.symbol("C1");
    let rhs_eq = arena.add(&[ln_abs_x, c1]);

    // Implicit solution: lhs_integral(v) = ln|x| + C1
    // Back-substitute v = y/x:  lhs(y/x) - ln|x| - C1 = 0
    let y_over_x = arena.div(func, var);
    let lhs_backsub = crate::transforms::subs::subs(arena, lhs_integral, v, y_over_x);
    let lhs_backsub = crate::transforms::eval::eval(arena, lhs_backsub);

    let implicit = arena.sub(lhs_backsub, rhs_eq);
    let implicit = crate::transforms::eval::eval(arena, implicit);

    // Try to solve for y explicitly (several roots: see `try_exact_ode`).
    let solutions = known_roots(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::transforms::eval::eval(arena, solutions[0].value);
        return Some(OdeResult {
            solution: sol,
            constants: vec![c1],
            branches: Vec::new(),
        });
    }

    // Return implicit form.
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
        branches: Vec::new(),
    })
}
