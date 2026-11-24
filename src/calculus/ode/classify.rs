//! ODE classification (part of [`ode`](super)).

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// ODE classification
// ═══════════════════════════════════════════════════════════════════════════

/// ODE classification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdeType {
    /// y' = f(x) — simple separable, no y dependence
    SimpleSeparable,
    /// y' = f(x)*g(y) — full separable
    FullSeparable,
    /// y' + a*y = f(x) — first-order linear with constant coefficients
    FirstOrderLinearCC,
    /// y' + P(x)*y = Q(x) — first-order linear with variable coefficients
    FirstOrderLinearVC,
    /// M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x — exact first-order
    ExactFirstOrder,
    /// a*y'' + b*y' + c*y = 0 — second-order linear constant-coefficient homogeneous
    SecondOrderLinearCCHomogeneous,
    /// a*y'' + b*y' + c*y = f(x) — second-order linear constant-coefficient nonhomogeneous
    SecondOrderLinearCCNonHomogeneous,
    /// y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1) — Bernoulli equation
    Bernoulli,
    /// a·x²·y'' + b·x·y' + c·y = 0 — Euler-Cauchy equation
    EulerCauchy,
    /// y'' + p·y' + q·y = g(x) solved via variation of parameters
    VariationOfParameters,
    /// y' = f(y/x) — homogeneous coefficient (degree-0 homogeneous RHS)
    HomogeneousCoefficient,
    /// F(y, y', y'') = 0 (no explicit x) — reducible via p = y'
    NthOrderReducible,
    /// Σ a_k·y^(k) = g(x) with constant coefficients, order ≥ 3
    /// (orders 1 and 2 use the dedicated variants above)
    NthOrderLinearConstCoeff,
    /// M(x,y) + N(x,y)·y' = 0, non-exact but with an integrating factor
    /// μ(x) or μ(y)
    IntegratingFactor,
    /// y = x·y' + f(y') — Clairaut equation
    Clairaut,
    /// y' = q₀(x) + q₁(x)·y + q₂(x)·y² — Riccati equation (needs a
    /// particular solution; see [`solve_riccati`])
    Riccati,
    /// Unrecognized ODE type
    Unknown,
}

/// Classify an ODE without solving it.
///
/// The ODE is given as `expr = 0` where `expr` may contain derivative nodes.
/// Returns the recognized [`OdeType`].
pub fn classify_ode(arena: &mut Arena, expr: ExprId, func: ExprId, var: ExprId) -> OdeType {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return OdeType::Unknown,
    };
    let func_sym = match arena.node(func) {
        ExprNode::Symbol(sid) => *sid,
        _ => return OdeType::Unknown,
    };

    // Check for second-order: look for Derivative(Derivative(func, var), var)
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));
    let d3y_dx3 = arena.intern(ExprNode::Derivative(d2y_dx2, var));

    // Order ≥ 3 with constant coefficients.
    if expr_contains(arena, expr, d3y_dx3) {
        if let Some(ode) = extract_linear_cc(arena, expr, func, var, func_sym)
            && ode.coeffs.len() >= 4
        {
            return OdeType::NthOrderLinearConstCoeff;
        }
        return OdeType::Unknown;
    }

    if expr_contains(arena, expr, d2y_dx2) {
        // Try to verify it matches a*y'' + b*y' + c*y [+ f(x)] = 0 pattern
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut all_const_coeff = true;
            let mut has_y2 = false;
            let mut forcing: Vec<ExprId> = Vec::new();
            for &child in children {
                let (_coeff, term) = arena.as_coeff_term(child);
                if term == d2y_dx2 {
                    has_y2 = true;
                } else if term == dy_dx {
                    // OK — y' term with constant coefficient
                } else if term == func {
                    // OK — y term with constant coefficient
                } else if !contains_sym(arena, child, func_sym) {
                    // Term free of y — nonhomogeneous forcing term
                    forcing.push(child);
                } else {
                    all_const_coeff = false;
                    break;
                }
            }
            if has_y2 && all_const_coeff {
                if forcing.is_empty() {
                    return OdeType::SecondOrderLinearCCHomogeneous;
                }
                // Undetermined coefficients apply to poly × exp × {sin, cos}
                // forcing; anything else needs variation of parameters.
                let all_uc = forcing
                    .iter()
                    .all(|&t| parse_forcing_term(arena, t, var).is_some());
                if all_uc {
                    return OdeType::SecondOrderLinearCCNonHomogeneous;
                }
                if try_variation_of_parameters(arena, expr, func, var, func_sym, var_sym).is_some()
                {
                    return OdeType::VariationOfParameters;
                }
                return OdeType::Unknown;
            }
        }
        // Check for Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
        if try_euler_cauchy(arena, expr, func, var, func_sym, var_sym).is_some() {
            return OdeType::EulerCauchy;
        }
        // Check for nth-order reducible: F(y, y', y'') = 0 with no explicit x.
        {
            let d2_ph = arena.symbol("__d2_cls");
            let d1_ph = arena.symbol("__d1_cls");
            let stripped = crate::transforms::subs::subs(arena, expr, d2y_dx2, d2_ph);
            let stripped = crate::transforms::subs::subs(arena, stripped, dy_dx, d1_ph);
            if !contains_sym(arena, stripped, var_sym) {
                return OdeType::NthOrderReducible;
            }
        }

        // Even if we can't fully classify, it has a second derivative
        return OdeType::Unknown;
    }

    // Check for first-order
    if expr_contains(arena, expr, dy_dx) {
        // Check if func appears outside derivative terms
        if !contains_sym_outside_deriv(arena, expr, func_sym, dy_dx) {
            return OdeType::SimpleSeparable;
        }

        // Check if it matches a linear pattern: y' + P(x)*y = Q(x)
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut is_linear = true;
            let mut has_var_coeff = false;

            for &child in children {
                let (coeff, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    // dy/dx term — fine
                } else if term == func {
                    // Constant coefficient on y — fine
                } else if contains_sym(arena, child, func_sym) {
                    // Check if it's of the form P(x)*y (linear in y)
                    if let Some(px) = extract_coeff_of_func(arena, child, func, func_sym, var_sym) {
                        let _ = coeff; // suppress warning
                        if contains_sym(arena, px, var_sym) {
                            has_var_coeff = true;
                        }
                    } else {
                        is_linear = false;
                        break;
                    }
                }
                // Otherwise it's f(x) — acceptable
            }
            if is_linear {
                if has_var_coeff {
                    return OdeType::FirstOrderLinearVC;
                }
                return OdeType::FirstOrderLinearCC;
            }
        }

        // Check for exact ODE: M + N·y' = 0 with ∂M/∂y = ∂N/∂x
        if let Some((m_ex, n_ex)) = extract_m_n(arena, expr, func, var)
            && (contains_sym(arena, m_ex, func_sym) || contains_sym(arena, n_ex, func_sym))
        {
            let dm_dy = crate::transforms::diff::diff(arena, m_ex, func);
            let dn_dx = crate::transforms::diff::diff(arena, n_ex, var);
            let check = arena.sub(dm_dy, dn_dx);
            let check = crate::transforms::eval::eval(arena, check);
            let check = crate::transforms::expand::expand(arena, check);
            let check = crate::transforms::eval::eval(arena, check);
            if check == arena.zero {
                return OdeType::ExactFirstOrder;
            }
        }

        // Check for Clairaut: y = x·y' + f(y')  (y' appears nonlinearly)
        {
            let p = arena.symbol("__clairaut_p_cls");
            if clairaut_f(arena, expr, func, var, func_sym, var_sym, p).is_some() {
                return OdeType::Clairaut;
            }
        }

        // Check for Bernoulli: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
        if let ExprNode::Add(ref bn_children) = arena.node(expr).clone() {
            let mut bn_has_dy = false;
            let mut bn_has_yn = false;
            let mut bn_ok = true;
            let mut bn_has_free = false;
            for &child in bn_children {
                let (_, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    bn_has_dy = true;
                } else if !contains_sym(arena, child, func_sym) {
                    bn_has_free = true;
                } else if extract_coeff_of_func(arena, child, func, func_sym, var_sym).is_some() {
                    // linear in y — OK
                } else if extract_bernoulli_term(arena, child, func, func_sym, var_sym).is_some() {
                    bn_has_yn = true;
                } else {
                    bn_ok = false;
                    break;
                }
            }
            if bn_has_dy && bn_has_yn && bn_ok && !bn_has_free {
                return OdeType::Bernoulli;
            }
        }

        // Check for homogeneous coefficient: y' = f(y/x)
        // Substitute y = v*x in the RHS; if result is free of x → homogeneous
        if let ExprNode::Add(ref hc_children) = arena.node(expr).clone() {
            let mut hc_has_dy = false;
            let mut hc_other: Vec<ExprId> = Vec::new();
            for &child in hc_children {
                let (_, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    hc_has_dy = true;
                } else {
                    hc_other.push(child);
                }
            }
            if hc_has_dy && !hc_other.is_empty() {
                let hc_rhs = if hc_other.len() == 1 {
                    arena.neg(hc_other[0])
                } else {
                    let s = arena.add(&hc_other);
                    arena.neg(s)
                };
                // RHS must depend on both x and y
                if contains_sym(arena, hc_rhs, func_sym) && contains_sym(arena, hc_rhs, var_sym) {
                    // Degree-0 homogeneity: f(x, v·x) must be free of x after
                    // cancellation.
                    let v_cls = arena.symbol("__v_cls");
                    let vx = arena.mul(&[v_cls, var]);
                    let sub = crate::transforms::subs::subs(arena, hc_rhs, func, vx);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    let sub = crate::transforms::expand::expand(arena, sub);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    let sub = arena.cancel_expr(sub, var);
                    let sub = crate::transforms::eval::eval(arena, sub);
                    if !contains_sym(arena, sub, var_sym) {
                        return OdeType::HomogeneousCoefficient;
                    }
                }
            }
        }

        // Check if it's a full separable: y' = f(x)*g(y)
        // Quick check: if the RHS (after extracting y') factors cleanly
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut has_deriv = false;
            let mut other_terms: Vec<ExprId> = Vec::new();

            for &child in children {
                let (_coeff, term) = arena.as_coeff_term(child);
                if term == dy_dx {
                    has_deriv = true;
                } else {
                    other_terms.push(child);
                }
            }

            if has_deriv && !other_terms.is_empty() {
                let rhs = if other_terms.len() == 1 {
                    arena.neg(other_terms[0])
                } else {
                    let sum = arena.add(&other_terms);
                    arena.neg(sum)
                };
                let factors = collect_mul_factors(arena, rhs);
                let mut can_separate = true;
                for &factor in &factors {
                    let has_x = contains_sym(arena, factor, var_sym);
                    let has_y = contains_sym(arena, factor, func_sym);
                    if has_x && has_y {
                        can_separate = false;
                        break;
                    }
                }
                if can_separate {
                    return OdeType::FullSeparable;
                }
            }
        }

        // Non-exact with an integrating factor μ(x) or μ(y).
        if try_integrating_factor_ode(arena, expr, func, var, func_sym, var_sym).is_some() {
            return OdeType::IntegratingFactor;
        }

        // Riccati: y' = q0 + q1·y + q2·y² with q0, q2 ≠ 0 (checked last: a
        // separable or otherwise directly solvable Riccati keeps the more
        // specific class above).
        if riccati_coeffs(arena, expr, func, var).is_some() {
            return OdeType::Riccati;
        }

        return OdeType::Unknown;
    }

    OdeType::Unknown
}

/// Check if `haystack` contains the sub-expression `needle` anywhere
/// (iteratively: [`walk::contains`](crate::base::walk::contains)).
pub(super) fn expr_contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    crate::base::walk::contains(arena, haystack, needle)
}

/// Check if `expr` contains `sym` in a position that is NOT inside `deriv_node`.
///
/// This detects whether `func` (as a bare symbol) appears outside derivative
/// sub-expressions — i.e., `y` appears outside `dy/dx`.  An explicit stack
/// over the shared DAG, each node visited once (0.28 recursed, and a deep
/// ODE expression could overflow the stack).
fn contains_sym_outside_deriv(
    arena: &Arena,
    expr: ExprId,
    sym: SymbolId,
    deriv_node: ExprId,
) -> bool {
    let mut stack = vec![expr];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(id) = stack.pop() {
        // The derivative node is skipped: don't look inside.
        if id == deriv_node || !seen.insert(id) {
            continue;
        }
        let node = arena.node(id);
        if let ExprNode::Symbol(s) = node {
            if *s == sym {
                return true;
            }
            continue;
        }
        node.for_each_child(|c| stack.push(c));
    }
    false
}
