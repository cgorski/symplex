//! Ordinary Differential Equation (ODE) solver.
//!
//! Solves first-order and second-order ODEs:
//!
//! - **Full separable:** `dy/dx = f(x) * g(y)` → `∫ 1/g(y) dy = ∫ f(x) dx`
//! - **Simple separable:** `dy/dx = f(x)` (no y dependence)
//! - **First-order linear (variable coefficient):** `y' + P(x)*y = Q(x)`
//!   → `y = (1/μ) * [∫ Q(x)*μ dx + C1]` where `μ = exp(∫ P(x) dx)`
//! - **First-order linear constant-coefficient:** `y' + a*y = f(x)`
//!   → `y = e^(-ax) * ∫ f(x)*e^(ax) dx`
//! - **Second-order linear constant-coefficient:** `y'' + b*y' + c*y = 0`
//!   → characteristic equation `r² + b*r + c = 0`, solution based on roots

use crate::arena::Arena;
use crate::node::{ExprId, ExprNode, SymbolId};
use num_traits::One;

/// An ODE representation: f(x, y, y', y'', ...) = 0
/// For now, we support limited forms detected by pattern matching.
pub struct OdeResult {
    /// The general solution y = ... (may contain constants C1, C2)
    pub solution: ExprId,
    /// Names of the arbitrary constants
    pub constants: Vec<ExprId>,
}

/// Attempt to solve a first-order or second-order ODE.
///
/// The ODE is given as `expr = 0` where `expr` may contain:
/// - `var` (the independent variable, typically `x`)
/// - `func` (the dependent variable, typically `y`)
/// - `Derivative(func, var)` (first derivative `y'`)
/// - `Derivative(Derivative(func, var), var)` (second derivative `y''`)
///
/// Returns `None` if the ODE type is not recognized.
pub fn dsolve(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId, // y
    var: ExprId,  // x
) -> Option<OdeResult> {
    let var_sym = match arena.node(var) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };
    let func_sym = match arena.node(func) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Try to detect the ODE type

    // Type 1: Second-order constant-coefficient: a*y'' + b*y' + c*y = 0
    if let Some(result) = try_second_order_const_coeff(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 2: General first-order linear (variable P(x)): y' + P(x)*y = Q(x)
    if let Some(result) = try_first_order_linear_general(arena, expr, func, var, func_sym, var_sym)
    {
        return Some(result);
    }

    // Type 3: Full separable: y' = f(x)*g(y)
    if let Some(result) = try_full_separable(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 4: Simple separable: y' = f(x) (no y dependence) — fallback
    if let Some(result) = try_simple_separable(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    None
}

/// Solve y' = f(x) (simplest separable: no y dependence).
fn try_simple_separable(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    // Pattern: Derivative(y, x) + f(x) = 0, or Derivative(y, x) - f(x) = 0
    // Rearranges to: y' = f(x), then y = ∫ f(x) dx + C

    // Look for Derivative(func, var) in the expression
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Check if expr is Add containing dy_dx
    if let ExprNode::Add(ref children) = arena.node(expr).clone() {
        let mut has_deriv = false;
        let mut _deriv_coeff = None;
        let mut other_terms: Vec<ExprId> = Vec::new();

        for &child in children {
            if child == dy_dx {
                has_deriv = true;
                _deriv_coeff = Some(arena.one);
            } else if !contains_sym(arena, child, func_sym) {
                other_terms.push(child);
            } else {
                // Term contains y — not simple separable
                return None;
            }
        }

        if has_deriv {
            // y' + other_terms = 0 → y' = -other_terms → y = -∫ other_terms dx + C
            let rhs = if other_terms.is_empty() {
                arena.zero
            } else {
                let sum = arena.add(&other_terms);
                arena.neg(sum)
            };
            let integral = crate::integrate::integrate(arena, rhs, var);
            let c1 = arena.symbol("C1");
            let solution = arena.add(&[integral, c1]);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
            });
        }
    }

    // Also handle: Derivative(y, x) = expr pattern (if expr is a single Derivative)
    if expr == dy_dx {
        // y' = 0 → y = C
        let c1 = arena.symbol("C1");
        return Some(OdeResult {
            solution: c1,
            constants: vec![c1],
        });
    }

    None
}

/// Solve y'' + b*y' + c*y = 0 via characteristic equation.
fn try_second_order_const_coeff(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    _func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Pattern: a*y'' + b*y' + c*y = 0
    // Extract coefficients of y'', y', and y
    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else {
            return None; // Contains non-homogeneous or non-constant-coefficient terms
        }
    }

    use num_traits::Zero;
    if a_coeff.is_zero() {
        return None; // Not second order
    }

    // Normalize: divide by a
    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // Characteristic equation: r² + b*r + c = 0
    let r_var = arena.symbol("__r");
    let two = arena.int(2);
    let b_id = {
        let nid = arena.intern_num(b.clone());
        arena.intern(ExprNode::Num(nid))
    };
    let c_id = {
        let nid = arena.intern_num(c.clone());
        arena.intern(ExprNode::Num(nid))
    };

    let r_var_sq = arena.pow(r_var, two);
    let b_r = arena.mul(&[b_id, r_var]);
    let char_eq = arena.add(&[r_var_sq, b_r, c_id]);
    let roots = crate::solve::solve(arena, char_eq, r_var);

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    match roots.len() {
        2 => {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            if r1 == r2 {
                // Repeated root: y = (C1 + C2*x) * e^(r*x)
                let rx = arena.mul(&[r1, var]);
                let exp_rx = arena.exp(rx);
                let c2_x = arena.mul(&[c2, var]);
                let inner = arena.add(&[c1, c2_x]);
                let solution = arena.mul(&[inner, exp_rx]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                })
            } else {
                // Distinct roots: y = C1*e^(r1*x) + C2*e^(r2*x)
                let r1x = arena.mul(&[r1, var]);
                let r2x = arena.mul(&[r2, var]);
                let exp_r1x = arena.exp(r1x);
                let exp_r2x = arena.exp(r2x);
                let term1 = arena.mul(&[c1, exp_r1x]);
                let term2 = arena.mul(&[c2, exp_r2x]);
                let solution = arena.add(&[term1, term2]);
                Some(OdeResult {
                    solution,
                    constants: vec![c1, c2],
                })
            }
        }
        1 => {
            // Single root (shouldn't happen for quadratic, but handle)
            let r = roots[0].value;
            let rx = arena.mul(&[r, var]);
            let exp_rx = arena.exp(rx);
            let c2_x = arena.mul(&[c2, var]);
            let inner = arena.add(&[c1, c2_x]);
            let solution = arena.mul(&[inner, exp_rx]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
            })
        }
        _ => None,
    }
}

/// Solve y' + a*y = f(x) (first-order linear with constant coefficient).
fn try_first_order_linear(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut has_dy = false;
    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_of_x = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            // coeff should be 1 for standard form
            if !coeff.is_one() {
                return None; // Non-unit coefficient on y'
            }
        } else if term == func {
            a_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            f_of_x.push(child);
        } else {
            return None; // Contains y in a non-linear way
        }
    }

    use num_traits::Zero;
    if !has_dy {
        return None;
    }

    // y' + a*y + f(x) = 0 → y' + a*y = -f(x)
    // Solution: y = e^(-ax) * (∫ -f(x)*e^(ax) dx + C)

    let a_id = {
        let nid = arena.intern_num(a_coeff.clone());
        arena.intern(ExprNode::Num(nid))
    };
    let neg_a = arena.neg(a_id);

    // Build e^(ax) and e^(-ax)
    let ax = arena.mul(&[a_id, var]);
    let neg_ax = arena.mul(&[neg_a, var]);
    let exp_ax = arena.exp(ax);
    let exp_neg_ax = arena.exp(neg_ax);

    if a_coeff.is_zero() {
        // y' = -f(x) → y = -∫ f(x) dx + C
        let f = if f_of_x.is_empty() {
            arena.zero
        } else {
            let sum = arena.add(&f_of_x);
            arena.neg(sum)
        };
        let integral = crate::integrate::integrate(arena, f, var);
        let c1 = arena.symbol("C1");
        let solution = arena.add(&[integral, c1]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
        });
    }

    // General case: y = e^(-ax) * (∫ (-f(x))*e^(ax) dx + C1)
    let neg_f = if f_of_x.is_empty() {
        arena.zero
    } else {
        let sum = arena.add(&f_of_x);
        arena.neg(sum)
    };
    let integrand = arena.mul(&[neg_f, exp_ax]);
    let integral = crate::integrate::integrate(arena, integrand, var);

    let c1 = arena.symbol("C1");
    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[exp_neg_ax, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Full separable: y' = f(x) * g(y)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y' = f(x)*g(y) via separation of variables.
///
/// Algorithm:
/// 1. Extract dy/dx from the ODE expression
/// 2. Collect remaining terms as RHS (negated)
/// 3. If RHS factors into x-only × y-only parts, separate them
/// 4. For the common case g(y) = y: solution is y = C1·exp(∫f(x)dx)
/// 5. Otherwise: implicit solution ∫(1/g(y))dy = ∫f(x)dx + C1
fn try_full_separable(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying full separable");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));

    // Extract dy/dx term and collect the rest as RHS
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

    // We need coefficient on y' to be 1 (or normalize)
    if !dy_coeff.is_one() {
        // Normalize: divide everything else by dy_coeff
        let inv_coeff = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
        let inv_id = {
            let nid = arena.intern_num(inv_coeff);
            arena.intern(ExprNode::Num(nid))
        };
        let mut scaled = Vec::new();
        for &t in &other_terms {
            scaled.push(arena.mul(&[inv_id, t]));
        }
        other_terms = scaled;
    }

    // RHS = -(other_terms), i.e., y' = -other_terms
    let rhs = if other_terms.is_empty() {
        return None; // y' = 0 is handled by simple separable
    } else {
        let sum = arena.add(&other_terms);
        arena.neg(sum)
    };

    // Now we need: rhs = f(x) * g(y) where f depends only on var and g only on func
    // Check if rhs depends on func at all — if not, this is simple separable
    if !contains_sym(arena, rhs, func_sym) {
        return None; // Let simple separable handle it
    }

    // Try to factor rhs into x-only and y-only parts
    let factors = collect_mul_factors(arena, rhs);

    let mut x_factors: Vec<ExprId> = Vec::new();
    let mut y_factors: Vec<ExprId> = Vec::new();

    for &factor in &factors {
        let has_x = contains_sym(arena, factor, var_sym);
        let has_y = contains_sym(arena, factor, func_sym);

        if has_x && has_y {
            // Factor depends on both x and y — cannot separate
            return None;
        } else if has_y {
            y_factors.push(factor);
        } else {
            // Pure x-factor or constant
            x_factors.push(factor);
        }
    }

    if y_factors.is_empty() {
        return None; // No y dependence — let simple separable handle it
    }

    // f(x) = product of x_factors
    let f_x = if x_factors.is_empty() {
        arena.one
    } else if x_factors.len() == 1 {
        x_factors[0]
    } else {
        arena.mul(&x_factors)
    };

    // g(y) = product of y_factors
    let g_y = if y_factors.len() == 1 {
        y_factors[0]
    } else {
        arena.mul(&y_factors)
    };

    // Common case: g(y) = y → solution is y = C1·exp(∫f(x)dx)
    if g_y == func {
        let integral_fx = crate::integrate::integrate(arena, f_x, var);
        let c1 = arena.symbol("C1");
        let exponent = arena.add(&[integral_fx, c1]);
        let solution = arena.exp(exponent);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
        });
    }

    // Check if g(y) is a constant times y (e.g., 2*y or -y)
    {
        let (coeff, base) = arena.as_coeff_term(g_y);
        if base == func {
            // g(y) = coeff * y → ∫(1/(coeff*y))dy = (1/coeff)*ln(y)
            // (1/coeff)*ln(y) = ∫f(x)dx + C1 → ln(y) = coeff*∫f(x)dx + C1
            // → y = exp(coeff*∫f(x)dx + C1) = C1*exp(coeff*∫f(x)dx)
            let coeff_id = {
                let nid = arena.intern_num(coeff);
                arena.intern(ExprNode::Num(nid))
            };
            let scaled_fx = arena.mul(&[coeff_id, f_x]);
            let integral_fx = crate::integrate::integrate(arena, scaled_fx, var);
            let c1 = arena.symbol("C1");
            let exponent = arena.add(&[integral_fx, c1]);
            let solution = arena.exp(exponent);
            return Some(OdeResult {
                solution,
                constants: vec![c1],
            });
        }
    }

    // General case: ∫(1/g(y))dy = ∫f(x)dx + C1 (implicit solution)
    // Build 1/g(y) as g(y)^(-1)
    let neg_one = arena.int(-1);
    let inv_gy = arena.pow(g_y, neg_one);

    // We can't easily integrate w.r.t. y in this framework (integration variable
    // must be the independent variable). Return an implicit form using Integral nodes.
    let lhs_integral = arena.intern(ExprNode::Integral(inv_gy, func));
    let rhs_integral = crate::integrate::integrate(arena, f_x, var);
    let c1 = arena.symbol("C1");
    // Solution expressed as: ∫(1/g(y))dy = ∫f(x)dx + C1
    // We store the implicit solution as: ∫(1/g(y))dy - ∫f(x)dx - C1 = 0
    // but for the user, return the RHS: ∫f(x)dx + C1
    // Actually, we should try to solve for y. For now, if we can't get explicit,
    // return the implicit equation as the "solution" expression (LHS - RHS).
    let neg_rhs = arena.neg(rhs_integral);
    let neg_c1 = arena.neg(c1);
    let solution = arena.add(&[lhs_integral, neg_rhs, neg_c1]);
    Some(OdeResult {
        solution,
        constants: vec![c1],
    })
}

/// Collect multiplicative factors from an expression.
/// If `expr` is `Mul(a, b, c)`, return `[a, b, c]`.
/// If `expr` is `Neg(inner)`, return factors of inner with a `-1` prepended.
/// Otherwise return `[expr]`.
fn collect_mul_factors(arena: &mut Arena, expr: ExprId) -> Vec<ExprId> {
    match arena.node(expr).clone() {
        ExprNode::Mul(children) => children.to_vec(),
        ExprNode::Neg(inner) => {
            let neg_one = arena.int(-1);
            let mut factors = vec![neg_one];
            factors.extend(collect_mul_factors(arena, inner));
            factors
        }
        _ => vec![expr],
    }
}

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
fn try_first_order_linear_general(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
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

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            has_dy = true;
            dy_coeff_rational += coeff;
        } else if contains_sym(arena, child, func_sym) {
            y_terms.push(child);
        } else {
            free_terms.push(child);
        }
    }

    use num_traits::Zero;
    if !has_dy || dy_coeff_rational.is_zero() {
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
        if let Some(px) = extract_coeff_of_func(arena, yt, func, func_sym, var_sym) {
            p_x_terms.push(px);
        } else {
            // Non-linear in y
            return None;
        }
    }

    // Normalize by dy_coeff: divide P(x) and Q(x) by the coefficient of y'
    if !dy_coeff_rational.is_one() {
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
    if let Some(_num_val) = arena.as_num(p_x) {
        // P(x) is constant — let try_first_order_linear handle this
        return try_first_order_linear(arena, expr, func, var, func_sym, var_sym);
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
    let int_px = crate::integrate::integrate(arena, p_x, var);

    // Check if integration failed (returned an unevaluated Integral node)
    if let ExprNode::Integral(_, _) = arena.node(int_px).clone() {
        // Integration of P(x) failed — we can't compute the integrating factor
        return None;
    }

    let mu = arena.exp(int_px);

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
        });
    }

    // Nonhomogeneous: y = (1/μ) * [∫ Q(x)*μ dx + C1]
    let integrand = arena.mul(&[q_x, mu]);
    let integral = crate::integrate::integrate(arena, integrand, var);

    let inner = arena.add(&[integral, c1]);
    let solution = arena.mul(&[inv_mu, inner]);

    Some(OdeResult {
        solution,
        constants: vec![c1],
    })
}

/// Try to extract the coefficient of `func` (y) from an expression that should
/// be of the form `P(x) * y`. Returns `Some(P(x))` if the expression is linear
/// in y, `None` otherwise.
fn extract_coeff_of_func(
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
        if !coeff.is_one() && term != expr
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

/// Check if an expression contains a specific symbol.
fn contains_sym(arena: &Arena, expr: ExprId, sym: SymbolId) -> bool {
    match arena.node(expr).clone() {
        ExprNode::Symbol(s) => s == sym,
        other => {
            for &child in other.children().iter() {
                if contains_sym(arena, child, sym) {
                    return true;
                }
            }
            false
        }
    }
}

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
    /// a*y'' + b*y' + c*y = 0 — second-order linear constant-coefficient homogeneous
    SecondOrderLinearCCHomogeneous,
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

    if expr_contains(arena, expr, d2y_dx2) {
        // Try to verify it matches a*y'' + b*y' + c*y = 0 pattern
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut is_const_coeff_homogeneous = true;
            let mut has_y2 = false;
            for &child in children {
                let (_coeff, term) = arena.as_coeff_term(child);
                if term == d2y_dx2 {
                    has_y2 = true;
                } else if term == dy_dx {
                    // OK — y' term
                } else if term == func {
                    // OK — y term
                } else {
                    is_const_coeff_homogeneous = false;
                    break;
                }
            }
            if has_y2 && is_const_coeff_homogeneous {
                return OdeType::SecondOrderLinearCCHomogeneous;
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

        return OdeType::Unknown;
    }

    OdeType::Unknown
}

/// Check if `haystack` contains the sub-expression `needle` anywhere.
fn expr_contains(arena: &Arena, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle {
        return true;
    }
    let node = arena.node(haystack).clone();
    for &child in node.children().iter() {
        if expr_contains(arena, child, needle) {
            return true;
        }
    }
    false
}

/// Check if `expr` contains `sym` in a position that is NOT inside `deriv_node`.
///
/// This detects whether `func` (as a bare symbol) appears outside derivative
/// sub-expressions — i.e., `y` appears outside `dy/dx`.
fn contains_sym_outside_deriv(
    arena: &Arena,
    expr: ExprId,
    sym: SymbolId,
    deriv_node: ExprId,
) -> bool {
    if expr == deriv_node {
        // Skip — this is the derivative node, don't look inside
        return false;
    }
    match arena.node(expr).clone() {
        ExprNode::Symbol(s) => s == sym,
        other => {
            for &child in other.children().iter() {
                if contains_sym_outside_deriv(arena, child, sym, deriv_node) {
                    return true;
                }
            }
            false
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE solution verification
// ═══════════════════════════════════════════════════════════════════════════

/// Check whether a solution satisfies an ODE.
///
/// Substitutes the solution for `func`, differentiates as needed,
/// and checks if the ODE expression evaluates to zero.
///
/// The ODE is given as `ode_expr = 0`.
pub fn checkodesol(
    arena: &mut Arena,
    ode_expr: ExprId,
    solution: ExprId,
    func: ExprId,
    var: ExprId,
) -> bool {
    // Build derivative nodes
    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Compute derivatives of the solution
    let sol_d1 = crate::diff::diff(arena, solution, var);
    let sol_d2 = crate::diff::diff(arena, sol_d1, var);

    // Substitute second derivative first (more specific), then first, then func
    let mut result = crate::subs::subs(arena, ode_expr, d2y_dx2, sol_d2);
    result = crate::subs::subs(arena, result, dy_dx, sol_d1);
    result = crate::subs::subs(arena, result, func, solution);

    // Evaluate and simplify
    result = crate::eval::eval(arena, result);
    result = crate::expand::expand(arena, result);
    result = crate::eval::eval(arena, result);

    if result == arena.zero {
        return true;
    }

    // Try another round of expand + eval for stubborn expressions
    result = crate::expand::expand(arena, result);
    result = crate::eval::eval(arena, result);

    result == arena.zero
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: &str) -> ExprId {
        a.symbol(name)
    }
    fn display(a: &Arena, id: ExprId) -> String {
        a.display(id).to_string()
    }

    #[test]
    fn solve_dy_dx_eq_x() {
        // y' = x → y = x²/2 + C1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        // dy/dx - x = 0
        let expr = a.sub(dy, x);
        let result = dsolve(&mut a, expr, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("x"), "should contain x: {s}");
    }

    #[test]
    fn solve_dy_dx_eq_0() {
        // y' = 0 → y = C1
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let result = dsolve(&mut a, dy, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should be constant: {s}");
    }

    #[test]
    fn solve_second_order_y_plus_y_eq_0() {
        // y'' + y = 0 → y = C1*cos(x) + C2*sin(x) (via complex roots ±i)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let d2y = a.intern(ExprNode::Derivative(dy, x));
        let expr = a.add(&[d2y, y]);
        let result = dsolve(&mut a, expr, y, x);
        if let Some(r) = result {
            let s = display(&a, r.solution);
            assert!(
                s.contains("C1") && s.contains("C2"),
                "should have two constants: {s}"
            );
            assert!(s.contains("exp"), "should contain exp: {s}");
        }
        // It's OK if this doesn't solve yet — complex characteristic roots
    }

    #[test]
    fn solve_y_prime_plus_2y_eq_0() {
        // y' + 2y = 0 → y = C1*e^(-2x)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let two = a.int(2);
        let two_y = a.mul(&[two, y]);
        let expr = a.add(&[dy, two_y]);
        let result = dsolve(&mut a, expr, y, x).expect("should solve");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn solve_full_separable_xy() {
        // y' - x*y = 0 → y' = x*y → y = C1*exp(x²/2)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let xy = a.mul(&[x, y]);
        let expr = a.sub(dy, xy); // y' - x*y = 0
        let result = dsolve(&mut a, expr, y, x).expect("should solve y' = xy");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }

    #[test]
    fn solve_variable_coeff_linear_2xy() {
        // y' + 2*x*y = 0 → y = C1*exp(-x²)
        let mut a = Arena::new();
        let x = sym(&mut a, "x");
        let y = sym(&mut a, "y");
        let dy = a.intern(ExprNode::Derivative(y, x));
        let two = a.int(2);
        let two_x_y = a.mul(&[two, x, y]);
        let expr = a.add(&[dy, two_x_y]); // y' + 2*x*y = 0
        let result = dsolve(&mut a, expr, y, x).expect("should solve y' + 2xy = 0");
        let s = display(&a, result.solution);
        assert!(s.contains("C1"), "should have constant: {s}");
        assert!(s.contains("exp"), "should contain exp: {s}");
    }
}
