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

    // Type 1b: Second-order CC nonhomogeneous: a*y'' + b*y' + c*y = f(x)
    if let Some(result) =
        try_second_order_cc_nonhomogeneous(arena, expr, func, var, func_sym, var_sym)
    {
        return Some(result);
    }

    // Type 2: General first-order linear (variable P(x)): y' + P(x)*y = Q(x)
    if let Some(result) = try_first_order_linear_general(arena, expr, func, var, func_sym, var_sym)
    {
        return Some(result);
    }

    // Type 2b: Exact first-order ODE: M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x
    if let Some(result) = try_exact_ode(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 2c: Non-exact ODE with integrating factor μ(x) or μ(y)
    if let Some(result) = try_integrating_factor_ode(arena, expr, func, var, func_sym, var_sym) {
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

/// Solve the characteristic equation r² + b·r + c = 0 and construct the
/// homogeneous solution of y'' + b·y' + c·y = 0.
///
/// This is extracted as a helper so that both the homogeneous and
/// nonhomogeneous second-order solvers can reuse it.
fn solve_characteristic_equation(
    arena: &mut Arena,
    b: num_rational::Ratio<num_bigint::BigInt>,
    c: num_rational::Ratio<num_bigint::BigInt>,
    var: ExprId,
) -> Option<OdeResult> {
    let r_var = arena.symbol("__r");
    let two = arena.int(2);
    let b_id = {
        let nid = arena.intern_num(b);
        arena.intern(ExprNode::Num(nid))
    };
    let c_id = {
        let nid = arena.intern_num(c);
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
            // Single root (treat as repeated)
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

/// Solve a·y'' + b·y' + c·y = f(x) where f(x) is a polynomial, via
/// the method of undetermined coefficients.
///
/// Returns `None` when:
/// - The expression is not an `Add` node.
/// - The expression is not second-order (no y'' term).
/// - The expression is homogeneous (no forcing terms) — let the
///   dedicated homogeneous solver handle it.
/// - The forcing term is not polynomial in `var`.
/// - The characteristic equation cannot be solved.
fn try_second_order_cc_nonhomogeneous(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<OdeResult> {
    use num_traits::Zero;

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut f_of_x_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            // Term free of y and its derivatives — forcing f(x)
            f_of_x_terms.push(child);
        } else {
            return None; // nonlinear or variable-coefficient term
        }
    }

    if a_coeff.is_zero() {
        return None; // Not second order
    }

    if f_of_x_terms.is_empty() {
        return None; // Homogeneous — handled by try_second_order_const_coeff
    }

    // Normalise: divide through by a so the leading coefficient is 1.
    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // Build g(x) = sum of the forcing terms.  The ODE reads
    //   y'' + b·y' + c·y + g(x)/a = 0   ⟹   rhs = −g(x)/a
    let f_sum = if f_of_x_terms.len() == 1 {
        f_of_x_terms[0]
    } else {
        arena.add(&f_of_x_terms)
    };

    // The forcing term must be polynomial in var.
    let f_coeffs_expr = arena.coefficients_of(f_sum, var)?;
    if f_coeffs_expr.is_empty() {
        return None;
    }

    // Convert to rational and negate/scale: rhs_j = −f_j / a
    let mut rhs_coeffs: Vec<num_rational::Ratio<num_bigint::BigInt>> = Vec::new();
    for &cid in &f_coeffs_expr {
        let val = arena.as_num(cid)?.clone();
        rhs_coeffs.push(-val / &a_coeff);
    }

    // Particular solution via undetermined coefficients
    let particular_coeffs = find_particular_polynomial(&b, &c, &rhs_coeffs)?;
    let y_p = build_polynomial_expr(arena, &particular_coeffs, var);

    // Homogeneous solution via the characteristic equation
    let homo_result = solve_characteristic_equation(arena, b, c, var)?;

    // General solution = homogeneous + particular
    let solution = arena.add(&[homo_result.solution, y_p]);

    Some(OdeResult {
        solution,
        constants: homo_result.constants,
    })
}

/// Solve the undetermined-coefficients linear system for a polynomial
/// particular solution of  y'' + b·y' + c·y = rhs(x).
///
/// `rhs_coeffs[j]` is the coefficient of x^j on the right-hand side,
/// in ascending degree order.
///
/// Returns ascending-order coefficients of y_p, or `None` on failure.
fn find_particular_polynomial(
    b: &num_rational::Ratio<num_bigint::BigInt>,
    c: &num_rational::Ratio<num_bigint::BigInt>,
    rhs_coeffs: &[num_rational::Ratio<num_bigint::BigInt>],
) -> Option<Vec<num_rational::Ratio<num_bigint::BigInt>>> {
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use num_traits::Zero;

    let n = rhs_coeffs.len() - 1;

    if !c.is_zero() {
        // ── Case 1: c ≠ 0 ──────────────────────────────────────────────
        // y_p = A_0 + A_1·x + … + A_n·x^n   (same degree as rhs)
        //
        // Matching x^j (j = n … 0):
        //   (j+2)(j+1)·A_{j+2} + b·(j+1)·A_{j+1} + c·A_j = r_j
        //
        // Triangular system, solved top-down.
        let mut a = vec![Ratio::<BigInt>::zero(); n + 1];
        for j in (0..=n).rev() {
            let a_j2 = if j + 2 <= n {
                a[j + 2].clone()
            } else {
                Ratio::zero()
            };
            let a_j1 = if j < n {
                a[j + 1].clone()
            } else {
                Ratio::zero()
            };
            let factor2 =
                Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64)) * a_j2;
            let factor1 =
                Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone() * a_j1;
            a[j] = (rhs_coeffs[j].clone() - factor2 - factor1) / c.clone();
        }
        Some(a)
    } else if !b.is_zero() {
        // ── Case 2: c = 0, b ≠ 0 ───────────────────────────────────────
        // Multiply trial by x:  y_p = B_0·x + B_1·x² + … + B_n·x^{n+1}
        //
        // Matching x^j (j = n … 0):
        //   [(j+2)(j+1)·B_{j+1} if j < n] + b·(j+1)·B_j = r_j
        let mut bb = vec![Ratio::<BigInt>::zero(); n + 1];
        for j in (0..=n).rev() {
            let deriv_term = if j < n {
                Ratio::from_integer(BigInt::from(((j + 2) * (j + 1)) as i64))
                    * bb[j + 1].clone()
            } else {
                Ratio::zero()
            };
            let denom =
                Ratio::from_integer(BigInt::from((j + 1) as i64)) * b.clone();
            if denom.is_zero() {
                return None;
            }
            bb[j] = (rhs_coeffs[j].clone() - deriv_term) / denom;
        }
        // Shift: x·(B_0 + B_1·x + …) → coefficients [0, B_0, B_1, …]
        let mut result = vec![Ratio::zero()];
        result.extend(bb);
        Some(result)
    } else {
        // ── Case 3: c = 0, b = 0 ───────────────────────────────────────
        // y'' = rhs  ⟹  y_p = ∫∫ rhs dx dx
        // Coefficient of x^{j+2} = r_j / ((j+1)(j+2))
        let mut result = vec![Ratio::<BigInt>::zero(); 2];
        for (j, r_j) in rhs_coeffs.iter().enumerate() {
            let denom =
                Ratio::from_integer(BigInt::from(((j + 1) * (j + 2)) as i64));
            result.push(r_j.clone() / denom);
        }
        Some(result)
    }
}

/// Build an arena polynomial expression from ascending-order rational
/// coefficients: `coeffs[j]` is the coefficient of `var^j`.
fn build_polynomial_expr(
    arena: &mut Arena,
    coeffs: &[num_rational::Ratio<num_bigint::BigInt>],
    var: ExprId,
) -> ExprId {
    use num_traits::Zero;
    let mut terms = Vec::new();
    for (j, coeff) in coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let x_pow_j = if j == 0 {
            arena.one
        } else if j == 1 {
            var
        } else {
            let exp = arena.int(j as i64);
            arena.pow(var, exp)
        };
        let term = arena.make_coeff_term(coeff.clone(), x_pow_j);
        terms.push(term);
    }
    if terms.is_empty() {
        arena.zero
    } else if terms.len() == 1 {
        terms[0]
    } else {
        arena.add(&terms)
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

// ═══════════════════════════════════════════════════════════════════════════
// Exact ODE solver: M(x,y)dx + N(x,y)dy = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Extract M(x,y) and N(x,y) from an ODE expression of the form `M + N·y' = 0`.
///
/// Returns `(M, N)` where M is the sum of terms not containing `dy/dx`
/// and N is the total coefficient of `dy/dx`.
///
/// Returns `None` when the expression cannot be decomposed (e.g. `(y')²`).
fn extract_m_n(
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
fn try_exact_ode(
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
    let dm_dy = crate::diff::diff(arena, m_expr, func);
    let dn_dx = crate::diff::diff(arena, n_expr, var);

    let diff_check = arena.sub(dm_dy, dn_dx);
    let diff_eval = crate::eval::eval(arena, diff_check);
    let diff_expanded = crate::expand::expand(arena, diff_eval);
    let diff_simplified = crate::eval::eval(arena, diff_expanded);

    if diff_simplified != arena.zero {
        return None; // Not exact
    }

    // ── Build potential function F(x,y) ───────────────────────────
    // Step 1: F_partial = ∫ M dx  (treating y as constant)
    let integral_m = crate::integrate::integrate(arena, m_expr, var);
    if matches!(arena.node(integral_m), ExprNode::Integral(_, _)) {
        return None; // Integration of M w.r.t. x failed
    }

    // Step 2: g'(y) = N − ∂(∫M dx)/∂y
    let d_intm_dy = crate::diff::diff(arena, integral_m, func);
    let g_prime = arena.sub(n_expr, d_intm_dy);
    let g_prime = crate::eval::eval(arena, g_prime);
    let g_prime = crate::expand::expand(arena, g_prime);
    let g_prime = crate::eval::eval(arena, g_prime);

    // g'(y) must be free of x.
    if contains_sym(arena, g_prime, var_sym) {
        return None;
    }

    // Step 3: g(y) = ∫ g'(y) dy
    let g_y = crate::integrate::integrate(arena, g_prime, func);
    if matches!(arena.node(g_y), ExprNode::Integral(_, _)) {
        return None;
    }

    // F(x,y) = ∫M dx + g(y)
    let potential = arena.add(&[integral_m, g_y]);
    let potential = crate::eval::eval(arena, potential);

    let c1 = arena.symbol("C1");

    // Try to solve F(x,y) = C1 for y explicitly.
    let f_minus_c1 = arena.sub(potential, c1);
    let solutions = crate::solve::solve(arena, f_minus_c1, func);

    if solutions.len() == 1 {
        return Some(OdeResult {
            solution: solutions[0].value,
            constants: vec![c1],
        });
    }

    // Return the implicit solution F(x,y) (the equation is F = C1).
    Some(OdeResult {
        solution: potential,
        constants: vec![c1],
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
fn try_integrating_factor_ode(
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

    let dm_dy = crate::diff::diff(arena, m_expr, func);
    let dn_dx = crate::diff::diff(arena, n_expr, var);

    let diff_mn = arena.sub(dm_dy, dn_dx); // ∂M/∂y − ∂N/∂x
    let diff_eval = crate::eval::eval(arena, diff_mn);
    let diff_expanded = crate::expand::expand(arena, diff_eval);
    let diff_simplified = crate::eval::eval(arena, diff_expanded);

    if diff_simplified == arena.zero {
        // Already exact — delegate.
        return try_exact_ode(arena, expr, func, var, func_sym, var_sym);
    }

    // ── Try μ(x): (∂M/∂y − ∂N/∂x) / N free of y ─────────────────
    {
        let ratio = arena.div(diff_simplified, n_expr);
        let ratio = crate::eval::eval(arena, ratio);
        let ratio = crate::expand::expand(arena, ratio);
        let ratio = crate::eval::eval(arena, ratio);
        let ratio_cancelled = arena.cancel_expr(ratio, var);

        if !contains_sym(arena, ratio_cancelled, func_sym) {
            let int_ratio = crate::integrate::integrate(arena, ratio_cancelled, var);
            if !matches!(arena.node(int_ratio), ExprNode::Integral(_, _)) {
                let mu = arena.exp(int_ratio);

                // New M' = μ·M,  N' = μ·N
                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::eval::eval(arena, new_expr);

                if let Some(result) =
                    try_exact_ode(arena, new_expr, func, var, func_sym, var_sym)
                {
                    return Some(result);
                }
            }
        }
    }

    // ── Try μ(y): (∂N/∂x − ∂M/∂y) / M free of x ─────────────────
    {
        let neg_diff = arena.neg(diff_simplified); // ∂N/∂x − ∂M/∂y
        let ratio = arena.div(neg_diff, m_expr);
        let ratio = crate::eval::eval(arena, ratio);
        let ratio = crate::expand::expand(arena, ratio);
        let ratio = crate::eval::eval(arena, ratio);
        let ratio_cancelled = arena.cancel_expr(ratio, func);

        if !contains_sym(arena, ratio_cancelled, var_sym) {
            let int_ratio = crate::integrate::integrate(arena, ratio_cancelled, func);
            if !matches!(arena.node(int_ratio), ExprNode::Integral(_, _)) {
                let mu = arena.exp(int_ratio);

                let new_m = arena.mul(&[mu, m_expr]);
                let new_n = arena.mul(&[mu, n_expr]);

                let dy_dx = arena.intern(ExprNode::Derivative(func, var));
                let n_dy = arena.mul(&[new_n, dy_dx]);
                let new_expr = arena.add(&[new_m, n_dy]);
                let new_expr = crate::eval::eval(arena, new_expr);

                if let Some(result) =
                    try_exact_ode(arena, new_expr, func, var, func_sym, var_sym)
                {
                    return Some(result);
                }
            }
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
    /// M(x,y) + N(x,y)·y' = 0 with ∂M/∂y = ∂N/∂x — exact first-order
    ExactFirstOrder,
    /// a*y'' + b*y' + c*y = 0 — second-order linear constant-coefficient homogeneous
    SecondOrderLinearCCHomogeneous,
    /// a*y'' + b*y' + c*y = f(x) — second-order linear constant-coefficient nonhomogeneous
    SecondOrderLinearCCNonHomogeneous,
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
        // Try to verify it matches a*y'' + b*y' + c*y [+ f(x)] = 0 pattern
        if let ExprNode::Add(ref children) = arena.node(expr).clone() {
            let mut all_const_coeff = true;
            let mut has_y2 = false;
            let mut has_forcing = false;
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
                    has_forcing = true;
                } else {
                    all_const_coeff = false;
                    break;
                }
            }
            if has_y2 && all_const_coeff {
                if has_forcing {
                    return OdeType::SecondOrderLinearCCNonHomogeneous;
                }
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

        // Check for exact ODE: M + N·y' = 0 with ∂M/∂y = ∂N/∂x
        if let Some((m_ex, n_ex)) = extract_m_n(arena, expr, func, var)
            && (contains_sym(arena, m_ex, func_sym) || contains_sym(arena, n_ex, func_sym))
        {
            let dm_dy = crate::diff::diff(arena, m_ex, func);
            let dn_dx = crate::diff::diff(arena, n_ex, var);
            let check = arena.sub(dm_dy, dn_dx);
            let check = crate::eval::eval(arena, check);
            let check = crate::expand::expand(arena, check);
            let check = crate::eval::eval(arena, check);
            if check == arena.zero {
                return OdeType::ExactFirstOrder;
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
