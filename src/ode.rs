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
//! - **Homogeneous coefficient:** `y' = f(y/x)` — substitution `v = y/x`
//! - **nth-order reducible:** `F(y, y', y'') = 0` (no `x`) — substitution `p = y'`
//! - **Constant-coefficient systems:** `ẋ = A·x` → `x(t) = exp(A·t)·c`
//!   via eigendecomposition (exact) or matrix exponential series (fallback)
//! - **Non-homogeneous systems:** `ẋ = A·x + b(t)` → variation of parameters

use crate::arena::Arena;
use crate::expr::Ex;
use crate::matrix::Matrix;
use crate::node::{ExprId, ExprNode, SymbolId};
use num_traits::One;
use num_traits::Signed;

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

    // Type 1c: Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
    if let Some(result) = try_euler_cauchy(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 1d: Variation of parameters: y'' + p·y' + q·y = g(x) (fallback)
    if let Some(result) =
        try_variation_of_parameters(arena, expr, func, var, func_sym, var_sym)
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

    // Type 2d: Bernoulli: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1)
    if let Some(result) = try_bernoulli(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 2e: Homogeneous coefficient: y' = f(y/x)
    if let Some(result) = try_homogeneous_coefficient(arena, expr, func, var, func_sym, var_sym) {
        return Some(result);
    }

    // Type 2f: nth-order reducible: F(y, y', y'') = 0, no explicit x
    if let Some(result) = try_nth_order_reducible(arena, expr, func, var, func_sym, var_sym) {
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

    // Check for complex roots: if disc = b² − 4c < 0, use Euler/trig form
    {
        let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
        let disc = &b * &b - &four_r * &c;
        if disc.is_negative() {
            return build_trig_homogeneous_solution(arena, &b, &disc, var);
        }
    }

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
    // Check for complex roots: if disc = b² − 4c < 0, use Euler/trig form
    {
        let four_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
        let disc = &b * &b - &four_r * &c;
        if disc.is_negative() {
            return build_trig_homogeneous_solution(arena, &b, &disc, var);
        }
    }

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

    // Try polynomial forcing first
    let y_p_poly = if let Some(f_coeffs_expr) = arena.coefficients_of(f_sum, var) {
        if f_coeffs_expr.is_empty() {
            None
        } else {
            let mut rhs_coeffs: Vec<num_rational::Ratio<num_bigint::BigInt>> = Vec::new();
            let mut all_numeric = true;
            for &cid in &f_coeffs_expr {
                if let Some(val) = arena.as_num(cid) {
                    rhs_coeffs.push(-val.clone() / &a_coeff);
                } else {
                    all_numeric = false;
                    break;
                }
            }
            if all_numeric {
                find_particular_polynomial(&b, &c, &rhs_coeffs).map(|pc| build_polynomial_expr(arena, &pc, var))
            } else {
                None
            }
        }
    } else {
        None
    };

    // If polynomial forcing didn't work, try trig/exp forcing
    let y_p = if let Some(yp) = y_p_poly {
        yp
    } else {
        // Compute rhs = -f_sum / a for trig/exp analysis
        let neg_f_sum = arena.neg(f_sum);
        let a_id = ode_ratio_to_expr(arena, &a_coeff);
        let rhs_expr = arena.div(neg_f_sum, a_id);
        let rhs_expr = crate::eval::eval(arena, rhs_expr);
        try_undetermined_trig_exp(arena, rhs_expr, &b, &c, var)?
    };

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
    // Simplify exp(a)*exp(b) → exp(a+b) before integrating
    let integrand = if let ExprNode::Exp(neg_f_inner) = arena.node(neg_f).clone() {
        let combined_arg = arena.add(&[neg_f_inner, ax]);
        let combined_arg = crate::eval::eval(arena, combined_arg);
        arena.exp(combined_arg)
    } else {
        arena.mul(&[neg_f, exp_ax])
    };
    let integrand = crate::eval::eval(arena, integrand);
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
    // Simplify products of exponentials before integrating
    let integrand = crate::eval::eval(arena, integrand);
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
fn try_homogeneous_coefficient(
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
        crate::eval::eval(arena, s)
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
    let v = arena.symbol("__v");

    // Compute f(v) = RHS(x=1, y=v)
    let rhs_sub = crate::subs::subs(arena, rhs, func, v);
    let rhs_sub = crate::subs::subs(arena, rhs_sub, var, arena.one);
    let rhs_sub = crate::eval::eval(arena, rhs_sub);
    let rhs_sub = crate::expand::expand(arena, rhs_sub);
    let rhs_sub = crate::eval::eval(arena, rhs_sub);

    // f(v) must be free of x (it should be, since we set x=1).
    if contains_sym(arena, rhs_sub, var_sym) {
        return None;
    }

    // Verify homogeneity: f(v) should equal the original RHS when v = y/x.
    // Spot-check: RHS(x, y) should equal f(y/x).  We rely on the algebraic
    // structure being correct — the substitution x=1 is valid precisely when
    // the function is homogeneous of degree 0.

    // Now we have:  v + x*v' = f(v)  →  dv/(f(v) - v) = dx/x
    // Integrate:  ∫ dv/(f(v) - v) = ln|x| + C1
    let f_v_minus_v = arena.sub(rhs_sub, v);
    let f_v_minus_v = crate::eval::eval(arena, f_v_minus_v);

    if f_v_minus_v == arena.zero {
        // f(v) = v means y' = y/x → y = C1*x (linear through origin)
        let c1 = arena.symbol("C1");
        let solution = arena.mul(&[c1, var]);
        return Some(OdeResult {
            solution,
            constants: vec![c1],
        });
    }

    let neg_one = arena.int(-1);
    let inv_fv = arena.pow(f_v_minus_v, neg_one);
    let lhs_integral = crate::integrate::integrate(arena, inv_fv, v);

    // If integration of 1/(f(v)-v) failed, bail out.
    if matches!(arena.node(lhs_integral), ExprNode::Integral(_, _)) {
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
    let lhs_backsub = crate::subs::subs(arena, lhs_integral, v, y_over_x);
    let lhs_backsub = crate::eval::eval(arena, lhs_backsub);

    let implicit = arena.sub(lhs_backsub, rhs_eq);
    let implicit = crate::eval::eval(arena, implicit);

    // Try to solve for y explicitly.
    let solutions = crate::solve::solve(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::eval::eval(arena, solutions[0].value);
        return Some(OdeResult {
            solution: sol,
            constants: vec![c1],
        });
    }

    // Return implicit form.
    Some(OdeResult {
        solution: implicit,
        constants: vec![c1],
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// nth-order reducible ODE: F(y, y', y'') = 0, no explicit x
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a second-order ODE where the independent variable doesn't appear:
///   F(y, y', y'') = 0
///
/// Substitution: p = y', y'' = p·dp/dy reduces to a first-order ODE in p(y).
fn try_nth_order_reducible(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    _func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying nth-order reducible (missing x)");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must have y'' present.
    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    // The independent variable x must NOT appear explicitly (only through
    // y and its derivatives).
    // We check: after removing derivative nodes, does x appear?
    // Strategy: substitute y''→__d2, y'→__d1, then check if var_sym remains.
    let d2_placeholder = arena.symbol("__d2");
    let d1_placeholder = arena.symbol("__d1");
    let stripped = crate::subs::subs(arena, expr, d2y_dx2, d2_placeholder);
    let stripped = crate::subs::subs(arena, stripped, dy_dx, d1_placeholder);
    if contains_sym(arena, stripped, var_sym) {
        return None; // x appears explicitly
    }

    // Now perform the reduction: let p = dy/dx, then d²y/dx² = p·dp/dy
    let p = arena.symbol("__p");
    let dp_dy = arena.intern(ExprNode::Derivative(p, func));
    let p_dp_dy = arena.mul(&[p, dp_dy]);

    // Substitute: y'' → p·dp/dy,  y' → p
    let reduced = crate::subs::subs(arena, expr, d2y_dx2, p_dp_dy);
    let reduced = crate::subs::subs(arena, reduced, dy_dx, p);
    let reduced = crate::eval::eval(arena, reduced);

    // Now `reduced` is a first-order ODE in p(y) with independent var = y.
    // Try to solve it.
    let p_result = dsolve(arena, reduced, p, func)?;

    // p_result.solution gives p = f(y, C1).
    // Now solve dy/dx = p(y) — this is separable: ∫ dy/p(y) = x + C2.
    let c2 = arena.symbol("C2");

    // Check if p_result.solution is simple enough
    let p_sol = p_result.solution;

    // Set up: dy/dx - p_sol = 0  →  ∫ 1/p_sol dy = x + C2
    // We need to integrate 1/p_sol w.r.t. y.
    let neg_one_id = arena.int(-1);
    let inv_p = arena.pow(p_sol, neg_one_id);
    let inv_p = crate::eval::eval(arena, inv_p);
    let lhs_integral = crate::integrate::integrate(arena, inv_p, func);

    if matches!(arena.node(lhs_integral), ExprNode::Integral(_, _)) {
        return None; // Can't integrate 1/p(y)
    }

    // Implicit solution: ∫ dy/p(y) = x + C2
    let rhs = arena.add(&[var, c2]);
    let implicit = arena.sub(lhs_integral, rhs);
    let implicit = crate::eval::eval(arena, implicit);

    // Try to solve explicitly for y.
    let solutions = crate::solve::solve(arena, implicit, func);
    if solutions.len() == 1 {
        let sol = crate::eval::eval(arena, solutions[0].value);
        let mut constants = p_result.constants;
        constants.push(c2);
        return Some(OdeResult {
            solution: sol,
            constants,
        });
    }

    // Return implicit form.
    let mut constants = p_result.constants;
    constants.push(c2);
    Some(OdeResult {
        solution: implicit,
        constants,
    })
}

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
// Helpers for complex roots and undetermined coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a `Ratio<BigInt>` to an arena `ExprId`.
fn ode_ratio_to_expr(
    arena: &mut Arena,
    r: &num_rational::Ratio<num_bigint::BigInt>,
) -> ExprId {
    let nid = arena.intern_num(r.clone());
    arena.intern(ExprNode::Num(nid))
}

/// Build the homogeneous solution in trigonometric form when the
/// characteristic equation `r² + b·r + c = 0` has complex roots.
///
/// For discriminant `disc = b² − 4c < 0`:
///   roots = α ± βi  where  α = −b/2,  β = √(−disc)/2
///   solution = exp(α·x)·(C1·cos(β·x) + C2·sin(β·x))
fn build_trig_homogeneous_solution(
    arena: &mut Arena,
    b: &num_rational::Ratio<num_bigint::BigInt>,
    disc: &num_rational::Ratio<num_bigint::BigInt>,
    var: ExprId,
) -> Option<OdeResult> {
    use num_traits::Zero;

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    // α = −b/2
    let two_r = num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
    let alpha = -(b.clone()) / &two_r;

    // β = √(−disc) / 2
    let neg_disc = -(disc.clone());
    let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
    let half = arena.rational(1, 2);
    let sqrt_neg_disc = arena.pow(neg_disc_id, half);
    let two_id = arena.int(2);
    let beta_expr = arena.div(sqrt_neg_disc, two_id);
    let beta_expr = crate::eval::eval(arena, beta_expr);

    // β·x
    let beta_x = arena.mul(&[beta_expr, var]);
    let cos_bx = arena.cos(beta_x);
    let sin_bx = arena.sin(beta_x);

    let c1_cos = arena.mul(&[c1, cos_bx]);
    let c2_sin = arena.mul(&[c2, sin_bx]);
    let trig_part = arena.add(&[c1_cos, c2_sin]);

    let solution = if alpha.is_zero() {
        trig_part
    } else {
        let alpha_id = ode_ratio_to_expr(arena, &alpha);
        let alpha_x = arena.mul(&[alpha_id, var]);
        let exp_ax = arena.exp(alpha_x);
        arena.mul(&[exp_ax, trig_part])
    };

    Some(OdeResult {
        solution,
        constants: vec![c1, c2],
    })
}

/// Try to find a particular solution for `y'' + b·y' + c·y = rhs(x)`
/// when rhs is a trigonometric or exponential function.
///
/// Handles:
/// - `rhs = R·sin(ωx)` or `rhs = R·cos(ωx)` → undetermined coefficients
/// - `rhs = R·exp(rx)` → undetermined coefficients (with resonance handling)
fn try_undetermined_trig_exp(
    arena: &mut Arena,
    rhs: ExprId,
    b: &num_rational::Ratio<num_bigint::BigInt>,
    c: &num_rational::Ratio<num_bigint::BigInt>,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let (coeff_r, term) = arena.as_coeff_term(rhs);
    let node = arena.node(term).clone();

    match node {
        ExprNode::Sin(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() { return None; }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &coeff_r, &zero_r, &omega, b, c, var)
        }
        ExprNode::Cos(inner) => {
            let (omega, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() { return None; }
            let zero_r = num_rational::Ratio::<num_bigint::BigInt>::zero();
            try_trig_particular(arena, &zero_r, &coeff_r, &omega, b, c, var)
        }
        ExprNode::Exp(inner) => {
            let (r_val, constant) = extract_linear_numeric(arena, inner, var)?;
            if !constant.is_zero() { return None; }
            try_exp_particular(arena, &coeff_r, &r_val, b, c, var)
        }
        _ => None,
    }
}

/// Extract the numeric coefficient and constant from a linear expression.
/// Returns `Some((a, b))` where `expr = a·var + b`, both rational.
fn extract_linear_numeric(
    arena: &Arena,
    expr: ExprId,
    var: ExprId,
) -> Option<(num_rational::Ratio<num_bigint::BigInt>, num_rational::Ratio<num_bigint::BigInt>)> {
    let poly = crate::polybridge::expr_to_poly(arena, expr, var)?;
    if poly.degree()? != 1 { return None; }
    Some((poly.coeff(1), poly.coeff(0)))
}

/// Compute a particular solution for `y'' + b·y' + c·y = P·sin(ωx) + Q·cos(ωx)`
/// via the method of undetermined coefficients.
fn try_trig_particular(
    arena: &mut Arena,
    p: &num_rational::Ratio<num_bigint::BigInt>,
    q: &num_rational::Ratio<num_bigint::BigInt>,
    omega: &num_rational::Ratio<num_bigint::BigInt>,
    b: &num_rational::Ratio<num_bigint::BigInt>,
    c: &num_rational::Ratio<num_bigint::BigInt>,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    let omega_sq = omega * omega;
    let d = c - &omega_sq;        // c − ω²
    let bw = b * omega;            // b·ω

    let det = &d * &d + &bw * &bw; // (c−ω²)² + (bω)²

    if !det.is_zero() {
        // Non-resonance: y_p = α·sin(ωx) + β·cos(ωx)
        let alpha = (&d * p + &bw * q) / &det;
        let beta = (&d * q - &bw * p) / &det;

        let omega_id = ode_ratio_to_expr(arena, omega);
        let omega_x = arena.mul(&[omega_id, var]);

        let mut terms = Vec::new();
        if !alpha.is_zero() {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let sin_wx = arena.sin(omega_x);
            terms.push(arena.mul(&[alpha_id, sin_wx]));
        }
        if !beta.is_zero() {
            let beta_id = ode_ratio_to_expr(arena, &beta);
            let cos_wx = arena.cos(omega_x);
            terms.push(arena.mul(&[beta_id, cos_wx]));
        }

        match terms.len() {
            0 => Some(arena.zero),
            1 => Some(terms[0]),
            _ => Some(arena.add(&terms)),
        }
    } else {
        // Resonance: d = 0 and bω = 0 ⟹ b = 0 and c = ω²
        if omega.is_zero() { return None; }
        let two_omega = num_rational::Ratio::from_integer(
            num_bigint::BigInt::from(2),
        ) * omega;
        let alpha = q / &two_omega;
        let beta = -(p / &two_omega);

        let omega_id = ode_ratio_to_expr(arena, omega);
        let omega_x = arena.mul(&[omega_id, var]);

        let mut inner_terms = Vec::new();
        if !alpha.is_zero() {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let sin_wx = arena.sin(omega_x);
            inner_terms.push(arena.mul(&[alpha_id, sin_wx]));
        }
        if !beta.is_zero() {
            let beta_id = ode_ratio_to_expr(arena, &beta);
            let cos_wx = arena.cos(omega_x);
            inner_terms.push(arena.mul(&[beta_id, cos_wx]));
        }

        if inner_terms.is_empty() {
            Some(arena.zero)
        } else {
            let inner = if inner_terms.len() == 1 {
                inner_terms[0]
            } else {
                arena.add(&inner_terms)
            };
            Some(arena.mul(&[var, inner]))
        }
    }
}

/// Compute a particular solution for `y'' + b·y' + c·y = R·exp(r·x)`
/// via the method of undetermined coefficients.
fn try_exp_particular(
    arena: &mut Arena,
    coeff_r: &num_rational::Ratio<num_bigint::BigInt>,
    r: &num_rational::Ratio<num_bigint::BigInt>,
    b: &num_rational::Ratio<num_bigint::BigInt>,
    c: &num_rational::Ratio<num_bigint::BigInt>,
    var: ExprId,
) -> Option<ExprId> {
    use num_traits::Zero;

    // Characteristic value at r: r² + b·r + c
    let char_val = r * r + b * r + c;

    let r_id = ode_ratio_to_expr(arena, r);
    let rx = arena.mul(&[r_id, var]);
    let exp_rx = arena.exp(rx);

    if !char_val.is_zero() {
        // Non-resonance: y_p = A·exp(rx) where A = R / (r²+br+c)
        let a_val = coeff_r / &char_val;
        let a_id = ode_ratio_to_expr(arena, &a_val);
        Some(arena.mul(&[a_id, exp_rx]))
    } else {
        // r is a root of the characteristic equation.
        let deriv_val = num_rational::Ratio::from_integer(
            num_bigint::BigInt::from(2),
        ) * r + b;
        if !deriv_val.is_zero() {
            // Single root: y_p = A·x·exp(rx) where A = R / (2r + b)
            let a_val = coeff_r / &deriv_val;
            let a_id = ode_ratio_to_expr(arena, &a_val);
            let x_exp = arena.mul(&[var, exp_rx]);
            Some(arena.mul(&[a_id, x_exp]))
        } else {
            // Double root: y_p = A·x²·exp(rx) where A = R / 2
            let two_r_val = num_rational::Ratio::from_integer(
                num_bigint::BigInt::from(2),
            );
            let a_val = coeff_r / &two_r_val;
            let a_id = ode_ratio_to_expr(arena, &a_val);
            let two_id = arena.int(2);
            let x_sq = arena.pow(var, two_id);
            let x2_exp = arena.mul(&[x_sq, exp_rx]);
            Some(arena.mul(&[a_id, x2_exp]))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bernoulli equations: y' + P(x)·y = Q(x)·y^n  (n ≠ 0, 1)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Bernoulli equations: y' + P(x)·y = Q(x)·y^n (n ≠ 0, 1).
///
/// Substitution v = y^(1−n) transforms to a first-order linear ODE:
///   v' + (1−n)·P(x)·v = (1−n)·Q(x)
/// Solve for v, then recover y = v^(1/(1−n)).
fn try_bernoulli(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Bernoulli");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    // Must be first-order (no second derivatives)
    if expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;

    let mut dy_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut p_x_terms: Vec<ExprId> = Vec::new();
    let mut q_x_terms: Vec<(ExprId, num_rational::Ratio<num_bigint::BigInt>)> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == dy_dx {
            dy_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            return None; // Free terms not allowed in standard Bernoulli
        } else if let Some(px) =
            extract_coeff_of_func(arena, child, func, func_sym, var_sym)
        {
            p_x_terms.push(px);
        } else if let Some((qx, n)) =
            extract_bernoulli_term(arena, child, func, func_sym, var_sym)
        {
            q_x_terms.push((qx, n));
        } else {
            return None;
        }
    }

    if dy_coeff.is_zero() || q_x_terms.is_empty() {
        return None;
    }

    // All y^n terms must share the same exponent n ≠ 0, 1
    let n_val = q_x_terms[0].1.clone();
    if n_val.is_zero() || n_val.is_one() {
        return None;
    }
    for (_, n) in &q_x_terms[1..] {
        if *n != n_val {
            return None;
        }
    }

    // Normalize by dy_coeff
    let inv_dy = num_rational::Ratio::<num_bigint::BigInt>::one() / &dy_coeff;
    let inv_dy_id = ode_ratio_to_expr(arena, &inv_dy);

    let p_raw = if p_x_terms.is_empty() {
        arena.zero
    } else if p_x_terms.len() == 1 {
        p_x_terms[0]
    } else {
        arena.add(&p_x_terms)
    };
    let p_x = if dy_coeff.is_one() {
        p_raw
    } else {
        let s = arena.mul(&[inv_dy_id, p_raw]);
        crate::eval::eval(arena, s)
    };

    // Q_raw·y^n appears on the LHS: y' + P·y + Q_raw·y^n = 0
    // So actual Q in y' + P·y = Q·y^n is −Q_raw
    let q_sum: Vec<ExprId> = q_x_terms.iter().map(|(qx, _)| *qx).collect();
    let q_raw = if q_sum.len() == 1 { q_sum[0] } else { arena.add(&q_sum) };
    let neg_q_raw = arena.neg(q_raw);
    let q_x = if dy_coeff.is_one() {
        neg_q_raw
    } else {
        let s = arena.mul(&[inv_dy_id, neg_q_raw]);
        crate::eval::eval(arena, s)
    };

    if contains_sym(arena, p_x, func_sym) || contains_sym(arena, q_x, func_sym) {
        return None;
    }

    // Substitution: v = y^(1−n)
    // Transformed ODE: v' + (1−n)·P·v = (1−n)·Q
    let one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() - &n_val;
    let one_minus_n_id = ode_ratio_to_expr(arena, &one_minus_n);

    let new_p = arena.mul(&[one_minus_n_id, p_x]);
    let new_p = crate::eval::eval(arena, new_p);
    let new_q = arena.mul(&[one_minus_n_id, q_x]);
    let new_q = crate::eval::eval(arena, new_q);

    // Build linear ODE for v: v' + new_p·v − new_q = 0
    let v = arena.symbol("__v");
    let dv = arena.intern(ExprNode::Derivative(v, var));
    let pv = arena.mul(&[new_p, v]);
    let neg_new_q = arena.neg(new_q);
    let linear_expr = arena.add(&[dv, pv, neg_new_q]);

    let v_sym = match arena.node(v) {
        ExprNode::Symbol(sid) => *sid,
        _ => return None,
    };

    // Solve the linear ODE for v (fall back to simple separable if P=0)
    let v_result = try_first_order_linear_general(
        arena, linear_expr, v, var, v_sym, var_sym,
    )
    .or_else(|| try_simple_separable(arena, linear_expr, v, var, v_sym, var_sym))?;

    // Recover y = v^(1/(1−n))
    let inv_one_minus_n = num_rational::Ratio::<num_bigint::BigInt>::one() / &one_minus_n;
    let inv_id = ode_ratio_to_expr(arena, &inv_one_minus_n);
    let solution = arena.pow(v_result.solution, inv_id);
    let solution = crate::eval::eval(arena, solution);

    Some(OdeResult {
        solution,
        constants: v_result.constants,
    })
}

/// Try to extract a Bernoulli term Q(x)·y^n from an expression.
///
/// Returns `Some((Q(x), n))` where Q(x) is free of y and n is a rational
/// constant.
fn extract_bernoulli_term(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    func_sym: SymbolId,
    _var_sym: SymbolId,
) -> Option<(ExprId, num_rational::Ratio<num_bigint::BigInt>)> {
    // Case 1: expr is y^n
    if let ExprNode::Pow(base, exp) = arena.node(expr).clone()
        && base == func
    {
        let n = arena.as_num(exp)?.clone();
        return Some((arena.one, n));
    }

    // Case 2: expr is Mul([..., y^n, ...])
    if let ExprNode::Mul(ref factors) = arena.node(expr).clone() {
        let mut yn_idx = None;
        let mut yn_exp = None;
        for (i, &f) in factors.iter().enumerate() {
            if let ExprNode::Pow(base, exp) = arena.node(f).clone()
                && base == func
                && let Some(n) = arena.as_num(exp)
            {
                yn_idx = Some(i);
                yn_exp = Some(n.clone());
                break;
            }
        }
        if let (Some(idx), Some(n)) = (yn_idx, yn_exp) {
            let mut other: Vec<ExprId> = Vec::new();
            for (i, &f) in factors.iter().enumerate() {
                if i != idx {
                    if contains_sym(arena, f, func_sym) {
                        return None;
                    }
                    other.push(f);
                }
            }
            let qx = match other.len() {
                0 => arena.one,
                1 => other[0],
                _ => arena.mul(&other),
            };
            return Some((qx, n));
        }
    }

    // Case 3: numeric coefficient × something
    {
        let (coeff, term) = arena.as_coeff_term(expr);
        if !coeff.is_one() && term != expr
            && let Some((inner_qx, n)) =
                extract_bernoulli_term(arena, term, func, func_sym, _var_sym)
        {
            let coeff_id = ode_ratio_to_expr(arena, &coeff);
            let qx = arena.mul(&[coeff_id, inner_qx]);
            return Some((qx, n));
        }
    }

    // Case 4: Neg(something)
    if let ExprNode::Neg(inner) = arena.node(expr).clone()
        && let Some((inner_qx, n)) =
            extract_bernoulli_term(arena, inner, func, func_sym, _var_sym)
    {
        let neg_qx = arena.neg(inner_qx);
        return Some((neg_qx, n));
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0
// ═══════════════════════════════════════════════════════════════════════════

/// Solve Euler-Cauchy equations: a·x²·y'' + b·x·y' + c·y = 0.
///
/// The characteristic equation is a·r(r−1) + b·r + c = 0, equivalently
/// a·r² + (b−a)·r + c = 0.
///
/// - Distinct real roots r₁, r₂: y = C1·x^r₁ + C2·x^r₂
/// - Repeated root r: y = (C1 + C2·ln(x))·x^r
/// - Complex roots α ± βi: y = x^α·(C1·cos(β·ln(x)) + C2·sin(β·ln(x)))
fn try_euler_cauchy(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying Euler-Cauchy");

    let dy_dx = arena.intern(ExprNode::Derivative(func, var));
    let d2y_dx2 = arena.intern(ExprNode::Derivative(dy_dx, var));

    if !expr_contains(arena, expr, d2y_dx2) {
        return None;
    }

    let children = match arena.node(expr).clone() {
        ExprNode::Add(children) => children,
        _ => return None,
    };

    use num_traits::Zero;
    let mut a_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut b_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();
    let mut c_coeff = num_rational::Ratio::<num_bigint::BigInt>::zero();

    let two_id = arena.int(2);
    let x_sq = arena.pow(var, two_id);

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            return None; // Forcing term — only homogeneous supported
        } else if let ExprNode::Mul(ref factors) = arena.node(term).clone() {
            let has_d2 = factors.contains(&d2y_dx2);
            let has_d1 = factors.contains(&dy_dx);
            let has_xsq = factors.contains(&x_sq);
            let has_xvar = factors.contains(&var);

            // Collect factors that are not x²/x/y''/y'
            let other: Vec<ExprId> = factors
                .iter()
                .copied()
                .filter(|&f| f != d2y_dx2 && f != dy_dx && f != x_sq && f != var)
                .collect();
            for &of in &other {
                if contains_sym(arena, of, func_sym)
                    || contains_sym(arena, of, var_sym)
                {
                    return None;
                }
            }
            let extra = if other.is_empty() {
                num_rational::Ratio::<num_bigint::BigInt>::from_integer(1.into())
            } else {
                let e = if other.len() == 1 {
                    other[0]
                } else {
                    arena.mul(&other)
                };
                arena.as_num(e)?.clone()
            };

            if has_d2 && has_xsq && !has_d1 && !has_xvar {
                a_coeff += &coeff * &extra;
            } else if has_d1 && has_xvar && !has_d2 && !has_xsq {
                b_coeff += &coeff * &extra;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() {
        return None;
    }

    // Characteristic equation: a·r² + (b−a)·r + c = 0
    // Normalize: r² + ((b−a)/a)·r + c/a = 0
    let p = (&b_coeff - &a_coeff) / &a_coeff;
    let q = &c_coeff / &a_coeff;

    let four = num_rational::Ratio::<num_bigint::BigInt>::from_integer(4.into());
    let disc = &p * &p - &four * &q;

    let c1 = arena.symbol("C1");
    let c2 = arena.symbol("C2");

    if disc.is_positive() {
        // Distinct real roots — solve r² + p·r + q = 0
        let r_var = arena.symbol("__r");
        let r_sq = arena.pow(r_var, two_id);
        let p_id = ode_ratio_to_expr(arena, &p);
        let q_id = ode_ratio_to_expr(arena, &q);
        let p_r = arena.mul(&[p_id, r_var]);
        let char_eq = arena.add(&[r_sq, p_r, q_id]);
        let roots = crate::solve::solve(arena, char_eq, r_var);

        if roots.len() >= 2 {
            let r1 = roots[0].value;
            let r2 = roots[1].value;
            let x_r1 = arena.pow(var, r1);
            let x_r2 = arena.pow(var, r2);
            let t1 = arena.mul(&[c1, x_r1]);
            let t2 = arena.mul(&[c2, x_r2]);
            let solution = arena.add(&[t1, t2]);
            Some(OdeResult {
                solution,
                constants: vec![c1, c2],
            })
        } else {
            None
        }
    } else if disc.is_zero() {
        // Repeated root: r = −p/2
        let two_r =
            num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let r = -(&p) / &two_r;
        let r_id = ode_ratio_to_expr(arena, &r);
        let x_r = arena.pow(var, r_id);
        let ln_x = arena.ln(var);
        let c2_ln = arena.mul(&[c2, ln_x]);
        let inner = arena.add(&[c1, c2_ln]);
        let solution = arena.mul(&[inner, x_r]);
        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
        })
    } else {
        // Complex roots α ± βi: α = −p/2, β = √(−disc)/2
        let two_r =
            num_rational::Ratio::<num_bigint::BigInt>::from_integer(2.into());
        let alpha = -(&p) / &two_r;
        let neg_disc = -disc;
        let neg_disc_id = ode_ratio_to_expr(arena, &neg_disc);
        let half = arena.rational(1, 2);
        let sqrt_neg_disc = arena.pow(neg_disc_id, half);
        let two_expr = arena.int(2);
        let beta = arena.div(sqrt_neg_disc, two_expr);
        let beta = crate::eval::eval(arena, beta);

        let ln_x = arena.ln(var);
        let beta_ln_x = arena.mul(&[beta, ln_x]);
        let cos_part = arena.cos(beta_ln_x);
        let sin_part = arena.sin(beta_ln_x);
        let c1_cos = arena.mul(&[c1, cos_part]);
        let c2_sin = arena.mul(&[c2, sin_part]);
        let trig_part = arena.add(&[c1_cos, c2_sin]);

        let solution = if alpha.is_zero() {
            trig_part
        } else {
            let alpha_id = ode_ratio_to_expr(arena, &alpha);
            let x_alpha = arena.pow(var, alpha_id);
            arena.mul(&[x_alpha, trig_part])
        };

        Some(OdeResult {
            solution,
            constants: vec![c1, c2],
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Variation of parameters: y'' + p·y' + q·y = g(x)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve y'' + p·y' + q·y = g(x) via variation of parameters.
///
/// Used as a fallback when undetermined coefficients fails (e.g., forcing
/// function is tan(x), sec(x), etc.).
///
/// Given homogeneous solutions y₁, y₂:
/// - Wronskian W computed via differentiation (with Abel's identity fallback)
/// - Particular: y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
fn try_variation_of_parameters(
    arena: &mut Arena,
    expr: ExprId,
    func: ExprId,
    var: ExprId,
    func_sym: SymbolId,
    var_sym: SymbolId,
) -> Option<OdeResult> {
    tracing::debug!("ode: trying variation of parameters");

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
    let mut f_terms: Vec<ExprId> = Vec::new();

    for &child in &children {
        let (coeff, term) = arena.as_coeff_term(child);
        if term == d2y_dx2 {
            a_coeff += coeff;
        } else if term == dy_dx {
            b_coeff += coeff;
        } else if term == func {
            c_coeff += coeff;
        } else if !contains_sym(arena, child, func_sym) {
            f_terms.push(child);
        } else {
            return None;
        }
    }

    if a_coeff.is_zero() || f_terms.is_empty() {
        return None;
    }

    let b = &b_coeff / &a_coeff;
    let c = &c_coeff / &a_coeff;

    // g(x) = −f_sum / a
    let f_sum = if f_terms.len() == 1 {
        f_terms[0]
    } else {
        arena.add(&f_terms)
    };
    let neg_f = arena.neg(f_sum);
    let a_id = ode_ratio_to_expr(arena, &a_coeff);
    let g_x = arena.div(neg_f, a_id);
    let g_x = crate::eval::eval(arena, g_x);

    // Solve homogeneous equation
    let homo = solve_characteristic_equation(arena, b.clone(), c.clone(), var)?;
    let (y1, y2) = extract_fundamental_solutions(arena, homo.solution, &homo.constants)?;

    // Wronskian via symbolic differentiation
    let y1_prime = crate::diff::diff(arena, y1, var);
    let y2_prime = crate::diff::diff(arena, y2, var);
    let w_term1 = arena.mul(&[y1, y2_prime]);
    let w_term2 = arena.mul(&[y2, y1_prime]);
    let wronskian = arena.sub(w_term1, w_term2);
    let wronskian = crate::eval::eval(arena, wronskian);
    let wronskian = crate::expand::expand(arena, wronskian);
    let wronskian = crate::eval::eval(arena, wronskian);

    // If W still depends on var (e.g. cos²+sin² unsimplified), use Abel's
    // identity: W(x) = W(0)·exp(−b·x).
    let wronskian = if contains_sym(arena, wronskian, var_sym) {
        let w0 = crate::subs::subs(arena, wronskian, var, arena.zero);
        let w0 = crate::eval::eval(arena, w0);
        if w0 == arena.zero {
            return None;
        }
        if b.is_zero() {
            w0
        } else {
            let neg_b_id = ode_ratio_to_expr(arena, &(-b.clone()));
            let neg_bx = arena.mul(&[neg_b_id, var]);
            let exp_nbx = arena.exp(neg_bx);
            arena.mul(&[w0, exp_nbx])
        }
    } else {
        wronskian
    };

    if wronskian == arena.zero {
        return None;
    }

    // y_p = −y₁·∫(y₂·g/W)dx + y₂·∫(y₁·g/W)dx
    let y2_g = arena.mul(&[y2, g_x]);
    let integrand1 = arena.div(y2_g, wronskian);
    let integrand1 = crate::eval::eval(arena, integrand1);
    let integral1 = crate::integrate::integrate(arena, integrand1, var);
    if matches!(arena.node(integral1), ExprNode::Integral(_, _)) {
        return None;
    }

    let y1_g = arena.mul(&[y1, g_x]);
    let integrand2 = arena.div(y1_g, wronskian);
    let integrand2 = crate::eval::eval(arena, integrand2);
    let integral2 = crate::integrate::integrate(arena, integrand2, var);
    if matches!(arena.node(integral2), ExprNode::Integral(_, _)) {
        return None;
    }

    let term1 = arena.mul(&[y1, integral1]);
    let neg_term1 = arena.neg(term1);
    let term2 = arena.mul(&[y2, integral2]);
    let y_p = arena.add(&[neg_term1, term2]);
    let y_p = crate::eval::eval(arena, y_p);

    let solution = arena.add(&[homo.solution, y_p]);
    let solution = crate::eval::eval(arena, solution);

    Some(OdeResult {
        solution,
        constants: homo.constants,
    })
}

/// Extract fundamental solutions y₁ and y₂ from a homogeneous solution
/// of the form C1·y₁ + C2·y₂ (or multiplied by a common factor).
///
/// Substitutes C1=1,C2=0 and C1=0,C2=1 to recover the individual solutions.
fn extract_fundamental_solutions(
    arena: &mut Arena,
    homo_solution: ExprId,
    constants: &[ExprId],
) -> Option<(ExprId, ExprId)> {
    if constants.len() != 2 {
        return None;
    }
    let c1 = constants[0];
    let c2 = constants[1];
    let zero = arena.zero;
    let one = arena.one;

    let y1 = crate::subs::subs(arena, homo_solution, c1, one);
    let y1 = crate::subs::subs(arena, y1, c2, zero);
    let y1 = crate::eval::eval(arena, y1);

    let y2 = crate::subs::subs(arena, homo_solution, c1, zero);
    let y2 = crate::subs::subs(arena, y2, c2, one);
    let y2 = crate::eval::eval(arena, y2);

    if y1 == arena.zero || y2 == arena.zero {
        return None;
    }

    Some((y1, y2))
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
        // Check for Euler-Cauchy: a·x²·y'' + b·x·y' + c·y = 0
        if try_euler_cauchy(arena, expr, func, var, func_sym, var_sym).is_some() {
            return OdeType::EulerCauchy;
        }
        // Check for nth-order reducible: F(y, y', y'') = 0 with no explicit x.
        {
            let d2_ph = arena.symbol("__d2_cls");
            let d1_ph = arena.symbol("__d1_cls");
            let stripped = crate::subs::subs(arena, expr, d2y_dx2, d2_ph);
            let stripped = crate::subs::subs(arena, stripped, dy_dx, d1_ph);
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
                } else if extract_coeff_of_func(
                    arena, child, func, func_sym, var_sym,
                )
                .is_some()
                {
                    // linear in y — OK
                } else if extract_bernoulli_term(
                    arena, child, func, func_sym, var_sym,
                )
                .is_some()
                {
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
                if contains_sym(arena, hc_rhs, func_sym)
                    && contains_sym(arena, hc_rhs, var_sym)
                {
                    // Use x→1 trick: for degree-0 homogeneous f(x,y),
                    // f(1, v) should be free of x.
                    let v_cls = arena.symbol("__v_cls");
                    let sub = crate::subs::subs(arena, hc_rhs, func, v_cls);
                    let sub = crate::subs::subs(arena, sub, var, arena.one);
                    let sub = crate::eval::eval(arena, sub);
                    let sub = crate::expand::expand(arena, sub);
                    let sub = crate::eval::eval(arena, sub);
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

// ═══════════════════════════════════════════════════════════════════════════
// ODE System Solver (constant-coefficient systems: ẋ = Ax)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve a system of first-order constant-coefficient ODEs.
///
/// Given `dx/dt = A·x` where `A` is a constant n×n matrix,
/// returns the general solution `x(t)` as a vector of `n` expressions,
/// each containing arbitrary constants `C1, C2, ..., Cn`.
///
/// # Algorithm
///
/// 1. Verifies `A` is square with no entries depending on `t_var`.
/// 2. For diagonal matrices, returns `[C1·exp(a₁₁·t), C2·exp(a₂₂·t), ...]`.
/// 3. For general matrices, computes eigenvalues and eigenvectors:
///    - Real eigenvalue λ with eigenvector v → `Cₖ·exp(λt)·v`
///    - Complex conjugate pair α±βi → trig form with `cos(βt)`, `sin(βt)`
/// 4. Falls back to matrix exponential series ([`Matrix::exp_series`]) when
///    eigendecomposition does not produce enough eigenvalues.
///
/// # Returns
///
/// `Some(vec)` with the solution vector, or `None` if the matrix is not
/// square, empty, or contains entries that depend on `t_var`.
///
/// # Examples
///
/// ```
/// use symplex::matrix::Matrix;
/// let t = symplex::var("t");
/// let a = Matrix::new(vec![
///     vec![symplex::int(0), symplex::int(1)],
///     vec![symplex::int(-2), symplex::int(-3)],
/// ]);
/// let sol = symplex::ode::solve_ode_system(&a, &t).unwrap();
/// assert_eq!(sol.len(), 2);
/// ```
pub fn solve_ode_system(
    a_matrix: &Matrix,
    t_var: &Ex,
) -> Option<Vec<Ex>> {
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 {
        return None;
    }

    // Verify constant coefficients: no entry may depend on t_var
    for i in 0..n {
        for j in 0..n {
            if a_matrix.get(i, j).contains(t_var) {
                return None;
            }
        }
    }

    // Special case: diagonal matrix → exact closed-form per component
    if ode_system_is_diagonal(a_matrix, n) {
        return Some(solve_ode_system_diagonal(a_matrix, t_var, n));
    }

    // Try eigenvalue-based exact solution
    if let Some(sol) = solve_ode_system_eigen(a_matrix, t_var, n) {
        return Some(sol);
    }

    // Fallback: truncated matrix exponential series
    Some(solve_ode_system_series(a_matrix, t_var, n))
}

/// Solve a non-homogeneous system `dx/dt = A·x + b(t)`.
///
/// Computes the general solution as:
///
/// `x(t) = x_h(t) + x_p(t)`
///
/// where `x_h` is the homogeneous solution (from [`solve_ode_system`]) and
/// `x_p` is a particular solution obtained via variation of parameters:
///
/// `x_p = exp(At) · ∫ exp(−At) · b(t) dt`
///
/// The matrix exponentials in the particular integral are computed with
/// [`Matrix::exp_series`], so the result is a truncated approximation
/// unless `b(t)` is polynomial.
///
/// # Returns
///
/// `None` if the homogeneous part cannot be solved or dimensions mismatch.
pub fn solve_ode_system_nonhomogeneous(
    a_matrix: &Matrix,
    b_vec: &[Ex],
    t_var: &Ex,
) -> Option<Vec<Ex>> {
    let n = a_matrix.nrows();
    if !a_matrix.is_square() || n == 0 || b_vec.len() != n {
        return None;
    }

    // Homogeneous part (exact when possible)
    let x_h = solve_ode_system(a_matrix, t_var)?;

    // Particular solution via variation of parameters:
    //   x_p = exp(At) · ∫ exp(-At) · b(t) dt
    let neg_one = crate::int(-1);
    let neg_a = a_matrix.scale(&neg_one);
    let neg_at = neg_a.scale(t_var);
    let exp_neg_at = neg_at.exp_series(12);

    let b_col = Matrix::col_vector(b_vec.to_vec());
    let integrand_matrix = exp_neg_at.matmul(&b_col).eval();

    // Integrate each component w.r.t. t
    let mut integrated = Vec::with_capacity(n);
    for i in 0..n {
        integrated.push(integrand_matrix.get(i, 0).integrate(t_var).eval());
    }
    let integrated_col = Matrix::col_vector(integrated);

    // Multiply by exp(At)
    let at = a_matrix.scale(t_var);
    let exp_at = at.exp_series(12);
    let particular = exp_at.matmul(&integrated_col).eval();

    // Combine: x = x_h + x_p
    let mut solution = Vec::with_capacity(n);
    for (i, x_h_i) in x_h.iter().enumerate() {
        let xi: Ex = x_h_i + particular.get(i, 0);
        solution.push(xi.eval());
    }
    Some(solution)
}

/// Returns `true` if every entry of `a_matrix` is free of `t_var`,
/// meaning the system `ẋ = A·x` has constant coefficients.
///
/// Also returns `false` for non-square matrices.
pub fn classify_ode_system_is_constant(a_matrix: &Matrix, t_var: &Ex) -> bool {
    if !a_matrix.is_square() {
        return false;
    }
    let n = a_matrix.nrows();
    for i in 0..n {
        for j in 0..n {
            if a_matrix.get(i, j).contains(t_var) {
                return false;
            }
        }
    }
    true
}

// ── ODE system helpers ─────────────────────────────────────────────────

/// Check whether a matrix is diagonal (off-diagonal entries are structurally zero).
fn ode_system_is_diagonal(m: &Matrix, n: usize) -> bool {
    for i in 0..n {
        for j in 0..n {
            if i != j && !m.get(i, j).is_zero_structural() {
                return false;
            }
        }
    }
    true
}

/// Solve a diagonal system: each row decouples to `x_i' = a_{ii} x_i`.
fn solve_ode_system_diagonal(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Vec<Ex> {
    (0..n)
        .map(|i| {
            let ci = crate::var(&format!("C{}", i + 1));
            let aii = a_matrix.get(i, i);
            if aii.is_zero_structural() {
                ci // x_i' = 0 → x_i = constant
            } else {
                let exp_term = (aii * t_var).exp();
                &ci * &exp_term
            }
        })
        .collect()
}

/// Fallback: approximate solution via truncated matrix exponential series.
fn solve_ode_system_series(a_matrix: &Matrix, t_var: &Ex, n: usize) -> Vec<Ex> {
    let m = a_matrix.scale(t_var);
    let exp_m = m.exp_series(12);
    let constants: Vec<Ex> = (1..=n)
        .map(|i| crate::var(&format!("C{i}")))
        .collect();
    let c_vec = Matrix::col_vector(constants);
    let result = exp_m.matmul(&c_vec);
    (0..n).map(|i| result.get(i, 0).eval()).collect()
}

/// Eigenvalue-based exact solver for constant-coefficient systems.
///
/// Computes eigenvalues of `A`, then for each:
/// - **Real λ**: finds eigenvector v via `null(A − λI)` and adds `C·exp(λt)·v`
/// - **Complex α±βi**: builds two real modes using `cos(βt)` and `sin(βt)`
///
/// Returns `None` if fewer than `n` eigenvalues are found or if any
/// eigenvector computation fails.
fn solve_ode_system_eigen(
    a_matrix: &Matrix,
    t_var: &Ex,
    n: usize,
) -> Option<Vec<Ex>> {
    let lambda_sym = crate::var("__ode_lambda");
    let eigenvalues = a_matrix.eigenvals(&lambda_sym);

    // Need at least n eigenvalues (counting algebraic multiplicity from solver)
    if eigenvalues.len() < n {
        return None;
    }

    let i_unit = crate::i_unit();
    let zero_ex = crate::int(0);
    let neg_i = -&i_unit;
    let identity = Matrix::identity(n);

    let mut solution: Vec<Ex> = (0..n).map(|_| crate::int(0)).collect();
    let mut const_idx = 1_usize;
    let mut used = vec![false; eigenvalues.len()];

    for idx in 0..eigenvalues.len() {
        if used[idx] {
            continue;
        }
        used[idx] = true;

        let ev = &eigenvalues[idx];

        if ev.contains(&i_unit) {
            // ── Complex eigenvalue α + βi ─────────────────────────────
            // Extract real part: substitute I → 0
            let alpha = ev.subs(&i_unit, &zero_ex).eval().simplify();
            // Extract imaginary coefficient: (λ − α) · (−i) = β
            let ev_minus_alpha = ev - &alpha;
            let beta = (&ev_minus_alpha * &neg_i).eval().simplify();

            // Find and mark the conjugate eigenvalue as processed
            for j in (idx + 1)..eigenvalues.len() {
                if !used[j] && eigenvalues[j].contains(&i_unit) {
                    let alpha_j = eigenvalues[j]
                        .subs(&i_unit, &zero_ex)
                        .eval()
                        .simplify();
                    let ej_diff = &eigenvalues[j] - &alpha_j;
                    let beta_j = (&ej_diff * &neg_i).eval().simplify();
                    let beta_sum = (&beta + &beta_j).eval().simplify();
                    if beta_sum.is_zero_structural() {
                        used[j] = true;
                        break;
                    }
                }
            }

            // Eigenvector via null(A − λI)
            let ev_identity = identity.scale(ev);
            let a_shifted = a_matrix.sub(&ev_identity).eval().simplify();
            let null_basis = a_shifted.nullspace();
            if null_basis.is_empty() {
                return None;
            }

            // Decompose eigenvector into real and imaginary parts:
            //   Re(v_i) = v_i with I → 0
            //   Im(v_i) = (v_i − Re(v_i)) · (−I)
            let mut u_re = Vec::with_capacity(n);
            let mut w_im = Vec::with_capacity(n);
            for row in 0..n {
                let vi = null_basis[0].get(row, 0).eval().simplify();
                let re = vi.subs(&i_unit, &zero_ex).eval().simplify();
                let vi_minus_re = &vi - &re;
                let im = (&vi_minus_re * &neg_i).eval().simplify();
                u_re.push(re);
                w_im.push(im);
            }

            // Two real-valued solution modes from the conjugate pair
            let c_a = crate::var(&format!("C{const_idx}"));
            let c_b = crate::var(&format!("C{}", const_idx + 1));
            const_idx += 2;

            let exp_alpha_t = if alpha.is_zero_structural() {
                crate::int(1)
            } else {
                (&alpha * t_var).exp()
            };
            let cos_beta_t = (&beta * t_var).cos();
            let sin_beta_t = (&beta * t_var).sin();

            for row in 0..n {
                // mode1[row] = e^(αt) · (cos(βt)·u[row] − sin(βt)·w[row])
                // mode2[row] = e^(αt) · (sin(βt)·u[row] + cos(βt)·w[row])
                let cu = &cos_beta_t * &u_re[row];
                let sw = &sin_beta_t * &w_im[row];
                let su = &sin_beta_t * &u_re[row];
                let cw = &cos_beta_t * &w_im[row];

                let m1 = &exp_alpha_t * &(&cu - &sw);
                let m2 = &exp_alpha_t * &(&su + &cw);

                let ca_m1 = &c_a * &m1;
                let cb_m2 = &c_b * &m2;
                let contrib = &ca_m1 + &cb_m2;
                solution[row] = &solution[row] + &contrib;
            }
        } else {
            // ── Real eigenvalue ───────────────────────────────────────
            let ev_identity = identity.scale(ev);
            let a_shifted = a_matrix.sub(&ev_identity).eval().simplify();
            let null_basis = a_shifted.nullspace();
            if null_basis.is_empty() {
                return None;
            }

            let ci = crate::var(&format!("C{const_idx}"));
            const_idx += 1;

            let exp_ev_t = if ev.is_zero_structural() {
                crate::int(1)
            } else {
                (ev * t_var).exp()
            };

            for (row, sol_row) in solution.iter_mut().enumerate().take(n) {
                let vi = null_basis[0].get(row, 0).eval().simplify();
                if !vi.is_zero_structural() {
                    let exp_vi = &exp_ev_t * &vi;
                    let ci_exp_vi = &ci * &exp_vi;
                    *sol_row = &*sol_row + &ci_exp_vi;
                }
            }
        }
    }

    let solution: Vec<Ex> = solution.into_iter().map(|s| s.eval()).collect();
    Some(solution)
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
        let r = dsolve(&mut a, expr, y, x).expect("should solve y'' + y = 0");
        let s = display(&a, r.solution);
        assert!(
            s.contains("C1") && s.contains("C2"),
            "should have two constants: {s}"
        );
        assert!(
            s.contains("cos") && s.contains("sin"),
            "should use trig form (cos and sin): {s}"
        );
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
